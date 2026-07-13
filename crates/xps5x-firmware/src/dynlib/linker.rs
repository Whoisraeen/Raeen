//! SCE relocation applier / linker — the final LM1 step: lays a module's
//! `PT_LOAD` segments into a flat image at `base`, then walks its SCE
//! relocations, resolving each symbol relocation's NID through the
//! [`ModuleRegistry`] (HLE trampoline / LLE export / unresolved stub) and
//! patching the relocation slot.
//!
//! # Link-time marker addresses
//!
//! [`HLE_TRAMPOLINE_BASE`] and [`UNRESOLVED_STUB_ADDR`] are **not real code
//! addresses** — they are distinct, obviously-synthetic high sentinel
//! ranges a later runtime milestone will recognize and trap (dispatching to
//! the HLE registry, or diagnosing an unresolved import) rather than ever
//! executing as code. LM1's job is only to reach a *linked* state with
//! every relocation slot holding a defined value.
//!
//! # Image layout assumption
//!
//! `link_module` assumes the module's `PT_LOAD` segment `p_vaddr`s are
//! already image-relative, starting near 0 (as this crate's synthetic test
//! fixtures build them) — the flat `image` buffer is sized to
//! `max(seg.vaddr + seg.mem_size)` and each segment's bytes are copied to
//! `image[vaddr..]` directly. `base` is folded in only when *writing*
//! resolved addresses into relocation slots (`R_X86_64_RELATIVE`, and any
//! `Lle` export address, are added to `base`/looked up independent of the
//! image's own vaddr space). A real `.sprx`'s segments may need an explicit
//! rebase before this image-relative assumption holds; that's out of LM1
//! scope.

use std::collections::HashMap;

use xps5x_core::error::FirmwareError;
use xps5x_hle::HleRegistry;

use crate::dynlib::DynlibData;
use crate::registry::{ModuleRegistry, Resolver};
use crate::sprx::SprxModule;

/// Deterministic base address for synthetic HLE-trampoline slots. See the
/// module docs: this is a link-time marker, not a real code address.
pub const HLE_TRAMPOLINE_BASE: u64 = 0x0000_4000_0000_0000;
/// Sentinel written into a relocation slot whose symbol resolved to
/// neither HLE nor a loaded LLE export. See the module docs.
pub const UNRESOLVED_STUB_ADDR: u64 = 0x0000_5000_0000_0000;

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;

/// One HLE import resolved during linking: which `library::function` a
/// relocation's slot now points at, via its deterministic trampoline
/// address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HleTrampoline {
    pub library: String,
    pub function: String,
    pub addr: u64,
}

/// The result of [`link_module`]: a flat, relocated image plus a record of
/// what each symbol relocation resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedModule {
    pub image: Vec<u8>,
    pub base: u64,
    /// NIDs that resolved to [`Resolver::Unresolved`] — logged, non-fatal.
    pub unresolved: Vec<u64>,
    /// Distinct HLE imports resolved, in first-encountered order, deduped
    /// by NID (a NID referenced by more than one relocation reuses the same
    /// trampoline address and appears here once).
    pub hle_trampolines: Vec<HleTrampoline>,
    /// [`SprxModule::entry`] (the ELF `e_entry`), carried through unchanged
    /// as an offset into `image` — a later milestone that gives modules a
    /// non-zero load bias will need to rebase this (see the module docs'
    /// "Image layout assumption").
    pub entry: u64,
}

/// Lay `module`'s `PT_LOAD` segments into a flat image at `base` and apply
/// every relocation in `dynlib.relocations`. See the module docs for the
/// image-layout assumption and the meaning of the HLE-trampoline /
/// unresolved-stub marker addresses.
///
/// Never panics: every offset/index derived from `module`/`dynlib` is
/// bounds-checked and returns [`FirmwareError::MalformedDynlibData`] on
/// overflow or out-of-range access. An unresolved import is recorded in
/// `unresolved` and does not fail the call; only a genuinely unsupported
/// relocation type fails it ([`FirmwareError::UnsupportedRelocation`]).
pub fn link_module(
    module: &SprxModule,
    dynlib: &DynlibData,
    registry: &ModuleRegistry,
    hle: &HleRegistry,
    base: u64,
) -> Result<LinkedModule, FirmwareError> {
    let mut image = vec![0u8; image_size(module)?];

    for seg in &module.segments {
        let start = usize::try_from(seg.vaddr).map_err(|_| {
            FirmwareError::MalformedDynlibData(format!("segment vaddr {:#x} overflows usize", seg.vaddr))
        })?;
        let end = start.checked_add(seg.data.len()).ok_or_else(|| {
            FirmwareError::MalformedDynlibData(format!(
                "segment vaddr {:#x} + data len {:#x} overflows",
                seg.vaddr,
                seg.data.len()
            ))
        })?;
        if end > image.len() {
            return Err(FirmwareError::MalformedDynlibData(format!(
                "segment range [{start:#x}, {end:#x}) exceeds image size {:#x}",
                image.len()
            )));
        }
        image[start..end].copy_from_slice(&seg.data);
    }

    let mut unresolved = Vec::new();
    let mut hle_trampolines: Vec<HleTrampoline> = Vec::new();
    let mut hle_addrs: HashMap<u64, u64> = HashMap::new();

    for reloc in &dynlib.relocations {
        let r_type = (reloc.info & 0xFFFF_FFFF) as u32;
        let r_sym = reloc.info >> 32;

        let value = match r_type {
            R_X86_64_RELATIVE => base.wrapping_add(reloc.addend as u64),
            R_X86_64_64 | R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT => {
                let sym_index = usize::try_from(r_sym).map_err(|_| {
                    FirmwareError::MalformedDynlibData(format!("relocation r_sym {r_sym:#x} overflows usize"))
                })?;
                let symbol = dynlib.symbols.get(sym_index).ok_or_else(|| {
                    FirmwareError::MalformedDynlibData(format!(
                        "relocation r_sym {sym_index} is out of range of symbols (len {})",
                        dynlib.symbols.len()
                    ))
                })?;

                match registry.resolve(hle, &module.name, symbol.nid) {
                    Resolver::Hle { library, function } => {
                        let nid = symbol.nid;
                        let addr = *hle_addrs.entry(nid).or_insert_with(|| {
                            let addr = HLE_TRAMPOLINE_BASE + (hle_trampolines.len() as u64 * 8);
                            hle_trampolines.push(HleTrampoline { library, function, addr });
                            addr
                        });
                        if r_type == R_X86_64_64 {
                            addr.wrapping_add(reloc.addend as u64)
                        } else {
                            addr
                        }
                    }
                    Resolver::Lle { addr } => addr.wrapping_add(reloc.addend as u64),
                    Resolver::Unresolved => {
                        unresolved.push(symbol.nid);
                        UNRESOLVED_STUB_ADDR
                    }
                }
            }
            other => return Err(FirmwareError::UnsupportedRelocation(other)),
        };

        write_slot(&mut image, reloc.offset, value)?;
    }

    Ok(LinkedModule {
        image,
        base,
        unresolved,
        hle_trampolines,
        entry: module.entry,
    })
}

/// Size the flat image to `max(seg.vaddr + seg.mem_size)` over `module`'s
/// `PT_LOAD` segments (also covering each segment's actual file-backed
/// `data` length, in case it exceeds `mem_size` for malformed input).
/// Bounds-checked: overflow returns [`FirmwareError::MalformedDynlibData`].
fn image_size(module: &SprxModule) -> Result<usize, FirmwareError> {
    let mut max_end: u64 = 0;
    for seg in &module.segments {
        let mem_end = seg.vaddr.checked_add(seg.mem_size).ok_or_else(|| {
            FirmwareError::MalformedDynlibData(format!(
                "segment vaddr {:#x} + mem_size {:#x} overflows u64",
                seg.vaddr, seg.mem_size
            ))
        })?;
        let data_end = seg
            .vaddr
            .checked_add(seg.data.len() as u64)
            .ok_or_else(|| {
                FirmwareError::MalformedDynlibData(format!(
                    "segment vaddr {:#x} + data len overflows u64",
                    seg.vaddr
                ))
            })?;
        max_end = max_end.max(mem_end).max(data_end);
    }
    usize::try_from(max_end)
        .map_err(|_| FirmwareError::MalformedDynlibData(format!("image size {max_end:#x} overflows usize")))
}

/// Write `value` little-endian into `image[offset .. offset + 8]`,
/// bounds-checked. Never panics; out-of-range `offset` returns
/// [`FirmwareError::MalformedDynlibData`].
fn write_slot(image: &mut [u8], offset: u64, value: u64) -> Result<(), FirmwareError> {
    let start = usize::try_from(offset)
        .map_err(|_| FirmwareError::MalformedDynlibData(format!("relocation offset {offset:#x} overflows usize")))?;
    let end = start
        .checked_add(8)
        .ok_or_else(|| FirmwareError::MalformedDynlibData(format!("relocation offset {offset:#x} + 8 overflows")))?;
    if end > image.len() {
        return Err(FirmwareError::MalformedDynlibData(format!(
            "relocation slot [{start:#x}, {end:#x}) exceeds image size {:#x}",
            image.len()
        )));
    }
    image[start..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynlib::nid::{nid_of, NidDatabase};
    use crate::dynlib::{DynSymbol, SceRela, SymbolExport};
    use crate::sprx::SprxSegment;

    /// A single-`PT_LOAD`-segment module big enough (`mem_size` bytes) to
    /// hold whatever relocation slots a test writes into it. Entirely
    /// synthetic — no real firmware bytes.
    fn test_module(mem_size: u64) -> SprxModule {
        SprxModule {
            name: "someModule".to_string(),
            e_type: 0xFE18, // ET_SCE_DYNAMIC
            segments: vec![SprxSegment {
                vaddr: 0,
                data: vec![0u8; mem_size as usize],
                flags: 5,
                mem_size,
            }],
            dynlib_data: None,
            relro: None,
            dynamic: None,
            entry: 0,
        }
    }

    fn empty_registry() -> (HleRegistry, ModuleRegistry) {
        let hle = HleRegistry::new();
        let db = NidDatabase::from_hle_names(hle.registered_names());
        (hle, ModuleRegistry::new(db))
    }

    fn read_slot(image: &[u8], offset: u64) -> u64 {
        let start = offset as usize;
        u64::from_le_bytes(image[start..start + 8].try_into().unwrap())
    }

    #[test]
    fn relative_relocation_writes_base_plus_addend() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x100);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x8,
                info: R_X86_64_RELATIVE as u64, // r_sym = 0, unused for RELATIVE
                addend: 0x20,
            }],
            ..Default::default()
        };

        let base = 0x1_0000_0000u64;
        let linked = link_module(&module, &dynlib, &registry, &hle, base).expect("relative reloc links");

        assert_eq!(read_slot(&linked.image, 0x8), base + 0x20);
        assert!(linked.unresolved.is_empty());
        assert!(linked.hle_trampolines.is_empty());
        assert_eq!(linked.base, base);
    }

    #[test]
    fn jump_slot_import_resolves_to_hle_trampoline() {
        let (hle, registry) = empty_registry();
        let (library, function) = hle
            .registered_names()
            .into_iter()
            .next()
            .expect("HleRegistry::new() registers at least one function");
        let nid = nid_of(&function);

        let module = test_module(0x100);
        let dynlib = DynlibData {
            symbols: vec![DynSymbol { nid, value: 0, is_import: true }],
            relocations: vec![SceRela {
                offset: 0x10,
                info: R_X86_64_JUMP_SLOT as u64, // r_sym = 0
                addend: 0,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("HLE-resolved reloc links");

        assert_eq!(read_slot(&linked.image, 0x10), HLE_TRAMPOLINE_BASE);
        assert_eq!(linked.hle_trampolines.len(), 1);
        assert_eq!(linked.hle_trampolines[0].library, library);
        assert_eq!(linked.hle_trampolines[0].function, function);
        assert_eq!(linked.hle_trampolines[0].addr, HLE_TRAMPOLINE_BASE);
        assert!(linked.unresolved.is_empty());
    }

    #[test]
    fn repeated_hle_import_reuses_same_trampoline_address() {
        let (hle, registry) = empty_registry();
        let (_, function) = hle
            .registered_names()
            .into_iter()
            .next()
            .expect("HleRegistry::new() registers at least one function");
        let nid = nid_of(&function);

        let module = test_module(0x100);
        let dynlib = DynlibData {
            symbols: vec![DynSymbol { nid, value: 0, is_import: true }],
            relocations: vec![
                SceRela { offset: 0x10, info: R_X86_64_JUMP_SLOT as u64, addend: 0 },
                SceRela { offset: 0x18, info: R_X86_64_GLOB_DAT as u64, addend: 0 },
            ],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("links");

        assert_eq!(read_slot(&linked.image, 0x10), HLE_TRAMPOLINE_BASE);
        assert_eq!(read_slot(&linked.image, 0x18), HLE_TRAMPOLINE_BASE);
        // Same NID referenced twice -> one trampoline record, not two.
        assert_eq!(linked.hle_trampolines.len(), 1);
    }

    #[test]
    fn import_resolves_to_lle_export_plus_addend() {
        let (hle, mut registry) = empty_registry();
        let nid = nid_of("someUniqueLleOnlyExport");
        registry.register_module_exports("otherModule", &[SymbolExport { nid, value: 0x9999 }]);

        let module = test_module(0x100);
        let dynlib = DynlibData {
            symbols: vec![DynSymbol { nid, value: 0, is_import: true }],
            relocations: vec![SceRela {
                offset: 0x18,
                info: R_X86_64_64 as u64, // r_sym = 0
                addend: 0x5,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("LLE-resolved reloc links");

        assert_eq!(read_slot(&linked.image, 0x18), 0x9999 + 0x5);
        assert!(linked.unresolved.is_empty());
        assert!(linked.hle_trampolines.is_empty());
    }

    #[test]
    fn unknown_nid_writes_stub_and_is_recorded_unresolved_without_failing() {
        let (hle, registry) = empty_registry();
        let nid = nid_of("totallyUnknownFunctionNameNobodyRegistered");

        let module = test_module(0x100);
        let dynlib = DynlibData {
            symbols: vec![DynSymbol { nid, value: 0, is_import: true }],
            relocations: vec![SceRela {
                offset: 0x20,
                info: R_X86_64_GLOB_DAT as u64, // r_sym = 0
                addend: 0,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("unresolved import is non-fatal");

        assert_eq!(read_slot(&linked.image, 0x20), UNRESOLVED_STUB_ADDR);
        assert_eq!(linked.unresolved, vec![nid]);
    }

    #[test]
    fn entry_offset_propagates_from_module_to_linked_module() {
        let (hle, registry) = empty_registry();
        let mut module = test_module(0x100);
        module.entry = 0x40;
        let dynlib = DynlibData::default();

        let linked = link_module(&module, &dynlib, &registry, &hle, 0x1000).expect("links");
        assert_eq!(linked.entry, 0x40);
    }

    #[test]
    fn unsupported_relocation_type_errors() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x100);
        let dynlib = DynlibData {
            relocations: vec![SceRela { offset: 0x0, info: 99, addend: 0 }],
            ..Default::default()
        };

        let err = link_module(&module, &dynlib, &registry, &hle, 0).unwrap_err();
        assert!(matches!(err, FirmwareError::UnsupportedRelocation(99)));
    }

    #[test]
    fn relocation_offset_past_image_end_errors_not_panics() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x10);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x1_0000, // far past the 0x10-byte image
                info: R_X86_64_RELATIVE as u64,
                addend: 0,
            }],
            ..Default::default()
        };

        let err = link_module(&module, &dynlib, &registry, &hle, 0).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn relocation_r_sym_past_symbols_len_errors_not_panics() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x100);
        let dynlib = DynlibData {
            symbols: Vec::new(), // r_sym 5 is out of range of an empty table
            relocations: vec![SceRela {
                offset: 0x8,
                info: (5u64 << 32) | R_X86_64_JUMP_SLOT as u64,
                addend: 0,
            }],
            ..Default::default()
        };

        let err = link_module(&module, &dynlib, &registry, &hle, 0).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }
}
