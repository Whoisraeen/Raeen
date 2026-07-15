//! SCE relocation applier / linker — the final LM1 step: lays a module's
//! `PT_LOAD` segments into a flat image at `base`, then walks its SCE
//! relocations, resolving each symbol relocation's NID through the
//! [`ModuleRegistry`] (HLE trampoline / LLE export / unresolved stub) and
//! patching the relocation slot.
//!
//! # Link-time marker addresses
//!
//! [`HLE_TRAMPOLINE_BASE`] and [`UNRESOLVED_STUB_BASE`] are **not real code
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
/// Base of the per-NID unresolved-stub region. A relocation whose symbol
/// resolved to neither HLE nor a loaded LLE export gets
/// `UNRESOLVED_STUB_BASE + i * 8`, where `i` indexes
/// [`LinkedModule::unresolved_stubs`] — exactly the scheme
/// [`HLE_TRAMPOLINE_BASE`] uses.
///
/// # Why per-NID and not one sentinel
///
/// Every unresolved symbol used to share this single address. When the guest
/// called it the runtime could only report `Faulted { addr: 0x5000_0000_0000 }`
/// — the one thing it could not say was *which import the guest wanted*, which
/// is the only thing worth knowing. Giving each NID its own slot makes the
/// faulting address itself the identity of the missing function, so the fault
/// reports "guest called `<nid>` from `<library>`, unimplemented" and names the
/// next thing to implement.
///
/// The region is deduped by NID, so it is bounded by the module's distinct
/// import count (876 on the measured retail title ≈ 7 KB), not by its
/// relocation count (87k).
pub const UNRESOLVED_STUB_BASE: u64 = 0x0000_5000_0000_0000;

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;
const R_X86_64_DTPMOD64: u32 = 16;
const R_X86_64_DTPOFF64: u32 = 17;
const R_X86_64_TPOFF64: u32 = 18;

/// The TLS module ID this (single, main) module's `DTPMOD64` relocations
/// resolve to. There is exactly one guest module with TLS today; a general
/// dynamic-TLS module table comes with `sceKernelLoadStartModule` (M1-D).
const MAIN_TLS_MODULE_ID: u64 = 1;

/// One HLE import resolved during linking: which `library::function` a
/// relocation's slot now points at, via its deterministic trampoline
/// address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HleTrampoline {
    pub library: String,
    pub function: String,
    pub addr: u64,
}

/// One relocation whose symbol resolved to neither HLE nor a loaded LLE
/// export. Recorded per **relocation**, not per symbol: a single NID
/// referenced by 48292 relocations appears 48292 times, so a raw count of
/// [`LinkedModule::unresolved`] measures patched slots, not missing functions.
///
/// `r_type` is what makes the count readable, because the two are wildly
/// different work items:
///
/// * `R_X86_64_JUMP_SLOT` — a PLT entry the guest will **call**. One per
///   distinct imported function. This is the real HLE gap.
/// * `R_X86_64_64` / `R_X86_64_GLOB_DAT` — a **data pointer** slot. On a big
///   C++ title these are dominated by RTTI: every polymorphic class's typeinfo
///   object points at a handful of `__cxxabiv1` vtables, so a few symbols
///   generate tens of thousands of relocations that are never called.
///
///
/// Measured on a retail title, 87414 import relocations break down as 86592
/// `R_X86_64_64`, 64 `GLOB_DAT`, and **758** `JUMP_SLOT`. The honest size of
/// the HLE gap is 758, not 87414.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnresolvedImport {
    pub nid: u64,
    /// The ELF relocation type (`R_X86_64_*`) of the slot left unresolved.
    pub r_type: u32,
}

/// One **distinct** unresolved NID and the stub address reserved for it.
/// Deduped: a NID referenced by many relocations owns exactly one slot, and
/// every one of those relocations is patched with that slot's address.
///
/// This is the reverse map the runtime needs: a fault at `addr` names `nid`,
/// and `library` says which library should have supplied it — turning an
/// opaque access violation into "implement this function next".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedStub {
    pub nid: u64,
    /// The importing library's name (from the module's import-library table),
    /// if the symbol's `#lib#` id maps to one. `None` when the module carries
    /// no library table, or the id is not in it.
    pub library: Option<String>,
    /// `UNRESOLVED_STUB_BASE + i * 8`.
    pub addr: u64,
}

/// The result of [`link_module`]: a flat, relocated image plus a record of
/// what each symbol relocation resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedModule {
    pub image: Vec<u8>,
    pub base: u64,
    /// Relocations whose symbol resolved to [`Resolver::Unresolved`] — logged,
    /// non-fatal. **One entry per relocation**; see [`UnresolvedImport`].
    ///
    /// Every entry is a genuine import: [`link_module`] skips defined symbols
    /// before it ever consults the registry, so nothing here is an artifact of
    /// misclassifying the module's own symbols.
    pub unresolved: Vec<UnresolvedImport>,
    /// The distinct unresolved NIDs, in first-encountered order — index `i`
    /// owns stub address `UNRESOLVED_STUB_BASE + i * 8`. The runtime inverts
    /// this to name the import a faulting guest call wanted.
    pub unresolved_stubs: Vec<UnresolvedStub>,
    /// Distinct HLE imports resolved, in first-encountered order, deduped
    /// by NID (a NID referenced by more than one relocation reuses the same
    /// trampoline address and appears here once).
    pub hle_trampolines: Vec<HleTrampoline>,
    /// [`SprxModule::entry`] (the ELF `e_entry`), carried through unchanged
    /// as an offset into `image` — a later milestone that gives modules a
    /// non-zero load bias will need to rebase this (see the module docs'
    /// "Image layout assumption").
    pub entry: u64,
    /// The module's `PT_TLS` static TLS template (M1-B, wall #2), carried
    /// through from [`SprxModule::tls`] so the runtime can materialize the
    /// per-thread TLS block whose layout this module's `TPOFF64`/`DTPOFF64`
    /// relocations were resolved against.
    pub tls: Option<crate::sprx::TlsTemplate>,
    /// Image offset of the `PT_SCE_PROCPARAM` block (from
    /// [`SprxModule::procparam`]'s vaddr), if the module has one. The runtime
    /// exposes `base + procparam_offset` as the guest address
    /// `sceKernelGetProcParam` returns.
    pub procparam_offset: Option<u64>,
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
            FirmwareError::MalformedDynlibData(format!(
                "segment vaddr {:#x} overflows usize",
                seg.vaddr
            ))
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

    let mut unresolved: Vec<UnresolvedImport> = Vec::new();
    let mut unresolved_stubs: Vec<UnresolvedStub> = Vec::new();
    let mut stub_addrs: HashMap<u64, u64> = HashMap::new();
    let mut hle_trampolines: Vec<HleTrampoline> = Vec::new();
    let mut hle_addrs: HashMap<u64, u64> = HashMap::new();

    // nid -> importing library name, so an unresolved stub can say which
    // library owes the guest this function. `library_index` indexes
    // `import_libs` (the library table) — never `import_modules`; crossing
    // them renames every library (see `dynlib::DT_SCE_NEEDED_MODULE_1`).
    let lib_names: HashMap<u16, &str> = dynlib
        .import_libs
        .iter()
        .map(|(id, n)| (*id, n.as_str()))
        .collect();
    let lib_of_nid: HashMap<u64, &str> = dynlib
        .imports
        .iter()
        .filter_map(|s| lib_names.get(&s.library_index).map(|n| (s.nid, *n)))
        .collect();

    for reloc in &dynlib.relocations {
        let r_type = (reloc.info & 0xFFFF_FFFF) as u32;
        let r_sym = reloc.info >> 32;

        let value = match r_type {
            R_X86_64_RELATIVE => base.wrapping_add(reloc.addend as u64),
            // TLS relocations (M1-B, wall #2). Symbol semantics: `r_sym == 0`
            // means a local TLS reference whose template-relative offset is
            // carried entirely in the addend; otherwise the symbol's `value`
            // is its offset within the module's `PT_TLS` template.
            R_X86_64_TPOFF64 => {
                // Variant-II x86-64 static TLS: the block sits immediately
                // below the TCB the FS base points at, so the fs-relative
                // offset of template offset `o` is `o - block_size`
                // (negative, wrapped). `TlsTemplate::block_size` is the
                // single source of truth both here and in the runtime's
                // block placement — they must agree exactly.
                let tls = module.tls.as_ref().ok_or_else(|| {
                    FirmwareError::MalformedDynlibData(
                        "TPOFF64 relocation in a module with no PT_TLS segment".to_string(),
                    )
                })?;
                let sym_off = tls_symbol_offset(dynlib, r_sym)?;
                sym_off
                    .wrapping_add(reloc.addend as u64)
                    .wrapping_sub(tls.block_size())
            }
            R_X86_64_DTPMOD64 => MAIN_TLS_MODULE_ID,
            R_X86_64_DTPOFF64 => {
                if module.tls.is_none() {
                    return Err(FirmwareError::MalformedDynlibData(
                        "DTPOFF64 relocation in a module with no PT_TLS segment".to_string(),
                    ));
                }
                let sym_off = tls_symbol_offset(dynlib, r_sym)?;
                sym_off.wrapping_add(reloc.addend as u64)
            }
            R_X86_64_64 | R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT => {
                let sym_index = usize::try_from(r_sym).map_err(|_| {
                    FirmwareError::MalformedDynlibData(format!(
                        "relocation r_sym {r_sym:#x} overflows usize"
                    ))
                })?;
                let symbol = dynlib.symbols.get(sym_index).ok_or_else(|| {
                    FirmwareError::MalformedDynlibData(format!(
                        "relocation r_sym {sym_index} is out of range of symbols (len {})",
                        dynlib.symbols.len()
                    ))
                })?;

                // A **defined** symbol is not an import: its slot holds the
                // symbol's own address, and there is nothing to look up. Only
                // undefined symbols name something another module must supply.
                //
                // Resolving defined symbols through the NID registry (as this
                // used to do unconditionally) is actively destructive on a real
                // title: its ~717k relocations mostly target its OWN internal
                // symbols, whose NIDs are naturally absent from the HLE
                // registry, so every one of them had `UNRESOLVED_STUB_BASE`
                // written into it — corrupting the module's internal pointers
                // and vtables wholesale, and guaranteeing a fault at the stub
                // the moment any of them was used. It went unnoticed because
                // in-tree fixtures only ever relocate imports.
                if !symbol.is_import {
                    let value =
                        base.wrapping_add(symbol.value)
                            .wrapping_add(if r_type == R_X86_64_64 {
                                reloc.addend as u64
                            } else {
                                0
                            });
                    write_slot(&mut image, reloc.offset, value)?;
                    continue;
                }

                match registry.resolve(hle, &module.name, symbol.nid) {
                    Resolver::Hle { library, function } => {
                        let nid = symbol.nid;
                        let addr = *hle_addrs.entry(nid).or_insert_with(|| {
                            let addr = HLE_TRAMPOLINE_BASE + (hle_trampolines.len() as u64 * 8);
                            hle_trampolines.push(HleTrampoline {
                                library,
                                function,
                                addr,
                            });
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
                        let nid = symbol.nid;
                        unresolved.push(UnresolvedImport { nid, r_type });
                        // Deliberately NOT `+ addend`, unlike the HLE arm: the
                        // stub address IS the symbol's identity, and the
                        // runtime inverts it by exact slot. An addend would
                        // land mid-slot and lose the NID — the whole point.
                        *stub_addrs.entry(nid).or_insert_with(|| {
                            let addr = UNRESOLVED_STUB_BASE + (unresolved_stubs.len() as u64 * 8);
                            unresolved_stubs.push(UnresolvedStub {
                                nid,
                                library: lib_of_nid.get(&nid).map(|s| s.to_string()),
                                addr,
                            });
                            addr
                        })
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
        unresolved_stubs,
        hle_trampolines,
        entry: module.entry,
        tls: module.tls.clone(),
        procparam_offset: module.procparam.as_ref().map(|p| p.vaddr),
    })
}

/// The template-relative offset a TLS relocation's symbol contributes:
/// 0 for `r_sym == 0` (local reference, offset in the addend), else the
/// symbol's `value`. Out-of-range `r_sym` is
/// [`FirmwareError::MalformedDynlibData`], mirroring the symbol lookup for
/// the address relocation types above.
fn tls_symbol_offset(dynlib: &DynlibData, r_sym: u64) -> Result<u64, FirmwareError> {
    if r_sym == 0 {
        return Ok(0);
    }
    let sym_index = usize::try_from(r_sym).map_err(|_| {
        FirmwareError::MalformedDynlibData(format!("relocation r_sym {r_sym:#x} overflows usize"))
    })?;
    let symbol = dynlib.symbols.get(sym_index).ok_or_else(|| {
        FirmwareError::MalformedDynlibData(format!(
            "TLS relocation r_sym {sym_index} is out of range of symbols (len {})",
            dynlib.symbols.len()
        ))
    })?;
    Ok(symbol.value)
}

/// Size the flat image to `max(seg.vaddr + seg.mem_size)` over `module`'s
/// `PT_LOAD` segments (also covering each segment's actual file-backed
/// `data` length, in case it exceeds `mem_size` for malformed input).
/// Bounds-checked: overflow returns [`FirmwareError::MalformedDynlibData`].
pub fn image_size(module: &SprxModule) -> Result<usize, FirmwareError> {
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
    usize::try_from(max_end).map_err(|_| {
        FirmwareError::MalformedDynlibData(format!("image size {max_end:#x} overflows usize"))
    })
}

/// Write `value` little-endian into `image[offset .. offset + 8]`,
/// bounds-checked. Never panics; out-of-range `offset` returns
/// [`FirmwareError::MalformedDynlibData`].
fn write_slot(image: &mut [u8], offset: u64, value: u64) -> Result<(), FirmwareError> {
    let start = usize::try_from(offset).map_err(|_| {
        FirmwareError::MalformedDynlibData(format!("relocation offset {offset:#x} overflows usize"))
    })?;
    let end = start.checked_add(8).ok_or_else(|| {
        FirmwareError::MalformedDynlibData(format!("relocation offset {offset:#x} + 8 overflows"))
    })?;
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
    use crate::dynlib::nid::{NidDatabase, nid_of};
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
            tls: None,
            procparam: None,
        }
    }

    /// `test_module` plus a `PT_TLS` template: 3 bytes of `.tdata`
    /// (`[0xAB, 0xCD, 0xEF]`), `mem_size` 0x30 (so 0x2D bytes of `.tbss`),
    /// align 0x20 — `block_size()` = 0x40.
    fn test_module_with_tls(mem_size: u64) -> SprxModule {
        let mut module = test_module(mem_size);
        module.tls = Some(crate::sprx::TlsTemplate {
            vaddr: 0x800,
            data: vec![0xAB, 0xCD, 0xEF],
            mem_size: 0x30,
            align: 0x20,
        });
        module
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
        let linked =
            link_module(&module, &dynlib, &registry, &hle, base).expect("relative reloc links");

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
            symbols: vec![DynSymbol {
                nid,
                value: 0,
                is_import: true,
            }],
            relocations: vec![SceRela {
                offset: 0x10,
                info: R_X86_64_JUMP_SLOT as u64, // r_sym = 0
                addend: 0,
            }],
            ..Default::default()
        };

        let linked =
            link_module(&module, &dynlib, &registry, &hle, 0).expect("HLE-resolved reloc links");

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
            symbols: vec![DynSymbol {
                nid,
                value: 0,
                is_import: true,
            }],
            relocations: vec![
                SceRela {
                    offset: 0x10,
                    info: R_X86_64_JUMP_SLOT as u64,
                    addend: 0,
                },
                SceRela {
                    offset: 0x18,
                    info: R_X86_64_GLOB_DAT as u64,
                    addend: 0,
                },
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
            symbols: vec![DynSymbol {
                nid,
                value: 0,
                is_import: true,
            }],
            relocations: vec![SceRela {
                offset: 0x18,
                info: R_X86_64_64 as u64, // r_sym = 0
                addend: 0x5,
            }],
            ..Default::default()
        };

        let linked =
            link_module(&module, &dynlib, &registry, &hle, 0).expect("LLE-resolved reloc links");

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
            symbols: vec![DynSymbol {
                nid,
                value: 0,
                is_import: true,
            }],
            relocations: vec![SceRela {
                offset: 0x20,
                info: R_X86_64_GLOB_DAT as u64, // r_sym = 0
                addend: 0,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0)
            .expect("unresolved import is non-fatal");

        assert_eq!(read_slot(&linked.image, 0x20), UNRESOLVED_STUB_BASE);
        assert_eq!(
            linked.unresolved,
            vec![UnresolvedImport {
                nid,
                r_type: R_X86_64_GLOB_DAT,
            }]
        );
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

    // --- TLS relocations (M1-B, wall #2) ---------------------------------

    #[test]
    fn tpoff64_writes_negative_block_relative_offset() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData {
            // r_sym == 0: local TLS reference, offset carried in the addend.
            relocations: vec![SceRela {
                offset: 0x10,
                info: R_X86_64_TPOFF64 as u64,
                addend: 0x8,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("TPOFF64 links");

        // block_size = 0x40 (0x30 rounded to align 0x20, min 16); the
        // fs-relative offset of template offset 0x8 is 0x8 - 0x40 = -0x38.
        assert_eq!(read_slot(&linked.image, 0x10), (-0x38i64) as u64);
    }

    #[test]
    fn tpoff64_uses_symbol_value_for_nonzero_r_sym() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData {
            symbols: vec![
                DynSymbol {
                    nid: 0,
                    value: 0,
                    is_import: false,
                },
                DynSymbol {
                    nid: nid_of("someTlsVar"),
                    value: 0x10,
                    is_import: false,
                },
            ],
            relocations: vec![SceRela {
                offset: 0x18,
                info: (1u64 << 32) | R_X86_64_TPOFF64 as u64,
                addend: 0x4,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("TPOFF64 links");
        // 0x10 (symbol) + 0x4 (addend) - 0x40 (block) = -0x2C.
        assert_eq!(read_slot(&linked.image, 0x18), (-0x2Ci64) as u64);
    }

    #[test]
    fn dtpmod64_writes_main_module_id() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x20,
                info: R_X86_64_DTPMOD64 as u64,
                addend: 0,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("DTPMOD64 links");
        assert_eq!(
            read_slot(&linked.image, 0x20),
            1,
            "single-module world: TLS module id is 1"
        );
    }

    #[test]
    fn dtpoff64_writes_template_relative_offset() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x28,
                info: R_X86_64_DTPOFF64 as u64,
                addend: 0xC,
            }],
            ..Default::default()
        };

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("DTPOFF64 links");
        assert_eq!(read_slot(&linked.image, 0x28), 0xC);
    }

    #[test]
    fn tpoff64_without_pt_tls_is_a_hard_link_error() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x100); // no TLS template
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x10,
                info: R_X86_64_TPOFF64 as u64,
                addend: 0,
            }],
            ..Default::default()
        };

        let err = link_module(&module, &dynlib, &registry, &hle, 0).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn tls_template_is_carried_into_linked_module() {
        let (hle, registry) = empty_registry();
        let module = test_module_with_tls(0x100);
        let dynlib = DynlibData::default();

        let linked = link_module(&module, &dynlib, &registry, &hle, 0).expect("links");
        let tls = linked.tls.expect("TLS template carried through");
        assert_eq!(tls.data, vec![0xAB, 0xCD, 0xEF]);
        assert_eq!(tls.mem_size, 0x30);
        assert_eq!(tls.block_size(), 0x40);
    }

    #[test]
    fn unsupported_relocation_type_errors() {
        let (hle, registry) = empty_registry();
        let module = test_module(0x100);
        let dynlib = DynlibData {
            relocations: vec![SceRela {
                offset: 0x0,
                info: 99,
                addend: 0,
            }],
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
