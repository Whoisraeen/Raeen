//! `.sprx` module parser.
//!
//! Parses the (already-plaintext) inner ELF recovered by
//! [`crate::crypto::decrypt_self`] into a [`SprxModule`]: the loadable
//! `PT_LOAD` segments plus the raw bytes of the Sony-specific
//! `PT_SCE_DYNLIBDATA` and `PT_SCE_RELRO` program headers. This is pure
//! structural ELF parsing — no keys, no decryption, no key material.
//!
//! Header and program-header decoding reuses `goblin::elf::Elf::parse`, the
//! same primitive `xps5x_loader::elf::parse_elf` uses. `parse_elf` itself
//! doesn't expose the raw `PT_SCE_DYNLIBDATA` byte range, so this module
//! re-walks the program headers directly and slices the raw file bytes,
//! bounds-checking every offset/size against the buffer — malformed or
//! truncated input returns a [`FirmwareError`], never panics.

use goblin::elf::program_header::{ProgramHeader, PT_DYNAMIC, PT_LOAD, PT_TLS};
use tracing::{debug, info, warn};
use xps5x_core::error::FirmwareError;

/// PS5-specific ELF types (mirrors `xps5x_loader::elf`; not re-exported
/// from there, so restated here).
const ET_SCE_EXEC: u16 = 0xFE00;
const ET_SCE_DYNEXEC: u16 = 0xFE10;
const ET_SCE_DYNAMIC: u16 = 0xFE18;

/// PS5-specific program header types (mirrors `xps5x_loader::elf`).
const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;
#[allow(dead_code)] // named for documentation parity with xps5x-loader/src/elf.rs; not matched directly below
const PT_SCE_PROCPARAM: u32 = 0x6100_0001;
const PT_SCE_MODULE_PARAM: u32 = 0x6100_0002;
const PT_SCE_RELRO: u32 = 0x6100_0010;

/// One loadable (or RELRO) segment: its virtual address, raw file bytes,
/// program-header flags, and in-memory size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SprxSegment {
    /// Virtual address this segment loads at (`p_vaddr`).
    pub vaddr: u64,
    /// File-backed bytes (`data[p_offset..p_offset + p_filesz]`).
    pub data: Vec<u8>,
    /// Program-header flags (`p_flags`; bit 0 = X, bit 1 = W, bit 2 = R).
    pub flags: u32,
    /// In-memory size (`p_memsz`; may exceed `data.len()` for BSS).
    pub mem_size: u64,
}

/// The module's `PT_TLS` segment: the initialization template for its
/// static TLS block (M1-B, wall #2). `data` is the file-backed `.tdata`
/// image; `mem_size` covers `.tdata` + zero-initialized `.tbss`; `align`
/// is `p_align`. The runtime materializes one such block per thread
/// (variant-II x86-64 TLS: the block sits immediately *below* the TCB the
/// FS base points at), and the linker resolves `TPOFF64`/`DTPOFF64`
/// relocations against this template's layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsTemplate {
    /// `p_vaddr` — where the template's file bytes live in the image
    /// (informational; TLS offsets are template-relative, not image-relative).
    pub vaddr: u64,
    /// File-backed `.tdata` initialization bytes.
    pub data: Vec<u8>,
    /// Total in-memory size (`p_memsz`): `.tdata` + `.tbss`.
    pub mem_size: u64,
    /// Required alignment (`p_align`).
    pub align: u64,
}

impl TlsTemplate {
    /// The static TLS block size both the linker (computing `TPOFF64`
    /// offsets) and the runtime (placing the block below the TCB) must
    /// agree on: `mem_size` rounded up to `max(align, 16)`. 16 is the
    /// x86-64 psABI minimum TCB alignment — folding it in here keeps the
    /// TCB address (block base + this size) properly aligned even for a
    /// template declaring a smaller `p_align`. Self-consistency between
    /// the two consumers is the load-bearing property.
    pub fn block_size(&self) -> u64 {
        let align = self.align.max(16);
        self.mem_size.div_ceil(align).saturating_mul(align)
    }
}

/// A parsed `.sprx` (SCE dynamic ELF) module: its loadable segments plus
/// the raw `PT_SCE_DYNLIBDATA` blob for [`crate::dynlib`] to decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SprxModule {
    /// Module name, derived from `PT_SCE_MODULE_PARAM` if trivially
    /// available, else `"module"`.
    pub name: String,
    /// The ELF `e_type` (one of `ET_SCE_EXEC`/`ET_SCE_DYNEXEC`/`ET_SCE_DYNAMIC`).
    pub e_type: u16,
    /// Every `PT_LOAD` program header, in file order.
    pub segments: Vec<SprxSegment>,
    /// Raw `PT_SCE_DYNLIBDATA` segment bytes, if present.
    pub dynlib_data: Option<Vec<u8>>,
    /// The `PT_SCE_RELRO` segment, if present.
    pub relro: Option<SprxSegment>,
    /// Raw `PT_DYNAMIC` segment bytes, if present — the `Elf64_Dyn` array
    /// consumed by [`crate::dynlib::parse_sce_dynamic`] to locate the
    /// string/symbol/relocation tables within `dynlib_data`.
    pub dynamic: Option<Vec<u8>>,
    /// The ELF header's `e_entry`. [`crate::dynlib::linker::link_module`]
    /// carries this through to [`crate::dynlib::linker::LinkedModule::entry`]
    /// unchanged, treating it as an *image offset* rather than a virtual
    /// address — valid because this crate's synthetic/homebrew modules build
    /// `PT_LOAD` segments whose `p_vaddr`s already start at 0 (see
    /// `linker.rs`'s module docs). A real `.sprx` with a non-zero load bias
    /// would need `entry - load_bias` here; that's out of LM1/RT1b scope.
    pub entry: u64,
    /// The `PT_TLS` segment (the static TLS initialization template), if
    /// present (M1-B, wall #2).
    pub tls: Option<TlsTemplate>,
}

/// Parse a plaintext inner ELF into an [`SprxModule`].
///
/// Accepts `e_type` of `ET_SCE_DYNAMIC`, `ET_SCE_DYNEXEC`, or `ET_SCE_EXEC`
/// — a plain non-SCE `e_type` (e.g. standard `ET_DYN`/`ET_EXEC`) is out of
/// LM1 scope (homebrew `.sprx` are SCE dynamic) and rejected with
/// [`FirmwareError::MalformedDynlibData`].
///
/// Every program-header-derived offset/size is bounds-checked against
/// `elf`; a truncated or otherwise malformed image returns
/// [`FirmwareError::MalformedDynlibData`], never panics.
pub fn parse_sprx(elf: &[u8]) -> Result<SprxModule, FirmwareError> {
    let parsed = goblin::elf::Elf::parse(elf)
        .map_err(|e| FirmwareError::MalformedDynlibData(format!("ELF parse error: {e}")))?;

    let e_type = parsed.header.e_type;
    if !matches!(e_type, ET_SCE_EXEC | ET_SCE_DYNEXEC | ET_SCE_DYNAMIC) {
        return Err(FirmwareError::MalformedDynlibData(format!(
            "unsupported e_type {e_type:#x}: expected an SCE dynamic/executable type \
             (ET_SCE_EXEC/ET_SCE_DYNEXEC/ET_SCE_DYNAMIC) — homebrew .sprx are SCE dynamic"
        )));
    }

    info!(
        "Parsing .sprx module: e_type={:#x}, phnum={}",
        e_type,
        parsed.program_headers.len()
    );

    let mut segments = Vec::new();
    let mut dynlib_data = None;
    let mut relro = None;
    let mut dynamic = None;
    let mut tls = None;
    let mut has_module_param = false;

    for phdr in &parsed.program_headers {
        match phdr.p_type {
            PT_TLS => {
                debug!(
                    "  PT_TLS: vaddr={:#x} filesz={:#x} memsz={:#x} align={:#x}",
                    phdr.p_vaddr, phdr.p_filesz, phdr.p_memsz, phdr.p_align
                );
                tls = Some(TlsTemplate {
                    vaddr: phdr.p_vaddr,
                    data: slice_phdr(elf, phdr, "PT_TLS")?,
                    mem_size: phdr.p_memsz,
                    align: phdr.p_align,
                });
            }
            PT_LOAD => {
                let data = slice_phdr(elf, phdr, "PT_LOAD")?;
                debug!(
                    "  PT_LOAD: vaddr={:#x} filesz={:#x} memsz={:#x} flags={:#x}",
                    phdr.p_vaddr, phdr.p_filesz, phdr.p_memsz, phdr.p_flags
                );
                segments.push(SprxSegment {
                    vaddr: phdr.p_vaddr,
                    data,
                    flags: phdr.p_flags,
                    mem_size: phdr.p_memsz,
                });
            }
            PT_SCE_DYNLIBDATA => {
                debug!(
                    "  PT_SCE_DYNLIBDATA: offset={:#x} size={:#x}",
                    phdr.p_offset, phdr.p_filesz
                );
                dynlib_data = Some(slice_phdr(elf, phdr, "PT_SCE_DYNLIBDATA")?);
            }
            PT_SCE_RELRO => {
                debug!(
                    "  PT_SCE_RELRO: vaddr={:#x} filesz={:#x}",
                    phdr.p_vaddr, phdr.p_filesz
                );
                let data = slice_phdr(elf, phdr, "PT_SCE_RELRO")?;
                relro = Some(SprxSegment {
                    vaddr: phdr.p_vaddr,
                    data,
                    flags: phdr.p_flags,
                    mem_size: phdr.p_memsz,
                });
            }
            PT_DYNAMIC => {
                debug!(
                    "  PT_DYNAMIC: offset={:#x} size={:#x}",
                    phdr.p_offset, phdr.p_filesz
                );
                dynamic = Some(slice_phdr(elf, phdr, "PT_DYNAMIC")?);
            }
            PT_SCE_MODULE_PARAM => {
                // TODO(LM1+): decode SceModuleParam to extract the real
                // module name string; for now we only note its presence.
                // Bounds-check it even though we don't consume the bytes
                // yet, so a malformed PT_SCE_MODULE_PARAM still surfaces
                // as an error rather than being silently ignored.
                let _ = slice_phdr(elf, phdr, "PT_SCE_MODULE_PARAM")?;
                has_module_param = true;
            }
            other => {
                debug!("  Skipping program header type {other:#x}");
            }
        }
    }

    if has_module_param {
        debug!("PT_SCE_MODULE_PARAM present but name extraction is not yet implemented");
    }
    if segments.is_empty() {
        warn!("Parsed .sprx module has no PT_LOAD segments");
    }

    Ok(SprxModule {
        name: "module".to_string(),
        e_type,
        segments,
        dynlib_data,
        relro,
        dynamic,
        entry: parsed.header.e_entry,
        tls,
    })
}

/// Slice `elf[p_offset .. p_offset + p_filesz]`, bounds-checked. Never
/// panics; returns [`FirmwareError::MalformedDynlibData`] on overflow or
/// out-of-bounds ranges.
fn slice_phdr(elf: &[u8], phdr: &ProgramHeader, label: &str) -> Result<Vec<u8>, FirmwareError> {
    let start = phdr.p_offset as usize;
    let len = phdr.p_filesz as usize;
    let end = start.checked_add(len).ok_or_else(|| {
        FirmwareError::MalformedDynlibData(format!("{label} offset/filesz overflow"))
    })?;
    if end > elf.len() {
        return Err(FirmwareError::MalformedDynlibData(format!(
            "{label} range [{start:#x}, {end:#x}) exceeds buffer size {:#x}",
            elf.len()
        )));
    }
    Ok(elf[start..end].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EHDR_SIZE: usize = 64;
    const PHDR_SIZE: usize = 56;
    const EM_X86_64: u16 = 62;

    /// One program header to synthesize, plus the file-backed bytes it
    /// should point at.
    struct PhdrSpec {
        p_type: u32,
        p_flags: u32,
        p_vaddr: u64,
        data: Vec<u8>,
    }

    /// Build a synthetic ELF64 image by hand: header, then a contiguous
    /// program header table, then each phdr's segment bytes laid out
    /// contiguously right after. Never real firmware bytes.
    fn build_elf(e_type: u16, phdrs: &[PhdrSpec]) -> Vec<u8> {
        let phnum = phdrs.len();
        let phoff = EHDR_SIZE as u64;

        let mut header = vec![0u8; EHDR_SIZE];
        header[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
        header[4] = 2; // ELFCLASS64
        header[5] = 1; // ELFDATA2LSB
        header[6] = 1; // EV_CURRENT
        header[16..18].copy_from_slice(&e_type.to_le_bytes());
        header[18..20].copy_from_slice(&EM_X86_64.to_le_bytes());
        header[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
        header[32..40].copy_from_slice(&phoff.to_le_bytes()); // e_phoff
        header[52..54].copy_from_slice(&(EHDR_SIZE as u16).to_le_bytes()); // e_ehsize
        header[54..56].copy_from_slice(&(PHDR_SIZE as u16).to_le_bytes()); // e_phentsize
        header[56..58].copy_from_slice(&(phnum as u16).to_le_bytes()); // e_phnum

        let mut offset = (EHDR_SIZE + phnum * PHDR_SIZE) as u64;
        let mut phdr_bytes = Vec::new();
        let mut seg_bytes = Vec::new();
        for spec in phdrs {
            let mut ph = [0u8; PHDR_SIZE];
            ph[0..4].copy_from_slice(&spec.p_type.to_le_bytes());
            ph[4..8].copy_from_slice(&spec.p_flags.to_le_bytes());
            ph[8..16].copy_from_slice(&offset.to_le_bytes()); // p_offset
            ph[16..24].copy_from_slice(&spec.p_vaddr.to_le_bytes()); // p_vaddr
            ph[32..40].copy_from_slice(&(spec.data.len() as u64).to_le_bytes()); // p_filesz
            ph[40..48].copy_from_slice(&(spec.data.len() as u64).to_le_bytes()); // p_memsz
            phdr_bytes.extend_from_slice(&ph);

            seg_bytes.extend_from_slice(&spec.data);
            offset += spec.data.len() as u64;
        }

        let mut buf = header;
        buf.extend_from_slice(&phdr_bytes);
        buf.extend_from_slice(&seg_bytes);
        buf
    }

    #[test]
    fn dynamic_module_with_load_and_dynlibdata() {
        let load_bytes = vec![0xAAu8; 16];
        let dynlib_blob = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02];

        let elf = build_elf(
            ET_SCE_DYNAMIC,
            &[
                PhdrSpec {
                    p_type: PT_LOAD,
                    p_flags: 5, // R+X
                    p_vaddr: 0x1000,
                    data: load_bytes.clone(),
                },
                PhdrSpec {
                    p_type: PT_SCE_DYNLIBDATA,
                    p_flags: 4,
                    p_vaddr: 0,
                    data: dynlib_blob.clone(),
                },
            ],
        );

        let module = parse_sprx(&elf).expect("valid synthetic SCE dynamic ELF parses");
        assert_eq!(module.e_type, ET_SCE_DYNAMIC);
        assert_eq!(module.segments.len(), 1);
        assert_eq!(module.segments[0].vaddr, 0x1000);
        assert_eq!(module.segments[0].data, load_bytes);
        assert_eq!(module.segments[0].flags, 5);
        assert_eq!(module.dynlib_data, Some(dynlib_blob));
        assert_eq!(module.relro, None);
        assert_eq!(module.dynamic, None);
    }

    #[test]
    fn dynamic_segment_is_captured() {
        let load_bytes = vec![0xAAu8; 16];
        let dynlib_blob = vec![0x01, 0x02];
        // A minimal synthetic Elf64_Dyn array: one (d_tag, d_val) pair
        // followed by DT_NULL. Not a real firmware fragment.
        let mut dynamic_bytes = Vec::new();
        dynamic_bytes.extend_from_slice(&0x6100_0035u64.to_le_bytes()); // DT_SCE_STRTAB
        dynamic_bytes.extend_from_slice(&0u64.to_le_bytes());
        dynamic_bytes.extend_from_slice(&0u64.to_le_bytes()); // DT_NULL
        dynamic_bytes.extend_from_slice(&0u64.to_le_bytes());

        let elf = build_elf(
            ET_SCE_DYNAMIC,
            &[
                PhdrSpec {
                    p_type: PT_LOAD,
                    p_flags: 5,
                    p_vaddr: 0x1000,
                    data: load_bytes,
                },
                PhdrSpec {
                    p_type: PT_SCE_DYNLIBDATA,
                    p_flags: 4,
                    p_vaddr: 0,
                    data: dynlib_blob,
                },
                PhdrSpec {
                    p_type: PT_DYNAMIC,
                    p_flags: 6,
                    p_vaddr: 0x4000,
                    data: dynamic_bytes.clone(),
                },
            ],
        );

        let module = parse_sprx(&elf).expect("valid synthetic ELF with PT_DYNAMIC parses");
        assert_eq!(module.dynamic, Some(dynamic_bytes));
    }

    #[test]
    fn relro_segment_is_captured() {
        let load_bytes = vec![0x11u8; 8];
        let dynlib_blob = vec![0x01, 0x02];
        let relro_bytes = vec![0x22u8; 4];

        let elf = build_elf(
            ET_SCE_DYNAMIC,
            &[
                PhdrSpec {
                    p_type: PT_LOAD,
                    p_flags: 6,
                    p_vaddr: 0x2000,
                    data: load_bytes,
                },
                PhdrSpec {
                    p_type: PT_SCE_DYNLIBDATA,
                    p_flags: 4,
                    p_vaddr: 0,
                    data: dynlib_blob,
                },
                PhdrSpec {
                    p_type: PT_SCE_RELRO,
                    p_flags: 4,
                    p_vaddr: 0x3000,
                    data: relro_bytes.clone(),
                },
            ],
        );

        let module = parse_sprx(&elf).expect("valid synthetic ELF with RELRO parses");
        let relro = module.relro.expect("relro segment present");
        assert_eq!(relro.vaddr, 0x3000);
        assert_eq!(relro.data, relro_bytes);
    }

    #[test]
    fn tls_segment_is_captured_with_memsz_and_align() {
        let load_bytes = vec![0x11u8; 8];
        let tdata = vec![0xAB, 0xCD, 0xEF];

        let mut elf = build_elf(
            ET_SCE_DYNAMIC,
            &[
                PhdrSpec {
                    p_type: PT_LOAD,
                    p_flags: 5,
                    p_vaddr: 0,
                    data: load_bytes,
                },
                PhdrSpec {
                    p_type: PT_TLS,
                    p_flags: 4,
                    p_vaddr: 0x800,
                    data: tdata.clone(),
                },
            ],
        );
        // `build_elf` sets p_memsz == p_filesz and leaves p_align 0; a real
        // PT_TLS has p_memsz > p_filesz (.tbss) and a nonzero p_align —
        // poke both into the second program header directly.
        let tls_phdr_off = EHDR_SIZE + PHDR_SIZE; // second phdr
        elf[tls_phdr_off + 40..tls_phdr_off + 48].copy_from_slice(&0x30u64.to_le_bytes()); // p_memsz
        elf[tls_phdr_off + 48..tls_phdr_off + 56].copy_from_slice(&0x20u64.to_le_bytes()); // p_align

        let module = parse_sprx(&elf).expect("valid synthetic ELF with PT_TLS parses");
        let tls = module.tls.expect("TLS template present");
        assert_eq!(tls.vaddr, 0x800);
        assert_eq!(tls.data, tdata);
        assert_eq!(tls.mem_size, 0x30);
        assert_eq!(tls.align, 0x20);
        // block_size: 0x30 rounded up to max(0x20, 16) = 0x40.
        assert_eq!(tls.block_size(), 0x40);
    }

    #[test]
    fn module_without_pt_tls_has_no_template() {
        let elf = build_elf(
            ET_SCE_DYNAMIC,
            &[PhdrSpec {
                p_type: PT_LOAD,
                p_flags: 5,
                p_vaddr: 0,
                data: vec![0u8; 8],
            }],
        );
        let module = parse_sprx(&elf).expect("parses");
        assert_eq!(module.tls, None);
    }

    #[test]
    fn entry_point_is_captured_from_e_entry() {
        let load_bytes = vec![0xAAu8; 16];
        let mut elf = build_elf(
            ET_SCE_DYNAMIC,
            &[PhdrSpec {
                p_type: PT_LOAD,
                p_flags: 5,
                p_vaddr: 0,
                data: load_bytes,
            }],
        );
        let entry = 0x1234u64;
        elf[24..32].copy_from_slice(&entry.to_le_bytes()); // e_entry

        let module = parse_sprx(&elf).expect("valid synthetic ELF parses");
        assert_eq!(module.entry, entry);
    }

    #[test]
    fn zero_e_entry_defaults_to_zero() {
        let load_bytes = vec![0xAAu8; 16];
        let elf = build_elf(
            ET_SCE_DYNAMIC,
            &[PhdrSpec {
                p_type: PT_LOAD,
                p_flags: 5,
                p_vaddr: 0,
                data: load_bytes,
            }],
        );

        let module = parse_sprx(&elf).expect("valid synthetic ELF parses");
        assert_eq!(module.entry, 0);
    }

    #[test]
    fn non_sce_e_type_is_rejected() {
        const ET_DYN: u16 = 3;
        let elf = build_elf(ET_DYN, &[]);

        let err = parse_sprx(&elf).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn truncated_phdr_table_does_not_panic() {
        let load_bytes = vec![0xAAu8; 16];
        let elf = build_elf(
            ET_SCE_DYNAMIC,
            &[PhdrSpec {
                p_type: PT_LOAD,
                p_flags: 5,
                p_vaddr: 0x1000,
                data: load_bytes,
            }],
        );
        // Cut the buffer off partway through the (only) program header.
        let truncated = &elf[..EHDR_SIZE + 10];

        let err = parse_sprx(truncated).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn segment_range_beyond_buffer_does_not_panic() {
        let load_bytes = vec![0xAAu8; 16];
        let mut elf = build_elf(
            ET_SCE_DYNAMIC,
            &[PhdrSpec {
                p_type: PT_LOAD,
                p_flags: 5,
                p_vaddr: 0x1000,
                data: load_bytes,
            }],
        );
        // Truncate away the segment's file-backed bytes while the program
        // header still claims the original p_filesz.
        elf.truncate(EHDR_SIZE + PHDR_SIZE);

        let err = parse_sprx(&elf).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }
}
