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

use goblin::elf::program_header::{PT_DYNAMIC, PT_GNU_EH_FRAME, PT_LOAD, PT_TLS, ProgramHeader};
use tracing::{debug, info, warn};
use xps5x_core::error::FirmwareError;

/// PS5-specific ELF types (mirrors `xps5x_loader::elf`; not re-exported
/// from there, so restated here).
const ET_SCE_EXEC: u16 = 0xFE00;
const ET_SCE_DYNEXEC: u16 = 0xFE10;
const ET_SCE_DYNAMIC: u16 = 0xFE18;

/// PS5-specific program header types (mirrors `xps5x_loader::elf`).
const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;
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

/// ELF unwind metadata retained from one guest module.
///
/// These are module-relative virtual addresses. The process composer and
/// runtime rebase them exactly like `PT_LOAD`; keeping them in the loader
/// avoids title-specific address tables and lets the guest C++ runtime find
/// exception tables for every loaded executable or PRX.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnwindInfo {
    /// `PT_GNU_EH_FRAME` (`.eh_frame_hdr`) virtual address, or zero.
    pub eh_frame_hdr_vaddr: u64,
    /// `.eh_frame` virtual address, decoded from the section table or header.
    pub eh_frame_vaddr: u64,
    /// `.eh_frame` byte size. For stripped images this is conservatively
    /// inferred as the distance to the following `.eh_frame_hdr`.
    pub eh_frame_size: u64,
    /// First `PT_LOAD` virtual address and in-memory size, exposed by the
    /// Orbis module-info ABI.
    pub seg0_vaddr: u64,
    pub seg0_size: u64,
    /// Full loadable module range used to resolve a PC to its owner.
    pub image_vaddr: u64,
    pub image_size: u64,
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

/// One module's slot in the **process-wide** static TLS area (variant-II
/// x86-64: every block sits below the thread pointer, the main module's
/// nearest to it).
///
/// # Why a process has a layout, not a block
///
/// Each module with a `PT_TLS` owns its own thread-locals, addressed two
/// interchangeable ways the ELF TLS ABI requires to alias: initial-exec
/// (`TPOFF64`, an `fs`-relative offset baked in at link time) and
/// general-dynamic (`DTPMOD64`/`DTPOFF64` through `__tls_get_addr`). Both are
/// only coherent if the process assigns every module a *distinct* region and
/// resolves both models against the same assignment.
///
/// This used to collapse to "the main module's block": every `DTPMOD64` in the
/// process resolved to module 1 and every `TPOFF64` was computed against the
/// module's own block size, as if each module sat alone below the TCB. On the
/// measured retail title four modules carry `PT_TLS` (the eboot, `libc.prx`
/// `memsz=0x478` *with* a `0x188`-byte init image, `libcohtml`,
/// `libRenoirCore`), so all four aliased the eboot's block: libc's initialized
/// TLS (errno, locale, strtok state) was never materialized, and a
/// thread-local written through one model read back zero through the other —
/// surfacing as a null-pointer crash deep inside the title's UI renderer with
/// nothing pointing at TLS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticTlsModule {
    /// Module name, for diagnostics.
    pub name: String,
    /// The TLS module ID this module's `DTPMOD64` relocations resolve to
    /// (1 = the main executable, dependencies count up in load order).
    pub module_id: u64,
    /// Distance from the thread pointer **down** to this module's block:
    /// the block occupies `[tp - tp_offset, tp - tp_offset + mem_size)`,
    /// and the module's `TPOFF64` values are `template_offset - tp_offset`.
    pub tp_offset: u64,
    /// The module's `PT_TLS` template, copied into place per thread.
    pub template: TlsTemplate,
}

/// Total bytes of static TLS below the thread pointer for `layout` —
/// the farthest module's `tp_offset`, which by construction covers every
/// block. Zero for an empty layout (no module has TLS).
pub fn static_tls_total(layout: &[StaticTlsModule]) -> u64 {
    layout.iter().map(|m| m.tp_offset).max().unwrap_or(0)
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
    /// The `PT_SCE_PROCPARAM` segment — the process-parameter block a real
    /// `sceKernelGetProcParam` returns a pointer to (it carries the SDK
    /// version, magic, and process metadata). Captured here (vaddr + bytes)
    /// so the runtime can expose its guest address instead of dropping it;
    /// [`proc_param_sdk_version`] pulls the SDK version out of it.
    pub procparam: Option<SprxSegment>,
    /// Loadable range and exception-unwind tables for this ELF.
    pub unwind: Option<UnwindInfo>,
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
    let mut procparam = None;
    let mut eh_frame_hdr = None;
    let mut has_module_param = false;

    for phdr in &parsed.program_headers {
        match phdr.p_type {
            PT_GNU_EH_FRAME => {
                let data = slice_phdr(elf, phdr, "PT_GNU_EH_FRAME")?;
                debug!(
                    "  PT_GNU_EH_FRAME: vaddr={:#x} filesz={:#x}",
                    phdr.p_vaddr, phdr.p_filesz
                );
                eh_frame_hdr = Some((phdr.p_vaddr, data));
            }
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
            PT_SCE_PROCPARAM => {
                debug!(
                    "  PT_SCE_PROCPARAM: vaddr={:#x} filesz={:#x}",
                    phdr.p_vaddr, phdr.p_filesz
                );
                procparam = Some(SprxSegment {
                    vaddr: phdr.p_vaddr,
                    data: slice_phdr(elf, phdr, "PT_SCE_PROCPARAM")?,
                    flags: phdr.p_flags,
                    mem_size: phdr.p_memsz,
                });
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

    let unwind = build_unwind_info(&parsed, &segments, eh_frame_hdr.as_ref());

    Ok(SprxModule {
        name: "module".to_string(),
        e_type,
        segments,
        dynlib_data,
        relro,
        dynamic,
        entry: parsed.header.e_entry,
        tls,
        procparam,
        unwind,
    })
}

fn build_unwind_info(
    elf: &goblin::elf::Elf<'_>,
    segments: &[SprxSegment],
    eh_frame_hdr: Option<&(u64, Vec<u8>)>,
) -> Option<UnwindInfo> {
    let first = segments.first()?;
    let image_vaddr = segments.iter().map(|s| s.vaddr).min()?;
    let image_end = segments
        .iter()
        .filter_map(|s| s.vaddr.checked_add(s.mem_size))
        .max()?;

    let section = elf
        .section_headers
        .iter()
        .find(|section| elf.shdr_strtab.get_at(section.sh_name) == Some(".eh_frame"));
    let eh_frame_hdr_vaddr = eh_frame_hdr.map_or(0, |(vaddr, _)| *vaddr);
    let decoded_vaddr =
        eh_frame_hdr.and_then(|(vaddr, bytes)| decode_eh_frame_pointer(bytes, *vaddr));
    let eh_frame_vaddr =
        section.map_or_else(|| decoded_vaddr.unwrap_or(0), |section| section.sh_addr);
    let eh_frame_size = section.map_or_else(
        || {
            eh_frame_hdr_vaddr
                .checked_sub(eh_frame_vaddr)
                .filter(|_| eh_frame_vaddr != 0)
                .unwrap_or(0)
        },
        |section| section.sh_size,
    );

    Some(UnwindInfo {
        eh_frame_hdr_vaddr,
        eh_frame_vaddr,
        eh_frame_size,
        seg0_vaddr: first.vaddr,
        seg0_size: first.mem_size,
        image_vaddr,
        image_size: image_end.saturating_sub(image_vaddr),
    })
}

/// Decode the first pointer in a GNU `.eh_frame_hdr`. PS5 executables use
/// the standard version-1 header and commonly encode this as
/// `DW_EH_PE_pcrel | DW_EH_PE_sdata4` (`0x1b`). Support the fixed-width forms
/// too; unsupported variable-length encodings degrade to an absent pointer.
fn decode_eh_frame_pointer(header: &[u8], header_vaddr: u64) -> Option<u64> {
    if header.len() < 5 || header[0] != 1 || header[1] == 0xff {
        return None;
    }
    let encoding = header[1];
    let field_vaddr = header_vaddr.checked_add(4)?;
    let raw = match encoding & 0x0f {
        0x00 => i64::from_le_bytes(header.get(4..12)?.try_into().ok()?),
        0x03 => u32::from_le_bytes(header.get(4..8)?.try_into().ok()?) as i64,
        0x04 => {
            let value = u64::from_le_bytes(header.get(4..12)?.try_into().ok()?);
            i64::try_from(value).ok()?
        }
        0x0b => i32::from_le_bytes(header.get(4..8)?.try_into().ok()?) as i64,
        0x0c => i64::from_le_bytes(header.get(4..12)?.try_into().ok()?),
        _ => return None,
    };
    let base = match encoding & 0x70 {
        0x00 => 0,
        0x10 => field_vaddr,
        _ => return None,
    };
    if raw >= 0 {
        base.checked_add(raw as u64)
    } else {
        base.checked_sub(raw.unsigned_abs())
    }
}

/// Extract the SDK version from a `PT_SCE_PROCPARAM` block, if present and
/// large enough. The Orbis process-parameter block lays out (little-endian):
/// `u64 size`, `u32 magic`, `u32 entry_count`, `u32 sdk_version`, ... — the
/// SDK version is at byte offset 16. Returns `None` when there's no
/// procparam or it's too short to hold that field, so callers degrade
/// gracefully rather than reading past the block.
pub fn proc_param_sdk_version(module: &SprxModule) -> Option<u32> {
    let seg = module.procparam.as_ref()?;
    let bytes = seg.data.get(16..20)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
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
    fn gnu_eh_frame_header_recovers_stripped_unwind_range() {
        let header_vaddr = 0x4000u64;
        let frame_vaddr = 0x3000u64;
        let relative = (frame_vaddr as i64 - (header_vaddr + 4) as i64) as i32;
        let mut eh_frame_hdr = vec![1, 0x1b, 0x03, 0x3b];
        eh_frame_hdr.extend_from_slice(&relative.to_le_bytes());
        eh_frame_hdr.extend_from_slice(&0u32.to_le_bytes());

        let elf = build_elf(
            ET_SCE_DYNAMIC,
            &[
                PhdrSpec {
                    p_type: PT_LOAD,
                    p_flags: 5,
                    p_vaddr: 0x1000,
                    data: vec![0; 0x20],
                },
                PhdrSpec {
                    p_type: PT_GNU_EH_FRAME,
                    p_flags: 4,
                    p_vaddr: header_vaddr,
                    data: eh_frame_hdr,
                },
            ],
        );

        let module = parse_sprx(&elf).expect("synthetic unwind ELF parses");
        let unwind = module.unwind.expect("PT_LOAD creates module metadata");
        assert_eq!(unwind.eh_frame_hdr_vaddr, header_vaddr);
        assert_eq!(unwind.eh_frame_vaddr, frame_vaddr);
        assert_eq!(unwind.eh_frame_size, header_vaddr - frame_vaddr);
        assert_eq!(unwind.seg0_vaddr, 0x1000);
        assert_eq!(unwind.seg0_size, 0x20);
        assert_eq!(unwind.image_vaddr, 0x1000);
        assert_eq!(unwind.image_size, 0x20);
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

    const PT_SCE_PROCPARAM_T: u32 = 0x6100_0001;

    #[test]
    fn procparam_segment_is_captured_and_sdk_version_extracted() {
        // A minimal proc-param block: u64 size, u32 magic, u32 entry_count,
        // u32 sdk_version (0x09000000) at byte offset 16.
        let mut pp = vec![0u8; 24];
        pp[0..8].copy_from_slice(&24u64.to_le_bytes());
        pp[8..12].copy_from_slice(&0x4942_524Fu32.to_le_bytes()); // magic
        pp[16..20].copy_from_slice(&0x0900_0000u32.to_le_bytes()); // sdk_version

        let elf = build_elf(
            ET_SCE_DYNAMIC,
            &[
                PhdrSpec {
                    p_type: PT_LOAD,
                    p_flags: 5,
                    p_vaddr: 0,
                    data: vec![0u8; 8],
                },
                PhdrSpec {
                    p_type: PT_SCE_PROCPARAM_T,
                    p_flags: 4,
                    p_vaddr: 0x5000,
                    data: pp,
                },
            ],
        );

        let module = parse_sprx(&elf).expect("valid synthetic ELF with PT_SCE_PROCPARAM parses");
        let seg = module
            .procparam
            .as_ref()
            .expect("procparam captured, not dropped");
        assert_eq!(seg.vaddr, 0x5000);
        assert_eq!(proc_param_sdk_version(&module), Some(0x0900_0000));
    }

    #[test]
    fn no_procparam_yields_no_sdk_version() {
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
        assert_eq!(module.procparam, None);
        assert_eq!(proc_param_sdk_version(&module), None);
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
