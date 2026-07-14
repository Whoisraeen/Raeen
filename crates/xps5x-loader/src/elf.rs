//! ELF64 parser for PS5 executables.
//!
//! PS5 executables are standard ELF64 binaries targeting x86-64 with
//! Sony-specific extensions in the dynamic section. This module parses
//! the ELF headers, program headers, and section headers to extract
//! loadable segments and dynamic linking information.

use crate::{LoadedBinary, LoadedSegment};
use tracing::{debug, info, warn};
use xps5x_core::error::LoaderError;

/// PS5-specific ELF OS/ABI value.
#[allow(dead_code)] // reserved: e_ident[EI_OSABI] validation not yet enforced
const ELFOSABI_FREEBSD: u8 = 9;

/// PS5-specific ELF type for Signed ELF (SCE).
const ET_SCE_EXEC: u16 = 0xFE00;
const ET_SCE_DYNEXEC: u16 = 0xFE10;
const ET_SCE_DYNAMIC: u16 = 0xFE18;

/// PS5-specific program header types.
const PT_SCE_DYNLIBDATA: u32 = 0x61000000;
const PT_SCE_PROCPARAM: u32 = 0x61000001;
const PT_SCE_MODULE_PARAM: u32 = 0x61000002;
const PT_SCE_RELRO: u32 = 0x61000010;
#[allow(dead_code)] // reserved: SCE comment segment not yet consumed
const PT_SCE_COMMENT: u32 = 0x6FFFFF00;

/// Parse an ELF64 binary from raw bytes.
///
/// Returns a `LoadedBinary` containing all loadable segments,
/// the entry point, and dynamic library dependencies.
pub fn parse_elf(data: &[u8]) -> Result<LoadedBinary, LoaderError> {
    use goblin::elf::Elf;

    info!("Parsing ELF binary ({} bytes)", data.len());

    // Validate magic bytes.
    if data.len() < 4 || data[0..4] != [0x7F, b'E', b'L', b'F'] {
        let magic = if data.len() >= 4 {
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        } else {
            0
        };
        return Err(LoaderError::InvalidElfMagic(magic));
    }

    // Validate 64-bit ELF.
    if data.len() < 5 || data[4] != 2 {
        return Err(LoaderError::UnsupportedElfClass(
            data.get(4).copied().unwrap_or(0),
        ));
    }

    let elf = Elf::parse(data).map_err(|e| LoaderError::SegmentLoadFailed {
        address: 0,
        size: 0,
        reason: format!("ELF parse error: {e}"),
    })?;

    // Validate architecture (x86-64).
    if elf.header.e_machine != goblin::elf::header::EM_X86_64 {
        return Err(LoaderError::UnsupportedArchitecture(elf.header.e_machine));
    }

    debug!(
        "ELF type: {:#x}, entry: {:#x}, phnum: {}, shnum: {}",
        elf.header.e_type,
        elf.header.e_entry,
        elf.program_headers.len(),
        elf.section_headers.len()
    );

    // Check for PS5-specific ELF types.
    match elf.header.e_type {
        ET_SCE_EXEC | ET_SCE_DYNEXEC | ET_SCE_DYNAMIC => {
            info!(
                "Detected PS5 SCE executable (type {:#x})",
                elf.header.e_type
            );
        }
        goblin::elf::header::ET_EXEC | goblin::elf::header::ET_DYN => {
            info!(
                "Detected standard ELF executable (type {:#x})",
                elf.header.e_type
            );
        }
        _ => {
            warn!(
                "Unknown ELF type: {:#x}, attempting to load anyway",
                elf.header.e_type
            );
        }
    }

    // Extract loadable segments.
    let mut segments = Vec::new();
    for phdr in &elf.program_headers {
        match phdr.p_type {
            goblin::elf::program_header::PT_LOAD => {
                let offset = phdr.p_offset as usize;
                let file_size = phdr.p_filesz as usize;

                let segment_data = if offset + file_size <= data.len() {
                    let mut buf = vec![0u8; phdr.p_memsz as usize];
                    buf[..file_size].copy_from_slice(&data[offset..offset + file_size]);
                    buf
                } else {
                    warn!(
                        "Segment at {:#x} extends beyond file (offset {:#x} + size {:#x} > file size {:#x})",
                        phdr.p_vaddr,
                        offset,
                        file_size,
                        data.len()
                    );
                    vec![0u8; phdr.p_memsz as usize]
                };

                debug!(
                    "  PT_LOAD: vaddr={:#x} memsz={:#x} filesz={:#x} flags={:#x}",
                    phdr.p_vaddr, phdr.p_memsz, phdr.p_filesz, phdr.p_flags
                );

                segments.push(LoadedSegment {
                    vaddr: phdr.p_vaddr,
                    mem_size: phdr.p_memsz,
                    file_size: phdr.p_filesz,
                    data: segment_data,
                    readable: phdr.p_flags & 0x4 != 0,
                    writable: phdr.p_flags & 0x2 != 0,
                    executable: phdr.p_flags & 0x1 != 0,
                });
            }
            PT_SCE_DYNLIBDATA => {
                debug!(
                    "  PT_SCE_DYNLIBDATA: offset={:#x} size={:#x}",
                    phdr.p_offset, phdr.p_filesz
                );
            }
            PT_SCE_PROCPARAM => {
                debug!(
                    "  PT_SCE_PROCPARAM: vaddr={:#x} size={:#x}",
                    phdr.p_vaddr, phdr.p_filesz
                );
            }
            PT_SCE_MODULE_PARAM => {
                debug!(
                    "  PT_SCE_MODULE_PARAM: vaddr={:#x} size={:#x}",
                    phdr.p_vaddr, phdr.p_filesz
                );
            }
            PT_SCE_RELRO => {
                debug!(
                    "  PT_SCE_RELRO: vaddr={:#x} size={:#x}",
                    phdr.p_vaddr, phdr.p_filesz
                );
            }
            _ => {
                debug!("  Skipping program header type {:#x}", phdr.p_type);
            }
        }
    }

    // Extract needed libraries from dynamic section.
    let needed_libraries: Vec<String> = elf.libraries.iter().map(|lib| lib.to_string()).collect();

    if !needed_libraries.is_empty() {
        info!("Required libraries: {:?}", needed_libraries);
    }

    let module_name = elf
        .soname
        .map(|s| s.to_string())
        .unwrap_or_else(|| "eboot.bin".to_string());

    let is_dynamic = elf.header.e_type == goblin::elf::header::ET_DYN
        || elf.header.e_type == ET_SCE_DYNEXEC
        || elf.header.e_type == ET_SCE_DYNAMIC;

    info!(
        "ELF loaded: entry={:#x}, segments={}, libraries={}, dynamic={}",
        elf.header.e_entry,
        segments.len(),
        needed_libraries.len(),
        is_dynamic
    );

    Ok(LoadedBinary {
        entry_point: elf.header.e_entry,
        segments,
        needed_libraries,
        module_name,
        is_dynamic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_magic() {
        let data = [0x00, 0x01, 0x02, 0x03];
        let result = parse_elf(&data);
        assert!(matches!(result, Err(LoaderError::InvalidElfMagic(_))));
    }

    #[test]
    fn test_too_short() {
        let data = [0x7F, b'E', b'L'];
        let result = parse_elf(&data);
        assert!(matches!(result, Err(LoaderError::InvalidElfMagic(_))));
    }

    #[test]
    fn test_32bit_elf_rejected() {
        // Valid magic but 32-bit class.
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
        data[4] = 1; // ELFCLASS32
        let result = parse_elf(&data);
        assert!(matches!(result, Err(LoaderError::UnsupportedElfClass(1))));
    }
}
