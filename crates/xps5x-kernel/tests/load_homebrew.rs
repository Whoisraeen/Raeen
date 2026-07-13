//! End-to-end test: parse a minimal ELF and map it into the kernel's
//! emulated address space (the milestone M1 boot path).

use xps5x_kernel::OrbisKernel;

/// Build a minimal, valid ELF64 x86-64 executable with a single
/// R+X PT_LOAD segment containing `code`, entering at `vaddr`.
fn build_minimal_elf(vaddr: u64, code: &[u8]) -> Vec<u8> {
    const EHSIZE: usize = 64;
    const PHENTSIZE: usize = 56;
    const SEG_OFFSET: u64 = 0x1000; // File offset of the segment data.

    let mut buf = vec![0u8; SEG_OFFSET as usize + code.len()];

    // ── ELF header ──────────────────────────────────────────
    buf[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB (little-endian)
    buf[6] = 1; // EV_CURRENT
    buf[7] = 9; // ELFOSABI_FREEBSD (PS5-like)
    buf[16..18].copy_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
    buf[18..20].copy_from_slice(&62u16.to_le_bytes()); // e_machine = EM_X86_64
    buf[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
    buf[24..32].copy_from_slice(&vaddr.to_le_bytes()); // e_entry
    buf[32..40].copy_from_slice(&(EHSIZE as u64).to_le_bytes()); // e_phoff
    buf[40..48].copy_from_slice(&0u64.to_le_bytes()); // e_shoff
    buf[48..52].copy_from_slice(&0u32.to_le_bytes()); // e_flags
    buf[52..54].copy_from_slice(&(EHSIZE as u16).to_le_bytes()); // e_ehsize
    buf[54..56].copy_from_slice(&(PHENTSIZE as u16).to_le_bytes()); // e_phentsize
    buf[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum
    // e_shentsize / e_shnum / e_shstrndx remain 0.

    // ── Program header (PT_LOAD) at offset EHSIZE ───────────
    let ph = EHSIZE;
    buf[ph..ph + 4].copy_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
    buf[ph + 4..ph + 8].copy_from_slice(&5u32.to_le_bytes()); // p_flags = R|X
    buf[ph + 8..ph + 16].copy_from_slice(&SEG_OFFSET.to_le_bytes()); // p_offset
    buf[ph + 16..ph + 24].copy_from_slice(&vaddr.to_le_bytes()); // p_vaddr
    buf[ph + 24..ph + 32].copy_from_slice(&vaddr.to_le_bytes()); // p_paddr
    buf[ph + 32..ph + 40].copy_from_slice(&(code.len() as u64).to_le_bytes()); // p_filesz
    buf[ph + 40..ph + 48].copy_from_slice(&(code.len() as u64).to_le_bytes()); // p_memsz
    buf[ph + 48..ph + 56].copy_from_slice(&0x1000u64.to_le_bytes()); // p_align

    // ── Segment data ────────────────────────────────────────
    buf[SEG_OFFSET as usize..].copy_from_slice(code);

    buf
}

#[test]
fn loads_minimal_homebrew_elf_into_memory() {
    // `mov eax, 42; ret`
    let code = [0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3];
    let entry = 0x0040_0000;
    let elf = build_minimal_elf(entry, &code);

    let kernel = OrbisKernel::new();
    let image = kernel
        .load_executable_from_bytes(&elf)
        .expect("homebrew ELF should load");

    // Image metadata reflects the single segment.
    assert_eq!(image.entry_point, entry);
    assert_eq!(image.base_address, entry);
    assert_eq!(image.segment_count, 1);
    assert!(!image.is_dynamic);

    // The module was registered with the kernel.
    assert!(kernel.modules.contains_key(&image.module_id));

    // The entry point is a mapped, executable region.
    let region = kernel
        .memory
        .region_containing(entry)
        .expect("entry point must be mapped");
    assert!(region.protection.contains(xps5x_core::types::MemoryProtection::EXEC));

    // The code bytes are readable back at the entry point.
    let read_back = kernel.memory.read(entry, code.len()).expect("code readable");
    assert_eq!(read_back, code);
}

#[test]
fn rejects_non_elf_bytes() {
    let kernel = OrbisKernel::new();
    let result = kernel.load_executable_from_bytes(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00]);
    assert!(result.is_err());
}
