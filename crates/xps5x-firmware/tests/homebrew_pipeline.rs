//! LM1 acceptance: a fully synthetic homebrew `.sprx` — a plaintext SELF
//! wrapping an `ET_SCE_DYNAMIC` ELF with one `PT_LOAD`, a `PT_SCE_DYNLIBDATA`
//! declaring one import, and a `PT_DYNAMIC` pointing at it — flows through
//! [`xps5x_firmware::load_module`] end to end: SELF passthrough -> `.sprx`
//! parse -> `PT_SCE_DYNLIBDATA` decode -> NID link against the HLE registry.
//!
//! Entirely hand-built buffers. No real firmware bytes anywhere, and
//! `NoKeysProvider` throughout — this milestone requires no keys.

use xps5x_firmware::crypto::NoKeysProvider;
use xps5x_firmware::dynlib::nid::{encode_nid, nid_of, NidDatabase};
use xps5x_firmware::{load_module, ModuleRegistry, HLE_TRAMPOLINE_BASE, UNRESOLVED_STUB_ADDR};
use xps5x_hle::HleRegistry;

const EHDR_SIZE: usize = 64;
const PHDR_SIZE: usize = 56;
const EM_X86_64: u16 = 62;
const ET_SCE_DYNAMIC: u16 = 0xFE18;

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;

const DT_SCE_JMPREL: u64 = 0x6100_0029;
const DT_SCE_PLTRELSZ: u64 = 0x6100_002D;
const DT_SCE_STRTAB: u64 = 0x6100_0035;
const DT_SCE_STRSZ: u64 = 0x6100_0037;
const DT_SCE_SYMTAB: u64 = 0x6100_0039;
const DT_SCE_SYMENT: u64 = 0x6100_003B;
const DT_SCE_SYMTABSZ: u64 = 0x6100_003F;
const DT_NULL: u64 = 0;

const R_X86_64_JUMP_SLOT: u64 = 7;

/// SELF layout constants, mirroring `crypto::self_crypto`'s private ones
/// (they aren't public, so replicated here for this hand-built fixture).
const SELF_MAGIC: u32 = 0x4F15D17E;
const SELF_HEADER_SIZE: usize = 32;
const SELF_ENTRY_SIZE: usize = 32;

/// A reloc slot inside the `PT_LOAD` segment (which is 0x100 bytes at
/// vaddr 0).
const RELOC_SLOT_OFFSET: u64 = 0x10;

/// One program header to synthesize, plus the file-backed bytes it should
/// point at. Mirrors `sprx.rs`'s private test helper (not importable, so
/// replicated here).
struct PhdrSpec {
    p_type: u32,
    p_flags: u32,
    p_vaddr: u64,
    data: Vec<u8>,
}

/// Build a synthetic ELF64 image by hand: header, then a contiguous program
/// header table, then each phdr's segment bytes laid out contiguously right
/// after. Mirrors `sprx.rs`'s `build_elf` test helper.
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

/// Build a SELF header (32 bytes) + entry table + segment payloads, all
/// plaintext (properties = 0 => passthrough). Mirrors `self_crypto.rs`'s
/// `build_self` test helper (single-segment case).
fn build_plaintext_self(inner_elf: &[u8]) -> Vec<u8> {
    let header_size = SELF_HEADER_SIZE + SELF_ENTRY_SIZE;

    let mut buf = vec![0u8; header_size];
    buf[0..4].copy_from_slice(&SELF_MAGIC.to_le_bytes());
    buf[4] = 1; // version
    buf[5] = 0; // mode
    buf[6] = 1; // endian
    buf[7] = 0; // attributes
    buf[8..12].copy_from_slice(&0u32.to_le_bytes()); // key_type
    buf[12..14].copy_from_slice(&(header_size as u16).to_le_bytes());
    buf[14..16].copy_from_slice(&0u16.to_le_bytes()); // meta_size
    buf[24..26].copy_from_slice(&1u16.to_le_bytes()); // num_entries
    buf[26..28].copy_from_slice(&0u16.to_le_bytes()); // flags

    let base = SELF_HEADER_SIZE;
    buf[base..base + 8].copy_from_slice(&0u64.to_le_bytes()); // properties: plaintext
    buf[base + 8..base + 16].copy_from_slice(&(header_size as u64).to_le_bytes()); // offset
    buf[base + 16..base + 24].copy_from_slice(&(inner_elf.len() as u64).to_le_bytes()); // compressed_size
    buf[base + 24..base + 32].copy_from_slice(&(inner_elf.len() as u64).to_le_bytes()); // uncompressed_size

    buf.extend_from_slice(inner_elf);

    let file_size = buf.len() as u64;
    buf[16..24].copy_from_slice(&file_size.to_le_bytes());

    buf
}

/// Build the `PT_SCE_DYNLIBDATA` blob (strtab + one undefined `Elf64_Sym` +
/// one `Elf64_Rela` JMPREL entry) and the matching `PT_DYNAMIC` bytes (SCE
/// tags pointing at the strtab/symtab/jmprel offsets+sizes *within the
/// dynlibdata blob*), for a single import identified by `import_nid`.
fn build_dynlib_and_dynamic(import_nid: u64) -> (Vec<u8>, Vec<u8>) {
    // strtab: [0] reserved null byte, then the NUL-terminated encoded-NID
    // name (the "<nid>#<lib_id>#<mod_id>" convention `dynlib::mod` expects).
    let import_name = format!("{}#A#A", encode_nid(import_nid));
    let mut strtab = vec![0u8];
    let import_off = strtab.len() as u32;
    strtab.extend_from_slice(import_name.as_bytes());
    strtab.push(0);

    // symtab: one undefined (import) Elf64_Sym at index 0.
    // st_name, st_info=0, st_other=0, st_shndx=0, st_value=0, st_size=0.
    let mut symtab = Vec::new();
    symtab.extend_from_slice(&import_off.to_le_bytes());
    symtab.push(0);
    symtab.push(0);
    symtab.extend_from_slice(&0u16.to_le_bytes());
    symtab.extend_from_slice(&0u64.to_le_bytes());
    symtab.extend_from_slice(&0u64.to_le_bytes());
    assert_eq!(symtab.len(), 24, "one Elf64_Sym record");

    // JMPREL: one Elf64_Rela entry: r_offset = a slot inside the PT_LOAD
    // image, r_info = (0 << 32) | R_X86_64_JUMP_SLOT, addend = 0.
    let mut jmprel = Vec::new();
    jmprel.extend_from_slice(&RELOC_SLOT_OFFSET.to_le_bytes());
    // r_sym = 0 (the import's symtab index) << 32 | R_X86_64_JUMP_SLOT.
    jmprel.extend_from_slice(&R_X86_64_JUMP_SLOT.to_le_bytes());
    jmprel.extend_from_slice(&0i64.to_le_bytes());
    assert_eq!(jmprel.len(), 24, "one Elf64_Rela record");

    let strtab_off = 0u64;
    let symtab_off = strtab.len() as u64;
    let jmprel_off = symtab_off + symtab.len() as u64;

    let mut blob = Vec::new();
    blob.extend_from_slice(&strtab);
    blob.extend_from_slice(&symtab);
    blob.extend_from_slice(&jmprel);

    // PT_DYNAMIC: Elf64_Dyn (d_tag, d_val) pairs, terminated by DT_NULL.
    let mut dynamic = Vec::new();
    let mut push_tag = |tag: u64, val: u64| {
        dynamic.extend_from_slice(&tag.to_le_bytes());
        dynamic.extend_from_slice(&val.to_le_bytes());
    };
    push_tag(DT_SCE_STRTAB, strtab_off);
    push_tag(DT_SCE_STRSZ, strtab.len() as u64);
    push_tag(DT_SCE_SYMTAB, symtab_off);
    push_tag(DT_SCE_SYMTABSZ, symtab.len() as u64);
    push_tag(DT_SCE_SYMENT, 24);
    push_tag(DT_SCE_JMPREL, jmprel_off);
    push_tag(DT_SCE_PLTRELSZ, jmprel.len() as u64);
    push_tag(DT_NULL, 0);

    (blob, dynamic)
}

/// Build a fully synthetic homebrew `.sprx` (plaintext SELF wrapping an
/// `ET_SCE_DYNAMIC` ELF) with one `PT_LOAD`, `PT_SCE_DYNLIBDATA`, and
/// `PT_DYNAMIC` importing `import_nid` via a single `JUMP_SLOT` relocation.
fn build_homebrew_sprx(import_nid: u64) -> Vec<u8> {
    build_homebrew_sprx_with_entry(import_nid, 0)
}

/// Same as [`build_homebrew_sprx`], but patches `e_entry` to `entry` — used
/// to prove RT1b's entry-offset plumbing (`SprxModule::entry` ->
/// `LinkedModule::entry`) end to end through the real [`load_module`]
/// pipeline, not just the unit-level extraction/propagation tests in
/// `sprx.rs`/`dynlib/linker.rs`.
fn build_homebrew_sprx_with_entry(import_nid: u64, entry: u64) -> Vec<u8> {
    let (dynlib_blob, dynamic_bytes) = build_dynlib_and_dynamic(import_nid);

    let load_bytes = vec![0u8; 0x100];

    let mut elf = build_elf(
        ET_SCE_DYNAMIC,
        &[
            PhdrSpec {
                p_type: PT_LOAD,
                p_flags: 6, // R+W
                p_vaddr: 0,
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
                p_vaddr: 0x2000,
                data: dynamic_bytes,
            },
        ],
    );
    elf[24..32].copy_from_slice(&entry.to_le_bytes()); // e_entry

    build_plaintext_self(&elf)
}

fn read_slot(image: &[u8], offset: u64) -> u64 {
    let start = offset as usize;
    u64::from_le_bytes(image[start..start + 8].try_into().unwrap())
}

#[test]
fn homebrew_sprx_links_import_against_hle_trampoline() {
    let hle = HleRegistry::new();
    let (lib, func) = hle
        .registered_names()
        .into_iter()
        .next()
        .expect("HLE registers at least one fn");
    let import_nid = nid_of(&func);

    let sprx = build_homebrew_sprx(import_nid);

    let db = NidDatabase::from_hle_names(hle.registered_names());
    let mut registry = ModuleRegistry::new(db);

    let base = 0x8000_0000u64;
    let linked = load_module(&sprx, &NoKeysProvider, &mut registry, &hle, base)
        .expect("fully synthetic homebrew .sprx links end-to-end against HLE");

    assert_eq!(read_slot(&linked.image, RELOC_SLOT_OFFSET), HLE_TRAMPOLINE_BASE);
    assert_eq!(linked.hle_trampolines.len(), 1);
    assert_eq!(linked.hle_trampolines[0].library, lib);
    assert_eq!(linked.hle_trampolines[0].function, func);
    assert!(linked.unresolved.is_empty());
}

#[test]
fn homebrew_sprx_with_unknown_import_is_unresolved_not_fatal() {
    let hle = HleRegistry::new();
    let (_, func) = hle
        .registered_names()
        .into_iter()
        .next()
        .expect("HLE registers at least one fn");
    // A real HLE-registered NID, used only to seed a non-empty NidDatabase;
    // the module itself imports a bogus NID absent from any registered name.
    let _ = nid_of(&func);
    let bogus_nid = nid_of("totallyUnknownHomebrewImportNobodyRegistered");

    let sprx = build_homebrew_sprx(bogus_nid);

    let db = NidDatabase::from_hle_names(hle.registered_names());
    let mut registry = ModuleRegistry::new(db);

    let base = 0x8000_0000u64;
    let linked = load_module(&sprx, &NoKeysProvider, &mut registry, &hle, base)
        .expect("an unresolved import is non-fatal");

    assert_eq!(read_slot(&linked.image, RELOC_SLOT_OFFSET), UNRESOLVED_STUB_ADDR);
    assert_eq!(linked.unresolved, vec![bogus_nid]);
    assert!(linked.hle_trampolines.is_empty());
}

#[test]
fn homebrew_sprx_entry_point_propagates_through_load_module() {
    let hle = HleRegistry::new();
    let (_, func) = hle
        .registered_names()
        .into_iter()
        .next()
        .expect("HLE registers at least one fn");
    let import_nid = nid_of(&func);

    let sprx = build_homebrew_sprx_with_entry(import_nid, 0x40);

    let db = NidDatabase::from_hle_names(hle.registered_names());
    let mut registry = ModuleRegistry::new(db);

    let base = 0x8000_0000u64;
    let linked = load_module(&sprx, &NoKeysProvider, &mut registry, &hle, base)
        .expect("fully synthetic homebrew .sprx links end-to-end against HLE");

    assert_eq!(linked.entry, 0x40, "e_entry rides along as an image offset through the whole load_module pipeline");
}
