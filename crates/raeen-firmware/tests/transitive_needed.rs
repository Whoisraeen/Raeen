//! Transitive `DT_NEEDED` loading (M1-D closure): [`load_process`] must
//! file-load not only the eboot's direct NEEDED `.prx`s but the NEEDEDs of
//! those dependencies too, so an import satisfied only by a *transitive*
//! module resolves — with diamonds loading the shared module once, and a
//! missing transitive file degrading to unresolved-with-warning, never an
//! error.
//!
//! Entirely hand-built buffers, `NoKeysProvider` throughout — no real
//! firmware bytes anywhere. The ELF/SELF builders mirror
//! `homebrew_pipeline.rs`'s (private helpers aren't importable, so
//! replicated here), extended with export symbols and `DT_NEEDED` entries.
//! Transitive-only modules live under `sce_module/` so the root-level plugin
//! scan cannot pre-place them: only the transitive walk can find them.

use raeen_firmware::crypto::NoKeysProvider;
use raeen_firmware::dynlib::nid::{NidDatabase, encode_nid, nid_of};
use raeen_firmware::{ModuleRegistry, UNRESOLVED_STUB_BASE, load_process};
use raeen_hle::HleRegistry;

const EHDR_SIZE: usize = 64;
const PHDR_SIZE: usize = 56;
const EM_X86_64: u16 = 62;
const ET_SCE_DYNAMIC: u16 = 0xFE18;

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;

const DT_NEEDED: u64 = 1;
const DT_SCE_JMPREL: u64 = 0x6100_0029;
const DT_SCE_PLTRELSZ: u64 = 0x6100_002D;
const DT_SCE_STRTAB: u64 = 0x6100_0035;
const DT_SCE_STRSZ: u64 = 0x6100_0037;
const DT_SCE_SYMTAB: u64 = 0x6100_0039;
const DT_SCE_SYMENT: u64 = 0x6100_003B;
const DT_SCE_SYMTABSZ: u64 = 0x6100_003F;
const DT_SCE_NEEDED_MODULE_1: u64 = 0x6100_0045;
const DT_SCE_IMPORT_LIB_1: u64 = 0x6100_0049;
const DT_NULL: u64 = 0;

const R_X86_64_JUMP_SLOT: u64 = 7;

/// SELF layout constants, mirroring `crypto::self_crypto`'s private ones
/// (they aren't public, so replicated here for this hand-built fixture).
const SELF_MAGIC: u32 = 0x4F15D17E;
const SELF_HEADER_SIZE: usize = 32;
const SELF_ENTRY_SIZE: usize = 32;

/// A reloc slot inside the `PT_LOAD` segment (0x100 bytes at vaddr 0).
const RELOC_SLOT_OFFSET: u64 = 0x10;
/// Module vaddr every test export is placed at (inside the PT_LOAD).
const EXPORT_VADDR: u64 = 0x40;

/// Guest base every process is loaded at.
const BASE: u64 = 0x8000_0000;

/// One program header to synthesize, plus the file-backed bytes it should
/// point at. Mirrors `homebrew_pipeline.rs`'s helper.
struct PhdrSpec {
    p_type: u32,
    p_flags: u32,
    p_vaddr: u64,
    data: Vec<u8>,
}

/// Build a synthetic ELF64 image by hand: header, then a contiguous program
/// header table, then each phdr's segment bytes laid out contiguously right
/// after. Mirrors `homebrew_pipeline.rs`'s helper.
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
/// plaintext (properties = 0 => passthrough). Mirrors
/// `homebrew_pipeline.rs`'s helper.
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

/// What one synthetic module carries. A real module mixes these freely; the
/// tests below need only one concern per module.
#[derive(Default)]
struct ModuleSpec<'a> {
    /// `DT_NEEDED` file names this module declares (its own dependencies).
    needed: &'a [&'a str],
    /// One imported symbol: `(nid, provider)`. The provider names both the
    /// import-library (id 0) and needed-module (id 1) tables, so strict
    /// provider-aware linking resolves the import against that module's
    /// registered exports.
    import: Option<(u64, &'a str)>,
    /// One exported symbol: `(nid, vaddr inside the module's PT_LOAD)`.
    export: Option<(u64, u64)>,
}

fn add_str(s: &str, strtab: &mut Vec<u8>) -> u64 {
    let off = strtab.len() as u64;
    strtab.extend_from_slice(s.as_bytes());
    strtab.push(0);
    off
}

fn push_sym(st_name: u64, st_shndx: u16, st_value: u64, symtab: &mut Vec<u8>) {
    symtab.extend_from_slice(&(st_name as u32).to_le_bytes());
    symtab.push(0); // st_info
    symtab.push(0); // st_other
    symtab.extend_from_slice(&st_shndx.to_le_bytes());
    symtab.extend_from_slice(&st_value.to_le_bytes());
    symtab.extend_from_slice(&0u64.to_le_bytes()); // st_size
    assert_eq!(symtab.len() % 24, 0, "whole Elf64_Sym records");
}

/// Build the `PT_SCE_DYNLIBDATA` blob and matching `PT_DYNAMIC` bytes for a
/// [`ModuleSpec`]: strtab + symtab (import first, export second) + one
/// JMPREL entry for the import, with the SCE tags pointing at them.
fn build_dynlib_and_dynamic(spec: &ModuleSpec) -> (Vec<u8>, Vec<u8>) {
    // Symbol names: the import carries "<nid>#<lib>#<mod>" (library id 0 =
    // "A", module id 1 = "B", per `dynlib::decode_symbols`; both tables
    // below name the provider, matching the real ABI). The export is a bare
    // "<nid>".
    let mut strtab = vec![0u8];
    let import_name_off = spec
        .import
        .map(|(nid, _)| add_str(&format!("{}#A#B", encode_nid(nid)), &mut strtab));
    let provider_off = spec
        .import
        .map(|(_, provider)| add_str(provider, &mut strtab));
    let export_name_off = spec
        .export
        .map(|(nid, _)| add_str(&encode_nid(nid), &mut strtab));
    let needed_offs: Vec<u64> = spec
        .needed
        .iter()
        .map(|n| add_str(n, &mut strtab))
        .collect();

    // symtab: the import (undefined: st_shndx 0, st_value 0) at index 0 when
    // present — the JMPREL's r_sym — then the export (defined: st_shndx 1)
    // at its vaddr.
    let mut symtab = Vec::new();
    if let Some(off) = import_name_off {
        push_sym(off, 0, 0, &mut symtab);
    }
    if let (Some(off), Some((_, vaddr))) = (export_name_off, spec.export) {
        push_sym(off, 1, vaddr, &mut symtab);
    }

    // JMPREL: one Elf64_Rela for the import, r_sym = 0.
    let mut jmprel = Vec::new();
    if spec.import.is_some() {
        jmprel.extend_from_slice(&RELOC_SLOT_OFFSET.to_le_bytes());
        jmprel.extend_from_slice(&R_X86_64_JUMP_SLOT.to_le_bytes());
        jmprel.extend_from_slice(&0i64.to_le_bytes());
    }

    let strtab_off = 0u64;
    let symtab_off = strtab.len() as u64;
    let jmprel_off = symtab_off + symtab.len() as u64;

    let mut blob = Vec::new();
    blob.extend_from_slice(&strtab);
    blob.extend_from_slice(&symtab);
    blob.extend_from_slice(&jmprel);

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
    // Standard DT_NEEDED: d_val is a bare strtab offset, one per dependency.
    for off in &needed_offs {
        push_tag(DT_NEEDED, *off);
    }
    if let Some(off) = provider_off {
        push_tag(DT_SCE_IMPORT_LIB_1, off); // library id 0
        push_tag(DT_SCE_NEEDED_MODULE_1, (1u64 << 48) | off); // module id 1
    }
    if spec.import.is_some() {
        push_tag(DT_SCE_JMPREL, jmprel_off);
        push_tag(DT_SCE_PLTRELSZ, jmprel.len() as u64);
    }
    push_tag(DT_NULL, 0);

    (blob, dynamic)
}

/// Build a fully synthetic `.prx`-shaped module (plaintext SELF wrapping an
/// `ET_SCE_DYNAMIC` ELF): one `PT_LOAD`, the `PT_SCE_DYNLIBDATA` blob, and
/// `PT_DYNAMIC` describing a [`ModuleSpec`].
fn build_module(spec: &ModuleSpec) -> Vec<u8> {
    let (dynlib_blob, dynamic_bytes) = build_dynlib_and_dynamic(spec);

    let load_bytes = vec![0u8; 0x100];

    let elf = build_elf(
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

    build_plaintext_self(&elf)
}

/// A unique temp app directory holding the given `(name, bytes)` module
/// files; a name containing `/` (e.g. `sce_module/libdepb.prx`) creates its
/// subdirectory. Same pattern as `raeen-kernel`'s filesystem tests.
fn app_dir(tag: &str, files: &[(&str, Vec<u8>)]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("raeen-transitive-{tag}-{}", std::process::id()));
    for (name, bytes) in files {
        let path = dir.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
    dir
}

fn load(dir: &std::path::Path, main: &[u8]) -> raeen_firmware::LoadedProcess {
    let hle = HleRegistry::new();
    let mut registry = ModuleRegistry::new(NidDatabase::from_hle(&hle));
    load_process(main, dir, &NoKeysProvider, &mut registry, &hle, BASE)
        .expect("synthetic modules must load; unresolved imports are non-fatal")
}

fn read_slot(image: &[u8], offset: u64) -> u64 {
    let start = offset as usize;
    u64::from_le_bytes(image[start..start + 8].try_into().unwrap())
}

/// main → libdepa.prx → libdepb.prx: the eboot imports a NID only the
/// depth-2 module exports. libdepb ships in `sce_module/`, invisible to the
/// root-level plugin scan, so only the transitive walk can load it.
#[test]
fn transitive_needed_chain_resolves_import_from_depth_two_module() {
    let export_nid = nid_of("transitiveChainExportOnlyDepBProvides");
    let main = build_module(&ModuleSpec {
        needed: &["libdepa.prx"],
        import: Some((export_nid, "libdepb")),
        export: None,
    });
    let dep_a = build_module(&ModuleSpec {
        needed: &["libdepb.prx"],
        ..Default::default()
    });
    let dep_b = build_module(&ModuleSpec {
        export: Some((export_nid, EXPORT_VADDR)),
        ..Default::default()
    });
    let dir = app_dir(
        "chain",
        &[("libdepa.prx", dep_a), ("sce_module/libdepb.prx", dep_b)],
    );

    let process = load(&dir, &main);

    let names: Vec<&str> = process
        .dependencies
        .iter()
        .map(|d| d.name.as_str())
        .collect();
    assert_eq!(
        names,
        ["libdepa.prx", "libdepb.prx"],
        "the direct dep loads first, then its own NEEDED (breadth-first)"
    );
    let dep_b_offset = process.dependencies[1].image_offset;
    assert_eq!(
        read_slot(&process.linked.image, RELOC_SLOT_OFFSET),
        BASE + dep_b_offset + EXPORT_VADDR,
        "the eboot's import resolved to libdepb's export at its absolute guest address"
    );
    assert!(process.linked.unresolved.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

/// main → libdepa.prx -↘
/// main → libdepb.prx -→ libdepc.prx: two paths to the shared module must
/// still load it exactly once (the visit-set makes the walk a fixpoint).
#[test]
fn transitive_needed_diamond_loads_the_shared_module_once() {
    let export_nid = nid_of("diamondSharedExportOnlyDepCProvides");
    let main = build_module(&ModuleSpec {
        needed: &["libdepa.prx", "libdepb.prx"],
        import: Some((export_nid, "libdepc")),
        export: None,
    });
    let dep_a = build_module(&ModuleSpec {
        needed: &["libdepc.prx"],
        ..Default::default()
    });
    let dep_b = build_module(&ModuleSpec {
        needed: &["libdepc.prx"],
        ..Default::default()
    });
    let dep_c = build_module(&ModuleSpec {
        export: Some((export_nid, EXPORT_VADDR)),
        ..Default::default()
    });
    let dir = app_dir(
        "diamond",
        &[
            ("libdepa.prx", dep_a),
            ("libdepb.prx", dep_b),
            ("sce_module/libdepc.prx", dep_c),
        ],
    );

    let process = load(&dir, &main);

    assert_eq!(
        process
            .dependencies
            .iter()
            .filter(|d| d.name == "libdepc.prx")
            .count(),
        1,
        "two dependencies naming libdepc must load it exactly once"
    );
    assert_eq!(process.dependencies.len(), 3);
    let dep_c_offset = process
        .dependencies
        .iter()
        .find(|d| d.name == "libdepc.prx")
        .unwrap()
        .image_offset;
    assert_eq!(
        read_slot(&process.linked.image, RELOC_SLOT_OFFSET),
        BASE + dep_c_offset + EXPORT_VADDR,
        "the eboot's import resolved to the single libdepc's export"
    );
    assert!(process.linked.unresolved.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

/// main → libdepa.prx → libmissing.prx (shipped nowhere): the miss must
/// degrade to a warning plus an unresolved import — exactly like a missing
/// direct dep — never fail the whole load.
#[test]
fn missing_transitive_needed_degrades_to_unresolved_not_error() {
    let missing_nid = nid_of("missingTransitiveExportNobodyProvides");
    let main = build_module(&ModuleSpec {
        needed: &["libdepa.prx"],
        import: Some((missing_nid, "libmissing")),
        export: None,
    });
    let dep_a = build_module(&ModuleSpec {
        needed: &["libmissing.prx"],
        ..Default::default()
    });
    let dir = app_dir("missing", &[("libdepa.prx", dep_a)]);

    // `load` asserts Ok: a missing transitive file is non-fatal.
    let process = load(&dir, &main);

    assert_eq!(
        process.dependencies.len(),
        1,
        "only libdepa loaded; the missing transitive module is skipped"
    );
    assert_eq!(
        process.dependencies[0].name, "libdepa.prx",
        "libdepa still loaded and linked even though its own NEEDED is missing"
    );
    assert_eq!(
        read_slot(&process.linked.image, RELOC_SLOT_OFFSET),
        UNRESOLVED_STUB_BASE,
        "the first (and only) unresolved NID owns stub slot 0"
    );
    assert_eq!(
        process
            .linked
            .unresolved
            .iter()
            .map(|u| u.nid)
            .collect::<Vec<_>>(),
        vec![missing_nid]
    );

    let _ = std::fs::remove_dir_all(&dir);
}
