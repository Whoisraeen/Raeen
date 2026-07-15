//! Sony dynamic-linking data: NID hashing (`nid`), and the
//! `PT_SCE_DYNLIBDATA` decoder that turns the raw blob captured by
//! [`crate::sprx::parse_sprx`] plus the `PT_DYNAMIC` tags captured
//! alongside it into typed imports/exports/relocations.
//!
//! # Format
//!
//! `PT_SCE_DYNLIBDATA` is a blob holding a string table, an ELF64 symbol
//! table, and ELF64 RELA relocation tables. It is addressed by standard
//! `Elf64_Dyn` `(d_tag, d_val)` pairs living in the module's `PT_DYNAMIC`
//! segment ([`parse_sce_dynamic`]), where `d_val` is (for table tags) an
//! offset into the dynlibdata blob or a size. This module implements the
//! common documented SCE dynamic-tag set; see the RE note on
//! [`parse_dynlibdata`] for what is community-RE-derived versus
//! structurally certain.
//!
//! Every offset/size read from `dyn_tags` is bounds-checked against the
//! blob (`checked_add`, then compared against `blob.len()`); a table that
//! runs past the end of the blob, or a record that doesn't evenly divide
//! its table, returns [`FirmwareError::MalformedDynlibData`] — never a
//! panic.

pub mod linker;
pub mod nid;

use tracing::{debug, info, warn};
use xps5x_core::error::FirmwareError;

/// SCE dynamic tags (see module docs). Values are the community-documented
/// common set; `PT_DYNAMIC` also carries plenty of standard ELF tags this
/// module simply ignores (it only looks for the tags below).
/// Standard ELF `DT_NEEDED` — its value is a strtab offset naming a
/// dependency module (M1-D, wall #4). OpenOrbis-style toolchains emit these
/// alongside the SCE tags; each entry is a `.prx` the module expects loaded.
const DT_NEEDED: u64 = 1;

// The standard (non-SCE) dynamic tags. Real PS5 titles use these — with
// **virtual addresses** — rather than the `DT_SCE_*` tags' offsets into a
// `PT_SCE_DYNLIBDATA` blob. See [`standard_dynamic_view`].
const DT_PLTRELSZ: u64 = 2;
const DT_STRTAB: u64 = 5;
const DT_SYMTAB: u64 = 6;
const DT_RELA: u64 = 7;
const DT_RELASZ: u64 = 8;
const DT_RELAENT: u64 = 9;
const DT_STRSZ: u64 = 10;
const DT_SYMENT: u64 = 11;
const DT_JMPREL: u64 = 23;

/// `DT_SCE_IMPORT_LIB`: one per library this module imports from. The value
/// packs `(library_id << 48) | (version << 32) | name_strtab_offset` — so it
/// maps a [`SymbolRef::library_index`] to a human-readable library name.
/// Verified against a real title (`0x0001_0101_000036d8` => id 1, name at
/// `0x36d8` = "libcohtml.Prospero.prx").
const DT_SCE_IMPORT_LIB: u64 = 0x6100_0045;

const DT_SCE_HASH: u64 = 0x6100_0025;
const DT_SCE_PLTRELSZ: u64 = 0x6100_002D;
const DT_SCE_JMPREL: u64 = 0x6100_0029;
const DT_SCE_RELA: u64 = 0x6100_002F;
const DT_SCE_RELASZ: u64 = 0x6100_0031;
const DT_SCE_RELAENT: u64 = 0x6100_0033;
const DT_SCE_STRTAB: u64 = 0x6100_0035;
const DT_SCE_STRSZ: u64 = 0x6100_0037;
const DT_SCE_SYMTAB: u64 = 0x6100_0039;
const DT_SCE_SYMENT: u64 = 0x6100_003B;
const DT_SCE_HASHSZ: u64 = 0x6100_003D;
const DT_SCE_SYMTABSZ: u64 = 0x6100_003F;

/// Size in bytes of an `Elf64_Sym` record.
const ELF64_SYM_SIZE: u64 = 24;
/// Size in bytes of an `Elf64_Rela` record.
const ELF64_RELA_SIZE: u64 = 24;
/// Size in bytes of an `Elf64_Dyn` `(d_tag, d_val)` pair.
const ELF64_DYN_SIZE: usize = 16;
/// `DT_NULL`: terminates the `Elf64_Dyn` array.
const DT_NULL: u64 = 0;

/// One import: an undefined symbol a module needs resolved (by HLE or by
/// another loaded module's export), identified by its NID plus the
/// best-effort library/module indices from its strtab name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolRef {
    pub nid: u64,
    pub module_index: u16,
    pub library_index: u16,
}

/// One export: a defined symbol this module provides to others, at a given
/// (module-relative) virtual address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolExport {
    pub nid: u64,
    pub value: u64,
}

/// One `Elf64_Rela` entry, decoded but not yet interpreted. `r_type = info
/// & 0xffff_ffff`; `r_sym = info >> 32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceRela {
    pub offset: u64,
    pub info: u64,
    pub addend: i64,
}

/// One `Elf64_Sym` record, decoded in original symtab order so that
/// `symbols[i]` corresponds to `r_sym == i` (the `r_sym = info >> 32` field
/// of an [`SceRela`]) — the index a symbol relocation names its symbol by.
/// This is a strict superset of the classification already split into
/// [`DynlibData::imports`]/[`DynlibData::exports`]: every symtab entry,
/// import or export, appears here at its table index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynSymbol {
    pub nid: u64,
    pub value: u64,
    pub is_import: bool,
}

/// The decoded contents of a `PT_SCE_DYNLIBDATA` blob.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DynlibData {
    pub imports: Vec<SymbolRef>,
    pub exports: Vec<SymbolExport>,
    /// Every decoded symtab entry, in table order — `symbols[i]` is the
    /// symbol named by `r_sym == i` in an [`SceRela`]'s `info` field.
    pub symbols: Vec<DynSymbol>,
    /// `DT_SCE_RELA` entries followed by `DT_SCE_JMPREL` entries, in that
    /// order.
    pub relocations: Vec<SceRela>,
    /// TODO(LM1+): the documented tag set consumed here does not include a
    /// `DT_SCE_NEEDED_MODULE`-style tag with a confirmed layout, so this is
    /// always empty until that's verified against a real module.
    pub needed_modules: Vec<String>,
    /// `library_index` -> library name, from the [`DT_SCE_IMPORT_LIB`] tags.
    ///
    /// This is what makes an unresolved import *actionable*: a NID is a one-way
    /// hash, so an unresolved one cannot be turned back into a function name —
    /// but every import symbol carries a [`SymbolRef::library_index`], and this
    /// maps that to a real name ("libSceAgc.prx", ...). Grouping unresolved
    /// imports by library turns "688 unknown NIDs" into a prioritized list of
    /// which libraries to implement.
    pub import_libs: Vec<(u16, String)>,
}

/// Decode the `Elf64_Dyn` `(d_tag: u64, d_val: u64)` array from a raw
/// `PT_DYNAMIC` segment, stopping at `DT_NULL`. Bounds-checked: a `PT_DYNAMIC`
/// array that runs out of bytes before reaching `DT_NULL` is
/// [`FirmwareError::MalformedDynlibData`], never a panic.
pub fn parse_sce_dynamic(dynamic_bytes: &[u8]) -> Result<Vec<(u64, u64)>, FirmwareError> {
    let mut tags = Vec::new();
    let mut offset = 0usize;
    loop {
        let end = offset.checked_add(ELF64_DYN_SIZE).ok_or_else(|| {
            FirmwareError::MalformedDynlibData("PT_DYNAMIC entry offset overflow".to_string())
        })?;
        if end > dynamic_bytes.len() {
            return Err(FirmwareError::MalformedDynlibData(format!(
                "PT_DYNAMIC array truncated before DT_NULL: entry at {offset:#x}, buffer size {:#x}",
                dynamic_bytes.len()
            )));
        }
        let chunk = &dynamic_bytes[offset..end];
        let d_tag = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let d_val = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
        if d_tag == DT_NULL {
            break;
        }
        tags.push((d_tag, d_val));
        offset = end;
    }
    Ok(tags)
}

/// Decode a `PT_SCE_DYNLIBDATA` blob into imports/exports/relocations, using
/// `dyn_tags` (from [`parse_sce_dynamic`]) to locate the string table,
/// symbol table, and relocation tables within `blob`.
///
/// # RE note (design §7 open item)
///
/// Exact SCE tag numbering and symbol-record layout are community-RE-derived
/// and may need iteration against real modules. This implements the
/// documented common tag set (`DT_SCE_STRTAB`/`STRSZ`, `SYMTAB`/`SYMTABSZ`/
/// `SYMENT`, `RELA`/`RELASZ`/`RELAENT`, `JMPREL`/`PLTRELSZ`). What's
/// structurally certain and asserted by tests: the NID, the defined/
/// undefined classification, and every relocation field. What's best-effort
/// (`// TODO(LM1+): verify index encoding`): the `lib_id`/`mod_id` indices
/// parsed from the `"<nid>#<lib_id>#<mod_id>"` strtab name convention.
/// What [`standard_dynamic_view`] hands to [`parse_dynlibdata`]: a
/// virtual-address-indexed image, and the `DT_SCE_*`-shaped tags whose values
/// are offsets into it.
pub type StandardDynamicView = (Vec<u8>, Vec<(u64, u64)>);

/// Present a module that uses the **standard, vaddr-based** dynamic model as
/// something [`parse_dynlibdata`] can consume.
///
/// Two dynamic models exist in the wild:
///
/// * **`PT_SCE_DYNLIBDATA` blob** — `DT_SCE_*` tags give *offsets into that
///   blob*. This is what homebrew/`.sprx` fixtures use, and what
///   [`parse_dynlibdata`] natively speaks.
/// * **Standard tags** — `DT_STRTAB`/`DT_SYMTAB`/`DT_RELA`/`DT_JMPREL` give
///   **virtual addresses**, and there is no `PT_SCE_DYNLIBDATA` segment at all.
///   Real PS5 titles use this (verified on a retail eboot: `DT_STRTAB` =
///   `0xe416840`, which is exactly a `PT_LOAD`'s `p_vaddr`).
///
/// Rather than duplicate the decoder, this flattens the `PT_LOAD` segments into
/// an image **indexed by virtual address** — so a vaddr *is* an offset into it —
/// and rewrites the standard tags to their `DT_SCE_*` equivalents with their
/// values unchanged. `parse_dynlibdata(&image, &tags)` then works verbatim.
///
/// Returns `None` if this module doesn't use the standard model (no
/// `DT_STRTAB`), so the caller can fall back to the blob path.
///
/// Size tags are carried across as-is; note real titles supply
/// `DT_SCE_SYMTABSZ` even while addressing the symtab via standard `DT_SYMTAB`
/// (standard ELF has no symtab-size tag), so both are consulted.
pub fn standard_dynamic_view(
    segments: &[crate::sprx::SprxSegment],
    dyn_tags: &[(u64, u64)],
) -> Option<StandardDynamicView> {
    // No standard string table => not this model; let the caller use the blob.
    tag_val(dyn_tags, DT_STRTAB)?;

    // Flatten to a vaddr-indexed image: index == virtual address.
    let end = segments
        .iter()
        .map(|s| s.vaddr.saturating_add(s.data.len() as u64))
        .max()?;
    let mut image = vec![0u8; usize::try_from(end).ok()?];
    for seg in segments {
        let at = usize::try_from(seg.vaddr).ok()?;
        let stop = at.checked_add(seg.data.len())?;
        image.get_mut(at..stop)?.copy_from_slice(&seg.data);
    }

    // Same value, SCE-equivalent tag: the vaddr is now an image offset.
    const MAP: &[(u64, u64)] = &[
        (DT_STRTAB, DT_SCE_STRTAB),
        (DT_STRSZ, DT_SCE_STRSZ),
        (DT_SYMTAB, DT_SCE_SYMTAB),
        (DT_SYMENT, DT_SCE_SYMENT),
        (DT_RELA, DT_SCE_RELA),
        (DT_RELASZ, DT_SCE_RELASZ),
        (DT_RELAENT, DT_SCE_RELAENT),
        (DT_JMPREL, DT_SCE_JMPREL),
        (DT_PLTRELSZ, DT_SCE_PLTRELSZ),
    ];
    let mut out: Vec<(u64, u64)> = Vec::new();
    for &(std_tag, sce_tag) in MAP {
        if let Some(v) = tag_val(dyn_tags, std_tag) {
            out.push((sce_tag, v));
        }
    }
    // Preserve any genuine SCE tags already present (e.g. DT_SCE_SYMTABSZ,
    // which real titles supply because standard ELF has no symtab-size tag),
    // plus DT_NEEDED so module names still resolve.
    //
    // Only the tags this function *mapped* above are skipped. Do NOT dedupe by
    // tag generally: DT_NEEDED and DT_SCE_IMPORT_LIB are inherently REPEATED
    // (one per dependency / imported library — ~50 each on a real title), so a
    // "keep the first of each tag" filter silently drops all but one, leaving
    // exactly one library name and one NEEDED entry.
    for &(t, v) in dyn_tags {
        let is_interesting = t == DT_NEEDED || (0x6100_0000..0x6200_0000).contains(&t);
        let already_mapped = MAP.iter().any(|&(_, sce)| sce == t);
        if is_interesting && !already_mapped {
            out.push((t, v));
        }
    }

    info!(
        "module uses the standard vaddr-based dynamic model ({} tag(s) mapped, {:#x}-byte vaddr image)",
        out.len(),
        image.len()
    );
    Some((image, out))
}

pub fn parse_dynlibdata(blob: &[u8], dyn_tags: &[(u64, u64)]) -> Result<DynlibData, FirmwareError> {
    let strtab = match (
        tag_val(dyn_tags, DT_SCE_STRTAB),
        tag_val(dyn_tags, DT_SCE_STRSZ),
    ) {
        (Some(off), Some(sz)) => slice_range(blob, off, sz, "DT_SCE_STRTAB")?,
        _ => &[],
    };

    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut symbols = Vec::new();
    if let (Some(sym_off), Some(sym_sz)) = (
        tag_val(dyn_tags, DT_SCE_SYMTAB),
        tag_val(dyn_tags, DT_SCE_SYMTABSZ),
    ) {
        let syment = tag_val(dyn_tags, DT_SCE_SYMENT).unwrap_or(ELF64_SYM_SIZE);
        let symtab = slice_range(blob, sym_off, sym_sz, "DT_SCE_SYMTAB")?;
        decode_symbols(
            symtab,
            syment,
            strtab,
            &mut imports,
            &mut exports,
            &mut symbols,
        )?;
    }

    let mut relocations = Vec::new();
    if let (Some(rela_off), Some(rela_sz)) = (
        tag_val(dyn_tags, DT_SCE_RELA),
        tag_val(dyn_tags, DT_SCE_RELASZ),
    ) {
        let relaent = tag_val(dyn_tags, DT_SCE_RELAENT).unwrap_or(ELF64_RELA_SIZE);
        let rela = slice_range(blob, rela_off, rela_sz, "DT_SCE_RELA")?;
        decode_relas(rela, relaent, &mut relocations)?;
    }
    if let (Some(jmp_off), Some(jmp_sz)) = (
        tag_val(dyn_tags, DT_SCE_JMPREL),
        tag_val(dyn_tags, DT_SCE_PLTRELSZ),
    ) {
        // No documented DT_SCE_JMPREL-specific entsize tag; DT_SCE_JMPREL
        // entries are Elf64_Rela records like DT_SCE_RELA.
        let jmprel = slice_range(blob, jmp_off, jmp_sz, "DT_SCE_JMPREL")?;
        decode_relas(jmprel, ELF64_RELA_SIZE, &mut relocations)?;
    }

    // DT_SCE_HASH/DT_SCE_HASHSZ are bounds-checked-if-present but otherwise
    // unused for LM1 (per the plan, the SCE symbol hash table may be
    // ignored for now).
    if let (Some(hash_off), Some(hash_sz)) = (
        tag_val(dyn_tags, DT_SCE_HASH),
        tag_val(dyn_tags, DT_SCE_HASHSZ),
    ) {
        slice_range(blob, hash_off, hash_sz, "DT_SCE_HASH")?;
    }

    // M1-D (wall #4): collect every `DT_NEEDED` dependency name from the
    // strtab. These are *recognized and surfaced* here; whether each is
    // HLE-covered (or needs a real file-backed load) is the loader's call —
    // see `load_module`'s NEEDED logging.
    let mut needed_modules = Vec::new();
    for &(tag, val) in dyn_tags {
        if tag != DT_NEEDED {
            continue;
        }
        let Ok(off) = usize::try_from(val) else {
            tracing::warn!("DT_NEEDED strtab offset {val:#x} overflows usize; skipping entry");
            continue;
        };
        match read_cstr(strtab, off) {
            Some(name) if !name.is_empty() => needed_modules.push(name),
            _ => {
                tracing::warn!(
                    "DT_NEEDED strtab offset {off:#x} is out of range (strtab len {}) or empty; skipping entry",
                    strtab.len()
                );
            }
        }
    }

    // `DT_SCE_IMPORT_LIB`: library_index -> name. See `DynlibData::import_libs`
    // for why this matters (it's what makes an unresolved NID actionable).
    let mut import_libs = Vec::new();
    for &(tag, val) in dyn_tags {
        if tag != DT_SCE_IMPORT_LIB {
            continue;
        }
        let id = (val >> 48) as u16;
        let off = (val & 0xFFFF_FFFF) as usize;
        match read_cstr(strtab, off) {
            Some(name) if !name.is_empty() => import_libs.push((id, name)),
            _ => tracing::debug!(
                "DT_SCE_IMPORT_LIB id {id} name offset {off:#x} is out of range (strtab len {}) \
                 or empty; skipping",
                strtab.len()
            ),
        }
    }

    Ok(DynlibData {
        imports,
        exports,
        symbols,
        relocations,
        needed_modules,
        import_libs,
    })
}

/// Look up the (first) value for `tag` in `dyn_tags`.
fn tag_val(dyn_tags: &[(u64, u64)], tag: u64) -> Option<u64> {
    dyn_tags.iter().find(|(t, _)| *t == tag).map(|(_, v)| *v)
}

/// Slice `blob[offset .. offset + size]`, bounds-checked. Never panics;
/// returns [`FirmwareError::MalformedDynlibData`] on overflow or
/// out-of-bounds ranges.
fn slice_range<'a>(
    blob: &'a [u8],
    offset: u64,
    size: u64,
    label: &str,
) -> Result<&'a [u8], FirmwareError> {
    let start = usize::try_from(offset).map_err(|_| {
        FirmwareError::MalformedDynlibData(format!("{label} offset {offset:#x} overflows usize"))
    })?;
    let len = usize::try_from(size).map_err(|_| {
        FirmwareError::MalformedDynlibData(format!("{label} size {size:#x} overflows usize"))
    })?;
    let end = start.checked_add(len).ok_or_else(|| {
        FirmwareError::MalformedDynlibData(format!("{label} offset+size overflow"))
    })?;
    if end > blob.len() {
        return Err(FirmwareError::MalformedDynlibData(format!(
            "{label} range [{start:#x}, {end:#x}) exceeds blob size {:#x}",
            blob.len()
        )));
    }
    Ok(&blob[start..end])
}

/// Read a NUL-terminated (or table-end-terminated) string out of `strtab`
/// starting at `start`. Returns `None` if `start` is past the end of the
/// table (bounds-checked, never panics/indexes out of range).
fn read_cstr(strtab: &[u8], start: usize) -> Option<String> {
    if start > strtab.len() {
        return None;
    }
    let end = strtab[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|p| start + p)
        .unwrap_or(strtab.len());
    Some(String::from_utf8_lossy(&strtab[start..end]).into_owned())
}

/// Decode `symtab` as a sequence of `entsize`-byte `Elf64_Sym` records,
/// resolving each symbol's name against `strtab` and classifying it as an
/// import (undefined) or export (defined).
///
/// `symbols` receives one [`DynSymbol`] per symtab record, **in table
/// order**, regardless of whether the record's name decoded as an NID — a
/// relocation's `r_sym = info >> 32` indexes the *original* `Elf64_Sym`
/// table, so `symbols[i]` must line up with index `i` even for entries this
/// function otherwise skips out of `imports`/`exports`. A record whose name
/// doesn't decode gets a `nid: 0` placeholder (never a panic, never a
/// dropped index).
fn decode_symbols(
    symtab: &[u8],
    entsize: u64,
    strtab: &[u8],
    imports: &mut Vec<SymbolRef>,
    exports: &mut Vec<SymbolExport>,
    symbols: &mut Vec<DynSymbol>,
) -> Result<(), FirmwareError> {
    let entsize = usize::try_from(entsize).map_err(|_| {
        FirmwareError::MalformedDynlibData("DT_SCE_SYMENT overflows usize".to_string())
    })?;
    if (entsize as u64) < ELF64_SYM_SIZE {
        return Err(FirmwareError::MalformedDynlibData(format!(
            "DT_SCE_SYMENT {entsize} is smaller than Elf64_Sym ({ELF64_SYM_SIZE})"
        )));
    }
    if entsize == 0 || !symtab.len().is_multiple_of(entsize) {
        return Err(FirmwareError::MalformedDynlibData(format!(
            "symbol table size {} is not a multiple of entsize {entsize} (truncated symbol record)",
            symtab.len()
        )));
    }

    for chunk in symtab.chunks(entsize) {
        let st_name = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let st_shndx = u16::from_le_bytes(chunk[6..8].try_into().unwrap());
        let st_value = u64::from_le_bytes(chunk[8..16].try_into().unwrap());

        let is_import = st_shndx == 0 || st_value == 0;

        let Some(name) = read_cstr(strtab, st_name as usize) else {
            warn!(
                "symbol st_name {st_name:#x} is out of range of the strtab (len {}); skipping \
                 classification but keeping its symtab index (r_sym alignment)",
                strtab.len()
            );
            symbols.push(DynSymbol {
                nid: 0,
                value: st_value,
                is_import,
            });
            continue;
        };

        let (nid_part, rest) = match name.split_once('#') {
            Some((n, r)) => (n, Some(r)),
            None => (name.as_str(), None),
        };

        let Some(symbol_nid) = nid::decode_nid(nid_part) else {
            debug!(
                "symbol name {name:?} does not decode as an SCE NID; skipping classification but \
                 keeping its symtab index (r_sym alignment)"
            );
            symbols.push(DynSymbol {
                nid: 0,
                value: st_value,
                is_import,
            });
            continue;
        };

        symbols.push(DynSymbol {
            nid: symbol_nid,
            value: st_value,
            is_import,
        });

        if st_shndx == 0 || st_value == 0 {
            // TODO(LM1+): verify index encoding. `rest` is expected to be
            // "<lib_id>#<mod_id>" per the documented SCE strtab convention,
            // but `decode_nid` requires >=8 decoded bytes and single/double
            // -character indices decode to fewer, so this commonly falls
            // back to 0 until the real (likely much simpler) index encoding
            // is confirmed against a real module.
            let mut parts = rest.unwrap_or("").split('#');
            // `decode_index`, not `decode_nid`: these are short, variable-length
            // fields (`<nid>#<lib>#<mod>`, e.g. `rTXw65xmLIA#l#l`), and
            // `decode_nid` requires >= 8 decoded bytes so it returned None for
            // every one of them — silently making every import's library index
            // 0, which matches no DT_SCE_IMPORT_LIB entry (real ids start at 1).
            let library_index = parts.next().and_then(nid::decode_index).unwrap_or(0);
            let module_index = parts.next().and_then(nid::decode_index).unwrap_or(0);
            imports.push(SymbolRef {
                nid: symbol_nid,
                module_index,
                library_index,
            });
        } else {
            exports.push(SymbolExport {
                nid: symbol_nid,
                value: st_value,
            });
        }
    }

    Ok(())
}

/// Decode `data` as a sequence of `entsize`-byte `Elf64_Rela` records,
/// appending each to `out`.
fn decode_relas(data: &[u8], entsize: u64, out: &mut Vec<SceRela>) -> Result<(), FirmwareError> {
    let entsize = usize::try_from(entsize).map_err(|_| {
        FirmwareError::MalformedDynlibData("relocation entsize overflows usize".to_string())
    })?;
    if (entsize as u64) < ELF64_RELA_SIZE {
        return Err(FirmwareError::MalformedDynlibData(format!(
            "relocation entsize {entsize} is smaller than Elf64_Rela ({ELF64_RELA_SIZE})"
        )));
    }
    if entsize == 0 || !data.len().is_multiple_of(entsize) {
        return Err(FirmwareError::MalformedDynlibData(format!(
            "relocation table size {} is not a multiple of entsize {entsize} (truncated relocation record)",
            data.len()
        )));
    }

    for chunk in data.chunks(entsize) {
        let offset = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let info = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
        let addend = i64::from_le_bytes(chunk[16..24].try_into().unwrap());
        out.push(SceRela {
            offset,
            info,
            addend,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::nid::{encode_nid, nid_of};
    use super::*;

    /// M1-D (wall #4): `DT_NEEDED` entries resolve their strtab names into
    /// `needed_modules`; an out-of-range offset is skipped, not fatal.
    #[test]
    fn dt_needed_entries_are_collected_from_the_strtab() {
        let mut strtab = vec![0u8];
        let lib_a_off = strtab.len() as u64;
        strtab.extend_from_slice(b"libSceLibcInternal.sprx\0");
        let lib_b_off = strtab.len() as u64;
        strtab.extend_from_slice(b"libSceFios2.sprx\0");
        let blob = strtab.clone();

        let dyn_tags = vec![
            (DT_SCE_STRTAB, 0u64),
            (DT_SCE_STRSZ, strtab.len() as u64),
            (DT_NEEDED, lib_a_off),
            (DT_NEEDED, lib_b_off),
            (DT_NEEDED, 0x9999), // out of range: skipped with a warning
        ];

        let dynlib = parse_dynlibdata(&blob, &dyn_tags).expect("parses");
        assert_eq!(
            dynlib.needed_modules,
            vec![
                "libSceLibcInternal.sprx".to_string(),
                "libSceFios2.sprx".to_string()
            ]
        );
    }

    /// Build a synthetic dynlibdata blob laid out as: strtab, symtab, RELA
    /// table, JMPREL table (each section contiguous). Returns the blob plus
    /// the `dyn_tags` describing it. Entirely synthetic — no real firmware
    /// bytes.
    struct Fixture {
        blob: Vec<u8>,
        dyn_tags: Vec<(u64, u64)>,
        import_nid: u64,
        export_nid: u64,
    }

    fn build_fixture() -> Fixture {
        let import_name = format!("{}#A#A", encode_nid(nid_of("someImport")));
        let export_name = encode_nid(nid_of("someExport"));

        // strtab: [0] reserved null byte, then each NUL-terminated name.
        let mut strtab = vec![0u8];
        let import_off = strtab.len() as u32;
        strtab.extend_from_slice(import_name.as_bytes());
        strtab.push(0);
        let export_off = strtab.len() as u32;
        strtab.extend_from_slice(export_name.as_bytes());
        strtab.push(0);

        // symtab: one undefined (import) symbol, one defined (export) symbol.
        let mut symtab = Vec::new();
        // Import: st_name, st_info=0, st_other=0, st_shndx=0, st_value=0, st_size=0.
        symtab.extend_from_slice(&import_off.to_le_bytes());
        symtab.push(0);
        symtab.push(0);
        symtab.extend_from_slice(&0u16.to_le_bytes());
        symtab.extend_from_slice(&0u64.to_le_bytes());
        symtab.extend_from_slice(&0u64.to_le_bytes());
        // Export: st_name, st_info=0x11, st_other=0, st_shndx=1, st_value=0x2000, st_size=8.
        symtab.extend_from_slice(&export_off.to_le_bytes());
        symtab.push(0x11);
        symtab.push(0);
        symtab.extend_from_slice(&1u16.to_le_bytes());
        symtab.extend_from_slice(&0x2000u64.to_le_bytes());
        symtab.extend_from_slice(&8u64.to_le_bytes());

        // One DT_SCE_RELA entry.
        let mut rela = Vec::new();
        rela.extend_from_slice(&0x1000u64.to_le_bytes()); // r_offset
        rela.extend_from_slice(&((5u64 << 32) | 7u64).to_le_bytes()); // r_info: sym=5, type=7
        rela.extend_from_slice(&0x10i64.to_le_bytes()); // r_addend

        // One DT_SCE_JMPREL entry.
        let mut jmprel = Vec::new();
        jmprel.extend_from_slice(&0x2000u64.to_le_bytes());
        jmprel.extend_from_slice(&((3u64 << 32) | 1u64).to_le_bytes());
        jmprel.extend_from_slice(&0i64.to_le_bytes());

        let strtab_off = 0u64;
        let symtab_off = strtab.len() as u64;
        let rela_off = symtab_off + symtab.len() as u64;
        let jmprel_off = rela_off + rela.len() as u64;

        let mut blob = Vec::new();
        blob.extend_from_slice(&strtab);
        blob.extend_from_slice(&symtab);
        blob.extend_from_slice(&rela);
        blob.extend_from_slice(&jmprel);

        let dyn_tags = vec![
            (DT_SCE_STRTAB, strtab_off),
            (DT_SCE_STRSZ, strtab.len() as u64),
            (DT_SCE_SYMTAB, symtab_off),
            (DT_SCE_SYMTABSZ, symtab.len() as u64),
            (DT_SCE_SYMENT, ELF64_SYM_SIZE),
            (DT_SCE_RELA, rela_off),
            (DT_SCE_RELASZ, rela.len() as u64),
            (DT_SCE_RELAENT, ELF64_RELA_SIZE),
            (DT_SCE_JMPREL, jmprel_off),
            (DT_SCE_PLTRELSZ, jmprel.len() as u64),
        ];

        Fixture {
            blob,
            dyn_tags,
            import_nid: nid_of("someImport"),
            export_nid: nid_of("someExport"),
        }
    }

    #[test]
    fn parses_imports_exports_and_relocations_in_order() {
        let fx = build_fixture();
        let data = parse_dynlibdata(&fx.blob, &fx.dyn_tags).expect("synthetic dynlibdata parses");

        assert_eq!(data.imports.len(), 1);
        assert_eq!(data.imports[0].nid, fx.import_nid);

        assert_eq!(data.exports.len(), 1);
        assert_eq!(data.exports[0].nid, fx.export_nid);
        assert_eq!(data.exports[0].value, 0x2000);

        // `symbols` is the ordered symtab: index 0 is the import symbol,
        // index 1 is the export symbol, matching the fixture's symtab layout.
        assert_eq!(data.symbols.len(), 2);
        assert_eq!(
            data.symbols[0],
            DynSymbol {
                nid: fx.import_nid,
                value: 0,
                is_import: true,
            }
        );
        assert_eq!(
            data.symbols[1],
            DynSymbol {
                nid: fx.export_nid,
                value: 0x2000,
                is_import: false,
            }
        );

        assert_eq!(data.relocations.len(), 2);
        assert_eq!(
            data.relocations[0],
            SceRela {
                offset: 0x1000,
                info: (5u64 << 32) | 7u64,
                addend: 0x10,
            }
        );
        assert_eq!(
            data.relocations[1],
            SceRela {
                offset: 0x2000,
                info: (3u64 << 32) | 1u64,
                addend: 0,
            }
        );
    }

    /// `SceRela::info >> 32` (`r_sym`) indexes the *original* symtab, so
    /// `DynlibData::symbols[r_sym]` must be the symbol the relocation names
    /// — even when that symbol isn't the first or only entry in the table.
    #[test]
    fn symbols_are_indexable_by_relocation_r_sym() {
        // strtab: two throwaway names before the target, so its symtab
        // index (2) isn't trivially 0.
        let mut strtab = vec![0u8];
        let mut name_offsets = Vec::new();
        for n in ["dummyA", "dummyB", "targetImport"] {
            let off = strtab.len() as u32;
            strtab.extend_from_slice(encode_nid(nid_of(n)).as_bytes());
            strtab.push(0);
            name_offsets.push(off);
        }

        // symtab: three undefined (import) symbols; the target is index 2.
        let mut symtab = Vec::new();
        for &name_off in &name_offsets {
            symtab.extend_from_slice(&name_off.to_le_bytes());
            symtab.push(0);
            symtab.push(0);
            symtab.extend_from_slice(&0u16.to_le_bytes());
            symtab.extend_from_slice(&0u64.to_le_bytes());
            symtab.extend_from_slice(&0u64.to_le_bytes());
        }

        // One RELA entry whose r_sym (2) names the target symbol.
        let r_sym: u64 = 2;
        let r_type: u64 = 1; // R_X86_64_64
        let mut rela = Vec::new();
        rela.extend_from_slice(&0x100u64.to_le_bytes());
        rela.extend_from_slice(&((r_sym << 32) | r_type).to_le_bytes());
        rela.extend_from_slice(&0i64.to_le_bytes());

        let strtab_off = 0u64;
        let symtab_off = strtab.len() as u64;
        let rela_off = symtab_off + symtab.len() as u64;

        let mut blob = Vec::new();
        blob.extend_from_slice(&strtab);
        blob.extend_from_slice(&symtab);
        blob.extend_from_slice(&rela);

        let dyn_tags = vec![
            (DT_SCE_STRTAB, strtab_off),
            (DT_SCE_STRSZ, strtab.len() as u64),
            (DT_SCE_SYMTAB, symtab_off),
            (DT_SCE_SYMTABSZ, symtab.len() as u64),
            (DT_SCE_SYMENT, ELF64_SYM_SIZE),
            (DT_SCE_RELA, rela_off),
            (DT_SCE_RELASZ, rela.len() as u64),
            (DT_SCE_RELAENT, ELF64_RELA_SIZE),
        ];

        let data = parse_dynlibdata(&blob, &dyn_tags).expect("synthetic dynlibdata parses");
        assert_eq!(data.symbols.len(), 3);

        let reloc = data.relocations[0];
        let resolved_r_sym = (reloc.info >> 32) as usize;
        assert_eq!(resolved_r_sym, 2);
        assert_eq!(data.symbols[resolved_r_sym].nid, nid_of("targetImport"));
        assert!(data.symbols[resolved_r_sym].is_import);
    }

    #[test]
    fn table_offset_past_blob_end_errors_not_panics() {
        let fx = build_fixture();
        let mut dyn_tags = fx.dyn_tags.clone();
        // Push DT_SCE_STRTAB way out of bounds.
        for tag in dyn_tags.iter_mut() {
            if tag.0 == DT_SCE_STRTAB {
                tag.1 = fx.blob.len() as u64 + 1000;
            }
        }
        let err = parse_dynlibdata(&fx.blob, &dyn_tags).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn table_size_past_blob_end_errors_not_panics() {
        let fx = build_fixture();
        let mut dyn_tags = fx.dyn_tags.clone();
        for tag in dyn_tags.iter_mut() {
            if tag.0 == DT_SCE_SYMTABSZ {
                tag.1 += 10_000;
            }
        }
        let err = parse_dynlibdata(&fx.blob, &dyn_tags).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn truncated_symbol_record_errors_not_panics() {
        let fx = build_fixture();
        let mut dyn_tags = fx.dyn_tags.clone();
        for tag in dyn_tags.iter_mut() {
            if tag.0 == DT_SCE_SYMTABSZ {
                // Not a multiple of 24: a truncated trailing record.
                tag.1 = 10;
            }
        }
        let err = parse_dynlibdata(&fx.blob, &dyn_tags).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn truncated_rela_record_errors_not_panics() {
        let fx = build_fixture();
        let mut dyn_tags = fx.dyn_tags.clone();
        for tag in dyn_tags.iter_mut() {
            if tag.0 == DT_SCE_RELASZ {
                tag.1 = 10;
            }
        }
        let err = parse_dynlibdata(&fx.blob, &dyn_tags).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn missing_tags_yield_empty_data_not_error() {
        // No dyn_tags at all: nothing to decode, but not malformed.
        let data = parse_dynlibdata(&[], &[]).expect("empty tags is not an error");
        assert!(data.imports.is_empty());
        assert!(data.exports.is_empty());
        assert!(data.relocations.is_empty());
        assert!(data.needed_modules.is_empty());
    }

    #[test]
    fn parse_sce_dynamic_reads_entries_until_dt_null() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&DT_SCE_STRTAB.to_le_bytes());
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&DT_SCE_STRSZ.to_le_bytes());
        bytes.extend_from_slice(&0x20u64.to_le_bytes());
        bytes.extend_from_slice(&DT_NULL.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        // Trailing garbage after DT_NULL must be ignored.
        bytes.extend_from_slice(&[0xFFu8; 16]);

        let tags = parse_sce_dynamic(&bytes).expect("well-formed Elf64_Dyn array parses");
        assert_eq!(tags, vec![(DT_SCE_STRTAB, 0x100), (DT_SCE_STRSZ, 0x20)]);
    }

    #[test]
    fn parse_sce_dynamic_truncated_before_dt_null_errors() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&DT_SCE_STRTAB.to_le_bytes());
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        // Cut off partway through the second entry; no DT_NULL reached.
        bytes.extend_from_slice(&DT_SCE_STRSZ.to_le_bytes());

        let err = parse_sce_dynamic(&bytes).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }

    #[test]
    fn parse_sce_dynamic_empty_input_errors() {
        // Zero bytes: never reaches a DT_NULL terminator.
        let err = parse_sce_dynamic(&[]).unwrap_err();
        assert!(matches!(err, FirmwareError::MalformedDynlibData(_)));
    }
}
