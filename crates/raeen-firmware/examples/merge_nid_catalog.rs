//! Merge a candidate name list into the verified NID catalog.
//!
//! Raeen's `nid_names.txt` is a hash-verified `<hex-nid> <name>` table
//! (shadPS4-derived, 94,247 entries). SharpEmu ships a much larger candidate
//! name list (`scripts/ps5_names.txt`, ~154k names). Because a NID is
//! `SHA-1(name || salt)`, a name that hashes to a NID *is* its preimage — so
//! we can admit SharpEmu's names through the identical hash gate the existing
//! table uses, with no trust in SharpEmu's own labels. A verified run of a real
//! 2026-07-16 boot showed 18 imports our table could not name; this recovers 11
//! of them, including the whole `libSceNpAuthAuthorizedAppDialog` set and
//! `sceAgcGetIsTrinityMode`.
//!
//! Usage (from repo root):
//!   cargo run -p raeen-firmware --example merge_nid_catalog -- \
//!     C:/Users/whoisraeen/Documents/sharpemu/scripts/ps5_names.txt
//!
//! Existing entries always win a NID collision (SharpEmu lacks only
//! `__sys_dynlib_unload_prx`, which ours keeps). The `all_names_hash_to_their_nid`
//! test re-proves the merged table, so a bad candidate name can only fail to
//! enter — it can never corrupt the table.
//!
//! Provenance: SharpEmu name list (GPL-2.0-or-later), attributed in
//! THIRD_PARTY_NOTICES.md. Public symbol names only — no Sony code/keys.

use std::collections::BTreeMap;
use std::path::Path;

use raeen_firmware::dynlib::nid::nid_of;

fn main() {
    let candidate_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: merge_nid_catalog <candidate-names.txt>");
        std::process::exit(2);
    });

    // The catalog lives next to its source, discovered relative to the crate.
    let catalog = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/dynlib/nid_names.txt");

    // 1) Load the existing verified table. Existing names win every collision.
    let existing = std::fs::read_to_string(&catalog).expect("read nid_names.txt");
    let mut table: BTreeMap<u64, String> = BTreeMap::new();
    for line in existing.lines() {
        if let Some((nid, name)) = line.split_once(' ')
            && let Ok(nid) = u64::from_str_radix(nid, 16)
        {
            table.insert(nid, name.to_owned());
        }
    }
    let before = table.len();

    // 2) Fold in the candidate names through the SAME hash gate. Skip only
    //    provable non-imports: `.L*` assembler-local labels and leading-`/`
    //    path artifacts can never be a cross-module NID, so they are pure noise.
    let candidates = std::fs::read_to_string(&candidate_path).expect("read candidate list");
    let (mut added, mut skipped_junk) = (0usize, 0usize);
    for raw in candidates.lines() {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        if name.starts_with(".L") || name.starts_with('/') {
            skipped_junk += 1;
            continue;
        }
        let nid = nid_of(name);
        // `entry().or_insert()` keeps the existing name on any collision — the
        // verified table is authoritative where the two overlap.
        table.entry(nid).or_insert_with(|| {
            added += 1;
            name.to_owned()
        });
    }

    // 3) Write the merged table, sorted by NID (BTreeMap iteration order),
    //    matching the file's existing shape exactly.
    let mut out = String::with_capacity(table.len() * 48);
    for (nid, name) in &table {
        out.push_str(&format!("{nid:016x} {name}\n"));
    }
    std::fs::write(&catalog, out).expect("write nid_names.txt");

    eprintln!("existing: {before}");
    eprintln!("skipped junk (.L*, /*): {skipped_junk}");
    eprintln!("added (new hash-verified names): {added}");
    eprintln!("total: {}", table.len());

    // Spot-check the decisive recoveries from the real boot log.
    for (nid_hex, want) in [
        ("05f0436466ed8bb0", "sceAgcGetIsTrinityMode"),
        ("f42895ed2872eaa4", "sceNpAuthAuthorizedAppDialogInitialize"),
        ("ab6cbfc032155990", "sceKernelSyncOnAddressWake"),
    ] {
        let nid = u64::from_str_radix(nid_hex, 16).unwrap();
        match table.get(&nid) {
            Some(name) if name == want => eprintln!("  recovered {nid_hex} -> {name}"),
            Some(other) => eprintln!("  WARN {nid_hex} -> {other} (expected {want})"),
            None => eprintln!("  MISSING {nid_hex} ({want})"),
        }
    }
}
