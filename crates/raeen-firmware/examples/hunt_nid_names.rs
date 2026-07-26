//! Targeted dictionary attack on anonymous NIDs.
//!
//! Some imports have no hash-proven name in any public list (PS5-internal
//! `libSceAgc` helpers, `libSceSaveData` native exports, …). Public symbol
//! names in one library family are built from a small CamelCase vocabulary,
//! so: extract each family's token vocabulary from the *existing* verified
//! catalog, generate `prefix + 1..=3 tokens` candidates, and hash every one.
//! A candidate that lands on a target NID is a proven preimage — the same
//! admission gate `nid_names.txt` itself uses (see `nid_names.rs`). No
//! external data, no guessing into the table: only hash-verified hits are
//! reported, and `--write` folds them into the catalog.
//!
//! Usage (from repo root):
//!   cargo run -p raeen-firmware --example hunt_nid_names -- [--deep] [--write] \
//!     [hex-nid ...]
//!
//! With no hex arguments, hunts the nine anonymous NIDs measured by
//! `cargo xtask nids coverage` on 2026-07-25 (six `libSceAgc`, one
//! `libSceAudioIn`, one `libSceSaveData_native`, one `libSceVideoRecordingP`).
//! `--deep` adds a pruned 4-token pass for families that still have targets.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use raeen_firmware::dynlib::nid::nid_of;

/// The nine anonymous NIDs from the 2026-07-25 nine-game coverage run.
const MEASURED_TARGETS: [&str; 9] = [
    "2247ddb7fac8a821", // libSceAgc (GTA V, Avatar)
    "81092a90bb6d729c", // libSceAgc (GTA V)
    "93413bbe482a02e1", // libSceAgc (GTA V)
    "acfe712dd39fdba9", // libSceAgc (GTA V)
    "cc0451e5a0a69286", // libSceAgc (GTA V)
    "ed66b769e2607955", // libSceAgc (GTA V)
    "5fee237484bbe4fd", // libSceAudioIn (ASTRO.BOT)
    "46f203f0155e83e8", // libSceSaveData_native (Minecraft)
    "8904ba0d4b4bc9b1", // libSceVideoRecordingP (GTA V)
];

/// Name prefixes to hunt, mapped to the provider that imports them. Family
/// vocabularies come from verified catalog names with the same prefix.
const FAMILIES: [(&str, &str); 4] = [
    ("sceAgc", "libSceAgc"),
    ("sceAudioIn", "libSceAudioIn"),
    ("sceSaveData", "libSceSaveData"),
    ("sceVideoRecording", "libSceVideoRecordingP"),
];

/// Split a CamelCase tail into tokens: case boundaries and alpha/digit
/// boundaries (`DcbDrawIndex2Auto` -> `Dcb Draw Index 2 Auto`). Only feeds
/// candidate generation; the hash gate decides what is true.
fn tokenize(tail: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut prev: Option<char> = None;
    for ch in tail.chars() {
        let boundary = match prev {
            Some(p) => {
                (p.is_lowercase() && ch.is_uppercase()) || (p.is_alphabetic() != ch.is_alphabetic())
            }
            None => false,
        };
        if boundary && !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
        current.push(ch);
        prev = Some(ch);
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let deep = args.iter().any(|arg| arg == "--deep");
    let write = args.iter().any(|arg| arg == "--write");
    let target_args: Vec<&str> = args
        .iter()
        .filter(|arg| !arg.starts_with("--"))
        .map(String::as_str)
        .collect();
    let target_hex: Vec<&str> = if target_args.is_empty() {
        MEASURED_TARGETS.to_vec()
    } else {
        target_args
    };
    let mut remaining: HashSet<u64> = target_hex
        .iter()
        .map(|hex| u64::from_str_radix(hex, 16).expect("hex nid"))
        .collect();
    println!("hunting {} anonymous NIDs", remaining.len());

    // 1) Per-family token vocabularies from the verified catalog.
    let catalog_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/dynlib/nid_names.txt");
    let catalog = std::fs::read_to_string(&catalog_path).expect("read nid_names.txt");
    let mut vocab: HashMap<&str, Vec<String>> = HashMap::new();
    let mut hits: BTreeMap<u64, String> = BTreeMap::new();
    for (prefix, _) in FAMILIES {
        let mut freq: BTreeMap<String, usize> = BTreeMap::new();
        for line in catalog.lines() {
            let Some((_, name)) = line.split_once(' ') else {
                continue;
            };
            if let Some(tail) = name.strip_prefix(prefix) {
                for token in tokenize(tail) {
                    *freq.entry(token).or_insert(0) += 1;
                }
            }
        }
        let mut tokens: Vec<(String, usize)> = freq.into_iter().collect();
        tokens.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let tokens: Vec<String> = tokens.into_iter().map(|(token, _)| token).collect();
        println!("{prefix}: {} tokens from catalog", tokens.len());
        vocab.insert(prefix, tokens);
    }

    // 2) Candidate stream: prefix + 1..=3 tokens (no repeated token), each
    //    also tried with the common `GetSize` suffix. Early-exit per family
    //    once its targets are all found.
    let mut tried = 0u64;
    for (prefix, _) in FAMILIES {
        let tokens = &vocab[prefix];
        let family_targets: Vec<u64> = remaining.iter().copied().collect();
        if family_targets.is_empty() {
            break;
        }
        let mut check = |candidate: String, remaining: &mut HashSet<u64>| {
            tried += 1;
            let nid = nid_of(&candidate);
            if remaining.remove(&nid) {
                println!("HIT {nid:016x} -> {candidate}");
                hits.insert(nid, candidate);
            }
        };
        for a in tokens {
            check(format!("{prefix}{a}"), &mut remaining);
            check(format!("{prefix}{a}GetSize"), &mut remaining);
            for b in tokens {
                if b == a {
                    continue;
                }
                check(format!("{prefix}{a}{b}"), &mut remaining);
                check(format!("{prefix}{a}{b}GetSize"), &mut remaining);
                for c in tokens {
                    if c == a || c == b {
                        continue;
                    }
                    check(format!("{prefix}{a}{b}{c}"), &mut remaining);
                    check(format!("{prefix}{a}{b}{c}GetSize"), &mut remaining);
                }
            }
        }
        println!("{prefix} pass done; {} targets remain", remaining.len());
    }

    // 3) Optional deep pass: 4 tokens from each family's 40 most common
    //    tokens (40^4 = 2.56M per family, no suffix — bounded).
    if deep && !remaining.is_empty() {
        for (prefix, _) in FAMILIES {
            let tokens: Vec<&String> = vocab[prefix].iter().take(40).collect();
            for a in &tokens {
                for b in &tokens {
                    for c in &tokens {
                        for d in &tokens {
                            tried += 1;
                            let candidate = format!("{prefix}{a}{b}{c}{d}");
                            let nid = nid_of(&candidate);
                            if remaining.remove(&nid) {
                                println!("HIT {nid:016x} -> {candidate}");
                                hits.insert(nid, candidate);
                            }
                        }
                    }
                }
            }
            println!(
                "{prefix} deep pass done; {} targets remain",
                remaining.len()
            );
        }
    }

    println!("tried {tried} candidates");
    if !remaining.is_empty() {
        let mut remaining: Vec<u64> = remaining.into_iter().collect();
        remaining.sort();
        for nid in &remaining {
            println!("still anonymous: {nid:016x}");
        }
    }

    // 4) Fold verified hits into the catalog (identical shape/gate as
    //    merge_nid_catalog: existing entries win, sorted by NID).
    if write && !hits.is_empty() {
        let mut table: BTreeMap<u64, String> = BTreeMap::new();
        for line in catalog.lines() {
            if let Some((nid, name)) = line.split_once(' ')
                && let Ok(nid) = u64::from_str_radix(nid, 16)
            {
                table.insert(nid, name.to_owned());
            }
        }
        let mut added = 0usize;
        for (nid, name) in &hits {
            if table.insert(*nid, name.clone()).is_none() {
                added += 1;
            }
        }
        let mut out = String::with_capacity(table.len() * 48);
        for (nid, name) in &table {
            out.push_str(&format!("{nid:016x} {name}\n"));
        }
        std::fs::write(&catalog_path, out).expect("write nid_names.txt");
        println!("folded {added} verified names into the catalog");
    } else if !hits.is_empty() {
        println!("dry run — re-run with --write to fold hits into the catalog");
    }
}
