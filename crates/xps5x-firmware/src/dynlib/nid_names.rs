//! NID → symbol-name recovery.
//!
//! A NID is `SHA-1(name || salt)` truncated to 8 bytes (see [`super::nid`]), so
//! it is a one-way hash: given only a NID, the name cannot be computed. Every
//! unresolved import a title asks for is therefore anonymous by construction,
//! and an anonymous import cannot be implemented — you cannot write
//! `sceKernelGetGPI` if all you have is `0xe285d87bd5e69344`. That was the
//! standing binding constraint on all HLE work.
//!
//! The way out is a **dictionary attack**: take a large list of candidate
//! names, hash each one, and keep the ones that land on a NID we care about.
//! [`NID_NAMES`] is the precomputed result of doing exactly that.
//!
//! ## Why this is trustworthy (and why the dictionary's accuracy is irrelevant)
//!
//! The candidate list came from a third party, but **no entry is taken on
//! faith**. An entry is admitted only if our own [`super::nid::nid_of`]
//! reproduces its NID from its name. Since a NID is a SHA-1 hash, a name that
//! hashes to it *is* a preimage — that is proof, not evidence. A wrong, stale,
//! or malicious dictionary entry cannot enter the table; it can only fail to.
//! `all_names_hash_to_their_nid` re-proves the whole table from scratch on
//! every test run, so this property is enforced, not just documented.
//!
//! This distinction matters, because name lists differ in kind. Lists that
//! *bind* a human label to a NID string (SharpEmu's `ExportName`, Kyty's
//! `LIB_FUNC` C++ symbols) never have to hash correctly and demonstrably do
//! not — SharpEmu labels `HV4j+E0MBHE` as `sceAgcCreateInterpolantMapping`,
//! but that name hashes to `pdEV7bI6COI`. Such a label is a guess. A
//! hash-verified name is not. Only hash-verified names belong here.
//!
//! ## Provenance
//!
//! Candidates come from two community sources, both admitted only through the
//! hash gate above (existing names win every collision, so the first source
//! stays authoritative where the two overlap):
//!
//! - shadPS4's `src/core/aerolib/aerolib.inl` (GPL-2.0-or-later, © 2024 shadPS4
//!   Emulator Project) — 94,247 of its 94,276 entries pass.
//! - SharpEmu's `scripts/ps5_names.txt` (GPL-2.0-or-later) — a ~154k candidate
//!   name list; folding it in through the same gate added 55,658 new
//!   hash-verified names (149,905 total), recovering symbols the first source
//!   lacked, notably the whole `libSceNpAuthAuthorizedAppDialog` set and
//!   `sceAgcGetIsTrinityMode`. `.L*` assembler-locals and `/`-prefixed path
//!   artifacts are dropped (never cross-module NIDs).
//!
//! Both are permitted by `.claude/skills/clean-room` ("NID names from community
//! databases OK") and attributed in `THIRD_PARTY_NOTICES.md`. Regenerate the
//! first source with `gen_nid_names.py`; fold in the second with the
//! `merge_nid_catalog` example (both kept beside the data for auditability).
//!
//! These are public symbol names, not Sony code — no SDK headers, blobs, or
//! keys are involved.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Hash-verified `<hex-nid> <name>` pairs, one per line, sorted by NID.
const NID_NAMES: &str = include_str!("nid_names.txt");

/// Lazily parsed NID → name index over [`NID_NAMES`].
fn table() -> &'static HashMap<u64, &'static str> {
    static TABLE: OnceLock<HashMap<u64, &'static str>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut map = HashMap::with_capacity(96_000);
        for line in NID_NAMES.lines() {
            let Some((nid, name)) = line.split_once(' ') else {
                continue;
            };
            if let Ok(nid) = u64::from_str_radix(nid, 16) {
                map.insert(nid, name);
            }
        }
        map
    })
}

/// The symbol name for `nid`, if a hash-verified one is known.
///
/// A `Some` is authoritative: the name provably hashes to `nid`. A `None` means
/// the dictionary did not contain it — the NID may still be a real, callable
/// import (PS5-only libraries such as `libSceAgc` are the usual gap, since the
/// dictionary is PS4-derived).
pub fn name_of(nid: u64) -> Option<&'static str> {
    table().get(&nid).copied()
}

/// Render `nid` for a human: the recovered name when known, else the encoded
/// NID string. Diagnostics should prefer this over printing a bare hash.
pub fn describe(nid: u64) -> String {
    match name_of(nid) {
        Some(name) => name.to_string(),
        None => super::nid::encode_nid(nid),
    }
}

/// Number of hash-verified names available.
pub fn len() -> usize {
    table().len()
}

#[cfg(test)]
mod tests {
    use super::super::nid::{encode_nid, nid_of};
    use super::*;

    /// THE safety property this module rests on: every name in the table is a
    /// verified SHA-1 preimage of its NID, recomputed by our own hasher. If a
    /// single entry failed, the dictionary would be untrusted input rather than
    /// proven fact — so this test checks all ~94k, not a sample.
    #[test]
    fn all_names_hash_to_their_nid() {
        let mut checked = 0usize;
        for (&nid, &name) in table() {
            assert_eq!(
                nid_of(name),
                nid,
                "table entry {name:?} does not hash to its NID {nid:#018x} — \
                 the table must contain only hash-proven names"
            );
            checked += 1;
        }
        assert!(
            checked > 90_000,
            "expected the full dictionary, only verified {checked} entries"
        );
    }

    /// The lookup that unblocked ASTRO.BOT: the title's fatal import was
    /// reported only as `nid 0xe285d87bd5e69344 (4oXYe9Xmk0Q)`.
    #[test]
    fn names_the_astro_bot_blocker() {
        let nid = 0xe285_d87b_d5e6_9344;
        assert_eq!(encode_nid(nid), "4oXYe9Xmk0Q", "pins the measured NID");
        assert_eq!(name_of(nid), Some("sceKernelGetGPI"));
        assert_eq!(describe(nid), "sceKernelGetGPI");
    }

    /// The 13 libkernel NIDs measured as missing from ALL THREE loadable retail
    /// titles. Six are bare POSIX/libc symbols, which is precisely why the
    /// prior `sceKernel<Verb><Noun>` brute force could never reach them.
    #[test]
    fn names_every_libkernel_nid_shared_by_all_three_titles() {
        for (nid, expected) in [
            (0x21a7_c8d8_fc5c_3e74u64, "scePthreadMutexTimedlock"),
            (0x361a_6ca7_1763_10a5, "_nanosleep"),
            (0x5400_dcdc_c350_ddc3, "signal"),
            (0x540c_ecc2_f4ce_0b32, "unlink"),
            (0x5a5c_8403_fb0b_0dfd, "sceKernelTruncate"),
            (0x73b6_674f_b575_07df, "rmdir"),
            (0x763c_713a_65ba_fdac, "__progname"),
            (0x7e02_2c43_5d31_6150, "sceKernelChmod"),
            (0x8479_5941_49e5_c523, "getrusage"),
            (0xd02a_bc8a_92ab_f67d, "sceKernelUtimes"),
            (0xde4e_a4c7_fcce_3924, "sceKernelMlock"),
            (0xe763_5c61_4f7e_944a, "sceKernelRename"),
            (
                0xfd84_d6fa_a5dc_dc24,
                "sceKernelInternalMemoryGetModuleSegmentInfo",
            ),
        ] {
            assert_eq!(name_of(nid), Some(expected), "nid {nid:#018x}");
            // Independently re-derive: the name must hash back to the NID.
            assert_eq!(nid_of(expected), nid, "{expected} must hash to its NID");
        }
    }

    /// An unknown NID must degrade to the encoded string, never panic or lie.
    #[test]
    fn unknown_nid_falls_back_to_the_encoded_form() {
        let nid = nid_of("xps5xDefinitelyNotARealSonySymbolName");
        assert_eq!(name_of(nid), None);
        assert_eq!(describe(nid), encode_nid(nid));
    }
}
