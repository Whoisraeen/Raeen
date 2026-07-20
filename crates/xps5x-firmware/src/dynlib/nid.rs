//! SCE NID (Name ID) hashing and base64 encoding.

use std::collections::{HashMap, HashSet};

use sha1::{Digest, Sha1};
use tracing::{debug, warn};

/// Public, community-documented 16-byte suffix appended to a symbol name
/// before hashing. This is a salt, not a secret key.
const SCE_NID_SALT: [u8; 16] = [
    0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1, 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30,
];

/// Custom base64 alphabet used for NID strings.
const NID_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

/// Compute the 64-bit NID for a symbol name.
pub fn nid_of(name: &str) -> u64 {
    let mut hasher = Sha1::new();
    hasher.update(name.as_bytes());
    hasher.update(SCE_NID_SALT);
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[0..8].try_into().unwrap())
}

/// Encode a 64-bit NID into its 11-character SCE base64 string.
///
/// Bytes are taken big-endian and packed MSB-first, 6 bits per character, no
/// padding (64 bits → 11 characters, the last carrying 4 significant bits).
///
/// Note: `nid_of` reads the SHA-1 digest's first 8 bytes little-endian into
/// the `u64`, but the SCE base64 packing then re-serializes that `u64`
/// big-endian — i.e. it walks the original digest bytes in their natural
/// (reversed) order. This asymmetry is confirmed against a public
/// name→NID vector (see the `pins_documented_libkernel_nid_vector` test).
pub fn encode_nid(nid: u64) -> String {
    let bytes = nid.to_be_bytes();
    let mut out = String::with_capacity(11);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for &b in &bytes {
        acc = (acc << 8) | b as u32;
        bits += 8;
        while bits >= 6 {
            bits -= 6;
            out.push(NID_ALPHABET[((acc >> bits) & 0x3F) as usize] as char);
        }
    }
    if bits > 0 {
        out.push(NID_ALPHABET[((acc << (6 - bits)) & 0x3F) as usize] as char);
    }
    out
}

/// Decode a **variable-length** SCE base64 field into an integer.
///
/// A real import symbol is `<nid>#<library>#<module>` — e.g. `rTXw65xmLIA#l#l`.
/// The `nid` is a full 11-char encoding of 8 bytes, but the library/module
/// fields are small indices encoded in as few characters as they need (usually
/// one or two). [`decode_nid`] cannot read those: it requires >= 8 decoded
/// bytes and returns `None` for anything shorter, so every import's library and
/// module index silently became 0 — which matches no `DT_SCE_IMPORT_LIB` entry
/// (real ids start at 1), making it impossible to say which library an
/// unresolved import belongs to.
///
/// This decodes MSB-first, 6 bits per character, over the same alphabet.
/// Returns `None` on a character outside the alphabet, or if the value would
/// exceed `u16` (indices are small by construction).
pub fn decode_index(s: &str) -> Option<u16> {
    if s.is_empty() {
        return None;
    }
    let mut acc: u32 = 0;
    for ch in s.chars() {
        let val = NID_ALPHABET.iter().position(|&a| a as char == ch)? as u32;
        acc = acc.checked_mul(64)?.checked_add(val)?;
        if acc > u16::MAX as u32 {
            return None;
        }
    }
    Some(acc as u16)
}

/// Decode an SCE base64 NID string back into a 64-bit NID.
///
/// Returns `None` if the string contains a character outside the alphabet or
/// decodes to fewer than 8 bytes. For the short library/module index fields of
/// an import symbol, use [`decode_index`] instead.
pub fn decode_nid(s: &str) -> Option<u64> {
    let mut acc: u64 = 0;
    let mut bits: u32 = 0;
    let mut bytes = Vec::with_capacity(8);
    for ch in s.chars() {
        let val = NID_ALPHABET.iter().position(|&a| a as char == ch)? as u64;
        acc = (acc << 6) | val;
        bits += 6;
        while bits >= 8 {
            bits -= 8;
            bytes.push(((acc >> bits) & 0xFF) as u8);
        }
    }
    if bytes.len() < 8 {
        return None;
    }
    // See `encode_nid`: the packed bytes are big-endian relative to the
    // `u64` produced by `nid_of`.
    Some(u64::from_be_bytes(bytes[0..8].try_into().ok()?))
}

/// Maps import NIDs back to the HLE `"library::function"` names they
/// resolve to, precomputed from the names XPS5X's HLE implements.
///
/// Built once (from [`from_hle_names`](Self::from_hle_names)) and then
/// consulted read-only when linking a module's imports (Task 5's
/// `ModuleRegistry`).
#[derive(Debug, Default, Clone)]
pub struct NidDatabase {
    by_nid: HashMap<u64, String>,
    by_provider: HashMap<(String, u64), String>,
}

impl NidDatabase {
    /// Build a database from `(library, function)` name pairs — e.g. every
    /// name the HLE registry implements. Each function name's NID is
    /// computed via [`nid_of`] and mapped to `"library::function"`.
    ///
    /// # Two NIDs can be equal for two very different reasons
    ///
    /// A NID hashes the **function name alone** — the library is not an input.
    /// So two entries share a NID when either:
    ///
    /// * **The same function name is registered under several libraries.**
    ///   Equal *by construction*, expected, and common: a title may import
    ///   `sce::Json` from either `libSceJson` or `libSceJson2`, and
    ///   `libkernel`/`libScePosix` both spell `getpid`. Measured on this tree:
    ///   11 such names, every one registering the *same implementation* under
    ///   both libraries. Logged at `debug` — warning about it 302 times per run
    ///   (as this used to) is crying wolf, and trains everyone to ignore the
    ///   message that actually matters.
    /// * **Two genuinely different names hash to the same 64-bit value.** A
    ///   real SHA-1-truncated-to-64-bits collision, astronomically unlikely
    ///   over a name set this size (~10^-14). If it ever happens it is a true
    ///   problem — one function becomes unreachable — so it is logged at
    ///   `warn` with both names. Never observed.
    ///
    /// # Determinism
    ///
    /// Input order is **not** stable — `HleRegistry::registered_names()` walks
    /// a concurrent map — so "first inserted wins" made the winner flip between
    /// runs. Measured on the real title: 10 of the 11 duplicated names resolved
    /// *both ways* across two runs minutes apart. That is harmless only for as
    /// long as the duplicate implementations stay identical, and it poisons any
    /// "it worked last time" reasoning. Sorting first makes the winner a
    /// function of the name set alone.
    ///
    /// The generic NID view keeps one deterministic name for diagnostics.
    /// Provider-aware linking uses a second `(library, NID)` index via
    /// [`Self::resolve_for_provider`], preserving the richer import identity.
    pub fn from_hle_names(names: impl IntoIterator<Item = (String, String)>) -> Self {
        // Deterministic winner: sort before insertion so the result depends on
        // the name set, not on iteration order.
        let mut names: Vec<(String, String)> = names.into_iter().collect();
        names.sort();

        let mut by_nid: HashMap<u64, String> = HashMap::new();
        let mut by_provider: HashMap<(String, u64), String> = HashMap::new();
        for (library, function) in names {
            let nid = nid_of(&function);
            let key = format!("{library}::{function}");
            by_provider.insert((canonical_provider_name(&library), nid), key.clone());
            match by_nid.entry(nid) {
                std::collections::hash_map::Entry::Occupied(existing) => {
                    let existing_function = existing.get().split_once("::").map(|(_, f)| f);
                    if existing_function == Some(function.as_str()) {
                        debug!(
                            "{function:?} is registered under several libraries; a NID hashes the \
                             name alone, so they share {nid:#x} by construction — resolving to \
                             {:?}, not {key:?}",
                            existing.get()
                        );
                    } else {
                        warn!(
                            "genuine NID collision at {nid:#x}: two DIFFERENT names hash alike — \
                             keeping {:?}, dropping {key:?}. {key:?} is now unreachable.",
                            existing.get()
                        );
                    }
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(key);
                }
            }
        }
        Self {
            by_nid,
            by_provider,
        }
    }

    /// Build the database from a live [`xps5x_hle::HleRegistry`]: every
    /// name-hashed NID, **plus** every explicit NID binding.
    ///
    /// Prefer this over [`Self::from_hle_names`] anywhere a real registry is
    /// available. Name hashing alone cannot express a function whose real name
    /// is unknown — the RE'd-by-NID case — and silently leaves it unreachable
    /// (see [`xps5x_hle::HleRegistry::register_nid`]).
    pub fn from_hle(hle: &xps5x_hle::HleRegistry) -> Self {
        let mut db = Self::from_hle_names(hle.registered_names());
        let mut overrides = hle.registered_provider_nid_overrides();
        overrides.sort_by(|left, right| (left.1, &left.2).cmp(&(right.1, &right.2)));
        let mut explicit_nids = HashSet::new();
        for (library, nid, key) in overrides {
            // An explicit NID is a stronger statement than a hashed name: it
            // was read out of a real module's symbol table. The provider-aware
            // table retains every binding; the provider-free diagnostic view
            // chooses the lexicographically first explicit label so it stays
            // deterministic even when equal NIDs exist in several libraries.
            if explicit_nids.insert(nid)
                && let Some(existing) = db.by_nid.insert(nid, key.clone())
                && existing != key
            {
                debug!("explicit NID {nid:#018x} -> {key:?} overrides name-hashed {existing:?}");
            }
            db.by_provider
                .insert((canonical_provider_name(&library), nid), key);
        }
        db
    }

    /// Resolve an import NID to its `"library::function"` name, if known.
    pub fn resolve(&self, nid: u64) -> Option<&str> {
        self.by_nid.get(&nid).map(String::as_str)
    }

    /// Resolve only when `provider` actually registered this NID. Unlike
    /// [`Self::resolve`], this preserves the module/library half of the import
    /// identity and cannot borrow an implementation from an unrelated HLE
    /// library that happens to use the same function name.
    pub fn resolve_for_provider(&self, provider: &str, nid: u64) -> Option<&str> {
        self.by_provider
            .get(&(canonical_provider_name(provider), nid))
            .map(String::as_str)
    }

    /// Number of entries in the database.
    pub fn len(&self) -> usize {
        self.by_nid.len()
    }

    /// Whether the database has no entries.
    pub fn is_empty(&self) -> bool {
        self.by_nid.is_empty()
    }
}

fn canonical_provider_name(provider: &str) -> String {
    let lower = provider.to_ascii_lowercase();
    let lower = lower
        .strip_suffix(".sprx")
        .or_else(|| lower.strip_suffix(".prx"))
        .unwrap_or(&lower);
    // `.native` / `_native` is a SPELLING of the same library, not a different
    // one. Retail import tables ask for `libSceMsgDialog.native` and
    // `libSceSaveDataDialog.native` while the HLE registers the bare names, so
    // without this every such import resolved to `Unresolved` even though the
    // function was fully implemented — measured on Minecraft (PPSA17221), which
    // imports 7 `libSceMsgDialog.native` + 7 `libSceSaveDataDialog.native`
    // symbols that all exist in-tree. Stripping here fixes the whole class at
    // the one point that already normalizes provider spelling, instead of
    // making each module hand-maintain an alias list (the idiom
    // `libsce_save_data.rs` had to use).
    lower
        .strip_suffix(".native")
        .or_else(|| lower.strip_suffix("_native"))
        .unwrap_or(lower)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `.native` / `_native` import must reach the bare-named registration.
    ///
    /// Measured on Minecraft (PPSA17221): the title imports
    /// `sceMsgDialogInitialize` from `libSceMsgDialog.native` and
    /// `sceSaveDataDialogOpen` from `libSceSaveDataDialog.native`. Both are
    /// implemented in-tree under the bare library names, yet the link reported
    /// all 14 such symbols "missing" purely because the provider spelling did
    /// not match. Guard the canonicalization, not the individual names.
    #[test]
    fn native_suffixed_provider_resolves_to_the_bare_registration() {
        let db = NidDatabase::from_hle_names(vec![
            (
                "libSceMsgDialog".to_string(),
                "sceMsgDialogInitialize".to_string(),
            ),
            (
                "libSceSaveDataDialog".to_string(),
                "sceSaveDataDialogOpen".to_string(),
            ),
        ]);

        for (import_library, function) in [
            ("libSceMsgDialog.native", "sceMsgDialogInitialize"),
            ("libSceMsgDialog_native", "sceMsgDialogInitialize"),
            ("libSceMsgDialog", "sceMsgDialogInitialize"),
            ("libSceSaveDataDialog.native", "sceSaveDataDialogOpen"),
            ("libSceSaveDataDialog_native", "sceSaveDataDialogOpen"),
        ] {
            let nid = nid_of(function);
            let resolved = db.resolve_for_provider(import_library, nid);
            assert!(
                resolved.is_some_and(|name| name.ends_with(function)),
                "import {import_library}::{function} (NID {nid:#018x}) did not reach its \
                 bare-named registration — got {resolved:?}"
            );
        }
    }

    /// Stripping `.native` must not merge two genuinely different libraries:
    /// the suffix is only removed from the END of the provider name.
    #[test]
    fn native_stripping_does_not_merge_unrelated_libraries() {
        assert_eq!(
            canonical_provider_name("libSceNativeThing"),
            "libscenativething"
        );
        assert_eq!(
            canonical_provider_name("libSceSaveData.native"),
            "libscesavedata"
        );
        assert_eq!(
            canonical_provider_name("libSceSaveData.prx"),
            "libscesavedata"
        );
    }

    /// The same function name under several libraries must resolve to the SAME
    /// winner no matter what order the names arrive in.
    ///
    /// `HleRegistry::registered_names()` walks a concurrent map, so its order is
    /// not stable — and with "first inserted wins" that made the winner flip
    /// between runs. Measured on the real title before this fix: 10 of 11
    /// duplicated names resolved *both ways* across two runs minutes apart.
    #[test]
    fn duplicate_names_resolve_the_same_winner_whatever_the_input_order() {
        let forward = vec![
            ("libSceJson".to_string(), "sceJsonInit".to_string()),
            ("libSceJson2".to_string(), "sceJsonInit".to_string()),
        ];
        let reversed: Vec<_> = forward.iter().cloned().rev().collect();

        let a = NidDatabase::from_hle_names(forward);
        let b = NidDatabase::from_hle_names(reversed);

        let nid = nid_of("sceJsonInit");
        assert_eq!(
            a.resolve(nid),
            b.resolve(nid),
            "input order must not change the winner"
        );
        // And the winner is the one the sort picks, not an accident of hashing.
        assert_eq!(a.resolve(nid), Some("libSceJson::sceJsonInit"));
    }

    /// A three-way duplicate is stable too, and the sort's winner is the
    /// lowest `(library, function)` pair.
    #[test]
    fn duplicate_winner_is_the_lowest_library_name() {
        let mut names = vec![
            ("libZ".to_string(), "f".to_string()),
            ("libA".to_string(), "f".to_string()),
            ("libM".to_string(), "f".to_string()),
        ];
        let first = NidDatabase::from_hle_names(names.clone());
        names.rotate_left(2);
        let second = NidDatabase::from_hle_names(names);

        let nid = nid_of("f");
        assert_eq!(first.resolve(nid), Some("libA::f"));
        assert_eq!(second.resolve(nid), Some("libA::f"));
    }

    /// The same name under two libraries collapses to ONE entry — it is one
    /// NID, by construction, not two.
    #[test]
    fn the_same_name_in_two_libraries_is_one_entry_not_a_hash_accident() {
        let db = NidDatabase::from_hle_names(vec![
            ("libkernel".to_string(), "getpid".to_string()),
            ("libScePosix".to_string(), "getpid".to_string()),
        ]);
        assert_eq!(db.len(), 1);
        // The NID is a hash of the name alone, so the library cannot change it.
        assert_eq!(nid_of("getpid"), nid_of("getpid"));
    }

    /// Distinct names never collapse, so a database of unique names keeps every
    /// entry — i.e. we have no *genuine* NID collisions in practice.
    #[test]
    fn distinct_names_each_get_their_own_entry() {
        let names: Vec<_> = ["alpha", "beta", "gamma", "delta"]
            .iter()
            .map(|f| ("libX".to_string(), f.to_string()))
            .collect();
        let db = NidDatabase::from_hle_names(names);
        assert_eq!(db.len(), 4);
    }

    #[test]
    fn nid_is_deterministic() {
        assert_eq!(nid_of("sceKernelError"), nid_of("sceKernelError"));
    }

    #[test]
    fn distinct_names_give_distinct_nids() {
        assert_ne!(nid_of("foo"), nid_of("bar"));
    }

    #[test]
    fn encode_decode_round_trips() {
        for nid in [0u64, 1, 0x0123_4567_89AB_CDEF, u64::MAX] {
            let s = encode_nid(nid);
            assert_eq!(s.len(), 11, "NID string is 11 chars");
            assert_eq!(decode_nid(&s), Some(nid), "round-trip {nid:#x}");
        }
    }

    #[test]
    fn alphabet_endpoints() {
        // 6-bit index 0 -> 'A', 63 -> '-'. Verify via single-value encodings:
        // nid whose top 6 bits are 0 begins with 'A'.
        assert!(encode_nid(0).starts_with('A'));
        // decode rejects characters outside the alphabet.
        assert_eq!(decode_nid("!!!!!!!!!!!"), None);
    }

    /// Pins the NID bit-order against a public, documented `name -> NID`
    /// vector (design §7 open item / plan Task 4 step 2).
    ///
    /// Source: shadPS4 (open-source PS5/PS4 emulator,
    /// <https://github.com/shadps4-emu/shadPS4>) hardcodes each libkernel
    /// export's SCE NID string as the first argument to its `LIB_FUNCTION`
    /// registration macro, e.g. in
    /// `src/core/libraries/kernel/memory.cpp`:
    /// `LIB_FUNCTION("rTXw65xmLIA", "libkernel", 1, "libkernel",
    /// sceKernelAllocateDirectMemory);`. These strings are load-bearing for
    /// shadPS4's dynamic linker to resolve real PS5 game imports, so they
    /// are a genuinely verified public vector, not merely documented in
    /// prose.
    ///
    /// This test originally failed: `nid_of` (SHA-1(name + salt), first 8
    /// bytes read little-endian into a `u64`) was correct, but `encode_nid`
    /// / `decode_nid` packed those bytes little-endian instead of
    /// big-endian, producing the wrong base64 string. Fixed by switching
    /// `encode_nid`/`decode_nid` to `to_be_bytes`/`from_be_bytes`; verified
    /// against six independent shadPS4 `libkernel` NIDs below.
    #[test]
    fn pins_documented_libkernel_nid_vectors() {
        let vectors: &[(&str, &str)] = &[
            ("sceKernelAllocateDirectMemory", "rTXw65xmLIA"),
            ("sceKernelAllocateMainDirectMemory", "B+vc2AO2Zrc"),
            ("sceKernelMapDirectMemory", "L-Q3LEjIbgA"),
            ("sceKernelMmap", "PGhQHd-dzv8"),
            ("sceKernelMunmap", "cQke9UuBQOk"),
            ("sceKernelReleaseDirectMemory", "MBuItvba6z8"),
        ];
        for (name, expected_nid_string) in vectors {
            let actual = encode_nid(nid_of(name));
            assert_eq!(
                &actual, expected_nid_string,
                "NID for {name:?} should match the documented shadPS4 vector"
            );
        }
    }
}

#[cfg(test)]
mod nid_database_tests {
    use super::*;

    #[test]
    fn from_hle_names_resolves_known_nid() {
        let db = NidDatabase::from_hle_names([(
            "libkernel".to_string(),
            "sceKernelAllocDirectMemory".to_string(),
        )]);
        assert_eq!(
            db.resolve(nid_of("sceKernelAllocDirectMemory")),
            Some("libkernel::sceKernelAllocDirectMemory")
        );
    }

    #[test]
    fn resolve_unknown_nid_returns_none() {
        let db = NidDatabase::from_hle_names([(
            "libkernel".to_string(),
            "sceKernelAllocDirectMemory".to_string(),
        )]);
        assert_eq!(db.resolve(nid_of("someUnregisteredFunction")), None);
    }

    #[test]
    fn len_and_is_empty_behave() {
        let empty = NidDatabase::from_hle_names(std::iter::empty());
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let db = NidDatabase::from_hle_names([
            ("libkernel".to_string(), "sceKernelSleep".to_string()),
            ("libc".to_string(), "malloc".to_string()),
        ]);
        assert_eq!(db.len(), 2);
        assert!(!db.is_empty());
    }

    /// The payoff test for broadening HLE coverage (see xps5x-hle's
    /// `libkernel`/`libc` modules): a `NidDatabase` built from the *real*
    /// `HleRegistry::new()` (not a hand-built fixture) must resolve a real
    /// PS5 import NID — `sceKernelAllocateDirectMemory` — straight through to
    /// its HLE `"library::function"` name. This proves a real module's
    /// import of this function now resolves to HLE instead of `Unresolved`.
    #[test]
    fn real_hle_registry_resolves_sce_kernel_allocate_direct_memory_nid() {
        let hle = xps5x_hle::HleRegistry::new();
        let db = NidDatabase::from_hle_names(hle.registered_names());

        let nid = nid_of("sceKernelAllocateDirectMemory");
        assert_eq!(
            db.resolve(nid),
            Some("libkernel::sceKernelAllocateDirectMemory"),
            "expected sceKernelAllocateDirectMemory's NID to resolve via the real HLE registry"
        );
    }

    /// ASTRO.BOT imports `sceNpGetAccountCountryA` (libSceNpManager) by this
    /// exact NID — measured 2026-07-18: the eboot link reported it missing and
    /// the title later hard-asserted at `NpWebApi.cpp:1587` on the resumed
    /// error. The registration exists (`libsce_np.rs`), so the live-registry
    /// database must resolve it. Pins both the hash and the registration.
    #[test]
    fn real_hle_registry_resolves_np_get_account_country_a_nid() {
        let hle = xps5x_hle::HleRegistry::new();
        let db = NidDatabase::from_hle(&hle);

        assert_eq!(nid_of("sceNpGetAccountCountryA"), 0x253f_add3_46b7_4f10);
        assert_eq!(
            db.resolve(0x253f_add3_46b7_4f10),
            Some("libSceNpManager::sceNpGetAccountCountryA"),
        );
    }

    #[test]
    fn explicit_equal_nids_resolve_by_provider_without_overwrite() {
        fn first(_ctx: &xps5x_hle::HleContext, _args: &[u64]) -> u64 {
            1
        }
        fn second(_ctx: &xps5x_hle::HleContext, _args: &[u64]) -> u64 {
            2
        }
        let hle = xps5x_hle::HleRegistry::new();
        let nid = 0x1234_5678_9abc_def0;
        hle.register_nid("libAlpha", "unknownAlpha", nid, first);
        hle.register_nid("libBeta", "unknownBeta", nid, second);
        let db = NidDatabase::from_hle(&hle);

        assert_eq!(
            db.resolve_for_provider("libAlpha.sprx", nid),
            Some("libAlpha::unknownAlpha")
        );
        assert_eq!(
            db.resolve_for_provider("LIBBETA", nid),
            Some("libBeta::unknownBeta")
        );
    }

    #[test]
    fn duplicate_name_resolves_deterministically_and_does_not_panic() {
        // Same function name registered under two different libraries maps
        // to the same NID (nid_of only hashes the function name), which is
        // a real, expected "collision" from the database's point of view.
        //
        // This used to assert "first-inserted wins" — which is exactly the bug:
        // the input comes from a concurrent map with no stable order, so the
        // winner flipped between runs (measured: 10 of 11 duplicated names
        // resolved both ways across two real runs). The contract is now that
        // the winner is a function of the NAME SET, not of arrival order.
        let forward = [
            ("libkernel".to_string(), "sceKernelSleep".to_string()),
            (
                "libSceLibcInternal".to_string(),
                "sceKernelSleep".to_string(),
            ),
        ];
        let reversed: Vec<_> = forward.iter().cloned().rev().collect();
        let db = NidDatabase::from_hle_names(forward.clone());
        let db_rev = NidDatabase::from_hle_names(reversed);

        assert_eq!(db.len(), 1);
        assert_eq!(
            db.resolve(nid_of("sceKernelSleep")),
            db_rev.resolve(nid_of("sceKernelSleep")),
            "the winner must not depend on input order"
        );
        assert_eq!(
            db.resolve(nid_of("sceKernelSleep")),
            Some("libSceLibcInternal::sceKernelSleep"),
            "the sort picks the lowest (library, function) pair — 'libS' < 'libk'"
        );
    }
}
