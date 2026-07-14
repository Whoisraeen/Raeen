//! SCE NID (Name ID) hashing and base64 encoding.

use std::collections::HashMap;

use sha1::{Digest, Sha1};
use tracing::warn;

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

/// Decode an SCE base64 NID string back into a 64-bit NID.
///
/// Returns `None` if the string contains a character outside the alphabet or
/// decodes to fewer than 8 bytes.
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
}

impl NidDatabase {
    /// Build a database from `(library, function)` name pairs — e.g. every
    /// name the HLE registry implements. Each function name's NID is
    /// computed via [`nid_of`] and mapped to `"library::function"`.
    ///
    /// NID collisions are possible-but-astronomically-unlikely (SHA-1
    /// truncated to 64 bits over a modest name set). On a collision the
    /// first-inserted mapping wins and the collision is logged via `warn!`
    /// — this never panics.
    pub fn from_hle_names(names: impl IntoIterator<Item = (String, String)>) -> Self {
        let mut by_nid = HashMap::new();
        for (library, function) in names {
            let nid = nid_of(&function);
            let key = format!("{library}::{function}");
            match by_nid.entry(nid) {
                std::collections::hash_map::Entry::Occupied(existing) => {
                    warn!(
                        "NID collision for {:#x}: keeping {:?}, dropping {:?}",
                        nid,
                        existing.get(),
                        key
                    );
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(key);
                }
            }
        }
        Self { by_nid }
    }

    /// Resolve an import NID to its `"library::function"` name, if known.
    pub fn resolve(&self, nid: u64) -> Option<&str> {
        self.by_nid.get(&nid).map(String::as_str)
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn collision_keeps_first_and_does_not_panic() {
        // Same function name registered under two different libraries maps
        // to the same NID (nid_of only hashes the function name), which is
        // a real, expected "collision" from the database's point of view.
        let db = NidDatabase::from_hle_names([
            ("libkernel".to_string(), "sceKernelSleep".to_string()),
            (
                "libSceLibcInternal".to_string(),
                "sceKernelSleep".to_string(),
            ),
        ]);
        assert_eq!(db.len(), 1);
        assert_eq!(
            db.resolve(nid_of("sceKernelSleep")),
            Some("libkernel::sceKernelSleep"),
            "first-inserted mapping wins on collision"
        );
    }
}
