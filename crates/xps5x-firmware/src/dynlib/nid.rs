//! SCE NID (Name ID) hashing and base64 encoding.

use sha1::{Digest, Sha1};

/// Public, community-documented 16-byte suffix appended to a symbol name
/// before hashing. This is a salt, not a secret key.
const SCE_NID_SALT: [u8; 16] = [
    0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1, 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30,
];

/// Custom base64 alphabet used for NID strings.
const NID_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

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
/// Bytes are taken little-endian and packed MSB-first, 6 bits per character,
/// no padding (64 bits → 11 characters, the last carrying 4 significant bits).
pub fn encode_nid(nid: u64) -> String {
    let bytes = nid.to_le_bytes();
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
    Some(u64::from_le_bytes(bytes[0..8].try_into().ok()?))
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
}
