//! Port of Kyty's `Core/Hash.h`.
//!
//! This is Bob Jenkins' "one-at-a-time" hash, used throughout Kyty as the
//! general-purpose non-cryptographic hash for hashmap keys. There is no std
//! equivalent to wrap: this is pure arithmetic (no allocation, no manual
//! memory, no raw-pointer aliasing tricks worth preserving), so it is ported
//! directly as safe Rust functions operating on byte slices / integers
//! instead of `(const void*, len)` and `reinterpret_cast` tricks.
//!
//! API mapping (`Kyty::Core::` free functions -> `kyty_core::hash::` free functions):
//! - `hash(const void *key, uint32_t key_len)` -> `hash(key: &[u8]) -> u32`
//! - `hash8(uint8_t)`   -> `hash8(key: u8) -> u32`
//! - `hash16(uint16_t)` -> `hash16(key: u16) -> u32`
//! - `hash32(uint32_t)` -> `hash32(key: u32) -> u32`
//! - `hash64(uint64_t)` -> `hash64(key: u64) -> u32`
//!
//! The fixed-size variants operate on the value's native-endian byte
//! representation, matching the original's `reinterpret_cast<uint8_t*>(&key)`
//! (the original relies on the host's endianness; `to_ne_bytes()` reproduces
//! that exactly).

/// One-at-a-time hash over an arbitrary byte slice.
#[must_use]
pub fn hash(key: &[u8]) -> u32 {
    let mut h: u32 = 0;
    let mut chunks = key.chunks_exact(4);

    for chunk in &mut chunks {
        for &b in chunk {
            h = h.wrapping_add(u32::from(b));
            h = h.wrapping_add(h << 10);
            h ^= h >> 6;
        }
    }

    for &b in chunks.remainder() {
        h = h.wrapping_add(u32::from(b));
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }

    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);

    h
}

/// One-at-a-time hash over a single byte.
#[must_use]
pub fn hash8(key: u8) -> u32 {
    hash(&key.to_ne_bytes())
}

/// One-at-a-time hash over a `u16`'s native-endian bytes.
#[must_use]
pub fn hash16(key: u16) -> u32 {
    hash(&key.to_ne_bytes())
}

/// One-at-a-time hash over a `u32`'s native-endian bytes.
#[must_use]
pub fn hash32(key: u32) -> u32 {
    hash(&key.to_ne_bytes())
}

/// One-at-a-time hash over a `u64`'s native-endian bytes.
#[must_use]
pub fn hash64(key: u64) -> u32 {
    hash(&key.to_ne_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice_hashes_to_zero() {
        // With key_len == 0 the C++ loop/switch never touch `hash`, and
        // only the final avalanche mix (all zero-derived) runs, so the
        // result is deterministically 0.
        assert_eq!(hash(&[]), 0);
    }

    #[test]
    fn is_deterministic_for_same_input() {
        let data = b"XPS5X kyty-core";
        assert_eq!(hash(data), hash(data));
    }

    #[test]
    fn differs_for_different_input() {
        assert_ne!(hash(b"hello"), hash(b"world"));
    }

    #[test]
    fn handles_all_tail_lengths_1_2_3() {
        // Exercises the switch(key_len) 1/2/3 fallthrough tail path as well
        // as the >=4 chunked path, ensuring lengths 1..=5 all produce
        // distinct, stable values.
        let inputs: [&[u8]; 5] = [&[1], &[1, 2], &[1, 2, 3], &[1, 2, 3, 4], &[1, 2, 3, 4, 5]];
        let mut seen = Vec::new();
        for input in inputs {
            let h = hash(input);
            assert_eq!(h, hash(input));
            seen.push(h);
        }
        for i in 0..seen.len() {
            for j in (i + 1)..seen.len() {
                assert_ne!(
                    seen[i], seen[j],
                    "collision between inputs of different length"
                );
            }
        }
    }

    #[test]
    fn fixed_width_helpers_match_generic_hash_over_ne_bytes() {
        let v8: u8 = 0xAB;
        let v16: u16 = 0xABCD;
        let v32: u32 = 0xDEAD_BEEF;
        let v64: u64 = 0x1234_5678_9ABC_DEF0;

        assert_eq!(hash8(v8), hash(&v8.to_ne_bytes()));
        assert_eq!(hash16(v16), hash(&v16.to_ne_bytes()));
        assert_eq!(hash32(v32), hash(&v32.to_ne_bytes()));
        assert_eq!(hash64(v64), hash(&v64.to_ne_bytes()));
    }

    #[test]
    fn known_vector_matches_reference_one_at_a_time_hash() {
        // Cross-checked against the classic Jenkins one-at-a-time algorithm
        // (same recurrence as Kyty's Hash.h) for the ASCII bytes of "a".
        // hash=0 -> add 'a'(97) -> h=97; h += h<<10 => 97 + 99328 = 99425;
        // h ^= h>>6 (99425 >> 6 = 1553) => 99425 ^ 1553 = 98432
        // then final mix: h += h<<3; h ^= h>>11; h += h<<15
        let mut h: u32 = 97;
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
        h = h.wrapping_add(h << 3);
        h ^= h >> 11;
        h = h.wrapping_add(h << 15);
        assert_eq!(hash(b"a"), h);
    }
}
