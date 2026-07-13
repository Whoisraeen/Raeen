//! Port of Kyty's `include/Kyty/Sys/SysSwapByteOrder.h`
//! (`Kyty::SwapByteOrder16/32/64`, the generic `Kyty::SwapByteOrder<T>`
//! template, and `Kyty::NoSwapByteOrder<T>`).
//!
//! The original header is header-only and picks an intrinsic per compiler
//! (`_byteswap_ushort`/`_byteswap_ulong`/`_byteswap_uint64` on MSVC,
//! `__builtin_bswap32`/`__builtin_bswap64` on Clang, manual shifts
//! otherwise). Rust's integer primitives already expose an intrinsic-backed
//! [`swap_bytes`](u32::swap_bytes) on every target, so the fixed-width free
//! functions here are thin wrappers over that — no `unsafe`, no
//! platform-specific dispatch needed.
//!
//! Kyty's generic `template <typename T> void SwapByteOrder(T& x)` dispatches
//! at compile time on `sizeof(x)` (2, 4, or 8 bytes) and on `std::is_signed_v`.
//! Rust has no `sizeof`-based generic dispatch, so that template is ported as
//! a small [`SwapByteOrder`] trait implemented for the six built-in integer
//! types Kyty's callers actually instantiate it with (`u16`/`i16`, `u32`/
//! `i32`, `u64`/`i64`); the free function [`swap_byte_order`] mirrors the call
//! site `Kyty::SwapByteOrder(x)`. `Kyty::NoSwapByteOrder<T>` (an intentional
//! no-op, used where a format is already the desired endianness) is ported as
//! [`no_swap_byte_order`].

/// Swap the byte order of a 16-bit value.
///
/// Kyty: `SwapByteOrder16`. Maps directly to [`u16::swap_bytes`].
#[must_use]
pub fn swap_byte_order_16(value: u16) -> u16 {
    value.swap_bytes()
}

/// Swap the byte order of a 32-bit value.
///
/// Kyty: `SwapByteOrder32`. Maps directly to [`u32::swap_bytes`].
#[must_use]
pub fn swap_byte_order_32(value: u32) -> u32 {
    value.swap_bytes()
}

/// Swap the byte order of a 64-bit value.
///
/// Kyty: `SwapByteOrder64` (falls back to composing two 32-bit swaps in the
/// portable C++ path). Maps directly to [`u64::swap_bytes`].
#[must_use]
pub fn swap_byte_order_64(value: u64) -> u64 {
    value.swap_bytes()
}

/// Port of Kyty's generic `template <typename T> void SwapByteOrder(T& x)`.
///
/// Implemented for the integer widths the C++ template's `sizeof(x) == 2 / 4
/// / 8` branches actually swap, signed and unsigned alike (the C++ template
/// leaves any other width untouched — a `T` outside these widths simply has
/// no matching `impl` here, which is the static-dispatch equivalent).
pub trait SwapByteOrder {
    /// Swap this value's byte order in place.
    fn swap_byte_order(&mut self);
}

macro_rules! impl_swap_byte_order {
    ($($t:ty),+ $(,)?) => {
        $(
            impl SwapByteOrder for $t {
                fn swap_byte_order(&mut self) {
                    *self = self.swap_bytes();
                }
            }
        )+
    };
}

impl_swap_byte_order!(u16, i16, u32, i32, u64, i64);

/// Free-function form of [`SwapByteOrder::swap_byte_order`], mirroring the
/// call site `Kyty::SwapByteOrder(x)`.
pub fn swap_byte_order<T: SwapByteOrder>(x: &mut T) {
    x.swap_byte_order();
}

/// Port of Kyty's `template <typename T> void NoSwapByteOrder(T& x)`: an
/// intentional no-op, used at call sites where the data is already in the
/// desired endianness and no conversion should happen.
pub fn no_swap_byte_order<T>(_x: &mut T) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_16_reverses_bytes() {
        assert_eq!(swap_byte_order_16(0x1234), 0x3412);
        assert_eq!(swap_byte_order_16(0x0000), 0x0000);
        assert_eq!(swap_byte_order_16(0xFFFF), 0xFFFF);
        assert_eq!(swap_byte_order_16(0x00FF), 0xFF00);
    }

    #[test]
    fn swap_32_reverses_bytes() {
        assert_eq!(swap_byte_order_32(0x1234_5678), 0x7856_3412);
        assert_eq!(swap_byte_order_32(0x0000_0001), 0x0100_0000);
        assert_eq!(swap_byte_order_32(0xFFFF_FFFF), 0xFFFF_FFFF);
    }

    #[test]
    fn swap_64_reverses_bytes() {
        assert_eq!(
            swap_byte_order_64(0x0123_4567_89AB_CDEFu64),
            0xEFCD_AB89_6745_2301u64
        );
        assert_eq!(swap_byte_order_64(0), 0);
        assert_eq!(swap_byte_order_64(u64::MAX), u64::MAX);
    }

    #[test]
    fn swap_is_involutive() {
        // Swapping twice returns the original value, for every width.
        assert_eq!(swap_byte_order_16(swap_byte_order_16(0xBEEF)), 0xBEEF);
        assert_eq!(
            swap_byte_order_32(swap_byte_order_32(0xDEAD_BEEF)),
            0xDEAD_BEEF
        );
        assert_eq!(
            swap_byte_order_64(swap_byte_order_64(0x1122_3344_5566_7788)),
            0x1122_3344_5566_7788
        );
    }

    #[test]
    fn generic_trait_swaps_unsigned() {
        let mut v16: u16 = 0x1234;
        swap_byte_order(&mut v16);
        assert_eq!(v16, 0x3412);

        let mut v32: u32 = 0x1234_5678;
        swap_byte_order(&mut v32);
        assert_eq!(v32, 0x7856_3412);

        let mut v64: u64 = 0x0123_4567_89AB_CDEF;
        swap_byte_order(&mut v64);
        assert_eq!(v64, 0xEFCD_AB89_6745_2301);
    }

    #[test]
    fn generic_trait_swaps_signed_matching_bit_pattern_of_unsigned() {
        // Kyty's template reinterprets the signed value as unsigned, swaps,
        // then casts back — i.e. the bit pattern swap matches the unsigned
        // swap of the same width.
        let mut u: u16 = 0x1234;
        swap_byte_order(&mut u);

        let mut s: i16 = 0x1234;
        swap_byte_order(&mut s);

        assert_eq!(s as u16, u);
    }

    #[test]
    fn generic_trait_negative_signed_round_trips() {
        let original: i32 = -1; // 0xFFFF_FFFF, byte-swap of all-ones is itself
        let mut x = original;
        swap_byte_order(&mut x);
        assert_eq!(x, -1);

        let original2: i64 = i64::MIN; // 0x8000_0000_0000_0000
        let mut y = original2;
        swap_byte_order(&mut y);
        swap_byte_order(&mut y);
        assert_eq!(y, original2);
    }

    #[test]
    fn no_swap_leaves_value_unchanged() {
        let mut v: u32 = 0xDEAD_BEEF;
        no_swap_byte_order(&mut v);
        assert_eq!(v, 0xDEAD_BEEF);

        let mut s = String::from("unchanged");
        no_swap_byte_order(&mut s);
        assert_eq!(s, "unchanged");
    }
}
