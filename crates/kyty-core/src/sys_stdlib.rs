//! Port of Kyty's `Sys::sys_strtoX` family
//! (`reference/kyty/source/include/Kyty/Sys/SysStdlib.h`, which is just
//! `#include "Kyty/Sys/Linux/SysLinuxStdlib.h"` +
//! `#include "Kyty/Sys/Windows/SysWindowsStdlib.h"`; Raeen targets Windows
//! only, so this ports `include/Kyty/Sys/Windows/SysWindowsStdlib.h` — six
//! `inline` one-liners forwarding to the C runtime: `sys_strtof`/`sys_strtod`
//! -> `strtof`/`strtod`, `sys_strtoi32`/`sys_strtoui32` -> `strtol`/`strtoul`,
//! `sys_strtoi64`/`sys_strtoui64` -> MSVC's `_strtoi64`/`_strtoui64` (the
//! 64-bit `strtoll`/`strtoull` equivalents on Windows)).
//!
//! # Std mapping
//!
//! These functions are pure C-runtime string-to-number parsers with no
//! Windows-specific behavior (`_strtoi64`/`_strtoui64` are just MSVC's names
//! for `strtoll`/`strtoull`), so — per the port conventions' preference for
//! Rust std where it fully covers the behavior — they are reimplemented here
//! as safe, dependency-free prefix parsers replicating the C `strtoX`
//! contract, rather than calling out to libc via FFI. Rust's `str::parse`
//! requires the *whole* string to be a valid number and has no "parse a
//! leading numeric prefix, ignore trailing garbage" mode, so a small manual
//! parser is the faithful equivalent (this mirrors the same tradeoff already
//! made for [`crate::string8::String8::to_int32`] and friends, which port the
//! same C semantics at the `String8` level; this module ports the lower-level
//! `sys_strtoX` primitives themselves).
//!
//! The original C signature is `T sys_strtoX(const char *nptr, char
//! **endptr, [int base])`: `nptr` is the NUL-terminated input, and `endptr`
//! is an optional out-param receiving a pointer just past the last character
//! consumed. Rust has no raw C strings in safe code, so `nptr`/`endptr` are
//! ported as a `&str` input plus a `usize` "characters consumed" return
//! value standing in for `endptr - nptr` (`0` exactly when the C function
//! would leave `*endptr == nptr`, i.e. "no conversion could be performed").
//! Callers that want the C `endptr` pointer itself can reconstruct it as
//! `&input[consumed..]`.
//!
//! Overflow behavior matches the C runtime exactly: `sys_strtoi32`/
//! `sys_strtoi64` clamp to `[i32::MIN, i32::MAX]`/`[i64::MIN, i64::MAX]`
//! (like `strtol`/`_strtoi64` clamping to `LONG_MIN`/`LONG_MAX` and setting
//! `errno = ERANGE`, minus the `errno` signal, which Kyty's callers never
//! checked), and `sys_strtoui32`/`sys_strtoui64` wrap modulo 2^32/2^64 on a
//! negative subject sequence (like `strtoul`/`_strtoui64`, e.g.
//! `strtoul("-1", ...)` == `UINT_MAX`).
//!
//! Recognizing `inf`/`nan` literals (which `strtod`/`strtof` do) is not
//! implemented, matching the same documented divergence already accepted for
//! [`crate::string8::String8::to_double`].
//!
//! Gated to `#[cfg(windows)]`: Raeen's only target is Windows, and this file
//! is specifically the port of the Windows-side header (the Linux side,
//! `SysLinuxStdlib.h`, is out of scope).

#![cfg(windows)]

/// Parses a `strtol`/`strtoul`-style leading numeric prefix: optional ASCII
/// whitespace, optional sign, then digits in `base` (or auto-detected when
/// `base == 0`, matching C's `strtol(..., 0)`: `0x`/`0X` prefix -> hex,
/// leading `0` -> octal, else decimal). Returns `(negative, magnitude,
/// consumed)`; `magnitude` saturates at `u128::MAX` rather than wrapping so
/// callers can clamp/truncate to their target width afterwards. `consumed`
/// is `0` (matching `*endptr == nptr`) if no digits were found.
fn parse_int_prefix(s: &str, base: i32) -> (bool, u128, usize) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    while i < len && bytes[i].is_ascii_whitespace() {
        i += 1;
    }

    let mut negative = false;
    if i < len && (bytes[i] == b'+' || bytes[i] == b'-') {
        negative = bytes[i] == b'-';
        i += 1;
    }

    let base: u32 = if base == 16 || base == 0 {
        if i + 1 < len && bytes[i] == b'0' && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
            i += 2;
            16
        } else if base == 0 {
            if i < len && bytes[i] == b'0' { 8 } else { 10 }
        } else {
            16
        }
    } else {
        base.clamp(2, 36) as u32
    };

    let mut magnitude: u128 = 0;
    let mut any_digits = false;
    let mut end = i;
    while i < len {
        let digit = match bytes[i] {
            c @ b'0'..=b'9' => u32::from(c - b'0'),
            c @ b'a'..=b'z' => u32::from(c - b'a') + 10,
            c @ b'A'..=b'Z' => u32::from(c - b'A') + 10,
            _ => break,
        };
        if digit >= base {
            break;
        }
        any_digits = true;
        magnitude = magnitude
            .saturating_mul(u128::from(base))
            .saturating_add(u128::from(digit));
        i += 1;
        end = i;
    }

    if !any_digits {
        return (false, 0, 0);
    }
    (negative, magnitude, end)
}

/// Parses a `strtod`/`strtof`-style leading floating-point prefix (optional
/// whitespace, optional sign, digits, optional `.digits`, optional
/// `[eE][+-]digits`). Returns `(value, consumed)`; `(0.0, 0)` if no valid
/// numeric prefix is present (matching `*endptr == nptr`). `inf`/`nan`
/// literals are not recognized (see module doc comment).
fn parse_float_prefix(s: &str) -> (f64, usize) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    while i < len && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;

    if i < len && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }

    let mut has_digits = false;
    while i < len && bytes[i].is_ascii_digit() {
        i += 1;
        has_digits = true;
    }
    if i < len && bytes[i] == b'.' {
        i += 1;
        while i < len && bytes[i].is_ascii_digit() {
            i += 1;
            has_digits = true;
        }
    }

    if !has_digits {
        return (0.0, 0);
    }

    let mut end = i;
    if i < len && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < len && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        let exp_digits_start = j;
        while j < len && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_digits_start {
            end = j;
        }
    }

    let value = s[start..end].parse::<f64>().unwrap_or(0.0);
    (value, end)
}

/// `sys_strtof(const char *nptr, char **endptr)`: ports Windows'
/// `strtof(nptr, endptr)`. Returns `(value, consumed)` in place of the
/// `endptr` out-param (see module doc comment).
pub fn sys_strtof(nptr: &str) -> (f32, usize) {
    let (value, consumed) = parse_float_prefix(nptr);
    (value as f32, consumed)
}

/// `sys_strtod(const char *nptr, char **endptr)`: ports Windows'
/// `strtod(nptr, endptr)`.
pub fn sys_strtod(nptr: &str) -> (f64, usize) {
    parse_float_prefix(nptr)
}

/// `sys_strtoi32(const char *nptr, char **endptr, int base)`: ports
/// Windows' `strtol(nptr, endptr, base)`, clamped to `int32_t` range.
pub fn sys_strtoi32(nptr: &str, base: i32) -> (i32, usize) {
    let (negative, magnitude, consumed) = parse_int_prefix(nptr, base);
    let signed: i128 = if negative {
        -(magnitude as i128)
    } else {
        magnitude as i128
    };
    (
        signed.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32,
        consumed,
    )
}

/// `sys_strtoui32(const char *nptr, char **endptr, int base)`: ports
/// Windows' `strtoul(nptr, endptr, base)`, wrapping modulo 2^32 on a
/// negative subject sequence (matching C's `strtoul` behavior).
pub fn sys_strtoui32(nptr: &str, base: i32) -> (u32, usize) {
    let (negative, magnitude, consumed) = parse_int_prefix(nptr, base);
    let val = magnitude.min(u128::from(u32::MAX)) as u32;
    (if negative { val.wrapping_neg() } else { val }, consumed)
}

/// `sys_strtoi64(const char *nptr, char **endptr, int base)`: ports
/// Windows' `_strtoi64(nptr, endptr, base)`, clamped to `int64_t` range.
pub fn sys_strtoi64(nptr: &str, base: i32) -> (i64, usize) {
    let (negative, magnitude, consumed) = parse_int_prefix(nptr, base);
    let signed: i128 = if negative {
        -(magnitude as i128)
    } else {
        magnitude as i128
    };
    (
        signed.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64,
        consumed,
    )
}

/// `sys_strtoui64(const char *nptr, char **endptr, int base)`: ports
/// Windows' `_strtoui64(nptr, endptr, base)`, wrapping modulo 2^64 on a
/// negative subject sequence (matching C's `_strtoui64` behavior).
pub fn sys_strtoui64(nptr: &str, base: i32) -> (u64, usize) {
    let (negative, magnitude, consumed) = parse_int_prefix(nptr, base);
    let val = magnitude.min(u128::from(u64::MAX)) as u64;
    (if negative { val.wrapping_neg() } else { val }, consumed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strtof_basic() {
        let (v, consumed) = sys_strtof("3.5xyz");
        assert!((v - 3.5_f32).abs() < 1e-6);
        assert_eq!(consumed, 3);
    }

    #[test]
    fn strtof_no_conversion() {
        let (v, consumed) = sys_strtof("   notanumber");
        assert_eq!(v, 0.0);
        assert_eq!(consumed, 0);
    }

    #[test]
    fn strtod_basic_and_exponent() {
        let (v, consumed) = sys_strtod("  -2.5e2trailing");
        assert!((v - (-250.0)).abs() < 1e-9);
        assert_eq!(consumed, "  -2.5e2".len());
    }

    #[test]
    fn strtod_no_digits() {
        assert_eq!(sys_strtod(""), (0.0, 0));
        assert_eq!(sys_strtod("   "), (0.0, 0));
        assert_eq!(sys_strtod("e5"), (0.0, 0));
    }

    #[test]
    fn strtoi32_decimal_and_sign() {
        assert_eq!(sys_strtoi32("42", 10), (42, 2));
        let (v, consumed) = sys_strtoi32("  -7rest", 10);
        assert_eq!(v, -7);
        assert_eq!(consumed, "  -7".len());
    }

    #[test]
    fn strtoi32_hex_and_base_zero_autodetect() {
        assert_eq!(sys_strtoi32("ff", 16), (255, 2));
        assert_eq!(sys_strtoi32("0x1A", 0), (26, 4));
        assert_eq!(sys_strtoi32("017", 0), (15, 3)); // octal auto-detect
    }

    #[test]
    fn strtoi32_overflow_clamps() {
        assert_eq!(sys_strtoi32("99999999999", 10).0, i32::MAX);
        assert_eq!(sys_strtoi32("-99999999999", 10).0, i32::MIN);
    }

    #[test]
    fn strtoi32_no_digits_reports_zero_consumed() {
        assert_eq!(sys_strtoi32("notanumber", 10), (0, 0));
    }

    #[test]
    fn strtoui32_negative_wraps_like_c_strtoul() {
        let (v, consumed) = sys_strtoui32("-1", 10);
        assert_eq!(v, u32::MAX);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn strtoi64_and_strtoui64_basic() {
        assert_eq!(sys_strtoi64("123456789012", 10), (123_456_789_012, 12));
        assert_eq!(sys_strtoi64("-5", 10).0, -5);
        assert_eq!(sys_strtoui64("100", 10), (100, 3));
        assert_eq!(sys_strtoui64("-1", 10).0, u64::MAX);
    }

    #[test]
    fn strtoi64_overflow_clamps() {
        assert_eq!(sys_strtoi64("99999999999999999999999", 10).0, i64::MAX);
        assert_eq!(sys_strtoi64("-99999999999999999999999", 10).0, i64::MIN);
    }

    #[test]
    fn trailing_garbage_ignored_matching_c_semantics() {
        assert_eq!(sys_strtoi32("123abc", 10), (123, 3));
        let (v, consumed) = sys_strtof("1.5abc");
        assert!((v - 1.5_f32).abs() < 1e-6);
        assert_eq!(consumed, 3);
    }
}
