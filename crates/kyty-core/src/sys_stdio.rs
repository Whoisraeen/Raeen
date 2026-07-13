//! Port of Kyty's `include/Kyty/Sys/Windows/SysWindowsStdio.h` (the Windows
//! implementation of the cross-platform `include/Kyty/Sys/SysStdio.h`
//! interface; compare `include/Kyty/Sys/Linux/SysLinuxStdio.h`, whose
//! `vsnprintf`-based logic is the same shape).
//!
//! ```cpp
//! inline uint32_t sys_vscprintf(const char *format, va_list argptr)
//! {
//!     int len = _vscprintf(format, argptr);
//!     return len < 0 ? 0 : len;
//! }
//!
//! inline uint32_t sys_vsnprintf(char *dest, size_t count, const char *format, va_list args)
//! {
//!     int len = _vsnprintf_s(dest, count+1, count, format, args);
//!     return len < 0 ? 0 : len;
//! }
//! ```
//!
//! These two functions exist purely to support `Core::String::Printf` /
//! `Core::String8::Printf`: `sys_vscprintf` measures how many bytes a C
//! `printf`-style `format` + `va_list` would produce (so the caller can size a
//! buffer), and `sys_vsnprintf` renders that same format into a caller-owned
//! buffer of at most `count` bytes, NUL-terminating it — mirroring MSVC's
//! `_vscprintf`/`_vsnprintf_s` (`_vsnprintf_s` is called with a literal
//! `count`, not `_TRUNCATE`, so — given a buffer of `count + 1` bytes as every
//! call site here provides — it always succeeds and returns the number of
//! bytes written, never truncating unexpectedly).
//!
//! Rust has no safe equivalent of a C format string parsed against a
//! `va_list` (this crate's `string.rs`/`string8.rs` already document that
//! `Printf`/`FromPrintf` are intentionally *not* ported for exactly that
//! reason — callers use `format!` instead), so this module keeps Kyty's
//! *buffer-sizing-and-truncated-copy* API and semantics, but operating on an
//! already-rendered Rust `&str` (the output of the caller's own `format!`)
//! rather than a C format string + `va_list`:
//!
//! - [`sys_vscprintf`] maps to the rendered string's byte length ([`str::len`]),
//!   matching `_vscprintf`'s "how many bytes would this produce" contract.
//! - [`sys_vsnprintf`] copies at most `count` bytes of the rendered string
//!   into `dest` and NUL-terminates it, matching `_vsnprintf_s`'s
//!   truncated-copy contract, without needing `unsafe` or a `va_list`.
//!
//! Windows-specific (see `SysWindowsStdio.h`); gated with `#[cfg(windows)]`
//! so the crate still builds on non-Windows targets.

/// Kyty: `sys_vscprintf(const char *format, va_list argptr)`.
///
/// C++ asks the CRT how many bytes `vsnprintf`-style formatting of `format`
/// against `argptr` would produce (excluding the NUL terminator), clamping a
/// negative CRT error result to `0`. Since Rust has no `va_list`/C format
/// string, this operates on the string the caller already rendered with
/// `format!` — its byte length is exactly that "would-be-written" count.
#[cfg(windows)]
#[must_use]
pub fn sys_vscprintf(formatted: &str) -> u32 {
    u32::try_from(formatted.len()).unwrap_or(u32::MAX)
}

/// Kyty: `sys_vsnprintf(char *dest, size_t count, const char *format, va_list args)`.
///
/// C++ renders `format`/`args` into `dest` (a buffer of `count + 1` bytes, per
/// every call site in Kyty), writing at most `count` bytes plus a NUL
/// terminator, and returns the number of bytes written (excluding the NUL),
/// clamping a negative CRT error result to `0`.
///
/// Ported here as a truncating byte-copy of an already-rendered `&str` into
/// `dest`: copies `min(count, formatted.len(), dest.len().saturating_sub(1))`
/// bytes, writes a `0` terminator immediately after them, and returns the
/// number of bytes copied. `dest` should be at least `count + 1` bytes long
/// (mirroring the C buffer sizing), but this never panics or writes out of
/// bounds even if `dest` is smaller.
#[cfg(windows)]
pub fn sys_vsnprintf(dest: &mut [u8], count: usize, formatted: &str) -> u32 {
    if dest.is_empty() {
        return 0;
    }

    let capacity = count.min(dest.len() - 1);
    let bytes = formatted.as_bytes();
    let written = bytes.len().min(capacity);

    dest[..written].copy_from_slice(&bytes[..written]);
    dest[written] = 0;

    u32::try_from(written).unwrap_or(u32::MAX)
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn vscprintf_returns_byte_length_of_rendered_string() {
        assert_eq!(sys_vscprintf(""), 0);
        assert_eq!(sys_vscprintf("hello"), 5);
        assert_eq!(sys_vscprintf(&format!("{} + {} = {}", 2, 2, 4)), 9);
    }

    #[test]
    fn vscprintf_counts_bytes_not_chars_for_multibyte_utf8() {
        // "héllo": 'é' is 2 bytes in UTF-8, matching a byte-oriented
        // _vscprintf (not a codepoint count).
        let s = "h\u{00e9}llo";
        assert_eq!(sys_vscprintf(s), s.len() as u32);
        assert_eq!(sys_vscprintf(s), 6);
    }

    #[test]
    fn vsnprintf_copies_full_string_and_nul_terminates_when_buffer_fits() {
        let formatted = "hello";
        let mut dest = [0xAAu8; 16];
        let len = sys_vsnprintf(&mut dest, formatted.len(), formatted);
        assert_eq!(len, 5);
        assert_eq!(&dest[..5], b"hello");
        assert_eq!(dest[5], 0);
    }

    #[test]
    fn vsnprintf_matches_kyty_call_pattern_buffer_sized_len_plus_one() {
        // Mirrors `String::Printf`: buffer is `len + 1` bytes, `count == len`.
        let formatted = "the quick brown fox";
        let len = sys_vscprintf(formatted) as usize;
        let mut buffer = vec![0u8; len + 1];
        let written = sys_vsnprintf(&mut buffer, len, formatted);
        assert_eq!(written as usize, len);
        assert_eq!(&buffer[..len], formatted.as_bytes());
        assert_eq!(buffer[len], 0);
    }

    #[test]
    fn vsnprintf_truncates_when_count_is_smaller_than_string() {
        let formatted = "0123456789";
        let mut dest = [0xAAu8; 8];
        let len = sys_vsnprintf(&mut dest, 4, formatted);
        assert_eq!(len, 4);
        assert_eq!(&dest[..4], b"0123");
        assert_eq!(dest[4], 0);
        // Bytes past the terminator are left untouched (matches the C
        // contract: only `count` bytes + NUL are guaranteed written).
        assert_eq!(dest[5], 0xAA);
    }

    #[test]
    fn vsnprintf_clamps_to_dest_len_without_panicking_when_dest_is_smaller_than_count() {
        let formatted = "0123456789";
        let mut dest = [0xAAu8; 3];
        // Ask for far more than `dest` can hold; must not panic or overrun.
        let len = sys_vsnprintf(&mut dest, 100, formatted);
        assert_eq!(len, 2); // 2 data bytes + 1 NUL == dest.len()
        assert_eq!(&dest[..2], b"01");
        assert_eq!(dest[2], 0);
    }

    #[test]
    fn vsnprintf_on_empty_dest_returns_zero_without_panicking() {
        let mut dest: [u8; 0] = [];
        assert_eq!(sys_vsnprintf(&mut dest, 5, "hello"), 0);
    }

    #[test]
    fn vsnprintf_on_empty_formatted_string_writes_only_nul_terminator() {
        let mut dest = [0xAAu8; 4];
        let len = sys_vsnprintf(&mut dest, 4, "");
        assert_eq!(len, 0);
        assert_eq!(dest[0], 0);
    }
}
