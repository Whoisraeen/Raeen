//! Minimal C `printf`-style formatter for the HLE libc print family
//! (M1-C, wall #3).
//!
//! Formats a guest-supplied format string against the SysV integer argument
//! registers the dispatcher captured. Deliberately scoped to what M1-era
//! homebrew actually prints:
//!
//! - Conversions: `%s` `%c` `%d` `%i` `%u` `%x` `%X` `%p` `%%`
//! - Length modifiers `l`, `ll`, `z`, `t` (64-bit), `h`, `hh` (truncating),
//!   parsed and honored for integer width
//! - Flags `0` and `-`, plus a numeric field width, for integers and
//!   strings; precision (`.N`) for `%s`
//!
//! Anything else (`%f`, `%g`, `%n`, `*`-width, ...) is emitted verbatim
//! (the raw specifier text) with a `warn!` — visibly wrong output over
//! silently dropped output, and never a crash. Floating-point conversions
//! also can't be honest yet for a deeper reason: SysV passes them in XMM
//! registers, which the trampoline dispatcher does not capture.
//!
//! Variadic reality check: the dispatcher captures only the 6 SysV integer
//! registers, so a call can format at most 5 (printf) / 3 (snprintf)
//! variadic values; further conversions consume "arguments" that read as 0.
//! That is an honest, documented limit of the register-only dispatch, not a
//! parsing bug.

use crate::GuestMemory;
use tracing::warn;

/// Cap on how many bytes [`read_cstr`] will scan for a NUL terminator —
/// same bound (and rationale) as `libc.rs`'s `STRLEN_MAX_SCAN`.
pub(crate) const CSTR_MAX_SCAN: u64 = 1 << 20; // 1 MiB

/// Read a NUL-terminated guest string at `addr`, bounded by
/// [`CSTR_MAX_SCAN`]. Returns `None` if `addr` is unreadable at its very
/// first byte; an unterminated-but-readable run is returned truncated at
/// the cap (with a warning) rather than failing — real titles' strings are
/// terminated, and a partial read is more diagnosable than nothing.
pub(crate) fn read_cstr(mem: &dyn GuestMemory, addr: u64) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut byte = [0u8; 1];
    let mut off: u64 = 0;
    loop {
        let Some(cur) = addr.checked_add(off) else {
            warn!("read_cstr: address {addr:#x} + {off} overflowed");
            break;
        };
        if !mem.read(cur, &mut byte) {
            if off == 0 {
                return None;
            }
            warn!("read_cstr: string at {addr:#x} ran out of readable memory after {off} bytes");
            break;
        }
        if byte[0] == 0 {
            break;
        }
        out.push(byte[0]);
        off += 1;
        if off >= CSTR_MAX_SCAN {
            warn!("read_cstr: string at {addr:#x} unterminated after {CSTR_MAX_SCAN} bytes; truncating");
            break;
        }
    }
    Some(out)
}

/// How a parsed length modifier reshapes the raw 64-bit register value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Length {
    /// No modifier: C default-promotes variadic integers to `int`.
    Int,
    /// `hh`: `signed`/`unsigned char`.
    Char,
    /// `h`: `short`.
    Short,
    /// `l` / `ll` / `z` / `t` / `j`: 64-bit on the PS5's LP64 ABI.
    Long,
}

/// Format `fmt` (raw guest bytes, NUL already stripped) against `args`,
/// reading `%s` pointees through `mem`. Returns the formatted bytes.
pub(crate) fn format_c(
    fmt: &[u8],
    args: &mut dyn Iterator<Item = u64>,
    mem: &dyn GuestMemory,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(fmt.len());
    let mut i = 0usize;

    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }

        // Parse one conversion specification starting at the '%'.
        let spec_start = i;
        i += 1; // past '%'

        // Flags.
        let mut zero_pad = false;
        let mut left_align = false;
        while i < fmt.len() {
            match fmt[i] {
                b'0' => zero_pad = true,
                b'-' => left_align = true,
                // Parsed-and-ignored flags: still consumed so the
                // conversion character is found, but not honored.
                b'+' | b' ' | b'#' => {}
                _ => break,
            }
            i += 1;
        }

        // Field width (digits only; '*' is unsupported and falls through to
        // the verbatim-emit arm below via the unknown-conversion path).
        let mut width = 0usize;
        while i < fmt.len() && fmt[i].is_ascii_digit() {
            width = width
                .saturating_mul(10)
                .saturating_add((fmt[i] - b'0') as usize);
            i += 1;
        }

        // Precision (".N", digits only).
        let mut precision: Option<usize> = None;
        if i < fmt.len() && fmt[i] == b'.' {
            i += 1;
            let mut p = 0usize;
            while i < fmt.len() && fmt[i].is_ascii_digit() {
                p = p
                    .saturating_mul(10)
                    .saturating_add((fmt[i] - b'0') as usize);
                i += 1;
            }
            precision = Some(p);
        }

        // Length modifier.
        let mut length = Length::Int;
        while i < fmt.len() {
            match fmt[i] {
                b'l' | b'z' | b't' | b'j' => length = Length::Long,
                b'h' => {
                    length = if length == Length::Short {
                        Length::Char
                    } else {
                        Length::Short
                    };
                }
                _ => break,
            }
            i += 1;
        }

        let Some(&conv) = fmt.get(i) else {
            // Trailing lone '%...' with no conversion char: emit verbatim.
            out.extend_from_slice(&fmt[spec_start..]);
            break;
        };
        i += 1;

        match conv {
            b'%' => out.push(b'%'),
            b'c' => {
                let v = args.next().unwrap_or(0);
                pad(&mut out, &[(v & 0xFF) as u8], width, left_align, false);
            }
            b's' => {
                let ptr = args.next().unwrap_or(0);
                let mut s = match read_cstr(mem, ptr) {
                    Some(s) => s,
                    None => {
                        warn!("printf %s: unreadable guest string pointer {ptr:#x}");
                        format!("<bad ptr {ptr:#x}>").into_bytes()
                    }
                };
                if let Some(p) = precision {
                    s.truncate(p);
                }
                pad(&mut out, &s, width, left_align, false);
            }
            b'd' | b'i' => {
                let raw = args.next().unwrap_or(0);
                let v: i64 = match length {
                    Length::Int => raw as u32 as i32 as i64,
                    Length::Char => raw as u8 as i8 as i64,
                    Length::Short => raw as u16 as i16 as i64,
                    Length::Long => raw as i64,
                };
                pad(
                    &mut out,
                    v.to_string().as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                );
            }
            b'u' => {
                let raw = args.next().unwrap_or(0);
                let v: u64 = match length {
                    Length::Int => raw as u32 as u64,
                    Length::Char => raw as u8 as u64,
                    Length::Short => raw as u16 as u64,
                    Length::Long => raw,
                };
                pad(
                    &mut out,
                    v.to_string().as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                );
            }
            b'x' | b'X' => {
                let raw = args.next().unwrap_or(0);
                let v: u64 = match length {
                    Length::Int => raw as u32 as u64,
                    Length::Char => raw as u8 as u64,
                    Length::Short => raw as u16 as u64,
                    Length::Long => raw,
                };
                let s = if conv == b'x' {
                    format!("{v:x}")
                } else {
                    format!("{v:X}")
                };
                pad(&mut out, s.as_bytes(), width, left_align, zero_pad);
            }
            b'p' => {
                let v = args.next().unwrap_or(0);
                pad(
                    &mut out,
                    format!("{v:#x}").as_bytes(),
                    width,
                    left_align,
                    false,
                );
            }
            other => {
                // Unsupported conversion: emit the whole raw specifier
                // verbatim (visibly wrong beats silently dropped), consume
                // no argument — a float would be in an XMM register the
                // dispatcher never captured anyway.
                warn!(
                    "printf: unsupported conversion '%{}' emitted verbatim",
                    (other as char).escape_default()
                );
                out.extend_from_slice(&fmt[spec_start..i]);
            }
        }
    }

    out
}

/// Append `s` to `out`, padded to `width`: left-aligned pads with trailing
/// spaces; right-aligned pads with leading spaces, or leading zeros when
/// `zero_pad` (numeric conversions only — callers pass `false` otherwise).
/// Zero-padding a negative number keeps the sign first (`-0042`).
fn pad(out: &mut Vec<u8>, s: &[u8], width: usize, left_align: bool, zero_pad: bool) {
    let pad_n = width.saturating_sub(s.len());
    if pad_n == 0 {
        out.extend_from_slice(s);
        return;
    }
    if left_align {
        out.extend_from_slice(s);
        out.extend(std::iter::repeat_n(b' ', pad_n));
    } else if zero_pad {
        if let Some((&b'-', digits)) = s.split_first() {
            out.push(b'-');
            out.extend(std::iter::repeat_n(b'0', pad_n));
            out.extend_from_slice(digits);
        } else {
            out.extend(std::iter::repeat_n(b'0', pad_n));
            out.extend_from_slice(s);
        }
    } else {
        out.extend(std::iter::repeat_n(b' ', pad_n));
        out.extend_from_slice(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestMemory;

    /// Format against a `TestMemory` whose bytes are irrelevant (no `%s`).
    fn fmt_no_mem(fmt: &str, args: &[u64]) -> String {
        let mem = TestMemory::new(16);
        let mut it = args.iter().copied();
        String::from_utf8(format_c(fmt.as_bytes(), &mut it, &mem)).unwrap()
    }

    #[test]
    fn plain_text_passes_through() {
        assert_eq!(fmt_no_mem("hello world\n", &[]), "hello world\n");
    }

    #[test]
    fn percent_d_formats_signed_int_with_default_int_promotion() {
        assert_eq!(fmt_no_mem("v=%d", &[42]), "v=42");
        // A negative `int` arrives as a sign-extended (or zero-garbage-high)
        // register; only the low 32 bits are the value.
        assert_eq!(fmt_no_mem("v=%d", &[(-7i32) as u32 as u64]), "v=-7");
        // High garbage above bit 31 must not leak into a plain %d.
        assert_eq!(fmt_no_mem("v=%d", &[0xFFFF_FFFF_0000_002Au64]), "v=42");
    }

    #[test]
    fn percent_ld_uses_all_64_bits() {
        assert_eq!(fmt_no_mem("v=%ld", &[(-7i64) as u64]), "v=-7");
        assert_eq!(fmt_no_mem("v=%lld", &[9_000_000_000u64]), "v=9000000000");
        assert_eq!(fmt_no_mem("v=%zu", &[9_000_000_000u64]), "v=9000000000");
    }

    #[test]
    fn percent_u_x_p_and_escapes() {
        assert_eq!(fmt_no_mem("%u", &[0xFFFF_FFFFu64]), "4294967295");
        assert_eq!(fmt_no_mem("%x", &[0xABCDu64]), "abcd");
        assert_eq!(fmt_no_mem("%X", &[0xABCDu64]), "ABCD");
        assert_eq!(fmt_no_mem("%p", &[0x1000u64]), "0x1000");
        assert_eq!(fmt_no_mem("100%%", &[]), "100%");
        assert_eq!(fmt_no_mem("%c!", &[b'A' as u64]), "A!");
    }

    #[test]
    fn width_and_zero_padding() {
        assert_eq!(fmt_no_mem("[%5d]", &[42]), "[   42]");
        assert_eq!(fmt_no_mem("[%-5d]", &[42]), "[42   ]");
        assert_eq!(fmt_no_mem("[%05d]", &[42]), "[00042]");
        assert_eq!(fmt_no_mem("[%08x]", &[0xABCu64]), "[00000abc]");
        assert_eq!(fmt_no_mem("[%05d]", &[(-42i32) as u32 as u64]), "[-0042]");
    }

    #[test]
    fn percent_s_reads_guest_string_with_width_and_precision() {
        let mem = TestMemory::new(64);
        assert!(mem.write(0x10, b"world\0"));
        let mut it = [0x10u64].iter().copied();
        assert_eq!(
            String::from_utf8(format_c(b"hello %s", &mut it, &mem)).unwrap(),
            "hello world"
        );

        let mut it = [0x10u64].iter().copied();
        assert_eq!(
            String::from_utf8(format_c(b"[%8s]", &mut it, &mem)).unwrap(),
            "[   world]"
        );

        let mut it = [0x10u64].iter().copied();
        assert_eq!(
            String::from_utf8(format_c(b"[%.3s]", &mut it, &mem)).unwrap(),
            "[wor]"
        );
    }

    #[test]
    fn percent_s_with_unreadable_pointer_reports_instead_of_crashing() {
        let mem = TestMemory::new(16);
        let mut it = [0xDEAD_0000u64].iter().copied();
        let s = String::from_utf8(format_c(b"%s", &mut it, &mem)).unwrap();
        assert_eq!(s, "<bad ptr 0xdead0000>");
    }

    #[test]
    fn unsupported_conversion_is_emitted_verbatim_and_consumes_no_argument() {
        // %f is emitted raw; the one argument then feeds %d, proving %f
        // consumed nothing.
        assert_eq!(fmt_no_mem("%f then %d", &[42]), "%f then 42");
    }

    #[test]
    fn missing_arguments_read_as_zero() {
        assert_eq!(fmt_no_mem("%d %d", &[7]), "7 0");
    }

    #[test]
    fn read_cstr_bounds() {
        let mem = TestMemory::new(8);
        assert!(mem.write(0, b"abc\0"));
        assert_eq!(read_cstr(&mem, 0).unwrap(), b"abc");
        // First byte unreadable -> None.
        assert!(read_cstr(&mem, 0x100).is_none());
        // Readable but unterminated to the end of memory -> truncated, not None.
        assert!(mem.write(4, b"defg")); // bytes 4..8, no NUL, memory ends
        assert_eq!(read_cstr(&mem, 4).unwrap(), b"defg");
    }
}
