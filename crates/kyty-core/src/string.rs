//! Port of Kyty's `Kyty::Core::String` / `Kyty::Core::StringList` / (the
//! subset of) `Kyty::Core::Char` that `String` needs
//! (`reference/kyty/source/include/Kyty/Core/String.h`,
//! `reference/kyty/source/lib/Core/src/String.cpp`).
//!
//! # Internal encoding (read from the C++ source)
//!
//! `String::DataType = SimpleArray<char32_t>`: Kyty's `String` is a
//! ref-counted, copy-on-write array of **UTF-32 code points**, not bytes —
//! unlike [`crate::string8::String8`] (a byte string used for filenames/CSV/
//! arbitrary narrow data), `Core::String` is "a proper Unicode string" (see
//! that module's doc comment). The backing array always carries one hidden
//! trailing `U+0000` that every method accounts for by hand (`Size()` returns
//! `m_data->Size() - 1`, `RemoveAt` re-appends `'\0'` if it got removed, ...).
//!
//! # Std mapping
//!
//! `String`'s entire public surface — `Size`/`operator[]`/`At`/`Mid`/`Left`/
//! `Right`/`FindIndex`/... — indexes and slices by **code-point position**,
//! not byte offset. A `std::string::String` (UTF-8 bytes) would need to
//! re-decode on every positional access to honor that contract, so the
//! faithful thin wrapper here is `Vec<char>` (Rust's `char` is a Unicode
//! *scalar value* — the same domain as a *valid* `char32_t`). This mirrors
//! exactly how [`crate::simple_array::SimpleArray<T>`] wraps `Vec<T>` and how
//! [`String8`] wraps `Vec<u8>`:
//! - the ref-counted/CoW `SimpleArray<char32_t>*` indirection is dropped —
//!   `Vec<char>` + `#[derive(Clone)]` (eager deep copy) is observably
//!   equivalent without unsafe code, manual refcounting, or raw pointers;
//! - the hidden trailing `'\0'` bookkeeping is dropped entirely — `Vec<char>`
//!   already knows its own length.
//!
//! For interop with ordinary UTF-8 text: [`String::utf8_str`] returns the
//! already-ported [`String8`] (mirroring the C++ alias `Utf8 = Vector<char>`
//! and the `explicit String(const Utf8&)` constructor, ported here as
//! [`String::from_string8`]/[`From<&String8>`]); `impl From<&str> for String`
//! and `impl Display for String` interop with Rust's own UTF-8
//! `std::string::String` for callers that just want ordinary text.
//!
//! Rust's `char` cannot represent the C++ `char32_t` domain exactly (no
//! surrogate halves `U+D800..=U+DFFF`, nothing above `U+10FFFF`) — decoding
//! any such (malformed-input-only) value maps it to `U+FFFD` (REPLACEMENT
//! CHARACTER) instead. This is an accepted, documented divergence, reachable
//! only via malformed input, exactly like `String8`'s documented
//! locale/table divergences.
//!
//! `Char`'s Unicode property tables (`CharUcd`'s `g_char_prop_p`/
//! `g_char_prop_r`, driving `IsAlpha`/`IsDecimal`/`IsSpace`/`ToUpper`/
//! `ToLower`/...) are **not** ported (per project convention); this module
//! uses Rust `char::is_alphabetic`/`is_numeric`/`is_alphanumeric`/
//! `is_uppercase`/`is_lowercase`/`is_whitespace`/`to_uppercase`/
//! `to_lowercase` instead. Because `char::to_uppercase()`/`to_lowercase()`
//! can yield *more than one* code point for a handful of characters (e.g.
//! `'ß'.to_uppercase()` == `"SS"`), while Kyty's `Char::ToUpper`/`ToLower`
//! (table-driven `case_offset`) always map exactly one input code point to
//! exactly one output code point (preserving `Size()`), only the *first*
//! code point of Rust's full case mapping is kept — a documented divergence
//! for a full-case-folding edge case Kyty's own callers never hit.
//!
//! `Char::ToCp866`/`ToCp1251`/`ReadCp866`/`ReadCp1251` (fixed legacy 8-bit
//! codepage conversion tables, *not* general Unicode data) have no std
//! equivalent and are ported verbatim as small lookup tables/`match`
//! statements below.
//! `Char::ToUtf8`/`ToUtf16`/`ToUtf32`/`ReadUtf8`/`ReadUtf16`/`ReadUtf32` (hand
//! -rolled reimplementations of the *standard* UTF-8/UTF-16 algorithms) are
//! instead ported via Rust's own std encoders/decoders
//! (`char::encode_utf8`/`str::encode_utf16`/`char::decode_utf16`/
//! `str::from_utf8`), which implement the identical algorithm safely. Kyty's
//! `Read*` recursively skip every `U+FEFF` (BOM) code point anywhere in the
//! stream (not just a leading BOM) — reproduced here by filtering `U+FEFF`
//! out of the decoded stream.
//!
//! `Printf`/`FromPrintf` (C varargs `printf` over a `va_list`) are
//! intentionally **not** ported, for the same reasoning as `String8`: no
//! faithful *safe* Rust equivalent exists. Use `format!` + `String::from`.
//!
//! `Hash()` is ported using the already-ported [`crate::hash::hash`] (Kyty's
//! own one-at-a-time hash) over the little-endian byte representation of
//! each stored code point (as `u32`), matching
//! `SimpleArray<char32_t>::Hash()`'s raw-byte-hash semantics byte-for-byte
//! (unlike `simple_array.rs`'s own `Hash()`, which documents switching to
//! Rust's `Hash` trait — that divergence isn't needed here since Kyty's own
//! hash function is already available in this crate).
//!
//! `StringList` (C++: `class StringList : public Vector<String>`, plus
//! `Contains`/`Concat`/`Equal`/`EqualNoCase`) is ported as a thin wrapper over
//! `Vec<String>` exposing just the members callers need, mirroring
//! `StringList8`'s own approach.

use crate::byte_buffer::ByteBuffer;
use crate::string8::String8;
use std::char::{REPLACEMENT_CHARACTER, decode_utf16};
use std::string::String as StdString;

/// `STRING_INVALID_INDEX`: sentinel returned by the `Find*` family when no
/// match exists, and passed by callers who want `FindLastIndex`'s "search
/// from the end" default.
pub const INVALID_INDEX: u32 = u32::MAX;

/// `String::Case`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Case {
    Insensitive = 0,
    Sensitive = 1,
}

/// `String::SplitType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SplitType {
    WithEmptyParts,
    SplitNoEmptyParts,
}

/// `Kyty::Core::String`: a thin wrapper over `Vec<char>` exposing the
/// original's method names/semantics. See the module doc comment for the
/// full std-mapping rationale.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct String {
    data: Vec<char>,
}

fn char_to_upper(c: char) -> char {
    c.to_uppercase().next().unwrap_or(c)
}

fn char_to_lower(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

fn char_eq_no_case(a: char, b: char) -> bool {
    char_to_upper(a) == char_to_upper(b) || char_to_lower(a) == char_to_lower(b)
}

fn compare_equal(a: &[char], b: &[char], from1: usize, from2: usize, size: usize) -> bool {
    a[from1..from1 + size] == b[from2..from2 + size]
}

fn compare_equal_no_case(a: &[char], b: &[char], from1: usize, from2: usize, size: usize) -> bool {
    (0..size).all(|i| char_eq_no_case(a[from1 + i], b[from2 + i]))
}

// ---------------------------------------------------------------------
// CP866 / CP1251: fixed legacy 8-bit codepage conversion tables (not
// Unicode property data — see module doc comment for why these are ported
// verbatim while `CharUcd` is not).
// ---------------------------------------------------------------------

#[rustfmt::skip]
const CP866_DECODE: [u32; 128] = [
    0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
    0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427, 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
    0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F,
    0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562, 0x2556, 0x2555, 0x2563, 0x2551, 0x2557, 0x255D, 0x255C, 0x255B, 0x2510,
    0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x255E, 0x255F, 0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x2567,
    0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256B, 0x256A, 0x2518, 0x250C, 0x2588, 0x2584, 0x258C, 0x2590, 0x2580,
    0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F,
    0x0401, 0x0451, 0x0404, 0x0454, 0x0407, 0x0457, 0x040E, 0x045E, 0x00B0, 0x2219, 0x00B7, 0x221A, 0x2116, 0x00A4, 0x25A0, 0x00A0,
];

#[rustfmt::skip]
const CP1251_DECODE: [u32; 128] = [
    0x0402, 0x0403, 0x201A, 0x0453, 0x201E, 0x2026, 0x2020, 0x2021, 0x20AC, 0x2030, 0x0409, 0x2039, 0x040A, 0x040C, 0x040B, 0x040F,
    0x0452, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x0401, 0x2122, 0x0459, 0x203A, 0x045A, 0x045C, 0x045B, 0x045F,
    0x00A0, 0x040E, 0x045E, 0x0408, 0x00A4, 0x0490, 0x00A6, 0x00A7, 0x0401, 0x00A9, 0x0404, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x0407,
    0x00B0, 0x00B1, 0x0406, 0x0456, 0x0491, 0x00B5, 0x00B6, 0x00B7, 0x0451, 0x2116, 0x0454, 0x00BB, 0x0458, 0x0405, 0x0455, 0x0457,
    0x0410, 0x0411, 0x0412, 0x0413, 0x0414, 0x0415, 0x0416, 0x0417, 0x0418, 0x0419, 0x041A, 0x041B, 0x041C, 0x041D, 0x041E, 0x041F,
    0x0420, 0x0421, 0x0422, 0x0423, 0x0424, 0x0425, 0x0426, 0x0427, 0x0428, 0x0429, 0x042A, 0x042B, 0x042C, 0x042D, 0x042E, 0x042F,
    0x0430, 0x0431, 0x0432, 0x0433, 0x0434, 0x0435, 0x0436, 0x0437, 0x0438, 0x0439, 0x043A, 0x043B, 0x043C, 0x043D, 0x043E, 0x043F,
    0x0440, 0x0441, 0x0442, 0x0443, 0x0444, 0x0445, 0x0446, 0x0447, 0x0448, 0x0449, 0x044A, 0x044B, 0x044C, 0x044D, 0x044E, 0x044F,
];

fn cp866_decode_byte(b: u8) -> char {
    if b < 128 {
        return b as char;
    }
    char::from_u32(CP866_DECODE[(b - 128) as usize]).unwrap_or(REPLACEMENT_CHARACTER)
}

fn cp1251_decode_byte(b: u8) -> char {
    if b < 128 {
        return b as char;
    }
    char::from_u32(CP1251_DECODE[(b - 128) as usize]).unwrap_or(REPLACEMENT_CHARACTER)
}

#[rustfmt::skip]
fn cp866_encode_char(ch: char) -> u8 {
    let u = ch as u32;
    if u < 128 {
        return u as u8;
    }
    match u {
        0x0410 => 128, 0x0411 => 129, 0x0412 => 130, 0x0413 => 131,
        0x0414 => 132, 0x0415 => 133, 0x0416 => 134, 0x0417 => 135,
        0x0418 => 136, 0x0419 => 137, 0x041A => 138, 0x041B => 139,
        0x041C => 140, 0x041D => 141, 0x041E => 142, 0x041F => 143,

        0x0420 => 144, 0x0421 => 145, 0x0422 => 146, 0x0423 => 147,
        0x0424 => 148, 0x0425 => 149, 0x0426 => 150, 0x0427 => 151,
        0x0428 => 152, 0x0429 => 153, 0x042A => 154, 0x042B => 155,
        0x042C => 156, 0x042D => 157, 0x042E => 158, 0x042F => 159,

        0x0430 => 160, 0x0431 => 161, 0x0432 => 162, 0x0433 => 163,
        0x0434 => 164, 0x0435 => 165, 0x0436 => 166, 0x0437 => 167,
        0x0438 => 168, 0x0439 => 169, 0x043A => 170, 0x043B => 171,
        0x043C => 172, 0x043D => 173, 0x043E => 174, 0x043F => 175,

        0x2591 => 176, 0x2592 => 177, 0x2593 => 178, 0x2502 => 179,
        0x2524 => 180, 0x2561 => 181, 0x2562 => 182, 0x2556 => 183,
        0x2555 => 184, 0x2563 => 185, 0x2551 => 186, 0x2557 => 187,
        0x255D => 188, 0x255C => 189, 0x255B => 190, 0x2510 => 191,

        0x2514 => 192, 0x2534 => 193, 0x252C => 194, 0x251C => 195,
        0x2500 => 196, 0x253C => 197, 0x255E => 198, 0x255F => 199,
        0x255A => 200, 0x2554 => 201, 0x2569 => 202, 0x2566 => 203,
        0x2560 => 204, 0x2550 => 205, 0x256C => 206, 0x2567 => 207,

        0x2568 => 208, 0x2564 => 209, 0x2565 => 210, 0x2559 => 211,
        0x2558 => 212, 0x2552 => 213, 0x2553 => 214, 0x256B => 215,
        0x256A => 216, 0x2518 => 217, 0x250C => 218, 0x2588 => 219,
        0x2584 => 220, 0x258C => 221, 0x2590 => 222, 0x2580 => 223,

        0x0440 => 224, 0x0441 => 225, 0x0442 => 226, 0x0443 => 227,
        0x0444 => 228, 0x0445 => 229, 0x0446 => 230, 0x0447 => 231,
        0x0448 => 232, 0x0449 => 233, 0x044A => 234, 0x044B => 235,
        0x044C => 236, 0x044D => 237, 0x044E => 238, 0x044F => 239,

        0x0401 => 240, 0x0451 => 241, 0x0404 => 242, 0x0454 => 243,
        0x0407 => 244, 0x0457 => 245, 0x040E => 246, 0x045E => 247,
        0x00B0 => 248, 0x2219 => 249, 0x00B7 => 250, 0x221A => 251,
        0x2116 => 252, 0x00A4 => 253, 0x25A0 => 254, 0x00A0 => 255,

        0x2193 => 25,
        0x2191 => 24,
        0x2192 => 26,
        0x2190 => 27,

        _ => 240,
    }
}

#[rustfmt::skip]
fn cp1251_encode_char(ch: char) -> u8 {
    let u = ch as u32;
    if u < 128 {
        return u as u8;
    }
    match u {
        0x0402 => 128, 0x0403 => 129, 0x201A => 130, 0x0453 => 131,
        0x201E => 132, 0x2026 => 133, 0x2020 => 134, 0x2021 => 135,
        0x20AC => 136, 0x2030 => 137, 0x0409 => 138, 0x2039 => 139,
        0x040A => 140, 0x040C => 141, 0x040B => 142, 0x040F => 143,

        0x0452 => 144, 0x2018 => 145, 0x2019 => 146, 0x201C => 147,
        0x201D => 148, 0x2022 => 149, 0x2013 => 150, 0x2014 => 151,
        0x2122 => 153, 0x0459 => 154, 0x203A => 155,
        0x045A => 156, 0x045C => 157, 0x045B => 158, 0x045F => 159,

        0x00A0 => 160, 0x040E => 161, 0x045E => 162, 0x0408 => 163,
        0x00A4 => 164, 0x0490 => 165, 0x00A6 => 166, 0x00A7 => 167,
        0x0401 => 168, 0x00A9 => 169, 0x0404 => 170, 0x00AB => 171,
        0x00AC => 172, 0x00AD => 173, 0x00AE => 174, 0x0407 => 175,

        0x00B0 => 176, 0x00B1 => 177, 0x0406 => 178, 0x0456 => 179,
        0x0491 => 180, 0x00B5 => 181, 0x00B6 => 182, 0x00B7 => 183,
        0x0451 => 184, 0x2116 => 185, 0x0454 => 186, 0x00BB => 187,
        0x0458 => 188, 0x0405 => 189, 0x0455 => 190, 0x0457 => 191,

        0x0410 => 192, 0x0411 => 193, 0x0412 => 194, 0x0413 => 195,
        0x0414 => 196, 0x0415 => 197, 0x0416 => 198, 0x0417 => 199,
        0x0418 => 200, 0x0419 => 201, 0x041A => 202, 0x041B => 203,
        0x041C => 204, 0x041D => 205, 0x041E => 206, 0x041F => 207,

        0x0420 => 208, 0x0421 => 209, 0x0422 => 210, 0x0423 => 211,
        0x0424 => 212, 0x0425 => 213, 0x0426 => 214, 0x0427 => 215,
        0x0428 => 216, 0x0429 => 217, 0x042A => 218, 0x042B => 219,
        0x042C => 220, 0x042D => 221, 0x042E => 222, 0x042F => 223,

        0x0430 => 224, 0x0431 => 225, 0x0432 => 226, 0x0433 => 227,
        0x0434 => 228, 0x0435 => 229, 0x0436 => 230, 0x0437 => 231,
        0x0438 => 232, 0x0439 => 233, 0x043A => 234, 0x043B => 235,
        0x043C => 236, 0x043D => 237, 0x043E => 238, 0x043F => 239,

        0x0440 => 240, 0x0441 => 241, 0x0442 => 242, 0x0443 => 243,
        0x0444 => 244, 0x0445 => 245, 0x0446 => 246, 0x0447 => 247,
        0x0448 => 248, 0x0449 => 249, 0x044A => 250, 0x044B => 251,
        0x044C => 252, 0x044D => 253, 0x044E => 254, 0x044F => 255,

        0x2193 => 25,
        0x2191 => 24,
        0x2192 => 26,
        0x2190 => 27,

        _ => 240,
    }
}

impl String {
    /// `String()`: empty string.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// `String(char32_t ch, uint32_t repeat = 1)`. Rust has no default
    /// arguments; pass `1` explicitly for the C++ default.
    pub fn from_char(ch: char, repeat: u32) -> Self {
        Self {
            data: vec![ch; repeat as usize],
        }
    }

    /// `static String FromUtf8(const char* utf8_str)`. Decodes with Rust's
    /// own UTF-8 decoder (lossy — invalid sequences become `U+FFFD`, per
    /// module doc comment) and drops every `U+FEFF` (BOM), matching
    /// `Char::ReadUtf8`'s recursive BOM skip.
    pub fn from_utf8_bytes(bytes: &[u8]) -> Self {
        let text = StdString::from_utf8_lossy(bytes);
        Self {
            data: text.chars().filter(|&c| c != '\u{FEFF}').collect(),
        }
    }

    /// `explicit String(const Utf8& utf8)`.
    pub fn from_string8(utf8: &String8) -> Self {
        Self::from_utf8_bytes(utf8.as_bytes())
    }

    /// `static String FromUtf16(const char16_t* utf16_str)`.
    pub fn from_utf16(units: &[u16]) -> Self {
        let data = decode_utf16(units.iter().copied())
            .map(|r| r.unwrap_or(REPLACEMENT_CHARACTER))
            .filter(|&c| c != '\u{FEFF}')
            .collect();
        Self { data }
    }

    /// `static String FromUtf32(const char32_t* utf32_str)`.
    pub fn from_utf32(code_points: &[u32]) -> Self {
        let data = code_points
            .iter()
            .map(|&cp| char::from_u32(cp).unwrap_or(REPLACEMENT_CHARACTER))
            .filter(|&c| c != '\u{FEFF}')
            .collect();
        Self { data }
    }

    /// `static String FromCp866(const char* utf8_str)` (parameter is really
    /// CP866 bytes despite the C++ name).
    pub fn from_cp866(bytes: &[u8]) -> Self {
        Self {
            data: bytes.iter().map(|&b| cp866_decode_byte(b)).collect(),
        }
    }

    /// `static String FromCp1251(const char* utf8_str)`.
    pub fn from_cp1251(bytes: &[u8]) -> Self {
        Self {
            data: bytes.iter().map(|&b| cp1251_decode_byte(b)).collect(),
        }
    }

    /// `Size()`.
    pub fn size(&self) -> u32 {
        self.data.len() as u32
    }

    /// `IsEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `IsInvalid()`. In C++ this reports whether `m_data == nullptr`, a
    /// state only reachable via a moved-from `String`; Rust's ownership model
    /// makes that unreachable through safe code, so this always returns
    /// `false` (kept for API parity, matching `String8::is_invalid`).
    pub fn is_invalid(&self) -> bool {
        false
    }

    /// `Clear()`.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// `At(uint32_t index) const`.
    pub fn at(&self, index: u32) -> &char {
        crate::exit_if!(index >= self.size());
        &self.data[index as usize]
    }

    /// `operator[](uint32_t)` (mutable form).
    #[allow(clippy::should_implement_trait)]
    pub fn index_mut(&mut self, index: u32) -> &mut char {
        crate::exit_if!(index >= self.size());
        &mut self.data[index as usize]
    }

    /// `operator[](uint32_t) const`.
    #[allow(clippy::should_implement_trait)]
    pub fn index(&self, index: u32) -> &char {
        self.at(index)
    }

    /// `GetData()` (mutable view).
    pub fn get_data_mut(&mut self) -> &mut [char] {
        &mut self.data
    }

    /// `GetData() const` / `GetDataConst() const`.
    pub fn get_data_const(&self) -> &[char] {
        &self.data
    }

    /// `utf8_str()`: encodes with Rust's own (standard-conformant) UTF-8
    /// encoder into the already-ported [`String8`].
    pub fn utf8_str(&self) -> String8 {
        let s: StdString = self.data.iter().collect();
        String8::from(s)
    }

    /// `utf16_str()`.
    pub fn utf16_str(&self) -> Vec<u16> {
        let s: StdString = self.data.iter().collect();
        s.encode_utf16().collect()
    }

    /// `utf32_str()`.
    pub fn utf32_str(&self) -> Vec<u32> {
        self.data.iter().map(|&c| c as u32).collect()
    }

    /// `cp866_str()`.
    pub fn cp866_str(&self) -> Vec<u8> {
        self.data.iter().map(|&c| cp866_encode_char(c)).collect()
    }

    /// `cp1251_str()`.
    pub fn cp1251_str(&self) -> Vec<u8> {
        self.data.iter().map(|&c| cp1251_encode_char(c)).collect()
    }

    /// `Equal(const String& src)`.
    pub fn equal(&self, other: &String) -> bool {
        self.data == other.data
    }

    /// `Equal(char32_t ch)`.
    pub fn equal_char(&self, ch: char) -> bool {
        self.data.len() == 1 && self.data[0] == ch
    }

    /// `Equal(const char* utf8_str)`.
    pub fn equal_str(&self, s: &str) -> bool {
        self.equal(&String::from_utf8_bytes(s.as_bytes()))
    }

    /// `EqualNoCase(const String& src)`.
    pub fn equal_no_case(&self, other: &String) -> bool {
        self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(&a, &b)| char_eq_no_case(a, b))
    }

    /// `EqualNoCase(char32_t ch)`.
    pub fn equal_no_case_char(&self, ch: char) -> bool {
        self.data.len() == 1 && char_eq_no_case(self.data[0], ch)
    }

    /// `EqualNoCase(const char* utf8_str)`.
    pub fn equal_no_case_str(&self, s: &str) -> bool {
        self.equal_no_case(&String::from_utf8_bytes(s.as_bytes()))
    }

    /// `Mid(uint32_t first, uint32_t count)`.
    pub fn mid(&self, first: u32, count: u32) -> String {
        let size = self.size();
        if first >= size {
            return String::new();
        }
        let mut count = count;
        if u64::from(first) + u64::from(count) > u64::from(size) {
            count = size - first;
        }
        if first == 0 && count == size {
            return self.clone();
        }
        String {
            data: self.data[first as usize..(first + count) as usize].to_vec(),
        }
    }

    /// `Mid(uint32_t first)` (single-argument overload).
    pub fn mid_from(&self, first: u32) -> String {
        self.mid(first, self.size().saturating_sub(first))
    }

    /// `Left(uint32_t count)`.
    pub fn left(&self, count: u32) -> String {
        self.mid(0, count)
    }

    /// `Right(uint32_t count)`.
    pub fn right(&self, count: u32) -> String {
        let size = self.size();
        if count >= size {
            return self.clone();
        }
        self.mid(size - count, count)
    }

    /// `ToUpper()`. See module doc comment for the "first code point only"
    /// divergence from Rust's full Unicode case folding.
    pub fn to_upper(&self) -> String {
        String {
            data: self.data.iter().map(|&c| char_to_upper(c)).collect(),
        }
    }

    /// `ToLower()`.
    pub fn to_lower(&self) -> String {
        String {
            data: self.data.iter().map(|&c| char_to_lower(c)).collect(),
        }
    }

    /// `TrimRight()`.
    pub fn trim_right(&self) -> String {
        let size = self.size();
        for i in 0..size {
            if !self.data[(size - i - 1) as usize].is_whitespace() {
                return self.mid(0, size - i);
            }
        }
        String::new()
    }

    /// `TrimLeft()`.
    pub fn trim_left(&self) -> String {
        let size = self.size();
        for i in 0..size {
            if !self.data[i as usize].is_whitespace() {
                return self.mid(i, size - i);
            }
        }
        String::new()
    }

    /// `Trim()`.
    pub fn trim(&self) -> String {
        let size = self.size();
        let mut left_pos = size;
        let mut count = 0u32;
        for i in 0..size {
            if !self.data[i as usize].is_whitespace() {
                left_pos = i;
                break;
            }
        }
        for i in 0..size.saturating_sub(left_pos) {
            if !self.data[(size - i - 1) as usize].is_whitespace() {
                count = size - left_pos - i;
                break;
            }
        }
        self.mid(left_pos, count)
    }

    /// `Simplify()`.
    pub fn simplify(&self) -> String {
        let mut data = Vec::with_capacity(self.data.len());
        let mut prev_space = true;
        for &c in &self.data {
            if c.is_whitespace() {
                if prev_space {
                    continue;
                }
                prev_space = true;
            } else {
                prev_space = false;
            }
            data.push(c);
        }
        String { data }.trim_right()
    }

    /// `ReplaceChar(char32_t old_char, char32_t new_char, Case cs = Sensitive)`.
    pub fn replace_char(&self, old_char: char, new_char: char, cs: Case) -> String {
        let data = self
            .data
            .iter()
            .map(|&c| {
                let matched = match cs {
                    Case::Sensitive => c == old_char,
                    Case::Insensitive => char_eq_no_case(c, old_char),
                };
                if matched { new_char } else { c }
            })
            .collect();
        String { data }
    }

    /// `ReplaceStr(const String& old_str, const String& new_str, Case cs = Sensitive)`.
    pub fn replace_str(&self, old_str: &String, new_str: &String, cs: Case) -> String {
        let mut result = String::new();
        let mut start: u32 = 0;
        let extra: u32 = u32::from(old_str.is_empty());
        let sep_size = old_str.size();

        loop {
            let end = self.find_index(old_str, start + extra, cs);
            if end == INVALID_INDEX {
                break;
            }
            if start != end {
                result += &self.mid(start, end - start);
            }
            result += new_str;
            start = end + sep_size;
        }

        if start != self.size() {
            result += &self.mid(start, self.size() - start);
        }
        result
    }

    /// `RemoveAt(uint32_t index, uint32_t count = 1)`.
    pub fn remove_at(&self, index: u32, count: u32) -> String {
        let size = self.size();
        if index >= size {
            return self.clone();
        }
        let mut count = count;
        if u64::from(index) + u64::from(count) > u64::from(size) {
            count = size - index;
        }
        let mut data = self.data.clone();
        let start = index as usize;
        data.drain(start..start + count as usize);
        String { data }
    }

    /// `RemoveChar(char32_t ch, Case cs = Sensitive)`.
    pub fn remove_char(&self, ch: char, cs: Case) -> String {
        let data = self
            .data
            .iter()
            .copied()
            .filter(|&c| match cs {
                Case::Sensitive => c != ch,
                Case::Insensitive => !char_eq_no_case(c, ch),
            })
            .collect();
        String { data }
    }

    /// `RemoveStr(const String& str, Case cs = Sensitive)`. Kyty's own
    /// implementation is structurally `ReplaceStr` with an empty
    /// replacement, so this is ported as exactly that call, matching
    /// `String8::remove_str`.
    pub fn remove_str(&self, s: &String, cs: Case) -> String {
        self.replace_str(s, &String::new(), cs)
    }

    /// `RemoveLast(uint32_t num)`.
    pub fn remove_last(&self, num: u32) -> String {
        let size = self.size();
        if num >= size {
            return String::new();
        }
        self.left(size - num)
    }

    /// `RemoveFirst(uint32_t num)`.
    pub fn remove_first(&self, num: u32) -> String {
        let size = self.size();
        if num >= size {
            return String::new();
        }
        self.right(size - num)
    }

    /// `InsertAt(uint32_t index, const String& str)`.
    pub fn insert_at(&self, index: u32, s: &String) -> String {
        let size = self.size();
        let mut result = self.mid(0, index);
        result.data.extend_from_slice(&s.data);
        result
            .data
            .extend_from_slice(&self.mid(index, size.saturating_sub(index)).data);
        result
    }

    /// `SafeLua()`.
    pub fn safe_lua(&self) -> String {
        self.replace_str(&String::from("\\"), &String::from("\\\\"), Case::Sensitive)
            .replace_str(&String::from("'"), &String::from("\\'"), Case::Sensitive)
    }

    /// `SafeCsv()`.
    pub fn safe_csv(&self) -> String {
        let add_space = self.starts_with_char('+', Case::Sensitive)
            || self.starts_with_char('=', Case::Sensitive)
            || self.starts_with_char('-', Case::Sensitive);

        let needs_quoting = self.contains_char('"', Case::Sensitive)
            || self.contains_char(';', Case::Sensitive)
            || self.contains_char('+', Case::Sensitive)
            || self.contains_char('=', Case::Sensitive)
            || self.contains_char('-', Case::Sensitive);

        if needs_quoting {
            let mut r = String::from("\"");
            if add_space {
                r += ' ';
            }
            r += &self.replace_str(&String::from("\""), &String::from("\"\""), Case::Sensitive);
            r += '"';
            return r;
        }
        self.clone()
    }

    /// `FindIndex(const String& str, uint32_t from = 0, Case cs = Sensitive)`.
    pub fn find_index(&self, s: &String, from: u32, cs: Case) -> u32 {
        let size = self.size();
        if from >= size {
            return INVALID_INDEX;
        }
        let str_size = s.size();
        if str_size == 0 {
            return from;
        }
        match cs {
            Case::Sensitive => {
                let mut i = from;
                while i + str_size <= size {
                    if compare_equal(&self.data, &s.data, i as usize, 0, str_size as usize) {
                        return i;
                    }
                    i += 1;
                }
            }
            Case::Insensitive => {
                let mut i = from;
                while i + str_size <= size {
                    if compare_equal_no_case(&self.data, &s.data, i as usize, 0, str_size as usize)
                    {
                        return i;
                    }
                    i += 1;
                }
            }
        }
        INVALID_INDEX
    }

    /// `FindLastIndex(const String& str, uint32_t from = STRING_INVALID_INDEX, Case cs = Sensitive)`.
    ///
    /// Diverges from the original in one edge case (matching
    /// `String8::find_last_index`): if `str` is longer than `self`, C++
    /// underflows `size - str_size` (unsigned wraparound / UB); this always
    /// safely returns [`INVALID_INDEX`] instead.
    pub fn find_last_index(&self, s: &String, from: u32, cs: Case) -> u32 {
        let size = self.size();
        if size == 0 {
            return INVALID_INDEX;
        }
        let mut from = from;
        if from >= size {
            from = size - 1;
        }
        let str_size = s.size();
        if str_size == 0 {
            return from;
        }
        if str_size > size {
            return INVALID_INDEX;
        }
        if from + str_size > size {
            from = size - str_size;
        }
        match cs {
            Case::Sensitive => {
                for i in (0..=from).rev() {
                    if compare_equal(&self.data, &s.data, i as usize, 0, str_size as usize) {
                        return i;
                    }
                }
            }
            Case::Insensitive => {
                for i in (0..=from).rev() {
                    if compare_equal_no_case(&self.data, &s.data, i as usize, 0, str_size as usize)
                    {
                        return i;
                    }
                }
            }
        }
        INVALID_INDEX
    }

    /// `FindIndex(char32_t chr, uint32_t from = 0, Case cs = Sensitive)`.
    pub fn find_index_char(&self, ch: char, from: u32, cs: Case) -> u32 {
        let size = self.size();
        if from >= size {
            return INVALID_INDEX;
        }
        match cs {
            Case::Sensitive => {
                for i in from..size {
                    if self.data[i as usize] == ch {
                        return i;
                    }
                }
            }
            Case::Insensitive => {
                for i in from..size {
                    if char_eq_no_case(self.data[i as usize], ch) {
                        return i;
                    }
                }
            }
        }
        INVALID_INDEX
    }

    /// `FindLastIndex(char32_t chr, uint32_t from = STRING_INVALID_INDEX, Case cs = Sensitive)`.
    pub fn find_last_index_char(&self, ch: char, from: u32, cs: Case) -> u32 {
        let size = self.size();
        if size == 0 {
            return INVALID_INDEX;
        }
        let mut from = from;
        if from >= size {
            from = size - 1;
        }
        match cs {
            Case::Sensitive => {
                for i in (0..=from).rev() {
                    if self.data[i as usize] == ch {
                        return i;
                    }
                }
            }
            Case::Insensitive => {
                for i in (0..=from).rev() {
                    if char_eq_no_case(self.data[i as usize], ch) {
                        return i;
                    }
                }
            }
        }
        INVALID_INDEX
    }

    /// `IndexValid(uint32_t index)`.
    pub fn index_valid(&self, index: u32) -> bool {
        index < self.size()
    }

    /// `ContainsStr(const String& str, Case cs = Sensitive)`.
    pub fn contains_str(&self, s: &String, cs: Case) -> bool {
        if s.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }
        self.find_index(s, 0, cs) != INVALID_INDEX
    }

    /// `ContainsAnyStr(const StringList& list, Case cs = Sensitive)`.
    pub fn contains_any_str(&self, list: &StringList, cs: Case) -> bool {
        list.iter().any(|s| self.contains_str(s, cs))
    }

    /// `ContainsAllStr(const StringList& list, Case cs = Sensitive)`.
    pub fn contains_all_str(&self, list: &StringList, cs: Case) -> bool {
        list.iter().all(|s| self.contains_str(s, cs))
    }

    /// `ContainsChar(char32_t chr, Case cs = Sensitive)`.
    pub fn contains_char(&self, ch: char, cs: Case) -> bool {
        if self.is_empty() {
            return false;
        }
        self.find_index_char(ch, 0, cs) != INVALID_INDEX
    }

    /// `ContainsAnyChar(const String& list, Case cs = Sensitive)`.
    pub fn contains_any_char(&self, list: &String, cs: Case) -> bool {
        list.data.iter().any(|&c| self.contains_char(c, cs))
    }

    /// `ContainsAllChar(const String& list, Case cs = Sensitive)`.
    pub fn contains_all_char(&self, list: &String, cs: Case) -> bool {
        list.data.iter().all(|&c| self.contains_char(c, cs))
    }

    /// `EndsWith(const String& str, Case cs = Sensitive)`.
    pub fn ends_with(&self, s: &String, cs: Case) -> bool {
        let str_size = s.size();
        if str_size == 0 {
            return true;
        }
        let size = self.size();
        if size == 0 || str_size > size {
            return false;
        }
        self.find_last_index(s, size - 1, cs) == size - str_size
    }

    /// `StartsWith(const String& str, Case cs = Sensitive)`.
    pub fn starts_with(&self, s: &String, cs: Case) -> bool {
        if s.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }
        self.find_index(s, 0, cs) == 0
    }

    /// `EndsWith(char32_t chr, Case cs = Sensitive)`.
    pub fn ends_with_char(&self, ch: char, cs: Case) -> bool {
        let size = self.size();
        if size == 0 {
            return false;
        }
        self.find_last_index_char(ch, size - 1, cs) == size - 1
    }

    /// `StartsWith(char32_t chr, Case cs = Sensitive)`.
    pub fn starts_with_char(&self, ch: char, cs: Case) -> bool {
        if self.is_empty() {
            return false;
        }
        self.find_index_char(ch, 0, cs) == 0
    }

    /// `DirectoryWithoutFilename()`.
    pub fn directory_without_filename(&self) -> String {
        match self.find_last_index_char('/', INVALID_INDEX, Case::Sensitive) {
            INVALID_INDEX => String::new(),
            index => self.left(index + 1),
        }
    }

    /// `FilenameWithoutDirectory()`.
    pub fn filename_without_directory(&self) -> String {
        match self.find_last_index_char('/', INVALID_INDEX, Case::Sensitive) {
            INVALID_INDEX => self.clone(),
            index => self.mid_from(index + 1),
        }
    }

    /// `FilenameWithoutExtension()`.
    pub fn filename_without_extension(&self) -> String {
        match self.find_last_index_char('.', INVALID_INDEX, Case::Sensitive) {
            INVALID_INDEX => self.clone(),
            index => self.left(index),
        }
    }

    /// `ExtensionWithoutFilename()`.
    pub fn extension_without_filename(&self) -> String {
        match self.find_last_index_char('.', INVALID_INDEX, Case::Sensitive) {
            INVALID_INDEX => String::new(),
            index => self.mid_from(index),
        }
    }

    /// `FixFilenameSlash()`.
    pub fn fix_filename_slash(&self) -> String {
        self.replace_char('\\', '/', Case::Sensitive)
    }

    /// `FixDirectorySlash()`.
    pub fn fix_directory_slash(&self) -> String {
        let mut s = self.replace_char('\\', '/', Case::Sensitive);
        if !s.ends_with_char('/', Case::Sensitive) {
            s += '/';
        }
        s
    }

    /// `Split(const String& sep, SplitType type = SplitNoEmptyParts, Case cs = Sensitive)`.
    pub fn split(&self, sep: &String, split_type: SplitType, cs: Case) -> StringList {
        let mut list = StringList::new();
        let mut start: u32 = 0;
        let extra: u32 = u32::from(sep.is_empty());
        let sep_size = sep.size();

        loop {
            let end = self.find_index(sep, start + extra, cs);
            if end == INVALID_INDEX {
                break;
            }
            if start != end || split_type == SplitType::WithEmptyParts {
                list.add(self.mid(start, end - start));
            }
            start = end + sep_size;
        }

        if start != self.size() || split_type == SplitType::WithEmptyParts {
            list.add(self.mid(start, self.size() - start));
        }
        list
    }

    /// `Split(char32_t sep, SplitType type = SplitNoEmptyParts, Case cs = Sensitive)`.
    pub fn split_char(&self, sep: char, split_type: SplitType, cs: Case) -> StringList {
        let mut list = StringList::new();
        let mut start: u32 = 0;

        loop {
            let end = self.find_index_char(sep, start, cs);
            if end == INVALID_INDEX {
                break;
            }
            if start != end || split_type == SplitType::WithEmptyParts {
                list.add(self.mid(start, end - start));
            }
            start = end + 1;
        }

        if start != self.size() || split_type == SplitType::WithEmptyParts {
            list.add(self.mid(start, self.size() - start));
        }
        list
    }

    /// `ToUint32(int base = 10)`. Delegates to [`String8::to_uint32`] via
    /// [`Self::utf8_str`], matching the C++'s own
    /// `sys_strtoui32(utf8_str().GetData(), ...)`.
    pub fn to_uint32(&self, base: i32) -> u32 {
        self.utf8_str().to_uint32(base)
    }

    /// `ToUint64(int base = 10)`.
    pub fn to_uint64(&self, base: i32) -> u64 {
        self.utf8_str().to_uint64(base)
    }

    /// `ToInt32(int base = 10)`.
    pub fn to_int32(&self, base: i32) -> i32 {
        self.utf8_str().to_int32(base)
    }

    /// `ToInt64(int base = 10)`.
    pub fn to_int64(&self, base: i32) -> i64 {
        self.utf8_str().to_int64(base)
    }

    /// `ToDouble()`.
    pub fn to_double(&self) -> f64 {
        self.utf8_str().to_double()
    }

    /// `ToFloat()`.
    pub fn to_float(&self) -> f32 {
        self.utf8_str().to_float()
    }

    /// `Hash()`: delegates to Kyty's `Core::hash` one-at-a-time hash over the
    /// little-endian bytes of each stored code point (see module doc
    /// comment).
    pub fn hash(&self) -> u32 {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for &c in &self.data {
            bytes.extend_from_slice(&(c as u32).to_le_bytes());
        }
        crate::hash::hash(&bytes)
    }

    /// `HexToBin()`. Uses `char::to_digit(16)` (std) in place of
    /// `Char::HexDigit`'s `CharUcd`-table lookup — see module doc comment.
    pub fn hex_to_bin(&self) -> ByteBuffer {
        let mut out = Vec::new();
        let mut p: u8 = 0;
        for (i, &c) in self.data.iter().enumerate() {
            let digit = c.to_digit(16).unwrap_or(0) as u8;
            if i % 2 == 0 {
                p = digit * 16;
            } else {
                p += digit;
                out.push(p);
                p = 0;
            }
        }
        ByteBuffer::from(out)
    }

    /// `static String HexFromBin(const ByteBuffer& bin)`.
    pub fn hex_from_bin(bin: &ByteBuffer) -> String {
        let mut r = String::new();
        for &b in bin.get_data() {
            r += &String::from(format!("{b:02X}"));
        }
        r
    }

    /// `EqualAscii(const char* ascii_str)`.
    pub fn equal_ascii(&self, ascii: &str) -> bool {
        self.data.iter().copied().eq(ascii.chars())
    }

    /// `EqualAsciiNoCase(const char* ascii_str)`.
    pub fn equal_ascii_no_case(&self, ascii: &str) -> bool {
        self.data.len() == ascii.chars().count()
            && self
                .data
                .iter()
                .zip(ascii.chars())
                .all(|(&a, b)| char_eq_no_case(a, b))
    }

    /// `IsAlpha()`. Uses `char::is_alphabetic` (std) in place of
    /// `Char::IsAlpha`'s `CharUcd`-table lookup — see module doc comment.
    pub fn is_alpha(&self) -> bool {
        self.data.iter().all(|c| c.is_alphabetic())
    }

    /// `SortChars()`.
    pub fn sort_chars(&self) -> String {
        if self.is_empty() {
            return String::new();
        }
        let mut data = self.data.clone();
        data.sort_unstable();
        String { data }
    }
}

impl std::ops::Index<u32> for String {
    type Output = char;
    fn index(&self, index: u32) -> &char {
        String::index(self, index)
    }
}
impl std::ops::IndexMut<u32> for String {
    fn index_mut(&mut self, index: u32) -> &mut char {
        String::index_mut(self, index)
    }
}

impl std::ops::AddAssign<&String> for String {
    /// `operator+=(const String& src)`.
    fn add_assign(&mut self, other: &String) {
        self.data.extend_from_slice(&other.data);
    }
}
impl std::ops::AddAssign<char> for String {
    /// `operator+=(char32_t ch)`.
    fn add_assign(&mut self, ch: char) {
        self.data.push(ch);
    }
}
impl std::ops::AddAssign<&str> for String {
    /// `operator+=(const char* utf8_str)`.
    fn add_assign(&mut self, s: &str) {
        self.data.extend(String::from_utf8_bytes(s.as_bytes()).data);
    }
}

impl std::ops::Add<&String> for String {
    type Output = String;
    /// `operator+(const String&, const String&)`.
    fn add(mut self, other: &String) -> String {
        self += other;
        self
    }
}
impl std::ops::Add<char> for String {
    type Output = String;
    /// `operator+(const String&, char32_t)`.
    fn add(mut self, ch: char) -> String {
        self += ch;
        self
    }
}
impl std::ops::Add<&str> for String {
    type Output = String;
    /// `operator+(const String&, const char*)`.
    fn add(mut self, s: &str) -> String {
        self += s;
        self
    }
}
impl std::ops::Add<String> for &str {
    type Output = String;
    /// `operator+(const char*, const String&)`.
    fn add(self, other: String) -> String {
        let mut r = String::from(self);
        r += &other;
        r
    }
}
impl std::ops::Add<String> for char {
    type Output = String;
    /// `operator+(char32_t, const String&)`.
    fn add(self, other: String) -> String {
        let mut r = String::from_char(self, 1);
        r += &other;
        r
    }
}

impl PartialEq<str> for String {
    fn eq(&self, other: &str) -> bool {
        self.equal_str(other)
    }
}
impl PartialEq<&str> for String {
    fn eq(&self, other: &&str) -> bool {
        self.equal_str(other)
    }
}

impl std::fmt::Display for String {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s: StdString = self.data.iter().collect();
        write!(f, "{s}")
    }
}

impl From<&str> for String {
    fn from(s: &str) -> Self {
        String::from_utf8_bytes(s.as_bytes())
    }
}
impl From<StdString> for String {
    fn from(s: StdString) -> Self {
        String::from_utf8_bytes(s.as_bytes())
    }
}
impl From<char> for String {
    fn from(ch: char) -> Self {
        String::from_char(ch, 1)
    }
}
impl From<&String8> for String {
    fn from(utf8: &String8) -> Self {
        String::from_string8(utf8)
    }
}

/// `Kyty::Core::StringList`: a thin wrapper over `Vec<String>` exposing the
/// original's `Vector<String>`-inherited surface that `String`'s own methods
/// need, plus `StringList`'s own additions (`Contains`/`Concat`/`Equal`/
/// `EqualNoCase`), mirroring `StringList8`'s own approach.
#[derive(Debug, Clone, Default)]
pub struct StringList {
    items: Vec<String>,
}

impl StringList {
    /// `StringList()` (via `using Vector<String>::Vector`).
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// `Size()`.
    pub fn size(&self) -> u32 {
        self.items.len() as u32
    }

    /// `IsEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// `Add(const T&)`.
    pub fn add(&mut self, s: String) {
        self.items.push(s);
    }

    /// `At(uint32_t index) const`.
    pub fn at(&self, index: u32) -> &String {
        &self.items[index as usize]
    }

    /// `begin() const`/`end() const` (const iteration).
    pub fn iter(&self) -> std::slice::Iter<'_, String> {
        self.items.iter()
    }

    /// `StringList::Contains(const String& str, Case cs = Sensitive)`.
    pub fn contains(&self, s: &String, cs: Case) -> bool {
        self.items.iter().any(|item| item.contains_str(s, cs))
    }

    /// `StringList::Concat(const String& str)`: joins all elements with
    /// `str` as separator.
    pub fn concat(&self, sep: &String) -> String {
        let mut r = String::new();
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                r += sep;
            }
            r += item;
        }
        r
    }

    /// `StringList::Concat(char32_t chr)`.
    pub fn concat_char(&self, chr: char) -> String {
        let mut r = String::new();
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                r += chr;
            }
            r += item;
        }
        r
    }

    /// `StringList::Equal(const StringList& str)`.
    pub fn equal(&self, other: &StringList) -> bool {
        self.items.len() == other.items.len()
            && self
                .items
                .iter()
                .zip(other.items.iter())
                .all(|(a, b)| a.equal(b))
    }

    /// `StringList::EqualNoCase(const StringList& str)`.
    pub fn equal_no_case(&self, other: &StringList) -> bool {
        self.items.len() == other.items.len()
            && self
                .items
                .iter()
                .zip(other.items.iter())
                .all(|(a, b)| a.equal_no_case(b))
    }
}

impl PartialEq for StringList {
    fn eq(&self, other: &Self) -> bool {
        self.equal(other)
    }
}
impl Eq for StringList {}

impl std::ops::Index<u32> for StringList {
    type Output = String;
    fn index(&self, index: u32) -> &String {
        &self.items[index as usize]
    }
}

impl From<Vec<String>> for StringList {
    fn from(items: Vec<String>) -> Self {
        Self { items }
    }
}

impl IntoIterator for StringList {
    type Item = String;
    type IntoIter = std::vec::IntoIter<String>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}
impl<'a> IntoIterator for &'a StringList {
    type Item = &'a String;
    type IntoIter = std::slice::Iter<'a, String>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let s = String::new();
        assert_eq!(s.size(), 0);
        assert!(s.is_empty());
        assert!(!s.is_invalid());
    }

    #[test]
    fn from_char_repeat() {
        let s = String::from_char('x', 3);
        assert_eq!(s.size(), 3);
        assert_eq!(s.to_string(), "xxx");
        assert!(String::from_char('x', 0).is_empty());
    }

    #[test]
    fn from_str_round_trip() {
        let s = String::from("héllo");
        assert_eq!(s.size(), 5); // code-point count, not byte count
        assert_eq!(s.to_string(), "héllo");
        assert_eq!(s.utf8_str().as_bytes(), "héllo".as_bytes());
    }

    #[test]
    fn from_utf8_strips_bom() {
        let bytes = "\u{FEFF}abc".as_bytes();
        let s = String::from_utf8_bytes(bytes);
        assert_eq!(s.to_string(), "abc");
    }

    #[test]
    fn utf16_round_trip_with_surrogate_pair() {
        // U+1F600 (an emoji) requires a UTF-16 surrogate pair.
        let s = String::from("😀");
        let units = s.utf16_str();
        assert_eq!(units.len(), 2);
        let back = String::from_utf16(&units);
        assert_eq!(back, s);
    }

    #[test]
    fn utf32_round_trip_and_bom_skip() {
        let cps = vec![0xFEFFu32, 'a' as u32, 'b' as u32];
        let s = String::from_utf32(&cps);
        assert_eq!(s.to_string(), "ab");
        assert_eq!(s.utf32_str(), vec!['a' as u32, 'b' as u32]);
    }

    #[test]
    fn cp866_round_trip() {
        let s = String::from("АБВ"); // Cyrillic
        let bytes = s.cp866_str();
        assert_eq!(bytes, vec![128, 129, 130]);
        let back = String::from_cp866(&bytes);
        assert_eq!(back, s);
    }

    #[test]
    fn cp1251_round_trip() {
        let s = String::from("АБВ");
        let bytes = s.cp1251_str();
        assert_eq!(bytes, vec![192, 193, 194]);
        let back = String::from_cp1251(&bytes);
        assert_eq!(back, s);
    }

    #[test]
    fn index_and_at() {
        let mut s = String::from("abc");
        assert_eq!(s[0], 'a');
        assert_eq!(*s.at(2), 'c');
        s[1] = 'Z';
        assert_eq!(s.to_string(), "aZc");
    }

    #[test]
    fn equal_variants() {
        let a = String::from("abc");
        assert!(a.equal(&String::from("abc")));
        assert!(!a.equal(&String::from("abd")));
        assert!(String::from("x").equal_char('x'));
        assert!(a.equal_str("abc"));
        assert_eq!(a, String::from("abc"));
        assert_eq!(a, "abc");

        assert!(a.equal_no_case(&String::from("ABC")));
        assert!(String::from("K").equal_no_case_char('k'));
        assert!(a.equal_no_case_str("ABC"));
    }

    #[test]
    fn add_assign_and_concatenation() {
        let mut s = String::from("foo");
        s += &String::from("bar");
        s += '!';
        s += "?";
        assert_eq!(s.to_string(), "foobar!?");

        let combined = String::from("a") + &String::from("b") + 'c';
        assert_eq!(combined.to_string(), "abc");

        let prefixed = "pre-" + String::from("fix");
        assert_eq!(prefixed.to_string(), "pre-fix");

        let charred = 'x' + String::from("yz");
        assert_eq!(charred.to_string(), "xyz");
    }

    #[test]
    fn mid_left_right() {
        let s = String::from("hello world");
        assert_eq!(s.mid(0, 5).to_string(), "hello");
        assert_eq!(s.mid(6, 100).to_string(), "world"); // count clamped
        assert!(s.mid(100, 1).is_empty()); // first >= size
        assert_eq!(s.mid_from(6).to_string(), "world");
        assert_eq!(s.left(3).to_string(), "hel");
        assert_eq!(s.right(3).to_string(), "rld");
        assert_eq!(s.right(100).to_string(), "hello world");
    }

    #[test]
    fn to_upper_and_lower() {
        let s = String::from("Hello");
        assert_eq!(s.to_upper().to_string(), "HELLO");
        assert_eq!(s.to_lower().to_string(), "hello");
    }

    #[test]
    fn trim_variants() {
        let s = String::from("  hi there  ");
        assert_eq!(s.trim_left().to_string(), "hi there  ");
        assert_eq!(s.trim_right().to_string(), "  hi there");
        assert_eq!(s.trim().to_string(), "hi there");
        assert!(String::from("   ").trim().is_empty());
    }

    #[test]
    fn simplify_collapses_whitespace_runs() {
        let s = String::from("  a\t\tb   c  ");
        assert_eq!(s.simplify().to_string(), "a\tb c");
    }

    #[test]
    fn replace_char_and_str_with_case() {
        let s = String::from("a-B-c");
        assert_eq!(
            s.replace_char('-', '_', Case::Sensitive).to_string(),
            "a_B_c"
        );
        assert_eq!(
            s.replace_char('b', '_', Case::Insensitive).to_string(),
            "a-_-c"
        );

        let s2 = String::from("foo bar foo");
        let replaced = s2.replace_str(&String::from("foo"), &String::from("baz"), Case::Sensitive);
        assert_eq!(replaced.to_string(), "baz bar baz");
    }

    #[test]
    fn remove_at_char_str_last_first() {
        let s = String::from("hello");
        assert_eq!(s.remove_at(1, 2).to_string(), "hlo");
        assert_eq!(s.remove_char('l', Case::Sensitive).to_string(), "heo");
        assert_eq!(s.remove_last(2).to_string(), "hel");
        assert_eq!(s.remove_first(2).to_string(), "llo");

        let s2 = String::from("foo bar foo");
        assert_eq!(
            s2.remove_str(&String::from("foo"), Case::Sensitive)
                .to_string(),
            " bar "
        );
    }

    #[test]
    fn insert_at_middle() {
        let s = String::from("helloworld");
        let inserted = s.insert_at(5, &String::from(" "));
        assert_eq!(inserted.to_string(), "hello world");
    }

    #[test]
    fn find_index_variants_and_case_insensitive() {
        let s = String::from("abcABCabc");
        assert_eq!(s.find_index(&String::from("bc"), 0, Case::Sensitive), 1);
        assert_eq!(s.find_index(&String::from("BC"), 0, Case::Insensitive), 1);
        assert_eq!(
            s.find_index(&String::from("zz"), 0, Case::Sensitive),
            INVALID_INDEX
        );
        assert_eq!(
            s.find_last_index(&String::from("abc"), INVALID_INDEX, Case::Sensitive),
            6
        );
        assert_eq!(s.find_index_char('C', 0, Case::Insensitive), 2);
        assert_eq!(
            s.find_last_index_char('a', INVALID_INDEX, Case::Sensitive),
            6
        );
    }

    #[test]
    fn contains_starts_ends_with() {
        let s = String::from("Hello World");
        assert!(s.contains_str(&String::from("wor"), Case::Insensitive));
        assert!(!s.contains_str(&String::from("wor"), Case::Sensitive));
        assert!(s.contains_char('W', Case::Sensitive));
        assert!(s.starts_with(&String::from("hello"), Case::Insensitive));
        assert!(s.ends_with(&String::from("world"), Case::Insensitive));
        assert!(s.starts_with_char('H', Case::Sensitive));
        assert!(s.ends_with_char('d', Case::Sensitive));

        let any_list = StringList::from(vec![String::from("xyz"), String::from("wor")]);
        assert!(s.contains_any_str(&any_list, Case::Insensitive));
        let all_list = StringList::from(vec![String::from("hello"), String::from("world")]);
        assert!(s.contains_all_str(&all_list, Case::Insensitive));

        assert!(s.contains_any_char(&String::from("xyzW"), Case::Sensitive));
        assert!(!s.contains_all_char(&String::from("xyz"), Case::Sensitive));
    }

    #[test]
    fn path_helpers() {
        let path = String::from("/usr/local/bin.exe");
        assert_eq!(path.directory_without_filename().to_string(), "/usr/local/");
        assert_eq!(path.filename_without_directory().to_string(), "bin.exe");
        assert_eq!(
            path.filename_without_extension().to_string(),
            "/usr/local/bin"
        );
        assert_eq!(path.extension_without_filename().to_string(), ".exe");

        let win = String::from(r"a\b\c");
        assert_eq!(win.fix_filename_slash().to_string(), "a/b/c");
        let dir = String::from(r"a\b");
        assert_eq!(dir.fix_directory_slash().to_string(), "a/b/");
    }

    #[test]
    fn split_str_and_char_with_empty_parts() {
        let s = String::from("a,,b,c");
        let no_empty = s.split(
            &String::from(","),
            SplitType::SplitNoEmptyParts,
            Case::Sensitive,
        );
        assert_eq!(no_empty.size(), 3);
        let with_empty = s.split(
            &String::from(","),
            SplitType::WithEmptyParts,
            Case::Sensitive,
        );
        assert_eq!(with_empty.size(), 4);

        let s2 = String::from("a/b//c");
        let parts = s2.split_char('/', SplitType::SplitNoEmptyParts, Case::Sensitive);
        assert_eq!(parts.size(), 3);
        assert_eq!(parts.at(2).to_string(), "c");
    }

    #[test]
    fn safe_lua_and_csv() {
        let s = String::from(r"it's a \test\");
        assert_eq!(s.safe_lua().to_string(), r"it\'s a \\test\\");

        assert_eq!(String::from("plain").safe_csv().to_string(), "plain");
        assert_eq!(
            String::from("has \"quote\"").safe_csv().to_string(),
            "\"has \"\"quote\"\"\""
        );
        assert_eq!(
            String::from("=SUM(A1)").safe_csv().to_string(),
            "\" =SUM(A1)\""
        );
    }

    #[test]
    fn numeric_conversions() {
        assert_eq!(String::from("42").to_uint32(10), 42);
        assert_eq!(String::from("-7").to_int32(10), -7);
        assert_eq!(String::from("ff").to_uint32(16), 255);
        assert!((String::from("3.5").to_double() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn hash_is_deterministic_and_content_based() {
        let a = String::from("kyty");
        let b = String::from("kyty");
        let c = String::from("other");
        assert_eq!(a.hash(), b.hash());
        assert_ne!(a.hash(), c.hash());
    }

    #[test]
    fn hex_to_bin_and_back() {
        let bin = String::from("4B79747921").hex_to_bin(); // "Kyty!"
        assert_eq!(bin.get_data(), b"Kyty!");
        let back = String::hex_from_bin(&bin);
        assert_eq!(back.to_string(), "4B79747921");
    }

    #[test]
    fn equal_ascii_variants() {
        let s = String::from("Hello");
        assert!(s.equal_ascii("Hello"));
        assert!(!s.equal_ascii("hello"));
        assert!(s.equal_ascii_no_case("HELLO"));
    }

    #[test]
    fn is_alpha_and_sort_chars() {
        assert!(String::from("Hello").is_alpha());
        assert!(!String::from("Hello1").is_alpha());
        assert_eq!(String::from("dcba").sort_chars().to_string(), "abcd");
        assert!(String::new().sort_chars().is_empty());
    }

    #[test]
    fn string_list_contains_concat_equal() {
        let list = StringList::from(vec![String::from("foo"), String::from("bar")]);
        assert!(list.contains(&String::from("oo"), Case::Sensitive));
        assert_eq!(list.concat(&String::from(", ")).to_string(), "foo, bar");
        assert_eq!(list.concat_char('-').to_string(), "foo-bar");

        let same = StringList::from(vec![String::from("foo"), String::from("bar")]);
        let upper = StringList::from(vec![String::from("FOO"), String::from("BAR")]);
        assert_eq!(list, same);
        assert!(!list.equal(&upper));
        assert!(list.equal_no_case(&upper));
    }

    #[test]
    fn from_string8_interop() {
        let s8 = String8::from("kyty");
        let s = String::from(&s8);
        assert_eq!(s.to_string(), "kyty");
        assert_eq!(s.utf8_str(), s8);
    }
}
