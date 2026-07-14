//! Port of Kyty's `Kyty::Core::String8` / `Kyty::Core::StringList8`
//! (`reference/kyty/source/include/Kyty/Core/String8.h`,
//! `reference/kyty/source/lib/Core/src/String8.cpp`).
//!
//! # Std mapping
//!
//! In C++, `String8` is a ref-counted, copy-on-write wrapper around
//! `SimpleArray<char>` (Kyty's own hand-rolled dynamic array): `m_data` is a
//! shared, manually refcounted pointer, mutating accessors call
//! `CopyOnWrite`/`CopyPtr` to deep-copy lazily on first write, and the backing
//! array always carries one extra trailing `'\0'` byte that every method has
//! to account for by hand (`Size()` returns `m_data->Size() - 1`, `RemoveAt`
//! re-appends a `'\0'` if it got removed, etc).
//!
//! `String8` here is a **byte string**, not a Unicode string: Kyty never
//! assumed its `char` content was valid UTF-8 (it's used for filenames, CSV
//! fields, Lua-escaped text, arbitrary narrow-encoded data, ...), so unlike
//! the (separate, not-yet-ported) `Kyty::Core::String` — which is a proper
//! Unicode string — the faithful std mapping for `String8` is `Vec<u8>`, per
//! the port conventions' explicit "byte content that must round-trip
//! losslessly maps to `Vec<u8>`" guidance. `String8` below is a thin wrapper
//! over `Vec<u8>`:
//! - the refcounted/CoW `SimpleArray<char>*` dance is dropped entirely —
//!   `Vec<u8>` + `#[derive(Clone)]` (eager deep copy) is observably
//!   equivalent (same final contents, same mutation semantics) without any
//!   unsafe code, manual refcounting, or raw pointers;
//! - the trailing `'\0'` bookkeeping is dropped entirely — `Vec<u8>` already
//!   knows its own length, so there is nothing to keep in sync;
//! - Kyty's `char` (a single narrow byte) is ported as plain `u8` throughout
//!   this API (`Equal(char)`, `operator[]`, `ReplaceChar`, ...).
//!
//! `Char::IsSpace` (`std::isspace` under the "C" locale, the only locale
//! Kyty relies on) is ported as `u8::is_ascii_whitespace`. The two agree on
//! space/tab/LF/FF/CR; `is_ascii_whitespace` additionally excludes
//! `'\x0B'` (vertical tab), which `isspace` includes. This is an accepted,
//! documented divergence (matches the port conventions' carve-out for
//! locale/table-derived helpers) — vertical tab is essentially never present
//! in real-world text processed by this API.
//!
//! `Printf`/`FromPrintf` (C varargs `printf` over a `va_list`) have no
//! faithful *safe* Rust equivalent (`va_list` is not portable/safe stable
//! Rust, and pulling in a printf-compatible crate is unwarranted for this
//! module) and are intentionally **not** ported. Callers should build the
//! formatted text with Rust's `format!`/`write!` and construct a `String8`
//! from the result (`String8::from(format!(...))`).
//!
//! `c_str()`/`GetData()` (raw `char*`/`const char*` pointer accessors used in
//! C++ to interoperate with C APIs expecting a NUL-terminated buffer) are not
//! ported as pointer accessors — per the port conventions, Rust ownership
//! makes that C-interop scaffolding unnecessary here. Use [`String8::as_bytes`]
//! (a plain, safe `&[u8]` view) instead; if a NUL-terminated `CStr` is ever
//! needed for FFI, build one explicitly with `std::ffi::CString`.
//!
//! `ToUint32`/`ToInt32`/`ToUint64`/`ToInt64`/`ToDouble`/`ToFloat` port Kyty's
//! `sys_strtoi32`/`sys_strtoui64`/`sys_strtod`/etc, which are thin wrappers
//! over C's `strtol`/`strtoul`/`strtoll`/`strtoull`/`strtod`/`strtof`. Rust's
//! `str::parse` requires the *entire* string to be a valid number and has no
//! "parse a leading numeric prefix, ignore trailing garbage" mode, so these
//! are ported as small manual prefix parsers (below) that replicate the C
//! `strtoX` contract (skip leading ASCII whitespace, optional sign, parse a
//! leading run of digits in the given base, `0` if no digits found, magnitude
//! clamped/wrapped to the target width) without pulling in `libc`. `base = 0`
//! reproduces `strtol`'s auto-detection (`0x`/`0X` prefix -> hex, leading `0`
//! -> octal, else decimal). Recognizing `inf`/`nan` literals (which
//! `strtod`/`strtof` do) is not implemented — an accepted, documented
//! divergence for an edge case that Kyty's own callers do not rely on.
//!
//! `StringList8` (C++: `class StringList8 : public Vector<String8>`, plus
//! `Contains`/`Concat`/`Equal`) is ported as a thin wrapper over `Vec<String8>`
//! exposing just the members `String8`'s own methods need (`Split` returns
//! one, `ContainsAnyStr`/`ContainsAllStr` take one) plus the `StringList8`-own
//! methods, rather than pulling in the already-ported generic `Vector<T>` —
//! there is no behavior here that needs the generic type.

use crate::hash;

/// `STRING8_INVALID_INDEX`: sentinel returned by the `Find*` family when no
/// match exists.
pub const INVALID_INDEX: u32 = u32::MAX;

/// `String8::SplitType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitType {
    WithEmptyParts,
    SplitNoEmptyParts,
}

/// Kyty's `Kyty::Core::String8`: a thin wrapper over `Vec<u8>` exposing the
/// original's method names/semantics. See the module doc comment for the
/// full std-mapping rationale.
#[derive(Debug, Clone, Default)]
pub struct String8 {
    data: Vec<u8>,
}

// ---------------------------------------------------------------------
// strtoX-style prefix parsing helpers backing To{U,I}nt{32,64}/ToDouble/ToFloat.
// ---------------------------------------------------------------------

/// Parses a `strtol`/`strtoul`-style leading numeric prefix: optional ASCII
/// whitespace, optional sign, then digits in `base` (or auto-detected when
/// `base == 0`, matching C's `strtol(..., 0)`). Returns `(negative, magnitude)`;
/// `magnitude` saturates at `u128::MAX` rather than wrapping, so callers can
/// clamp/truncate to their target width afterwards. `(false, 0)` if no digits
/// were found (matching `strtoX`'s "0, *endptr == nptr" case).
fn parse_int_prefix(bytes: &[u8], mut base: u32) -> (bool, u128) {
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

    if base == 16 || base == 0 {
        if i + 1 < len && bytes[i] == b'0' && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
            i += 2;
            base = 16;
        } else if base == 0 {
            base = if i < len && bytes[i] == b'0' { 8 } else { 10 };
        }
    }

    let mut magnitude: u128 = 0;
    let mut any_digits = false;
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
    }

    if !any_digits {
        return (false, 0);
    }
    (negative, magnitude)
}

/// Parses a `strtod`/`strtof`-style leading floating-point prefix (optional
/// whitespace, optional sign, digits, optional `.digits`, optional
/// `[eE][+-]digits`). Returns `0.0` if no valid numeric prefix is present.
/// `inf`/`nan` literals are not recognized (see module doc comment).
fn parse_float_prefix(bytes: &[u8]) -> f64 {
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
        return 0.0;
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

    std::str::from_utf8(&bytes[start..end])
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
}

impl String8 {
    /// `String8()`: empty string.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// `String8(char ch, uint32_t repeat = 1)`. Rust has no default
    /// arguments; pass `1` explicitly for the C++ default.
    pub fn from_char(ch: u8, repeat: u32) -> Self {
        Self {
            data: vec![ch; repeat as usize],
        }
    }

    /// `explicit String8(const char* array, uint32_t size)`.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data: bytes.to_vec(),
        }
    }

    /// `Size()`: number of bytes (excludes the internal `'\0'` bookkeeping
    /// C++ had to do by hand; see module doc comment).
    pub fn size(&self) -> u32 {
        self.data.len() as u32
    }

    /// `IsEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `IsInvalid()`. In C++ this reports whether `m_data == nullptr`, a
    /// state only reachable by using a moved-from `String8` (`String8&&`
    /// sets the source's `m_data` to `nullptr`, and the object remains
    /// nominally alive/usable afterwards). Rust's ownership model makes that
    /// state unreachable through safe code — a moved-from value simply
    /// cannot be named again, the borrow checker rejects it at compile time
    /// — so this always returns `false`. Kept for API parity.
    pub fn is_invalid(&self) -> bool {
        false
    }

    /// `Clear()`.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// `At(uint32_t index) const`.
    pub fn at(&self, index: u32) -> &u8 {
        &self.data[index as usize]
    }

    /// `GetData()` (mutable view).
    pub fn get_data(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// `GetData() const` / `GetDataConst()`.
    pub fn get_data_const(&self) -> &[u8] {
        &self.data
    }

    /// Safe, allocation-free byte view (replaces C++'s raw-pointer
    /// `c_str()`/`GetDataConst()` for Rust callers; see module doc comment).
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// `Equal(const String8& src)`.
    pub fn equal(&self, other: &String8) -> bool {
        self.data == other.data
    }

    /// `Equal(char ch)`: true iff `self` is the single-byte string `ch`.
    pub fn equal_char(&self, ch: u8) -> bool {
        self.data.len() == 1 && self.data[0] == ch
    }

    /// `Equal(const char* utf8_str)`.
    pub fn equal_str(&self, s: &str) -> bool {
        self.data == s.as_bytes()
    }

    /// `Mid(uint32_t first, uint32_t count)`.
    pub fn mid(&self, first: u32, count: u32) -> String8 {
        let size = self.size();
        if first >= size {
            return String8::new();
        }
        let mut count = count;
        if u64::from(first) + u64::from(count) > u64::from(size) {
            count = size - first;
        }
        if first == 0 && count == size {
            return self.clone();
        }
        String8::from_bytes(&self.data[first as usize..(first + count) as usize])
    }

    /// `Mid(uint32_t first)` (single-argument overload).
    pub fn mid_from(&self, first: u32) -> String8 {
        self.mid(first, self.size().saturating_sub(first))
    }

    /// `Left(uint32_t count)`.
    pub fn left(&self, count: u32) -> String8 {
        self.mid(0, count)
    }

    /// `Right(uint32_t count)`.
    pub fn right(&self, count: u32) -> String8 {
        let size = self.size();
        if count >= size {
            return self.clone();
        }
        self.mid(size - count, count)
    }

    /// `TrimRight()`.
    pub fn trim_right(&self) -> String8 {
        let size = self.size();
        for i in 0..size {
            if !self.data[(size - i - 1) as usize].is_ascii_whitespace() {
                return self.mid(0, size - i);
            }
        }
        String8::new()
    }

    /// `TrimLeft()`.
    pub fn trim_left(&self) -> String8 {
        let size = self.size();
        for i in 0..size {
            if !self.data[i as usize].is_ascii_whitespace() {
                return self.mid(i, size - i);
            }
        }
        String8::new()
    }

    /// `Trim()`.
    pub fn trim(&self) -> String8 {
        let size = self.size();
        let mut left_pos = size;
        let mut count = 0u32;
        for i in 0..size {
            if !self.data[i as usize].is_ascii_whitespace() {
                left_pos = i;
                break;
            }
        }
        for i in 0..size.saturating_sub(left_pos) {
            if !self.data[(size - i - 1) as usize].is_ascii_whitespace() {
                count = size - left_pos - i;
                break;
            }
        }
        self.mid(left_pos, count)
    }

    /// `Simplify()`: collapses runs of whitespace, keeping the byte the run
    /// started with (not necessarily `' '`), drops leading whitespace
    /// entirely, and trims trailing whitespace — exactly Kyty's algorithm.
    pub fn simplify(&self) -> String8 {
        let mut data = Vec::with_capacity(self.data.len());
        let mut prev_space = true;
        for &c in &self.data {
            if c.is_ascii_whitespace() {
                if prev_space {
                    continue;
                }
                prev_space = true;
            } else {
                prev_space = false;
            }
            data.push(c);
        }
        String8 { data }.trim_right()
    }

    /// `ReplaceChar(char old_char, char new_char)`.
    pub fn replace_char(&self, old_char: u8, new_char: u8) -> String8 {
        let data = self
            .data
            .iter()
            .map(|&b| if b == old_char { new_char } else { b })
            .collect();
        String8 { data }
    }

    /// `ReplaceStr(const String8& old_str, const String8& new_str)`.
    pub fn replace_str(&self, old_str: &String8, new_str: &String8) -> String8 {
        let mut result = String8::new();
        let mut start: u32 = 0;
        let extra: u32 = u32::from(old_str.is_empty());
        let sep_size = old_str.size();

        loop {
            let end = self.find_index(old_str, start + extra);
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
    pub fn remove_at(&self, index: u32, count: u32) -> String8 {
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
        String8 { data }
    }

    /// `RemoveChar(char ch)`.
    pub fn remove_char(&self, ch: u8) -> String8 {
        let data: Vec<u8> = self.data.iter().copied().filter(|&b| b != ch).collect();
        String8 { data }
    }

    /// `RemoveStr(const String8& str)`. Kyty's own implementation is
    /// structurally identical to `ReplaceStr` with an empty replacement (the
    /// loop bodies match exactly, minus the `str += new_str` append), so
    /// this is ported as exactly that call rather than duplicating the loop.
    pub fn remove_str(&self, str_: &String8) -> String8 {
        self.replace_str(str_, &String8::new())
    }

    /// `RemoveLast(uint32_t num)`.
    pub fn remove_last(&self, num: u32) -> String8 {
        let size = self.size();
        if num >= size {
            return String8::new();
        }
        self.left(size - num)
    }

    /// `RemoveFirst(uint32_t num)`.
    pub fn remove_first(&self, num: u32) -> String8 {
        let size = self.size();
        if num >= size {
            return String8::new();
        }
        self.right(size - num)
    }

    /// `InsertAt(uint32_t index, const String8& str)`.
    pub fn insert_at(&self, index: u32, str_: &String8) -> String8 {
        let size = self.size();
        let mut result = self.mid(0, index);
        result += str_;
        result += &self.mid(index, size.saturating_sub(index));
        result
    }

    /// `SafeLua()`.
    pub fn safe_lua(&self) -> String8 {
        self.replace_str(&String8::from("\\"), &String8::from("\\\\"))
            .replace_str(&String8::from("'"), &String8::from("\\'"))
    }

    /// `SafeCsv()`.
    pub fn safe_csv(&self) -> String8 {
        let add_space = self.starts_with_char(b'+')
            || self.starts_with_char(b'=')
            || self.starts_with_char(b'-');

        let needs_quoting = self.contains_char(b'"')
            || self.contains_char(b';')
            || self.contains_char(b'+')
            || self.contains_char(b'=')
            || self.contains_char(b'-');

        if needs_quoting {
            let mut r = String8::from("\"");
            if add_space {
                r += b' ';
            }
            r += &self.replace_str(&String8::from("\""), &String8::from("\"\""));
            r += b'"';
            return r;
        }
        self.clone()
    }

    /// `FindIndex(const String8& str, uint32_t from = 0)`.
    pub fn find_index(&self, s: &String8, from: u32) -> u32 {
        let size = self.size();
        if from >= size {
            return INVALID_INDEX;
        }
        let str_size = s.size();
        if str_size == 0 {
            return from;
        }
        if str_size > size {
            return INVALID_INDEX;
        }
        let last_start = size - str_size;
        let mut i = from;
        while i <= last_start {
            if self.data[i as usize..(i + str_size) as usize] == s.data[..] {
                return i;
            }
            i += 1;
        }
        INVALID_INDEX
    }

    /// `FindLastIndex(const String8& str, uint32_t from = STRING8_INVALID_INDEX)`.
    ///
    /// Diverges from the original in one edge case: if `str` is longer than
    /// `self`, C++ computes `size - str_size` as an unsigned underflow and
    /// walks off the end of the buffer (undefined behavior); this always
    /// safely returns [`INVALID_INDEX`] instead, which is the only sensible
    /// answer ("longer string can never be found").
    pub fn find_last_index(&self, s: &String8, from: u32) -> u32 {
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
        for i in (0..=from).rev() {
            if self.data[i as usize..(i + str_size) as usize] == s.data[..] {
                return i;
            }
        }
        INVALID_INDEX
    }

    /// `FindIndex(char chr, uint32_t from = 0)`.
    pub fn find_index_char(&self, ch: u8, from: u32) -> u32 {
        let size = self.size();
        if from >= size {
            return INVALID_INDEX;
        }
        for i in from..size {
            if self.data[i as usize] == ch {
                return i;
            }
        }
        INVALID_INDEX
    }

    /// `FindLastIndex(char chr, uint32_t from = STRING8_INVALID_INDEX)`.
    pub fn find_last_index_char(&self, ch: u8, from: u32) -> u32 {
        let size = self.size();
        if size == 0 {
            return INVALID_INDEX;
        }
        let mut from = from;
        if from >= size {
            from = size - 1;
        }
        for i in (0..=from).rev() {
            if self.data[i as usize] == ch {
                return i;
            }
        }
        INVALID_INDEX
    }

    /// `IndexValid(uint32_t index)`.
    pub fn index_valid(&self, index: u32) -> bool {
        index < self.size()
    }

    /// `ContainsStr(const String8& str)`.
    pub fn contains_str(&self, s: &String8) -> bool {
        if s.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }
        self.find_index(s, 0) != INVALID_INDEX
    }

    /// `ContainsAnyStr(const StringList8& list)`.
    pub fn contains_any_str(&self, list: &StringList8) -> bool {
        list.iter().any(|s| self.contains_str(s))
    }

    /// `ContainsAllStr(const StringList8& list)`.
    pub fn contains_all_str(&self, list: &StringList8) -> bool {
        list.iter().all(|s| self.contains_str(s))
    }

    /// `ContainsChar(char chr)`.
    pub fn contains_char(&self, ch: u8) -> bool {
        if self.is_empty() {
            return false;
        }
        self.find_index_char(ch, 0) != INVALID_INDEX
    }

    /// `ContainsAnyChar(const String8& list)`.
    pub fn contains_any_char(&self, list: &String8) -> bool {
        list.data.iter().any(|&c| self.contains_char(c))
    }

    /// `ContainsAllChar(const String8& list)`.
    pub fn contains_all_char(&self, list: &String8) -> bool {
        list.data.iter().all(|&c| self.contains_char(c))
    }

    /// `EndsWith(const String8& str)`.
    pub fn ends_with(&self, s: &String8) -> bool {
        let str_size = s.size();
        if str_size == 0 {
            return true;
        }
        let size = self.size();
        if size == 0 || str_size > size {
            return false;
        }
        self.find_last_index(s, size - 1) == size - str_size
    }

    /// `StartsWith(const String8& str)`.
    pub fn starts_with(&self, s: &String8) -> bool {
        if s.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }
        self.find_index(s, 0) == 0
    }

    /// `EndsWith(char chr)`.
    pub fn ends_with_char(&self, ch: u8) -> bool {
        let size = self.size();
        if size == 0 {
            return false;
        }
        self.find_last_index_char(ch, size - 1) == size - 1
    }

    /// `StartsWith(char chr)`.
    pub fn starts_with_char(&self, ch: u8) -> bool {
        if self.is_empty() {
            return false;
        }
        self.find_index_char(ch, 0) == 0
    }

    /// `DirectoryWithoutFilename()`.
    pub fn directory_without_filename(&self) -> String8 {
        match self.find_last_index_char(b'/', INVALID_INDEX) {
            INVALID_INDEX => String8::new(),
            index => self.left(index + 1),
        }
    }

    /// `FilenameWithoutDirectory()`.
    pub fn filename_without_directory(&self) -> String8 {
        match self.find_last_index_char(b'/', INVALID_INDEX) {
            INVALID_INDEX => self.clone(),
            index => self.mid_from(index + 1),
        }
    }

    /// `FilenameWithoutExtension()`.
    pub fn filename_without_extension(&self) -> String8 {
        match self.find_last_index_char(b'.', INVALID_INDEX) {
            INVALID_INDEX => self.clone(),
            index => self.left(index),
        }
    }

    /// `ExtensionWithoutFilename()`.
    pub fn extension_without_filename(&self) -> String8 {
        match self.find_last_index_char(b'.', INVALID_INDEX) {
            INVALID_INDEX => String8::new(),
            index => self.mid_from(index),
        }
    }

    /// `FixFilenameSlash()`.
    pub fn fix_filename_slash(&self) -> String8 {
        self.replace_char(b'\\', b'/')
    }

    /// `FixDirectorySlash()`.
    pub fn fix_directory_slash(&self) -> String8 {
        let mut s = self.replace_char(b'\\', b'/');
        if !s.ends_with_char(b'/') {
            s += b'/';
        }
        s
    }

    /// `Split(const String8& sep, SplitType type = SplitNoEmptyParts)`.
    pub fn split(&self, sep: &String8, split_type: SplitType) -> StringList8 {
        let mut list = StringList8::new();
        let mut start: u32 = 0;
        let extra: u32 = u32::from(sep.is_empty());
        let sep_size = sep.size();

        loop {
            let end = self.find_index(sep, start + extra);
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

    /// `Split(char sep, SplitType type = SplitNoEmptyParts)`.
    pub fn split_char(&self, sep: u8, split_type: SplitType) -> StringList8 {
        let mut list = StringList8::new();
        let mut start: u32 = 0;

        loop {
            let end = self.find_index_char(sep, start);
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

    /// `ToUint32(int base = 10)`.
    pub fn to_uint32(&self, base: i32) -> u32 {
        let (negative, magnitude) = parse_int_prefix(&self.data, base as u32);
        let val = magnitude.min(u128::from(u32::MAX)) as u32;
        if negative { val.wrapping_neg() } else { val }
    }

    /// `ToUint64(int base = 10)`.
    pub fn to_uint64(&self, base: i32) -> u64 {
        let (negative, magnitude) = parse_int_prefix(&self.data, base as u32);
        let val = magnitude.min(u128::from(u64::MAX)) as u64;
        if negative { val.wrapping_neg() } else { val }
    }

    /// `ToInt32(int base = 10)`. Matches `sys_strtoi32`'s clamp to
    /// `[INT32_MIN, INT32_MAX]` on overflow.
    pub fn to_int32(&self, base: i32) -> i32 {
        let (negative, magnitude) = parse_int_prefix(&self.data, base as u32);
        let signed: i128 = if negative {
            -(magnitude as i128)
        } else {
            magnitude as i128
        };
        signed.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
    }

    /// `ToInt64(int base = 10)`. Matches `strtoll`'s clamp to
    /// `[INT64_MIN, INT64_MAX]` on overflow.
    pub fn to_int64(&self, base: i32) -> i64 {
        let (negative, magnitude) = parse_int_prefix(&self.data, base as u32);
        let signed: i128 = if negative {
            -(magnitude as i128)
        } else {
            magnitude as i128
        };
        signed.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
    }

    /// `ToDouble()`.
    pub fn to_double(&self) -> f64 {
        parse_float_prefix(&self.data)
    }

    /// `ToFloat()`.
    pub fn to_float(&self) -> f32 {
        parse_float_prefix(&self.data) as f32
    }

    /// `Hash()`: delegates to Kyty's `Core::hash` one-at-a-time hash (see
    /// [`crate::hash`]), same as the C++ `SimpleArray<char>::Hash()` this
    /// wraps.
    pub fn hash(&self) -> u32 {
        hash::hash(&self.data)
    }

    /// `SortChars()`.
    pub fn sort_chars(&self) -> String8 {
        if self.is_empty() {
            return String8::new();
        }
        let mut data = self.data.clone();
        data.sort_unstable();
        String8 { data }
    }
}

impl PartialEq for String8 {
    fn eq(&self, other: &Self) -> bool {
        self.equal(other)
    }
}
impl Eq for String8 {}

impl PartialEq<str> for String8 {
    fn eq(&self, other: &str) -> bool {
        self.equal_str(other)
    }
}
impl PartialEq<&str> for String8 {
    fn eq(&self, other: &&str) -> bool {
        self.equal_str(other)
    }
}

impl std::fmt::Display for String8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.data))
    }
}

impl From<&str> for String8 {
    fn from(s: &str) -> Self {
        Self {
            data: s.as_bytes().to_vec(),
        }
    }
}
impl From<String> for String8 {
    fn from(s: String) -> Self {
        Self {
            data: s.into_bytes(),
        }
    }
}
impl From<Vec<u8>> for String8 {
    fn from(data: Vec<u8>) -> Self {
        Self { data }
    }
}
impl From<u8> for String8 {
    fn from(ch: u8) -> Self {
        String8::from_char(ch, 1)
    }
}

impl std::ops::Index<u32> for String8 {
    type Output = u8;
    fn index(&self, index: u32) -> &u8 {
        &self.data[index as usize]
    }
}
impl std::ops::IndexMut<u32> for String8 {
    fn index_mut(&mut self, index: u32) -> &mut u8 {
        &mut self.data[index as usize]
    }
}

impl std::ops::AddAssign<&String8> for String8 {
    /// `operator+=(const String8& src)`.
    fn add_assign(&mut self, other: &String8) {
        self.data.extend_from_slice(&other.data);
    }
}
impl std::ops::AddAssign<u8> for String8 {
    /// `operator+=(char ch)`.
    fn add_assign(&mut self, ch: u8) {
        self.data.push(ch);
    }
}
impl std::ops::AddAssign<&str> for String8 {
    /// `operator+=(const char* utf8_str)`.
    fn add_assign(&mut self, s: &str) {
        self.data.extend_from_slice(s.as_bytes());
    }
}

impl std::ops::Add<&String8> for String8 {
    type Output = String8;
    /// `operator+(const String8&, const String8&)`.
    fn add(mut self, other: &String8) -> String8 {
        self += other;
        self
    }
}
impl std::ops::Add<u8> for String8 {
    type Output = String8;
    /// `operator+(const String8&, char)`.
    fn add(mut self, ch: u8) -> String8 {
        self += ch;
        self
    }
}
impl std::ops::Add<&str> for String8 {
    type Output = String8;
    /// `operator+(const String8&, const char*)`.
    fn add(mut self, s: &str) -> String8 {
        self += s;
        self
    }
}

/// Kyty's `Kyty::Core::StringList8`: a thin wrapper over `Vec<String8>`
/// exposing the original's `Vector<String8>`-inherited surface that
/// `String8`'s own methods need, plus `StringList8`'s own additions
/// (`Contains`/`Concat`/`Equal`).
#[derive(Debug, Clone, Default)]
pub struct StringList8 {
    items: Vec<String8>,
}

impl StringList8 {
    /// `StringList8()` (via `using Vector<String8>::Vector`).
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
    pub fn add(&mut self, s: String8) {
        self.items.push(s);
    }

    /// `At(uint32_t index) const`.
    pub fn at(&self, index: u32) -> &String8 {
        &self.items[index as usize]
    }

    /// `begin()`/`end()` (const iteration).
    pub fn iter(&self) -> std::slice::Iter<'_, String8> {
        self.items.iter()
    }

    /// `StringList8::Contains(const String8& str)`.
    pub fn contains(&self, str_: &String8) -> bool {
        self.items.iter().any(|s| s.contains_str(str_))
    }

    /// `StringList8::Concat(const String8& str)`: joins all elements with
    /// `str` as separator.
    pub fn concat(&self, sep: &String8) -> String8 {
        let mut r = String8::new();
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                r += sep;
            }
            r += item;
        }
        r
    }

    /// `StringList8::Concat(char chr)`.
    pub fn concat_char(&self, chr: u8) -> String8 {
        let mut r = String8::new();
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                r += chr;
            }
            r += item;
        }
        r
    }

    /// `StringList8::Equal(const StringList8& str)`.
    pub fn equal(&self, other: &StringList8) -> bool {
        self.items.len() == other.items.len()
            && self
                .items
                .iter()
                .zip(other.items.iter())
                .all(|(a, b)| a.equal(b))
    }
}

impl PartialEq for StringList8 {
    fn eq(&self, other: &Self) -> bool {
        self.equal(other)
    }
}
impl Eq for StringList8 {}

impl std::ops::Index<u32> for StringList8 {
    type Output = String8;
    fn index(&self, index: u32) -> &String8 {
        &self.items[index as usize]
    }
}

impl From<Vec<String8>> for StringList8 {
    fn from(items: Vec<String8>) -> Self {
        Self { items }
    }
}

impl IntoIterator for StringList8 {
    type Item = String8;
    type IntoIter = std::vec::IntoIter<String8>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}
impl<'a> IntoIterator for &'a StringList8 {
    type Item = &'a String8;
    type IntoIter = std::slice::Iter<'a, String8>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let s = String8::new();
        assert_eq!(s.size(), 0);
        assert!(s.is_empty());
        assert!(!s.is_invalid());
    }

    #[test]
    fn from_char_repeat() {
        let s = String8::from_char(b'x', 3);
        assert_eq!(s.size(), 3);
        assert_eq!(s.as_bytes(), b"xxx");

        let empty = String8::from_char(b'x', 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn from_str_and_bytes() {
        let a = String8::from("hello");
        assert_eq!(a.as_bytes(), b"hello");
        let b = String8::from_bytes(b"world");
        assert_eq!(b.as_bytes(), b"world");
        let c: String8 = String::from("owned").into();
        assert_eq!(c.as_bytes(), b"owned");
    }

    #[test]
    fn clear_resets_to_empty() {
        let mut s = String8::from("abc");
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.size(), 0);
    }

    #[test]
    fn index_and_at() {
        let mut s = String8::from("abc");
        assert_eq!(s[0], b'a');
        assert_eq!(*s.at(2), b'c');
        s[1] = b'Z';
        assert_eq!(s.as_bytes(), b"aZc");
    }

    #[test]
    fn equal_variants() {
        let a = String8::from("abc");
        assert!(a.equal(&String8::from("abc")));
        assert!(!a.equal(&String8::from("abd")));
        assert!(String8::from("x").equal_char(b'x'));
        assert!(!a.equal_char(b'a')); // "abc" != "a"
        assert!(a.equal_str("abc"));
        assert_eq!(a, String8::from("abc"));
        assert_eq!(a, "abc");
    }

    #[test]
    fn add_assign_and_concatenation() {
        let mut s = String8::from("foo");
        s += &String8::from("bar");
        s += b'!';
        s += "?";
        assert_eq!(s.as_bytes(), b"foobar!?");

        let combined = String8::from("a") + &String8::from("b") + b'c';
        assert_eq!(combined.as_bytes(), b"abc");
    }

    #[test]
    fn mid_basic_and_out_of_range() {
        let s = String8::from("hello world");
        assert_eq!(s.mid(0, 5).as_bytes(), b"hello");
        assert_eq!(s.mid(6, 5).as_bytes(), b"world");
        assert_eq!(s.mid(6, 100).as_bytes(), b"world"); // count clamped
        assert!(s.mid(100, 1).is_empty()); // first >= size
        assert_eq!(s.mid_from(6).as_bytes(), b"world");
    }

    #[test]
    fn left_and_right() {
        let s = String8::from("hello");
        assert_eq!(s.left(3).as_bytes(), b"hel");
        assert_eq!(s.right(3).as_bytes(), b"llo");
        assert_eq!(s.right(100).as_bytes(), b"hello"); // count >= size
    }

    #[test]
    fn trim_variants() {
        let s = String8::from("  hi there  ");
        assert_eq!(s.trim_left().as_bytes(), b"hi there  ");
        assert_eq!(s.trim_right().as_bytes(), b"  hi there");
        assert_eq!(s.trim().as_bytes(), b"hi there");
        assert!(String8::from("   ").trim().is_empty());
    }

    #[test]
    fn simplify_collapses_whitespace_runs() {
        let s = String8::from("  a\t\tb   c  ");
        // Leading run dropped; internal runs collapsed to their first byte
        // ('\t' for the a/b gap); trailing trimmed.
        assert_eq!(s.simplify().as_bytes(), b"a\tb c");
    }

    #[test]
    fn replace_char_and_replace_str() {
        let s = String8::from("a-b-c");
        assert_eq!(s.replace_char(b'-', b'_').as_bytes(), b"a_b_c");

        let s2 = String8::from("foo bar foo");
        let replaced = s2.replace_str(&String8::from("foo"), &String8::from("baz"));
        assert_eq!(replaced.as_bytes(), b"baz bar baz");

        // Empty old_str: the `extra = 1` guard (avoids an infinite loop
        // matching a zero-width needle at the same spot forever) means
        // matches only land strictly *between* bytes, never at the very
        // start or end: new_str gets inserted once between 'a' and 'b'.
        let inserted = String8::from("ab").replace_str(&String8::new(), &String8::from("-"));
        assert_eq!(inserted.as_bytes(), b"a-b");
    }

    #[test]
    fn remove_at_variants() {
        let s = String8::from("hello");
        assert_eq!(s.remove_at(1, 2).as_bytes(), b"hlo");
        assert_eq!(s.remove_at(4, 10).as_bytes(), b"hell"); // count clamped
        assert_eq!(s.remove_at(10, 1).as_bytes(), b"hello"); // index >= size: no-op
    }

    #[test]
    fn remove_char_and_remove_str() {
        let s = String8::from("banana");
        assert_eq!(s.remove_char(b'a').as_bytes(), b"bnn");

        let s2 = String8::from("foo bar foo");
        assert_eq!(s2.remove_str(&String8::from("foo")).as_bytes(), b" bar ");
    }

    #[test]
    fn remove_last_and_first() {
        let s = String8::from("hello");
        assert_eq!(s.remove_last(2).as_bytes(), b"hel");
        assert_eq!(s.remove_first(2).as_bytes(), b"llo");
        assert!(s.remove_last(100).is_empty());
        assert!(s.remove_first(100).is_empty());
    }

    #[test]
    fn insert_at_middle_and_ends() {
        let s = String8::from("helloworld");
        let inserted = s.insert_at(5, &String8::from(" "));
        assert_eq!(inserted.as_bytes(), b"hello world");

        let s2 = String8::from("bc");
        assert_eq!(s2.insert_at(0, &String8::from("a")).as_bytes(), b"abc");
    }

    #[test]
    fn find_index_str_variants() {
        let s = String8::from("abcabc");
        assert_eq!(s.find_index(&String8::from("bc"), 0), 1);
        assert_eq!(s.find_index(&String8::from("bc"), 2), 4);
        assert_eq!(s.find_index(&String8::from("zz"), 0), INVALID_INDEX);
        assert_eq!(s.find_index(&String8::new(), 3), 3); // empty needle
    }

    #[test]
    fn find_last_index_str_variants() {
        let s = String8::from("abcabc");
        assert_eq!(s.find_last_index(&String8::from("bc"), INVALID_INDEX), 4);
        assert_eq!(
            s.find_last_index(&String8::from("zz"), INVALID_INDEX),
            INVALID_INDEX
        );
        // Needle longer than haystack: safely INVALID_INDEX (no UB replicated).
        assert_eq!(
            s.find_last_index(&String8::from("way too long!!"), INVALID_INDEX),
            INVALID_INDEX
        );
    }

    #[test]
    fn find_index_and_last_index_char() {
        let s = String8::from("hello");
        assert_eq!(s.find_index_char(b'l', 0), 2);
        assert_eq!(s.find_index_char(b'l', 3), 3);
        assert_eq!(s.find_index_char(b'z', 0), INVALID_INDEX);
        assert_eq!(s.find_last_index_char(b'l', INVALID_INDEX), 3);
        assert_eq!(s.find_last_index_char(b'z', INVALID_INDEX), INVALID_INDEX);
    }

    #[test]
    fn index_valid() {
        let s = String8::from("ab");
        assert!(s.index_valid(0));
        assert!(s.index_valid(1));
        assert!(!s.index_valid(2));
    }

    #[test]
    fn contains_str_and_char_variants() {
        let s = String8::from("hello world");
        assert!(s.contains_str(&String8::from("wor")));
        assert!(!s.contains_str(&String8::from("xyz")));
        assert!(s.contains_char(b'w'));
        assert!(!s.contains_char(b'z'));

        let any_list = StringList8::from(vec![String8::from("xyz"), String8::from("wor")]);
        assert!(s.contains_any_str(&any_list));
        let all_list = StringList8::from(vec![String8::from("hello"), String8::from("world")]);
        assert!(s.contains_all_str(&all_list));
        let not_all = StringList8::from(vec![String8::from("hello"), String8::from("nope")]);
        assert!(!s.contains_all_str(&not_all));

        assert!(s.contains_any_char(&String8::from("xyzw")));
        assert!(s.contains_all_char(&String8::from("helo")));
        assert!(!s.contains_all_char(&String8::from("helz")));
    }

    #[test]
    fn starts_and_ends_with() {
        let s = String8::from("hello world");
        assert!(s.starts_with(&String8::from("hello")));
        assert!(!s.starts_with(&String8::from("world")));
        assert!(s.ends_with(&String8::from("world")));
        assert!(!s.ends_with(&String8::from("hello")));
        assert!(s.starts_with_char(b'h'));
        assert!(s.ends_with_char(b'd'));
        assert!(!s.ends_with(&String8::from("way too long to fit")));
    }

    #[test]
    fn path_helpers() {
        let path = String8::from("/usr/local/bin.exe");
        assert_eq!(path.directory_without_filename().as_bytes(), b"/usr/local/");
        assert_eq!(path.filename_without_directory().as_bytes(), b"bin.exe");
        assert_eq!(
            path.filename_without_extension().as_bytes(),
            b"/usr/local/bin"
        );
        assert_eq!(path.extension_without_filename().as_bytes(), b".exe");

        let no_slash = String8::from("bin.exe");
        assert_eq!(no_slash.directory_without_filename().as_bytes(), b"");
        assert_eq!(no_slash.filename_without_directory().as_bytes(), b"bin.exe");
    }

    #[test]
    fn slash_fixups() {
        let win = String8::from(r"a\b\c");
        assert_eq!(win.fix_filename_slash().as_bytes(), b"a/b/c");

        let dir = String8::from(r"a\b");
        assert_eq!(dir.fix_directory_slash().as_bytes(), b"a/b/");
        let already = String8::from("a/b/");
        assert_eq!(already.fix_directory_slash().as_bytes(), b"a/b/");
    }

    #[test]
    fn split_str_with_and_without_empty_parts() {
        let s = String8::from("a,,b,c");
        let no_empty = s.split(&String8::from(","), SplitType::SplitNoEmptyParts);
        assert_eq!(no_empty.size(), 3);
        assert_eq!(no_empty.at(0).as_bytes(), b"a");
        assert_eq!(no_empty.at(1).as_bytes(), b"b");
        assert_eq!(no_empty.at(2).as_bytes(), b"c");

        let with_empty = s.split(&String8::from(","), SplitType::WithEmptyParts);
        assert_eq!(with_empty.size(), 4);
        assert_eq!(with_empty.at(1).as_bytes(), b"");
    }

    #[test]
    fn split_char_variant() {
        let s = String8::from("a/b//c");
        let parts = s.split_char(b'/', SplitType::SplitNoEmptyParts);
        assert_eq!(parts.size(), 3);
        assert_eq!(parts.at(2).as_bytes(), b"c");
    }

    #[test]
    fn safe_lua_escapes_backslash_and_quote() {
        let s = String8::from(r"it's a \test\");
        let escaped = s.safe_lua();
        assert_eq!(escaped.as_bytes(), br"it\'s a \\test\\".as_slice());
    }

    #[test]
    fn safe_csv_quotes_when_needed() {
        let plain = String8::from("plain");
        assert_eq!(plain.safe_csv().as_bytes(), b"plain");

        let with_quote = String8::from("has \"quote\"");
        assert_eq!(with_quote.safe_csv().as_bytes(), b"\"has \"\"quote\"\"\"");

        let formula = String8::from("=SUM(A1)");
        assert_eq!(formula.safe_csv().as_bytes(), b"\" =SUM(A1)\"");
    }

    #[test]
    fn to_integers_with_bases_signs_and_overflow() {
        assert_eq!(String8::from("42").to_uint32(10), 42);
        assert_eq!(String8::from("  -7").to_int32(10), -7);
        assert_eq!(String8::from("ff").to_uint32(16), 255);
        assert_eq!(String8::from("0x1A").to_uint32(0), 26); // base 0 auto-detect
        assert_eq!(String8::from("123abc").to_uint32(10), 123); // trailing garbage ignored
        assert_eq!(String8::from("notanumber").to_uint32(10), 0);
        assert_eq!(String8::from("99999999999").to_int32(10), i32::MAX); // overflow clamps
        assert_eq!(String8::from("-99999999999").to_int32(10), i32::MIN);
    }

    #[test]
    fn to_int64_and_uint64() {
        assert_eq!(String8::from("123456789012").to_int64(10), 123_456_789_012);
        assert_eq!(String8::from("-5").to_int64(10), -5);
        assert_eq!(String8::from("100").to_uint64(10), 100);
    }

    #[test]
    fn to_double_and_float() {
        assert!((String8::from("3.5").to_double() - 3.5).abs() < 1e-9);
        assert!((String8::from("-2.5e2xyz").to_double() - (-250.0)).abs() < 1e-9);
        assert_eq!(String8::from("notanumber").to_double(), 0.0);
        assert!((String8::from("1.5").to_float() - 1.5_f32).abs() < 1e-6);
    }

    #[test]
    fn hash_is_deterministic_and_matches_content() {
        let a = String8::from("kyty");
        let b = String8::from("kyty");
        let c = String8::from("other");
        assert_eq!(a.hash(), b.hash());
        assert_ne!(a.hash(), c.hash());
        assert_eq!(a.hash(), crate::hash::hash(b"kyty"));
    }

    #[test]
    fn sort_chars_orders_bytes() {
        let s = String8::from("dcba");
        assert_eq!(s.sort_chars().as_bytes(), b"abcd");
        assert!(String8::new().sort_chars().is_empty());
    }

    #[test]
    fn string_list_contains_concat_equal() {
        let list = StringList8::from(vec![String8::from("foo"), String8::from("bar")]);
        assert!(list.contains(&String8::from("oo")));
        assert!(!list.contains(&String8::from("zzz")));

        assert_eq!(list.concat(&String8::from(", ")).as_bytes(), b"foo, bar");
        assert_eq!(list.concat_char(b'-').as_bytes(), b"foo-bar");

        let same = StringList8::from(vec![String8::from("foo"), String8::from("bar")]);
        let different = StringList8::from(vec![String8::from("foo"), String8::from("baz")]);
        assert!(list.equal(&same));
        assert_eq!(list, same);
        assert!(!list.equal(&different));
    }
}
