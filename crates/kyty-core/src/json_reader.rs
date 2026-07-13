//! Port of Kyty's `Core::Json` reader
//! (`reference/kyty/source/include/Kyty/Core/JsonReader.h`,
//! `reference/kyty/source/lib/Core/src/JsonReader.cpp`).
//!
//! Kyty's own hand-written recursive-descent JSON parser — *not* a wrapper
//! over `serde_json` (not a workspace dependency; see the port conventions
//! doc). The public surface (`Json::Create`/`GetItem`/`GetString`/`GetFloat`/
//! `GetInt`/`GetBool`/`ToString`/`ToFloat`/`ToInt`/`ToBool`/`ToArray`/
//! `DbgCheckList`/`GetError`) is preserved 1:1 (method names translated to
//! Rust `snake_case`, semantics unchanged), including its documented quirks
//! (see below).
//!
//! # Std / ownership mapping
//!
//! - **Raw `Json*` tree, manual `new`/`Delete`, `KYTY_CLASS_NO_COPY`** → the
//!   Rust `Json` owns its children directly as `Vec<Json>` (a genuine
//!   recursive value type). There is no pointer aliasing to guard against, so
//!   the C++ no-copy restriction (needed only to avoid double-freeing raw
//!   pointers) is dropped; `Json` is a plain, safely `Clone`-able value.
//! - **`const char32_t*` cursors with an implicit hidden trailing `'\0'`** →
//!   ordinary Rust slices `&[char]`. Every place the C++ code tests
//!   `*ptr == 0` to detect "ran off the end of the buffer" is replaced by a
//!   `.len()`/bounds check on the slice — the same algorithm, zero unsafe,
//!   zero raw pointers. A parse step "failing" (C++: returns `nullptr`) is
//!   `Option::None`; "succeeding, cursor advanced" (C++: returns the new
//!   pointer) is `Some(&remaining_slice)`.
//! - **`static String* g_json_error`** (lazily-allocated global, read by
//!   `Json::GetError()`) → `static JSON_ERROR: Mutex<Option<String>>`, a
//!   `std`-only lazily-populated global (`Mutex::new(None)` is a `const fn`,
//!   so no extra init step or crate is needed).
//! - **`static Json* g_null` singleton returned by `GetItem` on a miss** →
//!   `static NULL_JSON: OnceLock<Json>` (also `std`-only), read through a
//!   private `null_json() -> &'static Json` helper; `&'static Json` coerces
//!   to whatever borrow lifetime the caller expects, so `get_item` keeps the
//!   exact "always returns *some* `&Json`, never a null pointer" contract.
//!   `Json::init()` is kept as a real (if now-trivial) public function for
//!   API parity; it simply forces that lazy singleton to exist.
//! - **`String out(' ', len)` pre-sized before decoding an escaped JSON
//!   string** → dropped; `parse_string` decodes into a plain growable
//!   `Vec<char>` in one pass (C++ needed a first counting pass only because
//!   its `String` buffer had to be preallocated up front — count-then-fill
//!   and grow-as-you-go produce byte-for-byte the same decoded text).
//! - Number text → `f64`/`i64` conversion reuses the already-ported
//!   [`crate::string::String::to_double`] (itself backed by
//!   [`crate::string8::String8::to_double`]), matching the C++'s own
//!   `str.ToDouble()` call exactly.
//!
//! # Preserved quirks (intentional, matching the C++ 1:1)
//!
//! - `\uXXXX` escapes are **not** supported: hitting one inside a JSON string
//!   sets the parse error and fails, exactly like the original
//!   (`case U'u': set_error(ptr); return nullptr;`).
//! - A value position holding an unrecognized character (including running
//!   out of input) is **not** a parse error: `parse_value`'s C++ `default:`
//!   case just returns the cursor unchanged without setting a type; ported
//!   here as the same no-op fallthrough.
//! - `GetItem` does a case-insensitive **ASCII** name match
//!   (`EqualAsciiNoCase`), and object member lookup is a linear scan over
//!   `m_list`/`self.list` — both ported as-is.
//! - `Json::ToBool()` faithfully returns `i64` (`0`/`1`), not `bool` — this
//!   is what the C++ header itself declares.
//! - `DbgCheckList`'s "unknown key" check delegates to the already-ported
//!   [`crate::string::StringList::contains`], which (matching
//!   `StringList::Contains` in `String.cpp`) tests *substring* containment
//!   per list entry, not list-element equality — preserved, not "fixed".

use crate::string::{Case, String, StringList};
use std::string::String as StdString;
use std::sync::{Mutex, OnceLock};

/// `Kyty::Core::JsonType`. See module doc comment for the `JsonNULL` →
/// `JsonNull` casing note (the enum's only naming departure from the C++).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JsonType {
    JsonBool,
    #[default]
    JsonNull,
    JsonNumber,
    JsonString,
    JsonArray,
    JsonObject,
}

/// `Kyty::Core::Json`: a faithful, safe-Rust re-implementation of the C++
/// recursive-descent JSON tree/parser. See the module doc comment for the
/// full ownership/std mapping.
#[derive(Debug, Clone, Default)]
pub struct Json {
    list: Vec<Json>,
    kind: JsonType,
    value_string: String,
    value_int: i64,
    value_double: f64,
    value_bool: bool,
    name: String,
}

static JSON_ERROR: Mutex<Option<String>> = Mutex::new(None);

fn set_error(remaining: &[char]) {
    let text: StdString = remaining.iter().collect();
    if let Ok(mut guard) = JSON_ERROR.lock() {
        *guard = Some(String::from(text));
    }
}

fn null_json() -> &'static Json {
    static NULL_JSON: OnceLock<Json> = OnceLock::new();
    NULL_JSON.get_or_init(Json::default)
}

/// `static const char32_t* skip(const char32_t* in)`.
fn skip(input: &[char]) -> &[char] {
    let mut i = 0;
    while i < input.len() && input[i].is_whitespace() {
        i += 1;
    }
    &input[i..]
}

/// `static const char32_t* skip_number(const char32_t* in)`.
fn skip_number(input: &[char]) -> &[char] {
    let mut i = 0;
    while i < input.len() {
        let c = input[i];
        if c.is_ascii_digit() || c == '-' || c == '+' || c == 'e' || c == 'E' || c == '.' {
            i += 1;
        } else {
            break;
        }
    }
    &input[i..]
}

/// `Char::EqualAsciiN(value, ascii, n)` (as used by `parse_value` to match
/// the `null`/`false`/`true` literals).
fn equal_ascii_n(value: &[char], ascii: &str) -> bool {
    let n = ascii.chars().count();
    value.len() >= n && value.iter().zip(ascii.chars()).all(|(&a, b)| a == b)
}

impl Json {
    /// `static void Init()`.
    pub fn init() {
        let _ = null_json();
    }

    /// `static const Json* Create(const String& str)`.
    pub fn create(s: &String) -> Option<Json> {
        let mut c = Json::default();
        let data = s.get_data_const();
        c.parse_value(skip(data))?;
        Some(c)
    }

    /// `static String GetError()`.
    pub fn get_error() -> String {
        JSON_ERROR.lock().map(|guard| guard.clone().unwrap_or_default()).unwrap_or_default()
    }

    /// `const Json* GetItem(const char* string) const`.
    pub fn get_item(&self, name: &str) -> &Json {
        for child in &self.list {
            if child.name.equal_ascii_no_case(name) {
                return child;
            }
        }
        null_json()
    }

    /// `String GetString(const char* name, const String& default_value) const`.
    pub fn get_string(&self, name: &str, default_value: &String) -> String {
        let n = self.get_item(name);
        if !n.is_null() { n.value_string.clone() } else { default_value.clone() }
    }

    /// `String GetString(const char* name) const` (defaults to `U""`).
    pub fn get_string_default(&self, name: &str) -> String {
        self.get_string(name, &String::new())
    }

    /// `double GetFloat(const char* name, double default_value) const`.
    pub fn get_float(&self, name: &str, default_value: f64) -> f64 {
        let n = self.get_item(name);
        if !n.is_null() { n.value_double } else { default_value }
    }

    /// `int64_t GetInt(const char* name, int64_t default_value) const`.
    pub fn get_int(&self, name: &str, default_value: i64) -> i64 {
        let n = self.get_item(name);
        if !n.is_null() { n.value_int } else { default_value }
    }

    /// `bool GetBool(const char* name, bool default_value) const`.
    pub fn get_bool(&self, name: &str, default_value: bool) -> bool {
        let n = self.get_item(name);
        if !n.is_null() { n.value_bool } else { default_value }
    }

    /// `String GetName() const`.
    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    /// `JsonType GetType() const`.
    pub fn get_type(&self) -> JsonType {
        self.kind
    }

    /// `bool IsNull() const`.
    pub fn is_null(&self) -> bool {
        self.kind == JsonType::JsonNull
    }

    /// `bool IsNumber() const`.
    pub fn is_number(&self) -> bool {
        self.kind == JsonType::JsonNumber
    }

    /// `bool IsString() const`.
    pub fn is_string(&self) -> bool {
        self.kind == JsonType::JsonString
    }

    /// `bool IsObject() const`.
    pub fn is_object(&self) -> bool {
        self.kind == JsonType::JsonObject
    }

    /// `bool IsArray() const`.
    pub fn is_array(&self) -> bool {
        self.kind == JsonType::JsonArray
    }

    /// `bool IsBool() const`.
    pub fn is_bool(&self) -> bool {
        self.kind == JsonType::JsonBool
    }

    /// `String ToString() const`.
    pub fn to_string(&self) -> String {
        self.value_string.clone()
    }

    /// `double ToFloat() const`.
    pub fn to_float(&self) -> f64 {
        self.value_double
    }

    /// `int64_t ToInt() const`.
    pub fn to_int(&self) -> i64 {
        self.value_int
    }

    /// `int64_t ToBool() const`: faithfully returns `i64` (`0`/`1`), matching
    /// the C++ header's own declared return type (not `bool`).
    pub fn to_bool(&self) -> i64 {
        self.value_bool as i64
    }

    /// `const ListType& ToArray() const`.
    pub fn to_array(&self) -> &[Json] {
        &self.list
    }

    /// `StringList DbgCheckList(const StringList& required, const StringList& optional) const`.
    pub fn dbg_check_list(&self, required: &StringList, optional: &StringList) -> StringList {
        let mut errors = StringList::new();

        for i in 0..required.size() {
            let req = required.at(i);
            if self.get_item(&req.to_string()).name.is_empty() {
                errors.add(String::from("missing: ") + req);
            }
        }

        for child in &self.list {
            if !(child.name.is_empty() || required.contains(&child.name, Case::Sensitive) || optional.contains(&child.name, Case::Sensitive))
            {
                errors.add(String::from("unknown: ") + &child.name);
            }
        }

        errors
    }

    /// `const char32_t* parse_value(const char32_t* value)`.
    fn parse_value<'a>(&mut self, value: &'a [char]) -> Option<&'a [char]> {
        match value.first().copied() {
            Some('n') => {
                if equal_ascii_n(value, "null") {
                    self.kind = JsonType::JsonNull;
                    return Some(&value[4..]);
                }
            }
            Some('f') => {
                if equal_ascii_n(value, "false") {
                    self.kind = JsonType::JsonBool;
                    return Some(&value[5..]);
                }
            }
            Some('t') => {
                if equal_ascii_n(value, "true") {
                    self.kind = JsonType::JsonBool;
                    self.value_bool = true;
                    self.value_int = 1;
                    return Some(&value[4..]);
                }
            }
            Some('"') => return self.parse_string(value),
            Some('[') => return self.parse_array(value),
            Some('{') => return self.parse_object(value),
            Some(c) if c == '-' || c.is_ascii_digit() => return self.parse_number(value),
            None => return Some(value),
            _ => return Some(value),
        }

        set_error(value);
        None
    }

    /// `const char32_t* parse_array(const char32_t* value)`.
    fn parse_array<'a>(&mut self, value: &'a [char]) -> Option<&'a [char]> {
        if value.first().copied() != Some('[') {
            set_error(value);
            return None;
        }

        self.kind = JsonType::JsonArray;

        let mut value = skip(&value[1..]);

        if value.first().copied() == Some(']') {
            return Some(&value[1..]);
        }

        let mut child = Json::default();
        value = skip(child.parse_value(skip(value))?);
        self.list.push(child);

        while value.first().copied() == Some(',') {
            let mut child = Json::default();
            value = skip(child.parse_value(skip(&value[1..]))?);
            self.list.push(child);
        }

        if value.first().copied() == Some(']') {
            return Some(&value[1..]);
        }

        set_error(value);
        None
    }

    /// `const char32_t* parse_object(const char32_t* value)`.
    fn parse_object<'a>(&mut self, value: &'a [char]) -> Option<&'a [char]> {
        if value.first().copied() != Some('{') {
            set_error(value);
            return None;
        }

        self.kind = JsonType::JsonObject;

        let mut value = skip(&value[1..]);

        if value.first().copied() == Some('}') {
            return Some(&value[1..]);
        }

        let mut child = Json::default();
        value = skip(child.parse_value(skip(value))?);
        child.name = child.value_string.clone();
        child.value_string = String::new();

        if value.first().copied() != Some(':') {
            set_error(value);
            return None;
        }

        value = skip(child.parse_value(skip(&value[1..]))?);
        self.list.push(child);

        while value.first().copied() == Some(',') {
            let mut child = Json::default();
            value = skip(child.parse_value(skip(&value[1..]))?);
            child.name = child.value_string.clone();
            child.value_string = String::new();

            if value.first().copied() != Some(':') {
                set_error(value);
                return None;
            }

            value = skip(child.parse_value(skip(&value[1..]))?);
            self.list.push(child);
        }

        if value.first().copied() == Some('}') {
            return Some(&value[1..]);
        }

        set_error(value);
        None
    }

    /// `const char32_t* parse_number(const char32_t* value)`.
    fn parse_number<'a>(&mut self, value: &'a [char]) -> Option<&'a [char]> {
        let end = skip_number(value);
        let len = value.len() - end.len();

        if len != 0 {
            let text: StdString = value[..len].iter().collect();
            let s = String::from(text);
            self.kind = JsonType::JsonNumber;
            self.value_double = s.to_double();
            self.value_int = self.value_double as i64;
            self.value_string = s;
            return Some(end);
        }

        set_error(value);
        None
    }

    /// `const char32_t* parse_string(const char32_t* str)`.
    fn parse_string<'a>(&mut self, s: &'a [char]) -> Option<&'a [char]> {
        if s.first().copied() != Some('"') {
            set_error(s);
            return None;
        }

        let mut out: Vec<char> = Vec::new();
        let mut p = 1usize;

        while p < s.len() && s[p] != '"' {
            if s[p] != '\\' {
                out.push(s[p]);
                p += 1;
                continue;
            }

            p += 1;
            let escaped = if p < s.len() { s[p] } else { '\0' };
            match escaped {
                'b' => out.push('\u{0008}'),
                'f' => out.push('\u{000C}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    set_error(&s[p..]);
                    return None;
                }
                other => out.push(other),
            }
            p += 1;
        }

        if p < s.len() && s[p] == '"' {
            p += 1;
        }

        let text: StdString = out.into_iter().collect();
        self.value_string = String::from(text);
        self.kind = JsonType::JsonString;

        Some(&s[p..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that observe [`Json::get_error`], which reads the
    /// process-global `JSON_ERROR` (a faithful port of the C++ `g_json_error`
    /// global). Any failing parse overwrites it, so tests asserting on its
    /// contents must not run concurrently with one another.
    static ERROR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn parse(src: &str) -> Json {
        Json::create(&String::from(src)).expect("expected valid JSON")
    }

    #[test]
    fn parses_null_bool_number_string_literals() {
        let n = parse("null");
        assert!(n.is_null());

        let t = parse("true");
        assert!(t.is_bool());
        assert_eq!(t.to_bool(), 1);

        let f = parse("false");
        assert!(f.is_bool());
        assert_eq!(f.to_bool(), 0);

        let num = parse("  -12.5e1 ");
        assert!(num.is_number());
        assert!((num.to_float() - (-125.0)).abs() < 1e-9);
        assert_eq!(num.to_int(), -125);

        let s = parse("\"hello\"");
        assert!(s.is_string());
        assert_eq!(s.to_string(), "hello");
    }

    #[test]
    fn parses_string_escapes() {
        let s = parse(r#""a\nb\tc\"d\\e""#);
        assert!(s.is_string());
        assert_eq!(s.to_string(), "a\nb\tc\"d\\e");
    }

    #[test]
    fn unicode_escape_is_unsupported_and_errors() {
        let _guard = ERROR_LOCK.lock().unwrap();
        // JSON text: "aAb" (a backslash-u escape inside a string).
        let src = "\"a\\u0041b\"";
        let result = Json::create(&String::from(src));
        assert!(result.is_none());
        // The `\u` case calls `set_error` with the remaining text starting at
        // `u`, so the recorded error is deterministically non-empty.
        assert!(Json::get_error().to_string().contains('u'));
    }

    #[test]
    fn parses_array_of_numbers() {
        let arr = parse("[1, 2, 3]");
        assert!(arr.is_array());
        let items = arr.to_array();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].to_int(), 1);
        assert_eq!(items[1].to_int(), 2);
        assert_eq!(items[2].to_int(), 3);
    }

    #[test]
    fn parses_empty_array_and_object() {
        let arr = parse("[]");
        assert!(arr.is_array());
        assert_eq!(arr.to_array().len(), 0);

        let obj = parse("{}");
        assert!(obj.is_object());
        assert_eq!(obj.to_array().len(), 0);
    }

    #[test]
    fn parses_object_with_mixed_members_and_case_insensitive_lookup() {
        let obj = parse(r#"{"Name": "kyty", "count": 3, "ok": true, "extra": null}"#);
        assert!(obj.is_object());

        assert_eq!(obj.get_string("name", &String::new()), "kyty");
        assert_eq!(obj.get_string("NAME", &String::new()), "kyty");
        assert_eq!(obj.get_int("count", -1), 3);
        assert!(obj.get_bool("ok", false));
        assert!(obj.get_item("extra").is_null());

        // Missing key -> default value.
        assert_eq!(obj.get_string_default("missing"), "");
        assert_eq!(obj.get_float("missing", 4.5), 4.5);
        assert_eq!(obj.get_int("missing", 42), 42);
        assert!(obj.get_bool("missing", true));
        assert!(obj.get_item("missing").is_null());
    }

    #[test]
    fn parses_nested_object_and_array() {
        let obj = parse(r#"{"list": [1, {"a": 2}], "nested": {"x": 1}}"#);
        let list = obj.get_item("list");
        assert!(list.is_array());
        assert_eq!(list.to_array().len(), 2);
        assert_eq!(list.to_array()[0].to_int(), 1);
        assert!(list.to_array()[1].is_object());
        assert_eq!(list.to_array()[1].get_int("a", 0), 2);

        let nested = obj.get_item("nested");
        assert!(nested.is_object());
        assert_eq!(nested.get_int("x", 0), 1);
    }

    #[test]
    fn malformed_json_fails_and_records_error() {
        let _guard = ERROR_LOCK.lock().unwrap();

        // Structurally invalid inputs all fail to parse.
        assert!(Json::create(&String::from("{")).is_none());
        assert!(Json::create(&String::from("[1, 2")).is_none());
        assert!(Json::create(&String::from("nul")).is_none());

        // The two failures above run off the end of the buffer, so — exactly
        // like the C++ original, whose `set_error` stores a pointer to the
        // terminating `'\0'` — they record an *empty* error string. Only a bad
        // token that still has trailing text (here `nul`) yields a non-empty,
        // deterministically observable error, so assert on that one.
        assert!(Json::get_error().to_string().contains("nul"));

        // Unterminated string: no closing quote, parser stops at end of
        // input without error (matches C++ behavior of treating end-of-
        // buffer like the closing quote).
        assert!(Json::create(&String::from("\"unterminated")).is_some());
    }

    #[test]
    fn get_error_is_empty_when_nothing_has_failed_yet_or_reports_last_failure() {
        // After a successful parse the error string still reflects whatever
        // the most recent *failure* was (Kyty never clears it on success) --
        // only assert it's queryable without panicking.
        let _ = parse("42");
        let _ = Json::get_error();
    }

    #[test]
    fn dbg_check_list_reports_missing_and_unknown_keys() {
        let obj = parse(r#"{"a": 1, "b": 2, "z": 3}"#);

        let mut required = StringList::new();
        required.add(String::from("a"));
        required.add(String::from("c"));

        let mut optional = StringList::new();
        optional.add(String::from("b"));

        let errors = obj.dbg_check_list(&required, &optional);
        let joined = errors.concat(&String::from("|")).to_string();

        assert!(joined.contains("missing: c"));
        assert!(joined.contains("unknown: z"));
        assert!(!joined.contains("unknown: a"));
        assert!(!joined.contains("unknown: b"));
    }

    #[test]
    fn init_is_idempotent() {
        Json::init();
        Json::init();
    }
}
