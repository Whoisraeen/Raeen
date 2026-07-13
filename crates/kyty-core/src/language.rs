//! Port of Kyty's `Core/Language` (`reference/kyty/source/include/Kyty/Core/Language.h`
//! + `reference/kyty/source/lib/Core/src/Language.cpp`).
//!
//! Kyty's `Language` is a namespace of free functions (not a class) built
//! around a process-global `Hashmap<String, LanguageId>* g_lang_map`
//! (2-letter ISO code -> [`LanguageId`]), populated once by `Language::Init()`
//! and then read by every other function in the namespace. The rest of the
//! module is static Unicode data tables (per-language alphabets, the fixed
//! ASCII punctuation/numeric lists, and English/Russian month/day names) plus
//! small query functions over them.
//!
//! # Std mapping
//!
//! - The namespace of free functions -> free functions directly in this
//!   module (`crate::language::*`), mirroring `Kyty::Core::Language::*`.
//! - The global `Hashmap<String, LanguageId>*` (heap-allocated by `Init()`,
//!   read afterwards, never freed) -> a private `std::sync::OnceLock<
//!   HashMap<String, LanguageId>>`, populated lazily on first access via
//!   `get_or_init`. This is a deliberate, documented divergence from the
//!   original's contract ("call `Init()` once at startup, or every other
//!   function dereferences a null pointer"): the Rust port is safe to call in
//!   any order — the first call to *any* function that needs the map builds
//!   it — so [`init`] is kept only for call-site parity with ported code that
//!   calls `Language::Init()` explicitly; it is not required for correctness.
//!   A plain `std::collections::HashMap` is used (not the crate's own
//!   [`crate::hashmap::Hashmap`], whose `Start`/`Next`/... iteration cursor is
//!   `RefCell`-backed and therefore not `Sync` — unusable inside a `static`).
//! - `Vector<char32_t>`/`String::Sort()`-based `string_remove_duplicates`
//!   (dedupe-then-sort helper used by every `Get*List` function) -> a private
//!   `Vec<char>` built the same way (linear "already seen?" scan, matching
//!   the original's `Contains` behavior/complexity) then `sort_unstable`
//!   (Rust `char` order is Unicode scalar value order, exactly the
//!   `char32_t` numeric order the original sorts by).
//! - `LanguageId GetId(const String& id)` unknown-code -> `LanguageId::Unknown`
//!   is preserved via `HashMap::get(..).copied().unwrap_or(LanguageId::Unknown)`
//!   (the same "default value" shape as the original's
//!   `Hashmap::Get(key, default)`).
//! - The two C++ overloads of `GetLettersList` (`(const String&)` and
//!   `(LanguageId)`) become two distinct Rust names, since Rust has no
//!   overloading: [`get_letters_list`] (by 2-letter code, delegates to
//!   [`get_id`]) and [`get_letters_list_by_id`] (by [`LanguageId`] directly),
//!   matching this crate's existing convention for disambiguating overloads
//!   by suffix (see `string.rs`'s `find_index`/`find_index_char`).
//! - `FOR_HASH`-driven `GetLanguages()` iterates the map in whatever bucket
//!   order Kyty's hand-rolled hash table produced (unspecified, not
//!   documented as stable); this port iterates `HashMap::keys()`, whose order
//!   is likewise unspecified — callers cannot rely on ordering in either
//!   implementation, so this is not an observable divergence.
//!
//! `LanguageId` here is the *canonical*, fully-populated (`Init`-derived)
//! enum: `Kyty::Core::LanguageId` from `Language.h`, with every variant the
//! header declares (`Unknown`, `German`, `English`, `French`, `Italian`,
//! `Portuguese`, `Russian`, `Spanish`), in declaration order. Note
//! `date_time.rs` currently carries its own *minimal stand-in* `LanguageId`
//! (documented there as "delete once `Language` is ported for real") — that
//! stand-in and this module's enum are intentionally two separate types for
//! now; wiring `date_time.rs` over to this one is an integration step, not
//! part of this port.
//!
//! Every `Get*` function that receives a `LanguageId`/2-letter code Kyty does
//! not have data for (`Unknown`, or any of `German`/`French`/`Italian`/
//! `Portuguese`/`Spanish` in the three `GetNameOf*` family — the original's
//! own tables only cover English and Russian there) calls `EXIT("unknown
//! language\n")`; ported here as `crate::exit!("unknown language")`, matching
//! this crate's established convention (see `date_time.rs`).

use crate::string::{String, StringList};
use std::collections::HashMap;
use std::sync::OnceLock;

/// `Kyty::Core::LanguageId` (`Language.h`). Declaration order preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageId {
    Unknown,
    German,
    English,
    French,
    Italian,
    Portuguese,
    Russian,
    Spanish,
}

// ---------------------------------------------------------------------
// Static data tables (`g_list_*`/`g_alphabet_*`/`g_*_month_*`/`g_*_day_*` in
// Language.cpp), verbatim.
// ---------------------------------------------------------------------

const LIST_NUMERIC: &str = "0123456789";
const LIST_PUNCTUATION: &str = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const LIST_PUNCTUATION_2: &str = "ªº¡¿";

const ALPHABET_ENGLISH_1: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const ALPHABET_ENGLISH_2: &str = "abcdefghijklmnopqrstuvwxyz";

const ALPHABET_RUSSIAN_1: &str = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ";
const ALPHABET_RUSSIAN_2: &str = "абвгдеёжзийклмнопрстуфхцчшщьыъэюя";

const ALPHABET_GERMAN_1: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜẞ";
const ALPHABET_GERMAN_2: &str = "abcdefghijklmnopqrstuvwxyzäöüß";

const ALPHABET_FRENCH_1: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZÀÂÆÈÉÊËÎÏÔŒÙÛÜŸÇ";
const ALPHABET_FRENCH_2: &str = "abcdefghijklmnopqrstuvwxyzàâæèéêëîïôœùûüÿç";

const ALPHABET_ITALIAN_1: &str = "ABCDEFGHILMNOPQRSTUVÀÈÉÌÍÏÒÓÙÚ";
const ALPHABET_ITALIAN_2: &str = "abcdefghilmnopqrstuvàèéìíïòóùú";

const ALPHABET_SPANISH_1: &str = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚÜ";
const ALPHABET_SPANISH_2: &str = "abcdefghijklmnñopqrstuvwxyzáéíóúü";

const ALPHABET_PORTUGUESE_1: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÉÊÍÒÓÔÕÚÜÇ";
const ALPHABET_PORTUGUESE_2: &str = "abcdefghijklmnopqrstuvwxyzàáâãéêíòóôõúüç";

const SHORT_MONTH_ENGLISH: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const MONTH_ENGLISH: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

const SHORT_MONTH_RUSSIAN: [&str; 12] = [
    "Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек",
];
const MONTH_RUSSIAN: [&str; 12] = [
    "Январь",
    "Февраль",
    "Март",
    "Апрель",
    "Май",
    "Июнь",
    "Июль",
    "Август",
    "Сентябрь",
    "Октябрь",
    "Ноябрь",
    "Декабрь",
];

const SHORT_DAY_ENGLISH: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const DAY_ENGLISH: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

const SHORT_DAY_RUSSIAN: [&str; 7] = ["Пнд", "Втн", "Срд", "Чтв", "Птн", "Сбт", "Вск"];
const DAY_RUSSIAN: [&str; 7] = [
    "Понедельник",
    "Вторник",
    "Среда",
    "Четверг",
    "Пятница",
    "Суббота",
    "Воскресенье",
];

// ---------------------------------------------------------------------
// `g_lang_map` (`static Hashmap<String, LanguageId>* g_lang_map`).
// ---------------------------------------------------------------------

static LANG_MAP: OnceLock<HashMap<String, LanguageId>> = OnceLock::new();

fn lang_map() -> &'static HashMap<String, LanguageId> {
    LANG_MAP.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert(String::from("de"), LanguageId::German);
        m.insert(String::from("en"), LanguageId::English);
        m.insert(String::from("fr"), LanguageId::French);
        m.insert(String::from("it"), LanguageId::Italian);
        m.insert(String::from("pt"), LanguageId::Portuguese);
        m.insert(String::from("ru"), LanguageId::Russian);
        m.insert(String::from("es"), LanguageId::Spanish);
        m
    })
}

/// `Language::Init()`. See the module doc comment: the map is actually
/// populated lazily on first use regardless of whether this is called, so
/// this exists only for call-site parity with ported code that calls
/// `Language::Init()` explicitly at startup.
pub fn init() {
    let _ = lang_map();
}

/// Private `string_remove_duplicates(const String& s)`: dedupe (preserving
/// first-seen order, like the original's linear `Contains` scan) then sort
/// ascending by code point.
fn string_remove_duplicates(s: &String) -> String {
    let mut chars: Vec<char> = Vec::new();
    for &ch in s.get_data_const() {
        if !chars.contains(&ch) {
            chars.push(ch);
        }
    }
    chars.sort_unstable();

    let mut ret = String::new();
    for ch in chars {
        ret += ch;
    }
    ret
}

fn alphabet_for(lang_id: LanguageId) -> (&'static str, &'static str) {
    match lang_id {
        LanguageId::English => (ALPHABET_ENGLISH_1, ALPHABET_ENGLISH_2),
        LanguageId::Russian => (ALPHABET_RUSSIAN_1, ALPHABET_RUSSIAN_2),
        LanguageId::German => (ALPHABET_GERMAN_1, ALPHABET_GERMAN_2),
        LanguageId::French => (ALPHABET_FRENCH_1, ALPHABET_FRENCH_2),
        LanguageId::Italian => (ALPHABET_ITALIAN_1, ALPHABET_ITALIAN_2),
        LanguageId::Spanish => (ALPHABET_SPANISH_1, ALPHABET_SPANISH_2),
        LanguageId::Portuguese => (ALPHABET_PORTUGUESE_1, ALPHABET_PORTUGUESE_2),
        LanguageId::Unknown => crate::exit!("unknown language"),
    }
}

/// `String Language::GetLettersList(LanguageId lang_id)`.
pub fn get_letters_list_by_id(lang_id: LanguageId) -> String {
    let (upper, lower) = alphabet_for(lang_id);
    let mut ret = String::new();
    ret += upper;
    ret += lower;
    string_remove_duplicates(&ret)
}

/// `String Language::GetLettersList(const String& id)`.
pub fn get_letters_list(id: &String) -> String {
    get_letters_list_by_id(get_id(id))
}

/// `String Language::GetNumericList(const String& /*id*/)`. The `id`
/// parameter is unused in the original too (every language shares the same
/// digit list); kept for API parity.
pub fn get_numeric_list(_id: &String) -> String {
    let mut ret = String::new();
    ret += LIST_NUMERIC;
    string_remove_duplicates(&ret)
}

/// `String Language::GetPunctuationList(const String& id)`.
pub fn get_punctuation_list(id: &String) -> String {
    let mut ret = String::new();
    ret += LIST_PUNCTUATION;

    if get_id(id) == LanguageId::Spanish {
        ret += LIST_PUNCTUATION_2;
    }

    string_remove_duplicates(&ret)
}

/// `String Language::GetCharList(const String& id)`.
pub fn get_char_list(id: &String) -> String {
    let mut ret = String::new();
    ret += LIST_NUMERIC;
    ret += LIST_PUNCTUATION;

    match get_id(id) {
        LanguageId::English => {
            ret += ALPHABET_ENGLISH_1;
            ret += ALPHABET_ENGLISH_2;
        }
        LanguageId::Russian => {
            ret += ALPHABET_RUSSIAN_1;
            ret += ALPHABET_RUSSIAN_2;
        }
        LanguageId::German => {
            ret += ALPHABET_GERMAN_1;
            ret += ALPHABET_GERMAN_2;
        }
        LanguageId::French => {
            ret += ALPHABET_FRENCH_1;
            ret += ALPHABET_FRENCH_2;
        }
        LanguageId::Italian => {
            ret += ALPHABET_ITALIAN_1;
            ret += ALPHABET_ITALIAN_2;
        }
        LanguageId::Spanish => {
            ret += LIST_PUNCTUATION_2;
            ret += ALPHABET_SPANISH_1;
            ret += ALPHABET_SPANISH_2;
        }
        LanguageId::Portuguese => {
            ret += ALPHABET_PORTUGUESE_1;
            ret += ALPHABET_PORTUGUESE_2;
        }
        LanguageId::Unknown => crate::exit!("unknown language"),
    }

    string_remove_duplicates(&ret)
}

/// `String Language::GetCharListAll()`.
pub fn get_char_list_all() -> String {
    let mut ret = String::new();
    ret += LIST_NUMERIC;
    ret += LIST_PUNCTUATION;
    ret += LIST_PUNCTUATION_2;
    ret += ALPHABET_ENGLISH_1;
    ret += ALPHABET_ENGLISH_2;
    ret += ALPHABET_RUSSIAN_1;
    ret += ALPHABET_RUSSIAN_2;
    ret += ALPHABET_GERMAN_1;
    ret += ALPHABET_GERMAN_2;
    ret += ALPHABET_FRENCH_1;
    ret += ALPHABET_FRENCH_2;
    ret += ALPHABET_ITALIAN_1;
    ret += ALPHABET_ITALIAN_2;
    ret += ALPHABET_SPANISH_1;
    ret += ALPHABET_SPANISH_2;
    ret += ALPHABET_PORTUGUESE_1;
    ret += ALPHABET_PORTUGUESE_2;

    string_remove_duplicates(&ret)
}

/// `String Language::GetNameOfMonth(int month, LanguageId lang_id)`.
pub fn get_name_of_month(month: i32, lang_id: LanguageId) -> String {
    crate::exit_if!(!(1..=12).contains(&month));
    match lang_id {
        LanguageId::English => String::from(MONTH_ENGLISH[(month - 1) as usize]),
        LanguageId::Russian => String::from(MONTH_RUSSIAN[(month - 1) as usize]),
        _ => crate::exit!("unknown language"),
    }
}

/// `String Language::GetNameOfMonthShort(int month, LanguageId lang_id)`.
pub fn get_name_of_month_short(month: i32, lang_id: LanguageId) -> String {
    crate::exit_if!(!(1..=12).contains(&month));
    match lang_id {
        LanguageId::English => String::from(SHORT_MONTH_ENGLISH[(month - 1) as usize]),
        LanguageId::Russian => String::from(SHORT_MONTH_RUSSIAN[(month - 1) as usize]),
        _ => crate::exit!("unknown language"),
    }
}

/// `String Language::GetNameOfDay(int day, LanguageId lang_id)`.
pub fn get_name_of_day(day: i32, lang_id: LanguageId) -> String {
    crate::exit_if!(!(1..=7).contains(&day));
    match lang_id {
        LanguageId::English => String::from(DAY_ENGLISH[(day - 1) as usize]),
        LanguageId::Russian => String::from(DAY_RUSSIAN[(day - 1) as usize]),
        _ => crate::exit!("unknown language"),
    }
}

/// `String Language::GetNameOfDayShort(int day, LanguageId lang_id)`.
pub fn get_name_of_day_short(day: i32, lang_id: LanguageId) -> String {
    crate::exit_if!(!(1..=7).contains(&day));
    match lang_id {
        LanguageId::English => String::from(SHORT_DAY_ENGLISH[(day - 1) as usize]),
        LanguageId::Russian => String::from(SHORT_DAY_RUSSIAN[(day - 1) as usize]),
        _ => crate::exit!("unknown language"),
    }
}

/// `LanguageId Language::GetId(const String& id)`.
pub fn get_id(id: &String) -> LanguageId {
    lang_map().get(id).copied().unwrap_or(LanguageId::Unknown)
}

/// `StringList Language::GetLanguages()`.
pub fn get_languages() -> StringList {
    let mut list = StringList::new();
    for key in lang_map().keys() {
        list.add(key.clone());
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_idempotent_and_optional() {
        // Calling Init() explicitly (parity with ported call sites) must not
        // panic, and must not change behavior vs. never calling it.
        init();
        init();
        assert_eq!(get_id(&String::from("en")), LanguageId::English);
    }

    #[test]
    fn get_id_maps_known_codes() {
        assert_eq!(get_id(&String::from("de")), LanguageId::German);
        assert_eq!(get_id(&String::from("en")), LanguageId::English);
        assert_eq!(get_id(&String::from("fr")), LanguageId::French);
        assert_eq!(get_id(&String::from("it")), LanguageId::Italian);
        assert_eq!(get_id(&String::from("pt")), LanguageId::Portuguese);
        assert_eq!(get_id(&String::from("ru")), LanguageId::Russian);
        assert_eq!(get_id(&String::from("es")), LanguageId::Spanish);
    }

    #[test]
    fn get_id_unknown_code_is_unknown() {
        assert_eq!(get_id(&String::from("xx")), LanguageId::Unknown);
        assert_eq!(get_id(&String::from("")), LanguageId::Unknown);
    }

    #[test]
    fn get_languages_has_all_seven_codes_regardless_of_order() {
        let langs = get_languages();
        assert_eq!(langs.size(), 7);
        for code in ["de", "en", "fr", "it", "pt", "ru", "es"] {
            assert!(langs.contains(&String::from(code), crate::string::Case::Sensitive));
        }
    }

    #[test]
    fn get_letters_list_by_id_dedupes_and_sorts() {
        // English upper+lower has no overlap, so nothing is deduped; ASCII
        // sort puts every uppercase letter before every lowercase letter.
        let letters = get_letters_list_by_id(LanguageId::English);
        assert_eq!(letters.to_string(), "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
    }

    #[test]
    fn get_letters_list_by_code_delegates_to_get_id() {
        assert_eq!(get_letters_list(&String::from("en")).to_string(), get_letters_list_by_id(LanguageId::English).to_string());
    }

    #[test]
    #[should_panic(expected = "unknown language")]
    fn get_letters_list_by_id_unknown_panics() {
        let _ = get_letters_list_by_id(LanguageId::Unknown);
    }

    #[test]
    fn get_numeric_list_is_sorted_digits() {
        assert_eq!(get_numeric_list(&String::from("en")).to_string(), "0123456789");
        // The `id` argument is unused in the original; any code (or an
        // unknown one) yields the exact same result.
        assert_eq!(get_numeric_list(&String::from("xx")).to_string(), "0123456789");
    }

    #[test]
    fn get_punctuation_list_adds_spanish_extras_only_for_spanish() {
        let en = get_punctuation_list(&String::from("en"));
        let es = get_punctuation_list(&String::from("es"));
        assert!(!en.contains_char('¡', crate::string::Case::Sensitive));
        assert!(es.contains_char('¡', crate::string::Case::Sensitive));
        assert!(es.contains_char('¿', crate::string::Case::Sensitive));
        assert!(es.contains_char('!', crate::string::Case::Sensitive));
    }

    #[test]
    fn get_char_list_combines_numeric_punctuation_and_alphabet() {
        let en = get_char_list(&String::from("en"));
        assert!(en.contains_char('5', crate::string::Case::Sensitive));
        assert!(en.contains_char('!', crate::string::Case::Sensitive));
        assert!(en.contains_char('A', crate::string::Case::Sensitive));
        assert!(en.contains_char('z', crate::string::Case::Sensitive));
        assert!(!en.contains_char('Ä', crate::string::Case::Sensitive));

        let de = get_char_list(&String::from("de"));
        assert!(de.contains_char('Ä', crate::string::Case::Sensitive));
        assert!(de.contains_char('ß', crate::string::Case::Sensitive));
    }

    #[test]
    fn get_char_list_spanish_includes_extra_punctuation() {
        let es = get_char_list(&String::from("es"));
        assert!(es.contains_char('¿', crate::string::Case::Sensitive));
        assert!(es.contains_char('Ñ', crate::string::Case::Sensitive));
    }

    #[test]
    #[should_panic(expected = "unknown language")]
    fn get_char_list_unknown_code_panics() {
        let _ = get_char_list(&String::from("xx"));
    }

    #[test]
    fn get_char_list_all_contains_every_language() {
        let all = get_char_list_all();
        assert!(all.contains_char('a', crate::string::Case::Sensitive));
        assert!(all.contains_char('А', crate::string::Case::Sensitive)); // Cyrillic A
        assert!(all.contains_char('ß', crate::string::Case::Sensitive));
        assert!(all.contains_char('ç', crate::string::Case::Sensitive));
        assert!(all.contains_char('ñ', crate::string::Case::Sensitive));
        assert!(all.contains_char('¡', crate::string::Case::Sensitive));
        assert!(all.contains_char('0', crate::string::Case::Sensitive));
    }

    #[test]
    fn get_name_of_month_english_and_russian() {
        assert_eq!(get_name_of_month(1, LanguageId::English).to_string(), "January");
        assert_eq!(get_name_of_month(12, LanguageId::English).to_string(), "December");
        assert_eq!(get_name_of_month(1, LanguageId::Russian).to_string(), "Январь");
    }

    #[test]
    fn get_name_of_month_short_english_and_russian() {
        assert_eq!(get_name_of_month_short(7, LanguageId::English).to_string(), "Jul");
        assert_eq!(get_name_of_month_short(7, LanguageId::Russian).to_string(), "Июл");
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn get_name_of_month_out_of_range_panics() {
        let _ = get_name_of_month(13, LanguageId::English);
    }

    #[test]
    #[should_panic(expected = "unknown language")]
    fn get_name_of_month_german_panics() {
        // Kyty's own table only ever covered English + Russian; every other
        // language hits EXIT("unknown language").
        let _ = get_name_of_month(1, LanguageId::German);
    }

    #[test]
    fn get_name_of_day_english_and_russian() {
        assert_eq!(get_name_of_day(1, LanguageId::English).to_string(), "Monday");
        assert_eq!(get_name_of_day(7, LanguageId::English).to_string(), "Sunday");
        assert_eq!(get_name_of_day(1, LanguageId::Russian).to_string(), "Понедельник");
    }

    #[test]
    fn get_name_of_day_short_english_and_russian() {
        assert_eq!(get_name_of_day_short(1, LanguageId::English).to_string(), "Mon");
        assert_eq!(get_name_of_day_short(1, LanguageId::Russian).to_string(), "Пнд");
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn get_name_of_day_out_of_range_panics() {
        let _ = get_name_of_day(0, LanguageId::English);
    }
}
