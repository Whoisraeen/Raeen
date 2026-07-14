//! Port of Kyty's `Core/DateTime` (`reference/kyty/source/include/Kyty/Core/DateTime.h`
//! + `lib/Core/src/DateTime.cpp`).
//!
//! Three value types, exactly as in Kyty:
//! - [`Date`] — a proleptic-Gregorian calendar date stored as a Julian Day
//!   Number (`jd_t` -> [`Jd`] = `i32`).
//! - [`Time`] — a time-of-day stored as milliseconds since midnight.
//! - [`DateTime`] — a `(Date, Time)` pair.
//!
//! std mapping used by this port:
//! - `Kyty::Core::String` (UTF-32) -> `std::string::String` / `&str`. The
//!   `String` module itself is not ported yet, so `ToString()` here returns a
//!   plain `String` and format strings are plain `&str` (all format specifiers
//!   are ASCII, so byte/char indexing is equivalent to Kyty's code-point
//!   indexing).
//! - `Math::floordiv<jd_t>` (`Kyty/Math/MathAll.h`) -> the private `floordiv`
//!   free function below, ported verbatim (the Math module itself is a
//!   separate, not-yet-ported subsystem; this one function is trivial
//!   arithmetic with no allocation/pointer content worth pulling in a whole
//!   module for).
//! - `Kyty::Sys::sys_get_system_time[_utc]` (`Kyty/Sys/SysTimer.h`) -> the
//!   `Sys` layer is not ported yet, so `from_system`/`from_system_utc` here
//!   are both implemented directly against `std::time::SystemTime`. Rust's
//!   std has no cross-platform local-timezone API, so — pending a ported
//!   `Sys` timer module — `from_system` currently behaves identically to
//!   `from_system_utc` (UTC); this is a documented, intentional gap, not a
//!   silent behavior change for the *format specifiers and calendar math*
//!   this module owns.
//! - `Kyty::Core::LanguageId` comes from the ported [`crate::language`]
//!   subsystem ([`LanguageId`] is re-exported here for call-site convenience).
//!   The private month/day name tables below remain local to this module,
//!   covering exactly what `Date`/`DateTime::ToString` need (English +
//!   Russian, matching what Kyty's `Language.cpp` actually implements — every
//!   other enumerator is `EXIT("unknown language")` in the original too).

use crate::language::LanguageId;
use std::time::{SystemTime, UNIX_EPOCH};

/// `jd_t` — a Julian Day Number.
pub type Jd = i32;

pub const DATE_JD_INVALID: Jd = i32::MIN;
pub const TIME_MS_INVALID: i32 = -1;
pub const TIME_MS_IN_DAY: i32 = 24 * 3600 * 1000;

pub const MONTH_JANUARY: i32 = 1;
pub const MONTH_FEBRUARY: i32 = 2;
pub const MONTH_MARCH: i32 = 3;
pub const MONTH_APRIL: i32 = 4;
pub const MONTH_MAY: i32 = 5;
pub const MONTH_JUNE: i32 = 6;
pub const MONTH_JULY: i32 = 7;
pub const MONTH_AUGUST: i32 = 8;
pub const MONTH_SEPTEMBER: i32 = 9;
pub const MONTH_OCTOBER: i32 = 10;
pub const MONTH_NOVEMBER: i32 = 11;
pub const MONTH_DECEMBER: i32 = 12;

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

fn name_of_month(month: i32, lang_id: LanguageId) -> &'static str {
    crate::exit_if!(!(1..=12).contains(&month));
    match lang_id {
        LanguageId::English => MONTH_ENGLISH[(month - 1) as usize],
        LanguageId::Russian => MONTH_RUSSIAN[(month - 1) as usize],
        _ => crate::exit!("unknown language"),
    }
}

fn name_of_month_short(month: i32, lang_id: LanguageId) -> &'static str {
    crate::exit_if!(!(1..=12).contains(&month));
    match lang_id {
        LanguageId::English => SHORT_MONTH_ENGLISH[(month - 1) as usize],
        LanguageId::Russian => SHORT_MONTH_RUSSIAN[(month - 1) as usize],
        _ => crate::exit!("unknown language"),
    }
}

fn name_of_day(day: i32, lang_id: LanguageId) -> &'static str {
    crate::exit_if!(!(1..=7).contains(&day));
    match lang_id {
        LanguageId::English => DAY_ENGLISH[(day - 1) as usize],
        LanguageId::Russian => DAY_RUSSIAN[(day - 1) as usize],
        _ => crate::exit!("unknown language"),
    }
}

fn name_of_day_short(day: i32, lang_id: LanguageId) -> &'static str {
    crate::exit_if!(!(1..=7).contains(&day));
    match lang_id {
        LanguageId::English => SHORT_DAY_ENGLISH[(day - 1) as usize],
        LanguageId::Russian => SHORT_DAY_RUSSIAN[(day - 1) as usize],
        _ => crate::exit!("unknown language"),
    }
}

/// Port of `Math::floordiv<jd_t>` (`Kyty/Math/MathAll.h`): integer division
/// that rounds towards negative infinity (unlike Rust/C++ `/`, which
/// truncates towards zero).
fn floordiv(a: Jd, b: Jd) -> Jd {
    let (mut a, mut b) = (a, b);
    if b < 0 {
        if a < 0 {
            return a / b;
        }
        b = -b;
        a = -a;
    }
    (a - if a < 0 { b - 1 } else { 0 }) / b
}

/// Port of the free function `ymd_to_jd` in `DateTime.cpp` (the
/// Fliegel & Van Flandern algorithm).
fn ymd_to_jd(year: i32, month: i32, day: i32) -> Jd {
    let mut t_year = year;
    let t_month = month;
    let t_day = day;

    if t_year < 0 {
        t_year += 1;
    }

    let a = floordiv(14 - t_month, 12);
    let y = t_year + 4800 - a;
    let m = t_month + 12 * a - 3;
    t_day + floordiv(153 * m + 2, 5) + 365 * y + floordiv(y, 4) - floordiv(y, 100)
        + floordiv(y, 400)
        - 32045
}

/// Port of the free function `jd_to_ymd` in `DateTime.cpp`.
fn jd_to_ymd(jd: Jd) -> (i32, i32, i32) {
    let a = jd + 32044;
    let b = floordiv(4 * a + 3, 146097);
    let c = a - floordiv(146097 * b, 4);
    let d = floordiv(4 * c + 3, 1461);
    let e = c - floordiv(1461 * d, 4);
    let m = floordiv(5 * e + 2, 153);

    let t_day = e - floordiv(153 * m + 2, 5) + 1;
    let t_month = m + 3 - 12 * floordiv(m, 10);
    let mut t_year = 100 * b + d - 4800 + floordiv(m, 10);

    if t_year <= 0 {
        t_year -= 1;
    }

    (t_year, t_month, t_day)
}

fn hms_to_ms(hour: i32, minute: i32, second: i32, msec: i32) -> i32 {
    hour * 60 * 60 * 1000 + minute * 60 * 1000 + second * 1000 + msec
}

fn ms_to_hms(ms: i32) -> (i32, i32, i32, i32) {
    let hour = ms / (60 * 60 * 1000);
    let minute = (ms % (60 * 60 * 1000)) / (60 * 1000);
    let second = (ms / 1000) % 60;
    let msec = ms % 1000;
    (hour, minute, second, msec)
}

/// `Kyty::Core::Date`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    jd: Jd,
}

impl Default for Date {
    fn default() -> Self {
        Date {
            jd: DATE_JD_INVALID,
        }
    }
}

impl Date {
    /// `explicit Date(jd_t d)`.
    #[must_use]
    pub fn new(jd: Jd) -> Self {
        Date { jd }
    }

    /// `explicit Date(int year, int month, int day)`; month in `[1..12]`, day in `[1..31]`.
    #[must_use]
    pub fn from_ymd(year: i32, month: i32, day: i32) -> Self {
        let mut d = Date::default();
        d.set(year, month, day);
        d
    }

    /// `Date::FromSystem()`. See the module doc comment: pending a ported
    /// `Sys` timer module, this is identical to [`Date::from_system_utc`].
    #[must_use]
    pub fn from_system() -> Self {
        Self::from_system_utc()
    }

    /// `Date::FromSystemUTC()`.
    #[must_use]
    pub fn from_system_utc() -> Self {
        let (y, m, d, ..) = now_utc_ymd_hms();
        Date::from_ymd(y, m, d)
    }

    /// `Date::FromMacros(const String& date)` — parses a `__DATE__`-style
    /// string, e.g. `"Jul 12 2026"`.
    #[must_use]
    pub fn from_macros(date: &str) -> Self {
        const MONTHS: [&str; 12] = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];

        let parts: Vec<&str> = date.split_whitespace().collect();
        if parts.len() != 3 {
            return Date::default();
        }

        // `String::EqualAscii` (no "NoCase" suffix) is a case-sensitive exact
        // match in Kyty, matching `__DATE__`'s fixed "Jan".."Dec" casing.
        let month = match MONTHS.iter().position(|m| parts[0] == *m) {
            Some(idx) => (idx + 1) as i32,
            None => return Date::default(),
        };

        let day: i32 = match parts[1].parse() {
            Ok(v) => v,
            Err(_) => return Date::default(),
        };
        let year: i32 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => return Date::default(),
        };

        Date::from_ymd(year, month, day)
    }

    /// `void Set(int year, int month, int day)`.
    pub fn set(&mut self, year: i32, month: i32, day: i32) {
        if Date::is_valid(year, month, day) {
            self.jd = ymd_to_jd(year, month, day);
        } else {
            self.jd = DATE_JD_INVALID;
        }
    }

    /// `void Set(jd_t jd)`.
    pub fn set_jd(&mut self, jd: Jd) {
        self.jd = jd;
    }

    /// `void Get(int* year, int* month, int* day) const` — Rust has no output
    /// parameters, so this returns all three (`0, 0, 0` when invalid, exactly
    /// as the original does for every null-checked-out pointer it was given).
    #[must_use]
    pub fn get(&self) -> (i32, i32, i32) {
        if self.is_invalid() {
            (0, 0, 0)
        } else {
            jd_to_ymd(self.jd)
        }
    }

    #[must_use]
    pub fn is_invalid(&self) -> bool {
        self.jd == DATE_JD_INVALID
    }

    #[must_use]
    pub fn days_in_month(&self) -> i32 {
        if self.is_invalid() {
            return 0;
        }
        let (year, month, _) = jd_to_ymd(self.jd);
        if month == 2 && Date::is_leap_year_static(year) {
            return 29;
        }
        Date::days_in_month_static(month)
    }

    #[must_use]
    pub fn days_in_year(&self) -> i32 {
        if self.is_invalid() {
            return 0;
        }
        if self.is_leap_year() { 366 } else { 365 }
    }

    #[must_use]
    pub fn is_leap_year(&self) -> bool {
        if self.is_invalid() {
            return false;
        }
        let (year, _, _) = jd_to_ymd(self.jd);
        Date::is_leap_year_static(year)
    }

    #[must_use]
    pub fn year(&self) -> i32 {
        if self.is_invalid() {
            return 0;
        }
        jd_to_ymd(self.jd).0
    }

    #[must_use]
    pub fn month(&self) -> i32 {
        if self.is_invalid() {
            return 0;
        }
        jd_to_ymd(self.jd).1
    }

    #[must_use]
    pub fn day(&self) -> i32 {
        if self.is_invalid() {
            return 0;
        }
        jd_to_ymd(self.jd).2
    }

    #[must_use]
    pub fn julian_day(&self) -> Jd {
        self.jd
    }

    #[must_use]
    pub fn day_of_week(&self) -> i32 {
        if self.is_invalid() {
            return 0;
        }
        if self.jd >= 0 {
            (self.jd % 7) + 1
        } else {
            ((self.jd + 1) % 7) + 7
        }
    }

    #[must_use]
    pub fn day_of_year(&self) -> i32 {
        if self.is_invalid() {
            return 0;
        }
        let d = ymd_to_jd(self.year(), 1, 1);
        self.jd - d + 1
    }

    #[must_use]
    pub fn quarter_of_year(&self) -> i32 {
        if self.is_invalid() {
            return 0;
        }
        (self.month() - 1) / 3 + 1
    }

    /// `String Date::ToString(const char* format, LanguageId lang_id) const`.
    ///
    /// ```text
    /// YYYY       - 4-digit year
    /// YYY,YY,Y   - Last 3, 2, or 1 digit(s) of year.
    /// Q          - Quarter of year (1, 2, 3, 4; JAN-MAR = 1).
    /// MM         - Month (01-12; JAN = 01).
    /// MON        - Abbreviated name of month.
    /// MONTH      - Name of month
    /// D          - Day of week (1-7).
    /// DAY        - Name of day.
    /// DY         - Abbreviated name of day.
    /// DD         - Day of month (01-31).
    /// DDD        - Day of year (001-366).
    /// J          - Julian day
    /// ```
    #[must_use]
    pub fn to_string(&self, format: &str, lang_id: LanguageId) -> String {
        if self.is_invalid() {
            return String::new();
        }
        let chars: Vec<char> = format.chars().collect();
        let mut out = String::new();
        let mut i = 0usize;
        while i < chars.len() {
            let consumed = format_date(self, &chars, i, &mut out, lang_id);
            if consumed == 0 {
                out.push(chars[i]);
                i += 1;
            } else {
                i += consumed;
            }
        }
        out
    }

    #[must_use]
    pub fn is_valid(year: i32, month: i32, day: i32) -> bool {
        if year == 0 {
            return false;
        }
        (day >= 1)
            && (day <= Date::days_in_month_static(month)
                || (day == 29 && month == 2 && Date::is_leap_year_static(year)))
    }

    #[must_use]
    pub fn days_in_month_static(month: i32) -> i32 {
        const DAYS: [i32; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let m = if !(1..=12).contains(&month) { 0 } else { month };
        DAYS[m as usize]
    }

    #[must_use]
    pub fn days_in_year_static(year: i32) -> i32 {
        if Date::is_leap_year_static(year) {
            366
        } else {
            365
        }
    }

    #[must_use]
    pub fn is_leap_year_static(year: i32) -> bool {
        let year = if year < 1 { year + 1 } else { year };
        (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
    }
}

impl std::ops::Add<i32> for Date {
    type Output = Date;
    fn add(self, days: i32) -> Date {
        Date::new(self.jd + days)
    }
}

impl std::ops::Sub<i32> for Date {
    type Output = Date;
    fn sub(self, days: i32) -> Date {
        Date::new(self.jd - days)
    }
}

impl std::ops::AddAssign<i32> for Date {
    fn add_assign(&mut self, days: i32) {
        self.jd += days;
    }
}

impl std::ops::SubAssign<i32> for Date {
    fn sub_assign(&mut self, days: i32) {
        self.jd -= days;
    }
}

/// `format_date()` free function in `DateTime.cpp`. Returns the number of
/// `chars` consumed (0 means "no format specifier matched here").
fn format_date(d: &Date, chars: &[char], i: usize, out: &mut String, lang_id: LanguageId) -> usize {
    let matches = |pat: &str| -> bool {
        let pc: Vec<char> = pat.chars().collect();
        i + pc.len() <= chars.len() && chars[i..i + pc.len()] == pc[..]
    };

    if matches("YYYY") {
        let y = d.year();
        crate::exit_if!(!(0..=9999).contains(&y));
        out.push_str(&format!("{y:04}"));
        4
    } else if matches("YYY") {
        let y = d.year();
        crate::exit_if!(!(0..=9999).contains(&y));
        out.push_str(&format!("{y:04}")[1..]);
        3
    } else if matches("YY") {
        let y = d.year();
        crate::exit_if!(!(0..=9999).contains(&y));
        out.push_str(&format!("{y:04}")[2..]);
        2
    } else if matches("Y") {
        let y = d.year();
        crate::exit_if!(!(0..=9999).contains(&y));
        out.push_str(&format!("{y:04}")[3..]);
        1
    } else if matches("Q") {
        out.push_str(&format!("{}", d.quarter_of_year()));
        1
    } else if matches("MM") {
        out.push_str(&format!("{:02}", d.month()));
        2
    } else if matches("MONTH") {
        out.push_str(name_of_month(d.month(), lang_id));
        5
    } else if matches("MON") {
        out.push_str(name_of_month_short(d.month(), lang_id));
        3
    } else if matches("DDD") {
        out.push_str(&format!("{:03}", d.day_of_year()));
        3
    } else if matches("DD") {
        out.push_str(&format!("{:02}", d.day()));
        2
    } else if matches("DY") {
        out.push_str(name_of_day_short(d.day_of_week(), lang_id));
        2
    } else if matches("DAY") {
        out.push_str(name_of_day(d.day_of_week(), lang_id));
        3
    } else if matches("D") {
        out.push_str(&format!("{}", d.day_of_week()));
        1
    } else if matches("J") {
        out.push_str(&format!("{}", d.julian_day()));
        1
    } else {
        0
    }
}

/// `Kyty::Core::Time`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Time {
    ms: i32,
}

impl Default for Time {
    fn default() -> Self {
        Time {
            ms: TIME_MS_INVALID,
        }
    }
}

impl Time {
    /// `explicit Time(int msec)`.
    #[must_use]
    pub fn new(msec: i32) -> Self {
        if !(0..TIME_MS_IN_DAY).contains(&msec) {
            Time {
                ms: TIME_MS_INVALID,
            }
        } else {
            Time { ms: msec }
        }
    }

    /// `explicit Time(int hour24, int minute, int second, int msec = 0)`.
    #[must_use]
    pub fn from_hms(hour24: i32, minute: i32, second: i32, msec: i32) -> Self {
        let mut t = Time::default();
        t.set(hour24, minute, second, msec);
        t
    }

    /// `Time::FromSystem()`. See the module doc comment: identical to
    /// [`Time::from_system_utc`] pending a ported `Sys` timer module.
    #[must_use]
    pub fn from_system() -> Self {
        Self::from_system_utc()
    }

    /// `Time::FromSystemUTC()`.
    #[must_use]
    pub fn from_system_utc() -> Self {
        let (_, _, _, h, m, s, ms) = now_utc_ymd_hms();
        Time::from_hms(h, m, s, ms)
    }

    /// `void Set(int hour24, int minute, int second, int msec = 0)`.
    pub fn set(&mut self, hour24: i32, minute: i32, second: i32, msec: i32) {
        if Time::is_valid(hour24, minute, second, msec) {
            self.ms = hms_to_ms(hour24, minute, second, msec);
        } else {
            self.ms = TIME_MS_INVALID;
        }
    }

    /// `void Set(int msec)`.
    pub fn set_ms(&mut self, msec: i32) {
        self.ms = msec;
    }

    /// `void Get(int* hour24, int* minute, int* second, int* msec = nullptr) const`
    /// — returns all four (`-1, -1, -1, -1` when invalid).
    #[must_use]
    pub fn get(&self) -> (i32, i32, i32, i32) {
        if self.is_invalid() {
            (-1, -1, -1, -1)
        } else {
            ms_to_hms(self.ms)
        }
    }

    #[must_use]
    pub fn is_invalid(&self) -> bool {
        self.ms < 0 || self.ms >= TIME_MS_IN_DAY
    }

    #[must_use]
    pub fn hour12(&self) -> i32 {
        if self.is_invalid() {
            return -1;
        }
        let h = ms_to_hms(self.ms).0 % 12;
        if h == 0 { 12 } else { h }
    }

    #[must_use]
    pub fn hour24(&self) -> i32 {
        if self.is_invalid() {
            return -1;
        }
        ms_to_hms(self.ms).0
    }

    #[must_use]
    pub fn is_am(&self) -> bool {
        if self.is_invalid() {
            return false;
        }
        ms_to_hms(self.ms).0 < 12
    }

    #[must_use]
    pub fn is_pm(&self) -> bool {
        if self.is_invalid() {
            return false;
        }
        ms_to_hms(self.ms).0 >= 12
    }

    #[must_use]
    pub fn minute(&self) -> i32 {
        if self.is_invalid() {
            return -1;
        }
        ms_to_hms(self.ms).1
    }

    #[must_use]
    pub fn second(&self) -> i32 {
        if self.is_invalid() {
            return -1;
        }
        ms_to_hms(self.ms).2
    }

    #[must_use]
    pub fn msec(&self) -> i32 {
        if self.is_invalid() {
            return -1;
        }
        ms_to_hms(self.ms).3
    }

    #[must_use]
    pub fn msec_total(&self) -> i32 {
        self.ms
    }

    /// `String Time::ToString(const char* format) const`.
    ///
    /// ```text
    /// HH    - Hour of day (1-12)
    /// HH12  - Hour of day (1-12)
    /// HH24  - Hour of day (0-23)
    /// MI    - Minute (0-59)
    /// SS    - Second (0-59)
    /// SSSSS - Seconds past midnight (0-86399)
    /// FFF   - Milliseconds
    /// AM    - AM or PM
    /// A.M.  - A.M. or P.M.
    /// ```
    #[must_use]
    pub fn to_string(&self, format: &str) -> String {
        if self.is_invalid() {
            return String::new();
        }
        let chars: Vec<char> = format.chars().collect();
        let mut out = String::new();
        let mut i = 0usize;
        while i < chars.len() {
            let consumed = format_time(self, &chars, i, &mut out);
            if consumed == 0 {
                out.push(chars[i]);
                i += 1;
            } else {
                i += consumed;
            }
        }
        out
    }

    #[must_use]
    pub fn is_valid(hour: i32, minute: i32, second: i32, msec: i32) -> bool {
        (0..=23).contains(&hour)
            && (0..=59).contains(&minute)
            && (0..=59).contains(&second)
            && (0..=999).contains(&msec)
    }
}

fn format_time(t: &Time, chars: &[char], i: usize, out: &mut String) -> usize {
    let matches = |pat: &str| -> bool {
        let pc: Vec<char> = pat.chars().collect();
        i + pc.len() <= chars.len() && chars[i..i + pc.len()] == pc[..]
    };

    if matches("HH24") {
        out.push_str(&format!("{:02}", t.hour24()));
        4
    } else if matches("HH12") {
        out.push_str(&format!("{:02}", t.hour12()));
        4
    } else if matches("HH") {
        out.push_str(&format!("{:02}", t.hour12()));
        2
    } else if matches("MI") {
        out.push_str(&format!("{:02}", t.minute()));
        2
    } else if matches("SSSSS") {
        out.push_str(&format!("{:05}", t.msec_total() / 1000));
        5
    } else if matches("SS") {
        out.push_str(&format!("{:02}", t.second()));
        2
    } else if matches("FFF") {
        out.push_str(&format!("{:03}", t.msec()));
        3
    } else if matches("AM") {
        out.push_str(if t.is_am() { "AM" } else { "PM" });
        2
    } else if matches("A.M.") {
        out.push_str(if t.is_am() { "A.M." } else { "P.M." });
        4
    } else {
        0
    }
}

impl std::ops::Add<i32> for Time {
    type Output = Time;
    fn add(self, secs: i32) -> Time {
        let ms_secs = secs * 1000;
        crate::exit_if!(!(-TIME_MS_IN_DAY..=TIME_MS_IN_DAY).contains(&ms_secs));
        if self.is_invalid() {
            return Time::default();
        }
        let mut r = self.ms + ms_secs;
        if r < 0 {
            r += TIME_MS_IN_DAY;
        }
        if r >= TIME_MS_IN_DAY {
            r -= TIME_MS_IN_DAY;
        }
        Time::new(r)
    }
}

impl std::ops::Sub<i32> for Time {
    type Output = Time;
    fn sub(self, secs: i32) -> Time {
        let ms_secs = secs * 1000;
        crate::exit_if!(!(-TIME_MS_IN_DAY..=TIME_MS_IN_DAY).contains(&ms_secs));
        if self.is_invalid() {
            return Time::default();
        }
        let mut r = self.ms - ms_secs;
        if r < 0 {
            r += TIME_MS_IN_DAY;
        }
        if r >= TIME_MS_IN_DAY {
            r -= TIME_MS_IN_DAY;
        }
        Time::new(r)
    }
}

impl std::ops::AddAssign<i32> for Time {
    fn add_assign(&mut self, secs: i32) {
        let ms_secs = secs * 1000;
        crate::exit_if!(!(-TIME_MS_IN_DAY..=TIME_MS_IN_DAY).contains(&ms_secs));
        if !self.is_invalid() {
            self.ms += ms_secs;
            if self.ms < 0 {
                self.ms += TIME_MS_IN_DAY;
            }
            if self.ms >= TIME_MS_IN_DAY {
                self.ms -= TIME_MS_IN_DAY;
            }
        }
    }
}

impl std::ops::SubAssign<i32> for Time {
    fn sub_assign(&mut self, secs: i32) {
        let ms_secs = secs * 1000;
        crate::exit_if!(!(-TIME_MS_IN_DAY..=TIME_MS_IN_DAY).contains(&ms_secs));
        if !self.is_invalid() {
            self.ms -= ms_secs;
            if self.ms < 0 {
                self.ms += TIME_MS_IN_DAY;
            }
            if self.ms >= TIME_MS_IN_DAY {
                self.ms -= TIME_MS_IN_DAY;
            }
        }
    }
}

/// `Kyty::Core::DateTime`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DateTime {
    date: Date,
    time: Time,
}

impl DateTime {
    /// `explicit DateTime(const Date& d)`.
    #[must_use]
    pub fn from_date(d: Date) -> Self {
        DateTime {
            date: d,
            time: Time::default(),
        }
    }

    /// `explicit DateTime(const Time& t)`.
    #[must_use]
    pub fn from_time(t: Time) -> Self {
        DateTime {
            date: Date::default(),
            time: t,
        }
    }

    /// `explicit DateTime(const Date& d, const Time& t)` (and the
    /// argument-order-swapped overload — Rust has named parameters, so one
    /// function covers both).
    #[must_use]
    pub fn from_date_time(d: Date, t: Time) -> Self {
        DateTime { date: d, time: t }
    }

    /// `DateTime::FromSystem()`. See the module doc comment: identical to
    /// [`DateTime::from_system_utc`] pending a ported `Sys` timer module.
    #[must_use]
    pub fn from_system() -> Self {
        Self::from_system_utc()
    }

    /// `DateTime::FromSystemUTC()`.
    #[must_use]
    pub fn from_system_utc() -> Self {
        let (y, mo, d, h, mi, s, ms) = now_utc_ymd_hms();
        DateTime::from_date_time(Date::from_ymd(y, mo, d), Time::from_hms(h, mi, s, ms))
    }

    /// `DateTime::FromSQLiteJulian(double jd)`.
    #[must_use]
    pub fn from_sqlite_julian(jd: f64) -> Self {
        let shifted = jd + 0.5;
        let i = shifted.trunc();
        let f = shifted - i;
        DateTime::from_date_time(
            Date::new(i as Jd),
            Time::new((f * f64::from(TIME_MS_IN_DAY)) as i32),
        )
    }

    /// `double DateTime::ToSQLiteJulian() const`.
    #[must_use]
    pub fn to_sqlite_julian(&self) -> f64 {
        -0.5 + f64::from(self.date.julian_day())
            + f64::from(self.time.msec_total()) / f64::from(TIME_MS_IN_DAY)
    }

    /// `int64_t DateTime::ToSQLiteJulianInt64() const`.
    #[must_use]
    pub fn to_sqlite_julian_int64(&self) -> i64 {
        -(i64::from(TIME_MS_IN_DAY)) / 2
            + i64::from(self.date.julian_day()) * i64::from(TIME_MS_IN_DAY)
            + i64::from(self.time.msec_total())
    }

    /// `DateTime::FromUnix(double seconds)`.
    #[must_use]
    pub fn from_unix(seconds: f64) -> Self {
        Self::from_sqlite_julian(2_440_587.5 + seconds / 86400.0)
    }

    /// `double DateTime::ToUnix() const`.
    #[must_use]
    pub fn to_unix(&self) -> f64 {
        (self.to_sqlite_julian() - 2_440_587.5) * 86400.0
    }

    /// `uint64_t DateTime::DistanceMs(const DateTime& other) const`.
    #[must_use]
    pub fn distance_ms(&self, other: &DateTime) -> u64 {
        crate::exit_if!(self.is_invalid() || other.is_invalid());

        if *other == *self {
            return 0;
        }
        if *other > *self {
            return other.distance_ms(self);
        }

        let j1 = i64::from(other.date.julian_day());
        let j2 = i64::from(self.date.julian_day());

        ((j2 - j1 - 1) * i64::from(TIME_MS_IN_DAY)
            + (i64::from(TIME_MS_IN_DAY) - i64::from(other.time.msec_total()))
            + i64::from(self.time.msec_total())) as u64
    }

    #[must_use]
    pub fn is_invalid(&self) -> bool {
        self.date.is_invalid() || self.time.is_invalid()
    }

    pub fn set_date(&mut self, d: Date) {
        self.date = d;
    }

    pub fn set_time(&mut self, t: Time) {
        self.time = t;
    }

    #[must_use]
    pub fn get_date(&self) -> Date {
        self.date
    }

    pub fn get_date_mut(&mut self) -> &mut Date {
        &mut self.date
    }

    #[must_use]
    pub fn get_time(&self) -> Time {
        self.time
    }

    pub fn get_time_mut(&mut self) -> &mut Time {
        &mut self.time
    }

    /// `String DateTime::ToString(const char* format, LanguageId lang_id) const`
    /// — combines [`Date`]'s and [`Time`]'s format specifiers (see their
    /// `to_string` doc comments).
    #[must_use]
    pub fn to_string(&self, format: &str, lang_id: LanguageId) -> String {
        if self.is_invalid() {
            return String::new();
        }
        let chars: Vec<char> = format.chars().collect();
        let mut out = String::new();
        let mut i = 0usize;
        while i < chars.len() {
            let mut consumed = format_date(&self.date, &chars, i, &mut out, lang_id);
            if consumed == 0 {
                consumed = format_time(&self.time, &chars, i, &mut out);
            }
            if consumed == 0 {
                out.push(chars[i]);
                i += 1;
            } else {
                i += consumed;
            }
        }
        out
    }
}

impl PartialOrd for DateTime {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DateTime {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `operator<`: m_date < other.m_date || (m_date == other.m_date && m_time < other.m_time)
        self.date.cmp(&other.date).then(self.time.cmp(&other.time))
    }
}

/// Shared helper backing `from_system`/`from_system_utc` on all three types:
/// current UTC wall-clock time decomposed the same way Kyty's `SysTimeStruct`
/// is, via this module's own `jd_to_ymd`/`ms_to_hms` so the result is
/// self-consistent with `Date`/`Time`'s own calendar math.
fn now_utc_ymd_hms() -> (i32, i32, i32, i32, i32, i32, i32) {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let epoch_jd = ymd_to_jd(1970, 1, 1);
    let total_secs = dur.as_secs() as i64;
    let days = total_secs.div_euclid(86400);
    let secs_of_day = total_secs.rem_euclid(86400) as i32;
    let ms_of_day = secs_of_day * 1000 + (dur.subsec_millis() as i32);
    let jd = epoch_jd + days as Jd;
    let (y, mo, d) = jd_to_ymd(jd);
    let (h, mi, s, ms) = ms_to_hms(ms_of_day);
    (y, mo, d, h, mi, s, ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_from_ymd_roundtrips() {
        let d = Date::from_ymd(2026, 7, 12);
        assert!(!d.is_invalid());
        assert_eq!(d.get(), (2026, 7, 12));
        assert_eq!(d.year(), 2026);
        assert_eq!(d.month(), 7);
        assert_eq!(d.day(), 12);
    }

    #[test]
    fn date_invalid_month_day_combo() {
        let d = Date::from_ymd(2026, 2, 30);
        assert!(d.is_invalid());
        assert_eq!(d.get(), (0, 0, 0));
        assert_eq!(d.year(), 0);
    }

    #[test]
    fn date_year_zero_is_invalid() {
        assert!(!Date::is_valid(0, 1, 1));
        assert!(Date::from_ymd(0, 1, 1).is_invalid());
    }

    #[test]
    fn leap_year_rules() {
        assert!(Date::is_leap_year_static(2000));
        assert!(!Date::is_leap_year_static(1900));
        assert!(Date::is_leap_year_static(2024));
        assert!(!Date::is_leap_year_static(2023));
        // Confirms Feb 29 2024 actually constructs as a valid date.
        assert!(!Date::from_ymd(2024, 2, 29).is_invalid());
    }

    #[test]
    fn days_in_month_and_year() {
        let d = Date::from_ymd(2024, 2, 1);
        assert_eq!(d.days_in_month(), 29);
        assert_eq!(d.days_in_year(), 366);

        let d = Date::from_ymd(2023, 2, 1);
        assert_eq!(d.days_in_month(), 28);
        assert_eq!(d.days_in_year(), 365);

        assert_eq!(Date::days_in_month_static(4), 30);
        assert_eq!(Date::days_in_month_static(13), 0);

        assert_eq!(Date::days_in_year_static(2024), 366);
        assert_eq!(Date::days_in_year_static(2023), 365);
    }

    #[test]
    fn day_of_week_matches_known_date() {
        // 2026-07-12 is a Sunday.
        let d = Date::from_ymd(2026, 7, 12);
        assert_eq!(d.day_of_week(), 7);
        // 2026-07-13 is a Monday.
        let d2 = Date::from_ymd(2026, 7, 13);
        assert_eq!(d2.day_of_week(), 1);
    }

    #[test]
    fn day_of_year_and_quarter() {
        let d = Date::from_ymd(2026, 1, 1);
        assert_eq!(d.day_of_year(), 1);
        assert_eq!(d.quarter_of_year(), 1);

        let d = Date::from_ymd(2026, 12, 31);
        assert_eq!(d.day_of_year(), 365);
        assert_eq!(d.quarter_of_year(), 4);

        let d = Date::from_ymd(2026, 4, 1);
        assert_eq!(d.quarter_of_year(), 2);
    }

    #[test]
    fn date_arithmetic_operators() {
        let d = Date::from_ymd(2026, 7, 12);
        let d2 = d + 1;
        assert_eq!(d2.get(), (2026, 7, 13));
        let d3 = d2 - 1;
        assert_eq!(d3, d);

        let mut d4 = d;
        d4 += 30;
        assert_eq!(d4.get(), (2026, 8, 11));
        d4 -= 30;
        assert_eq!(d4, d);
    }

    #[test]
    fn date_ordering() {
        let a = Date::from_ymd(2026, 1, 1);
        let b = Date::from_ymd(2026, 1, 2);
        assert!(a < b);
        assert!(b > a);
        assert!(a <= a);
        assert_ne!(a, b);
    }

    #[test]
    fn date_to_string_default_format() {
        let d = Date::from_ymd(2026, 7, 12);
        assert_eq!(d.to_string("YYYY.MM.DD", LanguageId::English), "2026.07.12");
    }

    #[test]
    fn date_to_string_names_and_julian() {
        let d = Date::from_ymd(2026, 7, 12); // Sunday
        assert_eq!(
            d.to_string("DAY, MON DD YYYY", LanguageId::English),
            "Sunday, Jul 12 2026"
        );
        assert_eq!(d.to_string("DY", LanguageId::English), "Sun");
        assert_eq!(d.to_string("MONTH", LanguageId::English), "July");
        assert_eq!(
            d.to_string("J", LanguageId::English),
            format!("{}", d.julian_day())
        );
        assert_eq!(d.to_string("Q D", LanguageId::English), "3 7");
    }

    #[test]
    fn date_to_string_russian() {
        let d = Date::from_ymd(2026, 7, 12);
        assert_eq!(d.to_string("MONTH", LanguageId::Russian), "Июль");
    }

    #[test]
    fn date_to_string_invalid_is_empty() {
        let d = Date::default();
        assert_eq!(d.to_string("YYYY", LanguageId::English), "");
    }

    #[test]
    fn date_from_macros() {
        let d = Date::from_macros("Jul 12 2026");
        assert_eq!(d.get(), (2026, 7, 12));

        assert!(Date::from_macros("garbage").is_invalid());
        assert!(Date::from_macros("Xyz 12 2026").is_invalid());
    }

    #[test]
    fn time_from_hms_roundtrips() {
        let t = Time::from_hms(13, 45, 30, 250);
        assert!(!t.is_invalid());
        assert_eq!(t.get(), (13, 45, 30, 250));
        assert_eq!(t.hour24(), 13);
        assert_eq!(t.hour12(), 1);
        assert!(t.is_pm());
        assert!(!t.is_am());
    }

    #[test]
    fn time_hour12_midnight_and_noon() {
        let midnight = Time::from_hms(0, 0, 0, 0);
        assert_eq!(midnight.hour12(), 12);
        assert!(midnight.is_am());

        let noon = Time::from_hms(12, 0, 0, 0);
        assert_eq!(noon.hour12(), 12);
        assert!(noon.is_pm());
    }

    #[test]
    fn time_invalid_values() {
        assert!(Time::from_hms(24, 0, 0, 0).is_invalid());
        assert!(Time::from_hms(-1, 0, 0, 0).is_invalid());
        assert!(Time::from_hms(0, 60, 0, 0).is_invalid());
        assert_eq!(Time::default().get(), (-1, -1, -1, -1));
    }

    #[test]
    fn time_new_clamps_out_of_range_to_invalid() {
        assert!(Time::new(-1).is_invalid());
        assert!(Time::new(TIME_MS_IN_DAY).is_invalid());
        assert!(!Time::new(0).is_invalid());
    }

    #[test]
    fn time_add_sub_wraps_within_day() {
        let t = Time::from_hms(23, 59, 50, 0);
        let t2 = t + 20; // 20 seconds later wraps past midnight
        assert_eq!(t2.get(), (0, 0, 10, 0));

        let t3 = Time::from_hms(0, 0, 5, 0) - 10;
        assert_eq!(t3.get(), (23, 59, 55, 0));
    }

    #[test]
    fn time_add_assign_sub_assign() {
        let mut t = Time::from_hms(10, 0, 0, 0);
        t += 3600;
        assert_eq!(t.get(), (11, 0, 0, 0));
        t -= 3600;
        assert_eq!(t.get(), (10, 0, 0, 0));
    }

    #[test]
    fn time_to_string() {
        let t = Time::from_hms(13, 5, 9, 42);
        assert_eq!(t.to_string("HH24:MI:SS"), "13:05:09");
        assert_eq!(t.to_string("HH12:MI AM"), "01:05 PM");
        assert_eq!(t.to_string("SSSSS"), "47109"); // (13*3600+5*60+9) seconds past midnight
        assert_eq!(t.to_string("FFF"), "042");
        assert_eq!(t.to_string("A.M."), "P.M.");
    }

    #[test]
    fn time_to_string_invalid_is_empty() {
        assert_eq!(Time::default().to_string("HH24:MI:SS"), "");
    }

    #[test]
    fn datetime_construction_and_ordering() {
        let dt1 =
            DateTime::from_date_time(Date::from_ymd(2026, 7, 12), Time::from_hms(10, 0, 0, 0));
        let dt2 =
            DateTime::from_date_time(Date::from_ymd(2026, 7, 12), Time::from_hms(11, 0, 0, 0));
        assert!(dt1 < dt2);
        assert!(dt2 > dt1);
        assert_ne!(dt1, dt2);

        let dt3 = DateTime::from_date(Date::from_ymd(2026, 7, 12));
        assert!(dt3.get_time().is_invalid());
        assert!(dt3.is_invalid());
    }

    #[test]
    fn datetime_distance_ms() {
        let dt1 = DateTime::from_date_time(Date::from_ymd(2026, 7, 12), Time::from_hms(0, 0, 0, 0));
        let dt2 = DateTime::from_date_time(Date::from_ymd(2026, 7, 13), Time::from_hms(0, 0, 0, 0));
        assert_eq!(dt1.distance_ms(&dt2), TIME_MS_IN_DAY as u64);
        assert_eq!(dt2.distance_ms(&dt1), TIME_MS_IN_DAY as u64);
        assert_eq!(dt1.distance_ms(&dt1), 0);
    }

    #[test]
    fn datetime_sqlite_julian_roundtrip() {
        let dt = DateTime::from_date_time(Date::from_ymd(2026, 7, 12), Time::from_hms(12, 0, 0, 0));
        let jd = dt.to_sqlite_julian();
        let back = DateTime::from_sqlite_julian(jd);
        assert_eq!(back.get_date().get(), (2026, 7, 12));
        assert_eq!(back.get_time().hour24(), 12);
    }

    #[test]
    fn datetime_unix_roundtrip() {
        let dt = DateTime::from_date_time(Date::from_ymd(2026, 7, 12), Time::from_hms(0, 0, 0, 0));
        let unix = dt.to_unix();
        let back = DateTime::from_unix(unix);
        assert_eq!(back.get_date().get(), (2026, 7, 12));
        assert_eq!(back.get_time().hour24(), 0);
    }

    #[test]
    fn datetime_from_unix_epoch() {
        let dt = DateTime::from_unix(0.0);
        assert_eq!(dt.get_date().get(), (1970, 1, 1));
        assert_eq!(dt.get_time().get(), (0, 0, 0, 0));
    }

    #[test]
    fn datetime_to_string_combines_date_and_time() {
        let dt = DateTime::from_date_time(Date::from_ymd(2026, 7, 12), Time::from_hms(9, 5, 0, 0));
        assert_eq!(
            dt.to_string("YYYY.MM.DD HH24:MI:SS", LanguageId::English),
            "2026.07.12 09:05:00"
        );
    }

    #[test]
    fn datetime_to_string_invalid_is_empty() {
        assert_eq!(
            DateTime::default().to_string("YYYY.MM.DD", LanguageId::English),
            ""
        );
    }

    #[test]
    fn from_system_utc_is_not_invalid() {
        // Sanity check only (the value depends on wall-clock time); confirms
        // the SystemTime -> jd/ms conversion produces a valid Date/Time.
        let dt = DateTime::from_system_utc();
        assert!(!dt.is_invalid());
        assert!(dt.get_date().year() >= 2026);

        let d = Date::from_system();
        assert!(!d.is_invalid());
        let t = Time::from_system();
        assert!(!t.is_invalid());
    }
}
