//! Port of Kyty's `Sys::SysTimer` (Windows) —
//! `reference/kyty/source/include/Kyty/Sys/SysTimer.h` (the cross-platform
//! header, which for this API is nothing but an `IWYU pragma: export` of the
//! per-platform headers) and
//! `reference/kyty/source/include/Kyty/Sys/Windows/SysWindowsTimer.h` (the
//! Windows implementation this module ports; every function is `inline` in
//! the header itself, there is no separate `.cpp`).
//!
//! XPS5X is Windows-first, so only the Windows implementation is ported.
//! This whole module is gated `#[cfg(windows)]`.
//!
//! std mapping used by this port:
//! - `QueryPerformanceFrequency`/`QueryPerformanceCounter` ->
//!   [`std::time::Instant`], exactly like `crate::timer::Timer` (see that
//!   module's doc comment for the rationale): ticks are nanoseconds since a
//!   process-wide lazily-initialized epoch, so `sys_query_performance_frequency()`
//!   is fixed at `1_000_000_000`. This module keeps its *own* epoch,
//!   independent of `crate::timer`'s — both are faithful re-implementations
//!   of "a monotonic counter", not required to share a single global instant,
//!   and `crate::timer` is an already-ported Core module this task must not
//!   touch.
//! - `GetSystemTime`/`GetLocalTime`/`FileTimeToSystemTime`/
//!   `SystemTimeToFileTime` -> [`std::time::SystemTime`] plus a pure-Rust
//!   proleptic-Gregorian calendar <-> day-count conversion (Howard Hinnant's
//!   public-domain `days_from_civil`/`civil_from_days` algorithm,
//!   <http://howardhinnant.github.io/date_algorithms.html>), instead of
//!   calling the Win32 APIs. This needs no FFI and is exact over the
//!   FILETIME epoch's range. `SysFileTimeStruct::time` (a Win32 `FILETIME`,
//!   i.e. two `u32`s forming 100ns ticks since 1601-01-01 UTC) is represented
//!   here as a single `u64` tick count — same value, thinner shape (per this
//!   crate's porting convention: preserve API + semantics, not the exact
//!   struct layout).
//! - Rust's std has no cross-platform local-timezone API. Matching the same
//!   documented, intentional gap already taken in `crate::date_time`
//!   (`Date::from_system` == `Date::from_system_utc` pending a real
//!   timezone source), [`sys_get_system_time`] (local time, `GetLocalTime`)
//!   currently behaves identically to [`sys_get_system_time_utc`]
//!   (`GetSystemTime`).
//!
//! Kyty's out-parameter style (`void f(const In& in, Out& out)`) is replaced
//! by returning the value, per this crate's porting convention.

#![cfg(windows)]

use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Ticks-per-second for [`sys_query_performance_frequency`] /
/// [`sys_query_performance_counter`]'s tick domain (nanoseconds).
const PERFORMANCE_FREQUENCY: u64 = 1_000_000_000;

/// Process-wide monotonic epoch that [`sys_query_performance_counter`]
/// measures from. See the module doc comment for why this is a separate
/// epoch from `crate::timer::Timer`'s.
fn performance_epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Number of 100ns FILETIME ticks per calendar unit.
const TICKS_PER_DAY: u64 = 864_000_000_000;
const TICKS_PER_HOUR: u64 = 36_000_000_000;
const TICKS_PER_MINUTE: u64 = 600_000_000;
const TICKS_PER_SECOND: u64 = 10_000_000;
const TICKS_PER_MS: u64 = 10_000;

/// Ticks between the FILETIME epoch (1601-01-01 00:00:00 UTC) and the Unix
/// epoch (1970-01-01 00:00:00 UTC) — the same constant Kyty's
/// `sys_time_t_to_system` uses (`116444736000000000`).
const FILETIME_UNIX_DIFF_TICKS: i64 = 116_444_736_000_000_000;
/// Days between the two epochs above (`116444736000000000 / 864000000000`).
const FILETIME_UNIX_DIFF_DAYS: i64 = 134_774;

/// Kyty `SysTimeStruct` — a calendar timestamp (date + time-of-day) plus an
/// `is_invalid` flag, mirroring the Windows header's fields 1:1
/// (`uint16_t` -> `u16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SysTimeStruct {
    pub year: u16,
    pub month: u16,
    pub day: u16,
    pub hour: u16,
    pub minute: u16,
    pub second: u16,
    pub milliseconds: u16,
    pub is_invalid: bool,
}

/// Kyty `SysFileTimeStruct` — a Win32 `FILETIME` (100ns ticks since
/// 1601-01-01 UTC) plus an `is_invalid` flag. See the module doc comment for
/// why `time` (a two-`u32` struct in the original) is a single `u64` here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SysFileTimeStruct {
    pub ticks: u64,
    pub is_invalid: bool,
}

/// Howard Hinnant's `days_from_civil`: days since 1970-01-01 for a
/// proleptic-Gregorian `(y, m, d)`, `m` in `[1, 12]`.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Howard Hinnant's `civil_from_days`: the inverse of [`days_from_civil`].
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Kyty `sys_file_to_system_time_utc(const SysFileTimeStruct& f, SysTimeStruct& t)`.
#[must_use]
pub fn sys_file_to_system_time_utc(f: &SysFileTimeStruct) -> SysTimeStruct {
    if f.is_invalid {
        return SysTimeStruct { is_invalid: true, ..Default::default() };
    }

    let total_days = (f.ticks / TICKS_PER_DAY) as i64 - FILETIME_UNIX_DIFF_DAYS;
    let day_ticks = f.ticks % TICKS_PER_DAY;

    let (year, month, day) = civil_from_days(total_days);
    let hour = day_ticks / TICKS_PER_HOUR;
    let minute = (day_ticks % TICKS_PER_HOUR) / TICKS_PER_MINUTE;
    let mut second = (day_ticks % TICKS_PER_MINUTE) / TICKS_PER_SECOND;
    let milliseconds = (day_ticks % TICKS_PER_SECOND) / TICKS_PER_MS;

    // Kyty: `t.Second = (s.wSecond == 60 ? 59 : s.wSecond);` — clamp a leap
    // second down to 59. Our own arithmetic never produces 60, but the
    // clamp is kept for behavioral fidelity.
    if second == 60 {
        second = 59;
    }

    SysTimeStruct {
        year: year as u16,
        month: month as u16,
        day: day as u16,
        hour: hour as u16,
        minute: minute as u16,
        second: second as u16,
        milliseconds: milliseconds as u16,
        is_invalid: false,
    }
}

/// Kyty `sys_time_t_to_system(time_t t, SysTimeStruct& s)`.
#[must_use]
pub fn sys_time_t_to_system(t: i64) -> SysTimeStruct {
    // Port of `Int32x32To64(t, 10000000) + 116444736000000000`: converts a
    // `time_t` (seconds since the Unix epoch, UTC) to FILETIME ticks, then
    // reassembles via `sys_file_to_system_time_utc`, exactly as the original
    // does.
    let ticks = t * 10_000_000 + FILETIME_UNIX_DIFF_TICKS;
    let ft = SysFileTimeStruct { ticks: ticks as u64, is_invalid: false };
    sys_file_to_system_time_utc(&ft)
}

/// Kyty `sys_system_to_file_time_utc(const SysTimeStruct& f, SysFileTimeStruct& t)`.
#[must_use]
pub fn sys_system_to_file_time_utc(f: &SysTimeStruct) -> SysFileTimeStruct {
    if f.is_invalid {
        return SysFileTimeStruct { is_invalid: true, ..Default::default() };
    }

    let days = days_from_civil(i64::from(f.year), i64::from(f.month), i64::from(f.day)) + FILETIME_UNIX_DIFF_DAYS;
    let ticks = (days as u64) * TICKS_PER_DAY
        + u64::from(f.hour) * TICKS_PER_HOUR
        + u64::from(f.minute) * TICKS_PER_MINUTE
        + u64::from(f.second) * TICKS_PER_SECOND
        + u64::from(f.milliseconds) * TICKS_PER_MS;

    SysFileTimeStruct { ticks, is_invalid: false }
}

/// Shared helper backing both `sys_get_system_time` and
/// `sys_get_system_time_utc` (see the module doc comment on why both return
/// UTC for now).
fn now_utc() -> SysTimeStruct {
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let total_secs = dur.as_secs() as i64;
    let days = total_secs.div_euclid(86400);
    let secs_of_day = total_secs.rem_euclid(86400);

    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let mut second = secs_of_day % 60;
    if second == 60 {
        second = 59;
    }

    SysTimeStruct {
        year: year as u16,
        month: month as u16,
        day: day as u16,
        hour: hour as u16,
        minute: minute as u16,
        second: second as u16,
        milliseconds: dur.subsec_millis() as u16,
        is_invalid: false,
    }
}

/// Kyty `sys_get_system_time(SysTimeStruct& t)` — "Retrieves the current
/// local date and time" (`GetLocalTime`). See the module doc comment: this
/// currently behaves identically to [`sys_get_system_time_utc`].
#[must_use]
pub fn sys_get_system_time() -> SysTimeStruct {
    now_utc()
}

/// Kyty `sys_get_system_time_utc(SysTimeStruct& t)` — "Retrieves the current
/// system date and time in Coordinated Universal Time (UTC)" (`GetSystemTime`).
#[must_use]
pub fn sys_get_system_time_utc() -> SysTimeStruct {
    now_utc()
}

/// Kyty `sys_query_performance_frequency(uint64_t* freq)`.
#[must_use]
pub fn sys_query_performance_frequency() -> u64 {
    PERFORMANCE_FREQUENCY
}

/// Kyty `sys_query_performance_counter(uint64_t* counter)`.
#[must_use]
pub fn sys_query_performance_counter() -> u64 {
    performance_epoch().elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn performance_frequency_is_nanosecond_domain() {
        assert_eq!(sys_query_performance_frequency(), 1_000_000_000);
    }

    #[test]
    fn performance_counter_is_monotonic_and_advances() {
        let a = sys_query_performance_counter();
        sleep(Duration::from_millis(5));
        let b = sys_query_performance_counter();
        assert!(b > a);
    }

    #[test]
    fn filetime_epoch_zero_is_1601_01_01_midnight() {
        let ft = SysFileTimeStruct { ticks: 0, is_invalid: false };
        let t = sys_file_to_system_time_utc(&ft);
        assert!(!t.is_invalid);
        assert_eq!((t.year, t.month, t.day), (1601, 1, 1));
        assert_eq!((t.hour, t.minute, t.second, t.milliseconds), (0, 0, 0, 0));
    }

    #[test]
    fn filetime_unix_diff_ticks_is_1970_01_01_midnight() {
        let ft = SysFileTimeStruct { ticks: FILETIME_UNIX_DIFF_TICKS as u64, is_invalid: false };
        let t = sys_file_to_system_time_utc(&ft);
        assert_eq!((t.year, t.month, t.day), (1970, 1, 1));
        assert_eq!((t.hour, t.minute, t.second, t.milliseconds), (0, 0, 0, 0));
    }

    #[test]
    fn time_t_zero_is_unix_epoch() {
        let t = sys_time_t_to_system(0);
        assert!(!t.is_invalid);
        assert_eq!((t.year, t.month, t.day), (1970, 1, 1));
        assert_eq!((t.hour, t.minute, t.second), (0, 0, 0));
    }

    #[test]
    fn time_t_known_date() {
        // 1_752_364_800 == 2025-07-13 00:00:00 UTC.
        let t = sys_time_t_to_system(1_752_364_800);
        assert_eq!((t.year, t.month, t.day), (2025, 7, 13));
        assert_eq!((t.hour, t.minute, t.second), (0, 0, 0));
    }

    #[test]
    fn system_to_file_time_and_back_roundtrips() {
        let original = SysTimeStruct {
            year: 2026,
            month: 7,
            day: 12,
            hour: 13,
            minute: 45,
            second: 30,
            milliseconds: 250,
            is_invalid: false,
        };
        let ft = sys_system_to_file_time_utc(&original);
        assert!(!ft.is_invalid);
        let back = sys_file_to_system_time_utc(&ft);
        assert_eq!(back, original);
    }

    #[test]
    fn invalid_file_time_propagates() {
        let ft = SysFileTimeStruct { ticks: 12345, is_invalid: true };
        let t = sys_file_to_system_time_utc(&ft);
        assert!(t.is_invalid);
    }

    #[test]
    fn invalid_system_time_propagates() {
        let s = SysTimeStruct { is_invalid: true, ..Default::default() };
        let ft = sys_system_to_file_time_utc(&s);
        assert!(ft.is_invalid);
    }

    #[test]
    fn get_system_time_utc_is_valid_and_recent() {
        let t = sys_get_system_time_utc();
        assert!(!t.is_invalid);
        assert!(t.year >= 2026);
        assert!((1..=12).contains(&t.month));
        assert!((1..=31).contains(&t.day));
        assert!(t.hour <= 23);
        assert!(t.minute <= 59);
        assert!(t.second <= 59);
    }

    #[test]
    fn get_system_time_matches_documented_utc_gap() {
        // Both should report the same wall-clock second (allowing for the
        // (extremely unlikely) chance the two calls straddle a second
        // boundary).
        let local = sys_get_system_time();
        let utc = sys_get_system_time_utc();
        assert!(!local.is_invalid && !utc.is_invalid);
        let diff = (i64::from(local.hour) * 3600 + i64::from(local.minute) * 60 + i64::from(local.second))
            - (i64::from(utc.hour) * 3600 + i64::from(utc.minute) * 60 + i64::from(utc.second));
        assert!(diff.abs() <= 1);
    }

    #[test]
    fn civil_day_conversions_are_inverse() {
        let cases = [(1970, 1, 1), (2026, 7, 12), (2000, 2, 29), (1601, 1, 1), (1900, 3, 1)];
        for (y, m, d) in cases {
            let days = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(days), (y, m, d), "roundtrip failed for {y}-{m}-{d}");
        }
    }
}
