//! HLE libSceRtc — real-time-clock calendar math.
//!
//! A faithful Rust port of the **pure** (deterministic, host-time-free) half of
//! SharpEmu's `RtcExports` (GPL-2.0): leap-year / days-in-month / day-of-week
//! queries, `SceRtcDateTime` validation, tick arithmetic (add days/hours/… to a
//! microsecond tick), and tick comparison. These need no host clock, so they
//! port completely and are deterministically testable. The `GetCurrent*Tick`
//! family (which samples the host clock) is a separate follow-up.
//!
//! An Rtc **tick** is microseconds since `0001-01-01 00:00:00`; the tick
//! resolution is 1,000,000 ticks/second.

use crate::{HleContext, HleRegistry};
use tracing::debug;

const OK: u64 = 0;

// libSceRtc error codes (returned in the low 32 bits, read by the guest as a
// negative `int`).
const ERR_INVALID_POINTER: u64 = 0x80B5_0002;
const ERR_INVALID_VALUE: u64 = 0x80B5_0003;
const ERR_INVALID_ARG: u64 = 0x80B5_0004;
const ERR_INVALID_YEAR: u64 = 0x80B5_0008;
const ERR_INVALID_MONTH: u64 = 0x80B5_0009;
const ERR_INVALID_DAY: u64 = 0x80B5_000A;
const ERR_INVALID_HOUR: u64 = 0x80B5_000B;
const ERR_INVALID_MINUTE: u64 = 0x80B5_000C;
const ERR_INVALID_SECOND: u64 = 0x80B5_000D;
const ERR_INVALID_MICROSECOND: u64 = 0x80B5_000E;

/// Microseconds from `0001-01-01` (the Rtc tick epoch) to the Unix epoch
/// (`1970-01-01`). Adding host Unix-microseconds to this yields an Rtc tick.
const UNIX_EPOCH_TICKS: u64 = 62_135_596_800_000_000;

const MICROSECONDS_PER_SECOND: u64 = 1_000_000;
const MICROSECONDS_PER_MINUTE: u64 = 60 * MICROSECONDS_PER_SECOND;
const MICROSECONDS_PER_HOUR: u64 = 60 * MICROSECONDS_PER_MINUTE;
const MICROSECONDS_PER_DAY: u64 = 24 * MICROSECONDS_PER_HOUR;
const MICROSECONDS_PER_WEEK: u64 = 7 * MICROSECONDS_PER_DAY;

/// `SCE_RTC_STRING_BUFSIZE`: the exact size of the `char *` buffer
/// `sceRtcFormatRFC3339`/`Precise`/`LocalTime` render into. Callers declare it
/// as a stack local, so every byte past this is a caller local, a saved
/// register, or the stack-protector canary.
const RFC3339_BUFSIZE: usize = 32;

/// The largest legal `timeZoneMinutes` magnitude: ±14 h, RFC 3339's maximum
/// UTC offset. The value arrives in a guest register, so bounding it is what
/// keeps the rendered suffix two digits wide (`±hh:mm`) instead of however many
/// digits an arbitrary `int` needs.
const MAX_TZ_OFFSET_MINUTES: i32 = 14 * 60;

/// Register the libSceRtc HLE functions.
pub fn register(registry: &HleRegistry) {
    // Return-code-only lifecycle around a clock that is always available (the
    // host's); there is nothing to set up or release, so OK is complete.
    registry.register("libSceRtc", "sceRtcInit", hle_ok);
    registry.register("libSceRtc", "sceRtcEnd", hle_ok);
    registry.register(
        "libSceRtc",
        "sceRtcGetTickResolution",
        hle_get_tick_resolution,
    );
    // Wall-clock "current tick" family — all report the host UTC time as an
    // Rtc tick (offline: no separate network clock).
    registry.register("libSceRtc", "sceRtcGetCurrentTick", hle_get_current_tick);
    registry.register(
        "libSceRtc",
        "sceRtcGetCurrentNetworkTick",
        hle_get_current_tick,
    );
    registry.register(
        "libSceRtc",
        "sceRtcGetCurrentRawNetworkTick",
        hle_get_current_tick,
    );
    registry.register(
        "libSceRtc",
        "sceRtcGetCurrentAdNetworkTick",
        hle_get_current_tick,
    );
    registry.register(
        "libSceRtc",
        "sceRtcGetCurrentDebugNetworkTick",
        hle_get_current_tick,
    );
    registry.register("libSceRtc", "sceRtcGetCurrentClock", hle_get_current_clock);
    registry.register(
        "libSceRtc",
        "sceRtcGetCurrentClockLocalTime",
        hle_get_current_clock,
    );
    registry.register(
        "libSceRtc",
        "sceRtcConvertUtcToLocalTime",
        hle_convert_utc_to_local_time,
    );
    registry.register("libSceRtc", "sceRtcIsLeapYear", hle_is_leap_year);
    registry.register("libSceRtc", "sceRtcGetDaysInMonth", hle_get_days_in_month);
    registry.register("libSceRtc", "sceRtcGetDayOfWeek", hle_get_day_of_week);
    registry.register("libSceRtc", "sceRtcCheckValid", hle_check_valid);
    registry.register("libSceRtc", "sceRtcGetTick", hle_get_tick);
    registry.register("libSceRtc", "sceRtcGetTime_t", hle_get_time_t);
    registry.register("libSceRtc", "sceRtcSetTick", hle_set_tick);
    registry.register("libSceRtc", "sceRtcSetTime_t", hle_set_time_t);
    registry.register("libSceRtc", "sceRtcFormatRFC3339", hle_format_rfc3339);
    registry.register("libSceRtc", "sceRtcParseRFC3339", hle_parse_rfc3339);
    registry.register("libSceRtc", "sceRtcCompareTick", hle_compare_tick);
    registry.register("libSceRtc", "sceRtcTickAddTicks", hle_tick_add::<1>);
    registry.register("libSceRtc", "sceRtcTickAddMicroseconds", hle_tick_add::<1>);
    registry.register(
        "libSceRtc",
        "sceRtcTickAddSeconds",
        hle_tick_add::<MICROSECONDS_PER_SECOND>,
    );
    registry.register(
        "libSceRtc",
        "sceRtcTickAddMinutes",
        hle_tick_add::<MICROSECONDS_PER_MINUTE>,
    );
    registry.register(
        "libSceRtc",
        "sceRtcTickAddHours",
        hle_tick_add::<MICROSECONDS_PER_HOUR>,
    );
    registry.register(
        "libSceRtc",
        "sceRtcTickAddDays",
        hle_tick_add::<MICROSECONDS_PER_DAY>,
    );
    registry.register(
        "libSceRtc",
        "sceRtcTickAddWeeks",
        hle_tick_add::<MICROSECONDS_PER_WEEK>,
    );
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    OK
}

/// Gregorian leap-year rule.
fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Days in `month` (1-12) of `year`. Caller must pass a valid month.
fn days_in_month(year: i32, month: i32) -> i32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Day of week for a valid date, 0 = Sunday .. 6 = Saturday (Sakamoto's
/// method), matching SharpEmu's `DateTime.DayOfWeek`.
fn day_of_week(mut year: i32, month: i32, day: i32) -> i32 {
    const T: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    if month < 3 {
        year -= 1;
    }
    (year + year / 4 - year / 100 + year / 400 + T[(month - 1) as usize] + day).rem_euclid(7)
}

/// `sceRtcGetTickResolution()`: microsecond ticks → 1,000,000 per second.
fn hle_get_tick_resolution(_ctx: &HleContext, _args: &[u64]) -> u64 {
    MICROSECONDS_PER_SECOND
}

/// The current host UTC time as an Rtc tick (microseconds since 0001-01-01).
/// A pre-epoch host clock clamps to the epoch base rather than wrapping.
fn current_tick() -> u64 {
    let unix_micros = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_micros() as u64,
        Err(_) => 0,
    };
    UNIX_EPOCH_TICKS.saturating_add(unix_micros)
}

/// `sceRtcGetCurrentTick(SceRtcTick *out)` and the network-tick aliases: write
/// the current host UTC time as an Rtc tick. Offline, the "network" clock is
/// just the wall clock.
fn hle_get_current_tick(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    if out == 0 {
        return ERR_INVALID_POINTER;
    }
    if !ctx.mem.write(out, &current_tick().to_le_bytes()) {
        return ERR_INVALID_POINTER;
    }
    OK
}

/// A calendar date-time broken out from an Rtc tick.
struct DateTime {
    year: u16,
    month: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    micro: u32,
}

/// Convert an Rtc tick (µs since 0001-01-01) to a broken-out date-time. The
/// date is derived with Howard Hinnant's `civil_from_days` algorithm; the
/// time-of-day is the intra-day remainder.
fn tick_to_datetime(tick: u64) -> DateTime {
    // Microseconds since the Unix epoch (ticks before 1970 clamp to 0).
    let unix_micros = tick.saturating_sub(UNIX_EPOCH_TICKS);
    let days = (unix_micros / MICROSECONDS_PER_DAY) as i64;
    let rem = unix_micros % MICROSECONDS_PER_DAY;

    // civil_from_days: days since 1970-01-01 → (year, month, day).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = year + i64::from(month <= 2);

    DateTime {
        year: year as u16,
        month: month as u16,
        day: day as u16,
        hour: (rem / MICROSECONDS_PER_HOUR) as u16,
        minute: ((rem % MICROSECONDS_PER_HOUR) / MICROSECONDS_PER_MINUTE) as u16,
        second: ((rem % MICROSECONDS_PER_MINUTE) / MICROSECONDS_PER_SECOND) as u16,
        micro: (rem % MICROSECONDS_PER_SECOND) as u32,
    }
}

/// Write a `DateTime` into a 16-byte guest `SceRtcDateTime` at `addr`.
fn write_datetime(ctx: &HleContext, addr: u64, dt: &DateTime) -> bool {
    let mut b = [0u8; 16];
    b[0..2].copy_from_slice(&dt.year.to_le_bytes());
    b[2..4].copy_from_slice(&dt.month.to_le_bytes());
    b[4..6].copy_from_slice(&dt.day.to_le_bytes());
    b[6..8].copy_from_slice(&dt.hour.to_le_bytes());
    b[8..10].copy_from_slice(&dt.minute.to_le_bytes());
    b[10..12].copy_from_slice(&dt.second.to_le_bytes());
    b[12..16].copy_from_slice(&dt.micro.to_le_bytes());
    ctx.mem.write(addr, &b)
}

/// `sceRtcGetCurrentClock(SceRtcDateTime *out, tz)` / `...LocalTime(out)`:
/// write the current wall-clock time as a broken-out date-time. Offline there
/// is no timezone database, so local time is treated as UTC.
fn hle_get_current_clock(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    if out == 0 {
        return ERR_INVALID_POINTER;
    }
    let dt = tick_to_datetime(current_tick());
    if !write_datetime(ctx, out, &dt) {
        return ERR_INVALID_POINTER;
    }
    OK
}

/// `sceRtcIsLeapYear(year)`: 1 if leap, 0 if not, error for an out-of-range year.
fn hle_is_leap_year(_ctx: &HleContext, args: &[u64]) -> u64 {
    let year = args.first().copied().unwrap_or(0) as i32;
    if !(1..=9999).contains(&year) {
        return ERR_INVALID_YEAR;
    }
    u64::from(is_leap(year))
}

/// `sceRtcGetDaysInMonth(year, month)`: the number of days, or an error.
fn hle_get_days_in_month(_ctx: &HleContext, args: &[u64]) -> u64 {
    let year = args.first().copied().unwrap_or(0) as i32;
    let month = args.get(1).copied().unwrap_or(0) as i32;
    if !(1..=9999).contains(&year) {
        return ERR_INVALID_YEAR;
    }
    if !(1..=12).contains(&month) {
        return ERR_INVALID_MONTH;
    }
    days_in_month(year, month) as u64
}

/// `sceRtcGetDayOfWeek(year, month, day)`: 0 (Sun) .. 6 (Sat), or an error.
fn hle_get_day_of_week(_ctx: &HleContext, args: &[u64]) -> u64 {
    let year = args.first().copied().unwrap_or(0) as i32;
    let month = args.get(1).copied().unwrap_or(0) as i32;
    let day = args.get(2).copied().unwrap_or(0) as i32;
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return ERR_INVALID_ARG;
    }
    if day < 1 || day > days_in_month(year, month) {
        return ERR_INVALID_ARG;
    }
    day_of_week(year, month, day) as u64
}

/// Inverse of [`tick_to_datetime`]: a broken-out UTC date-time to an Rtc tick
/// (µs since 0001-01-01), via Howard Hinnant's `days_from_civil`.
fn datetime_to_tick(dt: &DateTime) -> u64 {
    let y = i64::from(dt.year) - i64::from(dt.month <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let m = i64::from(dt.month);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(dt.day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    let days = era * 146_097 + doe - 719_468; // days since 1970-01-01
    let unix_micros = days * MICROSECONDS_PER_DAY as i64
        + i64::from(dt.hour) * MICROSECONDS_PER_HOUR as i64
        + i64::from(dt.minute) * MICROSECONDS_PER_MINUTE as i64
        + i64::from(dt.second) * MICROSECONDS_PER_SECOND as i64
        + i64::from(dt.micro);
    (unix_micros + UNIX_EPOCH_TICKS as i64).max(0) as u64
}

/// Read a 16-byte guest `SceRtcDateTime` at `addr` into a [`DateTime`].
fn read_datetime(ctx: &HleContext, addr: u64) -> Option<DateTime> {
    let mut b = [0u8; 16];
    if !ctx.mem.read(addr, &mut b) {
        return None;
    }
    let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    Some(DateTime {
        year: u16_at(0),
        month: u16_at(2),
        day: u16_at(4),
        hour: u16_at(6),
        minute: u16_at(8),
        second: u16_at(10),
        micro: u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
    })
}

/// `sceRtcGetTick(const SceRtcDateTime *in, SceRtcTick *out)`: convert a
/// calendar date-time to a 64-bit tick. Unreal reads it right after
/// `sceRtcGetCurrentClock` to timestamp its log/frame clock.
fn hle_get_tick(ctx: &HleContext, args: &[u64]) -> u64 {
    let in_ptr = args.first().copied().unwrap_or(0);
    let out_ptr = args.get(1).copied().unwrap_or(0);
    if in_ptr == 0 || out_ptr == 0 {
        return ERR_INVALID_POINTER;
    }
    let Some(dt) = read_datetime(ctx, in_ptr) else {
        return ERR_INVALID_POINTER;
    };
    let tick = datetime_to_tick(&dt);
    if !ctx.mem.write(out_ptr, &tick.to_le_bytes()) {
        return ERR_INVALID_POINTER;
    }
    OK
}

/// `sceRtcGetTime_t(const SceRtcDateTime *in, time_t *out)`: convert a
/// calendar date-time to Unix seconds. Pre-epoch date-times clamp to 0
/// (shadPS4 `rtc.cpp` does the same — re-derived, not ported).
fn hle_get_time_t(ctx: &HleContext, args: &[u64]) -> u64 {
    let in_ptr = args.first().copied().unwrap_or(0);
    let out_ptr = args.get(1).copied().unwrap_or(0);
    if in_ptr == 0 || out_ptr == 0 {
        return ERR_INVALID_POINTER;
    }
    let Some(dt) = read_datetime(ctx, in_ptr) else {
        return ERR_INVALID_POINTER;
    };
    let tick = datetime_to_tick(&dt);
    let time_t = tick.saturating_sub(UNIX_EPOCH_TICKS) / MICROSECONDS_PER_SECOND;
    if !ctx.mem.write(out_ptr, &time_t.to_le_bytes()) {
        return ERR_INVALID_POINTER;
    }
    OK
}

/// `sceRtcSetTick(SceRtcDateTime *out, const SceRtcTick *in)`: the inverse —
/// break a 64-bit tick back into a calendar date-time.
fn hle_set_tick(ctx: &HleContext, args: &[u64]) -> u64 {
    let out_ptr = args.first().copied().unwrap_or(0);
    let in_ptr = args.get(1).copied().unwrap_or(0);
    if in_ptr == 0 || out_ptr == 0 {
        return ERR_INVALID_POINTER;
    }
    let mut buf = [0u8; 8];
    if !ctx.mem.read(in_ptr, &mut buf) {
        return ERR_INVALID_POINTER;
    }
    let dt = tick_to_datetime(u64::from_le_bytes(buf));
    if !write_datetime(ctx, out_ptr, &dt) {
        return ERR_INVALID_POINTER;
    }
    OK
}

/// `sceRtcSetTime_t(SceRtcDateTime *out, time_t seconds)`: break a Unix
/// `time_t` out into a calendar date-time. shadPS4 `rtc.cpp:982` (NID
/// `bDEVVP4bTjQ`): tick = seconds * 1e6 + UNIX_EPOCH_TICKS; a negative
/// `time_t` is INVALID_VALUE on SDK >= 3.00 (Raeen reports SDK 9.00, see
/// `libkernel::GEN5_SDK_VERSION`, so the modern branch always applies).
fn hle_set_time_t(ctx: &HleContext, args: &[u64]) -> u64 {
    let out_ptr = args.first().copied().unwrap_or(0);
    let seconds = args.get(1).copied().unwrap_or(0) as i64;
    if out_ptr == 0 {
        return ERR_INVALID_POINTER;
    }
    if seconds < 0 {
        return ERR_INVALID_VALUE;
    }
    let Some(unix_micros) = (seconds as u64).checked_mul(MICROSECONDS_PER_SECOND) else {
        return ERR_INVALID_VALUE;
    };
    let tick = UNIX_EPOCH_TICKS.saturating_add(unix_micros);
    let dt = tick_to_datetime(tick);
    if !write_datetime(ctx, out_ptr, &dt) {
        return ERR_INVALID_POINTER;
    }
    OK
}

/// `sceRtcFormatRFC3339(char *out, const SceRtcTick *tickUtc, int
/// timeZoneMinutes)`: render a tick as
/// `YYYY-MM-DDTHH:MM:SS.ff(Z|±hh:mm)`.
///
/// Semantics follow shadPS4 `sceRtcFormatRFC3339Precise` (`rtc.cpp:285`, NID
/// `WJ3rqFwymew` routes there): a NULL tick formats the CURRENT time; the
/// timezone offset in minutes is added to the tick before breakdown and then
/// rendered as the offset suffix (`Z` when 0); the fractional part is two
/// digits (centiseconds), always present.
///
/// # Buffer safety
///
/// The out buffer is exactly [`RFC3339_BUFSIZE`] bytes and is nearly always a
/// caller **stack local**, so the rendered text — terminator included — must
/// fit. Two inputs could previously push it past 32 bytes and smash the
/// caller's frame:
///
/// * `timeZoneMinutes` arrives in a guest register and was formatted with
///   `{:02}`, which is a *minimum* width: `i32::MAX` rendered a
///   `+35791394:07` suffix (12 chars) and produced a 36-byte string.
/// * A tick far in the future breaks down to a 5-digit year.
///
/// Both are now rejected as argument errors (a real RFC 3339 offset cannot
/// exceed ±14 h, and the ABI's `SceRtcTick` range ends at year 9999), and the
/// store itself goes through the out-buffer guard so the ABI size is enforced
/// even if a future edit reintroduces a longer rendering.
fn hle_format_rfc3339(ctx: &HleContext, args: &[u64]) -> u64 {
    let out_ptr = args.first().copied().unwrap_or(0);
    let tick_ptr = args.get(1).copied().unwrap_or(0);
    let tz_minutes = args.get(2).copied().unwrap_or(0) as i32;
    if out_ptr == 0 {
        return ERR_INVALID_POINTER;
    }
    if !(-MAX_TZ_OFFSET_MINUTES..=MAX_TZ_OFFSET_MINUTES).contains(&tz_minutes) {
        return ERR_INVALID_ARG;
    }
    let tick = if tick_ptr == 0 {
        current_tick()
    } else {
        let mut buf = [0u8; 8];
        if !ctx.mem.read(tick_ptr, &mut buf) {
            return ERR_INVALID_POINTER;
        }
        u64::from_le_bytes(buf)
    };
    let shifted = if tz_minutes >= 0 {
        tick.saturating_add(tz_minutes as u64 * MICROSECONDS_PER_MINUTE)
    } else {
        tick.saturating_sub(tz_minutes.unsigned_abs() as u64 * MICROSECONDS_PER_MINUTE)
    };
    let dt = tick_to_datetime(shifted);
    if dt.year > 9999 {
        // A 5-digit year would widen the string past the ABI buffer; the
        // `SceRtcTick` range the ABI documents ends at 9999-12-31.
        return ERR_INVALID_VALUE;
    }
    let mut text = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:02}",
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second,
        dt.micro / 10_000, // two fractional digits (centiseconds), shadPS4
    );
    if tz_minutes == 0 {
        text.push('Z');
    } else {
        let sign = if tz_minutes < 0 { '-' } else { '+' };
        let abs = tz_minutes.unsigned_abs();
        text.push_str(&format!("{sign}{:02}:{:02}", abs / 60, abs % 60));
    }
    text.push('\0');
    // Exactly the ABI buffer, never more: the guard clamps (and logs once) if
    // the rendering ever outgrows `SCE_RTC_STRING_BUFSIZE` again.
    if !ctx.write_out_struct(
        "libSceRtc::sceRtcFormatRFC3339",
        out_ptr,
        RFC3339_BUFSIZE,
        text.as_bytes(),
    ) {
        return ERR_INVALID_POINTER;
    }
    OK
}

/// `sceRtcParseRFC3339(SceRtcTick *tickUtc, const char *text)`: parse
/// `YYYY-MM-DDTHH:MM:SS[.f...](Z|±hh:mm)` into a UTC tick.
///
/// Field positions follow shadPS4 `sceRtcParseRFC3339` (`rtc.cpp:796`, NID
/// `99bMGglFW3I`); fractional digits beyond what is present are 0, and the
/// numeric UTC-offset suffix is SUBTRACTED from the parsed local time to
/// reach UTC (RFC 3339 §4.2: local = UTC + offset ⇒ UTC = local − offset;
/// shadPS4 applies the offset with the opposite sign, which round-trips its
/// own formatter but inverts real offsets — deliberately not mirrored).
fn hle_parse_rfc3339(ctx: &HleContext, args: &[u64]) -> u64 {
    let tick_out = args.first().copied().unwrap_or(0);
    let text_ptr = args.get(1).copied().unwrap_or(0);
    if tick_out == 0 || text_ptr == 0 {
        return ERR_INVALID_POINTER;
    }
    let mut raw = Vec::new();
    // RFC 3339 date-times fit well inside 64 bytes; read until NUL or cap.
    for offset in 0..64u64 {
        let mut byte = [0u8; 1];
        if !ctx.mem.read(text_ptr + offset, &mut byte) {
            return ERR_INVALID_POINTER;
        }
        if byte[0] == 0 {
            break;
        }
        raw.push(byte[0]);
    }
    let text = String::from_utf8_lossy(&raw).into_owned();
    let bytes = text.as_bytes();
    // Minimum: "YYYY-MM-DDTHH:MM:SS" (19 bytes).
    if bytes.len() < 19 {
        return ERR_INVALID_VALUE;
    }
    let digits = |range: std::ops::Range<usize>| -> Option<u32> {
        text.get(range).and_then(|s| s.parse::<u32>().ok())
    };
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) = (
        digits(0..4),
        digits(5..7),
        digits(8..10),
        digits(11..13),
        digits(14..16),
        digits(17..19),
    ) else {
        return ERR_INVALID_VALUE;
    };
    // Optional fraction: ".d+" — scale whatever digits are present to µs.
    let mut cursor = 19;
    let mut micro: u32 = 0;
    if bytes.get(cursor) == Some(&b'.') {
        cursor += 1;
        let mut scale = 100_000u32;
        while let Some(digit) = bytes.get(cursor).filter(|b| b.is_ascii_digit()) {
            micro += u32::from(digit - b'0') * scale;
            scale /= 10;
            cursor += 1;
            if scale == 0 {
                // Skip sub-microsecond digits.
                while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
                    cursor += 1;
                }
                break;
            }
        }
    }
    let dt = DateTime {
        year: year as u16,
        month: month as u16,
        day: day as u16,
        hour: hour as u16,
        minute: minute as u16,
        second: second as u16,
        micro,
    };
    let mut tick = datetime_to_tick(&dt);
    // Timezone suffix: Z, or ±hh:mm subtracted to reach UTC.
    match bytes.get(cursor).copied() {
        Some(b'Z' | b'z') | None => {}
        Some(sign @ (b'+' | b'-')) => {
            let (Some(hh), Some(mm)) = (
                digits(cursor + 1..cursor + 3),
                digits(cursor + 4..cursor + 6),
            ) else {
                return ERR_INVALID_VALUE;
            };
            let offset_us = u64::from(hh * 60 + mm) * MICROSECONDS_PER_MINUTE;
            tick = if sign == b'+' {
                tick.saturating_sub(offset_us)
            } else {
                tick.saturating_add(offset_us)
            };
        }
        Some(_) => return ERR_INVALID_VALUE,
    }
    if !ctx.mem.write(tick_out, &tick.to_le_bytes()) {
        return ERR_INVALID_POINTER;
    }
    OK
}

/// `sceRtcConvertUtcToLocalTime(tickUtc*, tickLocal*)`: read the UTC tick,
/// shift it by the host's timezone bias, write the local tick. SharpEmu's
/// `RtcConvertUtcToLocalTime` (TimeZoneInfo.Local); the bias is the Windows
/// `GetTimeZoneInformation` value (UTC = local + bias ⇒ local = UTC − bias).
fn hle_convert_utc_to_local_time(ctx: &HleContext, args: &[u64]) -> u64 {
    let utc_ptr = args.first().copied().unwrap_or(0);
    let local_ptr = args.get(1).copied().unwrap_or(0);
    if utc_ptr == 0 || local_ptr == 0 {
        return ERR_INVALID_POINTER;
    }
    let mut buf = [0u8; 8];
    if !ctx.mem.read(utc_ptr, &mut buf) {
        return ERR_INVALID_POINTER;
    }
    let utc = u64::from_le_bytes(buf) as i64;
    let local = utc - host_timezone_bias_microseconds();
    if local < 0 {
        return ERR_INVALID_VALUE;
    }
    if !ctx.mem.write(local_ptr, &(local as u64).to_le_bytes()) {
        return ERR_INVALID_POINTER;
    }
    OK
}

/// The host's UTC bias in microseconds (positive when local time is BEHIND
/// UTC, e.g. US timezones). Cached — the bias can only change on a TZ change
/// the title has no way to trigger.
fn host_timezone_bias_microseconds() -> i64 {
    #[cfg(windows)]
    {
        use std::sync::OnceLock;
        static BIAS_US: OnceLock<i64> = OnceLock::new();
        *BIAS_US.get_or_init(|| {
            use windows_sys::Win32::System::Time::{GetTimeZoneInformation, TIME_ZONE_INFORMATION};
            // SAFETY: plain out-param struct on a live thread; no aliasing.
            let mut info: TIME_ZONE_INFORMATION = unsafe { std::mem::zeroed() };
            let _ = unsafe { GetTimeZoneInformation(&mut info) };
            i64::from(info.Bias) * MICROSECONDS_PER_MINUTE as i64
        })
    }
    #[cfg(not(windows))]
    {
        0 // honest offline default: local == UTC until a platform hook lands
    }
}

/// `sceRtcCheckValid(const SceRtcDateTime *)`: validate each field in order,
/// returning the first field error (or OK). Struct layout: year/month/day/
/// hour/minute/second are `u16` at 0/2/4/6/8/10; microsecond is `u32` at 12.
fn hle_check_valid(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    if ptr == 0 {
        return ERR_INVALID_POINTER;
    }
    let mut buf = [0u8; 16];
    if !ctx.mem.read(ptr, &mut buf) {
        return ERR_INVALID_POINTER;
    }
    let u16_at = |o: usize| u16::from_le_bytes([buf[o], buf[o + 1]]) as i32;
    let year = u16_at(0);
    let month = u16_at(2);
    let day = u16_at(4);
    let hour = u16_at(6);
    let minute = u16_at(8);
    let second = u16_at(10);
    let micro = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);

    if !(1..=9999).contains(&year) {
        return ERR_INVALID_YEAR;
    }
    if !(1..=12).contains(&month) {
        return ERR_INVALID_MONTH;
    }
    if day < 1 || day > days_in_month(year, month) {
        return ERR_INVALID_DAY;
    }
    if hour > 23 {
        return ERR_INVALID_HOUR;
    }
    if minute > 59 {
        return ERR_INVALID_MINUTE;
    }
    if second > 59 {
        return ERR_INVALID_SECOND;
    }
    if micro > 999_999 {
        return ERR_INVALID_MICROSECOND;
    }
    OK
}

/// `sceRtcCompareTick(t1, t2)`: read two ticks, return -1 / 0 / 1.
fn hle_compare_tick(ctx: &HleContext, args: &[u64]) -> u64 {
    let a = args.first().copied().unwrap_or(0);
    let b = args.get(1).copied().unwrap_or(0);
    if a == 0 || b == 0 {
        return ERR_INVALID_POINTER;
    }
    let (mut ba, mut bb) = ([0u8; 8], [0u8; 8]);
    if !ctx.mem.read(a, &mut ba) || !ctx.mem.read(b, &mut bb) {
        return ERR_INVALID_POINTER;
    }
    let (t1, t2) = (u64::from_le_bytes(ba), u64::from_le_bytes(bb));
    match t1.cmp(&t2) {
        std::cmp::Ordering::Less => (-1i64) as u64,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// `sceRtcTickAdd*(dst, src, n)`: `*dst = *src + n * UNIT` (microseconds),
/// where `n` is signed. Underflow past zero or overflow → error.
fn hle_tick_add<const UNIT: u64>(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    let n = args.get(2).copied().unwrap_or(0) as i64;
    if dst == 0 || src == 0 {
        return ERR_INVALID_POINTER;
    }
    let mut buf = [0u8; 8];
    if !ctx.mem.read(src, &mut buf) {
        return ERR_INVALID_POINTER;
    }
    let source = u64::from_le_bytes(buf) as i64;
    let Some(delta) = n.checked_mul(UNIT as i64) else {
        return ERR_INVALID_VALUE;
    };
    let result = match source.checked_add(delta) {
        Some(r) if r >= 0 => r as u64,
        _ => return ERR_INVALID_VALUE,
    };
    if !ctx.mem.write(dst, &result.to_le_bytes()) {
        return ERR_INVALID_POINTER;
    }
    debug!("sceRtcTickAdd(unit={UNIT}): {source} + {n}*{UNIT} = {result}");
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    fn ctx_env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    /// `datetime_to_tick` must be the exact inverse of `tick_to_datetime`, and
    /// `sceRtcGetTick`/`sceRtcSetTick` round-trip through guest memory.
    #[test]
    fn get_tick_and_set_tick_round_trip() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // A known moment: 2026-07-17 20:36:00.123456 UTC. Write the datetime,
        // GetTick it, SetTick the result back, and require an identical datetime.
        let dt_in = DateTime {
            year: 2026,
            month: 7,
            day: 17,
            hour: 20,
            minute: 36,
            second: 0,
            micro: 123_456,
        };
        assert!(write_datetime(&ctx, 0x40, &dt_in));
        assert_eq!(hle_get_tick(&ctx, &[0x40, 0x80]), OK);
        // Break the tick back out into a fresh datetime slot and compare bytes.
        assert_eq!(hle_set_tick(&ctx, &[0xC0, 0x80]), OK);
        let mut orig = [0u8; 16];
        let mut round = [0u8; 16];
        assert!(ctx.mem.read(0x40, &mut orig));
        assert!(ctx.mem.read(0xC0, &mut round));
        assert_eq!(orig, round, "GetTick->SetTick must round-trip");

        // And the pure inverse holds for an arbitrary tick.
        for tick in [
            UNIX_EPOCH_TICKS,
            UNIX_EPOCH_TICKS + 1,
            63_000_000_000_000_000,
        ] {
            let dt = tick_to_datetime(tick);
            assert_eq!(datetime_to_tick(&dt), tick, "tick {tick} must round-trip");
        }
    }

    #[test]
    fn get_tick_rejects_null_pointers() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_get_tick(&ctx, &[0, 0x80]), ERR_INVALID_POINTER);
        assert_eq!(hle_get_tick(&ctx, &[0x40, 0]), ERR_INVALID_POINTER);
    }

    #[test]
    fn leap_year_and_days_in_month() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_is_leap_year(&ctx, &[2000]), 1, "2000 divisible by 400");
        assert_eq!(
            hle_is_leap_year(&ctx, &[1900]),
            0,
            "1900 divisible by 100 not 400"
        );
        assert_eq!(hle_is_leap_year(&ctx, &[2024]), 1);
        assert_eq!(hle_is_leap_year(&ctx, &[2023]), 0);
        assert_eq!(hle_is_leap_year(&ctx, &[0]), ERR_INVALID_YEAR);
        // February: 29 in a leap year, 28 otherwise.
        assert_eq!(hle_get_days_in_month(&ctx, &[2000, 2]), 29);
        assert_eq!(hle_get_days_in_month(&ctx, &[2001, 2]), 28);
        assert_eq!(hle_get_days_in_month(&ctx, &[2001, 4]), 30);
        assert_eq!(hle_get_days_in_month(&ctx, &[2001, 12]), 31);
        assert_eq!(hle_get_days_in_month(&ctx, &[2001, 13]), ERR_INVALID_MONTH);
    }

    #[test]
    fn day_of_week_matches_known_dates() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // 2000-01-01 was a Saturday (6); 2024-02-29 was a Thursday (4).
        assert_eq!(hle_get_day_of_week(&ctx, &[2000, 1, 1]), 6);
        assert_eq!(hle_get_day_of_week(&ctx, &[2024, 2, 29]), 4);
        // 2023-12-25 was a Monday (1).
        assert_eq!(hle_get_day_of_week(&ctx, &[2023, 12, 25]), 1);
        // Invalid day (Feb 30) → error.
        assert_eq!(hle_get_day_of_week(&ctx, &[2001, 2, 30]), ERR_INVALID_ARG);
    }

    #[test]
    fn check_valid_reports_the_first_bad_field() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let write_dt = |y: u16, mo: u16, d: u16, h: u16, mi: u16, s: u16, us: u32| {
            let mut b = [0u8; 16];
            b[0..2].copy_from_slice(&y.to_le_bytes());
            b[2..4].copy_from_slice(&mo.to_le_bytes());
            b[4..6].copy_from_slice(&d.to_le_bytes());
            b[6..8].copy_from_slice(&h.to_le_bytes());
            b[8..10].copy_from_slice(&mi.to_le_bytes());
            b[10..12].copy_from_slice(&s.to_le_bytes());
            b[12..16].copy_from_slice(&us.to_le_bytes());
            assert!(ctx.mem.write(0x40, &b));
        };
        write_dt(2024, 2, 29, 12, 30, 45, 500_000);
        assert_eq!(hle_check_valid(&ctx, &[0x40]), OK, "a real date is valid");
        write_dt(2023, 2, 29, 0, 0, 0, 0); // Feb 29 in a non-leap year
        assert_eq!(hle_check_valid(&ctx, &[0x40]), ERR_INVALID_DAY);
        write_dt(2024, 13, 1, 0, 0, 0, 0);
        assert_eq!(hle_check_valid(&ctx, &[0x40]), ERR_INVALID_MONTH);
        write_dt(2024, 1, 1, 24, 0, 0, 0);
        assert_eq!(hle_check_valid(&ctx, &[0x40]), ERR_INVALID_HOUR);
        assert_eq!(hle_check_valid(&ctx, &[0]), ERR_INVALID_POINTER);
    }

    #[test]
    fn current_tick_is_a_plausible_recent_wall_clock() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Rtc tick for 2020-01-01 = UNIX_EPOCH_TICKS + (50 years of µs).
        let tick_2020 = UNIX_EPOCH_TICKS + 1_577_836_800 * MICROSECONDS_PER_SECOND;
        assert_eq!(hle_get_current_tick(&ctx, &[0x100]), OK);
        let mut buf = [0u8; 8];
        assert!(ctx.mem.read(0x100, &mut buf));
        let t1 = u64::from_le_bytes(buf);
        assert!(t1 > tick_2020, "current tick must be after 2020 (got {t1})");
        // A second sample is monotonic non-decreasing.
        assert_eq!(hle_get_current_tick(&ctx, &[0x108]), OK);
        assert!(ctx.mem.read(0x108, &mut buf));
        assert!(u64::from_le_bytes(buf) >= t1);
        // NULL out-pointer → error.
        assert_eq!(hle_get_current_tick(&ctx, &[0]), ERR_INVALID_POINTER);
    }

    #[test]
    fn tick_to_datetime_recovers_known_dates() {
        // 2000-01-01 00:00:00: 10957 days after the Unix epoch.
        let tick_2000 = UNIX_EPOCH_TICKS + 10_957 * MICROSECONDS_PER_DAY;
        let dt = tick_to_datetime(tick_2000);
        assert_eq!((dt.year, dt.month, dt.day), (2000, 1, 1));
        assert_eq!((dt.hour, dt.minute, dt.second), (0, 0, 0));
        // Add 13h 37m 42.5s and check the time-of-day breakdown.
        let t = tick_2000
            + 13 * MICROSECONDS_PER_HOUR
            + 37 * MICROSECONDS_PER_MINUTE
            + 42 * MICROSECONDS_PER_SECOND
            + 500_000;
        let dt = tick_to_datetime(t);
        assert_eq!((dt.year, dt.month, dt.day), (2000, 1, 1));
        assert_eq!(
            (dt.hour, dt.minute, dt.second, dt.micro),
            (13, 37, 42, 500_000)
        );
    }

    #[test]
    fn get_current_clock_writes_a_plausible_datetime() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_get_current_clock(&ctx, &[0x40, 0]), OK);
        let mut b = [0u8; 16];
        assert!(ctx.mem.read(0x40, &mut b));
        let year = u16::from_le_bytes([b[0], b[1]]);
        let month = u16::from_le_bytes([b[2], b[3]]);
        assert!(
            (2020..=2100).contains(&year),
            "plausible current year (got {year})"
        );
        assert!((1..=12).contains(&month), "valid month (got {month})");
        assert_eq!(hle_get_current_clock(&ctx, &[0, 0]), ERR_INVALID_POINTER);
    }

    /// `sceRtcSetTime_t` breaks a Unix time into the calendar (shadPS4
    /// semantics: tick = t*1e6 + epoch base; negative → INVALID_VALUE).
    #[test]
    fn set_time_t_converts_unix_seconds() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // 2000-01-01 00:00:00 UTC = 946684800.
        assert_eq!(hle_set_time_t(&ctx, &[0x40, 946_684_800]), OK);
        let mut b = [0u8; 16];
        assert!(ctx.mem.read(0x40, &mut b));
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        assert_eq!(
            (u16_at(0), u16_at(2), u16_at(4), u16_at(6)),
            (2000, 1, 1, 0)
        );
        assert_eq!(
            hle_set_time_t(&ctx, &[0x40, (-5i64) as u64]),
            ERR_INVALID_VALUE
        );
        assert_eq!(hle_set_time_t(&ctx, &[0, 1]), ERR_INVALID_POINTER);
    }

    /// Format renders shadPS4's RFC 3339 shape and Parse inverts it; a
    /// numeric offset moves the tick the RFC-correct direction (UTC = local
    /// − offset).
    #[test]
    fn rfc3339_format_and_parse_round_trip() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // A known tick: 2026-07-17 20:36:00.120000 UTC.
        let dt = DateTime {
            year: 2026,
            month: 7,
            day: 17,
            hour: 20,
            minute: 36,
            second: 0,
            micro: 120_000,
        };
        let tick = datetime_to_tick(&dt);
        assert!(ctx.mem.write(0x40, &tick.to_le_bytes()));

        // UTC render.
        assert_eq!(hle_format_rfc3339(&ctx, &[0x100, 0x40, 0]), OK);
        let mut buf = [0u8; 32];
        assert!(ctx.mem.read(0x100, &mut buf));
        let end = buf.iter().position(|&b| b == 0).unwrap();
        let text = std::str::from_utf8(&buf[..end]).unwrap();
        assert_eq!(text, "2026-07-17T20:36:00.12Z");

        // Parse inverts it back to the same tick, modulo the sub-centisecond
        // truncation the two-digit fraction implies.
        assert!(ctx.mem.write(0x200, text.as_bytes()));
        assert!(ctx.mem.write(0x200 + end as u64, &[0u8]));
        assert_eq!(hle_parse_rfc3339(&ctx, &[0x60, 0x200]), OK);
        let mut parsed = [0u8; 8];
        assert!(ctx.mem.read(0x60, &mut parsed));
        assert_eq!(u64::from_le_bytes(parsed), tick);

        // +02:00 render: the wall-clock text moves forward two hours...
        assert_eq!(hle_format_rfc3339(&ctx, &[0x100, 0x40, 120]), OK);
        assert!(ctx.mem.read(0x100, &mut buf));
        let end = buf.iter().position(|&b| b == 0).unwrap();
        let text = std::str::from_utf8(&buf[..end]).unwrap();
        assert_eq!(text, "2026-07-17T22:36:00.12+02:00");
        // ...and parsing that text lands on the SAME UTC tick (offset is
        // subtracted — RFC 3339, not shadPS4's inverted sign).
        assert!(ctx.mem.write(0x200, text.as_bytes()));
        assert!(ctx.mem.write(0x200 + end as u64, &[0u8]));
        assert_eq!(hle_parse_rfc3339(&ctx, &[0x60, 0x200]), OK);
        assert!(ctx.mem.read(0x60, &mut parsed));
        assert_eq!(u64::from_le_bytes(parsed), tick);

        // Garbage is INVALID_VALUE, not a bogus tick.
        assert!(ctx.mem.write(0x200, b"not-a-date\0"));
        assert_eq!(hle_parse_rfc3339(&ctx, &[0x60, 0x200]), ERR_INVALID_VALUE);
        assert_eq!(hle_parse_rfc3339(&ctx, &[0, 0x200]), ERR_INVALID_POINTER);
    }

    /// `sceRtcFormatRFC3339` must never write past `SCE_RTC_STRING_BUFSIZE`.
    ///
    /// The caller's buffer is a 32-byte **stack local** (`char
    /// buf[SCE_RTC_STRING_BUFSIZE]`), and `timeZoneMinutes` arrives in a guest
    /// register. `{:02}` is a MINIMUM field width, so an out-of-range offset
    /// renders as many digits as it needs: `i32::MAX` minutes formats an
    /// `+35791394:07` suffix, pushing the string to 36 bytes and dropping 4
    /// bytes onto the caller's saved registers / `__stack_chk_guard` canary.
    /// A real offset cannot exceed ±14 h, so an out-of-range one is an
    /// argument error — never a longer string.
    #[test]
    fn rfc3339_format_never_writes_past_the_32_byte_abi_buffer() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let tick = datetime_to_tick(&DateTime {
            year: 2026,
            month: 7,
            day: 17,
            hour: 20,
            minute: 36,
            second: 0,
            micro: 120_000,
        });
        assert!(ctx.mem.write(0x40, &tick.to_le_bytes()));
        // Poison the 8 bytes immediately after the caller's 32-byte buffer:
        // these stand in for the canary and the saved registers above it.
        const CANARY: [u8; 8] = [0xCA; 8];
        assert!(ctx.mem.write(0x100 + RFC3339_BUFSIZE as u64, &CANARY));

        for tz in [i32::MAX, i32::MIN, 841, -841, 100_000] {
            assert_eq!(
                hle_format_rfc3339(&ctx, &[0x100, 0x40, tz as u32 as u64]),
                ERR_INVALID_ARG,
                "an out-of-range timeZoneMinutes ({tz}) must be rejected"
            );
            let mut after = [0u8; 8];
            assert!(ctx.mem.read(0x100 + RFC3339_BUFSIZE as u64, &mut after));
            assert_eq!(
                after, CANARY,
                "formatting must not touch a byte past the 32-byte ABI buffer"
            );
        }

        // The widest legal offset (±14:00) still fits, terminator included.
        for tz in [840i32, -840] {
            assert_eq!(
                hle_format_rfc3339(&ctx, &[0x100, 0x40, tz as u32 as u64]),
                OK
            );
            let mut buf = [0u8; RFC3339_BUFSIZE];
            assert!(ctx.mem.read(0x100, &mut buf));
            assert!(
                buf.contains(&0),
                "the rendered string must be NUL-terminated inside the buffer"
            );
            let mut after = [0u8; 8];
            assert!(ctx.mem.read(0x100 + RFC3339_BUFSIZE as u64, &mut after));
            assert_eq!(after, CANARY);
        }
    }

    #[test]
    fn tick_add_and_compare() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // src tick at 0x100 = 1_000_000 (1 second past epoch base).
        assert!(ctx.mem.write(0x100, &1_000_000u64.to_le_bytes()));
        // Add 2 seconds → dst at 0x108 = 3_000_000.
        assert_eq!(
            hle_tick_add::<MICROSECONDS_PER_SECOND>(&ctx, &[0x108, 0x100, 2]),
            OK
        );
        let mut buf = [0u8; 8];
        assert!(ctx.mem.read(0x108, &mut buf));
        assert_eq!(u64::from_le_bytes(buf), 3_000_000);
        // Compare: 0x100 (1e6) < 0x108 (3e6) → -1; reversed → 1; equal → 0.
        assert_eq!(hle_compare_tick(&ctx, &[0x100, 0x108]), (-1i64) as u64);
        assert_eq!(hle_compare_tick(&ctx, &[0x108, 0x100]), 1);
        assert_eq!(hle_compare_tick(&ctx, &[0x100, 0x100]), 0);
        // Subtracting past zero → error, not a wrap. (n = -5, as a u64 arg.)
        assert_eq!(
            hle_tick_add::<MICROSECONDS_PER_SECOND>(&ctx, &[0x108, 0x100, (-5i64) as u64]),
            ERR_INVALID_VALUE
        );
    }
}
