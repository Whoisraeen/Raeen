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

const MICROSECONDS_PER_SECOND: u64 = 1_000_000;
const MICROSECONDS_PER_MINUTE: u64 = 60 * MICROSECONDS_PER_SECOND;
const MICROSECONDS_PER_HOUR: u64 = 60 * MICROSECONDS_PER_MINUTE;
const MICROSECONDS_PER_DAY: u64 = 24 * MICROSECONDS_PER_HOUR;
const MICROSECONDS_PER_WEEK: u64 = 7 * MICROSECONDS_PER_DAY;

/// Register the libSceRtc HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceRtc", "sceRtcInit", hle_ok);
    registry.register("libSceRtc", "sceRtcEnd", hle_ok);
    registry.register(
        "libSceRtc",
        "sceRtcGetTickResolution",
        hle_get_tick_resolution,
    );
    registry.register("libSceRtc", "sceRtcIsLeapYear", hle_is_leap_year);
    registry.register("libSceRtc", "sceRtcGetDaysInMonth", hle_get_days_in_month);
    registry.register("libSceRtc", "sceRtcGetDayOfWeek", hle_get_day_of_week);
    registry.register("libSceRtc", "sceRtcCheckValid", hle_check_valid);
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
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
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
