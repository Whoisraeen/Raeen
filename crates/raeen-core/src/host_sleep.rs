//! Host sleep primitive and Windows timer-resolution control.
//!
//! # The measured problem
//!
//! A guest sleep is only as precise as the host primitive underneath it, and on
//! Windows there are three tiers that differ by more than four orders of
//! magnitude. Measured on this project's reference host (12 logical cores,
//! Windows 11, rustc 1.97, release build, 200–300 samples per row):
//!
//! | requested | default waitable timer | `std::thread::sleep` | `Condvar::wait_timeout` |
//! |-----------|-----------------------|----------------------|-------------------------|
//! | 1 µs      | 15 103 µs             | 531 µs               | —                       |
//! | 100 µs    | 15 233 µs             | 532 µs               | 15 394 µs               |
//! | 1 ms      | 15 540 µs             | 1 435 µs             | 15 564 µs               |
//! | 10 ms     | 15 525 µs             | 10 231 µs            | 15 454 µs               |
//!
//! Two conclusions drove this module's design, and both contradict the obvious
//! assumption:
//!
//! * **`std::thread::sleep` is not the coarse one.** Rust's Windows
//!   implementation already parks on a high-resolution waitable timer, so the
//!   guest sleep path was never paying the ~15.6 ms system tick. Its floor is
//!   ~530 µs — the practical resolution of a high-resolution waitable timer.
//!   That floor is still a 531× multiplier on a 1 µs request, which is what
//!   this module's spin phase exists for.
//! * **The ~15.6 ms tick is paid by every *condition variable* timed wait** —
//!   `Condvar::wait_timeout` and `parking_lot`'s `wait_for` alike. Those are
//!   the primitives behind the kernel event-flag, equeue and semaphore wait
//!   slices, and no waitable timer can fix them: a notifiable wait must stay a
//!   condition-variable wait. The only lever is the process timer resolution,
//!   which is why [`arm_high_resolution_timer`] exists alongside the sleep
//!   path. Measured with `timeBeginPeriod(1)`: a 1 ms `wait_timeout` drops from
//!   15 564 µs to 3 020 µs, and a 10 ms one from 15 454 µs to 10 457 µs.
//!
//! # Why not a yielding spin
//!
//! Every reference emulator studied for this (kytyps5 spins below 1 ms,
//! SharpEmu below 100 µs, shadPS4 not at all) uses some spin phase, and two of
//! them yield inside it. Measured under 30 saturating threads on 12 cores —
//! the emulator's own steady state — a yielding spin is catastrophic: a
//! `SwitchToThread`/`yield_now` loop cost **60–190 ms** for a sub-millisecond
//! request, because a thread that yields goes to the back of a ready queue it
//! then has to traverse. A `PAUSE`-only spin held its core and stayed exact.
//!
//! So: `PAUSE` only, never `yield_now`, and the spin phase is admission-limited
//! to [`MAX_CONCURRENT_SPINNERS`] threads so ~30 guest threads cannot
//! collectively burn a 12-core host.
//!
//! # Guest semantics
//!
//! Sleeping **less** than requested is a correctness bug, not an optimisation.
//! Every path here loops against an [`Instant`] deadline and
//! [`timer_due_time_100ns`] rounds the timer's due time *up*, so no strategy
//! can return early. Sleeping *more* is merely lost time.
//!
//! Behaviour was informed by studying (not copying) `reference/kytyps5`
//! (`src/common/threads.cpp` thread-local `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`),
//! `reference/shadps4` (`src/common/thread.cpp` `AccurateSleep`) and
//! `reference/sharpemu` (`src/SharpEmu.Libs/HostTiming.cs` tiered ladder). The
//! thread-local cached-timer shape already existed in-tree at
//! `raeen-hle/src/libsce_video_out.rs`; this module generalises it, and that
//! copy has since been collapsed into [`sleep_until`] so there is one timer
//! primitive rather than two that can drift apart.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Longest request that may be honoured by spinning instead of parking.
///
/// 100 µs matches the threshold SharpEmu's `HostTiming` settled on. Below the
/// ~530 µs floor of a high-resolution waitable timer, parking cannot honour the
/// request at all — a 100 µs park measured 5.2× long — so a bounded `PAUSE`
/// spin is the only way to be accurate. Above it, parking is within ~2× and
/// spinning would cost real core time.
pub const DEFAULT_SPIN_CEILING: Duration = Duration::from_micros(100);

/// How many threads may be inside the spin phase simultaneously.
///
/// The spin phase can never burn more core time than the guest asked to sleep,
/// but ~30 guest threads each spinning their own request would still saturate a
/// 12-core host. Two slots covers the one or two frame-pacing threads that
/// actually issue short sleeps; every other thread parks.
pub const MAX_CONCURRENT_SPINNERS: usize = 2;

/// Residual that is spun out rather than re-parked, so a park can never return
/// early.
///
/// A waitable timer's due time is expressed on a different clock than
/// [`Instant`] (QPC), so the two can disagree by well under a microsecond even
/// though [`timer_due_time_100ns`] rounds up. Spinning that residual is exact
/// and free; re-parking it would cost another ~530 µs floor.
const NEVER_EARLY_TOP_UP: Duration = Duration::from_micros(2);

/// Host facts the strategy choice depends on, passed explicitly so [`plan`]
/// stays a pure function testable without a host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCapabilities {
    /// Whether a high-resolution waitable timer can be created on this host.
    pub high_resolution_timer: bool,
    /// Requests at or below this length may spin.
    pub spin_ceiling: Duration,
    /// Whether a spin admission slot is free (see [`MAX_CONCURRENT_SPINNERS`]).
    pub spin_slot_available: bool,
}

impl HostCapabilities {
    /// The capabilities of a host with no high-resolution timer and no spin
    /// budget — the most degraded configuration, and the one every fallback
    /// path must remain correct on.
    #[must_use]
    pub const fn degraded() -> Self {
        Self {
            high_resolution_timer: false,
            spin_ceiling: Duration::ZERO,
            spin_slot_available: false,
        }
    }
}

/// How the host will honour one guest sleep request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepPlan {
    /// A zero-length request: give up the rest of this thread's quantum.
    ///
    /// `usleep(0)` is a yield request, not a sleep, and this is the one place a
    /// yield is correct — there is no deadline to overshoot.
    Yield,
    /// Hold the core and `PAUSE` to the deadline. Exact, and bounded by
    /// [`HostCapabilities::spin_ceiling`].
    Spin,
    /// Park on a high-resolution waitable timer.
    HighResolutionTimer,
    /// Park with `std::thread::sleep` — no high-resolution timer available.
    Fallback,
}

/// Choose the strategy for one request. Pure: the entire decision is a function
/// of the request and the host's capabilities.
///
/// Order matters. A zero request is a yield whatever the host can do; a short
/// request prefers the spin because no park can honour it; and the fallback is
/// reached only when the host cannot create a high-resolution timer at all,
/// which is the pre-Windows-10-1803 and non-Windows case.
#[must_use]
pub fn plan(requested: Duration, caps: HostCapabilities) -> SleepPlan {
    if requested.is_zero() {
        return SleepPlan::Yield;
    }
    if caps.spin_slot_available && requested <= caps.spin_ceiling {
        return SleepPlan::Spin;
    }
    if caps.high_resolution_timer {
        return SleepPlan::HighResolutionTimer;
    }
    SleepPlan::Fallback
}

/// Choose the strategy for a wait to an absolute deadline. `None` means the
/// deadline has already passed and the caller must return immediately.
///
/// That `None` is the one input on which a deadline wait and a duration wait
/// must disagree. [`plan`] maps a zero request to [`SleepPlan::Yield`] because
/// `usleep(0)` *is* a yield request. A deadline that has already passed is not
/// a request for anything — it is a wait that is merely late, and a thread
/// pacing to a fixed grid that surrenders its quantum on the way out of a late
/// wait misses the next edge too.
#[must_use]
pub fn plan_until(remaining: Duration, caps: HostCapabilities) -> Option<SleepPlan> {
    (!remaining.is_zero()).then(|| plan(remaining, caps))
}

/// A waitable timer's relative due time in 100 ns units, **negative** as the
/// Win32 ABI requires for a relative time.
///
/// Rounds **up**, and that is a correctness requirement rather than a
/// nicety: truncating (`nanos / 100`) turns any request under 100 ns into a due
/// time of zero, which `SetWaitableTimer` signals immediately — the guest would
/// sleep for nothing at all. shadPS4's `AccurateSleep` has exactly that
/// truncation.
///
/// Saturates at `-i64::MAX` (~29 000 years) rather than wrapping. The sign is
/// load-bearing: a positive due time is an *absolute* Win32 time in 1601, which
/// signals instantly, so an overflow here would silently defeat every sleep.
#[must_use]
pub fn timer_due_time_100ns(requested: Duration) -> i64 {
    // `units` lands in `0..=i64::MAX`, so the negation cannot overflow.
    let units = i64::try_from(requested.as_nanos().div_ceil(100)).unwrap_or(i64::MAX);
    -units
}

/// Whether a thread may enter the spin phase given how many are already in it.
///
/// Pure so the admission rule is testable without threads. The comparison is
/// against the count *before* this thread joins.
#[must_use]
pub fn spin_admitted(spinners_before: usize, max: usize) -> bool {
    spinners_before < max
}

/// Parse a spin-ceiling override (`RAEEN_SLEEP_SPIN_US`).
///
/// `Some("0")` disables the spin phase entirely — the escape hatch for a host
/// where holding a core is worse than sleeping long. Anything unparseable keeps
/// the default rather than failing a launch over a diagnostic knob.
#[must_use]
pub fn spin_ceiling_from_env(raw: Option<&str>) -> Duration {
    match raw {
        None => DEFAULT_SPIN_CEILING,
        Some(text) => match text.trim().parse::<u64>() {
            Ok(micros) => Duration::from_micros(micros),
            Err(_) => DEFAULT_SPIN_CEILING,
        },
    }
}

// ---------------------------------------------------------------------------
// Requested-vs-actual histogram
// ---------------------------------------------------------------------------

/// Number of buckets in [`SleepHistogram`]; see [`bucket_index`].
pub const BUCKETS: usize = 9;

/// Human-readable span of each bucket, indexed by [`bucket_index`].
pub const BUCKET_LABELS: [&str; BUCKETS] = [
    "0",
    "1-9us",
    "10-99us",
    "100-499us",
    "500-999us",
    "1-1.9ms",
    "2-4.9ms",
    "5-19.9ms",
    ">=20ms",
];

/// Which bucket a requested duration falls in.
///
/// The ladder is deliberately dense below 1 ms: that is where the host floor
/// dominates and where the multiplier is therefore interesting. Above 20 ms the
/// host is within a few percent and one bucket suffices.
#[must_use]
pub fn bucket_index(requested: Duration) -> usize {
    match requested.as_micros() {
        0 => 0,
        1..=9 => 1,
        10..=99 => 2,
        100..=499 => 3,
        500..=999 => 4,
        1_000..=1_999 => 5,
        2_000..=4_999 => 6,
        5_000..=19_999 => 7,
        _ => 8,
    }
}

/// One bucket's aggregate, read out of a [`SleepHistogram`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketSummary {
    /// Index into [`BUCKET_LABELS`].
    pub bucket: usize,
    /// How many sleeps landed in this bucket.
    pub calls: u64,
    /// Sum of what the guest asked for.
    pub requested_ns: u128,
    /// Sum of what the host actually spent.
    pub actual_ns: u128,
    /// Longest single sleep observed in this bucket.
    pub worst_actual_ns: u64,
}

impl BucketSummary {
    /// Actual over requested. `None` when nothing landed here, or when the
    /// bucket is the zero-request bucket (dividing by a zero request is
    /// meaningless, not infinite).
    #[must_use]
    pub fn multiplier(&self) -> Option<f64> {
        if self.calls == 0 || self.requested_ns == 0 {
            return None;
        }
        Some(self.actual_ns as f64 / self.requested_ns as f64)
    }
}

/// Per-bucket counters. Relaxed atomics: this is a diagnostic, and a lost
/// increment under contention is cheaper than serialising every guest sleep.
#[derive(Debug, Default)]
struct Bucket {
    calls: AtomicU64,
    requested_ns: AtomicU64,
    actual_ns: AtomicU64,
    worst_actual_ns: AtomicU64,
}

impl Bucket {
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            requested_ns: AtomicU64::new(0),
            actual_ns: AtomicU64::new(0),
            worst_actual_ns: AtomicU64::new(0),
        }
    }
}

/// Requested-vs-actual sleep durations, bucketed by request size.
///
/// This is the instrument that turns "sleeps cost more than the guest asked"
/// into a number. Without it the multiplier is an assumption: on this host the
/// status quo is 531× at 1 µs but only 1.02× at 10 ms, and which of those a
/// title pays depends entirely on what it requests — which only a histogram of
/// a real run can say.
///
/// Two caveats when reading a real run.
///
/// It records **host parks**, not guest calls. `libkernel`'s sleep handlers
/// slice any request longer than 100 ms so teardown stays observable, so a guest
/// `usleep(500000)` appears as five 100 ms samples rather than one 500 ms
/// sample. Every request at or below 100 ms — the entire interesting range, and
/// every per-frame sleep a title issues — is exactly one park, so only the top
/// bucket is affected, and its multiplier is ~1.0 in any case.
///
/// It also records [`sleep`] only, never [`sleep_until`]. The deadline entry
/// point's callers are internal frame pacing, not guest sleep requests: a
/// `sceVideoOutWaitVblank` wait is anywhere from 0 to a full 16.6 ms period
/// depending only on where in the frame the guest called it, so its
/// actual-over-requested ratio measures the title's frame time and not the
/// host's sleep precision. Including it would also swamp the instrument — at
/// 60 Hz it alone produces ~60 samples a second, enough to trip the report
/// threshold on its own and to bury a title's real millisecond-scale sleeps
/// under pacing waits in the same buckets.
///
/// Constructible standalone so tests can feed synthetic samples; the process
/// instance is [`global`].
#[derive(Debug)]
pub struct SleepHistogram {
    buckets: [Bucket; BUCKETS],
    total_calls: AtomicU64,
}

impl SleepHistogram {
    /// An empty histogram.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buckets: [const { Bucket::new() }; BUCKETS],
            total_calls: AtomicU64::new(0),
        }
    }

    /// Record one sleep. Returns the running total call count, so a caller can
    /// decide when to emit a report without a second counter.
    pub fn record(&self, requested: Duration, actual: Duration) -> u64 {
        let bucket = &self.buckets[bucket_index(requested)];
        bucket.calls.fetch_add(1, Ordering::Relaxed);
        bucket.requested_ns.fetch_add(
            u64::try_from(requested.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        let actual_ns = u64::try_from(actual.as_nanos()).unwrap_or(u64::MAX);
        bucket.actual_ns.fetch_add(actual_ns, Ordering::Relaxed);
        bucket
            .worst_actual_ns
            .fetch_max(actual_ns, Ordering::Relaxed);
        self.total_calls.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Read every non-empty bucket.
    #[must_use]
    pub fn snapshot(&self) -> Vec<BucketSummary> {
        self.buckets
            .iter()
            .enumerate()
            .filter_map(|(bucket, counters)| {
                let calls = counters.calls.load(Ordering::Relaxed);
                (calls > 0).then(|| BucketSummary {
                    bucket,
                    calls,
                    requested_ns: u128::from(counters.requested_ns.load(Ordering::Relaxed)),
                    actual_ns: u128::from(counters.actual_ns.load(Ordering::Relaxed)),
                    worst_actual_ns: counters.worst_actual_ns.load(Ordering::Relaxed),
                })
            })
            .collect()
    }

    /// Total sleeps recorded.
    #[must_use]
    pub fn total_calls(&self) -> u64 {
        self.total_calls.load(Ordering::Relaxed)
    }
}

impl Default for SleepHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Render a snapshot as a fixed-width table.
///
/// Pure over its rows so the report's arithmetic is testable without sleeping —
/// the project forbids wall-clock tests, and a report that silently divides
/// wrongly is worse than no report.
#[must_use]
pub fn format_report(rows: &[BucketSummary]) -> String {
    let mut out = String::from(
        "SLEEP HISTOGRAM (requested vs actual)\n  request      calls    mean req    mean act     mult    worst\n",
    );
    if rows.is_empty() {
        out.push_str("  <no sleeps recorded>\n");
        return out;
    }
    for row in rows {
        let calls = row.calls.max(1);
        let mean_req_us = row.requested_ns as f64 / calls as f64 / 1000.0;
        let mean_act_us = row.actual_ns as f64 / calls as f64 / 1000.0;
        let mult = row
            .multiplier()
            .map_or_else(|| "     n/a".to_owned(), |m| format!("{m:8.2}"));
        out.push_str(&format!(
            "  {:<10} {:>6} {:>9.1}us {:>9.1}us {mult} {:>7.1}ms\n",
            BUCKET_LABELS[row.bucket],
            row.calls,
            mean_req_us,
            mean_act_us,
            row.worst_actual_ns as f64 / 1e6,
        ));
    }
    out
}

/// The process-wide histogram.
pub fn global() -> &'static SleepHistogram {
    static GLOBAL: SleepHistogram = SleepHistogram::new();
    &GLOBAL
}

/// Whether `RAEEN_TIME_SLEEP` armed the histogram.
///
/// Recording costs a bucket index plus four relaxed atomics per sleep, which is
/// noise against a ≥1 µs sleep — but the periodic report is log volume, so it
/// stays opt-in like the sibling `RAEEN_TIME_HLE` instrument.
fn histogram_armed() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| std::env::var_os("RAEEN_TIME_SLEEP").is_some())
}

/// How many recorded sleeps between reports. At the ~120 sleeps/s a title's
/// pacing thread issues, this is a report every ~30 s.
const REPORT_EVERY: u64 = 4096;

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Threads currently inside the spin phase.
static SPINNERS: AtomicUsize = AtomicUsize::new(0);

/// Holds one spin admission slot for as long as it is alive.
struct SpinSlot;

impl SpinSlot {
    /// Claim a slot, or `None` when [`MAX_CONCURRENT_SPINNERS`] are already
    /// spinning. Uses a CAS loop rather than `fetch_add` so an over-limit
    /// attempt never transiently blocks a thread that would have been admitted.
    fn claim() -> Option<Self> {
        let mut current = SPINNERS.load(Ordering::Relaxed);
        loop {
            if !spin_admitted(current, MAX_CONCURRENT_SPINNERS) {
                return None;
            }
            match SPINNERS.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(Self),
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for SpinSlot {
    fn drop(&mut self) {
        SPINNERS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Whether a spin slot is currently free. Advisory — [`SpinSlot::claim`] is the
/// authority — and only used to build [`HostCapabilities`].
fn spin_slot_available() -> bool {
    spin_admitted(SPINNERS.load(Ordering::Relaxed), MAX_CONCURRENT_SPINNERS)
}

/// The configured spin ceiling, read from the environment once.
fn spin_ceiling() -> Duration {
    static CEILING: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *CEILING
        .get_or_init(|| spin_ceiling_from_env(std::env::var("RAEEN_SLEEP_SPIN_US").ok().as_deref()))
}

/// This host's capabilities right now.
fn capabilities() -> HostCapabilities {
    HostCapabilities {
        high_resolution_timer: high_resolution_timer_supported(),
        spin_ceiling: spin_ceiling(),
        spin_slot_available: spin_slot_available(),
    }
}

/// `PAUSE`-spin until `deadline`. Never yields — see the module docs: a yield
/// here measured 60–190 ms on an oversubscribed host.
fn pause_spin_until(deadline: Instant) {
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

/// Park until `deadline`, never returning early.
///
/// The loop is the never-early guarantee: after any park, whatever is left is
/// either spun out (if within [`NEVER_EARLY_TOP_UP`]) or parked again. In
/// practice a high-resolution timer overshoots, so this runs once.
fn park_until(deadline: Instant, use_timer: bool) {
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return;
        }
        if remaining <= NEVER_EARLY_TOP_UP {
            pause_spin_until(deadline);
            return;
        }
        if use_timer && park_on_high_resolution_timer(remaining) {
            continue;
        }
        std::thread::sleep(remaining);
    }
}

/// Carry out an already-chosen plan against `deadline`.
///
/// The deadline is the authority, not the duration the plan was chosen from:
/// every arm below aims at the same absolute instant, so no strategy can drift
/// by re-reading the clock on its way in.
fn execute(chosen: SleepPlan, deadline: Instant) {
    match chosen {
        SleepPlan::Yield => std::thread::yield_now(),
        SleepPlan::Spin => {
            // The slot may have been taken between `capabilities()` and here;
            // parking is always a correct answer, so fall through rather than
            // waiting for a slot.
            match SpinSlot::claim() {
                Some(_slot) => pause_spin_until(deadline),
                None => park_until(deadline, high_resolution_timer_supported()),
            }
        }
        SleepPlan::HighResolutionTimer => park_until(deadline, true),
        SleepPlan::Fallback => park_until(deadline, false),
    }
}

/// Block until `deadline`, then return. Never returns early; returns at once if
/// the deadline has already passed.
///
/// The deadline-shaped entry point, for a caller pacing to a fixed grid rather
/// than sleeping for a length of time — `sceVideoOutWaitVblank` waiting for
/// edge *n* of the vblank schedule, say. Such a caller must not be made to
/// subtract and call [`sleep`]: the gap between the caller reading the clock
/// and [`sleep`] reading it again lands *past* the deadline, so every wait
/// overshoots by that gap and an epoch-anchored grid drifts by exactly as much
/// as the per-call relative sleeps it exists to avoid.
///
/// **Not recorded in the [`SleepHistogram`]**, unlike [`sleep`] — see that
/// type's documentation for why.
pub fn sleep_until(deadline: Instant) {
    let remaining = deadline.saturating_duration_since(Instant::now());
    let Some(chosen) = plan_until(remaining, capabilities()) else {
        return;
    };
    execute(chosen, deadline);
}

/// Sleep the calling thread for at least `requested`.
///
/// The guest-facing entry point: `TimeSubsystem::sleep` forwards to it, so every
/// guest `usleep`/`nanosleep`/`sleep` shares one measured strategy. Never
/// returns before `requested` has elapsed on the monotonic clock. A caller
/// pacing to an absolute deadline wants [`sleep_until`] instead — both run the
/// same strategy table, and only this one is recorded in the histogram.
pub fn sleep(requested: Duration) {
    let start = Instant::now();
    execute(plan(requested, capabilities()), start + requested);
    if histogram_armed() {
        let count = global().record(requested, start.elapsed());
        if count.is_multiple_of(REPORT_EVERY) {
            tracing::info!("{}", format_report(&global().snapshot()));
        }
    }
}

// ---------------------------------------------------------------------------
// Windows backend
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod windows_backend {
    use std::time::Duration;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::Media::{timeBeginPeriod, timeEndPeriod};
    use windows_sys::Win32::System::Threading::{
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CREATE_WAITABLE_TIMER_MANUAL_RESET,
        CreateWaitableTimerExW, SetWaitableTimer, TIMER_MODIFY_STATE, WaitForSingleObject,
    };

    /// Wait indefinitely for one object; not exported by windows-sys 0.59.
    const INFINITE: u32 = 0xFFFF_FFFF;

    /// Object-access right to wait on a handle; not exported by windows-sys 0.59.
    const SYNCHRONIZATION: u32 = 0x0010_0000;

    /// The millisecond period requested from the multimedia timer.
    const TIMER_PERIOD_MS: u32 = 1;

    /// One manual-reset high-resolution waitable timer, owned by one thread.
    ///
    /// Thread-local and cached for the thread's life: shadPS4 creates and closes
    /// a kernel handle on *every* sleep, which is two extra syscalls on the hot
    /// path. The handle must not be shared — several guest threads park
    /// concurrently, and a shared manual-reset timer would wake all of them.
    struct HighResTimer(HANDLE);

    // SAFETY: the handle is only ever touched from the thread whose
    // thread-local owns it; this impl exists solely because `thread_local!`
    // requires the type to be sized and droppable there, not to share it.
    impl HighResTimer {
        fn new() -> Option<Self> {
            // SAFETY: an unnamed manual-reset timer; null attributes and name
            // are valid. The high-resolution flag is ignored by builds that
            // predate it, degrading to a normal (coarse) timer rather than
            // failing — which is why `probe` checks creation, not precision.
            let handle = unsafe {
                CreateWaitableTimerExW(
                    std::ptr::null(),
                    std::ptr::null(),
                    CREATE_WAITABLE_TIMER_MANUAL_RESET | CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                    TIMER_MODIFY_STATE | SYNCHRONIZATION,
                )
            };
            (!handle.is_null()).then_some(Self(handle))
        }

        /// Arm for `remaining` and wait. `false` means `SetWaitableTimer`
        /// failed and nothing was waited on, so the caller must fall back.
        fn park(&self, remaining: Duration) -> bool {
            let due = super::timer_due_time_100ns(remaining);
            // SAFETY: `self.0` is a live timer owned by this thread; `due`
            // points at a local `i64`; no APC completion routine is used.
            let armed = unsafe { SetWaitableTimer(self.0, &due, 0, None, std::ptr::null(), 0) };
            if armed == 0 {
                return false;
            }
            // SAFETY: waiting on this thread's own timer, which signals at the
            // due time, so INFINITE cannot hang.
            unsafe { WaitForSingleObject(self.0, INFINITE) };
            true
        }
    }

    impl Drop for HighResTimer {
        fn drop(&mut self) {
            // SAFETY: closing a handle this thread created and owns.
            unsafe { CloseHandle(self.0) };
        }
    }

    thread_local! {
        static TIMER: Option<HighResTimer> = HighResTimer::new();
    }

    /// Whether a waitable timer can be created at all on this host. Probed once
    /// per process on a throwaway handle so the answer is available before any
    /// thread has its own.
    pub fn supported() -> bool {
        static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *SUPPORTED.get_or_init(|| {
            let probe = HighResTimer::new();
            let ok = probe.is_some();
            if !ok {
                tracing::warn!(
                    "no high-resolution waitable timer on this host — guest sleeps fall back to \
                     std::thread::sleep"
                );
            }
            drop(probe);
            ok
        })
    }

    /// Park this thread on its own high-resolution timer. `false` when the
    /// thread has no timer or arming failed; the caller then parks another way.
    pub fn park(remaining: Duration) -> bool {
        TIMER.with(|timer| timer.as_ref().is_some_and(|timer| timer.park(remaining)))
    }

    /// Raise the process timer resolution to 1 ms. Idempotent; `false` when the
    /// request was rejected.
    pub fn arm_timer_resolution() -> bool {
        static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ARMED.get_or_init(|| {
            // SAFETY: no preconditions; returns TIMERR_NOERROR (0) on success.
            // Released by `release_timer_resolution`, and unconditionally by
            // Windows when the process exits.
            let result = unsafe { timeBeginPeriod(TIMER_PERIOD_MS) };
            if result == 0 {
                tracing::info!(
                    "timer resolution raised to {TIMER_PERIOD_MS} ms — condition-variable timed \
                     waits are otherwise quantised to the ~15.6 ms system tick"
                );
                true
            } else {
                tracing::warn!(
                    "timeBeginPeriod({TIMER_PERIOD_MS}) rejected ({result}) — kernel wait slices \
                     stay at the default ~15.6 ms granularity"
                );
                false
            }
        })
    }

    /// Give back the raised timer resolution.
    ///
    /// Pairs [`arm_timer_resolution`]. Windows drops a process's request when
    /// the process exits, so this exists for a host that wants the system back
    /// at its default while the emulator is still running (an idle Shell, say)
    /// rather than as a leak fix.
    pub fn release_timer_resolution() {
        // SAFETY: no preconditions. Harmless if `timeBeginPeriod` was never
        // called or already released — Windows returns TIMERR_NOCANDO.
        unsafe { timeEndPeriod(TIMER_PERIOD_MS) };
    }
}

#[cfg(windows)]
fn high_resolution_timer_supported() -> bool {
    windows_backend::supported()
}

#[cfg(windows)]
fn park_on_high_resolution_timer(remaining: Duration) -> bool {
    windows_backend::park(remaining)
}

/// Raise the process timer resolution so *condition-variable* timed waits stop
/// being quantised to the ~15.6 ms system tick.
///
/// This is the only lever for the kernel wait slices. `Condvar::wait_timeout`
/// and `parking_lot`'s `wait_for` both round their timeout to the system tick,
/// so every event-flag, equeue and semaphore wait shorter than one tick costs a
/// full tick — measured 15 394 µs for a 100 µs wait, and 15 564 µs for a 1 ms
/// wait. A waitable timer cannot substitute: a notifiable wait must remain a
/// condition-variable wait or the notify path is lost, and preserving prompt
/// notification is worth more than precision.
///
/// Measured effect on this host: a 1 ms `wait_timeout` drops 15 564 µs →
/// 3 020 µs, a 10 ms one 15 454 µs → 10 457 µs. The residual is condition
/// variable wake latency, not timer granularity, so this narrows the loss
/// rather than eliminating it.
///
/// Idempotent, and safe to call from any thread on any host. Returns whether
/// the process is now running at raised resolution. Always `false` off Windows,
/// where no such global setting exists.
pub fn arm_high_resolution_timer() -> bool {
    #[cfg(windows)]
    {
        windows_backend::arm_timer_resolution()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// Release what [`arm_high_resolution_timer`] raised. No-op off Windows, and
/// no-op if the resolution was never raised.
pub fn release_high_resolution_timer() {
    #[cfg(windows)]
    {
        windows_backend::release_timer_resolution();
    }
}

#[cfg(not(windows))]
fn high_resolution_timer_supported() -> bool {
    // No portable equivalent. `std::thread::sleep` maps to `nanosleep`, which
    // is already nanosecond-granular on Linux/macOS, so the fallback path is
    // not a degradation there.
    false
}

#[cfg(not(windows))]
fn park_on_high_resolution_timer(_remaining: Duration) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole strategy table, as a decision rather than as timing. Every row
    /// is the answer this module must give for one (request, host) pair.
    #[test]
    fn plan_covers_every_request_and_host_combination() {
        let full = HostCapabilities {
            high_resolution_timer: true,
            spin_ceiling: DEFAULT_SPIN_CEILING,
            spin_slot_available: true,
        };

        // Zero is a yield whatever the host offers — even fully degraded.
        assert_eq!(plan(Duration::ZERO, full), SleepPlan::Yield);
        assert_eq!(
            plan(Duration::ZERO, HostCapabilities::degraded()),
            SleepPlan::Yield
        );

        // At or below the ceiling, spin: no park can honour 100 us.
        assert_eq!(plan(Duration::from_micros(1), full), SleepPlan::Spin);
        assert_eq!(plan(Duration::from_micros(100), full), SleepPlan::Spin);

        // One microsecond past the ceiling, park.
        assert_eq!(
            plan(Duration::from_micros(101), full),
            SleepPlan::HighResolutionTimer
        );
        assert_eq!(
            plan(Duration::from_millis(16), full),
            SleepPlan::HighResolutionTimer
        );

        // No slot free: a short request parks instead of spinning. This is the
        // admission limit doing its job, not a fallback.
        let no_slot = HostCapabilities {
            spin_slot_available: false,
            ..full
        };
        assert_eq!(
            plan(Duration::from_micros(1), no_slot),
            SleepPlan::HighResolutionTimer
        );

        // Spin disabled by configuration behaves like no slot.
        let no_spin = HostCapabilities {
            spin_ceiling: Duration::ZERO,
            ..full
        };
        assert_eq!(
            plan(Duration::from_micros(1), no_spin),
            SleepPlan::HighResolutionTimer
        );

        // No high-resolution timer: everything above the ceiling falls back,
        // and the fallback must still be reachable with no spin budget at all.
        let no_timer = HostCapabilities {
            high_resolution_timer: false,
            ..full
        };
        assert_eq!(
            plan(Duration::from_millis(5), no_timer),
            SleepPlan::Fallback
        );
        assert_eq!(
            plan(Duration::from_millis(5), HostCapabilities::degraded()),
            SleepPlan::Fallback
        );
        assert_eq!(
            plan(Duration::from_micros(1), HostCapabilities::degraded()),
            SleepPlan::Fallback
        );
    }

    /// The due time must round UP. Truncating turns a sub-100 ns request into
    /// due time zero, which `SetWaitableTimer` signals immediately — the guest
    /// sleeps for nothing, which is a correctness bug and not a rounding nit.
    #[test]
    fn timer_due_time_rounds_up_and_never_reaches_zero() {
        assert_eq!(timer_due_time_100ns(Duration::from_nanos(1)), -1);
        assert_eq!(timer_due_time_100ns(Duration::from_nanos(99)), -1);
        assert_eq!(timer_due_time_100ns(Duration::from_nanos(100)), -1);
        assert_eq!(timer_due_time_100ns(Duration::from_nanos(101)), -2);
        assert_eq!(timer_due_time_100ns(Duration::from_micros(1)), -10);
        assert_eq!(timer_due_time_100ns(Duration::from_millis(1)), -10_000);
        assert_eq!(timer_due_time_100ns(Duration::from_secs(1)), -10_000_000);

        // Zero is the one input that may be zero; `plan` never routes it here.
        assert_eq!(timer_due_time_100ns(Duration::ZERO), 0);

        // A wild guest value saturates instead of wrapping into a POSITIVE due
        // time, which Win32 would read as an absolute time in 1601.
        let saturated = timer_due_time_100ns(Duration::from_secs(u64::MAX));
        assert!(saturated < 0, "due time must stay negative (relative)");
        assert_eq!(saturated, -i64::MAX);
    }

    /// A deadline wait and a duration wait must agree on every input but one.
    ///
    /// Zero is that input: `usleep(0)` is a yield request, but a deadline that
    /// has already passed is a wait that is merely late, and yielding there
    /// would cost a frame-pacing caller the *next* edge as well. Everywhere
    /// else the two entry points must resolve to the same strategy, or the
    /// vblank path and the guest sleep path would drift back apart — which is
    /// the whole reason there is one module here rather than two.
    #[test]
    fn plan_until_matches_plan_except_on_a_passed_deadline() {
        let full = HostCapabilities {
            high_resolution_timer: true,
            spin_ceiling: DEFAULT_SPIN_CEILING,
            spin_slot_available: true,
        };

        assert_eq!(plan(Duration::ZERO, full), SleepPlan::Yield);
        assert_eq!(plan_until(Duration::ZERO, full), None);
        assert_eq!(
            plan_until(Duration::ZERO, HostCapabilities::degraded()),
            None
        );

        let no_slot = HostCapabilities {
            spin_slot_available: false,
            ..full
        };
        for remaining in [
            Duration::from_nanos(1),
            Duration::from_micros(1),
            Duration::from_micros(100),
            Duration::from_micros(101),
            // A 60 Hz vblank period: the request the collapsed video_out path
            // actually issues, and it must park on the timer.
            Duration::from_nanos(1_000_000_000 / 60),
            Duration::from_millis(100),
        ] {
            for caps in [full, no_slot, HostCapabilities::degraded()] {
                assert_eq!(
                    plan_until(remaining, caps),
                    Some(plan(remaining, caps)),
                    "deadline and duration waits must choose alike for {remaining:?}"
                );
            }
        }

        assert_eq!(
            plan_until(Duration::from_nanos(1_000_000_000 / 60), full),
            Some(SleepPlan::HighResolutionTimer),
            "a vblank-length wait parks rather than spinning a core for 16.6ms"
        );
    }

    /// The admission rule, as arithmetic. Guards ~30 guest threads from
    /// collectively spinning a 12-core host.
    #[test]
    fn spin_admission_stops_at_the_limit() {
        assert!(spin_admitted(0, MAX_CONCURRENT_SPINNERS));
        assert!(spin_admitted(
            MAX_CONCURRENT_SPINNERS - 1,
            MAX_CONCURRENT_SPINNERS
        ));
        assert!(!spin_admitted(
            MAX_CONCURRENT_SPINNERS,
            MAX_CONCURRENT_SPINNERS
        ));
        assert!(!spin_admitted(30, MAX_CONCURRENT_SPINNERS));
        // A zero limit disables spinning entirely.
        assert!(!spin_admitted(0, 0));
    }

    #[test]
    fn spin_ceiling_env_parses_and_defaults() {
        assert_eq!(spin_ceiling_from_env(None), DEFAULT_SPIN_CEILING);
        assert_eq!(
            spin_ceiling_from_env(Some("250")),
            Duration::from_micros(250)
        );
        assert_eq!(
            spin_ceiling_from_env(Some(" 40 ")),
            Duration::from_micros(40)
        );
        // Zero is meaningful: it disables the spin phase.
        assert_eq!(spin_ceiling_from_env(Some("0")), Duration::ZERO);
        // Garbage keeps the default rather than failing a launch.
        assert_eq!(spin_ceiling_from_env(Some("soon")), DEFAULT_SPIN_CEILING);
        assert_eq!(spin_ceiling_from_env(Some("")), DEFAULT_SPIN_CEILING);
        assert_eq!(spin_ceiling_from_env(Some("-5")), DEFAULT_SPIN_CEILING);
    }

    /// Bucket boundaries, exhaustively at the edges. A histogram that
    /// mis-buckets reports a wrong multiplier, which is worse than reporting
    /// none — it would send the next investigation at the wrong request size.
    #[test]
    fn bucket_boundaries_are_exact() {
        assert_eq!(bucket_index(Duration::ZERO), 0);
        assert_eq!(
            bucket_index(Duration::from_nanos(1)),
            0,
            "sub-microsecond rounds into the zero bucket"
        );
        assert_eq!(bucket_index(Duration::from_micros(1)), 1);
        assert_eq!(bucket_index(Duration::from_micros(9)), 1);
        assert_eq!(bucket_index(Duration::from_micros(10)), 2);
        assert_eq!(bucket_index(Duration::from_micros(99)), 2);
        assert_eq!(bucket_index(Duration::from_micros(100)), 3);
        assert_eq!(bucket_index(Duration::from_micros(499)), 3);
        assert_eq!(bucket_index(Duration::from_micros(500)), 4);
        assert_eq!(bucket_index(Duration::from_micros(999)), 4);
        assert_eq!(bucket_index(Duration::from_millis(1)), 5);
        assert_eq!(bucket_index(Duration::from_micros(1_999)), 5);
        assert_eq!(bucket_index(Duration::from_millis(2)), 6);
        assert_eq!(bucket_index(Duration::from_micros(4_999)), 6);
        assert_eq!(bucket_index(Duration::from_millis(5)), 7);
        assert_eq!(bucket_index(Duration::from_micros(19_999)), 7);
        assert_eq!(bucket_index(Duration::from_millis(20)), 8);
        assert_eq!(bucket_index(Duration::from_secs(1)), 8);
        // Every index the ladder can produce has a label.
        for micros in [0u64, 1, 10, 100, 500, 1_000, 2_000, 5_000, 20_000] {
            let index = bucket_index(Duration::from_micros(micros));
            assert!(index < BUCKETS);
            assert!(!BUCKET_LABELS[index].is_empty());
        }
    }

    /// The histogram aggregates synthetic samples — a synthetic clock, not a
    /// real one, so the test is deterministic and instant.
    #[test]
    fn histogram_aggregates_requested_versus_actual() {
        let hist = SleepHistogram::new();
        assert_eq!(hist.total_calls(), 0);
        assert!(hist.snapshot().is_empty());

        // Three 1 us requests that each cost 500 us: the status-quo 500x floor.
        for _ in 0..3 {
            hist.record(Duration::from_micros(1), Duration::from_micros(500));
        }
        // One 10 ms request that cost 10.2 ms: near-exact.
        hist.record(Duration::from_millis(10), Duration::from_micros(10_200));

        assert_eq!(hist.total_calls(), 4);
        let rows = hist.snapshot();
        assert_eq!(rows.len(), 2, "only non-empty buckets are reported");

        let tiny = rows.iter().find(|r| r.bucket == 1).expect("1-9us bucket");
        assert_eq!(tiny.calls, 3);
        assert_eq!(tiny.requested_ns, 3_000);
        assert_eq!(tiny.actual_ns, 1_500_000);
        assert_eq!(tiny.worst_actual_ns, 500_000);
        let multiplier = tiny.multiplier().expect("a nonzero request has a ratio");
        assert!(
            (multiplier - 500.0).abs() < 1e-9,
            "1us requested / 500us actual is a 500x multiplier, got {multiplier}"
        );

        let big = rows
            .iter()
            .find(|r| r.bucket == 7)
            .expect("5-19.9ms bucket");
        assert_eq!(big.calls, 1);
        let multiplier = big.multiplier().expect("a nonzero request has a ratio");
        assert!(
            (multiplier - 1.02).abs() < 1e-9,
            "10ms requested / 10.2ms actual is 1.02x, got {multiplier}"
        );
    }

    /// A zero-length request has no meaningful multiplier. Reporting `inf` (or
    /// panicking on a divide) would make the whole report unusable for a title
    /// that calls `usleep(0)` as a yield.
    #[test]
    fn zero_request_has_no_multiplier() {
        let hist = SleepHistogram::new();
        hist.record(Duration::ZERO, Duration::from_micros(3));
        let rows = hist.snapshot();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bucket, 0);
        assert_eq!(rows[0].multiplier(), None);
        let report = format_report(&rows);
        assert!(report.contains("n/a"), "zero bucket reports n/a: {report}");
        assert!(
            !report.contains("inf"),
            "no infinities in the report: {report}"
        );
    }

    #[test]
    fn report_formats_rows_and_the_empty_case() {
        assert!(format_report(&[]).contains("<no sleeps recorded>"));

        let rows = [BucketSummary {
            bucket: 3,
            calls: 16_277,
            requested_ns: 16_277 * 200_000,
            actual_ns: 16_277 * 8_300_000,
            worst_actual_ns: 41_000_000,
        }];
        let report = format_report(&rows);
        assert!(report.contains("100-499us"), "{report}");
        assert!(report.contains("16277"), "{report}");
        // 200us requested vs 8300us actual is 41.5x.
        assert!(
            report.contains("41.50"),
            "multiplier must be shown: {report}"
        );
        assert!(report.contains("41.0ms"), "worst case in ms: {report}");
    }

    /// Arming and releasing the process timer resolution must be safe to call
    /// repeatedly and in any order, on any host. This asserts the contract, not
    /// a resolution change — the latter is unobservable without timing.
    #[test]
    fn timer_resolution_arm_is_idempotent_and_release_is_safe() {
        let first = arm_high_resolution_timer();
        let second = arm_high_resolution_timer();
        assert_eq!(first, second, "arming is idempotent");
        #[cfg(not(windows))]
        assert!(!first, "no process timer resolution exists off Windows");
        release_high_resolution_timer();
        // Releasing twice, and releasing something never armed, are both no-ops.
        release_high_resolution_timer();
    }
}
