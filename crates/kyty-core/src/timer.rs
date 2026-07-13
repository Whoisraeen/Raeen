//! Port of Kyty's `Core::Timer`
//! (`reference/kyty/source/include/Kyty/Core/Timer.h` +
//! `reference/kyty/source/lib/Core/src/Timer.cpp`).
//!
//! Kyty's `Timer` is a pausable stopwatch built on the platform's
//! `Kyty::Sys::SysTimer` primitives (`sys_query_performance_frequency` /
//! `sys_query_performance_counter`, i.e. Windows `QueryPerformanceCounter` /
//! `QueryPerformanceFrequency` or the POSIX `clock_gettime`-based
//! equivalent). It records a `m_StartTime` tick on `Start()`; while running,
//! elapsed ticks are `now - m_StartTime`. `Pause()` freezes elapsed time by
//! snapshotting `m_PauseTime`; `Resume()` folds the paused interval back into
//! `m_StartTime` (`m_StartTime += now - m_PauseTime`) so that time spent
//! paused is excluded from all subsequent readings.
//!
//! Rust mapping: the OS performance-counter pair is replaced by
//! [`std::time::Instant`], which is std's monotonic, high-resolution clock
//! and needs no FFI. Because `Instant` does not expose a raw
//! platform "tick" value comparable across process runs, this port defines
//! its own tick domain: ticks are nanoseconds elapsed since a process-wide
//! lazily-initialized epoch `Instant` (`epoch()`), and the "frequency" is
//! therefore fixed at `1_000_000_000` (nanoseconds per second). This
//! preserves the original's observable contract exactly (`get_ticks() as f64
//! / get_frequency() as f64` == elapsed seconds, `is_paused()` freezes the
//! reading, `resume()` excludes paused time) while using only `std`.
//!
//! Method names are the `snake_case` equivalents of Kyty's `PascalCase` API
//! (`Start` -> `start`, `GetTimeMs` -> `get_time_ms`, ...), per this crate's
//! porting convention.

use crate::exit_if;
use std::sync::OnceLock;
use std::time::Instant;

/// Ticks-per-second for the tick domain used by this port (nanoseconds).
/// Equivalent to what Kyty's `sys_query_performance_frequency` would return
/// on a platform with nanosecond-resolution counters.
const FREQUENCY: u64 = 1_000_000_000;

/// Process-wide monotonic epoch that [`Timer::query_performance_counter`]
/// measures ticks from. Lazily initialized on first use so that tick values
/// are comparable (via subtraction) for the lifetime of the process, exactly
/// like a real hardware performance counter.
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Rust port of `Kyty::Core::Timer`. A pausable stopwatch: [`Timer::start`]
/// begins timing, [`Timer::pause`]/[`Timer::resume`] can suspend and resume
/// it (excluding the paused interval from elapsed time), and `get_time_*`/
/// `get_ticks` read the elapsed time so far.
#[derive(Debug)]
pub struct Timer {
    is_paused: bool,
    frequency: u64,
    start_time: u64,
    pause_time: u64,
}

impl Timer {
    /// Kyty `Timer::Timer()`: captures the tick frequency; the timer starts
    /// out paused (matching the C++ default member initializer
    /// `m_is_paused = true`) and must be started with [`Timer::start`].
    #[must_use]
    pub fn new() -> Self {
        Self { is_paused: true, frequency: Self::query_performance_frequency(), start_time: 0, pause_time: 0 }
    }

    /// Kyty `Timer::Start()`: (re)starts the stopwatch from now, discarding
    /// any previous run.
    pub fn start(&mut self) {
        self.start_time = Self::query_performance_counter();
        self.is_paused = false;
    }

    /// Kyty `Timer::Pause()`: freezes the elapsed-time reading. `EXIT_IF`s
    /// (panics) if the timer is already paused.
    pub fn pause(&mut self) {
        exit_if!(self.is_paused);

        self.pause_time = Self::query_performance_counter();
        self.is_paused = true;
    }

    /// Kyty `Timer::Resume()`: un-freezes the stopwatch, folding the time
    /// spent paused back into the start reference so it is excluded from
    /// elapsed-time readings. `EXIT_IF`s (panics) if the timer is not
    /// currently paused.
    pub fn resume(&mut self) {
        exit_if!(!self.is_paused);

        let current_time = Self::query_performance_counter();
        self.start_time += current_time - self.pause_time;
        self.is_paused = false;
    }

    /// Kyty `Timer::IsPaused() const`.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    /// Kyty `Timer::GetTimeMs() const`: elapsed time in milliseconds.
    #[must_use]
    pub fn get_time_ms(&self) -> f64 {
        1000.0 * (self.get_ticks() as f64) / (self.frequency as f64)
    }

    /// Kyty `Timer::GetTimeS() const`: elapsed time in seconds.
    #[must_use]
    pub fn get_time_s(&self) -> f64 {
        (self.get_ticks() as f64) / (self.frequency as f64)
    }

    /// Kyty `Timer::GetTicks() const`: elapsed time in raw ticks (see module
    /// docs for the tick domain used by this port).
    #[must_use]
    pub fn get_ticks(&self) -> u64 {
        if self.is_paused {
            self.pause_time - self.start_time
        } else {
            Self::query_performance_counter() - self.start_time
        }
    }

    /// Kyty `Timer::GetFrequency() const`: ticks-per-second for this timer.
    #[must_use]
    pub fn get_frequency(&self) -> u64 {
        self.frequency
    }

    /// Kyty `Timer::QueryPerformanceFrequency()` (static).
    #[must_use]
    pub fn query_performance_frequency() -> u64 {
        FREQUENCY
    }

    /// Kyty `Timer::QueryPerformanceCounter()` (static).
    #[must_use]
    pub fn query_performance_counter() -> u64 {
        epoch().elapsed().as_nanos() as u64
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn new_timer_starts_paused_with_zero_elapsed() {
        let t = Timer::new();
        assert!(t.is_paused());
        assert_eq!(t.get_ticks(), 0);
        assert_eq!(t.get_time_ms(), 0.0);
        assert_eq!(t.get_time_s(), 0.0);
    }

    #[test]
    fn frequency_matches_static_query() {
        let t = Timer::new();
        assert_eq!(t.get_frequency(), Timer::query_performance_frequency());
        assert_eq!(t.get_frequency(), FREQUENCY);
    }

    #[test]
    fn start_unpauses_and_time_advances() {
        let mut t = Timer::new();
        t.start();
        assert!(!t.is_paused());
        sleep(Duration::from_millis(20));
        let ticks = t.get_ticks();
        assert!(ticks > 0);
        // Sanity: at least ~10ms worth of ticks elapsed (generous tolerance
        // for slow/loaded CI machines), and get_time_ms/get_time_s agree
        // with get_ticks/frequency.
        assert!(t.get_time_ms() >= 10.0);
        assert!((t.get_time_s() - t.get_time_ms() / 1000.0).abs() < 1e-6);
    }

    #[test]
    fn pause_freezes_elapsed_time() {
        let mut t = Timer::new();
        t.start();
        sleep(Duration::from_millis(15));
        t.pause();
        assert!(t.is_paused());
        let frozen = t.get_ticks();
        sleep(Duration::from_millis(15));
        // While paused, repeated reads must not advance.
        assert_eq!(t.get_ticks(), frozen);
    }

    #[test]
    fn resume_excludes_paused_interval() {
        let mut t = Timer::new();
        t.start();
        sleep(Duration::from_millis(10));
        t.pause();
        let before_pause = t.get_ticks();
        sleep(Duration::from_millis(50)); // time that should NOT count
        t.resume();
        assert!(!t.is_paused());
        // Immediately after resume, elapsed ticks should be close to what
        // they were right before pausing (the long sleep must be excluded),
        // not inflated by the ~50ms spent paused.
        let after_resume = t.get_ticks();
        assert!(
            after_resume >= before_pause,
            "elapsed time must not go backwards across resume"
        );
        let growth_ns = after_resume - before_pause;
        assert!(
            growth_ns < Duration::from_millis(40).as_nanos() as u64,
            "paused interval leaked into elapsed time: grew by {growth_ns}ns"
        );
    }

    #[test]
    #[should_panic]
    fn pause_while_already_paused_panics() {
        let mut t = Timer::new();
        t.start();
        t.pause();
        t.pause(); // EXIT_IF(m_is_paused) equivalent
    }

    #[test]
    #[should_panic]
    fn resume_while_not_paused_panics() {
        let mut t = Timer::new();
        t.start();
        t.resume(); // EXIT_IF(!m_is_paused) equivalent
    }

    #[test]
    fn restarting_discards_previous_run() {
        let mut t = Timer::new();
        t.start();
        sleep(Duration::from_millis(20));
        t.pause();
        assert!(t.get_ticks() > 0);

        // Start() again resets the stopwatch, matching Kyty's behavior of
        // unconditionally overwriting m_StartTime.
        t.start();
        assert!(!t.is_paused());
        let ticks_just_after_restart = t.get_ticks();
        assert!(ticks_just_after_restart < Duration::from_millis(10).as_nanos() as u64);
    }

    #[test]
    fn static_query_performance_counter_is_monotonic() {
        let a = Timer::query_performance_counter();
        sleep(Duration::from_millis(1));
        let b = Timer::query_performance_counter();
        assert!(b >= a);
    }
}
