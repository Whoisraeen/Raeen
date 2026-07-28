//! The unified GPU completion clock (`RAEEN_UNIFIED_GPU_CLOCK`, default OFF).
//!
//! Checklist item 5, step 3. RELEASE_MEM GPU-timestamp fences were written by
//! TWO disagreeing clocks: the HLE's eager submit-time writer used the
//! session-monotonic kernel clock in nanoseconds, while the GPU worker's
//! in-stream `cp_op_release_mem` used a process-local `1, 2, 3, …` counter —
//! so the same fence address was double-written with values from different
//! domains, and whichever write landed last decided what the guest polled. A
//! guest comparing a fresh fence against an earlier ns-scale sample could see
//! its "clock" jump backward by twelve orders of magnitude.
//!
//! Under the gate BOTH writers draw from this one process-global clock:
//! monotonic nanoseconds, forced strictly increasing (and therefore never
//! zero), the shape a hardware GPU core-clock counter has. With the gate off
//! (the default) nothing here is consulted and both legacy clocks behave
//! bit-identically to before — the flip is A/B territory (the ledger flags
//! ASTRO.BOT's timestamp-fence hang as the regression risk), so the default
//! stays OFF until a live title verifies it.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// `RAEEN_UNIFIED_GPU_CLOCK=1` (transition gate, default OFF): route every
/// RELEASE_MEM GPU-timestamp write — the HLE's eager submit-time write AND the
/// GPU worker's in-stream write — through [`next_unified_gpu_timestamp`].
/// Read per call, like the `RAEEN_TRACE_*` gates, so tests can flip it per
/// case.
#[must_use]
pub fn unified_gpu_clock_enabled() -> bool {
    std::env::var_os("RAEEN_UNIFIED_GPU_CLOCK").is_some()
}

/// Nanoseconds of process-monotonic time since this clock was first read.
fn monotonic_nanos() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    let start = *START.get_or_init(Instant::now);
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Next value for a RELEASE_MEM GPU-timestamp fence under the unified-clock
/// gate: process-monotonic nanoseconds, forced strictly increasing across
/// calls from EVERY writer (and therefore never zero), the way the hardware
/// clock counter is. One `AtomicU64` is the whole synchronization: two writers
/// can never mint the same or a decreasing value.
#[must_use]
pub fn next_unified_gpu_timestamp() -> u64 {
    static CLOCK: AtomicU64 = AtomicU64::new(0);
    let now = monotonic_nanos();
    let mut next = now.max(1);
    let _ = CLOCK.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |prev| {
        next = now.max(prev.saturating_add(1)).max(1);
        Some(next)
    });
    next
}

/// The timestamp source installed into every session `CommandProcessor`
/// (`CommandProcessor::set_timestamp_source`): gate ON → the unified clock;
/// gate OFF → `None`, and the CP falls back to its legacy process-local
/// counter — bit-identical default behavior.
pub(crate) fn cp_timestamp_source() -> Option<u64> {
    unified_gpu_clock_enabled().then(next_unified_gpu_timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the env-gate tests in this module (the gate is process
    /// state) — same pattern as raeen-hle's `SIDEFX_ENV_LOCK`.
    static CLOCK_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn unified_timestamps_are_nonzero_and_strictly_increasing() {
        let mut previous = 0u64;
        for _ in 0..1000 {
            let ts = next_unified_gpu_timestamp();
            assert!(ts > previous, "strictly increasing: {previous} -> {ts}");
            previous = ts;
        }
        assert_ne!(previous, 0);
    }

    #[test]
    fn the_cp_source_declines_with_the_gate_off_and_ticks_with_it_on() {
        let _guard = CLOCK_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        struct EnvReset;
        impl Drop for EnvReset {
            fn drop(&mut self) {
                unsafe { std::env::remove_var("RAEEN_UNIFIED_GPU_CLOCK") };
            }
        }
        let _reset = EnvReset;

        unsafe { std::env::remove_var("RAEEN_UNIFIED_GPU_CLOCK") };
        assert!(!unified_gpu_clock_enabled());
        assert_eq!(
            cp_timestamp_source(),
            None,
            "gate off: the CP keeps its legacy counter"
        );

        unsafe { std::env::set_var("RAEEN_UNIFIED_GPU_CLOCK", "1") };
        assert!(unified_gpu_clock_enabled());
        let first = cp_timestamp_source().expect("gate on: the unified clock");
        let second = cp_timestamp_source().expect("gate on: the unified clock");
        assert!(second > first, "one clock, strictly increasing");
    }
}
