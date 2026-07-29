//! A free-running **host** vblank source: display refreshes that happen because
//! time passed, not because the guest asked.
//!
//! # The defect this closes
//!
//! Raeen advanced its vblank sequence from exactly two places, and both were
//! guest calls: `sceVideoOutSubmitFlip` and `sceVideoOutWaitVblank`
//! (`libsce_video_out.rs`). That works for a *polling* frame loop. It deadlocks
//! an **event-driven** one. A guest thread that
//!
//! 1. opens video out,
//! 2. registers a vblank event on an equeue (`sceVideoOutAddVblankEvent` — we
//!    return `SCE_OK` and store the registration), then
//! 3. blocks in `sceKernelWaitEqueue` for the first vblank **before** it submits
//!    its first flip,
//!
//! waits forever: the only two things that could fire that event are the two
//! calls it is now blocked from making. Zero CPU, zero output, zero errors.
//! Same **ack-but-never-deliver** class as `sceKernelRaiseException` and the
//! `HLE_ERROR` sentinel leak. Full diagnosis:
//! `docs/silent-zero-frame-cluster.md` sections 3 and 5;
//! design notes and the A/B procedure: `docs/host-vblank-source.md`.
//!
//! KytyPS5 (`reference/kytyps5`, MIT © InoriRus / Nmzik) has no such hole: its
//! window loop ticks vblank every displayed host frame regardless of the guest —
//! `GameShowWindow` calls `VideoOutBeginVblank()` → `VideoOutFlipWindow(0)` →
//! `VideoOutEndVblank()` (`src/graphics/presentation/window/window.cpp:350-354`),
//! and those advance the pre-vblank / vblank counters and trigger the VideoOut
//! events of every opened handle (`videoOut.cpp:649-686`), paced against
//! `Config::GetVblankFrequency()` (`videoOut.cpp:402`). This module is that
//! structure, ported to Raeen's equeue model.
//!
//! # Why this needed a seam rather than a patch
//!
//! [`HleContext`](crate::HleContext) is a borrowed struct of `&dyn` references
//! with a lifetime, so no host thread can hold one — which is why this was left
//! unimplemented when it was first diagnosed. But a vblank delivery only ever
//! touches two things: the `kernel_equeue_events` map on
//! [`raeen_kernel::OrbisKernel`], and a
//! [`WaitSubsystem::wake`](raeen_core::subsystems::WaitSubsystem::wake). It
//! never reads guest memory, allocates, or submits to the GPU. `OrbisKernel`
//! already implements `WaitSubsystem`, which is `Send + Sync`, and the kernel is
//! already `Arc`-shared by the runtime. So the seam is just those two arguments
//! split out of the context:
//! [`wake_equeue_via`](crate::kernel_equeue::wake_equeue_via) and
//! [`trigger_vblank_events_via`](crate::libsce_video_out::trigger_vblank_events_via).
//! A background thread holding `Weak<OrbisKernel>` upgrades and passes
//! `&*kernel` for both. No `HleContext`, no new locking, no change to any
//! guest-visible ABI.
//!
//! # One owner
//!
//! While this source is running it is the **sole** advancer of
//! `video_out_vblank_count`, and the two guest-driven advances stand down (they
//! check [`owns_sequence`]). Both sources advancing would let a title's frame
//! sequence run ahead of the display clock it is measuring. The flip *events* a
//! flip fires are unaffected — a flip really did complete; only the "…therefore
//! a refresh happened" inference is dropped.
//!
//! # Default off
//!
//! Titles render today (Minecraft ~13,400 frames; ASTRO.BOT holds `rendering`),
//! and an unverified pacing change must not be able to regress them. Disabled —
//! the default — the whole feature costs one relaxed [`AtomicBool`] load and a
//! not-taken branch at each of the two guest advance sites, no thread, and no
//! behavior change of any kind. Enable with `RAEEN_HOST_VBLANK` (see
//! [`configured_host_vblank_period`]).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use raeen_kernel::OrbisKernel;

/// Set for exactly as long as a [`HostVblankSource`] thread is alive. Read by
/// the guest-driven advance sites in `libsce_video_out.rs`.
static OWNS_SEQUENCE: AtomicBool = AtomicBool::new(false);

/// Is a host vblank source the owner of the process vblank sequence?
///
/// `false` (the default, and whenever the source could not start) means the
/// legacy guest-driven advances in `sceVideoOutSubmitFlip` /
/// `sceVideoOutWaitVblank` stay in charge — so a build with the feature
/// disabled behaves exactly as it did before this module existed.
#[must_use]
pub fn owns_sequence() -> bool {
    OWNS_SEQUENCE.load(Ordering::Relaxed)
}

/// Environment tokens that mean "off" even though they are non-empty.
const OFF_TOKENS: [&str; 4] = ["0", "off", "false", "no"];

/// Resolve the host vblank source's period from the environment.
///
/// `host` is `RAEEN_HOST_VBLANK`, `hz` is `RAEEN_VBLANK_HZ`:
///
/// | `RAEEN_HOST_VBLANK` | result |
/// |---|---|
/// | unset | `None` — disabled (the default) |
/// | `0`, `off`, `false`, `no` | `None` — explicitly disabled |
/// | a rate in 24..=480 | that rate, overriding `RAEEN_VBLANK_HZ` |
/// | any other value (`1`, `on`, empty, …) | the configured display rate |
///
/// The last row inherits `RAEEN_VBLANK_HZ` through the *existing*
/// [`vblank_period`](crate::libsce_video_out::vblank_period) — this deliberately
/// does not introduce a second refresh setting. That also means
/// `RAEEN_VBLANK_HZ=0`, the explicit unpaced benchmark mode
/// (`cargo xtask compat run --profile max-fps`), yields `None` here: with no
/// display clock there is no host source to run, and the guest-driven advances
/// correctly keep ownership. Requesting a rate directly (`RAEEN_HOST_VBLANK=60`)
/// is how a benchmark run gets both an unpaced guest and a real vblank clock.
#[must_use]
pub fn configured_host_vblank_period(host: Option<&str>, hz: Option<&str>) -> Option<Duration> {
    let host = host?;
    if OFF_TOKENS.contains(&host.trim().to_ascii_lowercase().as_str()) {
        return None;
    }
    match host.trim().parse::<u64>() {
        Ok(rate @ 24..=480) => Some(Duration::from_nanos(1_000_000_000 / rate)),
        _ => crate::libsce_video_out::configured_vblank_period(hz),
    }
}

/// A running host vblank source: one named background thread plus its stop
/// flag. Dropping it stops and joins the thread, so the source cannot outlive
/// the scope that owns it.
pub struct HostVblankSource {
    /// Cleared to stop; the thread checks it on both sides of every wait, so it
    /// can never deliver a refresh after teardown began.
    stop: Arc<AtomicBool>,
    /// `None` once joined, making [`HostVblankSource::stop`] idempotent.
    thread: Option<std::thread::JoinHandle<()>>,
    /// The period the thread is pacing at, for logging and tests.
    period: Duration,
}

impl HostVblankSource {
    /// Start the source from the environment, or return `None` when it is
    /// disabled (the default), has no display clock, or the thread cannot be
    /// spawned.
    ///
    /// Holds a [`Weak`], never an [`Arc`]: the source must not keep a guest
    /// process's kernel alive, and a failed upgrade is the thread's cue that the
    /// process is gone — which makes "wake into a torn-down kernel"
    /// unrepresentable rather than merely unlikely.
    #[must_use]
    pub fn start_from_env(kernel: &Arc<OrbisKernel>) -> Option<Self> {
        let period = configured_host_vblank_period(
            std::env::var("RAEEN_HOST_VBLANK").ok().as_deref(),
            std::env::var("RAEEN_VBLANK_HZ").ok().as_deref(),
        )?;
        Self::start_with_period(kernel, period)
    }

    /// Start at an explicit period. Separate from [`Self::start_from_env`] so
    /// tests drive the lifecycle without touching process environment.
    #[must_use]
    pub fn start_with_period(kernel: &Arc<OrbisKernel>, period: Duration) -> Option<Self> {
        if period.is_zero() {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let weak = Arc::downgrade(kernel);
        let thread_stop = Arc::clone(&stop);
        // Claim ownership BEFORE the thread exists. Claiming it from inside the
        // thread would leave a window in which the guest-driven sites still
        // advance while the host source is already coming up — the exact
        // double-count this flag prevents.
        OWNS_SEQUENCE.store(true, Ordering::Relaxed);
        let thread = std::thread::Builder::new()
            .name("raeen-host-vblank".to_owned())
            .spawn(move || run(&weak, period, &thread_stop))
            .inspect_err(|error| {
                OWNS_SEQUENCE.store(false, Ordering::Relaxed);
                tracing::warn!(%error, "host vblank source could not start a thread");
            })
            .ok()?;
        tracing::info!(
            period_us = period.as_micros(),
            hz = 1_000_000_000u64 / period.as_nanos().max(1) as u64,
            "host vblank source running — it now owns the vblank sequence \
             (RAEEN_HOST_VBLANK)"
        );
        Some(Self {
            stop,
            thread: Some(thread),
            period,
        })
    }

    /// The period this source paces at.
    #[must_use]
    pub fn period(&self) -> Duration {
        self.period
    }

    /// Stop delivering and join the thread. Idempotent.
    ///
    /// Ownership of the sequence is released **first**, so the guest-driven
    /// advances resume before the last host refresh could possibly land; then
    /// the join is a real handshake, not a timed wait, so no thread outlives
    /// this call. Worst-case latency is one period (≤ 41 ms at the slowest
    /// selectable rate): the thread is parked on an absolute display edge and
    /// checks the stop flag the moment it wakes.
    pub fn stop(&mut self) {
        OWNS_SEQUENCE.store(false, Ordering::Relaxed);
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            // A panicked ticker is a bug worth a line, not a reason to poison
            // the shutdown path of a process that is already exiting.
            if thread.join().is_err() {
                tracing::warn!("host vblank thread panicked");
            }
        }
    }
}

impl Drop for HostVblankSource {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The ticker loop. Paced on the **shared** absolute edge grid
/// (`vblank_epoch() + n·period`), the same one `sceVideoOutWaitVblank` waits
/// on — so a guest parked in that call resumes on the very edge this thread
/// just delivered, rather than on a second clock beating against it.
fn run(kernel: &Weak<OrbisKernel>, period: Duration, stop: &AtomicBool) {
    let mut delivered = 0u64;
    while !stop.load(Ordering::Relaxed) {
        crate::libsce_video_out::wait_next_host_vblank_edge(period);
        // Re-check after the wait: teardown may have begun while parked, and a
        // wake delivered after `stop()` returned is exactly what must not
        // happen.
        if stop.load(Ordering::Relaxed) {
            break;
        }
        // A dead kernel means the guest process is gone; there is nothing to
        // deliver to and nothing to keep alive.
        let Some(kernel) = kernel.upgrade() else {
            break;
        };
        crate::libsce_video_out::host_vblank_refresh(&kernel, &*kernel);
        delivered += 1;
    }
    tracing::debug!(delivered, "host vblank source stopped");
}

/// [`OWNS_SEQUENCE`] is process-global, matching the process-global vblank clock
/// it governs. Serialize every test that reads or writes it so a parallel
/// `cargo test` cannot flip it under another test's assertions. Module level, not
/// inside `mod tests`: `libsce_video_out`'s own tests need it to check that the
/// guest-driven advance sites stand down.
#[cfg(test)]
static OWNERSHIP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Hold the vblank-sequence ownership flag for the body of a test and always
/// release it, so a failing assertion cannot leak a `true` flag into later tests.
#[cfg(test)]
pub(crate) struct OwnershipGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

#[cfg(test)]
impl OwnershipGuard {
    /// Pretend a host source is running.
    pub(crate) fn claimed() -> Self {
        let guard = OWNERSHIP_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        OWNS_SEQUENCE.store(true, Ordering::Relaxed);
        Self(guard)
    }

    /// Pin the default (no host source) against a concurrent test.
    pub(crate) fn released() -> Self {
        let guard = OWNERSHIP_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        OWNS_SEQUENCE.store(false, Ordering::Relaxed);
        Self(guard)
    }
}

#[cfg(test)]
impl Drop for OwnershipGuard {
    fn drop(&mut self) {
        OWNS_SEQUENCE.store(false, Ordering::Relaxed);
    }
}

/// A [`WaitSubsystem`](raeen_core::subsystems::WaitSubsystem) that records wakes
/// instead of performing them, so "the tick woke the right queue for the right
/// reason" is a plain assertion — no threads, no timing, no sleeping. This is
/// the whole point of the seam being `&dyn WaitSubsystem`: the wake is
/// observable without a second thread to synchronize with.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct RecordingWaker {
    wakes: std::sync::Mutex<
        Vec<(
            raeen_core::subsystems::WaitKey,
            raeen_core::subsystems::WakeReason,
        )>,
    >,
}

#[cfg(test)]
impl RecordingWaker {
    pub(crate) fn wakes(
        &self,
    ) -> Vec<(
        raeen_core::subsystems::WaitKey,
        raeen_core::subsystems::WakeReason,
    )> {
        self.wakes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[cfg(test)]
impl raeen_core::subsystems::WaitSubsystem for RecordingWaker {
    fn wait_until(
        &self,
        _key: raeen_core::subsystems::WaitKey,
        _timeout: Duration,
        _terminating: &dyn Fn() -> bool,
        _ready: &mut dyn FnMut() -> bool,
    ) -> raeen_core::subsystems::WaitOutcome {
        unreachable!("the host vblank source never waits through the subsystem")
    }

    fn wake(
        &self,
        key: raeen_core::subsystems::WaitKey,
        reason: raeen_core::subsystems::WakeReason,
    ) {
        self.wakes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push((key, reason));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raeen_core::subsystems::WakeReason;

    #[test]
    fn env_gate_is_off_by_default_and_inherits_the_display_rate() {
        // Unset and the explicit off tokens: no source, so no pacing change can
        // reach a title that renders today.
        assert_eq!(configured_host_vblank_period(None, None), None);
        for off in ["0", "off", "false", "no", "OFF", " off "] {
            assert_eq!(
                configured_host_vblank_period(Some(off), None),
                None,
                "{off:?} must disable the source"
            );
        }
        // A rate overrides RAEEN_VBLANK_HZ.
        assert_eq!(
            configured_host_vblank_period(Some("120"), Some("60")),
            Some(Duration::from_nanos(1_000_000_000 / 120))
        );
        // Enable tokens inherit the configured display rate rather than adding
        // a second refresh setting.
        for on in ["1", "on", "true", "", "yes"] {
            assert_eq!(
                configured_host_vblank_period(Some(on), Some("120")),
                Some(Duration::from_nanos(1_000_000_000 / 120)),
                "{on:?} must inherit RAEEN_VBLANK_HZ"
            );
            assert_eq!(
                configured_host_vblank_period(Some(on), None),
                Some(Duration::from_nanos(1_000_000_000 / 60)),
                "{on:?} must default to 60 Hz"
            );
        }
        // Unpaced benchmark mode has no display clock, so there is no host
        // source to run and the guest-driven advances keep ownership.
        assert_eq!(configured_host_vblank_period(Some("1"), Some("0")), None);
        // ...unless the rate is requested directly.
        assert_eq!(
            configured_host_vblank_period(Some("60"), Some("0")),
            Some(Duration::from_nanos(1_000_000_000 / 60))
        );
    }

    /// The headline: a refresh delivered with **no guest call at all** advances
    /// the sequence, triggers the registered event, and wakes its queue.
    #[test]
    fn a_host_refresh_advances_the_sequence_and_wakes_a_registered_waiter() {
        let _guard = OwnershipGuard::claimed();
        let kernel = OrbisKernel::new();
        let eq = kernel.create_equeue(0);
        // Register a vblank event exactly as `sceVideoOutAddVblankEvent` does,
        // then never call another guest function.
        kernel.kernel_equeue_events.insert(
            (eq, 0x40),
            raeen_kernel::EqueueUserEvent {
                udata: 0xBEEF,
                filter: -13,
                ..Default::default()
            },
        );
        let waker = RecordingWaker::default();

        let sequence = crate::libsce_video_out::host_vblank_refresh(&kernel, &waker);

        assert_eq!(sequence, 1, "the first host refresh is sequence 1");
        assert_eq!(kernel.video_out_vblank_count.load(Ordering::Relaxed), 1);
        let event = kernel.kernel_equeue_events.get(&(eq, 0x40)).unwrap();
        assert!(event.triggered, "the vblank event must be delivered");
        assert_eq!(event.udata, 0xBEEF, "the registration's udata survives");
        assert_eq!(
            event.data as u64,
            0x40 | (1 << 16),
            "data carries ident | sequence << 16, as every trigger site encodes it"
        );
        drop(event);

        // And the queue was woken, so a thread parked in `sceKernelWaitEqueue`
        // observes it now rather than on its next 50 ms slice.
        let wakes = waker.wakes();
        assert_eq!(
            wakes.len(),
            1,
            "exactly one wake, for the one queue: {wakes:?}"
        );
        assert_eq!(wakes[0].0.class, "kernel-equeue");
        assert_eq!(wakes[0].0.object, eq);
        assert_eq!(
            wakes[0].0.guest_thread, 0,
            "no guest thread caused this wake"
        );
        assert_eq!(wakes[0].1, WakeReason::Signal);
    }

    /// One refresh reaches **every** registered queue and both event classes,
    /// mirroring KytyPS5's per-opened-handle loop.
    #[test]
    fn a_host_refresh_reaches_every_registered_queue_and_both_classes() {
        let _guard = OwnershipGuard::claimed();
        let kernel = OrbisKernel::new();
        let first = kernel.create_equeue(0);
        let second = kernel.create_equeue(0);
        let registration = raeen_kernel::EqueueUserEvent {
            filter: -13,
            ..Default::default()
        };
        kernel
            .kernel_equeue_events
            .insert((first, 0x40), registration);
        // Pre-vblank-start on the second queue, plus an unrelated user event
        // that must be left alone.
        kernel
            .kernel_equeue_events
            .insert((second, 0x41), registration);
        kernel
            .kernel_equeue_events
            .insert((second, 0x7), raeen_kernel::EqueueUserEvent::default());
        let waker = RecordingWaker::default();

        crate::libsce_video_out::host_vblank_refresh(&kernel, &waker);

        assert!(
            kernel
                .kernel_equeue_events
                .get(&(first, 0x40))
                .unwrap()
                .triggered
        );
        assert!(
            kernel
                .kernel_equeue_events
                .get(&(second, 0x41))
                .unwrap()
                .triggered
        );
        assert!(
            !kernel
                .kernel_equeue_events
                .get(&(second, 0x7))
                .unwrap()
                .triggered,
            "a non-VideoOut user event must not be triggered by a display refresh"
        );
        let mut woken: Vec<u64> = waker.wakes().iter().map(|(key, _)| key.object).collect();
        woken.sort_unstable();
        let mut expected = [first, second];
        expected.sort_unstable();
        assert_eq!(woken, expected, "both queues woken, each exactly once");
    }

    /// A refresh with nothing registered is pure counter traffic: no wake, so an
    /// enabled source costs a title that never registers a vblank event nothing
    /// but a DashMap scan per period.
    #[test]
    fn a_host_refresh_with_no_registration_wakes_nobody() {
        let _guard = OwnershipGuard::claimed();
        let kernel = OrbisKernel::new();
        let waker = RecordingWaker::default();
        assert_eq!(
            crate::libsce_video_out::host_vblank_refresh(&kernel, &waker),
            1
        );
        assert!(waker.wakes().is_empty());
    }

    /// Teardown: ownership is released, the thread joins (not times out), and
    /// nothing is left running. The `Weak` also means a dropped kernel ends the
    /// thread on its own.
    #[test]
    fn stopping_releases_ownership_and_joins_the_thread() {
        let _guard = OwnershipGuard::released();
        let kernel = Arc::new(OrbisKernel::new());
        // The fastest selectable rate, so the loop is certain to be inside its
        // wait rather than mid-spawn when the stop lands.
        let mut source =
            HostVblankSource::start_with_period(&kernel, Duration::from_nanos(1_000_000_000 / 480))
                .expect("a positive period starts a source");
        assert!(
            owns_sequence(),
            "a running source owns the vblank sequence immediately, \
             before the thread's first tick"
        );
        assert_eq!(source.period(), Duration::from_nanos(1_000_000_000 / 480));

        source.stop();

        assert!(
            !owns_sequence(),
            "stopping hands the sequence back to the guest-driven sites"
        );
        source.stop(); // idempotent: a second stop must not panic on a taken handle.
        // Dropping is also `stop()`; it must not double-join either.
        drop(source);
        assert!(!owns_sequence());
        // Nothing kept the kernel alive but this binding.
        assert_eq!(Arc::strong_count(&kernel), 1, "the source held only a Weak");
    }

    /// End to end through the real environment read: with `RAEEN_HOST_VBLANK`
    /// unset — the state every existing launch is in — no source starts and the
    /// guest-driven advances keep ownership. Reads the environment rather than
    /// mutating it, so it cannot race a parallel test (and needs no `unsafe`
    /// `set_var` on edition 2024).
    #[test]
    fn start_from_env_is_a_no_op_unless_the_flag_is_set() {
        let _guard = OwnershipGuard::released();
        if std::env::var_os("RAEEN_HOST_VBLANK").is_some() {
            // Someone is deliberately A/B-ing; this assertion does not apply.
            return;
        }
        let kernel = Arc::new(OrbisKernel::new());
        assert!(HostVblankSource::start_from_env(&kernel).is_none());
        assert!(
            !owns_sequence(),
            "default off means the guest still owns it"
        );
    }

    #[test]
    fn a_zero_period_cannot_start_a_source() {
        let _guard = OwnershipGuard::released();
        let kernel = Arc::new(OrbisKernel::new());
        assert!(HostVblankSource::start_with_period(&kernel, Duration::ZERO).is_none());
        assert!(
            !owns_sequence(),
            "a source that never started must not claim ownership"
        );
    }
}
