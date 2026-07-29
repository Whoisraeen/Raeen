//! Ordered GPU side effects — the in-stream event / EOP-interrupt / flip
//! hand-off (checklist item 5, steps 4–5).
//!
//! With `RAEEN_DEFER_GPU_SIDE_EFFECTS` OFF (the default) the HLE applies
//! these side effects **eagerly at submit time** from its decode pass, before
//! the GPU worker has executed anything — so an event or flip sequenced
//! behind an unexecuted `WAIT_REG_MEM` becomes guest-visible early. Under the
//! gate the eager duplicates are skipped and the GPU worker's command
//! processor records each side effect **in PM4 stream order** as it executes
//! the packet ([`kyty_graphics::run::SideEffect`]); the session publishes
//! them here, and the HLE drains this queue from its observation points
//! (submit, `sceKernelWaitEqueue`'s poll loop, the VideoOut status calls) and
//! applies them with kernel/VideoOut authority. Effects therefore become
//! visible no earlier than their in-stream execution, in execution order.
//!
//! The queue is process-global for the same reason `AgcGpuSession` is: the
//! worker thread has no kernel handle (`OrbisKernel` is not process-global,
//! and the HLE seam rule keeps kernel types out of this crate), while the
//! HLE has kernel authority on every call but no thread of its own.
//!
//! # Waking the observers
//!
//! A publish is only half of a hand-off: the guest thread that must observe it
//! is usually parked in `sceKernelWaitEqueue`, and nothing in this crate can
//! reach the kernel condition variable it is parked on. Before
//! [`SideEffectObserverWaker`] existed, the HLE compensated by shortening that
//! wait's internal park slice from 50 ms to 1 ms whenever the defer gate was
//! on — bounding observation latency by polling ~30 guest threads 50× more
//! often, for a queue that is empty almost every time.
//!
//! So the notify is *injected* instead. `raeen-hle` installs a waker at launch
//! ([`set_observer_waker`]) that holds a `Weak<OrbisKernel>`, and [`publish`]
//! calls it after queuing — the same shape `raeen-hle`'s `host_vblank` uses to
//! deliver events from a host thread, and for the same reason: the wake needs
//! only a `WaitSubsystem`, not an `HleContext`. With nothing installed (tests,
//! the Shell, a headless GPU harness) a publish is exactly what it always was.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// One guest-visible completion side effect executed in-stream by the GPU
/// worker, awaiting HLE delivery. Mirrors
/// [`kyty_graphics::run::SideEffect`] — no Kyty type crosses the HLE seam.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OrderedGpuSideEffect {
    /// Standard `IT_EVENT_WRITE`: signal kernel equeue events with this id.
    EventWrite {
        /// Event type (low 6 bits of the packet's first body dword).
        event_id: u32,
    },
    /// AGC `RELEASE_MEM` end-of-pipe interrupt (graphics-core equeue events).
    EopInterrupt {
        /// The packet's INT_CTXID dword (0 when absent).
        context_id: u32,
    },
    /// AGC flip packet: a VideoOut flip embedded in the command stream.
    Flip {
        /// VideoOut handle the title opened.
        video_out_handle: u32,
        /// Registered display-buffer slot to scan out.
        display_buffer_index: u32,
        /// `SceVideoOutFlipMode`.
        flip_mode: u32,
        /// The title's opaque completion argument.
        flip_arg: u64,
    },
}

impl From<kyty_graphics::run::SideEffect> for OrderedGpuSideEffect {
    fn from(effect: kyty_graphics::run::SideEffect) -> Self {
        match effect {
            kyty_graphics::run::SideEffect::EventWrite { event_id } => {
                Self::EventWrite { event_id }
            }
            kyty_graphics::run::SideEffect::EopInterrupt { context_id } => {
                Self::EopInterrupt { context_id }
            }
            kyty_graphics::run::SideEffect::Flip {
                video_out_handle,
                display_buffer_index,
                flip_mode,
                flip_arg,
            } => Self::Flip {
                video_out_handle,
                display_buffer_index,
                flip_mode,
                flip_arg,
            },
        }
    }
}

/// `RAEEN_DEFER_GPU_SIDE_EFFECTS=1` (transition gate, default OFF): stop
/// applying CP-executed side effects eagerly at submit and let the GPU
/// worker's in-stream execution order them instead. This is THE single
/// reader both crates consult (raeen-hle delegates here), so the eager path
/// and the worker publish can never disagree about the policy. Read per
/// call, like the `RAEEN_TRACE_*` gates, so tests can flip it per case.
#[must_use]
pub fn defer_gpu_side_effects() -> bool {
    std::env::var_os("RAEEN_DEFER_GPU_SIDE_EFFECTS").is_some()
}

/// Bound on undelivered effects. A title produces a handful of events and at
/// most a couple of flips per frame while the HLE drains on every submit, so
/// reaching this means delivery stopped entirely; dropping the newest (with a
/// rate-limited warn) keeps the oldest — still-ordered — prefix deliverable.
const MAX_PENDING: usize = 65_536;

static PENDING: Mutex<VecDeque<OrderedGpuSideEffect>> = Mutex::new(VecDeque::new());
/// Cheap emptiness probe so the HLE's poll-loop drain call sites do not take
/// the lock on the (overwhelmingly common) empty queue.
static PENDING_LEN: AtomicUsize = AtomicUsize::new(0);
static OVERFLOW_WARNED: AtomicUsize = AtomicUsize::new(0);

/// Are there effects queued for delivery?
///
/// One acquire load, no lock — cheap enough to sit inside a kernel wait's
/// readiness predicate, which is exactly where the HLE uses it: a waiter that
/// sees `true` leaves its park to run the drain. Pairs with [`publish`]'s
/// release store, so a waiter re-checking after a notify cannot miss the
/// effects that notify was announcing.
#[must_use]
pub fn has_pending() -> bool {
    PENDING_LEN.load(Ordering::Acquire) != 0
}

/// The notify seam a publish uses to reach whoever is parked waiting to
/// observe these effects.
///
/// Deliberately expressed as "wake the observers", not "wake equeue N": this
/// crate must not depend on `raeen-hle`/`raeen-kernel`, and the GPU worker has
/// no idea which queues the effects it just recorded will end up firing —
/// applying them is the HLE's job, and it happens *after* the wake. The
/// implementation lives in `raeen_hle::gpu_side_effect_waker`.
pub trait SideEffectObserverWaker: Send + Sync {
    /// Wake every thread that could be parked waiting to observe a published
    /// effect. Returns how many waits were notified — diagnostics only, and
    /// the deterministic seam the HLE's tests assert on.
    fn wake_side_effect_observers(&self) -> usize;
}

/// Installed once per launch by the process that owns the guest's kernel.
/// `None` (the default) makes a publish behave exactly as it did before this
/// seam existed.
static OBSERVER_WAKER: Mutex<Option<Arc<dyn SideEffectObserverWaker>>> = Mutex::new(None);
/// How many publishes have notified through [`OBSERVER_WAKER`] — see
/// [`observer_wake_count`].
static OBSERVER_WAKES: AtomicU64 = AtomicU64::new(0);

/// Install the process-wide observer waker, replacing any previous one.
///
/// Callers should hold the returned installation for the life of the guest
/// process and [`clear_observer_waker`] on teardown; `raeen-hle` wraps both in
/// an RAII guard. The waker must hold a `Weak` to anything it wakes — a
/// process-global `Arc<OrbisKernel>` would keep a finished guest's kernel
/// alive forever.
pub fn set_observer_waker(waker: Arc<dyn SideEffectObserverWaker>) {
    *OBSERVER_WAKER.lock().unwrap_or_else(|e| e.into_inner()) = Some(waker);
}

/// Uninstall the observer waker. Idempotent.
pub fn clear_observer_waker() {
    *OBSERVER_WAKER.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// How many times a publish has notified an installed observer waker.
///
/// Diagnostic, and the deterministic test seam for "the publish woke the
/// waiters rather than leaving them on their park slice" — the same role
/// `OrbisKernel::semaphore_wake_count` plays for semaphore producers. A
/// publish with an empty queue, or with no waker installed, does not count.
#[must_use]
pub fn observer_wake_count() -> u64 {
    OBSERVER_WAKES.load(Ordering::Relaxed)
}

/// Notify the installed waker, if any, that the queue just grew.
///
/// The registry lock is released before the wake: the woken threads take the
/// kernel's notification lock and then this module's `PENDING` lock, and a
/// producer holding a third lock across that hand-off is how a lock cycle gets
/// introduced later by someone who did not know it was load-bearing.
fn notify_observers() {
    let waker = {
        let guard = OBSERVER_WAKER.lock().unwrap_or_else(|e| e.into_inner());
        guard.clone()
    };
    let Some(waker) = waker else { return };
    OBSERVER_WAKES.fetch_add(1, Ordering::Relaxed);
    waker.wake_side_effect_observers();
}

/// Publish side effects the GPU worker just executed, in execution order, and
/// wake whoever is parked waiting to observe them.
pub fn publish(effects: impl IntoIterator<Item = OrderedGpuSideEffect>) {
    let mut queued = 0usize;
    {
        let mut queue = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        for effect in effects {
            if queue.len() >= MAX_PENDING {
                if OVERFLOW_WARNED.fetch_add(1, Ordering::Relaxed) == 0 {
                    tracing::warn!(
                        cap = MAX_PENDING,
                        "ordered GPU side-effect queue overflowed — HLE delivery has stopped; \
                         dropping the newest effects"
                    );
                }
                break;
            }
            queue.push_back(effect);
            queued += 1;
        }
        // Released before the notify below, and read by `has_pending` under
        // the kernel's notification lock: store-then-notify is what closes the
        // check-then-sleep race against a waiter that parked a moment ago.
        PENDING_LEN.store(queue.len(), Ordering::Release);
    }
    // Nothing queued is nothing to observe — an empty publish (or one that hit
    // the overflow cap on its first effect) must not cost a process-wide wake.
    if queued != 0 {
        notify_observers();
    }
}

/// Publish a CP walk's recorded side effects **iff the defer gate is ON**.
/// Gate OFF is the eager-duplicate policy: the HLE already applied every one
/// of these at submit time, so publishing them again would double-deliver
/// (a flip is not idempotent) — they are dropped here instead.
pub(crate) fn publish_cp_side_effects(effects: Vec<kyty_graphics::run::SideEffect>) {
    if effects.is_empty() || !defer_gpu_side_effects() {
        return;
    }
    publish(effects.into_iter().map(OrderedGpuSideEffect::from));
}

/// Drain every pending effect, in publish order. Cheap when empty (one
/// relaxed atomic load, no lock) — safe to call from poll loops.
#[must_use]
pub fn drain() -> Vec<OrderedGpuSideEffect> {
    if PENDING_LEN.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let mut queue = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    PENDING_LEN.store(0, Ordering::Release);
    queue.drain(..).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The queue and the gate are process state; serialize the tests that
    /// touch them (same pattern as raeen-hle's `SIDEFX_ENV_LOCK`).
    static QUEUE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvReset;
    impl Drop for EnvReset {
        fn drop(&mut self) {
            unsafe { std::env::remove_var("RAEEN_DEFER_GPU_SIDE_EFFECTS") };
        }
    }

    #[test]
    fn publish_then_drain_preserves_order_and_empties() {
        let _guard = QUEUE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = drain();
        let effects = [
            OrderedGpuSideEffect::EventWrite { event_id: 5 },
            OrderedGpuSideEffect::EopInterrupt { context_id: 7 },
            OrderedGpuSideEffect::Flip {
                video_out_handle: 1,
                display_buffer_index: 2,
                flip_mode: 0,
                flip_arg: 9,
            },
        ];
        publish(effects);
        assert!(has_pending(), "a publish leaves the queue observable");
        assert_eq!(drain(), effects.to_vec());
        assert!(drain().is_empty(), "a drain empties the queue");
        assert!(!has_pending(), "a drain clears the probe too");
    }

    /// A waker that records calls instead of waking anything, so "the publish
    /// notified the observers" is a plain assertion — no kernel, no threads, no
    /// timing. Counts its own calls as well as going through
    /// [`observer_wake_count`], because the two would silently diverge if the
    /// notify were ever moved off the counted path.
    #[derive(Default)]
    struct CountingWaker(AtomicU64);

    impl SideEffectObserverWaker for CountingWaker {
        fn wake_side_effect_observers(&self) -> usize {
            self.0.fetch_add(1, Ordering::Relaxed);
            3
        }
    }

    /// Uninstall on the way out so a failing assertion cannot leak a waker into
    /// the tests that run after it.
    struct WakerReset;
    impl Drop for WakerReset {
        fn drop(&mut self) {
            clear_observer_waker();
        }
    }

    /// The seam that replaced the 1 ms poll: a publish notifies the installed
    /// waker exactly once, and an empty one does not notify at all.
    #[test]
    fn a_publish_notifies_the_installed_observer_waker_once() {
        let _guard = QUEUE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _reset = WakerReset;
        let _ = drain();
        let waker = Arc::new(CountingWaker::default());
        set_observer_waker(waker.clone());

        let before = observer_wake_count();
        publish([OrderedGpuSideEffect::EventWrite { event_id: 5 }]);
        assert_eq!(waker.0.load(Ordering::Relaxed), 1);
        assert_eq!(observer_wake_count(), before + 1);

        // Three effects are one hand-off, so one wake — not one per effect.
        publish([
            OrderedGpuSideEffect::EventWrite { event_id: 6 },
            OrderedGpuSideEffect::EopInterrupt { context_id: 1 },
            OrderedGpuSideEffect::EventWrite { event_id: 7 },
        ]);
        assert_eq!(waker.0.load(Ordering::Relaxed), 2);
        assert_eq!(observer_wake_count(), before + 2);

        // Nothing queued is nothing to observe: a process-wide wake per empty
        // publish would be the polling cost back under a different name.
        publish([]);
        assert_eq!(
            waker.0.load(Ordering::Relaxed),
            2,
            "an empty publish must not wake anybody"
        );
        assert_eq!(observer_wake_count(), before + 2);

        let _ = drain();
    }

    /// Uninstalling really uninstalls, and a publish with no waker installed —
    /// every test binary, the Shell, any headless GPU harness — is exactly what
    /// it was before the seam existed.
    #[test]
    fn a_publish_with_no_waker_installed_is_unchanged() {
        let _guard = QUEUE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _reset = WakerReset;
        let _ = drain();
        let waker = Arc::new(CountingWaker::default());
        set_observer_waker(waker.clone());
        clear_observer_waker();

        let before = observer_wake_count();
        publish([OrderedGpuSideEffect::EopInterrupt { context_id: 9 }]);

        assert_eq!(waker.0.load(Ordering::Relaxed), 0);
        assert_eq!(observer_wake_count(), before);
        assert_eq!(
            drain(),
            vec![OrderedGpuSideEffect::EopInterrupt { context_id: 9 }],
            "the effect is still queued for the next observation point"
        );
    }

    /// Gate OFF drops the worker's duplicate, so there is nothing to observe
    /// and nobody is woken. The wake must follow what was actually queued, not
    /// the fact that a publish was attempted.
    #[test]
    fn a_dropped_cp_publish_wakes_nobody() {
        let _guard = QUEUE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _reset = WakerReset;
        let _env = EnvReset;
        let _ = drain();
        let waker = Arc::new(CountingWaker::default());
        set_observer_waker(waker.clone());
        let recorded = vec![kyty_graphics::run::SideEffect::EventWrite { event_id: 0x2A }];

        unsafe { std::env::remove_var("RAEEN_DEFER_GPU_SIDE_EFFECTS") };
        publish_cp_side_effects(recorded.clone());
        assert_eq!(waker.0.load(Ordering::Relaxed), 0);
        assert!(!has_pending());

        unsafe { std::env::set_var("RAEEN_DEFER_GPU_SIDE_EFFECTS", "1") };
        publish_cp_side_effects(recorded);
        assert_eq!(waker.0.load(Ordering::Relaxed), 1);
        assert!(has_pending());

        let _ = drain();
    }

    #[test]
    fn cp_side_effects_are_published_only_under_the_defer_gate() {
        let _guard = QUEUE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _reset = EnvReset;
        let _ = drain();
        let recorded = vec![kyty_graphics::run::SideEffect::Flip {
            video_out_handle: 1,
            display_buffer_index: 0,
            flip_mode: 0,
            flip_arg: 4,
        }];

        // Gate OFF: the eager submit path already applied the effect — the
        // worker's duplicate must be dropped, not double-delivered.
        unsafe { std::env::remove_var("RAEEN_DEFER_GPU_SIDE_EFFECTS") };
        publish_cp_side_effects(recorded.clone());
        assert!(drain().is_empty(), "gate off drops the worker duplicate");

        // Gate ON: the worker's in-stream execution is the only source.
        unsafe { std::env::set_var("RAEEN_DEFER_GPU_SIDE_EFFECTS", "1") };
        publish_cp_side_effects(recorded);
        assert_eq!(
            drain(),
            vec![OrderedGpuSideEffect::Flip {
                video_out_handle: 1,
                display_buffer_index: 0,
                flip_mode: 0,
                flip_arg: 4,
            }]
        );
    }
}
