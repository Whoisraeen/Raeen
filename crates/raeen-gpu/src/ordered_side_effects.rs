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

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// Publish side effects the GPU worker just executed, in execution order.
pub fn publish(effects: impl IntoIterator<Item = OrderedGpuSideEffect>) {
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
    }
    PENDING_LEN.store(queue.len(), Ordering::Release);
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
        assert_eq!(drain(), effects.to_vec());
        assert!(drain().is_empty(), "a drain empties the queue");
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
