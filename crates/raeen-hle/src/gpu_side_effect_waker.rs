//! The GPU worker's notify into the kernel: how a published ordered side
//! effect wakes the guest threads parked waiting to observe it.
//!
//! # The defect this closes
//!
//! Under `RAEEN_DEFER_GPU_SIDE_EFFECTS` the GPU worker records completion side
//! effects (events, EOP interrupts, flips) in PM4 execution order and publishes
//! them to [`raeen_gpu::ordered_side_effects`]; the HLE drains that queue at the
//! guest's observation points, one of which is the `sceKernelWaitEqueue` poll
//! loop. But the worker is a host thread with no kernel handle, so a publish
//! could not wake anybody: a guest thread already parked in that wait would sit
//! there until its internal park slice expired, whatever the worker had just
//! made deliverable.
//!
//! The stopgap was to shorten that slice from 50 ms to 1 ms whenever the defer
//! gate was on. It bounded the latency by polling — ~30 guest threads waking
//! 50× more often, essentially always to find the queue empty, on a title
//! already fighting for CPU. This module makes the publish notify instead, and
//! `kernel_equeue`'s slice went back to one value for every wait.
//!
//! # Why this needed a seam rather than a call
//!
//! `raeen-gpu` must not depend on `raeen-hle` or `raeen-kernel`, so the worker
//! cannot call the wake. Nor could the wake be handed over as an
//! [`HleContext`](crate::HleContext): that is a borrowed struct of `&dyn`
//! references with a lifetime, which no host thread can hold. Same two walls
//! [`crate::host_vblank`] hit, and the same way through — a wake needs only a
//! [`WaitSubsystem`](raeen_core::subsystems::WaitSubsystem), which `OrbisKernel`
//! implements and which is `Send + Sync`. So `raeen-gpu` declares the trait
//! ([`raeen_gpu::ordered_side_effects::SideEffectObserverWaker`]), this module
//! implements it over a `Weak<OrbisKernel>`, and the launching process installs
//! one for the life of the guest.
//!
//! # Why it wakes every queue
//!
//! The worker knows what it *executed*, not what that will *fire*: an effect is
//! translated into equeue triggers later, by whichever guest thread drains it
//! (`libsce_agc::apply_ordered_gpu_side_effects`). So there is no queue to
//! address at publish time and the wake goes to all of them
//! ([`crate::kernel_equeue::wake_all_equeues_via`]). Each woken waiter
//! re-evaluates its readiness predicate, which reports the pending queue, drains
//! it, and parks again if the effects were not for it. A title has a handful of
//! queues and a publish happens per submission, so this is orders of magnitude
//! below the 1 ms poll it replaces.

use std::sync::{Arc, Weak};

use raeen_core::subsystems::WakeReason;
use raeen_kernel::OrbisKernel;

/// Diagnostics `guest_thread` label for a wake that no guest thread caused —
/// the GPU worker is a host thread, exactly like the host vblank source.
const GPU_WORKER_GUEST_THREAD: u64 = 0;

/// Wakes every live event queue of one guest process when the GPU worker
/// publishes ordered side effects.
///
/// Holds a [`Weak`], never an [`Arc`]: this is installed in a process-global
/// slot, and a strong reference there would keep a finished guest's kernel
/// alive for the rest of the host process. A failed upgrade means the process
/// is gone and the wake is simply dropped.
pub struct EqueueSideEffectWaker {
    kernel: Weak<OrbisKernel>,
}

impl EqueueSideEffectWaker {
    /// A waker for one guest process's kernel, not yet installed.
    ///
    /// [`SideEffectWakerGuard::install`] is what production uses; this exists so
    /// a test can invoke `wake_side_effect_observers` — the exact call
    /// [`publish`] makes — without routing through the process-global waker slot
    /// or effect queue, both of which a parallel test can legitimately disturb.
    ///
    /// [`publish`]: raeen_gpu::ordered_side_effects::publish
    #[must_use]
    pub fn for_kernel(kernel: &Arc<OrbisKernel>) -> Self {
        Self {
            kernel: Arc::downgrade(kernel),
        }
    }
}

impl raeen_gpu::ordered_side_effects::SideEffectObserverWaker for EqueueSideEffectWaker {
    fn wake_side_effect_observers(&self) -> usize {
        let Some(kernel) = self.kernel.upgrade() else {
            return 0;
        };
        crate::kernel_equeue::wake_all_equeues_via(
            &kernel,
            &*kernel,
            GPU_WORKER_GUEST_THREAD,
            // The same reason the eager submit path signals with: what the
            // waiter is being told is that GPU work reached a point the guest
            // can observe.
            WakeReason::SubmissionComplete,
        )
    }
}

/// An installed [`EqueueSideEffectWaker`], uninstalled on drop.
///
/// Bind it to a local in the scope that owns the guest's `Arc<OrbisKernel>`, as
/// [`crate::host_vblank::HostVblankSource`] is bound, so the process-global slot
/// cannot outlive the run that filled it and a second launch in the same
/// process cannot inherit the first one's dead `Weak`.
#[must_use = "dropping the guard immediately uninstalls the waker"]
pub struct SideEffectWakerGuard(());

impl SideEffectWakerGuard {
    /// Install a waker for `kernel`, replacing any previous one.
    pub fn install(kernel: &Arc<OrbisKernel>) -> Self {
        raeen_gpu::ordered_side_effects::set_observer_waker(Arc::new(
            EqueueSideEffectWaker::for_kernel(kernel),
        ));
        tracing::debug!(
            "GPU ordered side-effect publishes now wake equeue waiters directly \
             (no polling slice)"
        );
        Self(())
    }
}

impl Drop for SideEffectWakerGuard {
    fn drop(&mut self) {
        raeen_gpu::ordered_side_effects::clear_observer_waker();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raeen_gpu::ordered_side_effects::{
        OrderedGpuSideEffect, SideEffectObserverWaker, observer_wake_count, publish,
    };

    /// The waker slot and the effect queue are process-global; serialize with
    /// every other test that touches them.
    fn sidefx_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::SIDEFX_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// The wake reaches every live queue, labelled as a host-thread wake, and
    /// with the reason the eager submit path uses — asserted through the
    /// `&dyn WaitSubsystem` seam so no thread or timing is involved.
    #[test]
    fn the_waker_wakes_every_live_queue_as_a_host_thread() {
        let kernel = OrbisKernel::new();
        let queues = [
            kernel.create_equeue(0),
            kernel.create_equeue(0),
            kernel.create_equeue(0),
        ];
        let recorder = crate::host_vblank::RecordingWaker::default();

        let woken = crate::kernel_equeue::wake_all_equeues_via(
            &kernel,
            &recorder,
            GPU_WORKER_GUEST_THREAD,
            WakeReason::SubmissionComplete,
        );

        assert_eq!(woken, queues.len());
        let wakes = recorder.wakes();
        assert_eq!(
            wakes.len(),
            queues.len(),
            "one wake per live queue: {wakes:?}"
        );
        let mut woken_objects: Vec<u64> = wakes.iter().map(|(key, _)| key.object).collect();
        woken_objects.sort_unstable();
        let mut expected = queues;
        expected.sort_unstable();
        assert_eq!(woken_objects, expected);
        for (key, reason) in &wakes {
            assert_eq!(key.class, "kernel-equeue");
            assert_eq!(
                key.guest_thread, GPU_WORKER_GUEST_THREAD,
                "no guest thread caused a GPU worker publish"
            );
            assert_eq!(*reason, WakeReason::SubmissionComplete);
        }
    }

    /// With no queue open there is nobody to wake, and that is not an error —
    /// a publish before the title creates its first equeue must not panic or
    /// fabricate a wake.
    #[test]
    fn a_process_with_no_queues_is_woken_zero_times() {
        let kernel = OrbisKernel::new();
        let recorder = crate::host_vblank::RecordingWaker::default();
        assert_eq!(
            crate::kernel_equeue::wake_all_equeues_via(
                &kernel,
                &recorder,
                GPU_WORKER_GUEST_THREAD,
                WakeReason::SubmissionComplete,
            ),
            0
        );
        assert!(recorder.wakes().is_empty());
    }

    /// End to end through the installed seam: a `raeen_gpu` publish — the call
    /// the GPU worker makes, with no knowledge of any kernel — reaches this
    /// crate and wakes the process's queues.
    #[test]
    fn a_publish_reaches_the_installed_waker() {
        let _guard = sidefx_lock();
        let _ = raeen_gpu::ordered_side_effects::drain();
        let kernel = Arc::new(OrbisKernel::new());
        kernel.create_equeue(0);
        kernel.create_equeue(0);
        let _installed = SideEffectWakerGuard::install(&kernel);

        let before = observer_wake_count();
        publish([OrderedGpuSideEffect::EventWrite { event_id: 0x2A }]);

        assert_eq!(
            observer_wake_count(),
            before + 1,
            "a non-empty publish must notify exactly once"
        );
        // The effect itself is deliberately not asserted on here: the queue is
        // process-global and any parallel test in this binary that reaches a
        // drain point may legitimately consume it. What this test owns is that
        // the publish reached the installed waker.
        let _ = raeen_gpu::ordered_side_effects::drain();
    }

    /// The guard is the whole lifetime: dropping it uninstalls, so a finished
    /// run cannot be woken and its kernel is not kept alive by the global slot.
    #[test]
    fn the_guard_uninstalls_and_holds_only_a_weak() {
        let _guard = sidefx_lock();
        let _ = raeen_gpu::ordered_side_effects::drain();
        let kernel = Arc::new(OrbisKernel::new());
        kernel.create_equeue(0);

        {
            let _installed = SideEffectWakerGuard::install(&kernel);
            assert_eq!(
                Arc::strong_count(&kernel),
                1,
                "the installed waker must hold only a Weak"
            );
        }

        let before = observer_wake_count();
        publish([OrderedGpuSideEffect::EventWrite { event_id: 7 }]);
        assert_eq!(
            observer_wake_count(),
            before,
            "an uninstalled waker must not be notified"
        );
        let _ = raeen_gpu::ordered_side_effects::drain();
    }

    /// A wake after the guest process is gone is a no-op, not an upgrade of a
    /// dangling pointer — the reason the waker holds a `Weak`.
    #[test]
    fn a_dead_kernel_wakes_nothing() {
        let kernel = Arc::new(OrbisKernel::new());
        kernel.create_equeue(0);
        let waker = EqueueSideEffectWaker::for_kernel(&kernel);
        assert_eq!(waker.wake_side_effect_observers(), 1);
        drop(kernel);
        assert_eq!(waker.wake_side_effect_observers(), 0);
    }
}
