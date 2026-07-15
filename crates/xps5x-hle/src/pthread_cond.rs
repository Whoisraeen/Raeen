//! pthread **condition variables**, to the exact extent the single-guest-thread
//! model can implement them honestly.
//!
//! # What is correct here, and what is missing
//!
//! XPS5X runs exactly one guest execution context (see `pthread_sync`'s module
//! docs and M1-E in the progress ledger). That is not merely a limitation for
//! condition variables — it *determines* their semantics, and splits this API
//! cleanly in two:
//!
//! * **`signal` / `broadcast` are genuinely correct, not stubs.** They wake
//!   waiting threads. With one guest thread, the caller *is* that thread, so a
//!   waiter cannot exist — anything waiting would be blocked and unable to make
//!   this call. Waking nobody and returning 0 is exactly what POSIX specifies
//!   for a condition variable with no waiters.
//! * **`wait` / `timedwait` cannot be implemented at all**, and are deliberately
//!   left unregistered so a guest that calls one gets a loud, self-naming
//!   `UnimplementedImport` fault. Every alternative lies:
//!   returning 0 fakes a spurious wakeup (POSIX permits those), so the guest
//!   re-checks a predicate only another thread could have set and **spins
//!   forever**; blocking really deadlocks, since no other thread can ever
//!   signal; and returning an error makes correct guest code take a failure
//!   path it has no reason to take. A missing import stops with the function's
//!   name attached — a livelock stops with nothing. See
//!   `pthread_sync::register_posix` for the same reasoning applied to
//!   `pthread_create`.
//!
//! When M1-E lands real guest threads, `wait`/`timedwait` belong here and
//! `signal`/`broadcast` grow a real wait queue — at which point the reasoning
//! above stops holding and these must be revisited together.
//!
//! Only the POSIX spellings are registered: a NID hashes the function name
//! alone, and the measured retail title imports `pthread_cond_*` from
//! `libScePosix` and never mentions `scePthreadCond*`.

use tracing::debug;

use crate::{HleContext, HleRegistry};

/// POSIX success. These entry points return errno directly (0 = success),
/// matching `pthread_sync`.
const OK: u64 = 0;
const EINVAL: u64 = 22;

/// Register the condition-variable entry points that can be implemented
/// correctly under one guest thread. `wait`/`timedwait` are intentionally
/// absent — see the module docs.
pub fn register(registry: &HleRegistry) {
    registry.register("libScePosix", "pthread_cond_init", hle_cond_init);
    registry.register("libScePosix", "pthread_cond_destroy", hle_cond_destroy);
    registry.register("libScePosix", "pthread_cond_signal", hle_cond_signal);
    registry.register("libScePosix", "pthread_cond_broadcast", hle_cond_broadcast);
    registry.register("libScePosix", "pthread_condattr_init", hle_condattr_ok);
    registry.register("libScePosix", "pthread_condattr_destroy", hle_condattr_ok);
    registry.register("libScePosix", "pthread_condattr_setclock", hle_condattr_ok);
}

/// `pthread_cond_init(cond, attr)`. A condition variable carries no state we
/// need while it can have no waiters, so this only validates the pointer.
fn hle_cond_init(_ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    debug!("pthread_cond_init(cond={cond:#x})");
    if cond == 0 { EINVAL } else { OK }
}

/// `pthread_cond_destroy(cond)`.
fn hle_cond_destroy(_ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    debug!("pthread_cond_destroy(cond={cond:#x})");
    if cond == 0 { EINVAL } else { OK }
}

/// `pthread_cond_signal(cond)` — wake one waiter.
///
/// Correct, not a stub: with one guest thread there are no waiters to wake, and
/// POSIX defines signalling a condition variable with no waiters as a no-op
/// returning success.
fn hle_cond_signal(_ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    debug!("pthread_cond_signal(cond={cond:#x}) [no waiters possible: one guest thread]");
    if cond == 0 { EINVAL } else { OK }
}

/// `pthread_cond_broadcast(cond)` — wake all waiters. Same reasoning as
/// [`hle_cond_signal`].
fn hle_cond_broadcast(_ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    debug!("pthread_cond_broadcast(cond={cond:#x}) [no waiters possible: one guest thread]");
    if cond == 0 { EINVAL } else { OK }
}

/// `pthread_condattr_init/destroy/setclock` — attribute objects carry nothing
/// that affects behaviour while there are no waiters.
fn hle_condattr_ok(_ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    if attr == 0 { EINVAL } else { OK }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestAllocator, TestMemory, test_ctx};

    fn fixture() -> (xps5x_kernel::OrbisKernel, TestMemory, TestAllocator) {
        (
            xps5x_kernel::OrbisKernel::new(),
            TestMemory::new(0x1000),
            TestAllocator::new(0x8000),
        )
    }

    #[test]
    fn signal_and_broadcast_succeed_with_no_waiters() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_cond_signal(&ctx, &[0x1000]), OK);
        assert_eq!(hle_cond_broadcast(&ctx, &[0x1000]), OK);
    }

    #[test]
    fn init_destroy_round_trip() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_cond_init(&ctx, &[0x2000, 0]), OK);
        assert_eq!(hle_cond_destroy(&ctx, &[0x2000]), OK);
    }

    #[test]
    fn null_cond_is_einval_not_a_silent_success() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_cond_init(&ctx, &[0]), EINVAL);
        assert_eq!(hle_cond_destroy(&ctx, &[0]), EINVAL);
        assert_eq!(hle_cond_signal(&ctx, &[0]), EINVAL);
        assert_eq!(hle_cond_broadcast(&ctx, &[0]), EINVAL);
        assert_eq!(hle_condattr_ok(&ctx, &[0]), EINVAL);
    }

    /// `wait`/`timedwait` must stay UNREGISTERED until real guest threads
    /// exist. Registering them cannot be done honestly under one thread: 0
    /// fakes a spurious wakeup and the guest spins forever, blocking really
    /// deadlocks, and an error sends correct code down a failure path. An
    /// unresolved import at least reports its own name.
    #[test]
    fn wait_is_deliberately_not_implemented() {
        let registry = HleRegistry::new();
        assert!(
            !registry.is_implemented("libScePosix", "pthread_cond_wait"),
            "pthread_cond_wait cannot be implemented under one guest thread — a fake return \
             livelocks the guest instead of naming the missing capability (M1-E)"
        );
        assert!(!registry.is_implemented("libScePosix", "pthread_cond_timedwait"));
        // ...while the half that IS implementable is registered.
        assert!(registry.is_implemented("libScePosix", "pthread_cond_broadcast"));
        assert!(registry.is_implemented("libScePosix", "pthread_cond_signal"));
    }
}
