//! Host-backed pthread condition variables for concurrent native guest
//! workers. Wait atomically releases the associated guest mutex while holding
//! the condition generation lock, sleeps, then reacquires before returning.

use tracing::debug;

use crate::{HleContext, HleRegistry};

/// POSIX success. These entry points return errno directly (0 = success),
/// matching `pthread_sync`.
const OK: u64 = 0;
const EINVAL: u64 = 22;
const ETIMEDOUT: u64 = 60;

pub fn register(registry: &HleRegistry) {
    for library in ["libScePosix", "libkernel"] {
        registry.register(library, "pthread_cond_init", hle_cond_init);
        registry.register(library, "pthread_cond_destroy", hle_cond_destroy);
        registry.register(library, "pthread_cond_wait", hle_cond_wait);
        registry.register(library, "pthread_cond_timedwait", hle_cond_timedwait);
        registry.register(library, "pthread_cond_signal", hle_cond_signal);
        registry.register(library, "pthread_cond_broadcast", hle_cond_broadcast);
    }
    registry.register("libkernel", "scePthreadCondInit", hle_cond_init);
    registry.register("libkernel", "scePthreadCondWait", hle_cond_wait);
    registry.register("libkernel", "scePthreadCondTimedwait", hle_cond_timedwait);
    registry.register("libkernel", "scePthreadCondSignal", hle_cond_signal);
    registry.register("libkernel", "scePthreadCondBroadcast", hle_cond_broadcast);
    registry.register("libScePosix", "pthread_condattr_init", hle_condattr_ok);
    registry.register("libScePosix", "pthread_condattr_destroy", hle_condattr_ok);
    registry.register("libScePosix", "pthread_condattr_setclock", hle_condattr_ok);
}

/// `pthread_cond_init(cond, attr)`. A condition variable carries no state we
/// need while it can have no waiters, so this only validates the pointer.
fn hle_cond_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    debug!("pthread_cond_init(cond={cond:#x})");
    if cond == 0 {
        EINVAL
    } else {
        ctx.kernel.pthread_conds.insert(
            cond,
            std::sync::Arc::new(xps5x_kernel::PthreadCond::default()),
        );
        OK
    }
}

/// `pthread_cond_destroy(cond)`.
fn hle_cond_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    debug!("pthread_cond_destroy(cond={cond:#x})");
    if cond == 0 {
        EINVAL
    } else {
        ctx.kernel.pthread_conds.remove(&cond);
        OK
    }
}

fn condition(ctx: &HleContext, cond: u64) -> std::sync::Arc<xps5x_kernel::PthreadCond> {
    ctx.kernel
        .pthread_conds
        .entry(cond)
        .or_insert_with(|| std::sync::Arc::new(xps5x_kernel::PthreadCond::default()))
        .clone()
}

fn hle_cond_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    wait_core(ctx, args, None)
}

fn hle_cond_timedwait(ctx: &HleContext, args: &[u64]) -> u64 {
    let timeout = args
        .get(2)
        .copied()
        .filter(|ptr| *ptr != 0)
        .and_then(|ptr| {
            let mut raw = [0u8; 16];
            ctx.mem.read(ptr, &mut raw).then(|| {
                let secs = i64::from_le_bytes(raw[..8].try_into().expect("fixed slice"));
                let nanos = i64::from_le_bytes(raw[8..].try_into().expect("fixed slice"));
                let target = std::time::UNIX_EPOCH
                    + std::time::Duration::new(
                        secs.max(0) as u64,
                        nanos.clamp(0, 999_999_999) as u32,
                    );
                target
                    .duration_since(std::time::SystemTime::now())
                    .unwrap_or_default()
            })
        });
    wait_core(ctx, args, timeout)
}

fn wait_core(ctx: &HleContext, args: &[u64], timeout: Option<std::time::Duration>) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    let mutex = args.get(1).copied().unwrap_or(0);
    if cond == 0 || mutex == 0 {
        return EINVAL;
    }
    let state = condition(ctx, cond);
    let mut generation = state.generation.lock();
    let observed = *generation;
    let unlock = crate::pthread_sync::mutex_unlock_for_cond(ctx, mutex);
    if unlock != OK {
        return unlock;
    }
    let started = std::time::Instant::now();
    let mut timed_out = false;
    while *generation == observed && !ctx.guest_threads.process_is_terminating() {
        let slice = timeout
            .map(|limit| limit.saturating_sub(started.elapsed()))
            .unwrap_or(std::time::Duration::from_millis(10))
            .min(std::time::Duration::from_millis(10));
        if timeout.is_some() && slice.is_zero() {
            timed_out = true;
            break;
        }
        state.changed.wait_for(&mut generation, slice);
    }
    drop(generation);
    let relock = crate::pthread_sync::mutex_lock_for_cond(ctx, mutex);
    if relock != OK {
        return relock;
    }
    if timed_out { ETIMEDOUT } else { OK }
}

/// `pthread_cond_signal(cond)` — wake one waiter.
///
/// Correct, not a stub: with one guest thread there are no waiters to wake, and
/// POSIX defines signalling a condition variable with no waiters as a no-op
/// returning success.
fn hle_cond_signal(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    if cond == 0 {
        return EINVAL;
    }
    let state = condition(ctx, cond);
    *state.generation.lock() += 1;
    state.changed.notify_one();
    OK
}

/// `pthread_cond_broadcast(cond)` — wake all waiters. Same reasoning as
/// [`hle_cond_signal`].
fn hle_cond_broadcast(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    if cond == 0 {
        return EINVAL;
    }
    let state = condition(ctx, cond);
    *state.generation.lock() += 1;
    state.changed.notify_all();
    OK
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

    #[test]
    fn wait_is_registered_for_real_guest_workers() {
        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libScePosix", "pthread_cond_wait"));
        assert!(registry.is_implemented("libScePosix", "pthread_cond_timedwait"));
        assert!(registry.is_implemented("libScePosix", "pthread_cond_broadcast"));
        assert!(registry.is_implemented("libScePosix", "pthread_cond_signal"));
    }
}
