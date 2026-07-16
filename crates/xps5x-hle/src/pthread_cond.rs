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
const COND_OBJECT_SIZE: u64 = 0x100;

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

/// `pthread_cond_init(cond, attr)`. Orbis condition variables are opaque
/// pointer handles: initialize `*cond`, and retain the same host state under
/// both the guest pointer slot and its allocated handle. Guest libc inspects
/// the slot directly, so leaving it zero after reporting success can make it
/// mistake its own initialized condition for a static/uninitialized object.
fn hle_cond_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    debug!("pthread_cond_init(cond={cond:#x})");
    if cond == 0 {
        EINVAL
    } else {
        let Some(handle) = ctx.alloc.alloc(COND_OBJECT_SIZE, 0x10) else {
            return EINVAL;
        };
        if !ctx.mem.write(cond, &handle.to_le_bytes()) {
            ctx.alloc.free(handle);
            return EINVAL;
        }
        let state = std::sync::Arc::new(xps5x_kernel::PthreadCond::default());
        ctx.kernel.pthread_conds.insert(cond, state.clone());
        ctx.kernel.pthread_conds.insert(handle, state);
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
        let handle = read_handle(ctx, cond);
        ctx.kernel.pthread_conds.remove(&cond);
        if let Some(handle) = handle {
            ctx.kernel.pthread_conds.remove(&handle);
            ctx.alloc.free(handle);
        }
        let _ = ctx.mem.write(cond, &0u64.to_le_bytes());
        OK
    }
}

fn read_handle(ctx: &HleContext, cond: u64) -> Option<u64> {
    let mut bytes = [0u8; 8];
    ctx.mem
        .read(cond, &mut bytes)
        .then(|| u64::from_le_bytes(bytes))
        .filter(|handle| *handle != 0)
}

/// Resolve an initialized condition, lazily materializing a zero-initialized
/// static object with the same opaque-handle ABI as `pthread_cond_init`.
fn condition(ctx: &HleContext, cond: u64) -> Option<std::sync::Arc<xps5x_kernel::PthreadCond>> {
    if let Some(state) = ctx.kernel.pthread_conds.get(&cond) {
        return Some(state.clone());
    }
    if let Some(handle) = read_handle(ctx, cond)
        && let Some(state) = ctx.kernel.pthread_conds.get(&handle)
    {
        return Some(state.clone());
    }

    let handle = ctx.alloc.alloc(COND_OBJECT_SIZE, 0x10)?;
    if !ctx.mem.write(cond, &handle.to_le_bytes()) {
        ctx.alloc.free(handle);
        return None;
    }
    let state = std::sync::Arc::new(xps5x_kernel::PthreadCond::default());
    ctx.kernel.pthread_conds.insert(cond, state.clone());
    ctx.kernel.pthread_conds.insert(handle, state.clone());
    Some(state)
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
    let Some(state) = condition(ctx, cond) else {
        return EINVAL;
    };
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
        let wait = state.changed.wait_for(&mut generation, slice);
        // POSIX explicitly permits spurious condition-variable wakeups. Treat
        // the bounded host wait as one so an orphaned/stale guest waiter can
        // re-check its own predicate, while still polling process termination
        // without pinning a host thread forever inside the VEH.
        if timeout.is_none() && wait.timed_out() {
            break;
        }
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
    let Some(state) = condition(ctx, cond) else {
        return EINVAL;
    };
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
    let Some(state) = condition(ctx, cond) else {
        return EINVAL;
    };
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
    use crate::{GuestMemory, TestAllocator, TestMemory, test_ctx};

    fn fixture() -> (xps5x_kernel::OrbisKernel, TestMemory, TestAllocator) {
        (
            xps5x_kernel::OrbisKernel::new(),
            TestMemory::new(0x4000),
            TestAllocator::new(0x2000),
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

    #[test]
    fn static_wait_materializes_an_opaque_handle_and_can_wake_spuriously() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cond = 0x100;
        let mutex = 0x200;
        assert_eq!(crate::pthread_sync::mutex_lock_for_cond(&ctx, mutex), OK);

        let started = std::time::Instant::now();
        assert_eq!(hle_cond_wait(&ctx, &[cond, mutex]), OK);
        assert!(started.elapsed() < std::time::Duration::from_millis(100));

        let mut bytes = [0u8; 8];
        assert!(mem.read(cond, &mut bytes));
        let handle = u64::from_le_bytes(bytes);
        assert_ne!(handle, 0);
        assert!(kernel.pthread_conds.contains_key(&cond));
        assert!(kernel.pthread_conds.contains_key(&handle));
        // POSIX requires the mutex to be reacquired before wait returns.
        assert_eq!(crate::pthread_sync::mutex_unlock_for_cond(&ctx, mutex), OK);
    }
}
