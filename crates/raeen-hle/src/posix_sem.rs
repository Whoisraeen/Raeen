//! HLE **POSIX semaphores** (`sem_init`/`sem_wait`/`sem_timedwait`/
//! `sem_post`/`sem_destroy`) under `libScePosix`.
//!
//! These are **address-based** objects: the guest owns a `sem_t` and every
//! call names it by pointer, so state is keyed by that guest address
//! (`OrbisKernel::posix_semaphores`) — deliberately distinct from the
//! handle-based `sceKernelCreateSema` family in `kernel_semaphore`.
//!
//! Blocking is real (FMOD's worker threads park on these): a waiter sleeps on
//! a host condvar in bounded slices, re-checking process termination each
//! slice so a terminating process can never deadlock on a semaphore nobody
//! will ever post — the same discipline as `pthread_cond`'s `wait_core`.
//!
//! # Return convention
//!
//! POSIX `sem_*` return `0`/`-1` with `errno`, not an errno value directly
//! (unlike `pthread_mutex_*`). Failures here return `-1` **and** store the
//! errno in the calling thread's `__error()` slot via
//! [`crate::libkernel::set_guest_errno`].

use crate::{HleContext, HleRegistry};
use raeen_kernel::PosixSem;
use std::sync::Arc;
use tracing::debug;

const OK: u64 = 0;
const MINUS_ONE: u64 = u64::MAX;
// POSIX errno values (FreeBSD/Orbis numbering).
const EINTR: i32 = 4;
const EINVAL: i32 = 22;
const ETIMEDOUT: i32 = 60;

/// Cooperative wait slice: short enough that termination is noticed promptly.
const WAIT_SLICE: std::time::Duration = std::time::Duration::from_millis(10);

/// Register the POSIX semaphore HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libScePosix", "sem_init", hle_sem_init);
    registry.register("libScePosix", "sem_destroy", hle_sem_destroy);
    registry.register("libScePosix", "sem_wait", hle_sem_wait);
    registry.register("libScePosix", "sem_trywait", hle_sem_trywait);
    registry.register("libScePosix", "sem_timedwait", hle_sem_timedwait);
    registry.register("libScePosix", "sem_post", hle_sem_post);
    registry.register("libScePosix", "sem_getvalue", hle_sem_getvalue);

    // scePthreadSem* — Sony's counting-semaphore API. Address-based like the
    // POSIX sem_* family and SHARING the same kernel state (posix_semaphores),
    // so a title mixing the two spellings sees one object; they differ only in
    // the return convention (SCE error codes rather than -1/errno). Ported from
    // SharpEmu `KernelSemaphoreCompatExports.cs` (#424, a60bfc9), which delegates
    // each scePthreadSem* to its Posix counterpart, adding the private-flag check
    // on Init and the BUSY->TRY_AGAIN translation on Trywait.
    registry.register("libkernel", "scePthreadSemInit", hle_pthread_sem_init);
    registry.register("libkernel", "scePthreadSemWait", hle_pthread_sem_wait);
    registry.register("libkernel", "scePthreadSemTrywait", hle_pthread_sem_trywait);
    registry.register("libkernel", "scePthreadSemPost", hle_pthread_sem_post);
    registry.register("libkernel", "scePthreadSemDestroy", hle_pthread_sem_destroy);
}

// SCE return convention for scePthreadSem*: 0 on success, real Orbis
// `SCE_KERNEL_ERROR_*` codes (`0x8002_0000 | errno`) on failure.
const SCE_OK: u64 = 0;
const SCE_EINVAL: u64 = 0x8002_0016;
/// `SCE_KERNEL_ERROR_EAGAIN` — SharpEmu's `ORBIS_GEN2_ERROR_TRY_AGAIN`.
const SCE_EAGAIN: u64 = 0x8002_0023;
/// `SCE_KERNEL_ERROR_EINTR`.
const SCE_EINTR: u64 = 0x8002_0004;

/// `scePthreadSemInit(sem, flag, value, name)`: only private semaphores are
/// supported (`flag == 0`); anything else is `EINVAL`. Otherwise fresh state is
/// registered with `value` counts, replacing any existing object at `sem`.
fn hle_pthread_sem_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let sem = args.first().copied().unwrap_or(0);
    let flag = args.get(1).copied().unwrap_or(0);
    let value = args.get(2).copied().unwrap_or(0);
    if flag != 0 || sem == 0 || value > i64::MAX as u64 {
        return SCE_EINVAL;
    }
    let state = Arc::new(PosixSem::default());
    *state.count.lock() = value as i64;
    ctx.kernel.posix_semaphores.insert(sem, state);
    debug!("scePthreadSemInit(sem={sem:#x}, value={value})");
    SCE_OK
}

/// `scePthreadSemWait(sem)`: decrement, parking until a post supplies a count
/// (real blocking, shared with `sem_wait`), reporting SCE codes.
fn hle_pthread_sem_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    match acquire(ctx, args.first().copied().unwrap_or(0), None) {
        Acquire::Ok => SCE_OK,
        Acquire::Interrupted => SCE_EINTR,
        // No deadline is supplied, so `TimedOut` cannot occur; treat any
        // non-acquire outcome other than interruption as an invalid semaphore.
        Acquire::Invalid | Acquire::TimedOut => SCE_EINVAL,
    }
}

/// `scePthreadSemTrywait(sem)`: consume an available count or report
/// `TRY_AGAIN` (SharpEmu maps the POSIX `BUSY`/`EAGAIN` onto `TRY_AGAIN`).
fn hle_pthread_sem_trywait(ctx: &HleContext, args: &[u64]) -> u64 {
    let sem = args.first().copied().unwrap_or(0);
    if sem == 0 {
        return SCE_EINVAL;
    }
    let state = semaphore(ctx, sem);
    let mut count = state.count.lock();
    if *count > 0 {
        *count -= 1;
        SCE_OK
    } else {
        SCE_EAGAIN
    }
}

/// `scePthreadSemPost(sem)`: supply one count and wake a parked waiter.
fn hle_pthread_sem_post(ctx: &HleContext, args: &[u64]) -> u64 {
    let sem = args.first().copied().unwrap_or(0);
    if sem == 0 {
        return SCE_EINVAL;
    }
    let state = semaphore(ctx, sem);
    let mut count = state.count.lock();
    if *count == i64::MAX {
        return SCE_EINVAL; // EOVERFLOW territory; EINVAL is the honest reject
    }
    *count += 1;
    state.posted.notify_one();
    SCE_OK
}

/// `scePthreadSemDestroy(sem)`: drop the tracked state. Destroying an unknown
/// semaphore is `EINVAL` (it names no initialized object).
fn hle_pthread_sem_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let sem = args.first().copied().unwrap_or(0);
    if sem == 0 || ctx.kernel.posix_semaphores.remove(&sem).is_none() {
        return SCE_EINVAL;
    }
    debug!("scePthreadSemDestroy(sem={sem:#x})");
    SCE_OK
}

fn fail(ctx: &HleContext, errno: i32) -> u64 {
    crate::libkernel::set_guest_errno(ctx, errno);
    MINUS_ONE
}

/// Fetch (creating on first touch) the semaphore state for a guest `sem_t`
/// address. Implicit creation (count 0) covers a semaphore reached without
/// `sem_init` — waiting on it parks exactly like waiting on a legitimately
/// empty one, instead of faulting.
fn semaphore(ctx: &HleContext, sem: u64) -> Arc<PosixSem> {
    ctx.kernel
        .posix_semaphores
        .entry(sem)
        .or_insert_with(|| Arc::new(PosixSem::default()))
        .clone()
}

/// `sem_init(sem, pshared, value)`: register fresh state with `value` counts.
/// Re-initializing an existing semaphore replaces its state (POSIX leaves
/// this undefined; replacing matches what a fresh object would observe).
fn hle_sem_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let sem = args.first().copied().unwrap_or(0);
    let value = args.get(2).copied().unwrap_or(0);
    if sem == 0 || value > i64::MAX as u64 {
        return fail(ctx, EINVAL);
    }
    let state = Arc::new(PosixSem::default());
    *state.count.lock() = value as i64;
    ctx.kernel.posix_semaphores.insert(sem, state);
    debug!("sem_init(sem={sem:#x}, value={value})");
    OK
}

/// `sem_destroy(sem)`: drop the state. Destroying an unknown semaphore is
/// `EINVAL` (it names no initialized object).
fn hle_sem_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let sem = args.first().copied().unwrap_or(0);
    if sem == 0 || ctx.kernel.posix_semaphores.remove(&sem).is_none() {
        return fail(ctx, EINVAL);
    }
    debug!("sem_destroy(sem={sem:#x})");
    OK
}

/// Outcome of a blocking acquire, shared by the POSIX (`sem_wait`) and SCE
/// (`scePthreadSemWait`) wrappers so the parking loop — including the
/// process-termination self-heal — lives in exactly one place.
enum Acquire {
    Ok,
    Interrupted,
    TimedOut,
    Invalid,
}

/// Shared acquire body. `deadline == None` waits indefinitely (bounded by
/// process termination); `Some` yields [`Acquire::TimedOut`] once the
/// wall-clock deadline passes without an available count. Returns an outcome so
/// each ABI wrapper can format its own return convention (POSIX `-1`/errno vs
/// SCE error code).
fn acquire(ctx: &HleContext, sem: u64, deadline: Option<std::time::SystemTime>) -> Acquire {
    if sem == 0 {
        return Acquire::Invalid;
    }
    let state = semaphore(ctx, sem);
    let mut count = state.count.lock();
    loop {
        if *count > 0 {
            *count -= 1;
            return Acquire::Ok;
        }
        if ctx.guest_threads.process_is_terminating() {
            // Unblock a parked worker so process teardown can finish.
            return Acquire::Interrupted;
        }
        // A queued Orbis exception interrupts this wait, and then it RESUMES —
        // *not* `Acquire::Interrupted`/`EINTR`, which is reserved for teardown.
        // The count guard is released across delivery: `sem_post` takes it, and a
        // handler that acknowledges by posting is the normal case (see
        // `crate::exception`).
        if crate::exception::pending_at_wait_slice(ctx) {
            drop(count);
            crate::exception::deliver_at_wait_slice(ctx);
            count = state.count.lock();
            continue;
        }
        if let Some(deadline) = deadline {
            let Ok(remaining) = deadline.duration_since(std::time::SystemTime::now()) else {
                return Acquire::TimedOut;
            };
            state.posted.wait_for(&mut count, remaining.min(WAIT_SLICE));
        } else {
            state.posted.wait_for(&mut count, WAIT_SLICE);
        }
    }
}

/// Shared wait body for the POSIX `sem_*` family: `-1` plus errno on failure,
/// `0` on success. `EINTR` is the POSIX shape of "the wait was interrupted".
fn wait_core(ctx: &HleContext, sem: u64, deadline: Option<std::time::SystemTime>) -> u64 {
    match acquire(ctx, sem, deadline) {
        Acquire::Ok => OK,
        Acquire::Interrupted => fail(ctx, EINTR),
        Acquire::TimedOut => fail(ctx, ETIMEDOUT),
        Acquire::Invalid => fail(ctx, EINVAL),
    }
}

/// `sem_wait(sem)`: decrement, parking until a `sem_post` supplies a count.
fn hle_sem_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    wait_core(ctx, args.first().copied().unwrap_or(0), None)
}

/// `sem_trywait(sem)`: decrement if immediately available, else
/// `EWOULDBLOCK`-shaped `EAGAIN` (35 on FreeBSD, where EAGAIN==EWOULDBLOCK).
fn hle_sem_trywait(ctx: &HleContext, args: &[u64]) -> u64 {
    const EAGAIN: i32 = 35;
    let sem = args.first().copied().unwrap_or(0);
    if sem == 0 {
        return fail(ctx, EINVAL);
    }
    let state = semaphore(ctx, sem);
    let mut count = state.count.lock();
    if *count > 0 {
        *count -= 1;
        OK
    } else {
        drop(count);
        fail(ctx, EAGAIN)
    }
}

/// `sem_timedwait(sem, abstime)`: `sem_wait` bounded by an **absolute**
/// `CLOCK_REALTIME` deadline (a `timespec`: `tv_sec` then `tv_nsec`, both
/// 64-bit). An already-expired deadline still succeeds if a count is
/// available (per POSIX), otherwise reports `ETIMEDOUT`.
fn hle_sem_timedwait(ctx: &HleContext, args: &[u64]) -> u64 {
    let sem = args.first().copied().unwrap_or(0);
    let abstime = args.get(1).copied().unwrap_or(0);
    let mut raw = [0u8; 16];
    if abstime == 0 || !ctx.mem.read(abstime, &mut raw) {
        return fail(ctx, EINVAL);
    }
    let secs = i64::from_le_bytes(raw[..8].try_into().expect("fixed slice"));
    let nanos = i64::from_le_bytes(raw[8..].try_into().expect("fixed slice"));
    if !(0..1_000_000_000).contains(&nanos) {
        return fail(ctx, EINVAL);
    }
    let deadline =
        std::time::UNIX_EPOCH + std::time::Duration::new(secs.max(0) as u64, nanos as u32);
    wait_core(ctx, sem, Some(deadline))
}

/// `sem_post(sem)`: supply one count and wake a parked waiter.
fn hle_sem_post(ctx: &HleContext, args: &[u64]) -> u64 {
    let sem = args.first().copied().unwrap_or(0);
    if sem == 0 {
        return fail(ctx, EINVAL);
    }
    let state = semaphore(ctx, sem);
    let mut count = state.count.lock();
    if *count == i64::MAX {
        drop(count);
        return fail(ctx, EINVAL); // EOVERFLOW territory; EINVAL is the honest reject
    }
    *count += 1;
    state.posted.notify_one();
    OK
}

/// `sem_getvalue(sem, sval)`: write the current count through `sval`.
fn hle_sem_getvalue(ctx: &HleContext, args: &[u64]) -> u64 {
    let sem = args.first().copied().unwrap_or(0);
    let sval = args.get(1).copied().unwrap_or(0);
    if sem == 0 || sval == 0 {
        return fail(ctx, EINVAL);
    }
    let state = semaphore(ctx, sem);
    let count = *state.count.lock();
    let clamped = count.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    if !ctx.mem.write(sval, &clamped.to_le_bytes()) {
        return fail(ctx, EINVAL);
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn ctx_env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x800);
        (kernel, mem, alloc)
    }

    /// Write a `timespec` for `now + offset_ms` at guest address 0x100.
    fn write_abstime(mem: &crate::TestMemory, offset_ms: i64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let target_ns = (now.as_nanos() as i64) + offset_ms * 1_000_000;
        let secs = target_ns.div_euclid(1_000_000_000);
        let nanos = target_ns.rem_euclid(1_000_000_000);
        assert!(mem.write(0x100, &secs.to_le_bytes()));
        assert!(mem.write(0x108, &nanos.to_le_bytes()));
    }

    #[test]
    fn post_then_wait_returns_immediately() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let sem = 0x200u64;
        assert_eq!(hle_sem_init(&ctx, &[sem, 0, 0]), OK);
        assert_eq!(hle_sem_post(&ctx, &[sem]), OK);
        let started = std::time::Instant::now();
        assert_eq!(hle_sem_wait(&ctx, &[sem]), OK);
        assert!(
            started.elapsed() < std::time::Duration::from_millis(100),
            "a posted semaphore must be taken without parking"
        );
        // Count is back to zero: trywait now reports EAGAIN.
        assert_eq!(hle_sem_trywait(&ctx, &[sem]), MINUS_ONE);
    }

    #[test]
    fn init_value_supplies_initial_counts() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let sem = 0x210u64;
        assert_eq!(hle_sem_init(&ctx, &[sem, 0, 2]), OK);
        assert_eq!(hle_sem_wait(&ctx, &[sem]), OK);
        assert_eq!(hle_sem_wait(&ctx, &[sem]), OK);
        assert_eq!(hle_sem_trywait(&ctx, &[sem]), MINUS_ONE);
        // sem_getvalue reads the live count.
        assert_eq!(hle_sem_post(&ctx, &[sem]), OK);
        assert_eq!(hle_sem_getvalue(&ctx, &[sem, 0x300]), OK);
        let mut b = [0u8; 4];
        assert!(mem.read(0x300, &mut b));
        assert_eq!(i32::from_le_bytes(b), 1);
    }

    #[test]
    fn timedwait_times_out_on_an_empty_semaphore() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let sem = 0x220u64;
        assert_eq!(hle_sem_init(&ctx, &[sem, 0, 0]), OK);
        write_abstime(&mem, 30); // 30 ms from now
        let started = std::time::Instant::now();
        assert_eq!(hle_sem_timedwait(&ctx, &[sem, 0x100]), MINUS_ONE);
        let elapsed = started.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_millis(20),
            "timedwait must actually wait toward the deadline (waited {elapsed:?})"
        );
        // The errno slot holds ETIMEDOUT.
        let slot = crate::libkernel::hle_error_addr(&ctx, &[]);
        let mut e = [0u8; 4];
        assert!(mem.read(slot, &mut e));
        assert_eq!(i32::from_le_bytes(e), ETIMEDOUT);
    }

    #[test]
    fn timedwait_with_a_past_deadline_still_takes_an_available_count() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let sem = 0x230u64;
        assert_eq!(hle_sem_init(&ctx, &[sem, 0, 1]), OK);
        write_abstime(&mem, -1000); // already expired
        assert_eq!(hle_sem_timedwait(&ctx, &[sem, 0x100]), OK);
    }

    #[test]
    fn destroy_removes_state_and_rejects_unknowns() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let sem = 0x240u64;
        assert_eq!(hle_sem_init(&ctx, &[sem, 0, 1]), OK);
        assert_eq!(hle_sem_destroy(&ctx, &[sem]), OK);
        assert!(!kernel.posix_semaphores.contains_key(&sem));
        assert_eq!(hle_sem_destroy(&ctx, &[sem]), MINUS_ONE);
        assert_eq!(hle_sem_init(&ctx, &[0, 0, 0]), MINUS_ONE);
    }

    #[test]
    fn cross_thread_post_wakes_a_parked_waiter() {
        // The real FMOD shape: one thread parks in sem_wait, another posts.
        // Kernel state is shared; each thread builds its own ctx over it.
        let kernel = std::sync::Arc::new(raeen_kernel::OrbisKernel::new());
        let sem = 0x250u64;
        {
            let mem = crate::TestMemory::new(0x1000);
            let alloc = crate::TestAllocator::new(0x800);
            let ctx = test_ctx(&kernel, &mem, &alloc);
            assert_eq!(hle_sem_init(&ctx, &[sem, 0, 0]), OK);
        }
        let waiter_kernel = kernel.clone();
        let waiter = std::thread::spawn(move || {
            let mem = crate::TestMemory::new(0x1000);
            let alloc = crate::TestAllocator::new(0x800);
            let ctx = test_ctx(&waiter_kernel, &mem, &alloc);
            hle_sem_wait(&ctx, &[sem])
        });
        std::thread::sleep(std::time::Duration::from_millis(30));
        {
            let mem = crate::TestMemory::new(0x1000);
            let alloc = crate::TestAllocator::new(0x800);
            let ctx = test_ctx(&kernel, &mem, &alloc);
            assert_eq!(hle_sem_post(&ctx, &[sem]), OK);
        }
        assert_eq!(waiter.join().unwrap(), OK);
    }

    #[test]
    fn registered_under_libsce_posix() {
        let registry = HleRegistry::new();
        for name in [
            "sem_init",
            "sem_destroy",
            "sem_wait",
            "sem_trywait",
            "sem_timedwait",
            "sem_post",
            "sem_getvalue",
        ] {
            assert!(
                registry.is_implemented("libScePosix", name),
                "libScePosix::{name} must be registered"
            );
        }
    }

    #[test]
    fn pthread_sem_family_registered_under_libkernel() {
        let registry = HleRegistry::new();
        for name in [
            "scePthreadSemInit",
            "scePthreadSemWait",
            "scePthreadSemTrywait",
            "scePthreadSemPost",
            "scePthreadSemDestroy",
        ] {
            assert!(
                registry.is_implemented("libkernel", name),
                "libkernel::{name} must be registered"
            );
        }
    }

    #[test]
    fn pthread_sem_uses_sce_return_convention_and_private_flag_check() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let sem = 0x260u64;
        // Non-private (flag != 0) is rejected with the SCE invalid-argument code.
        assert_eq!(hle_pthread_sem_init(&ctx, &[sem, 1, 0, 0]), SCE_EINVAL);
        // Private init, then post/wait share one object and return SCE codes.
        assert_eq!(hle_pthread_sem_init(&ctx, &[sem, 0, 0, 0]), SCE_OK);
        assert_eq!(hle_pthread_sem_trywait(&ctx, &[sem]), SCE_EAGAIN);
        assert_eq!(hle_pthread_sem_post(&ctx, &[sem]), SCE_OK);
        assert_eq!(hle_pthread_sem_trywait(&ctx, &[sem]), SCE_OK);
        // Destroy removes state; a second destroy is EINVAL.
        assert_eq!(hle_pthread_sem_destroy(&ctx, &[sem]), SCE_OK);
        assert_eq!(hle_pthread_sem_destroy(&ctx, &[sem]), SCE_EINVAL);
    }

    #[test]
    fn pthread_sem_and_posix_sem_share_one_object() {
        // A title that inits with scePthreadSemInit and waits with sem_wait (or
        // vice versa) must see the same counting semaphore — both key on the
        // guest address in kernel.posix_semaphores.
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let sem = 0x270u64;
        assert_eq!(hle_pthread_sem_init(&ctx, &[sem, 0, 1, 0]), SCE_OK);
        // POSIX sem_wait consumes the count scePthreadSemInit seeded.
        assert_eq!(hle_sem_wait(&ctx, &[sem]), OK);
        // POSIX sem_post feeds a scePthreadSemWait.
        assert_eq!(hle_sem_post(&ctx, &[sem]), OK);
        assert_eq!(hle_pthread_sem_wait(&ctx, &[sem]), SCE_OK);
    }
}
