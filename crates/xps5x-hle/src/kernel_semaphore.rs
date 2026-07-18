//! HLE libkernel **counting semaphores** (`sceKernelCreate/Signal/Wait/Poll/
//! Cancel/DeleteSema`).
//!
//! A faithful Rust port of SharpEmu's `KernelSemaphoreCompatExports` (GPL-2.0).
//! The count arithmetic (Create/Signal/Poll/Cancel/Delete) lives in the kernel
//! (`OrbisKernel::kernel_semaphores`). `Wait` **truly blocks** on the shared
//! `semaphore_signal` condvar until another guest thread signals the count or a
//! finite timeout expires — with real concurrent guest threads a producer *does*
//! signal, so the old instant-`ETIMEDOUT` was wrong: a job-system title (e.g.
//! ASTRO.BOT `Semaphore.cpp:63`) asserts when its worker threads' waits time out
//! at boot. A NULL timeout waits forever (never synthesizes `ETIMEDOUT`); a
//! finite one returns `ETIMEDOUT` only when the deadline genuinely passes. This
//! mirrors the same conversion already done for event flags (`kernel_eventflag`).

use crate::{HleContext, HleRegistry};
use tracing::debug;

const OK: u64 = 0;
// SCE kernel error codes (`0x8002_0000 | errno`).
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016; // 22
const SCE_KERNEL_ERROR_ESRCH: u64 = 0x8002_0003; // 3 (no such semaphore)
const SCE_KERNEL_ERROR_EFAULT: u64 = 0x8002_000E; // 14
const SCE_KERNEL_ERROR_EBUSY: u64 = 0x8002_0010; // 16 (poll not satisfied)
const SCE_KERNEL_ERROR_ETIMEDOUT: u64 = 0x8002_003C; // 60

const MAX_NAME_LEN: usize = 31;

/// Register the semaphore HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "sceKernelCreateSema", hle_create);
    registry.register("libkernel", "sceKernelDeleteSema", hle_delete);
    registry.register("libkernel", "sceKernelSignalSema", hle_signal);
    registry.register("libkernel", "sceKernelWaitSema", hle_wait);
    registry.register("libkernel", "sceKernelPollSema", hle_poll);
    registry.register("libkernel", "sceKernelCancelSema", hle_cancel);
}

/// `sceKernelCreateSema(out, name, attr, initCount, maxCount, opt)`.
fn hle_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    let name_ptr = args.get(1).copied().unwrap_or(0);
    let attr = args.get(2).copied().unwrap_or(0);
    let initial = args.get(3).copied().unwrap_or(0) as i32;
    let max = args.get(4).copied().unwrap_or(0) as i32;
    let opt = args.get(5).copied().unwrap_or(0);

    if out == 0 || name_ptr == 0 || attr > 2 || initial < 0 || max <= 0 || initial > max || opt != 0
    {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let mut name_buf = [0u8; MAX_NAME_LEN + 2];
    if !ctx.mem.read(name_ptr, &mut name_buf) {
        return SCE_KERNEL_ERROR_EFAULT;
    }

    let handle = ctx.kernel.create_semaphore(initial, max);
    if !ctx.mem.write(out, &handle.to_le_bytes()) {
        ctx.kernel.kernel_semaphores.remove(&handle);
        return SCE_KERNEL_ERROR_EFAULT;
    }
    debug!("sceKernelCreateSema -> handle {handle:#x} init {initial} max {max}");
    OK
}

/// `sceKernelDeleteSema(handle)`.
fn hle_delete(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    if ctx.kernel.kernel_semaphores.remove(&handle).is_none() {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    OK
}

/// `sceKernelSignalSema(handle, signalCount)`: add to the count, up to the
/// ceiling (overflowing the max is an error).
fn hle_signal(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    let signal = args.get(1).copied().unwrap_or(0) as i32;
    {
        let Some(mut sem) = ctx.kernel.kernel_semaphores.get_mut(&handle) else {
            return SCE_KERNEL_ERROR_ESRCH;
        };
        if signal < 1 || sem.count > sem.max_count - signal {
            return SCE_KERNEL_ERROR_EINVAL;
        }
        sem.count += signal;
    }
    // Wake every blocked `WaitSema` so it re-checks the new count. Notifying
    // under the shared lock closes the check-then-sleep race (see hle_wait).
    let (lock, cvar) = &ctx.kernel.semaphore_signal;
    let _guard = lock.lock().unwrap();
    cvar.notify_all();
    OK
}

/// Non-blocking consume: `EINVAL` on a bad `need`, `unavailable_err` if the
/// count is short, else decrement and `OK`. Used by Poll, and by Wait for its
/// first (fast-path) attempt.
fn try_consume(ctx: &HleContext, handle: u32, need: i32, unavailable_err: u64) -> u64 {
    let Some(mut sem) = ctx.kernel.kernel_semaphores.get_mut(&handle) else {
        return SCE_KERNEL_ERROR_ESRCH;
    };
    if need < 1 || need > sem.max_count {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    if sem.count < need {
        return unavailable_err;
    }
    sem.count -= need;
    OK
}

/// `sceKernelPollSema(handle, needCount)`: non-blocking; `EBUSY` if the count
/// isn't available.
fn hle_poll(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    let need = args.get(1).copied().unwrap_or(0) as i32;
    try_consume(ctx, handle, need, SCE_KERNEL_ERROR_EBUSY)
}

/// `sceKernelWaitSema(handle, needCount, timeout)`: consume `needCount` from the
/// semaphore, **blocking** on the shared `semaphore_signal` condvar until a
/// producer thread signals enough count. `timeout` is `SceKernelUseconds*` (a
/// u32 of microseconds); NULL waits forever and never synthesizes a timeout — a
/// finite value returns `ETIMEDOUT` only once its deadline genuinely passes.
fn hle_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    let need = args.get(1).copied().unwrap_or(0) as i32;
    let timeout_ptr = args.get(2).copied().unwrap_or(0);

    // Fast path: if the count is already available (or the args are bad, or the
    // handle is unknown), settle it without touching the condvar.
    let fast = try_consume(ctx, handle, need, SCE_KERNEL_ERROR_EBUSY);
    if fast != SCE_KERNEL_ERROR_EBUSY {
        return fast; // OK, EINVAL, or ESRCH
    }

    let deadline = if timeout_ptr == 0 {
        None
    } else {
        let mut raw = [0u8; 4];
        if !ctx.mem.read(timeout_ptr, &mut raw) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        Some(std::time::Instant::now() + std::time::Duration::from_micros(u64::from(u32::from_le_bytes(raw))))
    };

    let (lock, cvar) = &ctx.kernel.semaphore_signal;
    let mut guard = lock.lock().unwrap();
    // Cap each host wait so process teardown and any missed notify are noticed
    // promptly regardless of the guest's (possibly infinite) timeout.
    let slice = std::time::Duration::from_millis(100);
    loop {
        // Honor process teardown: a parked worker MUST wake and unwind, or
        // `terminate_and_reap`'s join hangs forever waiting for it (the cond and
        // event-flag waits honor this too — returning here lets the thread reach
        // a termination checkpoint in dispatch). This is the counterpart to
        // making the wait truly block: an infinite block must still be escapable.
        if ctx.guest_threads.process_is_terminating() {
            return OK;
        }
        // Re-check under the condvar lock, serialised against Signal's notify.
        let attempt = try_consume(ctx, handle, need, SCE_KERNEL_ERROR_EBUSY);
        if attempt != SCE_KERNEL_ERROR_EBUSY {
            return attempt; // OK, EINVAL, or ESRCH (deleted while waiting)
        }
        // A NULL timeout waits forever (never synthesizes a timeout the guest
        // asserts on); a finite one returns ETIMEDOUT once the deadline passes.
        let wait = match deadline {
            None => slice,
            Some(dl) => {
                let remaining = dl.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    return SCE_KERNEL_ERROR_ETIMEDOUT;
                }
                remaining.min(slice)
            }
        };
        let (g, _) = cvar.wait_timeout(guard, wait).unwrap();
        guard = g;
    }
}

/// `sceKernelCancelSema(handle, setCount, waiterCountOut)`: reset the count
/// (to `setCount` if ≥ 0, else the... just `setCount`), report 0 waiters.
fn hle_cancel(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    let set_count = args.get(1).copied().unwrap_or(0) as i32;
    let waiter_out = args.get(2).copied().unwrap_or(0);
    let Some(mut sem) = ctx.kernel.kernel_semaphores.get_mut(&handle) else {
        return SCE_KERNEL_ERROR_ESRCH;
    };
    if waiter_out != 0 && !ctx.mem.write(waiter_out, &0u32.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    // A negative setCount means "reset to the max"; clamp into range.
    sem.count = set_count.clamp(0, sem.max_count);
    drop(sem);
    // Cancel wakes every waiter so they re-evaluate against the reset count.
    let (lock, cvar) = &ctx.kernel.semaphore_signal;
    let _guard = lock.lock().unwrap();
    cvar.notify_all();
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn ctx_env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    fn create(ctx: &HleContext, initial: u64, max: u64) -> u32 {
        assert!(ctx.mem.write(0x40, b"sem\0"));
        assert_eq!(hle_create(ctx, &[0x100, 0x40, 0, initial, max, 0]), OK);
        let mut b = [0u8; 4];
        assert!(ctx.mem.read(0x100, &mut b));
        u32::from_le_bytes(b)
    }

    #[test]
    fn signal_poll_and_wait_track_the_count() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let h = create(&ctx, 1, 4);
        // Poll 1 → OK, count 0. Poll 1 again → EBUSY.
        assert_eq!(hle_poll(&ctx, &[h as u64, 1]), OK);
        assert_eq!(kernel.kernel_semaphores.get(&h).unwrap().count, 0);
        assert_eq!(hle_poll(&ctx, &[h as u64, 1]), SCE_KERNEL_ERROR_EBUSY);
        // Signal 3 → count 3. Wait 2 → OK, count 1.
        assert_eq!(hle_signal(&ctx, &[h as u64, 3]), OK);
        assert_eq!(hle_wait(&ctx, &[h as u64, 2, 0]), OK);
        assert_eq!(kernel.kernel_semaphores.get(&h).unwrap().count, 1);
    }

    #[test]
    fn signal_cannot_exceed_max_count() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let h = create(&ctx, 2, 3);
        // count 2, max 3: signalling 2 would overflow to 4 → EINVAL, count unchanged.
        assert_eq!(hle_signal(&ctx, &[h as u64, 2]), SCE_KERNEL_ERROR_EINVAL);
        assert_eq!(kernel.kernel_semaphores.get(&h).unwrap().count, 2);
        // signalling 1 reaches exactly max → OK.
        assert_eq!(hle_signal(&ctx, &[h as u64, 1]), OK);
        assert_eq!(kernel.kernel_semaphores.get(&h).unwrap().count, 3);
    }

    #[test]
    fn wait_with_finite_timeout_times_out_when_count_unavailable() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let h = create(&ctx, 0, 2);
        // A finite timeout (2000 us) at 0x200: nothing signals → ETIMEDOUT.
        assert!(mem.write(0x200, &2000u32.to_le_bytes()));
        assert_eq!(
            hle_wait(&ctx, &[h as u64, 1, 0x200]),
            SCE_KERNEL_ERROR_ETIMEDOUT
        );
        // need beyond max → EINVAL on the fast path (never blocks).
        assert_eq!(hle_wait(&ctx, &[h as u64, 5, 0x200]), SCE_KERNEL_ERROR_EINVAL);
    }

    /// The blocking-wait contract: a `WaitSema` with a NULL (forever) timeout
    /// must sleep until another thread signals the count, NOT instantly time out.
    /// This is the ASTRO.BOT `Semaphore.cpp:63` regression — its worker threads
    /// wait on an empty job semaphore at boot and asserted on ETIMEDOUT.
    #[test]
    fn wait_blocks_forever_until_another_thread_signals() {
        let kernel = std::sync::Arc::new(xps5x_kernel::OrbisKernel::new());
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(kernel.as_ref(), &mem, &alloc);
        let h = create(&ctx, 0, 4);
        let signaller = std::thread::spawn({
            let k2 = std::sync::Arc::clone(&kernel);
            move || {
                std::thread::sleep(std::time::Duration::from_millis(30));
                let mem2 = crate::TestMemory::new(0x100);
                let alloc2 = crate::TestAllocator::new(0);
                let ctx2 = test_ctx(k2.as_ref(), &mem2, &alloc2);
                assert_eq!(hle_signal(&ctx2, &[u64::from(h), 1]), OK);
            }
        });
        let start = std::time::Instant::now();
        // timeout ptr 0 = wait forever; must return OK only after the signal.
        assert_eq!(hle_wait(&ctx, &[u64::from(h), 1, 0]), OK);
        let elapsed = start.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_millis(25),
            "the wait must have BLOCKED for the producer, not spun (took {elapsed:?})"
        );
        signaller.join().unwrap();
    }

    #[test]
    fn create_validation_and_lifecycle() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x40, b"sem\0"));
        // max <= 0, initial > max, attr > 2, non-zero opt, NULL out → EINVAL.
        assert_eq!(
            hle_create(&ctx, &[0x100, 0x40, 0, 0, 0, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(
            hle_create(&ctx, &[0x100, 0x40, 0, 5, 3, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(
            hle_create(&ctx, &[0x100, 0x40, 3, 0, 1, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(
            hle_create(&ctx, &[0, 0x40, 0, 0, 1, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
        // Valid create → delete → second delete ESRCH; ops on unknown → ESRCH.
        let h = create(&ctx, 0, 1);
        assert_eq!(hle_delete(&ctx, &[h as u64]), OK);
        assert_eq!(hle_delete(&ctx, &[h as u64]), SCE_KERNEL_ERROR_ESRCH);
        assert_eq!(hle_signal(&ctx, &[0xDEAD, 1]), SCE_KERNEL_ERROR_ESRCH);
    }

    #[test]
    fn cancel_resets_count_and_reports_zero_waiters() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let h = create(&ctx, 3, 5);
        assert_eq!(hle_cancel(&ctx, &[h as u64, 1, 0x108]), OK);
        assert_eq!(kernel.kernel_semaphores.get(&h).unwrap().count, 1);
        let mut w = [0u8; 4];
        assert!(mem.read(0x108, &mut w));
        assert_eq!(u32::from_le_bytes(w), 0);
    }
}
