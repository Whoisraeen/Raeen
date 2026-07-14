//! HLE libkernel **counting semaphores** (`sceKernelCreate/Signal/Wait/Poll/
//! Cancel/DeleteSema`).
//!
//! A faithful Rust port of SharpEmu's `KernelSemaphoreCompatExports` (GPL-2.0).
//! The count arithmetic (Create/Signal/Poll/Cancel/Delete) is **fully correct**
//! and lives in the kernel (`OrbisKernel::kernel_semaphores`). `Wait` on an
//! empty semaphore blocks a real thread until another signals; under XPS5X's
//! single-active-execution model there is no other thread, so `Wait` decrements
//! immediately when the count is available (the common same-thread case) and
//! otherwise reports a timeout rather than hanging. True blocking waits arrive
//! with the M1-E scheduler.

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
    let Some(mut sem) = ctx.kernel.kernel_semaphores.get_mut(&handle) else {
        return SCE_KERNEL_ERROR_ESRCH;
    };
    if signal < 1 || sem.count > sem.max_count - signal {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    sem.count += signal;
    OK
}

/// Shared body for Poll (never blocks) and Wait (decrements if available).
fn poll_or_wait(ctx: &HleContext, args: &[u64], unavailable_err: u64) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    let need = args.get(1).copied().unwrap_or(0) as i32;
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
    poll_or_wait(ctx, args, SCE_KERNEL_ERROR_EBUSY)
}

/// `sceKernelWaitSema(handle, needCount, timeout)`: decrements immediately if
/// the count is available; otherwise reports a timeout (no other thread can
/// signal it under single-active-execution).
fn hle_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    poll_or_wait(ctx, args, SCE_KERNEL_ERROR_ETIMEDOUT)
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
    fn wait_times_out_when_count_unavailable() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let h = create(&ctx, 0, 2);
        // count 0, need 1 → nothing else can signal → timeout.
        assert_eq!(
            hle_wait(&ctx, &[h as u64, 1, 0]),
            SCE_KERNEL_ERROR_ETIMEDOUT
        );
        // need beyond max → EINVAL.
        assert_eq!(hle_wait(&ctx, &[h as u64, 5, 0]), SCE_KERNEL_ERROR_EINVAL);
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
