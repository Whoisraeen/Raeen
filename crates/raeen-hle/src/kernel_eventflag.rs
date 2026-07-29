//! HLE libkernel **event flags** (`sceKernelCreate/Set/Clear/Poll/Wait/Cancel/
//! DeleteEventFlag`).
//!
//! An event flag is a 64-bit set of condition bits a title sets, clears, and
//! waits on. A faithful Rust port of SharpEmu's `KernelEventFlagCompatExports`
//! (GPL-2.0). The bit state (Create/Set/Clear/Poll/Cancel/Delete) is **fully
//! correct** and lives in the kernel (`OrbisKernel::kernel_event_flags`).
//!
//! `Wait` blocks a real thread until the pattern is satisfied; under Raeen's
//! single-active-execution model there is no *other* thread to satisfy an
//! unmet condition, so `Wait` completes immediately when the pattern is
//! already satisfied (the common same-thread case) and otherwise reports a
//! timeout rather than hanging. True cross-thread blocking waits arrive with
//! the M1-E scheduler.

use crate::{HleContext, HleRegistry};
use raeen_core::subsystems::{EventUpdate, WaitKey, WaitOutcome, WakeReason};

const OK: u64 = 0;
// SCE kernel error codes (`0x8002_0000 | errno`).
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016; // 22
const SCE_KERNEL_ERROR_ESRCH: u64 = 0x8002_0003; // 3 (no such flag)
const SCE_KERNEL_ERROR_EFAULT: u64 = 0x8002_000E; // 14
const SCE_KERNEL_ERROR_ENOMEM: u64 = 0x8002_000C; // 12
const SCE_KERNEL_ERROR_EBUSY: u64 = 0x8002_0010; // 16 (poll not satisfied)
const SCE_KERNEL_ERROR_ETIMEDOUT: u64 = 0x8002_003C; // 60

// Wait-mode bits.
const WAIT_AND: u64 = 0x01;
const WAIT_OR: u64 = 0x02;
const CLEAR_ALL: u64 = 0x10;
const CLEAR_PATTERN: u64 = 0x20;

const MAX_NAME_LEN: usize = 31;

/// Register the event-flag HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "sceKernelCreateEventFlag", hle_create);
    registry.register("libkernel", "sceKernelDeleteEventFlag", hle_delete);
    registry.register("libkernel", "sceKernelSetEventFlag", hle_set);
    registry.register("libkernel", "sceKernelClearEventFlag", hle_clear);
    registry.register("libkernel", "sceKernelPollEventFlag", hle_poll);
    registry.register("libkernel", "sceKernelWaitEventFlag", hle_wait);
    registry.register("libkernel", "sceKernelCancelEventFlag", hle_cancel);
}

/// Attributes: queue mode ∈ {0, FIFO=1, PRIO=2}, thread mode ∈ {0, SINGLE=0x10,
/// MULTI=0x20}, no other bits.
fn valid_attributes(attr: u32) -> bool {
    let queue = attr & 0x0F;
    let thread = attr & 0xF0;
    matches!(queue, 0..=2) && matches!(thread, 0 | 0x10 | 0x20) && (attr & !0x33) == 0
}

/// Wait mode: condition ∈ {AND, OR}, clear ∈ {0, ALL, PATTERN}, no other bits.
fn valid_wait_mode(mode: u64) -> bool {
    let cond = mode & 0x0F;
    let clear = mode & 0xF0;
    (cond == WAIT_AND || cond == WAIT_OR)
        && matches!(clear, 0 | CLEAR_ALL | CLEAR_PATTERN)
        && (mode & !0x33) == 0
}

/// Whether `bits` satisfies `pattern` under the wait condition.
fn is_satisfied(bits: u64, pattern: u64, mode: u64) -> bool {
    if mode & 0x0F == WAIT_AND {
        bits & pattern == pattern
    } else {
        bits & pattern != 0
    }
}

/// `sceKernelCreateEventFlag(out, name, attr, initialPattern, opt)`.
fn hle_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    let name_ptr = args.get(1).copied().unwrap_or(0);
    let attr = args.get(2).copied().unwrap_or(0) as u32;
    let initial = args.get(3).copied().unwrap_or(0);
    let opt = args.get(4).copied().unwrap_or(0);

    if out == 0 || name_ptr == 0 || opt != 0 || !valid_attributes(attr) {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    // Read the name (bounded); a name longer than 31 bytes is invalid.
    let mut name_buf = [0u8; MAX_NAME_LEN + 2];
    if !ctx.mem.read(name_ptr, &mut name_buf) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if !name_buf.contains(&0) {
        return SCE_KERNEL_ERROR_EINVAL; // no NUL within 32 bytes → too long
    }

    let Some(handle) = ctx.services.create_event(attr, initial) else {
        return SCE_KERNEL_ERROR_ENOMEM;
    };
    if !ctx.mem.write(out, &handle.to_le_bytes()) {
        ctx.services.delete_event(handle);
        return SCE_KERNEL_ERROR_EFAULT;
    }
    // Log the NAME, not just the handle. Orbis event flags are named by the
    // subsystem that owns them, so when a guest thread parks forever on a flag
    // nothing sets, the name is what identifies which subsystem was supposed to
    // signal it — the handle alone says nothing.
    let name = name_buf
        .split(|&b| b == 0)
        .next()
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .unwrap_or_default();
    tracing::info!(
        "sceKernelCreateEventFlag {name:?} -> handle {handle:#x} attr {attr:#x} bits {initial:#x}"
    );
    OK
}

/// `sceKernelDeleteEventFlag(handle)`.
fn hle_delete(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    if !ctx.services.delete_event(handle) {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    ctx.services.wake(
        WaitKey {
            class: "event-flag",
            object: handle,
            guest_thread: ctx.guest_threads.current_thread(),
        },
        WakeReason::Deleted,
    );
    OK
}

/// `sceKernelSetEventFlag(handle, pattern)`: OR `pattern` into the bits, then
/// wake every flag waiter so a blocked `WaitEventFlag` re-checks.
fn hle_set(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let pattern = args.get(1).copied().unwrap_or(0);
    if ctx
        .services
        .update_event(handle, EventUpdate::Set(pattern))
        .is_none()
    {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    ctx.services.wake(
        WaitKey {
            class: "event-flag",
            object: handle,
            guest_thread: ctx.guest_threads.current_thread(),
        },
        WakeReason::Set,
    );
    OK
}

/// `sceKernelClearEventFlag(handle, pattern)`: `bits &= pattern` (Orbis
/// semantics — keeps only the bits in `pattern`; pass `~mask` to clear `mask`).
fn hle_clear(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let pattern = args.get(1).copied().unwrap_or(0);
    if ctx
        .services
        .update_event(handle, EventUpdate::Keep(pattern))
        .is_none()
    {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    ctx.services.wake(
        WaitKey {
            class: "event-flag",
            object: handle,
            guest_thread: ctx.guest_threads.current_thread(),
        },
        WakeReason::Clear,
    );
    OK
}

/// Apply the wait's clear mode to the flag after a satisfied poll/wait.
fn apply_clear(bits: &mut u64, pattern: u64, mode: u64) {
    match mode & 0xF0 {
        CLEAR_ALL => *bits = 0,
        CLEAR_PATTERN => *bits &= !pattern,
        _ => {}
    }
}

/// Shared body for Poll (never blocks) and Wait (completes if already
/// satisfied). Writes the observed bits to `result_ptr` when non-null.
fn poll_or_wait(ctx: &HleContext, args: &[u64], unsatisfied_err: u64) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let pattern = args.get(1).copied().unwrap_or(0);
    let mode = args.get(2).copied().unwrap_or(0);
    let result_ptr = args.get(3).copied().unwrap_or(0);

    let Some(mut ef) = ctx.kernel.kernel_event_flags.get_mut(&handle) else {
        return SCE_KERNEL_ERROR_ESRCH;
    };
    if pattern == 0 || !valid_wait_mode(mode) {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    // Report the observed bits regardless of satisfaction.
    if result_ptr != 0 && !ctx.mem.write(result_ptr, &ef.bits.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if !is_satisfied(ef.bits, pattern, mode) {
        return unsatisfied_err;
    }
    apply_clear(&mut ef.bits, pattern, mode);
    OK
}

/// `sceKernelPollEventFlag(handle, pattern, waitMode, result)`: non-blocking;
/// `EBUSY` if the pattern isn't currently satisfied.
fn hle_poll(ctx: &HleContext, args: &[u64]) -> u64 {
    poll_or_wait(ctx, args, SCE_KERNEL_ERROR_EBUSY)
}

/// `sceKernelWaitEventFlag(handle, pattern, waitMode, result, timeout)`:
/// block on the shared flag-changed condvar until the pattern is satisfied or
/// the deadline passes. The old poll-and-instant-ETIMEDOUT was built for the
/// single-active-execution model; with real guest threads a producer CAN set
/// the bits — returning instantly made five Dragon Ball threads hot-spin at
/// 100% CPU and starve the producer (measured via RAEEN_STALL_DUMP).
fn hle_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let pattern = args.get(1).copied().unwrap_or(0);
    let mode = args.get(2).copied().unwrap_or(0);
    let result_ptr = args.get(3).copied().unwrap_or(0);
    let timeout_ptr = args.get(4).copied().unwrap_or(0);

    if pattern == 0 || !valid_wait_mode(mode) {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    if !ctx.kernel.kernel_event_flags.contains_key(&handle) {
        return SCE_KERNEL_ERROR_ESRCH;
    }

    // Timeout is `SceKernelUseconds*` (NULL = wait forever). A NULL timeout
    // waits forever and NEVER synthesizes ETIMEDOUT; a finite one returns
    // ETIMEDOUT only once its real deadline passes. `wait_until` is called in
    // bounded 50ms slices so process teardown is noticed promptly and an
    // infinite wait stays escapable — but the guest's timeout is honored by
    // LOOPING those slices to the real deadline, exactly like `sceKernelWaitSema`.
    //
    // The prior code passed a single `min(us, 50ms)` slice straight to
    // `wait_until` and mapped its TimedOut to ETIMEDOUT, so ANY wait longer than
    // 50ms — a forever wait, or a finite `>50ms` one — returned a spurious
    // ETIMEDOUT after 50ms. A worker waiting on a producer slower than one slice
    // then proceeded as if the wait had failed, reading not-yet-ready state.
    let deadline = if timeout_ptr == 0 {
        None
    } else {
        // `SceKernelUseconds` is a `uint32_t`; read 4 bytes, not 8. Reading 8
        // pulled the adjacent guest word into the high 32 bits, ballooning a
        // bounded wait into a near-unbounded one, and spuriously faulted a valid
        // 4-byte timeout sitting at a page boundary. Matches the sibling readers
        // in `kernel_semaphore.rs` and `kernel_equeue.rs`.
        let mut raw = [0u8; 4];
        if !ctx.mem.read(timeout_ptr, &mut raw) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        Some(
            std::time::Instant::now()
                + std::time::Duration::from_micros(u64::from(u32::from_le_bytes(raw))),
        )
    };
    let write_failed = std::cell::Cell::new(false);
    let deleted = std::cell::Cell::new(false);
    let mut ready = || {
        let Some(mut ef) = ctx.kernel.kernel_event_flags.get_mut(&handle) else {
            deleted.set(true);
            return true;
        };
        if result_ptr != 0 && !ctx.mem.write(result_ptr, &ef.bits.to_le_bytes()) {
            write_failed.set(true);
            return true;
        }
        if is_satisfied(ef.bits, pattern, mode) {
            apply_clear(&mut ef.bits, pattern, mode);
            return true;
        }
        false
    };
    let slice = std::time::Duration::from_millis(50);
    loop {
        // An infinite block must still be escapable so teardown's join doesn't
        // hang on a parked worker (the semaphore/cond waits honor this too).
        if ctx.guest_threads.process_is_terminating() {
            return OK;
        }
        // Shared with the equeue and semaphore waits: a slice expiry is never
        // the guest's timeout, and a finite deadline nearer than the slice wins.
        let now = std::time::Instant::now();
        if raeen_core::subsystems::guest_deadline_reached(deadline, now) {
            return SCE_KERNEL_ERROR_ETIMEDOUT;
        }
        let wait = raeen_core::subsystems::park_slice(deadline, now, slice);
        let outcome = ctx.services.wait_until(
            WaitKey {
                class: "event-flag",
                object: handle,
                guest_thread: ctx.guest_threads.current_thread(),
            },
            wait,
            &|| ctx.guest_threads.process_is_terminating(),
            &mut ready,
        );
        if write_failed.get() {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        if deleted.get() {
            return SCE_KERNEL_ERROR_ESRCH;
        }
        if outcome == WaitOutcome::Ready {
            return OK;
        }
        // Slice elapsed unsatisfied: loop again — forever for a NULL timeout,
        // otherwise until the real deadline above trips ETIMEDOUT.
    }
}

/// `sceKernelCancelEventFlag(handle, setPattern, waiterCountOut)`: force the
/// bits to `setPattern`, report 0 waiters, and wake every flag waiter.
fn hle_cancel(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let set_pattern = args.get(1).copied().unwrap_or(0);
    let waiter_out = args.get(2).copied().unwrap_or(0);
    if ctx.services.event_bits(handle).is_none() {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    if waiter_out != 0 && !ctx.mem.write(waiter_out, &0u32.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    ctx.services
        .update_event(handle, EventUpdate::Replace(set_pattern));
    ctx.services.wake(
        WaitKey {
            class: "event-flag",
            object: handle,
            guest_thread: ctx.guest_threads.current_thread(),
        },
        WakeReason::Cancel,
    );
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
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    /// Create a flag (name "ef" at 0x40) with the given attr/initial bits,
    /// returning its handle.
    fn create(ctx: &HleContext, attr: u64, initial: u64) -> u64 {
        assert!(ctx.mem.write(0x40, b"ef\0"));
        assert_eq!(hle_create(ctx, &[0x100, 0x40, attr, initial, 0]), OK);
        let mut b = [0u8; 8];
        assert!(ctx.mem.read(0x100, &mut b));
        u64::from_le_bytes(b)
    }

    #[test]
    fn set_clear_and_poll_track_bits() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let h = create(&ctx, 0, 0);
        // Set bits 0b0110.
        assert_eq!(hle_set(&ctx, &[h, 0b0110]), OK);
        assert_eq!(kernel.kernel_event_flags.get(&h).unwrap().bits, 0b0110);
        // Poll AND for 0b0100 → satisfied (no clear).
        assert_eq!(hle_poll(&ctx, &[h, 0b0100, WAIT_AND, 0x108]), OK);
        let mut r = [0u8; 8];
        assert!(mem.read(0x108, &mut r));
        assert_eq!(u64::from_le_bytes(r), 0b0110, "poll reports observed bits");
        // Poll AND for 0b1000 → not present → EBUSY.
        assert_eq!(
            hle_poll(&ctx, &[h, 0b1000, WAIT_AND, 0]),
            SCE_KERNEL_ERROR_EBUSY
        );
        // Clear: bits &= 0b0100 keeps only bit 2.
        assert_eq!(hle_clear(&ctx, &[h, 0b0100]), OK);
        assert_eq!(kernel.kernel_event_flags.get(&h).unwrap().bits, 0b0100);
    }

    #[test]
    fn poll_with_clear_pattern_consumes_bits() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let h = create(&ctx, 0, 0b1111);
        // OR-wait for bit 0 with CLEAR_PATTERN → satisfied, clears bit 0.
        assert_eq!(hle_poll(&ctx, &[h, 0b0001, WAIT_OR | CLEAR_PATTERN, 0]), OK);
        assert_eq!(kernel.kernel_event_flags.get(&h).unwrap().bits, 0b1110);
        // CLEAR_ALL zeroes everything.
        assert_eq!(hle_poll(&ctx, &[h, 0b0010, WAIT_OR | CLEAR_ALL, 0]), OK);
        assert_eq!(kernel.kernel_event_flags.get(&h).unwrap().bits, 0);
    }

    #[test]
    fn wait_completes_when_satisfied_else_times_out() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let h = create(&ctx, 0, 0b0001);
        // Already satisfied → OK immediately.
        assert_eq!(hle_wait(&ctx, &[h, 0b0001, WAIT_AND, 0, 0]), OK);
        // Unsatisfied with a finite one-millisecond SceKernelUseconds timeout
        // expires. A NULL timeout means wait forever and must not be used for
        // this assertion.
        assert!(mem.write(0x120, &1_000u32.to_le_bytes()));
        assert_eq!(
            hle_wait(&ctx, &[h, 0b1000, WAIT_AND, 0, 0x120]),
            SCE_KERNEL_ERROR_ETIMEDOUT
        );
    }

    /// The blocking-wait contract: a waiter must sleep until a producer sets
    /// the bits, not spin-poll — the Dragon Ball hot-spin regression test.
    #[test]
    fn wait_blocks_until_another_thread_sets_the_flag() {
        let kernel = std::sync::Arc::new(raeen_kernel::OrbisKernel::new());
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(kernel.as_ref(), &mem, &alloc);
        let h = create(&ctx, 0, 0);
        let setter = std::thread::spawn({
            let k2 = std::sync::Arc::clone(&kernel);
            move || {
                // Inside the 50 ms forever-slice: a NULL-timeout wait wakes on
                // the flag change instead of timing out or spinning.
                std::thread::sleep(std::time::Duration::from_millis(30));
                // hle_set touches only kernel state (bits + the condvar), so
                // the producer gets its own memory; the shared kernel is Sync.
                let mem2 = crate::TestMemory::new(0x100);
                let alloc2 = crate::TestAllocator::new(0);
                let ctx2 = test_ctx(k2.as_ref(), &mem2, &alloc2);
                assert_eq!(hle_set(&ctx2, &[h, 0b0001]), OK);
            }
        });
        let start = std::time::Instant::now();
        assert_eq!(
            hle_wait(&ctx, &[h, 0b0001, WAIT_AND, 0, 0]),
            OK,
            "the waiter must complete once the producer sets the bit"
        );
        let elapsed = start.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_millis(25),
            "the wait must have BLOCKED for the producer, not spun (took {elapsed:?})"
        );
        setter.join().unwrap();
    }

    /// A NULL (forever) wait must keep waiting past the internal 50 ms slice.
    /// The old single-slice code returned ETIMEDOUT after 50 ms even for a
    /// forever wait, so a producer slower than one slice spuriously "timed out"
    /// the waiter — which then proceeded on not-yet-ready state. Regression:
    /// the producer sets the bit at 80 ms (> one slice).
    #[test]
    fn forever_wait_survives_past_the_internal_slice() {
        let kernel = std::sync::Arc::new(raeen_kernel::OrbisKernel::new());
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(kernel.as_ref(), &mem, &alloc);
        let h = create(&ctx, 0, 0);
        let setter = std::thread::spawn({
            let k2 = std::sync::Arc::clone(&kernel);
            move || {
                std::thread::sleep(std::time::Duration::from_millis(80));
                let mem2 = crate::TestMemory::new(0x100);
                let alloc2 = crate::TestAllocator::new(0);
                let ctx2 = test_ctx(k2.as_ref(), &mem2, &alloc2);
                assert_eq!(hle_set(&ctx2, &[h, 0b0001]), OK);
            }
        });
        let start = std::time::Instant::now();
        assert_eq!(
            hle_wait(&ctx, &[h, 0b0001, WAIT_AND, 0, 0]),
            OK,
            "a forever wait must not time out at the 50 ms slice boundary"
        );
        assert!(
            start.elapsed() >= std::time::Duration::from_millis(75),
            "must have waited for the 80 ms producer, not returned early at 50 ms"
        );
        setter.join().unwrap();
    }

    #[test]
    fn create_validation_and_lifecycle() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // NULL out / name, non-zero opt, or bad attrs → EINVAL.
        assert!(mem.write(0x40, b"ef\0"));
        assert_eq!(
            hle_create(&ctx, &[0, 0x40, 0, 0, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(
            hle_create(&ctx, &[0x100, 0x40, 0, 0, 0x999]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(
            hle_create(&ctx, &[0x100, 0x40, 0x44, 0, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
        // Valid create, then delete; second delete → ESRCH.
        let h = create(&ctx, 0x21, 0); // FIFO | SINGLE
        assert_eq!(hle_delete(&ctx, &[h]), OK);
        assert_eq!(hle_delete(&ctx, &[h]), SCE_KERNEL_ERROR_ESRCH);
        // Ops on an unknown handle → ESRCH.
        assert_eq!(hle_set(&ctx, &[0xDEAD, 1]), SCE_KERNEL_ERROR_ESRCH);
    }

    #[test]
    fn cancel_forces_bits_and_reports_zero_waiters() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let h = create(&ctx, 0, 0b0101);
        assert_eq!(hle_cancel(&ctx, &[h, 0b1010, 0x108]), OK);
        assert_eq!(kernel.kernel_event_flags.get(&h).unwrap().bits, 0b1010);
        let mut w = [0u8; 4];
        assert!(mem.read(0x108, &mut w));
        assert_eq!(
            u32::from_le_bytes(w),
            0,
            "zero waiters under single execution"
        );
    }
}
