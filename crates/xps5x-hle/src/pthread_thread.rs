//! HLE libkernel pthread **thread identity & control** (`scePthreadSelf`,
//! `Equal`, `Getthreadid`, `Yield`, `Rename`).
//!
//! The small, stateless pthread calls a title makes constantly: "who am I",
//! "are these two handles the same thread", "yield the CPU", "name this
//! thread". Under XPS5X's single-active-execution model there is exactly one
//! guest thread, so these are **complete and exactly correct** — `Self` and
//! `Getthreadid` return that one handle, `Equal` compares, `Yield` has nothing
//! to switch to, and `Rename` is accepted. Cross-checked against SharpEmu's
//! `KernelPthreadCompatExports` (GPL-2.0). Real multi-threading (distinct
//! handles per thread) arrives with the M1-E scheduler.

use crate::{HleContext, HleRegistry};
use tracing::info;

/// `SCE_OK`.
const OK: u64 = 0;

/// The single active guest thread's handle / unique id.
#[cfg(test)]
const CURRENT_THREAD: u64 = 1;

/// Register the pthread thread-identity/control HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "scePthreadSelf", hle_self);
    registry.register("libkernel", "scePthreadEqual", hle_equal);
    registry.register("libkernel", "scePthreadGetthreadid", hle_getthreadid);
    registry.register("libkernel", "scePthreadYield", hle_yield);
    registry.register("libkernel", "scePthreadRename", hle_rename);
    // POSIX / GNU-extension spellings libKernel also exports under distinct NIDs.
    // Shipped middleware compiled against POSIX/GNU headers links these, not the
    // `scePthread*` forms — same one-active-thread semantics. Ported from
    // SharpEmu `KernelPthreadCompatExports.cs`: `pthread_yield` (#426, 5d7d8e0,
    // NID B5GmVDKwpn0 → `PthreadYield`) and `pthread_rename_np` (#450, 0c467e8,
    // NID 9vyP6Z7bqzc → `PthreadRename`), each delegating to the `sce*` body.
    registry.register("libkernel", "pthread_yield", hle_yield);
    registry.register("libkernel", "pthread_rename_np", hle_rename);
    // Real priority bookkeeping: Setprio records, Getprio reads it back
    // (default: `PthreadAttr::default().sched_priority`). Supersedes the old
    // libkernel `hle_ok_stub` for Setprio, which silently dropped the value so
    // a later Getprio had nothing truthful to report.
    registry.register("libkernel", "scePthreadSetprio", hle_setprio);
    registry.register("libkernel", "scePthreadGetprio", hle_getprio);

    // POSIX spellings — different NIDs, same semantics, and these are the ones
    // a real title imports (from `libScePosix`). `pthread_self` returns the
    // thread handle and `pthread_equal` a boolean, exactly as the `sce*` forms
    // do; `sched_yield` returns 0 like `scePthreadYield`.
    //
    // `pthread_create`/`pthread_join` are deliberately NOT registered — there
    // is no second guest execution context yet, and answering them with a fake
    // success would turn a loud, self-naming fault into a silent livelock. See
    // `pthread_sync::register_posix` for the full reasoning.
    registry.register("libScePosix", "pthread_self", hle_self);
    registry.register("libScePosix", "pthread_equal", hle_equal);
    registry.register("libScePosix", "sched_yield", hle_yield);
    registry.register(
        "libScePosix",
        "sched_get_priority_max",
        hle_sched_get_priority_max,
    );
    registry.register(
        "libScePosix",
        "sched_get_priority_min",
        hle_sched_get_priority_min,
    );
}

/// Orbis thread-priority range: numerically the kernel accepts 256 (highest
/// urgency) through 767 (lowest, the FreeBSD-derived rtprio span the default
/// `PthreadAttr` priority of 700 sits inside). POSIX defines "max" as the
/// numerically largest schedulable value, so `sched_get_priority_max` reports
/// 767 and `min` 256 — matching shadPS4's POSIX sched shims.
const ORBIS_SCHED_PRIORITY_MAX: u64 = 767;
const ORBIS_SCHED_PRIORITY_MIN: u64 = 256;

/// `sched_get_priority_max(policy)`.
fn hle_sched_get_priority_max(_ctx: &HleContext, _args: &[u64]) -> u64 {
    ORBIS_SCHED_PRIORITY_MAX
}

/// `sched_get_priority_min(policy)`.
fn hle_sched_get_priority_min(_ctx: &HleContext, _args: &[u64]) -> u64 {
    ORBIS_SCHED_PRIORITY_MIN
}

/// `SCE_KERNEL_ERROR_EINVAL` — the `scePthread*` (libkernel) ABI's invalid
/// argument code (`0x8002_0000 | errno`), matching `pthread_sync`'s SCE side.
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;
/// `SCE_KERNEL_ERROR_EFAULT`.
const SCE_KERNEL_ERROR_EFAULT: u64 = 0x8002_000E;

/// Resolve a caller-supplied thread handle: `0` means "the calling thread"
/// (the same convention `scePthreadRename` honors above).
fn resolve_thread(ctx: &HleContext, thread: u64) -> u64 {
    if thread == 0 {
        ctx.guest_threads.current_thread()
    } else {
        thread
    }
}

/// `scePthreadSetprio(thread, prio)`: record the requested Orbis priority for
/// the thread. XPS5X does not map guest priorities onto host scheduling (only
/// contention order could differ, never correctness), but the value must be
/// RECORDED so `scePthreadGetprio` reads back what was set — shadPS4's
/// pthread priority model does the same bookkeeping.
fn hle_setprio(ctx: &HleContext, args: &[u64]) -> u64 {
    let thread = resolve_thread(ctx, args.first().copied().unwrap_or(0));
    let prio = args.get(1).copied().unwrap_or(0) as i32;
    ctx.kernel.thread_priorities.insert(thread, prio);
    OK
}

/// `scePthreadGetprio(thread, int *prio)`: report the priority recorded by
/// `scePthreadSetprio`, or the default attribute priority
/// (`PthreadAttr::default().sched_priority`, inside the 256..=767 rtprio span)
/// when the thread never set one.
fn hle_getprio(ctx: &HleContext, args: &[u64]) -> u64 {
    let thread = resolve_thread(ctx, args.first().copied().unwrap_or(0));
    let prio_out = args.get(1).copied().unwrap_or(0);
    if prio_out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let prio = ctx
        .kernel
        .thread_priorities
        .get(&thread)
        .map(|p| *p)
        .unwrap_or(xps5x_kernel::PthreadAttr::default().sched_priority);
    if !ctx.mem.write(prio_out, &prio.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    OK
}

/// `scePthreadSelf()`: the calling thread's handle (the one guest thread).
fn hle_self(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.guest_threads.current_thread()
}

/// `scePthreadEqual(t1, t2)`: 1 if the two handles are the same thread, else 0.
fn hle_equal(_ctx: &HleContext, args: &[u64]) -> u64 {
    let t1 = args.first().copied().unwrap_or(0);
    let t2 = args.get(1).copied().unwrap_or(0);
    u64::from(t1 == t2)
}

/// `scePthreadGetthreadid()`: the calling thread's unique numeric id.
fn hle_getthreadid(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.guest_threads.current_thread()
}

/// `scePthreadYield()`: hint a reschedule. Native guest threads are already
/// host-scheduled, so no additional scheduler action is required.
fn hle_yield(_ctx: &HleContext, _args: &[u64]) -> u64 {
    OK
}

/// `scePthreadRename(thread, name)`: name the thread (accepted; the name is
/// diagnostic only). Logs the requested name when readable.
fn hle_rename(ctx: &HleContext, args: &[u64]) -> u64 {
    let thread = args.first().copied().unwrap_or(0);
    let name_ptr = args.get(1).copied().unwrap_or(0);
    if name_ptr != 0 {
        let mut buf = [0u8; 32];
        if ctx.mem.read(name_ptr, &mut buf) {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            if let Ok(name) = std::str::from_utf8(&buf[..end]) {
                // Name the TARGET thread. `thread` is the handle
                // `scePthreadCreate` wrote back, and that handle IS the guest
                // thread id (`GuestThreads::create` writes the same value it
                // reports as `guest_thread`), so it needs no translation.
                //
                // Keying on `current_thread()` instead assumed a self-rename.
                // Titles name threads from the SPAWNER: Minecraft's main thread
                // creates "RakThread" and names it, so the old code labelled the
                // MAIN thread "RakThread" and left the real one unnamed —
                // misattributing every thread in a fault or stall report. A zero
                // handle still means "me".
                let target = if thread == 0 {
                    ctx.guest_threads.current_thread()
                } else {
                    thread
                };
                info!(thread = target, name, "guest pthread named");
                ctx.kernel.thread_names.insert(target, name.to_owned());
            }
        }
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    fn ctx_env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    #[test]
    fn self_and_getthreadid_return_the_one_thread_handle() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_self(&ctx, &[]), CURRENT_THREAD);
        assert_eq!(hle_getthreadid(&ctx, &[]), CURRENT_THREAD);
    }

    #[test]
    fn equal_compares_handles() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_equal(&ctx, &[5, 5]), 1, "same handle → equal");
        assert_eq!(hle_equal(&ctx, &[5, 6]), 0, "different handles → not equal");
        // scePthreadSelf() equals itself.
        let me = hle_self(&ctx, &[]);
        assert_eq!(hle_equal(&ctx, &[me, me]), 1);
    }

    #[test]
    fn sched_priority_range_is_the_orbis_rtprio_span() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_sched_get_priority_max(&ctx, &[1]), 767);
        assert_eq!(hle_sched_get_priority_min(&ctx, &[1]), 256);
        // The default PthreadAttr priority must sit inside the reported range.
        let default_priority = u64::try_from(xps5x_kernel::PthreadAttr::default().sched_priority)
            .expect("default priority is positive");
        assert!((256..=767).contains(&default_priority));

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libScePosix", "sched_get_priority_max"));
        assert!(registry.is_implemented("libScePosix", "sched_get_priority_min"));
    }

    #[test]
    fn getprio_reads_back_what_setprio_stored_with_a_sane_default() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Unset: the default attribute priority (700) is reported.
        assert_eq!(hle_getprio(&ctx, &[CURRENT_THREAD, 0x40]), OK);
        let mut buf = [0u8; 4];
        assert!(crate::GuestMemory::read(&mem, 0x40, &mut buf));
        assert_eq!(i32::from_le_bytes(buf), 700);
        // Setprio records; Getprio reads it back (thread 0 = "me").
        assert_eq!(hle_setprio(&ctx, &[0, 512]), OK);
        assert_eq!(hle_getprio(&ctx, &[CURRENT_THREAD, 0x40]), OK);
        assert!(crate::GuestMemory::read(&mem, 0x40, &mut buf));
        assert_eq!(i32::from_le_bytes(buf), 512);
        // NULL out-pointer is EINVAL, not a silent success.
        assert_eq!(
            hle_getprio(&ctx, &[CURRENT_THREAD, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libkernel", "scePthreadSetprio"));
        assert!(registry.is_implemented("libkernel", "scePthreadGetprio"));
    }

    #[test]
    fn yield_and_rename_succeed() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_yield(&ctx, &[]), OK);
        // Rename with a NULL name pointer is still accepted.
        assert_eq!(hle_rename(&ctx, &[CURRENT_THREAD, 0]), OK);
    }

    #[test]
    fn posix_and_gnu_pthread_aliases_are_registered_under_libkernel() {
        let registry = HleRegistry::new();
        for name in [
            "scePthreadYield",
            "scePthreadRename",
            "pthread_yield",
            "pthread_rename_np",
        ] {
            assert!(
                registry.is_implemented("libkernel", name),
                "libkernel::{name} must be registered"
            );
        }
    }
}
