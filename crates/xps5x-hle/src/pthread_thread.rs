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
use tracing::debug;

/// `SCE_OK`.
const OK: u64 = 0;

/// The single active guest thread's handle / unique id.
const CURRENT_THREAD: u64 = 1;

/// Register the pthread thread-identity/control HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "scePthreadSelf", hle_self);
    registry.register("libkernel", "scePthreadEqual", hle_equal);
    registry.register("libkernel", "scePthreadGetthreadid", hle_getthreadid);
    registry.register("libkernel", "scePthreadYield", hle_yield);
    registry.register("libkernel", "scePthreadRename", hle_rename);
}

/// `scePthreadSelf()`: the calling thread's handle (the one guest thread).
fn hle_self(_ctx: &HleContext, _args: &[u64]) -> u64 {
    CURRENT_THREAD
}

/// `scePthreadEqual(t1, t2)`: 1 if the two handles are the same thread, else 0.
fn hle_equal(_ctx: &HleContext, args: &[u64]) -> u64 {
    let t1 = args.first().copied().unwrap_or(0);
    let t2 = args.get(1).copied().unwrap_or(0);
    u64::from(t1 == t2)
}

/// `scePthreadGetthreadid()`: the calling thread's unique numeric id.
fn hle_getthreadid(_ctx: &HleContext, _args: &[u64]) -> u64 {
    CURRENT_THREAD
}

/// `scePthreadYield()`: hint a reschedule. With one guest thread there is
/// nothing to switch to, so this succeeds immediately.
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
                debug!("scePthreadRename(thread={thread:#x}, name={name:?})");
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
    fn yield_and_rename_succeed() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_yield(&ctx, &[]), OK);
        // Rename with a NULL name pointer is still accepted.
        assert_eq!(hle_rename(&ctx, &[CURRENT_THREAD, 0]), OK);
    }
}
