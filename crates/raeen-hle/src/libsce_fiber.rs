//! HLE libSceFiber — the **safely-portable** slice of the fiber API.
//!
//! `sceFiberRun`/`Switch`/`ReturnToThread`/`Initialize` transfer control into a
//! fiber's guest entry, which needs the per-thread guest-execution machinery
//! (the M1-E scheduler). Stubbing those would be the "no-op instead of real
//! execution" anti-pattern (a title's fiber logic would silently never run), so
//! they are deliberately left unimplemented.
//!
//! What *is* portable — and ported here — are the fiber API's pure
//! configuration/profiling calls that don't run any fiber:
//! `sceFiberOptParamInitialize` (stamp an option-param signature) and the
//! `sceFiber{Start,Stop}ContextSizeCheck` profiling toggle (state in
//! `OrbisKernel::fiber_context_size_check`). Ported faithfully from SharpEmu's
//! `FiberExports` (GPL-2.0).

use crate::{HleContext, HleRegistry};
use std::sync::atomic::Ordering;

const OK: u64 = 0;
// libSceFiber error codes.
const FIBER_ERROR_NULL: u64 = 0x8059_0001;
const FIBER_ERROR_ALIGNMENT: u64 = 0x8059_0002;
const FIBER_ERROR_INVALID: u64 = 0x8059_0004;
const FIBER_ERROR_STATE: u64 = 0x8059_0006;

/// Magic written into an initialized `SceFiberOptParam`.
const FIBER_OPT_SIGNATURE: u32 = 0xBB40_E64D;

/// Register the safely-portable libSceFiber functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceFiber",
        "sceFiberOptParamInitialize",
        hle_opt_param_init,
    );
    registry.register(
        "libSceFiber",
        "sceFiberStartContextSizeCheck",
        hle_start_context_size_check,
    );
    registry.register(
        "libSceFiber",
        "sceFiberStopContextSizeCheck",
        hle_stop_context_size_check,
    );

    // Fiber creation/teardown carry real state. The control-transfer calls
    // (Run/Switch/ReturnToThread/GetSelf) are registered so a trampoline exists,
    // but they are INTERCEPTED in the runtime VEH dispatch (`fiber.rs`) — they
    // swap the whole guest CONTEXT to resume another fiber natively, which an
    // ordinary `-> u64` handler cannot express — so the placeholder below is
    // never reached. Names match those the linker reports for ASTRO.BOT.
    registry.register(
        "libSceFiber",
        "_sceFiberInitializeImpl",
        hle_fiber_initialize,
    );
    registry.register("libSceFiber", "sceFiberInitialize", hle_fiber_initialize);
    registry.register("libSceFiber", "sceFiberFinalize", hle_fiber_finalize);
    registry.register("libSceFiber", "sceFiberRun", hle_fiber_transfer_placeholder);
    registry.register(
        "libSceFiber",
        "sceFiberSwitch",
        hle_fiber_transfer_placeholder,
    );
    registry.register(
        "libSceFiber",
        "sceFiberReturnToThread",
        hle_fiber_transfer_placeholder,
    );
    registry.register(
        "libSceFiber",
        "sceFiberGetSelf",
        hle_fiber_transfer_placeholder,
    );
}

/// `_sceFiberInitializeImpl(fiber, name, entry, arg_on_initialize, addr_context,
/// size_context, opt_param, build_ver)` — records the fiber's config into the
/// guest `SceFiber` struct (offsets from shadPS4 `fiber.h`); first-run register
/// setup happens later, at `sceFiberRun` time, in `fiber.rs`.
fn hle_fiber_initialize(ctx: &HleContext, args: &[u64]) -> u64 {
    let fiber = args.first().copied().unwrap_or(0);
    if fiber == 0 {
        return FIBER_ERROR_NULL;
    }
    let name = args.get(1).copied().unwrap_or(0);
    let entry = args.get(2).copied().unwrap_or(0);
    let arg_on_init = args.get(3).copied().unwrap_or(0);
    let addr_context = args.get(4).copied().unwrap_or(0);
    let size_context = args.get(5).copied().unwrap_or(0);
    let w32 = |off: u64, v: u32| {
        let _ = ctx.mem.write(fiber + off, &v.to_le_bytes());
    };
    let w64 = |off: u64, v: u64| {
        let _ = ctx.mem.write(fiber + off, &v.to_le_bytes());
    };
    w32(0x00, 0xdef1_649c); // magic_start
    w32(0x04, 2); // state = Idle
    w64(0x08, entry);
    w64(0x10, arg_on_init);
    w64(0x18, addr_context);
    w64(0x20, size_context);
    if name != 0 {
        let mut nb = [0u8; 31];
        let _ = ctx.mem.read(name, &mut nb);
        let mut buf = [0u8; 32];
        buf[..31].copy_from_slice(&nb);
        let _ = ctx.mem.write(fiber + 0x28, &buf);
    }
    w32(0x50, 0); // flags (SetFpuRegs is applied by fiber.rs first-run seed)
    w32(0x68, 0xb375_92a0); // magic_end
    // Stamp the stack guard at the base of the context buffer (fiber.cpp:212).
    if addr_context != 0 {
        let _ = ctx
            .mem
            .write(addr_context, &0x7149_f2ca_7149_f2cau64.to_le_bytes());
    }
    OK
}

/// `sceFiberFinalize(fiber)` — CAS the guest `state` field Idle(2) → Terminated(3)
/// and drop any suspended snapshot; a non-Idle fiber is a state error.
fn hle_fiber_finalize(ctx: &HleContext, args: &[u64]) -> u64 {
    let fiber = args.first().copied().unwrap_or(0);
    if fiber == 0 {
        return FIBER_ERROR_NULL;
    }
    let mut state = [0u8; 4];
    if ctx.mem.read(fiber + 0x04, &mut state) && u32::from_le_bytes(state) != 2 {
        return FIBER_ERROR_STATE;
    }
    let _ = ctx.mem.write(fiber + 0x04, &3u32.to_le_bytes());
    ctx.kernel.fibers.remove(&fiber);
    OK
}

/// Registered only so the control-transfer NIDs get a trampoline; the real work
/// is `fiber::handle` in the runtime VEH dispatch, which intercepts before this
/// runs. Reaching it means the interception regressed.
fn hle_fiber_transfer_placeholder(_ctx: &HleContext, _args: &[u64]) -> u64 {
    tracing::error!(
        "libSceFiber transfer function reached its HLE handler — it should have been \
         intercepted in dispatch::fiber; the fiber context-swap did not run"
    );
    OK
}

/// `sceFiberOptParamInitialize(optParam)`: stamp the option-param signature.
/// `optParam` must be non-NULL and 8-byte aligned.
fn hle_opt_param_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let opt_param = args.first().copied().unwrap_or(0);
    if opt_param == 0 {
        return FIBER_ERROR_NULL;
    }
    if opt_param & 7 != 0 {
        return FIBER_ERROR_ALIGNMENT;
    }
    if !ctx.mem.write(opt_param, &FIBER_OPT_SIGNATURE.to_le_bytes()) {
        // The real API has no memory-fault path here; a bad pointer is an
        // alignment/argument error.
        return FIBER_ERROR_ALIGNMENT;
    }
    OK
}

/// `sceFiberStartContextSizeCheck(flags)`: begin context-size profiling. `flags`
/// must be 0; starting twice without stopping is a state error.
fn hle_start_context_size_check(ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) != 0 {
        return FIBER_ERROR_INVALID;
    }
    if ctx
        .kernel
        .fiber_context_size_check
        .swap(1, Ordering::Relaxed)
        == 0
    {
        OK
    } else {
        FIBER_ERROR_STATE
    }
}

/// `sceFiberStopContextSizeCheck()`: end context-size profiling. Stopping when
/// not started is a state error.
fn hle_stop_context_size_check(ctx: &HleContext, _args: &[u64]) -> u64 {
    if ctx
        .kernel
        .fiber_context_size_check
        .swap(0, Ordering::Relaxed)
        == 1
    {
        OK
    } else {
        FIBER_ERROR_STATE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    fn ctx_env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x40);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    #[test]
    fn opt_param_init_stamps_signature_and_validates() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // NULL and misaligned params are rejected.
        assert_eq!(hle_opt_param_init(&ctx, &[0]), FIBER_ERROR_NULL);
        assert_eq!(hle_opt_param_init(&ctx, &[0x14]), FIBER_ERROR_ALIGNMENT);
        // An 8-aligned param gets the signature stamped.
        assert_eq!(hle_opt_param_init(&ctx, &[0x10]), OK);
        let mut b = [0u8; 4];
        assert!(crate::GuestMemory::read(&mem, 0x10, &mut b));
        assert_eq!(u32::from_le_bytes(b), FIBER_OPT_SIGNATURE);
    }

    #[test]
    fn context_size_check_toggles_with_state_errors() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Non-zero flags → invalid.
        assert_eq!(
            hle_start_context_size_check(&ctx, &[1]),
            FIBER_ERROR_INVALID
        );
        // Start, then a second start is a state error; stop, then a second stop.
        assert_eq!(hle_start_context_size_check(&ctx, &[0]), OK);
        assert_eq!(hle_start_context_size_check(&ctx, &[0]), FIBER_ERROR_STATE);
        assert_eq!(hle_stop_context_size_check(&ctx, &[]), OK);
        assert_eq!(hle_stop_context_size_check(&ctx, &[]), FIBER_ERROR_STATE);
    }
}
