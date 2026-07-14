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
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
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
