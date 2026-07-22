//! HLE libSceCoredump — the crash-handler registration handshake.
//!
//! A faithful Rust port of SharpEmu's `sceCoredumpRegisterCoredumpHandler`
//! (from `KernelExports`, GPL-2.0). A title registers a coredump handler
//! (function pointer + user context) that the OS would invoke when the process
//! crashes to let the title write extra crash data. Raeen records the
//! registration but never invokes it — a real invocation would require the
//! runtime's fault path to call back into guest code, which is deferred. The
//! call reports success so a title's crash-reporting init completes.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::Ordering;

const OK: u64 = 0;

/// Register the libSceCoredump functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceCoredump",
        "sceCoredumpRegisterCoredumpHandler",
        hle_register_handler,
    );
}

/// `sceCoredumpRegisterCoredumpHandler(handler, context, ...)`: records the
/// handler pointer + user context. The handler is never invoked (no crash-path
/// callback yet).
fn hle_register_handler(ctx: &HleContext, args: &[u64]) -> u64 {
    ctx.kernel
        .coredump_handler
        .store(args.first().copied().unwrap_or(0), Ordering::Relaxed);
    ctx.kernel
        .coredump_handler_context
        .store(args.get(1).copied().unwrap_or(0), Ordering::Relaxed);
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    #[test]
    fn register_handler_records_pointers() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_register_handler(&ctx, &[0xCAFE, 0xF00D]), OK);
        assert_eq!(kernel.coredump_handler.load(Ordering::Relaxed), 0xCAFE);
        assert_eq!(
            kernel.coredump_handler_context.load(Ordering::Relaxed),
            0xF00D
        );
    }
}
