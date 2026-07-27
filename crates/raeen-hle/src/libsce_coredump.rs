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
    registry.register(
        "libSceCoredump",
        "sceCoredumpUnregisterCoredumpHandler",
        hle_unregister_handler,
    );
    registry.register(
        "libSceCoredump",
        "sceCoredumpWriteUserData",
        hle_write_user_data,
    );
}

/// `sceCoredumpUnregisterCoredumpHandler()`: clear the recorded handler.
fn hle_unregister_handler(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.kernel.coredump_handler.store(0, Ordering::Relaxed);
    ctx.kernel
        .coredump_handler_context
        .store(0, Ordering::Relaxed);
    OK
}

/// `sceCoredumpWriteUserData(const void *data, size_t size)`: called from
/// inside a title's coredump handler to append its crash annotation to the
/// dump. Raeen produces no Sony-format coredump file, so the data is
/// validated (readable guest range) and accepted without being persisted —
/// the crash path must not fail inside the handler.
fn hle_write_user_data(ctx: &HleContext, args: &[u64]) -> u64 {
    let data = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    if data == 0 {
        return OK; // nothing to write; benign inside a crash handler
    }
    // Probe the first byte only — validating the claimed span byte-by-byte
    // inside a crash path buys nothing, and the data is not persisted.
    let mut probe = [0u8; 1];
    let readable = ctx.mem.read(data, &mut probe);
    tracing::debug!(
        "sceCoredumpWriteUserData(data={data:#x}, size={size}) -> accepted (not persisted; \
         readable={readable})"
    );
    OK
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
