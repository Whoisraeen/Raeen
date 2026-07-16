//! HLE libSceNpUniversalDataSystem — the UDS (Universal Data System) event
//! telemetry lifecycle.
//!
//! A faithful Rust port of SharpEmu's `NpUniversalDataSystemExports` (GPL-2.0).
//! UDS is PSN gameplay-event telemetry. XPS5X has no PSN backend, so this is an
//! honest handshake stub: `Initialize` validates and touches its parameter
//! block, `CreateContext`/`CreateHandle` hand back a fixed context id (1) / a
//! monotonic handle, and register/destroy succeed. No event is ever recorded or
//! transmitted.
//!
//! Error codes are ported verbatim from SharpEmu: the lib-specific
//! `0x8055_3102` (invalid argument) and the generic memory-fault mapped to the
//! real Orbis `EFAULT` (`0x8002_000E`), all as plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};

const OK: u64 = 0;
const UDS_ERROR_INVALID_ARGUMENT: u64 = 0x8055_3102;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

// SharpEmu's `_nextHandle`, starting at 1 (incremented before first use).
static NEXT_HANDLE: AtomicI32 = AtomicI32::new(1);

/// Register the libSceNpUniversalDataSystem functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemInitialize",
        hle_initialize,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemCreateContext",
        hle_create_context,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemCreateHandle",
        hle_create_handle,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemRegisterContext",
        |_, _| OK,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemDestroyHandle",
        |_, _| OK,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemCreateEvent",
        hle_create_event,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemDestroyEvent",
        |_, _| OK,
    );
    // Telemetry sink: accepting the post is all a title observes.
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemPostEvent",
        |_, _| OK,
    );
}

/// `sceNpUniversalDataSystemCreateEvent(param*, _, eventOut*, altOut*)` —
/// SharpEmu: null param is invalid-argument; the new event id is written to
/// the first writable of args 2/3. Minecraft's activity thread dies without
/// this ("activityTerminate" events).
fn hle_create_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    if param == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    let event = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    for out in [
        args.get(2).copied().unwrap_or(0),
        args.get(3).copied().unwrap_or(0),
    ] {
        if out != 0 && ctx.mem.write(out, &event.to_le_bytes()) {
            return OK;
        }
    }
    SCE_ERROR_MEMORY_FAULT
}

/// `sceNpUniversalDataSystemInitialize(param *)`: a null param is a
/// lib-specific invalid-argument error; otherwise the 16-byte parameter block
/// is read (validating readability) and the call succeeds.
fn hle_initialize(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    if param == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    let mut buf = [0u8; 16];
    if ctx.mem.read(param, &mut buf) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceNpUniversalDataSystemCreateContext(context *)`: a null out-pointer is a
/// benign success (matching SharpEmu); otherwise the fixed context id `1` is
/// written back.
fn hle_create_context(ctx: &HleContext, args: &[u64]) -> u64 {
    let context = args.first().copied().unwrap_or(0);
    if context == 0 {
        return OK;
    }
    if ctx.mem.write(context, &1i32.to_le_bytes()) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceNpUniversalDataSystemCreateHandle(out0 *, out1 *)`: a fresh handle is
/// written to whichever of the two out-pointers is non-null and writable
/// (SharpEmu tries `Rdi` then `Rsi`, both with a nil check). If neither can be
/// written, a memory fault is returned.
fn hle_create_handle(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed) + 1;
    let out0 = args.first().copied().unwrap_or(0);
    let out1 = args.get(1).copied().unwrap_or(0);
    let bytes = handle.to_le_bytes();
    let wrote =
        (out0 != 0 && ctx.mem.write(out0, &bytes)) || (out1 != 0 && ctx.mem.write(out1, &bytes));
    if wrote { OK } else { SCE_ERROR_MEMORY_FAULT }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            xps5x_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x100),
            crate::TestAllocator::new(0),
        )
    }

    #[test]
    fn initialize_validates_param() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_initialize(&ctx, &[0]), UDS_ERROR_INVALID_ARGUMENT);
        assert_eq!(hle_initialize(&ctx, &[0x10]), OK);
        assert_eq!(hle_initialize(&ctx, &[0xFFFF_0000]), SCE_ERROR_MEMORY_FAULT);
    }

    #[test]
    fn create_context_writes_one_and_allows_null() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_create_context(&ctx, &[0]), OK); // null → benign OK
        assert_eq!(hle_create_context(&ctx, &[0x20]), OK);
        let mut id = [0u8; 4];
        assert!(mem.read(0x20, &mut id));
        assert_eq!(i32::from_le_bytes(id), 1);
    }

    #[test]
    fn create_handle_prefers_first_writable_out() {
        NEXT_HANDLE.store(1, Ordering::Relaxed);
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // First handle is 2 (counter starts at 1, incremented before use).
        assert_eq!(hle_create_handle(&ctx, &[0x30, 0x40]), OK);
        let mut h = [0u8; 4];
        assert!(mem.read(0x30, &mut h));
        assert_eq!(i32::from_le_bytes(h), 2);
        // Rdi null → falls back to Rsi.
        assert_eq!(hle_create_handle(&ctx, &[0, 0x50]), OK);
        assert!(mem.read(0x50, &mut h));
        assert_eq!(i32::from_le_bytes(h), 3);
        // Neither writable → memory fault.
        assert_eq!(
            hle_create_handle(&ctx, &[0xFFFF_0000, 0]),
            SCE_ERROR_MEMORY_FAULT
        );
    }
}
