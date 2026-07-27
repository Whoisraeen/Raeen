//! HLE libSceNpUniversalDataSystem — the UDS (Universal Data System) event
//! telemetry lifecycle.
//!
//! A faithful Rust port of SharpEmu's `NpUniversalDataSystemExports` (GPL-2.0).
//! UDS is PSN gameplay-event telemetry. Raeen has no PSN backend, so this is an
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
    // EventProperty family — SharpEmu ports (`EventPropertyObjectSetString` /
    // `SetArray`): the property-object pointer must be non-null and readable,
    // pointer-valued payloads must be readable when non-null, and the call
    // otherwise succeeds without recording anything. Measured: ASTRO.BOT's
    // job-assert telemetry path dies on the unresolved-import stub without
    // these (`SetString` fault confirmed 2026-07-21, siblings called from the
    // same path per the boot missing-NID list).
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyObjectSetString",
        hle_property_set_string,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyObjectSetArray",
        hle_property_set_array,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyArraySetObject",
        hle_property_set_array,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyArraySetString",
        hle_property_set_string,
    );
    // Scalar variants carry their payload by value, not by pointer, so only
    // the property-object pointer is validated (SharpEmu has no reference for
    // these; fail-soft success is all a no-PSN-backend telemetry sink can do).
    for name in [
        "sceNpUniversalDataSystemEventPropertyObjectSetInt32",
        "sceNpUniversalDataSystemEventPropertyObjectSetInt64",
        "sceNpUniversalDataSystemEventPropertyObjectSetUInt32",
        "sceNpUniversalDataSystemEventPropertyObjectSetUInt64",
        "sceNpUniversalDataSystemEventPropertyObjectSetBool",
        // Array-scalar variant (measured GTA V import): the float payload
        // travels in XMM0, so the integer slice carries only the object
        // pointer — the same validation shape as the scalar family.
        "sceNpUniversalDataSystemEventPropertyArraySetFloat32",
    ] {
        registry.register("libSceNpUniversalDataSystem", name, hle_property_set_scalar);
    }
    // `sceNpUniversalDataSystemAbortHandle(handle)`: nothing asynchronous ever
    // runs in this no-backend telemetry sink, so there is no work to abort.
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemAbortHandle",
        |_, _| OK,
    );
    // Lifecycle bookkeeping a no-backend implementation can simply accept.
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemCreateEventPropertyObject",
        |_, _| OK,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemDestroyContext",
        |_, _| OK,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemTerminate",
        |_, _| OK,
    );
}

/// One readable byte at `addr` — SharpEmu's `TryRead(addr, stackalloc[1])`
/// probe.
fn readable(ctx: &HleContext, addr: u64) -> bool {
    ctx.mem.read(addr, &mut [0u8; 1])
}

/// `sceNpUniversalDataSystemEventPropertyObjectSetString(event, obj, value)`:
/// both the property object and the string pointer must be non-null and
/// readable (SharpEmu `NpUniversalDataSystemEventPropertyObjectSetString`).
fn hle_property_set_string(ctx: &HleContext, args: &[u64]) -> u64 {
    let obj = args.get(1).copied().unwrap_or(0);
    let value = args.get(2).copied().unwrap_or(0);
    if obj == 0 || value == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    if readable(ctx, obj) && readable(ctx, value) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceNpUniversalDataSystemEventPropertyObjectSetArray(event, obj, values)`:
/// the property object must be non-null and readable; a non-null array pointer
/// must be readable (SharpEmu `NpUniversalDataSystemEventPropertyObjectSetArray`).
fn hle_property_set_array(ctx: &HleContext, args: &[u64]) -> u64 {
    let obj = args.get(1).copied().unwrap_or(0);
    let values = args.get(2).copied().unwrap_or(0);
    if obj == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    if !readable(ctx, obj) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    if values != 0 && !readable(ctx, values) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// Scalar `EventPropertyObjectSet{Int32,Int64,UInt32,UInt64,Bool}`: the value
/// travels by register, so only the property-object pointer is checked.
fn hle_property_set_scalar(ctx: &HleContext, args: &[u64]) -> u64 {
    let obj = args.get(1).copied().unwrap_or(0);
    if obj == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    if readable(ctx, obj) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
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
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            raeen_kernel::OrbisKernel::new(),
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

    #[test]
    fn property_set_string_validates_both_pointers() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_property_set_string(&ctx, &[1, 0, 0x20]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_property_set_string(&ctx, &[1, 0x10, 0]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(hle_property_set_string(&ctx, &[1, 0x10, 0x20]), OK);
        assert_eq!(
            hle_property_set_string(&ctx, &[1, 0xFFFF_0000, 0x20]),
            SCE_ERROR_MEMORY_FAULT
        );
        assert_eq!(
            hle_property_set_string(&ctx, &[1, 0x10, 0xFFFF_0000]),
            SCE_ERROR_MEMORY_FAULT
        );
    }

    #[test]
    fn property_set_array_allows_null_payload_but_not_bad_payload() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_property_set_array(&ctx, &[1, 0, 0x20]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(hle_property_set_array(&ctx, &[1, 0x10, 0]), OK);
        assert_eq!(hle_property_set_array(&ctx, &[1, 0x10, 0x20]), OK);
        assert_eq!(
            hle_property_set_array(&ctx, &[1, 0x10, 0xFFFF_0000]),
            SCE_ERROR_MEMORY_FAULT
        );
    }

    #[test]
    fn property_set_scalar_checks_only_the_object() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_property_set_scalar(&ctx, &[1, 0, 42]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        // A by-value payload (42) is not dereferenced.
        assert_eq!(hle_property_set_scalar(&ctx, &[1, 0x10, 42]), OK);
        assert_eq!(
            hle_property_set_scalar(&ctx, &[1, 0xFFFF_0000, 42]),
            SCE_ERROR_MEMORY_FAULT
        );
    }

    #[test]
    fn event_property_family_is_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceNpUniversalDataSystemEventPropertyObjectSetString",
            "sceNpUniversalDataSystemEventPropertyObjectSetArray",
            "sceNpUniversalDataSystemEventPropertyArraySetObject",
            "sceNpUniversalDataSystemEventPropertyArraySetString",
            "sceNpUniversalDataSystemEventPropertyObjectSetInt32",
            "sceNpUniversalDataSystemEventPropertyObjectSetInt64",
            "sceNpUniversalDataSystemEventPropertyObjectSetUInt32",
            "sceNpUniversalDataSystemEventPropertyObjectSetUInt64",
            "sceNpUniversalDataSystemEventPropertyObjectSetBool",
            "sceNpUniversalDataSystemCreateEventPropertyObject",
            "sceNpUniversalDataSystemDestroyContext",
            "sceNpUniversalDataSystemTerminate",
        ] {
            assert!(
                registry.is_implemented("libSceNpUniversalDataSystem", name),
                "{name} must be registered"
            );
        }
    }
}
