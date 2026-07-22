//! HLE libSceNpTrophy2 — the PS5 (Gen5) trophy context/handle lifecycle.
//!
//! A faithful Rust port of SharpEmu's `NpTrophy2Exports` (GPL-2.0). A title
//! creates a trophy *context* and *handle* (each a monotonically increasing
//! `int` id written back to a guest out-pointer), registers the context, may
//! register an unlock callback, and shows the trophy list. Raeen has no trophy
//! backend, so every call after id allocation is an honest handshake stub that
//! reports success — the guest gets valid context/handle ids and its trophy
//! bookkeeping proceeds, but no trophy is ever actually unlocked or displayed.
//!
//! SharpEmu's synthetic `OrbisGen2Result` codes are mapped to the real Orbis
//! kernel codes (`EINVAL`/`EFAULT`) as plain zero-extended `u64`, matching the
//! already-ported [`crate::libsce_ampr`] convention.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};

const OK: u64 = 0;
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;
/// SharpEmu's `ORBIS_GEN2_ERROR_NOT_FOUND` maps to the real Orbis
/// `SCE_KERNEL_ERROR_ENOENT` (both `0x8002_0002`).
const SCE_ERROR_NOT_FOUND: u64 = 0x8002_0002;

// SharpEmu's `_nextContext` / `_nextHandle`, both starting at 1.
static NEXT_CONTEXT: AtomicI32 = AtomicI32::new(1);
static NEXT_HANDLE: AtomicI32 = AtomicI32::new(1);

/// Register the libSceNpTrophy2 functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceNpTrophy2",
        "sceNpTrophy2CreateContext",
        hle_create_context,
    );
    registry.register(
        "libSceNpTrophy2",
        "sceNpTrophy2CreateHandle",
        hle_create_handle,
    );
    registry.register("libSceNpTrophy2", "sceNpTrophy2DestroyContext", |_, _| OK);
    registry.register("libSceNpTrophy2", "sceNpTrophy2DestroyHandle", |_, _| OK);
    registry.register("libSceNpTrophy2", "sceNpTrophy2AbortHandle", |_, _| OK);
    registry.register("libSceNpTrophy2", "sceNpTrophy2RegisterContext", |_, _| OK);
    registry.register(
        "libSceNpTrophy2",
        "sceNpTrophy2RegisterUnlockCallback",
        |_, _| OK,
    );
    registry.register(
        "libSceNpTrophy2",
        "sceNpTrophy2UnregisterUnlockCallback",
        |_, _| OK,
    );
    registry.register("libSceNpTrophy2", "sceNpTrophy2ShowTrophyList", |_, _| OK);
    // `sceNpTrophy2GetTrophyInfo(context, handle, trophyId, details*, data*)`:
    // report "no such trophy" rather than succeeding. Succeeding would require
    // filling both `SceNpTrophy2Details` and `SceNpTrophy2Data`, whose exact
    // layouts are not confirmed here — a title trusting zeroed details would read
    // an empty name and grade 0 as real data. NOT_FOUND is a documented outcome
    // callers already handle, so it degrades along a path the game tests.
    // SharpEmu `NpTrophy2Exports.cs` (#450, 0c467e8) returns
    // `ORBIS_GEN2_ERROR_NOT_FOUND`.
    registry.register("libSceNpTrophy2", "sceNpTrophy2GetTrophyInfo", |_, _| {
        SCE_ERROR_NOT_FOUND
    });
}

/// Write the current `next` id (int32, little-endian) to `out_address`, then —
/// only on a successful write — advance the counter. Mirrors SharpEmu's
/// `WriteIdAndReturn` (no id is consumed on a memory fault).
fn write_id_and_return(ctx: &HleContext, out_address: u64, next: &AtomicI32) -> u64 {
    if out_address == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let id = next.load(Ordering::Relaxed);
    if !ctx.mem.write(out_address, &id.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    next.fetch_add(1, Ordering::Relaxed);
    OK
}

/// `sceNpTrophy2CreateContext(context *)`: write a fresh context id.
fn hle_create_context(ctx: &HleContext, args: &[u64]) -> u64 {
    write_id_and_return(ctx, args.first().copied().unwrap_or(0), &NEXT_CONTEXT)
}

/// `sceNpTrophy2CreateHandle(handle *)`: write a fresh handle id.
fn hle_create_handle(ctx: &HleContext, args: &[u64]) -> u64 {
    write_id_and_return(ctx, args.first().copied().unwrap_or(0), &NEXT_HANDLE)
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
    fn create_context_and_handle_write_monotonic_ids() {
        NEXT_CONTEXT.store(1, Ordering::Relaxed);
        NEXT_HANDLE.store(1, Ordering::Relaxed);
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Null out-pointer is rejected without consuming an id.
        assert_eq!(hle_create_context(&ctx, &[0]), SCE_ERROR_INVALID_ARGUMENT);

        assert_eq!(hle_create_context(&ctx, &[0x10]), OK);
        assert_eq!(hle_create_context(&ctx, &[0x20]), OK);
        let mut id = [0u8; 4];
        assert!(mem.read(0x10, &mut id));
        assert_eq!(i32::from_le_bytes(id), 1);
        assert!(mem.read(0x20, &mut id));
        assert_eq!(i32::from_le_bytes(id), 2);

        // Handles use an independent counter, also starting at 1.
        assert_eq!(hle_create_handle(&ctx, &[0x30]), OK);
        assert!(mem.read(0x30, &mut id));
        assert_eq!(i32::from_le_bytes(id), 1);

        // A memory fault does not consume an id.
        assert_eq!(
            hle_create_context(&ctx, &[0xFFFF_0000]),
            SCE_ERROR_MEMORY_FAULT
        );
        assert_eq!(hle_create_context(&ctx, &[0x40]), OK);
        assert!(mem.read(0x40, &mut id));
        assert_eq!(i32::from_le_bytes(id), 3);
    }

    #[test]
    fn get_trophy_info_reports_not_found_without_writing_details() {
        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libSceNpTrophy2", "sceNpTrophy2GetTrophyInfo"));
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // context, handle, trophyId, details*, data* — always NOT_FOUND.
        assert_eq!(
            registry.call(
                &ctx,
                "libSceNpTrophy2",
                "sceNpTrophy2GetTrophyInfo",
                &[1, 1, 0, 0x10, 0x40],
            ),
            Some(SCE_ERROR_NOT_FOUND)
        );
    }
}
