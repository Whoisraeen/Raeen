//! HLE libSceNpTrophy2 — the PS5 (Gen5) trophy context/handle lifecycle.
//!
//! Started as a faithful Rust port of SharpEmu's `NpTrophy2Exports`
//! (GPL-2.0); the lifecycle is now *real*: contexts and handles are tracked
//! in a live table (monotonic int ids written back to guest out-pointers,
//! matching SharpEmu), `RegisterContext` records the registration, and the
//! icon getters follow KytyPS5's `LibNpTrophy2` (MIT lineage) — `*size = 0`
//! plus the one **confirmed** Trophy2 error code, `0x8055_3911`
//! (`NP_TROPHY2_ERROR_ICON_FILE_NOT_FOUND`).
//!
//! ## Where the unlocks actually are
//!
//! The PS5 library exports **no** `sceNpTrophy2UnlockTrophy` and no
//! `sceNpTrophy2GetTrophyUnlockState` — verified against both KytyPS5's
//! `LibNpTrophy2` NID table and the full `sceNpTrophy2*` name inventory
//! (SharpEmu `ps5_names.txt`). On PS5, titles unlock trophies by posting a
//! UDS event (`libSceNpUniversalDataSystem`); see
//! [`crate::libsce_np_universal_data`], which persists unlocks into the local
//! [`raeen_core::trophies::TrophyStore`].
//!
//! ## Why the info getters still refuse
//!
//! `GetGameInfo`/`GetGroupInfo`/`GetTrophyInfo` (+ `*InfoArray`) must fill
//! details structs whose *layouts* are now confirmed (KytyPS5 static-asserts:
//! `NpTrophy2GameDetails` 152 B, `NpTrophy2Details` 1312 B, …) but whose
//! *contents* — trophy names, descriptions, grades, group/total counts —
//! live in the title's encrypted trophy pack, which Raeen cannot parse
//! (permanent no-Sony-keys wall). KytyPS5 fabricates a one-bronze-trophy
//! game titled "Kyty"; that fabrication is deliberately **not** ported.
//! NOT_FOUND is a documented outcome callers already handle, so it degrades
//! along a path the game tests. The local unlock ledger (counts + times)
//! surfaces in the Shell's per-game overlay instead.
//!
//! SharpEmu's synthetic `OrbisGen2Result` codes are mapped to the real Orbis
//! kernel codes (`EINVAL`/`EFAULT`) as plain zero-extended `u64`, matching the
//! already-ported [`crate::libsce_ampr`] convention. Trophy2-specific error
//! values beyond `0x8055_3911` are unconfirmed in any license-compatible
//! source (the PS4 `0x8055_16xx` family from shadPS4 does **not** carry over:
//! Trophy2's icon-not-found offset is `0x11` where PS4's is `0x14`), so no
//! other `0x8055_39xx` value is invented here.

use crate::{HleContext, HleRegistry};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{LazyLock, Mutex};

const OK: u64 = 0;
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;
/// SharpEmu's `ORBIS_GEN2_ERROR_NOT_FOUND` maps to the real Orbis
/// `SCE_KERNEL_ERROR_ENOENT` (both `0x8002_0002`).
const SCE_ERROR_NOT_FOUND: u64 = 0x8002_0002;
/// The one *confirmed* Trophy2-specific error code:
/// `NP_TROPHY2_ERROR_ICON_FILE_NOT_FOUND` (KytyPS5 `libNet.cpp`,
/// `LibNpTrophy2`, `0x80553911`).
const NP_TROPHY2_ERROR_ICON_FILE_NOT_FOUND: u64 = 0x8055_3911;

// SharpEmu's `_nextContext` / `_nextHandle`, both starting at 1.
static NEXT_CONTEXT: AtomicI32 = AtomicI32::new(1);
static NEXT_HANDLE: AtomicI32 = AtomicI32::new(1);

/// Live contexts (id → registered?) and handles. Real bookkeeping so the
/// lifecycle is inspectable and diagnosable; *return-code* policy on unknown
/// ids follows KytyPS5 (tolerant OK, loud log) because the Trophy2-specific
/// error values a strict path would need are unconfirmed — see module docs.
#[derive(Default)]
struct Lifecycle {
    contexts: HashMap<i32, bool>,
    handles: HashSet<i32>,
}

static LIFECYCLE: LazyLock<Mutex<Lifecycle>> = LazyLock::new(Mutex::default);

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
    registry.register(
        "libSceNpTrophy2",
        "sceNpTrophy2DestroyContext",
        hle_destroy_context,
    );
    registry.register(
        "libSceNpTrophy2",
        "sceNpTrophy2DestroyHandle",
        hle_destroy_handle,
    );
    // Abort cancels in-flight async work on the handle; nothing here is ever
    // asynchronous, so the handle stays live (matches KytyPS5).
    registry.register("libSceNpTrophy2", "sceNpTrophy2AbortHandle", |_, _| OK);
    registry.register(
        "libSceNpTrophy2",
        "sceNpTrophy2RegisterContext",
        hle_register_context,
    );
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
    // Details-bearing getters: NOT_FOUND — the definitions live in the
    // encrypted trophy pack (see module docs; never fabricate names/counts).
    // `sceNpTrophy2GetTrophyInfo(context, handle, trophyId, details*, data*)`
    // SharpEmu `NpTrophy2Exports.cs` (#450, 0c467e8) returns
    // `ORBIS_GEN2_ERROR_NOT_FOUND`.
    for name in [
        "sceNpTrophy2GetTrophyInfo",
        // `(context, handle, details*, data*)`
        "sceNpTrophy2GetGameInfo",
        // `(context, handle, groupId, details*, data*)` — KytyPS5 signature.
        "sceNpTrophy2GetGroupInfo",
        // `(context, handle, offset, limit, details[], data[], count*)`.
        "sceNpTrophy2GetGroupInfoArray",
        "sceNpTrophy2GetTrophyInfoArray",
    ] {
        registry.register("libSceNpTrophy2", name, |_, _| SCE_ERROR_NOT_FOUND);
    }
    // Icon getters: honest not-supported along KytyPS5's exact path —
    // `*size = 0`, return the confirmed `ICON_FILE_NOT_FOUND` (0x80553911).
    // `sceNpTrophy2GetGameIcon(context, handle, buffer*, size*)`.
    registry.register("libSceNpTrophy2", "sceNpTrophy2GetGameIcon", |ctx, args| {
        hle_icon_not_found(ctx, args.get(3).copied().unwrap_or(0))
    });
    // `sceNpTrophy2GetGroupIcon(context, handle, groupId, buffer*, size*)`.
    registry.register(
        "libSceNpTrophy2",
        "sceNpTrophy2GetGroupIcon",
        |ctx, args| hle_icon_not_found(ctx, args.get(4).copied().unwrap_or(0)),
    );
    // `sceNpTrophy2GetTrophyIcon(context, handle, trophyId, buffer*, size*)`.
    registry.register(
        "libSceNpTrophy2",
        "sceNpTrophy2GetTrophyIcon",
        |ctx, args| hle_icon_not_found(ctx, args.get(4).copied().unwrap_or(0)),
    );
}

/// Write the current `next` id (int32, little-endian) to `out_address`, then —
/// only on a successful write — advance the counter. Mirrors SharpEmu's
/// `WriteIdAndReturn` (no id is consumed on a memory fault).
fn write_id_and_return(ctx: &HleContext, out_address: u64, next: &AtomicI32) -> Option<i32> {
    if out_address == 0 {
        return None;
    }
    let id = next.load(Ordering::Relaxed);
    if !ctx.mem.write(out_address, &id.to_le_bytes()) {
        return Some(-1); // sentinel: fault (distinguished by caller)
    }
    next.fetch_add(1, Ordering::Relaxed);
    Some(id)
}

/// `sceNpTrophy2CreateContext(context*, userId, serviceLabel, options)`:
/// write a fresh context id and track it live (unregistered).
fn hle_create_context(ctx: &HleContext, args: &[u64]) -> u64 {
    match write_id_and_return(ctx, args.first().copied().unwrap_or(0), &NEXT_CONTEXT) {
        None => SCE_ERROR_INVALID_ARGUMENT,
        Some(-1) => SCE_ERROR_MEMORY_FAULT,
        Some(id) => {
            LIFECYCLE.lock().unwrap().contexts.insert(id, false);
            OK
        }
    }
}

/// `sceNpTrophy2CreateHandle(handle*)`: write a fresh handle id and track it.
fn hle_create_handle(ctx: &HleContext, args: &[u64]) -> u64 {
    match write_id_and_return(ctx, args.first().copied().unwrap_or(0), &NEXT_HANDLE) {
        None => SCE_ERROR_INVALID_ARGUMENT,
        Some(-1) => SCE_ERROR_MEMORY_FAULT,
        Some(id) => {
            LIFECYCLE.lock().unwrap().handles.insert(id);
            OK
        }
    }
}

/// `sceNpTrophy2DestroyContext(context)`: retire the context. Unknown ids
/// log loudly but return OK (KytyPS5 parity — see [`Lifecycle`]).
fn hle_destroy_context(_ctx: &HleContext, args: &[u64]) -> u64 {
    let id = args.first().copied().unwrap_or(0) as i32;
    if LIFECYCLE.lock().unwrap().contexts.remove(&id).is_none() {
        tracing::warn!(context = id, "sceNpTrophy2DestroyContext: unknown context");
    }
    OK
}

/// `sceNpTrophy2DestroyHandle(handle)`: retire the handle (same policy).
fn hle_destroy_handle(_ctx: &HleContext, args: &[u64]) -> u64 {
    let id = args.first().copied().unwrap_or(0) as i32;
    if !LIFECYCLE.lock().unwrap().handles.remove(&id) {
        tracing::warn!(handle = id, "sceNpTrophy2DestroyHandle: unknown handle");
    }
    OK
}

/// `sceNpTrophy2RegisterContext(context, handle, options)`: mark the context
/// registered. Unknown context/handle logs loudly but returns OK
/// (KytyPS5 parity — see [`Lifecycle`]).
fn hle_register_context(_ctx: &HleContext, args: &[u64]) -> u64 {
    let context = args.first().copied().unwrap_or(0) as i32;
    let handle = args.get(1).copied().unwrap_or(0) as i32;
    let mut lifecycle = LIFECYCLE.lock().unwrap();
    if !lifecycle.handles.contains(&handle) {
        tracing::warn!(
            context,
            handle,
            "sceNpTrophy2RegisterContext: unknown handle"
        );
    }
    match lifecycle.contexts.get_mut(&context) {
        Some(registered) => *registered = true,
        None => {
            tracing::warn!(context, "sceNpTrophy2RegisterContext: unknown context");
        }
    }
    OK
}

/// KytyPS5 icon-getter shape: write `*size = 0` (when the out-pointer is
/// non-null and writable) and report the confirmed `ICON_FILE_NOT_FOUND`.
fn hle_icon_not_found(ctx: &HleContext, size_ptr: u64) -> u64 {
    if size_ptr != 0 {
        // `size_t*` — 8 bytes. A fault here is still icon-not-found (the
        // real library reads the file before touching outputs).
        let _ = ctx.mem.write(size_ptr, &0u64.to_le_bytes());
    }
    NP_TROPHY2_ERROR_ICON_FILE_NOT_FOUND
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
    fn lifecycle_tracks_create_register_destroy() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_create_context(&ctx, &[0x50]), OK);
        let mut raw = [0u8; 4];
        assert!(mem.read(0x50, &mut raw));
        let context = i32::from_le_bytes(raw);

        assert_eq!(hle_create_handle(&ctx, &[0x60]), OK);
        assert!(mem.read(0x60, &mut raw));
        let handle = i32::from_le_bytes(raw);

        // Live and unregistered after create.
        assert_eq!(
            LIFECYCLE.lock().unwrap().contexts.get(&context),
            Some(&false)
        );
        assert!(LIFECYCLE.lock().unwrap().handles.contains(&handle));

        // RegisterContext flips the registered bit.
        assert_eq!(
            hle_register_context(&ctx, &[context as u64, handle as u64, 0]),
            OK
        );
        assert_eq!(
            LIFECYCLE.lock().unwrap().contexts.get(&context),
            Some(&true)
        );

        // Destroy retires both; a second destroy logs but still returns OK
        // (KytyPS5 return-code parity).
        assert_eq!(hle_destroy_context(&ctx, &[context as u64]), OK);
        assert_eq!(hle_destroy_handle(&ctx, &[handle as u64]), OK);
        assert!(!LIFECYCLE.lock().unwrap().contexts.contains_key(&context));
        assert!(!LIFECYCLE.lock().unwrap().handles.contains(&handle));
        assert_eq!(hle_destroy_context(&ctx, &[context as u64]), OK);
        assert_eq!(hle_destroy_handle(&ctx, &[handle as u64]), OK);
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

    #[test]
    fn info_and_icon_family_is_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceNpTrophy2GetGameInfo",
            "sceNpTrophy2GetGroupInfo",
            "sceNpTrophy2GetGroupInfoArray",
            "sceNpTrophy2GetTrophyInfoArray",
            "sceNpTrophy2GetGameIcon",
            "sceNpTrophy2GetGroupIcon",
            "sceNpTrophy2GetTrophyIcon",
        ] {
            assert!(
                registry.is_implemented("libSceNpTrophy2", name),
                "{name} must be registered"
            );
        }
    }

    #[test]
    fn icon_getters_zero_size_and_report_icon_not_found() {
        let registry = HleRegistry::new();
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x80, &u64::MAX.to_le_bytes()));
        // GetGameIcon(context, handle, buffer, size*) — size is arg 3.
        assert_eq!(
            registry.call(
                &ctx,
                "libSceNpTrophy2",
                "sceNpTrophy2GetGameIcon",
                &[1, 1, 0, 0x80],
            ),
            Some(NP_TROPHY2_ERROR_ICON_FILE_NOT_FOUND)
        );
        let mut size = [0u8; 8];
        assert!(mem.read(0x80, &mut size));
        assert_eq!(u64::from_le_bytes(size), 0);
        // GetTrophyIcon(context, handle, trophyId, buffer, size*) — arg 4.
        assert!(mem.write(0x90, &u64::MAX.to_le_bytes()));
        assert_eq!(
            registry.call(
                &ctx,
                "libSceNpTrophy2",
                "sceNpTrophy2GetTrophyIcon",
                &[1, 1, 3, 0, 0x90],
            ),
            Some(NP_TROPHY2_ERROR_ICON_FILE_NOT_FOUND)
        );
        assert!(mem.read(0x90, &mut size));
        assert_eq!(u64::from_le_bytes(size), 0);
        // A null size pointer is tolerated.
        assert_eq!(
            registry.call(
                &ctx,
                "libSceNpTrophy2",
                "sceNpTrophy2GetGroupIcon",
                &[1, 1, 0, 0, 0],
            ),
            Some(NP_TROPHY2_ERROR_ICON_FILE_NOT_FOUND)
        );
    }
}
