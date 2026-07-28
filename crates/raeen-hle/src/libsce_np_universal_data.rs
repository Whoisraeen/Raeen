//! HLE libSceNpUniversalDataSystem — the UDS (Universal Data System) event
//! lifecycle.
//!
//! Originally a port of SharpEmu's `NpUniversalDataSystemExports` (GPL-2.0);
//! signatures were then corrected against KytyPS5's `LibNpUniversalDataSystem`
//! (`libNet.cpp`, MIT lineage), whose prototypes are field-proven on real PS5
//! titles: property setters are `(object, key, value)` — SharpEmu had the
//! object in the *second* slot — and `sceNpUniversalDataSystemCreateEvent` is
//! `(eventName*, propObject, newEvent**, propPtr**)`.
//!
//! UDS is PSN gameplay-event telemetry, and Raeen has no PSN backend — but
//! UDS is also **the PS5 trophy-unlock path**: `libSceNpTrophy2` exports no
//! `UnlockTrophy` (see [`crate::libsce_np_trophy2`]); titles unlock trophies
//! by posting the reserved system event `_UnlockTrophy` whose property object
//! carries the integer `_trophy_id`. Those two names are the publicly
//! documented UDS trophy protocol from PS5 homebrew RE; they are *not*
//! confirmed by an in-tree license-compatible source yet, so every
//! `PostEvent` logs its event name — the first real unlock session confirms
//! or refutes the binding at a glance. Everything else about the path is
//! real: event names are read from guest memory at `CreateEvent`, integer
//! properties are recorded per property object, and a posted `_UnlockTrophy`
//! persists write-through into the per-title
//! [`raeen_core::trophies::TrophyStore`] next to the save-data host map.
//!
//! No telemetry is ever recorded or transmitted; `PostEvent` returns OK
//! (fire-and-forget, matching KytyPS5) whether or not it unlocked anything —
//! a repeated unlock is a store-level no-op, never a guest-visible error.
//!
//! Error codes: the lib-specific `0x8055_3102` (invalid argument — KytyPS5
//! `NP_UNIVERSAL_DATA_SYSTEM_ERROR_INVALID_ARGUMENT`, same value SharpEmu
//! uses) and the generic memory-fault mapped to the real Orbis `EFAULT`
//! (`0x8002_000E`), all as plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};
use raeen_core::trophies::{TrophyStore, UnlockOutcome};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

const OK: u64 = 0;
const UDS_ERROR_INVALID_ARGUMENT: u64 = 0x8055_3102;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

/// Reserved UDS event name that unlocks a trophy, and the property key that
/// carries the trophy id. Publicly documented PS5 homebrew RE — unverified
/// in-tree; every PostEvent logs its name so a live session can confirm.
const UNLOCK_TROPHY_EVENT: &str = "_UnlockTrophy";
const TROPHY_ID_PROPERTY: &str = "_trophy_id";

// SharpEmu's `_nextHandle`, starting at 1 (incremented before first use).
static NEXT_HANDLE: AtomicI32 = AtomicI32::new(1);

/// Opaque guest-visible ids for events / property objects / property arrays.
/// The guest never dereferences them (they are handles it passes back), so
/// any distinctive non-zero value works; starting high keeps them apart from
/// the small context/handle ints in logs.
static NEXT_OBJECT_ID: AtomicU64 = AtomicU64::new(0x0001_0000);

/// One created (not yet destroyed) UDS event.
struct EventRecord {
    /// Event name read from guest memory at `CreateEvent`.
    name: String,
    /// The property object linked at `CreateEvent` (caller-supplied or
    /// freshly allocated) — where `_trophy_id` lands for unlock events.
    prop_object: u64,
}

/// Live UDS state: events by id, and integer/string properties per property
/// object. Property maps are keyed by the raw object value the guest passed,
/// so objects we did not hand out still accumulate properties.
#[derive(Default)]
struct UdsState {
    events: HashMap<u64, EventRecord>,
    int_props: HashMap<u64, HashMap<String, i64>>,
    string_props: HashMap<u64, HashMap<String, String>>,
}

static UDS: LazyLock<Mutex<UdsState>> = LazyLock::new(Mutex::default);

/// Cached per-title trophy store, keyed by its backing path — reloaded when
/// the active title (savedata root) changes.
static TROPHIES: LazyLock<Mutex<Option<TrophyStore>>> = LazyLock::new(Mutex::default);

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
        hle_destroy_event,
    );
    // The PS5 trophy-unlock path (see module docs); every other event is a
    // telemetry sink where accepting the post is all a title observes.
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemPostEvent",
        hle_post_event,
    );
    // EventProperty family — KytyPS5 prototypes `(object, key, value)`.
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyObjectSetString",
        hle_object_set_string,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyObjectSetArray",
        hle_object_set_array,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyArraySetObject",
        hle_array_set_value_ptr,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyArraySetString",
        hle_array_set_value_ptr,
    );
    // Integer scalars carry their payload by value; the value is recorded so
    // a later `_UnlockTrophy` post can find its `_trophy_id`.
    for name in [
        "sceNpUniversalDataSystemEventPropertyObjectSetInt32",
        "sceNpUniversalDataSystemEventPropertyObjectSetInt64",
        "sceNpUniversalDataSystemEventPropertyObjectSetUInt32",
        "sceNpUniversalDataSystemEventPropertyObjectSetUInt64",
        "sceNpUniversalDataSystemEventPropertyObjectSetBool",
    ] {
        registry.register("libSceNpUniversalDataSystem", name, hle_object_set_scalar);
    }
    // Float payloads travel in XMM0, so the integer arg slice carries only
    // (array) — validate the array handle, record nothing (measured GTA V
    // import; KytyPS5 `ArraySetFloat32`).
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemEventPropertyArraySetFloat32",
        |_, args| {
            if args.first().copied().unwrap_or(0) == 0 {
                UDS_ERROR_INVALID_ARGUMENT
            } else {
                OK
            }
        },
    );
    // `sceNpUniversalDataSystemAbortHandle(handle)`: nothing asynchronous ever
    // runs in this no-backend implementation, so there is no work to abort.
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemAbortHandle",
        |_, _| OK,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemCreateEventPropertyObject",
        hle_create_property_object,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemDestroyEventPropertyObject",
        hle_destroy_property_object,
    );
    // Property arrays are inert containers here (KytyPS5 parity): create
    // hands out a fresh opaque id, destroy accepts anything.
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemCreateEventPropertyArray",
        hle_create_property_object,
    );
    registry.register(
        "libSceNpUniversalDataSystem",
        "sceNpUniversalDataSystemDestroyEventPropertyArray",
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

/// Read a guest C string, or `None` when the pointer is unreadable.
fn read_guest_str(ctx: &HleContext, addr: u64) -> Option<String> {
    crate::fmt::read_cstr(ctx.mem, addr).map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

/// Persist a trophy unlock for the running title. The store lives next to
/// the title's save-data host root (`savedata/<title>-trophies.json`) and is
/// cached until the savedata root — i.e. the title — changes.
fn unlock_trophy(ctx: &HleContext, trophy_id: i32) {
    let path = TrophyStore::path_for_savedata_root(&ctx.kernel.filesystem.savedata_root());
    let mut guard = TROPHIES.lock().unwrap();
    if guard.as_ref().is_none_or(|store| store.path() != path) {
        *guard = Some(TrophyStore::load(path));
    }
    let store = guard.as_mut().expect("just installed");
    match store.unlock_now(trophy_id) {
        Ok(UnlockOutcome::NewlyUnlocked) => {
            tracing::info!(
                trophy_id,
                total = store.unlocked_count(),
                store = %store.path().display(),
                "trophy unlocked"
            );
        }
        Ok(UnlockOutcome::AlreadyUnlocked) => {
            tracing::info!(trophy_id, "trophy unlock repeated — already unlocked");
        }
        Err(err) => {
            tracing::warn!(
                trophy_id,
                store = %store.path().display(),
                error = %err,
                "trophy unlock could not be persisted"
            );
        }
    }
}

/// `sceNpUniversalDataSystemCreateEvent(eventName*, propObject, newEvent**,
/// propPtr**)` — KytyPS5 prototype. The event name is read from guest memory
/// and recorded; the linked property object is the caller's when supplied,
/// otherwise freshly allocated; `*propPtr` (when non-null) receives it.
fn hle_create_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let name_ptr = args.first().copied().unwrap_or(0);
    let caller_prop = args.get(1).copied().unwrap_or(0);
    let event_out = args.get(2).copied().unwrap_or(0);
    let prop_out = args.get(3).copied().unwrap_or(0);
    if name_ptr == 0 || event_out == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    let Some(name) = read_guest_str(ctx, name_ptr) else {
        return SCE_ERROR_MEMORY_FAULT;
    };
    let event_id = NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed);
    let prop_object = if caller_prop != 0 {
        caller_prop
    } else {
        NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)
    };
    // Pointer-sized outs: these are `Event**` / `PropertyObject**`.
    if !ctx.mem.write(event_out, &event_id.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    if prop_out != 0 && !ctx.mem.write(prop_out, &prop_object.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    tracing::debug!(event_id, name = %name, prop_object, "UDS event created");
    UDS.lock()
        .unwrap()
        .events
        .insert(event_id, EventRecord { name, prop_object });
    OK
}

/// `sceNpUniversalDataSystemDestroyEvent(event)`: retire the record.
fn hle_destroy_event(_ctx: &HleContext, args: &[u64]) -> u64 {
    let event = args.first().copied().unwrap_or(0);
    UDS.lock().unwrap().events.remove(&event);
    OK
}

/// `sceNpUniversalDataSystemPostEvent(context, handle, event, options)`:
/// fire-and-forget. A posted `_UnlockTrophy` event persists its
/// `_trophy_id` into the local trophy store; everything else is logged so a
/// live session shows exactly which UDS events a title emits.
fn hle_post_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let event = args.get(2).copied().unwrap_or(0);
    let (name, trophy_id) = {
        let uds = UDS.lock().unwrap();
        let Some(record) = uds.events.get(&event) else {
            tracing::debug!(event, "UDS PostEvent on unknown event — accepted");
            return OK;
        };
        let trophy_id = uds
            .int_props
            .get(&record.prop_object)
            .and_then(|props| props.get(TROPHY_ID_PROPERTY))
            .copied();
        (record.name.clone(), trophy_id)
    };
    tracing::info!(event, name = %name, "UDS PostEvent");
    if name == UNLOCK_TROPHY_EVENT {
        match trophy_id {
            Some(id) => unlock_trophy(ctx, id as i32),
            None => tracing::warn!(
                event,
                "UDS {UNLOCK_TROPHY_EVENT} posted without a {TROPHY_ID_PROPERTY} property — \
                 nothing unlocked"
            ),
        }
    }
    OK
}

/// `sceNpUniversalDataSystemEventPropertyObjectSetString(object, key*,
/// value*)`: object/key/value must be non-null (KytyPS5); key and value are
/// read from guest memory and recorded.
fn hle_object_set_string(ctx: &HleContext, args: &[u64]) -> u64 {
    let object = args.first().copied().unwrap_or(0);
    let key_ptr = args.get(1).copied().unwrap_or(0);
    let value_ptr = args.get(2).copied().unwrap_or(0);
    if object == 0 || key_ptr == 0 || value_ptr == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    let (Some(key), Some(value)) = (read_guest_str(ctx, key_ptr), read_guest_str(ctx, value_ptr))
    else {
        return SCE_ERROR_MEMORY_FAULT;
    };
    UDS.lock()
        .unwrap()
        .string_props
        .entry(object)
        .or_default()
        .insert(key, value);
    OK
}

/// Integer scalar setters `(object, key*, value)` — value by register.
/// Recorded as `i64` so `_trophy_id` survives whichever width the title used.
fn hle_object_set_scalar(ctx: &HleContext, args: &[u64]) -> u64 {
    let object = args.first().copied().unwrap_or(0);
    let key_ptr = args.get(1).copied().unwrap_or(0);
    let value = args.get(2).copied().unwrap_or(0);
    if object == 0 || key_ptr == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    let Some(key) = read_guest_str(ctx, key_ptr) else {
        return SCE_ERROR_MEMORY_FAULT;
    };
    UDS.lock()
        .unwrap()
        .int_props
        .entry(object)
        .or_default()
        .insert(key, value as i64);
    OK
}

/// `sceNpUniversalDataSystemEventPropertyObjectSetArray(object, key*, value,
/// valuePtr**)`: object/key non-null; `*valuePtr` (when non-null) receives
/// the caller's array or a fresh one (KytyPS5).
fn hle_object_set_array(ctx: &HleContext, args: &[u64]) -> u64 {
    let object = args.first().copied().unwrap_or(0);
    let key_ptr = args.get(1).copied().unwrap_or(0);
    let value = args.get(2).copied().unwrap_or(0);
    let value_out = args.get(3).copied().unwrap_or(0);
    if object == 0 || key_ptr == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    if !readable(ctx, key_ptr) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    if value_out != 0 {
        let array = if value != 0 {
            value
        } else {
            NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)
        };
        if !ctx.mem.write(value_out, &array.to_le_bytes()) {
            return SCE_ERROR_MEMORY_FAULT;
        }
    }
    OK
}

/// `EventPropertyArraySet{Object,String}(array, value)`: both non-null
/// (KytyPS5); array contents are not modeled (inert container).
fn hle_array_set_value_ptr(_ctx: &HleContext, args: &[u64]) -> u64 {
    let array = args.first().copied().unwrap_or(0);
    let value = args.get(1).copied().unwrap_or(0);
    if array == 0 || value == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    OK
}

/// `sceNpUniversalDataSystemCreateEventPropertyObject(newObject**)` (also
/// serves `CreateEventPropertyArray`): allocate an opaque id and write it
/// back — KytyPS5 writes `*new_object`; the previous always-OK stub left the
/// out-pointer untouched, handing the guest uninitialized garbage.
fn hle_create_property_object(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    if out == 0 {
        return UDS_ERROR_INVALID_ARGUMENT;
    }
    let id = NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed);
    if !ctx.mem.write(out, &id.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceNpUniversalDataSystemDestroyEventPropertyObject(object)`: drop any
/// recorded properties.
fn hle_destroy_property_object(_ctx: &HleContext, args: &[u64]) -> u64 {
    let object = args.first().copied().unwrap_or(0);
    let mut uds = UDS.lock().unwrap();
    uds.int_props.remove(&object);
    uds.string_props.remove(&object);
    OK
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
            crate::TestMemory::new(0x200),
            crate::TestAllocator::new(0),
        )
    }

    /// Read the u64 the HLE wrote at `addr`.
    fn read_u64(mem: &crate::TestMemory, addr: u64) -> u64 {
        let mut raw = [0u8; 8];
        assert!(mem.read(addr, &mut raw));
        u64::from_le_bytes(raw)
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
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_create_handle(&ctx, &[0x30, 0x40]), OK);
        let mut h = [0u8; 4];
        assert!(mem.read(0x30, &mut h));
        let first = i32::from_le_bytes(h);
        assert!(first >= 2, "counter starts at 1, incremented before use");
        // Rdi null → falls back to Rsi.
        assert_eq!(hle_create_handle(&ctx, &[0, 0x50]), OK);
        assert!(mem.read(0x50, &mut h));
        assert_eq!(i32::from_le_bytes(h), first + 1);
        // Neither writable → memory fault.
        assert_eq!(
            hle_create_handle(&ctx, &[0xFFFF_0000, 0]),
            SCE_ERROR_MEMORY_FAULT
        );
    }

    #[test]
    fn create_event_records_the_guest_name_and_writes_both_outs() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x10, b"gameStart\0"));
        // Null name / null event-out are invalid (KytyPS5).
        assert_eq!(
            hle_create_event(&ctx, &[0, 0, 0x40, 0x48]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_create_event(&ctx, &[0x10, 0, 0, 0x48]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        // Unreadable name pointer is a fault.
        assert_eq!(
            hle_create_event(&ctx, &[0xFFFF_0000, 0, 0x40, 0x48]),
            SCE_ERROR_MEMORY_FAULT
        );

        assert_eq!(hle_create_event(&ctx, &[0x10, 0, 0x40, 0x48]), OK);
        let event = read_u64(&mem, 0x40);
        let prop = read_u64(&mem, 0x48);
        assert_ne!(event, 0);
        assert_ne!(prop, 0);
        assert_ne!(event, prop);
        {
            let uds = UDS.lock().unwrap();
            let record = uds.events.get(&event).expect("event recorded");
            assert_eq!(record.name, "gameStart");
            assert_eq!(record.prop_object, prop);
        }
        // A caller-supplied property object is linked, not replaced.
        assert_eq!(hle_create_event(&ctx, &[0x10, 0xBEEF, 0x40, 0x48]), OK);
        let event2 = read_u64(&mem, 0x40);
        assert_eq!(read_u64(&mem, 0x48), 0xBEEF);
        assert_eq!(
            UDS.lock().unwrap().events.get(&event2).unwrap().prop_object,
            0xBEEF
        );
        // Destroy retires the record.
        assert_eq!(hle_destroy_event(&ctx, &[event]), OK);
        assert!(!UDS.lock().unwrap().events.contains_key(&event));
    }

    #[test]
    fn object_set_scalar_records_the_key_and_value() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x10, b"_trophy_id\0"));
        assert_eq!(
            hle_object_set_scalar(&ctx, &[0, 0x10, 7]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_object_set_scalar(&ctx, &[0x7001, 0, 7]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_object_set_scalar(&ctx, &[0x7001, 0xFFFF_0000, 7]),
            SCE_ERROR_MEMORY_FAULT
        );
        assert_eq!(hle_object_set_scalar(&ctx, &[0x7001, 0x10, 7]), OK);
        assert_eq!(
            UDS.lock()
                .unwrap()
                .int_props
                .get(&0x7001)
                .and_then(|p| p.get("_trophy_id"))
                .copied(),
            Some(7)
        );
    }

    #[test]
    fn object_set_string_validates_and_records() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x10, b"mode\0"));
        assert!(mem.write(0x20, b"creative\0"));
        assert_eq!(
            hle_object_set_string(&ctx, &[0x7002, 0, 0x20]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_object_set_string(&ctx, &[0x7002, 0x10, 0]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_object_set_string(&ctx, &[0x7002, 0x10, 0xFFFF_0000]),
            SCE_ERROR_MEMORY_FAULT
        );
        assert_eq!(hle_object_set_string(&ctx, &[0x7002, 0x10, 0x20]), OK);
        assert_eq!(
            UDS.lock()
                .unwrap()
                .string_props
                .get(&0x7002)
                .and_then(|p| p.get("mode"))
                .cloned(),
            Some("creative".to_owned())
        );
    }

    #[test]
    fn create_property_object_writes_a_fresh_id() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_create_property_object(&ctx, &[0]),
            UDS_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_create_property_object(&ctx, &[0xFFFF_0000]),
            SCE_ERROR_MEMORY_FAULT
        );
        assert_eq!(hle_create_property_object(&ctx, &[0x60]), OK);
        assert_ne!(read_u64(&mem, 0x60), 0);
    }

    #[test]
    fn post_unlock_trophy_persists_into_the_store_idempotently() {
        let (kernel, mem, alloc) = env();
        let savedata = std::env::temp_dir()
            .join(format!("raeen-uds-unlock-{}", std::process::id()))
            .join("TestTitle");
        let _ = std::fs::remove_dir_all(savedata.parent().unwrap());
        std::fs::create_dir_all(&savedata).unwrap();
        kernel.filesystem.set_savedata_directory(&savedata);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x10, b"_UnlockTrophy\0"));
        assert!(mem.write(0x20, b"_trophy_id\0"));
        assert_eq!(hle_create_event(&ctx, &[0x10, 0, 0x40, 0x48]), OK);
        let event = read_u64(&mem, 0x40);
        let prop = read_u64(&mem, 0x48);
        assert_eq!(hle_object_set_scalar(&ctx, &[prop, 0x20, 42]), OK);

        // Post twice: first persists, second is a store-level no-op — both
        // return OK (fire-and-forget).
        assert_eq!(hle_post_event(&ctx, &[1, 1, event, 0]), OK);
        assert_eq!(hle_post_event(&ctx, &[1, 1, event, 0]), OK);

        let store_path = TrophyStore::path_for_savedata_root(&savedata);
        assert_eq!(
            store_path,
            savedata.parent().unwrap().join("TestTitle-trophies.json"),
            "store must sit next to, not inside, the save root"
        );
        let store = TrophyStore::load(&store_path);
        assert!(store.is_unlocked(42));
        assert_eq!(store.unlocked_count(), 1);

        // An unlock event without a trophy id unlocks nothing and still OKs.
        assert_eq!(hle_create_event(&ctx, &[0x10, 0, 0x50, 0]), OK);
        let bare_event = read_u64(&mem, 0x50);
        assert_eq!(hle_post_event(&ctx, &[1, 1, bare_event, 0]), OK);
        assert_eq!(TrophyStore::load(&store_path).unlocked_count(), 1);

        // A non-trophy event never touches the store.
        assert!(mem.write(0x30, b"gameEnd\0"));
        assert_eq!(hle_create_event(&ctx, &[0x30, 0, 0x50, 0]), OK);
        let other = read_u64(&mem, 0x50);
        assert_eq!(hle_post_event(&ctx, &[1, 1, other, 0]), OK);
        assert_eq!(TrophyStore::load(&store_path).unlocked_count(), 1);

        let _ = std::fs::remove_dir_all(savedata.parent().unwrap());
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
            "sceNpUniversalDataSystemDestroyEventPropertyObject",
            "sceNpUniversalDataSystemCreateEventPropertyArray",
            "sceNpUniversalDataSystemDestroyEventPropertyArray",
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
