//! HLE libSceJson / libSceJson2 — `sce::Json` C++ object model.
//!
//! Started as a faithful port of SharpEmu's `Json` exports (GPL-2.0) covering
//! the lifecycle NIDs (`MemAllocator`/`Initializer`/`InitParameter2`, `Value`
//! scalar construction), then extended to the full Object/Array/String/
//! Parser/iterator surface ASTRO.BOT imports from `libSceJson2`.
//!
//! ## Model
//!
//! Every `sce::Json` class has an **opaque C++ layout** we do not know, so the
//! payloads live host-side in maps keyed by the guest object's `this` pointer
//! (see [`JsonValue`]). What makes the aggregate methods more than stubs is
//! that every method returning a *reference* hands back a **stable guest
//! address** that later calls resolve through the same maps:
//!
//! * An **Object entry** is a guest allocation (an "entry block") laid out as
//!   `[String key @ +0x0][Value @ +0x8]`: `operator[]` returns
//!   `entry + 0x8` (a `Value&`), iterator deref returns `entry` (a `Pair&`
//!   whose `.first` resolves via [`STRINGS`] and `.second` via [`VALUES`]).
//!   The `+0x8` offset assumes `sizeof(sce::Json::String) == 8` (SharpEmu's
//!   measured `StringObjectSize`); deref additionally refreshes payload
//!   aliases at `+0x18`/`+0x20` in case the title's headers inlined a bigger
//!   `String` into `Pair`.
//! * An **Array element** is a guest-allocated `Value` anchor.
//! * Scalar getters (`getBoolean`/`getInteger`/`getUInteger`/`getReal`)
//!   return `const T&` per the SDK ABI — a pointer to an 8-byte guest slot we
//!   allocate per value and rewrite on every call (SharpEmu returns
//!   `this + 0x10` into a guessed object mirror; a dedicated slot avoids
//!   betting on `sizeof(Value)`).
//! * `begin`/`end` return `iterator` **by value**; the iterator has an
//!   imported (user-provided) destructor, so it is non-trivially destructible
//!   and the Itanium ABI returns it via a hidden sret pointer: `args[0]` is
//!   the return slot, `args[1]` is `this`. Iterator state lives host-side
//!   keyed by that slot address.
//!
//! `Parser::parse` is real: guest bytes → `serde_json` → host value tree with
//! guest-visible anchors for every child. Error codes (`0x80920101` invalid
//! token, `0x80920105` empty buffer) follow SharpEmu's libSceJson port.
//!
//! C++ ABI conventions throughout: a **constructor/fluent setter returns
//! `this`**, a **destructor returns 0** and frees host state, an sret method
//! returns the hidden pointer in `rax`.

use crate::{HleContext, HleRegistry};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard};
use tracing::{debug, warn};

const OK: u64 = 0;
const SCE_KERNEL_ERROR_ENOMEM: u64 = 0x8002_000c;
const SCE_KERNEL_ERROR_EFAULT: u64 = 0x8002_000e;
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;
/// Parser error codes as used by SharpEmu's libSceJson port (GPL-2.0).
const SCE_JSON_PARSER_ERROR_INVALID_TOKEN: u64 = 0x8092_0101;
const SCE_JSON_PARSER_ERROR_EMPTY_BUFFER: u64 = 0x8092_0105;

/// Refuse to parse documents larger than this — a wild guest length must not
/// become a multi-gigabyte host allocation.
const MAX_PARSE_BYTES: u64 = 64 * 1024 * 1024;

/// Guest bytes reserved for a synthesized `Value` anchor (SharpEmu's measured
/// `sizeof(sce::Json::Value)`; the guest never reads the bytes, but keeping
/// real-object size means nothing else is ever mapped inside it).
const VALUE_ANCHOR_BYTES: u64 = 0x20;
/// Guest bytes reserved for a synthesized `String` anchor.
const STRING_ANCHOR_BYTES: u64 = 0x10;
/// Guest bytes for the scalar-reference slot behind `getInteger` et al.
const SCALAR_SLOT_BYTES: u64 = 8;
/// One Object entry block: `[String @ 0x0][Value @ 0x8]`, padded so the
/// `Pair::second` aliases below stay inside the allocation.
const ENTRY_BLOCK_BYTES: u64 = 0x30;
/// Offset of the entry's `Value` — `sizeof(sce::Json::String)` == 8 per
/// SharpEmu's `StringObjectSize`.
const ENTRY_VALUE_OFFSET: u64 = 0x8;
/// Hedge offsets for `Pair::second` if the title compiled against a larger
/// inline `String` (libc++ `std::string` is 0x18; a padded wrapper 0x20).
/// Iterator deref refreshes read-only payload aliases at these offsets.
const ENTRY_VALUE_ALIAS_OFFSETS: &[u64] = &[0x18, 0x20];
/// `end()` sentinel: resolves to the container's current length.
const ITER_END: u64 = u64::MAX;

/// The `sce::Json` lifecycle methods, by mangled name. `true` = returns `this`
/// (constructors + fluent setters); `false` = returns 0 (destructors).
const RET_THIS: &[&str] = &[
    "_ZN3sce4Json12MemAllocatorC2Ev",   // MemAllocator()
    "_ZN3sce4Json11InitializerC1Ev",    // Initializer()
    "_ZN3sce4Json14InitParameter2C1Ev", // InitParameter2()
    "_ZN3sce4Json14InitParameter212setAllocatorEPNS0_12MemAllocatorEPv", // setAllocator()
    "_ZN3sce4Json14InitParameter217setFileBufferSizeEm", // setFileBufferSize()
];
const RET_ZERO: &[&str] = &[
    "_ZN3sce4Json12MemAllocatorD2Ev", // ~MemAllocator()
    "_ZN3sce4Json11InitializerD1Ev",  // ~Initializer()
];
const INITIALIZE: &[&str] = &[
    "_ZN3sce4Json11Initializer10initializeEPKNS0_13InitParameterE", // initialize(InitParameter*)
    "_ZN3sce4Json11Initializer10initializeEPKNS0_14InitParameter2E", // initialize(InitParameter2*)
];

/// What a guest `sce::Json::Value` holds. Kept **host-side**, keyed by the
/// guest object's `this` pointer.
///
/// A `Value` is an opaque C++ variant whose layout we do not know, so the
/// object's own bytes are left alone and the payload lives here instead. That
/// is what makes these more than stubs: a constructor or `set` that only
/// returned `this` would leave the guest reading uninitialized memory back out
/// of `get`/`serialize`, which is silent corruption. Storing the value means it
/// round-trips.
///
/// Aggregates hold the guest address that keys [`OBJECTS`] / [`ARRAYS`]; that
/// same address doubles as the `const Object&`/`const Array&` the reference-
/// returning getters hand back to the guest.
#[derive(Debug, Clone, Default, PartialEq)]
enum JsonValue {
    #[default]
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Double(f64),
    Str(String),
    /// Guest address keying [`OBJECTS`].
    Object(u64),
    /// Guest address keying [`ARRAYS`].
    Array(u64),
}

/// `sce::Json::ValueType` tag, SDK order (matches SharpEmu's mapping).
fn type_tag(value: &JsonValue) -> u64 {
    match value {
        JsonValue::Null => 0,
        JsonValue::Bool(_) => 1,
        JsonValue::Int(_) => 2,
        JsonValue::UInt(_) => 3,
        JsonValue::Double(_) => 4,
        JsonValue::Str(_) => 5,
        JsonValue::Array(_) => 6,
        JsonValue::Object(_) => 7,
    }
}

/// An `sce::Json::Object`'s members, in insertion order: `(key, entry block)`.
/// The entry block is a guest allocation holding the key `String` at `+0x0`
/// (keys [`STRINGS`]) and the member `Value` at [`ENTRY_VALUE_OFFSET`] (keys
/// [`VALUES`]).
type ObjectEntries = Vec<(String, u64)>;

/// Which container class an iterator walks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IterKind {
    Object,
    Array,
}

/// Host-side `Object::iterator` / `Array::iterator` state, keyed by the guest
/// address of the iterator object (usually a stack slot the compiler passed as
/// the sret pointer to `begin`/`end`).
#[derive(Debug, Clone, Copy)]
struct IterState {
    kind: IterKind,
    container: u64,
    index: u64,
}

/// Live `sce::Json::Value` payloads, by guest `this`.
static VALUES: LazyLock<Mutex<HashMap<u64, JsonValue>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Live `sce::Json::String` contents, by guest `this`. Separate from
/// [`VALUES`] because `String` is its own class in the library, even though a
/// `Value` can hold one.
static STRINGS: LazyLock<Mutex<HashMap<u64, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Live `sce::Json::Object` containers, by guest address (a real guest
/// `Object`'s `this`, or an allocated anchor for an object nested in a value
/// tree).
static OBJECTS: LazyLock<Mutex<HashMap<u64, ObjectEntries>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Live `sce::Json::Array` containers: element `Value` anchors in order.
static ARRAYS: LazyLock<Mutex<HashMap<u64, Vec<u64>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Live iterators, keyed by the iterator object's guest address.
static ITERS: LazyLock<Mutex<HashMap<u64, IterState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Per-value 8-byte guest slot backing the `const T&` scalar getters.
static SCALAR_SLOTS: LazyLock<Mutex<HashMap<u64, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Per-value `String` anchor backing `getString`'s `const String&`.
static STR_ANCHORS: LazyLock<Mutex<HashMap<u64, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Per-value cached null-`Value` anchor for out-of-range / missing-key
/// `operator[]` lookups (the SDK returns a null-value reference there).
static NULL_CHILDREN: LazyLock<Mutex<HashMap<u64, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Per-value cached empty containers for `getObject`/`getArray` on a value of
/// the wrong type.
static FALLBACK_OBJECTS: LazyLock<Mutex<HashMap<u64, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static FALLBACK_ARRAYS: LazyLock<Mutex<HashMap<u64, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Per-`String` guest `c_str()` buffer: `(address, capacity)`.
static CSTR_BUFS: LazyLock<Mutex<HashMap<u64, (u64, u64)>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// `Initializer::setGlobalNullAccessCallBack`: stored for fidelity, never
/// invoked (this HLE degrades wrong-type accesses to defaults instead of
/// calling back into guest code) — same policy as SharpEmu.
static NULL_ACCESS_CALLBACK: AtomicU64 = AtomicU64::new(0);
static NULL_ACCESS_CALLBACK_CTX: AtomicU64 = AtomicU64::new(0);

fn values() -> MutexGuard<'static, HashMap<u64, JsonValue>> {
    VALUES.lock().unwrap_or_else(|p| p.into_inner())
}
fn strings() -> MutexGuard<'static, HashMap<u64, String>> {
    STRINGS.lock().unwrap_or_else(|p| p.into_inner())
}
fn objects() -> MutexGuard<'static, HashMap<u64, ObjectEntries>> {
    OBJECTS.lock().unwrap_or_else(|p| p.into_inner())
}
fn arrays() -> MutexGuard<'static, HashMap<u64, Vec<u64>>> {
    ARRAYS.lock().unwrap_or_else(|p| p.into_inner())
}
fn iters() -> MutexGuard<'static, HashMap<u64, IterState>> {
    ITERS.lock().unwrap_or_else(|p| p.into_inner())
}
fn scalar_slots() -> MutexGuard<'static, HashMap<u64, u64>> {
    SCALAR_SLOTS.lock().unwrap_or_else(|p| p.into_inner())
}
fn str_anchors() -> MutexGuard<'static, HashMap<u64, u64>> {
    STR_ANCHORS.lock().unwrap_or_else(|p| p.into_inner())
}
fn null_children() -> MutexGuard<'static, HashMap<u64, u64>> {
    NULL_CHILDREN.lock().unwrap_or_else(|p| p.into_inner())
}
fn fallback_objects() -> MutexGuard<'static, HashMap<u64, u64>> {
    FALLBACK_OBJECTS.lock().unwrap_or_else(|p| p.into_inner())
}
fn fallback_arrays() -> MutexGuard<'static, HashMap<u64, u64>> {
    FALLBACK_ARRAYS.lock().unwrap_or_else(|p| p.into_inner())
}
fn cstr_bufs() -> MutexGuard<'static, HashMap<u64, (u64, u64)>> {
    CSTR_BUFS.lock().unwrap_or_else(|p| p.into_inner())
}

/// The container a payload owns, if any. Used to avoid freeing a container
/// that the replacing payload still references.
fn container_of(value: &JsonValue) -> Option<(IterKind, u64)> {
    match value {
        JsonValue::Object(c) => Some((IterKind::Object, *c)),
        JsonValue::Array(c) => Some((IterKind::Array, *c)),
        _ => None,
    }
}

/// Read a value payload (default null for untracked addresses — a `Value` the
/// guest built without an out-of-line constructor degrades to null rather than
/// faulting, matching SharpEmu).
fn get_value(addr: u64) -> JsonValue {
    let guard = values();
    guard.get(&addr).cloned().unwrap_or_default()
}

/// Allocate a guest anchor. Address 0 is rejected so an anchor can always be
/// used where a null pointer means "no object".
fn alloc_anchor(ctx: &HleContext, size: u64) -> Option<u64> {
    ctx.alloc.alloc(size, 0x10).filter(|a| *a != 0)
}

/// Allocate a child `Value` anchor holding `payload`.
fn new_child(ctx: &HleContext, payload: JsonValue) -> Option<u64> {
    let anchor = alloc_anchor(ctx, VALUE_ANCHOR_BYTES)?;
    values().insert(anchor, payload);
    Some(anchor)
}

/// Allocate an Object entry block: key `String` at `+0x0`, member `Value` at
/// [`ENTRY_VALUE_OFFSET`].
fn new_entry(ctx: &HleContext, key: &str, payload: JsonValue) -> Option<u64> {
    let entry = alloc_anchor(ctx, ENTRY_BLOCK_BYTES)?;
    strings().insert(entry, key.to_owned());
    values().insert(entry + ENTRY_VALUE_OFFSET, payload);
    Some(entry)
}

/// Store `value` for the object at `this` and return `this` — the C++ ABI
/// return for both a constructor and a fluent setter. Frees the host state of
/// a replaced aggregate payload (unless the new payload still references the
/// same container).
fn store(this: u64, value: JsonValue) -> u64 {
    if this == 0 {
        return 0;
    }
    let new_container = container_of(&value);
    let old = values().insert(this, value);
    if let Some(old) = old
        && container_of(&old) != new_container
    {
        free_payload(&old);
    }
    this
}

/// Free the container (and its whole subtree) an aggregate payload owns.
fn free_payload(value: &JsonValue) {
    match value {
        JsonValue::Object(c) => free_object_container(*c),
        JsonValue::Array(c) => free_array_container(*c),
        _ => {}
    }
}

/// Drop an Object container and every entry subtree under it.
fn free_object_container(container: u64) {
    let entries = objects().remove(&container);
    if let Some(entries) = entries {
        for (_, entry) in entries {
            strings().remove(&entry);
            // The Pair::second aliases are shallow clones sharing the real
            // child's container ids: plain-remove them so the subtree is only
            // freed once, through the real child below.
            for off in ENTRY_VALUE_ALIAS_OFFSETS {
                values().remove(&(entry + off));
            }
            free_value_at(entry + ENTRY_VALUE_OFFSET);
        }
    }
}

/// Drop an Array container and every element subtree under it.
fn free_array_container(container: u64) {
    let elems = arrays().remove(&container);
    if let Some(elems) = elems {
        for anchor in elems {
            free_value_at(anchor);
        }
    }
}

/// Fully drop the value at `addr`: its payload subtree plus every derived
/// anchor cache (scalar slot, string anchor, null child, fallbacks).
fn free_value_at(addr: u64) {
    let old = values().remove(&addr);
    if let Some(old) = old {
        free_payload(&old);
    }
    scalar_slots().remove(&addr);
    let anchor = str_anchors().remove(&addr);
    if let Some(anchor) = anchor {
        strings().remove(&anchor);
        cstr_bufs().remove(&anchor);
    }
    null_children().remove(&addr);
    let fb = fallback_objects().remove(&addr);
    if let Some(c) = fb {
        objects().remove(&c);
    }
    let fb = fallback_arrays().remove(&addr);
    if let Some(c) = fb {
        arrays().remove(&c);
    }
}

/// Deep-copy a payload: aggregates get fresh containers and fresh child
/// anchors (C++ copy semantics — the source may be destroyed afterwards).
/// Guest-allocator exhaustion degrades to null with a loud log.
fn clone_payload(ctx: &HleContext, src: &JsonValue) -> JsonValue {
    match src {
        JsonValue::Object(c) => {
            let entries = {
                let guard = objects();
                guard.get(c).cloned().unwrap_or_default()
            };
            let Some(container) = alloc_anchor(ctx, VALUE_ANCHOR_BYTES) else {
                warn!("sceJson: allocator exhausted cloning an Object; degrading to null");
                return JsonValue::Null;
            };
            let mut out = Vec::with_capacity(entries.len());
            for (key, entry) in entries {
                let child = get_value(entry + ENTRY_VALUE_OFFSET);
                let cloned = clone_payload(ctx, &child);
                match new_entry(ctx, &key, cloned) {
                    Some(new) => out.push((key, new)),
                    None => {
                        warn!("sceJson: allocator exhausted cloning Object member '{key}'");
                        break;
                    }
                }
            }
            objects().insert(container, out);
            JsonValue::Object(container)
        }
        JsonValue::Array(c) => {
            let elems = {
                let guard = arrays();
                guard.get(c).cloned().unwrap_or_default()
            };
            let Some(container) = alloc_anchor(ctx, VALUE_ANCHOR_BYTES) else {
                warn!("sceJson: allocator exhausted cloning an Array; degrading to null");
                return JsonValue::Null;
            };
            let mut out = Vec::with_capacity(elems.len());
            for elem in elems {
                let cloned = clone_payload(ctx, &get_value(elem));
                match new_child(ctx, cloned) {
                    Some(new) => out.push(new),
                    None => {
                        warn!("sceJson: allocator exhausted cloning an Array element");
                        break;
                    }
                }
            }
            arrays().insert(container, out);
            JsonValue::Array(container)
        }
        other => other.clone(),
    }
}

/// Payload → `serde_json` value (for `serialize`/`toString`).
fn to_serde(value: &JsonValue) -> serde_json::Value {
    match value {
        JsonValue::Null => serde_json::Value::Null,
        JsonValue::Bool(b) => serde_json::Value::Bool(*b),
        JsonValue::Int(i) => serde_json::Value::Number((*i).into()),
        JsonValue::UInt(u) => serde_json::Value::Number((*u).into()),
        JsonValue::Double(d) => serde_json::Number::from_f64(*d)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        JsonValue::Str(s) => serde_json::Value::String(s.clone()),
        JsonValue::Object(c) => {
            let entries = {
                let guard = objects();
                guard.get(c).cloned().unwrap_or_default()
            };
            let mut map = serde_json::Map::with_capacity(entries.len());
            for (key, entry) in entries {
                map.insert(key, to_serde(&get_value(entry + ENTRY_VALUE_OFFSET)));
            }
            serde_json::Value::Object(map)
        }
        JsonValue::Array(c) => {
            let elems = {
                let guard = arrays();
                guard.get(c).cloned().unwrap_or_default()
            };
            serde_json::Value::Array(elems.iter().map(|a| to_serde(&get_value(*a))).collect())
        }
    }
}

/// `serde_json` value → payload tree with guest anchors for every child.
/// `None` = guest allocator exhausted.
fn build_payload(ctx: &HleContext, value: &serde_json::Value) -> Option<JsonValue> {
    Some(match value {
        serde_json::Value::Null => JsonValue::Null,
        serde_json::Value::Bool(b) => JsonValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                JsonValue::Int(i)
            } else if let Some(u) = n.as_u64() {
                JsonValue::UInt(u)
            } else {
                JsonValue::Double(n.as_f64()?)
            }
        }
        serde_json::Value::String(s) => JsonValue::Str(s.clone()),
        serde_json::Value::Array(items) => {
            let container = alloc_anchor(ctx, VALUE_ANCHOR_BYTES)?;
            let mut elems = Vec::with_capacity(items.len());
            for item in items {
                let payload = build_payload(ctx, item)?;
                elems.push(new_child(ctx, payload)?);
            }
            arrays().insert(container, elems);
            JsonValue::Array(container)
        }
        serde_json::Value::Object(map) => {
            let container = alloc_anchor(ctx, VALUE_ANCHOR_BYTES)?;
            let mut entries = Vec::with_capacity(map.len());
            for (key, item) in map {
                let payload = build_payload(ctx, item)?;
                entries.push((key.clone(), new_entry(ctx, key, payload)?));
            }
            objects().insert(container, entries);
            JsonValue::Object(container)
        }
    })
}

/// A cached per-value null `Value` anchor — what `operator[]` misses return so
/// the guest still holds a dereferenceable `const Value&`.
fn null_child(ctx: &HleContext, owner: u64) -> u64 {
    let cached = {
        let guard = null_children();
        guard.get(&owner).copied()
    };
    let anchor = match cached {
        Some(anchor) => anchor,
        None => {
            let Some(anchor) = alloc_anchor(ctx, VALUE_ANCHOR_BYTES) else {
                warn!("sceJson: allocator exhausted creating a null-value reference");
                return 0;
            };
            null_children().insert(owner, anchor);
            anchor
        }
    };
    values().insert(anchor, JsonValue::Null);
    anchor
}

/// Get-or-create the member `Value` for `key` in `container`, returning its
/// guest address (`Value&`). Creates the container on first touch.
fn object_member(ctx: &HleContext, container: u64, key: &str) -> u64 {
    let found = {
        let guard = objects();
        guard.get(&container).and_then(|entries| {
            entries
                .iter()
                .find(|(k, _)| k.as_str() == key)
                .map(|(_, e)| *e)
        })
    };
    if let Some(entry) = found {
        return entry + ENTRY_VALUE_OFFSET;
    }
    let Some(entry) = new_entry(ctx, key, JsonValue::Null) else {
        warn!("sceJson: allocator exhausted inserting Object member '{key}'");
        return 0;
    };
    objects()
        .entry(container)
        .or_default()
        .push((key.to_owned(), entry));
    entry + ENTRY_VALUE_OFFSET
}

/// Make the value at `this` hold an Object (converting in place if needed) and
/// return the container address.
fn ensure_object(ctx: &HleContext, this: u64) -> u64 {
    if this == 0 {
        return 0;
    }
    if let JsonValue::Object(c) = get_value(this) {
        return c;
    }
    let Some(container) = alloc_anchor(ctx, VALUE_ANCHOR_BYTES) else {
        warn!("sceJson: allocator exhausted converting a value to Object");
        return 0;
    };
    objects().insert(container, Vec::new());
    store(this, JsonValue::Object(container));
    container
}

/// Make the value at `this` hold an Array and return the container address.
fn ensure_array(ctx: &HleContext, this: u64) -> u64 {
    if this == 0 {
        return 0;
    }
    if let JsonValue::Array(c) = get_value(this) {
        return c;
    }
    let Some(container) = alloc_anchor(ctx, VALUE_ANCHOR_BYTES) else {
        warn!("sceJson: allocator exhausted converting a value to Array");
        return 0;
    };
    arrays().insert(container, Vec::new());
    store(this, JsonValue::Array(container));
    container
}

/// Write `bits` into the value's scalar-reference slot and return the slot
/// address — the `const T&` the scalar getters return.
fn scalar_ref(ctx: &HleContext, this: u64, bits: u64) -> u64 {
    let cached = {
        let guard = scalar_slots();
        guard.get(&this).copied()
    };
    let slot = match cached {
        Some(slot) => slot,
        None => {
            let Some(slot) = ctx.alloc.alloc(SCALAR_SLOT_BYTES, 8).filter(|s| *s != 0) else {
                warn!("sceJson: allocator exhausted creating a scalar-reference slot");
                return 0;
            };
            scalar_slots().insert(this, slot);
            slot
        }
    };
    if !ctx.mem.write(slot, &bits.to_le_bytes()) {
        warn!("sceJson: scalar slot {slot:#x} is not writable guest memory");
    }
    slot
}

// ---------------------------------------------------------------------------
// Value: scalar constructors / setters (SharpEmu-derived)
// ---------------------------------------------------------------------------

/// `Value()` — a default-constructed value is JSON null.
fn hle_value_null(_ctx: &HleContext, args: &[u64]) -> u64 {
    store(args.first().copied().unwrap_or(0), JsonValue::Null)
}

/// `Value(bool)` / `set(bool)`.
fn hle_value_bool(_ctx: &HleContext, args: &[u64]) -> u64 {
    let v = args.get(1).copied().unwrap_or(0) != 0;
    store(args.first().copied().unwrap_or(0), JsonValue::Bool(v))
}

/// `Value(long)` / `set(long)`.
fn hle_value_int(_ctx: &HleContext, args: &[u64]) -> u64 {
    let v = args.get(1).copied().unwrap_or(0) as i64;
    store(args.first().copied().unwrap_or(0), JsonValue::Int(v))
}

/// `Value(unsigned long)` / `set(unsigned long)`.
fn hle_value_uint(_ctx: &HleContext, args: &[u64]) -> u64 {
    let v = args.get(1).copied().unwrap_or(0);
    store(args.first().copied().unwrap_or(0), JsonValue::UInt(v))
}

/// `Value(double)` / `set(double)`. The `double` arrives in **XMM0**, not an
/// integer register, so it comes from the float-argument channel.
fn hle_value_double(ctx: &HleContext, args: &[u64]) -> u64 {
    let v = ctx.float_arg_f64(0);
    store(args.first().copied().unwrap_or(0), JsonValue::Double(v))
}

/// `Value(const char *)` / `set(const char *)`.
fn hle_value_cstr(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.get(1).copied().unwrap_or(0);
    let text = crate::fmt::read_cstr(ctx.mem, ptr)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default();
    store(args.first().copied().unwrap_or(0), JsonValue::Str(text))
}

/// `Value(const Value &)` / `operator=` / `set(const Value &)`: deep-copy the
/// source's payload (aggregates get fresh containers).
fn hle_value_copy(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    if this == src {
        return this; // self-assignment must not free the payload it copies
    }
    let payload = clone_payload(ctx, &get_value(src));
    store(this, payload)
}

/// `~Value()`: drop the payload subtree and every derived anchor.
fn hle_value_dtor(_ctx: &HleContext, args: &[u64]) -> u64 {
    if let Some(this) = args.first().copied() {
        free_value_at(this);
    }
    OK
}

/// `Value::clear()`: back to null (frees an aggregate subtree).
fn hle_value_clear(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    if this != 0 {
        store(this, JsonValue::Null);
    }
    OK
}

/// `Value(ValueType)` / `set(ValueType)`: reset the value to the default of the
/// named type. Tag order per the SDK: null/bool/int/uint/real/string/array/
/// object. Aggregate tags get a fresh empty container; an unrecognised code
/// falls back to null.
fn hle_value_set_type(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let value = match args.get(1).copied().unwrap_or(0) {
        1 => JsonValue::Bool(false),
        2 => JsonValue::Int(0),
        3 => JsonValue::UInt(0),
        4 => JsonValue::Double(0.0),
        5 => JsonValue::Str(String::new()),
        6 => match alloc_anchor(ctx, VALUE_ANCHOR_BYTES) {
            Some(c) => {
                arrays().insert(c, Vec::new());
                JsonValue::Array(c)
            }
            None => JsonValue::Null,
        },
        7 => match alloc_anchor(ctx, VALUE_ANCHOR_BYTES) {
            Some(c) => {
                objects().insert(c, Vec::new());
                JsonValue::Object(c)
            }
            None => JsonValue::Null,
        },
        _ => JsonValue::Null,
    };
    store(this, value)
}

/// `Value(const String &)` / `set(const String &)`: take the text from the
/// `sce::Json::String` the guest passes and hold it as a JSON string.
fn hle_value_from_string(_ctx: &HleContext, args: &[u64]) -> u64 {
    let src = args.get(1).copied().unwrap_or(0);
    let text = {
        let guard = strings();
        guard.get(&src).cloned().unwrap_or_default()
    };
    store(args.first().copied().unwrap_or(0), JsonValue::Str(text))
}

/// `Value(const Object &)` / `set(const Object &)`: deep-copy the container.
fn hle_value_from_object(ctx: &HleContext, args: &[u64]) -> u64 {
    let src = args.get(1).copied().unwrap_or(0);
    let payload = clone_payload(ctx, &JsonValue::Object(src));
    store(args.first().copied().unwrap_or(0), payload)
}

/// `Value(const Array &)` / `set(const Array &)`: deep-copy the container.
fn hle_value_from_array(ctx: &HleContext, args: &[u64]) -> u64 {
    let src = args.get(1).copied().unwrap_or(0);
    let payload = clone_payload(ctx, &JsonValue::Array(src));
    store(args.first().copied().unwrap_or(0), payload)
}

// ---------------------------------------------------------------------------
// Value: getters
// ---------------------------------------------------------------------------

/// `getType() const` → `ValueType` by value.
fn hle_value_get_type(_ctx: &HleContext, args: &[u64]) -> u64 {
    type_tag(&get_value(args.first().copied().unwrap_or(0)))
}

/// `count() const`: number of children of an aggregate, 0 otherwise.
fn hle_value_count(_ctx: &HleContext, args: &[u64]) -> u64 {
    match get_value(args.first().copied().unwrap_or(0)) {
        JsonValue::Object(c) => {
            let guard = objects();
            guard.get(&c).map_or(0, |e| e.len()) as u64
        }
        JsonValue::Array(c) => {
            let guard = arrays();
            guard.get(&c).map_or(0, |e| e.len()) as u64
        }
        _ => 0,
    }
}

/// `getBoolean() const` → `const bool&` (pointer to a guest slot).
fn hle_value_get_boolean(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let b = match get_value(this) {
        JsonValue::Bool(b) => b,
        JsonValue::Int(i) => i != 0,
        JsonValue::UInt(u) => u != 0,
        JsonValue::Double(d) => d != 0.0,
        _ => false,
    };
    scalar_ref(ctx, this, u64::from(b))
}

/// `getInteger() const` → `const int64_t&`.
fn hle_value_get_integer(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let i = match get_value(this) {
        JsonValue::Int(i) => i,
        JsonValue::UInt(u) => u as i64,
        JsonValue::Double(d) => d as i64,
        JsonValue::Bool(b) => i64::from(b),
        _ => 0,
    };
    scalar_ref(ctx, this, i as u64)
}

/// `getUInteger() const` → `const uint64_t&`.
fn hle_value_get_uinteger(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let u = match get_value(this) {
        JsonValue::UInt(u) => u,
        JsonValue::Int(i) => i as u64,
        JsonValue::Double(d) => d as u64,
        JsonValue::Bool(b) => u64::from(b),
        _ => 0,
    };
    scalar_ref(ctx, this, u)
}

/// `getReal() const` → `const double&`.
fn hle_value_get_real(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let d = match get_value(this) {
        JsonValue::Double(d) => d,
        JsonValue::Int(i) => i as f64,
        JsonValue::UInt(u) => u as f64,
        JsonValue::Bool(b) => f64::from(u8::from(b)),
        _ => 0.0,
    };
    scalar_ref(ctx, this, u64::from_le_bytes(d.to_le_bytes()))
}

/// `getString() const` → `const String&`: a stable per-value `String` anchor,
/// its text refreshed on every call.
fn hle_value_get_string(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let text = match get_value(this) {
        JsonValue::Str(s) => s,
        other => {
            debug!(
                "sceJson getString on a non-string value at {this:#x} (type {})",
                type_tag(&other)
            );
            String::new()
        }
    };
    let cached = {
        let guard = str_anchors();
        guard.get(&this).copied()
    };
    let anchor = match cached {
        Some(anchor) => anchor,
        None => {
            let Some(anchor) = alloc_anchor(ctx, STRING_ANCHOR_BYTES) else {
                warn!("sceJson: allocator exhausted creating a getString anchor");
                return 0;
            };
            str_anchors().insert(this, anchor);
            anchor
        }
    };
    strings().insert(anchor, text);
    anchor
}

/// `getObject() const` → `const Object&`. Wrong type degrades to a cached
/// empty container (the SDK would hand out `s_nullobject`).
fn hle_value_get_object(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    if let JsonValue::Object(c) = get_value(this) {
        return c;
    }
    debug!("sceJson getObject on a non-object value at {this:#x}");
    let cached = {
        let guard = fallback_objects();
        guard.get(&this).copied()
    };
    match cached {
        Some(c) => c,
        None => {
            let Some(c) = alloc_anchor(ctx, VALUE_ANCHOR_BYTES) else {
                warn!("sceJson: allocator exhausted creating a null-object reference");
                return 0;
            };
            objects().insert(c, Vec::new());
            fallback_objects().insert(this, c);
            c
        }
    }
}

/// `getArray() const` → `const Array&`.
fn hle_value_get_array(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    if let JsonValue::Array(c) = get_value(this) {
        return c;
    }
    debug!("sceJson getArray on a non-array value at {this:#x}");
    let cached = {
        let guard = fallback_arrays();
        guard.get(&this).copied()
    };
    match cached {
        Some(c) => c,
        None => {
            let Some(c) = alloc_anchor(ctx, VALUE_ANCHOR_BYTES) else {
                warn!("sceJson: allocator exhausted creating a null-array reference");
                return 0;
            };
            arrays().insert(c, Vec::new());
            fallback_arrays().insert(this, c);
            c
        }
    }
}

/// `operator[](const char*) const` → `const Value&` (null-value reference on a
/// missing key or wrong type, per the SDK's null-access model).
fn hle_value_ix_cstr(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let key_ptr = args.get(1).copied().unwrap_or(0);
    let key = crate::fmt::read_cstr(ctx.mem, key_ptr)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default();
    if let JsonValue::Object(c) = get_value(this) {
        let found = {
            let guard = objects();
            guard.get(&c).and_then(|entries| {
                entries
                    .iter()
                    .find(|(k, _)| k.as_str() == key)
                    .map(|(_, e)| *e)
            })
        };
        if let Some(entry) = found {
            return entry + ENTRY_VALUE_OFFSET;
        }
    }
    debug!("sceJson operator[\"{key}\"] miss on value {this:#x}");
    null_child(ctx, this)
}

/// `operator[](size_t) const` / `getValue(size_t) const` → `const Value&`.
fn hle_value_ix_index(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let index = args.get(1).copied().unwrap_or(0);
    if let JsonValue::Array(c) = get_value(this) {
        let found = {
            let guard = arrays();
            guard
                .get(&c)
                .and_then(|elems| elems.get(index as usize).copied())
        };
        if let Some(anchor) = found {
            return anchor;
        }
    }
    debug!("sceJson operator[{index}] miss on value {this:#x}");
    null_child(ctx, this)
}

/// `referValue(const String& key)` → `Value&`: converts to Object as needed
/// and get-or-creates the member.
fn hle_value_refer_value(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let key_ptr = args.get(1).copied().unwrap_or(0);
    let key = {
        let guard = strings();
        guard.get(&key_ptr).cloned()
    }
    .unwrap_or_else(|| {
        debug!("sceJson referValue: key String at {key_ptr:#x} is untracked; using \"\"");
        String::new()
    });
    let container = ensure_object(ctx, this);
    if container == 0 {
        return 0;
    }
    object_member(ctx, container, &key)
}

/// `referObject()` → `Object&` (converts a non-object value in place).
fn hle_value_refer_object(ctx: &HleContext, args: &[u64]) -> u64 {
    ensure_object(ctx, args.first().copied().unwrap_or(0))
}

/// `referArray()` → `Array&`.
fn hle_value_refer_array(ctx: &HleContext, args: &[u64]) -> u64 {
    ensure_array(ctx, args.first().copied().unwrap_or(0))
}

/// `toString(String& out) const`: a string value yields its raw text, anything
/// else its JSON serialization (SharpEmu behavior). Returns 0.
fn hle_value_to_string(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    if out != 0 {
        let payload = get_value(this);
        let text = match &payload {
            JsonValue::Str(s) => s.clone(),
            other => to_serde(other).to_string(),
        };
        strings().insert(out, text);
    }
    OK
}

/// `serialize(String& out)`: always the JSON text. Returns 0.
fn hle_value_serialize(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    if out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    strings().insert(out, to_serde(&get_value(this)).to_string());
    OK
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// `static Parser::parse(Value& out, const char* buf, size_t len)`: real JSON
/// parsing into the host value tree.
fn hle_parser_parse(ctx: &HleContext, args: &[u64]) -> u64 {
    let value_out = args.first().copied().unwrap_or(0);
    let buffer = args.get(1).copied().unwrap_or(0);
    let length = args.get(2).copied().unwrap_or(0);
    if value_out == 0 || buffer == 0 || length == 0 {
        return SCE_JSON_PARSER_ERROR_EMPTY_BUFFER;
    }
    if length > MAX_PARSE_BYTES {
        warn!("sceJson Parser::parse: {length} bytes exceeds the {MAX_PARSE_BYTES} cap");
        return SCE_JSON_PARSER_ERROR_INVALID_TOKEN;
    }
    let mut bytes = vec![0u8; length as usize];
    if !ctx.mem.read(buffer, &mut bytes) {
        warn!("sceJson Parser::parse: buffer {buffer:#x}+{length} is not readable guest memory");
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let parsed: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(parsed) => parsed,
        Err(err) => {
            debug!("sceJson Parser::parse: malformed document ({err})");
            return SCE_JSON_PARSER_ERROR_INVALID_TOKEN;
        }
    };
    let Some(payload) = build_payload(ctx, &parsed) else {
        warn!("sceJson Parser::parse: guest allocator exhausted while building the value tree");
        return SCE_KERNEL_ERROR_ENOMEM;
    };
    store(value_out, payload);
    OK
}

// ---------------------------------------------------------------------------
// Object
// ---------------------------------------------------------------------------

/// `Object()`.
fn hle_object_ctor(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    if this == 0 {
        return 0;
    }
    free_object_container(this); // stale stack-address reuse
    objects().insert(this, Vec::new());
    this
}

/// `Object(const Object&)` / `operator=`: deep-copy every member subtree.
fn hle_object_copy(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    if this == 0 {
        return 0;
    }
    if this == src {
        return this;
    }
    let entries = {
        let guard = objects();
        guard.get(&src).cloned().unwrap_or_default()
    };
    free_object_container(this);
    let mut out = Vec::with_capacity(entries.len());
    for (key, entry) in entries {
        let payload = clone_payload(ctx, &get_value(entry + ENTRY_VALUE_OFFSET));
        match new_entry(ctx, &key, payload) {
            Some(new) => out.push((key, new)),
            None => {
                warn!("sceJson: allocator exhausted copying Object member '{key}'");
                break;
            }
        }
    }
    objects().insert(this, out);
    this
}

/// `~Object()`.
fn hle_object_dtor(_ctx: &HleContext, args: &[u64]) -> u64 {
    if let Some(this) = args.first().copied() {
        free_object_container(this);
    }
    OK
}

/// `Object::clear()`: drop members, keep the container alive.
fn hle_object_clear(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    if this != 0 {
        free_object_container(this);
        objects().insert(this, Vec::new());
    }
    OK
}

/// `Object::operator[](const String& key)` → `Value&` (std::map semantics:
/// inserts a null member for a new key).
fn hle_object_index(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let key_ptr = args.get(1).copied().unwrap_or(0);
    if this == 0 {
        return 0;
    }
    let key = {
        let guard = strings();
        guard.get(&key_ptr).cloned()
    }
    .unwrap_or_else(|| {
        debug!("sceJson Object[]: key String at {key_ptr:#x} is untracked; using \"\"");
        String::new()
    });
    object_member(ctx, this, &key)
}

// ---------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------

/// `Array()`.
fn hle_array_ctor(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    if this == 0 {
        return 0;
    }
    free_array_container(this);
    arrays().insert(this, Vec::new());
    this
}

/// `~Array()`.
fn hle_array_dtor(_ctx: &HleContext, args: &[u64]) -> u64 {
    if let Some(this) = args.first().copied() {
        free_array_container(this);
    }
    OK
}

/// `Array::push_back(const Value&)`: deep-copies the value into a fresh
/// element anchor (C++ copies into the container).
fn hle_array_push_back(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    if this == 0 {
        return 0;
    }
    let payload = clone_payload(ctx, &get_value(src));
    let Some(anchor) = new_child(ctx, payload) else {
        warn!("sceJson: allocator exhausted in Array::push_back");
        return 0;
    };
    arrays().entry(this).or_default().push(anchor);
    OK
}

/// `Array::size() const`.
fn hle_array_size(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let guard = arrays();
    guard.get(&this).map_or(0, |e| e.len()) as u64
}

/// `Array::empty() const`.
fn hle_array_empty(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let empty = {
        let guard = arrays();
        guard.get(&this).is_none_or(|e| e.is_empty())
    };
    u64::from(empty)
}

/// `Array::back() const` → `const Value&`.
fn hle_array_back(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let last = {
        let guard = arrays();
        guard.get(&this).and_then(|e| e.last().copied())
    };
    match last {
        Some(anchor) => anchor,
        None => {
            debug!("sceJson Array::back on an empty array {this:#x}");
            null_child(ctx, this)
        }
    }
}

// ---------------------------------------------------------------------------
// Iterators (Object + Array)
// ---------------------------------------------------------------------------

/// `begin`/`end` common path: the iterator is returned **by value** through a
/// hidden sret pointer (`args[0]`) because it has a user-provided (imported)
/// destructor; `args[1]` is the container `this`. Registers host state under
/// the return slot and returns it in `rax` per the Itanium ABI.
fn make_iter(args: &[u64], kind: IterKind, index: u64) -> u64 {
    let sret = args.first().copied().unwrap_or(0);
    let container = args.get(1).copied().unwrap_or(0);
    if sret == 0 {
        return 0;
    }
    iters().insert(
        sret,
        IterState {
            kind,
            container,
            index,
        },
    );
    sret
}

fn hle_object_begin(_ctx: &HleContext, args: &[u64]) -> u64 {
    make_iter(args, IterKind::Object, 0)
}
fn hle_object_end(_ctx: &HleContext, args: &[u64]) -> u64 {
    make_iter(args, IterKind::Object, ITER_END)
}
fn hle_array_begin(_ctx: &HleContext, args: &[u64]) -> u64 {
    make_iter(args, IterKind::Array, 0)
}
fn hle_array_end(_ctx: &HleContext, args: &[u64]) -> u64 {
    make_iter(args, IterKind::Array, ITER_END)
}

/// Resolve an iterator's position; the `end()` sentinel means "current
/// container length" so `it != end()` keeps working while members are added.
fn iter_pos(state: &IterState) -> u64 {
    if state.index != ITER_END {
        return state.index;
    }
    match state.kind {
        IterKind::Object => {
            let guard = objects();
            guard.get(&state.container).map_or(0, |e| e.len()) as u64
        }
        IterKind::Array => {
            let guard = arrays();
            guard.get(&state.container).map_or(0, |e| e.len()) as u64
        }
    }
}

/// `iterator::operator!=(const iterator&) const` → bool. Untracked iterators
/// compare equal so a guest loop terminates instead of spinning forever.
fn hle_iter_ne(_ctx: &HleContext, args: &[u64]) -> u64 {
    let a = {
        let guard = iters();
        guard.get(&args.first().copied().unwrap_or(0)).copied()
    };
    let b = {
        let guard = iters();
        guard.get(&args.get(1).copied().unwrap_or(0)).copied()
    };
    match (a, b) {
        (Some(a), Some(b)) => u64::from(
            a.kind != b.kind || a.container != b.container || iter_pos(&a) != iter_pos(&b),
        ),
        (None, None) => 0,
        _ => {
            warn!("sceJson iterator!=: comparing an untracked iterator; forcing loop exit");
            0
        }
    }
}

/// `iterator::operator++()` → `iterator&` (`this`).
fn hle_iter_pp(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let mut guard = iters();
    match guard.get_mut(&this) {
        Some(state) if state.index != ITER_END => state.index += 1,
        Some(_) => {}
        None => warn!("sceJson iterator++ on an untracked iterator at {this:#x}"),
    }
    drop(guard);
    this
}

/// `iterator::operator*() const`. For an Array iterator this is the element
/// `Value&`. For an Object iterator it is a `Pair&` — the entry block, whose
/// key `String` sits at `+0x0` and member `Value` at [`ENTRY_VALUE_OFFSET`];
/// payload aliases are refreshed at the hedge offsets in case the title
/// compiled `Pair::second` at a different `sizeof(String)`.
fn hle_iter_deref(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let state = {
        let guard = iters();
        guard.get(&this).copied()
    };
    let Some(state) = state else {
        warn!("sceJson iterator* on an untracked iterator at {this:#x}");
        return 0;
    };
    match state.kind {
        IterKind::Array => {
            let elem = {
                let guard = arrays();
                guard
                    .get(&state.container)
                    .and_then(|e| e.get(state.index as usize).copied())
            };
            match elem {
                Some(anchor) => anchor,
                None => {
                    warn!(
                        "sceJson Array iterator* past the end (index {})",
                        state.index
                    );
                    null_child(ctx, this)
                }
            }
        }
        IterKind::Object => {
            let entry = {
                let guard = objects();
                guard
                    .get(&state.container)
                    .and_then(|e| e.get(state.index as usize).map(|(_, entry)| *entry))
            };
            let Some(entry) = entry else {
                warn!(
                    "sceJson Object iterator* past the end (index {})",
                    state.index
                );
                return new_entry(ctx, "", JsonValue::Null).unwrap_or(0);
            };
            let child = get_value(entry + ENTRY_VALUE_OFFSET);
            let mut guard = values();
            for off in ENTRY_VALUE_ALIAS_OFFSETS {
                guard.insert(entry + off, child.clone());
            }
            drop(guard);
            entry
        }
    }
}

/// `~iterator()`.
fn hle_iter_dtor(_ctx: &HleContext, args: &[u64]) -> u64 {
    if let Some(this) = args.first().copied() {
        iters().remove(&this);
    }
    OK
}

// ---------------------------------------------------------------------------
// String
// ---------------------------------------------------------------------------

/// `String()` / `String(const char *)`: keep the text host-side, keyed by the
/// guest object, and return `this`. Same reasoning as [`JsonValue`] — the
/// class's layout is opaque, so storing the payload here is what makes a later
/// read return what was written.
fn hle_string_ctor(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    if this == 0 {
        return 0;
    }
    let text = match args.get(1).copied().unwrap_or(0) {
        0 => String::new(),
        ptr => crate::fmt::read_cstr(ctx.mem, ptr)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default(),
    };
    strings().insert(this, text);
    this
}

/// `String(const String&)` / `operator=`: copy the source's text.
fn hle_string_copy(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    if this == 0 {
        return 0;
    }
    if this == src {
        return this;
    }
    let text = {
        let guard = strings();
        guard.get(&src).cloned().unwrap_or_default()
    };
    strings().insert(this, text);
    this
}

/// `~String()`: drop the contents.
fn hle_string_dtor(_ctx: &HleContext, args: &[u64]) -> u64 {
    if let Some(this) = args.first().copied() {
        strings().remove(&this);
        cstr_bufs().remove(&this);
    }
    OK
}

/// `String::c_str() const`: materialize the host-side text into a cached guest
/// buffer (reused while the capacity fits) and return its address. Also
/// mirrors the buffer pointer into the object's first 8 bytes — SharpEmu's
/// hedge for titles whose headers inlined `data()` as `*(const char**)this`.
fn hle_string_c_str(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    if this == 0 {
        return 0;
    }
    let text = {
        let guard = strings();
        guard.get(&this).cloned().unwrap_or_default()
    };
    let needed = (text.len() as u64 + 1).max(0x10);
    let cached = {
        let guard = cstr_bufs();
        guard.get(&this).copied()
    };
    let buf = match cached {
        Some((buf, cap)) if cap >= needed => buf,
        _ => {
            let Some(buf) = ctx.alloc.alloc(needed, 0x10).filter(|b| *b != 0) else {
                warn!("sceJson String::c_str: guest allocator exhausted");
                return 0;
            };
            buf
        }
    };
    let mut bytes = text.into_bytes();
    bytes.push(0);
    if !ctx.mem.write(buf, &bytes) {
        warn!("sceJson String::c_str: buffer {buf:#x} is not writable guest memory");
        return 0;
    }
    cstr_bufs().insert(this, (buf, needed));
    let _ = ctx.mem.write(this, &buf.to_le_bytes());
    buf
}

/// `String::length() const` (bytes).
fn hle_string_length(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let guard = strings();
    guard.get(&this).map_or(0, |s| s.len()) as u64
}

/// `String::empty() const`.
fn hle_string_empty(_ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let empty = {
        let guard = strings();
        guard.get(&this).is_none_or(|s| s.is_empty())
    };
    u64::from(empty)
}

/// `String::operator==(const char*) const`.
fn hle_string_eq_cstr(ctx: &HleContext, args: &[u64]) -> u64 {
    let this = args.first().copied().unwrap_or(0);
    let text = {
        let guard = strings();
        guard.get(&this).cloned().unwrap_or_default()
    };
    match crate::fmt::read_cstr(ctx.mem, args.get(1).copied().unwrap_or(0)) {
        Some(bytes) => u64::from(bytes == text.as_bytes()),
        None => 0,
    }
}

// ---------------------------------------------------------------------------
// Initializer extras
// ---------------------------------------------------------------------------

/// `Initializer::terminate()`: OK (EINVAL for a null `this`, like
/// `initialize`).
fn hle_terminate(_ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) == 0 {
        SCE_KERNEL_ERROR_EINVAL
    } else {
        OK
    }
}

/// `Initializer::setGlobalNullAccessCallBack(cb, void* ctx)`: store-and-OK.
/// The callback is kept for fidelity but never invoked — this HLE degrades
/// wrong-type accesses to defaults instead of calling back into guest code
/// (SharpEmu policy). Arg layout per SharpEmu: `this`, callback, context.
fn hle_set_global_null_access_callback(_ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    NULL_ACCESS_CALLBACK.store(args.get(1).copied().unwrap_or(0), Ordering::Relaxed);
    NULL_ACCESS_CALLBACK_CTX.store(args.get(2).copied().unwrap_or(0), Ordering::Relaxed);
    OK
}

/// `Value` construction/assignment, as `(mangled name, handler)`. Each
/// constructor and its matching `set` overload share a handler: both take
/// `(this, value)` and both may return `this`.
///
/// Itanium emits two constructor symbols per class — `C1` (complete object) and
/// `C2` (base object) — and two destructors (`D1`/`D2`); which one a title links
/// depends on its call site, so both are bound.
const VALUE_METHODS: &[(&str, crate::HleFunction)] = &[
    ("_ZN3sce4Json5ValueC1Ev", hle_value_null), // Value()
    ("_ZN3sce4Json5ValueC2Ev", hle_value_null),
    ("_ZN3sce4Json5ValueC1Eb", hle_value_bool), // Value(bool)
    ("_ZN3sce4Json5ValueC2Eb", hle_value_bool),
    ("_ZN3sce4Json5Value3setEb", hle_value_bool), // set(bool)
    ("_ZN3sce4Json5ValueC1El", hle_value_int),    // Value(long)
    ("_ZN3sce4Json5ValueC2El", hle_value_int),
    ("_ZN3sce4Json5Value3setEl", hle_value_int), // set(long)
    ("_ZN3sce4Json5ValueC1Em", hle_value_uint),  // Value(unsigned long)
    ("_ZN3sce4Json5ValueC2Em", hle_value_uint),
    ("_ZN3sce4Json5Value3setEm", hle_value_uint), // set(unsigned long)
    ("_ZN3sce4Json5ValueC1Ed", hle_value_double), // Value(double)
    ("_ZN3sce4Json5ValueC2Ed", hle_value_double),
    ("_ZN3sce4Json5Value3setEd", hle_value_double), // set(double)
    ("_ZN3sce4Json5ValueC1EPKc", hle_value_cstr),   // Value(const char*)
    ("_ZN3sce4Json5ValueC2EPKc", hle_value_cstr),
    ("_ZN3sce4Json5Value3setEPKc", hle_value_cstr), // set(const char*)
    ("_ZN3sce4Json5ValueC1ERKS1_", hle_value_copy), // Value(const Value&)
    ("_ZN3sce4Json5ValueC2ERKS1_", hle_value_copy),
    ("_ZN3sce4Json5Value3setERKS1_", hle_value_copy), // set(const Value&)
    ("_ZN3sce4Json5ValueaSERKS1_", hle_value_copy),   // operator=(const Value&)
    ("_ZN3sce4Json5ValueD1Ev", hle_value_dtor),       // ~Value()
    ("_ZN3sce4Json5ValueD2Ev", hle_value_dtor),
    ("_ZN3sce4Json5Value5clearEv", hle_value_clear), // clear()
    // `sce::Json::String` — its own class; a `Value` can be built from one.
    ("_ZN3sce4Json6StringC1Ev", hle_string_ctor), // String()
    ("_ZN3sce4Json6StringC2Ev", hle_string_ctor),
    ("_ZN3sce4Json6StringC1EPKc", hle_string_ctor), // String(const char*)
    ("_ZN3sce4Json6StringC2EPKc", hle_string_ctor),
    ("_ZN3sce4Json6StringC1ERKS1_", hle_string_copy), // String(const String&)
    ("_ZN3sce4Json6StringC2ERKS1_", hle_string_copy),
    ("_ZN3sce4Json6StringaSERKS1_", hle_string_copy), // operator=(const String&)
    ("_ZN3sce4Json6StringD1Ev", hle_string_dtor),     // ~String()
    ("_ZN3sce4Json6StringD2Ev", hle_string_dtor),
    ("_ZNK3sce4Json6String5c_strEv", hle_string_c_str), // c_str()
    ("_ZNK3sce4Json6String6lengthEv", hle_string_length), // length()
    ("_ZNK3sce4Json6String5emptyEv", hle_string_empty), // empty()
    ("_ZNK3sce4Json6StringeqEPKc", hle_string_eq_cstr), // operator==(const char*)
    // Value <- String / Object / Array bridges.
    ("_ZN3sce4Json5ValueC1ERKNS0_6StringE", hle_value_from_string), // Value(const String&)
    ("_ZN3sce4Json5ValueC2ERKNS0_6StringE", hle_value_from_string),
    (
        "_ZN3sce4Json5Value3setERKNS0_6StringE",
        hle_value_from_string,
    ), // set(const String&)
    ("_ZN3sce4Json5ValueC1ERKNS0_6ObjectE", hle_value_from_object), // Value(const Object&)
    ("_ZN3sce4Json5ValueC2ERKNS0_6ObjectE", hle_value_from_object),
    (
        "_ZN3sce4Json5Value3setERKNS0_6ObjectE",
        hle_value_from_object,
    ), // set(const Object&)
    ("_ZN3sce4Json5ValueC1ERKNS0_5ArrayE", hle_value_from_array), // Value(const Array&)
    ("_ZN3sce4Json5ValueC2ERKNS0_5ArrayE", hle_value_from_array),
    ("_ZN3sce4Json5Value3setERKNS0_5ArrayE", hle_value_from_array), // set(const Array&)
    // Value(ValueType) / set(ValueType).
    ("_ZN3sce4Json5ValueC1ENS0_9ValueTypeE", hle_value_set_type),
    ("_ZN3sce4Json5ValueC2ENS0_9ValueTypeE", hle_value_set_type),
    ("_ZN3sce4Json5Value3setENS0_9ValueTypeE", hle_value_set_type),
    // Value getters.
    ("_ZNK3sce4Json5Value7getTypeEv", hle_value_get_type),
    ("_ZNK3sce4Json5Value5countEv", hle_value_count),
    ("_ZNK3sce4Json5Value10getBooleanEv", hle_value_get_boolean),
    ("_ZNK3sce4Json5Value10getIntegerEv", hle_value_get_integer),
    ("_ZNK3sce4Json5Value11getUIntegerEv", hle_value_get_uinteger),
    ("_ZNK3sce4Json5Value7getRealEv", hle_value_get_real),
    ("_ZNK3sce4Json5Value9getStringEv", hle_value_get_string),
    ("_ZNK3sce4Json5Value9getObjectEv", hle_value_get_object),
    ("_ZNK3sce4Json5Value8getArrayEv", hle_value_get_array),
    ("_ZNK3sce4Json5ValueixEPKc", hle_value_ix_cstr), // operator[](const char*)
    ("_ZNK3sce4Json5ValueixEm", hle_value_ix_index),  // operator[](size_t)
    ("_ZNK3sce4Json5Value8getValueEm", hle_value_ix_index), // getValue(size_t)
    (
        "_ZN3sce4Json5Value10referValueERKNS0_6StringE",
        hle_value_refer_value,
    ),
    ("_ZN3sce4Json5Value11referObjectEv", hle_value_refer_object),
    ("_ZN3sce4Json5Value10referArrayEv", hle_value_refer_array),
    (
        "_ZNK3sce4Json5Value8toStringERNS0_6StringE",
        hle_value_to_string,
    ),
    (
        "_ZN3sce4Json5Value9serializeERNS0_6StringE",
        hle_value_serialize,
    ),
    // Object.
    ("_ZN3sce4Json6ObjectC1Ev", hle_object_ctor),
    ("_ZN3sce4Json6ObjectC2Ev", hle_object_ctor),
    ("_ZN3sce4Json6ObjectC1ERKS1_", hle_object_copy),
    ("_ZN3sce4Json6ObjectC2ERKS1_", hle_object_copy),
    ("_ZN3sce4Json6ObjectaSERKS1_", hle_object_copy),
    ("_ZN3sce4Json6ObjectD1Ev", hle_object_dtor),
    ("_ZN3sce4Json6ObjectD2Ev", hle_object_dtor),
    ("_ZN3sce4Json6Object5clearEv", hle_object_clear),
    ("_ZN3sce4Json6ObjectixERKNS0_6StringE", hle_object_index),
    ("_ZNK3sce4Json6Object5beginEv", hle_object_begin),
    ("_ZNK3sce4Json6Object3endEv", hle_object_end),
    ("_ZNK3sce4Json6Object8iteratorneERKS2_", hle_iter_ne),
    ("_ZN3sce4Json6Object8iteratorppEv", hle_iter_pp),
    ("_ZNK3sce4Json6Object8iteratordeEv", hle_iter_deref),
    ("_ZN3sce4Json6Object8iteratorD1Ev", hle_iter_dtor),
    ("_ZN3sce4Json6Object8iteratorD2Ev", hle_iter_dtor),
    // Array.
    ("_ZN3sce4Json5ArrayC1Ev", hle_array_ctor),
    ("_ZN3sce4Json5ArrayC2Ev", hle_array_ctor),
    ("_ZN3sce4Json5ArrayD1Ev", hle_array_dtor),
    ("_ZN3sce4Json5ArrayD2Ev", hle_array_dtor),
    (
        "_ZN3sce4Json5Array9push_backERKNS0_5ValueE",
        hle_array_push_back,
    ),
    ("_ZNK3sce4Json5Array4sizeEv", hle_array_size),
    ("_ZNK3sce4Json5Array5emptyEv", hle_array_empty),
    ("_ZNK3sce4Json5Array4backEv", hle_array_back),
    ("_ZNK3sce4Json5Array5beginEv", hle_array_begin),
    ("_ZNK3sce4Json5Array3endEv", hle_array_end),
    ("_ZNK3sce4Json5Array8iteratorneERKS2_", hle_iter_ne),
    ("_ZN3sce4Json5Array8iteratorppEv", hle_iter_pp),
    ("_ZNK3sce4Json5Array8iteratordeEv", hle_iter_deref),
    ("_ZN3sce4Json5Array8iteratorD1Ev", hle_iter_dtor),
    ("_ZN3sce4Json5Array8iteratorD2Ev", hle_iter_dtor),
    // Parser + Initializer extras.
    (
        "_ZN3sce4Json6Parser5parseERNS0_5ValueEPKcm",
        hle_parser_parse,
    ),
    ("_ZN3sce4Json11Initializer9terminateEv", hle_terminate),
    (
        "_ZN3sce4Json11Initializer27setGlobalNullAccessCallBackEPFRKNS0_5ValueENS0_9ValueTypeEPS3_PvES7_",
        hle_set_global_null_access_callback,
    ),
];

/// Register the `sce::Json` lifecycle under both `libSceJson` and
/// `libSceJson2` (a title may import from either).
pub fn register(registry: &HleRegistry) {
    for lib in ["libSceJson", "libSceJson2"] {
        for &f in RET_THIS {
            registry.register(lib, f, hle_ret_this);
        }
        for &f in RET_ZERO {
            registry.register(lib, f, hle_ret_zero);
        }
        for &f in INITIALIZE {
            registry.register(lib, f, hle_initialize);
        }
        for &(name, handler) in VALUE_METHODS {
            registry.register(lib, name, handler);
        }
    }
}

/// C++ constructor / fluent setter: return `this` (the first argument).
fn hle_ret_this(_ctx: &HleContext, args: &[u64]) -> u64 {
    args.first().copied().unwrap_or(0)
}

/// C++ destructor: returns void (`rax = 0`).
fn hle_ret_zero(_ctx: &HleContext, _args: &[u64]) -> u64 {
    OK
}

/// `Initializer::initialize(this, param)`: OK, or `EINVAL` for a null `this`.
fn hle_initialize(_ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) == 0 {
        SCE_KERNEL_ERROR_EINVAL
    } else {
        OK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GuestMemory;

    /// One shared guest-memory size for all tests; each test uses a disjoint
    /// address range because the payload maps are process-global statics and
    /// the tests run in parallel.
    const MEM_BYTES: u64 = 0x10_0000;

    fn fixtures() -> (xps5x_kernel::OrbisKernel, crate::TestMemory) {
        (
            xps5x_kernel::OrbisKernel::new(),
            crate::TestMemory::new(MEM_BYTES as usize),
        )
    }

    fn read_u64(mem: &crate::TestMemory, addr: u64) -> u64 {
        let mut b = [0u8; 8];
        assert!(mem.read(addr, &mut b), "slot {addr:#x} must be readable");
        u64::from_le_bytes(b)
    }

    /// The point of the host-side model: a `set` must actually STORE, so the
    /// payload survives to a later read. A stub that only returned `this` would
    /// pass a "returns this" assertion while leaving the guest with garbage.
    #[test]
    fn value_payloads_round_trip_and_die_with_the_object() {
        let (kernel, mem, alloc) = (
            xps5x_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x100),
            crate::TestAllocator::new(0),
        );
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        let peek = |this: u64| values().get(&this).cloned();
        const A: u64 = 0xA000;
        const B: u64 = 0xB000;

        // Construct null, then re-`set` through several types.
        assert_eq!(hle_value_null(&ctx, &[A]), A, "a constructor returns this");
        assert_eq!(peek(A), Some(JsonValue::Null));
        hle_value_bool(&ctx, &[A, 1]);
        assert_eq!(peek(A), Some(JsonValue::Bool(true)));
        hle_value_int(&ctx, &[A, (-7i64) as u64]);
        assert_eq!(peek(A), Some(JsonValue::Int(-7)));
        hle_value_uint(&ctx, &[A, 9]);
        assert_eq!(peek(A), Some(JsonValue::UInt(9)));

        // A string is copied out of guest memory.
        assert!(mem.write(0x40, b"hello\0"));
        hle_value_cstr(&ctx, &[A, 0x40]);
        assert_eq!(peek(A), Some(JsonValue::Str("hello".to_owned())));

        // Copy-construction takes the source's payload.
        hle_value_copy(&ctx, &[B, A]);
        assert_eq!(peek(B), Some(JsonValue::Str("hello".to_owned())));

        // `double` comes from the XMM float channel, which test_ctx zeroes.
        hle_value_double(&ctx, &[A]);
        assert_eq!(peek(A), Some(JsonValue::Double(0.0)));

        // Destruction drops the payload; the other object is untouched.
        assert_eq!(hle_value_dtor(&ctx, &[A]), 0);
        assert_eq!(peek(A), None);
        assert_eq!(peek(B), Some(JsonValue::Str("hello".to_owned())));
        hle_value_dtor(&ctx, &[B]);

        // A null `this` is not a crash and stores nothing.
        assert_eq!(hle_value_null(&ctx, &[0]), 0);
    }

    #[test]
    fn constructors_return_this_destructors_return_zero() {
        let (kernel, mem, alloc) = (
            xps5x_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x10),
            crate::TestAllocator::new(0),
        );
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        // A constructor returns its `this` pointer (arg0).
        assert_eq!(hle_ret_this(&ctx, &[0xCAFE]), 0xCAFE);
        // A destructor returns 0.
        assert_eq!(hle_ret_zero(&ctx, &[0xCAFE]), 0);
        // initialize: OK for a real `this`, EINVAL for null.
        assert_eq!(hle_initialize(&ctx, &[0xCAFE, 0x1234]), OK);
        assert_eq!(hle_initialize(&ctx, &[0, 0x1234]), SCE_KERNEL_ERROR_EINVAL);
    }

    #[test]
    fn all_lifecycle_nids_resolve_under_both_libraries() {
        let reg = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        for lib in ["libSceJson", "libSceJson2"] {
            // A constructor resolves and returns `this`.
            assert_eq!(
                reg.call(&ctx, lib, "_ZN3sce4Json11InitializerC1Ev", &[0x99]),
                Some(0x99),
                "{lib} ctor must resolve"
            );
            // A representative of the new surface resolves too.
            assert_eq!(
                reg.call(&ctx, lib, "_ZN3sce4Json5ArrayC1Ev", &[0x98]),
                Some(0x98),
                "{lib} Array ctor must resolve"
            );
        }
    }

    /// Parser::parse builds a navigable tree: object member lookup, nested
    /// array indexing, and scalar/string reference getters all resolve through
    /// guest-visible anchors. Address range: objects 0x1000.., allocator
    /// 0x10000...
    #[test]
    fn parse_then_navigate_members_and_scalars() {
        let (kernel, mem) = fixtures();
        let alloc = crate::TestAllocator::new(0x1_0000);
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        const V: u64 = 0x1000;

        let doc = br#"{"name":"astro","count":42,"flags":[true,false]}"#;
        assert!(mem.write(0x1800, doc));
        assert_eq!(hle_parser_parse(&ctx, &[V, 0x1800, doc.len() as u64]), OK);
        assert_eq!(hle_value_get_type(&ctx, &[V]), 7, "root is an object");
        assert_eq!(hle_value_count(&ctx, &[V]), 3);

        // operator[]("name") -> string member.
        assert!(mem.write(0x1900, b"name\0"));
        let name = hle_value_ix_cstr(&ctx, &[V, 0x1900]);
        assert_ne!(name, 0);
        assert_eq!(hle_value_get_type(&ctx, &[name]), 5);
        let sref = hle_value_get_string(&ctx, &[name]);
        assert_eq!(strings().get(&sref).cloned().as_deref(), Some("astro"));
        // ... and c_str materializes readable guest bytes.
        let cbuf = hle_string_c_str(&ctx, &[sref]);
        assert_ne!(cbuf, 0);
        let mut got = [0u8; 6];
        assert!(mem.read(cbuf, &mut got));
        assert_eq!(&got, b"astro\0");

        // operator[]("count") -> integer through a scalar reference slot.
        assert!(mem.write(0x1910, b"count\0"));
        let count = hle_value_ix_cstr(&ctx, &[V, 0x1910]);
        assert_eq!(hle_value_get_type(&ctx, &[count]), 2);
        let slot = hle_value_get_integer(&ctx, &[count]);
        assert_eq!(read_u64(&mem, slot) as i64, 42);

        // operator[]("flags")[0] -> boolean.
        assert!(mem.write(0x1920, b"flags\0"));
        let flags = hle_value_ix_cstr(&ctx, &[V, 0x1920]);
        assert_eq!(hle_value_get_type(&ctx, &[flags]), 6);
        assert_eq!(hle_value_count(&ctx, &[flags]), 2);
        let first = hle_value_ix_index(&ctx, &[flags, 0]);
        let bslot = hle_value_get_boolean(&ctx, &[first]);
        assert_eq!(read_u64(&mem, bslot) & 1, 1);

        // A missing key still hands back a dereferenceable null value.
        assert!(mem.write(0x1930, b"absent\0"));
        let missing = hle_value_ix_cstr(&ctx, &[V, 0x1930]);
        assert_ne!(missing, 0);
        assert_eq!(hle_value_get_type(&ctx, &[missing]), 0);

        // Destroying the root frees the whole subtree host-side.
        assert_eq!(hle_value_dtor(&ctx, &[V]), 0);
        assert!(values().get(&name).is_none(), "member freed with the root");
        assert!(
            values().get(&first).is_none(),
            "element freed with the root"
        );
    }

    /// Array push_back copies the value (later mutation of the source must not
    /// leak in), size/empty/back track contents. Range: objects 0x2000..,
    /// allocator 0x30000...
    #[test]
    fn array_push_back_size_back() {
        let (kernel, mem) = fixtures();
        let alloc = crate::TestAllocator::new(0x3_0000);
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        const ARR: u64 = 0x2000;
        const V1: u64 = 0x2100;

        assert_eq!(hle_array_ctor(&ctx, &[ARR]), ARR);
        assert_eq!(hle_array_empty(&ctx, &[ARR]), 1);

        hle_value_int(&ctx, &[V1, 7]);
        assert_eq!(hle_array_push_back(&ctx, &[ARR, V1]), OK);
        hle_value_int(&ctx, &[V1, 8]);
        hle_array_push_back(&ctx, &[ARR, V1]);

        assert_eq!(hle_array_size(&ctx, &[ARR]), 2);
        assert_eq!(hle_array_empty(&ctx, &[ARR]), 0);

        // back() is the second push; the copy is deep, so mutating V1 after
        // the fact must not change it.
        hle_value_int(&ctx, &[V1, 99]);
        let back = hle_array_back(&ctx, &[ARR]);
        assert_eq!(values().get(&back).cloned(), Some(JsonValue::Int(8)));

        // Iterate: begin != end, two elements, then exhausted.
        const IT: u64 = 0x2200;
        const END: u64 = 0x2280;
        assert_eq!(
            hle_array_begin(&ctx, &[IT, ARR]),
            IT,
            "sret ptr is returned"
        );
        assert_eq!(hle_array_end(&ctx, &[END, ARR]), END);
        assert_eq!(hle_iter_ne(&ctx, &[IT, END]), 1);
        let e0 = hle_iter_deref(&ctx, &[IT]);
        assert_eq!(values().get(&e0).cloned(), Some(JsonValue::Int(7)));
        hle_iter_pp(&ctx, &[IT]);
        assert_eq!(hle_iter_ne(&ctx, &[IT, END]), 1);
        hle_iter_pp(&ctx, &[IT]);
        assert_eq!(hle_iter_ne(&ctx, &[IT, END]), 0, "walk terminates");
        hle_iter_dtor(&ctx, &[IT]);
        hle_iter_dtor(&ctx, &[END]);

        // Destruction frees the elements.
        hle_array_dtor(&ctx, &[ARR]);
        assert!(values().get(&back).is_none());
        hle_value_dtor(&ctx, &[V1]);
    }

    /// Build an Object through operator[], then walk it with begin/end
    /// iterators: deref yields an entry whose key String sits at +0x0 and
    /// member Value at the entry-value offset. Range: objects 0x3000..,
    /// allocator 0x50000...
    #[test]
    fn object_index_and_iterator_walk() {
        let (kernel, mem) = fixtures();
        let alloc = crate::TestAllocator::new(0x5_0000);
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        const OBJ: u64 = 0x3000;
        const KEY1: u64 = 0x3100;
        const KEY2: u64 = 0x3180;

        assert_eq!(hle_object_ctor(&ctx, &[OBJ]), OBJ);

        assert!(mem.write(0x3800, b"alpha\0"));
        assert!(mem.write(0x3810, b"beta\0"));
        hle_string_ctor(&ctx, &[KEY1, 0x3800]);
        hle_string_ctor(&ctx, &[KEY2, 0x3810]);

        // operator[] inserts null members and returns stable Value addresses.
        let a = hle_object_index(&ctx, &[OBJ, KEY1]);
        let b = hle_object_index(&ctx, &[OBJ, KEY2]);
        assert_ne!(a, 0);
        assert_ne!(b, a);
        hle_value_int(&ctx, &[a, 1]);
        hle_value_int(&ctx, &[b, 2]);
        assert_eq!(
            hle_object_index(&ctx, &[OBJ, KEY1]),
            a,
            "same key resolves to the same member"
        );

        // Iterator walk, in insertion order.
        const IT: u64 = 0x3200;
        const END: u64 = 0x3280;
        assert_eq!(hle_object_begin(&ctx, &[IT, OBJ]), IT);
        assert_eq!(hle_object_end(&ctx, &[END, OBJ]), END);
        assert_eq!(hle_iter_ne(&ctx, &[IT, END]), 1);
        let entry = hle_iter_deref(&ctx, &[IT]);
        assert_eq!(strings().get(&entry).cloned().as_deref(), Some("alpha"));
        assert_eq!(
            values().get(&(entry + ENTRY_VALUE_OFFSET)).cloned(),
            Some(JsonValue::Int(1))
        );
        hle_iter_pp(&ctx, &[IT]);
        let entry = hle_iter_deref(&ctx, &[IT]);
        assert_eq!(strings().get(&entry).cloned().as_deref(), Some("beta"));
        hle_iter_pp(&ctx, &[IT]);
        assert_eq!(hle_iter_ne(&ctx, &[IT, END]), 0);
        hle_iter_dtor(&ctx, &[IT]);
        hle_iter_dtor(&ctx, &[END]);

        // Copies are deep: mutate the original, the copy is unaffected.
        const COPY: u64 = 0x3300;
        assert_eq!(hle_object_copy(&ctx, &[COPY, OBJ]), COPY);
        hle_value_int(&ctx, &[a, 111]);
        let copy_a = hle_object_index(&ctx, &[COPY, KEY1]);
        assert_eq!(values().get(&copy_a).cloned(), Some(JsonValue::Int(1)));

        // clear drops the members but keeps the object alive.
        hle_object_clear(&ctx, &[OBJ]);
        assert!(values().get(&a).is_none());
        assert!(objects().get(&OBJ).is_some_and(|e| e.is_empty()));

        hle_object_dtor(&ctx, &[OBJ]);
        hle_object_dtor(&ctx, &[COPY]);
        hle_string_dtor(&ctx, &[KEY1]);
        hle_string_dtor(&ctx, &[KEY2]);
    }

    /// serialize emits JSON that parses back to the same document; toString on
    /// a string value yields raw (unquoted) text. Range: objects 0x4000..,
    /// allocator 0x70000...
    #[test]
    fn serialize_round_trips_and_tostring_is_raw() {
        let (kernel, mem) = fixtures();
        let alloc = crate::TestAllocator::new(0x7_0000);
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        const V: u64 = 0x4000;
        const OUT: u64 = 0x4100;

        let doc = br#"{"a":[1,2.5,"z"],"b":true,"c":null,"n":-3}"#;
        assert!(mem.write(0x4800, doc));
        assert_eq!(hle_parser_parse(&ctx, &[V, 0x4800, doc.len() as u64]), OK);

        assert_eq!(hle_value_serialize(&ctx, &[V, OUT]), OK);
        let text = strings().get(&OUT).cloned().expect("serialized text");
        let round: serde_json::Value = serde_json::from_str(&text).expect("valid JSON out");
        let original: serde_json::Value = serde_json::from_slice(doc).unwrap();
        assert_eq!(round, original, "serialize must round-trip the document");

        // toString of a string value is the raw text, not a JSON quote.
        const SV: u64 = 0x4200;
        assert!(mem.write(0x4900, b"plain\0"));
        hle_value_cstr(&ctx, &[SV, 0x4900]);
        assert_eq!(hle_value_to_string(&ctx, &[SV, OUT]), OK);
        assert_eq!(strings().get(&OUT).cloned().as_deref(), Some("plain"));

        hle_value_dtor(&ctx, &[V]);
        hle_value_dtor(&ctx, &[SV]);
        strings().remove(&OUT);
    }

    /// Parse error paths, the null-access callback store, terminate, refer*
    /// conversion, and String comparisons. Range: objects 0x5000.., allocator
    /// 0x90000...
    #[test]
    fn parse_errors_callback_refer_and_string_ops() {
        let (kernel, mem) = fixtures();
        let alloc = crate::TestAllocator::new(0x9_0000);
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        const V: u64 = 0x5000;

        // Parser error family.
        assert_eq!(
            hle_parser_parse(&ctx, &[V, 0x5800, 0]),
            SCE_JSON_PARSER_ERROR_EMPTY_BUFFER
        );
        assert!(mem.write(0x5800, b"{not json"));
        assert_eq!(
            hle_parser_parse(&ctx, &[V, 0x5800, 9]),
            SCE_JSON_PARSER_ERROR_INVALID_TOKEN
        );
        assert_eq!(
            hle_parser_parse(&ctx, &[V, 0x5800, MAX_PARSE_BYTES + 1]),
            SCE_JSON_PARSER_ERROR_INVALID_TOKEN
        );

        // The measured first wall: store-and-OK, EINVAL on null this.
        assert_eq!(
            hle_set_global_null_access_callback(&ctx, &[0x5010, 0xCB00, 0xCC00]),
            OK
        );
        assert_eq!(NULL_ACCESS_CALLBACK.load(Ordering::Relaxed), 0xCB00);
        assert_eq!(NULL_ACCESS_CALLBACK_CTX.load(Ordering::Relaxed), 0xCC00);
        assert_eq!(
            hle_set_global_null_access_callback(&ctx, &[0, 1, 2]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(hle_terminate(&ctx, &[0x5010]), OK);
        assert_eq!(hle_terminate(&ctx, &[0]), SCE_KERNEL_ERROR_EINVAL);

        // referValue converts a null value into an object and creates members.
        const KEY: u64 = 0x5100;
        assert!(mem.write(0x5810, b"speed\0"));
        hle_string_ctor(&ctx, &[KEY, 0x5810]);
        hle_value_null(&ctx, &[V]);
        let member = hle_value_refer_value(&ctx, &[V, KEY]);
        assert_ne!(member, 0);
        assert_eq!(hle_value_get_type(&ctx, &[V]), 7, "converted to object");
        hle_value_int(&ctx, &[member, 5]);
        assert_eq!(
            hle_value_refer_value(&ctx, &[V, KEY]),
            member,
            "same member on re-lookup"
        );
        let obj = hle_value_refer_object(&ctx, &[V]);
        assert_ne!(obj, 0);
        assert_eq!(hle_value_count(&ctx, &[V]), 1);

        // String copy/assign/length/empty/eq.
        const S2: u64 = 0x5200;
        assert_eq!(hle_string_copy(&ctx, &[S2, KEY]), S2);
        assert_eq!(hle_string_length(&ctx, &[S2]), 5);
        assert_eq!(hle_string_empty(&ctx, &[S2]), 0);
        assert_eq!(hle_string_eq_cstr(&ctx, &[S2, 0x5810]), 1);
        assert!(mem.write(0x5820, b"other\0"));
        assert_eq!(hle_string_eq_cstr(&ctx, &[S2, 0x5820]), 0);

        hle_string_dtor(&ctx, &[S2]);
        hle_string_dtor(&ctx, &[KEY]);
        hle_value_dtor(&ctx, &[V]);
    }
}
