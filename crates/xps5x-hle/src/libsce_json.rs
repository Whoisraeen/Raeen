//! HLE libSceJson / libSceJson2 — `sce::Json` C++ object lifecycle.
//!
//! A faithful port of SharpEmu's `Json` exports (GPL-2.0). These are the
//! Itanium-ABI-mangled `sce::Json` construction/configuration methods a title
//! links against: `MemAllocator`/`Initializer`/`InitParameter2` constructors,
//! destructors, the fluent `set*` configurators, and `Initializer::initialize`.
//!
//! Following the C++ ABI: a **constructor returns `this`** (the object pointer
//! in the first argument), a **destructor returns 0**, the fluent **setters
//! return `this`**, and **`initialize` returns OK** (or `EINVAL` for a null
//! `this`). Actual JSON *parsing* is not exercised by these lifecycle NIDs;
//! without them a title using `sce::Json` hits an unresolved import and dies.

use crate::{HleContext, HleRegistry};

const OK: u64 = 0;
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;

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
#[derive(Debug, Clone, Default, PartialEq)]
enum JsonValue {
    #[default]
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Double(f64),
    Str(String),
}

/// Live `sce::Json::Value` payloads, by guest `this`.
static VALUES: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<u64, JsonValue>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Store `value` for the object at `this` and return `this` — the C++ ABI
/// return for both a constructor and a fluent setter.
fn store(this: u64, value: JsonValue) -> u64 {
    if this == 0 {
        return 0;
    }
    VALUES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(this, value);
    this
}

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

/// `Value(const Value &)` / `set(const Value &)`: copy the source's payload.
fn hle_value_copy(_ctx: &HleContext, args: &[u64]) -> u64 {
    let src = args.get(1).copied().unwrap_or(0);
    let value = VALUES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(&src)
        .cloned()
        .unwrap_or_default();
    store(args.first().copied().unwrap_or(0), value)
}

/// `~Value()`: drop the payload. Returns void.
fn hle_value_dtor(_ctx: &HleContext, args: &[u64]) -> u64 {
    if let Some(this) = args.first().copied() {
        VALUES
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&this);
    }
    OK
}

/// Live `sce::Json::String` contents, by guest `this`. Separate from
/// [`VALUES`] because `String` is its own class in the library, even though a
/// `Value` can hold one.
static STRINGS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<u64, String>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

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
    STRINGS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(this, text);
    this
}

/// `Value(ValueType)` / `set(ValueType)`: reset the value to the default of the
/// named type.
///
/// `sce::Json::ValueType` is the library's tag enum, ordered
/// null/boolean/integer/uinteger/real/string/array/object — the SDK's
/// `kValueType*` order. An unrecognised code falls back to null rather than
/// guessing, which is the same "no value" state a default-constructed `Value`
/// holds.
fn hle_value_set_type(_ctx: &HleContext, args: &[u64]) -> u64 {
    let value = match args.get(1).copied().unwrap_or(0) {
        1 => JsonValue::Bool(false),
        2 => JsonValue::Int(0),
        3 => JsonValue::UInt(0),
        4 => JsonValue::Double(0.0),
        5 => JsonValue::Str(String::new()),
        // 0 (null) and the aggregate types (array/object) have no scalar
        // payload to seed; they stay null until something is stored into them.
        _ => JsonValue::Null,
    };
    store(args.first().copied().unwrap_or(0), value)
}

/// `Value(const String &)` / `set(const String &)`: take the text from the
/// `sce::Json::String` the guest passes and hold it as a JSON string.
fn hle_value_from_string(_ctx: &HleContext, args: &[u64]) -> u64 {
    let src = args.get(1).copied().unwrap_or(0);
    let text = STRINGS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(&src)
        .cloned()
        .unwrap_or_default();
    store(args.first().copied().unwrap_or(0), JsonValue::Str(text))
}

/// `~String()`: drop the contents.
fn hle_string_dtor(_ctx: &HleContext, args: &[u64]) -> u64 {
    if let Some(this) = args.first().copied() {
        STRINGS
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&this);
    }
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
    ("_ZN3sce4Json5ValueD1Ev", hle_value_dtor),       // ~Value()
    ("_ZN3sce4Json5ValueD2Ev", hle_value_dtor),
    // `sce::Json::String` — its own class; a `Value` can be built from one.
    ("_ZN3sce4Json6StringC1Ev", hle_string_ctor), // String()
    ("_ZN3sce4Json6StringC2Ev", hle_string_ctor),
    ("_ZN3sce4Json6StringC1EPKc", hle_string_ctor), // String(const char*)
    ("_ZN3sce4Json6StringC2EPKc", hle_string_ctor),
    ("_ZN3sce4Json6StringD1Ev", hle_string_dtor), // ~String()
    ("_ZN3sce4Json6StringD2Ev", hle_string_dtor),
    // Value <- String bridge.
    ("_ZN3sce4Json5ValueC1ERKNS0_6StringE", hle_value_from_string), // Value(const String&)
    ("_ZN3sce4Json5ValueC2ERKNS0_6StringE", hle_value_from_string),
    (
        "_ZN3sce4Json5Value3setERKNS0_6StringE",
        hle_value_from_string,
    ), // set(const String&)
    // Value(ValueType) / set(ValueType).
    ("_ZN3sce4Json5ValueC1ENS0_9ValueTypeE", hle_value_set_type),
    ("_ZN3sce4Json5ValueC2ENS0_9ValueTypeE", hle_value_set_type),
    ("_ZN3sce4Json5Value3setENS0_9ValueTypeE", hle_value_set_type),
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
        let peek = |this: u64| VALUES.lock().unwrap().get(&this).cloned();
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
        }
    }
}
