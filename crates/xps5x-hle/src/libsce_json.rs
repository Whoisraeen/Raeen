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
