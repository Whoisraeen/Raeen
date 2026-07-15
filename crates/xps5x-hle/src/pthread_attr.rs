//! HLE libkernel pthread **thread-attribute** objects (`scePthreadAttr*`).
//!
//! A faithful Rust port of SharpEmu's thread-attribute state
//! (`PthreadAttrState`, GPL-2.0). A title configures a `pthread_attr_t` — stack
//! size, detach state, guard size, scheduling params — before
//! `scePthreadCreate`, then reads fields back. This is pure configuration data
//! with **no runtime dependency**, so it ports completely (unlike thread
//! creation itself, which needs the M1-E scheduler). State lives in the kernel
//! (`OrbisKernel::pthread_attrs`), keyed by both the guest `pthread_attr_t`
//! address and the opaque handle `Init` allocates and writes into `*attr`.

use crate::{HleContext, HleRegistry};
use tracing::debug;
use xps5x_kernel::PthreadAttr;

const OK: u64 = 0;
const EINVAL: u64 = 22;

/// Size of the opaque attribute object handed to the guest.
const ATTR_OBJECT_SIZE: u64 = 0x40;

/// Register the pthread thread-attribute HLE functions.
pub fn register(registry: &HleRegistry) {
    register_posix(registry);

    registry.register("libkernel", "scePthreadAttrInit", hle_attr_init);
    registry.register("libkernel", "scePthreadAttrDestroy", hle_attr_destroy);
    registry.register(
        "libkernel",
        "scePthreadAttrSetdetachstate",
        hle_set_detachstate,
    );
    registry.register(
        "libkernel",
        "scePthreadAttrGetdetachstate",
        hle_get_detachstate,
    );
    registry.register("libkernel", "scePthreadAttrSetstacksize", hle_set_stacksize);
    registry.register("libkernel", "scePthreadAttrGetstacksize", hle_get_stacksize);
    registry.register("libkernel", "scePthreadAttrSetguardsize", hle_set_guardsize);
    registry.register("libkernel", "scePthreadAttrGetguardsize", hle_get_guardsize);
    registry.register(
        "libkernel",
        "scePthreadAttrSetschedpolicy",
        hle_set_schedpolicy,
    );
}

/// The POSIX spellings, under `libScePosix`.
///
/// Different NIDs from the `scePthread*` forms (a NID hashes the name alone),
/// and these are the ones a real title imports. Same signatures and the same
/// POSIX return convention (0 / positive errno), so the implementations alias
/// directly.
fn register_posix(registry: &HleRegistry) {
    registry.register("libScePosix", "pthread_attr_init", hle_attr_init);
    registry.register("libScePosix", "pthread_attr_destroy", hle_attr_destroy);
    registry.register(
        "libScePosix",
        "pthread_attr_setdetachstate",
        hle_set_detachstate,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_getdetachstate",
        hle_get_detachstate,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_setstacksize",
        hle_set_stacksize,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_getstacksize",
        hle_get_stacksize,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_setguardsize",
        hle_set_guardsize,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_getguardsize",
        hle_get_guardsize,
    );
}

/// Resolve the attr-state key for a guest `pthread_attr_t` address: the address
/// if registered, else the handle it points at.
fn resolve_key(ctx: &HleContext, attr_addr: u64) -> Option<u64> {
    if ctx.kernel.pthread_attrs.contains_key(&attr_addr) {
        return Some(attr_addr);
    }
    let mut buf = [0u8; 8];
    if ctx.mem.read(attr_addr, &mut buf) {
        let handle = u64::from_le_bytes(buf);
        if handle != 0 && ctx.kernel.pthread_attrs.contains_key(&handle) {
            return Some(handle);
        }
    }
    None
}

/// `scePthreadAttrInit(attr)`: allocate an opaque object, write its handle into
/// `*attr`, and register default attribute state.
fn hle_attr_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr_addr = args.first().copied().unwrap_or(0);
    if attr_addr == 0 {
        return EINVAL;
    }
    let Some(handle) = ctx.alloc.alloc(ATTR_OBJECT_SIZE, 0x10) else {
        return EINVAL;
    };
    if !ctx.mem.write(attr_addr, &handle.to_le_bytes()) {
        return EINVAL;
    }
    let state = PthreadAttr::default();
    ctx.kernel.pthread_attrs.insert(attr_addr, state);
    ctx.kernel.pthread_attrs.insert(handle, state);
    debug!("scePthreadAttrInit(attr={attr_addr:#x}) -> handle {handle:#x}");
    OK
}

/// `scePthreadAttrDestroy(attr)`.
fn hle_attr_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr_addr = args.first().copied().unwrap_or(0);
    if attr_addr == 0 {
        return EINVAL;
    }
    let Some(key) = resolve_key(ctx, attr_addr) else {
        return EINVAL;
    };
    ctx.kernel.pthread_attrs.remove(&key);
    if key != attr_addr {
        ctx.kernel.pthread_attrs.remove(&attr_addr);
    }
    let _ = ctx.mem.write(attr_addr, &0u64.to_le_bytes());
    OK
}

/// Apply `update` to the attr state for `attr_addr` (creating a default if the
/// title never called `Init` — a static `pthread_attr_t`).
fn with_attr(ctx: &HleContext, attr_addr: u64, update: impl FnOnce(&mut PthreadAttr)) -> u64 {
    if attr_addr == 0 {
        return EINVAL;
    }
    let key = resolve_key(ctx, attr_addr).unwrap_or_else(|| {
        ctx.kernel
            .pthread_attrs
            .insert(attr_addr, PthreadAttr::default());
        attr_addr
    });
    let mut entry = ctx.kernel.pthread_attrs.get_mut(&key).unwrap();
    update(&mut entry);
    OK
}

/// Read the attr state for `attr_addr`, or the default if unset.
fn read_attr(ctx: &HleContext, attr_addr: u64) -> Option<PthreadAttr> {
    let key = resolve_key(ctx, attr_addr)?;
    ctx.kernel.pthread_attrs.get(&key).map(|e| *e)
}

fn hle_set_detachstate(ctx: &HleContext, args: &[u64]) -> u64 {
    let state = args.get(1).copied().unwrap_or(0) as i32;
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.detach_state = state
    })
}

fn hle_set_stacksize(ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.get(1).copied().unwrap_or(0);
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.stack_size = size
    })
}

fn hle_set_guardsize(ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.get(1).copied().unwrap_or(0);
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.guard_size = size
    })
}

fn hle_set_schedpolicy(ctx: &HleContext, args: &[u64]) -> u64 {
    let policy = args.get(1).copied().unwrap_or(0) as i32;
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.sched_policy = policy
    })
}

/// `scePthreadAttrGetdetachstate(attr, int *out)`.
fn hle_get_detachstate(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    let Some(state) = read_attr(ctx, attr) else {
        return EINVAL;
    };
    if out == 0 || !ctx.mem.write(out, &state.detach_state.to_le_bytes()) {
        return EINVAL;
    }
    OK
}

/// `scePthreadAttrGetstacksize(attr, size_t *out)`.
fn hle_get_stacksize(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    let Some(state) = read_attr(ctx, attr) else {
        return EINVAL;
    };
    if out == 0 || !ctx.mem.write(out, &state.stack_size.to_le_bytes()) {
        return EINVAL;
    }
    OK
}

/// `scePthreadAttrGetguardsize(attr, size_t *out)`.
fn hle_get_guardsize(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    let Some(state) = read_attr(ctx, attr) else {
        return EINVAL;
    };
    if out == 0 || !ctx.mem.write(out, &state.guard_size.to_le_bytes()) {
        return EINVAL;
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn ctx_env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x1000);
        (kernel, mem, alloc)
    }

    #[test]
    fn init_writes_a_handle_and_defaults() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x100;
        assert_eq!(hle_attr_init(&ctx, &[attr]), OK);
        let mut buf = [0u8; 8];
        assert!(mem.read(attr, &mut buf));
        assert!(u64::from_le_bytes(buf) != 0, "init writes a handle");
        // Default stack size (1 MiB) is readable back.
        let out = 0x200;
        assert_eq!(hle_get_stacksize(&ctx, &[attr, out]), OK);
        let mut sz = [0u8; 8];
        assert!(mem.read(out, &mut sz));
        assert_eq!(u64::from_le_bytes(sz), 0x10_0000);
        assert_eq!(hle_attr_init(&ctx, &[0]), EINVAL);
    }

    #[test]
    fn set_then_get_round_trips_stacksize_and_detachstate() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x300;
        hle_attr_init(&ctx, &[attr]);
        // Stack size round-trips.
        assert_eq!(hle_set_stacksize(&ctx, &[attr, 0x20_0000]), OK);
        let out = 0x400;
        assert_eq!(hle_get_stacksize(&ctx, &[attr, out]), OK);
        let mut sz = [0u8; 8];
        assert!(mem.read(out, &mut sz));
        assert_eq!(u64::from_le_bytes(sz), 0x20_0000);
        // Detach state round-trips (1 = detached).
        assert_eq!(hle_set_detachstate(&ctx, &[attr, 1]), OK);
        let dout = 0x408;
        assert_eq!(hle_get_detachstate(&ctx, &[attr, dout]), OK);
        let mut ds = [0u8; 4];
        assert!(mem.read(dout, &mut ds));
        assert_eq!(i32::from_le_bytes(ds), 1);
    }

    #[test]
    fn get_on_unknown_attr_errors_and_destroy_clears() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Get before init → EINVAL (no state, and *attr reads 0).
        assert_eq!(hle_get_stacksize(&ctx, &[0x500, 0x600]), EINVAL);
        let attr = 0x700;
        hle_attr_init(&ctx, &[attr]);
        assert_eq!(hle_attr_destroy(&ctx, &[attr]), OK);
        assert!(!kernel.pthread_attrs.contains_key(&attr));
        let mut buf = [0u8; 8];
        assert!(mem.read(attr, &mut buf));
        assert_eq!(u64::from_le_bytes(buf), 0, "destroy zeroes the handle");
    }
}
