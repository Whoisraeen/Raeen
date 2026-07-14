//! HLE libkernel pthread **thread-specific data** (TLS keys).
//!
//! A faithful Rust port of SharpEmu's pthread-key TLS (GPL-2.0). A title
//! creates a key (`scePthreadKeyCreate`), stores a per-thread pointer under it
//! (`Setspecific`), and reads it back (`Getspecific`) — the mechanism libc and
//! runtimes use for thread-local storage. Under XPS5X's single-active-execution
//! model there is one guest thread, so the (thread, key) → value map is exactly
//! correct with **no runtime dependency** — this module ports completely.
//!
//! Key registry and the value map live in the kernel
//! (`OrbisKernel::pthread_tls_keys` / `pthread_tls_values`).

use crate::{HleContext, HleRegistry};
use tracing::debug;

const OK: u64 = 0;
const EINVAL: u64 = 22;

/// The single active guest thread's handle (single-active-execution model).
const CURRENT_THREAD: u64 = 1;

/// Register the pthread TLS-key HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "scePthreadKeyCreate", hle_key_create);
    registry.register("libkernel", "scePthreadKeyDelete", hle_key_delete);
    registry.register("libkernel", "scePthreadSetspecific", hle_setspecific);
    registry.register("libkernel", "scePthreadGetspecific", hle_getspecific);
}

/// `scePthreadKeyCreate(pthread_key_t *key, destructor)`: allocate a key and
/// write it into `*key`.
fn hle_key_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let out_key = args.first().copied().unwrap_or(0);
    let destructor = args.get(1).copied().unwrap_or(0);
    if out_key == 0 {
        return EINVAL;
    }
    let key = ctx.kernel.pthread_key_create(destructor);
    if !ctx.mem.write(out_key, &key.to_le_bytes()) {
        // Roll back the registration if we can't hand the key back.
        ctx.kernel.pthread_tls_keys.remove(&key);
        return EINVAL;
    }
    debug!("scePthreadKeyCreate -> key {key}, destructor {destructor:#x}");
    OK
}

/// `scePthreadKeyDelete(key)`: drop the key and every thread's value for it.
fn hle_key_delete(ctx: &HleContext, args: &[u64]) -> u64 {
    let key = args.first().copied().unwrap_or(0) as i32;
    if ctx.kernel.pthread_tls_keys.remove(&key).is_none() {
        return EINVAL;
    }
    // Remove any stored specific values for this key across all threads.
    ctx.kernel.pthread_tls_values.retain(|k, _| k.1 != key);
    OK
}

/// `scePthreadSetspecific(key, value)`: store `value` for the current thread.
fn hle_setspecific(ctx: &HleContext, args: &[u64]) -> u64 {
    let key = args.first().copied().unwrap_or(0) as i32;
    let value = args.get(1).copied().unwrap_or(0);
    if !ctx.kernel.pthread_tls_keys.contains_key(&key) {
        return EINVAL;
    }
    ctx.kernel
        .pthread_tls_values
        .insert((CURRENT_THREAD, key), value);
    OK
}

/// `scePthreadGetspecific(key)`: return the current thread's value (0 if unset
/// or the key is unknown). The value is the return value, per the ABI.
fn hle_getspecific(ctx: &HleContext, args: &[u64]) -> u64 {
    let key = args.first().copied().unwrap_or(0) as i32;
    if !ctx.kernel.pthread_tls_keys.contains_key(&key) {
        return 0;
    }
    ctx.kernel
        .pthread_tls_values
        .get(&(CURRENT_THREAD, key))
        .map(|v| *v)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    fn ctx_env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x800);
        (kernel, mem, alloc)
    }

    /// Create a key via the HLE path, returning the id written into guest memory.
    fn create_key(ctx: &HleContext, out: u64, destructor: u64) -> i32 {
        assert_eq!(hle_key_create(ctx, &[out, destructor]), OK);
        let mut buf = [0u8; 4];
        assert!(ctx.mem.read(out, &mut buf));
        i32::from_le_bytes(buf)
    }

    #[test]
    fn set_then_get_returns_the_stored_value() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let key = create_key(&ctx, 0x100, 0);
        // Unset → 0.
        assert_eq!(hle_getspecific(&ctx, &[key as u64]), 0);
        // Store then read back.
        assert_eq!(hle_setspecific(&ctx, &[key as u64, 0xDEAD_BEEF]), OK);
        assert_eq!(hle_getspecific(&ctx, &[key as u64]), 0xDEAD_BEEF);
    }

    #[test]
    fn distinct_keys_are_independent() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let k1 = create_key(&ctx, 0x100, 0);
        let k2 = create_key(&ctx, 0x104, 0);
        assert_ne!(k1, k2, "each create yields a fresh key");
        hle_setspecific(&ctx, &[k1 as u64, 111]);
        hle_setspecific(&ctx, &[k2 as u64, 222]);
        assert_eq!(hle_getspecific(&ctx, &[k1 as u64]), 111);
        assert_eq!(hle_getspecific(&ctx, &[k2 as u64]), 222);
    }

    #[test]
    fn setspecific_on_unknown_key_errors() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_setspecific(&ctx, &[999, 1]), EINVAL);
        // getspecific on an unknown key is a benign 0 (per the ABI).
        assert_eq!(hle_getspecific(&ctx, &[999]), 0);
    }

    #[test]
    fn delete_removes_the_key_and_its_values() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let key = create_key(&ctx, 0x100, 0);
        hle_setspecific(&ctx, &[key as u64, 42]);
        assert_eq!(hle_key_delete(&ctx, &[key as u64]), OK);
        // Value is gone and the key is unregistered.
        assert!(!kernel.pthread_tls_keys.contains_key(&key));
        assert_eq!(hle_getspecific(&ctx, &[key as u64]), 0);
        // Deleting an unknown key → EINVAL.
        assert_eq!(hle_key_delete(&ctx, &[key as u64]), EINVAL);
    }
}
