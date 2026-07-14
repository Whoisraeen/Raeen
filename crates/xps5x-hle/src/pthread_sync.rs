//! HLE libkernel pthread **mutex** synchronization.
//!
//! A faithful Rust port of the core mutex state machine from SharpEmu's
//! `KernelPthreadCompatExports` (GPL-2.0). A mutex's state (type, owner,
//! recursion count) lives in the kernel (`OrbisKernel::pthread_mutexes`),
//! keyed by both the guest `pthread_mutex_t` address and the opaque handle
//! `Init` allocates and writes into `*mutex`. `Lock`/`Unlock` honor per-type
//! semantics: a recursive mutex re-locks by count, an error-check mutex
//! reports `EDEADLK` on self-relock, and normal/adaptive mutexes are lenient.
//!
//! Under XPS5X's single-active-execution model there is exactly one guest
//! thread, so ownership reduces to recursion + type tracking — which is
//! precisely correct for that model. The blocking/contention path (real
//! waiters) and condition variables need the per-thread scheduler and are out
//! of scope here (they stay stubbed until M1-E stands the runtime up).

use crate::{HleContext, HleRegistry};
use tracing::debug;
use xps5x_kernel::PthreadMutex;

// Orbis `scePthreadMutex*` return POSIX errno directly (0 = success).
const OK: u64 = 0;
const EPERM: u64 = 1;
const EDEADLK: u64 = 11;
const EBUSY: u64 = 16;
const EINVAL: u64 = 22;

// pthread mutex types.
const MUTEX_ERRORCHECK: i32 = 1;
const MUTEX_RECURSIVE: i32 = 2;
const MUTEX_NORMAL: i32 = 3;
const MUTEX_ADAPTIVE: i32 = 4;

/// The single active guest thread's handle (single-active-execution model).
const CURRENT_THREAD: u64 = 1;

/// Size of the opaque mutex object handed to the guest.
const MUTEX_OBJECT_SIZE: u64 = 0x100;

/// Register the pthread mutex HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "scePthreadMutexInit", hle_mutex_init);
    registry.register("libkernel", "scePthreadMutexDestroy", hle_mutex_destroy);
    registry.register("libkernel", "scePthreadMutexLock", hle_mutex_lock);
    registry.register("libkernel", "scePthreadMutexTrylock", hle_mutex_trylock);
    registry.register("libkernel", "scePthreadMutexUnlock", hle_mutex_unlock);
    registry.register("libkernel", "scePthreadMutexattrInit", hle_mutexattr_init);
    registry.register(
        "libkernel",
        "scePthreadMutexattrDestroy",
        hle_mutexattr_destroy,
    );
    registry.register(
        "libkernel",
        "scePthreadMutexattrSettype",
        hle_mutexattr_settype,
    );
}

/// Normalize a caller-supplied mutex type to a known value (default: normal,
/// matching a zero-initialized `pthread_mutexattr_t`).
fn normalize_type(ty: i32) -> i32 {
    match ty {
        MUTEX_ERRORCHECK | MUTEX_RECURSIVE | MUTEX_NORMAL | MUTEX_ADAPTIVE => ty,
        _ => MUTEX_NORMAL,
    }
}

/// Resolve the mutex state key for a guest `pthread_mutex_t` address: the
/// address itself if registered, else the handle it points at if that is
/// registered. Returns `None` if neither is known.
fn resolve_key(ctx: &HleContext, mutex_addr: u64) -> Option<u64> {
    if ctx.kernel.pthread_mutexes.contains_key(&mutex_addr) {
        return Some(mutex_addr);
    }
    let mut buf = [0u8; 8];
    if ctx.mem.read(mutex_addr, &mut buf) {
        let handle = u64::from_le_bytes(buf);
        if handle != 0 && ctx.kernel.pthread_mutexes.contains_key(&handle) {
            return Some(handle);
        }
    }
    None
}

/// `scePthreadMutexInit(mutex, attr, name)`: allocate an opaque mutex object,
/// write its handle into `*mutex`, and register fresh state.
fn hle_mutex_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let mutex_addr = args.first().copied().unwrap_or(0);
    let attr_addr = args.get(1).copied().unwrap_or(0);
    if mutex_addr == 0 {
        return EINVAL;
    }

    // Resolve the type from the attribute object (default: normal).
    let ty = if attr_addr != 0 {
        ctx.kernel
            .pthread_mutex_attrs
            .get(&attr_addr)
            .map(|t| *t)
            .unwrap_or(MUTEX_NORMAL)
    } else {
        MUTEX_NORMAL
    };

    let Some(handle) = ctx.alloc.alloc(MUTEX_OBJECT_SIZE, 0x10) else {
        return EINVAL;
    };
    if !ctx.mem.write(mutex_addr, &handle.to_le_bytes()) {
        return EINVAL;
    }

    let state = PthreadMutex {
        ty,
        owner: 0,
        recursion: 0,
    };
    ctx.kernel.pthread_mutexes.insert(mutex_addr, state);
    ctx.kernel.pthread_mutexes.insert(handle, state);
    debug!("scePthreadMutexInit(mutex={mutex_addr:#x}) -> handle {handle:#x}, type {ty}");
    OK
}

/// `scePthreadMutexDestroy(mutex)`: drop the state and zero the guest handle.
fn hle_mutex_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let mutex_addr = args.first().copied().unwrap_or(0);
    if mutex_addr == 0 {
        return EINVAL;
    }
    let Some(key) = resolve_key(ctx, mutex_addr) else {
        return EINVAL;
    };
    ctx.kernel.pthread_mutexes.remove(&key);
    if key != mutex_addr {
        ctx.kernel.pthread_mutexes.remove(&mutex_addr);
    }
    let _ = ctx.mem.write(mutex_addr, &0u64.to_le_bytes());
    OK
}

fn hle_mutex_lock(ctx: &HleContext, args: &[u64]) -> u64 {
    lock_core(ctx, args.first().copied().unwrap_or(0), false)
}

fn hle_mutex_trylock(ctx: &HleContext, args: &[u64]) -> u64 {
    lock_core(ctx, args.first().copied().unwrap_or(0), true)
}

/// The lock state machine. With one active guest thread, a mutex either is
/// free (acquire it) or is already held by this thread (per-type re-lock
/// behavior). A missing mutex is created implicitly (guest static
/// initializers never call `Init`).
fn lock_core(ctx: &HleContext, mutex_addr: u64, try_only: bool) -> u64 {
    if mutex_addr == 0 {
        return EINVAL;
    }
    let key = resolve_key(ctx, mutex_addr).unwrap_or_else(|| {
        // Implicit creation for a statically-initialized mutex.
        ctx.kernel.pthread_mutexes.insert(
            mutex_addr,
            PthreadMutex {
                ty: MUTEX_NORMAL,
                owner: 0,
                recursion: 0,
            },
        );
        mutex_addr
    });

    let mut entry = ctx.kernel.pthread_mutexes.get_mut(&key).unwrap();
    if entry.owner == CURRENT_THREAD {
        // Already held by (the only) thread.
        match entry.ty {
            MUTEX_RECURSIVE => {
                entry.recursion += 1;
                OK
            }
            MUTEX_NORMAL | MUTEX_ADAPTIVE => {
                if try_only {
                    EBUSY
                } else {
                    // Lenient: real normal-mutex self-relock is UB; we bump.
                    entry.recursion += 1;
                    OK
                }
            }
            _ => {
                // Error-check: self-relock is a detected deadlock.
                if try_only {
                    EBUSY
                } else {
                    EDEADLK
                }
            }
        }
    } else {
        // Free → acquire.
        entry.owner = CURRENT_THREAD;
        entry.recursion = 1;
        OK
    }
}

/// `scePthreadMutexUnlock(mutex)`: drop one recursion level, releasing at zero.
fn hle_mutex_unlock(ctx: &HleContext, args: &[u64]) -> u64 {
    let mutex_addr = args.first().copied().unwrap_or(0);
    if mutex_addr == 0 {
        return EINVAL;
    }
    let Some(key) = resolve_key(ctx, mutex_addr) else {
        return EINVAL;
    };
    let mut entry = ctx.kernel.pthread_mutexes.get_mut(&key).unwrap();
    let lenient = matches!(entry.ty, MUTEX_NORMAL | MUTEX_ADAPTIVE);

    if entry.recursion <= 0 {
        // Not locked.
        return if lenient { OK } else { EINVAL };
    }
    if !lenient && entry.owner != CURRENT_THREAD {
        return EPERM;
    }
    entry.recursion -= 1;
    if entry.recursion == 0 {
        entry.owner = 0;
    }
    OK
}

/// `scePthreadMutexattrInit(attr)`: register a default (normal) attribute.
fn hle_mutexattr_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr_addr = args.first().copied().unwrap_or(0);
    if attr_addr == 0 {
        return EINVAL;
    }
    ctx.kernel
        .pthread_mutex_attrs
        .insert(attr_addr, MUTEX_NORMAL);
    OK
}

/// `scePthreadMutexattrDestroy(attr)`.
fn hle_mutexattr_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr_addr = args.first().copied().unwrap_or(0);
    if attr_addr == 0 {
        return EINVAL;
    }
    ctx.kernel.pthread_mutex_attrs.remove(&attr_addr);
    OK
}

/// `scePthreadMutexattrSettype(attr, type)`.
fn hle_mutexattr_settype(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr_addr = args.first().copied().unwrap_or(0);
    let ty = args.get(1).copied().unwrap_or(0) as i32;
    if attr_addr == 0 {
        return EINVAL;
    }
    ctx.kernel
        .pthread_mutex_attrs
        .insert(attr_addr, normalize_type(ty));
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_ctx, GuestMemory};

    /// Build a context with memory large enough for a mutex pointer slot plus
    /// the allocator's object arena (based well past the pointer slots).
    fn ctx_env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x1000); // objects live at 0x1000+
        (kernel, mem, alloc)
    }

    #[test]
    fn init_writes_a_handle_and_registers_state() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x100;
        assert_eq!(hle_mutex_init(&ctx, &[mutex, 0, 0]), OK);
        // A non-zero handle was written into *mutex.
        let mut buf = [0u8; 8];
        assert!(mem.read(mutex, &mut buf));
        let handle = u64::from_le_bytes(buf);
        assert!(handle != 0, "init must write an opaque handle");
        // State is registered under both the address and the handle.
        assert!(kernel.pthread_mutexes.contains_key(&mutex));
        assert!(kernel.pthread_mutexes.contains_key(&handle));
        // NULL mutex → EINVAL.
        assert_eq!(hle_mutex_init(&ctx, &[0, 0, 0]), EINVAL);
    }

    #[test]
    fn lock_then_unlock_balances() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x200;
        hle_mutex_init(&ctx, &[mutex, 0, 0]);
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);
        // After locking, the (only) thread owns it with recursion 1.
        let key = resolve_key(&ctx, mutex).unwrap();
        assert_eq!(kernel.pthread_mutexes.get(&key).unwrap().recursion, 1);
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
        assert_eq!(kernel.pthread_mutexes.get(&key).unwrap().recursion, 0);
        assert_eq!(kernel.pthread_mutexes.get(&key).unwrap().owner, 0);
        // Unlocking an already-free normal mutex is lenient (OK).
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
    }

    #[test]
    fn recursive_mutex_counts_nested_locks() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x80;
        hle_mutexattr_init(&ctx, &[attr]);
        assert_eq!(
            hle_mutexattr_settype(&ctx, &[attr, MUTEX_RECURSIVE as u64]),
            OK
        );
        let mutex = 0x300;
        hle_mutex_init(&ctx, &[mutex, attr, 0]);
        // Three nested locks → recursion 3; three unlocks to release.
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);
        let key = resolve_key(&ctx, mutex).unwrap();
        assert_eq!(kernel.pthread_mutexes.get(&key).unwrap().recursion, 3);
        for _ in 0..3 {
            assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
        }
        assert_eq!(kernel.pthread_mutexes.get(&key).unwrap().recursion, 0);
    }

    #[test]
    fn errorcheck_mutex_reports_deadlock_on_self_relock() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x88;
        hle_mutexattr_init(&ctx, &[attr]);
        hle_mutexattr_settype(&ctx, &[attr, MUTEX_ERRORCHECK as u64]);
        let mutex = 0x400;
        hle_mutex_init(&ctx, &[mutex, attr, 0]);
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);
        // Second lock by the same (only) thread → EDEADLK; trylock → EBUSY.
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), EDEADLK);
        assert_eq!(hle_mutex_trylock(&ctx, &[mutex]), EBUSY);
    }

    #[test]
    fn destroy_removes_state_and_zeroes_handle() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x500;
        hle_mutex_init(&ctx, &[mutex, 0, 0]);
        assert_eq!(hle_mutex_destroy(&ctx, &[mutex]), OK);
        let mut buf = [0u8; 8];
        assert!(mem.read(mutex, &mut buf));
        assert_eq!(
            u64::from_le_bytes(buf),
            0,
            "destroy zeroes the guest handle"
        );
        assert!(!kernel.pthread_mutexes.contains_key(&mutex));
        // Destroying an unknown mutex → EINVAL.
        assert_eq!(hle_mutex_destroy(&ctx, &[0x999]), EINVAL);
    }
}
