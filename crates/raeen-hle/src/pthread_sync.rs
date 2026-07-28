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
//! Ownership uses the runtime's current guest-thread handle. Contended locks
//! park cooperatively on private host wait objects and re-check process
//! termination. Unlock transfers ownership directly to the FIFO head while
//! holding the state lock, preventing a new arrival from barging ahead.

use crate::{HleContext, HleFunction, HleRegistry};
use raeen_kernel::{PthreadMutex, PthreadRwlock};
use std::sync::Arc;
use tracing::debug;

// The shared mutex state machine (`lock_core`) works in POSIX errno (0 =
// success); the `libScePosix` pthread_* entry points return these directly.
const OK: u64 = 0;
const EPERM: u64 = 1;
const EDEADLK: u64 = 11;
const EBUSY: u64 = 16;
const EINVAL: u64 = 22;
const ETIMEDOUT: u64 = 60;

// The `libkernel` `scePthreadMutex*` ABI returns SCE_KERNEL_ERROR_* codes
// (0x8002_00xx), NOT bare POSIX errno. The title's own libc `_Mtx_trylock`
// (C11 threads wrapper) maps SCE EBUSY -> _Thrd_busy(3) but treats a bare
// POSIX 16 as an error, constructing std::system_error(EINVAL) — which was
// UNCAUGHT and killed Minecraft's "Streaming Pool" asset threads (~22-frame
// unwind -> sceKernelDebugRaiseExceptionOnReleaseMode), stalling boot before
// the menu. Mirrors the scePthreadCondTimedwait SCE/POSIX split.
const SCE_KERNEL_ERROR_EPERM: u64 = 0x8002_0001;
const SCE_KERNEL_ERROR_EDEADLK: u64 = 0x8002_0023;
const SCE_KERNEL_ERROR_EBUSY: u64 = 0x8002_0010;
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;
const SCE_KERNEL_ERROR_ETIMEDOUT: u64 = 0x8002_003C;

/// Translate a POSIX errno from the shared state machine into the SCE error
/// code the `scePthreadMutex*` (libkernel) ABI returns. 0 stays 0.
fn posix_to_sce(err: u64) -> u64 {
    match err {
        OK => OK,
        EPERM => SCE_KERNEL_ERROR_EPERM,
        EDEADLK => SCE_KERNEL_ERROR_EDEADLK,
        EBUSY => SCE_KERNEL_ERROR_EBUSY,
        EINVAL => SCE_KERNEL_ERROR_EINVAL,
        ETIMEDOUT => SCE_KERNEL_ERROR_ETIMEDOUT,
        other => other,
    }
}

// pthread mutex types.
const MUTEX_ERRORCHECK: i32 = 1;
const MUTEX_RECURSIVE: i32 = 2;
const MUTEX_NORMAL: i32 = 3;
const MUTEX_ADAPTIVE: i32 = 4;

#[cfg(test)]
const CURRENT_THREAD: u64 = 1;

/// Publish only genuinely parked calls to the existing in-flight diagnostic
/// map. Dispatch-wide HLE timing is intentionally opt-in because formatting
/// and indexing every import distorts the hot path; a contended lock is rare
/// enough to record unconditionally and is exactly what a stall report needs.
struct InFlightWait<'a> {
    calls: &'a dashmap::DashMap<u64, String>,
    thread: u64,
    previous: Option<String>,
}

impl<'a> InFlightWait<'a> {
    fn new(calls: &'a dashmap::DashMap<u64, String>, thread: u64, description: String) -> Self {
        let previous = calls.insert(thread, description);
        Self {
            calls,
            thread,
            previous,
        }
    }
}

impl Drop for InFlightWait<'_> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            self.calls.insert(self.thread, previous);
        } else {
            self.calls.remove(&self.thread);
        }
    }
}

/// Size of the opaque mutex object handed to the guest.
const MUTEX_OBJECT_SIZE: u64 = 0x100;

/// Register the pthread mutex HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "scePthreadMutexInit", hle_mutex_init);
    registry.register("libkernel", "scePthreadMutexDestroy", hle_mutex_destroy);
    registry.register("libkernel", "scePthreadMutexLock", hle_sce_mutex_lock);
    registry.register("libkernel", "scePthreadMutexTrylock", hle_sce_mutex_trylock);
    registry.register(
        "libkernel",
        "scePthreadMutexTimedlock",
        hle_sce_mutex_timedlock,
    );
    registry.register("libkernel", "scePthreadMutexUnlock", hle_sce_mutex_unlock);
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
    registry.register(
        "libkernel",
        "scePthreadMutexattrSetprotocol",
        hle_mutexattr_setprotocol,
    );

    registry.register("libkernel", "scePthreadRwlockInit", hle_rwlock_init);
    registry.register("libkernel", "scePthreadRwlockDestroy", hle_rwlock_destroy);
    registry.register("libkernel", "scePthreadRwlockRdlock", hle_rwlock_rdlock);
    registry.register(
        "libkernel",
        "scePthreadRwlockTryrdlock",
        hle_rwlock_tryrdlock,
    );
    registry.register("libkernel", "scePthreadRwlockWrlock", hle_rwlock_wrlock);
    registry.register(
        "libkernel",
        "scePthreadRwlockTrywrlock",
        hle_rwlock_trywrlock,
    );
    registry.register("libkernel", "scePthreadRwlockUnlock", hle_rwlock_unlock);
    // Timed variants: `scePthreadRwlockTimed{rd,wr}lock(rwlock, usec)` — like
    // `scePthreadMutexTimedlock`, the second argument is a RELATIVE
    // `SceKernelUseconds` count passed by value, and timeout must surface as
    // the SCE code 0x8002003C (see the Minecraft `_Mtx_trylock` lesson above:
    // a bare POSIX 60 is unclassifiable by the title's own libc wrappers).
    registry.register(
        "libkernel",
        "scePthreadRwlockTimedrdlock",
        hle_sce_rwlock_timedrdlock,
    );
    registry.register(
        "libkernel",
        "scePthreadRwlockTimedwrlock",
        hle_sce_rwlock_timedwrlock,
    );
    registry.register("libkernel", "scePthreadRwlockattrInit", hle_rwlockattr_ok);
    registry.register(
        "libkernel",
        "scePthreadRwlockattrDestroy",
        hle_rwlockattr_ok,
    );

    register_posix(registry);
}

/// Register the **POSIX** spellings of the same functions through the two
/// provider names used by title and runtime modules.
///
/// # Why both spellings are needed
///
/// A NID hashes the function **name** alone, so `pthread_mutex_lock` and
/// `scePthreadMutexLock` are entirely different symbols with different NIDs.
/// Implementing only the Sony spelling leaves the POSIX one unresolved — and a
/// real title imports the POSIX ones: the measured retail eboot imports
/// `pthread_mutex_lock` and friends from `libScePosix`, and does **not** import
/// `scePthreadMutexLock` at all.
///
/// # Why aliasing is honest here (and not everywhere)
///
/// These implementations already use the POSIX convention — `0` on success, a
/// **positive** `errno` on failure (`EPERM`/`EDEADLK`/`EBUSY`/`EINVAL` above) —
/// which is exactly what `pthread_*` returns. So the same function pointer is
/// correct for both names with no conversion.
///
/// Deliberately **not** aliased, because no honest mapping exists yet:
///
/// * `pthread_create` / `pthread_join` / `pthread_detach` — Raeen has no second
///   guest execution context (M1-E). The existing `hle_pthread_create` returns
///   `1`, which under this ABI reads as `EPERM`, and never writes the out-param.
///   Wiring the POSIX name to it would swap a loud, self-identifying fault for a
///   guest that silently believes thread creation failed — or livelocks waiting
///   on a worker that never runs. An unresolved import names itself; a wrong
///   return value does not.
/// * `pthread_cond_*` / `pthread_condattr_*` — no implementation exists.
/// * `sem_*` — POSIX semaphores are address-based; `kernel_semaphore`'s are
///   handle-based. Different objects, not a rename.
fn register_posix(registry: &HleRegistry) {
    register_posix_abi(registry, "pthread_mutex_init", hle_mutex_init);
    register_posix_abi(registry, "pthread_mutex_destroy", hle_mutex_destroy);
    register_posix_abi(registry, "pthread_mutex_lock", hle_mutex_lock);
    register_posix_abi(registry, "pthread_mutex_trylock", hle_mutex_trylock);
    register_posix_abi(registry, "pthread_mutex_unlock", hle_mutex_unlock);
    register_posix_abi(registry, "pthread_mutexattr_init", hle_mutexattr_init);
    register_posix_abi(registry, "pthread_mutexattr_destroy", hle_mutexattr_destroy);
    register_posix_abi(registry, "pthread_mutexattr_settype", hle_mutexattr_settype);
    // `pthread_mutexattr_setprotocol` shares the sce body's ABI and return
    // convention (0 / positive errno), so the alias is exact — see
    // `hle_mutexattr_setprotocol` for why the protocol is validated but not
    // modelled. Measured missing from a retail import table.
    register_posix_abi(
        registry,
        "pthread_mutexattr_setprotocol",
        hle_mutexattr_setprotocol,
    );

    register_posix_abi(registry, "pthread_rwlock_init", hle_rwlock_init);
    register_posix_abi(registry, "pthread_rwlock_destroy", hle_rwlock_destroy);
    register_posix_abi(registry, "pthread_rwlock_rdlock", hle_rwlock_rdlock);
    // The POSIX try-variants must be the NON-blocking bodies. They were wired
    // to the blocking ones, so a guest probing a contended rwlock with
    // tryrdlock/trywrlock parked instead of getting EBUSY — a hang where the
    // guest expected a fast negative answer. (The SCE spellings above were
    // always correct, which is why this went unnoticed.)
    register_posix_abi(registry, "pthread_rwlock_tryrdlock", hle_rwlock_tryrdlock);
    register_posix_abi(registry, "pthread_rwlock_wrlock", hle_rwlock_wrlock);
    register_posix_abi(registry, "pthread_rwlock_trywrlock", hle_rwlock_trywrlock);
    register_posix_abi(registry, "pthread_rwlock_unlock", hle_rwlock_unlock);
}

fn register_posix_abi(registry: &HleRegistry, function: &str, implementation: HleFunction) {
    registry.register("libScePosix", function, implementation);
    registry.register("libkernel", function, implementation);
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

    let state = PthreadMutex::shared(ty);
    ctx.kernel
        .pthread_mutexes
        .insert(mutex_addr, Arc::clone(&state));
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
    let mut handle_bytes = [0u8; 8];
    let handle = ctx
        .mem
        .read(mutex_addr, &mut handle_bytes)
        .then(|| u64::from_le_bytes(handle_bytes))
        .filter(|handle| *handle != 0);
    let Some(key) = resolve_key(ctx, mutex_addr) else {
        return EINVAL;
    };
    ctx.kernel.pthread_mutexes.remove(&key);
    ctx.kernel.pthread_mutexes.remove(&mutex_addr);
    if let Some(handle) = handle {
        ctx.kernel.pthread_mutexes.remove(&handle);
        ctx.alloc.free(handle);
    }
    let _ = ctx.mem.write(mutex_addr, &0u64.to_le_bytes());
    OK
}

fn hle_mutex_lock(ctx: &HleContext, args: &[u64]) -> u64 {
    lock_core(ctx, args.first().copied().unwrap_or(0), false, None)
}

fn hle_mutex_trylock(ctx: &HleContext, args: &[u64]) -> u64 {
    lock_core(ctx, args.first().copied().unwrap_or(0), true, None)
}

/// SCE `scePthreadMutexTrylock` (libkernel): the shared state machine in SCE
/// error codes. Critically, a contended try returns SCE EBUSY (0x8002_0010),
/// which the title's libc `_Mtx_trylock` maps to `_Thrd_busy`; a bare POSIX 16
/// was mis-read as an error and thrown.
fn hle_sce_mutex_trylock(ctx: &HleContext, args: &[u64]) -> u64 {
    posix_to_sce(hle_mutex_trylock(ctx, args))
}

/// SCE `scePthreadMutexLock`/`scePthreadMutexTimedlock` (libkernel): SCE-coded.
/// The success path is unchanged (0); only error/timeout codes are translated.
fn hle_sce_mutex_lock(ctx: &HleContext, args: &[u64]) -> u64 {
    posix_to_sce(hle_mutex_lock(ctx, args))
}

fn hle_sce_mutex_timedlock(ctx: &HleContext, args: &[u64]) -> u64 {
    posix_to_sce(hle_mutex_timedlock(ctx, args))
}

/// SCE `scePthreadMutexUnlock` (libkernel): SCE-coded (success unchanged).
fn hle_sce_mutex_unlock(ctx: &HleContext, args: &[u64]) -> u64 {
    posix_to_sce(hle_mutex_unlock(ctx, args))
}

/// `scePthreadMutexTimedlock(mutex, usec)`: `scePthreadMutexLock` with a
/// relative microsecond timeout (`SceKernelUseconds`), returning `ETIMEDOUT`
/// if the mutex could not be acquired before the deadline. A timeout of 0
/// behaves like `Trylock` (an already-expired deadline).
fn hle_mutex_timedlock(ctx: &HleContext, args: &[u64]) -> u64 {
    let mutex = args.first().copied().unwrap_or(0);
    let usec = args.get(1).copied().unwrap_or(0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_micros(usec);
    lock_core(ctx, mutex, false, Some(deadline))
}

const MUTEX_CONTENTION_WARN_AFTER: std::time::Duration = std::time::Duration::from_secs(3);
const MUTEX_DEADLOCK_STABLE_OWNER_AFTER: std::time::Duration = std::time::Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MutexWaitDiagnostic {
    LongContention,
    ProbableDeadlock,
}

/// Classify one wait without equating ordinary lock contention with deadlock.
///
/// Minecraft's streaming pool can keep a waiter parked for several seconds
/// while ownership rotates through workers. That is a convoy: slow, but still
/// making progress. A probable deadlock requires one continuously observed
/// owner for a much longer interval.
fn classify_mutex_wait(
    total_wait: std::time::Duration,
    stable_owner_wait: std::time::Duration,
    reported_contention: bool,
    reported_deadlock: bool,
) -> Option<MutexWaitDiagnostic> {
    if !reported_deadlock && stable_owner_wait >= MUTEX_DEADLOCK_STABLE_OWNER_AFTER {
        Some(MutexWaitDiagnostic::ProbableDeadlock)
    } else if !reported_contention && total_wait >= MUTEX_CONTENTION_WARN_AFTER {
        Some(MutexWaitDiagnostic::LongContention)
    } else {
        None
    }
}

/// Sample the current instruction pointer of one guest thread for a rare
/// long-contention diagnostic. The handle is duplicated while its DashMap
/// guard is held, then the owner is suspended only for GetThreadContext and
/// resumed before any formatting, symbol lookup, or logging can run.
#[cfg(windows)]
fn sample_guest_thread_rip(kernel: &raeen_kernel::OrbisKernel, thread: u64) -> Option<u64> {
    use windows_sys::Win32::Foundation::{CloseHandle, DUPLICATE_SAME_ACCESS, DuplicateHandle};
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, GetThreadContext};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, ResumeThread, SuspendThread};

    const CONTEXT_CONTROL_AMD64: u32 = 0x0010_0001;

    #[repr(align(16))]
    struct Aligned(CONTEXT);

    let process = unsafe { GetCurrentProcess() };
    let mut duplicate = std::ptr::null_mut();
    {
        let source = kernel.host_thread_handles.get(&thread)?;
        let ok = unsafe {
            DuplicateHandle(
                process,
                *source.value() as *mut _,
                process,
                &mut duplicate,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        };
        if ok == 0 {
            return None;
        }
    }

    let handle = duplicate;
    unsafe {
        if SuspendThread(handle) == u32::MAX {
            CloseHandle(handle);
            return None;
        }
        let mut context: Aligned = std::mem::zeroed();
        context.0.ContextFlags = CONTEXT_CONTROL_AMD64;
        let rip = (GetThreadContext(handle, &mut context.0) != 0).then_some(context.0.Rip);
        // Resume and close before returning: the sampled thread must not be
        // left suspended on any path out of here.
        ResumeThread(handle);
        CloseHandle(handle);
        rip
    }
}

#[cfg(not(windows))]
fn sample_guest_thread_rip(_kernel: &raeen_kernel::OrbisKernel, _thread: u64) -> Option<u64> {
    None
}

fn format_guest_thread_site(kernel: &raeen_kernel::OrbisKernel, thread: u64) -> String {
    let Some(rip) = sample_guest_thread_rip(kernel, thread) else {
        return "<unavailable>".to_owned();
    };
    kernel.unwind_module_for_addr(rip).map_or_else(
        || format!("{rip:#x}"),
        |module| format!("{}+{:#x}", module.name, rip - module.start),
    )
}

/// Resolve a bounded set of guest return addresses retained in the owner's
/// acquisition frame. This runs only after three seconds of contention; doing
/// the same walk on every hot mutex acquisition would distort the title.
fn format_guest_acquire_stack(ctx: &HleContext, rsp: u64) -> String {
    if rsp == 0 {
        return "<unavailable>".to_owned();
    }
    let mut sites = Vec::new();
    for index in 0..128u64 {
        let mut bytes = [0u8; 8];
        if !ctx.mem.read(rsp.wrapping_add(index * 8), &mut bytes) {
            break;
        }
        let address = u64::from_le_bytes(bytes);
        let Some(module) = ctx.kernel.unwind_module_for_addr(address) else {
            continue;
        };
        let site = format!("{}+{:#x}", module.name, address - module.start);
        if sites.last() != Some(&site) {
            sites.push(site);
        }
        if sites.len() == 8 {
            break;
        }
    }
    if sites.is_empty() {
        "<no guest return addresses>".to_owned()
    } else {
        sites.join(" <- ")
    }
}

/// The lock state machine. With one active guest thread, a mutex either is
/// free (acquire it) or is already held by this thread (per-type re-lock
/// behavior). A missing mutex is created implicitly (guest static
/// initializers never call `Init`). A contended lock with a `deadline`
/// reports `ETIMEDOUT` once the deadline passes.
fn lock_core(
    ctx: &HleContext,
    mutex_addr: u64,
    try_only: bool,
    deadline: Option<std::time::Instant>,
) -> u64 {
    if mutex_addr == 0 {
        return EINVAL;
    }
    let key = if let Some(key) = resolve_key(ctx, mutex_addr) {
        key
    } else {
        // Implicit creation for a statically-initialized mutex, and it MUST be
        // atomic. Two guest threads first-touching the same static mutex both
        // miss `resolve_key`; a plain check-then-insert lets the second
        // `insert(mutex_addr, {owner: 0})` clobber the first thread's
        // freshly-taken ownership back to free, so the loop below then hands the
        // "mutex" to both threads and there is no mutual exclusion at all —
        // exactly when a title first spins up its worker pool on a shared static
        // lock. `entry` serializes the miss so the state is published once.
        // Mirrors `pthread_cond.rs::condition` (and defers the handle-alias
        // insert until the shard guard is dropped, so two colliding keys can't
        // self-deadlock).
        let state = PthreadMutex::shared(MUTEX_NORMAL);
        // Orbis mutexes are opaque pointer slots, so success must also
        // materialize *mutex; guest libc checks that pointer directly between
        // pthread calls. Only the thread that wins the vacant entry does this.
        let new_handle = match ctx.kernel.pthread_mutexes.entry(mutex_addr) {
            dashmap::mapref::entry::Entry::Occupied(_) => None,
            dashmap::mapref::entry::Entry::Vacant(slot) => {
                let Some(handle) = ctx.alloc.alloc(MUTEX_OBJECT_SIZE, 0x10) else {
                    return EINVAL;
                };
                if !ctx.mem.write(mutex_addr, &handle.to_le_bytes()) {
                    ctx.alloc.free(handle);
                    return EINVAL;
                }
                slot.insert(Arc::clone(&state));
                Some(handle)
            }
        };
        // Keep a fallback alias for callers that pass the opaque handle itself
        // instead of its slot — inserted only after the entry shard guard above
        // is dropped. The slot remains the canonical state key.
        if let Some(handle) = new_handle {
            ctx.kernel.pthread_mutexes.insert(handle, state);
        }
        mutex_addr
    };

    let current = ctx.guest_threads.current_thread();
    let spin_start = std::time::Instant::now();
    let Some(shared) = ctx
        .kernel
        .pthread_mutexes
        .get(&key)
        .map(|entry| Arc::clone(entry.value()))
    else {
        return EINVAL;
    };
    // Parked waiting, not spinning: the guard is held across the condvar
    // wait, which releases it while parked and reacquires on wake. The old
    // `yield_now()` spin loop burned a full host core per blocked guest
    // thread — Minecraft's seven in-game "Streaming Pool" workers spinning on
    // one mutex starved the owner itself down to 4 FPS.
    let (waiter, mut last_owner) = {
        let mut state = shared.state.lock();
        if state.owner == current {
            return match state.ty {
                MUTEX_RECURSIVE => {
                    state.recursion += 1;
                    OK
                }
                MUTEX_NORMAL | MUTEX_ADAPTIVE => {
                    if try_only {
                        EBUSY
                    } else {
                        state.recursion += 1;
                        OK
                    }
                }
                _ => {
                    if try_only {
                        EBUSY
                    } else {
                        EDEADLK
                    }
                }
            };
        }
        if state.owner == 0 && !state.has_waiters() {
            state.owner = current;
            state.recursion = 1;
            state.owner_acquire_site = ctx.caller_return_addr;
            state.owner_acquire_rsp = ctx.caller_rsp;
            return OK;
        }
        if try_only || ctx.guest_threads.process_is_terminating() {
            return EBUSY;
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return ETIMEDOUT;
        }
        let waiter = state.enqueue_waiter(current, ctx.caller_return_addr, ctx.caller_rsp);
        if state.try_grant_head() == Some(current) {
            return OK;
        }
        (waiter, state.owner)
    };
    let mut owner_since = std::time::Instant::now();
    let mut owner_changes = 0u64;
    let mut reported_contention = false;
    let mut reported_deadlock = false;
    let _in_flight_wait = InFlightWait::new(
        &ctx.kernel.in_flight_hle,
        current,
        format!("libkernel::scePthreadMutexLock(waiting mutex={key:#x})"),
    );
    loop {
        if waiter.wait_for_signal(std::time::Duration::from_millis(10)) {
            if reported_contention {
                tracing::info!(
                    mutex = format_args!("{key:#x}"),
                    waiter = current,
                    wait_ms = spin_start.elapsed().as_millis(),
                    owner_changes,
                    "scePthreadMutexLock acquired after long contention"
                );
            }
            return OK;
        }
        if ctx.guest_threads.process_is_terminating() {
            return abandon_mutex_wait(&shared, &waiter, EBUSY);
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return abandon_mutex_wait(&shared, &waiter, ETIMEDOUT);
        }

        let (owner, owner_acquire_site, owner_acquire_rsp, ty, recursion, queued) = {
            let state = shared.state.lock();
            (
                state.owner,
                state.owner_acquire_site,
                state.owner_acquire_rsp,
                state.ty,
                state.recursion,
                state.waiter_count(),
            )
        };
        let now = std::time::Instant::now();
        if owner != 0 && owner != last_owner {
            if last_owner != 0 {
                owner_changes += 1;
            }
            last_owner = owner;
            owner_since = now;
        }
        // Name long waits, but do not call every >3 s wait a deadlock.
        // Minecraft's streaming lock has measured rotating owners: a convoy
        // that eventually acquires. Only one owner observed continuously for
        // 30 s is elevated to the soak harness's probable-deadlock signal.
        let total_wait = spin_start.elapsed();
        let stable_owner_wait = owner_since.elapsed();
        if let Some(diagnostic) = classify_mutex_wait(
            total_wait,
            stable_owner_wait,
            reported_contention,
            reported_deadlock,
        ) {
            let owner_name = ctx
                .kernel
                .thread_names
                .get(&owner)
                .map_or_else(|| "<unnamed>".to_owned(), |n| n.clone());
            let self_name = ctx
                .kernel
                .thread_names
                .get(&current)
                .map_or_else(|| "<unnamed>".to_owned(), |n| n.clone());
            let owner_site = format_guest_thread_site(ctx.kernel, owner);
            let owner_acquired_at = ctx
                .kernel
                .unwind_module_for_addr(owner_acquire_site)
                .map_or_else(
                    || {
                        if owner_acquire_site == 0 {
                            "<unavailable>".to_owned()
                        } else {
                            format!("{owner_acquire_site:#x}")
                        }
                    },
                    |module| format!("{}+{:#x}", module.name, owner_acquire_site - module.start),
                );
            let owner_wait = ctx
                .kernel
                .in_flight_hle
                .get(&owner)
                .map_or_else(|| "<guest code>".to_owned(), |entry| entry.clone());
            let owner_acquire_stack = format_guest_acquire_stack(ctx, owner_acquire_rsp);
            match diagnostic {
                MutexWaitDiagnostic::LongContention => {
                    reported_contention = true;
                    tracing::warn!(
                        mutex = format_args!("{key:#x}"),
                        waiter = current,
                        waiter_name = %self_name,
                        owner,
                        owner_name = %owner_name,
                        owner_acquired_at = %owner_acquired_at,
                        owner_acquire_stack = %owner_acquire_stack,
                        owner_site = %owner_site,
                        owner_wait = %owner_wait,
                        ty,
                        recursion,
                        queued,
                        owner_changes,
                        "scePthreadMutexLock waiting >3s — long contention"
                    );
                }
                MutexWaitDiagnostic::ProbableDeadlock => {
                    reported_deadlock = true;
                    tracing::error!(
                        mutex = format_args!("{key:#x}"),
                        waiter = current,
                        waiter_name = %self_name,
                        owner,
                        owner_name = %owner_name,
                        owner_acquired_at = %owner_acquired_at,
                        owner_acquire_stack = %owner_acquire_stack,
                        owner_site = %owner_site,
                        owner_wait = %owner_wait,
                        ty,
                        recursion,
                        queued,
                        owner_changes,
                        owner_stable_ms = stable_owner_wait.as_millis(),
                        "scePthreadMutexLock waiting >30s with one owner — probable deadlock"
                    );
                }
            }
        }
        // Bounded so process termination and deadlines are observed even if
        // no unlock ever notifies; a wake races are re-checked by the loop.
    }
}

/// Leave the acquisition FIFO after timeout or termination. If the direct
/// handoff already dequeued this waiter, it owns the mutex and reports success.
fn abandon_mutex_wait(
    shared: &raeen_kernel::PthreadMutexShared,
    waiter: &Arc<raeen_kernel::GuestWaiter>,
    give_up: u64,
) -> u64 {
    let mut state = shared.state.lock();
    if !state.cancel_waiter(waiter) {
        return OK;
    }
    state.try_grant_head();
    give_up
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
    let Some(shared) = ctx
        .kernel
        .pthread_mutexes
        .get(&key)
        .map(|entry| Arc::clone(entry.value()))
    else {
        return EINVAL;
    };
    let mut state = shared.state.lock();
    let lenient = matches!(state.ty, MUTEX_NORMAL | MUTEX_ADAPTIVE);
    if state.recursion <= 0 {
        return if lenient { OK } else { EINVAL };
    }
    if state.owner != ctx.guest_threads.current_thread() {
        return EPERM;
    }
    state.recursion -= 1;
    if state.recursion == 0 {
        state.owner = 0;
        state.owner_acquire_site = 0;
        state.owner_acquire_rsp = 0;
        // Transfer ownership under the state lock and wake exactly the FIFO
        // head. Clearing then notifying allowed arrivals to barge ahead of the
        // selected waiter, starving Minecraft's streaming workers.
        state.try_grant_head();
    }
    OK
}

/// Condition-wait bridge: release the supplied mutex using the same owner
/// checks as the public pthread entry point.
pub(crate) fn mutex_unlock_for_cond(ctx: &HleContext, mutex: u64) -> u64 {
    hle_mutex_unlock(ctx, &[mutex])
}

/// Condition-wait bridge: reacquire the supplied mutex before returning to
/// guest code.
pub(crate) fn mutex_lock_for_cond(ctx: &HleContext, mutex: u64) -> u64 {
    lock_core(ctx, mutex, false, None)
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

/// `scePthreadMutexattrSetprotocol(attr, protocol)`: select the mutex's
/// priority protocol — `PTHREAD_PRIO_NONE` (0), `PTHREAD_PRIO_INHERIT` (1), or
/// `PTHREAD_PRIO_PROTECT` (2).
///
/// The protocol is validated and accepted but deliberately **not modelled**:
/// Raeen maps guest mutexes onto host mutexes, which expose no priority
/// inheritance. That choice only affects scheduling latency under contention —
/// never mutual exclusion — so ignoring it cannot corrupt guest state, unlike
/// faking a handle would. An out-of-range protocol is still rejected so a guest
/// that checks the ABI sees POSIX behaviour.
///
/// Measured: Until Dawn calls this during early thread setup and stops dead
/// when it is unresolved.
fn hle_mutexattr_setprotocol(_ctx: &HleContext, args: &[u64]) -> u64 {
    let attr_addr = args.first().copied().unwrap_or(0);
    let protocol = args.get(1).copied().unwrap_or(0) as i32;
    if attr_addr == 0 || !(0..=2).contains(&protocol) {
        return EINVAL;
    }
    OK
}

/// Size of the opaque rwlock object handed to the guest.
const RWLOCK_OBJECT_SIZE: u64 = 0x100;

/// Resolve the rwlock state key for a guest `pthread_rwlock_t` address.
fn resolve_rwlock_key(ctx: &HleContext, addr: u64) -> Option<u64> {
    if let Some(state) = ctx.kernel.pthread_rwlocks.get(&addr) {
        return Some(state.state.lock().key);
    }
    let mut buf = [0u8; 8];
    if ctx.mem.read(addr, &mut buf) {
        let handle = u64::from_le_bytes(buf);
        if handle != 0
            && let Some(state) = ctx.kernel.pthread_rwlocks.get(&handle)
        {
            return Some(state.state.lock().key);
        }
    }
    None
}

/// `scePthreadRwlockInit(rwlock, attr)`: allocate an opaque object, write its
/// handle into `*rwlock`, and register fresh (unlocked) state.
fn hle_rwlock_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    if addr == 0 {
        return EINVAL;
    }
    let Some(handle) = ctx.alloc.alloc(RWLOCK_OBJECT_SIZE, 0x10) else {
        return EINVAL;
    };
    if !ctx.mem.write(addr, &handle.to_le_bytes()) {
        return EINVAL;
    }
    let state = PthreadRwlock::shared(addr);
    ctx.kernel.pthread_rwlocks.insert(addr, Arc::clone(&state));
    ctx.kernel.pthread_rwlocks.insert(handle, state);
    debug!("scePthreadRwlockInit(rwlock={addr:#x}) -> handle {handle:#x}");
    OK
}

/// `scePthreadRwlockDestroy(rwlock)`.
fn hle_rwlock_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    if addr == 0 {
        return EINVAL;
    }
    let Some(key) = resolve_rwlock_key(ctx, addr) else {
        return EINVAL;
    };
    let Some(state) = ctx
        .kernel
        .pthread_rwlocks
        .get(&key)
        .map(|entry| Arc::clone(entry.value()))
    else {
        return EINVAL;
    };
    let aliases: Vec<u64> = ctx
        .kernel
        .pthread_rwlocks
        .iter()
        .filter(|entry| Arc::ptr_eq(entry.value(), &state))
        .map(|entry| *entry.key())
        .collect();
    for alias in aliases {
        ctx.kernel.pthread_rwlocks.remove(&alias);
    }
    // Drop any per-thread read-hold accounting for this rwlock too, so a
    // destroyed-and-recycled address can't inherit stale read depths.
    ctx.kernel
        .pthread_rwlock_read_holds
        .retain(|(_, k), _| *k != key);
    let mut handle_bytes = [0u8; 8];
    let handle = if ctx.mem.read(key, &mut handle_bytes) {
        u64::from_le_bytes(handle_bytes)
    } else {
        0
    };
    if handle != 0 {
        ctx.alloc.free(handle);
    }
    let _ = ctx.mem.write(key, &0u64.to_le_bytes());
    OK
}

/// Resolve (creating implicitly for static initializers) the rwlock state key.
fn resolve_or_create_rwlock(ctx: &HleContext, addr: u64) -> u64 {
    if let Some(key) = resolve_rwlock_key(ctx, addr) {
        return key;
    }
    // Implicit creation for a statically-initialized rwlock, and it MUST be
    // atomic. Two guest threads first-touching the same static rwlock (typically
    // a reader and a writer) both miss `resolve_rwlock_key`; a plain
    // check-then-insert lets the second `insert(default)` clobber the first
    // thread's freshly-taken hold (writer/readers reset to 0), so a reader and a
    // writer both "acquire" and the writer rehashes the guest's container while
    // the reader is still walking it — the observed simultaneous null/dangling
    // bucket-walk fault on the reader threads. `entry().or_insert_with` publishes
    // the state exactly once and never overwrites an existing hold. Mirrors
    // `pthread_cond.rs::condition`.
    ctx.kernel
        .pthread_rwlocks
        .entry(addr)
        .or_insert_with(|| PthreadRwlock::shared(addr));
    addr
}

/// `scePthreadRwlockRdlock`/`Tryrdlock(rwlock)`: add a read hold. With one
/// guest thread a read lock never blocks; it nests by reader count.
fn hle_rwlock_rdlock(ctx: &HleContext, args: &[u64]) -> u64 {
    rwlock_rdlock_core(ctx, args, false)
}

fn hle_rwlock_tryrdlock(ctx: &HleContext, args: &[u64]) -> u64 {
    rwlock_rdlock_core(ctx, args, true)
}

fn rwlock_rdlock_core(ctx: &HleContext, args: &[u64], try_only: bool) -> u64 {
    rwlock_rdlock_deadline(ctx, args.first().copied().unwrap_or(0), try_only, None)
}

fn rwlock_rdlock_deadline(
    ctx: &HleContext,
    addr: u64,
    try_only: bool,
    deadline: Option<std::time::Instant>,
) -> u64 {
    if addr == 0 {
        return EINVAL;
    }
    let key = resolve_or_create_rwlock(ctx, addr);
    let current = ctx.guest_threads.current_thread();
    let Some(shared) = ctx
        .kernel
        .pthread_rwlocks
        .get(&key)
        .map(|entry| Arc::clone(entry.value()))
    else {
        return EINVAL;
    };
    // Parked, not spinning — see the note in `lock_core`.
    let mut state = shared.state.lock();
    loop {
        if state.writer == 0 || state.writer == current {
            state.readers += 1;
            drop(state);
            // Record this thread's read hold so a later unlock can prove the
            // caller actually owns one before releasing it (see
            // `hle_rwlock_unlock`).
            *ctx.kernel
                .pthread_rwlock_read_holds
                .entry((current, key))
                .or_insert(0) += 1;
            return OK;
        }
        if try_only || ctx.guest_threads.process_is_terminating() {
            return EBUSY;
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return ETIMEDOUT;
        }
        let _ = shared
            .released
            .wait_for(&mut state, std::time::Duration::from_millis(10));
    }
}

/// `scePthreadRwlockWrlock`/`Trywrlock(rwlock)`: acquire (or recurse) the write
/// hold. Free → own it; already write-owned by this thread → recurse.
fn hle_rwlock_wrlock(ctx: &HleContext, args: &[u64]) -> u64 {
    rwlock_wrlock_core(ctx, args, false)
}

fn hle_rwlock_trywrlock(ctx: &HleContext, args: &[u64]) -> u64 {
    rwlock_wrlock_core(ctx, args, true)
}

fn rwlock_wrlock_core(ctx: &HleContext, args: &[u64], try_only: bool) -> u64 {
    rwlock_wrlock_deadline(ctx, args.first().copied().unwrap_or(0), try_only, None)
}

fn rwlock_wrlock_deadline(
    ctx: &HleContext,
    addr: u64,
    try_only: bool,
    deadline: Option<std::time::Instant>,
) -> u64 {
    if addr == 0 {
        return EINVAL;
    }
    let key = resolve_or_create_rwlock(ctx, addr);
    let current = ctx.guest_threads.current_thread();
    let Some(shared) = ctx
        .kernel
        .pthread_rwlocks
        .get(&key)
        .map(|entry| Arc::clone(entry.value()))
    else {
        return EINVAL;
    };
    // Parked, not spinning — see the note in `lock_core`.
    let mut state = shared.state.lock();
    loop {
        if state.writer == current {
            state.writer_recursion += 1;
            return OK;
        }
        if state.writer == 0 && state.readers == 0 {
            state.writer = current;
            state.writer_recursion = 1;
            return OK;
        }
        if try_only || ctx.guest_threads.process_is_terminating() {
            return EBUSY;
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return ETIMEDOUT;
        }
        let _ = shared
            .released
            .wait_for(&mut state, std::time::Duration::from_millis(10));
    }
}

/// `scePthreadRwlockTimedrdlock(rwlock, SceKernelUseconds usec)`: a read lock
/// bounded by a relative microsecond timeout. `usec == 0` behaves like an
/// already-expired deadline (a Tryrdlock whose failure code is ETIMEDOUT, per
/// the mutex Timedlock convention above). Errors are SCE-coded.
fn hle_sce_rwlock_timedrdlock(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let usec = args.get(1).copied().unwrap_or(0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_micros(usec);
    posix_to_sce(rwlock_rdlock_deadline(ctx, addr, false, Some(deadline)))
}

/// `scePthreadRwlockTimedwrlock(rwlock, SceKernelUseconds usec)` — the write
/// twin of [`hle_sce_rwlock_timedrdlock`].
fn hle_sce_rwlock_timedwrlock(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let usec = args.get(1).copied().unwrap_or(0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_micros(usec);
    posix_to_sce(rwlock_wrlock_deadline(ctx, addr, false, Some(deadline)))
}

/// `scePthreadRwlockUnlock(rwlock)`: release the thread's write hold (recursion
/// first) or one read hold; `EPERM` if it holds neither.
fn hle_rwlock_unlock(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    if addr == 0 {
        return EINVAL;
    }
    let Some(key) = resolve_rwlock_key(ctx, addr) else {
        return EINVAL;
    };
    let current = ctx.guest_threads.current_thread();

    // Release the most-recently-acquired hold first (LIFO). The only way a
    // thread holds both a write hold and a read hold is `wrlock` then `rdlock`
    // (a writer taking a reentrant read, admitted in `rwlock_rdlock_deadline`),
    // so a read hold is always newer than the write hold and must be dropped
    // first. Dropping the write hold first would clear `writer` while
    // `readers >= 1`, silently downgrading an intended-exclusive section to a
    // shared one — a second reader could then observe data mid-write. This
    // mirrors KytyPS5 (`RwlockRemoveReader` first).
    //
    // Release one of THIS thread's read holds — never another thread's. The
    // shared `readers` count cannot say who holds a read, so the per-(thread,
    // rwlock) depth map is the ownership check the bare count can't provide.
    let read_hold_drained = ctx
        .kernel
        .pthread_rwlock_read_holds
        .get_mut(&(current, key))
        .map(|mut depth| {
            *depth -= 1;
            *depth == 0
        });
    if let Some(drained) = read_hold_drained {
        if drained {
            ctx.kernel.pthread_rwlock_read_holds.remove(&(current, key));
        }
        if let Some(shared) = ctx
            .kernel
            .pthread_rwlocks
            .get(&key)
            .map(|entry| Arc::clone(entry.value()))
        {
            let mut state = shared.state.lock();
            if state.readers > 0 {
                state.readers -= 1;
            }
            let drained = state.readers == 0;
            drop(state);
            if drained {
                // Last reader out: a parked writer can take it now.
                shared.released.notify_all();
            }
        }
        return OK;
    }

    // No read hold: release the write hold (recursion first). A stray or
    // duplicated unlock from a thread that holds neither is rejected (EPERM)
    // rather than letting a writer in behind a live reader's back.
    let Some(shared) = ctx
        .kernel
        .pthread_rwlocks
        .get(&key)
        .map(|entry| Arc::clone(entry.value()))
    else {
        return EINVAL;
    };
    let mut state = shared.state.lock();
    if state.writer == current && state.writer_recursion > 0 {
        state.writer_recursion -= 1;
        let released = state.writer_recursion == 0;
        if released {
            state.writer = 0;
        }
        drop(state);
        if released {
            // Writers are exclusive but readers are not — wake everyone and
            // let the loops re-check.
            shared.released.notify_all();
        }
        return OK;
    }
    EPERM
}

/// `scePthreadRwlockattrInit`/`Destroy`: accepted (no attribute state modelled).
fn hle_rwlockattr_ok(_ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) == 0 {
        return EINVAL;
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    /// Build a context with memory large enough for a mutex pointer slot plus
    /// the allocator's object arena (based well past the pointer slots).
    fn ctx_env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x1000); // objects live at 0x1000+
        (kernel, mem, alloc)
    }

    #[test]
    fn mutex_wait_diagnostic_distinguishes_contention_from_a_stable_owner() {
        use std::time::Duration;

        assert_eq!(
            classify_mutex_wait(Duration::from_secs(3), Duration::from_secs(3), false, false,),
            Some(MutexWaitDiagnostic::LongContention),
        );
        assert_eq!(
            classify_mutex_wait(
                Duration::from_secs(29),
                Duration::from_secs(29),
                true,
                false,
            ),
            None,
        );
        assert_eq!(
            classify_mutex_wait(
                Duration::from_secs(30),
                Duration::from_secs(30),
                true,
                false,
            ),
            Some(MutexWaitDiagnostic::ProbableDeadlock),
        );
        assert_eq!(
            classify_mutex_wait(Duration::from_secs(60), Duration::from_secs(2), true, false,),
            None,
            "a waiter that observed a recent owner change is in a convoy, not a deadlock",
        );
    }

    #[test]
    fn parked_wait_diagnostic_restores_the_dispatch_entry() {
        let calls = dashmap::DashMap::new();
        calls.insert(7, "libkernel::outer".to_owned());
        {
            let _wait = InFlightWait::new(
                &calls,
                7,
                "libkernel::scePthreadMutexLock(waiting mutex=0x1234)".to_owned(),
            );
            assert_eq!(
                calls.get(&7).as_deref().map(String::as_str),
                Some("libkernel::scePthreadMutexLock(waiting mutex=0x1234)")
            );
        }
        assert_eq!(
            calls.get(&7).as_deref().map(String::as_str),
            Some("libkernel::outer")
        );

        {
            let _wait = InFlightWait::new(&calls, 8, "libkernel::scePthreadMutexLock".to_owned());
            assert!(calls.contains_key(&8));
        }
        assert!(
            !calls.contains_key(&8),
            "a wait with no outer timed call must remove its temporary entry"
        );
    }

    #[test]
    fn posix_pthread_names_are_available_through_both_abi_providers() {
        let registry = HleRegistry::new();
        for provider in ["libScePosix", "libkernel"] {
            for name in [
                "pthread_mutex_lock",
                "pthread_mutex_unlock",
                "pthread_rwlock_rdlock",
                "pthread_rwlock_wrlock",
                "pthread_rwlock_unlock",
                "pthread_mutexattr_setprotocol",
            ] {
                assert!(
                    registry.is_implemented(provider, name),
                    "missing {provider}::{name}"
                );
            }
        }
    }

    #[test]
    fn mutexattr_setprotocol_validates_the_protocol_range() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x200;
        assert_eq!(hle_mutexattr_init(&ctx, &[attr]), OK);
        // PRIO_NONE / PRIO_INHERIT / PRIO_PROTECT are all accepted...
        for protocol in [0u64, 1, 2] {
            assert_eq!(hle_mutexattr_setprotocol(&ctx, &[attr, protocol]), OK);
        }
        // ...and anything out of range is POSIX EINVAL.
        assert_eq!(hle_mutexattr_setprotocol(&ctx, &[attr, 3]), EINVAL);
        assert_eq!(hle_mutexattr_setprotocol(&ctx, &[0, 1]), EINVAL);
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
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&key)
                .unwrap()
                .state
                .lock()
                .recursion,
            1
        );
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&key)
                .unwrap()
                .state
                .lock()
                .recursion,
            0
        );
        assert_eq!(
            kernel.pthread_mutexes.get(&key).unwrap().state.lock().owner,
            0
        );
        // Unlocking an already-free normal mutex is lenient (OK).
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
    }

    #[test]
    fn slot_and_opaque_handle_share_one_mutex_state() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x280;
        assert_eq!(hle_mutex_init(&ctx, &[mutex, 0, 0]), OK);

        let mut bytes = [0u8; 8];
        assert!(mem.read(mutex, &mut bytes));
        let handle = u64::from_le_bytes(bytes);
        assert_ne!(handle, 0);

        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);
        assert_eq!(
            hle_mutex_trylock(&ctx, &[handle]),
            EBUSY,
            "the slot and its opaque handle must not expose independent locks"
        );
        assert_eq!(hle_mutex_unlock(&ctx, &[handle]), OK);
        assert_eq!(
            hle_mutex_unlock(&ctx, &[mutex]),
            OK,
            "unlock through the handle must release the slot-visible state"
        );
    }

    #[test]
    fn static_mutex_lock_materializes_and_destroy_clears_opaque_handle() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x300;

        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);
        let mut bytes = [0u8; 8];
        assert!(mem.read(mutex, &mut bytes));
        let handle = u64::from_le_bytes(bytes);
        assert_ne!(handle, 0);
        assert!(kernel.pthread_mutexes.contains_key(&mutex));
        assert!(kernel.pthread_mutexes.contains_key(&handle));
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
        assert_eq!(hle_mutex_destroy(&ctx, &[mutex]), OK);
        assert!(mem.read(mutex, &mut bytes));
        assert_eq!(u64::from_le_bytes(bytes), 0);
        assert!(!kernel.pthread_mutexes.contains_key(&mutex));
        assert!(!kernel.pthread_mutexes.contains_key(&handle));
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
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&key)
                .unwrap()
                .state
                .lock()
                .recursion,
            3
        );
        for _ in 0..3 {
            assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
        }
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&key)
                .unwrap()
                .state
                .lock()
                .recursion,
            0
        );
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

    /// The shared kernel state behind a guest mutex, for driving the waiter FIFO
    /// directly. All handoff tests below are deterministic: no host threads and
    /// no sleeps — waiters are enqueued through the queue API and the assertion
    /// is on the per-waiter wake bit plus `waiter_count`.
    fn mutex_state(ctx: &HleContext, mutex: u64) -> Arc<raeen_kernel::PthreadMutexShared> {
        let key = resolve_key(ctx, mutex).expect("mutex is registered");
        ctx.kernel
            .pthread_mutexes
            .get(&key)
            .map(|entry| Arc::clone(entry.value()))
            .expect("mutex state exists")
    }

    /// The regression this port exists for. Unlock must **hand ownership over**
    /// to the FIFO head while holding the state lock, not clear `owner` and hope
    /// the woken waiter wins the reacquire against every arriving thread.
    #[test]
    fn unlock_hands_ownership_to_the_head_waiter_and_wakes_only_it() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x600;
        hle_mutex_init(&ctx, &[mutex, 0, 0]);
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);

        let shared = mutex_state(&ctx, mutex);
        let (first, second) = {
            let mut state = shared.state.lock();
            (
                state.enqueue_waiter(0x77, 0, 0),
                state.enqueue_waiter(0x88, 0, 0),
            )
        };

        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);

        let state = shared.state.lock();
        assert_eq!(
            state.owner, 0x77,
            "the mutex must be OWNED by the head waiter after unlock, never free"
        );
        assert_eq!(state.recursion, 1);
        assert!(first.is_signaled());
        assert!(
            !second.is_signaled(),
            "handoff wakes exactly the granted waiter, not the whole queue"
        );
        assert_eq!(state.waiter_count(), 1);
    }

    /// The four mutex types keep their unlock semantics under handoff: a
    /// RECURSIVE mutex only hands off at the last level.
    #[test]
    fn recursive_unlock_hands_off_only_at_the_final_level() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x90;
        hle_mutexattr_init(&ctx, &[attr]);
        hle_mutexattr_settype(&ctx, &[attr, MUTEX_RECURSIVE as u64]);
        let mutex = 0x610;
        hle_mutex_init(&ctx, &[mutex, attr, 0]);
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);

        let shared = mutex_state(&ctx, mutex);
        let waiter = shared.state.lock().enqueue_waiter(0x77, 0, 0);

        // Inner unlock: still ours, nobody granted anything.
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
        {
            let state = shared.state.lock();
            assert_eq!(state.owner, CURRENT_THREAD);
            assert_eq!(state.recursion, 1);
        }
        assert!(
            !waiter.is_signaled(),
            "a nested unlock must not hand the mutex away"
        );

        // Outer unlock: handoff.
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
        assert!(waiter.is_signaled());
        assert_eq!(shared.state.lock().owner, 0x77);
    }

    /// The lenient NORMAL/ADAPTIVE self-relock still counts recursion, and each
    /// level must be unwound before the queue is served.
    #[test]
    fn lenient_normal_self_relock_still_unwinds_before_handoff() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x620;
        hle_mutex_init(&ctx, &[mutex, 0, 0]);
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);
        // NORMAL self-relock is deliberately lenient rather than EDEADLK.
        assert_eq!(hle_mutex_lock(&ctx, &[mutex]), OK);

        let shared = mutex_state(&ctx, mutex);
        let waiter = shared.state.lock().enqueue_waiter(0x77, 0, 0);
        assert_eq!(shared.state.lock().recursion, 2);

        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
        assert!(!waiter.is_signaled());
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);
        assert!(waiter.is_signaled());
        assert_eq!(shared.state.lock().owner, 0x77);

        // The lenient `recursion <= 0` case survives: unlocking a NORMAL mutex
        // that is not held is OK, and must not disturb the new owner.
        let mutex2 = 0x628;
        hle_mutex_init(&ctx, &[mutex2, 0, 0]);
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex2]), OK);
    }

    /// `EPERM` on a non-owner unlock, and the queued waiter must not be granted
    /// anything by the rejected call.
    #[test]
    fn unlock_from_a_non_owner_is_eperm_and_grants_nothing() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x630;
        hle_mutex_init(&ctx, &[mutex, 0, 0]);
        let shared = mutex_state(&ctx, mutex);
        let waiter = {
            let mut state = shared.state.lock();
            state.owner = 0x99; // some other guest thread holds it
            state.recursion = 1;
            state.enqueue_waiter(0x77, 0, 0)
        };

        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), EPERM);
        assert_eq!(hle_sce_mutex_unlock(&ctx, &[mutex]), SCE_KERNEL_ERROR_EPERM);
        let state = shared.state.lock();
        assert_eq!(state.owner, 0x99, "a rejected unlock changes nothing");
        assert!(!waiter.is_signaled());
        assert_eq!(state.waiter_count(), 1);
    }

    /// Anti-barging: a free mutex with a queued waiter belongs to that waiter, so
    /// an arriving `Trylock` must report EBUSY rather than jump the queue.
    #[test]
    fn trylock_does_not_barge_past_a_queued_waiter() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x640;
        hle_mutex_init(&ctx, &[mutex, 0, 0]);
        // Free and unqueued: trylock succeeds, as always.
        assert_eq!(hle_mutex_trylock(&ctx, &[mutex]), OK);
        assert_eq!(hle_mutex_unlock(&ctx, &[mutex]), OK);

        let shared = mutex_state(&ctx, mutex);
        let waiter = shared.state.lock().enqueue_waiter(0x77, 0, 0);
        // Free, but someone is queued ahead of us.
        assert_eq!(shared.state.lock().owner, 0);
        assert_eq!(hle_mutex_trylock(&ctx, &[mutex]), EBUSY);
        assert!(
            !waiter.is_signaled(),
            "a refused trylock must not consume the queue"
        );
    }

    /// Owner-death recovery still works with a queue attached: the dead owner's
    /// mutex is handed to its head waiter, not merely marked free (which the
    /// anti-barging path would then refuse to anyone, wedging the mutex).
    #[test]
    fn owner_death_hands_the_mutex_to_the_head_waiter() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mutex = 0x650;
        hle_mutex_init(&ctx, &[mutex, 0, 0]);
        let shared = mutex_state(&ctx, mutex);
        let waiter = {
            let mut state = shared.state.lock();
            state.owner = 0xdead;
            state.recursion = 3;
            state.enqueue_waiter(0x77, 0, 0)
        };

        let summary = kernel.release_locks_owned_by(0xdead);
        assert_eq!(summary.mutexes, 1);
        assert!(waiter.is_signaled());
        let state = shared.state.lock();
        assert_eq!(state.owner, 0x77);
        assert_eq!(state.recursion, 1);
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

    /// The timed rwlock variants honor their relative-microsecond timeout:
    /// free → immediate acquire; held by another thread → SCE ETIMEDOUT
    /// (0x8002003C) once the deadline passes, never a bare POSIX 60.
    #[test]
    fn rwlock_timed_variants_acquire_or_time_out_with_sce_codes() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let rw = 0x300u64;
        assert_eq!(hle_rwlock_init(&ctx, &[rw, 0]), OK);

        // Uncontended: both timed locks succeed immediately, even with usec=0.
        assert_eq!(hle_sce_rwlock_timedrdlock(&ctx, &[rw, 0]), OK);
        assert_eq!(hle_rwlock_unlock(&ctx, &[rw]), OK);
        assert_eq!(hle_sce_rwlock_timedwrlock(&ctx, &[rw, 1000]), OK);
        assert_eq!(hle_rwlock_unlock(&ctx, &[rw]), OK);

        // Contended by ANOTHER thread (id 99 ≠ the test's current thread 1):
        // the wait expires and reports the SCE timeout code.
        let key = resolve_rwlock_key(&ctx, rw).unwrap();
        {
            let state = kernel.pthread_rwlocks.get(&key).unwrap();
            let mut state = state.state.lock();
            state.writer = 99;
            state.writer_recursion = 1;
        }
        let started = std::time::Instant::now();
        assert_eq!(
            hle_sce_rwlock_timedrdlock(&ctx, &[rw, 2000]),
            SCE_KERNEL_ERROR_ETIMEDOUT
        );
        assert_eq!(
            hle_sce_rwlock_timedwrlock(&ctx, &[rw, 2000]),
            SCE_KERNEL_ERROR_ETIMEDOUT
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "a timed lock must not spin unbounded"
        );

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libkernel", "scePthreadRwlockTimedrdlock"));
        assert!(registry.is_implemented("libkernel", "scePthreadRwlockTimedwrlock"));
    }

    #[test]
    fn rwlock_read_holds_nest_and_balance() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let rw = 0x600;
        assert_eq!(hle_rwlock_init(&ctx, &[rw, 0]), OK);
        // Two read locks nest; two unlocks balance.
        assert_eq!(hle_rwlock_rdlock(&ctx, &[rw]), OK);
        assert_eq!(hle_rwlock_rdlock(&ctx, &[rw]), OK);
        let key = resolve_rwlock_key(&ctx, rw).unwrap();
        assert_eq!(
            kernel
                .pthread_rwlocks
                .get(&key)
                .unwrap()
                .state
                .lock()
                .readers,
            2
        );
        assert_eq!(hle_rwlock_unlock(&ctx, &[rw]), OK);
        assert_eq!(hle_rwlock_unlock(&ctx, &[rw]), OK);
        assert_eq!(
            kernel
                .pthread_rwlocks
                .get(&key)
                .unwrap()
                .state
                .lock()
                .readers,
            0
        );
        // Unlocking with nothing held → EPERM.
        assert_eq!(hle_rwlock_unlock(&ctx, &[rw]), EPERM);
    }

    #[test]
    fn rwlock_slot_and_opaque_handle_share_state_and_read_ownership() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let rw = 0x680;
        assert_eq!(hle_rwlock_init(&ctx, &[rw, 0]), OK);

        let mut bytes = [0u8; 8];
        assert!(mem.read(rw, &mut bytes));
        let handle = u64::from_le_bytes(bytes);
        assert_ne!(handle, 0);

        assert_eq!(hle_rwlock_rdlock(&ctx, &[rw]), OK);
        assert_eq!(
            hle_rwlock_trywrlock(&ctx, &[handle]),
            EBUSY,
            "a reader held through the slot must block a writer through its handle"
        );
        assert_eq!(
            hle_rwlock_unlock(&ctx, &[handle]),
            OK,
            "read ownership must be visible through either alias"
        );
        assert_eq!(hle_rwlock_unlock(&ctx, &[rw]), EPERM);
    }

    #[test]
    fn rwlock_write_hold_is_exclusive_and_recursive() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let rw = 0x700;
        hle_rwlock_init(&ctx, &[rw, 0]);
        assert_eq!(hle_rwlock_wrlock(&ctx, &[rw]), OK);
        assert_eq!(hle_rwlock_wrlock(&ctx, &[rw]), OK); // recursive write
        let key = resolve_rwlock_key(&ctx, rw).unwrap();
        {
            let s = kernel.pthread_rwlocks.get(&key).unwrap();
            let s = s.state.lock();
            assert_eq!(s.writer, CURRENT_THREAD);
            assert_eq!(s.writer_recursion, 2);
        }
        assert_eq!(hle_rwlock_unlock(&ctx, &[rw]), OK);
        assert_eq!(hle_rwlock_unlock(&ctx, &[rw]), OK);
        let s = kernel.pthread_rwlocks.get(&key).unwrap();
        assert_eq!(
            s.state.lock().writer,
            0,
            "write hold released at recursion 0"
        );
    }

    #[test]
    fn rwlock_destroy_removes_state_and_zeroes_handle() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let rw = 0x800;
        hle_rwlock_init(&ctx, &[rw, 0]);
        assert_eq!(hle_rwlock_destroy(&ctx, &[rw]), OK);
        let mut buf = [0u8; 8];
        assert!(mem.read(rw, &mut buf));
        assert_eq!(u64::from_le_bytes(buf), 0);
        assert!(!kernel.pthread_rwlocks.contains_key(&rw));
        assert_eq!(hle_rwlock_destroy(&ctx, &[0xABC]), EINVAL);
    }

    /// A stray or duplicated `scePthreadRwlockUnlock` from a thread that holds
    /// no read hold must not steal another thread's: it returns EPERM and leaves
    /// the real reader's hold (and the shared count) intact, so a writer stays
    /// out while that reader is still inside the lock. Regression for the
    /// shared-reader-count desync (sync-audit finding #3).
    #[test]
    fn rwlock_unlock_from_non_holder_is_rejected_and_preserves_other_readers() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let rw = 0x900u64;
        assert_eq!(hle_rwlock_init(&ctx, &[rw, 0]), OK);
        let key = resolve_rwlock_key(&ctx, rw).unwrap();

        // Another thread (99 ≠ the test's CURRENT_THREAD) holds one read hold —
        // exactly the state its own `Rdlock` would leave: shared count bumped
        // and a per-thread depth recorded.
        const OTHER: u64 = 99;
        kernel
            .pthread_rwlocks
            .get(&key)
            .unwrap()
            .state
            .lock()
            .readers = 1;
        kernel.pthread_rwlock_read_holds.insert((OTHER, key), 1);

        // The test thread holds nothing, so its unlock is rejected outright and
        // touches neither the shared count nor thread 99's per-thread hold.
        assert_eq!(hle_rwlock_unlock(&ctx, &[rw]), EPERM);
        assert_eq!(
            kernel
                .pthread_rwlocks
                .get(&key)
                .unwrap()
                .state
                .lock()
                .readers,
            1
        );
        assert_eq!(
            *kernel.pthread_rwlock_read_holds.get(&(OTHER, key)).unwrap(),
            1
        );

        // And the live reader still keeps writers out (before the fix, the stray
        // unlock would have zeroed the count and let this Trywrlock succeed).
        assert_eq!(hle_rwlock_trywrlock(&ctx, &[rw]), EBUSY);
    }

    /// The POSIX `pthread_rwlock_try*` names must resolve to the non-blocking
    /// bodies. They were registered to the blocking ones, so a guest probing a
    /// write-held rwlock parked instead of receiving EBUSY. Registration is
    /// compared by function pointer so a future re-wiring fails here.
    #[test]
    fn posix_rwlock_try_names_are_the_non_blocking_bodies() {
        let registry = HleRegistry::new();
        for (name, expected) in [
            (
                "pthread_rwlock_tryrdlock",
                hle_rwlock_tryrdlock as HleFunction,
            ),
            (
                "pthread_rwlock_trywrlock",
                hle_rwlock_trywrlock as HleFunction,
            ),
        ] {
            let mut found = false;
            for library in ["libScePosix", "libkernel", "libc", "libSceLibcInternal"] {
                let key = format!("{library}::{name}");
                if let Some(entry) = registry.functions.get(&key) {
                    found = true;
                    assert!(
                        std::ptr::fn_addr_eq(*entry.value(), expected),
                        "{key} must be the non-blocking body"
                    );
                }
            }
            assert!(found, "{name} is not registered under any provider");
        }
    }
}
