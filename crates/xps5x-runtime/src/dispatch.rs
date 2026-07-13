//! The Vectored Exception Handler (VEH) that services HLE trampoline calls
//! (design doc §2/§4), plus [`run`], which arms it around a single native
//! call into mapped guest code.
//!
//! # Genuine faults are not (yet) converted to `RuntimeError::Faulted`
//!
//! The VEH only ever resumes execution for `EXCEPTION_ACCESS_VIOLATION`s
//! whose faulting address lands inside the active trampoline guard region
//! (a `call` through an HLE-resolved import slot). Any other exception —
//! including a genuine wild guest fault — is passed on with
//! `EXCEPTION_CONTINUE_SEARCH` and is *not* swallowed or silently converted
//! into a `Faulted` return: doing that safely for an arbitrary faulting
//! program counter would require a saved recovery point (a
//! setjmp/longjmp-equivalent host register snapshot taken before calling
//! the guest, restored on fault) that RT0 does not implement yet, because
//! rebuilding it correctly on top of Rust's own stack/unwind machinery is
//! its own piece of delicate `unsafe` work. `RuntimeError::Faulted` is kept
//! in the public API for callers to already match on; wiring it up is
//! deferred to a later milestone (see the design doc §7 RT1/RT2 notes and
//! the crate-level module docs).

use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};
use std::cell::Cell;
use std::sync::Mutex;

use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    RemoveVectoredExceptionHandler,
};

use xps5x_firmware::HleTrampoline;
use xps5x_hle::HleRegistry;

use crate::RuntimeError;
use crate::trampoline::{self, TrampolineGuard};

/// Serializes calls to [`run`] (and thus [`crate::execute_linked`]): RT0
/// supports exactly one active native guest execution at a time (design doc
/// §4/§6/§9 — "RT0 is single-threaded-execution"). Held for the *entire*
/// duration of the guest call, setup through teardown; that's also what
/// makes `ACTIVE_CONTEXT` below safe to touch from the VEH without a lock of
/// its own — only one OS thread is ever "inside" a guarded call at a time,
/// and the VEH only ever runs synchronously on that same thread.
static CALL_LOCK: Mutex<()> = Mutex::new(());

/// The context [`veh_callback`] consults while a guest call is in flight.
/// Written by [`run`] before calling the guest, read (and, on an unresolved
/// trampoline, written back to) by the VEH, cleared by [`run`] after the
/// call returns. See `CALL_LOCK`'s doc comment for why this never needs its
/// own synchronization.
struct ActiveContext {
    trampolines: *const [HleTrampoline],
    hle: *const HleRegistry,
    region_base: u64,
    region_len: u64,
    error: Cell<Option<RuntimeError>>,
}

/// `AtomicPtr<T>` is `Send + Sync` regardless of `T` — see `CALL_LOCK`'s doc
/// comment for why non-concurrent, single-thread-only access to the pointee
/// is guaranteed here despite that.
static ACTIVE_CONTEXT: AtomicPtr<ActiveContext> = AtomicPtr::new(ptr::null_mut());

/// Call `entry` (a mapped guest function) with `args`, servicing any HLE
/// trampoline calls it makes via a VEH for the duration of the call (design
/// doc §2).
///
/// # Safety
/// `entry` must be a valid function pointer, callable with the
/// `extern "sysv64"` calling convention, into memory the caller mapped
/// specifically so it can be executed (i.e. a live
/// [`crate::mem::MappedImage`]'s contents) — calling it runs that code
/// natively on the current thread. `trampolines` and `hle` must outlive this
/// call (guaranteed by `execute_linked`'s borrows). `guard` must be the
/// [`TrampolineGuard`] whose region covers every address `trampolines`
/// resolves.
pub(crate) unsafe fn run(
    entry: unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64,
    args: [u64; 6],
    trampolines: &[HleTrampoline],
    hle: &HleRegistry,
    guard: &TrampolineGuard,
) -> Result<u64, RuntimeError> {
    // Serialize: only one native guest execution at a time (RT0).
    let _lock = CALL_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

    let ctx = ActiveContext {
        trampolines: trampolines as *const [HleTrampoline],
        hle: hle as *const HleRegistry,
        region_base: guard.base(),
        region_len: guard.len(),
        error: Cell::new(None),
    };

    // SAFETY: `veh_callback` has the `unsafe extern "system" fn(*mut
    // EXCEPTION_POINTERS) -> i32` signature `AddVectoredExceptionHandler`
    // requires. Registered as the first handler (`1`) so it sees the
    // exception before any other handler in the process; removed before
    // this function returns on every path below, so it never outlives `ctx`
    // or this call.
    let handle = unsafe { AddVectoredExceptionHandler(1, Some(veh_callback)) };
    if handle.is_null() {
        return Err(RuntimeError::MapFailed);
    }

    // `ctx` is a local; its address is stable for the rest of this function
    // (never moved), which is exactly the lifetime the VEH needs it for.
    ACTIVE_CONTEXT.store(&ctx as *const ActiveContext as *mut ActiveContext, Ordering::Release);

    // SAFETY: `entry` is a valid `sysv64` function pointer into mapped,
    // executable guest code per this function's safety contract. The VEH is
    // armed (just above) for the entire duration of this call, so any guest
    // `call [import_slot]` into the guarded trampoline region is trapped
    // and serviced by `veh_callback` rather than crashing the process.
    let result = unsafe { entry(args[0], args[1], args[2], args[3], args[4], args[5]) };

    ACTIVE_CONTEXT.store(ptr::null_mut(), Ordering::Release);
    // SAFETY: `handle` is exactly the handle `AddVectoredExceptionHandler`
    // returned above, removed exactly once.
    unsafe {
        RemoveVectoredExceptionHandler(handle);
    }

    match ctx.error.take() {
        Some(err) => Err(err),
        None => Ok(result),
    }
}

/// The VEH callback. Only services `EXCEPTION_ACCESS_VIOLATION`s whose
/// faulting address falls inside the currently-active `TrampolineGuard`
/// region; everything else is passed on with `EXCEPTION_CONTINUE_SEARCH` —
/// see this module's doc comment for why genuine faults are never
/// swallowed.
///
/// # Safety
/// Called by the OS (via the Vectored Exception Handler mechanism) with a
/// valid `info` pointer for the duration of this callback.
unsafe extern "system" fn veh_callback(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: `info` and the `ExceptionRecord`/`ContextRecord` pointers it
    // contains are supplied by the OS and valid for the duration of this
    // callback (the `AddVectoredExceptionHandler` contract).
    let info = unsafe { &*info };
    let record = unsafe { &*info.ExceptionRecord };

    if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    let ctx_ptr = ACTIVE_CONTEXT.load(Ordering::Acquire);
    if ctx_ptr.is_null() {
        // No execute_linked call is in flight (or we're outside its guarded
        // window) — not ours to handle.
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: a non-null `ctx_ptr` was stored by `run` and points at a
    // still-live stack local for the entire duration of the guarded call —
    // which we are necessarily inside right now, since the OS delivers
    // vectored exceptions synchronously on the same thread that faulted,
    // and `run`'s call to `entry` (the only place a guest-code fault can
    // originate) is on this thread's stack below us.
    let ctx = unsafe { &*ctx_ptr };

    // SAFETY: `info.ContextRecord` is valid per the VEH contract; mutable
    // access is required to redirect execution (Rip/Rsp/Rax) below.
    let context = unsafe { &mut *info.ContextRecord };
    let fault_addr = context.Rip;

    if fault_addr < ctx.region_base || fault_addr >= ctx.region_base + ctx.region_len {
        // Outside our guarded trampoline region: a genuine guest fault.
        // Never swallow it — hand it to the next handler / default
        // unhandled-exception path. See this module's doc comment.
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // SAFETY: `ctx.trampolines`/`ctx.hle` were stored by `run` from live
    // `&[HleTrampoline]`/`&HleRegistry` references that, per `run`'s safety
    // contract, outlive this call.
    let trampolines = unsafe { &*ctx.trampolines };
    let hle = unsafe { &*ctx.hle };

    let result = match trampoline::resolve(fault_addr, trampolines) {
        Some(t) => {
            // SysV integer argument registers (design doc §3).
            let args = [context.Rdi, context.Rsi, context.Rdx, context.Rcx, context.R8, context.R9];
            hle.call(&t.library, &t.function, &args).unwrap_or(0)
        }
        None => {
            // A call landed in the guarded region but names no known
            // trampoline (out of range of this module's table) — record it
            // so `run` surfaces `UnresolvedTrampoline` after the call
            // returns, but still service the call as a 0-returning stub so
            // we can safely resume (design doc §7 step 2's suggested
            // approach) rather than needing an unwind-style abort.
            ctx.error.set(Some(RuntimeError::UnresolvedTrampoline(fault_addr)));
            0
        }
    };

    context.Rax = result;

    // Emulate the `call` instruction's target returning. The CPU already
    // pushed the return address before faulting on the instruction fetch at
    // the (guarded) trampoline address — `call`'s push-then-jump always
    // precedes the fetch of the first instruction at the new Rip, so by the
    // time we're here [Rsp] holds that return address.
    //
    // SAFETY: `Rsp` was set by the guest's own `call` instruction per
    // standard x86-64 `call` semantics and points at 8 bytes of the guest's
    // (mapped, valid) stack holding the pushed return address.
    let ret_addr = unsafe { core::ptr::read(context.Rsp as *const u64) };
    context.Rip = ret_addr;
    context.Rsp = context.Rsp.wrapping_add(8);

    EXCEPTION_CONTINUE_EXECUTION
}
