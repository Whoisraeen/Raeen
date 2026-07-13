//! The Vectored Exception Handler (VEH) that services HLE trampoline calls
//! (design doc §2/§4), plus [`run`], which arms it around a single native
//! call into mapped guest code.
//!
//! # Genuine-fault recovery (RT1a)
//!
//! A guest fault *outside* the trampoline guard region — a genuine wild
//! access violation, not an HLE call — is recovered as `Err(Faulted { addr
//! })` instead of crashing the process. This is Windows' setjmp/longjmp
//! equivalent, built from `RtlCaptureContext` plus a manual `CONTEXT`
//! overwrite (the same "edit the delivered `CONTEXT` and
//! `EXCEPTION_CONTINUE_EXECUTION`" mechanism the trampoline path below
//! already uses, just with a much bigger jump):
//!
//! 1. Before calling the guest, [`run`] calls `RtlCaptureContext` to
//!    snapshot the calling thread's *entire* register state — Rip, Rsp,
//!    Rbp, every GPR — into a stack-local `CONTEXT` (`recovery_ctx`). This
//!    is the recovery point: "the state of this thread right before it
//!    called the guest."
//! 2. `run` stores a pointer to `recovery_ctx` in [`ActiveContext`], then
//!    calls `entry(...)`.
//! 3. If `entry` faults genuinely (access violation outside the guard
//!    region), [`veh_callback`] records `RuntimeError::Faulted { addr:
//!    context.Rip }`, copies `*recovery_ctx` over the OS-delivered
//!    `ContextRecord` (a plain struct copy — `CONTEXT` is `Copy`), and
//!    returns `EXCEPTION_CONTINUE_EXECUTION`. The OS then resumes the
//!    faulting thread with *that* (restored) register state, which lands
//!    it back at the exact instruction after step 1's `RtlCaptureContext`
//!    call — i.e. exactly as if `entry(...)` had "returned" — discarding
//!    every stack frame `entry()` (and anything it called) pushed below
//!    that point.
//! 4. `run` distinguishes this "resumed after a fault" arrival from the
//!    original "about to call the guest" arrival using
//!    `ActiveContext::resumed`, a `Cell<bool>` armed immediately before the
//!    guest call. A register-only context restore never touches memory, so
//!    this `Cell` (ordinary host stack memory) reliably survives the jump
//!    and tells the two arrivals apart — see `run`'s doc comment for the
//!    full control-flow argument.
//!
//! Servicing an HLE trampoline call (the pre-existing mechanism, unchanged)
//! never touches `recovery_ctx`/`resumed` and never restores; it only
//! happens for faults *inside* the guard region. Non-access-violation
//! exceptions are still passed on with `EXCEPTION_CONTINUE_SEARCH`,
//! unchanged.

use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};
use std::cell::Cell;
use std::sync::Mutex;

use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
    EXCEPTION_POINTERS, RemoveVectoredExceptionHandler, RtlCaptureContext,
};

use xps5x_firmware::HleTrampoline;
use xps5x_hle::{GuestAllocator, GuestMemory, HleContext, HleRegistry};
use xps5x_kernel::OrbisKernel;

use crate::RuntimeError;
use crate::trampoline::{self, TrampolineGuard};

/// Serializes [`crate::execute_linked`] end to end: RT0 supports exactly
/// one active native guest execution at a time (design doc §4/§6/§9 —
/// "RT0 is single-threaded-execution"). [`call_lock`] is acquired by
/// `execute_linked` itself (not just around the guest call in [`run`]),
/// because `TrampolineGuard::reserve`'s fixed-address `VirtualAlloc` (in
/// `trampoline.rs`) is process-global state just as much as `run`'s guest
/// call is — two concurrent `execute_linked` calls racing to reserve the
/// same fixed `HLE_TRAMPOLINE_BASE` region would spuriously fail with
/// `MapFailed`, lock or no lock inside `run` alone. Holding it for the
/// whole pipeline is also what makes `ACTIVE_CONTEXT` below safe to touch
/// from the VEH without a lock of its own — only one OS thread is ever
/// "inside" a guarded call at a time, and the VEH only ever runs
/// synchronously on that same thread.
static CALL_LOCK: Mutex<()> = Mutex::new(());

/// Acquire [`CALL_LOCK`] for the entire [`crate::execute_linked`] pipeline.
/// Called once, at the very top of `execute_linked`; the returned guard is
/// held (as a local binding) until that function returns. `run` no longer
/// acquires `CALL_LOCK` itself — it's not reentrant, and the lock must
/// already be held by the time `run` is called.
pub(crate) fn call_lock() -> std::sync::MutexGuard<'static, ()> {
    CALL_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The context [`veh_callback`] consults while a guest call is in flight.
/// Written by [`run`] before calling the guest, read (and, on an unresolved
/// trampoline, written back to) by the VEH, cleared by [`run`] after the
/// call returns. See `CALL_LOCK`'s doc comment for why this never needs its
/// own synchronization.
struct ActiveContext {
    trampolines: *const [HleTrampoline],
    hle: *const HleRegistry,
    /// The live kernel an HLE call's [`HleContext::kernel`] borrows from.
    /// Stored as a raw pointer for the same reason `trampolines`/`hle` are
    /// (see `CALL_LOCK`'s doc comment): only one OS thread is ever "inside"
    /// a guarded call at a time, and the VEH only ever runs synchronously
    /// on that same thread, so a non-atomic raw pointer read here is sound.
    kernel: *const OrbisKernel,
    /// The guest memory view an HLE call's [`HleContext::mem`] borrows
    /// from — a trait object pointer (fat pointer) for the same reason.
    mem: *const dyn GuestMemory,
    /// The guest allocator an HLE call's [`HleContext::alloc`] borrows
    /// from — a trait object pointer (fat pointer), same reasoning as `mem`.
    alloc: *const dyn GuestAllocator,
    region_base: u64,
    region_len: u64,
    error: Cell<Option<RuntimeError>>,
    /// Armed (`true`) by `run` immediately before it calls the guest, and
    /// read by `run` right after the `RtlCaptureContext` recovery point —
    /// on *both* the original arrival there and, if the guest faults
    /// genuinely, the restored arrival (see this module's doc comment).
    /// `veh_callback` never writes this; only `run` does. It has to live
    /// here (not a plain local in `run`) because it must survive the
    /// register-only context restore that jumps back to before `entry` was
    /// called — ordinary host stack memory like this `Cell` is untouched
    /// by that restore, which is exactly why it works as the signal.
    resumed: Cell<bool>,
    /// Pointer to `run`'s stack-local recovery `CONTEXT`, populated by
    /// `RtlCaptureContext` before the guest call. On a genuine fault,
    /// `veh_callback` copies `*recovery_ctx` over the OS-delivered
    /// `ContextRecord`; it is never read on the (unchanged) trampoline
    /// path.
    recovery_ctx: *const CONTEXT,
    /// The memory slot [`crate::stack::call_on_guest_stack`] saves the host
    /// RSP into before switching to the guest stack, and restores it from
    /// afterward (design doc §2/§4, RT2c-a). Lives here (not a plain local
    /// in `run`) purely so its address is easy to hand to the asm
    /// trampoline via `Cell::as_ptr`; it is never read or written by
    /// `veh_callback` or by the `resumed`/recovery mechanism — those are
    /// entirely unaffected by which stack the guest runs on (see this
    /// module's doc comment and `run`'s doc comment for why). `Cell<u64>`
    /// has the same in-memory representation as `u64` (guaranteed:
    /// `Cell`/`UnsafeCell` are `#[repr(transparent)]`), so `as_ptr` yields a
    /// plain `*mut u64` the asm block can address directly.
    host_rsp: Cell<u64>,
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
/// [`crate::arena::GuestArena`]'s image sub-region) — calling it runs that
/// code natively on the current thread. `trampolines`, `hle`, `kernel`, `mem`,
/// and `alloc` must outlive this call (guaranteed by `execute_linked`'s
/// borrows). `guard` must be the [`TrampolineGuard`] whose region covers
/// every address `trampolines` resolves. `guest_rsp_top` must be a
/// 16-byte-aligned address that is the top of a committed, writable region
/// big enough to serve as `entry`'s stack (i.e.
/// [`crate::arena::GuestArena::stack_top`]) — see
/// [`crate::stack::call_on_guest_stack`]'s doc comment for the full
/// RSP-switch contract.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run(
    entry: unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64,
    args: [u64; 6],
    trampolines: &[HleTrampoline],
    hle: &HleRegistry,
    kernel: &OrbisKernel,
    mem: &dyn GuestMemory,
    alloc: &dyn GuestAllocator,
    guard: &TrampolineGuard,
    guest_rsp_top: u64,
) -> Result<u64, RuntimeError> {
    // Serialization (only one native guest execution at a time, RT0) is
    // provided by the caller: `execute_linked` holds `call_lock()` for its
    // entire pipeline, not just this call — see `CALL_LOCK`'s doc comment.

    // The RT1a recovery point's storage. A stack local here (not inside
    // `ctx`) so its address is stable for exactly this call's duration,
    // same as `ctx` itself; `ctx.recovery_ctx` below just points at it.
    //
    // SAFETY: zero-initializing a `CONTEXT` is valid — it's a plain
    // `repr(C)` struct of integers/arrays with no pointer/validity
    // invariants — and `RtlCaptureContext` fully populates it below before
    // it is ever read (by `veh_callback`, on a genuine fault; `run` itself
    // never reads it).
    let mut recovery_ctx: CONTEXT = unsafe { core::mem::zeroed() };

    // `dyn GuestMemory` written bare (as `ActiveContext.mem`'s field type is)
    // carries an implicit `'static` bound, unlike `&dyn GuestMemory`, whose
    // bound follows the reference's own lifetime — so building the raw
    // pointer needs an explicit lifetime-erasing transmute here, the same
    // trick applied implicitly by the plain `as *const _` casts just below
    // for `trampolines`/`hle`/`kernel` (those aren't trait objects, so they
    // don't hit this). Sound for the same reason those are: `run`'s safety
    // contract requires `mem` to outlive this call, it is only ever
    // dereferenced synchronously on this thread while `ctx` is alive (see
    // `CALL_LOCK`'s doc comment), and `ACTIVE_CONTEXT` is cleared before
    // `run` returns.
    //
    // SAFETY: `&dyn GuestMemory` and `&'static dyn GuestMemory` have
    // identical layout (data pointer + vtable pointer); this only widens the
    // *type-level* lifetime, which the actual borrow (`mem`, tied to
    // `execute_linked`'s stack frame for this whole call) still outlives in
    // practice.
    let mem_erased: &'static dyn GuestMemory = unsafe { core::mem::transmute::<&dyn GuestMemory, &'static dyn GuestMemory>(mem) };

    // SAFETY: same reasoning as `mem_erased` immediately above, applied to
    // `alloc` instead of `mem` — `&dyn GuestAllocator` carries the same
    // implicit non-`'static` bound that `&dyn GuestMemory` does, so building
    // `ActiveContext.alloc`'s raw pointer needs the same lifetime-erasing
    // transmute. Sound for the same reason: `run`'s safety contract requires
    // `alloc` to outlive this call, it is only ever dereferenced
    // synchronously on this thread while `ctx` is alive, and
    // `ACTIVE_CONTEXT` is cleared before `run` returns.
    let alloc_erased: &'static dyn GuestAllocator =
        unsafe { core::mem::transmute::<&dyn GuestAllocator, &'static dyn GuestAllocator>(alloc) };

    let ctx = ActiveContext {
        trampolines: trampolines as *const [HleTrampoline],
        hle: hle as *const HleRegistry,
        kernel: kernel as *const OrbisKernel,
        mem: mem_erased as *const dyn GuestMemory,
        alloc: alloc_erased as *const dyn GuestAllocator,
        region_base: guard.base(),
        region_len: guard.len(),
        error: Cell::new(None),
        resumed: Cell::new(false),
        recovery_ctx: &recovery_ctx as *const CONTEXT,
        host_rsp: Cell::new(0),
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

    // SAFETY: `RtlCaptureContext` only requires a valid, writable `CONTEXT`
    // out-pointer, which `&mut recovery_ctx` is. It captures the calling
    // thread's complete register state as of right now: `Rip` becomes the
    // address right after this call returns, and `Rsp`/`Rbp`/every GPR
    // reflect exactly this point in `run`'s frame. This is RT1a's recovery
    // point — see this module's doc comment and `veh_callback`.
    unsafe { RtlCaptureContext(&mut recovery_ctx) };

    // Execution reaches this exact point in two different ways (see this
    // module's doc comment for the full mechanism):
    //  1. Normally, immediately after the `RtlCaptureContext` call above
    //     returns. `ctx.resumed` is still `false` (just initialized), so
    //     the `else` arm runs: it arms `resumed`, then calls the guest.
    //  2. If the guest then faults genuinely (outside the trampoline guard
    //     region), `veh_callback` overwrites the OS-delivered exception
    //     context with a copy of `recovery_ctx` and returns
    //     `EXCEPTION_CONTINUE_EXECUTION`. The OS resumes this thread with
    //     that restored register state, landing back at this exact
    //     instruction with the exact registers arrival (1) had — except
    //     `ctx.resumed` now reads `true`, because it was set right before
    //     the guest call in arrival (1), and a register-only context
    //     restore never touches memory (`ctx.resumed` lives in ordinary
    //     host stack memory, not a register). So this second arrival takes
    //     the `if` branch: `entry` is *not* called again — its frame, and
    //     everything `run` pushed while inside it, is gone (`Rsp` was reset
    //     by the restore) — and `ctx.error` already holds `Faulted { addr
    //     }`, set by `veh_callback` before it restored us here.
    let result = if ctx.resumed.get() {
        0
    } else {
        ctx.resumed.set(true);
        // SAFETY: `entry` is a valid `sysv64` function pointer into mapped,
        // executable guest code per this function's safety contract. The
        // VEH is armed (just above) for the entire duration of this call,
        // so any guest `call [import_slot]` into the guarded trampoline
        // region is trapped and serviced by `veh_callback`, and any
        // genuine wild fault is recovered via the `RtlCaptureContext`
        // snapshot just taken above, rather than crashing the process.
        //
        // `entry` now runs on the guest's own stack (`guest_rsp_top`),
        // switched to and back by `call_on_guest_stack`, instead of directly
        // on this (host) stack — this does not change the recovery argument
        // above: `recovery_ctx` was captured with the *host* RSP before this
        // switch ever happens, so a genuine fault still restores the host
        // RSP and abandons the guest stack entirely (`call_on_guest_stack`'s
        // own doc comment covers the RSP-switch mechanism itself; see this
        // module's doc comment for the full compatibility argument).
        // `ctx.host_rsp`'s address is passed as the save slot via
        // `Cell::as_ptr` (an ordinary host-stack field, unreachable from
        // guest memory).
        let r = unsafe { crate::stack::call_on_guest_stack(entry, args, guest_rsp_top, ctx.host_rsp.as_ptr()) };
        // Returned normally: no genuine fault occurred on this call.
        // Disarm so `resumed` doesn't linger set for no reason (it's about
        // to be dropped along with `ctx` regardless, but this keeps the
        // invariant "armed only while the guest call is actually in
        // flight" honest).
        ctx.resumed.set(false);
        r
    };

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

/// The VEH callback. Services `EXCEPTION_ACCESS_VIOLATION`s whose faulting
/// address falls inside the currently-active `TrampolineGuard` region (an
/// HLE call) by dispatching to HLE and resuming the guest; genuine faults
/// outside that region are recovered via the `RtlCaptureContext`-based
/// restore described in this module's doc comment. Any other exception, or
/// an access violation with no `execute_linked` call in flight, is passed
/// on with `EXCEPTION_CONTINUE_SEARCH`.
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
        // Recover rather than crash (RT1a): record the error, then
        // overwrite the delivered context with the pre-call snapshot `run`
        // took via `RtlCaptureContext`, and resume there. See this module's
        // doc comment for the full control-flow argument.
        ctx.error.set(Some(RuntimeError::Faulted { addr: fault_addr }));

        // SAFETY: `ctx.recovery_ctx` was populated by `run`'s
        // `RtlCaptureContext` call before it called the guest, and is
        // still valid here: it points at a stack local in `run`'s frame,
        // which is necessarily still live on this same thread's stack
        // below this callback (vectored exceptions are delivered
        // synchronously on the faulting thread, and `run`'s call to
        // `entry` — the only place a guest fault can originate — is on
        // this thread's stack beneath us). `CONTEXT` is `Copy`, so this is
        // a plain struct copy, not a move of anything with drop glue.
        // Overwriting `*context` and returning
        // `EXCEPTION_CONTINUE_EXECUTION` below makes the OS resume this
        // thread with exactly that (restored) register state.
        *context = unsafe { *ctx.recovery_ctx };

        return EXCEPTION_CONTINUE_EXECUTION;
    }

    // SAFETY: `ctx.trampolines`/`ctx.hle`/`ctx.kernel`/`ctx.mem`/`ctx.alloc`
    // were stored by `run` from live `&[HleTrampoline]`/`&HleRegistry`/
    // `&OrbisKernel`/`&dyn GuestMemory`/`&dyn GuestAllocator` references
    // that, per `run`'s safety contract, outlive this call.
    let trampolines = unsafe { &*ctx.trampolines };
    let hle = unsafe { &*ctx.hle };
    let kernel = unsafe { &*ctx.kernel };
    let mem = unsafe { &*ctx.mem };
    let alloc = unsafe { &*ctx.alloc };

    let result = match trampoline::resolve(fault_addr, trampolines) {
        Some(t) => {
            // SysV integer argument registers (design doc §3).
            let args = [context.Rdi, context.Rsi, context.Rdx, context.Rcx, context.R8, context.R9];
            let hle_ctx = HleContext { kernel, mem, alloc };
            hle.call(&hle_ctx, &t.library, &t.function, &args).unwrap_or(0)
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
