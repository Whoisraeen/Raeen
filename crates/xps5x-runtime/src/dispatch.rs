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

use crate::RunOutcome;
use crate::RuntimeError;
use crate::trampoline::{self, TrampolineGuard};

/// The `(library, function)` set that ends a process-mode run instead of
/// being serviced-and-resumed like an ordinary HLE call (design doc §4, wall
/// #1 W1b): `_start` ends the program by calling one of these, and never
/// returns to its caller in a well-formed program. `veh_callback` recognizes
/// a resolved trampoline call against this table *before* dispatching to
/// `hle.call`, and answers it with the exit-longjmp instead (see
/// [`veh_callback`]'s doc comment).
const TERMINATING_FUNCTIONS: &[(&str, &str)] = &[
    ("libc", "exit"),
    ("libc", "exit_group"),
    ("libc", "_exit"),
    ("libkernel", "sceKernelExit"),
];

/// Whether `(library, function)` names a [`TERMINATING_FUNCTIONS`] entry.
fn is_terminating(library: &str, function: &str) -> bool {
    TERMINATING_FUNCTIONS
        .iter()
        .any(|&(l, f)| l == library && f == function)
}

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
    CALL_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
    /// The FS base to restore after this call (RT2c-b, design doc §3),
    /// captured via `tls::read_fsbase()` before `RtlCaptureContext` — only
    /// when TLS is active for this call (see `tls_active` below). Read back
    /// in the shared continuation after the `if`/`else` in `run`, which runs
    /// on *both* the normal-return and RT1a-fault-recovery arrivals.
    ///
    /// Lives here (memory reached through `ACTIVE_CONTEXT`), not a plain
    /// local in `run`, for the same reason `resumed` does (see its doc
    /// comment just below): the RT2c-b spike found that a value carried as
    /// an ordinary local across `RtlCaptureContext`, then read again after a
    /// possible fault-driven resume, is not reliably restored — the fix,
    /// verified 20/20 clean runs in the spike, is storing it in memory
    /// reached through this struct instead, exactly like `resumed`/`error`
    /// already do.
    orig_fsbase: Cell<u64>,
    /// Whether TLS (fsbase set/restore) is active for this call — set once,
    /// before `RtlCaptureContext`, from `tcb.is_some() &&
    /// tls::fsgsbase_available()`, and read again in the shared continuation
    /// (which runs on both arrivals). Same "must live in `ctx`, not a local
    /// carried across `RtlCaptureContext`" reasoning as `orig_fsbase` above.
    tls_active: Cell<bool>,
    /// Set by `veh_callback` (design doc §4, wall #1 W1b) when a resolved
    /// trampoline call targets a [`TERMINATING_FUNCTIONS`] entry, alongside
    /// `exited`, just before it performs the same `recovery_ctx` longjmp the
    /// genuine-fault path uses. Lives here — not a plain local in `run` —
    /// for the exact same "must survive a register-only context restore"
    /// reason `resumed`/`orig_fsbase` do (see `resumed`'s doc comment):
    /// `veh_callback` writes it *before* the longjmp, and `run` reads it
    /// only after control has already jumped back to the recovery point.
    exit_code: Cell<u64>,
    /// Set to `true` by `veh_callback` alongside `exit_code`, immediately
    /// before the exit-longjmp (design doc §4). `run`'s shared continuation
    /// checks this *first* (before `error`) on the resumed arrival: a
    /// terminating call and a genuine fault are mutually exclusive —
    /// `veh_callback` only ever sets one of `exited`/`error` for a given
    /// call — but checking `exited` first matches the design doc's
    /// documented arrival order exactly.
    exited: Cell<bool>,
}

thread_local! {
    /// The active-call context [`veh_callback`] consults, held **per OS
    /// thread** (M1-E step 1, spec §6): the VEH is process-wide but runs
    /// synchronously on the faulting thread, so each thread's fault must find
    /// *its own* in-flight `run` call's context. `run` sets this to its
    /// stack-local `&ctx` before entering the guest and clears it to null
    /// after — a thread that faults with no guest call in flight (null slot)
    /// falls through with `EXCEPTION_CONTINUE_SEARCH`.
    ///
    /// `const`-initialized to a null `Cell` (spec §6 I4): no lazy `Once`, no
    /// `Drop`, so even a first touch inside the VEH (an unrelated access
    /// violation on a thread that never ran guest code, hitting the
    /// process-wide handler) is a cheap, non-allocating pointer read. Today
    /// exactly one guest call is ever in flight (`CALL_LOCK` still serializes
    /// all execution); this per-thread form is the substrate a real second
    /// guest thread will use — the pointee is only ever touched synchronously
    /// on the owning thread, so a plain `Cell` (not an atomic) is correct.
    static ACTIVE_CONTEXT: Cell<*mut ActiveContext> = const { Cell::new(ptr::null_mut()) };
}

/// Call the guest — via `call_guest`, invoked exactly once — servicing any
/// HLE trampoline calls it makes via a VEH for the duration of the call
/// (design doc §2), and distinguishing three ways that call can end: a
/// normal return (`RunOutcome::Returned`), the guest calling a terminating
/// function like `exit` (`RunOutcome::Exited`, design doc §4, wall #1 W1b),
/// or a genuine fault (`Err(Faulted)`, RT1a). `execute_linked` (function
/// mode, via [`crate::stack::call_on_guest_stack`]) and `execute_process`
/// (process mode `_start` entry, via
/// [`crate::stack::enter_guest_at_start`]) both funnel through this single
/// function so the fault-recovery/exit-longjmp machinery below is never
/// duplicated (design doc §8).
///
/// # Safety
/// `call_guest` must, when called, transfer control to mapped guest code the
/// caller set up specifically so it can be executed (i.e. a live
/// [`crate::arena::GuestArena`]'s image sub-region) exactly as
/// `call_on_guest_stack`'s or `enter_guest_at_start`'s own safety contract
/// requires, and return the value that should surface as
/// `RunOutcome::Returned` on a normal, non-terminating, non-faulting return.
/// It must be safe to call exactly once, synchronously, on this thread, for
/// the duration this function's VEH and RT1a recovery point are armed (both
/// are set up before it is called, and torn down/read only after it
/// returns). `trampolines`, `hle`, `kernel`, `mem`, and `alloc` must outlive
/// this call (guaranteed by `execute_linked`/`execute_process`'s borrows).
/// `guard` must be the [`TrampolineGuard`] whose region covers every address
/// `trampolines` resolves. `tcb`, if `Some`, is the guest TCB address (from
/// [`crate::arena::GuestArena::setup_main_tcb`]) to point the FS base at for
/// the duration of the guest call (RT2c-b, design doc §3); `None` means TLS
/// isn't set up for this call (e.g. FSGSBASE unavailable), in which case no
/// fsbase instruction is ever executed.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run(
    trampolines: &[HleTrampoline],
    hle: &HleRegistry,
    kernel: &OrbisKernel,
    mem: &dyn GuestMemory,
    alloc: &dyn GuestAllocator,
    guard: &TrampolineGuard,
    tcb: Option<u64>,
    call_guest: impl FnOnce() -> u64,
) -> Result<RunOutcome, RuntimeError> {
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
    let mem_erased: &'static dyn GuestMemory =
        unsafe { core::mem::transmute::<&dyn GuestMemory, &'static dyn GuestMemory>(mem) };

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
        orig_fsbase: Cell::new(0),
        tls_active: Cell::new(false),
        exit_code: Cell::new(0),
        exited: Cell::new(false),
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
    ACTIVE_CONTEXT.with(|slot| slot.set(&ctx as *const ActiveContext as *mut ActiveContext));

    // RT2c-b (design doc §3): if a guest TCB was set up and FSGSBASE is
    // available, capture the current FS base into `ctx.orig_fsbase` —
    // memory reached through `ACTIVE_CONTEXT` (already stored just above) —
    // *before* `RtlCaptureContext`, so it can be restored in the shared
    // continuation below regardless of which arrival gets there. See
    // `ActiveContext::orig_fsbase`'s doc comment and the RT2c-b spike report
    // for why this must live in `ctx`, not a plain local carried across
    // `RtlCaptureContext`. If FSGSBASE is unavailable or no TCB was set up,
    // `ctx.tls_active` stays `false` and no fsbase instruction is ever
    // executed for this call (honest degradation).
    if tcb.is_some() && crate::tls::fsgsbase_available() {
        ctx.tls_active.set(true);
        // SAFETY: `fsgsbase_available()` just returned `true`, so `RDFSBASE`
        // is permitted on this CPU. This only reads the current FS base; it
        // does not modify any CPU state.
        ctx.orig_fsbase.set(unsafe { crate::tls::read_fsbase() });
    }

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
        if ctx.tls_active.get() {
            // SAFETY: `ctx.tls_active` is only `true` when
            // `fsgsbase_available()` returned `true` (checked above, before
            // `RtlCaptureContext`) and `tcb.is_some()` (checked in that same
            // condition, so the `expect` below cannot fail). This `else`
            // branch only ever runs on the original, not-yet-faulted
            // arrival (a resumed arrival always takes the `if` branch
            // above), so `tcb` here is still exactly the guest TCB address
            // `execute_linked` passed in — a valid guest address inside the
            // arena, set up by `GuestArena::setup_main_tcb`. Setting FS base
            // here, immediately before the guest call, does not touch GS or
            // any other CPU state.
            unsafe {
                crate::tls::write_fsbase(tcb.expect("tls_active implies tcb.is_some()"));
            }
        }
        // SAFETY: `call_guest`'s contract (this function's `# Safety`
        // section) guarantees it transfers control to mapped, executable
        // guest code exactly as `call_on_guest_stack`'s/
        // `enter_guest_at_start`'s own safety contract requires. The VEH is
        // armed (just above) for the entire duration of this call, so any
        // guest `call [import_slot]` into the guarded trampoline region is
        // trapped and serviced by `veh_callback`, and any genuine wild fault
        // is recovered via the `RtlCaptureContext` snapshot just taken
        // above, rather than crashing the process.
        //
        // `entry` runs on the guest's own stack, switched to and back (or,
        // for a process-mode `_start` entry, never back at all on a
        // well-formed run — see `enter_guest_at_start`'s doc comment) by
        // whichever of `call_on_guest_stack`/`enter_guest_at_start`
        // `call_guest` wraps, instead of directly on this (host) stack —
        // this does not change the recovery argument above: `recovery_ctx`
        // was captured with the *host* RSP before this switch ever happens,
        // so a genuine fault (or an exit-longjmp) still restores the host
        // RSP and abandons the guest stack entirely.
        //
        // Note: invoking `call_guest` here is an ordinary (safe-typed)
        // closure call — the `unsafe` obligations described above are
        // discharged by whichever `unsafe { crate::stack::... }` block the
        // caller built the closure from (see `execute_linked`/
        // `execute_process`), not by this call site itself.
        let r = call_guest();
        // Returned normally: no genuine fault or exit-longjmp occurred on
        // this call. Disarm so `resumed` doesn't linger set for no reason
        // (it's about to be dropped along with `ctx` regardless, but this
        // keeps the invariant "armed only while the guest call is actually
        // in flight" honest).
        ctx.resumed.set(false);
        r
    };

    if ctx.tls_active.get() {
        // SAFETY: `ctx.tls_active` being `true` guarantees FSGSBASE is
        // available (see where it's set, before `RtlCaptureContext`, above).
        // This is the shared continuation reached after the `if`/`else`
        // above — i.e. on *both* the normal-return and RT1a-fault-recovery
        // arrivals (see this module's doc comment) — so it restores the FS
        // base captured into `ctx.orig_fsbase` before the guest call,
        // reading it fresh from memory here rather than trusting a local
        // carried across `RtlCaptureContext`.
        unsafe {
            crate::tls::write_fsbase(ctx.orig_fsbase.get());
        }
    }

    ACTIVE_CONTEXT.with(|slot| slot.set(ptr::null_mut()));
    // SAFETY: `handle` is exactly the handle `AddVectoredExceptionHandler`
    // returned above, removed exactly once.
    unsafe {
        RemoveVectoredExceptionHandler(handle);
    }

    // Design doc §4: on the resumed arrival, an exit-family termination is
    // checked first, then a genuine fault; only the arrival that actually
    // fell out of `call_guest` normally reaches the ordinary-return arm.
    // `veh_callback` only ever sets one of `ctx.exited`/`ctx.error` for a
    // given call (the exit-longjmp and the fault-recovery longjmp are
    // mutually exclusive outcomes of the same guarded call), so the order
    // between the first two arms doesn't change behavior — it's written
    // this way to match the design doc's documented arrival order exactly.
    if ctx.exited.get() {
        return Ok(RunOutcome::Exited(ctx.exit_code.get()));
    }
    match ctx.error.take() {
        Some(err) => Err(err),
        None => Ok(RunOutcome::Returned(result)),
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

    let ctx_ptr = ACTIVE_CONTEXT.with(|slot| slot.get());
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
        ctx.error
            .set(Some(RuntimeError::Faulted { addr: fault_addr }));

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
            if is_terminating(&t.library, &t.function) {
                // Design doc §4 (wall #1 W1b): `_start` ends the program by
                // calling `exit`/`exit_group`/`_exit` (or `sceKernelExit`) —
                // read the exit code from SysV arg 0 (`Rdi`), record it plus
                // the `exited` flag in `ctx` (memory, returns-twice-safe —
                // see `ActiveContext::exited`'s doc comment), then perform
                // the *same* longjmp the genuine-fault path below uses:
                // overwrite the delivered context with `run`'s pre-call
                // `RtlCaptureContext` snapshot and resume there, instead of
                // servicing this call and resuming the guest. `run`
                // distinguishes this arrival from a genuine fault by
                // checking `ctx.exited` first.
                ctx.exit_code.set(context.Rdi);
                ctx.exited.set(true);

                // SAFETY: same reasoning as the genuine-fault restore below
                // — `ctx.recovery_ctx` points at a still-live stack local in
                // `run`'s frame, necessarily below this callback on the same
                // thread's stack (vectored exceptions are delivered
                // synchronously on the faulting thread). `CONTEXT` is
                // `Copy`, so this is a plain struct copy.
                *context = unsafe { *ctx.recovery_ctx };
                return EXCEPTION_CONTINUE_EXECUTION;
            }

            // SysV integer argument registers (args 1-6, design doc §3),
            // followed by the first `STACK_ARGS` on-stack integer arguments
            // (args 7+). At this trap `Rsp` points at the `call`-pushed return
            // address, so arg7 is at `[Rsp+8]`, arg8 at `[Rsp+16]`, … . Reads
            // are bounds-checked against the guest arena (0 if unmapped), so
            // functions with ≤6 args are unaffected — the extra slots are
            // simply never consulted by their HLE bodies.
            const STACK_ARGS: usize = 8;
            let mut args = [0u64; 6 + STACK_ARGS];
            args[0] = context.Rdi;
            args[1] = context.Rsi;
            args[2] = context.Rdx;
            args[3] = context.Rcx;
            args[4] = context.R8;
            args[5] = context.R9;
            for i in 0..STACK_ARGS {
                let slot = context.Rsp.wrapping_add(8 + (i as u64) * 8);
                let mut buf = [0u8; 8];
                if mem.read(slot, &mut buf) {
                    args[6 + i] = u64::from_le_bytes(buf);
                }
            }
            let hle_ctx = HleContext { kernel, mem, alloc };
            hle.call(&hle_ctx, &t.library, &t.function, &args)
                .unwrap_or(0)
        }
        None => {
            // A call landed in the guarded region but names no known
            // trampoline (out of range of this module's table) — record it
            // so `run` surfaces `UnresolvedTrampoline` after the call
            // returns, but still service the call as a 0-returning stub so
            // we can safely resume (design doc §7 step 2's suggested
            // approach) rather than needing an unwind-style abort.
            ctx.error
                .set(Some(RuntimeError::UnresolvedTrampoline(fault_addr)));
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
