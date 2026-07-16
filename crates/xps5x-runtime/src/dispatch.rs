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
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter, Register};
use windows_sys::Win32::Foundation::{
    CloseHandle, EXCEPTION_ACCESS_VIOLATION, EXCEPTION_ILLEGAL_INSTRUCTION,
};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, CONTEXT_ALL_AMD64, EXCEPTION_CONTINUE_EXECUTION,
    EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS, GetThreadContext, RtlCaptureContext,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT,
    THREAD_SUSPEND_RESUME,
};

use xps5x_firmware::{HleTrampoline, UnresolvedStub};
use xps5x_hle::{
    GuestAllocator, GuestCallCompletion, GuestCallRequest, GuestCallScheduler, GuestMemory,
    GuestThreadScheduler, HleContext, HleRegistry,
};
use xps5x_kernel::OrbisKernel;

use crate::RunOutcome;
use crate::RuntimeError;
use crate::stub;
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
    // libkernel's real export of _exit (NID hashed from "_exit") — libc.prx
    // and the eboot both import it directly.
    ("libkernel", "_exit"),
];

/// Whether `(library, function)` names a [`TERMINATING_FUNCTIONS`] entry.
fn is_terminating(library: &str, function: &str) -> bool {
    TERMINATING_FUNCTIONS
        .iter()
        .any(|&(l, f)| l == library && f == function)
}

/// Find a process-termination trampoline suitable for the Orbis `_start`
/// `exit_fn` argument. Retail entries preserve RSI and call through it when
/// startup terminates; handing them zero turns an orderly shutdown into a
/// secondary execute-at-null fault.
pub(crate) fn process_exit_trampoline(trampolines: &[HleTrampoline]) -> Option<u64> {
    trampolines
        .iter()
        .find(|trampoline| is_terminating(&trampoline.library, &trampoline.function))
        .map(|trampoline| trampoline.addr)
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
/// whole pipeline still enforces RT0's single guest execution. The
/// `ACTIVE_CONTEXT` route itself is thread-local and the VEH only reads the
/// faulting OS thread's slot, which is the property later workers rely on.
static CALL_LOCK: Mutex<()> = Mutex::new(());

/// Opt-in native-execution watchdog used by retail-title bring-up. A timeout
/// must produce the guest RIP/stack/register state before the launcher kills
/// the process; otherwise a spin loop and a blocked host wait are
/// indistinguishable from a silent log tail.
struct RunWatchdog {
    cancelled: Arc<AtomicBool>,
}

impl Drop for RunWatchdog {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

fn arm_run_watchdog() -> Option<RunWatchdog> {
    let delay_ms = std::env::var("XPS5X_GUEST_WATCHDOG_MS")
        .ok()?
        .parse::<u64>()
        .ok()
        .filter(|delay| *delay > 0)?;
    // SAFETY: this only reads the caller's stable OS thread identifier.
    let thread_id = unsafe { GetCurrentThreadId() };
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    std::thread::spawn(move || {
        for sample in 1..=3 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            if worker_cancelled.load(Ordering::Acquire) {
                return;
            }

            // SAFETY: `thread_id` names the still-running thread whose `run`
            // call owns this watchdog. The requested rights are exactly those
            // needed to suspend it and read its integer/control context.
            let thread =
                unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, 0, thread_id) };
            if thread.is_null() {
                return;
            }
            // SAFETY: the handle has THREAD_SUSPEND_RESUME rights. A failure is
            // reported as `u32::MAX`; in that case no matching ResumeThread is
            // due.
            let suspend_count = unsafe { SuspendThread(thread) };
            if suspend_count == u32::MAX {
                // SAFETY: `thread` is an owned non-null Win32 handle.
                unsafe { CloseHandle(thread) };
                return;
            }

            // SAFETY: a zeroed CONTEXT is valid storage; ContextFlags selects
            // all AMD64 state before GetThreadContext fills it while the target
            // is suspended.
            let mut context = unsafe { core::mem::zeroed::<CONTEXT>() };
            context.ContextFlags = CONTEXT_ALL_AMD64;
            let got_context = unsafe { GetThreadContext(thread, &mut context) } != 0;

            // Resume before logging: the target might have been suspended while
            // holding tracing's internal lock.
            // SAFETY: this balances the successful SuspendThread above, then
            // closes the owned handle exactly once.
            unsafe {
                ResumeThread(thread);
                CloseHandle(thread);
            }
            if got_context && !worker_cancelled.load(Ordering::Acquire) {
                tracing::error!(
                    "guest watchdog sample {sample}/3 after {} ms: rip={:#x} rsp={:#x} \
                     rbp={:#x} rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} \
                     rdi={:#x} fs_rearms={} hle_enter={} hle_exit={} last_hle={} \
                     last_hle_return={:#x}",
                    delay_ms * sample,
                    context.Rip,
                    context.Rsp,
                    context.Rbp,
                    context.Rax,
                    context.Rbx,
                    context.Rcx,
                    context.Rdx,
                    context.Rsi,
                    context.Rdi,
                    FSBASE_REARMS.load(Ordering::Relaxed),
                    HLE_ENTERS.load(Ordering::Relaxed),
                    HLE_EXITS.load(Ordering::Relaxed),
                    LAST_HLE_INDEX.load(Ordering::Relaxed),
                    LAST_HLE_RETURN.load(Ordering::Relaxed),
                );
            }
        }
    });
    Some(RunWatchdog { cancelled })
}

/// The process-wide VEH installation. The callback itself owns no process
/// pointer: it consults the const-initialized thread-local `ACTIVE_CONTEXT`,
/// so leaving one handler installed is safe for non-guest threads and avoids
/// stacking/removing handlers while future guest workers are live.
static VEH_HANDLE: OnceLock<usize> = OnceLock::new();

fn ensure_veh() -> Result<(), RuntimeError> {
    let handle = *VEH_HANDLE.get_or_init(|| {
        // SAFETY: `veh_callback` has the required Windows callback ABI and is
        // valid for the lifetime of the process. It only dereferences a
        // context when this faulting OS thread has installed one in TLS.
        unsafe { AddVectoredExceptionHandler(1, Some(veh_callback)) as usize }
    });
    if handle == 0 {
        Err(RuntimeError::MapFailed)
    } else {
        Ok(())
    }
}

/// Process-wide count of FS-base re-arms performed by [`veh_callback`] (see
/// [`ActiveContext::guest_fsbase`]).
///
/// Every increment is one guest `fs:`-relative access that trapped because
/// Windows had discarded our FS base at a context switch, and was transparently
/// re-armed and retried. In steady state this ticks roughly once per scheduler
/// quantum in which the guest touches TLS or its `fs:0x28` stack canary, so it
/// is a cheap, honest health/perf signal: a plausible rate is ~tens/second per
/// guest thread, while a runaway rate would mean the re-arm is not sticking.
///
/// It is also what lets the re-arm's tests assert the mechanism **actually
/// fired**, rather than passing because the guest happened never to be
/// preempted (in which case the base still matches and the re-arm arm is
/// skipped entirely) — see `guest_tls_survives_preemption_via_fsbase_rearm` and
/// `genuine_wild_fault_after_preemption_recovers_instead_of_looping_the_veh`.
static FSBASE_REARMS: AtomicU64 = AtomicU64::new(0);
static HLE_ENTERS: AtomicU64 = AtomicU64::new(0);
static HLE_EXITS: AtomicU64 = AtomicU64::new(0);
static LAST_HLE_INDEX: AtomicU64 = AtomicU64::new(u64::MAX);
static LAST_HLE_RETURN: AtomicU64 = AtomicU64::new(0);
static TRACE_HLE: OnceLock<bool> = OnceLock::new();
static TRACE_HLE_INDEX: OnceLock<Option<u64>> = OnceLock::new();

/// The number of FS-base re-arms performed since process start — see
/// [`FSBASE_REARMS`]. Monotonic; never reset.
pub fn fsbase_rearm_count() -> u64 {
    FSBASE_REARMS.load(Ordering::Relaxed)
}

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

/// How many recent HLE calls [`CallTrace`] remembers.
///
/// 256 was too few to be useful on a real title, and the way it failed is worth
/// recording. Minecraft's background worker throws a C++ exception, unwinds it,
/// and then deliberately aborts (`mov dword ptr [0], 0xDEADC0DE`). By the time
/// it faults, the whole 256-entry window holds nothing but the mutex/free
/// teardown *after* the throw — the call that actually failed had long since
/// scrolled out, which is how a savedata task's failure came to look like a
/// memory bug. The window has to be wide enough to still contain the cause when
/// the guest finally admits something went wrong.
///
/// Costs one fixed array of 40-byte cells per guarded call (~160 KB), sized once
/// at construction, never grown. `push` stays allocation-free — only the report
/// allocates, and only after the guest has already stopped.
const CALL_TRACE_LEN: usize = 4096;

/// A bounded ring of the most recent HLE calls, for explaining a fault.
///
/// # Why this exists
///
/// A guest that faults reading address 0 is almost never wrong on its own — it
/// is holding a null we handed it. Once a title gets past its own init, the
/// most common failure is an HLE stub returning `0` where the guest expects a
/// real object, and the guest then dereferencing it several calls later. The
/// faulting instruction is in *guest* code and names nothing; the culprit is in
/// the call history.
///
/// Measured motivation: the retail title's first fault after libc init is
/// exactly this shape — `Faulted { access: 0, kind: Read }` at `eboot+0x80A288`,
/// with no import to blame.
///
/// # Discipline
///
/// Written from the VEH and read by [`run`], so it uses `Cell` like the rest of
/// [`ActiveContext`]: it must survive the register-only context restore that the
/// genuine-fault path performs (a `RefCell` borrow held across that jump would
/// never be released). Nothing here allocates or logs — [`run`] does the
/// formatting, on the normal path, once the guest is no longer running.
struct CallTrace {
    /// Next slot to write; total calls made is not tracked beyond this ring.
    head: Cell<usize>,
    /// Whether the ring has wrapped (so all slots are live).
    wrapped: Cell<bool>,
    /// `(trampoline index, return value, first three arguments)`.
    ///
    /// Keeping arg0 lets the post-fault reporter recover the PCs passed to
    /// `sceKernelGetModuleInfoForUnwind`. That produces a useful guest
    /// backtrace without allocating or logging in the VEH.
    entries: [Cell<(u32, u64, [u64; 3])>; CALL_TRACE_LEN],
}

impl CallTrace {
    fn new() -> Self {
        Self {
            head: Cell::new(0),
            wrapped: Cell::new(false),
            entries: [const { Cell::new((0u32, 0u64, [0u64; 3])) }; CALL_TRACE_LEN],
        }
    }

    /// Record one serviced HLE call. Called from the VEH — allocation-free.
    fn push(&self, trampoline_idx: u32, ret: u64, args: [u64; 3]) {
        let h = self.head.get();
        self.entries[h].set((trampoline_idx, ret, args));
        let next = h + 1;
        if next == CALL_TRACE_LEN {
            self.head.set(0);
            self.wrapped.set(true);
        } else {
            self.head.set(next);
        }
    }

    /// The recorded calls, oldest first.
    fn entries_oldest_first(&self) -> Vec<(u32, u64, [u64; 3])> {
        let h = self.head.get();
        let mut out = Vec::with_capacity(CALL_TRACE_LEN);
        if self.wrapped.get() {
            for i in h..CALL_TRACE_LEN {
                out.push(self.entries[i].get());
            }
        }
        for i in 0..h {
            out.push(self.entries[i].get());
        }
        out
    }
}

/// The context [`veh_callback`] consults while a guest call is in flight.
/// Written by [`run`] before calling the guest, read (and, on an unresolved
/// trampoline, written back to) by the VEH, cleared by [`run`] after the
/// call returns. See `CALL_LOCK`'s doc comment for why this never needs its
/// own synchronization.
struct ActiveContext {
    trampolines: *const [HleTrampoline],
    /// The per-NID unresolved-stub table, so a fault inside
    /// `[UNRESOLVED_STUB_BASE, +len*8)` can name the import the guest wanted
    /// instead of reporting a bare address. Raw pointer for the same reason as
    /// `trampolines` (see `CALL_LOCK`).
    unresolved_stubs: *const [UnresolvedStub],
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
    thread_scheduler: *const dyn GuestThreadScheduler,
    current_thread: u64,
    /// This thread's static TLS block base (`tcb - tls_block_size`), when it
    /// has one. Served to HLE via
    /// [`GuestThreadScheduler::current_static_tls_block`] so `__tls_get_addr`
    /// can resolve the main module's thread-locals to the same storage
    /// `TPOFF64` accesses use, rather than to a second, zeroed block.
    ///
    /// Per-call rather than per-scheduler, exactly like `current_thread`: one
    /// `ActiveContext` guards one thread's guest execution.
    static_tls_block: Option<u64>,
    thread_exit: Cell<Option<u64>>,
    region_base: u64,
    region_len: u64,
    /// The guarded slot after the invalid-trampoline diagnostic sentinel.
    /// Guest callbacks return here so the VEH can finish the original HLE
    /// call and resume its caller.
    callback_return_addr: u64,
    tls_rearm_trampoline: u64,
    /// Top-level guest calls return through the same guarded address when no
    /// nested HLE callback frame is active.
    returned: Cell<bool>,
    retval: Cell<u64>,
    /// At most one guest callback can be requested by a single HLE handler.
    pending_guest_call: Cell<Option<GuestCallRequest>>,
    /// HLE currently executing on this OS thread. A GuestMemory write can
    /// itself fault (for example, when a bad output pointer names RX code),
    /// recursively entering the VEH before the call can be added to the
    /// completed-call trace. Keep attribution in the per-thread active
    /// context so the recovered fault names the actual writer.
    active_hle: Cell<Option<(u64, [u64; 6])>>,
    /// Nested guest callbacks are legal (an initializer may call another API
    /// that invokes guest code), so retain their original HLE return frames.
    callback_frames: RefCell<Vec<GuestCallbackFrame>>,
    error: Cell<Option<RuntimeError>>,
    /// Register state and instruction bytes captured before a genuine guest
    /// fault is long-jumped back to the host recovery point.
    fault_snapshot: Cell<Option<FaultSnapshot>>,
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
    /// The guest TCB address this call's FS base must point at whenever guest
    /// code runs — i.e. the value `run` wrote with `WRFSBASE` before entering
    /// the guest. Meaningful only while `tls_active` is `true`.
    ///
    /// Read by `veh_callback` to **re-arm** the FS base after Windows discards
    /// it. `tls.rs`'s `fsbase_does_not_survive_preemption_on_windows` pins the
    /// platform behaviour: a user-set FS base is cleared to 0 by the first
    /// context switch (a bare timer-interrupt preemption suffices — no syscall
    /// needed), because Windows restores the base from its own notion of the
    /// thread's base (0 for native x64 threads). There is no notification we
    /// could hook to re-set it — but the resulting fault *is* the notification:
    /// the guest's next `fs:`-relative access reads a near-null address and
    /// traps, and `veh_callback` re-arms and retries the instruction. Same
    /// "must live in `ctx`, not a local carried across `RtlCaptureContext`"
    /// reasoning as `orig_fsbase`.
    guest_fsbase: Cell<u64>,
    /// Set by `veh_callback` (design doc §4, wall #1 W1b) when a resolved
    /// trampoline call targets a [`TERMINATING_FUNCTIONS`] entry, alongside
    /// `exited`, just before it performs the same `recovery_ctx` longjmp the
    /// genuine-fault path uses. Lives here — not a plain local in `run` —
    /// for the exact same "must survive a register-only context restore"
    /// reason `resumed`/`orig_fsbase` do (see `resumed`'s doc comment):
    /// `veh_callback` writes it *before* the longjmp, and `run` reads it
    /// only after control has already jumped back to the recovery point.
    /// The most recent serviced HLE calls, for explaining a fault that names
    /// nothing (see [`CallTrace`]).
    trace: CallTrace,
    exit_code: Cell<u64>,
    /// Set to `true` by `veh_callback` alongside `exit_code`, immediately
    /// before the exit-longjmp (design doc §4). `run`'s shared continuation
    /// checks this *first* (before `error`) on the resumed arrival: a
    /// terminating call and a genuine fault are mutually exclusive —
    /// `veh_callback` only ever sets one of `exited`/`error` for a given
    /// call — but checking `exited` first matches the design doc's
    /// documented arrival order exactly.
    exited: Cell<bool>,
    /// Whether `recovery_ctx` has been populated and is safe to long-jump to.
    ///
    /// Set to `true` **immediately after** `RtlCaptureContext` returns and
    /// before the guest is entered; `veh_callback` returns
    /// `EXCEPTION_CONTINUE_SEARCH` while it is `false`. This closes the window
    /// between `ACTIVE_CONTEXT` being installed and `recovery_ctx` being
    /// captured: a fault in that window (e.g. inside `RtlCaptureContext`
    /// itself, or the fsbase capture just before it) must NOT be hijacked as a
    /// genuine fault, because doing `*context = *ctx.recovery_ctx` from the
    /// still-zeroed buffer would set `Rip = 0` and spin the VEH in an infinite
    /// fault loop until the host stack overflows. Defense in depth: with the
    /// `AlignedContext` fix the `movaps` `#GP` that used to open this window is
    /// gone, but any other pre-capture fault is now handled safely too.
    armed: Cell<bool>,
}

#[derive(Debug, Clone, Copy)]
struct GuestCallbackFrame {
    original_return: u64,
    hle_result: u64,
    completion: Option<GuestCallCompletion>,
}

impl GuestCallScheduler for ActiveContext {
    fn request(&self, request: GuestCallRequest) -> bool {
        if self.pending_guest_call.get().is_some() {
            return false;
        }
        self.pending_guest_call.set(Some(request));
        true
    }
}

impl GuestThreadScheduler for ActiveContext {
    fn create(&self, thread_out: u64, attr: u64, entry: u64, arg: u64) -> u64 {
        // SAFETY: `thread_scheduler` is installed by `run` from a scheduler
        // that outlives this guarded call.
        unsafe { &*self.thread_scheduler }.create(thread_out, attr, entry, arg)
    }

    fn join(&self, thread: u64, retval_out: u64) -> u64 {
        if thread == self.current_thread {
            return 0x8002_0023;
        }
        // SAFETY: same lifetime invariant as `create`.
        unsafe { &*self.thread_scheduler }.join(thread, retval_out)
    }

    fn detach(&self, thread: u64) -> u64 {
        // SAFETY: same lifetime invariant as `create`.
        unsafe { &*self.thread_scheduler }.detach(thread)
    }

    fn request_exit(&self, retval: u64) -> bool {
        self.thread_exit.set(Some(retval));
        true
    }

    fn current_thread(&self) -> u64 {
        self.current_thread
    }

    fn current_static_tls_block(&self) -> Option<u64> {
        self.static_tls_block
    }

    fn request_process_exit(&self, code: u64) {
        // SAFETY: same lifetime invariant as `create`.
        unsafe { &*self.thread_scheduler }.request_process_exit(code);
    }

    fn process_is_terminating(&self) -> bool {
        // SAFETY: same lifetime invariant as `create`.
        unsafe { &*self.thread_scheduler }.process_is_terminating()
    }
}

/// Resume native guest code through the guest-side WRFSBASE stub whenever
/// this run owns a TCB. A VEH executes with Windows' thread environment, and
/// returning directly can silently expose the host TEB to the next `fs:`
/// access when that address happens to be readable. Staging the write after
/// NtContinue closes that leak at every HLE/callback/syscall safe point.
fn resume_guest_with_tls(
    ctx: &ActiveContext,
    context: &mut CONTEXT,
    mem: &dyn GuestMemory,
    target_rip: u64,
    target_rsp: u64,
) {
    context.Rip = target_rip;
    context.Rsp = target_rsp;
    if !ctx.tls_active.get() {
        return;
    }

    let Some(staged_rsp) = target_rsp.checked_sub(16) else {
        return;
    };
    if mem.write(staged_rsp, &context.R11.to_le_bytes())
        && mem.write(staged_rsp + 8, &target_rip.to_le_bytes())
    {
        context.Rsp = staged_rsp;
        context.R11 = ctx.guest_fsbase.get();
        context.Rip = ctx.tls_rearm_trampoline;
        FSBASE_REARMS.fetch_add(1, Ordering::Relaxed);
    }
}

struct UnsupportedGuestThreads;

impl GuestThreadScheduler for UnsupportedGuestThreads {
    fn create(&self, _thread_out: u64, _attr: u64, _entry: u64, _arg: u64) -> u64 {
        0x8002_000B
    }

    fn join(&self, _thread: u64, _retval_out: u64) -> u64 {
        0x8002_0003
    }

    fn detach(&self, _thread: u64) -> u64 {
        0x8002_0003
    }

    fn request_exit(&self, _retval: u64) -> bool {
        false
    }

    fn current_thread(&self) -> u64 {
        1
    }

    fn request_process_exit(&self, _code: u64) {}

    fn process_is_terminating(&self) -> bool {
        false
    }
}

static UNSUPPORTED_GUEST_THREADS: UnsupportedGuestThreads = UnsupportedGuestThreads;

#[derive(Clone, Copy)]
struct FaultSnapshot {
    rip: u64,
    rsp: u64,
    rbp: u64,
    rax: u64,
    rbx: u64,
    rcx: u64,
    rdx: u64,
    rsi: u64,
    rdi: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    // Callee-saved registers: at a deliberate-abort site the argument
    // registers are already clobbered by the assert handler that just
    // returned, but the values it was CALLED with (message pointers
    // especially) were staged from these and usually survive.
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    bytes: [u8; 16],
    bytes_read: bool,
}

/// A `CONTEXT` forced to the SDK's mandated 16-byte alignment.
///
/// windows-sys 0.59 declares x86-64 `CONTEXT` as plain `#[repr(C)]`, whose
/// largest field (`M128A`) is only 8-aligned, so `align_of::<CONTEXT>() == 8`.
/// But the Windows SDK declares it `DECLSPEC_ALIGN(16)`, and
/// `RtlCaptureContext` stores the XMM registers with `movaps`, which raises
/// `#GP` on a 16-misaligned destination (surfaced on Windows as an
/// `EXCEPTION_ACCESS_VIOLATION` at address `-1`). A bare `CONTEXT` stack local
/// lands at a 16-aligned slot only by chance of the current frame layout; any
/// change to `run`'s frame (e.g. adding `ActiveContext` fields) can silently
/// flip it to 8-mod-16 and detonate the fault loop described on `armed`.
/// Wrapping in an `align(16)` newtype makes the alignment guaranteed rather
/// than incidental.
#[repr(C, align(16))]
struct AlignedContext(CONTEXT);

fn never_returns(value: core::convert::Infallible) -> ! {
    match value {}
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
/// or a genuine fault (`Err(Faulted)`, RT1a). Function entries, dependency
/// initializers, and process `_start` all enter through
/// [`crate::stack::enter_guest`] and funnel through this function so the
/// fault-recovery/exit-longjmp machinery is never duplicated.
///
/// # Safety
/// `call_guest` must, when called, transfer control to mapped guest code the
/// caller set up specifically so it can be executed (i.e. a live
/// [`crate::arena::GuestArena`]'s image sub-region) exactly as
/// [`crate::stack::enter_guest`]'s safety contract requires. It diverges;
/// normal guest returns fault at the guarded return trampoline and resume
/// through the captured recovery context.
/// It must be safe to call exactly once, synchronously, on this thread, for
/// the duration this function's process-wide VEH and per-call RT1a recovery
/// point are armed. The handler is installed before entry and remains for the
/// process lifetime; the recovery context is read only before this returns.
/// `trampolines`, `hle`, `kernel`, `mem`, and `alloc` must outlive
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
    unresolved_stubs: &[UnresolvedStub],
    hle: &HleRegistry,
    kernel: &OrbisKernel,
    mem: &dyn GuestMemory,
    alloc: &dyn GuestAllocator,
    guard: &TrampolineGuard,
    tcb: Option<u64>,
    // This thread's static TLS block base — `tcb - tls_block_size`, which only
    // the caller can compute since only it knows the module's `PT_TLS` size.
    // `None` when the module has no `PT_TLS`.
    static_tls_block: Option<u64>,
    guest_threads: Option<&dyn GuestThreadScheduler>,
    current_thread: u64,
    call_guest: impl FnOnce() -> core::convert::Infallible,
) -> Result<RunOutcome, RuntimeError> {
    let _watchdog = arm_run_watchdog();
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
    // never reads it). Wrapped in `AlignedContext` so `RtlCaptureContext`'s
    // `movaps` XMM stores hit a 16-aligned buffer (see `AlignedContext`).
    let mut recovery = AlignedContext(unsafe { core::mem::zeroed::<CONTEXT>() });
    debug_assert_eq!(
        (&recovery as *const AlignedContext as usize) % 16,
        0,
        "recovery CONTEXT must be 16-aligned for RtlCaptureContext's movaps stores"
    );

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
    let thread_scheduler = guest_threads.unwrap_or(&UNSUPPORTED_GUEST_THREADS);
    // SAFETY: shared schedulers are Arc-owned by the process, while the
    // fallback is static; either way the scheduler outlives this run.
    let thread_scheduler_erased: &'static dyn GuestThreadScheduler = unsafe {
        core::mem::transmute::<&dyn GuestThreadScheduler, &'static dyn GuestThreadScheduler>(
            thread_scheduler,
        )
    };

    let ctx = ActiveContext {
        trampolines: trampolines as *const [HleTrampoline],
        unresolved_stubs: unresolved_stubs as *const [UnresolvedStub],
        hle: hle as *const HleRegistry,
        kernel: kernel as *const OrbisKernel,
        mem: mem_erased as *const dyn GuestMemory,
        alloc: alloc_erased as *const dyn GuestAllocator,
        thread_scheduler: thread_scheduler_erased as *const dyn GuestThreadScheduler,
        current_thread,
        static_tls_block,
        thread_exit: Cell::new(None),
        region_base: guard.base(),
        region_len: guard.len(),
        callback_return_addr: guard.return_trampoline(),
        tls_rearm_trampoline: guard.tls_rearm_trampoline(),
        returned: Cell::new(false),
        retval: Cell::new(0),
        pending_guest_call: Cell::new(None),
        active_hle: Cell::new(None),
        callback_frames: RefCell::new(Vec::new()),
        error: Cell::new(None),
        fault_snapshot: Cell::new(None),
        resumed: Cell::new(false),
        recovery_ctx: &recovery.0 as *const CONTEXT,
        orig_fsbase: Cell::new(0),
        tls_active: Cell::new(false),
        guest_fsbase: Cell::new(0),
        trace: CallTrace::new(),
        exit_code: Cell::new(0),
        exited: Cell::new(false),
        armed: Cell::new(false),
    };

    ensure_veh()?;

    if let Ok(value) = std::env::var("XPS5X_DIAGNOSTIC_CODE_ADDR") {
        let value = value.trim_start_matches("0x");
        if let Ok(address) = u64::from_str_radix(value, 16)
            && let Some(disassembly) = diagnostic_disassembly(mem, address, 0x400)
        {
            tracing::info!("diagnostic guest disassembly from {address:#x}:\n{disassembly}");
        }
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
    if let Some(tcb_addr) = tcb.filter(|_| crate::tls::fsgsbase_available()) {
        ctx.tls_active.set(true);
        // The address `veh_callback` re-arms the FS base to after Windows
        // discards it at a context switch (see `guest_fsbase`'s doc comment).
        ctx.guest_fsbase.set(tcb_addr);
        // SAFETY: `fsgsbase_available()` just returned `true`, so `RDFSBASE`
        // is permitted on this CPU. This only reads the current FS base; it
        // does not modify any CPU state.
        ctx.orig_fsbase.set(unsafe { crate::tls::read_fsbase() });
    }

    // SAFETY: `RtlCaptureContext` only requires a valid, writable, 16-aligned
    // `CONTEXT` out-pointer, which `&mut recovery.0` (an `AlignedContext`
    // field) is. It captures the calling thread's complete register state as
    // of right now: `Rip` becomes the address right after this call returns,
    // and `Rsp`/`Rbp`/every GPR reflect exactly this point in `run`'s frame.
    // This is RT1a's recovery point — see this module's doc comment and
    // `veh_callback`.
    unsafe { RtlCaptureContext(&mut recovery.0) };

    // `recovery` is now populated — from here a genuine-fault longjmp to it is
    // safe. Arm the VEH gate (idempotent across the resumed-arrival re-run of
    // this point). Any fault *before* here (incl. inside `RtlCaptureContext`)
    // sees `armed == false` and is passed through, never hijacked to a
    // zeroed context. See `ActiveContext::armed`.
    ctx.armed.set(true);

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
    if !ctx.resumed.get() {
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
        // guest code exactly as `enter_guest`'s safety contract requires. The VEH is
        // armed (just above) for the entire duration of this call, so any
        // guest `call [import_slot]` into the guarded trampoline region is
        // trapped and serviced by `veh_callback`, and any genuine wild fault
        // is recovered via the `RtlCaptureContext` snapshot just taken
        // above, rather than crashing the process.
        //
        // `entry` runs on the guest's own stack. `enter_guest` switches to it
        // and never returns through this host frame; every terminal path
        // restores the pre-entry context instead —
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
        // This rustc warns about applying a diverging eliminator to an
        // uninhabited result even though that is precisely the stable
        // `FnOnce -> Infallible` spelling used in place of `FnOnce -> !`.
        #[allow(unreachable_code)]
        never_returns(call_guest())
    }

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
        Some(err) => {
            // A fault that names nothing is usually our fault, not the guest's:
            // an HLE stub handed back a null (or a bogus handle) and the guest
            // dereferenced it later. The faulting instruction is in guest code
            // and identifies no symbol, so print the recent HLE history — that
            // is where the culprit is. Done HERE, not in the VEH: this
            // thread's active context is cleared and the guest is no longer
            // running, so this can safely allocate and log.
            if matches!(
                err,
                RuntimeError::Faulted { .. } | RuntimeError::UnimplementedImport { .. }
            ) {
                log_call_trace(&ctx, trampolines, &err);
            }
            Err(err)
        }
        None if ctx.returned.get() => Ok(RunOutcome::Returned(ctx.retval.get())),
        None => Err(RuntimeError::MapFailed),
    }
}

/// Print the most recent HLE calls preceding a fault, oldest first.
///
/// Deliberately reports the **return value** of each call alongside its name: a
/// `-> 0x0` a few lines above a `Faulted { access: 0 }` is very often the whole
/// story.
fn read_fault_cstr(mem: &dyn GuestMemory, address: u64) -> Option<String> {
    if address == 0 {
        return None;
    }
    let mut bytes = Vec::with_capacity(128);
    for offset in 0..128u64 {
        let mut byte = [0u8; 1];
        if !mem.read(address.wrapping_add(offset), &mut byte) {
            return None;
        }
        if byte[0] == 0 {
            break;
        }
        if !(byte[0].is_ascii_graphic() || byte[0] == b' ' || byte[0] == b'\t') {
            return None;
        }
        bytes.push(byte[0]);
    }
    if bytes.is_empty() {
        None
    } else {
        String::from_utf8(bytes).ok()
    }
}

fn log_call_trace(ctx: &ActiveContext, trampolines: &[HleTrampoline], err: &RuntimeError) {
    if let Some((idx, args)) = ctx.active_hle.get() {
        if let Some(t) = trampolines.get(idx as usize) {
            tracing::warn!(
                "fault occurred inside HLE #{idx} {}::{} args=[{:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}]",
                t.library,
                t.function,
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
            );
        }
    }
    if let Some(snapshot) = ctx.fault_snapshot.get() {
        let instruction = if snapshot.bytes_read {
            snapshot
                .bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            "<unreadable>".to_string()
        };
        tracing::warn!(
            "fault snapshot: rip={:#x} rsp={:#x} rbp={:#x}\n  \
             rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x}\n  \
             rsi={:#x} rdi={:#x} r8={:#x} r9={:#x} r10={:#x} r11={:#x}\n  \
             instruction bytes: {instruction}",
            snapshot.rip,
            snapshot.rsp,
            snapshot.rbp,
            snapshot.rax,
            snapshot.rbx,
            snapshot.rcx,
            snapshot.rdx,
            snapshot.rsi,
            snapshot.rdi,
            snapshot.r8,
            snapshot.r9,
            snapshot.r10,
            snapshot.r11,
        );

        // Which module was the guest actually in? A bare rip names nothing in a
        // multi-module process: eyeballing 0x1000111c640c against the wrong
        // dependency's base twice put this investigation in the eboot when the
        // fault was in libRenoirCore.PS5.prx all along. The kernel's unwind table
        // already carries every loaded module's [start, end) — it is populated
        // for `sceKernelGetModuleInfoForUnwind` — so the answer costs one lookup.
        //
        // SAFETY: `ctx.kernel` is the live kernel installed by `run` for this
        // guarded call, on the same thread, and this diagnostic runs
        // synchronously before that runner drops it.
        let kernel = unsafe { &*ctx.kernel };
        match kernel.unwind_module_for_addr(snapshot.rip) {
            Some(module) => tracing::warn!(
                "fault module: {} at +{:#x} (module {:#x}..{:#x})",
                module.name,
                snapshot.rip - module.start,
                module.start,
                module.end
            ),
            // Not in any loaded module: the guest was executing somewhere it was
            // never given code — a wild jump, or a call through a slot holding
            // something that is not a function.
            None => tracing::warn!(
                "fault module: rip {:#x} is in NO loaded module — the guest jumped somewhere it \
                 has no code",
                snapshot.rip
            ),
        }

        // The GuestMemory object is owned by the still-live process runner;
        // this diagnostic runs synchronously before that runner drops it and
        // after the VEH has stopped consulting the active context.
        let mem = unsafe { &*ctx.mem };
        let fault_window_start = snapshot.rip.saturating_sub(32);
        let mut fault_window = [0u8; 64];
        if mem.read(fault_window_start, &mut fault_window) {
            let bytes = fault_window
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            tracing::warn!(
                "guest fault-site bytes: {fault_window_start:#x}..{:#x} (RIP +0x20): {bytes}",
                fault_window_start + fault_window.len() as u64
            );
        }
        // What each register POINTS AT, previewed as text. When a guest
        // aborts deliberately, the abort call's arguments (file, function,
        // and — decisively — the formatted exception/assert MESSAGE) are
        // still sitting in argument registers, and the message is the one
        // fact no register dump or stack walk can recover. Bounded reads,
        // printable-ASCII runs only, and only registers that actually point
        // at readable guest memory produce a line.
        for (name, value) in [
            ("rax", snapshot.rax),
            ("rcx", snapshot.rcx),
            ("rdx", snapshot.rdx),
            ("rsi", snapshot.rsi),
            ("rdi", snapshot.rdi),
            ("r8", snapshot.r8),
            ("r9", snapshot.r9),
            ("r10", snapshot.r10),
            ("rbx", snapshot.rbx),
            ("r12", snapshot.r12),
            ("r13", snapshot.r13),
            ("r14", snapshot.r14),
            ("r15", snapshot.r15),
        ] {
            let mut preview = [0u8; 96];
            if value == 0 || !mem.read(value, &mut preview) {
                continue;
            }
            let printable: String = preview
                .iter()
                .take_while(|&&byte| byte != 0)
                .map(|&byte| {
                    if (0x20..0x7f).contains(&byte) {
                        byte as char
                    } else {
                        '.'
                    }
                })
                .collect();
            // Only worth a line if it plausibly IS text: mostly printable
            // and at least a few characters long.
            let text_like = printable.len() >= 4
                && printable.chars().filter(|c| *c != '.').count() * 4 >= printable.len() * 3;
            if text_like {
                tracing::warn!("register text preview: {name} -> {value:#x} \"{printable}\"");
            }
        }

        let mut stack_words = Vec::new();
        for index in 0..16u64 {
            let address = snapshot.rsp.wrapping_add(index * 8);
            let mut bytes = [0u8; 8];
            if !mem.read(address, &mut bytes) {
                break;
            }
            stack_words.push((address, u64::from_le_bytes(bytes)));
        }
        if !stack_words.is_empty() {
            let formatted = stack_words
                .iter()
                .map(|(address, value)| format!("{address:#x}:{value:#x}"))
                .collect::<Vec<_>>()
                .join(" ");
            tracing::warn!("guest stack qwords: {formatted}");

            // A wild indirect jump can fault before a normal function
            // prologue, so RSP[0] is not necessarily the return address. Scan
            // the bounded stack sample for values that actually fall inside a
            // loaded module and show their code windows. This turns a jump into
            // data (as seen in a title-supplied libc) into an actionable caller
            // without guessing which stack word owns the return slot.
            let mut shown = 0usize;
            for (stack_address, candidate) in &stack_words {
                let Some(module) = kernel.unwind_module_for_addr(*candidate) else {
                    continue;
                };
                let window_start = candidate.saturating_sub(16);
                let mut return_window = [0u8; 32];
                if !mem.read(window_start, &mut return_window) {
                    continue;
                }
                let bytes = return_window
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                tracing::warn!(
                    "guest stack code candidate at {stack_address:#x}: {}+{:#x} ({candidate:#x}), \
                     bytes {bytes}",
                    module.name,
                    candidate - module.start,
                );
                shown += 1;
                if shown == 6 {
                    break;
                }
            }
        }
    }
    let entries = ctx.trace.entries_oldest_first();
    if entries.is_empty() {
        tracing::warn!("{err} — no HLE calls were serviced before it");
        return;
    }
    tracing::warn!(
        "{err}\n  the {} most recent HLE call(s) before it (oldest first) — a call returning \
         0x0 here is the usual cause of a null dereference in guest code:",
        entries.len()
    );
    let mut unwind_pcs = Vec::new();
    for &(idx, _ret, args) in &entries {
        let is_unwind = trampolines
            .get(idx as usize)
            .is_some_and(|t| t.function == "sceKernelGetModuleInfoForUnwind");
        if is_unwind && unwind_pcs.last().copied() != Some(args[0]) {
            unwind_pcs.push(args[0]);
        }
    }
    if !unwind_pcs.is_empty() {
        let pcs = unwind_pcs
            .iter()
            .map(|pc| format!("{pc:#x}"))
            .collect::<Vec<_>>()
            .join(" -> ");
        tracing::warn!("guest unwind PC lookups (most recent exception): {pcs}");
    }
    let mem = unsafe { &*ctx.mem };
    let mut recent_strings = Vec::new();
    for &(idx, _ret, args) in &entries {
        let Some(t) = trampolines.get(idx as usize) else {
            continue;
        };
        let pointers: &[u64] = match t.function.as_str() {
            "strlen" => &args[..1],
            "strcpy" | "strncpy" => &args[1..2],
            "strcmp" => &args[..2],
            _ => continue,
        };
        for &pointer in pointers {
            if let Some(value) = read_fault_cstr(mem, pointer) {
                let item = format!("{}({pointer:#x})={value:?}", t.function);
                if !recent_strings.contains(&item) {
                    recent_strings.push(item);
                }
            }
        }
    }
    if !recent_strings.is_empty() {
        let start = recent_strings.len().saturating_sub(24);
        tracing::warn!(
            "recent guest strings observed by libc:\n    {}",
            recent_strings[start..].join("\n    ")
        );
    }
    let mut failures = Vec::new();
    for (idx, ret, _args) in entries {
        let marker = if is_orbis_error(ret) {
            "  <-- ERROR"
        } else {
            ""
        };
        match trampolines.get(idx as usize) {
            Some(t) => {
                tracing::warn!("    {}::{} -> {ret:#x}{marker}", t.library, t.function);
                if !marker.is_empty() {
                    let item = format!("{}::{} -> {ret:#x}", t.library, t.function);
                    if !failures.contains(&item) {
                        failures.push(item);
                    }
                }
            }
            None => tracing::warn!("    <trampoline #{idx}> -> {ret:#x}{marker}"),
        }
    }
    // The list above is thousands of lines and almost all of it succeeded. What
    // a reader actually wants is the handful that did not: a guest that throws,
    // asserts, or dereferences a null it was handed is usually reacting to one
    // of these, and picking them out by eye across 4096 entries is not a thing
    // anyone should have to do.
    if failures.is_empty() {
        tracing::warn!("no HLE call returned an Orbis error before this fault");
    } else {
        tracing::warn!(
            "HLE calls that returned an ERROR before this fault ({} distinct) — \
             the guest's own failure handling usually starts at one of these:\n    {}",
            failures.len(),
            failures.join("\n    ")
        );
    }
}

/// Whether an HLE return value looks like an Orbis error code rather than a
/// result.
///
/// Orbis errors are 32-bit and always have the top bit set (`0x8...`), e.g.
/// `SCE_KERNEL_ERROR_EINVAL = 0x8002_0016`. A legitimate pointer cannot be
/// confused with one: the guest arena is based at [`crate::GUEST_ARENA_BASE`]
/// (16 TiB), so every real guest pointer is far wider than 32 bits. A plain
/// small integer result (a length, a count, a fd) never has bit 31 set either.
///
/// This is a heuristic for a *diagnostic*, not a control-flow decision — a false
/// positive costs one misleading line in a fault report, never a behaviour
/// change.
fn is_orbis_error(ret: u64) -> bool {
    ret <= u32::MAX as u64 && (ret as u32) & 0x8000_0000 != 0
}

fn instruction_uses_fs(mem: &dyn GuestMemory, rip: u64) -> bool {
    let mut bytes = [0u8; 15];
    if !mem.read(rip, &mut bytes) {
        return false;
    }
    let mut decoder = Decoder::with_ip(64, &bytes, rip, DecoderOptions::NONE);
    decoder.decode().segment_prefix() == Register::FS
}

fn diagnostic_disassembly(mem: &dyn GuestMemory, start: u64, len: usize) -> Option<String> {
    let mut bytes = vec![0u8; len];
    if !mem.read(start, &mut bytes) {
        return None;
    }
    let mut decoder = Decoder::with_ip(64, &bytes, start, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    let mut lines = Vec::new();
    while decoder.can_decode() && lines.len() < 48 {
        let instruction = decoder.decode();
        let mut rendered = String::new();
        formatter.format(&instruction, &mut rendered);
        lines.push(format!("{:#x}: {rendered}", instruction.ip()));
    }
    Some(lines.join("\n"))
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

    let is_access_violation = record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION;
    let is_illegal_instruction = record.ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION;
    if !is_access_violation && !is_illegal_instruction {
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

    if !ctx.armed.get() {
        // A fault in the window between `ACTIVE_CONTEXT` being installed and
        // `recovery_ctx` being captured (e.g. inside `RtlCaptureContext`, or
        // the fsbase read just before it). `recovery_ctx` is still zeroed, so
        // hijacking this would longjmp to `Rip = 0` and spin the VEH into an
        // infinite fault loop. Pass it through instead. See
        // `ActiveContext::armed`.
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // SAFETY: `info.ContextRecord` is valid per the VEH contract; mutable
    // access is required to redirect execution (Rip/Rsp/Rax) below.
    let context = unsafe { &mut *info.ContextRecord };
    let fault_addr = context.Rip;

    // Process exit is cooperative across native guest workers. Every import,
    // guarded return, or FS-rearm fault is a safe point at which a worker can
    // abandon its guest stack through the already-captured recovery context.
    // A trap-free worker intentionally keeps teardown waiting; force-killing
    // a host thread could leave locks and Rust frames corrupted.
    if ctx.process_is_terminating() {
        ctx.exited.set(true);
        *context = unsafe { *ctx.recovery_ctx };
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    if is_illegal_instruction {
        let mem = unsafe { &*ctx.mem };
        let mut marker = [0u8; 2];
        if !mem.read(fault_addr, &mut marker)
            || marker != xps5x_firmware::dynlib::linker::SYSCALL_TRAP_BYTES
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }

        // FreeBSD/Orbis x86-64 syscall ABI: number in RAX and six arguments
        // in RDI, RSI, RDX, R10, R8, R9. A successful syscall clears CF and
        // returns its value in RAX; an error sets CF and returns errno in RAX.
        // RCX/R11 receive the post-syscall RIP/RFLAGS just as real hardware's
        // SYSCALL instruction would clobber them.
        let number = context.Rax;
        let args = [
            context.Rdi,
            context.Rsi,
            context.Rdx,
            context.R10,
            context.R8,
            context.R9,
        ];
        let return_rip = fault_addr.wrapping_add(marker.len() as u64);
        context.Rcx = return_rip;
        context.R11 = context.EFlags as u64;
        let kernel = unsafe { &*ctx.kernel };
        match kernel.dispatch_syscall(number, &args) {
            Ok(value) => {
                context.Rax = value;
                context.EFlags &= !1;
                tracing::debug!("Orbis syscall {number}({args:x?}) -> {value:#x}");
            }
            Err(error) => {
                let errno = match error {
                    xps5x_core::error::KernelError::UnimplementedSyscall { .. } => 78, // ENOSYS
                    xps5x_core::error::KernelError::MmapFailed { .. } => 12,           // ENOMEM
                    xps5x_core::error::KernelError::InvalidMemoryAccess(_) => 14,      // EFAULT
                    xps5x_core::error::KernelError::ThreadCreationFailed(_) => 11,     // EAGAIN
                    xps5x_core::error::KernelError::FileNotFound(_) => 2,              // ENOENT
                    xps5x_core::error::KernelError::PermissionDenied(_) => 13,         // EACCES
                };
                context.Rax = errno;
                context.EFlags |= 1;
                tracing::warn!("Orbis syscall {number}({args:x?}) -> errno {errno} ({error})");
            }
        }
        let return_rsp = context.Rsp;
        resume_guest_with_tls(ctx, context, mem, return_rip, return_rsp);
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    // A callback requested by an HLE handler returned through the dedicated
    // extra guard slot. Finish its synchronization update and resume exactly
    // where the original import call would have returned.
    if fault_addr == ctx.callback_return_addr {
        if let Some(frame) = ctx.callback_frames.borrow_mut().pop() {
            // SAFETY: `ctx.mem` is the live guest memory view stored by `run`
            // for this guarded call; callback returns are delivered
            // synchronously on the same thread before `run` can tear it down.
            let mem = unsafe { &*ctx.mem };
            if let Some(completion) = frame.completion {
                let _ = mem.atomic_store_u32(completion.address, completion.success_u32);
            }
            context.Rax = frame.hle_result;
            resume_guest_with_tls(ctx, context, mem, frame.original_return, context.Rsp);
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        // No nested HLE callback owns this return, so the top-level guest
        // entry returned normally. Capture RAX and recover the complete host
        // register context, including RSP, from `run`'s pre-entry snapshot.
        ctx.retval.set(context.Rax);
        ctx.returned.set(true);
        // SAFETY: recovery was captured and armed before guest entry, points
        // into the still-live `run` frame on this thread, and CONTEXT is Copy.
        *context = unsafe { *ctx.recovery_ctx };
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    if fault_addr < ctx.region_base || fault_addr >= ctx.region_base + ctx.region_len {
        // Before treating this as a genuine fault: is it just Windows having
        // discarded our FS base at the last context switch?
        //
        // `tls.rs`'s `fsbase_does_not_survive_preemption_on_windows` pins the
        // platform behaviour — a user-set FS base is cleared to 0 by the first
        // preemption (no syscall needed). Guest code compiled with TLS or a
        // stack protector then reads `fs:[...]`, which with a zeroed base
        // resolves to a near-null address and traps here. Windows gives us no
        // notification we could hook to re-set the base, but this fault *is*
        // the notification: re-arm and retry the faulting instruction.
        //
        // Retrying is safe and cannot loop: the retry runs with the FS base
        // restored, so if the access still faults, the base now matches
        // `guest_fsbase`, this arm is skipped, and the genuine-fault path below
        // runs — i.e. at most one extra trap per genuinely-faulting access. The
        // steady-state cost is one trap per preemption that precedes an `fs:`
        // access (~one per scheduler quantum), which is negligible.
        //
        // Nothing here disturbs the RT1a recovery contract: we resume the
        // *faulting* instruction with the delivered context untouched, and the
        // x64 `CONTEXT` has no FS-base field, so the re-armed base survives the
        // `EXCEPTION_CONTINUE_EXECUTION` return (the property the original
        // RT2c-b spike did verify).
        if ctx.tls_active.get() {
            let want = ctx.guest_fsbase.get();
            let access_addr = record.ExceptionInformation.get(1).copied().unwrap_or(0) as u64;
            let mem = unsafe { &*ctx.mem };
            // During exception dispatch Windows can temporarily present the
            // guest FS base to RDFSBASE, then clear it again in NtContinue.
            // The effective fault address is stronger evidence: an FS-prefixed
            // access resolving in the low 4 GiB (or the sign-extended top
            // range produced by a negative TPOFF from base zero) necessarily
            // executed without the high guest TCB base.
            let zero_based_fs_fault = instruction_uses_fs(mem, context.Rip)
                && (access_addr < 0x1_0000_0000 || access_addr >= 0xFFFF_8000_0000_0000);
            // SAFETY: `tls_active` is only ever `true` when
            // `fsgsbase_available()` returned `true` (checked in `run` before
            // `RtlCaptureContext`), so `RDFSBASE`/`WRFSBASE` are permitted on
            // this CPU. We are on the faulting thread (vectored exceptions are
            // delivered synchronously), which is exactly the thread whose FS
            // base must be re-armed.
            if unsafe { crate::tls::read_fsbase() } != want || zero_based_fs_fault {
                // Returning directly after WRFSBASE is racy: Windows resumes
                // through NtContinue and may restore FS=0 again before the
                // faulting instruction executes. Stage a tiny guest-side
                // trampoline instead, so WRFSBASE is the first instruction
                // after the OS has completed exception return.
                let Some(staged_rsp) = context.Rsp.checked_sub(16) else {
                    return EXCEPTION_CONTINUE_SEARCH;
                };
                if mem.write(staged_rsp, &context.R11.to_le_bytes())
                    && mem.write(staged_rsp + 8, &context.Rip.to_le_bytes())
                {
                    context.Rsp = staged_rsp;
                    context.R11 = want;
                    context.Rip = ctx.tls_rearm_trampoline;
                    FSBASE_REARMS.fetch_add(1, Ordering::Relaxed);
                    return EXCEPTION_CONTINUE_EXECUTION;
                }
            }
        }

        // Second chance, same shape as the FS re-arm above: is the guest simply
        // touching a range it RESERVED but that carries no memory yet?
        //
        // `sceKernelReserveVirtualRange` hands out address space, not memory,
        // and titles reserve far more than they touch (Until Dawn: 512 GiB) —
        // so a reservation cannot be committed eagerly. But those titles then
        // use the reservation directly, and this fault is the only notification
        // that they have. Back the page and retry the instruction.
        //
        // Cannot loop: the retry runs against committed memory, and if the
        // access still faults, `commit_on_demand` returns `false` the second
        // time (the page is now in `sparse_mappings`), so the genuine-fault path
        // below runs — at most one extra trap per genuinely-faulting access,
        // exactly the FS-re-arm bargain. A wild pointer that happens to land in
        // the sparse tail but outside every reservation is declined here and
        // still reported as the fault it is.
        //
        // SAFETY: `ctx.alloc` is the live allocator stored by `run` for this
        // guarded call; vectored exceptions are delivered synchronously on the
        // faulting thread, so it cannot be torn down underneath us.
        let access_addr = record.ExceptionInformation.get(1).copied().unwrap_or(0) as u64;
        if unsafe { &*ctx.alloc }.commit_on_demand(access_addr) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        // Outside our guarded trampoline region: a genuine guest fault.
        // Recover rather than crash (RT1a): record the error, then
        // overwrite the delivered context with the pre-call snapshot `run`
        // took via `RtlCaptureContext`, and resume there. See this module's
        // doc comment for the full control-flow argument.
        //
        // But first: is this an *unresolved import* the guest just called?
        // The linker gives every distinct missing NID its own stub slot at
        // `UNRESOLVED_STUB_BASE + i*8` and patches that symbol's relocations
        // with it, so the faulting address IS the symbol's identity. That is
        // the difference between "guest fault at 0x5000_0000_0000" (which
        // names nothing) and "guest called nid 0x… — implement it next".
        //
        // The stub region is deliberately never mapped: reaching it always
        // traps, and it traps *here*, in the generic arm, because it is
        // outside the HLE guard window. Recovery is identical either way —
        // only the reported error differs.
        //
        // What was the instruction touching, and how? Windows puts the access
        // type in ExceptionInformation[0] and the faulting data address in
        // [1]. `Rip` alone only says *where* the guest was; the access address
        // is usually what identifies the bug.
        let access = record.ExceptionInformation.get(1).copied().unwrap_or(0) as u64;
        let kind = RuntimeError::fault_kind(
            record.ExceptionInformation.first().copied().unwrap_or(0) as u64,
        );

        // SAFETY: `ctx.unresolved_stubs` was set by `run` from a live borrow
        // that outlives the guarded call, and only one thread is inside a
        // guarded call at a time (see `CALL_LOCK`), so this read is sound.
        let stubs = unsafe { &*ctx.unresolved_stubs };

        // An unresolved import can be reached two ways, and both must name it:
        //  * CALLED  — Rip *is* the stub (execution jumped there); or
        //  * READ    — the guest loaded/dereferenced the slot, so Rip is
        //              ordinary code and the *access* address is the stub.
        // Only the first was handled before, which meant a data import left at
        // a stub reported an anonymous `Faulted` at whatever innocent
        // instruction happened to read it — hiding the actual missing symbol.
        let err = match stub::resolve(fault_addr, stubs).or_else(|| stub::resolve(access, stubs)) {
            Some(s) => RuntimeError::UnimplementedImport {
                nid: s.nid,
                addr: fault_addr,
            },
            None => RuntimeError::Faulted {
                addr: fault_addr,
                access,
                kind,
            },
        };
        ctx.error.set(Some(err));

        let mut bytes = [0u8; 16];
        // SAFETY: `ctx.mem` points to the live guest memory view for the
        // duration of this guarded call (same invariant used below for HLE
        // dispatch). `read` is bounds-checked and leaves diagnosis honest if
        // RIP itself is not readable.
        let mem = unsafe { &*ctx.mem };
        // A failed callback must release any once/in-progress state it owns.
        // Unwind every nested frame because the recovery jump abandons the
        // whole guest stack, not merely the innermost callback.
        for frame in ctx.callback_frames.borrow_mut().drain(..).rev() {
            if let Some(completion) = frame.completion {
                let _ = mem.atomic_store_u32(completion.address, completion.failure_u32);
            }
        }
        let bytes_read = mem.read(context.Rip, &mut bytes);
        ctx.fault_snapshot.set(Some(FaultSnapshot {
            rip: context.Rip,
            rsp: context.Rsp,
            rbp: context.Rbp,
            rax: context.Rax,
            rbx: context.Rbx,
            rcx: context.Rcx,
            rdx: context.Rdx,
            rsi: context.Rsi,
            rdi: context.Rdi,
            r8: context.R8,
            r9: context.R9,
            r10: context.R10,
            r11: context.R11,
            r12: context.R12,
            r13: context.R13,
            r14: context.R14,
            r15: context.R15,
            bytes,
            bytes_read,
        }));

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
            let idx = fault_addr.wrapping_sub(ctx.region_base) / 8;
            let mut return_bytes = [0u8; 8];
            if mem.read(context.Rsp, &mut return_bytes) {
                LAST_HLE_RETURN.store(u64::from_le_bytes(return_bytes), Ordering::Relaxed);
            }
            LAST_HLE_INDEX.store(idx, Ordering::Relaxed);
            HLE_ENTERS.fetch_add(1, Ordering::Relaxed);
            let trace_index = *TRACE_HLE_INDEX.get_or_init(|| {
                std::env::var("XPS5X_TRACE_HLE_INDEX")
                    .ok()
                    .and_then(|value| value.parse().ok())
            });
            if *TRACE_HLE.get_or_init(|| std::env::var_os("XPS5X_TRACE_HLE").is_some())
                || trace_index == Some(idx)
            {
                let read_u64 = |addr| {
                    let mut bytes = [0u8; 8];
                    mem.read(addr, &mut bytes)
                        .then(|| u64::from_le_bytes(bytes))
                        .unwrap_or(0)
                };
                tracing::info!(
                    "HLE enter #{idx}: {}::{} rip={:#x} return={:#x} \
                     args=[{:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}] \
                     pointees=[{:#x}, {:#x}] r14={:#x} *r14={:#x} frame_return={:#x}",
                    t.library,
                    t.function,
                    context.Rip,
                    LAST_HLE_RETURN.load(Ordering::Relaxed),
                    context.Rdi,
                    context.Rsi,
                    context.Rdx,
                    context.Rcx,
                    context.R8,
                    context.R9,
                    read_u64(context.Rdi),
                    read_u64(context.Rsi),
                    context.R14,
                    read_u64(context.R14),
                    read_u64(context.Rbp.wrapping_add(8)),
                );
                if trace_index == Some(idx) {
                    let mut frame = context.Rbp;
                    let mut chain = Vec::new();
                    for _ in 0..12 {
                        let previous = read_u64(frame);
                        let return_addr = read_u64(frame.wrapping_add(8));
                        if return_addr == 0 || previous <= frame {
                            break;
                        }
                        let location = kernel
                            .unwind_module_for_addr(return_addr)
                            .map(|module| {
                                format!("{}+{:#x}", module.name, return_addr - module.start)
                            })
                            .unwrap_or_else(|| format!("{return_addr:#x}"));
                        chain.push(location);
                        frame = previous;
                    }
                    if !chain.is_empty() {
                        tracing::info!("HLE #{idx} frame chain: {}", chain.join(" -> "));
                    }
                    let return_addr = LAST_HLE_RETURN.load(Ordering::Relaxed);
                    let caller_start = return_addr & !0xff;
                    if let Some(disassembly) = diagnostic_disassembly(mem, caller_start, 0x100) {
                        tracing::info!(
                            "HLE #{idx} caller disassembly from {caller_start:#x}:\n{disassembly}"
                        );
                    }
                    let frame_return = read_u64(context.Rbp.wrapping_add(8));
                    let frame_caller_start = frame_return & !0xff;
                    if frame_return != 0
                        && let Some(disassembly) =
                            diagnostic_disassembly(mem, frame_caller_start, 0x100)
                    {
                        tracing::info!(
                            "HLE #{idx} frame caller disassembly from {frame_caller_start:#x}:\n\
                             {disassembly}"
                        );
                    }
                    let callback = context.Rsi;
                    if t.function.contains("Once")
                        && let Some(disassembly) = diagnostic_disassembly(mem, callback, 0x100)
                    {
                        tracing::info!(
                            "HLE #{idx} callback disassembly from {callback:#x}:\n{disassembly}"
                        );
                    }
                }
            }
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
                ctx.request_process_exit(context.Rdi);

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
            // `[Rsp]` at the trap is the `call`-pushed return address — the
            // guest instruction that called this HLE function. Threaded into
            // the context purely as diagnostic provenance.
            let caller_return_addr = {
                let mut buf = [0u8; 8];
                if mem.read(context.Rsp, &mut buf) {
                    u64::from_le_bytes(buf)
                } else {
                    0
                }
            };
            let hle_ctx = HleContext {
                kernel,
                mem,
                alloc,
                guest_calls: ctx,
                guest_threads: ctx,
                caller_return_addr,
                caller_rsp: context.Rsp,
            };
            ctx.active_hle
                .set(Some((idx, args[..6].try_into().unwrap())));
            let ret = hle
                .call(&hle_ctx, &t.library, &t.function, &args)
                .unwrap_or(0);
            ctx.active_hle.set(None);
            HLE_EXITS.fetch_add(1, Ordering::Relaxed);
            if ctx.process_is_terminating() {
                ctx.exited.set(true);
                *context = unsafe { *ctx.recovery_ctx };
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            if let Some(retval) = ctx.thread_exit.take() {
                ctx.retval.set(retval);
                ctx.returned.set(true);
                // SAFETY: the recovery context is armed and remains live for
                // this run; pthread exit abandons the guest stack exactly like
                // a normal top-level return.
                *context = unsafe { *ctx.recovery_ctx };
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            // Remember it: if the guest later faults on a null we handed back,
            // this history is the only thing that names the culprit (see
            // `CallTrace`). Index, not name — no allocation in the VEH.
            ctx.trace.push(idx as u32, ret, [args[0], args[1], args[2]]);
            ret
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

    if let Some(request) = ctx.pending_guest_call.take() {
        let mut original_return = [0u8; 8];
        let mut entry_probe = [0u8; 1];
        if request.entry >= 0x1_0000
            && mem.read(request.entry, &mut entry_probe)
            && mem.read(context.Rsp, &mut original_return)
            && mem.write(context.Rsp, &ctx.callback_return_addr.to_le_bytes())
        {
            ctx.callback_frames.borrow_mut().push(GuestCallbackFrame {
                original_return: u64::from_le_bytes(original_return),
                hle_result: result,
                completion: request.completion,
            });
            let callback_rsp = context.Rsp;
            context.Rdi = request.args[0];
            context.Rsi = request.args[1];
            context.Rdx = request.args[2];
            context.Rcx = request.args[3];
            context.R8 = request.args[4];
            context.R9 = request.args[5];
            context.Rax = 0;
            resume_guest_with_tls(ctx, context, mem, request.entry, callback_rsp);
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        if let Some(completion) = request.completion {
            let _ = mem.atomic_store_u32(completion.address, completion.failure_u32);
        }
    }

    context.Rax = result;

    // Emulate the `call` instruction's target returning. The CPU already
    // pushed the return address before faulting on the instruction fetch at
    // the (guarded) trampoline address — `call`'s push-then-jump always
    // precedes the fetch of the first instruction at the new Rip, so by the
    // time we're here [Rsp] holds that return address.
    //
    // RSP is guest-controlled. Read it through the arena bounds check instead
    // of dereferencing it in the VEH: a corrupt stack pointer must become a
    // recoverable guest fault, never a recursive host exception.
    let mut return_bytes = [0u8; 8];
    if !mem.read(context.Rsp, &mut return_bytes) {
        ctx.error.set(Some(RuntimeError::Faulted {
            addr: fault_addr,
            access: context.Rsp,
            kind: crate::FaultKind::Read,
        }));
        *context = unsafe { *ctx.recovery_ctx };
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    let ret_addr = u64::from_le_bytes(return_bytes);
    let return_rsp = context.Rsp.wrapping_add(8);
    resume_guest_with_tls(ctx, context, mem, ret_addr, return_rsp);

    EXCEPTION_CONTINUE_EXECUTION
}
