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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter, Register};
use windows_sys::Win32::Foundation::{
    CloseHandle, EXCEPTION_ACCESS_VIOLATION, EXCEPTION_BREAKPOINT, EXCEPTION_ILLEGAL_INSTRUCTION,
    EXCEPTION_INT_DIVIDE_BY_ZERO, EXCEPTION_INT_OVERFLOW,
};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, CONTEXT_ALL_AMD64, EXCEPTION_CONTINUE_EXECUTION,
    EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS, GetThreadContext, RtlCaptureContext,
    RtlRestoreContext,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentThreadId, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT,
    THREAD_SUSPEND_RESUME,
};

use raeen_core::blockers::{self, BlockerCategory};
use raeen_core::frame_path::{self, Phase};
use raeen_core::subsystems::GpuSubmissionSubsystem;
use raeen_firmware::{HleTrampoline, UnresolvedStub};
use raeen_hle::{
    GuestAllocator, GuestCallCompletion, GuestCallError, GuestCallRequest, GuestCallScheduler,
    GuestMemory, GuestThreadScheduler, HleContext, HleRegistry,
};
use raeen_kernel::OrbisKernel;

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
    // The `libSceLibcInternal` ABI alias of the same libc functions: provider-
    // free resolution deterministically picks the lexicographically first
    // registration for a shared name, which is this library — an import of
    // `exit` resolves here, and must terminate the run just the same.
    ("libSceLibcInternal", "exit"),
    ("libSceLibcInternal", "exit_group"),
    ("libSceLibcInternal", "_exit"),
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

/// Serializes top-level [`crate::execute_linked`] sessions end to end.
/// [`call_lock`] is acquired by `execute_linked` itself (not just around the
/// initial guest call in [`run`]), because `TrampolineGuard::reserve`'s
/// fixed-address `VirtualAlloc` (in `trampoline.rs`) is process-global state:
/// two concurrent process launches racing to reserve the same fixed
/// `HLE_TRAMPOLINE_BASE` region would spuriously fail with `MapFailed`.
///
/// This does **not** serialize guest pthread workers belonging to that process.
/// Those workers may execute [`run`] concurrently and rely on thread-local
/// [`ACTIVE_CONTEXT`] slots; the VEH only reads the faulting OS thread's slot.
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
    let delay_ms = std::env::var("RAEEN_GUEST_WATCHDOG_MS")
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
static HLE_VEH_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static HLE_DIRECT_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static LAST_HLE_INDEX: AtomicU64 = AtomicU64::new(u64::MAX);
static LAST_HLE_RETURN: AtomicU64 = AtomicU64::new(0);
static GUEST_WATCHPOINT_HITS: AtomicU64 = AtomicU64::new(0);
static TRACE_HLE: OnceLock<bool> = OnceLock::new();
static TRACE_HLE_INDEX: OnceLock<Option<u64>> = OnceLock::new();
static TRACE_HLE_FUNCTION: OnceLock<Option<String>> = OnceLock::new();
static TRACE_EINVAL: OnceLock<bool> = OnceLock::new();
/// `RAEEN_TRAP_CXA_THROW`, read once.
///
/// It sits as the last operand of an `||` chain whose earlier operands are
/// false on a default run, so `||` never short-circuits it away: without this
/// cache every HLE call — ~15k/second on the measured main thread — paid a
/// Windows environment-block scan plus the `OsString` allocation `var_os`
/// returns. Same treatment [`TRACE_EINVAL`] already has one line above.
static TRAP_CXA: OnceLock<bool> = OnceLock::new();
static WATCH_REQUEST: OnceLock<Option<GuestWatchRequest>> = OnceLock::new();
static WATCH_CANARY_FRAME_DEPTH: OnceLock<usize> = OnceLock::new();

const EXCEPTION_SINGLE_STEP_CODE: i32 = 0x8000_0004_u32 as i32;
const CONTEXT_DEBUG_REGISTERS_AMD64_FLAG: u32 = 0x0010_0010;
const STACK_CANARY_SLOT_FROM_RBP: u64 = 0x30;

#[derive(Debug)]
enum GuestWatchRequest {
    /// Arm an eight-byte write watch at the exact guest address supplied by
    /// `RAEEN_WATCH_ADDR`.
    Explicit(u64),
    /// Read one guest pointer from `RAEEN_WATCH_PTR_ADDR` and watch the
    /// eight-byte value it names. This is useful for imported data symbols:
    /// the module-relative GOT address is stable while the HLE data-page
    /// address stored there is process-local.
    Indirect(u64),
    /// When this HLE function returns, watch a guest frame's stack-protector
    /// slot at `[rbp - 0x30]`. `RAEEN_WATCH_CANARY_FRAME_DEPTH` selects how
    /// many saved-frame-pointer links to follow (default 0 = the frame that
    /// called HLE).
    ///
    /// This is the useful automatic mode for a deterministic canary smash:
    /// set `RAEEN_WATCH_CANARY_AFTER_HLE=sceKernelGetdents`, and the next
    /// guest instruction that changes the protected slot is reported.
    CallerCanaryAfter(String),
}

#[derive(Debug, Clone, Copy)]
struct ActiveGuestWatch {
    address: u64,
    initial_value: u64,
    armed_by_hle_index: u64,
}

fn parse_watch_address(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    let digits = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    if digits.is_empty() {
        return None;
    }
    u64::from_str_radix(digits, 16).ok()
}

fn parse_watch_frame_depth(value: &str) -> Option<usize> {
    value.trim().parse().ok().filter(|depth| *depth <= 64)
}

fn watch_canary_frame_depth() -> usize {
    *WATCH_CANARY_FRAME_DEPTH.get_or_init(|| {
        let Ok(value) = std::env::var("RAEEN_WATCH_CANARY_FRAME_DEPTH") else {
            return 0;
        };
        match parse_watch_frame_depth(&value) {
            Some(depth) => depth,
            None => {
                tracing::warn!(
                    "ignoring invalid RAEEN_WATCH_CANARY_FRAME_DEPTH={value:?}; \
                     expected an integer from 0 through 64"
                );
                0
            }
        }
    })
}

fn stack_canary_address(
    mem: &dyn GuestMemory,
    mut frame_pointer: u64,
    frame_depth: usize,
) -> Option<u64> {
    for _ in 0..frame_depth {
        frame_pointer = read_guest_u64(mem, frame_pointer)?;
    }
    frame_pointer.checked_sub(STACK_CANARY_SLOT_FROM_RBP)
}

/// Encode one local x64 data breakpoint in DR7.
///
/// L0 is enabled, RW0=01 selects writes, and LEN0 uses Intel's non-linear
/// length encoding (00=1, 01=2, 11=4, 10=8). Execute and I/O breakpoints are
/// deliberately unsupported here.
fn write_watch_dr7(byte_len: usize) -> Option<u64> {
    let len_bits = match byte_len {
        1 => 0b00,
        2 => 0b01,
        4 => 0b11,
        8 => 0b10,
        _ => return None,
    };
    // Bit 10 is architecturally fixed to one. Windows normalizes it when an
    // exception CONTEXT is resumed, so include it in the value we compare and
    // restore instead of reporting every HLE boundary as lost debug state.
    Some((1 << 10) | 1 | (0b01 << 16) | (len_bits << 18))
}

fn guest_watch_request() -> Option<&'static GuestWatchRequest> {
    WATCH_REQUEST
        .get_or_init(|| {
            if let Ok(value) = std::env::var("RAEEN_WATCH_ADDR") {
                match parse_watch_address(&value) {
                    Some(address) => return Some(GuestWatchRequest::Explicit(address)),
                    None => tracing::warn!(
                        "ignoring invalid RAEEN_WATCH_ADDR={value:?}; expected hexadecimal address"
                    ),
                }
            }
            if let Ok(value) = std::env::var("RAEEN_WATCH_PTR_ADDR") {
                match parse_watch_address(&value) {
                    Some(address) => return Some(GuestWatchRequest::Indirect(address)),
                    None => tracing::warn!(
                        "ignoring invalid RAEEN_WATCH_PTR_ADDR={value:?}; \
                         expected hexadecimal address"
                    ),
                }
            }
            std::env::var("RAEEN_WATCH_CANARY_AFTER_HLE")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(GuestWatchRequest::CallerCanaryAfter)
        })
        .as_ref()
}

/// Whether the per-call stall instruments — [`OrbisKernel::in_flight_hle`],
/// [`OrbisKernel::hle_call_time`] and [`OrbisKernel::recent_hle_calls`] — must be
/// populated.
///
/// `RAEEN_STALL_DUMP` **implies** them. It used to imply nothing, and that cost a
/// full hardware investigation: the Blasphemous II runs set `RAEEN_STALL_DUMP`
/// alone, so `in_flight_hle` stayed empty and every dump printed
/// `IN-FLIGHT HLE: <none — all threads between calls>` while 14 threads sat
/// inside `sceKernelWaitSema`. The one run that also set `RAEEN_TIME_HLE` named
/// them correctly — the *stall* never changed, only the arming did, and the
/// difference was read as the stall having changed class.
///
/// A diagnostic knob that silently reports "nothing" when it was never armed is
/// worse than one that is off, so the reader arms everything its own report
/// depends on.
fn stall_instruments_armed() -> bool {
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| {
        std::env::var_os("RAEEN_TIME_HLE").is_some()
            || std::env::var_os("RAEEN_STALL_DUMP").is_some()
    })
}

/// `RAEEN_CALL_STATS`: count every HLE call per `library::function`, split
/// into a boot window (first 30 s) and steady state after. A title that
/// POLLS a readiness value in short cycles never shows up in the in-flight
/// or timing views (each call is fast) — but its poll function dominates the
/// steady-state call ranking. Relaxed atomics on [`OrbisKernel::hle_call_counts`].
static CALL_STATS: OnceLock<bool> = OnceLock::new();

/// Boundary between the two [`CALL_STATS`] windows.
const CALL_STATS_BOOT_WINDOW: std::time::Duration = std::time::Duration::from_secs(30);

/// The number of FS-base re-arms performed since process start — see
/// [`FSBASE_REARMS`]. Monotonic; never reset.
pub fn fsbase_rearm_count() -> u64 {
    FSBASE_REARMS.load(Ordering::Relaxed)
}

/// Number of guest data-write watchpoints consumed by the VEH.
///
/// This diagnostic counter is monotonic and process-wide. It exists so a
/// controlled native-guest fixture can prove the complete DR0/DR7 ->
/// `EXCEPTION_SINGLE_STEP` -> DR6.B0 path, rather than treating successful
/// register encoding as evidence that Windows delivered the trap.
pub fn guest_watchpoint_hit_count() -> u64 {
    GUEST_WATCHPOINT_HITS.load(Ordering::Relaxed)
}

/// Monotonic HLE dispatch counters used by compatibility reports and the
/// direct-vs-VEH benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HleDispatchMetrics {
    pub entered: u64,
    pub exited: u64,
    pub veh: u64,
    pub direct: u64,
}

pub fn hle_dispatch_metrics() -> HleDispatchMetrics {
    HleDispatchMetrics {
        entered: HLE_ENTERS.load(Ordering::Relaxed),
        exited: HLE_EXITS.load(Ordering::Relaxed),
        veh: HLE_VEH_DISPATCHES.load(Ordering::Relaxed),
        direct: HLE_DIRECT_DISPATCHES.load(Ordering::Relaxed),
    }
}

/// Acquire [`CALL_LOCK`] for the entire [`crate::execute_linked`] pipeline.
/// Called once, at the very top of `execute_linked`; the returned guard is
/// held (as a local binding) until that function returns. `run` does not
/// acquire `CALL_LOCK` itself: process-owned pthread workers also call `run`
/// and are intentionally concurrent within the already-active session.
pub(crate) fn call_lock() -> std::sync::MutexGuard<'static, ()> {
    CALL_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Diagnostic "guest GIL" behind the `RAEEN_SINGLE_THREAD_GUEST` knob.
///
/// When enabled, only one guest thread runs guest *native* code at a time. The
/// token is released whenever a thread traps out into an HLE call — so a
/// blocking `sceKernelWaitSema`/mutex/cond wait still lets another guest thread
/// run (a strict "one thread, ever" scheme deadlocks any persistent worker
/// pool) — and re-acquired before the guest resumes. A guest that spins in
/// native code on a shared flag without ever calling an HLE function can still
/// deadlock here; the opt-in `RAEEN_GUEST_WATCHDOG_MS` surfaces that.
///
/// Purpose: a bring-up A/B. If a crash that reproduces with parallel guest
/// threads stops reproducing under this flag, the bug is a guest data race and
/// the fix belongs in a guest-visible synchronization primitive that is not
/// actually excluding — not in the faulting guest code. Default off: when the
/// env var is unset every hook below is a `None` guard and changes nothing.
static GUEST_GIL_HELD: Mutex<bool> = Mutex::new(false);
static GUEST_GIL_IDLE: std::sync::Condvar = std::sync::Condvar::new();

fn single_thread_guest() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RAEEN_SINGLE_THREAD_GUEST").is_some())
}

fn guest_gil_acquire() {
    let mut held = GUEST_GIL_HELD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    while *held {
        held = GUEST_GIL_IDLE
            .wait(held)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }
    *held = true;
}

fn guest_gil_release() {
    let mut held = GUEST_GIL_HELD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *held = false;
    GUEST_GIL_IDLE.notify_one();
}

/// Holds the guest GIL for one guest-native execution span (the whole of
/// [`run`]). Drops on every return arm — including the recovery jump, which
/// restores `run`'s own frame — so a fault or exit releases it too. `None`
/// (and a no-op) unless [`single_thread_guest`].
struct GuestGilHold;
impl GuestGilHold {
    fn acquire() -> Option<Self> {
        single_thread_guest().then(|| {
            guest_gil_acquire();
            GuestGilHold
        })
    }
}
impl Drop for GuestGilHold {
    fn drop(&mut self) {
        guest_gil_release();
    }
}

/// Temporarily yields the guest GIL for the duration of an HLE call, re-
/// acquiring when dropped — i.e. before the guest resumes. Scoped to the HLE
/// trampoline arm so it drops on every exit from it (normal, terminating,
/// pthread-exit). `None` (and a no-op) unless [`single_thread_guest`].
struct GuestGilYield;
impl GuestGilYield {
    fn during_hle() -> Option<Self> {
        single_thread_guest().then(|| {
            guest_gil_release();
            GuestGilYield
        })
    }
}
impl Drop for GuestGilYield {
    fn drop(&mut self) {
        guest_gil_acquire();
    }
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

/// How many of the most recent calls a fault publishes to
/// `OrbisKernel::recent_hle_calls` for the crash report.
///
/// The report is a markdown file a user pastes into a GitHub issue, so this is
/// a readability budget, not a diagnostic one — the full 4096-entry ring still
/// goes to the log at DEBUG. Matches the cap the `RAEEN_TRACE_EINVAL`
/// diagnostic writer uses for the same deque.
const CRASH_REPORT_RING_LEN: usize = 24;

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
/// call returns. The slot containing this value is thread-local, and the VEH
/// consults it synchronously on that same OS thread, so the fields need no
/// cross-thread synchronization.
struct ActiveContext {
    trampolines: *const [HleTrampoline],
    /// The per-NID unresolved-stub table, so a fault inside
    /// `[UNRESOLVED_STUB_BASE, +len*8)` can name the import the guest wanted
    /// instead of reporting a bare address. The process-owned backing data
    /// outlives every worker `run` call and this thread-local context is cleared
    /// before that call returns.
    unresolved_stubs: *const [UnresolvedStub],
    hle: *const HleRegistry,
    /// The live kernel an HLE call's [`HleContext::kernel`] borrows from.
    /// Stored as a raw pointer because the process-owned kernel outlives each
    /// worker `run` call, and the VEH only accesses it synchronously on the
    /// same OS thread before that thread's context slot is cleared.
    kernel: *const OrbisKernel,
    /// The guest memory view an HLE call's [`HleContext::mem`] borrows
    /// from — a trait object pointer (fat pointer) for the same reason.
    mem: *const dyn GuestMemory,
    /// The guest allocator an HLE call's [`HleContext::alloc`] borrows
    /// from — a trait object pointer (fat pointer), same reasoning as `mem`.
    alloc: *const dyn GuestAllocator,
    gpu: *const dyn GpuSubmissionSubsystem,
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
    /// Red-zone-safe rearm stub (`ret 0x80` twin of the plain WRFSBASE stub):
    /// staging lands at `[rsp-144, rsp-128)`, below the SysV red zone, so
    /// resuming an ARBITRARY interrupted instruction cannot overwrite a leaf
    /// function's live red-zone locals.
    tls_rearm_trampoline_redzone: u64,
    /// Top-level guest calls return through the same guarded address when no
    /// nested HLE callback frame is active.
    returned: Cell<bool>,
    retval: Cell<u64>,
    /// At most one guest callback can be requested by a single HLE handler.
    pending_guest_call: Cell<Option<GuestCallRequest>>,
    /// Whether the HLE handler currently executing on this thread was
    /// dispatched through [`direct_hle_gateway`] (the executable leaf bridge)
    /// rather than the VEH. Set/restored around the gateway's `hle.call` so
    /// [`ActiveContext::call_guest`] can refuse synchronous guest re-entry on
    /// that path: the generated direct bridge re-bases RSP to a fixed private
    /// host-stack top on *every* entry (`trampoline::direct_bridge_code`), so
    /// a callback running on that stack that called another direct-dispatched
    /// import would have its live frames clobbered. Save/restore (not a bare
    /// set/clear) because a VEH-path callback may legitimately call a direct
    /// leaf import, nesting a gateway dispatch inside a VEH dispatch.
    direct_gateway_active: Cell<bool>,
    /// HLE currently executing on this OS thread. A GuestMemory write can
    /// itself fault (for example, when a bad output pointer names RX code),
    /// recursively entering the VEH before the call can be added to the
    /// completed-call trace. Keep attribution in the per-thread active
    /// context so the recovered fault names the actual writer.
    active_hle: Cell<Option<(u64, [u64; 6])>>,
    /// Nested guest callbacks are legal (an initializer may call another API
    /// that invokes guest code), so retain their original HLE return frames.
    ///
    /// `Cell`, not `RefCell`, and for the same reason every other field here is
    /// (see the discipline note on [`CallTrace`]) — but this one was measured
    /// the hard way. Unwinding these frames writes each frame's completion word
    /// to a **guest-controlled** address; that write can itself fault, which
    /// re-enters the VEH *on this thread* and reaches the same unwind code. With
    /// a `RefCell` the outer `borrow_mut()` was still live, so the inner borrow
    /// panicked — a Rust panic inside a vectored exception handler, i.e. an
    /// `abort` that Windows reports as `0xC0000409`
    /// (`STATUS_STACK_BUFFER_OVERRUN`) with no fault report at all. A `Cell`
    /// cannot fail: [`Cell::take`] moves the whole vector out before any guest
    /// memory is touched, so a re-entrant handler simply finds it empty.
    ///
    /// Every access must therefore follow the same shape: `take()` (or
    /// take/modify/`set()`) with **no guest access under the temporary**, then
    /// touch guest memory. Taking leaves `Vec::new()`, which neither allocates
    /// nor frees, so this stays VEH-safe.
    callback_frames: Cell<Vec<GuestCallbackFrame>>,
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

const DIRECT_HOST_STACK_SIZE: usize = 256 * 1024;

/// The direct HLE gateway's per-run host stack: a dedicated VirtualAlloc'd
/// region with a `PAGE_NOACCESS` guard page at its LOW end. The bridge
/// re-bases RSP to `top` on every direct-dispatched import, and the deepest
/// frame chains (GPU submit, the mutex-contention stack walk) grow downward —
/// with the old heap `Box<[u8]>` an overflow wrote silently into whatever
/// allocation the allocator had placed below it. Overrunning this region
/// faults AT THE STORE on the guard page, which the VEH catches and
/// classifies like any other host fault.
struct DirectHostStack {
    /// Allocation base == the guard page. Usable stack is `[base+4096, top)`.
    base: u64,
    top: u64,
}

impl DirectHostStack {
    const GUARD: usize = 4096;

    fn new() -> Result<Self, RuntimeError> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualProtect,
        };
        let total = DIRECT_HOST_STACK_SIZE + Self::GUARD;
        // SAFETY: fresh anonymous reserve+commit, released exactly once in
        // `Drop`. No address is published until the call succeeds.
        let raw = unsafe {
            VirtualAlloc(
                core::ptr::null(),
                total,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if raw.is_null() {
            return Err(RuntimeError::MapFailed);
        }
        let mut old = 0u32;
        // SAFETY: the first page of the region committed just above.
        let armed = unsafe { VirtualProtect(raw, Self::GUARD, PAGE_NOACCESS, &mut old) } != 0;
        if !armed {
            // SAFETY: `raw` is the live allocation base from the call above.
            unsafe { VirtualFree(raw, 0, MEM_RELEASE) };
            return Err(RuntimeError::MapFailed);
        }
        let base = raw as u64;
        Ok(Self {
            base,
            top: base + total as u64,
        })
    }
}

impl Drop for DirectHostStack {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        // SAFETY: `base` is the exact allocation base `new` obtained, and the
        // bridge only uses the stack inside guarded guest calls that end
        // before this run frame unwinds.
        unsafe { VirtualFree(self.base as *mut core::ffi::c_void, 0, MEM_RELEASE) };
    }
}

#[repr(C)]
struct DirectThreadState {
    context: *mut ActiveContext,
    host_stack_top: u64,
    previous_teb_slot: u64,
}

/// Read the x64 TEB's `NT_TIB.ArbitraryUserPointer` (`GS:[0x28]`).
///
/// Windows owns and preserves GS as the TEB base across context switches,
/// while Raeen deliberately replaces FS with the guest TCB. The arbitrary
/// pointer field is the application-owned per-thread slot used by the direct
/// HLE bridge so a preemption cannot turn its state pointer into unrelated TEB
/// bytes. The previous value is restored when the guarded guest call ends.
fn direct_state_slot() -> u64 {
    let value: u64;
    // SAFETY: on Windows x86-64 GS addresses the current thread's valid TEB;
    // offset 0x28 is the pointer-sized ArbitraryUserPointer field. This is a
    // read only and neither changes the segment base nor dereferences `value`.
    unsafe {
        core::arch::asm!(
            "mov {value}, qword ptr gs:[0x28]",
            value = out(reg) value,
            options(nostack, preserves_flags, readonly)
        );
    }
    value
}

fn set_direct_state_slot(value: u64) {
    // SAFETY: same TEB layout invariant as `direct_state_slot`. The field is
    // explicitly application-owned, the value points at a live boxed
    // DirectThreadState, and `run` restores the prior value before returning.
    unsafe {
        core::arch::asm!(
            "mov qword ptr gs:[0x28], {value}",
            value = in(reg) value,
            options(nostack, preserves_flags)
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct GuestCallbackFrame {
    original_return: u64,
    hle_result: u64,
    /// Whether the HLE call that scheduled this callback was registered
    /// float-returning — the result lands in guest XMM0 as well as RAX
    /// when the frame completes (see [`apply_hle_result`]).
    float_return: bool,
    completion: Option<GuestCallCompletion>,
}

impl ActiveContext {
    /// Push one nested guest-callback frame.
    ///
    /// Take/modify/set rather than a borrow: see [`ActiveContext::callback_frames`].
    /// Nothing between the take and the set may touch guest memory — a fault
    /// there would re-enter the VEH and find the stack momentarily empty, losing
    /// the outer frames' completion rollback.
    fn push_callback_frame(&self, frame: GuestCallbackFrame) {
        let mut frames = self.callback_frames.take();
        frames.push(frame);
        self.callback_frames.set(frames);
    }

    /// Pop the innermost nested guest-callback frame, if any.
    fn pop_callback_frame(&self) -> Option<GuestCallbackFrame> {
        let mut frames = self.callback_frames.take();
        let frame = frames.pop();
        self.callback_frames.set(frames);
        frame
    }

    /// Take **every** nested callback frame, innermost first — the order in
    /// which an abandoned guest stack must roll its completions back.
    ///
    /// The whole vector leaves the cell before the caller writes a single
    /// completion word, which is the entire point (see
    /// [`ActiveContext::callback_frames`]): those writes go to guest-controlled
    /// addresses and may fault, re-entering this handler, and a re-entrant
    /// unwind must find nothing left to unwind rather than a live borrow.
    fn take_callback_frames_innermost_first(&self) -> std::vec::IntoIter<GuestCallbackFrame> {
        let mut frames = self.callback_frames.take();
        frames.reverse();
        frames.into_iter()
    }
}

impl GuestCallScheduler for ActiveContext {
    fn request(&self, request: GuestCallRequest) -> bool {
        if self.pending_guest_call.get().is_some() {
            return false;
        }
        self.pending_guest_call.set(Some(request));
        true
    }

    /// Synchronous guest re-entry from inside an HLE handler (checklist item
    /// 7): call `entry` with up to six SysV integer arguments **on the
    /// current guest thread**, returning its RAX to the handler mid-call.
    ///
    /// # Stack strategy
    ///
    /// On the VEH dispatch path the handler is *already executing on the
    /// guest stack*: Windows delivers vectored exceptions synchronously on
    /// the faulting thread's current stack, and the guest was on its own
    /// (arena-backed) stack when it called the import slot. A plain native
    /// call from here therefore runs the callback on the guest stack, below
    /// the trapped frame and the kernel's exception-dispatch frames, with
    /// the compiler maintaining the SysV 16-byte call alignment. That is the
    /// architecturally sound choice over a dedicated callback stack region
    /// for two reasons: (a) the TLS re-arm and FS-rearm paths stage a tiny
    /// return trampoline at `[rsp-16]` via `GuestMemory::write`, which is
    /// bounds-checked against the arena — a callback on a host-allocated
    /// side stack would make every re-arm under it fail; and (b) the guest
    /// stack is exactly where the callback would have run on real hardware
    /// (below its caller's frame), so guest stack-walking code sees a
    /// plausible chain.
    ///
    /// # Nesting
    ///
    /// The callback may itself call HLE imports; each traps into a nested
    /// VEH dispatch against this same `ActiveContext` (recovery context,
    /// TLS/FS re-arm, on-demand commit, and terminating-function arms all
    /// stay armed), and such a nested handler may call `call_guest` again.
    /// Depth is bounded only by guest stack space; depth 2 (HLE → callback →
    /// HLE → callback) is pinned by `nested_call_guest_composes_to_depth_two`
    /// in `tests/execute.rs`.
    ///
    /// # Unwind composition
    ///
    /// A genuine fault in the callback, a `request_exit` from a nested HLE
    /// handler (`__stack_chk_fail`, `abort`, `scePthreadExit`), a
    /// terminating-function call (`exit`), or cooperative process
    /// termination all longjmp to `run`'s recovery context exactly as they
    /// do outside a callback. Control then **never returns here**, so the
    /// interrupted HLE handler cannot mistake a fatal unwind for a
    /// successful callback return. The longjmp abandons this Rust frame
    /// without running its drops — the same established trade every HLE
    /// handler already makes for faults raised under `GuestMemory` accesses;
    /// nothing in this frame owns a resource beyond the saved cells below,
    /// and those die with the run's `ActiveContext`.
    ///
    /// # Dispatch paths
    ///
    /// VEH only. The direct leaf gateway is refused loudly (see
    /// `direct_gateway_active`): its generated bridge re-bases RSP to a
    /// fixed private host-stack top on every entry, so a nested
    /// direct-dispatched import called from a callback would clobber the
    /// outer gateway's live frames, and re-arm staging writes would target
    /// non-arena host memory. `trampoline::direct_dispatchable` deliberately
    /// lists only imports that never re-enter guest code — an import that
    /// needs callbacks belongs on the VEH path.
    fn call_guest(&self, entry: u64, args: [u64; 6]) -> Result<u64, GuestCallError> {
        if self.direct_gateway_active.get() {
            tracing::error!(
                "synchronous guest callback to {entry:#x} requested from the direct leaf \
                 gateway — refused: the direct bridge re-bases RSP to a fixed host stack \
                 top on every entry, so re-entering guest code here is unsound. Route the \
                 requesting import through the VEH path (remove it from \
                 `direct_dispatchable`)."
            );
            return Err(GuestCallError::Unsupported);
        }
        if entry == 0 {
            return Err(GuestCallError::NullEntry);
        }
        debug_assert!(
            self.armed.get(),
            "call_guest is only reachable from an HLE handler dispatched under an armed run"
        );

        // Preserve the interrupted call's per-thread dispatch state: a nested
        // HLE call made by the callback overwrites `active_hle` (fault
        // attribution would then name the wrong handler) and would consume a
        // deferred `pending_guest_call` the *outer* handler already scheduled
        // (attaching it to the nested call's return instead of the outer
        // one's). Both are restored after the callback returns.
        let saved_active_hle = self.active_hle.take();
        let saved_pending_call = self.pending_guest_call.take();

        // Diagnostic single-thread mode: guest native code is about to run on
        // this thread, so hold the guest GIL for the callback's duration (the
        // dispatching HLE arm released it via `GuestGilYield`). No-op unless
        // RAEEN_SINGLE_THREAD_GUEST is set.
        let gil = GuestGilHold::acquire();

        // SAFETY: `entry` names guest-supplied code in the identity-mapped
        // arena (or an executable import slot), exactly the class of address
        // `stack::enter_guest` jumps to — and the same recovery machinery
        // defines every outcome: this thread is inside an HLE handler
        // dispatched by `run`, so the process VEH is installed, this
        // thread's `ACTIVE_CONTEXT` names `self`, and `self.armed` is true
        // with a populated `recovery_ctx`. If `entry` is not valid code the
        // first instruction fetch faults outside the guard region and the
        // genuine-fault arm longjmps to `run` (reported as `Err(Faulted)` —
        // this call never returns); HLE imports the callback makes trap and
        // are serviced by nested VEH dispatch; FS-base staleness re-arms on
        // fault. The callback is assumed to follow the SysV ABI (callee-
        // saved registers, DF=0) — the same assumption the runtime makes of
        // all guest code it enters. The transmute itself only reinterprets
        // an address as a `sysv64` function pointer; no lifetime or validity
        // beyond "callable now, on this thread" is claimed.
        let callback = unsafe {
            core::mem::transmute::<
                usize,
                unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64,
            >(entry as usize)
        };
        // SAFETY: discharged by the block above — armed recovery + VEH make
        // this call's fault/exit/unwind behavior fully defined, and the
        // current stack is the guest stack (VEH dispatch runs on the
        // faulting thread's stack), so callback frames land below the
        // trapped guest frame just as a native call there would.
        let ret = unsafe { callback(args[0], args[1], args[2], args[3], args[4], args[5]) };
        drop(gil);

        self.pending_guest_call.set(saved_pending_call);
        self.active_hle.set(saved_active_hle);
        Ok(ret)
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

    fn set_priority(&self, thread: u64, priority: i32) -> bool {
        // SAFETY: same lifetime invariant as `create`.
        unsafe { &*self.thread_scheduler }.set_priority(thread, priority)
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

    // Stage BELOW the SysV red zone (`[rsp-128, rsp)`), not at `[rsp-16, rsp)`:
    // several callers resume an ARBITRARY interrupted instruction (preempted
    // `fs:` access, watchpoint step, CPUID/syscall traps), where a leaf
    // function's live locals may occupy the red zone — including the
    // `fs:0x28` stack canary a function is about to re-check. The red-zone
    // stub's `ret 0x80` discards the 128-byte gap so RSP still lands exactly
    // on `target_rsp`. Call-boundary resumes tolerate the lower staging
    // trivially (everything below RSP is dead there).
    let Some(staged_rsp) = target_rsp.checked_sub(144) else {
        return;
    };
    if mem.write(staged_rsp, &context.R11.to_le_bytes())
        && mem.write(staged_rsp + 8, &target_rip.to_le_bytes())
    {
        context.Rsp = staged_rsp;
        context.R11 = ctx.guest_fsbase.get();
        context.Rip = ctx.tls_rearm_trampoline_redzone;
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

/// Deliver an HLE call's result to the suspended guest context: always in
/// RAX, and — when the function was registered float-returning
/// ([`raeen_hle::HleRegistry::register_float`]) — ALSO in XMM0, the SysV
/// floating-point return register. Without the second write a `double`/
/// `float` result would never reach the caller: SysV does not look in RAX
/// for it. Used by every HLE writeback site on the VEH path; the direct
/// gateway performs the same delivery in its generated float bridge
/// (`trampoline::direct_bridge_code_float`), keeping the two call paths
/// register-identical.
fn apply_hle_result(context: &mut CONTEXT, result: u64, float_return: bool) {
    context.Rax = result;
    if float_return {
        // SAFETY: identical invariant to the float-argument capture further
        // down — `Anonymous` unions the per-register view with `FltSave`
        // (XSAVE_FORMAT) over the same bytes, and the VEH's exception
        // context is CONTEXT_FULL (CONTEXT_ALL_AMD64), so slot 0's Low/High
        // are valid to write; the resume commits the whole record to the
        // thread. High is zeroed so an f32 result's upper lane is
        // deterministic rather than stale argument bits.
        let xmm0 = unsafe { &mut context.Anonymous.FltSave.XmmRegisters[0] };
        xmm0.Low = result;
        xmm0.High = 0;
    }
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
    /// process-wide handler) is a cheap, non-allocating pointer read. At most
    /// one `run` call is active on a given OS thread, while multiple process
    /// workers may run concurrently on different threads. The pointee is only
    /// touched synchronously on its owning thread, so a plain `Cell` (not an
    /// atomic) is correct.
    static ACTIVE_CONTEXT: Cell<*mut ActiveContext> = const { Cell::new(ptr::null_mut()) };

    /// One hardware data breakpoint owned by this guest OS thread. DR0/DR7
    /// are thread-local CPU state, and Windows delivers the resulting
    /// single-step exception synchronously on the same thread, so the
    /// attribution must be thread-local as well.
    static ACTIVE_GUEST_WATCH: Cell<Option<ActiveGuestWatch>> = const { Cell::new(None) };
}

/// Target of the generated executable leaf-import bridge. The bridge switches
/// to a private host stack before entering this function.
#[allow(clippy::too_many_arguments)]
pub(crate) extern "sysv64" fn direct_hle_gateway(
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    index: u64,
    guest_bridge_rsp: u64,
    context: u64,
    _guest_rax: u64,
    xmm0: u64,
    xmm1: u64,
    xmm2: u64,
    xmm3: u64,
    xmm4: u64,
    xmm5: u64,
    xmm6: u64,
    xmm7: u64,
) -> u64 {
    let dispatch = || {
        let ctx = unsafe { &*(context as *const ActiveContext) };
        let trampolines = unsafe { &*ctx.trampolines };
        let Some(t) = trampolines.get(index as usize) else {
            ctx.error
                .set(Some(RuntimeError::UnresolvedTrampoline(index)));
            return 0;
        };
        let hle = unsafe { &*ctx.hle };
        let kernel = unsafe { &*ctx.kernel };
        let mem = unsafe { &*ctx.mem };
        let alloc = unsafe { &*ctx.alloc };

        let mut args = [0u64; 14];
        args[..6].copy_from_slice(&[a0, a1, a2, a3, a4, a5]);
        // The slot call pushed an internal return address. The original guest
        // return is next, followed by SysV argument seven.
        for i in 0..8 {
            let mut bytes = [0u8; 8];
            if mem.read(guest_bridge_rsp + 16 + i as u64 * 8, &mut bytes) {
                args[6 + i] = u64::from_le_bytes(bytes);
            }
        }
        let mut caller = [0u8; 8];
        let caller_return_addr = if mem.read(guest_bridge_rsp + 8, &mut caller) {
            u64::from_le_bytes(caller)
        } else {
            0
        };
        LAST_HLE_RETURN.store(caller_return_addr, Ordering::Relaxed);
        LAST_HLE_INDEX.store(index, Ordering::Relaxed);
        HLE_ENTERS.fetch_add(1, Ordering::Relaxed);
        HLE_DIRECT_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        // The earliest thing this tree can observe that proves guest code is
        // genuinely executing: nothing can see the guest's first *instruction*,
        // but a trampoline call means it ran far enough to reach an import.
        // After the first one this is a relaxed load of a shared-clean cache
        // line and a not-taken branch — no `fetch_add` — which is what makes it
        // safe on a ~15k-call/second path.
        frame_path::record_phase(Phase::FirstHleCall);
        let _hle_yield = GuestGilYield::during_hle();
        let timed = stall_instruments_armed();
        let started = timed.then(std::time::Instant::now);
        // Read `CALL_STATS` into a local *before* the `||`: as a short-circuited
        // second operand it was skipped whenever `timed` was already true, and
        // the `CALL_STATS.get()` below then depended on some earlier VEH dispatch
        // having initialized it. Widening `timed` to include `RAEEN_STALL_DUMP`
        // made that ordering assumption much easier to hit, so remove it.
        let call_stats = *CALL_STATS.get_or_init(|| std::env::var_os("RAEEN_CALL_STATS").is_some());
        let function_key = (timed || call_stats).then(|| format!("{}::{}", t.library, t.function));
        if call_stats {
            let counters = kernel
                .hle_call_counts
                .entry(function_key.as_ref().expect("call-stat key").clone())
                .or_default();
            let (boot, steady) = counters.value();
            if kernel.uptime() < CALL_STATS_BOOT_WINDOW {
                boot.fetch_add(1, Ordering::Relaxed);
            } else {
                steady.fetch_add(1, Ordering::Relaxed);
            }
        }
        if timed {
            kernel.in_flight_hle.insert(
                ctx.current_thread(),
                function_key.as_ref().expect("timing key").clone(),
            );
        }

        let hle_ctx = HleContext {
            kernel,
            services: kernel,
            gpu: unsafe { &*ctx.gpu },
            mem,
            alloc,
            guest_calls: ctx,
            guest_threads: ctx,
            caller_return_addr,
            caller_rsp: guest_bridge_rsp + 8,
            float_args: [xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7],
            // No CONTEXT exists on this path: the direct bridge arrives by a
            // plain `call`, not a trap, and only the argument registers were
            // preserved. `trampoline::direct_dispatchable` lists exactly the
            // imports that never re-enter guest code, so nothing dispatched
            // here can need a machine context (see `HleContext::caller_gprs`).
            caller_gprs: None,
        };
        ctx.active_hle
            .set(Some((index, args[..6].try_into().unwrap())));
        // Keep diagnostic parity with the VEH dispatcher. Without this, every
        // executable-gateway call vanished from RAEEN_TRACE_EINVAL's recent
        // call ring, so a thread parked in a direct mutex/condvar handler
        // appeared as "between calls" in STALL_DUMP even while it was the
        // measured owner of a title-wide streaming convoy.
        if *TRACE_EINVAL.get_or_init(|| std::env::var_os("RAEEN_TRACE_EINVAL").is_some())
            || stall_instruments_armed()
            || *TRAP_CXA.get_or_init(|| std::env::var_os("RAEEN_TRAP_CXA_THROW").is_some())
        {
            let tid = ctx.current_thread();
            let ring = kernel.recent_hle_calls.entry(tid).or_default();
            let mut q = ring.lock();
            if q.len() >= 24 {
                q.pop_front();
            }
            q.push_back(format!("{}::{}", t.library, t.function));
        }
        // Mark this dispatch as gateway-hosted for the duration of the
        // handler, so a `call_guest` attempt from inside it is refused (see
        // `ActiveContext::direct_gateway_active`). Save/restore rather than
        // set/clear: this gateway may itself be nested under a VEH-path
        // callback. The early longjmp exits below never restore the flag,
        // but they abandon this whole ActiveContext with the run.
        let prev_direct_gateway = ctx.direct_gateway_active.replace(true);
        let result = hle
            .call(&hle_ctx, &t.library, &t.function, &args)
            .unwrap_or(0);
        ctx.direct_gateway_active.set(prev_direct_gateway);
        // Two comparisons on a value already in a register; see
        // `record_orbis_error` for why the error path itself is worth counting.
        if is_orbis_error(result) {
            record_orbis_error(index, trampolines.len(), &t.library, &t.function, result);
        }
        if timed {
            kernel.in_flight_hle.remove(&ctx.current_thread());
            let micros = started.map_or(0, |s| s.elapsed().as_micros());
            let mut entry = kernel
                .hle_call_time
                .entry((
                    ctx.current_thread(),
                    function_key.as_ref().expect("timing key").clone(),
                ))
                .or_default();
            entry.0 += 1;
            entry.1 += micros;
        }
        ctx.active_hle.set(None);

        if let Some(request) = ctx.pending_guest_call.take() {
            if let Some(completion) = request.completion {
                let _ = mem.atomic_store_u32(completion.address, completion.failure_u32);
            }
            tracing::error!(
                "{}::{} requested a guest callback through the direct leaf gateway",
                t.library,
                t.function
            );
        }
        ctx.trace
            .push(index as u32, result, [args[0], args[1], args[2]]);
        HLE_EXITS.fetch_add(1, Ordering::Relaxed);

        // Another guest thread can terminate the process while this worker is
        // inside an executable leaf thunk. The VEH path observes that flag at
        // every HLE boundary and restores `recovery_ctx`; the direct path must
        // do the same instead of returning to guest code forever. The old
        // error-only branch left the worker alive after the main thread
        // faulted, producing millions of calls and filling the 64 MiB log in
        // seconds during process teardown.
        if ctx.process_is_terminating() {
            ctx.exited.set(true);
            // SAFETY: `direct_hle_gateway` is reachable only while `run` has
            // armed this ActiveContext and populated `recovery_ctx`. Restoring
            // it abandons the guest/direct-host stacks and resumes `run` at
            // the same recovery point used by `veh_callback`.
            unsafe { RtlRestoreContext(ctx.recovery_ctx, ptr::null()) };
            std::process::abort();
        }
        if let Some(retval) = ctx.thread_exit.take() {
            ctx.retval.set(retval);
            ctx.returned.set(true);
            // SAFETY: identical recovery-context invariant to the process-exit
            // branch above; this mirrors the VEH pthread-exit path.
            unsafe { RtlRestoreContext(ctx.recovery_ctx, ptr::null()) };
            std::process::abort();
        }

        result
    };

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(dispatch)).unwrap_or_else(|_| {
        tracing::error!("HLE handler panicked inside executable thunk gateway; aborting runner");
        std::process::abort()
    })
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
    gpu: &dyn GpuSubmissionSubsystem,
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
    // Diagnostic single-thread mode (RAEEN_SINGLE_THREAD_GUEST): serialize guest
    // native execution across all guest threads, yielding around HLE calls (see
    // `GuestGilHold`). Acquired before the watchdog so a wait for the token is
    // not mistaken for a guest hang; released on every return arm of `run` (the
    // recovery jump restores this frame, so faults and exits release it too).
    let _guest_gil = GuestGilHold::acquire();
    let _watchdog = arm_run_watchdog();
    // The top-level caller holds `call_lock()` to protect the process-wide
    // fixed mappings. Process-owned pthread workers may call `run` concurrently;
    // their dispatch state is isolated by thread-local `ACTIVE_CONTEXT` slots.

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
    // dereferenced synchronously on this thread while `ctx` is alive, and
    // this thread's `ACTIVE_CONTEXT` slot is cleared before `run` returns.
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
    // SAFETY: process GPU services are Arc-owned; the function-mode fallback
    // is a clone of the installed process handle. Both outlive this guarded
    // call and the pointer is cleared with the rest of `ActiveContext`.
    let gpu_erased: &'static dyn GpuSubmissionSubsystem = unsafe {
        core::mem::transmute::<&dyn GpuSubmissionSubsystem, &'static dyn GpuSubmissionSubsystem>(
            gpu,
        )
    };
    let thread_scheduler = guest_threads.unwrap_or(&UNSUPPORTED_GUEST_THREADS);
    // SAFETY: shared schedulers are Arc-owned by the process, while the
    // fallback is static; either way the scheduler outlives this run.
    let thread_scheduler_erased: &'static dyn GuestThreadScheduler = unsafe {
        core::mem::transmute::<&dyn GuestThreadScheduler, &'static dyn GuestThreadScheduler>(
            thread_scheduler,
        )
    };

    let direct_host_stack = DirectHostStack::new()?;
    let direct_host_stack_top = direct_host_stack.top;
    let mut direct_state = Box::new(DirectThreadState {
        context: ptr::null_mut(),
        host_stack_top: direct_host_stack_top,
        previous_teb_slot: 0,
    });

    let ctx = ActiveContext {
        trampolines: trampolines as *const [HleTrampoline],
        unresolved_stubs: unresolved_stubs as *const [UnresolvedStub],
        hle: hle as *const HleRegistry,
        kernel: kernel as *const OrbisKernel,
        mem: mem_erased as *const dyn GuestMemory,
        alloc: alloc_erased as *const dyn GuestAllocator,
        gpu: gpu_erased as *const dyn GpuSubmissionSubsystem,
        thread_scheduler: thread_scheduler_erased as *const dyn GuestThreadScheduler,
        current_thread,
        static_tls_block,
        thread_exit: Cell::new(None),
        region_base: guard.base(),
        region_len: guard.len(),
        callback_return_addr: guard.return_trampoline(),
        tls_rearm_trampoline_redzone: guard.tls_rearm_trampoline_redzone(),
        returned: Cell::new(false),
        retval: Cell::new(0),
        pending_guest_call: Cell::new(None),
        direct_gateway_active: Cell::new(false),
        active_hle: Cell::new(None),
        callback_frames: Cell::new(Vec::new()),
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
    direct_state.context = &ctx as *const ActiveContext as *mut ActiveContext;

    ensure_veh()?;

    if let Ok(value) = std::env::var("RAEEN_DIAGNOSTIC_CODE_ADDR") {
        let value = value.trim_start_matches("0x");
        if let Ok(address) = u64::from_str_radix(value, 16)
            && let Some(disassembly) = diagnostic_disassembly(mem, address, 0x400)
        {
            tracing::info!("diagnostic guest disassembly from {address:#x}:\n{disassembly}");
        }
    }

    // `ctx` is a local; its address is stable for the rest of this function
    // (never moved), which is exactly the lifetime the VEH needs it for.
    ACTIVE_GUEST_WATCH.with(|slot| slot.set(None));
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
        direct_state.previous_teb_slot = direct_state_slot();
        set_direct_state_slot(direct_state.as_ref() as *const DirectThreadState as u64);
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

    set_direct_state_slot(direct_state.previous_teb_slot);

    ACTIVE_GUEST_WATCH.with(|slot| slot.set(None));
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
        Some(mut err) => {
            // A host fault is an emulator bug that the VEH could only record as
            // addresses. Name the HLE call that was in flight and say plainly
            // whose fault it is — allocation and logging are safe here (the
            // guest is stopped and this thread's context is cleared), which is
            // exactly why the VEH left `hle: None` for us to fill in.
            if let RuntimeError::HostFaulted {
                rip,
                access,
                kind,
                hle,
            } = &mut err
            {
                *hle = ctx
                    .active_hle
                    .get()
                    .and_then(|(idx, _)| trampolines.get(idx as usize))
                    .map(|t| format!("{}::{}", t.library, t.function));
                tracing::error!(
                    rip = format_args!("{rip:#x}"),
                    access = format_args!("{access:#x}"),
                    kind = format_args!("{kind}"),
                    hle = hle.as_deref().unwrap_or("<none>"),
                    "HOST FAULT: emulator code faulted while a guest call was in \
                     flight — this is a Raeen bug, not a guest fault. The guest run was \
                     abandoned at its recovery point; host frames below the fault were \
                     discarded without unwinding."
                );
                log_call_trace(&ctx, trampolines, &err);
                return Err(err);
            }
            // A divide-error trap. The VEH could only record addresses; name the
            // HLE call in flight here (allocation and logging are safe now) and
            // say plainly what died, because until this arm existed the whole
            // process vanished with exit code `0xC000_0094` and an empty log.
            if let RuntimeError::IntegerDivideFault {
                rip,
                cause,
                origin,
                hle,
            } = &mut err
            {
                *hle = ctx
                    .active_hle
                    .get()
                    .and_then(|(idx, _)| trampolines.get(idx as usize))
                    .map(|t| format!("{}::{}", t.library, t.function));
                let guidance = match origin {
                    crate::FaultOrigin::Guest => {
                        "guest code divided by a value that was zero. Very often that zero came \
                         FROM US: an HLE stub returning 0 (or leaving an out-parameter \
                         untouched) for a grain size, sample rate, stride, element size, or \
                         frequency the title then divides by. The recent-HLE-call trace below \
                         is the place to look."
                    }
                    crate::FaultOrigin::Host => {
                        "emulator code took a divide-error trap. This cannot come from safe \
                         Rust integer division (rustc emits an explicit zero check and panics \
                         instead), so it is a C/C++ dependency or inline assembly."
                    }
                };
                tracing::error!(
                    rip = format_args!("{rip:#x}"),
                    cause = format_args!("{cause}"),
                    origin = format_args!("{origin}"),
                    hle = hle.as_deref().unwrap_or("<none>"),
                    "INTEGER DIVIDE FAULT: {guidance}"
                );
                log_call_trace(&ctx, trampolines, &err);
                return Err(err);
            }
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
    if let Some((idx, args)) = ctx.active_hle.get()
        && let Some(t) = trampolines.get(idx as usize)
    {
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
        // TEMP-DIAG (2026-07-23, ASTRO.BOT +0xe03f1a NULL-base fault diagnosis;
        // REMOVE after the investigation): the snapshot struct already captures
        // r12-r15 but the report never printed them — the faulting voice/list
        // pointer lives in r14. Also, with RAEEN_DUMP_VOICE_LIST set, walk the
        // SAL voice list observed at module+0xe7f6020 ([mgr]->+0x38->+0x10
        // chain, node link at voice+0xe8) to show which node is half-linked.
        tracing::warn!(
            "TEMP-DIAG snapshot r12={:#x} r13={:#x} r14={:#x} r15={:#x}",
            snapshot.r12,
            snapshot.r13,
            snapshot.r14,
            snapshot.r15,
        );
        if std::env::var_os("RAEEN_DUMP_VOICE_LIST").is_some() {
            let read_q = |addr: u64| -> Option<u64> {
                let mut b = [0u8; 8];
                if unsafe { &*ctx.mem }.read(addr, &mut b) {
                    Some(u64::from_le_bytes(b))
                } else {
                    None
                }
            };
            let mut lines = String::new();
            let mgr = read_q(0x10000e7f6020);
            lines.push_str(&format!("mgr=[0xe7f6020]={mgr:#x?}\n"));
            if let Some(mgr) = mgr {
                let container = read_q(mgr.wrapping_add(0x38));
                lines.push_str(&format!("  [mgr+0x38]={container:#x?}\n"));
                if let Some(container) = container {
                    let mut node = read_q(container.wrapping_add(0x10)).unwrap_or(0);
                    for i in 0..32 {
                        if node == 0 {
                            lines.push_str(&format!("  node[{i}] = 0 (end)\n"));
                            break;
                        }
                        let link = read_q(node.wrapping_add(0xe8));
                        let flag120 = read_q(node.wrapping_add(0x120)).map(|v| v & 0xff);
                        lines.push_str(&format!(
                            "  node[{i}]={node:#x} [+0xe8]={link:#x?} [+0x120]&0xff={flag120:#x?}\n"
                        ));
                        match link {
                            Some(l) if l != 0 => node = read_q(l.wrapping_add(0x10)).unwrap_or(0),
                            _ => {
                                lines.push_str("  ^^ NULL/unreadable link — half-linked node\n");
                                break;
                            }
                        }
                    }
                }
            }
            tracing::warn!("TEMP-DIAG voice list walk:\n{lines}");
        }

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

        // Guest stack return-address walk. A fault inside a leaf libc routine
        // (memcpy/strlen/…) names the routine in the register dump but not WHO
        // called it with the bad pointer — the return-address chain does. Scan
        // qwords upward from RSP for values that land in the loaded guest image
        // (plausible return addresses) and log them module-relative so
        // `--dump-vaddr` can decode the caller.
        {
            let mut stack = [0u8; 512];
            if mem.read(snapshot.rsp, &mut stack) {
                let mut chain = Vec::new();
                for qw in stack.chunks_exact(8) {
                    let v = u64::from_le_bytes(qw.try_into().unwrap_or([0; 8]));
                    if (crate::GUEST_ARENA_BASE..crate::GUEST_ARENA_BASE + 0x2000_0000).contains(&v)
                    {
                        chain.push(format!("+{:#x}", v - crate::GUEST_ARENA_BASE));
                        if chain.len() >= 12 {
                            break;
                        }
                    }
                }
                if !chain.is_empty() {
                    tracing::warn!(
                        "guest stack return-addr chain (module-relative): {}",
                        chain.join(" <- ")
                    );
                }
            }
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
        tracing::error!("{err} — no HLE calls were serviced before it");
        return;
    }

    // ---- Publish the ring for the crash report. ----
    //
    // This `CallTrace` lives in the faulting thread's `ActiveContext` and dies
    // with the run, so the crash report — assembled afterwards in `raeen-gui`
    // from `kernel.recent_hle_calls` — used to find nothing. That DashMap was
    // only ever written by the `RAEEN_TRACE_EINVAL` diagnostic, so on a normal
    // run the report's "Recent HLE calls" section rendered `<none recorded>`
    // for a fault whose own log line said "35 HLE call(s) recorded"
    // (measured: Dead Cells PPSA15552, 2026-07-29). That section is the one
    // `.github/ISSUE_TEMPLATE/game-report.yml` asks users to paste, so shipping
    // it blank threw away the most useful thing a report can carry.
    //
    // Written oldest-first (the deque convention the diagnostic writer uses and
    // the report's renderer reverses), and the return value is kept: a `-> 0x0`
    // a few calls before a null dereference is the usual cause, and it is the
    // same lead the distillation below promotes.
    {
        // SAFETY: `ctx.kernel` is the live kernel installed by `run` for this
        // call; this runs on the normal path after the guest has stopped, on
        // the same thread, before the context slot is cleared.
        let kernel = unsafe { &*ctx.kernel };
        let start = entries.len().saturating_sub(CRASH_REPORT_RING_LEN);
        let published: Vec<String> = entries[start..]
            .iter()
            .map(|&(idx, ret, _args)| match trampolines.get(idx as usize) {
                Some(t) => format!("{}::{} -> {ret:#x}", t.library, t.function),
                None => format!("<trampoline #{idx}> -> {ret:#x}"),
            })
            .collect();
        let ring = kernel
            .recent_hle_calls
            .entry(ctx.current_thread())
            .or_default();
        let mut queue = ring.lock();
        // Replace rather than append: this ring is authoritative (always-on and
        // 4096 deep) where the diagnostic writer's is opt-in and 24 deep.
        queue.clear();
        queue.extend(published);
    }

    // ---- Distilled crash report: emitted FIRST and PROMINENTLY. ----
    //
    // The fault site (registers, module+offset, fault-site bytes, stack walk,
    // register text previews) has already been logged above at WARN — that is
    // part (a) of the report. What follows are the *distilled* leads a reader
    // actually acts on: the HLE calls that returned an error, the
    // pointer-returning calls that handed the guest a null, and the guest's own
    // unwind chain.
    //
    // The full ~4096-entry ring used to be spilled right here, one `tracing::warn!`
    // per entry — ~4096 log events, each paying a String alloc, full formatting,
    // and a Mutex lock through the ConsoleLayer; three faults produced ~11,000
    // WARN lines (92% of the log). It is now assembled into ONE String and
    // emitted ONCE at DEBUG at the end of this function. Nothing is lost: the
    // leads are promoted, the raw ring is demoted.

    // Lead line, promoted to ERROR so the single most important fact is not
    // buried among the WARN-level detail lines around it.
    tracing::error!(
        "{err} — {} HLE call(s) recorded before the fault; distilled leads follow at WARN, \
         the full oldest-first ring once at DEBUG",
        entries.len()
    );

    let mem = unsafe { &*ctx.mem };

    // (b) HLE calls that returned an Orbis error before the fault. A guest that
    // throws, asserts, or dereferences a null it was handed is usually reacting
    // to one of these; picking them out by eye across 4096 entries is not a
    // thing anyone should have to do.
    let mut failures = Vec::new();
    for &(idx, ret, _args) in &entries {
        if !is_orbis_error(ret) {
            continue;
        }
        if let Some(t) = trampolines.get(idx as usize) {
            let item = format!("{}::{} -> {ret:#x}", t.library, t.function);
            if !failures.contains(&item) {
                failures.push(item);
            }
        }
    }
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

    // (c) pointer-returning HLE calls that handed the guest 0x0. A second,
    // independent lens on the same call ring. The Orbis-error filter above only
    // catches returns in the 0x8xxx_xxxx range; it is blind to the more common
    // retail failure — a pointer/handle-returning call that handed back 0x0.
    // That null is not an error code, so nothing flags it, yet it is the usual
    // value a guest dereferences a few calls later (the faulting read of a small
    // offset like 0x10/0x28 is `null->field`).
    //
    // The registry carries no return-type metadata, so classify by
    // self-calibration from the ring itself: a function is "pointer-returning"
    // if it returned a *readable guest address* on at least one recorded call.
    // If it ALSO returned 0x0 on another call, those zeros are the candidates. A
    // function whose 0x0 is a normal success (mutex unlock, etc.) never returns a
    // readable pointer, so it is excluded automatically; a timestamp/counter
    // returns large values that are not mapped guest addresses, so it is excluded
    // too.
    let readable = |value: u64| value != 0 && mem.read(value, &mut [0u8; 1]);
    let mut profiles: std::collections::BTreeMap<(&str, &str), (bool, usize)> =
        std::collections::BTreeMap::new();
    for &(idx, ret, _args) in &entries {
        let Some(t) = trampolines.get(idx as usize) else {
            continue;
        };
        let profile = profiles
            .entry((t.library.as_str(), t.function.as_str()))
            .or_insert((false, 0));
        if ret == 0 {
            profile.1 += 1;
        } else if readable(ret) {
            profile.0 = true;
        }
    }
    let mut null_pointer_calls: Vec<(String, usize)> = profiles
        .into_iter()
        .filter(|&(_, (returned_pointer, zeros))| returned_pointer && zeros > 0)
        .map(|((library, function), (_, zeros))| (format!("{library}::{function}"), zeros))
        .collect();
    null_pointer_calls.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if !null_pointer_calls.is_empty() {
        let rendered = null_pointer_calls
            .iter()
            .map(|(name, zeros)| {
                format!("{name} -> 0x0 (×{zeros}; also returned a live pointer elsewhere in this window)")
            })
            .collect::<Vec<_>>()
            .join("\n    ");
        tracing::warn!(
            "pointer-returning HLE calls that handed the guest 0x0 before this fault \
             ({} distinct) — a null from one of these is the usual thing a guest \
             dereferences into the fault above:\n    {}",
            null_pointer_calls.len(),
            rendered,
        );
    }

    // (d) guest unwind PC lookups: the addresses the guest's own C++ unwinder
    // asked about while walking the most recent exception. The last one is
    // typically the throw site.
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

    // Supplementary: recent NUL-terminated guest strings observed passing
    // through libc string routines — often the assert/exception message.
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

    // ---- Full raw ring: ONE DEBUG event, not one WARN per entry. ----
    //
    // Assembling the whole oldest-first ring into a single String and emitting
    // it once replaces the ~4096 log events / String allocs / Mutex locks the
    // old per-entry loop paid on every fault. A `-> 0x0` a few lines above the
    // fault is still the usual cause of a null dereference in guest code, so the
    // per-entry detail is preserved verbatim — just demoted to DEBUG.
    use std::fmt::Write as _;
    let mut ring = String::with_capacity(entries.len() * 48);
    let _ = write!(
        ring,
        "full HLE call ring before the fault ({} entries, oldest first) — a call \
         returning 0x0 here is the usual cause of a null dereference in guest code:",
        entries.len()
    );
    for &(idx, ret, args) in &entries {
        let marker = if is_orbis_error(ret) {
            "  <-- ERROR"
        } else {
            ""
        };
        match trampolines.get(idx as usize) {
            Some(t) => {
                let _ = write!(
                    ring,
                    "\n    {}::{}({args:x?}) -> {ret:#x}{marker}",
                    t.library, t.function,
                );
            }
            None => {
                let _ = write!(ring, "\n    <trampoline #{idx}> -> {ret:#x}{marker}");
            }
        }
    }
    if std::env::var_os("RAEEN_CRASH_RING_WARN").is_some() {
        let tail = ring
            .lines()
            .skip(entries.len().saturating_sub(64) + 1)
            .collect::<Vec<_>>()
            .join("\n");
        tracing::warn!("last HLE calls before the fault (diagnostic):\n{tail}");
    }
    tracing::debug!("{ring}");
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

/// One blocker slot per trampoline index, so a repeated error return never
/// touches the blockers mutex, hashes, or allocates.
///
/// Sized from the live trampoline table the first time any HLE call returns an
/// error. A later run in the same process with a *larger* table would index
/// past the end; that case falls back to [`blockers::record`] rather than
/// dropping the event. It only arises in test binaries — the Shell launches one
/// process per title.
static ORBIS_ERROR_SLOTS: OnceLock<Box<[OnceLock<Option<blockers::Slot>>]>> = OnceLock::new();

/// Count one Orbis error return against `library::function`.
///
/// This is the site that makes the measured silent-zero-frame cluster
/// diagnosable at all. Three of its four titles fail on an ordinary
/// **registered** HLE function saying no — `sceKernelDlsym(handle=0,
/// symbol='scriptingGetMem') ENOENT` for Blasphemous II and Subnautica BZ
/// (`docs/silent-zero-frame-cluster.md`) — so no unimplemented-shim or
/// unresolved-NID hook can ever fire for them, and a report carrying only those
/// hooks prints a correct, useless `<none recorded>`.
///
/// An Orbis error code is one of the few values safe to put in the intern
/// `subject`: it is a bounded enumeration, unlike an address. It pins the FIRST
/// code an import returned, and a later different code from the same import is
/// counted against that entry rather than interning a second one — which keeps
/// the 32-key category cap spent on distinct *imports*, the axis a reader acts
/// on, rather than on one chatty import's errno variety.
#[inline]
fn record_orbis_error(
    index: u64,
    trampoline_count: usize,
    library: &str,
    function: &str,
    ret: u64,
) {
    let slots =
        ORBIS_ERROR_SLOTS.get_or_init(|| (0..trampoline_count).map(|_| OnceLock::new()).collect());
    let Some(cell) = slots.get(index as usize) else {
        blockers::record(
            BlockerCategory::OrbisError,
            format!("{library}::{function}"),
            ret,
            || format!("returned {ret:#x} (first observed code)"),
        );
        return;
    };
    if let Some(slot) = *cell.get_or_init(|| {
        blockers::intern(
            BlockerCategory::OrbisError,
            format!("{library}::{function}"),
            ret,
            || format!("returned {ret:#x} (first observed code)"),
        )
    }) {
        blockers::bump(slot);
    }
}

/// Whose code was running when an access violation arrived, once the fast
/// paths (HLE trampoline dispatch, FS-base re-arm, demand-commit) have all
/// declined it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AvOrigin {
    /// Guest code faulted: a wild pointer, a null dereference, a jump into
    /// data. The title's problem, recovered and reported as
    /// [`RuntimeError::Faulted`].
    Guest,
    /// The fault names an unresolved-import stub, by `rip` (the guest CALLED a
    /// missing import) or by access address (it READ a slot still pointing at
    /// one). Reported as [`RuntimeError::UnimplementedImport`] — the runtime's
    /// most actionable outcome, and the hot inventory path.
    UnresolvedStub,
    /// **Our** code faulted: a bad dereference inside a Rust HLE handler, or
    /// inside something one called. Reported as
    /// [`RuntimeError::HostFaulted`], never laundered as a guest fault.
    Host,
}

/// Decide whose fault an access violation is.
///
/// Pure, so the rules can be pinned by table tests rather than by provoking
/// real faults. Every input is a fact the VEH already has (or can get without
/// allocating): the access type from `ExceptionInformation[0]`, whether `rip`
/// reads back through [`GuestMemory`] (i.e. is guest-mapped, committed memory),
/// whether the arena's VMA map positively attributes `rip` to the host process,
/// and whether either address resolves to an unresolved-import stub.
///
/// # The rules, and why
///
/// 1. **Stub involvement wins.** Both the CALLED and the READ shapes must keep
///    naming the missing NID; that reporting (and the non-strict resume-with-0
///    inventory path) is unchanged by this classification.
/// 2. **`rip` in guest memory ⇒ guest.** The overwhelmingly common case, and
///    the one whose behaviour must not move at all.
/// 3. **An instruction *fetch* failure is always the guest's.** A guest that
///    `jmp`s to `0x8`, calls through a null function pointer, or returns into
///    data faults with the access type `Execute` at an address that is by
///    definition not executable — and therefore not guest-readable either. Host
///    code cannot produce this shape: it is running, so its own page is mapped
///    executable. Tests in `tests/execute.rs` deliberately observe `argc` by
///    jumping to it (`Faulted { addr: 1 }`), so this rule is load-bearing.
/// 4. **A read/write from host-owned `rip` is ours.** The faulting instruction
///    was fetched and executed, so its page *is* mapped executable — and the
///    map says the page belongs to the host process. Nothing else can be true
///    of it.
/// 5. **Anything else stays a guest fault.** Unknown access types, addresses
///    the map cannot answer for: the pre-existing path, unchanged. Host
///    attribution is a positive claim only.
fn classify_access_violation(
    kind: crate::FaultKind,
    rip_is_guest_memory: bool,
    rip_is_host_owned: bool,
    rip_is_stub: bool,
    access_is_stub: bool,
) -> AvOrigin {
    if rip_is_stub || access_is_stub {
        return AvOrigin::UnresolvedStub;
    }
    if rip_is_guest_memory {
        return AvOrigin::Guest;
    }
    if matches!(kind, crate::FaultKind::Read | crate::FaultKind::Write) && rip_is_host_owned {
        return AvOrigin::Host;
    }
    AvOrigin::Guest
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

fn read_guest_u64(mem: &dyn GuestMemory, address: u64) -> Option<u64> {
    let mut bytes = [0u8; 8];
    mem.read(address, &mut bytes)
        .then(|| u64::from_le_bytes(bytes))
}

/// Return every valid x64 instruction ending at `rip_after`.
///
/// A data breakpoint is delivered after the write has retired, so Windows
/// reports the following RIP. x64 has no architecturally unique backwards
/// decode point; retaining every 1..=15-byte candidate is more honest than
/// guessing one and calling it the writer. The true instruction is always in
/// this bounded set, and the surrounding disassembly logged with it resolves
/// any ambiguity.
fn writer_instruction_candidates(mem: &dyn GuestMemory, rip_after: u64) -> Vec<String> {
    let mut candidates = Vec::new();
    for distance in 1..=15u64 {
        let Some(start) = rip_after.checked_sub(distance) else {
            continue;
        };
        let mut bytes = [0u8; 15];
        if !mem.read(start, &mut bytes) {
            continue;
        }
        let mut decoder = Decoder::with_ip(64, &bytes, start, DecoderOptions::NONE);
        let instruction = decoder.decode();
        if instruction.is_invalid() || instruction.next_ip() != rip_after {
            continue;
        }
        let mut rendered = String::new();
        let mut formatter = IntelFormatter::new();
        formatter.format(&instruction, &mut rendered);
        candidates.push(format!("{start:#x}: {rendered}"));
    }
    candidates
}

fn maybe_arm_guest_watchpoint(
    context: &mut CONTEXT,
    mem: &dyn GuestMemory,
    function: &str,
    hle_index: u64,
) {
    if let Some(active) = ACTIVE_GUEST_WATCH.with(|slot| slot.get()) {
        let expected_dr7 = write_watch_dr7(8).expect("8-byte watchpoints are supported");
        if context.Dr0 != active.address || context.Dr7 != expected_dr7 {
            tracing::warn!(
                "guest write watch debug registers were not preserved across HLE #{hle_index} \
                 ({function}); restoring dr0={:#x} dr7={expected_dr7:#x} \
                 (observed dr0={:#x} dr7={:#x})",
                active.address,
                context.Dr0,
                context.Dr7
            );
            context.ContextFlags |= CONTEXT_DEBUG_REGISTERS_AMD64_FLAG;
            context.Dr0 = active.address;
            context.Dr6 = 0;
            context.Dr7 = expected_dr7;
        }
        return;
    }
    let Some(request) = guest_watch_request() else {
        return;
    };
    let address = match request {
        GuestWatchRequest::Explicit(address) => *address,
        GuestWatchRequest::Indirect(pointer_address) => {
            let Some(address) = read_guest_u64(mem, *pointer_address) else {
                tracing::warn!(
                    "cannot arm indirect guest write watch: unreadable pointer at {pointer_address:#x}"
                );
                return;
            };
            address
        }
        GuestWatchRequest::CallerCanaryAfter(target) => {
            if target != function {
                return;
            }
            let frame_depth = watch_canary_frame_depth();
            let Some(address) = stack_canary_address(mem, context.Rbp, frame_depth) else {
                tracing::warn!(
                    "cannot auto-arm canary watch after {function}: cannot resolve depth \
                     {frame_depth} from rbp {:#x}",
                    context.Rbp,
                );
                return;
            };
            address
        }
    };
    if !address.is_multiple_of(8) {
        tracing::warn!(
            "cannot arm 8-byte guest write watch at unaligned address {address:#x}; \
             choose an aligned RAEEN_WATCH_ADDR"
        );
        return;
    }
    let Some(initial_value) = read_guest_u64(mem, address) else {
        tracing::warn!("cannot arm guest write watch at unreadable address {address:#x}");
        return;
    };

    context.ContextFlags |= CONTEXT_DEBUG_REGISTERS_AMD64_FLAG;
    context.Dr0 = address;
    context.Dr6 = 0;
    context.Dr7 = write_watch_dr7(8).expect("8-byte watchpoints are supported");
    ACTIVE_GUEST_WATCH.with(|slot| {
        slot.set(Some(ActiveGuestWatch {
            address,
            initial_value,
            armed_by_hle_index: hle_index,
        }));
    });
    tracing::info!(
        "armed guest write watch address={address:#x} initial={initial_value:#018x} \
         after HLE #{hle_index} ({function}), canary_frame_depth={}",
        watch_canary_frame_depth()
    );
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
    let is_breakpoint = record.ExceptionCode == EXCEPTION_BREAKPOINT;
    let is_single_step = record.ExceptionCode == EXCEPTION_SINGLE_STEP_CODE;
    // An x86 divide-error trap (`#DE`). Not handling this used to mean a `div`
    // by zero anywhere — guest or host — took `EXCEPTION_CONTINUE_SEARCH` and
    // killed the process with `0xC000_0094` and no log line whatsoever. See
    // `RuntimeError::IntegerDivideFault`.
    let divide_fault = match record.ExceptionCode {
        EXCEPTION_INT_DIVIDE_BY_ZERO => Some(crate::DivideFault::ByZero),
        EXCEPTION_INT_OVERFLOW => Some(crate::DivideFault::Overflow),
        _ => None,
    };
    if !is_access_violation
        && !is_illegal_instruction
        && !is_breakpoint
        && !is_single_step
        && divide_fault.is_none()
    {
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

    if is_single_step {
        // DR6.B0 identifies the local DR0 breakpoint. Do not consume an
        // unrelated trap-flag/debugger single-step merely because a guest
        // data watch is active.
        if context.Dr6 & 1 == 0 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let Some(watch) = ACTIVE_GUEST_WATCH.with(|slot| slot.take()) else {
            return EXCEPTION_CONTINUE_SEARCH;
        };
        GUEST_WATCHPOINT_HITS.fetch_add(1, Ordering::Relaxed);
        // Disable the local data breakpoint before doing any more work. The
        // watched address is guest memory, but logging/unwinding below can
        // inspect that memory and must not recursively re-enter this handler.
        context.ContextFlags |= CONTEXT_DEBUG_REGISTERS_AMD64_FLAG;
        context.Dr0 = 0;
        context.Dr6 = 0;
        context.Dr7 = 0;

        let mem = unsafe { &*ctx.mem };
        let current_value = read_guest_u64(mem, watch.address);
        let candidates = writer_instruction_candidates(mem, context.Rip);
        let candidate_text = if candidates.is_empty() {
            "<no valid backwards decode>".to_string()
        } else {
            candidates.join(" | ")
        };
        let module = unsafe { &*ctx.kernel }
            .unwind_module_for_addr(context.Rip)
            .map(|module| {
                format!(
                    "{}+{:#x}",
                    module.name,
                    context.Rip.saturating_sub(module.start)
                )
            })
            .unwrap_or_else(|| format!("{:#x}", context.Rip));
        let trigger = unsafe { &*ctx.trampolines }
            .get(watch.armed_by_hle_index as usize)
            .map(|trampoline| format!("{}::{}", trampoline.library, trampoline.function))
            .unwrap_or_else(|| format!("HLE #{}", watch.armed_by_hle_index));
        tracing::error!(
            "guest write watchpoint hit: address={:#x} initial={:#018x} current={} \
             rip_after={:#x} ({module}) writer_candidates=[{candidate_text}] armed_by={trigger}",
            watch.address,
            watch.initial_value,
            current_value
                .map(|value| format!("{value:#018x}"))
                .unwrap_or_else(|| "<unreadable>".to_string()),
            context.Rip,
        );
        if let Some(disassembly) =
            diagnostic_disassembly(mem, context.Rip.saturating_sub(0x20), 0x80)
        {
            tracing::error!(
                "guest write watchpoint surrounding disassembly (rip_after={:#x}):\n{disassembly}",
                context.Rip
            );
        }

        // A Windows exception dispatch can clear the user-set FS base just
        // like a scheduler preemption. Resume through the established TLS
        // re-arm stub so the instruction after the writer sees the guest TCB,
        // not the host's cleared FS base.
        let resume_rip = context.Rip;
        let resume_rsp = context.Rsp;
        resume_guest_with_tls(ctx, context, mem, resume_rip, resume_rsp);
        return EXCEPTION_CONTINUE_EXECUTION;
    }

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

    // An x86 divide-error trap. There is nothing to emulate and nothing to
    // retry — resuming the same instruction would re-fault forever — so the
    // only useful response is the one a genuine guest fault already gets:
    // record what happened, roll back abandoned callback completions, snapshot
    // the registers, and recover through `run`'s pre-entry context. That turns
    // a silent `0xC000_0094` process kill into a report naming the instruction,
    // its bytes (which encode the divisor register), the register file, and the
    // HLE call in flight. See `RuntimeError::IntegerDivideFault`.
    if let Some(cause) = divide_fault {
        // SAFETY: `ctx.mem` outlives the guarded call, per `ActiveContext`.
        let mem = unsafe { &*ctx.mem };
        // Same test the illegal-instruction arm uses: guest code lives at or
        // above the arena base, emulator code below it.
        let origin = if fault_addr >= crate::GUEST_ARENA_BASE {
            crate::FaultOrigin::Guest
        } else {
            crate::FaultOrigin::Host
        };
        ctx.error.set(Some(RuntimeError::IntegerDivideFault {
            rip: fault_addr,
            cause,
            origin,
            hle: None,
        }));
        // Frames OUT of the cell first, writes second: each `atomic_store_u32`
        // targets a guest-controlled address and may fault straight back into
        // this handler. See `ActiveContext::callback_frames`.
        for frame in ctx.take_callback_frames_innermost_first() {
            if let Some(completion) = frame.completion {
                let _ = mem.atomic_store_u32(completion.address, completion.failure_u32);
            }
        }
        let mut bytes = [0u8; 16];
        // Host code is not in guest memory, so a host `#DE`'s bytes read back
        // as unreadable rather than as whatever `read` left in the buffer.
        let bytes_read = origin == crate::FaultOrigin::Guest && mem.read(fault_addr, &mut bytes);
        if !bytes_read {
            bytes = [0u8; 16];
        }
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
        // SAFETY: identical invariant to every other recovery in this handler —
        // `recovery_ctx` points at `run`'s still-live stack snapshot on this
        // same synchronously-faulting thread, and CONTEXT is Copy.
        *context = unsafe { *ctx.recovery_ctx };
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    if is_breakpoint {
        // A one-shot export trap (`RAEEN_TRAP_MODULE_EXPORTS`): an `int3`
        // planted on a module export's entry byte. `ExceptionAddress` is the
        // `int3` itself for breakpoint exceptions (the kernel rewinds it), so
        // use the record — not `Rip` — as the authoritative trap address.
        // `take_hit` logs once and restores the original byte; resume at the
        // SAME address through the TLS-rearm stub, exactly like the syscall
        // path, so the next `fs:` access never sees the host TEB. Anything
        // not in the trap map (debugger int3, guest int3 padding) is passed
        // on unchanged.
        // SAFETY: `ctx.mem` outlives the guarded call, per `ActiveContext`.
        let mem = unsafe { &*ctx.mem };
        let fault = record.ExceptionAddress as u64;
        let trap_regs = crate::export_trap::TrapRegisters {
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
            r12: context.R12,
            r13: context.R13,
            r14: context.R14,
            r15: context.R15,
        };
        if let Some(hit) = crate::export_trap::take_hit(fault, mem, &trap_regs) {
            if let Some((register, value)) = hit.register_write {
                use crate::export_trap::TrapRegister;
                match register {
                    TrapRegister::Rax => context.Rax = value,
                    TrapRegister::Rcx => context.Rcx = value,
                    TrapRegister::Rdx => context.Rdx = value,
                    TrapRegister::Rbx => context.Rbx = value,
                    TrapRegister::Rsp => context.Rsp = value,
                    TrapRegister::Rbp => context.Rbp = value,
                    TrapRegister::Rsi => context.Rsi = value,
                    TrapRegister::Rdi => context.Rdi = value,
                    TrapRegister::R8 => context.R8 = value,
                    TrapRegister::R9 => context.R9 = value,
                    TrapRegister::R10 => context.R10 = value,
                    TrapRegister::R11 => context.R11 = value,
                    TrapRegister::R12 => context.R12 = value,
                    TrapRegister::R13 => context.R13 = value,
                    TrapRegister::R14 => context.R14 = value,
                    TrapRegister::R15 => context.R15 = value,
                }
            }
            resume_guest_with_tls(ctx, context, mem, hit.resume_rip, context.Rsp);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }

    if is_illegal_instruction {
        let mem = unsafe { &*ctx.mem };
        let mut marker = [0u8; 2];
        if mem.read(fault_addr, &mut marker)
            && marker == raeen_firmware::dynlib::linker::CPUID_TRAP_BYTES
        {
            let leaf = context.Rax as u32;
            let subleaf = context.Rcx as u32;
            let (eax, ebx, ecx, edx) = raeen_kernel::hypervisor::ps5_cpuid(leaf, subleaf);
            context.Rax = u64::from(eax);
            context.Rbx = u64::from(ebx);
            context.Rcx = u64::from(ecx);
            context.Rdx = u64::from(edx);
            let return_rip = fault_addr.wrapping_add(marker.len() as u64);
            if std::env::var_os("RAEEN_TRACE_CPUID").is_some() {
                tracing::info!(
                    "guest CPUID leaf={leaf:#010x} subleaf={subleaf:#010x} -> \
                     eax={eax:#010x} ebx={ebx:#010x} ecx={ecx:#010x} edx={edx:#010x}"
                );
            }
            resume_guest_with_tls(ctx, context, mem, return_rip, context.Rsp);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if marker != raeen_firmware::dynlib::linker::SYSCALL_TRAP_BYTES {
            // Not our syscall trap. If the bad instruction is in guest code
            // (arena range), it is a genuine undefined opcode — a wild jump into
            // data, a corrupted vtable/function-pointer call landing off-code, a
            // `ud2`. Passing it through kills the whole process with no report
            // (the exact "unreported SIGILL" seen bringing up ASTRO.BOT's worker
            // threads). Record it as an execute-fault and recover through `run`'s
            // snapshot instead, so it surfaces as "guest fault at <rip>" with the
            // offending bytes — which names the crash site. Host-code illegal
            // instructions (below the arena) are still passed through untouched.
            if fault_addr >= crate::GUEST_ARENA_BASE {
                ctx.error.set(Some(RuntimeError::Faulted {
                    addr: fault_addr,
                    access: fault_addr,
                    kind: crate::FaultKind::Execute,
                }));
                // Frames OUT of the cell first, writes second: each
                // `atomic_store_u32` targets a guest-controlled address and may
                // fault straight back into this handler. See
                // `ActiveContext::callback_frames`.
                for frame in ctx.take_callback_frames_innermost_first() {
                    if let Some(completion) = frame.completion {
                        let _ = mem.atomic_store_u32(completion.address, completion.failure_u32);
                    }
                }
                let mut bytes = [0u8; 16];
                let bytes_read = mem.read(fault_addr, &mut bytes);
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
                // SAFETY: identical invariant to the genuine-fault recovery
                // below — `recovery_ctx` points at `run`'s still-live stack
                // snapshot on this same (synchronously-faulting) thread, and
                // CONTEXT is Copy.
                *context = unsafe { *ctx.recovery_ctx };
                return EXCEPTION_CONTINUE_EXECUTION;
            }
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
                if value == 22
                    && *TRACE_EINVAL
                        .get_or_init(|| std::env::var_os("RAEEN_TRACE_EINVAL").is_some())
                {
                    tracing::warn!(
                        "EINVAL(22) from syscall {number}({args:x?}) on thread {}",
                        ctx.current_thread()
                    );
                }
            }
            Err(error) => {
                let errno = match error {
                    raeen_core::error::KernelError::UnimplementedSyscall { .. } => 78, // ENOSYS
                    raeen_core::error::KernelError::MmapFailed { .. } => 12,           // ENOMEM
                    raeen_core::error::KernelError::InvalidMemoryAccess(_) => 14,      // EFAULT
                    raeen_core::error::KernelError::ThreadCreationFailed(_) => 11,     // EAGAIN
                    raeen_core::error::KernelError::FileNotFound(_) => 2,              // ENOENT
                    raeen_core::error::KernelError::PermissionDenied(_) => 13,         // EACCES
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
        // Popped out of the cell before any guest write below, for the reason
        // documented on `ActiveContext::callback_frames`.
        if let Some(frame) = ctx.pop_callback_frame() {
            // SAFETY: `ctx.mem` is the live guest memory view stored by `run`
            // for this guarded call; callback returns are delivered
            // synchronously on the same thread before `run` can tear it down.
            let mem = unsafe { &*ctx.mem };
            if let Some(completion) = frame.completion {
                let _ = mem.atomic_store_u32(completion.address, completion.success_u32);
            }
            apply_hle_result(context, frame.hle_result, frame.float_return);
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
                && !(0x1_0000_0000..0xFFFF_8000_0000_0000).contains(&access_addr);
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
                // after the OS has completed exception return. The staging
                // sits BELOW the red zone (`[rsp-144, rsp-128)`, red-zone
                // stub) — the interrupted instruction is mid-function, so a
                // leaf frame's live red-zone locals must survive the rearm.
                let Some(staged_rsp) = context.Rsp.checked_sub(144) else {
                    return EXCEPTION_CONTINUE_SEARCH;
                };
                if mem.write(staged_rsp, &context.R11.to_le_bytes())
                    && mem.write(staged_rsp + 8, &context.Rip.to_le_bytes())
                {
                    context.Rsp = staged_rsp;
                    context.R11 = want;
                    context.Rip = ctx.tls_rearm_trampoline_redzone;
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

        // SAFETY: `ctx.unresolved_stubs` was set by `run` from process-owned
        // storage that outlives this worker call. The VEH runs synchronously on
        // the same OS thread, and that thread's context slot is cleared before
        // `run` returns, so this read is sound.
        let stubs = unsafe { &*ctx.unresolved_stubs };

        // Whose fault is this, actually? Every access violation delivered on a
        // thread with an armed context used to be recorded as a *guest* fault at
        // `Rip` — including one raised inside a Rust HLE handler, which then
        // surfaced as "guest fault at 0x7ff…" and pointed the investigation at
        // the title instead of at us. Classify first; only the `Host` verdict
        // changes any behaviour (see `classify_access_violation`).
        //
        // SAFETY: `ctx.mem`/`ctx.alloc` are the live views installed by `run`
        // for this guarded call, on this synchronously-faulting thread — the
        // same invariant every other access in this arm relies on.
        let probe_mem = unsafe { &*ctx.mem };
        let mut probe = [0u8; 1];
        let origin = classify_access_violation(
            kind,
            probe_mem.read(fault_addr, &mut probe),
            unsafe { &*ctx.alloc }.address_is_host_owned(fault_addr),
            stub::resolve(fault_addr, stubs).is_some(),
            stub::resolve(access, stubs).is_some(),
        );
        if origin == AvOrigin::Host {
            // Recover exactly as a guest fault does — the Shell survives and the
            // run reports — but under a name that says whose bug it is. The HLE
            // call is resolved to a name by `run`, not here: the VEH must not
            // allocate, and `active_hle` already survives the longjmp.
            ctx.error.set(Some(RuntimeError::HostFaulted {
                rip: fault_addr,
                access,
                kind,
                hle: None,
            }));
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
                // The bytes come from guest memory, which by definition does not
                // contain this instruction; report them as unread rather than
                // printing whatever `read` left in the buffer.
                bytes: [0u8; 16],
                bytes_read: false,
            }));
            // Roll back abandoned callback completions for the same reason the
            // guest-fault path does: the longjmp below discards the whole guest
            // stack, nested callbacks included.
            for frame in ctx.take_callback_frames_innermost_first() {
                if let Some(completion) = frame.completion {
                    let _ = probe_mem.atomic_store_u32(completion.address, completion.failure_u32);
                }
            }
            // SAFETY: identical invariant to the genuine-fault recovery below —
            // `recovery_ctx` points at `run`'s still-live stack snapshot on this
            // same synchronously-faulting thread, and CONTEXT is Copy.
            *context = unsafe { *ctx.recovery_ctx };
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        // An unresolved import can be reached two ways, and both must name it:
        //  * CALLED  — Rip *is* the stub (execution jumped there); or
        //  * READ    — the guest loaded/dereferenced the slot, so Rip is
        //              ordinary code and the *access* address is the stub.
        // Only the first was handled before, which meant a data import left at
        // a stub reported an anonymous `Faulted` at whatever innocent
        // instruction happened to read it — hiding the actual missing symbol.
        // A CALLED unresolved import is a compatibility-inventory event by
        // default, not an immediate process terminator. Return zero and resume
        // at the caller so one 180-second run can expose the complete missing
        // NID surface instead of stopping at the first optional service.
        //
        // `RAEEN_STRICT_NIDS=1` restores the hard-fail behavior for ABI
        // development and acceptance tests. Data READs remain hard failures:
        // unlike a call, there is no return address and no honest value to
        // synthesize for an unresolved imported object.
        static STRICT_NIDS: OnceLock<bool> = OnceLock::new();
        if let Some(s) = stub::resolve(fault_addr, stubs) {
            // SAFETY: `ctx.mem` and `ctx.kernel` are the live process objects
            // installed by `run`; the VEH is synchronous on that same thread.
            let mem = unsafe { &*ctx.mem };
            let kernel = unsafe { &*ctx.kernel };
            let mut ret = [0u8; 8];
            let has_return = mem.read(context.Rsp, &mut ret);
            let return_addr = has_return.then(|| u64::from_le_bytes(ret));
            let caller = return_addr.and_then(|addr| kernel.unwind_module_for_addr(addr));
            let calling_module = caller
                .as_ref()
                .map_or_else(|| "<unknown>".to_owned(), |module| module.name.clone());
            let caller_offset = caller
                .as_ref()
                .zip(return_addr)
                .map(|(module, addr)| addr.saturating_sub(module.start));
            let library = s.library.as_deref().unwrap_or("<unknown>");
            let function = raeen_firmware::dynlib::nid_names::describe(s.nid);
            let (first_occurrence, count) =
                kernel.record_unresolved_nid_call(s.nid, &function, library, &calling_module);
            // The kernel inventory above is process-local and reachable only
            // from a live `OrbisKernel`; the blocker table is what a crash
            // report, the console and the IPC digest read. Counting on every
            // occurrence rather than gating on `first_occurrence` keeps the
            // reported ×N honest, and `record`'s lock is nothing beside the
            // hardware exception, `describe`, and the three `String`s this path
            // already pays per call. The NID is safe as the intern subject: a
            // title's import table is fixed, so the key set is bounded.
            blockers::record(
                BlockerCategory::UnresolvedNid,
                function.clone(),
                s.nid,
                || format!("library={library} caller={calling_module}"),
            );
            let strict =
                *STRICT_NIDS.get_or_init(|| std::env::var_os("RAEEN_STRICT_NIDS").is_some());
            if first_occurrence {
                tracing::warn!(
                    nid = format_args!("{:#018x}", s.nid),
                    library,
                    function,
                    calling_module,
                    caller_offset = caller_offset.map(|offset| format!("{offset:#x}")),
                    count,
                    strict,
                    "UNRESOLVED NID CALLED"
                );
            }
            if !strict && let Some(return_addr) = return_addr {
                context.Rip = return_addr;
                context.Rsp = context.Rsp.wrapping_add(8);
                // Phase 0 deliberately uses the least-assumptive success
                // value. Implemented HLE must replace this before compatibility
                // can be published; the inventory makes every gap visible.
                context.Rax = 0;
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }

        let err = match stub::resolve(fault_addr, stubs).or_else(|| stub::resolve(access, stubs)) {
            Some(s) => RuntimeError::UnimplementedImport {
                nid: s.nid,
                library: s.library.clone(),
                stub_addr: s.addr,
                rip: fault_addr,
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
        // whole guest stack, not merely the innermost callback. Frames come out
        // of the cell before the first (guest-addressed, faultable) completion
        // write — see `ActiveContext::callback_frames`.
        for frame in ctx.take_callback_frames_innermost_first() {
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

    let (result, float_return) = match trampoline::resolve(fault_addr, trampolines) {
        Some(t) => {
            let idx = fault_addr.wrapping_sub(ctx.region_base) / 8;
            let mut return_bytes = [0u8; 8];
            if mem.read(context.Rsp, &mut return_bytes) {
                LAST_HLE_RETURN.store(u64::from_le_bytes(return_bytes), Ordering::Relaxed);
            }
            LAST_HLE_INDEX.store(idx, Ordering::Relaxed);
            HLE_ENTERS.fetch_add(1, Ordering::Relaxed);
            HLE_VEH_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            // Twin of the direct gateway's stamp: whichever dispatch path a
            // title's first import happens to take, `phase=first_hle_call` has
            // to mean the same thing. See the direct site for the cost argument.
            frame_path::record_phase(Phase::FirstHleCall);
            // Yield the diagnostic guest GIL for the duration of this HLE call so
            // a blocking wait (semaphore/mutex/cond) lets another guest thread
            // run; re-acquired when `_hle_yield` drops at the end of this arm,
            // before the guest resumes.
            let _hle_yield = GuestGilYield::during_hle();
            let trace_index = *TRACE_HLE_INDEX.get_or_init(|| {
                std::env::var("RAEEN_TRACE_HLE_INDEX")
                    .ok()
                    .and_then(|value| value.parse().ok())
            });
            let trace_function =
                TRACE_HLE_FUNCTION.get_or_init(|| std::env::var("RAEEN_TRACE_HLE_FUNCTION").ok());
            let trace_this_function = trace_function
                .as_deref()
                .is_some_and(|name| name == t.function);
            if *TRACE_HLE.get_or_init(|| std::env::var_os("RAEEN_TRACE_HLE").is_some())
                || trace_index == Some(idx)
                || trace_this_function
            {
                let read_u64 = |addr| {
                    let mut bytes = [0u8; 8];
                    if mem.read(addr, &mut bytes) {
                        u64::from_le_bytes(bytes)
                    } else {
                        0
                    }
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
                if trace_index == Some(idx) || trace_this_function {
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

            // Native-function detour (diagnostic, RAEEN_TRAP_*): entry/return
            // trampolines that log-and-continue a native guest function so its
            // divergent behavior under native execution can be observed.
            if t.library == crate::native_trap::TRAP_LIBRARY
                && crate::native_trap::handle(
                    context,
                    mem,
                    u64::from(unsafe { GetCurrentThreadId() }),
                )
            {
                return EXCEPTION_CONTINUE_EXECUTION;
            }

            // libSceFiber control transfer (sceFiberRun / Switch / ReturnToThread
            // / GetSelf) rewrites the guest CONTEXT to resume a different fiber
            // (or the thread) executing natively on its own stack — handled here,
            // NOT as a normal HLE call, because it swaps Rip/Rsp/all GPRs. Same
            // "overwrite the delivered CONTEXT + continue" seam as above.
            if crate::fiber::handle(
                &t.function,
                kernel,
                mem,
                context,
                u64::from(unsafe { GetCurrentThreadId() }),
            ) {
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
            // SysV passes float/double arguments in XMM0..XMM7, which never
            // appear in the integer `args` slice. Hand the low half of each to
            // the handler so a function like `sincosf(float, float*, float*)`
            // can read its value instead of guessing it.
            //
            // SAFETY: `context` is the live trap CONTEXT captured by the VEH.
            // `Anonymous` is a union of `FltSave` (XSAVE_FORMAT) and the
            // per-register view; both alias the same bytes, so reading the
            // `XmmRegisters` array is valid however the OS filled it. The
            // floating-point state is always present here — CONTEXT_FLOATING_POINT
            // is part of the CONTEXT_FULL an exception handler receives — and
            // `zip` bounds the read to the 16-entry array. `Low` is a plain
            // `u64` field of a `repr(C)` POD.
            let float_args = {
                let mut xmm = [0u64; 8];
                let saved = unsafe { &context.Anonymous.FltSave.XmmRegisters };
                for (slot, reg) in xmm.iter_mut().zip(saved.iter()) {
                    *slot = reg.Low;
                }
                xmm
            };
            let hle_ctx = HleContext {
                kernel,
                services: kernel,
                gpu: unsafe { &*ctx.gpu },
                mem,
                alloc,
                guest_calls: ctx,
                guest_threads: ctx,
                caller_return_addr,
                caller_rsp: context.Rsp,
                float_args,
                // The interrupted guest thread's full integer register file.
                // The argument slice covers only RDI/RSI/RDX/RCX/R8/R9; an HLE
                // function that must hand the guest a *machine context* — the
                // `ucontext_t` an Orbis exception handler receives — needs the
                // callee-saved set too, above all RBP, which a managed
                // runtime's collector unwinds the suspended thread through.
                caller_gprs: Some(raeen_hle::GuestGpRegs {
                    rdi: context.Rdi,
                    rsi: context.Rsi,
                    rdx: context.Rdx,
                    rcx: context.Rcx,
                    r8: context.R8,
                    r9: context.R9,
                    rax: context.Rax,
                    rbx: context.Rbx,
                    rbp: context.Rbp,
                    r10: context.R10,
                    r11: context.R11,
                    r12: context.R12,
                    r13: context.R13,
                    r14: context.R14,
                    r15: context.R15,
                    rflags: u64::from(context.EFlags),
                    // The guest's TCB. `guest_fsbase` is the value this run
                    // installed; `tls_active` is false on a CPU without
                    // FSGSBASE or a module without `PT_TLS`, and 0 is then the
                    // honest answer rather than a stale base.
                    fsbase: if ctx.tls_active.get() {
                        ctx.guest_fsbase.get()
                    } else {
                        0
                    },
                }),
            };
            ctx.active_hle
                .set(Some((idx, args[..6].try_into().unwrap())));
            // Record this call in the guest thread's recent-call ring so the
            // __cxa_throw trap can report what led to a throw (host threads are
            // pooled; the guest thread id is stable). Gated to the trap run.
            if *TRACE_EINVAL.get_or_init(|| std::env::var_os("RAEEN_TRACE_EINVAL").is_some())
                || stall_instruments_armed()
                || *TRAP_CXA.get_or_init(|| std::env::var_os("RAEEN_TRAP_CXA_THROW").is_some())
            {
                let tid = ctx.current_thread();
                let ring = kernel.recent_hle_calls.entry(tid).or_default();
                let mut q = ring.lock();
                if q.len() >= 24 {
                    q.pop_front();
                }
                q.push_back(format!("{}::{}", t.library, t.function));
            }
            // Where does a stalled thread's wall-clock actually GO? The call
            // ring names which calls a thread made but not how long each took,
            // so a thread parked for minutes inside one wait looks identical to
            // one cycling through thousands of fast calls. Timing each call and
            // accumulating per (thread, function) separates those two.
            let timed = stall_instruments_armed();
            let started = timed.then(std::time::Instant::now);
            // `RAEEN_CALL_STATS`: per-function call counter, split into boot
            // (first 30 s) and steady-state windows. See `CALL_STATS`.
            if *CALL_STATS.get_or_init(|| std::env::var_os("RAEEN_CALL_STATS").is_some()) {
                let counters = kernel
                    .hle_call_counts
                    .entry(format!("{}::{}", t.library, t.function))
                    .or_default();
                let (boot, steady) = counters.value();
                if kernel.uptime() < CALL_STATS_BOOT_WINDOW {
                    boot.fetch_add(1, Ordering::Relaxed);
                } else {
                    steady.fetch_add(1, Ordering::Relaxed);
                }
            }
            // Name the in-flight call BEFORE dispatching it, so a thread that
            // blocks in a host wait deep inside the call (and never returns) can
            // be pinned to the exact function it is parked in.
            if timed {
                kernel.in_flight_hle.insert(
                    ctx.current_thread(),
                    format!("{}::{}", t.library, t.function),
                );
            }
            let ret = hle
                .call(&hle_ctx, &t.library, &t.function, &args)
                .unwrap_or(0);
            // Twin of the direct gateway's site — see `record_orbis_error`.
            // Deliberately independent of the `RAEEN_TRACE_EINVAL` warning
            // further down: that one is opt-in and prints, this one is always
            // on and counts.
            if is_orbis_error(ret) {
                record_orbis_error(idx, trampolines.len(), &t.library, &t.function, ret);
            }
            if timed {
                kernel.in_flight_hle.remove(&ctx.current_thread());
                let micros = started.map_or(0, |s| s.elapsed().as_micros());
                let mut entry = kernel
                    .hle_call_time
                    .entry((
                        ctx.current_thread(),
                        format!("{}::{}", t.library, t.function),
                    ))
                    .or_default();
                entry.0 += 1;
                entry.1 += micros;
            }
            // Diagnostic: surface every EINVAL (22) an HLE call returns, so a
            // std::system_error("invalid argument") thrown by a guest C++
            // threading primitive can be traced to the exact HLE function and
            // arguments that produced it.
            // Any small non-zero errno, or an SCE 0x8002_xxxx error — a PS5
            // title's own libc wrappers can map an unexpected one to EINVAL and
            // throw. `scePthreadGetthreadid`/`scePthreadSelf` legitimately
            // return small thread ids, so exclude them.
            let is_error_return =
                (ret != 0 && ret < 0x100) || (0x8002_0000..0x8003_0000).contains(&ret);
            let is_thread_id_fn =
                t.function.contains("Getthreadid") || t.function.contains("PthreadSelf");
            if is_error_return
                && !is_thread_id_fn
                && *TRACE_EINVAL.get_or_init(|| std::env::var_os("RAEEN_TRACE_EINVAL").is_some())
            {
                tracing::warn!(
                    "errno-return ({ret:#x}) from {}::{} args=[{:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x}] (thread {})",
                    t.library,
                    t.function,
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    ctx.current_thread(),
                );
            }
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
            maybe_arm_guest_watchpoint(context, mem, &t.function, idx);
            // Remember it: if the guest later faults on a null we handed back,
            // this history is the only thing that names the culprit (see
            // `CallTrace`). Index, not name — no allocation in the VEH.
            ctx.trace.push(idx as u32, ret, [args[0], args[1], args[2]]);
            (ret, hle.returns_float(&t.library, &t.function))
        }
        None => {
            // A call landed in the guarded region but names no known
            // trampoline (out of range of this module's table) — record it
            // so `run` surfaces `UnresolvedTrampoline` after the call
            // returns, but still service the call as a 0-returning stub so
            // we can safely resume (design doc §7 step 2's suggested
            // approach) rather than needing an unwind-style abort.
            let linker_visible = raeen_firmware::HLE_TRAMPOLINE_BASE
                .wrapping_add(fault_addr.wrapping_sub(ctx.region_base));
            ctx.error
                .set(Some(RuntimeError::UnresolvedTrampoline(linker_visible)));
            (0, false)
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
            ctx.push_callback_frame(GuestCallbackFrame {
                original_return: u64::from_le_bytes(original_return),
                hle_result: result,
                float_return,
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

    apply_hle_result(context, result, float_return);

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct WatchTestMemory(RefCell<Vec<u8>>);

    impl WatchTestMemory {
        fn new(len: usize) -> Self {
            Self(RefCell::new(vec![0; len]))
        }
    }

    impl GuestMemory for WatchTestMemory {
        fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool {
            let Ok(start) = usize::try_from(guest_addr) else {
                return false;
            };
            let memory = self.0.borrow();
            let Some(end) = start.checked_add(out.len()) else {
                return false;
            };
            let Some(source) = memory.get(start..end) else {
                return false;
            };
            out.copy_from_slice(source);
            true
        }

        fn write(&self, guest_addr: u64, data: &[u8]) -> bool {
            let Ok(start) = usize::try_from(guest_addr) else {
                return false;
            };
            let mut memory = self.0.borrow_mut();
            let Some(end) = start.checked_add(data.len()) else {
                return false;
            };
            let Some(destination) = memory.get_mut(start..end) else {
                return false;
            };
            destination.copy_from_slice(data);
            true
        }
    }

    #[test]
    fn direct_state_teb_slot_round_trips_without_touching_guest_fs() {
        let original = direct_state_slot();
        let marker = 0x5241_4545_4E50_4830;
        set_direct_state_slot(marker);
        assert_eq!(direct_state_slot(), marker);
        set_direct_state_slot(original);
        assert_eq!(direct_state_slot(), original);
    }

    /// The float-return channel on the VEH path: a float-marked HLE result
    /// lands in guest XMM0 (and RAX); an ordinary result touches RAX only.
    /// This is the exact writeback both the main HLE return and the nested
    /// guest-callback return perform.
    #[test]
    fn apply_hle_result_writes_xmm0_only_for_float_returns() {
        const ONE_F64: u64 = 0x3FF0_0000_0000_0000; // 1.0

        // SAFETY: CONTEXT is a plain repr(C) POD (Copy); an all-zero record
        // is a valid starting state for this write-only test, and the union
        // reads below view bytes the helper just wrote.
        let mut context: CONTEXT = unsafe { std::mem::zeroed() };
        apply_hle_result(&mut context, ONE_F64, true);
        assert_eq!(context.Rax, ONE_F64);
        // SAFETY: per the zeroing note above — `Anonymous` aliases `FltSave`
        // with the per-register view over the same live bytes.
        let xmm0 = unsafe { &context.Anonymous.FltSave.XmmRegisters[0] };
        assert_eq!(xmm0.Low, ONE_F64, "the f64 bits must reach guest XMM0");
        assert_eq!(xmm0.High, 0);

        // SAFETY: same POD zeroing reasoning.
        let mut context: CONTEXT = unsafe { std::mem::zeroed() };
        apply_hle_result(&mut context, 7, false);
        assert_eq!(context.Rax, 7);
        // SAFETY: same union-view reasoning.
        let xmm0 = unsafe { &context.Anonymous.FltSave.XmmRegisters[0] };
        assert_eq!(xmm0.Low, 0, "integer results leave guest XMM0 untouched");
    }

    use crate::FaultKind;

    /// The overwhelmingly common case, and the one that must not move: guest
    /// code dereferenced something bad. `rip` reads back through the arena, so
    /// the guest was running.
    #[test]
    fn a_read_fault_in_guest_code_is_a_guest_fault() {
        for kind in [FaultKind::Read, FaultKind::Write, FaultKind::Execute] {
            assert_eq!(
                classify_access_violation(kind, true, false, false, false),
                AvOrigin::Guest,
                "{kind} at a guest-mapped rip is the guest's"
            );
        }
    }

    /// The regression this classification exists to fix: a null dereference
    /// inside a Rust HLE handler used to be reported as `Faulted { addr: <host
    /// rip> }` — "guest fault at 0x7ff…" for an emulator bug.
    #[test]
    fn a_read_or_write_fault_in_host_owned_code_is_a_host_fault() {
        for kind in [FaultKind::Read, FaultKind::Write] {
            assert_eq!(
                classify_access_violation(kind, false, true, false, false),
                AvOrigin::Host,
                "{kind} from a host-owned rip is ours, not the guest's"
            );
        }
    }

    /// A guest `jmp rax` into `argc`, a call through a null function pointer, a
    /// `ret` into data: the *fetch* fails, so `rip` is unmapped by definition and
    /// no host page is involved. `tests/execute.rs` observes `argc` exactly this
    /// way (`Faulted { addr: 1 }`), so this rule is load-bearing — host code
    /// cannot produce this shape, because it is running.
    #[test]
    fn an_instruction_fetch_failure_is_always_a_guest_fault() {
        assert_eq!(
            classify_access_violation(FaultKind::Execute, false, true, false, false),
            AvOrigin::Guest,
            "an execute fault is a wild guest jump even into host-owned space"
        );
        assert_eq!(
            classify_access_violation(FaultKind::Execute, false, false, false, false),
            AvOrigin::Guest,
        );
    }

    /// Both unresolved-import shapes keep their existing reporting: CALLED
    /// (`rip` is the stub) and READ (the access address is). This is the hot
    /// inventory path — a host verdict must never shadow it.
    #[test]
    fn stub_addresses_keep_the_unresolved_import_path() {
        assert_eq!(
            classify_access_violation(FaultKind::Execute, false, false, true, false),
            AvOrigin::UnresolvedStub,
            "a CALLED unresolved import must still name its NID"
        );
        assert_eq!(
            classify_access_violation(FaultKind::Read, false, false, false, true),
            AvOrigin::UnresolvedStub,
            "a READ through a stub-valued relocation must still name its NID"
        );
        // Even when the map calls the rip host-owned — an HLE handler
        // dereferencing a stub-valued pointer the guest handed it still names the
        // missing symbol, which is the actionable half of that report.
        assert_eq!(
            classify_access_violation(FaultKind::Read, false, true, false, true),
            AvOrigin::UnresolvedStub,
        );
    }

    /// Host attribution is a positive claim only. An address the VMA map cannot
    /// answer for, or an access type Windows reports as something other than
    /// read/write/execute (a `#GP` shape), stays on the pre-existing guest path.
    #[test]
    fn unattributed_addresses_stay_on_the_guest_path() {
        assert_eq!(
            classify_access_violation(FaultKind::Read, false, false, false, false),
            AvOrigin::Guest,
            "not guest-mapped and not positively host-owned is not a host verdict"
        );
        assert_eq!(
            classify_access_violation(FaultKind::Other(7), false, true, false, false),
            AvOrigin::Guest,
            "an unrecognized access type must not be reclassified"
        );
    }

    /// An f32 result's bits arrive zero-extended, with the upper lane of
    /// XMM0 cleared (not stale argument state).
    #[test]
    fn apply_hle_result_zero_extends_f32_bits() {
        const ONE_F32_BITS: u64 = 0x3F80_0000; // 1.0f32, zero-extended
        // SAFETY: same POD zeroing reasoning as the sibling test.
        let mut context: CONTEXT = unsafe { std::mem::zeroed() };
        apply_hle_result(&mut context, ONE_F32_BITS, true);
        // SAFETY: same union-view reasoning.
        let xmm0 = unsafe { &context.Anonymous.FltSave.XmmRegisters[0] };
        assert_eq!(xmm0.Low, ONE_F32_BITS);
        assert_eq!(f32::from_bits(xmm0.Low as u32), 1.0);
        assert_eq!(xmm0.High, 0);
    }

    #[test]
    fn eight_byte_write_watchpoint_uses_x64_dr7_encoding() {
        // Fixed bit 10 + L0=1, RW0=01 (write), LEN0=10 (8 bytes).
        assert_eq!(write_watch_dr7(8), Some(0x9_0401));
        assert_eq!(write_watch_dr7(1), Some(0x1_0401));
        assert_eq!(write_watch_dr7(3), None);
    }

    #[test]
    fn watch_address_accepts_prefixed_and_plain_hex() {
        assert_eq!(parse_watch_address("0x1234"), Some(0x1234));
        assert_eq!(parse_watch_address("ABCDEF"), Some(0xAB_CDEF));
        assert_eq!(parse_watch_address("0x"), None);
        assert_eq!(parse_watch_address("not-an-address"), None);
    }

    #[test]
    fn canary_watch_follows_requested_parent_frame_depth() {
        let memory = WatchTestMemory::new(0x400);
        let helper_rbp = 0x100;
        let parent_rbp = 0x200u64;
        assert!(memory.write(helper_rbp, &parent_rbp.to_le_bytes()));

        assert_eq!(
            stack_canary_address(&memory, helper_rbp, 0),
            Some(helper_rbp - STACK_CANARY_SLOT_FROM_RBP)
        );
        assert_eq!(
            stack_canary_address(&memory, helper_rbp, 1),
            Some(parent_rbp - STACK_CANARY_SLOT_FROM_RBP)
        );
        assert_eq!(stack_canary_address(&memory, helper_rbp, 2), None);
    }

    #[test]
    fn watch_frame_depth_is_bounded() {
        assert_eq!(parse_watch_frame_depth("0"), Some(0));
        assert_eq!(parse_watch_frame_depth(" 64 "), Some(64));
        assert_eq!(parse_watch_frame_depth("65"), None);
        assert_eq!(parse_watch_frame_depth("-1"), None);
        assert_eq!(parse_watch_frame_depth("not-a-number"), None);
    }
}
