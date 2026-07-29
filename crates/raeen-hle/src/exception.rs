//! Orbis exception (signal) delivery to a guest-installed handler.
//!
//! `sceKernelInstallExceptionHandler(signum, handler)` registers a process-wide
//! handler; `sceKernelRaiseException(thread, signum)` raises `signum` **at a
//! named thread**. The handler runs with the FreeBSD signal ABI:
//!
//! ```text
//! void handler(int signum, ucontext_t *uctx)
//! ```
//!
//! # Why this exists
//!
//! Registration has been modelled since the first HLE pass, but `Raise` only
//! logged and returned `SCE_OK`. That is not a harmless stub: the one signal
//! titles actually raise is `SIGUSR1` (30), and it is what a managed runtime's
//! stop-the-world collector uses to suspend a thread. Acknowledging without
//! delivering means the collector waits forever for a suspension that never
//! happens. Measured on Subnautica Below Zero: 180 s, 1.4 s of CPU, zero flips,
//! and this exact line as its first blocker —
//!
//! ```text
//! sceKernelRaiseException: guest handler is registered but asynchronous
//! delivery is not implemented; acknowledging target_thread=0x1 signum=30
//! ```
//!
//! # The delivery model: raise queues, the target thread delivers
//!
//! A guest signal handler must run **on the target thread's own stack**, with
//! that thread's TLS and its guest frames below it. The raising thread cannot
//! run it: hijacking another live host worker from outside is precisely the
//! corruption `raeen-runtime`'s cooperative-exit model exists to avoid, and the
//! handler would see the wrong TCB.
//!
//! So `Raise` records a [`raeen_kernel::PendingException`] against the target
//! thread and returns. Every HLE dispatch is a **safe point**: the guest is
//! stopped at a known instruction boundary with its full register file captured
//! ([`crate::HleContext::caller_gprs`]), and `raeen-runtime` can synchronously
//! re-enter guest code from there ([`crate::GuestCallScheduler::call_guest`]).
//! [`deliver_pending`] is called at the end of every dispatch, so the target
//! thread picks up its own signal at its next import call — which for a
//! self-raise is the same call that raised it.
//!
//! This is the same shape SharpEmu arrived at (GPL-2.0; `DirectExecutionBackend`
//! queues a raise for the owning executor to consume "at its next HLE boundary,
//! where the original guest thread is safely paused"), and the layout constants
//! below are cross-checked against shadPS4's `Ucontext`/`Mcontext`
//! (`core/libraries/kernel/threads/exception.h`, GPL-2.0). Re-implemented in
//! Rust; no code copied.
//!
//! # A signal must also interrupt a thread that is already blocked
//!
//! "Delivery at the next HLE call" is not enough, and the shape that proves it
//! is the one this module was written for. Measured on Blasphemous II
//! (Unity/IL2CPP, `docs/host-park-stall.md`): an installed handler for signum 30
//! (SIGUSR1), a raise from `t15`, and all fifteen guest threads parked forever —
//! thirteen inside `sceKernelWaitSema`, the main thread inside
//! `pthread_cond_wait`, and the raiser itself inside `sceKernelWaitSema` waiting
//! for the acknowledgement. **Not one delivery ever happened**, because
//! [`deliver_pending`] ran only *after* an HLE handler returned and none of those
//! handlers ever returns. Even the deferral counter never moved, so
//! [`GATEWAY_STALL_WARNING`] was unreachable and the stall was completely silent.
//!
//! POSIX is the model: a signal **interrupts** a blocking wait. Raeen's blocking
//! waits are already bounded re-check loops, so the fix is a second delivery
//! point — [`deliver_at_wait_slice`] — called by every one of them, once per
//! slice, plus a prompt wake ([`wake_target_for_exception`]) so the latency is
//! the wake rather than the slice.
//!
//! **After the handler returns the wait RESUMES**; it does not return an error.
//! Two independent reasons, both from references that boot Unity titles:
//! shadPS4 installs a title's handler with `SA_RESTART`
//! (`core/libraries/kernel/threads/exception.cpp`, GPL-2.0), which *is* the
//! "restart the interrupted call" flag; and KytyPS5 polls exactly this way —
//! `KernelDispatchPendingSignalForCurrentThread` is called from inside its
//! semaphore and condition wait loops, between `m_mutex.Unlock()` and
//! `m_mutex.Lock()`, after which the loop simply continues
//! (`src/kernel/semaphore.cpp`, `src/libs/libKernel.cpp`; Kyty/MIT lineage).
//! Structure and behaviour studied; re-implemented in Rust, no code copied.
//!
//! The alternative — returning something like `EINTR` — is not available: Orbis
//! `sceKernelWaitSema` has no such error in its set, and handing a guest's own
//! pthread wrapper a code it cannot classify is a failure mode this tree has
//! already measured and fixed once (see `pthread_cond.rs` on POSIX `60` versus
//! SCE `0x8002003C` turning into an uncaught `std::system_error`).
//!
//! # What is delivered, and what is not
//!
//! Delivered: the correct `signum`, a pointer to a real per-thread
//! `ucontext_t`, `mc_rip`/`mc_rsp` naming the interrupted guest instruction and
//! stack, the complete integer register file, `mc_fsbase`, and `mc_len`.
//!
//! **Not** delivered, and named here rather than hidden:
//!
//! * **Timing outside a wait.** A thread spinning in pure guest compute with no
//!   imports and no blocking wait is still only interrupted at its next import.
//!   Hardware would deliver immediately.
//! * **Direct-gateway-only threads.** `raeen-runtime`'s direct leaf gateway
//!   (`trampoline::direct_dispatchable`) reaches its imports by a plain `call`
//!   on a private host stack and therefore *cannot* re-enter guest code, nor
//!   does it capture a machine context ([`HleContext::caller_gprs`] is `None`
//!   there). The blocking waits are consequently **off** that list — a call that
//!   parks the thread is not the per-call-overhead-dominated leaf the list
//!   exists for — but a thread whose only imports are the ones that remain
//!   (`scePthreadMutexLock` and friends) still never reaches a delivering safe
//!   point. The raise is requeued rather than dropped, and after
//!   [`GATEWAY_STALL_WARNING`] consecutive deferrals a single `warn` names the
//!   condition and the `RAEEN_DISABLE_DIRECT_HLE=1` escape hatch, which routes
//!   every import through the VEH path where delivery always works.
//! * **`mc_fpstate`.** Left zeroed with `mc_fpformat = _MC_FPFMT_NODEV` and
//!   `mc_ownedfp = _MC_FPOWNED_NONE`, which is the ABI's way of saying "no FP
//!   state here" — a handler that trusts those fields reads no garbage, but a
//!   handler that needs the XMM file will not find it.
//! * **Segment selectors** (`mc_cs`/`mc_ss`/`mc_ds`/`mc_es`/`mc_fs`/`mc_gs`),
//!   `mc_trapno`, `mc_err`, `mc_addr`: zero. There was no trap, so there is no
//!   honest value.
//! * **Resuming *from* the ucontext.** The handler returns normally to us; a
//!   handler that modifies the context and expects the thread to resume from it
//!   is not supported, and nothing signals that refusal.

use tracing::{debug, info, warn};

use crate::{GuestCallError, HleContext};

/// `sizeof(ucontext_t)` on Orbis/FreeBSD amd64.
pub(crate) const UCONTEXT_SIZE: u64 = 0x500;
/// Byte offset of `uc_mcontext` inside `ucontext_t`: a 16-byte `uc_sigmask`
/// followed by 0x30 bytes of private fields.
///
/// # Measured against the alternatives
///
/// The references disagree: shadPS4 places `uc_mcontext` here at 0x40
/// (`Sigset uc_sigmask; int field1_0x10[12]`), stock FreeBSD amd64 at 0x10, and
/// KytyPS5 at 0x00. All three share the same WITHIN-mcontext offsets, so the
/// choice decides which of our registers a guest finds where it expects
/// `mc_rip`.
///
/// All three were measured against Blasphemous II, whose IL2CPP collector faults
/// after its GC handshake:
///
/// | `UC_MCONTEXT` | where the collector dies |
/// |---|---|
/// | **0x40** (this) | `Il2CppUserAssemblies.prx+0x2b24ab`, reading `0x145b03c0` |
/// | 0x10 | `libc.prx+0x48e0`, reading `0xffff_ffff_ffff_ffff` |
/// | 0x00 | `libc.prx+0x48e0`, reading `0xffff_ffff_ffff_ffff` |
///
/// So the layout is **not** that fault's cause — no offset avoids it — and 0x40
/// is the least bad: the other two move the failure into libc, 0x00 by
/// overwriting `uc_sigmask` outright. Keep 0x40 until a third source settles the
/// real Orbis layout; the collector fault is a separate defect.
pub(crate) const UC_MCONTEXT: u64 = 0x40;
/// `sizeof(mcontext_t)`, the value the ABI requires in `mc_len`.
pub(crate) const MCONTEXT_LEN: u64 = 0x480;

// `mcontext_t` field offsets, relative to [`UC_MCONTEXT`]. Register order is
// FreeBSD's, which is *not* SysV argument order — `mc_rdi` first, `mc_rax`
// after `mc_r9`.
const MC_RDI: u64 = 0x08;
const MC_RSI: u64 = 0x10;
const MC_RDX: u64 = 0x18;
const MC_RCX: u64 = 0x20;
const MC_R8: u64 = 0x28;
const MC_R9: u64 = 0x30;
const MC_RAX: u64 = 0x38;
const MC_RBX: u64 = 0x40;
const MC_RBP: u64 = 0x48;
const MC_R10: u64 = 0x50;
const MC_R11: u64 = 0x58;
const MC_R12: u64 = 0x60;
const MC_R13: u64 = 0x68;
const MC_R14: u64 = 0x70;
const MC_R15: u64 = 0x78;
const MC_RIP: u64 = 0xA0;
const MC_RFLAGS: u64 = 0xB0;
const MC_RSP: u64 = 0xB8;
const MC_LEN: u64 = 0xC8;
const MC_FPFORMAT: u64 = 0xD0;
const MC_OWNEDFP: u64 = 0xD8;
const MC_FSBASE: u64 = 0x440;
const MC_GSBASE: u64 = 0x448;

/// `_MC_FPFMT_NODEV` — "no FP state in this context".
const MC_FPFMT_NODEV: u64 = 0x1_0000;
/// `_MC_FPOWNED_NONE` — "FP state not used".
const MC_FPOWNED_NONE: u64 = 0x2_0000;

/// How many deliveries are logged at `info` before dropping to `debug`.
///
/// A collector that suspends every cycle raises continuously; the first few are
/// the ones that prove delivery works, and the rest would flood the log.
const VERBOSE_DELIVERIES: u64 = 8;
static DELIVERIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Consecutive deliveries refused because the dispatch path could not re-enter
/// guest code. Reset by any successful delivery; see [`GATEWAY_STALL_WARNING`].
static DEFERRALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many consecutive direct-gateway deferrals before saying so out loud.
///
/// A thread whose only imports are direct-dispatchable (`scePthreadMutexLock`,
/// `scePthreadCondWait`, `sceKernelWaitSema` — see `raeen-runtime`'s
/// `trampoline::direct_dispatchable`) never reaches a delivering safe point, so
/// its signal waits forever and the raiser stalls. That is a *silent* stall
/// otherwise, which is the exact failure mode this whole module exists to
/// remove, so it gets a named line with the workaround.
const GATEWAY_STALL_WARNING: u64 = 64;
static GATEWAY_STALL_REPORTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Exceptions delivered from **inside** a blocking wait rather than after a
/// completed HLE call. Diagnostics: a non-zero value is the positive evidence
/// that [`deliver_at_wait_slice`] is doing its job.
static WAIT_DELIVERIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Total exceptions successfully delivered to a guest handler this process.
#[must_use]
pub fn delivered_count() -> u64 {
    DELIVERIES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Prompt wakes issued by [`wake_target_for_exception`]. Diagnostics, and the
/// only race-free way for a test to assert that a raise really did try to wake
/// its target (the wake itself lands in whichever host primitive the target is
/// parked on, which nothing can observe from outside).
static EXCEPTION_WAKES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many of [`delivered_count`] were delivered from inside a blocking wait.
#[must_use]
pub fn wait_delivered_count() -> u64 {
    WAIT_DELIVERIES.load(std::sync::atomic::Ordering::Relaxed)
}

/// How many times a raise woke its target out of a blocking wait. Monotonic; a
/// caller comparing two reads sees a lower bound on the wakes in between.
#[must_use]
pub fn exception_wake_count() -> u64 {
    EXCEPTION_WAKES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the calling guest thread has an exception waiting for it.
///
/// The predicate a blocking wait tests **before** releasing its own
/// notification lock, so the lock is only dropped when there is really something
/// to deliver. One relaxed atomic load in the (overwhelmingly common) case where
/// no title has raised anything; see
/// [`raeen_kernel::OrbisKernel::has_pending_exception_for`].
pub(crate) fn pending_at_wait_slice(ctx: &HleContext) -> bool {
    ctx.kernel
        .has_pending_exception_for(ctx.guest_threads.current_thread())
}

/// **The one place a blocking HLE wait delivers a queued Orbis exception.**
///
/// Every blocking wait in `raeen-hle` calls this once per slice —
/// `kernel_semaphore`, `pthread_cond`, `kernel_eventflag`, `kernel_equeue`, and
/// `libsce_posix`'s `sleep`. Returns whether a guest handler ran.
///
/// # The contract every call site must honour
///
/// **No lock may be held.** This runs guest code on the calling thread via
/// [`crate::GuestCallScheduler::call_guest`], and that guest code is free to
/// call straight back into the HLE — a stop-the-world collector's handler
/// acknowledging a suspension typically calls `sceKernelSignalSema` or
/// `pthread_cond_signal`. If the wait still held the notification lock its own
/// producer takes, that call would deadlock on a non-reentrant host mutex and
/// the "fix" would be strictly worse than the stall it replaced. Wait sites that
/// park under a lock must therefore release it, call this, and re-acquire —
/// which is exactly what KytyPS5 does around its own
/// `KernelDispatchPendingSignalForCurrentThread`.
///
/// Sites gate on [`pending_at_wait_slice`] so the release/re-acquire only
/// happens when there is work.
///
/// # Why this is not a spurious wake
///
/// Nothing here touches the wait's own condition, and the caller resumes waiting
/// afterwards. The wait's real predicate, its deadline, and its FIFO position are
/// all untouched; the only observable difference is that guest code ran on this
/// thread in the middle of the wait — which is precisely what a signal is.
pub(crate) fn deliver_at_wait_slice(ctx: &HleContext) -> bool {
    if !deliver_pending(ctx) {
        return false;
    }
    WAIT_DELIVERIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Wake the target thread out of whatever blocking wait it is parked in, so it
/// reaches [`deliver_at_wait_slice`] now rather than at its next slice.
///
/// Called by `sceKernelRaiseException` **after** the exception is queued: a wake
/// that overtook the queue would find nothing and be wasted. Returns how many
/// condition-variable waiters were interrupted (diagnostics/tests).
///
/// Without this, a signal costs up to one wait slice — 100 ms for
/// `sceKernelWaitSema` — and a collector that suspends every cycle pays that
/// per thread, per cycle. With it the latency is a host unpark.
///
/// Three wake paths, because Raeen's waits park on three different primitives:
///
/// * the [`WaitSubsystem`](raeen_core::subsystems::WaitSubsystem) seam, which
///   covers every wait built on `OrbisKernel::wait_until` (event flags, event
///   queues). Expressed over the trait rather than the concrete kernel so a test
///   can assert the wake with a recording double instead of a second thread.
/// * the semaphore condvars — the process-wide one behind `sceKernelWaitSema`
///   and each live POSIX `sem_t`'s own.
/// * an interrupt — *not* a signal — of the target's `pthread_cond` and futex
///   (`sceKernelSyncOnAddress*`) waiters; see
///   [`raeen_kernel::OrbisKernel::interrupt_cond_waiters_of`] and
///   [`raeen_kernel::SyncAddressTable::interrupt_waiters_of`].
///
/// All of them are safe to invoke when the target is not waiting at all, and none
/// changes any guest-visible condition: a woken wait re-checks its own predicate
/// and parks again if it is still unsatisfied. That is what keeps this from
/// shortening a wait the guest asked for.
///
/// Not covered: `scePthreadMutexLock`'s ownership FIFO, because a mutex wait is
/// not a delivery site (it stays on the direct leaf gateway, which cannot run
/// guest code). Waking it would achieve nothing.
pub(crate) fn wake_target_for_exception(
    waker: &dyn raeen_core::subsystems::WaitSubsystem,
    kernel: &raeen_kernel::OrbisKernel,
    target: u64,
) -> usize {
    EXCEPTION_WAKES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    waker.wake(
        raeen_core::subsystems::WaitKey {
            class: "orbis-exception",
            object: target,
            guest_thread: target,
        },
        raeen_core::subsystems::WakeReason::Signal,
    );
    kernel.notify_semaphore_slices();
    kernel.interrupt_cond_waiters_of(target) + kernel.sync_addresses.interrupt_waiters_of(target)
}

/// The Orbis signals `sceKernelInstallExceptionHandler` accepts: SIGHUP(1),
/// SIGILL(4), SIGFPE(8), SIGBUS(10), SIGSEGV(11), SIGUSR1(30).
///
/// Matches shadPS4's `orbis_allowed_signals` and SharpEmu's `AllowedSignals`.
#[must_use]
pub fn signal_allowed(signum: i32) -> bool {
    matches!(signum, 1 | 4 | 8 | 10 | 11 | 30)
}

/// The guest scratch `ucontext_t` for `thread`, allocated on first use.
///
/// One region per thread, kept for the process lifetime: two threads can be
/// inside their handlers simultaneously, and recycling one buffer would let one
/// delivery rewrite the context another handler is still reading.
fn ucontext_for(ctx: &HleContext, thread: u64) -> Option<u64> {
    if let Some(existing) = ctx.kernel.exception_contexts.get(&thread) {
        return Some(*existing);
    }
    let address = ctx.alloc.alloc(UCONTEXT_SIZE, 16)?;
    // Racing allocations both succeed; the first to publish wins and the loser's
    // block is simply never used again (bounded: at most one per thread).
    Some(
        *ctx.kernel
            .exception_contexts
            .entry(thread)
            .or_insert(address),
    )
}

/// Write the interrupted thread's machine context into the `ucontext_t` at
/// `address`, returning whether every field landed in guest memory.
///
/// Zero-fills first: a handler reading an unmodelled field must see a defined
/// zero, not whatever the allocator's block previously held.
fn write_ucontext(ctx: &HleContext, address: u64, rip: u64, rsp: u64) -> bool {
    let zeros = [0u8; UCONTEXT_SIZE as usize];
    if !ctx.mem.write(address, &zeros) {
        return false;
    }
    let regs = ctx.caller_gprs.unwrap_or_default();
    let mc = address + UC_MCONTEXT;
    let fields = [
        (MC_RDI, regs.rdi),
        (MC_RSI, regs.rsi),
        (MC_RDX, regs.rdx),
        (MC_RCX, regs.rcx),
        (MC_R8, regs.r8),
        (MC_R9, regs.r9),
        (MC_RAX, regs.rax),
        (MC_RBX, regs.rbx),
        (MC_RBP, regs.rbp),
        (MC_R10, regs.r10),
        (MC_R11, regs.r11),
        (MC_R12, regs.r12),
        (MC_R13, regs.r13),
        (MC_R14, regs.r14),
        (MC_R15, regs.r15),
        (MC_RIP, rip),
        (MC_RFLAGS, regs.rflags),
        (MC_RSP, rsp),
        (MC_LEN, MCONTEXT_LEN),
        (MC_FPFORMAT, MC_FPFMT_NODEV),
        (MC_OWNEDFP, MC_FPOWNED_NONE),
        (MC_FSBASE, regs.fsbase),
        (MC_GSBASE, 0),
    ];
    // One-shot: what a guest reading `mc_rip` would get under each candidate
    // `uc_mcontext` offset. Ours is 0x40 (shadPS4's RE'd layout); stock FreeBSD
    // amd64 puts it at 0x10 and KytyPS5 at 0x00, and all three share the same
    // WITHIN-mcontext offsets. So a guest compiled against a different one reads
    // one of our other registers where it expects the instruction pointer — and
    // Blasphemous II's IL2CPP collector dies dereferencing 0x145b03c0, a value
    // with no guest-arena base. This line says whether that value is sitting in
    // R12 or R14, which would identify the layout as the cause.
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!(
            uc = format_args!("{address:#x}"),
            uc_mcontext_offset = format_args!("{UC_MCONTEXT:#x}"),
            real_rip = format_args!("{rip:#x}"),
            real_rsp = format_args!("{rsp:#x}"),
            at_uc_plus_0xa0_is_r12 = format_args!("{:#x}", regs.r12),
            at_uc_plus_0xb0_is_r14 = format_args!("{:#x}", regs.r14),
            rbp = format_args!("{:#x}", regs.rbp),
            rbx = format_args!("{:#x}", regs.rbx),
            "ucontext layout probe: what a guest expecting mcontext at 0x00 or 0x10 \
             would read where it wants mc_rip"
        );
    }
    fields
        .iter()
        .all(|(offset, value)| ctx.mem.write(mc + offset, &value.to_le_bytes()))
}

/// Deliver the exception queued for the calling guest thread, if any.
///
/// Called at the end of every HLE dispatch — the safe point. Returns whether a
/// handler ran.
///
/// The fast path is a single relaxed atomic load
/// ([`raeen_kernel::OrbisKernel::has_pending_exceptions`]); a run in which no
/// title ever raises pays nothing beyond that.
pub(crate) fn deliver_pending(ctx: &HleContext) -> bool {
    if !ctx.kernel.has_pending_exceptions() {
        return false;
    }
    // A thread being torn down must not be re-entered into guest code.
    if ctx.guest_threads.process_is_terminating() {
        return false;
    }
    let thread = ctx.guest_threads.current_thread();
    // `claim` also refuses while this thread is already inside a handler: the
    // handler's own imports are safe points too, and a re-entrant claim would
    // nest deliveries until the guest stack ran out.
    let Some(pending) = ctx.kernel.claim_pending_exception(thread) else {
        return false;
    };

    let Some(uctx) = ucontext_for(ctx, thread) else {
        // Requeued, not dropped: the next safe point may allocate successfully.
        ctx.kernel.requeue_pending_exception(thread, pending);
        warn!(
            thread,
            signum = pending.signum,
            "sceKernelRaiseException delivery deferred: could not allocate a {UCONTEXT_SIZE}-byte \
             guest ucontext for the exception handler"
        );
        return false;
    };

    // `caller_return_addr`/`caller_rsp` describe where the guest is stopped: the
    // instruction it resumes at, and `[rsp]` holding that same address. A real
    // signal frame reports the interrupted RSP, which is one slot above the
    // `call`-pushed return address.
    let rip = ctx.caller_return_addr;
    let rsp = ctx.caller_rsp.wrapping_add(8);
    if !write_ucontext(ctx, uctx, rip, rsp) {
        ctx.kernel.finish_exception_delivery(thread);
        warn!(
            thread,
            signum = pending.signum,
            uctx = format_args!("{uctx:#x}"),
            "sceKernelRaiseException delivery ABANDONED: the guest ucontext scratch is not \
             writable, so the handler cannot be given a machine context. The raise is dropped \
             rather than calling the handler with a garbage pointer."
        );
        return false;
    }

    let delivered = DELIVERIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if delivered < VERBOSE_DELIVERIES {
        info!(
            thread,
            signum = pending.signum,
            handler = format_args!("{:#x}", pending.handler),
            raised_by = pending.raised_by,
            rip = format_args!("{rip:#x}"),
            "delivering sceKernelRaiseException to the guest handler"
        );
    } else {
        debug!(
            thread,
            signum = pending.signum,
            "delivering sceKernelRaiseException to the guest handler"
        );
    }

    // `call_guest` runs the handler on THIS thread's guest stack, below the
    // trapped frame, and returns its RAX. A fault, a `request_exit`, or process
    // termination under the handler unwinds the whole guest call and **never
    // returns here** — so the `finish` below cannot mistake a fatal unwind for a
    // completed handler. The delivering mark dies with the run's context in that
    // case, which is correct: the thread is gone.
    let result = ctx.guest_calls.call_guest(
        pending.handler,
        [pending.signum.unsigned_abs().into(), uctx, 0, 0, 0, 0],
    );
    ctx.kernel.finish_exception_delivery(thread);

    match result {
        Ok(_) => {
            DEFERRALS.store(0, std::sync::atomic::Ordering::Relaxed);
            true
        }
        Err(GuestCallError::NullEntry) => {
            warn!(
                thread,
                signum = pending.signum,
                "sceKernelRaiseException delivery REFUSED: the installed handler is a null \
                 pointer. Dropped rather than jumped to."
            );
            false
        }
        Err(GuestCallError::Unsupported) => {
            // The direct leaf gateway cannot re-enter guest code. Put it back:
            // the next import on the VEH path — which is every import that can
            // reach guest code — delivers it.
            ctx.kernel.requeue_pending_exception(thread, pending);
            let deferrals = DEFERRALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if deferrals >= GATEWAY_STALL_WARNING
                && !GATEWAY_STALL_REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                warn!(
                    thread,
                    signum = pending.signum,
                    deferrals,
                    "sceKernelRaiseException delivery is STALLED: this thread keeps reaching only \
                     direct-gateway imports, which cannot re-enter guest code, so its exception \
                     handler never runs and whatever raised the signal is waiting on it. Set \
                     RAEEN_DISABLE_DIRECT_HLE=1 to route every import through the VEH path, where \
                     delivery always works."
                );
            } else {
                debug!(
                    thread,
                    signum = pending.signum,
                    "sceKernelRaiseException delivery deferred past the direct leaf gateway"
                );
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestCallRequest, GuestCallScheduler, GuestMemory, GuestThreadScheduler};
    use raeen_kernel::PendingException;

    /// A `GuestCallScheduler` that records the callback it was asked to make
    /// instead of running guest code, so delivery is testable without a native
    /// runtime. `outcome` selects what `call_guest` reports back.
    struct RecordingCalls {
        calls: std::cell::RefCell<Vec<(u64, [u64; 6])>>,
        outcome: Result<u64, GuestCallError>,
    }

    impl RecordingCalls {
        fn ok() -> Self {
            Self {
                calls: std::cell::RefCell::new(Vec::new()),
                outcome: Ok(0),
            }
        }

        fn failing(error: GuestCallError) -> Self {
            Self {
                calls: std::cell::RefCell::new(Vec::new()),
                outcome: Err(error),
            }
        }
    }

    impl GuestCallScheduler for RecordingCalls {
        fn request(&self, _request: GuestCallRequest) -> bool {
            false
        }

        fn call_guest(&self, entry: u64, args: [u64; 6]) -> Result<u64, GuestCallError> {
            self.calls.borrow_mut().push((entry, args));
            self.outcome
        }
    }

    /// A scheduler double that reports a caller-chosen current thread, so a
    /// cross-thread raise (the real shape: a collector thread raising at the
    /// main thread) can be distinguished from a self-raise.
    struct ThreadIs(u64);

    impl GuestThreadScheduler for ThreadIs {
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
            self.0
        }
        fn request_process_exit(&self, _code: u64) {}
        fn process_is_terminating(&self) -> bool {
            false
        }
    }

    fn read_u64(mem: &crate::TestMemory, at: u64) -> u64 {
        let mut buf = [0u8; 8];
        assert!(mem.read(at, &mut buf), "ucontext field at {at:#x} readable");
        u64::from_le_bytes(buf)
    }

    /// The whole point: a queued raise must actually call the registered guest
    /// handler, with the FreeBSD signal ABI (`signum`, `ucontext*`).
    #[test]
    fn a_queued_raise_calls_the_registered_guest_handler() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        // Keep the ucontext away from address 0 so a null-vs-allocated mistake
        // cannot pass by accident.
        let alloc = crate::TestAllocator::new(0x1000);
        let calls = RecordingCalls::ok();
        let ctx = crate::HleContext {
            caller_return_addr: 0x1234_5678,
            caller_rsp: 0x2000,
            caller_gprs: Some(crate::GuestGpRegs {
                rbp: 0xBBBB_0000,
                r15: 0xFFFF_0001,
                fsbase: 0x7F00_0000,
                ..Default::default()
            }),
            ..crate::test_ctx_with_guest_calls(&kernel, &mem, &alloc, &calls)
        };

        kernel.queue_pending_exception(
            1,
            PendingException {
                signum: 30,
                handler: 0xDEAD_BEEF,
                raised_by: 4,
            },
        );
        assert!(kernel.has_pending_exceptions());

        assert!(
            deliver_pending(&ctx),
            "a raise queued for the current thread must be delivered at the safe point"
        );

        let calls = calls.calls.borrow();
        assert_eq!(calls.len(), 1, "exactly one handler invocation");
        let (entry, args) = calls[0];
        assert_eq!(entry, 0xDEAD_BEEF, "the installed handler must be called");
        assert_eq!(args[0], 30, "arg0 is the Orbis signal number");
        let uctx = args[1];
        assert_ne!(uctx, 0, "arg1 must be a real ucontext pointer");

        // The machine context the handler receives must describe the
        // *interrupted* guest thread, not zeros.
        let mc = uctx + UC_MCONTEXT;
        assert_eq!(read_u64(&mem, mc + MC_RIP), 0x1234_5678);
        assert_eq!(
            read_u64(&mem, mc + MC_RSP),
            0x2008,
            "mc_rsp must be the interrupted RSP — one slot above the pushed return address"
        );
        assert_eq!(read_u64(&mem, mc + MC_RBP), 0xBBBB_0000);
        assert_eq!(read_u64(&mem, mc + MC_R15), 0xFFFF_0001);
        assert_eq!(read_u64(&mem, mc + MC_FSBASE), 0x7F00_0000);
        assert_eq!(
            read_u64(&mem, mc + MC_LEN),
            MCONTEXT_LEN,
            "mc_len must be sizeof(mcontext_t) or a handler cannot version-check it"
        );
        assert_eq!(read_u64(&mem, mc + MC_FPFORMAT), MC_FPFMT_NODEV);
        assert_eq!(read_u64(&mem, mc + MC_OWNEDFP), MC_FPOWNED_NONE);

        assert!(
            !kernel.has_pending_exceptions(),
            "a delivered exception must be consumed"
        );
        assert!(
            !kernel.exception_delivery_active.contains_key(&1),
            "the delivering mark must be released once the handler returns"
        );
    }

    /// Delivery is per-thread: thread 1's queued signal must not be handed to
    /// thread 2 just because thread 2 reached a safe point first.
    #[test]
    fn a_raise_is_only_delivered_on_its_target_thread() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x1000);
        let calls = RecordingCalls::ok();
        let other = ThreadIs(2);
        let ctx = crate::HleContext {
            guest_threads: &other,
            ..crate::test_ctx_with_guest_calls(&kernel, &mem, &alloc, &calls)
        };

        kernel.queue_pending_exception(
            1,
            PendingException {
                signum: 30,
                handler: 0xDEAD_BEEF,
                raised_by: 2,
            },
        );

        assert!(
            !deliver_pending(&ctx),
            "thread 2 must not consume thread 1's signal"
        );
        assert!(calls.calls.borrow().is_empty());
        assert!(
            kernel.has_pending_exceptions(),
            "the target thread's signal must still be waiting for it"
        );
    }

    /// A safe point reached *inside* a handler must not start a second
    /// delivery — that recursion is bounded only by guest stack space.
    #[test]
    fn delivery_does_not_recurse_while_a_handler_is_running() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x1000);
        let calls = RecordingCalls::ok();
        let ctx = crate::HleContext {
            ..crate::test_ctx_with_guest_calls(&kernel, &mem, &alloc, &calls)
        };

        // Model "already inside the handler for thread 1".
        kernel.exception_delivery_active.insert(1, ());
        kernel.queue_pending_exception(
            1,
            PendingException {
                signum: 30,
                handler: 0xDEAD_BEEF,
                raised_by: 4,
            },
        );

        assert!(!deliver_pending(&ctx), "a nested claim must be refused");
        assert!(calls.calls.borrow().is_empty());
        assert!(
            kernel.has_pending_exceptions(),
            "the refused signal stays queued for after the handler returns"
        );
    }

    /// The direct leaf gateway cannot re-enter guest code. A raise must survive
    /// that dispatch path rather than being silently consumed by it.
    #[test]
    fn an_undeliverable_dispatch_path_requeues_instead_of_dropping() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x1000);
        let calls = RecordingCalls::failing(GuestCallError::Unsupported);
        let ctx = crate::test_ctx_with_guest_calls(&kernel, &mem, &alloc, &calls);

        kernel.queue_pending_exception(
            1,
            PendingException {
                signum: 30,
                handler: 0xDEAD_BEEF,
                raised_by: 4,
            },
        );

        assert!(!deliver_pending(&ctx));
        assert!(
            kernel.has_pending_exceptions(),
            "a path that cannot call guest code must leave the raise for one that can"
        );
        assert!(
            !kernel.exception_delivery_active.contains_key(&1),
            "the delivering mark must not be left set on a deferred delivery"
        );
    }

    /// A null handler must be refused by name, never jumped to.
    #[test]
    fn a_null_handler_is_refused_not_jumped_to() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x1000);
        let calls = RecordingCalls::failing(GuestCallError::NullEntry);
        let ctx = crate::test_ctx_with_guest_calls(&kernel, &mem, &alloc, &calls);

        kernel.queue_pending_exception(
            1,
            PendingException {
                signum: 30,
                handler: 0,
                raised_by: 4,
            },
        );

        assert!(!deliver_pending(&ctx));
        assert!(
            !kernel.has_pending_exceptions(),
            "an unjumpable handler is dropped, not retried forever"
        );
    }

    /// The zero-pending fast path must not touch the map at all, and must not
    /// invent a delivery.
    #[test]
    fn nothing_is_delivered_when_nothing_was_raised() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x100);
        let calls = RecordingCalls::ok();
        let ctx = crate::test_ctx_with_guest_calls(&kernel, &mem, &alloc, &calls);

        assert!(!kernel.has_pending_exceptions());
        assert!(!deliver_pending(&ctx));
        assert!(calls.calls.borrow().is_empty());
    }

    /// A thread that exits with an undelivered signal must not leave the
    /// per-call fast path permanently disarmed.
    #[test]
    fn a_dead_threads_signal_is_discarded_and_rearms_the_fast_path() {
        let kernel = raeen_kernel::OrbisKernel::new();
        kernel.queue_pending_exception(
            7,
            PendingException {
                signum: 30,
                handler: 0xDEAD_BEEF,
                raised_by: 1,
            },
        );
        assert!(kernel.has_pending_exceptions());
        assert!(kernel.discard_pending_exception(7));
        assert!(
            !kernel.has_pending_exceptions(),
            "a dead thread's entry must be removed, or every later HLE call pays a map lookup"
        );
        assert!(!kernel.discard_pending_exception(7), "idempotent");
    }

    /// Newest wins: a collector that raises again before the previous signal was
    /// picked up must not build a backlog.
    #[test]
    fn a_second_raise_replaces_an_undelivered_one() {
        let kernel = raeen_kernel::OrbisKernel::new();
        assert!(!kernel.queue_pending_exception(
            1,
            PendingException {
                signum: 30,
                handler: 0x1000,
                raised_by: 4,
            }
        ));
        assert!(
            kernel.queue_pending_exception(
                1,
                PendingException {
                    signum: 30,
                    handler: 0x2000,
                    raised_by: 5,
                }
            ),
            "the second raise must report that it replaced an undelivered one"
        );
        assert_eq!(
            kernel.claim_pending_exception(1).map(|p| p.handler),
            Some(0x2000)
        );
    }

    /// Queueing an exception must **wake** the target rather than let it sit out
    /// its wait slice — up to 100 ms for `sceKernelWaitSema`. Asserted through the
    /// injectable wait seam with a recording double, so promptness is a recorded
    /// wake with the right key and reason rather than a measured duration.
    #[test]
    fn a_queued_raise_wakes_the_target_out_of_its_wait() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let recorder = crate::host_vblank::RecordingWaker::default();
        kernel.queue_pending_exception(
            5,
            PendingException {
                signum: 30,
                handler: 0x1234,
                raised_by: 9,
            },
        );

        let interrupted = wake_target_for_exception(&recorder, &kernel, 5);

        let wakes = recorder.wakes();
        assert_eq!(wakes.len(), 1, "exactly one wait-subsystem wake");
        assert_eq!(wakes[0].0.class, "orbis-exception");
        assert_eq!(wakes[0].0.object, 5, "the wake must name the TARGET thread");
        assert_eq!(
            wakes[0].1,
            raeen_core::subsystems::WakeReason::Signal,
            "the reason must read as a signal in a diagnostic trace, not as a queue event"
        );
        assert_eq!(
            interrupted, 0,
            "no condition waiter exists, so none is interrupted"
        );
        assert!(
            kernel.has_pending_exception_for(5),
            "waking must not consume the queued exception — the target still has to claim it"
        );
        assert!(
            !kernel.has_pending_exception_for(1),
            "and no other thread may see it"
        );
    }

    /// The in-wait chokepoint and the post-dispatch one share a body, but the
    /// in-wait deliveries must be separately countable: a run whose only evidence
    /// is `delivered_count` cannot distinguish "signals worked" from "signals
    /// worked only for threads that were not blocked", which is exactly the
    /// distinction the Blasphemous II stall turned on.
    #[test]
    fn an_in_wait_delivery_is_counted_as_one() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x1000);
        let calls = RecordingCalls::ok();
        let ctx = crate::test_ctx_with_guest_calls(&kernel, &mem, &alloc, &calls);

        let before = wait_delivered_count();
        assert!(!pending_at_wait_slice(&ctx), "nothing queued yet");
        assert!(!deliver_at_wait_slice(&ctx));
        assert_eq!(
            wait_delivered_count(),
            before,
            "a chokepoint call with nothing pending must not be counted as a delivery"
        );

        kernel.queue_pending_exception(
            1,
            PendingException {
                signum: 30,
                handler: 0xDEAD_BEEF,
                raised_by: 4,
            },
        );
        assert!(pending_at_wait_slice(&ctx));
        assert!(deliver_at_wait_slice(&ctx));
        assert_eq!(calls.calls.borrow().len(), 1, "the handler ran");
        assert!(
            wait_delivered_count() > before,
            "an in-wait delivery must be attributed to the in-wait chokepoint"
        );
    }

    /// The layout constants are load-bearing ABI. Pin them against the
    /// published FreeBSD amd64 `ucontext_t`/`mcontext_t` shape so a future edit
    /// cannot quietly shift a register by eight bytes.
    #[test]
    fn ucontext_layout_matches_the_freebsd_amd64_abi() {
        assert_eq!(UC_MCONTEXT, 0x40, "uc_sigmask[16] + 0x30 private bytes");
        assert_eq!(MCONTEXT_LEN, 0x480);
        // mcontext_t: mc_onstack then the register block in FreeBSD's order.
        assert_eq!(
            [
                MC_RDI, MC_RSI, MC_RDX, MC_RCX, MC_R8, MC_R9, MC_RAX, MC_RBX, MC_RBP, MC_R10,
                MC_R11, MC_R12, MC_R13, MC_R14, MC_R15,
            ],
            [
                0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70,
                0x78,
            ]
        );
        // mc_fpstate[104] starts at 0x100 and runs to 0x440, where the bases sit.
        assert_eq!(MC_FSBASE, 0x100 + 104 * 8);
        assert_eq!(MC_GSBASE, MC_FSBASE + 8);
        // uc_mcontext must fit inside the ucontext.
        const { assert!(UC_MCONTEXT + MCONTEXT_LEN <= UCONTEXT_SIZE) };
    }

    /// Only the signals the real kernel accepts may be installed — a title that
    /// probes an unsupported signal must get EINVAL, not a handler slot.
    #[test]
    fn only_the_orbis_allowed_signals_are_accepted() {
        for signum in [1, 4, 8, 10, 11, 30] {
            assert!(signal_allowed(signum), "signal {signum} is allowed");
        }
        for signum in [0, 2, 3, 5, 6, 7, 9, 29, 31, 32, 127] {
            assert!(!signal_allowed(signum), "signal {signum} is not allowed");
        }
    }
}
