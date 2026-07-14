//! # XPS5X Runtime — RT0
//!
//! Executes a [`xps5x_firmware::LinkedModule`] natively: maps its `image`
//! into host memory, guards the HLE trampoline region the LM1 linker
//! addressed relocation slots against, arms a Vectored Exception Handler,
//! and calls the guest entry function directly on the host thread as a
//! foreign `extern "sysv64"` function pointer. See the design doc
//! (`docs/superpowers/specs/2026-07-13-xps5x-runtime-design.md`) for the
//! full mechanism, ABI boundary, and safety/trust-boundary discussion.
//!
//! RT0 is Windows-first (design doc §7/§9): [`execute_linked`]'s mechanism
//! (`mem`/`trampoline`/`dispatch`) is Win32-API-specific and gated
//! `#[cfg(target_os = "windows")]`, but the public function signature is
//! platform-independent so a POSIX `sigaction`/`SIGSEGV` backend can slot in
//! at a later milestone without callers changing.
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "windows")]
mod arena;
#[cfg(target_os = "windows")]
mod dispatch;
#[cfg(target_os = "windows")]
mod process;
#[cfg(target_os = "windows")]
mod stack;
#[cfg(target_os = "windows")]
mod tls;
#[cfg(target_os = "windows")]
mod trampoline;

use thiserror::Error;
use xps5x_firmware::LinkedModule;
use xps5x_hle::HleRegistry;
use xps5x_kernel::OrbisKernel;

/// How a guest call ended (design doc §3/§4, wall #1): [`execute_linked`]'s
/// function-mode calls only ever produce `Returned` (mapped straight to its
/// `Ok(u64)` return), but [`execute_process`]'s `_start` entries can also end
/// via `exit`/`exit_group`/`_exit` (or `sceKernelExit`), which surfaces as
/// `Exited` instead of a normal return — `_start` never returns to its
/// caller in a well-formed program, so `Exited` is the expected, honest way
/// a process-mode run ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOutcome {
    /// The guest entry point returned normally; the value is its `RAX`.
    Returned(u64),
    /// The guest called a terminating function (`exit`-family); the value
    /// is the exit code passed to it (SysV arg 0, i.e. `Rdi`).
    Exited(u64),
}

/// The guest address space's fixed base (design doc §2/§3): a 4 GiB host
/// region reserved at this exact address by [`arena::GuestArena`] (RT2 Task
/// 2), identity-mapped so guest address `A` is host address `A`. High and
/// normally free, clear of the trampoline guard at `0x4000_0000_0000` and
/// the unresolved-stub sentinel at `0x5000_0000_0000`.
///
/// This is the single source of truth for the link base: the LM1 linker must
/// link a module so guest vaddr `0` lands here, and `xps5x-gui`'s
/// `FirmwareLauncher` passes this as the load base (RT2 Task 3). Exported
/// unconditionally (not `cfg(windows)`-gated) since it is a pure constant —
/// only [`arena::GuestArena`]'s reservation mechanism is Windows-specific.
pub const GUEST_ARENA_BASE: u64 = 0x0000_1000_0000_0000;

/// Errors [`execute_linked`] can return (design doc §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RuntimeError {
    /// Guest memory (the image mapping or the trampoline guard region)
    /// could not be established, or `entry_offset` did not point within the
    /// mapped image.
    #[error("failed to map guest memory")]
    MapFailed,
    /// A guest `call` hit a trampoline slot with no corresponding
    /// [`xps5x_firmware::HleTrampoline`] entry — surfaced, not silently
    /// ignored (design doc §5). The faulting address is reported for
    /// diagnostics.
    #[error("call to unresolved HLE trampoline at {0:#x}")]
    UnresolvedTrampoline(u64),
    /// A genuine guest fault (an access violation) outside the trampoline
    /// guard region — e.g. a wild pointer dereference in guest code, not an
    /// HLE call. Recovered rather than crashing the process (RT1a): the VEH
    /// restores a pre-call register snapshot taken via `RtlCaptureContext`
    /// (see `dispatch.rs`'s module doc comment for the exact mechanism).
    /// `addr` is the faulting instruction's `Rip`.
    #[error("guest fault at {addr:#x}")]
    Faulted { addr: u64 },
    /// More than 6 integer/pointer arguments were requested — RT0 only
    /// marshals the SysV integer argument registers (design doc §3).
    #[error("more than 6 arguments requested (RT0 marshals only the SysV integer registers)")]
    TooManyArgs,
}

/// Integer/pointer arguments RT0 marshals: SysV RDI, RSI, RDX, RCX, R8, R9
/// (design doc §3).
const MAX_ARGS: usize = 6;

/// Run `module`'s function at `entry_offset` (an offset into
/// `module.image`) natively, passing `args` (up to 6 integer/pointer
/// values, SysV) and servicing every HLE trampoline call it makes through
/// `hle`. Each serviced call gets an [`xps5x_hle::HleContext`] built from
/// `kernel` and a real, identity-mapped [`arena::GuestArena`] (design doc
/// §2/§5's dispatch-context milestone), passed as both the
/// [`xps5x_hle::GuestMemory`] and [`xps5x_hle::GuestAllocator`] views — so
/// HLE functions can touch the kernel, read/write guest memory, and allocate
/// real guest heap/mmap memory, not just log. Returns the guest function's
/// `RAX` on success. See the design doc §2 for the full trap-and-dispatch
/// mechanism and §5 for this signature.
///
/// **Requirement:** `module` must have been linked with `link_module`'s
/// `base` set to [`GUEST_ARENA_BASE`] — the arena always maps `module.image`
/// at `GUEST_ARENA_BASE` (identity: guest address `A` is host address `A`),
/// so any `R_X86_64_RELATIVE` relocation baked in at a different base would
/// resolve to the wrong host address. `entry_offset` is still a plain offset
/// into `module.image` (host addr = `GUEST_ARENA_BASE + entry_offset`).
#[cfg(target_os = "windows")]
pub fn execute_linked(
    module: &LinkedModule,
    hle: &HleRegistry,
    kernel: &OrbisKernel,
    entry_offset: u64,
    args: &[u64],
) -> Result<u64, RuntimeError> {
    if args.len() > MAX_ARGS {
        return Err(RuntimeError::TooManyArgs);
    }
    let mut padded = [0u64; MAX_ARGS];
    padded[..args.len()].copy_from_slice(args);

    // RT0 supports exactly one active native guest execution at a time
    // (design doc §4/§6/§9); held for this entire function, not just the
    // guest call inside `dispatch::run` below — see `dispatch::CALL_LOCK`'s
    // doc comment for why the trampoline guard reservation just below also
    // needs this. It also serializes construction of the fixed-base
    // `GuestArena` below, which only one caller may hold at a time (design
    // doc §2/§9, `arena::GuestArena`'s doc comment).
    let _call_lock = dispatch::call_lock();

    let arena = arena::GuestArena::new(&module.image)?;
    // Expose the module's PT_SCE_PROCPARAM block (if any) to the guest via
    // `sceKernelGetProcParam`: its guest address is the arena base plus the
    // segment's image offset (identity-mapped). `0` clears any stale value
    // from a prior run.
    kernel.set_proc_param_addr(
        module
            .procparam_offset
            .map_or(0, |off| GUEST_ARENA_BASE + off),
    );
    let entry_ptr = arena.entry_ptr(entry_offset)?;
    let guard = trampoline::TrampolineGuard::reserve(module.hle_trampolines.len())?;

    // RT2c-b (design doc §3): set up a minimal main-thread TCB so the guest
    // can use `fs:`-relative TLS, but only when FSGSBASE is actually
    // available — on a CPU without it, `tcb` stays `None` and `dispatch::run`
    // never executes an fsbase instruction (honest degradation, not a
    // fragile half-working `fsbase`). `setup_main_tcb` allocates from the
    // same arena the guest otherwise uses, so this fails closed (`None`)
    // rather than panicking if the heap allocation itself fails. The
    // module's `PT_TLS` template (if any) is materialized below the TCB
    // (M1-B) so TLS-relocated accesses resolve against real init data.
    let tcb = if tls::fsgsbase_available() {
        arena.setup_main_tcb(module.tls.as_ref())
    } else {
        None
    };

    // SAFETY: `entry_ptr` is a host address inside `arena`'s
    // `PAGE_EXECUTE_READWRITE` image sub-region, at the caller-specified
    // `entry_offset` into `module.image` — code the LM1 pipeline produced,
    // and the only thing this crate ever executes (design doc §6).
    // Transmuting a data pointer to an `extern "sysv64"` function pointer
    // matches the guest ABI (design doc §3); actually calling it happens
    // inside `dispatch::run`, guarded by the VEH armed there.
    let entry: unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64 =
        unsafe { core::mem::transmute::<*const u8, _>(entry_ptr) };

    // `arena` doubles as both the guest-memory view and the guest allocator
    // HLE calls get: it is identity-mapped (guest address `A` is host
    // address `A`) and implements both `GuestMemory` and `GuestAllocator`
    // (see `arena.rs`).
    //
    // SAFETY: `entry` is exactly the function pointer
    // `call_on_guest_stack`'s safety contract requires (a valid `sysv64`
    // pointer into the `GuestArena` we just built), called on
    // `arena.stack_top()` (the 16-aligned top of `arena`'s own committed,
    // writable stack region, RT2c-a, design doc §2/§4) — satisfying
    // `dispatch::run`'s `call_guest` contract. `module.hle_trampolines`,
    // `hle`, `kernel`, and `arena` (as both `&dyn GuestMemory` and `&dyn
    // GuestAllocator`) all outlive this call (borrowed for its entire
    // duration); `guard`'s region covers every address
    // `module.hle_trampolines` can resolve (it was sized from that same
    // table, immediately above).
    let outcome = unsafe {
        dispatch::run(
            &module.hle_trampolines,
            hle,
            kernel,
            &arena,
            &arena,
            &guard,
            tcb,
            // No inner `unsafe {}` here: this closure literal is written
            // directly inside the `unsafe { dispatch::run(...) }` block
            // below, so it's already inside that unsafe scope (rustc flags
            // a nested one as `unused_unsafe`) — the SAFETY justification
            // for this call is the comment on that outer block.
            || crate::stack::call_on_guest_stack(entry, padded, arena.stack_top()),
        )
    }?;
    // Function-mode callers only ever care about the value: a bare `exit()`
    // call from a synthetic stub is unusual but harmless here (design doc
    // §4) — it surfaces as `Exited(code)`, which is treated exactly like an
    // ordinary return of `code`.
    Ok(match outcome {
        RunOutcome::Returned(v) | RunOutcome::Exited(v) => v,
    })
}

/// RT0 is Windows-first; a POSIX backend lands at a later milestone without
/// changing this signature (design doc §7/§9).
#[cfg(not(target_os = "windows"))]
pub fn execute_linked(
    _module: &LinkedModule,
    _hle: &HleRegistry,
    _kernel: &OrbisKernel,
    _entry_offset: u64,
    _args: &[u64],
) -> Result<u64, RuntimeError> {
    Err(RuntimeError::MapFailed)
}

/// Run `module` as a real ELF process (design doc §2/§3, wall #1): build the
/// initial `argc`/`argv`/`envp`/`auxv` process stack (§2) and enter
/// `module.entry` as `_start` — no pushed return address, so the guest's
/// first instruction sees `rsp` pointing at `argc` — servicing HLE trampoline
/// calls exactly as [`execute_linked`] does. A well-formed `_start` never
/// returns; it ends the program via `exit`/`exit_group`/`_exit`
/// (`RunOutcome::Exited`, §4). A malformed `_start` that returns anyway, or
/// that faults, is recovered as `Err(Faulted)` (RT1a) rather than crashing.
///
/// **Requirement:** same as [`execute_linked`] — `module` must have been
/// linked with `link_module`'s `base` set to [`GUEST_ARENA_BASE`].
#[cfg(target_os = "windows")]
pub fn execute_process(
    module: &LinkedModule,
    hle: &HleRegistry,
    kernel: &OrbisKernel,
    argv: &[&str],
    envp: &[&str],
) -> Result<RunOutcome, RuntimeError> {
    // RT0 single-active-execution invariant (design doc §4/§6/§9) — see
    // `dispatch::CALL_LOCK`'s doc comment.
    let _call_lock = dispatch::call_lock();

    let arena = arena::GuestArena::new(&module.image)?;
    // Expose the module's PT_SCE_PROCPARAM block (if any) to the guest via
    // `sceKernelGetProcParam`: its guest address is the arena base plus the
    // segment's image offset (identity-mapped). `0` clears any stale value
    // from a prior run.
    kernel.set_proc_param_addr(
        module
            .procparam_offset
            .map_or(0, |off| GUEST_ARENA_BASE + off),
    );
    let entry_ptr = arena.entry_ptr(module.entry)?;
    let guard = trampoline::TrampolineGuard::reserve(module.hle_trampolines.len())?;

    // RT2c-b (design doc §3): same honest-degradation TCB setup as
    // `execute_linked` above, including the module's `PT_TLS` template and
    // the fs:0x28 canary (M1-B).
    let tcb = if tls::fsgsbase_available() {
        arena.setup_main_tcb(module.tls.as_ref())
    } else {
        None
    };

    // Lay out the process stack in the arena's stack region (design doc §2);
    // `process::build_process_stack` writes only through `&arena` (bounds-
    // checked `GuestMemory`), never panicking on `argv`/`envp` content.
    let process_rsp = process::build_process_stack(arena.stack_top(), argv, envp, &arena)?;

    // SAFETY: same reasoning as `execute_linked`'s `entry` transmute above —
    // `entry_ptr` is a host address inside `arena`'s `PAGE_EXECUTE_READWRITE`
    // image sub-region, at `module.entry` (the ELF `e_entry`), which is the
    // only thing this crate ever executes (design doc §6). This function
    // pointer is never actually *called* with these six registers as
    // arguments (a `_start` entry ignores them and reads `argc`/`argv`/
    // `envp` off the stack instead) — see `stack::enter_guest_at_start`'s doc
    // comment — but the same fn-pointer type is reused here for consistency
    // with `execute_linked`, since `entry_ptr` (a `*const u8`) needs some
    // callable type to become the address `enter_guest_at_start`'s asm jumps
    // to.
    let entry: unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64 =
        unsafe { core::mem::transmute::<*const u8, _>(entry_ptr) };

    // SAFETY: `entry` is exactly the function pointer
    // `enter_guest_at_start`'s safety contract requires; `process_rsp` is the
    // 16-aligned process-stack pointer `build_process_stack` just computed,
    // inside `arena`'s own committed, writable stack region — satisfying
    // `dispatch::run`'s `call_guest` contract. The remaining arguments carry
    // the same safety argument as `execute_linked`'s call to `dispatch::run`
    // above.
    unsafe {
        dispatch::run(
            &module.hle_trampolines,
            hle,
            kernel,
            &arena,
            &arena,
            &guard,
            tcb,
            // Same "no inner `unsafe {}`" note as `execute_linked`'s closure
            // above — already inside the outer `unsafe { dispatch::run(...)
            // }` block's scope.
            || crate::stack::enter_guest_at_start(entry, process_rsp),
        )
    }
}

/// RT0 is Windows-first; a POSIX backend lands at a later milestone without
/// changing this signature (design doc §7/§9).
#[cfg(not(target_os = "windows"))]
pub fn execute_process(
    _module: &LinkedModule,
    _hle: &HleRegistry,
    _kernel: &OrbisKernel,
    _argv: &[&str],
    _envp: &[&str],
) -> Result<RunOutcome, RuntimeError> {
    Err(RuntimeError::MapFailed)
}
