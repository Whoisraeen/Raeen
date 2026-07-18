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
// Conflicts with our MSRV: `is_multiple_of` is stable since 1.87, we target 1.85.
#![allow(clippy::manual_is_multiple_of)]

#[cfg(target_os = "windows")]
mod arena;
#[cfg(target_os = "windows")]
mod dispatch;
mod fiber;
#[cfg(target_os = "windows")]
mod process;
#[cfg(target_os = "windows")]
mod stack;
#[cfg(target_os = "windows")]
mod stub;
#[cfg(target_os = "windows")]
mod thread;
/// Diagnostic: sample every guest thread's RIP (see [`thread::sample_guest_rips`]).
/// Windows-only, like the rest of the execution core.
#[cfg(target_os = "windows")]
pub use thread::sample_guest_rips;
/// Diagnostic: shallow HOST backtrace per guest thread, symbolized to
/// `module+offset` — names where a stalled thread is parked in our code / ntdll.
#[cfg(target_os = "windows")]
pub use thread::{host_module_for_addr, sample_host_backtraces};
#[cfg(target_os = "windows")]
mod tls;
#[cfg(target_os = "windows")]
mod trampoline;
// Deliberately not `cfg(windows)`: the address map is pure bookkeeping with no
// host calls, so it builds and tests everywhere. `pub` because the kernel/HLE
// memory calls are its consumers.
pub mod vmm;

/// Diagnostic: how many times the VEH has re-armed a guest FS base that
/// Windows discarded at a context switch (see `dispatch::fsbase_rearm_count`).
/// Windows-only, like the rest of the execution core.
#[cfg(target_os = "windows")]
pub use dispatch::fsbase_rearm_count;

use thiserror::Error;
use xps5x_firmware::LinkedModule;
use xps5x_hle::{GuestMemory, HleRegistry};
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

/// The base of the static TLS **area** sitting immediately below `tcb`
/// (variant-II x86-64: every module's block grows downward from the thread
/// pointer), or `None` when nothing in the process has static TLS.
///
/// This is the low end of the storage the linker's `TPOFF64` offsets resolve
/// into, and the base `__tls_get_addr` resolves static TLS module ids against
/// (via the kernel's per-module area offsets) — the ELF TLS ABI requires a
/// thread-local reached through the general-dynamic model to land on the same
/// storage as through initial-exec. Handing out a second block instead is a
/// measured Minecraft crash: its single `.tdata` pointer read back as `NULL`
/// from the uninitialized copy.
fn static_tls_block(
    tcb: Option<u64>,
    tls_layout: &[xps5x_firmware::StaticTlsModule],
) -> Option<u64> {
    let total = xps5x_firmware::static_tls_total(tls_layout);
    match tcb {
        Some(tcb) if total > 0 => tcb.checked_sub(total),
        _ => None,
    }
}

/// Publish the process's static TLS layout to the kernel, translating each
/// module's `tp_offset` (distance below the thread pointer) into its offset
/// from the area's LOW end — the form `__tls_get_addr` adds to
/// `current_static_tls_block()`.
fn register_static_tls_layout(
    kernel: &OrbisKernel,
    tls_layout: &[xps5x_firmware::StaticTlsModule],
) {
    let total = xps5x_firmware::static_tls_total(tls_layout);
    kernel.set_static_tls_area_offsets(
        tls_layout
            .iter()
            .map(|m| (m.module_id, total - m.tp_offset)),
    );
}

/// How a faulting guest instruction was touching the address it faulted on —
/// decoded from `EXCEPTION_RECORD.ExceptionInformation[0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    Read,
    Write,
    /// A DEP violation: the guest tried to *execute* the address. This is what
    /// a call through an unresolved import slot looks like.
    Execute,
    /// Windows reported a code this doesn't know; the raw value is carried so
    /// the report stays honest rather than guessing.
    Other(u64),
}

impl std::fmt::Display for FaultKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FaultKind::Read => f.write_str("read"),
            FaultKind::Write => f.write_str("write"),
            FaultKind::Execute => f.write_str("execute"),
            FaultKind::Other(v) => write!(f, "access-type-{v}"),
        }
    }
}

impl RuntimeError {
    /// Decode `EXCEPTION_RECORD.ExceptionInformation[0]` into a [`FaultKind`].
    pub(crate) fn fault_kind(v: u64) -> FaultKind {
        match v {
            0 => FaultKind::Read,
            1 => FaultKind::Write,
            8 => FaultKind::Execute,
            other => FaultKind::Other(other),
        }
    }
}

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
    ///
    /// `addr` is the faulting instruction's `Rip` — *where the guest was*.
    /// `access` is what it was **touching** when it faulted, which is usually
    /// the part that identifies the bug: `Rip` alone says a module is unhappy,
    /// while the access address says *why* (a null deref, a wild pointer, or —
    /// very commonly here — a read through a relocation slot left pointing at
    /// [`xps5x_firmware::UNRESOLVED_STUB_BASE`]). Windows hands both to the VEH
    /// in `EXCEPTION_RECORD`; this used to discard the access address.
    #[error("guest fault at {addr:#x} ({kind} {access:#x})")]
    Faulted {
        addr: u64,
        /// The virtual address the faulting instruction tried to touch
        /// (`ExceptionInformation[1]`).
        access: u64,
        /// How it was touching it — "read", "write", or "execute"
        /// (`ExceptionInformation[0]`: 0, 1, and 8 respectively).
        kind: FaultKind,
    },
    /// The guest **called an import nothing implements**: execution reached a
    /// per-NID unresolved stub (`UNRESOLVED_STUB_BASE + i * 8`), which the
    /// linker wrote into that symbol's relocation slots.
    ///
    /// This is the single most actionable outcome the runtime produces. It
    /// used to surface as `Faulted { addr: 0x5000_0000_0000 }` — every missing
    /// import shared one address, so the fault could not say *which* function
    /// the guest wanted. `nid` names it; map it through
    /// [`xps5x_firmware::LinkedModule::unresolved_stubs`] for the library.
    #[error("guest called unimplemented import nid {nid:#018x} (stub {addr:#x})")]
    UnimplementedImport { nid: u64, addr: u64 },
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
    let guest_rsp = arena
        .stack_top()
        .checked_sub(8)
        .ok_or(RuntimeError::MapFailed)?;
    if !arena.write(guest_rsp, &guard.return_trampoline().to_le_bytes()) {
        return Err(RuntimeError::MapFailed);
    }

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
        arena.setup_main_tcb(&module.tls_layout)
    } else {
        None
    };
    let static_tls_block = static_tls_block(tcb, &module.tls_layout);
    register_static_tls_layout(kernel, &module.tls_layout);

    // SAFETY: `entry_ptr` is a host address inside `arena`'s
    // `PAGE_EXECUTE_READWRITE` image sub-region, at the caller-specified
    // `entry_offset` into `module.image` — code the LM1 pipeline produced,
    // and the only thing this crate ever executes (design doc §6).
    // Transmuting a data pointer to an `extern "sysv64"` function pointer
    // matches the guest ABI (design doc §3); actually calling it happens
    // inside `dispatch::run`, guarded by the VEH armed there.
    let entry = entry_ptr as u64;

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
            &module.unresolved_stubs,
            hle,
            kernel,
            &arena,
            &arena,
            &guard,
            tcb,
            static_tls_block,
            None,
            1,
            // No inner `unsafe {}` here: this closure literal is written
            // directly inside the `unsafe { dispatch::run(...) }` block
            // below, so it's already inside that unsafe scope (rustc flags
            // a nested one as `unused_unsafe`) — the SAFETY justification
            // for this call is the comment on that outer block.
            || crate::stack::enter_guest(entry, guest_rsp, padded),
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
    let _call_lock = dispatch::call_lock();
    let arena = arena::GuestArena::new(&module.image)?;
    let guard = trampoline::TrampolineGuard::reserve(module.hle_trampolines.len())?;
    execute_process_mapped(module, hle, kernel, &arena, &guard, None, argv, envp)
}

/// Execute a process whose complete runtime state is Arc-owned. This is the
/// M1-E/C2 entry used by real titles: workers may retain clones after
/// `scePthreadCreate` without borrowing launcher stack frames or allowing the
/// arena/trampoline guard to be unmapped underneath them.
#[cfg(target_os = "windows")]
pub fn execute_process_shared(
    module: std::sync::Arc<LinkedModule>,
    hle: std::sync::Arc<HleRegistry>,
    kernel: std::sync::Arc<OrbisKernel>,
    argv: &[&str],
    envp: &[&str],
) -> Result<RunOutcome, RuntimeError> {
    let _call_lock = dispatch::call_lock();
    let arena = std::sync::Arc::new(arena::GuestArena::new(&module.image)?);
    let guard = std::sync::Arc::new(trampoline::TrampolineGuard::reserve(
        module.hle_trampolines.len(),
    )?);
    let process = thread::GuestProcess::create(module, hle, kernel, arena, guard);
    // The main guest thread is id 1 and runs on THIS host thread — it never goes
    // through `GuestProcess::create`'s spawn path, so record its handle here or
    // the RIP sampler would be blind to the one thread that drives boot.
    thread::record_host_thread_handle(&process.kernel, 1);
    let result = execute_process_mapped(
        &process.module,
        &process.hle,
        &process.kernel,
        &process.arena,
        &process.guard,
        Some(&process),
        argv,
        envp,
    );
    let process_exit_was_requested = process.requested_exit_code().is_some();
    let fallback_code = match &result {
        Ok(RunOutcome::Exited(code)) => *code,
        Ok(RunOutcome::Returned(_)) => 0,
        Err(_) => 1,
    };
    process.terminate_and_reap(fallback_code);
    if process_exit_was_requested {
        Ok(RunOutcome::Exited(
            process.requested_exit_code().unwrap_or(fallback_code),
        ))
    } else {
        result
    }
}

#[cfg(target_os = "windows")]
#[allow(clippy::too_many_arguments)]
fn execute_process_mapped(
    module: &LinkedModule,
    hle: &HleRegistry,
    kernel: &OrbisKernel,
    arena: &arena::GuestArena,
    guard: &trampoline::TrampolineGuard,
    guest_threads: Option<&dyn xps5x_hle::GuestThreadScheduler>,
    argv: &[&str],
    envp: &[&str],
) -> Result<RunOutcome, RuntimeError> {
    let unwind_modules = module
        .unwind_modules
        .iter()
        .filter_map(|loaded| {
            let module_base = GUEST_ARENA_BASE.checked_add(loaded.image_offset)?;
            let rebase = |vaddr: u64| {
                if vaddr == 0 {
                    Some(0)
                } else {
                    module_base.checked_add(vaddr)
                }
            };
            let start = module_base.checked_add(loaded.unwind.image_vaddr)?;
            Some(xps5x_kernel::UnwindModuleInfo {
                name: loaded.name.clone(),
                start,
                end: start.checked_add(loaded.unwind.image_size)?,
                eh_frame_hdr_addr: rebase(loaded.unwind.eh_frame_hdr_vaddr)?,
                eh_frame_addr: rebase(loaded.unwind.eh_frame_vaddr)?,
                eh_frame_size: loaded.unwind.eh_frame_size,
                seg0_addr: module_base.checked_add(loaded.unwind.seg0_vaddr)?,
                seg0_size: loaded.unwind.seg0_size,
            })
        })
        .collect();
    kernel.set_unwind_modules(unwind_modules);
    for loaded in &module.unwind_modules {
        let Some(module_base) = GUEST_ARENA_BASE.checked_add(loaded.image_offset) else {
            continue;
        };
        let Some(load_start) = module_base.checked_add(loaded.unwind.image_vaddr) else {
            continue;
        };
        let exports = loaded.exports.iter().filter_map(|export| {
            module_base
                .checked_add(export.value)
                .map(|addr| (export.nid, addr))
        });
        let initialized = loaded.image_offset == 0
            || module
                .module_inits
                .iter()
                .any(|init| init.name == loaded.name);
        kernel.register_lle_module(
            loaded.name.clone(),
            load_start,
            loaded.unwind.image_size,
            loaded
                .init_vaddr
                .and_then(|init| module_base.checked_add(init)),
            initialized,
            exports,
        );
    }
    kernel.set_proc_param_addr(
        module
            .procparam_offset
            .map_or(0, |off| GUEST_ARENA_BASE + off),
    );
    let entry_ptr = arena.entry_ptr(module.entry)?;

    // RT2c-b (design doc §3): same honest-degradation TCB setup as
    // `execute_linked` above, including the module's `PT_TLS` template and
    // the fs:0x28 canary (M1-B).
    let tcb = if tls::fsgsbase_available() {
        arena.setup_main_tcb(&module.tls_layout)
    } else {
        None
    };
    let static_tls_block = static_tls_block(tcb, &module.tls_layout);
    register_static_tls_layout(kernel, &module.tls_layout);

    // Lay out the process stack in the arena's stack region (design doc §2);
    // `process::build_process_stack` writes only through `&arena` (bounds-
    // checked `GuestMemory`), never panicking on `argv`/`envp` content.
    let process_rsp = process::build_process_stack(arena.stack_top(), argv, envp, arena)?;
    // Dependency initializers are ordinary called functions. Give them a
    // return slot below the process-parameter block; the guarded address in
    // that slot lets dispatch recover the host context without retaining a
    // host RSP anywhere outside the captured CONTEXT.
    let module_rsp = process_rsp.checked_sub(16).ok_or(RuntimeError::MapFailed)?;
    if !arena.write(module_rsp, &guard.return_trampoline().to_le_bytes()) {
        return Err(RuntimeError::MapFailed);
    }

    // SAFETY: same reasoning as `execute_linked`'s `entry` transmute above —
    // `entry_ptr` is a host address inside `arena`'s `PAGE_EXECUTE_READWRITE`
    // image sub-region, at `module.entry` (the ELF `e_entry`), which is the
    // only thing this crate ever executes (design doc §6). This function
    // pointer is never actually *called* through this Rust type — the entry is
    // reached by a bare `jmp` — but `entry_ptr` (a `*const u8`) needs some
    // callable type to become the address `enter_guest_at_start`'s asm jumps
    // to, and the same fn-pointer type is reused here for consistency with
    // `execute_linked`. The Orbis `_start(params, exit_fn)` register arguments
    // are supplied explicitly below; see `stack::enter_guest_at_start`'s doc.
    let entry = entry_ptr as u64;
    let exit_fn = dispatch::process_exit_trampoline(&module.hle_trampolines).unwrap_or_else(|| {
        tracing::warn!(
            "process imports no terminating HLE function; Orbis _start exit_fn remains null"
        );
        0
    });

    // SAFETY: `entry` is exactly the function pointer
    // `enter_guest_at_start`'s safety contract requires; `process_rsp` is the
    // 8-mod-16 Orbis process-stack pointer `build_process_stack` just computed,
    // inside `arena`'s own committed, writable stack region — satisfying
    // `dispatch::run`'s `call_guest` contract. The remaining arguments carry
    // the same safety argument as `execute_linked`'s call to `dispatch::run`
    // above.
    // Diagnostic: XPS5X_SKIP_MAIN_INIT skips the MAIN module's module_start
    // call (the LAST entry in `module_inits`, after the dependencies'). A title
    // whose crt0 `_start` runs the same init array again gets ctors run TWICE —
    // measured on ASTRO.BOT: a list-adding ctor then builds a cyclic list its
    // own walk spins on (t1 frozen at a `mov rdx,[rcx]; lea rcx,[rdx+0x10];
    // jnz` cycle). Skipping the loader's call leaves exactly one run (crt0's).
    let skip_main_init = std::env::var_os("XPS5X_SKIP_MAIN_INIT").is_some();
    let init_count = module.module_inits.len();
    for (i, init) in module.module_inits.iter().enumerate() {
        if skip_main_init && i + 1 == init_count {
            tracing::warn!(
                "{}: XPS5X_SKIP_MAIN_INIT — loader's module_start skipped (crt0 will run it)",
                init.name
            );
            continue;
        }
        let Ok(ptr) = arena.entry_ptr(init.image_offset) else {
            tracing::warn!(
                "{}: module_start at +{:#x} is outside the image — skipping",
                init.name,
                init.image_offset
            );
            continue;
        };
        tracing::info!(
            "{}: calling module_start (+{:#x})",
            init.name,
            init.image_offset
        );
        // SAFETY: `ptr` is executable guest code in the live arena;
        // `module_rsp` points at the guarded return address and dispatch arms
        // recovery before `enter_guest` transfers control.
        let outcome = unsafe {
            dispatch::run(
                &module.hle_trampolines,
                &module.unresolved_stubs,
                hle,
                kernel,
                arena,
                arena,
                guard,
                tcb,
                static_tls_block,
                guest_threads,
                1,
                || crate::stack::enter_guest(ptr as u64, module_rsp, [0; 6]),
            )
        }?;
        match outcome {
            RunOutcome::Returned(rc) => {
                tracing::info!("{}: module_start returned {rc:#x}", init.name);
            }
            RunOutcome::Exited(code) => return Ok(RunOutcome::Exited(code)),
        }
    }

    // Process mode deliberately keeps the parameter block at RSP for the
    // legacy stack-shaped entry fixture while also passing it in RDI per the
    // real Orbis ABI. A malformed `_start` return still faults through argc
    // and is recovered as a genuine guest fault.
    unsafe {
        dispatch::run(
            &module.hle_trampolines,
            &module.unresolved_stubs,
            hle,
            kernel,
            arena,
            arena,
            guard,
            tcb,
            static_tls_block,
            guest_threads,
            1,
            // Same "no inner `unsafe {}`" note as `execute_linked`'s closure
            // above — already inside the outer `unsafe { dispatch::run(...)
            // }` block's scope.
            // Orbis `_start(params /* rdi */, exit_fn /* rsi */)`: `params`
            // points at the `argc, argv[], NULL, envp[], NULL, auxv` block
            // `build_process_stack` just wrote (which is exactly what
            // `process_rsp` addresses), NOT at the stack in the Linux sense —
            // a real title does `mov r14d,[rdi]` / `lea r15,[rdi+8]` and would
            // null-deref with rdi=0. `exit_fn` is a real terminating HLE
            // trampoline. Retail crt0 preserves RSI and calls through it when
            // startup terminates; the VEH turns that call into the normal
            // process-exit longjmp.
            || crate::stack::enter_guest(entry, process_rsp, [process_rsp, exit_fn, 0, 0, 0, 0]),
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
