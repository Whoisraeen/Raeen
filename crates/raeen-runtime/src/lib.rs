//! # Raeen Runtime — RT0
//!
//! Executes a [`raeen_firmware::LinkedModule`] natively: maps its `image`
//! into host memory, guards the HLE trampoline region the LM1 linker
//! addressed relocation slots against, arms a Vectored Exception Handler,
//! and calls the guest entry function directly on the host thread as a
//! foreign `extern "sysv64"` function pointer. See the design doc
//! (`docs/superpowers/specs/2026-07-13-raeen-runtime-design.md`) for the
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
/// Diagnostic: one-shot `int3` traps on a module's exports
/// (`RAEEN_TRAP_MODULE_EXPORTS`). Pure bookkeeping — builds and tests
/// everywhere; the VEH glue is in `dispatch`.
pub mod export_trap;
mod fiber;
pub mod native_trap;
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
#[cfg(target_os = "windows")]
pub use thread::{GuestProcess, GuestProcessHandle, GuestProcessSnapshot};
/// Diagnostic: shallow HOST backtrace per guest thread, symbolized to
/// `module+offset` — names where a stalled thread is parked in our code / ntdll.
/// [`thread::host_wait_primitive`] classifies the innermost frame, which is what
/// lets the stall monitor count a thread parked outside any HLE call.
#[cfg(target_os = "windows")]
pub use thread::{
    HostThreadSample, host_module_for_addr, host_wait_primitive, sample_host_backtraces,
};
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
pub use dispatch::{
    HleDispatchMetrics, fsbase_rearm_count, guest_watchpoint_hit_count, hle_dispatch_metrics,
};

use raeen_firmware::LinkedModule;
#[cfg(target_os = "windows")]
use raeen_firmware::ModuleInitRole;
use raeen_hle::{GuestMemory, HleRegistry};
use raeen_kernel::OrbisKernel;
use thiserror::Error;

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

/// Who runs the **main executable's** `DT_INIT` initializer — the decision that
/// makes a real title boot instead of double-constructing its globals.
///
/// A retail crt0 `_start` walks the executable's own init array, so a loader
/// that *also* calls the main initializer runs those constructors twice.
/// Measured on ASTRO.BOT, a list-building constructor then formed a cyclic list
/// its own later walk spun on forever (t1 frozen at `module+0x7426c00`). The
/// two runtime entry points therefore differ:
///
/// * [`CrtOwnsMainInit`](EntryPolicy::CrtOwnsMainInit) — [`execute_process`]:
///   a genuine crt0 `_start` is entered, and it runs the main initializer
///   itself, so the loader runs **only the dependency** initializers.
/// * [`LoaderOwnsMainInit`](EntryPolicy::LoaderOwnsMainInit) — [`execute_linked`]
///   (direct function/module execution): no crt0 is entered, so nothing else
///   will run the main initializer; the loader runs **every** initializer,
///   main included. Function-mode fixtures carry no initializers, so this is a
///   no-op for them, but a directly-run module with constructors gets them.
///
/// Internal to the runtime: the two public entry points ([`execute_process`],
/// [`execute_linked`]) each fix their own policy, so this never appears in a
/// public signature.
#[cfg(target_os = "windows")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryPolicy {
    /// The guest crt0 `_start` owns the main initializer; the loader withholds it.
    CrtOwnsMainInit,
    /// No crt0 runs; the loader owns every initializer, main included.
    LoaderOwnsMainInit,
}

/// Whether the loader itself invokes an initializer with `role` under `policy`.
///
/// The whole truth table: the loader runs every initializer **except** the main
/// executable's own under [`EntryPolicy::CrtOwnsMainInit`], which the crt0
/// `_start` runs instead (running it here too is the ASTRO.BOT double-init
/// hang; see [`EntryPolicy`]).
#[cfg(target_os = "windows")]
fn loader_runs_initializer(policy: EntryPolicy, role: ModuleInitRole) -> bool {
    !matches!(
        (policy, role),
        (EntryPolicy::CrtOwnsMainInit, ModuleInitRole::Main)
    )
}

/// `RAEEN_SKIP_MAIN_INIT` used to be required to reach the proven boot path;
/// deferring the main initializer to crt0 is now the default, so the variable
/// is deprecated. Warn once (per process entry) if it is still set so a stale
/// launcher configuration is visible rather than silently ignored.
#[cfg(target_os = "windows")]
fn warn_if_deprecated_skip_main_init_set() {
    if std::env::var_os("RAEEN_SKIP_MAIN_INIT").is_some() {
        tracing::warn!(
            "RAEEN_SKIP_MAIN_INIT is set but no longer needed — deferring the main \
             executable's initializer to crt0 is the default (EntryPolicy::CrtOwnsMainInit). \
             The variable is deprecated and ignored; remove it from your launch environment."
        );
    }
}

/// The guest address space's fixed base (design doc §2/§3): a 4 GiB host
/// region reserved at this exact address by [`arena::GuestArena`] (RT2 Task
/// 2), identity-mapped so guest address `A` is host address `A`. High and
/// normally free, clear of the trampoline guard at `0x4000_0000_0000` and
/// the unresolved-stub sentinel at `0x5000_0000_0000`.
///
/// This is the single source of truth for the link base: the LM1 linker must
/// link a module so guest vaddr `0` lands here, and `raeen-gui`'s
/// `FirmwareLauncher` passes this as the load base (RT2 Task 3). Exported
/// unconditionally (not `cfg(windows)`-gated) since it is a pure constant —
/// only [`arena::GuestArena`]'s reservation mechanism is Windows-specific.
pub const GUEST_ARENA_BASE: u64 = 0x0000_1000_0000_0000;

/// What [`reserve_title_va_window`] claimed at process startup, for the caller
/// to log once logging exists (the call itself runs *before* logging setup —
/// even a tracing subscriber allocates, and beating every host allocation is
/// the whole point).
#[derive(Debug, Clone, Default)]
pub struct TitleVaWindowReport {
    /// The defended windows as `(inclusive start, exclusive end)`. A title maps
    /// direct memory at more than one literal base, so more than one range has
    /// to be held for it.
    pub windows: Vec<(u64, u64)>,
    /// How many disjoint free holes were claimed as `MEM_RESERVE` blocks.
    pub reserved_blocks: usize,
    /// Total bytes of address space now reserved for guest fixed-address maps.
    pub reserved_bytes: u64,
    /// Human-readable descriptions of host regions that were ALREADY inside
    /// the window when the reservation ran — each one is address space a guest
    /// fixed map can no longer be served at. Empty when the call ran early
    /// enough, which is the invariant the Shell maintains.
    pub squatters: Vec<String>,
}

/// Claim the guest title fixed-VA window before the host process can allocate
/// into it. See `arena::reserve_title_va_window` for the mechanism and the
/// measured Shell-vs-CLI failure it closes. Idempotent; call it as the first
/// statement of `main`.
#[cfg(target_os = "windows")]
pub use arena::reserve_title_va_window;

/// Non-Windows: no native runtime, nothing to defend. Kept so callers
/// (`raeen-gui`'s `main`) stay `cfg`-free.
#[cfg(not(target_os = "windows"))]
pub fn reserve_title_va_window() -> &'static TitleVaWindowReport {
    static EMPTY: TitleVaWindowReport = TitleVaWindowReport {
        windows: Vec::new(),
        reserved_blocks: 0,
        reserved_bytes: 0,
        squatters: Vec::new(),
    };
    &EMPTY
}

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
    tls_layout: &[raeen_firmware::StaticTlsModule],
) -> Option<u64> {
    let total = raeen_firmware::static_tls_total(tls_layout);
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
    tls_layout: &[raeen_firmware::StaticTlsModule],
) {
    let total = raeen_firmware::static_tls_total(tls_layout);
    kernel.set_static_tls_area_offsets(
        tls_layout
            .iter()
            .map(|m| (m.module_id, total - m.tp_offset)),
    );
}

/// The guest thread id both `execute_linked` and `execute_process` hand
/// `dispatch::run` for the initial thread.
#[cfg(target_os = "windows")]
const MAIN_GUEST_THREAD: u64 = 1;

/// Publish the main thread's guest stack bounds so the HLE out-buffer guard can
/// recognise a caller local.
///
/// The HLE needs "is this out-pointer inside the calling thread's stack?" to be
/// answerable *exactly*, because an oversized out-struct write onto a caller
/// frame is what smashes the caller's adjacent locals and its
/// `__stack_chk_guard` canary — the crash GTA V and Until Dawn both die of.
/// Registering the region costs one map insert per launch and removes the need
/// for any address-range heuristic; `raeen_runtime::thread` does the same for
/// each `scePthreadCreate` worker's own stack.
#[cfg(target_os = "windows")]
fn register_main_thread_stack(kernel: &OrbisKernel, arena: &arena::GuestArena) {
    let (base, top) = arena.stack_region();
    kernel
        .guest_thread_stacks
        .insert(MAIN_GUEST_THREAD, (base, top));
}

/// Publish this process's HLE trampoline table to the kernel, keyed by function
/// name, so `sceKernelDlsym` can answer with a guest-callable address.
///
/// Imports never need this — the linker already wrote each import's trampoline
/// address into its relocation slot. `dlsym` is the sole caller that has to turn
/// a *name* into an address at run time, with no relocation to consult, and the
/// trampoline table is the only place a name-to-address mapping exists.
///
/// Must run before guest code does. It is cheap (one map insert per distinct
/// import) and idempotent, so every process entry point calls it unconditionally
/// rather than trying to decide whether this title uses `dlsym`.
/// Bind this process's guest address space to the VFS so the `memory:`
/// pseudo-file scheme can serve reads out of it.
///
/// Scaleform/GFx (GTA V's UI layer) hands the file layer a pointer+length URI
/// instead of a path — `memory:$<hex addr>,<len>,<flags>:<name>` — which is not
/// a host path and must never be walked as one. `raeen-kernel` sits below this
/// crate and cannot name [`arena::GuestArena`], so the VFS declares a read-only
/// seam and the runtime supplies the arena here.
///
/// The VFS keeps only a `Weak`, so this does not extend the arena past the end of
/// the run: when the process finishes and the arena drops, `memory:` opens go
/// back to a named, counted refusal on their own.
#[cfg(target_os = "windows")]
fn bind_memory_scheme_address_space(
    kernel: &OrbisKernel,
    arena: &std::sync::Arc<arena::GuestArena>,
) {
    let source: std::sync::Arc<dyn raeen_kernel::filesystem::GuestByteSource> =
        std::sync::Arc::clone(arena) as _;
    kernel.filesystem.set_guest_byte_source(&source);
}

#[cfg(target_os = "windows")]
fn publish_hle_exports_for_dlsym(
    kernel: &OrbisKernel,
    trampolines: &[raeen_firmware::HleTrampoline],
) {
    for trampoline in trampolines {
        kernel.register_hle_export_addr(&trampoline.function, trampoline.addr);
    }
    tracing::debug!(
        "published {} HLE trampoline(s) to sceKernelDlsym",
        kernel.hle_export_addr_count()
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

/// Which x86 divide-error (`#DE`) condition Windows reported for a faulting
/// `div`/`idiv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivideFault {
    /// `EXCEPTION_INT_DIVIDE_BY_ZERO` (`0xC000_0094`) — the divisor was zero.
    ByZero,
    /// `EXCEPTION_INT_OVERFLOW` (`0xC000_0095`) — an `idiv` whose quotient does
    /// not fit the destination register (`INT_MIN / -1`). A different cause but
    /// the *same instruction and the same trap*, so it is classified here rather
    /// than left to fall through unreported.
    Overflow,
}

impl std::fmt::Display for DivideFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DivideFault::ByZero => f.write_str("divide by zero"),
            DivideFault::Overflow => f.write_str("quotient overflow"),
        }
    }
}

/// Whose code was executing when a fault was delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultOrigin {
    /// The faulting instruction is in the guest arena — the title's own code.
    Guest,
    /// The faulting instruction is below the arena — emulator (or dependency)
    /// code. Our bug.
    Host,
}

impl std::fmt::Display for FaultOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FaultOrigin::Guest => f.write_str("guest"),
            FaultOrigin::Host => f.write_str("HOST"),
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
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RuntimeError {
    /// Guest memory (the image mapping or the trampoline guard region)
    /// could not be established, or `entry_offset` did not point within the
    /// mapped image.
    #[error("failed to map guest memory")]
    MapFailed,
    /// A guest `call` hit a trampoline slot with no corresponding
    /// [`raeen_firmware::HleTrampoline`] entry — surfaced, not silently
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
    /// [`raeen_firmware::UNRESOLVED_STUB_BASE`]). Windows hands both to the VEH
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
    /// [`raeen_firmware::LinkedModule::unresolved_stubs`] for the library.
    #[error(
        "guest called unimplemented import nid {nid:#018x} from {library:?} (stub {stub_addr:#x}, rip {rip:#x})"
    )]
    UnimplementedImport {
        nid: u64,
        library: Option<String>,
        stub_addr: u64,
        rip: u64,
    },
    /// An access violation that happened while **host** (emulator) code was
    /// running — not guest code. Our bug, not the title's.
    ///
    /// The VEH used to report every access violation on a guest-executing
    /// thread as [`Self::Faulted`], including one raised by a null dereference
    /// inside a Rust HLE handler (or a driver called from one). That laundered
    /// an emulator bug into "guest fault at 0x7ff…", which reads as a title
    /// problem and sends the investigation to the wrong place entirely.
    ///
    /// Classification (see `dispatch::classify_access_violation`): the faulting
    /// `rip` is not guest-readable memory, is not an unresolved-import stub, and
    /// the arena's VMA map positively attributes it to the host process
    /// ([`crate::vmm::VmaType::Foreign`]) — plus the access is a read or write,
    /// never an instruction fetch, because a *fetch* failure at an unmapped
    /// address is the ordinary shape of a guest wild jump and stays a guest
    /// fault.
    ///
    /// `hle` names the HLE call that was executing on this thread, when one was
    /// (`library::function`) — filled in by `run` after recovery, since the VEH
    /// itself must not allocate.
    ///
    /// One residual ambiguity, recorded rather than hidden: guest code that
    /// jumps into an executable host page and then faults also lands here. The
    /// reported `rip` and the call trace distinguish the two by inspection.
    #[error(
        "HOST fault in emulator code at rip {rip:#x} ({kind} {access:#x}){}",
        match hle { Some(name) => format!(" during {name}"), None => String::new() }
    )]
    HostFaulted {
        /// The host instruction pointer that faulted.
        rip: u64,
        /// The address the host instruction tried to touch.
        access: u64,
        /// How it was touching it (never `Execute` — see the type docs).
        kind: FaultKind,
        /// `library::function` of the HLE call in flight, if any.
        hle: Option<String>,
    },
    /// An x86 divide-error trap (`#DE`): a `div`/`idiv` executed with a zero
    /// divisor, or an `idiv` whose quotient overflowed.
    ///
    /// # Why this is its own outcome
    ///
    /// The VEH used to service only `ACCESS_VIOLATION`, `ILLEGAL_INSTRUCTION`
    /// and `BREAKPOINT`; every other exception took `EXCEPTION_CONTINUE_SEARCH`.
    /// With no other handler installed, a `#DE` therefore killed the whole
    /// process with `STATUS_INTEGER_DIVIDE_BY_ZERO` (`0xC000_0094`) and **no log
    /// line at all** — measured on A Plague Tale Requiem, which died at 40.8 s
    /// with zero errors logged and nothing to name the site. Unhandled meant
    /// invisible by construction.
    ///
    /// Classified rather than merely logged, because the two origins want
    /// opposite responses. A [`FaultOrigin::Guest`] `#DE` is very often *our*
    /// fault at one remove: an HLE stub handed the title a zero (a grain, sample
    /// rate, stride, element size, tick frequency) and the title divided by it —
    /// so the recent-HLE-call trace printed alongside is the actionable part. A
    /// [`FaultOrigin::Host`] `#DE` cannot come from safe Rust integer division
    /// (rustc emits an explicit zero check and panics instead), so it points at
    /// a C/C++ dependency or inline assembly.
    ///
    /// `hle` names the HLE call in flight, filled in after recovery for the same
    /// reason as [`Self::HostFaulted`]'s: the VEH must not allocate.
    #[error(
        "{origin} integer-divide fault at rip {rip:#x} ({cause}){}",
        match hle { Some(name) => format!(" during {name}"), None => String::new() }
    )]
    IntegerDivideFault {
        /// The faulting `div`/`idiv` instruction's address.
        rip: u64,
        /// Zero divisor, or quotient overflow.
        cause: DivideFault,
        /// Guest code or emulator code — decided by whether `rip` is in the
        /// guest arena.
        origin: FaultOrigin,
        /// `library::function` of the HLE call in flight, if any.
        hle: Option<String>,
    },
    /// More than 6 integer/pointer arguments were requested — RT0 only
    /// marshals the SysV integer argument registers (design doc §3).
    #[error("more than 6 arguments requested (RT0 marshals only the SysV integer registers)")]
    TooManyArgs,
}

/// Integer/pointer arguments RT0 marshals: SysV RDI, RSI, RDX, RCX, R8, R9
/// (design doc §3).
const MAX_ARGS: usize = 6;

#[cfg(target_os = "windows")]
struct GpuShutdownGuard(raeen_gpu::GpuProcessSession);

#[cfg(target_os = "windows")]
impl Drop for GpuShutdownGuard {
    fn drop(&mut self) {
        self.0.shutdown();
    }
}

/// Run `module`'s function at `entry_offset` (an offset into
/// `module.image`) natively, passing `args` (up to 6 integer/pointer
/// values, SysV) and servicing every HLE trampoline call it makes through
/// `hle`. Each serviced call gets an [`raeen_hle::HleContext`] built from
/// `kernel` and a real, identity-mapped [`arena::GuestArena`] (design doc
/// §2/§5's dispatch-context milestone), passed as both the
/// [`raeen_hle::GuestMemory`] and [`raeen_hle::GuestAllocator`] views — so
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

    // Only one top-level process session can own the fixed guest/trampoline
    // mappings at a time, so the guard covers construction and the whole
    // launch pipeline. Guest pthread workers inside that process may still
    // execute concurrently through their thread-local dispatch contexts.
    let _call_lock = dispatch::call_lock();

    let arena = std::sync::Arc::new(arena::GuestArena::new(&module.image)?);
    arena::maybe_enable_wx_image(&arena, &module.executable_ranges);
    let gpu = GpuShutdownGuard(raeen_gpu::AgcGpuSession::new_process(arena.clone()));
    raeen_gpu::AgcGpuSession::install_process(&gpu.0);
    bind_memory_scheme_address_space(kernel, &arena);
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
    publish_hle_exports_for_dlsym(kernel, &module.hle_trampolines);
    let guard = trampoline::TrampolineGuard::reserve(&module.hle_trampolines, |t| {
        hle.returns_float(&t.library, &t.function)
    })?;
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
    register_main_thread_stack(kernel, &arena);

    // Direct function/module execution enters no crt0, so the loader owns every
    // initializer this module carries, main included (EntryPolicy::
    // LoaderOwnsMainInit). Function-mode fixtures (`link_module` leaves
    // `module_inits` empty) run none; a directly executed module that *does*
    // carry constructors gets them before its entry — and never the double run
    // that only a crt0-owning process entry could cause. An initializer that
    // terminates the process surfaces as the run's value, exactly as the entry
    // point's own `exit` would.
    if let Some(RunOutcome::Exited(code)) = run_module_initializers(
        EntryPolicy::LoaderOwnsMainInit,
        module,
        hle,
        kernel,
        arena.as_ref(),
        &guard,
        &gpu.0,
        tcb,
        static_tls_block,
        None,
        guest_rsp,
    )? {
        return Ok(code);
    }

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
            arena.as_ref(),
            arena.as_ref(),
            &gpu.0,
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
    let arena = std::sync::Arc::new(arena::GuestArena::new(&module.image)?);
    arena::maybe_enable_wx_image(&arena, &module.executable_ranges);
    publish_hle_exports_for_dlsym(kernel, &module.hle_trampolines);
    let guard = trampoline::TrampolineGuard::reserve(&module.hle_trampolines, |t| {
        hle.returns_float(&t.library, &t.function)
    })?;
    let gpu = raeen_gpu::AgcGpuSession::new_process(arena.clone());
    raeen_gpu::AgcGpuSession::install_process(&gpu);
    bind_memory_scheme_address_space(kernel, &arena);
    let result = execute_process_mapped(
        module,
        hle,
        kernel,
        arena.as_ref(),
        &guard,
        &gpu,
        None,
        argv,
        envp,
    );
    gpu.shutdown();
    kernel.log_unresolved_nid_inventory();
    result
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
    execute_process_shared_inner(module, hle, kernel, argv, envp, |_| {})
}

/// Execute a process and publish its ownership handle before entering guest
/// code. Shell/session controllers use the handle for cooperative termination
/// and diagnostics; the runtime still owns teardown and joins all workers.
#[cfg(target_os = "windows")]
pub fn execute_process_shared_with_control(
    module: std::sync::Arc<LinkedModule>,
    hle: std::sync::Arc<HleRegistry>,
    kernel: std::sync::Arc<OrbisKernel>,
    argv: &[&str],
    envp: &[&str],
    on_start: impl FnOnce(GuestProcessHandle),
) -> Result<RunOutcome, RuntimeError> {
    execute_process_shared_inner(module, hle, kernel, argv, envp, on_start)
}

#[cfg(target_os = "windows")]
fn execute_process_shared_inner(
    module: std::sync::Arc<LinkedModule>,
    hle: std::sync::Arc<HleRegistry>,
    kernel: std::sync::Arc<OrbisKernel>,
    argv: &[&str],
    envp: &[&str],
    on_start: impl FnOnce(GuestProcessHandle),
) -> Result<RunOutcome, RuntimeError> {
    let _call_lock = dispatch::call_lock();
    let arena = std::sync::Arc::new(arena::GuestArena::new(&module.image)?);
    arena::maybe_enable_wx_image(&arena, &module.executable_ranges);
    publish_hle_exports_for_dlsym(&kernel, &module.hle_trampolines);
    let guard = std::sync::Arc::new(trampoline::TrampolineGuard::reserve(
        &module.hle_trampolines,
        |t| hle.returns_float(&t.library, &t.function),
    )?);
    let process = thread::GuestProcess::create(module, hle, kernel, arena, guard);
    on_start(process.clone());
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
        &process.gpu,
        Some(&process),
        argv,
        envp,
    );
    thread::release_host_thread_handle(&process.kernel, 1);
    let process_exit_was_requested = process.requested_exit_code().is_some();
    let fallback_code = match &result {
        Ok(RunOutcome::Exited(code)) => *code,
        Ok(RunOutcome::Returned(_)) => 0,
        Err(_) => 1,
    };
    process.terminate_and_reap(fallback_code);
    process.kernel.log_unresolved_nid_inventory();
    if process_exit_was_requested {
        Ok(RunOutcome::Exited(
            process.requested_exit_code().unwrap_or(fallback_code),
        ))
    } else {
        result
    }
}

/// Run the module initializers the loader owns under `policy`, in schedule
/// order (dependencies first, main last), each on `init_rsp` (which must
/// address a guarded return trampoline).
///
/// Returns `Ok(Some(Exited(code)))` if an initializer terminated the whole
/// process (an `exit`-family call from a constructor), which the caller
/// propagates instead of continuing to the entry point; `Ok(None)` to continue.
/// Under [`EntryPolicy::CrtOwnsMainInit`] the main executable's initializer is
/// withheld here (the crt0 `_start` runs it) — the ASTRO.BOT double-init fix.
#[cfg(target_os = "windows")]
#[allow(clippy::too_many_arguments)]
fn run_module_initializers(
    policy: EntryPolicy,
    module: &LinkedModule,
    hle: &HleRegistry,
    kernel: &OrbisKernel,
    arena: &arena::GuestArena,
    guard: &trampoline::TrampolineGuard,
    gpu: &dyn raeen_core::subsystems::GpuSubmissionSubsystem,
    tcb: Option<u64>,
    static_tls_block: Option<u64>,
    guest_threads: Option<&dyn raeen_hle::GuestThreadScheduler>,
    init_rsp: u64,
) -> Result<Option<RunOutcome>, RuntimeError> {
    use raeen_core::diagnostics::DiagnosticKind;

    // A per-entry ordinal in schedule order, stable across runs. The
    // process-scoped `DiagnosticRecorder` assigns the cross-category sequence
    // number; this identifies the initializer within the init phase itself.
    for (index, init) in module.module_inits.iter().enumerate() {
        let ordinal = index + 1;
        if !loader_runs_initializer(policy, init.role) {
            tracing::info!(
                "{}: {} module_start (+{:#x}) deferred to crt0 (process entry owns it)",
                init.name,
                init.role,
                init.image_offset
            );
            kernel.diagnostics.record(
                1,
                DiagnosticKind::ModuleInit,
                init.name.clone(),
                init.image_offset,
                format!("role={} ordinal={ordinal} deferred-to-crt0", init.role),
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
            "{}: calling {} module_start (+{:#x})",
            init.name,
            init.role,
            init.image_offset
        );
        kernel.diagnostics.record(
            1,
            DiagnosticKind::ModuleInit,
            init.name.clone(),
            init.image_offset,
            format!("role={} ordinal={ordinal} run", init.role),
        );
        // SAFETY: `ptr` is executable guest code in the live arena; `init_rsp`
        // points at the guarded return address and dispatch arms recovery
        // before `enter_guest` transfers control — the same contract the
        // process-entry and function-entry calls satisfy.
        let outcome = unsafe {
            dispatch::run(
                &module.hle_trampolines,
                &module.unresolved_stubs,
                hle,
                kernel,
                arena,
                arena,
                gpu,
                guard,
                tcb,
                static_tls_block,
                guest_threads,
                1,
                || crate::stack::enter_guest(ptr as u64, init_rsp, [0; 6]),
            )
        }?;
        match outcome {
            RunOutcome::Returned(rc) => {
                tracing::info!("{}: module_start returned {rc:#x}", init.name);
                // An initializer that RETURNED is the honest proof of this
                // phase: one that faults or exits leaves the loop through the
                // other arm, and the phase must not claim initializers ran when
                // one of them killed the process.
                raeen_core::frame_path::record_phase(
                    raeen_core::frame_path::Phase::InitializersRan,
                );
            }
            RunOutcome::Exited(code) => return Ok(Some(RunOutcome::Exited(code))),
        }
    }
    Ok(None)
}

#[cfg(target_os = "windows")]
#[allow(clippy::too_many_arguments)]
fn execute_process_mapped(
    module: &LinkedModule,
    hle: &HleRegistry,
    kernel: &OrbisKernel,
    arena: &arena::GuestArena,
    guard: &trampoline::TrampolineGuard,
    gpu: &dyn raeen_core::subsystems::GpuSubmissionSubsystem,
    guest_threads: Option<&dyn raeen_hle::GuestThreadScheduler>,
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
            Some(raeen_kernel::UnwindModuleInfo {
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
    register_main_thread_stack(kernel, arena);

    // Lay out the process stack in the arena's stack region (design doc §2);
    // `process::build_process_stack` writes only through `&arena` (bounds-
    // checked `GuestMemory`), never panicking on `argv`/`envp` content.
    let process_rsp = process::build_process_stack(arena.stack_top(), argv, envp, arena)?;
    // The process stack starts with argc followed immediately by the argv
    // pointer table. libc's getargc/getargv may run in a dependency
    // initializer before the executable's `_start`, so publish it now.
    let process_argv = process_rsp.checked_add(8).ok_or(RuntimeError::MapFailed)?;
    kernel.set_process_args(argv.len() as u64, process_argv);
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
    // Process mode enters a genuine crt0 `_start`, which walks the executable's
    // own init array itself — so the loader runs only the DEPENDENCY
    // initializers and withholds the main executable's (EntryPolicy::
    // CrtOwnsMainInit). Running the main initializer here too double-constructs
    // the title's globals: measured on ASTRO.BOT, a list-adding ctor then builds
    // a cyclic list its own walk spins on forever (t1 frozen at a
    // `mov rdx,[rcx]; lea rcx,[rdx+0x10]; jnz` cycle at `module+0x7426c00`).
    // This is the default correctness path; no environment variable required.
    warn_if_deprecated_skip_main_init_set();
    if let Some(exited) = run_module_initializers(
        EntryPolicy::CrtOwnsMainInit,
        module,
        hle,
        kernel,
        arena,
        guard,
        gpu,
        tcb,
        static_tls_block,
        guest_threads,
        module_rsp,
    )? {
        return Ok(exited);
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
            gpu,
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
