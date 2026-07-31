//! # Raeen Kernel
//!
//! High-Level Emulation (HLE) of the PS5's Orbis OS kernel.
//!
//! The PS5 runs a heavily modified FreeBSD kernel (Orbis OS). This crate
//! translates PS5 syscalls to host OS equivalents, manages the emulated
//! virtual address space, implements PS5 threading primitives, and provides
//! a virtual filesystem mapping PS5 paths to host directories.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │              PS5 Game (x86-64 binary)            │
//! │                  ↓ syscall                        │
//! ├──────────────────────────────────────────────────┤
//! │            Syscall Dispatcher                     │
//! │   ┌──────────┬──────────┬───────────┬─────────┐  │
//! │   │  File    │  Memory  │  Thread   │  Other  │  │
//! │   │  Ops     │  Ops     │  Ops      │         │  │
//! │   └──────────┴──────────┴───────────┴─────────┘  │
//! ├──────────────────────────────────────────────────┤
//! │           Host OS (Win32 / POSIX)                │
//! └──────────────────────────────────────────────────┘
//! ```

pub mod aio;
pub mod filesystem;
pub mod hypervisor;
pub mod memory;
pub mod process;
pub mod syscalls;
pub mod threading;

pub use process::ProcessImage;
// Re-exported so HLE module-loading calls can construct kernel module-table
// entries without a direct raeen-core type path (M1-D).
pub use raeen_core::types::ModuleInfo;

use dashmap::DashMap;
use parking_lot::RwLock;
use raeen_core::diagnostics::DiagnosticRecorder;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Captured guest console output (M1-C): everything the guest writes via
/// `printf`/`puts`/`write(1|2, ...)` lands here, byte-for-byte, so the
/// Shell can surface real guest stdout and tests can assert on it. Each
/// write is also mirrored to the host log (`tracing`, target `"guest"`)
/// line-buffered, so `cargo run` output shows guest prints live.
///
/// Bounded: once `MAX_CONSOLE_BYTES` is exceeded the oldest bytes are
/// dropped (keeping the tail) — a guest print loop must not grow host
/// memory without bound.
pub struct Console {
    buf: parking_lot::Mutex<Vec<u8>>,
    /// Bytes of the current, not-yet-newline-terminated line, staged for
    /// the host-log mirror only (the raw `buf` above is unaffected).
    pending_line: parking_lot::Mutex<Vec<u8>>,
}

/// Cap on [`Console`]'s retained output. 4 MiB of text is far more than any
/// M1-era homebrew prints; the tail is what a user debugging a title needs.
const MAX_CONSOLE_BYTES: usize = 4 << 20;

impl Console {
    fn new() -> Self {
        Self {
            buf: parking_lot::Mutex::new(Vec::new()),
            pending_line: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Append raw guest output bytes. Complete lines are mirrored to the
    /// host log (target `"guest"`) as they form.
    pub fn write_bytes(&self, bytes: &[u8]) {
        {
            let mut buf = self.buf.lock();
            buf.extend_from_slice(bytes);
            if buf.len() > MAX_CONSOLE_BYTES {
                let drop_n = buf.len() - MAX_CONSOLE_BYTES;
                buf.drain(..drop_n);
            }
        }

        let mut pending = self.pending_line.lock();
        pending.extend_from_slice(bytes);
        while let Some(nl) = pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = pending.drain(..=nl).collect();
            tracing::info!(target: "guest", "{}", String::from_utf8_lossy(&line[..line.len() - 1]));
        }
        // Bound the pending fragment too (a guest spewing without newlines).
        if pending.len() > MAX_CONSOLE_BYTES {
            let drop_n = pending.len() - MAX_CONSOLE_BYTES;
            pending.drain(..drop_n);
        }
    }

    /// The captured output so far, lossily decoded as UTF-8.
    pub fn contents(&self) -> String {
        String::from_utf8_lossy(&self.buf.lock()).into_owned()
    }

    /// Total captured bytes (post-truncation).
    pub fn len(&self) -> usize {
        self.buf.lock().len()
    }

    /// Whether nothing has been captured yet.
    pub fn is_empty(&self) -> bool {
        self.buf.lock().is_empty()
    }
}

/// One guest-registered NP state callback and its delivery status.
///
/// The two registration spellings carry DIFFERENT argument layouts, so the
/// form must be remembered or the delivery scrambles the callback's view of
/// its own userdata: legacy `sceNpRegisterStateCallback` is
/// `(userId, state, SceNpId *npId, void *userdata)` while the A/toolkit forms
/// are `(userId, state, void *userdata)` (shadPS4 `np_manager.h:34-40`).
/// Delivering the 4-argument layout to a 3-argument callback would hand it the
/// (NULL) `npId` as its userdata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NpStateCallbackRegistration {
    /// Guest entry point of the callback.
    pub entry: u64,
    /// The `userdata` pointer given at registration, passed back verbatim.
    pub userdata: u64,
    /// Legacy 4-argument form (`npId` pointer before `userdata`).
    pub legacy_np_id_arg: bool,
    /// The initial state has been scheduled through `sceNpCheckCallback`.
    pub notified: bool,
}

/// AMPR gather/scatter continuation state for one command buffer: after a
/// `ReadFile*` append, the file id sticks while the destination and file
/// offset continue immediately past the bytes just read, so the follow-up
/// `ReadFileGather` (new file offset, continued destination) /
/// `ReadFileScatter` (new destination, continued file offset) /
/// `ReadFileGatherScatter` (both given) know where "next" is (KytyPS5
/// `libAmpr.cpp` `CommandBufferState` gather_scatter_* fields).
#[derive(Clone, Copy, Debug, Default)]
pub struct AmprGatherScatterState {
    /// APR file id of the read stream being gathered/scattered.
    pub file_id: u32,
    /// Guest address one byte past the previous read's destination.
    pub next_destination: u64,
    /// File offset one byte past the previous read's range.
    pub next_file_offset: u64,
}

/// One Orbis exception (signal) raised at a guest thread and awaiting delivery
/// to the process handler installed for `signum`.
///
/// Recorded by `sceKernelRaiseException` when the target is another thread, and
/// consumed at that thread's next HLE safe point — the only place the runtime
/// can legally re-enter guest code on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingException {
    /// Orbis signal number (`sceKernelRaiseException`'s second argument;
    /// SIGUSR1 = 30 is the one titles actually raise — it is what Unity's
    /// stop-the-world collector uses to suspend a thread).
    pub signum: i32,
    /// The guest handler address installed for `signum` at raise time. Latched
    /// here rather than re-read at delivery so a concurrent
    /// `sceKernelRemoveExceptionHandler` cannot turn a queued raise into a
    /// jump through a stale slot.
    pub handler: u64,
    /// The guest thread that raised it. Diagnostic only.
    pub raised_by: u64,
}

/// The emulated PS5 kernel state.
///
/// Holds all kernel-level state: memory map, file descriptors,
/// threads, loaded modules, and system configuration.
pub struct OrbisKernel {
    /// Captured guest console output (stdout/stderr + libc print family).
    pub console: Console,
    /// Virtual memory manager.
    pub memory: Arc<memory::VirtualMemoryManager>,
    /// Thread manager.
    pub threads: Arc<threading::ThreadManager>,
    /// Virtual filesystem.
    pub filesystem: Arc<filesystem::VirtualFileSystem>,
    /// Host-threadpool-backed kernel AIO engine (`sceKernelAio*`). Performs
    /// its file I/O through [`Self::filesystem`] — the same descriptor table
    /// the synchronous read/write path uses. Workers spawn lazily on the
    /// first submit.
    pub aio: aio::AioEngine,
    /// Stable, process-scoped HLE/wait/event/task/GPU event stream.
    pub diagnostics: Arc<DiagnosticRecorder>,
    /// Distinct unresolved callable imports observed by this guest process,
    /// keyed by `(NID, resolved name, import library, calling module)` and
    /// counted. Keeping this process-owned makes the default fail-soft policy produce a
    /// complete, de-duplicated compatibility inventory without one title
    /// suppressing another title's first occurrence.
    unresolved_nid_calls: DashMap<(u64, String, String, String), u64>,
    /// Monotonic epoch owned by this guest process rather than a host-global
    /// static, so consecutive title launches do not inherit elapsed time.
    started_at: std::time::Instant,
    /// Loaded modules (.sprx / .elf).
    pub modules: DashMap<u32, ModuleInfo>,
    /// Handle-scoped NID -> absolute guest addresses for real LLE modules.
    lle_module_exports: DashMap<u32, HashMap<u64, u64>>,
    /// Function name -> the guest-callable HLE trampoline address that stands
    /// for it in *this* process, published by the runtime from the process-wide
    /// trampoline table (`LinkedModule::hle_trampolines`).
    ///
    /// Only `sceKernelDlsym` reads this. Imports never do: the linker has
    /// already written each import's trampoline address straight into its
    /// relocation slot. `dlsym` is the one caller that has to turn a *name*
    /// into a callable address at run time, with no relocation to consult.
    hle_export_addrs: DashMap<String, u64>,
    /// Next module ID to assign.
    next_module_id: RwLock<u32>,
    /// Syscall statistics (for debugging).
    pub syscall_stats: DashMap<u64, u64>,
    /// Guest thread names (`scePthreadRename`), keyed by guest thread id —
    /// purely diagnostic, so a dying thread can be identified by name.
    pub thread_names: DashMap<u64, String>,
    /// Guest thread scheduling priorities (`scePthreadSetprio`), keyed by
    /// guest thread id. Recorded so `scePthreadGetprio` reads back exactly
    /// what the title set; Raeen does not map these onto host scheduling.
    pub thread_priorities: DashMap<u64, i32>,
    /// Guest thread scheduling policies (`scePthreadSetschedparam`), keyed by
    /// guest thread id. Same bookkeeping class as [`Self::thread_priorities`]:
    /// recorded so `scePthreadGetschedparam` reports the policy the title set,
    /// defaulting to `PthreadAttr::default().sched_policy` when never set.
    pub thread_sched_policies: DashMap<u64, i32>,
    /// Host (OS) thread handle for each guest thread id, recorded as the thread
    /// starts. It lets diagnostics sample a title stuck between HLE calls and,
    /// under the runtime's A/B gate, lets `scePthreadSetprio` update the live
    /// host scheduler rather than only its readback bookkeeping. On Windows
    /// these are duplicated `HANDLE`s owned for the thread lifetime.
    pub host_thread_handles: DashMap<u64, u64>,
    /// Each live guest thread's stack as `[base, top)`, keyed by guest thread
    /// id — the arena's stack region for the main thread, and the freshly
    /// allocated stack for every `scePthreadCreate` worker. Registered by the
    /// runtime as a thread starts and removed as it exits.
    ///
    /// This is what lets the HLE answer "is this out-pointer a caller local?"
    /// **exactly** instead of guessing from an address range: a secondary
    /// thread's stack comes out of the same arena heap as ordinary
    /// allocations, so nothing about its address distinguishes it. The
    /// out-buffer guard (`raeen_hle::out_buffer`) uses the answer to refuse
    /// bulk-initializing a caller frame, which is the difference between an
    /// oversized HLE write being harmless and it smashing the caller's
    /// `__stack_chk_guard` canary.
    pub guest_thread_stacks: DashMap<u64, (u64, u64)>,
    /// Per-guest-thread ring of the most recent HLE `library::function` calls,
    /// keyed by guest thread id. Populated only when diagnostics ask for it
    /// (the __cxa_throw trap), so a thread that throws can report the exact
    /// calls that led there — host threads are pooled, so a host-ThreadId
    /// correlation is unreliable; the guest thread id is not.
    pub recent_hle_calls: DashMap<u64, parking_lot::Mutex<std::collections::VecDeque<String>>>,
    /// The HLE call each guest thread is CURRENTLY inside (set on entry, cleared
    /// on return), keyed by guest thread id. The recent-call ring records on
    /// entry too, but a call that never returns (a thread blocked in a host wait
    /// deep inside an HLE call) is indistinguishable there from one that returned
    /// — this names the exact in-flight call a stalled thread is parked in.
    pub in_flight_hle: DashMap<u64, String>,
    /// Guest address of the main module's `PT_SCE_PROCPARAM` block, set by
    /// the runtime at load time (0 = none). `sceKernelGetProcParam` returns
    /// this so a title reads its real process-parameter block (SDK version,
    /// etc.) instead of a stub pointer.
    proc_param_addr: std::sync::atomic::AtomicU64,
    /// Process entry arguments recorded from the runtime-built initial stack.
    /// `getargc` and `getargv` expose these to libc module initializers.
    process_argc: std::sync::atomic::AtomicU64,
    process_argv: std::sync::atomic::AtomicU64,
    /// Real ELF ranges and exception tables for the currently executing
    /// process. The runtime replaces this atomically before entering `_start`.
    unwind_modules: RwLock<Vec<UnwindModuleInfo>>,
    /// The current controller snapshot as a 12-byte Orbis `ScePadData` input
    /// prefix (buttons + sticks + triggers), or `None` when the host has not
    /// pushed live input — in which case `scePadReadState` reports a neutral
    /// pad. The Shell updates this each frame from its `InputManager`; the
    /// HLE `scePadReadState` reads it. See [`set_pad_state`](Self::set_pad_state).
    pad_state: parking_lot::Mutex<Option<[u8; 12]>>,
    /// The title's newest `scePadSetVibration` request as `(sequence,
    /// largeMotor, smallMotor)`. The sequence starts at 0 (= never set) and
    /// increments on every guest call — including calls that repeat the same
    /// motor values, because a repeated call is how a title refreshes its
    /// vibration on real hardware and the host router keys its safety
    /// auto-stop off that freshness. Read by the host each frame (Shell
    /// in-process, or the isolated runner's input thread which forwards it
    /// over the frame IPC). See [`set_pad_rumble`](Self::set_pad_rumble).
    pad_rumble: parking_lot::Mutex<(u64, u8, u8)>,
    /// Whether libSceUserService has delivered this process's initial login
    /// event. A new guest process gets a fresh kernel and therefore a fresh
    /// event; keeping this here avoids a host-global flag leaking across
    /// consecutive title launches.
    user_service_login_event_delivered: std::sync::atomic::AtomicBool,
    /// Guest pthread mutex state, keyed by both the guest `pthread_mutex_t`
    /// address and its allocated opaque handle (both map to the same logical
    /// mutex). Manipulated by the HLE `pthread_sync` module — per-process
    /// state, so it lives here rather than in a global.
    pub pthread_mutexes: DashMap<u64, Arc<PthreadMutexShared>>,
    /// Guest pthread mutex-attribute type, keyed by the attr object address.
    pub pthread_mutex_attrs: DashMap<u64, i32>,
    /// The process's futex-style parking lot: threads blocked in
    /// `sceKernelSyncOnAddress{Wait,Wait32,Wait64}` (and the `futex` syscall),
    /// keyed by the watched guest address. Per-process, like the other lock
    /// tables.
    pub sync_addresses: SyncAddressTable,
    /// Guest pthread read-write lock state, keyed by both the guest
    /// `pthread_rwlock_t` address and its allocated handle.
    pub pthread_rwlocks: DashMap<u64, Arc<PthreadRwlockShared>>,
    /// Per-`(guest thread id, rwlock key)` read-hold recursion depth.
    /// [`PthreadRwlock::readers`] is a single shared count that cannot say
    /// *which* thread holds a read; without that, a stray or duplicated
    /// `scePthreadRwlockUnlock` from a non-holder would decrement (and
    /// eventually free) another thread's read hold, letting a writer in while a
    /// reader is still inside the lock. This gates the read-release on the
    /// caller actually owning one. See `raeen-hle` `pthread_sync`.
    pub pthread_rwlock_read_holds: DashMap<(u64, u64), u32>,
    /// Diagnostic (`RAEEN_TIME_HLE`): `(guest thread id, "library::function")`
    /// -> (call count, total microseconds inside that call). Attributes a
    /// stalled thread's wall-clock to the specific call it is parked in, which
    /// [`Self::recent_hle_calls`] cannot do — the ring names the calls but not
    /// their duration, so one long wait and a fast spin look identical.
    pub hle_call_time: DashMap<(u64, String), (u64, u128)>,
    /// Diagnostic (`RAEEN_CALL_STATS`): `"library::function"` -> (calls during
    /// the first 30 s of the run, calls after). A title that POLLS a readiness
    /// value in its steady state ranks the polled function at the top of the
    /// second window — [`Self::hle_call_time`] cannot show this because it is
    /// per-thread and only populated under `RAEEN_TIME_HLE`'s timing overhead.
    /// Relaxed atomics; incremented in the dispatch path only when the env var
    /// is set.
    pub hle_call_counts:
        DashMap<String, (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64)>,
    /// Guest pthread condition-variable wait queues, keyed by object address.
    pub pthread_conds: DashMap<u64, Arc<PthreadCond>>,
    /// Clock id set on a guest `pthread_condattr_t` by
    /// `pthread_condattr_setclock`, keyed by the attr's address. Read once by
    /// `pthread_cond_init` to fix the new cond's clock, then irrelevant —
    /// POSIX attr objects are inputs to init, not live links.
    pub pthread_condattr_clocks: DashMap<u64, u64>,
    /// Guest pthread thread-attribute objects, keyed by both the guest
    /// `pthread_attr_t` address and its allocated handle.
    pub pthread_attrs: DashMap<u64, PthreadAttr>,
    /// Process exception handlers installed through
    /// `sceKernelInstallExceptionHandler`, keyed by Orbis signal number.
    ///
    /// Delivery is a runtime concern; keeping the registration in process
    /// state still gives install/remove/raise coherent ABI behavior.
    pub exception_handlers: DashMap<i32, u64>,
    /// Orbis exceptions raised with `sceKernelRaiseException` at a thread that
    /// is **not** the caller, awaiting delivery at that thread's next HLE
    /// safe point. Keyed by target guest thread id.
    ///
    /// One slot per thread, newest wins: a raise is a level, not a queue —
    /// the guest handler for a repeated signal runs once per observation, and
    /// an unbounded backlog would let a stop-the-world collector that raises
    /// each cycle accumulate deliveries it no longer wants.
    ///
    /// Cross-thread rather than immediate because a guest signal handler must
    /// run *on the target thread's own stack*: hijacking a running host worker
    /// from outside is exactly the corruption `raeen-runtime`'s cooperative
    /// exit model avoids. See `raeen-hle`'s `exception` module.
    pub pending_exceptions: DashMap<u64, PendingException>,
    /// `pending_exceptions.len()`, cached as a relaxed atomic.
    ///
    /// Every HLE call consults the pending set (the safe point is the dispatch
    /// boundary), and `DashMap::is_empty` locks every shard to sum lengths —
    /// unacceptable on that path. Reads of this counter are the fast "nothing
    /// to deliver" answer; only a non-zero value pays for a map lookup.
    pending_exception_count: std::sync::atomic::AtomicUsize,
    /// Threads currently executing a guest exception handler. Delivery is a
    /// re-entrant path (the handler makes HLE calls, each of which is another
    /// safe point), so a thread inside its handler must not be handed the
    /// next signal until it returns.
    pub exception_delivery_active: DashMap<u64, ()>,
    /// Per-thread guest scratch for the `ucontext_t` handed to an exception
    /// handler, keyed by guest thread id. Allocated once per thread on first
    /// delivery and reused: a handler receives a pointer to it and must not
    /// see it recycled underneath a concurrent delivery on another thread.
    pub exception_contexts: DashMap<u64, u64>,
    /// Kernel event flags, keyed by handle.
    pub kernel_event_flags: DashMap<u64, EventFlag>,
    /// Next event-flag handle to hand out.
    kernel_event_flag_next: std::sync::atomic::AtomicU64,
    /// Live event slots used for atomic resource-quota admission.
    kernel_event_flag_live: std::sync::atomic::AtomicU32,
    /// Kernel counting semaphores, keyed by handle.
    pub kernel_semaphores: DashMap<u32, Semaphore>,
    /// "Save data memory" blobs, keyed by `SceUserServiceUserId`. This is the
    /// mountless per-user save blob (`sceSaveDataSetupSaveDataMemory2` sizes it,
    /// `Get/Set/Sync` transfer it) — distinct from the mounted `/savedata0` VFS.
    /// In-memory for the session (not persisted across launches), zero-filled at
    /// setup; see `raeen-hle`'s `libsce_save_data` save-memory functions.
    pub save_data_memory: DashMap<i32, Vec<u8>>,
    /// POSIX (`sem_*`) semaphores, keyed by the guest `sem_t` address. These
    /// are address-based objects distinct from the handle-based
    /// `sceKernelCreateSema` family — see the `raeen-hle` `posix_sem` module.
    pub posix_semaphores: DashMap<u64, Arc<PosixSem>>,
    /// Next semaphore handle to hand out.
    kernel_semaphore_next: std::sync::atomic::AtomicU32,
    /// Kernel event queues (existence), keyed by handle → attributes.
    pub kernel_equeues: DashMap<u64, u32>,
    /// Registered user events on event queues, keyed by (equeue, ident).
    pub kernel_equeue_events: DashMap<(u64, u64), EqueueUserEvent>,
    /// Next event-queue handle to hand out.
    kernel_equeue_next: std::sync::atomic::AtomicU64,
    /// APR (`sceKernelAprResolve*`) file registry: the FNV-1a id of a guest
    /// path → its resolved host path, so a later AMPR read-by-id finds the
    /// file. SharpEmu's `AmprFileRegistry` model.
    pub appr_files: DashMap<u32, String>,
    /// APR async-read host-handle cache: APR id → an open host `File` used
    /// for positional (`seek_read`/`read_at`) reads, so a title re-reading
    /// one asset does not re-open it per command record. SharpEmu
    /// `AmprExports._hostFileCache` (keyed there by host path; the APR id
    /// already maps 1:1 to one resolved path here).
    pub appr_file_handles: DashMap<u32, std::fs::File>,
    /// Guest-registered NP (PSN) state callbacks awaiting delivery of the
    /// initial account state through `sceNpCheckCallback`.
    ///
    /// On real hardware the system queues the current sign-in state at
    /// registration and invokes the callback on the title's own thread the
    /// next time it pumps `sceNpCheckCallback` (shadPS4
    /// `np_manager.cpp` `DispatchPendingNpStateCallbacks`, whose init comment
    /// says exactly this: events are "delivered on the game's thread during
    /// sceNpCheckCallback"). Registering the callback but never firing it
    /// starves any title whose UI waits for the state event rather than
    /// polling `sceNpGetState` — measured on Minecraft's post-"Get started"
    /// screen, which pumps `sceNpCheckCallback` ~10x/s forever with a blank
    /// page.
    pub np_state_callbacks: parking_lot::Mutex<Vec<NpStateCallbackRegistration>>,
    /// Live offline NP-auth request handles. Authentication never reaches a
    /// host service, but titles still expect Create/Get/Delete to have a
    /// coherent process-local lifetime instead of faulting on unresolved
    /// imports.
    pub np_auth_requests: DashMap<i32, ()>,
    /// Next NP-auth request id; the first issued id is `0x1000_0001`.
    pub np_auth_next_request: std::sync::atomic::AtomicI32,
    /// APR ids already named by the once-per-id missing-file warn in
    /// `sceAmprAprCommandBufferReadFile` — the "name the miss" diagnostic
    /// stays visible without spamming one warn per frame.
    pub appr_missing_warned: DashMap<u32, ()>,
    /// Offline epoll instances (`sceNetEpoll*`): epoll id → registered
    /// (fd, events, udata) tuples. No host-network backend exists, so a Wait
    /// always reports no events after its timeout — the honest offline model.
    pub kernel_epolls: DashMap<u32, Vec<(i32, u32, u64)>>,
    /// Next epoll id to hand out.
    kernel_epoll_next: std::sync::atomic::AtomicU32,
    /// APR command-buffer submissions (synchronous model): submission id →
    /// command buffer, completed at submit time; `sceKernelAprWaitCommandBuffer`
    /// just consumes the entry. SharpEmu's `_submittedCommandBuffers` model.
    pub appr_submissions: DashMap<u32, u64>,
    /// Next APR submission id to hand out.
    appr_next_submission: std::sync::atomic::AtomicU32,
    /// The Agc driver's registered default resource owner (`sceAgcDriver*`).
    pub agc_default_owner: std::sync::atomic::AtomicU32,
    /// Whether the per-process Agc resource-registration arena is active.
    pub agc_resource_registration_initialized: std::sync::atomic::AtomicBool,
    /// Maximum number of Agc resource owners accepted for this process.
    pub agc_resource_registration_max_owners: std::sync::atomic::AtomicU32,
    /// Next candidate Agc owner handle.
    pub agc_next_owner: std::sync::atomic::AtomicU32,
    /// Registered Agc owner handles and their diagnostic names.
    pub agc_resource_owners: DashMap<u32, String>,
    /// Next candidate Agc resource handle.
    pub agc_next_resource: std::sync::atomic::AtomicU32,
    /// Guest GPU allocations registered for diagnostics and submission.
    pub agc_resources: DashMap<u32, AgcResource>,
    /// Number of structurally valid AGC DCBs submitted by this process.
    pub agc_submission_count: std::sync::atomic::AtomicU64,
    /// Draw packets observed across valid AGC DCB submissions.
    pub agc_draw_packet_count: std::sync::atomic::AtomicU64,
    /// Compute-dispatch packets observed across valid AGC DCB submissions.
    pub agc_dispatch_packet_count: std::sync::atomic::AtomicU64,
    /// VideoOut flip packets observed across valid AGC DCB submissions.
    pub agc_flip_packet_count: std::sync::atomic::AtomicU64,
    /// Last RELEASE_MEM GPU-timestamp fence value written for this session
    /// (per-session so a relaunch restarts with the session's monotonic clock
    /// instead of counting up from a prior session's final value).
    pub agc_gpu_timestamp: std::sync::atomic::AtomicU64,
    /// Most recently submitted DCB address (diagnostic capture metadata).
    pub agc_last_dcb_address: std::sync::atomic::AtomicU64,
    /// Most recently submitted DCB length in DWORDs.
    pub agc_last_dcb_dwords: std::sync::atomic::AtomicU32,
    /// Guest address of the process's materialized AGC register-defaults
    /// block (`sceAgcGetRegisterDefaults2[Internal]`), or 0 before the first
    /// call. The guest walks pointers inside this block, so it is allocated
    /// once in guest memory and cached here for every later call.
    pub agc_register_defaults_addr: std::sync::atomic::AtomicU64,
    /// Graphics PM4 the title has BUILT after its last DCB submit but not yet
    /// submitted (KytyPS5 `g_pending_graphics_segment`, agc.cpp L214-264):
    /// the segment starts where the last submitted DCB ended and grows with
    /// each contiguous command-buffer allocation in that range. An ACB whose
    /// waits depend on a `RELEASE_MEM` in this unsubmitted tail would park
    /// forever; the HLE flushes the segment as a DCB before every ACB submit.
    pub agc_pending_graphics_segment: std::sync::Mutex<AgcPendingGraphicsSegment>,
    /// Display buffers registered by VideoOut, keyed by `(port, slot)`.
    pub video_out_buffers: DashMap<(i32, i32), VideoOutBuffer>,
    /// Completed VideoOut flips for this process.
    /// Bytes of direct memory currently allocated, against
    /// [`Self::DIRECT_MEMORY_SIZE`]. A real PS5 exposes a fixed direct-memory
    /// budget and titles *discover* it by allocating until the kernel refuses —
    /// Dragon Ball allocates 1 GiB in a loop expecting ENOMEM after ~13 GiB.
    /// Without a budget that loop consumed ~900 GiB of host address space and
    /// then fell over on placement instead of ending normally.
    pub direct_memory_allocated: std::sync::atomic::AtomicU64,
    pub video_out_flip_count: std::sync::atomic::AtomicU64,
    /// Process-local vertical-blank sequence used by frame pacing APIs.
    pub video_out_vblank_count: std::sync::atomic::AtomicU64,
    /// Guest correlation value from the most recently completed flip.
    pub video_out_last_flip_arg: std::sync::atomic::AtomicI64,
    /// Buffer slot selected by the most recently completed flip.
    pub video_out_current_buffer: std::sync::atomic::AtomicI32,
    /// libc fixed-capacity heaps keyed by their guest mspace handle.
    pub libc_mspaces: DashMap<u64, LibcMspace>,
    /// Active allocations carved from libc mspaces, keyed by guest address.
    pub libc_mspace_allocations: DashMap<u64, LibcMspaceAllocation>,
    /// One shared (lock, condvar) signalled whenever any event flag's bits
    /// change (Set/Clear/Cancel). Waiters re-check their own pattern on wake —
    /// a single condvar is correct under spurious wakeups and avoids per-flag
    /// registration churn.
    pub event_flag_signal: (std::sync::Mutex<()>, std::sync::Condvar),
    /// One shared (lock, condvar) signalled whenever any counting semaphore's
    /// state changes (Signal/Cancel/Delete). A blocked `WaitSema` re-checks its
    /// own count on wake. Same rationale as [`event_flag_signal`]: with real
    /// guest threads a producer thread *can* signal, so a waiter must block
    /// until it does rather than instantly time out.
    ///
    /// Notify through [`Self::wake_semaphore_waiters`] rather than reaching in
    /// here, so every producer is counted.
    pub semaphore_signal: (std::sync::Mutex<()>, std::sync::Condvar),
    /// How many times semaphore waiters were woken — see
    /// [`Self::wake_semaphore_waiters`].
    semaphore_wakes: std::sync::atomic::AtomicU64,
    /// AMPR command-buffer write offsets, keyed by the command-buffer address
    /// (the current write cursor `sceAmprCommandBufferGetCurrentOffset` reads).
    pub ampr_write_offsets: DashMap<u64, u64>,
    /// AMPR per-command-buffer appended-record counts, keyed by the
    /// command-buffer address (`sceAmprCommandBufferGetNumCommands` reads;
    /// SharpEmu's `AmprCommandBufferState.CommandCount` — zeroed on
    /// construct/reset, incremented per appended record).
    pub ampr_command_counts: DashMap<u64, u64>,
    /// AMPR per-command-buffer gather/scatter continuation state, keyed by
    /// the command-buffer address. Seeded/advanced by every successful
    /// `sceAmprAprCommandBufferReadFile*` append (KytyPS5 `libAmpr.cpp`
    /// `AppendReadFileRecord`: file id sticks, destination and file offset
    /// continue past the previous read); presence in the map = "valid".
    /// Cleared by construct/reset/destruct and
    /// `sceAmprAprCommandBufferResetGatherScatterState`.
    pub ampr_gather_scatter: DashMap<u64, AmprGatherScatterState>,
    /// AMPR per-command-buffer "type" flag word (`sceAmprCommandBufferGetType`
    /// reads it), host-tracked because Raeen's guest-visible command-buffer
    /// struct layout has no type field at +0x00 (KytyPS5 keeps these bits in
    /// the guest header; the observable flag semantics are the same):
    /// `0x0001_0000` = gather/scatter state valid, `0x0002_0000` = an APR
    /// map window is active (MapBegin seen, MapEnd pending).
    pub ampr_type_flags: DashMap<u64, u32>,
    /// The libSceFiber context-size-check profiling toggle (0 = off, 1 = on).
    pub fiber_context_size_check: std::sync::atomic::AtomicU32,
    /// Suspended-fiber snapshots keyed by the guest `SceFiber` address: the
    /// saved guest registers plus the `*arg_on_run` out-pointer the fiber passed
    /// when it suspended (poked with the resuming call's arg before it resumes).
    /// Present iff the fiber has yielded at least once; absent means "never run",
    /// so `sceFiberRun` builds the first-run entry frame instead.
    pub fibers: DashMap<u64, (GuestRegs, u64)>,
    /// Per host-thread fiber state — the `sceFiberRun` return anchor and the
    /// fiber currently owned — keyed by host (OS) thread id.
    pub fiber_threads: DashMap<u64, FiberThreadState>,
    /// Guest network sockets (offline — no host connectivity), keyed by fd.
    pub kernel_sockets: DashMap<i32, GuestSocket>,
    /// Next socket fd to hand out (a high range, distinct from VFS fds).
    kernel_socket_next: std::sync::atomic::AtomicI32,
    /// Live socket slots used for atomic descriptor-quota admission.
    kernel_socket_live: std::sync::atomic::AtomicU32,
    /// Registered pthread TLS keys → their destructor address (0 = none).
    pub pthread_tls_keys: DashMap<i32, u64>,
    /// Thread-local specific values, keyed by (thread handle, TLS key).
    pub pthread_tls_values: DashMap<(u64, i32), u64>,
    /// Dynamic TLS blocks allocated by `libkernel::__tls_get_addr`, keyed by
    /// `(guest thread, module identifier)`.
    pub dynamic_tls_blocks: DashMap<(u64, u64), u64>,
    /// The process's static TLS layout: TLS module id → offset of that
    /// module's block *within* the per-thread static area (from the area's
    /// low end; the area spans `[tp - total, tp)`). Registered once at
    /// launch from the linker's layout so `__tls_get_addr` resolves a
    /// static module to the SAME storage its `TPOFF64` accesses use —
    /// per-module, which module 1's block base alone cannot express once a
    /// process has more than one module with TLS.
    pub static_tls_area_offsets: DashMap<u64, u64>,
    /// Per-thread guest addresses returned by libkernel `__error()`.
    pub errno_slots: DashMap<u64, u64>,
    /// Save-data transaction resources, keyed by resource id -> memory size.
    pub save_data_transaction_resources: DashMap<i32, u64>,
    /// Next save-data transaction resource id.
    pub save_data_next_transaction_resource: std::sync::atomic::AtomicI32,
    /// Save-data metadata values keyed by (mount point, parameter type).
    pub save_data_params: DashMap<(String, u32), Vec<u8>>,
    /// libc/rtld callback pointers registered for this guest process.
    pub thread_dtors_callback: std::sync::atomic::AtomicU64,
    pub thread_atexit_count_callback: std::sync::atomic::AtomicU64,
    pub thread_atexit_report_callback: std::sync::atomic::AtomicU64,
    /// Guest application heap API table registered with the rtld.
    pub application_heap_api: std::sync::atomic::AtomicU64,
    /// Process-owned libc trace table and coredump callback pointers. These
    /// contain guest addresses and must never survive into another launch.
    pub libc_trace_storage: std::sync::atomic::AtomicU64,
    pub coredump_handler: std::sync::atomic::AtomicU64,
    pub coredump_handler_context: std::sync::atomic::AtomicU64,
    /// Process-owned pacing state for AudioOut ports: `(grain, frequency)`.
    audio_out_ports: DashMap<u32, (u32, u32)>,
    audio_out_next_port: std::sync::atomic::AtomicU32,
    audio_out_live_ports: std::sync::atomic::AtomicU32,
    /// Whether the optional libSceTextToSpeech2 service is initialized for
    /// this guest process. The audio synthesis surface is layered on top of
    /// this lifecycle state rather than using process-global statics.
    pub text_to_speech2_initialized: std::sync::atomic::AtomicBool,
    /// Next TLS key id to hand out.
    pthread_tls_next_key: std::sync::atomic::AtomicI32,
    /// libSceHttp contexts (existence set for `sceHttpTerm`), keyed by context
    /// id → recorded pool size. Ported from SharpEmu HttpExports (GPL-2.0).
    pub http_contexts: DashMap<i32, u64>,
    /// Next libSceHttp context id (increment-before-use; first id is 1).
    pub http_next_context: std::sync::atomic::AtomicI32,
    /// libSceHttp templates, keyed by template id → owning context id (so
    /// `sceHttpTerm` can cascade-remove all of a context's templates).
    pub http_templates: DashMap<i32, i32>,
    /// Next libSceHttp template id (starts at 0x1000).
    pub http_next_template: std::sync::atomic::AtomicI32,
    /// libSceHttp2 contexts, keyed by context id → recorded pool size.
    pub http2_contexts: DashMap<i32, u64>,
    /// Next libSceHttp2 context id (increment-before-use; first id is 1).
    pub http2_next_context: std::sync::atomic::AtomicI32,
    /// libSceHttp2 templates, keyed by template id -> owning HTTP2 context.
    pub http2_templates: DashMap<i32, i32>,
    /// Next libSceHttp2 template id (starts at 0x1000).
    pub http2_next_template: std::sync::atomic::AtomicI32,
    /// libSceSsl contexts, keyed by context id → recorded pool size.
    pub ssl_contexts: DashMap<i32, u64>,
    /// Next libSceSsl context id (increment-before-use; first id is 1).
    pub ssl_next_context: std::sync::atomic::AtomicI32,
}

/// A guest network socket. Raeen models **no host connectivity**, so a socket
/// can be created and bound (bookkeeping the guest reads back via
/// `getsockname`) but `connect` never succeeds. Ported from SharpEmu's socket
/// state (GPL-2.0), minus the real host-TCP path.
#[derive(Clone, Copy, Debug, Default)]
pub struct GuestSocket {
    /// Bound IPv4 address (network byte order, as the guest supplied it).
    pub bound_ip: [u8; 4],
    /// Bound port (host byte order).
    pub bound_port: u16,
    /// Whether `bind` has been called.
    pub bound: bool,
    /// Per-descriptor flags returned by `fcntl(F_GETFD)` (for example
    /// close-on-exec). Raeen does not currently replace a guest process image,
    /// but retaining the value is observable through a later `F_GETFD`.
    pub descriptor_flags: i32,
    /// File-status flags returned by `fcntl(F_GETFL)`, including nonblocking.
    pub status_flags: i32,
}

/// A user event registered on a kernel event queue (`EVFILT_USER`). Ported
/// from SharpEmu's event-queue registration (GPL-2.0). `Trigger` marks it
/// pending with `udata`; `WaitEqueue` delivers pending events and (edge)
/// clears them.
#[derive(Clone, Copy, Debug)]
pub struct EqueueUserEvent {
    /// User data delivered with the event.
    pub udata: u64,
    /// Whether the event is currently pending (triggered, not yet delivered).
    pub triggered: bool,
    /// Trigger count (delivered as the event's `fflags`).
    pub fflags: u32,
    /// Kernel event filter (`EVFILT_USER`, VideoOut, graphics, ...).
    pub filter: i16,
    /// Filter-specific signed event payload.
    pub data: i64,
}

impl Default for EqueueUserEvent {
    fn default() -> Self {
        Self {
            udata: 0,
            triggered: false,
            fflags: 0,
            filter: -11,
            data: 0,
        }
    }
}

/// The unsubmitted graphics tail of the title's command ring, in guest byte
/// addresses (KytyPS5 `PendingGraphicsSegment`, agc.cpp L214-218). `start` is
/// where the last submitted DCB ended; `end` grows with contiguous
/// command-buffer allocations; `range_end` bounds how far the segment may
/// grow (start + 0xfffff dwords). All zero when nothing is tracked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AgcPendingGraphicsSegment {
    pub start: u64,
    pub end: u64,
    pub range_end: u64,
}

/// Metadata supplied through `sceAgcDriverRegisterResource`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgcResource {
    pub owner: u32,
    pub address: u64,
    pub size: u64,
    pub name: String,
    pub resource_type: u32,
    pub flags: u32,
}

/// Gen5 VideoOut buffer attributes needed to interpret a presented image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoOutBufferAttribute {
    pub pixel_format: u64,
    pub tiling_mode: u32,
    pub width: u32,
    pub height: u32,
    /// `pitchInPixel` — the buffer's row stride in PIXELS, which is NOT
    /// necessarily `width`.
    ///
    /// A title may allocate a display buffer whose rows are padded (or share a
    /// wider allocation) and present only a `width`-wide window of each row.
    /// Reading such a buffer at a `width` stride walks diagonally through it
    /// instead of down it, which renders as horizontal striping. `0` means the
    /// guest left the field unset, in which case a tightly-packed row
    /// (`pitch == width`) is the right assumption.
    pub pitch_pixels: u32,
    pub option: u64,
    pub dcc_clear_color: u64,
    pub dcc_control: u32,
}

/// One guest display buffer captured by `sceVideoOutRegisterBuffers2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoOutBuffer {
    pub set_index: i32,
    pub address: u64,
    pub metadata: u64,
    pub attribute: VideoOutBufferAttribute,
}

/// Process-local state for a libc mspace created over guest-owned memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibcMspace {
    pub base: u64,
    pub capacity: u64,
    pub next_offset: u64,
    pub peak_offset: u64,
    pub active_bytes: u64,
    pub name: String,
    /// Reclaimed blocks `(offset_from_base, size)`, kept sorted by offset and
    /// coalesced. Malloc reuses a fitting free block before bumping `next_offset`,
    /// so malloc/free churn does not exhaust a fixed-capacity mspace — native
    /// dlmalloc reclaims, and a bump-only allocator that doesn't makes a title's
    /// global heap OOM after enough turnover (measured on ASTRO.BOT).
    pub free_list: Vec<(u64, u64)>,
}

/// One allocation carved out of a libc mspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LibcMspaceAllocation {
    pub mspace: u64,
    pub size: u64,
}

/// A kernel counting semaphore. Ported from SharpEmu's `KernelSemaphoreState`
/// (GPL-2.0). The count is fully correct under single-active-execution; a
/// blocking `Wait` on an empty semaphore needs the M1-E scheduler.
#[derive(Clone, Copy, Debug, Default)]
pub struct Semaphore {
    /// Current available count.
    pub count: i32,
    /// Maximum count (ceiling for `Signal`).
    pub max_count: i32,
}

/// A kernel event flag: a 64-bit set of condition bits a title waits on / sets
/// / clears. Ported from SharpEmu's `EventFlagState` (GPL-2.0). The bit state
/// is fully correct under single-active-execution; true cross-thread blocking
/// waits need the M1-E scheduler.
#[derive(Clone, Copy, Debug, Default)]
pub struct EventFlag {
    /// Current condition bits.
    pub bits: u64,
    /// Creation attributes (queue/thread mode).
    pub attributes: u32,
}

/// A guest pthread thread-attribute object (`pthread_attr_t`) — the stack
/// size, detach state, guard size, and scheduling parameters a title sets
/// before `scePthreadCreate`. Pure configuration data with no runtime
/// dependency; defaults match SharpEmu's `PthreadAttrState.Default` (GPL-2.0).
#[derive(Clone, Copy, Debug)]
pub struct PthreadAttr {
    /// 0 = joinable, 1 = detached.
    pub detach_state: i32,
    /// Requested stack size in bytes (default 1 MiB).
    pub stack_size: u64,
    /// Guest-requested stack base. The current scheduler may allocate its own
    /// stack, but attribute getters must still round-trip the configured value.
    pub stack_address: u64,
    /// Guard-page size in bytes (default 4 KiB).
    pub guard_size: u64,
    /// Scheduling policy (default 1).
    pub sched_policy: i32,
    /// Scheduling priority.
    pub sched_priority: i32,
    /// SCE-specific "solo scheduler" flag (`scePthreadAttrSetsolosched`): the
    /// title asks that the thread run on a dedicated scheduling context. Pure
    /// bookkeeping — Raeen maps guest threads onto host threads, which are
    /// already independently scheduled, so the flag has no host action but is
    /// recorded so it reads back exactly as set.
    pub solo_sched: bool,
}

impl Default for PthreadAttr {
    fn default() -> Self {
        Self {
            detach_state: 0,
            stack_size: 0x10_0000,
            stack_address: 0,
            guard_size: 0x1000,
            sched_policy: 1,
            sched_priority: 700,
            solo_sched: false,
        }
    }
}

/// State of a guest pthread read-write lock. Both the guest pointer slot and
/// its opaque handle alias one shared host state object; `key` keeps per-thread
/// read ownership canonical across those aliases.
#[derive(Clone, Copy, Debug)]
pub struct PthreadRwlock {
    /// Canonical guest pointer-slot address for this logical rwlock.
    pub key: u64,
    /// Outstanding read-lock holds across all guest threads.
    pub readers: i32,
    /// Write-owning thread handle (0 = no writer).
    pub writer: u64,
    /// Write-lock recursion depth.
    pub writer_recursion: i32,
}

/// One guest rwlock's shared state plus the host condvar its HLE waiters
/// park on — the rwlock counterpart of [`PthreadMutexShared`], for the same
/// reason: a spinning waiter burns a full host core that the lock's own owner
/// needs to make progress. Notified whenever a writer releases or the last
/// reader drains.
pub struct PthreadRwlockShared {
    pub state: parking_lot::Mutex<PthreadRwlock>,
    pub released: parking_lot::Condvar,
}

impl PthreadRwlock {
    /// Create one shared state object for a guest rwlock and all its aliases.
    pub fn shared(key: u64) -> Arc<PthreadRwlockShared> {
        Arc::new(PthreadRwlockShared {
            state: parking_lot::Mutex::new(Self {
                key,
                readers: 0,
                writer: 0,
                writer_recursion: 0,
            }),
            released: parking_lot::Condvar::new(),
        })
    }

    /// Test-only: a shared rwlock pre-seeded with reader/writer state.
    #[cfg(test)]
    pub(crate) fn shared_for_test(
        key: u64,
        readers: i32,
        writer: u64,
        writer_recursion: i32,
    ) -> Arc<PthreadRwlockShared> {
        Arc::new(PthreadRwlockShared {
            state: parking_lot::Mutex::new(Self {
                key,
                readers,
                writer,
                writer_recursion,
            }),
            released: parking_lot::Condvar::new(),
        })
    }
}

/// How many locks of each kind [`OrbisKernel::release_locks_owned_by`] freed
/// from a dying thread. Reported in the fault-path log so the deadlock-cascade
/// recovery is visible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LockReleaseSummary {
    /// Mutexes whose owner was the dead thread.
    pub mutexes: usize,
    /// Rwlocks whose writer was the dead thread.
    pub rwlock_writers: usize,
    /// Rwlock read holds the dead thread still had.
    pub rwlock_read_holds: usize,
    /// Condition-variable wait-queue entries the dead thread left parked.
    ///
    /// Not a lock, but the same class of leak and the same consequence. A
    /// waiter is removed from the FIFO *by the signaler*, so a dead thread's
    /// entry stays queued: the next `scePthreadCondSignal` pops it, wakes a
    /// thread that will never look, and returns having "signalled one" — the
    /// live waiter behind it misses its wakeup. That is a lost wakeup per dead
    /// waiter, which is a deadlock with extra steps.
    pub cond_waiters: usize,
}

impl LockReleaseSummary {
    /// Whether any lock at all was released (worth logging).
    pub fn any(&self) -> bool {
        self.mutexes != 0
            || self.rwlock_writers != 0
            || self.rwlock_read_holds != 0
            || self.cond_waiters != 0
    }
}

/// State of a guest pthread mutex. The kernel map wraps this in one shared
/// `Arc<Mutex<_>>` per logical guest mutex because Orbis exposes both a pointer
/// slot and an opaque handle for the same object. Ported from SharpEmu's
/// `PthreadMutexState` (GPL-2.0). See the `raeen-hle` `pthread_sync` module for
/// the state machine.
#[derive(Debug)]
pub struct PthreadMutex {
    /// Mutex type: 1 = error-check, 2 = recursive, 3 = normal, 4 = adaptive.
    pub ty: i32,
    /// Owning thread handle (0 = unlocked).
    pub owner: u64,
    /// Lock recursion count (0 = unlocked).
    pub recursion: i32,
    /// Guest return address of the lock call that acquired ownership.
    ///
    /// Diagnostic only: a long-contention report can now identify the start
    /// of the critical section instead of sampling an arbitrary instruction
    /// several seconds later. Zero means unavailable or unlocked.
    pub owner_acquire_site: u64,
    /// Guest stack pointer at the lock import boundary. Read only after a
    /// long-contention threshold, while the owning critical-section frame is
    /// still live, to recover parent guest return addresses.
    pub owner_acquire_rsp: u64,
    /// Threads blocked on this mutex, oldest first.
    waiters: std::collections::VecDeque<PthreadMutexWaiter>,
}

#[derive(Debug)]
struct PthreadMutexWaiter {
    signal: Arc<GuestWaiter>,
    acquire_site: u64,
    acquire_rsp: u64,
}

/// One guest mutex's shared state and FIFO of private waiter signals.
/// Before host-backed parking existed, every blocked guest thread spun on
/// `yield_now()` at full host-CPU — measured on Minecraft's in-game
/// "Streaming Pool" workers, where 7 spinning waiters starved the very owner
/// they were waiting for (4 FPS in-world). Direct FIFO handoff now transfers
/// ownership under `state` and wakes only the selected waiter.
pub struct PthreadMutexShared {
    pub state: parking_lot::Mutex<PthreadMutex>,
}

impl PthreadMutex {
    /// Create the single shared state object that every guest-visible alias of
    /// this mutex must reference.
    pub fn shared(ty: i32) -> Arc<PthreadMutexShared> {
        Arc::new(PthreadMutexShared {
            state: parking_lot::Mutex::new(Self {
                ty,
                owner: 0,
                recursion: 0,
                owner_acquire_site: 0,
                owner_acquire_rsp: 0,
                waiters: std::collections::VecDeque::new(),
            }),
        })
    }

    /// Test-only: a shared mutex pre-seeded with an owner/recursion, for the
    /// owner-death recovery tests.
    #[cfg(test)]
    pub(crate) fn shared_for_test(ty: i32, owner: u64, recursion: i32) -> Arc<PthreadMutexShared> {
        Arc::new(PthreadMutexShared {
            state: parking_lot::Mutex::new(Self {
                ty,
                owner,
                recursion,
                owner_acquire_site: 0,
                owner_acquire_rsp: 0,
                waiters: std::collections::VecDeque::new(),
            }),
        })
    }

    /// Join the acquisition FIFO. A previous entry for the same guest thread
    /// is stale because one thread cannot wait on two acquisitions at once.
    #[must_use]
    pub fn enqueue_waiter(
        &mut self,
        thread: u64,
        acquire_site: u64,
        acquire_rsp: u64,
    ) -> Arc<GuestWaiter> {
        if thread != 0 {
            self.waiters.retain(|waiter| waiter.signal.thread != thread);
        }
        let waiter = Arc::new(GuestWaiter::new(thread));
        self.waiters.push_back(PthreadMutexWaiter {
            signal: Arc::clone(&waiter),
            acquire_site,
            acquire_rsp,
        });
        waiter
    }

    /// Remove a waiter that timed out or is terminating. `false` means a
    /// concurrent handoff already dequeued it, so the caller owns the mutex.
    pub fn cancel_waiter(&mut self, waiter: &Arc<GuestWaiter>) -> bool {
        let Some(position) = self
            .waiters
            .iter()
            .position(|candidate| Arc::ptr_eq(&candidate.signal, waiter))
        else {
            return false;
        };
        self.waiters.remove(position);
        true
    }

    /// Transfer a free mutex directly to the oldest waiter and wake only it.
    ///
    /// Ported from SharpEmu's GPL-2.0 `TryGrantMutexWaiterLocked` behavior.
    /// Ownership changes while the state lock is held, preventing arrivals
    /// from barging ahead of the selected waiter.
    pub fn try_grant_head(&mut self) -> Option<u64> {
        if self.owner != 0 {
            return None;
        }
        let waiter = self.waiters.pop_front()?;
        self.owner = waiter.signal.thread;
        self.recursion = 1;
        self.owner_acquire_site = waiter.acquire_site;
        self.owner_acquire_rsp = waiter.acquire_rsp;
        waiter.signal.wake();
        Some(waiter.signal.thread)
    }

    #[must_use]
    pub fn has_waiters(&self) -> bool {
        !self.waiters.is_empty()
    }

    #[must_use]
    pub fn waiter_count(&self) -> usize {
        self.waiters.len()
    }
}

/// Host-backed state for one POSIX (`sem_*`) semaphore: the available count
/// plus a condvar waiters sleep on until `sem_post` raises it. Keyed by the
/// guest `sem_t` address in [`OrbisKernel::posix_semaphores`].
#[derive(Debug, Default)]
pub struct PosixSem {
    /// Current available count (waiters sleep while it is zero).
    pub count: parking_lot::Mutex<i64>,
    /// Waiters sleep here until `sem_post` increments the count.
    pub posted: parking_lot::Condvar,
}

/// One parked guest thread — the single waiter primitive shared by every
/// blocking HLE synchronization object in the tree.
///
/// A waiter owns its wake bit so a wake-one can wake exactly one queued thread.
/// An object-wide generation counter cannot provide that contract: once
/// incremented, every waiter observes the new generation on its next bounded
/// host wake and a signal-one silently becomes a broadcast.
///
/// The private bit is also what closes the *unpark race*. A waiter registers
/// (under whichever queue lock owns it), then drops that lock before parking.
/// A wake landing inside that window sets `signaled` under the waiter's own
/// lock, so [`Self::wait_for_signal`] sees it already set and returns at once
/// instead of parking on a wake that will never come again.
///
/// Three users: [`PthreadCond`] (`scePthreadCondWait`), [`PthreadMutex`]'s
/// ownership-handoff FIFO (`scePthreadMutexUnlock`), and [`SyncAddressTable`]
/// (`sceKernelSyncOnAddressWait` / `sys_futex`).
#[derive(Debug)]
pub struct GuestWaiter {
    thread: u64,
    /// Lock-free mirror of `signaled`, stored **first** in [`Self::wake`].
    ///
    /// This is what [`Self::spin_for_signal`] polls: a freshly granted waiter
    /// observes its wake with a single atomic load instead of a host park /
    /// unpark round trip. It is a fast-path hint only — the mutex-protected
    /// `signaled` bool below remains the single source of truth for the park
    /// path, so the spin phase cannot introduce a lost wake: a waiter that
    /// gives up spinning falls into [`Self::wait_for_signal`], which re-checks
    /// `signaled` under its lock before ever sleeping.
    signaled_fast: std::sync::atomic::AtomicBool,
    signaled: parking_lot::Mutex<bool>,
    changed: parking_lot::Condvar,
}

impl GuestWaiter {
    fn new(thread: u64) -> Self {
        Self {
            thread,
            signaled_fast: std::sync::atomic::AtomicBool::new(false),
            signaled: parking_lot::Mutex::new(false),
            changed: parking_lot::Condvar::new(),
        }
    }

    fn wake(&self) {
        // Release-store the fast flag before taking the waiter's lock: a
        // spinning waiter that Acquire-loads it true is guaranteed to see
        // every write the waker made before waking (e.g. the mutex ownership
        // transfer `try_grant_head` performed under the state lock).
        self.signaled_fast
            .store(true, std::sync::atomic::Ordering::Release);
        *self.signaled.lock() = true;
        self.changed.notify_one();
    }

    /// End this waiter's current host park **without** granting it anything.
    ///
    /// The inverse of [`Self::wake`] in the one respect that matters: `signaled`
    /// stays clear, so [`Self::wait_for_signal`] returns `false` and the caller's
    /// slice loop re-runs its checks and parks again. Nothing about the guest's
    /// condition is claimed to have changed.
    ///
    /// This is how a reason *outside* the wait's own object — a queued Orbis
    /// exception for this thread — reaches a parked waiter promptly instead of
    /// waiting out its slice. Using `wake` for that would make the guest's
    /// `pthread_cond_wait` return as though the condition had been signalled.
    ///
    /// Safe to call on a waiter that is not parked: a `notify_one` with nobody
    /// waiting is a no-op, and the flag it does not touch cannot be lost.
    pub fn interrupt(&self) {
        self.changed.notify_one();
    }

    /// The guest thread handle this waiter parks on behalf of.
    #[must_use]
    pub fn thread(&self) -> u64 {
        self.thread
    }

    /// Whether this waiter has been selected by signal/broadcast.
    #[must_use]
    pub fn is_signaled(&self) -> bool {
        *self.signaled.lock()
    }

    /// Sleep for one bounded host slice, returning true only after a real
    /// condition-variable wake selected this waiter.
    #[must_use]
    pub fn wait_for_signal(&self, timeout: std::time::Duration) -> bool {
        let mut signaled = self.signaled.lock();
        if !*signaled {
            self.changed.wait_for(&mut signaled, timeout);
        }
        *signaled
    }

    /// Bounded busy-wait for this waiter's grant **before** falling back to
    /// [`Self::wait_for_signal`] — the adaptive-mutex spin phase.
    ///
    /// The measured problem this exists for: Minecraft's libc allocator lock
    /// is held for sub-microsecond critical sections at very high frequency
    /// (MAIN alone: 138k `scePthreadMutexLock` calls in ~15 s, ~5.3 s inside
    /// them). Under strict park-per-contention every handoff costs a host
    /// wakeup — microseconds to tens of microseconds — thousands of times per
    /// second, serialized through one FIFO. Spinning a few microseconds first
    /// lets a waiter observe the granting side's flag store without any kernel
    /// transition, exactly like glibc's `PTHREAD_MUTEX_ADAPTIVE_NP`.
    ///
    /// Returns `true` once the grant is observed; `false` when the budget is
    /// exhausted (the caller must then park — a wake landing in that
    /// transition is still observed by `wait_for_signal`'s locked flag check).
    /// The ladder: the first [`SPIN_BEFORE_YIELD`] iterations are pure
    /// `spin_loop` pauses; past that, every 64th iteration escalates to
    /// `yield_now` so an oversubscribed host makes progress.
    ///
    /// FIFO order, anti-barging, timeouts, and cancellation are untouched:
    /// spinning only changes *how* the already-selected waiter notices its
    /// grant, never *who* is selected.
    #[must_use]
    pub fn spin_for_signal(&self, budget: u32) -> bool {
        /// Iterations of pure `spin_loop` before the yield ladder kicks in.
        const SPIN_BEFORE_YIELD: u32 = 1024;
        for iteration in 0..budget {
            if self
                .signaled_fast
                .load(std::sync::atomic::Ordering::Acquire)
            {
                return true;
            }
            if iteration < SPIN_BEFORE_YIELD || !iteration.is_multiple_of(64) {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        self.signaled_fast
            .load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Default pre-park spin budget for contended guest locks, in
/// [`GuestWaiter::spin_for_signal`] iterations. ~2000 pause-hinted loads is on
/// the order of tens of microseconds on a Zen-class core — comparable to a
/// single park/unpark round trip, so the spin phase never costs meaningfully
/// more than the park it replaces, while a sub-microsecond malloc-class
/// critical section is caught in the first few iterations.
pub const DEFAULT_GUEST_WAITER_SPIN: u32 = 2000;

/// Parse an `RAEEN_MUTEX_SPIN` override. `0` disables spinning entirely
/// (restoring pure park-per-contention); unparseable values keep the default.
fn parse_spin_budget(raw: Option<&str>) -> u32 {
    raw.and_then(|value| value.trim().parse().ok())
        .unwrap_or(DEFAULT_GUEST_WAITER_SPIN)
}

/// The process-wide pre-park spin budget for guest lock waiters, read once
/// from `RAEEN_MUTEX_SPIN` (iterations; `0` disables spinning, absent/invalid
/// = [`DEFAULT_GUEST_WAITER_SPIN`]). Env-overridable so a live soak can A/B
/// spin budgets without a rebuild.
#[must_use]
pub fn guest_waiter_spin_budget() -> u32 {
    static BUDGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| parse_spin_budget(std::env::var("RAEEN_MUTEX_SPIN").ok().as_deref()))
}

/// Host-backed FIFO wait queue of [`GuestWaiter`]s.
///
/// The generic park/unpark queue behind `scePthreadCond*` and
/// `sceKernelSyncOnAddress*`. Wake selection completes *while holding the queue
/// lock*, which is what lets [`Self::cancel_waiter`] returning `false` mean
/// "a waker already took me, and my private wake bit is ready to observe" —
/// the timeout/wake race resolves without a second round of locking.
#[derive(Debug, Default)]
pub struct GuestWaitQueue {
    waiters: parking_lot::Mutex<std::collections::VecDeque<std::sync::Arc<GuestWaiter>>>,
}

impl GuestWaitQueue {
    /// Join the FIFO. The caller must publish itself here *before* releasing
    /// whatever lock guards the condition it is about to wait on, so no wake
    /// can slip into the gap.
    #[must_use]
    pub fn enqueue_waiter(&self, thread: u64) -> std::sync::Arc<GuestWaiter> {
        let waiter = std::sync::Arc::new(GuestWaiter::new(thread));
        self.waiters.lock().push_back(waiter.clone());
        waiter
    }

    /// Remove a waiter that timed out or whose process is terminating.
    ///
    /// False means a waker already removed it and, because wake selection is
    /// completed while holding the same queue lock, its private wake bit is
    /// ready to observe.
    pub fn cancel_waiter(&self, waiter: &std::sync::Arc<GuestWaiter>) -> bool {
        let mut waiters = self.waiters.lock();
        let Some(position) = waiters
            .iter()
            .position(|candidate| std::sync::Arc::ptr_eq(candidate, waiter))
        else {
            return false;
        };
        waiters.remove(position);
        true
    }

    /// Wake the oldest queued waiter, preserving wake-one semantics.
    pub fn signal_one(&self) -> bool {
        let mut waiters = self.waiters.lock();
        let Some(waiter) = waiters.pop_front() else {
            return false;
        };
        waiter.wake();
        true
    }

    /// Wake up to `count` waiters in FIFO order, returning how many were woken.
    /// `usize::MAX` is the wake-all spelling.
    pub fn signal_many(&self, count: usize) -> usize {
        let mut waiters = self.waiters.lock();
        let mut woken = 0;
        while woken < count {
            let Some(waiter) = waiters.pop_front() else {
                break;
            };
            waiter.wake();
            woken += 1;
        }
        woken
    }

    /// Wake the queued waiter belonging to one guest thread.
    pub fn signal_thread(&self, thread: u64) -> bool {
        let mut waiters = self.waiters.lock();
        let Some(position) = waiters.iter().position(|waiter| waiter.thread == thread) else {
            return false;
        };
        let waiter = waiters
            .remove(position)
            .expect("position came from the same locked queue");
        waiter.wake();
        true
    }

    /// Wake and remove every currently queued waiter.
    pub fn broadcast(&self) -> usize {
        let mut waiters = self.waiters.lock();
        let count = waiters.len();
        for waiter in waiters.drain(..) {
            waiter.wake();
        }
        count
    }

    /// End the host park of every queued waiter belonging to `thread` without
    /// signalling any of them, returning how many were interrupted.
    ///
    /// The entries **stay queued** — this is not a dequeue, and it does not
    /// consume a wake that a later `signal_one` owes to somebody. See
    /// [`GuestWaiter::interrupt`] for why that distinction is the whole point.
    pub fn interrupt_waiters_of(&self, thread: u64) -> usize {
        let waiters = self.waiters.lock();
        let mut interrupted = 0;
        for waiter in waiters.iter().filter(|waiter| waiter.thread == thread) {
            waiter.interrupt();
            interrupted += 1;
        }
        interrupted
    }

    /// Number of currently parked waiters, primarily for diagnostics/tests.
    #[must_use]
    pub fn waiter_count(&self) -> usize {
        self.waiters.lock().len()
    }

    /// Drop every entry belonging to a thread that no longer exists, returning
    /// how many were removed.
    ///
    /// Thread-death cleanup, not a wake: the entries are discarded, never
    /// signalled. Signalling a dead waiter is precisely the bug this prevents —
    /// [`Self::signal_one`] pops the oldest entry and considers the signal
    /// delivered, so one abandoned entry silently swallows one wakeup that a
    /// live waiter needed.
    pub fn remove_waiters_of(&self, thread: u64) -> usize {
        let mut waiters = self.waiters.lock();
        let before = waiters.len();
        waiters.retain(|waiter| waiter.thread != thread);
        before - waiters.len()
    }
}

/// Host-backed FIFO wait queue for one guest pthread condition: a
/// [`GuestWaitQueue`] plus the condition's chosen clock.
#[derive(Debug, Default)]
pub struct PthreadCond {
    queue: GuestWaitQueue,
    /// Which clock this condition's `pthread_cond_timedwait` deadlines are on,
    /// as chosen by `pthread_condattr_setclock` before `pthread_cond_init`.
    ///
    /// POSIX lets a condition variable pick its clock, and the deadline is
    /// meaningless without knowing which one: `CLOCK_MONOTONIC` counts from an
    /// arbitrary origin (here, process start) while the default
    /// `CLOCK_REALTIME` counts from the Unix epoch. Reading a monotonic
    /// deadline as a realtime one puts it ~1.78e9 seconds in the past, so every
    /// wait expires instantly and the caller spins.
    ///
    /// `false` (the `Default`) is `CLOCK_REALTIME` — the POSIX default, and the
    /// right answer for a statically-initialized cond that never saw an attr.
    pub monotonic: std::sync::atomic::AtomicBool,
}

impl PthreadCond {
    /// Join the condition's FIFO before the caller releases its guest mutex.
    #[must_use]
    pub fn enqueue_waiter(&self, thread: u64) -> std::sync::Arc<GuestWaiter> {
        self.queue.enqueue_waiter(thread)
    }

    /// Remove a waiter that timed out or whose process is terminating.
    ///
    /// False means a signaler already removed it and, because wake selection is
    /// completed while holding the same queue lock, its private wake bit is
    /// ready to observe.
    pub fn cancel_waiter(&self, waiter: &std::sync::Arc<GuestWaiter>) -> bool {
        self.queue.cancel_waiter(waiter)
    }

    /// Wake the oldest queued waiter, preserving pthread signal-one semantics.
    pub fn signal_one(&self) -> bool {
        self.queue.signal_one()
    }

    /// Wake the queued waiter belonging to one guest thread.
    pub fn signal_thread(&self, thread: u64) -> bool {
        self.queue.signal_thread(thread)
    }

    /// Wake and remove every currently queued waiter.
    pub fn broadcast(&self) -> usize {
        self.queue.broadcast()
    }

    /// End the host park of this condition's waiters belonging to `thread`
    /// without signalling them — see [`GuestWaitQueue::interrupt_waiters_of`].
    pub fn interrupt_waiters_of(&self, thread: u64) -> usize {
        self.queue.interrupt_waiters_of(thread)
    }

    /// Number of currently parked waiters, primarily for diagnostics/tests.
    #[must_use]
    pub fn waiter_count(&self) -> usize {
        self.queue.waiter_count()
    }

    /// Discard every entry belonging to a dead thread — see
    /// [`GuestWaitQueue::remove_waiters_of`]. Used by owner-death recovery: an
    /// abandoned entry would otherwise absorb a `signal_one` that a live waiter
    /// needed.
    pub fn remove_waiters_of(&self, thread: u64) -> usize {
        self.queue.remove_waiters_of(thread)
    }
}

/// The process-wide **address-keyed parking lot** behind libkernel's
/// `sceKernelSyncOnAddress{Wait,Wait32,Wait64,Wake}` and the `futex` syscall.
///
/// These are the PS5's futex primitives: a thread parks on a *guest address*
/// until another thread writes that address and wakes it. Guest runtimes build
/// their own spinlocks and work queues on top, and call the wait in a hot loop
/// — so a wait that returns success without ever parking turns the guest's
/// "block until the word changes" into a busy-spin with no forward progress.
///
/// One [`GuestWaitQueue`] is materialized per watched address on first wait, so
/// a wake reaches exactly the threads parked on that address (rather than a
/// process-wide broadcast). Queues are kept after they drain: a guest futex
/// address is reused for the lifetime of the object that owns it, and dropping
/// the queue would race a wait that has registered but not yet parked.
///
/// The **value compare** belongs to the caller, not here: it needs
/// [`GuestMemory`](crate::GuestMemory)-style access that this crate's kernel
/// state does not own. The caller must enqueue *first* and read the watched word
/// *second* (see the `raeen-hle` `libkernel` module) — that order is what makes
/// the compare race-free, because a waker that writes the word after our read
/// necessarily sees us already queued.
///
/// Ported from SharpEmu's `KernelSyncOnAddressCompatExports` (GPL-2.0): the
/// address-keyed park/wake shape and the bounded self-heal deadline are theirs;
/// the per-waiter FIFO replaces their per-address wake *generation* counter, and
/// the enqueue-then-compare ordering replaces the compare value they could not
/// recover.
#[derive(Debug, Default)]
pub struct SyncAddressTable {
    queues: DashMap<u64, Arc<GuestWaitQueue>>,
}

impl SyncAddressTable {
    /// The wait queue for `address`, created on first use.
    #[must_use]
    pub fn queue(&self, address: u64) -> Arc<GuestWaitQueue> {
        Arc::clone(
            self.queues
                .entry(address)
                .or_insert_with(|| Arc::new(GuestWaitQueue::default()))
                .value(),
        )
    }

    /// Park the calling guest thread on `address`. The returned waiter must be
    /// either woken or [`GuestWaitQueue::cancel_waiter`]-ed by the caller.
    #[must_use]
    pub fn enqueue(&self, address: u64, thread: u64) -> Arc<GuestWaiter> {
        self.queue(address).enqueue_waiter(thread)
    }

    /// Wake up to `count` waiters parked on `address` in FIFO order, returning
    /// how many were woken. `usize::MAX` is the wake-all spelling.
    ///
    /// An address nobody is parked on is not an error — a wake that arrives
    /// before its wait is the common uncontended case.
    pub fn wake(&self, address: u64, count: usize) -> usize {
        self.queues
            .get(&address)
            .map_or(0, |queue| queue.signal_many(count))
    }

    /// End the host park of every futex waiter belonging to `thread`, across
    /// every watched address, without signalling any of them — see
    /// [`GuestWaitQueue::interrupt_waiters_of`]. Returns how many were
    /// interrupted.
    ///
    /// The watched word is untouched and no entry leaves its queue, so a woken
    /// waiter re-runs its slice checks (including the queued-exception one) and
    /// parks again.
    pub fn interrupt_waiters_of(&self, thread: u64) -> usize {
        self.queues
            .iter()
            .map(|entry| entry.value().interrupt_waiters_of(thread))
            .sum()
    }

    /// How many threads are parked on `address` (diagnostics/tests).
    #[must_use]
    pub fn waiter_count(&self, address: u64) -> usize {
        self.queues
            .get(&address)
            .map_or(0, |queue| queue.waiter_count())
    }

    /// Number of distinct addresses that have ever been waited on.
    #[must_use]
    pub fn tracked_addresses(&self) -> usize {
        self.queues.len()
    }
}

#[cfg(test)]
mod pthread_cond_wait_queue_tests {
    use super::PthreadCond;

    #[test]
    fn signal_wakes_only_the_oldest_waiter() {
        let cond = PthreadCond::default();
        let first = cond.enqueue_waiter(11);
        let second = cond.enqueue_waiter(22);

        assert!(cond.signal_one());
        assert!(first.is_signaled());
        assert!(
            !second.is_signaled(),
            "one condition signal must not become a delayed broadcast"
        );
        assert_eq!(cond.waiter_count(), 1);

        assert!(cond.signal_one());
        assert!(second.is_signaled());
        assert_eq!(cond.waiter_count(), 0);
    }
}

#[cfg(test)]
mod pthread_mutex_handoff_tests {
    use super::PthreadMutex;

    const MUTEX_NORMAL: i32 = 3;

    #[test]
    fn release_hands_ownership_to_the_head_waiter_and_wakes_only_it() {
        let shared = PthreadMutex::shared(MUTEX_NORMAL);
        let mut state = shared.state.lock();
        state.owner = 0x100;
        state.recursion = 1;

        let first = state.enqueue_waiter(0x201, 0x1000_0201, 0x2000_0201);
        let second = state.enqueue_waiter(0x202, 0x1000_0202, 0x2000_0202);
        assert_eq!(state.waiter_count(), 2);
        assert_eq!(state.try_grant_head(), None);

        state.owner = 0;
        state.recursion = 0;
        assert_eq!(state.try_grant_head(), Some(0x201));
        assert_eq!(state.owner, 0x201);
        assert_eq!(state.recursion, 1);
        assert_eq!(
            state.owner_acquire_site, 0x1000_0201,
            "direct handoff must retain the selected waiter's guest call site"
        );
        assert_eq!(state.owner_acquire_rsp, 0x2000_0201);
        assert!(first.is_signaled());
        assert!(!second.is_signaled());
        assert_eq!(state.waiter_count(), 1);
    }

    #[test]
    fn cancelling_or_reenqueuing_cannot_leave_a_stale_fifo_head() {
        let shared = PthreadMutex::shared(MUTEX_NORMAL);
        let mut state = shared.state.lock();
        state.owner = 0x100;
        state.recursion = 1;

        let timed_out = state.enqueue_waiter(0x201, 0x1000_0201, 0x2000_0201);
        let stale = state.enqueue_waiter(0x202, 0x1000_dead, 0x2000_dead);
        let live = state.enqueue_waiter(0x202, 0x1000_0202, 0x2000_0202);
        assert!(state.cancel_waiter(&timed_out));
        assert_eq!(state.waiter_count(), 1);

        state.owner = 0;
        state.recursion = 0;
        assert_eq!(state.try_grant_head(), Some(0x202));
        assert!(!timed_out.is_signaled());
        assert!(!stale.is_signaled());
        assert!(live.is_signaled());
        assert_eq!(state.owner_acquire_site, 0x1000_0202);
        assert_eq!(state.owner_acquire_rsp, 0x2000_0202);
    }
}

/// Interrupting a parked waiter for a reason its wait object knows nothing about
/// — a queued Orbis exception — must never look like that object having been
/// signalled. These pin the distinction, which is the difference between a signal
/// reaching a blocked thread and a guest's `pthread_cond_wait` returning with its
/// predicate still false.
#[cfg(test)]
mod waiter_interrupt_tests {
    use super::{GuestWaitQueue, OrbisKernel, PendingException, PthreadCond, SyncAddressTable};
    use std::sync::Arc;

    #[test]
    fn interrupt_wakes_the_park_without_claiming_a_signal() {
        let queue = GuestWaitQueue::default();
        let mine = queue.enqueue_waiter(11);
        let other = queue.enqueue_waiter(22);

        assert_eq!(queue.interrupt_waiters_of(11), 1);
        assert!(
            !mine.is_signaled(),
            "the wake bit means 'the condition was signalled' — an interrupt must not set it"
        );
        assert!(!other.is_signaled());
        assert_eq!(
            queue.waiter_count(),
            2,
            "an interrupt is not a dequeue: both waiters keep their FIFO places, so no \
             later signal_one is stolen"
        );

        // A signal still works normally afterwards, and still selects the head.
        assert!(queue.signal_one());
        assert!(mine.is_signaled());
        assert!(!other.is_signaled());
        assert_eq!(queue.waiter_count(), 1);
    }

    #[test]
    fn interrupting_a_thread_with_no_waiters_is_a_no_op() {
        let queue = GuestWaitQueue::default();
        let waiter = queue.enqueue_waiter(11);
        assert_eq!(queue.interrupt_waiters_of(99), 0);
        assert!(!waiter.is_signaled());
        assert_eq!(queue.waiter_count(), 1);
    }

    #[test]
    fn futex_waiters_are_interrupted_across_every_watched_address() {
        let table = SyncAddressTable::default();
        let first = table.enqueue(0x1000, 7);
        let second = table.enqueue(0x2000, 7);
        let stranger = table.enqueue(0x2000, 8);

        assert_eq!(table.interrupt_waiters_of(7), 2);
        assert!(!first.is_signaled());
        assert!(!second.is_signaled());
        assert!(!stranger.is_signaled());
        assert_eq!(table.waiter_count(0x1000), 1);
        assert_eq!(table.waiter_count(0x2000), 2);
    }

    /// `pthread_conds` deliberately registers one condition under both the guest
    /// pointer and its opaque handle. A caller trusting the returned count must
    /// not be told there were twice as many waiters as exist.
    #[test]
    fn an_aliased_condition_is_visited_once() {
        let kernel = OrbisKernel::new();
        let cond = Arc::new(PthreadCond::default());
        let waiter = cond.enqueue_waiter(3);
        kernel.pthread_conds.insert(0x1000, Arc::clone(&cond));
        kernel.pthread_conds.insert(0x9000, cond);

        assert_eq!(kernel.interrupt_cond_waiters_of(3), 1);
        assert!(!waiter.is_signaled());
        assert_eq!(kernel.interrupt_cond_waiters_of(4), 0);
    }

    #[test]
    fn a_pending_exception_is_visible_only_to_its_own_target() {
        let kernel = OrbisKernel::new();
        assert!(!kernel.has_pending_exception_for(1));
        kernel.queue_pending_exception(
            1,
            PendingException {
                signum: 30,
                handler: 0x1000,
                raised_by: 2,
            },
        );
        assert!(kernel.has_pending_exception_for(1));
        assert!(
            !kernel.has_pending_exception_for(2),
            "the per-thread predicate must not degrade into the process-wide one — every \
             blocking wait in the tree would then release its lock and attempt a delivery \
             that is not for it"
        );
        assert!(kernel.discard_pending_exception(1));
        assert!(!kernel.has_pending_exception_for(1));
    }

    /// Waking semaphore slices must not invent counts. It is a pure "re-run your
    /// checks" notification, safe with nobody parked.
    #[test]
    fn notifying_semaphore_slices_changes_no_count() {
        let kernel = OrbisKernel::new();
        let handle = kernel.create_semaphore(2, 4);
        let posix = Arc::new(super::PosixSem::default());
        *posix.count.lock() = 5;
        kernel.posix_semaphores.insert(0x800, Arc::clone(&posix));

        kernel.notify_semaphore_slices();

        assert_eq!(kernel.kernel_semaphores.get(&handle).unwrap().count, 2);
        assert_eq!(*posix.count.lock(), 5);
    }
}

#[cfg(test)]
mod guest_waiter_spin_tests {
    use super::{DEFAULT_GUEST_WAITER_SPIN, GuestWaitQueue, PthreadMutex, parse_spin_budget};

    /// The spin→park transition regression: a grant that lands exactly after
    /// the spin gives up and before the park must not be lost. The zero-length
    /// timeout proves it is `wait_for_signal`'s locked flag check — not a
    /// condvar notification — that observes the wake.
    #[test]
    fn grant_landing_at_the_spin_to_park_transition_is_not_lost() {
        let queue = GuestWaitQueue::default();
        let waiter = queue.enqueue_waiter(7);

        // Spin exhausts its budget with no grant in sight.
        assert!(!waiter.spin_for_signal(64));

        // The grant lands in the spin→park window...
        assert!(queue.signal_one());

        // ...and the park path still observes it without ever sleeping.
        assert!(waiter.wait_for_signal(std::time::Duration::ZERO));
    }

    /// A wake completed on another host thread is visible to the very first
    /// spin iteration (Release store in `wake`, Acquire load in the spin).
    #[test]
    fn completed_wake_is_visible_to_the_first_spin_iteration() {
        let queue = GuestWaitQueue::default();
        let waiter = queue.enqueue_waiter(7);
        std::thread::scope(|scope| {
            scope
                .spawn(|| assert!(queue.signal_one()))
                .join()
                .expect("waker thread");
        });
        assert!(waiter.spin_for_signal(1));
        assert!(waiter.is_signaled(), "the park-path flag is set too");
    }

    /// The mutex FIFO's direct handoff is observable by a spinning waiter:
    /// `try_grant_head` transfers ownership under the state lock and its
    /// `wake` makes that transfer visible to the spin without any park.
    #[test]
    fn mutex_handoff_grant_is_observable_by_a_spinning_waiter() {
        const MUTEX_NORMAL: i32 = 3;
        let shared = PthreadMutex::shared(MUTEX_NORMAL);
        let mut state = shared.state.lock();
        state.owner = 0x100;
        state.recursion = 1;
        let waiter = state.enqueue_waiter(0x201, 0, 0);

        assert!(!waiter.spin_for_signal(16), "no grant while owned");

        state.owner = 0;
        state.recursion = 0;
        assert_eq!(state.try_grant_head(), Some(0x201));
        assert!(waiter.spin_for_signal(1));
        assert_eq!(state.owner, 0x201);
    }

    /// `RAEEN_MUTEX_SPIN` parsing: 0 disables, integers override, garbage and
    /// absence keep the default.
    #[test]
    fn spin_budget_env_parsing() {
        assert_eq!(parse_spin_budget(None), DEFAULT_GUEST_WAITER_SPIN);
        assert_eq!(parse_spin_budget(Some("0")), 0);
        assert_eq!(parse_spin_budget(Some(" 500 ")), 500);
        assert_eq!(
            parse_spin_budget(Some("not-a-number")),
            DEFAULT_GUEST_WAITER_SPIN
        );
        assert_eq!(parse_spin_budget(Some("-3")), DEFAULT_GUEST_WAITER_SPIN);
    }
}

/// Runtime-rebased ELF metadata used by `sceKernelGetModuleInfoForUnwind`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnwindModuleInfo {
    pub name: String,
    pub start: u64,
    pub end: u64,
    pub eh_frame_hdr_addr: u64,
    pub eh_frame_addr: u64,
    pub eh_frame_size: u64,
    pub seg0_addr: u64,
    pub seg0_size: u64,
}

/// A snapshot of the guest integer CPU context — enough to suspend a guest
/// fiber and later resume it executing NATIVELY on its own stack (SharpEmu's
/// `GuestCpuContinuation` model). The runtime's fiber module captures this from,
/// and applies it to, the Windows `CONTEXT` in the VEH handler; `rip`/`rsp` name
/// where and on which stack the guest resumes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuestRegs {
    pub rip: u64,
    pub rsp: u64,
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rflags: u64,
    pub mxcsr: u32,
    pub fpucw: u16,
}

/// Per host-thread fiber state.
#[derive(Clone, Copy, Debug, Default)]
pub struct FiberThreadState {
    /// The thread's resume snapshot — where its `sceFiberRun` call returns to.
    pub root: GuestRegs,
    /// The guest `SceFiber` address currently running on this thread (0 = none).
    pub current_fiber: u64,
    /// The `*arg_on_return` out-pointer the active `sceFiberRun` passed in `rdx`,
    /// written with the fiber's return value when it calls `sceFiberReturnToThread`.
    pub root_arg_slot: u64,
}

impl OrbisKernel {
    /// Explicit per-process resource ceilings. These are emulator safeguards,
    /// not claims about undocumented retail limits: guest-controlled handle
    /// creation must fail instead of growing host memory without bound.
    pub const MAX_EVENT_FLAGS: u32 = 4096;
    pub const MAX_OFFLINE_SOCKETS: u32 = 1024;
    pub const MAX_AUDIO_OUT_PORTS: u32 = 32;

    fn try_claim_slot(counter: &std::sync::atomic::AtomicU32, limit: u32) -> bool {
        counter
            .fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |live| (live < limit).then_some(live + 1),
            )
            .is_ok()
    }

    /// Release every pthread mutex currently owned by `thread`, returning how
    /// many were freed. Called when a guest thread is torn down mid-execution
    /// (e.g. `sceKernelDebugRaiseException`) so it does not leave mutexes
    /// locked forever — waiters on a dead owner would otherwise spin or block
    /// indefinitely. This is owner-death recovery, not a normal unlock: it
    /// clears ownership and recursion outright regardless of level.
    pub fn release_mutexes_owned_by(&self, thread: u64) -> usize {
        let mut released = 0;
        let mut visited = HashSet::new();
        for entry in &self.pthread_mutexes {
            let shared = entry.value();
            if !visited.insert(Arc::as_ptr(shared) as usize) {
                continue;
            }
            let mut state = shared.state.lock();
            if state.owner == thread {
                state.owner = 0;
                state.recursion = 0;
                state.owner_acquire_site = 0;
                state.owner_acquire_rsp = 0;
                released += 1;
                // Wake every parked waiter — the lock they were waiting for
                // just became available via owner death.
                state.try_grant_head();
            }
        }
        released
    }

    /// Record an Orbis exception raised at `target` for delivery at that
    /// thread's next HLE safe point, returning whether it replaced an
    /// already-queued (undelivered) one.
    ///
    /// Newest wins — see [`OrbisKernel::pending_exceptions`].
    pub fn queue_pending_exception(&self, target: u64, pending: PendingException) -> bool {
        let replaced = self.pending_exceptions.insert(target, pending).is_some();
        self.sync_pending_exception_count();
        replaced
    }

    /// Whether any thread has an exception awaiting delivery.
    ///
    /// The fast path every HLE dispatch takes: one relaxed atomic load, no map
    /// locking. See [`OrbisKernel::pending_exception_count`].
    pub fn has_pending_exceptions(&self) -> bool {
        self.pending_exception_count
            .load(std::sync::atomic::Ordering::Relaxed)
            != 0
    }

    /// Whether **this** thread has an exception awaiting delivery.
    ///
    /// Gated on the relaxed count first, so the common answer costs the same
    /// single atomic load as [`OrbisKernel::has_pending_exceptions`] and the map
    /// is only touched once a raise really is outstanding.
    ///
    /// Exists for the blocking waits: a wait that holds its own notification
    /// lock must decide *whether to release it* before attempting delivery, and
    /// the answer has to be available without releasing it first.
    pub fn has_pending_exception_for(&self, thread: u64) -> bool {
        self.has_pending_exceptions() && self.pending_exceptions.contains_key(&thread)
    }

    /// The live guest thread stack registered for `thread`, as `[base, top)`.
    ///
    /// The authoritative answer to "where is this thread's stack?", and the only
    /// one Raeen has: guest stacks are arena-owned, so they are in neither
    /// [`Self::memory`]'s region table nor any address range that distinguishes
    /// them from ordinary heap objects. `None` means the thread is unknown or
    /// already reaped.
    #[must_use]
    pub fn guest_stack_of(&self, thread: u64) -> Option<(u64, u64)> {
        let (base, top) = *self.guest_thread_stacks.get(&thread)?;
        (base < top).then_some((base, top))
    }

    /// The live guest thread stack that contains `addr`, as `[base, top)`.
    ///
    /// Lets the address-keyed queries a title uses to discover stack extents
    /// (`sceKernelIsStack`, `sceKernelVirtualQuery`) answer for an arena-owned
    /// stack instead of reporting "not mapped". Linear in the number of live
    /// guest threads, which is why only those two rare calls use it — never a
    /// per-call hot path.
    #[must_use]
    pub fn guest_stack_containing(&self, addr: u64) -> Option<(u64, u64)> {
        self.guest_thread_stacks.iter().find_map(|entry| {
            let (base, top) = *entry.value();
            (base < top && addr >= base && addr < top).then_some((base, top))
        })
    }

    /// Wake every thread parked in a **semaphore** wait — both the Orbis
    /// counting semaphores (`sceKernelWaitSema`) and the POSIX ones
    /// (`sem_wait`) — so each re-runs its per-slice checks now.
    ///
    /// Not a signal: no count is touched, so a woken waiter re-reads it, finds it
    /// unchanged and parks again. The point is the *other* per-slice checks —
    /// process teardown, and (the reason this exists) a queued Orbis exception,
    /// which would otherwise wait out the wait's 100 ms slice.
    pub fn notify_semaphore_slices(&self) {
        {
            let (lock, cvar) = &self.semaphore_signal;
            let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            cvar.notify_all();
        }
        // POSIX semaphores park per object on their own condvar, so there is no
        // process-wide notify to piggyback on. Bounded by the number of live
        // `sem_t`s, and only reached on a raise.
        for entry in self.posix_semaphores.iter() {
            entry.value().posted.notify_all();
        }
    }

    /// Interrupt — **without signalling** — every `pthread_cond` waiter parked on
    /// behalf of `thread`, returning how many were interrupted.
    ///
    /// A condition waiter parks on its own [`GuestWaiter`] with a 10 ms slice and
    /// treats its private wake bit as "the condition was signalled". Waking it
    /// with [`GuestWaitQueue::signal_thread`] would therefore be a *lie*: the
    /// wait would return `OK` to the guest as though `pthread_cond_signal` had
    /// run. [`GuestWaiter::interrupt`] leaves the bit clear, so the waiter merely
    /// returns from its host park early, re-runs its slice checks, and — finding
    /// no wake — parks again. That is what makes this safe to call for a reason
    /// the condition variable knows nothing about.
    ///
    /// Deduplicated by identity, because `pthread_conds` deliberately keys the
    /// *same* state under both the guest cond pointer and its allocated opaque
    /// handle (see `raeen-hle`'s `pthread_cond::condition`). Visiting a condition
    /// twice would interrupt harmlessly but report twice as many waiters as
    /// exist, which is exactly the kind of doubled count a caller would trust.
    pub fn interrupt_cond_waiters_of(&self, thread: u64) -> usize {
        let mut visited: Vec<*const PthreadCond> = Vec::new();
        let mut interrupted = 0;
        for entry in self.pthread_conds.iter() {
            let identity = Arc::as_ptr(entry.value());
            if visited.contains(&identity) {
                continue;
            }
            visited.push(identity);
            interrupted += entry.value().interrupt_waiters_of(thread);
        }
        interrupted
    }

    /// Claim the exception queued for `thread`, if any, marking the thread as
    /// *delivering* so a nested safe point inside the handler does not claim
    /// the next one.
    ///
    /// Returns `None` when nothing is queued **or** when this thread is already
    /// inside a handler. The caller must pair a `Some` with
    /// [`OrbisKernel::finish_exception_delivery`] or
    /// [`OrbisKernel::requeue_pending_exception`].
    pub fn claim_pending_exception(&self, thread: u64) -> Option<PendingException> {
        if !self.has_pending_exceptions() {
            return None;
        }
        if self.exception_delivery_active.contains_key(&thread) {
            return None;
        }
        let pending = self.pending_exceptions.remove(&thread)?.1;
        self.sync_pending_exception_count();
        self.exception_delivery_active.insert(thread, ());
        Some(pending)
    }

    /// Release the delivering mark [`OrbisKernel::claim_pending_exception`]
    /// took. Idempotent.
    pub fn finish_exception_delivery(&self, thread: u64) {
        self.exception_delivery_active.remove(&thread);
    }

    /// Put a claimed exception back because delivery could not be attempted
    /// (no guest-callback capability on this dispatch path), and clear the
    /// delivering mark.
    ///
    /// Requeues only if nothing newer arrived in the meantime — a fresher raise
    /// supersedes the one we failed to deliver.
    pub fn requeue_pending_exception(&self, thread: u64, pending: PendingException) {
        self.pending_exceptions.entry(thread).or_insert(pending);
        self.sync_pending_exception_count();
        self.exception_delivery_active.remove(&thread);
    }

    /// Drop every trace of `thread` from the exception machinery, returning
    /// whether an undelivered exception was discarded.
    ///
    /// Called on thread exit: an exception raised at a thread that then dies has
    /// nowhere to be delivered, and leaving the entry behind would keep
    /// [`OrbisKernel::has_pending_exceptions`] permanently true — turning the
    /// per-call fast path into a map lookup for the rest of the run.
    pub fn discard_pending_exception(&self, thread: u64) -> bool {
        let discarded = self.pending_exceptions.remove(&thread).is_some();
        self.sync_pending_exception_count();
        self.exception_delivery_active.remove(&thread);
        self.exception_contexts.remove(&thread);
        discarded
    }

    fn sync_pending_exception_count(&self) {
        self.pending_exception_count.store(
            self.pending_exceptions.len(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Release EVERY lock a dying `thread` held — mutex ownership, rwlock
    /// write-ownership, and rwlock read holds — returning how many of each were
    /// freed. Superset of [`release_mutexes_owned_by`].
    ///
    /// This is the fault-path companion to the guest-called
    /// `sceKernelDebugRaiseException` recovery. A worker thread that a
    /// host-detected fault tears down mid-execution (a VEH-trapped null/wild
    /// dereference that ends `dispatch::run` with `RuntimeError::Faulted`) never
    /// runs its C++ unlock/cleanup, so any lock it held stays held forever. Now
    /// that mutexes and rwlocks TRULY block (post the create-race fix), every
    /// waiter on such a lock hangs indefinitely — the measured
    /// "scePthreadMutexLock stuck >3s — deadlock" cascade on ASTRO.BOT (one
    /// worker faults holding a mutex, 22 others wedge behind it). Clearing the
    /// dead thread's locks lets the title limp on instead of deadlocking.
    ///
    /// Owner-death recovery, not a normal unlock: ownership and recursion are
    /// cleared outright regardless of level, and a read hold gives back its full
    /// share of the shared reader count.
    ///
    /// **Runs on every thread exit path**, not only the faulting one, and is
    /// idempotent: a thread that unlocked everything it held (the normal case)
    /// scans the maps, finds nothing owned, and reports an all-zero summary. It
    /// has to be unconditional — a guest thread that leaves a critical section
    /// via `scePthreadExit`, or one abandoned mid-call by cooperative process
    /// termination, ends in `Ok(Returned)`/`Ok(Exited)` having skipped its C++
    /// unlock just as thoroughly as a faulting one does.
    ///
    /// # What is *not* released, and why
    ///
    /// Kernel counting semaphores ([`Semaphore`]), POSIX semaphores
    /// ([`PosixSem`]) and event flags ([`EventFlag`]) carry **no owner** — a
    /// count and a bitmask, with no record of which thread took a unit or set a
    /// bit. There is nothing here to attribute to the dying thread, and
    /// inventing an owner would be wrong for the common producer/consumer use,
    /// where the waiter is never the one expected to post back. Recovering a
    /// dead thread's semaphore units would require the wait paths to keep a
    /// per-thread ledger of successful acquisitions (and event flags a per-thread
    /// record of bits set), which only has a defensible meaning for the
    /// mutex-shaped `initial == max == 1` usage. Left unimplemented deliberately;
    /// see `docs/sharpemu-port/veh-hardening.md`.
    pub fn release_locks_owned_by(&self, thread: u64) -> LockReleaseSummary {
        let mut summary = LockReleaseSummary::default();

        let mut visited = HashSet::new();
        for entry in &self.pthread_mutexes {
            let shared = entry.value();
            if !visited.insert(Arc::as_ptr(shared) as usize) {
                continue;
            }
            let mut state = shared.state.lock();
            if state.owner == thread {
                state.owner = 0;
                state.recursion = 0;
                state.owner_acquire_site = 0;
                state.owner_acquire_rsp = 0;
                summary.mutexes += 1;
                // Owner died holding this — wake the parked waiters.
                state.try_grant_head();
            }
        }

        // Read holds first: collect the dead thread's holds (removing them from
        // the per-thread depth map), then give each rwlock back the reader-count
        // share those holds contributed. `readers` is a single shared count, so
        // a hold left behind would keep a writer locked out forever.
        let mut dead_holds: Vec<(u64, u32)> = Vec::new();
        self.pthread_rwlock_read_holds
            .retain(|&(t, rwlock_key), &mut depth| {
                if t == thread {
                    dead_holds.push((rwlock_key, depth));
                    false
                } else {
                    true
                }
            });
        for (rwlock_key, depth) in dead_holds {
            if let Some(rw) = self
                .pthread_rwlocks
                .get(&rwlock_key)
                .map(|entry| Arc::clone(entry.value()))
            {
                let mut state = rw.state.lock();
                state.readers = state.readers.saturating_sub(depth as i32).max(0);
                let drained = state.readers == 0;
                drop(state);
                if drained {
                    // Last reader gone via owner death — a parked writer can
                    // now proceed.
                    rw.released.notify_all();
                }
            }
            summary.rwlock_read_holds += 1;
        }

        let mut visited = HashSet::new();
        for entry in &self.pthread_rwlocks {
            let shared = entry.value();
            if !visited.insert(Arc::as_ptr(shared) as usize) {
                continue;
            }
            let mut state = shared.state.lock();
            if state.writer == thread {
                state.writer = 0;
                state.writer_recursion = 0;
                summary.rwlock_writers += 1;
                drop(state);
                // Writer died holding it — wake every parked reader/writer.
                shared.released.notify_all();
            }
        }

        // Condition-variable queues last: a dead thread's entry is a stolen
        // wakeup, not a held lock, but it wedges the same waiters. Discarded,
        // never signalled (see `PthreadCond::remove_waiters_of`).
        let mut visited = HashSet::new();
        for entry in &self.pthread_conds {
            let cond = entry.value();
            if !visited.insert(Arc::as_ptr(cond) as usize) {
                continue;
            }
            summary.cond_waiters += cond.remove_waiters_of(thread);
        }

        summary
    }

    /// Create a new kernel instance with default configuration.
    pub fn new() -> Self {
        tracing::info!("Initializing Orbis kernel HLE");
        let filesystem = Arc::new(filesystem::VirtualFileSystem::new());
        Self {
            console: Console::new(),
            memory: Arc::new(memory::VirtualMemoryManager::new()),
            threads: Arc::new(threading::ThreadManager::new()),
            aio: aio::AioEngine::new(Arc::clone(&filesystem)),
            filesystem,
            diagnostics: Arc::new(DiagnosticRecorder::from_env()),
            unresolved_nid_calls: DashMap::new(),
            started_at: std::time::Instant::now(),
            modules: DashMap::new(),
            lle_module_exports: DashMap::new(),
            hle_export_addrs: DashMap::new(),
            next_module_id: RwLock::new(1),
            syscall_stats: DashMap::new(),
            thread_names: DashMap::new(),
            thread_priorities: DashMap::new(),
            thread_sched_policies: DashMap::new(),
            host_thread_handles: DashMap::new(),
            guest_thread_stacks: DashMap::new(),
            recent_hle_calls: DashMap::new(),
            in_flight_hle: DashMap::new(),
            proc_param_addr: std::sync::atomic::AtomicU64::new(0),
            process_argc: std::sync::atomic::AtomicU64::new(0),
            process_argv: std::sync::atomic::AtomicU64::new(0),
            unwind_modules: RwLock::new(Vec::new()),
            pad_state: parking_lot::Mutex::new(None),
            pad_rumble: parking_lot::Mutex::new((0, 0, 0)),
            user_service_login_event_delivered: std::sync::atomic::AtomicBool::new(false),
            pthread_mutexes: DashMap::new(),
            pthread_mutex_attrs: DashMap::new(),
            sync_addresses: SyncAddressTable::default(),
            pthread_rwlocks: DashMap::new(),
            pthread_rwlock_read_holds: DashMap::new(),
            hle_call_time: DashMap::new(),
            hle_call_counts: DashMap::new(),
            pthread_conds: DashMap::new(),
            pthread_condattr_clocks: DashMap::new(),
            pthread_attrs: DashMap::new(),
            exception_handlers: DashMap::new(),
            pending_exceptions: DashMap::new(),
            pending_exception_count: std::sync::atomic::AtomicUsize::new(0),
            exception_delivery_active: DashMap::new(),
            exception_contexts: DashMap::new(),
            kernel_event_flags: DashMap::new(),
            kernel_event_flag_next: std::sync::atomic::AtomicU64::new(1),
            kernel_event_flag_live: std::sync::atomic::AtomicU32::new(0),
            kernel_semaphores: DashMap::new(),
            save_data_memory: DashMap::new(),
            kernel_semaphore_next: std::sync::atomic::AtomicU32::new(1),
            posix_semaphores: DashMap::new(),
            kernel_equeues: DashMap::new(),
            kernel_equeue_events: DashMap::new(),
            kernel_equeue_next: std::sync::atomic::AtomicU64::new(1),
            appr_files: DashMap::new(),
            appr_file_handles: DashMap::new(),
            np_state_callbacks: parking_lot::Mutex::new(Vec::new()),
            np_auth_requests: DashMap::new(),
            np_auth_next_request: std::sync::atomic::AtomicI32::new(0x1000_0000),
            appr_missing_warned: DashMap::new(),
            appr_submissions: DashMap::new(),
            appr_next_submission: std::sync::atomic::AtomicU32::new(1),
            kernel_epolls: DashMap::new(),
            kernel_epoll_next: std::sync::atomic::AtomicU32::new(1),
            agc_default_owner: std::sync::atomic::AtomicU32::new(1),
            agc_resource_registration_initialized: std::sync::atomic::AtomicBool::new(false),
            agc_resource_registration_max_owners: std::sync::atomic::AtomicU32::new(0),
            agc_next_owner: std::sync::atomic::AtomicU32::new(1),
            agc_resource_owners: DashMap::new(),
            agc_next_resource: std::sync::atomic::AtomicU32::new(1),
            agc_resources: DashMap::new(),
            agc_submission_count: std::sync::atomic::AtomicU64::new(0),
            agc_draw_packet_count: std::sync::atomic::AtomicU64::new(0),
            agc_dispatch_packet_count: std::sync::atomic::AtomicU64::new(0),
            agc_flip_packet_count: std::sync::atomic::AtomicU64::new(0),
            agc_gpu_timestamp: std::sync::atomic::AtomicU64::new(0),
            agc_last_dcb_address: std::sync::atomic::AtomicU64::new(0),
            agc_last_dcb_dwords: std::sync::atomic::AtomicU32::new(0),
            agc_register_defaults_addr: std::sync::atomic::AtomicU64::new(0),
            agc_pending_graphics_segment: std::sync::Mutex::new(
                AgcPendingGraphicsSegment::default(),
            ),
            video_out_buffers: DashMap::new(),
            direct_memory_allocated: std::sync::atomic::AtomicU64::new(0),
            video_out_flip_count: std::sync::atomic::AtomicU64::new(0),
            video_out_vblank_count: std::sync::atomic::AtomicU64::new(0),
            video_out_last_flip_arg: std::sync::atomic::AtomicI64::new(0),
            video_out_current_buffer: std::sync::atomic::AtomicI32::new(0),
            libc_mspaces: DashMap::new(),
            libc_mspace_allocations: DashMap::new(),
            event_flag_signal: (std::sync::Mutex::new(()), std::sync::Condvar::new()),
            semaphore_signal: (std::sync::Mutex::new(()), std::sync::Condvar::new()),
            semaphore_wakes: std::sync::atomic::AtomicU64::new(0),
            ampr_write_offsets: DashMap::new(),
            ampr_command_counts: DashMap::new(),
            ampr_gather_scatter: DashMap::new(),
            ampr_type_flags: DashMap::new(),
            fiber_context_size_check: std::sync::atomic::AtomicU32::new(0),
            fibers: DashMap::new(),
            fiber_threads: DashMap::new(),
            kernel_sockets: DashMap::new(),
            kernel_socket_next: std::sync::atomic::AtomicI32::new(0x4000_0000),
            kernel_socket_live: std::sync::atomic::AtomicU32::new(0),
            pthread_tls_keys: DashMap::new(),
            pthread_tls_values: DashMap::new(),
            dynamic_tls_blocks: DashMap::new(),
            static_tls_area_offsets: DashMap::new(),
            errno_slots: DashMap::new(),
            save_data_transaction_resources: DashMap::new(),
            save_data_next_transaction_resource: std::sync::atomic::AtomicI32::new(0),
            save_data_params: DashMap::new(),
            thread_dtors_callback: std::sync::atomic::AtomicU64::new(0),
            thread_atexit_count_callback: std::sync::atomic::AtomicU64::new(0),
            thread_atexit_report_callback: std::sync::atomic::AtomicU64::new(0),
            application_heap_api: std::sync::atomic::AtomicU64::new(0),
            libc_trace_storage: std::sync::atomic::AtomicU64::new(0),
            coredump_handler: std::sync::atomic::AtomicU64::new(0),
            coredump_handler_context: std::sync::atomic::AtomicU64::new(0),
            audio_out_ports: DashMap::new(),
            audio_out_next_port: std::sync::atomic::AtomicU32::new(1),
            audio_out_live_ports: std::sync::atomic::AtomicU32::new(0),
            text_to_speech2_initialized: std::sync::atomic::AtomicBool::new(false),
            pthread_tls_next_key: std::sync::atomic::AtomicI32::new(0),
            http_contexts: DashMap::new(),
            http_next_context: std::sync::atomic::AtomicI32::new(0),
            http_templates: DashMap::new(),
            http_next_template: std::sync::atomic::AtomicI32::new(0x1000),
            http2_contexts: DashMap::new(),
            http2_next_context: std::sync::atomic::AtomicI32::new(0),
            http2_templates: DashMap::new(),
            http2_next_template: std::sync::atomic::AtomicI32::new(0x1000),
            ssl_contexts: DashMap::new(),
            ssl_next_context: std::sync::atomic::AtomicI32::new(0),
        }
    }

    /// Wall-clock elapsed since this kernel instance was created (= since the
    /// title launch began). Used by diagnostics that split measurements into a
    /// boot window and a steady-state window.
    pub fn uptime(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }

    /// Claim the process's initial UserService login event.
    ///
    /// Returns `true` for exactly one caller. If that caller cannot write the
    /// event to guest memory it must call
    /// [`restore_initial_user_login_event`](Self::restore_initial_user_login_event)
    /// so a later valid request can consume it.
    pub fn claim_initial_user_login_event(&self) -> bool {
        self.user_service_login_event_delivered
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
    }

    /// Make a failed initial-login delivery available to the next caller.
    pub fn restore_initial_user_login_event(&self) {
        self.user_service_login_event_delivered
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Replace the loaded-process unwind table. Entries with empty or
    /// inverted ranges are discarded so address lookup remains unambiguous.
    pub fn set_unwind_modules(&self, modules: Vec<UnwindModuleInfo>) {
        let mut table = self.unwind_modules.write();
        *table = modules
            .into_iter()
            .filter(|module| module.start < module.end)
            .collect();
    }

    /// Register the process's static TLS layout: for each TLS module id, the
    /// offset of its block within the per-thread static area (see
    /// [`Self::static_tls_area_offsets`]). Called once at launch, replacing
    /// any previous registration.
    pub fn set_static_tls_area_offsets(&self, offsets: impl IntoIterator<Item = (u64, u64)>) {
        self.static_tls_area_offsets.clear();
        for (module_id, offset) in offsets {
            self.static_tls_area_offsets.insert(module_id, offset);
        }
    }

    /// Find the ELF owning `addr`, using half-open load ranges.
    pub fn unwind_module_for_addr(&self, addr: u64) -> Option<UnwindModuleInfo> {
        self.unwind_modules
            .read()
            .iter()
            .find(|module| module.start <= addr && addr < module.end)
            .cloned()
    }

    /// Record one call through an unresolved import.
    ///
    /// Returns `(first_occurrence, count)`. The caller uses the first bit to
    /// emit one structured warning per distinct compatibility gap while the
    /// count preserves hot-call information for the process-exit inventory.
    pub fn record_unresolved_nid_call(
        &self,
        nid: u64,
        function: &str,
        library: &str,
        calling_module: &str,
    ) -> (bool, u64) {
        let mut entry = self
            .unresolved_nid_calls
            .entry((
                nid,
                function.to_owned(),
                library.to_owned(),
                calling_module.to_owned(),
            ))
            .or_insert(0);
        *entry = entry.saturating_add(1);
        (*entry == 1, *entry)
    }

    /// The process-local unresolved-call inventory as sorted, formatted
    /// lines — one per distinct `(nid, function, library, caller)`, with call
    /// counts. Consumed by the crash-report assembly as well as the log dump
    /// below, so both say exactly the same thing.
    pub fn unresolved_nid_inventory(&self) -> Vec<String> {
        let mut inventory = self
            .unresolved_nid_calls
            .iter()
            .map(|entry| {
                let ((nid, function, library, calling_module), count) = entry.pair();
                format!(
                    "{nid:#018x} {function} library={library} caller={calling_module} calls={count}"
                )
            })
            .collect::<Vec<_>>();
        inventory.sort();
        inventory
    }

    /// Emit the complete process-local unresolved-call inventory as one
    /// deterministic event. First-occurrence warnings remain useful when a
    /// timed compatibility run is externally terminated before clean teardown.
    pub fn log_unresolved_nid_inventory(&self) {
        let inventory = self.unresolved_nid_inventory();
        if !inventory.is_empty() {
            tracing::warn!(
                entries = inventory.len(),
                inventory = ?inventory,
                "UNRESOLVED NID INVENTORY: default fail-soft calls observed"
            );
        }
    }

    /// Create an event flag with `attributes` and `initial_bits`, returning its
    /// handle. See the `raeen-hle` `kernel_eventflag` module.
    pub fn create_event_flag(&self, attributes: u32, initial_bits: u64) -> Option<u64> {
        if !Self::try_claim_slot(&self.kernel_event_flag_live, Self::MAX_EVENT_FLAGS) {
            return None;
        }
        let handle = self
            .kernel_event_flag_next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.kernel_event_flags.insert(
            handle,
            EventFlag {
                bits: initial_bits,
                attributes,
            },
        );
        Some(handle)
    }

    /// Create a counting semaphore with `initial`/`max` count, returning its
    /// handle. See the `raeen-hle` `kernel_semaphore` module.
    pub fn create_semaphore(&self, initial: i32, max: i32) -> u32 {
        let handle = self
            .kernel_semaphore_next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.kernel_semaphores.insert(
            handle,
            Semaphore {
                count: initial,
                max_count: max,
            },
        );
        handle
    }

    /// Wake every thread parked in `sceKernelWaitSema` so each re-checks its own
    /// count, and count the wake.
    ///
    /// The single funnel for semaphore notifications. It exists because the
    /// three producers had three copies of "lock, notify_all" and one of them —
    /// `DeleteSema` — simply did not have it, leaving waiters to discover the
    /// deletion when their internal 100 ms slice next expired. A counted funnel
    /// makes that class of omission provable in a test rather than only
    /// observable as latency.
    ///
    /// Notifying while holding the lock is what closes the check-then-sleep
    /// race: `hle_wait` re-checks the count under this same lock, so a notify
    /// can only land either before that check (seen) or after the waiter has
    /// atomically released the lock into `wait_timeout` (delivered).
    pub fn wake_semaphore_waiters(&self) {
        let (lock, cvar) = &self.semaphore_signal;
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        cvar.notify_all();
        self.semaphore_wakes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// How many semaphore wakes this process has issued.
    ///
    /// Diagnostic, and the deterministic seam the wait-slice audit tests use: a
    /// producer that does not appear here leaves its waiters on the slice.
    #[must_use]
    pub fn semaphore_wake_count(&self) -> u64 {
        self.semaphore_wakes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Create an event queue with `attributes`, returning its handle. See the
    /// `raeen-hle` `kernel_equeue` module.
    pub fn create_equeue(&self, attributes: u32) -> u64 {
        let handle = self
            .kernel_equeue_next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.kernel_equeues.insert(handle, attributes);
        handle
    }

    /// FNV-1a id of a guest path — the deterministic APR file id, matching
    /// SharpEmu's `AmprFileRegistry.ComputeFileId`.
    pub fn appr_file_id(guest_path: &str) -> u32 {
        let mut hash: u32 = 2_166_136_261;
        for &b in guest_path.as_bytes() {
            hash ^= u32::from(b);
            hash = hash.wrapping_mul(16_777_619);
        }
        hash
    }

    /// Register a resolved guest→host path pair, returning its deterministic
    /// id. A later AMPR read-by-id looks the file up here.
    pub fn appr_register_file(&self, guest_path: &str, host_path: String) -> u32 {
        let id = Self::appr_file_id(guest_path);
        self.appr_files.insert(id, host_path);
        id
    }

    /// The host path an APR id resolved to, if any.
    pub fn appr_host_path(&self, id: u32) -> Option<String> {
        self.appr_files.get(&id).map(|p| p.clone())
    }

    /// Allocate an APR command-buffer submission id.
    pub fn appr_add_submission(&self, command_buffer: u64) -> u32 {
        let id = self
            .appr_next_submission
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.appr_submissions.insert(id, command_buffer);
        id
    }

    /// Allocate a fresh (offline) socket fd. See the `raeen-hle`
    /// `kernel_socket` module.
    pub fn create_socket(&self) -> Option<i32> {
        if !Self::try_claim_slot(&self.kernel_socket_live, Self::MAX_OFFLINE_SOCKETS) {
            return None;
        }
        let fd = self
            .kernel_socket_next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.kernel_sockets.insert(fd, GuestSocket::default());
        Some(fd)
    }

    /// Close an offline socket and release one descriptor-table slot.
    pub fn close_socket(&self, fd: i32) -> bool {
        if self.kernel_sockets.remove(&fd).is_none() {
            return false;
        }
        self.kernel_socket_live
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        true
    }

    /// Open one process-owned AudioOut pacing port.
    pub fn open_audio_out_port(&self, grain: u32, frequency: u32) -> Option<u32> {
        if !Self::try_claim_slot(&self.audio_out_live_ports, Self::MAX_AUDIO_OUT_PORTS) {
            return None;
        }
        let handle = self
            .audio_out_next_port
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.audio_out_ports.insert(handle, (grain, frequency));
        Some(handle)
    }

    /// Return pacing parameters for a live AudioOut port.
    pub fn audio_out_port(&self, handle: u32) -> Option<(u32, u32)> {
        self.audio_out_ports.get(&handle).map(|port| *port)
    }

    /// Close a process-owned AudioOut port and release its quota slot.
    pub fn close_audio_out_port(&self, handle: u32) -> bool {
        if self.audio_out_ports.remove(&handle).is_none() {
            return false;
        }
        self.audio_out_live_ports
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        true
    }

    /// Allocate a fresh (offline) epoll id with an empty registration set.
    pub fn create_epoll(&self) -> u32 {
        let id = self
            .kernel_epoll_next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.kernel_epolls.insert(id, Vec::new());
        id
    }

    /// Allocate a fresh pthread TLS key registered with `destructor` (0 = none),
    /// returning its id. See the `raeen-hle` `pthread_tls` module.
    pub fn pthread_key_create(&self, destructor: u64) -> i32 {
        let key = self
            .pthread_tls_next_key
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.pthread_tls_keys.insert(key, destructor);
        key
    }

    /// Push the current controller state (a 12-byte Orbis `ScePadData` input
    /// prefix) from the host, for `scePadReadState` to report. Called each
    /// frame by the Shell from its `InputManager`.
    pub fn set_pad_state(&self, data: [u8; 12]) {
        *self.pad_state.lock() = Some(data);
    }

    /// The current controller snapshot, or `None` if the host hasn't pushed
    /// live input yet (the HLE then reports a neutral pad).
    pub fn pad_state(&self) -> Option<[u8; 12]> {
        *self.pad_state.lock()
    }

    /// Record a guest `scePadSetVibration` request. Every call bumps the
    /// sequence — even with unchanged motor values — so the host can tell a
    /// refreshed vibration from a stale one (the safety auto-stop keys off
    /// this freshness). Called by the HLE `hle_pad_set_vibration`.
    pub fn set_pad_rumble(&self, large: u8, small: u8) {
        let mut rumble = self.pad_rumble.lock();
        rumble.0 = rumble.0.wrapping_add(1);
        rumble.1 = large;
        rumble.2 = small;
    }

    /// The newest vibration request as `(sequence, largeMotor, smallMotor)`.
    /// Sequence 0 means no title ever called `scePadSetVibration`.
    pub fn pad_rumble(&self) -> (u64, u8, u8) {
        *self.pad_rumble.lock()
    }

    /// Record the guest address of the main module's process-parameter block
    /// (see [`proc_param_addr`](Self::proc_param_addr)).
    pub fn set_proc_param_addr(&self, addr: u64) {
        self.proc_param_addr
            .store(addr, std::sync::atomic::Ordering::Relaxed);
    }

    /// The guest address `sceKernelGetProcParam` returns, or `0` if no
    /// `PT_SCE_PROCPARAM` was present in the loaded module.
    pub fn proc_param_addr(&self) -> u64 {
        self.proc_param_addr
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record the argc value and guest `char **argv` table built for `_start`.
    pub fn set_process_args(&self, argc: u64, argv: u64) {
        self.process_argv
            .store(argv, std::sync::atomic::Ordering::Release);
        self.process_argc
            .store(argc, std::sync::atomic::Ordering::Release);
    }

    /// Return the process entry arguments, or `(0, 0)` before process setup.
    pub fn process_args(&self) -> (u64, u64) {
        (
            self.process_argc.load(std::sync::atomic::Ordering::Acquire),
            self.process_argv.load(std::sync::atomic::Ordering::Acquire),
        )
    }

    /// Register a loaded module with the kernel.
    pub fn register_module(&self, mut info: ModuleInfo) -> u32 {
        let mut next_id = self.next_module_id.write();
        let id = *next_id;
        info.id = id;
        *next_id += 1;
        tracing::info!(
            "Registered module: id={}, name='{}', base={:#x}, size={:#x}",
            id,
            info.name,
            info.base_address,
            info.size
        );
        self.modules.insert(id, info);
        id
    }

    /// Look up a module by name.
    pub fn find_module(&self, name: &str) -> Option<ModuleInfo> {
        self.modules
            .iter()
            .find(|entry| entry.value().name == name)
            .map(|entry| entry.value().clone())
    }

    /// Register or update a preplaced real module and its callable exports.
    pub fn register_lle_module(
        &self,
        name: String,
        base_address: u64,
        size: u64,
        entry_point: Option<u64>,
        initialized: bool,
        exports: impl IntoIterator<Item = (u64, u64)>,
    ) -> u32 {
        let id = self.find_module(&name).map_or_else(
            || {
                self.register_module(ModuleInfo {
                    id: 0,
                    name,
                    base_address,
                    size,
                    entry_point,
                    initialized,
                })
            },
            |module| module.id,
        );
        self.lle_module_exports
            .insert(id, exports.into_iter().collect());
        id
    }

    /// Mark a loaded module initialized after its `DT_INIT` callback has been
    /// accepted for execution.
    pub fn mark_module_initialized(&self, id: u32) {
        if let Some(mut module) = self.modules.get_mut(&id) {
            module.initialized = true;
        }
    }

    /// Resolve one export from a real module handle.
    pub fn resolve_lle_export(&self, handle: u32, nid: u64) -> Option<u64> {
        self.lle_module_exports
            .get(&handle)
            .and_then(|exports| exports.get(&nid).copied())
    }

    /// How many exports a module handle carries, or `None` if the handle names
    /// no registered module at all.
    ///
    /// Purely diagnostic, and the distinction is the whole point: a failed
    /// `sceKernelDlsym` against a handle with *zero* exports means the module
    /// was never wired up, while the same failure against a handle with *many*
    /// means the symbol genuinely is not in its export table — two completely
    /// different bugs that are indistinguishable from an `ENOENT` alone.
    pub fn lle_export_count(&self, handle: u32) -> Option<usize> {
        self.lle_module_exports.get(&handle).map(|e| e.len())
    }

    /// The handle of the **main program**.
    ///
    /// Module ids are handed out from 1 upward by [`Self::register_module`], in
    /// load order, and the executable is registered first — so the lowest id
    /// carrying an export table is the main program. `sceKernelDlsym(0, ...)`
    /// resolves against it: handle 0 is the Orbis rtld's reserved id for the
    /// executable, **not** a "search everything" scope (`RTLD_DEFAULT`) and not
    /// an invalid handle.
    #[must_use]
    pub fn main_lle_module_handle(&self) -> Option<u32> {
        self.lle_module_exports
            .iter()
            .map(|entry| *entry.key())
            .min()
    }

    /// Resolve `nid` against **every** loaded module, in load order (ascending
    /// handle), returning the first match as `(handle, address)`.
    ///
    /// The load-order walk is what makes this deterministic: `lle_module_exports`
    /// is a `DashMap` whose iteration order is arbitrary, and two modules may
    /// legally export the same NID, so an unordered "first hit" would resolve
    /// the same symbol to different addresses on different runs.
    #[must_use]
    pub fn resolve_lle_export_anywhere(&self, nid: u64) -> Option<(u32, u64)> {
        let mut handles: Vec<u32> = self
            .lle_module_exports
            .iter()
            .map(|entry| *entry.key())
            .collect();
        handles.sort_unstable();
        handles.into_iter().find_map(|handle| {
            self.resolve_lle_export(handle, nid)
                .map(|addr| (handle, addr))
        })
    }

    /// Publish one process-wide HLE trampoline address under the function name
    /// it stands for, so `sceKernelDlsym` can hand the guest a callable address
    /// for an HLE-implemented function.
    ///
    /// First registration wins. Two libraries may export the same function name
    /// under different NIDs (`libkernel::stat` and `libScePosix::stat`), and
    /// both trampolines dispatch to the same implementation, so which one a
    /// name-keyed lookup returns does not matter — but it must not *change*
    /// between calls.
    pub fn register_hle_export_addr(&self, function: &str, addr: u64) {
        self.hle_export_addrs
            .entry(function.to_string())
            .or_insert(addr);
    }

    /// The guest-callable trampoline address for an HLE-implemented function
    /// name, if this process reserved one.
    #[must_use]
    pub fn resolve_hle_export_addr(&self, function: &str) -> Option<u64> {
        self.hle_export_addrs.get(function).map(|entry| *entry)
    }

    /// How many HLE trampolines this process published to `dlsym`. Diagnostic:
    /// zero means the runtime never called [`Self::register_hle_export_addr`],
    /// which is a wiring bug rather than a missing symbol.
    #[must_use]
    pub fn hle_export_addr_count(&self) -> usize {
        self.hle_export_addrs.len()
    }

    /// Dispatch a syscall.
    pub fn dispatch_syscall(
        &self,
        number: u64,
        args: &[u64],
    ) -> Result<u64, raeen_core::error::KernelError> {
        // Track syscall statistics.
        self.syscall_stats
            .entry(number)
            .and_modify(|count| *count += 1)
            .or_insert(1);

        syscalls::dispatch(self, number, args)
    }
}

impl Default for OrbisKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl raeen_core::subsystems::TimeSubsystem for OrbisKernel {
    fn monotonic_elapsed(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }

    fn wall_clock(&self) -> std::time::SystemTime {
        std::time::SystemTime::now()
    }

    /// Every guest `usleep`/`nanosleep`/`sleep` lands here, so the host
    /// strategy is chosen in exactly one place.
    ///
    /// This was `std::thread::sleep`, whose Windows behaviour is unspecified and
    /// has changed across toolchains (it was a plain `Sleep()`, i.e. the ~15.6 ms
    /// system tick, before Rust ~1.79). `raeen_core::host_sleep` owns the
    /// strategy instead — a bounded `PAUSE` spin below 100 µs, a high-resolution
    /// waitable timer above it, and `std::thread::sleep` only where neither is
    /// available — with a requested-vs-actual histogram behind
    /// `RAEEN_TIME_SLEEP`. Never returns before `duration` has elapsed.
    fn sleep(&self, duration: std::time::Duration) {
        raeen_core::host_sleep::sleep(duration);
    }
}

impl raeen_core::subsystems::WaitSubsystem for OrbisKernel {
    fn wait_until(
        &self,
        key: raeen_core::subsystems::WaitKey,
        timeout: std::time::Duration,
        terminating: &dyn Fn() -> bool,
        ready: &mut dyn FnMut() -> bool,
    ) -> raeen_core::subsystems::WaitOutcome {
        use raeen_core::diagnostics::DiagnosticKind;
        use raeen_core::subsystems::WaitOutcome;

        // The park below is `Condvar::wait_timeout`, whose timeout is rounded to
        // the Windows system tick: measured, a 100 us wait costs 15 394 us and a
        // 1 ms wait 15 564 us. That quantisation is the single largest source of
        // lost time in the kernel wait paths, and raising the process timer
        // resolution is the only lever for it — a waitable timer cannot help
        // because this wait must stay notifiable. Armed here rather than at
        // construction so an idle Shell leaves the system at its default; the
        // request is process-wide and idempotent, so first waiter wins and every
        // other condition-variable wait (semaphore, pthread cond) benefits too.
        raeen_core::host_sleep::arm_high_resolution_timer();

        self.diagnostics.record(
            key.guest_thread,
            DiagnosticKind::WaitBegin,
            key.class,
            key.object,
            format!("timeout_us={}", timeout.as_micros()),
        );
        let deadline = std::time::Instant::now() + timeout;
        let (lock, cvar) = &self.event_flag_signal;
        let mut guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let outcome = loop {
            if ready() {
                break WaitOutcome::Ready;
            }
            if terminating() {
                break WaitOutcome::Terminating;
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break WaitOutcome::TimedOut;
            }
            let (next, _) = cvar
                .wait_timeout(guard, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard = next;
        };
        self.diagnostics.record(
            key.guest_thread,
            DiagnosticKind::WaitEnd,
            key.class,
            key.object,
            format!("{outcome:?}"),
        );
        outcome
    }

    fn wake(
        &self,
        key: raeen_core::subsystems::WaitKey,
        reason: raeen_core::subsystems::WakeReason,
    ) {
        use raeen_core::diagnostics::DiagnosticKind;
        self.diagnostics.record(
            key.guest_thread,
            DiagnosticKind::Wake,
            key.class,
            key.object,
            format!("{reason:?}"),
        );
        let (lock, cvar) = &self.event_flag_signal;
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        cvar.notify_all();
    }
}

impl raeen_core::subsystems::EventSubsystem for OrbisKernel {
    fn create_event(&self, attributes: u32, initial: u64) -> Option<u64> {
        let handle = self.create_event_flag(attributes, initial)?;
        self.diagnostics.record(
            0,
            raeen_core::diagnostics::DiagnosticKind::EventTransition,
            "event-flag",
            handle,
            format!("create attributes={attributes:#x} bits={initial:#x}"),
        );
        Some(handle)
    }

    fn delete_event(&self, handle: u64) -> bool {
        let removed = self.kernel_event_flags.remove(&handle).is_some();
        if removed {
            self.kernel_event_flag_live
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            self.diagnostics.record(
                0,
                raeen_core::diagnostics::DiagnosticKind::EventTransition,
                "event-flag",
                handle,
                "delete",
            );
        }
        removed
    }

    fn update_event(
        &self,
        handle: u64,
        update: raeen_core::subsystems::EventUpdate,
    ) -> Option<u64> {
        use raeen_core::diagnostics::DiagnosticKind;
        use raeen_core::subsystems::EventUpdate;
        let mut event = self.kernel_event_flags.get_mut(&handle)?;
        event.bits = match update {
            EventUpdate::Set(pattern) => event.bits | pattern,
            EventUpdate::Keep(pattern) => event.bits & pattern,
            EventUpdate::Replace(pattern) => pattern,
        };
        let bits = event.bits;
        drop(event);
        self.diagnostics.record(
            0,
            DiagnosticKind::EventTransition,
            "event-flag",
            handle,
            format!("{update:?} -> {bits:#x}"),
        );
        Some(bits)
    }

    fn event_bits(&self, handle: u64) -> Option<u64> {
        self.kernel_event_flags.get(&handle).map(|event| event.bits)
    }
}

impl raeen_core::subsystems::VfsSubsystem for OrbisKernel {
    fn open(&self, path: &str, flags: i32, mode: u32) -> std::io::Result<i32> {
        self.filesystem.open(path, flags, mode)
    }

    fn read(&self, fd: i32, count: usize) -> std::io::Result<Vec<u8>> {
        self.filesystem.read(fd, count)
    }

    fn read_into(&self, fd: i32, out: &mut [u8]) -> std::io::Result<usize> {
        self.filesystem.read_into(fd, out)
    }

    fn write(&self, fd: i32, bytes: &[u8]) -> std::io::Result<usize> {
        self.filesystem.write(fd, bytes)
    }

    fn sync(&self, fd: i32) -> std::io::Result<()> {
        self.filesystem.sync(fd)
    }

    fn close(&self, fd: i32) -> std::io::Result<()> {
        self.filesystem.close(fd)
    }
}

impl raeen_core::subsystems::NetworkSubsystem for OrbisKernel {
    fn mode(&self) -> raeen_core::subsystems::NetworkMode {
        raeen_core::subsystems::NetworkMode::Offline
    }

    fn create_socket(&self) -> Option<i32> {
        OrbisKernel::create_socket(self)
    }

    fn socket_exists(&self, fd: i32) -> bool {
        self.kernel_sockets.contains_key(&fd)
    }

    fn close_socket(&self, fd: i32) -> bool {
        OrbisKernel::close_socket(self, fd)
    }
}

#[cfg(test)]
mod subsystem_resource_tests {
    use super::OrbisKernel;
    use raeen_core::subsystems::EventSubsystem;
    use std::sync::Arc;

    #[test]
    fn event_contract_reports_exhaustion_and_delete_releases_a_slot() {
        let kernel = OrbisKernel::new();
        let mut handles = Vec::new();
        for _ in 0..OrbisKernel::MAX_EVENT_FLAGS {
            handles.push(kernel.create_event(0, 0).expect("event slot"));
        }
        assert_eq!(kernel.create_event(0, 0), None);
        assert!(kernel.delete_event(handles[0]));
        assert!(kernel.create_event(0, 0).is_some());
    }

    #[test]
    fn network_contract_reports_exhaustion_and_close_releases_a_slot() {
        let kernel = OrbisKernel::new();
        let mut fds = Vec::new();
        for _ in 0..OrbisKernel::MAX_OFFLINE_SOCKETS {
            fds.push(kernel.create_socket().expect("socket slot"));
        }
        assert_eq!(kernel.create_socket(), None);
        assert!(kernel.close_socket(fds[0]));
        assert!(kernel.create_socket().is_some());
    }

    #[test]
    fn unresolved_nid_inventory_deduplicates_per_process_and_calling_module() {
        let kernel = OrbisKernel::new();
        assert_eq!(
            kernel.record_unresolved_nid_call(0x1234, "sceExample", "libSceExample", "eboot.bin"),
            (true, 1)
        );
        assert_eq!(
            kernel.record_unresolved_nid_call(0x1234, "sceExample", "libSceExample", "eboot.bin"),
            (false, 2)
        );
        assert_eq!(
            kernel.record_unresolved_nid_call(0x1234, "sceExample", "libSceExample", "plugin.prx"),
            (true, 1),
            "a second calling module is a distinct compatibility gap"
        );
        assert_eq!(kernel.unresolved_nid_calls.len(), 2);
        // The formatted inventory is sorted and carries the call counts —
        // the exact lines the crash report and the log dump both emit.
        assert_eq!(
            kernel.unresolved_nid_inventory(),
            vec![
                "0x0000000000001234 sceExample library=libSceExample caller=eboot.bin calls=2"
                    .to_string(),
                "0x0000000000001234 sceExample library=libSceExample caller=plugin.prx calls=1"
                    .to_string(),
            ]
        );
    }

    /// A guest worker torn down by a host-detected fault never runs its unlock
    /// path, so every lock it held would stay held forever and its waiters would
    /// hang (the measured "scePthreadMutexLock stuck >3s — deadlock" cascade).
    /// `release_locks_owned_by` must free the dead thread's mutex ownership,
    /// rwlock write ownership, and rwlock read holds (giving back the shared
    /// reader count) while leaving OTHER threads' locks untouched.
    #[test]
    fn release_locks_owned_by_frees_only_the_dead_threads_mutexes_rwlocks_and_read_holds() {
        use super::{PthreadMutex, PthreadRwlock};
        let kernel = OrbisKernel::new();
        let dead = 7u64;
        let live = 9u64;

        // Two mutexes held by the dead thread, one held by a live thread.
        kernel
            .pthread_mutexes
            .insert(0x1000, PthreadMutex::shared_for_test(3, dead, 1));
        kernel
            .pthread_mutexes
            .insert(0x1008, PthreadMutex::shared_for_test(2, dead, 3));
        kernel
            .pthread_mutexes
            .insert(0x1010, PthreadMutex::shared_for_test(3, live, 1));
        // The pointer slot and opaque handle are two keys for one logical
        // mutex. Cleanup must release and count that shared object once.
        let dead_mutex_alias = {
            let entry = kernel.pthread_mutexes.get(&0x1000).unwrap();
            Arc::clone(entry.value())
        };
        kernel.pthread_mutexes.insert(0x1100, dead_mutex_alias);

        // One rwlock write-owned by the dead thread; one read-shared: the dead
        // thread holds 2 read recursions, the live thread holds 1 — reader count
        // is the shared sum (3).
        kernel
            .pthread_rwlocks
            .insert(0x2000, PthreadRwlock::shared_for_test(0x2000, 0, dead, 2));
        kernel
            .pthread_rwlocks
            .insert(0x2008, PthreadRwlock::shared_for_test(0x2008, 3, 0, 0));
        let dead_rwlock_alias = {
            let entry = kernel.pthread_rwlocks.get(&0x2000).unwrap();
            Arc::clone(entry.value())
        };
        kernel.pthread_rwlocks.insert(0x2100, dead_rwlock_alias);
        kernel.pthread_rwlock_read_holds.insert((dead, 0x2008), 2);
        kernel.pthread_rwlock_read_holds.insert((live, 0x2008), 1);

        let summary = kernel.release_locks_owned_by(dead);
        assert_eq!(summary.mutexes, 2);
        assert_eq!(summary.rwlock_writers, 1);
        assert_eq!(summary.rwlock_read_holds, 1);
        assert!(summary.any());

        // Dead thread's mutexes are cleared; the live thread's is untouched.
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&0x1000)
                .unwrap()
                .state
                .lock()
                .owner,
            0
        );
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&0x1000)
                .unwrap()
                .state
                .lock()
                .recursion,
            0
        );
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&0x1008)
                .unwrap()
                .state
                .lock()
                .owner,
            0
        );
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&0x1100)
                .unwrap()
                .state
                .lock()
                .owner,
            0
        );
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&0x1010)
                .unwrap()
                .state
                .lock()
                .owner,
            live
        );

        // The write lock is released; the read-shared lock loses the dead
        // thread's 2 holds (3 -> 1) and keeps the live thread's hold.
        assert_eq!(
            kernel
                .pthread_rwlocks
                .get(&0x2000)
                .unwrap()
                .state
                .lock()
                .writer,
            0
        );
        assert_eq!(
            kernel
                .pthread_rwlocks
                .get(&0x2100)
                .unwrap()
                .state
                .lock()
                .writer,
            0
        );
        assert_eq!(
            kernel
                .pthread_rwlocks
                .get(&0x2000)
                .unwrap()
                .state
                .lock()
                .writer_recursion,
            0
        );
        assert_eq!(
            kernel
                .pthread_rwlocks
                .get(&0x2008)
                .unwrap()
                .state
                .lock()
                .readers,
            1
        );
        assert!(
            kernel
                .pthread_rwlock_read_holds
                .get(&(dead, 0x2008))
                .is_none()
        );
        assert_eq!(
            *kernel
                .pthread_rwlock_read_holds
                .get(&(live, 0x2008))
                .unwrap(),
            1
        );

        // Idempotent: a second call on the same (now clean) thread frees nothing.
        assert!(!kernel.release_locks_owned_by(dead).any());
    }

    /// A dead thread's condition-variable wait entry is a stolen wakeup: the
    /// next `signal_one` pops it, wakes nobody, and reports success — so the
    /// live waiter queued behind it never runs. Thread-death cleanup must drop
    /// the entry (never signal it) and leave the live waiter first in line.
    #[test]
    fn release_locks_owned_by_drops_the_dead_threads_cond_waiters_without_signalling_them() {
        use super::PthreadCond;
        let kernel = OrbisKernel::new();
        let dead = 7u64;
        let live = 9u64;

        let cond = Arc::new(PthreadCond::default());
        let dead_waiter = cond.enqueue_waiter(dead);
        let live_waiter = cond.enqueue_waiter(live);
        kernel.pthread_conds.insert(0x3000, Arc::clone(&cond));
        // Object address and handle alias one queue; it must be visited once.
        kernel.pthread_conds.insert(0x3100, Arc::clone(&cond));

        // A second condition the dead thread never touched stays untouched.
        let other = Arc::new(PthreadCond::default());
        let other_waiter = other.enqueue_waiter(live);
        kernel.pthread_conds.insert(0x3200, other);

        let summary = kernel.release_locks_owned_by(dead);
        assert_eq!(
            summary.cond_waiters, 1,
            "exactly one abandoned entry, counted once across both aliases"
        );
        assert!(summary.any());
        assert_eq!(cond.waiter_count(), 1, "the live waiter stays queued");
        assert!(
            !dead_waiter.is_signaled(),
            "a dead waiter is discarded, never woken — waking it is the bug"
        );
        assert!(!live_waiter.is_signaled());
        assert!(!other_waiter.is_signaled());

        // The live waiter is now the one a signal reaches — the lost wakeup is
        // what this cleanup restores.
        assert!(cond.signal_one());
        assert!(live_waiter.is_signaled());

        // Idempotent on a queue with nothing of the dead thread's left.
        assert_eq!(kernel.release_locks_owned_by(dead).cond_waiters, 0);
    }
}
