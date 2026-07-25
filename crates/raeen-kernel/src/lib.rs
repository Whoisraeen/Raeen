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
    /// Stable, process-scoped HLE/wait/event/task/GPU event stream.
    pub diagnostics: Arc<DiagnosticRecorder>,
    /// Monotonic epoch owned by this guest process rather than a host-global
    /// static, so consecutive title launches do not inherit elapsed time.
    started_at: std::time::Instant,
    /// Loaded modules (.sprx / .elf).
    pub modules: DashMap<u32, ModuleInfo>,
    /// Handle-scoped NID -> absolute guest addresses for real LLE modules.
    lle_module_exports: DashMap<u32, HashMap<u64, u64>>,
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
    /// Host (OS) thread handle for each guest thread id, recorded as the thread
    /// starts. Purely diagnostic: it lets a sampler suspend a guest thread and
    /// read its RIP, which is the ONLY way to see where a title is stuck when it
    /// spins in its own code and makes no HLE calls (the call ring goes blank).
    /// On Windows these are duplicated `HANDLE`s owned for the process lifetime.
    pub host_thread_handles: DashMap<u64, u64>,
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
    /// Whether libSceUserService has delivered this process's initial login
    /// event. A new guest process gets a fresh kernel and therefore a fresh
    /// event; keeping this here avoids a host-global flag leaking across
    /// consecutive title launches.
    user_service_login_event_delivered: std::sync::atomic::AtomicBool,
    /// Guest pthread mutex state, keyed by both the guest `pthread_mutex_t`
    /// address and its allocated opaque handle (both map to the same logical
    /// mutex). Manipulated by the HLE `pthread_sync` module — per-process
    /// state, so it lives here rather than in a global.
    pub pthread_mutexes: DashMap<u64, Arc<parking_lot::Mutex<PthreadMutex>>>,
    /// Guest pthread mutex-attribute type, keyed by the attr object address.
    pub pthread_mutex_attrs: DashMap<u64, i32>,
    /// Guest pthread read-write lock state, keyed by both the guest
    /// `pthread_rwlock_t` address and its allocated handle.
    pub pthread_rwlocks: DashMap<u64, Arc<parking_lot::Mutex<PthreadRwlock>>>,
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
    /// count changes (Signal/Cancel). A blocked `WaitSema` re-checks its own
    /// count on wake. Same rationale as [`event_flag_signal`]: with real guest
    /// threads a producer thread *can* signal, so a waiter must block until it
    /// does rather than instantly time out.
    pub semaphore_signal: (std::sync::Mutex<()>, std::sync::Condvar),
    /// AMPR command-buffer write offsets, keyed by the command-buffer address
    /// (the current write cursor `sceAmprCommandBufferGetCurrentOffset` reads).
    pub ampr_write_offsets: DashMap<u64, u64>,
    /// AMPR per-command-buffer appended-record counts, keyed by the
    /// command-buffer address (`sceAmprCommandBufferGetNumCommands` reads;
    /// SharpEmu's `AmprCommandBufferState.CommandCount` — zeroed on
    /// construct/reset, incremented per appended record).
    pub ampr_command_counts: DashMap<u64, u64>,
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
    /// Guard-page size in bytes (default 4 KiB).
    pub guard_size: u64,
    /// Scheduling policy (default 1).
    pub sched_policy: i32,
    /// Scheduling priority.
    pub sched_priority: i32,
}

impl Default for PthreadAttr {
    fn default() -> Self {
        Self {
            detach_state: 0,
            stack_size: 0x10_0000,
            guard_size: 0x1000,
            sched_policy: 1,
            sched_priority: 700,
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

impl PthreadRwlock {
    /// Create one shared state object for a guest rwlock and all its aliases.
    pub fn shared(key: u64) -> Arc<parking_lot::Mutex<Self>> {
        Arc::new(parking_lot::Mutex::new(Self {
            key,
            readers: 0,
            writer: 0,
            writer_recursion: 0,
        }))
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
}

impl LockReleaseSummary {
    /// Whether any lock at all was released (worth logging).
    pub fn any(&self) -> bool {
        self.mutexes != 0 || self.rwlock_writers != 0 || self.rwlock_read_holds != 0
    }
}

/// State of a guest pthread mutex. The kernel map wraps this in one shared
/// `Arc<Mutex<_>>` per logical guest mutex because Orbis exposes both a pointer
/// slot and an opaque handle for the same object. Ported from SharpEmu's
/// `PthreadMutexState` (GPL-2.0). See the `raeen-hle` `pthread_sync` module for
/// the state machine.
#[derive(Clone, Copy, Debug)]
pub struct PthreadMutex {
    /// Mutex type: 1 = error-check, 2 = recursive, 3 = normal, 4 = adaptive.
    pub ty: i32,
    /// Owning thread handle (0 = unlocked).
    pub owner: u64,
    /// Lock recursion count (0 = unlocked).
    pub recursion: i32,
}

impl PthreadMutex {
    /// Create the single shared state object that every guest-visible alias of
    /// this mutex must reference.
    pub fn shared(ty: i32) -> Arc<parking_lot::Mutex<Self>> {
        Arc::new(parking_lot::Mutex::new(Self {
            ty,
            owner: 0,
            recursion: 0,
        }))
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

/// Host-backed generation wait queue for one guest pthread condition.
#[derive(Debug, Default)]
pub struct PthreadCond {
    /// Incremented by signal/broadcast while holding `generation`'s mutex.
    pub generation: parking_lot::Mutex<u64>,
    /// Waiters sleep here until the generation changes.
    pub changed: parking_lot::Condvar,
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
            let state = entry.value();
            if !visited.insert(Arc::as_ptr(state) as usize) {
                continue;
            }
            let mut state = state.lock();
            if state.owner == thread {
                state.owner = 0;
                state.recursion = 0;
                released += 1;
            }
        }
        released
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
    pub fn release_locks_owned_by(&self, thread: u64) -> LockReleaseSummary {
        let mut summary = LockReleaseSummary::default();

        let mut visited = HashSet::new();
        for entry in &self.pthread_mutexes {
            let state = entry.value();
            if !visited.insert(Arc::as_ptr(state) as usize) {
                continue;
            }
            let mut state = state.lock();
            if state.owner == thread {
                state.owner = 0;
                state.recursion = 0;
                summary.mutexes += 1;
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
                let mut rw = rw.lock();
                rw.readers = rw.readers.saturating_sub(depth as i32).max(0);
            }
            summary.rwlock_read_holds += 1;
        }

        let mut visited = HashSet::new();
        for entry in &self.pthread_rwlocks {
            let state = entry.value();
            if !visited.insert(Arc::as_ptr(state) as usize) {
                continue;
            }
            let mut state = state.lock();
            if state.writer == thread {
                state.writer = 0;
                state.writer_recursion = 0;
                summary.rwlock_writers += 1;
            }
        }

        summary
    }

    /// Create a new kernel instance with default configuration.
    pub fn new() -> Self {
        tracing::info!("Initializing Orbis kernel HLE");
        Self {
            console: Console::new(),
            memory: Arc::new(memory::VirtualMemoryManager::new()),
            threads: Arc::new(threading::ThreadManager::new()),
            filesystem: Arc::new(filesystem::VirtualFileSystem::new()),
            diagnostics: Arc::new(DiagnosticRecorder::from_env()),
            started_at: std::time::Instant::now(),
            modules: DashMap::new(),
            lle_module_exports: DashMap::new(),
            next_module_id: RwLock::new(1),
            syscall_stats: DashMap::new(),
            thread_names: DashMap::new(),
            thread_priorities: DashMap::new(),
            host_thread_handles: DashMap::new(),
            recent_hle_calls: DashMap::new(),
            in_flight_hle: DashMap::new(),
            proc_param_addr: std::sync::atomic::AtomicU64::new(0),
            process_argc: std::sync::atomic::AtomicU64::new(0),
            process_argv: std::sync::atomic::AtomicU64::new(0),
            unwind_modules: RwLock::new(Vec::new()),
            pad_state: parking_lot::Mutex::new(None),
            user_service_login_event_delivered: std::sync::atomic::AtomicBool::new(false),
            pthread_mutexes: DashMap::new(),
            pthread_mutex_attrs: DashMap::new(),
            pthread_rwlocks: DashMap::new(),
            pthread_rwlock_read_holds: DashMap::new(),
            hle_call_time: DashMap::new(),
            hle_call_counts: DashMap::new(),
            pthread_conds: DashMap::new(),
            pthread_condattr_clocks: DashMap::new(),
            pthread_attrs: DashMap::new(),
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
            ampr_write_offsets: DashMap::new(),
            ampr_command_counts: DashMap::new(),
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

    fn sleep(&self, duration: std::time::Duration) {
        std::thread::sleep(duration);
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
        kernel.pthread_mutexes.insert(
            0x1000,
            Arc::new(parking_lot::Mutex::new(PthreadMutex {
                ty: 3,
                owner: dead,
                recursion: 1,
            })),
        );
        kernel.pthread_mutexes.insert(
            0x1008,
            Arc::new(parking_lot::Mutex::new(PthreadMutex {
                ty: 2,
                owner: dead,
                recursion: 3,
            })),
        );
        kernel.pthread_mutexes.insert(
            0x1010,
            Arc::new(parking_lot::Mutex::new(PthreadMutex {
                ty: 3,
                owner: live,
                recursion: 1,
            })),
        );
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
        kernel.pthread_rwlocks.insert(
            0x2000,
            Arc::new(parking_lot::Mutex::new(PthreadRwlock {
                key: 0x2000,
                readers: 0,
                writer: dead,
                writer_recursion: 2,
            })),
        );
        kernel.pthread_rwlocks.insert(
            0x2008,
            Arc::new(parking_lot::Mutex::new(PthreadRwlock {
                key: 0x2008,
                readers: 3,
                writer: 0,
                writer_recursion: 0,
            })),
        );
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
        assert_eq!(kernel.pthread_mutexes.get(&0x1000).unwrap().lock().owner, 0);
        assert_eq!(
            kernel
                .pthread_mutexes
                .get(&0x1000)
                .unwrap()
                .lock()
                .recursion,
            0
        );
        assert_eq!(kernel.pthread_mutexes.get(&0x1008).unwrap().lock().owner, 0);
        assert_eq!(kernel.pthread_mutexes.get(&0x1100).unwrap().lock().owner, 0);
        assert_eq!(
            kernel.pthread_mutexes.get(&0x1010).unwrap().lock().owner,
            live
        );

        // The write lock is released; the read-shared lock loses the dead
        // thread's 2 holds (3 -> 1) and keeps the live thread's hold.
        assert_eq!(
            kernel.pthread_rwlocks.get(&0x2000).unwrap().lock().writer,
            0
        );
        assert_eq!(
            kernel.pthread_rwlocks.get(&0x2100).unwrap().lock().writer,
            0
        );
        assert_eq!(
            kernel
                .pthread_rwlocks
                .get(&0x2000)
                .unwrap()
                .lock()
                .writer_recursion,
            0
        );
        assert_eq!(
            kernel.pthread_rwlocks.get(&0x2008).unwrap().lock().readers,
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
}
