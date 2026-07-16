//! # XPS5X Kernel
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
// entries without a direct xps5x-core type path (M1-D).
pub use xps5x_core::types::ModuleInfo;

use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
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
    /// Loaded modules (.sprx / .elf).
    pub modules: DashMap<u32, ModuleInfo>,
    /// Handle-scoped NID -> absolute guest addresses for real LLE modules.
    lle_module_exports: DashMap<u32, HashMap<u64, u64>>,
    /// Next module ID to assign.
    next_module_id: RwLock<u32>,
    /// Syscall statistics (for debugging).
    pub syscall_stats: DashMap<u64, u64>,
    /// Guest address of the main module's `PT_SCE_PROCPARAM` block, set by
    /// the runtime at load time (0 = none). `sceKernelGetProcParam` returns
    /// this so a title reads its real process-parameter block (SDK version,
    /// etc.) instead of a stub pointer.
    proc_param_addr: std::sync::atomic::AtomicU64,
    /// Real ELF ranges and exception tables for the currently executing
    /// process. The runtime replaces this atomically before entering `_start`.
    unwind_modules: RwLock<Vec<UnwindModuleInfo>>,
    /// The current controller snapshot as a 12-byte Orbis `ScePadData` input
    /// prefix (buttons + sticks + triggers), or `None` when the host has not
    /// pushed live input — in which case `scePadReadState` reports a neutral
    /// pad. The Shell updates this each frame from its `InputManager`; the
    /// HLE `scePadReadState` reads it. See [`set_pad_state`](Self::set_pad_state).
    pad_state: parking_lot::Mutex<Option<[u8; 12]>>,
    /// Guest pthread mutex state, keyed by both the guest `pthread_mutex_t`
    /// address and its allocated opaque handle (both map to the same logical
    /// mutex). Manipulated by the HLE `pthread_sync` module — per-process
    /// state, so it lives here rather than in a global.
    pub pthread_mutexes: DashMap<u64, PthreadMutex>,
    /// Guest pthread mutex-attribute type, keyed by the attr object address.
    pub pthread_mutex_attrs: DashMap<u64, i32>,
    /// Guest pthread read-write lock state, keyed by both the guest
    /// `pthread_rwlock_t` address and its allocated handle.
    pub pthread_rwlocks: DashMap<u64, PthreadRwlock>,
    /// Guest pthread condition-variable wait queues, keyed by object address.
    pub pthread_conds: DashMap<u64, Arc<PthreadCond>>,
    /// Guest pthread thread-attribute objects, keyed by both the guest
    /// `pthread_attr_t` address and its allocated handle.
    pub pthread_attrs: DashMap<u64, PthreadAttr>,
    /// Kernel event flags, keyed by handle.
    pub kernel_event_flags: DashMap<u64, EventFlag>,
    /// Next event-flag handle to hand out.
    kernel_event_flag_next: std::sync::atomic::AtomicU64,
    /// Kernel counting semaphores, keyed by handle.
    pub kernel_semaphores: DashMap<u32, Semaphore>,
    /// POSIX (`sem_*`) semaphores, keyed by the guest `sem_t` address. These
    /// are address-based objects distinct from the handle-based
    /// `sceKernelCreateSema` family — see the `xps5x-hle` `posix_sem` module.
    pub posix_semaphores: DashMap<u64, Arc<PosixSem>>,
    /// Next semaphore handle to hand out.
    kernel_semaphore_next: std::sync::atomic::AtomicU32,
    /// Kernel event queues (existence), keyed by handle → attributes.
    pub kernel_equeues: DashMap<u64, u32>,
    /// Registered user events on event queues, keyed by (equeue, ident).
    pub kernel_equeue_events: DashMap<(u64, u64), EqueueUserEvent>,
    /// Next event-queue handle to hand out.
    kernel_equeue_next: std::sync::atomic::AtomicU64,
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
    /// AMPR command-buffer write offsets, keyed by the command-buffer address
    /// (the current write cursor `sceAmprCommandBufferGetCurrentOffset` reads).
    pub ampr_write_offsets: DashMap<u64, u64>,
    /// The libSceFiber context-size-check profiling toggle (0 = off, 1 = on).
    pub fiber_context_size_check: std::sync::atomic::AtomicU32,
    /// Guest network sockets (offline — no host connectivity), keyed by fd.
    pub kernel_sockets: DashMap<i32, GuestSocket>,
    /// Next socket fd to hand out (a high range, distinct from VFS fds).
    kernel_socket_next: std::sync::atomic::AtomicI32,
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

/// A guest network socket. XPS5X models **no host connectivity**, so a socket
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
    /// close-on-exec). XPS5X does not currently replace a guest process image,
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

/// State of a guest pthread read-write lock. Single-active-execution means one
/// guest thread, so the lock never truly contends — read/write acquisition
/// reduces to reader-count + writer-recursion tracking (leniently, matching
/// SharpEmu's `PthreadRwlockState`, GPL-2.0). See `xps5x-hle` `pthread_sync`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PthreadRwlock {
    /// Outstanding read-lock holds by the (single) thread.
    pub readers: i32,
    /// Write-owning thread handle (0 = no writer).
    pub writer: u64,
    /// Write-lock recursion depth.
    pub writer_recursion: i32,
}

/// State of a guest pthread mutex. Under XPS5X's single-active-execution model
/// there is one guest thread, so a mutex reduces to its type plus owner /
/// recursion tracking — which is exactly correct for that model. Ported from
/// SharpEmu's `PthreadMutexState` (GPL-2.0). See the `xps5x-hle` `pthread_sync`
/// module for the state machine.
#[derive(Clone, Copy, Debug)]
pub struct PthreadMutex {
    /// Mutex type: 1 = error-check, 2 = recursive, 3 = normal, 4 = adaptive.
    pub ty: i32,
    /// Owning thread handle (0 = unlocked).
    pub owner: u64,
    /// Lock recursion count (0 = unlocked).
    pub recursion: i32,
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

impl OrbisKernel {
    /// Create a new kernel instance with default configuration.
    pub fn new() -> Self {
        tracing::info!("Initializing Orbis kernel HLE");
        Self {
            console: Console::new(),
            memory: Arc::new(memory::VirtualMemoryManager::new()),
            threads: Arc::new(threading::ThreadManager::new()),
            filesystem: Arc::new(filesystem::VirtualFileSystem::new()),
            modules: DashMap::new(),
            lle_module_exports: DashMap::new(),
            next_module_id: RwLock::new(1),
            syscall_stats: DashMap::new(),
            proc_param_addr: std::sync::atomic::AtomicU64::new(0),
            unwind_modules: RwLock::new(Vec::new()),
            pad_state: parking_lot::Mutex::new(None),
            pthread_mutexes: DashMap::new(),
            pthread_mutex_attrs: DashMap::new(),
            pthread_rwlocks: DashMap::new(),
            pthread_conds: DashMap::new(),
            pthread_attrs: DashMap::new(),
            kernel_event_flags: DashMap::new(),
            kernel_event_flag_next: std::sync::atomic::AtomicU64::new(1),
            kernel_semaphores: DashMap::new(),
            kernel_semaphore_next: std::sync::atomic::AtomicU32::new(1),
            posix_semaphores: DashMap::new(),
            kernel_equeues: DashMap::new(),
            kernel_equeue_events: DashMap::new(),
            kernel_equeue_next: std::sync::atomic::AtomicU64::new(1),
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
            agc_last_dcb_address: std::sync::atomic::AtomicU64::new(0),
            agc_last_dcb_dwords: std::sync::atomic::AtomicU32::new(0),
            agc_register_defaults_addr: std::sync::atomic::AtomicU64::new(0),
            video_out_buffers: DashMap::new(),
            video_out_flip_count: std::sync::atomic::AtomicU64::new(0),
            video_out_vblank_count: std::sync::atomic::AtomicU64::new(0),
            video_out_last_flip_arg: std::sync::atomic::AtomicI64::new(0),
            video_out_current_buffer: std::sync::atomic::AtomicI32::new(0),
            libc_mspaces: DashMap::new(),
            libc_mspace_allocations: DashMap::new(),
            ampr_write_offsets: DashMap::new(),
            fiber_context_size_check: std::sync::atomic::AtomicU32::new(0),
            kernel_sockets: DashMap::new(),
            kernel_socket_next: std::sync::atomic::AtomicI32::new(0x4000_0000),
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
    /// handle. See the `xps5x-hle` `kernel_eventflag` module.
    pub fn create_event_flag(&self, attributes: u32, initial_bits: u64) -> u64 {
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
        handle
    }

    /// Create a counting semaphore with `initial`/`max` count, returning its
    /// handle. See the `xps5x-hle` `kernel_semaphore` module.
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
    /// `xps5x-hle` `kernel_equeue` module.
    pub fn create_equeue(&self, attributes: u32) -> u64 {
        let handle = self
            .kernel_equeue_next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.kernel_equeues.insert(handle, attributes);
        handle
    }

    /// Allocate a fresh (offline) socket fd. See the `xps5x-hle`
    /// `kernel_socket` module.
    pub fn create_socket(&self) -> i32 {
        let fd = self
            .kernel_socket_next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.kernel_sockets.insert(fd, GuestSocket::default());
        fd
    }

    /// Allocate a fresh pthread TLS key registered with `destructor` (0 = none),
    /// returning its id. See the `xps5x-hle` `pthread_tls` module.
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
    ) -> Result<u64, xps5x_core::error::KernelError> {
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
