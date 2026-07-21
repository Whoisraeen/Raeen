//! # XPS5X HLE
//!
//! High-Level Emulation of PS5 system libraries.
//!
//! PS5 games link against Sony's proprietary `.sprx` libraries.
//! Rather than loading encrypted firmware modules, XPS5X re-implements
//! these libraries' exported functions, routing calls to the appropriate
//! emulator subsystem.
//!
//! ## Implemented Libraries
//!
//! | Library | Status | Routes to |
//! |:---|:---|:---|
//! | libkernel.sprx | Partial | xps5x-kernel |
//! | libc.sprx | Partial | Rust std |
//! | libSceGnmDriver.sprx | Stub | xps5x-gpu |
//! | libSceVideoOut.sprx | Stub | xps5x-gpu (Vulkan swapchain) |
//! | libSceAudioOut.sprx | Stub | xps5x-audio |
//! | libScePad.sprx | Stub | xps5x-input |
//! | libSceNet.sprx | Stub | Host networking |
//! | libSceSaveData.sprx | Stub | xps5x-kernel (VFS) |
//! | libSceSysmodule.sprx | Partial | Module registry |

pub(crate) mod fmt;
pub mod kernel_equeue;
pub mod kernel_eventflag;
pub mod kernel_semaphore;
pub mod kernel_socket;
pub mod libc;
pub mod libkernel;
pub mod libsce_acm;
pub mod libsce_agc;
pub(crate) mod libsce_agc_reg_defaults;
pub mod libsce_ampr;
pub mod libsce_app_content;
pub mod libsce_audio_out;
pub mod libsce_audio_out2;
pub mod libsce_common_dialog;
pub mod libsce_content_export;
pub mod libsce_coredump;
pub mod libsce_disc_map;
pub mod libsce_fiber;
pub mod libsce_font;
pub mod libsce_gnm_driver;
pub mod libsce_http;
pub mod libsce_json;
pub mod libsce_libc_internal;
pub mod libsce_media;
pub mod libsce_net;
pub mod libsce_np;
pub mod libsce_np_entitlement;
pub mod libsce_np_session_signaling;
pub mod libsce_np_trophy2;
pub mod libsce_np_universal_data;
pub mod libsce_np_web_api2;
pub mod libsce_pad;
pub mod libsce_peripheral;
pub mod libsce_playgo;
pub mod libsce_posix;
pub mod libsce_random;
pub mod libsce_rtc;
pub mod libsce_save_data;
pub mod libsce_save_data_dialog;
pub mod libsce_share;
pub mod libsce_ssl;
pub mod libsce_sysmodule;
pub mod libsce_system_service;
pub mod libsce_text_to_speech2;
pub mod libsce_user_service;
pub mod libsce_video_out;
pub mod posix_sem;
pub mod pthread_attr;
pub mod pthread_cond;
pub mod pthread_sync;
pub mod pthread_thread;
pub mod pthread_tls;

use dashmap::DashMap;
use tracing::{debug, info, warn};
use xps5x_core::diagnostics::DiagnosticKind;
use xps5x_core::subsystems::{GpuSubmissionSubsystem, KernelSubsystems};

/// Access to the guest (emulated PS5) address space from an HLE function.
///
/// Every implementation must be bounds-checked: an out-of-bounds
/// `guest_addr`/length combination returns `false` (touching nothing)
/// rather than panicking or reading/writing outside the guest's actual
/// backing storage. An HLE function handed a wild pointer by buggy or
/// malicious guest code must never be able to turn that into a host OOB
/// access or a panic.
pub trait GuestMemory {
    /// Read `out.len()` bytes starting at `guest_addr` into `out`. Returns
    /// `false` (leaving `out`'s contents unspecified) if the read would
    /// fall outside the guest's mapped memory.
    fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool;
    /// Write `data` starting at `guest_addr`. Returns `false` (writing
    /// nothing) if the write would fall outside the guest's mapped memory.
    fn write(&self, guest_addr: u64, data: &[u8]) -> bool;

    /// Write into the guest CODE image (instrumentation patches: export-trap
    /// `int3`, `native_trap` prologues, one-shot restores). Distinct from
    /// [`write`] because a W^X backend makes the code image read-only, so a
    /// plain data write there would fault; a code patch must transiently lift
    /// the write bar. The default is a plain `write` — correct for the
    /// permissive RWX default and for test memories; a W^X arena overrides it
    /// to toggle page protection around the store.
    fn patch_code(&self, guest_addr: u64, data: &[u8]) -> bool {
        self.write(guest_addr, data)
    }

    /// Apply a guest `mprotect`: change the host protection of the committed
    /// pages in `[addr, addr+len)` to the Orbis CPU-protection bitset `prot`
    /// (`CPU_READ`=1, `CPU_WRITE`=2, `CPU_EXEC`=4, `NO_ACCESS`=0). The default
    /// is a no-op returning `true` — the historical "protections not remapped"
    /// behaviour — so a title that only queries or that runs without
    /// enforcement is unaffected. A real arena overrides it (behind an opt-in
    /// gate) to actually re-protect the pages, turning a write to a page the
    /// guest marked read-only into a trap instead of silent success.
    fn protect(&self, _addr: u64, _len: u64, _prot: u32) -> bool {
        true
    }

    /// Validate a whole guest range for the requested access without exposing
    /// a host pointer. Backends should override this with their authoritative
    /// map; the default probes one byte per 4 KiB page plus the last byte.
    fn validate_range(&self, range: GuestRange, _access: GuestAccess) -> bool {
        if range.is_empty() {
            return true;
        }
        let Some(last) = range.end().and_then(|end| end.checked_sub(1)) else {
            return false;
        };
        let mut probe = [0u8; 1];
        let mut address = range.start().raw();
        loop {
            if !self.read(address, &mut probe) {
                return false;
            }
            if address == last {
                return true;
            }
            address = address.saturating_add(0x1000).min(last);
        }
    }

    /// Whether the entire range may be entered as native guest code.
    fn is_executable_range(&self, _range: GuestRange) -> bool {
        false
    }

    /// Whether the GPU command path may consume this guest range.
    fn is_gpu_visible_range(&self, _range: GuestRange) -> bool {
        false
    }

    /// Atomic 32-bit load used for guest synchronization words. The default
    /// is suitable for single-threaded test memories; native runtimes must
    /// override it with a real host atomic operation.
    fn atomic_load_u32(&self, guest_addr: u64) -> Option<u32> {
        let mut bytes = [0u8; 4];
        self.read(guest_addr, &mut bytes)
            .then(|| u32::from_le_bytes(bytes))
    }

    /// Compare/exchange a 32-bit guest synchronization word, returning the
    /// observed value. See [`GuestMemory::atomic_load_u32`].
    fn atomic_compare_exchange_u32(&self, guest_addr: u64, current: u32, new: u32) -> Option<u32> {
        let observed = self.atomic_load_u32(guest_addr)?;
        if observed == current && !self.write(guest_addr, &new.to_le_bytes()) {
            return None;
        }
        Some(observed)
    }

    /// Atomic 32-bit store used to complete or roll back guest callbacks.
    fn atomic_store_u32(&self, guest_addr: u64, value: u32) -> bool {
        self.write(guest_addr, &value.to_le_bytes())
    }
}

/// An untrusted address received from guest registers or memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GuestAddress(u64);

impl GuestAddress {
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// A checked-for-overflow address/length pair. This is still untrusted until
/// converted into one of the capability types below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GuestRange {
    start: GuestAddress,
    len: u64,
}

impl GuestRange {
    #[must_use]
    pub fn new(start: GuestAddress, len: u64) -> Option<Self> {
        start.raw().checked_add(len)?;
        Some(Self { start, len })
    }

    #[must_use]
    pub const fn start(self) -> GuestAddress {
        self.start
    }

    #[must_use]
    pub const fn len(self) -> u64 {
        self.len
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn end(self) -> Option<u64> {
        self.start.raw().checked_add(self.len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuestAccess {
    Read,
    Write,
    ReadWrite,
}

/// Proof that `memory` accepted a complete range for a specific access mode.
/// Construction is private to the validating API, so downstream code cannot
/// accidentally relabel a raw integer as mapped guest memory.
pub struct ValidatedGuestRange<'a> {
    memory: &'a dyn GuestMemory,
    range: GuestRange,
    access: GuestAccess,
}

impl<'a> ValidatedGuestRange<'a> {
    #[must_use]
    pub fn validate(
        memory: &'a dyn GuestMemory,
        range: GuestRange,
        access: GuestAccess,
    ) -> Option<Self> {
        memory.validate_range(range, access).then_some(Self {
            memory,
            range,
            access,
        })
    }

    #[must_use]
    pub const fn range(&self) -> GuestRange {
        self.range
    }

    #[must_use]
    pub const fn access(&self) -> GuestAccess {
        self.access
    }

    pub fn read(&self, out: &mut [u8]) -> bool {
        out.len() as u64 == self.range.len()
            && self.access != GuestAccess::Write
            && self.memory.read(self.range.start().raw(), out)
    }

    pub fn write(&self, data: &[u8]) -> bool {
        data.len() as u64 == self.range.len()
            && self.access != GuestAccess::Read
            && self.memory.write(self.range.start().raw(), data)
    }
}

/// Validated executable mapping. Used for guest entries/callbacks, not ordinary
/// data ranges.
pub struct ExecutableGuestMapping<'a>(ValidatedGuestRange<'a>);

impl<'a> ExecutableGuestMapping<'a> {
    #[must_use]
    pub fn validate(memory: &'a dyn GuestMemory, range: GuestRange) -> Option<Self> {
        memory
            .is_executable_range(range)
            .then_some(Self(ValidatedGuestRange {
                memory,
                range,
                access: GuestAccess::Read,
            }))
    }

    #[must_use]
    pub const fn range(&self) -> GuestRange {
        self.0.range()
    }
}

/// Validated guest memory that may be consumed by the GPU submission path.
pub struct GpuVisibleGuestRange<'a>(ValidatedGuestRange<'a>);

impl<'a> GpuVisibleGuestRange<'a> {
    #[must_use]
    pub fn validate(memory: &'a dyn GuestMemory, range: GuestRange) -> Option<Self> {
        memory
            .is_gpu_visible_range(range)
            .then_some(Self(ValidatedGuestRange {
                memory,
                range,
                access: GuestAccess::ReadWrite,
            }))
    }

    #[must_use]
    pub const fn range(&self) -> GuestRange {
        self.0.range()
    }
}

/// Host-side staging bound for one bulk HLE memory operation. Large guest
/// mappings remain valid, but an ABI adapter must not allocate or loop across
/// attacker-sized lengths in one call.
pub(crate) const MAX_HLE_BULK_BYTES: u64 = 256 << 20;

pub(crate) fn zero_guest_range(memory: &dyn GuestMemory, addr: u64, len: u64) -> bool {
    if len == 0 {
        return true;
    }
    if len > MAX_HLE_BULK_BYTES {
        return false;
    }
    let Some(range) = GuestRange::new(GuestAddress::new(addr), len) else {
        return false;
    };
    if ValidatedGuestRange::validate(memory, range, GuestAccess::Write).is_none() {
        return false;
    }

    static ZEROES: [u8; 64 * 1024] = [0; 64 * 1024];
    let mut written = 0u64;
    while written < len {
        let chunk = (len - written).min(ZEROES.len() as u64) as usize;
        if !memory.write(addr + written, &ZEROES[..chunk]) {
            return false;
        }
        written += chunk as u64;
    }
    true
}

/// Allocates and releases guest memory on behalf of an HLE function —
/// `malloc`/`free`/`realloc`/`mmap`/`munmap`'s underlying mechanism.
///
/// Every method is total: an exhausted arena, an overflowing size/alignment
/// request, or an unrecognized address returns a sentinel (`None`, or simply
/// doing nothing) rather than panicking. Nothing calls this trait's methods
/// yet (that lands in RT2 Task 3/4/5, once a real implementation —
/// `xps5x-runtime`'s `GuestArena` — exists); it is threaded through
/// [`HleContext`] now so every call site is ready ahead of that.
pub trait GuestAllocator {
    /// Allocate at least `size` bytes, aligned to `align`, returning the
    /// guest address of the new block, or `None` if the request cannot be
    /// satisfied (exhausted arena, overflowing size/align, ...).
    fn alloc(&self, size: u64, align: u64) -> Option<u64>;
    /// Release a block previously returned by `alloc`/`realloc`/`mmap`. An
    /// unrecognized `addr` is simply ignored.
    fn free(&self, addr: u64);
    /// Resize the block at `addr` to `new_size`, returning the (possibly
    /// new) guest address, or `None` if the request cannot be satisfied —
    /// `addr` is left untouched in that case.
    fn realloc(&self, addr: u64, new_size: u64) -> Option<u64>;
    /// Reserve a `length`-byte region aligned to `align`, returning its
    /// guest address, or `None` if the request cannot be satisfied.
    fn mmap(&self, length: u64, align: u64) -> Option<u64>;
    /// Reserve address space without making it readable or writable. The
    /// default keeps small test allocators source-compatible; native runtimes
    /// override this so large sparse ranges do not consume the committed mmap
    /// pool.
    fn reserve(&self, length: u64, align: u64) -> Option<u64> {
        self.mmap(length, align)
    }
    /// Commit a mapping at a caller-selected address inside a prior virtual
    /// reservation. Test allocators reject fixed mappings by default.
    fn map_at(&self, addr: u64, length: u64, align: u64) -> Option<u64> {
        if addr == 0 {
            self.mmap(length, align)
        } else {
            None
        }
    }
    /// Back `addr` with memory if it falls in a range the guest reserved but
    /// that carries no memory yet, returning whether the faulting access should
    /// be retried. Real titles reserve far more address space than they touch
    /// (Until Dawn opens with a 512 GiB reservation) and then use it directly,
    /// so reservations must be committed lazily, page by page, on first touch.
    /// The default declines: an allocator without a sparse reservation model
    /// has nothing to commit, and a `false` here simply leaves the access to be
    /// reported as a genuine fault exactly as before.
    fn commit_on_demand(&self, _addr: u64) -> bool {
        false
    }
    /// Release a `length`-byte region previously returned by `mmap` starting
    /// at `addr`. An unrecognized `addr` is simply ignored.
    fn munmap(&self, addr: u64, length: u64);
}

/// A memory update the runtime performs after a requested guest callback
/// returns. The failure value is restored if that callback faults before
/// completing, so synchronization state is never left permanently wedged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestCallCompletion {
    pub address: u64,
    pub success_u32: u32,
    pub failure_u32: u32,
}

/// A guest function call requested while servicing an HLE import.
///
/// The runtime resumes guest execution at `entry` using the active guest
/// stack/TLS context, then transparently returns to the instruction after the
/// original import call. This is generic infrastructure for pthread once
/// initializers and other PS5 APIs that accept guest callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestCallRequest {
    pub entry: u64,
    pub args: [u64; 6],
    pub completion: Option<GuestCallCompletion>,
}

/// Sink through which an HLE implementation can request one deferred guest
/// call. A request returns `false` if this dispatch already has one pending.
pub trait GuestCallScheduler {
    fn request(&self, request: GuestCallRequest) -> bool;
}

/// Runtime-owned guest pthread lifecycle. HLE knows the Orbis ABI, while the
/// native runtime owns stacks, TCBs, host threads, and guarded guest entry.
pub trait GuestThreadScheduler {
    fn create(&self, thread_out: u64, attr: u64, entry: u64, arg: u64) -> u64;
    fn join(&self, thread: u64, retval_out: u64) -> u64;
    fn detach(&self, thread: u64) -> u64;
    fn request_exit(&self, retval: u64) -> bool;
    fn current_thread(&self) -> u64;
    /// Mark the whole guest process as terminating with `code`.
    fn request_process_exit(&self, code: u64);
    /// Whether another guest thread has requested process termination.
    fn process_is_terminating(&self) -> bool;
    /// The base of *this* thread's static TLS block: the storage the main
    /// module's `PT_TLS` template was copied into, which the linker's `TPOFF64`
    /// offsets resolve against (variant II — the block sits immediately below
    /// the TCB).
    ///
    /// `__tls_get_addr` needs it because the ELF TLS ABI requires that a
    /// thread-local reached through the general-dynamic model resolve to the
    /// *same address* as the same variable reached through initial-exec.
    /// Handing back separate storage gives one variable two homes, and only one
    /// of them holds its initialized value.
    ///
    /// `None` when the thread has no static block — a module with no `PT_TLS`,
    /// a CPU without FSGSBASE, or a test double. Callers must then fall back to
    /// dynamic storage rather than assume an address.
    fn current_static_tls_block(&self) -> Option<u64> {
        None
    }
}

/// Everything an HLE function may touch: the emulated kernel (memory,
/// threads, filesystem, ...), the guest's address space, and the guest
/// allocator.
///
/// This is the dispatch-context milestone: before it existed, an HLE
/// function was a bare `fn(&[u64]) -> u64` with no way to read/write guest
/// pointers or reach a live [`xps5x_kernel::OrbisKernel`] — every stub was
/// necessarily a no-op that just logged and returned a plausible value. Now
/// every HLE call gets all three, so functions like `memcpy`/`strlen`/
/// `sceKernelMapFlexibleMemory` can do the real operation.
pub struct HleContext<'a> {
    /// The live emulated kernel (memory manager, thread manager, VFS, ...).
    pub kernel: &'a xps5x_kernel::OrbisKernel,
    /// XPS5X-owned service contracts. New HLE code should translate the ABI
    /// and call these interfaces instead of reaching into kernel fields.
    pub services: &'a dyn KernelSubsystems,
    /// Process-owned GPU submission boundary. No Kyty type crosses this seam.
    pub gpu: &'a dyn GpuSubmissionSubsystem,
    /// The guest's address space, as seen from wherever this call
    /// originated (e.g. the runtime's mapped module image).
    pub mem: &'a dyn GuestMemory,
    /// The guest allocator backing `malloc`/`mmap` and friends. Not yet
    /// consumed by any HLE function body — see [`GuestAllocator`]'s doc
    /// comment.
    pub alloc: &'a dyn GuestAllocator,
    /// Deferred guest callbacks executed by the native runtime after the HLE
    /// handler returns, while the caller's guest register/stack context is
    /// still active.
    pub guest_calls: &'a dyn GuestCallScheduler,
    /// Process-scoped guest pthread scheduler supplied by the runtime.
    pub guest_threads: &'a dyn GuestThreadScheduler,
    /// The guest return address of the current HLE call — `[rsp]` at the
    /// trap, i.e. the instruction the caller resumes at. Diagnostic only
    /// (0 when unavailable / in tests): lets a handler report *where in the
    /// guest* a call like `sceKernelDebugRaiseException` originated, which
    /// `--dump-vaddr` then turns into the failing assert's code bytes.
    pub caller_return_addr: u64,
    /// The guest stack pointer at the trap (`rsp`, pointing at the pushed
    /// return address). Diagnostic only (0 when unavailable): lets a handler
    /// walk the caller's stack for a return-address chain.
    pub caller_rsp: u64,
    /// The **floating-point** arguments: the low 64 bits of `XMM0..XMM7`, in
    /// SysV order.
    ///
    /// The integer arguments a handler receives as `&[u64]` come from
    /// `RDI/RSI/RDX/RCX/R8/R9`, but the SysV ABI passes `float`/`double`
    /// arguments in the XMM registers instead — they are *not* in that slice
    /// and are invisible without this. A handler for a function like
    /// `sincosf(float x, float *s, float *c)` reads `x` here and its two
    /// pointers from the integer slice.
    ///
    /// Interpret as `f32::from_bits(bits as u32)` for a `float` argument, or
    /// `f64::from_bits(bits)` for a `double`. Zeroed in tests and on any path
    /// that has no register context.
    pub float_args: [u64; 8],
}

impl HleContext<'_> {
    /// The `n`-th SysV floating-point argument as an `f32` (a `float`), read
    /// from `XMM{n}`'s low half. Out-of-range indices yield `0.0` rather than
    /// panicking, matching how the integer-argument slice degrades.
    #[must_use]
    pub fn float_arg_f32(&self, n: usize) -> f32 {
        f32::from_bits(self.float_args.get(n).copied().unwrap_or(0) as u32)
    }

    /// The `n`-th SysV floating-point argument as an `f64` (a `double`).
    #[must_use]
    pub fn float_arg_f64(&self, n: usize) -> f64 {
        f64::from_bits(self.float_args.get(n).copied().unwrap_or(0))
    }
}

/// HLE function signature: takes a dispatch context and **integer** arguments,
/// returns a result. Floating-point arguments arrive separately, in
/// [`HleContext::float_args`] — the SysV ABI passes them in XMM registers.
pub type HleFunction = fn(&HleContext, &[u64]) -> u64;

fn canonical_provider_name(provider: &str) -> String {
    let lower = provider.to_ascii_lowercase();
    let lower = lower
        .strip_suffix(".sprx")
        .or_else(|| lower.strip_suffix(".prx"))
        .unwrap_or(&lower);
    // Keep in lockstep with `xps5x_firmware::dynlib::nid::canonical_provider_name`
    // — the two halves of one provider identity. `.native` / `_native` is a
    // spelling of the same library (retail imports `libSceMsgDialog.native`).
    lower
        .strip_suffix(".native")
        .or_else(|| lower.strip_suffix("_native"))
        .unwrap_or(lower)
        .to_string()
}

/// Registry of all HLE'd library functions.
pub struct HleRegistry {
    /// Map of "library::function" → implementation.
    functions: DashMap<String, HleFunction>,
    /// Explicit NID → `"library::function"` bindings for functions whose real
    /// name is unknown — see [`HleRegistry::register_nid`].
    nid_overrides: DashMap<(String, u64), String>,
}

impl HleRegistry {
    /// Create and populate the HLE registry with all implemented functions.
    pub fn new() -> Self {
        info!("Initializing HLE registry");
        let registry = Self {
            functions: DashMap::new(),
            nid_overrides: DashMap::new(),
        };

        // Register all implemented HLE functions.
        libkernel::register(&registry);
        // Real pthread mutex/rwlock state machine — supersedes libkernel stubs.
        pthread_sync::register(&registry);
        // pthread thread-attribute objects (stack size / detach state / …).
        pthread_attr::register(&registry);
        // pthread condition variables — signal/broadcast only; wait needs M1-E.
        pthread_cond::register(&registry);
        // pthread thread-specific data (TLS keys).
        pthread_tls::register(&registry);
        // pthread thread identity/control (self / equal / yield / rename).
        pthread_thread::register(&registry);
        // Kernel event flags (create/set/clear/poll/wait/cancel/delete).
        kernel_eventflag::register(&registry);
        // Kernel counting semaphores (create/signal/wait/poll/cancel/delete).
        kernel_semaphore::register(&registry);
        // POSIX (address-based) semaphores — sem_init/wait/timedwait/post.
        posix_sem::register(&registry);
        // Kernel event queues + user events (supersedes libkernel's equeue stubs).
        kernel_equeue::register(&registry);
        // BSD sockets (offline) + pure net helpers (htons/inet_pton/bzero).
        kernel_socket::register(&registry);
        libc::register(&registry);
        // The POSIX-named view of the kernel (`gettimeofday`, ...). A real
        // title's own libc.prx calls these spellings, which are distinct NIDs
        // from the `sce*` ones — see the module docs.
        libsce_posix::register(&registry);
        libsce_sysmodule::register(&registry);
        libsce_video_out::register(&registry);
        libsce_pad::register(&registry);
        libsce_playgo::register(&registry);
        libsce_system_service::register(&registry);
        libsce_text_to_speech2::register(&registry);
        libsce_user_service::register(&registry);
        libsce_audio_out::register(&registry);
        libsce_save_data::register(&registry);
        libsce_save_data_dialog::register(&registry);
        libsce_common_dialog::register(&registry);
        libsce_font::register(&registry);
        libsce_content_export::register(&registry);
        libsce_app_content::register(&registry);
        libsce_np::register(&registry);
        libsce_net::register(&registry);
        libsce_disc_map::register(&registry);
        libsce_rtc::register(&registry);
        libsce_random::register(&registry);
        libsce_peripheral::register(&registry);
        libsce_json::register(&registry);
        libsce_libc_internal::register(&registry);
        libsce_fiber::register(&registry);
        libsce_media::register(&registry);
        libsce_agc::register(&registry);
        libsce_acm::register(&registry);
        libsce_ampr::register(&registry);
        // SharpEmu-ported service-library handshake stubs (no host backend):
        libsce_np_trophy2::register(&registry);
        libsce_np_universal_data::register(&registry);
        libsce_np_web_api2::register(&registry);
        libsce_np_entitlement::register(&registry);
        libsce_np_session_signaling::register(&registry);
        libsce_http::register(&registry);
        libsce_ssl::register(&registry);
        libsce_audio_out2::register(&registry);
        libsce_coredump::register(&registry);
        libsce_share::register(&registry);

        info!(
            "HLE registry: {} functions registered",
            registry.functions.len()
        );
        registry
    }

    /// Register an HLE function.
    pub fn register(&self, library: &str, function: &str, implementation: HleFunction) {
        let key = format!("{}::{}", library, function);
        debug!("HLE register: {}", key);
        self.functions.insert(key, implementation);
    }

    /// Register a function whose real name is **unknown**, binding it to an
    /// explicit NID.
    ///
    /// # Why this is needed
    ///
    /// Resolution normally derives a NID by hashing the function *name*, which
    /// only works when the name is known. Plenty of RE'd functions are known
    /// only by the NID string in a title's symbol table, and the convention
    /// (from SharpEmu) is to name them `<lib>Unknown<NIDSTRING>`. Hashing THAT
    /// placeholder produces a completely different NID, so the implementation
    /// is **unreachable by construction** — it exists, it is registered, and no
    /// import can ever resolve to it.
    ///
    /// That is not hypothetical: `sceAgcUnknownQj7QZpgr9Uw` was implemented and
    /// dead, while the measured retail title imports exactly that NID
    /// (`qj7QZpgr9Uw` = `0xaa3e_d066_982b_f54c`) and reported it missing.
    ///
    /// `function` is still recorded as the human label (logs, diagnostics); the
    /// **NID is the identity**.
    pub fn register_nid(
        &self,
        library: &str,
        function: &str,
        nid: u64,
        implementation: HleFunction,
    ) {
        let key = format!("{}::{}", library, function);
        debug!("HLE register by NID {nid:#018x}: {key}");
        self.nid_overrides
            .insert((canonical_provider_name(library), nid), key.clone());
        self.functions.insert(key, implementation);
    }

    /// Explicit `NID -> "library::function"` bindings (see
    /// [`Self::register_nid`]). The NID database must apply these *in addition*
    /// to the name-hashed ones, or these functions stay unreachable.
    pub fn registered_nid_overrides(&self) -> Vec<(u64, String)> {
        self.nid_overrides
            .iter()
            .map(|entry| (entry.key().1, entry.value().clone()))
            .collect()
    }

    /// Provider-aware explicit bindings used by the firmware linker. Unlike
    /// the legacy diagnostic view above, this preserves the library half of
    /// the import identity when two providers use the same numeric NID.
    pub fn registered_provider_nid_overrides(&self) -> Vec<(String, u64, String)> {
        self.nid_overrides
            .iter()
            .map(|entry| (entry.key().0.clone(), entry.key().1, entry.value().clone()))
            .collect()
    }

    /// Look up and call an HLE function, giving it `ctx` (the kernel +
    /// guest memory) alongside its integer arguments.
    pub fn call(
        &self,
        ctx: &HleContext,
        library: &str,
        function: &str,
        args: &[u64],
    ) -> Option<u64> {
        let key = format!("{}::{}", library, function);
        if let Some(func) = self.functions.get(&key) {
            debug!("HLE call: {}({:?})", key, args);
            let thread = ctx.guest_threads.current_thread();
            if ctx.kernel.diagnostics.is_enabled() {
                let detail = args
                    .iter()
                    .take(14)
                    .map(|arg| format!("{arg:#x}"))
                    .collect::<Vec<_>>()
                    .join(",");
                ctx.kernel.diagnostics.record(
                    thread,
                    DiagnosticKind::HleEnter,
                    &key,
                    ctx.caller_return_addr,
                    detail,
                );
            }
            let result = func(ctx, args);
            ctx.kernel.diagnostics.record(
                thread,
                DiagnosticKind::HleExit,
                &key,
                ctx.caller_return_addr,
                format!("return={result:#x}"),
            );
            Some(result)
        } else {
            warn!("HLE: unimplemented function {}", key);
            None
        }
    }

    /// Check if a function is implemented.
    pub fn is_implemented(&self, library: &str, function: &str) -> bool {
        let key = format!("{}::{}", library, function);
        self.functions.contains_key(&key)
    }

    /// Every registered function as `(library, function)` pairs.
    ///
    /// Each internal key is `"library::function"`; this splits on the first
    /// `"::"` to recover the pair. Used to seed a `NidDatabase` from what the
    /// HLE registry actually implements.
    pub fn registered_names(&self) -> Vec<(String, String)> {
        self.functions
            .iter()
            .filter_map(|entry| {
                entry
                    .key()
                    .split_once("::")
                    .map(|(library, function)| (library.to_string(), function.to_string()))
            })
            .collect()
    }

    /// Function names registered under more than one library whose
    /// registrations do **not** all share the same implementation.
    ///
    /// # Why this matters
    ///
    /// A NID hashes the function **name** alone, so `libSceJson::X` and
    /// `libSceJson2::X` are indistinguishable to `NidDatabase`/`ModuleRegistry`
    /// — one of them wins and the other is unreachable. That is harmless only
    /// while both register the *same* implementation, which is true today
    /// (measured: 11 duplicated names, all sharing an implementation).
    ///
    /// The day someone gives two same-named functions different bodies, a guest
    /// importing from one library will silently run the other's code. This
    /// surfaces that as data instead of leaving it to be discovered by a
    /// mis-executing game — see the `duplicate_names_share_one_implementation`
    /// test, and `xps5x_firmware`'s `NidDatabase::from_hle_names` docs for the
    /// fix if it ever fires (resolution must then key on the library too).
    pub fn duplicate_name_conflicts(&self) -> Vec<String> {
        let mut by_function: std::collections::HashMap<String, Vec<(String, usize)>> =
            std::collections::HashMap::new();
        for entry in self.functions.iter() {
            let Some((library, function)) = entry.key().split_once("::") else {
                continue;
            };
            by_function
                .entry(function.to_string())
                .or_default()
                .push((library.to_string(), *entry.value() as usize));
        }

        let mut conflicts: Vec<String> = by_function
            .into_iter()
            .filter_map(|(function, regs)| {
                let first = regs.first()?.1;
                if regs.iter().all(|(_, addr)| *addr == first) {
                    return None;
                }
                let mut libs: Vec<&str> = regs.iter().map(|(l, _)| l.as_str()).collect();
                libs.sort_unstable();
                Some(format!("{function} (in {})", libs.join(", ")))
            })
            .collect();
        conflicts.sort();
        conflicts
    }
}

impl Default for HleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
/// A tiny in-memory [`GuestMemory`] backed by a `Vec<u8>`, for unit tests
/// that need to exercise real read/write behavior without a runtime.
pub(crate) struct TestMemory(std::cell::RefCell<Vec<u8>>);

#[cfg(test)]
impl TestMemory {
    pub(crate) fn new(size: usize) -> Self {
        Self(std::cell::RefCell::new(vec![0u8; size]))
    }
}

#[cfg(test)]
impl GuestMemory for TestMemory {
    fn read(&self, guest_addr: u64, out: &mut [u8]) -> bool {
        let Ok(addr) = usize::try_from(guest_addr) else {
            return false;
        };
        let buf = self.0.borrow();
        let Some(end) = addr.checked_add(out.len()) else {
            return false;
        };
        if end > buf.len() {
            return false;
        }
        out.copy_from_slice(&buf[addr..end]);
        true
    }

    fn write(&self, guest_addr: u64, data: &[u8]) -> bool {
        let Ok(addr) = usize::try_from(guest_addr) else {
            return false;
        };
        let mut buf = self.0.borrow_mut();
        let Some(end) = addr.checked_add(data.len()) else {
            return false;
        };
        if end > buf.len() {
            return false;
        }
        buf[addr..end].copy_from_slice(data);
        true
    }

    fn validate_range(&self, range: GuestRange, _access: GuestAccess) -> bool {
        range
            .end()
            .and_then(|end| usize::try_from(end).ok())
            .is_some_and(|end| end <= self.0.borrow().len())
    }

    fn is_executable_range(&self, range: GuestRange) -> bool {
        !range.is_empty() && self.validate_range(range, GuestAccess::Read)
    }

    fn is_gpu_visible_range(&self, range: GuestRange) -> bool {
        self.validate_range(range, GuestAccess::ReadWrite)
    }
}

#[cfg(test)]
/// A minimal in-memory [`GuestAllocator`] test double, for unit tests that
/// need a complete [`HleContext`] but don't exercise allocation behavior
/// (nothing calls `ctx.alloc` yet — see [`GuestAllocator`]'s doc comment).
/// `alloc`/`mmap` are a bump allocator over a `Cell<u64>`; `free`/`munmap`
/// are no-ops; `realloc` always bumps a fresh block rather than reusing
/// `addr`.
pub(crate) struct TestAllocator(std::cell::Cell<u64>);

#[cfg(test)]
impl TestAllocator {
    pub(crate) fn new(base: u64) -> Self {
        Self(std::cell::Cell::new(base))
    }

    fn bump(&self, size: u64, align: u64) -> Option<u64> {
        let align = align.max(1);
        let cur = self.0.get();
        let aligned = cur.checked_add(align - 1)? & !(align - 1);
        let next = aligned.checked_add(size)?;
        self.0.set(next);
        Some(aligned)
    }
}

#[cfg(test)]
impl GuestAllocator for TestAllocator {
    fn alloc(&self, size: u64, align: u64) -> Option<u64> {
        self.bump(size, align)
    }

    fn free(&self, _addr: u64) {}

    fn realloc(&self, _addr: u64, new_size: u64) -> Option<u64> {
        self.bump(new_size, 1)
    }

    fn mmap(&self, length: u64, align: u64) -> Option<u64> {
        self.bump(length, align)
    }

    /// Model [`GuestArena::map_at`]: an aligned request is satisfied at exactly
    /// the requested address. The default trait impl declines every non-zero
    /// address, which would make a caller that honors a guest's requested
    /// address look like a caller that ignores it.
    fn map_at(&self, addr: u64, length: u64, align: u64) -> Option<u64> {
        if addr == 0 {
            return self.mmap(length, align);
        }
        if addr % align.max(1) != 0 {
            return None;
        }
        Some(addr)
    }

    fn munmap(&self, _addr: u64, _length: u64) {}
}

/// Build an [`HleContext`] over a test kernel, [`TestMemory`], and
/// [`TestAllocator`]. Defined at the crate root (not inside `mod tests`) so
/// every submodule's own `#[cfg(test)] mod tests` can reach it as
/// `crate::test_ctx` — Rust visibility lets descendant modules see their
/// ancestors' private items.
#[cfg(test)]
pub(crate) fn test_ctx<'a>(
    kernel: &'a xps5x_kernel::OrbisKernel,
    mem: &'a TestMemory,
    alloc: &'a TestAllocator,
) -> HleContext<'a> {
    struct NoGpu;
    impl GpuSubmissionSubsystem for NoGpu {
        fn submit(&self, _words: Vec<u32>, _queue: xps5x_core::subsystems::GpuQueue) {}
        fn map_shader_metadata(
            &self,
            _code_address: u64,
            _data: xps5x_core::subsystems::ShaderMappedData,
        ) {
        }
        fn present_scanout(
            &self,
            _address: u64,
            _descriptor: Option<xps5x_core::subsystems::ScanoutDescriptor>,
        ) {
        }
        fn wait_idle(&self) {}
        fn stats(&self) -> xps5x_core::subsystems::GpuSubmissionStats {
            xps5x_core::subsystems::GpuSubmissionStats::default()
        }
    }
    static NO_GPU: NoGpu = NoGpu;
    test_ctx_with_gpu(kernel, mem, alloc, &NO_GPU)
}

#[cfg(test)]
pub(crate) fn test_ctx_with_gpu<'a>(
    kernel: &'a xps5x_kernel::OrbisKernel,
    mem: &'a TestMemory,
    alloc: &'a TestAllocator,
    gpu: &'a dyn GpuSubmissionSubsystem,
) -> HleContext<'a> {
    struct NoGuestCalls;
    impl GuestCallScheduler for NoGuestCalls {
        fn request(&self, _request: GuestCallRequest) -> bool {
            false
        }
    }
    static NO_GUEST_CALLS: NoGuestCalls = NoGuestCalls;
    struct NoGuestThreads;
    impl GuestThreadScheduler for NoGuestThreads {
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
    static NO_GUEST_THREADS: NoGuestThreads = NoGuestThreads;
    HleContext {
        kernel,
        services: kernel,
        gpu,
        mem,
        alloc,
        guest_calls: &NO_GUEST_CALLS,
        guest_threads: &NO_GUEST_THREADS,
        caller_return_addr: 0,
        caller_rsp: 0,
        float_args: [0; 8],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_memory_capabilities_reject_overflow_and_out_of_bounds_ranges() {
        let memory = TestMemory::new(0x100);
        assert!(GuestRange::new(GuestAddress::new(u64::MAX), 2).is_none());

        let mapped = GuestRange::new(GuestAddress::new(0x20), 4).unwrap();
        let validated =
            ValidatedGuestRange::validate(&memory, mapped, GuestAccess::ReadWrite).unwrap();
        assert!(validated.write(&[1, 2, 3, 4]));
        let mut bytes = [0u8; 4];
        assert!(validated.read(&mut bytes));
        assert_eq!(bytes, [1, 2, 3, 4]);
        assert!(ExecutableGuestMapping::validate(&memory, mapped).is_some());
        assert!(GpuVisibleGuestRange::validate(&memory, mapped).is_some());

        let outside = GuestRange::new(GuestAddress::new(0xFF), 2).unwrap();
        assert!(ValidatedGuestRange::validate(&memory, outside, GuestAccess::Read).is_none());
        assert!(ExecutableGuestMapping::validate(&memory, outside).is_none());
        assert!(GpuVisibleGuestRange::validate(&memory, outside).is_none());
    }

    #[test]
    fn registered_names_splits_library_and_function() {
        let registry = HleRegistry::new();
        let names = registry.registered_names();
        assert!(
            !names.is_empty(),
            "HleRegistry::new() should register some functions"
        );
        assert_eq!(names.len(), registry.functions.len());

        // Every name must round-trip: is_implemented(lib, func) is true for
        // each pair we enumerated.
        for (library, function) in &names {
            assert!(
                registry.is_implemented(library, function),
                "registered_names produced ({library}, {function}) that is_implemented doesn't recognize"
            );
        }
    }

    #[test]
    fn new_registers_substantially_more_than_the_original_three_libraries() {
        let registry = HleRegistry::new();
        // Before this change, `new()` only wired up libSceSysmodule (3
        // functions), libSceVideoOut (5 functions), and libScePad (4
        // functions) — 12 functions total. Broadening libkernel/libc
        // coverage should push this well past that.
        assert!(
            registry.functions.len() > 12,
            "expected substantially more than the original 3-library baseline (12 functions), got {}",
            registry.functions.len()
        );
    }

    #[test]
    fn representative_libkernel_and_libc_functions_are_implemented_and_callable() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = TestMemory::new(0x1000);
        let alloc = TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let samples: &[(&str, &str)] = &[
            ("libkernel", "sceKernelAllocateDirectMemory"),
            ("libkernel", "scePthreadCreate"),
            ("libc", "malloc"),
            ("libc", "memcpy"),
        ];
        for (library, function) in samples {
            assert!(
                registry.is_implemented(library, function),
                "expected {library}::{function} to be implemented"
            );
            let result = registry.call(&ctx, library, function, &[1, 2, 3, 4]);
            assert!(
                result.is_some(),
                "{library}::{function} should return a value, not None"
            );
        }
    }

    #[test]
    fn registered_names_reflects_manual_registration() {
        let registry = HleRegistry {
            functions: DashMap::new(),
            nid_overrides: DashMap::new(),
        };
        fn stub(_ctx: &HleContext, _args: &[u64]) -> u64 {
            0
        }
        registry.register("libFoo", "someFunction", stub);

        let names = registry.registered_names();
        assert_eq!(
            names,
            vec![("libFoo".to_string(), "someFunction".to_string())]
        );
    }

    #[test]
    fn explicit_equal_nids_remain_distinct_per_provider() {
        fn first(_ctx: &HleContext, _args: &[u64]) -> u64 {
            1
        }
        fn second(_ctx: &HleContext, _args: &[u64]) -> u64 {
            2
        }
        let registry = HleRegistry {
            functions: DashMap::new(),
            nid_overrides: DashMap::new(),
        };
        let nid = 0x1234_5678_9abc_def0;
        registry.register_nid("libAlpha", "unknownAlpha", nid, first);
        registry.register_nid("libBeta", "unknownBeta", nid, second);

        let mut bindings = registry.registered_provider_nid_overrides();
        bindings.sort();
        assert_eq!(bindings.len(), 2);
        assert!(bindings.contains(&(
            "libalpha".to_string(),
            nid,
            "libAlpha::unknownAlpha".to_string()
        )));
        assert!(bindings.contains(&(
            "libbeta".to_string(),
            nid,
            "libBeta::unknownBeta".to_string()
        )));
    }

    /// **A NID cannot tell two libraries apart.** It hashes the function name
    /// alone, so if the same name is registered under several libraries they
    /// share one NID and exactly one of them is reachable — whichever
    /// `NidDatabase` picks.
    ///
    /// That is safe only while every such registration runs the same code, and
    /// today it does (`libSceJson`/`libSceJson2` register from one loop; both
    /// `sceNpGameIntentInitialize` registrations are `hle_ok`;
    /// `libkernel`/`libScePosix` `getpid` share `hle_getpid`).
    ///
    /// If this test fails, someone gave two same-named functions different
    /// bodies, and a guest importing from one library is now silently running
    /// the other's. Do NOT "fix" it by renaming: make resolution key on the
    /// library too — the import symbol carries a `library_index`, and
    /// `xps5x_firmware`'s `NidDatabase::from_hle_names` documents the change.
    #[test]
    fn duplicate_names_share_one_implementation() {
        let registry = HleRegistry::new();
        let conflicts = registry.duplicate_name_conflicts();
        assert!(
            conflicts.is_empty(),
            "these function names are registered under multiple libraries with DIFFERENT \
             implementations, but resolution is by NID (= the name alone), so one of each pair \
             is unreachable and guests will silently run the wrong one: {conflicts:#?}"
        );
    }

    /// The detector must actually detect — otherwise the test above passes
    /// vacuously and pins nothing.
    #[test]
    fn duplicate_name_conflicts_flags_a_genuine_divergence() {
        fn a(_ctx: &HleContext, _args: &[u64]) -> u64 {
            1
        }
        fn b(_ctx: &HleContext, _args: &[u64]) -> u64 {
            2
        }
        let registry = HleRegistry {
            functions: DashMap::new(),
            nid_overrides: DashMap::new(),
        };
        // Same name, two libraries, SAME impl -> not a conflict.
        registry.register("libOne", "shared", a);
        registry.register("libTwo", "shared", a);
        // Same name, two libraries, DIFFERENT impls -> a conflict.
        registry.register("libOne", "diverged", a);
        registry.register("libTwo", "diverged", b);

        let conflicts = registry.duplicate_name_conflicts();
        assert_eq!(conflicts.len(), 1, "got {conflicts:#?}");
        assert!(conflicts[0].starts_with("diverged (in libOne, libTwo)"));
    }
}
