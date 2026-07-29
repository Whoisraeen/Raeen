//! Process-owned native guest pthread execution.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use raeen_core::diagnostics::DiagnosticKind;
use raeen_core::subsystems::GpuSubmissionSubsystem;
use raeen_firmware::LinkedModule;
use raeen_gpu::{AgcGpuSession, GpuProcessSession};
use raeen_hle::{
    ExecutableGuestMapping, GuestAccess, GuestAddress, GuestAllocator, GuestMemory, GuestRange,
    GuestThreadScheduler, HleRegistry, ValidatedGuestRange,
};
use raeen_kernel::OrbisKernel;

use crate::arena::GuestArena;
use crate::dispatch;
use crate::trampoline::TrampolineGuard;
use crate::{RunOutcome, RuntimeError};

const SCE_OK: u64 = 0;
const SCE_KERNEL_ERROR_ESRCH: u64 = 0x8002_0003;
const SCE_KERNEL_ERROR_EAGAIN: u64 = 0x8002_000B;
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;

const GUEST_PTHREAD_MIN_STACK_SIZE: u64 = 0x1_0000;
const GUEST_PTHREAD_MAX_STACK_SIZE: u64 = 0x400_0000;

/// Emulator-owned space below the stack size visible through pthread attrs.
///
/// Minecraft requests a 1 MiB stack for its streaming pool, then calls a
/// compiler-produced function with a fixed 0x14a778-byte frame while opening a
/// world. Allocating exactly the reported size crosses the stack base before
/// the function can run. KytyPS5 independently models the platform with a
/// separate 1 MiB runtime reserve for emulator-owned stacks; keep that
/// headroom out of `PthreadAttr::stack_size` so Set/Get still round-trip the
/// title's value exactly.
const GUEST_PTHREAD_RUNTIME_HEADROOM: u64 = 0x10_0000;

fn allocated_guest_stack_size(requested: u64) -> u64 {
    requested.clamp(GUEST_PTHREAD_MIN_STACK_SIZE, GUEST_PTHREAD_MAX_STACK_SIZE)
        + GUEST_PTHREAD_RUNTIME_HEADROOM
}

/// Opt-in A/B gate for mapping Orbis pthread priorities onto live Windows
/// threads. KytyPS5 and SharpEmu independently use the same Orbis thresholds:
/// <=478 is high, >=733 is low, and the middle band is normal.
fn host_thread_priority_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RAEEN_HOST_THREAD_PRIORITY").is_some())
}

#[cfg(windows)]
fn windows_thread_priority(orbis_priority: i32) -> i32 {
    use windows_sys::Win32::System::Threading::{
        THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL,
    };
    if orbis_priority <= 478 {
        THREAD_PRIORITY_HIGHEST
    } else if orbis_priority >= 733 {
        THREAD_PRIORITY_LOWEST
    } else {
        THREAD_PRIORITY_NORMAL
    }
}

#[cfg(windows)]
fn set_windows_thread_priority(handle: u64, orbis_priority: i32) -> bool {
    use windows_sys::Win32::System::Threading::SetThreadPriority;
    // SAFETY: callers pass either a live duplicated thread handle owned by the
    // process table or `GetCurrentThread`'s pseudo-handle. The priority value is
    // one of Windows' documented constants.
    unsafe { SetThreadPriority(handle as *mut _, windows_thread_priority(orbis_priority)) != 0 }
}

struct GuestThread {
    join: Option<JoinHandle<Result<RunOutcome, RuntimeError>>>,
    detached: bool,
}

/// Record the calling OS thread's handle under `guest_thread`.
///
/// This is the only way to see a title that stops making HLE calls: with a
/// handle diagnostics can suspend the thread and read its RIP. The same handle
/// also applies live guest-priority changes under the scheduler A/B gate. The
/// duplicate is owned by the map until this guest thread exits.
#[cfg(windows)]
pub(crate) fn record_host_thread_handle(kernel: &OrbisKernel, guest_thread: u64) {
    use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThread};

    let mut dup = std::ptr::null_mut();
    // SAFETY: `GetCurrentProcess`/`GetCurrentThread` return pseudo-handles valid
    // for this call, and `dup` is a valid out-param. On success the duplicate is
    // a real handle owned by `host_thread_handles`.
    let ok = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            GetCurrentThread(),
            GetCurrentProcess(),
            &mut dup,
            0,
            0,
            DUPLICATE_SAME_ACCESS,
        )
    };
    if ok != 0
        && let Some(old) = kernel.host_thread_handles.insert(guest_thread, dup as u64)
    {
        // SAFETY: `old` was a real duplicated handle owned by this map.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(old as *mut _) };
    }
}

#[cfg(not(windows))]
pub(crate) fn record_host_thread_handle(_kernel: &OrbisKernel, _guest_thread: u64) {}

#[cfg(windows)]
pub(crate) fn release_host_thread_handle(kernel: &OrbisKernel, guest_thread: u64) {
    if let Some((_, raw)) = kernel.host_thread_handles.remove(&guest_thread) {
        // SAFETY: the map owns real handles created by `DuplicateHandle`, and
        // removal transfers that single ownership here.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(raw as *mut _) };
    }
}

#[cfg(not(windows))]
pub(crate) fn release_host_thread_handle(_kernel: &OrbisKernel, _guest_thread: u64) {}

/// Duplicate every sampled handle while its DashMap entry is still guarded.
/// The returned handles are privately owned by the sampler, so a concurrent
/// guest-thread exit may close/remove the process-table handle without making
/// the sampler operate on a stale or subsequently reused Windows handle value.
#[cfg(windows)]
fn duplicate_host_thread_handles(kernel: &OrbisKernel) -> Vec<(u64, u64)> {
    use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let process = unsafe { GetCurrentProcess() };
    kernel
        .host_thread_handles
        .iter()
        .filter_map(|entry| {
            let mut duplicate = std::ptr::null_mut();
            // SAFETY: the map guard pins ownership of `entry.value()` for this
            // call; source/target process pseudo-handles are valid, and the
            // successful duplicate is transferred to the returned vector.
            let ok = unsafe {
                DuplicateHandle(
                    process,
                    *entry.value() as *mut _,
                    process,
                    &mut duplicate,
                    0,
                    0,
                    DUPLICATE_SAME_ACCESS,
                )
            };
            (ok != 0).then_some((*entry.key(), duplicate as u64))
        })
        .collect()
}

/// Suspend each guest thread briefly and read its instruction pointer.
///
/// This answers the one question the HLE-call ring cannot: where is a title
/// stuck when it is spinning in guest code and calling nothing? Returns
/// `(guest_thread_id, rip)`.
///
/// Suspending threads is only safe because of what this does NOT do while one is
/// stopped: the handle list is copied out first (so no map shard stays locked
/// across a suspend), and nothing is logged until every thread is resumed —
/// either would deadlock against a thread suspended inside the same lock.
#[cfg(windows)]
#[must_use]
pub fn sample_guest_rips(kernel: &OrbisKernel) -> Vec<(u64, u64)> {
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, GetThreadContext};
    use windows_sys::Win32::System::Threading::{ResumeThread, SuspendThread};

    /// `CONTEXT_CONTROL` for AMD64 — RIP/RSP/RFLAGS only, which is all we read.
    const CONTEXT_CONTROL_AMD64: u32 = 0x0010_0001;

    #[repr(align(16))]
    struct Aligned(CONTEXT);

    // Own private duplicates BEFORE suspending anything: no DashMap shard stays
    // guarded during a suspend, and concurrent thread teardown cannot make a
    // copied integer stale or retarget it through Windows handle reuse.
    let handles = duplicate_host_thread_handles(kernel);

    let mut out = Vec::with_capacity(handles.len());
    for (id, raw) in handles {
        let handle = raw as *mut core::ffi::c_void;
        // SAFETY: `handle` is a live duplicated thread handle owned by the
        // kernel map. Suspend/GetThreadContext/Resume are balanced on every
        // path, and `ctx` is 16-byte aligned as GetThreadContext requires. The
        // sampler runs on its own host thread, never a guest one, so it cannot
        // suspend itself.
        unsafe {
            if SuspendThread(handle) == u32::MAX {
                windows_sys::Win32::Foundation::CloseHandle(handle);
                continue;
            }
            let mut ctx: Aligned = std::mem::zeroed();
            ctx.0.ContextFlags = CONTEXT_CONTROL_AMD64;
            if GetThreadContext(handle, &mut ctx.0) != 0 {
                out.push((id, ctx.0.Rip));
            }
            ResumeThread(handle);
            windows_sys::Win32::Foundation::CloseHandle(handle);
        }
    }
    out
}

#[cfg(not(windows))]
#[must_use]
pub fn sample_guest_rips(_kernel: &OrbisKernel) -> Vec<(u64, u64)> {
    Vec::new()
}

/// Resolve a HOST (Windows) code address to `module+0xoffset`, or `None` if it
/// is not inside a loaded module. Diagnostic only — symbolizes where a stalled
/// guest thread is parked in *our* code / ntdll.
#[cfg(windows)]
#[must_use]
pub fn host_module_for_addr(addr: u64) -> Option<String> {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleFileNameW, GetModuleHandleExW,
    };
    let mut hmod: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: FROM_ADDRESS reinterprets the name pointer as the address to look
    // up; UNCHANGED_REFCOUNT means the returned handle must not be freed. Both
    // out-params are valid local storage.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            addr as *const u16,
            &mut hmod,
        )
    };
    if ok == 0 || hmod.is_null() {
        return None;
    }
    let mut buf = [0u16; 260];
    // SAFETY: `buf` holds `buf.len()` u16s; the call writes at most that many.
    let len = unsafe { GetModuleFileNameW(hmod, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    let path = String::from_utf16_lossy(&buf[..len.min(buf.len())]);
    let name = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_owned();
    Some(format!("{name}+{:#x}", addr.wrapping_sub(hmod as u64)))
}

/// Resolve a host address to a `function+disp` name via dbghelp + the process
/// PDB. `None` if the symbol can't be found (e.g. system DLLs without symbols).
/// Best-effort and lazily initialized; a 200 MB PDB loads on the first hit.
#[cfg(windows)]
#[must_use]
pub fn symbolize_host_addr(addr: u64) -> Option<String> {
    use std::sync::Once;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        SYMBOL_INFO, SymFromAddr, SymInitialize, SymSetOptions,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SYMOPT_UNDNAME | SYMOPT_LOAD_LINES.
    //
    // `SYMOPT_DEFERRED_LOADS` (0x4) is deliberately **absent**, and its absence
    // is the whole reason this function returns anything for our own module.
    // Measured against the Blasphemous II stall capture: with deferred loads on,
    // `SymFromAddr` answered `ERROR_MOD_NOT_FOUND` (126) for every `raeen.exe`
    // address while still naming `ntdll`/`KERNELBASE` frames — those resolve from
    // the DLLs' export tables and need no PDB, so the failure looked like "our
    // frames just have no symbols" instead of "the PDB was never loaded". Every
    // host backtrace in that capture therefore printed our frames as bare
    // `raeen.exe+0x84df7f` offsets, which is most of why they were unreadable.
    const OPTS: u32 = 0x0000_0002 | 0x0000_0010;
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        SymSetOptions(OPTS);
        SymInitialize(GetCurrentProcess(), core::ptr::null(), 1);
    });
    const MAX_SYM_NAME: usize = 1024;
    // SYMBOL_INFO ends in a variable-length name array; over-allocate as u64s so
    // the struct stays 8-aligned and there's room for the name past its tail.
    let words = std::mem::size_of::<SYMBOL_INFO>() / 8 + MAX_SYM_NAME / 8 + 2;
    let mut buf = vec![0u64; words];
    let info = buf.as_mut_ptr().cast::<SYMBOL_INFO>();
    let mut disp = 0u64;
    // SAFETY: `info` points at `words*8` bytes of zeroed storage — more than
    // `SizeOfStruct + MaxNameLen`; the required header fields are set first.
    let ok = unsafe {
        (*info).SizeOfStruct = std::mem::size_of::<SYMBOL_INFO>() as u32;
        (*info).MaxNameLen = MAX_SYM_NAME as u32;
        SymFromAddr(GetCurrentProcess(), addr, &mut disp, info)
    };
    if ok == 0 {
        return None;
    }
    // SAFETY: on success dbghelp wrote `NameLen` name bytes into the trailing
    // `Name` array; read exactly that many, clamped to our allocation.
    let (name_ptr, name_len) = unsafe {
        (
            (*info).Name.as_ptr().cast::<u8>(),
            ((*info).NameLen as usize).min(MAX_SYM_NAME),
        )
    };
    let bytes = unsafe { std::slice::from_raw_parts(name_ptr, name_len) };
    let name = String::from_utf8_lossy(bytes);
    if name.is_empty() {
        None
    } else if disp == 0 {
        Some(name.into_owned())
    } else {
        Some(format!("{name}+{disp:#x}"))
    }
}

/// One sampled host thread: where its RIP is, whether that place is a kernel
/// wait, and the shallow backtrace that reached it.
#[derive(Debug, Clone)]
pub struct HostThreadSample {
    /// Guest thread id (the key `OrbisKernel` tracks host handles under).
    pub thread: u64,
    /// Host instruction pointer at the moment of sampling.
    pub rip: u64,
    /// `Some(primitive)` when the RIP is inside a Windows kernel wait — i.e. the
    /// thread is **parked**, not running. `None` means either genuinely running
    /// or unclassifiable (see [`host_wait_primitive`]).
    pub parked_in: Option<&'static str>,
    /// `frame <- frame <- …`, innermost first.
    pub chain: String,
}

/// Classify a symbolized host frame as a Windows kernel wait, returning the
/// primitive's normalized name.
///
/// This is how the stall monitor decides that a thread is *parked* rather than
/// running, and it is deliberately a whitelist of syscall entry points rather
/// than "the frame is in ntdll": a thread inside `RtlAllocateHeap` is also in
/// ntdll and is very much running. `None` therefore means "not known to be
/// waiting", never "known to be running" — a distinction the report keeps.
///
/// The names accepted cover both spellings ntdll exports for each syscall
/// (`Nt*` and `Zw*` are the same code at the same address; which one a
/// symbolizer picks is arbitrary), and the trailing `+0xdisp` that
/// [`symbolize_host_addr`] appends.
#[must_use]
pub fn host_wait_primitive(symbol: &str) -> Option<&'static str> {
    // `module+0xoff(Name+0xdisp)` or `Name+0xdisp` or `Name`.
    let name = symbol
        .rsplit('(')
        .next()
        .unwrap_or(symbol)
        .trim_end_matches(')');
    let name = name.split('+').next().unwrap_or(name).trim();
    let bare = name
        .strip_prefix("Zw")
        .or_else(|| name.strip_prefix("Nt"))?;
    Some(match bare {
        // The `WaitOnAddress` futex. Every Rust `std::sync::Mutex`/`Condvar` on
        // Windows and every `parking_lot` park lands here, which is exactly why
        // seeing it is NOT evidence of one library over the other.
        "WaitForAlertByThreadId" => "WaitOnAddress futex (std or parking_lot)",
        "WaitForSingleObject" => "WaitForSingleObject",
        "WaitForMultipleObjects" => "WaitForMultipleObjects",
        "SignalAndWaitForSingleObject" => "SignalAndWaitForSingleObject",
        "WaitForKeyedEvent" => "keyed-event wait",
        "DelayExecution" => "Sleep",
        "RemoveIoCompletion" | "RemoveIoCompletionEx" => "I/O completion wait",
        "WaitForWorkViaWorkerFactory" => "thread-pool idle",
        "WaitForDebugEvent" => "debug-event wait",
        _ => return None,
    })
}

/// Suspend each guest host-thread and walk a shallow HOST backtrace: the RIP
/// plus stack qwords that resolve to a loaded module (return addresses), each
/// symbolized to `module+offset` (plus `function` when a PDB symbol resolves).
/// Names exactly where a stalled thread is parked — e.g. an ntdll wait reached
/// through our dispatch/arena/GPU code. Diagnostic only.
///
/// The per-thread [`HostThreadSample::parked_in`] is what lets the stall monitor
/// count a thread that is stuck in a host wait *outside* any HLE call. Before it
/// existed the monitor's only notion of "stalled" was "currently inside an HLE
/// call", so a run in which every thread was parked reported nothing at all.
#[cfg(windows)]
#[must_use]
pub fn sample_host_backtraces(kernel: &OrbisKernel) -> Vec<HostThreadSample> {
    use windows_sys::Win32::System::Diagnostics::Debug::{
        CONTEXT, GetThreadContext, ReadProcessMemory,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, ResumeThread, SuspendThread};
    const CONTEXT_CONTROL_AMD64: u32 = 0x0010_0001;
    #[repr(align(16))]
    struct Aligned(CONTEXT);

    let proc = unsafe { GetCurrentProcess() };
    let handles = duplicate_host_thread_handles(kernel);
    let mut out = Vec::with_capacity(handles.len());
    for (id, raw) in handles {
        let handle = raw as *mut core::ffi::c_void;
        // SAFETY: `handle` is a live duplicated thread handle owned by the kernel
        // map; suspend/get-context/resume are balanced; `ctx` is 16-aligned; the
        // sampler runs on its own host thread so it never suspends itself.
        let (ok, rip, rsp) = unsafe {
            if SuspendThread(handle) == u32::MAX {
                windows_sys::Win32::Foundation::CloseHandle(handle);
                continue;
            }
            let mut ctx: Aligned = std::mem::zeroed();
            ctx.0.ContextFlags = CONTEXT_CONTROL_AMD64;
            let ok = GetThreadContext(handle, &mut ctx.0);
            let r = (ok, ctx.0.Rip, ctx.0.Rsp);
            ResumeThread(handle);
            windows_sys::Win32::Foundation::CloseHandle(handle);
            r
        };
        if ok == 0 {
            continue;
        }
        // A frame label is `module+offset`, plus `(function)` when the PDB has a
        // symbol — only computed for KEPT frames, so the per-qword scan stays
        // cheap. The bool is "this address is a function's first byte", which a
        // *return* address never is: that is how the stack scan below rejects
        // function pointers and vtable slots it would otherwise print as frames.
        let label = |a: u64| -> Option<(String, bool)> {
            host_module_for_addr(a).map(|m| match symbolize_host_addr(a) {
                // `symbolize_host_addr` omits `+0xdisp` exactly when disp == 0.
                Some(f) => {
                    let entry = !f.contains("+0x");
                    (format!("{m}({f})"), entry)
                }
                None => (m, false),
            })
        };
        let (rip_label, _) = label(rip).unwrap_or_else(|| (format!("{rip:#x}"), false));
        let parked_in = host_wait_primitive(&rip_label);
        let mut frames = vec![rip_label];
        // Poor-man's backtrace: scan up the stack for qwords that land inside a
        // loaded module (return addresses); skip non-code data.
        let mut sp = rsp;
        let mut scanned = 0u32;
        while frames.len() < 12 && scanned < 512 {
            let mut word = [0u8; 8];
            let mut got = 0usize;
            // SAFETY: reads this process's own committed stack via
            // ReadProcessMemory, which reports failure instead of faulting.
            let read_ok = unsafe {
                ReadProcessMemory(
                    proc,
                    sp as *const core::ffi::c_void,
                    word.as_mut_ptr().cast(),
                    8,
                    &mut got,
                )
            };
            if read_ok == 0 || got != 8 {
                break;
            }
            if let Some((sym, entry)) = label(u64::from_le_bytes(word)) {
                // A function entry is a taken address, not a return site; and a
                // qword repeated in adjacent slots is one saved pointer, not two
                // frames. Both were pure noise in the Blasphemous II capture.
                if !entry && frames.last() != Some(&sym) {
                    frames.push(sym);
                }
            }
            sp = sp.wrapping_add(8);
            scanned += 1;
        }
        out.push(HostThreadSample {
            thread: id,
            rip,
            parked_in,
            chain: frames.join(" <- "),
        });
    }
    out
}

#[cfg(not(windows))]
#[must_use]
pub fn sample_host_backtraces(_kernel: &OrbisKernel) -> Vec<HostThreadSample> {
    Vec::new()
}

/// Every resource a guest worker can outlive its creator with is Arc-owned.
/// This is the C2 ownership boundary: no worker borrows launcher or
/// `execute_process` stack state.
pub struct GuestProcess {
    pub(crate) module: Arc<LinkedModule>,
    pub(crate) hle: Arc<HleRegistry>,
    pub(crate) kernel: Arc<OrbisKernel>,
    pub(crate) arena: Arc<GuestArena>,
    pub(crate) guard: Arc<TrampolineGuard>,
    /// Process-owned GPU register state, shader cache, framebuffers, and
    /// ordered submission worker. The Shell holds only an observer clone.
    pub(crate) gpu: GpuProcessSession,
    next_thread: AtomicU64,
    threads: Mutex<HashMap<u64, GuestThread>>,
    lifecycle: Mutex<()>,
    terminating: AtomicBool,
    exit_code: AtomicU64,
}

#[derive(Clone)]
pub struct GuestProcessHandle(Arc<GuestProcess>);

impl GuestProcessHandle {
    /// A non-owning view of the process for observers (Shell quit/diagnostics)
    /// that must not keep the guest arena's fixed-base mapping alive past
    /// teardown. Only one such mapping can exist per host process, so any
    /// strong handle retained by a session controller races the next launch's
    /// reservation; a `Weak` simply fails to upgrade once the run is over.
    #[must_use]
    pub fn downgrade(&self) -> std::sync::Weak<GuestProcess> {
        Arc::downgrade(&self.0)
    }
}

/// Read-only ownership census for diagnostics/UI. It exposes lifecycle facts,
/// never the unsafe arena/guard internals themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestProcessSnapshot {
    pub image_bytes: usize,
    pub loaded_modules: usize,
    pub static_tls_modules: usize,
    pub guest_threads: usize,
    pub kernel_handles: usize,
    pub gpu: raeen_core::subsystems::GpuSubmissionStats,
    pub diagnostic_events: usize,
}

impl std::ops::Deref for GuestProcessHandle {
    type Target = GuestProcess;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GuestProcess {
    pub(crate) fn create(
        module: Arc<LinkedModule>,
        hle: Arc<HleRegistry>,
        kernel: Arc<OrbisKernel>,
        arena: Arc<GuestArena>,
        guard: Arc<TrampolineGuard>,
    ) -> GuestProcessHandle {
        let gpu = AgcGpuSession::new_process(arena.clone());
        AgcGpuSession::install_process(&gpu);
        kernel.diagnostics.record(
            1,
            DiagnosticKind::TaskOwned,
            "guest-main",
            1,
            "process owner",
        );
        GuestProcessHandle(Arc::new(Self {
            module,
            hle,
            kernel,
            arena,
            guard,
            gpu,
            next_thread: AtomicU64::new(2),
            threads: Mutex::new(HashMap::new()),
            lifecycle: Mutex::new(()),
            terminating: AtomicBool::new(false),
            exit_code: AtomicU64::new(0),
        }))
    }

    fn attributes(&self, attr: u64) -> (u64, bool, i32) {
        let state = (attr != 0)
            .then(|| self.kernel.pthread_attrs.get(&attr).map(|state| *state))
            .flatten();
        let requested = state.map_or(0x10_0000, |state| state.stack_size);
        (
            allocated_guest_stack_size(requested),
            state.is_some_and(|state| state.detach_state != 0),
            state.map_or_else(
                || raeen_kernel::PthreadAttr::default().sched_priority,
                |state| state.sched_priority,
            ),
        )
    }

    fn begin_termination(&self, code: u64) {
        if self
            .terminating
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.exit_code.store(code, Ordering::Release);
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> GuestProcessSnapshot {
        let guest_threads = self
            .threads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
            + 1;
        GuestProcessSnapshot {
            image_bytes: self.module.image.len(),
            loaded_modules: self.module.unwind_modules.len().max(1),
            static_tls_modules: self.module.tls_layout.len(),
            guest_threads,
            kernel_handles: self.kernel.modules.len()
                + self.kernel.kernel_event_flags.len()
                + self.kernel.kernel_equeues.len()
                + self.kernel.kernel_sockets.len(),
            gpu: self.gpu.stats(),
            diagnostic_events: self.kernel.diagnostics.snapshot().len(),
        }
    }

    /// Cooperatively request process termination. Blocking HLE waits observe
    /// this flag and native execution exits at the next runtime/HLE boundary;
    /// [`GuestProcessHandle::terminate_and_reap`] remains the sole owner of
    /// joining workers and draining GPU work. Defined on `GuestProcess` (not
    /// the handle) so a session controller holding only a `Weak<GuestProcess>`
    /// can upgrade and request termination without owning the arena.
    pub fn request_termination(&self, code: u64) {
        self.begin_termination(code);
        self.kernel.event_flag_signal.1.notify_all();
    }
}

impl GuestProcessHandle {
    #[must_use]
    pub fn is_terminating(&self) -> bool {
        self.terminating.load(Ordering::Acquire)
    }

    /// Stop accepting new workers and wait until every internally retained
    /// host handle has left guest dispatch. Detached is a guest-visible
    /// joinability state only; it never discards the runtime's safety handle.
    pub(crate) fn terminate_and_reap(&self, code: u64) {
        self.begin_termination(code);
        let _lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let handles: Vec<_> = {
            let mut threads = self
                .threads
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            threads
                .drain()
                .filter_map(|(handle, record)| record.join.map(|join| (handle, join)))
                .collect()
        };
        for (handle, join) in handles {
            if let Err(payload) = join.join() {
                tracing::warn!(
                    "guest thread {handle:#x} panicked during process teardown: {payload:?}"
                );
            }
        }
        self.gpu.shutdown();
        self.kernel.diagnostics.record(
            1,
            DiagnosticKind::TaskReleased,
            "guest-main",
            1,
            format!("process exit={code}"),
        );
    }

    pub(crate) fn requested_exit_code(&self) -> Option<u64> {
        self.terminating
            .load(Ordering::Acquire)
            .then(|| self.exit_code.load(Ordering::Acquire))
    }
}

impl GuestThreadScheduler for GuestProcessHandle {
    fn create(&self, thread_out: u64, attr: u64, entry: u64, arg: u64) -> u64 {
        let _lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.terminating.load(Ordering::Acquire) {
            return SCE_KERNEL_ERROR_EAGAIN;
        }
        let executable_entry = GuestRange::new(GuestAddress::new(entry), 1)
            .and_then(|range| ExecutableGuestMapping::validate(self.arena.as_ref(), range))
            .is_some();
        let writable_thread_out = GuestRange::new(GuestAddress::new(thread_out), 8)
            .and_then(|range| {
                ValidatedGuestRange::validate(self.arena.as_ref(), range, GuestAccess::Write)
            })
            .is_some();
        if thread_out == 0
            || entry < 0x1_0000
            || !executable_entry
            || !writable_thread_out
            || !self.arena.write(thread_out, &0u64.to_le_bytes())
        {
            return SCE_KERNEL_ERROR_EINVAL;
        }

        let (stack_size, detached, priority) = self.attributes(attr);
        // Round down to a 16-byte multiple so the entry RSP (`stack_base +
        // stack_size - 8`, over a 16-aligned base) meets the SysV AMD64 entry
        // contract (RSP ≡ 8 mod 16). A guest-supplied odd stack size would
        // otherwise misalign the worker's first aligned-SSE store (#GP).
        let stack_size = stack_size & !0xF;
        let Some(stack_base) = self.arena.alloc(stack_size, 16) else {
            return SCE_KERNEL_ERROR_EAGAIN;
        };
        let Some(stack_rsp) = stack_base
            .checked_add(stack_size)
            .and_then(|top| top.checked_sub(8))
        else {
            self.arena.free(stack_base);
            return SCE_KERNEL_ERROR_EAGAIN;
        };
        if !self
            .arena
            .write(stack_rsp, &self.guard.return_trampoline().to_le_bytes())
        {
            self.arena.free(stack_base);
            return SCE_KERNEL_ERROR_EAGAIN;
        }

        let Some(tcb) = self.arena.setup_thread_tcb(&self.module.tls_layout) else {
            self.arena.free(stack_base);
            return SCE_KERNEL_ERROR_EAGAIN;
        };
        let tls_area = raeen_firmware::static_tls_total(&self.module.tls_layout);
        let tcb_base = tcb - tls_area;
        // Only a process with at least one `PT_TLS` has a static area; without
        // one, `tcb_base == tcb` and there is no thread-local storage to
        // alias, so `__tls_get_addr` must fall back to its dynamic path rather
        // than be handed the TCB itself.
        let static_tls_block = (tls_area > 0).then_some(tcb_base);
        let handle = self.next_thread.fetch_add(1, Ordering::Relaxed);
        self.kernel.thread_priorities.insert(handle, priority);
        let process = self.clone();
        let host = std::thread::Builder::new()
            .name(format!("raeen-guest-{handle}"))
            .spawn(move || {
                tracing::info!(
                    guest_thread = handle,
                    entry,
                    stack_base,
                    stack_size,
                    orbis_priority = priority,
                    "guest pthread started"
                );
                // Stamped where the worker actually BEGINS, not where it is
                // reaped: `terminate_and_reap`'s join loop would place the
                // phase at teardown, and a hard-killed stalled title — exactly
                // the class this chain exists to diagnose — never reaches it.
                raeen_core::frame_path::record_phase(
                    raeen_core::frame_path::Phase::FirstGuestThread,
                );
                process.kernel.diagnostics.record(
                    handle,
                    DiagnosticKind::TaskOwned,
                    "guest-thread",
                    handle,
                    format!("entry={entry:#x}"),
                );
                record_host_thread_handle(&process.kernel, handle);
                #[cfg(windows)]
                if host_thread_priority_enabled() {
                    use windows_sys::Win32::System::Threading::GetCurrentThread;
                    // SAFETY: returns a pseudo-handle valid on this thread.
                    let current = unsafe { GetCurrentThread() } as u64;
                    if !set_windows_thread_priority(current, priority) {
                        tracing::warn!(
                            guest_thread = handle,
                            orbis_priority = priority,
                            "failed to apply guest pthread priority to host thread"
                        );
                    }
                }
                // SAFETY: all process resources are Arc-owned by this worker;
                // entry and stack were validated in the live identity-mapped
                // arena, and dispatch installs this OS thread's TLS context
                // before the diverging transfer.
                let result = unsafe {
                    dispatch::run(
                        &process.module.hle_trampolines,
                        &process.module.unresolved_stubs,
                        &process.hle,
                        &process.kernel,
                        &*process.arena,
                        &*process.arena,
                        &process.gpu,
                        &process.guard,
                        Some(tcb),
                        static_tls_block,
                        Some(&process),
                        handle,
                        || crate::stack::enter_guest(entry, stack_rsp, [arg, 0, 0, 0, 0, 0]),
                    )
                };
                // A worker torn down by a host-detected fault (a VEH-trapped
                // null/wild dereference that ends `dispatch::run` with
                // `RuntimeError::Faulted`, or any other abnormal error) never ran
                // its C++ unlock/cleanup, so every lock it still held stays held
                // forever. Now that mutexes and rwlocks truly block, its waiters
                // hang indefinitely — the measured "scePthreadMutexLock stuck >3s
                // — deadlock" cascade during ASTRO.BOT level transitions.
                //
                // This runs on EVERY exit path, not just the errored one. The
                // "a clean Returned/Exited already unlocked what it held"
                // assumption this code used to make is false in both directions:
                //   * `Ok(Returned)` is also how `scePthreadExit` ends a worker.
                //     A thread that calls it from inside a critical section — the
                //     ordinary shape of a C++ worker that throws, catches, and
                //     exits, or of any thread cancelled by its own logic — never
                //     reaches its unlock either, and left the mutex held forever.
                //   * `Ok(Exited)` is cooperative process termination: the VEH
                //     abandons the guest stack at the next safe point, which is
                //     just as abrupt as a fault, only tidier at the top.
                // Unconditional is also cheap and idempotent: a thread holding
                // nothing produces an all-zero summary and logs nothing.
                let freed = process.kernel.release_locks_owned_by(handle);
                let exit = match &result {
                    Ok(RunOutcome::Returned(_)) => "returned/pthread-exit",
                    Ok(RunOutcome::Exited(_)) => "process-exit",
                    Err(_) => "faulted",
                };
                // A guest thread's exit was logged NOWHERE unless it happened to
                // die holding a lock, and `RunOutcome` reached a human only if
                // some other guest thread later called `scePthreadJoin`. A
                // detached worker therefore died in complete silence — and the
                // only evidence was its absence from the next stall dump's
                // thread inventory.
                //
                // That cost a whole investigation. Blasphemous II's IL2CPP
                // collector raises SIGUSR1, waits for the mutator's
                // acknowledgement, and must then set "resumeEvent" to release
                // the world. It acknowledges, and then its guest thread ends
                // without ever setting the flag — so every other thread parks on
                // `resumeEvent` forever. Whether it RETURNED or FAULTED is the
                // whole question, and neither was recoverable after the fact.
                //
                // Unconditional, at INFO: one line per guest thread for a whole
                // session, and the thread that ends unexpectedly names itself.
                let name = process
                    .kernel
                    .thread_names
                    .get(&handle)
                    .map(|n| n.clone())
                    .unwrap_or_default();
                tracing::info!(
                    guest_thread = handle,
                    name = %name,
                    exit,
                    outcome = ?result,
                    "guest thread exited"
                );
                if freed.any() {
                    tracing::warn!(
                        guest_thread = handle,
                        exit,
                        mutexes = freed.mutexes,
                        rwlock_writers = freed.rwlock_writers,
                        rwlock_read_holds = freed.rwlock_read_holds,
                        cond_waiters = freed.cond_waiters,
                        "guest worker exited still holding sync state; released it so waiters can proceed"
                    );
                }
                // An exception raised at a thread that then exits has nowhere to
                // be delivered. Discard it — and, just as importantly, keep the
                // pending set empty, because a stale entry would leave
                // `has_pending_exceptions` true forever and turn every later HLE
                // call's one-atomic fast path into a map lookup.
                if process.kernel.discard_pending_exception(handle) {
                    tracing::warn!(
                        guest_thread = handle,
                        "guest worker exited with an undelivered sceKernelRaiseException; \
                         discarded (the raiser's wait for it will not be satisfied)"
                    );
                }
                process
                    .kernel
                    .pthread_tls_values
                    .retain(|(thread, _), _| *thread != handle);
                process
                    .kernel
                    .dynamic_tls_blocks
                    .retain(|(thread, _), block| {
                        if *thread == handle {
                            process.arena.free(*block);
                            false
                        } else {
                            true
                        }
                    });
                // Drop the stack registration BEFORE the stack memory is
                // returned to the arena: once freed, the same address range can
                // be handed out as an ordinary heap object, and a stale
                // registration would make the guard misclassify that object as
                // a caller frame and refuse to initialize it.
                process.kernel.guest_thread_stacks.remove(&handle);
                process.arena.free(tcb_base);
                process.arena.free(stack_base);
                release_host_thread_handle(&process.kernel, handle);
                process.kernel.diagnostics.record(
                    handle,
                    DiagnosticKind::TaskReleased,
                    "guest-thread",
                    handle,
                    "host worker exited",
                );
                result
            });
        let Ok(host) = host else {
            // The worker closure never ran, so its in-closure cleanup will not
            // free this thread's TCB / static-TLS block — release it here
            // alongside the stack, or repeated spawn failures leak the guest
            // heap one TCB at a time.
            self.arena.free(tcb_base);
            self.arena.free(stack_base);
            return SCE_KERNEL_ERROR_EAGAIN;
        };

        self.threads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                handle,
                GuestThread {
                    join: Some(host),
                    detached,
                },
            );
        if !self.arena.write(thread_out, &handle.to_le_bytes()) {
            return SCE_KERNEL_ERROR_EINVAL;
        }
        SCE_OK
    }

    fn join(&self, thread: u64, retval_out: u64) -> u64 {
        let host = {
            let mut threads = self
                .threads
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(record) = threads.get_mut(&thread) else {
                return SCE_KERNEL_ERROR_ESRCH;
            };
            if record.detached {
                return SCE_KERNEL_ERROR_EINVAL;
            }
            let Some(host) = record.join.take() else {
                return SCE_KERNEL_ERROR_EINVAL;
            };
            host
        };

        let retval = match host.join() {
            Ok(Ok(RunOutcome::Returned(value) | RunOutcome::Exited(value))) => value,
            Ok(Err(err)) => {
                tracing::warn!("guest thread {thread:#x} faulted: {err}");
                return SCE_KERNEL_ERROR_EINVAL;
            }
            Err(_) => return SCE_KERNEL_ERROR_EINVAL,
        };
        self.threads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&thread);
        if retval_out != 0 && !self.arena.write(retval_out, &retval.to_le_bytes()) {
            return SCE_KERNEL_ERROR_EINVAL;
        }
        SCE_OK
    }

    fn detach(&self, thread: u64) -> u64 {
        let mut threads = self
            .threads
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(record) = threads.get_mut(&thread) else {
            return SCE_KERNEL_ERROR_ESRCH;
        };
        if record.detached {
            return SCE_KERNEL_ERROR_EINVAL;
        }
        record.detached = true;
        SCE_OK
    }

    fn set_priority(&self, thread: u64, priority: i32) -> bool {
        if !host_thread_priority_enabled() {
            return true;
        }
        #[cfg(windows)]
        {
            let Some(handle) = self.kernel.host_thread_handles.get(&thread) else {
                return false;
            };
            set_windows_thread_priority(*handle, priority)
        }
        #[cfg(not(windows))]
        {
            let _ = (thread, priority);
            false
        }
    }

    // The two calls below are per-thread questions, and the process cannot
    // answer them: HLE always holds the running thread's `ActiveContext`
    // (`dispatch.rs`, `guest_threads: ctx`), which answers both from its own
    // per-run state and never delegates them here. These exist only to satisfy
    // the trait. Do not "fix" them into real-looking answers — a process-wide
    // exit flag or a hardcoded handle would be wrong for every worker.
    fn request_exit(&self, _retval: u64) -> bool {
        false
    }

    fn current_thread(&self) -> u64 {
        1
    }

    fn request_process_exit(&self, code: u64) {
        self.begin_termination(code);
    }

    fn process_is_terminating(&self) -> bool {
        self.terminating.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use super::windows_thread_priority;
    use super::{
        GUEST_PTHREAD_MAX_STACK_SIZE, GUEST_PTHREAD_MIN_STACK_SIZE, GUEST_PTHREAD_RUNTIME_HEADROOM,
        allocated_guest_stack_size, host_wait_primitive,
    };

    /// The exact frame label every thread in the Blasphemous II stall capture
    /// carried. Classifying it is what turns "STALL_DUMP (0 threads)" into a
    /// count, so it is pinned by test rather than left to the format's mercy.
    #[test]
    fn wait_on_address_frame_is_recognized_as_a_park() {
        let frame = "ntdll.dll+0x163cb4(ZwWaitForAlertByThreadId+0x14)";
        assert_eq!(
            host_wait_primitive(frame),
            Some("WaitOnAddress futex (std or parking_lot)")
        );
        // Both ntdll spellings are the same syscall at the same address, so the
        // symbolizer's choice between them must not change the verdict.
        assert_eq!(
            host_wait_primitive("ntdll.dll+0x163cb4(NtWaitForAlertByThreadId+0x14)"),
            host_wait_primitive(frame)
        );
    }

    /// `None` must mean "not known to be waiting", and in particular a thread
    /// inside the ntdll heap is running. The capture's stack scan surfaced
    /// `RtlAllocateHeap` frames, so a module-based rule would have called a busy
    /// allocator a park.
    #[test]
    fn running_and_unsymbolized_frames_are_not_parks() {
        for frame in [
            "ntdll.dll+0x3e732(RtlAllocateHeap+0xad2)",
            "ntdll.dll+0x7a81d(RtlReAllocateHeap+0x4d)",
            "KERNELBASE.dll+0xde558(WaitOnAddress+0x38)", // a caller, not the syscall
            "raeen.exe+0x84df7f",
            "0x7ff8c3dc3cb4",
            "",
        ] {
            assert_eq!(
                host_wait_primitive(frame),
                None,
                "must not claim a park for {frame:?}"
            );
        }
    }

    #[test]
    fn the_other_windows_wait_syscalls_are_classified() {
        for (frame, expect) in [
            (
                "ntdll.dll+0x1(NtWaitForSingleObject+0x14)",
                "WaitForSingleObject",
            ),
            ("ntdll.dll+0x1(ZwDelayExecution+0x14)", "Sleep"),
            (
                "ntdll.dll+0x1(NtWaitForMultipleObjects)",
                "WaitForMultipleObjects",
            ),
            ("ntdll.dll+0x1(ZwWaitForKeyedEvent+0x2)", "keyed-event wait"),
            (
                "ntdll.dll+0x1(ZwWaitForWorkViaWorkerFactory+0x9)",
                "thread-pool idle",
            ),
        ] {
            assert_eq!(host_wait_primitive(frame), Some(expect), "for {frame:?}");
        }
    }

    #[test]
    fn runtime_owned_pthread_stack_adds_headroom_without_changing_requested_size() {
        let requested = 0x10_0000;
        assert_eq!(allocated_guest_stack_size(requested), 0x20_0000);
        assert_eq!(requested, 0x10_0000);
    }

    #[test]
    fn runtime_owned_pthread_stack_clamps_request_before_adding_headroom() {
        assert_eq!(
            allocated_guest_stack_size(1),
            GUEST_PTHREAD_MIN_STACK_SIZE + GUEST_PTHREAD_RUNTIME_HEADROOM
        );
        assert_eq!(
            allocated_guest_stack_size(u64::MAX),
            GUEST_PTHREAD_MAX_STACK_SIZE + GUEST_PTHREAD_RUNTIME_HEADROOM
        );
    }

    #[cfg(windows)]
    #[test]
    fn orbis_priorities_map_to_the_reference_host_bands() {
        use windows_sys::Win32::System::Threading::{
            THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL,
        };

        assert_eq!(windows_thread_priority(256), THREAD_PRIORITY_HIGHEST);
        assert_eq!(windows_thread_priority(478), THREAD_PRIORITY_HIGHEST);
        assert_eq!(windows_thread_priority(479), THREAD_PRIORITY_NORMAL);
        assert_eq!(windows_thread_priority(700), THREAD_PRIORITY_NORMAL);
        assert_eq!(windows_thread_priority(732), THREAD_PRIORITY_NORMAL);
        assert_eq!(windows_thread_priority(733), THREAD_PRIORITY_LOWEST);
        assert_eq!(windows_thread_priority(767), THREAD_PRIORITY_LOWEST);
    }
}
