//! Process-owned native guest pthread execution.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use xps5x_firmware::LinkedModule;
use xps5x_hle::{GuestAllocator, GuestMemory, GuestThreadScheduler, HleRegistry};
use xps5x_kernel::OrbisKernel;
use xps5x_core::diagnostics::DiagnosticKind;
use xps5x_gpu::{AgcGpuSession, GpuProcessSession};
use xps5x_core::subsystems::GpuSubmissionSubsystem;

use crate::arena::GuestArena;
use crate::dispatch;
use crate::trampoline::TrampolineGuard;
use crate::{RunOutcome, RuntimeError};

const SCE_OK: u64 = 0;
const SCE_KERNEL_ERROR_ESRCH: u64 = 0x8002_0003;
const SCE_KERNEL_ERROR_EAGAIN: u64 = 0x8002_000B;
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;

struct GuestThread {
    join: Option<JoinHandle<Result<RunOutcome, RuntimeError>>>,
    detached: bool,
}

/// Record the calling OS thread's handle under `guest_thread`.
///
/// Diagnostic only, and the only way to see a title that stops making HLE calls:
/// when the guest spins inside its own code the per-thread call ring freezes and
/// says nothing about *where*. With a handle we can suspend the thread and read
/// its RIP. The duplicate is owned by the map for the process lifetime.
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
    if ok != 0 {
        kernel.host_thread_handles.insert(guest_thread, dup as u64);
    }
}

#[cfg(not(windows))]
pub(crate) fn record_host_thread_handle(_kernel: &OrbisKernel, _guest_thread: u64) {}

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

    // Copy the handles out BEFORE suspending anything: holding a DashMap shard
    // guard across a suspend can stop a thread that needs the same shard.
    let handles: Vec<(u64, u64)> = kernel
        .host_thread_handles
        .iter()
        .map(|e| (*e.key(), *e.value()))
        .collect();

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
                continue;
            }
            let mut ctx: Aligned = std::mem::zeroed();
            ctx.0.ContextFlags = CONTEXT_CONTROL_AMD64;
            if GetThreadContext(handle, &mut ctx.0) != 0 {
                out.push((id, ctx.0.Rip));
            }
            ResumeThread(handle);
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
    // SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES.
    const OPTS: u32 = 0x0000_0002 | 0x0000_0004 | 0x0000_0010;
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

/// Suspend each guest host-thread and walk a shallow HOST backtrace: the RIP
/// plus stack qwords that resolve to a loaded module (return addresses), each
/// symbolized to `module+offset` (plus `function` when a PDB symbol resolves).
/// Names exactly where a stalled thread is parked — e.g. an ntdll wait reached
/// through our dispatch/arena/GPU code. Diagnostic only.
#[cfg(windows)]
#[must_use]
pub fn sample_host_backtraces(kernel: &OrbisKernel) -> Vec<(u64, String)> {
    use windows_sys::Win32::System::Diagnostics::Debug::{
        CONTEXT, GetThreadContext, ReadProcessMemory,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, ResumeThread, SuspendThread};
    const CONTEXT_CONTROL_AMD64: u32 = 0x0010_0001;
    #[repr(align(16))]
    struct Aligned(CONTEXT);

    let proc = unsafe { GetCurrentProcess() };
    let handles: Vec<(u64, u64)> = kernel
        .host_thread_handles
        .iter()
        .map(|e| (*e.key(), *e.value()))
        .collect();
    let mut out = Vec::with_capacity(handles.len());
    for (id, raw) in handles {
        let handle = raw as *mut core::ffi::c_void;
        // SAFETY: `handle` is a live duplicated thread handle owned by the kernel
        // map; suspend/get-context/resume are balanced; `ctx` is 16-aligned; the
        // sampler runs on its own host thread so it never suspends itself.
        let (ok, rip, rsp) = unsafe {
            if SuspendThread(handle) == u32::MAX {
                continue;
            }
            let mut ctx: Aligned = std::mem::zeroed();
            ctx.0.ContextFlags = CONTEXT_CONTROL_AMD64;
            let ok = GetThreadContext(handle, &mut ctx.0);
            let r = (ok, ctx.0.Rip, ctx.0.Rsp);
            ResumeThread(handle);
            r
        };
        if ok == 0 {
            continue;
        }
        // A frame label is `module+offset`, plus `(function)` when the PDB has a
        // symbol — only computed for KEPT frames, so the per-qword scan stays cheap.
        let label = |a: u64| -> Option<String> {
            host_module_for_addr(a).map(|m| match symbolize_host_addr(a) {
                Some(f) => format!("{m}({f})"),
                None => m,
            })
        };
        let mut frames = vec![label(rip).unwrap_or_else(|| format!("{rip:#x}"))];
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
            if let Some(sym) = label(u64::from_le_bytes(word)) {
                frames.push(sym);
            }
            sp = sp.wrapping_add(8);
            scanned += 1;
        }
        out.push((id, frames.join(" <- ")));
    }
    out
}

#[cfg(not(windows))]
#[must_use]
pub fn sample_host_backtraces(_kernel: &OrbisKernel) -> Vec<(u64, String)> {
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

/// Read-only ownership census for diagnostics/UI. It exposes lifecycle facts,
/// never the unsafe arena/guard internals themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestProcessSnapshot {
    pub image_bytes: usize,
    pub loaded_modules: usize,
    pub static_tls_modules: usize,
    pub guest_threads: usize,
    pub kernel_handles: usize,
    pub gpu: xps5x_core::subsystems::GpuSubmissionStats,
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
        let gpu = AgcGpuSession::new_process();
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

    fn attributes(&self, attr: u64) -> (u64, bool) {
        let state = (attr != 0)
            .then(|| self.kernel.pthread_attrs.get(&attr).map(|state| *state))
            .flatten();
        let requested = state.map_or(0x10_0000, |state| state.stack_size);
        (
            requested.clamp(0x1_0000, 0x400_0000),
            state.is_some_and(|state| state.detach_state != 0),
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
}

impl GuestProcessHandle {
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
        if thread_out == 0
            || entry < 0x1_0000
            || !self.arena.is_executable_address(entry)
            || !self.arena.write(thread_out, &0u64.to_le_bytes())
        {
            return SCE_KERNEL_ERROR_EINVAL;
        }

        let (stack_size, detached) = self.attributes(attr);
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
        let tls_area = xps5x_firmware::static_tls_total(&self.module.tls_layout);
        let tcb_base = tcb - tls_area;
        // Only a process with at least one `PT_TLS` has a static area; without
        // one, `tcb_base == tcb` and there is no thread-local storage to
        // alias, so `__tls_get_addr` must fall back to its dynamic path rather
        // than be handed the TCB itself.
        let static_tls_block = (tls_area > 0).then_some(tcb_base);
        let handle = self.next_thread.fetch_add(1, Ordering::Relaxed);
        let process = self.clone();
        let host = std::thread::Builder::new()
            .name(format!("xps5x-guest-{handle}"))
            .spawn(move || {
                tracing::info!(guest_thread = handle, entry, "guest pthread started");
                process.kernel.diagnostics.record(
                    handle,
                    DiagnosticKind::TaskOwned,
                    "guest-thread",
                    handle,
                    format!("entry={entry:#x}"),
                );
                record_host_thread_handle(&process.kernel, handle);
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
                process.arena.free(tcb_base);
                process.arena.free(stack_base);
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
