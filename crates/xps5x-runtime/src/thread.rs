//! Process-owned native guest pthread execution.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use xps5x_firmware::LinkedModule;
use xps5x_hle::{GuestAllocator, GuestMemory, GuestThreadScheduler, HleRegistry};
use xps5x_kernel::OrbisKernel;

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

/// Every resource a guest worker can outlive its creator with is Arc-owned.
/// This is the C2 ownership boundary: no worker borrows launcher or
/// `execute_process` stack state.
pub(crate) struct GuestProcess {
    pub(crate) module: Arc<LinkedModule>,
    pub(crate) hle: Arc<HleRegistry>,
    pub(crate) kernel: Arc<OrbisKernel>,
    pub(crate) arena: Arc<GuestArena>,
    pub(crate) guard: Arc<TrampolineGuard>,
    next_thread: AtomicU64,
    threads: Mutex<HashMap<u64, GuestThread>>,
    lifecycle: Mutex<()>,
    terminating: AtomicBool,
    exit_code: AtomicU64,
}

#[derive(Clone)]
pub(crate) struct GuestProcessHandle(Arc<GuestProcess>);

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
        GuestProcessHandle(Arc::new(Self {
            module,
            hle,
            kernel,
            arena,
            guard,
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

        let Some(tcb) = self.arena.setup_thread_tcb(self.module.tls.as_ref()) else {
            self.arena.free(stack_base);
            return SCE_KERNEL_ERROR_EAGAIN;
        };
        let tls_block_size = self.module.tls.as_ref().map_or(0, |tls| tls.block_size());
        let tcb_base = tcb - tls_block_size;
        // Only a module with a `PT_TLS` has a static block; without one,
        // `tcb_base == tcb` and there is no thread-local storage to alias, so
        // `__tls_get_addr` must fall back to its dynamic path rather than be
        // handed the TCB itself.
        let static_tls_block = self.module.tls.as_ref().map(|_| tcb_base);
        let handle = self.next_thread.fetch_add(1, Ordering::Relaxed);
        let process = self.clone();
        let host = std::thread::Builder::new()
            .name(format!("xps5x-guest-{handle}"))
            .spawn(move || {
                tracing::info!(guest_thread = handle, entry, "guest pthread started");
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
