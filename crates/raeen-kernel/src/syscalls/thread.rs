//! Thread management syscall handlers.
//!
//! Translates PS5 threading operations (thr_new, thr_self, thr_exit, futex)
//! to host OS threading primitives.

use crate::OrbisKernel;
use raeen_core::error::KernelError;
use tracing::{debug, warn};

/// sys_thr_new — Create a new thread.
pub fn sys_thr_new(
    kernel: &OrbisKernel,
    param_addr: u64,
    param_size: u64,
) -> Result<u64, KernelError> {
    debug!(
        "thr_new(param={:#x}, size={}) -> stubbed",
        param_addr, param_size
    );

    // In a full implementation:
    // 1. Read thr_param struct from emulated memory at param_addr
    // 2. Extract start_func, arg, stack_base, stack_size, tls_base
    // 3. Create a host thread running start_func(arg)
    // 4. Register the thread with the ThreadManager

    let tid = kernel.threads.create_thread(param_addr)?;
    debug!("thr_new -> tid={}", tid);
    Ok(0)
}

/// sys_thr_self — Get the current thread ID.
pub fn sys_thr_self(_kernel: &OrbisKernel, id_ptr: u64) -> Result<u64, KernelError> {
    let tid = std::thread::current().id();
    let numeric_tid: u64 = format!("{:?}", tid)
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or(1);

    debug!("thr_self(id_ptr={:#x}) -> tid={}", id_ptr, numeric_tid);
    // In a full implementation, write numeric_tid to emulated memory at id_ptr.
    Ok(0)
}

/// sys_thr_exit — Terminate the calling thread.
pub fn sys_thr_exit(_kernel: &OrbisKernel, status: u64) -> Result<u64, KernelError> {
    debug!("thr_exit(status={}) -> thread terminating", status as i64);
    // In a full implementation, this would clean up thread-local state
    // and notify joiners.
    Ok(0)
}

/// sys_futex — Fast userspace locking.
///
/// Shares [`OrbisKernel::sync_addresses`] with libkernel's
/// `sceKernelSyncOnAddress{Wait,Wake}` HLE entry points, so a guest that reaches
/// the same watched word through the syscall and through the library still parks
/// in one queue. The syscall path is not the launch hot path today (launches use
/// HLE trampolines, not `syscall` intercept), but a split parking lot would be a
/// silent lost-wake bug the day it is.
///
/// The **value compare** is deliberately absent here: it needs guest-memory
/// access this crate does not own, and the HLE path (which does) performs it.
/// Skipping the compare only costs an unnecessary park, which the bounded slice
/// below then releases — it never loses a wake.
pub fn sys_futex(kernel: &OrbisKernel, args: &[u64]) -> Result<u64, KernelError> {
    let uaddr = args[0];
    let op = args[1] as i32;
    let val = args[2] as u32;

    const FUTEX_WAIT: i32 = 0;
    const FUTEX_WAKE: i32 = 1;
    /// Bounded park, matching the HLE self-heal: a missed wake resolves into the
    /// caller re-checking its own condition rather than a hang.
    const SELF_HEAL: std::time::Duration = std::time::Duration::from_millis(100);

    match op & 0x7F {
        FUTEX_WAIT => {
            let queue = kernel.sync_addresses.queue(uaddr);
            let waiter = queue.enqueue_waiter(0);
            let woken = waiter.wait_for_signal(SELF_HEAL);
            if !woken {
                queue.cancel_waiter(&waiter);
            }
            debug!("futex_wait(uaddr={uaddr:#x}, val={val}) -> woken={woken}");
            Ok(0)
        }
        FUTEX_WAKE => {
            let count = if val == 0 { usize::MAX } else { val as usize };
            let woken = kernel.sync_addresses.wake(uaddr, count);
            debug!("futex_wake(uaddr={uaddr:#x}, val={val}) -> woke {woken}");
            Ok(0)
        }
        _ => {
            warn!("futex: unsupported operation {}", op);
            Ok(0)
        }
    }
}
