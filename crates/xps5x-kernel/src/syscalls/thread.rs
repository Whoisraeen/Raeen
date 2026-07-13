//! Thread management syscall handlers.
//!
//! Translates PS5 threading operations (thr_new, thr_self, thr_exit, futex)
//! to host OS threading primitives.

use crate::OrbisKernel;
use tracing::{debug, warn};
use xps5x_core::error::KernelError;

/// sys_thr_new — Create a new thread.
pub fn sys_thr_new(
    kernel: &OrbisKernel,
    param_addr: u64,
    param_size: u64,
) -> Result<u64, KernelError> {
    debug!("thr_new(param={:#x}, size={}) -> stubbed", param_addr, param_size);

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
pub fn sys_thr_self(
    _kernel: &OrbisKernel,
    id_ptr: u64,
) -> Result<u64, KernelError> {
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
pub fn sys_thr_exit(
    _kernel: &OrbisKernel,
    status: u64,
) -> Result<u64, KernelError> {
    debug!("thr_exit(status={}) -> thread terminating", status as i64);
    // In a full implementation, this would clean up thread-local state
    // and notify joiners.
    Ok(0)
}

/// sys_futex — Fast userspace locking.
///
/// Implements a subset of the futex operations used by PS5 games.
pub fn sys_futex(
    _kernel: &OrbisKernel,
    args: &[u64],
) -> Result<u64, KernelError> {
    let uaddr = args[0];
    let op = args[1] as i32;
    let val = args[2] as u32;

    const FUTEX_WAIT: i32 = 0;
    const FUTEX_WAKE: i32 = 1;

    match op & 0x7F {
        FUTEX_WAIT => {
            debug!("futex_wait(uaddr={:#x}, val={})", uaddr, val);
            // In a full implementation:
            // 1. Read the value at uaddr from emulated memory
            // 2. If it matches val, block the thread
            // 3. If it doesn't match, return EAGAIN
            Ok(0)
        }
        FUTEX_WAKE => {
            debug!("futex_wake(uaddr={:#x}, val={})", uaddr, val);
            // In a full implementation:
            // Wake up to val threads blocked on uaddr.
            Ok(0)
        }
        _ => {
            warn!("futex: unsupported operation {}", op);
            Ok(0)
        }
    }
}
