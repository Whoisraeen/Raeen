//! Process management syscall handlers.
//!
//! Handles nanosleep, sysctl, and other process-level operations.

use crate::OrbisKernel;
use tracing::debug;
use xps5x_core::error::KernelError;

/// sys_nanosleep — High-resolution sleep.
pub fn sys_nanosleep(
    _kernel: &OrbisKernel,
    req_addr: u64,
    rem_addr: u64,
) -> Result<u64, KernelError> {
    // In a full implementation, read timespec from emulated memory.
    // For now, sleep a minimal amount.
    debug!(
        "nanosleep(req={:#x}, rem={:#x}) -> sleeping",
        req_addr, rem_addr
    );
    std::thread::sleep(std::time::Duration::from_millis(1));
    Ok(0)
}

/// sys_sysctl — Retrieve system information.
///
/// PS5 games use sysctl to query hardware info (CPU count, memory size, etc.).
/// We return spoofed PS5-like values.
pub fn sys_sysctl(_kernel: &OrbisKernel, args: &[u64]) -> Result<u64, KernelError> {
    let name_addr = args[0];
    let name_len = args[1] as u32;

    debug!(
        "sysctl(name={:#x}, namelen={}) -> spoofed",
        name_addr, name_len
    );

    // Common sysctl queries from PS5 games:
    // - hw.ncpu -> 8
    // - hw.physmem -> 16 GB
    // - kern.ostype -> "FreeBSD"
    // - kern.osrelease -> "11.0-RELEASE"
    //
    // In a full implementation, we'd read the MIB from emulated memory
    // and return appropriate spoofed values.

    Ok(0)
}
