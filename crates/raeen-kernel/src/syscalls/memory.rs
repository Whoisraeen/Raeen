//! Memory management syscall handlers.
//!
//! Translates PS5 mmap/munmap/mprotect to host memory operations.

use crate::OrbisKernel;
use raeen_core::error::KernelError;
use tracing::debug;

/// sys_mmap — Map pages of memory.
pub fn sys_mmap(
    kernel: &OrbisKernel,
    addr: u64,
    length: u64,
    prot: u32,
    flags: u32,
    fd: i32,
    offset: u64,
) -> Result<u64, KernelError> {
    let mem = &kernel.memory;

    let mapped_addr = mem.mmap(addr, length, prot, flags, fd, offset)?;
    debug!(
        "mmap(addr={:#x}, len={:#x}, prot={:#x}, flags={:#x}) -> {:#x}",
        addr, length, prot, flags, mapped_addr
    );
    Ok(mapped_addr)
}

/// sys_munmap — Unmap pages of memory.
pub fn sys_munmap(kernel: &OrbisKernel, addr: u64, length: u64) -> Result<u64, KernelError> {
    let mem = &kernel.memory;
    mem.munmap(addr, length)?;
    debug!("munmap(addr={:#x}, len={:#x}) -> success", addr, length);
    Ok(0)
}

/// sys_mprotect — Set protection on a region of memory.
pub fn sys_mprotect(
    kernel: &OrbisKernel,
    addr: u64,
    length: u64,
    prot: u32,
) -> Result<u64, KernelError> {
    let mem = &kernel.memory;
    mem.mprotect(addr, length, prot)?;
    debug!(
        "mprotect(addr={:#x}, len={:#x}, prot={:#x}) -> success",
        addr, length, prot
    );
    Ok(0)
}
