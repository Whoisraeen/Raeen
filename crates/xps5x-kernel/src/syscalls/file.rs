//! File I/O syscall handlers.
//!
//! Translates PS5 file operations to host filesystem calls
//! through the virtual filesystem layer.

use crate::OrbisKernel;
use tracing::{debug, warn};
use xps5x_core::error::KernelError;

/// sys_read — Read from a file descriptor.
pub fn sys_read(kernel: &OrbisKernel, fd: i32, _buf_addr: u64, count: u64) -> Result<u64, KernelError> {
    let vfs = &kernel.filesystem;

    match vfs.read(fd, count as usize) {
        Ok(data) => {
            let bytes_read = data.len() as u64;
            // In a real implementation, we'd write `data` to the emulated
            // memory at `buf_addr`. For now, log and return bytes read.
            debug!("read(fd={}) -> {} bytes", fd, bytes_read);
            Ok(bytes_read)
        }
        Err(e) => {
            warn!("read(fd={}) failed: {}", fd, e);
            Ok(u64::MAX) // -1 in unsigned.
        }
    }
}

/// sys_write — Write to a file descriptor.
pub fn sys_write(kernel: &OrbisKernel, fd: i32, buf_addr: u64, count: u64) -> Result<u64, KernelError> {
    // Handle stdout (fd=1) and stderr (fd=2) specially.
    if fd == 1 || fd == 2 {
        // In a full implementation, we'd read `count` bytes from
        // emulated memory at `buf_addr` and print them.
        debug!("write to {} ({} bytes at {:#x})", if fd == 1 { "stdout" } else { "stderr" }, count, buf_addr);
        return Ok(count);
    }

    let vfs = &kernel.filesystem;
    // Placeholder: read data from emulated memory and write to VFS.
    let data = vec![0u8; count as usize]; // Placeholder data.
    match vfs.write(fd, &data) {
        Ok(bytes_written) => {
            debug!("write(fd={}) -> {} bytes", fd, bytes_written);
            Ok(bytes_written as u64)
        }
        Err(e) => {
            warn!("write(fd={}) failed: {}", fd, e);
            Ok(u64::MAX)
        }
    }
}

/// sys_open — Open a file.
pub fn sys_open(kernel: &OrbisKernel, path_addr: u64, flags: i32, mode: u32) -> Result<u64, KernelError> {
    // In a full implementation, we'd read the path string from
    // emulated memory at `path_addr`.
    let path = format!("<path@{:#x}>", path_addr);
    debug!("open('{}', flags={:#x}, mode={:#o})", path, flags, mode);

    let vfs = &kernel.filesystem;
    match vfs.open(&path, flags, mode) {
        Ok(fd) => {
            debug!("open('{}') -> fd={}", path, fd);
            Ok(fd as u64)
        }
        Err(e) => {
            warn!("open('{}') failed: {}", path, e);
            Ok(u64::MAX)
        }
    }
}

/// sys_close — Close a file descriptor.
pub fn sys_close(kernel: &OrbisKernel, fd: i32) -> Result<u64, KernelError> {
    let vfs = &kernel.filesystem;
    match vfs.close(fd) {
        Ok(()) => {
            debug!("close(fd={}) -> success", fd);
            Ok(0)
        }
        Err(e) => {
            warn!("close(fd={}) failed: {}", fd, e);
            Ok(u64::MAX)
        }
    }
}

/// sys_lseek — Reposition file offset.
pub fn sys_lseek(_kernel: &OrbisKernel, fd: i32, offset: i64, whence: i32) -> Result<u64, KernelError> {
    debug!("lseek(fd={}, offset={}, whence={}) -> stubbed", fd, offset, whence);
    Ok(0) // Stub: return beginning of file.
}

/// sys_fstat — Get file status.
pub fn sys_fstat(_kernel: &OrbisKernel, fd_or_path: u64, stat_buf: u64) -> Result<u64, KernelError> {
    debug!("fstat({}, buf={:#x}) -> stubbed", fd_or_path, stat_buf);
    // In a full implementation, we'd populate the stat buffer in
    // emulated memory. For now, return success.
    Ok(0)
}
