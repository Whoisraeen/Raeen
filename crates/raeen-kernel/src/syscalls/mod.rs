//! Syscall dispatcher and handlers.
//!
//! Routes PS5 (Orbis OS / FreeBSD) syscall numbers to their
//! emulated implementations.

pub mod file;
pub mod memory;
pub mod network;
pub mod process;
pub mod thread;

use crate::OrbisKernel;
use raeen_core::error::KernelError;
use tracing::{debug, warn};

// ─── FreeBSD / Orbis syscall numbers ───────────────────────────────
// These are based on FreeBSD 11 (PS5 kernel base) with Sony extensions.

/// read(2)
const SYS_READ: u64 = 3;
/// write(2)
const SYS_WRITE: u64 = 4;
/// open(2)
const SYS_OPEN: u64 = 5;
/// close(2)
const SYS_CLOSE: u64 = 6;
/// getpid(2)
const SYS_GETPID: u64 = 20;
/// access(2)
const SYS_ACCESS: u64 = 33;
/// ioctl(2)
const SYS_IOCTL: u64 = 54;
/// mmap(2)
const SYS_MMAP: u64 = 477;
/// munmap(2)
const SYS_MUNMAP: u64 = 73;
/// mprotect(2)
const SYS_MPROTECT: u64 = 74;
/// lseek(2)
const SYS_LSEEK: u64 = 478;
/// fstat(2)
const SYS_FSTAT: u64 = 551;
/// stat(2)
const SYS_STAT: u64 = 188;
/// nanosleep(2)
const SYS_NANOSLEEP: u64 = 240;
/// sysctl(2)
const SYS_SYSCTL: u64 = 202;

// ─── Sony custom syscalls (600+) ──────────────────────────────────
/// dynlib_dlsym
const SYS_DYNLIB_DLSYM: u64 = 591;
/// dynlib_load_prx
const SYS_DYNLIB_LOAD_PRX: u64 = 594;
/// dynlib_get_proc_param
const SYS_DYNLIB_GET_PROC_PARAM: u64 = 599;
/// thr_new
const SYS_THR_NEW: u64 = 455;
/// thr_self
const SYS_THR_SELF: u64 = 432;
/// thr_exit
const SYS_THR_EXIT: u64 = 431;
/// futex (Sony extension)
const SYS_FUTEX: u64 = 454;

// ─── Additional Sony-specific syscalls ────────────────────────────
/// regmgr_call (registry manager)
const SYS_REGMGR_CALL: u64 = 532;
/// randomized_path
const SYS_RANDOMIZED_PATH: u64 = 602;
/// get_authinfo
const SYS_GET_AUTHINFO: u64 = 612;

/// Dispatch a syscall to the appropriate handler.
///
/// Returns the syscall return value (rax) on success.
pub fn dispatch(kernel: &OrbisKernel, number: u64, args: &[u64]) -> Result<u64, KernelError> {
    match number {
        // ─── File I/O ──────────────────────────────────────
        SYS_READ => {
            debug!(
                "syscall: read(fd={}, buf={:#x}, count={})",
                args[0], args[1], args[2]
            );
            file::sys_read(kernel, args[0] as i32, args[1], args[2])
        }
        SYS_WRITE => {
            debug!(
                "syscall: write(fd={}, buf={:#x}, count={})",
                args[0], args[1], args[2]
            );
            file::sys_write(kernel, args[0] as i32, args[1], args[2])
        }
        SYS_OPEN => {
            debug!(
                "syscall: open(path={:#x}, flags={:#x}, mode={:#o})",
                args[0], args[1], args[2]
            );
            file::sys_open(kernel, args[0], args[1] as i32, args[2] as u32)
        }
        SYS_CLOSE => {
            debug!("syscall: close(fd={})", args[0]);
            file::sys_close(kernel, args[0] as i32)
        }
        SYS_LSEEK => {
            debug!(
                "syscall: lseek(fd={}, offset={}, whence={})",
                args[0], args[1] as i64, args[2]
            );
            file::sys_lseek(kernel, args[0] as i32, args[1] as i64, args[2] as i32)
        }
        SYS_FSTAT | SYS_STAT => {
            debug!("syscall: fstat/stat({})", args[0]);
            file::sys_fstat(kernel, args[0], args[1])
        }
        SYS_ACCESS => {
            debug!("syscall: access(path={:#x}, mode={})", args[0], args[1]);
            Ok(0) // Stub: always accessible.
        }

        // ─── Memory management ─────────────────────────────
        SYS_MMAP => {
            debug!(
                "syscall: mmap(addr={:#x}, len={:#x}, prot={:#x}, flags={:#x}, fd={}, offset={:#x})",
                args[0], args[1], args[2], args[3], args[4] as i32, args[5]
            );
            memory::sys_mmap(
                kernel,
                args[0],
                args[1],
                args[2] as u32,
                args[3] as u32,
                args[4] as i32,
                args[5],
            )
        }
        SYS_MUNMAP => {
            debug!("syscall: munmap(addr={:#x}, len={:#x})", args[0], args[1]);
            memory::sys_munmap(kernel, args[0], args[1])
        }
        SYS_MPROTECT => {
            debug!(
                "syscall: mprotect(addr={:#x}, len={:#x}, prot={:#x})",
                args[0], args[1], args[2]
            );
            memory::sys_mprotect(kernel, args[0], args[1], args[2] as u32)
        }

        // ─── Threading ─────────────────────────────────────
        SYS_THR_NEW => {
            debug!("syscall: thr_new(param={:#x})", args[0]);
            thread::sys_thr_new(kernel, args[0], args[1])
        }
        SYS_THR_SELF => {
            debug!("syscall: thr_self(id_ptr={:#x})", args[0]);
            thread::sys_thr_self(kernel, args[0])
        }
        SYS_THR_EXIT => {
            debug!("syscall: thr_exit(status={})", args[0] as i64);
            thread::sys_thr_exit(kernel, args[0])
        }
        SYS_FUTEX => {
            debug!(
                "syscall: futex(uaddr={:#x}, op={}, val={})",
                args[0], args[1], args[2]
            );
            thread::sys_futex(kernel, args)
        }

        // ─── Process ───────────────────────────────────────
        SYS_GETPID => {
            debug!("syscall: getpid()");
            Ok(1) // Emulated PID = 1.
        }
        SYS_NANOSLEEP => {
            debug!("syscall: nanosleep(req={:#x})", args[0]);
            process::sys_nanosleep(kernel, args[0], args[1])
        }

        // ─── System info ───────────────────────────────────
        SYS_SYSCTL => {
            debug!("syscall: sysctl(name={:#x}, namelen={})", args[0], args[1]);
            process::sys_sysctl(kernel, args)
        }
        SYS_IOCTL => {
            debug!("syscall: ioctl(fd={}, request={:#x})", args[0], args[1]);
            Ok(0) // Stub: success.
        }

        // ─── Sony-specific: Dynamic linking ────────────────
        SYS_DYNLIB_DLSYM => {
            debug!(
                "syscall: dynlib_dlsym(handle={}, symbol={:#x})",
                args[0], args[1]
            );
            Ok(0) // Stub.
        }
        SYS_DYNLIB_LOAD_PRX => {
            debug!("syscall: dynlib_load_prx(name={:#x})", args[0]);
            Ok(0) // Stub.
        }
        SYS_DYNLIB_GET_PROC_PARAM => {
            debug!(
                "syscall: dynlib_get_proc_param(param={:#x}, size={:#x})",
                args[0], args[1]
            );
            Ok(0) // Stub.
        }

        // ─── Sony-specific: Misc ───────────────────────────
        SYS_REGMGR_CALL => {
            debug!("syscall: regmgr_call (stubbed)");
            Ok(0)
        }
        SYS_RANDOMIZED_PATH => {
            debug!("syscall: randomized_path (stubbed)");
            Ok(0)
        }
        SYS_GET_AUTHINFO => {
            debug!("syscall: get_authinfo (stubbed)");
            Ok(0)
        }

        // ─── Unimplemented ─────────────────────────────────
        _ => {
            warn!("Unimplemented syscall: {} (args: {:?})", number, args);
            Err(KernelError::UnimplementedSyscall {
                number,
                name: format!("syscall_{}", number),
            })
        }
    }
}
