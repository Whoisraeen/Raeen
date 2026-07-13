//! HLE libkernel — Core kernel interface re-implementation.
//!
//! Clean-room re-implementation of the PS5 `libkernel.sprx` exports. Function
//! *names* below are factual PS5 API identifiers (not copyrightable); every
//! implementation is original.
//!
//! ## Stub status
//!
//! [`crate::HleFunction`] is `fn(&[u64]) -> u64` — a bare function pointer
//! with no access to a live [`xps5x_kernel::OrbisKernel`] /
//! [`xps5x_kernel::memory::VirtualMemoryManager`] instance and no access to
//! guest memory. So every function here is a self-contained stub: it logs
//! the call via `tracing` and returns a plausible value (an `SCE_OK`-style
//! `0`, a fake handle, or a fake address/size), but does **not** perform the
//! real operation (no memory is actually mapped, no thread is actually
//! created, out-parameters are not written). Routing these to a real
//! `OrbisKernel` instance is a later milestone that needs a richer dispatch
//! signature than a bare fn pointer.

use crate::HleRegistry;
use tracing::debug;

/// Register libkernel HLE functions.
pub fn register(registry: &HleRegistry) {
    // -- Memory --
    registry.register("libkernel", "sceKernelAllocateDirectMemory", hle_allocate_direct_memory);
    registry.register(
        "libkernel",
        "sceKernelAllocateMainDirectMemory",
        hle_allocate_main_direct_memory,
    );
    registry.register("libkernel", "sceKernelReleaseDirectMemory", hle_release_direct_memory);
    registry.register("libkernel", "sceKernelMapDirectMemory", hle_map_direct_memory);
    registry.register("libkernel", "sceKernelMapFlexibleMemory", hle_map_flexible_memory);
    registry.register("libkernel", "sceKernelMunmap", hle_munmap);
    registry.register("libkernel", "sceKernelMmap", hle_mmap);
    registry.register("libkernel", "sceKernelGetDirectMemorySize", hle_get_direct_memory_size);
    registry.register(
        "libkernel",
        "sceKernelAvailableFlexibleMemorySize",
        hle_available_flexible_memory_size,
    );
    registry.register("libkernel", "sceKernelSetVirtualRangeName", hle_set_virtual_range_name);

    // -- Thread / sync --
    registry.register("libkernel", "scePthreadCreate", hle_pthread_create);
    registry.register("libkernel", "scePthreadJoin", hle_pthread_join);
    registry.register("libkernel", "scePthreadExit", hle_pthread_exit);
    registry.register("libkernel", "scePthreadMutexInit", hle_pthread_mutex_init);
    registry.register("libkernel", "scePthreadMutexLock", hle_pthread_mutex_lock);
    registry.register("libkernel", "scePthreadMutexUnlock", hle_pthread_mutex_unlock);
    registry.register("libkernel", "scePthreadCondInit", hle_pthread_cond_init);
    registry.register("libkernel", "scePthreadCondWait", hle_pthread_cond_wait);
    registry.register("libkernel", "scePthreadCondSignal", hle_pthread_cond_signal);
    registry.register("libkernel", "sceKernelCreateEqueue", hle_create_equeue);
    registry.register("libkernel", "sceKernelWaitEqueue", hle_wait_equeue);

    // -- Misc / process / clock --
    registry.register("libkernel", "sceKernelGetProcessType", hle_get_process_type);
    registry.register("libkernel", "sceKernelGetCurrentCpu", hle_get_current_cpu);
    registry.register("libkernel", "sceKernelGettimeofday", hle_gettimeofday);
    registry.register("libkernel", "sceKernelClockGettime", hle_clock_gettime);
    registry.register("libkernel", "sceKernelGetTscFrequency", hle_get_tsc_frequency);
    registry.register("libkernel", "sceKernelUsleep", hle_usleep);
    registry.register("libkernel", "sceKernelGetProcParam", hle_get_proc_param);
    registry.register("libkernel", "sceKernelIsNeoMode", hle_is_neo_mode);
    registry.register("libkernel", "sceKernelGetCpumode", hle_get_cpumode);
    registry.register("libkernel", "sceKernelError", hle_kernel_error);
}

// ---------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------

/// Stub: always reports success (`SCE_OK` = 0). The physical-address
/// out-parameter is not written — no guest memory access from this fn
/// pointer signature.
fn hle_allocate_direct_memory(args: &[u64]) -> u64 {
    debug!(
        "sceKernelAllocateDirectMemory(searchStart={:#x}, searchEnd={:#x}, len={:#x}, alignment={:#x}, memoryType={})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0),
        args.get(3).copied().unwrap_or(0),
        args.get(4).copied().unwrap_or(0)
    );
    0
}

fn hle_allocate_main_direct_memory(args: &[u64]) -> u64 {
    debug!(
        "sceKernelAllocateMainDirectMemory(len={:#x}, alignment={:#x}, memoryType={})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    0
}

fn hle_release_direct_memory(args: &[u64]) -> u64 {
    debug!(
        "sceKernelReleaseDirectMemory(start={:#x}, len={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    0
}

fn hle_map_direct_memory(args: &[u64]) -> u64 {
    debug!(
        "sceKernelMapDirectMemory(len={:#x}, prot={}, alignment={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0),
        args.get(4).copied().unwrap_or(0)
    );
    0
}

fn hle_map_flexible_memory(args: &[u64]) -> u64 {
    debug!(
        "sceKernelMapFlexibleMemory(len={:#x}, prot={})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    0
}

fn hle_munmap(args: &[u64]) -> u64 {
    debug!(
        "sceKernelMunmap(addr={:#x}, len={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    0
}

/// Stub: returns a plausible fake mapped address rather than the real
/// `sceKernelMmap`'s written-through `void **res` out-parameter, since this
/// stub has no guest memory access.
fn hle_mmap(args: &[u64]) -> u64 {
    debug!(
        "sceKernelMmap(addr={:#x}, len={:#x}, prot={}, flags={})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0),
        args.get(3).copied().unwrap_or(0)
    );
    0x0000_2000_0000_0000
}

/// Stub: plausible fixed size (1 GiB), not the real configured direct-memory
/// pool size.
fn hle_get_direct_memory_size(_args: &[u64]) -> u64 {
    debug!("sceKernelGetDirectMemorySize()");
    0x4000_0000
}

/// Stub: plausible fixed size (256 MiB) of "available" flexible memory.
fn hle_available_flexible_memory_size(_args: &[u64]) -> u64 {
    debug!("sceKernelAvailableFlexibleMemorySize()");
    0x1000_0000
}

fn hle_set_virtual_range_name(args: &[u64]) -> u64 {
    debug!(
        "sceKernelSetVirtualRangeName(addr={:#x}, len={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    0
}

// ---------------------------------------------------------------------
// Thread / sync
// ---------------------------------------------------------------------

/// Stub: returns a fake, always-`1` thread handle. No thread is actually
/// spawned — real thread creation needs [`xps5x_kernel::threading`] wiring.
fn hle_pthread_create(_args: &[u64]) -> u64 {
    debug!("scePthreadCreate()");
    1
}

fn hle_pthread_join(args: &[u64]) -> u64 {
    debug!("scePthreadJoin(thread={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_exit(_args: &[u64]) -> u64 {
    // Real scePthreadExit never returns to the caller; the stub returns 0
    // since this fn-pointer signature has no way to unwind the guest thread.
    debug!("scePthreadExit()");
    0
}

fn hle_pthread_mutex_init(args: &[u64]) -> u64 {
    debug!("scePthreadMutexInit(mutex={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_mutex_lock(args: &[u64]) -> u64 {
    debug!("scePthreadMutexLock(mutex={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_mutex_unlock(args: &[u64]) -> u64 {
    debug!("scePthreadMutexUnlock(mutex={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_cond_init(args: &[u64]) -> u64 {
    debug!("scePthreadCondInit(cond={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_cond_wait(args: &[u64]) -> u64 {
    debug!("scePthreadCondWait(cond={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_cond_signal(args: &[u64]) -> u64 {
    debug!("scePthreadCondSignal(cond={:#x})", args.first().copied().unwrap_or(0));
    0
}

/// Stub: returns a fake, always-`1` event-queue handle.
fn hle_create_equeue(_args: &[u64]) -> u64 {
    debug!("sceKernelCreateEqueue()");
    1
}

fn hle_wait_equeue(args: &[u64]) -> u64 {
    debug!("sceKernelWaitEqueue(eq={:#x})", args.first().copied().unwrap_or(0));
    0
}

// ---------------------------------------------------------------------
// Misc / process / clock
// ---------------------------------------------------------------------

/// Stub: `0` = `SCE_KERNEL_PROCESS_TYPE_NORMAL`-style plausible value.
fn hle_get_process_type(_args: &[u64]) -> u64 {
    debug!("sceKernelGetProcessType()");
    0
}

/// Stub: always reports CPU 0.
fn hle_get_current_cpu(_args: &[u64]) -> u64 {
    debug!("sceKernelGetCurrentCpu()");
    0
}

fn hle_gettimeofday(_args: &[u64]) -> u64 {
    // Real function writes a `struct timeval` out-parameter; not writable
    // from this stub. Report success only.
    debug!("sceKernelGettimeofday()");
    0
}

fn hle_clock_gettime(args: &[u64]) -> u64 {
    debug!("sceKernelClockGettime(clockId={})", args.first().copied().unwrap_or(0));
    0
}

/// Stub: plausible PS5 base-clock TSC frequency (1.6 GHz).
fn hle_get_tsc_frequency(_args: &[u64]) -> u64 {
    debug!("sceKernelGetTscFrequency()");
    1_600_000_000
}

fn hle_usleep(args: &[u64]) -> u64 {
    debug!("sceKernelUsleep(usec={})", args.first().copied().unwrap_or(0));
    0
}

/// Stub: returns a fake, non-null pointer value. The real function returns
/// a pointer into the guest's process-parameter block, which this stub
/// cannot construct.
fn hle_get_proc_param(_args: &[u64]) -> u64 {
    debug!("sceKernelGetProcParam()");
    0x0000_1000_0000_0000
}

/// Stub: always reports base-mode (non-Neo/Pro) hardware.
fn hle_is_neo_mode(_args: &[u64]) -> u64 {
    debug!("sceKernelIsNeoMode()");
    0
}

/// Stub: always reports the base CPU mode.
fn hle_get_cpumode(_args: &[u64]) -> u64 {
    debug!("sceKernelGetCpumode()");
    0
}

/// Stub: no last-error state is tracked yet; always reports "no error".
fn hle_kernel_error(_args: &[u64]) -> u64 {
    debug!("sceKernelError()");
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_adds_expected_functions() {
        let registry = HleRegistry::new();
        for name in [
            "sceKernelAllocateDirectMemory",
            "sceKernelAllocateMainDirectMemory",
            "sceKernelReleaseDirectMemory",
            "sceKernelMapDirectMemory",
            "sceKernelMapFlexibleMemory",
            "sceKernelMunmap",
            "sceKernelMmap",
            "sceKernelGetDirectMemorySize",
            "sceKernelAvailableFlexibleMemorySize",
            "sceKernelSetVirtualRangeName",
            "scePthreadCreate",
            "scePthreadJoin",
            "scePthreadExit",
            "scePthreadMutexInit",
            "scePthreadMutexLock",
            "scePthreadMutexUnlock",
            "scePthreadCondInit",
            "scePthreadCondWait",
            "scePthreadCondSignal",
            "sceKernelCreateEqueue",
            "sceKernelWaitEqueue",
            "sceKernelGetProcessType",
            "sceKernelGetCurrentCpu",
            "sceKernelGettimeofday",
            "sceKernelClockGettime",
            "sceKernelGetTscFrequency",
            "sceKernelUsleep",
            "sceKernelGetProcParam",
            "sceKernelIsNeoMode",
            "sceKernelGetCpumode",
            "sceKernelError",
        ] {
            assert!(registry.is_implemented("libkernel", name), "missing libkernel::{name}");
            // Every registered stub must be callable without panicking.
            registry.call("libkernel", name, &[1, 2, 3, 4, 5, 6]);
        }
    }
}
