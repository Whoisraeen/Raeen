//! HLE libkernel — Core kernel interface re-implementation.
//!
//! Clean-room re-implementation of the PS5 `libkernel.sprx` exports. Function
//! *names* below are factual PS5 API identifiers (not copyrightable); every
//! implementation is original.
//!
//! ## Stub status
//!
//! Every HLE call now gets an [`crate::HleContext`] (a live
//! [`xps5x_kernel::OrbisKernel`] plus guest-memory access), so functions
//! *can* do real work. Most functions below still just log the call and
//! return a plausible value (an `SCE_OK`-style `0`, a fake handle, or a
//! fake address/size) — thread creation, event queues, and most
//! out-parameters still aren't backed by real state. `sceKernelAllocateDirectMemory`
//! and `sceKernelMapFlexibleMemory` are the exceptions: they route through
//! `ctx.kernel.memory.mmap` and write their out-parameter through
//! `ctx.mem`, as a proof that the context threads all the way through.
//! Broadening the rest is future work, not a limitation of the dispatch
//! signature anymore.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK` — the PS5 convention for "this call succeeded".
const SCE_OK: u64 = 0;

/// Generic failure sentinel used by the functions below that now attempt a
/// real operation (`ctx.kernel.memory.mmap`/`ctx.mem` access) and can
/// genuinely fail. Not a real `SCE_KERNEL_ERROR_*` code — just a nonzero
/// value distinguishable from `SCE_OK`.
const HLE_ERROR: u64 = 0xFFFF_FFFF;

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

/// Real signature: `sceKernelAllocateDirectMemory(off_t searchStart, off_t
/// searchEnd, size_t len, size_t alignment, int memoryType, off_t
/// *physAddrOut)`.
///
/// Partial integration: real hardware separates *reserving* physical
/// memory (this call) from *mapping* it into the process's virtual address
/// space (`sceKernelMapDirectMemory`). This HLE doesn't yet model physical
/// memory as distinct from virtual mappings, so as a documented shortcut
/// this routes straight through `ctx.kernel.memory.mmap` — treating the
/// "physical" address handed back as already virtual-mapped — and writes
/// it through the `physAddrOut` out-parameter (`args[5]`) via `ctx.mem`.
/// Returns `SCE_OK` on success; `HLE_ERROR` if the mmap fails or
/// `physAddrOut` is out of bounds (bounds-checked, never a panic/OOB).
fn hle_allocate_direct_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let search_start = args.first().copied().unwrap_or(0);
    let search_end = args.get(1).copied().unwrap_or(0);
    let len = args.get(2).copied().unwrap_or(0);
    let alignment = args.get(3).copied().unwrap_or(0);
    let memory_type = args.get(4).copied().unwrap_or(0);
    let phys_addr_out = args.get(5).copied().unwrap_or(0);
    debug!(
        "sceKernelAllocateDirectMemory(searchStart={search_start:#x}, searchEnd={search_end:#x}, len={len:#x}, alignment={alignment:#x}, memoryType={memory_type}, physAddrOut={phys_addr_out:#x})"
    );

    match ctx.kernel.memory.mmap(0, len, 0x3, 0, -1, 0) {
        Ok(addr) => {
            if phys_addr_out != 0 && !ctx.mem.write(phys_addr_out, &addr.to_le_bytes()) {
                warn!("sceKernelAllocateDirectMemory: physAddrOut {phys_addr_out:#x} out of bounds");
                return HLE_ERROR;
            }
            SCE_OK
        }
        Err(err) => {
            warn!("sceKernelAllocateDirectMemory: mmap failed: {err:?}");
            HLE_ERROR
        }
    }
}

fn hle_allocate_main_direct_memory(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelAllocateMainDirectMemory(len={:#x}, alignment={:#x}, memoryType={})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    0
}

fn hle_release_direct_memory(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelReleaseDirectMemory(start={:#x}, len={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    0
}

fn hle_map_direct_memory(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelMapDirectMemory(len={:#x}, prot={}, alignment={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0),
        args.get(4).copied().unwrap_or(0)
    );
    0
}

/// Real signature: `sceKernelMapFlexibleMemory(void **addrOut, size_t len,
/// int prot, int flags)`.
///
/// Routes through `ctx.kernel.memory.mmap` for real: maps `len` bytes with
/// `prot` and writes the resulting guest address through `addrOut`
/// (`args[0]`) via `ctx.mem`. Returns `SCE_OK` on success; `HLE_ERROR` if
/// the mmap fails or `addrOut` is out of bounds.
fn hle_map_flexible_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr_out = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0x3) as u32;
    debug!("sceKernelMapFlexibleMemory(addrOut={addr_out:#x}, len={len:#x}, prot={prot})");

    match ctx.kernel.memory.mmap(0, len, prot, 0, -1, 0) {
        Ok(addr) => {
            if addr_out != 0 && !ctx.mem.write(addr_out, &addr.to_le_bytes()) {
                warn!("sceKernelMapFlexibleMemory: addrOut {addr_out:#x} out of bounds");
                return HLE_ERROR;
            }
            SCE_OK
        }
        Err(err) => {
            warn!("sceKernelMapFlexibleMemory: mmap failed: {err:?}");
            HLE_ERROR
        }
    }
}

fn hle_munmap(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelMunmap(addr={:#x}, len={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    0
}

/// Stub: returns a plausible fake mapped address rather than actually
/// mapping through `ctx.kernel.memory` — `sceKernelMmap`'s real `void
/// **res` out-parameter semantics (and fd-backed mappings) are a later
/// milestone; `sceKernelMapFlexibleMemory`/`sceKernelAllocateDirectMemory`
/// above are this milestone's proof that the context threads through.
fn hle_mmap(_ctx: &HleContext, args: &[u64]) -> u64 {
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
fn hle_get_direct_memory_size(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetDirectMemorySize()");
    0x4000_0000
}

/// Stub: plausible fixed size (256 MiB) of "available" flexible memory.
fn hle_available_flexible_memory_size(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelAvailableFlexibleMemorySize()");
    0x1000_0000
}

fn hle_set_virtual_range_name(_ctx: &HleContext, args: &[u64]) -> u64 {
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
fn hle_pthread_create(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("scePthreadCreate()");
    1
}

fn hle_pthread_join(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePthreadJoin(thread={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_exit(_ctx: &HleContext, _args: &[u64]) -> u64 {
    // Real scePthreadExit never returns to the caller; the stub returns 0
    // since this fn-pointer signature has no way to unwind the guest thread.
    debug!("scePthreadExit()");
    0
}

fn hle_pthread_mutex_init(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePthreadMutexInit(mutex={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_mutex_lock(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePthreadMutexLock(mutex={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_mutex_unlock(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePthreadMutexUnlock(mutex={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_cond_init(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePthreadCondInit(cond={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_cond_wait(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePthreadCondWait(cond={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_pthread_cond_signal(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePthreadCondSignal(cond={:#x})", args.first().copied().unwrap_or(0));
    0
}

/// Stub: returns a fake, always-`1` event-queue handle.
fn hle_create_equeue(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelCreateEqueue()");
    1
}

fn hle_wait_equeue(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("sceKernelWaitEqueue(eq={:#x})", args.first().copied().unwrap_or(0));
    0
}

// ---------------------------------------------------------------------
// Misc / process / clock
// ---------------------------------------------------------------------

/// Stub: `0` = `SCE_KERNEL_PROCESS_TYPE_NORMAL`-style plausible value.
fn hle_get_process_type(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetProcessType()");
    0
}

/// Stub: always reports CPU 0.
fn hle_get_current_cpu(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetCurrentCpu()");
    0
}

fn hle_gettimeofday(_ctx: &HleContext, _args: &[u64]) -> u64 {
    // Real function writes a `struct timeval` out-parameter; not yet wired
    // up here. Report success only.
    debug!("sceKernelGettimeofday()");
    0
}

fn hle_clock_gettime(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("sceKernelClockGettime(clockId={})", args.first().copied().unwrap_or(0));
    0
}

/// Stub: plausible PS5 base-clock TSC frequency (1.6 GHz).
fn hle_get_tsc_frequency(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetTscFrequency()");
    1_600_000_000
}

fn hle_usleep(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("sceKernelUsleep(usec={})", args.first().copied().unwrap_or(0));
    0
}

/// Stub: returns a fake, non-null pointer value. The real function returns
/// a pointer into the guest's process-parameter block, which this stub
/// cannot construct.
fn hle_get_proc_param(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetProcParam()");
    0x0000_1000_0000_0000
}

/// Stub: always reports base-mode (non-Neo/Pro) hardware.
fn hle_is_neo_mode(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelIsNeoMode()");
    0
}

/// Stub: always reports the base CPU mode.
fn hle_get_cpumode(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetCpumode()");
    0
}

/// Stub: no last-error state is tracked yet; always reports "no error".
fn hle_kernel_error(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelError()");
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn register_adds_expected_functions() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let ctx = test_ctx(&kernel, &mem);
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
            registry.call(&ctx, "libkernel", name, &[1, 2, 3, 4, 5, 6]);
        }
    }

    #[test]
    fn map_flexible_memory_actually_maps_through_the_kernel_and_writes_addr_out() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let ctx = test_ctx(&kernel, &mem);

        let addr_out: u64 = 0x100;
        let result = registry
            .call(&ctx, "libkernel", "sceKernelMapFlexibleMemory", &[addr_out, 0x4000, 0x3])
            .unwrap();
        assert_eq!(result, SCE_OK);

        let mut mapped_addr_bytes = [0u8; 8];
        assert!(mem.read(addr_out, &mut mapped_addr_bytes));
        let mapped_addr = u64::from_le_bytes(mapped_addr_bytes);
        assert_ne!(mapped_addr, 0, "sceKernelMapFlexibleMemory should write a real mapped address");
        assert!(kernel.memory.is_mapped(mapped_addr), "the address written to addrOut must actually be mapped");
    }

    #[test]
    fn allocate_direct_memory_actually_maps_through_the_kernel_and_writes_phys_addr_out() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let ctx = test_ctx(&kernel, &mem);

        let phys_addr_out: u64 = 0x200;
        let result = registry
            .call(
                &ctx,
                "libkernel",
                "sceKernelAllocateDirectMemory",
                &[0, 0, 0x4000, 0, 0, phys_addr_out],
            )
            .unwrap();
        assert_eq!(result, SCE_OK);

        let mut addr_bytes = [0u8; 8];
        assert!(mem.read(phys_addr_out, &mut addr_bytes));
        let addr = u64::from_le_bytes(addr_bytes);
        assert_ne!(addr, 0);
        assert!(kernel.memory.is_mapped(addr));
    }
}
