//! HLE libkernel — Core kernel interface re-implementation.
//!
//! Clean-room re-implementation of the PS5 `libkernel.sprx` exports. Function
//! *names* below are factual PS5 API identifiers (not copyrightable); every
//! implementation is original.
//!
//! ## Stub status
//!
//! Every HLE call now gets an [`crate::HleContext`] (a live
//! [`xps5x_kernel::OrbisKernel`] plus guest-memory and guest-allocator
//! access), so functions *can* do real work. Most functions below still
//! just log the call and return a plausible value (an `SCE_OK`-style `0`,
//! a fake handle, or a fake address/size) — thread creation, event queues,
//! and most out-parameters still aren't backed by real state.
//! `sceKernelAllocateDirectMemory`, `sceKernelMapFlexibleMemory`, and
//! `sceKernelMmap` are the exceptions: they route through `ctx.alloc.mmap`
//! (the arena's mmap region, in production — `xps5x-runtime`'s
//! `GuestArena`) and record the mapping in `ctx.kernel.memory` so
//! `is_mapped`/`region_containing` see it, writing the resulting address
//! through their out-parameter (where the ABI has one) via `ctx.mem`.
//! `sceKernelMunmap` mirrors this on the way out. Broadening the rest is
//! future work, not a limitation of the dispatch signature anymore.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK` — the PS5 convention for "this call succeeded".
const SCE_OK: u64 = 0;

/// Generic failure sentinel used by the functions below that now attempt a
/// real operation (`ctx.kernel.memory.mmap`/`ctx.mem` access) and can
/// genuinely fail. Not a real `SCE_KERNEL_ERROR_*` code — just a nonzero
/// value distinguishable from `SCE_OK`.
const HLE_ERROR: u64 = 0xFFFF_FFFF;

/// Cap on how many bytes one `write` call will copy out of guest memory —
/// keeps a wild `count` from ballooning a host buffer. Generous for
/// console output.
const WRITE_MAX_BYTES: u64 = 1 << 20; // 1 MiB

/// `EBADF` as the sign-extended negative return `write(2)` produces on a
/// bad descriptor (the PS5's BSD libc returns `-1` and sets `errno`;
/// `sceKernelWrite` returns a negative error directly — either way the
/// caller sees "negative", which is the honest signal here).
const WRITE_EBADF: u64 = (-9i64) as u64;

/// Real `write(fd, buf, count)` / `sceKernelWrite` for the console
/// descriptors (M1-C): fd 1 (stdout) and fd 2 (stderr) copy `count` guest
/// bytes (bounded by [`WRITE_MAX_BYTES`]) to the kernel
/// [`xps5x_kernel::Console`] and return `count`. Any other fd has no
/// backing file table yet — logged loudly, returns [`WRITE_EBADF`], never
/// pretends to have written.
fn hle_write(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0);
    let buf = args.get(1).copied().unwrap_or(0);
    let count = args.get(2).copied().unwrap_or(0);
    debug!("write(fd={fd}, buf={buf:#x}, count={count:#x})");

    if fd != 1 && fd != 2 {
        warn!(
            "write: fd {fd} has no backing file table yet (only stdout/stderr are wired) — EBADF"
        );
        return WRITE_EBADF;
    }
    let capped = count.min(WRITE_MAX_BYTES);
    let Ok(len) = usize::try_from(capped) else {
        return WRITE_EBADF;
    };
    let mut bytes = vec![0u8; len];
    if !ctx.mem.read(buf, &mut bytes) {
        warn!("write: guest buffer [{buf:#x}, +{capped:#x}) is not readable — EFAULT-ish EBADF");
        return WRITE_EBADF;
    }
    ctx.kernel.console.write_bytes(&bytes);
    if capped < count {
        warn!("write: count {count:#x} capped to {capped:#x} (WRITE_MAX_BYTES)");
    }
    capped
}

/// `SCE_KERNEL_ERROR_EFAULT` (`0x8002000E`): the documented SCE convention
/// of `0x8002_0000 | errno` (EFAULT = 14) — returned when a guest hands a
/// module call an unreadable pointer.
const SCE_KERNEL_ERROR_EFAULT: u64 = 0x8002_000E;
/// `SCE_KERNEL_ERROR_ESRCH` (`0x80020003`): no such module handle.
const SCE_KERNEL_ERROR_ESRCH: u64 = 0x8002_0003;
/// `SCE_KERNEL_ERROR_ENOENT` (`0x80020002`): symbol/entity not found.
const SCE_KERNEL_ERROR_ENOENT: u64 = 0x8002_0002;

/// Real-enough `sceKernelLoadStartModule(path, argc, argv, flags, opt,
/// pRes)` (M1-D, wall #4): reads the guest path, then
///
/// 1. If a module with that name (or filename) is already registered with
///    the kernel, returns its existing handle — repeated loads of the same
///    module hand back the same handle, like the real call.
/// 2. Otherwise registers a *synthetic* module entry and returns its fresh
///    handle. This is honest for system `.prx`s (`libSceLibcInternal`,
///    `libSceFios2`, ...) because their functionality is HLE: every import
///    from them was already NID-resolved against the HLE registry at link
///    time, so the only thing the guest actually needs from this call is a
///    valid handle (SharpEmu's loader takes the same pseudo-handle
///    approach). A *user-supplied* `.prx` with real code is NOT loaded by
///    this path yet — that needs file-backed loading through the firmware
///    pipeline, logged loudly below as future work.
///
/// `pRes` (the module-local `module_start` result out-param), when non-null,
/// is written `0` — no real `module_start` runs for an HLE-backed module.
fn hle_load_start_module(ctx: &HleContext, args: &[u64]) -> u64 {
    let path_ptr = args.first().copied().unwrap_or(0);
    let res_ptr = args.get(5).copied().unwrap_or(0);
    debug!("sceKernelLoadStartModule(path={path_ptr:#x}, pRes={res_ptr:#x})");

    let Some(path_bytes) = crate::fmt::read_cstr(ctx.mem, path_ptr) else {
        warn!("sceKernelLoadStartModule: unreadable path pointer {path_ptr:#x} — EFAULT");
        return SCE_KERNEL_ERROR_EFAULT;
    };
    let path = String::from_utf8_lossy(&path_bytes).into_owned();
    // The registry keys modules by bare name; callers pass full guest paths
    // like "/system/common/lib/libSceSysmodule.sprx".
    let file_name = path.rsplit('/').next().unwrap_or(&path).to_string();
    let stem = file_name
        .trim_end_matches(".sprx")
        .trim_end_matches(".prx")
        .to_string();

    if res_ptr != 0 && !ctx.mem.write(res_ptr, &0u32.to_le_bytes()) {
        warn!("sceKernelLoadStartModule: pRes {res_ptr:#x} is not writable — EFAULT");
        return SCE_KERNEL_ERROR_EFAULT;
    }

    for candidate in [path.as_str(), file_name.as_str(), stem.as_str()] {
        if let Some(info) = ctx.kernel.find_module(candidate) {
            debug!(
                "sceKernelLoadStartModule: '{path}' already loaded as handle {}",
                info.id
            );
            return info.id as u64;
        }
    }

    warn!(
        "sceKernelLoadStartModule: '{path}' registered as an HLE-backed pseudo-module — its \
         imports resolve via NID against the HLE registry; file-backed .prx loading is not \
         implemented"
    );
    let id = ctx.kernel.register_module(xps5x_kernel::ModuleInfo {
        id: 0, // assigned by register_module
        name: stem,
        base_address: 0,
        size: 0,
        entry_point: None,
        initialized: true,
    });
    id as u64
}

/// `sceKernelStopUnloadModule(handle, ...)`: validates the handle against
/// the kernel module table and reports success without unloading — every
/// module this HLE hands out is HLE-backed (nothing to unload), and the
/// main module is never legitimately unloaded mid-run. An unknown handle is
/// a loud `ESRCH`, not a silent success.
fn hle_stop_unload_module(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    debug!("sceKernelStopUnloadModule(handle={handle})");

    let Ok(id) = u32::try_from(handle) else {
        return SCE_KERNEL_ERROR_ESRCH;
    };
    if ctx.kernel.modules.contains_key(&id) {
        SCE_OK
    } else {
        warn!("sceKernelStopUnloadModule: unknown module handle {handle} — ESRCH");
        SCE_KERNEL_ERROR_ESRCH
    }
}

/// `sceKernelDlsym(handle, symbol, addrOut)`: honestly unimplemented. The
/// HLE trampoline table is minted at link time (LM1) from the module's
/// declared imports; handing out a *new* callable guest address at runtime
/// needs dynamically-minted trampolines the dispatcher doesn't support yet.
/// Logs the requested symbol loudly and returns `ENOENT` — a title that
/// needs dlsym will show exactly what it asked for.
fn hle_dlsym(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let sym_ptr = args.get(1).copied().unwrap_or(0);
    let symbol = crate::fmt::read_cstr(ctx.mem, sym_ptr)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_else(|| format!("<unreadable {sym_ptr:#x}>"));
    warn!(
        "sceKernelDlsym(handle={handle}, symbol='{symbol}'): runtime symbol lookup needs \
         dynamically-minted trampolines (not implemented) — ENOENT"
    );
    SCE_KERNEL_ERROR_ENOENT
}

/// Register libkernel HLE functions.
pub fn register(registry: &HleRegistry) {
    // -- Memory --
    registry.register(
        "libkernel",
        "sceKernelAllocateDirectMemory",
        hle_allocate_direct_memory,
    );
    registry.register(
        "libkernel",
        "sceKernelAllocateMainDirectMemory",
        hle_allocate_main_direct_memory,
    );
    registry.register(
        "libkernel",
        "sceKernelReleaseDirectMemory",
        hle_release_direct_memory,
    );
    registry.register(
        "libkernel",
        "sceKernelMapDirectMemory",
        hle_map_direct_memory,
    );
    registry.register(
        "libkernel",
        "sceKernelMapFlexibleMemory",
        hle_map_flexible_memory,
    );
    registry.register("libkernel", "sceKernelMunmap", hle_munmap);
    registry.register("libkernel", "sceKernelMmap", hle_mmap);

    // -- File descriptors / console I/O (M1-C) --
    registry.register("libkernel", "write", hle_write);
    registry.register("libkernel", "sceKernelWrite", hle_write);

    // -- Module loading (M1-D) --
    registry.register(
        "libkernel",
        "sceKernelLoadStartModule",
        hle_load_start_module,
    );
    registry.register(
        "libkernel",
        "sceKernelStopUnloadModule",
        hle_stop_unload_module,
    );
    registry.register("libkernel", "sceKernelDlsym", hle_dlsym);
    registry.register(
        "libkernel",
        "sceKernelGetDirectMemorySize",
        hle_get_direct_memory_size,
    );
    registry.register(
        "libkernel",
        "sceKernelAvailableFlexibleMemorySize",
        hle_available_flexible_memory_size,
    );
    registry.register(
        "libkernel",
        "sceKernelSetVirtualRangeName",
        hle_set_virtual_range_name,
    );

    // -- Thread / sync --
    registry.register("libkernel", "scePthreadCreate", hle_pthread_create);
    registry.register("libkernel", "scePthreadJoin", hle_pthread_join);
    registry.register("libkernel", "scePthreadExit", hle_pthread_exit);
    registry.register("libkernel", "scePthreadMutexInit", hle_pthread_mutex_init);
    registry.register("libkernel", "scePthreadMutexLock", hle_pthread_mutex_lock);
    registry.register(
        "libkernel",
        "scePthreadMutexUnlock",
        hle_pthread_mutex_unlock,
    );
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
    registry.register(
        "libkernel",
        "sceKernelGetTscFrequency",
        hle_get_tsc_frequency,
    );
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
/// this now allocates from the arena's mmap region (`ctx.alloc.mmap`) —
/// treating the "physical" address handed back as already virtual-mapped —
/// records the mapping's metadata in `ctx.kernel.memory` (so
/// `is_mapped`/`region_containing` see it), and writes the address through
/// the `physAddrOut` out-parameter (`args[5]`) via `ctx.mem`. Returns
/// `SCE_OK` on success; `HLE_ERROR` if the arena is exhausted or
/// `physAddrOut` is out of bounds (bounds-checked, never a panic/OOB) — in
/// the latter case the just-recorded metadata is rolled back via
/// `remove_mapping` so no dangling record is left behind.
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

    const DEFAULT_PROT: u32 = 0x3; // R+W — matches the old `ctx.kernel.memory.mmap(0, len, 0x3, ...)` call this replaces.

    let Some(addr) = ctx.alloc.mmap(len, xps5x_core::PS5_PAGE_SIZE as u64) else {
        warn!("sceKernelAllocateDirectMemory: arena mmap failed (len={len:#x})");
        return HLE_ERROR;
    };
    ctx.kernel.memory.record_mapping(addr, len, DEFAULT_PROT);

    if phys_addr_out != 0 && !ctx.mem.write(phys_addr_out, &addr.to_le_bytes()) {
        warn!("sceKernelAllocateDirectMemory: physAddrOut {phys_addr_out:#x} out of bounds");
        ctx.kernel.memory.remove_mapping(addr);
        return HLE_ERROR;
    }
    SCE_OK
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
/// Allocates `len` bytes from the arena's mmap region (`ctx.alloc.mmap`),
/// records the mapping's metadata in `ctx.kernel.memory` (so
/// `is_mapped`/`region_containing` reflect it), and writes the resulting
/// guest address through `addrOut` (`args[0]`) via `ctx.mem`. Returns
/// `SCE_OK` on success; `HLE_ERROR` if the arena is exhausted or `addrOut`
/// is out of bounds — in the latter case `remove_mapping` rolls back the
/// just-recorded metadata so no dangling record is left behind.
fn hle_map_flexible_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr_out = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0x3) as u32;
    debug!("sceKernelMapFlexibleMemory(addrOut={addr_out:#x}, len={len:#x}, prot={prot})");

    let Some(addr) = ctx.alloc.mmap(len, xps5x_core::PS5_PAGE_SIZE as u64) else {
        warn!("sceKernelMapFlexibleMemory: arena mmap failed (len={len:#x})");
        return HLE_ERROR;
    };
    ctx.kernel.memory.record_mapping(addr, len, prot);

    if addr_out != 0 && !ctx.mem.write(addr_out, &addr.to_le_bytes()) {
        warn!("sceKernelMapFlexibleMemory: addrOut {addr_out:#x} out of bounds");
        ctx.kernel.memory.remove_mapping(addr);
        return HLE_ERROR;
    }
    SCE_OK
}

/// Releases a mapping previously returned by `sceKernelMapFlexibleMemory`/
/// `sceKernelAllocateDirectMemory`/`sceKernelMmap`: releases the arena
/// allocation (`ctx.alloc.munmap`, best-effort — see
/// [`xps5x_hle::GuestAllocator::munmap`]'s contract) and removes the VMM
/// metadata (`ctx.kernel.memory.remove_mapping`) so `is_mapped` stops
/// reporting the address as mapped. Always reports success (`SCE_OK`),
/// matching real `munmap`'s behavior on an already-unmapped/unrecognized
/// address.
fn hle_munmap(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    debug!("sceKernelMunmap(addr={addr:#x}, len={len:#x})");

    ctx.alloc.munmap(addr, len);
    ctx.kernel.memory.remove_mapping(addr);
    SCE_OK
}

/// Real signature: `sceKernelMmap(void *addr, size_t len, int prot, int
/// flags, int fd, off_t offset, void **res)`. This HLE only models
/// anonymous mappings — `fd`/`offset` (file-backed mmap) are ignored, a
/// documented shortcut, same as the rest of this file's memory functions.
///
/// Allocates `len` bytes from the arena's mmap region (`ctx.alloc.mmap`)
/// and records the mapping's metadata in `ctx.kernel.memory`, same as
/// `sceKernelMapFlexibleMemory` above. Unlike that function, this HLE
/// binding returns the mapped address directly as the call's result rather
/// than through an out-parameter (the real ABI's `void **res` is not
/// modeled here — no `args` slot maps cleanly to it, and every existing
/// caller of this HLE function already expects the address in the return
/// value); on failure returns `0` rather than `HLE_ERROR`, since `0` is
/// `sceKernelMmap`'s real `NULL`-ish failure convention for an
/// address-returning call.
fn hle_mmap(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0x3) as u32;
    let flags = args.get(3).copied().unwrap_or(0);
    debug!("sceKernelMmap(addr={addr:#x}, len={len:#x}, prot={prot}, flags={flags:#x})");

    let Some(mapped) = ctx.alloc.mmap(len, xps5x_core::PS5_PAGE_SIZE as u64) else {
        warn!("sceKernelMmap: arena mmap failed (len={len:#x})");
        return 0;
    };
    ctx.kernel.memory.record_mapping(mapped, len, prot);
    mapped
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
    debug!(
        "scePthreadJoin(thread={:#x})",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_pthread_exit(_ctx: &HleContext, _args: &[u64]) -> u64 {
    // Real scePthreadExit never returns to the caller; the stub returns 0
    // since this fn-pointer signature has no way to unwind the guest thread.
    debug!("scePthreadExit()");
    0
}

fn hle_pthread_mutex_init(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "scePthreadMutexInit(mutex={:#x})",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_pthread_mutex_lock(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "scePthreadMutexLock(mutex={:#x})",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_pthread_mutex_unlock(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "scePthreadMutexUnlock(mutex={:#x})",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_pthread_cond_init(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "scePthreadCondInit(cond={:#x})",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_pthread_cond_wait(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "scePthreadCondWait(cond={:#x})",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_pthread_cond_signal(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "scePthreadCondSignal(cond={:#x})",
        args.first().copied().unwrap_or(0)
    );
    0
}

/// Stub: returns a fake, always-`1` event-queue handle.
fn hle_create_equeue(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelCreateEqueue()");
    1
}

fn hle_wait_equeue(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelWaitEqueue(eq={:#x})",
        args.first().copied().unwrap_or(0)
    );
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
    debug!(
        "sceKernelClockGettime(clockId={})",
        args.first().copied().unwrap_or(0)
    );
    0
}

/// Stub: plausible PS5 base-clock TSC frequency (1.6 GHz).
fn hle_get_tsc_frequency(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetTscFrequency()");
    1_600_000_000
}

fn hle_usleep(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelUsleep(usec={})",
        args.first().copied().unwrap_or(0)
    );
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

    /// M1-C: `write(1, buf, n)` copies real guest bytes to the kernel
    /// console and returns `n`; stderr (fd 2) lands in the same capture; an
    /// unbacked fd is a loud EBADF, never a silent "success".
    #[test]
    fn write_to_stdout_and_stderr_reaches_the_console_and_bad_fd_is_ebadf() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"out\n"));
        assert!(mem.write(0x200, b"err\n"));
        assert_eq!(hle_write(&ctx, &[1, 0x100, 4]), 4);
        assert_eq!(hle_write(&ctx, &[2, 0x200, 4]), 4);
        assert_eq!(kernel.console.contents(), "out\nerr\n");

        assert_eq!(
            hle_write(&ctx, &[7, 0x100, 4]) as i64,
            -9,
            "unbacked fd must be EBADF"
        );
        assert_eq!(
            kernel.console.contents(),
            "out\nerr\n",
            "bad-fd write must not emit"
        );
    }

    /// M1-D: LoadStartModule returns a positive handle, writes 0 through
    /// `pRes`, and hands the *same* handle back for a repeated load of the
    /// same module.
    #[test]
    fn load_start_module_returns_stable_positive_handle_and_writes_pres() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"/system/common/lib/libSceSysmodule.sprx\0"));
        assert!(mem.write(0x300, &0xFFFF_FFFFu32.to_le_bytes())); // pRes poisoned

        let handle = hle_load_start_module(&ctx, &[0x100, 0, 0, 0, 0, 0x300]);
        assert!(
            (handle as i64) > 0,
            "expected a positive handle, got {handle:#x}"
        );

        let mut pres = [0u8; 4];
        assert!(mem.read(0x300, &mut pres));
        assert_eq!(u32::from_le_bytes(pres), 0, "pRes must be written 0");

        let again = hle_load_start_module(&ctx, &[0x100, 0, 0, 0, 0, 0]);
        assert_eq!(
            again, handle,
            "same module path must return the same handle"
        );

        // The registered name is the extension-stripped filename.
        assert!(kernel.find_module("libSceSysmodule").is_some());
    }

    #[test]
    fn load_start_module_with_unreadable_path_is_efault() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_load_start_module(&ctx, &[0xDEAD_0000, 0, 0, 0, 0, 0]),
            SCE_KERNEL_ERROR_EFAULT
        );
    }

    #[test]
    fn stop_unload_validates_the_handle_and_dlsym_is_honest_enoent() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"libFoo.sprx\0"));
        let handle = hle_load_start_module(&ctx, &[0x100, 0, 0, 0, 0, 0]);
        assert_eq!(hle_stop_unload_module(&ctx, &[handle]), SCE_OK);
        assert_eq!(
            hle_stop_unload_module(&ctx, &[9999]),
            SCE_KERNEL_ERROR_ESRCH
        );

        assert!(mem.write(0x200, b"sceSomeFunction\0"));
        assert_eq!(
            hle_dlsym(&ctx, &[handle, 0x200, 0x400]),
            SCE_KERNEL_ERROR_ENOENT
        );
    }

    #[test]
    fn write_with_unreadable_buffer_fails_loudly() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!((hle_write(&ctx, &[1, 0xDEAD_0000, 8]) as i64) < 0);
        assert!(kernel.console.is_empty());
    }

    #[test]
    fn register_adds_expected_functions() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
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
            assert!(
                registry.is_implemented("libkernel", name),
                "missing libkernel::{name}"
            );
            // Every registered stub must be callable without panicking.
            registry.call(&ctx, "libkernel", name, &[1, 2, 3, 4, 5, 6]);
        }
    }

    #[test]
    fn map_flexible_memory_actually_maps_through_the_arena_and_writes_addr_out() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        // Nonzero base: `TestAllocator` is a bump allocator, so its returned
        // address is real evidence the call routed through `ctx.alloc`
        // (`kernel.memory.mmap` would instead pick from its own
        // `next_anon_addr`, which starts far above this test's TestMemory).
        let alloc = crate::TestAllocator::new(0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let addr_out: u64 = 0x100;
        let result = registry
            .call(
                &ctx,
                "libkernel",
                "sceKernelMapFlexibleMemory",
                &[addr_out, 0x4000, 0x3],
            )
            .unwrap();
        assert_eq!(result, SCE_OK);

        let mut mapped_addr_bytes = [0u8; 8];
        assert!(mem.read(addr_out, &mut mapped_addr_bytes));
        let mapped_addr = u64::from_le_bytes(mapped_addr_bytes);
        assert_ne!(
            mapped_addr, 0,
            "sceKernelMapFlexibleMemory should write a real mapped address"
        );
        assert!(
            kernel.memory.is_mapped(mapped_addr),
            "the address written to addrOut must be recorded as mapped in the VMM"
        );

        // munmap must clear the recorded metadata.
        let munmap_result = registry
            .call(&ctx, "libkernel", "sceKernelMunmap", &[mapped_addr, 0x4000])
            .unwrap();
        assert_eq!(munmap_result, SCE_OK);
        assert!(
            !kernel.memory.is_mapped(mapped_addr),
            "sceKernelMunmap must remove the VMM mapping record"
        );
    }

    #[test]
    fn allocate_direct_memory_actually_maps_through_the_arena_and_writes_phys_addr_out() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

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

    #[test]
    fn mmap_returns_a_real_mapped_address_directly_and_munmap_clears_it() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // sceKernelMmap's ABI returns the mapped address as the call result
        // (not through an out-parameter), unlike sceKernelMapFlexibleMemory.
        let addr = registry
            .call(&ctx, "libkernel", "sceKernelMmap", &[0, 0x4000, 0x3, 0])
            .unwrap();
        assert_ne!(
            addr, 0,
            "sceKernelMmap should return a real mapped address, not the old fake sentinel"
        );
        assert!(
            kernel.memory.is_mapped(addr),
            "sceKernelMmap must record its mapping in the VMM"
        );

        let munmap_result = registry
            .call(&ctx, "libkernel", "sceKernelMunmap", &[addr, 0x4000])
            .unwrap();
        assert_eq!(munmap_result, SCE_OK);
        assert!(
            !kernel.memory.is_mapped(addr),
            "sceKernelMunmap must remove the VMM mapping record"
        );
    }
}
