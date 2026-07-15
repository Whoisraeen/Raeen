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

use crate::{GuestCallCompletion, GuestCallRequest, HleContext, HleRegistry};
use std::sync::atomic::Ordering;
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

/// Per-module dynamic TLS storage used by `__tls_get_addr`. This matches the
/// compatibility-layer size used by SharpEmu and is deliberately bounded so
/// a corrupt guest descriptor cannot request an unbounded host allocation.
const DYNAMIC_TLS_BLOCK_SIZE: u64 = 0x1_0000;

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

    let capped = count.min(WRITE_MAX_BYTES);
    let Ok(len) = usize::try_from(capped) else {
        return WRITE_EBADF;
    };
    let mut bytes = vec![0u8; len];
    if !ctx.mem.read(buf, &mut bytes) {
        warn!("write: guest buffer [{buf:#x}, +{capped:#x}) is not readable — EFAULT-ish EBADF");
        return WRITE_EBADF;
    }
    if capped < count {
        warn!("write: count {count:#x} capped to {capped:#x} (WRITE_MAX_BYTES)");
    }

    // fd 1/2 are the console; everything else routes to a VFS-backed file
    // descriptor (real write-back on close — savedata/output files persist).
    if fd == 1 || fd == 2 {
        ctx.kernel.console.write_bytes(&bytes);
        return capped;
    }
    match ctx.kernel.filesystem.write(fd as i32, &bytes) {
        Ok(n) => n as u64,
        Err(e) => {
            warn!("write: fd {fd} failed: {e} — EBADF");
            WRITE_EBADF
        }
    }
}

/// `EBADF` (bad file descriptor) as a sign-extended negative return.
const FILE_EBADF: u64 = (-9i64) as u64;
/// `ENOENT` (no such file) as a sign-extended negative return.
const FILE_ENOENT: u64 = (-2i64) as u64;
/// `EFAULT` (bad address) as a sign-extended negative return.
const FILE_EFAULT: u64 = (-14i64) as u64;
/// `EINVAL` (invalid argument) as a sign-extended negative return.
const FILE_EINVAL: u64 = (-22i64) as u64;
/// Cap on a single `read` transfer into guest memory (bounds host staging).
const READ_MAX_BYTES: u64 = 16 << 20; // 16 MiB

/// Real `open(path, flags, mode)` / `sceKernelOpen` (VFS-backed): resolves
/// the guest path through the kernel VFS (`/app0/…` → the game directory,
/// etc.), opens it, and returns a file descriptor (>= 3). A path that
/// resolves to no existing host file is a genuine `ENOENT` — homebrew that
/// probes for optional files gets the real "not found" instead of a fake
/// success. Console fds (0/1/2) are handled by `write`, not here.
fn hle_open(ctx: &HleContext, args: &[u64]) -> u64 {
    let path_ptr = args.first().copied().unwrap_or(0);
    let flags = args.get(1).copied().unwrap_or(0) as i32;
    let mode = args.get(2).copied().unwrap_or(0) as u32;
    debug!("open(path={path_ptr:#x}, flags={flags:#x}, mode={mode:#o})");

    let Some(path_bytes) = crate::fmt::read_cstr(ctx.mem, path_ptr) else {
        warn!("open: unreadable path pointer {path_ptr:#x} — EFAULT");
        return FILE_EFAULT;
    };
    let path = String::from_utf8_lossy(&path_bytes).into_owned();

    // A missing file is ENOENT *unless* the guest passed O_CREAT (then the VFS
    // creates it). O_CREAT is bit 0x200 in the Orbis/BSD flag set.
    const O_CREAT: i32 = 0x200;
    let creating = flags & O_CREAT != 0;
    match ctx.kernel.filesystem.resolve_path(&path) {
        Some(host) if host.exists() || creating => {}
        Some(host) => {
            warn!(
                "open: '{path}' → '{}' does not exist (no O_CREAT) — ENOENT",
                host.display()
            );
            return FILE_ENOENT;
        }
        None => {
            warn!("open: '{path}' matches no VFS mount — ENOENT");
            return FILE_ENOENT;
        }
    }

    match ctx.kernel.filesystem.open(&path, flags, mode) {
        Ok(fd) => fd as u64,
        Err(e) => {
            warn!("open: '{path}' failed: {e} — ENOENT");
            FILE_ENOENT
        }
    }
}

/// Real `read(fd, buf, count)` / `sceKernelRead` (VFS-backed): reads up to
/// `count` bytes (capped by [`READ_MAX_BYTES`]) from the open descriptor and
/// writes them into the guest buffer, returning the byte count actually read
/// (0 at EOF). Bad fd → `EBADF`; unwritable buffer → `EFAULT`.
fn hle_read(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let buf = args.get(1).copied().unwrap_or(0);
    let count = args.get(2).copied().unwrap_or(0).min(READ_MAX_BYTES);
    debug!("read(fd={fd}, buf={buf:#x}, count={count:#x})");

    let Ok(n) = usize::try_from(count) else {
        return FILE_EINVAL;
    };
    match ctx.kernel.filesystem.read(fd, n) {
        Ok(bytes) => {
            if bytes.is_empty() {
                return 0; // EOF (or an empty file) — a valid short read.
            }
            if !ctx.mem.write(buf, &bytes) {
                warn!(
                    "read: guest buffer {buf:#x} (+{}) not writable — EFAULT",
                    bytes.len()
                );
                return FILE_EFAULT;
            }
            bytes.len() as u64
        }
        Err(e) => {
            warn!("read: fd {fd} failed: {e} — EBADF");
            FILE_EBADF
        }
    }
}

/// Real `close(fd)` / `sceKernelClose`: closes the VFS descriptor. Unknown
/// fd → `EBADF`.
fn hle_close(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    debug!("close(fd={fd})");
    match ctx.kernel.filesystem.close(fd) {
        Ok(()) => SCE_OK,
        Err(_) => FILE_EBADF,
    }
}

/// `sceKernelFsync(fd)`: persist the VFS write-back buffer while leaving the
/// descriptor open. The SCE spelling returns the kernel errno encoding.
fn hle_fsync(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    match ctx.kernel.filesystem.sync(fd) {
        Ok(()) => SCE_OK,
        Err(e) => {
            warn!("fsync: fd {fd} failed: {e} â€” EBADF");
            0x8002_0009
        }
    }
}

/// Common Gen5 directory enumeration path. `sceKernelGetdirentries` supplies
/// an optional fourth `basep` argument; `sceKernelGetdents` does not.
fn hle_getdents(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let buffer = args.get(1).copied().unwrap_or(0);
    let requested = args.get(2).copied().unwrap_or(0).min(READ_MAX_BYTES);
    let basep = args.get(3).copied().unwrap_or(0);
    if buffer == 0 || requested < 512 {
        return 0x8002_0016;
    }
    let Ok(requested) = usize::try_from(requested) else {
        return 0x8002_0016;
    };
    let (payload, base) = match ctx.kernel.filesystem.getdents(fd, requested) {
        Ok(result) => result,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return 0x8002_0009,
        Err(_) => return 0x8002_0016,
    };
    if !payload.is_empty() && !ctx.mem.write(buffer, &payload) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if basep != 0 && !ctx.mem.write(basep, &(base as u64).to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    payload.len() as u64
}

/// Real `lseek(fd, offset, whence)` / `sceKernelLseek`: repositions the
/// descriptor and returns the new absolute offset. Bad fd → `EBADF`; bad
/// `whence` or a negative resulting position → `EINVAL`.
fn hle_lseek(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let offset = args.get(1).copied().unwrap_or(0) as i64;
    let whence = args.get(2).copied().unwrap_or(0) as i32;
    debug!("lseek(fd={fd}, offset={offset}, whence={whence})");
    match ctx.kernel.filesystem.seek(fd, offset, whence) {
        Ok(pos) => pos,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => FILE_EBADF,
        Err(_) => FILE_EINVAL,
    }
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
    // Optional write-throttling service exported under its own library. The
    // public name is not known; this exact NID is measured from the title and
    // the service only supplies kernel write pacing policy, so accepting the
    // default policy is a process-wide, title-independent fallback.
    registry.register_nid(
        "libkernel_write_throttling",
        "writeThrottlingUnknownYFC3dBBipj8",
        0x6050_B774_1062_A63F,
        hle_write_throttling_default,
    );
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

    // -- File I/O (real, VFS-backed) --
    registry.register("libkernel", "open", hle_open);
    registry.register("libkernel", "sceKernelOpen", hle_open);
    registry.register("libkernel", "read", hle_read);
    registry.register("libkernel", "sceKernelRead", hle_read);
    registry.register("libkernel", "close", hle_close);
    registry.register("libkernel", "sceKernelClose", hle_close);
    registry.register_nid(
        "libkernel",
        "sceKernelFsync",
        0x7d3c_7aea_5e62_5880,
        hle_fsync,
    );
    registry.register("libkernel", "sceKernelGetdents", hle_getdents);
    registry.register("libkernel", "sceKernelGetdirentries", hle_getdents);
    registry.register("libkernel", "getdirentries", hle_getdents);
    registry.register("libScePosix", "getdents", hle_getdents);
    registry.register("libkernel", "lseek", hle_lseek);
    registry.register("libkernel", "sceKernelLseek", hle_lseek);

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
    // POSIX spellings have distinct NIDs but the same ABI and scheduler path.
    registry.register("libScePosix", "pthread_create", hle_pthread_create);
    registry.register("libScePosix", "pthread_join", hle_pthread_join);
    registry.register("libScePosix", "pthread_detach", hle_pthread_detach);
    registry.register("libScePosix", "pthread_exit", hle_pthread_exit);
    // rtld thread-teardown hooks libc.prx registers during module_start.
    registry.register(
        "libkernel",
        "_sceKernelSetThreadDtors",
        hle_set_thread_dtors,
    );
    registry.register(
        "libkernel",
        "_sceKernelSetThreadAtexitCount",
        hle_set_thread_atexit_count,
    );
    registry.register(
        "libkernel",
        "_sceKernelSetThreadAtexitReport",
        hle_set_thread_atexit_report,
    );
    registry.register(
        "libkernel",
        "_sceKernelRtldThreadAtexitIncrement",
        hle_rtld_thread_atexit_increment,
    );
    registry.register(
        "libkernel",
        "_sceKernelRtldThreadAtexitDecrement",
        hle_rtld_thread_atexit_decrement,
    );
    registry.register("libkernel", "__tls_get_addr", hle_tls_get_addr);
    // scePthreadMutex* are registered by the `pthread_sync` module (real state
    // machine) — see xps5x_hle::pthread_sync.
    registry.register("libkernel", "scePthreadCondInit", hle_pthread_cond_init);
    registry.register("libkernel", "scePthreadCondWait", hle_pthread_cond_wait);
    registry.register("libkernel", "scePthreadCondSignal", hle_pthread_cond_signal);
    registry.register(
        "libkernel",
        "scePthreadCondBroadcast",
        hle_pthread_cond_signal,
    );
    registry.register("libkernel", "scePthreadCondDestroy", hle_pthread_cond_init);
    // pthread_cond_wait/timedwait and pthread_create/join are deliberately
    // NOT aliased under libScePosix, and scePthreadCondTimedwait is not
    // registered: with one guest thread every possible return value lies,
    // and an unresolved import at least names itself. See `pthread_cond`'s
    // module docs and `pthread_sync::register_posix` (M1-E).
    registry.register("libScePosix", "pthread_setschedparam", hle_ok_stub);
    registry.register("libScePosix", "fstat", hle_fstat);

    // -- Measured Minecraft libc.prx / eboot imports (real PS5 export names,
    // each verified by NID hash against the title's import table; semantics
    // cross-checked with SharpEmu + Kyty). The `_`-prefixed file/exit names
    // are libkernel's real exports of the plain POSIX calls.
    registry.register("libkernel", "_open", hle_open);
    registry.register("libkernel", "_read", hle_read);
    registry.register("libkernel", "_write", hle_write);
    registry.register("libkernel", "_close", hle_close);
    // `_exit` terminates the process: the runtime's exit family intercepts it
    // before dispatch (see xps5x_runtime::dispatch::TERMINATING_FUNCTIONS);
    // this registration exists so the import resolves to a trampoline.
    registry.register("libkernel", "_exit", hle_pthread_exit);
    registry.register("libkernel", "nanosleep", hle_nanosleep);
    registry.register("libkernel", "_sigprocmask", hle_sigprocmask);
    registry.register("libkernel", "_is_signal_return", hle_is_signal_return);
    registry.register(
        "libkernel",
        "_sceKernelRtldSetApplicationHeapAPI",
        hle_rtld_set_application_heap_api,
    );
    registry.register(
        "libkernel",
        "sceKernelIsAddressSanitizerEnabled",
        hle_zero_stub,
    );
    registry.register(
        "libkernel",
        "sceKernelGetSanitizerMallocReplaceExternal",
        hle_zero_stub,
    );
    registry.register(
        "libkernel",
        "sceKernelGetSanitizerNewReplaceExternal",
        hle_zero_stub,
    );
    registry.register("libkernel", "__error", hle_error_addr);
    registry.register("libkernel", "__pthread_cxa_finalize", hle_ok_stub);
    registry.register("libkernel", "__elf_phdr_match_addr", hle_zero_stub);
    registry.register("libkernel", "sceKernelMprotect", hle_ok_stub);
    registry.register(
        "libkernel",
        "sceKernelCheckReachability",
        hle_check_reachability,
    );
    registry.register("libkernel", "sceKernelUuidCreate", hle_uuid_create);
    registry.register(
        "libkernel",
        "sceKernelConvertUtcToLocaltime",
        hle_convert_time_identity,
    );
    registry.register(
        "libkernel",
        "sceKernelConvertLocaltimeToUtc",
        hle_convert_time_identity,
    );
    registry.register(
        "libkernel",
        "sceKernelGetModuleInfoForUnwind",
        hle_module_info_unavailable,
    );
    registry.register(
        "libkernel",
        "sceKernelGetModuleInfoFromAddr",
        hle_module_info_unavailable,
    );
    registry.register("libkernel", "sceKernelVirtualQuery", hle_virtual_query_stub);
    registry.register(
        "libkernel",
        "sceKernelReserveVirtualRange",
        hle_reserve_virtual_range,
    );
    // The Named variants share the plain calls' leading arguments (the name
    // pointer trails) so they route to the same handlers.
    registry.register(
        "libkernel",
        "sceKernelMapNamedFlexibleMemoryInternal",
        hle_map_flexible_memory,
    );
    registry.register(
        "libkernel",
        "sceKernelMapNamedDirectMemory",
        hle_map_direct_memory,
    );
    registry.register(
        "libkernel",
        "sceKernelDebugRaiseException",
        hle_debug_raise_exception,
    );
    registry.register(
        "libkernel",
        "sceKernelDebugRaiseExceptionOnReleaseMode",
        hle_debug_raise_exception,
    );
    // Filesystem metadata. Path-based operations resolve through the same VFS
    // mounts as open/read/write; no title-specific path handling lives here.
    registry.register("libkernel", "sceKernelMkdir", hle_mkdir);
    registry.register("libkernel", "sceKernelUnlink", hle_fs_enoent);
    registry.register("libkernel", "sceKernelRmdir", hle_fs_enoent);
    registry.register("libkernel", "sceKernelStat", hle_stat);
    registry.register("libkernel", "sceKernelFstat", hle_fstat);
    registry.register("libkernel", "sceKernelGetdents", hle_fs_enoent);
    // pthread surface libc/fmod touch during init — attr/priority/affinity
    // bookkeeping has no scheduler to talk to yet, so recording nothing and
    // returning success is faithful enough for a single-thread world.
    registry.register("libkernel", "scePthreadDetach", hle_pthread_detach);
    registry.register("libkernel", "scePthreadSetprio", hle_ok_stub);
    registry.register("libkernel", "scePthreadSetaffinity", hle_ok_stub);
    registry.register("libkernel", "scePthreadAttrSetaffinity", hle_ok_stub);
    registry.register("libkernel", "scePthreadAttrGetaffinity", hle_ok_stub);
    registry.register("libkernel", "scePthreadAttrGet", hle_ok_stub);
    registry.register("libkernel", "scePthreadAttrSetschedparam", hle_ok_stub);
    registry.register("libkernel", "scePthreadAttrGetschedparam", hle_ok_stub);
    registry.register("libkernel", "scePthreadAttrSetinheritsched", hle_ok_stub);
    registry.register("libkernel", "scePthreadGetname", hle_pthread_getname);
    registry.register("libkernel", "scePthreadOnce", hle_pthread_once);
    registry.register("libScePosix", "pthread_once", hle_pthread_once);
    // sceKernelCreateEqueue/WaitEqueue are registered by the `kernel_equeue`
    // module (real user-event queue) — see xps5x_hle::kernel_equeue.

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
    registry.register("libkernel", "sceKernelGetProcessTime", hle_get_process_time);
    registry.register(
        "libkernel",
        "sceKernelGetProcessTimeCounter",
        hle_get_process_time_counter,
    );
    registry.register(
        "libkernel",
        "sceKernelGetProcessTimeCounterFrequency",
        hle_get_process_time_counter_frequency,
    );
    registry.register("libkernel", "sceKernelGetProcParam", hle_get_proc_param);
    registry.register("libkernel", "sceKernelIsNeoMode", hle_is_neo_mode);
    registry.register("libkernel", "sceKernelGetCpumode", hle_get_cpumode);
    registry.register("libkernel", "sceKernelError", hle_kernel_error);
    registry.register(
        "libkernel",
        "sceKernelGetCompiledSdkVersion",
        hle_get_compiled_sdk_version,
    );
    registry.register("libkernel", "getpid", hle_getpid);
    registry.register("libkernel", "sceKernelGetProcessId", hle_getpid);
}

/// The PS5 (Gen5) compiled-SDK version XPS5X reports: `0x09000000` == SDK
/// 9.00 (same value SharpEmu's `Gen5CompiledSdkVersion` reports). Homebrew
/// commonly gates feature use on this.
const GEN5_SDK_VERSION: u32 = 0x0900_0000;

/// `SCE_KERNEL_ERROR_EINVAL` (`0x80020016`): invalid argument (EINVAL = 22).
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;

/// A fixed, plausible process id XPS5X reports for the single guest process.
const GUEST_PID: u64 = 0x2A2A; // arbitrary stable nonzero pid

/// Real `sceKernelGetCompiledSdkVersion(unsigned int *version)` (M1
/// hardening, reference SharpEmu KernelExports): writes the PS5 SDK version
/// through the out-param and returns `SCE_OK`. A NULL pointer is `EINVAL`
/// (matching SharpEmu), and an unwritable one is `EFAULT`.
fn hle_get_compiled_sdk_version(ctx: &HleContext, args: &[u64]) -> u64 {
    let version_ptr = args.first().copied().unwrap_or(0);
    debug!("sceKernelGetCompiledSdkVersion(version={version_ptr:#x})");
    if version_ptr == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    if !ctx.mem.write(version_ptr, &GEN5_SDK_VERSION.to_le_bytes()) {
        warn!(
            "sceKernelGetCompiledSdkVersion: version out-pointer {version_ptr:#x} not writable — EFAULT"
        );
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// `getpid()` / `sceKernelGetProcessId()`: the single guest process's pid.
/// A real, stable nonzero value (some homebrew keys temp paths / logs on it).
pub(crate) fn hle_getpid(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("getpid() -> {GUEST_PID:#x}");
    GUEST_PID
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

    if len == 0 || phys_addr_out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let page_size = xps5x_core::PS5_PAGE_SIZE as u64;
    let alignment = if alignment == 0 {
        page_size
    } else if alignment.is_power_of_two() {
        alignment.max(page_size)
    } else {
        return SCE_KERNEL_ERROR_EINVAL;
    };
    let Some(addr) = ctx.alloc.mmap(len, alignment) else {
        warn!("sceKernelAllocateDirectMemory: arena mmap failed (len={len:#x})");
        return HLE_ERROR;
    };
    ctx.kernel.memory.record_mapping(addr, len, DEFAULT_PROT);

    if !ctx.mem.write(phys_addr_out, &addr.to_le_bytes()) {
        warn!("sceKernelAllocateDirectMemory: physAddrOut {phys_addr_out:#x} out of bounds");
        ctx.kernel.memory.remove_mapping(addr);
        return HLE_ERROR;
    }
    SCE_OK
}

fn hle_allocate_main_direct_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelAllocateMainDirectMemory(len={:#x}, alignment={:#x}, memoryType={})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    hle_allocate_direct_memory(
        ctx,
        &[
            0,
            u64::MAX,
            args.first().copied().unwrap_or(0),
            args.get(1).copied().unwrap_or(0),
            args.get(2).copied().unwrap_or(0),
            args.get(3).copied().unwrap_or(0),
        ],
    )
}

fn hle_release_direct_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelReleaseDirectMemory(start={:#x}, len={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    let start = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    if start == 0 || len == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    ctx.alloc.munmap(start, len);
    ctx.kernel.memory.remove_mapping(start);
    SCE_OK
}

fn hle_map_direct_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr_out = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0) as u32;
    let direct_memory_start = args.get(4).copied().unwrap_or(0);
    let alignment = args.get(5).copied().unwrap_or(0);
    debug!(
        "sceKernelMapDirectMemory(addrOut={addr_out:#x}, len={len:#x}, prot={prot}, phys={direct_memory_start:#x}, alignment={alignment:#x})"
    );
    if addr_out == 0 || len == 0 || direct_memory_start == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let Some(region) = ctx.kernel.memory.region_containing(direct_memory_start) else {
        return SCE_KERNEL_ERROR_EINVAL;
    };
    let Some(request_end) = direct_memory_start.checked_add(len) else {
        return SCE_KERNEL_ERROR_EINVAL;
    };
    let Some(region_end) = region.vaddr.checked_add(region.size) else {
        return SCE_KERNEL_ERROR_EINVAL;
    };
    if request_end > region_end {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    ctx.kernel
        .memory
        .record_mapping(direct_memory_start, len, prot);
    if !ctx.mem.write(addr_out, &direct_memory_start.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
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

/// Plausible amount of "available" flexible memory reported to the guest
/// (256 MiB). A real PS5 exposes a few hundred MiB of flexible memory;
/// this fixed figure is enough for homebrew that sanity-checks headroom
/// before mapping.
const FLEXIBLE_MEMORY_SIZE: u64 = 0x1000_0000;

/// Real `sceKernelAvailableFlexibleMemorySize(size_t *sizeOut)`: writes the
/// available size through the out-param and returns `SCE_OK` — the previous
/// stub returned the size *as the return value*, which is the wrong ABI (a
/// guest reading `*sizeOut` got garbage). NULL/unwritable out-param is
/// `EFAULT`.
fn hle_available_flexible_memory_size(ctx: &HleContext, args: &[u64]) -> u64 {
    let size_out = args.first().copied().unwrap_or(0);
    debug!("sceKernelAvailableFlexibleMemorySize(sizeOut={size_out:#x})");
    if size_out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    if !ctx.mem.write(size_out, &FLEXIBLE_MEMORY_SIZE.to_le_bytes()) {
        warn!("sceKernelAvailableFlexibleMemorySize: sizeOut {size_out:#x} not writable — EFAULT");
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

fn hle_set_virtual_range_name(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelSetVirtualRangeName(addr={:#x}, len={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    0
}

fn hle_write_throttling_default(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "libkernel_write_throttling::YFC3dBBipj8(context={:#x}, policy={:#x}) -> default policy",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    SCE_OK
}

// ---------------------------------------------------------------------
// Thread / sync
// ---------------------------------------------------------------------

/// Stub: returns a fake, always-`1` thread handle. No thread is actually
/// spawned — real thread creation needs [`xps5x_kernel::threading`] wiring.
fn hle_pthread_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let thread_out = args.first().copied().unwrap_or(0);
    let attr = args.get(1).copied().unwrap_or(0);
    let entry = args.get(2).copied().unwrap_or(0);
    let arg = args.get(3).copied().unwrap_or(0);
    debug!("scePthreadCreate(out={thread_out:#x}, attr={attr:#x}, entry={entry:#x}, arg={arg:#x})");
    ctx.guest_threads.create(thread_out, attr, entry, arg)
}

fn hle_pthread_join(ctx: &HleContext, args: &[u64]) -> u64 {
    let thread = args.first().copied().unwrap_or(0);
    let retval_out = args.get(1).copied().unwrap_or(0);
    debug!("scePthreadJoin(thread={thread:#x}, retval={retval_out:#x})");
    ctx.guest_threads.join(thread, retval_out)
}

fn hle_pthread_exit(ctx: &HleContext, args: &[u64]) -> u64 {
    // Real scePthreadExit never returns to the caller; the stub returns 0
    // since this fn-pointer signature has no way to unwind the guest thread.
    let retval = args.first().copied().unwrap_or(0);
    debug!("scePthreadExit(retval={retval:#x})");
    if ctx.guest_threads.request_exit(retval) {
        SCE_OK
    } else {
        SCE_KERNEL_ERROR_EINVAL
    }
}

fn hle_pthread_detach(ctx: &HleContext, args: &[u64]) -> u64 {
    let thread = args.first().copied().unwrap_or(0);
    debug!("scePthreadDetach(thread={thread:#x})");
    ctx.guest_threads.detach(thread)
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

// ---------------------------------------------------------------------
// rtld thread-teardown hooks
// ---------------------------------------------------------------------
//
// libc.prx's `module_start` hands the rtld three guest callbacks for
// per-thread teardown before doing anything else, so an unresolved
// `_sceKernelSetThreadDtors` stops a real title inside libc init. The
// callbacks are recorded (a future real thread-exit path must invoke
// them) but nothing calls them yet — the main thread only ever exits the
// whole process. Process-global statics are per-guest-process here: the
// runtime's RT0 single-active-execution invariant allows one guest
// process per host process (see `xps5x_runtime::dispatch::CALL_LOCK`).
// Semantics cross-checked against SharpEmu's `KernelMemoryCompatExports`
// (GPL-2.0): Set* stores the pointer and returns OK; Increment/Decrement
// adjusts a guest-memory `u64` counter in place (saturating at zero) and
// returns the adjusted value.

/// `_sceKernelSetThreadDtors(dtors_fn)`.
fn hle_set_thread_dtors(ctx: &HleContext, args: &[u64]) -> u64 {
    let callback = args.first().copied().unwrap_or(0);
    debug!("_sceKernelSetThreadDtors(fn={callback:#x})");
    ctx.kernel
        .thread_dtors_callback
        .store(callback, Ordering::Relaxed);
    SCE_OK
}

/// `_sceKernelSetThreadAtexitCount(count_fn)`.
fn hle_set_thread_atexit_count(ctx: &HleContext, args: &[u64]) -> u64 {
    let callback = args.first().copied().unwrap_or(0);
    debug!("_sceKernelSetThreadAtexitCount(fn={callback:#x})");
    ctx.kernel
        .thread_atexit_count_callback
        .store(callback, Ordering::Relaxed);
    SCE_OK
}

/// `_sceKernelSetThreadAtexitReport(report_fn)`.
fn hle_set_thread_atexit_report(ctx: &HleContext, args: &[u64]) -> u64 {
    let callback = args.first().copied().unwrap_or(0);
    debug!("_sceKernelSetThreadAtexitReport(fn={callback:#x})");
    ctx.kernel
        .thread_atexit_report_callback
        .store(callback, Ordering::Relaxed);
    SCE_OK
}

/// Shared body of `_sceKernelRtldThreadAtexitIncrement`/`Decrement`: adjust
/// the guest `u64` at `counter_ptr` by `delta` (clamping below at zero) and
/// return the adjusted value.
fn rtld_thread_atexit_adjust(ctx: &HleContext, args: &[u64], delta: i64) -> u64 {
    let counter_ptr = args.first().copied().unwrap_or(0);
    if counter_ptr == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let mut bytes = [0u8; 8];
    if !ctx.mem.read(counter_ptr, &mut bytes) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let value = u64::from_le_bytes(bytes);
    let adjusted = if delta >= 0 {
        value.saturating_add(delta as u64)
    } else {
        value.saturating_sub(delta.unsigned_abs())
    };
    if !ctx.mem.write(counter_ptr, &adjusted.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    debug!("_sceKernelRtldThreadAtexit adjust {delta:+}: {value} -> {adjusted}");
    adjusted
}

fn hle_rtld_thread_atexit_increment(ctx: &HleContext, args: &[u64]) -> u64 {
    rtld_thread_atexit_adjust(ctx, args, 1)
}

fn hle_rtld_thread_atexit_decrement(ctx: &HleContext, args: &[u64]) -> u64 {
    rtld_thread_atexit_adjust(ctx, args, -1)
}

/// `__tls_get_addr(const tls_index*)` resolves a guest descriptor containing
/// `{ module_id: u64, offset: u64 }` to stable, zero-initialized storage.
///
/// This is the dynamic-TLS path used by file-backed modules such as libc.prx;
/// the main executable's compiler-emitted initial-exec TLS continues to use
/// the runtime's variant-II static block and `TPOFF64` relocations. The block
/// lives in the guest arena (never host-only memory), so the returned pointer
/// is directly dereferenceable by native guest code.
fn hle_tls_get_addr(ctx: &HleContext, args: &[u64]) -> u64 {
    let descriptor = args.first().copied().unwrap_or(0);
    if descriptor == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }

    let mut bytes = [0u8; 16];
    if !ctx.mem.read(descriptor, &mut bytes) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let module_id = u64::from_le_bytes(bytes[..8].try_into().expect("fixed slice"));
    let offset = u64::from_le_bytes(bytes[8..].try_into().expect("fixed slice"));
    if offset >= DYNAMIC_TLS_BLOCK_SIZE {
        warn!(
            "__tls_get_addr(module={module_id:#x}, offset={offset:#x}): offset exceeds bounded block"
        );
        return SCE_KERNEL_ERROR_EINVAL;
    }

    let thread = ctx.guest_threads.current_thread();
    let key = (thread, module_id);
    let base = if let Some(existing) = ctx.kernel.dynamic_tls_blocks.get(&key) {
        *existing
    } else {
        let Some(base) = ctx.alloc.alloc(DYNAMIC_TLS_BLOCK_SIZE, 16) else {
            warn!("__tls_get_addr(module={module_id:#x}): guest TLS allocation failed");
            return HLE_ERROR;
        };
        let zeroes = vec![0u8; DYNAMIC_TLS_BLOCK_SIZE as usize];
        if !ctx.mem.write(base, &zeroes) {
            ctx.alloc.free(base);
            return SCE_KERNEL_ERROR_EFAULT;
        }
        match ctx.kernel.dynamic_tls_blocks.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(existing) => {
                ctx.alloc.free(base);
                *existing.get()
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                vacant.insert(base);
                debug!(
                    "__tls_get_addr: thread {thread:#x}, module {module_id:#x} -> block {base:#x}"
                );
                base
            }
        }
    };

    base + offset
}

// ---------------------------------------------------------------------
// libc.prx boot surface (measured Minecraft imports)
// ---------------------------------------------------------------------

/// The malloc/free/posix_memalign table libc hands the rtld via
/// `_sceKernelRtldSetApplicationHeapAPI(void *api[])`. Recorded so a future
/// heap-replacement path can consult it; nothing reads it back yet (guest
/// malloc is HLE'd directly).
/// Success for calls whose side effect has nothing to act on yet (signal
/// masks, memory protections, pthread attr bookkeeping in a single-thread
/// world). Logged at debug so a misbehaving title can still be traced.
fn hle_ok_stub(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `0` where the ABI reads the return as a pointer/bool ("no replacement
/// table", "sanitizer disabled", "no matching phdr").
fn hle_zero_stub(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0
}

/// `_sceKernelRtldSetApplicationHeapAPI(api[])`: record libc's heap table.
fn hle_rtld_set_application_heap_api(ctx: &HleContext, args: &[u64]) -> u64 {
    let api = args.first().copied().unwrap_or(0);
    debug!("_sceKernelRtldSetApplicationHeapAPI(api={api:#x})");
    ctx.kernel
        .application_heap_api
        .store(api, Ordering::Relaxed);
    SCE_OK
}

/// `__error()`: address of the calling thread's lazily allocated `errno`.
fn hle_error_addr(ctx: &HleContext, _args: &[u64]) -> u64 {
    let thread = ctx.guest_threads.current_thread();
    if let Some(existing) = ctx.kernel.errno_slots.get(&thread) {
        return *existing;
    }
    let Some(slot) = ctx.alloc.alloc(8, 8) else {
        warn!("__error: guest arena exhausted; returning NULL errno address");
        return 0;
    };
    let _ = ctx.mem.write(slot, &0u64.to_le_bytes());
    match ctx.kernel.errno_slots.entry(thread) {
        dashmap::mapref::entry::Entry::Occupied(existing) => {
            ctx.alloc.free(slot);
            *existing.get()
        }
        dashmap::mapref::entry::Entry::Vacant(vacant) => {
            vacant.insert(slot);
            slot
        }
    }
}

/// `nanosleep(req, rem)`: honor the requested sleep (bounded like
/// `sceKernelUsleep`), report zero time remaining.
fn hle_nanosleep(ctx: &HleContext, args: &[u64]) -> u64 {
    let req = args.first().copied().unwrap_or(0);
    let rem = args.get(1).copied().unwrap_or(0);
    let mut secs = [0u8; 8];
    let mut nanos = [0u8; 8];
    if req == 0 || !ctx.mem.read(req, &mut secs) || !ctx.mem.read(req + 8, &mut nanos) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let secs = u64::from_le_bytes(secs);
    let nanos = u64::from_le_bytes(nanos);
    // Same guard as sceKernelUsleep: never let a wild guest value hang the
    // host for minutes.
    const MAX_SLEEP_MS: u64 = 100;
    let ms = secs
        .saturating_mul(1000)
        .saturating_add(nanos / 1_000_000)
        .min(MAX_SLEEP_MS);
    debug!("nanosleep({secs}s + {nanos}ns) -> sleeping {ms}ms");
    std::thread::sleep(std::time::Duration::from_millis(ms));
    if rem != 0 {
        let _ = ctx.mem.write(rem, &[0u8; 16]);
    }
    SCE_OK
}

/// `_sigprocmask(how, set, oset)`: no signals are ever delivered to the
/// guest, so the mask is trivially empty — write an all-zero old mask.
fn hle_sigprocmask(ctx: &HleContext, args: &[u64]) -> u64 {
    let oset = args.get(2).copied().unwrap_or(0);
    if oset != 0 {
        // sigset_t on Orbis is 16 bytes.
        let _ = ctx.mem.write(oset, &[0u8; 16]);
    }
    SCE_OK
}

/// Whether an address is the kernel's signal-return trampoline. XPS5X does
/// not deliver guest signals, so it never installs such a trampoline.
fn hle_is_signal_return(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "_is_signal_return(pc={:#x}) -> false (guest signals disabled)",
        args.first().copied().unwrap_or(0)
    );
    0
}

/// `sceKernelUuidCreate(SceKernelUuid *out)`: 16 bytes of per-call entropy
/// (RandomState-seeded, like the runtime's stack canary — no `rand` dep).
fn hle_uuid_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    if out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    use std::hash::{BuildHasher, Hasher};
    let mut bytes = [0u8; 16];
    for half in 0..2 {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u64(half as u64);
        bytes[half * 8..][..8].copy_from_slice(&hasher.finish().to_le_bytes());
    }
    if !ctx.mem.write(out, &bytes) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// `sceKernelConvertUtcToLocaltime` / `ConvertLocaltimeToUtc`: XPS5X's guest
/// clock runs in UTC, so the conversion is the identity — write the input
/// `time_t` back through the output pointer.
fn hle_convert_time_identity(ctx: &HleContext, args: &[u64]) -> u64 {
    let time = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    if out != 0 && !ctx.mem.write(out, &time.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// `sceKernelGetModuleInfoForUnwind` / `GetModuleInfoFromAddr`: no module
/// metadata table is exposed to the guest yet. An honest ESRCH beats a
/// half-filled info struct the unwinder would chase into the weeds.
fn hle_module_info_unavailable(_ctx: &HleContext, args: &[u64]) -> u64 {
    warn!(
        "sceKernelGetModuleInfo*(addr={:#x}): module info not implemented — returning ESRCH",
        args.first().copied().unwrap_or(0)
    );
    SCE_KERNEL_ERROR_ESRCH
}

/// `sceKernelVirtualQuery(addr, flags, info, infoSize)`: no query surface
/// yet; an honest EFAULT tells the caller nothing was written.
fn hle_virtual_query_stub(_ctx: &HleContext, args: &[u64]) -> u64 {
    warn!(
        "sceKernelVirtualQuery(addr={:#x}): not implemented — returning EFAULT",
        args.first().copied().unwrap_or(0)
    );
    SCE_KERNEL_ERROR_EFAULT
}

/// `sceKernelCheckReachability(addr, ...)`: verify that the leading byte of
/// the supplied guest address is mapped. Public symbol lists expose the name
/// but not a stronger ABI contract; this bounded probe is therefore the
/// narrowest behavior that distinguishes a real address from null/wild input.
fn hle_check_reachability(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let mut byte = [0u8; 1];
    let reachable = addr != 0 && ctx.mem.read(addr, &mut byte);
    debug!(
        "sceKernelCheckReachability(addr={addr:#x}, arg1={:#x}) -> {}",
        args.get(1).copied().unwrap_or(0),
        if reachable { "OK" } else { "EFAULT" }
    );
    if reachable {
        SCE_OK
    } else {
        SCE_KERNEL_ERROR_EFAULT
    }
}

/// `sceKernelReserveVirtualRange(void **addrInOut, size_t len, int flags,
/// size_t alignment)`: carve the range from the arena's mmap region and
/// write its address back — a reservation the guest can later map over.
fn hle_reserve_virtual_range(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr_inout = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let align = args
        .get(3)
        .copied()
        .unwrap_or(0)
        .max(xps5x_core::PS5_PAGE_SIZE as u64);
    if addr_inout == 0 || len == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let Some(addr) = ctx.alloc.mmap(len, align) else {
        warn!("sceKernelReserveVirtualRange: arena mmap failed (len={len:#x})");
        return HLE_ERROR;
    };
    if !ctx.mem.write(addr_inout, &addr.to_le_bytes()) {
        ctx.alloc.munmap(addr, len);
        return SCE_KERNEL_ERROR_EFAULT;
    }
    debug!("sceKernelReserveVirtualRange(len={len:#x}, align={align:#x}) -> {addr:#x}");
    SCE_OK
}

/// `sceKernelDebugRaiseException*`: the title is reporting a fatal
/// condition. Log it loudly; returning lets the guest continue into
/// whatever it does next (usually an exit path).
fn hle_debug_raise_exception(_ctx: &HleContext, args: &[u64]) -> u64 {
    warn!(
        "sceKernelDebugRaiseException(code={:#x}, arg={:#x}) — guest reported a fatal condition",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
    );
    SCE_OK
}

/// `sceKernelMkdir(path, mode)`: create the directory beneath a registered
/// VFS mount. The VFS rejects unmounted and traversing paths.
fn hle_mkdir(ctx: &HleContext, args: &[u64]) -> u64 {
    let path_ptr = args.first().copied().unwrap_or(0);
    let Some(path_bytes) = crate::fmt::read_cstr(ctx.mem, path_ptr) else {
        return SCE_KERNEL_ERROR_EFAULT;
    };
    let path = String::from_utf8_lossy(&path_bytes);
    match ctx.kernel.filesystem.create_dir_all(&path) {
        Ok(()) => SCE_OK,
        Err(error) => {
            warn!("sceKernelMkdir('{path}') failed: {error}");
            SCE_KERNEL_ERROR_ENOENT
        }
    }
}

const ORBIS_STAT_SIZE: usize = 120;
const ORBIS_MODE_DIRECTORY: u16 = 0x41ff;
const ORBIS_MODE_REGULAR: u16 = 0x81ff;

/// `sceKernelStat(path, stat_out)`: report real metadata for any mounted VFS
/// path. The 120-byte layout is the public Orbis/FreeBSD ABI used by titles.
fn hle_stat(ctx: &HleContext, args: &[u64]) -> u64 {
    let path_ptr = args.first().copied().unwrap_or(0);
    let stat_out = args.get(1).copied().unwrap_or(0);
    if stat_out == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let Some(path_bytes) = crate::fmt::read_cstr(ctx.mem, path_ptr) else {
        return SCE_KERNEL_ERROR_EFAULT;
    };
    let path = String::from_utf8_lossy(&path_bytes);
    match ctx.kernel.filesystem.metadata(&path) {
        Ok(metadata) => write_orbis_stat(ctx, stat_out, &metadata),
        Err(error) => {
            debug!("sceKernelStat('{path}') failed: {error}");
            SCE_KERNEL_ERROR_ENOENT
        }
    }
}

fn write_orbis_stat(ctx: &HleContext, stat_out: u64, metadata: &std::fs::Metadata) -> u64 {
    let is_directory = metadata.is_dir();
    let size = if is_directory { 65_536 } else { metadata.len() };
    let mut stat = [0u8; ORBIS_STAT_SIZE];
    stat[4..8].copy_from_slice(&1u32.to_le_bytes()); // stable nonzero inode
    stat[8..10].copy_from_slice(
        &(if is_directory {
            ORBIS_MODE_DIRECTORY
        } else {
            ORBIS_MODE_REGULAR
        })
        .to_le_bytes(),
    );
    stat[10..12].copy_from_slice(&1u16.to_le_bytes());
    write_orbis_timespec(&mut stat[24..40], metadata.accessed().ok());
    write_orbis_timespec(&mut stat[40..56], metadata.modified().ok());
    write_orbis_timespec(&mut stat[56..72], metadata.modified().ok());
    stat[72..80].copy_from_slice(&size.to_le_bytes());
    let blocks = if is_directory {
        128
    } else {
        size.div_ceil(512)
    };
    stat[80..88].copy_from_slice(&blocks.to_le_bytes());
    stat[88..92].copy_from_slice(&(if is_directory { 65_536u32 } else { 512u32 }).to_le_bytes());
    write_orbis_timespec(&mut stat[104..120], metadata.created().ok());
    if ctx.mem.write(stat_out, &stat) {
        SCE_OK
    } else {
        SCE_KERNEL_ERROR_EFAULT
    }
}

fn write_orbis_timespec(out: &mut [u8], value: Option<std::time::SystemTime>) {
    let Some(duration) = value.and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
    else {
        return;
    };
    out[..8].copy_from_slice(&(duration.as_secs() as i64).to_le_bytes());
    out[8..16].copy_from_slice(&(duration.subsec_nanos() as i64).to_le_bytes());
}

/// Shared honest-failure stub for path metadata calls with no VFS backing.
fn hle_fs_enoent(ctx: &HleContext, args: &[u64]) -> u64 {
    let arg0 = args.first().copied().unwrap_or(0);
    warn!(
        "filesystem metadata call ({:?} / {arg0:#x}): no VFS backing — ENOENT",
        crate::fmt::read_cstr(ctx.mem, arg0).unwrap_or_default()
    );
    SCE_KERNEL_ERROR_ENOENT
}

/// `fstat`/`sceKernelFstat(fd, stat_out)`: report regular-file size for VFS
/// descriptors and a zero-sized character-like record for console fds.
fn hle_fstat(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0);
    let stat_out = args.get(1).copied().unwrap_or(0);
    let size = if fd <= 2 {
        0
    } else if let Some(size) = ctx.kernel.filesystem.file_size(fd as i32) {
        size
    } else {
        warn!("fstat(fd={fd}): no file table backing — EBADF");
        return WRITE_EBADF;
    };
    if stat_out != 0 {
        let mut stat = [0u8; ORBIS_STAT_SIZE];
        stat[8..10].copy_from_slice(&ORBIS_MODE_REGULAR.to_le_bytes());
        stat[10..12].copy_from_slice(&1u16.to_le_bytes());
        stat[72..80].copy_from_slice(&size.to_le_bytes());
        stat[80..88].copy_from_slice(&size.div_ceil(512).to_le_bytes());
        stat[88..92].copy_from_slice(&512u32.to_le_bytes());
        if !ctx.mem.write(stat_out, &stat) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
    }
    SCE_OK
}

/// `scePthreadGetname(thread, name_out)`: the only thread is the main one.
fn hle_pthread_getname(ctx: &HleContext, args: &[u64]) -> u64 {
    let name_out = args.get(1).copied().unwrap_or(0);
    if name_out != 0 {
        let _ = ctx.mem.write(name_out, b"main\0");
    }
    SCE_OK
}

/// `scePthreadOnce` / POSIX `pthread_once`: claim the control word and defer
/// its initializer to the runtime's active guest-callback mechanism. The
/// runtime marks it done only after the guest routine returns and restores it
/// to uninitialized if the callback faults.
fn hle_pthread_once(ctx: &HleContext, args: &[u64]) -> u64 {
    const ONCE_UNINITIALIZED: u32 = 0;
    const ONCE_IN_PROGRESS: u32 = 1;
    const ONCE_DONE: u32 = 2;
    const SCE_KERNEL_ERROR_EAGAIN: u64 = 0x8002_000B;

    let once = args.first().copied().unwrap_or(0);
    let init = args.get(1).copied().unwrap_or(0);
    if once == 0 || init < 0x1_0000 {
        return SCE_KERNEL_ERROR_EINVAL;
    }

    let mut entry_probe = [0u8; 1];
    if !ctx.mem.read(init, &mut entry_probe) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    loop {
        match ctx.mem.atomic_load_u32(once) {
            Some(ONCE_DONE) => return SCE_OK,
            Some(ONCE_IN_PROGRESS) => {
                if ctx.guest_threads.process_is_terminating() {
                    return SCE_KERNEL_ERROR_EAGAIN;
                }
                std::thread::yield_now();
            }
            Some(ONCE_UNINITIALIZED) => {
                match ctx.mem.atomic_compare_exchange_u32(
                    once,
                    ONCE_UNINITIALIZED,
                    ONCE_IN_PROGRESS,
                ) {
                    Some(ONCE_UNINITIALIZED) => break,
                    Some(_) => continue,
                    None => return SCE_KERNEL_ERROR_EFAULT,
                }
            }
            Some(_) => return SCE_KERNEL_ERROR_EINVAL,
            None => return SCE_KERNEL_ERROR_EFAULT,
        }
    }
    let requested = ctx.guest_calls.request(GuestCallRequest {
        entry: init,
        args: [0; 6],
        completion: Some(GuestCallCompletion {
            address: once,
            success_u32: ONCE_DONE,
            failure_u32: ONCE_UNINITIALIZED,
        }),
    });
    if !requested {
        let _ = ctx.mem.atomic_store_u32(once, ONCE_UNINITIALIZED);
        return SCE_KERNEL_ERROR_EAGAIN;
    }

    debug!("pthread_once(once={once:#x}, init={init:#x}) -> deferred guest call");
    SCE_OK
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

/// Host wall-clock time since the Unix epoch as `(seconds, sub-second
/// nanos)`. Clamps a pre-epoch host clock to zero rather than panicking.
fn host_realtime() -> (i64, i64) {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => (d.as_secs() as i64, d.subsec_nanos() as i64),
        Err(_) => (0, 0),
    }
}

/// Real `sceKernelGettimeofday(struct timeval *tp, ...)` (M1 hardening):
/// writes the host wall-clock time as a PS5 `timeval` — two little-endian
/// `int64_t`s, `tv_sec` then `tv_usec` — through `tp`. A homebrew that
/// timestamps or measures elapsed time now reads real, advancing values
/// instead of zero. Unwritable `tp` is a loud `EFAULT`.
pub(crate) fn hle_gettimeofday(ctx: &HleContext, args: &[u64]) -> u64 {
    let tp = args.first().copied().unwrap_or(0);
    debug!("sceKernelGettimeofday(tp={tp:#x})");
    if tp == 0 {
        return SCE_OK; // NULL tp is a defined no-op in the real API.
    }
    let (sec, nanos) = host_realtime();
    let mut buf = [0u8; 16];
    buf[0..8].copy_from_slice(&sec.to_le_bytes());
    buf[8..16].copy_from_slice(&(nanos / 1_000).to_le_bytes()); // tv_usec
    if !ctx.mem.write(tp, &buf) {
        warn!("sceKernelGettimeofday: timeval out-pointer {tp:#x} not writable — EFAULT");
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// The PS5's BSD-derived `CLOCK_MONOTONIC` id. `CLOCK_REALTIME` is 0.
const CLOCK_MONOTONIC: u64 = 4;

/// Real `sceKernelClockGettime(clockId, struct timespec *tp)` (M1
/// hardening): writes a PS5 `timespec` — two little-endian `int64_t`s,
/// `tv_sec` then `tv_nsec` — through `tp`. `CLOCK_MONOTONIC` reports time
/// since a fixed process-start reference (never goes backwards); every
/// other clock id reports host wall-clock (`CLOCK_REALTIME` semantics).
pub(crate) fn hle_clock_gettime(ctx: &HleContext, args: &[u64]) -> u64 {
    let clock_id = args.first().copied().unwrap_or(0);
    let tp = args.get(1).copied().unwrap_or(0);
    debug!("sceKernelClockGettime(clockId={clock_id}, tp={tp:#x})");
    if tp == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }

    let (sec, nsec) = if clock_id == CLOCK_MONOTONIC {
        let elapsed = process_start().elapsed();
        (elapsed.as_secs() as i64, elapsed.subsec_nanos() as i64)
    } else {
        host_realtime()
    };
    let mut buf = [0u8; 16];
    buf[0..8].copy_from_slice(&sec.to_le_bytes());
    buf[8..16].copy_from_slice(&nsec.to_le_bytes());
    if !ctx.mem.write(tp, &buf) {
        warn!("sceKernelClockGettime: timespec out-pointer {tp:#x} not writable — EFAULT");
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// Frequency (Hz) of the process-time counter XPS5X exposes: a nanosecond
/// domain, so `GetProcessTimeCounter` returns elapsed nanoseconds and
/// `GetProcessTimeCounterFrequency` returns `1_000_000_000`.
const PROCESS_TIME_COUNTER_HZ: u64 = 1_000_000_000;

/// Real `sceKernelGetProcessTime()`: microseconds elapsed since the process
/// started (a `u64` return, not an out-param). Titles use this for frame
/// timing and delta-time.
fn hle_get_process_time(_ctx: &HleContext, _args: &[u64]) -> u64 {
    let us = process_start().elapsed().as_micros();
    debug!("sceKernelGetProcessTime() -> {us}us");
    u64::try_from(us).unwrap_or(u64::MAX)
}

/// Real `sceKernelGetProcessTimeCounter()`: elapsed nanoseconds since process
/// start (paired with [`PROCESS_TIME_COUNTER_HZ`]). Monotonic.
fn hle_get_process_time_counter(_ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::try_from(process_start().elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// `sceKernelGetProcessTimeCounterFrequency()`: the counter's frequency in
/// Hz — the divisor a title applies to the counter to get seconds.
fn hle_get_process_time_counter_frequency(_ctx: &HleContext, _args: &[u64]) -> u64 {
    PROCESS_TIME_COUNTER_HZ
}

/// A fixed monotonic reference captured on first use, so `CLOCK_MONOTONIC`
/// reports a stable, never-decreasing elapsed time across the process.
fn process_start() -> std::time::Instant {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    *START.get_or_init(std::time::Instant::now)
}

/// Stub: plausible PS5 base-clock TSC frequency (1.6 GHz).
fn hle_get_tsc_frequency(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetTscFrequency()");
    1_600_000_000
}

/// Upper bound on how long one `sceKernelUsleep` will actually block the
/// host thread, so a wild/huge `usec` (or a guest bug) can't wedge the
/// emulator. 1 second is far longer than any per-frame sleep a title
/// issues; a larger request is honored up to this cap and logged.
const USLEEP_MAX: std::time::Duration = std::time::Duration::from_secs(1);

/// Real `sceKernelUsleep(usec)` (M1 hardening): actually sleeps the host
/// thread for `usec` microseconds (capped by [`USLEEP_MAX`]). A no-op sleep
/// makes timing-driven homebrew busy-spin and burn 100% CPU; a real sleep
/// yields, matching what the title expects between frames.
pub(crate) fn hle_usleep(_ctx: &HleContext, args: &[u64]) -> u64 {
    let usec = args.first().copied().unwrap_or(0);
    debug!("sceKernelUsleep(usec={usec})");
    let requested = std::time::Duration::from_micros(usec);
    let dur = requested.min(USLEEP_MAX);
    if dur < requested {
        warn!(
            "sceKernelUsleep: {usec}us capped to {}us (USLEEP_MAX)",
            dur.as_micros()
        );
    }
    std::thread::sleep(dur);
    SCE_OK
}

/// Stub: returns a fake, non-null pointer value. The real function returns
/// a pointer into the guest's process-parameter block, which this stub
/// cannot construct.
/// `sceKernelGetProcParam()`: returns the guest address of the loaded
/// module's `PT_SCE_PROCPARAM` block — the process-parameter block carrying
/// the SDK version and process metadata — when the runtime recorded one at
/// load time. Falls back to a plausible non-null sentinel only when the
/// module had no procparam segment (so a caller never gets NULL).
fn hle_get_proc_param(ctx: &HleContext, _args: &[u64]) -> u64 {
    let addr = ctx.kernel.proc_param_addr();
    debug!("sceKernelGetProcParam() -> {addr:#x}");
    if addr != 0 {
        addr
    } else {
        0x0000_1000_0000_0000 // no PT_SCE_PROCPARAM in the module — sentinel
    }
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

    /// libc.prx's `module_start` registers its per-thread destructor hooks
    /// with the rtld before doing anything else — a missing
    /// `_sceKernelSetThreadDtors` was the exact wall a real title (Minecraft)
    /// died on. The three Set* calls record a guest callback pointer and
    /// return `SCE_OK`; the Increment/Decrement pair adjusts a guest-memory
    /// `u64` counter in place and returns the adjusted value (never
    /// underflowing past zero).
    #[test]
    fn rtld_thread_dtor_family_records_callbacks_and_adjusts_the_counter() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        for name in [
            "_sceKernelSetThreadDtors",
            "_sceKernelSetThreadAtexitCount",
            "_sceKernelSetThreadAtexitReport",
        ] {
            assert_eq!(
                registry.call(&ctx, "libkernel", name, &[0x12_3456]),
                Some(SCE_OK),
                "{name} must accept a callback and return SCE_OK"
            );
        }

        // Counter at guest 0x100 starts at 5; +1 → 6, -1 -1 → 4.
        assert!(mem.write(0x100, &5u64.to_le_bytes()));
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "_sceKernelRtldThreadAtexitIncrement",
                &[0x100]
            ),
            Some(6)
        );
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "_sceKernelRtldThreadAtexitDecrement",
                &[0x100]
            ),
            Some(5)
        );
        let mut counter = [0u8; 8];
        assert!(mem.read(0x100, &mut counter));
        assert_eq!(
            u64::from_le_bytes(counter),
            5,
            "adjustment must be written back"
        );

        // Decrementing a zero counter saturates at zero, never wraps.
        assert!(mem.write(0x108, &0u64.to_le_bytes()));
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "_sceKernelRtldThreadAtexitDecrement",
                &[0x108]
            ),
            Some(0)
        );

        // NULL counter is EINVAL; an unmapped counter address is EFAULT.
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "_sceKernelRtldThreadAtexitIncrement",
                &[0]
            ),
            Some(SCE_KERNEL_ERROR_EINVAL)
        );
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "_sceKernelRtldThreadAtexitIncrement",
                &[0xFFFF_0000]
            ),
            Some(SCE_KERNEL_ERROR_EFAULT)
        );
    }

    /// The measured Minecraft libc.prx boot surface: every import name in the
    /// batch resolves, and the ones with real behavior behave.
    #[test]
    fn minecraft_libc_boot_surface_resolves_and_behaves() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x40000);
        let alloc = crate::TestAllocator::new(0x1000);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Every name in the batch is registered (a typo here = an unresolved
        // NID at link time).
        for (lib, name) in [
            ("libkernel", "_open"),
            ("libkernel", "_read"),
            ("libkernel", "_write"),
            ("libkernel", "_close"),
            ("libkernel", "_exit"),
            ("libkernel", "nanosleep"),
            ("libkernel", "_sigprocmask"),
            ("libkernel", "_sceKernelRtldSetApplicationHeapAPI"),
            ("libkernel", "sceKernelIsAddressSanitizerEnabled"),
            ("libkernel", "sceKernelGetSanitizerMallocReplaceExternal"),
            ("libkernel", "sceKernelGetSanitizerNewReplaceExternal"),
            ("libkernel", "__error"),
            ("libkernel", "__pthread_cxa_finalize"),
            ("libkernel", "__elf_phdr_match_addr"),
            ("libkernel", "sceKernelMprotect"),
            ("libkernel", "sceKernelCheckReachability"),
            ("libkernel", "sceKernelUuidCreate"),
            ("libkernel", "sceKernelConvertUtcToLocaltime"),
            ("libkernel", "sceKernelConvertLocaltimeToUtc"),
            ("libkernel", "sceKernelGetModuleInfoForUnwind"),
            ("libkernel", "sceKernelGetModuleInfoFromAddr"),
            ("libkernel", "sceKernelVirtualQuery"),
            ("libkernel", "sceKernelReserveVirtualRange"),
            ("libkernel", "sceKernelMapNamedFlexibleMemoryInternal"),
            ("libkernel", "sceKernelMapNamedDirectMemory"),
            ("libkernel", "sceKernelDebugRaiseException"),
            ("libkernel", "sceKernelDebugRaiseExceptionOnReleaseMode"),
            ("libkernel", "sceKernelMkdir"),
            ("libkernel", "sceKernelUnlink"),
            ("libkernel", "sceKernelRmdir"),
            ("libkernel", "sceKernelStat"),
            ("libkernel", "sceKernelFstat"),
            ("libkernel", "sceKernelGetdents"),
            ("libkernel", "scePthreadDetach"),
            ("libkernel", "scePthreadSetprio"),
            ("libkernel", "scePthreadSetaffinity"),
            ("libkernel", "scePthreadAttrSetaffinity"),
            ("libkernel", "scePthreadAttrGetaffinity"),
            ("libkernel", "scePthreadAttrGet"),
            ("libkernel", "scePthreadAttrSetschedparam"),
            ("libkernel", "scePthreadAttrGetschedparam"),
            ("libkernel", "scePthreadAttrSetinheritsched"),
            ("libkernel", "scePthreadGetname"),
            ("libkernel", "scePthreadOnce"),
            ("libScePosix", "pthread_once"),
            ("libkernel", "_is_signal_return"),
            ("libkernel", "__tls_get_addr"),
            ("libkernel", "scePthreadCondBroadcast"),
            ("libkernel", "scePthreadCondDestroy"),
            ("libScePosix", "pthread_setschedparam"),
            ("libScePosix", "fstat"),
        ] {
            assert!(
                registry.is_implemented(lib, name),
                "{lib}::{name} must be registered"
            );
        }

        // __error hands out one stable guest errno slot the guest can write.
        let errno_addr = hle_error_addr(&ctx, &[]);
        assert_ne!(errno_addr, 0, "__error must allocate a guest slot");
        assert_eq!(hle_error_addr(&ctx, &[]), errno_addr, "slot must be stable");
        assert!(mem.write(errno_addr, &42u64.to_le_bytes()));
        assert_eq!(hle_check_reachability(&ctx, &[0x100]), SCE_OK);
        assert_eq!(hle_check_reachability(&ctx, &[0]), SCE_KERNEL_ERROR_EFAULT);
        assert_eq!(
            hle_check_reachability(&ctx, &[0xFFFF_0000]),
            SCE_KERNEL_ERROR_EFAULT
        );

        // Dynamic TLS descriptors are `{ module_id, offset }`. libc asks
        // libkernel for one stable zero-initialized block per module; two
        // calls for the same module must alias, while a different module
        // must not. Returning zero here merely moves the crash to libc's
        // first load/store through the result.
        assert!(mem.write(0x100, &7u64.to_le_bytes()));
        assert!(mem.write(0x108, &0x28u64.to_le_bytes()));
        let tls_7 = hle_tls_get_addr(&ctx, &[0x100]);
        assert_ne!(tls_7, 0);
        assert_eq!(hle_tls_get_addr(&ctx, &[0x100]), tls_7);
        let mut zero = [0xAAu8; 8];
        assert!(mem.read(tls_7, &mut zero));
        assert_eq!(zero, [0; 8], "new TLS storage must be zero-filled");

        assert!(mem.write(0x120, &8u64.to_le_bytes()));
        assert!(mem.write(0x128, &0x28u64.to_le_bytes()));
        let tls_8 = hle_tls_get_addr(&ctx, &[0x120]);
        assert_ne!(tls_8, tls_7, "different modules need distinct TLS blocks");
        assert_eq!(hle_tls_get_addr(&ctx, &[0]), SCE_KERNEL_ERROR_EINVAL);
        assert_eq!(hle_tls_get_addr(&ctx, &[0x3FFF8]), SCE_KERNEL_ERROR_EFAULT);

        // UuidCreate writes 16 nonzero-entropy bytes.
        assert_eq!(hle_uuid_create(&ctx, &[0x300]), SCE_OK);
        let mut uuid = [0u8; 16];
        assert!(mem.read(0x300, &mut uuid));
        assert_ne!(uuid, [0u8; 16], "uuid must not be all zeros");
        assert_eq!(hle_uuid_create(&ctx, &[0]), SCE_KERNEL_ERROR_EINVAL);

        // Time conversion is the identity: input time_t written through out.
        assert_eq!(
            hle_convert_time_identity(&ctx, &[0x1234_5678, 0x380]),
            SCE_OK
        );
        let mut t = [0u8; 8];
        assert!(mem.read(0x380, &mut t));
        assert_eq!(u64::from_le_bytes(t), 0x1234_5678);

        // nanosleep with a wild request is bounded, returns OK, zeroes rem.
        assert!(mem.write(0x400, &u64::MAX.to_le_bytes())); // tv_sec
        assert!(mem.write(0x408, &0u64.to_le_bytes())); // tv_nsec
        assert!(mem.write(0x410, &u64::MAX.to_le_bytes())); // rem, pre-poisoned
        assert_eq!(hle_nanosleep(&ctx, &[0x400, 0x410]), SCE_OK);
        let mut rem = [0u8; 8];
        assert!(mem.read(0x410, &mut rem));
        assert_eq!(u64::from_le_bytes(rem), 0, "rem must report zero remaining");

        // ReserveVirtualRange writes a real arena address through addrInOut.
        assert_eq!(
            hle_reserve_virtual_range(&ctx, &[0x500, 0x4000, 0, 0]),
            SCE_OK
        );
        let mut reserved = [0u8; 8];
        assert!(mem.read(0x500, &mut reserved));
        assert_ne!(u64::from_le_bytes(reserved), 0);

        // The heap-API table pointer is recorded.
        assert_eq!(
            hle_rtld_set_application_heap_api(&ctx, &[0xBEEF_0000]),
            SCE_OK
        );
        assert_eq!(
            kernel.application_heap_api.load(Ordering::Relaxed),
            0xBEEF_0000
        );

        // fstat: console fds report a zeroed stat, others EBADF.
        assert_eq!(hle_fstat(&ctx, &[1, 0x600]), SCE_OK);
        assert_eq!(hle_fstat(&ctx, &[9, 0x600]) as i64, -9);
    }

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

    /// M1 hardening: the clock functions write real, plausible time into
    /// their guest out-params instead of leaving them zero.
    #[test]
    fn gettimeofday_and_clock_gettime_write_real_time() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // gettimeofday: tv_sec must be a recent Unix timestamp (> 2020).
        assert_eq!(hle_gettimeofday(&ctx, &[0x100]), SCE_OK);
        let mut tv = [0u8; 16];
        assert!(mem.read(0x100, &mut tv));
        let tv_sec = i64::from_le_bytes(tv[0..8].try_into().unwrap());
        let tv_usec = i64::from_le_bytes(tv[8..16].try_into().unwrap());
        assert!(
            tv_sec > 1_577_836_800,
            "tv_sec must be a real epoch time, got {tv_sec}"
        );
        assert!(
            (0..1_000_000).contains(&tv_usec),
            "tv_usec must be in [0,1e6), got {tv_usec}"
        );

        // clock_gettime(CLOCK_MONOTONIC): tv_sec/tv_nsec well-formed, nsec in range.
        assert_eq!(hle_clock_gettime(&ctx, &[CLOCK_MONOTONIC, 0x200]), SCE_OK);
        let mut ts = [0u8; 16];
        assert!(mem.read(0x200, &mut ts));
        let tv_nsec = i64::from_le_bytes(ts[8..16].try_into().unwrap());
        assert!(
            (0..1_000_000_000).contains(&tv_nsec),
            "tv_nsec must be in [0,1e9), got {tv_nsec}"
        );

        // Unwritable out-pointer → EFAULT, not a panic.
        assert_eq!(
            hle_gettimeofday(&ctx, &[0xDEAD_0000]),
            SCE_KERNEL_ERROR_EFAULT
        );
        assert_eq!(
            hle_clock_gettime(&ctx, &[0, 0xDEAD_0000]),
            SCE_KERNEL_ERROR_EFAULT
        );
    }

    /// Real VFS-backed file I/O: a homebrew opens a file under /app0,
    /// reads its bytes into a guest buffer, seeks, reads again, and closes —
    /// all against a real host temp file mounted into the VFS.
    #[test]
    fn savedata_write_through_hle_open_write_close_persists_to_host() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp = std::env::temp_dir().join(format!("xps5x-hle-savewrite-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        // open("/app0/save.dat", O_WRONLY|O_CREAT|O_TRUNC) through the HLE.
        assert!(mem.write(0x100, b"/app0/save.dat\0"));
        use xps5x_kernel::filesystem::open_flags::*;
        let flags = (O_WRONLY | O_CREAT | O_TRUNC) as u64;
        let fd = hle_open(&ctx, &[0x100, flags, 0o644]);
        assert!(
            (fd as i64) >= 3,
            "open must return a real fd, got {}",
            fd as i64
        );

        // write("PROGRESS") through the HLE write() (non-console fd → VFS).
        assert!(mem.write(0x200, b"PROGRESS"));
        assert_eq!(hle_write(&ctx, &[fd, 0x200, 8]), 8);

        // fsync() persists while keeping the descriptor open.
        assert_eq!(hle_fsync(&ctx, &[fd]), SCE_OK);
        assert_eq!(std::fs::read(tmp.join("save.dat")).unwrap(), b"PROGRESS");
        assert_eq!(hle_fsync(&ctx, &[0x7fff]), 0x8002_0009);

        // close() remains valid after fsync and does not lose data.
        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        assert_eq!(std::fs::read(tmp.join("save.dat")).unwrap(), b"PROGRESS");

        let registry = HleRegistry::new();
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0x7d3c_7aea_5e62_5880 && key == "libkernel::sceKernelFsync"
                })
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn file_io_open_read_seek_close_against_a_real_host_file() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Mount a temp dir at /app0/ and drop a real file in it.
        let tmp = std::env::temp_dir().join(format!("xps5x-hle-fileio-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("data.bin"), b"HELLO_WORLD").unwrap();
        // Point the existing /app0/ mount at the temp dir (set_game_directory
        // updates it in place; a fresh `mount` would append a second, lower-
        // priority /app0/ that resolve_path never reaches).
        kernel.filesystem.set_game_directory(&tmp);

        // open("/app0/data.bin")
        assert!(mem.write(0x100, b"/app0/data.bin\0"));
        let fd = hle_open(&ctx, &[0x100, 0, 0]);
        assert!(
            (fd as i64) >= 3,
            "open must return a real fd, got {}",
            fd as i64
        );

        // read 5 bytes → "HELLO"
        let n = hle_read(&ctx, &[fd, 0x200, 5]);
        assert_eq!(n, 5);
        let mut buf = [0u8; 5];
        assert!(mem.read(0x200, &mut buf));
        assert_eq!(&buf, b"HELLO");

        // lseek to absolute offset 6 (SEEK_SET), read 5 → "WORLD"
        assert_eq!(hle_lseek(&ctx, &[fd, 6, 0]), 6);
        let n2 = hle_read(&ctx, &[fd, 0x210, 5]);
        assert_eq!(n2, 5);
        let mut buf2 = [0u8; 5];
        assert!(mem.read(0x210, &mut buf2));
        assert_eq!(&buf2, b"WORLD");

        // SEEK_END gives the file size (11).
        assert_eq!(hle_lseek(&ctx, &[fd, 0, 2]), 11);
        // A read at EOF returns 0.
        assert_eq!(hle_read(&ctx, &[fd, 0x220, 5]), 0);

        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        // Double close / bad fd → EBADF.
        assert_eq!(hle_close(&ctx, &[fd]) as i64, -9);
        assert_eq!(hle_read(&ctx, &[fd, 0x220, 5]) as i64, -9);

        // open of a missing file → ENOENT.
        assert!(mem.write(0x300, b"/app0/nope.bin\0"));
        assert_eq!(hle_open(&ctx, &[0x300, 0, 0]) as i64, -2);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn directory_open_and_getdents_expose_gen5_records() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x2000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let tmp = std::env::temp_dir().join(format!("xps5x-hle-dir-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("packs")).unwrap();
        std::fs::write(tmp.join("manifest.json"), b"{}").unwrap();
        kernel.filesystem.set_game_directory(&tmp);
        assert!(mem.write(0x100, b"/app0\0"));
        let fd = hle_open(&ctx, &[0x100, 0, 0]);
        assert!((fd as i64) >= 3);
        assert_eq!(hle_getdents(&ctx, &[fd, 0x400, 1024, 0x200]), 1024);
        let mut first = [0u8; 512];
        assert!(mem.read(0x400, &mut first));
        assert_eq!(u16::from_le_bytes(first[4..6].try_into().unwrap()), 512);
        assert!(matches!(first[6], 4 | 8));
        assert_ne!(first[7], 0);
        assert_eq!(hle_getdents(&ctx, &[fd, 0x400, 1024]), 0);
        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn stat_and_mkdir_use_mounted_vfs_roots() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let tmp = std::env::temp_dir().join(format!("xps5x-hle-stat-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        kernel.filesystem.set_temp_directory(&tmp);

        assert!(mem.write(0x100, b"/temp0/cache\0"));
        assert_eq!(hle_mkdir(&ctx, &[0x100, 0o755]), SCE_OK);
        assert!(tmp.join("cache").is_dir());

        assert_eq!(hle_stat(&ctx, &[0x100, 0x200]), SCE_OK);
        let mut stat = [0u8; ORBIS_STAT_SIZE];
        assert!(mem.read(0x200, &mut stat));
        assert_eq!(
            u16::from_le_bytes(stat[8..10].try_into().unwrap()),
            ORBIS_MODE_DIRECTORY
        );
        assert_eq!(u64::from_le_bytes(stat[72..80].try_into().unwrap()), 65_536);

        // The exact mount root is a valid directory too; it must not require
        // a trailing slash to resolve.
        assert!(mem.write(0x120, b"/temp0\0"));
        assert_eq!(hle_stat(&ctx, &[0x120, 0x300]), SCE_OK);

        assert!(mem.write(0x140, b"/temp0/missing\0"));
        assert_eq!(hle_stat(&ctx, &[0x140, 0x400]), SCE_KERNEL_ERROR_ENOENT);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// AvailableFlexibleMemorySize writes the size through its out-param
    /// (not the return value) and returns OK; NULL → EINVAL.
    #[test]
    fn available_flexible_memory_writes_out_param() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_available_flexible_memory_size(&ctx, &[0x40]), SCE_OK);
        let mut v = [0u8; 8];
        assert!(mem.read(0x40, &mut v));
        assert_eq!(u64::from_le_bytes(v), FLEXIBLE_MEMORY_SIZE);
        assert_eq!(
            hle_available_flexible_memory_size(&ctx, &[0]),
            SCE_KERNEL_ERROR_EINVAL
        );
    }

    /// M1 hardening: GetCompiledSdkVersion writes the PS5 SDK version out and
    /// validates its pointer; getpid returns a stable nonzero pid.
    #[test]
    fn get_proc_param_returns_the_runtime_recorded_address() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Before load: no procparam → the non-null sentinel (never NULL).
        assert_ne!(hle_get_proc_param(&ctx, &[]), 0);

        // Runtime records the block's guest address; GetProcParam returns it.
        kernel.set_proc_param_addr(0x1234_5000);
        assert_eq!(hle_get_proc_param(&ctx, &[]), 0x1234_5000);
    }

    #[test]
    fn compiled_sdk_version_and_getpid() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_compiled_sdk_version(&ctx, &[0x40]), SCE_OK);
        let mut v = [0u8; 4];
        assert!(mem.read(0x40, &mut v));
        assert_eq!(u32::from_le_bytes(v), GEN5_SDK_VERSION, "PS5 SDK 9.00");

        assert_eq!(
            hle_get_compiled_sdk_version(&ctx, &[0]),
            SCE_KERNEL_ERROR_EINVAL,
            "NULL → EINVAL"
        );
        assert_eq!(
            hle_get_compiled_sdk_version(&ctx, &[0xDEAD_0000]),
            SCE_KERNEL_ERROR_EFAULT
        );

        assert_ne!(hle_getpid(&ctx, &[]), 0, "pid must be nonzero");
        assert_eq!(
            hle_getpid(&ctx, &[]),
            hle_getpid(&ctx, &[]),
            "pid is stable"
        );
    }

    /// Process-time counters advance monotonically and agree on their domain.
    #[test]
    fn process_time_counters_advance_and_are_consistent() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_get_process_time_counter_frequency(&ctx, &[]),
            PROCESS_TIME_COUNTER_HZ
        );
        let t0 = hle_get_process_time(&ctx, &[]);
        let c0 = hle_get_process_time_counter(&ctx, &[]);
        std::thread::sleep(std::time::Duration::from_millis(2));
        let t1 = hle_get_process_time(&ctx, &[]);
        let c1 = hle_get_process_time_counter(&ctx, &[]);
        assert!(t1 >= t0, "process time must be monotonic");
        assert!(c1 > c0, "process-time counter must advance");
        // Counter is nanoseconds, GetProcessTime is microseconds — same clock.
        assert!(c1 >= t1 * 1000, "counter (ns) must be ~1000× the time (us)");
    }

    /// usleep sleeps a real (bounded) amount and returns OK.
    #[test]
    fn usleep_sleeps_and_caps() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let t0 = std::time::Instant::now();
        assert_eq!(hle_usleep(&ctx, &[2000]), SCE_OK); // 2 ms
        assert!(
            t0.elapsed() >= std::time::Duration::from_millis(1),
            "usleep must actually sleep"
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
            "sceKernelOpen",
            "sceKernelRead",
            "sceKernelClose",
            "sceKernelLseek",
            "sceKernelGetCompiledSdkVersion",
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
    fn main_direct_memory_maps_the_reserved_identity_range_and_writes_addr_out() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x10_0000);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelAllocateMainDirectMemory",
                &[0x8000, 0x2000, 0, 0x100]
            ),
            Some(SCE_OK)
        );
        let mut bytes = [0u8; 8];
        assert!(mem.read(0x100, &mut bytes));
        let physical = u64::from_le_bytes(bytes);
        assert_ne!(physical, 0);
        assert_eq!(physical & 0x1FFF, 0);

        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelMapNamedDirectMemory",
                &[0x108, 0x8000, 0x32, 0, physical, 0x2000, 0]
            ),
            Some(SCE_OK)
        );
        assert!(mem.read(0x108, &mut bytes));
        assert_eq!(u64::from_le_bytes(bytes), physical);
        assert!(kernel.memory.is_mapped(physical));

        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelReleaseDirectMemory",
                &[physical, 0x8000]
            ),
            Some(SCE_OK)
        );
        assert!(!kernel.memory.is_mapped(physical));
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
