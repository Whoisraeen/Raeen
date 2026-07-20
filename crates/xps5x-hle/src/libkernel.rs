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

/// The TLS module ID the linker writes into the main module's `DTPMOD64`
/// relocation slots, and therefore the id a general-dynamic access to the
/// executable's own thread-locals arrives here with.
///
/// Must equal `xps5x_firmware`'s `MAIN_TLS_MODULE_ID`. It is duplicated rather
/// than imported because the dependency runs the other way — `xps5x-firmware`
/// depends on this crate to resolve NIDs at link time, so importing it back
/// would be a cycle. Pinned against the linker's value by
/// `main_tls_module_id_matches_the_linkers` in `xps5x-firmware`.
const MAIN_TLS_MODULE_ID: u64 = 1;

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
    match ctx.services.write(fd as i32, &bytes) {
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

/// Store a POSIX errno value in the calling guest thread's `__error()` slot.
pub(crate) fn set_guest_errno(ctx: &HleContext, errno: i32) {
    let slot = hle_error_addr(ctx, &[]);
    if slot != 0 && !ctx.mem.write(slot, &errno.to_le_bytes()) {
        warn!("failed to write errno={errno} to guest slot {slot:#x}");
    }
}

/// Adapt this module's internal `-errno` file result to POSIX `-1` + errno.
fn file_result_posix(ctx: &HleContext, result: u64) -> u64 {
    let signed = result as i64;
    if (-4095..0).contains(&signed) {
        set_guest_errno(ctx, (-signed) as i32);
        (-1i64) as u64
    } else {
        result
    }
}

/// Adapt this module's internal `-errno` file result to an SCE kernel error.
fn file_result_sce(result: u64) -> u64 {
    let signed = result as i64;
    if (-4095..0).contains(&signed) {
        0x8002_0000 | (-signed as u64)
    } else {
        result
    }
}

fn sce_result_posix(ctx: &HleContext, result: u64) -> u64 {
    if result & 0xffff_0000 == 0x8002_0000 {
        set_guest_errno(ctx, (result & 0xffff) as i32);
        (-1i64) as u64
    } else {
        result
    }
}

fn hle_posix_open(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_open(ctx, args))
}

fn hle_sce_open(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_open(ctx, args))
}

fn hle_posix_read(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_read(ctx, args))
}

fn hle_sce_read(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_read(ctx, args))
}

fn hle_posix_write(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_write(ctx, args))
}

fn hle_sce_write(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_write(ctx, args))
}

fn hle_posix_close(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_close(ctx, args))
}

fn hle_sce_close(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_close(ctx, args))
}

fn hle_posix_lseek(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_lseek(ctx, args))
}

fn hle_sce_lseek(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_lseek(ctx, args))
}

fn hle_posix_getdents(ctx: &HleContext, args: &[u64]) -> u64 {
    sce_result_posix(ctx, hle_getdents(ctx, args, false))
}

fn hle_posix_getdirentries(ctx: &HleContext, args: &[u64]) -> u64 {
    sce_result_posix(ctx, hle_getdents(ctx, args, true))
}

fn hle_sce_getdents(ctx: &HleContext, args: &[u64]) -> u64 {
    hle_getdents(ctx, args, false)
}

fn hle_sce_getdirentries(ctx: &HleContext, args: &[u64]) -> u64 {
    hle_getdents(ctx, args, true)
}

fn hle_sce_fsync(ctx: &HleContext, args: &[u64]) -> u64 {
    hle_fsync(ctx, args)
}

/// If `missing` is a font file (`.otf`/`.ttf`/`.ttc`) that does not exist,
/// return the file *name* of a shipped sibling font in the same directory to
/// substitute for it (alphabetically first, for determinism), or `None` if the
/// directory ships no other font. Used by [`hle_open`]'s font fallback.
fn font_fallback_sibling(missing: &std::path::Path) -> Option<String> {
    let is_font = |p: &std::path::Path| {
        p.extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| matches!(x.to_ascii_lowercase().as_str(), "otf" | "ttf" | "ttc"))
    };
    if !is_font(missing) {
        return None;
    }
    let dir = missing.parent()?;
    let mut candidates: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_font(p))
        .filter_map(|p| p.file_name()?.to_str().map(String::from))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

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

    // RE probe (XPS5X_TRACE_UI): the caller of the routes.json open is the
    // Ore-UI/Gameface route-table processor — on the UI-INIT path (unlike the
    // render chain, which is a proven dead-end). Its return-addr + guest-stack
    // chain point at the code that decides whether to navigate to a route; aim
    // `xps5x --disas` there to find the never-taken CreateView/LoadURL branch.
    if std::env::var_os("XPS5X_TRACE_UI").is_some() && path.contains("routes.json") {
        const BASE: u64 = 0x0000_1000_0000_0000;
        const SPAN: u64 = 0x1_0000_0000;
        let mut chain: Vec<String> = Vec::new();
        let mut word = [0u8; 8];
        for slot in 0..512u64 {
            let Some(a) = ctx.caller_rsp.checked_add(slot * 8) else {
                break;
            };
            if !ctx.mem.read(a, &mut word) {
                break;
            }
            let v = u64::from_le_bytes(word);
            if (BASE..BASE + SPAN).contains(&v) {
                chain.push(format!("{:#x}", v - BASE));
                if chain.len() >= 16 {
                    break;
                }
            }
        }
        warn!(
            path = %path,
            caller = format_args!("{:#x}", ctx.caller_return_addr.wrapping_sub(BASE)),
            chain = %chain.join(" <- "),
            "TRACE_UI: routes.json opened by the route-table processor"
        );
        // Dump the UI-manager singleton's live vtable so `xps5x --disas` can map
        // which slot is Navigate/LoadURL/CreateView. Read obj = *[0xE39E098],
        // vtable = *[obj+0], then slots 0..24 as arena-relative fn pointers.
        let rd = |a: u64| -> Option<u64> {
            let mut w = [0u8; 8];
            ctx.mem.read(a, &mut w).then(|| u64::from_le_bytes(w))
        };
        // The manager calls [obj+0x10]/[obj+0x18] DIRECTLY — the function
        // pointers are embedded in the object (a C-style dispatch table), not
        // behind a separate C++ vtable. Dump obj+i*8 as fn pointers by byte
        // OFFSET so a code-pointer slot (Navigate/LoadURL/CreateView) stands out.
        if let Some(obj) = rd(BASE + 0xE39_E098).filter(|&o| o != 0) {
            let slots: Vec<String> = (0..24u64)
                .filter_map(|i| match rd(obj + i * 8) {
                    Some(f) if (BASE..BASE + SPAN).contains(&f) => {
                        Some(format!("+{:#x}={:#x}", i * 8, f - BASE))
                    }
                    _ => None,
                })
                .collect();
            warn!(
                obj = format_args!("{:#x}", obj.wrapping_sub(BASE)),
                "TRACE_UI: UI-manager code-pointer slots (offset=target): {}",
                slots.join(" ")
            );
        }
    }

    // A missing file is ENOENT *unless* the guest passed O_CREAT (then the VFS
    // creates it). O_CREAT is bit 0x200 in the Orbis/BSD flag set.
    const O_CREAT: i32 = 0x200;
    let creating = flags & O_CREAT != 0;
    match ctx.kernel.filesystem.resolve_path(&path) {
        Some(host) if host.exists() || creating => {}
        Some(host) => {
            // Font-file fallback. The title reads its fonts with its OWN
            // OpenType renderer and null-dereferences if an open fails, yet it
            // references font variants / PS5 system fonts it does not ship
            // (e.g. FuturaStd-Medium, SIE-ShinGoPr6N, HeiseiMaruGo). Substitute
            // a shipped sibling font so the renderer parses valid tables
            // (codepoints the substitute lacks are handled by the title's own
            // cmap-miss path) instead of faulting on a null font object. Only
            // triggers on a genuinely-missing font whose directory ships another.
            if !creating
                && let Some(fb_name) = font_fallback_sibling(&host)
                && let Some(slash) = path.rfind('/')
            {
                let fb_path = format!("{}/{fb_name}", &path[..slash]);
                warn!("open: '{path}' missing — substituting shipped font '{fb_path}'");
                return match ctx.services.open(&fb_path, flags, mode) {
                    Ok(fd) => fd as u64,
                    Err(e) => {
                        warn!("open: font substitute '{fb_path}' failed: {e} — ENOENT");
                        FILE_ENOENT
                    }
                };
            }
            warn!(
                "open: '{path}' → '{}' does not exist (no O_CREAT) — ENOENT",
                host.display()
            );
            return FILE_ENOENT;
        }
        None if path == "/" && !creating => {}
        None => {
            warn!("open: '{path}' matches no VFS mount — ENOENT");
            return FILE_ENOENT;
        }
    }

    match ctx.services.open(&path, flags, mode) {
        Ok(fd) => {
            // Name every SUCCESSFUL open too. Only failures were logged before,
            // which makes "the title never touched this file" and "it opened it
            // fine" indistinguishable in a boot trace — the exact ambiguity that
            // hid whether the Ore-UI menu HTML is ever loaded.
            debug!("open: '{path}' -> fd {fd}");
            fd as u64
        }
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
    match ctx.services.read(fd, n) {
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

/// Real `pread(fd, buf, nbyte, offset)` / `sceKernelPread` (VFS-backed):
/// reads up to `nbyte` bytes at absolute `offset` without moving the
/// descriptor's cursor — streaming loaders issue these concurrently with
/// sequential reads on the same fd. Measured: ASTRO.BOT's asset streamer
/// calls it during boot (its import was the first unresolved-NID fault once
/// boot reached the streaming path).
fn hle_pread(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let buf = args.get(1).copied().unwrap_or(0);
    let count = args.get(2).copied().unwrap_or(0).min(READ_MAX_BYTES);
    let offset = args.get(3).copied().unwrap_or(0);
    debug!("pread(fd={fd}, buf={buf:#x}, count={count:#x}, offset={offset:#x})");

    if (offset as i64) < 0 {
        return FILE_EINVAL;
    }
    let Ok(n) = usize::try_from(count) else {
        return FILE_EINVAL;
    };
    match ctx.kernel.filesystem.pread(fd, n, offset) {
        Ok(bytes) => {
            if bytes.is_empty() {
                return 0; // EOF (or a read wholly past it) — a valid short read.
            }
            if !ctx.mem.write(buf, &bytes) {
                warn!(
                    "pread: guest buffer {buf:#x} (+{}) not writable — EFAULT",
                    bytes.len()
                );
                return FILE_EFAULT;
            }
            bytes.len() as u64
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => FILE_EBADF,
        Err(_) => FILE_EINVAL,
    }
}

fn hle_posix_pread(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_pread(ctx, args))
}

fn hle_sce_pread(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_pread(ctx, args))
}

/// Real `close(fd)` / `sceKernelClose`: closes the VFS descriptor. Unknown
/// fd → `EBADF`.
fn hle_close(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    debug!("close(fd={fd})");
    if ctx.services.socket_exists(fd) {
        return if ctx.services.close_socket(fd) {
            SCE_OK
        } else {
            FILE_EBADF
        };
    }
    match ctx.services.close(fd) {
        Ok(()) => SCE_OK,
        Err(_) => FILE_EBADF,
    }
}

/// `sceKernelFsync(fd)`: persist the VFS write-back buffer while leaving the
/// descriptor open. The SCE spelling returns the kernel errno encoding.
fn hle_fsync(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    match ctx.services.sync(fd) {
        Ok(()) => SCE_OK,
        Err(e) => {
            warn!("fsync: fd {fd} failed: {e} â€” EBADF");
            0x8002_0009
        }
    }
}

/// `sceKernelAioInitializeParam(param, size)`: the AIO scheduler parameter
/// block, done **synchronously** (mission's measured call: Dragon Ball hits
/// this right after its engine allocator comes up). There is no async AIO
/// backend yet — the honest model is "the init succeeded; requests complete
/// immediately when they arrive". Zero the block (a clean, defined default)
/// rather than leaving guest garbage the title might read back as a schedule.
fn hle_aio_initialize_param(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    let size = usize::try_from(args.get(1).copied().unwrap_or(0)).unwrap_or(0);
    debug!("sceKernelAioInitializeParam(param={param:#x}, size={size:#x})");
    if param == 0 || size == 0 || size > 0x10000 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let bytes = vec![0u8; size];
    if !ctx.mem.write(param, &bytes) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// `sceKernelAioInitializeImpl(...)`: starts the AIO scheduler — in the
/// synchronous model this is a no-op that must report success so the title
/// proceeds to submit requests (which complete immediately when they arrive).
fn hle_aio_initialize_impl(ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelAioInitializeImpl(args=[{:#x}, {:#x}, {:#x}, {:#x}]) -> 0 (synchronous AIO model)",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0),
        args.get(3).copied().unwrap_or(0),
    );
    let _ = ctx;
    SCE_OK
}

/// Record types inside an AMPR command buffer (SharpEmu `AmprExports`).
const APR_RECORD_READ_FILE: u32 = 1;
const APR_RECORD_KERNEL_EVENT_QUEUE: u32 = 2;
const APR_RECORD_WRITE_ADDRESS: u32 = 3;

/// Complete one AMPR command buffer synchronously: walk its records and do
/// the work a console does async — read files by APR id, fire completion
/// events, write completion addresses. SharpEmu `AmprExports.CompleteCommandBuffer`.
fn apr_complete_command_buffer(ctx: &HleContext, cb: u64) -> u64 {
    // The visible struct carries data ptr @0x08 / size @0x10; the write
    // cursor is host-tracked (`ampr_write_offsets`).
    let mut data = [0u8; 8];
    let mut size = [0u8; 8];
    if !ctx.mem.read(cb + 0x08, &mut data) || !ctx.mem.read(cb + 0x10, &mut size) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let buffer = u64::from_le_bytes(data);
    let end = ctx
        .kernel
        .ampr_write_offsets
        .get(&cb)
        .map(|o| *o)
        .unwrap_or(0)
        .min(u64::from_le_bytes(size));

    let mut offset = 0;
    while offset < end {
        let mut ty = [0u8; 4];
        if !ctx.mem.read(buffer + offset, &mut ty) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        let record = buffer + offset;
        match u32::from_le_bytes(ty) {
            APR_RECORD_READ_FILE => {
                // [0x04]=fileId [0x08]=destination [0x10]=size [0x18]=fileOffset [0x20]=bytesRead
                let mut f = [0u8; 0x28];
                if !ctx.mem.read(record + 4, &mut f) {
                    return SCE_KERNEL_ERROR_EFAULT;
                }
                let file_id = u32::from_le_bytes(f[0..4].try_into().expect("fixed slice"));
                let destination =
                    u64::from_le_bytes(f[0x04..0x0c].try_into().expect("fixed slice"));
                let size = u64::from_le_bytes(f[0x0c..0x14].try_into().expect("fixed slice"));
                let file_offset =
                    u64::from_le_bytes(f[0x14..0x1c].try_into().expect("fixed slice"));
                let read = match ctx.kernel.appr_host_path(file_id).and_then(|host| {
                    let file = std::fs::File::open(&host).ok()?;
                    use std::io::{Read, Seek, SeekFrom};
                    let mut file = std::io::BufReader::new(file);
                    file.seek(SeekFrom::Start(file_offset)).ok()?;
                    let mut buf = vec![0u8; size.min(64 << 20) as usize];
                    let n = file.read(&mut buf).ok()?;
                    buf.truncate(n);
                    Some(buf)
                }) {
                    Some(bytes) => {
                        let n = bytes.len() as u64;
                        if !ctx.mem.write(destination, &bytes) {
                            return SCE_KERNEL_ERROR_EFAULT;
                        }
                        n
                    }
                    None => {
                        // Missing file: zero-fill (SharpEmu's documented behavior
                        // — games queue speculative reads and consume on success).
                        let zeros = [0u8; 4096];
                        let mut written = 0;
                        while written < size {
                            let chunk = (size - written).min(zeros.len() as u64) as usize;
                            if !ctx.mem.write(destination + written, &zeros[..chunk]) {
                                break;
                            }
                            written += chunk as u64;
                        }
                        size
                    }
                };
                if !ctx.mem.write(record + 0x20, &read.to_le_bytes()) {
                    return SCE_KERNEL_ERROR_EFAULT;
                }
                offset += 0x30;
            }
            APR_RECORD_KERNEL_EVENT_QUEUE => {
                // [0x08]=equeue [0x10]=ident [0x18]=userData [0x20]=data
                let mut f = [0u8; 0x28];
                if !ctx.mem.read(record + 8, &mut f) {
                    return SCE_KERNEL_ERROR_EFAULT;
                }
                let equeue = u64::from_le_bytes(f[0..8].try_into().expect("fixed slice"));
                let ident = u64::from_le_bytes(f[8..16].try_into().expect("fixed slice"));
                let user_data = u64::from_le_bytes(f[16..24].try_into().expect("fixed slice"));
                let data = u64::from_le_bytes(f[24..32].try_into().expect("fixed slice"));
                let _ = data;
                if let Some(mut ev) = ctx.kernel.kernel_equeue_events.get_mut(&(equeue, ident)) {
                    ev.triggered = true;
                    ev.udata = user_data;
                    ev.fflags += 1;
                } else {
                    ctx.kernel.kernel_equeue_events.insert(
                        (equeue, ident),
                        xps5x_kernel::EqueueUserEvent {
                            triggered: true,
                            udata: user_data,
                            fflags: 1,
                            ..Default::default()
                        },
                    );
                }
                offset += 0x30;
            }
            APR_RECORD_WRITE_ADDRESS => {
                // [0x08]=address [0x10]=value
                let mut f = [0u8; 0x10];
                if !ctx.mem.read(record + 8, &mut f) {
                    return SCE_KERNEL_ERROR_EFAULT;
                }
                let address = u64::from_le_bytes(f[0..8].try_into().expect("fixed slice"));
                let value = u64::from_le_bytes(f[8..16].try_into().expect("fixed slice"));
                if !ctx.mem.write(address, &value.to_le_bytes()) {
                    return SCE_KERNEL_ERROR_EFAULT;
                }
                offset += 0x20;
            }
            other => {
                warn!("APR command buffer: unknown record type {other} at +{offset:#x}");
                return SCE_KERNEL_ERROR_EINVAL;
            }
        }
    }
    SCE_OK
}

/// `sceKernelAprSubmitCommandBufferAndGetResult(cb, priority, resultAddress,
/// outSubmissionId)` — SharpEmu `KernelAprCompatExports`: record the
/// submission, complete the buffer synchronously, write the id + result.
fn hle_apr_submit_and_get_result(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let priority = args.get(1).copied().unwrap_or(0);
    let result_address = args.get(2).copied().unwrap_or(0);
    let out_submission_id = args.get(3).copied().unwrap_or(0);
    if cb == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let submission_id = ctx.kernel.appr_add_submission(cb);
    let result = apr_complete_command_buffer(ctx, cb);
    if result != SCE_OK {
        return result;
    }
    if out_submission_id != 0
        && !ctx
            .mem
            .write(out_submission_id, &submission_id.to_le_bytes())
    {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if result_address != 0 && !ctx.mem.write(result_address, &0u64.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    debug!(
        "sceKernelAprSubmitCommandBufferAndGetResult(cb={cb:#x}, priority={priority}) -> submission {submission_id}"
    );
    SCE_OK
}

/// `sceKernelAprSubmitCommandBuffer(cb, priority)` — submit without a result.
fn hle_apr_submit(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    ctx.kernel.appr_add_submission(cb);
    apr_complete_command_buffer(ctx, cb)
}

/// `sceKernelAprWaitCommandBuffer(submissionId, ..)`: the synchronous model
/// completed everything at submit time — consume the entry, report OK when
/// it existed, NOT_FOUND otherwise (SharpEmu).
fn hle_apr_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    let submission_id = args.first().copied().unwrap_or(0) as u32;
    if ctx.kernel.appr_submissions.remove(&submission_id).is_some() {
        SCE_OK
    } else {
        SCE_KERNEL_ERROR_ESRCH
    }
}

/// `sceKernelAprResolveFilepathsToIdsAndFileSizes(pathList, count,
/// idsAddress, sizesAddress)` — SharpEmu `KernelMemoryCompatExports` ABI.
/// Each entry in `pathList` is a uint64 pointer to a NUL-terminated path;
/// resolve it through the VFS, write the deterministic FNV-1a id (registered
/// for later AMPR reads) and the file size. A missing file gets id
/// `0xFFFF_FFFF` + size 0 and the batch CONTINUES (a patch/DLC path may be
/// legitimately absent; the caller checks per-file results).
fn hle_apr_resolve_filepaths_to_ids_and_file_sizes(ctx: &HleContext, args: &[u64]) -> u64 {
    let path_list = args.first().copied().unwrap_or(0);
    let count = args.get(1).copied().unwrap_or(0);
    let ids_address = args.get(2).copied().unwrap_or(0);
    let sizes_address = args.get(3).copied().unwrap_or(0);
    if sizes_address == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    apr_resolve_batch(
        ctx,
        None,
        path_list,
        count,
        ids_address,
        sizes_address,
        /* missing_is_error = */ false,
    )
}

/// One path entry's resolution: the registered APR id and the file size, or
/// `None` when the path does not resolve to an existing host file.
fn apr_resolve_one(ctx: &HleContext, guest_path: &str) -> Option<(u32, u64)> {
    let host = ctx.kernel.filesystem.resolve_path(guest_path)?;
    let meta = std::fs::metadata(&host).ok()?;
    let id = ctx
        .kernel
        .appr_register_file(guest_path, host.display().to_string());
    Some((id, meta.len()))
}

/// Shared body of the `sceKernelAprResolveFilepaths*` family.
///
/// `path_list` is an array of `count` `uint64` pointers to NUL-terminated
/// guest paths (SharpEmu `KernelMemoryCompatExports` ABI). Each resolves
/// through the VFS and registers a deterministic FNV-1a id for later AMPR
/// reads. `prefix` (the `WithPrefix` variants) is prepended verbatim to every
/// entry before resolution. Out-arrays are optional (0 = skip that output).
///
/// A missing file either continues the batch with id `0xFFFF_FFFF` + size 0
/// (`missing_is_error == false` — the `AndFileSizes` behavior: patch/DLC
/// files may legitimately be absent and the caller checks per-file results)
/// or aborts with `ENOENT` (`true` — SharpEmu's `ToIds` behavior).
fn apr_resolve_batch(
    ctx: &HleContext,
    prefix: Option<&str>,
    path_list: u64,
    count: u64,
    ids_address: u64,
    sizes_address: u64,
    missing_is_error: bool,
) -> u64 {
    const MAX_PATHS: u64 = 1024;
    if path_list == 0 || count == 0 || count > MAX_PATHS {
        return SCE_KERNEL_ERROR_EINVAL;
    }

    for i in 0..count {
        let mut raw = [0u8; 8];
        if !ctx.mem.read(path_list + i * 8, &mut raw) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        let text_ptr = u64::from_le_bytes(raw);
        let text = crate::fmt::read_cstr(ctx.mem, text_ptr).unwrap_or_default();
        let mut guest_path = String::from_utf8_lossy(&text).into_owned();
        if let Some(prefix) = prefix {
            guest_path = format!("{prefix}{guest_path}");
        }

        let (id, size) = match apr_resolve_one(ctx, &guest_path) {
            Some(resolved) => resolved,
            None if missing_is_error => {
                debug!("AprResolveFilepaths: '{guest_path}' not found — aborting batch (ENOENT)");
                return SCE_KERNEL_ERROR_ENOENT;
            }
            None => (u32::MAX, 0),
        };

        if ids_address != 0 && !ctx.mem.write(ids_address + i * 4, &id.to_le_bytes()) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        if sizes_address != 0 && !ctx.mem.write(sizes_address + i * 8, &size.to_le_bytes()) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
    }
    SCE_OK
}

/// `sceKernelAprResolveFilepathsToIds(pathList, count, ids)` — the ids-only
/// sibling (SharpEmu `KernelMemoryCompatExports`, NID `WT-5NKy42fw`): a
/// missing file aborts the batch with `ENOENT` (unlike the `AndFileSizes`
/// form, which reports per-file misses and continues).
fn hle_apr_resolve_filepaths_to_ids(ctx: &HleContext, args: &[u64]) -> u64 {
    let path_list = args.first().copied().unwrap_or(0);
    let count = args.get(1).copied().unwrap_or(0);
    let ids_address = args.get(2).copied().unwrap_or(0);
    if ids_address == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    apr_resolve_batch(ctx, None, path_list, count, ids_address, 0, true)
}

/// `sceKernelAprResolveFilepathsWithPrefixToIds(prefix, pathList, count,
/// ids)`: the `ToIds` form with a shared path prefix (arg 0) prepended to
/// every entry. Neither reference implements the `WithPrefix` variants
/// (shadPS4 carries only aerolib stubs; SharpEmu omits them), so the
/// argument order is the natural extension of the verified non-prefix ABI:
/// the prefix leads and the remaining arguments shift right by one.
fn hle_apr_resolve_filepaths_with_prefix_to_ids(ctx: &HleContext, args: &[u64]) -> u64 {
    let prefix_ptr = args.first().copied().unwrap_or(0);
    let path_list = args.get(1).copied().unwrap_or(0);
    let count = args.get(2).copied().unwrap_or(0);
    let ids_address = args.get(3).copied().unwrap_or(0);
    if prefix_ptr == 0 || ids_address == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let Some(prefix) = crate::fmt::read_cstr(ctx.mem, prefix_ptr) else {
        return SCE_KERNEL_ERROR_EFAULT;
    };
    let prefix = String::from_utf8_lossy(&prefix).into_owned();
    apr_resolve_batch(ctx, Some(&prefix), path_list, count, ids_address, 0, true)
}

/// `sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes(prefix, pathList,
/// count, ids, sizes)` — see [`hle_apr_resolve_filepaths_with_prefix_to_ids`]
/// for the (inferred) prefix ABI; per-file misses continue the batch like the
/// verified non-prefix `AndFileSizes` form.
fn hle_apr_resolve_filepaths_with_prefix_to_ids_and_file_sizes(
    ctx: &HleContext,
    args: &[u64],
) -> u64 {
    let prefix_ptr = args.first().copied().unwrap_or(0);
    let path_list = args.get(1).copied().unwrap_or(0);
    let count = args.get(2).copied().unwrap_or(0);
    let ids_address = args.get(3).copied().unwrap_or(0);
    let sizes_address = args.get(4).copied().unwrap_or(0);
    if prefix_ptr == 0 || sizes_address == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let Some(prefix) = crate::fmt::read_cstr(ctx.mem, prefix_ptr) else {
        return SCE_KERNEL_ERROR_EFAULT;
    };
    let prefix = String::from_utf8_lossy(&prefix).into_owned();
    apr_resolve_batch(
        ctx,
        Some(&prefix),
        path_list,
        count,
        ids_address,
        sizes_address,
        false,
    )
}

/// The `*ForEach` resolve variants (`ToIdsForEach`,
/// `ToIdsAndFileSizesForEach`, `WithPrefixToIdsForEach`,
/// `WithPrefixToIdsAndFileSizesForEach`).
///
/// **Honest partial:** no reference implements these (shadPS4 has only
/// aerolib stubs; SharpEmu omits them), so beyond `pathList`/`count` — which
/// the non-`ForEach` forms fix — the trailing argument layout is unverified:
/// `ForEach` strongly suggests a per-file guest callback, and XPS5X must not
/// call back into the guest here nor write through register slots that may
/// hold a code pointer. So this resolves and REGISTERS every path in the APR
/// id table (the durable side effect later `sceAmprApr*ReadFile` calls
/// depend on), writes nothing, warns loudly once per entry point, and
/// reports success.
fn apr_resolve_foreach(
    ctx: &HleContext,
    args: &[u64],
    name: &'static str,
    with_prefix: bool,
) -> u64 {
    let (prefix, path_list, count) = if with_prefix {
        let prefix_ptr = args.first().copied().unwrap_or(0);
        let prefix = crate::fmt::read_cstr(ctx.mem, prefix_ptr)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
        (
            prefix,
            args.get(1).copied().unwrap_or(0),
            args.get(2).copied().unwrap_or(0),
        )
    } else {
        (
            None,
            args.first().copied().unwrap_or(0),
            args.get(1).copied().unwrap_or(0),
        )
    };

    static WARNED: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    if let Ok(mut warned) = WARNED.lock()
        && !warned.contains(&name)
    {
        warned.push(name);
        warn!(
            "{name}: ForEach ABI unverified (no reference implementation) — registering \
             {count} path(s) in the APR id table, invoking NO guest callback and writing \
             NO out-array; later AMPR reads by id still resolve"
        );
    }

    if path_list == 0 || count == 0 || count > 1024 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    for i in 0..count {
        let mut raw = [0u8; 8];
        if !ctx.mem.read(path_list + i * 8, &mut raw) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        let text = crate::fmt::read_cstr(ctx.mem, u64::from_le_bytes(raw)).unwrap_or_default();
        let mut guest_path = String::from_utf8_lossy(&text).into_owned();
        if let Some(prefix) = &prefix {
            guest_path = format!("{prefix}{guest_path}");
        }
        let _ = apr_resolve_one(ctx, &guest_path);
    }
    SCE_OK
}

fn hle_apr_resolve_filepaths_to_ids_foreach(ctx: &HleContext, args: &[u64]) -> u64 {
    apr_resolve_foreach(ctx, args, "sceKernelAprResolveFilepathsToIdsForEach", false)
}

fn hle_apr_resolve_filepaths_to_ids_and_file_sizes_foreach(ctx: &HleContext, args: &[u64]) -> u64 {
    apr_resolve_foreach(
        ctx,
        args,
        "sceKernelAprResolveFilepathsToIdsAndFileSizesForEach",
        false,
    )
}

fn hle_apr_resolve_filepaths_with_prefix_to_ids_foreach(ctx: &HleContext, args: &[u64]) -> u64 {
    apr_resolve_foreach(
        ctx,
        args,
        "sceKernelAprResolveFilepathsWithPrefixToIdsForEach",
        true,
    )
}

fn hle_apr_resolve_filepaths_with_prefix_to_ids_and_file_sizes_foreach(
    ctx: &HleContext,
    args: &[u64],
) -> u64 {
    apr_resolve_foreach(
        ctx,
        args,
        "sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizesForEach",
        true,
    )
}

/// `sceKernelAprSubmitCommandBufferAndGetId(cb, priority, outSubmissionId)` —
/// SharpEmu `KernelAprCompatExports` (NID `qvMUCyyaCSI`): like
/// `SubmitCommandBufferAndGetResult` but the third argument is the submission
/// id out-pointer (required) and there is no result address.
fn hle_apr_submit_and_get_id(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let out_submission_id = args.get(2).copied().unwrap_or(0);
    if cb == 0 || out_submission_id == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let submission_id = ctx.kernel.appr_add_submission(cb);
    let result = apr_complete_command_buffer(ctx, cb);
    if result != SCE_OK {
        return result;
    }
    if !ctx
        .mem
        .write(out_submission_id, &submission_id.to_le_bytes())
    {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// `sceKernelAprGetFileStat(id, SceKernelStat *stat)` — SharpEmu
/// `KernelMemoryCompatExports` (NID `ApkYaHb8Sek`): stat an APR file by the
/// id `AprResolveFilepaths*` registered, writing the standard 120-byte Orbis
/// stat. An unregistered id or a vanished host file is `ENOENT` (SharpEmu
/// observed Void Terrarium null-dereferencing when this was missing).
fn hle_apr_get_file_stat(ctx: &HleContext, args: &[u64]) -> u64 {
    let file_id = args.first().copied().unwrap_or(0) as u32;
    let stat_out = args.get(1).copied().unwrap_or(0);
    if stat_out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let Some(host) = ctx.kernel.appr_host_path(file_id) else {
        debug!("sceKernelAprGetFileStat(id={file_id:#x}): id not registered — ENOENT");
        return SCE_KERNEL_ERROR_ENOENT;
    };
    match std::fs::metadata(&host) {
        Ok(metadata) => write_orbis_stat(ctx, stat_out, &metadata),
        Err(_) => SCE_KERNEL_ERROR_ENOENT,
    }
}

/// `sceKernelAprGetFileSize(id, uint64_t *size)`: report the real VFS file
/// size for a registered APR id.
///
/// The exact signature is not reversed anywhere public — SharpEmu returns a
/// bare success stub (NID `WvEu7yl3Ivg`, "argument layout is unknown") — so
/// this takes the layout every sibling APR call uses (id first, out-pointer
/// second) and fails EINVAL/ENOENT rather than guessing further.
fn hle_apr_get_file_size(ctx: &HleContext, args: &[u64]) -> u64 {
    let file_id = args.first().copied().unwrap_or(0) as u32;
    let size_out = args.get(1).copied().unwrap_or(0);
    if size_out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let Some(host) = ctx.kernel.appr_host_path(file_id) else {
        debug!("sceKernelAprGetFileSize(id={file_id:#x}): id not registered — ENOENT");
        return SCE_KERNEL_ERROR_ENOENT;
    };
    match std::fs::metadata(&host) {
        Ok(metadata) => {
            if !ctx.mem.write(size_out, &metadata.len().to_le_bytes()) {
                return SCE_KERNEL_ERROR_EFAULT;
            }
            SCE_OK
        }
        Err(_) => SCE_KERNEL_ERROR_ENOENT,
    }
}

/// Common Gen5 directory enumeration path. `sceKernelGetdirentries` supplies
/// an optional fourth `basep` argument; `sceKernelGetdents` does not.
fn hle_getdents(ctx: &HleContext, args: &[u64], has_basep: bool) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let buffer = args.get(1).copied().unwrap_or(0);
    let requested = args.get(2).copied().unwrap_or(0).min(READ_MAX_BYTES);
    // `sceKernelGetdents`/POSIX `getdents` have only three arguments. RCX is
    // therefore caller-clobbered garbage at the HLE trap and must never be
    // interpreted as an output pointer. Only the four-argument
    // `getdirentries` spellings own `basep`.
    let basep = if has_basep {
        args.get(3).copied().unwrap_or(0)
    } else {
        0
    };
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
        warn!(
            "getdents: buffer write failed — buffer={buffer:#x} len={} (fd={fd})",
            payload.len()
        );
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
/// 1. If a preplaced real module matches, schedules its `DT_INIT` once and
///    returns its existing handle. Repeated loads keep the same handle.
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
/// `pRes` is initialized to zero; deferred guest callbacks currently do not
/// copy the eventual initializer return value back into it.
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
            if !info.initialized {
                if let Some(entry) = info.entry_point {
                    let requested = ctx.guest_calls.request(GuestCallRequest {
                        entry,
                        args: [
                            args.get(1).copied().unwrap_or(0),
                            args.get(2).copied().unwrap_or(0),
                            0,
                            0,
                            0,
                            0,
                        ],
                        completion: None,
                    });
                    if !requested {
                        return 0x8002_000B; // SCE_KERNEL_ERROR_EAGAIN
                    }
                    tracing::info!(
                        "sceKernelLoadStartModule: scheduling '{}' DT_INIT at {entry:#x}",
                        info.name
                    );
                } else {
                    tracing::info!("sceKernelLoadStartModule: '{}' has no DT_INIT", info.name);
                }
                ctx.kernel.mark_module_initialized(info.id);
            }
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

/// `sceKernelDlsym(handle, symbol, addrOut)`: resolve exports from a real,
/// preplaced LLE module. Such exports are already directly executable guest
/// addresses and do not need a newly minted HLE trampoline.
fn hle_dlsym(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let sym_ptr = args.get(1).copied().unwrap_or(0);
    let addr_out = args.get(2).copied().unwrap_or(0);
    let Some(symbol_bytes) = crate::fmt::read_cstr(ctx.mem, sym_ptr) else {
        return SCE_KERNEL_ERROR_EFAULT;
    };
    if addr_out == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let Ok(handle) = u32::try_from(handle) else {
        return SCE_KERNEL_ERROR_ESRCH;
    };

    use sha1::{Digest, Sha1};
    const SCE_NID_SALT: [u8; 16] = [
        0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1, 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52,
        0x30,
    ];
    let mut hasher = Sha1::new();
    hasher.update(&symbol_bytes);
    hasher.update(SCE_NID_SALT);
    let digest = hasher.finalize();
    let nid = u64::from_le_bytes(digest[..8].try_into().expect("SHA-1 has 20 bytes"));
    let symbol = String::from_utf8_lossy(&symbol_bytes);
    let Some(addr) = ctx.kernel.resolve_lle_export(handle, nid) else {
        // Say which of the two very different bugs this is. A handle with no
        // exports at all was never wired up; a handle with many means the symbol
        // genuinely is not in that module's export table — and an ENOENT alone
        // cannot tell them apart, which is exactly how this failure was misread
        // as a memory bug for two sessions.
        match ctx.kernel.lle_export_count(handle) {
            Some(count) => warn!(
                "sceKernelDlsym(handle={handle}, symbol='{symbol}', nid={nid:#018x}): not among \
                 that module's {count} export(s) — ENOENT"
            ),
            None => warn!(
                "sceKernelDlsym(handle={handle}, symbol='{symbol}'): handle names NO registered \
                 module — ENOENT"
            ),
        }
        return SCE_KERNEL_ERROR_ENOENT;
    };
    if !ctx.mem.write(addr_out, &addr.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    debug!("sceKernelDlsym(handle={handle}, symbol='{symbol}') -> {addr:#x}");
    SCE_OK
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
    // BatchMap2 is BatchMap plus a trailing flags argument the batch semantics
    // already imply (fixed-address); one handler serves both.
    registry.register("libkernel", "sceKernelBatchMap", hle_batch_map);
    registry.register("libkernel", "sceKernelBatchMap2", hle_batch_map);

    // -- File descriptors / console I/O (M1-C) --
    registry.register("libkernel", "write", hle_posix_write);
    registry.register("libkernel", "sceKernelWrite", hle_sce_write);

    // Diagnostic C++ ABI trap — only used when the linker force-routes this
    // NID (XPS5X_TRAP_CXA_THROW); otherwise the shipped libc's real
    // __cxa_throw is used. Registered by NID (for import redirection) AND by
    // name (so a runtime-patched jump into an appended trampoline dispatches
    // here via the VEH's name-based hle.call).
    registry.register_nid("libc", "__cxa_throw", 0xbe4b_ae2d_f867_4992, hle_cxa_throw);
    registry.register("libc", "__cxa_throw", hle_cxa_throw);

    // -- File I/O (real, VFS-backed) --
    registry.register("libkernel", "open", hle_posix_open);
    registry.register("libkernel", "sceKernelOpen", hle_sce_open);
    registry.register("libkernel", "read", hle_posix_read);
    registry.register("libkernel", "sceKernelRead", hle_sce_read);
    registry.register("libkernel", "close", hle_posix_close);
    registry.register("libkernel", "sceKernelClose", hle_sce_close);
    // The same POSIX fd calls under the title's other provider — the measured
    // Minecraft eboot imports open/read/write/close/lseek naming libScePosix
    // (a NID hashes the name alone, so both providers see identical NIDs and
    // only the provider-aware registration differs). Same class of alias as
    // `libScePosix::getdents` below.
    registry.register("libScePosix", "open", hle_posix_open);
    registry.register("libScePosix", "read", hle_posix_read);
    registry.register("libScePosix", "write", hle_posix_write);
    registry.register("libScePosix", "close", hle_posix_close);
    registry.register("libScePosix", "lseek", hle_posix_lseek);
    registry.register_nid(
        "libkernel",
        "sceKernelFsync",
        0x7d3c_7aea_5e62_5880,
        hle_sce_fsync,
    );
    registry.register("libkernel", "sceKernelGetdents", hle_sce_getdents);
    registry.register("libkernel", "sceKernelGetdirentries", hle_sce_getdirentries);
    registry.register("libkernel", "getdirentries", hle_posix_getdirentries);
    registry.register("libScePosix", "getdents", hle_posix_getdents);
    registry.register("libkernel", "lseek", hle_posix_lseek);
    registry.register("libkernel", "sceKernelLseek", hle_sce_lseek);
    // pread: positional read. Registered under both the SCE and POSIX names
    // and both provider libraries (resolution is provider-aware — see the
    // clock_gettime lesson: a symbol only registered under one library is
    // unresolved for a title importing it from the other).
    registry.register("libkernel", "sceKernelPread", hle_sce_pread);
    registry.register("libkernel", "pread", hle_posix_pread);
    registry.register("libScePosix", "pread", hle_posix_pread);
    registry.register(
        "libkernel",
        "sceKernelAioInitializeParam",
        hle_aio_initialize_param,
    );
    registry.register(
        "libkernel",
        "sceKernelAioInitializeImpl",
        hle_aio_initialize_impl,
    );
    registry.register(
        "libkernel",
        "sceKernelAprResolveFilepathsToIdsAndFileSizes",
        hle_apr_resolve_filepaths_to_ids_and_file_sizes,
    );
    registry.register(
        "libkernel",
        "sceKernelAprSubmitCommandBufferAndGetResult",
        hle_apr_submit_and_get_result,
    );
    registry.register(
        "libkernel",
        "sceKernelAprSubmitCommandBuffer",
        hle_apr_submit,
    );
    registry.register("libkernel", "sceKernelAprWaitCommandBuffer", hle_apr_wait);
    registry.register(
        "libkernel",
        "sceKernelAprSubmitCommandBufferAndGetId",
        hle_apr_submit_and_get_id,
    );
    registry.register(
        "libkernel",
        "sceKernelAprResolveFilepathsToIds",
        hle_apr_resolve_filepaths_to_ids,
    );
    registry.register(
        "libkernel",
        "sceKernelAprResolveFilepathsWithPrefixToIds",
        hle_apr_resolve_filepaths_with_prefix_to_ids,
    );
    registry.register(
        "libkernel",
        "sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes",
        hle_apr_resolve_filepaths_with_prefix_to_ids_and_file_sizes,
    );
    registry.register(
        "libkernel",
        "sceKernelAprResolveFilepathsToIdsForEach",
        hle_apr_resolve_filepaths_to_ids_foreach,
    );
    registry.register(
        "libkernel",
        "sceKernelAprResolveFilepathsToIdsAndFileSizesForEach",
        hle_apr_resolve_filepaths_to_ids_and_file_sizes_foreach,
    );
    registry.register(
        "libkernel",
        "sceKernelAprResolveFilepathsWithPrefixToIdsForEach",
        hle_apr_resolve_filepaths_with_prefix_to_ids_foreach,
    );
    registry.register(
        "libkernel",
        "sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizesForEach",
        hle_apr_resolve_filepaths_with_prefix_to_ids_and_file_sizes_foreach,
    );
    registry.register(
        "libkernel",
        "sceKernelAprGetFileStat",
        hle_apr_get_file_stat,
    );
    registry.register(
        "libkernel",
        "sceKernelAprGetFileSize",
        hle_apr_get_file_size,
    );

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
        "sceKernelAvailableDirectMemorySize",
        hle_available_direct_memory_size,
    );
    // PS5 Pro ("Trinity") query — the libkernel twin of libSceAgc's
    // `sceAgcGetIsTrinityMode`. Base PS5, so false. Provider-aware resolution
    // keys on the importing LIBRARY, so the same answer is registered under
    // `libSceAgc` too (NID 0x05f0436466ed8bb0, name recovered via the SharpEmu
    // catalogue merge; appeared as an unnamed unresolved import in a real run).
    registry.register("libkernel", "sceKernelIsTrinityMode", hle_is_trinity_mode);
    registry.register("libSceAgc", "sceAgcGetIsTrinityMode", hle_is_trinity_mode);
    // Address-parking primitives (futex-like): a thread waits until another
    // writes the watched word and wakes it. XPS5X has no true parking lot here,
    // so `Wait` returns after a short slice as a PERMITTED SPURIOUS WAKEUP (the
    // same model as scePthreadCondWait, pthread_cond.rs) — the caller re-checks
    // its condition and either proceeds or waits again, so no deadlock and no
    // tight spin. `Wake` reports success. Names recovered via the catalogue
    // merge (Wake = 0xab6cbfc032155990); both appeared unnamed in a real run.
    registry.register(
        "libkernel",
        "sceKernelSyncOnAddressWait",
        hle_sync_on_address_wait,
    );
    registry.register(
        "libkernel",
        "sceKernelSyncOnAddressWake",
        hle_sync_on_address_wake,
    );
    registry.register(
        "libkernel",
        "sceKernelSetVirtualRangeName",
        hle_set_virtual_range_name,
    );
    registry.register(
        "libkernel",
        "sceKernelClearVirtualRangeName",
        hle_clear_virtual_range_name,
    );
    registry.register(
        "libkernel",
        "sceKernelDirectMemoryQuery",
        hle_direct_memory_query,
    );
    registry.register(
        "libkernel",
        "sceKernelConfiguredFlexibleMemorySize",
        hle_configured_flexible_memory_size,
    );
    // The console's "open PS id" — a libkernel export the title imports from
    // the wrapper library `libSceOpenPsId` (provider-aware resolution keys on
    // the importing library, so both spellings are registered).
    registry.register("libSceOpenPsId", "sceKernelGetOpenPsId", hle_get_open_ps_id);
    registry.register("libkernel", "sceKernelGetOpenPsId", hle_get_open_ps_id);

    // -- Thread / sync --
    registry.register("libkernel", "scePthreadCreate", hle_pthread_create);
    registry.register(
        "libScePosix",
        "pthread_create_name_np",
        hle_pthread_create_name_np,
    );
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
    // scePthreadCond* are registered by the `pthread_cond` module (real state
    // machine: the wait releases and reacquires the guest mutex around a
    // generation-counted sleep) — see xps5x_hle::pthread_cond.
    //
    // They used to ALSO be registered here as no-op stubs that returned success
    // without waiting or releasing the mutex. Those only ever lost the race
    // because `pthread_cond::register` runs after this function and
    // `HleRegistry::register` is last-write-wins — reordering the two calls
    // would have silently given every title a `cond_wait` that never releases
    // its mutex, deadlocking any guest thread pool. Removed rather than left as
    // shadowed dead code.
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
    registry.register("libkernel", "_open", hle_posix_open);
    registry.register("libkernel", "_read", hle_posix_read);
    registry.register("libkernel", "_write", hle_posix_write);
    registry.register("libkernel", "_close", hle_posix_close);
    // `_exit` terminates the process: the runtime's exit family intercepts it
    // before dispatch (see xps5x_runtime::dispatch::TERMINATING_FUNCTIONS);
    // this registration exists so the import resolves to a trampoline.
    registry.register("libkernel", "_exit", hle_pthread_exit);
    registry.register("libkernel", "nanosleep", hle_nanosleep);
    // libkernel's real export of the same call under the underscore spelling
    // (a distinct NID — a NID hashes the name alone), and the sce spelling
    // (`sceKernelNanosleep`, same req/rem timespec ABI, measured ASTRO.BOT).
    registry.register("libkernel", "_nanosleep", hle_nanosleep);
    registry.register("libkernel", "sceKernelNanosleep", hle_nanosleep);
    registry.register("libkernel", "getrusage", hle_getrusage);
    registry.register("libkernel", "signal", hle_posix_signal);
    registry.register("libkernel", "sceKernelMlock", hle_mlock);
    registry.register(
        "libkernel",
        "sceKernelMapDirectMemory2",
        hle_map_direct_memory2,
    );
    registry.register(
        "libkernel",
        "sceKernelInternalMemoryGetModuleSegmentInfo",
        hle_internal_get_module_segment_info,
    );
    // POSIX memory spellings the measured title imports from libScePosix.
    registry.register("libScePosix", "mprotect", hle_posix_mprotect);
    registry.register("libScePosix", "munmap", hle_munmap);
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
    registry.register("libkernel", "sceKernelMprotect", hle_kernel_mprotect);
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
        hle_get_module_info_for_unwind,
    );
    registry.register(
        "libkernel",
        "sceKernelGetModuleInfoFromAddr",
        hle_module_info_unavailable,
    );
    registry.register("libkernel", "sceKernelVirtualQuery", hle_virtual_query);
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
    // The public (non-`Internal`) spelling was missing: measured, Until Dawn
    // imports `sceKernelMapNamedFlexibleMemory` (NID 0x98bf0d0c7f3a8902) and
    // stopped its boot there even though the mapping behaviour was implemented.
    registry.register(
        "libkernel",
        "sceKernelMapNamedFlexibleMemory",
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
    registry.register(
        "libkernel",
        "sceKernelDebugWriteCppExceptionInfo",
        hle_debug_write_cpp_exception_info,
    );
    // Filesystem metadata. Path-based operations resolve through the same VFS
    // mounts as open/read/write; no title-specific path handling lives here.
    registry.register("libkernel", "sceKernelMkdir", hle_mkdir);
    registry.register("libkernel", "sceKernelUnlink", hle_sce_unlink);
    registry.register("libkernel", "sceKernelRmdir", hle_sce_rmdir);
    registry.register("libkernel", "unlink", hle_posix_unlink);
    registry.register("libkernel", "rmdir", hle_posix_rmdir);
    registry.register("libkernel", "sceKernelRename", hle_sce_rename);
    registry.register("libkernel", "sceKernelTruncate", hle_sce_truncate);
    registry.register("libkernel", "sceKernelSync", hle_sce_sync);
    registry.register("libkernel", "sceKernelChmod", hle_path_metadata_accept);
    registry.register("libkernel", "sceKernelUtimes", hle_path_metadata_accept);
    registry.register("libkernel", "sceKernelStat", hle_stat);
    registry.register("libkernel", "sceKernelFstat", hle_fstat);
    // pthread surface libc/fmod touch during init — attr/priority/affinity
    // bookkeeping has no scheduler to talk to yet, so recording nothing and
    // returning success is faithful enough for a single-thread world.
    registry.register("libkernel", "scePthreadDetach", hle_pthread_detach);
    // scePthreadSetprio/Getprio are registered by `pthread_thread` (real
    // priority bookkeeping) — the old hle_ok_stub here dropped the value.
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
    registry.register("libkernel", "sceKernelClockGetres", hle_clock_getres);
    registry.register(
        "libkernel",
        "sceKernelGetTscFrequency",
        hle_get_tsc_frequency,
    );
    registry.register("libkernel", "sceKernelReadTsc", hle_read_tsc);
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
    registry.register("libkernel", "sceKernelGetGPI", hle_get_gpi);
}

/// `sceKernelGetGPI()`: read the General Purpose **Input** lines.
///
/// These are physical DIP switches that exist only on development kits; on a
/// retail console there is nothing wired to them and the read is defined to
/// come back zero. Returning 0 is therefore the **real retail behavior**, not a
/// placeholder — cross-checked against shadPS4 (GPL-2.0-or-later), whose
/// `sceKernelGetGPI` is likewise `return ORBIS_OK` under the comment "stubbed
/// on non-devkit consoles" (`core/libraries/kernel/kernel.cpp:231`).
///
/// Measured: this is ASTRO.BOT's first hard stop (NID `0xe285d87bd5e69344`,
/// encoded `4oXYe9Xmk0Q`), reached from its allocator-init path.
fn hle_get_gpi(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetGPI() -> 0 (no devkit GPI switches on retail)");
    SCE_OK
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
/// The direct-memory budget reported to and enforced on titles: ~13.375 GiB,
/// the commonly-measured game-usable direct memory of a retail PS5. Titles
/// size their pools by allocating until the kernel refuses, so both the
/// refusal and the reported size must model the console, not the host.
pub(crate) const PS5_DIRECT_MEMORY_SIZE: u64 = 0x3_5800_0000;

/// `SCE_KERNEL_ERROR_EAGAIN` — what the real allocator returns when the
/// direct-memory budget cannot satisfy the request.
const SCE_KERNEL_ERROR_EAGAIN_ALLOC: u64 = 0x8002_000B;

/// `ENOMEM` — no block satisfies the request (`sceKernelAvailableDirectMemorySize`).
const SCE_KERNEL_ERROR_ENOMEM: u64 = 0x8002_000C;

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
    // Enforce the console's direct-memory budget BEFORE touching the host
    // arena. A real PS5 refuses once the budget is spent, and titles rely on
    // that refusal to discover how much memory exists: Dragon Ball allocates
    // 1 GiB in a loop until ENOMEM and sizes its pools from the total. With no
    // budget that loop "succeeded" ~900 times, consumed the entire host mapping
    // window, and then died on placement instead of ending normally.
    {
        use std::sync::atomic::Ordering;
        let mut current = ctx.kernel.direct_memory_allocated.load(Ordering::Relaxed);
        loop {
            let Some(next) = current
                .checked_add(len)
                .filter(|n| *n <= PS5_DIRECT_MEMORY_SIZE)
            else {
                debug!(
                    "sceKernelAllocateDirectMemory: budget exhausted \
                     (allocated={current:#x} + len={len:#x} > {PS5_DIRECT_MEMORY_SIZE:#x}) — EAGAIN"
                );
                return SCE_KERNEL_ERROR_EAGAIN_ALLOC;
            };
            match ctx.kernel.direct_memory_allocated.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(seen) => current = seen,
            }
        }
    }
    let Some(addr) = ctx.alloc.mmap(len, alignment) else {
        warn!("sceKernelAllocateDirectMemory: arena mmap failed (len={len:#x})");
        ctx.kernel
            .direct_memory_allocated
            .fetch_sub(len, std::sync::atomic::Ordering::Relaxed);
        return HLE_ERROR;
    };
    ctx.kernel.memory.record_mapping(addr, len, DEFAULT_PROT);

    if !ctx.mem.write(phys_addr_out, &addr.to_le_bytes()) {
        warn!("sceKernelAllocateDirectMemory: physAddrOut {phys_addr_out:#x} out of bounds");
        ctx.kernel.memory.remove_mapping(addr);
        return HLE_ERROR;
    }
    if std::env::var_os("XPS5X_TRACE_DIRECT_MEMORY").is_some() {
        warn!(
            "direct-memory trace: allocate len={len:#x} alignment={alignment:#x} type={memory_type} -> phys={addr:#x}"
        );
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
    if std::env::var_os("XPS5X_TRACE_DIRECT_MEMORY").is_some() {
        warn!("direct-memory trace: release phys={start:#x} len={len:#x}");
    }
    // `start` is a physical-memory OFFSET; 0 is a perfectly valid one (a title
    // whose direct-memory pool begins at physical 0 releases [0, len)). Only a
    // zero length is invalid. Rejecting start==0 returned SCE EINVAL, and the
    // title's C++ direct-memory RAII wrapper turned that into an uncaught
    // std::system_error("invalid argument") that killed its Streaming Pool /
    // Rendering Pool workers. Release is best-effort/idempotent: freeing an
    // untracked range is a no-op that still reports success.
    if len == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    ctx.alloc.munmap(start, len);
    ctx.kernel.memory.remove_mapping(start);
    // Return the bytes to the direct-memory budget. Saturating: release is
    // best-effort/idempotent (see above), so an untracked or double release
    // must not underflow the counter.
    let _ = ctx.kernel.direct_memory_allocated.fetch_update(
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
        |v| Some(v.saturating_sub(len)),
    );
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
    let mut requested_bytes = [0u8; 8];
    if !ctx.mem.read(addr_out, &mut requested_bytes) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let requested = u64::from_le_bytes(requested_bytes);
    let page_size = xps5x_core::PS5_PAGE_SIZE as u64;
    if alignment != 0 && !alignment.is_power_of_two() {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let alignment = if alignment == 0 {
        page_size
    } else {
        alignment.max(page_size)
    };
    // Two cases, and getting either one wrong silently corrupts a title.
    //
    // A guest-supplied address is NOT advisory in practice. Minecraft asks for
    // a specific address and then writes to *that* address without ever reading
    // the out-param back, so publishing somewhere else leaves it scribbling on
    // unmapped memory. Honor the request.
    //
    // With no requested address there is nothing to honor, and the answer is
    // NOT a fresh region: `sceKernelAllocateDirectMemory` hands out an address
    // `ctx.alloc.mmap` already committed, so the direct memory is live arena
    // memory at `direct_memory_start`. Publishing that address keeps the
    // mapping and the direct memory the same storage; a fresh region would
    // detach the guest's writes from its own direct memory and leak when
    // `sceKernelReleaseDirectMemory` freed the physical range.
    //
    // GAP: when an address IS requested, the mapping is backed by its own
    // memory rather than aliasing `direct_memory_start`. True aliasing needs
    // file-backed sections; reservations cannot do it. Harmless while a title
    // reaches direct memory only through the mapping it asked for (all four
    // measured titles do), wrong the day one writes via the mapping and reads
    // via the physical address.
    let mapped = if requested == 0 {
        if direct_memory_start % alignment != 0 {
            warn!(
                "sceKernelMapDirectMemory: phys {direct_memory_start:#x} does not satisfy \
                 alignment {alignment:#x}"
            );
            return SCE_KERNEL_ERROR_EINVAL;
        }
        Some(direct_memory_start)
    } else {
        ctx.alloc.map_at(requested, len, alignment)
    };
    let Some(mapped) = mapped else {
        warn!("sceKernelMapDirectMemory: cannot map len={len:#x} at requested={requested:#x}");
        return HLE_ERROR;
    };
    ctx.kernel.memory.record_mapping(mapped, len, prot);
    if !ctx.mem.write(addr_out, &mapped.to_le_bytes()) {
        if requested != 0 {
            ctx.alloc.munmap(mapped, len);
        }
        ctx.kernel.memory.remove_mapping(mapped);
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if std::env::var_os("XPS5X_TRACE_DIRECT_MEMORY").is_some() {
        warn!(
            "direct-memory trace: map phys={direct_memory_start:#x} len={len:#x} requested={requested:#x} alignment={alignment:#x} -> {mapped:#x}"
        );
    }
    SCE_OK
}

/// `sceKernelBatchMap(entries, numEntries, numEntriesOut)` /
/// `sceKernelBatchMap2(..., flags)` — perform a batch of map/unmap/protect
/// operations in one call.
///
/// Both Dragon Ball and Until Dawn fault on exactly this import (nid
/// 0xd92284c7a6d2abfe) right after sizing their memory pools — Unreal's PS5
/// allocator carves its reserved VA range into pages with batched fixed-address
/// direct-memory maps. Entry layout (32 bytes, cross-checked against shadPS4's
/// `OrbisKernelBatchMapEntry` and the OpenOrbis headers):
/// `{ start: u64, offset: u64 (phys for MAP_DIRECT), length: u64,
///    protection: u8, type: u8, pad: u16, operation: u32 }`.
///
/// `numEntriesOut` is updated as entries complete, so on error the title knows
/// how many succeeded (the real kernel does the same).
fn hle_batch_map(ctx: &HleContext, args: &[u64]) -> u64 {
    const OP_MAP_DIRECT: u32 = 0;
    const OP_UNMAP: u32 = 1;
    const OP_PROTECT: u32 = 2;
    const OP_MAP_FLEXIBLE: u32 = 3;
    const OP_TYPE_PROTECT: u32 = 4;
    const ENTRY_SIZE: u64 = 32;

    let entries = args.first().copied().unwrap_or(0);
    let num = args.get(1).copied().unwrap_or(0) as u32;
    let num_out = args.get(2).copied().unwrap_or(0);
    debug!("sceKernelBatchMap(entries={entries:#x}, num={num}, numOut={num_out:#x})");
    if entries == 0 || num == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let page = xps5x_core::PS5_PAGE_SIZE as u64;
    let mut done: u32 = 0;
    let mut status = SCE_OK;
    for i in 0..num {
        let mut e = [0u8; ENTRY_SIZE as usize];
        if !ctx.mem.read(entries + u64::from(i) * ENTRY_SIZE, &mut e) {
            status = SCE_KERNEL_ERROR_EFAULT;
            break;
        }
        let start = u64::from_le_bytes(e[0x00..0x08].try_into().unwrap());
        let offset = u64::from_le_bytes(e[0x08..0x10].try_into().unwrap());
        let length = u64::from_le_bytes(e[0x10..0x18].try_into().unwrap());
        let prot = u32::from(e[0x18]);
        let operation = u32::from_le_bytes(e[0x1c..0x20].try_into().unwrap());
        if std::env::var_os("XPS5X_TRACE_DIRECT_MEMORY").is_some() {
            warn!(
                "batch-map[{i}]: op={operation} start={start:#x} offset={offset:#x} \
                 len={length:#x} prot={prot:#x}"
            );
        }
        let ok = match operation {
            OP_MAP_DIRECT | OP_MAP_FLEXIBLE => {
                if length == 0 {
                    false
                } else if start != 0 {
                    // Batch maps are fixed-address: the title owns the layout
                    // (typically inside its own reserved range) and will use
                    // `start` directly.
                    let mapped = ctx.alloc.map_at(start, length, page).is_some();
                    if mapped {
                        ctx.kernel.memory.record_mapping(start, length, prot);
                    }
                    mapped
                } else if operation == OP_MAP_DIRECT && offset != 0 {
                    // No fixed address: direct memory is live arena storage at
                    // its physical offset (see hle_map_direct_memory) — the
                    // mapping IS that storage.
                    ctx.kernel.memory.record_mapping(offset, length, prot);
                    true
                } else {
                    ctx.alloc.mmap(length, page).is_some_and(|addr| {
                        ctx.kernel.memory.record_mapping(addr, length, prot);
                        true
                    })
                }
            }
            OP_UNMAP => {
                ctx.alloc.munmap(start, length);
                ctx.kernel.memory.remove_mapping(start);
                true
            }
            // Apply the protection under XPS5X_ENFORCE_MPROTECT; a no-op
            // otherwise (the arena default), matching the standalone
            // sceKernelMprotect path.
            OP_PROTECT | OP_TYPE_PROTECT => ctx.mem.protect(start, length, prot),
            other => {
                warn!("sceKernelBatchMap: unknown operation {other} at entry {i}");
                false
            }
        };
        if !ok {
            status = SCE_KERNEL_ERROR_EINVAL;
            break;
        }
        done += 1;
    }
    if num_out != 0 && !ctx.mem.write(num_out, &done.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    status
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
    // Must agree with the allocator's enforced budget: a title that reads this
    // then allocates it expects the allocations to succeed, and one that
    // allocates-until-refused expects the total to be about this.
    PS5_DIRECT_MEMORY_SIZE
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

/// Real `sceKernelAvailableDirectMemorySize(off_t searchStart, off_t searchEnd,
/// size_t alignment, off_t *physAddrOut, size_t *sizeOut)`: report the largest
/// free direct-memory block inside `[searchStart, searchEnd)` honouring
/// `alignment`, writing its physical base and size through the two out-params
/// (shadPS4 `memory.cpp:120`, NID `C0f7TJcbfac`).
///
/// XPS5X models direct memory as one contiguous pool
/// ([`PS5_DIRECT_MEMORY_SIZE`]) rather than a fragmenting physical allocator,
/// so the answer is the caller's own window clamped to that pool and aligned
/// up. That is honest for its purpose — callers use this to size an allocation
/// they are about to make — and never over-reports the pool.
///
/// A null out-param is `EINVAL`; an empty window is `ENOMEM`, matching shadPS4.
///
/// Measured: A Plague Tale Requiem stops its boot on this import.
fn hle_available_direct_memory_size(ctx: &HleContext, args: &[u64]) -> u64 {
    let search_start = args.first().copied().unwrap_or(0);
    let search_end = args.get(1).copied().unwrap_or(0);
    let alignment = args.get(2).copied().unwrap_or(0);
    let phys_addr_out = args.get(3).copied().unwrap_or(0);
    let size_out = args.get(4).copied().unwrap_or(0);
    debug!(
        "sceKernelAvailableDirectMemorySize(start={search_start:#x}, end={search_end:#x}, \
         align={alignment:#x})"
    );
    if phys_addr_out == 0 || size_out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }

    let align = alignment.max(xps5x_core::PS5_PAGE_SIZE as u64);
    let end = search_end.min(PS5_DIRECT_MEMORY_SIZE);
    // Align the base up, then floor the length to the same granularity.
    let start = search_start.div_ceil(align).saturating_mul(align);
    let window = end.saturating_sub(start);

    // Report what is actually still FREE, not the size of the window. A title
    // uses this to size its pools, so over-reporting makes it commit to an
    // allocation it can never get: the measured UE titles probe with
    // `sceKernelAllocateDirectMemory` until ENOMEM (which is the deliberate
    // discovery mechanism — see `hle_allocate_direct_memory`), and answering
    // "the whole pool is free" after 13.8 GiB is already handed out contradicts
    // that probe.
    let used = ctx
        .kernel
        .direct_memory_allocated
        .load(std::sync::atomic::Ordering::Relaxed);
    let free = PS5_DIRECT_MEMORY_SIZE.saturating_sub(used);
    let size = (window.min(free) / align).saturating_mul(align);
    if size == 0 {
        return SCE_KERNEL_ERROR_ENOMEM;
    }

    if !ctx.mem.write(phys_addr_out, &start.to_le_bytes())
        || !ctx.mem.write(size_out, &size.to_le_bytes())
    {
        warn!("sceKernelAvailableDirectMemorySize: out-param not writable — EFAULT");
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// `sceKernelIsTrinityMode()`: is this console a PS5 **Pro** ("Trinity")? The
/// libkernel twin of `libSceAgc`'s `sceAgcGetIsTrinityMode`; XPS5X emulates a
/// base PS5, so both answer false.
///
/// Name recovered for NID `0xb54e5eddff604a25` (`tU5e3f9gSiU`) from SharpEmu's
/// aerolib catalogue. Measured: Until Dawn stops its boot on this import.
fn hle_is_trinity_mode(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0
}

/// `sceKernelSyncOnAddressWait(addr, ...)`: park until the watched address is
/// woken. No true parking lot yet, so this returns after a short slice as a
/// permitted spurious wakeup (mirrors `scePthreadCondWait`) — the caller
/// re-checks its own condition and either proceeds or waits again. Safe: it
/// never deadlocks and the slice keeps it off a tight spin.
fn hle_sync_on_address_wait(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelSyncOnAddressWait(addr={:#x}) -> spurious wakeup",
        args.first().copied().unwrap_or(0)
    );
    std::thread::sleep(std::time::Duration::from_millis(10));
    SCE_OK
}

/// `sceKernelSyncOnAddressWake(addr, count)`: wake up to `count` parkers on
/// `addr`. Parkers here self-release on their spurious-wakeup slice, so this is
/// an accounted no-op that reports success.
fn hle_sync_on_address_wake(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelSyncOnAddressWake(addr={:#x}, count={})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
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

/// `sceKernelClearVirtualRangeName(addr, len)`: the inverse of
/// [`hle_set_virtual_range_name`]. Range names are diagnostic-only in this
/// model (Set does not record them), so clearing is an accepted no-op.
fn hle_clear_virtual_range_name(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelClearVirtualRangeName(addr={:#x}, len={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    SCE_OK
}

/// `SCE_KERNEL_ERROR_EACCES` (`0x8002000D`, errno 13): shadPS4's
/// `DirectMemoryQuery` code for "no allocated direct memory owns this offset".
const SCE_KERNEL_ERROR_EACCES: u64 = 0x8002_000D;

/// `sceKernelDirectMemoryQuery(offset, flags, SceKernelDirectMemoryQueryInfo
/// *info, infoSize)` — shadPS4 `memory.cpp` (NID `BHouLQzh0X0`): find the
/// allocated direct-memory region containing `offset` (`flags == 1` searches
/// forward to the next allocated region) and report `{ u64 start; u64 end;
/// s32 memoryType }` (shadPS4's `OrbisQueryInfo`).
///
/// XPS5X's direct-memory allocator hands out arena addresses and records each
/// allocation via `kernel.memory.record_mapping`, so the honest answer is the
/// recorded region containing (or following) the queried address. Unallocated
/// offsets are `EACCES`, matching shadPS4.
fn hle_direct_memory_query(ctx: &HleContext, args: &[u64]) -> u64 {
    let offset = args.first().copied().unwrap_or(0);
    let flags = args.get(1).copied().unwrap_or(0) as i32;
    let info_out = args.get(2).copied().unwrap_or(0);
    let info_size = args.get(3).copied().unwrap_or(0x14);
    debug!(
        "sceKernelDirectMemoryQuery(offset={offset:#x}, flags={flags}, infoSize={info_size:#x})"
    );
    if info_out == 0 || info_size < 0x14 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let region = ctx.kernel.memory.region_containing(offset).or_else(|| {
        if flags == 1 {
            ctx.kernel.memory.region_at_or_after(offset)
        } else {
            None
        }
    });
    let Some(region) = region else {
        debug!("sceKernelDirectMemoryQuery: no allocated region owns {offset:#x} — EACCES");
        return SCE_KERNEL_ERROR_EACCES;
    };
    let mut info = [0u8; 0x14];
    info[0..8].copy_from_slice(&region.vaddr.to_le_bytes());
    info[8..16].copy_from_slice(&(region.vaddr + region.size).to_le_bytes());
    // Memory type: XPS5X models one CPU-coherent pool (Onion / WB_ONION = 0).
    info[16..20].copy_from_slice(&0i32.to_le_bytes());
    if !ctx.mem.write(info_out, &info) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// `sceKernelConfiguredFlexibleMemorySize(size_t *sizeOut)` — shadPS4
/// `memory.cpp` (NID `n1-v6FgU7MQ`): the total configured flexible-memory
/// budget. Must agree with what `sceKernelAvailableFlexibleMemorySize`
/// reports ([`FLEXIBLE_MEMORY_SIZE`]) — a title that reads both expects
/// configured >= available.
fn hle_configured_flexible_memory_size(ctx: &HleContext, args: &[u64]) -> u64 {
    let size_out = args.first().copied().unwrap_or(0);
    debug!("sceKernelConfiguredFlexibleMemorySize(sizeOut={size_out:#x})");
    if size_out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    if !ctx.mem.write(size_out, &FLEXIBLE_MEMORY_SIZE.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// `sceKernelGetOpenPsId(uint8_t id[16])` (library `libSceOpenPsId`): the
/// console's 16-byte "open PS id". shadPS4 carries only the aerolib stub;
/// XPS5X reports a deterministic per-install-independent constant — stable
/// across runs so a title keying caches/telemetry buckets on it never sees
/// the id change, and obviously synthetic in logs.
fn hle_get_open_ps_id(ctx: &HleContext, args: &[u64]) -> u64 {
    const OPEN_PS_ID: [u8; 16] = *b"XPS5X-OpenPsId\x00\x01";
    let id_out = args.first().copied().unwrap_or(0);
    debug!("sceKernelGetOpenPsId(id={id_out:#x})");
    if id_out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    if !ctx.mem.write(id_out, &OPEN_PS_ID) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
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

/// `pthread_create_name_np(out, attr, entry, arg, name)` — the Posix-named
/// twin of `scePthreadCreate`: create, then record the diagnostic name
/// against the id written back (measured: Dragon Ball names every worker
/// thread at spawn).
fn hle_pthread_create_name_np(ctx: &HleContext, args: &[u64]) -> u64 {
    let thread_out = args.first().copied().unwrap_or(0);
    let attr = args.get(1).copied().unwrap_or(0);
    let entry = args.get(2).copied().unwrap_or(0);
    let arg = args.get(3).copied().unwrap_or(0);
    let name_ptr = args.get(4).copied().unwrap_or(0);
    debug!("pthread_create_name_np(out={thread_out:#x}, entry={entry:#x}, name={name_ptr:#x})");
    let rc = ctx.guest_threads.create(thread_out, attr, entry, arg);
    if rc != SCE_OK || name_ptr == 0 || thread_out == 0 {
        return rc;
    }
    let mut id_bytes = [0u8; 8];
    if !ctx.mem.read(thread_out, &mut id_bytes) {
        return rc;
    }
    let target = u64::from_le_bytes(id_bytes);
    let mut buf = [0u8; 32];
    if !ctx.mem.read(name_ptr, &mut buf) {
        return rc;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    if let Ok(name) = std::str::from_utf8(&buf[..end]) {
        ctx.kernel.thread_names.insert(target, name.to_owned());
    }
    rc
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
/// `{ module_id: u64, offset: u64 }` to that thread-local's storage.
///
/// Every module in the process's static TLS layout resolves into the
/// **static** area the runtime built from all the modules' `PT_TLS` templates
/// (via the kernel's per-module area offsets); only a module loaded outside
/// that layout gets bounded, zero-initialized dynamic storage. The storage
/// lives in the guest arena (never host-only memory), so the returned pointer
/// is directly dereferenceable by native guest code.
///
/// # Why a static module must not get a dynamic block
///
/// It used to, twice, and both were measured Minecraft crashes. Code reaches
/// thread-locals through the general-dynamic model — this function — as well
/// as through `TPOFF64` (`fs`-relative), and the ELF TLS ABI requires both to
/// land on the same address. Returning fresh storage here gives one variable
/// two homes, and only the static one is ever initialized from `.tdata`.
///
/// First the *main executable*: its `PT_TLS` is `tdata=0x8 memsz=0x78`, a
/// single initialized pointer at offset 0. It asked for
/// `{module: 1, offset: 0}`, read the zeroed copy back as `NULL`, and
/// dereferenced it. Then the same shape again for *dependencies*:
/// `libRenoirCore.PS5.prx` (the title's UI renderer) writes a thread context
/// pointer into its own TLS and reads it back general-dynamically — while
/// every `DTPMOD64` in the process still resolved to module 1, so four
/// modules' thread-locals (eboot, libc.prx, libcohtml, libRenoirCore) aliased
/// the eboot's block and none of the others' `.tdata` was ever materialized.
/// The fix is the process-wide layout this function now consults first.
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

    // A static TLS module's storage already exists and is already initialized:
    // the runtime copied every module's `.tdata` into the per-thread static
    // area, and the linker's `TPOFF64` offsets are computed against the same
    // layout the kernel's per-module area offsets describe. Alias that storage
    // rather than allocate beside it — the ELF TLS ABI requires the
    // general-dynamic and initial-exec models to land on the SAME address.
    //
    // `offset` is not bounds-checked against the module's block here: it comes
    // from the linker's own `DTPOFF64`, computed against the same template the
    // block was sized from, so the two agree by construction — the same
    // agreement `TPOFF64` already relies on.
    if let Some(area_offset) = ctx
        .kernel
        .static_tls_area_offsets
        .get(&module_id)
        .map(|entry| *entry)
        && let Some(area_base) = ctx.guest_threads.current_static_tls_block()
    {
        return area_base + area_offset + offset;
    }

    // No registered layout (single-module fixtures, test doubles): the main
    // module is the whole TLS world and the static block is its block. A
    // `None` block means there is no static TLS at all, and the dynamic path
    // below is then the honest answer.
    if module_id == MAIN_TLS_MODULE_ID
        && let Some(base) = ctx.guest_threads.current_static_tls_block()
    {
        return base + offset;
    }

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
pub(crate) fn hle_error_addr(ctx: &HleContext, _args: &[u64]) -> u64 {
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
    ctx.services.sleep(std::time::Duration::from_millis(ms));
    if rem != 0 {
        let _ = ctx.mem.write(rem, &[0u8; 16]);
    }
    SCE_OK
}

/// Size of FreeBSD's `struct rusage`: two `timeval`s (16 bytes each) plus
/// fourteen `long` counters — 144 bytes.
const RUSAGE_SIZE: usize = 144;

/// `getrusage(who, rusage*)`: report zeroed resource usage. XPS5X keeps no
/// per-process rusage accounting; all-zero counters are well-formed values a
/// caller can add/subtract safely (unlike an unwritten struct).
fn hle_getrusage(ctx: &HleContext, args: &[u64]) -> u64 {
    let who = args.first().copied().unwrap_or(0) as i64;
    let usage = args.get(1).copied().unwrap_or(0);
    debug!("getrusage(who={who}, usage={usage:#x}) -> zeroed counters");
    if usage == 0 || !ctx.mem.write(usage, &[0u8; RUSAGE_SIZE]) {
        set_guest_errno(ctx, 14); // EFAULT
        return (-1i64) as u64;
    }
    0
}

/// `signal(sig, handler)`: accept the registration and report the previous
/// handler as `SIG_DFL` (0). XPS5X never delivers guest signals (see
/// `_sigprocmask`/`_is_signal_return`), so recording the handler would only
/// promise a callback that can never arrive.
fn hle_posix_signal(_ctx: &HleContext, args: &[u64]) -> u64 {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        warn!(
            "signal(sig={}, handler={:#x}): accepted, but guest signals are never delivered \
             (logged once)",
            args.first().copied().unwrap_or(0),
            args.get(1).copied().unwrap_or(0)
        );
    });
    0 // previous handler = SIG_DFL
}

/// `sceKernelMlock(addr, len)`: guest memory is host-resident by construction
/// (the arena is committed memory, never paged out by us), so the lock request
/// is already satisfied.
fn hle_mlock(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelMlock(addr={:#x}, len={:#x}) -> OK (arena memory is resident)",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    SCE_OK
}

/// POSIX `mprotect(addr, len, prot)`: page protections are not remapped per
/// guest request yet (the arena stays RWX for HLE trampolines); accepting the
/// request is the same shortcut `sceKernelMprotect` takes.
fn hle_posix_mprotect(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0) as u32;
    debug!("mprotect(addr={addr:#x}, len={len:#x}, prot={prot:#x})");
    // Apply the protection when enforcement is on; a no-op otherwise (the arena
    // default). POSIX `mprotect` returns 0 on success.
    if ctx.mem.protect(addr, len, prot) {
        0
    } else {
        FILE_EINVAL
    }
}

/// `sceKernelMprotect(void *addr, size_t len, int prot)`. Same shape as the
/// POSIX call; applies the protection under `XPS5X_ENFORCE_MPROTECT`, no-op
/// otherwise.
fn hle_kernel_mprotect(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0) as u32;
    debug!("sceKernelMprotect(addr={addr:#x}, len={len:#x}, prot={prot:#x})");
    if ctx.mem.protect(addr, len, prot) {
        SCE_OK
    } else {
        SCE_KERNEL_ERROR_EINVAL
    }
}

/// Real signature: `sceKernelMapDirectMemory2(void **addrOut, size_t len, int
/// type, int prot, int flags, off_t directMemoryStart, size_t alignment)`.
///
/// Identical to `sceKernelMapDirectMemory` except for the extra `type`
/// argument at position 2 (which shifts `prot`/`flags`/`phys`/`alignment` one
/// slot right). The type only selects a physical-memory pool on hardware, so
/// it is logged and otherwise ignored — the arguments are reshuffled into the
/// existing implementation.
fn hle_map_direct_memory2(ctx: &HleContext, args: &[u64]) -> u64 {
    let memory_type = args.get(2).copied().unwrap_or(0);
    debug!("sceKernelMapDirectMemory2(type={memory_type}) -> sceKernelMapDirectMemory");
    hle_map_direct_memory(
        ctx,
        &[
            args.first().copied().unwrap_or(0), // addrOut
            args.get(1).copied().unwrap_or(0),  // len
            args.get(3).copied().unwrap_or(0),  // prot
            args.get(4).copied().unwrap_or(0),  // flags
            args.get(5).copied().unwrap_or(0),  // directMemoryStart
            args.get(6).copied().unwrap_or(0),  // alignment
        ],
    )
}

/// `sceKernelInternalMemoryGetModuleSegmentInfo(out)`: internal SDK call with
/// no public ABI contract. Filling a structure of unknown layout would hand
/// the guest garbage it trusts, so this fails loudly (once) with `EINVAL` and
/// writes nothing.
fn hle_internal_get_module_segment_info(_ctx: &HleContext, args: &[u64]) -> u64 {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        warn!(
            "sceKernelInternalMemoryGetModuleSegmentInfo(out={:#x}): ABI unknown — EINVAL \
             (logged once)",
            args.first().copied().unwrap_or(0)
        );
    });
    SCE_KERNEL_ERROR_EINVAL
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

/// Module-info calls without an implemented ABI return ESRCH rather than
/// exposing a half-filled structure the guest would chase into invalid memory.
fn hle_module_info_unavailable(_ctx: &HleContext, args: &[u64]) -> u64 {
    warn!(
        "sceKernelGetModuleInfo*(addr={:#x}): module info not implemented — returning ESRCH",
        args.first().copied().unwrap_or(0)
    );
    SCE_KERNEL_ERROR_ESRCH
}

/// `sceKernelGetModuleInfoForUnwind(addr, flags, info)`.
///
/// The 304-byte Orbis ABI is `size`, a 256-byte name, then the exception
/// header/table and first load-segment address/size. The caller initializes
/// `size`; matching the kernel contract prevents writes into older, shorter
/// structure revisions.
pub(crate) fn hle_get_module_info_for_unwind(ctx: &HleContext, args: &[u64]) -> u64 {
    const INFO_SIZE: usize = 304;
    const NAME_OFFSET: usize = 8;
    const NAME_SIZE: usize = 256;

    let addr = args.first().copied().unwrap_or(0);
    let flags = args.get(1).copied().unwrap_or(0);
    let info_addr = args.get(2).copied().unwrap_or(0);
    if flags >= 3 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    if info_addr == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let mut caller_size = [0u8; 8];
    if !ctx.mem.read(info_addr, &mut caller_size) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if u64::from_le_bytes(caller_size) < INFO_SIZE as u64 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let Some(module) = ctx.kernel.unwind_module_for_addr(addr) else {
        warn!("sceKernelGetModuleInfoForUnwind(addr={addr:#x}): no loaded ELF owns address");
        return SCE_KERNEL_ERROR_ESRCH;
    };

    let mut info = [0u8; INFO_SIZE];
    info[0..8].copy_from_slice(&(INFO_SIZE as u64).to_le_bytes());
    let name = module.name.as_bytes();
    let name_len = name.len().min(NAME_SIZE - 1);
    info[NAME_OFFSET..NAME_OFFSET + name_len].copy_from_slice(&name[..name_len]);
    info[264..272].copy_from_slice(&module.eh_frame_hdr_addr.to_le_bytes());
    info[272..280].copy_from_slice(&module.eh_frame_addr.to_le_bytes());
    info[280..288].copy_from_slice(&module.eh_frame_size.to_le_bytes());
    info[288..296].copy_from_slice(&module.seg0_addr.to_le_bytes());
    info[296..304].copy_from_slice(&module.seg0_size.to_le_bytes());
    if !ctx.mem.write(info_addr, &info) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    debug!(
        "sceKernelGetModuleInfoForUnwind({addr:#x}) -> {} [{:#x}, {:#x}) eh_frame={:#x}+{:#x}",
        module.name, module.start, module.end, module.eh_frame_addr, module.eh_frame_size
    );
    SCE_OK
}

/// `sceKernelVirtualQuery(addr, flags, info, infoSize)`: no query surface
/// yet; an honest EFAULT tells the caller nothing was written.
#[allow(dead_code)]
fn hle_virtual_query_stub(_ctx: &HleContext, args: &[u64]) -> u64 {
    warn!(
        "sceKernelVirtualQuery(addr={:#x}): not implemented — returning EFAULT",
        args.first().copied().unwrap_or(0)
    );
    SCE_KERNEL_ERROR_EFAULT
}

/// Return the mapped or reserved region containing `addr` (or the next
/// region for flag bit 0) using the 72-byte Gen4/Gen5 query-info ABI.
fn hle_virtual_query(ctx: &HleContext, args: &[u64]) -> u64 {
    const INFO_SIZE: usize = 72;
    const FIND_NEXT: i32 = 1;
    const SCE_KERNEL_ERROR_EACCES: u64 = 0x8002_000D;

    let address = args.first().copied().unwrap_or(0);
    let flags = args.get(1).copied().unwrap_or(0) as i32;
    let info = args.get(2).copied().unwrap_or(0);
    let info_size = args.get(3).copied().unwrap_or(0);
    if info == 0 || info_size < INFO_SIZE as u64 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let region = ctx.kernel.memory.region_containing(address).or_else(|| {
        (flags & FIND_NEXT != 0)
            .then(|| ctx.kernel.memory.region_at_or_after(address))
            .flatten()
    });
    let Some(region) = region else {
        return SCE_KERNEL_ERROR_EACCES;
    };
    let Some(end) = region.vaddr.checked_add(region.size) else {
        return SCE_KERNEL_ERROR_EFAULT;
    };

    let mut payload = [0u8; INFO_SIZE];
    payload[0..8].copy_from_slice(&region.vaddr.to_le_bytes());
    payload[8..16].copy_from_slice(&end.to_le_bytes());
    payload[24..28].copy_from_slice(&region.protection.bits().to_le_bytes());
    if !region.protection.is_empty() {
        payload[32] |= 0x10; // is_committed
    }
    let name = region.name.as_deref().unwrap_or("mapped").as_bytes();
    let name_len = name.len().min(31);
    payload[33..33 + name_len].copy_from_slice(&name[..name_len]);
    if !ctx.mem.write(info, &payload) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
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
    let Some(addr) = ctx.alloc.reserve(len, align) else {
        warn!("sceKernelReserveVirtualRange: address-space reservation failed (len={len:#x})");
        return HLE_ERROR;
    };
    ctx.kernel.memory.record_mapping(addr, len, 0);
    if !ctx.mem.write(addr_inout, &addr.to_le_bytes()) {
        ctx.alloc.munmap(addr, len);
        ctx.kernel.memory.remove_mapping(addr);
        return SCE_KERNEL_ERROR_EFAULT;
    }
    debug!("sceKernelReserveVirtualRange(len={len:#x}, align={align:#x}) -> {addr:#x}");
    SCE_OK
}

/// Diagnostic trap for the C++ ABI `__cxa_throw(void* obj, std::type_info*
/// tinfo, void (*dest)(void*))`. Reads the thrown exception's type name
/// (`tinfo->__type_name`, an offset-8 `const char*` in the Itanium ABI) and
/// logs it, then terminates the calling thread — the exception these worker
/// threads throw is uncaught anyway, so the end state (dead thread) is the
/// same, and this NAMES it instead of leaving it anonymous. Only reachable
/// when the linker force-routes `__cxa_throw` here (XPS5X_TRAP_CXA_THROW);
/// normal runs use the shipped libc's real `__cxa_throw`.
fn hle_cxa_throw(ctx: &HleContext, args: &[u64]) -> u64 {
    let obj = args.first().copied().unwrap_or(0);
    let tinfo = args.get(1).copied().unwrap_or(0);
    let thread = ctx.guest_threads.current_thread();
    let name = ctx
        .kernel
        .thread_names
        .get(&thread)
        .map_or_else(|| "<unnamed>".to_owned(), |entry| entry.clone());

    let read_u64 = |addr: u64| {
        let mut b = [0u8; 8];
        ctx.mem
            .read(addr, &mut b)
            .then(|| u64::from_le_bytes(b))
            .filter(|v| *v != 0)
    };
    let type_name = (tinfo != 0)
        .then(|| {
            read_u64(tinfo.wrapping_add(8))
                .and_then(|name_ptr| crate::fmt::read_cstr(ctx.mem, name_ptr))
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        })
        .flatten()
        .unwrap_or_else(|| "<unreadable>".to_owned());

    // Dump the raw exception object and probe each qword for a string, plus a
    // small int (the errno) — the libc++ system_error layout varies, so show
    // the shape rather than guess an offset.
    let mut object_dump = String::new();
    if obj != 0 {
        for i in 0..8u64 {
            let mut b = [0u8; 8];
            if !ctx.mem.read(obj.wrapping_add(i * 8), &mut b) {
                break;
            }
            let v = u64::from_le_bytes(b);
            let s = (v > 0x1000 && v < 0x1_0000_0000_0000)
                .then(|| crate::fmt::read_cstr(ctx.mem, v))
                .flatten()
                .filter(|bytes| bytes.iter().all(|c| c.is_ascii_graphic() || *c == b' '))
                .map(|bytes| format!(" str={:?}", String::from_utf8_lossy(&bytes)))
                .unwrap_or_default();
            object_dump.push_str(&format!(" +{}={:#x}{s}", i * 8, v));
        }
    }
    // Stack code-address chain: the frame above __throw_system_error is the
    // threading primitive (std::mutex/condition_variable/thread) that failed.
    let mut chain = Vec::new();
    if ctx.caller_rsp != 0 {
        for i in 0..96u64 {
            let mut b = [0u8; 8];
            if !ctx.mem.read(ctx.caller_rsp.wrapping_add(i * 8), &mut b) {
                break;
            }
            let v = u64::from_le_bytes(b);
            if (0x1000_0000_0000..0x1000_2000_0000).contains(&v) {
                chain.push(format!("{v:#x}"));
            }
        }
    }

    // The exact HLE calls this guest thread made before throwing — the most
    // reliable "what failed" signal (host threads are pooled).
    let recent = ctx
        .kernel
        .recent_hle_calls
        .get(&thread)
        .map(|ring| {
            let q = ring.lock();
            q.iter().cloned().collect::<Vec<_>>().join(" ")
        })
        .unwrap_or_default();

    warn!(
        "__cxa_throw trap: thread {thread} ('{name}') throws '{type_name}' \
         (obj={obj:#x}, ra={:#x}) object[{object_dump} ] recent=[{recent}] stack=[{}]",
        ctx.caller_return_addr,
        chain.join(" ")
    );
    ctx.guest_threads.request_exit(0xa002_0008);
    0
}

/// `sceKernelDebugRaiseException*`: the title is reporting a fatal
/// condition. Log it loudly; returning lets the guest continue into
/// whatever it does next (usually an exit path).
fn hle_debug_raise_exception(ctx: &HleContext, args: &[u64]) -> u64 {
    let code = args.first().copied().unwrap_or(0);
    let thread = ctx.guest_threads.current_thread();
    let name = ctx
        .kernel
        .thread_names
        .get(&thread)
        .map_or_else(|| "<unnamed>".to_owned(), |entry| entry.clone());
    warn!(
        "sceKernelDebugRaiseException(code={code:#x}, arg={:#x}) on thread {thread} \
         ('{name}') from guest ra={:#x} — guest reported a fatal condition; \
         terminating the calling guest thread",
        args.get(1).copied().unwrap_or(0),
        ctx.caller_return_addr,
    );
    // The exact HLE calls this guest thread made before the fatal condition —
    // names what it was doing (defeats host-thread pooling). Populated when
    // XPS5X_TRACE_EINVAL / XPS5X_TRAP_CXA_THROW is set.
    if let Some(ring) = ctx.kernel.recent_hle_calls.get(&thread) {
        let recent = ring.lock().iter().cloned().collect::<Vec<_>>().join(" ");
        if !recent.is_empty() {
            warn!("  recent HLE calls: {recent}");
        }
    }
    // Walk the caller's stack for return addresses into the loaded image, so
    // the call chain INTO libc's terminate handler (and thus the throw
    // origin) is greppable. Diagnostic only; bounded and read-only.
    if ctx.caller_rsp != 0 {
        let mut chain = Vec::new();
        for i in 0..256u64 {
            let mut buf = [0u8; 8];
            if !ctx.mem.read(ctx.caller_rsp.wrapping_add(i * 8), &mut buf) {
                break;
            }
            let val = u64::from_le_bytes(buf);
            // Return addresses land inside the composed guest image
            // (0x1000_0000_0000 .. +~300 MB); stack data / small ints don't.
            if (0x1000_0000_0000..0x1000_2000_0000).contains(&val) {
                chain.push(format!("{val:#x}"));
            }
        }
        if !chain.is_empty() {
            warn!("  fatal-thread stack code-addrs: {}", chain.join(" "));
        }
    }
    // A thread killed mid-execution never runs its C++ cleanup, so any mutex
    // it holds would stay locked forever — and our mutex lock is a spin-loop
    // that only frees on owner-unlock, so every other thread waiting on that
    // mutex spins indefinitely (a silent, total deadlock; measured on
    // Minecraft when its Streaming/REST/Watchdog workers abort while the
    // render thread waits on a mutex they held). Release this dying thread's
    // held mutexes so waiters can make progress, mirroring what robust-mutex
    // owner-death recovery does on real systems (EOWNERDEAD).
    let released = ctx.kernel.release_mutexes_owned_by(thread);
    if released > 0 {
        warn!("  released {released} mutex(es) held by dying thread {thread} ('{name}')");
    }

    // On hardware this never returns — the process is killed. Returning here
    // is measurably worse than stopping the thread: the caller's code ends at
    // the call instruction (noreturn), so a return "executes" whatever bytes
    // follow — on the measured title, a jump through a null slot that then
    // gets reported as OUR wild-jump bug. Exit the calling thread instead;
    // other threads keep running so the run stays observable.
    ctx.guest_threads.request_exit(code);
    SCE_OK
}

/// `sceKernelDebugWriteCppExceptionInfo`: record C++ exception diagnostics.
///
/// This is a reporting sink used after the language runtime has already
/// gathered its unwind state. The structure ABI is not public enough to
/// inspect safely here, so keep the guest pointer opaque and log only the
/// bounded register arguments supplied by the HLE dispatcher.
fn hle_debug_write_cpp_exception_info(_ctx: &HleContext, args: &[u64]) -> u64 {
    warn!(
        "sceKernelDebugWriteCppExceptionInfo(info={:#x}, arg1={:#x}, arg2={:#x}, arg3={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0),
        args.get(3).copied().unwrap_or(0),
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

/// Map a host I/O error to the POSIX errno the guest expects, as this
/// module's internal negative-errno convention (see [`file_result_posix`] /
/// [`file_result_sce`]).
fn io_error_to_file_result(error: &std::io::Error) -> u64 {
    use std::io::ErrorKind;
    let errno: i64 = match error.kind() {
        ErrorKind::NotFound => 2,           // ENOENT
        ErrorKind::PermissionDenied => 13,  // EACCES
        ErrorKind::AlreadyExists => 17,     // EEXIST
        ErrorKind::DirectoryNotEmpty => 66, // ENOTEMPTY (FreeBSD)
        ErrorKind::InvalidInput => 22,      // EINVAL
        _ => 5,                             // EIO
    };
    (-errno) as u64
}

/// Read the guest path argument at `args[index]`, or `None` on a bad pointer.
fn read_guest_path(ctx: &HleContext, args: &[u64], index: usize) -> Option<String> {
    let ptr = args.get(index).copied().unwrap_or(0);
    let bytes = crate::fmt::read_cstr(ctx.mem, ptr)?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// `unlink(path)` / `sceKernelUnlink`: remove a file beneath a mounted VFS
/// root (internal negative-errno convention; see the wrappers below).
fn hle_unlink_core(ctx: &HleContext, args: &[u64]) -> u64 {
    let Some(path) = read_guest_path(ctx, args, 0) else {
        return FILE_EFAULT;
    };
    match ctx.kernel.filesystem.remove_file(&path) {
        Ok(()) => SCE_OK,
        Err(e) => {
            warn!("unlink('{path}') failed: {e}");
            io_error_to_file_result(&e)
        }
    }
}

/// `rmdir(path)` / `sceKernelRmdir`: remove an empty directory beneath a
/// mounted VFS root.
fn hle_rmdir_core(ctx: &HleContext, args: &[u64]) -> u64 {
    let Some(path) = read_guest_path(ctx, args, 0) else {
        return FILE_EFAULT;
    };
    match ctx.kernel.filesystem.remove_dir(&path) {
        Ok(()) => SCE_OK,
        Err(e) => {
            warn!("rmdir('{path}') failed: {e}");
            io_error_to_file_result(&e)
        }
    }
}

/// `sceKernelRename(from, to)`: rename between mounted VFS paths.
fn hle_rename_core(ctx: &HleContext, args: &[u64]) -> u64 {
    let (Some(from), Some(to)) = (read_guest_path(ctx, args, 0), read_guest_path(ctx, args, 1))
    else {
        return FILE_EFAULT;
    };
    match ctx.kernel.filesystem.rename(&from, &to) {
        Ok(()) => SCE_OK,
        Err(e) => {
            warn!("rename('{from}' -> '{to}') failed: {e}");
            io_error_to_file_result(&e)
        }
    }
}

/// `sceKernelTruncate(path, length)`: set the file's length.
fn hle_truncate_core(ctx: &HleContext, args: &[u64]) -> u64 {
    let Some(path) = read_guest_path(ctx, args, 0) else {
        return FILE_EFAULT;
    };
    let length = args.get(1).copied().unwrap_or(0);
    match ctx.kernel.filesystem.truncate(&path, length) {
        Ok(()) => SCE_OK,
        Err(e) => {
            warn!("truncate('{path}', {length}) failed: {e}");
            io_error_to_file_result(&e)
        }
    }
}

fn hle_sce_unlink(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_unlink_core(ctx, args))
}

fn hle_posix_unlink(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_unlink_core(ctx, args))
}

fn hle_sce_rmdir(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_rmdir_core(ctx, args))
}

fn hle_posix_rmdir(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_rmdir_core(ctx, args))
}

fn hle_sce_rename(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_rename_core(ctx, args))
}

fn hle_sce_truncate(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_truncate_core(ctx, args))
}

/// `sceKernelSync()`: flush everything to stable storage. The VFS persists
/// dirty descriptors on `fsync`/`close`, and host writes are already durable
/// from the guest's perspective, so a global sync has nothing further to do.
fn hle_sce_sync(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelSync() -> OK (VFS persists on fsync/close)");
    SCE_OK
}

/// `sceKernelChmod(path, mode)` / `sceKernelUtimes(path, times)`: the VFS
/// stores no guest permission bits or timestamps of its own (host metadata is
/// authoritative), so a resolvable existing path is accepted with a debug log
/// and an unresolvable one is a real `ENOENT`.
fn hle_path_metadata_accept(ctx: &HleContext, args: &[u64]) -> u64 {
    let Some(path) = read_guest_path(ctx, args, 0) else {
        return SCE_KERNEL_ERROR_EFAULT;
    };
    match ctx.kernel.filesystem.resolve_path(&path) {
        Some(host) if host.exists() => {
            debug!("chmod/utimes('{path}') accepted (no guest metadata modeled)");
            SCE_OK
        }
        _ => {
            warn!("chmod/utimes('{path}'): not a mounted existing path — ENOENT");
            SCE_KERNEL_ERROR_ENOENT
        }
    }
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
fn host_realtime(ctx: &HleContext) -> (i64, i64) {
    match ctx
        .services
        .wall_clock()
        .duration_since(std::time::UNIX_EPOCH)
    {
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
    let (sec, nanos) = host_realtime(ctx);
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
pub(crate) const CLOCK_MONOTONIC: u64 = 4;

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
        let elapsed = ctx.services.monotonic_elapsed();
        (elapsed.as_secs() as i64, elapsed.subsec_nanos() as i64)
    } else {
        host_realtime(ctx)
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

/// Real `sceKernelClockGetres(clockId, struct timespec *res)`: the clock's
/// resolution as a PS5 `timespec`. The nanosecond domain here matches
/// `sceKernelClockGettime`, so report {0, 1} — 1 ns — for every clock id.
fn hle_clock_getres(ctx: &HleContext, args: &[u64]) -> u64 {
    let clock_id = args.first().copied().unwrap_or(0);
    let res = args.get(1).copied().unwrap_or(0);
    debug!("sceKernelClockGetres(clockId={clock_id}, res={res:#x})");
    if res == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let mut buf = [0u8; 16];
    buf[8..16].copy_from_slice(&1i64.to_le_bytes()); // tv_nsec = 1
    if !ctx.mem.write(res, &buf) {
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
fn hle_get_process_time(ctx: &HleContext, _args: &[u64]) -> u64 {
    let us = ctx.services.monotonic_elapsed().as_micros();
    debug!("sceKernelGetProcessTime() -> {us}us");
    u64::try_from(us).unwrap_or(u64::MAX)
}

/// Real `sceKernelGetProcessTimeCounter()`: elapsed nanoseconds since process
/// start (paired with [`PROCESS_TIME_COUNTER_HZ`]). Monotonic.
fn hle_get_process_time_counter(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::try_from(ctx.services.monotonic_elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// `sceKernelGetProcessTimeCounterFrequency()`: the counter's frequency in
/// Hz — the divisor a title applies to the counter to get seconds.
fn hle_get_process_time_counter_frequency(_ctx: &HleContext, _args: &[u64]) -> u64 {
    PROCESS_TIME_COUNTER_HZ
}

/// A fixed monotonic reference captured on first use, so `CLOCK_MONOTONIC`
/// reports a stable, never-decreasing elapsed time across the process.
pub(crate) fn process_start() -> std::time::Instant {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    *START.get_or_init(std::time::Instant::now)
}

/// Plausible PS5 base-clock TSC frequency (1.6 GHz). Reported by
/// `sceKernelGetTscFrequency` and used to scale [`hle_read_tsc`] so the two are
/// self-consistent.
const TSC_FREQ_HZ: u64 = 1_600_000_000;

/// Stub: plausible PS5 base-clock TSC frequency (1.6 GHz).
fn hle_get_tsc_frequency(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceKernelGetTscFrequency()");
    TSC_FREQ_HZ
}

/// Real `sceKernelReadTsc()`: the CPU timestamp counter, a raw `u64` return.
/// Titles poll it for fine-grained timing (busy-waits, audio pacing,
/// profiling). Counting the host TSC directly would drift from the 1.6 GHz we
/// report via [`hle_get_tsc_frequency`], breaking any `(tsc2 - tsc1) / freq`
/// elapsed-seconds math; instead derive it from the monotonic process clock at
/// exactly that rate. A missing ReadTsc left audio/timing threads spinning on
/// a never-advancing counter.
fn hle_read_tsc(ctx: &HleContext, _args: &[u64]) -> u64 {
    let nanos = ctx.services.monotonic_elapsed().as_nanos();
    // ticks = nanos * TSC_FREQ_HZ / 1e9; u128 math avoids overflow before the
    // final narrowing.
    u64::try_from(nanos * u128::from(TSC_FREQ_HZ) / 1_000_000_000).unwrap_or(u64::MAX)
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
pub(crate) fn hle_usleep(ctx: &HleContext, args: &[u64]) -> u64 {
    let usec = args.first().copied().unwrap_or(0);
    debug!("sceKernelUsleep(usec={usec})");
    // A guest that sleeps in a tight loop is polling for something that never
    // arrives — the shape of every boot stall seen so far. The sleep itself says
    // nothing; the CALLER does, because `--dump-vaddr` turns that address into
    // the loop's condition. Report each distinct caller once so a stalled title
    // names its own spin site without flooding the log.
    if std::env::var_os("XPS5X_TRACE_SPIN").is_some() && ctx.caller_return_addr != 0 {
        static SEEN: std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<u64>>> =
            std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        let first = SEEN
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(ctx.caller_return_addr);
        if first {
            tracing::info!(
                "SPIN: sceKernelUsleep({usec}us) from guest {:#x} (thread {})",
                ctx.caller_return_addr,
                ctx.guest_threads.current_thread()
            );
        }
    }
    let requested = std::time::Duration::from_micros(usec);
    let dur = requested.min(USLEEP_MAX);
    if dur < requested {
        warn!(
            "sceKernelUsleep: {usec}us capped to {}us (USLEEP_MAX)",
            dur.as_micros()
        );
    }
    ctx.services.sleep(dur);
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
    if std::env::var_os("XPS5X_TRACE_PROCPARAM").is_some() {
        let r = |a: u64| -> u64 {
            let mut b = [0u8; 8];
            if ctx.mem.read(a, &mut b) {
                u64::from_le_bytes(b)
            } else {
                0xDEAD_DEAD_DEAD_DEAD
            }
        };
        // libc's _malloc_init reads SceProcParam[+0x38] -> SceLibcParam, and
        // fails (heap never inits -> native sceLibcMspaceCreate returns null)
        // unless SceProcParam size >= 0x40, SceLibcParam ptr != 0, its size
        // >= 0x40, and version fields (+8 >= 2, +0xc == 1) hold.
        let libc = r(addr + 0x38);
        warn!(
            "PROCPARAM addr={addr:#x} size={:#x} libcParam@0x38={libc:#x}",
            r(addr)
        );
        if libc != 0 && libc >> 48 == 0 {
            let ver = r(libc + 8);
            warn!(
                "  SceLibcParam size={:#x} ver[+8]={:#x} ver[+0xc]={:#x} p[+0x30]={:#x} p[+0x38]={:#x}",
                r(libc),
                ver & 0xffff_ffff,
                ver >> 32,
                r(libc + 0x30),
                r(libc + 0x38)
            );
        }
    }
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
    use crate::{GuestMemory, GuestThreadScheduler, test_ctx};

    /// A scheduler that reports a static TLS block, as the real runtime does for
    /// any module with a `PT_TLS`. `test_ctx`'s double deliberately does not, so
    /// it exercises the dynamic path instead.
    struct StaticTlsThreads(u64);

    impl GuestThreadScheduler for StaticTlsThreads {
        fn create(&self, _thread_out: u64, _attr: u64, _entry: u64, _arg: u64) -> u64 {
            0x8002_000B
        }
        fn join(&self, _thread: u64, _retval_out: u64) -> u64 {
            0x8002_0003
        }
        fn detach(&self, _thread: u64) -> u64 {
            0x8002_0003
        }
        fn request_exit(&self, _retval: u64) -> bool {
            false
        }
        fn current_thread(&self) -> u64 {
            1
        }
        fn request_process_exit(&self, _code: u64) {}
        fn process_is_terminating(&self) -> bool {
            false
        }
        fn current_static_tls_block(&self) -> Option<u64> {
            Some(self.0)
        }
    }

    /// Write a `tls_index { module_id, offset }` descriptor at `at`.
    fn write_tls_index(mem: &crate::TestMemory, at: u64, module_id: u64, offset: u64) {
        assert!(mem.write(at, &module_id.to_le_bytes()));
        assert!(mem.write(at + 8, &offset.to_le_bytes()));
    }

    /// DirectMemoryQuery reports the recorded allocation containing (or, with
    /// flags==1, following) the queried offset — shadPS4's OrbisQueryInfo
    /// `{start, end, memoryType}` — and EACCES for unallocated space.
    #[test]
    fn direct_memory_query_reports_recorded_regions() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        kernel.memory.record_mapping(0x40_0000, 0x8000, 0x3);

        // Inside the region.
        assert_eq!(
            hle_direct_memory_query(&ctx, &[0x40_1000, 0, 0x100, 0x18]),
            SCE_OK
        );
        let mut info = [0u8; 0x14];
        assert!(mem.read(0x100, &mut info));
        assert_eq!(
            u64::from_le_bytes(info[0..8].try_into().unwrap()),
            0x40_0000
        );
        assert_eq!(
            u64::from_le_bytes(info[8..16].try_into().unwrap()),
            0x40_8000
        );

        // Below it: flags=0 → EACCES; flags=1 → finds the next region.
        assert_eq!(
            hle_direct_memory_query(&ctx, &[0x10_0000, 0, 0x100, 0x18]),
            SCE_KERNEL_ERROR_EACCES
        );
        assert_eq!(
            hle_direct_memory_query(&ctx, &[0x10_0000, 1, 0x100, 0x18]),
            SCE_OK
        );
        assert!(mem.read(0x100, &mut info));
        assert_eq!(
            u64::from_le_bytes(info[0..8].try_into().unwrap()),
            0x40_0000
        );

        // NULL / undersized info out-param is EINVAL.
        assert_eq!(
            hle_direct_memory_query(&ctx, &[0x40_1000, 0, 0, 0x18]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(
            hle_direct_memory_query(&ctx, &[0x40_1000, 0, 0x100, 0x8]),
            SCE_KERNEL_ERROR_EINVAL
        );
    }

    /// ConfiguredFlexibleMemorySize agrees with the Available report's model.
    #[test]
    fn configured_flexible_memory_size_writes_the_pool_size() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_configured_flexible_memory_size(&ctx, &[0x100]), SCE_OK);
        let mut size = [0u8; 8];
        assert!(mem.read(0x100, &mut size));
        assert_eq!(u64::from_le_bytes(size), FLEXIBLE_MEMORY_SIZE);
        assert_eq!(
            hle_configured_flexible_memory_size(&ctx, &[0]),
            SCE_KERNEL_ERROR_EINVAL
        );
    }

    /// GetOpenPsId writes a stable, nonzero 16-byte id.
    #[test]
    fn get_open_ps_id_is_deterministic() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_get_open_ps_id(&ctx, &[0x100]), SCE_OK);
        assert_eq!(hle_get_open_ps_id(&ctx, &[0x200]), SCE_OK);
        let (mut a, mut b) = ([0u8; 16], [0u8; 16]);
        assert!(mem.read(0x100, &mut a));
        assert!(mem.read(0x200, &mut b));
        assert_eq!(a, b, "the id must be stable across calls");
        assert_ne!(a, [0u8; 16]);
        assert_eq!(hle_get_open_ps_id(&ctx, &[0]), SCE_KERNEL_ERROR_EINVAL);

        let registry = crate::HleRegistry::new();
        assert!(registry.is_implemented("libSceOpenPsId", "sceKernelGetOpenPsId"));
        assert!(registry.is_implemented("libkernel", "sceKernelGetOpenPsId"));
    }

    /// The APR id table resolves GetFileSize/GetFileStat to real host file
    /// metadata, and SubmitCommandBufferAndGetId writes its id out-param.
    #[test]
    fn apr_get_file_size_and_stat_use_the_registered_id_table() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // A real host file with a known size.
        let host = std::env::temp_dir().join(format!(
            "xps5x_apr_test_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&host, [0xAAu8; 1234]).expect("temp file");
        let id = kernel.appr_register_file("/app0/assets/level.bin", host.display().to_string());

        assert_eq!(hle_apr_get_file_size(&ctx, &[u64::from(id), 0x100]), SCE_OK);
        let mut size = [0u8; 8];
        assert!(mem.read(0x100, &mut size));
        assert_eq!(u64::from_le_bytes(size), 1234);

        assert_eq!(hle_apr_get_file_stat(&ctx, &[u64::from(id), 0x200]), SCE_OK);
        let mut stat = [0u8; ORBIS_STAT_SIZE];
        assert!(mem.read(0x200, &mut stat));
        // st_size at +72 in the 120-byte Orbis stat; regular-file mode at +8.
        assert_eq!(u64::from_le_bytes(stat[72..80].try_into().unwrap()), 1234);
        assert_eq!(
            u16::from_le_bytes(stat[8..10].try_into().unwrap()),
            ORBIS_MODE_REGULAR
        );

        // Unregistered id → ENOENT; NULL out-param → EINVAL.
        assert_eq!(
            hle_apr_get_file_size(&ctx, &[0xDEAD, 0x100]),
            SCE_KERNEL_ERROR_ENOENT
        );
        assert_eq!(
            hle_apr_get_file_stat(&ctx, &[0xDEAD, 0x200]),
            SCE_KERNEL_ERROR_ENOENT
        );
        assert_eq!(
            hle_apr_get_file_size(&ctx, &[u64::from(id), 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
        let _ = std::fs::remove_file(&host);

        // SubmitCommandBufferAndGetId: an empty (zero-length) command buffer
        // completes and the submission id lands in the third argument.
        assert!(mem.write(0x508, &0x600u64.to_le_bytes())); // cb data ptr
        assert!(mem.write(0x510, &0u64.to_le_bytes())); // cb size 0
        assert_eq!(hle_apr_submit_and_get_id(&ctx, &[0x500, 0, 0x300]), SCE_OK);
        let mut sub = [0u8; 4];
        assert!(mem.read(0x300, &mut sub));
        assert_ne!(u32::from_le_bytes(sub), 0);
        assert_eq!(
            hle_apr_submit_and_get_id(&ctx, &[0x500, 0, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
    }

    /// The ForEach resolve variants register paths in the APR table without
    /// writing through their unverified trailing arguments.
    #[test]
    fn apr_foreach_registers_paths_without_writing_out_arrays() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // One path entry pointing at an unresolvable guest path (no VFS mount
        // in the test kernel): the call still succeeds and touches nothing.
        assert!(mem.write(0x100, b"/app0/x.bin\0"));
        assert!(mem.write(0x200, &0x100u64.to_le_bytes()));
        let sentinel = 0x300u64;
        assert!(mem.write(sentinel, &[0xEEu8; 16]));
        assert_eq!(
            hle_apr_resolve_filepaths_to_ids_foreach(&ctx, &[0x200, 1, sentinel, sentinel]),
            SCE_OK
        );
        let mut probe = [0u8; 16];
        assert!(mem.read(sentinel, &mut probe));
        assert_eq!(probe, [0xEEu8; 16], "unverified out-args must be untouched");
        assert_eq!(
            hle_apr_resolve_filepaths_to_ids_foreach(&ctx, &[0, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
    }

    /// The ELF TLS ABI requires that a thread-local reached through the
    /// general-dynamic model (`__tls_get_addr`) resolve to the *same address* as
    /// the same variable reached through initial-exec — which the runtime
    /// resolves against the static block the module's `PT_TLS` was copied into.
    ///
    /// This used to hand back a freshly allocated, zero-initialized block
    /// instead, giving one variable two homes: a write through one model was
    /// invisible through the other, and only the static copy ever held the
    /// `.tdata` initializer. A PIC-built executable reaches its own
    /// thread-locals through *both*, so this is not a hypothetical.
    #[test]
    fn tls_get_addr_resolves_the_main_module_against_the_static_block() {
        const STATIC_BLOCK: u64 = 0x4000;
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x8000);
        let threads = StaticTlsThreads(STATIC_BLOCK);
        let mut ctx = test_ctx(&kernel, &mem, &alloc);
        ctx.guest_threads = &threads;

        write_tls_index(&mem, 0x100, MAIN_TLS_MODULE_ID, 0x18);
        assert_eq!(
            hle_tls_get_addr(&ctx, &[0x100]),
            STATIC_BLOCK + 0x18,
            "the main module's TLS must alias the static block, not a copy"
        );

        // Offset 0 is the case that mattered on Minecraft: its whole `.tdata` is
        // a single 8-byte value there.
        write_tls_index(&mem, 0x200, MAIN_TLS_MODULE_ID, 0);
        assert_eq!(hle_tls_get_addr(&ctx, &[0x200]), STATIC_BLOCK);
    }

    /// Only the main module has a static block. Anything else must still get
    /// bounded dynamic storage, and must not be handed the main module's.
    #[test]
    fn tls_get_addr_still_gives_other_modules_their_own_dynamic_block() {
        const STATIC_BLOCK: u64 = 0x4000;
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x8000);
        let threads = StaticTlsThreads(STATIC_BLOCK);
        let mut ctx = test_ctx(&kernel, &mem, &alloc);
        ctx.guest_threads = &threads;

        write_tls_index(&mem, 0x100, MAIN_TLS_MODULE_ID + 1, 0);
        let other = hle_tls_get_addr(&ctx, &[0x100]);
        assert_ne!(
            other, STATIC_BLOCK,
            "a dependency must not land in the executable's TLS block"
        );

        // And it is stable: the same module resolves to the same storage twice,
        // or a thread-local would lose its value between reads.
        write_tls_index(&mem, 0x200, MAIN_TLS_MODULE_ID + 1, 0);
        assert_eq!(hle_tls_get_addr(&ctx, &[0x200]), other);
    }

    /// A dependency in the process's static TLS layout must resolve into the
    /// static area at ITS slot — not the main module's, and not a dynamic
    /// block.
    ///
    /// This is the measured Minecraft crash the layout exists for: every
    /// `DTPMOD64` in the process used to resolve to module 1, so
    /// `libRenoirCore.PS5.prx`'s thread context pointer — written through
    /// initial-exec against its own slot — read back zero through
    /// general-dynamic, and the title's UI renderer dereferenced the null on
    /// every thread that touched it.
    #[test]
    fn tls_get_addr_resolves_a_static_layout_dependency_at_its_own_slot() {
        const AREA_BASE: u64 = 0x4000;
        // Variant-II layout: main at tp-0x20 (area offset 0x80 of a 0xa0-byte
        // area), the dependency below it at tp-0xa0 (area offset 0).
        const MAIN_AREA_OFF: u64 = 0x80;
        const DEP_AREA_OFF: u64 = 0;
        let kernel = xps5x_kernel::OrbisKernel::new();
        kernel.set_static_tls_area_offsets([(1u64, MAIN_AREA_OFF), (2u64, DEP_AREA_OFF)]);
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x8000);
        let threads = StaticTlsThreads(AREA_BASE);
        let mut ctx = test_ctx(&kernel, &mem, &alloc);
        ctx.guest_threads = &threads;

        write_tls_index(&mem, 0x100, 2, 0x8);
        assert_eq!(
            hle_tls_get_addr(&ctx, &[0x100]),
            AREA_BASE + DEP_AREA_OFF + 0x8,
            "a static-layout dependency must alias its own slot in the static area"
        );

        // The main module keeps its slot too — registered layout, not the
        // legacy module-1 fallback.
        write_tls_index(&mem, 0x200, 1, 0x10);
        assert_eq!(
            hle_tls_get_addr(&ctx, &[0x200]),
            AREA_BASE + MAIN_AREA_OFF + 0x10
        );

        // A module OUTSIDE the layout (runtime-loaded) still gets dynamic
        // storage, never a slice of the static area.
        write_tls_index(&mem, 0x300, 7, 0);
        let dynamic = hle_tls_get_addr(&ctx, &[0x300]);
        assert!(
            !(AREA_BASE..AREA_BASE + 0x100).contains(&dynamic),
            "a module outside the layout must not land in the static area"
        );
    }

    #[test]
    fn unwind_module_info_writes_the_complete_orbis_structure() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        kernel.set_unwind_modules(vec![xps5x_kernel::UnwindModuleInfo {
            name: "libc.prx".to_string(),
            start: 0x1000_0000,
            end: 0x1010_0000,
            eh_frame_hdr_addr: 0x100f_0000,
            eh_frame_addr: 0x100d_0000,
            eh_frame_size: 0x20_000,
            seg0_addr: 0x1000_0000,
            seg0_size: 0x90_000,
        }]);
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, &304u64.to_le_bytes()));

        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelGetModuleInfoForUnwind",
                &[0x1000_1234, 0, 0x100]
            ),
            Some(SCE_OK)
        );
        let mut info = [0u8; 304];
        assert!(mem.read(0x100, &mut info));
        assert_eq!(u64::from_le_bytes(info[0..8].try_into().unwrap()), 304);
        assert_eq!(&info[8..17], b"libc.prx\0");
        assert_eq!(
            u64::from_le_bytes(info[264..272].try_into().unwrap()),
            0x100f_0000
        );
        assert_eq!(
            u64::from_le_bytes(info[272..280].try_into().unwrap()),
            0x100d_0000
        );
        assert_eq!(
            u64::from_le_bytes(info[280..288].try_into().unwrap()),
            0x20_000
        );
        assert_eq!(
            u64::from_le_bytes(info[288..296].try_into().unwrap()),
            0x1000_0000
        );
        assert_eq!(
            u64::from_le_bytes(info[296..304].try_into().unwrap()),
            0x90_000
        );

        assert!(mem.write(0x100, &303u64.to_le_bytes()));
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelGetModuleInfoForUnwind",
                &[0x1000_1234, 0, 0x100]
            ),
            Some(SCE_KERNEL_ERROR_EINVAL)
        );
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelGetModuleInfoForUnwind",
                &[0x2000_0000, 0, 0x100]
            ),
            Some(SCE_KERNEL_ERROR_EINVAL),
            "caller size validation happens before address lookup"
        );
    }

    #[test]
    fn dlsym_resolves_a_real_module_export_by_symbol_name() {
        use sha1::{Digest, Sha1};
        const SALT: [u8; 16] = [
            0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1, 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5,
            0x52, 0x30,
        ];
        let mut hash = Sha1::new();
        hash.update(b"CreateDecoder");
        hash.update(SALT);
        let digest = hash.finalize();
        let nid = u64::from_le_bytes(digest[..8].try_into().unwrap());

        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let handle = kernel.register_lle_module(
            "plugin.prx".to_string(),
            0x1000_0000,
            0x10000,
            None,
            true,
            [(nid, 0x1000_4321)],
        );
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, b"CreateDecoder\0"));

        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelDlsym",
                &[handle as u64, 0x100, 0x200]
            ),
            Some(SCE_OK)
        );
        let mut out = [0u8; 8];
        assert!(mem.read(0x200, &mut out));
        assert_eq!(u64::from_le_bytes(out), 0x1000_4321);
    }

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
            ("libkernel", "sceKernelDebugWriteCppExceptionInfo"),
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
            ("libScePosix", "open"),
            ("libScePosix", "read"),
            ("libScePosix", "write"),
            ("libScePosix", "close"),
            ("libScePosix", "lseek"),
            ("libkernel", "__stack_chk_fail"),
        ] {
            assert!(
                registry.is_implemented(lib, name),
                "{lib}::{name} must be registered"
            );
        }

        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelDebugWriteCppExceptionInfo",
                &[0x1234, 1, 2, 3]
            ),
            Some(SCE_OK),
            "exception diagnostics must not replace the guest's unwind result"
        );

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
        let reserved = u64::from_le_bytes(reserved);
        assert_ne!(reserved, 0);

        // The reservation is immediately visible through the real 72-byte
        // VirtualQuery ABI, but is not committed until mapped.
        assert_eq!(
            hle_virtual_query(&ctx, &[reserved + 0x100, 0, 0x700, 72]),
            SCE_OK
        );
        let mut query = [0u8; 72];
        assert!(mem.read(0x700, &mut query));
        assert_eq!(
            u64::from_le_bytes(query[0..8].try_into().unwrap()),
            reserved
        );
        assert_eq!(
            u64::from_le_bytes(query[8..16].try_into().unwrap()),
            reserved + 0x4000
        );
        assert_eq!(query[32] & 0x10, 0, "reservation is not committed");

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
    fn rename_truncate_unlink_and_rmdir_mutate_the_host_through_the_vfs() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        // Nonzero base: the errno slot is arena-allocated, and address 0 would
        // read as "no errno slot".
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp = std::env::temp_dir().join(format!("xps5x-hle-fsmut-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("dir")).unwrap();
        std::fs::write(tmp.join("old.dat"), b"0123456789").unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        // sceKernelRename moves the real host file.
        assert!(mem.write(0x100, b"/app0/old.dat\0"));
        assert!(mem.write(0x140, b"/app0/new.dat\0"));
        assert_eq!(hle_sce_rename(&ctx, &[0x100, 0x140]), SCE_OK);
        assert!(!tmp.join("old.dat").exists());
        assert_eq!(std::fs::read(tmp.join("new.dat")).unwrap(), b"0123456789");
        // Renaming the now-missing source is a real SCE ENOENT.
        assert_eq!(hle_sce_rename(&ctx, &[0x100, 0x140]), 0x8002_0002);

        // sceKernelTruncate shortens it.
        assert_eq!(hle_sce_truncate(&ctx, &[0x140, 4]), SCE_OK);
        assert_eq!(std::fs::read(tmp.join("new.dat")).unwrap(), b"0123");

        // unlink (POSIX spelling) removes it; a second unlink is -1 + ENOENT.
        assert_eq!(hle_posix_unlink(&ctx, &[0x140]), SCE_OK);
        assert!(!tmp.join("new.dat").exists());
        assert_eq!(hle_posix_unlink(&ctx, &[0x140]), (-1i64) as u64);
        let errno_slot = hle_error_addr(&ctx, &[]);
        let mut errno = [0u8; 4];
        assert!(mem.read(errno_slot, &mut errno));
        assert_eq!(i32::from_le_bytes(errno), 2, "errno must hold ENOENT");

        // rmdir removes the empty directory (SCE spelling).
        assert!(mem.write(0x180, b"/app0/dir\0"));
        assert_eq!(hle_sce_rmdir(&ctx, &[0x180]), SCE_OK);
        assert!(!tmp.join("dir").exists());
        assert_eq!(hle_sce_rmdir(&ctx, &[0x180]), 0x8002_0002);

        // Chmod/Utimes accept an existing path and reject a missing one.
        assert!(mem.write(0x1c0, b"/app0/\0"));
        assert_eq!(hle_path_metadata_accept(&ctx, &[0x1c0, 0o755]), SCE_OK);
        assert_eq!(hle_path_metadata_accept(&ctx, &[0x180, 0o755]), 0x8002_0002);

        // Sync is a global no-op success.
        assert_eq!(hle_sce_sync(&ctx, &[]), SCE_OK);

        // A traversing path never reaches the host.
        assert!(mem.write(0x200, b"/app0/../escape\0"));
        assert_ne!(hle_sce_unlink(&ctx, &[0x200]), SCE_OK);

        let registry = HleRegistry::new();
        for name in [
            "sceKernelRename",
            "sceKernelTruncate",
            "sceKernelSync",
            "sceKernelChmod",
            "sceKernelUtimes",
            "unlink",
            "rmdir",
            "sceKernelUnlink",
            "sceKernelRmdir",
            "_nanosleep",
            "getrusage",
            "signal",
            "sceKernelMlock",
            "sceKernelMapDirectMemory2",
            "sceKernelInternalMemoryGetModuleSegmentInfo",
            "sceKernelAddWriteEvent",
            "sceKernelDeleteWriteEvent",
            "scePthreadMutexTimedlock",
        ] {
            assert!(
                registry.is_implemented("libkernel", name),
                "libkernel::{name} must be registered"
            );
        }
        for name in ["mprotect", "munmap", "select", "accept", "listen", "recv"] {
            assert!(
                registry.is_implemented("libScePosix", name),
                "libScePosix::{name} must be registered"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn getrusage_zero_fills_and_map_direct_memory2_reorders_arguments() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100000);
        let alloc = crate::TestAllocator::new(0x40000);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // getrusage zero-fills the full 144-byte struct.
        assert!(mem.write(0x400, &[0xAAu8; RUSAGE_SIZE]));
        assert_eq!(hle_getrusage(&ctx, &[0, 0x400]), 0);
        let mut usage = [0xAAu8; RUSAGE_SIZE];
        assert!(mem.read(0x400, &mut usage));
        assert_eq!(usage, [0u8; RUSAGE_SIZE]);
        // NULL pointer → -1 (with errno EFAULT).
        assert_eq!(hle_getrusage(&ctx, &[0, 0]), (-1i64) as u64);

        // MapDirectMemory2: allocate direct memory, then map it with the
        // extra `type` argument at position 2 — must succeed exactly like
        // sceKernelMapDirectMemory with the trailing args shifted one right.
        assert_eq!(
            hle_allocate_direct_memory(&ctx, &[0, u64::MAX, 0x8000, 0, 0, 0x600]),
            SCE_OK
        );
        let mut phys_bytes = [0u8; 8];
        assert!(mem.read(0x600, &mut phys_bytes));
        let phys = u64::from_le_bytes(phys_bytes);
        assert!(mem.write(0x700, &0u64.to_le_bytes())); // addrOut in/out = 0
        assert_eq!(
            hle_map_direct_memory2(&ctx, &[0x700, 0x8000, /*type*/ 3, 0x3, 0, phys, 0]),
            SCE_OK
        );
        let mut mapped = [0u8; 8];
        assert!(mem.read(0x700, &mut mapped));
        assert_eq!(u64::from_le_bytes(mapped), phys);

        // signal() reports SIG_DFL and Mlock succeeds.
        assert_eq!(hle_posix_signal(&ctx, &[11, 0x1234]), 0);
        assert_eq!(hle_mlock(&ctx, &[0x1000, 0x1000]), SCE_OK);
        assert_eq!(
            hle_internal_get_module_segment_info(&ctx, &[0x400]),
            SCE_KERNEL_ERROR_EINVAL
        );
    }

    #[test]
    fn file_io_open_read_seek_close_against_a_real_host_file() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x800);
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

        // Plain/POSIX open returns -1 and sets per-thread errno, whereas the
        // sceKernel spelling returns the kernel error encoding directly.
        assert_eq!(hle_posix_open(&ctx, &[0x300, 0, 0]) as i64, -1);
        let errno = hle_error_addr(&ctx, &[]);
        let mut errno_bytes = [0u8; 4];
        assert!(mem.read(errno, &mut errno_bytes));
        assert_eq!(i32::from_le_bytes(errno_bytes), 2);
        assert_eq!(hle_sce_open(&ctx, &[0x300, 0, 0]), 0x8002_0002);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `pread` reads at an absolute offset WITHOUT moving the cursor — a
    /// streaming loader interleaves preads with sequential reads on one fd.
    /// Measured: ASTRO.BOT's asset streamer imports sceKernelPread.
    #[test]
    fn pread_reads_at_offset_without_moving_the_cursor() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp = std::env::temp_dir().join(format!("xps5x-hle-pread-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("data.bin"), b"HELLO_WORLD").unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        assert!(mem.write(0x100, b"/app0/data.bin\0"));
        let fd = hle_open(&ctx, &[0x100, 0, 0]);
        assert!((fd as i64) >= 3);

        // pread "WORLD" at offset 6, then a sequential read still starts at 0.
        assert_eq!(hle_pread(&ctx, &[fd, 0x200, 5, 6]), 5);
        let mut buf = [0u8; 5];
        assert!(mem.read(0x200, &mut buf));
        assert_eq!(&buf, b"WORLD");
        assert_eq!(hle_read(&ctx, &[fd, 0x210, 5]), 5);
        let mut buf2 = [0u8; 5];
        assert!(mem.read(0x210, &mut buf2));
        assert_eq!(&buf2, b"HELLO");

        // Reads at/past EOF are a valid 0-byte short read; a partial tail is
        // returned short, not padded.
        assert_eq!(hle_pread(&ctx, &[fd, 0x220, 5, 11]), 0);
        assert_eq!(hle_pread(&ctx, &[fd, 0x220, 5, 9]), 2);
        // Negative offset → EINVAL; bad fd → EBADF.
        assert_eq!(hle_pread(&ctx, &[fd, 0x220, 5, u64::MAX]) as i64, -22);
        assert_eq!(hle_pread(&ctx, &[999, 0x220, 5, 0]) as i64, -9);

        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
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
        let registry = HleRegistry::new();
        let stale_arg4 = [0xA5u8; 8];
        assert!(mem.write(0x200, &stale_arg4));
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelGetdents",
                &[fd, 0x400, 1024, 0x200]
            ),
            Some(512),
            "the registry must retain the VFS handler rather than a later stub"
        );
        let mut after = [0u8; 8];
        assert!(mem.read(0x200, &mut after));
        assert_eq!(
            after, stale_arg4,
            "three-argument sceKernelGetdents must not treat stale RCX as getdirentries basep"
        );
        let mut first = [0u8; 512];
        assert!(mem.read(0x400, &mut first));
        let first_len = u16::from_le_bytes(first[4..6].try_into().unwrap()) as usize;
        assert_eq!(first_len, 512);
        assert!(matches!(first[6], 4 | 8));
        assert_ne!(first[7], 0);
        assert_eq!(hle_getdents(&ctx, &[fd, 0x400, 1024], false), 512);
        assert_eq!(hle_getdents(&ctx, &[fd, 0x400, 1024], false), 512);
        assert_eq!(hle_getdents(&ctx, &[fd, 0x400, 1024], false), 512);
        assert_eq!(hle_getdents(&ctx, &[fd, 0x400, 1024], false), 0);
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

    /// With no address requested, a mapped direct-memory range must stay the
    /// *same storage* the guest allocated rather than a freshly allocated
    /// region, which would pass a naive "addr is non-zero" check while silently
    /// giving the guest a view disconnected from its own direct memory — and
    /// would leak once `sceKernelReleaseDirectMemory` freed the physical range.
    #[test]
    fn mapping_direct_memory_without_a_requested_address_publishes_the_allocated_storage() {
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

        // No address requested: the out-param starts zeroed.
        assert!(mem.write(0x108, &0u64.to_le_bytes()));
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
        let mapped = u64::from_le_bytes(bytes);
        assert_eq!(
            mapped, physical,
            "the mapping must publish the allocated direct memory itself"
        );
        assert!(kernel.memory.is_mapped(mapped));
    }

    /// A guest that supplies an address expects the mapping THERE. Minecraft
    /// asks for one and then writes to it without ever reading the out-param
    /// back, so publishing anywhere else leaves it writing to unmapped memory
    /// (measured: `memset: dst 0x100102000000 out of bounds`).
    #[test]
    fn mapping_direct_memory_at_a_requested_address_honors_that_address() {
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

        let requested = 0x20_0000u64;
        assert!(mem.write(0x108, &requested.to_le_bytes()));
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
        assert_eq!(
            u64::from_le_bytes(bytes),
            requested,
            "a requested address must be honored, not silently relocated"
        );
        assert!(kernel.memory.is_mapped(requested));
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
