//! HLE libkernel — Core kernel interface re-implementation.
//!
//! Clean-room re-implementation of the PS5 `libkernel.sprx` exports. Function
//! *names* below are factual PS5 API identifiers (not copyrightable); every
//! implementation is original.
//!
//! ## Stub status
//!
//! Every HLE call now gets an [`crate::HleContext`] (a live
//! [`raeen_kernel::OrbisKernel`] plus guest-memory and guest-allocator
//! access), so functions *can* do real work. Most functions below still
//! just log the call and return a plausible value (an `SCE_OK`-style `0`,
//! a fake handle, or a fake address/size) — thread creation, event queues,
//! and most out-parameters still aren't backed by real state.
//! `sceKernelAllocateDirectMemory`, `sceKernelMapFlexibleMemory`, and
//! `sceKernelMmap` are the exceptions: they route through `ctx.alloc.mmap`
//! (the arena's mmap region, in production — `raeen-runtime`'s
//! `GuestArena`) and record the mapping in `ctx.kernel.memory` so
//! `is_mapped`/`region_containing` see it, writing the resulting address
//! through their out-parameter (where the ABI has one) via `ctx.mem`.
//! `sceKernelMunmap` mirrors this on the way out. Broadening the rest is
//! future work, not a limitation of the dispatch signature anymore.

use crate::{
    GuestAccess, GuestAddress, GuestCallCompletion, GuestCallRequest, GuestRange, HleContext,
    HleRegistry, MAX_HLE_BULK_BYTES,
};
use std::sync::atomic::Ordering;
use tracing::{debug, info, warn};

/// `SCE_OK` — the PS5 convention for "this call succeeded".
const SCE_OK: u64 = 0;

/// `MAP_FIXED` — the guest demands the address it passed in, and the kernel
/// must refuse rather than relocate. Without it a requested address is only a
/// hint (shadPS4 `MemoryManager::MapMemory` takes its `SearchFree` path when
/// `Fixed` is clear).
const MAP_FIXED: u32 = 0x10;

/// Cap on how many bytes one `write` call will copy out of guest memory —
/// keeps a wild `count` from ballooning a host buffer. Generous for
/// console output.
const WRITE_CHUNK_BYTES: u64 = 1 << 20; // 1 MiB

/// Per-module dynamic TLS storage used by `__tls_get_addr`. This matches the
/// compatibility-layer size used by SharpEmu and is deliberately bounded so
/// a corrupt guest descriptor cannot request an unbounded host allocation.
const DYNAMIC_TLS_BLOCK_SIZE: u64 = 0x1_0000;

/// The TLS module ID the linker writes into the main module's `DTPMOD64`
/// relocation slots, and therefore the id a general-dynamic access to the
/// executable's own thread-locals arrives here with.
///
/// Must equal `raeen_firmware`'s `MAIN_TLS_MODULE_ID`. It is duplicated rather
/// than imported because the dependency runs the other way — `raeen-firmware`
/// depends on this crate to resolve NIDs at link time, so importing it back
/// would be a cycle. Pinned against the linker's value by
/// `main_tls_module_id_matches_the_linkers` in `raeen-firmware`.
const MAIN_TLS_MODULE_ID: u64 = 1;

/// `EBADF` as the sign-extended negative return `write(2)` produces on a
/// bad descriptor (the PS5's BSD libc returns `-1` and sets `errno`;
/// `sceKernelWrite` returns a negative error directly — either way the
/// caller sees "negative", which is the honest signal here).
const WRITE_EBADF: u64 = (-9i64) as u64;

fn trace_file_io(ctx: &HleContext) -> bool {
    std::env::var("RAEEN_TRACE_FILE_IO_AFTER_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u128>().ok())
        .is_some_and(|after_ms| ctx.kernel.uptime().as_millis() >= after_ms)
}

/// Real `write(fd, buf, count)` / `sceKernelWrite` for the console
/// descriptors (M1-C): fd 1 (stdout) and fd 2 (stderr) copy `count` guest
/// bytes (streamed through [`WRITE_CHUNK_BYTES`] staging) to the kernel
/// [`raeen_kernel::Console`] and return `count`. Any other fd has no
/// backing file table yet — logged loudly, returns [`WRITE_EBADF`], never
/// pretends to have written.
fn hle_write(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0);
    let buf = args.get(1).copied().unwrap_or(0);
    let count = args.get(2).copied().unwrap_or(0);
    debug!("write(fd={fd}, buf={buf:#x}, count={count:#x})");

    if count > MAX_HLE_BULK_BYTES {
        warn!(
            "write: refusing attacker-sized transfer count={count:#x}, \
             maximum={MAX_HLE_BULK_BYTES:#x}"
        );
        return FILE_EINVAL;
    }
    let Some(range) = GuestRange::new(GuestAddress::new(buf), count) else {
        return FILE_EFAULT;
    };
    if !ctx.mem.validate_range(range, GuestAccess::Read) {
        warn!("write: guest buffer [{buf:#x}, +{count:#x}) is not readable — EFAULT");
        return FILE_EFAULT;
    }

    // The 1 MiB value bounds host staging only. Stream the complete valid
    // guest request so large save records are not silently truncated.
    let mut staging = vec![0u8; count.min(WRITE_CHUNK_BYTES) as usize];
    let mut transferred = 0u64;
    while transferred < count {
        let chunk_len = (count - transferred).min(WRITE_CHUNK_BYTES) as usize;
        let chunk = &mut staging[..chunk_len];
        if !ctx.mem.read(buf + transferred, chunk) {
            return if transferred == 0 {
                FILE_EFAULT
            } else {
                transferred
            };
        }

        let written = if fd == 1 || fd == 2 {
            ctx.kernel.console.write_bytes(chunk);
            if chunk
                .windows(b"Fatal error".len())
                .any(|window| window == b"Fatal error")
                || chunk
                    .windows(b"unreachable code".len())
                    .any(|window| window == b"unreachable code")
            {
                report_guest_fatal_console(ctx, chunk);
            }
            chunk_len
        } else {
            match ctx.services.write(fd as i32, chunk) {
                Ok(written) if written <= chunk_len => written,
                Ok(_) => {
                    return if transferred == 0 {
                        FILE_EIO
                    } else {
                        transferred
                    };
                }
                Err(error) => {
                    warn!("write: fd {fd} failed after {transferred:#x} byte(s): {error}");
                    return if transferred == 0 {
                        WRITE_EBADF
                    } else {
                        transferred
                    };
                }
            }
        };
        transferred += written as u64;
        if written < chunk_len {
            break;
        }
    }
    transferred
}

/// Surface the call boundary around a fatal message emitted by a guest
/// runtime. The recent-call ring is populated only for diagnostic runs
/// (`RAEEN_TRACE_EINVAL` or `RAEEN_TRAP_CXA_THROW`), so normal execution pays
/// only the bounded substring check in [`hle_write`].
fn report_guest_fatal_console(ctx: &HleContext, bytes: &[u8]) {
    let thread = ctx.guest_threads.current_thread();
    let message = String::from_utf8_lossy(bytes);
    let recent = ctx
        .kernel
        .recent_hle_calls
        .get(&thread)
        .map(|ring| ring.lock().iter().cloned().collect::<Vec<_>>().join(" <- "))
        .unwrap_or_default();
    let mut chain = Vec::new();
    for slot in 0..128u64 {
        let mut word = [0u8; 8];
        if !ctx
            .mem
            .read(ctx.caller_rsp.wrapping_add(slot * 8), &mut word)
        {
            break;
        }
        let address = u64::from_le_bytes(word);
        if (0x1000_0000_0000..0x1000_2000_0000).contains(&address) {
            chain.push(format!("{address:#x}"));
            if chain.len() >= 24 {
                break;
            }
        }
    }
    warn!(
        "guest fatal console on thread {thread}: {:?}; writer_ra={:#x}; recent=[{}]; \
         stack=[{}]",
        message.trim(),
        ctx.caller_return_addr,
        recent,
        chain.join(" "),
    );
}

/// `EBADF` (bad file descriptor) as a sign-extended negative return.
const FILE_EBADF: u64 = (-9i64) as u64;
/// `ENOENT` (no such file) as a sign-extended negative return.
const FILE_ENOENT: u64 = (-2i64) as u64;
/// `EFAULT` (bad address) as a sign-extended negative return.
const FILE_EFAULT: u64 = (-14i64) as u64;
/// `EINVAL` (invalid argument) as a sign-extended negative return.
const FILE_EINVAL: u64 = (-22i64) as u64;
/// `EIO` (i/o error) as a sign-extended negative return.
const FILE_EIO: u64 = (-5i64) as u64;
/// `ENOMEM` (out of memory) as a sign-extended negative return.
const FILE_ENOMEM: u64 = (-12i64) as u64;
/// `EACCES` (permission denied) as a sign-extended negative return.
const FILE_EACCES: u64 = (-13i64) as u64;
/// Cap on a single `read` transfer into guest memory (bounds host staging).
/// Per-iteration transfer size for `read`/`pread`.
///
/// A request may be much larger (up to [`MAX_HLE_BULK_BYTES`]); chunking keeps
/// alternate/test `GuestMemory` backends from allocating the whole request at
/// once. This is not an ABI-visible cap. GTA V reads its 25,445,380-byte
/// `rpf.cache` in one call, so silently clamping the call to 16 MiB drops the
/// tail of the archive index and makes valid packaged shaders appear missing.
const READ_CHUNK_BYTES: u64 = 1 << 20; // 1 MiB

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

    // RE probe (RAEEN_TRACE_UI): the caller of the routes.json open is the
    // Ore-UI/Gameface route-table processor — on the UI-INIT path (unlike the
    // render chain, which is a proven dead-end). Its return-addr + guest-stack
    // chain point at the code that decides whether to navigate to a route; aim
    // `raeen --disas` there to find the never-taken CreateView/LoadURL branch.
    if std::env::var_os("RAEEN_TRACE_UI").is_some() && path.contains("routes.json") {
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
        // Dump the UI-manager singleton's live vtable so `raeen --disas` can map
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

    // Let the VFS perform the one authoritative resolve+open. The old path
    // resolved here to preflight existence and then `VfsSubsystem::open`
    // resolved the same path again. Commercial titles issue thousands of
    // successful asset opens while holding their own streaming locks, so that
    // duplicate sandbox walk/canonicalization turned world transitions into
    // multi-second mutex convoys.
    //
    // A missing file is ENOENT unless the guest passed O_CREAT (the VFS creates
    // it). O_CREAT is bit 0x200 in the Orbis/BSD flag set.
    const O_CREAT: i32 = 0x200;
    let creating = flags & O_CREAT != 0;
    let built_in_device = matches!(path.as_str(), "/dev/random" | "/dev/urandom");

    match ctx.services.open(&path, flags, mode) {
        Ok(fd) => {
            // Name every SUCCESSFUL open too. Only failures were logged before,
            // which makes "the title never touched this file" and "it opened it
            // fine" indistinguishable in a boot trace — the exact ambiguity that
            // hid whether the Ore-UI menu HTML is ever loaded.
            debug!("open: '{path}' -> fd {fd}");
            if trace_file_io(ctx) {
                info!(
                    elapsed_ms = ctx.kernel.uptime().as_millis(),
                    "file trace: open path='{path}' flags={flags:#x} mode={mode:#o} -> fd={fd}"
                );
            }
            if built_in_device && std::env::var_os("RAEEN_TRACE_ENTROPY").is_some() {
                info!("entropy device open: path='{path}' flags={flags:#x} -> fd={fd}");
            }
            fd as u64
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                let trace_missing =
                    trace_file_io(ctx) || std::env::var_os("RAEEN_TRACE_MISSING_FILES").is_some();
                let is_font = std::path::Path::new(&path)
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        matches!(
                            extension.to_ascii_lowercase().as_str(),
                            "otf" | "ttf" | "ttc"
                        )
                    });
                // Resolve a failed path a second time only for the two
                // diagnostics that need its host spelling: font substitution
                // and explicit file tracing. Ordinary optional-file probes
                // retain the single VFS sandbox walk above.
                let host = (!built_in_device && (is_font || trace_missing))
                    .then(|| ctx.kernel.filesystem.resolve_path(&path))
                    .flatten();

                // Font-file fallback. The title reads its fonts with its OWN
                // OpenType renderer and null-dereferences if an open fails, yet
                // references variants/system fonts it does not ship. Substitute
                // a shipped sibling so the renderer still parses valid tables.
                if !creating
                    && let Some(host) = host.as_deref()
                    && let Some(fb_name) = font_fallback_sibling(host)
                    && let Some(slash) = path.rfind('/')
                {
                    let fb_path = format!("{}/{fb_name}", &path[..slash]);
                    warn!("open: '{path}' missing — substituting shipped font '{fb_path}'");
                    return match ctx.services.open(&fb_path, flags, mode) {
                        Ok(fd) => fd as u64,
                        Err(error) => {
                            warn!("open: font substitute '{fb_path}' failed: {error} — ENOENT");
                            FILE_ENOENT
                        }
                    };
                }

                // Missing optional files are normal guest-handled probes. Keep
                // production logging quiet; the opt-in trace retains the full
                // guest/host path pair used by compatibility diagnostics.
                if let Some(host) = host {
                    debug!(
                        "open: '{path}' → '{}' does not exist (no O_CREAT) — ENOENT",
                        host.display()
                    );
                    if trace_missing {
                        info!(
                            elapsed_ms = ctx.kernel.uptime().as_millis(),
                            host = %host.display(),
                            "file trace: missing path='{path}' flags={flags:#x} mode={mode:#o} -> ENOENT"
                        );
                    }
                } else {
                    debug!("open: '{path}' does not exist — ENOENT");
                    if e.to_string().contains("path is not mounted") {
                        warn!("open: '{path}' matches no VFS mount — ENOENT");
                    }
                }
                return FILE_ENOENT;
            }

            // Map the host error to the matching errno instead of calling every
            // failure ENOENT. An out-of-memory open — the eager whole-file
            // buffer failing under host commit pressure — reported as "file not
            // found" cost a full debugging cycle chasing a missing file that was
            // right there on disk. A title also branches on errno (retry vs give
            // up), so the distinction is load-bearing, not cosmetic.
            let errno = match e.kind() {
                std::io::ErrorKind::NotFound => FILE_ENOENT,
                std::io::ErrorKind::PermissionDenied => FILE_EACCES,
                std::io::ErrorKind::OutOfMemory => FILE_ENOMEM,
                _ => FILE_EIO,
            };
            warn!("open: '{path}' failed: {e} (errno {})", -(errno as i64));
            errno
        }
    }
}

/// Real `read(fd, buf, count)` / `sceKernelRead` (VFS-backed): reads up to
/// `count` bytes from the open descriptor and writes them into the guest
/// buffer, returning the byte count actually read (0 at EOF). Large valid
/// requests are streamed in bounded chunks instead of being silently
/// truncated. Bad fd → `EBADF`; unwritable buffer → `EFAULT`.
fn hle_read(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let buf = args.get(1).copied().unwrap_or(0);
    let count = args.get(2).copied().unwrap_or(0);
    debug!("read(fd={fd}, buf={buf:#x}, count={count:#x})");
    let traced_path = trace_file_io(ctx)
        .then(|| ctx.kernel.filesystem.open_path(fd))
        .flatten();
    let entropy = ctx.kernel.filesystem.is_random_device(fd);
    let traced_guard = entropy
        .then(|| {
            std::env::var("RAEEN_TRACE_GUARD_ADDR")
                .ok()
                .and_then(|value| {
                    u64::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok()
                })
        })
        .flatten();
    let read_guard = |address| {
        let mut bytes = [0u8; 8];
        ctx.mem
            .read(address, &mut bytes)
            .then(|| u64::from_le_bytes(bytes))
    };
    let guard_before = traced_guard.and_then(read_guard);

    if count > MAX_HLE_BULK_BYTES {
        return FILE_EINVAL;
    }
    let Some(range) = GuestRange::new(GuestAddress::new(buf), count) else {
        return FILE_EFAULT;
    };
    if !ctx.mem.validate_range(range, GuestAccess::Write) {
        warn!("read: guest buffer {buf:#x} (+{count}) not writable — EFAULT");
        return FILE_EFAULT;
    }

    let mut transferred = 0u64;
    while transferred < count {
        let chunk_len = (count - transferred).min(READ_CHUNK_BYTES) as usize;
        let Some(chunk_addr) = buf.checked_add(transferred) else {
            return if transferred == 0 {
                FILE_EFAULT
            } else {
                transferred
            };
        };
        let mut read_result = None;
        let guest_fill = ctx.mem.fill_write(chunk_addr, chunk_len, &mut |out| {
            let result = ctx.services.read_into(fd, out);
            let read = result.as_ref().copied().unwrap_or(0);
            read_result = Some(result);
            read
        });
        let Some(filled) = guest_fill else {
            return if transferred == 0 {
                FILE_EFAULT
            } else {
                transferred
            };
        };
        match read_result.unwrap_or(Ok(filled)) {
            Ok(read) if read <= chunk_len => {
                debug_assert_eq!(filled, read);
                transferred += read as u64;
                if read < chunk_len {
                    break;
                }
            }
            Ok(_) => {
                return if transferred == 0 {
                    FILE_EIO
                } else {
                    transferred
                };
            }
            Err(error) => {
                if transferred == 0 {
                    warn!("read: fd {fd} failed: {error} — EBADF");
                    return FILE_EBADF;
                }
                break;
            }
        }
    }

    if let Some(path) = traced_path.as_deref() {
        info!(
            elapsed_ms = ctx.kernel.uptime().as_millis(),
            "file trace: read fd={fd} path='{path}' guest_buf={buf:#x} \
             count={count:#x} -> {transferred:#x} byte(s)"
        );
    }
    if entropy && std::env::var_os("RAEEN_TRACE_ENTROPY").is_some() {
        info!(
            "entropy device read: fd={fd} guest_buf={buf:#x} count={count:#x} -> {} \
             byte(s), guard={:#x?}->{:#x?}",
            transferred,
            guard_before,
            traced_guard.and_then(read_guard),
        );
    }
    transferred
}

fn pread_error(error: &std::io::Error) -> u64 {
    if error.kind() == std::io::ErrorKind::NotFound {
        FILE_EBADF
    } else {
        FILE_EINVAL
    }
}

fn pread_chunk(
    ctx: &HleContext,
    fd: i32,
    guest_addr: u64,
    len: usize,
    offset: u64,
) -> Result<usize, u64> {
    let mut read_result = None;
    let guest_fill = ctx.mem.fill_write(guest_addr, len, &mut |out| {
        let result = ctx.kernel.filesystem.pread_into(fd, out, offset);
        let read = result.as_ref().copied().unwrap_or(0);
        read_result = Some(result);
        read
    });
    let Some(filled) = guest_fill else {
        return Err(FILE_EFAULT);
    };
    match read_result.unwrap_or(Ok(filled)) {
        Ok(read) if read <= len => {
            debug_assert_eq!(filled, read);
            Ok(read)
        }
        Ok(_) => Err(FILE_EIO),
        Err(error) => Err(pread_error(&error)),
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
    let count = args.get(2).copied().unwrap_or(0);
    let offset = args.get(3).copied().unwrap_or(0);
    debug!("pread(fd={fd}, buf={buf:#x}, count={count:#x}, offset={offset:#x})");
    let traced_path = trace_file_io(ctx)
        .then(|| ctx.kernel.filesystem.open_path(fd))
        .flatten();

    if (offset as i64) < 0 {
        return FILE_EINVAL;
    }
    if count > MAX_HLE_BULK_BYTES {
        return FILE_EINVAL;
    }
    let Some(range) = GuestRange::new(GuestAddress::new(buf), count) else {
        return FILE_EFAULT;
    };
    if !ctx.mem.validate_range(range, GuestAccess::Write) {
        warn!("pread: guest buffer {buf:#x} (+{count}) not writable — EFAULT");
        return FILE_EFAULT;
    }

    let mut transferred = 0u64;
    while transferred < count {
        let chunk_len = (count - transferred).min(READ_CHUNK_BYTES) as usize;
        let Some(chunk_addr) = buf.checked_add(transferred) else {
            return if transferred == 0 {
                FILE_EFAULT
            } else {
                transferred
            };
        };
        let Some(chunk_offset) = offset.checked_add(transferred) else {
            return if transferred == 0 {
                FILE_EINVAL
            } else {
                transferred
            };
        };
        match pread_chunk(ctx, fd, chunk_addr, chunk_len, chunk_offset) {
            Ok(read) => {
                transferred += read as u64;
                if read < chunk_len {
                    break;
                }
            }
            Err(error) if transferred == 0 => return error,
            Err(_) => break,
        }
    }

    if let Some(path) = traced_path.as_deref() {
        info!(
            elapsed_ms = ctx.kernel.uptime().as_millis(),
            "file trace: pread fd={fd} path='{path}' guest_buf={buf:#x} \
             count={count:#x} offset={offset:#x} -> {transferred:#x} byte(s)"
        );
    }
    transferred
}

fn hle_posix_pread(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_pread(ctx, args))
}

fn hle_sce_pread(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_pread(ctx, args))
}

/// Real `pwrite(fd, buf, nbyte, offset)` / `sceKernelPwrite` (VFS-backed):
/// writes up to `nbyte` bytes at absolute `offset` WITHOUT moving the
/// descriptor's cursor — the write-side twin of [`hle_pread`]. The VFS
/// write-back buffer makes the bytes durable on fsync/close, exactly as
/// `write` does. Bad fd (or a read-only one) → `EBADF`; negative offset →
/// `EINVAL`; unreadable guest buffer → `EFAULT`.
fn hle_pwrite(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let buf = args.get(1).copied().unwrap_or(0);
    let count = args.get(2).copied().unwrap_or(0);
    let offset = args.get(3).copied().unwrap_or(0);
    debug!("pwrite(fd={fd}, buf={buf:#x}, count={count:#x}, offset={offset:#x})");

    if (offset as i64) < 0 {
        return FILE_EINVAL;
    }
    if count > MAX_HLE_BULK_BYTES {
        return FILE_EINVAL;
    }
    let Some(range) = GuestRange::new(GuestAddress::new(buf), count) else {
        return FILE_EFAULT;
    };
    if !ctx.mem.validate_range(range, GuestAccess::Read) {
        warn!("pwrite: guest buffer [{buf:#x}, +{count:#x}) is not readable — EFAULT");
        return FILE_EFAULT;
    }

    let mut staging = vec![0u8; count.min(WRITE_CHUNK_BYTES) as usize];
    let mut transferred = 0u64;
    while transferred < count {
        let chunk_len = (count - transferred).min(WRITE_CHUNK_BYTES) as usize;
        let chunk = &mut staging[..chunk_len];
        if !ctx.mem.read(buf + transferred, chunk) {
            return if transferred == 0 {
                FILE_EFAULT
            } else {
                transferred
            };
        }
        let Some(chunk_offset) = offset.checked_add(transferred) else {
            return if transferred == 0 {
                FILE_EINVAL
            } else {
                transferred
            };
        };
        match ctx.kernel.filesystem.pwrite(fd, chunk, chunk_offset) {
            Ok(written) if written <= chunk_len => {
                transferred += written as u64;
                if written < chunk_len {
                    break;
                }
            }
            Ok(_) => {
                return if transferred == 0 {
                    FILE_EIO
                } else {
                    transferred
                };
            }
            Err(error) => {
                use std::io::ErrorKind;
                let result = match error.kind() {
                    ErrorKind::NotFound | ErrorKind::PermissionDenied => FILE_EBADF,
                    _ => FILE_EINVAL,
                };
                return if transferred == 0 {
                    result
                } else {
                    transferred
                };
            }
        }
    }
    transferred
}

fn hle_posix_pwrite(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_pwrite(ctx, args))
}

fn hle_sce_pwrite(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_pwrite(ctx, args))
}

/// Real `ftruncate(fd, length)` / `sceKernelFtruncate` (VFS-backed): resize
/// the OPEN descriptor to `length` bytes. The VFS resizes the descriptor's
/// write-back buffer (drop the tail / zero-fill the extension), so the new
/// length survives the flush-on-close — truncating the host file out from
/// under a dirty buffer would be undone the moment it flushed. Bad fd (or a
/// read-only one) → `EBADF`; negative length → `EINVAL`.
fn hle_ftruncate(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0) as i32;
    let length = args.get(1).copied().unwrap_or(0);
    debug!("ftruncate(fd={fd}, length={length:#x})");

    if (length as i64) < 0 {
        return FILE_EINVAL;
    }
    match ctx.kernel.filesystem.ftruncate(fd, length) {
        Ok(()) => SCE_OK,
        Err(e) => {
            use std::io::ErrorKind;
            match e.kind() {
                ErrorKind::NotFound | ErrorKind::PermissionDenied => FILE_EBADF,
                _ => io_error_to_file_result(&e),
            }
        }
    }
}

fn hle_posix_ftruncate(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_ftruncate(ctx, args))
}

fn hle_sce_ftruncate(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_ftruncate(ctx, args))
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

/// Fixed size of the `sceKernelAioInitializeParam(param)` scheduler parameter
/// block, done **synchronously** (mission's measured call: Dragon Ball hits
/// this right after its engine allocator comes up). There is no async AIO
/// backend yet — the honest model is "the init succeeded; requests complete
/// immediately when they arrive". Zero the block (a clean, defined default)
/// rather than leaving guest garbage the title might read back as a schedule.
const AIO_INIT_PARAM_SIZE: usize = 0x3c;

/// The ABI takes only `param`; `args[1]` is stale register state. SharpEmu's
/// independently-derived Gen4/Gen5 layout fixes the block at 0x3c bytes, which
/// matches the size Until Dawn immediately passes to `InitializeImpl`.
///
/// # Why the zero-fill is conditional
///
/// `size` is a guest register, and this is the only place in this file that
/// turns a caller-supplied size into a **write length** rather than into a gate
/// (compare `sceKernelVirtualQuery`, which checks `info_size >= 72` and then
/// writes exactly 72). A block that is really a caller *local* — or a register
/// that is stale rather than a real size — would let this memset up to 64 KiB
/// of the caller's frame, taking out its locals, saved registers, and
/// `__stack_chk_guard` canary. So the bulk clear happens only for a block that
/// is provably not a caller local ([`crate::out_buffer`]); on a stack block the
/// HLE writes nothing and still reports success, because zeroing a parameter
/// block the caller filled in itself is a write the ABI never promised.
fn hle_aio_initialize_param(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    debug!("sceKernelAioInitializeParam(param={param:#x})");
    if param == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    // Fixed ABI size (never the guest register), AND only when the block
    // is provably not a caller local: rules 3 and 4 together.
    if !ctx.zero_out_object(
        "libkernel::sceKernelAioInitializeParam",
        param,
        AIO_INIT_PARAM_SIZE,
        0,
    ) {
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

/// Record types inside an AMPR command buffer (SharpEmu `AmprExports`;
/// type 4 is Raeen's self-sizing skip record from the KytyPS5-ported
/// nop/marker/wait/map commands — see `libsce_ampr::NOP_RECORD_TYPE`).
const APR_RECORD_READ_FILE: u32 = 1;
const APR_RECORD_KERNEL_EVENT_QUEUE: u32 = 2;
const APR_RECORD_WRITE_ADDRESS: u32 = 3;
const APR_RECORD_NOP: u32 = 4;

/// Complete one AMPR command buffer synchronously: walk its records and do
/// the completion work a console does async — fire completion events and
/// write completion addresses. ReadFile records are no-ops here because the
/// file data was read into guest memory EAGERLY at record-append time
/// (SharpEmu `AmprExports.CompleteCommandBuffer`, AmprExports.cs:550-608).
/// On success the buffer's host write cursor is consumed so a re-submit
/// without Reset cannot re-fire stale records.
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
                // No-op at completion: the read ran EAGERLY at record-append
                // time and the bytes (plus bytesRead @0x20) are already in
                // guest memory — SharpEmu `AmprExports.CompleteCommandBuffer`
                // skips ReadFile records for the same reason
                // (AmprExports.cs:578-580).
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
                        raeen_kernel::EqueueUserEvent {
                            triggered: true,
                            udata: user_data,
                            fflags: 1,
                            ..Default::default()
                        },
                    );
                }
                crate::kernel_equeue::wake_equeue(
                    ctx,
                    equeue,
                    raeen_core::subsystems::WakeReason::SubmissionComplete,
                );
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
            APR_RECORD_NOP => {
                // Self-sizing skip record ([type][total_size][payload…]):
                // nops, markers, dropped waits, and map bookkeeping — no
                // completion effect. `total_size` includes the 8-byte header.
                let mut sz = [0u8; 4];
                if !ctx.mem.read(record + 4, &mut sz) {
                    return SCE_KERNEL_ERROR_EFAULT;
                }
                let total = u64::from(u32::from_le_bytes(sz));
                if total < 8 || (total & 3) != 0 {
                    warn!(
                        "APR command buffer: corrupt skip record (total_size {total:#x}) at +{offset:#x}"
                    );
                    return SCE_KERNEL_ERROR_EINVAL;
                }
                offset += total;
            }
            other => {
                warn!("APR command buffer: unknown record type {other} at +{offset:#x}");
                return SCE_KERNEL_ERROR_EINVAL;
            }
        }
    }
    // Completion consumes the buffer: drop the host write cursor so a
    // re-submit without an explicit sceAmprAprCommandBufferReset can never
    // re-fire stale equeue/write-address records into guest addresses the
    // title may have repurposed (the ASTRO.BOT pause-menu heap-poisoning
    // vector). The record bytes stay in the guest buffer; only the host
    // cursor is dropped, so Reset + re-append is unaffected.
    ctx.kernel.ampr_write_offsets.remove(&cb);
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
    // The `resultAddress` slot: 8 zero bytes where the block is provably not a
    // caller local, 4 where it is. No in-tree evidence pins this slot's width —
    // the sibling `outSubmissionId` is a 4-byte `u32` (`appr_add_submission`
    // returns `u32`), and an SCE completion code is an `int`, but the SharpEmu
    // port this follows stores 8. The value is zero either way, so an off-stack
    // block keeps exactly the behavior it had, while a caller local can no
    // longer lose 4 bytes of the *next* local to a possibly-too-wide store.
    if result_address != 0
        && !ctx.zero_out_object(
            "libkernel::sceKernelAprSubmitCommandBufferAndGetResult",
            result_address,
            8,
            4,
        )
    {
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
/// `ForEach` strongly suggests a per-file guest callback, and Raeen must not
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
    let requested = args.get(2).copied().unwrap_or(0).min(MAX_HLE_BULK_BYTES);
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
    let id = ctx.kernel.register_module(raeen_kernel::ModuleInfo {
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

/// The `SCE_NID_SALT` every Orbis export name is hashed with: `NID =
/// LE_u64(SHA1(name || salt)[..8])`. A module's export table stores NIDs, not
/// names, so a name-keyed `dlsym` has to hash before it can look anything up.
const SCE_NID_SALT: [u8; 16] = [
    0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1, 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30,
];

/// Hash an export name to its NID.
fn nid_of_symbol(name: &[u8]) -> u64 {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(name);
    hasher.update(SCE_NID_SALT);
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-1 has 20 bytes"))
}

/// Where a `sceKernelDlsym` hit came from — for the resolution log line, so a
/// pointer handed to the guest can always be traced back to who supplied it.
enum DlsymHit {
    /// An export of the module the caller named (or of the main program, for
    /// handle 0).
    Module(u32),
    /// An export of some *other* loaded module, found by the load-order sweep.
    OtherModule(u32),
    /// An HLE trampoline: Raeen implements this system function itself.
    Hle,
}

/// `sceKernelDlsym(handle, symbol, addrOut)` — resolve an export name to a
/// guest-callable address.
///
/// # Handle 0 is the main program, not "search everywhere"
///
/// This is the bug this function was rewritten to fix. Handle 0 was treated as
/// an ordinary module id, matched nothing (ids start at 1), and returned an
/// error — which is what stopped Unity/IL2CPP titles dead, because IL2CPP asks
/// for its scripting allocator through `sceKernelDlsym(0, "scriptingGetMem")`
/// during startup.
///
/// The POSIX reflex is to read a null handle as `RTLD_DEFAULT` (global scope).
/// **Orbis does not.** KytyPS5's `RuntimeLinker::FindProgramById`
/// (`src/loader/runtimeLinker.cpp:1532`) reserves id 0 for the main program and
/// returns `m_programs.front()`; `unique_id` is handed out from 1. Our module
/// ids follow the same rule, so handle 0 maps to
/// [`OrbisKernel::main_lle_module_handle`].
///
/// # Resolution order
///
/// 1. The named module (handle 0 = main program).
/// 2. Every other loaded module, in load order. Not Kyty behaviour — this
///    follows SharpEmu's `DispatchKernelDynlibDlsym`, which tries the handle
///    and then falls back to a process-wide symbol sweep. Logged distinctly
///    when it hits, because a symbol found *outside* the module the guest
///    named means our handle bookkeeping disagrees with the title's.
/// 3. Raeen's own HLE trampolines by name. `dlsym` is the only caller that has
///    to turn a name into an address at run time; imports were already
///    relocated. Both references do the equivalent — Kyty returns emulator
///    functions from `KernelDlsym`, SharpEmu calls it a "runtime symbol".
///
/// # Failure
///
/// `SCE_KERNEL_ERROR_ESRCH`, matching KytyPS5's `KernelDlsym`
/// (`src/libs/libKernel.cpp:226`), which returns `ESRCH` both for an unknown
/// handle and for a symbol absent from a known module. The out-pointer is left
/// untouched and every miss is logged with module, NID, and name — never a
/// fabricated address, because the guest calls straight through whatever this
/// writes.
fn hle_dlsym(ctx: &HleContext, args: &[u64]) -> u64 {
    let raw_handle = args.first().copied().unwrap_or(0);
    let sym_ptr = args.get(1).copied().unwrap_or(0);
    let addr_out = args.get(2).copied().unwrap_or(0);
    // Both null guards are explicit, matching KytyPS5's `KernelDlsym`, rather
    // than left to `read_cstr` happening to fail on guest address 0 — that only
    // holds while nothing maps the zero page, which is not a property this
    // function should depend on.
    if addr_out == 0 || sym_ptr == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let Some(symbol_bytes) = crate::fmt::read_cstr(ctx.mem, sym_ptr) else {
        return SCE_KERNEL_ERROR_EFAULT;
    };
    let Ok(raw_handle) = u32::try_from(raw_handle) else {
        warn!("sceKernelDlsym(handle={raw_handle}): handle does not fit a module id — ESRCH");
        return SCE_KERNEL_ERROR_ESRCH;
    };
    let symbol = String::from_utf8_lossy(&symbol_bytes).into_owned();
    let nid = nid_of_symbol(&symbol_bytes);

    // Handle 0 names the executable. Resolving it to `None` here is not the
    // "unknown handle" case — it means no module registered an export table at
    // all, which the miss diagnostics below report separately.
    let scope = if raw_handle == 0 {
        ctx.kernel.main_lle_module_handle()
    } else {
        Some(raw_handle)
    };

    let hit = scope
        .and_then(|handle| {
            ctx.kernel
                .resolve_lle_export(handle, nid)
                .map(|addr| (addr, DlsymHit::Module(handle)))
        })
        .or_else(|| {
            ctx.kernel
                .resolve_lle_export_anywhere(nid)
                .map(|(handle, addr)| (addr, DlsymHit::OtherModule(handle)))
        })
        .or_else(|| {
            ctx.kernel
                .resolve_hle_export_addr(&symbol)
                .map(|addr| (addr, DlsymHit::Hle))
        });

    let Some((addr, source)) = hit else {
        dlsym_miss(ctx, raw_handle, scope, &symbol, nid);
        return SCE_KERNEL_ERROR_ESRCH;
    };

    if !ctx.mem.write(addr_out, &addr.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    match source {
        DlsymHit::Module(handle) => debug!(
            "sceKernelDlsym(handle={raw_handle}, symbol='{symbol}') -> {addr:#x} (module {handle})"
        ),
        DlsymHit::OtherModule(handle) => warn!(
            "sceKernelDlsym(handle={raw_handle}, symbol='{symbol}') -> {addr:#x} found in module \
             {handle}, NOT the module the guest named — resolved, but our module handles disagree \
             with the title's"
        ),
        DlsymHit::Hle => debug!(
            "sceKernelDlsym(handle={raw_handle}, symbol='{symbol}') -> {addr:#x} (Raeen HLE \
             trampoline)"
        ),
    }
    SCE_OK
}

/// Report a `sceKernelDlsym` miss precisely enough to act on.
///
/// The three failures below need three different fixes, and a bare `ESRCH`
/// cannot tell them apart — which is exactly how the handle-0 bug survived
/// several sessions being read as a memory fault.
fn dlsym_miss(ctx: &HleContext, raw_handle: u32, scope: Option<u32>, symbol: &str, nid: u64) {
    let hle_published = ctx.kernel.hle_export_addr_count();
    match scope {
        // No module has an export table: the loader never registered one, so
        // nothing could ever resolve here.
        None => warn!(
            "sceKernelDlsym(handle={raw_handle}, symbol='{symbol}', nid={nid:#018x}): NO module \
             has a registered export table ({hle_published} HLE trampoline(s) published) — ESRCH"
        ),
        Some(handle) => match ctx.kernel.lle_export_count(handle) {
            Some(count) => warn!(
                "sceKernelDlsym(handle={raw_handle}, symbol='{symbol}', nid={nid:#018x}): not \
                 among module {handle}'s {count} export(s), nor any other loaded module, nor the \
                 {hle_published} published HLE trampoline(s) — ESRCH"
            ),
            None => warn!(
                "sceKernelDlsym(handle={raw_handle}, symbol='{symbol}', nid={nid:#018x}): handle \
                 names NO registered module — ESRCH"
            ),
        },
    }
}

/// `scriptingGetMem(alignment, size)` — the aligned allocator Unity's IL2CPP
/// scripting backend fetches from libkernel via
/// `sceKernelDlsym(0, "scriptingGetMem", &fn)` during startup.
///
/// # Why this exists at all
///
/// It is **not** an export of any guest module, and no amount of module-table
/// searching will find it: it is a hook the runtime is expected to supply. Both
/// references that boot Unity titles special-case exactly this name —
/// KytyPS5 returns its own `KernelApplicationHeapGetMem`
/// (`src/libs/libKernel.cpp:203`), and SharpEmu aliases the name in
/// `TryResolveRuntimeSymbolAlias` (`DirectExecutionBackend.Imports.cs:2098`).
///
/// # Signature
///
/// `(alignment, size)`, from KytyPS5 — which boots Blasphemous II in-game, and
/// whose implementation clamps `alignment` up to `0x10` and rejects a
/// non-power-of-two, a guard only worth writing against observed arguments.
/// SharpEmu instead aliases the name to plain `malloc(size)`; that disagrees,
/// and the reference that actually runs the title wins.
///
/// The power-of-two check is kept for the same reason KytyPS5 has it: it is a
/// **self-test on the signature**. If the real first argument were a size
/// rather than an alignment it would almost never be a power of two, so a
/// mis-read ABI shows up as a loud null return instead of a plausible-looking
/// pointer the guest then writes `size` bytes through.
fn hle_scripting_get_mem(ctx: &HleContext, args: &[u64]) -> u64 {
    let requested_alignment = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    let alignment = requested_alignment.max(0x10);

    if !alignment.is_power_of_two() {
        warn!(
            "scriptingGetMem(alignment={requested_alignment:#x}, size={size:#x}): alignment is \
             not a power of two — refusing to allocate. Either the guest passed a bad alignment \
             or this function's (alignment, size) signature is wrong for this title"
        );
        return 0;
    }

    match ctx.alloc.alloc(size.max(1), alignment) {
        Some(addr) => {
            // Same live-block table the malloc family keeps, so
            // `malloc_usable_size` does not report a scripting block as dead.
            crate::libc::track_alloc(ctx, addr, size);
            debug!("scriptingGetMem(alignment={alignment:#x}, size={size:#x}) -> {addr:#x}");
            addr
        }
        None => {
            warn!(
                "scriptingGetMem(alignment={alignment:#x}, size={size:#x}): guest heap exhausted \
                 — returning null"
            );
            0
        }
    }
}

/// `scriptingFreeMem(ptr)` — the release half of the Unity scripting allocator
/// pair. Unambiguous whatever the rest of the family's shape turns out to be:
/// one pointer argument, no return.
///
/// Deliberately paired with [`hle_scripting_get_mem`]: handing IL2CPP an
/// allocator with no matching deallocator makes every scripting free leak.
fn hle_scripting_free_mem(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    if ptr != 0 {
        crate::libc::track_free(ctx, ptr);
        ctx.alloc.free(ptr);
    }
    SCE_OK
}

/// HLE functions that must get a trampoline address even though **no module
/// imports them** — the process loader reserves one for each
/// (`ProcessTables::reserve_hle_export`).
///
/// A normal HLE function earns its trampoline from a relocation: some module
/// imported it, so the linker minted an address and wrote it into the import
/// slot. These have no importer anywhere in the process. The guest reaches them
/// only by asking `sceKernelDlsym` for the name, and `dlsym` can only answer
/// with an address that already exists.
pub const DLSYM_RESERVED_EXPORTS: &[(&str, &str)] = &[
    ("libkernel", "scriptingGetMem"),
    ("libkernel", "scriptingFreeMem"),
];

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
        "sceKernelCheckedReleaseDirectMemory",
        hle_checked_release_direct_memory,
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
    // NID (RAEEN_TRAP_CXA_THROW); otherwise the shipped libc's real
    // __cxa_throw is used. Registered by NID (for import redirection) AND by
    // name (so a runtime-patched jump into an appended trampoline dispatches
    // here via the VEH's name-based hle.call).
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
    registry.register("libkernel", "sceKernelFsync", hle_sce_fsync);
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
    // pwrite/ftruncate: the write-side positional and resize calls, registered
    // under the same provider pair as pread (resolution is provider-aware).
    registry.register("libkernel", "sceKernelPwrite", hle_sce_pwrite);
    registry.register("libkernel", "pwrite", hle_posix_pwrite);
    registry.register("libScePosix", "pwrite", hle_posix_pwrite);
    registry.register("libkernel", "sceKernelFtruncate", hle_sce_ftruncate);
    registry.register("libkernel", "ftruncate", hle_posix_ftruncate);
    registry.register("libScePosix", "ftruncate", hle_posix_ftruncate);
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
    // The Unity/IL2CPP scripting allocator hooks. Registered under `libkernel`
    // because that is the library IL2CPP fetches them from, and registered at
    // all so `load_process` can reserve trampolines for them — `dlsym` needs a
    // guest-callable address to hand back, and only a reserved trampoline has
    // one. See `hle_scripting_get_mem` for the signature's provenance.
    //
    // `scriptingRealloc` / `scriptingCalloc` are deliberately NOT registered.
    // SharpEmu aliases them to libc `realloc`/`calloc`, but if `scriptingGetMem`
    // really is `(alignment, size)` then this family does not use libc argument
    // order, and guessing wrong on a *resize* corrupts the heap. An honest
    // `ESRCH` naming the symbol is the correct answer until one is measured.
    registry.register("libkernel", "scriptingGetMem", hle_scripting_get_mem);
    registry.register("libkernel", "scriptingFreeMem", hle_scripting_free_mem);
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
    // Address-parking primitives (the PS5's futex): a thread waits until another
    // writes the watched word and wakes it. REAL parking lot — an address-keyed
    // FIFO in `OrbisKernel::sync_addresses`, with a value compare on entry for
    // the sized variants and `Wake` releasing N waiters on that exact address.
    // See `sync_on_address_wait` for the enqueue-then-compare ordering that makes
    // it race-free, and for why the generic (unsized) `Wait` skips the compare.
    // Names recovered via the catalogue merge (Wake = 0xab6cbfc032155990);
    // Wait/Wake both appeared unnamed in a real run.
    registry.register(
        "libkernel",
        "sceKernelSyncOnAddressWait",
        hle_sync_on_address_wait,
    );
    registry.register(
        "libkernel",
        "sceKernelSyncOnAddressWait32",
        hle_sync_on_address_wait32,
    );
    registry.register(
        "libkernel",
        "sceKernelSyncOnAddressWait64",
        hle_sync_on_address_wait64,
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
    // machine) — see raeen_hle::pthread_sync.
    // scePthreadCond* are registered by the `pthread_cond` module (real state
    // machine: the wait releases and reacquires the guest mutex around a
    // generation-counted sleep) — see raeen_hle::pthread_cond.
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
    // pthread_setschedparam/pthread_getschedparam are registered by
    // `pthread_thread` (real sched-param bookkeeping, which runs after this
    // function and wins) — the old hle_ok_stub here dropped the value, the
    // same lesson as scePthreadSetprio below.
    // POSIX `fstat` returns -1 + errno; `sceKernelFstat` (below) returns an SCE
    // code. Registering the same raw handler under both spellings gave both the
    // wrong convention. Also registered under `libkernel`, which exports the
    // plain POSIX spellings too (see `read`/`write` above).
    registry.register("libScePosix", "fstat", hle_posix_fstat);
    registry.register("libkernel", "fstat", hle_posix_fstat);

    // -- Measured Minecraft libc.prx / eboot imports (real PS5 export names,
    // each verified by NID hash against the title's import table; semantics
    // cross-checked with SharpEmu + Kyty). The `_`-prefixed file/exit names
    // are libkernel's real exports of the plain POSIX calls.
    registry.register("libkernel", "_open", hle_posix_open);
    registry.register("libkernel", "_read", hle_posix_read);
    registry.register("libkernel", "_write", hle_posix_write);
    registry.register("libkernel", "_close", hle_posix_close);
    // `_exit` terminates the process: the runtime's exit family intercepts it
    // before dispatch (see raeen_runtime::dispatch::TERMINATING_FUNCTIONS);
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
        "sceKernelMtypeprotect",
        hle_kernel_mtypeprotect,
    );
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
    registry.register("libkernel", "sceKernelIsStack", hle_is_stack);
    registry.register(
        "libkernel",
        "sceKernelQueryMemoryProtection",
        hle_query_memory_protection,
    );
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
    for library in ["libkernel", "libkernel_unity"] {
        registry.register(
            library,
            "sceKernelInstallExceptionHandler",
            hle_install_exception_handler,
        );
        registry.register(
            library,
            "sceKernelRemoveExceptionHandler",
            hle_remove_exception_handler,
        );
        registry.register_incomplete(
            library,
            "sceKernelRaiseException",
            hle_raise_exception,
            "handler state is modeled but asynchronous guest-thread delivery is not implemented",
        );
    }
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
    registry.register("libkernel", "sceKernelFstat", hle_sce_fstat);
    // Plain POSIX spellings of the same two metadata calls — different NIDs
    // from the sce* forms, registered under both providers (the measured
    // imports name `libScePosix`; `unlink`/`rmdir` above show libkernel
    // exports the plain spellings too). `-1` + `errno` convention, like
    // `hle_posix_unlink`.
    registry.register("libkernel", "rename", hle_posix_rename);
    registry.register("libScePosix", "rename", hle_posix_rename);
    registry.register("libkernel", "stat", hle_posix_stat);
    registry.register("libScePosix", "stat", hle_posix_stat);
    // pthread surface libc/fmod touch during init — attr/priority/affinity
    // bookkeeping has no scheduler to talk to yet, so recording nothing and
    // returning success is faithful enough for a single-thread world.
    registry.register("libkernel", "scePthreadDetach", hle_pthread_detach);
    // scePthreadSetprio/Getprio are registered by `pthread_thread` (real
    // priority bookkeeping) — the old hle_ok_stub here dropped the value.
    registry.register("libkernel", "scePthreadSetaffinity", hle_ok_stub);
    registry.register(
        "libkernel",
        "scePthreadGetaffinity",
        hle_pthread_getaffinity,
    );
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
    // module (real user-event queue) — see raeen_hle::kernel_equeue.

    // -- Misc / process / clock --
    registry.register("libkernel", "sceKernelGetProcessType", hle_get_process_type);
    registry.register("libkernel", "sceKernelGetCurrentCpu", hle_get_current_cpu);
    registry.register("libkernel", "sceKernelGettimeofday", hle_gettimeofday);
    registry.register("libkernel", "sceKernelClockGettime", hle_clock_gettime);
    registry.register("libkernel", "sceKernelClockGetres", hle_clock_getres);
    // POSIX `clock_getres(clockId, timespec *res)` — libKernel exports it under a
    // distinct NID and shipped middleware links the plain name (SharpEmu #450,
    // 0c467e8, `smIj7eqzZE8`; DOOM's party.prx blocked on it). Same 1 ns domain
    // as `sceKernelClockGetres`, but a NULL `res` is accepted per POSIX.
    registry.register("libkernel", "clock_getres", hle_clock_getres_posix);
    // POSIX `getpagesize()` — reports the 16 KiB Orbis page, not the host 4 KiB
    // (SharpEmu #450). An allocator rounding to the host value produces sub-page
    // offsets every mapping call here rejects for misalignment.
    registry.register("libkernel", "getpagesize", hle_getpagesize);
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
    registry.register("libkernel", "getargc", hle_getargc);
    registry.register("libkernel", "getargv", hle_getargv);
    registry.register("libkernel", "sceKernelGetGPI", hle_get_gpi);
    registry.register("libkernel", "sceKernelSetGPO", hle_set_gpo);
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

/// `sceKernelSetGPO(bits)`: drive the General Purpose **Output** lines.
///
/// GPO is the output half of the same devkit-only mechanism as GPI (see
/// [`hle_get_gpi`]): on development kits these lines drive external hardware
/// (LEDs/switches on the test rig); a retail console has nothing wired to
/// them, so setting them is a defined no-op that succeeds. Accepting the call
/// is therefore the **real retail behavior** — the bits genuinely go nowhere
/// because there is genuinely nothing to receive them. Unlike SharpEmu we do
/// NOT loop the value back into `sceKernelGetGPI`: output state is not input
/// state, and folding it into GPI would report switches that were never set.
fn hle_set_gpo(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceKernelSetGPO(bits={:#x}) -> 0 (no devkit GPO lines on retail)",
        args.first().copied().unwrap_or(0)
    );
    SCE_OK
}

/// The PS5 (Gen5) compiled-SDK version Raeen reports: `0x09000000` == SDK
/// 9.00 (same value SharpEmu's `Gen5CompiledSdkVersion` reports). Homebrew
/// commonly gates feature use on this.
const GEN5_SDK_VERSION: u32 = 0x0900_0000;

/// `SCE_KERNEL_ERROR_EINVAL` (`0x80020016`): invalid argument (EINVAL = 22).
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;

/// A fixed, plausible process id Raeen reports for the single guest process.
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

/// Return the process argument count recorded from the runtime-built initial
/// stack. Semantics cross-checked against shadPS4's GPL-2.0 kernel exports.
fn hle_getargc(ctx: &HleContext, _args: &[u64]) -> u64 {
    let (argc, _) = ctx.kernel.process_args();
    debug!("getargc() -> {argc}");
    argc
}

/// Return the guest address of the process `char **argv` table.
fn hle_getargv(ctx: &HleContext, _args: &[u64]) -> u64 {
    let (_, argv) = ctx.kernel.process_args();
    debug!("getargv() -> {argv:#x}");
    argv
}

// ---------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------

fn trace_guest_vmm(message: std::fmt::Arguments<'_>) {
    if std::env::var_os("RAEEN_TRACE_GUEST_VMM").is_some() {
        warn!("guest VMM: {message}");
    }
}

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
/// `SCE_OK` on success; `SCE_KERNEL_ERROR_EAGAIN` if the budget or the arena
/// cannot satisfy the request, and `SCE_KERNEL_ERROR_EFAULT` if `physAddrOut`
/// is out of bounds (bounds-checked, never a panic/OOB) — in the latter case
/// the just-recorded metadata is rolled back via `remove_mapping` so no
/// dangling record is left behind.
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
    let page_size = raeen_core::PS5_PAGE_SIZE as u64;
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
        // Same code as the budget refusal above, and for the same reason: the
        // real allocator answers "I could not give you this memory" with
        // `EAGAIN` (shadPS4 `sceKernelAllocateDirectMemory`). A title sizing its
        // pools by allocating until refusal must see a status it recognises.
        warn!("sceKernelAllocateDirectMemory: arena mmap failed (len={len:#x}) — EAGAIN");
        ctx.kernel
            .direct_memory_allocated
            .fetch_sub(len, std::sync::atomic::Ordering::Relaxed);
        return SCE_KERNEL_ERROR_EAGAIN_ALLOC;
    };
    // Remember the allocation's `type` against its physical range, so the
    // mapping made from it can echo that type back through
    // `sceKernelVirtualQuery`.
    ctx.kernel.memory.record_mapping_of_kind(
        addr,
        len,
        DEFAULT_PROT,
        raeen_core::types::MappingKind::Direct,
        addr,
        i32::try_from(memory_type).unwrap_or(0),
    );
    trace_guest_vmm(format_args!(
        "allocate-direct search={search_start:#x}..{search_end:#x} len={len:#x} \
         align={alignment:#x} type={memory_type} -> phys={addr:#x}"
    ));

    if !ctx.mem.write(phys_addr_out, &addr.to_le_bytes()) {
        warn!("sceKernelAllocateDirectMemory: physAddrOut {phys_addr_out:#x} out of bounds");
        // Full rollback: the arena mapping AND the budget charge were both
        // made above — leaving either behind leaks it for the process
        // lifetime (a title probing out-param validity would otherwise
        // inflate its own measured footprint until ENOMEM).
        ctx.alloc.munmap(addr, len);
        ctx.kernel.memory.remove_mapping(addr);
        ctx.kernel
            .direct_memory_allocated
            .fetch_sub(len, std::sync::atomic::Ordering::Relaxed);
        // An unwritable out-parameter is a bad address, which is `EFAULT` —
        // the same code every other out-param guard in this file returns.
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if std::env::var_os("RAEEN_TRACE_DIRECT_MEMORY").is_some() {
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
    if std::env::var_os("RAEEN_TRACE_DIRECT_MEMORY").is_some() {
        warn!("direct-memory trace: release phys={start:#x} len={len:#x}");
    }
    trace_guest_vmm(format_args!("release-direct phys={start:#x} len={len:#x}"));
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

/// `sceKernelCheckedReleaseDirectMemory(start, len)`: the strict twin of
/// `sceKernelReleaseDirectMemory`. The unchecked form is best-effort by
/// contract (freeing an untracked range is a successful no-op, see above);
/// the CHECKED form validates and refuses:
///
/// * `start`/`len` must be page-aligned (`EINVAL`) — cross-checked against
///   SharpEmu's `KernelCheckedReleaseDirectMemory` (GPL-2.0);
/// * the whole range must lie inside one TRACKED direct-memory allocation,
///   else `ENOENT` — a title probing with a wild or double-freed range gets
///   a real refusal instead of a silent "success".
///
/// A zero length is valid and releases nothing (`SCE_OK`), as on the
/// unchecked path.
fn hle_checked_release_direct_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let start = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    debug!("sceKernelCheckedReleaseDirectMemory(start={start:#x}, len={len:#x})");

    let page_size = raeen_core::PS5_PAGE_SIZE as u64;
    if start % page_size != 0 || len % page_size != 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    if len == 0 {
        return SCE_OK;
    }
    let Some(end) = start.checked_add(len) else {
        return SCE_KERNEL_ERROR_EINVAL;
    };
    let covers = ctx
        .kernel
        .memory
        .region_containing(start)
        .filter(|region| region.kind == raeen_core::types::MappingKind::Direct)
        .and_then(|region| region.vaddr.checked_add(region.size))
        .is_some_and(|region_end| end <= region_end);
    if !covers {
        debug!(
            "sceKernelCheckedReleaseDirectMemory: [{start:#x}, {end:#x}) is not a tracked \
             direct-memory allocation — ENOENT"
        );
        return SCE_KERNEL_ERROR_ENOENT;
    }
    hle_release_direct_memory(ctx, args)
}

fn hle_map_direct_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr_out = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0) as u32;
    let flags = args.get(3).copied().unwrap_or(0) as u32;
    let direct_memory_start = args.get(4).copied().unwrap_or(0);
    let alignment = args.get(5).copied().unwrap_or(0);
    debug!(
        "sceKernelMapDirectMemory(addrOut={addr_out:#x}, len={len:#x}, prot={prot}, flags={flags:#x}, phys={direct_memory_start:#x}, alignment={alignment:#x})"
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
    let fixed = flags & MAP_FIXED != 0;
    let page_size = raeen_core::PS5_PAGE_SIZE as u64;
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
        // A requested address is mandatory only under `MAP_FIXED`. Without that
        // flag Orbis treats it as a hint and is free to place the mapping
        // elsewhere, reporting where through `addrOut` — so a hint we cannot
        // serve must not sink the call. Falling back to `direct_memory_start`
        // is exactly the answer the no-hint branch above gives, which keeps the
        // mapping and the direct memory one storage rather than detaching them.
        ctx.alloc.map_at(requested, len, alignment).or_else(|| {
            if fixed || !direct_memory_start.is_multiple_of(alignment) {
                return None;
            }
            warn!(
                "sceKernelMapDirectMemory: hint {requested:#x} unavailable for len={len:#x}; \
                 MAP_FIXED is clear, so publishing {direct_memory_start:#x} instead"
            );
            Some(direct_memory_start)
        })
    };
    let Some(mapped) = mapped else {
        // `ENOMEM` is what the real kernel reports for a fixed mapping it
        // cannot place (shadPS4 `MemoryManager::MapMemory`). It must be a real
        // `SCE_KERNEL_ERROR_*`: the guest branches on this value, and ASTRO.BOT
        // asserts on it — a sentinel it cannot classify leaves it dereferencing
        // whatever it computed from an unmapped base.
        warn!(
            "sceKernelMapDirectMemory: cannot map len={len:#x} at requested={requested:#x} \
             (fixed={fixed}) — ENOMEM"
        );
        return SCE_KERNEL_ERROR_ENOMEM;
    };
    // Record it as DIRECT, carrying the physical offset and the type the guest
    // allocated it with — a title reads its own mappings back, and
    // `sceKernelVirtualQuery` must agree with the map it just performed.
    ctx.kernel.memory.record_mapping_of_kind(
        mapped,
        len,
        prot,
        raeen_core::types::MappingKind::Direct,
        direct_memory_start,
        ctx.kernel
            .memory
            .direct_allocation_type(direct_memory_start)
            .unwrap_or(0),
    );
    trace_guest_vmm(format_args!(
        "map-direct requested={requested:#x} phys={direct_memory_start:#x} len={len:#x} \
         align={alignment:#x} prot={prot:#x} -> {mapped:#x}"
    ));
    if !ctx.mem.write(addr_out, &mapped.to_le_bytes()) {
        if requested != 0 {
            ctx.alloc.munmap(mapped, len);
        }
        ctx.kernel.memory.remove_mapping(mapped);
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if std::env::var_os("RAEEN_TRACE_DIRECT_MEMORY").is_some() {
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
    let page = raeen_core::PS5_PAGE_SIZE as u64;
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
        if std::env::var_os("RAEEN_TRACE_DIRECT_MEMORY").is_some() {
            warn!(
                "batch-map[{i}]: op={operation} start={start:#x} offset={offset:#x} \
                 len={length:#x} prot={prot:#x}"
            );
        }
        trace_guest_vmm(format_args!(
            "batch[{i}] op={operation} start={start:#x} phys={offset:#x} \
             len={length:#x} prot={prot:#x}"
        ));
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
            // Apply the protection under RAEEN_ENFORCE_MPROTECT; a no-op
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
/// `SCE_OK` on success; `SCE_KERNEL_ERROR_ENOMEM` if the arena is exhausted
/// and `SCE_KERNEL_ERROR_EFAULT` if `addrOut` is out of bounds — in the
/// latter case `remove_mapping` rolls back the just-recorded metadata so no
/// dangling record is left behind.
fn hle_map_flexible_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr_out = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0x3) as u32;
    debug!("sceKernelMapFlexibleMemory(addrOut={addr_out:#x}, len={len:#x}, prot={prot})");

    let Some(addr) = ctx.alloc.mmap(len, raeen_core::PS5_PAGE_SIZE as u64) else {
        warn!("sceKernelMapFlexibleMemory: arena mmap failed (len={len:#x}) — ENOMEM");
        return SCE_KERNEL_ERROR_ENOMEM;
    };
    ctx.kernel.memory.record_mapping(addr, len, prot);
    trace_guest_vmm(format_args!(
        "map-flexible len={len:#x} prot={prot:#x} -> {addr:#x}"
    ));

    if addr_out != 0 && !ctx.mem.write(addr_out, &addr.to_le_bytes()) {
        warn!("sceKernelMapFlexibleMemory: addrOut {addr_out:#x} out of bounds — EFAULT");
        ctx.kernel.memory.remove_mapping(addr);
        return SCE_KERNEL_ERROR_EFAULT;
    }
    SCE_OK
}

/// Releases a mapping previously returned by `sceKernelMapFlexibleMemory`/
/// `sceKernelAllocateDirectMemory`/`sceKernelMmap`: releases the arena
/// allocation (`ctx.alloc.munmap`, best-effort — see
/// [`raeen_hle::GuestAllocator::munmap`]'s contract) and removes the VMM
/// metadata (`ctx.kernel.memory.remove_mapping`) so `is_mapped` stops
/// reporting the address as mapped. Always reports success (`SCE_OK`),
/// matching real `munmap`'s behavior on an already-unmapped/unrecognized
/// address.
fn hle_munmap(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    debug!("sceKernelMunmap(addr={addr:#x}, len={len:#x})");
    trace_guest_vmm(format_args!("unmap addr={addr:#x} len={len:#x}"));

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
/// value); on failure returns `0`, since `0` is `sceKernelMmap`'s real
/// `NULL`-ish failure convention for an address-returning call. This is the
/// convention every address-returning export in this file follows — an
/// `SCE_KERNEL_ERROR_*` code, or the `0xffffffff` sentinel this file used to
/// carry, is a pointer the caller will happily dereference.
fn hle_mmap(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0x3) as u32;
    let flags = args.get(3).copied().unwrap_or(0);
    debug!("sceKernelMmap(addr={addr:#x}, len={len:#x}, prot={prot}, flags={flags:#x})");

    let Some(mapped) = ctx.alloc.mmap(len, raeen_core::PS5_PAGE_SIZE as u64) else {
        warn!("sceKernelMmap: arena mmap failed (len={len:#x})");
        return 0;
    };
    ctx.kernel.memory.record_mapping(mapped, len, prot);
    trace_guest_vmm(format_args!(
        "mmap hint={addr:#x} len={len:#x} prot={prot:#x} flags={flags:#x} -> {mapped:#x}"
    ));
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
/// Raeen models direct memory as one contiguous pool
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

    let align = alignment.max(raeen_core::PS5_PAGE_SIZE as u64);
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
/// libkernel twin of `libSceAgc`'s `sceAgcGetIsTrinityMode`; Raeen emulates a
/// base PS5, so both answer false.
///
/// Name recovered for NID `0xb54e5eddff604a25` (`tU5e3f9gSiU`) from SharpEmu's
/// aerolib catalogue. Measured: Until Dawn stops its boot on this import.
fn hle_is_trinity_mode(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0
}

/// `EAGAIN` (errno 11) in SCE coding: the futex value compare failed, so the
/// guest must re-read its own condition instead of parking. Same numeric code as
/// [`SCE_KERNEL_ERROR_EAGAIN_ALLOC`], spelled separately because here it is the
/// futex "value already changed" answer, not an allocation failure.
const SCE_KERNEL_ERROR_EAGAIN: u64 = 0x8002_000B;
/// `ETIMEDOUT` (errno 60) in SCE coding. Same split as
/// `scePthreadCondTimedwait` / `scePthreadMutexTimedlock`: a bare POSIX 60 is
/// unclassifiable by the title's own libc wrappers.
const SCE_KERNEL_ERROR_ETIMEDOUT: u64 = 0x8002_003C;

/// How long a wait with **no decodable deadline** stays parked before returning
/// success as a permitted spurious wakeup.
///
/// This is the safety net, not the mechanism: real releases come from
/// [`hle_sync_on_address_wake`], which wakes the queued waiter immediately. It
/// exists so a genuinely missed wake — or a guest that drives these through a
/// path Raeen has not resolved — self-heals into the caller re-checking its own
/// condition instead of hanging. Kept large on purpose: a short bound turns
/// every parked waiter into a hot re-poll that steals host bandwidth from the
/// threads making progress, including the one that would issue the wake.
const SYNC_ADDRESS_SELF_HEAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Park slice, so process teardown and deadlines are observed even when no wake
/// ever arrives. Not a spurious guest wakeup: the loop re-parks.
const SYNC_ADDRESS_SLICE: std::time::Duration = std::time::Duration::from_millis(10);

/// Largest value accepted as a *relative microsecond* timeout (60 s). Anything
/// larger is far more likely a guest pointer or an unset register than a real
/// deadline, and is treated as "no deadline" rather than gambled on.
const SYNC_ADDRESS_MAX_TIMEOUT_US: u64 = 60_000_000;

/// The compare width of a `sceKernelSyncOnAddressWait*` variant.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SyncWidth {
    /// `sceKernelSyncOnAddressWait32` — compare the 32-bit word at the address.
    Bits32,
    /// `sceKernelSyncOnAddressWait64` — compare the 64-bit word at the address.
    Bits64,
}

/// Read the watched word, or `None` if the address is not mapped.
fn read_sync_word(ctx: &HleContext, addr: u64, width: SyncWidth) -> Option<u64> {
    match width {
        SyncWidth::Bits32 => {
            let mut buf = [0u8; 4];
            ctx.mem
                .read(addr, &mut buf)
                .then(|| u64::from(u32::from_le_bytes(buf)))
        }
        SyncWidth::Bits64 => {
            let mut buf = [0u8; 8];
            ctx.mem
                .read(addr, &mut buf)
                .then(|| u64::from_le_bytes(buf))
        }
    }
}

/// Turn the raw timeout argument into a deadline.
///
/// 0 is the futex `NULL`-timeout spelling: wait indefinitely (bounded only by
/// [`SYNC_ADDRESS_SELF_HEAL`]). A plausible microsecond count becomes a real
/// deadline that must surface as `ETIMEDOUT`. Anything else is not trusted — see
/// [`SYNC_ADDRESS_MAX_TIMEOUT_US`].
fn decode_sync_timeout(raw: u64) -> Option<std::time::Instant> {
    if raw == 0 || raw > SYNC_ADDRESS_MAX_TIMEOUT_US {
        return None;
    }
    Some(std::time::Instant::now() + std::time::Duration::from_micros(raw))
}

/// The shared body of every `sceKernelSyncOnAddressWait*` variant: a real
/// address-keyed futex wait.
///
/// `width` is `None` for the generic `sceKernelSyncOnAddressWait`, whose
/// argument layout beyond the address is not recovered (SharpEmu records the
/// same gap) — that variant parks without a compare rather than gamble on which
/// register holds the expected value. The `*32`/`*64` spellings name their own
/// width, so they do the real compare.
///
/// **Ordering is the whole correctness argument.** The waiter joins the
/// address's FIFO *before* the watched word is read. A waker that writes the
/// word after our read therefore necessarily finds us already queued and wakes
/// us; a waker that wrote before our read is observed by the compare and we
/// return `EAGAIN` without parking. There is no window where a wake is lost, and
/// no need for the wake-generation counter SharpEmu used to approximate this.
fn sync_on_address_wait(ctx: &HleContext, args: &[u64], width: Option<SyncWidth>) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    if addr == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let expected = args.get(1).copied().unwrap_or(0);
    let deadline = decode_sync_timeout(args.get(2).copied().unwrap_or(0));

    let queue = ctx.kernel.sync_addresses.queue(addr);
    let waiter = queue.enqueue_waiter(ctx.guest_threads.current_thread());

    if let Some(width) = width {
        let expected = match width {
            SyncWidth::Bits32 => expected & u64::from(u32::MAX),
            SyncWidth::Bits64 => expected,
        };
        match read_sync_word(ctx, addr, width) {
            None => {
                if !queue.cancel_waiter(&waiter) {
                    return SCE_OK;
                }
                warn!("sceKernelSyncOnAddressWait: addr={addr:#x} not readable — EFAULT");
                return SCE_KERNEL_ERROR_EFAULT;
            }
            Some(observed) if observed != expected => {
                // The condition already moved. Per the futex contract the guest
                // must NOT park: it re-reads and proceeds. Returning success here
                // (as the old stub did) is what let a guest treat "0" as "the
                // value changed" and livelock on a word that never moves again.
                if !queue.cancel_waiter(&waiter) {
                    return SCE_OK;
                }
                debug!(
                    "sceKernelSyncOnAddressWait(addr={addr:#x}) observed={observed:#x} \
                     expected={expected:#x} -> EAGAIN"
                );
                return SCE_KERNEL_ERROR_EAGAIN;
            }
            Some(_) => {}
        }
    }

    // Spin-then-park, sharing the mutex path's waiter primitive and budget
    // (`RAEEN_MUTEX_SPIN`; 0 disables): a wake that lands within the spin
    // budget is observed without a host park/unpark round trip. The waiter is
    // already enqueued, so FIFO wake order and the compare-before-park futex
    // contract above are unchanged.
    let spin_budget = raeen_kernel::guest_waiter_spin_budget();
    if spin_budget > 0 && waiter.spin_for_signal(spin_budget) {
        return SCE_OK;
    }
    let parked_since = std::time::Instant::now();
    loop {
        if waiter.wait_for_signal(SYNC_ADDRESS_SLICE) {
            return SCE_OK;
        }
        if ctx.guest_threads.process_is_terminating() {
            queue.cancel_waiter(&waiter);
            return SCE_OK;
        }
        if let Some(deadline) = deadline {
            if std::time::Instant::now() >= deadline {
                // Wake and timeout can race; the queue lock decides. If we are
                // already gone a waker took us, and success is the truth.
                if !queue.cancel_waiter(&waiter) {
                    return SCE_OK;
                }
                return SCE_KERNEL_ERROR_ETIMEDOUT;
            }
        } else if parked_since.elapsed() >= SYNC_ADDRESS_SELF_HEAL {
            if !queue.cancel_waiter(&waiter) {
                return SCE_OK;
            }
            // Permitted spurious wakeup: the guest re-checks its own condition
            // and either proceeds or waits again. Never a hang.
            return SCE_OK;
        }
    }
}

/// `sceKernelSyncOnAddressWait(addr, ...)`: park until another thread wakes this
/// guest address. Generic variant — parks without a value compare, see
/// [`sync_on_address_wait`].
fn hle_sync_on_address_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    sync_on_address_wait(ctx, args, None)
}

/// `sceKernelSyncOnAddressWait32(addr, expected, timeout_us)`: park only while
/// the 32-bit word at `addr` still equals `expected`.
fn hle_sync_on_address_wait32(ctx: &HleContext, args: &[u64]) -> u64 {
    sync_on_address_wait(ctx, args, Some(SyncWidth::Bits32))
}

/// `sceKernelSyncOnAddressWait64(addr, expected, timeout_us)`: the 64-bit twin.
fn hle_sync_on_address_wait64(ctx: &HleContext, args: &[u64]) -> u64 {
    sync_on_address_wait(ctx, args, Some(SyncWidth::Bits64))
}

/// `sceKernelSyncOnAddressWake(addr, count)`: wake up to `count` threads parked
/// on `addr`, oldest first.
///
/// `count` of 1 is wake-one; a large or unset value is wake-all (SharpEmu's
/// reading of the same argument). Waking an address nobody is parked on is not
/// an error — a wake that beats its wait is the uncontended case, and the
/// waiter's own compare-on-entry catches it.
fn hle_sync_on_address_wake(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    if addr == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let requested = args.get(1).copied().unwrap_or(0) as i64;
    let count = if requested > 0 && requested < i64::from(i32::MAX) {
        usize::try_from(requested).unwrap_or(usize::MAX)
    } else {
        usize::MAX
    };
    let woken = ctx.kernel.sync_addresses.wake(addr, count);
    debug!(
        "sceKernelSyncOnAddressWake(addr={addr:#x}, count={requested}) -> woke {woken}, \
         {} still parked",
        ctx.kernel.sync_addresses.waiter_count(addr)
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
/// Raeen's direct-memory allocator hands out arena addresses and records each
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
    // Memory type: Raeen models one CPU-coherent pool (Onion / WB_ONION = 0).
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
/// Raeen reports a deterministic per-install-independent constant — stable
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
/// spawned — real thread creation needs [`raeen_kernel::threading`] wiring.
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
// process per host process (see `raeen_runtime::dispatch::CALL_LOCK`).
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
            "__tls_get_addr(module={module_id:#x}, offset={offset:#x}): offset exceeds bounded \
             block — NULL"
        );
        // `NULL`, not a status: see the allocation failure below. An
        // `SCE_KERNEL_ERROR_*` handed back from an address-returning call is a
        // pointer into low memory that the caller will happily dereference.
        return 0;
    }

    let thread = ctx.guest_threads.current_thread();
    let key = (thread, module_id);
    let base = if let Some(existing) = ctx.kernel.dynamic_tls_blocks.get(&key) {
        *existing
    } else {
        let Some(base) = ctx.alloc.alloc(DYNAMIC_TLS_BLOCK_SIZE, 16) else {
            // `__tls_get_addr` returns an ADDRESS, not a status, so no
            // `SCE_KERNEL_ERROR_*` belongs here — `NULL` is the only value a
            // caller could conceivably test, and it is the same convention
            // `hle_mmap` already uses for an address-returning failure. The
            // sentinel `0xffffffff` was strictly worse: a caller that ignores
            // the result dereferences it as a real pointer.
            warn!("__tls_get_addr(module={module_id:#x}): guest TLS allocation failed — NULL");
            return 0;
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

/// `getrusage(who, rusage*)`: report zeroed resource usage. Raeen keeps no
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
/// handler as `SIG_DFL` (0). Raeen never delivers guest signals (see
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
    trace_guest_vmm(format_args!(
        "mprotect addr={addr:#x} len={len:#x} prot={prot:#x}"
    ));
    // Apply the protection when enforcement is on; a no-op otherwise (the arena
    // default). POSIX `mprotect` returns 0 on success and **-1 with `errno`
    // set** on failure — it used to return the internal `-22` raw, which a
    // guest comparing against `-1` never recognised as an error, and `errno`
    // stayed stale.
    if ctx.mem.protect(addr, len, prot) {
        0
    } else {
        file_result_posix(ctx, FILE_EINVAL)
    }
}

/// `sceKernelMprotect(void *addr, size_t len, int prot)`. Same shape as the
/// POSIX call; applies the protection under `RAEEN_ENFORCE_MPROTECT`, no-op
/// otherwise.
fn hle_kernel_mprotect(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let prot = args.get(2).copied().unwrap_or(0) as u32;
    debug!("sceKernelMprotect(addr={addr:#x}, len={len:#x}, prot={prot:#x})");
    trace_guest_vmm(format_args!(
        "kernel-mprotect addr={addr:#x} len={len:#x} prot={prot:#x}"
    ));
    if ctx.mem.protect(addr, len, prot) {
        SCE_OK
    } else {
        SCE_KERNEL_ERROR_EINVAL
    }
}

/// `sceKernelMtypeprotect(void *addr, size_t len, int type, int prot)`.
///
/// The memory type selects/cache-tags the direct-memory pool on hardware.
/// Raeen does not yet expose that tag to the GPU memory tracker, but the host
/// protection transition is real and matches the shared mprotect path. This
/// behavior is cross-checked against SharpEmu and KytyPS5 (GPL-2.0/MIT).
fn hle_kernel_mtypeprotect(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let len = args.get(1).copied().unwrap_or(0);
    let memory_type = args.get(2).copied().unwrap_or(0) as u32;
    let prot = args.get(3).copied().unwrap_or(0) as u32;
    debug!(
        "sceKernelMtypeprotect(addr={addr:#x}, len={len:#x}, type={memory_type:#x}, prot={prot:#x})"
    );
    if addr == 0 || len == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
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

/// Whether an address is the kernel's signal-return trampoline. Raeen does
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

/// `sceKernelConvertUtcToLocaltime` / `ConvertLocaltimeToUtc`: Raeen's guest
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

    // `SceKernelVirtualQueryInfo`, 72 bytes (shadPS4
    // `core/libraries/kernel/memory.h` `OrbisVirtualQueryInfo`):
    //   0x00 start, 0x08 end, 0x10 offset, 0x18 protection, 0x1C memory_type,
    //   0x20 flags (is_flexible|is_direct|is_stack|is_pooled|is_committed),
    //   0x21 name[32].
    //
    // `offset`, `memory_type` and every kind bit used to be left zero, so a
    // direct-memory mapping read back as anonymous, type 0, offset 0. Titles
    // do not just make mappings, they verify them: Minecraft's embedded V8
    // maps direct memory (measured type 12) and queries the range immediately
    // afterwards — the disagreement tripped its `UNREACHABLE()` and killed the
    // process the moment the UI handled a button press.
    let mut payload = [0u8; INFO_SIZE];
    payload[0..8].copy_from_slice(&region.vaddr.to_le_bytes());
    payload[8..16].copy_from_slice(&end.to_le_bytes());
    payload[16..24].copy_from_slice(&region.direct_offset.to_le_bytes());
    payload[24..28].copy_from_slice(&region.protection.bits().to_le_bytes());
    payload[28..32].copy_from_slice(&region.direct_memory_type.to_le_bytes());
    payload[32] |= region.kind.query_flag_bit();
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

/// `sceKernelIsStack(addr, startOut, endOut)`: report whether `addr` lies in
/// a region the kernel tracks as a thread STACK. Returns 1 with the region's
/// bounds when it does; 0 otherwise.
///
/// Raeen records no `MappingKind::Stack` regions today (guest stacks are
/// arena-owned, outside the VMM's region table), so the honest answer for
/// every current query is 0. The outputs are still ZEROED in that case rather
/// than left untouched: the caller (libc's VM allocator probing whether a
/// range is a pthread stack) consumes them unconditionally, and stale stack
/// garbage there becomes an invalid fixed-range reservation — the failure
/// SharpEmu's `KernelIsStack` (GPL-2.0) documents, whose always-zero answer
/// this matches for non-stack ranges.
fn hle_is_stack(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let start_out = args.get(1).copied().unwrap_or(0);
    let end_out = args.get(2).copied().unwrap_or(0);
    debug!("sceKernelIsStack(addr={addr:#x}, startOut={start_out:#x}, endOut={end_out:#x})");

    let region = ctx.kernel.memory.region_containing(addr);
    let stack = region.filter(|r| r.kind == raeen_core::types::MappingKind::Stack);
    let (start, end) = match &stack {
        Some(region) => (region.vaddr, region.vaddr.saturating_add(region.size)),
        None => (0, 0),
    };
    if start_out != 0 && !ctx.mem.write(start_out, &start.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if end_out != 0 && !ctx.mem.write(end_out, &end.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    u64::from(stack.is_some())
}

/// `sceKernelQueryMemoryProtection(addr, startOut, endOut, protOut)`: report
/// the bounds and protection bits of the tracked mapping containing `addr`,
/// from the same region table `sceKernelVirtualQuery` reads. Any out-pointer
/// may be NULL. An address inside no tracked mapping is `ENOENT` (matching
/// SharpEmu's `KernelQueryMemoryProtection`, GPL-2.0) — distinguishing
/// "unmapped" from "mapped with no access", which `EACCES` would not.
///
/// `endOut` is the region's INCLUSIVE last byte (`base + size - 1`), the
/// convention SharpEmu documents for this call; `sceKernelVirtualQuery`'s
/// exclusive end is a different ABI and does not apply here.
fn hle_query_memory_protection(ctx: &HleContext, args: &[u64]) -> u64 {
    let addr = args.first().copied().unwrap_or(0);
    let start_out = args.get(1).copied().unwrap_or(0);
    let end_out = args.get(2).copied().unwrap_or(0);
    let prot_out = args.get(3).copied().unwrap_or(0);
    debug!(
        "sceKernelQueryMemoryProtection(addr={addr:#x}, startOut={start_out:#x}, \
         endOut={end_out:#x}, protOut={prot_out:#x})"
    );

    let Some(region) = ctx.kernel.memory.region_containing(addr) else {
        debug!("sceKernelQueryMemoryProtection: {addr:#x} is in no tracked mapping — ENOENT");
        return SCE_KERNEL_ERROR_ENOENT;
    };
    let inclusive_end = region.vaddr.saturating_add(region.size).saturating_sub(1);
    let prot = region.protection.bits() as i32;
    if start_out != 0 && !ctx.mem.write(start_out, &region.vaddr.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if end_out != 0 && !ctx.mem.write(end_out, &inclusive_end.to_le_bytes()) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if prot_out != 0 && !ctx.mem.write(prot_out, &prot.to_le_bytes()) {
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
    let flags = args.get(2).copied().unwrap_or(0) as u32;
    let page = raeen_core::PS5_PAGE_SIZE as u64;
    let requested_align = args.get(3).copied().unwrap_or(0);
    if addr_inout == 0 || len == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    if len % page != 0
        || (requested_align != 0 && (!requested_align.is_power_of_two() || requested_align < page))
    {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let align = requested_align.max(page);
    let mut requested_bytes = [0u8; 8];
    if !ctx.mem.read(addr_inout, &mut requested_bytes) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let requested = u64::from_le_bytes(requested_bytes);
    let fixed = flags & MAP_FIXED != 0;
    let Some(addr) = ctx.alloc.reserve_with_hint(requested, len, align, fixed) else {
        warn!(
            "sceKernelReserveVirtualRange: address-space reservation failed \
             (hint={requested:#x}, len={len:#x}, flags={flags:#x}, align={align:#x}, \
              fixed={fixed}) — ENOMEM"
        );
        // Out of address space is `ENOMEM`, the same code shadPS4 returns when
        // its VMA search finds nothing that fits.
        return SCE_KERNEL_ERROR_ENOMEM;
    };
    ctx.kernel.memory.record_mapping(addr, len, 0);
    trace_guest_vmm(format_args!(
        "reserve hint={requested:#x} len={len:#x} flags={flags:#x} \
         align={align:#x} fixed={fixed} -> {addr:#x}"
    ));
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
/// when the linker force-routes `__cxa_throw` here (RAEEN_TRAP_CXA_THROW);
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
    // RAEEN_TRACE_EINVAL / RAEEN_TRAP_CXA_THROW is set.
    if let Some(ring) = ctx.kernel.recent_hle_calls.get(&thread) {
        let recent = ring.lock().iter().cloned().collect::<Vec<_>>().join(" ");
        if !recent.is_empty() {
            warn!("  recent HLE calls: {recent}");
        }
    }
    // Walk the caller's stack for return addresses into the loaded image, so
    // the call chain INTO libc's terminate handler (and thus the throw
    // origin) is greppable. Diagnostic only; bounded and read-only.
    let chain = crate::guest_stack_code_addrs(ctx);
    if !chain.is_empty() {
        warn!("  fatal-thread stack code-addrs: {}", chain.join(" "));
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

fn exception_signal_allowed(signum: i32) -> bool {
    crate::exception::signal_allowed(signum)
}

/// Install/remove keep the process-visible handler table; `Raise` queues real
/// delivery to the target thread (see the [`crate::exception`] module).
fn hle_install_exception_handler(ctx: &HleContext, args: &[u64]) -> u64 {
    const SCE_KERNEL_ERROR_EAGAIN: u64 = 0x8002_000B;
    let signum = args.first().copied().unwrap_or(u64::MAX) as i32;
    let handler = args.get(1).copied().unwrap_or(0);
    if !exception_signal_allowed(signum) || handler == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    match ctx.kernel.exception_handlers.entry(signum) {
        dashmap::mapref::entry::Entry::Occupied(_) => SCE_KERNEL_ERROR_EAGAIN,
        dashmap::mapref::entry::Entry::Vacant(entry) => {
            entry.insert(handler);
            SCE_OK
        }
    }
}

fn hle_remove_exception_handler(ctx: &HleContext, args: &[u64]) -> u64 {
    let signum = args.first().copied().unwrap_or(u64::MAX) as i32;
    if !exception_signal_allowed(signum) {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    ctx.kernel.exception_handlers.remove(&signum);
    SCE_OK
}

/// `sceKernelRaiseException(thread, signum)`: raise `signum` at a named guest
/// thread, running the process handler installed for it.
///
/// Queues the exception against the target thread rather than calling the
/// handler here. A guest signal handler must execute on the *target* thread's
/// own stack and TLS — the raising thread cannot run it — so delivery happens at
/// that thread's next HLE safe point. For a self-raise the safe point is this
/// very call, so the handler runs as soon as this returns. See
/// [`crate::exception`] for the full model and the honest list of what delivery
/// does not yet reproduce.
///
/// `thread` is a `ScePthread`, which in this runtime *is* the guest thread id
/// (`GuestThreads::create` reports the same value it writes back), so it needs
/// no translation.
fn hle_raise_exception(ctx: &HleContext, args: &[u64]) -> u64 {
    let target_thread = args.first().copied().unwrap_or(0);
    let signum = args.get(1).copied().unwrap_or(u64::MAX) as i32;
    if target_thread == 0 || !(0..128).contains(&signum) {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    // No handler for this signal is not an error: the kernel accepts the raise
    // and there is simply no user callback to run.
    let Some(handler) = ctx.kernel.exception_handlers.get(&signum).map(|h| *h) else {
        debug!(
            target_thread = format_args!("{target_thread:#x}"),
            signum, "sceKernelRaiseException: no handler installed for this signal"
        );
        return SCE_OK;
    };
    let raised_by = ctx.guest_threads.current_thread();
    let replaced = ctx.kernel.queue_pending_exception(
        target_thread,
        raeen_kernel::PendingException {
            signum,
            handler,
            raised_by,
        },
    );
    if replaced {
        // Not fatal, but worth naming: the target has not reached a safe point
        // since the previous raise, so that one is superseded and never runs.
        debug!(
            target_thread = format_args!("{target_thread:#x}"),
            signum, "sceKernelRaiseException superseded an undelivered raise"
        );
    }
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
const ORBIS_MODE_CHARACTER: u16 = 0x21ff;

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
    if matches!(path.as_ref(), "/dev/random" | "/dev/urandom") {
        return write_orbis_device_stat(ctx, stat_out);
    }
    match ctx.kernel.filesystem.metadata(&path) {
        Ok(metadata) => write_orbis_stat(ctx, stat_out, &metadata),
        Err(error) => {
            debug!("sceKernelStat('{path}') failed: {error}");
            if std::env::var_os("RAEEN_DIAG_STAT_MISS").is_some() {
                static REPORTED: std::sync::LazyLock<dashmap::DashMap<String, ()>> =
                    std::sync::LazyLock::new(dashmap::DashMap::new);
                if REPORTED.len() < 256 && REPORTED.insert(path.to_string(), ()).is_none() {
                    warn!("sceKernelStat('{path}') -> ENOENT ({error})");
                }
            }
            SCE_KERNEL_ERROR_ENOENT
        }
    }
}

/// POSIX `stat(path, stat_out)`: the `-1` + `errno` twin of
/// [`hle_stat`]'s SCE return convention, for the plain POSIX spelling
/// imported from `libScePosix`.
fn hle_posix_stat(ctx: &HleContext, args: &[u64]) -> u64 {
    sce_result_posix(ctx, hle_stat(ctx, args))
}

fn write_orbis_device_stat(ctx: &HleContext, stat_out: u64) -> u64 {
    let mut stat = [0u8; ORBIS_STAT_SIZE];
    stat[4..8].copy_from_slice(&1u32.to_le_bytes());
    stat[8..10].copy_from_slice(&ORBIS_MODE_CHARACTER.to_le_bytes());
    stat[10..12].copy_from_slice(&1u16.to_le_bytes());
    if ctx.mem.write(stat_out, &stat) {
        SCE_OK
    } else {
        SCE_KERNEL_ERROR_EFAULT
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

fn hle_posix_rename(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_rename_core(ctx, args))
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
/// descriptors, a directory record (`S_IFDIR`, dirent-listing size, 0x8000
/// block size — shadPS4's directory fstat values) for directory fds, and a
/// zero-sized character-like record for console fds. A directory posing as a
/// zero-length regular file (the old behavior) breaks directory walkers that
/// branch on `st_mode`.
///
/// Returns this module's internal `-errno` convention, like every other
/// file-family primitive here, so [`file_result_posix`] and [`file_result_sce`]
/// can adapt it per export name. It previously returned `-9` (a raw internal
/// value) on a bad fd but `0x8002_000E` (an SCE code) on a fault, and was
/// registered **raw** under both `libScePosix::fstat` and
/// `libkernel::sceKernelFstat` — so the POSIX spelling never returned `-1`
/// (a guest's `if (fstat(...) == -1)` could not fire, and `errno` was never
/// set) and the SCE spelling reported `-9` instead of
/// `SCE_KERNEL_ERROR_EBADF`.
fn hle_fstat(ctx: &HleContext, args: &[u64]) -> u64 {
    let fd = args.first().copied().unwrap_or(0);
    let stat_out = args.get(1).copied().unwrap_or(0);
    let character_device = fd <= 2 || ctx.kernel.filesystem.is_random_device(fd as i32);
    let directory = !character_device && ctx.kernel.filesystem.is_directory(fd as i32);
    let size = if character_device {
        0
    } else if let Some(file_size) = ctx.kernel.filesystem.file_size(fd as i32) {
        file_size
    } else {
        warn!("fstat(fd={fd}): no file table backing — EBADF");
        return WRITE_EBADF;
    };
    if stat_out != 0 {
        let mut stat = [0u8; ORBIS_STAT_SIZE];
        let mode = if character_device {
            ORBIS_MODE_CHARACTER
        } else if directory {
            ORBIS_MODE_DIRECTORY
        } else {
            ORBIS_MODE_REGULAR
        };
        stat[8..10].copy_from_slice(&mode.to_le_bytes());
        stat[10..12].copy_from_slice(&1u16.to_le_bytes());
        stat[72..80].copy_from_slice(&size.to_le_bytes());
        let blocks: u64 = if directory { 8 } else { size.div_ceil(512) };
        stat[80..88].copy_from_slice(&blocks.to_le_bytes());
        let blksize: u32 = if directory { 0x8000 } else { 512 };
        stat[88..92].copy_from_slice(&blksize.to_le_bytes());
        if !ctx.mem.write(stat_out, &stat) {
            return FILE_EFAULT;
        }
    }
    SCE_OK
}

/// POSIX `fstat`: `-1` with `errno` set.
fn hle_posix_fstat(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_posix(ctx, hle_fstat(ctx, args))
}

/// `sceKernelFstat`: a negative `SCE_KERNEL_ERROR_*` code.
fn hle_sce_fstat(ctx: &HleContext, args: &[u64]) -> u64 {
    file_result_sce(hle_fstat(ctx, args))
}

/// `scePthreadGetaffinity(thread, mask_out)`: report the CPU cores this thread
/// may run on. No scheduler binds guest threads to host cores yet, so every
/// core a PS5 title is granted is reported available. `SceKernelCpumask` is the
/// 64-bit ABI type; 0x7f is the 7-core mask a title sees (core 7 is the OS's).
fn hle_pthread_getaffinity(ctx: &HleContext, args: &[u64]) -> u64 {
    let mask_out = args.get(1).copied().unwrap_or(0);
    if mask_out != 0 {
        let _ = ctx.mem.write(mask_out, &0x7fu64.to_le_bytes());
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
    let trace_once = std::env::var_os("RAEEN_TRACE_PTHREAD_ONCE").is_some();
    let mut logged_wait = false;
    loop {
        match ctx.mem.atomic_load_u32(once) {
            Some(ONCE_DONE) => {
                if trace_once {
                    info!(
                        "pthread_once trace: thread={} once={once:#x} init={init:#x} already done",
                        ctx.guest_threads.current_thread()
                    );
                }
                return SCE_OK;
            }
            Some(ONCE_IN_PROGRESS) => {
                if ctx.guest_threads.process_is_terminating() {
                    return SCE_KERNEL_ERROR_EAGAIN;
                }
                if trace_once && !logged_wait {
                    info!(
                        "pthread_once trace: thread={} once={once:#x} init={init:#x} waiting",
                        ctx.guest_threads.current_thread()
                    );
                    logged_wait = true;
                }
                // Back off instead of spinning. The initializer runs on
                // ANOTHER guest thread, so a bare `yield_now` here burns a
                // full host core racing the very thread it waits on — the
                // same starvation the pthread mutex/rwlock waits just traded
                // for parked waits. There is no condvar to park on (the flag
                // lives in guest memory and is polled), so a short sleep is
                // the honest equivalent: `once` initializers are rare and
                // short, and 200us is invisible next to one.
                std::thread::sleep(std::time::Duration::from_micros(200));
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

    if trace_once {
        info!(
            "pthread_once trace: thread={} once={once:#x} init={init:#x} claimed callback",
            ctx.guest_threads.current_thread()
        );
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

/// POSIX `clock_getres(clockId, struct timespec *res)`: the plain-named alias
/// of [`hle_clock_getres`]. libKernel exports the same routine under two NIDs
/// (SharpEmu #450, `smIj7eqzZE8`). Identical 1 ns resolution, but POSIX permits
/// a NULL `res` (the caller only wants to validate the clock id), which is
/// accepted as a success rather than faulting.
fn hle_clock_getres_posix(ctx: &HleContext, args: &[u64]) -> u64 {
    let res = args.get(1).copied().unwrap_or(0);
    if res == 0 {
        return SCE_OK;
    }
    hle_clock_getres(ctx, args)
}

/// POSIX `getpagesize()`: the guest page granularity. Reports the PS5's 16 KiB
/// `OrbisPageSize`, the granularity every mapping call in this HLE aligns
/// against — NOT the host's 4 KiB (SharpEmu #450). An allocator that rounded to
/// the host value would produce sub-page offsets that `mmap`/`mprotect` reject
/// for misalignment.
fn hle_getpagesize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    raeen_core::PS5_PAGE_SIZE as u64
}

/// Frequency (Hz) of the process-time counter Raeen exposes: a nanosecond
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
    if std::env::var_os("RAEEN_TRACE_SPIN").is_some() && ctx.caller_return_addr != 0 {
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
    if std::env::var_os("RAEEN_TRACE_PROCPARAM").is_some() {
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

    #[test]
    fn aio_initialize_param_ignores_stale_rsi_and_preserves_frame_guard() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let param = 0x100;
        let guard = param + AIO_INIT_PARAM_SIZE as u64;

        assert!(mem.write(param, &[0xAA; AIO_INIT_PARAM_SIZE]));
        assert!(mem.write(guard, &[0xCC; 16]));
        assert_eq!(
            hle_aio_initialize_param(&ctx, &[param, 0x88]),
            SCE_OK,
            "the measured stale RSI is not an ABI size argument"
        );

        let mut initialized = [0xFF; AIO_INIT_PARAM_SIZE];
        assert!(mem.read(param, &mut initialized));
        assert_eq!(initialized, [0; AIO_INIT_PARAM_SIZE]);
        let mut guard_bytes = [0; 16];
        assert!(mem.read(guard, &mut guard_bytes));
        assert_eq!(
            guard_bytes, [0xCC; 16],
            "fixed-size init must not erase the adjacent stack canary"
        );

        assert_eq!(
            hle_aio_initialize_param(&ctx, &[0]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(
            hle_aio_initialize_param(&ctx, &[0x1000 - 8]),
            SCE_KERNEL_ERROR_EFAULT
        );
    }

    /// DirectMemoryQuery reports the recorded allocation containing (or, with
    /// flags==1, following) the queried offset — shadPS4's OrbisQueryInfo
    /// `{start, end, memoryType}` — and EACCES for unallocated space.
    ///
    /// A direct-memory mapping must read back through `sceKernelVirtualQuery`
    /// as what the guest actually mapped: `is_direct` set, the physical offset
    /// echoed at 0x10, and the allocation `type` at 0x1C. All three used to be
    /// zero, so a title verifying its own mapping saw "anonymous, type 0,
    /// offset 0" — which is what tripped Minecraft's embedded V8 into
    /// `UNREACHABLE()` on the first button press.
    ///
    /// Layout pinned against shadPS4's `OrbisVirtualQueryInfo`
    /// (`core/libraries/kernel/memory.h`): start, end, offset, protection,
    /// memory_type, flags{flexible,direct,stack,pooled,committed}, name[32].
    #[test]
    fn virtual_query_reports_direct_mappings_with_offset_and_type() {
        const IS_DIRECT: u8 = 0x02;
        const IS_COMMITTED: u8 = 0x10;
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let phys = 0x5bc1_0000u64;
        let mapped = 0x40_0000u64;
        kernel.memory.record_mapping_of_kind(
            mapped,
            0x8000,
            0x3,
            raeen_core::types::MappingKind::Direct,
            phys,
            12, // the type Minecraft's allocator measured
        );

        assert_eq!(
            hle_virtual_query(&ctx, &[mapped + 0x1000, 0, 0x100, 72]),
            SCE_OK
        );
        let mut info = [0u8; 72];
        assert!(mem.read(0x100, &mut info));

        assert_eq!(u64::from_le_bytes(info[0..8].try_into().unwrap()), mapped);
        assert_eq!(
            u64::from_le_bytes(info[8..16].try_into().unwrap()),
            mapped + 0x8000
        );
        assert_eq!(
            u64::from_le_bytes(info[16..24].try_into().unwrap()),
            phys,
            "offset must echo the direct-memory physical start"
        );
        assert_eq!(
            i32::from_le_bytes(info[28..32].try_into().unwrap()),
            12,
            "memory_type must echo the allocation type"
        );
        assert_eq!(info[32] & IS_DIRECT, IS_DIRECT, "is_direct must be set");
        assert_eq!(info[32] & IS_COMMITTED, IS_COMMITTED);

        // An anonymous mapping still reports no kind bits and no offset/type.
        kernel.memory.record_mapping(0x80_0000, 0x4000, 0x3);
        assert_eq!(hle_virtual_query(&ctx, &[0x80_0000, 0, 0x200, 72]), SCE_OK);
        let mut anon = [0u8; 72];
        assert!(mem.read(0x200, &mut anon));
        assert_eq!(u64::from_le_bytes(anon[16..24].try_into().unwrap()), 0);
        assert_eq!(i32::from_le_bytes(anon[28..32].try_into().unwrap()), 0);
        assert_eq!(anon[32] & IS_DIRECT, 0, "anonymous is not direct");
    }

    /// `sceKernelAioInitializeParam` must not memset a caller frame.
    ///
    /// Its `size` argument is a guest register, and the block it names is
    /// frequently a caller local. Zeroing `size` bytes there — up to 64 KiB —
    /// takes out the caller's neighbouring locals, its saved registers, and its
    /// `__stack_chk_guard` canary, which is exactly the `__stack_chk_fail`
    /// death GTA V hits. Off-stack (an engine-allocator block) the clear still
    /// happens in full, because that is the useful behavior.
    #[test]
    fn aio_initialize_param_never_zeroes_a_caller_frame() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x2000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Guest thread 1 (what the test scheduler reports) owns [0x1000, 0x2000)
        // as its stack — the same registration the runtime performs per thread.
        kernel.guest_thread_stacks.insert(1, (0x1000, 0x2000));

        // A caller local at 0x1400 with live frame bytes above it.
        assert!(mem.write(0x1400, &[0xAAu8; 0x400]));
        assert_eq!(hle_aio_initialize_param(&ctx, &[0x1400, 0x400]), SCE_OK);
        let mut frame = [0u8; 0x400];
        assert!(mem.read(0x1400, &mut frame));
        assert!(
            frame.iter().all(|&b| b == 0xAA),
            "a stack-resident param block must not be bulk-zeroed"
        );

        // The same call against a heap block still clears it — but only the
        // fixed ABI block, never a byte past it. The write length comes from
        // `AIO_INIT_PARAM_SIZE`, not from the caller's register (rule 3), so
        // the bytes above the block must survive even off-stack.
        assert!(mem.write(0x200, &[0xAAu8; 0x40]));
        assert_eq!(hle_aio_initialize_param(&ctx, &[0x200, 0x40]), SCE_OK);
        let mut heap = [0xFFu8; 0x40];
        assert!(mem.read(0x200, &mut heap));
        assert!(
            heap[..AIO_INIT_PARAM_SIZE].iter().all(|&b| b == 0),
            "an off-stack param block keeps the defined-default zero fill"
        );
        assert!(
            heap[AIO_INIT_PARAM_SIZE..].iter().all(|&b| b == 0xAA),
            "nothing past the fixed ABI block may be written"
        );

        // A null param is still rejected.
        assert_eq!(
            hle_aio_initialize_param(&ctx, &[0, 0x40]),
            SCE_KERNEL_ERROR_EINVAL
        );
        // A hostile/stale size register is simply ignored now: the block is a
        // fixed 0x3c, so a bogus 0x20000 can neither fault nor widen the write.
        assert!(mem.write(0x200, &[0xAAu8; 0x40]));
        assert_eq!(hle_aio_initialize_param(&ctx, &[0x200, 0x20000]), SCE_OK);
        let mut after = [0xFFu8; 0x40];
        assert!(mem.read(0x200, &mut after));
        assert!(
            after[AIO_INIT_PARAM_SIZE..].iter().all(|&b| b == 0xAA),
            "a stale size register must not widen the write"
        );
    }

    #[test]
    fn direct_memory_query_reports_recorded_regions() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // A real host file with a known size.
        let host = std::env::temp_dir().join(format!(
            "raeen_apr_test_{}_{}.bin",
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

    /// Regression for the APR stale-record re-fire: after a command buffer
    /// completes, its records must never execute again unless the title
    /// explicitly rewinds via sceAmprAprCommandBufferReset. Completion also
    /// consumes the host write cursor (`ampr_write_offsets`).
    #[test]
    fn apr_completion_does_not_reexecute_stale_readfile_records() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let data = [0x5Au8; 64];
        let host = std::env::temp_dir().join(format!(
            "raeen_apr_stale_test_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&host, data).expect("temp file");
        let id = kernel.appr_register_file("/app0/assets/stale.bin", host.display().to_string());

        // Command-buffer struct @0x100, record buffer @0x200, destination @0x300.
        assert!(mem.write(0x108, &0x200u64.to_le_bytes())); // cb data ptr
        assert!(mem.write(0x110, &0x100u64.to_le_bytes())); // cb size
        kernel.ampr_write_offsets.insert(0x100, 0);
        kernel.ampr_command_counts.insert(0x100, 0);
        assert_eq!(
            crate::libsce_ampr::hle_apr_read_file(
                &ctx,
                &[0x100, 0, 0, u64::from(id), 0x300, 64, 0]
            ),
            SCE_OK
        );
        assert_eq!(hle_apr_submit(&ctx, &[0x100, 0]), SCE_OK);
        // First completion leaves the (eagerly read) bytes in place...
        let mut probe = [0u8; 64];
        assert!(mem.read(0x300, &mut probe));
        assert_eq!(probe, data);
        // ...and consumes the write cursor so nothing can re-fire.
        assert!(!kernel.ampr_write_offsets.contains_key(&0x100));

        // The title repurposes the destination; a re-submit WITHOUT Reset
        // must not re-execute the stale ReadFile record into it.
        assert!(mem.write(0x300, &[0xEEu8; 64]));
        assert_eq!(hle_apr_submit(&ctx, &[0x100, 0]), SCE_OK);
        assert!(mem.read(0x300, &mut probe));
        assert_eq!(probe, [0xEEu8; 64], "stale ReadFile record re-fired");
        let _ = std::fs::remove_file(&host);
    }

    /// The ForEach resolve variants register paths in the APR table without
    /// writing through their unverified trailing arguments.
    #[test]
    fn apr_foreach_registers_paths_without_writing_out_arrays() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
        kernel.set_unwind_modules(vec![raeen_kernel::UnwindModuleInfo {
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
        let kernel = raeen_kernel::OrbisKernel::new();
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

    /// The Unity/IL2CPP blocker. `sceKernelDlsym(0, ...)` must resolve against
    /// the **main program**, per KytyPS5's `RuntimeLinker::FindProgramById`
    /// ("Id 0 is reserved for main program"). Handle 0 used to be treated as an
    /// ordinary module id, which matches nothing because ids start at 1.
    #[test]
    fn dlsym_null_handle_resolves_against_the_main_program() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let main = kernel.register_lle_module(
            "eboot.bin".to_string(),
            0x1000_0000,
            0x10000,
            None,
            true,
            [(nid_of_symbol(b"MainOnlySymbol"), 0x1000_1111)],
        );
        // A second module, registered later, must not be what handle 0 names.
        let plugin = kernel.register_lle_module(
            "plugin.prx".to_string(),
            0x2000_0000,
            0x10000,
            None,
            true,
            [(nid_of_symbol(b"PluginOnlySymbol"), 0x2000_2222)],
        );
        assert!(main < plugin, "the executable is registered first");
        assert_eq!(kernel.main_lle_module_handle(), Some(main));

        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, b"MainOnlySymbol\0"));

        assert_eq!(hle_dlsym(&ctx, &[0, 0x100, 0x200]), SCE_OK);
        let mut out = [0u8; 8];
        assert!(mem.read(0x200, &mut out));
        assert_eq!(u64::from_le_bytes(out), 0x1000_1111);
    }

    /// SharpEmu's `DispatchKernelDynlibDlsym` falls back to a process-wide
    /// symbol sweep when the named handle misses. The sweep is load-ordered so
    /// two modules exporting one NID always resolve the same way.
    #[test]
    fn dlsym_falls_back_to_other_loaded_modules_in_load_order() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let nid = nid_of_symbol(b"SharedSymbol");
        let first = kernel.register_lle_module(
            "eboot.bin".to_string(),
            0x1000_0000,
            0x10000,
            None,
            true,
            [],
        );
        let second = kernel.register_lle_module(
            "a.prx".to_string(),
            0x2000_0000,
            0x10000,
            None,
            true,
            [(nid, 0x2000_2222)],
        );
        let third = kernel.register_lle_module(
            "b.prx".to_string(),
            0x3000_0000,
            0x10000,
            None,
            true,
            [(nid, 0x3000_3333)],
        );
        assert!(first < second && second < third);

        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, b"SharedSymbol\0"));

        // Handle 0 -> main program (no exports) -> sweep finds `a.prx`, the
        // earlier-loaded of the two exporters, not `b.prx`.
        assert_eq!(hle_dlsym(&ctx, &[0, 0x100, 0x200]), SCE_OK);
        let mut out = [0u8; 8];
        assert!(mem.read(0x200, &mut out));
        assert_eq!(u64::from_le_bytes(out), 0x2000_2222);
        assert_eq!(
            kernel.resolve_lle_export_anywhere(nid),
            Some((second, 0x2000_2222))
        );
    }

    /// A symbol that exists nowhere must still fail, loudly and without
    /// touching the out-pointer — the guest calls straight through whatever
    /// `dlsym` writes there, so a fabricated address is worse than an error.
    #[test]
    fn dlsym_null_handle_still_fails_for_a_genuinely_absent_symbol() {
        let kernel = raeen_kernel::OrbisKernel::new();
        kernel.register_lle_module(
            "eboot.bin".to_string(),
            0x1000_0000,
            0x10000,
            None,
            true,
            [(nid_of_symbol(b"SomethingElse"), 0x1000_1111)],
        );
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, b"NoSuchSymbolAnywhere\0"));
        assert!(mem.write(0x200, &0xDEAD_BEEF_u64.to_le_bytes()));

        assert_eq!(hle_dlsym(&ctx, &[0, 0x100, 0x200]), SCE_KERNEL_ERROR_ESRCH);
        let mut out = [0u8; 8];
        assert!(mem.read(0x200, &mut out));
        assert_eq!(
            u64::from_le_bytes(out),
            0xDEAD_BEEF,
            "the out-pointer must be untouched on a miss"
        );
    }

    /// With no module registered at all there is no main program, so handle 0
    /// resolves to nothing — and that is a *different* diagnosis from "the
    /// symbol is missing", which is why the miss path reports it separately.
    #[test]
    fn dlsym_null_handle_with_no_modules_is_esrch_not_a_panic() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, b"scriptingGetMem\0"));
        assert_eq!(kernel.main_lle_module_handle(), None);
        assert_eq!(hle_dlsym(&ctx, &[0, 0x100, 0x200]), SCE_KERNEL_ERROR_ESRCH);
    }

    /// `dlsym` must also see Raeen's own HLE trampolines: they are how a
    /// symbol the *emulator* implements becomes a guest-callable address.
    /// This is the path `scriptingGetMem` resolves through — it is not an
    /// export of any guest module and never will be.
    #[test]
    fn dlsym_resolves_scripting_get_mem_through_the_published_hle_trampoline() {
        let kernel = raeen_kernel::OrbisKernel::new();
        kernel.register_lle_module(
            "eboot.bin".to_string(),
            0x1000_0000,
            0x10000,
            None,
            true,
            [],
        );
        // Stands in for `publish_hle_exports_for_dlsym`, which the runtime
        // calls with the process-wide trampoline table before guest entry.
        const TRAMPOLINE: u64 = 0x0000_4000_0000_00A8;
        kernel.register_hle_export_addr("scriptingGetMem", TRAMPOLINE);

        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, b"scriptingGetMem\0"));

        assert_eq!(hle_dlsym(&ctx, &[0, 0x100, 0x200]), SCE_OK);
        let mut out = [0u8; 8];
        assert!(mem.read(0x200, &mut out));
        assert_eq!(u64::from_le_bytes(out), TRAMPOLINE);
    }

    /// A guest export of the same name must win over the HLE trampoline: the
    /// title's own implementation is the authoritative one, and the HLE entry
    /// is only a fallback for names nothing exports.
    #[test]
    fn dlsym_prefers_a_guest_module_export_over_the_hle_trampoline() {
        let kernel = raeen_kernel::OrbisKernel::new();
        kernel.register_lle_module(
            "eboot.bin".to_string(),
            0x1000_0000,
            0x10000,
            None,
            true,
            [(nid_of_symbol(b"scriptingGetMem"), 0x1000_7777)],
        );
        kernel.register_hle_export_addr("scriptingGetMem", 0x0000_4000_0000_00A8);

        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, b"scriptingGetMem\0"));

        assert_eq!(hle_dlsym(&ctx, &[0, 0x100, 0x200]), SCE_OK);
        let mut out = [0u8; 8];
        assert!(mem.read(0x200, &mut out));
        assert_eq!(u64::from_le_bytes(out), 0x1000_7777);
    }

    /// A null out-pointer is EFAULT before anything else, matching KytyPS5's
    /// `addr == nullptr` guard.
    #[test]
    fn dlsym_rejects_a_null_out_pointer_and_an_unreadable_symbol() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, b"anything\0"));
        assert_eq!(hle_dlsym(&ctx, &[0, 0x100, 0]), SCE_KERNEL_ERROR_EFAULT);
        assert_eq!(hle_dlsym(&ctx, &[0, 0, 0x200]), SCE_KERNEL_ERROR_EFAULT);
        assert_eq!(
            hle_dlsym(&ctx, &[0, 0xDEAD_0000, 0x200]),
            SCE_KERNEL_ERROR_EFAULT
        );
    }

    /// `scriptingGetMem(alignment, size)` returns an aligned guest block.
    /// The signature is KytyPS5's `KernelApplicationHeapGetMem`.
    #[test]
    fn scripting_get_mem_allocates_aligned_guest_memory() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x10_0000);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let addr = hle_scripting_get_mem(&ctx, &[0x40, 0x200]);
        assert_ne!(addr, 0, "a healthy heap must satisfy the request");
        assert_eq!(addr % 0x40, 0, "the requested alignment must be honored");

        // Alignment below 0x10 is clamped up, exactly as KytyPS5 does.
        let small = hle_scripting_get_mem(&ctx, &[1, 0x10]);
        assert_ne!(small, 0);
        assert_eq!(small % 0x10, 0);

        assert_eq!(hle_scripting_free_mem(&ctx, &[addr]), SCE_OK);
        assert_eq!(
            hle_scripting_free_mem(&ctx, &[0]),
            SCE_OK,
            "freeing null is a no-op, not a fault"
        );
    }

    /// The signature self-test: a non-power-of-two first argument means our
    /// `(alignment, size)` reading is wrong for this title, so return null
    /// rather than a pointer the guest would write `size` bytes through.
    #[test]
    fn scripting_get_mem_refuses_a_non_power_of_two_alignment() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x10_0000);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_scripting_get_mem(&ctx, &[0x30, 0x200]), 0);
    }

    /// The dlsym-only exports must be registered, or `load_process` would
    /// reserve trampolines for functions no dispatch could service.
    #[test]
    fn dlsym_reserved_exports_are_all_registered() {
        let registry = HleRegistry::new();
        for (library, function) in DLSYM_RESERVED_EXPORTS {
            assert!(
                registry.is_implemented(library, function),
                "{library}::{function} is reserved for dlsym but not implemented"
            );
        }
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
            ("libkernel", "sceKernelMtypeprotect"),
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_load_start_module(&ctx, &[0xDEAD_0000, 0, 0, 0, 0, 0]),
            SCE_KERNEL_ERROR_EFAULT
        );
    }

    #[test]
    fn stop_unload_validates_the_handle_and_dlsym_is_honest_esrch() {
        let kernel = raeen_kernel::OrbisKernel::new();
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

        // KytyPS5's `KernelDlsym` answers ESRCH for a symbol absent from a
        // known module, not ENOENT.
        assert!(mem.write(0x200, b"sceSomeFunction\0"));
        assert_eq!(
            hle_dlsym(&ctx, &[handle, 0x200, 0x400]),
            SCE_KERNEL_ERROR_ESRCH
        );
        let mut untouched = [0u8; 8];
        assert!(mem.read(0x400, &mut untouched));
        assert_eq!(
            u64::from_le_bytes(untouched),
            0,
            "a miss must leave the out-pointer alone — the guest calls through whatever is there"
        );
    }

    /// M1 hardening: the clock functions write real, plausible time into
    /// their guest out-params instead of leaving them zero.
    #[test]
    fn gettimeofday_and_clock_gettime_write_real_time() {
        let kernel = raeen_kernel::OrbisKernel::new();
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

    /// POSIX `clock_getres` (plain-named alias) tolerates a NULL `res` per POSIX,
    /// writes the 1 ns resolution otherwise, and `getpagesize` reports the 16 KiB
    /// Orbis page — both SharpEmu #450 exports registered under libkernel.
    #[test]
    fn posix_clock_getres_and_getpagesize() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // NULL res is a POSIX-legal success (unlike the faulting sce* form).
        assert_eq!(hle_clock_getres_posix(&ctx, &[CLOCK_MONOTONIC, 0]), SCE_OK);
        // A real out-pointer receives {tv_sec=0, tv_nsec=1}.
        assert_eq!(
            hle_clock_getres_posix(&ctx, &[CLOCK_MONOTONIC, 0x100]),
            SCE_OK
        );
        let mut ts = [0u8; 16];
        assert!(mem.read(0x100, &mut ts));
        assert_eq!(i64::from_le_bytes(ts[0..8].try_into().unwrap()), 0);
        assert_eq!(i64::from_le_bytes(ts[8..16].try_into().unwrap()), 1);

        // getpagesize reports the 16 KiB Orbis page, not the host 4 KiB.
        assert_eq!(hle_getpagesize(&ctx, &[]), raeen_core::PS5_PAGE_SIZE as u64);
        assert_eq!(hle_getpagesize(&ctx, &[]), 0x4000);

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libkernel", "clock_getres"));
        assert!(registry.is_implemented("libkernel", "getpagesize"));
    }

    /// Real VFS-backed file I/O: a homebrew opens a file under /app0,
    /// reads its bytes into a guest buffer, seeks, reads again, and closes —
    /// all against a real host temp file mounted into the VFS.
    #[test]
    fn savedata_write_through_hle_open_write_close_persists_to_host() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp = std::env::temp_dir().join(format!("raeen-hle-savewrite-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        // open("/app0/save.dat", O_WRONLY|O_CREAT|O_TRUNC) through the HLE.
        assert!(mem.write(0x100, b"/app0/save.dat\0"));
        use raeen_kernel::filesystem::open_flags::*;
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
        assert!(registry.is_implemented("libkernel", "sceKernelFsync"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn savedata_write_larger_than_one_mib_is_not_truncated() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let payload_len = (WRITE_CHUNK_BYTES + 0x34567) as usize;
        let mem = crate::TestMemory::new(payload_len + 0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp =
            std::env::temp_dir().join(format!("raeen-hle-large-savewrite-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        assert!(mem.write(0x100, b"/app0/large-save.dat\0"));
        use raeen_kernel::filesystem::open_flags::*;
        let fd = hle_open(&ctx, &[0x100, (O_WRONLY | O_CREAT | O_TRUNC) as u64, 0o644]);
        assert!((fd as i64) >= 3, "open must return a real fd, got {fd:#x}");

        let payload = vec![0xA5; payload_len];
        assert!(mem.write(0x800, &payload));
        assert_eq!(
            hle_write(&ctx, &[fd, 0x800, payload_len as u64]),
            payload_len as u64,
            "a valid large save write must report the full transfer"
        );
        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        assert_eq!(
            std::fs::read(tmp.join("large-save.dat")).unwrap(),
            payload,
            "the bytes after the old 1 MiB staging limit must reach the host file"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn rename_truncate_unlink_and_rmdir_mutate_the_host_through_the_vfs() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        // Nonzero base: the errno slot is arena-allocated, and address 0 would
        // read as "no errno slot".
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp = std::env::temp_dir().join(format!("raeen-hle-fsmut-{}", std::process::id()));
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

    /// `pwrite`/`ftruncate` through the HLE surface: positional writes land at
    /// their offset without moving the cursor, truncation resizes the open
    /// descriptor (and survives the flush-on-close), and the SCE/POSIX return
    /// conventions wrap the same core.
    #[test]
    fn pwrite_and_ftruncate_go_through_the_vfs_like_write_and_truncate() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        // Nonzero alloc base: the POSIX adapters write `errno` through an
        // arena-allocated slot, and address 0 reads as "no errno slot".
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp = std::env::temp_dir().join(format!("raeen-hle-pwrite-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        // open("/app0/blob.bin", O_RDWR|O_CREAT|O_TRUNC) and seed it.
        assert!(mem.write(0x100, b"/app0/blob.bin\0"));
        use raeen_kernel::filesystem::open_flags::*;
        let flags = (O_RDWR | O_CREAT | O_TRUNC) as u64;
        let fd = hle_open(&ctx, &[0x100, flags, 0o644]);
        assert!((fd as i64) >= 3, "open must return a real fd, got {fd:#x}");
        assert!(mem.write(0x200, b"aaaa"));
        assert_eq!(hle_write(&ctx, &[fd, 0x200, 4]), 4);

        // sceKernelPwrite at offset 1: lands at the offset, cursor stays at 4.
        assert!(mem.write(0x210, b"ZZ"));
        assert_eq!(hle_sce_pwrite(&ctx, &[fd, 0x210, 2, 1]), 2);
        assert_eq!(
            hle_lseek(&ctx, &[fd, 0, 1]),
            4,
            "pwrite must not move the cursor"
        );
        let mut back = [0u8; 4];
        assert_eq!(hle_pread(&ctx, &[fd, 0x300, 4, 0]), 4);
        assert!(mem.read(0x300, &mut back));
        assert_eq!(&back, b"aZZa");

        // POSIX pwrite failure shape: a negative offset is -1 + EINVAL (22).
        assert_eq!(
            hle_posix_pwrite(&ctx, &[fd, 0x210, 2, u64::MAX]),
            (-1i64) as u64
        );
        let errno_slot = hle_error_addr(&ctx, &[]);
        let mut errno = [0u8; 4];
        assert!(mem.read(errno_slot, &mut errno));
        assert_eq!(i32::from_le_bytes(errno), 22, "errno must hold EINVAL");

        // sceKernelFtruncate shortens the open descriptor; the cursor is
        // untouched (POSIX keeps the file offset), the flush keeps 3 bytes.
        assert_eq!(hle_sce_ftruncate(&ctx, &[fd, 3]), SCE_OK);
        assert_eq!(
            hle_lseek(&ctx, &[fd, 0, 1]),
            4,
            "ftruncate must not move the cursor"
        );
        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        assert_eq!(std::fs::read(tmp.join("blob.bin")).unwrap(), b"aZZ");

        // ftruncate on a read-only fd is EBADF (SCE encoding); a negative
        // length is EINVAL.
        let fd = hle_open(&ctx, &[0x100, O_RDONLY as u64, 0]);
        assert!((fd as i64) >= 3);
        assert_eq!(hle_sce_ftruncate(&ctx, &[fd, 1]), 0x8002_0009);
        assert_eq!(
            hle_posix_ftruncate(&ctx, &[fd, (-1i64) as u64]),
            (-1i64) as u64
        );
        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        // Unknown fds are EBADF for both calls.
        assert_eq!(hle_sce_pwrite(&ctx, &[0x7fff, 0x210, 1, 0]), 0x8002_0009);
        assert_eq!(hle_sce_ftruncate(&ctx, &[0x7fff, 1]), 0x8002_0009);

        let registry = HleRegistry::new();
        for (lib, name) in [
            ("libkernel", "sceKernelPwrite"),
            ("libkernel", "pwrite"),
            ("libScePosix", "pwrite"),
            ("libkernel", "sceKernelFtruncate"),
            ("libkernel", "ftruncate"),
            ("libScePosix", "ftruncate"),
        ] {
            assert!(
                registry.is_implemented(lib, name),
                "{lib}::{name} must be registered"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The plain POSIX spellings `rename`/`stat`: same VFS behavior as the
    /// sce* forms, `-1` + `errno` on failure.
    #[test]
    fn posix_rename_and_stat_adapt_the_sce_metadata_calls() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp = std::env::temp_dir().join(format!("raeen-hle-posixfs-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("old.dat"), b"0123456789").unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        // rename() moves the real host file; a missing source is -1 + ENOENT.
        assert!(mem.write(0x100, b"/app0/old.dat\0"));
        assert!(mem.write(0x140, b"/app0/new.dat\0"));
        assert_eq!(hle_posix_rename(&ctx, &[0x100, 0x140]), 0);
        assert_eq!(std::fs::read(tmp.join("new.dat")).unwrap(), b"0123456789");
        assert_eq!(hle_posix_rename(&ctx, &[0x100, 0x140]), (-1i64) as u64);
        let errno_slot = hle_error_addr(&ctx, &[]);
        let mut errno = [0u8; 4];
        assert!(mem.read(errno_slot, &mut errno));
        assert_eq!(i32::from_le_bytes(errno), 2, "errno must hold ENOENT");

        // stat() writes the 120-byte Orbis record (regular-file mode + size).
        assert_eq!(hle_posix_stat(&ctx, &[0x140, 0x400]), 0);
        let mut mode = [0u8; 2];
        assert!(mem.read(0x400 + 8, &mut mode));
        assert_eq!(u16::from_le_bytes(mode), ORBIS_MODE_REGULAR);
        let mut size = [0u8; 8];
        assert!(mem.read(0x400 + 72, &mut size));
        assert_eq!(u64::from_le_bytes(size), 10);
        // A missing path is -1 + ENOENT.
        assert!(mem.write(0x180, b"/app0/absent.dat\0"));
        assert_eq!(hle_posix_stat(&ctx, &[0x180, 0x400]), (-1i64) as u64);
        assert!(mem.read(errno_slot, &mut errno));
        assert_eq!(i32::from_le_bytes(errno), 2);

        let registry = HleRegistry::new();
        for (lib, name) in [
            ("libkernel", "rename"),
            ("libScePosix", "rename"),
            ("libkernel", "stat"),
            ("libScePosix", "stat"),
        ] {
            assert!(
                registry.is_implemented(lib, name),
                "{lib}::{name} must be registered"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `sceKernelQueryMemoryProtection` reports the tracked mapping's bounds
    /// (inclusive end) and protection bits; `sceKernelIsStack` answers 1 with
    /// bounds only for a region recorded as a stack.
    #[test]
    fn memory_protection_query_and_is_stack_read_the_region_table() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // A direct mapping, R+W.
        kernel.memory.record_mapping_of_kind(
            0x10_0000,
            0x4000,
            0x3,
            raeen_core::types::MappingKind::Direct,
            0x80,
            12,
        );
        assert_eq!(
            hle_query_memory_protection(&ctx, &[0x10_0100, 0x100, 0x108, 0x110]),
            SCE_OK
        );
        let mut words = [0u8; 24];
        assert!(mem.read(0x100, &mut words));
        assert_eq!(
            u64::from_le_bytes(words[0..8].try_into().unwrap()),
            0x10_0000
        );
        assert_eq!(
            u64::from_le_bytes(words[8..16].try_into().unwrap()),
            0x10_0000 + 0x4000 - 1,
            "end is the inclusive last byte"
        );
        assert_eq!(i32::from_le_bytes(words[16..20].try_into().unwrap()), 0x3);
        // NULL out-pointers are permitted; an untracked address is ENOENT.
        assert_eq!(
            hle_query_memory_protection(&ctx, &[0x10_0100, 0, 0, 0]),
            SCE_OK
        );
        assert_eq!(
            hle_query_memory_protection(&ctx, &[0x30_0000, 0x100, 0x108, 0x110]),
            SCE_KERNEL_ERROR_ENOENT
        );

        // IsStack: 0 (with zeroed outputs) for a non-stack region...
        assert!(mem.write(0x120, &0xAAAA_AAAAu64.to_le_bytes()));
        assert!(mem.write(0x128, &0xBBBB_BBBBu64.to_le_bytes()));
        assert_eq!(hle_is_stack(&ctx, &[0x10_0100, 0x120, 0x128]), 0);
        let mut out = [0u8; 16];
        assert!(mem.read(0x120, &mut out));
        assert_eq!(out, [0u8; 16], "non-stack outputs must be zeroed");
        // ...and 1 with the region bounds for a recorded stack.
        kernel.memory.record_mapping_of_kind(
            0x20_0000,
            0x8000,
            0x3,
            raeen_core::types::MappingKind::Stack,
            0,
            0,
        );
        assert_eq!(hle_is_stack(&ctx, &[0x20_4000, 0x120, 0x128]), 1);
        assert!(mem.read(0x120, &mut out));
        assert_eq!(u64::from_le_bytes(out[0..8].try_into().unwrap()), 0x20_0000);
        assert_eq!(
            u64::from_le_bytes(out[8..16].try_into().unwrap()),
            0x20_0000 + 0x8000
        );

        let registry = HleRegistry::new();
        for name in [
            "sceKernelIsStack",
            "sceKernelQueryMemoryProtection",
            "sceKernelCheckedReleaseDirectMemory",
            "sceKernelSetGPO",
        ] {
            assert!(
                registry.is_implemented("libkernel", name),
                "libkernel::{name} must be registered"
            );
        }
    }

    /// `sceKernelCheckedReleaseDirectMemory` validates where the unchecked
    /// form is best-effort: misaligned → EINVAL, untracked → ENOENT, and a
    /// real allocation releases (returning its budget) exactly once.
    #[test]
    fn checked_release_direct_memory_validates_then_releases() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x40000);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let page = raeen_core::PS5_PAGE_SIZE as u64;
        // Misalignment and zero-length contract.
        assert_eq!(
            hle_checked_release_direct_memory(&ctx, &[0x123, page]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(
            hle_checked_release_direct_memory(&ctx, &[0x40000, 0x123]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(hle_checked_release_direct_memory(&ctx, &[0, 0]), SCE_OK);
        // An untracked (or already-released) range is ENOENT...
        assert_eq!(
            hle_checked_release_direct_memory(&ctx, &[0x40000, page]),
            SCE_KERNEL_ERROR_ENOENT
        );

        // ...while a real direct allocation releases cleanly.
        assert_eq!(
            hle_allocate_direct_memory(&ctx, &[0, u64::MAX, page * 2, 0, 0, 0x600]),
            SCE_OK
        );
        let mut phys_bytes = [0u8; 8];
        assert!(mem.read(0x600, &mut phys_bytes));
        let phys = u64::from_le_bytes(phys_bytes);
        assert_eq!(phys % page, 0, "direct allocations are page-aligned");
        assert_eq!(
            hle_checked_release_direct_memory(&ctx, &[phys, page * 2]),
            SCE_OK
        );
        assert!(
            !kernel.memory.is_mapped(phys),
            "the release must drop the region record"
        );
        // A second checked release of the same range is now ENOENT (the
        // unchecked form stays a successful no-op by contract).
        assert_eq!(
            hle_checked_release_direct_memory(&ctx, &[phys, page * 2]),
            SCE_KERNEL_ERROR_ENOENT
        );
        assert_eq!(hle_release_direct_memory(&ctx, &[phys, page * 2]), SCE_OK);

        // SetGPO succeeds on retail (no devkit output lines to drive).
        assert_eq!(hle_set_gpo(&ctx, &[0xFF]), SCE_OK);
    }

    /// A failed `physAddrOut` write must roll back EVERYTHING the allocate
    /// charged: the arena mapping, the region record, and the direct-memory
    /// budget. Before the fix the budget refund was missing, so a title
    /// probing out-param validity leaked its measured footprint to ENOMEM.
    #[test]
    fn allocate_direct_memory_out_write_failure_refunds_budget_and_arena() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let page = raeen_core::PS5_PAGE_SIZE as u64;
        let before = kernel
            .direct_memory_allocated
            .load(std::sync::atomic::Ordering::Relaxed);
        // physAddrOut = 0x2000 is outside the 0x1000-byte test memory. The code
        // must be a real Orbis status (`EFAULT`), never the old `0xffffffff`
        // sentinel — a guest cannot classify that as failure.
        assert_eq!(
            hle_allocate_direct_memory(&ctx, &[0, u64::MAX, page, 0, 0, 0x2000]),
            SCE_KERNEL_ERROR_EFAULT
        );
        let after = kernel
            .direct_memory_allocated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(before, after, "failed out-write must refund the budget");
        // And the arena mapping must be gone: a retry with a VALID out param
        // must see the full budget and succeed.
        assert_eq!(
            hle_allocate_direct_memory(&ctx, &[0, u64::MAX, page, 0, 0, 0x600]),
            SCE_OK
        );
        assert_eq!(
            kernel
                .direct_memory_allocated
                .load(std::sync::atomic::Ordering::Relaxed),
            before + page,
            "only the successful allocate stays charged"
        );
    }

    /// Direct memory allocated 2 MiB-aligned, plus an `addrOut` in-value the
    /// test allocator cannot serve at that alignment. Shared by the two
    /// hint-failure tests below so both exercise the same refusal.
    fn direct_memory_with_unservable_hint(
        ctx: &HleContext,
        mem: &crate::TestMemory,
    ) -> (u64, u64, u64, u64) {
        let align = 0x20_0000u64; // 2 MiB
        let len = 0x8000u64;
        assert_eq!(
            hle_allocate_direct_memory(ctx, &[0, u64::MAX, len, align, 0, 0x600]),
            SCE_OK
        );
        let mut phys_bytes = [0u8; 8];
        assert!(mem.read(0x600, &mut phys_bytes));
        let phys = u64::from_le_bytes(phys_bytes);
        assert!(
            phys.is_multiple_of(align),
            "the allocation must satisfy the 2 MiB alignment it asked for"
        );
        // Page-aligned but NOT 2 MiB-aligned, so `map_at` declines it — the
        // deterministic stand-in for "a host allocation already owns this VA".
        let hint = raeen_core::PS5_PAGE_SIZE as u64;
        assert!(!hint.is_multiple_of(align));
        assert!(mem.write(0x700, &hint.to_le_bytes()));
        (0x700, len, align, phys)
    }

    /// The ASTRO.BOT regression, pinned at the ABI. Measured 2026-07-28
    /// (`artifacts/compat/raw/baseline-1785273714952/PPSA21564-*.stdout.log`):
    /// the title's `DirectMemoryAllocator` mapped 0xc8000000 bytes at the
    /// literal 0x1000000000, host thread stacks had landed inside that range,
    /// and this call answered `0xffffffff` — not an Orbis status at all. The
    /// guest logged `sceKernelMapDirectMemory error 0xffffffff`, asserted at
    /// `DirectMemoryAllocator.cpp:122`, and executed `int 0x41`; 128 presented
    /// frames became 0.
    ///
    /// Under `MAP_FIXED` the guest demanded that exact address, so refusing is
    /// correct — but it must refuse with `ENOMEM`, the code the real kernel
    /// uses for a fixed mapping it cannot place.
    #[test]
    fn map_direct_memory_refuses_an_unplaceable_fixed_address_with_enomem() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x20_0000);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let (addr_out, len, align, phys) = direct_memory_with_unservable_hint(&ctx, &mem);
        assert_eq!(
            hle_map_direct_memory(&ctx, &[addr_out, len, 0x3, MAP_FIXED as u64, phys, align]),
            SCE_KERNEL_ERROR_ENOMEM,
            "a fixed map that cannot be placed is ENOMEM, never 0xffffffff"
        );
    }

    /// Without `MAP_FIXED` the requested address is a HINT: Orbis places the
    /// mapping wherever it can and reports where through `addrOut` (shadPS4
    /// `MemoryManager::MapMemory` takes its `SearchFree` path whenever `Fixed`
    /// is clear). So an unservable hint must not sink the call — it falls back
    /// to the physical address, which is what the no-hint branch already
    /// publishes and keeps the mapping and the direct memory one storage.
    #[test]
    fn map_direct_memory_falls_back_to_the_physical_address_for_a_plain_hint() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x20_0000);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let (addr_out, len, align, phys) = direct_memory_with_unservable_hint(&ctx, &mem);
        assert_eq!(
            hle_map_direct_memory(&ctx, &[addr_out, len, 0x3, 0, phys, align]),
            SCE_OK,
            "a hint we cannot serve must not fail the call"
        );
        let mut published = [0u8; 8];
        assert!(mem.read(addr_out, &mut published));
        assert_eq!(
            u64::from_le_bytes(published),
            phys,
            "the guest must be told where the mapping actually landed"
        );
        assert!(
            kernel.memory.is_mapped(phys),
            "the fallback mapping must be recorded like any other"
        );
    }

    /// `sceKernelMapFlexibleMemory` and `sceKernelReserveVirtualRange` shared
    /// the same `0xffffffff` sentinel. Both must report Orbis statuses: no
    /// address space is `ENOMEM`, an unwritable out-parameter is `EFAULT`.
    #[test]
    fn flexible_map_and_virtual_reserve_report_orbis_statuses_on_failure() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        // A bump allocator started just below `u64::MAX` cannot satisfy any
        // real length, so `mmap`/`reserve` return `None` deterministically.
        let alloc = crate::TestAllocator::new(u64::MAX - 0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let page = raeen_core::PS5_PAGE_SIZE as u64;
        assert_eq!(
            hle_map_flexible_memory(&ctx, &[0x600, page, 0x3, 0]),
            SCE_KERNEL_ERROR_ENOMEM
        );
        assert!(mem.write(0x700, &0u64.to_le_bytes()));
        assert_eq!(
            hle_reserve_virtual_range(&ctx, &[0x700, page, 0, 0]),
            SCE_KERNEL_ERROR_ENOMEM
        );

        // With a working allocator the out-parameter guards take over, and an
        // out-of-bounds `addrOut` is EFAULT rather than the old sentinel.
        let alloc = crate::TestAllocator::new(page);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_map_flexible_memory(&ctx, &[0x2000, page, 0x3, 0]),
            SCE_KERNEL_ERROR_EFAULT
        );
    }

    #[test]
    fn getrusage_zero_fills_and_map_direct_memory2_reorders_arguments() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Mount a temp dir at /app0/ and drop a real file in it.
        let tmp = std::env::temp_dir().join(format!("raeen-hle-fileio-{}", std::process::id()));
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

    #[test]
    fn read_efault_does_not_consume_the_file_cursor() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let tmp =
            std::env::temp_dir().join(format!("raeen-hle-read-efault-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("data.bin"), b"FIRST_SECOND").unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        assert!(mem.write(0x20, b"/app0/data.bin\0"));
        let fd = hle_open(&ctx, &[0x20, 0, 0]);
        assert!((fd as i64) >= 3);

        assert_eq!(
            hle_read(&ctx, &[fd, 0x3ff, 5]) as i64,
            -14,
            "an invalid guest destination must return EFAULT"
        );
        assert_eq!(hle_read(&ctx, &[fd, 0x100, 5]), 5);
        let mut first = [0u8; 5];
        assert!(mem.read(0x100, &mut first));
        assert_eq!(
            &first, b"FIRST",
            "EFAULT must be detected before the host read advances the fd"
        );

        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn dev_urandom_opens_and_fills_guest_memory() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"/dev/urandom\0"));
        let fd = hle_open(&ctx, &[0x100, 0, 0]);
        assert!(
            (fd as i64) >= 3,
            "random device open returned {}",
            fd as i64
        );
        assert_eq!(hle_stat(&ctx, &[0x100, 0x300]), SCE_OK);
        assert_eq!(hle_fstat(&ctx, &[fd, 0x400]), SCE_OK);
        let mut path_mode = [0u8; 2];
        let mut fd_mode = [0u8; 2];
        assert!(mem.read(0x308, &mut path_mode));
        assert!(mem.read(0x408, &mut fd_mode));
        assert_eq!(u16::from_le_bytes(path_mode), ORBIS_MODE_CHARACTER);
        assert_eq!(u16::from_le_bytes(fd_mode), ORBIS_MODE_CHARACTER);

        assert!(mem.write(0x200, &[0xAA; 32]));
        assert_eq!(hle_read(&ctx, &[fd, 0x200, 32]), 32);
        let mut random = [0xAA; 32];
        assert!(mem.read(0x200, &mut random));
        assert_ne!(random, [0xAA; 32], "entropy read must overwrite the buffer");
        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
    }

    /// `pread` reads at an absolute offset WITHOUT moving the cursor — a
    /// streaming loader interleaves preads with sequential reads on one fd.
    /// Measured: ASTRO.BOT's asset streamer imports sceKernelPread.
    #[test]
    fn pread_reads_at_offset_without_moving_the_cursor() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp = std::env::temp_dir().join(format!("raeen-hle-pread-{}", std::process::id()));
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
    fn read_and_pread_stream_past_the_old_sixteen_mib_cap() {
        const OLD_CAP: usize = 16 << 20;
        let payload_len = OLD_CAP + 0x23456;
        let guest_buf = 0x1000u64;
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(payload_len + guest_buf as usize);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tmp = std::env::temp_dir().join(format!("raeen-hle-large-read-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let payload: Vec<u8> = (0..payload_len)
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        std::fs::write(tmp.join("archive-index.bin"), &payload).unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        assert!(mem.write(0x100, b"/app0/archive-index.bin\0"));
        let fd = hle_open(&ctx, &[0x100, 0, 0]);
        assert!((fd as i64) >= 3);

        assert_eq!(
            hle_pread(&ctx, &[fd, guest_buf, payload_len as u64, 0]),
            payload_len as u64,
            "pread must not report the old 16 MiB clamp as a successful short read"
        );
        let mut actual = vec![0u8; payload_len];
        assert!(mem.read(guest_buf, &mut actual));
        assert_eq!(actual, payload);

        actual.fill(0);
        assert_eq!(
            hle_read(&ctx, &[fd, guest_buf, payload_len as u64]),
            payload_len as u64,
            "sequential read must stream the full archive index too"
        );
        assert!(mem.read(guest_buf, &mut actual));
        assert_eq!(actual, payload);

        assert_eq!(
            hle_pread(
                &ctx,
                &[fd, guest_buf, MAX_HLE_BULK_BYTES.saturating_add(1), 0],
            ),
            FILE_EINVAL,
            "attacker-sized reads remain bounded instead of allocating"
        );

        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn directory_open_and_getdents_expose_gen5_records() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x2000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let tmp = std::env::temp_dir().join(format!("raeen-hle-dir-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("packs")).unwrap();
        std::fs::write(tmp.join("manifest.json"), b"{}").unwrap();
        kernel.filesystem.set_game_directory(&tmp);
        assert!(mem.write(0x100, b"/app0\0"));
        let fd = hle_open(&ctx, &[0x100, 0, 0]);
        assert!((fd as i64) >= 3);
        let registry = HleRegistry::new();
        let stale_arg4 = [0xA5u8; 8];
        assert!(mem.write(0x200, &stale_arg4));
        // `.`, `..`, `packs`, `manifest.json` pack into ONE 512-byte block.
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
        // Walk the packed records: every d_reclen stays inside the returned
        // payload (a d_reclen of 512 per record is exactly the overflow that
        // smashed Until Dawn's stack canary), and the four names appear.
        let mut block = [0u8; 512];
        assert!(mem.read(0x400, &mut block));
        let mut names = Vec::new();
        let mut offset = 0usize;
        while offset < block.len() {
            let reclen =
                u16::from_le_bytes(block[offset + 4..offset + 6].try_into().unwrap()) as usize;
            let namlen = block[offset + 7] as usize;
            assert!(matches!(block[offset + 6], 4 | 8));
            assert!(reclen > 8 + namlen, "d_reclen covers the record");
            assert!(
                offset + reclen <= block.len(),
                "d_reclen must stay inside the returned payload"
            );
            names.push(
                std::str::from_utf8(&block[offset + 8..offset + 8 + namlen])
                    .unwrap()
                    .to_string(),
            );
            offset += reclen;
        }
        names.sort();
        assert_eq!(names, [".", "..", "manifest.json", "packs"]);
        // The listing was fully consumed by the first call: EOF is 0 bytes.
        assert_eq!(hle_getdents(&ctx, &[fd, 0x400, 1024], false), 0);
        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A directory descriptor must fstat as a directory: `S_IFDIR` mode, the
    /// packed-dirent listing size, and shadPS4's directory block geometry —
    /// not as a zero-length regular file.
    #[test]
    fn fstat_reports_directory_mode_size_and_block_geometry() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let tmp = std::env::temp_dir().join(format!("raeen-hle-fstat-dir-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("save.dat"), b"s").unwrap();
        kernel.filesystem.set_game_directory(&tmp);

        assert!(mem.write(0x100, b"/app0\0"));
        let fd = hle_open(&ctx, &[0x100, 0, 0]);
        assert!((fd as i64) >= 3);

        assert_eq!(hle_fstat(&ctx, &[fd, 0x200]), SCE_OK);
        let mut stat = [0u8; ORBIS_STAT_SIZE];
        assert!(mem.read(0x200, &mut stat));
        assert_eq!(
            u16::from_le_bytes(stat[8..10].try_into().unwrap()),
            ORBIS_MODE_DIRECTORY,
            "a directory fd must report S_IFDIR, not a regular file"
        );
        let size = u64::from_le_bytes(stat[72..80].try_into().unwrap());
        assert!(
            size >= 512 && size.is_multiple_of(512),
            "st_size is the 512-aligned dirent listing, got {size}"
        );
        assert_eq!(
            u64::from_le_bytes(stat[80..88].try_into().unwrap()),
            8,
            "st_blocks"
        );
        assert_eq!(
            u32::from_le_bytes(stat[88..92].try_into().unwrap()),
            0x8000,
            "st_blksize"
        );
        assert_eq!(hle_close(&ctx, &[fd]), SCE_OK);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn stat_and_mkdir_use_mounted_vfs_roots() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let tmp = std::env::temp_dir().join(format!("raeen-hle-stat-{}", std::process::id()));
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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

    #[test]
    fn getargc_and_getargv_return_runtime_process_stack_state() {
        let kernel = raeen_kernel::OrbisKernel::new();
        kernel.set_process_args(3, 0x1234_5008);
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_getargc(&ctx, &[]), 3);
        assert_eq!(hle_getargv(&ctx, &[]), 0x1234_5008);
    }

    /// Process-time counters advance monotonically and agree on their domain.
    #[test]
    fn process_time_counters_advance_and_are_consistent() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!((hle_write(&ctx, &[1, 0xDEAD_0000, 8]) as i64) < 0);
        assert!(kernel.console.is_empty());
    }

    #[test]
    fn register_adds_expected_functions() {
        let registry = HleRegistry::new();
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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

    #[test]
    fn exception_handler_family_validates_and_preserves_process_state() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_install_exception_handler(&ctx, &[30, 0x1234]), SCE_OK);
        assert_eq!(kernel.exception_handlers.get(&30).map(|v| *v), Some(0x1234));
        assert_eq!(
            hle_install_exception_handler(&ctx, &[30, 0x5678]),
            0x8002_000B
        );
        assert_eq!(
            kernel.exception_handlers.get(&30).map(|v| *v),
            Some(0x1234),
            "a duplicate registration must not replace the installed handler"
        );
        assert_eq!(hle_raise_exception(&ctx, &[1, 30]), SCE_OK);
        assert_eq!(hle_remove_exception_handler(&ctx, &[30]), SCE_OK);
        assert!(!kernel.exception_handlers.contains_key(&30));
        assert_eq!(
            hle_install_exception_handler(&ctx, &[2, 0x1234]),
            SCE_KERNEL_ERROR_EINVAL
        );
    }

    // ---------------------------------------------------------------------
    // sceKernelSyncOnAddress* — the futex parking lot.
    //
    // Deterministic by construction: nothing here starts a host thread or
    // sleeps. `Wait` is only ever called on a value-mismatch (which returns
    // without parking), and the park side is exercised by driving the
    // address-keyed queue directly and asserting the per-waiter wake bit plus
    // `waiter_count` — the same shape as the `PthreadCond` queue tests.
    // ---------------------------------------------------------------------

    /// All four spellings must be resolvable, or a guest futex import lands on a
    /// null jump instead of the parking lot.
    #[test]
    fn sync_on_address_exports_are_all_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceKernelSyncOnAddressWait",
            "sceKernelSyncOnAddressWait32",
            "sceKernelSyncOnAddressWait64",
            "sceKernelSyncOnAddressWake",
        ] {
            assert!(
                registry.is_implemented("libkernel", name),
                "missing libkernel::{name}"
            );
        }
    }

    #[test]
    fn sync_on_address_rejects_a_null_address() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_sync_on_address_wait(&ctx, &[0, 0, 0]),
            SCE_KERNEL_ERROR_EINVAL
        );
        assert_eq!(
            hle_sync_on_address_wake(&ctx, &[0, 1]),
            SCE_KERNEL_ERROR_EINVAL
        );
    }

    /// The futex contract, and the bug the old stub had: if the watched word no
    /// longer holds the expected value the caller must be told to re-check
    /// (`EAGAIN`) rather than parked — and it must **not** be handed a bare 0,
    /// which a guest reads as "the value changed" and loops on forever.
    #[test]
    fn wait32_returns_eagain_without_parking_when_the_value_already_moved() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let addr = 0x200u64;
        assert!(mem.write(addr, &7u32.to_le_bytes()));

        // Expecting 0, memory holds 7.
        assert_eq!(
            hle_sync_on_address_wait32(&ctx, &[addr, 0, 0]),
            SCE_KERNEL_ERROR_EAGAIN
        );
        assert_eq!(
            kernel.sync_addresses.waiter_count(addr),
            0,
            "a mismatching wait must leave no waiter behind"
        );
    }

    /// The compare is width-correct: `Wait32` looks at 32 bits and ignores the
    /// high half of both the memory word and the expected value.
    #[test]
    fn wait32_and_wait64_compare_at_their_own_width() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let addr = 0x300u64;
        assert!(mem.write(addr, &0xdead_beef_0000_0001u64.to_le_bytes()));

        // Low 32 bits are 1: a 32-bit wait expecting 2 mismatches...
        assert_eq!(
            hle_sync_on_address_wait32(&ctx, &[addr, 2, 0]),
            SCE_KERNEL_ERROR_EAGAIN
        );
        // ...and the high half of `expected` is masked off, so 0xffff_ffff_0000_0002
        // is still a mismatch against 1 rather than an accidental match.
        assert_eq!(
            hle_sync_on_address_wait32(&ctx, &[addr, 0xffff_ffff_0000_0002, 0]),
            SCE_KERNEL_ERROR_EAGAIN
        );
        // A 64-bit wait sees the whole word, so expecting only the low half is a
        // mismatch where a 32-bit wait would have matched.
        assert_eq!(
            hle_sync_on_address_wait64(&ctx, &[addr, 1, 0]),
            SCE_KERNEL_ERROR_EAGAIN
        );
    }

    #[test]
    fn wait32_on_unmapped_memory_is_efault() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_sync_on_address_wait32(&ctx, &[0x9000_0000, 0, 0]),
            SCE_KERNEL_ERROR_EFAULT
        );
        assert_eq!(kernel.sync_addresses.waiter_count(0x9000_0000), 0);
    }

    /// `Wake` reaches the queue for its own address only, in FIFO order, and a
    /// count of 1 is wake-one rather than a broadcast. Driven through the HLE
    /// entry point so the argument decoding is covered too.
    #[test]
    fn wake_releases_fifo_waiters_on_that_address_only() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let watched = 0x400u64;
        let other = 0x408u64;

        let first = kernel.sync_addresses.enqueue(watched, 11);
        let second = kernel.sync_addresses.enqueue(watched, 22);
        let elsewhere = kernel.sync_addresses.enqueue(other, 33);

        assert_eq!(hle_sync_on_address_wake(&ctx, &[watched, 1]), SCE_OK);
        assert!(first.is_signaled());
        assert!(
            !second.is_signaled(),
            "count=1 is wake-one, not a broadcast"
        );
        assert!(!elsewhere.is_signaled(), "a wake is address-scoped");
        assert_eq!(kernel.sync_addresses.waiter_count(watched), 1);

        // A count of 0 (unset register) is the wake-all spelling.
        assert_eq!(hle_sync_on_address_wake(&ctx, &[watched, 0]), SCE_OK);
        assert!(second.is_signaled());
        assert_eq!(kernel.sync_addresses.waiter_count(watched), 0);
        assert_eq!(kernel.sync_addresses.waiter_count(other), 1);
    }

    /// Waking an address nobody waits on is success, not an error: a wake that
    /// beats its wait is the ordinary uncontended case, and the waiter's
    /// compare-on-entry is what catches it.
    #[test]
    fn wake_with_no_parked_waiter_succeeds() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_sync_on_address_wake(&ctx, &[0x500, 1]), SCE_OK);
    }

    /// The timeout argument decoder. 0 is the futex `NULL` spelling (no
    /// deadline); a plausible microsecond count becomes a real deadline; an
    /// implausible one (a guest pointer, say) is refused rather than gambled on,
    /// so a mis-decoded register can never manufacture an instant timeout.
    #[test]
    fn timeout_decoding_only_trusts_plausible_microsecond_counts() {
        assert!(decode_sync_timeout(0).is_none(), "0 == no deadline");
        assert!(decode_sync_timeout(1_000).is_some());
        assert!(decode_sync_timeout(SYNC_ADDRESS_MAX_TIMEOUT_US).is_some());
        assert!(
            decode_sync_timeout(SYNC_ADDRESS_MAX_TIMEOUT_US + 1).is_none(),
            "an absurd count is not a deadline"
        );
        assert!(
            decode_sync_timeout(0x7fff_0000_0000).is_none(),
            "a guest pointer must not be read as microseconds"
        );
    }
}
