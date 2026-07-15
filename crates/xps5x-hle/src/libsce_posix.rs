//! `libScePosix` — the POSIX-named view of the Orbis kernel.
//!
//! # Why this library exists
//!
//! A PS5 module's `libkernel` **module** exports several **libraries**, and
//! `libScePosix` is the one holding the plain POSIX spellings: `gettimeofday`,
//! `clock_gettime`, `usleep`, ... The Sony-prefixed names (`sceKernelGettimeofday`)
//! and the POSIX ones are *different symbols* with *different NIDs* — a NID is
//! a hash of the function name alone — so implementing only the `sce*` spelling
//! leaves the POSIX one unresolved even though the behaviour is already written.
//!
//! That is not hypothetical: the measured retail title's own `libc.prx` calls
//! `gettimeofday` during early init and died on it, while
//! `sceKernelGettimeofday` had been implemented and working for weeks.
//!
//! So this module is deliberately thin: it maps POSIX names onto the
//! implementations that already exist rather than duplicating them. Where a
//! POSIX name is *identical* to one `libc` already registers (`malloc`,
//! `memcpy`, `strlen`, ...), nothing is needed here at all — the NID is the
//! same, and `ModuleRegistry::resolve` matches on NID, not on the declaring
//! library.
//!
//! # Semantics
//!
//! POSIX functions report failure as `-1` with `errno` set, whereas the `sce*`
//! entry points return a negative SCE error code directly. Where an aliased
//! implementation's return convention differs, the wrapper adapts it — see
//! [`posix_gettimeofday`]. Anything whose convention cannot be adapted honestly
//! is left unregistered rather than wired up wrongly: an unresolved import
//! fails loudly and names itself, while a wrong return value corrupts silently.

use tracing::debug;

use crate::libkernel;
use crate::{HleContext, HleRegistry};

/// Register the POSIX-named entry points.
pub fn register(registry: &HleRegistry) {
    registry.register("libScePosix", "gettimeofday", posix_gettimeofday);
    registry.register("libScePosix", "clock_gettime", posix_clock_gettime);
    registry.register("libScePosix", "usleep", posix_usleep);
    registry.register("libScePosix", "getpid", libkernel::hle_getpid);
    registry.register_nid("libScePosix", "fcntl", 0xf276_35f5_b2a8_8999, posix_fcntl);
}

/// Turn an SCE return (`0` on success, negative error code on failure) into the
/// POSIX convention (`0` on success, `-1` on failure).
///
/// `errno` is deliberately **not** set: XPS5X has no guest `errno` yet, and
/// inventing one that never updates would be worse than leaving it — a caller
/// that checks `errno` after a `-1` would read stale memory and misbehave in a
/// way far harder to trace than a missing symbol. Callers that only test the
/// return value (the overwhelming majority, and every caller seen so far in the
/// measured title) get correct behaviour.
fn sce_to_posix(rc: u64) -> u64 {
    if (rc as i64) < 0 { (-1i64) as u64 } else { 0 }
}

/// POSIX `gettimeofday(struct timeval *tp, struct timezone *tzp)`.
///
/// Same `timeval` layout as [`libkernel::hle_gettimeofday`] (two little-endian
/// `int64_t`s: `tv_sec`, then `tv_usec`), which does the real work. `tzp` is
/// ignored — it is obsolete in POSIX and every real caller passes NULL.
fn posix_gettimeofday(ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("gettimeofday(tp={:#x})", args.first().copied().unwrap_or(0));
    sce_to_posix(libkernel::hle_gettimeofday(ctx, args))
}

/// POSIX `clock_gettime(clockid_t clk_id, struct timespec *tp)`.
fn posix_clock_gettime(ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "clock_gettime(clk_id={}, tp={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    sce_to_posix(libkernel::hle_clock_gettime(ctx, args))
}

/// POSIX `usleep(useconds_t usec)` — really sleeps, bounded, via the libkernel
/// implementation.
fn posix_usleep(ctx: &HleContext, args: &[u64]) -> u64 {
    sce_to_posix(libkernel::hle_usleep(ctx, args))
}

/// POSIX `fcntl(fd, command, argument)` for descriptor/status flag commands.
/// These are the commands used by libc and ordinary C++ file streams; handle
/// duplication/locking remain explicit failures until their shared-open-file
/// semantics are modeled.
fn posix_fcntl(ctx: &HleContext, args: &[u64]) -> u64 {
    const F_GETFD: i32 = 1;
    const F_SETFD: i32 = 2;
    const F_GETFL: i32 = 3;
    const F_SETFL: i32 = 4;

    let fd = args.first().copied().unwrap_or(0) as i32;
    let command = args.get(1).copied().unwrap_or(0) as i32;
    let argument = args.get(2).copied().unwrap_or(0) as i32;
    match command {
        F_GETFD => {
            if ctx.kernel.filesystem.flags(fd).is_some() {
                0
            } else {
                ctx.kernel
                    .kernel_sockets
                    .get(&fd)
                    .map_or((-1i64) as u64, |socket| socket.descriptor_flags as u64)
            }
        }
        F_SETFD => {
            if ctx.kernel.filesystem.flags(fd).is_some() {
                // XPS5X does not replace a guest process image yet, so the
                // close-on-exec bit has no effect for VFS descriptors.
                0
            } else if let Some(mut socket) = ctx.kernel.kernel_sockets.get_mut(&fd) {
                socket.descriptor_flags = argument;
                0
            } else {
                (-1i64) as u64
            }
        }
        F_GETFL => ctx.kernel.filesystem.flags(fd).map_or_else(
            || {
                ctx.kernel
                    .kernel_sockets
                    .get(&fd)
                    .map_or((-1i64) as u64, |socket| socket.status_flags as u64)
            },
            |flags| flags as u64,
        ),
        F_SETFL => {
            if ctx.kernel.filesystem.set_status_flags(fd, argument).is_ok() {
                0
            } else if let Some(mut socket) = ctx.kernel.kernel_sockets.get_mut(&fd) {
                socket.status_flags = argument;
                0
            } else {
                (-1i64) as u64
            }
        }
        _ => (-1i64) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: the NID-level assertions for these names live in `xps5x-firmware`
    // (`libsce_posix_names_resolve_the_nids_the_real_title_asked_for`) — NID
    // hashing lives there, and firmware depends on this crate, not the reverse.

    #[test]
    fn registered_under_libsce_posix() {
        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libScePosix", "gettimeofday"));
        assert!(registry.is_implemented("libScePosix", "clock_gettime"));
        assert!(registry.is_implemented("libScePosix", "usleep"));
        assert!(registry.is_implemented("libScePosix", "getpid"));
    }

    #[test]
    fn sce_to_posix_maps_error_to_minus_one_and_success_to_zero() {
        assert_eq!(sce_to_posix(0), 0);
        assert_eq!(sce_to_posix((-9i64) as u64), (-1i64) as u64);
        // A positive SCE return is still success.
        assert_eq!(sce_to_posix(5), 0);
    }

    #[test]
    fn fcntl_gets_and_sets_status_flags_and_has_retail_nid() {
        use crate::{TestAllocator, TestMemory, test_ctx};
        use xps5x_kernel::filesystem::open_flags::*;

        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = TestMemory::new(0x1000);
        let alloc = TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let tmp = std::env::temp_dir().join(format!("xps5x-fcntl-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("file"), b"x").unwrap();
        kernel.filesystem.set_game_directory(&tmp);
        let fd = kernel.filesystem.open("/app0/file", O_RDONLY, 0).unwrap();
        assert_eq!(posix_fcntl(&ctx, &[fd as u64, 3]), O_RDONLY as u64);
        assert_eq!(posix_fcntl(&ctx, &[fd as u64, 4, O_APPEND as u64]), 0);
        assert_eq!(posix_fcntl(&ctx, &[fd as u64, 3]), O_APPEND as u64);
        assert_eq!(posix_fcntl(&ctx, &[0x7fff, 3]), (-1i64) as u64);

        let socket = kernel.create_socket();
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 3]), 0);
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 4, 0x800]), 0);
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 3]), 0x800);
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 2, 1]), 0);
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 1]), 1);

        let registry = HleRegistry::new();
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| { *nid == 0xf276_35f5_b2a8_8999 && key == "libScePosix::fcntl" })
        );
        kernel.filesystem.close(fd).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
