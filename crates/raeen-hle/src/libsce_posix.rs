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
    registry.register("libScePosix", "fcntl", posix_fcntl);
    registry.register("libScePosix", "sleep", posix_sleep);
    registry.register("libkernel", "sleep", posix_sleep);

    // The `libkernel` module exports these POSIX names under BOTH its
    // `libScePosix` library and its `libkernel` library, and resolution is
    // provider-aware (`ModuleRegistry::resolve` keys on the importing symbol's
    // provider library, not the NID alone). The measured ASTRO.BOT title imports
    // `clock_gettime` (NID 0x94b313f6f240724d) naming provider library
    // `libkernel`, so the `libScePosix` registration above does not satisfy it —
    // register the same thin POSIX-ABI adapter under `libkernel` too. See the
    // `libkernel::register` POSIX-spelling block for the sibling time/memory
    // aliases done the same way.
    //
    // Measured per title: ASTRO.BOT stopped on `clock_gettime` and Minecraft on
    // `gettimeofday` (NID 0x9fcf2fc770b99d6f), both naming `libkernel`. `usleep`
    // is the same family and is aliased with them rather than waiting for a
    // third title to trip over it. (`getpid` and `nanosleep` already have
    // `libkernel` registrations in `libkernel::register`.)
    registry.register("libkernel", "clock_gettime", posix_clock_gettime);
    registry.register("libkernel", "gettimeofday", posix_gettimeofday);
    registry.register("libkernel", "usleep", posix_usleep);
}

/// Turn an SCE return (`0` on success, negative error code on failure) into the
/// POSIX convention (`0` on success, `-1` **with `errno` set** on failure).
///
/// `errno` used to be deliberately skipped because Raeen had no guest `errno`.
/// It does now — `libkernel::set_guest_errno` writes the per-thread
/// `__error()` slot — so a POSIX-named export leaving it stale is a defect, not
/// a conservative choice: a caller that tests `-1` and then reads `errno` sees
/// whatever a previous call left there. An `SCE_KERNEL_ERROR_*` code carries
/// the errno in its low 16 bits (`0x8002_xxxx`); a bare internal `-errno`
/// negative is used directly. `EINVAL` backs anything unrecognisable, since a
/// wrong-but-defined errno still beats a stale one.
///
/// Failure is judged on the **32-bit** sign as well as the 64-bit one, because
/// an `int`-returning export's error reaches the guest in EAX: `0x8002_000E` is
/// a positive `i64` but a negative `int`.
fn sce_to_posix(ctx: &HleContext, rc: u64) -> u64 {
    const EINVAL: i32 = 22;
    let signed = rc as i64;
    // An `int`-returning handler's failure reaches the guest in EAX, so the
    // sign that matters is the **32-bit** one: `SCE_KERNEL_ERROR_EFAULT`
    // (`0x8002_000E`) is a positive `i64` but a negative `int`. Testing only
    // `(rc as i64) < 0` — as this did — mapped every `0x8002_xxxx` failure to
    // POSIX *success*, so `gettimeofday` reported 0 while writing nothing.
    let narrow = rc as u32 as i32;
    if signed >= 0 && narrow >= 0 {
        return 0;
    }
    let errno = if rc & 0xffff_0000 == 0x8002_0000 {
        i32::try_from(rc & 0xffff).unwrap_or(EINVAL)
    } else if (-4095..0).contains(&signed) {
        (-signed) as i32
    } else {
        EINVAL
    };
    libkernel::set_guest_errno(ctx, if errno == 0 { EINVAL } else { errno });
    (-1i64) as u64
}

/// POSIX `gettimeofday(struct timeval *tp, struct timezone *tzp)`.
///
/// Same `timeval` layout as [`libkernel::hle_gettimeofday`] (two little-endian
/// `int64_t`s: `tv_sec`, then `tv_usec`), which does the real work. `tzp` is
/// ignored — it is obsolete in POSIX and every real caller passes NULL.
fn posix_gettimeofday(ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("gettimeofday(tp={:#x})", args.first().copied().unwrap_or(0));
    sce_to_posix(ctx, libkernel::hle_gettimeofday(ctx, args))
}

/// POSIX `clock_gettime(clockid_t clk_id, struct timespec *tp)`.
fn posix_clock_gettime(ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "clock_gettime(clk_id={}, tp={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    sce_to_posix(ctx, libkernel::hle_clock_gettime(ctx, args))
}

/// POSIX `usleep(useconds_t usec)` — really sleeps, bounded, via the libkernel
/// implementation.
fn posix_usleep(ctx: &HleContext, args: &[u64]) -> u64 {
    sce_to_posix(ctx, libkernel::hle_usleep(ctx, args))
}

/// POSIX `sleep(unsigned int seconds)`: really sleeps the host thread,
/// returning 0 once the full interval elapsed — or the number of UNSLEPT
/// seconds if the guest process began terminating mid-sleep (the teardown
/// case maps onto POSIX's "interrupted by a signal" shape: the caller is
/// going away, and a partial count is the honest report of what was slept).
///
/// The wait is sliced so teardown is noticed within 100 ms rather than after
/// the whole interval — a `sleep(3600)` during shutdown must not pin a dying
/// process open. Unlike `usleep` there is no duration cap: the guest asked
/// for seconds-scale blocking and a real console would honor it.
fn posix_sleep(ctx: &HleContext, args: &[u64]) -> u64 {
    let seconds = args.first().copied().unwrap_or(0);
    debug!("sleep(seconds={seconds})");
    const SLICE: std::time::Duration = std::time::Duration::from_millis(100);
    let mut remaining = std::time::Duration::from_secs(seconds);
    while !remaining.is_zero() {
        if ctx.guest_threads.process_is_terminating() {
            // POSIX reports the unslept whole seconds (rounded up).
            return u64::try_from(remaining.as_millis().div_ceil(1000)).unwrap_or(u64::MAX);
        }
        let slice = remaining.min(SLICE);
        ctx.services.sleep(slice);
        remaining = remaining.saturating_sub(slice);
    }
    0
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
                // Raeen does not replace a guest process image yet, so the
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

    // NOTE: the NID-level assertions for these names live in `raeen-firmware`
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

    /// The measured ASTRO.BOT title imports `clock_gettime` naming provider
    /// library `libkernel`, not `libScePosix`; resolution is provider-aware, so
    /// it must be registered under `libkernel` too (the NID-level provider check
    /// lives in `raeen-firmware`'s `hle_nid_coverage`).
    #[test]
    fn clock_gettime_is_also_registered_under_libkernel() {
        let registry = HleRegistry::new();
        assert!(
            registry.is_implemented("libkernel", "clock_gettime"),
            "the libkernel-library import the title actually issues must resolve"
        );
    }

    #[test]
    fn sce_to_posix_maps_error_to_minus_one_and_sets_errno() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        // A non-zero allocator base: `__error()` treats a zero address as
        // allocation failure, so a bump allocator starting at 0 has no slot.
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        let errno_slot = libkernel::hle_error_addr(&ctx, &[]);
        assert_ne!(errno_slot, 0, "the per-thread errno slot must exist");
        let read_errno = || {
            let mut buf = [0u8; 4];
            assert!(crate::GuestMemory::read(&mem, errno_slot, &mut buf));
            i32::from_le_bytes(buf)
        };

        assert_eq!(sce_to_posix(&ctx, 0), 0);
        // A positive SCE return is still success.
        assert_eq!(sce_to_posix(&ctx, 5), 0);

        // A bare internal `-errno`.
        assert_eq!(sce_to_posix(&ctx, (-9i64) as u64), (-1i64) as u64);
        assert_eq!(read_errno(), 9, "EBADF must reach the guest errno slot");

        // An `SCE_KERNEL_ERROR_*` code carries errno in its low 16 bits.
        assert_eq!(sce_to_posix(&ctx, 0x8002_0016), (-1i64) as u64);
        assert_eq!(read_errno(), 22, "EINVAL from 0x80020016");

        // Anything unrecognisable still leaves a DEFINED errno, never a stale
        // one — that is the whole point of setting it.
        assert_eq!(sce_to_posix(&ctx, 0x9999_9999), (-1i64) as u64);
        assert_eq!(read_errno(), 22);
    }

    #[test]
    fn fcntl_gets_and_sets_status_flags_and_has_retail_nid() {
        use crate::{TestAllocator, TestMemory, test_ctx};
        use raeen_kernel::filesystem::open_flags::*;

        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = TestMemory::new(0x1000);
        let alloc = TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let tmp = std::env::temp_dir().join(format!("raeen-fcntl-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("file"), b"x").unwrap();
        kernel.filesystem.set_game_directory(&tmp);
        let fd = kernel.filesystem.open("/app0/file", O_RDONLY, 0).unwrap();
        assert_eq!(posix_fcntl(&ctx, &[fd as u64, 3]), O_RDONLY as u64);
        assert_eq!(posix_fcntl(&ctx, &[fd as u64, 4, O_APPEND as u64]), 0);
        assert_eq!(posix_fcntl(&ctx, &[fd as u64, 3]), O_APPEND as u64);
        assert_eq!(posix_fcntl(&ctx, &[0x7fff, 3]), (-1i64) as u64);

        let socket = kernel.create_socket().expect("socket quota available");
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 3]), 0);
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 4, 0x800]), 0);
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 3]), 0x800);
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 2, 1]), 0);
        assert_eq!(posix_fcntl(&ctx, &[socket as u32 as u64, 1]), 1);

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libScePosix", "fcntl"));
        kernel.filesystem.close(fd).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn sleep_really_sleeps_and_is_registered_under_both_providers() {
        use crate::{TestAllocator, TestMemory, test_ctx};

        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = TestMemory::new(0x100);
        let alloc = TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // sleep(0) completes without waiting.
        let started = std::time::Instant::now();
        assert_eq!(posix_sleep(&ctx, &[0]), 0);
        assert!(started.elapsed() < std::time::Duration::from_millis(500));

        // A completed interval returns 0 after really sleeping it through
        // (the kernel's sleep service is a real host sleep).
        let started = std::time::Instant::now();
        assert_eq!(posix_sleep(&ctx, &[1]), 0);
        assert!(started.elapsed() >= std::time::Duration::from_millis(900));

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libScePosix", "sleep"));
        assert!(registry.is_implemented("libkernel", "sleep"));
    }
}
