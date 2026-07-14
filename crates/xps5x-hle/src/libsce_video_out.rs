//! HLE libSceVideoOut — display output / flip (present) management.
//!
//! A title's render loop calls `sceVideoOutSubmitFlip` to present a buffer,
//! then waits for that flip to *complete* — polling `sceVideoOutGetFlipStatus`
//! (or an event) until the flip count advances and no flip is pending — before
//! reusing the buffer for the next frame. XPS5X doesn't present to a real
//! swapchain yet, but it must report flips as **completing** or the render
//! loop stalls. So `SubmitFlip` bumps a global flip counter and records the
//! `flipArg`, and `GetFlipStatus` reports that count with zero pending — the
//! loop advances every frame. `GetResolutionStatus` reports a 1080p display
//! so the title sizes its framebuffers. Real swapchain present is the M2/M3
//! follow-up (behind `xps5x-gpu`).

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use tracing::debug;

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// Default display width reported by `GetResolutionStatus` (1080p).
const DISPLAY_WIDTH: u32 = 1920;
/// Default display height.
const DISPLAY_HEIGHT: u32 = 1080;

/// Total flips submitted (== completed, since XPS5X never leaves one
/// pending). `GetFlipStatus` reports this so a waiting render loop advances.
static FLIP_COUNT: AtomicU64 = AtomicU64::new(0);
/// The `flipArg` from the most recent `SubmitFlip` (a title correlates the
/// completion it's waiting for by this value).
static LAST_FLIP_ARG: AtomicI64 = AtomicI64::new(0);

/// Register libSceVideoOut HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceVideoOut", "sceVideoOutOpen", hle_open);
    registry.register("libSceVideoOut", "sceVideoOutClose", hle_close);
    registry.register("libSceVideoOut", "sceVideoOutSetFlipRate", hle_ok);
    registry.register("libSceVideoOut", "sceVideoOutRegisterBuffers", hle_ok);
    registry.register("libSceVideoOut", "sceVideoOutSubmitFlip", hle_submit_flip);
    registry.register(
        "libSceVideoOut",
        "sceVideoOutGetFlipStatus",
        hle_get_flip_status,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutGetResolutionStatus",
        hle_get_resolution_status,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutGetVblankStatus",
        hle_get_vblank_status,
    );
    registry.register("libSceVideoOut", "sceVideoOutAddFlipEvent", hle_ok);
    registry.register("libSceVideoOut", "sceVideoOutSetBufferAttribute", hle_ok);
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

fn hle_open(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceVideoOutOpen(userId={}, busType={}, index={})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    1 // video-out handle
}

fn hle_close(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceVideoOutClose(handle={})",
        args.first().copied().unwrap_or(0)
    );
    SCE_OK
}

/// `sceVideoOutSubmitFlip(handle, bufferIndex, flipMode, flipArg)`: records
/// the flip as completed (bumps [`FLIP_COUNT`], stores `flipArg`) so a
/// subsequent `GetFlipStatus` shows the render loop it can proceed.
fn hle_submit_flip(_ctx: &HleContext, args: &[u64]) -> u64 {
    let buffer_index = args.get(1).copied().unwrap_or(0);
    let flip_arg = args.get(3).copied().unwrap_or(0) as i64;
    debug!("sceVideoOutSubmitFlip(bufferIndex={buffer_index}, flipArg={flip_arg})");
    LAST_FLIP_ARG.store(flip_arg, Ordering::Relaxed);
    FLIP_COUNT.fetch_add(1, Ordering::Relaxed);
    SCE_OK
}

/// `sceVideoOutGetFlipStatus(handle, SceVideoOutFlipStatus *status)`: reports
/// the completed flip count, the last `flipArg`, and **zero pending** flips —
/// so a title waiting on flip completion always sees it done and advances.
fn hle_get_flip_status(ctx: &HleContext, args: &[u64]) -> u64 {
    let status_ptr = args.get(1).copied().unwrap_or(0);
    if status_ptr == 0 {
        return SCE_OK;
    }
    // SceVideoOutFlipStatus (64 bytes): count@0, processTime@8, tsc@16,
    // flipArg@24, submitTsc@32, reserved@40, gcQueueNum@48,
    // flipPendingNum@52, currentBuffer@56, reserved@60.
    let mut buf = [0u8; 64];
    let count = FLIP_COUNT.load(Ordering::Relaxed);
    buf[0..8].copy_from_slice(&count.to_le_bytes());
    buf[24..32].copy_from_slice(&LAST_FLIP_ARG.load(Ordering::Relaxed).to_le_bytes());
    // flipPendingNum@52 stays 0 (nothing pending) and currentBuffer@56 stays 0.
    if !ctx.mem.write(status_ptr, &buf) {
        debug!("sceVideoOutGetFlipStatus: status out-ptr {status_ptr:#x} not writable");
    }
    SCE_OK
}

/// `sceVideoOutGetResolutionStatus(handle, SceVideoOutResolutionStatus
/// *status)`: reports a 1920x1080 display so the title sizes its buffers.
fn hle_get_resolution_status(ctx: &HleContext, args: &[u64]) -> u64 {
    let status_ptr = args.get(1).copied().unwrap_or(0);
    if status_ptr == 0 {
        return SCE_OK;
    }
    // width@0, height@4, paneWidth@8, paneHeight@12 (all u32).
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&DISPLAY_WIDTH.to_le_bytes());
    buf[4..8].copy_from_slice(&DISPLAY_HEIGHT.to_le_bytes());
    buf[8..12].copy_from_slice(&DISPLAY_WIDTH.to_le_bytes());
    buf[12..16].copy_from_slice(&DISPLAY_HEIGHT.to_le_bytes());
    if !ctx.mem.write(status_ptr, &buf) {
        debug!("sceVideoOutGetResolutionStatus: status out-ptr {status_ptr:#x} not writable");
    }
    SCE_OK
}

/// `sceVideoOutGetVblankStatus(handle, SceVideoOutVblankStatus *status)`:
/// reports the flip count as the vblank count (a monotonically-advancing
/// frame counter is enough for a title's frame-timing loop).
fn hle_get_vblank_status(ctx: &HleContext, args: &[u64]) -> u64 {
    let status_ptr = args.get(1).copied().unwrap_or(0);
    if status_ptr == 0 {
        return SCE_OK;
    }
    let mut buf = [0u8; 32];
    buf[0..8].copy_from_slice(&FLIP_COUNT.load(Ordering::Relaxed).to_le_bytes()); // count@0
    if !ctx.mem.write(status_ptr, &buf) {
        debug!("sceVideoOutGetVblankStatus: status out-ptr {status_ptr:#x} not writable");
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn submit_flip_advances_the_reported_flip_count() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Read the current count, submit a flip with a distinctive flipArg,
        // then confirm GetFlipStatus reports an advanced count + that arg +
        // zero pending (so a render loop would proceed).
        let before = {
            hle_get_flip_status(&ctx, &[1, 0x100]);
            let mut b = [0u8; 8];
            assert!(mem.read(0x100, &mut b));
            u64::from_le_bytes(b)
        };

        assert_eq!(hle_submit_flip(&ctx, &[1, 0, 1, 0xABCD]), SCE_OK);

        hle_get_flip_status(&ctx, &[1, 0x200]);
        let mut st = [0u8; 64];
        assert!(mem.read(0x200, &mut st));
        let count = u64::from_le_bytes(st[0..8].try_into().unwrap());
        let flip_arg = i64::from_le_bytes(st[24..32].try_into().unwrap());
        let pending = i32::from_le_bytes(st[52..56].try_into().unwrap());
        assert!(count > before, "flip count must advance after SubmitFlip");
        assert_eq!(flip_arg, 0xABCD, "the submitted flipArg is reported back");
        assert_eq!(pending, 0, "no flip pending → render loop proceeds");
    }

    #[test]
    fn resolution_status_reports_1080p() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_resolution_status(&ctx, &[1, 0x100]), SCE_OK);
        let mut r = [0u8; 8];
        assert!(mem.read(0x100, &mut r));
        assert_eq!(u32::from_le_bytes(r[0..4].try_into().unwrap()), 1920);
        assert_eq!(u32::from_le_bytes(r[4..8].try_into().unwrap()), 1080);
    }
}
