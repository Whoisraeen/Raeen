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
const VIDEO_OUT_ERROR_INVALID_VALUE: u64 = 0x8029_0001;
const VIDEO_OUT_ERROR_INVALID_ADDRESS: u64 = 0x8029_0002;
const VIDEO_OUT_ERROR_RESOURCE_BUSY: u64 = 0x8029_0009;
const VIDEO_OUT_ERROR_INVALID_HANDLE: u64 = 0x8029_000B;
const VIDEO_OUT_ERROR_INVALID_OPTION: u64 = 0x8029_001A;
/// Default display width reported by `GetResolutionStatus` (1080p).
const DISPLAY_WIDTH: u32 = 1920;
/// Default display height.
const DISPLAY_HEIGHT: u32 = 1080;
/// Nominal refresh rate reported by `GetOutputStatus`.
const DISPLAY_REFRESH_HZ: u64 = 60;

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
        "sceVideoOutGetOutputStatus",
        hle_get_output_status,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutGetVblankStatus",
        hle_get_vblank_status,
    );
    registry.register("libSceVideoOut", "sceVideoOutAddFlipEvent", hle_ok);
    registry.register("libSceVideoOut", "sceVideoOutSetBufferAttribute", hle_ok);
    registry.register_nid(
        "libSceVideoOut",
        "sceVideoOutSetBufferAttribute2",
        0x3e34_b9b8_04b0_715f,
        hle_set_buffer_attribute2,
    );
    registry.register_nid(
        "libSceVideoOut",
        "sceVideoOutRegisterBuffers2",
        0xaca0_54b6_046b_b5b9,
        hle_register_buffers2,
    );
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

/// `sceVideoOutGetOutputStatus(handle, SceVideoOutOutputStatus *status)`:
/// reports an attached 1080p display at 60 Hz. The PS5 structure is 0x30
/// bytes; the first fields are the resolution class, connection state, and
/// refresh rate. Keeping the reserved tail zero makes this forward-compatible
/// with titles that inspect only fields defined by their SDK revision.
fn hle_get_output_status(ctx: &HleContext, args: &[u64]) -> u64 {
    let status_ptr = args.get(1).copied().unwrap_or(0);
    if status_ptr == 0 {
        return SCE_OK;
    }

    let mut buf = [0u8; 0x30];
    let resolution_class = if DISPLAY_WIDTH >= 3840 || DISPLAY_HEIGHT >= 2160 {
        2i32
    } else {
        1i32
    };
    buf[0x00..0x04].copy_from_slice(&resolution_class.to_le_bytes());
    buf[0x04..0x08].copy_from_slice(&1i32.to_le_bytes()); // display connected
    buf[0x08..0x10].copy_from_slice(&DISPLAY_REFRESH_HZ.to_le_bytes());
    if !ctx.mem.write(status_ptr, &buf) {
        debug!("sceVideoOutGetOutputStatus: status out-ptr {status_ptr:#x} not writable");
    }
    SCE_OK
}

fn read_buffer_attribute2(
    ctx: &HleContext,
    address: u64,
) -> Option<xps5x_kernel::VideoOutBufferAttribute> {
    let mut bytes = [0u8; 0x50];
    if address == 0 || !ctx.mem.read(address, &mut bytes) {
        return None;
    }
    Some(xps5x_kernel::VideoOutBufferAttribute {
        tiling_mode: u32::from_le_bytes(bytes[0x04..0x08].try_into().ok()?),
        width: u32::from_le_bytes(bytes[0x0C..0x10].try_into().ok()?),
        height: u32::from_le_bytes(bytes[0x10..0x14].try_into().ok()?),
        option: u64::from_le_bytes(bytes[0x18..0x20].try_into().ok()?),
        pixel_format: u64::from_le_bytes(bytes[0x20..0x28].try_into().ok()?),
        dcc_clear_color: u64::from_le_bytes(bytes[0x28..0x30].try_into().ok()?),
        dcc_control: u32::from_le_bytes(bytes[0x30..0x34].try_into().ok()?),
    })
}

/// Build a Gen5 `SceVideoOutBufferAttribute2` in guest memory.
fn hle_set_buffer_attribute2(ctx: &HleContext, args: &[u64]) -> u64 {
    let address = args.first().copied().unwrap_or(0);
    if address == 0 {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    let mut bytes = [0u8; 0x50];
    bytes[0x04..0x08].copy_from_slice(&(args.get(2).copied().unwrap_or(0) as u32).to_le_bytes());
    bytes[0x0C..0x10].copy_from_slice(&(args.get(3).copied().unwrap_or(0) as u32).to_le_bytes());
    bytes[0x10..0x14].copy_from_slice(&(args.get(4).copied().unwrap_or(0) as u32).to_le_bytes());
    bytes[0x18..0x20].copy_from_slice(&args.get(5).copied().unwrap_or(0).to_le_bytes());
    bytes[0x20..0x28].copy_from_slice(&args.get(1).copied().unwrap_or(0).to_le_bytes());
    bytes[0x28..0x30].copy_from_slice(&args.get(7).copied().unwrap_or(0).to_le_bytes());
    bytes[0x30..0x34].copy_from_slice(&(args.get(6).copied().unwrap_or(0) as u32).to_le_bytes());
    if !ctx.mem.write(address, &bytes) {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    SCE_OK
}

/// Capture Gen5 display-buffer addresses and their layout so submit/flip can
/// hand the selected guest image to the graphics backend.
fn hle_register_buffers2(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as i32;
    let set_index = args.get(1).copied().unwrap_or(0) as i32;
    let start_index = args.get(2).copied().unwrap_or(0) as i32;
    let buffers_address = args.get(3).copied().unwrap_or(0);
    let buffer_count = args.get(4).copied().unwrap_or(0) as i32;
    let attribute_address = args.get(5).copied().unwrap_or(0);
    let category = args.get(6).copied().unwrap_or(0);
    let option = args.get(7).copied().unwrap_or(0);

    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    if buffers_address == 0 {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    if attribute_address == 0 {
        return VIDEO_OUT_ERROR_INVALID_OPTION;
    }
    if start_index < 0
        || buffer_count < 1
        || buffer_count > 16
        || start_index.saturating_add(buffer_count) > 16
    {
        return VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if category != 0 || option != 0 {
        return VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    let Some(attribute) = read_buffer_attribute2(ctx, attribute_address) else {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    };

    let mut pending = Vec::with_capacity(buffer_count as usize);
    for i in 0..buffer_count {
        let entry = buffers_address + i as u64 * 0x20;
        let mut bytes = [0u8; 16];
        if !ctx.mem.read(entry, &mut bytes) {
            return VIDEO_OUT_ERROR_INVALID_ADDRESS;
        }
        let address = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let metadata = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        if address == 0 {
            return VIDEO_OUT_ERROR_INVALID_ADDRESS;
        }
        let slot = start_index + i;
        if ctx.kernel.video_out_buffers.contains_key(&(handle, slot)) {
            return VIDEO_OUT_ERROR_RESOURCE_BUSY;
        }
        pending.push((slot, address, metadata));
    }
    for (slot, address, metadata) in pending {
        ctx.kernel.video_out_buffers.insert(
            (handle, slot),
            xps5x_kernel::VideoOutBuffer {
                set_index,
                address,
                metadata,
                attribute,
            },
        );
    }
    debug!(
        "sceVideoOutRegisterBuffers2(handle={handle}, set={set_index}, start={start_index}, count={buffer_count}, {}x{}, format={:#x})",
        attribute.width, attribute.height, attribute.pixel_format
    );
    set_index as u64
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

    #[test]
    fn output_status_reports_a_connected_1080p_60_hz_display() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_output_status(&ctx, &[1, 0x100]), SCE_OK);
        let mut status = [0u8; 0x30];
        assert!(mem.read(0x100, &mut status));
        assert_eq!(i32::from_le_bytes(status[0..4].try_into().unwrap()), 1);
        assert_eq!(i32::from_le_bytes(status[4..8].try_into().unwrap()), 1);
        assert_eq!(
            u64::from_le_bytes(status[8..16].try_into().unwrap()),
            DISPLAY_REFRESH_HZ
        );
        assert!(status[16..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn gen5_buffer_registration_captures_presentable_guest_images() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_set_buffer_attribute2(
                &ctx,
                &[0x100, 0x8100_0000_2200_0000, 0, 1920, 1080, 0, 0, 0]
            ),
            SCE_OK
        );
        assert!(mem.write(0x200, &0x4000u64.to_le_bytes()));
        assert!(mem.write(0x208, &0u64.to_le_bytes()));
        assert!(mem.write(0x220, &0x8000u64.to_le_bytes()));
        assert!(mem.write(0x228, &0u64.to_le_bytes()));
        assert_eq!(
            hle_register_buffers2(&ctx, &[1, 3, 0, 0x200, 2, 0x100, 0, 0]),
            3
        );
        let first = kernel.video_out_buffers.get(&(1, 0)).unwrap();
        assert_eq!(first.set_index, 3);
        assert_eq!(first.address, 0x4000);
        assert_eq!(first.attribute.width, 1920);
        assert_eq!(first.attribute.height, 1080);
        assert_eq!(first.attribute.pixel_format, 0x8100_0000_2200_0000);
        drop(first);
        assert_eq!(
            kernel.video_out_buffers.get(&(1, 1)).unwrap().address,
            0x8000
        );

        let registry = HleRegistry::new();
        let overrides = registry.registered_nid_overrides();
        assert!(overrides.iter().any(|(nid, key)| {
            *nid == 0xaca0_54b6_046b_b5b9 && key == "libSceVideoOut::sceVideoOutRegisterBuffers2"
        }));
        assert!(overrides.iter().any(|(nid, key)| {
            *nid == 0x3e34_b9b8_04b0_715f && key == "libSceVideoOut::sceVideoOutSetBufferAttribute2"
        }));
    }
}
