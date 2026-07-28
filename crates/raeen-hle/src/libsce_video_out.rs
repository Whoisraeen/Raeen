//! HLE libSceVideoOut — display output / flip (present) management.
//!
//! A title's render loop calls `sceVideoOutSubmitFlip` to present a buffer,
//! then waits for that flip to *complete* — polling `sceVideoOutGetFlipStatus`
//! (or an event) until the flip count advances and no flip is pending — before
//! reusing the buffer for the next frame. Raeen doesn't present to a real
//! swapchain yet, but it must report flips as **completing** or the render
//! loop stalls. So `SubmitFlip` bumps a global flip counter and records the
//! `flipArg`, and `GetFlipStatus` reports that count with zero pending — the
//! loop advances every frame. `GetResolutionStatus` reports a 1080p display
//! so the title sizes its framebuffers. Real swapchain present is the M2/M3
//! follow-up (behind `raeen-gpu`).

use crate::{HleContext, HleRegistry};
use std::sync::atomic::Ordering;
use tracing::debug;

/// `SCE_OK`.
const SCE_OK: u64 = 0;
const VIDEO_OUT_ERROR_INVALID_VALUE: u64 = 0x8029_0001;
const VIDEO_OUT_ERROR_INVALID_ADDRESS: u64 = 0x8029_0002;
const VIDEO_OUT_ERROR_RESOURCE_BUSY: u64 = 0x8029_0009;
const VIDEO_OUT_ERROR_INVALID_HANDLE: u64 = 0x8029_000B;
const VIDEO_OUT_ERROR_INVALID_EVENT: u64 = 0x8029_000D;
const VIDEO_OUT_ERROR_UNSUPPORTED_OUTPUT_MODE: u64 = 0x8029_0016;
const VIDEO_OUT_ERROR_INVALID_OPTION: u64 = 0x8029_001A;
/// SCE "memory fault" (`0x8002_0000 | EFAULT`) — SharpEmu returns this generic
/// kernel error (not a VideoOut one) when an event/options block is unreadable.
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;
/// Kernel-event `ident` of a VideoOut **flip** event (SharpEmu
/// `SceVideoOutInternalEventFlip`).
const VIDEO_OUT_EVENT_FLIP_ID: u64 = 0x6;
/// Kernel-event `ident` of a VideoOut **vblank** event (SharpEmu
/// `SceVideoOutInternalEventVblank`).
const VIDEO_OUT_EVENT_VBLANK_ID: u64 = 0x40;
/// `SceKernelEvent.filter` for VideoOut events.
const KERNEL_EVENT_FILTER_VIDEO_OUT: i16 = -13;
/// Size of the `SceVideoOutOutputOptions` block (SharpEmu
/// `VideoOutOutputOptionsSize`).
const OUTPUT_OPTIONS_SIZE: usize = 0x40;
/// `SceVideoOutOutputMode` values accepted by `IsOutputSupported` (SharpEmu).
const OUTPUT_MODE_DEFAULT: u64 = 1;
const OUTPUT_MODE_119_88_HZ: u64 = 0xF;
/// Default display width reported by `GetResolutionStatus` (1080p).
const DISPLAY_WIDTH: u32 = 1920;
/// Default display height.
const DISPLAY_HEIGHT: u32 = 1080;
/// Nominal refresh rate reported by `GetOutputStatus`.
const DISPLAY_REFRESH_HZ: u64 = 60;

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
    registry.register("libSceVideoOut", "sceVideoOutWaitVblank", hle_wait_vblank);
    registry.register(
        "libSceVideoOut",
        "sceVideoOutAddFlipEvent",
        hle_add_flip_event,
    );
    registry.register("libSceVideoOut", "sceVideoOutSetBufferAttribute", hle_ok);
    registry.register(
        "libSceVideoOut",
        "sceVideoOutSetBufferAttribute2",
        hle_set_buffer_attribute2,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutRegisterBuffers2",
        hle_register_buffers2,
    );
    // Color pipeline + flip-state queries (measured ASTRO.BOT imports).
    registry.register(
        "libSceVideoOut",
        "sceVideoOutColorSettingsSetGamma_",
        hle_color_settings_set_gamma,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutAdjustColor_",
        hle_adjust_color,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutSubmitChangeBufferAttribute2",
        hle_submit_change_buffer_attribute2,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutIsFlipPending",
        hle_is_flip_pending,
    );
    // UE5 pair (Until Dawn PPSA15421 + Dragon Ball Sparking Zero PPSA15210 —
    // identical libSceVideoOut gap set) + A Plague Tale Requiem trio. Every
    // name hashes to the NID the titles import (verified with --imports).
    registry.register(
        "libSceVideoOut",
        "sceVideoOutIsOutputSupported",
        hle_is_output_supported,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutConfigureOutput",
        hle_configure_output,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutInitializeOutputOptions",
        hle_initialize_output_options,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutSetWindowModeMargins",
        hle_set_window_mode_margins,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutUnregisterBuffers",
        hle_unregister_buffers,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutAddVblankEvent",
        hle_add_vblank_event,
    );
    registry.register("libSceVideoOut", "sceVideoOutGetEventId", hle_get_event_id);
    registry.register(
        "libSceVideoOut",
        "sceVideoOutVrrPegToFixedRate",
        hle_vrr_fixed_rate,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutVrrUnpegFromFixedRate",
        hle_vrr_fixed_rate,
    );
    // Host input sampling is already immediate; there is no display-latency
    // controller to wait on, so this synchronization hint completes now.
    registry.register_incomplete(
        "libSceVideoOut",
        "sceVideoOutLatencyControlWaitBeforeInput",
        hle_ok,
        "latency controller is not modeled; synchronization hint completes immediately",
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

/// `sceVideoOutAddFlipEvent(equeue, handle, udata)`: register a VideoOut
/// event that is edge-triggered whenever a direct or AGC-embedded flip
/// completes.
fn hle_add_flip_event(ctx: &HleContext, args: &[u64]) -> u64 {
    add_video_out_event(ctx, args, VIDEO_OUT_EVENT_FLIP_ID, "flip")
}

/// `sceVideoOutAddVblankEvent(equeue, handle, udata)`: register a VideoOut
/// event triggered on display vblank. Ported from SharpEmu
/// `VideoOutAddVblankEvent` (VideoOutExports.cs, NID `Xru92wHJRmg`): same
/// (equeue, handle, udata) ABI as AddFlipEvent, re-registration replaces the
/// existing registration for that queue (here: same `(equeue, ident)` key).
/// SharpEmu starts a 60 Hz vblank thread; Raeen instead ticks vblank events
/// from `sceVideoOutWaitVblank` and from every completed flip (a flip implies
/// a display refresh), which keeps event-driven frame loops advancing without
/// a host timer thread.
fn hle_add_vblank_event(ctx: &HleContext, args: &[u64]) -> u64 {
    add_video_out_event(ctx, args, VIDEO_OUT_EVENT_VBLANK_ID, "vblank")
}

fn add_video_out_event(ctx: &HleContext, args: &[u64], ident: u64, kind: &str) -> u64 {
    let equeue = args.first().copied().unwrap_or(0);
    let handle = args.get(1).copied().unwrap_or(0) as i32;
    let udata = args.get(2).copied().unwrap_or(0);
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    if !ctx.kernel.kernel_equeues.contains_key(&equeue) {
        return VIDEO_OUT_ERROR_INVALID_OPTION;
    }
    ctx.kernel.kernel_equeue_events.insert(
        (equeue, ident),
        raeen_kernel::EqueueUserEvent {
            udata,
            filter: KERNEL_EVENT_FILTER_VIDEO_OUT,
            ..Default::default()
        },
    );
    debug!(equeue, handle, udata, "registered VideoOut {kind} event");
    SCE_OK
}

/// Trigger every registered VideoOut vblank event. `data` carries the vblank
/// sequence in the upper bits over the ident, mirroring the flip-event
/// encoding this file already uses (SharpEmu `GetEventData` decodes
/// `data >> 16`).
fn trigger_vblank_events(ctx: &HleContext, count: u64) {
    let event_hint = VIDEO_OUT_EVENT_VBLANK_ID | ((count & 0x0000_ffff_ffff_ffff) << 16);
    let mut queues = Vec::new();
    for mut event in ctx.kernel.kernel_equeue_events.iter_mut() {
        if event.key().1 == VIDEO_OUT_EVENT_VBLANK_ID
            && event.filter == KERNEL_EVENT_FILTER_VIDEO_OUT
        {
            let eq = event.key().0;
            event.triggered = true;
            event.fflags = event.fflags.saturating_add(1);
            event.data = event_hint as i64;
            if !queues.contains(&eq) {
                queues.push(eq);
            }
        }
    }
    for eq in queues {
        crate::kernel_equeue::wake_equeue(ctx, eq, raeen_core::subsystems::WakeReason::Signal);
    }
}

/// `sceVideoOutSubmitFlip(handle, bufferIndex, flipMode, flipArg)`: records
/// the flip as completed (bumps process-local state and stores `flipArg`) so a
/// subsequent `GetFlipStatus` shows the render loop it can proceed.
fn hle_submit_flip(ctx: &HleContext, args: &[u64]) -> u64 {
    let buffer_index = args.get(1).copied().unwrap_or(0);
    let flip_arg = args.get(3).copied().unwrap_or(0) as i64;
    debug!("sceVideoOutSubmitFlip(bufferIndex={buffer_index}, flipArg={flip_arg})");
    ctx.kernel
        .video_out_last_flip_arg
        .store(flip_arg, Ordering::Relaxed);
    ctx.kernel
        .video_out_current_buffer
        .store(buffer_index as i32, Ordering::Relaxed);
    ctx.kernel
        .video_out_flip_count
        .fetch_add(1, Ordering::Relaxed);
    // Route the flipped buffer to the GPU present path. A title composites its
    // UI across several render targets and flips to ONE of them; the GPU
    // otherwise presents the last-drawn target (often a black background). Look
    // up the guest address the title registered for this buffer slot and hand
    // it to the session — it presents that render target when it has content
    // and falls back to the last-drawn baseline otherwise.
    let handle = args.first().copied().unwrap_or(1) as i32;
    if let Some(buffer) = ctx
        .kernel
        .video_out_buffers
        .get(&(handle, buffer_index as i32))
    {
        // Thread the buffer's layout to the GPU so a CPU-drawn 2D buffer (no
        // GPU render target) can be presented straight from guest memory (M3).
        // The Gen5 attribute carries no separate pitch, so a tightly-packed
        // linear row (pitch == width) is assumed.
        let attr = buffer.attribute;
        let descriptor = raeen_core::subsystems::ScanoutDescriptor {
            width: attr.width,
            height: attr.height,
            pitch_pixels: attr.width,
            pixel_format: attr.pixel_format,
            tiling_mode: attr.tiling_mode,
        };
        ctx.gpu.present_scanout(buffer.address, Some(descriptor));
    }
    let event_hint = VIDEO_OUT_EVENT_FLIP_ID | ((flip_arg as u64 & 0x0000_ffff_ffff_ffff) << 16);
    let mut flip_queues = Vec::new();
    for mut event in ctx.kernel.kernel_equeue_events.iter_mut() {
        if event.key().1 == VIDEO_OUT_EVENT_FLIP_ID && event.filter == KERNEL_EVENT_FILTER_VIDEO_OUT
        {
            let eq = event.key().0;
            event.triggered = true;
            event.fflags = event.fflags.saturating_add(1);
            event.data = event_hint as i64;
            if !flip_queues.contains(&eq) {
                flip_queues.push(eq);
            }
        }
    }
    for eq in flip_queues {
        crate::kernel_equeue::wake_equeue(
            ctx,
            eq,
            raeen_core::subsystems::WakeReason::SubmissionComplete,
        );
    }
    // A completed flip implies a display refresh: advance the vblank sequence
    // and wake any vblank-parked frame loop (Raeen has no host vblank timer
    // thread — see `hle_add_vblank_event`).
    let vblanks = ctx
        .kernel
        .video_out_vblank_count
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    trigger_vblank_events(ctx, vblanks);
    SCE_OK
}

/// Complete a flip encoded inside an AGC submission. Gen5 games commonly use
/// this packet path instead of calling `sceVideoOutSubmitFlip` directly.
pub(crate) fn submit_flip_from_agc(
    ctx: &HleContext,
    handle: u32,
    buffer_index: u32,
    flip_mode: u32,
    flip_arg: u64,
) -> u64 {
    hle_submit_flip(
        ctx,
        &[
            u64::from(handle),
            u64::from(buffer_index),
            u64::from(flip_mode),
            flip_arg,
        ],
    )
}

/// `sceVideoOutGetFlipStatus(handle, SceVideoOutFlipStatus *status)`: reports
/// the completed flip count, the last `flipArg`, and **zero pending** flips —
/// so a title waiting on flip completion always sees it done and advances.
fn hle_get_flip_status(ctx: &HleContext, args: &[u64]) -> u64 {
    // Deliver any flips the GPU worker executed in-stream since the guest
    // last observed (no-op unless `RAEEN_DEFER_GPU_SIDE_EFFECTS` filled the
    // queue) — this status read IS the observation point for flip counts.
    crate::libsce_agc::apply_ordered_gpu_side_effects(ctx);
    let status_ptr = args.get(1).copied().unwrap_or(0);
    if status_ptr == 0 {
        return SCE_OK;
    }
    // SceVideoOutFlipStatus (64 bytes): count@0, processTime@8, tsc@16,
    // flipArg@24, submitTsc@32, reserved@40, gcQueueNum@48,
    // flipPendingNum@52, currentBuffer@56, reserved@60.
    let mut buf = [0u8; 64];
    let count = ctx.kernel.video_out_flip_count.load(Ordering::Relaxed);
    buf[0..8].copy_from_slice(&count.to_le_bytes());
    buf[24..32].copy_from_slice(
        &ctx.kernel
            .video_out_last_flip_arg
            .load(Ordering::Relaxed)
            .to_le_bytes(),
    );
    // flipPendingNum@52 stays 0 (nothing pending).
    buf[56..60].copy_from_slice(
        &ctx.kernel
            .video_out_current_buffer
            .load(Ordering::Relaxed)
            .to_le_bytes(),
    );
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
) -> Option<raeen_kernel::VideoOutBufferAttribute> {
    let mut bytes = [0u8; 0x50];
    if address == 0 || !ctx.mem.read(address, &mut bytes) {
        return None;
    }
    Some(raeen_kernel::VideoOutBufferAttribute {
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
        || !(1..=16).contains(&buffer_count)
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
            raeen_kernel::VideoOutBuffer {
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

/// `sceVideoOutColorSettingsSetGamma_(SceVideoOutColorSettings *settings,
/// float gamma)` — SharpEmu `VideoOutExports` (NID `DYhhWbJSeRg`): the gamma
/// arrives in XMM0; validate it is finite and inside [0.1, 2.0], then store
/// it as the settings object's leading float. Raeen's present path applies no
/// gamma yet, so the stored value is bookkeeping the later `AdjustColor_`
/// call reads back.
fn hle_color_settings_set_gamma(ctx: &HleContext, args: &[u64]) -> u64 {
    let settings = args.first().copied().unwrap_or(0);
    if settings == 0 {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    let gamma = ctx.float_arg_f32(0);
    if !gamma.is_finite() || !(0.1..=2.0).contains(&gamma) {
        return VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if !ctx.mem.write(settings, &gamma.to_bits().to_le_bytes()) {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    debug!("sceVideoOutColorSettingsSetGamma_(settings={settings:#x}, gamma={gamma})");
    SCE_OK
}

/// `sceVideoOutAdjustColor_(handle, const SceVideoOutColorSettings
/// *settings)` — SharpEmu `VideoOutExports` (NID `pv9CI5VC+R0`): accept the
/// settings built by `ColorSettingsSetGamma_`. The gamma is read (validating
/// the pointer) and logged; no display color pipeline exists to apply it to,
/// which only affects final image tint, never guest progress.
fn hle_adjust_color(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as i32;
    let settings = args.get(1).copied().unwrap_or(0);
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    if settings == 0 {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    let mut gamma = [0u8; 4];
    if !ctx.mem.read(settings, &mut gamma) {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    debug!(
        "sceVideoOutAdjustColor_(handle={handle}, gamma={}) -> accepted (no host color pipeline)",
        f32::from_bits(u32::from_le_bytes(gamma))
    );
    SCE_OK
}

/// `sceVideoOutSubmitChangeBufferAttribute2(handle, index, const
/// SceVideoOutBufferAttribute2 *attribute)`: re-describe an already
/// registered buffer slot (Gen5 titles switch pixel format / DCC state
/// between scenes). The registered entry's attribute is updated in place so
/// the present path sees the new layout; the slot's address is unchanged.
/// Signature mirrors the Gen4 `sceVideoOutSubmitChangeBufferAttribute`
/// (handle, index, attribute) with the Attribute2 block this file already
/// decodes for `RegisterBuffers2`.
fn hle_submit_change_buffer_attribute2(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as i32;
    let index = args.get(1).copied().unwrap_or(0) as i32;
    let attribute_address = args.get(2).copied().unwrap_or(0);
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    let Some(attribute) = read_buffer_attribute2(ctx, attribute_address) else {
        // Accept-and-log rather than fail: the address may be a layout this
        // decoder does not model yet, and a rejected attribute change would
        // stop a render loop over a cosmetic re-description.
        debug!(
            "sceVideoOutSubmitChangeBufferAttribute2(handle={handle}, index={index}): \
             attribute at {attribute_address:#x} unreadable — accepted without update"
        );
        return SCE_OK;
    };
    if let Some(mut buffer) = ctx.kernel.video_out_buffers.get_mut(&(handle, index)) {
        buffer.attribute = attribute;
        debug!(
            "sceVideoOutSubmitChangeBufferAttribute2(handle={handle}, index={index}) -> \
             {}x{} format={:#x}",
            attribute.width, attribute.height, attribute.pixel_format
        );
    } else {
        debug!(
            "sceVideoOutSubmitChangeBufferAttribute2(handle={handle}, index={index}): \
             slot not registered — accepted"
        );
    }
    SCE_OK
}

/// `sceVideoOutIsFlipPending(handle)`: how many submitted flips have not yet
/// completed. Raeen completes every flip synchronously at submit
/// (`hle_submit_flip`), so the honest answer is always 0 — matching SharpEmu's
/// `VideoOutIsFlipPending` (NID `zgXifHT9ErY`), which reports 0 after
/// validating the handle.
fn hle_is_flip_pending(ctx: &HleContext, args: &[u64]) -> u64 {
    // Complete any worker-executed in-stream flips first, so "0 pending"
    // stays honest under `RAEEN_DEFER_GPU_SIDE_EFFECTS` (no-op otherwise).
    crate::libsce_agc::apply_ordered_gpu_side_effects(ctx);
    let handle = args.first().copied().unwrap_or(0) as i32;
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    0
}

/// `sceVideoOutGetVblankStatus(handle, SceVideoOutVblankStatus *status)`:
/// reports the flip count as the vblank count (a monotonically-advancing
/// frame counter is enough for a title's frame-timing loop).
fn hle_get_vblank_status(ctx: &HleContext, args: &[u64]) -> u64 {
    // A completed in-stream flip advances the vblank sequence — deliver any
    // the worker executed before reporting (no-op with the gate off).
    crate::libsce_agc::apply_ordered_gpu_side_effects(ctx);
    let status_ptr = args.get(1).copied().unwrap_or(0);
    if status_ptr == 0 {
        return SCE_OK;
    }
    let mut buf = [0u8; 32];
    buf[0..8].copy_from_slice(
        &ctx.kernel
            .video_out_vblank_count
            .load(Ordering::Relaxed)
            .to_le_bytes(),
    ); // count@0
    if !ctx.mem.write(status_ptr, &buf) {
        debug!("sceVideoOutGetVblankStatus: status out-ptr {status_ptr:#x} not writable");
    }
    SCE_OK
}

/// Nominal display refresh period. Default 60 Hz matches the base-console
/// display contract; `RAEEN_VBLANK_HZ=120` selects the PS5's 120 Hz output
/// mode. `RAEEN_VBLANK_HZ=0` is the explicit unpaced benchmark mode used by
/// `cargo xtask compat run --profile max-fps`; it still advances the guest's
/// vblank sequence and events, but does not sleep. The Shell only exposes
/// 24–480 Hz, so production launches cannot select this accidentally.
///
/// MEASURED (stage C): the old unconditional 16.667 ms sleep was the
/// whole-title FPS ceiling — Minecraft's flip loop paced off this wait while
/// the GPU path had ~8x headroom (min flip interval 2.03 ms vs p50 16.5 ms).
fn configured_vblank_period(value: Option<&str>) -> Option<std::time::Duration> {
    match value.and_then(|value| value.parse::<u64>().ok()) {
        Some(0) => None,
        Some(hz @ 24..=480) => Some(std::time::Duration::from_nanos(1_000_000_000 / hz)),
        _ => Some(std::time::Duration::from_nanos(1_000_000_000 / 60)),
    }
}

fn vblank_period() -> Option<std::time::Duration> {
    static PERIOD: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *PERIOD
        .get_or_init(|| configured_vblank_period(std::env::var("RAEEN_VBLANK_HZ").ok().as_deref()))
}

/// Wait until the next vblank edge on the process-wide schedule.
///
/// Edges are anchored to a fixed epoch (`epoch + n·period`), not to "now +
/// period": per-call relative sleeps drift and quantize. The wait itself is
/// drift-compensated accurate sleep (concept from shadPS4's
/// `AccurateSleep`/`Timer`, GPL-2.0 — reimplemented): an absolute deadline, a
/// high-resolution OS wait for the bulk, and a bounded final spin so a late
/// signal never overshoots a full edge. Windows' default timer resolution
/// (~15.6 ms) rounds ANY shorter `thread::sleep` up to a full tick — the
/// stage-C measurement saw exactly that signature (20.2 ms intervals from a
/// 16.7 ms sleep) — so the bulk wait goes through a high-resolution waitable
/// timer (~0.5 ms) instead of `Sleep`, and only the last millisecond spins.
/// That removes up to a whole tick of yield-spin per wait from the guest core
/// the title parks here. At 120 Hz the whole 8.3 ms wait is one timer signal.
fn wait_next_vblank_edge() {
    let Some(period) = vblank_period() else {
        return;
    };
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let epoch = *EPOCH.get_or_init(std::time::Instant::now);
    let elapsed = epoch.elapsed();
    let next = (elapsed.as_nanos() / period.as_nanos() + 1) as u64;
    let deadline = std::time::Duration::from_nanos(next.saturating_mul(period.as_nanos() as u64));
    /// The final slice of every wait is a spin: an OS signal can arrive early,
    /// and overshooting an edge is worse than spinning a millisecond.
    const SPIN_REMAINDER: std::time::Duration = std::time::Duration::from_millis(1);
    /// Coarse-sleep granularity of the no-high-res-timer fallback.
    const TIMER_TICK: std::time::Duration = std::time::Duration::from_millis(17);
    loop {
        let now = epoch.elapsed();
        if now >= deadline {
            return;
        }
        let remaining = deadline - now;
        if remaining <= SPIN_REMAINDER {
            std::thread::yield_now();
            continue;
        }
        #[cfg(windows)]
        let served = precise_wait::wait(remaining - SPIN_REMAINDER);
        #[cfg(not(windows))]
        let served = false;
        if !served {
            // Fallback without a high-resolution timer: coarse-sleep, never
            // past the spin window, and re-derive the remaining time each lap.
            std::thread::sleep((remaining - SPIN_REMAINDER).min(TIMER_TICK));
        }
    }
}

/// High-resolution waitable-timer sleep (Windows 10 1803+). One unnamed
/// manual-reset timer per waiting thread, created lazily; thread exit closes
/// it. A process may park several guest threads in `sceVideoOutWaitVblank` /
/// kernel waits, so the handle is thread-local, never shared.
#[cfg(windows)]
mod precise_wait {
    use std::time::Duration;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::{
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CREATE_WAITABLE_TIMER_MANUAL_RESET,
        CreateWaitableTimerExW, SetWaitableTimer, TIMER_MODIFY_STATE, WaitForSingleObject,
    };

    /// Wait indefinitely for one object; not exported by windows-sys 0.59.
    const INFINITE: u32 = 0xFFFF_FFFF;

    /// Object-access right to wait on a handle; not exported by windows-sys 0.59.
    const SYNCHRONIZATION: u32 = 0x0010_0000;

    thread_local! {
        static HIGH_RES_TIMER: Option<HighResTimer> = HighResTimer::new();
    }

    struct HighResTimer(HANDLE);

    impl HighResTimer {
        fn new() -> Option<Self> {
            // SAFETY: creating an unnamed manual-reset timer; null attributes
            // and name are valid. The high-resolution flag is ignored on
            // Windows builds that predate it, degrading to a normal timer.
            let handle = unsafe {
                CreateWaitableTimerExW(
                    std::ptr::null(),
                    std::ptr::null(),
                    CREATE_WAITABLE_TIMER_MANUAL_RESET | CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                    TIMER_MODIFY_STATE | SYNCHRONIZATION,
                )
            };
            (!handle.is_null()).then_some(Self(handle))
        }

        fn wait(&self, remaining: Duration) {
            // Negative due time = relative, in 100 ns units.
            let due = -((remaining.as_nanos() / 100).min(i64::MAX as u128) as i64);
            // SAFETY: `self.0` is a live timer owned by this thread; the
            // due-time pointer is valid; no APC completion routine is used.
            let ok = unsafe { SetWaitableTimer(self.0, &due, 0, None, std::ptr::null(), 0) };
            if ok != 0 {
                // SAFETY: waiting on this thread's own timer; it signals at
                // the due time, so INFINITE cannot hang.
                unsafe { WaitForSingleObject(self.0, INFINITE) };
            }
        }
    }

    impl Drop for HighResTimer {
        fn drop(&mut self) {
            // SAFETY: closing a handle this thread created and owns.
            unsafe { CloseHandle(self.0) };
        }
    }

    /// Sleep for `remaining` at sub-millisecond precision. Returns `false`
    /// when no high-resolution timer could be created (caller falls back).
    pub fn wait(remaining: Duration) -> bool {
        HIGH_RES_TIMER.with(|timer| {
            let Some(timer) = timer else {
                return false;
            };
            timer.wait(remaining);
            true
        })
    }
}

/// `sceVideoOutWaitVblank(handle)`: pace the native guest thread to the next
/// display edge on the process-wide vblank schedule (default 60 Hz;
/// `RAEEN_VBLANK_HZ` selects other modes), then advance the process-local
/// vblank sequence.
fn hle_wait_vblank(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as i32;
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    // Deliver any worker-executed in-stream flips before the frame-pacing
    // wait, so their flip/vblank events are pending when the guest resumes
    // (no-op unless `RAEEN_DEFER_GPU_SIDE_EFFECTS` filled the queue).
    crate::libsce_agc::apply_ordered_gpu_side_effects(ctx);
    wait_next_vblank_edge();
    let vblanks = ctx
        .kernel
        .video_out_vblank_count
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    trigger_vblank_events(ctx, vblanks);
    SCE_OK
}

/// `sceVideoOutIsOutputSupported(handle, mode, options, reservedPtr,
/// reserved)`: is the requested output mode available on this display?
/// Ported from SharpEmu `VideoOutIsOutputSupported` (VideoOutExports.cs, NID
/// `Nv8c-Kb+DUM`): reserved args must be zero; a non-null options block must
/// read as `0x40` zero bytes; the mode must be Default (1) or 119.88 Hz
/// (0xF). Returns **1 = supported / 0 = unsupported** directly. Raeen reports
/// a 60 Hz display, so the 119.88 Hz VRR-class mode is honestly unsupported.
fn hle_is_output_supported(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as i32;
    let mode = args.get(1).copied().unwrap_or(0);
    let options = args.get(2).copied().unwrap_or(0);
    let reserved_ptr = args.get(3).copied().unwrap_or(0);
    let reserved = args.get(4).copied().unwrap_or(0);
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    if reserved_ptr != 0 || reserved != 0 {
        return VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if options != 0 {
        let mut block = [0u8; OUTPUT_OPTIONS_SIZE];
        if !ctx.mem.read(options, &mut block) {
            return SCE_ERROR_MEMORY_FAULT;
        }
        if block.iter().any(|byte| *byte != 0) {
            return VIDEO_OUT_ERROR_INVALID_OPTION;
        }
    }
    if mode != OUTPUT_MODE_DEFAULT && mode != OUTPUT_MODE_119_88_HZ {
        return VIDEO_OUT_ERROR_UNSUPPORTED_OUTPUT_MODE;
    }
    u64::from(mode == OUTPUT_MODE_DEFAULT || DISPLAY_REFRESH_HZ >= 119)
}

/// `sceVideoOutConfigureOutput(handle, ...)`: apply an output configuration.
/// Ported from SharpEmu `VideoOutConfigureOutput` (VideoOutExports.cs, NID
/// `w0hLuNarQxY`), which validates the handle and returns OK without storing
/// anything — the port model has no output-mode fields to update yet.
fn hle_configure_output(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as i32;
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    debug!("sceVideoOutConfigureOutput(handle={handle}) -> accepted");
    SCE_OK
}

/// `sceVideoOutInitializeOutputOptions(options)`: zero-initialize a
/// `SceVideoOutOutputOptions` block (0x40 bytes). Ported from SharpEmu
/// `VideoOutInitializeOutputOptions` (VideoOutExports.cs, NID `+I4K03i3EL0`).
fn hle_initialize_output_options(ctx: &HleContext, args: &[u64]) -> u64 {
    let options = args.first().copied().unwrap_or(0);
    if options == 0 {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    if !ctx.mem.write(options, &[0u8; OUTPUT_OPTIONS_SIZE]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    SCE_OK
}

/// `sceVideoOutSetWindowModeMargins(handle, top, bottom)`: window-mode
/// letterbox margins. Ported from SharpEmu `VideoOutSetWindowModeMargins`
/// (VideoOutExports.cs, NID `MTxxrOCeSig`): validate the handle, accept the
/// margins (it discards them too — margins only shift the presented image).
fn hle_set_window_mode_margins(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as i32;
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    debug!(
        "sceVideoOutSetWindowModeMargins(handle={handle}, top={}, bottom={}) -> accepted",
        args.get(1).copied().unwrap_or(0) as i32,
        args.get(2).copied().unwrap_or(0) as i32
    );
    SCE_OK
}

/// `sceVideoOutUnregisterBuffers(handle, setIndex)`: drop every display
/// buffer registered under attribute-set `setIndex`. Ported from SharpEmu
/// `VideoOutUnregisterBuffers` (VideoOutExports.cs, NID `N5KDtkIjjJ4`): a
/// negative or never-registered set index is `INVALID_VALUE`; on success the
/// group and its buffer slots are cleared (here: the `(handle, slot)` entries
/// whose `set_index` matches are removed from the port model).
fn hle_unregister_buffers(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as i32;
    let set_index = args.get(1).copied().unwrap_or(0) as i32;
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    if set_index < 0 {
        return VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    let slots: Vec<(i32, i32)> = ctx
        .kernel
        .video_out_buffers
        .iter()
        .filter(|entry| entry.key().0 == handle && entry.set_index == set_index)
        .map(|entry| *entry.key())
        .collect();
    if slots.is_empty() {
        return VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    for slot in &slots {
        ctx.kernel.video_out_buffers.remove(slot);
    }
    debug!(
        "sceVideoOutUnregisterBuffers(handle={handle}, set={set_index}) -> {} slot(s) cleared",
        slots.len()
    );
    SCE_OK
}

/// `sceVideoOutGetEventId(const SceKernelEvent *event)`: classify a delivered
/// VideoOut kernel event — returns **0 = flip, 1 = vblank** (positive return,
/// not an out-param). Ported from SharpEmu `VideoOutGetEventId`
/// (VideoOutExports.cs, NID `U2JJtSqNKZI`): reads `ident` (u64 @ +0x00) and
/// `filter` (i16 @ +0x08) from the guest event struct; a non-VideoOut filter
/// or unknown ident is `INVALID_EVENT`. The idents match what this module
/// registers/triggers: flip = 0x6, vblank = 0x40.
fn hle_get_event_id(ctx: &HleContext, args: &[u64]) -> u64 {
    let event = args.first().copied().unwrap_or(0);
    if event == 0 {
        return VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    let mut ident = [0u8; 8];
    let mut filter = [0u8; 2];
    if !ctx.mem.read(event, &mut ident) || !ctx.mem.read(event + 0x08, &mut filter) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    if i16::from_le_bytes(filter) != KERNEL_EVENT_FILTER_VIDEO_OUT {
        return VIDEO_OUT_ERROR_INVALID_EVENT;
    }
    match u64::from_le_bytes(ident) {
        VIDEO_OUT_EVENT_FLIP_ID => 0,
        VIDEO_OUT_EVENT_VBLANK_ID => 1,
        _ => VIDEO_OUT_ERROR_INVALID_EVENT,
    }
}

/// `sceVideoOutVrrPegToFixedRate(handle, ...)` /
/// `sceVideoOutVrrUnpegFromFixedRate(handle)`: pin/unpin the variable
/// refresh rate to a fixed rate. No reference implements these Gen5 exports
/// (absent from SharpEmu/Kyty — SharpEmu has no Vrr entry points at all), and
/// Raeen reports a fixed 60 Hz display with no VRR hardware to steer, so both
/// are accepted as OK no-ops with the arguments recorded for future RE.
/// Measured Until Dawn + Dragon Ball Sparking Zero imports (NIDs
/// `5tRaBjtdTzY` / `T4ucGB8CsnM`).
fn hle_vrr_fixed_rate(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as i32;
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    debug!(
        "sceVideoOutVrr[Peg|Unpeg]FixedRate(handle={handle}, arg1={:#x}) -> accepted (fixed 60 Hz display, no VRR)",
        args.get(1).copied().unwrap_or(0)
    );
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    /// Ordered GPU side effects (checklist item 5, step 5): a flip the GPU
    /// worker executed in-stream and published is completed by the flip-status
    /// read's drain — the status call IS the observation point for flip
    /// visibility.
    #[test]
    fn get_flip_status_delivers_worker_published_flips() {
        // The hand-off queue is process-global: serialize with every other
        // test that touches it.
        let _guard = crate::SIDEFX_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ = raeen_gpu::ordered_side_effects::drain();
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        raeen_gpu::ordered_side_effects::publish([
            raeen_gpu::ordered_side_effects::OrderedGpuSideEffect::Flip {
                video_out_handle: 1,
                display_buffer_index: 0,
                flip_mode: 1,
                flip_arg: 0x77,
            },
        ]);
        hle_get_flip_status(&ctx, &[1, 0x200]);
        let mut st = [0u8; 64];
        assert!(mem.read(0x200, &mut st));
        let count = u64::from_le_bytes(st[0..8].try_into().unwrap());
        let flip_arg = i64::from_le_bytes(st[24..32].try_into().unwrap());
        assert_eq!(count, 1, "the worker's flip completes at the status read");
        assert_eq!(flip_arg, 0x77, "with the packet's flip arg");
    }

    #[test]
    fn submit_flip_advances_the_reported_flip_count() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
    fn vblank_period_has_an_explicit_unpaced_benchmark_mode() {
        assert_eq!(configured_vblank_period(Some("0")), None);
        assert_eq!(
            configured_vblank_period(Some("120")),
            Some(std::time::Duration::from_nanos(1_000_000_000 / 120))
        );
        for invalid in [None, Some("23"), Some("481"), Some("not-a-rate")] {
            assert_eq!(
                configured_vblank_period(invalid),
                Some(std::time::Duration::from_nanos(1_000_000_000 / 60))
            );
        }
    }

    /// Two consecutive vblank waits must land one full period apart: the first
    /// anchors to the next edge, the second to the edge after. A wait that
    /// returned early (broken timer) or a full tick late (uncompensated drift)
    /// fails the window. Skipped in the unpaced benchmark mode, where the wait
    /// contract is explicitly void.
    #[test]
    fn consecutive_vblank_waits_land_one_period_apart() {
        let Some(period) = vblank_period() else {
            eprintln!("unpaced benchmark mode (RAEEN_VBLANK_HZ=0): skipping");
            return;
        };
        wait_next_vblank_edge();
        let start = std::time::Instant::now();
        wait_next_vblank_edge();
        let interval = start.elapsed();
        assert!(
            interval >= period - std::time::Duration::from_millis(3),
            "wait returned {interval:?} after the previous edge — early vs period {period:?}"
        );
        assert!(
            interval <= period + std::time::Duration::from_millis(10),
            "wait took {interval:?} for one period {period:?} — late signal or drift"
        );
    }

    #[test]
    fn wait_vblank_advances_a_separate_frame_sequence() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_wait_vblank(&ctx, &[1]), SCE_OK);
        assert_eq!(kernel.video_out_vblank_count.load(Ordering::Relaxed), 1);
        assert_eq!(kernel.video_out_flip_count.load(Ordering::Relaxed), 0);
        assert_eq!(hle_wait_vblank(&ctx, &[99]), VIDEO_OUT_ERROR_INVALID_HANDLE);

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libSceVideoOut", "sceVideoOutWaitVblank"));
    }

    #[test]
    fn agc_flip_triggers_registered_video_out_event() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = kernel.create_equeue(0);
        assert_eq!(hle_add_flip_event(&ctx, &[eq, 1, 0xCAFE]), SCE_OK);
        assert_eq!(submit_flip_from_agc(&ctx, 1, 2, 1, 0x1234), SCE_OK);
        let event = kernel.kernel_equeue_events.get(&(eq, 6)).unwrap();
        assert!(event.triggered);
        assert_eq!(event.filter, -13);
        assert_eq!(event.udata, 0xCAFE);
        assert_eq!(event.data as u64, 6 | (0x1234 << 16));
    }

    #[test]
    fn color_and_flip_state_calls_validate_and_report() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let mut ctx = test_ctx(&kernel, &mem, &alloc);

        // Gamma arrives in XMM0: 1.25 is stored into the settings object.
        ctx.float_args[0] = u64::from(1.25f32.to_bits());
        assert_eq!(hle_color_settings_set_gamma(&ctx, &[0x100]), SCE_OK);
        let mut g = [0u8; 4];
        assert!(mem.read(0x100, &mut g));
        assert_eq!(f32::from_bits(u32::from_le_bytes(g)), 1.25);
        // Out-of-range gamma is rejected (SharpEmu validates 0.1..=2.0).
        ctx.float_args[0] = u64::from(5.0f32.to_bits());
        assert_eq!(
            hle_color_settings_set_gamma(&ctx, &[0x100]),
            VIDEO_OUT_ERROR_INVALID_VALUE
        );
        assert_eq!(
            hle_color_settings_set_gamma(&ctx, &[0]),
            VIDEO_OUT_ERROR_INVALID_ADDRESS
        );

        // AdjustColor accepts the settings on the open port only.
        assert_eq!(hle_adjust_color(&ctx, &[1, 0x100]), SCE_OK);
        assert_eq!(
            hle_adjust_color(&ctx, &[7, 0x100]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );

        // No flip is ever left pending (flips complete at submit).
        assert_eq!(hle_is_flip_pending(&ctx, &[1]), 0);
        assert_eq!(hle_submit_flip(&ctx, &[1, 0, 1, 7]), SCE_OK);
        assert_eq!(hle_is_flip_pending(&ctx, &[1]), 0);
        assert_eq!(
            hle_is_flip_pending(&ctx, &[9]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
    }

    /// SubmitChangeBufferAttribute2 re-describes a registered slot in place.
    #[test]
    fn change_buffer_attribute2_updates_the_registered_slot() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Register one buffer at slot 0 with a 1920x1080 attribute.
        assert_eq!(
            hle_set_buffer_attribute2(
                &ctx,
                &[0x100, 0x8100_0000_2200_0000, 0, 1920, 1080, 0, 0, 0]
            ),
            SCE_OK
        );
        assert!(mem.write(0x200, &0x4000u64.to_le_bytes()));
        assert!(mem.write(0x208, &0u64.to_le_bytes()));
        assert_eq!(
            hle_register_buffers2(&ctx, &[1, 0, 0, 0x200, 1, 0x100, 0, 0]),
            0
        );

        // Re-describe it as 1280x720 with a different format.
        assert_eq!(
            hle_set_buffer_attribute2(&ctx, &[0x300, 0x1234, 0, 1280, 720, 0, 0, 0]),
            SCE_OK
        );
        assert_eq!(
            hle_submit_change_buffer_attribute2(&ctx, &[1, 0, 0x300]),
            SCE_OK
        );
        let buffer = kernel.video_out_buffers.get(&(1, 0)).unwrap();
        assert_eq!(buffer.attribute.width, 1280);
        assert_eq!(buffer.attribute.height, 720);
        assert_eq!(buffer.attribute.pixel_format, 0x1234);
        assert_eq!(buffer.address, 0x4000, "address is unchanged");
        drop(buffer);

        assert_eq!(
            hle_submit_change_buffer_attribute2(&ctx, &[3, 0, 0x300]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
    }

    #[test]
    fn resolution_status_reports_1080p() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        assert!(registry.is_implemented("libSceVideoOut", "sceVideoOutRegisterBuffers2"));
        assert!(registry.is_implemented("libSceVideoOut", "sceVideoOutSetBufferAttribute2"));
    }

    /// SharpEmu `VideoOutIsOutputSupported` parity on a 60 Hz display.
    #[test]
    fn is_output_supported_reports_default_only_on_a_60_hz_display() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Default mode: supported (1). 119.88 Hz on a 60 Hz display: not (0).
        assert_eq!(hle_is_output_supported(&ctx, &[1, 1, 0, 0, 0]), 1);
        assert_eq!(hle_is_output_supported(&ctx, &[1, 0xF, 0, 0, 0]), 0);
        // Unknown mode / reserved args / bad handle → the SharpEmu errors.
        assert_eq!(
            hle_is_output_supported(&ctx, &[1, 7, 0, 0, 0]),
            VIDEO_OUT_ERROR_UNSUPPORTED_OUTPUT_MODE
        );
        assert_eq!(
            hle_is_output_supported(&ctx, &[1, 1, 0, 0x10, 0]),
            VIDEO_OUT_ERROR_INVALID_VALUE
        );
        assert_eq!(
            hle_is_output_supported(&ctx, &[9, 1, 0, 0, 0]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
        // A non-null options block must be all-zero.
        assert!(mem.write(0x100, &[0u8; OUTPUT_OPTIONS_SIZE]));
        assert_eq!(hle_is_output_supported(&ctx, &[1, 1, 0x100, 0, 0]), 1);
        assert!(mem.write(0x100, &[1u8]));
        assert_eq!(
            hle_is_output_supported(&ctx, &[1, 1, 0x100, 0, 0]),
            VIDEO_OUT_ERROR_INVALID_OPTION
        );
    }

    /// InitializeOutputOptions zeroes the 0x40-byte block; ConfigureOutput and
    /// SetWindowModeMargins accept on the open port only.
    #[test]
    fn output_configuration_calls_validate_and_accept() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, &[0xAAu8; OUTPUT_OPTIONS_SIZE]));
        assert_eq!(hle_initialize_output_options(&ctx, &[0x100]), SCE_OK);
        let mut block = [0u8; OUTPUT_OPTIONS_SIZE];
        assert!(mem.read(0x100, &mut block));
        assert!(block.iter().all(|byte| *byte == 0));
        assert_eq!(
            hle_initialize_output_options(&ctx, &[0]),
            VIDEO_OUT_ERROR_INVALID_ADDRESS
        );

        assert_eq!(hle_configure_output(&ctx, &[1]), SCE_OK);
        assert_eq!(
            hle_configure_output(&ctx, &[5]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
        assert_eq!(hle_set_window_mode_margins(&ctx, &[1, 32, 32]), SCE_OK);
        assert_eq!(
            hle_set_window_mode_margins(&ctx, &[5, 0, 0]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
        // VRR peg/unpeg accepts on the open port (no VRR hardware modeled).
        assert_eq!(hle_vrr_fixed_rate(&ctx, &[1, 60]), SCE_OK);
        assert_eq!(
            hle_vrr_fixed_rate(&ctx, &[3]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
    }

    /// UnregisterBuffers drops exactly the slots registered under the given
    /// attribute set, per SharpEmu semantics.
    #[test]
    fn unregister_buffers_clears_the_attribute_sets_slots() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        assert!(kernel.video_out_buffers.contains_key(&(1, 0)));
        assert!(kernel.video_out_buffers.contains_key(&(1, 1)));

        // A set index that was never registered → INVALID_VALUE.
        assert_eq!(
            hle_unregister_buffers(&ctx, &[1, 9]),
            VIDEO_OUT_ERROR_INVALID_VALUE
        );
        assert_eq!(hle_unregister_buffers(&ctx, &[1, 3]), SCE_OK);
        assert!(!kernel.video_out_buffers.contains_key(&(1, 0)));
        assert!(!kernel.video_out_buffers.contains_key(&(1, 1)));
        // Second unregister of the same set: nothing left → INVALID_VALUE.
        assert_eq!(
            hle_unregister_buffers(&ctx, &[1, 3]),
            VIDEO_OUT_ERROR_INVALID_VALUE
        );
        assert_eq!(
            hle_unregister_buffers(&ctx, &[7, 3]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
    }

    /// Vblank events registered via AddVblankEvent fire on WaitVblank and on
    /// flip completion; GetEventId classifies flip (0) vs vblank (1) from the
    /// delivered SceKernelEvent per SharpEmu.
    #[test]
    fn vblank_events_fire_and_get_event_id_classifies() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = kernel.create_equeue(0);

        assert_eq!(hle_add_vblank_event(&ctx, &[eq, 1, 0xBEEF]), SCE_OK);
        assert_eq!(
            hle_add_vblank_event(&ctx, &[eq, 9, 0]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
        {
            let event = kernel
                .kernel_equeue_events
                .get(&(eq, VIDEO_OUT_EVENT_VBLANK_ID))
                .unwrap();
            assert!(!event.triggered);
            assert_eq!(event.filter, KERNEL_EVENT_FILTER_VIDEO_OUT);
            assert_eq!(event.udata, 0xBEEF);
        }
        // WaitVblank ticks the sequence and wakes the registration.
        assert_eq!(hle_wait_vblank(&ctx, &[1]), SCE_OK);
        {
            let event = kernel
                .kernel_equeue_events
                .get(&(eq, VIDEO_OUT_EVENT_VBLANK_ID))
                .unwrap();
            assert!(event.triggered);
            assert_eq!(event.data as u64, VIDEO_OUT_EVENT_VBLANK_ID | (1 << 16));
        }
        // A completed flip also implies a vblank tick.
        if let Some(mut event) = kernel
            .kernel_equeue_events
            .get_mut(&(eq, VIDEO_OUT_EVENT_VBLANK_ID))
        {
            event.triggered = false;
        }
        assert_eq!(hle_submit_flip(&ctx, &[1, 0, 1, 5]), SCE_OK);
        assert!(
            kernel
                .kernel_equeue_events
                .get(&(eq, VIDEO_OUT_EVENT_VBLANK_ID))
                .unwrap()
                .triggered
        );

        // GetEventId reads ident@0 + filter@8 from the guest event struct.
        let mut event_struct = [0u8; 0x20];
        event_struct[0..8].copy_from_slice(&VIDEO_OUT_EVENT_FLIP_ID.to_le_bytes());
        event_struct[8..10].copy_from_slice(&KERNEL_EVENT_FILTER_VIDEO_OUT.to_le_bytes());
        assert!(mem.write(0x300, &event_struct));
        assert_eq!(hle_get_event_id(&ctx, &[0x300]), 0, "flip event → 0");
        event_struct[0..8].copy_from_slice(&VIDEO_OUT_EVENT_VBLANK_ID.to_le_bytes());
        assert!(mem.write(0x300, &event_struct));
        assert_eq!(hle_get_event_id(&ctx, &[0x300]), 1, "vblank event → 1");
        // Wrong filter or unknown ident → INVALID_EVENT.
        event_struct[8..10].copy_from_slice(&(-11i16).to_le_bytes());
        assert!(mem.write(0x300, &event_struct));
        assert_eq!(
            hle_get_event_id(&ctx, &[0x300]),
            VIDEO_OUT_ERROR_INVALID_EVENT
        );
        event_struct[0..8].copy_from_slice(&0x99u64.to_le_bytes());
        event_struct[8..10].copy_from_slice(&KERNEL_EVENT_FILTER_VIDEO_OUT.to_le_bytes());
        assert!(mem.write(0x300, &event_struct));
        assert_eq!(
            hle_get_event_id(&ctx, &[0x300]),
            VIDEO_OUT_ERROR_INVALID_EVENT
        );
        assert_eq!(
            hle_get_event_id(&ctx, &[0]),
            VIDEO_OUT_ERROR_INVALID_ADDRESS
        );

        // The whole measured UE5-pair + Plague Tale VideoOut set registers.
        let registry = HleRegistry::new();
        for name in [
            "sceVideoOutIsOutputSupported",
            "sceVideoOutConfigureOutput",
            "sceVideoOutInitializeOutputOptions",
            "sceVideoOutSetWindowModeMargins",
            "sceVideoOutUnregisterBuffers",
            "sceVideoOutAddVblankEvent",
            "sceVideoOutGetEventId",
            "sceVideoOutVrrPegToFixedRate",
            "sceVideoOutVrrUnpegFromFixedRate",
        ] {
            assert!(
                registry.is_implemented("libSceVideoOut", name),
                "missing libSceVideoOut::{name}"
            );
        }
    }
}
