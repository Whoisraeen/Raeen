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
use raeen_core::frame_path::{self, Stage};
use std::sync::atomic::Ordering;
use tracing::{debug, info};

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
/// SCE "no such entry" (`0x8002_0000 | ENOENT`) — the generic *kernel* error
/// KytyPS5's `KernelDeleteEvent` (`src/kernel/eventQueue.cpp:315`) returns when
/// the `(ident, filter)` pair being deleted was never registered. The
/// `sceVideoOutDelete*Event` family delegates to it, so it surfaces that error
/// rather than a `libSceVideoOut` one.
const SCE_KERNEL_ERROR_ENOENT: u64 = 0x8002_0002;
/// Kernel-event `ident` of a VideoOut **flip** event (SharpEmu
/// `SceVideoOutInternalEventFlip`).
const VIDEO_OUT_EVENT_FLIP_ID: u64 = 0x6;
/// Kernel-event `ident` of a VideoOut **vblank** event (SharpEmu
/// `SceVideoOutInternalEventVblank`).
const VIDEO_OUT_EVENT_VBLANK_ID: u64 = 0x40;
/// Kernel-event `ident` of a VideoOut **pre-vblank-start** event — the leading
/// edge of the same display refresh whose trailing edge is the vblank event
/// (KytyPS5 fires them from `VblankBegin` / `VblankEnd`, videoOut.cpp:649-685).
///
/// No reference supplies an *internal* ident for it: SharpEmu has no pre-vblank
/// path at all, and KytyPS5 keys its kevents by the **public** event id (0, 1,
/// 2, 8) rather than by the internal idents this module inherited from SharpEmu
/// (flip `0x6`, vblank `0x40`). So this value is Raeen-chosen — adjacent to the
/// vblank ident because it is the same refresh, and deliberately not a small
/// integer a guest is likely to pick for one of its own user events on the same
/// queue. Only distinctness is guest-visible: a title classifies a delivered
/// event through [`hle_get_event_id`], which maps this back to the public id 2
/// (KytyPS5 `VIDEO_OUT_EVENT_PRE_VBLANK_START`).
const VIDEO_OUT_EVENT_PRE_VBLANK_START_ID: u64 = 0x41;
/// Kernel-event `ident` of a VideoOut **output-mode** event. Raeen-chosen on
/// the same basis as [`VIDEO_OUT_EVENT_PRE_VBLANK_START_ID`]; maps to the public
/// id 8 (KytyPS5 `VIDEO_OUT_EVENT_SET_MODE`).
const VIDEO_OUT_EVENT_OUTPUT_MODE_ID: u64 = 0x42;
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
    registry.register(
        "libSceVideoOut",
        "sceVideoOutSetFlipRate",
        hle_set_flip_rate,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutRegisterBuffers",
        hle_register_buffers,
    );
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
    // `sceVideoOutSetBufferAttribute(SceVideoOutBufferAttribute *attr, format,
    // tilingMode, aspectRatio, width, height, pitchInPixel)` reads as a setter
    // but is really a *filler*: it zeroes the caller's attribute struct and
    // writes all seven fields into it. This stub writes none of them, and the
    // struct is what the title hands to `sceVideoOutRegisterBuffers` — which is
    // implemented and does read the layout back. `SetBufferAttribute2` (below)
    // is the path that actually records it.
    registry.register_incomplete(
        "libSceVideoOut",
        "sceVideoOutSetBufferAttribute",
        hle_ok,
        "reports success without filling the caller's SceVideoOutBufferAttribute out-struct",
    );
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
    // The rest of the VideoOut event family (KytyPS5 implements seven entry
    // points, Raeen previously implemented two — docs/silent-zero-frame-cluster.md
    // section 3). Blasphemous II (PPSA13580) is the measured import: its
    // `sceVideoOutDeleteFlipEvent` was unresolved in the baseline-1785285421268
    // run. The other four are not observed as imports in any title measured so
    // far; they are registered because a title that can *add* an event class it
    // cannot *delete* leaks the registration, and because an unresolved NID in
    // this family is a hard stop rather than a degraded frame.
    registry.register(
        "libSceVideoOut",
        "sceVideoOutDeleteFlipEvent",
        hle_delete_flip_event,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutDeleteVblankEvent",
        hle_delete_vblank_event,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutAddPreVblankStartEvent",
        hle_add_pre_vblank_start_event,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutDeletePreVblankStartEvent",
        hle_delete_pre_vblank_start_event,
    );
    registry.register(
        "libSceVideoOut",
        "sceVideoOutAddOutputModeEvent",
        hle_add_output_mode_event,
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

/// `sceVideoOutSetFlipRate(handle, rate)`: accepted as-is. Recorded so a title
/// that opens a handle and then stalls is distinguishable from one that never
/// chose a presentation cadence at all.
fn hle_set_flip_rate(_ctx: &HleContext, _args: &[u64]) -> u64 {
    frame_path::record(Stage::FlipRateSet);
    SCE_OK
}

/// `sceVideoOutRegisterBuffers(...)`: the pre-Gen5 registration entry point.
/// Accepted as-is; `sceVideoOutRegisterBuffers2` is the path that records
/// layout. Both count toward the frame path's buffers-registered rung.
fn hle_register_buffers(_ctx: &HleContext, _args: &[u64]) -> u64 {
    frame_path::record(Stage::BuffersRegistered);
    SCE_OK
}

fn hle_open(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "sceVideoOutOpen(userId={}, busType={}, index={})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    frame_path::record(Stage::VideoOutOpen);
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
    add_video_out_event(ctx, args, VIDEO_OUT_EVENT_FLIP_ID, "flip", None)
}

/// `sceVideoOutDeleteFlipEvent(equeue, handle)`: drop the flip registration
/// this queue holds. Ported from KytyPS5 `VideoOutDeleteFlipEvent`
/// (videoOut.cpp:1059, NID `-Ozn0F1AFRg`).
///
/// This is the one entry point in the missing set with **measured** evidence:
/// Blasphemous II imports it and our loader reported it unresolved
/// (`artifacts/compat/raw/baseline-1785285421268/PPSA13580-b5469945261a.stdout.log:322`).
fn hle_delete_flip_event(ctx: &HleContext, args: &[u64]) -> u64 {
    delete_video_out_event(ctx, args, VIDEO_OUT_EVENT_FLIP_ID, "flip")
}

/// `sceVideoOutAddVblankEvent(equeue, handle, udata)`: register a VideoOut
/// event triggered on display vblank. Ported from SharpEmu
/// `VideoOutAddVblankEvent` (VideoOutExports.cs, NID `Xru92wHJRmg`): same
/// (equeue, handle, udata) ABI as AddFlipEvent, re-registration replaces the
/// existing registration for that queue (here: same `(equeue, ident)` key).
/// SharpEmu starts a 60 Hz vblank thread. Raeen ticks vblank events from
/// `sceVideoOutWaitVblank` and from every completed flip (a flip implies a
/// display refresh), which keeps a **polling** frame loop advancing without a
/// host timer thread — but deadlocks an event-driven one that waits for its
/// first vblank *before* its first flip, because the only two tickers are the
/// two calls it is blocked from making.
///
/// [`crate::host_vblank`] is the timer thread that closes that hole
/// (`RAEEN_HOST_VBLANK`, default off). While it runs it owns the sequence and
/// the two guest-driven advances stand down.
fn hle_add_vblank_event(ctx: &HleContext, args: &[u64]) -> u64 {
    add_video_out_event(ctx, args, VIDEO_OUT_EVENT_VBLANK_ID, "vblank", None)
}

/// `sceVideoOutDeleteVblankEvent(equeue, handle)`: the mirror of
/// [`hle_add_vblank_event`]. Ported from KytyPS5 `VideoOutDeleteVblankEvent`
/// (videoOut.cpp:1069, NID `oNOQn3knW6s`).
fn hle_delete_vblank_event(ctx: &HleContext, args: &[u64]) -> u64 {
    delete_video_out_event(ctx, args, VIDEO_OUT_EVENT_VBLANK_ID, "vblank")
}

/// `sceVideoOutAddPreVblankStartEvent(equeue, handle, udata)`: register for the
/// *leading* edge of a display refresh. Ported from KytyPS5
/// `VideoOutAddPreVblankStartEvent` (videoOut.cpp:1085, NID `keipklF0pMY`).
///
/// KytyPS5 splits the refresh in two — `VblankBegin` fires pre-vblank-start,
/// `VblankEnd` fires vblank (videoOut.cpp:649-685). Raeen has a single vblank
/// tick (see [`hle_add_vblank_event`]), so [`trigger_vblank_events`] fires both
/// classes from it, each carrying the same sequence number. That collapses the
/// intra-frame ordering KytyPS5 preserves, but it does deliver: registering an
/// event class that nothing ever triggers is the exact "we ack, we never
/// deliver" failure this work exists to close.
fn hle_add_pre_vblank_start_event(ctx: &HleContext, args: &[u64]) -> u64 {
    add_video_out_event(
        ctx,
        args,
        VIDEO_OUT_EVENT_PRE_VBLANK_START_ID,
        "pre-vblank-start",
        None,
    )
}

/// `sceVideoOutDeletePreVblankStartEvent(equeue, handle)`: the mirror of
/// [`hle_add_pre_vblank_start_event`]. Ported from KytyPS5
/// `VideoOutDeletePreVblankStartEvent` (videoOut.cpp:1074, NID `elWQ9vERF-Q`).
fn hle_delete_pre_vblank_start_event(ctx: &HleContext, args: &[u64]) -> u64 {
    delete_video_out_event(
        ctx,
        args,
        VIDEO_OUT_EVENT_PRE_VBLANK_START_ID,
        "pre-vblank-start",
    )
}

/// `sceVideoOutAddOutputModeEvent(equeue, handle, udata)`: register for display
/// output-mode changes. Ported from KytyPS5 `VideoOutAddOutputModeEvent`
/// (videoOut.cpp:1091, NID `kmSe30JTs+E`).
///
/// KytyPS5 registers this class **already triggered**, with the handle's
/// current output mode as the payload (`RegisterVideoOutEvent`,
/// videoOut.cpp:366-376: `initially_triggered = kind == OutputMode`) — so a
/// title that registers and then blocks for the current mode is answered
/// immediately instead of waiting for a mode change that may never come. Raeen
/// drives a fixed display and never changes mode after registration, which
/// makes that initial delivery the *only* one; getting it wrong would park an
/// output-mode-driven init path forever. The payload is
/// [`OUTPUT_MODE_DEFAULT`], the mode `sceVideoOutIsOutputSupported` accepts.
fn hle_add_output_mode_event(ctx: &HleContext, args: &[u64]) -> u64 {
    add_video_out_event(
        ctx,
        args,
        VIDEO_OUT_EVENT_OUTPUT_MODE_ID,
        "output-mode",
        Some(OUTPUT_MODE_DEFAULT),
    )
}

/// Shared body of the `sceVideoOutAdd*Event` family.
///
/// `initial_trigger` is `Some(payload)` for a class that KytyPS5 registers
/// pre-triggered (output-mode only); the payload is encoded into `data` with
/// the same `ident | payload << 16` layout every trigger site in this file
/// uses, so `sceVideoOutGetEventData` decodes it identically to a later
/// delivery.
fn add_video_out_event(
    ctx: &HleContext,
    args: &[u64],
    ident: u64,
    kind: &str,
    initial_trigger: Option<u64>,
) -> u64 {
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
            triggered: initial_trigger.is_some(),
            fflags: u32::from(initial_trigger.is_some()),
            data: initial_trigger
                .map(|payload| (ident | ((payload & 0x0000_ffff_ffff_ffff) << 16)) as i64)
                .unwrap_or(0),
        },
    );
    debug!(equeue, handle, udata, "registered VideoOut {kind} event");
    SCE_OK
}

/// Shared body of the `sceVideoOutDelete*Event` family: the exact mirror of
/// [`add_video_out_event`] — the same handle and equeue validation, then the
/// `(equeue, ident)` key that Add inserted is removed.
///
/// Ported from KytyPS5 `DeleteVideoOutEvent` (videoOut.cpp:389). Deleting a
/// registration that was never made is [`SCE_KERNEL_ERROR_ENOENT`], the generic
/// kernel error KytyPS5 surfaces by delegating to `KernelDeleteEvent`
/// (eventQueue.cpp:315) — not a `libSceVideoOut` error.
fn delete_video_out_event(ctx: &HleContext, args: &[u64], ident: u64, kind: &str) -> u64 {
    let equeue = args.first().copied().unwrap_or(0);
    let handle = args.get(1).copied().unwrap_or(0) as i32;
    if handle != 1 {
        return VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    if !ctx.kernel.kernel_equeues.contains_key(&equeue) {
        return VIDEO_OUT_ERROR_INVALID_OPTION;
    }
    if ctx
        .kernel
        .kernel_equeue_events
        .remove(&(equeue, ident))
        .is_none()
    {
        debug!(
            equeue,
            handle, "no VideoOut {kind} event registered to delete"
        );
        return SCE_KERNEL_ERROR_ENOENT;
    }
    debug!(equeue, handle, "deleted VideoOut {kind} event");
    SCE_OK
}

/// Trigger every registered VideoOut vblank **and pre-vblank-start** event.
/// `data` carries the vblank sequence in the upper bits over the ident,
/// mirroring the flip-event encoding this file already uses (SharpEmu
/// `GetEventData` decodes `data >> 16`).
///
/// KytyPS5 fires the two classes from opposite ends of one display refresh
/// (`VblankBegin` / `VblankEnd`, videoOut.cpp:649-685) with independent
/// counters. Raeen has a single tick point, so both fire here with the same
/// sequence number — a registered pre-vblank-start event is delivered rather
/// than parked forever, at the cost of the intra-frame ordering.
fn trigger_vblank_events(ctx: &HleContext, count: u64) {
    trigger_vblank_events_via(
        ctx.kernel,
        ctx.services,
        ctx.guest_threads.current_thread(),
        count,
    );
}

/// The [`HleContext`]-free body of [`trigger_vblank_events`].
///
/// Everything a vblank delivery touches is either an [`raeen_kernel::OrbisKernel`]
/// field (`kernel_equeue_events`) or a [`WaitSubsystem::wake`] — it never reads
/// or writes guest memory, never allocates guest memory, and never submits to
/// the GPU. So it does not need the `mem` / `alloc` / `gpu` / `guest_calls`
/// borrows an [`HleContext`] carries, and a host thread holding an
/// `Arc<OrbisKernel>` can call it: `OrbisKernel` implements `WaitSubsystem`,
/// which is `Send + Sync`.
///
/// `guest_thread` is the diagnostics label for the wake (`0` from a host
/// thread). See [`crate::kernel_equeue::wake_equeue_via`].
///
/// [`WaitSubsystem::wake`]: raeen_core::subsystems::WaitSubsystem::wake
pub(crate) fn trigger_vblank_events_via(
    kernel: &raeen_kernel::OrbisKernel,
    waker: &dyn raeen_core::subsystems::WaitSubsystem,
    guest_thread: u64,
    count: u64,
) {
    let sequence = (count & 0x0000_ffff_ffff_ffff) << 16;
    let mut queues = Vec::new();
    for mut event in kernel.kernel_equeue_events.iter_mut() {
        let ident = event.key().1;
        if (ident == VIDEO_OUT_EVENT_VBLANK_ID || ident == VIDEO_OUT_EVENT_PRE_VBLANK_START_ID)
            && event.filter == KERNEL_EVENT_FILTER_VIDEO_OUT
        {
            let eq = event.key().0;
            event.triggered = true;
            event.fflags = event.fflags.saturating_add(1);
            event.data = (ident | sequence) as i64;
            if !queues.contains(&eq) {
                queues.push(eq);
            }
        }
    }
    for eq in queues {
        crate::kernel_equeue::wake_equeue_via(
            waker,
            eq,
            guest_thread,
            raeen_core::subsystems::WakeReason::Signal,
        );
    }
}

/// One **host-driven** display refresh: advance the process vblank sequence and
/// deliver it to every registered vblank / pre-vblank-start event. Returns the
/// new sequence number.
///
/// This is the guest-independent tick KytyPS5 runs from its window loop
/// (`GameShowWindow` → `VideoOutBeginVblank` / `VideoOutEndVblank`,
/// `src/graphics/presentation/window/window.cpp:350-354` and
/// `videoOut.cpp:649-686`). Called only by [`crate::host_vblank`]; when it is
/// running it is the **sole** advancer of `video_out_vblank_count` (see
/// [`crate::host_vblank::owns_sequence`]).
///
/// "Every opened handle" in KytyPS5 terms is "every registered vblank event" in
/// Raeen: our registrations are keyed by `(equeue, ident)` and `sceVideoOutOpen`
/// only ever hands out handle 1, so the per-handle loop and the per-registration
/// loop cover the same set.
pub(crate) fn host_vblank_refresh(
    kernel: &raeen_kernel::OrbisKernel,
    waker: &dyn raeen_core::subsystems::WaitSubsystem,
) -> u64 {
    let sequence = kernel
        .video_out_vblank_count
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    trigger_vblank_events_via(kernel, waker, HOST_VBLANK_GUEST_THREAD, sequence);
    sequence
}

/// Diagnostics `guest_thread` label for a wake that no guest thread caused.
const HOST_VBLANK_GUEST_THREAD: u64 = 0;

/// `sceVideoOutSubmitFlip(handle, bufferIndex, flipMode, flipArg)`: records
/// the flip as completed (bumps process-local state and stores `flipArg`) so a
/// subsequent `GetFlipStatus` shows the render loop it can proceed.
fn hle_submit_flip(ctx: &HleContext, args: &[u64]) -> u64 {
    let buffer_index = args.get(1).copied().unwrap_or(0);
    let flip_arg = args.get(3).copied().unwrap_or(0) as i64;
    debug!("sceVideoOutSubmitFlip(bufferIndex={buffer_index}, flipArg={flip_arg})");
    frame_path::record(Stage::FlipSubmitted);
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
        //
        // The pitch comes from the guest's `pitchInPixel`, not from the width.
        // Assuming `pitch == width` reads a padded buffer diagonally: with a
        // real stride of 2*width, presented row 0 is the left half of guest row
        // 0, row 1 the right half of guest row 0, row 2 the left half of guest
        // row 1 — so if the padding is unwritten, every other row comes out
        // uniformly dark. `effective_scanout_pitch` falls back to the width for
        // an unset or implausible value, which is the previous behavior.
        let attr = buffer.attribute;
        let descriptor = raeen_core::subsystems::ScanoutDescriptor {
            width: attr.width,
            height: attr.height,
            pitch_pixels: effective_scanout_pitch(attr.pitch_pixels, attr.width),
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
    // and wake any vblank-parked frame loop.
    //
    // ONE OWNER. When the host vblank source is running it is the sole advancer
    // of the sequence, and this inference is dropped — a flip that happens to
    // land between two host edges must not manufacture an extra refresh, or the
    // sequence a title uses for frame timing runs ahead of the display clock it
    // is supposed to measure. The flip events above still fire either way: a
    // flip really did complete, and that is not an inference.
    if !crate::host_vblank::owns_sequence() {
        let vblanks = ctx
            .kernel
            .video_out_vblank_count
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        trigger_vblank_events(ctx, vblanks);
    }
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
    let width = u32::from_le_bytes(bytes[0x0C..0x10].try_into().ok()?);
    // `pitchInPixel` at 0x14 — the one gap this decoder used to leave. Offsets
    // 0x04 (tilingMode), 0x0C (width), 0x10 (height) and 0x18 (option) match
    // the documented `SceVideoOutBufferAttribute` layout exactly, and in that
    // layout 0x14 is `pitchInPixel`. Skipping it and assuming `pitch == width`
    // is only correct for a tightly-packed buffer; for a padded one it makes
    // every row read start mid-row, which presents as horizontal striping.
    let pitch_pixels = u32::from_le_bytes(bytes[0x14..0x18].try_into().ok()?);
    if pitch_pixels != 0 && pitch_pixels != width {
        // Bounded: this runs on buffer registration, not per flip. Logged
        // because a pitch that differs from the width is exactly the condition
        // that used to be invisible, and naming it makes the next run decisive.
        info!(
            width,
            pitch_pixels,
            accepted = effective_scanout_pitch(pitch_pixels, width),
            "VideoOut buffer declares a row pitch wider than its visible width"
        );
    }
    Some(raeen_kernel::VideoOutBufferAttribute {
        tiling_mode: u32::from_le_bytes(bytes[0x04..0x08].try_into().ok()?),
        width,
        height: u32::from_le_bytes(bytes[0x10..0x14].try_into().ok()?),
        pitch_pixels,
        option: u64::from_le_bytes(bytes[0x18..0x20].try_into().ok()?),
        pixel_format: u64::from_le_bytes(bytes[0x20..0x28].try_into().ok()?),
        dcc_clear_color: u64::from_le_bytes(bytes[0x28..0x30].try_into().ok()?),
        dcc_control: u32::from_le_bytes(bytes[0x30..0x34].try_into().ok()?),
    })
}

/// Force the old `pitch == width` assumption: set to `width` to ignore the
/// guest's declared `pitchInPixel` entirely.
///
/// Exists so a suspected pitch regression can be A/B'd against a single run
/// without a rebuild — the scanout stride is not something a screenshot lets
/// you infer, so being able to toggle it is worth an env var.
pub const SCANOUT_PITCH_ENV: &str = "RAEEN_SCANOUT_PITCH";

/// The row stride, in pixels, to read a display buffer with.
///
/// Prefers the guest's declared `pitchInPixel`, but only when it is *plausible*
/// as a stride: at least the visible width (a shorter row cannot hold one) and
/// no more than four times it. The upper bound is deliberate — this field was
/// previously never decoded, so a title that leaves garbage there must not be
/// able to turn a working present into a failed read. Anything implausible
/// falls back to `width`, which is exactly the previous behavior, so a buffer
/// that really is tightly packed is bit-for-bit unaffected.
fn effective_scanout_pitch(declared: u32, width: u32) -> u32 {
    // Cached: this runs per flip, and an env lookup allocates and takes a lock.
    static FORCED_TO_WIDTH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let forced = *FORCED_TO_WIDTH.get_or_init(|| {
        std::env::var(SCANOUT_PITCH_ENV).is_ok_and(|v| v.eq_ignore_ascii_case("width"))
    });
    if forced {
        return width;
    }
    if declared >= width && declared <= width.saturating_mul(4) {
        declared
    } else {
        width
    }
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
    frame_path::record(Stage::BuffersRegistered);
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
pub(crate) fn configured_vblank_period(value: Option<&str>) -> Option<std::time::Duration> {
    match value.and_then(|value| value.parse::<u64>().ok()) {
        Some(0) => None,
        Some(hz @ 24..=480) => Some(std::time::Duration::from_nanos(1_000_000_000 / hz)),
        _ => Some(std::time::Duration::from_nanos(1_000_000_000 / 60)),
    }
}

pub(crate) fn vblank_period() -> Option<std::time::Duration> {
    static PERIOD: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *PERIOD
        .get_or_init(|| configured_vblank_period(std::env::var("RAEEN_VBLANK_HZ").ok().as_deref()))
}

/// Wait until the next vblank edge on the process-wide schedule.
///
/// Edges are anchored to a fixed epoch (`epoch + n·period`), not to "now +
/// period": per-call relative sleeps drift and quantize. That anchoring is this
/// module's job. The *waiting* is `raeen_core::host_sleep`'s, which owns the one
/// measured host sleep strategy for the process — including the never-early
/// guarantee this schedule depends on, since a wait that returned before its
/// edge would hand the guest a frame that has not been scanned out yet.
///
/// This path used to carry its own copy of the thread-local
/// `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION` wait, which `host_sleep` was later
/// generalised from. Two copies of a timer primitive is two chances to drift,
/// and the copy here had a due time that truncated instead of rounding up.
///
/// Two claims from the original stage-C note did not survive `host_sleep`'s
/// measurements and are not repeated here. The 20.2 ms intervals seen from a
/// 16.7 ms sleep were real, but `std::thread::sleep` is not what quantises to
/// the ~15.6 ms tick — Rust already parks it on a high-resolution timer, and a
/// 10 ms sleep measures 10.2 ms. The tick is paid by condition-variable timed
/// waits, and the yield-spin this path used for its final millisecond measured
/// 60–190 ms under the emulator's own thread load, which is the likelier source
/// of an overshot edge. `host_sleep` parks that millisecond instead.
///
/// The trait exists so the edge arithmetic can be tested against a deterministic
/// clock: host scheduling load cannot turn a schedule unit test into a false
/// failure.
pub(crate) trait VblankClock {
    fn elapsed(&self) -> std::time::Duration;
    fn wait_until(&self, deadline: std::time::Duration);
}

pub(crate) struct HostVblankClock {
    epoch: std::time::Instant,
}

impl VblankClock for HostVblankClock {
    fn elapsed(&self) -> std::time::Duration {
        self.epoch.elapsed()
    }

    fn wait_until(&self, deadline: std::time::Duration) {
        // Hand `host_sleep` the absolute edge, not a remaining duration.
        // Subtracting here and letting `host_sleep::sleep` re-read the clock
        // would push every wait past its edge by the gap between the two
        // reads — reintroducing exactly the per-call drift the epoch-anchored
        // grid exists to remove.
        let Some(deadline) = self.epoch.checked_add(deadline) else {
            // Unreachable with a real schedule: `wait_next_vblank_edge_with_clock`
            // never picks an edge more than one period past now, and
            // `configured_vblank_period` bounds the period to 24-480 Hz. With
            // no representable edge to wait for, do not wait.
            return;
        };
        raeen_core::host_sleep::sleep_until(deadline);
    }
}

pub(crate) fn wait_next_vblank_edge_with_clock(
    period: std::time::Duration,
    clock: &impl VblankClock,
) {
    let elapsed = clock.elapsed();
    let next = elapsed.as_nanos() / period.as_nanos() + 1;
    let deadline_nanos = next.saturating_mul(period.as_nanos());
    let deadline = std::time::Duration::from_nanos(deadline_nanos.min(u64::MAX as u128) as u64);
    clock.wait_until(deadline);
}

/// The one process-wide vblank epoch. Absolute edges are `epoch + n·period`,
/// so `sceVideoOutWaitVblank` and the host vblank source
/// ([`crate::host_vblank`]) land on the **same** edge grid instead of on two
/// clocks that beat against each other.
pub(crate) fn vblank_epoch() -> std::time::Instant {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    *EPOCH.get_or_init(std::time::Instant::now)
}

/// Block until the next edge of the shared vblank grid.
pub(crate) fn wait_next_host_vblank_edge(period: std::time::Duration) {
    wait_next_vblank_edge_with_clock(
        period,
        &HostVblankClock {
            epoch: vblank_epoch(),
        },
    );
}

fn wait_next_vblank_edge() {
    let Some(period) = vblank_period() else {
        return;
    };
    wait_next_host_vblank_edge(period);
}

#[cfg(test)]
#[derive(Default)]
struct ManualVblankClock {
    now: std::cell::Cell<std::time::Duration>,
    deadlines: std::cell::RefCell<Vec<std::time::Duration>>,
}

#[cfg(test)]
impl ManualVblankClock {
    fn deadlines(&self) -> Vec<std::time::Duration> {
        self.deadlines.borrow().clone()
    }
}

#[cfg(test)]
impl VblankClock for ManualVblankClock {
    fn elapsed(&self) -> std::time::Duration {
        self.now.get()
    }

    fn wait_until(&self, deadline: std::time::Duration) {
        assert!(deadline >= self.now.get());
        self.deadlines.borrow_mut().push(deadline);
        self.now.set(deadline);
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
    // ONE OWNER (see `hle_submit_flip`). The pacing wait above is what the guest
    // asked for and always happens; only the *advance* is conditional. With the
    // host source running, the edge this wait returned on is the same absolute
    // edge that source ticks on — it shares the epoch and the period — so the
    // sequence the guest observes on resuming was advanced by the host tick for
    // this very refresh, not by a second count for it.
    if !crate::host_vblank::owns_sequence() {
        let vblanks = ctx
            .kernel
            .video_out_vblank_count
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        trigger_vblank_events(ctx, vblanks);
    }
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
/// VideoOut kernel event — returns **0 = flip, 1 = vblank, 2 = pre-vblank-start,
/// 8 = output-mode** (positive return, not an out-param). Ported from SharpEmu
/// `VideoOutGetEventId` (VideoOutExports.cs, NID `U2JJtSqNKZI`) for the first
/// two; the remaining public ids are KytyPS5's `VIDEO_OUT_EVENT_*`
/// (videoOut.cpp:39-42). Reads `ident` (u64 @ +0x00) and `filter` (i16 @ +0x08)
/// from the guest event struct; a non-VideoOut filter or unknown ident is
/// `INVALID_EVENT`.
///
/// This is the only place the internal idents this module keys registrations by
/// (flip `0x6`, vblank `0x40`, and the Raeen-chosen `0x41` / `0x42`) become
/// guest-visible numbers, which is why their exact values are internal.
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
        VIDEO_OUT_EVENT_PRE_VBLANK_START_ID => 2,
        VIDEO_OUT_EVENT_OUTPUT_MODE_ID => 8,
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

    /// Two consecutive vblank waits must choose consecutive absolute edges.
    /// The clock is deterministic: host scheduling load cannot turn this
    /// schedule-unit test into a false failure.
    #[test]
    fn consecutive_vblank_waits_land_one_period_apart() {
        let period = std::time::Duration::from_nanos(1_000_000_000 / 60);
        let clock = ManualVblankClock::default();
        wait_next_vblank_edge_with_clock(period, &clock);
        assert_eq!(clock.elapsed(), period);
        wait_next_vblank_edge_with_clock(period, &clock);
        assert_eq!(clock.elapsed(), period * 2);
        assert_eq!(clock.deadlines(), vec![period, period * 2]);
    }

    #[test]
    fn wait_vblank_advances_a_separate_frame_sequence() {
        // The guest-driven vblank advance under test only happens while no
        // host vblank source owns the sequence; pin that against a concurrent
        // ownership test (crate::host_vblank).
        let _vblank_owner = crate::host_vblank::OwnershipGuard::released();
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

    /// ONE OWNER, the double-tick rule. With a host vblank source running, the
    /// two guest-driven advance sites must contribute **zero** sequence numbers:
    /// a flip plus a `WaitVblank` leave the count exactly where the host source
    /// put it. Without this, a title's frame sequence outruns the display clock
    /// it is measuring, and any timestamp-fence logic keyed to it drifts.
    #[test]
    fn a_running_host_source_is_the_only_advancer_of_the_vblank_sequence() {
        let _vblank_owner = crate::host_vblank::OwnershipGuard::claimed();
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = kernel.create_equeue(0);
        assert_eq!(hle_add_vblank_event(&ctx, &[eq, 1, 0xBEEF]), SCE_OK);
        assert_eq!(hle_add_flip_event(&ctx, &[eq, 1, 0xCAFE]), SCE_OK);

        // One host refresh: sequence 1, vblank event delivered.
        let waker = crate::host_vblank::RecordingWaker::default();
        assert_eq!(host_vblank_refresh(&kernel, &waker), 1);

        // Now the guest does both of the things that used to advance it.
        assert_eq!(hle_submit_flip(&ctx, &[1, 0, 1, 0xABCD]), SCE_OK);
        assert_eq!(hle_wait_vblank(&ctx, &[1]), SCE_OK);

        assert_eq!(
            kernel.video_out_vblank_count.load(Ordering::Relaxed),
            1,
            "the host source is the sole advancer: a flip and a WaitVblank add nothing"
        );
        // The flip itself is NOT an inference and still completes + fires its
        // own event class — only the implied refresh is dropped.
        assert_eq!(kernel.video_out_flip_count.load(Ordering::Relaxed), 1);
        let flip = kernel
            .kernel_equeue_events
            .get(&(eq, VIDEO_OUT_EVENT_FLIP_ID))
            .expect("flip registration");
        assert!(
            flip.triggered,
            "a completed flip still fires its flip event"
        );
        assert_eq!(flip.data as u64, VIDEO_OUT_EVENT_FLIP_ID | (0xABCD << 16));
        drop(flip);
        // And the vblank event still carries the HOST sequence, not a guest one.
        let vblank = kernel
            .kernel_equeue_events
            .get(&(eq, VIDEO_OUT_EVENT_VBLANK_ID))
            .expect("vblank registration");
        assert_eq!(vblank.data as u64, VIDEO_OUT_EVENT_VBLANK_ID | (1 << 16));
    }

    /// The default: with no host source, `sceVideoOutSubmitFlip` advances and
    /// delivers the vblank sequence exactly as it did before this feature
    /// existed. This is the regression guard for Minecraft / ASTRO.BOT — a
    /// disabled host source must change nothing observable.
    #[test]
    fn with_no_host_source_a_flip_still_implies_a_refresh() {
        let _vblank_owner = crate::host_vblank::OwnershipGuard::released();
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = kernel.create_equeue(0);
        assert_eq!(hle_add_vblank_event(&ctx, &[eq, 1, 0xBEEF]), SCE_OK);

        assert_eq!(hle_submit_flip(&ctx, &[1, 0, 1, 0xABCD]), SCE_OK);

        assert_eq!(kernel.video_out_vblank_count.load(Ordering::Relaxed), 1);
        let vblank = kernel
            .kernel_equeue_events
            .get(&(eq, VIDEO_OUT_EVENT_VBLANK_ID))
            .expect("vblank registration");
        assert!(vblank.triggered);
        assert_eq!(vblank.data as u64, VIDEO_OUT_EVENT_VBLANK_ID | (1 << 16));
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

    /// Captures the `ScanoutDescriptor` a flip publishes. The pitch is not
    /// observable any other way — it leaves the HLE only through this call.
    struct RecordingGpu(std::sync::Mutex<Vec<raeen_core::subsystems::ScanoutDescriptor>>);

    impl RecordingGpu {
        fn new() -> Self {
            Self(std::sync::Mutex::new(Vec::new()))
        }

        fn only_descriptor(&self) -> raeen_core::subsystems::ScanoutDescriptor {
            let seen = self.0.lock().unwrap();
            assert_eq!(seen.len(), 1, "expected exactly one flip: {seen:?}");
            seen[0]
        }
    }

    impl crate::GpuSubmissionSubsystem for RecordingGpu {
        fn submit(&self, _words: Vec<u32>, _queue: raeen_core::subsystems::GpuQueue) {}
        fn map_shader_metadata(
            &self,
            _code_address: u64,
            _data: raeen_core::subsystems::ShaderMappedData,
        ) {
        }
        fn present_scanout(
            &self,
            _address: u64,
            descriptor: Option<raeen_core::subsystems::ScanoutDescriptor>,
        ) {
            if let Some(descriptor) = descriptor {
                self.0.lock().unwrap().push(descriptor);
            }
        }
        fn wait_idle(&self) {}
        fn stats(&self) -> raeen_core::subsystems::GpuSubmissionStats {
            raeen_core::subsystems::GpuSubmissionStats::default()
        }
    }

    /// Register one 1920x1080 buffer whose guest attribute declares
    /// `pitchInPixel = declared_pitch` at offset 0x14, flip it, and return the
    /// descriptor that reached the GPU.
    ///
    /// The struct is written byte-by-byte rather than through
    /// `hle_set_buffer_attribute2` because that filler takes no pitch argument
    /// — a retail title fills the struct itself, which is exactly the path
    /// under test.
    fn flip_with_declared_pitch(declared_pitch: u32) -> raeen_core::subsystems::ScanoutDescriptor {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let gpu = RecordingGpu::new();
        let ctx = crate::test_ctx_with_gpu(&kernel, &mem, &alloc, &gpu);

        let mut attribute = [0u8; 0x50];
        attribute[0x04..0x08].copy_from_slice(&1u32.to_le_bytes()); // tilingMode = linear
        attribute[0x0C..0x10].copy_from_slice(&1920u32.to_le_bytes()); // width
        attribute[0x10..0x14].copy_from_slice(&1080u32.to_le_bytes()); // height
        attribute[0x14..0x18].copy_from_slice(&declared_pitch.to_le_bytes()); // pitchInPixel
        attribute[0x20..0x28].copy_from_slice(&0x8000_2000u64.to_le_bytes()); // pixelFormat
        assert!(mem.write(0x100, &attribute));

        assert!(mem.write(0x200, &0x4000u64.to_le_bytes()));
        assert!(mem.write(0x208, &0u64.to_le_bytes()));
        assert_eq!(
            hle_register_buffers2(&ctx, &[1, 0, 0, 0x200, 1, 0x100, 0, 0]),
            0
        );
        assert_eq!(hle_submit_flip(&ctx, &[1, 0, 1, 0xABCD]), SCE_OK);
        gpu.only_descriptor()
    }

    /// A padded display buffer must be presented at the stride the guest
    /// declared, not at its visible width.
    ///
    /// Assuming `pitch == width` reads such a buffer diagonally: presented row 0
    /// is the left half of guest row 0, row 1 the RIGHT half of guest row 0, row
    /// 2 the left half of guest row 1. When the padding is unwritten that is
    /// exactly "every other row uniformly dark" across the whole frame.
    #[test]
    fn a_padded_display_buffer_keeps_the_guest_declared_row_pitch() {
        let descriptor = flip_with_declared_pitch(3840);
        assert_eq!(descriptor.width, 1920);
        assert_eq!(descriptor.height, 1080);
        assert_eq!(
            descriptor.pitch_pixels, 3840,
            "the declared pitch must survive to the scanout descriptor; \
             substituting the width here is what stripes the frame"
        );
    }

    /// The row mapping a pitch implies, stated as the arithmetic the present
    /// path performs — so the *consequence* of the pitch is pinned here, not
    /// just the value.
    ///
    /// Guards the case no scanout test covered: a stride wider than the visible
    /// width, over more than one row.
    #[test]
    fn a_wider_pitch_maps_each_presented_row_to_a_whole_guest_row() {
        let descriptor = flip_with_declared_pitch(3840);
        let row_bytes = descriptor.pitch_pixels as usize * 4;
        // Presented row y must start exactly y whole guest rows in.
        assert_eq!(row_bytes, 15_360);
        for y in 0..descriptor.height as usize {
            assert_eq!(y * row_bytes, y * 15_360, "row {y} start");
        }
        // The wrong stride aliases row 1 onto the middle of guest row 0 — the
        // striping signature. Assert the two mappings genuinely differ, so this
        // test fails if the pitch silently collapses back to the width.
        let wrong_row_bytes = descriptor.width as usize * 4;
        assert_ne!(
            row_bytes, wrong_row_bytes,
            "a padded buffer must not map rows at the visible width"
        );
        assert_eq!(wrong_row_bytes * 2, row_bytes, "the 2x aliasing");
    }

    /// A guest that leaves `pitchInPixel` unset gets the tightly-packed
    /// assumption — the previous behavior, bit for bit. This is what keeps
    /// already-working titles unaffected.
    #[test]
    fn an_unset_pitch_falls_back_to_a_tightly_packed_row() {
        assert_eq!(flip_with_declared_pitch(0).pitch_pixels, 1920);
    }

    /// This field was never decoded before, so a title with garbage there must
    /// not be able to turn a working present into a failed read.
    #[test]
    fn an_implausible_pitch_falls_back_to_the_width() {
        // Narrower than the visible width: cannot hold a row.
        assert_eq!(flip_with_declared_pitch(64).pitch_pixels, 1920);
        // Absurdly wide: would size a read no buffer can satisfy.
        assert_eq!(flip_with_declared_pitch(0xDEAD_BEEF).pitch_pixels, 1920);
    }

    /// The pitch policy in isolation, including the exact acceptance boundary.
    #[test]
    fn effective_scanout_pitch_accepts_only_plausible_strides() {
        // Equal, and the whole plausible range up to 4x, are taken verbatim.
        assert_eq!(effective_scanout_pitch(1920, 1920), 1920);
        assert_eq!(effective_scanout_pitch(1984, 1920), 1984);
        assert_eq!(effective_scanout_pitch(3840, 1920), 3840);
        assert_eq!(effective_scanout_pitch(7680, 1920), 7680);
        // Just past 4x is rejected.
        assert_eq!(effective_scanout_pitch(7681, 1920), 1920);
        // Unset and short are rejected.
        assert_eq!(effective_scanout_pitch(0, 1920), 1920);
        assert_eq!(effective_scanout_pitch(1919, 1920), 1920);
        // A zero width cannot overflow the 4x bound.
        assert_eq!(effective_scanout_pitch(0, 0), 0);
        assert_eq!(effective_scanout_pitch(u32::MAX, 0), 0);
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
        // The guest-driven vblank advance under test only happens while no
        // host vblank source owns the sequence; pin that against a concurrent
        // ownership test (crate::host_vblank).
        let _vblank_owner = crate::host_vblank::OwnershipGuard::released();
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
            "sceVideoOutDeleteFlipEvent",
            "sceVideoOutDeleteVblankEvent",
            "sceVideoOutAddPreVblankStartEvent",
            "sceVideoOutDeletePreVblankStartEvent",
            "sceVideoOutAddOutputModeEvent",
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

    /// `sceVideoOutDeleteFlipEvent` is the exact mirror of AddFlipEvent: same
    /// handle/equeue validation, and the registration Add inserted is gone
    /// afterwards — a later flip no longer triggers anything on that queue.
    ///
    /// This is the measured gap: Blasphemous II (PPSA13580) imports this NID
    /// and our loader reported it unresolved.
    #[test]
    fn delete_flip_event_undoes_add_flip_event() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = kernel.create_equeue(0);

        assert_eq!(hle_add_flip_event(&ctx, &[eq, 1, 0xCAFE]), SCE_OK);
        assert!(
            kernel
                .kernel_equeue_events
                .contains_key(&(eq, VIDEO_OUT_EVENT_FLIP_ID))
        );

        // Validation mirrors the Add path exactly, and rejects *before*
        // removing anything.
        assert_eq!(
            hle_delete_flip_event(&ctx, &[eq, 9]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
        assert_eq!(
            hle_delete_flip_event(&ctx, &[eq + 0x1000, 1]),
            VIDEO_OUT_ERROR_INVALID_OPTION
        );
        assert!(
            kernel
                .kernel_equeue_events
                .contains_key(&(eq, VIDEO_OUT_EVENT_FLIP_ID)),
            "a rejected delete must not drop the registration"
        );

        assert_eq!(hle_delete_flip_event(&ctx, &[eq, 1]), SCE_OK);
        assert!(
            !kernel
                .kernel_equeue_events
                .contains_key(&(eq, VIDEO_OUT_EVENT_FLIP_ID))
        );
        // Deleting a registration that is not there is ENOENT, not success.
        assert_eq!(
            hle_delete_flip_event(&ctx, &[eq, 1]),
            SCE_KERNEL_ERROR_ENOENT
        );

        // And the delete actually took effect where it matters: a completed
        // flip no longer resurrects or triggers the event.
        assert_eq!(submit_flip_from_agc(&ctx, 1, 0, 1, 0x1234), SCE_OK);
        assert!(
            !kernel
                .kernel_equeue_events
                .contains_key(&(eq, VIDEO_OUT_EVENT_FLIP_ID))
        );
    }

    /// The vblank mirror, checked the same way — including that the vblank tick
    /// (which fires from WaitVblank and from every flip) no longer reaches the
    /// deleted registration.
    #[test]
    fn delete_vblank_event_undoes_add_vblank_event() {
        // The guest-driven vblank advance under test only happens while no
        // host vblank source owns the sequence; pin that against a concurrent
        // ownership test (crate::host_vblank).
        let _vblank_owner = crate::host_vblank::OwnershipGuard::released();
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = kernel.create_equeue(0);

        assert_eq!(hle_add_vblank_event(&ctx, &[eq, 1, 0xBEEF]), SCE_OK);
        assert_eq!(
            hle_delete_vblank_event(&ctx, &[eq, 9]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
        assert_eq!(hle_delete_vblank_event(&ctx, &[eq, 1]), SCE_OK);
        assert_eq!(
            hle_delete_vblank_event(&ctx, &[eq, 1]),
            SCE_KERNEL_ERROR_ENOENT
        );

        assert_eq!(hle_wait_vblank(&ctx, &[1]), SCE_OK);
        assert!(
            !kernel
                .kernel_equeue_events
                .contains_key(&(eq, VIDEO_OUT_EVENT_VBLANK_ID)),
            "a vblank tick must not resurrect a deleted registration"
        );
    }

    /// Pre-vblank-start registers, is **delivered** by the vblank tick (Raeen
    /// has one tick point where KytyPS5 has two), classifies as public id 2,
    /// and deletes.
    #[test]
    fn pre_vblank_start_events_register_fire_and_delete() {
        // The guest-driven vblank advance under test only happens while no
        // host vblank source owns the sequence; pin that against a concurrent
        // ownership test (crate::host_vblank).
        let _vblank_owner = crate::host_vblank::OwnershipGuard::released();
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = kernel.create_equeue(0);

        assert_eq!(
            hle_add_pre_vblank_start_event(&ctx, &[eq, 9, 0]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
        assert_eq!(
            hle_add_pre_vblank_start_event(&ctx, &[eq, 1, 0xF00D]),
            SCE_OK
        );
        {
            let event = kernel
                .kernel_equeue_events
                .get(&(eq, VIDEO_OUT_EVENT_PRE_VBLANK_START_ID))
                .unwrap();
            assert!(!event.triggered, "not triggered until a refresh");
            assert_eq!(event.filter, KERNEL_EVENT_FILTER_VIDEO_OUT);
            assert_eq!(event.udata, 0xF00D);
        }

        // The ident is distinct from flip and vblank — the three classes are
        // separate registrations on one queue, not one key overwriting another.
        assert_ne!(
            VIDEO_OUT_EVENT_PRE_VBLANK_START_ID,
            VIDEO_OUT_EVENT_VBLANK_ID
        );
        assert_ne!(VIDEO_OUT_EVENT_PRE_VBLANK_START_ID, VIDEO_OUT_EVENT_FLIP_ID);
        assert_eq!(hle_add_vblank_event(&ctx, &[eq, 1, 0xBEEF]), SCE_OK);

        // One vblank tick delivers both classes, each carrying its own ident
        // with the shared sequence number above it.
        assert_eq!(hle_wait_vblank(&ctx, &[1]), SCE_OK);
        {
            let pre = kernel
                .kernel_equeue_events
                .get(&(eq, VIDEO_OUT_EVENT_PRE_VBLANK_START_ID))
                .unwrap();
            assert!(pre.triggered, "we must deliver, not merely acknowledge");
            assert_eq!(
                pre.data as u64,
                VIDEO_OUT_EVENT_PRE_VBLANK_START_ID | (1 << 16)
            );
            assert_eq!(pre.udata, 0xF00D);
            let vblank = kernel
                .kernel_equeue_events
                .get(&(eq, VIDEO_OUT_EVENT_VBLANK_ID))
                .unwrap();
            assert!(vblank.triggered);
            assert_eq!(vblank.data as u64, VIDEO_OUT_EVENT_VBLANK_ID | (1 << 16));
            assert_eq!(vblank.udata, 0xBEEF);
        }

        // GetEventId classifies the delivered event as the public id 2.
        let mut event_struct = [0u8; 0x20];
        event_struct[0..8].copy_from_slice(&VIDEO_OUT_EVENT_PRE_VBLANK_START_ID.to_le_bytes());
        event_struct[8..10].copy_from_slice(&KERNEL_EVENT_FILTER_VIDEO_OUT.to_le_bytes());
        assert!(mem.write(0x300, &event_struct));
        assert_eq!(
            hle_get_event_id(&ctx, &[0x300]),
            2,
            "pre-vblank-start event → 2"
        );

        assert_eq!(hle_delete_pre_vblank_start_event(&ctx, &[eq, 1]), SCE_OK);
        assert!(
            !kernel
                .kernel_equeue_events
                .contains_key(&(eq, VIDEO_OUT_EVENT_PRE_VBLANK_START_ID))
        );
        assert_eq!(
            hle_delete_pre_vblank_start_event(&ctx, &[eq, 1]),
            SCE_KERNEL_ERROR_ENOENT
        );
        // Deleting pre-vblank leaves the vblank registration alone.
        assert!(
            kernel
                .kernel_equeue_events
                .contains_key(&(eq, VIDEO_OUT_EVENT_VBLANK_ID))
        );
    }

    /// An output-mode event registers **already triggered** with the current
    /// mode as its payload (KytyPS5 `RegisterVideoOutEvent`,
    /// videoOut.cpp:366-376). Raeen never changes output mode, so this initial
    /// delivery is the only one a title will get — if it were not pending at
    /// registration, an output-mode-driven init would park forever.
    #[test]
    fn output_mode_event_registers_already_triggered_with_the_current_mode() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = kernel.create_equeue(0);

        assert_eq!(
            hle_add_output_mode_event(&ctx, &[eq, 9, 0]),
            VIDEO_OUT_ERROR_INVALID_HANDLE
        );
        assert_eq!(hle_add_output_mode_event(&ctx, &[eq, 1, 0xABBA]), SCE_OK);
        let event = kernel
            .kernel_equeue_events
            .get(&(eq, VIDEO_OUT_EVENT_OUTPUT_MODE_ID))
            .unwrap();
        assert!(event.triggered, "registered pending, per KytyPS5");
        assert_eq!(event.fflags, 1);
        assert_eq!(event.filter, KERNEL_EVENT_FILTER_VIDEO_OUT);
        assert_eq!(event.udata, 0xABBA);
        assert_eq!(
            event.data as u64,
            VIDEO_OUT_EVENT_OUTPUT_MODE_ID | (OUTPUT_MODE_DEFAULT << 16),
            "payload is the current output mode, in the shared data layout"
        );
        drop(event);

        // A vblank tick must not disturb it — output-mode is its own class.
        assert_eq!(hle_wait_vblank(&ctx, &[1]), SCE_OK);
        assert_eq!(
            kernel
                .kernel_equeue_events
                .get(&(eq, VIDEO_OUT_EVENT_OUTPUT_MODE_ID))
                .unwrap()
                .data as u64,
            VIDEO_OUT_EVENT_OUTPUT_MODE_ID | (OUTPUT_MODE_DEFAULT << 16)
        );

        let mut event_struct = [0u8; 0x20];
        event_struct[0..8].copy_from_slice(&VIDEO_OUT_EVENT_OUTPUT_MODE_ID.to_le_bytes());
        event_struct[8..10].copy_from_slice(&KERNEL_EVENT_FILTER_VIDEO_OUT.to_le_bytes());
        assert!(mem.write(0x300, &event_struct));
        assert_eq!(hle_get_event_id(&ctx, &[0x300]), 8, "output-mode event → 8");
    }

    /// The Add path for the classes that are *not* pre-triggered must stay
    /// un-triggered, so a title cannot mistake registration for a first
    /// delivery. Guards the `initial_trigger` plumbing added for output-mode.
    #[test]
    fn only_output_mode_registers_pre_triggered() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = kernel.create_equeue(0);

        for (add, ident) in [
            (
                hle_add_flip_event as fn(&HleContext, &[u64]) -> u64,
                VIDEO_OUT_EVENT_FLIP_ID,
            ),
            (hle_add_vblank_event, VIDEO_OUT_EVENT_VBLANK_ID),
            (
                hle_add_pre_vblank_start_event,
                VIDEO_OUT_EVENT_PRE_VBLANK_START_ID,
            ),
        ] {
            assert_eq!(add(&ctx, &[eq, 1, 0]), SCE_OK);
            let event = kernel.kernel_equeue_events.get(&(eq, ident)).unwrap();
            assert!(!event.triggered, "ident {ident:#x} must register idle");
            assert_eq!(event.fflags, 0);
            assert_eq!(event.data, 0);
        }
    }
}
