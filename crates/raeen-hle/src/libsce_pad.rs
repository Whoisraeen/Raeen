//! HLE libScePad — Controller input interface.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// Register libScePad HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libScePad", "scePadInit", hle_pad_init);
    registry.register("libScePad", "scePadOpen", hle_pad_open);
    registry.register("libScePad", "scePadReadState", hle_pad_read_state);
    registry.register("libScePad", "scePadRead", hle_pad_read_state);
    registry.register("libScePad", "scePadSetVibration", hle_pad_set_vibration);
    // Motion/tilt/feature setters the title configures at HID init. It asserts
    // on a non-OK return (ASTRO.BOT PsPadPpr.cpp:108 on scePadSetTiltCorrectionState
    // returning garbage), so acknowledge them — we model no motion hardware, but
    // "configured OK" lets the pad setup proceed.
    for f in [
        "scePadSetTiltCorrectionState",
        "scePadSetAngularVelocityDeadbandState",
        "scePadSetMotionSensorState",
        "scePadSetVibrationMode",
        "scePadResetOrientation",
        "scePadResetLightBar",
        "scePadSetLightBar",
        "scePadSetVolumeGain",
        "scePadSetProcessPrivilege",
        "scePadDeviceClassParseData",
        "scePadSetButtonRemappingInfo",
    ] {
        registry.register("libScePad", f, hle_pad_ok);
    }
    registry.register("libScePad", "scePadClose", hle_pad_close);
    registry.register(
        "libScePad",
        "scePadGetControllerInformation",
        hle_pad_get_controller_information,
    );
    registry.register(
        "libScePad",
        "scePadSetTriggerEffect",
        hle_pad_set_trigger_effect,
    );
    registry.register(
        "libScePad",
        "scePadGetTriggerEffectState",
        hle_pad_get_trigger_effect_state,
    );
}

/// `SCE_PAD_ERROR_INVALID_HANDLE` (SharpEmu `OrbisPadErrorInvalidHandle`).
const PAD_ERROR_INVALID_HANDLE: u64 = 0x8092_0003;
/// `SCE_PAD_ERROR_INVALID_ARG`.
const PAD_ERROR_INVALID_ARG: u64 = 0x8092_0001;

/// The one pad handle `scePadOpen` hands out.
const PRIMARY_PAD_HANDLE: u64 = 1;

/// `scePadClose(handle)`: release the handle. The handle model is a single
/// always-open primary pad (`scePadOpen` returns 1 unconditionally), so close
/// validates the handle and succeeds without tearing anything down — exactly
/// SharpEmu `PadExports.PadClose` (NID `6ncge5+l5Qs`): primary handle → 0,
/// anything else → invalid-handle.
fn hle_pad_close(_ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    debug!("scePadClose(handle={handle})");
    if handle == PRIMARY_PAD_HANDLE {
        0
    } else {
        PAD_ERROR_INVALID_HANDLE
    }
}

/// `scePadGetControllerInformation(handle, ScePadControllerInformation
/// *info)`: report a connected standard (DualSense) controller. The 0x1C-byte
/// layout and values are SharpEmu `PadExports.PadGetControllerInformation`
/// (NID `gjP9-KQzoUk`): touchpad density 44.86, 1920x943 touch resolution,
/// stick deadzones 30/30, port type standard (0), connectedCount 1,
/// connected 1, deviceClass 0 at +0x10.
fn hle_pad_get_controller_information(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let info_ptr = args.get(1).copied().unwrap_or(0);
    debug!("scePadGetControllerInformation(handle={handle}, info={info_ptr:#x})");
    if handle != PRIMARY_PAD_HANDLE {
        return PAD_ERROR_INVALID_HANDLE;
    }
    if info_ptr == 0 {
        return PAD_ERROR_INVALID_ARG;
    }
    let mut info = [0u8; 0x1C];
    info[0x00..0x04].copy_from_slice(&44.86f32.to_le_bytes()); // touchpad pixel density
    info[0x04..0x06].copy_from_slice(&1920u16.to_le_bytes()); // touchpad width
    info[0x06..0x08].copy_from_slice(&943u16.to_le_bytes()); // touchpad height
    info[0x08] = 30; // left stick deadzone
    info[0x09] = 30; // right stick deadzone
    info[0x0A] = 0; // connection type: standard/local
    info[0x0B] = 1; // connected count
    info[0x0C] = 1; // connected
    info[0x10..0x14].copy_from_slice(&0i32.to_le_bytes()); // deviceClass: standard
    if !ctx.mem.write(info_ptr, &info) {
        warn!("scePadGetControllerInformation: info out-ptr {info_ptr:#x} not writable");
        return PAD_ERROR_INVALID_ARG;
    }
    0
}

/// `scePadSetTriggerEffect(handle, const ScePadTriggerEffectParam *param)`:
/// accept the adaptive-trigger command. SharpEmu decodes the 120-byte param
/// (trigger mask + two 56-byte per-trigger commands) into host rumble; Raeen
/// has no trigger-rumble backend yet, so the command is validated (handle +
/// non-NULL, readable param) and acknowledged — the title's haptics loop only
/// checks the return code.
fn hle_pad_set_trigger_effect(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let param_ptr = args.get(1).copied().unwrap_or(0);
    if handle != PRIMARY_PAD_HANDLE {
        return PAD_ERROR_INVALID_HANDLE;
    }
    if param_ptr == 0 {
        return PAD_ERROR_INVALID_ARG;
    }
    let mut mask = [0u8; 1];
    if !ctx.mem.read(param_ptr, &mut mask) {
        return PAD_ERROR_INVALID_ARG;
    }
    debug!(
        "scePadSetTriggerEffect(handle={handle}, triggerMask={:#x}) -> accepted",
        mask[0]
    );
    0
}

/// `scePadGetTriggerEffectState(handle, state)`: report both adaptive
/// triggers as idle (no effect active). No public reference implements this
/// (PS5-only; absent from SharpEmu and shadPS4), so the out-buffer is treated
/// as the minimal per-trigger state pair — 2 bytes, one per trigger, 0 =
/// off/idle — which is consistent with never having accepted an effect into a
/// host backend in [`hle_pad_set_trigger_effect`].
fn hle_pad_get_trigger_effect_state(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let state_ptr = args.get(1).copied().unwrap_or(0);
    if handle != PRIMARY_PAD_HANDLE {
        return PAD_ERROR_INVALID_HANDLE;
    }
    if state_ptr == 0 {
        return PAD_ERROR_INVALID_ARG;
    }
    if !ctx.mem.write(state_ptr, &[0u8; 2]) {
        return PAD_ERROR_INVALID_ARG;
    }
    0
}

/// A libScePad configuration setter with no modeled hardware effect that the
/// title only checks the SCE-OK return of.
fn hle_pad_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0
}

fn hle_pad_init(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("scePadInit()");
    0
}

fn hle_pad_open(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "scePadOpen(userId={}, type={}, index={})",
        args[0], args[1], args[2]
    );
    1 // Return pad handle = 1.
}

/// Real `scePadReadState(handle, ScePadData *data)` (M3 input): writes a
/// valid Orbis `ScePadData` input prefix (buttons + sticks + triggers, see
/// [`raeen_input::ControllerState::to_orbis_pad_data`]) into the guest
/// buffer and returns `1` (one state read).
///
/// Two real fixes over the old stub: (1) it writes a **valid** struct
/// (previously nothing — the guest read uninitialized memory), and (2) it
/// returns `1`, not `0` — a homebrew input loop reads state until the return
/// is positive, so the old `0` made it spin/hang forever.
///
/// The state written is the host's current controller snapshot
/// (`ctx.kernel.pad_state()`, pushed each frame by the Shell) when live input
/// is available, else a neutral default (controller connected, no buttons,
/// sticks centered) so a guest polling before any input still gets a
/// well-formed, non-garbage state and its read loop makes progress.
fn hle_pad_read_state(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let data = args.get(1).copied().unwrap_or(0);
    debug!("scePadReadState(handle={handle}, data={data:#x})");

    let state = ctx
        .kernel
        .pad_state()
        .unwrap_or_else(|| raeen_input::ControllerState::default().to_orbis_pad_data());
    if data != 0 && !ctx.mem.write(data, &state) {
        warn!("scePadReadState: ScePadData out-buffer {data:#x} not writable");
        return 0; // no state read
    }
    1 // one ScePadData written
}

fn hle_pad_set_vibration(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("scePadSetVibration(handle={}, ...)", args[0]);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    /// M3: scePadReadState writes a valid, non-garbage ScePadData prefix into
    /// the guest buffer and returns 1 (one state read) so a homebrew read
    /// loop makes progress instead of spinning on the old `0`.
    #[test]
    fn pad_read_state_writes_valid_neutral_state_and_returns_one() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Pre-poison the buffer so we can tell real bytes were written.
        assert!(mem.write(0x100, &[0xEE; 12]));
        let ret = hle_pad_read_state(&ctx, &[1, 0x100]);
        assert_eq!(ret, 1, "one ScePadData state must be reported read");

        let mut buf = [0u8; 12];
        assert!(mem.read(0x100, &mut buf));
        // Neutral: buttons == 0, sticks centered at 128.
        assert_eq!(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]), 0);
        assert_eq!(buf[4], 128);
        assert_eq!(buf[7], 128);
    }

    /// Live input pushed onto the kernel by the host flows through
    /// scePadReadState to the guest buffer (the DualSense routing path).
    #[test]
    fn pad_read_state_reflects_host_pushed_live_state() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Host pushes a state with the Cross button held (bit 0x4000) and the
        // left stick pushed right (byte 4 = 255).
        let mut live = [0u8; 12];
        live[0..4].copy_from_slice(&0x0000_4000u32.to_le_bytes());
        live[4] = 255;
        live[5] = 128;
        live[6] = 128;
        live[7] = 128;
        kernel.set_pad_state(live);

        assert_eq!(hle_pad_read_state(&ctx, &[1, 0x100]), 1);
        let mut buf = [0u8; 12];
        assert!(mem.read(0x100, &mut buf));
        assert_eq!(buf, live, "guest must read the host-pushed live pad state");
    }

    /// The DualSense info/close/trigger family validates the primary handle
    /// and reports a connected standard controller (SharpEmu PadExports
    /// layout: connectedCount@0x0B == 1, connected@0x0C == 1).
    #[test]
    fn controller_information_close_and_trigger_effects() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_pad_get_controller_information(&ctx, &[1, 0x100]), 0);
        let mut info = [0u8; 0x1C];
        assert!(mem.read(0x100, &mut info));
        assert_eq!(
            f32::from_le_bytes(info[0..4].try_into().unwrap()),
            44.86,
            "touchpad density"
        );
        assert_eq!(info[0x0B], 1, "connectedCount");
        assert_eq!(info[0x0C], 1, "connected");
        assert_eq!(
            hle_pad_get_controller_information(&ctx, &[2, 0x100]),
            PAD_ERROR_INVALID_HANDLE
        );
        assert_eq!(
            hle_pad_get_controller_information(&ctx, &[1, 0]),
            PAD_ERROR_INVALID_ARG
        );

        // Trigger effect: accepted on the primary handle, then reported idle.
        assert!(mem.write(0x200, &[0x03u8]));
        assert_eq!(hle_pad_set_trigger_effect(&ctx, &[1, 0x200]), 0);
        assert_eq!(hle_pad_get_trigger_effect_state(&ctx, &[1, 0x300]), 0);
        let mut state = [0xFFu8; 2];
        assert!(mem.read(0x300, &mut state));
        assert_eq!(state, [0, 0], "both triggers idle");

        assert_eq!(hle_pad_close(&ctx, &[1]), 0);
        assert_eq!(hle_pad_close(&ctx, &[5]), PAD_ERROR_INVALID_HANDLE);
    }

    #[test]
    fn pad_read_state_unwritable_buffer_returns_zero() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_pad_read_state(&ctx, &[1, 0xDEAD_0000]), 0);
    }
}
