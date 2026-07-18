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
/// [`xps5x_input::ControllerState::to_orbis_pad_data`]) into the guest
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
        .unwrap_or_else(|| xps5x_input::ControllerState::default().to_orbis_pad_data());
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
        let kernel = xps5x_kernel::OrbisKernel::new();
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
        let kernel = xps5x_kernel::OrbisKernel::new();
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

    #[test]
    fn pad_read_state_unwritable_buffer_returns_zero() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_pad_read_state(&ctx, &[1, 0xDEAD_0000]), 0);
    }
}
