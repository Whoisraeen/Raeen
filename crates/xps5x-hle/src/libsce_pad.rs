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
/// The state written is currently the neutral default (controller connected,
/// no buttons, sticks centered): live host input is not yet routed through
/// `HleContext` to the `InputManager`, so actual button presses are the
/// follow-up. But a guest now gets a well-formed, non-garbage state and its
/// read loop makes progress.
fn hle_pad_read_state(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let data = args.get(1).copied().unwrap_or(0);
    debug!("scePadReadState(handle={handle}, data={data:#x})");

    let state = xps5x_input::ControllerState::default().to_orbis_pad_data();
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
    use crate::{test_ctx, GuestMemory};

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

    #[test]
    fn pad_read_state_unwritable_buffer_returns_zero() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_pad_read_state(&ctx, &[1, 0xDEAD_0000]), 0);
    }
}
