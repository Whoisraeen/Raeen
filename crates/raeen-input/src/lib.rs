//! # Raeen Input
//!
//! DualSense controller emulation and generic gamepad support.
//! Provides haptic feedback, adaptive trigger translation, and
//! fallback to XInput/SDL for non-DualSense controllers.

pub mod adaptive_triggers;
pub mod dualsense;
pub mod haptics;
pub mod hid;
pub mod native;
pub mod scripted;
pub mod xinput;

pub use native::NativeGamepads;
pub use scripted::InputScript;

use tracing::info;

/// Controller state — represents the current input from a connected controller.
#[derive(Debug, Clone, Default)]
pub struct ControllerState {
    // ─── Buttons ───────────────────────────────────
    pub cross: bool,    // ✕
    pub circle: bool,   // ○
    pub square: bool,   // □
    pub triangle: bool, // △
    pub l1: bool,
    pub r1: bool,
    pub l3: bool, // Left stick click
    pub r3: bool, // Right stick click
    pub options: bool,
    pub create: bool, // Share/Create button
    pub ps_button: bool,
    pub touchpad_click: bool,
    pub dpad_up: bool,
    pub dpad_down: bool,
    pub dpad_left: bool,
    pub dpad_right: bool,

    // ─── Analog ────────────────────────────────────
    pub left_stick_x: f32, // -1.0 to 1.0
    pub left_stick_y: f32,
    pub right_stick_x: f32,
    pub right_stick_y: f32,
    pub l2_trigger: f32, // 0.0 to 1.0
    pub r2_trigger: f32,

    // ─── Motion ────────────────────────────────────
    pub gyro_x: f32,
    pub gyro_y: f32,
    pub gyro_z: f32,
    pub accel_x: f32,
    pub accel_y: f32,
    pub accel_z: f32,

    // ─── Touchpad ──────────────────────────────────
    pub touch1_active: bool,
    pub touch1_x: f32,
    pub touch1_y: f32,
    pub touch2_active: bool,
    pub touch2_x: f32,
    pub touch2_y: f32,
}

/// Orbis `ScePadButtonDataOffset` masks (the documented libScePad button
/// bits — cross-checked against SharpEmu's DualSense reader: UP=0x10,
/// RIGHT=0x20, DOWN=0x40, LEFT=0x80). A homebrew ANDs `ScePadData::buttons`
/// against these to test each digital button.
pub mod pad_button {
    pub const L3: u32 = 0x0000_0002;
    pub const R3: u32 = 0x0000_0004;
    pub const OPTIONS: u32 = 0x0000_0008;
    pub const UP: u32 = 0x0000_0010;
    pub const RIGHT: u32 = 0x0000_0020;
    pub const DOWN: u32 = 0x0000_0040;
    pub const LEFT: u32 = 0x0000_0080;
    pub const L2: u32 = 0x0000_0100;
    pub const R2: u32 = 0x0000_0200;
    pub const L1: u32 = 0x0000_0400;
    pub const R1: u32 = 0x0000_0800;
    pub const TRIANGLE: u32 = 0x0000_1000;
    pub const CIRCLE: u32 = 0x0000_2000;
    pub const CROSS: u32 = 0x0000_4000;
    pub const SQUARE: u32 = 0x0000_8000;
    pub const TOUCH_PAD: u32 = 0x0010_0000;
}

/// Size in bytes of the `ScePadData` input prefix [`ControllerState::to_orbis_pad_data`]
/// produces: `buttons` (u32) + `leftStick`/`rightStick` (2×u8 each) +
/// `analogButtons` L2/R2 (u8 each) + 2 padding bytes. This is the stable,
/// universally-read leading region of the Orbis `ScePadData` struct; the
/// extended motion/touch/timestamp fields that follow are not populated
/// here (they'd need their exact per-SDK offsets).
pub const ORBIS_PAD_DATA_PREFIX_LEN: usize = 12;

/// Total size in bytes of a complete Orbis/Prospero `ScePadData` struct — the
/// full buffer `scePadReadState` fills. Verified against shadPS4 `pad.h`,
/// KytyPS5 `padData.h` (`static_assert(sizeof == 120)`), and OpenOrbis. The
/// `connected` flag lives at `0x4C` and `connectedCount` at `0x68`, well past
/// the 12-byte input prefix — so `hle_pad_read_state` must write the WHOLE
/// struct or the guest reads `connected` as garbage and drops all input.
pub const ORBIS_PAD_DATA_LEN: usize = 120;

/// Map an analog stick axis (`-1.0..=1.0`, this crate's convention) to the
/// Orbis `u8` encoding (`0`=min, `128`=center, `255`=max).
fn stick_to_u8(v: f32) -> u8 {
    // Map [-1, 1] onto [0, 255] with center 128: -1 -> 0, 0 -> 128, 1 -> 255.
    ((v.clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Map an analog trigger (`0.0..=1.0`) to the Orbis `u8` encoding (`0..=255`).
fn trigger_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8
}

impl ControllerState {
    /// The packed `buttons` bitfield for this state, using the [`pad_button`]
    /// masks — what a guest reads at `ScePadData::buttons`.
    #[must_use]
    pub fn orbis_buttons(&self) -> u32 {
        use pad_button as b;
        let mut out = 0u32;
        for (pressed, mask) in [
            (self.cross, b::CROSS),
            (self.circle, b::CIRCLE),
            (self.square, b::SQUARE),
            (self.triangle, b::TRIANGLE),
            (self.l1, b::L1),
            (self.r1, b::R1),
            (self.l3, b::L3),
            (self.r3, b::R3),
            (self.options, b::OPTIONS),
            (self.dpad_up, b::UP),
            (self.dpad_down, b::DOWN),
            (self.dpad_left, b::LEFT),
            (self.dpad_right, b::RIGHT),
            (self.touchpad_click, b::TOUCH_PAD),
            (self.l2_trigger > 0.5, b::L2),
            (self.r2_trigger > 0.5, b::R2),
        ] {
            if pressed {
                out |= mask;
            }
        }
        out
    }

    /// Encode this state as the leading [`ORBIS_PAD_DATA_PREFIX_LEN`] bytes of
    /// an Orbis `ScePadData`: little-endian `buttons`, then
    /// `leftStick.{x,y}`, `rightStick.{x,y}`, `analogButtons.{l2,r2}`, and two
    /// padding bytes. A guest's `scePadReadState` reads exactly these fields
    /// for digital + analog input.
    #[must_use]
    pub fn to_orbis_pad_data(&self) -> [u8; ORBIS_PAD_DATA_PREFIX_LEN] {
        let mut d = [0u8; ORBIS_PAD_DATA_PREFIX_LEN];
        d[0..4].copy_from_slice(&self.orbis_buttons().to_le_bytes());
        d[4] = stick_to_u8(self.left_stick_x);
        d[5] = stick_to_u8(self.left_stick_y);
        d[6] = stick_to_u8(self.right_stick_x);
        d[7] = stick_to_u8(self.right_stick_y);
        d[8] = trigger_to_u8(self.l2_trigger);
        d[9] = trigger_to_u8(self.r2_trigger);
        // d[10..12] padding stays zero.
        d
    }
}

/// Input manager — handles controller enumeration and polling.
pub struct InputManager {
    /// Whether DualSense-specific features are enabled.
    pub dualsense_features: bool,
    /// Controller deadzone.
    pub deadzone: f32,
}

impl InputManager {
    pub fn new(dualsense_features: bool, deadzone: f32) -> Self {
        info!(
            "Input manager created (DualSense features={}, deadzone={:.2})",
            dualsense_features, deadzone
        );
        Self {
            dualsense_features,
            deadzone,
        }
    }

    /// Apply deadzone to an analog axis value.
    pub fn apply_deadzone(&self, value: f32) -> f32 {
        if value.abs() < self.deadzone {
            0.0
        } else {
            let sign = value.signum();
            let magnitude = (value.abs() - self.deadzone) / (1.0 - self.deadzone);
            sign * magnitude
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_neutral_buttons_zero_sticks_centered() {
        let d = ControllerState::default().to_orbis_pad_data();
        assert_eq!(
            u32::from_le_bytes([d[0], d[1], d[2], d[3]]),
            0,
            "no buttons"
        );
        assert_eq!(d[4], 128, "left stick x centered");
        assert_eq!(d[5], 128, "left stick y centered");
        assert_eq!(d[6], 128, "right stick x centered");
        assert_eq!(d[7], 128, "right stick y centered");
        assert_eq!(d[8], 0, "L2 released");
        assert_eq!(d[9], 0, "R2 released");
    }

    #[test]
    fn buttons_map_to_documented_orbis_masks() {
        let s = ControllerState {
            cross: true,
            dpad_up: true,
            options: true,
            ..Default::default()
        };
        let b = s.orbis_buttons();
        assert_eq!(b & pad_button::CROSS, pad_button::CROSS);
        assert_eq!(b & pad_button::UP, pad_button::UP);
        assert_eq!(b & pad_button::OPTIONS, pad_button::OPTIONS);
        assert_eq!(b & pad_button::CIRCLE, 0, "unpressed buttons stay clear");
    }

    #[test]
    fn analog_extremes_encode_to_byte_range() {
        let s = ControllerState {
            left_stick_x: -1.0,
            left_stick_y: 1.0,
            l2_trigger: 1.0,
            ..Default::default()
        };
        let d = s.to_orbis_pad_data();
        assert_eq!(d[4], 0, "-1.0 -> 0");
        assert_eq!(d[5], 255, "1.0 -> 255");
        assert_eq!(d[8], 255, "full L2 -> 255");
    }

    #[test]
    fn analog_trigger_over_half_sets_the_digital_l2_r2_bit() {
        let pressed = ControllerState {
            l2_trigger: 0.9,
            ..Default::default()
        };
        assert_eq!(pressed.orbis_buttons() & pad_button::L2, pad_button::L2);
        let released = ControllerState {
            l2_trigger: 0.1,
            ..Default::default()
        };
        assert_eq!(released.orbis_buttons() & pad_button::L2, 0);
    }
}
