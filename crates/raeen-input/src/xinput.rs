//! Xbox / XInput-compatible controller reader (Windows).
//!
//! Clean-room port of SharpEmu's `WindowsXInputReader.cs` (GPL-2.0-or-later,
//! © SharpEmu Emulator Project). Mapping-DB-free: the fixed `XINPUT_GAMEPAD`
//! layout is translated directly, so Steam-Input / DS4Windows / generic-HID
//! pads — which gilrs rejects with an all-zeros UUID ("No mapping found for
//! UUID 00000000-…") — still produce input. Steam surfaces most pads as an
//! XInput device, so this path alone recovers the dead-button case.
//!
//! The pure [`translate`] function (below) carries the whole mapping and is
//! unit-tested without any device; the Windows FFI in `imp` only feeds it.

use crate::ControllerState;

// XINPUT_GAMEPAD wButtons bit values (SharpEmu parity / XInput headers).
const XINPUT_DPAD_UP: u16 = 0x0001;
const XINPUT_DPAD_DOWN: u16 = 0x0002;
const XINPUT_DPAD_LEFT: u16 = 0x0004;
const XINPUT_DPAD_RIGHT: u16 = 0x0008;
const XINPUT_START: u16 = 0x0010;
const XINPUT_BACK: u16 = 0x0020;
const XINPUT_LEFT_THUMB: u16 = 0x0040;
const XINPUT_RIGHT_THUMB: u16 = 0x0080;
const XINPUT_LEFT_SHOULDER: u16 = 0x0100;
const XINPUT_RIGHT_SHOULDER: u16 = 0x0200;
const XINPUT_A: u16 = 0x1000;
const XINPUT_B: u16 = 0x2000;
const XINPUT_X: u16 = 0x4000;
const XINPUT_Y: u16 = 0x8000;

/// The fixed `XINPUT_GAMEPAD` payload, decoupled from the FFI struct so the
/// translation stays a pure, device-free function.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct XPadRaw {
    pub buttons: u16,
    pub left_trigger: u8,
    pub right_trigger: u8,
    pub thumb_lx: i16,
    pub thumb_ly: i16,
    pub thumb_rx: i16,
    pub thumb_ry: i16,
}

/// Map a signed 16-bit thumbstick axis to this crate's `-1.0..=1.0` encoding.
fn axis(v: i16) -> f32 {
    (v as f32 / 32768.0).clamp(-1.0, 1.0)
}

/// Translate a raw `XINPUT_GAMEPAD` into a [`ControllerState`].
///
/// Buttons: A→Cross, B→Circle, X→Square, Y→Triangle, LeftShoulder→L1,
/// RightShoulder→R1, thumbs→L3/R3, Start→Options, Back→TouchPad, D-pad
/// pass-through. Triggers become analog L2/R2 (`0..=1`); the pipeline derives
/// the digital L2/R2 bit from that value. Stick X passes through; stick Y is
/// inverted because XInput's Y grows upward while this crate (like the Orbis
/// encoding) puts "up" at the low byte.
#[must_use]
pub fn translate(pad: &XPadRaw) -> ControllerState {
    let b = pad.buttons;
    ControllerState {
        cross: b & XINPUT_A != 0,
        circle: b & XINPUT_B != 0,
        square: b & XINPUT_X != 0,
        triangle: b & XINPUT_Y != 0,
        l1: b & XINPUT_LEFT_SHOULDER != 0,
        r1: b & XINPUT_RIGHT_SHOULDER != 0,
        l3: b & XINPUT_LEFT_THUMB != 0,
        r3: b & XINPUT_RIGHT_THUMB != 0,
        options: b & XINPUT_START != 0,
        touchpad_click: b & XINPUT_BACK != 0,
        dpad_up: b & XINPUT_DPAD_UP != 0,
        dpad_down: b & XINPUT_DPAD_DOWN != 0,
        dpad_left: b & XINPUT_DPAD_LEFT != 0,
        dpad_right: b & XINPUT_DPAD_RIGHT != 0,
        left_stick_x: axis(pad.thumb_lx),
        left_stick_y: -axis(pad.thumb_ly),
        right_stick_x: axis(pad.thumb_rx),
        right_stick_y: -axis(pad.thumb_ry),
        l2_trigger: pad.left_trigger as f32 / 255.0,
        r2_trigger: pad.right_trigger as f32 / 255.0,
        ..Default::default()
    }
}

#[cfg(windows)]
mod imp {
    use super::{translate, ControllerState, XPadRaw};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use windows_sys::Win32::UI::Input::XboxController::{XInputGetState, XINPUT_STATE};

    const ERROR_SUCCESS: u32 = 0;
    const SLOT_COUNT: u32 = 4;

    /// Latest snapshot of the first connected XInput slot, or `None`.
    pub type Shared = Arc<Mutex<Option<ControllerState>>>;

    /// Spawn the background reader thread and return the shared cache. The
    /// thread idle-polls the four XInput slots and never needs joining.
    #[must_use]
    pub fn spawn() -> Shared {
        let shared: Shared = Arc::new(Mutex::new(None));
        let worker = shared.clone();
        let _ = std::thread::Builder::new()
            .name("xinput-reader".into())
            .spawn(move || read_loop(&worker));
        shared
    }

    fn poll_slot(slot: u32) -> Option<XPadRaw> {
        // SAFETY: XINPUT_STATE is a plain-integer POD; we hand XInputGetState a
        // valid zeroed out-param and only read `Gamepad` on ERROR_SUCCESS.
        let mut state: XINPUT_STATE = unsafe { std::mem::zeroed() };
        let rc = unsafe { XInputGetState(slot, &mut state) };
        if rc != ERROR_SUCCESS {
            return None;
        }
        let g = state.Gamepad;
        Some(XPadRaw {
            buttons: g.wButtons,
            left_trigger: g.bLeftTrigger,
            right_trigger: g.bRightTrigger,
            thumb_lx: g.sThumbLX,
            thumb_ly: g.sThumbLY,
            thumb_rx: g.sThumbRX,
            thumb_ry: g.sThumbRY,
        })
    }

    fn find_slot() -> Option<u32> {
        (0..SLOT_COUNT).find(|&slot| poll_slot(slot).is_some())
    }

    fn read_loop(shared: &Shared) {
        loop {
            let Some(slot) = find_slot() else {
                *shared.lock().unwrap() = None;
                std::thread::sleep(Duration::from_millis(1000));
                continue;
            };
            tracing::info!(slot, "XInput (Xbox-compatible) controller connected");
            while let Some(raw) = poll_slot(slot) {
                *shared.lock().unwrap() = Some(translate(&raw));
                std::thread::sleep(Duration::from_millis(8));
            }
            tracing::info!(slot, "XInput controller disconnected");
            *shared.lock().unwrap() = None;
            std::thread::sleep(Duration::from_millis(1000));
        }
    }
}

#[cfg(windows)]
pub use imp::{spawn, Shared};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_buttons_map_to_playstation_layout() {
        let s = translate(&XPadRaw {
            buttons: XINPUT_A | XINPUT_B | XINPUT_X | XINPUT_Y,
            ..Default::default()
        });
        assert!(s.cross, "A -> Cross");
        assert!(s.circle, "B -> Circle");
        assert!(s.square, "X -> Square");
        assert!(s.triangle, "Y -> Triangle");
    }

    #[test]
    fn shoulders_thumbs_menu_and_dpad_map() {
        let s = translate(&XPadRaw {
            buttons: XINPUT_LEFT_SHOULDER
                | XINPUT_RIGHT_SHOULDER
                | XINPUT_LEFT_THUMB
                | XINPUT_RIGHT_THUMB
                | XINPUT_START
                | XINPUT_BACK
                | XINPUT_DPAD_UP
                | XINPUT_DPAD_RIGHT,
            ..Default::default()
        });
        assert!(s.l1 && s.r1, "shoulders -> L1/R1");
        assert!(s.l3 && s.r3, "thumbs -> L3/R3");
        assert!(s.options, "Start -> Options");
        assert!(s.touchpad_click, "Back -> TouchPad");
        assert!(s.dpad_up && s.dpad_right, "d-pad passes through");
        assert!(!s.dpad_down && !s.dpad_left, "unpressed d-pad stays clear");
    }

    #[test]
    fn triggers_become_analog_and_cross_the_digital_threshold() {
        let s = translate(&XPadRaw {
            left_trigger: 255,
            right_trigger: 0,
            ..Default::default()
        });
        assert_eq!(s.l2_trigger, 1.0, "full left trigger -> 1.0");
        assert_eq!(s.r2_trigger, 0.0, "released right trigger -> 0.0");
        // The pipeline's orbis encoding sets the digital L2 bit at >0.5.
        assert_eq!(s.orbis_buttons() & crate::pad_button::L2, crate::pad_button::L2);
        assert_eq!(s.orbis_buttons() & crate::pad_button::R2, 0);
    }

    #[test]
    fn sticks_scale_and_invert_y_only() {
        // Full right + full up on the left stick.
        let s = translate(&XPadRaw {
            thumb_lx: 32767,
            thumb_ly: 32767,
            thumb_rx: -32768,
            thumb_ry: -32768,
            ..Default::default()
        });
        assert!((s.left_stick_x - 1.0).abs() < 1e-3, "X right -> +1");
        // XInput up is +32767; inverted to -1 so the Orbis low byte means up.
        assert!((s.left_stick_y + 1.0).abs() < 1e-3, "Y up -> -1 (inverted)");
        assert!((s.right_stick_x + 1.0).abs() < 1e-3, "X left -> -1");
        assert!((s.right_stick_y - 1.0).abs() < 1e-3, "Y down -> +1 (inverted)");

        // Centered stick round-trips to the Orbis center byte (128).
        let center = translate(&XPadRaw::default()).to_orbis_pad_data();
        assert_eq!(center[4], 128, "centered left stick x -> 128");
        assert_eq!(center[5], 128, "centered left stick y -> 128");
    }

    #[test]
    fn neutral_pad_presses_nothing() {
        assert_eq!(translate(&XPadRaw::default()).orbis_buttons(), 0);
    }
}
