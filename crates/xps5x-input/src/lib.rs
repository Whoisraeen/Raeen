//! # XPS5X Input
//!
//! DualSense controller emulation and generic gamepad support.
//! Provides haptic feedback, adaptive trigger translation, and
//! fallback to XInput/SDL for non-DualSense controllers.

pub mod adaptive_triggers;
pub mod dualsense;
pub mod haptics;

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
