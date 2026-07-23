//! Native, mapping-DB-free gamepad readers, merged into the Shell's pad
//! pipeline ahead of gilrs and the keyboard fallback.
//!
//! Two backends run on their own background threads (SharpEmu design): the
//! XInput reader ([`crate::xinput`]) for Xbox / Steam-Input / generic-HID
//! pads, and the raw-HID reader ([`crate::hid`]) for the DualSense. Both cache
//! their latest snapshot behind a `Mutex`; [`NativeGamepads::poll`] returns one
//! per frame, DualSense preferred (it is the real PS5 pad).
//!
//! On non-Windows targets the whole thing degrades to a no-op so the crate
//! still builds; the native readers are Windows-only today.

use crate::ControllerState;

/// Owns the background native-controller readers and hands the Shell the
/// latest snapshot each frame.
#[cfg(windows)]
pub struct NativeGamepads {
    xinput: crate::xinput::Shared,
    dualsense: crate::hid::Shared,
}

#[cfg(windows)]
impl NativeGamepads {
    /// Start the background XInput + DualSense-HID readers. Cheap and
    /// idempotent-friendly to hold once for the app's lifetime.
    #[must_use]
    pub fn start() -> Self {
        Self {
            xinput: crate::xinput::spawn(),
            dualsense: crate::hid::spawn(),
        }
    }

    /// Latest native snapshot — DualSense preferred over XInput — or `None`
    /// when no native controller is connected. Sticks are raw (`-1..=1`); the
    /// caller applies its configured deadzone.
    #[must_use]
    pub fn poll(&self) -> Option<ControllerState> {
        if let Some(ds) = self.dualsense.lock().ok().and_then(|g| g.clone()) {
            return Some(ds);
        }
        self.xinput.lock().ok().and_then(|g| g.clone())
    }
}

/// No-op stand-in on non-Windows targets (native readers are Windows-only).
#[cfg(not(windows))]
pub struct NativeGamepads;

#[cfg(not(windows))]
impl NativeGamepads {
    #[must_use]
    pub fn start() -> Self {
        Self
    }

    #[must_use]
    pub fn poll(&self) -> Option<ControllerState> {
        None
    }
}
