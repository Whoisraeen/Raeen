//! Native, mapping-DB-free gamepad readers, merged into the Shell's pad
//! pipeline ahead of gilrs and the keyboard fallback.
//!
//! Two backends run on their own background threads (SharpEmu design): the
//! XInput reader ([`crate::xinput`]) for Xbox / Steam-Input / generic-HID
//! pads, and the raw-HID reader ([`crate::hid`]) for DualSense / DualShock 4.
//! Both cache
//! their latest snapshot behind a `Mutex`; [`NativeGamepads::poll`] returns one
//! per frame, native Sony HID preferred (DualSense is the real PS5 pad).
//!
//! On non-Windows targets the whole thing degrades to a no-op so the crate
//! still builds; the native readers are Windows-only today.

use crate::ControllerState;

/// Owns the background native-controller readers and hands the Shell the
/// latest snapshot each frame.
#[cfg(windows)]
pub struct NativeGamepads {
    xinput: crate::xinput::XInputPads,
    sony_hid: crate::hid::DualSense,
}

#[cfg(windows)]
impl NativeGamepads {
    /// Start the background XInput + DualSense-HID readers (and the DualSense
    /// rumble writer). Cheap and idempotent-friendly to hold once for the
    /// app's lifetime.
    #[must_use]
    pub fn start() -> Self {
        Self {
            xinput: crate::xinput::spawn(),
            sony_hid: crate::hid::spawn(),
        }
    }

    /// Latest native snapshot — Sony HID preferred over XInput — or `None`
    /// when no native controller is connected. Sticks are raw (`-1..=1`); the
    /// caller applies its configured deadzone.
    #[must_use]
    pub fn poll(&self) -> Option<ControllerState> {
        if let Some(sony) = self.sony_hid.input.lock().ok().and_then(|g| g.clone()) {
            return Some(sony);
        }
        self.xinput.input.lock().ok().and_then(|g| g.clone())
    }

    /// Route a rumble command (Orbis `0..=255` motor bytes; large =
    /// low-frequency/strong, small = high-frequency/weak) to whatever native
    /// controller is connected: the DualSense gets a HID output report, a
    /// DualShock 4 remains input-only, and an XInput pad gets
    /// `XInputSetState`. All sinks no-op when their device is absent, so this
    /// is safe to call unconditionally.
    pub fn set_rumble(&self, large: u8, small: u8) {
        self.sony_hid.set_rumble(large, small);
        self.xinput.set_rumble(large, small);
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

    pub fn set_rumble(&self, _large: u8, _small: u8) {}
}
