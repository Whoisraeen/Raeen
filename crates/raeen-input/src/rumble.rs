//! Guest → host vibration (rumble) plumbing shared by the HLE, the isolated
//! runner, and the Shell.
//!
//! `scePadSetVibration` stores the title's motor request on the session
//! kernel (`OrbisKernel::set_pad_rumble`). From there it travels as a single
//! encoded `u64` *rumble word* — either read directly (in-process session) or
//! published through the frame-IPC header (isolated runner child → Shell).
//! The Shell feeds the word into a [`RumbleRouter`] each frame, which decides
//! what the physical controller motors should do:
//!
//! * routes the newest guest request to hardware when Settings ▸ Controllers ▸
//!   DualSense Features is ON, and forces silence when it is OFF;
//! * de-duplicates writes so an idle guest costs zero hardware output reports;
//! * safety auto-stop: if the guest never refreshes a non-zero vibration for
//!   [`AUTO_STOP_AFTER`], the motors are stopped. Real firmware persists an
//!   output report indefinitely (shadPS4 mirrors that by passing an unbounded
//!   duration to `SDL_RumbleGamepad`), but titles refresh vibration far more
//!   often than 5 s, and a killed guest must never leave motors stuck;
//! * silences the motors the moment the session (the rumble source) is gone.

use std::time::Duration;

/// Requested motor intensities, `0` = off, `255` = full — exactly the Orbis
/// `ScePadVibrationParam` fields (largeMotor = left/low-frequency motor,
/// smallMotor = right/high-frequency motor; layout cross-checked against
/// shadPS4 `OrbisPadVibrationParam` and SharpEmu `PadSetVibration`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RumbleState {
    pub large: u8,
    pub small: u8,
}

impl RumbleState {
    /// Both motors off.
    pub const SILENT: Self = Self { large: 0, small: 0 };

    #[must_use]
    pub fn new(large: u8, small: u8) -> Self {
        Self { large, small }
    }

    #[must_use]
    pub fn is_silent(&self) -> bool {
        *self == Self::SILENT
    }
}

/// Pack a rumble request into the single-`u64` wire format used by the
/// kernel and the frame-IPC header: bits 0..8 = small motor, 8..16 = large
/// motor, 16..64 = sequence. The sequence increments on **every**
/// `scePadSetVibration` call (even with unchanged values), so a title
/// re-asserting its vibration refreshes the router's auto-stop deadline the
/// same way it refreshes real hardware.
#[must_use]
pub fn encode_word(seq: u64, state: RumbleState) -> u64 {
    (seq << 16) | ((state.large as u64) << 8) | state.small as u64
}

/// Unpack a rumble word. `None` when the sequence is 0 — the "no title ever
/// called `scePadSetVibration`" initial state, indistinguishable from an
/// absent publisher, so both decode to "no rumble source".
#[must_use]
pub fn decode_word(word: u64) -> Option<(u64, RumbleState)> {
    let seq = word >> 16;
    if seq == 0 {
        return None;
    }
    Some((seq, RumbleState::new((word >> 8) as u8, word as u8)))
}

/// Stop the motors after this long without the guest refreshing a non-zero
/// vibration. See the module docs for why this exists despite real firmware
/// persisting output reports indefinitely.
pub const AUTO_STOP_AFTER: Duration = Duration::from_secs(5);

/// Per-frame rumble arbiter between the guest's requests and the physical
/// controller. Pure state machine over caller-supplied timestamps
/// (`Duration` since any fixed epoch), so every rule is unit-testable
/// without hardware or sleeping.
#[derive(Debug, Default)]
pub struct RumbleRouter {
    /// Sequence of the newest guest request consumed.
    last_seq: Option<u64>,
    /// When that request (or its refresh) was observed.
    last_refresh: Option<Duration>,
    /// What the guest currently wants the motors to do.
    target: RumbleState,
    /// What the hardware was last told to do; `None` until the first update
    /// so the initial command (even "silent") is always emitted, clearing any
    /// stale motor state left by a previous process.
    applied: Option<RumbleState>,
}

impl RumbleRouter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume the newest rumble source snapshot and return the command to
    /// send to the physical controller, or `None` when hardware already
    /// matches. `source` is `None` when there is no running session (or the
    /// guest has never vibrated) — the motors are then silenced. `enabled`
    /// is the Settings ▸ DualSense Features toggle: OFF drops vibration at
    /// this gate while the guest-side state keeps flowing, so flipping it
    /// back ON mid-session picks the current vibration right up.
    pub fn update(
        &mut self,
        now: Duration,
        source: Option<(u64, RumbleState)>,
        enabled: bool,
    ) -> Option<RumbleState> {
        match source {
            Some((seq, state)) => {
                if self.last_seq != Some(seq) {
                    self.last_seq = Some(seq);
                    self.last_refresh = Some(now);
                    self.target = state;
                }
            }
            None => {
                // Session gone (or never vibrated): decay immediately.
                self.last_seq = None;
                self.last_refresh = None;
                self.target = RumbleState::SILENT;
            }
        }
        // Safety auto-stop: a non-zero vibration the guest stopped refreshing.
        if !self.target.is_silent()
            && self
                .last_refresh
                .is_some_and(|at| now.saturating_sub(at) >= AUTO_STOP_AFTER)
        {
            self.target = RumbleState::SILENT;
        }
        let desired = if enabled {
            self.target
        } else {
            RumbleState::SILENT
        };
        if self.applied == Some(desired) {
            return None;
        }
        self.applied = Some(desired);
        Some(desired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    #[test]
    fn word_round_trips_and_seq_zero_means_no_source() {
        let state = RumbleState::new(200, 17);
        let word = encode_word(9, state);
        assert_eq!(decode_word(word), Some((9, state)));
        assert_eq!(decode_word(0), None, "initial kernel state is no-source");
        assert_eq!(
            decode_word(encode_word(0, RumbleState::new(255, 255))),
            None,
            "seq 0 is reserved for never-set regardless of motor bits"
        );
    }

    #[test]
    fn first_update_clears_stale_hardware_then_dedupes() {
        let mut router = RumbleRouter::new();
        // First frame ever: emit silent once (clear stale motors)…
        assert_eq!(
            router.update(secs(0), None, true),
            Some(RumbleState::SILENT)
        );
        // …then nothing while nothing changes.
        assert_eq!(router.update(secs(1), None, true), None);
    }

    #[test]
    fn guest_request_reaches_hardware_once_and_stops_when_session_ends() {
        let mut router = RumbleRouter::new();
        router.update(secs(0), None, true);
        let on = RumbleState::new(255, 128);
        assert_eq!(router.update(secs(1), Some((1, on)), true), Some(on));
        // Same seq re-observed: no duplicate hardware write.
        assert_eq!(router.update(secs(2), Some((1, on)), true), None);
        // Session ends → immediate silence (no stuck motors).
        assert_eq!(
            router.update(secs(3), None, true),
            Some(RumbleState::SILENT)
        );
    }

    #[test]
    fn settings_toggle_gates_hardware_but_not_guest_state() {
        let mut router = RumbleRouter::new();
        let on = RumbleState::new(90, 0);
        // Disabled: guest request is dropped at the hardware gate. The very
        // first update emits the silent baseline once.
        assert_eq!(
            router.update(secs(0), Some((1, on)), false),
            Some(RumbleState::SILENT)
        );
        assert_eq!(router.update(secs(1), Some((1, on)), false), None);
        // Re-enabling mid-session picks the current vibration right up.
        assert_eq!(router.update(secs(2), Some((1, on)), true), Some(on));
        // Disabling again silences immediately.
        assert_eq!(
            router.update(secs(3), Some((1, on)), false),
            Some(RumbleState::SILENT)
        );
    }

    #[test]
    fn unrefreshed_vibration_auto_stops_after_five_seconds() {
        let mut router = RumbleRouter::new();
        let on = RumbleState::new(255, 255);
        router.update(secs(0), None, true);
        assert_eq!(router.update(secs(1), Some((1, on)), true), Some(on));
        // 4.9s after the request: still rumbling.
        assert_eq!(
            router.update(secs(1) + Duration::from_millis(4900), Some((1, on)), true),
            None
        );
        // 5s without a refresh: safety stop.
        assert_eq!(
            router.update(secs(6), Some((1, on)), true),
            Some(RumbleState::SILENT)
        );
        // A fresh guest call (new seq, same values) restarts the vibration —
        // exactly how a title keep-alive refreshes real hardware.
        assert_eq!(router.update(secs(7), Some((2, on)), true), Some(on));
        assert_eq!(router.update(secs(11), Some((2, on)), true), None);
        assert_eq!(
            router.update(secs(12), Some((2, on)), true),
            Some(RumbleState::SILENT)
        );
    }

    #[test]
    fn guest_clearing_vibration_stops_without_waiting_for_auto_stop() {
        let mut router = RumbleRouter::new();
        router.update(secs(0), None, true);
        let on = RumbleState::new(10, 20);
        assert_eq!(router.update(secs(1), Some((1, on)), true), Some(on));
        assert_eq!(
            router.update(secs(2), Some((2, RumbleState::SILENT)), true),
            Some(RumbleState::SILENT)
        );
        // A silent target never re-triggers the auto-stop path.
        assert_eq!(
            router.update(secs(30), Some((2, RumbleState::SILENT)), true),
            None
        );
    }
}
