//! Deterministic controller-state replay for title acceptance runs.
//!
//! The isolated runner normally publishes only live native controller input.
//! Nightly compatibility runs also need repeatable button sequences, so
//! `RAEEN_INPUT_SCRIPT` can supply semicolon-separated state snapshots:
//!
//! ```text
//! 0:neutral;180000:cross;180250:neutral;185000:cross+options
//! ```
//!
//! Timestamps are milliseconds since the runner's input thread started. Each
//! state remains active until the next snapshot. The feature is opt-in and
//! does not alter the physical-controller path when the variable is absent.

use crate::ControllerState;
use std::time::Duration;

#[derive(Debug, Clone)]
struct ScriptEvent {
    at: Duration,
    state: ControllerState,
}

/// Parsed deterministic controller-state timeline.
#[derive(Debug, Clone)]
pub struct InputScript {
    events: Vec<ScriptEvent>,
}

impl InputScript {
    /// Parse the compact `milliseconds:button+button;...` replay format.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut events = Vec::new();
        let mut previous_ms = None;
        for (index, raw_event) in spec.split(';').enumerate() {
            let raw_event = raw_event.trim();
            if raw_event.is_empty() {
                continue;
            }
            let (at, buttons) = raw_event.split_once(':').ok_or_else(|| {
                format!("input event {} must be '<milliseconds>:<state>'", index + 1)
            })?;
            let at_ms = at.trim().parse::<u64>().map_err(|_| {
                format!(
                    "input event {} has invalid millisecond timestamp '{}'",
                    index + 1,
                    at.trim()
                )
            })?;
            if previous_ms.is_some_and(|previous| at_ms < previous) {
                return Err(format!(
                    "input event {} timestamp {at_ms} precedes the previous event",
                    index + 1
                ));
            }
            previous_ms = Some(at_ms);
            events.push(ScriptEvent {
                at: Duration::from_millis(at_ms),
                state: parse_state(buttons.trim(), index + 1)?,
            });
        }
        if events.is_empty() {
            return Err("input script contains no events".to_string());
        }
        Ok(Self { events })
    }

    /// State active at `elapsed`, or `None` before the first scripted event.
    #[must_use]
    pub fn state_at(&self, elapsed: Duration) -> Option<ControllerState> {
        let end = self.events.partition_point(|event| event.at <= elapsed);
        end.checked_sub(1)
            .map(|index| self.events[index].state.clone())
    }

    /// Number of timeline snapshots, for diagnostics.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the parsed timeline is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

fn parse_state(buttons: &str, event_index: usize) -> Result<ControllerState, String> {
    let mut state = ControllerState::default();
    if buttons.eq_ignore_ascii_case("neutral") || buttons.eq_ignore_ascii_case("none") {
        return Ok(state);
    }
    if buttons.is_empty() {
        return Err(format!("input event {event_index} has an empty state"));
    }
    for raw_button in buttons.split('+') {
        let button = raw_button.trim().to_ascii_lowercase();
        match button.as_str() {
            "cross" | "x" => state.cross = true,
            "circle" => state.circle = true,
            "square" => state.square = true,
            "triangle" => state.triangle = true,
            "options" | "start" => state.options = true,
            "create" | "share" => state.create = true,
            "l1" => state.l1 = true,
            "r1" => state.r1 = true,
            "l2" => state.l2_trigger = 1.0,
            "r2" => state.r2_trigger = 1.0,
            "l3" => state.l3 = true,
            "r3" => state.r3 = true,
            "up" | "dpad_up" => state.dpad_up = true,
            "down" | "dpad_down" => state.dpad_down = true,
            "left" | "dpad_left" => state.dpad_left = true,
            "right" | "dpad_right" => state.dpad_right = true,
            "touchpad" | "touch_pad" => state.touchpad_click = true,
            _ => {
                return Err(format!(
                    "input event {event_index} names unknown button '{raw_button}'"
                ));
            }
        }
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pad_button;

    #[test]
    fn timeline_holds_each_snapshot_until_the_next() {
        let script = InputScript::parse("100:cross;250:neutral;400:up+r2").expect("valid script");
        assert!(script.state_at(Duration::from_millis(99)).is_none());
        assert_eq!(
            script
                .state_at(Duration::from_millis(100))
                .unwrap()
                .orbis_buttons(),
            pad_button::CROSS
        );
        assert_eq!(
            script
                .state_at(Duration::from_millis(399))
                .unwrap()
                .orbis_buttons(),
            0
        );
        assert_eq!(
            script
                .state_at(Duration::from_secs(5))
                .unwrap()
                .orbis_buttons(),
            pad_button::UP | pad_button::R2
        );
        assert_eq!(script.len(), 3);
        assert!(!script.is_empty());
    }

    #[test]
    fn aliases_and_chords_map_to_orbis_buttons() {
        let script = InputScript::parse("0:x+start+touch_pad").unwrap();
        assert_eq!(
            script.state_at(Duration::ZERO).unwrap().orbis_buttons(),
            pad_button::CROSS | pad_button::OPTIONS | pad_button::TOUCH_PAD
        );
    }

    #[test]
    fn malformed_timelines_fail_closed() {
        assert!(InputScript::parse("").is_err());
        assert!(InputScript::parse("cross").is_err());
        assert!(InputScript::parse("later:cross").is_err());
        assert!(InputScript::parse("10:cross;9:neutral").is_err());
        assert!(InputScript::parse("10:home").is_err());
        assert!(InputScript::parse("10:").is_err());
    }
}
