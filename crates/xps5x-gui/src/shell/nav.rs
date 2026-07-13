//! Focus model + input mapping (spec §3, §9, §10).
//!
//! Pure state machine: `(NavState, NavInput) -> (NavState, NavAction)`. No
//! egui, no I/O — this is what the table-driven navigation tests exercise.
//! `shell/mod.rs` is responsible for translating keyboard/gamepad events
//! into [`NavInput`] and for acting on the returned [`NavAction`].
//!
//! SM1 adds a third mode, [`NavMode::ControlCenterOption`], for Control
//! Center cards that expose a selectable option list (e.g. Power: Rest
//! Mode / Restart / Turn Off). Confirming on a card with `option_count() >
//! 0` drills into that list; Confirming an option there yields
//! [`NavAction::ActivateOption`] and returns to the card view.

/// A single navigation input, already normalized from keyboard or gamepad.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavInput {
    Left,
    Right,
    Confirm,
    /// PS/Guide button (keyboard: `C`).
    Guide,
    Back,
}

/// A side effect the caller must perform in response to a [`NavInput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavAction {
    None,
    /// Launch the rail item at this index.
    Launch(usize),
    OpenControlCenter,
    CloseControlCenter,
    /// Confirm selected `option` within Control Center card `card`.
    ActivateOption { card: usize, option: usize },
}

/// Which surface currently owns navigation input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavMode {
    Home,
    ControlCenter,
    /// Drilled into the focused Control Center card's own option list.
    ControlCenterOption,
}

/// The Shell's full navigation state: which surface is active, and the
/// focus index within each of the rail, the Control Center row, and (when
/// drilled in) the focused card's option list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NavState {
    pub mode: NavMode,
    pub rail_index: usize,
    pub rail_len: usize,
    pub cc_index: usize,
    pub cc_len: usize,
    pub cc_option_index: usize,
    /// Number of selectable options each Control Center card exposes; `0`
    /// for cards that are display-only. Indexed in parallel with the
    /// Control Center row (`cc_index`).
    cc_option_counts: Vec<usize>,
}

impl NavState {
    /// Construct with every Control Center card display-only (no options).
    /// Kept as a convenience constructor for tests and any future caller
    /// with an all-display-only Control Center row; the Shell itself uses
    /// [`NavState::with_cc_options`] to wire in Power's option count.
    #[allow(dead_code)]
    pub fn new(rail_len: usize, cc_len: usize) -> Self {
        Self::with_cc_options(rail_len, cc_len, vec![0; cc_len])
    }

    /// Construct with an explicit per-card option count, in the same order
    /// as the Control Center row.
    pub fn with_cc_options(rail_len: usize, cc_len: usize, cc_option_counts: Vec<usize>) -> Self {
        Self { mode: NavMode::Home, rail_index: 0, rail_len, cc_index: 0, cc_len, cc_option_index: 0, cc_option_counts }
    }

    /// Apply one input, mutating focus in place and returning the resulting
    /// action (if any).
    pub fn apply(&mut self, input: NavInput) -> NavAction {
        match self.mode {
            NavMode::Home => self.apply_home(input),
            NavMode::ControlCenter => self.apply_control_center(input),
            NavMode::ControlCenterOption => self.apply_control_center_option(input),
        }
    }

    fn focused_option_count(&self) -> usize {
        self.cc_option_counts.get(self.cc_index).copied().unwrap_or(0)
    }

    fn apply_home(&mut self, input: NavInput) -> NavAction {
        match input {
            NavInput::Left => {
                self.rail_index = self.rail_index.saturating_sub(1);
                NavAction::None
            }
            NavInput::Right => {
                if self.rail_len > 0 {
                    self.rail_index = (self.rail_index + 1).min(self.rail_len - 1);
                }
                NavAction::None
            }
            NavInput::Confirm => NavAction::Launch(self.rail_index),
            NavInput::Guide => {
                self.mode = NavMode::ControlCenter;
                self.cc_index = 0;
                NavAction::OpenControlCenter
            }
            NavInput::Back => NavAction::None,
        }
    }

    fn apply_control_center(&mut self, input: NavInput) -> NavAction {
        match input {
            NavInput::Left => {
                self.cc_index = self.cc_index.saturating_sub(1);
                NavAction::None
            }
            NavInput::Right => {
                if self.cc_len > 0 {
                    self.cc_index = (self.cc_index + 1).min(self.cc_len - 1);
                }
                NavAction::None
            }
            NavInput::Confirm => {
                if self.focused_option_count() > 0 {
                    self.mode = NavMode::ControlCenterOption;
                    self.cc_option_index = 0;
                }
                NavAction::None
            }
            NavInput::Guide | NavInput::Back => {
                self.mode = NavMode::Home;
                NavAction::CloseControlCenter
            }
        }
    }

    fn apply_control_center_option(&mut self, input: NavInput) -> NavAction {
        let count = self.focused_option_count();
        match input {
            NavInput::Left => {
                self.cc_option_index = self.cc_option_index.saturating_sub(1);
                NavAction::None
            }
            NavInput::Right => {
                if count > 0 {
                    self.cc_option_index = (self.cc_option_index + 1).min(count - 1);
                }
                NavAction::None
            }
            NavInput::Confirm => {
                let action = NavAction::ActivateOption { card: self.cc_index, option: self.cc_option_index };
                self.mode = NavMode::ControlCenter;
                action
            }
            NavInput::Back => {
                self.mode = NavMode::ControlCenter;
                NavAction::None
            }
            NavInput::Guide => {
                self.mode = NavMode::Home;
                NavAction::CloseControlCenter
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table-driven: (starting state, input) -> (expected rail/cc index,
    /// expected mode, expected action).
    #[test]
    fn home_navigation_table() {
        struct Case {
            name: &'static str,
            start_index: usize,
            rail_len: usize,
            input: NavInput,
            expect_index: usize,
            expect_action: NavAction,
        }

        let cases = [
            Case { name: "right moves forward", start_index: 0, rail_len: 5, input: NavInput::Right, expect_index: 1, expect_action: NavAction::None },
            Case { name: "left clamps at zero", start_index: 0, rail_len: 5, input: NavInput::Left, expect_index: 0, expect_action: NavAction::None },
            Case { name: "right clamps at the end", start_index: 4, rail_len: 5, input: NavInput::Right, expect_index: 4, expect_action: NavAction::None },
            Case { name: "left moves backward", start_index: 3, rail_len: 5, input: NavInput::Left, expect_index: 2, expect_action: NavAction::None },
            Case { name: "confirm launches focused index", start_index: 2, rail_len: 5, input: NavInput::Confirm, expect_index: 2, expect_action: NavAction::Launch(2) },
        ];

        for case in cases {
            let mut nav = NavState::new(case.rail_len, 11);
            nav.rail_index = case.start_index;
            let action = nav.apply(case.input);
            assert_eq!(action, case.expect_action, "case: {}", case.name);
            assert_eq!(nav.rail_index, case.expect_index, "case: {}", case.name);
            assert_eq!(nav.mode, NavMode::Home, "case: {}", case.name);
        }
    }

    #[test]
    fn guide_opens_control_center_and_resets_its_focus() {
        let mut nav = NavState::new(9, 11);
        nav.rail_index = 5;
        nav.cc_index = 7; // stale from a previous session
        let action = nav.apply(NavInput::Guide);
        assert_eq!(action, NavAction::OpenControlCenter);
        assert_eq!(nav.mode, NavMode::ControlCenter);
        assert_eq!(nav.cc_index, 0);
        // Home focus is preserved for when we return.
        assert_eq!(nav.rail_index, 5);
    }

    #[test]
    fn control_center_navigation_table() {
        struct Case {
            name: &'static str,
            start_index: usize,
            cc_len: usize,
            input: NavInput,
            expect_index: usize,
            expect_action: NavAction,
            expect_mode: NavMode,
        }

        let cases = [
            Case { name: "right moves forward", start_index: 0, cc_len: 11, input: NavInput::Right, expect_index: 1, expect_action: NavAction::None, expect_mode: NavMode::ControlCenter },
            Case { name: "left clamps at zero", start_index: 0, cc_len: 11, input: NavInput::Left, expect_index: 0, expect_action: NavAction::None, expect_mode: NavMode::ControlCenter },
            Case { name: "right clamps at the end", start_index: 10, cc_len: 11, input: NavInput::Right, expect_index: 10, expect_action: NavAction::None, expect_mode: NavMode::ControlCenter },
            Case { name: "escape (Back) closes", start_index: 4, cc_len: 11, input: NavInput::Back, expect_index: 4, expect_action: NavAction::CloseControlCenter, expect_mode: NavMode::Home },
            Case { name: "guide toggles closed", start_index: 4, cc_len: 11, input: NavInput::Guide, expect_index: 4, expect_action: NavAction::CloseControlCenter, expect_mode: NavMode::Home },
        ];

        for case in cases {
            let mut nav = NavState::new(9, case.cc_len);
            nav.mode = NavMode::ControlCenter;
            nav.cc_index = case.start_index;
            let action = nav.apply(case.input);
            assert_eq!(action, case.expect_action, "case: {}", case.name);
            assert_eq!(nav.cc_index, case.expect_index, "case: {}", case.name);
            assert_eq!(nav.mode, case.expect_mode, "case: {}", case.name);
        }
    }

    #[test]
    fn empty_rail_does_not_panic_on_right() {
        let mut nav = NavState::new(0, 0);
        assert_eq!(nav.apply(NavInput::Right), NavAction::None);
        assert_eq!(nav.rail_index, 0);
    }

    #[test]
    fn confirm_on_a_display_only_card_does_not_drill_in() {
        let mut nav = NavState::with_cc_options(9, 3, vec![0, 0, 3]);
        nav.mode = NavMode::ControlCenter;
        nav.cc_index = 0;
        let action = nav.apply(NavInput::Confirm);
        assert_eq!(action, NavAction::None);
        assert_eq!(nav.mode, NavMode::ControlCenter);
    }

    /// Table-driven: the Control Center's option-drilldown transitions
    /// (e.g. Power's Rest Mode / Restart / Turn Off list).
    #[test]
    fn control_center_option_navigation_table() {
        struct Case {
            name: &'static str,
            start_mode: NavMode,
            start_option_index: usize,
            input: NavInput,
            expect_mode: NavMode,
            expect_option_index: usize,
            expect_action: NavAction,
        }

        // Card index 2 has 3 options (mirrors Power in `control_center::ITEMS`).
        let option_counts = vec![0, 0, 3];
        let power_card = 2;

        let cases = [
            Case { name: "confirm on the options card drills in", start_mode: NavMode::ControlCenter, start_option_index: 0, input: NavInput::Confirm, expect_mode: NavMode::ControlCenterOption, expect_option_index: 0, expect_action: NavAction::None },
            Case { name: "right moves within options", start_mode: NavMode::ControlCenterOption, start_option_index: 0, input: NavInput::Right, expect_mode: NavMode::ControlCenterOption, expect_option_index: 1, expect_action: NavAction::None },
            Case { name: "right clamps at the last option", start_mode: NavMode::ControlCenterOption, start_option_index: 2, input: NavInput::Right, expect_mode: NavMode::ControlCenterOption, expect_option_index: 2, expect_action: NavAction::None },
            Case { name: "left clamps at zero", start_mode: NavMode::ControlCenterOption, start_option_index: 0, input: NavInput::Left, expect_mode: NavMode::ControlCenterOption, expect_option_index: 0, expect_action: NavAction::None },
            Case { name: "confirm activates the selected option and returns to the card", start_mode: NavMode::ControlCenterOption, start_option_index: 1, input: NavInput::Confirm, expect_mode: NavMode::ControlCenter, expect_option_index: 1, expect_action: NavAction::ActivateOption { card: power_card, option: 1 } },
            Case { name: "back leaves option mode without activating", start_mode: NavMode::ControlCenterOption, start_option_index: 1, input: NavInput::Back, expect_mode: NavMode::ControlCenter, expect_option_index: 1, expect_action: NavAction::None },
            Case { name: "guide closes control center entirely from option mode", start_mode: NavMode::ControlCenterOption, start_option_index: 1, input: NavInput::Guide, expect_mode: NavMode::Home, expect_option_index: 1, expect_action: NavAction::CloseControlCenter },
        ];

        for case in cases {
            let mut nav = NavState::with_cc_options(9, option_counts.len(), option_counts.clone());
            nav.mode = case.start_mode;
            nav.cc_index = power_card;
            nav.cc_option_index = case.start_option_index;
            let action = nav.apply(case.input);
            assert_eq!(action, case.expect_action, "case: {}", case.name);
            assert_eq!(nav.mode, case.expect_mode, "case: {}", case.name);
            assert_eq!(nav.cc_option_index, case.expect_option_index, "case: {}", case.name);
        }
    }
}
