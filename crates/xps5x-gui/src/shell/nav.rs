//! Focus model + input mapping (spec §3, §9).
//!
//! Pure state machine: `(NavState, NavInput) -> (NavState, NavAction)`. No
//! egui, no I/O — this is what the table-driven navigation tests exercise.
//! `shell/mod.rs` is responsible for translating keyboard/gamepad events
//! into [`NavInput`] and for acting on the returned [`NavAction`].

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
}

/// Which surface currently owns navigation input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavMode {
    Home,
    ControlCenter,
}

/// The Shell's full navigation state: which surface is active, and the
/// focus index within each of the rail and the Control Center row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NavState {
    pub mode: NavMode,
    pub rail_index: usize,
    pub rail_len: usize,
    pub cc_index: usize,
    pub cc_len: usize,
}

impl NavState {
    pub fn new(rail_len: usize, cc_len: usize) -> Self {
        Self { mode: NavMode::Home, rail_index: 0, rail_len, cc_index: 0, cc_len }
    }

    /// Apply one input, mutating focus in place and returning the resulting
    /// action (if any).
    pub fn apply(&mut self, input: NavInput) -> NavAction {
        match self.mode {
            NavMode::Home => self.apply_home(input),
            NavMode::ControlCenter => self.apply_control_center(input),
        }
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
            NavInput::Confirm => NavAction::None,
            NavInput::Guide | NavInput::Back => {
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
}
