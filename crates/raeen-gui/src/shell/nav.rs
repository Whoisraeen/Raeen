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
//!
//! SM2a adds two more things (spec §10 SM2):
//! - A [`NavMode::Settings`] mode with its own two-dimensional focus
//!   (`settings_section` + `settings_row`), navigated with Up/Down (which
//!   also crosses section boundaries) and adjusted with Left/Right/Confirm.
//! - A [`RailTab`] (Games/Media) on the Home rail, switched with
//!   [`NavInput::Tab`]. Confirm behaves differently per tab: on Games,
//!   confirming the wired-up Settings tile opens Settings instead of
//!   launching; on Media it reports [`NavAction::LaunchMedia`] since there's
//!   no media-playback engine to hand off to yet.
//!
//! The concept-mock Home adds [`NavMode::Pills`]: Up from the rail moves
//! focus into the pill navigation row (Store / My games / Media / Library /
//! Settings / "…"), Left/Right move along it, Down/Back return to the rail,
//! and Confirm activates the focused pill — tab pills switch the rail tab,
//! Settings opens the Settings surface, and Store/Library jump the rail to
//! their app tiles.

/// A single navigation input, already normalized from keyboard or gamepad.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavInput {
    Left,
    Right,
    /// Settings mode only: move row focus up (crossing into the previous
    /// section at the top of the current one).
    Up,
    /// Settings mode only: move row focus down (crossing into the next
    /// section at the bottom of the current one).
    Down,
    Confirm,
    /// PS/Guide button (keyboard: `C`).
    Guide,
    Back,
    /// Switch the Home rail between Games and Media (keyboard `Tab`,
    /// gamepad L1/R1 — spec §10 SM2).
    Tab,
    /// Options/Triangle (keyboard `O`, gamepad North): on a focused game tile,
    /// opens that title's per-game settings overlay. Inspired by SharpEmu's
    /// per-game settings dialog, but reachable straight from the tile you are
    /// looking at rather than a right-click menu.
    Options,
}

/// A side effect the caller must perform in response to a [`NavInput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavAction {
    None,
    /// Launch the Games-rail item at this index.
    Launch(usize),
    /// Confirm on a Media-rail tile. There's no media-playback engine to
    /// hand off to yet, so the caller just logs/no-ops (spec §10 SM2).
    LaunchMedia(usize),
    OpenControlCenter,
    CloseControlCenter,
    /// Confirm selected `option` within Control Center card `card`.
    ActivateOption {
        card: usize,
        option: usize,
    },
    /// The Settings tile was confirmed on the Games rail.
    OpenSettings,
    /// Settings was left via Back — the caller should persist any changes.
    CloseSettings,
    /// The focused Settings row's value should step by `delta` (`-1`/`1`) —
    /// e.g. resolution scale, volume, deadzone, or the theme selector.
    /// `nav.rs` has no notion of what a row actually is; the caller maps
    /// `(section, row)` to a concrete field.
    AdjustSetting {
        section: usize,
        row: usize,
        delta: i32,
    },
    /// The focused Settings row was confirmed — a bool toggle, or a
    /// semantic action (add/remove a game folder) only the caller knows how
    /// to interpret.
    ActivateSetting {
        section: usize,
        row: usize,
    },
    /// The Home rail's active tab changed.
    SwitchTab(RailTab),
    /// Options was pressed on the game at this Games-rail index — open its
    /// per-game settings overlay.
    OpenGameOptions {
        index: usize,
    },
    /// The Game Options overlay was left via Back — the caller should persist
    /// the edited overrides.
    CloseGameOptions,
    /// The focused Game Options row's value should step by `delta` (`-1`/`1`).
    AdjustGameOption {
        row: usize,
        delta: i32,
    },
    /// The focused Game Options row was confirmed — toggle its override on/off
    /// (or, on the Reset row, clear every override).
    ActivateGameOption {
        row: usize,
    },
    /// Back was pressed at the top level (Home, nothing overlaid). The caller
    /// leaves fullscreen if it is in it, so the window chrome comes back and
    /// the app can be closed normally; in a window it is the exit gesture.
    LeaveShell,
}

/// Which Home rail tab is active (spec §10 SM2: Games shows the library,
/// Media shows the built-in media apps). Both tabs share the same rail
/// rendering and focus model — only the backing item list differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RailTab {
    #[default]
    Games,
    Media,
}

impl RailTab {
    fn toggled(self) -> Self {
        match self {
            RailTab::Games => RailTab::Media,
            RailTab::Media => RailTab::Games,
        }
    }
}

// Pill-row focus indices, in display order (the leading icon-only pill is
// decorative and not focusable). `home.rs` renders labels in this same
// order — the two must agree. Only functional destinations get a pill
// (PS5-authentic: no dead Store/Library/"…" chrome).
pub const PILL_MY_GAMES: usize = 0;
pub const PILL_MEDIA: usize = 1;
pub const PILL_SETTINGS: usize = 2;
pub const PILL_COUNT: usize = 3;

/// Which surface currently owns navigation input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavMode {
    Home,
    /// Focus is in Home's pill navigation row (reached with Up from the
    /// rail); Home still renders underneath, only input routing differs.
    Pills,
    ControlCenter,
    /// Drilled into the focused Control Center card's own option list.
    ControlCenterOption,
    /// Full-screen Settings surface (spec §10 SM2).
    Settings,
    /// Per-game settings overlay for the focused title (opened with Options
    /// from Home). Home still renders underneath; only input routing differs.
    GameOptions,
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

    /// Which Home rail is active.
    pub tab: RailTab,
    games_rail_len: usize,
    media_rail_len: usize,

    /// Focused pill while `mode == Pills` (one of the `PILL_*` indices).
    pub pill_index: usize,

    /// Games-rail index that opens Settings on Confirm instead of
    /// launching (the "Settings" app tile). `None` if the caller never
    /// wired one up (e.g. a bare test fixture with no Settings tile).
    settings_tile_index: Option<usize>,
    pub settings_section: usize,
    pub settings_row: usize,
    /// Number of rows in each Settings section, in display order.
    /// `settings_row_counts.len()` is the section count.
    settings_row_counts: Vec<usize>,

    /// Focused row within the per-game Game Options overlay.
    pub game_options_row: usize,
    /// Number of rows the Game Options overlay exposes.
    game_options_row_count: usize,
}

impl NavState {
    /// Construct with every Control Center card display-only (no options)
    /// and no Settings/Media wiring. Kept as a convenience constructor for
    /// tests and any future caller that only needs the Home/Control Center
    /// graph; the Shell itself chains [`NavState::with_settings`] and
    /// [`NavState::with_media_rail_len`] onto [`NavState::with_cc_options`].
    #[allow(dead_code)]
    pub fn new(rail_len: usize, cc_len: usize) -> Self {
        Self::with_cc_options(rail_len, cc_len, vec![0; cc_len])
    }

    /// Construct with an explicit per-card option count, in the same order
    /// as the Control Center row.
    pub fn with_cc_options(rail_len: usize, cc_len: usize, cc_option_counts: Vec<usize>) -> Self {
        Self {
            mode: NavMode::Home,
            rail_index: 0,
            rail_len,
            cc_index: 0,
            cc_len,
            cc_option_index: 0,
            cc_option_counts,
            tab: RailTab::Games,
            games_rail_len: rail_len,
            media_rail_len: 0,
            pill_index: PILL_MY_GAMES,
            settings_tile_index: None,
            settings_section: 0,
            settings_row: 0,
            settings_row_counts: Vec::new(),
            game_options_row: 0,
            game_options_row_count: 0,
        }
    }

    /// Builder: wire up which Games-rail index opens Settings, and the
    /// per-section row-count table Settings navigation uses.
    pub fn with_settings(
        mut self,
        settings_tile_index: Option<usize>,
        settings_row_counts: Vec<usize>,
    ) -> Self {
        self.settings_tile_index = settings_tile_index;
        self.settings_row_counts = settings_row_counts;
        self
    }

    /// Builder: the Media tab's rail length (spec §10 SM2).
    pub fn with_media_rail_len(mut self, media_rail_len: usize) -> Self {
        self.media_rail_len = media_rail_len;
        self
    }

    /// Builder: how many rows the per-game Game Options overlay exposes.
    pub fn with_game_options(mut self, row_count: usize) -> Self {
        self.game_options_row_count = row_count;
        self
    }

    /// Whether the Games-rail tile at `index` is a built-in app tile
    /// (Settings) rather than a launchable game. Used to gate Options: only
    /// real games have per-game settings.
    fn is_app_tile(&self, index: usize) -> bool {
        self.settings_tile_index == Some(index)
    }

    /// Replace the Games rail after a library rescan: new rail length and the
    /// (possibly moved) Settings tile index. Rail focus is clamped back into
    /// range; every other piece of navigation state (mode, tab, Settings
    /// focus) is untouched, so a rescan from inside Settings stays in
    /// Settings.
    pub fn set_games_rail(&mut self, rail_len: usize, settings_tile_index: Option<usize>) {
        self.games_rail_len = rail_len;
        self.settings_tile_index = settings_tile_index;
        if self.tab == RailTab::Games {
            self.rail_len = rail_len;
            self.rail_index = self.rail_index.min(rail_len.saturating_sub(1));
        }
    }

    /// Replace the Settings section/row-count table — e.g. after adding or
    /// removing a game folder changes the Game Folders section's row count
    /// — clamping the current section/row focus back into range.
    pub fn set_settings_row_counts(&mut self, counts: Vec<usize>) {
        self.settings_row_counts = counts;
        if self.settings_section >= self.settings_row_counts.len() {
            self.settings_section = self.settings_row_counts.len().saturating_sub(1);
        }
        let rows = self
            .settings_row_counts
            .get(self.settings_section)
            .copied()
            .unwrap_or(0);
        self.settings_row = if rows == 0 {
            0
        } else {
            self.settings_row.min(rows - 1)
        };
    }

    /// Apply one input, mutating focus in place and returning the resulting
    /// action (if any).
    pub fn apply(&mut self, input: NavInput) -> NavAction {
        match self.mode {
            NavMode::Home => self.apply_home(input),
            NavMode::Pills => self.apply_pills(input),
            NavMode::ControlCenter => self.apply_control_center(input),
            NavMode::ControlCenterOption => self.apply_control_center_option(input),
            NavMode::Settings => self.apply_settings(input),
            NavMode::GameOptions => self.apply_game_options(input),
        }
    }

    /// Switch the active rail tab, resetting rail length and focus.
    fn set_tab(&mut self, tab: RailTab) {
        self.tab = tab;
        self.rail_len = match tab {
            RailTab::Games => self.games_rail_len,
            RailTab::Media => self.media_rail_len,
        };
        self.rail_index = 0;
    }

    /// The pill that corresponds to the active rail tab — where pill focus
    /// starts when Up moves it out of the rail.
    fn active_tab_pill(&self) -> usize {
        match self.tab {
            RailTab::Games => PILL_MY_GAMES,
            RailTab::Media => PILL_MEDIA,
        }
    }

    fn focused_option_count(&self) -> usize {
        self.cc_option_counts
            .get(self.cc_index)
            .copied()
            .unwrap_or(0)
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
            NavInput::Up => {
                self.mode = NavMode::Pills;
                self.pill_index = self.active_tab_pill();
                NavAction::None
            }
            NavInput::Down => NavAction::None,
            NavInput::Confirm => match self.tab {
                RailTab::Games => {
                    if self.settings_tile_index == Some(self.rail_index) {
                        self.mode = NavMode::Settings;
                        self.settings_section = 0;
                        self.settings_row = 0;
                        NavAction::OpenSettings
                    } else {
                        NavAction::Launch(self.rail_index)
                    }
                }
                RailTab::Media => NavAction::LaunchMedia(self.rail_index),
            },
            NavInput::Guide => {
                self.mode = NavMode::ControlCenter;
                self.cc_index = 0;
                NavAction::OpenControlCenter
            }
            // Back at the top level is the way OUT of the Shell. Fullscreen is
            // borderless with no decorations, so without this there is no
            // close button, no title bar, and nothing Esc can do — the app is
            // a trap (reported from a real session). The caller decides what
            // "leave" means: drop out of fullscreen if we are in it, otherwise
            // it is the user's explicit exit gesture.
            NavInput::Back => NavAction::LeaveShell,
            NavInput::Tab => {
                self.set_tab(self.tab.toggled());
                NavAction::SwitchTab(self.tab)
            }
            NavInput::Options => {
                // Per-game settings only apply to real games, and only the
                // Games rail carries them.
                if self.tab == RailTab::Games
                    && self.rail_len > 0
                    && !self.is_app_tile(self.rail_index)
                {
                    self.mode = NavMode::GameOptions;
                    self.game_options_row = 0;
                    NavAction::OpenGameOptions {
                        index: self.rail_index,
                    }
                } else {
                    NavAction::None
                }
            }
        }
    }

    fn apply_pills(&mut self, input: NavInput) -> NavAction {
        match input {
            NavInput::Left => {
                self.pill_index = self.pill_index.saturating_sub(1);
                NavAction::None
            }
            NavInput::Right => {
                self.pill_index = (self.pill_index + 1).min(PILL_COUNT - 1);
                NavAction::None
            }
            NavInput::Up => NavAction::None,
            NavInput::Down | NavInput::Back => {
                self.mode = NavMode::Home;
                NavAction::None
            }
            NavInput::Confirm => match self.pill_index {
                PILL_MY_GAMES => {
                    self.set_tab(RailTab::Games);
                    self.mode = NavMode::Home;
                    NavAction::SwitchTab(RailTab::Games)
                }
                PILL_MEDIA => {
                    self.set_tab(RailTab::Media);
                    self.mode = NavMode::Home;
                    NavAction::SwitchTab(RailTab::Media)
                }
                PILL_SETTINGS => {
                    self.mode = NavMode::Settings;
                    self.settings_section = 0;
                    self.settings_row = 0;
                    NavAction::OpenSettings
                }
                _ => NavAction::None,
            },
            NavInput::Guide => {
                self.mode = NavMode::ControlCenter;
                self.cc_index = 0;
                NavAction::OpenControlCenter
            }
            NavInput::Tab => {
                self.set_tab(self.tab.toggled());
                NavAction::SwitchTab(self.tab)
            }
            NavInput::Options => NavAction::None,
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
            NavInput::Up | NavInput::Down | NavInput::Tab | NavInput::Options => NavAction::None,
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
            NavInput::Up | NavInput::Down | NavInput::Tab | NavInput::Options => NavAction::None,
            NavInput::Confirm => {
                let action = NavAction::ActivateOption {
                    card: self.cc_index,
                    option: self.cc_option_index,
                };
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

    fn apply_settings(&mut self, input: NavInput) -> NavAction {
        let section_count = self.settings_row_counts.len();
        match input {
            NavInput::Up => {
                if self.settings_row > 0 {
                    self.settings_row -= 1;
                } else if self.settings_section > 0 {
                    self.settings_section -= 1;
                    let rows = self
                        .settings_row_counts
                        .get(self.settings_section)
                        .copied()
                        .unwrap_or(0);
                    self.settings_row = rows.saturating_sub(1);
                }
                NavAction::None
            }
            NavInput::Down => {
                let rows = self
                    .settings_row_counts
                    .get(self.settings_section)
                    .copied()
                    .unwrap_or(0);
                if rows > 0 && self.settings_row + 1 < rows {
                    self.settings_row += 1;
                } else if self.settings_section + 1 < section_count {
                    self.settings_section += 1;
                    self.settings_row = 0;
                }
                NavAction::None
            }
            NavInput::Left => NavAction::AdjustSetting {
                section: self.settings_section,
                row: self.settings_row,
                delta: -1,
            },
            NavInput::Right => NavAction::AdjustSetting {
                section: self.settings_section,
                row: self.settings_row,
                delta: 1,
            },
            NavInput::Confirm => NavAction::ActivateSetting {
                section: self.settings_section,
                row: self.settings_row,
            },
            NavInput::Back => {
                self.mode = NavMode::Home;
                NavAction::CloseSettings
            }
            NavInput::Guide | NavInput::Tab | NavInput::Options => NavAction::None,
        }
    }

    /// Navigate the per-game Game Options overlay: Up/Down move the focused row,
    /// Left/Right adjust its value, Confirm toggles the override, Back closes
    /// and persists.
    fn apply_game_options(&mut self, input: NavInput) -> NavAction {
        let count = self.game_options_row_count;
        match input {
            NavInput::Up => {
                self.game_options_row = self.game_options_row.saturating_sub(1);
                NavAction::None
            }
            NavInput::Down => {
                if count > 0 {
                    self.game_options_row = (self.game_options_row + 1).min(count - 1);
                }
                NavAction::None
            }
            NavInput::Left => NavAction::AdjustGameOption {
                row: self.game_options_row,
                delta: -1,
            },
            NavInput::Right => NavAction::AdjustGameOption {
                row: self.game_options_row,
                delta: 1,
            },
            NavInput::Confirm => NavAction::ActivateGameOption {
                row: self.game_options_row,
            },
            NavInput::Back => {
                self.mode = NavMode::Home;
                NavAction::CloseGameOptions
            }
            NavInput::Guide | NavInput::Tab | NavInput::Options => NavAction::None,
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
            Case {
                name: "right moves forward",
                start_index: 0,
                rail_len: 5,
                input: NavInput::Right,
                expect_index: 1,
                expect_action: NavAction::None,
            },
            Case {
                name: "left clamps at zero",
                start_index: 0,
                rail_len: 5,
                input: NavInput::Left,
                expect_index: 0,
                expect_action: NavAction::None,
            },
            Case {
                name: "right clamps at the end",
                start_index: 4,
                rail_len: 5,
                input: NavInput::Right,
                expect_index: 4,
                expect_action: NavAction::None,
            },
            Case {
                name: "left moves backward",
                start_index: 3,
                rail_len: 5,
                input: NavInput::Left,
                expect_index: 2,
                expect_action: NavAction::None,
            },
            Case {
                name: "confirm launches focused index",
                start_index: 2,
                rail_len: 5,
                input: NavInput::Confirm,
                expect_index: 2,
                expect_action: NavAction::Launch(2),
            },
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

    /// Back at the top level must report [`NavAction::LeaveShell`], and must
    /// NOT do so while an overlay owns input — otherwise Esc would drop the
    /// user out of fullscreen (or quit) instead of closing the Control
    /// Center / Settings / Game Options they actually meant to leave.
    ///
    /// Regression guard for a real trap: fullscreen is borderless, so with
    /// Back a no-op at Home there was no close button, no title bar, and no
    /// key that did anything — the app could only be left via an
    /// undiscoverable Control Center path or by killing the process.
    #[test]
    fn back_leaves_the_shell_only_at_the_top_level() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11])
            .with_settings(Some(8), vec![4])
            .with_game_options(5);

        // Top level: Back is the way out.
        assert_eq!(nav.apply(NavInput::Back), NavAction::LeaveShell);
        assert_eq!(nav.mode, NavMode::Home, "leaving is the caller's job");

        // Control Center overlaid: Back closes it, and does NOT leave.
        nav.apply(NavInput::Guide);
        assert_eq!(nav.mode, NavMode::ControlCenter);
        assert_eq!(nav.apply(NavInput::Back), NavAction::CloseControlCenter);

        // Settings open: Back closes Settings, and does NOT leave.
        nav.rail_index = 8;
        assert_eq!(nav.apply(NavInput::Confirm), NavAction::OpenSettings);
        assert_eq!(nav.apply(NavInput::Back), NavAction::CloseSettings);

        // Game Options open: Back closes the overlay, and does NOT leave.
        nav.rail_index = 2;
        assert_eq!(
            nav.apply(NavInput::Options),
            NavAction::OpenGameOptions { index: 2 }
        );
        assert_eq!(nav.apply(NavInput::Back), NavAction::CloseGameOptions);

        // Back at the pill row returns to the rail rather than leaving.
        nav.apply(NavInput::Up);
        assert_eq!(nav.mode, NavMode::Pills);
        assert_eq!(nav.apply(NavInput::Back), NavAction::None);
        assert_eq!(nav.mode, NavMode::Home);

        // ...and now that we are back at the top level, Back leaves again.
        assert_eq!(nav.apply(NavInput::Back), NavAction::LeaveShell);
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
            Case {
                name: "right moves forward",
                start_index: 0,
                cc_len: 11,
                input: NavInput::Right,
                expect_index: 1,
                expect_action: NavAction::None,
                expect_mode: NavMode::ControlCenter,
            },
            Case {
                name: "left clamps at zero",
                start_index: 0,
                cc_len: 11,
                input: NavInput::Left,
                expect_index: 0,
                expect_action: NavAction::None,
                expect_mode: NavMode::ControlCenter,
            },
            Case {
                name: "right clamps at the end",
                start_index: 10,
                cc_len: 11,
                input: NavInput::Right,
                expect_index: 10,
                expect_action: NavAction::None,
                expect_mode: NavMode::ControlCenter,
            },
            Case {
                name: "escape (Back) closes",
                start_index: 4,
                cc_len: 11,
                input: NavInput::Back,
                expect_index: 4,
                expect_action: NavAction::CloseControlCenter,
                expect_mode: NavMode::Home,
            },
            Case {
                name: "guide toggles closed",
                start_index: 4,
                cc_len: 11,
                input: NavInput::Guide,
                expect_index: 4,
                expect_action: NavAction::CloseControlCenter,
                expect_mode: NavMode::Home,
            },
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
            Case {
                name: "confirm on the options card drills in",
                start_mode: NavMode::ControlCenter,
                start_option_index: 0,
                input: NavInput::Confirm,
                expect_mode: NavMode::ControlCenterOption,
                expect_option_index: 0,
                expect_action: NavAction::None,
            },
            Case {
                name: "right moves within options",
                start_mode: NavMode::ControlCenterOption,
                start_option_index: 0,
                input: NavInput::Right,
                expect_mode: NavMode::ControlCenterOption,
                expect_option_index: 1,
                expect_action: NavAction::None,
            },
            Case {
                name: "right clamps at the last option",
                start_mode: NavMode::ControlCenterOption,
                start_option_index: 2,
                input: NavInput::Right,
                expect_mode: NavMode::ControlCenterOption,
                expect_option_index: 2,
                expect_action: NavAction::None,
            },
            Case {
                name: "left clamps at zero",
                start_mode: NavMode::ControlCenterOption,
                start_option_index: 0,
                input: NavInput::Left,
                expect_mode: NavMode::ControlCenterOption,
                expect_option_index: 0,
                expect_action: NavAction::None,
            },
            Case {
                name: "confirm activates the selected option and returns to the card",
                start_mode: NavMode::ControlCenterOption,
                start_option_index: 1,
                input: NavInput::Confirm,
                expect_mode: NavMode::ControlCenter,
                expect_option_index: 1,
                expect_action: NavAction::ActivateOption {
                    card: power_card,
                    option: 1,
                },
            },
            Case {
                name: "back leaves option mode without activating",
                start_mode: NavMode::ControlCenterOption,
                start_option_index: 1,
                input: NavInput::Back,
                expect_mode: NavMode::ControlCenter,
                expect_option_index: 1,
                expect_action: NavAction::None,
            },
            Case {
                name: "guide closes control center entirely from option mode",
                start_mode: NavMode::ControlCenterOption,
                start_option_index: 1,
                input: NavInput::Guide,
                expect_mode: NavMode::Home,
                expect_option_index: 1,
                expect_action: NavAction::CloseControlCenter,
            },
        ];

        for case in cases {
            let mut nav = NavState::with_cc_options(9, option_counts.len(), option_counts.clone());
            nav.mode = case.start_mode;
            nav.cc_index = power_card;
            nav.cc_option_index = case.start_option_index;
            let action = nav.apply(case.input);
            assert_eq!(action, case.expect_action, "case: {}", case.name);
            assert_eq!(nav.mode, case.expect_mode, "case: {}", case.name);
            assert_eq!(
                nav.cc_option_index, case.expect_option_index,
                "case: {}",
                case.name
            );
        }
    }

    // --- SM2a: Settings mode ------------------------------------------------

    #[test]
    fn confirm_on_the_settings_tile_opens_settings_instead_of_launching() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11])
            .with_settings(Some(8), vec![4, 3, 2, 3, 1, 1]);
        nav.rail_index = 8;
        let action = nav.apply(NavInput::Confirm);
        assert_eq!(action, NavAction::OpenSettings);
        assert_eq!(nav.mode, NavMode::Settings);
        assert_eq!(nav.settings_section, 0);
        assert_eq!(nav.settings_row, 0);
    }

    #[test]
    fn confirm_on_a_non_settings_tile_still_launches() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11]).with_settings(Some(8), vec![4]);
        nav.rail_index = 2;
        assert_eq!(nav.apply(NavInput::Confirm), NavAction::Launch(2));
        assert_eq!(nav.mode, NavMode::Home);
    }

    #[test]
    fn no_settings_tile_wired_up_never_diverts_confirm() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11]); // settings_tile_index left None
        nav.rail_index = 0;
        assert_eq!(nav.apply(NavInput::Confirm), NavAction::Launch(0));
    }

    /// Table-driven: Settings' Up/Down row navigation, including crossing
    /// section boundaries and clamping at the very top/bottom.
    #[test]
    fn settings_row_navigation_crosses_section_boundaries() {
        struct Case {
            name: &'static str,
            start_section: usize,
            start_row: usize,
            input: NavInput,
            expect_section: usize,
            expect_row: usize,
        }

        let counts = vec![4, 3, 2]; // three sections for this table

        let cases = [
            Case {
                name: "down within a section",
                start_section: 0,
                start_row: 0,
                input: NavInput::Down,
                expect_section: 0,
                expect_row: 1,
            },
            Case {
                name: "down at the last row of a section moves to the next section",
                start_section: 0,
                start_row: 3,
                input: NavInput::Down,
                expect_section: 1,
                expect_row: 0,
            },
            Case {
                name: "down at the very last row clamps",
                start_section: 2,
                start_row: 1,
                input: NavInput::Down,
                expect_section: 2,
                expect_row: 1,
            },
            Case {
                name: "up within a section",
                start_section: 1,
                start_row: 2,
                input: NavInput::Up,
                expect_section: 1,
                expect_row: 1,
            },
            Case {
                name: "up at the first row of a section moves to the previous section's last row",
                start_section: 1,
                start_row: 0,
                input: NavInput::Up,
                expect_section: 0,
                expect_row: 3,
            },
            Case {
                name: "up at the very first row clamps",
                start_section: 0,
                start_row: 0,
                input: NavInput::Up,
                expect_section: 0,
                expect_row: 0,
            },
        ];

        for case in cases {
            let mut nav = NavState::with_cc_options(9, 11, vec![0; 11])
                .with_settings(Some(8), counts.clone());
            nav.mode = NavMode::Settings;
            nav.settings_section = case.start_section;
            nav.settings_row = case.start_row;
            let action = nav.apply(case.input);
            assert_eq!(action, NavAction::None, "case: {}", case.name);
            assert_eq!(
                nav.settings_section, case.expect_section,
                "case: {}",
                case.name
            );
            assert_eq!(nav.settings_row, case.expect_row, "case: {}", case.name);
        }
    }

    #[test]
    fn settings_left_right_and_confirm_report_the_focused_row() {
        let mut nav =
            NavState::with_cc_options(9, 11, vec![0; 11]).with_settings(Some(8), vec![4, 3]);
        nav.mode = NavMode::Settings;
        nav.settings_section = 1;
        nav.settings_row = 2;
        assert_eq!(
            nav.apply(NavInput::Right),
            NavAction::AdjustSetting {
                section: 1,
                row: 2,
                delta: 1
            }
        );
        assert_eq!(
            nav.apply(NavInput::Left),
            NavAction::AdjustSetting {
                section: 1,
                row: 2,
                delta: -1
            }
        );
        assert_eq!(
            nav.apply(NavInput::Confirm),
            NavAction::ActivateSetting { section: 1, row: 2 }
        );
        // Adjusting/activating a row never itself leaves Settings.
        assert_eq!(nav.mode, NavMode::Settings);
    }

    #[test]
    fn back_leaves_settings_and_restores_home_focus() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11]).with_settings(Some(8), vec![4]);
        nav.rail_index = 8;
        assert_eq!(nav.apply(NavInput::Confirm), NavAction::OpenSettings);
        assert_eq!(nav.mode, NavMode::Settings);

        let action = nav.apply(NavInput::Back);
        assert_eq!(action, NavAction::CloseSettings);
        assert_eq!(nav.mode, NavMode::Home);
        // The Home rail focus (the Settings tile itself) survived the trip.
        assert_eq!(nav.rail_index, 8);
    }

    #[test]
    fn set_games_rail_clamps_focus_and_preserves_mode() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11])
            .with_settings(Some(8), vec![4])
            .with_media_rail_len(3);
        nav.mode = NavMode::Settings;
        nav.rail_index = 8;
        // Library shrank from 9 to 3 tiles; Settings tile moved to index 2.
        nav.set_games_rail(3, Some(2));
        assert_eq!(nav.rail_len, 3);
        assert_eq!(nav.rail_index, 2, "focus clamps into the new rail");
        assert_eq!(nav.mode, NavMode::Settings, "mode is untouched");
        // Confirm on the new Settings tile index diverts to Settings.
        nav.mode = NavMode::Home;
        nav.rail_index = 2;
        assert_eq!(nav.apply(NavInput::Confirm), NavAction::OpenSettings);
    }

    #[test]
    fn set_games_rail_on_the_media_tab_defers_to_tab_switch() {
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11]).with_media_rail_len(3);
        nav.apply(NavInput::Tab); // -> Media
        nav.rail_index = 1;
        nav.set_games_rail(2, None);
        // Media rail untouched now; the new Games length applies on switch.
        assert_eq!(nav.rail_len, 3);
        assert_eq!(nav.rail_index, 1);
        nav.apply(NavInput::Tab); // -> Games
        assert_eq!(nav.rail_len, 2);
    }

    #[test]
    fn set_settings_row_counts_clamps_focus_when_a_section_shrinks() {
        let mut nav =
            NavState::with_cc_options(9, 11, vec![0; 11]).with_settings(Some(8), vec![4, 3]);
        nav.mode = NavMode::Settings;
        nav.settings_section = 1;
        nav.settings_row = 2; // last row of a 3-row section
        nav.set_settings_row_counts(vec![4, 1]); // that section shrinks to 1 row
        assert_eq!(nav.settings_row, 0);
    }

    #[test]
    fn set_settings_row_counts_clamps_section_when_the_section_count_shrinks() {
        let mut nav =
            NavState::with_cc_options(9, 11, vec![0; 11]).with_settings(Some(8), vec![4, 3, 2]);
        nav.mode = NavMode::Settings;
        nav.settings_section = 2;
        nav.set_settings_row_counts(vec![4, 3]);
        assert_eq!(nav.settings_section, 1);
        assert_eq!(nav.settings_row, 0);
    }

    #[test]
    fn empty_settings_section_list_does_not_panic_on_up_or_down() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11]).with_settings(Some(8), vec![]);
        nav.mode = NavMode::Settings;
        assert_eq!(nav.apply(NavInput::Down), NavAction::None);
        assert_eq!(nav.apply(NavInput::Up), NavAction::None);
        assert_eq!(nav.settings_section, 0);
        assert_eq!(nav.settings_row, 0);
    }

    // --- SM2a: Games/Media tab switching -------------------------------------

    #[test]
    fn tab_switches_between_games_and_media_and_resets_rail_focus() {
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11]).with_media_rail_len(3);
        nav.rail_index = 4;

        let action = nav.apply(NavInput::Tab);
        assert_eq!(action, NavAction::SwitchTab(RailTab::Media));
        assert_eq!(nav.tab, RailTab::Media);
        assert_eq!(nav.rail_len, 3);
        assert_eq!(nav.rail_index, 0);

        nav.rail_index = 2;
        let action = nav.apply(NavInput::Tab);
        assert_eq!(action, NavAction::SwitchTab(RailTab::Games));
        assert_eq!(nav.tab, RailTab::Games);
        assert_eq!(nav.rail_len, 5);
        assert_eq!(nav.rail_index, 0);
    }

    #[test]
    fn media_tab_rail_navigation_clamps_independently_of_games_rail_len() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11]).with_media_rail_len(2);
        nav.apply(NavInput::Tab); // -> Media, rail_len == 2
        assert_eq!(nav.apply(NavInput::Right), NavAction::None);
        assert_eq!(nav.rail_index, 1);
        assert_eq!(nav.apply(NavInput::Right), NavAction::None);
        assert_eq!(
            nav.rail_index, 1,
            "clamps at the Media rail's own length, not the Games rail's"
        );
    }

    #[test]
    fn confirm_on_the_media_tab_reports_launch_media_not_launch() {
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11]).with_media_rail_len(3);
        nav.apply(NavInput::Tab); // -> Media
        nav.rail_index = 1;
        assert_eq!(nav.apply(NavInput::Confirm), NavAction::LaunchMedia(1));
        assert_eq!(nav.mode, NavMode::Home);
    }

    // --- Pill-row focus (concept-mock Home) ----------------------------------

    #[test]
    fn up_from_the_rail_enters_pills_on_the_active_tabs_pill() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11]).with_media_rail_len(3);
        assert_eq!(nav.apply(NavInput::Up), NavAction::None);
        assert_eq!(nav.mode, NavMode::Pills);
        assert_eq!(nav.pill_index, PILL_MY_GAMES);

        nav.apply(NavInput::Back); // back to the rail
        nav.apply(NavInput::Tab); // -> Media
        nav.apply(NavInput::Up);
        assert_eq!(nav.mode, NavMode::Pills);
        assert_eq!(
            nav.pill_index, PILL_MEDIA,
            "pill focus starts on the pill matching the active tab"
        );
    }

    /// Table-driven: pill-row movement and exits.
    #[test]
    fn pills_navigation_table() {
        struct Case {
            name: &'static str,
            start_pill: usize,
            input: NavInput,
            expect_pill: usize,
            expect_mode: NavMode,
        }

        let cases = [
            Case {
                name: "right moves forward",
                start_pill: PILL_MY_GAMES,
                input: NavInput::Right,
                expect_pill: PILL_MEDIA,
                expect_mode: NavMode::Pills,
            },
            Case {
                name: "left moves backward",
                start_pill: PILL_MEDIA,
                input: NavInput::Left,
                expect_pill: PILL_MY_GAMES,
                expect_mode: NavMode::Pills,
            },
            Case {
                name: "left clamps at the first pill",
                start_pill: PILL_MY_GAMES,
                input: NavInput::Left,
                expect_pill: PILL_MY_GAMES,
                expect_mode: NavMode::Pills,
            },
            Case {
                name: "right clamps at the last pill",
                start_pill: PILL_SETTINGS,
                input: NavInput::Right,
                expect_pill: PILL_SETTINGS,
                expect_mode: NavMode::Pills,
            },
            Case {
                name: "down returns to the rail",
                start_pill: PILL_MEDIA,
                input: NavInput::Down,
                expect_pill: PILL_MEDIA,
                expect_mode: NavMode::Home,
            },
            Case {
                name: "back returns to the rail",
                start_pill: PILL_MEDIA,
                input: NavInput::Back,
                expect_pill: PILL_MEDIA,
                expect_mode: NavMode::Home,
            },
            Case {
                name: "up stays put",
                start_pill: PILL_MY_GAMES,
                input: NavInput::Up,
                expect_pill: PILL_MY_GAMES,
                expect_mode: NavMode::Pills,
            },
        ];

        for case in cases {
            let mut nav = NavState::with_cc_options(9, 11, vec![0; 11]);
            nav.mode = NavMode::Pills;
            nav.pill_index = case.start_pill;
            let action = nav.apply(case.input);
            assert_eq!(action, NavAction::None, "case: {}", case.name);
            assert_eq!(nav.pill_index, case.expect_pill, "case: {}", case.name);
            assert_eq!(nav.mode, case.expect_mode, "case: {}", case.name);
        }
    }

    #[test]
    fn confirming_the_tab_pills_switches_the_rail_tab_and_returns_home() {
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11]).with_media_rail_len(3);
        nav.mode = NavMode::Pills;
        nav.pill_index = PILL_MEDIA;
        assert_eq!(
            nav.apply(NavInput::Confirm),
            NavAction::SwitchTab(RailTab::Media)
        );
        assert_eq!(nav.mode, NavMode::Home);
        assert_eq!(nav.tab, RailTab::Media);
        assert_eq!(nav.rail_len, 3);
        assert_eq!(nav.rail_index, 0);

        nav.apply(NavInput::Up);
        nav.pill_index = PILL_MY_GAMES;
        assert_eq!(
            nav.apply(NavInput::Confirm),
            NavAction::SwitchTab(RailTab::Games)
        );
        assert_eq!(nav.tab, RailTab::Games);
        assert_eq!(nav.rail_len, 5);
    }

    #[test]
    fn confirming_the_settings_pill_opens_settings() {
        let mut nav =
            NavState::with_cc_options(9, 11, vec![0; 11]).with_settings(Some(8), vec![4, 3]);
        nav.mode = NavMode::Pills;
        nav.pill_index = PILL_SETTINGS;
        assert_eq!(nav.apply(NavInput::Confirm), NavAction::OpenSettings);
        assert_eq!(nav.mode, NavMode::Settings);
        assert_eq!(nav.settings_section, 0);
        assert_eq!(nav.settings_row, 0);
    }

    #[test]
    fn guide_from_pills_opens_control_center() {
        let mut nav = NavState::with_cc_options(9, 11, vec![0; 11]);
        nav.mode = NavMode::Pills;
        assert_eq!(nav.apply(NavInput::Guide), NavAction::OpenControlCenter);
        assert_eq!(nav.mode, NavMode::ControlCenter);
        assert_eq!(nav.cc_index, 0);
    }

    #[test]
    fn media_tab_confirm_ignores_the_games_tab_settings_tile_index() {
        // Settings tile is Games-rail index 1; on the Media tab, index 1
        // must never be diverted into Settings.
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11])
            .with_settings(Some(1), vec![4])
            .with_media_rail_len(3);
        nav.apply(NavInput::Tab); // -> Media
        nav.rail_index = 1;
        assert_eq!(nav.apply(NavInput::Confirm), NavAction::LaunchMedia(1));
        assert_eq!(nav.mode, NavMode::Home);
    }

    #[test]
    fn options_on_a_game_tile_opens_game_options() {
        // Games rail of 5; index 4 is the Settings app tile, so index 2 is a
        // game.
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11])
            .with_settings(Some(4), vec![9])
            .with_game_options(5);
        nav.rail_index = 2;
        assert_eq!(
            nav.apply(NavInput::Options),
            NavAction::OpenGameOptions { index: 2 }
        );
        assert_eq!(nav.mode, NavMode::GameOptions);
        assert_eq!(nav.game_options_row, 0);
    }

    #[test]
    fn options_on_an_app_tile_is_a_no_op() {
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11])
            .with_settings(Some(4), vec![9])
            .with_game_options(5);
        // Focus the Settings app tile — it has no per-game settings.
        nav.rail_index = 4;
        assert_eq!(nav.apply(NavInput::Options), NavAction::None);
        assert_eq!(nav.mode, NavMode::Home);
    }

    #[test]
    fn options_on_media_tab_is_a_no_op() {
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11])
            .with_media_rail_len(3)
            .with_game_options(5);
        nav.apply(NavInput::Tab); // -> Media
        assert_eq!(nav.apply(NavInput::Options), NavAction::None);
        assert_eq!(nav.mode, NavMode::Home);
    }

    #[test]
    fn game_options_navigation_and_adjust() {
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11])
            .with_settings(Some(4), vec![9])
            .with_game_options(5);
        nav.rail_index = 2;
        nav.apply(NavInput::Options);
        // Down moves the focused row; clamps at the last row.
        assert_eq!(nav.apply(NavInput::Down), NavAction::None);
        assert_eq!(nav.game_options_row, 1);
        // Right/Left/Confirm surface adjust/activate actions for the caller.
        assert_eq!(
            nav.apply(NavInput::Right),
            NavAction::AdjustGameOption { row: 1, delta: 1 }
        );
        assert_eq!(
            nav.apply(NavInput::Confirm),
            NavAction::ActivateGameOption { row: 1 }
        );
    }

    // --- Property tests (proptest) -------------------------------------------

    proptest::proptest! {
        /// ANY input sequence keeps every focus index in bounds and the state
        /// machine panic-free — the invariant every render path relies on
        /// when it indexes rows/tiles by the nav state.
        #[test]
        fn any_input_sequence_keeps_focus_in_bounds(
            inputs in proptest::collection::vec(0usize..9, 0..96)
        ) {
            const ALL: [NavInput; 9] = [
                NavInput::Left,
                NavInput::Right,
                NavInput::Up,
                NavInput::Down,
                NavInput::Confirm,
                NavInput::Guide,
                NavInput::Back,
                NavInput::Tab,
                NavInput::Options,
            ];
            let counts = vec![4usize, 3, 2];
            let mut nav = NavState::with_cc_options(9, 5, vec![0, 0, 3, 0, 0])
                .with_settings(Some(8), counts.clone())
                .with_media_rail_len(3)
                .with_game_options(5);
            for index in inputs {
                nav.apply(ALL[index]);
                proptest::prop_assert!(nav.rail_index < nav.rail_len.max(1));
                proptest::prop_assert!(nav.pill_index < PILL_COUNT);
                proptest::prop_assert!(nav.cc_index < 5);
                proptest::prop_assert!(nav.cc_option_index < 3.max(1));
                proptest::prop_assert!(nav.settings_section < counts.len());
                proptest::prop_assert!(
                    nav.settings_row < counts[nav.settings_section].max(1)
                );
                proptest::prop_assert!(nav.game_options_row < 5);
            }
        }
    }

    #[test]
    fn game_options_back_closes_and_signals_persist() {
        let mut nav = NavState::with_cc_options(5, 11, vec![0; 11])
            .with_settings(Some(4), vec![9])
            .with_game_options(5);
        nav.rail_index = 2;
        nav.apply(NavInput::Options);
        assert_eq!(nav.apply(NavInput::Back), NavAction::CloseGameOptions);
        assert_eq!(nav.mode, NavMode::Home);
    }
}
