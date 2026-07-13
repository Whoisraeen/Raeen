//! Shell state machine: Boot → Home (+ Control Center overlay) → launch
//! transition → back to Home (spec §3, §5, §10 — SM0 scope).
//!
//! Owns navigation state, animated values, the library, the active
//! [`GameLauncher`], and a best-effort `gilrs` gamepad connection. Frame
//! driving lives in `app.rs`; this module is where input becomes
//! [`nav::NavAction`]s and where those actions become launcher calls.

pub mod anim;
pub mod boot;
pub mod control_center;
pub mod home;
pub mod icons;
pub mod nav;

use crate::launcher::{GameLauncher, SessionHandle, SessionState};
use crate::library::{Gradient, LibraryItem, MetaCache};
use crate::theme::Theme;
use anim::Animated;
use boot::BootSequence;
use egui::Key;
use home::HomeAnim;
use nav::{NavAction, NavInput, NavMode, NavState};

enum Screen {
    Boot(BootSequence),
    Home,
}

/// A game session launched from Home, tracked until it exits.
struct ActiveSession {
    handle: SessionHandle,
    title: String,
    /// Rail index that was focused when Play was pressed, so we return to
    /// the same tile on exit (spec §5).
    target_index: usize,
}

/// Cap on the Switcher's recent-titles history (spec §10).
const MAX_RECENT_TITLES: usize = 6;

/// The full Shell: navigation, animation, library, and the launcher seam.
pub struct Shell {
    theme: Theme,
    library: Vec<LibraryItem>,
    meta_cache: MetaCache,
    nav: NavState,
    screen: Screen,
    launcher: Box<dyn GameLauncher>,
    session: Option<ActiveSession>,
    gilrs: Option<gilrs::Gilrs>,
    /// Ids of launched titles, most-recent-first, deduplicated and capped —
    /// backs the Control Center's Switcher panel (spec §10).
    recent: Vec<String>,

    rail_offset: Animated,
    focus_pop: Animated,
    hero_from: Option<Gradient>,
    hero_to: Gradient,
    hero_t: Animated,
    cc_open: Animated,
    last_rail_index: usize,
}

impl Shell {
    pub fn new(theme: Theme, library: Vec<LibraryItem>, launcher: Box<dyn GameLauncher>) -> Self {
        let rail_len = library.len();
        let cc_len = control_center::ITEMS.len();
        let cc_option_counts: Vec<usize> = control_center::ITEMS.iter().map(|item| item.option_count()).collect();
        let hero_to = library.first().map(|i| i.art.hero()).unwrap_or(Gradient {
            hi: theme.palette.raised,
            mid: theme.palette.raised,
            lo: theme.palette.ground,
        });
        let meta_cache = MetaCache::from_items(&library);

        let gilrs = gilrs::Gilrs::new().ok();
        if gilrs.is_none() {
            tracing::warn!("gamepad support unavailable (gilrs init failed) — keyboard still works");
        }

        Self {
            theme,
            library,
            meta_cache,
            nav: NavState::with_cc_options(rail_len, cc_len, cc_option_counts),
            screen: Screen::Boot(BootSequence::new()),
            launcher,
            session: None,
            gilrs,
            recent: Vec::new(),
            rail_offset: Animated::new(0.0),
            focus_pop: Animated::with_speed(1.0, 12.0),
            hero_from: None,
            hero_to,
            hero_t: Animated::with_speed(1.0, 6.0),
            cc_open: Animated::with_speed(0.0, 11.0),
            last_rail_index: 0,
        }
    }

    /// Drive one frame: advance boot/animation state, route input, poll any
    /// active session, and draw.
    pub fn update(&mut self, ctx: &egui::Context) {
        if let Screen::Boot(boot) = &self.screen {
            boot::draw(ctx, &self.theme, boot);
            if boot.is_done() {
                self.screen = Screen::Home;
            } else {
                return;
            }
        }

        self.route_input(ctx);
        self.poll_session();
        self.tick_animations(ctx);
        self.draw(ctx);
    }

    fn route_input(&mut self, ctx: &egui::Context) {
        let inputs = self.poll_nav_inputs(ctx);

        // While a session is loading/running, the Shell only listens for
        // Back (quit-to-shell); rail/Control-Center navigation is parked.
        if let Some(session) = &self.session {
            if inputs.contains(&NavInput::Back) {
                let _ = self.launcher.quit(&session.handle);
            }
            return;
        }

        for input in inputs {
            match self.nav.apply(input) {
                NavAction::Launch(index) => self.begin_launch(index),
                NavAction::ActivateOption { card, option } => self.handle_cc_option(ctx, card, option),
                NavAction::OpenControlCenter | NavAction::CloseControlCenter | NavAction::None => {}
            }
        }
    }

    /// Handle Confirm on a Control Center card's option list (currently
    /// only Power: Rest Mode / Restart / Turn Off — spec §10). Rest/Restart
    /// are no-op stubs for SM1; Turn Off actually closes the Shell.
    fn handle_cc_option(&mut self, ctx: &egui::Context, card: usize, option: usize) {
        let Some(item) = control_center::ITEMS.get(card) else { return };
        if item.name != "Power" {
            return;
        }
        match option {
            0 => tracing::info!("Rest Mode requested (stub — no-op in SM1)"),
            1 => tracing::info!("Restart requested (stub — no-op in SM1)"),
            2 => {
                tracing::info!("Turn Off requested — closing XPS5X");
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            _ => {}
        }
    }

    fn poll_nav_inputs(&mut self, ctx: &egui::Context) -> Vec<NavInput> {
        let mut inputs = Vec::new();

        ctx.input(|i| {
            if i.key_pressed(Key::ArrowLeft) {
                inputs.push(NavInput::Left);
            }
            if i.key_pressed(Key::ArrowRight) {
                inputs.push(NavInput::Right);
            }
            if i.key_pressed(Key::Enter) {
                inputs.push(NavInput::Confirm);
            }
            if i.key_pressed(Key::Escape) {
                inputs.push(NavInput::Back);
            }
            if i.key_pressed(Key::C) {
                inputs.push(NavInput::Guide);
            }
        });

        if let Some(gilrs) = self.gilrs.as_mut() {
            while let Some(gilrs::Event { event, .. }) = gilrs.next_event() {
                match event {
                    gilrs::EventType::ButtonPressed(gilrs::Button::DPadLeft, _) => inputs.push(NavInput::Left),
                    gilrs::EventType::ButtonPressed(gilrs::Button::DPadRight, _) => inputs.push(NavInput::Right),
                    gilrs::EventType::ButtonPressed(gilrs::Button::South, _) => inputs.push(NavInput::Confirm),
                    gilrs::EventType::ButtonPressed(gilrs::Button::East, _) => inputs.push(NavInput::Back),
                    gilrs::EventType::ButtonPressed(gilrs::Button::Mode, _) => inputs.push(NavInput::Guide),
                    _ => {}
                }
            }
        }

        inputs
    }

    fn begin_launch(&mut self, index: usize) {
        let Some(item) = self.library.get(index) else { return };
        match self.launcher.launch(&item.launch) {
            Ok(handle) => {
                push_recent(&mut self.recent, &item.id, MAX_RECENT_TITLES);
                self.session = Some(ActiveSession { handle, title: item.title.clone(), target_index: index });
            }
            Err(err) => {
                tracing::warn!(title = %item.title, error = %err, "launch failed");
            }
        }
    }

    fn poll_session(&mut self) {
        let Some(session) = &self.session else { return };
        match self.launcher.session_state(&session.handle) {
            SessionState::Exited | SessionState::Faulted => {
                self.nav.rail_index = session.target_index;
                self.session = None;
            }
            SessionState::Loading | SessionState::Running => {}
        }
    }

    fn tick_animations(&mut self, ctx: &egui::Context) {
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        let mut animating = false;

        if self.nav.rail_index != self.last_rail_index {
            self.last_rail_index = self.nav.rail_index;
            self.focus_pop.value = 0.0;
            self.focus_pop.set_target(1.0);

            let new_hero = self.library.get(self.nav.rail_index).map(|i| i.art.hero());
            if let Some(new_hero) = new_hero {
                self.hero_from = Some(self.blended_hero());
                self.hero_to = new_hero;
                self.hero_t.value = 0.0;
                self.hero_t.set_target(1.0);
            }
        }

        let target_offset = -(self.nav.rail_index as f32) * (self.theme.metrics.tile_size + self.theme.metrics.tile_gap);
        self.rail_offset.set_target(target_offset);
        animating |= self.rail_offset.tick(dt);
        animating |= self.focus_pop.tick(dt);
        animating |= self.hero_t.tick(dt);

        self.cc_open.set_target(if self.nav.mode == NavMode::ControlCenter { 1.0 } else { 0.0 });
        animating |= self.cc_open.tick(dt);

        if animating {
            ctx.request_repaint();
        }
    }

    fn blended_hero(&self) -> Gradient {
        match self.hero_from {
            Some(from) => {
                let t = anim::ease_out_cubic(self.hero_t.value);
                Gradient {
                    hi: anim::lerp_color(from.hi, self.hero_to.hi, t),
                    mid: anim::lerp_color(from.mid, self.hero_to.mid, t),
                    lo: anim::lerp_color(from.lo, self.hero_to.lo, t),
                }
            }
            None => self.hero_to,
        }
    }

    fn draw(&self, ctx: &egui::Context) {
        let theme = self.theme.clone();
        let frame = egui::Frame::NONE.fill(theme.palette.ground);

        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            if let Some(session) = &self.session {
                draw_session_overlay(ui, &theme, session, self.launcher.as_ref());
                return;
            }

            let anim = HomeAnim {
                rail_offset: self.rail_offset.value,
                hero: self.blended_hero(),
                focus_pop: self.focus_pop.value,
            };
            home::draw(ui, &theme, &self.library, &self.nav, &anim, &self.meta_cache);

            let recent_titles = self.recent_titles();
            control_center::draw(ui, &theme, &self.nav, self.cc_open.value, &recent_titles);
        });
    }

    /// Resolve the recent-titles id history to display titles, most-recent-
    /// first, for the Switcher panel.
    fn recent_titles(&self) -> Vec<String> {
        self.recent
            .iter()
            .filter_map(|id| self.library.iter().find(|item| &item.id == id).map(|item| item.title.clone()))
            .collect()
    }
}

/// Record a launch in the recent-titles history: most-recent-first,
/// de-duplicated (a repeat launch moves to the front instead of appearing
/// twice), and capped at `cap` entries (spec §10).
fn push_recent(recent: &mut Vec<String>, id: &str, cap: usize) {
    recent.retain(|existing| existing != id);
    recent.insert(0, id.to_string());
    recent.truncate(cap);
}

fn draw_session_overlay(ui: &mut egui::Ui, theme: &Theme, session: &ActiveSession, launcher: &dyn GameLauncher) {
    let screen = ui.max_rect();
    let painter = ui.painter();
    painter.rect_filled(screen, 0.0, theme.palette.ground);

    let state = launcher.session_state(&session.handle);
    let (headline, sub) = match state {
        SessionState::Loading => (format!("Launching {}…", session.title), "Handing off to the engine".to_string()),
        SessionState::Running => (session.title.clone(), "Running — Esc to return to the Shell".to_string()),
        SessionState::Faulted | SessionState::Exited => (session.title.clone(), "Returning to Shell…".to_string()),
    };

    let center = screen.center();
    painter.text(
        egui::pos2(center.x, center.y - 12.0),
        egui::Align2::CENTER_CENTER,
        headline,
        egui::FontId::proportional(32.0),
        theme.palette.text,
    );
    painter.text(
        egui::pos2(center.x, center.y + 24.0),
        egui::Align2::CENTER_CENTER,
        sub,
        egui::FontId::proportional(15.0),
        theme.palette.text_dim,
    );

    ui.ctx().request_repaint();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_recent_orders_most_recent_first() {
        let mut recent = Vec::new();
        push_recent(&mut recent, "a", 6);
        push_recent(&mut recent, "b", 6);
        assert_eq!(recent, vec!["b".to_string(), "a".to_string()]);
    }

    #[test]
    fn push_recent_deduplicates_repeats() {
        let mut recent = Vec::new();
        push_recent(&mut recent, "a", 6);
        push_recent(&mut recent, "b", 6);
        push_recent(&mut recent, "a", 6);
        assert_eq!(recent, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn push_recent_is_capped() {
        let mut recent = Vec::new();
        for id in ["a", "b", "c", "d"] {
            push_recent(&mut recent, id, 2);
        }
        assert_eq!(recent, vec!["d".to_string(), "c".to_string()]);
    }

    #[test]
    fn push_recent_cap_zero_yields_empty_history() {
        let mut recent = Vec::new();
        push_recent(&mut recent, "a", 0);
        assert!(recent.is_empty());
    }
}
