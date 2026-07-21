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
pub mod media;
pub mod nav;
pub(crate) mod present;
pub mod settings;

use crate::launcher::{GameLauncher, SessionHandle, SessionState};
use crate::library::{Gradient, LaunchTarget, LibraryItem, MetaCache};
use crate::theme::{self, Theme};
use crate::updater::{self, UpdaterEvent, UpdaterState};
use anim::Animated;
use boot::BootSequence;
use egui::Key;
use home::HomeAnim;
use nav::{NavAction, NavInput, NavMode, NavState, RailTab};
use std::collections::HashMap;
use std::path::PathBuf;
use xps5x_core::config::EmulatorConfig;

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
    /// The Media tab's rail (spec §10 SM2) — built once, same as `library`.
    library_media: Vec<LibraryItem>,
    meta_cache: MetaCache,
    nav: NavState,
    screen: Screen,
    launcher: Box<dyn GameLauncher>,
    session: Option<ActiveSession>,
    /// Shows the running title's rendered frames. Persistent because it owns
    /// the GPU texture the frame is uploaded into.
    frame_view: present::GameFrameView,
    gilrs: Option<gilrs::Gilrs>,
    /// Ids of launched titles, most-recent-first, deduplicated and capped —
    /// backs the Control Center's Switcher panel (spec §10).
    recent: Vec<String>,

    /// Live settings, edited in place by the Settings screen and persisted
    /// via `EmulatorConfig::save` when the user backs out of Settings.
    config: EmulatorConfig,
    config_path: PathBuf,
    /// Root directory theme names are resolved under: `themes_root/<name>/
    /// theme.toml` (spec §6, §10 SM2b).
    themes_root: PathBuf,
    /// The active theme's background image, if any, uploaded to the GPU
    /// once per theme (re)load — `home.rs` draws this instead of the mesh
    /// gradient hero when present (spec §6).
    background_texture: Option<egui::TextureHandle>,
    /// Per-game cover images (user `cover.png`, else the title's own
    /// `sce_sys/icon0.png`), decoded and uploaded once at construction, keyed
    /// by `LibraryItem::id`. Games without one keep gradient + monogram art.
    cover_textures: HashMap<String, egui::TextureHandle>,
    /// Per-game key-art backgrounds (`sce_sys/pic1.png`, else `pic0.png`,
    /// shipped by the title next to its eboot), decoded once at construction,
    /// keyed by `LibraryItem::id`. The focused game's entry is drawn full-bleed
    /// behind the Home rail; anything without one falls back to the mesh
    /// gradient hero.
    game_backgrounds: HashMap<String, egui::TextureHandle>,
    /// The crossfading pair of key-art backgrounds for the Home hero: `to` is
    /// the focused game's art, `from` is the previously-focused one, dissolved
    /// by `hero_t`. Cloned handles out of `game_backgrounds` (or `None` for an
    /// app tile / a game with no key art).
    hero_bg_from: Option<egui::TextureHandle>,
    hero_bg_to: Option<egui::TextureHandle>,
    /// Scratch text-entry buffers for Settings' two path fields. Kept on
    /// `Shell` rather than `nav::NavState`, which stays free of raw text
    /// state so it can remain a pure, egui-free state machine.
    settings_new_folder_input: String,
    settings_key_provider_input: String,

    /// Auto-updater state machine (Settings → System). Worker threads
    /// (check/download) report back over the mpsc channel; `update()` pumps
    /// it every frame. The Shell never blocks on the network.
    updater_state: UpdaterState,
    updater_tx: std::sync::mpsc::Sender<UpdaterEvent>,
    updater_rx: std::sync::mpsc::Receiver<UpdaterEvent>,

    rail_offset: Animated,
    focus_pop: Animated,
    hero_from: Option<Gradient>,
    hero_to: Gradient,
    hero_t: Animated,
    cc_open: Animated,
    last_rail_index: usize,
    last_tab: RailTab,
}

impl Shell {
    /// `ctx` is used once here to install the initial theme's font (if any)
    /// and upload its background image (if any) — see [`Self::reload_theme`],
    /// which this delegates to for that work so construction and a later
    /// Settings-driven theme switch share one code path.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &egui::Context,
        theme: Theme,
        themes_root: PathBuf,
        library: Vec<LibraryItem>,
        launcher: Box<dyn GameLauncher>,
        config: EmulatorConfig,
        config_path: PathBuf,
    ) -> Self {
        let rail_len = library.len();
        let cc_len = control_center::ITEMS.len();
        let cc_option_counts: Vec<usize> = control_center::ITEMS
            .iter()
            .map(|item| item.option_count())
            .collect();
        let hero_to = library.first().map(|i| i.art.hero()).unwrap_or(Gradient {
            hi: theme.palette.raised,
            mid: theme.palette.raised,
            lo: theme.palette.ground,
        });
        let meta_cache = MetaCache::from_items(&library);

        let library_media = media::media_items();
        // Settings is always one of the built-in apps (spec §10 SM2); if a
        // caller ever hands the Shell a library without one, Confirm on
        // that index simply never matches and nothing opens Settings.
        let settings_tile_index = library.iter().position(|item| item.id == "settings");
        let settings_row_counts = settings::settings_row_counts(config.paths.game_folders.len());
        // Store / Game Library tiles, so their nav pills can jump the rail
        // there (same "absent tile → pill no-ops" contract as Settings).
        let store_tile_index = library.iter().position(|item| item.id == "store");
        let library_tile_index = library.iter().position(|item| item.id == "library");

        let nav = NavState::with_cc_options(rail_len, cc_len, cc_option_counts)
            .with_settings(settings_tile_index, settings_row_counts)
            .with_media_rail_len(library_media.len())
            .with_app_tiles(store_tile_index, library_tile_index);

        let gilrs = gilrs::Gilrs::new().ok();
        if gilrs.is_none() {
            tracing::warn!(
                "gamepad support unavailable (gilrs init failed) — keyboard still works"
            );
        }

        let settings_key_provider_input = config.paths.key_provider_path.display().to_string();

        // Channel only — the initial network check is kicked off by
        // `app.rs` via `start_update_check()`, so constructing a Shell in
        // tests never touches the network.
        let (updater_tx, updater_rx) = std::sync::mpsc::channel();

        theme::install_fonts(ctx, &theme);
        let background_texture = background_texture_for(ctx, &theme);
        let cover_textures = cover_textures_for(ctx, &library);
        let game_backgrounds = background_textures_for(ctx, &library);
        // Seed the hero background with the initially-focused tile's key art.
        let hero_bg_to = library
            .first()
            .and_then(|i| game_backgrounds.get(i.id.as_str()).cloned());

        Self {
            theme,
            library,
            library_media,
            meta_cache,
            nav,
            screen: Screen::Boot(BootSequence::new()),
            launcher,
            session: None,
            frame_view: present::GameFrameView::default(),
            gilrs,
            recent: Vec::new(),
            config,
            config_path,
            themes_root,
            background_texture,
            cover_textures,
            game_backgrounds,
            hero_bg_from: None,
            hero_bg_to,
            settings_new_folder_input: String::new(),
            settings_key_provider_input,
            updater_state: UpdaterState::default(),
            updater_tx,
            updater_rx,
            rail_offset: Animated::new(0.0),
            focus_pop: Animated::with_speed(1.0, 12.0),
            hero_from: None,
            hero_to,
            hero_t: Animated::with_speed(1.0, 6.0),
            cc_open: Animated::with_speed(0.0, 11.0),
            last_rail_index: 0,
            last_tab: RailTab::Games,
        }
    }

    fn active_items(&self) -> &[LibraryItem] {
        match self.nav.tab {
            RailTab::Games => &self.library,
            RailTab::Media => &self.library_media,
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
        self.pump_updater_events(ctx);
        self.poll_session();
        self.tick_animations(ctx);
        self.draw(ctx);
    }

    /// Kick off the startup update check (called once from `app.rs`, not
    /// from `Shell::new`, so unit tests never hit the network).
    pub fn start_update_check(&mut self) {
        self.updater_state = UpdaterState::Checking;
        updater::spawn_check(self.updater_tx.clone(), xps5x_core::VERSION.to_string());
    }

    /// Drain updater worker events into the state machine. A discovered
    /// update starts downloading immediately (updates apply only on
    /// restart, so the download is never disruptive); while a worker is in
    /// flight we keep repainting so its result shows without user input.
    fn pump_updater_events(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.updater_rx.try_recv() {
            match event {
                UpdaterEvent::UpToDate { latest } => {
                    self.updater_state = UpdaterState::UpToDate { latest };
                }
                UpdaterEvent::UpdateAvailable(info) => {
                    tracing::info!(tag = %info.tag, "update available — downloading");
                    self.updater_state = UpdaterState::Downloading {
                        tag: info.tag.clone(),
                    };
                    updater::spawn_download(self.updater_tx.clone(), info);
                }
                UpdaterEvent::Staged { tag, staged } => {
                    tracing::info!(tag = %tag, staged = %staged.display(), "update staged — restart to apply");
                    self.updater_state = UpdaterState::Staged { tag, staged };
                }
                UpdaterEvent::CheckFailed(err) => {
                    // A background update check that fails is benign: the project
                    // may have no published release yet (GitHub 404), or the host
                    // may be offline. The state is surfaced in the UI already, so
                    // logging it as a WARN on every launch is pure noise.
                    tracing::debug!(error = %err, "update check failed (non-fatal)");
                    self.updater_state = UpdaterState::Error(err);
                }
                UpdaterEvent::DownloadFailed(err) => {
                    // A download the user asked for that then failed is worth a
                    // warning — the intent was explicit.
                    tracing::warn!(error = %err, "update download failed");
                    self.updater_state = UpdaterState::Error(err);
                }
            }
        }
        if self.updater_state.is_busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(400));
        }
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
                NavAction::LaunchMedia(index) => self.confirm_media(index),
                NavAction::ActivateOption { card, option } => {
                    self.handle_cc_option(ctx, card, option)
                }
                NavAction::OpenSettings => self.enter_settings(),
                NavAction::CloseSettings => self.leave_settings(),
                NavAction::AdjustSetting {
                    section,
                    row,
                    delta,
                } => self.apply_setting_adjust(ctx, section, row, delta),
                NavAction::ActivateSetting { section, row } => {
                    self.apply_setting_activate(ctx, section, row)
                }
                NavAction::OpenControlCenter
                | NavAction::CloseControlCenter
                | NavAction::SwitchTab(_)
                | NavAction::None => {}
            }
        }
    }

    /// Confirm on a Media-tab tile (spec §10 SM2). There is no media
    /// playback engine yet, so this is a stub: just log it.
    fn confirm_media(&mut self, index: usize) {
        let title = self
            .library_media
            .get(index)
            .map(|i| i.title.as_str())
            .unwrap_or("unknown");
        tracing::info!(title, "media app confirmed (stub — no media engine yet)");
    }

    /// Confirm on the Settings tile: reset the Settings screen's scratch
    /// text-entry buffers to match the current config.
    fn enter_settings(&mut self) {
        self.settings_new_folder_input.clear();
        self.settings_key_provider_input =
            self.config.paths.key_provider_path.display().to_string();
    }

    /// Back out of Settings: fold the KeyProvider path text field back into
    /// config (it's edited free-form, not row-by-row like the other
    /// sections) and persist via the existing `EmulatorConfig::save` path.
    fn leave_settings(&mut self) {
        self.config.paths.key_provider_path =
            PathBuf::from(self.settings_key_provider_input.trim());
        if let Err(err) = self.config.save(&self.config_path) {
            // TODO: surface a save failure to the user in-UI instead of
            // just logging it (spec doesn't yet define a Settings toast/
            // error surface).
            tracing::warn!(error = %err, path = %self.config_path.display(), "failed to save settings");
        }
    }

    /// Step the config field addressed by `(section, row)` — see
    /// `settings::SETTINGS_SECTION_NAMES` for what each section is.
    /// Sections 3 (Game Folders) and 4 (Key Provider) are pure text-entry
    /// and have nothing to step with Left/Right. `ctx` is only needed by
    /// the Theme row, which reloads fonts/textures on the egui context.
    fn apply_setting_adjust(
        &mut self,
        ctx: &egui::Context,
        section: usize,
        row: usize,
        delta: i32,
    ) {
        match (section, row) {
            (0, 0) => {
                self.config.graphics.resolution_scale = settings::adjust_stepped(
                    self.config.graphics.resolution_scale,
                    delta,
                    0.25,
                    0.5,
                    4.0,
                )
            }
            (0, 1) => {
                // Apply live — the viewport command is what actually moves
                // the window in and out of fullscreen, not the config bit.
                self.config.general.fullscreen = !self.config.general.fullscreen;
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(
                    self.config.general.fullscreen,
                ));
            }
            (0, 2) => self.config.graphics.shader_cache = !self.config.graphics.shader_cache,
            (0, 3) => {
                self.config.graphics.validation_layers = !self.config.graphics.validation_layers
            }
            (0, 4) => self.config.general.vsync = !self.config.general.vsync,
            (1, 0) => self.config.audio.enabled = !self.config.audio.enabled,
            (1, 1) => {
                self.config.audio.volume =
                    settings::adjust_stepped(self.config.audio.volume, delta, 0.05, 0.0, 1.0)
            }
            (1, 2) => self.config.audio.spatial_audio = !self.config.audio.spatial_audio,
            (2, 0) => self.config.input.dualsense_features = !self.config.input.dualsense_features,
            (2, 1) => {
                self.config.input.deadzone =
                    settings::adjust_stepped(self.config.input.deadzone, delta, 0.05, 0.0, 1.0)
            }
            (5, 0) => self.cycle_theme(ctx, delta),
            (7, 0) => {
                self.config.debug.logging = !self.config.debug.logging;
                self.apply_log_settings();
            }
            (7, 1) => {
                self.config.debug.log_level =
                    settings::cycle_log_level(&self.config.debug.log_level, delta);
                self.apply_log_settings();
            }
            (7, 2) => self.config.debug.trace_syscalls = !self.config.debug.trace_syscalls,
            (7, 3) => self.config.debug.dump_gpu_commands = !self.config.debug.dump_gpu_commands,
            (7, 4) => self.config.debug.dump_shaders = !self.config.debug.dump_shaders,
            _ => {}
        }
    }

    /// Push the current Debug logging settings to the live tracing subscriber
    /// so Log Level / Logging take effect immediately, not just on next launch.
    /// Logging off silences all output (an `off` filter).
    fn apply_log_settings(&self) {
        xps5x_core::logging::set_level(if self.config.debug.logging {
            self.config.debug.log_level.as_str()
        } else {
            "off"
        });
    }

    /// Confirm on the focused Settings row. Video/Audio/Input/Theme all
    /// behave the same as an adjust with `delta: 1` (a toggle flips either
    /// way; the theme selector cycles forward) — spec's "Left/Right or
    /// Confirm to adjust". Game Folders and Key Provider have their own
    /// Confirm semantics.
    fn apply_setting_activate(&mut self, ctx: &egui::Context, section: usize, row: usize) {
        match section {
            0 | 1 | 2 | 5 | 7 => self.apply_setting_adjust(ctx, section, row, 1),
            3 => self.activate_game_folder_row(row),
            6 => self.activate_system_row(ctx, row),
            _ => {} // Key Provider (4): pure text-entry, nothing to "confirm".
        }
    }

    /// Confirm within the System section. Row 0 (Version) is display-only;
    /// row 1 is the updater's action row — what it does depends on where
    /// the state machine currently is.
    fn activate_system_row(&mut self, ctx: &egui::Context, row: usize) {
        if row != 1 {
            return;
        }
        match self.updater_state.clone() {
            UpdaterState::Idle | UpdaterState::UpToDate { .. } | UpdaterState::Error(_) => {
                self.start_update_check();
            }
            UpdaterState::Staged { staged, .. } => match updater::apply_staged(&staged) {
                Ok(()) => {
                    tracing::info!("update script launched — closing for swap");
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                Err(err) => {
                    tracing::warn!(error = %err, "failed to launch update script");
                    self.updater_state = UpdaterState::Error(err);
                }
            },
            UpdaterState::Checking | UpdaterState::Downloading { .. } => {}
        }
    }

    /// Confirm within the Game Folders section: the last row ("Add
    /// Folder") pushes the typed path; any other row removes that folder.
    fn activate_game_folder_row(&mut self, row: usize) {
        let folder_count = self.config.paths.game_folders.len();
        if row == folder_count {
            let trimmed = self.settings_new_folder_input.trim();
            if !trimmed.is_empty() {
                let path = PathBuf::from(trimmed);
                if !self.config.paths.game_folders.contains(&path) {
                    self.config.paths.game_folders.push(path);
                }
                self.settings_new_folder_input.clear();
            }
        } else if row < folder_count {
            self.config.paths.game_folders.remove(row);
        }
        self.nav
            .set_settings_row_counts(settings::settings_row_counts(
                self.config.paths.game_folders.len(),
            ));
    }

    /// Cycle `general.selected_theme` by `delta` steps through the themes
    /// installed under `themes_root` and reload the active theme from disk
    /// (spec §6, §10 SM2b: "selecting a theme updates `general.
    /// selected_theme` and reloads the active theme").
    fn cycle_theme(&mut self, ctx: &egui::Context, delta: i32) {
        let themes = settings::available_themes(&self.themes_root);
        if themes.is_empty() {
            return;
        }
        let current = themes
            .iter()
            .position(|t| t == &self.config.general.selected_theme)
            .unwrap_or(0);
        let len = themes.len() as i32;
        let next = (current as i32 + delta).rem_euclid(len) as usize;
        self.config.general.selected_theme = themes[next].clone();
        self.reload_theme(ctx);
    }

    /// Load `config.general.selected_theme` from `themes_root`, install its
    /// font (or fall back to egui's built-ins) into `ctx`, and (re)upload
    /// its background image, if any, to the GPU. Called once at
    /// construction (via [`Self::new`]) and again whenever Settings' theme
    /// selector changes the active theme.
    fn reload_theme(&mut self, ctx: &egui::Context) {
        let theme =
            theme::loader::load_theme(&self.themes_root, &self.config.general.selected_theme);
        theme::install_fonts(ctx, &theme);
        self.background_texture = background_texture_for(ctx, &theme);
        self.theme = theme;
    }

    /// Handle Confirm on a Control Center card's option list (currently
    /// only Power: Rest Mode / Restart / Turn Off — spec §10). Rest/Restart
    /// are no-op stubs for SM1; Turn Off actually closes the Shell.
    fn handle_cc_option(&mut self, ctx: &egui::Context, card: usize, option: usize) {
        let Some(item) = control_center::ITEMS.get(card) else {
            return;
        };
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

        // While an egui widget (e.g. one of Settings' text fields) holds
        // keyboard focus, let it consume typing/arrow keys instead of the
        // Shell stealing them for rail/section navigation. Gamepad input
        // never fights a text field, so it's still polled below regardless.
        let widget_has_focus = ctx.memory(|m| m.focused().is_some());

        if !widget_has_focus {
            ctx.input(|i| {
                if i.key_pressed(Key::ArrowLeft) {
                    inputs.push(NavInput::Left);
                }
                if i.key_pressed(Key::ArrowRight) {
                    inputs.push(NavInput::Right);
                }
                if i.key_pressed(Key::ArrowUp) {
                    inputs.push(NavInput::Up);
                }
                if i.key_pressed(Key::ArrowDown) {
                    inputs.push(NavInput::Down);
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
                if i.key_pressed(Key::Tab) {
                    inputs.push(NavInput::Tab);
                }
            });
        }

        if let Some(gilrs) = self.gilrs.as_mut() {
            while let Some(gilrs::Event { event, .. }) = gilrs.next_event() {
                match event {
                    gilrs::EventType::ButtonPressed(gilrs::Button::DPadLeft, _) => {
                        inputs.push(NavInput::Left)
                    }
                    gilrs::EventType::ButtonPressed(gilrs::Button::DPadRight, _) => {
                        inputs.push(NavInput::Right)
                    }
                    gilrs::EventType::ButtonPressed(gilrs::Button::DPadUp, _) => {
                        inputs.push(NavInput::Up)
                    }
                    gilrs::EventType::ButtonPressed(gilrs::Button::DPadDown, _) => {
                        inputs.push(NavInput::Down)
                    }
                    gilrs::EventType::ButtonPressed(gilrs::Button::South, _) => {
                        inputs.push(NavInput::Confirm)
                    }
                    gilrs::EventType::ButtonPressed(gilrs::Button::East, _) => {
                        inputs.push(NavInput::Back)
                    }
                    gilrs::EventType::ButtonPressed(gilrs::Button::Mode, _) => {
                        inputs.push(NavInput::Guide)
                    }
                    // Shoulder buttons (L1/R1) both toggle the two-item
                    // Games/Media tab (spec §10 SM2: "L1/R1 if easy").
                    gilrs::EventType::ButtonPressed(gilrs::Button::LeftTrigger, _) => {
                        inputs.push(NavInput::Tab)
                    }
                    gilrs::EventType::ButtonPressed(gilrs::Button::RightTrigger, _) => {
                        inputs.push(NavInput::Tab)
                    }
                    _ => {}
                }
            }
        }

        inputs
    }

    fn begin_launch(&mut self, index: usize) {
        let Some(item) = self.library.get(index) else {
            return;
        };
        match self.launcher.launch(&item.launch) {
            Ok(handle) => {
                push_recent(&mut self.recent, &item.id, MAX_RECENT_TITLES);
                self.session = Some(ActiveSession {
                    handle,
                    title: item.title.clone(),
                    target_index: index,
                });
            }
            Err(err) => {
                tracing::warn!(title = %item.title, error = %err, "launch failed");
            }
        }
    }

    fn poll_session(&mut self) {
        let Some(session) = &self.session else { return };
        // Only `Exited` clears the overlay. `Faulted` used to be lumped in
        // with `Exited` here, but that meant a synchronous fault (e.g. the
        // real firmware launcher's "no module file" case) would vanish
        // before the very first `draw` ever painted it — the user would
        // never see why. A fault now stays on screen, same as `Running`,
        // until the user presses Back (which calls `quit`, landing on
        // `Exited` on the next poll).
        if self.launcher.session_state(&session.handle) == SessionState::Exited {
            self.nav.rail_index = session.target_index;
            self.session = None;
            // Drop the last frame with the session, or the next launch opens on
            // the previous title's final image before it renders anything.
            self.frame_view.clear();
        }
    }

    fn tick_animations(&mut self, ctx: &egui::Context) {
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        let mut animating = false;

        if self.nav.rail_index != self.last_rail_index || self.nav.tab != self.last_tab {
            self.last_rail_index = self.nav.rail_index;
            self.last_tab = self.nav.tab;
            self.focus_pop.value = 0.0;
            self.focus_pop.set_target(1.0);

            // Resolve the newly-focused tile's hero gradient and its key-art
            // background in one borrow, then start both crossfades off the same
            // `hero_t` tween.
            let (new_hero, focused_bg) = {
                let item = self.active_items().get(self.nav.rail_index);
                (
                    item.map(|i| i.art.hero()),
                    item.and_then(|i| self.game_backgrounds.get(i.id.as_str()).cloned()),
                )
            };
            self.hero_bg_from = self.hero_bg_to.clone();
            self.hero_bg_to = focused_bg;
            if let Some(new_hero) = new_hero {
                self.hero_from = Some(self.blended_hero());
                self.hero_to = new_hero;
                self.hero_t.value = 0.0;
                self.hero_t.set_target(1.0);
            }
        }

        let target_offset = -(self.nav.rail_index as f32)
            * (self.theme.metrics.tile_size + self.theme.metrics.tile_gap);
        self.rail_offset.set_target(target_offset);
        animating |= self.rail_offset.tick(dt);
        animating |= self.focus_pop.tick(dt);
        animating |= self.hero_t.tick(dt);

        self.cc_open
            .set_target(if self.nav.mode == NavMode::ControlCenter {
                1.0
            } else {
                0.0
            });
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

    fn draw(&mut self, ctx: &egui::Context) {
        let theme = self.theme.clone();
        let frame = egui::Frame::NONE.fill(theme.palette.ground);

        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            if let Some(session) = &self.session {
                draw_session_overlay(
                    ui,
                    &theme,
                    session,
                    self.launcher.as_ref(),
                    &mut self.frame_view,
                );
                return;
            }

            if self.nav.mode == NavMode::Settings {
                settings::draw(
                    ui,
                    &theme,
                    &self.nav,
                    &self.config,
                    &mut self.settings_new_folder_input,
                    &mut self.settings_key_provider_input,
                    &self.updater_state,
                );
                return;
            }

            let anim = HomeAnim {
                rail_offset: self.rail_offset.value,
                hero: self.blended_hero(),
                hero_fade: anim::ease_out_cubic(self.hero_t.value),
                focus_pop: self.focus_pop.value,
            };
            let items = self.active_items();
            home::draw(
                ui,
                &theme,
                items,
                &self.nav,
                &anim,
                &self.meta_cache,
                self.background_texture.as_ref(),
                &self.cover_textures,
                self.hero_bg_from.as_ref(),
                self.hero_bg_to.as_ref(),
            );

            let recent_titles = self.recent_titles();
            control_center::draw(ui, &theme, &self.nav, self.cc_open.value, &recent_titles);
        });
    }

    /// Resolve the recent-titles id history to display titles, most-recent-
    /// first, for the Switcher panel.
    fn recent_titles(&self) -> Vec<String> {
        self.recent
            .iter()
            .filter_map(|id| {
                self.library
                    .iter()
                    .find(|item| &item.id == id)
                    .map(|item| item.title.clone())
            })
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

/// Upload `theme`'s background image (if any) to the GPU as a fresh
/// texture. `None` when the theme carries no background — `home.rs` falls
/// back to its mesh-gradient hero in that case (spec §6).
fn background_texture_for(ctx: &egui::Context, theme: &Theme) -> Option<egui::TextureHandle> {
    theme.assets.background.as_ref().map(|image| {
        ctx.load_texture(
            "xps5x-theme-background",
            (**image).clone(),
            egui::TextureOptions::LINEAR,
        )
    })
}

/// Decode + upload every scanned game's user-supplied cover image, keyed by
/// item id. Uses the theme loader's bounds-checked image path (covers are
/// untrusted content exactly like theme backgrounds); anything missing,
/// oversized, or malformed is simply skipped — that game keeps its
/// gradient + monogram tile art.
fn cover_textures_for(
    ctx: &egui::Context,
    library: &[LibraryItem],
) -> HashMap<String, egui::TextureHandle> {
    let mut textures = HashMap::new();
    for item in library {
        let Some(path) = &item.cover_path else {
            continue;
        };
        let Some(image) = theme::loader::load_image_file_capped(path) else {
            tracing::warn!(path = %path.display(), title = %item.title, "cover image failed to load — using gradient art");
            continue;
        };
        let texture = ctx.load_texture(
            format!("xps5x-cover-{}", item.id),
            image,
            egui::TextureOptions::LINEAR,
        );
        textures.insert(item.id.clone(), texture);
    }
    textures
}

/// Decode + upload each scanned game's key-art background — the title's own
/// `sce_sys/pic1.png` (or `pic0.png`), found next to its eboot — keyed by item
/// id. Same bounds-checked, skip-on-failure path as covers; a game with no key
/// art (or an app tile) simply has no entry and the Home hero falls back to its
/// mesh gradient.
fn background_textures_for(
    ctx: &egui::Context,
    library: &[LibraryItem],
) -> HashMap<String, egui::TextureHandle> {
    let mut textures = HashMap::new();
    for item in library {
        let LaunchTarget::Game { path } = &item.launch else {
            continue;
        };
        let Some(bg_path) = crate::library::scan::title_background(path) else {
            continue;
        };
        let Some(image) = theme::loader::load_image_file_capped(&bg_path) else {
            tracing::warn!(path = %bg_path.display(), title = %item.title, "background image failed to load — using gradient hero");
            continue;
        };
        let texture = ctx.load_texture(
            format!("xps5x-bg-{}", item.id),
            image,
            egui::TextureOptions::LINEAR,
        );
        textures.insert(item.id.clone(), texture);
    }
    textures
}

fn draw_session_overlay(
    ui: &mut egui::Ui,
    theme: &Theme,
    session: &ActiveSession,
    launcher: &dyn GameLauncher,
    frame_view: &mut present::GameFrameView,
) {
    let screen = ui.max_rect();
    ui.painter().rect_filled(screen, 0.0, theme.palette.ground);

    // The title's own frames, when it has rendered any. Painted before the
    // status text so the text stays legible over a bright frame.
    let presented = frame_view.paint(ui, screen);

    let state = launcher.session_state(&session.handle);
    // `session_detail` carries the engine's honest account of what actually
    // happened — a fault reason, or (for the real firmware launcher) a
    // "linked, not executed" summary — so the overlay never claims more
    // than SM3 actually does (spec: link, don't pretend to play).
    let detail = launcher.session_detail(&session.handle);
    let (headline, sub) = match state {
        SessionState::Loading => (
            format!("Launching {}…", session.title),
            "Handing off to the engine".to_string(),
        ),
        SessionState::Running => (
            session.title.clone(),
            detail.unwrap_or_else(|| "Running — Esc to return to the Shell".to_string()),
        ),
        SessionState::Faulted => (
            session.title.clone(),
            detail.unwrap_or_else(|| "Launch failed — Esc to return to the Shell".to_string()),
        ),
        SessionState::Exited => (session.title.clone(), "Returning to Shell…".to_string()),
    };

    // Once frames are arriving, the big centred title would sit on top of the
    // game. Step aside: a single dim line in the corner, so the screen is the
    // title's and the way out is still visible.
    let painter = ui.painter();
    match presented {
        present::Presented::Frame { rect } => {
            painter.text(
                egui::pos2(rect.left() + 12.0, rect.top() + 10.0),
                egui::Align2::LEFT_TOP,
                format!("{} — Esc to return to the Shell", session.title),
                egui::FontId::proportional(13.0),
                theme.palette.text_dim,
            );
        }
        present::Presented::NoFrameYet => {
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
        }
    }

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
