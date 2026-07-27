//! Shell state machine: Boot → Home (+ Control Center overlay) → launch
//! transition → back to Home (spec §3, §5, §10 — SM0 scope).
//!
//! Owns navigation state, animated values, the library, the active
//! [`GameLauncher`], and a best-effort `gilrs` gamepad connection. Frame
//! driving lives in `app.rs`; this module is where input becomes
//! [`nav::NavAction`]s and where those actions become launcher calls.

pub mod anim;
pub mod boot;
pub mod console;
pub mod control_center;
pub mod home;
pub mod icons;
pub mod ledger;
pub mod media;
pub mod nav;
pub mod per_game;
pub(crate) mod present;
pub mod settings;
pub mod sounds;

use crate::launcher::{GameLauncher, SessionHandle, SessionState};
use crate::library::{Gradient, LaunchTarget, LibraryItem, MetaCache};
use crate::theme::{self, Theme};
use crate::updater::{self, UpdaterEvent, UpdaterState};
use anim::Animated;
use boot::BootSequence;
use egui::Key;
use home::HomeAnim;
use nav::{NavAction, NavInput, NavMode, NavState, RailTab};
use raeen_core::config::EmulatorConfig;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

enum Screen {
    Boot(BootSequence),
    Home,
}

/// The selectable values for the Video ▸ Upscaler row: the `"off"` sentinel
/// followed by every registered present plugin (built-in vendor-neutral
/// plugins plus any user-supplied BYO plugin registered this run).
pub(crate) fn present_plugin_options() -> Vec<String> {
    let mut options = vec!["off".to_string()];
    options.extend(
        raeen_gpu::AgcGpuSession::present_plugins()
            .into_iter()
            .map(|(name, _caps)| name),
    );
    options
}

/// Apply the persisted present-plugin selection + upscale factor to the GPU
/// crate (the same process-wide sink the present path reads). `"off"` — and a
/// persisted plugin that is no longer registered — both restore the zero-cost
/// identity present path rather than silently pretending a plugin is active.
pub(crate) fn apply_present_plugin(graphics: &raeen_core::config::GraphicsConfig) {
    let active = graphics.upscaler != "off"
        && raeen_gpu::AgcGpuSession::select_present_plugin(&graphics.upscaler);
    if !active {
        raeen_gpu::AgcGpuSession::clear_present_plugin();
    }
    raeen_gpu::AgcGpuSession::set_present_output_scale(graphics.present_upscale);
}

/// A game session launched from Home, tracked until it exits.
struct ActiveSession {
    handle: SessionHandle,
    title: String,
    /// Library id, for the session ledger written on exit.
    item_id: String,
    /// Wall-clock start of the session, for the ledger's play time.
    started: std::time::Instant,
    /// Whether any poll observed this session in the `Faulted` state — the
    /// ledger remembers it so Home can be honest about the last session.
    faulted_seen: bool,
    /// Rail index that was focused when Play was pressed, so we return to
    /// the same tile on exit (spec §5).
    target_index: usize,
    /// The session's kernel, cached at launch so the per-frame controller
    /// push touches only the cheap `pad_state` mutex instead of re-locking the
    /// launcher's session map. `None` if the launcher shares no kernel
    /// (`StubLauncher`).
    kernel: Option<std::sync::Arc<raeen_kernel::OrbisKernel>>,
}

/// Cap on the Switcher's recent-titles history (spec §10).
const MAX_RECENT_TITLES: usize = 6;

/// User-supplied UI sound packs live under `sounds/<pack>/` (repo-root
/// relative, like `themes/` and `Games/`).
const SOUNDS_ROOT: &str = "sounds";
/// User-supplied wallpapers live directly under `wallpapers/`.
const WALLPAPERS_ROOT: &str = "wallpapers";

/// Stick deflection that registers as a menu-navigation step.
const STICK_NAV_PRESS: f32 = 0.6;
/// Deflection the stick must fall back under before it can step again —
/// lower than [`STICK_NAV_PRESS`] so jitter right at the press threshold
/// never double-steps.
const STICK_NAV_RELEASE: f32 = 0.4;

/// Turns analog left-stick motion into discrete [`NavInput`] steps, one per
/// flick: crossing [`STICK_NAV_PRESS`] emits, and nothing emits again on that
/// axis until the stick returns under [`STICK_NAV_RELEASE`] (or flicks to the
/// opposite side). Pure and egui/gilrs-free so it is unit-testable.
#[derive(Debug, Default)]
struct StickNav {
    latched_x: i8,
    latched_y: i8,
}

impl StickNav {
    /// One axis step: `-1`/`1` on a fresh press past the threshold, `None`
    /// while held, released, or jittering between the two thresholds.
    fn step_axis(latched: &mut i8, value: f32) -> Option<i8> {
        let dir = if value >= STICK_NAV_PRESS {
            1
        } else if value <= -STICK_NAV_PRESS {
            -1
        } else if value.abs() < STICK_NAV_RELEASE {
            0
        } else {
            // Between release and press: keep whatever is latched.
            *latched
        };
        let fresh = dir != 0 && dir != *latched;
        *latched = dir;
        fresh.then_some(dir)
    }

    /// Horizontal stick motion (`+1.0` = right) → Left/Right.
    fn update_x(&mut self, value: f32) -> Option<NavInput> {
        Self::step_axis(&mut self.latched_x, value).map(|dir| {
            if dir < 0 {
                NavInput::Left
            } else {
                NavInput::Right
            }
        })
    }

    /// Vertical stick motion (gilrs reports up as `+1.0`) → Up/Down.
    fn update_y(&mut self, value: f32) -> Option<NavInput> {
        Self::step_axis(&mut self.latched_y, value).map(|dir| {
            if dir < 0 {
                NavInput::Down
            } else {
                NavInput::Up
            }
        })
    }
}

/// Seconds the PS/Guide button must be held during a session to quit back to
/// the Shell. A hold (not a tap) so the button still reaches the guest for its
/// normal in-game use; only a deliberate press-and-hold exits.
const SESSION_QUIT_HOLD_SECS: f32 = 0.9;

/// Tracks a press-and-hold on the in-session quit button, firing once when the
/// hold crosses [`SESSION_QUIT_HOLD_SECS`]. Our pad-driven answer to SharpEmu's
/// "controller B/O closes the game" (PR #415): where SharpEmu forwards a face
/// button, we use a PS-button hold so a stray press never yanks the player out
/// mid-game, and the overlay shows the hold filling before it commits.
#[derive(Debug, Default)]
struct QuitHold {
    held_for: f32,
}

impl QuitHold {
    /// Advance by `dt` seconds given whether the quit button is held this frame.
    /// Returns `true` exactly on the frame the hold crosses the threshold.
    fn update(&mut self, held: bool, dt: f32) -> bool {
        if !held {
            self.held_for = 0.0;
            return false;
        }
        let was = self.held_for;
        self.held_for += dt;
        was < SESSION_QUIT_HOLD_SECS && self.held_for >= SESSION_QUIT_HOLD_SECS
    }

    /// Fraction of the hold completed, `0.0..=1.0` — drives the overlay's fill.
    fn progress(&self) -> f32 {
        (self.held_for / SESSION_QUIT_HOLD_SECS).clamp(0.0, 1.0)
    }

    fn reset(&mut self) {
        self.held_for = 0.0;
    }
}

/// The full Shell: navigation, animation, library, and the launcher seam.
pub struct Shell {
    theme: Theme,
    library: Vec<LibraryItem>,
    /// The Media tab's rail (spec §10 SM2) — built once, same as `library`.
    library_media: Vec<LibraryItem>,
    meta_cache: MetaCache,
    /// Real per-title play history (last played, time played, last fault),
    /// loaded once at startup and refreshed on every session launch/exit.
    ledgers: HashMap<String, ledger::TitleLedger>,
    nav: NavState,
    screen: Screen,
    launcher: Box<dyn GameLauncher>,
    session: Option<ActiveSession>,
    /// Shows the running title's rendered frames. Persistent because it owns
    /// the GPU texture the frame is uploaded into.
    frame_view: present::GameFrameView,
    gilrs: Option<gilrs::Gilrs>,
    /// Native, mapping-DB-free controller readers (XInput + raw-HID
    /// DualSense) running on background threads. Merged into the guest pad
    /// state ahead of gilrs and the keyboard, so pads gilrs rejects with an
    /// all-zeros UUID (Steam Input / DS4Windows / generic HID) still work.
    native_input: raeen_input::NativeGamepads,
    /// Latched left-stick state so analog flicks navigate menus like the
    /// D-pad, one step per flick (see [`StickNav`]).
    stick_nav: StickNav,
    /// The active UI sound pack (Settings ▸ Audio ▸ UI Sound Pack) — the
    /// silent pack when `"off"` or nothing decodes.
    sound_pack: sounds::SoundPack,
    /// Toast notifications (egui-notify): save failures, rescan results,
    /// update staged — surfaced in-UI instead of only in the log.
    toasts: egui_notify::Toasts,
    /// Filesystem watcher over the configured game folders; dropping a game
    /// in (or deleting one) rescans the library without touching Settings.
    /// `None` when the watcher backend failed — manual rescan still works.
    library_watcher: Option<notify::RecommendedWatcher>,
    /// Set by the watcher thread on any create/modify/remove under a game
    /// folder; drained by [`Self::tick_library_watcher`].
    library_dirty: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Time of the most recent watcher event — the rescan debounce anchor,
    /// so a multi-file copy triggers one rescan at the end, not dozens.
    library_dirty_since: Option<std::time::Instant>,
    /// egui's wgpu render state, captured on the first frame — the device/
    /// queue the guest-frame native present path uploads through.
    render_state: Option<eframe::egui_wgpu::RenderState>,
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

    /// The per-game overrides currently being edited in the Game Options
    /// overlay, and the library id they belong to. Loaded from disk when the
    /// overlay opens and persisted when it closes (see [`per_game`]).
    game_options_draft: per_game::PerGameSettings,
    game_options_target_id: Option<String>,
    /// Press-and-hold state for the in-session pad quit (SharpEmu PR #415).
    session_quit_hold: QuitHold,

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

    /// In-app log console (F10), SharpEmu-style — the terminal is optional.
    console: console::ConsolePane,
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
        let ledger_dir = ledger::store_dir(&config_path);
        let ledgers: HashMap<String, ledger::TitleLedger> = library
            .iter()
            .map(|item| (item.id.clone(), ledger::load(&ledger_dir, &item.id)))
            .collect();

        let library_media = media::media_items();
        // Settings is always one of the built-in apps (spec §10 SM2); if a
        // caller ever hands the Shell a library without one, Confirm on
        // that index simply never matches and nothing opens Settings.
        let settings_tile_index = library.iter().position(|item| item.id == "settings");
        let settings_row_counts = settings::settings_row_counts(
            config.paths.game_folders.len(),
            raeen_gpu::AgcGpuSession::present_plugins().len(),
        );
        let nav = NavState::with_cc_options(rail_len, cc_len, cc_option_counts)
            .with_settings(settings_tile_index, settings_row_counts)
            .with_media_rail_len(library_media.len())
            .with_game_options(per_game::ROW_COUNT);

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
        let background_texture = wallpaper_texture_for(ctx, &config.general.wallpaper)
            .or_else(|| background_texture_for(ctx, &theme));
        let cover_textures = cover_textures_for(ctx, &library);
        let game_backgrounds = background_textures_for(ctx, &library);
        // Seed the hero background with the initially-focused tile's key art.
        let hero_bg_to = library
            .first()
            .and_then(|i| game_backgrounds.get(i.id.as_str()).cloned());

        let mut shell = Self {
            theme,
            library,
            library_media,
            meta_cache,
            ledgers,
            nav,
            screen: Screen::Boot(BootSequence::new()),
            launcher,
            session: None,
            frame_view: present::GameFrameView::default(),
            gilrs,
            native_input: raeen_input::NativeGamepads::start(),
            stick_nav: StickNav::default(),
            sound_pack: sounds::SoundPack::load(
                std::path::Path::new(SOUNDS_ROOT),
                &config.general.sound_pack,
            ),
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
            game_options_draft: per_game::PerGameSettings::default(),
            game_options_target_id: None,
            session_quit_hold: QuitHold::default(),
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
            console: console::ConsolePane::default(),
            // Top-LEFT anchor: the Shell's top-left is free chrome space, and
            // a windowed shell wider than the display (seen in the field)
            // pushes the right edge — and any right-anchored toast — off
            // screen entirely.
            toasts: egui_notify::Toasts::default()
                .with_anchor(egui_notify::Anchor::TopLeft)
                .with_margin(egui::vec2(16.0, 16.0)),
            library_watcher: None,
            library_dirty: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            library_dirty_since: None,
            render_state: None,
        };
        shell.rebuild_library_watcher(ctx);
        shell
    }

    /// (Re)attach the filesystem watcher to the current game-folder list.
    /// Called at construction and whenever the folder list changes. A missing
    /// folder is skipped quietly (it may be created later — a rebuild then
    /// picks it up); a failed watcher backend downgrades to manual rescans.
    fn rebuild_library_watcher(&mut self, ctx: &egui::Context) {
        use notify::Watcher;
        self.library_watcher = None;
        let flag = std::sync::Arc::clone(&self.library_dirty);
        let repaint = ctx.clone();
        let handler = move |event: Result<notify::Event, notify::Error>| {
            if let Ok(event) = event
                && matches!(
                    event.kind,
                    notify::EventKind::Create(_)
                        | notify::EventKind::Modify(_)
                        | notify::EventKind::Remove(_)
                )
            {
                flag.store(true, Ordering::Relaxed);
                // Wake the UI loop so the debounce timer runs while idle.
                repaint.request_repaint();
            }
        };
        let mut watcher = match notify::recommended_watcher(handler) {
            Ok(watcher) => watcher,
            Err(err) => {
                tracing::warn!(error = %err, "game-folder watcher unavailable — rescan manually");
                return;
            }
        };
        let mut watched = 0usize;
        for folder in &self.config.paths.game_folders {
            match watcher.watch(folder, notify::RecursiveMode::Recursive) {
                Ok(()) => watched += 1,
                Err(err) => {
                    tracing::warn!(folder = %folder.display(), error = %err, "game folder not watched");
                }
            }
        }
        tracing::info!(
            watched,
            of = self.config.paths.game_folders.len(),
            "game-folder watcher attached"
        );
        self.library_watcher = Some(watcher);
    }

    /// Debounced reaction to watcher events: rescan once, 800 ms after the
    /// *last* filesystem event, so bulk copies coalesce into one rescan.
    fn tick_library_watcher(&mut self, ctx: &egui::Context) {
        if self.library_dirty.swap(false, Ordering::Relaxed) {
            self.library_dirty_since = Some(std::time::Instant::now());
        }
        let Some(since) = self.library_dirty_since else {
            return;
        };
        if since.elapsed() >= std::time::Duration::from_millis(800) {
            self.library_dirty_since = None;
            self.rescan_library(ctx);
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    fn active_items(&self) -> &[LibraryItem] {
        match self.nav.tab {
            RailTab::Games => &self.library,
            RailTab::Media => &self.library_media,
        }
    }

    /// Drive one frame: advance boot/animation state, route input, poll any
    /// active session, and draw. `render_state` (when the backend is wgpu)
    /// enables the zero-conversion guest-frame upload in [`present`].
    pub fn update(
        &mut self,
        ctx: &egui::Context,
        render_state: Option<&eframe::egui_wgpu::RenderState>,
    ) {
        if self.render_state.is_none() {
            self.render_state = render_state.cloned();
        }
        if let Screen::Boot(boot) = &self.screen {
            boot::draw(ctx, &self.theme, boot);
            if boot.is_done() {
                self.screen = Screen::Home;
            } else {
                return;
            }
        }

        // F10 toggles the log console from any screen (including in-game) —
        // checked before nav routing so nothing can shadow it.
        if ctx.input(|i| i.key_pressed(Key::F10)) {
            self.console.open = !self.console.open;
        }
        // F11 toggles fullscreen from any screen, same effect as Settings ▸
        // Video ▸ Fullscreen (which also keeps the config bit in sync).
        if ctx.input(|i| i.key_pressed(Key::F11)) {
            self.apply_setting_adjust(ctx, settings::SECTION_VIDEO, 1, 1);
        }

        self.route_input(ctx);
        self.pump_updater_events(ctx);
        self.tick_library_watcher(ctx);
        self.poll_session();
        self.push_pad_state(ctx);
        self.tick_session_quit(ctx);
        self.tick_animations(ctx);
        self.draw(ctx);
        // Drawn last: the console floats above every screen, toasts above it.
        self.console.ui(ctx);
        self.toasts.show(ctx);
    }

    /// Kick off the startup update check (called once from `app.rs`, not
    /// from `Shell::new`, so unit tests never hit the network).
    pub fn start_update_check(&mut self) {
        self.updater_state = UpdaterState::Checking;
        updater::spawn_check(self.updater_tx.clone(), raeen_core::VERSION.to_string());
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
                    self.toasts
                        .success(format!("Update {tag} downloaded — restart to apply"));
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
            let action = self.nav.apply(input);
            self.play_nav_sound(input, action);
            match action {
                NavAction::Launch(index) => self.begin_launch(index),
                NavAction::LaunchMedia(index) => self.confirm_media(index),
                NavAction::ActivateOption { card, option } => {
                    self.handle_cc_option(ctx, card, option)
                }
                NavAction::OpenSettings => self.enter_settings(),
                NavAction::CloseSettings => self.leave_settings(),
                NavAction::OpenGameOptions { index } => self.open_game_options(index),
                NavAction::CloseGameOptions => self.close_game_options(),
                NavAction::AdjustGameOption { row, delta } => {
                    self.game_options_draft.adjust(row, delta, &self.config)
                }
                NavAction::ActivateGameOption { row } => {
                    self.game_options_draft.toggle_override(row, &self.config)
                }
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

    /// Voice one navigation step through the active UI sound pack. Launch is
    /// deliberately absent — `begin_launch` plays the launch cue itself so
    /// pointer-initiated launches sound identical to pad ones.
    fn play_nav_sound(&self, input: NavInput, action: NavAction) {
        use sounds::UiSound;
        let sound = match action {
            NavAction::Launch(_) => return,
            NavAction::CloseControlCenter
            | NavAction::CloseSettings
            | NavAction::CloseGameOptions => UiSound::Back,
            _ => match input {
                NavInput::Left
                | NavInput::Right
                | NavInput::Up
                | NavInput::Down
                | NavInput::Tab => UiSound::Move,
                NavInput::Confirm | NavInput::Options | NavInput::Guide => UiSound::Confirm,
                NavInput::Back => UiSound::Back,
            },
        };
        self.sound_pack.play(sound);
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
            tracing::warn!(error = %err, path = %self.config_path.display(), "failed to save settings");
            self.toasts
                .error(format!("Settings could not be saved: {err}"));
        }
    }

    /// Directory this Shell's per-game override files live in
    /// (`<config_dir>/per_game`).
    fn per_game_dir(&self) -> PathBuf {
        per_game::PerGameSettings::store_dir(&self.config_path)
    }

    /// Open the Game Options overlay for the Games-rail item at `index`,
    /// loading its persisted overrides into the editable draft.
    fn open_game_options(&mut self, index: usize) {
        let Some(item) = self.library.get(index) else {
            self.nav.mode = nav::NavMode::Home;
            return;
        };
        let id = item.id.clone();
        self.game_options_draft = per_game::PerGameSettings::load(&self.per_game_dir(), &id);
        self.game_options_target_id = Some(id);
    }

    /// Persist the edited per-game overrides (an all-inherit draft deletes the
    /// file — see [`per_game::PerGameSettings::save`]).
    fn close_game_options(&mut self) {
        if let Some(id) = self.game_options_target_id.take() {
            self.game_options_draft.save(&self.per_game_dir(), &id);
        }
        self.game_options_draft = per_game::PerGameSettings::default();
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
                if self.config.general.fullscreen {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
                } else {
                    // Restore all normal-window properties explicitly. Some
                    // Windows compositors retain the borderless fullscreen
                    // geometry after only clearing the fullscreen flag.
                    ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Resizable(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                        self.config.general.window_width as f32,
                        self.config.general.window_height as f32,
                    )));
                    ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(
                        40.0, 40.0,
                    )));
                }
            }
            (0, 2) => self.config.graphics.shader_cache = !self.config.graphics.shader_cache,
            (0, 3) => {
                self.config.graphics.validation_layers = !self.config.graphics.validation_layers
            }
            (0, 4) => self.config.general.vsync = !self.config.general.vsync,
            (0, 5) => {
                self.config.graphics.frame_limit =
                    settings::cycle_frame_limit(self.config.graphics.frame_limit, delta)
            }
            (0, 6) => {
                self.config.graphics.gpu_device_index = settings::adjust_stepped_u32(
                    self.config.graphics.gpu_device_index,
                    delta,
                    1,
                    0,
                    8,
                )
            }
            (0, 7) => {
                self.config.general.window_width = settings::adjust_stepped_u32(
                    self.config.general.window_width,
                    delta,
                    160,
                    640,
                    7680,
                )
            }
            (0, 8) => {
                self.config.general.window_height = settings::adjust_stepped_u32(
                    self.config.general.window_height,
                    delta,
                    90,
                    480,
                    4320,
                )
            }
            (0, 9) => {
                // Cycle the present plugin (upscaler / frame gen) and apply live.
                let options = present_plugin_options();
                self.config.graphics.upscaler =
                    settings::cycle_upscaler(&self.config.graphics.upscaler, delta, &options);
                apply_present_plugin(&self.config.graphics);
            }
            (0, 10) => {
                self.config.graphics.present_upscale = settings::adjust_stepped(
                    self.config.graphics.present_upscale,
                    delta,
                    0.25,
                    1.0,
                    4.0,
                );
                apply_present_plugin(&self.config.graphics);
            }
            (1, 0) => {
                self.config.audio.enabled = !self.config.audio.enabled;
                raeen_audio::output::set_enabled(self.config.audio.enabled);
            }
            (1, 1) => {
                self.config.audio.volume =
                    settings::adjust_stepped(self.config.audio.volume, delta, 0.05, 0.0, 1.0);
                raeen_audio::output::set_volume(self.config.audio.volume);
            }
            (1, 2) => self.config.audio.spatial_audio = !self.config.audio.spatial_audio,
            (1, 3) => {
                let packs = sounds::available_packs(std::path::Path::new(SOUNDS_ROOT));
                self.config.general.sound_pack =
                    settings::cycle_option(&self.config.general.sound_pack, delta, &packs);
                self.sound_pack = sounds::SoundPack::load(
                    std::path::Path::new(SOUNDS_ROOT),
                    &self.config.general.sound_pack,
                );
                // Audible preview so cycling packs is self-demonstrating.
                self.sound_pack.play(sounds::UiSound::Confirm);
            }
            (2, 0) => self.config.input.dualsense_features = !self.config.input.dualsense_features,
            (2, 1) => {
                self.config.input.deadzone =
                    settings::adjust_stepped(self.config.input.deadzone, delta, 0.05, 0.0, 1.0)
            }
            (2, 2) => {
                self.config.input.controller_icon_style =
                    self.config.input.controller_icon_style.cycle(delta)
            }
            (settings::SECTION_THEME, 0) => self.cycle_theme(ctx, delta),
            (settings::SECTION_THEME, 1) => {
                let options = settings::available_wallpapers(std::path::Path::new(WALLPAPERS_ROOT));
                self.config.general.wallpaper =
                    settings::cycle_option(&self.config.general.wallpaper, delta, &options);
                self.refresh_background(ctx);
            }
            (settings::SECTION_ADVANCED, 0) => {
                self.config.debug.logging = !self.config.debug.logging;
                self.apply_log_settings();
            }
            (settings::SECTION_ADVANCED, 1) => {
                self.config.debug.log_level =
                    settings::cycle_log_level(&self.config.debug.log_level, delta);
                self.apply_log_settings();
            }
            (settings::SECTION_ADVANCED, 2) => {
                self.config.debug.trace_syscalls = !self.config.debug.trace_syscalls
            }
            (settings::SECTION_ADVANCED, 3) => {
                self.config.debug.dump_gpu_commands = !self.config.debug.dump_gpu_commands
            }
            (settings::SECTION_ADVANCED, 4) => {
                self.config.debug.dump_shaders = !self.config.debug.dump_shaders
            }
            (settings::SECTION_ADVANCED, 5) => {
                self.config.debug.dump_frames = !self.config.debug.dump_frames
            }
            (settings::SECTION_ADVANCED, 6) => {
                self.config.debug.call_stats = !self.config.debug.call_stats
            }
            (settings::SECTION_ADVANCED, 7) => {
                self.config.debug.stall_dump = !self.config.debug.stall_dump
            }
            _ => {}
        }
    }

    /// Push the current Debug logging settings to the live tracing subscriber
    /// so Log Level / Logging take effect immediately, not just on next launch.
    /// Logging off silences all output (an `off` filter).
    fn apply_log_settings(&self) {
        raeen_core::logging::set_level(if self.config.debug.logging {
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
            settings::SECTION_VIDEO
            | settings::SECTION_AUDIO
            | settings::SECTION_CONTROLLER
            | settings::SECTION_THEME
            | settings::SECTION_ADVANCED => self.apply_setting_adjust(ctx, section, row, 1),
            settings::SECTION_GAME_FOLDERS => self.activate_game_folder_row(ctx, row),
            settings::SECTION_PLUGINS => self.activate_plugin_row(row),
            settings::SECTION_SYSTEM => self.activate_system_row(ctx, row),
            _ => {} // Key Provider: pure text-entry, nothing to "confirm".
        }
    }

    /// Confirm within the Plugins section: a plugin row toggles that plugin
    /// active/inactive (and applies it live); the two trailing action rows
    /// rescan `plugins/` and open it in the host file manager.
    fn activate_plugin_row(&mut self, row: usize) {
        let plugins = raeen_gpu::AgcGpuSession::present_plugin_infos();
        if let Some(plugin) = plugins.get(row) {
            self.config.graphics.upscaler = if self.config.graphics.upscaler == plugin.name {
                "off".to_string()
            } else {
                plugin.name.clone()
            };
            apply_present_plugin(&self.config.graphics);
        } else if row == plugins.len() {
            self.rescan_plugins();
        } else if row == plugins.len() + 1 {
            open_plugins_folder();
        }
    }

    /// Re-scan `plugins/` for out-of-tree present plugins without a restart.
    /// New binaries register, refusals are re-recorded for the Plugins UI, and
    /// the section's row count follows the registry.
    fn rescan_plugins(&mut self) {
        // SAFETY: same trust boundary as the startup load in `main.rs` —
        // `plugins/` is the documented, user-controlled BYO plugin directory,
        // and rescanning is an explicit user action on that same directory.
        let loaded = unsafe {
            raeen_gpu::AgcGpuSession::load_present_plugins_from(std::path::Path::new("plugins"))
        };
        tracing::info!(count = loaded.len(), plugins = ?loaded, "plugins folder rescanned");
        self.toasts.info(if loaded.is_empty() {
            "Plugins rescanned — nothing new loaded".to_string()
        } else {
            format!("Plugins rescanned — loaded: {}", loaded.join(", "))
        });
        // The persisted selection may name a plugin that just (re)appeared.
        apply_present_plugin(&self.config.graphics);
        self.refresh_settings_row_counts();
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

    /// Confirm within the Game Folders section: a folder row removes that
    /// folder; "Add Folder" pushes the typed path; "Browse & Add Folder"
    /// opens the system folder picker; "Rescan Games" re-reads the folders.
    /// Any folder change rescans the library immediately, so the Home rail
    /// always reflects the folder list Settings shows.
    fn activate_game_folder_row(&mut self, ctx: &egui::Context, row: usize) {
        let folder_count = self.config.paths.game_folders.len();
        let mut folders_changed = false;
        if row < folder_count {
            self.config.paths.game_folders.remove(row);
            folders_changed = true;
        } else if row == folder_count {
            let trimmed = self.settings_new_folder_input.trim();
            if !trimmed.is_empty() {
                folders_changed = self.add_game_folder(PathBuf::from(trimmed));
                self.settings_new_folder_input.clear();
            }
        } else if row == folder_count + 1 {
            // Native folder picker. Blocks the UI thread for the dialog's
            // lifetime — standard native-app behavior, and the Shell has no
            // background work a modal pick would starve.
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Add Game Folder")
                .pick_folder()
            {
                folders_changed = self.add_game_folder(path);
            }
        } else if row == folder_count + 2 {
            self.rescan_library(ctx);
        }
        if folders_changed {
            self.refresh_settings_row_counts();
            self.rescan_library(ctx);
            // The watcher's folder set just changed with the config.
            self.rebuild_library_watcher(ctx);
        }
    }

    /// Append a game folder if it is not already configured. Returns whether
    /// the list changed.
    fn add_game_folder(&mut self, path: PathBuf) -> bool {
        if self.config.paths.game_folders.contains(&path) {
            return false;
        }
        self.config.paths.game_folders.push(path);
        true
    }

    /// Re-scan the configured game folders and swap the Home library in
    /// place: items, metadata, ledgers, cover/key-art textures, and the nav
    /// rail follow the new list; navigation mode and Settings focus are
    /// untouched, so a rescan from Settings stays in Settings.
    fn rescan_library(&mut self, ctx: &egui::Context) {
        let mut library = crate::app::scan_game_folders(&self.config.paths.game_folders);
        let scanned = library.len();
        if library.is_empty() {
            library = crate::library::sample_library();
            self.toasts
                .info("Library rescanned — no games found (showing samples)");
        } else {
            library.extend(crate::library::built_in_apps());
            self.toasts
                .success(format!("Library rescanned — {scanned} titles"));
        }

        self.meta_cache = MetaCache::from_items(&library);
        let ledger_dir = self.ledger_dir();
        self.ledgers = library
            .iter()
            .map(|item| (item.id.clone(), ledger::load(&ledger_dir, &item.id)))
            .collect();
        self.cover_textures = cover_textures_for(ctx, &library);
        self.game_backgrounds = background_textures_for(ctx, &library);
        let settings_tile_index = library.iter().position(|item| item.id == "settings");
        self.nav.set_games_rail(library.len(), settings_tile_index);
        self.library = library;
        // Old texture handles may be stale — force the hero refresh path in
        // `tick_animations` to re-resolve the focused tile's art.
        self.last_rail_index = usize::MAX;
        tracing::info!(count = self.library.len(), "game library rescanned");
    }

    /// Re-derive the Settings nav's per-section row counts from the live
    /// folder and plugin counts (both sections grow and shrink at runtime).
    fn refresh_settings_row_counts(&mut self) {
        self.nav
            .set_settings_row_counts(settings::settings_row_counts(
                self.config.paths.game_folders.len(),
                raeen_gpu::AgcGpuSession::present_plugins().len(),
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
        self.theme = theme;
        self.refresh_background(ctx);
    }

    /// Resolve the Home background: the configured wallpaper wins, else the
    /// active theme's own background, else none (mesh-gradient hero).
    fn refresh_background(&mut self, ctx: &egui::Context) {
        self.background_texture = wallpaper_texture_for(ctx, &self.config.general.wallpaper)
            .or_else(|| background_texture_for(ctx, &self.theme));
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
                tracing::info!("Turn Off requested — closing Raeen");
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
                // WASD mirrors the arrows for keyboard-first users; both are
                // guarded by `widget_has_focus`, so typing in a text field
                // never navigates.
                if i.key_pressed(Key::ArrowLeft) || i.key_pressed(Key::A) {
                    inputs.push(NavInput::Left);
                }
                if i.key_pressed(Key::ArrowRight) || i.key_pressed(Key::D) {
                    inputs.push(NavInput::Right);
                }
                if i.key_pressed(Key::ArrowUp) || i.key_pressed(Key::W) {
                    inputs.push(NavInput::Up);
                }
                if i.key_pressed(Key::ArrowDown) || i.key_pressed(Key::S) {
                    inputs.push(NavInput::Down);
                }
                if i.key_pressed(Key::Enter) || i.key_pressed(Key::Space) {
                    inputs.push(NavInput::Confirm);
                }
                if i.key_pressed(Key::Escape) || i.key_pressed(Key::Backspace) {
                    inputs.push(NavInput::Back);
                }
                if i.key_pressed(Key::C) {
                    inputs.push(NavInput::Guide);
                }
                if i.key_pressed(Key::Tab) {
                    inputs.push(NavInput::Tab);
                }
                if i.key_pressed(Key::O) {
                    inputs.push(NavInput::Options);
                }
                if i.pointer.secondary_clicked() {
                    inputs.push(NavInput::Back);
                }
                let scroll = i.smooth_scroll_delta;
                if scroll.y > 1.0 {
                    inputs.push(NavInput::Up);
                } else if scroll.y < -1.0 {
                    inputs.push(NavInput::Down);
                }
                if scroll.x > 1.0 {
                    inputs.push(NavInput::Left);
                } else if scroll.x < -1.0 {
                    inputs.push(NavInput::Right);
                }
            });
        }

        // The gamepad drives menu navigation only outside a session; while a
        // title runs, `push_pad_state` owns the gamepad and forwards it to the
        // guest, so face buttons like Cross/Circle reach the game rather than
        // the Shell (only the keyboard Esc quits back to the dashboard).
        if self.session.is_none()
            && let Some(gilrs) = self.gilrs.as_mut()
        {
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
                    // Triangle/North (the "Options" affordance on the Home
                    // button-hint bar) opens the focused game's per-game
                    // settings overlay. Start is the DualSense Options button
                    // itself, so it does the same.
                    gilrs::EventType::ButtonPressed(gilrs::Button::North, _)
                    | gilrs::EventType::ButtonPressed(gilrs::Button::Start, _) => {
                        inputs.push(NavInput::Options)
                    }
                    // Shoulder buttons (L1/R1) both toggle the two-item
                    // Games/Media tab (spec §10 SM2: "L1/R1 if easy").
                    gilrs::EventType::ButtonPressed(gilrs::Button::LeftTrigger, _) => {
                        inputs.push(NavInput::Tab)
                    }
                    gilrs::EventType::ButtonPressed(gilrs::Button::RightTrigger, _) => {
                        inputs.push(NavInput::Tab)
                    }
                    // Left stick navigates menus like the D-pad: a flick past
                    // the press threshold emits one step; the stick must
                    // return toward center before it can emit again.
                    gilrs::EventType::AxisChanged(gilrs::Axis::LeftStickX, value, _) => {
                        inputs.extend(self.stick_nav.update_x(value));
                    }
                    gilrs::EventType::AxisChanged(gilrs::Axis::LeftStickY, value, _) => {
                        inputs.extend(self.stick_nav.update_y(value));
                    }
                    _ => {}
                }
            }
        }

        inputs
    }

    /// Forward the physical gamepad's live state into the running guest each
    /// frame — the input producer that makes the guest's `scePadReadState`
    /// return real input. Analog sticks get Settings ▸ Controller ▸ Deadzone
    /// applied; the encoded 12-byte `ScePadData` prefix is written to the local
    /// session kernel and to the isolated runner's shared-memory input slot.
    /// This keeps one canonical host-device reader in the Shell while ensuring
    /// the child process's guest kernel receives the same snapshot.
    fn push_pad_state(&mut self, ctx: &egui::Context) {
        let (kernel, deadzone) = match self.session.as_ref() {
            Some(session) => (session.kernel.clone(), self.config.input.deadzone),
            None => return,
        };
        // Highest-priority source: the native, mapping-DB-free readers
        // (XInput + raw-HID DualSense, SharpEmu-ported). These recover pads
        // gilrs rejects with an all-zeros UUID (Steam Input / DS4Windows /
        // generic HID), where `read_pad`'s `is_pressed` never fires. Sticks
        // arrive raw, so apply the same configured deadzone as the gilrs path.
        let native = self.native_input.poll().map(|mut s| {
            s.left_stick_x = apply_deadzone(s.left_stick_x, deadzone);
            s.left_stick_y = apply_deadzone(s.left_stick_y, deadzone);
            s.right_stick_x = apply_deadzone(s.right_stick_x, deadzone);
            s.right_stick_y = apply_deadzone(s.right_stick_y, deadzone);
            s
        });
        // Live gamepad state — neutral if no controller is connected OR if gilrs
        // has no button mapping for it (the all-zeros-UUID / Steam-Input / generic
        // HID case, where `is_pressed(Button::South)` never fires). We no longer
        // early-return when gilrs is absent, so the keyboard path below still runs.
        let pad = if let Some(gilrs) = self.gilrs.as_mut() {
            // Drain events so gilrs's cached per-pad state is current this frame;
            // during a session `poll_nav_inputs` leaves the gamepad to us.
            while gilrs.next_event().is_some() {}
            match gilrs.gamepads().next() {
                Some((_, gamepad)) => read_pad(&gamepad, deadzone),
                None => raeen_input::ControllerState::default(),
            }
        } else {
            raeen_input::ControllerState::default()
        };
        // Merge priority: native → gilrs → keyboard. `merge_pad_states`
        // OR-merges buttons and prefers the first non-zero analog axis, so
        // ordering native first gives it stick priority; the keyboard fallback
        // still reaches the guest when nothing is mapped/connected. Only active
        // in-session (we early-returned above otherwise), so it can never fight
        // Shell navigation.
        let merged = merge_pad_states(
            native.unwrap_or_default(),
            merge_pad_states(pad, read_keyboard_pad(ctx)),
        );
        let encoded = merged.to_orbis_pad_data();
        if let Some(kernel) = kernel {
            kernel.set_pad_state(encoded);
        }
        raeen_gpu::frame_ipc::publish_pad_state(encoded);
    }

    /// While a title runs, hold the PS/Guide button to quit back to the Shell
    /// (SharpEmu PR #415, pad-driven). No-op outside a session. `push_pad_state`
    /// has already drained gilrs this frame, so the cached button state is
    /// current; the button is still forwarded to the guest meanwhile, so this
    /// never steals a normal in-game press.
    fn tick_session_quit(&mut self, ctx: &egui::Context) {
        if self.session.is_none() {
            self.session_quit_hold.reset();
            return;
        }
        let held = self
            .gilrs
            .as_ref()
            .and_then(|g| g.gamepads().next())
            .is_some_and(|(_, gamepad)| gamepad.is_pressed(gilrs::Button::Mode));
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        if self.session_quit_hold.update(held, dt) {
            if let Some(session) = &self.session {
                tracing::info!(title = %session.title, "PS-button hold — quitting to Shell");
                let _ = self.launcher.quit(&session.handle);
            }
            self.session_quit_hold.reset();
        }
        // Keep repainting while the hold fills so the overlay ring animates
        // without needing another input event.
        if held {
            ctx.request_repaint();
        }
    }

    fn begin_launch(&mut self, index: usize) {
        let Some(item) = self.library.get(index) else {
            return;
        };

        // Apply this title's per-game overrides on top of the global config,
        // then push the effective graphics/logging settings into the same
        // process-wide sinks the Shell uses for global settings. A title with
        // no overrides yields the global config unchanged — which also cleanly
        // resets any previous title's overrides back to baseline.
        let effective =
            per_game::PerGameSettings::load(&self.per_game_dir(), &item.id).effective(&self.config);
        raeen_gpu::AgcGpuSession::set_runtime_config(
            effective.graphics.validation_layers,
            effective.graphics.resolution_scale,
            effective.graphics.gpu_device_index,
            effective.graphics.shader_cache,
            effective.paths.shader_cache_dir.clone(),
        );
        apply_present_plugin(&effective.graphics);
        raeen_core::logging::set_level(if effective.debug.logging {
            effective.debug.log_level.as_str()
        } else {
            "off"
        });
        // Stage the Advanced dump/trace toggles + Frame Limit for the isolated
        // runner child, so "applies on the next launch" is actually true and
        // per-game overrides reach the guest process.
        crate::launcher::stage_runner_env(&effective);

        self.session_quit_hold.reset();
        match self.launcher.launch(&item.launch) {
            Ok(handle) => {
                self.sound_pack.play(sounds::UiSound::Launch);
                push_recent(&mut self.recent, &item.id, MAX_RECENT_TITLES);
                // Session ledger: stamp the launch immediately (a crash later
                // must not lose the "last played" fact).
                let ledger_dir = self.ledger_dir();
                let mut title_ledger = ledger::load(&ledger_dir, &item.id);
                title_ledger.last_played = ledger::now_unix();
                ledger::store(&ledger_dir, &item.id, &title_ledger);
                self.ledgers.insert(item.id.clone(), title_ledger);
                let kernel = self.launcher.session_kernel(&handle);
                self.session = Some(ActiveSession {
                    handle,
                    title: item.title.clone(),
                    item_id: item.id.clone(),
                    started: std::time::Instant::now(),
                    faulted_seen: false,
                    target_index: index,
                    kernel,
                });
            }
            Err(err) => {
                tracing::warn!(title = %item.title, error = %err, "launch failed");
            }
        }
    }

    fn poll_session(&mut self) {
        let state = {
            let Some(session) = &mut self.session else {
                return;
            };
            let state = self.launcher.session_state(&session.handle);
            if state == SessionState::Faulted {
                session.faulted_seen = true;
            }
            state
        };
        // Only `Exited` clears the overlay. `Faulted` used to be lumped in
        // with `Exited` here, but that meant a synchronous fault (e.g. the
        // real firmware launcher's "no module file" case) would vanish
        // before the very first `draw` ever painted it — the user would
        // never see why. A fault now stays on screen, same as `Running`,
        // until the user presses Back (which calls `quit`, landing on
        // `Exited` on the next poll).
        if state == SessionState::Exited {
            let session = self.session.take().expect("checked above");
            // Session ledger: accumulate play time, remember a fault.
            let ledger_dir = self.ledger_dir();
            let mut title_ledger = ledger::load(&ledger_dir, &session.item_id);
            title_ledger.total_play_secs = title_ledger
                .total_play_secs
                .saturating_add(session.started.elapsed().as_secs());
            title_ledger.last_faulted = session.faulted_seen;
            ledger::store(&ledger_dir, &session.item_id, &title_ledger);
            self.ledgers.insert(session.item_id.clone(), title_ledger);
            self.nav.rail_index = session.target_index;
            self.session_quit_hold.reset();
            // Drop the last frame with the session, or the next launch opens on
            // the previous title's final image before it renders anything.
            self.frame_view.clear();
        }
    }

    fn ledger_dir(&self) -> PathBuf {
        ledger::store_dir(&self.config_path)
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

        // Drilling into a card's option list (ControlCenterOption) must keep
        // the overlay open — it is the same surface, only input routing
        // differs.
        self.cc_open.set_target(
            if matches!(
                self.nav.mode,
                NavMode::ControlCenter | NavMode::ControlCenterOption
            ) {
                1.0
            } else {
                0.0
            },
        );
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
        let mut clicked_home_tile = None;
        let mut clicked_gear = false;
        let mut clicked_pill = None;
        let mut clicked_setting = None;
        let mut clicked_game_option = None;
        let mut clicked_cc = None;

        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            if let Some(session) = &self.session {
                draw_session_overlay(
                    ui,
                    &theme,
                    session,
                    self.launcher.as_ref(),
                    &mut self.frame_view,
                    self.session_quit_hold.progress(),
                    self.render_state.as_ref(),
                );
                return;
            }

            if self.nav.mode == NavMode::GameOptions {
                let title = self
                    .game_options_target_id
                    .as_ref()
                    .and_then(|id| self.library.iter().find(|item| &item.id == id))
                    .map(|item| item.title.clone())
                    .unwrap_or_default();
                clicked_game_option = per_game::draw(
                    ui,
                    &theme,
                    &self.nav,
                    &self.config,
                    &self.game_options_draft,
                    &title,
                );
                return;
            }

            if self.nav.mode == NavMode::Settings {
                let plugins = plugin_row_infos();
                let plugin_failures = raeen_gpu::AgcGpuSession::present_plugin_load_failures();
                clicked_setting = settings::draw(
                    ui,
                    &theme,
                    &self.nav,
                    &self.config,
                    &mut self.settings_new_folder_input,
                    &mut self.settings_key_provider_input,
                    &self.updater_state,
                    &plugins,
                    &plugin_failures,
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
            let home_response = home::draw(
                ui,
                &theme,
                items,
                &self.nav,
                &anim,
                &self.meta_cache,
                &self.ledgers,
                self.background_texture.as_ref(),
                &self.cover_textures,
                self.hero_bg_from.as_ref(),
                self.hero_bg_to.as_ref(),
                self.config.input.controller_icon_style,
            );
            clicked_home_tile = home_response.clicked_tile;
            clicked_gear = home_response.gear_clicked;
            clicked_pill = home_response.clicked_pill;

            let recent_titles = self.recent_titles();
            // Live card values — real config volume and the real connected
            // pad, computed fresh each frame the overlay is visible.
            let live = control_center::CcLive {
                sound: if self.config.audio.enabled {
                    format!(
                        "Host output · {}%",
                        (self.config.audio.volume * 100.0).round() as u32
                    )
                } else {
                    "Muted".to_string()
                },
                accessories: self
                    .gilrs
                    .as_ref()
                    .and_then(|g| g.gamepads().next().map(|(_, pad)| pad.name().to_string()))
                    .unwrap_or_else(|| "No controller connected".to_string()),
            };
            clicked_cc = control_center::draw(
                ui,
                &theme,
                &self.nav,
                self.cc_open.value,
                &recent_titles,
                &live,
            );
        });
        if clicked_gear {
            self.sound_pack.play(sounds::UiSound::Confirm);
            self.nav.mode = NavMode::Settings;
            self.nav.settings_section = 0;
            self.nav.settings_row = 0;
            self.enter_settings();
        }
        if let Some(pill) = clicked_pill {
            // Clicking a pill focuses and activates it in one go — the same
            // path a pad Confirm takes, so tab switching / opening Settings
            // stay in one place (`nav::apply_pills`).
            self.sound_pack.play(sounds::UiSound::Confirm);
            self.nav.mode = NavMode::Pills;
            self.nav.pill_index = pill;
            if self.nav.apply(NavInput::Confirm) == NavAction::OpenSettings {
                self.enter_settings();
            }
        }
        if let Some(click) = clicked_cc
            && matches!(
                self.nav.mode,
                NavMode::ControlCenter | NavMode::ControlCenterOption
            )
        {
            match click {
                control_center::CcClick::Card(index) => {
                    // Focus the clicked card; a second meaning (drilling into
                    // its option list) comes from the same Confirm the pad
                    // uses, so display-only cards simply focus.
                    self.sound_pack.play(sounds::UiSound::Move);
                    self.nav.mode = NavMode::ControlCenter;
                    self.nav.cc_index = index;
                    self.nav.apply(NavInput::Confirm);
                }
                control_center::CcClick::Option(option) => {
                    // Drill into the focused card's list if not already there,
                    // select the clicked line, and activate it.
                    self.sound_pack.play(sounds::UiSound::Confirm);
                    if self.nav.mode == NavMode::ControlCenter {
                        self.nav.apply(NavInput::Confirm);
                    }
                    if self.nav.mode == NavMode::ControlCenterOption {
                        self.nav.cc_option_index = option;
                        if let NavAction::ActivateOption { card, option } =
                            self.nav.apply(NavInput::Confirm)
                        {
                            self.handle_cc_option(ctx, card, option);
                        }
                    }
                }
                control_center::CcClick::Dismiss => {
                    // Same as pressing Guide: close the overlay entirely.
                    self.sound_pack.play(sounds::UiSound::Back);
                    self.nav.apply(NavInput::Guide);
                }
            }
        }
        if let Some(index) = clicked_home_tile {
            self.nav.mode = NavMode::Home;
            self.nav.rail_index = index;
            let action = self.nav.apply(NavInput::Confirm);
            self.play_nav_sound(NavInput::Confirm, action);
            match action {
                NavAction::Launch(index) => self.begin_launch(index),
                NavAction::LaunchMedia(index) => self.confirm_media(index),
                NavAction::OpenSettings => self.enter_settings(),
                _ => {}
            }
        }
        if let Some(clicked) = clicked_setting {
            match clicked {
                settings::SettingsClick::Section(section) => {
                    self.sound_pack.play(sounds::UiSound::Move);
                    self.nav.settings_section = section;
                    self.nav.settings_row = 0;
                }
                settings::SettingsClick::Row(row) => {
                    self.nav.settings_row = row;
                    // Text-entry rows retain native egui widget behavior;
                    // clicking any other row has the same meaning as Confirm
                    // (in Game Folders that includes remove-on-click for
                    // folder rows and the Browse/Rescan action rows).
                    let is_text_row = match self.nav.settings_section {
                        settings::SECTION_KEY_PROVIDER => true,
                        settings::SECTION_GAME_FOLDERS => {
                            row == self.config.paths.game_folders.len()
                        }
                        _ => false,
                    };
                    if !is_text_row {
                        self.sound_pack.play(sounds::UiSound::Confirm);
                        self.apply_setting_activate(ctx, self.nav.settings_section, row);
                    }
                }
            }
        }
        if let Some(per_game::GameOptionsClick::Row(row)) = clicked_game_option {
            // Clicking a row focuses it and toggles its override, matching a
            // Confirm — the same reach the pad has.
            self.nav.game_options_row = row;
            self.game_options_draft.toggle_override(row, &self.config);
        }
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

/// Resolve the live plugin registry into the display rows the Settings ▸
/// Plugins section draws. Active is the registry's word (what is actually
/// running), not the config's (what is persisted) — the two can differ when a
/// persisted plugin is no longer registered.
fn plugin_row_infos() -> Vec<settings::PluginRowInfo> {
    let active = raeen_gpu::AgcGpuSession::active_present_plugin();
    raeen_gpu::AgcGpuSession::present_plugin_infos()
        .into_iter()
        .map(|info| settings::PluginRowInfo {
            capabilities: settings::capability_label(&info.capabilities),
            source: info.source.as_deref().map_or_else(
                || "built-in".to_string(),
                |p| {
                    p.file_name().map_or_else(
                        || p.display().to_string(),
                        |f| f.to_string_lossy().into_owned(),
                    )
                },
            ),
            active: active.as_deref() == Some(info.name.as_str()),
            name: info.name,
        })
        .collect()
}

/// Open the BYO `plugins/` directory in the host's file manager, creating it
/// first so the user always lands somewhere real. Failures are logged, never
/// fatal — this is a convenience, not a load-bearing path.
fn open_plugins_folder() {
    let dir = std::path::Path::new("plugins");
    if let Err(err) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %err, "could not create plugins/ directory");
        return;
    }
    if let Err(err) = opener::open(dir) {
        tracing::warn!(error = %err, "could not open plugins/ in the file manager");
    }
}

/// Radial deadzone for a single analog axis, matching `read_pad`'s inline
/// closure and `raeen_input::InputManager::apply_deadzone`. Used to condition
/// the native (XInput / DualSense) sticks, which arrive without a deadzone.
fn apply_deadzone(v: f32, deadzone: f32) -> f32 {
    if v.abs() < deadzone {
        0.0
    } else {
        v.signum() * (v.abs() - deadzone) / (1.0 - deadzone).max(f32::EPSILON)
    }
}

/// Snapshot a connected gilrs gamepad into an [`raeen_input::ControllerState`],
/// applying `deadzone` to the analog sticks. Stick Y is inverted because gilrs
/// reports up as `+1.0` while the Orbis encoding puts up at the low byte.
fn read_pad(gamepad: &gilrs::Gamepad, deadzone: f32) -> raeen_input::ControllerState {
    use gilrs::{Axis, Button};
    let axis = |a: Axis| gamepad.axis_data(a).map_or(0.0, |d| d.value());
    let button_value = |b: Button| gamepad.button_data(b).map_or(0.0, |d| d.value());
    // Radial deadzone identical to `InputManager::apply_deadzone`, inlined so we
    // don't build (and log) an `InputManager` every frame.
    let dz = |v: f32| {
        if v.abs() < deadzone {
            0.0
        } else {
            v.signum() * (v.abs() - deadzone) / (1.0 - deadzone).max(f32::EPSILON)
        }
    };
    raeen_input::ControllerState {
        cross: gamepad.is_pressed(Button::South),
        circle: gamepad.is_pressed(Button::East),
        square: gamepad.is_pressed(Button::West),
        triangle: gamepad.is_pressed(Button::North),
        l1: gamepad.is_pressed(Button::LeftTrigger),
        r1: gamepad.is_pressed(Button::RightTrigger),
        l3: gamepad.is_pressed(Button::LeftThumb),
        r3: gamepad.is_pressed(Button::RightThumb),
        options: gamepad.is_pressed(Button::Start),
        create: gamepad.is_pressed(Button::Select),
        ps_button: gamepad.is_pressed(Button::Mode),
        dpad_up: gamepad.is_pressed(Button::DPadUp),
        dpad_down: gamepad.is_pressed(Button::DPadDown),
        dpad_left: gamepad.is_pressed(Button::DPadLeft),
        dpad_right: gamepad.is_pressed(Button::DPadRight),
        left_stick_x: dz(axis(Axis::LeftStickX)),
        left_stick_y: dz(-axis(Axis::LeftStickY)),
        right_stick_x: dz(axis(Axis::RightStickX)),
        right_stick_y: dz(-axis(Axis::RightStickY)),
        l2_trigger: button_value(Button::LeftTrigger2).clamp(0.0, 1.0),
        r2_trigger: button_value(Button::RightTrigger2).clamp(0.0, 1.0),
        ..Default::default()
    }
}

/// Keyboard → Orbis pad fallback, OR-merged with the gamepad in
/// [`ShellApp::push_pad_state`] so input reaches the guest even when no
/// controller is mapped/connected (gilrs reports "No mapping found" for some
/// driver/UUID combos, and a keyboard is always available). Layout: `WASD` =
/// left stick, arrow keys = D-pad, `Space` = Cross (✕), `B` = Circle (○),
/// `V` = Square (□), `C` = Triangle (△), `Q`/`E` = L1/R1, `1`/`3` = L2/R2,
/// `Z`/`X` = L3/R3, `Enter` = Options, `Tab` = Create. Consulted only in-session.
fn read_keyboard_pad(ctx: &egui::Context) -> raeen_input::ControllerState {
    use egui::Key;
    ctx.input(|i| {
        let axis = |neg: Key, pos: Key| {
            (if i.key_down(pos) { 1.0 } else { 0.0 }) - (if i.key_down(neg) { 1.0 } else { 0.0 })
        };
        raeen_input::ControllerState {
            cross: i.key_down(Key::Space),
            circle: i.key_down(Key::B),
            square: i.key_down(Key::V),
            triangle: i.key_down(Key::C),
            l1: i.key_down(Key::Q),
            r1: i.key_down(Key::E),
            l3: i.key_down(Key::Z),
            r3: i.key_down(Key::X),
            options: i.key_down(Key::Enter),
            create: i.key_down(Key::Tab),
            dpad_up: i.key_down(Key::ArrowUp),
            dpad_down: i.key_down(Key::ArrowDown),
            dpad_left: i.key_down(Key::ArrowLeft),
            dpad_right: i.key_down(Key::ArrowRight),
            left_stick_x: axis(Key::A, Key::D),
            left_stick_y: axis(Key::W, Key::S),
            l2_trigger: if i.key_down(Key::Num1) { 1.0 } else { 0.0 },
            r2_trigger: if i.key_down(Key::Num3) { 1.0 } else { 0.0 },
            ..Default::default()
        }
    })
}

/// OR-merge two controller snapshots: buttons are logically OR'd, the non-zero
/// analog axis wins (gamepad preferred), triggers take the max — so gamepad and
/// keyboard input both reach the guest without one zeroing the other.
fn merge_pad_states(
    a: raeen_input::ControllerState,
    b: raeen_input::ControllerState,
) -> raeen_input::ControllerState {
    let pick = |x: f32, y: f32| if x != 0.0 { x } else { y };
    raeen_input::ControllerState {
        cross: a.cross || b.cross,
        circle: a.circle || b.circle,
        square: a.square || b.square,
        triangle: a.triangle || b.triangle,
        l1: a.l1 || b.l1,
        r1: a.r1 || b.r1,
        l3: a.l3 || b.l3,
        r3: a.r3 || b.r3,
        options: a.options || b.options,
        create: a.create || b.create,
        ps_button: a.ps_button || b.ps_button,
        touchpad_click: a.touchpad_click || b.touchpad_click,
        dpad_up: a.dpad_up || b.dpad_up,
        dpad_down: a.dpad_down || b.dpad_down,
        dpad_left: a.dpad_left || b.dpad_left,
        dpad_right: a.dpad_right || b.dpad_right,
        left_stick_x: pick(a.left_stick_x, b.left_stick_x),
        left_stick_y: pick(a.left_stick_y, b.left_stick_y),
        right_stick_x: pick(a.right_stick_x, b.right_stick_x),
        right_stick_y: pick(a.right_stick_y, b.right_stick_y),
        l2_trigger: a.l2_trigger.max(b.l2_trigger),
        r2_trigger: a.r2_trigger.max(b.r2_trigger),
        ..Default::default()
    }
}

/// Upload `theme`'s background image (if any) to the GPU as a fresh
/// texture. `None` when the theme carries no background — `home.rs` falls
/// back to its mesh-gradient hero in that case (spec §6).
/// Load a user wallpaper (`wallpapers/<file>`) as the Home background
/// texture. `"off"`, a missing file, or a failed decode all yield `None` so
/// the theme background (or gradient) shows instead — never an error.
fn wallpaper_texture_for(ctx: &egui::Context, wallpaper: &str) -> Option<egui::TextureHandle> {
    if wallpaper == "off" || wallpaper.is_empty() {
        return None;
    }
    let path = std::path::Path::new(WALLPAPERS_ROOT).join(wallpaper);
    let Some(image) = theme::loader::load_image_file_capped(&path) else {
        tracing::warn!(path = %path.display(), "wallpaper failed to load — using theme background");
        return None;
    };
    let fitted = fit_texture_source(image, ctx.input(|i| i.max_texture_side));
    Some(ctx.load_texture("raeen-wallpaper", fitted, egui::TextureOptions::LINEAR))
}

fn background_texture_for(ctx: &egui::Context, theme: &Theme) -> Option<egui::TextureHandle> {
    theme.assets.background.as_ref().map(|image| {
        let fitted = fit_texture_source((**image).clone(), ctx.input(|i| i.max_texture_side));
        ctx.load_texture(
            "raeen-theme-background",
            fitted,
            egui::TextureOptions::LINEAR,
        )
    })
}

/// Downscale `image` (nearest-neighbour, aspect-preserving) so neither side
/// exceeds `max_side` — the running renderer's maximum texture dimension.
///
/// Titles legitimately ship 4K key art (`sce_sys/pic1.png`), and the image
/// loader's decode cap ([`theme::loader`]'s `MAX_IMAGE_DIM`) allows up to 4096
/// px, but an iGPU's egui/wgpu renderer can cap the texture side far lower
/// (e.g. 2048). Uploading art larger than that side used to panic egui and take
/// the whole Shell down on boot; here oversized art is shrunk to fit instead —
/// still shown, never a crash and never silently dropped.
fn fit_texture_source(image: egui::ColorImage, max_side: usize) -> egui::ColorImage {
    let [w, h] = image.size;
    if max_side == 0 || (w <= max_side && h <= max_side) {
        return image;
    }
    let scale = max_side as f32 / w.max(h) as f32;
    let nw = ((w as f32 * scale).floor() as usize).clamp(1, max_side);
    let nh = ((h as f32 * scale).floor() as usize).clamp(1, max_side);
    let mut pixels = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        let sy = (y * h / nh).min(h - 1);
        for x in 0..nw {
            let sx = (x * w / nw).min(w - 1);
            pixels.push(image.pixels[sy * w + sx]);
        }
    }
    egui::ColorImage {
        size: [nw, nh],
        pixels,
    }
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
    use rayon::prelude::*;
    let max_side = ctx.input(|i| i.max_texture_side);
    // Decode + downscale in parallel (pure CPU work, one image per title —
    // the startup/rescan hot spot for a big library); GPU upload stays
    // serial on this thread because it needs the egui context.
    let decoded: Vec<(String, egui::ColorImage)> = library
        .par_iter()
        .filter_map(|item| {
            let path = item.cover_path.as_ref()?;
            let Some(image) = theme::loader::load_image_file_capped(path) else {
                tracing::warn!(path = %path.display(), title = %item.title, "cover image failed to load — using gradient art");
                return None;
            };
            Some((item.id.clone(), fit_texture_source(image, max_side)))
        })
        .collect();
    decoded
        .into_iter()
        .map(|(id, fitted)| {
            let texture = ctx.load_texture(
                format!("raeen-cover-{id}"),
                fitted,
                egui::TextureOptions::LINEAR,
            );
            (id, texture)
        })
        .collect()
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
    use rayon::prelude::*;
    let max_side = ctx.input(|i| i.max_texture_side);
    // Same split as `cover_textures_for`: parallel decode of the (often 4K)
    // key art, serial upload.
    let decoded: Vec<(String, egui::ColorImage)> = library
        .par_iter()
        .filter_map(|item| {
            let LaunchTarget::Game { path } = &item.launch else {
                return None;
            };
            let bg_path = crate::library::scan::title_background(path)?;
            let Some(image) = theme::loader::load_image_file_capped(&bg_path) else {
                tracing::warn!(path = %bg_path.display(), title = %item.title, "background image failed to load — using gradient hero");
                return None;
            };
            Some((item.id.clone(), fit_texture_source(image, max_side)))
        })
        .collect();
    decoded
        .into_iter()
        .map(|(id, fitted)| {
            let texture = ctx.load_texture(
                format!("raeen-bg-{id}"),
                fitted,
                egui::TextureOptions::LINEAR,
            );
            (id, texture)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn draw_session_overlay(
    ui: &mut egui::Ui,
    theme: &Theme,
    session: &ActiveSession,
    launcher: &dyn GameLauncher,
    frame_view: &mut present::GameFrameView,
    quit_progress: f32,
    render_state: Option<&eframe::egui_wgpu::RenderState>,
) {
    let screen = ui.max_rect();
    ui.painter().rect_filled(screen, 0.0, theme.palette.ground);

    let state = launcher.session_state(&session.handle);
    // This is the guest process's completed VideoOut flip sequence, not an
    // egui repaint count. The process-local counter also covers AGC-encoded
    // flips because both paths converge on `sceVideoOutSubmitFlip`.
    let presented_frames = session
        .kernel
        .as_ref()
        .map(|kernel| kernel.video_out_flip_count.load(Ordering::Relaxed));

    // The title's own frames, when it has rendered any. Painted before the
    // status text so the text stays legible over a bright frame.
    let presented = frame_view.paint(ui, screen, presented_frames, render_state);

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
            detail.unwrap_or_else(|| "Running — Esc or hold PS to return to the Shell".to_string()),
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
    let presentation_bounds = match &presented {
        present::Presented::Frame { rect } => *rect,
        present::Presented::NoFrameYet => screen,
    };
    if state == SessionState::Running {
        frame_view.paint_fps(ui, presentation_bounds);
    }
    match presented {
        present::Presented::Frame { rect } => {
            painter.text(
                egui::pos2(rect.left() + 12.0, rect.top() + 10.0),
                egui::Align2::LEFT_TOP,
                format!("{} — Esc or hold PS to return to the Shell", session.title),
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

    // While the PS-button quit hold is filling, show a bottom-centered progress
    // bar so the player sees the exit committing before it fires (SharpEmu #415).
    if quit_progress > 0.0 {
        let painter = ui.painter();
        let bar_w = 240.0;
        let bar_h = 6.0;
        let bar_x = screen.center().x - bar_w / 2.0;
        let bar_y = screen.bottom() - 64.0;
        let track = egui::Rect::from_min_size(egui::pos2(bar_x, bar_y), egui::vec2(bar_w, bar_h));
        painter.rect_filled(
            track,
            3.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 26),
        );
        let fill = egui::Rect::from_min_size(
            track.min,
            egui::vec2(bar_w * quit_progress.clamp(0.0, 1.0), bar_h),
        );
        painter.rect_filled(fill, 3.0, theme.palette.focus);
        painter.text(
            egui::pos2(screen.center().x, bar_y - 14.0),
            egui::Align2::CENTER_BOTTOM,
            "Release to cancel — keep holding PS to quit",
            egui::FontId::proportional(13.0),
            theme.palette.text,
        );
    }

    ui.ctx().request_repaint();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stick_nav_emits_once_per_flick_with_hysteresis() {
        let mut nav = StickNav::default();
        // Ramping up: nothing until the press threshold.
        assert_eq!(nav.update_x(0.3), None);
        assert_eq!(nav.update_x(0.7), Some(NavInput::Right));
        // Held past the threshold: no repeat.
        assert_eq!(nav.update_x(0.9), None);
        assert_eq!(nav.update_x(1.0), None);
        // Jitter back between release (0.4) and press (0.6): still latched.
        assert_eq!(nav.update_x(0.5), None);
        assert_eq!(nav.update_x(0.7), None);
        // Full release re-arms the axis.
        assert_eq!(nav.update_x(0.1), None);
        assert_eq!(nav.update_x(0.8), Some(NavInput::Right));
    }

    #[test]
    fn stick_nav_opposite_flick_emits_without_a_center_stop() {
        let mut nav = StickNav::default();
        assert_eq!(nav.update_x(0.8), Some(NavInput::Right));
        // Snapping straight across to the other side is a fresh step.
        assert_eq!(nav.update_x(-0.8), Some(NavInput::Left));
        assert_eq!(nav.update_x(-0.9), None);
    }

    #[test]
    fn stick_nav_y_up_is_up_and_axes_are_independent() {
        let mut nav = StickNav::default();
        // gilrs reports up as +1.0.
        assert_eq!(nav.update_y(0.8), Some(NavInput::Up));
        assert_eq!(nav.update_y(0.0), None);
        assert_eq!(nav.update_y(-0.8), Some(NavInput::Down));
        // A held vertical deflection never blocks the horizontal axis.
        assert_eq!(nav.update_x(0.8), Some(NavInput::Right));
    }

    #[test]
    fn push_recent_orders_most_recent_first() {
        let mut recent = Vec::new();
        push_recent(&mut recent, "a", 6);
        push_recent(&mut recent, "b", 6);
        assert_eq!(recent, vec!["b".to_string(), "a".to_string()]);
    }

    #[test]
    fn quit_hold_fires_once_when_the_hold_crosses_the_threshold() {
        let mut hold = QuitHold::default();
        // A tap (well under the threshold) never fires and leaves no progress.
        assert!(!hold.update(true, 0.1));
        assert!(hold.progress() > 0.0 && hold.progress() < 1.0);
        // Releasing resets the hold.
        assert!(!hold.update(false, 0.1));
        assert_eq!(hold.progress(), 0.0);
        // A sustained hold fires exactly once as it crosses the threshold…
        assert!(!hold.update(true, SESSION_QUIT_HOLD_SECS - 0.05));
        assert!(hold.update(true, 0.1));
        // …and not again on subsequent held frames (edge-triggered).
        assert!(!hold.update(true, 0.1));
    }

    #[test]
    fn quit_hold_progress_is_clamped_and_resettable() {
        let mut hold = QuitHold::default();
        hold.update(true, SESSION_QUIT_HOLD_SECS * 4.0);
        assert_eq!(hold.progress(), 1.0);
        hold.reset();
        assert_eq!(hold.progress(), 0.0);
    }

    #[test]
    fn fit_texture_source_downscales_oversized_art_preserving_aspect() {
        // A 3840x2160 key art on a renderer capped at 2048 must shrink to fit
        // (this is the exact case that used to panic egui on boot).
        let big = egui::ColorImage {
            size: [3840, 2160],
            pixels: vec![egui::Color32::RED; 3840 * 2160],
        };
        let fitted = fit_texture_source(big, 2048);
        assert!(fitted.size[0] <= 2048 && fitted.size[1] <= 2048);
        // Longer side maps to the cap; aspect roughly preserved (16:9).
        assert_eq!(fitted.size[0], 2048);
        assert_eq!(fitted.size[1], 1152);
        assert_eq!(fitted.pixels.len(), fitted.size[0] * fitted.size[1]);
    }

    #[test]
    fn fit_texture_source_leaves_in_bounds_art_untouched() {
        let small = egui::ColorImage {
            size: [512, 512],
            pixels: vec![egui::Color32::BLUE; 512 * 512],
        };
        let fitted = fit_texture_source(small, 2048);
        assert_eq!(fitted.size, [512, 512]);
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
