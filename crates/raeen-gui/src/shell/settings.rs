//! Settings screen (spec §3 screen 5, §10 SM2) — PS5-style.
//!
//! A full-screen surface: an icon sidebar of sections on the left (Video,
//! Audio, Controller, Game Folders, Key Provider, Theme, Plugins, System,
//! Advanced) and the focused section's rows as full-width cards on the
//! right, bound to the real [`raeen_core::config::EmulatorConfig`].
//! Navigation (which section/row is focused, Up/Down between rows,
//! Left/Right/Confirm to adjust, Back to leave) lives in [`super::nav`] as
//! pure state; this module draws it and reports pointer interaction.
//!
//! Every row and sidebar entry is an *allocated* egui widget, so its hit
//! rect is exactly its painted rect — pointer clicks can never drift from
//! the visuals (the old fixed-pitch overlay grid did).
//!
//! Key Provider is a path field only — the Shell stores and displays the
//! string but never reads, parses, or otherwise touches key material (spec
//! §11). Theme is populated from installed `themes/<name>/theme.toml`
//! directories on disk (spec §6, §10 SM2b) via [`available_themes`];
//! Wallpaper and the UI Sound Pack come from user-supplied `wallpapers/`
//! and `sounds/<pack>/` trees.

use super::icons::{self, Glyph};
use super::nav::NavState;
use crate::theme::Theme;
use egui::{
    Align2, Color32, CursorIcon, FontId, Mesh, Pos2, Rect, RichText, Sense, Shape, StrokeKind,
    UiBuilder, vec2,
};
use raeen_core::config::EmulatorConfig;
use std::path::Path;

/// Pointer selection reported to the Shell after the Settings surface draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsClick {
    Section(usize),
    Row(usize),
}

/// Section names, in the order `nav::NavState::settings_section` indexes.
pub const SETTINGS_SECTION_NAMES: [&str; 9] = [
    "Video",
    "Audio",
    "Controller",
    "Game Folders",
    "Key Provider",
    "Theme",
    "Plugins",
    "System",
    "Advanced",
];

// Section indices, named so `shell/mod.rs`'s `(section, row)` dispatch and
// this file's draw dispatch cannot drift apart when a section is inserted.
pub const SECTION_VIDEO: usize = 0;
pub const SECTION_AUDIO: usize = 1;
pub const SECTION_CONTROLLER: usize = 2;
pub const SECTION_GAME_FOLDERS: usize = 3;
pub const SECTION_KEY_PROVIDER: usize = 4;
pub const SECTION_THEME: usize = 5;
pub const SECTION_PLUGINS: usize = 6;
pub const SECTION_SYSTEM: usize = 7;
pub const SECTION_ADVANCED: usize = 8;

/// The sidebar glyph for each section, in `SETTINGS_SECTION_NAMES` order.
fn section_glyph(section: usize) -> Glyph {
    match section {
        SECTION_VIDEO => Glyph::Monitor,
        SECTION_AUDIO => Glyph::Sound,
        SECTION_CONTROLLER => Glyph::Pad,
        SECTION_GAME_FOLDERS => Glyph::Folder,
        SECTION_KEY_PROVIDER => Glyph::Key,
        SECTION_THEME => Glyph::Palette,
        SECTION_PLUGINS => Glyph::Puzzle,
        SECTION_SYSTEM => Glyph::Info,
        SECTION_ADVANCED => Glyph::Wrench,
        _ => Glyph::Grid,
    }
}

/// Rows the Plugins section appends after its per-plugin rows: "Rescan
/// Plugins Folder" and "Open Plugins Folder".
pub const PLUGIN_ACTION_ROWS: usize = 2;

/// Rows the Game Folders section appends after its per-folder rows:
/// "Add Folder" (typed path), "Browse & Add Folder…" (native picker), and
/// "Rescan Games".
pub const GAME_FOLDER_ACTION_ROWS: usize = 3;

/// Number of rows in each section, in `SETTINGS_SECTION_NAMES` order. The
/// Game Folders section grows by one row per configured folder plus its
/// three fixed action rows; the Plugins section grows by one row per
/// registered present plugin plus its two fixed action rows — so this takes
/// both live counts.
pub fn settings_row_counts(game_folder_count: usize, plugin_count: usize) -> Vec<usize> {
    vec![
        11,
        4,
        3,
        game_folder_count + GAME_FOLDER_ACTION_ROWS,
        1,
        2,
        plugin_count + PLUGIN_ACTION_ROWS,
        2,
        8,
    ]
}

/// One registered present plugin, pre-resolved by the Shell into plain display
/// strings so this module stays free of `raeen-gpu` types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginRowInfo {
    /// The plugin's self-reported name (also the selection key).
    pub name: String,
    /// Human-readable capability summary ("Upscale · Frame Gen", …).
    pub capabilities: String,
    /// Where it came from: "built-in" or the binary's file name.
    pub source: String,
    /// Whether this plugin is the active present plugin right now.
    pub active: bool,
}

/// Human-readable capability summary for a plugin row. All-false capabilities
/// read as "Passthrough" (the plugin transforms nothing).
#[must_use]
pub fn capability_label(caps: &raeen_gpu::Capabilities) -> String {
    let mut parts = Vec::new();
    if caps.upscale {
        parts.push("Upscale");
    }
    if caps.frame_gen {
        parts.push("Frame Gen");
    }
    if caps.wants_depth {
        parts.push("Depth");
    }
    if caps.wants_motion_vectors {
        parts.push("Motion Vectors");
    }
    if parts.is_empty() {
        "Passthrough".to_string()
    } else {
        parts.join(" · ")
    }
}

/// The user-facing label for an `"off"`-or-name config value: `"off"` reads
/// as `Off`, anything else is shown verbatim.
#[must_use]
pub fn upscaler_label(name: &str) -> String {
    if name == "off" {
        "Off".to_string()
    } else {
        name.to_string()
    }
}

/// Cycle `current` through `options` by `delta` steps (wrapping). Pure and
/// generic over any string-option row: the upscaler, the UI sound pack, the
/// wallpaper. An unknown current value restarts from the first option; an
/// empty option set is a no-op.
#[must_use]
pub fn cycle_option(current: &str, delta: i32, options: &[String]) -> String {
    if options.is_empty() {
        return current.to_string();
    }
    let idx = options.iter().position(|o| o == current).unwrap_or(0) as i32;
    let n = options.len() as i32;
    let next = (((idx + delta) % n) + n) % n;
    options[next as usize].clone()
}

/// Cycle the selected present-plugin name through `options` by `delta` steps
/// (wrapping). Kept as a named alias of [`cycle_option`] for its call sites
/// and tests.
#[must_use]
pub fn cycle_upscaler(current: &str, delta: i32, options: &[String]) -> String {
    cycle_option(current, delta, options)
}

/// Frame-limit presets the Video ▸ Frame Limit row cycles through (guest vblank
/// cadence in Hz). 60 is native PS5.
pub const FRAME_LIMITS: [u32; 6] = [30, 60, 90, 120, 144, 240];

/// Cycle `current` through [`FRAME_LIMITS`] by `delta` steps (wrapping). A
/// non-preset current value snaps to its nearest preset first.
pub fn cycle_frame_limit(current: u32, delta: i32) -> u32 {
    let idx = FRAME_LIMITS
        .iter()
        .position(|&h| h == current)
        .unwrap_or_else(|| {
            FRAME_LIMITS
                .iter()
                .enumerate()
                .min_by_key(|(_, h)| (i64::from(**h) - i64::from(current)).abs())
                .map_or(1, |(i, _)| i)
        });
    let next = (idx as i32 + delta).rem_euclid(FRAME_LIMITS.len() as i32) as usize;
    FRAME_LIMITS[next]
}

/// Step an integer setting (window size, GPU device index) by `delta` steps of
/// `step`, clamped to `[min, max]`. The `u32` counterpart to [`adjust_stepped`].
pub fn adjust_stepped_u32(value: u32, delta: i32, step: u32, min: u32, max: u32) -> u32 {
    let stepped = i64::from(value) + i64::from(delta) * i64::from(step);
    stepped.clamp(i64::from(min), i64::from(max)) as u32
}

/// Log-level names the Debug section's "Log Level" row cycles through, in
/// increasing verbosity — the values `tracing`'s env-filter accepts.
pub const LOG_LEVELS: [&str; 5] = ["error", "warn", "info", "debug", "trace"];

/// Cycle `current` through [`LOG_LEVELS`] by `delta` steps (wrapping). An
/// unrecognized current value restarts from `info`.
pub fn cycle_log_level(current: &str, delta: i32) -> String {
    let idx = LOG_LEVELS.iter().position(|l| *l == current).unwrap_or(2);
    let next = (idx as i32 + delta).rem_euclid(LOG_LEVELS.len() as i32) as usize;
    LOG_LEVELS[next].to_string()
}

/// Step `value` by `delta` steps of size `step`, clamped to `[min, max]`.
/// Used for every numeric Settings row (resolution scale, volume,
/// deadzone) so they share one clamping rule instead of each hand-rolling
/// it slightly differently.
pub fn adjust_stepped(value: f32, delta: i32, step: f32, min: f32, max: f32) -> f32 {
    (value + delta as f32 * step).clamp(min, max)
}

/// Theme names the Settings screen's Theme selector can choose between:
/// `"default"` plus whatever's installed under `themes_root` (spec §6,
/// §10 SM2b).
pub fn available_themes(themes_root: &Path) -> Vec<String> {
    crate::theme::loader::list_themes(themes_root)
}

/// Wallpaper choices: `"off"` plus every image file under `root`
/// (user-supplied `wallpapers/`). Sorted for a stable cycle order.
pub fn available_wallpapers(root: &Path) -> Vec<String> {
    const IMAGE_EXTS: [&str; 4] = ["png", "jpg", "jpeg", "bmp"];
    let mut options = vec!["off".to_string()];
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut files: Vec<String> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| IMAGE_EXTS.iter().any(|x| e.eq_ignore_ascii_case(x)))
            })
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_string))
            .collect();
        files.sort();
        options.extend(files);
    }
    options
}

fn on_off(b: bool) -> &'static str {
    if b { "On" } else { "Off" }
}

// ─── Layout metrics ─────────────────────────────────────────────────────────

const MARGIN_X: f32 = 54.0;
const HEADER_H: f32 = 110.0;
const SIDEBAR_W: f32 = 280.0;
const SIDEBAR_ROW_H: f32 = 48.0;
const PANE_GAP: f32 = 46.0;
const PANE_MAX_W: f32 = 680.0;
const ROW_H: f32 = 46.0;
const ROW_GAP: f32 = 6.0;

/// Per-frame drawing context threaded through the section renderers: theme +
/// nav focus in, the clicked row (if any) out.
struct Rows<'a> {
    theme: &'a Theme,
    nav: &'a NavState,
    clicked: Option<usize>,
}

impl Rows<'_> {
    /// One PS5-style settings row card: rounded focus/hover fill, label on
    /// the left, value on the right. The hit rect is exactly the painted
    /// rect. Records a click into `self.clicked`.
    fn row(&mut self, ui: &mut egui::Ui, row_index: usize, label: &str, value: String) {
        let (rect, resp) = self.row_frame(ui, row_index);
        let focused = self.nav.settings_row == row_index;
        let painter = ui.painter();
        painter.text(
            Pos2::new(rect.left() + 20.0, rect.center().y),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(15.5),
            if focused {
                self.theme.palette.text
            } else {
                self.theme.palette.text_dim
            },
        );
        painter.text(
            Pos2::new(rect.right() - 20.0, rect.center().y),
            Align2::RIGHT_CENTER,
            value,
            FontId::proportional(14.5),
            if focused {
                self.theme.palette.focus
            } else {
                self.theme.palette.text_faint
            },
        );
        if resp.clicked() {
            self.clicked = Some(row_index);
        }
    }

    /// A row whose value side is a live text-entry widget (Add Folder,
    /// KeyProvider path). Clicking is left to the embedded `TextEdit`; the
    /// surrounding card only paints focus.
    fn text_row(&mut self, ui: &mut egui::Ui, row_index: usize, label: &str, input: &mut String) {
        let (rect, _resp) = self.row_frame(ui, row_index);
        let focused = self.nav.settings_row == row_index;
        ui.painter().text(
            Pos2::new(rect.left() + 20.0, rect.center().y),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(15.5),
            if focused {
                self.theme.palette.text
            } else {
                self.theme.palette.text_dim
            },
        );
        let edit_rect = Rect::from_min_max(
            Pos2::new(rect.left() + rect.width() * 0.42, rect.top() + 9.0),
            Pos2::new(rect.right() - 16.0, rect.bottom() - 9.0),
        );
        ui.put(
            edit_rect,
            egui::TextEdit::singleline(input).vertical_align(egui::Align::Center),
        );
    }

    /// Allocate + paint one row card's frame, returning its rect/response.
    fn row_frame(&mut self, ui: &mut egui::Ui, row_index: usize) -> (Rect, egui::Response) {
        let (rect, resp) =
            ui.allocate_exact_size(vec2(ui.available_width(), ROW_H), Sense::click());
        let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
        let focused = self.nav.settings_row == row_index;
        let painter = ui.painter();
        // Every row is a faint card; focus brightens it and adds the accent
        // bar, hover sits in between — the PS5's row treatment.
        let fill = if focused {
            self.theme.palette.raised.gamma_multiply(1.35)
        } else if resp.hovered() {
            self.theme.palette.raised.gamma_multiply(1.1)
        } else {
            self.theme.palette.raised.gamma_multiply(0.55)
        };
        painter.rect_filled(rect, 10.0, fill);
        if focused {
            painter.rect_filled(
                Rect::from_min_size(
                    Pos2::new(rect.left(), rect.top() + 9.0),
                    vec2(3.5, rect.height() - 18.0),
                ),
                2.0,
                self.theme.palette.focus,
            );
            painter.rect_stroke(
                rect,
                10.0,
                egui::Stroke::new(1.0, self.theme.palette.focus.gamma_multiply(0.5)),
                StrokeKind::Inside,
            );
        }
        ui.add_space(ROW_GAP);
        (rect, resp)
    }

    /// A faint explanatory footnote under a section's rows.
    fn hint(&self, ui: &mut egui::Ui, text: &str) {
        ui.add_space(8.0);
        ui.label(
            RichText::new(text)
                .color(self.theme.palette.text_faint)
                .size(12.0),
        );
    }
}

/// Vertical two-stop gradient (same approach as the Control Center's).
fn vertical_gradient(painter: &egui::Painter, rect: Rect, top: Color32, bottom: Color32) {
    let mut mesh = Mesh::default();
    mesh.colored_vertex(rect.left_top(), top);
    mesh.colored_vertex(rect.right_top(), top);
    mesh.colored_vertex(rect.left_bottom(), bottom);
    mesh.colored_vertex(rect.right_bottom(), bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 2, 3);
    painter.add(Shape::mesh(mesh));
}

/// Draw the full Settings surface. `new_folder_input`/`key_provider_input`
/// are Shell-owned scratch text buffers bound to their respective text
/// fields; `plugins`/`plugin_failures` are the pre-resolved Plugins rows.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    theme: &Theme,
    nav: &NavState,
    config: &EmulatorConfig,
    new_folder_input: &mut String,
    key_provider_input: &mut String,
    updater: &crate::updater::UpdaterState,
    plugins: &[PluginRowInfo],
    plugin_failures: &[String],
) -> Option<SettingsClick> {
    let screen = ui.max_rect();
    let painter = ui.painter().clone();
    painter.rect_filled(screen, 0.0, theme.palette.ground);
    // Subtle top sheen so the surface reads as lit, not flat.
    vertical_gradient(
        &painter,
        Rect::from_min_size(screen.min, vec2(screen.width(), 240.0)),
        Color32::from_rgba_unmultiplied(255, 255, 255, 10),
        Color32::TRANSPARENT,
    );

    painter.text(
        Pos2::new(screen.left() + MARGIN_X, screen.top() + 44.0),
        Align2::LEFT_CENTER,
        "Settings",
        FontId::proportional(30.0),
        theme.palette.text,
    );
    painter.line_segment(
        [
            Pos2::new(screen.left() + MARGIN_X, screen.top() + HEADER_H - 22.0),
            Pos2::new(screen.right() - MARGIN_X, screen.top() + HEADER_H - 22.0),
        ],
        egui::Stroke::new(1.0, theme.palette.line),
    );

    let mut clicked = None;

    // ── Sidebar ──
    let sidebar = Rect::from_min_size(
        Pos2::new(screen.left() + MARGIN_X, screen.top() + HEADER_H),
        vec2(SIDEBAR_W, screen.height() - HEADER_H - 60.0),
    );
    ui.scope_builder(UiBuilder::new().max_rect(sidebar), |ui| {
        for (i, name) in SETTINGS_SECTION_NAMES.iter().enumerate() {
            let (rect, resp) =
                ui.allocate_exact_size(vec2(ui.available_width(), SIDEBAR_ROW_H), Sense::click());
            let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
            let focused = i == nav.settings_section;
            let painter = ui.painter();
            if focused {
                painter.rect_filled(rect, SIDEBAR_ROW_H / 2.0, theme.palette.focus);
            } else if resp.hovered() {
                painter.rect_filled(
                    rect,
                    SIDEBAR_ROW_H / 2.0,
                    theme.palette.raised.gamma_multiply(1.2),
                );
            }
            let (icon_color, text_color) = if focused {
                (theme.palette.ground, theme.palette.ground)
            } else {
                (theme.palette.text_dim, theme.palette.text_dim)
            };
            icons::draw(
                painter,
                section_glyph(i),
                Pos2::new(rect.left() + 30.0, rect.center().y),
                17.0,
                icon_color,
            );
            painter.text(
                Pos2::new(rect.left() + 56.0, rect.center().y),
                Align2::LEFT_CENTER,
                *name,
                FontId::proportional(16.0),
                text_color,
            );
            if resp.clicked() {
                clicked = Some(SettingsClick::Section(i));
            }
            ui.add_space(6.0);
        }
    });

    // ── Section pane ──
    let pane_w = (screen.width() - MARGIN_X * 2.0 - SIDEBAR_W - PANE_GAP).min(PANE_MAX_W);
    let pane = Rect::from_min_size(
        Pos2::new(
            screen.left() + MARGIN_X + SIDEBAR_W + PANE_GAP,
            screen.top() + HEADER_H,
        ),
        vec2(pane_w.max(320.0), screen.height() - HEADER_H - 60.0),
    );
    let mut rows = Rows {
        theme,
        nav,
        clicked: None,
    };
    ui.scope_builder(UiBuilder::new().max_rect(pane), |ui| {
        ui.label(
            RichText::new(SETTINGS_SECTION_NAMES[nav.settings_section])
                .color(theme.palette.text)
                .size(20.0)
                .strong(),
        );
        ui.add_space(14.0);
        match nav.settings_section {
            SECTION_VIDEO => draw_video(ui, &mut rows, config),
            SECTION_AUDIO => draw_audio(ui, &mut rows, config),
            SECTION_CONTROLLER => draw_input(ui, &mut rows, config),
            SECTION_GAME_FOLDERS => draw_game_folders(ui, &mut rows, config, new_folder_input),
            SECTION_KEY_PROVIDER => draw_key_provider(ui, &mut rows, key_provider_input),
            SECTION_THEME => draw_theme(ui, &mut rows, config),
            SECTION_PLUGINS => draw_plugins(ui, &mut rows, plugins, plugin_failures),
            SECTION_SYSTEM => draw_system(ui, &mut rows, updater),
            SECTION_ADVANCED => draw_debug(ui, &mut rows, config),
            _ => {}
        }
    });
    if let Some(row) = rows.clicked {
        clicked = Some(SettingsClick::Row(row));
    }

    // ── Footer key hints ──
    painter.text(
        Pos2::new(screen.left() + MARGIN_X, screen.bottom() - 26.0),
        Align2::LEFT_CENTER,
        format!(
            "\u{2191}\u{2193}/Wheel Rows    \u{25C0}\u{25B6}/Enter/{} Adjust    Esc/{}/Right-click Back",
            config.input.controller_icon_style.confirm(),
            config.input.controller_icon_style.back(),
        ),
        FontId::proportional(13.0),
        theme.palette.text_dim,
    );

    clicked
}

fn draw_video(ui: &mut egui::Ui, rows: &mut Rows, config: &EmulatorConfig) {
    rows.row(
        ui,
        0,
        "Resolution Scale",
        format!("{:.2}x", config.graphics.resolution_scale),
    );
    rows.row(
        ui,
        1,
        "Fullscreen",
        on_off(config.general.fullscreen).to_string(),
    );
    rows.row(
        ui,
        2,
        "Shader Cache",
        on_off(config.graphics.shader_cache).to_string(),
    );
    rows.row(
        ui,
        3,
        "Validation Layers",
        on_off(config.graphics.validation_layers).to_string(),
    );
    rows.row(ui, 4, "VSync", on_off(config.general.vsync).to_string());
    rows.row(
        ui,
        5,
        "Frame Limit",
        format!("{} Hz", config.graphics.frame_limit),
    );
    rows.row(
        ui,
        6,
        "GPU Device",
        if config.graphics.gpu_device_index == 0 {
            "Auto (best)".to_string()
        } else {
            format!("Device {}", config.graphics.gpu_device_index)
        },
    );
    rows.row(
        ui,
        7,
        "Window Width",
        format!("{} px", config.general.window_width),
    );
    rows.row(
        ui,
        8,
        "Window Height",
        format!("{} px", config.general.window_height),
    );
    rows.row(
        ui,
        9,
        "Upscaler / Frame Gen",
        upscaler_label(&config.graphics.upscaler),
    );
    rows.row(
        ui,
        10,
        "Upscale Factor",
        format!("{:.2}x", config.graphics.present_upscale),
    );
    rows.hint(
        ui,
        "Window size applies when Fullscreen is Off. Frame Limit and GPU Device apply on the \
         next launch; VSync applies after restarting Raeen. Upscaler applies live; only Raeen's \
         built-in plugins ship — proprietary ones (e.g. DLSS) are user-supplied.",
    );
}

fn draw_audio(ui: &mut egui::Ui, rows: &mut Rows, config: &EmulatorConfig) {
    rows.row(
        ui,
        0,
        "Audio Enabled",
        on_off(config.audio.enabled).to_string(),
    );
    rows.row(
        ui,
        1,
        "Master Volume",
        format!("{}%", (config.audio.volume * 100.0).round() as i32),
    );
    rows.row(
        ui,
        2,
        "Spatial Audio",
        on_off(config.audio.spatial_audio).to_string(),
    );
    rows.row(
        ui,
        3,
        "UI Sound Pack",
        upscaler_label(&config.general.sound_pack),
    );
    rows.hint(
        ui,
        "Enabled, Volume, and the UI Sound Pack apply immediately. A pack is a folder under \
         sounds/ with move.wav, confirm.wav, back.wav, launch.wav (all optional). Spatial Audio \
         is reserved for the Tempest 3D engine and has no effect yet.",
    );
}

fn draw_input(ui: &mut egui::Ui, rows: &mut Rows, config: &EmulatorConfig) {
    rows.row(
        ui,
        0,
        "DualSense Features",
        on_off(config.input.dualsense_features).to_string(),
    );
    rows.row(
        ui,
        1,
        "Stick Deadzone",
        format!("{:.2}", config.input.deadzone),
    );
    rows.row(
        ui,
        2,
        "Button Icon Style",
        config.input.controller_icon_style.label().to_string(),
    );
    rows.hint(
        ui,
        "Deadzone and Icon Style apply immediately. DualSense Features (adaptive triggers, \
         haptics, lightbar) is reserved and has no effect yet.",
    );
}

fn draw_game_folders(
    ui: &mut egui::Ui,
    rows: &mut Rows,
    config: &EmulatorConfig,
    new_folder_input: &mut String,
) {
    for (i, path) in config.paths.game_folders.iter().enumerate() {
        rows.row(ui, i, "Folder", path.display().to_string());
    }
    let add_row = config.paths.game_folders.len();
    rows.text_row(ui, add_row, "Add Folder", new_folder_input);
    rows.row(
        ui,
        add_row + 1,
        "Browse & Add Folder\u{2026}",
        String::new(),
    );
    rows.row(ui, add_row + 2, "Rescan Games", String::new());
    rows.hint(
        ui,
        "Confirm on a folder row removes it. \"Browse & Add Folder\" opens the system folder \
         picker; typed paths still work via \"Add Folder\". Folder changes rescan the library \
         immediately — \"Rescan Games\" re-reads the folders after you add or remove games on disk.",
    );
}

fn draw_key_provider(ui: &mut egui::Ui, rows: &mut Rows, key_provider_input: &mut String) {
    rows.text_row(ui, 0, "KeyProvider Path", key_provider_input);
    rows.hint(
        ui,
        "Path only — Raeen never reads or handles key material here.",
    );
}

fn draw_theme(ui: &mut egui::Ui, rows: &mut Rows, config: &EmulatorConfig) {
    rows.row(ui, 0, "Theme", config.general.selected_theme.clone());
    rows.row(
        ui,
        1,
        "Wallpaper",
        upscaler_label(&config.general.wallpaper),
    );
    rows.hint(
        ui,
        "Left/Right or Confirm cycles. Themes install under themes/<name>/theme.toml; a \
         wallpaper is any image dropped into wallpapers/ and overrides the theme's background.",
    );
}

/// Plugins section: one row per registered present plugin, then the two
/// action rows and any load refusals from the last scan.
fn draw_plugins(
    ui: &mut egui::Ui,
    rows: &mut Rows,
    plugins: &[PluginRowInfo],
    plugin_failures: &[String],
) {
    for (i, plugin) in plugins.iter().enumerate() {
        let value = if plugin.active {
            format!("{} · {} · Active", plugin.capabilities, plugin.source)
        } else {
            format!("{} · {}", plugin.capabilities, plugin.source)
        };
        rows.row(ui, i, &plugin.name, value);
    }
    rows.row(ui, plugins.len(), "Rescan Plugins Folder", String::new());
    rows.row(ui, plugins.len() + 1, "Open Plugins Folder", String::new());
    rows.hint(
        ui,
        "Confirm activates a plugin (again to deactivate). Built-ins ship with Raeen; \
         everything else is user-supplied in plugins/ and loaded at startup or on rescan.",
    );
    if !plugin_failures.is_empty() {
        ui.add_space(10.0);
        ui.label(
            RichText::new("Refused on the last scan:")
                .color(rows.theme.palette.text)
                .size(13.0),
        );
        for failure in plugin_failures {
            ui.label(
                RichText::new(format!("\u{2022} {failure}"))
                    .color(rows.theme.palette.text_faint)
                    .size(12.0),
            );
        }
    }
}

/// System section: current version + the updater's single action row
/// ("Check for Updates" → "Download Update" → "Restart & Update" depending
/// on [`crate::updater::UpdaterState`]).
fn draw_system(ui: &mut egui::Ui, rows: &mut Rows, updater: &crate::updater::UpdaterState) {
    rows.row(ui, 0, "Version", format!("v{}", raeen_core::VERSION));
    rows.row(ui, 1, updater.action_label(), updater.status_line());
    rows.hint(
        ui,
        "Updates are fetched from GitHub Releases and applied on restart.",
    );
}

/// Debug section: developer diagnostics, all persisted to `config.toml`'s
/// `[debug]` table — logging verbosity plus the GPU/shader/syscall dump
/// toggles the runtime reads when a session launches.
fn draw_debug(ui: &mut egui::Ui, rows: &mut Rows, config: &EmulatorConfig) {
    rows.row(ui, 0, "Logging", on_off(config.debug.logging).to_string());
    rows.row(ui, 1, "Log Level", config.debug.log_level.clone());
    rows.row(
        ui,
        2,
        "Trace HLE Calls",
        on_off(config.debug.trace_syscalls).to_string(),
    );
    rows.row(
        ui,
        3,
        "Dump GPU Resources",
        on_off(config.debug.dump_gpu_commands).to_string(),
    );
    rows.row(
        ui,
        4,
        "Dump Shaders",
        on_off(config.debug.dump_shaders).to_string(),
    );
    rows.row(
        ui,
        5,
        "Dump Frames",
        on_off(config.debug.dump_frames).to_string(),
    );
    rows.row(
        ui,
        6,
        "Call Stats",
        on_off(config.debug.call_stats).to_string(),
    );
    rows.row(
        ui,
        7,
        "Stall Dump",
        on_off(config.debug.stall_dump).to_string(),
    );
    rows.hint(
        ui,
        "Developer diagnostics. Logging is live; the dumps and traces apply on the next launch. \
         A RAEEN_* variable set in the environment before starting Raeen overrides its toggle here.",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_names_and_row_counts_stay_in_step() {
        assert_eq!(
            settings_row_counts(0, 0).len(),
            SETTINGS_SECTION_NAMES.len()
        );
    }

    #[test]
    fn section_constants_match_the_names_table() {
        assert_eq!(SETTINGS_SECTION_NAMES[SECTION_VIDEO], "Video");
        assert_eq!(SETTINGS_SECTION_NAMES[SECTION_AUDIO], "Audio");
        assert_eq!(SETTINGS_SECTION_NAMES[SECTION_CONTROLLER], "Controller");
        assert_eq!(SETTINGS_SECTION_NAMES[SECTION_GAME_FOLDERS], "Game Folders");
        assert_eq!(SETTINGS_SECTION_NAMES[SECTION_KEY_PROVIDER], "Key Provider");
        assert_eq!(SETTINGS_SECTION_NAMES[SECTION_THEME], "Theme");
        assert_eq!(SETTINGS_SECTION_NAMES[SECTION_PLUGINS], "Plugins");
        assert_eq!(SETTINGS_SECTION_NAMES[SECTION_SYSTEM], "System");
        assert_eq!(SETTINGS_SECTION_NAMES[SECTION_ADVANCED], "Advanced");
    }

    #[test]
    fn game_folders_row_count_tracks_the_configured_folder_count() {
        // Just the action rows: Add Folder, Browse & Add, Rescan Games.
        assert_eq!(settings_row_counts(0, 0)[SECTION_GAME_FOLDERS], 3);
        // 2 folders + the three action rows.
        assert_eq!(settings_row_counts(2, 0)[SECTION_GAME_FOLDERS], 5);
    }

    #[test]
    fn plugins_row_count_tracks_the_registered_plugin_count() {
        // No plugins: just the Rescan + Open Folder action rows.
        assert_eq!(settings_row_counts(0, 0)[SECTION_PLUGINS], 2);
        // The built-in pair plus the action rows.
        assert_eq!(settings_row_counts(0, 2)[SECTION_PLUGINS], 4);
    }

    #[test]
    fn other_sections_row_counts_are_fixed() {
        let counts = settings_row_counts(5, 3);
        assert_eq!(counts[SECTION_VIDEO], 11); // + Frame Limit, GPU Device, Window W/H, Upscaler, Factor
        assert_eq!(counts[SECTION_AUDIO], 4); // + UI Sound Pack
        assert_eq!(counts[SECTION_CONTROLLER], 3);
        assert_eq!(counts[SECTION_KEY_PROVIDER], 1);
        assert_eq!(counts[SECTION_THEME], 2); // Theme + Wallpaper
        assert_eq!(counts[SECTION_SYSTEM], 2); // version + updater action
        assert_eq!(counts[SECTION_ADVANCED], 8);
    }

    #[test]
    fn capability_label_names_each_capability_and_passthrough() {
        use raeen_gpu::Capabilities;
        assert_eq!(capability_label(&Capabilities::default()), "Passthrough");
        assert_eq!(
            capability_label(&Capabilities {
                upscale: true,
                ..Default::default()
            }),
            "Upscale"
        );
        assert_eq!(
            capability_label(&Capabilities {
                upscale: true,
                frame_gen: true,
                wants_depth: true,
                wants_motion_vectors: true,
            }),
            "Upscale · Frame Gen · Depth · Motion Vectors"
        );
    }

    #[test]
    fn cycle_upscaler_wraps_and_labels() {
        let opts = vec![
            "off".to_string(),
            "passthrough".to_string(),
            "nearest".to_string(),
        ];
        assert_eq!(cycle_upscaler("off", 1, &opts), "passthrough");
        assert_eq!(cycle_upscaler("nearest", 1, &opts), "off"); // wraps top -> bottom
        assert_eq!(cycle_upscaler("off", -1, &opts), "nearest"); // wraps bottom -> top
        // Unknown current name starts from the first option.
        assert_eq!(cycle_upscaler("bogus", 1, &opts), "passthrough");
        // Empty option set is a no-op.
        assert_eq!(cycle_upscaler("off", 1, &[]), "off");
        assert_eq!(upscaler_label("off"), "Off");
        assert_eq!(upscaler_label("nearest"), "nearest");
    }

    #[test]
    fn cycle_frame_limit_wraps_and_snaps_to_nearest_preset() {
        assert_eq!(cycle_frame_limit(60, 1), 90);
        assert_eq!(cycle_frame_limit(60, -1), 30);
        assert_eq!(cycle_frame_limit(30, -1), 240); // wraps bottom -> top
        assert_eq!(cycle_frame_limit(240, 1), 30); // wraps top -> bottom
        assert_eq!(cycle_frame_limit(80, 1), 120); // 80 snaps to 90, then +1 -> 120
    }

    #[test]
    fn adjust_stepped_u32_steps_and_clamps() {
        assert_eq!(adjust_stepped_u32(1920, 1, 160, 640, 7680), 2080);
        assert_eq!(adjust_stepped_u32(640, -1, 160, 640, 7680), 640); // clamped at min
        assert_eq!(adjust_stepped_u32(7680, 1, 160, 640, 7680), 7680); // clamped at max
    }

    #[test]
    fn cycle_log_level_wraps_both_directions() {
        assert_eq!(cycle_log_level("info", 1), "debug");
        assert_eq!(cycle_log_level("info", -1), "warn");
        assert_eq!(cycle_log_level("trace", 1), "error"); // wraps top -> bottom
        assert_eq!(cycle_log_level("error", -1), "trace"); // wraps bottom -> top
        assert_eq!(cycle_log_level("nonsense", 1), "debug"); // unknown restarts at info
    }

    #[test]
    fn adjust_stepped_moves_by_step_and_clamps() {
        assert_eq!(adjust_stepped(1.0, 1, 0.25, 0.5, 4.0), 1.25);
        assert_eq!(adjust_stepped(1.0, -1, 0.25, 0.5, 4.0), 0.75);
        assert_eq!(adjust_stepped(4.0, 1, 0.25, 0.5, 4.0), 4.0);
        assert_eq!(adjust_stepped(0.5, -1, 0.25, 0.5, 4.0), 0.5);
    }

    #[test]
    fn available_themes_lists_at_least_the_default() {
        let themes = available_themes(std::path::Path::new("this/does/not/exist"));
        assert!(themes.contains(&"default".to_string()));
    }

    #[test]
    fn available_wallpapers_lists_off_plus_image_files_only() {
        let base = std::env::temp_dir().join(format!("raeen-wallpapers-{}", std::process::id()));
        std::fs::create_dir_all(&base).expect("mkdir");
        std::fs::write(base.join("b.png"), b"x").expect("write");
        std::fs::write(base.join("a.JPG"), b"x").expect("write");
        std::fs::write(base.join("notes.txt"), b"x").expect("write");
        std::fs::create_dir_all(base.join("subdir")).expect("mkdir");

        let options = available_wallpapers(&base);
        assert_eq!(options, vec!["off", "a.JPG", "b.png"]);

        // A missing directory is just "off".
        assert_eq!(
            available_wallpapers(Path::new("this/does/not/exist")),
            vec!["off"]
        );

        let _ = std::fs::remove_dir_all(&base);
    }
}
