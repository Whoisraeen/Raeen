//! Settings screen (spec §3 screen 5, §10 SM2).
//!
//! A full-screen surface with sections — Video, Audio, Input, Game Folders,
//! Key Provider, Theme, System, Debug — bound to the real
//! [`raeen_core::config::EmulatorConfig`]. Navigation (which section/row is
//! focused, Up/Down between rows, Left/Right/Confirm to adjust, Back to
//! leave) lives in [`super::nav`] as pure state; this module only draws the
//! six sections for whichever `(section, row)` is currently focused, and
//! exposes the pure data tables `shell/mod.rs` needs to drive that nav
//! state (section names, per-section row counts, and a couple of small
//! value-stepping helpers).
//!
//! Key Provider is a path field only — the Shell stores and displays the
//! string but never reads, parses, or otherwise touches key material (spec
//! §11). Theme is populated from installed `themes/<name>/theme.toml`
//! directories on disk (spec §6, §10 SM2b) via [`available_themes`];
//! selecting one updates `general.selected_theme` and `shell/mod.rs`
//! reloads the active [`Theme`] from disk.

use super::nav::NavState;
use crate::theme::Theme;
use egui::{Align, Layout, RichText, UiBuilder};
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

/// Rows the Plugins section appends after its per-plugin rows: "Rescan
/// Plugins Folder" and "Open Plugins Folder".
pub const PLUGIN_ACTION_ROWS: usize = 2;

/// Number of rows in each section, in `SETTINGS_SECTION_NAMES` order. The
/// Game Folders section grows by one row per configured folder plus a
/// trailing "Add Folder" row; the Plugins section grows by one row per
/// registered present plugin plus its two fixed action rows — so this takes
/// both live counts.
pub fn settings_row_counts(game_folder_count: usize, plugin_count: usize) -> Vec<usize> {
    vec![
        11,
        3,
        3,
        game_folder_count + 1,
        1,
        1,
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

/// The user-facing label for a present-plugin config value: `"off"` reads as
/// `Off`, any other name is shown verbatim (the plugin's own id).
#[must_use]
pub fn upscaler_label(name: &str) -> String {
    if name == "off" {
        "Off".to_string()
    } else {
        name.to_string()
    }
}

/// Cycle the selected present-plugin name through `options` by `delta` steps
/// (wrapping). `options` is the "off" sentinel followed by the registered
/// plugin names. Pure so it is unit-testable without the global GPU registry.
#[must_use]
pub fn cycle_upscaler(current: &str, delta: i32, options: &[String]) -> String {
    if options.is_empty() {
        return current.to_string();
    }
    let idx = options.iter().position(|o| o == current).unwrap_or(0) as i32;
    let n = options.len() as i32;
    let next = (((idx + delta) % n) + n) % n;
    options[next as usize].clone()
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
/// §10 SM2b). Thin wrapper over [`crate::theme::loader::list_themes`] kept
/// here so `shell/mod.rs` has one place to go for "what can Theme cycle
/// through" without reaching into `theme::loader` directly.
pub fn available_themes(themes_root: &Path) -> Vec<String> {
    crate::theme::loader::list_themes(themes_root)
}

fn on_off(b: bool) -> &'static str {
    if b { "On" } else { "Off" }
}

/// Draw one row: a focus-highlighted label on the left, its current value
/// dimmed on the right.
fn row(
    ui: &mut egui::Ui,
    theme: &Theme,
    nav: &NavState,
    row_index: usize,
    label: &str,
    value: String,
) {
    let focused = nav.settings_row == row_index;
    let color = if focused {
        theme.palette.focus
    } else {
        theme.palette.text
    };
    let prefix = if focused { "\u{25B6} " } else { "   " };
    ui.horizontal(|ui| {
        ui.set_width(500.0);
        ui.label(
            RichText::new(format!("{prefix}{label}"))
                .color(color)
                .size(15.0),
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(
                RichText::new(value)
                    .color(theme.palette.text_dim)
                    .size(14.0),
            );
        });
    });
    ui.add_space(10.0);
}

/// Draw the full Settings surface. `new_folder_input`/`key_provider_input`
/// are Shell-owned scratch text buffers (not part of `nav::NavState`, which
/// stays free of raw text-entry state) bound to their respective text
/// fields.
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
    ui.painter().rect_filled(screen, 0.0, theme.palette.ground);

    ui.scope_builder(UiBuilder::new().max_rect(screen), |ui| {
        ui.add_space(28.0);
        ui.horizontal(|ui| {
            ui.add_space(54.0);
            ui.label(
                RichText::new("SETTINGS")
                    .color(theme.palette.text)
                    .size(28.0)
                    .strong(),
            );
        });
        ui.add_space(24.0);

        ui.horizontal_top(|ui| {
            ui.add_space(54.0);

            ui.vertical(|ui| {
                ui.set_width(200.0);
                for (i, name) in SETTINGS_SECTION_NAMES.iter().enumerate() {
                    let focused = i == nav.settings_section;
                    let color = if focused {
                        theme.palette.focus
                    } else {
                        theme.palette.text_dim
                    };
                    ui.label(RichText::new(*name).color(color).size(17.0).strong());
                    ui.add_space(16.0);
                }
            });

            ui.add_space(40.0);

            ui.vertical(|ui| {
                ui.set_width(560.0);
                match nav.settings_section {
                    SECTION_VIDEO => draw_video(ui, theme, nav, config),
                    SECTION_AUDIO => draw_audio(ui, theme, nav, config),
                    SECTION_CONTROLLER => draw_input(ui, theme, nav, config),
                    SECTION_GAME_FOLDERS => {
                        draw_game_folders(ui, theme, nav, config, new_folder_input)
                    }
                    SECTION_KEY_PROVIDER => draw_key_provider(ui, theme, nav, key_provider_input),
                    SECTION_THEME => draw_theme(ui, theme, nav, config),
                    SECTION_PLUGINS => draw_plugins(ui, theme, nav, plugins, plugin_failures),
                    SECTION_SYSTEM => draw_system(ui, theme, nav, updater),
                    SECTION_ADVANCED => draw_debug(ui, theme, nav, config),
                    _ => {}
                }
            });
        });

        ui.add_space(24.0);
        ui.horizontal(|ui| {
            ui.add_space(54.0);
            ui.label(
                RichText::new(format!(
                    "\u{2191}\u{2193}/Wheel Rows    \u{25C0}\u{25B6}/Enter/{} Adjust    Esc/{}/Right-click Back",
                    config.input.controller_icon_style.confirm(),
                    config.input.controller_icon_style.back(),
                ))
                    .color(theme.palette.text_dim)
                    .size(13.0),
            );
        });
    });

    // The screen uses fixed explicit anchors. Put transparent interaction
    // rectangles over those same anchors so pointer users can select a
    // section or directly activate a setting without changing the painter-
    // driven visual layout.
    let section_top = screen.top() + 86.0;
    for section in 0..SETTINGS_SECTION_NAMES.len() {
        let rect = egui::Rect::from_min_size(
            egui::pos2(screen.left() + 54.0, section_top + section as f32 * 33.0),
            egui::vec2(200.0, 30.0),
        );
        if ui
            .interact(
                rect,
                ui.id().with(("settings-section", section)),
                egui::Sense::click(),
            )
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked()
        {
            return Some(SettingsClick::Section(section));
        }
    }
    let row_count =
        settings_row_counts(config.paths.game_folders.len(), plugins.len())[nav.settings_section];
    for row_index in 0..row_count {
        let rect = egui::Rect::from_min_size(
            egui::pos2(screen.left() + 294.0, section_top + row_index as f32 * 28.0),
            egui::vec2(560.0, 27.0),
        );
        if ui
            .interact(
                rect,
                ui.id()
                    .with(("settings-row", nav.settings_section, row_index)),
                egui::Sense::click(),
            )
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked()
        {
            return Some(SettingsClick::Row(row_index));
        }
    }
    None
}

fn draw_video(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig) {
    row(
        ui,
        theme,
        nav,
        0,
        "Resolution Scale",
        format!("{:.2}x", config.graphics.resolution_scale),
    );
    row(
        ui,
        theme,
        nav,
        1,
        "Fullscreen",
        on_off(config.general.fullscreen).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        2,
        "Shader Cache",
        on_off(config.graphics.shader_cache).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        3,
        "Validation Layers",
        on_off(config.graphics.validation_layers).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        4,
        "VSync",
        on_off(config.general.vsync).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        5,
        "Frame Limit",
        format!("{} Hz", config.graphics.frame_limit),
    );
    row(
        ui,
        theme,
        nav,
        6,
        "GPU Device",
        if config.graphics.gpu_device_index == 0 {
            "Auto (best)".to_string()
        } else {
            format!("Device {}", config.graphics.gpu_device_index)
        },
    );
    row(
        ui,
        theme,
        nav,
        7,
        "Window Width",
        format!("{} px", config.general.window_width),
    );
    row(
        ui,
        theme,
        nav,
        8,
        "Window Height",
        format!("{} px", config.general.window_height),
    );
    row(
        ui,
        theme,
        nav,
        9,
        "Upscaler / Frame Gen",
        upscaler_label(&config.graphics.upscaler),
    );
    row(
        ui,
        theme,
        nav,
        10,
        "Upscale Factor",
        format!("{:.2}x", config.graphics.present_upscale),
    );
    ui.add_space(10.0);
    ui.label(
        RichText::new(
            "Window size applies when Fullscreen is Off. Frame Limit and GPU Device apply on the next launch. \
             Upscaler applies live; only Raeen's built-in plugins ship — proprietary ones (e.g. DLSS) are user-supplied.",
        )
        .color(theme.palette.text_faint)
        .size(12.0),
    );
}

fn draw_audio(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig) {
    row(
        ui,
        theme,
        nav,
        0,
        "Audio Enabled",
        on_off(config.audio.enabled).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        1,
        "Master Volume",
        format!("{}%", (config.audio.volume * 100.0).round() as i32),
    );
    row(
        ui,
        theme,
        nav,
        2,
        "Spatial Audio",
        on_off(config.audio.spatial_audio).to_string(),
    );
}

fn draw_input(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig) {
    row(
        ui,
        theme,
        nav,
        0,
        "DualSense Features",
        on_off(config.input.dualsense_features).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        1,
        "Stick Deadzone",
        format!("{:.2}", config.input.deadzone),
    );
    row(
        ui,
        theme,
        nav,
        2,
        "Button Icon Style",
        config.input.controller_icon_style.label().to_string(),
    );
}

fn draw_game_folders(
    ui: &mut egui::Ui,
    theme: &Theme,
    nav: &NavState,
    config: &EmulatorConfig,
    new_folder_input: &mut String,
) {
    for (i, path) in config.paths.game_folders.iter().enumerate() {
        row(ui, theme, nav, i, "Folder", path.display().to_string());
    }
    let add_row = config.paths.game_folders.len();
    let focused = nav.settings_row == add_row;
    let color = if focused {
        theme.palette.focus
    } else {
        theme.palette.text
    };
    let prefix = if focused { "\u{25B6} " } else { "   " };
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{prefix}Add Folder"))
                .color(color)
                .size(15.0),
        );
        ui.text_edit_singleline(new_folder_input);
    });
    ui.add_space(10.0);
    ui.label(
        RichText::new(
            "Confirm on a folder row removes it; Confirm on \"Add Folder\" adds the typed path.",
        )
        .color(theme.palette.text_faint)
        .size(12.0),
    );
}

fn draw_key_provider(
    ui: &mut egui::Ui,
    theme: &Theme,
    nav: &NavState,
    key_provider_input: &mut String,
) {
    let focused = nav.settings_row == 0;
    let color = if focused {
        theme.palette.focus
    } else {
        theme.palette.text
    };
    let prefix = if focused { "\u{25B6} " } else { "   " };
    ui.label(
        RichText::new(format!("{prefix}KeyProvider Path"))
            .color(color)
            .size(15.0),
    );
    ui.add_space(6.0);
    ui.text_edit_singleline(key_provider_input);
    ui.add_space(10.0);
    ui.label(
        RichText::new("Path only — Raeen never reads or handles key material here.")
            .color(theme.palette.text_faint)
            .size(12.0),
    );
}

fn draw_theme(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig) {
    let focused = nav.settings_row == 0;
    let color = if focused {
        theme.palette.focus
    } else {
        theme.palette.text
    };
    let prefix = if focused { "\u{25B6} " } else { "   " };
    ui.label(
        RichText::new(format!("{prefix}Theme: {}", config.general.selected_theme))
            .color(color)
            .size(15.0),
    );
    ui.add_space(10.0);
    ui.label(
        RichText::new("Left/Right or Confirm cycles installed themes (themes/<name>/theme.toml).")
            .color(theme.palette.text_faint)
            .size(12.0),
    );
}

/// Plugins section: one row per registered present plugin (name on the left;
/// capabilities, origin, and the Active marker on the right), then the two
/// action rows. Confirm on a plugin row activates it (or deactivates it when
/// already active); the Shell owns that logic — this only draws.
fn draw_plugins(
    ui: &mut egui::Ui,
    theme: &Theme,
    nav: &NavState,
    plugins: &[PluginRowInfo],
    plugin_failures: &[String],
) {
    for (i, plugin) in plugins.iter().enumerate() {
        let value = if plugin.active {
            format!("{} · {} · Active", plugin.capabilities, plugin.source)
        } else {
            format!("{} · {}", plugin.capabilities, plugin.source)
        };
        row(ui, theme, nav, i, &plugin.name, value);
    }
    row(
        ui,
        theme,
        nav,
        plugins.len(),
        "Rescan Plugins Folder",
        String::new(),
    );
    row(
        ui,
        theme,
        nav,
        plugins.len() + 1,
        "Open Plugins Folder",
        String::new(),
    );
    ui.add_space(10.0);
    ui.label(
        RichText::new(
            "Confirm activates a plugin (again to deactivate). Built-ins ship with Raeen; \
             everything else is user-supplied in plugins/ and loaded at startup or on rescan.",
        )
        .color(theme.palette.text_faint)
        .size(12.0),
    );
    if !plugin_failures.is_empty() {
        ui.add_space(10.0);
        ui.label(
            RichText::new("Refused on the last scan:")
                .color(theme.palette.text)
                .size(13.0),
        );
        for failure in plugin_failures {
            ui.label(
                RichText::new(format!("\u{2022} {failure}"))
                    .color(theme.palette.text_faint)
                    .size(12.0),
            );
        }
    }
}

/// System section: current version + the updater's single action row
/// ("Check for Updates" → "Download Update" → "Restart & Update" depending
/// on [`crate::updater::UpdaterState`]).
fn draw_system(
    ui: &mut egui::Ui,
    theme: &Theme,
    nav: &NavState,
    updater: &crate::updater::UpdaterState,
) {
    row(
        ui,
        theme,
        nav,
        0,
        "Version",
        format!("v{}", raeen_core::VERSION),
    );
    row(
        ui,
        theme,
        nav,
        1,
        updater.action_label(),
        updater.status_line(),
    );
    ui.add_space(10.0);
    ui.label(
        RichText::new("Updates are fetched from GitHub Releases and applied on restart.")
            .color(theme.palette.text_faint)
            .size(12.0),
    );
}

/// Debug section: developer diagnostics, all persisted to `config.toml`'s
/// `[debug]` table — logging verbosity plus the GPU/shader/syscall dump
/// toggles the runtime reads when a session launches.
fn draw_debug(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig) {
    row(
        ui,
        theme,
        nav,
        0,
        "Logging",
        on_off(config.debug.logging).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        1,
        "Log Level",
        config.debug.log_level.clone(),
    );
    row(
        ui,
        theme,
        nav,
        2,
        "Trace HLE Calls",
        on_off(config.debug.trace_syscalls).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        3,
        "Dump GPU Resources",
        on_off(config.debug.dump_gpu_commands).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        4,
        "Dump Shaders",
        on_off(config.debug.dump_shaders).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        5,
        "Dump Frames",
        on_off(config.debug.dump_frames).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        6,
        "Call Stats",
        on_off(config.debug.call_stats).to_string(),
    );
    row(
        ui,
        theme,
        nav,
        7,
        "Stall Dump",
        on_off(config.debug.stall_dump).to_string(),
    );
    ui.add_space(10.0);
    ui.label(
        RichText::new(
            "Developer diagnostics. Logging is live; the dumps and traces apply on the next launch.",
        )
        .color(theme.palette.text_faint)
        .size(12.0),
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
        assert_eq!(settings_row_counts(0, 0)[SECTION_GAME_FOLDERS], 1); // just the "Add Folder" row
        assert_eq!(settings_row_counts(2, 0)[SECTION_GAME_FOLDERS], 3); // 2 folders + "Add Folder"
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
        assert_eq!(counts[SECTION_VIDEO], 11); // Video (+ Frame Limit, GPU Device, Window W/H, Upscaler, Factor)
        assert_eq!(counts[SECTION_AUDIO], 3); // Audio
        assert_eq!(counts[SECTION_CONTROLLER], 3); // Controller
        assert_eq!(counts[SECTION_KEY_PROVIDER], 1); // Key Provider
        assert_eq!(counts[SECTION_THEME], 1); // Theme
        assert_eq!(counts[SECTION_SYSTEM], 2); // System (version + updater action)
        assert_eq!(counts[SECTION_ADVANCED], 8); // Advanced (Logging, Log Level, 3 traces/dumps + 3 more)
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
}
