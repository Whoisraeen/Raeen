//! Settings screen (spec §3 screen 5, §10 SM2).
//!
//! A full-screen surface with six sections — Video, Audio, Input, Game
//! Folders, Key Provider, Theme — bound to the real
//! [`xps5x_core::config::EmulatorConfig`]. Navigation (which section/row is
//! focused, Up/Down between rows, Left/Right/Confirm to adjust, Back to
//! leave) lives in [`super::nav`] as pure state; this module only draws the
//! six sections for whichever `(section, row)` is currently focused, and
//! exposes the pure data tables `shell/mod.rs` needs to drive that nav
//! state (section names, per-section row counts, and a couple of small
//! value-stepping helpers).
//!
//! Key Provider is a path field only — the Shell stores and displays the
//! string but never reads, parses, or otherwise touches key material (spec
//! §11). Theme is a single-item selector for now; on-disk theme
//! installation (and populating this list from `themes/*` directories) is
//! SM2b, not this milestone (see `available_themes` below).

use super::nav::NavState;
use crate::theme::Theme;
use egui::{Align, Layout, RichText, UiBuilder};
use xps5x_core::config::EmulatorConfig;

/// Section names, in the order `nav::NavState::settings_section` indexes.
pub const SETTINGS_SECTION_NAMES: [&str; 6] = ["Video", "Audio", "Input", "Game Folders", "Key Provider", "Theme"];

/// Number of rows in each section, in `SETTINGS_SECTION_NAMES` order. The
/// Game Folders section grows by one row per configured folder plus a
/// trailing "Add Folder" row, so this takes the current folder count.
pub fn settings_row_counts(game_folder_count: usize) -> Vec<usize> {
    vec![4, 3, 2, game_folder_count + 1, 1, 1]
}

/// Step `value` by `delta` steps of size `step`, clamped to `[min, max]`.
/// Used for every numeric Settings row (resolution scale, volume,
/// deadzone) so they share one clamping rule instead of each hand-rolling
/// it slightly differently.
pub fn adjust_stepped(value: f32, delta: i32, step: f32, min: f32, max: f32) -> f32 {
    (value + delta as f32 * step).clamp(min, max)
}

/// Theme names the Settings screen's Theme selector can choose between.
/// SM2a ships only the in-code default theme, so this is a single-item
/// list. TODO(SM2b): populate this from installed `themes/<name>/theme.toml`
/// directories on disk instead (spec §6, §10).
pub fn available_themes() -> Vec<String> {
    vec!["default".to_string()]
}

fn on_off(b: bool) -> &'static str {
    if b { "On" } else { "Off" }
}

/// Draw one row: a focus-highlighted label on the left, its current value
/// dimmed on the right.
fn row(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, row_index: usize, label: &str, value: String) {
    let focused = nav.settings_row == row_index;
    let color = if focused { theme.palette.focus } else { theme.palette.text };
    let prefix = if focused { "\u{25B6} " } else { "   " };
    ui.horizontal(|ui| {
        ui.set_width(500.0);
        ui.label(RichText::new(format!("{prefix}{label}")).color(color).size(15.0));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(value).color(theme.palette.text_dim).size(14.0));
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
) {
    let screen = ui.max_rect();
    ui.painter().rect_filled(screen, 0.0, theme.palette.ground);

    ui.scope_builder(UiBuilder::new().max_rect(screen), |ui| {
        ui.add_space(28.0);
        ui.horizontal(|ui| {
            ui.add_space(54.0);
            ui.label(RichText::new("SETTINGS").color(theme.palette.text).size(28.0).strong());
        });
        ui.add_space(24.0);

        ui.horizontal_top(|ui| {
            ui.add_space(54.0);

            ui.vertical(|ui| {
                ui.set_width(200.0);
                for (i, name) in SETTINGS_SECTION_NAMES.iter().enumerate() {
                    let focused = i == nav.settings_section;
                    let color = if focused { theme.palette.focus } else { theme.palette.text_dim };
                    ui.label(RichText::new(*name).color(color).size(17.0).strong());
                    ui.add_space(16.0);
                }
            });

            ui.add_space(40.0);

            ui.vertical(|ui| {
                ui.set_width(560.0);
                match nav.settings_section {
                    0 => draw_video(ui, theme, nav, config),
                    1 => draw_audio(ui, theme, nav, config),
                    2 => draw_input(ui, theme, nav, config),
                    3 => draw_game_folders(ui, theme, nav, config, new_folder_input),
                    4 => draw_key_provider(ui, theme, nav, key_provider_input),
                    5 => draw_theme(ui, theme, nav, config),
                    _ => {}
                }
            });
        });

        ui.add_space(24.0);
        ui.horizontal(|ui| {
            ui.add_space(54.0);
            ui.label(
                RichText::new("\u{2191}\u{2193} Rows    \u{25C0}\u{25B6}/Enter Adjust    Esc Back")
                    .color(theme.palette.text_dim)
                    .size(13.0),
            );
        });
    });
}

fn draw_video(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig) {
    row(ui, theme, nav, 0, "Resolution Scale", format!("{:.2}x", config.graphics.resolution_scale));
    row(ui, theme, nav, 1, "Fullscreen", on_off(config.general.fullscreen).to_string());
    row(ui, theme, nav, 2, "Shader Cache", on_off(config.graphics.shader_cache).to_string());
    row(ui, theme, nav, 3, "Validation Layers", on_off(config.graphics.validation_layers).to_string());
}

fn draw_audio(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig) {
    row(ui, theme, nav, 0, "Audio Enabled", on_off(config.audio.enabled).to_string());
    row(ui, theme, nav, 1, "Master Volume", format!("{}%", (config.audio.volume * 100.0).round() as i32));
    row(ui, theme, nav, 2, "Spatial Audio", on_off(config.audio.spatial_audio).to_string());
}

fn draw_input(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig) {
    row(ui, theme, nav, 0, "DualSense Features", on_off(config.input.dualsense_features).to_string());
    row(ui, theme, nav, 1, "Stick Deadzone", format!("{:.2}", config.input.deadzone));
}

fn draw_game_folders(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig, new_folder_input: &mut String) {
    for (i, path) in config.paths.game_folders.iter().enumerate() {
        row(ui, theme, nav, i, "Folder", path.display().to_string());
    }
    let add_row = config.paths.game_folders.len();
    let focused = nav.settings_row == add_row;
    let color = if focused { theme.palette.focus } else { theme.palette.text };
    let prefix = if focused { "\u{25B6} " } else { "   " };
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{prefix}Add Folder")).color(color).size(15.0));
        ui.text_edit_singleline(new_folder_input);
    });
    ui.add_space(10.0);
    ui.label(
        RichText::new("Confirm on a folder row removes it; Confirm on \"Add Folder\" adds the typed path.")
            .color(theme.palette.text_faint)
            .size(12.0),
    );
}

fn draw_key_provider(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, key_provider_input: &mut String) {
    let focused = nav.settings_row == 0;
    let color = if focused { theme.palette.focus } else { theme.palette.text };
    let prefix = if focused { "\u{25B6} " } else { "   " };
    ui.label(RichText::new(format!("{prefix}KeyProvider Path")).color(color).size(15.0));
    ui.add_space(6.0);
    ui.text_edit_singleline(key_provider_input);
    ui.add_space(10.0);
    ui.label(
        RichText::new("Path only — XPS5X never reads or handles key material here.")
            .color(theme.palette.text_faint)
            .size(12.0),
    );
}

fn draw_theme(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, config: &EmulatorConfig) {
    let focused = nav.settings_row == 0;
    let color = if focused { theme.palette.focus } else { theme.palette.text };
    let prefix = if focused { "\u{25B6} " } else { "   " };
    ui.label(RichText::new(format!("{prefix}Theme: {}", config.general.selected_theme)).color(color).size(15.0));
    ui.add_space(10.0);
    ui.label(
        RichText::new("Left/Right or Confirm cycles themes. TODO(SM2b): populate from installed themes/ directories.")
            .color(theme.palette.text_faint)
            .size(12.0),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_names_and_row_counts_stay_in_step() {
        assert_eq!(settings_row_counts(0).len(), SETTINGS_SECTION_NAMES.len());
    }

    #[test]
    fn game_folders_row_count_tracks_the_configured_folder_count() {
        assert_eq!(settings_row_counts(0)[3], 1); // just the "Add Folder" row
        assert_eq!(settings_row_counts(2)[3], 3); // 2 folders + "Add Folder"
    }

    #[test]
    fn other_sections_row_counts_are_fixed() {
        let counts = settings_row_counts(5);
        assert_eq!(counts[0], 4); // Video
        assert_eq!(counts[1], 3); // Audio
        assert_eq!(counts[2], 2); // Input
        assert_eq!(counts[4], 1); // Key Provider
        assert_eq!(counts[5], 1); // Theme
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
        let themes = available_themes();
        assert!(themes.contains(&"default".to_string()));
    }
}
