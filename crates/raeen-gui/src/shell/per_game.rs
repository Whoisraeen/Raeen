//! Per-game settings overrides — a title carries its own graphics/logging
//! config that wins over the global [`EmulatorConfig`] at launch.
//!
//! This is our answer to (and improvement on) SharpEmu's per-game settings
//! (`PerGameSettings.cs` / `PerGameSettingsDialog.cs`, PRs #453 "PerGameSettings
//! Null toggles" and #430 "Gui Settings Null list Entries"): every field is an
//! `Option` that, when `None`, *inherits* the global value, and a title whose
//! overrides are all `None` persists nothing at all (its file is deleted). Like
//! SharpEmu, load is fully null-safe — a missing or corrupt file yields the
//! all-inherit default rather than failing the launch.
//!
//! Where SharpEmu's per-game panel overrides only logging/dynlib knobs, ours
//! overrides the settings that actually change how a title *renders*:
//! Resolution Scale (SharpEmu's global-only `RenderResolutionScale`, PR #468 —
//! here made per-title), GPU device selection, validation layers, and log
//! level. All four are re-applied through the same process-wide setters the
//! Shell uses for global settings (`AgcGpuSession::set_runtime_config`,
//! `logging::set_level`), so an un-overridden title cleanly resets any previous
//! title's overrides back to the global baseline.
//!
//! Overrides live in `<config_dir>/per_game/<title-id>.json` — the game's own
//! stable [`crate::library::LibraryItem::id`], sanitized for the filesystem —
//! mirroring SharpEmu's `user/custom_configs/<titleId>.json`.

use super::nav::NavState;
use super::settings;
use crate::compat;
use crate::theme::Theme;
use egui::{Align, Layout, RichText, UiBuilder};
use raeen_core::config::EmulatorConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Number of rows the Game Options overlay exposes: four override fields plus
/// a trailing "Reset to Global Defaults" row.
pub const ROW_COUNT: usize = 5;

/// One row in the Game Options overlay.
const ROW_RESOLUTION: usize = 0;
const ROW_GPU_DEVICE: usize = 1;
const ROW_VALIDATION: usize = 2;
const ROW_LOG_LEVEL: usize = 3;
const ROW_RESET: usize = 4;

/// A title's persisted per-game overrides. Every field is `None` = "inherit the
/// global setting"; a `Some` wins over the global value at launch.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PerGameSettings {
    /// Internal render-resolution scale (SharpEmu `RenderResolutionScale`,
    /// PR #468 — global there, per-title here). `1.0` = native PS5.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution_scale: Option<f32>,
    /// Force a specific Vulkan physical device for this title (0 = auto/best).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_device_index: Option<u32>,
    /// Enable GPU validation layers for this title only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_layers: Option<bool>,
    /// Log-level filter for this title's session (`error`..`trace`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_level: Option<String>,
}

/// Pointer selection reported to the Shell after the overlay draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameOptionsClick {
    Row(usize),
}

/// Read-only per-title trophy summary shown in the Game Options overlay,
/// computed from the local unlock store
/// (`savedata/<title>-trophies.json`, [`raeen_core::trophies::TrophyStore`])
/// when the overlay opens. Counts and times only — trophy *names/grades*
/// live in the title's encrypted trophy pack, which Raeen cannot parse, so
/// no name is ever shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrophySummary {
    /// Locally unlocked trophy count.
    pub unlocked: usize,
    /// Unix-ms timestamp of the most recent unlock.
    pub last_unlock_ms: Option<u64>,
}

impl TrophySummary {
    /// The one-line display, e.g.
    /// `Trophies: 3 unlocked · last 2026-07-27 18:22:05 UTC`.
    pub fn line(&self) -> String {
        match (self.unlocked, self.last_unlock_ms) {
            (0, _) => "Trophies: none unlocked yet".to_owned(),
            (n, Some(ms)) => format!(
                "Trophies: {n} unlocked \u{00B7} last {}",
                crate::crash_report::utc_display(ms / 1000)
            ),
            (n, None) => format!("Trophies: {n} unlocked"),
        }
    }
}

impl PerGameSettings {
    /// Every override is `None` — the title inherits the global config wholesale
    /// and therefore persists no file (mirrors SharpEmu `PerGameSettings.IsEmpty`).
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Directory per-game override files live in: `<config_dir>/per_game`.
    pub fn store_dir(config_path: &Path) -> PathBuf {
        let base = config_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("per_game")
    }

    /// Absolute path of a title's override file within `store_dir`.
    pub fn path_for(store_dir: &Path, id: &str) -> PathBuf {
        store_dir.join(format!("{}.json", sanitize_id(id)))
    }

    /// Load a title's overrides. A missing or unreadable/corrupt file yields the
    /// all-inherit default — never an error — so a bad file can never block a
    /// launch (SharpEmu PR #453/#430 null-safety, taken further: even malformed
    /// JSON degrades to "inherit global").
    pub fn load(store_dir: &Path, id: &str) -> Self {
        let path = Self::path_for(store_dir, id);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        match serde_json::from_str::<Self>(&text) {
            Ok(settings) => settings,
            Err(err) => {
                tracing::warn!(path = %path.display(), error = %err, "per-game settings malformed — using global defaults");
                Self::default()
            }
        }
    }

    /// Persist a title's overrides, or delete the file when nothing is
    /// overridden (keeps the store free of empty inherit-everything files).
    /// Best-effort: a write failure is logged, never surfaced as an error.
    pub fn save(&self, store_dir: &Path, id: &str) {
        let path = Self::path_for(store_dir, id);
        if self.is_empty() {
            if path.exists()
                && let Err(err) = std::fs::remove_file(&path)
            {
                tracing::warn!(path = %path.display(), error = %err, "failed to clear per-game settings");
            }
            return;
        }
        if let Err(err) = std::fs::create_dir_all(store_dir) {
            tracing::warn!(dir = %store_dir.display(), error = %err, "failed to create per-game settings dir");
            return;
        }
        match serde_json::to_string_pretty(self) {
            Ok(json) => {
                if let Err(err) = std::fs::write(&path, json) {
                    tracing::warn!(path = %path.display(), error = %err, "failed to save per-game settings");
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "failed to serialize per-game settings");
            }
        }
    }

    /// Whether `row` currently carries an override (drives the [Override]/[Global]
    /// badge and, at launch, whether the field wins over the global value).
    pub fn is_overridden(&self, row: usize) -> bool {
        match row {
            ROW_RESOLUTION => self.resolution_scale.is_some(),
            ROW_GPU_DEVICE => self.gpu_device_index.is_some(),
            ROW_VALIDATION => self.validation_layers.is_some(),
            ROW_LOG_LEVEL => self.log_level.is_some(),
            _ => false,
        }
    }

    /// Confirm on `row`: toggle whether it overrides. Enabling seeds the
    /// override from the current global value; disabling reverts to inherit.
    /// The trailing Reset row clears every override at once.
    pub fn toggle_override(&mut self, row: usize, global: &EmulatorConfig) {
        match row {
            ROW_RESOLUTION => {
                self.resolution_scale =
                    toggle(self.resolution_scale, global.graphics.resolution_scale)
            }
            ROW_GPU_DEVICE => {
                self.gpu_device_index =
                    toggle(self.gpu_device_index, global.graphics.gpu_device_index)
            }
            ROW_VALIDATION => {
                self.validation_layers =
                    toggle(self.validation_layers, global.graphics.validation_layers)
            }
            ROW_LOG_LEVEL => {
                self.log_level = match self.log_level {
                    Some(_) => None,
                    None => Some(global.debug.log_level.clone()),
                }
            }
            ROW_RESET => *self = Self::default(),
            _ => {}
        }
    }

    /// Left/Right on `row`: turn the override on (if it was inheriting) and step
    /// its value. A bool flips; the Reset row has nothing to step.
    pub fn adjust(&mut self, row: usize, delta: i32, global: &EmulatorConfig) {
        match row {
            ROW_RESOLUTION => {
                let cur = self
                    .resolution_scale
                    .unwrap_or(global.graphics.resolution_scale);
                self.resolution_scale = Some(settings::adjust_stepped(cur, delta, 0.25, 0.5, 4.0));
            }
            ROW_GPU_DEVICE => {
                let cur = self
                    .gpu_device_index
                    .unwrap_or(global.graphics.gpu_device_index);
                self.gpu_device_index = Some(settings::adjust_stepped_u32(cur, delta, 1, 0, 8));
            }
            ROW_VALIDATION => {
                let cur = self
                    .validation_layers
                    .unwrap_or(global.graphics.validation_layers);
                self.validation_layers = Some(!cur);
            }
            ROW_LOG_LEVEL => {
                let cur = self
                    .log_level
                    .clone()
                    .unwrap_or_else(|| global.debug.log_level.clone());
                self.log_level = Some(settings::cycle_log_level(&cur, delta));
            }
            _ => {}
        }
    }

    /// Fold this title's overrides into a copy of `global` — the effective
    /// config the launcher runs the title under. Un-overridden fields keep the
    /// global value, so a fresh (empty) override set returns `global` unchanged.
    pub fn effective(&self, global: &EmulatorConfig) -> EmulatorConfig {
        let mut cfg = global.clone();
        if let Some(v) = self.resolution_scale {
            cfg.graphics.resolution_scale = v;
        }
        if let Some(v) = self.gpu_device_index {
            cfg.graphics.gpu_device_index = v;
        }
        if let Some(v) = self.validation_layers {
            cfg.graphics.validation_layers = v;
        }
        if let Some(v) = &self.log_level {
            cfg.debug.log_level = v.clone();
        }
        cfg
    }

    /// The effective value string shown for `row`: the override if set, else the
    /// inherited global value.
    fn effective_value(&self, row: usize, global: &EmulatorConfig) -> String {
        match row {
            ROW_RESOLUTION => format!(
                "{:.2}x",
                self.resolution_scale
                    .unwrap_or(global.graphics.resolution_scale)
            ),
            ROW_GPU_DEVICE => {
                let idx = self
                    .gpu_device_index
                    .unwrap_or(global.graphics.gpu_device_index);
                if idx == 0 {
                    "Auto (best)".to_string()
                } else {
                    format!("Device {idx}")
                }
            }
            ROW_VALIDATION => {
                let on = self
                    .validation_layers
                    .unwrap_or(global.graphics.validation_layers);
                if on { "On" } else { "Off" }.to_string()
            }
            ROW_LOG_LEVEL => self
                .log_level
                .clone()
                .unwrap_or_else(|| global.debug.log_level.clone()),
            _ => String::new(),
        }
    }
}

/// Toggle an `Option<T>` override: `Some` clears to inherit, `None` seeds from
/// the global default.
fn toggle<T>(current: Option<T>, global_default: T) -> Option<T> {
    match current {
        Some(_) => None,
        None => Some(global_default),
    }
}

/// Replace filesystem-hostile characters in a title id so it is a safe file
/// stem (matches the spirit of SharpEmu `SanitizeTitleId`).
fn sanitize_id(id: &str) -> String {
    let cleaned: String = id
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "UNKNOWN".to_string()
    } else {
        cleaned
    }
}

const ROW_LABELS: [&str; ROW_COUNT] = [
    "Resolution Scale",
    "GPU Device",
    "Validation Layers",
    "Log Level",
    "Reset to Global Defaults",
];

/// Draw the Game Options overlay for `title`, bound to `draft` (the in-progress
/// override set) resolved against the global `config`. `badge` is the title's
/// resolved compatibility badge, shown with its one-line provenance
/// ("baseline YYYY-MM-DD" vs "last session") — absent for Untested titles.
/// Returns a pointer selection when a row is clicked, so mouse users get the
/// same reach as the pad's Up/Down + Confirm.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    theme: &Theme,
    nav: &NavState,
    config: &EmulatorConfig,
    draft: &PerGameSettings,
    title: &str,
<<<<<<< HEAD
    badge: Option<&compat::TitleBadge>,
=======
    trophies: Option<&TrophySummary>,
>>>>>>> worktree-agent-ab8451f5668e88c3c
) -> Option<GameOptionsClick> {
    let screen = ui.max_rect();
    ui.painter().rect_filled(screen, 0.0, theme.palette.ground);

    ui.scope_builder(UiBuilder::new().max_rect(screen), |ui| {
        ui.add_space(28.0);
        ui.horizontal(|ui| {
            ui.add_space(54.0);
            ui.label(
                RichText::new("GAME OPTIONS")
                    .color(theme.palette.text)
                    .size(28.0)
                    .strong(),
            );
        });
        ui.horizontal(|ui| {
            ui.add_space(54.0);
            ui.label(
                RichText::new(title)
                    .color(theme.palette.focus)
                    .size(16.0)
                    .strong(),
            );
            // Compatibility badge + provenance, on the title row so the
            // fixed-anchor row hit-rects below stay aligned. Untested titles
            // show nothing.
            if let Some(badge) = badge {
                ui.add_space(14.0);
                ui.label(
                    RichText::new("\u{25CF}")
                        .color(compat::badge_color(&theme.palette, badge.level))
                        .size(12.0),
                );
                ui.label(
                    RichText::new(format!(
                        "{} \u{00B7} {}",
                        badge.level.label(),
                        badge.provenance
                    ))
                    .color(theme.palette.text_dim)
                    .size(13.0),
                );
            }
        });
        ui.add_space(18.0);

        ui.horizontal(|ui| {
            ui.add_space(54.0);
            ui.label(
                RichText::new(
                    "Overrides for this title only. Rows marked Global inherit Settings ▸ Video.",
                )
                .color(theme.palette.text_faint)
                .size(12.0),
            );
        });
        ui.add_space(14.0);

        for row in 0..ROW_COUNT {
            draw_row(ui, theme, nav, config, draft, row);
        }

        // Trophy summary — drawn *after* the option rows so the fixed
        // pointer hit-rects above stay aligned with the painted rows.
        // Display-only: counts/times from the local unlock store, never a
        // trophy name (definitions are unavailable — see [`TrophySummary`]).
        if let Some(summary) = trophies {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.add_space(54.0);
                ui.label(
                    RichText::new(summary.line())
                        .color(theme.palette.text_dim)
                        .size(13.0),
                );
            });
        }

        ui.add_space(20.0);
        ui.horizontal(|ui| {
            ui.add_space(54.0);
            ui.label(
                RichText::new(format!(
                    "\u{2191}\u{2193} Rows    \u{25C0}\u{25B6} Adjust    Enter/{} Override on/off    Esc/{} Back",
                    config.input.controller_icon_style.confirm(),
                    config.input.controller_icon_style.back(),
                ))
                .color(theme.palette.text_dim)
                .size(13.0),
            );
        });
    });

    // Transparent hit rects over each row for pointer users (same anchors the
    // painter-driven rows use).
    let row_top = screen.top() + 150.0;
    for row in 0..ROW_COUNT {
        let rect = egui::Rect::from_min_size(
            egui::pos2(screen.left() + 54.0, row_top + row as f32 * 32.0),
            egui::vec2(620.0, 30.0),
        );
        if ui
            .interact(
                rect,
                ui.id().with(("game-option", row)),
                egui::Sense::click(),
            )
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked()
        {
            return Some(GameOptionsClick::Row(row));
        }
    }
    None
}

fn draw_row(
    ui: &mut egui::Ui,
    theme: &Theme,
    nav: &NavState,
    config: &EmulatorConfig,
    draft: &PerGameSettings,
    row: usize,
) {
    let focused = nav.game_options_row == row;
    let label_color = if focused {
        theme.palette.focus
    } else {
        theme.palette.text
    };
    let prefix = if focused { "\u{25B6} " } else { "   " };

    ui.horizontal(|ui| {
        ui.add_space(54.0);
        ui.set_width(560.0);
        ui.label(
            RichText::new(format!("{prefix}{}", ROW_LABELS[row]))
                .color(label_color)
                .size(15.0),
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if row == ROW_RESET {
                return;
            }
            let overridden = draft.is_overridden(row);
            ui.label(
                RichText::new(draft.effective_value(row, config))
                    .color(theme.palette.text_dim)
                    .size(14.0),
            );
            ui.add_space(12.0);
            let (badge, badge_color) = if overridden {
                ("[Override]", theme.palette.focus)
            } else {
                ("[Global]", theme.palette.text_faint)
            };
            ui.label(RichText::new(badge).color(badge_color).size(12.0).strong());
        });
    });
    ui.add_space(12.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn global() -> EmulatorConfig {
        EmulatorConfig::default()
    }

    #[test]
    fn default_inherits_everything_and_is_empty() {
        let pg = PerGameSettings::default();
        assert!(pg.is_empty());
        // Effective equals global for an empty override set.
        let g = global();
        let eff = pg.effective(&g);
        assert_eq!(eff.graphics.resolution_scale, g.graphics.resolution_scale);
        assert_eq!(eff.debug.log_level, g.debug.log_level);
    }

    #[test]
    fn adjust_turns_override_on_and_steps_from_global() {
        let g = global();
        let mut pg = PerGameSettings::default();
        assert!(!pg.is_overridden(ROW_RESOLUTION));
        pg.adjust(ROW_RESOLUTION, 1, &g);
        assert!(pg.is_overridden(ROW_RESOLUTION));
        // Global default is 1.0; one +0.25 step -> 1.25.
        assert_eq!(pg.resolution_scale, Some(1.25));
        assert_eq!(pg.effective(&g).graphics.resolution_scale, 1.25);
    }

    #[test]
    fn confirm_toggles_override_off_then_on() {
        let g = global();
        let mut pg = PerGameSettings::default();
        // Enable at the global value.
        pg.toggle_override(ROW_RESOLUTION, &g);
        assert_eq!(pg.resolution_scale, Some(g.graphics.resolution_scale));
        // Disable back to inherit.
        pg.toggle_override(ROW_RESOLUTION, &g);
        assert_eq!(pg.resolution_scale, None);
    }

    #[test]
    fn validation_adjust_flips_bool() {
        let g = global(); // validation defaults off
        let mut pg = PerGameSettings::default();
        pg.adjust(ROW_VALIDATION, 1, &g);
        assert_eq!(pg.validation_layers, Some(true));
        assert!(pg.effective(&g).graphics.validation_layers);
    }

    #[test]
    fn log_level_override_wins_over_global() {
        let g = global();
        let pg = PerGameSettings {
            log_level: Some("trace".to_string()),
            ..PerGameSettings::default()
        };
        assert_eq!(pg.effective(&g).debug.log_level, "trace");
    }

    #[test]
    fn reset_row_clears_all_overrides() {
        let g = global();
        let mut pg = PerGameSettings::default();
        pg.adjust(ROW_RESOLUTION, 1, &g);
        pg.adjust(ROW_VALIDATION, 1, &g);
        assert!(!pg.is_empty());
        pg.toggle_override(ROW_RESET, &g);
        assert!(pg.is_empty());
    }

    #[test]
    fn save_then_load_round_trips_and_empty_deletes() {
        let dir = std::env::temp_dir().join(format!("raeen-pergame-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let id = "CUSA-Test:01";
        let g = global();

        let mut pg = PerGameSettings::default();
        pg.adjust(ROW_RESOLUTION, 2, &g); // 1.0 -> 1.5
        pg.save(&dir, id);
        let loaded = PerGameSettings::load(&dir, id);
        assert_eq!(loaded.resolution_scale, Some(1.5));

        // Emptying and saving deletes the file; load falls back to default.
        let empty = PerGameSettings::default();
        empty.save(&dir, id);
        assert!(!PerGameSettings::path_for(&dir, id).exists());
        assert!(PerGameSettings::load(&dir, id).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_of_corrupt_file_is_null_safe() {
        let dir = std::env::temp_dir().join(format!("raeen-pergame-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let id = "BROKEN";
        std::fs::write(PerGameSettings::path_for(&dir, id), b"{ not valid json ]").unwrap();
        // Never panics or errors — degrades to inherit-global.
        assert!(PerGameSettings::load(&dir, id).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sanitize_id_strips_path_hostile_chars() {
        assert_eq!(sanitize_id("CUSA/12345:v2"), "CUSA_12345_v2");
        assert_eq!(sanitize_id("   "), "UNKNOWN");
        assert_eq!(sanitize_id("ok.title-1_2"), "ok.title-1_2");
    }

    #[test]
    fn trophy_summary_line_shows_counts_and_last_unlock_never_names() {
        assert_eq!(
            TrophySummary {
                unlocked: 0,
                last_unlock_ms: None
            }
            .line(),
            "Trophies: none unlocked yet"
        );
        // 2026-07-27 00:00:00 UTC = 1_785_110_400 s.
        assert_eq!(
            TrophySummary {
                unlocked: 3,
                last_unlock_ms: Some(1_785_110_400_000)
            }
            .line(),
            format!(
                "Trophies: 3 unlocked \u{00B7} last {}",
                crate::crash_report::utc_display(1_785_110_400)
            )
        );
        // A count without a timestamp (defensive) still renders honestly.
        assert_eq!(
            TrophySummary {
                unlocked: 2,
                last_unlock_ms: None
            }
            .line(),
            "Trophies: 2 unlocked"
        );
    }

    #[test]
    fn store_dir_is_sibling_of_config() {
        let dir = PerGameSettings::store_dir(Path::new("some/dir/config.toml"));
        assert_eq!(dir, PathBuf::from("some/dir").join("per_game"));
        // A bare relative config file falls back to "." as its base.
        let dir = PerGameSettings::store_dir(Path::new("config.toml"));
        assert_eq!(dir, PathBuf::from(".").join("per_game"));
    }
}
