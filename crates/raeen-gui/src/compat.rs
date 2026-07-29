//! Per-title compatibility badges for the Shell library.
//!
//! Two honest data sources, folded newest-wins:
//!
//! 1. The local baseline report `artifacts/compat/latest.json`, produced by
//!    `cargo xtask baseline run`. The JSON shape is owned by
//!    `xtask/src/schema.rs` (`RunReport` / `CompatResult` / `Stage` /
//!    `Metrics`); xtask is a binary crate, so the few fields the Shell needs
//!    are duplicated here — keep them byte-compatible with that file.
//! 2. The Shell's own session ledger (`shell/ledger.rs`): the most recent
//!    launch outcome this machine actually witnessed.
//!
//! ## Stage → badge mapping
//!
//! The stage order mirrors `xtask/src/baseline.rs::stage_rank` (Refused /
//! Detected < Crashed / Launching < Exited < TimedOut < Rendering) — do not
//! invent a divergent ordering. Whether a run presented frames is judged from
//! `flip_events`, exactly like the baseline diff does, because a rendering
//! run that lives to the timeout is still classified `TimedOut`.
//!
//! | Evidence | Badge |
//! |---|---|
//! | `Refused` / `Detected` / `Launching` / `Crashed` | **Broken** |
//! | `Exited` / `TimedOut` with zero flips (ran, never presented) | **Boots** |
//! | frames + render errors (`shader_errors + gpu_errors > 0`) or fps < 10 | **Menu** |
//! | frames, clean render path, 10 ≤ fps < 30 | **In-game** |
//! | frames, clean render path, fps ≥ 30 | **Playable** |
//! | no data at all | **Untested** (no badge is drawn — absence is the badge) |
//!
//! The Menu / In-game / Playable split cannot be observed directly by the
//! headless harness, so the thresholds are documented heuristics over what it
//! *can* measure: presented frames, render-path errors, and sustained fps.
//!
//! ## Newest-wins fold with the session ledger
//!
//! A ledger entry is evidence only if the title was actually launched
//! (`last_played > 0`). When the last local session is newer than the
//! baseline measurement:
//!
//! * a **faulted** session is strong evidence — the badge drops to
//!   **Broken**, provenance `"last session"`;
//! * a **clean** session is weak evidence — it proves the title boots but
//!   nothing more, so it can only *raise* the badge to **Boots** (when the
//!   baseline said Broken, or there is no baseline). It never downgrades a
//!   higher baseline badge, because a clean exit is consistent with Playable.
//!
//! When the baseline is newer (or the only source), the baseline wins and
//! provenance reads `"baseline YYYY-MM-DD"`.

use crate::library::{ItemKind, LibraryItem};
use crate::shell::ledger::TitleLedger;
use crate::theme::Palette;
use egui::Color32;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Where the baseline report lives, relative to the working directory —
/// the same repo-root convention as `Games/` and `themes/`.
pub const DEFAULT_BASELINE_PATH: &str = "artifacts/compat/latest.json";

// ---------------------------------------------------------------------------
// Mirrored xtask schema (source of truth: xtask/src/schema.rs)
// ---------------------------------------------------------------------------

/// Mirror of `xtask/src/schema.rs::Stage`. Serde names must stay identical.
///
/// This is the project's **one** outcome vocabulary — the compatibility
/// harness, `compat/schema-v1.json`, and the session report written for every
/// launch all speak it. A second copy declared elsewhere would be a fourth
/// spelling of the same seven words, so new consumers extend this enum rather
/// than declaring their own.
///
/// `Launching` is the [`Default`] because that is the honest state of a session
/// that has begun and not yet reported: a report written at t≈0 must not claim
/// an outcome it cannot know.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Detected,
    #[default]
    Launching,
    Rendering,
    Crashed,
    TimedOut,
    Exited,
    Refused,
}

impl Stage {
    /// Every stage, in the schema's declaration order.
    pub const ALL: [Stage; 7] = [
        Stage::Detected,
        Stage::Launching,
        Stage::Rendering,
        Stage::Crashed,
        Stage::TimedOut,
        Stage::Exited,
        Stage::Refused,
    ];

    /// The wire spelling — byte-identical to the serde name, so a report's
    /// `- Outcome:` line and its JSON sidecar can never disagree.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Stage::Detected => "detected",
            Stage::Launching => "launching",
            Stage::Rendering => "rendering",
            Stage::Crashed => "crashed",
            Stage::TimedOut => "timed_out",
            Stage::Exited => "exited",
            Stage::Refused => "refused",
        }
    }

    /// Inverse of [`Stage::slug`].
    #[must_use]
    pub fn from_slug(slug: &str) -> Option<Self> {
        Stage::ALL.into_iter().find(|s| s.slug() == slug)
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.slug())
    }
}

/// The subset of `xtask/src/schema.rs::Metrics` the badge mapping reads.
/// Every field defaults so a trimmed or older report still parses.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Metrics {
    #[serde(default)]
    pub flip_events: u64,
    #[serde(default)]
    pub shader_errors: u64,
    #[serde(default)]
    pub gpu_errors: u64,
    #[serde(default)]
    pub observed_fps: Option<f64>,
}

/// The subset of `xtask/src/schema.rs::CompatResult` the Shell needs.
#[derive(Debug, Clone, Deserialize)]
pub struct CompatResult {
    #[serde(default)]
    pub game_id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub measured_unix_ms: u128,
    pub stage: Stage,
    #[serde(default)]
    pub metrics: Metrics,
}

/// The subset of `xtask/src/schema.rs::RunReport` the Shell needs.
#[derive(Debug, Clone, Deserialize)]
pub struct RunReport {
    #[serde(default)]
    pub results: Vec<CompatResult>,
}

// ---------------------------------------------------------------------------
// Badge levels
// ---------------------------------------------------------------------------

/// Per-title compatibility level, ordered worst → best so `Ord` means
/// "further along". "Untested" is represented by the *absence* of a badge —
/// no chip is drawn, no noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BadgeLevel {
    Broken,
    Boots,
    Menu,
    InGame,
    Playable,
}

impl BadgeLevel {
    pub fn label(self) -> &'static str {
        match self {
            BadgeLevel::Broken => "Broken",
            BadgeLevel::Boots => "Boots",
            BadgeLevel::Menu => "Menu",
            BadgeLevel::InGame => "In-game",
            BadgeLevel::Playable => "Playable",
        }
    }
}

/// Frames-presented runs need render errors and fps below these to escape
/// "Menu"; "Playable" additionally needs [`PLAYABLE_MIN_FPS`].
const IN_GAME_MIN_FPS: f64 = 10.0;
const PLAYABLE_MIN_FPS: f64 = 30.0;

/// Map one baseline result to a badge level (see the module-level table).
pub fn badge_from_result(result: &CompatResult) -> BadgeLevel {
    match result.stage {
        Stage::Refused | Stage::Detected | Stage::Launching | Stage::Crashed => BadgeLevel::Broken,
        Stage::Exited | Stage::TimedOut | Stage::Rendering => {
            if result.metrics.flip_events == 0 {
                return BadgeLevel::Boots;
            }
            let fps = result.metrics.observed_fps.unwrap_or(0.0);
            let render_errors = result.metrics.shader_errors + result.metrics.gpu_errors;
            if render_errors > 0 || fps < IN_GAME_MIN_FPS {
                BadgeLevel::Menu
            } else if fps >= PLAYABLE_MIN_FPS {
                BadgeLevel::Playable
            } else {
                BadgeLevel::InGame
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Baseline index
// ---------------------------------------------------------------------------

/// One title's digested baseline evidence.
#[derive(Debug, Clone)]
pub struct BaselineEntry {
    pub level: BadgeLevel,
    pub measured_unix_ms: u128,
    /// Pre-rendered `YYYY-MM-DD` of the measurement, for provenance text.
    pub date: String,
}

/// The baseline report digested for lookup: keyed by title id
/// (`PPSA…`, from `game_id`) and by normalized title as a fallback for
/// library entries without a `param.json` identity.
#[derive(Debug, Clone, Default)]
pub struct CompatIndex {
    by_game_id: HashMap<String, BaselineEntry>,
    by_title: HashMap<String, BaselineEntry>,
}

fn normalize_title(title: &str) -> String {
    title.trim().to_lowercase()
}

impl CompatIndex {
    /// Load and digest `latest.json`. A missing, unreadable, or malformed
    /// file yields an empty index — never an error, never a panic. Badges
    /// simply don't appear.
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        match serde_json::from_str::<RunReport>(&text) {
            Ok(report) => Self::from_report(&report),
            Err(err) => {
                tracing::warn!(path = %path.display(), error = %err,
                    "compat baseline malformed — no badges");
                Self::default()
            }
        }
    }

    /// Digest a parsed report. Duplicate results for one title keep the
    /// newest measurement.
    pub fn from_report(report: &RunReport) -> Self {
        let mut index = Self::default();
        for result in &report.results {
            let entry = BaselineEntry {
                level: badge_from_result(result),
                measured_unix_ms: result.measured_unix_ms,
                date: unix_ms_to_date(result.measured_unix_ms),
            };
            if !result.game_id.trim().is_empty() {
                insert_newest(
                    &mut index.by_game_id,
                    result.game_id.trim().to_string(),
                    &entry,
                );
            }
            if !result.title.trim().is_empty() {
                insert_newest(&mut index.by_title, normalize_title(&result.title), &entry);
            }
        }
        index
    }

    /// Find a library item's baseline entry: real title id first
    /// (`param.json` `PPSA…` against the report's `game_id`), display title
    /// as fallback.
    pub fn lookup(&self, title_id: Option<&str>, title: &str) -> Option<&BaselineEntry> {
        if let Some(id) = title_id
            && let Some(entry) = self.by_game_id.get(id.trim())
        {
            return Some(entry);
        }
        self.by_title.get(&normalize_title(title))
    }

    /// Whether the index digested any results at all. Test-facing today —
    /// the Shell itself treats an empty index the same as any other (lookups
    /// simply miss).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_empty(&self) -> bool {
        self.by_game_id.is_empty() && self.by_title.is_empty()
    }
}

fn insert_newest(map: &mut HashMap<String, BaselineEntry>, key: String, entry: &BaselineEntry) {
    match map.get(&key) {
        Some(existing) if existing.measured_unix_ms >= entry.measured_unix_ms => {}
        _ => {
            map.insert(key, entry.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Session fold
// ---------------------------------------------------------------------------

/// The session ledger evidence the fold reads (adapter over
/// [`TitleLedger`], kept tiny so the fold is trivially testable).
#[derive(Debug, Clone, Copy)]
pub struct SessionOutcome {
    /// Unix seconds of the most recent launch; `0` = never launched
    /// (no evidence).
    pub last_played_unix: u64,
    /// Whether that session ended after faulting.
    pub faulted: bool,
}

impl SessionOutcome {
    pub fn from_ledger(ledger: &TitleLedger) -> Self {
        Self {
            last_played_unix: ledger.last_played,
            faulted: ledger.last_faulted,
        }
    }
}

/// A resolved badge: level plus one-line provenance for the per-game overlay
/// ("baseline 2026-07-27" vs "last session").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TitleBadge {
    pub level: BadgeLevel,
    pub provenance: String,
}

/// Fold baseline and session evidence, newest-wins (see module docs for the
/// clean-session asymmetry). `None` = Untested = no badge.
pub fn resolve_badge(
    baseline: Option<&BaselineEntry>,
    session: Option<SessionOutcome>,
) -> Option<TitleBadge> {
    let session = session.filter(|s| s.last_played_unix > 0);
    let baseline_badge = |entry: &BaselineEntry| TitleBadge {
        level: entry.level,
        provenance: format!("baseline {}", entry.date),
    };
    let session_badge = |faulted: bool| TitleBadge {
        level: if faulted {
            BadgeLevel::Broken
        } else {
            BadgeLevel::Boots
        },
        provenance: "last session".to_string(),
    };
    match (baseline, session) {
        (None, None) => None,
        (Some(entry), None) => Some(baseline_badge(entry)),
        (None, Some(session)) => Some(session_badge(session.faulted)),
        (Some(entry), Some(session)) => {
            let session_newer = (session.last_played_unix as u128) * 1000 > entry.measured_unix_ms;
            if !session_newer {
                return Some(baseline_badge(entry));
            }
            if session.faulted {
                return Some(session_badge(true));
            }
            // Clean newer session: weak evidence — only ever an upgrade.
            if entry.level < BadgeLevel::Boots {
                Some(session_badge(false))
            } else {
                Some(baseline_badge(entry))
            }
        }
    }
}

/// Resolve a badge for every *game* in the library (apps never wear one),
/// keyed by [`LibraryItem::id`] — the same key the Shell's ledger map uses.
pub fn badges_for_library(
    index: &CompatIndex,
    items: &[LibraryItem],
    ledgers: &HashMap<String, TitleLedger>,
) -> HashMap<String, TitleBadge> {
    let mut badges = HashMap::new();
    for item in items {
        if item.kind != ItemKind::Game {
            continue;
        }
        let baseline = index.lookup(item.title_id.as_deref(), &item.title);
        let session = ledgers
            .get(item.id.as_str())
            .map(SessionOutcome::from_ledger);
        if let Some(badge) = resolve_badge(baseline, session) {
            badges.insert(item.id.clone(), badge);
        }
    }
    badges
}

// ---------------------------------------------------------------------------
// Theme-derived badge colors
// ---------------------------------------------------------------------------

/// The accent color a badge chip's status dot wears, derived from the active
/// theme rather than hardcoded hex: each level keeps a fixed *semantic* hue
/// (green = further along, red = broken) but takes its saturation/lightness
/// family from the theme's `accent` token, so chips match a theme's vibrance
/// the way every other accent does.
pub fn badge_color(palette: &Palette, level: BadgeLevel) -> Color32 {
    // The theme's own hue is intentionally replaced per level; only its
    // saturation/lightness character carries over, clamped into a band where
    // every hue stays legible as a small dot over the dark chip fill.
    let (_, s, l) = rgb_to_hsl(palette.accent);
    let s = s.clamp(0.45, 0.85);
    let l = l.clamp(0.5, 0.62);
    let (hue, s_scale) = match level {
        BadgeLevel::Playable => (130.0, 1.0),
        BadgeLevel::InGame => (195.0, 1.0),
        BadgeLevel::Menu => (45.0, 1.0),
        BadgeLevel::Boots => (220.0, 0.35),
        BadgeLevel::Broken => (5.0, 1.0),
    };
    hsl_to_rgb(hue, s * s_scale, l)
}

fn rgb_to_hsl(c: Color32) -> (f32, f32, f32) {
    let r = c.r() as f32 / 255.0;
    let g = c.g() as f32 / 255.0;
    let b = c.b() as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    if (max - min).abs() < f32::EPSILON {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if (max - r).abs() < f32::EPSILON {
        ((g - b) / d).rem_euclid(6.0)
    } else if (max - g).abs() < f32::EPSILON {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    } * 60.0;
    (h, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Color32 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h.rem_euclid(360.0) / 60.0;
    let x = c * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    Color32::from_rgb(
        ((r1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((g1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((b1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

// ---------------------------------------------------------------------------
// Unix ms → civil date (UTC), for provenance text
// ---------------------------------------------------------------------------

/// `YYYY-MM-DD` (UTC) for a unix-epoch millisecond stamp. Dependency-free
/// civil-from-days conversion (Howard Hinnant's algorithm).
pub fn unix_ms_to_date(ms: u128) -> String {
    let days = (ms / 86_400_000) as i64;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((if m <= 2 { y + 1 } else { y }), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::default_theme;

    /// The outcome vocabulary is shared with `xtask/src/schema.rs` and
    /// `compat/schema-v1.json`. `slug` and the serde name are two spellings of
    /// one thing; a renamed variant that updated only one of them would split
    /// the vocabulary silently — every report would still parse, and the
    /// harness and the Shell would quietly disagree about what `timed_out`
    /// means. Pinning the literals here makes that a test failure.
    #[test]
    fn stage_slugs_match_their_serde_names_and_the_schema_literals() {
        for stage in Stage::ALL {
            let json = serde_json::to_string(&stage).expect("stage serializes");
            assert_eq!(
                json,
                format!("\"{}\"", stage.slug()),
                "slug and serde name must be the same word"
            );
            assert_eq!(Stage::from_slug(stage.slug()), Some(stage));
            let parsed: Stage = serde_json::from_str(&json).expect("round trip");
            assert_eq!(parsed, stage);
        }
        assert_eq!(
            Stage::ALL.map(Stage::slug),
            [
                "detected",
                "launching",
                "rendering",
                "crashed",
                "timed_out",
                "exited",
                "refused"
            ]
        );
        assert_eq!(Stage::from_slug("not_a_stage"), None);
        // A session that has begun and not yet reported is `launching` — the
        // report written at t=0 must not claim an outcome it cannot know.
        assert_eq!(Stage::default(), Stage::Launching);
    }

    fn result(stage: Stage, flips: u64, shader: u64, gpu: u64, fps: Option<f64>) -> CompatResult {
        CompatResult {
            game_id: "PPSA00000".to_string(),
            title: "Fixture".to_string(),
            measured_unix_ms: 1_785_136_298_906,
            stage,
            metrics: Metrics {
                flip_events: flips,
                shader_errors: shader,
                gpu_errors: gpu,
                observed_fps: fps,
            },
        }
    }

    #[test]
    fn stage_mapping_matches_the_documented_table() {
        // Never got going → Broken.
        for stage in [
            Stage::Refused,
            Stage::Detected,
            Stage::Launching,
            Stage::Crashed,
        ] {
            assert_eq!(
                badge_from_result(&result(stage, 500, 0, 0, Some(60.0))),
                BadgeLevel::Broken,
                "{stage:?} is Broken regardless of metrics"
            );
        }
        // Ran but never presented → Boots.
        assert_eq!(
            badge_from_result(&result(Stage::Exited, 0, 0, 0, None)),
            BadgeLevel::Boots
        );
        assert_eq!(
            badge_from_result(&result(Stage::TimedOut, 0, 0, 0, None)),
            BadgeLevel::Boots
        );
        // Frames with render errors (the ASTRO.BOT shape) → Menu.
        assert_eq!(
            badge_from_result(&result(Stage::TimedOut, 96, 83, 74, Some(0.7))),
            BadgeLevel::Menu
        );
        // Frames, clean, slideshow fps → Menu.
        assert_eq!(
            badge_from_result(&result(Stage::Rendering, 40, 0, 0, Some(4.0))),
            BadgeLevel::Menu
        );
        // Frames, clean, mid fps → In-game.
        assert_eq!(
            badge_from_result(&result(Stage::Rendering, 900, 0, 0, Some(20.0))),
            BadgeLevel::InGame
        );
        // Frames, clean, fast (the Minecraft shape) → Playable.
        assert_eq!(
            badge_from_result(&result(Stage::TimedOut, 13_536, 0, 0, Some(72.2))),
            BadgeLevel::Playable
        );
        // Frames but no fps sample → Menu, never a crash.
        assert_eq!(
            badge_from_result(&result(Stage::Rendering, 3, 0, 0, None)),
            BadgeLevel::Menu
        );
    }

    fn entry(level: BadgeLevel, measured_unix_ms: u128) -> BaselineEntry {
        BaselineEntry {
            level,
            measured_unix_ms,
            date: unix_ms_to_date(measured_unix_ms),
        }
    }

    #[test]
    fn fold_newest_wins_and_clean_sessions_only_upgrade() {
        let baseline = entry(BadgeLevel::Playable, 1_785_136_298_906); // 2026-07-27
        let newer: u64 = 1_785_200_000; // secs — newer than the baseline stamp
        let older: u64 = 1_700_000_000;

        // Newer faulted session overrides even a Playable baseline.
        let badge = resolve_badge(
            Some(&baseline),
            Some(SessionOutcome {
                last_played_unix: newer,
                faulted: true,
            }),
        )
        .unwrap();
        assert_eq!(badge.level, BadgeLevel::Broken);
        assert_eq!(badge.provenance, "last session");

        // Newer clean session never downgrades a higher baseline.
        let badge = resolve_badge(
            Some(&baseline),
            Some(SessionOutcome {
                last_played_unix: newer,
                faulted: false,
            }),
        )
        .unwrap();
        assert_eq!(badge.level, BadgeLevel::Playable);
        assert_eq!(badge.provenance, "baseline 2026-07-27");

        // Newer clean session upgrades a Broken baseline to Boots.
        let broken = entry(BadgeLevel::Broken, 1_785_136_298_906);
        let badge = resolve_badge(
            Some(&broken),
            Some(SessionOutcome {
                last_played_unix: newer,
                faulted: false,
            }),
        )
        .unwrap();
        assert_eq!(badge.level, BadgeLevel::Boots);
        assert_eq!(badge.provenance, "last session");

        // Older session (even a fault) loses to the baseline — newest wins.
        let badge = resolve_badge(
            Some(&baseline),
            Some(SessionOutcome {
                last_played_unix: older,
                faulted: true,
            }),
        )
        .unwrap();
        assert_eq!(badge.level, BadgeLevel::Playable);
        assert_eq!(badge.provenance, "baseline 2026-07-27");

        // Session only (no baseline): fault → Broken, clean → Boots.
        let badge = resolve_badge(
            None,
            Some(SessionOutcome {
                last_played_unix: older,
                faulted: false,
            }),
        )
        .unwrap();
        assert_eq!(badge.level, BadgeLevel::Boots);
        assert_eq!(badge.provenance, "last session");

        // Never launched, never measured → Untested → no badge.
        assert!(resolve_badge(None, None).is_none());
        assert!(
            resolve_badge(
                None,
                Some(SessionOutcome {
                    last_played_unix: 0,
                    faulted: true
                })
            )
            .is_none(),
            "last_played == 0 is 'never launched', not evidence"
        );
    }

    #[test]
    fn missing_or_malformed_baseline_yields_no_badges_and_never_panics() {
        let missing = CompatIndex::load(Path::new("does/not/exist/latest.json"));
        assert!(missing.is_empty());

        let dir = std::env::temp_dir().join(format!("raeen-compat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("latest.json");
        std::fs::write(&bad, b"{ not json at all ]").unwrap();
        assert!(CompatIndex::load(&bad).is_empty());
        // Valid JSON, wrong shape (stage is an unknown string) → also empty.
        std::fs::write(
            &bad,
            br#"{"results":[{"game_id":"X","title":"Y","stage":"summoned"}]}"#,
        )
        .unwrap();
        assert!(CompatIndex::load(&bad).is_empty());
        let _ = std::fs::remove_dir_all(&dir);

        // An empty index over a library with no ledgers → no badges at all.
        let items = crate::library::sample_library();
        let badges = badges_for_library(&CompatIndex::default(), &items, &HashMap::new());
        assert!(badges.is_empty());
    }

    #[test]
    fn index_matches_by_title_id_then_title_and_keeps_newest_duplicate() {
        let report = RunReport {
            results: vec![
                result(Stage::Crashed, 0, 0, 0, None), // PPSA00000 "Fixture", older stage below
                CompatResult {
                    measured_unix_ms: 1_785_200_000_000, // newer duplicate wins
                    ..result(Stage::Rendering, 500, 0, 0, Some(60.0))
                },
                CompatResult {
                    game_id: "PPSA11111".to_string(),
                    title: "Other Game".to_string(),
                    ..result(Stage::Exited, 0, 0, 0, None)
                },
            ],
        };
        let index = CompatIndex::from_report(&report);
        // Newest duplicate won: Playable, not Broken.
        let by_id = index.lookup(Some("PPSA00000"), "unrelated").unwrap();
        assert_eq!(by_id.level, BadgeLevel::Playable);
        // Title fallback (case/whitespace-insensitive) for items without ids.
        let by_title = index.lookup(None, "  other game ").unwrap();
        assert_eq!(by_title.level, BadgeLevel::Boots);
        // Unknown title id falls through to the title key.
        let fallthrough = index.lookup(Some("PPSA99999"), "Fixture").unwrap();
        assert_eq!(fallthrough.level, BadgeLevel::Playable);
        assert!(index.lookup(Some("PPSA99999"), "nope").is_none());
    }

    #[test]
    fn badges_for_library_keys_by_item_id_and_skips_apps() {
        let mut items = crate::library::sample_library();
        // Give one sample game a real identity matching the report.
        items[0].title_id = Some("PPSA00000".to_string());
        let report = RunReport {
            results: vec![result(Stage::TimedOut, 13_536, 0, 0, Some(72.2))],
        };
        let index = CompatIndex::from_report(&report);
        let mut ledgers = HashMap::new();
        // A different sample game faulted locally (never measured by xtask).
        ledgers.insert(
            items[1].id.clone(),
            TitleLedger {
                last_played: 1_700_000_000,
                total_play_secs: 60,
                last_faulted: true,
            },
        );
        let badges = badges_for_library(&index, &items, &ledgers);
        assert_eq!(
            badges.get(&items[0].id).unwrap().level,
            BadgeLevel::Playable
        );
        let faulted = badges.get(&items[1].id).unwrap();
        assert_eq!(faulted.level, BadgeLevel::Broken);
        assert_eq!(faulted.provenance, "last session");
        // Untested games and the Settings app tile get nothing.
        assert_eq!(badges.len(), 2);
        assert!(!badges.contains_key("settings"));
    }

    #[test]
    fn unix_ms_to_date_converts_known_stamps() {
        assert_eq!(unix_ms_to_date(0), "1970-01-01");
        // The real latest.json generation stamp from 2026-07-27.
        assert_eq!(unix_ms_to_date(1_785_136_298_906), "2026-07-27");
        // Leap-day sanity: 2024-02-29T12:00Z.
        assert_eq!(unix_ms_to_date(1_709_208_000_000), "2024-02-29");
    }

    #[test]
    fn badge_colors_are_distinct_and_theme_derived() {
        let palette = default_theme().palette;
        let levels = [
            BadgeLevel::Broken,
            BadgeLevel::Boots,
            BadgeLevel::Menu,
            BadgeLevel::InGame,
            BadgeLevel::Playable,
        ];
        let colors: Vec<Color32> = levels.iter().map(|&l| badge_color(&palette, l)).collect();
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(colors[i], colors[j], "{:?} vs {:?}", levels[i], levels[j]);
            }
        }
        // Semantic anchors: Playable leans green, Broken leans red.
        let playable = badge_color(&palette, BadgeLevel::Playable);
        assert!(playable.g() > playable.r() && playable.g() > playable.b());
        let broken = badge_color(&palette, BadgeLevel::Broken);
        assert!(broken.r() > broken.g() && broken.r() > broken.b());
    }
}
