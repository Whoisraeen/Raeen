//! `cargo xtask compat diff` — sweep-to-sweep regression classification.
//!
//! Compares two `RunReport` sweep JSONs per title (matched by content SHA-1,
//! the same identity `baseline diff` uses — a title ID can cover two installed
//! revisions) and assigns one verdict per title: IMPROVED / UNCHANGED /
//! REGRESSED / NEW / REMOVED. The rules are deliberately coarse and
//! noise-tolerant:
//!
//! * a stage-rank drop (`Rendering > TimedOut > Exited > Crashed/Launching >
//!   Refused/Detected`) is a regression; a rise is an improvement;
//! * a flip-count drop beyond `--flip-tolerance` percent (default 20) is a
//!   regression; a rise beyond the same tolerance — or first frames ever — is
//!   an improvement; anything inside the band is measurement noise;
//! * a wall-time class drop (`<5s`, `5-30s`, `30-120s`, `120s+`) is a
//!   regression: a title that used to survive the whole window and now dies in
//!   seconds regressed even when its stage label did not change. Exact wall
//!   times are never compared — scheduler noise is not a regression;
//! * a first blocker appearing where none was observed is a regression, a
//!   blocker clearing is an improvement, and a *changed* blocker is a note
//!   (moving between two different faults is triage information, not
//!   automatically better or worse). Blockers are compared after normalizing
//!   timestamps, `ThreadId(..)` tokens, and digit runs, because two runs of
//!   the same fault never share a literal log line.
//!
//! `compat diff` exits nonzero when any title regressed so scripts and CI can
//! gate on it. `compat run` prints the same table automatically against
//! `--baseline` or the previous sweep tracked by the latest-run pointer.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Result, bail};

use crate::baseline::stage_rank;
use crate::schema::{CompatResult, RunReport, SCHEMA_VERSION};
use crate::{option, read_json};

pub(crate) const DEFAULT_FLIP_TOLERANCE_PCT: f64 = 20.0;
/// Blocker excerpts in the details section stay one-line readable.
const MAX_BLOCKER_EXCERPT: usize = 160;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DiffOptions {
    pub flip_tolerance_pct: f64,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            flip_tolerance_pct: DEFAULT_FLIP_TOLERANCE_PCT,
        }
    }
}

/// Parse `--flip-tolerance` (shared by `compat diff` and `compat run`).
pub(crate) fn parse_options(args: &[String]) -> Result<DiffOptions> {
    let flip_tolerance_pct = match option(args, "--flip-tolerance") {
        Some(value) => match value.parse::<f64>() {
            Ok(pct) if pct.is_finite() && pct >= 0.0 => pct,
            _ => bail!("--flip-tolerance must be a non-negative percentage, got {value:?}"),
        },
        None => DEFAULT_FLIP_TOLERANCE_PCT,
    };
    Ok(DiffOptions { flip_tolerance_pct })
}

pub fn diff(args: &[String]) -> Result<()> {
    let options = parse_options(args)?;
    let positional = positionals(args, &["--flip-tolerance"]);
    let [old_path, new_path] = positional.as_slice() else {
        bail!("usage: cargo xtask compat diff <old.json> <new.json> [--flip-tolerance PCT]");
    };
    let old: RunReport = read_json(Path::new(old_path))?;
    let new: RunReport = read_json(Path::new(new_path))?;
    for (path, report) in [(old_path, &old), (new_path, &new)] {
        if report.schema_version != SCHEMA_VERSION {
            bail!(
                "{path} has schema {} (expected {SCHEMA_VERSION})",
                report.schema_version
            );
        }
    }
    let report = compute_diff(&old, &new, options);
    print!("{}", render_diff(&report, old_path, new_path, options));
    let regressed = report.count(Verdict::Regressed);
    if regressed > 0 {
        bail!("{regressed} title(s) REGRESSED");
    }
    Ok(())
}

/// Positional arguments, skipping flags and the values of value-taking flags.
fn positionals(args: &[String], value_flags: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if value_flags.contains(&arg.as_str()) {
            iter.next();
        } else if !arg.starts_with("--") {
            out.push(arg.clone());
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    Improved,
    Unchanged,
    Regressed,
    New,
    Removed,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Verdict::Improved => "IMPROVED",
            Verdict::Unchanged => "UNCHANGED",
            Verdict::Regressed => "REGRESSED",
            Verdict::New => "NEW",
            Verdict::Removed => "REMOVED",
        }
    }
}

/// Coarse wall-time buckets. Only a bucket change is signal; exact wall-time
/// deltas inside a bucket are scheduler noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum WallClass {
    Instant,
    Brief,
    Short,
    Sustained,
}

fn wall_class(wall_ms: u128) -> WallClass {
    match wall_ms {
        0..=4_999 => WallClass::Instant,
        5_000..=29_999 => WallClass::Brief,
        30_000..=119_999 => WallClass::Short,
        _ => WallClass::Sustained,
    }
}

impl WallClass {
    fn label(self) -> &'static str {
        match self {
            WallClass::Instant => "<5s",
            WallClass::Brief => "5-30s",
            WallClass::Short => "30-120s",
            WallClass::Sustained => "120s+",
        }
    }
}

/// Reduce a sanitized blocker line to its fault identity so two runs of the
/// same fault compare equal: drop the leading RFC3339 timestamp and
/// `ThreadId(..)` tokens, and collapse every decimal digit run to `#` (HLE
/// ring depths, fault counters, and line numbers vary run to run; addresses
/// are already `<ADDR>` from the runner's sanitizer). Normalization is used
/// only for equality — the report always displays the raw sanitized line.
fn normalize_blocker(line: &str) -> String {
    line.split_whitespace()
        .filter(|token| !is_timestamp(token) && !token.starts_with("ThreadId("))
        .map(collapse_digits)
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_timestamp(token: &str) -> bool {
    token.starts_with(|c: char| c.is_ascii_digit()) && token.contains('T') && token.ends_with('Z')
}

fn collapse_digits(token: &str) -> String {
    let mut out = String::with_capacity(token.len());
    let mut in_run = false;
    for c in token.chars() {
        if c.is_ascii_digit() {
            if !in_run {
                out.push('#');
                in_run = true;
            }
        } else {
            out.push(c);
            in_run = false;
        }
    }
    out
}

fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.into()
    } else {
        let cut: String = value.chars().take(max).collect();
        format!("{cut}…")
    }
}

pub(crate) struct TitleVerdict {
    title: String,
    game_id: String,
    verdict: Verdict,
    stage_cell: String,
    flips_cell: String,
    wall_cell: String,
    blocker_cell: String,
    /// Why this title is REGRESSED (empty otherwise).
    regressions: Vec<String>,
    /// Why this title is IMPROVED (may be non-empty on a REGRESSED title —
    /// a regression always outranks concurrent progress).
    improvements: Vec<String>,
    /// Neutral triage notes (e.g. the first blocker changed identity).
    notes: Vec<String>,
}

pub(crate) struct CompatDiff {
    old_build: String,
    new_build: String,
    old_machine: String,
    new_machine: String,
    rows: Vec<TitleVerdict>,
}

impl CompatDiff {
    pub(crate) fn count(&self, verdict: Verdict) -> usize {
        self.rows
            .iter()
            .filter(|row| row.verdict == verdict)
            .count()
    }
}

fn results_by_hash(report: &RunReport) -> BTreeMap<&str, &CompatResult> {
    report
        .results
        .iter()
        .map(|result| (result.content_sha1.as_str(), result))
        .collect()
}

pub(crate) fn compute_diff(old: &RunReport, new: &RunReport, options: DiffOptions) -> CompatDiff {
    let old_map = results_by_hash(old);
    let new_map = results_by_hash(new);
    let build_of = |report: &RunReport| {
        report
            .results
            .first()
            .map(|result| result.build_revision.clone())
            .unwrap_or_else(|| "unknown".into())
    };

    let mut rows = Vec::new();
    for (hash, previous) in &old_map {
        match new_map.get(hash) {
            Some(current) => rows.push(classify_title(previous, current, options)),
            None => rows.push(single_side_row(previous, Verdict::Removed)),
        }
    }
    for (hash, current) in &new_map {
        if !old_map.contains_key(hash) {
            rows.push(single_side_row(current, Verdict::New));
        }
    }
    rows.sort_by(|a, b| (&a.title, &a.game_id).cmp(&(&b.title, &b.game_id)));
    CompatDiff {
        old_build: build_of(old),
        new_build: build_of(new),
        old_machine: old.machine_id.clone(),
        new_machine: new.machine_id.clone(),
        rows,
    }
}

fn classify_title(old: &CompatResult, new: &CompatResult, options: DiffOptions) -> TitleVerdict {
    let mut regressions = Vec::new();
    let mut improvements = Vec::new();
    let mut notes = Vec::new();

    let (old_rank, new_rank) = (stage_rank(old.stage), stage_rank(new.stage));
    let stage_cell = if old.stage == new.stage {
        format!("{:?}", new.stage)
    } else {
        format!("{:?} -> {:?}", old.stage, new.stage)
    };
    if new_rank < old_rank {
        regressions.push(format!("stage {:?} -> {:?}", old.stage, new.stage));
    } else if new_rank > old_rank {
        improvements.push(format!("stage {:?} -> {:?}", old.stage, new.stage));
    }

    let (old_flips, new_flips) = (old.metrics.flip_events, new.metrics.flip_events);
    let tolerance = options.flip_tolerance_pct / 100.0;
    let flips_cell = if old_flips == new_flips {
        format!("{new_flips}")
    } else {
        let pct = (new_flips as f64 - old_flips as f64) / (old_flips as f64).max(1.0) * 100.0;
        format!("{old_flips} -> {new_flips} ({pct:+.1}%)")
    };
    if old_flips > 0 && (new_flips as f64) < old_flips as f64 * (1.0 - tolerance) {
        regressions.push(format!(
            "flips {old_flips} -> {new_flips} (beyond the -{:.0}% tolerance)",
            options.flip_tolerance_pct
        ));
    } else if old_flips == 0 && new_flips > 0 {
        improvements.push(format!(
            "started presenting frames (0 -> {new_flips} flips)"
        ));
    } else if old_flips > 0 && (new_flips as f64) > old_flips as f64 * (1.0 + tolerance) {
        improvements.push(format!(
            "flips {old_flips} -> {new_flips} (beyond the +{:.0}% tolerance)",
            options.flip_tolerance_pct
        ));
    }

    let (old_wall, new_wall) = (
        wall_class(old.metrics.wall_ms),
        wall_class(new.metrics.wall_ms),
    );
    let wall_cell = if old_wall == new_wall {
        new_wall.label().into()
    } else {
        format!("{} -> {}", old_wall.label(), new_wall.label())
    };
    if new_wall < old_wall {
        regressions.push(format!(
            "wall-time class {} -> {}",
            old_wall.label(),
            new_wall.label()
        ));
    } else if new_wall > old_wall {
        improvements.push(format!(
            "wall-time class {} -> {}",
            old_wall.label(),
            new_wall.label()
        ));
    }

    let old_blocker = old.evidence.first_blocker.as_deref();
    let new_blocker = new.evidence.first_blocker.as_deref();
    let blocker_cell = match (old_blocker, new_blocker) {
        (None, None) => "none".into(),
        (None, Some(blocker)) => {
            regressions.push(format!(
                "first blocker appeared: {}",
                truncate(blocker, MAX_BLOCKER_EXCERPT)
            ));
            "appeared".into()
        }
        (Some(_), None) => {
            improvements.push("first blocker cleared".into());
            "cleared".into()
        }
        (Some(before), Some(after)) => {
            if normalize_blocker(before) == normalize_blocker(after) {
                "unchanged".into()
            } else {
                notes.push(format!(
                    "first blocker changed: {}",
                    truncate(after, MAX_BLOCKER_EXCERPT)
                ));
                "changed".into()
            }
        }
    };

    let verdict = if !regressions.is_empty() {
        Verdict::Regressed
    } else if !improvements.is_empty() {
        Verdict::Improved
    } else {
        Verdict::Unchanged
    };
    TitleVerdict {
        title: new.title.clone(),
        game_id: new.game_id.clone(),
        verdict,
        stage_cell,
        flips_cell,
        wall_cell,
        blocker_cell,
        regressions,
        improvements,
        notes,
    }
}

fn single_side_row(result: &CompatResult, verdict: Verdict) -> TitleVerdict {
    TitleVerdict {
        title: result.title.clone(),
        game_id: result.game_id.clone(),
        verdict,
        stage_cell: format!("{:?}", result.stage),
        flips_cell: result.metrics.flip_events.to_string(),
        wall_cell: wall_class(result.metrics.wall_ms).label().into(),
        blocker_cell: if result.evidence.first_blocker.is_some() {
            "present".into()
        } else {
            "none".into()
        },
        regressions: Vec::new(),
        improvements: Vec::new(),
        notes: Vec::new(),
    }
}

pub(crate) fn render_diff(
    report: &CompatDiff,
    old_label: &str,
    new_label: &str,
    options: DiffOptions,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "compat diff: {old_label} (build {}) -> {new_label} (build {})\n",
        report.old_build, report.new_build
    ));
    out.push_str(&format!(
        "flip tolerance: ±{:.0}% (drops inside the band are noise, not regressions)\n",
        options.flip_tolerance_pct
    ));
    if report.old_machine != report.new_machine {
        out.push_str(&format!(
            "WARNING: reports come from different machines ({} vs {}); metric deltas are not comparable\n",
            report.old_machine, report.new_machine
        ));
    }
    out.push('\n');

    let headers = [
        "Title",
        "Verdict",
        "Stage",
        "Flips",
        "Wall",
        "First blocker",
    ];
    let cells: Vec<[String; 6]> = report
        .rows
        .iter()
        .map(|row| {
            [
                format!("{} ({})", row.title, row.game_id),
                row.verdict.label().into(),
                row.stage_cell.clone(),
                row.flips_cell.clone(),
                row.wall_cell.clone(),
                row.blocker_cell.clone(),
            ]
        })
        .collect();
    let mut widths = headers.map(str::len);
    for row in &cells {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.chars().count());
        }
    }
    let render_row = |columns: &[String; 6]| -> String {
        let padded: Vec<String> = columns
            .iter()
            .zip(widths)
            .map(|(cell, width)| format!("{cell:<width$}"))
            .collect();
        format!("| {} |\n", padded.join(" | "))
    };
    out.push_str(&render_row(&headers.map(String::from)));
    let rule: Vec<String> = widths.iter().map(|width| "-".repeat(*width)).collect();
    out.push_str(&format!("|-{}-|\n", rule.join("-|-")));
    for row in &cells {
        out.push_str(&render_row(row));
    }

    let mut details = String::new();
    for row in &report.rows {
        for reason in &row.regressions {
            details.push_str(&format!("  REGRESSED {}: {reason}\n", row.title));
        }
        for reason in &row.improvements {
            details.push_str(&format!("  IMPROVED  {}: {reason}\n", row.title));
        }
        for note in &row.notes {
            details.push_str(&format!("  note      {}: {note}\n", row.title));
        }
    }
    if !details.is_empty() {
        out.push_str("\ndetails:\n");
        out.push_str(&details);
    }

    let matched = report
        .rows
        .iter()
        .filter(|row| !matches!(row.verdict, Verdict::New | Verdict::Removed))
        .count();
    out.push_str(&format!(
        "\nsummary: {matched} matched title(s) — {} improved, {} unchanged, {} regressed; {} new, {} removed\n",
        report.count(Verdict::Improved),
        report.count(Verdict::Unchanged),
        report.count(Verdict::Regressed),
        report.count(Verdict::New),
        report.count(Verdict::Removed),
    ));
    out
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Stage;

    /// Two entries lifted verbatim from a real measured sweep
    /// (`artifacts/compat/phase0-final.json`, run-1785019657785), so the
    /// fixtures carry the exact production schema, not an invented one.
    const REAL_SWEEP_JSON: &str = r#"{
      "schema_version": 1,
      "generated_unix_ms": 1785020659920,
      "machine_id": "machine-ddfbe4e61ec9",
      "results": [
        {
          "schema_version": 1,
          "measured_unix_ms": 1785019838296,
          "run_id": "run-1785019657785",
          "build_revision": "01f7b613911a",
          "profile": "max-fps",
          "game_id": "PPSA17221",
          "title": "Minecraft",
          "content_sha1": "05b59012cd4ebea8be5ab195b62bd842d158deeb",
          "stage": "timed_out",
          "metrics": {
            "wall_ms": 180383,
            "cpu_ms": 429906,
            "peak_working_set_bytes": 1821728768,
            "exit_code": 1,
            "flip_events": 8192,
            "shader_errors": 0,
            "gpu_errors": 0,
            "audio_errors": 0,
            "input_events": 0,
            "observed_fps": null
          },
          "evidence": {
            "log_sha1": "455338d6d5cc6d33f1c77036f59c46d2ff5eef2c",
            "blocker_signature": null,
            "first_blocker": null,
            "measured": true
          }
        },
        {
          "schema_version": 1,
          "measured_unix_ms": 1785020299590,
          "run_id": "run-1785019657785",
          "build_revision": "01f7b613911a",
          "profile": "max-fps",
          "game_id": "PPSA01576",
          "title": "Avatar Frontiers of Pandora",
          "content_sha1": "d075058339cba635b3f8f1ed5b525347ea895c11",
          "stage": "crashed",
          "metrics": {
            "wall_ms": 57785,
            "cpu_ms": 22765,
            "peak_working_set_bytes": 1446645760,
            "exit_code": -1073740791,
            "flip_events": 102,
            "shader_errors": 708,
            "gpu_errors": 708,
            "audio_errors": 0,
            "input_events": 0,
            "observed_fps": null
          },
          "evidence": {
            "log_sha1": "022c0a66a6e62b546f423b29cd07806fcf30adba",
            "blocker_signature": "8b4f15ce703003df50b8f7d56f1b1e4addce97f6",
            "first_blocker": "2026-07-25T22:57:28.720305Z ERROR ThreadId(64) kyty_graphics::shader::parse: unknown smem instruction s_load_dwordx16, opcode = 0x4 at addr <ADDR> (hash0 = <ADDR>, crc32 = <ADDR>)",
            "measured": true
          }
        }
      ]
    }"#;

    fn real_sweep() -> RunReport {
        serde_json::from_str(REAL_SWEEP_JSON).expect("real sweep fixture parses")
    }

    fn find<'a>(report: &'a mut RunReport, title: &str) -> &'a mut CompatResult {
        report
            .results
            .iter_mut()
            .find(|result| result.title == title)
            .expect("title present")
    }

    fn verdict_of<'a>(diff: &'a CompatDiff, title: &str) -> &'a TitleVerdict {
        diff.rows
            .iter()
            .find(|row| row.title == title)
            .expect("row present")
    }

    #[test]
    fn a_sweep_diffed_against_itself_is_all_unchanged() {
        let report = real_sweep();
        let diff = compute_diff(&report, &report, DiffOptions::default());
        assert_eq!(diff.rows.len(), 2);
        assert!(
            diff.rows
                .iter()
                .all(|row| row.verdict == Verdict::Unchanged),
            "self-diff must not invent changes"
        );
        assert_eq!(diff.count(Verdict::Regressed), 0);
        // The Avatar blocker is byte-identical, so it must read "unchanged".
        assert_eq!(
            verdict_of(&diff, "Avatar Frontiers of Pandora").blocker_cell,
            "unchanged"
        );
    }

    #[test]
    fn flip_drops_inside_the_tolerance_band_are_noise() {
        let old = real_sweep();
        let mut new = real_sweep();
        // 8192 * 0.8 = 6553.6 — 6554 is inside the default -20% band.
        find(&mut new, "Minecraft").metrics.flip_events = 6554;
        let diff = compute_diff(&old, &new, DiffOptions::default());
        assert_eq!(verdict_of(&diff, "Minecraft").verdict, Verdict::Unchanged);
    }

    #[test]
    fn flip_drops_beyond_the_tolerance_regress() {
        let old = real_sweep();
        let mut new = real_sweep();
        find(&mut new, "Minecraft").metrics.flip_events = 6553;
        let diff = compute_diff(&old, &new, DiffOptions::default());
        let row = verdict_of(&diff, "Minecraft");
        assert_eq!(row.verdict, Verdict::Regressed);
        assert!(row.regressions[0].contains("flips 8192 -> 6553"));
        assert_eq!(diff.count(Verdict::Regressed), 1);
    }

    #[test]
    fn the_flip_tolerance_is_configurable() {
        let old = real_sweep();
        let mut new = real_sweep();
        // A 50% drop regresses at the default 20% but not at a 60% tolerance.
        find(&mut new, "Minecraft").metrics.flip_events = 4096;
        let strict = compute_diff(&old, &new, DiffOptions::default());
        assert_eq!(verdict_of(&strict, "Minecraft").verdict, Verdict::Regressed);
        let loose = compute_diff(
            &old,
            &new,
            DiffOptions {
                flip_tolerance_pct: 60.0,
            },
        );
        assert_eq!(verdict_of(&loose, "Minecraft").verdict, Verdict::Unchanged);
    }

    #[test]
    fn first_frames_ever_and_large_flip_gains_improve() {
        let mut old = real_sweep();
        let new = real_sweep();
        find(&mut old, "Avatar Frontiers of Pandora")
            .metrics
            .flip_events = 0;
        // 0 -> 102 flips: started presenting.
        let diff = compute_diff(&old, &new, DiffOptions::default());
        let row = verdict_of(&diff, "Avatar Frontiers of Pandora");
        assert_eq!(row.verdict, Verdict::Improved);
        assert!(row.improvements[0].contains("started presenting frames"));

        let mut gained = real_sweep();
        find(&mut gained, "Minecraft").metrics.flip_events = 16384;
        let diff = compute_diff(&real_sweep(), &gained, DiffOptions::default());
        assert_eq!(verdict_of(&diff, "Minecraft").verdict, Verdict::Improved);
    }

    #[test]
    fn a_stage_rank_drop_regresses_and_a_rise_improves() {
        let old = real_sweep();
        let mut new = real_sweep();
        {
            let minecraft = find(&mut new, "Minecraft");
            minecraft.stage = Stage::Crashed;
            minecraft.metrics.flip_events = 8192; // isolate the stage rule
        }
        let diff = compute_diff(&old, &new, DiffOptions::default());
        let row = verdict_of(&diff, "Minecraft");
        assert_eq!(row.verdict, Verdict::Regressed);
        assert!(row.regressions[0].contains("stage TimedOut -> Crashed"));

        let mut better = real_sweep();
        find(&mut better, "Avatar Frontiers of Pandora").stage = Stage::Rendering;
        let diff = compute_diff(&old, &better, DiffOptions::default());
        let row = verdict_of(&diff, "Avatar Frontiers of Pandora");
        assert_eq!(row.verdict, Verdict::Improved);
    }

    #[test]
    fn a_wall_class_drop_regresses_even_when_the_stage_label_is_unchanged() {
        let old = real_sweep();
        let mut new = real_sweep();
        // Crashed at 57.8s before, crashes at 3s now: same stage, worse.
        find(&mut new, "Avatar Frontiers of Pandora")
            .metrics
            .wall_ms = 3000;
        let diff = compute_diff(&old, &new, DiffOptions::default());
        let row = verdict_of(&diff, "Avatar Frontiers of Pandora");
        assert_eq!(row.verdict, Verdict::Regressed);
        assert!(row.regressions[0].contains("wall-time class 30-120s -> <5s"));
    }

    #[test]
    fn wall_noise_inside_a_class_is_not_a_regression() {
        let old = real_sweep();
        let mut new = real_sweep();
        find(&mut new, "Avatar Frontiers of Pandora")
            .metrics
            .wall_ms = 31000;
        let diff = compute_diff(&old, &new, DiffOptions::default());
        assert_eq!(
            verdict_of(&diff, "Avatar Frontiers of Pandora").verdict,
            Verdict::Unchanged
        );
    }

    #[test]
    fn blocker_appearance_regresses_clearing_improves_and_change_is_a_note() {
        let old = real_sweep();

        let mut appeared = real_sweep();
        find(&mut appeared, "Minecraft").evidence.first_blocker =
            Some("ERROR guest fault at <ADDR>".into());
        let diff = compute_diff(&old, &appeared, DiffOptions::default());
        let row = verdict_of(&diff, "Minecraft");
        assert_eq!(row.verdict, Verdict::Regressed);
        assert_eq!(row.blocker_cell, "appeared");

        let mut cleared = real_sweep();
        find(&mut cleared, "Avatar Frontiers of Pandora")
            .evidence
            .first_blocker = None;
        let diff = compute_diff(&old, &cleared, DiffOptions::default());
        let row = verdict_of(&diff, "Avatar Frontiers of Pandora");
        assert_eq!(row.verdict, Verdict::Improved);
        assert_eq!(row.blocker_cell, "cleared");

        let mut changed = real_sweep();
        find(&mut changed, "Avatar Frontiers of Pandora")
            .evidence
            .first_blocker = Some(
            "2026-07-26T01:02:03.000000Z ERROR ThreadId(02) other::module: different fault".into(),
        );
        let diff = compute_diff(&old, &changed, DiffOptions::default());
        let row = verdict_of(&diff, "Avatar Frontiers of Pandora");
        assert_eq!(
            row.verdict,
            Verdict::Unchanged,
            "a changed blocker alone is a note"
        );
        assert_eq!(row.blocker_cell, "changed");
        assert!(row.notes[0].contains("different fault"));
    }

    #[test]
    fn blocker_identity_survives_timestamps_thread_ids_and_counter_drift() {
        let before = "2026-07-25T22:50:42.832234Z ERROR ThreadId(01) raeen_runtime::dispatch: \
                      guest fault at <ADDR> (execute <ADDR>) — 4096 HLE call(s) recorded before the fault";
        let after = "2026-07-27T09:14:02.000001Z ERROR ThreadId(63) raeen_runtime::dispatch: \
                     guest fault at <ADDR> (execute <ADDR>) — 2733 HLE call(s) recorded before the fault";
        assert_eq!(normalize_blocker(before), normalize_blocker(after));
        let unrelated =
            "ERROR kyty_graphics::shader::parse: unknown smem instruction s_load_dwordx16";
        assert_ne!(normalize_blocker(before), normalize_blocker(unrelated));
    }

    #[test]
    fn a_regression_outranks_concurrent_improvements() {
        let old = real_sweep();
        let mut new = real_sweep();
        {
            let avatar = find(&mut new, "Avatar Frontiers of Pandora");
            avatar.stage = Stage::Rendering; // stage improved…
            avatar.metrics.flip_events = 10; // …but flips collapsed 102 -> 10
        }
        let diff = compute_diff(&old, &new, DiffOptions::default());
        let row = verdict_of(&diff, "Avatar Frontiers of Pandora");
        assert_eq!(row.verdict, Verdict::Regressed);
        assert!(!row.improvements.is_empty(), "progress is still reported");
    }

    #[test]
    fn titles_present_on_only_one_side_are_new_or_removed() {
        let mut old = real_sweep();
        let mut new = real_sweep();
        old.results.retain(|result| result.title == "Minecraft");
        new.results
            .retain(|result| result.title == "Avatar Frontiers of Pandora");
        let diff = compute_diff(&old, &new, DiffOptions::default());
        assert_eq!(verdict_of(&diff, "Minecraft").verdict, Verdict::Removed);
        assert_eq!(
            verdict_of(&diff, "Avatar Frontiers of Pandora").verdict,
            Verdict::New
        );
        // NEW/REMOVED never gate: nothing regressed here.
        assert_eq!(diff.count(Verdict::Regressed), 0);
    }

    #[test]
    fn rendering_carries_verdicts_reasons_and_the_summary_line() {
        let old = real_sweep();
        let mut new = real_sweep();
        find(&mut new, "Minecraft").metrics.flip_events = 100;
        let diff = compute_diff(&old, &new, DiffOptions::default());
        let rendered = render_diff(&diff, "old.json", "new.json", DiffOptions::default());
        assert!(rendered.contains("| Title"), "{rendered}");
        assert!(rendered.contains("REGRESSED"), "{rendered}");
        assert!(rendered.contains("UNCHANGED"), "{rendered}");
        assert!(rendered.contains("flips 8192 -> 100"), "{rendered}");
        assert!(
            rendered.contains("1 regressed"),
            "summary must count regressions: {rendered}"
        );
        assert!(rendered.contains("flip tolerance: ±20%"), "{rendered}");
    }

    #[test]
    fn machine_mismatch_is_called_out() {
        let old = real_sweep();
        let mut new = real_sweep();
        new.machine_id = "machine-other".into();
        let diff = compute_diff(&old, &new, DiffOptions::default());
        let rendered = render_diff(&diff, "a.json", "b.json", DiffOptions::default());
        assert!(rendered.contains("different machines"));
    }

    #[test]
    fn option_parsing_rejects_nonsense_tolerances() {
        assert!(parse_options(&["--flip-tolerance".into(), "-5".into()]).is_err());
        assert!(parse_options(&["--flip-tolerance".into(), "abc".into()]).is_err());
        let parsed = parse_options(&["--flip-tolerance".into(), "35".into()]).unwrap();
        assert_eq!(parsed.flip_tolerance_pct, 35.0);
        assert_eq!(
            parse_options(&[]).unwrap().flip_tolerance_pct,
            DEFAULT_FLIP_TOLERANCE_PCT
        );
    }

    #[test]
    fn positional_extraction_skips_value_flags() {
        let args: Vec<String> = ["old.json", "--flip-tolerance", "30", "new.json", "--strict"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            positionals(&args, &["--flip-tolerance"]),
            vec!["old.json".to_string(), "new.json".to_string()]
        );
    }
}
