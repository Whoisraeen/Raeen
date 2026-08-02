//! `cargo xtask baseline run` / `cargo xtask baseline diff` — the per-game
//! compatibility baseline and its regression tripwire.
//!
//! `run` is the native replacement for `scratch/run-baseline-parts.py`: it
//! measures each registered game as its own short-lived child process (robust
//! to a single silent death: one retry, then move on), writes one part file
//! per game, and merges into `artifacts/compat/latest.json` **only when every
//! game measured**. It never spawns cargo: the prebuilt `raeen.exe` must
//! already exist — building the GUI from inside xtask is exactly the shared
//! `target/` lock deadlock the python driver was written to avoid.
//!
//! `diff` compares two baseline reports (same `RunReport` schema) and prints
//! a human-readable regression/progress report: per-title stage changes,
//! exit-code changes, flip/present deltas, unresolved-NID count deltas, and
//! the newly-missing / newly-resolved NID lists. `--strict` turns detected
//! regressions into a nonzero exit for CI-style gating.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};

use crate::schema::{CompatResult, Registry, RunReport, SCHEMA_VERSION, Stage, UnresolvedNid};
use crate::{
    DEFAULT_REGISTRY, DEFAULT_RESULTS, git_output, has, machine_id, now_ms, option, read_json,
    run_one, safe_name, select_games, strip_ansi, write_json,
};

const DEFAULT_PARTS_DIR: &str = "artifacts/compat/baseline-parts";
/// Matches the python driver's per-game budget.
const DEFAULT_TIMEOUT_SECS: &str = "180";
const DEFAULT_ATTEMPTS: usize = 2;
/// The runtime's first-occurrence marker for a guest call into a per-NID
/// unresolved stub (`raeen-runtime/src/dispatch.rs`).
const UNRESOLVED_MARKER: &str = "UNRESOLVED NID CALLED";
/// Cap per-list NID rendering in `diff` output; the counts stay exact.
const MAX_LISTED_NIDS: usize = 20;

// ---------------------------------------------------------------------------
// baseline run
// ---------------------------------------------------------------------------

pub fn run(args: &[String]) -> Result<()> {
    let registry_path =
        PathBuf::from(option(args, "--registry").unwrap_or_else(|| DEFAULT_REGISTRY.into()));
    let registry: Registry = read_json(&registry_path)?;
    if registry.schema_version != SCHEMA_VERSION {
        bail!("unsupported registry schema {}", registry.schema_version);
    }
    let exe =
        PathBuf::from(option(args, "--exe").unwrap_or_else(|| "target/release/raeen.exe".into()));
    ensure_prebuilt_binary(&exe, has(args, "--allow-stale"))?;
    let timeout_secs = option(args, "--timeout")
        .unwrap_or_else(|| DEFAULT_TIMEOUT_SECS.into())
        .parse::<u64>()
        .context("--timeout must be an integer")?;
    let attempts = option(args, "--attempts")
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("--attempts must be an integer")?
        .unwrap_or(DEFAULT_ATTEMPTS)
        .max(1);
    let tier = option(args, "--tier").unwrap_or_else(|| "all".into());
    let profile = option(args, "--profile").unwrap_or_else(|| "max-fps".into());
    let output = PathBuf::from(option(args, "--output").unwrap_or_else(|| DEFAULT_RESULTS.into()));
    let selected = select_games(&registry.games, &tier)?;
    if selected.is_empty() {
        bail!("registry has no games; run `cargo xtask compat discover` first");
    }

    let run_id = format!("baseline-{}", now_ms());
    let parts_dir =
        PathBuf::from(option(args, "--parts-dir").unwrap_or_else(|| DEFAULT_PARTS_DIR.into()))
            .join(&run_id);
    fs::create_dir_all(&parts_dir)?;
    let raw_dir = PathBuf::from("artifacts/compat/raw").join(&run_id);
    fs::create_dir_all(&raw_dir)?;
    let build_revision = crate::build_identity();
    let machine = machine_id();

    let mut parts: Vec<RunReport> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    for game in &selected {
        let mut last_error = String::new();
        let mut measured = None;
        for attempt in 1..=attempts {
            println!(
                "baseline: {} ({}) attempt {attempt}/{attempts}",
                game.title, game.id
            );
            match run_one(
                &exe,
                game,
                &profile,
                timeout_secs,
                &run_id,
                &build_revision,
                &raw_dir,
            ) {
                Ok(result) => {
                    println!(
                        "  {:?} in {:.1}s",
                        result.stage,
                        result.metrics.wall_ms as f64 / 1000.0
                    );
                    measured = Some(result);
                    break;
                }
                Err(error) => {
                    last_error = format!("{error:#}");
                    eprintln!("  FAILED: {last_error}");
                }
            }
        }
        match measured {
            Some(result) => {
                let part = RunReport {
                    schema_version: SCHEMA_VERSION,
                    generated_unix_ms: now_ms(),
                    machine_id: machine.clone(),
                    results: vec![result],
                };
                let part_path = parts_dir.join(format!("{}.json", safe_name(&game.id)));
                write_json(&part_path, &part)?;
                parts.push(part);
            }
            None => failed.push(format!("{} ({last_error})", game.title)),
        }
    }

    let merged = merge_reports(&parts)?;
    write_json(&parts_dir.join("merged.json"), &merged)?;
    if publishable(merged.results.len(), &failed) {
        write_json(&output, &merged)?;
        println!(
            "COMPLETE: wrote {} ({} results)",
            output.display(),
            merged.results.len()
        );
        Ok(())
    } else {
        println!(
            "PARTIAL: {} measured, {} failed (merged report kept at {}; {} untouched)",
            merged.results.len(),
            failed.len(),
            parts_dir.join("merged.json").display(),
            output.display()
        );
        for failure in &failed {
            println!("  failed: {failure}");
        }
        bail!(
            "{} of {} games did not measure",
            failed.len(),
            selected.len()
        );
    }
}

/// The prebuilt runner must exist and should postdate HEAD; xtask must never
/// build it itself (a concurrent session sharing `target/` deadlocks cargo).
pub(crate) fn ensure_prebuilt_binary(exe: &Path, allow_stale: bool) -> Result<()> {
    let metadata = fs::metadata(exe).map_err(|_| {
        anyhow!(
            "{} does not exist. Build it in a SEPARATE invocation first \
             (`cargo build --release -p raeen-gui`); `xtask baseline` never \
             triggers a build because a concurrent cargo on the shared \
             target/ dir deadlocks.",
            exe.display()
        )
    })?;
    let Ok(modified) = metadata.modified() else {
        return Ok(());
    };
    let Some(head) = head_commit_time() else {
        eprintln!("warning: cannot read HEAD commit time; skipping staleness check");
        return Ok(());
    };
    if let Some(lag) = exe_staleness(modified, head) {
        let message = format!(
            "{} predates the HEAD commit by {}s — it cannot contain HEAD's \
             changes. Rebuild first (`cargo build --release -p raeen-gui`) or \
             pass --allow-stale to measure the old binary knowingly.",
            exe.display(),
            lag.as_secs()
        );
        if allow_stale {
            eprintln!("warning: {message}");
        } else {
            bail!(message);
        }
    }
    Ok(())
}

fn head_commit_time() -> Option<SystemTime> {
    let seconds = git_output(&["log", "-1", "--format=%ct"])
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    Some(UNIX_EPOCH + Duration::from_secs(seconds))
}

/// `Some(lag)` when the binary is older than the commit it is supposed to
/// contain; `None` when it is at least as new.
fn exe_staleness(exe_modified: SystemTime, head_commit: SystemTime) -> Option<Duration> {
    head_commit
        .duration_since(exe_modified)
        .ok()
        .filter(|lag| !lag.is_zero())
}

/// Concatenate per-game part reports into one `RunReport`, preserving order.
/// Every part must carry the current schema; the machine id must be a single
/// consensus value (parts are produced by this process, so a mismatch means
/// someone merged across machines — refuse rather than mislabel evidence).
fn merge_reports(parts: &[RunReport]) -> Result<RunReport> {
    let mut machine = None;
    let mut results = Vec::new();
    for part in parts {
        if part.schema_version != SCHEMA_VERSION {
            bail!(
                "part has schema {} (expected {SCHEMA_VERSION})",
                part.schema_version
            );
        }
        match &machine {
            None => machine = Some(part.machine_id.clone()),
            Some(existing) if *existing != part.machine_id => bail!(
                "parts disagree on machine id ({existing} vs {}); refusing to merge \
                 cross-machine evidence into one report",
                part.machine_id
            ),
            Some(_) => {}
        }
        results.extend(part.results.iter().cloned());
    }
    Ok(RunReport {
        schema_version: SCHEMA_VERSION,
        generated_unix_ms: now_ms(),
        machine_id: machine.unwrap_or_else(|| "unknown".into()),
        results,
    })
}

/// `latest.json` is only replaced by a complete, non-empty run: a partial
/// merge must never masquerade as the baseline (mirrors the python driver).
fn publishable(measured: usize, failed: &[String]) -> bool {
    measured > 0 && failed.is_empty()
}

// ---------------------------------------------------------------------------
// log harvesting (shared with `compat run` via `run_one`)
// ---------------------------------------------------------------------------

/// Harvest the unique unresolved imports a run actually called from its log.
/// The runtime logs each (library, nid) once, as a structured tracing line
/// containing [`UNRESOLVED_MARKER`] plus `nid=` / `library=` / `function=`
/// fields; ANSI is stripped before token scanning so colored field names
/// cannot hide a value.
pub(crate) fn parse_unresolved_nids(text: &str) -> Vec<UnresolvedNid> {
    let mut unique: BTreeMap<(String, String), Option<String>> = BTreeMap::new();
    for line in text.lines().filter(|line| line.contains(UNRESOLVED_MARKER)) {
        let line = strip_ansi(line);
        let field = |name: &str| {
            let prefix = format!("{name}=");
            line.split_whitespace()
                .find_map(|token| token.strip_prefix(&prefix))
                .map(str::to_string)
        };
        let Some(nid) = field("nid") else {
            continue;
        };
        let library = field("library").unwrap_or_else(|| "<unknown>".into());
        let function = field("function").filter(|name| !name.starts_with("nid_0x"));
        unique.entry((library, nid)).or_insert(function);
    }
    unique
        .into_iter()
        .map(|((library, nid), function)| UnresolvedNid {
            library,
            nid,
            function,
        })
        .collect()
}

/// Boot-outcome classification, extracted from the runner so the timeout /
/// exit-code / flip precedence is pinned by tests: a timeout outranks the
/// (killed) exit status, a reported failure outranks flips, and flips decide
/// between a rendering run and a silent clean exit.
pub(crate) fn classify_stage(
    timed_out: bool,
    exit_success: Option<bool>,
    flip_events: u64,
) -> Stage {
    if timed_out {
        Stage::TimedOut
    } else if exit_success == Some(false) {
        Stage::Crashed
    } else if flip_events > 0 {
        Stage::Rendering
    } else {
        Stage::Exited
    }
}

// ---------------------------------------------------------------------------
// baseline diff
// ---------------------------------------------------------------------------

pub fn diff(args: &[String]) -> Result<()> {
    let positional: Vec<&String> = args.iter().filter(|arg| !arg.starts_with("--")).collect();
    let old_path = positional.first().ok_or_else(|| {
        anyhow!("usage: cargo xtask baseline diff <old.json> [new.json] [--strict]")
    })?;
    let new_path = positional
        .get(1)
        .map(|value| value.as_str())
        .unwrap_or(DEFAULT_RESULTS);
    let old: RunReport = read_json(Path::new(old_path.as_str()))?;
    let new: RunReport = read_json(Path::new(new_path))?;
    let report = compute_diff(&old, &new);
    print!("{}", render_diff(&report, old_path, new_path));
    if has(args, "--strict") && report.has_regressions() {
        bail!("{} title(s) regressed", report.regressed_titles());
    }
    Ok(())
}

struct NidDelta {
    old_count: usize,
    new_count: usize,
    newly_missing: Vec<UnresolvedNid>,
    newly_resolved: Vec<UnresolvedNid>,
}

struct TitleDiff {
    title: String,
    game_id: String,
    old_stage: Stage,
    new_stage: Stage,
    old_exit: Option<i32>,
    new_exit: Option<i32>,
    old_flips: u64,
    new_flips: u64,
    old_fps: Option<f64>,
    new_fps: Option<f64>,
    /// `None` when either side never harvested NIDs — deltas would be lies.
    nid_delta: Option<NidDelta>,
    regressions: Vec<String>,
    progress: Vec<String>,
}

struct BaselineDiff {
    old_machine: String,
    new_machine: String,
    old_build: String,
    new_build: String,
    titles: Vec<TitleDiff>,
    only_old: Vec<String>,
    only_new: Vec<String>,
}

impl BaselineDiff {
    fn has_regressions(&self) -> bool {
        self.regressed_titles() > 0
    }

    fn regressed_titles(&self) -> usize {
        self.titles
            .iter()
            .filter(|title| !title.regressions.is_empty())
            .count()
    }
}

/// Coarse "further along is better" order. `TimedOut` means the process
/// survived the whole window (the runner kills it), so it sits above a clean
/// early exit; whether frames were presented is judged separately from
/// `flip_events`, because a rendering run that lives to the timeout is still
/// classified `TimedOut`.
pub(crate) fn stage_rank(stage: Stage) -> u8 {
    match stage {
        Stage::Refused | Stage::Detected => 0,
        Stage::Crashed | Stage::Launching => 1,
        Stage::Exited => 2,
        Stage::TimedOut => 3,
        Stage::Rendering => 4,
    }
}

fn compute_diff(old: &RunReport, new: &RunReport) -> BaselineDiff {
    let by_hash = |report: &RunReport| -> BTreeMap<String, CompatResult> {
        report
            .results
            .iter()
            .map(|result| (result.content_sha1.clone(), result.clone()))
            .collect()
    };
    let old_map = by_hash(old);
    let new_map = by_hash(new);
    let build_of = |report: &RunReport| {
        report
            .results
            .first()
            .map(|result| result.build_revision.clone())
            .unwrap_or_else(|| "unknown".into())
    };

    let mut titles = Vec::new();
    for (hash, previous) in &old_map {
        let Some(current) = new_map.get(hash) else {
            continue;
        };
        titles.push(diff_title(previous, current));
    }
    let only_old = old_map
        .iter()
        .filter(|(hash, _)| !new_map.contains_key(*hash))
        .map(|(_, result)| result.title.clone())
        .collect();
    let only_new = new_map
        .iter()
        .filter(|(hash, _)| !old_map.contains_key(*hash))
        .map(|(_, result)| result.title.clone())
        .collect();
    BaselineDiff {
        old_machine: old.machine_id.clone(),
        new_machine: new.machine_id.clone(),
        old_build: build_of(old),
        new_build: build_of(new),
        titles,
        only_old,
        only_new,
    }
}

fn diff_title(old: &CompatResult, new: &CompatResult) -> TitleDiff {
    let mut regressions = Vec::new();
    let mut progress = Vec::new();

    let (old_rank, new_rank) = (stage_rank(old.stage), stage_rank(new.stage));
    if new_rank < old_rank {
        regressions.push(format!("stage {:?} -> {:?}", old.stage, new.stage));
    } else if new_rank > old_rank {
        progress.push(format!("stage {:?} -> {:?}", old.stage, new.stage));
    }

    let (old_flips, new_flips) = (old.metrics.flip_events, new.metrics.flip_events);
    if old_flips > 0 && new_flips == 0 {
        regressions.push(format!(
            "stopped presenting frames ({old_flips} flips -> 0)"
        ));
    } else if old_flips == 0 && new_flips > 0 {
        progress.push(format!(
            "started presenting frames (0 flips -> {new_flips})"
        ));
    }

    let nid_delta = match (&old.evidence.unresolved_nids, &new.evidence.unresolved_nids) {
        (Some(old_nids), Some(new_nids)) => {
            let key = |nid: &UnresolvedNid| (nid.library.clone(), nid.nid.clone());
            let old_set: BTreeMap<_, _> =
                old_nids.iter().map(|nid| (key(nid), nid.clone())).collect();
            let new_set: BTreeMap<_, _> =
                new_nids.iter().map(|nid| (key(nid), nid.clone())).collect();
            let newly_missing: Vec<UnresolvedNid> = new_set
                .iter()
                .filter(|(key, _)| !old_set.contains_key(*key))
                .map(|(_, nid)| nid.clone())
                .collect();
            let newly_resolved: Vec<UnresolvedNid> = old_set
                .iter()
                .filter(|(key, _)| !new_set.contains_key(*key))
                .map(|(_, nid)| nid.clone())
                .collect();
            if !newly_missing.is_empty() {
                regressions.push(format!("{} newly-missing NID(s)", newly_missing.len()));
            }
            if !newly_resolved.is_empty() {
                progress.push(format!("{} newly-resolved NID(s)", newly_resolved.len()));
            }
            Some(NidDelta {
                old_count: old_set.len(),
                new_count: new_set.len(),
                newly_missing,
                newly_resolved,
            })
        }
        _ => None,
    };

    TitleDiff {
        title: new.title.clone(),
        game_id: new.game_id.clone(),
        old_stage: old.stage,
        new_stage: new.stage,
        old_exit: old.metrics.exit_code,
        new_exit: new.metrics.exit_code,
        old_flips,
        new_flips,
        old_fps: old.metrics.observed_fps,
        new_fps: new.metrics.observed_fps,
        nid_delta,
        regressions,
        progress,
    }
}

fn fmt_exit(code: Option<i32>) -> String {
    code.map(|value| value.to_string())
        .unwrap_or_else(|| "killed".into())
}

fn fmt_fps(fps: Option<f64>) -> String {
    fps.map(|value| format!("{value:.1}"))
        .unwrap_or_else(|| "n/a".into())
}

fn fmt_nid_list(list: &[UnresolvedNid]) -> String {
    let mut shown: Vec<String> = list
        .iter()
        .take(MAX_LISTED_NIDS)
        .map(|nid| {
            format!(
                "{} {} {}",
                nid.library,
                nid.nid,
                nid.function.as_deref().unwrap_or("<anonymous>")
            )
        })
        .collect();
    if list.len() > MAX_LISTED_NIDS {
        shown.push(format!("... and {} more", list.len() - MAX_LISTED_NIDS));
    }
    shown.join("\n      ")
}

fn render_diff(report: &BaselineDiff, old_path: &str, new_path: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "baseline diff: {old_path} (build {}) -> {new_path} (build {})\n",
        report.old_build, report.new_build
    ));
    if report.old_machine != report.new_machine {
        out.push_str(&format!(
            "WARNING: reports come from different machines ({} vs {}); metric deltas are not comparable\n",
            report.old_machine, report.new_machine
        ));
    }
    for title in &report.titles {
        out.push_str(&format!("\n== {} ({}) ==\n", title.title, title.game_id));
        out.push_str(&format!(
            "  stage: {:?} -> {:?}\n",
            title.old_stage, title.new_stage
        ));
        if title.old_exit != title.new_exit {
            out.push_str(&format!(
                "  exit code: {} -> {}\n",
                fmt_exit(title.old_exit),
                fmt_exit(title.new_exit)
            ));
        }
        out.push_str(&format!(
            "  flips: {} -> {} ({:+})\n",
            title.old_flips,
            title.new_flips,
            title.new_flips as i128 - title.old_flips as i128
        ));
        if title.old_fps.is_some() || title.new_fps.is_some() {
            out.push_str(&format!(
                "  observed fps: {} -> {}\n",
                fmt_fps(title.old_fps),
                fmt_fps(title.new_fps)
            ));
        }
        match &title.nid_delta {
            None => out.push_str(
                "  unresolved NIDs: not comparable (at least one run predates NID harvesting)\n",
            ),
            Some(delta) => {
                out.push_str(&format!(
                    "  unresolved NIDs called: {} -> {} ({:+})\n",
                    delta.old_count,
                    delta.new_count,
                    delta.new_count as i128 - delta.old_count as i128
                ));
                if !delta.newly_resolved.is_empty() {
                    out.push_str(&format!(
                        "    newly resolved ({}):\n      {}\n",
                        delta.newly_resolved.len(),
                        fmt_nid_list(&delta.newly_resolved)
                    ));
                }
                if !delta.newly_missing.is_empty() {
                    out.push_str(&format!(
                        "    newly missing ({}):\n      {}\n",
                        delta.newly_missing.len(),
                        fmt_nid_list(&delta.newly_missing)
                    ));
                }
            }
        }
        for regression in &title.regressions {
            out.push_str(&format!("  REGRESSION: {regression}\n"));
        }
        for item in &title.progress {
            out.push_str(&format!("  PROGRESS: {item}\n"));
        }
    }
    if !report.only_old.is_empty() {
        out.push_str(&format!(
            "\nonly in old report: {}\n",
            report.only_old.join(", ")
        ));
    }
    if !report.only_new.is_empty() {
        out.push_str(&format!(
            "\nonly in new report: {}\n",
            report.only_new.join(", ")
        ));
    }
    let regressed = report.regressed_titles();
    let progressed = report
        .titles
        .iter()
        .filter(|title| !title.progress.is_empty())
        .count();
    out.push_str(&format!(
        "\nsummary: {} matched title(s), {} regressed, {} progressed, {} only-old, {} only-new\n",
        report.titles.len(),
        regressed,
        progressed,
        report.only_old.len(),
        report.only_new.len()
    ));
    out
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Evidence, Metrics};

    fn nid(library: &str, nid: &str, function: Option<&str>) -> UnresolvedNid {
        UnresolvedNid {
            library: library.into(),
            nid: nid.into(),
            function: function.map(str::to_string),
        }
    }

    fn result(
        title: &str,
        sha: &str,
        stage: Stage,
        exit_code: Option<i32>,
        flips: u64,
        unresolved: Option<Vec<UnresolvedNid>>,
    ) -> CompatResult {
        CompatResult {
            schema_version: SCHEMA_VERSION,
            measured_unix_ms: 0,
            run_id: "baseline-test".into(),
            build_revision: "abc123".into(),
            profile: "max-fps".into(),
            game_id: format!("id-{title}"),
            title: title.into(),
            content_sha1: sha.into(),
            stage,
            metrics: Metrics {
                wall_ms: 1000,
                exit_code,
                flip_events: flips,
                ..Metrics::default()
            },
            evidence: Evidence {
                log_sha1: "0".repeat(40),
                blocker_signature: None,
                first_blocker: None,
                measured: true,
                unresolved_nids: unresolved,
            },
        }
    }

    fn report(machine: &str, results: Vec<CompatResult>) -> RunReport {
        RunReport {
            schema_version: SCHEMA_VERSION,
            generated_unix_ms: 0,
            machine_id: machine.into(),
            results,
        }
    }

    // -- timeout / exit-code classification --------------------------------

    #[test]
    fn timeout_outranks_exit_status_and_flips() {
        assert_eq!(classify_stage(true, Some(true), 500), Stage::TimedOut);
        assert_eq!(classify_stage(true, None, 0), Stage::TimedOut);
    }

    #[test]
    fn reported_failure_is_a_crash_even_with_flips() {
        assert_eq!(classify_stage(false, Some(false), 500), Stage::Crashed);
    }

    #[test]
    fn clean_exit_classifies_by_presented_frames() {
        assert_eq!(classify_stage(false, Some(true), 3), Stage::Rendering);
        assert_eq!(classify_stage(false, Some(true), 0), Stage::Exited);
        // Unknown exit status (kill/wait raced) without a timeout must not
        // invent a crash: flips still decide.
        assert_eq!(classify_stage(false, None, 1), Stage::Rendering);
    }

    // -- unresolved-NID log harvesting --------------------------------------

    #[test]
    fn parses_structured_unresolved_lines_through_ansi() {
        let log = concat!(
            "2026-07-27T07:00:00Z INFO boot: loaded eboot.bin\n",
            "2026-07-27T07:00:01Z \u{1b}[33mWARN\u{1b}[0m raeen_runtime::dispatch: ",
            "UNRESOLVED NID CALLED \u{1b}[3mnid\u{1b}[0m=0x00000000deadbeef ",
            "library=libSceVoice function=sceVoiceInit calling_module=eboot.bin count=1 strict=false\n",
            "some unrelated ERROR line\n",
        );
        let parsed = parse_unresolved_nids(log);
        assert_eq!(
            parsed,
            vec![nid(
                "libSceVoice",
                "0x00000000deadbeef",
                Some("sceVoiceInit")
            )]
        );
    }

    #[test]
    fn deduplicates_and_sorts_by_library_then_nid() {
        let log = concat!(
            "UNRESOLVED NID CALLED nid=0x02 library=libB function=b2\n",
            "UNRESOLVED NID CALLED nid=0x01 library=libB function=b1\n",
            "UNRESOLVED NID CALLED nid=0x01 library=libA function=a1\n",
            "UNRESOLVED NID CALLED nid=0x01 library=libB function=b1\n",
        );
        let parsed = parse_unresolved_nids(log);
        assert_eq!(
            parsed,
            vec![
                nid("libA", "0x01", Some("a1")),
                nid("libB", "0x01", Some("b1")),
                nid("libB", "0x02", Some("b2")),
            ]
        );
    }

    #[test]
    fn anonymous_nids_keep_no_fake_name_and_missing_fields_are_tolerated() {
        let log = concat!(
            // `describe` falls back to nid_0x… when the dictionary has no
            // hash-proven name; that placeholder must not persist as a name.
            "UNRESOLVED NID CALLED nid=0x0a library=libX function=nid_0x000000000000000a\n",
            // No library field at all: keep the call, mark provider unknown.
            "UNRESOLVED NID CALLED nid=0x0b function=orphan\n",
            // No nid field: unusable, skipped.
            "UNRESOLVED NID CALLED library=libY function=broken\n",
        );
        let parsed = parse_unresolved_nids(log);
        assert_eq!(
            parsed,
            vec![
                nid("<unknown>", "0x0b", Some("orphan")),
                nid("libX", "0x0a", None),
            ]
        );
    }

    // -- schema round-trip ---------------------------------------------------

    #[test]
    fn pre_harvest_reports_parse_and_reserialize_without_the_new_field() {
        // Exactly the shape every existing latest.json result carries.
        let old_json = r#"{
            "log_sha1": "aa",
            "blocker_signature": null,
            "first_blocker": null,
            "measured": true
        }"#;
        let evidence: Evidence = serde_json::from_str(old_json).expect("old evidence parses");
        assert_eq!(evidence.unresolved_nids, None);
        let back = serde_json::to_string(&evidence).expect("serializes");
        assert!(
            !back.contains("unresolved_nids"),
            "absent field must stay absent so old reports round-trip: {back}"
        );
    }

    #[test]
    fn harvested_nids_round_trip_including_measured_empty() {
        for value in [Some(Vec::new()), Some(vec![nid("libA", "0x01", None)])] {
            let result = result("T", "sha", Stage::Rendering, Some(0), 5, value.clone());
            let json = serde_json::to_string(&result).expect("serializes");
            assert!(json.contains("unresolved_nids"));
            let back: CompatResult = serde_json::from_str(&json).expect("parses");
            assert_eq!(back.evidence.unresolved_nids, value);
        }
    }

    // -- merge + publish gating ----------------------------------------------

    #[test]
    fn merge_concatenates_parts_in_order_with_consensus_machine() {
        let parts = vec![
            report(
                "machine-a",
                vec![result("A", "1", Stage::Rendering, Some(0), 5, None)],
            ),
            report(
                "machine-a",
                vec![result("B", "2", Stage::Crashed, Some(1), 0, None)],
            ),
        ];
        let merged = merge_reports(&parts).expect("merges");
        assert_eq!(merged.machine_id, "machine-a");
        assert_eq!(
            merged
                .results
                .iter()
                .map(|result| result.title.as_str())
                .collect::<Vec<_>>(),
            vec!["A", "B"]
        );
    }

    #[test]
    fn merge_refuses_cross_machine_parts_and_wrong_schema() {
        let cross = vec![
            report(
                "machine-a",
                vec![result("A", "1", Stage::Exited, Some(0), 0, None)],
            ),
            report(
                "machine-b",
                vec![result("B", "2", Stage::Exited, Some(0), 0, None)],
            ),
        ];
        assert!(merge_reports(&cross).is_err());
        let mut wrong = report("machine-a", Vec::new());
        wrong.schema_version = SCHEMA_VERSION + 1;
        assert!(merge_reports(&[wrong]).is_err());
    }

    #[test]
    fn latest_is_only_replaced_by_a_complete_nonempty_run() {
        assert!(publishable(3, &[]));
        assert!(!publishable(3, &["Minecraft (spawn failed)".into()]));
        assert!(!publishable(0, &[]));
    }

    // -- staleness -------------------------------------------------------------

    #[test]
    fn binary_older_than_head_is_stale_and_newer_is_not() {
        let head = UNIX_EPOCH + Duration::from_secs(1_000_000);
        let older = UNIX_EPOCH + Duration::from_secs(999_000);
        let newer = UNIX_EPOCH + Duration::from_secs(1_000_500);
        assert_eq!(exe_staleness(older, head), Some(Duration::from_secs(1000)));
        assert_eq!(exe_staleness(newer, head), None);
        assert_eq!(exe_staleness(head, head), None);
    }

    // -- diff --------------------------------------------------------------------

    #[test]
    fn diff_reports_progress_for_resolved_nids_and_new_frames() {
        let old = report(
            "m",
            vec![result(
                "Astro",
                "sha-a",
                Stage::Crashed,
                Some(1),
                0,
                Some(vec![
                    nid("libSceAgc", "0x01", Some("sceAgcSubmit")),
                    nid("libSceVoice", "0x02", None),
                ]),
            )],
        );
        let new = report(
            "m",
            vec![result(
                "Astro",
                "sha-a",
                Stage::Rendering,
                Some(0),
                120,
                Some(vec![nid("libSceVoice", "0x02", None)]),
            )],
        );
        let diff = compute_diff(&old, &new);
        assert_eq!(diff.titles.len(), 1);
        let title = &diff.titles[0];
        assert!(title.regressions.is_empty(), "{:?}", title.regressions);
        assert_eq!(title.progress.len(), 3); // stage up, frames started, 1 resolved
        let delta = title.nid_delta.as_ref().expect("both sides measured");
        assert_eq!((delta.old_count, delta.new_count), (2, 1));
        assert_eq!(
            delta.newly_resolved,
            vec![nid("libSceAgc", "0x01", Some("sceAgcSubmit"))]
        );
        assert!(delta.newly_missing.is_empty());
        assert!(!diff.has_regressions());
    }

    #[test]
    fn diff_flags_stage_drop_lost_frames_and_new_missing_nids_as_regressions() {
        let old = report(
            "m",
            vec![result(
                "MC",
                "sha-m",
                Stage::Rendering,
                Some(0),
                500,
                Some(vec![]),
            )],
        );
        let new = report(
            "m",
            vec![result(
                "MC",
                "sha-m",
                Stage::Crashed,
                Some(3),
                0,
                Some(vec![nid(
                    "libSceAmpr",
                    "0x0c",
                    Some("sceAmprCommandBufferX"),
                )]),
            )],
        );
        let diff = compute_diff(&old, &new);
        let title = &diff.titles[0];
        assert_eq!(title.regressions.len(), 3); // stage, frames, missing NID
        assert!(diff.has_regressions());
        assert_eq!(diff.regressed_titles(), 1);
        let rendered = render_diff(&diff, "old.json", "new.json");
        assert!(rendered.contains("REGRESSION: stage Rendering -> Crashed"));
        assert!(rendered.contains("exit code: 0 -> 3"));
        assert!(rendered.contains("newly missing (1)"));
        assert!(rendered.contains("sceAmprCommandBufferX"));
        assert!(rendered.contains("1 regressed"));
    }

    #[test]
    fn diff_declines_nid_deltas_when_old_run_predates_harvesting() {
        let old = report(
            "m",
            vec![result("UD", "sha-u", Stage::Crashed, Some(1), 0, None)],
        );
        let new = report(
            "m",
            vec![result(
                "UD",
                "sha-u",
                Stage::Crashed,
                Some(1),
                0,
                Some(vec![nid("libX", "0x01", None)]),
            )],
        );
        let diff = compute_diff(&old, &new);
        let title = &diff.titles[0];
        assert!(title.nid_delta.is_none());
        assert!(title.regressions.is_empty(), "no fake NID regression");
        let rendered = render_diff(&diff, "old.json", "new.json");
        assert!(rendered.contains("not comparable"));
    }

    #[test]
    fn diff_lists_unmatched_titles_and_warns_on_machine_mismatch() {
        let old = report(
            "machine-a",
            vec![
                result("Kept", "sha-1", Stage::Exited, Some(0), 0, None),
                result("Removed", "sha-2", Stage::Exited, Some(0), 0, None),
            ],
        );
        let new = report(
            "machine-b",
            vec![
                result("Kept", "sha-1", Stage::Exited, Some(0), 0, None),
                result("Added", "sha-3", Stage::Exited, Some(0), 0, None),
            ],
        );
        let diff = compute_diff(&old, &new);
        assert_eq!(diff.only_old, vec!["Removed".to_string()]);
        assert_eq!(diff.only_new, vec!["Added".to_string()]);
        let rendered = render_diff(&diff, "a.json", "b.json");
        assert!(rendered.contains("different machines"));
        assert!(rendered.contains("only in old report: Removed"));
        assert!(rendered.contains("only in new report: Added"));
    }

    #[test]
    fn stage_ranks_match_the_documented_order() {
        assert!(stage_rank(Stage::Rendering) > stage_rank(Stage::TimedOut));
        assert!(stage_rank(Stage::TimedOut) > stage_rank(Stage::Exited));
        assert!(stage_rank(Stage::Exited) > stage_rank(Stage::Crashed));
        assert!(stage_rank(Stage::Crashed) > stage_rank(Stage::Refused));
    }

    #[test]
    fn long_nid_lists_are_capped_in_rendering_with_exact_counts() {
        let many: Vec<UnresolvedNid> = (0..25)
            .map(|index| nid("libX", &format!("0x{index:02x}"), None))
            .collect();
        let rendered = fmt_nid_list(&many);
        assert!(rendered.contains("... and 5 more"));
    }
}
