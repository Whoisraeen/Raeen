//! `cargo xtask soak` — long-run liveness harness (checklist item 9).
//!
//! Launches ONE registered game via the prebuilt `raeen.exe` (same
//! prebuilt-binary / staleness rules as `baseline run`; xtask never builds)
//! and then, unlike the baseline's run-to-timeout-and-read-the-log model,
//! MONITORS the run live for the whole soak window:
//!
//! * **Frame-epoch advancement** — tails the child's stdout/stderr for the
//!   same flip telemetry the baseline harvests after the fact: the GPU
//!   worker's `WORKER TIMING … flips=N frame_ms=X` window lines (emitted
//!   every 32 completed presents under `RAEEN_TIME_WORKER=1`), the AGC
//!   `total_flips=N` progress lines, and per-call `sceVideoOutSubmitFlip`
//!   lines. Any high-water increase or new flip line is an epoch advance.
//! * **Deadlock warnings** — the runtime's own warning
//!   `scePthreadMutexLock stuck >3s — deadlock; naming the holder`
//!   (pthread_sync.rs), the line that located the Minecraft streaming-pool
//!   bug. One observed warning fails the soak immediately.
//! * **Process-tree resources** — CPU% and memory via `sysinfo`, sampled
//!   periodically over the whole tree rooted at the child.
//!
//! FAIL: no epoch advance for more than `--stall-secs` (default 10, armed
//! only after the first advance; `--boot-secs` budgets the boot itself), a
//! deadlock warning, or the process exiting before the deadline. On failure
//! the harness prints the frozen-window timestamps and the log tail and
//! exits nonzero. On success it prints a stability report (min/avg/max
//! epoch rate, worst stall, peak memory/CPU).
//!
//! Synthetic input: `--input <spec|file>` forwards a deterministic
//! controller timeline to the runner via `RAEEN_INPUT_SCRIPT` (+
//! `RAEEN_RUNNER_CHILD=1`, which starts the runner input thread — script
//! states win over Shell IPC and native pads). The format is
//! `raeen-input`'s compact replay spec (`0:neutral;180000:cross;…`,
//! milliseconds since the input thread started); the spec is validated here
//! before launch so a typo fails in seconds, not 30 minutes in. The final
//! snapshot holds forever, so a trailing `…:ls_up` keeps walking for the
//! rest of the soak. With `--input none` (default) the soak still detects
//! frozen frames and deadlocks, but only on the boot/idle path — reduced
//! coverage, noted in the report.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::baseline::ensure_prebuilt_binary;
use crate::schema::{GameRecord, Registry, SCHEMA_VERSION};
use crate::{
    DEFAULT_REGISTRY, git_output, has, max_metric, now_ms, option, read_json, strip_ansi,
    terminate_process_tree,
};

const DEFAULT_MINUTES: f64 = 30.0;
const DEFAULT_STALL_SECS: f64 = 10.0;
/// Budget for the FIRST epoch advance: Minecraft needs well over the stall
/// threshold to reach its first present, and a boot that never flips at all
/// is its own failure mode, so it gets its own (generous) clock.
const DEFAULT_BOOT_SECS: f64 = 180.0;
const DEFAULT_OUTPUT_DIR: &str = "artifacts/soak";
/// Log poll cadence. Liveness thresholds are 10s-scale, so up to one poll of
/// stamp skew on an observed line is noise.
const POLL: Duration = Duration::from_millis(500);
/// Resource sampling is a full system process sweep; every 4th poll (2 s) is
/// plenty for a 30-minute trend and stays above sysinfo's minimum CPU
/// update interval.
const RESOURCE_SAMPLE_EVERY: u32 = 4;
/// The runtime's contended-mutex forensic line (`pthread_sync.rs`): fires
/// once per lock call after 3 s of waiting and names mutex/owner/waiter.
const DEADLOCK_MARKER: &str = "scePthreadMutexLock stuck >3s";
/// Raw-log ring kept for the failure report.
const TAIL_LINES: usize = 80;
const TAIL_LINE_CAP: usize = 400;

// ---------------------------------------------------------------------------
// pure: line assembly from appended log bytes
// ---------------------------------------------------------------------------

/// Turns arbitrarily-chunked appended bytes into complete lines, holding the
/// trailing partial line until its newline arrives.
#[derive(Default)]
pub(crate) struct LineAssembler {
    partial: Vec<u8>,
}

impl LineAssembler {
    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Vec<String> {
        self.partial.extend_from_slice(chunk);
        let Some(last_newline) = self.partial.iter().rposition(|&byte| byte == b'\n') else {
            return Vec::new();
        };
        let complete: Vec<u8> = self.partial.drain(..=last_newline).collect();
        complete
            .split(|&byte| byte == b'\n')
            .filter(|segment| !segment.is_empty() || complete.is_empty())
            .map(|segment| {
                let segment = segment.strip_suffix(b"\r").unwrap_or(segment);
                String::from_utf8_lossy(segment).into_owned()
            })
            .filter(|line| !line.is_empty())
            .collect()
    }

    /// Hand back the unterminated remainder (used once, after the child is
    /// dead, so a final crash line without a newline is still observed).
    pub(crate) fn flush(&mut self) -> Option<String> {
        if self.partial.is_empty() {
            return None;
        }
        let line = String::from_utf8_lossy(&self.partial).into_owned();
        self.partial.clear();
        (!line.trim().is_empty()).then_some(line)
    }
}

// ---------------------------------------------------------------------------
// pure: liveness tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeadlockWarning {
    pub elapsed: Duration,
    pub mutex: String,
    pub owner: String,
    pub owner_name: String,
    pub waiter_name: String,
    pub line: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SoakFailure {
    Deadlock(DeadlockWarning),
    /// The run never produced a single epoch advance within the boot budget.
    NeverPresented {
        waited: Duration,
    },
    /// Epochs advanced, then froze past the stall limit.
    Stalled {
        last_advance: Duration,
        gap: Duration,
    },
    /// The child died (crash or clean exit) before the soak deadline.
    ProcessExited {
        exit_code: Option<i32>,
        at: Duration,
    },
}

/// Everything the monitor knows about liveness, fed one log line at a time.
/// Lines are stamped with the monitor's own elapsed clock at observation
/// (not parsed log timestamps): up to one poll interval of skew against
/// 10s-scale thresholds.
pub(crate) struct SoakTracker {
    stall_limit: Duration,
    boot_limit: Duration,
    flip_high_water: u64,
    flip_call_lines: u64,
    first_advance: Option<Duration>,
    last_advance: Option<Duration>,
    worst_closed_stall: Duration,
    frame_ms_samples: Vec<f64>,
    deadlocks: Vec<DeadlockWarning>,
}

impl SoakTracker {
    pub(crate) fn new(stall_limit: Duration, boot_limit: Duration) -> Self {
        Self {
            stall_limit,
            boot_limit,
            flip_high_water: 0,
            flip_call_lines: 0,
            first_advance: None,
            last_advance: None,
            worst_closed_stall: Duration::ZERO,
            frame_ms_samples: Vec::new(),
            deadlocks: Vec::new(),
        }
    }

    pub(crate) fn observe_line(&mut self, elapsed: Duration, raw: &str) {
        let line = strip_ansi(raw);
        if line.contains(DEADLOCK_MARKER) {
            self.deadlocks.push(parse_deadlock_line(elapsed, &line));
        }
        let mut advanced = false;
        // `max_metric` tokenizes on whitespace, so `flips=` cannot match a
        // `total_flips=` token and vice versa.
        let high_water = max_metric(&line, "flips").max(max_metric(&line, "total_flips"));
        if high_water > self.flip_high_water {
            self.flip_high_water = high_water;
            advanced = true;
        }
        if line.contains("sceVideoOutSubmitFlip") {
            self.flip_call_lines += 1;
            advanced = true;
        }
        for value in line
            .split_whitespace()
            .filter_map(|token| token.strip_prefix("frame_ms="))
            .filter_map(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            self.frame_ms_samples.push(value);
        }
        if advanced {
            self.record_advance(elapsed);
        }
    }

    fn record_advance(&mut self, elapsed: Duration) {
        if let Some(last) = self.last_advance {
            let gap = elapsed.saturating_sub(last);
            if gap > self.worst_closed_stall {
                self.worst_closed_stall = gap;
            }
        } else {
            self.first_advance = Some(elapsed);
        }
        self.last_advance = Some(elapsed);
    }

    /// Liveness verdict at `elapsed`. Deadlock outranks stall (it names the
    /// cause); the stall clock arms only once something has ever advanced,
    /// with the boot budget covering the window before that.
    pub(crate) fn check(&self, elapsed: Duration) -> Option<SoakFailure> {
        if let Some(warning) = self.deadlocks.first() {
            return Some(SoakFailure::Deadlock(warning.clone()));
        }
        match self.last_advance {
            Some(last) => {
                let gap = elapsed.saturating_sub(last);
                (gap > self.stall_limit).then_some(SoakFailure::Stalled {
                    last_advance: last,
                    gap,
                })
            }
            None => (elapsed >= self.boot_limit)
                .then_some(SoakFailure::NeverPresented { waited: elapsed }),
        }
    }

    /// Worst observed epoch gap, including the still-open one at the end of
    /// the run (a freeze that starts 9 s before the deadline must not hide).
    pub(crate) fn worst_stall(&self, final_elapsed: Duration) -> Option<Duration> {
        self.last_advance.map(|last| {
            self.worst_closed_stall
                .max(final_elapsed.saturating_sub(last))
        })
    }
}

/// Extract the named fields from the runtime's deadlock warning. Token-based:
/// a thread name containing spaces keeps only its first word here, but the
/// full line is preserved verbatim for the report.
pub(crate) fn parse_deadlock_line(elapsed: Duration, clean_line: &str) -> DeadlockWarning {
    let field = |name: &str| {
        let prefix = format!("{name}=");
        clean_line
            .split_whitespace()
            .find_map(|token| token.strip_prefix(&prefix))
            .unwrap_or("<unknown>")
            .to_string()
    };
    DeadlockWarning {
        elapsed,
        mutex: field("mutex"),
        owner: field("owner"),
        owner_name: field("owner_name"),
        waiter_name: field("waiter_name"),
        line: clean_line.chars().take(TAIL_LINE_CAP).collect(),
    }
}

// ---------------------------------------------------------------------------
// pure: resource accumulation + process-tree membership
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct ResourceStats {
    samples: u64,
    cpu_sum: f64,
    pub peak_cpu_pct: f64,
    pub peak_memory_bytes: u64,
}

impl ResourceStats {
    /// `cpu_pct` is the process-tree sum of per-core percentages (100 = one
    /// full core, sysinfo convention).
    pub(crate) fn record(&mut self, cpu_pct: f64, memory_bytes: u64) {
        self.samples += 1;
        self.cpu_sum += cpu_pct;
        if cpu_pct > self.peak_cpu_pct {
            self.peak_cpu_pct = cpu_pct;
        }
        if memory_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = memory_bytes;
        }
    }

    /// Track a memory reading without contributing a CPU sample (used for
    /// the first sysinfo sweep, whose CPU% is 0 by construction).
    pub(crate) fn record_memory(&mut self, memory_bytes: u64) {
        if memory_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = memory_bytes;
        }
    }

    pub(crate) fn avg_cpu_pct(&self) -> Option<f64> {
        (self.samples > 0).then(|| self.cpu_sum / self.samples as f64)
    }
}

/// Pids whose parent chain reaches `root` (plus `root` itself), from a
/// pid→parent snapshot. Cycle-safe: a corrupt chain stops at a repeat.
pub(crate) fn process_tree(root: u32, parents: &BTreeMap<u32, Option<u32>>) -> BTreeSet<u32> {
    let mut members = BTreeSet::from([root]);
    for &pid in parents.keys() {
        let mut visited = BTreeSet::new();
        let mut cursor = pid;
        let chain_reaches_root = loop {
            if members.contains(&cursor) {
                break true;
            }
            if !visited.insert(cursor) {
                break false;
            }
            match parents.get(&cursor) {
                Some(Some(parent)) => cursor = *parent,
                _ => break false,
            }
        };
        if chain_reaches_root {
            members.extend(visited);
            members.insert(pid);
        }
    }
    members
}

// ---------------------------------------------------------------------------
// pure: report
// ---------------------------------------------------------------------------

pub(crate) struct SoakReport {
    pub title: String,
    pub game_id: String,
    pub build_revision: String,
    pub input_description: String,
    pub planned: Duration,
    pub actual: Duration,
    pub flips: u64,
    pub flip_call_lines: u64,
    pub first_present: Option<Duration>,
    pub worst_stall: Option<Duration>,
    /// Presents per second averaged from first present to the end of the run.
    pub overall_flips_per_sec: Option<f64>,
    /// (min, avg, max) FPS over the GPU worker's 32-present windows.
    pub window_fps: Option<(f64, f64, f64)>,
    pub deadlock_count: usize,
    pub avg_cpu_pct: Option<f64>,
    pub peak_cpu_pct: Option<f64>,
    pub peak_memory_bytes: Option<u64>,
}

/// The launch-time facts a report carries verbatim (everything measured
/// lives in the tracker/resource accumulators).
pub(crate) struct RunMeta<'a> {
    pub title: &'a str,
    pub game_id: &'a str,
    pub build_revision: &'a str,
    pub input_description: &'a str,
    pub planned: Duration,
}

pub(crate) fn build_report(
    tracker: &SoakTracker,
    resources: &ResourceStats,
    meta: &RunMeta<'_>,
    actual: Duration,
) -> SoakReport {
    let overall_flips_per_sec = tracker.first_advance.and_then(|first| {
        let window = actual.saturating_sub(first).as_secs_f64();
        (window > 0.0 && tracker.flip_high_water > 0)
            .then(|| tracker.flip_high_water as f64 / window)
    });
    let window_fps = (!tracker.frame_ms_samples.is_empty()).then(|| {
        let fps: Vec<f64> = tracker
            .frame_ms_samples
            .iter()
            .map(|frame_ms| 1000.0 / frame_ms)
            .collect();
        let min = fps.iter().copied().fold(f64::INFINITY, f64::min);
        let max = fps.iter().copied().fold(0.0_f64, f64::max);
        let avg = fps.iter().sum::<f64>() / fps.len() as f64;
        (min, avg, max)
    });
    SoakReport {
        title: meta.title.to_string(),
        game_id: meta.game_id.to_string(),
        build_revision: meta.build_revision.to_string(),
        input_description: meta.input_description.to_string(),
        planned: meta.planned,
        actual,
        flips: tracker.flip_high_water,
        flip_call_lines: tracker.flip_call_lines,
        first_present: tracker.first_advance,
        worst_stall: tracker.worst_stall(actual),
        overall_flips_per_sec,
        window_fps,
        deadlock_count: tracker.deadlocks.len(),
        avg_cpu_pct: resources.avg_cpu_pct(),
        peak_cpu_pct: (resources.samples > 0).then_some(resources.peak_cpu_pct),
        peak_memory_bytes: (resources.samples > 0).then_some(resources.peak_memory_bytes),
    }
}

pub(crate) fn fmt_duration(duration: Duration) -> String {
    let total = duration.as_secs_f64();
    if total >= 60.0 {
        format!("{}m{:04.1}s", (total / 60.0) as u64, total % 60.0)
    } else {
        format!("{total:.1}s")
    }
}

fn fmt_opt_fps(value: Option<f64>) -> String {
    value
        .map(|fps| format!("{fps:.1}"))
        .unwrap_or_else(|| "n/a".into())
}

pub(crate) fn render_report(report: &SoakReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "soak report: {} ({}) build {}\n",
        report.title, report.game_id, report.build_revision
    ));
    out.push_str(&format!(
        "  duration: {} of {} planned\n",
        fmt_duration(report.actual),
        fmt_duration(report.planned)
    ));
    out.push_str(&format!("  input: {}\n", report.input_description));
    out.push_str(&format!(
        "  first present: {}\n",
        report
            .first_present
            .map(fmt_duration)
            .unwrap_or_else(|| "never".into())
    ));
    out.push_str(&format!(
        "  presented frames (flip high-water): {}\n",
        report.flips
    ));
    if report.flip_call_lines > 0 {
        out.push_str(&format!(
            "  sceVideoOutSubmitFlip log lines: {}\n",
            report.flip_call_lines
        ));
    }
    out.push_str(&format!(
        "  epoch rate: overall {} flips/s, windows min/avg/max {}/{}/{} fps\n",
        fmt_opt_fps(report.overall_flips_per_sec),
        fmt_opt_fps(report.window_fps.map(|(min, _, _)| min)),
        fmt_opt_fps(report.window_fps.map(|(_, avg, _)| avg)),
        fmt_opt_fps(report.window_fps.map(|(_, _, max)| max)),
    ));
    out.push_str(&format!(
        "  worst stall: {}\n",
        report
            .worst_stall
            .map(fmt_duration)
            .unwrap_or_else(|| "n/a".into())
    ));
    out.push_str(&format!("  deadlock warnings: {}\n", report.deadlock_count));
    out.push_str(&format!(
        "  cpu (process tree): avg {} peak {} (% of one core)\n",
        fmt_opt_fps(report.avg_cpu_pct),
        fmt_opt_fps(report.peak_cpu_pct)
    ));
    out.push_str(&format!(
        "  peak memory (process tree): {}\n",
        report
            .peak_memory_bytes
            .map(|bytes| format!("{:.0} MiB", bytes as f64 / 1_048_576.0))
            .unwrap_or_else(|| "n/a".into())
    ));
    out
}

pub(crate) fn render_failure(
    failure: &SoakFailure,
    report: &SoakReport,
    tail: &VecDeque<(Duration, String)>,
) -> String {
    let mut out = String::from("SOAK FAILED: ");
    match failure {
        SoakFailure::Deadlock(warning) => {
            out.push_str(&format!(
                "deadlock warning at {}: mutex {} held by thread {} ({}), waiter {}\n  line: {}\n",
                fmt_duration(warning.elapsed),
                warning.mutex,
                warning.owner,
                warning.owner_name,
                warning.waiter_name,
                warning.line
            ));
        }
        SoakFailure::NeverPresented { waited } => {
            out.push_str(&format!(
                "no frame-epoch advance within the {} boot budget\n",
                fmt_duration(*waited)
            ));
        }
        SoakFailure::Stalled { last_advance, gap } => {
            out.push_str(&format!(
                "frame epoch frozen for {} (window {} -> {}, no flip progress since)\n",
                fmt_duration(*gap),
                fmt_duration(*last_advance),
                fmt_duration(*last_advance + *gap)
            ));
        }
        SoakFailure::ProcessExited { exit_code, at } => {
            out.push_str(&format!(
                "process exited before the deadline at {} (exit code {})\n",
                fmt_duration(*at),
                exit_code
                    .map(|code| code.to_string())
                    .unwrap_or_else(|| "unknown/killed".into())
            ));
        }
    }
    out.push('\n');
    out.push_str(&render_report(report));
    if !tail.is_empty() {
        out.push_str(&format!("\nlog tail (last {} line(s)):\n", tail.len()));
        for (elapsed, line) in tail {
            out.push_str(&format!("  [{}] {}\n", fmt_duration(*elapsed), line));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// pure-ish: argument handling
// ---------------------------------------------------------------------------

pub(crate) fn parse_positive_f64(args: &[String], name: &str, default: f64) -> Result<f64> {
    let Some(value) = option(args, name) else {
        return Ok(default);
    };
    let parsed = value
        .parse::<f64>()
        .with_context(|| format!("{name} must be a number, got '{value}'"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        bail!("{name} must be positive, got '{value}'");
    }
    Ok(parsed)
}

/// Pick the soak target: `--game` matches id exactly, then id/title
/// case-insensitive substring (must be unique). Without `--game`, a single
/// registered game or a unique `minecraft`-tagged one is unambiguous.
pub(crate) fn select_soak_game<'a>(
    games: &'a [GameRecord],
    wanted: Option<&str>,
) -> Result<&'a GameRecord> {
    let ids = || {
        games
            .iter()
            .map(|game| format!("{} ({})", game.id, game.title))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if let Some(wanted) = wanted {
        if let Some(game) = games
            .iter()
            .find(|game| game.id.eq_ignore_ascii_case(wanted))
        {
            return Ok(game);
        }
        let needle = wanted.to_ascii_lowercase();
        let matches: Vec<&GameRecord> = games
            .iter()
            .filter(|game| {
                game.id.to_ascii_lowercase().contains(&needle)
                    || game.title.to_ascii_lowercase().contains(&needle)
            })
            .collect();
        return match matches.as_slice() {
            [] => bail!("--game '{wanted}' matches nothing; registered: {}", ids()),
            [only] => Ok(only),
            many => bail!(
                "--game '{wanted}' is ambiguous ({}); use an exact id",
                many.iter()
                    .map(|game| game.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
    }
    if let [only] = games {
        return Ok(only);
    }
    let minecraft: Vec<&GameRecord> = games
        .iter()
        .filter(|game| game.tags.iter().any(|tag| tag == "minecraft"))
        .collect();
    match minecraft.as_slice() {
        [only] => Ok(only),
        _ => bail!("--game is required (registered: {})", ids()),
    }
}

/// Resolve `--input`: `none`/absent → no synthetic input; otherwise an
/// existing file's contents or the literal value, validated as a
/// `raeen-input` replay spec. Returns the spec plus its event count.
pub(crate) fn resolve_input(arg: Option<&str>) -> Result<Option<(String, usize)>> {
    let Some(arg) = arg else { return Ok(None) };
    if arg.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    let path = Path::new(arg);
    let spec = if path.is_file() {
        fs::read_to_string(path)
            .with_context(|| format!("read input script {}", path.display()))?
            .trim()
            .to_string()
    } else {
        arg.to_string()
    };
    let script = raeen_input::InputScript::parse(&spec)
        .map_err(|error| anyhow::anyhow!("invalid --input script: {error}"))?;
    Ok(Some((spec, script.len())))
}

// ---------------------------------------------------------------------------
// impure: log cursor + resource sampler + driver
// ---------------------------------------------------------------------------

/// Incremental reader over a log file another process is appending to.
struct LogCursor {
    path: PathBuf,
    offset: u64,
    assembler: LineAssembler,
}

impl LogCursor {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            offset: 0,
            assembler: LineAssembler::default(),
        }
    }

    fn poll(&mut self) -> Vec<String> {
        let Ok(mut file) = File::open(&self.path) else {
            return Vec::new();
        };
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return Vec::new();
        }
        let mut chunk = Vec::new();
        let Ok(read) = file.read_to_end(&mut chunk) else {
            return Vec::new();
        };
        self.offset += read as u64;
        self.assembler.feed(&chunk)
    }

    fn flush(&mut self) -> Option<String> {
        self.assembler.flush()
    }
}

/// Process-tree CPU/memory sampler over sysinfo. The first sample after
/// process start reports 0% CPU by construction and is skipped by the driver.
struct TreeSampler {
    sys: sysinfo::System,
}

impl TreeSampler {
    fn new() -> Self {
        Self {
            sys: sysinfo::System::new(),
        }
    }

    /// `(cpu_pct_sum, memory_bytes_sum)` over the tree, `None` once the root
    /// process is gone.
    fn sample(&mut self, root: u32) -> Option<(f64, u64)> {
        self.sys
            .refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        self.sys.process(sysinfo::Pid::from_u32(root))?;
        let parents: BTreeMap<u32, Option<u32>> = self
            .sys
            .processes()
            .iter()
            .map(|(pid, process)| (pid.as_u32(), process.parent().map(|p| p.as_u32())))
            .collect();
        let tree = process_tree(root, &parents);
        let mut cpu = 0.0_f64;
        let mut memory = 0_u64;
        for pid in tree {
            if let Some(process) = self.sys.process(sysinfo::Pid::from_u32(pid)) {
                cpu += f64::from(process.cpu_usage());
                memory = memory.saturating_add(process.memory());
            }
        }
        Some((cpu, memory))
    }
}

fn push_tail(tail: &mut VecDeque<(Duration, String)>, elapsed: Duration, line: &str) {
    if tail.len() == TAIL_LINES {
        tail.pop_front();
    }
    let clean = strip_ansi(line);
    tail.push_back((elapsed, clean.chars().take(TAIL_LINE_CAP).collect()));
}

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
    let minutes = parse_positive_f64(args, "--minutes", DEFAULT_MINUTES)?;
    let stall_limit = Duration::from_secs_f64(parse_positive_f64(
        args,
        "--stall-secs",
        DEFAULT_STALL_SECS,
    )?);
    let boot_limit =
        Duration::from_secs_f64(parse_positive_f64(args, "--boot-secs", DEFAULT_BOOT_SECS)?);
    let deadline = Duration::from_secs_f64(minutes * 60.0);
    let wanted = option(args, "--game");
    let game = select_soak_game(&registry.games, wanted.as_deref())?;
    let local_path = game
        .local_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("registry {} has no local executable path", game.id))?;
    let input_arg = option(args, "--input");
    let input = resolve_input(input_arg.as_deref())?;
    let input_description = match &input {
        Some((_, events)) => format!("scripted ({events} event(s) via RAEEN_INPUT_SCRIPT)"),
        None => "none (boot/idle liveness only — no interaction coverage)".to_string(),
    };

    let run_id = format!("soak-{}", now_ms());
    let out_dir =
        PathBuf::from(option(args, "--output-dir").unwrap_or_else(|| DEFAULT_OUTPUT_DIR.into()))
            .join(&run_id);
    fs::create_dir_all(&out_dir)?;
    let build_revision =
        git_output(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|_| "unknown".into());
    let stdout_path = out_dir.join("stdout.log");
    let stderr_path = out_dir.join("stderr.log");

    println!(
        "soak: {} ({}) for {} (stall limit {}, boot budget {}, input: {})",
        game.title,
        game.id,
        fmt_duration(deadline),
        fmt_duration(stall_limit),
        fmt_duration(boot_limit),
        input_description
    );
    println!("soak: logs under {}", out_dir.display());

    let mut command = Command::new(&exe);
    command
        .arg("--run-eboot")
        .arg(local_path)
        // The 32-present WORKER TIMING window is the soak's primary epoch
        // signal (about twice a second at 60 FPS) as well as its FPS source.
        .env("RAEEN_TIME_WORKER", "1")
        .env("RAEEN_COMPAT_RUN_ID", &run_id)
        // Production default, pinned so the soak never depends on the host
        // environment. Unlike the baseline's max-fps profile the soak keeps
        // real vblank pacing: it must observe the run a player would get.
        .env("RAEEN_ASYNC_FLIP", "1")
        .env_remove("RAEEN_CALL_STATS")
        .stdout(Stdio::from(File::create(&stdout_path)?))
        .stderr(Stdio::from(File::create(&stderr_path)?));
    if let Some((spec, _)) = &input {
        // RAEEN_RUNNER_CHILD starts the runner's input thread, where a
        // scripted state wins over Shell IPC and native pads (gui main.rs).
        command
            .env("RAEEN_RUNNER_CHILD", "1")
            .env("RAEEN_INPUT_SCRIPT", spec);
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("launch {}", game.title))?;
    let pid = child.id();
    let started = Instant::now();

    let mut tracker = SoakTracker::new(stall_limit, boot_limit);
    let mut resources = ResourceStats::default();
    let mut sampler = TreeSampler::new();
    let mut cursors = [LogCursor::new(stdout_path), LogCursor::new(stderr_path)];
    let mut tail: VecDeque<(Duration, String)> = VecDeque::with_capacity(TAIL_LINES);
    let mut verdict: Option<SoakFailure> = None;
    let mut polls: u32 = 0;

    loop {
        let elapsed = started.elapsed();
        if elapsed >= deadline {
            break;
        }
        for cursor in &mut cursors {
            for line in cursor.poll() {
                tracker.observe_line(elapsed, &line);
                push_tail(&mut tail, elapsed, &line);
            }
        }
        if let Some(status) = child.try_wait()? {
            verdict = Some(SoakFailure::ProcessExited {
                exit_code: status.code(),
                at: elapsed,
            });
            break;
        }
        if let Some(failure) = tracker.check(elapsed) {
            verdict = Some(failure);
            break;
        }
        if polls.is_multiple_of(RESOURCE_SAMPLE_EVERY)
            && let Some((cpu, memory)) = sampler.sample(pid)
        {
            // The very first sysinfo sample always reports 0% CPU; keep the
            // memory reading but do not let the zero deflate the average.
            if polls == 0 {
                resources.record_memory(memory);
            } else {
                resources.record(cpu, memory);
            }
        }
        polls += 1;
        std::thread::sleep(POLL.min(deadline.saturating_sub(started.elapsed())));
    }

    terminate_process_tree(&mut child);
    let final_elapsed = started.elapsed();
    // Final drain (including unterminated last lines) so a crash message or a
    // late deadlock warning still reaches the tracker and the tail.
    for cursor in &mut cursors {
        for line in cursor.poll() {
            tracker.observe_line(final_elapsed, &line);
            push_tail(&mut tail, final_elapsed, &line);
        }
        if let Some(line) = cursor.flush() {
            tracker.observe_line(final_elapsed, &line);
            push_tail(&mut tail, final_elapsed, &line);
        }
    }
    if verdict.is_none() {
        // Catches a deadline-straddling freeze and any deadlock that only
        // surfaced in the final drain.
        verdict = tracker.check(final_elapsed);
    }

    let report = build_report(
        &tracker,
        &resources,
        &RunMeta {
            title: &game.title,
            game_id: &game.id,
            build_revision: &build_revision,
            input_description: &input_description,
            planned: deadline,
        },
        final_elapsed,
    );
    let report_path = out_dir.join("report.txt");
    match verdict {
        Some(failure) => {
            let rendered = render_failure(&failure, &report, &tail);
            fs::write(&report_path, &rendered)?;
            print!("{rendered}");
            println!("report: {}", report_path.display());
            bail!("soak failed: {failure:?}");
        }
        None => {
            let rendered = render_report(&report);
            fs::write(&report_path, &rendered)?;
            print!("{rendered}");
            println!(
                "SOAK PASSED: {} survived {} with no frozen window > {} and no deadlock warning",
                game.title,
                fmt_duration(final_elapsed),
                fmt_duration(stall_limit)
            );
            println!("report: {}", report_path.display());
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(value: u64) -> Duration {
        Duration::from_secs(value)
    }

    fn tracker() -> SoakTracker {
        SoakTracker::new(secs(10), secs(180))
    }

    fn worker_timing(flips: u64, frame_ms: f64) -> String {
        format!(
            "2026-07-28T00:00:00Z WARN raeen_gpu::agc_exec: flips={flips} submits=40 \
             window_ms=500 idle_pct=20 busy_pct=80 submit_pct=30 flush_pct=50 \
             frame_ms={frame_ms} worker_ms=12.0 WORKER TIMING: idle=waiting-on-guest"
        )
    }

    const DEADLOCK_LINE: &str = "2026-07-28T00:01:00Z WARN raeen_hle::pthread_sync: \
         mutex=0x1019a1d48c0 waiter=42 waiter_name=MAIN owner=7 owner_name=Streaming \
         ty=1 recursion=1 scePthreadMutexLock stuck >3s — deadlock; naming the holder";

    // -- line assembly -------------------------------------------------------

    #[test]
    fn assembler_holds_partial_lines_across_chunks() {
        let mut assembler = LineAssembler::default();
        assert!(assembler.feed(b"first ha").is_empty());
        assert_eq!(
            assembler.feed(b"lf\r\nsecond\nthird "),
            vec!["first half".to_string(), "second".to_string()]
        );
        assert_eq!(assembler.feed(b"part\n"), vec!["third part".to_string()]);
        assert!(assembler.flush().is_none());
    }

    #[test]
    fn assembler_flush_returns_the_unterminated_remainder_once() {
        let mut assembler = LineAssembler::default();
        assert!(assembler.feed(b"dying words without newline").is_empty());
        assert_eq!(
            assembler.flush().as_deref(),
            Some("dying words without newline")
        );
        assert!(assembler.flush().is_none());
    }

    #[test]
    fn assembler_is_lossy_on_invalid_utf8_not_panicking() {
        let mut assembler = LineAssembler::default();
        let lines = assembler.feed(b"ok \xff\xfe bytes\n");
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("ok "));
    }

    // -- epoch advancement ----------------------------------------------------

    #[test]
    fn worker_timing_flip_high_water_advances_the_epoch() {
        let mut t = tracker();
        t.observe_line(secs(30), &worker_timing(32, 16.0));
        assert_eq!(t.first_advance, Some(secs(30)));
        t.observe_line(secs(31), &worker_timing(64, 15.5));
        assert_eq!(t.last_advance, Some(secs(31)));
        assert_eq!(t.flip_high_water, 64);
        assert_eq!(t.frame_ms_samples, vec![16.0, 15.5]);
    }

    #[test]
    fn repeated_or_lower_flip_counts_are_not_progress() {
        let mut t = tracker();
        t.observe_line(secs(30), &worker_timing(64, 16.0));
        // A stale/duplicate count and a small per-submission `flips=` field
        // (the "captured AGC submission" debug line) must not reset the clock.
        t.observe_line(secs(40), &worker_timing(64, 16.0));
        t.observe_line(secs(41), "DEBUG captured AGC submission flips=1 draws=9");
        assert_eq!(t.last_advance, Some(secs(30)));
    }

    #[test]
    fn total_flips_and_videoout_lines_also_advance() {
        let mut t = tracker();
        t.observe_line(
            secs(5),
            "INFO AGC submission progress submissions=8 total_draws=100 total_flips=4",
        );
        assert_eq!(t.last_advance, Some(secs(5)));
        t.observe_line(secs(6), "TRACE hle: sceVideoOutSubmitFlip index=1");
        assert_eq!(t.last_advance, Some(secs(6)));
        assert_eq!(t.flip_call_lines, 1);
    }

    #[test]
    fn ansi_colored_flip_tokens_still_count() {
        let mut t = tracker();
        t.observe_line(secs(9), "\u{1b}[32mtotal_flips\u{1b}[0m=19 total_draws=80");
        assert_eq!(t.flip_high_water, 19);
        assert_eq!(t.last_advance, Some(secs(9)));
    }

    // -- stall detection over synthetic timelines ------------------------------

    #[test]
    fn boot_window_uses_the_boot_budget_not_the_stall_limit() {
        let t = tracker();
        assert_eq!(t.check(secs(179)), None, "still booting");
        assert_eq!(
            t.check(secs(180)),
            Some(SoakFailure::NeverPresented { waited: secs(180) })
        );
    }

    #[test]
    fn stall_clock_arms_only_after_first_advance() {
        let mut t = tracker();
        assert_eq!(t.check(secs(60)), None, "60s of boot silence is fine");
        t.observe_line(secs(61), &worker_timing(32, 16.0));
        assert_eq!(t.check(secs(71)), None, "exactly 10s is not > limit");
        assert_eq!(
            t.check(secs(72)),
            Some(SoakFailure::Stalled {
                last_advance: secs(61),
                gap: secs(11)
            })
        );
    }

    #[test]
    fn steady_timeline_never_reports_a_stall() {
        let mut t = tracker();
        for i in 0..100_u64 {
            t.observe_line(secs(30 + i), &worker_timing(32 * (i + 1), 16.0));
            assert_eq!(t.check(secs(30 + i)), None);
        }
        assert_eq!(t.worst_stall(secs(129)), Some(secs(1)));
    }

    #[test]
    fn worst_stall_includes_the_open_gap_at_the_deadline() {
        let mut t = tracker();
        t.observe_line(secs(30), &worker_timing(32, 16.0));
        t.observe_line(secs(35), &worker_timing(64, 16.0));
        t.observe_line(secs(36), &worker_timing(96, 16.0));
        // Closed worst gap is 5s; the run then goes quiet until 44s.
        assert_eq!(t.worst_stall(secs(44)), Some(secs(8)));
        assert_eq!(t.worst_stall(secs(38)), Some(secs(5)));
    }

    #[test]
    fn no_advance_ever_means_no_worst_stall() {
        assert_eq!(tracker().worst_stall(secs(500)), None);
    }

    // -- deadlock parsing -------------------------------------------------------

    #[test]
    fn deadlock_warning_fails_immediately_even_while_flips_advance() {
        let mut t = tracker();
        t.observe_line(secs(30), &worker_timing(32, 16.0));
        t.observe_line(secs(31), DEADLOCK_LINE);
        t.observe_line(secs(32), &worker_timing(64, 16.0));
        match t.check(secs(32)) {
            Some(SoakFailure::Deadlock(warning)) => {
                assert_eq!(warning.mutex, "0x1019a1d48c0");
                assert_eq!(warning.owner, "7");
                assert_eq!(warning.owner_name, "Streaming");
                assert_eq!(warning.waiter_name, "MAIN");
                assert_eq!(warning.elapsed, secs(31));
            }
            other => panic!("expected deadlock failure, got {other:?}"),
        }
    }

    #[test]
    fn deadlock_marker_is_found_through_ansi_and_fields_default_when_absent() {
        let colored = "\u{1b}[33mWARN\u{1b}[0m \u{1b}[3mmutex\u{1b}[0m=0xabc \
             scePthreadMutexLock stuck >3s — deadlock; naming the holder";
        let mut t = tracker();
        t.observe_line(secs(1), colored);
        assert_eq!(t.deadlocks.len(), 1);
        assert_eq!(t.deadlocks[0].mutex, "0xabc");
        assert_eq!(t.deadlocks[0].owner, "<unknown>");
    }

    #[test]
    fn cond_trace_and_ordinary_warns_are_not_deadlocks() {
        let mut t = tracker();
        t.observe_line(
            secs(1),
            "WARN TRACE_COND: waiting >3s — this cond has not been signalled cond=0x1 waiter=2",
        );
        t.observe_line(secs(2), "WARN shader: opcode not supported");
        assert!(t.deadlocks.is_empty());
    }

    // -- process tree -------------------------------------------------------------

    #[test]
    fn process_tree_follows_parent_chains_and_survives_cycles() {
        let parents: BTreeMap<u32, Option<u32>> = [
            (100, None),      // root
            (200, Some(100)), // child
            (300, Some(200)), // grandchild
            (400, Some(999)), // unrelated (parent unknown)
            (500, Some(600)), // cycle pair
            (600, Some(500)),
        ]
        .into_iter()
        .collect();
        assert_eq!(process_tree(100, &parents), BTreeSet::from([100, 200, 300]));
        assert_eq!(process_tree(700, &parents), BTreeSet::from([700]));
    }

    // -- resource stats -------------------------------------------------------------

    #[test]
    fn resource_stats_track_peak_and_average() {
        let mut stats = ResourceStats::default();
        assert_eq!(stats.avg_cpu_pct(), None);
        // First-sweep memory reading must not count as a 0% CPU sample.
        stats.record_memory(1_000);
        assert_eq!(stats.avg_cpu_pct(), None);
        stats.record(100.0, 1_000);
        stats.record(300.0, 5_000);
        stats.record(200.0, 2_000);
        assert_eq!(stats.avg_cpu_pct(), Some(200.0));
        assert_eq!(stats.peak_cpu_pct, 300.0);
        assert_eq!(stats.peak_memory_bytes, 5_000);
    }

    // -- report --------------------------------------------------------------------

    fn sample_report(t: &SoakTracker, actual: Duration) -> SoakReport {
        build_report(
            t,
            &{
                let mut stats = ResourceStats::default();
                stats.record(250.0, 3 * 1_048_576);
                stats
            },
            &RunMeta {
                title: "Minecraft",
                game_id: "PPSA00000",
                build_revision: "abc123",
                input_description: "none (boot/idle liveness only — no interaction coverage)",
                planned: secs(1800),
            },
            actual,
        )
    }

    #[test]
    fn report_computes_rates_from_windows_and_high_water() {
        let mut t = tracker();
        t.observe_line(secs(60), &worker_timing(32, 20.0)); // 50 fps window
        t.observe_line(secs(61), &worker_timing(64, 10.0)); // 100 fps window
        t.observe_line(secs(62), &worker_timing(96, 16.0)); // 62.5 fps window
        let report = sample_report(&t, secs(70));
        assert_eq!(report.flips, 96);
        assert_eq!(report.first_present, Some(secs(60)));
        // 96 flips over the 10s from first present to the end of the run.
        assert_eq!(report.overall_flips_per_sec, Some(9.6));
        let (min, avg, max) = report.window_fps.expect("windows measured");
        assert_eq!(min, 50.0);
        assert_eq!(max, 100.0);
        assert!((avg - 70.833).abs() < 0.01, "avg was {avg}");
        assert_eq!(report.worst_stall, Some(secs(8)));
        assert_eq!(report.peak_memory_bytes, Some(3 * 1_048_576));
    }

    #[test]
    fn success_report_renders_every_headline_number() {
        let mut t = tracker();
        t.observe_line(secs(60), &worker_timing(32, 20.0));
        t.observe_line(secs(61), &worker_timing(64, 10.0));
        let rendered = render_report(&sample_report(&t, secs(70)));
        assert!(rendered.contains("Minecraft (PPSA00000) build abc123"));
        assert!(rendered.contains("duration: 1m10.0s of 30m00.0s planned"));
        assert!(rendered.contains("presented frames (flip high-water): 64"));
        assert!(rendered.contains("windows min/avg/max 50.0/75.0/100.0 fps"));
        assert!(rendered.contains("worst stall: 9.0s"));
        assert!(rendered.contains("deadlock warnings: 0"));
        assert!(rendered.contains("peak memory (process tree): 3 MiB"));
        assert!(rendered.contains("avg 250.0 peak 250.0"));
        assert!(rendered.contains("input: none"));
    }

    #[test]
    fn report_without_any_presents_stays_honest() {
        let t = tracker();
        let report = sample_report(&t, secs(200));
        assert_eq!(report.flips, 0);
        assert_eq!(report.overall_flips_per_sec, None);
        assert_eq!(report.window_fps, None);
        assert_eq!(report.worst_stall, None);
        let rendered = render_report(&report);
        assert!(rendered.contains("first present: never"));
        assert!(rendered.contains("overall n/a flips/s"));
        assert!(rendered.contains("worst stall: n/a"));
    }

    #[test]
    fn failure_rendering_names_the_frozen_window_and_includes_the_tail() {
        let mut t = tracker();
        t.observe_line(secs(90), &worker_timing(32, 16.0));
        let failure = SoakFailure::Stalled {
            last_advance: secs(90),
            gap: secs(12),
        };
        let mut tail = VecDeque::new();
        tail.push_back((secs(101), "last words".to_string()));
        let rendered = render_failure(&failure, &sample_report(&t, secs(102)), &tail);
        assert!(rendered.starts_with("SOAK FAILED: frame epoch frozen for 12.0s"));
        assert!(rendered.contains("window 1m30.0s -> 1m42.0s"));
        assert!(rendered.contains("log tail (last 1 line(s)):"));
        assert!(rendered.contains("[1m41.0s] last words"));
    }

    #[test]
    fn deadlock_failure_rendering_names_mutex_and_holder() {
        let mut t = tracker();
        t.observe_line(secs(31), DEADLOCK_LINE);
        let Some(SoakFailure::Deadlock(warning)) = t.check(secs(31)) else {
            panic!("deadlock expected");
        };
        let rendered = render_failure(
            &SoakFailure::Deadlock(warning),
            &sample_report(&t, secs(31)),
            &VecDeque::new(),
        );
        assert!(rendered.contains("mutex 0x1019a1d48c0 held by thread 7 (Streaming)"));
        assert!(rendered.contains("deadlock warnings: 1"));
    }

    #[test]
    fn process_exit_rendering_reports_the_exit_code() {
        let t = tracker();
        let rendered = render_failure(
            &SoakFailure::ProcessExited {
                exit_code: Some(3),
                at: secs(45),
            },
            &sample_report(&t, secs(45)),
            &VecDeque::new(),
        );
        assert!(rendered.contains("process exited before the deadline at 45.0s (exit code 3)"));
    }

    #[test]
    fn durations_format_as_minutes_and_seconds() {
        assert_eq!(fmt_duration(Duration::from_secs_f64(9.96)), "10.0s");
        assert_eq!(fmt_duration(secs(60)), "1m00.0s");
        assert_eq!(fmt_duration(Duration::from_secs_f64(1912.34)), "31m52.3s");
    }

    // -- argument handling ------------------------------------------------------------

    fn game(id: &str, title: &str, tags: &[&str]) -> GameRecord {
        GameRecord {
            id: id.into(),
            title: title.into(),
            content_sha1: format!("sha-{id}"),
            executable_bytes: 1,
            relative_hint: "depth-1/eboot.bin".into(),
            local_path: Some(format!("C:/games/{id}/eboot.bin")),
            aliases: Vec::new(),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
        }
    }

    #[test]
    fn game_selection_prefers_exact_id_then_unique_substring() {
        let games = vec![
            game("PPSA01234", "Minecraft", &["minecraft"]),
            game("PPSA05678", "ASTRO's PLAYROOM", &["astro"]),
        ];
        assert_eq!(
            select_soak_game(&games, Some("ppsa01234")).unwrap().title,
            "Minecraft"
        );
        assert_eq!(
            select_soak_game(&games, Some("astro")).unwrap().id,
            "PPSA05678"
        );
        assert!(
            select_soak_game(&games, Some("PPSA0")).is_err(),
            "ambiguous"
        );
        assert!(select_soak_game(&games, Some("gta")).is_err(), "no match");
    }

    #[test]
    fn default_game_is_the_only_entry_or_the_unique_minecraft_tag() {
        let solo = vec![game("PPSA9", "Solo", &[])];
        assert_eq!(select_soak_game(&solo, None).unwrap().id, "PPSA9");
        let tagged = vec![
            game("PPSA01234", "Minecraft", &["minecraft"]),
            game("PPSA05678", "ASTRO's PLAYROOM", &["astro"]),
        ];
        assert_eq!(select_soak_game(&tagged, None).unwrap().title, "Minecraft");
        let ambiguous = vec![game("A", "Game A", &[]), game("B", "Game B", &[])];
        assert!(select_soak_game(&ambiguous, None).is_err());
    }

    #[test]
    fn input_none_or_absent_disables_scripting() {
        assert_eq!(resolve_input(None).unwrap(), None);
        assert_eq!(resolve_input(Some("none")).unwrap(), None);
        assert_eq!(resolve_input(Some("NONE")).unwrap(), None);
    }

    #[test]
    fn inline_input_specs_are_validated_by_the_real_parser() {
        let (spec, events) = resolve_input(Some("0:neutral;5000:cross;5250:neutral"))
            .unwrap()
            .expect("valid spec");
        assert_eq!(spec, "0:neutral;5000:cross;5250:neutral");
        assert_eq!(events, 3);
        let error = resolve_input(Some("0:warp_drive")).unwrap_err().to_string();
        assert!(error.contains("invalid --input script"), "{error}");
    }

    #[test]
    fn input_script_files_are_read_and_validated() {
        let dir = std::env::temp_dir().join(format!("raeen-soak-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("walk.txt");
        fs::write(&path, "0:neutral;1000:ls_up\n").unwrap();
        let (spec, events) = resolve_input(Some(path.to_str().unwrap()))
            .unwrap()
            .expect("file spec");
        assert_eq!(spec, "0:neutral;1000:ls_up");
        assert_eq!(events, 2);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn numeric_flags_reject_nonpositive_values_and_default_when_absent() {
        let args = |value: &str| vec!["--minutes".to_string(), value.to_string()];
        assert_eq!(parse_positive_f64(&[], "--minutes", 30.0).unwrap(), 30.0);
        assert_eq!(
            parse_positive_f64(&args("0.5"), "--minutes", 30.0).unwrap(),
            0.5
        );
        assert!(parse_positive_f64(&args("0"), "--minutes", 30.0).is_err());
        assert!(parse_positive_f64(&args("-3"), "--minutes", 30.0).is_err());
        assert!(parse_positive_f64(&args("soon"), "--minutes", 30.0).is_err());
    }
}
