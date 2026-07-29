//! Frame-path progress counters: what a title actually did on the way to a
//! pixel, and where the chain stopped.
//!
//! Every other diagnosis this project has made started from an error line. A
//! title that runs cleanly, presents nothing, and logs no error defeats that
//! entirely: the log tail is indistinguishable from a healthy run that simply
//! has verbose logging off. Four measured titles fail exactly this way
//! (`docs/silent-zero-frame-cluster.md`).
//!
//! The frame path is a strict prefix chain — a handle, then buffers, then
//! submits, then draws, then a published frame. Naming the **last stage
//! reached** converts "nothing logged" into a bounded question: everything
//! before that stage works, and the very next one is the suspect. That is a
//! per-title-independent answer, so this stays useful long after the current
//! cluster is fixed.
//!
//! # Cost
//!
//! Counting is **on by default**: a [`record`] is one relaxed load of a process
//! `AtomicBool`, one `fetch_add` on a 64-byte-aligned counter, and one relaxed
//! load — no allocation, and the clock is read at most once per stage for the
//! whole process. The counters were off by default until the diagnostics work
//! of 2026-07-29, which meant the one number that answers "where did it stop"
//! was missing from every report a user could produce. What `RAEEN_FRAME_PATH`
//! still controls is the *periodic reporter thread*, not the counting.
//!
//! Each counter is padded to its own cache line because the writers genuinely
//! sit on different cores: the GPU worker records
//! `DcbSubmitted`/`Draw`/`Dispatch`/`FramePublished` while guest threads record
//! `VideoOutOpen`/`BuffersRegistered`/`FlipRateSet`/`FlipSubmitted`. Sharing one
//! line would put a cross-core read-modify-write in the 75 FPS profile.
//!
//! # Why periodic, not only at exit
//!
//! The compatibility harness resolves a stalled title by killing it
//! (`child.kill()` in `xtask`), so an at-exit hook would never run for exactly
//! the titles this exists to diagnose. When enabled, a reporter thread logs the
//! summary on an interval, so the last line before a hard kill still carries
//! the full chain state.
//!
//! # Two halves of one chain
//!
//! [`Stage`] covers the *presentation* half, which begins at
//! `sceVideoOutOpen`. Everything before that — loading the process, linking its
//! dependencies, reaching `_start`, running initializers, the first HLE call,
//! the first guest thread — used to collapse into the single word `nothing`,
//! which described four completely different failures identically. [`Phase`] is
//! that upstream half, so `reached=nothing phase=deps_linked` and
//! `reached=nothing phase=first_guest_thread` are now different findings.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// One rung of the guest's path from "process started" to "pixel presented".
///
/// Declaration order is the causal order the stages must occur in, and
/// [`Stage::ALL`] relies on it: the summary reports the last stage with a
/// non-zero count as the furthest point reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    /// `sceVideoOutOpen` returned a handle.
    VideoOutOpen,
    /// `sceVideoOutRegisterBuffers`/`...Buffers2` accepted a buffer set.
    BuffersRegistered,
    /// `sceVideoOutSetFlipRate` selected a presentation cadence.
    FlipRateSet,
    /// A GPU submission (AGC DCB) reached the command processor.
    DcbSubmitted,
    /// A draw was translated and recorded.
    Draw,
    /// A compute dispatch was translated and recorded.
    Dispatch,
    /// `sceVideoOutSubmitFlip` (or the equivalent flip packet) asked for a
    /// buffer to be shown.
    FlipSubmitted,
    /// A complete frame reached the present path (`publish_frame`).
    FramePublished,
}

impl Stage {
    /// Every stage in causal order.
    pub const ALL: [Stage; 8] = [
        Stage::VideoOutOpen,
        Stage::BuffersRegistered,
        Stage::FlipRateSet,
        Stage::DcbSubmitted,
        Stage::Draw,
        Stage::Dispatch,
        Stage::FlipSubmitted,
        Stage::FramePublished,
    ];

    /// Stable short label used in the summary line and in tooling that greps
    /// for it. These are matched by `xtask`'s report builder, so treat them as
    /// part of the log's contract.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Stage::VideoOutOpen => "videoout_open",
            Stage::BuffersRegistered => "buffers_registered",
            Stage::FlipRateSet => "flip_rate_set",
            Stage::DcbSubmitted => "dcb_submitted",
            Stage::Draw => "draws",
            Stage::Dispatch => "dispatches",
            Stage::FlipSubmitted => "flips_submitted",
            Stage::FramePublished => "frames_published",
        }
    }

    /// Inverse of [`Stage::label`], for readers that parse a summary line or a
    /// report back into the enum (the Shell's status pane, `xtask`).
    #[must_use]
    pub fn from_label(label: &str) -> Option<Stage> {
        Stage::ALL.into_iter().find(|s| s.label() == label)
    }

    const fn index(self) -> usize {
        match self {
            Stage::VideoOutOpen => 0,
            Stage::BuffersRegistered => 1,
            Stage::FlipRateSet => 2,
            Stage::DcbSubmitted => 3,
            Stage::Draw => 4,
            Stage::Dispatch => 5,
            Stage::FlipSubmitted => 6,
            Stage::FramePublished => 7,
        }
    }
}

/// One rung of the guest's path from "process image loaded" to "the guest is
/// running its own threads" — the half of the chain that happens *before*
/// [`Stage::VideoOutOpen`].
///
/// Deliberately six rungs, not seven: nothing in the tree can observe "the
/// guest executed its first instruction" without a trap or a call, so
/// [`Phase::FirstHleCall`] carries that meaning rather than inventing a rung
/// that would always read as unreached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// The process image was composed and mapped.
    ProcessLoaded,
    /// `DT_NEEDED` dependencies were loaded and NID-linked.
    DepsLinked,
    /// Control was handed to the guest entry point (`_start`).
    EntryReached,
    /// A module initializer (`module_start`/`DT_INIT`) returned.
    InitializersRan,
    /// The guest called into HLE for the first time — the earliest proof that
    /// guest code is genuinely executing.
    FirstHleCall,
    /// The guest started a pthread of its own.
    FirstGuestThread,
}

impl Phase {
    /// Every phase in causal order.
    pub const ALL: [Phase; 6] = [
        Phase::ProcessLoaded,
        Phase::DepsLinked,
        Phase::EntryReached,
        Phase::InitializersRan,
        Phase::FirstHleCall,
        Phase::FirstGuestThread,
    ];

    /// Stable short label. Part of the log/report contract, exactly like
    /// [`Stage::label`].
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Phase::ProcessLoaded => "process_loaded",
            Phase::DepsLinked => "deps_linked",
            Phase::EntryReached => "entry_reached",
            Phase::InitializersRan => "initializers_ran",
            Phase::FirstHleCall => "first_hle_call",
            Phase::FirstGuestThread => "first_guest_thread",
        }
    }

    /// Inverse of [`Phase::label`].
    #[must_use]
    pub fn from_label(label: &str) -> Option<Phase> {
        Phase::ALL.into_iter().find(|p| p.label() == label)
    }

    const fn index(self) -> usize {
        match self {
            Phase::ProcessLoaded => 0,
            Phase::DepsLinked => 1,
            Phase::EntryReached => 2,
            Phase::InitializersRan => 3,
            Phase::FirstHleCall => 4,
            Phase::FirstGuestThread => 5,
        }
    }
}

/// Marker used in the summary when the guest never reached any stage. The
/// compat harness turns this into a first-blocker line, so a silent title stops
/// reporting "none logged".
pub const NOTHING_REACHED: &str = "nothing";

const STAGE_COUNT: usize = Stage::ALL.len();
const PHASE_COUNT: usize = Phase::ALL.len();

/// An `AtomicU64` on a cache line of its own. See the module docs on cost: the
/// stage counters are written from the GPU worker and from guest threads
/// concurrently, so packing them would make every record a cross-core RMW.
#[repr(align(64))]
pub struct PaddedU64(AtomicU64);

/// Gate for [`record`]. **On by default** — see the module docs. Kept as a
/// switch so tests and embedders can silence recording entirely.
static ENABLED: AtomicBool = AtomicBool::new(true);

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: PaddedU64 = PaddedU64(AtomicU64::new(0));
static COUNTS: [PaddedU64; STAGE_COUNT] = [ZERO; STAGE_COUNT];

/// "Never reached", for both [`FIRST_MS`] and [`PHASE_FIRST_MS`].
const NOT_REACHED: u64 = u64::MAX;
/// "Reached, but before [`mark_origin`] stamped the epoch, so the time is not
/// knowable". Distinct from [`NOT_REACHED`] so a reader is never handed an
/// invented `0 ms`.
const NO_EPOCH: u64 = u64::MAX - 1;

/// Milliseconds since [`mark_origin`] — which `main` stamps at process start —
/// at each stage's first occurrence. See [`NOT_REACHED`] / [`NO_EPOCH`].
#[allow(clippy::declare_interior_mutable_const)]
const NEVER: PaddedU64 = PaddedU64(AtomicU64::new(NOT_REACHED));
static FIRST_MS: [PaddedU64; STAGE_COUNT] = [NEVER; STAGE_COUNT];
static PHASE_FIRST_MS: [PaddedU64; PHASE_COUNT] = [NEVER; PHASE_COUNT];

static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Stamp the timing epoch. Call once, as early in `main` as possible — the
/// diagnostics timeline (this module and `crate::blockers`) is measured from
/// here, so every artifact in a session shares one origin.
///
/// Idempotent; the first call wins. Until it is called, first-occurrence times
/// read as [`None`] rather than as zero: seeding the epoch lazily at the first
/// recorded event would silently redefine it as "whenever the first thing
/// happened", and would make that first rung always print `@0ms`.
pub fn mark_origin() {
    let _ = ORIGIN.get_or_init(Instant::now);
}

/// Milliseconds since [`mark_origin`], or `None` if the epoch was never
/// stamped.
#[must_use]
pub fn since_origin_ms() -> Option<u64> {
    ORIGIN
        .get()
        .map(|origin| origin.elapsed().as_millis().min(u128::from(u64::MAX)) as u64)
}

/// Whether frame-path recording is on.
#[must_use]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Turn recording off (or back on). Exists for tests and for embedders that
/// want the counters silent; the emulator leaves it at its default `true`.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Note that the guest reached `phase`.
///
/// After the first occurrence this is one relaxed load of a shared-clean cache
/// line and a not-taken branch — no `fetch_add` — which matters because
/// [`Phase::FirstHleCall`] sits on the ~15k-calls/second HLE path.
#[inline]
pub fn record_phase(phase: Phase) {
    let cell = &PHASE_FIRST_MS[phase.index()].0;
    if cell.load(Ordering::Relaxed) != NOT_REACHED {
        return;
    }
    record_phase_slow(cell);
}

#[inline(never)]
fn record_phase_slow(cell: &AtomicU64) {
    let stamp = since_origin_ms().unwrap_or(NO_EPOCH);
    // Racing threads may both observe NOT_REACHED; `fetch_min` keeps the
    // earliest, and both sentinels are larger than any real millisecond value
    // so a real time always wins over NO_EPOCH.
    cell.fetch_min(stamp, Ordering::Relaxed);
}

/// Milliseconds from the epoch to the first occurrence of `phase`.
///
/// `None` covers both "never reached" and "reached before the epoch was
/// stamped"; use [`phase_reached`] to tell whether a phase happened at all.
#[must_use]
pub fn phase_first_ms(phase: Phase) -> Option<u64> {
    match PHASE_FIRST_MS[phase.index()].0.load(Ordering::Relaxed) {
        NOT_REACHED | NO_EPOCH => None,
        value => Some(value),
    }
}

/// Whether `phase` was reached at all, regardless of whether its time is known.
#[must_use]
pub fn phase_reached(phase: Phase) -> bool {
    PHASE_FIRST_MS[phase.index()].0.load(Ordering::Relaxed) != NOT_REACHED
}

/// The furthest phase the guest reached, or `None` if it reached none.
#[must_use]
pub fn furthest_phase() -> Option<Phase> {
    Phase::ALL.into_iter().rev().find(|&p| phase_reached(p))
}

/// Note that the guest reached `stage`.
///
/// Safe to call from any thread, including the GPU worker. A no-op unless
/// [`init_from_env`] turned recording on.
#[inline]
pub fn record(stage: Stage) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    record_slow(stage, 1);
}

/// Note `n` occurrences of `stage` at once.
///
/// Draws arrive as a per-submission batch total; counting the batch as one
/// would make the summary's draw figure meaningless next to `flips_submitted`.
#[inline]
pub fn record_n(stage: Stage, n: u64) {
    if n == 0 || !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    record_slow(stage, n);
}

/// Out-of-line remainder of [`record`], so the disabled path stays a load and a
/// branch with nothing to inline around it.
#[inline(never)]
fn record_slow(stage: Stage, n: u64) {
    let index = stage.index();
    COUNTS[index].0.fetch_add(n, Ordering::Relaxed);
    if FIRST_MS[index].0.load(Ordering::Relaxed) == NOT_REACHED {
        let stamp = since_origin_ms().unwrap_or(NO_EPOCH);
        // Racing threads may both see NOT_REACHED; `fetch_min` keeps the
        // earliest, and a real millisecond always beats the NO_EPOCH sentinel.
        FIRST_MS[index].0.fetch_min(stamp, Ordering::Relaxed);
    }
}

/// How many times `stage` was recorded.
#[must_use]
pub fn count(stage: Stage) -> u64 {
    COUNTS[stage.index()].0.load(Ordering::Relaxed)
}

/// Milliseconds from the epoch to the first occurrence of `stage`.
///
/// `None` means the time is not knowable — either the stage was never reached
/// or it happened before [`mark_origin`]. `count(stage) > 0` independently
/// proves "reached", so the two cases stay distinguishable and no substituted
/// zero exists anywhere.
#[must_use]
pub fn first_ms(stage: Stage) -> Option<u64> {
    match FIRST_MS[stage.index()].0.load(Ordering::Relaxed) {
        NOT_REACHED | NO_EPOCH => None,
        value => Some(value),
    }
}

/// The furthest stage the guest reached, or `None` if it reached none.
#[must_use]
pub fn furthest_stage() -> Option<Stage> {
    Stage::ALL.into_iter().rev().find(|&stage| count(stage) > 0)
}

/// The single-line summary written to the log.
///
/// Shape (stable; `xtask` parses `reached=`):
/// `frame path: reached=<label> | videoout_open=0 buffers_registered=0 ...`
///
/// A never-reached stage prints `0`; a reached stage prints
/// `<count>@<first_ms>ms`, or `<count>@?ms` when it was reached before the
/// epoch was stamped — never an invented `@0ms`.
///
/// `phase=` follows the `reached=` token and never precedes it: `xtask` parses
/// this line by taking the first whitespace token after `"frame path: reached="`.
#[must_use]
pub fn summary() -> String {
    let reached = furthest_stage().map_or(NOTHING_REACHED, Stage::label);
    let phase = furthest_phase().map_or(NOTHING_REACHED, Phase::label);
    let mut line = format!("frame path: reached={reached} phase={phase} |");
    for stage in Stage::ALL {
        let count = count(stage);
        if count == 0 {
            line.push_str(&format!(" {}=0", stage.label()));
        } else {
            match first_ms(stage) {
                Some(ms) => line.push_str(&format!(" {}={count}@{ms}ms", stage.label())),
                None => line.push_str(&format!(" {}={count}@?ms", stage.label())),
            }
        }
    }
    for phase in Phase::ALL {
        if !phase_reached(phase) {
            continue;
        }
        match phase_first_ms(phase) {
            Some(ms) => line.push_str(&format!(" {}=@{ms}ms", phase.label())),
            None => line.push_str(&format!(" {}=@?ms", phase.label())),
        }
    }
    line
}

/// A consistent read of the whole chain: 14 relaxed loads, no allocation
/// beyond the returned value.
///
/// "Consistent" is per-field, not a global snapshot — a live guest may advance
/// between two loads. That is deliberate and harmless: the chain is a strict
/// prefix, so a torn read can only ever understate progress.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Furthest upstream phase reached, if any.
    pub phase: Option<Phase>,
    /// Furthest presentation stage reached, if any.
    pub stage: Option<Stage>,
    /// Per-stage occurrence counts, in [`Stage::ALL`] order.
    pub counts: [u64; STAGE_COUNT],
    /// Per-stage first-occurrence times, in [`Stage::ALL`] order. `None` means
    /// "not reached, or reached before the epoch" — read it against `counts`.
    pub first_ms: [Option<u64>; STAGE_COUNT],
    /// Whether each phase was reached, in [`Phase::ALL`] order.
    pub phase_reached: [bool; PHASE_COUNT],
    /// Per-phase first-occurrence times, in [`Phase::ALL`] order.
    pub phase_first_ms: [Option<u64>; PHASE_COUNT],
}

impl Snapshot {
    /// The `reached=` label this snapshot would print.
    #[must_use]
    pub fn reached_label(&self) -> &'static str {
        self.stage.map_or(NOTHING_REACHED, Stage::label)
    }

    /// The `phase=` label this snapshot would print.
    #[must_use]
    pub fn phase_label(&self) -> &'static str {
        self.phase.map_or(NOTHING_REACHED, Phase::label)
    }

    /// The counter detail the summary line carries after the `|`, without the
    /// leading separator — the form the verdict wording quotes.
    #[must_use]
    pub fn counters(&self) -> String {
        let mut out = String::new();
        for (index, stage) in Stage::ALL.into_iter().enumerate() {
            if index > 0 {
                out.push(' ');
            }
            let count = self.counts[index];
            match (count, self.first_ms[index]) {
                (0, _) => out.push_str(&format!("{}=0", stage.label())),
                (n, Some(ms)) => out.push_str(&format!("{}={n}@{ms}ms", stage.label())),
                (n, None) => out.push_str(&format!("{}={n}@?ms", stage.label())),
            }
        }
        out
    }
}

/// Read the whole chain at once.
#[must_use]
pub fn snapshot() -> Snapshot {
    Snapshot {
        phase: furthest_phase(),
        stage: furthest_stage(),
        counts: Stage::ALL.map(count),
        first_ms: Stage::ALL.map(first_ms),
        phase_reached: Phase::ALL.map(phase_reached),
        phase_first_ms: Phase::ALL.map(phase_first_ms),
    }
}

/// The finding a silent zero-frame run reports, in one sentence.
///
/// Lives here, not in `xtask`, so the compatibility harness and the in-process
/// session report cannot drift into describing the same run differently — they
/// call this same function.
#[must_use]
pub fn silent_zero_frame_verdict(reached_label: &str, phase_label: &str, counters: &str) -> String {
    if reached_label == NOTHING_REACHED {
        // The upstream half is what makes this useful: "reached no stage" was
        // one word for four different failures until `phase` existed.
        let progress = if phase_label == NOTHING_REACHED {
            "and no load-path phase either — it never got as far as a composed process image"
                .to_string()
        } else {
            format!("its load path stopped after '{phase_label}'")
        };
        format!(
            "silent zero-frame run: the guest reached NO frame-path stage — it never opened a \
             video-out handle; {progress}. Counters: {counters}"
        )
    } else {
        format!(
            "silent zero-frame run: the frame path stopped after '{reached_label}' and never \
             published a frame. Counters: {counters}"
        )
    }
}

/// Parse the reporter interval from `RAEEN_FRAME_PATH`.
///
/// Any set value enables recording. A bare integer is the summary interval in
/// seconds; `1`, `true`, `yes`, `on` and the empty string mean "enabled at the
/// default interval". `0` enables recording with no periodic reporter, which
/// suits an in-process caller that logs [`summary`] itself.
#[must_use]
pub fn parse_interval(value: &str) -> Option<Duration> {
    const DEFAULT_SECS: u64 = 10;
    let trimmed = value.trim();
    if trimmed.is_empty() || matches!(trimmed, "1" | "true" | "yes" | "on") {
        return Some(Duration::from_secs(DEFAULT_SECS));
    }
    match trimmed.parse::<u64>() {
        Ok(0) => Some(Duration::ZERO),
        Ok(secs) => Some(Duration::from_secs(secs)),
        Err(_) => Some(Duration::from_secs(DEFAULT_SECS)),
    }
}

/// Default periodic-reporter interval when `RAEEN_FRAME_PATH` is unset.
///
/// The reporter is what makes a hard-killed stalled title leave its chain state
/// behind, and that is the failure class this module exists for — so it runs by
/// default. One `info!` line per minute is not a log-volume concern next to a
/// 64 MiB cap; set `RAEEN_FRAME_PATH=0` to silence it (counting continues).
pub const DEFAULT_REPORT_SECS: u64 = 60;

/// Start the periodic frame-path reporter.
///
/// Counting itself is always on (see the module docs) — this only controls the
/// periodic `frame path: reached=…` log line. `RAEEN_FRAME_PATH` chooses the
/// interval in seconds; `0` disables the reporter; unset means
/// [`DEFAULT_REPORT_SECS`].
///
/// Idempotent: a second call is ignored, so wiring this into both a library
/// entry point and the GUI is harmless. Returns whether the reporter is running.
pub fn init_from_env() -> bool {
    static STARTED: AtomicBool = AtomicBool::new(false);

    let interval = match std::env::var("RAEEN_FRAME_PATH") {
        Ok(raw) => parse_interval(&raw).unwrap_or(Duration::from_secs(DEFAULT_REPORT_SECS)),
        Err(_) => Duration::from_secs(DEFAULT_REPORT_SECS),
    };
    if interval.is_zero() || STARTED.swap(true, Ordering::Release) {
        return false;
    }
    // The epoch is normally stamped by `main`; stamping here too keeps a
    // library-only embedder from getting `None` timings forever.
    mark_origin();
    std::thread::Builder::new()
        .name("frame-path-report".into())
        .spawn(move || {
            loop {
                std::thread::sleep(interval);
                tracing::info!("{}", summary());
            }
        })
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The counters are process-global, so the assertions below are written to
    // hold regardless of what other tests in this binary recorded. Only
    // `parse_interval`, `summary` shape, and ordering are exercised.

    #[test]
    fn stage_labels_are_unique_and_ordered() {
        let labels: Vec<&str> = Stage::ALL.iter().map(|s| s.label()).collect();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "labels must be unique");
        for (index, stage) in Stage::ALL.into_iter().enumerate() {
            assert_eq!(stage.index(), index, "ALL must match index()");
        }
    }

    #[test]
    fn phase_labels_are_unique_and_ordered() {
        let labels: Vec<&str> = Phase::ALL.iter().map(|p| p.label()).collect();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "labels must be unique");
        for (index, phase) in Phase::ALL.into_iter().enumerate() {
            assert_eq!(phase.index(), index, "ALL must match index()");
            assert_eq!(Phase::from_label(phase.label()), Some(phase));
        }
        for stage in Stage::ALL {
            assert_eq!(Stage::from_label(stage.label()), Some(stage));
            // The two vocabularies must not collide: a reader parsing
            // `reached=<x> phase=<y>` has to be able to tell them apart.
            assert_eq!(Phase::from_label(stage.label()), None, "{stage:?} collides");
        }
        assert_eq!(Stage::from_label("not_a_stage"), None);
        assert_eq!(Phase::from_label("not_a_phase"), None);
    }

    /// `xtask` parses the summary by finding `"frame path: reached="` and
    /// taking the FIRST whitespace token after it. Adding `phase=` must not
    /// move that token, or every compat report's blocker silently regresses to
    /// a phase label.
    #[test]
    fn summary_keeps_the_prefix_xtask_parses_and_names_every_stage() {
        const MARKER: &str = "frame path: reached=";
        let line = summary();
        assert!(line.starts_with(MARKER), "{line}");
        let tail = line.split_once(MARKER).expect("marker present").1;
        let reached = tail.split_whitespace().next().expect("a reached token");
        assert!(
            reached == NOTHING_REACHED || Stage::from_label(reached).is_some(),
            "the first token after the marker must still be a STAGE, got {reached:?}"
        );
        assert!(
            line.contains(" phase="),
            "the phase must be present: {line}"
        );
        for stage in Stage::ALL {
            assert!(line.contains(stage.label()), "{line} is missing {stage:?}");
        }
    }

    /// Recording is on by default now, but a caller can still silence it —
    /// and while silenced no counter may move.
    #[test]
    fn recording_is_on_by_default_and_can_be_silenced() {
        assert!(enabled(), "counting must be on by default");
        set_enabled(false);
        let before: Vec<u64> = Stage::ALL.into_iter().map(count).collect();
        for stage in Stage::ALL {
            record(stage);
        }
        let after: Vec<u64> = Stage::ALL.into_iter().map(count).collect();
        set_enabled(true);
        assert_eq!(before, after, "a silenced recorder must not move a counter");
    }

    /// The whole point of the two sentinels: a stage reached before the epoch
    /// was stamped must NOT claim it happened at 0 ms.
    #[test]
    fn a_time_is_never_invented_for_a_reached_stage() {
        let snap = Snapshot {
            phase: Some(Phase::DepsLinked),
            stage: Some(Stage::DcbSubmitted),
            counts: [1, 2, 0, 91, 0, 0, 0, 0],
            first_ms: [Some(812), Some(840), None, None, None, None, None, None],
            phase_reached: [true, true, false, false, false, false],
            phase_first_ms: [Some(4), None, None, None, None, None],
        };
        let rendered = snap.counters();
        assert!(rendered.contains("videoout_open=1@812ms"), "{rendered}");
        // Reached (count 91) but no time: `?`, not a fabricated zero.
        assert!(rendered.contains("dcb_submitted=91@?ms"), "{rendered}");
        // Never reached: a bare zero with no time at all.
        assert!(rendered.contains("flip_rate_set=0"), "{rendered}");
        assert!(!rendered.contains("@0ms"), "no invented zero: {rendered}");
        assert_eq!(snap.reached_label(), "dcb_submitted");
        assert_eq!(snap.phase_label(), "deps_linked");
    }

    /// The verdict wording is shared with `xtask` so the harness and the
    /// in-process report cannot describe one run two ways.
    #[test]
    fn the_silent_zero_frame_verdict_distinguishes_four_failures() {
        let nothing_at_all = silent_zero_frame_verdict(NOTHING_REACHED, NOTHING_REACHED, "c");
        let linked = silent_zero_frame_verdict(NOTHING_REACHED, "deps_linked", "c");
        let threaded = silent_zero_frame_verdict(NOTHING_REACHED, "first_guest_thread", "c");
        let staged = silent_zero_frame_verdict("dcb_submitted", "first_guest_thread", "c");
        for verdict in [&nothing_at_all, &linked, &threaded, &staged] {
            assert!(verdict.starts_with("silent zero-frame run: "), "{verdict}");
            assert!(verdict.ends_with("Counters: c"), "{verdict}");
        }
        // The four cases that used to be one word are now four sentences.
        assert!(nothing_at_all.contains("never got as far as a composed process image"));
        assert!(linked.contains("stopped after 'deps_linked'"));
        assert!(threaded.contains("stopped after 'first_guest_thread'"));
        assert!(staged.contains("stopped after 'dcb_submitted'"));
        assert_ne!(linked, threaded);
    }

    #[test]
    fn interval_parsing_covers_the_documented_spellings() {
        assert_eq!(parse_interval(""), Some(Duration::from_secs(10)));
        assert_eq!(parse_interval("1"), Some(Duration::from_secs(10)));
        assert_eq!(parse_interval("on"), Some(Duration::from_secs(10)));
        assert_eq!(parse_interval("nonsense"), Some(Duration::from_secs(10)));
        assert_eq!(parse_interval("0"), Some(Duration::ZERO));
        assert_eq!(parse_interval("5"), Some(Duration::from_secs(5)));
        assert_eq!(parse_interval(" 30 "), Some(Duration::from_secs(30)));
    }
}
