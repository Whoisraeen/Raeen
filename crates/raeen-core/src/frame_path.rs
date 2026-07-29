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
//! Disabled (the default) every [`record`] is one relaxed load of a process
//! `AtomicBool` and a not-taken branch: no counter traffic, no clock read, no
//! allocation. Enable with `RAEEN_FRAME_PATH` (see [`init_from_env`]).
//!
//! # Why periodic, not only at exit
//!
//! The compatibility harness resolves a stalled title by killing it
//! (`child.kill()` in `xtask`), so an at-exit hook would never run for exactly
//! the titles this exists to diagnose. When enabled, a reporter thread logs the
//! summary on an interval, so the last line before a hard kill still carries
//! the full chain state.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// One rung of the guest's path from "process started" to "pixel presented".
///
/// Declaration order is the causal order the stages must occur in, and
/// [`Stage::ALL`] relies on it: the summary reports the last stage with a
/// non-zero count as the furthest point reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Marker used in the summary when the guest never reached any stage. The
/// compat harness turns this into a first-blocker line, so a silent title stops
/// reporting "none logged".
pub const NOTHING_REACHED: &str = "nothing";

const STAGE_COUNT: usize = Stage::ALL.len();

/// Gate for [`record`]. False until [`init_from_env`] enables it, which happens
/// once during process startup before any guest code runs.
static ENABLED: AtomicBool = AtomicBool::new(false);

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static COUNTS: [AtomicU64; STAGE_COUNT] = [ZERO; STAGE_COUNT];
/// Milliseconds since [`ENABLED`] was set, at each stage's first occurrence.
/// `u64::MAX` means "never reached" — distinguishable from "reached at 0 ms".
#[allow(clippy::declare_interior_mutable_const)]
const NEVER: AtomicU64 = AtomicU64::new(u64::MAX);
static FIRST_MS: [AtomicU64; STAGE_COUNT] = [NEVER; STAGE_COUNT];

static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Whether frame-path recording is on.
#[must_use]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
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
    COUNTS[index].fetch_add(n, Ordering::Relaxed);
    if FIRST_MS[index].load(Ordering::Relaxed) == u64::MAX {
        let elapsed = ORIGIN.get().map_or(0, |origin| {
            origin.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
        });
        // Racing threads may both see MAX; `fetch_min` keeps the earliest.
        FIRST_MS[index].fetch_min(elapsed, Ordering::Relaxed);
    }
}

/// How many times `stage` was recorded.
#[must_use]
pub fn count(stage: Stage) -> u64 {
    COUNTS[stage.index()].load(Ordering::Relaxed)
}

/// Milliseconds from enablement to the first occurrence of `stage`, or `None`
/// if it was never reached.
#[must_use]
pub fn first_ms(stage: Stage) -> Option<u64> {
    match FIRST_MS[stage.index()].load(Ordering::Relaxed) {
        u64::MAX => None,
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
/// A zero-count stage prints `0`; a reached stage prints `<count>@<first_ms>ms`
/// so the timing of the stall is visible without correlating timestamps.
#[must_use]
pub fn summary() -> String {
    let reached = furthest_stage().map_or(NOTHING_REACHED, Stage::label);
    let mut line = format!("frame path: reached={reached} |");
    for stage in Stage::ALL {
        let count = count(stage);
        match first_ms(stage) {
            Some(ms) => line.push_str(&format!(" {}={count}@{ms}ms", stage.label())),
            None => line.push_str(&format!(" {}=0", stage.label())),
        }
    }
    line
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

/// Turn on frame-path recording if `RAEEN_FRAME_PATH` is set, and start the
/// periodic reporter unless the interval is zero.
///
/// Idempotent: a second call is ignored, so wiring this into both a library
/// entry point and the GUI is harmless. Returns whether recording is on.
pub fn init_from_env() -> bool {
    let Some(raw) = std::env::var("RAEEN_FRAME_PATH").ok() else {
        return false;
    };
    let Some(interval) = parse_interval(&raw) else {
        return false;
    };
    if ENABLED.swap(true, Ordering::Release) {
        return true;
    }
    let _ = ORIGIN.set(Instant::now());
    if !interval.is_zero() {
        std::thread::Builder::new()
            .name("frame-path-report".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(interval);
                    tracing::info!("{}", summary());
                }
            })
            .ok();
    }
    true
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
    fn summary_names_every_stage_and_the_furthest_one() {
        let line = summary();
        assert!(line.starts_with("frame path: reached="), "{line}");
        for stage in Stage::ALL {
            assert!(line.contains(stage.label()), "{line} is missing {stage:?}");
        }
    }

    #[test]
    fn a_disabled_recorder_stays_silent() {
        // Nothing enabled this in the unit-test binary, so recording must not
        // move any counter — the guarantee that the default build pays nothing.
        assert!(!enabled());
        let before: Vec<u64> = Stage::ALL.into_iter().map(count).collect();
        for stage in Stage::ALL {
            record(stage);
        }
        let after: Vec<u64> = Stage::ALL.into_iter().map(count).collect();
        assert_eq!(before, after);
        assert_eq!(furthest_stage(), None);
        assert!(summary().contains(&format!("reached={NOTHING_REACHED}")));
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
