//! A report for **every** session, not only the ones that fault.
//!
//! # Why this exists
//!
//! `crash_report::CrashReport` was written from exactly one place: the `Err`
//! arm of the runner's `execute_process_shared` call. That covers a guest that
//! faults — and misses the largest measured failure class entirely. Four
//! titles in `docs/silent-zero-frame-cluster.md` run cleanly, present nothing,
//! log no error, and are resolved 180 seconds later by `child.kill()`. A killed
//! process runs no `Err` arm, no `Drop`, and no at-exit hook, so those four
//! sessions produced **zero artifacts**. The one question a user has — where did
//! it stop, and why — had no file to answer it.
//!
//! Two decisions follow from that, and neither is optional:
//!
//! * **Write from t≈0, synchronously, before the heartbeat thread exists.** A
//!   `__fastfail` (`0xC0000409`) termination or a death at 2 seconds must still
//!   leave a file. Write-then-sleep, never sleep-then-write.
//! * **Overwrite one pair in place.** A fixed stem per session means a
//!   heartbeat every ten seconds costs two files total, not one pair per tick,
//!   and gives the parent a known path to finalize when it kills the child.
//!
//! This is the same argument `raeen_core::frame_path` already makes for its
//! periodic reporter, applied to the artifact a human actually opens.
//!
//! # Honesty
//!
//! A heartbeat report is marked `provisional` and carries the outcome
//! `launching` or `rendering` — never a verdict it cannot have yet. The parent
//! promotes it to `timed_out`/`crashed` when it resolves the session
//! (`crash_report::finalize_report`). Nothing here guesses: a section with
//! nothing in it renders `<none recorded>`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use std::sync::Mutex;

use crate::compat::Stage;
use crate::crash_report::{self, CrashReport};

/// How often the heartbeat rewrites the report, in seconds.
pub const DEFAULT_HEARTBEAT_SECS: u64 = 10;

/// Environment override for the heartbeat interval. `0` writes the initial
/// report and never refreshes it.
pub const HEARTBEAT_SECS_ENV: &str = "RAEEN_SESSION_REPORT_SECS";

/// Seconds without a published frame before the heartbeat samples where the
/// guest threads actually are.
///
/// Deliberately not zero: sampling suspends guest threads briefly, and a title
/// legitimately spends its first seconds loading. Thirty seconds is past every
/// measured healthy first-frame time and well inside the harness's 180 s
/// timeout, so a stalled title is sampled several times before it is killed.
pub const STALL_SAMPLE_AFTER_SECS: u64 = 30;

/// What the report builder can see. Everything is optional because a session
/// can fail before any of it exists — which is precisely the case that used to
/// produce no artifact at all.
#[derive(Default)]
pub struct SessionInputs {
    pub eboot: PathBuf,
    /// The composed guest image and its dependency offsets, for resolving a
    /// faulting address to `module+offset`.
    ///
    /// Held as an `Arc`, never cloned: `linked.image` is the whole guest image
    /// (hundreds of megabytes for a retail title), and the heartbeat touches
    /// this every ten seconds.
    pub linked: Option<Arc<raeen_firmware::LinkedModule>>,
    pub dep_offsets: Vec<(String, u64)>,
    pub kernel: Option<Arc<raeen_kernel::OrbisKernel>>,
    /// Imports that linked to nothing. Distinct from what the guest *called*.
    pub linked_missing_nids: Vec<String>,
    /// Deliberately-incomplete shims this title imports.
    pub incomplete_imports: Vec<String>,
}

/// What a session looked like when the report was written — every input to
/// [`classify`] other than the run's own ending.
///
/// These are read from six different places at the one production call site
/// (`frame_path::snapshot()`, the process CPU clock, the session's start
/// instant), so grouping them buys type-checked field names in place of seven
/// same-typed positional arguments: `frames_published`, `cpu_ms`, `wall_ms` and
/// `elapsed_secs` are all `u64`-shaped, and `reached_label`/`phase_label`/
/// `counters` are all `&str`. Transposing any pair used to compile silently.
#[derive(Debug, Clone, Copy)]
pub struct RunObservation<'a> {
    /// `frame_path`'s `FramePublished` count.
    pub frames_published: u64,
    /// Furthest frame-path stage reached, as a label.
    pub reached_label: &'a str,
    /// Furthest frame-path phase reached, as a label.
    pub phase_label: &'a str,
    /// The rendered frame-path counter line.
    pub counters: &'a str,
    /// Guest-process CPU time, `None` if the host would not report it.
    pub cpu_ms: Option<u64>,
    /// Guest-process wall-clock time, `None` if unknown.
    pub wall_ms: Option<u64>,
    /// Seconds since the session started.
    pub elapsed_secs: u64,
}

/// Decide the outcome and the one-sentence verdict for a session.
///
/// Pure over its inputs so every ending — fault, clean exit, heartbeat,
/// refusal — is classified by one testable function rather than by whichever
/// call site happened to write the report.
///
/// `outcome` is `None` for a heartbeat (the session is still running).
///
/// Everything except `outcome` is an observation of the *same* run, so it
/// travels as one [`RunObservation`] rather than as seven positional arguments.
/// `outcome` stays separate because it is the discriminant this function
/// switches on, not another measurement.
#[must_use]
pub fn classify(
    outcome: Option<&Result<raeen_runtime::RunOutcome, raeen_runtime::RuntimeError>>,
    observed: &RunObservation<'_>,
) -> (Stage, String) {
    let &RunObservation {
        frames_published,
        reached_label,
        phase_label,
        counters,
        cpu_ms,
        wall_ms,
        elapsed_secs,
    } = observed;
    let (stage, mut verdict) = match outcome {
        Some(Err(error)) => (Stage::Crashed, describe_runtime_error(error)),
        Some(Ok(run)) if frames_published > 0 => (
            Stage::Rendering,
            format!(
                "The guest ran to completion ({run:?}) after publishing {frames_published} \
                 frame(s)."
            ),
        ),
        Some(Ok(run)) => (
            Stage::Exited,
            format!(
                "The guest ended ({run:?}) without ever publishing a frame. {}",
                // The one shared definition, called by the compatibility
                // harness too — so the report and the harness cannot describe
                // the same run two different ways.
                raeen_core::frame_path::silent_zero_frame_verdict(
                    reached_label,
                    phase_label,
                    counters
                )
            ),
        ),
        None if frames_published > 0 => (
            Stage::Rendering,
            format!(
                "Running — {frames_published} frame(s) published so far. No outcome yet \
                 (heartbeat at +{elapsed_secs} s)."
            ),
        ),
        None => (
            Stage::Launching,
            format!(
                "Running — no frame published yet. No outcome yet (heartbeat at \
                 +{elapsed_secs} s). {}",
                raeen_core::frame_path::silent_zero_frame_verdict(
                    reached_label,
                    phase_label,
                    counters
                )
            ),
        ),
    };

    // The CPU shape is what split the measured silent-zero-frame cluster into
    // two different bugs: three titles burned ~2% of a core across 180 s
    // (parked on a primitive that never fired) while one burned 99.7% (a guest
    // busy loop). Normalised to ONE core, not to wall: a single hot guest
    // thread on an 8-core host is only 12.5% of wall, and a ratio-to-wall test
    // would misfile it as parked.
    if let Some(shape) = cpu_shape(cpu_ms, wall_ms) {
        verdict.push(' ');
        verdict.push_str(&shape);
    }
    (stage, verdict)
}

/// `parked (CPU 3.2 s over 180.4 s wall = 1.8% of one core)`, or `None` when
/// the host could not be asked.
#[must_use]
pub fn cpu_shape(cpu_ms: Option<u64>, wall_ms: Option<u64>) -> Option<String> {
    let (cpu_ms, wall_ms) = (cpu_ms?, wall_ms?);
    if wall_ms == 0 {
        return None;
    }
    let ratio = cpu_ms as f64 / wall_ms as f64;
    let label = if ratio >= 0.8 {
        "at least one thread spinning"
    } else if ratio <= 0.1 {
        "parked"
    } else {
        "partially active"
    };
    Some(format!(
        "{label} (CPU {:.1} s over {:.1} s wall = {:.1}% of one core).",
        cpu_ms as f64 / 1000.0,
        wall_ms as f64 / 1000.0,
        ratio * 100.0
    ))
}

/// The fault one-liner for a runtime error — the same wording the session
/// overlay shows, so the two never disagree.
#[must_use]
pub fn describe_runtime_error(error: &raeen_runtime::RuntimeError) -> String {
    match error {
        raeen_runtime::RuntimeError::Faulted { addr, access, kind } => {
            format!("Guest fault at {addr:#x} ({kind} of {access:#x})")
        }
        raeen_runtime::RuntimeError::UnimplementedImport { nid, library, .. } => format!(
            "Unimplemented import: {} ({}) — nid {nid:#018x}",
            raeen_firmware::dynlib::nid_names::describe(*nid),
            library.as_deref().unwrap_or("<unknown library>")
        ),
        raeen_runtime::RuntimeError::IntegerDivideFault {
            rip, cause, origin, ..
        } => format!("{origin} integer-divide fault at {rip:#x} ({cause})"),
        other => format!("Runtime error: {other:?}"),
    }
}

/// Lock a session-report mutex, tolerating poisoning.
///
/// A panic elsewhere in the process must not cost the artifact that explains
/// the panic — that is the exact moment a report matters most.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Owns one session's report pair and keeps it fresh while the guest runs.
pub struct SessionReportWriter {
    dir: PathBuf,
    stem: String,
    md_path: PathBuf,
    started: Instant,
    started_unix: u64,
    inputs: Mutex<SessionInputs>,
    stall: Mutex<Option<String>>,
    guest_console: Mutex<Option<String>>,
    /// Set once the session has been finalized, so a heartbeat racing the
    /// final write cannot overwrite a verdict with `provisional`.
    finalized: std::sync::atomic::AtomicBool,
}

impl SessionReportWriter {
    /// Begin a session: resolve the stem, write the first report immediately,
    /// then start the heartbeat.
    ///
    /// The synchronous first write is the whole point — a process that dies in
    /// its first two seconds still leaves an artifact naming what it was doing.
    pub fn start(eboot: &Path) -> Arc<Self> {
        let started_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let (title_id, _, _) = crash_report::title_meta_for(eboot);
        let stem = format!(
            "{}_{}",
            crash_report::sanitize_component(&title_id),
            crash_report::utc_stamp(started_unix)
        );
        let dir = PathBuf::from(crash_report::REPORTS_DIR);
        let md_path = dir.join(format!("{stem}{}", crash_report::REPORT_SUFFIX));

        let writer = Arc::new(Self {
            dir,
            stem,
            md_path,
            started: Instant::now(),
            started_unix,
            inputs: Mutex::new(SessionInputs {
                eboot: eboot.to_path_buf(),
                ..SessionInputs::default()
            }),
            stall: Mutex::new(None),
            guest_console: Mutex::new(None),
            finalized: std::sync::atomic::AtomicBool::new(false),
        });

        // Before the thread: a hard death at t+2 s must still find a file here.
        writer.write_once();
        tracing::info!(
            report = %writer.md_path.display(),
            "session report opened — refreshed while the title runs, so a stall or a kill still \
             leaves an artifact"
        );

        let interval = std::env::var(HEARTBEAT_SECS_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_HEARTBEAT_SECS);
        if interval > 0 {
            let heartbeat = Arc::clone(&writer);
            let _ = std::thread::Builder::new()
                .name("session-report".into())
                .spawn(move || heartbeat.heartbeat_loop(interval));
        }
        writer
    }

    /// Supply the dependency offsets, once `load_process` has succeeded.
    pub fn attach_dependencies(&self, dep_offsets: Vec<(String, u64)>) {
        lock(&self.inputs).dep_offsets = dep_offsets;
    }

    /// Supply the composed image and the kernel, once linking has succeeded.
    ///
    /// Takes `Arc`s and never clones them: `linked.image` is the whole guest
    /// image, and the heartbeat reads this every tick.
    pub fn attach_image(
        &self,
        linked: Arc<raeen_firmware::LinkedModule>,
        kernel: Arc<raeen_kernel::OrbisKernel>,
    ) {
        let mut inputs = lock(&self.inputs);
        inputs.linked = Some(linked);
        inputs.kernel = Some(kernel);
    }

    /// Record the guest's own stdout/stderr — frequently the only statement of
    /// intent a title makes before it dies.
    pub fn set_guest_console(&self, text: String) {
        *lock(&self.guest_console) = Some(text);
    }

    /// Record what the stall observer saw, for the report's `## Stall` section.
    pub fn set_stall(&self, text: String) {
        *lock(&self.stall) = Some(text);
    }

    fn heartbeat_loop(self: Arc<Self>, interval: u64) {
        let period = std::time::Duration::from_secs(interval);
        let mut last_frames = u64::MAX;
        let mut unchanged_secs = 0u64;
        loop {
            std::thread::sleep(period);
            if self.finalized.load(std::sync::atomic::Ordering::Acquire) {
                return;
            }

            // Sample where the guest threads are only once forward progress
            // has genuinely stopped: suspending threads is not free, and a
            // title is allowed to spend its first seconds loading.
            let frames =
                raeen_core::frame_path::count(raeen_core::frame_path::Stage::FramePublished);
            if frames == last_frames {
                unchanged_secs = unchanged_secs.saturating_add(interval);
            } else {
                unchanged_secs = 0;
                last_frames = frames;
            }
            // Sample only once the frame counter has been static long enough;
            // `then().flatten()` keeps that gate and the `Option` in one
            // expression instead of nesting two `if`s.
            if let Some(sample) = (unchanged_secs >= STALL_SAMPLE_AFTER_SECS)
                .then(|| self.sample_stall())
                .flatten()
            {
                self.set_stall(sample);
            }
            self.write_once();
        }
    }

    /// Where each guest thread currently is, resolved to `module+offset`.
    ///
    /// `None` when there is no kernel yet or the sampler found nothing — an
    /// empty stall section is honest, an invented one is not.
    fn sample_stall(&self) -> Option<String> {
        let kernel = lock(&self.inputs).kernel.clone()?;
        let sampled = raeen_runtime::sample_guest_rips(&kernel);
        if sampled.is_empty() {
            return None;
        }
        let elapsed = self.started.elapsed().as_secs();
        let mut out = format!(
            "No frame published for at least {STALL_SAMPLE_AFTER_SECS} s (session +{elapsed} s). \
             Guest thread positions at the sample:\n"
        );
        for (tid, rip) in sampled {
            // A bare RIP names nothing 250 MB into a stripped C++ binary. The
            // owning module and offset are the whole actionable content, and
            // the thread's name says which of twenty workers is stuck.
            let name = kernel
                .thread_names
                .get(&tid)
                .map_or_else(String::new, |n| format!(" ({})", n.clone()));
            out.push_str(&format!("- t{tid}{name}: rip {rip:#x}\n"));
        }
        Some(out)
    }

    /// Rewrite the report pair from the current state, marked provisional.
    pub fn write_once(&self) {
        if self.finalized.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let report = self.build(None, true);
        self.persist(&report);
    }

    /// Write the final report for a session that ended on its own terms.
    pub fn write_final(
        &self,
        outcome: &Result<raeen_runtime::RunOutcome, raeen_runtime::RuntimeError>,
    ) {
        let report = self.build(Some(outcome), false);
        self.finalized
            .store(true, std::sync::atomic::Ordering::Release);
        self.persist(&report);
        tracing::info!(
            report = %self.md_path.display(),
            outcome = %report.outcome.slug(),
            "session report finalized"
        );
    }

    /// Write the final report for a session that never got as far as running.
    ///
    /// `refused` is one of the seven outcome slugs and is exactly right for
    /// "Encrypted module — decryption needs user-supplied keys", which is the
    /// most important message the clean-room doctrine requires this project to
    /// print and which previously escaped as a bare `Error:` on stderr with no
    /// artifact behind it.
    pub fn write_refused(&self, error: &anyhow::Error) {
        let mut report = self.build(None, false);
        report.outcome = Stage::Refused;
        // The whole `anyhow` chain: the outermost message is usually the least
        // specific one, and the cause is the actionable half.
        let chain: Vec<String> = error.chain().map(|cause| cause.to_string()).collect();
        report.fault = chain.first().cloned().unwrap_or_else(|| error.to_string());
        report.verdict = format!(
            "The session was refused before the guest ran: {}",
            chain.join(" — caused by: ")
        );
        raeen_core::blockers::record(
            raeen_core::blockers::BlockerCategory::HostError,
            "session refused",
            0,
            || chain.join(" — caused by: "),
        );
        report.blockers = raeen_core::blockers::ranked();
        self.finalized
            .store(true, std::sync::atomic::Ordering::Release);
        self.persist(&report);
        tracing::info!(
            report = %self.md_path.display(),
            "session report finalized (refused before execution)"
        );
    }

    fn persist(&self, report: &CrashReport) {
        match report.write_pair(&self.dir, &self.stem, self.started_unix) {
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(%error, "session report could not be written");
            }
        }
    }

    /// Assemble the report from whatever is currently known.
    fn build(
        &self,
        outcome: Option<&Result<raeen_runtime::RunOutcome, raeen_runtime::RuntimeError>>,
        provisional: bool,
    ) -> CrashReport {
        let inputs = lock(&self.inputs);
        let snapshot = raeen_core::frame_path::snapshot();
        let frames_published =
            snapshot.counts[raeen_core::frame_path::Stage::FramePublished as usize];
        let cpu_ms = crash_report::process_cpu_ms();
        let wall_ms = Some(self.started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64);

        let counters = snapshot.counters();
        let (stage, verdict) = classify(
            outcome,
            &RunObservation {
                frames_published,
                reached_label: snapshot.reached_label(),
                phase_label: snapshot.phase_label(),
                counters: &counters,
                cpu_ms,
                wall_ms,
                elapsed_secs: self.started.elapsed().as_secs(),
            },
        );

        let fault = match outcome {
            Some(Err(error)) => describe_runtime_error(error),
            Some(Ok(run)) => format!("No fault — the guest ended with {run:?}"),
            None => "No fault — the session was still running when this was written".to_string(),
        };

        // A faulting address is only resolvable to module+offset while the
        // composed image is around; that is why `linked` is held here.
        let fault_site = match (outcome, inputs.linked.as_ref()) {
            (Some(Err(raeen_runtime::RuntimeError::Faulted { addr, .. })), Some(linked)) => {
                locate(linked, &inputs.dep_offsets, *addr)
            }
            (
                Some(Err(raeen_runtime::RuntimeError::IntegerDivideFault { rip, .. })),
                Some(linked),
            ) => locate(linked, &inputs.dep_offsets, *rip),
            _ => None,
        };

        let (title_id, title, version) = crash_report::title_meta_for(&inputs.eboot);
        let (recent_hle, unresolved_nids, session_duration) = match inputs.kernel.as_ref() {
            Some(kernel) => (
                crash_report::recent_hle_for_report(kernel),
                kernel.unresolved_nid_inventory(),
                Some(kernel.uptime()),
            ),
            None => (Vec::new(), Vec::new(), None),
        };

        let gpu = raeen_gpu::AgcGpuSession::global();
        let shader = gpu.shader_stats();
        let gpu_summary = format!(
            "draws={} presented_frames={} shaders: fetched={} translated_ok={} failed={} \
             skipped_draws={}",
            gpu.draw_count(),
            gpu.present_epoch(),
            shader.distinct_fetched,
            shader.translated_ok,
            shader.translate_failed,
            gpu.shader_skip_count()
        );

        CrashReport {
            outcome: stage,
            provisional,
            verdict,
            title_id,
            title,
            version,
            session_duration,
            fault,
            fault_site,
            recent_hle,
            unresolved_nids,
            linked_missing_nids: inputs.linked_missing_nids.clone(),
            incomplete_imports: inputs.incomplete_imports.clone(),
            frame_path: Some(snapshot),
            blockers: raeen_core::blockers::ranked(),
            blockers_dropped: raeen_core::blockers::dropped_by_category()
                .into_iter()
                .map(|(category, count)| (category.slug().to_string(), count))
                .collect(),
            first_blocker: raeen_core::blockers::first().map(|b| b.line()),
            worst_blocker: raeen_core::blockers::worst().map(|b| b.line()),
            cpu_ms,
            wall_ms,
            stall: lock(&self.stall).clone(),
            gpu_summary: Some(gpu_summary),
            host: Some(crash_report::HostInfo::collect()),
            dump_path: None,
            log_path: Some(PathBuf::from("logs/raeen.log")),
            guest_console_tail: lock(&self.guest_console).clone(),
            host_gpu_notes: Vec::new(),
        }
    }
}

fn locate(
    linked: &raeen_firmware::LinkedModule,
    deps: &[(String, u64)],
    addr: u64,
) -> Option<crash_report::FaultSite> {
    match crash_report::locate_fault(&linked.image, deps, raeen_runtime::GUEST_ARENA_BASE, addr) {
        crash_report::FaultLocation::Site(site) => Some(site),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COUNTERS: &str = "videoout_open=1@812ms dcb_submitted=91@1204ms";

    /// The baseline observation these cases vary from: a run that submitted
    /// DCBs, published nothing, and reports no CPU shape. Each case overrides
    /// only the fields it is actually about (`..observed()`), so what a test is
    /// exercising is visible instead of being the third or sixth positional
    /// argument among seven same-typed ones.
    fn observed() -> RunObservation<'static> {
        RunObservation {
            frames_published: 0,
            reached_label: "dcb_submitted",
            phase_label: "first_guest_thread",
            counters: COUNTERS,
            cpu_ms: None,
            wall_ms: None,
            elapsed_secs: 0,
        }
    }

    #[test]
    fn classify_maps_every_ending_to_a_stage_and_a_verdict() {
        // A fault.
        let faulted = Err(raeen_runtime::RuntimeError::Faulted {
            addr: 0x0002_00a1_03c6,
            access: 0x0300_0000_0010,
            kind: raeen_runtime::FaultKind::Read,
        });
        let (stage, verdict) = classify(Some(&faulted), &observed());
        assert_eq!(stage, Stage::Crashed);
        assert!(verdict.contains("Guest fault at 0x200a103c6"), "{verdict}");

        // A clean exit that never presented — the silent zero-frame class.
        let exited = Ok(raeen_runtime::RunOutcome::Exited(0));
        let (stage, verdict) = classify(Some(&exited), &observed());
        assert_eq!(stage, Stage::Exited);
        assert!(verdict.contains("silent zero-frame run"), "{verdict}");
        // The verdict is the SHARED definition, so the compat harness and this
        // report cannot describe one run two ways.
        assert!(
            verdict.contains(&raeen_core::frame_path::silent_zero_frame_verdict(
                "dcb_submitted",
                "first_guest_thread",
                COUNTERS
            )),
            "{verdict}"
        );

        // A clean exit that did present.
        let (stage, _) = classify(
            Some(&exited),
            &RunObservation {
                frames_published: 900,
                reached_label: "frames_published",
                ..observed()
            },
        );
        assert_eq!(stage, Stage::Rendering);

        // A heartbeat: still running, and it says so instead of guessing.
        let (stage, verdict) = classify(
            None,
            &RunObservation {
                reached_label: "nothing",
                phase_label: "deps_linked",
                elapsed_secs: 40,
                ..observed()
            },
        );
        assert_eq!(stage, Stage::Launching);
        assert!(verdict.contains("No outcome yet"), "{verdict}");
        assert!(verdict.contains("+40 s"), "{verdict}");
    }

    /// The measured cluster split into two different bugs on exactly this
    /// number: three titles parked (~2% of a core over 180 s) and one spun
    /// (99.7%). A report that cannot tell them apart sends the reader down the
    /// wrong path.
    #[test]
    fn a_parked_stall_and_a_spinning_stall_are_told_apart() {
        let parked = classify(
            None,
            &RunObservation {
                reached_label: "nothing",
                cpu_ms: Some(3_200),
                wall_ms: Some(180_000),
                elapsed_secs: 180,
                ..observed()
            },
        )
        .1;
        assert!(parked.contains("parked"), "{parked}");
        assert!(!parked.contains("spinning"), "{parked}");

        let spinning = classify(
            None,
            &RunObservation {
                reached_label: "nothing",
                cpu_ms: Some(179_600),
                wall_ms: Some(180_000),
                elapsed_secs: 180,
                ..observed()
            },
        )
        .1;
        assert!(
            spinning.contains("at least one thread spinning"),
            "{spinning}"
        );

        // Unknown CPU says nothing at all rather than guessing a shape.
        assert_eq!(cpu_shape(None, Some(180_000)), None);
        assert_eq!(cpu_shape(Some(1), None), None);
        assert_eq!(cpu_shape(Some(1), Some(0)), None, "no divide by zero");
    }
}
