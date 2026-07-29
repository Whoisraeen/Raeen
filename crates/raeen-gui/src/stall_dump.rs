//! The `RAEEN_STALL_DUMP` report: what every guest thread is doing while a
//! title makes no progress.
//!
//! # Why this is its own module
//!
//! The report used to be built inline in `main.rs` and counted threads by how
//! many had entries in [`OrbisKernel::recent_hle_calls`](raeen_kernel::OrbisKernel::recent_hle_calls).
//! That produced `STALL_DUMP (0 threads)` on the Blasphemous II (PPSA13580)
//! captures — *while all 15 guest threads were frozen* — because the call ring is
//! only populated under a different env var. The same run printed
//! `IN-FLIGHT HLE: <none — all threads between calls>`, which is what
//! `in_flight_hle` looks like when nothing ever wrote to it.
//!
//! An instrument that reports "nothing" for "not armed" is worse than no
//! instrument: those two lines were read as *evidence* that the threads had
//! moved out of their guest waits and into host synchronization, and that reading
//! drove a whole investigation. (`raeen_runtime::dispatch::stall_instruments_armed`
//! now arms the instruments this report reads, so the two can no longer disagree.)
//!
//! So the rules here are:
//!
//! * **The thread inventory comes from the host thread sampler**, not from an
//!   opt-in ring. A thread the OS knows about is counted even if it has never
//!   made an HLE call.
//! * **"Parked" is a positive observation**, never an absence: it means the
//!   thread's host RIP is inside a named Windows wait syscall
//!   ([`raeen_runtime::host_wait_primitive`]).
//! * **Every field says which of "unknown" and "none" it means.** A parked thread
//!   with no in-flight HLE call reads `in host code between HLE calls`, because
//!   that *is* the interesting case; a thread with no sampled data reads
//!   `<unknown>`.
//!
//! # How long has it been parked
//!
//! [`StallTracker`] fingerprints each thread's state and remembers when that
//! fingerprint first appeared, so the duration is a **lower bound** measured from
//! the first dump that saw the current state — reported as `>=41.4s`, never as an
//! exact figure it cannot know. This costs nothing per HLE call, which matters:
//! the alternative (timestamping every dispatch) is paid by every call in the
//! run, including the fast ones that are not the problem.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// One guest thread as a single stall sample saw it.
///
/// Deliberately plain data — no runtime or kernel types — so the whole report can
/// be built and asserted on without a live guest process.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThreadSample {
    /// Guest thread id.
    pub thread: u64,
    /// The guest's own name for the thread, empty when it never set one.
    pub name: String,
    /// `library::function` the thread is currently inside. `None` means the
    /// thread is *not* in an HLE call — it is in guest code or in our runtime.
    pub in_flight: Option<String>,
    /// Most recent HLE calls, newest first.
    pub recent: Vec<String>,
    /// Guest RIP as `module+offset`, when the sampler resolved one.
    pub guest_site: Option<String>,
    /// The Windows wait primitive the thread's host RIP is inside, when it is
    /// inside one. `None` is "not observed waiting", not "observed running".
    pub parked_in: Option<String>,
    /// Shallow host backtrace, innermost frame first.
    pub chain: String,
}

impl ThreadSample {
    /// What distinguishes this thread's *state* from a different state. Two
    /// samples with equal fingerprints mean the thread has not moved.
    ///
    /// The guest RIP is included so a thread spinning in guest code — which makes
    /// no HLE calls at all and so has an unchanging in-flight field — is still
    /// seen to be moving.
    fn fingerprint(&self) -> String {
        format!(
            "{}|{}|{}|{}",
            self.in_flight.as_deref().unwrap_or("-"),
            self.parked_in.as_deref().unwrap_or("-"),
            self.guest_site.as_deref().unwrap_or("-"),
            self.recent.first().map_or("-", String::as_str),
        )
    }

    /// Whether this sample is a positive observation of a parked thread.
    #[must_use]
    pub fn is_parked(&self) -> bool {
        self.parked_in.is_some()
    }
}

/// How long a thread has held its current state, as observed across dumps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Held {
    /// Lower bound: time since the first dump that saw this state.
    pub since: Duration,
    /// How many consecutive dumps have seen it, this one included.
    pub samples: u32,
}

/// Remembers each thread's previous state so successive dumps can say how long
/// nothing has changed.
#[derive(Debug, Default)]
pub struct StallTracker {
    /// thread -> (state fingerprint, consecutive samples, first seen).
    seen: HashMap<u64, (String, u32, Instant)>,
}

impl StallTracker {
    /// Fold one sample set in, returning each thread's [`Held`] in the order the
    /// samples were given.
    ///
    /// A thread whose fingerprint changed restarts at `samples = 1`; a thread that
    /// disappeared is forgotten, so a recycled thread id cannot inherit a stale
    /// age.
    pub fn observe(&mut self, samples: &[ThreadSample], now: Instant) -> Vec<Held> {
        let mut next = HashMap::with_capacity(samples.len());
        let mut held = Vec::with_capacity(samples.len());
        for sample in samples {
            let fingerprint = sample.fingerprint();
            let (count, since) = match self.seen.get(&sample.thread) {
                Some((previous, count, first)) if *previous == fingerprint => {
                    (count.saturating_add(1), *first)
                }
                _ => (1, now),
            };
            held.push(Held {
                since: now.saturating_duration_since(since),
                samples: count,
            });
            next.insert(sample.thread, (fingerprint, count, since));
        }
        self.seen = next;
        held
    }
}

/// Render one stall dump.
///
/// `held[i]` describes `samples[i]`; a short `held` degrades to "age unknown"
/// rather than panicking, because a diagnostic must not be able to kill the run
/// it is diagnosing.
#[must_use]
pub fn format_report(
    samples: &[ThreadSample],
    held: &[Held],
    time_in_hle: &[String],
    console_tail: &str,
) -> String {
    let parked = samples.iter().filter(|s| s.is_parked()).count();
    let total = samples.len();
    let mut out = String::new();
    if total == 0 {
        // No guest thread has host state yet — say that, rather than printing a
        // zero that reads as "nothing is wrong".
        out.push_str("STALL_DUMP: no guest threads are registered yet (nothing to sample)");
        let _ = write!(out, "\nGUEST CONSOLE: {console_tail}");
        return out;
    }
    let _ = write!(
        out,
        "STALL_DUMP: {total} guest thread(s) — {parked} host-parked, {} not observed waiting",
        total - parked
    );
    if parked == total {
        out.push_str(
            "\nVERDICT: every guest thread is parked in a host wait. Nothing in the process \
             can advance on its own — the wake has to come from outside the guest, or it \
             never comes.",
        );
    }

    for (index, sample) in samples.iter().enumerate() {
        let name = if sample.name.is_empty() {
            String::new()
        } else {
            format!("({})", sample.name)
        };
        let age = match held.get(index) {
            Some(h) => format!(">={:.1}s over {} dump(s)", h.since.as_secs_f64(), h.samples),
            None => "age unknown".to_owned(),
        };
        let state = match (&sample.parked_in, &sample.in_flight) {
            // The case this report exists for: parked, but not inside any HLE
            // call — i.e. in our own runtime code between guest calls.
            (Some(primitive), None) => {
                format!("PARKED {age} in host code between HLE calls [{primitive}]")
            }
            (Some(primitive), Some(call)) => format!("PARKED {age} inside {call} [{primitive}]"),
            (None, Some(call)) => format!("in {call}, unchanged {age}"),
            (None, None) => format!("not in an HLE call, unchanged {age}"),
        };
        let _ = write!(out, "\nt{}{name} {state}", sample.thread);
        let _ = write!(
            out,
            "\n    last returned from: {}",
            if sample.recent.is_empty() {
                "<no HLE call recorded for this thread>".to_owned()
            } else {
                sample.recent.join(" <- ")
            }
        );
        let _ = write!(
            out,
            "\n    guest rip: {}",
            sample.guest_site.as_deref().unwrap_or("<unknown>")
        );
    }

    // Grouped, because in the capture that motivated this every worker printed
    // the same twelve frames and the repetition was most of the output.
    out.push_str("\nHOST BACKTRACES:");
    for (chain, threads) in group_chains(samples) {
        let ids = threads
            .iter()
            .map(|t| format!("t{t}"))
            .collect::<Vec<_>>()
            .join(",");
        let _ = write!(out, "\n  {ids}: {chain}");
    }

    if !time_in_hle.is_empty() {
        let _ = write!(out, "\nTIME IN HLE (top):\n{}", time_in_hle.join("\n"));
    }
    let _ = write!(out, "\nGUEST CONSOLE: {console_tail}");
    out
}

/// Group threads by identical host backtrace, preserving first-seen order.
fn group_chains(samples: &[ThreadSample]) -> Vec<(String, Vec<u64>)> {
    let mut groups: Vec<(String, Vec<u64>)> = Vec::new();
    for sample in samples {
        let chain = if sample.chain.is_empty() {
            "<not sampled>".to_owned()
        } else {
            sample.chain.clone()
        };
        match groups.iter_mut().find(|(existing, _)| *existing == chain) {
            Some((_, threads)) => threads.push(sample.thread),
            None => groups.push((chain, vec![sample.thread])),
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::{Held, StallTracker, ThreadSample, format_report};
    use std::time::{Duration, Instant};

    /// The Blasphemous II capture, reduced: 14 workers parked inside
    /// `sceKernelWaitSema` and the main thread inside `pthread_cond_wait`, all on
    /// the same host wait primitive.
    fn blasphemous_samples() -> Vec<ThreadSample> {
        let mut samples = vec![ThreadSample {
            thread: 1,
            name: "main".to_owned(),
            in_flight: Some("libScePosix::pthread_cond_wait".to_owned()),
            recent: vec!["libScePosix::read".to_owned()],
            guest_site: Some("eboot+0xb47c11".to_owned()),
            parked_in: Some("WaitOnAddress futex (std or parking_lot)".to_owned()),
            chain: "ntdll(ZwWaitForAlertByThreadId) <- GuestWaiter::wait_for_signal".to_owned(),
        }];
        for thread in 2..=15 {
            samples.push(ThreadSample {
                thread,
                name: String::new(),
                in_flight: Some("libkernel::sceKernelWaitSema".to_owned()),
                recent: vec!["libkernel::sceKernelSignalSema".to_owned()],
                guest_site: Some("eboot+0xb47bd0".to_owned()),
                parked_in: Some("WaitOnAddress futex (std or parking_lot)".to_owned()),
                chain: "ntdll(ZwWaitForAlertByThreadId) <- kernel_semaphore::hle_wait".to_owned(),
            });
        }
        samples
    }

    /// The regression this module exists for. The old report counted threads by
    /// call-ring entries and said `(0 threads)` for exactly this state.
    #[test]
    fn every_thread_parked_is_counted_not_reported_as_zero() {
        let samples = blasphemous_samples();
        let held = vec![
            Held {
                since: Duration::from_secs_f64(41.4),
                samples: 7,
            };
            samples.len()
        ];
        let report = format_report(&samples, &held, &[], "<empty>");
        assert!(
            report.starts_with("STALL_DUMP: 15 guest thread(s) — 15 host-parked, 0 not observed"),
            "header must count parked threads: {report}"
        );
        assert!(
            !report.contains("(0 threads)"),
            "the zero-thread wording must be gone: {report}"
        );
        assert!(
            report.contains("VERDICT: every guest thread is parked"),
            "a fully parked process must be called out: {report}"
        );
    }

    /// Deliverable: per parked thread, what it last returned from and how long it
    /// has been parked.
    #[test]
    fn a_parked_thread_reports_its_call_and_its_age() {
        let samples = blasphemous_samples();
        let held = vec![
            Held {
                since: Duration::from_secs_f64(41.4),
                samples: 7,
            };
            samples.len()
        ];
        let report = format_report(&samples, &held, &[], "<empty>");
        assert!(
            report.contains(
                "t2 PARKED >=41.4s over 7 dump(s) inside libkernel::sceKernelWaitSema \
                 [WaitOnAddress futex (std or parking_lot)]"
            ),
            "the in-flight call, age and primitive must all appear: {report}"
        );
        assert!(
            report.contains("last returned from: libkernel::sceKernelSignalSema"),
            "the previous call must appear: {report}"
        );
    }

    /// A thread parked with no in-flight HLE call is the case the mission named:
    /// it must read as "in host code between HLE calls", never as an absence.
    #[test]
    fn parked_outside_any_hle_call_is_named_explicitly() {
        let samples = vec![ThreadSample {
            thread: 4,
            name: String::new(),
            in_flight: None,
            recent: vec!["libkernel::sceKernelSignalSema".to_owned()],
            guest_site: None,
            parked_in: Some("WaitForSingleObject".to_owned()),
            chain: "ntdll(NtWaitForSingleObject)".to_owned(),
        }];
        let held = vec![Held {
            since: Duration::from_secs(12),
            samples: 3,
        }];
        let report = format_report(&samples, &held, &[], "<empty>");
        assert!(
            report.contains(
                "t4 PARKED >=12.0s over 3 dump(s) in host code between HLE calls \
                 [WaitForSingleObject]"
            ),
            "a park outside any HLE call must be stated: {report}"
        );
        assert!(
            report.contains("guest rip: <unknown>"),
            "an unsampled field must say unknown, not print as empty: {report}"
        );
    }

    /// Fifteen identical twelve-frame chains were most of the old output's bulk.
    #[test]
    fn identical_host_backtraces_are_grouped() {
        let samples = blasphemous_samples();
        let held = vec![
            Held {
                since: Duration::from_secs(6),
                samples: 1,
            };
            samples.len()
        ];
        let report = format_report(&samples, &held, &[], "<empty>");
        assert!(
            report.contains("t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15: ntdll"),
            "the 14 workers' shared chain must print once: {report}"
        );
        assert_eq!(
            report.matches("kernel_semaphore::hle_wait").count(),
            1,
            "the shared chain must not be repeated per thread: {report}"
        );
    }

    /// An empty thread list must not render as a zero that reads like health.
    #[test]
    fn no_threads_yet_says_so() {
        let report = format_report(&[], &[], &[], "<empty>");
        assert!(
            report.contains("no guest threads are registered yet"),
            "{report}"
        );
    }

    #[test]
    fn tracker_ages_an_unchanged_thread_and_resets_a_moving_one() {
        let mut tracker = StallTracker::default();
        let t0 = Instant::now();
        let samples = blasphemous_samples();

        let first = tracker.observe(&samples, t0);
        assert_eq!(first[0].samples, 1);
        assert_eq!(first[0].since, Duration::ZERO);

        let second = tracker.observe(&samples, t0 + Duration::from_secs(6));
        assert_eq!(second[0].samples, 2);
        assert_eq!(second[0].since, Duration::from_secs(6));

        let third = tracker.observe(&samples, t0 + Duration::from_secs(12));
        assert_eq!(third[0].samples, 3);
        assert_eq!(third[0].since, Duration::from_secs(12));

        // A thread that advanced to another call restarts, so a stale age can
        // never be attributed to a thread that is actually working.
        let mut moved = samples.clone();
        moved[0].in_flight = Some("libkernel::sceKernelSignalSema".to_owned());
        let fourth = tracker.observe(&moved, t0 + Duration::from_secs(18));
        assert_eq!(fourth[0].samples, 1);
        assert_eq!(fourth[0].since, Duration::ZERO);
        // The untouched workers keep accumulating.
        assert_eq!(fourth[1].samples, 4);
        assert_eq!(fourth[1].since, Duration::from_secs(18));
    }

    /// A thread that spins in pure guest code makes no HLE calls, so only its
    /// guest RIP can show that it is moving.
    #[test]
    fn a_moving_guest_rip_resets_the_age() {
        let mut tracker = StallTracker::default();
        let t0 = Instant::now();
        let spinning = |site: &str| {
            vec![ThreadSample {
                thread: 9,
                guest_site: Some(site.to_owned()),
                ..ThreadSample::default()
            }]
        };
        tracker.observe(&spinning("eboot+0x100"), t0);
        let moved = tracker.observe(&spinning("eboot+0x140"), t0 + Duration::from_secs(6));
        assert_eq!(moved[0].samples, 1, "a moving RIP is not a stall");
    }

    /// A thread that vanishes must not leave an age behind for a reused id.
    #[test]
    fn a_vanished_thread_is_forgotten() {
        let mut tracker = StallTracker::default();
        let t0 = Instant::now();
        let one = vec![ThreadSample {
            thread: 3,
            in_flight: Some("libkernel::sceKernelWaitSema".to_owned()),
            ..ThreadSample::default()
        }];
        tracker.observe(&one, t0);
        tracker.observe(&[], t0 + Duration::from_secs(6));
        let back = tracker.observe(&one, t0 + Duration::from_secs(12));
        assert_eq!(back[0].samples, 1, "a recycled id must not inherit an age");
    }
}
