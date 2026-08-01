//! Low-overhead sampled guest instruction-pointer profile.
//!
//! `RAEEN_STALL_DUMP` answers a broad deadlock question, but it intentionally
//! arms per-HLE timing and captures host stacks. Those instruments can perturb a
//! CPU-heavy retail transition. `RAEEN_SAMPLE_GUEST_RIPS=1` is the narrow probe:
//! every 100 ms it briefly samples each registered guest thread, buckets guest
//! PCs to 64-byte regions, and reports the hottest regions every five seconds.

use raeen_kernel::OrbisKernel;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);
const REPORT_INTERVAL: Duration = Duration::from_secs(5);
const GUEST_BUCKET_BYTES: u64 = 64;
const TOP_SITES: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SiteKey {
    thread: u64,
    site: String,
}

#[derive(Debug, Default)]
struct RipProfile {
    sweeps: u64,
    sampled_threads: u64,
    guest: HashMap<SiteKey, u64>,
    host: HashMap<SiteKey, u64>,
    names: HashMap<u64, String>,
}

impl RipProfile {
    fn observe(&mut self, kernel: &OrbisKernel, samples: Vec<(u64, u64)>) {
        self.sweeps = self.sweeps.saturating_add(1);
        self.sampled_threads = self.sampled_threads.saturating_add(samples.len() as u64);
        for (thread, rip) in samples {
            if let Some(name) = kernel.thread_names.get(&thread) {
                self.names.insert(thread, name.clone());
            }
            let (site, guest) = kernel.unwind_module_for_addr(rip).map_or_else(
                || {
                    (
                        raeen_runtime::symbolize_host_addr(rip)
                            .or_else(|| raeen_runtime::host_module_for_addr(rip))
                            .unwrap_or_else(|| format!("host:{rip:#x}")),
                        false,
                    )
                },
                |module| {
                    let offset = (rip - module.start) & !(GUEST_BUCKET_BYTES - 1);
                    (format!("{}+{offset:#x}", module.name), true)
                },
            );
            let counts = if guest {
                &mut self.guest
            } else {
                &mut self.host
            };
            *counts.entry(SiteKey { thread, site }).or_insert(0) += 1;
        }
    }

    fn take_report(&mut self) -> String {
        let report = format_report(
            self.sweeps,
            self.sampled_threads,
            &self.guest,
            &self.host,
            &self.names,
        );
        self.sweeps = 0;
        self.sampled_threads = 0;
        self.guest.clear();
        self.host.clear();
        report
    }
}

fn format_report(
    sweeps: u64,
    sampled_threads: u64,
    guest: &HashMap<SiteKey, u64>,
    host: &HashMap<SiteKey, u64>,
    names: &HashMap<u64, String>,
) -> String {
    let mut out =
        format!("GUEST_RIP_PROFILE: {sweeps} sweep(s), {sampled_threads} thread sample(s)");
    append_top(&mut out, "guest", guest, names, sweeps);
    append_host_threads(&mut out, host, names, sweeps);
    out
}

/// Host samples are reported per thread. A global top-N is misleading here:
/// every idle worker spends every sweep at the same ntdll wait address, so a
/// dozen legitimately sleeping threads crowd out the one active mutex owner
/// the profile exists to find.
fn append_host_threads(
    out: &mut String,
    counts: &HashMap<SiteKey, u64>,
    names: &HashMap<u64, String>,
    sweeps: u64,
) {
    let mut per_thread: HashMap<u64, (&SiteKey, u64, u64)> = HashMap::new();
    for (key, count) in counts {
        let slot = per_thread.entry(key.thread).or_insert((key, 0, 0));
        slot.2 = slot.2.saturating_add(*count);
        if *count > slot.1 || (*count == slot.1 && key.site < slot.0.site) {
            slot.0 = key;
            slot.1 = *count;
        }
    }
    let mut ranked: Vec<_> = per_thread.into_iter().collect();
    ranked.sort_unstable_by_key(|(thread, _)| *thread);
    out.push_str("\n  host/parked state per thread:");
    if ranked.is_empty() {
        out.push_str(" <none sampled>");
        return;
    }
    for (thread, (key, top_hits, host_samples)) in ranked {
        let name = names.get(&thread).map_or("<unnamed>", String::as_str);
        let host_share = if sweeps == 0 {
            0.0
        } else {
            host_samples as f64 * 100.0 / sweeps as f64
        };
        let _ = write!(
            out,
            "\n    t{thread} ('{name}') host {:>3}/{sweeps} ({host_share:>5.1}%); top {:>3} hit(s): {}",
            host_samples, top_hits, key.site
        );
    }
}

fn append_top(
    out: &mut String,
    label: &str,
    counts: &HashMap<SiteKey, u64>,
    names: &HashMap<u64, String>,
    sweeps: u64,
) {
    let mut ranked: Vec<_> = counts.iter().collect();
    ranked.sort_unstable_by(|(key_a, count_a), (key_b, count_b)| {
        count_b
            .cmp(count_a)
            .then_with(|| key_a.thread.cmp(&key_b.thread))
            .then_with(|| key_a.site.cmp(&key_b.site))
    });
    let _ = write!(out, "\n  top {label} sites:");
    if ranked.is_empty() {
        out.push_str(" <none sampled>");
        return;
    }
    for (key, count) in ranked.into_iter().take(TOP_SITES) {
        let name = names.get(&key.thread).map_or("<unnamed>", String::as_str);
        let sweep_share = if sweeps == 0 {
            0.0
        } else {
            *count as f64 * 100.0 / sweeps as f64
        };
        let _ = write!(
            out,
            "\n    t{} ('{}') {:>3} hit(s), {:>5.1}% of sweeps: {}",
            key.thread, name, count, sweep_share, key.site
        );
    }
}

pub fn spawn_if_enabled(kernel: Arc<OrbisKernel>) {
    if std::env::var_os("RAEEN_SAMPLE_GUEST_RIPS").is_none() {
        return;
    }
    tracing::info!(
        target: "rip_profile",
        "guest RIP sampling enabled (100 ms interval, 64-byte guest buckets; no per-HLE timing)"
    );
    let result = std::thread::Builder::new()
        .name("raeen-rip-profile".to_owned())
        .spawn(move || {
            let mut profile = RipProfile::default();
            let mut last_report = Instant::now();
            loop {
                std::thread::sleep(SAMPLE_INTERVAL);
                profile.observe(&kernel, raeen_runtime::sample_guest_rips(&kernel));
                if last_report.elapsed() >= REPORT_INTERVAL {
                    tracing::info!(target: "rip_profile", "{}", profile.take_report());
                    last_report = Instant::now();
                }
            }
        });
    if let Err(error) = result {
        tracing::warn!(target: "rip_profile", %error, "could not start guest RIP sampler");
    }
}

#[cfg(test)]
mod tests {
    use super::{HashMap, SiteKey, format_report};

    #[test]
    fn report_prioritizes_hot_sites_and_names_threads() {
        let guest = HashMap::from([
            (
                SiteKey {
                    thread: 6,
                    site: "module+0x8e25e00".to_owned(),
                },
                41,
            ),
            (
                SiteKey {
                    thread: 4,
                    site: "module+0x12340".to_owned(),
                },
                3,
            ),
        ]);
        let names = HashMap::from([(6, "Streaming Pool".to_owned())]);
        let report = format_report(50, 100, &guest, &HashMap::new(), &names);
        let hot = report.find("module+0x8e25e00").unwrap();
        let cold = report.find("module+0x12340").unwrap();
        assert!(hot < cold);
        assert!(report.contains("t6 ('Streaming Pool')  41 hit(s),  82.0%"));
        assert!(report.contains("host/parked state per thread: <none sampled>"));
    }
}
