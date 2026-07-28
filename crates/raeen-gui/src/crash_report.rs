//! Actionable crash reports (north star: "install → library → launch → logs →
//! settings → actionable crash reports").
//!
//! The runtime already *knows* everything a crash report needs — faulting
//! module+offset, the bytes at RIP, the recent HLE calls per guest thread,
//! the unresolved-NID call inventory, GPU counters, host facts — but until
//! now a human assembled that picture by hand from `logs/raeen.log` (see
//! `docs/gta5-blocker-analysis-2026-07-27.md` for what that costs). This
//! module turns those inputs into ONE self-contained file:
//!
//! ```text
//! logs/crashes/<title-id>_<UTC>.report.md
//! ```
//!
//! written next to the minidumps (`crashdump.rs`), so the report, the `.dmp`,
//! and the log form a single shareable bundle.
//!
//! Two writers, one format:
//! * the **runner child** (`--run-eboot`) writes the rich report when
//!   `execute_process` returns a fault — it has the kernel, the linked image,
//!   and the GPU session in-process;
//! * the **Shell** writes a fallback report only when the runner died hard
//!   enough that a minidump landed but no report did (the crash handler is a
//!   last-chance mechanism; a process that `abort()`s never runs its own
//!   report path). See [`ensure_report_for_crashed_runner`].
//!
//! Everything that decides *what the report says* is a pure function over
//! plain inputs (testable headlessly); the IO wrappers are thin.

use std::path::{Path, PathBuf};

/// Where crash artifacts live, relative to the working directory — shared
/// with the minidump server in `crashdump.rs` so dumps and reports pair up
/// in one folder.
pub const REPORTS_DIR: &str = "logs/crashes";

/// Report file suffix. The double extension keeps reports distinguishable
/// from any other markdown while still opening in a text editor.
pub const REPORT_SUFFIX: &str = ".report.md";

/// The report line the Shell's list view extracts as the fault one-liner.
/// Kept as a named constant so the renderer and the parser cannot drift.
const FAULT_LINE_PREFIX: &str = "- Fault: ";

/// Host facts, mirroring the `host system` block `main.rs` logs at startup.
#[derive(Debug, Clone, PartialEq)]
pub struct HostInfo {
    pub cpu: String,
    pub cores: usize,
    pub ram_gb: f64,
    pub os: String,
}

impl HostInfo {
    /// Measure the running host (CPU model, core count, RAM, OS build).
    pub fn collect() -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_cpu_all();
        sys.refresh_memory();
        Self {
            cpu: sys
                .cpus()
                .first()
                .map(|c| c.brand().trim().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            cores: sys.cpus().len(),
            ram_gb: sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            os: sysinfo::System::long_os_version().unwrap_or_else(|| "unknown".into()),
        }
    }
}

/// Where in the composed guest image a fault landed: which module owns the
/// address, the offset within that module, and the bytes there (decodable on
/// the spot — see `main.rs::report_fault_site` for why a bare RIP is useless).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultSite {
    pub module: String,
    pub offset: u64,
    pub rip_bytes: Vec<u8>,
}

/// Result of resolving a faulting guest address against the loaded image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultLocation {
    /// The address is below the guest arena — not guest code.
    BelowImage,
    /// The address is past the loaded image — not guest code.
    PastImage { image_len: usize },
    /// The address lands in a loaded module.
    Site(FaultSite),
}

/// Resolve `addr` to the owning module + offset + bytes over the composed
/// process image. `deps` are `(name, image_offset)` pairs for every loaded
/// dependency; the last one at or below the address owns it, and below them
/// all it is the eboot's. Pure over its inputs.
pub fn locate_fault(image: &[u8], deps: &[(String, u64)], base: u64, addr: u64) -> FaultLocation {
    let Some(offset) = addr.checked_sub(base) else {
        return FaultLocation::BelowImage;
    };
    let Ok(offset) = usize::try_from(offset) else {
        return FaultLocation::PastImage {
            image_len: image.len(),
        };
    };
    if offset >= image.len() {
        return FaultLocation::PastImage {
            image_len: image.len(),
        };
    }
    let (module, module_offset) = match deps
        .iter()
        .filter(|(_, off)| usize::try_from(*off).is_ok_and(|off| off <= offset))
        .max_by_key(|(_, off)| *off)
    {
        Some((name, off)) => (name.clone(), offset as u64 - off),
        None => ("eboot.bin".to_string(), offset as u64),
    };
    let end = (offset + 16).min(image.len());
    FaultLocation::Site(FaultSite {
        module,
        offset: module_offset,
        rip_bytes: image[offset..end].to_vec(),
    })
}

/// Everything one crash report says. Fields are plain data so the renderer
/// stays a pure function; collection happens at the two wiring sites.
#[derive(Debug, Clone, Default)]
pub struct CrashReport {
    /// Real title id from `sce_sys/param.json`, or the game-folder name.
    pub title_id: String,
    /// Display title, when metadata provides one.
    pub title: Option<String>,
    /// Content version from `param.json` (e.g. `01.000.000`).
    pub version: Option<String>,
    /// How long the guest ran before the fault.
    pub session_duration: Option<std::time::Duration>,
    /// The fault one-liner — what the session overlay shows, verbatim.
    pub fault: String,
    /// Module+offset+bytes when the faulting address resolved to guest code.
    pub fault_site: Option<FaultSite>,
    /// Most-recent-first HLE calls per guest thread: `(thread label, calls)`.
    pub recent_hle: Vec<(String, Vec<String>)>,
    /// De-duplicated unresolved-NID call inventory (one line per distinct
    /// `(nid, function, library, caller)`, with call counts).
    pub unresolved_nids: Vec<String>,
    /// GPU counters one-liner (draws, presented frames, shader cache), when
    /// the GPU session was reachable.
    pub gpu_summary: Option<String>,
    /// Host facts (CPU/cores/RAM/OS).
    pub host: Option<HostInfo>,
    /// Paired minidump, when one landed for this session.
    pub dump_path: Option<PathBuf>,
    /// The run's log file (rotated to `raeen.log.1` by the next run).
    pub log_path: Option<PathBuf>,
}

impl CrashReport {
    /// Render the whole report as self-contained markdown. Pure.
    pub fn render(&self, unix_secs: u64) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(out, "# Raeen crash report — {}", self.title_id);
        out.push('\n');
        let title_line = match (&self.title, &self.version) {
            (Some(title), Some(version)) => format!("{title} ({}) v{version}", self.title_id),
            (Some(title), None) => format!("{title} ({})", self.title_id),
            (None, Some(version)) => format!("{} v{version}", self.title_id),
            (None, None) => self.title_id.clone(),
        };
        let _ = writeln!(out, "- Title: {title_line}");
        let _ = writeln!(out, "{FAULT_LINE_PREFIX}{}", one_liner(&self.fault, 200));
        let _ = writeln!(out, "- Generated: {}", utc_display(unix_secs));
        if let Some(duration) = self.session_duration {
            let _ = writeln!(out, "- Session duration: {:.1} s", duration.as_secs_f64());
        }
        let _ = writeln!(out, "- Emulator: Raeen v{}", raeen_core::VERSION);

        let _ = writeln!(out, "\n## Fault\n\n{}", self.fault);
        if let Some(site) = &self.fault_site {
            let _ = writeln!(out, "\n- Module: {} at +{:#x}", site.module, site.offset);
            let bytes: Vec<String> = site.rip_bytes.iter().map(|b| format!("{b:02x}")).collect();
            let _ = writeln!(out, "- Bytes at RIP: {}", bytes.join(" "));
        }

        let _ = writeln!(out, "\n## Recent HLE calls (most recent first)\n");
        if self.recent_hle.is_empty() {
            let _ = writeln!(out, "<none recorded>");
        }
        for (thread, calls) in &self.recent_hle {
            let joined = if calls.is_empty() {
                "<no calls recorded>".to_string()
            } else {
                calls.join(" <- ")
            };
            let _ = writeln!(out, "- {thread}: {joined}");
        }

        let _ = writeln!(out, "\n## Unresolved-NID calls\n");
        if self.unresolved_nids.is_empty() {
            let _ = writeln!(out, "<none — every called import resolved>");
        }
        for line in &self.unresolved_nids {
            let _ = writeln!(out, "- {line}");
        }

        if let Some(gpu) = &self.gpu_summary {
            let _ = writeln!(out, "\n## GPU\n\n{gpu}");
        }

        if let Some(host) = &self.host {
            let _ = writeln!(out, "\n## Host\n");
            let _ = writeln!(out, "- CPU: {} ({} threads)", host.cpu, host.cores);
            let _ = writeln!(out, "- RAM: {:.1} GB", host.ram_gb);
            let _ = writeln!(out, "- OS: {}", host.os);
        }

        let _ = writeln!(out, "\n## Artifacts\n");
        match &self.dump_path {
            Some(dump) => {
                let _ = writeln!(out, "- Minidump: {}", dump.display());
            }
            None => {
                let _ = writeln!(out, "- Minidump: none (fault was caught in-process)");
            }
        }
        if let Some(log) = &self.log_path {
            let _ = writeln!(
                out,
                "- Log: {} (this run; the next run rotates it to raeen.log.1)",
                log.display()
            );
        }
        out
    }

    /// The report's file name: `<title-id>_<UTC>.report.md`.
    pub fn file_name(&self, unix_secs: u64) -> String {
        format!(
            "{}_{}{REPORT_SUFFIX}",
            sanitize_component(&self.title_id),
            utc_stamp(unix_secs)
        )
    }

    /// Write the rendered report under `dir` with the standard name.
    /// Creates `dir` if needed.
    pub fn write_to(&self, dir: &Path, unix_secs: u64) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(self.file_name(unix_secs));
        std::fs::write(&path, self.render(unix_secs))?;
        Ok(path)
    }

    /// [`Self::write_to`] stamped with the current wall clock.
    pub fn write_now(&self, dir: &Path) -> std::io::Result<PathBuf> {
        self.write_to(dir, now_unix_secs())
    }
}

/// Current wall clock as seconds since the Unix epoch.
fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Make a string safe as a file-name component: ASCII alphanumerics, `-`,
/// `.` pass through; everything else becomes `-`. Empty input reads as
/// `unknown-title` rather than producing `_2026….report.md`.
pub fn sanitize_component(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "unknown-title".to_string()
    } else {
        cleaned
    }
}

/// Truncate to `max_chars` characters, appending `…` when something was cut.
/// Character-based, so multi-byte input can never split a UTF-8 boundary.
pub fn one_liner(s: &str, max_chars: usize) -> String {
    let s = s.lines().next().unwrap_or_default();
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Days-since-epoch → (year, month, day). Howard Hinnant's `civil_from_days`,
/// exact over the whole `u64` timestamp range this uses.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// Compact UTC stamp for file names: `YYYYMMDD-HHMMSSZ`. Pure — no chrono.
pub fn utc_stamp(unix_secs: u64) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(unix_secs);
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}Z")
}

/// Human UTC display: `YYYY-MM-DD HH:MM:SS UTC`.
pub fn utc_display(unix_secs: u64) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(unix_secs);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02} UTC")
}

fn utc_parts(unix_secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (unix_secs / 86_400) as i64;
    let rem = unix_secs % 86_400;
    let (y, mo, d) = civil_from_days(days);
    (
        y,
        mo,
        d,
        (rem / 3600) as u32,
        (rem % 3600 / 60) as u32,
        (rem % 60) as u32,
    )
}

/// Turn a file-name stamp (`20260727-184205Z`) back into the short display
/// the Shell's list rows show (`2026-07-27 18:42 UTC`). Unparsable stamps
/// come back verbatim — better an odd row than a lost report.
pub fn display_stamp(stamp: &str) -> String {
    let bare = stamp.strip_suffix('Z').unwrap_or(stamp);
    let (date, time) = match bare.split_once('-') {
        Some(parts) => parts,
        None => return stamp.to_string(),
    };
    if date.len() != 8 || time.len() != 6 || !bare.chars().all(|c| c.is_ascii_digit() || c == '-') {
        return stamp.to_string();
    }
    format!(
        "{}-{}-{} {}:{} UTC",
        &date[0..4],
        &date[4..6],
        &date[6..8],
        &time[0..2],
        &time[2..4]
    )
}

/// One report file, pre-digested for the Shell's Settings ▸ System rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportListing {
    pub path: PathBuf,
    /// Title id parsed from the file name.
    pub title_id: String,
    /// Short UTC display parsed from the file name.
    pub when: String,
    /// The report's fault one-liner.
    pub fault: String,
}

/// Parse a report file name (`<title-id>_<stamp>.report.md`) into its title
/// id and display time. Pure; `None` for non-report names.
pub fn parse_report_name(file_name: &str) -> Option<(String, String)> {
    let stem = file_name.strip_suffix(REPORT_SUFFIX)?;
    let (title_id, stamp) = stem.rsplit_once('_')?;
    if title_id.is_empty() {
        return None;
    }
    Some((title_id.to_string(), display_stamp(stamp)))
}

/// Extract the fault one-liner from rendered report text. Pure.
pub fn parse_fault_line(text: &str) -> Option<String> {
    text.lines()
        .take(32)
        .find_map(|line| line.strip_prefix(FAULT_LINE_PREFIX))
        .map(str::to_string)
}

/// List the crash reports under `dir`, newest first (by modification time).
/// A missing directory is an empty list, never an error.
pub fn list_reports(dir: &Path) -> Vec<ReportListing> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<(std::time::SystemTime, ReportListing)> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?.to_string();
            let (title_id, when) = parse_report_name(&name)?;
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            // Bounded read: the one-liner lives in the header.
            let fault = std::fs::read_to_string(&path)
                .ok()
                .and_then(|text| parse_fault_line(&text))
                .unwrap_or_else(|| "<unreadable report>".to_string());
            Some((
                modified,
                ReportListing {
                    path,
                    title_id,
                    when,
                    fault,
                },
            ))
        })
        .collect();
    found.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    found.into_iter().map(|(_, listing)| listing).collect()
}

/// Newest file under `dir` with `extension`, modified at or after `since`.
fn newest_file_since(
    dir: &Path,
    since: std::time::SystemTime,
    matches: impl Fn(&str) -> bool,
) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if !matches(name) {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            (modified >= since).then_some((modified, path))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, path)| path)
}

/// Newest minidump written at or after `since`, if any.
pub fn newest_dump_since(dir: &Path, since: std::time::SystemTime) -> Option<PathBuf> {
    newest_file_since(dir, since, |name| {
        Path::new(name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("dmp"))
    })
}

/// Newest crash report written at or after `since`, if any.
pub fn newest_report_since(dir: &Path, since: std::time::SystemTime) -> Option<PathBuf> {
    newest_file_since(dir, since, |name| name.ends_with(REPORT_SUFFIX))
}

/// Title metadata for the report header: the real `sce_sys/param.json` id/
/// title/version when the package ships one, else the game folder's name.
pub fn title_meta_for(eboot: &Path) -> (String, Option<String>, Option<String>) {
    let dir = eboot.parent().unwrap_or_else(|| Path::new("."));
    let folder = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "unknown-title".to_string());
    match raeen_loader::pkg::scan_game_directory(dir) {
        Ok(meta) => {
            let title_id = if meta.title_id.is_empty() || meta.title_id == "UNKNOWN00000" {
                folder
            } else {
                meta.title_id
            };
            let title = (!meta.title.is_empty() && meta.title != "Unknown").then_some(meta.title);
            let version = (!meta.app_version.is_empty()).then_some(meta.app_version);
            (title_id, title, version)
        }
        Err(_) => (folder, None, None),
    }
}

/// Shell-side fallback, called when the isolated runner exited unsuccessfully.
///
/// * A report newer than `session_start` already exists → the runner child
///   wrote the rich one before dying; reference it.
/// * Otherwise, if a **minidump** landed for this session (a hard crash the
///   child could not report on), write a fallback report pairing that dump.
/// * Otherwise nothing — a runner that failed before executing anything
///   (missing eboot, load error) is a launch fault, not a crash, and the
///   session overlay already carries its message.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
pub fn ensure_report_for_crashed_runner(
    eboot: &Path,
    session_start: std::time::SystemTime,
    fault: &str,
) -> Option<PathBuf> {
    let dir = Path::new(REPORTS_DIR);
    if let Some(existing) = newest_report_since(dir, session_start) {
        return Some(existing);
    }
    let dump = newest_dump_since(dir, session_start)?;
    let (title_id, title, version) = title_meta_for(eboot);
    let report = CrashReport {
        title_id,
        title,
        version,
        session_duration: session_start.elapsed().ok(),
        fault: format!("{fault} — the runner died before it could describe the fault itself"),
        fault_site: None,
        recent_hle: Vec::new(),
        unresolved_nids: Vec::new(),
        gpu_summary: None,
        host: Some(HostInfo::collect()),
        dump_path: Some(dump),
        log_path: Some(PathBuf::from("logs/raeen.log")),
    };
    match report.write_now(dir) {
        Ok(path) => {
            tracing::info!(report = %path.display(), "crash report written for crashed runner");
            Some(path)
        }
        Err(error) => {
            tracing::warn!(%error, "crash report could not be written");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report() -> CrashReport {
        CrashReport {
            title_id: "PPSA17221".to_string(),
            title: Some("Minecraft".to_string()),
            version: Some("1.21.100".to_string()),
            session_duration: Some(std::time::Duration::from_secs_f64(84.25)),
            fault: "Guest fault at 0x200a103c6 (read of 0x30000000010)".to_string(),
            fault_site: Some(FaultSite {
                module: "libc.prx".to_string(),
                offset: 0x103c6,
                rip_bytes: vec![0x48, 0x8b, 0x07],
            }),
            recent_hle: vec![(
                "t3 (MainThread)".to_string(),
                vec![
                    "libkernel::sceKernelUsleep".to_string(),
                    "libc::clock_gettime".to_string(),
                ],
            )],
            unresolved_nids: vec![
                "0x00000000deadbeef sceAgcFoo library=libSceAgc caller=eboot.bin calls=12"
                    .to_string(),
            ],
            gpu_summary: Some("draws=0 presented_frames=3".to_string()),
            host: Some(HostInfo {
                cpu: "TestCPU".to_string(),
                cores: 24,
                ram_gb: 63.9,
                os: "Windows 11 Pro".to_string(),
            }),
            dump_path: Some(PathBuf::from("logs/crashes/raeen-runner-1.dmp")),
            log_path: Some(PathBuf::from("logs/raeen.log")),
        }
    }

    #[test]
    fn utc_stamp_and_display_are_exact() {
        assert_eq!(utc_stamp(0), "19700101-000000Z");
        // 2001-09-09 01:46:40 UTC — the classic billennium second.
        assert_eq!(utc_stamp(1_000_000_000), "20010909-014640Z");
        assert_eq!(utc_display(1_000_000_000), "2001-09-09 01:46:40 UTC");
    }

    #[test]
    fn display_stamp_round_trips_and_tolerates_garbage() {
        assert_eq!(display_stamp("20260727-184205Z"), "2026-07-27 18:42 UTC");
        assert_eq!(display_stamp("not-a-stamp"), "not-a-stamp");
    }

    #[test]
    fn file_name_is_title_id_underscore_utc() {
        let report = sample_report();
        assert_eq!(
            report.file_name(1_000_000_000),
            "PPSA17221_20010909-014640Z.report.md"
        );
        // Hostile ids sanitize instead of escaping the directory.
        let hostile = CrashReport {
            title_id: "../evil id".to_string(),
            ..CrashReport::default()
        };
        assert_eq!(
            hostile.file_name(0),
            "..-evil-id_19700101-000000Z.report.md"
        );
    }

    #[test]
    fn render_carries_every_section_and_the_fault_anchor() {
        let report = sample_report();
        let text = report.render(1_000_000_000);
        assert!(text.starts_with("# Raeen crash report — PPSA17221"));
        assert!(text.contains("- Title: Minecraft (PPSA17221) v1.21.100"));
        assert!(text.contains("- Fault: Guest fault at 0x200a103c6 (read of 0x30000000010)"));
        assert!(text.contains("- Session duration: 84.2 s"));
        assert!(text.contains("- Module: libc.prx at +0x103c6"));
        assert!(text.contains("- Bytes at RIP: 48 8b 07"));
        assert!(
            text.contains("t3 (MainThread): libkernel::sceKernelUsleep <- libc::clock_gettime")
        );
        assert!(text.contains("sceAgcFoo library=libSceAgc caller=eboot.bin calls=12"));
        assert!(text.contains("draws=0 presented_frames=3"));
        assert!(text.contains("- CPU: TestCPU (24 threads)"));
        assert!(text.contains("- Minidump: logs/crashes/raeen-runner-1.dmp"));
        assert!(text.contains("- Log: logs/raeen.log"));
        // The list view can get the one-liner back out.
        assert_eq!(
            parse_fault_line(&text).as_deref(),
            Some("Guest fault at 0x200a103c6 (read of 0x30000000010)")
        );
    }

    #[test]
    fn render_is_honest_about_missing_optionals() {
        let report = CrashReport {
            title_id: "NOVA00001".to_string(),
            fault: "Isolated runner stopped with exit code: 0xc0000005".to_string(),
            ..CrashReport::default()
        };
        let text = report.render(0);
        assert!(text.contains("<none recorded>"));
        assert!(text.contains("<none — every called import resolved>"));
        assert!(text.contains("- Minidump: none (fault was caught in-process)"));
        assert!(!text.contains("## GPU"));
        assert!(!text.contains("## Host"));
    }

    #[test]
    fn locate_fault_names_the_owning_module() {
        let image = vec![0xAAu8; 0x100];
        let deps = vec![
            ("libc.prx".to_string(), 0x40u64),
            ("libSceFoo.prx".to_string(), 0x80u64),
        ];
        // Below every dependency: the eboot owns it.
        match locate_fault(&image, &deps, 0x1000, 0x1010) {
            FaultLocation::Site(site) => {
                assert_eq!(site.module, "eboot.bin");
                assert_eq!(site.offset, 0x10);
                assert_eq!(site.rip_bytes.len(), 16);
            }
            other => panic!("expected a site, got {other:?}"),
        }
        // Inside the highest dependency at or below the address.
        match locate_fault(&image, &deps, 0x1000, 0x1090) {
            FaultLocation::Site(site) => {
                assert_eq!(site.module, "libSceFoo.prx");
                assert_eq!(site.offset, 0x10);
            }
            other => panic!("expected a site, got {other:?}"),
        }
        assert_eq!(
            locate_fault(&image, &deps, 0x1000, 0x0),
            FaultLocation::BelowImage
        );
        assert_eq!(
            locate_fault(&image, &deps, 0x1000, 0x2000),
            FaultLocation::PastImage { image_len: 0x100 }
        );
        // Bytes clamp at the image end instead of slicing past it.
        match locate_fault(&image, &deps, 0x1000, 0x10f8) {
            FaultLocation::Site(site) => assert_eq!(site.rip_bytes.len(), 8),
            other => panic!("expected a site, got {other:?}"),
        }
    }

    #[test]
    fn parse_report_name_accepts_reports_and_rejects_the_rest() {
        assert_eq!(
            parse_report_name("PPSA17221_20260727-184205Z.report.md"),
            Some(("PPSA17221".to_string(), "2026-07-27 18:42 UTC".to_string()))
        );
        assert_eq!(parse_report_name("raeen-runner-175.dmp"), None);
        assert_eq!(parse_report_name("notes.md"), None);
        assert_eq!(parse_report_name("_20260727-184205Z.report.md"), None);
    }

    #[test]
    fn one_liner_takes_first_line_and_truncates_on_char_boundaries() {
        assert_eq!(one_liner("short", 10), "short");
        assert_eq!(one_liner("first\nsecond", 10), "first");
        assert_eq!(one_liner("abcdefgh", 5), "abcd…");
        // Multi-byte input never splits a UTF-8 boundary.
        assert_eq!(one_liner("ééééé", 3), "éé…");
    }

    #[test]
    fn write_list_and_pairing_round_trip() {
        let dir =
            std::env::temp_dir().join(format!("raeen-crash-report-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // An empty/missing directory lists as empty and pairs nothing.
        assert!(list_reports(&dir).is_empty());
        assert_eq!(newest_dump_since(&dir, std::time::UNIX_EPOCH), None);

        let older = CrashReport {
            title_id: "NOVA00001".to_string(),
            fault: "older fault".to_string(),
            ..CrashReport::default()
        };
        let newer = sample_report();
        let older_path = older.write_to(&dir, 1_000_000_000).expect("write older");
        // Distinct mtimes so "newest first" is deterministic.
        std::thread::sleep(std::time::Duration::from_millis(30));
        let newer_path = newer.write_to(&dir, 1_000_000_100).expect("write newer");
        std::fs::write(dir.join("raeen-runner-1.dmp"), b"dump").expect("write dump");
        std::fs::write(dir.join("notes.md"), b"not a report").expect("write decoy");

        let listed = list_reports(&dir);
        assert_eq!(listed.len(), 2, "decoys and dumps are not reports");
        assert_eq!(listed[0].path, newer_path);
        assert_eq!(listed[0].title_id, "PPSA17221");
        assert_eq!(listed[0].when, "2001-09-09 01:48 UTC");
        assert!(listed[0].fault.starts_with("Guest fault at 0x200a103c6"));
        assert_eq!(listed[1].path, older_path);
        assert_eq!(listed[1].fault, "older fault");

        assert_eq!(
            newest_report_since(&dir, std::time::UNIX_EPOCH),
            Some(newer_path)
        );
        assert_eq!(
            newest_dump_since(&dir, std::time::UNIX_EPOCH),
            Some(dir.join("raeen-runner-1.dmp"))
        );
        // A "since" in the future pairs nothing.
        let future = std::time::SystemTime::now() + std::time::Duration::from_secs(3600);
        assert_eq!(newest_dump_since(&dir, future), None);
        assert_eq!(newest_report_since(&dir, future), None);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
