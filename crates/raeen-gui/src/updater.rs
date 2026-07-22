//! In-app auto-updater backed by GitHub Releases.
//!
//! Flow: a background thread asks the GitHub API for the latest release of
//! [`REPO`]; if its tag is a newer semantic version than the running
//! [`raeen_core::VERSION`], the Settings → System screen offers to download
//! the bare-exe release asset (published by `.github/workflows/release.yml`
//! next to the zip precisely so the updater never has to unpack an
//! archive). The download is staged next to the running executable; on
//! "Restart & Update" a tiny batch script waits for this process to exit,
//! swaps the staged exe into place, and relaunches.
//!
//! Everything that can be pure is pure (version parsing/ordering, release
//! JSON parsing, asset selection, swap-script generation) and unit-tested
//! offline; only [`check_latest`], [`download_to`], and [`apply_staged`]
//! touch the network/filesystem/process boundary.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;

/// GitHub repository releases are pulled from.
pub const REPO: &str = "Whoisraeen/Raeen";

/// The release asset the updater downloads: the bare executable uploaded by
/// the release workflow alongside the user-facing zip.
pub const EXE_ASSET_SUFFIX: &str = "-windows-x86_64.exe";

/// Hard cap on a downloaded update, far above any plausible raeen.exe.
const MAX_DOWNLOAD_BYTES: u64 = 512 * 1024 * 1024;

/// A release the updater considers installable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseInfo {
    /// Release tag as published, e.g. `v0.2.0` or `v0.2.0-alpha`.
    pub tag: String,
    /// Direct download URL of the bare-exe asset.
    pub exe_url: String,
}

/// Events the updater's worker threads report back to the Shell.
#[derive(Debug)]
pub enum UpdaterEvent {
    UpToDate { latest: String },
    UpdateAvailable(ReleaseInfo),
    CheckFailed(String),
    Staged { tag: String, staged: PathBuf },
    DownloadFailed(String),
}

/// UI-facing updater state machine, driven by [`UpdaterEvent`]s.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum UpdaterState {
    #[default]
    Idle,
    Checking,
    UpToDate {
        latest: String,
    },
    Downloading {
        tag: String,
    },
    Staged {
        tag: String,
        staged: PathBuf,
    },
    Error(String),
}

impl UpdaterState {
    /// One-line status for the Settings "Version" row's value column.
    pub fn status_line(&self) -> String {
        match self {
            UpdaterState::Idle => String::new(),
            UpdaterState::Checking => "Checking for updates…".to_string(),
            UpdaterState::UpToDate { latest } => format!("Up to date (latest {latest})"),
            UpdaterState::Downloading { tag } => format!("Downloading {tag}…"),
            UpdaterState::Staged { tag, .. } => format!("{tag} ready — restart to apply"),
            UpdaterState::Error(err) => format!("Update check failed: {err}"),
        }
    }

    /// Label of the Settings "System" action row for the current state.
    pub fn action_label(&self) -> &'static str {
        match self {
            UpdaterState::Idle | UpdaterState::UpToDate { .. } | UpdaterState::Error(_) => {
                "Check for Updates"
            }
            UpdaterState::Checking => "Checking…",
            UpdaterState::Downloading { .. } => "Downloading…",
            UpdaterState::Staged { .. } => "Restart & Update",
        }
    }

    /// Whether a worker thread is in flight (the Shell keeps repainting so
    /// its result is seen without user input).
    pub fn is_busy(&self) -> bool {
        matches!(
            self,
            UpdaterState::Checking | UpdaterState::Downloading { .. }
        )
    }
}

/// A parsed `vMAJOR.MINOR.PATCH[-pre]` version. Ordering follows semver's
/// core rule: numeric fields first, and a pre-release version sorts *below*
/// its release (`0.2.0-alpha < 0.2.0`). Pre-release identifiers compare
/// lexicographically — enough for the alpha/beta/rc ladder this project
/// tags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre: Option<String>,
}

impl Version {
    /// Parse `"v0.2.0-alpha"`, `"0.2.0"`, etc. `None` for anything that
    /// isn't three dot-separated integers (+ optional `-pre`).
    pub fn parse(s: &str) -> Option<Version> {
        let s = s.trim().strip_prefix('v').unwrap_or(s.trim());
        let (core, pre) = match s.split_once('-') {
            Some((core, pre)) if !pre.is_empty() => (core, Some(pre.to_string())),
            Some(_) => return None,
            None => (s, None),
        };
        let mut parts = core.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        Some(Version {
            major,
            minor,
            patch,
            pre,
        })
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.major, self.minor, self.patch)
            .cmp(&(other.major, other.minor, other.patch))
            .then_with(|| match (&self.pre, &other.pre) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Greater, // release > pre
                (Some(_), None) => std::cmp::Ordering::Less,
                (Some(a), Some(b)) => a.cmp(b),
            })
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// `true` only when both versions parse and `latest_tag` is strictly newer
/// than `current` — unparseable tags never trigger an update.
pub fn is_newer(latest_tag: &str, current: &str) -> bool {
    match (Version::parse(latest_tag), Version::parse(current)) {
        (Some(latest), Some(current)) => latest > current,
        _ => false,
    }
}

/// Parse a GitHub `releases/latest` JSON document into a [`ReleaseInfo`],
/// selecting the bare-exe asset by [`EXE_ASSET_SUFFIX`].
pub fn parse_latest_release(json: &str) -> Result<ReleaseInfo, String> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("release JSON malformed: {e}"))?;
    let tag = value["tag_name"]
        .as_str()
        .ok_or("release JSON has no tag_name")?
        .to_string();
    let assets = value["assets"].as_array().ok_or("release has no assets")?;
    let exe_url = assets
        .iter()
        .find_map(|asset| {
            let name = asset["name"].as_str()?;
            if name.ends_with(EXE_ASSET_SUFFIX) {
                asset["browser_download_url"].as_str().map(str::to_string)
            } else {
                None
            }
        })
        .ok_or_else(|| format!("release {tag} has no *{EXE_ASSET_SUFFIX} asset"))?;
    if !exe_url.starts_with("https://") {
        return Err(format!("refusing non-HTTPS download URL: {exe_url}"));
    }
    Ok(ReleaseInfo { tag, exe_url })
}

/// The batch script that performs the swap after this process exits:
/// wait for `pid` to go away, replace the exe with the staged download,
/// relaunch, self-delete. Pure so the exact contract is testable.
pub fn swap_script(exe: &Path, staged: &Path, pid: u32) -> String {
    format!(
        "@echo off\r\n\
         :wait\r\n\
         tasklist /FI \"PID eq {pid}\" 2>NUL | find \"{pid}\" >NUL\r\n\
         if not errorlevel 1 (\r\n\
             timeout /t 1 /nobreak >NUL\r\n\
             goto wait\r\n\
         )\r\n\
         move /Y \"{staged}\" \"{exe}\"\r\n\
         start \"\" \"{exe}\"\r\n\
         del \"%~f0\"\r\n",
        pid = pid,
        staged = staged.display(),
        exe = exe.display(),
    )
}

/// Where a downloaded update is staged: next to the running exe so the
/// final `move` is same-volume atomic.
pub fn staging_path(exe: &Path) -> PathBuf {
    exe.with_extension("exe.update")
}

/// Blocking: fetch the latest release from the GitHub API.
fn check_latest() -> Result<ReleaseInfo, String> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let body = ureq::get(&url)
        .set("User-Agent", "raeen-updater")
        .set("Accept", "application/vnd.github+json")
        .timeout(std::time::Duration::from_secs(15))
        .call()
        .map_err(|e| match e {
            ureq::Error::Status(404, _) => "no releases published yet".to_string(),
            other => other.to_string(),
        })?
        .into_string()
        .map_err(|e| e.to_string())?;
    parse_latest_release(&body)
}

/// Blocking: download `url` to `dest` (size-capped).
fn download_to(url: &str, dest: &Path) -> Result<(), String> {
    let response = ureq::get(url)
        .set("User-Agent", "raeen-updater")
        .timeout(std::time::Duration::from_secs(600))
        .call()
        .map_err(|e| e.to_string())?;
    let mut reader = response.into_reader().take(MAX_DOWNLOAD_BYTES);
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|e| format!("download failed: {e}"))?;
    if bytes.is_empty() {
        return Err("downloaded update is empty".to_string());
    }
    std::fs::write(dest, &bytes).map_err(|e| format!("could not stage update: {e}"))?;
    Ok(())
}

/// Spawn the non-blocking latest-release check; the outcome (relative to
/// `current_version`) arrives on `tx` as an [`UpdaterEvent`].
pub fn spawn_check(tx: Sender<UpdaterEvent>, current_version: String) {
    std::thread::spawn(move || {
        let event = match check_latest() {
            Ok(info) => {
                if is_newer(&info.tag, &current_version) {
                    UpdaterEvent::UpdateAvailable(info)
                } else {
                    UpdaterEvent::UpToDate { latest: info.tag }
                }
            }
            Err(err) => UpdaterEvent::CheckFailed(err),
        };
        let _ = tx.send(event);
    });
}

/// Spawn the non-blocking download of `info`'s exe asset into
/// [`staging_path`] next to the running executable.
pub fn spawn_download(tx: Sender<UpdaterEvent>, info: ReleaseInfo) {
    std::thread::spawn(move || {
        let event = (|| -> Result<UpdaterEvent, String> {
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let staged = staging_path(&exe);
            download_to(&info.exe_url, &staged)?;
            Ok(UpdaterEvent::Staged {
                tag: info.tag.clone(),
                staged,
            })
        })()
        .unwrap_or_else(UpdaterEvent::DownloadFailed);
        let _ = tx.send(event);
    });
}

/// Write and launch the swap script for a staged update. The caller is
/// responsible for closing the app right after this returns `Ok` — the
/// script waits for this process to exit before swapping.
pub fn apply_staged(staged: &Path) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    if !staged.is_file() {
        return Err(format!("staged update missing: {}", staged.display()));
    }
    let script = swap_script(&exe, staged, std::process::id());
    let script_path = exe.with_file_name("raeen-update.bat");
    std::fs::write(&script_path, script).map_err(|e| e.to_string())?;
    #[cfg(windows)]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", "/min"])
            .arg(&script_path)
            .spawn()
            .map_err(|e| format!("could not launch update script: {e}"))?;
        Ok(())
    }
    #[cfg(not(windows))]
    {
        Err("self-update is only implemented on Windows".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parses_with_and_without_v_and_pre() {
        assert_eq!(
            Version::parse("v1.2.3"),
            Some(Version {
                major: 1,
                minor: 2,
                patch: 3,
                pre: None
            })
        );
        assert_eq!(
            Version::parse("0.1.0-alpha"),
            Some(Version {
                major: 0,
                minor: 1,
                patch: 0,
                pre: Some("alpha".to_string())
            })
        );
        assert_eq!(Version::parse("1.2"), None);
        assert_eq!(Version::parse("1.2.3.4"), None);
        assert_eq!(Version::parse("v1.2.3-"), None);
        assert_eq!(Version::parse("garbage"), None);
    }

    #[test]
    fn ordering_numeric_then_release_over_prerelease() {
        assert!(is_newer("v0.2.0", "0.1.9"));
        assert!(is_newer("v0.1.10", "0.1.9"));
        assert!(is_newer("v1.0.0", "0.99.99"));
        assert!(is_newer("v0.2.0", "0.2.0-alpha")); // release > its pre
        assert!(is_newer("v0.2.0-beta", "0.2.0-alpha"));
        assert!(!is_newer("v0.2.0-alpha", "0.2.0"));
        assert!(!is_newer("v0.1.0", "0.1.0"));
        assert!(!is_newer("v0.1.0", "0.2.0"));
    }

    #[test]
    fn unparseable_tags_never_update() {
        assert!(!is_newer("nightly", "0.1.0"));
        assert!(!is_newer("v0.2.0", "unknown"));
    }

    #[test]
    fn parse_latest_release_picks_the_bare_exe_asset() {
        let json = r#"{
            "tag_name": "v0.2.0",
            "assets": [
                {"name": "raeen-windows-x86_64.zip",
                 "browser_download_url": "https://github.com/r/releases/download/v0.2.0/raeen-windows-x86_64.zip"},
                {"name": "raeen-windows-x86_64.exe",
                 "browser_download_url": "https://github.com/r/releases/download/v0.2.0/raeen-windows-x86_64.exe"}
            ]
        }"#;
        let info = parse_latest_release(json).unwrap();
        assert_eq!(info.tag, "v0.2.0");
        assert!(info.exe_url.ends_with("raeen-windows-x86_64.exe"));
    }

    #[test]
    fn parse_latest_release_rejects_missing_asset_and_bad_json() {
        let no_exe = r#"{"tag_name": "v0.2.0", "assets": [
            {"name": "source.zip", "browser_download_url": "https://x/source.zip"}
        ]}"#;
        assert!(parse_latest_release(no_exe).is_err());
        assert!(parse_latest_release("not json").is_err());
        assert!(parse_latest_release(r#"{"assets": []}"#).is_err());
    }

    #[test]
    fn parse_latest_release_refuses_plain_http() {
        let json = r#"{"tag_name": "v0.2.0", "assets": [
            {"name": "raeen-windows-x86_64.exe",
             "browser_download_url": "http://evil.example/raeen-windows-x86_64.exe"}
        ]}"#;
        assert!(parse_latest_release(json).is_err());
    }

    #[test]
    fn swap_script_waits_swaps_relaunches_and_self_deletes() {
        let exe = Path::new(r"C:\apps\raeen\raeen.exe");
        let staged = Path::new(r"C:\apps\raeen\raeen.exe.update");
        let script = swap_script(exe, staged, 4242);
        assert!(script.contains("PID eq 4242"));
        assert!(
            script
                .contains(r#"move /Y "C:\apps\raeen\raeen.exe.update" "C:\apps\raeen\raeen.exe""#)
        );
        assert!(script.contains(r#"start "" "C:\apps\raeen\raeen.exe""#));
        assert!(script.contains("del \"%~f0\""));
    }

    #[test]
    fn staging_path_sits_next_to_the_exe() {
        assert_eq!(
            staging_path(Path::new(r"C:\apps\raeen\raeen.exe")),
            PathBuf::from(r"C:\apps\raeen\raeen.exe.update")
        );
    }

    #[test]
    fn status_and_action_labels_track_state() {
        assert_eq!(UpdaterState::Idle.action_label(), "Check for Updates");
        let downloading = UpdaterState::Downloading {
            tag: "v0.2.0".to_string(),
        };
        assert_eq!(downloading.action_label(), "Downloading…");
        assert!(downloading.status_line().contains("v0.2.0"));
        assert!(downloading.is_busy());
        assert!(UpdaterState::Checking.is_busy());
        let staged = UpdaterState::Staged {
            tag: "v0.2.0".to_string(),
            staged: PathBuf::from(r"C:\x\raeen.exe.update"),
        };
        assert_eq!(staged.action_label(), "Restart & Update");
        assert!(!staged.is_busy());
    }
}
