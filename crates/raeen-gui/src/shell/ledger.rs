//! Per-title session ledger — the real play history this Shell witnesses.
//!
//! The context block under the Home rail promises "no fictional stats": this
//! is the honest data source behind it. Every launch stamps `last_played`,
//! every exit accumulates `total_play_secs`, and a session that faulted is
//! remembered so Home can say so instead of claiming "Ready to play".
//!
//! Files live beside the per-game override store (one small JSON per title,
//! same sanitized-id naming), and every read degrades to `default()` — a
//! corrupt ledger can never block a launch or a frame.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TitleLedger {
    /// Unix seconds of the most recent launch. `0` = never launched.
    #[serde(default)]
    pub last_played: u64,
    /// Accumulated in-session seconds across all launches.
    #[serde(default)]
    pub total_play_secs: u64,
    /// Whether the most recent session ended after faulting.
    #[serde(default)]
    pub last_faulted: bool,
}

impl TitleLedger {
    /// Compact play-time text ("2h 14m", "8m"); `None` under a minute.
    pub fn play_time_text(&self) -> Option<String> {
        let mins = self.total_play_secs / 60;
        match mins {
            0 => None,
            m if m < 60 => Some(format!("{m}m played")),
            m => Some(format!("{}h {}m played", m / 60, m % 60)),
        }
    }
}

/// Ledger directory: a `session_ledger/` sibling of the per-game override
/// store (both hang off the config file's directory).
pub fn store_dir(config_path: &Path) -> PathBuf {
    let base = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("session_ledger")
}

fn path_for(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{}.json", sanitize_id(id)))
}

/// Load a title's ledger; missing/corrupt files yield the empty default.
pub fn load(dir: &Path, id: &str) -> TitleLedger {
    let Ok(text) = std::fs::read_to_string(path_for(dir, id)) else {
        return TitleLedger::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

/// Persist a title's ledger. Best-effort: failures are logged, never fatal.
pub fn store(dir: &Path, id: &str, ledger: &TitleLedger) {
    if let Err(err) = std::fs::create_dir_all(dir) {
        tracing::warn!(dir = %dir.display(), error = %err, "session ledger dir");
        return;
    }
    let path = path_for(dir, id);
    match serde_json::to_string_pretty(ledger) {
        Ok(json) => {
            if let Err(err) = std::fs::write(&path, json) {
                tracing::warn!(path = %path.display(), error = %err, "session ledger write");
            }
        }
        Err(err) => tracing::warn!(error = %err, "session ledger serialize"),
    }
}

/// Current wall-clock as unix seconds (0 if the host clock is pre-epoch).
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Same filesystem-hostile-character policy as the per-game store. Also used
/// by the screenshot writer, so a screenshot file name can never smuggle path
/// separators from a library id.
pub(crate) fn sanitize_id(id: &str) -> String {
    let cleaned: String = id
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "UNKNOWN".to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory unique to this process **and** this call.
    ///
    /// This test used to reuse one fixed `raeen-ledger-test` path directly
    /// under the shared system temp dir, delete it, and immediately recreate
    /// it. Both halves of that are silent failure modes that only bite under
    /// the I/O load of a full `cargo test --workspace` run, which is why it
    /// passed every time it was run on its own:
    ///
    /// * the opening `remove_dir_all` was best-effort (`let _ = …`), so a
    ///   ledger left behind by a killed run — or one a virus scanner still
    ///   holds a handle on — survives it, and the "missing file → default"
    ///   assertion reads the *previous* run's data instead of a default;
    /// * on Windows a directory whose last handle has not closed lingers in
    ///   delete-pending state, and recreating that exact path fails with
    ///   access denied. [`store`] is best-effort by design — it reports that
    ///   through `tracing::warn!` and returns normally — so the round-trip
    ///   assertion below just sees an empty ledger, with nothing in the test
    ///   output to explain why.
    ///
    /// A path that is never reused cannot hit either one. The pid follows the
    /// convention every other temp-dir test in this crate uses; the counter
    /// separates calls within one process, and the nanosecond stamp separates
    /// runs that land on a recycled pid.
    fn scratch_dir() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "raeen-ledger-test-{}-{}-{nanos}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).expect("create ledger scratch dir");
        dir
    }

    #[test]
    fn round_trips_and_degrades_gracefully() {
        let dir = scratch_dir();
        // Missing file → default.
        let fresh = load(&dir, "PPSA_TEST");
        assert_eq!(fresh.last_played, 0);
        assert!(fresh.play_time_text().is_none());
        // Store and reload.
        let ledger = TitleLedger {
            last_played: 1234,
            total_play_secs: 8100,
            last_faulted: true,
        };
        store(&dir, "PPSA_TEST", &ledger);
        // `store` swallows I/O errors into a log line, and `load` turns *any*
        // read failure into the default ledger — so between them a write that
        // never happened surfaces as a bare `left: 0` value mismatch below,
        // with nothing in the test output naming the real cause. Check the raw
        // file here instead, so that failure reports itself. (Existence alone
        // is not enough: a stale file at the path would satisfy it.)
        let raw = std::fs::read_to_string(path_for(&dir, "PPSA_TEST"))
            .expect("store left no readable ledger file (it logs why via tracing::warn!)");
        assert!(
            raw.contains("1234"),
            "a ledger file exists but store did not write this ledger: {raw}"
        );
        let back = load(&dir, "PPSA_TEST");
        assert_eq!(back.last_played, 1234);
        assert_eq!(back.play_time_text().as_deref(), Some("2h 15m played"));
        assert!(back.last_faulted);
        // Corrupt file → default, no panic.
        std::fs::write(path_for(&dir, "PPSA_TEST"), "{not json").unwrap();
        assert_eq!(load(&dir, "PPSA_TEST").last_played, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn play_time_text_tiers() {
        let mut l = TitleLedger {
            total_play_secs: 59,
            ..TitleLedger::default()
        };
        assert!(l.play_time_text().is_none());
        l.total_play_secs = 60 * 8;
        assert_eq!(l.play_time_text().as_deref(), Some("8m played"));
        l.total_play_secs = 3600 * 3 + 60 * 5;
        assert_eq!(l.play_time_text().as_deref(), Some("3h 5m played"));
    }
}
