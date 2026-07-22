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

/// Same filesystem-hostile-character policy as the per-game store.
fn sanitize_id(id: &str) -> String {
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

    #[test]
    fn round_trips_and_degrades_gracefully() {
        let dir = std::env::temp_dir().join("raeen-ledger-test");
        let _ = std::fs::remove_dir_all(&dir);
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
        let mut l = TitleLedger::default();
        l.total_play_secs = 59;
        assert!(l.play_time_text().is_none());
        l.total_play_secs = 60 * 8;
        assert_eq!(l.play_time_text().as_deref(), Some("8m played"));
        l.total_play_secs = 3600 * 3 + 60 * 5;
        assert_eq!(l.play_time_text().as_deref(), Some("3h 5m played"));
    }
}
