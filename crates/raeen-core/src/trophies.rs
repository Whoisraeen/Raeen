//! Local per-title trophy unlock store.
//!
//! Raeen cannot parse a retail title's trophy pack (`TROPHY.TRP` /
//! `trophy2.ucp` are encrypted retail data — the permanent no-Sony-keys wall),
//! so trophy *definitions* (names, descriptions, grades, group layout, total
//! counts) are unavailable. What Raeen *can* own honestly is the unlock
//! ledger: which trophy ids this title unlocked locally and when. This module
//! is that ledger — a small JSON file per title, stored as a sibling of the
//! title's save-data host directory:
//!
//! ```text
//! savedata/<title>/                  ← save-data host map (VFS /savedata0)
//! savedata/<title>-trophies.json     ← this store
//! ```
//!
//! The file sits *next to* the title's save root, never inside it, so guest
//! save-slot enumeration through `/savedata0` can never see it.
//!
//! Consumers:
//! - `raeen-hle` (writer): the UDS trophy-unlock event path persists unlocks
//!   write-through as they happen.
//! - `raeen-gui` (reader): the per-game overlay shows "N trophies unlocked ·
//!   last <time>" and the running session polls the file to toast new unlocks.
//!
//! Load is null-safe like the Shell's other per-title JSON stores (per-game
//! settings, ledger): a missing or corrupt file degrades to an empty ledger,
//! never an error.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Current on-disk schema version.
pub const TROPHY_STORE_VERSION: u32 = 1;

/// Trophy-id capacity of the Orbis unlock-flag bitset
/// (`ORBIS_NP_TROPHY_FLAG_SETSIZE`, shadPS4 `np_trophy.h`). Ids are `0..128`.
pub const TROPHY_FLAG_SETSIZE: usize = 128;

/// What an unlock request found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnlockOutcome {
    /// First unlock of this id — persisted to disk.
    NewlyUnlocked,
    /// The id was already unlocked; the original timestamp is kept.
    AlreadyUnlocked,
}

/// On-disk shape. Unlock timestamps are Unix milliseconds (UTC).
#[derive(Debug, Default, Serialize, Deserialize)]
struct TrophyStoreFile {
    version: u32,
    /// trophy id → unlock time (Unix ms). `BTreeMap` keeps the file diffable.
    unlocks: BTreeMap<i32, u64>,
}

/// A loaded per-title unlock ledger, bound to its backing file.
#[derive(Debug)]
pub struct TrophyStore {
    path: PathBuf,
    unlocks: BTreeMap<i32, u64>,
}

impl TrophyStore {
    /// The store file for a title whose save-data host root is
    /// `savedata_root` (e.g. `savedata/Minecraft` →
    /// `savedata/Minecraft-trophies.json`). A rootless path (no final
    /// component) falls back to `<root>/trophies.json`.
    pub fn path_for_savedata_root(savedata_root: &Path) -> PathBuf {
        match savedata_root.file_name().and_then(|name| name.to_str()) {
            Some(name) => savedata_root.with_file_name(format!("{name}-trophies.json")),
            None => savedata_root.join("trophies.json"),
        }
    }

    /// Load the ledger at `path`. Missing, unreadable, or corrupt files all
    /// degrade to an empty ledger (logged, never an error) — a bad trophy
    /// file must never block anything.
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let unlocks = match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<TrophyStoreFile>(&text) {
                Ok(file) => file.unlocks,
                Err(err) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %err,
                        "trophy store malformed — starting empty (existing file kept until next unlock)"
                    );
                    BTreeMap::new()
                }
            },
            Err(_) => BTreeMap::new(),
        };
        Self { path, unlocks }
    }

    /// Backing file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Unlock `trophy_id` at `unix_ms`, writing through to disk when the id
    /// is new. A repeated unlock keeps the original timestamp, touches
    /// nothing on disk, and reports [`UnlockOutcome::AlreadyUnlocked`].
    pub fn unlock(&mut self, trophy_id: i32, unix_ms: u64) -> std::io::Result<UnlockOutcome> {
        if self.unlocks.contains_key(&trophy_id) {
            return Ok(UnlockOutcome::AlreadyUnlocked);
        }
        self.unlocks.insert(trophy_id, unix_ms);
        self.save()?;
        Ok(UnlockOutcome::NewlyUnlocked)
    }

    /// Unlock `trophy_id` stamped with the current wall clock.
    pub fn unlock_now(&mut self, trophy_id: i32) -> std::io::Result<UnlockOutcome> {
        self.unlock(trophy_id, now_unix_ms())
    }

    /// Whether `trophy_id` is recorded as unlocked.
    pub fn is_unlocked(&self, trophy_id: i32) -> bool {
        self.unlocks.contains_key(&trophy_id)
    }

    /// Number of unlocked trophies.
    pub fn unlocked_count(&self) -> usize {
        self.unlocks.len()
    }

    /// Unix-ms timestamp of the most recent unlock, if any.
    pub fn last_unlock_ms(&self) -> Option<u64> {
        self.unlocks.values().copied().max()
    }

    /// Unlocked ids in ascending order.
    pub fn unlocked_ids(&self) -> Vec<i32> {
        self.unlocks.keys().copied().collect()
    }

    /// The Orbis 128-trophy unlock-flag bitset (`OrbisNpTrophyFlagArray`
    /// layout: four little-endian `u32` masks, bit `id % 32` of word
    /// `id / 32` — shadPS4 `ORBIS_NP_TROPHY_FLAG_SET`). Ids outside
    /// `0..128` are ignored.
    pub fn flag_bits(&self) -> [u32; 4] {
        let mut bits = [0u32; 4];
        for &id in self.unlocks.keys() {
            if (0..TROPHY_FLAG_SETSIZE as i32).contains(&id) {
                bits[(id / 32) as usize] |= 1u32 << (id % 32);
            }
        }
        bits
    }

    /// Persist the ledger. Creates the parent directory on demand.
    fn save(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = TrophyStoreFile {
            version: TROPHY_STORE_VERSION,
            unlocks: self.unlocks.clone(),
        };
        let json = serde_json::to_string_pretty(&file).map_err(std::io::Error::other)?;
        std::fs::write(&self.path, json)
    }
}

/// Current wall clock as Unix milliseconds.
pub fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store_path(tag: &str) -> PathBuf {
        std::env::temp_dir()
            .join(format!("raeen-trophies-{tag}-{}", std::process::id()))
            .join("Title-trophies.json")
    }

    #[test]
    fn path_derivation_is_a_sibling_of_the_save_root() {
        let root = Path::new("savedata").join("Minecraft");
        assert_eq!(
            TrophyStore::path_for_savedata_root(&root),
            Path::new("savedata").join("Minecraft-trophies.json")
        );
        // Degenerate rootless path still yields somewhere writable.
        assert_eq!(
            TrophyStore::path_for_savedata_root(Path::new("/")),
            Path::new("/").join("trophies.json")
        );
    }

    #[test]
    fn unlock_round_trips_through_disk() {
        let path = temp_store_path("roundtrip");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());

        let mut store = TrophyStore::load(&path);
        assert_eq!(store.unlocked_count(), 0);
        assert_eq!(store.last_unlock_ms(), None);
        assert_eq!(
            store.unlock(3, 1_700_000_000_000).unwrap(),
            UnlockOutcome::NewlyUnlocked
        );
        assert_eq!(
            store.unlock(7, 1_700_000_000_500).unwrap(),
            UnlockOutcome::NewlyUnlocked
        );

        // A fresh load sees exactly what was persisted.
        let reloaded = TrophyStore::load(&path);
        assert_eq!(reloaded.unlocked_count(), 2);
        assert!(reloaded.is_unlocked(3));
        assert!(reloaded.is_unlocked(7));
        assert!(!reloaded.is_unlocked(4));
        assert_eq!(reloaded.last_unlock_ms(), Some(1_700_000_000_500));
        assert_eq!(reloaded.unlocked_ids(), vec![3, 7]);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn repeat_unlock_is_idempotent_and_keeps_the_original_timestamp() {
        let path = temp_store_path("idempotent");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());

        let mut store = TrophyStore::load(&path);
        assert_eq!(store.unlock(5, 100).unwrap(), UnlockOutcome::NewlyUnlocked);
        assert_eq!(
            store.unlock(5, 999).unwrap(),
            UnlockOutcome::AlreadyUnlocked
        );
        assert_eq!(store.unlocked_count(), 1);
        assert_eq!(TrophyStore::load(&path).last_unlock_ms(), Some(100));

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn corrupt_file_degrades_to_empty_without_error() {
        let path = temp_store_path("corrupt");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{ not json ]").unwrap();

        let mut store = TrophyStore::load(&path);
        assert_eq!(store.unlocked_count(), 0);
        // The store stays writable: the next unlock replaces the bad file.
        store.unlock(1, 42).unwrap();
        assert!(TrophyStore::load(&path).is_unlocked(1));

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn flag_bits_match_the_orbis_flag_array_layout() {
        let path = temp_store_path("bits");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());

        let mut store = TrophyStore::load(&path);
        store.unlock(0, 1).unwrap();
        store.unlock(31, 2).unwrap();
        store.unlock(32, 3).unwrap();
        store.unlock(127, 4).unwrap();
        // Out-of-range ids are ignored by the bitset (still stored by id).
        store.unlock(200, 5).unwrap();
        store.unlock(-3, 6).unwrap();

        let bits = store.flag_bits();
        assert_eq!(bits[0], (1 << 0) | (1 << 31));
        assert_eq!(bits[1], 1 << 0);
        assert_eq!(bits[2], 0);
        assert_eq!(bits[3], 1 << 31);
        assert_eq!(store.unlocked_count(), 6);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
