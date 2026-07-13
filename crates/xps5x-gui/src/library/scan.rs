//! Game-folder scanning.
//!
//! [`item_from_folder`] is the pure classifier: given a folder name, its
//! path, and whether an `eboot.bin` was found inside it, decide whether the
//! folder is a valid game install and build a [`LibraryItem`] for it. It
//! touches no filesystem state, so it's unit-testable without fixtures on
//! disk. [`scan_dir`] is the thin IO wrapper that walks a real directory and
//! calls the classifier for each entry.
//!
//! SM0 scope: folder name → title derivation and eboot presence only. Real
//! package metadata (title id, icon, genre…) lands with the metadata cache
//! in SM1 (spec §9).

use super::{ArtSource, ItemKind, LaunchTarget, LibraryItem};
use std::path::Path;

/// Classify a single game folder. Returns `None` if it isn't a valid game
/// install (no `eboot.bin` found).
pub fn item_from_folder(name: &str, path: &Path, has_eboot: bool) -> Option<LibraryItem> {
    if !has_eboot {
        return None;
    }
    if name.trim().is_empty() {
        return None;
    }

    Some(LibraryItem {
        id: name.to_string(),
        title: derive_title(name),
        kind: ItemKind::Game,
        art: ArtSource::placeholder(),
        meta: None,
        launch: LaunchTarget::Game { path: path.join("eboot.bin") },
    })
}

/// Turn a folder name like `nova-requiem` or `sable_horizon` into a display
/// title like "Nova Requiem" / "Sable Horizon".
fn derive_title(folder_name: &str) -> String {
    folder_name
        .split(['-', '_', ' '])
        .filter(|word| !word.is_empty())
        .map(title_case_word)
        .collect::<Vec<_>>()
        .join(" ")
}

fn title_case_word(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Walk `root` one level deep, classifying each subdirectory as a potential
/// game install. Non-directories, unreadable roots, and folders without an
/// `eboot.bin` are skipped, never fatal (spec §9).
pub fn scan_dir(root: &Path) -> Vec<LibraryItem> {
    let mut items = Vec::new();

    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return items,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let has_eboot = path.join("eboot.bin").is_file();
        if let Some(item) = item_from_folder(name, &path, has_eboot) {
            items.push(item);
        }
    }

    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn folder_with_eboot_becomes_a_game() {
        let path = PathBuf::from("Games/nova-requiem");
        let item = item_from_folder("nova-requiem", &path, true).expect("should classify");
        assert_eq!(item.title, "Nova Requiem");
        assert_eq!(item.kind, ItemKind::Game);
        match item.launch {
            LaunchTarget::Game { path } => assert_eq!(path, PathBuf::from("Games/nova-requiem/eboot.bin")),
            _ => panic!("expected a Game launch target"),
        }
    }

    #[test]
    fn folder_without_eboot_is_skipped() {
        let path = PathBuf::from("Games/not-a-game");
        assert!(item_from_folder("not-a-game", &path, false).is_none());
    }

    #[test]
    fn empty_name_is_skipped_even_with_eboot() {
        let path = PathBuf::from("Games/");
        assert!(item_from_folder("", &path, true).is_none());
        assert!(item_from_folder("   ", &path, true).is_none());
    }

    #[test]
    fn title_derivation_handles_separators_and_casing() {
        assert_eq!(derive_title("sable_horizon"), "Sable Horizon");
        assert_eq!(derive_title("astral-drift"), "Astral Drift");
        assert_eq!(derive_title("kingfall"), "Kingfall");
        assert_eq!(derive_title("neon verge"), "Neon Verge");
    }

    #[test]
    fn scan_dir_on_missing_root_returns_empty_not_panic() {
        let items = scan_dir(&PathBuf::from("this/path/does/not/exist/anywhere"));
        assert!(items.is_empty());
    }
}
