//! `eframe::App` wiring for the Raeen Shell.
//!
//! Owns the [`shell::Shell`] and drives it one frame at a time. All actual
//! screen logic (boot, Home, Control Center, navigation, animation) lives
//! under `shell/`; this file is intentionally thin.

use crate::launcher::FirmwareLauncher;
use crate::library::{LibraryItem, built_in_apps, sample_library, scan::scan_dir};
use crate::shell::Shell;
use crate::theme::loader::load_theme;
use std::collections::HashSet;
use std::path::PathBuf;

/// Root directory installed Shell themes live under: `themes/<name>/
/// theme.toml` (spec §6, §10 SM2b). Relative, like the default `Games`
/// scan root, so it resolves from wherever the Shell is launched.
const THEMES_ROOT: &str = "themes";

/// Top-level Raeen application state.
pub struct RaeenApp {
    shell: Shell,
}

impl RaeenApp {
    pub fn new(
        ctx: &egui::Context,
        config: raeen_core::config::EmulatorConfig,
        config_path: PathBuf,
    ) -> Self {
        let themes_root = PathBuf::from(THEMES_ROOT);
        // Load whichever theme Settings last selected (SM2a persisted the
        // field; SM2b is what actually resolves it to a `themes/<name>`
        // directory on disk), falling back field-by-field to the in-code
        // default for anything missing, invalid, or not yet installed.
        let theme = load_theme(&themes_root, &config.general.selected_theme);

        // Scan every configured game folder (the same list Settings ▸ Game
        // Folders shows, and whatever the installer's setup page wrote into
        // `config.paths.game_folders`); fall back to the mockup's sample
        // library (with its original gradient art) when nothing is found —
        // covers a fresh checkout with no game folders yet. Either way, the
        // built-in apps (Store, Game Library, Settings) are always appended so
        // Settings stays reachable from the Home rail (spec §10 SM2)
        // regardless of what a real scan turns up.
        //
        // This scan is the single source of truth for the Home library, so it
        // reads the configured folder list rather than a hard-wired `./Games`.
        // A Home scan pinned to one literal path would silently ignore a
        // folder the user (or the installer) added and leave those titles
        // invisible on Home while still listing the folder in Settings.
        let mut library = scan_game_folders(&config.paths.game_folders);
        if library.is_empty() {
            library = sample_library();
        } else {
            library.extend(built_in_apps());
        }

        // SM3: the Shell now hands launches to the real firmware spine
        // (`raeen_firmware::load_module`) instead of `StubLauncher`. It
        // links a selected module — SELF decrypt-or-passthrough -> `.sprx`
        // parse -> dynlibdata decode -> NID link against HLE — but does not
        // yet execute it; see `launcher::FirmwareLauncher`'s docs. Holds no
        // key material of its own, so encrypted retail modules fault with
        // an informative message rather than a crash.
        let launcher = Box::new(FirmwareLauncher::new());

        let mut shell = Shell::new(
            ctx,
            theme,
            themes_root,
            library,
            launcher,
            config,
            config_path,
        );
        // Fire the startup update check here (not in `Shell::new`) so unit
        // tests constructing a Shell never touch the network.
        shell.start_update_check();

        Self { shell }
    }
}

impl eframe::App for RaeenApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Frame boundary for the opt-in puffin profiler (RAEEN_PROFILE=1);
        // a no-op branch when scopes are off.
        puffin::GlobalProfiler::lock().new_frame();
        puffin::profile_scope!("shell_update");
        self.shell.update(ctx);
    }
}

/// Scan every configured game folder into one de-duplicated library list.
///
/// Folders are scanned in order and their results concatenated; a title whose
/// `id` was already seen from an earlier folder is skipped, so overlapping or
/// accidentally repeated folder entries never list the same game twice. An
/// empty folder list — or one whose folders are all missing — yields an empty
/// `Vec`, which the caller reads as "fall back to the sample library".
/// `pub(crate)` because the Shell's Settings ▸ Game Folders ▸ Rescan runs the
/// exact same scan at runtime.
pub(crate) fn scan_game_folders(folders: &[PathBuf]) -> Vec<LibraryItem> {
    let mut items: Vec<LibraryItem> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for folder in folders {
        for item in scan_dir(folder) {
            if seen.insert(item.id.clone()) {
                items.push(item);
            }
        }
    }
    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Create `<root>/<name>/eboot.bin` — the minimum a folder needs for
    /// [`scan_dir`] to classify it as a game (`find_eboot` accepts a flat
    /// `eboot.bin`; `item_from_folder` keys the item `id` off the folder name).
    fn make_game(root: &Path, name: &str) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("mkdir game folder");
        std::fs::write(dir.join("eboot.bin"), b"\x7fELF").expect("write eboot.bin");
    }

    #[test]
    fn scan_game_folders_reads_every_configured_folder() {
        let base = std::env::temp_dir().join(format!("raeen-scan-{}", std::process::id()));
        let lib_a = base.join("libA");
        let lib_b = base.join("libB");
        make_game(&lib_a, "Alpha");
        make_game(&lib_b, "Beta");

        let items = scan_game_folders(&[lib_a, lib_b]);
        let ids: HashSet<&str> = items.iter().map(|i| i.id.as_str()).collect();

        assert!(
            ids.contains("Alpha"),
            "title from the first folder is missing"
        );
        assert!(
            ids.contains("Beta"),
            "title from the second folder is missing"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn scan_game_folders_dedupes_a_folder_listed_twice() {
        let base = std::env::temp_dir().join(format!("raeen-scan-dup-{}", std::process::id()));
        let lib = base.join("lib");
        make_game(&lib, "Solo");

        let once = scan_game_folders(std::slice::from_ref(&lib));
        let twice = scan_game_folders(&[lib.clone(), lib]);

        assert_eq!(
            once.len(),
            twice.len(),
            "the same folder listed twice must not duplicate its titles"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn scan_game_folders_with_no_folders_is_empty() {
        // Empty in → empty out, which the caller turns into the sample library.
        assert!(scan_game_folders(&[]).is_empty());
    }

    #[test]
    fn scan_game_folders_skips_missing_folders() {
        let missing = PathBuf::from("this/folder/does/not/exist/raeen");
        assert!(scan_game_folders(&[missing]).is_empty());
    }
}
