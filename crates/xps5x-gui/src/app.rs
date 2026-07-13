//! `eframe::App` wiring for the XPS5X Shell.
//!
//! Owns the [`shell::Shell`] and drives it one frame at a time. All actual
//! screen logic (boot, Home, Control Center, navigation, animation) lives
//! under `shell/`; this file is intentionally thin.

use crate::launcher::FirmwareLauncher;
use crate::library::{built_in_apps, sample_library, scan::scan_dir};
use crate::shell::Shell;
use crate::theme::loader::load_theme;
use std::path::{Path, PathBuf};

/// Root directory installed Shell themes live under: `themes/<name>/
/// theme.toml` (spec §6, §10 SM2b). Relative, like the default `Games`
/// scan root, so it resolves from wherever the Shell is launched.
const THEMES_ROOT: &str = "themes";

/// Top-level XPS5X application state.
pub struct XPS5XApp {
    shell: Shell,
}

impl XPS5XApp {
    pub fn new(ctx: &egui::Context, config: xps5x_core::config::EmulatorConfig, config_path: PathBuf) -> Self {
        let themes_root = PathBuf::from(THEMES_ROOT);
        // Load whichever theme Settings last selected (SM2a persisted the
        // field; SM2b is what actually resolves it to a `themes/<name>`
        // directory on disk), falling back field-by-field to the in-code
        // default for anything missing, invalid, or not yet installed.
        let theme = load_theme(&themes_root, &config.general.selected_theme);

        // Scan the default game folder; fall back to the mockup's sample
        // library (with its original gradient art) when nothing is found —
        // covers a fresh checkout with no `Games/` folder yet. Either way,
        // the built-in apps (Store, Game Library, Settings) are always
        // appended so Settings stays reachable from the Home rail (spec
        // §10 SM2) regardless of what a real scan turns up.
        let mut library = scan_dir(Path::new("Games"));
        if library.is_empty() {
            library = sample_library();
        } else {
            library.extend(built_in_apps());
        }

        // SM3: the Shell now hands launches to the real firmware spine
        // (`xps5x_firmware::load_module`) instead of `StubLauncher`. It
        // links a selected module — SELF decrypt-or-passthrough -> `.sprx`
        // parse -> dynlibdata decode -> NID link against HLE — but does not
        // yet execute it; see `launcher::FirmwareLauncher`'s docs. Holds no
        // key material of its own, so encrypted retail modules fault with
        // an informative message rather than a crash.
        let launcher = Box::new(FirmwareLauncher::new());

        Self { shell: Shell::new(ctx, theme, themes_root, library, launcher, config, config_path) }
    }
}

impl eframe::App for XPS5XApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.shell.update(ctx);
    }
}
