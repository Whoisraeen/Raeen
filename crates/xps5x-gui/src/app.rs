//! `eframe::App` wiring for the XPS5X Shell.
//!
//! Owns the [`shell::Shell`] and drives it one frame at a time. All actual
//! screen logic (boot, Home, Control Center, navigation, animation) lives
//! under `shell/`; this file is intentionally thin.

use crate::launcher::StubLauncher;
use crate::library::{sample_library, scan::scan_dir};
use crate::shell::Shell;
use crate::theme::loader::load_theme;
use std::path::Path;
use std::time::Duration;

/// Top-level XPS5X application state.
pub struct XPS5XApp {
    shell: Shell,
}

impl XPS5XApp {
    pub fn new(_config: xps5x_core::config::EmulatorConfig) -> Self {
        // SM0 ships only the default theme; `load_theme` is the seam SM2
        // uses to install user themes from `themes/<name>`.
        let theme = load_theme(Path::new("themes/default"));

        // Scan the default game folder; fall back to the mockup's sample
        // library (with its original gradient art) when nothing is found —
        // covers a fresh checkout with no `Games/` folder yet.
        let mut library = scan_dir(Path::new("Games"));
        if library.is_empty() {
            library = sample_library();
        }

        let launcher = Box::new(StubLauncher::new(Duration::from_millis(900)));

        Self { shell: Shell::new(theme, library, launcher) }
    }
}

impl eframe::App for XPS5XApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.shell.update(ctx);
    }
}
