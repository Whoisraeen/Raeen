//! On-disk theme loading.
//!
//! A `Theme` directory is untrusted content once user themes exist (SM1):
//! `theme.toml` plus fonts/icons/sounds/backgrounds extracted from hardware
//! the user owns. The loader must bounds-check everything, execute no code,
//! and decode all images through safe decoders, falling back to
//! [`default_theme`] for anything missing or malformed.
//!
//! SM0 scope: the Shell only ships the default theme, so this is a thin
//! stub that always resolves to it. The `path` parameter is accepted now so
//! callers (Settings, in a later milestone) don't need to change shape when
//! real loading lands.

use super::{Theme, default_theme};
use std::path::Path;

/// Load a theme from `path`, falling back to the default theme.
///
/// SM0: always returns [`default_theme`] — no on-disk parsing yet.
pub fn load_theme(_path: &Path) -> Theme {
    default_theme()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn missing_path_falls_back_to_default() {
        let theme = load_theme(&PathBuf::from("does/not/exist"));
        assert_eq!(theme.name, default_theme().name);
    }
}
