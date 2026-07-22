//! Theme tokens for the XPS5X Shell.
//!
//! A [`Theme`] is a bundle of palette colors and layout metrics — the same
//! tokens the locked visual mockup encodes as CSS custom properties. The
//! Shell's renderer reads *only* these tokens; no colors or sizes are
//! hard-coded in widget code (spec §6).
//!
//! SM0 ships a single in-code [`default_theme`] built entirely from
//! original XPS5X colors. SM2b adds real on-disk loading via [`loader`]:
//! a theme directory (`themes/<name>/theme.toml` plus optional font/
//! background files a *user* supplies) resolves to a [`Theme`], falling
//! back field-by-field to [`default_theme`] for anything missing or
//! invalid. The repository itself only ever ships `themes/default/
//! theme.toml` — no binary assets (spec §11).

pub mod loader;

use egui::Color32;
use std::sync::Arc;

/// Parse a `0xRRGGBB` literal into a [`Color32`] at full opacity.
const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb(
        ((hex >> 16) & 0xFF) as u8,
        ((hex >> 8) & 0xFF) as u8,
        (hex & 0xFF) as u8,
    )
}

/// Palette tokens — mirrors the mockup's `:root` custom properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Palette {
    /// Base background ("ground").
    pub ground: Color32,
    /// Raised surface (tiles, cards) before hover/focus.
    pub raised: Color32,
    /// Hairline border color.
    pub line: Color32,
    /// Primary text.
    pub text: Color32,
    /// Secondary/dimmed text.
    pub text_dim: Color32,
    /// Faint tertiary text (dots, hints).
    pub text_faint: Color32,
    /// Brand accent.
    pub accent: Color32,
    /// Brighter accent used for highlights (stars, progress bars, kicker).
    pub accent_hi: Color32,
    /// Focus ring color.
    pub focus: Color32,
    /// Legibility scrim drawn over hero art.
    pub scrim: Color32,
    /// Control Center backdrop dim.
    pub cc_scrim: Color32,
}

/// Layout metrics — mirrors the mockup's px values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Top function-bar horizontal padding.
    pub topbar_padding_x: f32,
    /// Top function-bar top padding.
    pub topbar_padding_top: f32,
    /// Home content area horizontal padding.
    pub content_padding_x: f32,
    /// Home content area bottom padding.
    pub content_padding_bottom: f32,
    /// Rail tile edge length (unfocused).
    pub tile_size: f32,
    /// Gap between rail tiles.
    pub tile_gap: f32,
    /// Scale factor applied to the focused tile.
    pub tile_focus_scale: f32,
    /// Vertical lift (px) applied to the focused tile.
    pub tile_focus_lift: f32,
    /// Rail's left padding (anchors the first/focused tile).
    pub rail_padding_left: f32,
    /// Corner radius for tiles and cards.
    pub corner_radius: f32,
    /// Corner radius for pill buttons.
    pub button_radius: f32,
    /// Activity card size.
    pub card_size: egui::Vec2,
    /// Gap between activity cards.
    pub card_gap: f32,
    /// Control Center item (circle) diameter.
    pub cc_item_size: f32,
    /// Gap between Control Center items.
    pub cc_item_gap: f32,
}

/// Optional user-supplied presentation assets loaded from a theme directory
/// (spec §6). Every field is `None` for the in-code default and for any
/// on-disk theme that doesn't provide that particular asset — callers must
/// treat `None` as "use the built-in fallback", never as an error.
///
/// Wrapped in [`Arc`] (rather than owned `Vec`/`ColorImage`) because `Theme`
/// is cloned once per frame (`shell/mod.rs::draw`); an `Arc` clone is a
/// refcount bump, not a multi-megabyte copy.
#[derive(Debug, Clone, Default)]
pub struct ThemeAssets {
    /// Custom UI font bytes, ready for [`egui::FontData::from_owned`].
    /// Installed into the egui context once per theme (re)load via
    /// [`install_fonts`] — widgets never read this directly.
    pub font: Option<Arc<[u8]>>,
    /// Decoded hero/background image. `shell/mod.rs` turns this into an
    /// `egui::TextureHandle` once per theme (re)load; `home.rs` draws the
    /// handle, never the raw pixels.
    pub background: Option<Arc<egui::ColorImage>>,
}

/// A fully-resolved theme: palette + metrics + optional user assets.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Display name (from the theme's manifest, or the in-code default's).
    /// Not currently rendered anywhere in the Shell UI — Settings' theme
    /// selector shows the directory/id instead — but kept for a future
    /// theme-picker preview.
    #[allow(dead_code)]
    pub name: String,
    pub palette: Palette,
    pub metrics: Metrics,
    pub assets: ThemeAssets,
}

/// Install `theme`'s custom UI font (if any) into `ctx` as the primary
/// proportional/monospace family, falling back to egui's built-in fonts for
/// any glyph the custom font doesn't cover — and, when the theme carries no
/// font at all, resetting `ctx` back to those built-ins entirely. Call once
/// per theme (re)load, never per-frame.
pub fn install_fonts(ctx: &egui::Context, theme: &Theme) {
    let mut fonts = egui::FontDefinitions::default();
    if let Some(font_bytes) = &theme.assets.font {
        let name = "xps5x-theme-font".to_string();
        let data = egui::FontData::from_owned(font_bytes.to_vec());
        fonts.font_data.insert(name.clone(), Arc::new(data));
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, name.clone());
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .insert(0, name);
    } else {
        // No theme font: prefer the host's native UI fonts over egui's
        // bundled ones. The bundled proportional face is a LIGHT weight —
        // thin strokes plus grayscale AA read as blurry at 1:1 DPI (user
        // report, measured 96-DPI panel). Segoe UI regular and Consolas are
        // designed for on-screen legibility at Shell sizes. Missing files
        // (non-Windows hosts, trimmed installs) keep the built-ins; the
        // built-ins also remain as fallback for uncovered glyphs.
        #[cfg(windows)]
        for (file, name, family) in [
            (
                "C:/Windows/Fonts/segoeui.ttf",
                "segoe-ui",
                egui::FontFamily::Proportional,
            ),
            (
                "C:/Windows/Fonts/consola.ttf",
                "consolas",
                egui::FontFamily::Monospace,
            ),
        ] {
            if let Ok(bytes) = std::fs::read(file) {
                fonts.font_data.insert(
                    name.to_string(),
                    Arc::new(egui::FontData::from_owned(bytes)),
                );
                fonts
                    .families
                    .entry(family)
                    .or_default()
                    .insert(0, name.to_string());
            }
        }
    }
    ctx.set_fonts(fonts);
}

/// The default, original-asset XPS5X theme (spec §6, §11 — zero Sony assets).
pub fn default_theme() -> Theme {
    Theme {
        name: "Raeen Default".to_string(),
        palette: Palette {
            ground: rgb(0x0a1017),
            raised: rgb(0x141d29),
            line: Color32::from_rgba_premultiplied(255, 255, 255, 26),
            text: rgb(0xf3f7fc),
            text_dim: rgb(0xa7b6c8),
            text_faint: rgb(0x6b7c90),
            accent: rgb(0x1f8fff),
            accent_hi: rgb(0x57b0ff),
            focus: rgb(0xffffff),
            scrim: Color32::from_rgba_premultiplied(6, 10, 16, 235),
            cc_scrim: Color32::from_rgba_premultiplied(4, 7, 12, 140),
        },
        metrics: Metrics {
            topbar_padding_x: 88.0,
            topbar_padding_top: 44.0,
            content_padding_x: 88.0,
            content_padding_bottom: 44.0,
            tile_size: 224.0,
            tile_gap: 18.0,
            tile_focus_scale: 1.42,
            tile_focus_lift: 0.0,
            rail_padding_left: 88.0,
            corner_radius: 16.0,
            button_radius: 8.0,
            card_size: egui::vec2(250.0, 140.0),
            card_gap: 16.0,
            cc_item_size: 56.0,
            cc_item_gap: 14.0,
        },
        assets: ThemeAssets {
            font: None,
            background: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_theme_exposes_expected_tokens() {
        let theme = default_theme();
        assert_eq!(theme.name, "Raeen Default");
        assert_eq!(theme.palette.ground, rgb(0x0a1017));
        assert_eq!(theme.palette.text, rgb(0xf3f7fc));
        assert_eq!(theme.palette.accent_hi, rgb(0x57b0ff));
        assert_eq!(theme.metrics.tile_size, 224.0);
        assert_eq!(theme.metrics.rail_padding_left, 88.0);
        assert!(theme.metrics.tile_focus_scale > 1.0);
        assert!(theme.assets.font.is_none());
        assert!(theme.assets.background.is_none());
    }

    #[test]
    fn rgb_helper_decodes_channels() {
        let c = rgb(0x1f8fff);
        assert_eq!(c.r(), 0x1f);
        assert_eq!(c.g(), 0x8f);
        assert_eq!(c.b(), 0xff);
        assert_eq!(c.a(), 255);
    }
}
