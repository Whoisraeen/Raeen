//! Theme tokens for the XPS5X Shell.
//!
//! A [`Theme`] is a bundle of palette colors and layout metrics — the same
//! tokens the locked visual mockup encodes as CSS custom properties. The
//! Shell's renderer reads *only* these tokens; no colors or sizes are
//! hard-coded in widget code (spec §6).
//!
//! SM0 ships a single in-code [`default_theme`] built entirely from
//! original XPS5X colors. On-disk theme loading (user themes extracted from
//! hardware they own) arrives in SM1 via [`loader`].

pub mod loader;

use egui::Color32;

/// Parse a `0xRRGGBB` literal into a [`Color32`] at full opacity.
const fn rgb(hex: u32) -> Color32 {
    Color32::from_rgb(
        ((hex >> 16) & 0xFF) as u8,
        ((hex >> 8) & 0xFF) as u8,
        (hex & 0xFF) as u8,
    )
}

/// Palette tokens — mirrors the mockup's `:root` custom properties.
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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

/// A fully-resolved theme: palette + metrics.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Reserved for the Settings theme picker (SM2); not yet read by SM0.
    #[allow(dead_code)]
    pub name: String,
    pub palette: Palette,
    pub metrics: Metrics,
}

/// The default, original-asset XPS5X theme (spec §6, §11 — zero Sony assets).
pub fn default_theme() -> Theme {
    Theme {
        name: "XPS5X Default".to_string(),
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
            topbar_padding_x: 54.0,
            topbar_padding_top: 28.0,
            content_padding_x: 54.0,
            content_padding_bottom: 18.0,
            tile_size: 150.0,
            tile_gap: 22.0,
            tile_focus_scale: 1.13,
            tile_focus_lift: 14.0,
            rail_padding_left: 54.0,
            corner_radius: 12.0,
            button_radius: 8.0,
            card_size: egui::vec2(190.0, 104.0),
            card_gap: 14.0,
            cc_item_size: 60.0,
            cc_item_gap: 12.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_theme_exposes_expected_tokens() {
        let theme = default_theme();
        assert_eq!(theme.name, "XPS5X Default");
        assert_eq!(theme.palette.ground, rgb(0x0a1017));
        assert_eq!(theme.palette.text, rgb(0xf3f7fc));
        assert_eq!(theme.palette.accent_hi, rgb(0x57b0ff));
        assert_eq!(theme.metrics.tile_size, 150.0);
        assert_eq!(theme.metrics.rail_padding_left, 54.0);
        assert!(theme.metrics.tile_focus_scale > 1.0);
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
