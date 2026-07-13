//! Control Center overlay (spec §3, screen 3).
//!
//! Summoned by the PS/Guide button (keyboard `C`, gamepad Guide). A bottom
//! row of circular cards; the focused card lifts, turns solid, and shows a
//! summary panel above the row. Dims the Home screen behind it.

use super::icons::{self, Glyph};
use super::nav::NavState;
use crate::theme::Theme;
use egui::{Align2, Color32, FontId, Pos2, Stroke};

/// One Control Center entry: name, subtitle, and its glyph.
pub struct CcItem {
    pub name: &'static str,
    pub sub: &'static str,
    pub glyph: Glyph,
}

/// The fixed SM0 Control Center row (spec §3). Sound/Network/Power etc.
/// become functional in SM1 — for now every card is display-only.
pub const ITEMS: &[CcItem] = &[
    CcItem { name: "Home", sub: "Return to the home screen", glyph: Glyph::Home },
    CcItem { name: "Switcher", sub: "Recently played games and apps", glyph: Glyph::Switcher },
    CcItem { name: "Notifications", sub: "You're all caught up", glyph: Glyph::Bell },
    CcItem { name: "Game Base", sub: "3 friends online", glyph: Glyph::Friends },
    CcItem { name: "Music", sub: "Nothing playing", glyph: Glyph::Music },
    CcItem { name: "Sound", sub: "Output: TV speakers · 80%", glyph: Glyph::Sound },
    CcItem { name: "Microphone", sub: "Muted", glyph: Glyph::Mic },
    CcItem { name: "Accessories", sub: "Controller · 92%", glyph: Glyph::Pad },
    CcItem { name: "Profile", sub: "Player · Level 214", glyph: Glyph::Profile },
    CcItem { name: "Network", sub: "Connected · 940 Mbps", glyph: Glyph::Network },
    CcItem { name: "Power", sub: "Rest mode, turn off, restart", glyph: Glyph::Power },
];

/// Draw the Control Center overlay at `open_amount` (0.0 closed .. 1.0 fully
/// open), sliding up from the bottom and dimming the screen behind it.
pub fn draw(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, open_amount: f32) {
    if open_amount <= 0.0 {
        return;
    }
    let screen = ui.max_rect();
    let painter = ui.painter();

    // Backdrop scrim.
    let scrim_alpha = (theme.palette.cc_scrim.a() as f32 / 255.0) * open_amount;
    painter.rect_filled(
        screen,
        0.0,
        Color32::from_rgba_unmultiplied(
            theme.palette.cc_scrim.r(),
            theme.palette.cc_scrim.g(),
            theme.palette.cc_scrim.b(),
            (scrim_alpha * 255.0) as u8,
        ),
    );

    let item_size = theme.metrics.cc_item_size;
    let gap = theme.metrics.cc_item_gap;
    let content_x = theme.metrics.content_padding_x;
    let bottom_pad = 26.0;

    // Slide the whole overlay up from below the screen as it opens.
    let slide_off = (1.0 - open_amount) * (item_size + 140.0);

    let row_y = screen.bottom() - bottom_pad - item_size + slide_off;
    let row_start_x = content_x;

    // Summary panel above the row.
    if let Some(focused) = ITEMS.get(nav.cc_index) {
        let panel_y = row_y - 40.0;
        painter.text(
            Pos2::new(row_start_x, panel_y),
            Align2::LEFT_BOTTOM,
            focused.name,
            FontId::proportional(26.0),
            theme.palette.text.gamma_multiply(open_amount),
        );
        painter.text(
            Pos2::new(row_start_x, panel_y + 22.0),
            Align2::LEFT_TOP,
            focused.sub,
            FontId::proportional(14.0),
            theme.palette.text_dim.gamma_multiply(open_amount),
        );
    }

    for (i, item) in ITEMS.iter().enumerate() {
        let focused = i == nav.cc_index;
        let x = row_start_x + i as f32 * (item_size + gap) + item_size / 2.0;
        let lift = if focused { 6.0 * open_amount } else { 0.0 };
        let scale = if focused { 1.0 + 0.08 * open_amount } else { 1.0 };
        let radius = item_size / 2.0 * scale;
        let center = Pos2::new(x, row_y + item_size / 2.0 - lift);

        let bg = if focused {
            theme.palette.focus.gamma_multiply(open_amount.max(0.001))
        } else {
            Color32::from_rgba_unmultiplied(20, 29, 41, (178.0 * open_amount) as u8)
        };
        painter.circle_filled(center, radius, bg);
        painter.circle_stroke(center, radius, Stroke::new(1.0, theme.palette.line.gamma_multiply(open_amount)));

        let glyph_color = if focused { theme.palette.ground } else { theme.palette.text_dim.gamma_multiply(open_amount) };
        icons::draw(painter, item.glyph, center, radius * 0.85, glyph_color);

        if focused {
            painter.text(
                Pos2::new(center.x, center.y - radius - 16.0),
                Align2::CENTER_CENTER,
                item.name,
                FontId::proportional(13.0),
                theme.palette.text.gamma_multiply(open_amount),
            );
        }
    }
}
