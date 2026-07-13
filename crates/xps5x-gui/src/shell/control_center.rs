//! Control Center overlay (spec §3, screen 3; §10).
//!
//! Summoned by the PS/Guide button (keyboard `C`, gamepad Guide). A bottom
//! row of circular cards; the focused card lifts, turns solid, and shows a
//! structured summary panel above the row. Dims the Home screen behind it.
//!
//! SM1 makes the panels real: Power exposes a selectable option list
//! (Rest Mode / Restart / Turn Off — driven by `nav::NavMode::ControlCenterOption`),
//! Switcher shows recent-titles history, and Sound/Network/Profile/
//! Notifications/Game Base render labeled fields instead of a single line
//! of static text. Home/Music/Microphone/Accessories stay simple status
//! lines — nothing meaningful to make "functional" for them yet (spec §10).

use super::icons::{self, Glyph};
use super::nav::{NavMode, NavState};
use crate::theme::Theme;
use egui::{Align2, Color32, FontId, Pos2, Stroke};

/// What a Control Center card's summary panel renders.
pub enum CcPanelKind {
    /// A single status line (`CcItem::sub`).
    Simple,
    /// Labeled `(field, value)` rows — e.g. Sound's output device + volume.
    Fields(&'static [(&'static str, &'static str)]),
    /// Recently-played titles, most-recent-first (Shell-tracked history).
    Switcher,
    /// A selectable option list, confirmed via `NavMode::ControlCenterOption`.
    Power(&'static [&'static str]),
}

/// One Control Center entry: name, subtitle, glyph, and panel content.
pub struct CcItem {
    pub name: &'static str,
    pub sub: &'static str,
    pub glyph: Glyph,
    pub panel: CcPanelKind,
}

impl CcItem {
    /// Number of selectable options this card's panel exposes (`0` for
    /// display-only panels). Used to build `NavState`'s per-card option
    /// counts so Confirm only drills in where there's something to select.
    pub fn option_count(&self) -> usize {
        match self.panel {
            CcPanelKind::Power(options) => options.len(),
            _ => 0,
        }
    }
}

const POWER_OPTIONS: &[&str] = &["Rest Mode", "Restart", "Turn Off XPS5X"];

/// The fixed Control Center row (spec §3, §10).
pub const ITEMS: &[CcItem] = &[
    CcItem { name: "Home", sub: "Return to the home screen", glyph: Glyph::Home, panel: CcPanelKind::Simple },
    CcItem { name: "Switcher", sub: "Recently played games and apps", glyph: Glyph::Switcher, panel: CcPanelKind::Switcher },
    CcItem {
        name: "Notifications",
        sub: "You're all caught up",
        glyph: Glyph::Bell,
        panel: CcPanelKind::Fields(&[("Unread", "0"), ("Status", "All caught up")]),
    },
    CcItem {
        name: "Game Base",
        sub: "3 friends online",
        glyph: Glyph::Friends,
        panel: CcPanelKind::Fields(&[("Friends online", "3"), ("Party", "Not in a party")]),
    },
    CcItem { name: "Music", sub: "Nothing playing", glyph: Glyph::Music, panel: CcPanelKind::Simple },
    CcItem {
        name: "Sound",
        sub: "Output: TV speakers · 80%",
        glyph: Glyph::Sound,
        panel: CcPanelKind::Fields(&[("Output", "TV Speakers"), ("Volume", "80%")]),
    },
    CcItem { name: "Microphone", sub: "Muted", glyph: Glyph::Mic, panel: CcPanelKind::Simple },
    CcItem { name: "Accessories", sub: "Controller · 92%", glyph: Glyph::Pad, panel: CcPanelKind::Simple },
    CcItem {
        name: "Profile",
        sub: "Player · Level 214",
        glyph: Glyph::Profile,
        panel: CcPanelKind::Fields(&[("Player", "Player One"), ("Level", "214"), ("Trophies", "1,204")]),
    },
    CcItem {
        name: "Network",
        sub: "Connected · 940 Mbps",
        glyph: Glyph::Network,
        panel: CcPanelKind::Fields(&[("Status", "Connected"), ("Speed", "940 Mbps"), ("Type", "Wi-Fi 6")]),
    },
    CcItem { name: "Power", sub: "Rest mode, turn off, restart", glyph: Glyph::Power, panel: CcPanelKind::Power(POWER_OPTIONS) },
];

/// Draw the Control Center overlay at `open_amount` (0.0 closed .. 1.0 fully
/// open), sliding up from the bottom and dimming the screen behind it.
/// `recent_titles` is the Switcher card's most-recent-first history,
/// resolved to display titles by the caller.
pub fn draw(ui: &mut egui::Ui, theme: &Theme, nav: &NavState, open_amount: f32, recent_titles: &[String]) {
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
        let panel_ctx = PanelRenderCtx { open_amount, nav, recent_titles };
        draw_panel_content(painter, theme, Pos2::new(row_start_x, panel_y + 22.0), focused, &panel_ctx);
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

/// The bits of per-frame state a panel needs that aren't the item itself or
/// where to draw it — bundled to keep [`draw_panel_content`]'s argument
/// count reasonable.
#[derive(Clone, Copy)]
struct PanelRenderCtx<'a> {
    open_amount: f32,
    nav: &'a NavState,
    recent_titles: &'a [String],
}

/// Render the focused card's structured panel content starting at
/// `origin`, growing downward.
fn draw_panel_content(painter: &egui::Painter, theme: &Theme, origin: Pos2, item: &CcItem, ctx: &PanelRenderCtx) {
    let PanelRenderCtx { open_amount, nav, recent_titles } = *ctx;
    let x = origin.x;
    let mut y = origin.y;
    match &item.panel {
        CcPanelKind::Simple => {
            painter.text(
                Pos2::new(x, y),
                Align2::LEFT_TOP,
                item.sub,
                FontId::proportional(14.0),
                theme.palette.text_dim.gamma_multiply(open_amount),
            );
        }
        CcPanelKind::Fields(fields) => {
            for (label, value) in *fields {
                painter.text(
                    Pos2::new(x, y),
                    Align2::LEFT_TOP,
                    label.to_uppercase(),
                    FontId::proportional(11.0),
                    theme.palette.text_faint.gamma_multiply(open_amount),
                );
                painter.text(
                    Pos2::new(x + 150.0, y - 1.0),
                    Align2::LEFT_TOP,
                    *value,
                    FontId::proportional(14.0),
                    theme.palette.text.gamma_multiply(open_amount),
                );
                y += 22.0;
            }
        }
        CcPanelKind::Switcher => {
            if recent_titles.is_empty() {
                painter.text(
                    Pos2::new(x, y),
                    Align2::LEFT_TOP,
                    "No recent games.",
                    FontId::proportional(14.0),
                    theme.palette.text_dim.gamma_multiply(open_amount),
                );
            } else {
                for title in recent_titles.iter().take(5) {
                    painter.text(
                        Pos2::new(x, y),
                        Align2::LEFT_TOP,
                        format!("\u{2022} {title}"),
                        FontId::proportional(14.0),
                        theme.palette.text.gamma_multiply(open_amount),
                    );
                    y += 22.0;
                }
            }
        }
        CcPanelKind::Power(options) => {
            for (i, option) in options.iter().enumerate() {
                let selected = nav.mode == NavMode::ControlCenterOption && nav.cc_option_index == i;
                let color = if selected { theme.palette.focus } else { theme.palette.text_dim };
                let prefix = if selected { "\u{25B6} " } else { "   " };
                painter.text(
                    Pos2::new(x, y),
                    Align2::LEFT_TOP,
                    format!("{prefix}{option}"),
                    FontId::proportional(15.0),
                    color.gamma_multiply(open_amount),
                );
                y += 24.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_power_exposes_options() {
        for item in ITEMS {
            if item.name == "Power" {
                assert_eq!(item.option_count(), 3);
            } else {
                assert_eq!(item.option_count(), 0, "card: {}", item.name);
            }
        }
    }

    #[test]
    fn power_options_match_the_expected_labels() {
        let power = ITEMS.iter().find(|i| i.name == "Power").expect("Power card must exist");
        match power.panel {
            CcPanelKind::Power(options) => {
                assert_eq!(options, &["Rest Mode", "Restart", "Turn Off XPS5X"]);
            }
            _ => panic!("expected Power panel kind"),
        }
    }
}
