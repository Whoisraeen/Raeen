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
use egui::{Align2, Color32, FontId, Mesh, Pos2, Rect, Shape, Stroke, StrokeKind, vec2};

/// What a Control Center card's summary panel renders.
pub enum CcPanelKind {
    /// A single status line (`CcItem::sub`, or its live override).
    Simple,
    /// Recently-played titles, most-recent-first (Shell-tracked history).
    Switcher,
    /// A selectable option list, confirmed via `NavMode::ControlCenterOption`.
    Power(&'static [&'static str]),
}

/// Live values the Shell computes per frame for cards whose status is real
/// data (volume, connected controller) rather than a static string.
#[derive(Default)]
pub struct CcLive {
    /// Sound card status ("Host output · 80%", "Muted").
    pub sound: String,
    /// Accessories card status ("Xbox Wireless Controller", "No controller
    /// connected").
    pub accessories: String,
}

/// A card's effective status line: the live override when the card carries
/// real data, else its static subtitle.
fn effective_sub<'a>(item: &'a CcItem, live: &'a CcLive) -> &'a str {
    match item.name {
        "Sound" if !live.sound.is_empty() => &live.sound,
        "Accessories" if !live.accessories.is_empty() => &live.accessories,
        _ => item.sub,
    }
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
    CcItem {
        name: "Home",
        sub: "Return to the home screen",
        glyph: Glyph::Home,
        panel: CcPanelKind::Simple,
    },
    CcItem {
        name: "Switcher",
        sub: "Recently played games and apps",
        glyph: Glyph::Switcher,
        panel: CcPanelKind::Switcher,
    },
    // Only cards backed by something real: Sound and Accessories get live
    // values from the Shell (CcLive); the fictional Notifications / Game
    // Base / Music / Microphone / Profile / Network cards are gone on the
    // same principle that removed the dead Store tile.
    CcItem {
        name: "Sound",
        sub: "",
        glyph: Glyph::Sound,
        panel: CcPanelKind::Simple,
    },
    CcItem {
        name: "Accessories",
        sub: "",
        glyph: Glyph::Pad,
        panel: CcPanelKind::Simple,
    },
    CcItem {
        name: "Power",
        sub: "Rest mode, turn off, restart",
        glyph: Glyph::Power,
        panel: CcPanelKind::Power(POWER_OPTIONS),
    },
];

/// Draw the Control Center overlay at `open_amount` (0.0 closed .. 1.0 fully
/// open), sliding up from the bottom and dimming the screen behind it.
/// `recent_titles` is the Switcher card's most-recent-first history,
/// resolved to display titles by the caller.
pub fn draw(
    ui: &mut egui::Ui,
    theme: &Theme,
    nav: &NavState,
    open_amount: f32,
    recent_titles: &[String],
    live: &CcLive,
) {
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
    let bottom_pad = 30.0;

    // Bottom gradient so the icon row reads against any hero art.
    let ground = theme.palette.ground;
    let grad_rect = Rect::from_min_max(
        Pos2::new(screen.left(), screen.bottom() - 260.0),
        screen.max,
    );
    vertical_gradient(
        painter,
        grad_rect,
        Color32::TRANSPARENT,
        Color32::from_rgba_unmultiplied(
            ground.r(),
            ground.g(),
            ground.b(),
            (225.0 * open_amount) as u8,
        ),
    );

    // Slide the whole overlay up from below the screen as it opens.
    let slide_off = (1.0 - open_amount) * (item_size + 140.0);

    let row_y = screen.bottom() - bottom_pad - item_size + slide_off;
    // Icon row centered on the screen, like the console's control center.
    let total_w = ITEMS.len() as f32 * item_size + (ITEMS.len() as f32 - 1.0).max(0.0) * gap;
    let row_start_x = screen.center().x - total_w / 2.0;

    // Focused card's summary panel: a rounded card floating above the row,
    // anchored near the focused icon but clamped to the content margins.
    if let Some(focused) = ITEMS.get(nav.cc_index) {
        let panel_ctx = PanelRenderCtx {
            open_amount,
            nav,
            recent_titles,
            live,
        };
        let focused_cx = row_start_x + nav.cc_index as f32 * (item_size + gap) + item_size / 2.0;
        draw_panel_card(
            painter, theme, screen, focused_cx, row_y, content_x, focused, &panel_ctx,
        );
    }

    for (i, item) in ITEMS.iter().enumerate() {
        let focused = i == nav.cc_index;
        let x = row_start_x + i as f32 * (item_size + gap) + item_size / 2.0;
        let lift = if focused { 6.0 * open_amount } else { 0.0 };
        let scale = if focused {
            1.0 + 0.08 * open_amount
        } else {
            1.0
        };
        let radius = item_size / 2.0 * scale;
        let center = Pos2::new(x, row_y + item_size / 2.0 - lift);

        let bg = if focused {
            theme.palette.focus.gamma_multiply(open_amount.max(0.001))
        } else {
            Color32::from_rgba_unmultiplied(20, 29, 41, (178.0 * open_amount) as u8)
        };
        painter.circle_filled(center, radius, bg);
        painter.circle_stroke(
            center,
            radius,
            Stroke::new(1.0f32, theme.palette.line.gamma_multiply(open_amount)),
        );

        let glyph_color = if focused {
            theme.palette.ground
        } else {
            theme.palette.text_dim.gamma_multiply(open_amount)
        };
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
    live: &'a CcLive,
}

/// Number of content lines the focused card's panel renders — sizes the
/// floating card before [`draw_panel_content`] fills it in.
fn panel_line_count(item: &CcItem, recent_titles: &[String]) -> usize {
    match &item.panel {
        CcPanelKind::Simple => 1,
        CcPanelKind::Switcher => recent_titles.len().clamp(1, 5),
        CcPanelKind::Power(options) => options.len(),
    }
}

/// The focused card's summary panel: a rounded card floating above the icon
/// row, horizontally anchored near the focused icon.
#[allow(clippy::too_many_arguments)]
fn draw_panel_card(
    painter: &egui::Painter,
    theme: &Theme,
    screen: Rect,
    focused_cx: f32,
    row_y: f32,
    margin_x: f32,
    item: &CcItem,
    ctx: &PanelRenderCtx,
) {
    const PANEL_W: f32 = 380.0;
    const PAD: f32 = 20.0;
    const TITLE_H: f32 = 32.0;
    const LINE_H: f32 = 24.0;

    let open = ctx.open_amount;
    let h = PAD * 2.0 + TITLE_H + panel_line_count(item, ctx.recent_titles) as f32 * LINE_H;
    let x = (focused_cx - PANEL_W / 2.0).clamp(
        margin_x,
        (screen.right() - margin_x - PANEL_W).max(margin_x),
    );
    let rect = Rect::from_min_size(Pos2::new(x, row_y - 46.0 - h), vec2(PANEL_W, h));

    painter.rect_filled(
        rect,
        18.0,
        Color32::from_rgba_unmultiplied(16, 23, 32, (242.0 * open) as u8),
    );
    painter.rect_stroke(
        rect,
        18.0,
        Stroke::new(1.0f32, theme.palette.line.gamma_multiply(open)),
        StrokeKind::Inside,
    );

    painter.text(
        rect.min + vec2(PAD, PAD),
        Align2::LEFT_TOP,
        item.name,
        FontId::proportional(20.0),
        theme.palette.text.gamma_multiply(open),
    );
    draw_panel_content(
        painter,
        theme,
        Pos2::new(rect.left() + PAD, rect.top() + PAD + TITLE_H),
        item,
        ctx,
    );
}

/// Vertical two-stop gradient (egui has no CSS gradients, so build a
/// two-triangle mesh with per-vertex colors).
fn vertical_gradient(painter: &egui::Painter, rect: Rect, top: Color32, bottom: Color32) {
    let mut mesh = Mesh::default();
    mesh.colored_vertex(rect.left_top(), top);
    mesh.colored_vertex(rect.right_top(), top);
    mesh.colored_vertex(rect.left_bottom(), bottom);
    mesh.colored_vertex(rect.right_bottom(), bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 2, 3);
    painter.add(Shape::mesh(mesh));
}

/// Render the focused card's structured panel content starting at
/// `origin`, growing downward.
fn draw_panel_content(
    painter: &egui::Painter,
    theme: &Theme,
    origin: Pos2,
    item: &CcItem,
    ctx: &PanelRenderCtx,
) {
    let PanelRenderCtx {
        open_amount,
        nav,
        recent_titles,
        live,
    } = *ctx;
    let x = origin.x;
    let mut y = origin.y;
    match &item.panel {
        CcPanelKind::Simple => {
            painter.text(
                Pos2::new(x, y),
                Align2::LEFT_TOP,
                effective_sub(item, live),
                FontId::proportional(14.0),
                theme.palette.text_dim.gamma_multiply(open_amount),
            );
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
                let color = if selected {
                    theme.palette.focus
                } else {
                    theme.palette.text_dim
                };
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
        let power = ITEMS
            .iter()
            .find(|i| i.name == "Power")
            .expect("Power card must exist");
        match power.panel {
            CcPanelKind::Power(options) => {
                assert_eq!(options, &["Rest Mode", "Restart", "Turn Off XPS5X"]);
            }
            _ => panic!("expected Power panel kind"),
        }
    }
}
