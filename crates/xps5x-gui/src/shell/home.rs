//! Home screen: function bar, hero background, context block, tile rail
//! (spec §3, screen 2).
//!
//! Pure rendering — all animated values (rail slide, hero crossfade, focus
//! pop) are resolved by `shell/mod.rs` before calling [`draw`], so this
//! module stays declarative and every color/metric comes from the active
//! [`Theme`].

use super::anim::lerp_color;
use super::icons::{self, Glyph};
use super::nav::{NavState, RailTab};
use crate::library::{ArtSource, GlyphKind, Gradient, ItemKind, LibraryItem, MetaCache, TileGradient};
use crate::theme::Theme;
use egui::{Align, Color32, Layout, Mesh, Pos2, Rect, RichText, Sense, Shape, Stroke, StrokeKind, UiBuilder, vec2};

/// Animated values resolved once per frame by `shell/mod.rs`.
pub struct HomeAnim {
    /// Current x-shift (px) applied to every rail tile.
    pub rail_offset: f32,
    /// Current (possibly crossfading) hero gradient.
    pub hero: Gradient,
    /// 0.0 (just changed focus) .. 1.0 (settled) — eases the focused tile's
    /// scale/lift in rather than snapping it.
    pub focus_pop: f32,
}

pub fn draw(
    ui: &mut egui::Ui,
    theme: &Theme,
    items: &[LibraryItem],
    nav: &NavState,
    anim: &HomeAnim,
    meta_cache: &MetaCache,
    background: Option<&egui::TextureHandle>,
) {
    let screen = ui.max_rect();
    let painter = ui.painter().clone();

    painter.rect_filled(screen, 0.0, theme.palette.ground);
    draw_hero(&painter, screen, theme, &anim.hero, background);

    let topbar_h = 96.0;
    let topbar_rect = Rect::from_min_size(screen.min, vec2(screen.width(), topbar_h));
    draw_topbar(ui, theme, topbar_rect, nav.tab);

    let rail_h = theme.metrics.tile_size * theme.metrics.tile_focus_scale
        + theme.metrics.tile_focus_lift
        + 34.0
        + 36.0;
    let rail_rect = Rect::from_min_max(Pos2::new(screen.left(), screen.bottom() - rail_h), screen.max);

    let content_rect = Rect::from_min_max(Pos2::new(screen.left(), topbar_rect.bottom()), Pos2::new(screen.right(), rail_rect.top()));

    let focused = items.get(nav.rail_index);
    draw_context_block(ui, theme, content_rect, focused, meta_cache);
    draw_rail(ui, theme, rail_rect, items, nav, anim);
    draw_hints(&painter, theme, screen);
}

/// Paint the Home hero. When the active theme provides a background image
/// (spec §6 — a user's own local theme), that image is drawn stretched to
/// fill `rect` instead of the original mesh-gradient art; either way, the
/// legibility scrim on top is unconditional so context-block text stays
/// readable.
fn draw_hero(painter: &egui::Painter, rect: Rect, theme: &Theme, g: &Gradient, background: Option<&egui::TextureHandle>) {
    match background {
        Some(texture) => {
            let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
            painter.image(texture.id(), rect, uv, Color32::WHITE);
        }
        None => {
            // Approximate the mockup's upper-right radial glow with a 4-corner mesh.
            let top_right = g.hi;
            let top_left = lerp_color(g.hi, g.mid, 0.55);
            let bottom_right = lerp_color(g.mid, g.lo, 0.4);
            let bottom_left = g.lo;
            corner_gradient(painter, rect, top_left, top_right, bottom_left, bottom_right);
        }
    }

    // Legibility scrim: darker toward the bottom-left where text sits.
    let s = theme.palette.scrim;
    let scrim_tl = with_alpha(s, 0.55);
    let scrim_tr = with_alpha(s, 0.04);
    let scrim_bl = with_alpha(s, 0.92);
    let scrim_br = with_alpha(s, 0.5);
    corner_gradient(painter, rect, scrim_tl, scrim_tr, scrim_bl, scrim_br);
}

fn with_alpha(c: Color32, a: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (a.clamp(0.0, 1.0) * 255.0) as u8)
}

/// Paint a rect filled with a bilinear 4-corner gradient. egui has no CSS
/// gradients, so this builds a two-triangle mesh with per-vertex colors.
fn corner_gradient(painter: &egui::Painter, rect: Rect, top_left: Color32, top_right: Color32, bottom_left: Color32, bottom_right: Color32) {
    let mut mesh = Mesh::default();
    let i0 = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.left_top(), top_left);
    mesh.colored_vertex(rect.right_top(), top_right);
    mesh.colored_vertex(rect.left_bottom(), bottom_left);
    mesh.colored_vertex(rect.right_bottom(), bottom_right);
    mesh.add_triangle(i0, i0 + 1, i0 + 2);
    mesh.add_triangle(i0 + 1, i0 + 2, i0 + 3);
    painter.add(Shape::mesh(mesh));
}

fn tile_gradient_rect(painter: &egui::Painter, rect: Rect, rounding: f32, g: &TileGradient) {
    let mid = lerp_color(g.from, g.to, 0.5);
    corner_gradient_rounded(painter, rect, rounding, g.from, mid, mid, g.to);
}

/// Same as [`corner_gradient`] but clipped to a rounded-rect mask by simply
/// painting the gradient then re-cutting the corners with the background —
/// cheaper: we approximate by painting a rounded rect fill first (base
/// color) and layering the gradient mesh with the same rounding via clip.
fn corner_gradient_rounded(painter: &egui::Painter, rect: Rect, rounding: f32, top_left: Color32, top_right: Color32, bottom_left: Color32, bottom_right: Color32) {
    let clip = painter.clip_rect();
    let clipped = painter.with_clip_rect(rect.intersect(clip));
    // Rounded base so corners outside the rounded silhouette stay transparent-looking
    // against whatever is already painted behind (the rail background).
    clipped.rect_filled(rect, rounding, bottom_left);
    corner_gradient(&clipped, rect, top_left, top_right, bottom_left, bottom_right);
    clipped.rect_stroke(rect, rounding, Stroke::NONE, StrokeKind::Inside);
}

fn draw_topbar(ui: &mut egui::Ui, theme: &Theme, rect: Rect, active_tab: RailTab) {
    ui.scope_builder(UiBuilder::new().max_rect(rect), |ui| {
        ui.add_space(theme.metrics.topbar_padding_top);
        ui.horizontal(|ui| {
            ui.add_space(theme.metrics.topbar_padding_x);
            let games_color = if active_tab == RailTab::Games { theme.palette.text } else { theme.palette.text_faint };
            let media_color = if active_tab == RailTab::Media { theme.palette.text } else { theme.palette.text_faint };
            ui.label(RichText::new("Games").color(games_color).size(21.0).strong());
            ui.add_space(30.0);
            ui.label(RichText::new("Media").color(media_color).size(21.0).strong());

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_space(theme.metrics.topbar_padding_x);

                // Clock (right-most).
                let (time_s, date_s) = current_clock_strings();
                ui.vertical(|ui| {
                    ui.label(RichText::new(time_s).color(theme.palette.text).size(20.0).strong());
                    ui.label(RichText::new(date_s).color(theme.palette.text_dim).size(12.0));
                });
                ui.add_space(10.0);

                // Avatar.
                let (rect, _resp) = ui.allocate_exact_size(vec2(40.0, 40.0), Sense::hover());
                ui.painter().circle_filled(rect.center(), 20.0, theme.palette.accent);
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "X5", egui::FontId::proportional(14.0), theme.palette.text);
                ui.add_space(14.0);

                for glyph in [Glyph::Gear, Glyph::Bell, Glyph::Friends, Glyph::Search] {
                    icon_button(ui, theme, glyph);
                    ui.add_space(8.0);
                }
            });
        });
    });
}

fn icon_button(ui: &mut egui::Ui, theme: &Theme, glyph: Glyph) {
    let (rect, resp) = ui.allocate_exact_size(vec2(40.0, 40.0), Sense::hover());
    let bg = if resp.hovered() { Color32::from_rgba_unmultiplied(255, 255, 255, 30) } else { Color32::TRANSPARENT };
    ui.painter().circle_filled(rect.center(), 20.0, bg);
    icons::draw(ui.painter(), glyph, rect.center(), 22.0, theme.palette.text_dim);
}

fn draw_context_block(ui: &mut egui::Ui, theme: &Theme, rect: Rect, focused: Option<&LibraryItem>, meta_cache: &MetaCache) {
    let Some(item) = focused else { return };
    // Home reads a focused item's metadata from the cache rather than the
    // item directly, so rendering stays decoupled from how (or whether)
    // metadata was sourced (parsed `xps5x-title.toml`, or none) — spec §4.
    let meta = meta_cache.get(item.id.as_str());
    let max_width = rect.width().min(660.0);
    let text_rect = Rect::from_min_size(rect.min, vec2(max_width, rect.height()));

    ui.scope_builder(
        UiBuilder::new().max_rect(text_rect).layout(Layout::bottom_up(Align::Min)),
        |ui| {
            ui.add_space(theme.metrics.content_padding_bottom);

            // Activity cards (bottom-most element added first under bottom_up).
            // A game with no metadata renders no cards — just title + Play.
            if let Some(meta) = meta {
                ui.horizontal(|ui| {
                    ui.add_space(theme.metrics.content_padding_x);
                    for card in &meta.activity {
                        draw_activity_card(ui, theme, card);
                        ui.add_space(theme.metrics.card_gap);
                    }
                });
                ui.add_space(20.0);
            }

            // Actions row.
            ui.horizontal(|ui| {
                ui.add_space(theme.metrics.content_padding_x);
                play_button(ui, theme);
                ui.add_space(14.0);
                if meta.is_some() {
                    ghost_button(ui, theme, "•••  More");
                }
            });
            ui.add_space(20.0);

            // Meta row (rating/genre/players) or system subtitle.
            ui.horizontal(|ui| {
                ui.add_space(theme.metrics.content_padding_x);
                match meta {
                    Some(meta) => {
                        ui.label(RichText::new(stars(meta.rating)).color(theme.palette.accent_hi).size(14.0));
                        dot(ui, theme);
                        ui.label(RichText::new(&meta.genre).color(theme.palette.text_dim).size(14.0));
                        dot(ui, theme);
                        ui.label(RichText::new(&meta.players).color(theme.palette.text_dim).size(14.0));
                    }
                    None => {
                        ui.label(RichText::new(format!("Open {}", item.title)).color(theme.palette.text_dim).size(14.0));
                    }
                }
            });
            ui.add_space(12.0);

            // Wordmark.
            ui.horizontal(|ui| {
                ui.add_space(theme.metrics.content_padding_x);
                ui.label(RichText::new(&item.title).color(theme.palette.text).size(58.0).strong());
            });
            ui.add_space(10.0);

            // Kicker (top-most element, added last under bottom_up).
            ui.horizontal(|ui| {
                ui.add_space(theme.metrics.content_padding_x);
                let kicker = meta.map(|m| m.kicker.as_str()).unwrap_or("System");
                ui.label(
                    RichText::new(kicker.to_uppercase())
                        .color(theme.palette.accent_hi)
                        .size(12.0)
                        .strong(),
                );
            });
        },
    );
}

fn dot(ui: &mut egui::Ui, theme: &Theme) {
    let (rect, _) = ui.allocate_exact_size(vec2(10.0, 14.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 1.5, theme.palette.text_faint);
}

fn stars(rating: u8) -> String {
    let filled = rating.min(5) as usize;
    format!("{}{}", "\u{2605}".repeat(filled), "\u{2606}".repeat(5 - filled))
}

fn play_button(ui: &mut egui::Ui, theme: &Theme) {
    let (rect, _resp) = ui.allocate_exact_size(vec2(120.0, 48.0), Sense::hover());
    ui.painter().rect_filled(rect, theme.metrics.button_radius, theme.palette.focus);
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "\u{25B6}  Play", egui::FontId::proportional(16.0), theme.palette.ground);
}

fn ghost_button(ui: &mut egui::Ui, theme: &Theme, label: &str) {
    let galley = ui.painter().layout_no_wrap(label.to_string(), egui::FontId::proportional(16.0), theme.palette.text);
    let w = galley.size().x + 40.0;
    let (rect, _resp) = ui.allocate_exact_size(vec2(w, 48.0), Sense::hover());
    ui.painter().rect_filled(rect, theme.metrics.button_radius, Color32::from_rgba_unmultiplied(255, 255, 255, 20));
    ui.painter().rect_stroke(rect, theme.metrics.button_radius, Stroke::new(1.0, theme.palette.line), StrokeKind::Inside);
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(16.0), theme.palette.text);
}

fn draw_activity_card(ui: &mut egui::Ui, theme: &Theme, card: &crate::library::ActivityCard) {
    let size = theme.metrics.card_size;
    let (rect, _resp) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter().rect_filled(rect, theme.metrics.corner_radius, Color32::from_rgba_unmultiplied(20, 29, 41, 184));
    ui.painter().rect_stroke(rect, theme.metrics.corner_radius, Stroke::new(1.0, theme.palette.line), StrokeKind::Inside);

    let pad = 14.0;
    ui.painter().text(
        rect.min + vec2(pad, pad),
        egui::Align2::LEFT_TOP,
        card.top.to_uppercase(),
        egui::FontId::proportional(11.0),
        theme.palette.text_faint,
    );
    ui.painter().text(
        rect.min + vec2(pad, pad + 22.0),
        egui::Align2::LEFT_TOP,
        &card.main,
        egui::FontId::proportional(16.0),
        theme.palette.text,
    );
    if !card.sub.is_empty() {
        ui.painter().text(
            rect.min + vec2(pad, pad + 42.0),
            egui::Align2::LEFT_TOP,
            &card.sub,
            egui::FontId::proportional(12.0),
            theme.palette.text_dim,
        );
    }
    if let Some(progress) = card.progress {
        let bar_rect = Rect::from_min_size(rect.min + vec2(pad, size.y - pad - 5.0), vec2(size.x - pad * 2.0, 5.0));
        ui.painter().rect_filled(bar_rect, 2.5, Color32::from_rgba_unmultiplied(255, 255, 255, 30));
        let filled_w = bar_rect.width() * (progress as f32 / 100.0).clamp(0.0, 1.0);
        let filled_rect = Rect::from_min_size(bar_rect.min, vec2(filled_w, bar_rect.height()));
        ui.painter().rect_filled(filled_rect, 2.5, theme.palette.accent_hi);
    }
}

fn draw_rail(ui: &mut egui::Ui, theme: &Theme, rect: Rect, items: &[LibraryItem], nav: &NavState, anim: &HomeAnim) {
    let painter = ui.painter_at(rect);
    let m = &theme.metrics;
    let step = m.tile_size + m.tile_gap;
    let base_y = rect.top() + 40.0;

    for (i, item) in items.iter().enumerate() {
        let focused = i == nav.rail_index;
        let base_x = rect.left() + m.rail_padding_left + i as f32 * step + anim.rail_offset;

        if base_x + m.tile_size < rect.left() - 40.0 || base_x > rect.right() + 40.0 {
            continue; // off-screen, skip
        }

        let scale = if focused { 1.0 + (m.tile_focus_scale - 1.0) * anim.focus_pop } else { 1.0 };
        let lift = if focused { m.tile_focus_lift * anim.focus_pop } else { 0.0 };
        let size = m.tile_size * scale;
        let center_x = base_x + m.tile_size / 2.0;
        let bottom_y = base_y + m.tile_size - lift;
        let tile_rect = Rect::from_min_size(Pos2::new(center_x - size / 2.0, bottom_y - size), vec2(size, size));

        let tile_gradient = item.art.tile();
        tile_gradient_rect(&painter, tile_rect, m.corner_radius, &tile_gradient);

        match &item.art {
            ArtSource::App { glyph, .. } => {
                let g = match glyph {
                    GlyphKind::Bag => Glyph::Bag,
                    GlyphKind::Grid => Glyph::Grid,
                    GlyphKind::Gear => Glyph::Gear,
                    GlyphKind::Music => Glyph::Music,
                    GlyphKind::Video => Glyph::Video,
                    GlyphKind::Network => Glyph::Network,
                };
                icons::draw(&painter, g, tile_rect.center(), size * 0.32, Color32::from_rgba_unmultiplied(255, 255, 255, 235));
            }
            ArtSource::Game { .. } => {
                if item.kind == ItemKind::Game {
                    let initial = item.title.split_whitespace().next().unwrap_or(&item.title);
                    painter.text(
                        tile_rect.min + vec2(14.0, size - 22.0),
                        egui::Align2::LEFT_TOP,
                        initial,
                        egui::FontId::proportional(15.0),
                        Color32::WHITE,
                    );
                }
            }
        }

        if focused {
            painter.rect_stroke(tile_rect, m.corner_radius, Stroke::new(3.0, theme.palette.focus), StrokeKind::Inside);
            painter.text(
                Pos2::new(tile_rect.center().x, tile_rect.top() - 18.0),
                egui::Align2::CENTER_CENTER,
                &item.title,
                egui::FontId::proportional(14.0),
                theme.palette.text,
            );
        } else {
            painter.rect_stroke(tile_rect, m.corner_radius, Stroke::new(1.0, Color32::from_rgba_unmultiplied(0, 0, 0, 90)), StrokeKind::Inside);
        }
    }
}

fn draw_hints(painter: &egui::Painter, theme: &Theme, screen: Rect) {
    let hints = "\u{25C0} \u{25B6} Navigate    Enter Play    Tab Media/Games    C Control Center";
    painter.text(
        Pos2::new(screen.right() - theme.metrics.content_padding_x, screen.bottom() - 18.0),
        egui::Align2::RIGHT_BOTTOM,
        hints,
        egui::FontId::proportional(13.0),
        theme.palette.text_dim,
    );
}

/// UTC-based clock/date strings. No timezone/calendar crate is in the
/// workspace, so this uses a small epoch→civil-date conversion rather than
/// pulling in a new dependency for a cosmetic clock (spec §11 keeps deps to
/// the existing workspace set).
fn current_clock_strings() -> (String, String) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = (secs / 86400) as i64;
    let sec_of_day = secs % 86400;
    let hour24 = sec_of_day / 3600;
    let minute = (sec_of_day % 3600) / 60;
    let (h12, ampm) = match hour24 {
        0 => (12, "AM"),
        1..=11 => (hour24, "AM"),
        12 => (12, "PM"),
        _ => (hour24 - 12, "PM"),
    };
    let time_s = format!("{h12}:{minute:02} {ampm}");

    let (_year, _month, _day) = civil_from_days(days);
    let weekday = WEEKDAYS[(days.rem_euclid(7)) as usize];
    let month_name = MONTHS[(_month - 1) as usize];
    let date_s = format!("{weekday}, {month_name} {_day}");
    (time_s, date_s)
}

const WEEKDAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/// Howard Hinnant's `civil_from_days`: days-since-epoch (1970-01-01) to a
/// proleptic Gregorian (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_from_days_epoch_is_1970_01_01() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }

    #[test]
    fn civil_from_days_known_date() {
        // 2000-03-01 is day 11017 since the epoch.
        assert_eq!(civil_from_days(11017), (2000, 3, 1));
    }

    #[test]
    fn stars_renders_expected_glyph_counts() {
        assert_eq!(stars(0).chars().filter(|c| *c == '\u{2605}').count(), 0);
        assert_eq!(stars(5).chars().filter(|c| *c == '\u{2605}').count(), 5);
        assert_eq!(stars(3).chars().filter(|c| *c == '\u{2606}').count(), 2);
    }
}
