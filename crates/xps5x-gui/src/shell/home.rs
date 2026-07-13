//! Home screen: function bar, top tile rail, hero background, context block
//! (spec §3, screen 2).
//!
//! Layout mirrors the PS5 home screen: Games/Media tabs top-left with
//! search/settings/avatar/clock top-right, the tile rail directly under
//! them with the focused tile enlarged in place, and the focused game's
//! wordmark + activity cards over the hero art in the lower-left. All art
//! remains original (gradients + monograms — spec §11, zero Sony assets).
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
    /// scale in rather than snapping it.
    pub focus_pop: f32,
}

/// Height of the top function bar (tabs + status icons).
const TOPBAR_H: f32 = 72.0;
/// Breathing room between the function bar and the tile rail.
const RAIL_TOP_GAP: f32 = 10.0;

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

    let topbar_rect = Rect::from_min_size(screen.min, vec2(screen.width(), TOPBAR_H));
    draw_topbar(ui, theme, topbar_rect, nav.tab);

    // Tile rail: top of the screen, directly under the function bar, with
    // the focused tile enlarged in place — the PS5 home arrangement.
    let focused_size = theme.metrics.tile_size * theme.metrics.tile_focus_scale;
    let rail_rect = Rect::from_min_size(
        Pos2::new(screen.left(), topbar_rect.bottom() + RAIL_TOP_GAP),
        vec2(screen.width(), focused_size + 12.0),
    );
    draw_rail(ui, theme, rail_rect, items, nav, anim);

    let content_rect = Rect::from_min_max(Pos2::new(screen.left(), rail_rect.bottom()), screen.max);
    let focused = items.get(nav.rail_index);
    draw_context_block(ui, theme, content_rect, focused, meta_cache);
    draw_hints(&painter, theme, screen);

    // The focus ring pulses and the clock ticks even when nothing else is
    // animating, so keep the screen gently alive.
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));
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
            // Approximate full-bleed key art's upper-right glow with a
            // 4-corner mesh.
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
            ui.label(RichText::new("Games").color(games_color).size(22.0).strong());
            ui.add_space(26.0);
            ui.label(RichText::new("Media").color(media_color).size(22.0).strong());

            // Right cluster, right-most first: clock, avatar, settings,
            // search — reading left-to-right as search · gear · avatar ·
            // time, the console's top-bar order. Time only, no date.
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_space(theme.metrics.topbar_padding_x);

                ui.label(RichText::new(current_clock_string()).color(theme.palette.text).size(19.0).strong());
                ui.add_space(20.0);

                // Avatar.
                let (rect, _resp) = ui.allocate_exact_size(vec2(34.0, 34.0), Sense::hover());
                ui.painter().circle_filled(rect.center(), 17.0, theme.palette.accent);
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "X5", egui::FontId::proportional(13.0), theme.palette.text);
                ui.add_space(16.0);

                for glyph in [Glyph::Gear, Glyph::Search] {
                    icon_button(ui, theme, glyph);
                    ui.add_space(10.0);
                }
            });
        });
    });
}

fn icon_button(ui: &mut egui::Ui, theme: &Theme, glyph: Glyph) {
    let (rect, resp) = ui.allocate_exact_size(vec2(36.0, 36.0), Sense::hover());
    if resp.hovered() {
        ui.painter().circle_filled(rect.center(), 18.0, Color32::from_rgba_unmultiplied(255, 255, 255, 30));
    }
    icons::draw(ui.painter(), glyph, rect.center(), 20.0, theme.palette.text);
}

/// The focused item's hero block: wordmark, meta line, and activity cards,
/// anchored to the bottom-left with explicit painter rects. Deliberately
/// not an egui layout — nested `bottom_up` scopes mis-place tall rows and
/// their overflow grows the parent `Ui`'s rects, which then mis-anchors
/// everything drawn after Home (the Control Center).
fn draw_context_block(ui: &mut egui::Ui, theme: &Theme, rect: Rect, focused: Option<&LibraryItem>, meta_cache: &MetaCache) {
    let Some(item) = focused else { return };
    // Home reads a focused item's metadata from the cache rather than the
    // item directly, so rendering stays decoupled from how (or whether)
    // metadata was sourced (parsed `xps5x-title.toml`, or none) — spec §4.
    let meta = meta_cache.get(item.id.as_str());
    let painter = ui.painter();
    let m = &theme.metrics;
    let left = rect.left() + m.content_padding_x;
    let mut y = rect.bottom() - m.content_padding_bottom;

    // Activity cards row along the bottom edge. A game with no metadata
    // renders no cards — just the wordmark block.
    if let Some(meta) = meta {
        let card_top = y - m.card_size.y;
        let mut x = left;
        for card in &meta.activity {
            let card_rect = Rect::from_min_size(Pos2::new(x, card_top), m.card_size);
            draw_activity_card(painter, theme, card_rect, card);
            x += m.card_size.x + m.card_gap;
        }
        y = card_top - 30.0;
    }

    // Meta line (genre/players) or system subtitle — the PS5 home shows no
    // persistent Play button; Confirm on the tile launches.
    let meta_text = match meta {
        Some(meta) => format!("{}   \u{00B7}   {}", meta.genre, meta.players),
        None => format!("Open {}", item.title),
    };
    painter.text(
        Pos2::new(left, y),
        egui::Align2::LEFT_BOTTOM,
        meta_text,
        egui::FontId::proportional(15.0),
        theme.palette.text_dim,
    );
    y -= 15.0 + 14.0;

    // Wordmark.
    painter.text(
        Pos2::new(left, y),
        egui::Align2::LEFT_BOTTOM,
        &item.title,
        egui::FontId::proportional(64.0),
        theme.palette.text,
    );
}

fn draw_activity_card(painter: &egui::Painter, theme: &Theme, rect: Rect, card: &crate::library::ActivityCard) {
    let size = rect.size();
    painter.rect_filled(rect, theme.metrics.corner_radius, Color32::from_rgba_unmultiplied(14, 20, 28, 200));
    painter.rect_stroke(rect, theme.metrics.corner_radius, Stroke::new(1.0, theme.palette.line), StrokeKind::Inside);

    let pad = 16.0;
    painter.text(
        rect.min + vec2(pad, pad),
        egui::Align2::LEFT_TOP,
        card.top.to_uppercase(),
        egui::FontId::proportional(11.0),
        theme.palette.text_faint,
    );
    painter.text(
        rect.min + vec2(pad, pad + 24.0),
        egui::Align2::LEFT_TOP,
        &card.main,
        egui::FontId::proportional(17.0),
        theme.palette.text,
    );
    if !card.sub.is_empty() {
        painter.text(
            rect.min + vec2(pad, pad + 48.0),
            egui::Align2::LEFT_TOP,
            &card.sub,
            egui::FontId::proportional(13.0),
            theme.palette.text_dim,
        );
    }
    if let Some(progress) = card.progress {
        let bar_rect = Rect::from_min_size(rect.min + vec2(pad, size.y - pad - 5.0), vec2(size.x - pad * 2.0, 5.0));
        painter.rect_filled(bar_rect, 2.5, Color32::from_rgba_unmultiplied(255, 255, 255, 30));
        let filled_w = bar_rect.width() * (progress as f32 / 100.0).clamp(0.0, 1.0);
        let filled_rect = Rect::from_min_size(bar_rect.min, vec2(filled_w, bar_rect.height()));
        painter.rect_filled(filled_rect, 2.5, theme.palette.accent_hi);
    }
}

fn draw_rail(ui: &mut egui::Ui, theme: &Theme, rect: Rect, items: &[LibraryItem], nav: &NavState, anim: &HomeAnim) {
    // Slightly expanded clip so the focused tile's glow isn't cut off.
    let painter = ui.painter_at(rect.expand(10.0));
    let m = &theme.metrics;
    let step = m.tile_size + m.tile_gap;
    let anchor_x = rect.left() + m.rail_padding_left;
    let focused_size = m.tile_size * m.tile_focus_scale;
    // All tiles share the focused tile's vertical center.
    let center_y = rect.top() + focused_size / 2.0;
    // Extra room the growing focused tile opens up in front of its
    // followers, so they never sit underneath it.
    let extra = (focused_size - m.tile_size) * anim.focus_pop;
    let pulse = 0.5 + 0.5 * (ui.input(|i| i.time) as f32 * 2.4).sin();

    // Focused tile drawn last so its ring/glow sits above its neighbors.
    let mut order: Vec<usize> = (0..items.len()).filter(|&i| i != nav.rail_index).collect();
    if nav.rail_index < items.len() {
        order.push(nav.rail_index);
    }

    for i in order {
        let item = &items[i];
        let focused = i == nav.rail_index;

        let size = if focused { m.tile_size + (focused_size - m.tile_size) * anim.focus_pop } else { m.tile_size };
        let mut x = anchor_x + i as f32 * step + anim.rail_offset;
        if i > nav.rail_index {
            x += extra;
        }

        if x + size < rect.left() - 60.0 || x > rect.right() + 60.0 {
            continue; // off-screen, skip
        }

        // Passed tiles fade out as they slide behind the left anchor
        // instead of poking out of the margin.
        let alpha = ((x - (anchor_x - step)) / step).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            continue;
        }

        let tile_rect = Rect::from_min_size(Pos2::new(x, center_y - size / 2.0), vec2(size, size));
        let radius = m.corner_radius * size / m.tile_size;

        let g = item.art.tile();
        let faded = TileGradient { from: with_alpha(g.from, alpha), to: with_alpha(g.to, alpha) };
        tile_gradient_rect(&painter, tile_rect, radius, &faded);

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
                icons::draw(&painter, g, tile_rect.center(), size * 0.34, with_alpha(Color32::WHITE, 0.92 * alpha));
            }
            ArtSource::Game { .. } => {
                if item.kind == ItemKind::Game {
                    // Original stand-in key art: a large centered monogram
                    // over the gradient (spec §11 — never real box art).
                    let monogram = item.title.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_default();
                    painter.text(
                        tile_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        monogram,
                        egui::FontId::proportional(size * 0.42),
                        with_alpha(Color32::WHITE, 0.45 * alpha),
                    );
                }
            }
        }

        if focused {
            // Soft outer glow, then the crisp pulsing ring on the tile edge.
            for k in 1..=3 {
                let spread = k as f32 * 2.0;
                let a = 26.0 * (1.0 - k as f32 / 4.0) * (0.6 + 0.4 * pulse) / 255.0;
                painter.rect_stroke(
                    tile_rect.expand(spread),
                    radius + spread,
                    Stroke::new(3.0, with_alpha(theme.palette.focus, a)),
                    StrokeKind::Outside,
                );
            }
            let ring_a = (0.84 + 0.16 * pulse) * alpha;
            painter.rect_stroke(tile_rect, radius, Stroke::new(2.5, with_alpha(theme.palette.focus, ring_a)), StrokeKind::Outside);
        }
    }
}

fn draw_hints(painter: &egui::Painter, theme: &Theme, screen: Rect) {
    let hints = "\u{25C0} \u{25B6} Navigate    Enter Play    Tab Media/Games    C Control Center";
    painter.text(
        Pos2::new(screen.right() - theme.metrics.content_padding_x, screen.bottom() - 14.0),
        egui::Align2::RIGHT_BOTTOM,
        hints,
        egui::FontId::proportional(12.0),
        theme.palette.text_faint,
    );
}

/// UTC-based clock string (time only, like the console's top bar). No
/// timezone crate is in the workspace, so this stays a plain epoch
/// breakdown rather than pulling in a new dependency for a cosmetic clock
/// (spec §11 keeps deps to the existing workspace set).
fn current_clock_string() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let sec_of_day = secs % 86400;
    let hour24 = sec_of_day / 3600;
    let minute = (sec_of_day % 3600) / 60;
    let (h12, ampm) = match hour24 {
        0 => (12, "AM"),
        1..=11 => (hour24, "AM"),
        12 => (12, "PM"),
        _ => (hour24 - 12, "PM"),
    };
    format!("{h12}:{minute:02} {ampm}")
}
