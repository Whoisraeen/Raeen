//! Home screen: status bar, pill navigation, cover-tile rail, hero
//! background, and the focused game's title + play stats (spec §3, screen 2).
//!
//! Layout mirrors the locked concept mock: avatar + profile top-left with
//! status icons center and wifi + clock top-right; a pill tab row (Store /
//! My games / Media / Library / Settings) under it; a large cover-tile rail
//! with the focused tile wearing an offset accent ring; the focused game's
//! title with Time played / Progress / Last trophy stats below the rail;
//! and a button-hint bar along the bottom edge. All art remains original —
//! gradients, monograms, hand-drawn glyphs (spec §11, zero Sony assets).
//!
//! Pure rendering — all animated values (rail slide, hero crossfade, focus
//! pop) are resolved by `shell/mod.rs` before calling [`draw`], so this
//! module stays declarative. Everything is painted at explicit rects; the
//! anchored content is deliberately not an egui layout (nested `bottom_up`
//! scopes mis-place tall rows and their overflow grows the parent `Ui`'s
//! rects, mis-anchoring whatever draws after Home).

use super::anim::lerp_color;
use super::icons::{self, Glyph};
use super::ledger;
use super::nav::{self, NavMode, NavState, RailTab};
use crate::library::{
    ArtSource, GlyphKind, Gradient, ItemKind, LibraryItem, MetaCache, TileGradient,
};
use crate::theme::Theme;
use egui::epaint::RectShape;
use egui::{Align2, Color32, FontId, Mesh, Pos2, Rect, Shape, Stroke, StrokeKind, vec2};
use std::collections::HashMap;
use xps5x_core::config::ControllerIconStyle;

/// Animated values resolved once per frame by `shell/mod.rs`.
pub struct HomeAnim {
    /// Current x-shift (px) applied to every rail tile.
    pub rail_offset: f32,
    /// Current (possibly crossfading) hero gradient.
    pub hero: Gradient,
    /// Cross-dissolve progress for the per-game key-art background: 0.0 right
    /// after a focus change (previous art shown) .. 1.0 settled (new art
    /// shown). Shares the hero gradient tween's timing.
    pub hero_fade: f32,
    /// 0.0 (just changed focus) .. 1.0 (settled) — eases the focused tile's
    /// scale in rather than snapping it.
    pub focus_pop: f32,
}

// Concept-mock vertical anchors (1080p reference; the top block is fixed
// from the top edge, the hint bar from the bottom, like the mock).
/// Top of the avatar block.
const AVATAR_TOP: f32 = 44.0;
/// Avatar circle diameter.
const AVATAR_SIZE: f32 = 52.0;
/// Top of the pill navigation row.
const PILLS_TOP: f32 = 206.0;
/// Pill height (fully rounded).
const PILL_H: f32 = 42.0;
/// Gap between pills.
const PILL_GAP: f32 = 10.0;
/// Top of the cover-tile rail.
const RAIL_TOP: f32 = 306.0;
/// Trophy-gold accent for trophy glyphs (the one non-theme color, matching
/// the mock's gold trophy marks).
const GOLD: Color32 = Color32::from_rgb(230, 190, 92);

/// What the user clicked this frame, beyond rail navigation.
#[derive(Default)]
pub struct HomeResponse {
    /// A rail tile was clicked (index into `items`).
    pub clicked_tile: Option<usize>,
    /// The top-bar settings gear was clicked.
    pub gear_clicked: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    theme: &Theme,
    items: &[LibraryItem],
    nav: &NavState,
    anim: &HomeAnim,
    meta_cache: &MetaCache,
    ledgers: &HashMap<String, ledger::TitleLedger>,
    background: Option<&egui::TextureHandle>,
    covers: &HashMap<String, egui::TextureHandle>,
    bg_from: Option<&egui::TextureHandle>,
    bg_to: Option<&egui::TextureHandle>,
    controller_icons: ControllerIconStyle,
) -> HomeResponse {
    let screen = ui.max_rect();
    let painter = ui.painter().clone();
    let focused = items.get(nav.rail_index);

    painter.rect_filled(screen, 0.0, theme.palette.ground);

    // The theme background (if any) is the base; the focused game's key art
    // cross-dissolves on top of it (`bg_from` → `bg_to`), with the mesh
    // gradient as the ultimate fallback — all resolved inside `draw_hero`.
    draw_hero(
        &painter,
        screen,
        theme,
        &anim.hero,
        background,
        bg_from,
        bg_to,
        anim.hero_fade,
    );

    let gear_rect = draw_topbar(&painter, theme, screen);
    let gear_clicked = ui
        .interact(gear_rect, ui.id().with("topbar-gear"), egui::Sense::click())
        .clicked();

    // The pills/rail/context band uses fixed 1080p-reference anchors; on a
    // taller window (e.g. a portrait monitor) that top-anchoring leaves a
    // dead band across the middle of the frame. Shift the whole band down by
    // a share of the surplus height — 0.38 parks the rail slightly above
    // true center, where the PS5 sits it. Landscape 1080p gets 0 shift, so
    // the reference layout is untouched. System chrome (topbar, hint bar)
    // stays glued to the edges.
    let band_shift = (screen.height() - 1080.0).max(0.0) * 0.38;

    draw_nav_pills(&painter, theme, screen, nav, band_shift);

    let focused_size = theme.metrics.tile_size * theme.metrics.tile_focus_scale;
    let rail_rect = Rect::from_min_size(
        Pos2::new(screen.left(), screen.top() + RAIL_TOP + band_shift),
        vec2(screen.width(), focused_size + 16.0),
    );
    let clicked_tile = draw_rail(ui, theme, rail_rect, items, nav, anim, covers);

    draw_context_block(
        &painter,
        theme,
        screen,
        rail_rect.top() + focused_size,
        focused,
        meta_cache,
        ledgers,
    );
    draw_bottom_bar(&painter, theme, screen, controller_icons);

    // The focus ring pulses and the clock ticks even when nothing else is
    // animating, so keep the screen gently alive.
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(50));
    HomeResponse {
        clicked_tile,
        gear_clicked,
    }
}

/// Paint the Home hero. A user theme's background image (spec §6), else the
/// mesh-gradient art, forms the base; the focused game's own key art
/// (`sce_sys/pic1`/`pic0`) cross-dissolves on top as focus moves between titles
/// (`bg_from` fades out, `bg_to` fades in, by `fade`). The legibility scrim on
/// top is unconditional so foreground text stays readable over a photo or a
/// gradient alike.
#[allow(clippy::too_many_arguments)]
fn draw_hero(
    painter: &egui::Painter,
    rect: Rect,
    theme: &Theme,
    g: &Gradient,
    theme_bg: Option<&egui::TextureHandle>,
    bg_from: Option<&egui::TextureHandle>,
    bg_to: Option<&egui::TextureHandle>,
    fade: f32,
) {
    // Cover-fit every background texture to the window aspect: full 0..1 UVs
    // squashed 16:9 key art 2:1 on a portrait panel.
    let aspect = rect.width() / rect.height().max(1.0);

    // Base layer: a user theme's background image if present, else the
    // (crossfading) 4-corner mesh gradient that approximates key-art glow.
    match theme_bg {
        Some(texture) => {
            painter.image(texture.id(), rect, cover_uv(texture, aspect), Color32::WHITE);
        }
        None => {
            let top_right = g.hi;
            let top_left = lerp_color(g.hi, g.mid, 0.55);
            let bottom_right = lerp_color(g.mid, g.lo, 0.4);
            let bottom_left = g.lo;
            corner_gradient(
                painter,
                rect,
                top_left,
                top_right,
                bottom_left,
                bottom_right,
            );
        }
    }

    // Per-game key art, cross-dissolved on top of the base as focus moves: the
    // outgoing art fades out while the incoming art fades in. Either side may be
    // absent (an app tile, or a game with no key art), so a game→app move
    // dissolves the photo back to the base rather than snapping.
    let fade = fade.clamp(0.0, 1.0);
    if let Some(from) = bg_from {
        painter.image(
            from.id(),
            rect,
            cover_uv(from, aspect),
            with_alpha(Color32::WHITE, 1.0 - fade),
        );
    }
    if let Some(to) = bg_to {
        painter.image(
            to.id(),
            rect,
            cover_uv(to, aspect),
            with_alpha(Color32::WHITE, fade),
        );
    }

    // Legibility scrim: darker toward the bottom-left where text sits. Kept
    // light enough that real key art stays visible (the PS5 shows the art;
    // the old 0.92 bottom-left swallowed it in near-opaque navy).
    let s = theme.palette.scrim;
    let scrim_tl = with_alpha(s, 0.30);
    let scrim_tr = with_alpha(s, 0.04);
    let scrim_bl = with_alpha(s, 0.62);
    let scrim_br = with_alpha(s, 0.35);
    corner_gradient(painter, rect, scrim_tl, scrim_tr, scrim_bl, scrim_br);
}

fn with_alpha(c: Color32, a: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (a.clamp(0.0, 1.0) * 255.0) as u8)
}

/// Paint a rect filled with a bilinear 4-corner gradient. egui has no CSS
/// gradients, so this builds a two-triangle mesh with per-vertex colors.
fn corner_gradient(
    painter: &egui::Painter,
    rect: Rect,
    top_left: Color32,
    top_right: Color32,
    bottom_left: Color32,
    bottom_right: Color32,
) {
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
fn corner_gradient_rounded(
    painter: &egui::Painter,
    rect: Rect,
    rounding: f32,
    top_left: Color32,
    top_right: Color32,
    bottom_left: Color32,
    bottom_right: Color32,
) {
    let clip = painter.clip_rect();
    let clipped = painter.with_clip_rect(rect.intersect(clip));
    // Rounded base so corners outside the rounded silhouette stay transparent-looking
    // against whatever is already painted behind (the rail background).
    clipped.rect_filled(rect, rounding, bottom_left);
    corner_gradient(
        &clipped,
        rect,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    );
    clipped.rect_stroke(rect, rounding, Stroke::NONE, StrokeKind::Inside);
}

/// The host user's display name (Windows account name). Cached; empty on
/// exotic hosts, which draw_topbar treats as "no initial".
fn host_user_name() -> &'static str {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        std::env::var("USERNAME")
            .or_else(|_| std::env::var("USER"))
            .unwrap_or_default()
    })
}

/// Status bar, PS5 proportions: nothing on the left (the tab pills carry
/// identity below), a right-aligned cluster of settings / user avatar /
/// clock. Every element is real: the gear opens Settings (clickable — the
/// returned rect is the hit target), the avatar initial is the host
/// account, the clock is local time. No fabricated profile stats, and no
/// search glyph until a search surface exists.
fn draw_topbar(painter: &egui::Painter, theme: &Theme, screen: Rect) -> Rect {
    let margin = theme.metrics.topbar_padding_x;
    let av_r = 18.0;
    let center_y = screen.top() + AVATAR_TOP + AVATAR_SIZE / 2.0 - 8.0;

    // Clock, right-most.
    let time_galley = painter.layout_no_wrap(
        current_clock_string(),
        FontId::proportional(19.0),
        theme.palette.text,
    );
    let time_size = time_galley.size();
    let time_x = screen.right() - margin - time_size.x;
    painter.galley(
        Pos2::new(time_x, center_y - time_size.y / 2.0),
        time_galley,
        theme.palette.text,
    );

    // Avatar circle with the real host user's initial.
    let av_c = Pos2::new(time_x - 30.0 - av_r, center_y);
    painter.circle_filled(av_c, av_r, with_alpha(theme.palette.accent, 0.85));
    painter.circle_stroke(
        av_c,
        av_r,
        Stroke::new(1.2f32, with_alpha(theme.palette.focus, 0.6)),
    );
    if let Some(initial) = host_user_name().chars().next() {
        painter.text(
            av_c,
            Align2::CENTER_CENTER,
            initial.to_uppercase().to_string(),
            FontId::proportional(16.0),
            theme.palette.text,
        );
    }

    // Settings gear, left of the avatar; its rect is the click target.
    let gear_c = Pos2::new(av_c.x - av_r - 30.0, center_y);
    icons::draw(painter, Glyph::Gear, gear_c, 18.0, theme.palette.text_dim);
    Rect::from_center_size(gear_c, vec2(34.0, 34.0))
}

/// Pill tab row: an icon pill, then Store / My games / Media / Library /
/// Settings / "…". The active pill tracks the rail's Games/Media tab. When
/// pill focus is live (`NavMode::Pills`, entered with Up from the rail),
/// the focused pill wears an accent ring and a brighter fill; Confirm
/// activates it (see `nav::apply_pills`).
fn draw_nav_pills(
    painter: &egui::Painter,
    theme: &Theme,
    screen: Rect,
    nav: &NavState,
    band_shift: f32,
) {
    let y = screen.top() + PILLS_TOP + band_shift;
    let inactive_bg = Color32::from_rgba_unmultiplied(255, 255, 255, 20);
    let focused_bg = Color32::from_rgba_unmultiplied(255, 255, 255, 44);
    let mut x = screen.left() + theme.metrics.topbar_padding_x;

    // Leading icon-only pill (decorative, not focusable — see `nav::PILL_*`).
    let icon_rect = Rect::from_min_size(Pos2::new(x, y), vec2(PILL_H, PILL_H));
    painter.rect_filled(icon_rect, PILL_H / 2.0, inactive_bg);
    icons::draw(
        painter,
        Glyph::Grid,
        icon_rect.center(),
        16.0,
        theme.palette.text_dim,
    );
    x += PILL_H + PILL_GAP;

    // Label order must match the `nav::PILL_*` focus indices. Only
    // functional destinations get a pill (no dead Store/Library chrome).
    let labels = [
        ("My games", nav.tab == RailTab::Games),
        ("Media", nav.tab == RailTab::Media),
        ("Settings", false),
    ];
    debug_assert_eq!(labels.len(), nav::PILL_COUNT);
    for (i, (label, active)) in labels.into_iter().enumerate() {
        let focused = nav.mode == NavMode::Pills && i == nav.pill_index;
        let text_color = if active {
            theme.palette.ground
        } else {
            theme.palette.text_dim
        };
        let galley =
            painter.layout_no_wrap(label.to_string(), FontId::proportional(15.0), text_color);
        let galley_size = galley.size();
        let w = galley_size.x + 40.0;
        let rect = Rect::from_min_size(Pos2::new(x, y), vec2(w, PILL_H));
        let bg = if active {
            theme.palette.focus
        } else if focused {
            focused_bg
        } else {
            inactive_bg
        };
        painter.rect_filled(rect, PILL_H / 2.0, bg);
        if focused {
            let ring = rect.expand(4.0);
            painter.rect_stroke(
                ring,
                ring.height() / 2.0,
                Stroke::new(2.5f32, theme.palette.accent),
                StrokeKind::Outside,
            );
        }
        painter.galley(rect.center() - galley_size / 2.0, galley, text_color);
        x += w + PILL_GAP;
    }
}

fn draw_rail(
    ui: &mut egui::Ui,
    theme: &Theme,
    rect: Rect,
    items: &[LibraryItem],
    nav: &NavState,
    anim: &HomeAnim,
    covers: &HashMap<String, egui::TextureHandle>,
) -> Option<usize> {
    // Slightly expanded clip so the focused tile's offset ring isn't cut off.
    let painter = ui.painter_at(rect.expand(18.0));
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

    // Focused tile drawn last so its ring sits above its neighbors.
    let mut order: Vec<usize> = (0..items.len()).filter(|&i| i != nav.rail_index).collect();
    if nav.rail_index < items.len() {
        order.push(nav.rail_index);
    }

    let mut clicked = None;
    for i in order {
        let item = &items[i];
        let focused = i == nav.rail_index;

        let size = if focused {
            m.tile_size + (focused_size - m.tile_size) * anim.focus_pop
        } else {
            m.tile_size
        };
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
        if ui
            .interact(
                tile_rect,
                ui.id().with(("home-tile", i)),
                egui::Sense::click(),
            )
            .clicked()
        {
            clicked = Some(i);
        }
        // Optically constant corners: scaling the radius with the focused
        // tile's growth reads squircle-ish; the PS5 keeps rounding subtle
        // and stable across the size jump.
        let radius = m.corner_radius;

        // Soft drop shadow so tiles read as raised cards, especially over a
        // busy key-art background. Four low-alpha, downward-offset rounded
        // rects stack into a cheap penumbra (egui has no blur); the tile itself
        // is painted on top and hides the core, leaving only the falloff.
        for k in 0..4 {
            let spread = k as f32 * (size * 0.02);
            painter.rect_filled(
                tile_rect.translate(vec2(0.0, size * 0.03)).expand(spread),
                radius + spread,
                with_alpha(Color32::BLACK, 0.06 * alpha),
            );
        }

        let g = item.art.tile();
        let faded = TileGradient {
            from: with_alpha(g.from, alpha),
            to: with_alpha(g.to, alpha),
        };
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
                icons::draw(
                    &painter,
                    g,
                    tile_rect.center(),
                    size * 0.3,
                    with_alpha(Color32::WHITE, 0.92 * alpha),
                );
            }
            ArtSource::Game { .. } => {
                if let Some(texture) = covers.get(item.id.as_str()) {
                    // The user's own cover image (spec §11: user-supplied,
                    // like theme backgrounds), center-cropped to the square
                    // tile and tinted for the passed-tile fade.
                    let shape =
                        RectShape::filled(tile_rect, radius, with_alpha(Color32::WHITE, alpha))
                            .with_texture(texture.id(), cover_uv(texture, 1.0));
                    painter.add(Shape::Rect(shape));
                } else if item.kind == ItemKind::Game {
                    // Original stand-in key art (spec §11 — never real box
                    // art): a bold two-initial monogram with a ghosted echo
                    // for depth, plus the title small along the bottom so
                    // the tile self-identifies like a real cover instead of
                    // reading as a failed image load.
                    let monogram: String = item
                        .title
                        .split_whitespace()
                        .filter(|w| {
                            !matches!(
                                w.to_ascii_lowercase().as_str(),
                                "a" | "an" | "the" | "of" | "to"
                            )
                        })
                        .take(2)
                        .filter_map(|w| w.chars().next())
                        .flat_map(|c| c.to_uppercase())
                        .collect();
                    let clipped = painter.with_clip_rect(tile_rect);
                    clipped.text(
                        tile_rect.center() + vec2(size * 0.06, size * 0.08),
                        Align2::CENTER_CENTER,
                        &monogram,
                        FontId::proportional(size * 0.52),
                        with_alpha(Color32::WHITE, 0.10 * alpha),
                    );
                    clipped.text(
                        tile_rect.center() - vec2(0.0, size * 0.04),
                        Align2::CENTER_CENTER,
                        &monogram,
                        FontId::proportional(size * 0.34),
                        with_alpha(Color32::WHITE, 0.85 * alpha),
                    );
                    clipped.text(
                        Pos2::new(
                            tile_rect.left() + size * 0.09,
                            tile_rect.bottom() - size * 0.09,
                        ),
                        Align2::LEFT_BOTTOM,
                        &item.title,
                        FontId::proportional((size * 0.075).max(11.0)),
                        with_alpha(Color32::WHITE, 0.75 * alpha),
                    );
                }
            }
        }

        if focused {
            // PS5 signature focus: a thin white line hugging the tile, with
            // only a whisper of accent halo behind it (the old thick blue
            // ring + strong glow vanished into blue art). The white shimmer
            // pulse is what reads as "alive".
            let ring_rect = tile_rect.expand(4.0);
            let ring_radius = radius + 4.0;
            for k in 1..=2 {
                let spread = k as f32 * 3.0;
                let a = 0.06 * (1.0 - k as f32 / 3.0) * (0.6 + 0.4 * pulse);
                painter.rect_stroke(
                    ring_rect.expand(spread),
                    ring_radius + spread,
                    Stroke::new(4.0f32, with_alpha(theme.palette.accent, a)),
                    StrokeKind::Outside,
                );
            }
            let ring_a = (0.75 + 0.25 * pulse) * alpha;
            painter.rect_stroke(
                ring_rect,
                ring_radius,
                Stroke::new(2.5f32, with_alpha(theme.palette.focus, ring_a)),
                StrokeKind::Outside,
            );
        }
    }
    clicked
}

/// UV rect that center-crops `texture` to `target_aspect` (width/height):
/// whichever texture axis overshoots the target is trimmed equally on both
/// sides ("cover" fit — never stretch). `1.0` = the square rail tiles; the
/// hero passes the window aspect so 16:9 key art isn't squashed onto a
/// portrait screen.
fn cover_uv(texture: &egui::TextureHandle, target_aspect: f32) -> Rect {
    let size = texture.size_vec2();
    if size.x <= 0.0 || size.y <= 0.0 || target_aspect <= 0.0 {
        return Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
    }
    let tex_aspect = size.x / size.y;
    if tex_aspect > target_aspect {
        let keep = target_aspect / tex_aspect;
        let dx = (1.0 - keep) / 2.0;
        Rect::from_min_max(Pos2::new(dx, 0.0), Pos2::new(1.0 - dx, 1.0))
    } else {
        let keep = tex_aspect / target_aspect;
        let dy = (1.0 - keep) / 2.0;
        Rect::from_min_max(Pos2::new(0.0, dy), Pos2::new(1.0, 1.0 - dy))
    }
}

/// The focused item's block under the rail: big title, then the
/// Time played / Progress / Last trophy stat columns (or a plain "Open …"
/// line for built-in apps).
#[allow(clippy::too_many_arguments)]
fn draw_context_block(
    painter: &egui::Painter,
    theme: &Theme,
    screen: Rect,
    rail_bottom: f32,
    focused: Option<&LibraryItem>,
    meta_cache: &MetaCache,
    ledgers: &HashMap<String, ledger::TitleLedger>,
) {
    let Some(item) = focused else { return };
    // Home reads a focused item's metadata from the cache rather than the
    // item directly, so rendering stays decoupled from how (or whether)
    // metadata was sourced (parsed `xps5x-title.toml`, or none) — spec §4.
    let meta = meta_cache.get(item.id.as_str());
    let left = screen.left() + theme.metrics.content_padding_x;

    let title_galley = painter.layout_no_wrap(
        item.title.clone(),
        FontId::proportional(46.0),
        theme.palette.text,
    );
    let title_h = title_galley.size().y;
    painter.galley(
        Pos2::new(left, rail_bottom + 52.0),
        title_galley,
        theme.palette.text,
    );

    let stats_top = rail_bottom + 52.0 + title_h + 28.0;
    let Some(meta) = meta else {
        // Real package identity (sce_sys/param.json) + real session history
        // (the ledger) for games; plain "Open …" for built-in apps. The
        // status is honest: a title whose last session faulted says so
        // instead of claiming "Ready to play".
        if item.kind != ItemKind::Game {
            painter.text(
                Pos2::new(left, stats_top),
                Align2::LEFT_TOP,
                format!("Open {}", item.title),
                FontId::proportional(18.0),
                theme.palette.text_dim,
            );
            return;
        }
        let title_ledger = ledgers.get(item.id.as_str());
        let mut identity = Vec::new();
        if let Some(id) = &item.title_id {
            identity.push(id.clone());
        }
        if let Some(version) = &item.version {
            identity.push(format!("v{version}"));
        }
        if let Some(played) = title_ledger.and_then(|l| l.play_time_text()) {
            identity.push(played);
        }
        let faulted = title_ledger.is_some_and(|l| l.last_faulted);
        let (status, status_color) = if faulted {
            (
                "Crashed last session — check Console (F10)",
                Color32::from_rgb(255, 168, 100),
            )
        } else {
            ("Ready to play", theme.palette.text_dim)
        };
        let mut x = left;
        if !identity.is_empty() {
            let identity_galley = painter.layout_no_wrap(
                format!("{}  ·  ", identity.join("  ·  ")),
                FontId::proportional(18.0),
                theme.palette.text_dim,
            );
            let w = identity_galley.size().x;
            painter.galley(Pos2::new(x, stats_top), identity_galley, theme.palette.text_dim);
            x += w;
        }
        painter.text(
            Pos2::new(x, stats_top),
            Align2::LEFT_TOP,
            status,
            FontId::proportional(18.0),
            status_color,
        );
        return;
    };

    let dash = "\u{2014}".to_string();
    let time_played = if meta.time_played.is_empty() {
        dash.clone()
    } else {
        meta.time_played.clone()
    };
    let progress = meta
        .progress_percent()
        .map(|p| format!("{p}%"))
        .unwrap_or_else(|| dash.clone());
    let last_trophy = if meta.last_trophy.is_empty() {
        dash
    } else {
        meta.last_trophy.clone()
    };

    let columns: [(&str, String, bool); 3] = [
        ("Time played", time_played, false),
        ("Progress", progress, false),
        ("Last trophy", last_trophy, true),
    ];
    let mut x = left;
    for (label, value, with_trophy) in columns {
        let label_galley = painter.layout_no_wrap(
            label.to_string(),
            FontId::proportional(18.0),
            theme.palette.text_faint,
        );
        let value_galley =
            painter.layout_no_wrap(value, FontId::proportional(32.0), theme.palette.text);
        let label_w = label_galley.size().x;
        let value_w = value_galley.size().x;
        let value_h = value_galley.size().y;
        painter.galley(
            Pos2::new(x, stats_top),
            label_galley,
            theme.palette.text_faint,
        );

        let value_y = stats_top + 32.0;
        let mut value_x = x;
        if with_trophy {
            icons::draw(
                painter,
                Glyph::Trophy,
                Pos2::new(x + 13.0, value_y + value_h / 2.0),
                26.0,
                GOLD,
            );
            value_x += 38.0;
        }
        painter.galley(
            Pos2::new(value_x, value_y),
            value_galley,
            theme.palette.text,
        );

        let col_w = label_w.max(value_w + if with_trophy { 38.0 } else { 0.0 });
        x += col_w + 72.0;
    }
}

/// Bottom hint bar: circled button glyphs with labels on the left (Play /
/// Search / Options, the mock's row), chat + capture status glyphs on the
/// right.
fn draw_bottom_bar(
    painter: &egui::Painter,
    theme: &Theme,
    screen: Rect,
    controller_icons: ControllerIconStyle,
) {
    let margin = theme.metrics.content_padding_x;
    let y = screen.bottom() - theme.metrics.content_padding_bottom;
    let circle_r = 11.0;
    let fill = Color32::from_rgba_unmultiplied(255, 255, 255, 14);

    // Only hints for actions that exist: Play (Confirm) and Options. The
    // old Search hint had no search overlay behind it, and the decorative
    // chat/capture glyphs reported nothing.
    let entries = [
        (Some(Glyph::Cross), controller_icons.confirm(), "Play"),
        (Some(Glyph::Menu), "", "Options"),
    ];
    let mut x = screen.left() + margin + circle_r;
    for (glyph, button_label, label) in entries {
        let c = Pos2::new(x, y);
        painter.circle_filled(c, circle_r, fill);
        painter.circle_stroke(c, circle_r, Stroke::new(1.4f32, theme.palette.line));
        if matches!(controller_icons, ControllerIconStyle::PlayStation) || button_label.is_empty() {
            icons::draw(
                painter,
                glyph.expect("entries have glyphs"),
                c,
                10.0,
                theme.palette.text_dim,
            );
        } else {
            painter.text(
                c,
                Align2::CENTER_CENTER,
                button_label,
                FontId::proportional(11.0),
                theme.palette.text_dim,
            );
        }
        let galley = painter.layout_no_wrap(
            label.to_string(),
            FontId::proportional(13.0),
            theme.palette.text_faint,
        );
        let galley_size = galley.size();
        painter.galley(
            Pos2::new(x + circle_r + 10.0, y - galley_size.y / 2.0),
            galley,
            theme.palette.text_faint,
        );
        x += circle_r + 10.0 + galley_size.x + 44.0;
    }
}

/// Local wall-clock, formatted 12-hour ("4:44 PM"). On Windows this is
/// Win32 `GetLocalTime` via the existing workspace `windows-sys` dep — no
/// new crates (spec §11); other platforms fall back to a raw epoch
/// breakdown, i.e. UTC.
fn current_clock_string() -> String {
    let (hour24, minute) = local_hour_minute();
    let (h12, ampm) = match hour24 {
        0 => (12, "AM"),
        1..=11 => (hour24, "AM"),
        12 => (12, "PM"),
        _ => (hour24 - 12, "PM"),
    };
    format!("{h12}:{minute:02} {ampm}")
}

#[cfg(windows)]
fn local_hour_minute() -> (u32, u32) {
    use windows_sys::Win32::Foundation::SYSTEMTIME;
    use windows_sys::Win32::System::SystemInformation::GetLocalTime;
    let mut st: SYSTEMTIME = unsafe { std::mem::zeroed() };
    unsafe { GetLocalTime(&mut st) };
    (st.wHour as u32, st.wMinute as u32)
}

#[cfg(not(windows))]
fn local_hour_minute() -> (u32, u32) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    (((secs % 86400) / 3600) as u32, ((secs % 3600) / 60) as u32)
}
