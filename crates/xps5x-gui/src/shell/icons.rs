//! Original, hand-drawn vector glyphs for the Shell's system icons.
//!
//! Deliberately simple primitive shapes (circles, lines, rounded rects) —
//! not recreations of Sony's iconography (spec §11 — zero Sony assets).
//! Every glyph is painted directly with `egui::Painter`, sized to fit
//! within a `size`×`size` box centered at `center`.

use egui::{Color32, Painter, Pos2, Stroke, StrokeKind, vec2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    Search,
    Friends,
    Bell,
    Gear,
    Home,
    Switcher,
    Music,
    Sound,
    Mic,
    Pad,
    Profile,
    Network,
    Power,
    Bag,
    Grid,
}

pub fn draw(painter: &Painter, glyph: Glyph, center: Pos2, size: f32, color: Color32) {
    let stroke = Stroke::new((size * 0.09).max(1.3), color);
    let r = size * 0.5;

    match glyph {
        Glyph::Search => {
            let c = center + vec2(-r * 0.18, -r * 0.18);
            painter.circle_stroke(c, r * 0.55, stroke);
            let dir = vec2(0.72, 0.72) * (r * 0.55);
            painter.line_segment([c + dir, c + dir * 1.7], stroke);
        }
        Glyph::Friends => {
            painter.circle_stroke(center + vec2(-r * 0.32, 0.0), r * 0.36, stroke);
            painter.circle_stroke(center + vec2(r * 0.32, -r * 0.05), r * 0.30, stroke);
        }
        Glyph::Bell => {
            painter.circle_stroke(center + vec2(0.0, -r * 0.1), r * 0.5, stroke);
            painter.line_segment(
                [center + vec2(-r * 0.25, r * 0.45), center + vec2(r * 0.25, r * 0.45)],
                stroke,
            );
        }
        Glyph::Gear => {
            painter.circle_stroke(center, r * 0.4, stroke);
            for i in 0..8 {
                let angle = (i as f32) * std::f32::consts::TAU / 8.0;
                let dir = vec2(angle.cos(), angle.sin());
                painter.line_segment([center + dir * r * 0.55, center + dir * r * 0.9], stroke);
            }
        }
        Glyph::Home => {
            painter.line_segment([center + vec2(-r * 0.85, 0.0), center + vec2(0.0, -r * 0.75)], stroke);
            painter.line_segment([center + vec2(0.0, -r * 0.75), center + vec2(r * 0.85, 0.0)], stroke);
            let base = egui::Rect::from_min_max(
                center + vec2(-r * 0.5, -r * 0.05),
                center + vec2(r * 0.5, r * 0.8),
            );
            painter.rect_stroke(base, 1.0, stroke, StrokeKind::Outside);
        }
        Glyph::Switcher => {
            let a = egui::Rect::from_min_max(center + vec2(-r * 0.85, -r * 0.55), center + vec2(-r * 0.05, r * 0.55));
            let b = egui::Rect::from_min_max(center + vec2(r * 0.05, -r * 0.55), center + vec2(r * 0.85, r * 0.55));
            painter.rect_stroke(a, 2.0, stroke, StrokeKind::Outside);
            painter.rect_stroke(b, 2.0, stroke, StrokeKind::Outside);
        }
        Glyph::Music => {
            painter.circle_filled(center + vec2(-r * 0.35, r * 0.45), r * 0.22, color);
            painter.circle_filled(center + vec2(r * 0.4, r * 0.25), r * 0.22, color);
            painter.line_segment([center + vec2(-r * 0.15, r * 0.45), center + vec2(-r * 0.15, -r * 0.7)], stroke);
            painter.line_segment([center + vec2(r * 0.6, r * 0.25), center + vec2(r * 0.6, -r * 0.55)], stroke);
            painter.line_segment([center + vec2(-r * 0.15, -r * 0.7), center + vec2(r * 0.6, -r * 0.55)], stroke);
        }
        Glyph::Sound => {
            let body = egui::Rect::from_min_max(center + vec2(-r * 0.75, -r * 0.3), center + vec2(-r * 0.25, r * 0.3));
            painter.rect_stroke(body, 1.0, stroke, StrokeKind::Outside);
            painter.line_segment([center + vec2(-r * 0.25, -r * 0.3), center + vec2(r * 0.1, -r * 0.65)], stroke);
            painter.line_segment([center + vec2(-r * 0.25, r * 0.3), center + vec2(r * 0.1, r * 0.65)], stroke);
            painter.line_segment([center + vec2(r * 0.1, -r * 0.65), center + vec2(r * 0.1, r * 0.65)], stroke);
            painter.arc(center + vec2(r * 0.1, 0.0), r * 0.55, -0.5..=0.5, stroke);
        }
        Glyph::Mic => {
            let body = egui::Rect::from_center_size(center + vec2(0.0, -r * 0.15), vec2(r * 0.5, r * 0.9));
            painter.rect_stroke(body, r * 0.25, stroke, StrokeKind::Outside);
            painter.arc(center + vec2(0.0, 0.1), r * 0.65, 0.35..=(std::f32::consts::PI - 0.35), stroke);
            painter.line_segment([center + vec2(0.0, r * 0.55), center + vec2(0.0, r * 0.85)], stroke);
        }
        Glyph::Pad => {
            let body = egui::Rect::from_center_size(center, vec2(r * 1.7, r * 1.0));
            painter.rect_stroke(body, r * 0.4, stroke, StrokeKind::Outside);
            painter.circle_filled(center + vec2(r * 0.45, -r * 0.05), r * 0.1, color);
            painter.circle_filled(center + vec2(r * 0.65, r * 0.1), r * 0.1, color);
            painter.line_segment([center + vec2(-r * 0.65, 0.0), center + vec2(-r * 0.35, 0.0)], stroke);
            painter.line_segment([center + vec2(-r * 0.5, -r * 0.15), center + vec2(-r * 0.5, r * 0.15)], stroke);
        }
        Glyph::Profile => {
            painter.circle_stroke(center + vec2(0.0, -r * 0.35), r * 0.32, stroke);
            painter.arc(center + vec2(0.0, r * 0.85), r * 0.65, (std::f32::consts::PI + 0.3)..=(-0.3), stroke);
        }
        Glyph::Network => {
            painter.arc(center + vec2(0.0, r * 0.2), r * 0.85, (-2.6)..=(-0.55), stroke);
            painter.arc(center + vec2(0.0, r * 0.2), r * 0.5, (-2.5)..=(-0.65), stroke);
            painter.circle_filled(center + vec2(0.0, r * 0.75), r * 0.1, color);
        }
        Glyph::Power => {
            painter.arc(center + vec2(0.0, r * 0.1), r * 0.55, 0.7..=(std::f32::consts::TAU - 0.7), stroke);
            painter.line_segment([center + vec2(0.0, -r * 0.7), center + vec2(0.0, r * 0.05)], stroke);
        }
        Glyph::Bag => {
            let body = egui::Rect::from_min_max(center + vec2(-r * 0.6, -r * 0.15), center + vec2(r * 0.6, r * 0.75));
            painter.rect_stroke(body, 2.0, stroke, StrokeKind::Outside);
            painter.arc(center + vec2(0.0, -r * 0.15), r * 0.35, (std::f32::consts::PI + 0.2)..=(-0.2), stroke);
        }
        Glyph::Grid => {
            let s = r * 0.55;
            for dx in [-1.0_f32, 1.0] {
                for dy in [-1.0_f32, 1.0] {
                    let c = center + vec2(dx * r * 0.42, dy * r * 0.42);
                    let rect = egui::Rect::from_center_size(c, vec2(s, s));
                    painter.rect_stroke(rect, 1.0, stroke, StrokeKind::Outside);
                }
            }
        }
    }
}

/// Small helper: draw an arc using line segments (egui's `Painter` has no
/// built-in arc primitive, so this approximates one).
trait ArcExt {
    fn arc(&self, center: Pos2, radius: f32, angle_range: std::ops::RangeInclusive<f32>, stroke: Stroke);
}

impl ArcExt for Painter {
    fn arc(&self, center: Pos2, radius: f32, angle_range: std::ops::RangeInclusive<f32>, stroke: Stroke) {
        const SEGMENTS: usize = 16;
        let start = *angle_range.start();
        let end = *angle_range.end();
        let mut prev = center + vec2(start.cos(), start.sin()) * radius;
        for i in 1..=SEGMENTS {
            let t = i as f32 / SEGMENTS as f32;
            let angle = start + (end - start) * t;
            let p = center + vec2(angle.cos(), angle.sin()) * radius;
            self.line_segment([prev, p], stroke);
            prev = p;
        }
    }
}
