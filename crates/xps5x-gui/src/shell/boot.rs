//! Boot sequence — brief power-on animation before Home (spec §3, screen 1).
//!
//! Pure timing logic (testable) plus a small `draw` helper. The mockup's
//! boot screen is a centered wordmark plus a shimmering progress bar that
//! fades to the Home screen after a fixed delay.

use crate::theme::Theme;
use std::time::{Duration, Instant};

/// Drives the boot screen's timing: logo hold, then a crossfade into Home.
#[derive(Debug)]
pub struct BootSequence {
    started: Instant,
    hold_duration: Duration,
    fade_duration: Duration,
}

impl BootSequence {
    pub fn new() -> Self {
        Self::with_durations(Duration::from_millis(1400), Duration::from_millis(700))
    }

    pub fn with_durations(hold_duration: Duration, fade_duration: Duration) -> Self {
        Self {
            started: Instant::now(),
            hold_duration,
            fade_duration,
        }
    }

    /// 0.0 at boot start, 1.0 once the crossfade into Home has finished.
    pub fn fade_alpha(&self) -> f32 {
        let elapsed = self.started.elapsed().as_secs_f32();
        let hold = self.hold_duration.as_secs_f32();
        let fade = self.fade_duration.as_secs_f32();
        if fade <= 0.0 {
            return if elapsed >= hold { 1.0 } else { 0.0 };
        }
        ((elapsed - hold) / fade).clamp(0.0, 1.0)
    }

    /// A looping 0..1 shimmer position for the progress bar.
    pub fn shimmer(&self) -> f32 {
        let period = 1.4_f32;
        let elapsed = self.started.elapsed().as_secs_f32();
        (elapsed % period) / period
    }

    pub fn is_done(&self) -> bool {
        self.fade_alpha() >= 1.0
    }
}

impl Default for BootSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// Draw the boot overlay (wordmark + shimmering bar), fading it out as
/// `boot.fade_alpha()` progresses. `alpha` is `1.0 - fade_alpha()` (opacity
/// of the boot layer itself).
pub fn draw(ctx: &egui::Context, theme: &Theme, boot: &BootSequence) {
    let alpha = 1.0 - boot.fade_alpha();
    if alpha <= 0.0 {
        return;
    }

    egui::Area::new(egui::Id::new("xps5x_boot"))
        .fixed_pos(egui::pos2(0.0, 0.0))
        .order(egui::Order::Foreground)
        .interactable(false)
        .show(ctx, |ui| {
            let screen = ctx.screen_rect();
            let painter = ui.painter();

            let bg = egui::Color32::from_rgba_unmultiplied(5, 8, 13, (alpha * 255.0) as u8);
            painter.rect_filled(screen, 0.0, bg);

            let center = screen.center();
            let logo_color = theme.palette.text.gamma_multiply(alpha);
            painter.text(
                egui::pos2(center.x, center.y - 20.0),
                egui::Align2::CENTER_CENTER,
                "Raeen",
                egui::FontId::proportional(34.0),
                logo_color,
            );

            let bar_w = 180.0;
            let bar_h = 3.0;
            let bar_rect = egui::Rect::from_center_size(
                egui::pos2(center.x, center.y + 26.0),
                egui::vec2(bar_w, bar_h),
            );
            painter.rect_filled(
                bar_rect,
                bar_h / 2.0,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, (31.0 * alpha) as u8),
            );

            let shimmer_w = bar_w * 0.4;
            let shimmer_x =
                bar_rect.left() - shimmer_w + boot.shimmer() * (bar_w + shimmer_w * 2.0);
            let shimmer_rect = egui::Rect::from_min_size(
                egui::pos2(shimmer_x, bar_rect.top()),
                egui::vec2(shimmer_w, bar_h),
            )
            .intersect(bar_rect);
            if shimmer_rect.width() > 0.0 {
                painter.rect_filled(
                    shimmer_rect,
                    bar_h / 2.0,
                    theme.palette.accent_hi.gamma_multiply(alpha),
                );
            }
        });

    ctx.request_repaint();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fade_alpha_is_zero_during_hold() {
        let boot =
            BootSequence::with_durations(Duration::from_secs(60), Duration::from_millis(200));
        assert_eq!(boot.fade_alpha(), 0.0);
        assert!(!boot.is_done());
    }

    #[test]
    fn fade_alpha_reaches_one_immediately_with_zero_durations() {
        let boot = BootSequence::with_durations(Duration::ZERO, Duration::ZERO);
        assert_eq!(boot.fade_alpha(), 1.0);
        assert!(boot.is_done());
    }

    #[test]
    fn shimmer_is_bounded() {
        let boot = BootSequence::new();
        let s = boot.shimmer();
        assert!((0.0..1.0).contains(&s));
    }
}
