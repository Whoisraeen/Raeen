//! Getting rendered guest frames onto the screen.
//!
//! Until this existed the GPU's only output was `.ppm` files on disk
//! (`RAEEN_DUMP_FRAMES`): a title could render perfectly and the window would
//! still be black, because nothing ever carried a frame to it. Every question
//! about rendering had to be answered by grepping logs and doing arithmetic.
//!
//! This is deliberately NOT the M3 swapchain. `libSceVideoOut` flip →
//! `VK_KHR_swapchain` is the real presentation path and is still owed. This
//! takes the frame the offscreen renderer already read back to host memory and
//! hands it to egui as a texture, which costs nothing extra today: the renderer
//! reads every draw back to the CPU anyway (the round trip that makes it slow),
//! so the pixels are already sitting in host memory. When render targets move
//! GPU-side that readback becomes one-per-flip instead of one-per-draw, and this
//! keeps working — it only ever asked for "the latest frame, as host pixels".
//!
//! As of GPU stage C that is where things stand: ONE readback per flip (only
//! the flipped/presented target crosses to the CPU; other render targets stay
//! GPU-side). Going below one — true zero-copy presentation — was assessed and
//! deliberately not built: eframe renders through wgpu on its own device, while
//! the guest GPU renders through ash on a second `VkDevice`, so a shared image
//! needs either (a) re-hosting the whole guest renderer on wgpu's raw Vulkan
//! device, or (b) `VK_KHR_external_memory_win32` export/import plus an
//! `egui_wgpu` paint-callback path. Both are real integrations (M3 swapchain
//! territory), not an optimization of this view.

use std::time::{Duration, Instant};

/// Minimum wall-clock interval between rate updates. The guest flip counter is
/// process-owned and monotonic, so measuring its delta over real elapsed time
/// stays honest even when egui repaints faster (or slower) than the title.
const FPS_SAMPLE_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Default)]
struct PresentedFrameRate {
    baseline: Option<(Instant, u64)>,
    fps: Option<f64>,
}

impl PresentedFrameRate {
    fn observe(&mut self, now: Instant, presented_frames: Option<u64>) {
        let Some(presented_frames) = presented_frames else {
            self.reset();
            return;
        };
        let Some((baseline_at, baseline_frames)) = self.baseline else {
            self.baseline = Some((now, presented_frames));
            return;
        };

        // A lower count means a new/reset process. Never turn that into a huge
        // wrapped delta; establish a fresh baseline for the new title instead.
        if presented_frames < baseline_frames {
            self.baseline = Some((now, presented_frames));
            self.fps = None;
            return;
        }

        let elapsed = now.saturating_duration_since(baseline_at);
        if elapsed < FPS_SAMPLE_INTERVAL {
            return;
        }
        self.fps = Some((presented_frames - baseline_frames) as f64 / elapsed.as_secs_f64());
        self.baseline = Some((now, presented_frames));
    }

    fn label(&self) -> String {
        self.fps
            .map_or_else(|| "-- FPS".to_string(), |fps| format!("{fps:.0} FPS"))
    }

    fn reset(&mut self) {
        self.baseline = None;
        self.fps = None;
    }
}

/// Presents the newest rendered guest frame, if there is one.
#[derive(Default)]
pub(crate) struct GameFrameView {
    texture: Option<egui::TextureHandle>,
    /// The GPU session's `present_epoch` when `texture` was last refreshed.
    /// `last_image()` is now an `Arc` bump (no frame copy), but REFRESHING the
    /// texture still costs a full 8 MB (1080p RGBA) `ColorImage` build plus a
    /// CPU→GPU upload into egui's own wgpu device — a whole-frame crossing. The
    /// UI repaints far faster than a title renders, so this gate keeps that
    /// crossing to once per newly PUBLISHED complete frame.
    ///
    /// The epoch advances only when the GPU worker publishes a COMPLETE
    /// (fence-read-back or flip-time-snapshot) frame — or a splash comes down —
    /// so it is the async-flip-safe refresh signal: it never fires while
    /// `last_image` is a half-read frame, and it does not chase the guest-side
    /// VideoOut flip counter, which under the bounded async flip races ahead of
    /// the frames the worker has actually finished. This is what lets the Shell
    /// follow the freshest completed frame instead of black-screening on the
    /// async path. It also subsumes the old draw-count and flip-count triggers:
    /// a CPU-drawn scanout that changes with no GPU draw still bumps the epoch
    /// when it is published.
    shown_at_epoch: u64,
    frame_rate: PresentedFrameRate,
}

/// What the viewer managed to show, so the caller can say something honest
/// about an empty screen instead of leaving the user guessing.
pub(crate) enum Presented {
    /// A frame was painted into this rect.
    Frame { rect: egui::Rect },
    /// The title has not rendered anything yet.
    NoFrameYet,
}

impl GameFrameView {
    /// Paint the latest guest frame into `screen`, letterboxed.
    pub(crate) fn paint(
        &mut self,
        ui: &egui::Ui,
        screen: egui::Rect,
        presented_frames: Option<u64>,
    ) -> Presented {
        self.frame_rate.observe(Instant::now(), presented_frames);
        let session = raeen_gpu::AgcGpuSession::global();
        // Refresh on the GPU worker's published-frame epoch, NOT the guest-side
        // flip counter (`presented_frames`, still used only for the FPS badge):
        // under the bounded async flip the guest flips ahead of the frames the
        // worker has actually read back, so gating on the flip counter would
        // upload while `last_image` is still an older/None frame. The epoch
        // advances only when a COMPLETE frame is published, so the texture only
        // ever receives whole, finished frames — never a half-read one.
        let epoch = session.present_epoch();
        if epoch != self.shown_at_epoch || self.texture.is_none() {
            // Deliberately does NOT `wait_idle()`: submission is asynchronous,
            // and blocking the UI thread until the GPU drained is exactly the
            // stall this Shell already had once. A viewer wants the latest frame
            // that exists, not a consistent one — a torn or one-frame-stale
            // image is invisible to a human and costs nothing.
            if let Some(image) = session.last_image() {
                let size = [image.width as usize, image.height as usize];
                if size[0] > 0 && size[1] > 0 && image.pixels.len() == size[0] * size[1] * 4 {
                    let color = egui::ColorImage::from_rgba_unmultiplied(size, &image.pixels);
                    match &mut self.texture {
                        Some(texture) => texture.set(color, egui::TextureOptions::LINEAR),
                        None => {
                            self.texture = Some(ui.ctx().load_texture(
                                "guest-frame",
                                color,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                    }
                    self.shown_at_epoch = epoch;
                }
            }
        }

        let Some(texture) = &self.texture else {
            return Presented::NoFrameYet;
        };
        let rect = letterbox(screen, texture.size_vec2());
        ui.painter().image(
            texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        Presented::Frame { rect }
    }

    /// Paint the measured guest presentation rate. This deliberately consumes
    /// only `sceVideoOut` flip-count deltas; egui repaint cadence is irrelevant.
    pub(crate) fn paint_fps(&self, ui: &egui::Ui, bounds: egui::Rect) {
        let badge = egui::Rect::from_min_size(
            egui::pos2(bounds.right() - 92.0, bounds.top() + 12.0),
            egui::vec2(80.0, 30.0),
        );
        ui.painter().rect_filled(
            badge,
            6.0,
            egui::Color32::from_rgba_unmultiplied(8, 11, 18, 210),
        );
        ui.painter().text(
            badge.center(),
            egui::Align2::CENTER_CENTER,
            self.frame_rate.label(),
            egui::FontId::monospace(15.0),
            egui::Color32::WHITE,
        );
    }

    /// Drop the frame when a session ends, so the next launch cannot open on the
    /// previous title's last frame.
    pub(crate) fn clear(&mut self) {
        self.texture = None;
        self.shown_at_epoch = 0;
        self.frame_rate.reset();
    }
}

/// Fit `content` inside `screen` preserving aspect ratio.
///
/// A guest frame is whatever resolution the title chose (1920x1080 for the
/// measured titles) and the window is whatever the user dragged it to, so the
/// two almost never agree. Stretching to fill would silently misreport what the
/// emulator rendered — the thing this view exists to show honestly.
fn letterbox(screen: egui::Rect, content: egui::Vec2) -> egui::Rect {
    if content.x <= 0.0 || content.y <= 0.0 {
        return screen;
    }
    let scale = (screen.width() / content.x).min(screen.height() / content.y);
    egui::Rect::from_center_size(screen.center(), content * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presented_frame_rate_uses_guest_flip_delta_over_real_time() {
        let start = Instant::now();
        let mut rate = PresentedFrameRate::default();
        rate.observe(start, Some(10));
        // UI repaints do not themselves add frames.
        rate.observe(start + Duration::from_millis(250), Some(10));
        assert_eq!(rate.label(), "-- FPS");
        rate.observe(start + Duration::from_secs(1), Some(130));
        assert_eq!(rate.label(), "120 FPS");
    }

    #[test]
    fn presented_frame_rate_reports_zero_when_guest_stops_flipping() {
        let start = Instant::now();
        let mut rate = PresentedFrameRate::default();
        rate.observe(start, Some(7));
        rate.observe(start + FPS_SAMPLE_INTERVAL, Some(7));
        assert_eq!(rate.label(), "0 FPS");
    }

    #[test]
    fn presented_frame_rate_rebaselines_when_process_counter_resets() {
        let start = Instant::now();
        let mut rate = PresentedFrameRate::default();
        rate.observe(start, Some(100));
        rate.observe(start + Duration::from_secs(1), Some(160));
        assert_eq!(rate.label(), "60 FPS");
        rate.observe(start + Duration::from_secs(2), Some(2));
        assert_eq!(rate.label(), "-- FPS");
    }

    #[test]
    fn letterbox_preserves_aspect_and_fits_inside_the_screen() {
        let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 800.0));
        // 16:9 content in a square window pillarboxes: full width, shorter height.
        let fitted = letterbox(screen, egui::vec2(1920.0, 1080.0));
        assert!((fitted.width() - 800.0).abs() < 0.01);
        assert!((fitted.height() - 450.0).abs() < 0.01);
        assert!(screen.contains_rect(fitted));
        assert_eq!(fitted.center(), screen.center());
    }

    #[test]
    fn letterbox_fits_tall_content_by_height() {
        let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 400.0));
        let fitted = letterbox(screen, egui::vec2(1000.0, 1000.0));
        assert!((fitted.height() - 400.0).abs() < 0.01);
        assert!((fitted.width() - 400.0).abs() < 0.01);
        assert!(screen.contains_rect(fitted));
    }

    /// A zero-sized frame must not produce a NaN rect (scale would divide by
    /// zero) — an unrendered target reports 0x0 before the first draw lands.
    #[test]
    fn degenerate_content_falls_back_to_the_screen_rect() {
        let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 400.0));
        assert_eq!(letterbox(screen, egui::vec2(0.0, 0.0)), screen);
    }
}
