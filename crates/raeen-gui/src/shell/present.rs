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
/// Keep the child-runner sequence space distinct from the Shell's process-local
/// `AgcGpuSession::present_epoch()`. Both counters start near zero; without a
/// source bit, the first remote frame could numerically equal the splash epoch
/// and leave the splash texture cached until a second game frame arrived.
const REMOTE_EPOCH_BIT: u64 = 1 << 63;

fn display_epoch(local_epoch: u64, remote: Option<&raeen_gpu::frame_ipc::RemoteFrame>) -> u64 {
    remote.map_or(local_epoch, |frame| frame.epoch | REMOTE_EPOCH_BIT)
}

/// Whether the viewer must upload pixels for this observation.
///
/// `has_displayed_frame` is deliberately source-neutral. The native-wgpu path
/// stores its texture id in `displayed` and leaves the legacy managed
/// `texture` empty, so using `texture.is_none()` here turns every Shell repaint
/// into another full-frame upload even when the published epoch is unchanged.
fn needs_frame_refresh(epoch: u64, shown_at_epoch: u64, has_displayed_frame: bool) -> bool {
    epoch != shown_at_epoch || !has_displayed_frame
}

/// Rolling window of per-frame times backing the performance HUD. Sized so a
/// 60 FPS title keeps ~2 seconds of history — long enough for "worst" to catch
/// a hitch, short enough that recovery is visible quickly.
const FRAME_STAT_WINDOW: usize = 120;

/// Per-frame time statistics derived from published-frame epochs (the same
/// signal that refreshes the texture — it only advances when the GPU worker
/// publishes a COMPLETE frame, so this measures what the title actually
/// delivered, not egui's repaint cadence). Pure and clock-injected, so it is
/// unit-testable; no puffin, no profiler — always available.
#[derive(Default)]
struct FrameTimeStats {
    /// Wall-clock + epoch of the previous observation.
    last: Option<(Instant, u64)>,
    /// Per-frame milliseconds, oldest first, capped at [`FRAME_STAT_WINDOW`].
    samples: std::collections::VecDeque<f32>,
}

impl FrameTimeStats {
    /// Feed one observation of the display epoch. Multiple frames published
    /// between observations (the Shell repaints slower than the title renders)
    /// share the elapsed time evenly — each contributes one window sample, so
    /// the window stays frame-weighted, not observation-weighted.
    fn observe(&mut self, now: Instant, epoch: u64) {
        let Some((last_at, last_epoch)) = self.last else {
            self.last = Some((now, epoch));
            return;
        };
        if epoch == last_epoch {
            return;
        }
        // A source flip (local splash ↔ remote runner, the REMOTE_EPOCH_BIT)
        // or a counter reset (new process) is not a frame delta — rebaseline
        // instead of turning it into a bogus sample.
        if (epoch ^ last_epoch) & REMOTE_EPOCH_BIT != 0 || epoch < last_epoch {
            self.samples.clear();
            self.last = Some((now, epoch));
            return;
        }
        let frames = epoch - last_epoch;
        let per_frame_ms =
            now.saturating_duration_since(last_at).as_secs_f32() * 1000.0 / frames as f32;
        for _ in 0..frames.min(FRAME_STAT_WINDOW as u64) {
            if self.samples.len() == FRAME_STAT_WINDOW {
                self.samples.pop_front();
            }
            self.samples.push_back(per_frame_ms);
        }
        self.last = Some((now, epoch));
    }

    /// Mean frame time over the window, in milliseconds.
    fn avg_ms(&self) -> Option<f32> {
        if self.samples.is_empty() {
            return None;
        }
        Some(self.samples.iter().sum::<f32>() / self.samples.len() as f32)
    }

    /// Worst (longest) frame time over the window, in milliseconds.
    fn worst_ms(&self) -> Option<f32> {
        self.samples.iter().copied().reduce(f32::max)
    }

    /// Frames per second implied by the window's mean frame time.
    fn fps(&self) -> Option<f32> {
        self.avg_ms().filter(|ms| *ms > 0.0).map(|ms| 1000.0 / ms)
    }

    fn reset(&mut self) {
        self.last = None;
        self.samples.clear();
    }
}

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

/// A frame texture owned directly on egui's wgpu device, registered as a
/// native egui texture. Written with one `queue.write_texture` per published
/// frame — no `ColorImage` conversion, no per-frame allocation, no egui
/// texture-delta copy. Reused across frames (and titles) while the size
/// matches; freed and re-registered on resize.
struct NativeFrame {
    /// Kept alive for the registered view; rewritten in place every frame.
    texture: eframe::egui_wgpu::wgpu::Texture,
    id: egui::TextureId,
    width: u32,
    height: u32,
}

/// Presents the newest rendered guest frame, if there is one.
#[derive(Default)]
pub(crate) struct GameFrameView {
    texture: Option<egui::TextureHandle>,
    /// The zero-conversion present path (see [`NativeFrame`]); `texture`
    /// remains the fallback when no wgpu render state exists (tests,
    /// non-wgpu backends) or for non-RGBA8 frames.
    native: Option<NativeFrame>,
    /// What `paint` last uploaded and should draw: texture id + pixel size.
    /// `None` until the first complete frame (or after [`Self::clear`]).
    displayed: Option<(egui::TextureId, egui::Vec2)>,
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
    /// Rolling frame-time window over published-frame epochs — feeds the
    /// performance HUD (Settings ▸ Advanced ▸ Performance HUD / F3).
    frame_stats: FrameTimeStats,
    present_timing: Option<raeen_gpu::PresentTiming>,
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
    /// Upload `image` through the zero-conversion native-wgpu path. Returns
    /// the egui texture id + size to draw, or `None` when this frame can't
    /// take it (non-RGBA8) and must use the `ColorImage` fallback.
    fn upload_native(
        &mut self,
        render_state: &eframe::egui_wgpu::RenderState,
        image: &raeen_gpu::RenderedImage,
    ) -> Option<(egui::TextureId, egui::Vec2)> {
        use eframe::egui_wgpu::wgpu;
        if image.bytes_per_pixel != 4 {
            return None;
        }
        let (width, height) = (image.width, image.height);
        let recreate = self
            .native
            .as_ref()
            .is_none_or(|n| n.width != width || n.height != height);
        if recreate {
            if let Some(old) = self.native.take() {
                render_state.renderer.write().free_texture(&old.id);
            }
            let texture = render_state
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("raeen-guest-frame"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // Same encoding egui uses for its managed textures, so the
                    // native path is pixel-identical to the ColorImage path.
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let id = render_state.renderer.write().register_native_texture(
                &render_state.device,
                &view,
                wgpu::FilterMode::Linear,
            );
            self.native = Some(NativeFrame {
                texture,
                id,
                width,
                height,
            });
        }
        let native = self.native.as_ref().expect("just ensured above");
        render_state.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &native.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &image.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Some((native.id, egui::vec2(width as f32, height as f32)))
    }

    /// Paint the latest guest frame into `screen`, letterboxed.
    pub(crate) fn paint(
        &mut self,
        ui: &egui::Ui,
        screen: egui::Rect,
        presented_frames: Option<u64>,
        render_state: Option<&eframe::egui_wgpu::RenderState>,
    ) -> Presented {
        let session = raeen_gpu::AgcGpuSession::global();
        let remote = raeen_gpu::frame_ipc::latest_remote_frame();
        // The isolated runner owns the real VideoOut counter. Share it
        // separately from frame pixels so a static/reused complete image still
        // reports live presentation cadence instead of `0 FPS`. The old frame
        // sequence remains a compatibility fallback for a pre-v4 child.
        let measured_frames = raeen_gpu::frame_ipc::latest_remote_present_count()
            .or_else(|| remote.as_ref().map(|frame| frame.epoch / 2))
            .or(presented_frames);
        self.frame_rate.observe(Instant::now(), measured_frames);
        // Refresh on the GPU worker's published-frame epoch, NOT the guest-side
        // flip counter (`presented_frames`, still used only for the FPS badge):
        // under the bounded async flip the guest flips ahead of the frames the
        // worker has actually read back, so gating on the flip counter would
        // upload while `last_image` is still an older/None frame. The epoch
        // advances only when a COMPLETE frame is published, so the texture only
        // ever receives whole, finished frames — never a half-read one.
        let epoch = display_epoch(session.present_epoch(), remote.as_ref());
        self.frame_stats.observe(Instant::now(), epoch);
        if needs_frame_refresh(epoch, self.shown_at_epoch, self.displayed.is_some()) {
            // Deliberately does NOT `wait_idle()`: submission is asynchronous,
            // and blocking the UI thread until the GPU drained is exactly the
            // stall this Shell already had once. A viewer wants the latest frame
            // that exists, not a consistent one — a torn or one-frame-stale
            // image is invisible to a human and costs nothing.
            let image = remote
                .as_ref()
                .map(|frame| std::sync::Arc::clone(&frame.image))
                .or_else(|| session.last_image());
            if let Some(image) = image {
                let size = [image.width as usize, image.height as usize];
                if size[0] > 0 && size[1] > 0 && image.pixels.len() == size[0] * size[1] * 4 {
                    let upload_at = Instant::now();
                    // Preferred: one direct write into a native wgpu texture.
                    // Fallback (no wgpu render state, or a format the native
                    // path declines): egui's ColorImage upload, which costs a
                    // full-frame conversion copy first.
                    let uploaded = render_state
                        .and_then(|rs| self.upload_native(rs, &image))
                        .or_else(|| {
                            let color =
                                egui::ColorImage::from_rgba_unmultiplied(size, &image.pixels);
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
                            self.texture
                                .as_ref()
                                .map(|texture| (texture.id(), texture.size_vec2()))
                        });
                    self.displayed = uploaded;
                    let mut timing = remote
                        .as_ref()
                        .map_or_else(|| session.present_timing(), |frame| frame.timing);
                    timing.egui_upload_us = upload_at.elapsed().as_micros() as u64;
                    self.present_timing = Some(timing);
                    let timing_epoch = remote
                        .as_ref()
                        .map_or_else(|| session.present_epoch(), |frame| frame.epoch);
                    if timing_epoch <= 16 || timing_epoch.is_power_of_two() {
                        tracing::info!(
                            epoch = timing_epoch,
                            worker_drain_us = timing.worker_drain_us,
                            fence_wait_us = timing.fence_wait_us,
                            readback_us = timing.readback_us,
                            srgb_encode_us = timing.srgb_encode_us,
                            egui_upload_us = timing.egui_upload_us,
                            "PRESENT TIMING: latest completed frame"
                        );
                    }
                    self.shown_at_epoch = epoch;
                }
            }
        }

        let Some((texture_id, size)) = self.displayed else {
            return Presented::NoFrameYet;
        };
        let rect = letterbox(screen, size);
        ui.painter().image(
            texture_id,
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
            egui::pos2(bounds.right() - 372.0, bounds.top() + 12.0),
            egui::vec2(360.0, 48.0),
        );
        ui.painter().rect_filled(
            badge,
            6.0,
            egui::Color32::from_rgba_unmultiplied(8, 11, 18, 210),
        );
        ui.painter().text(
            egui::pos2(badge.right() - 10.0, badge.top() + 8.0),
            egui::Align2::RIGHT_TOP,
            self.frame_rate.label(),
            egui::FontId::monospace(15.0),
            egui::Color32::WHITE,
        );
        let timing = self.present_timing.unwrap_or_default();
        ui.painter().text(
            egui::pos2(badge.left() + 10.0, badge.bottom() - 8.0),
            egui::Align2::LEFT_BOTTOM,
            format!(
                "drain {:.1}  fence {:.1}  read {:.1}  sRGB {:.1}  UI {:.1} ms",
                timing.worker_drain_us as f64 / 1000.0,
                timing.fence_wait_us as f64 / 1000.0,
                timing.readback_us as f64 / 1000.0,
                timing.srgb_encode_us as f64 / 1000.0,
                timing.egui_upload_us as f64 / 1000.0,
            ),
            egui::FontId::monospace(11.0),
            egui::Color32::from_gray(205),
        );
    }

    /// The performance HUD (Settings ▸ Advanced ▸ Performance HUD, or F3): a
    /// superset of [`Self::paint_fps`] — epoch-derived FPS plus rolling
    /// avg/worst frame time, the flip-count rate, and the always-available
    /// present timing counters. Painter + explicit rects in the same top-right
    /// corner slot as the plain badge (only one of the two is drawn per frame),
    /// semi-transparent so the game stays visible beneath it. Works without
    /// puffin/RAEEN_PROFILE — everything here comes from `PresentTiming` and
    /// the published-frame epochs.
    pub(crate) fn paint_perf_hud(&self, ui: &egui::Ui, bounds: egui::Rect) {
        let panel = egui::Rect::from_min_size(
            egui::pos2(bounds.right() - 372.0, bounds.top() + 12.0),
            egui::vec2(360.0, 96.0),
        );
        let painter = ui.painter();
        painter.rect_filled(
            panel,
            6.0,
            egui::Color32::from_rgba_unmultiplied(8, 11, 18, 210),
        );
        let headline = match (
            self.frame_stats.fps(),
            self.frame_stats.avg_ms(),
            self.frame_stats.worst_ms(),
        ) {
            (Some(fps), Some(avg), Some(worst)) => {
                format!("{fps:.0} FPS   avg {avg:.1} ms   worst {worst:.1} ms")
            }
            _ => "-- FPS   no published frames yet".to_string(),
        };
        let timing = self.present_timing.unwrap_or_default();
        painter.text(
            egui::pos2(panel.left() + 12.0, panel.top() + 10.0),
            egui::Align2::LEFT_TOP,
            headline,
            egui::FontId::monospace(15.0),
            egui::Color32::WHITE,
        );
        painter.text(
            egui::pos2(panel.left() + 12.0, panel.top() + 36.0),
            egui::Align2::LEFT_TOP,
            format!(
                "flips {}   upload {:.2} ms",
                self.frame_rate.label(),
                timing.egui_upload_us as f64 / 1000.0,
            ),
            egui::FontId::monospace(12.0),
            egui::Color32::from_gray(205),
        );
        painter.text(
            egui::pos2(panel.left() + 12.0, panel.top() + 58.0),
            egui::Align2::LEFT_TOP,
            format!(
                "drain {:.1}  fence {:.1}  read {:.1}  sRGB {:.1} ms",
                timing.worker_drain_us as f64 / 1000.0,
                timing.fence_wait_us as f64 / 1000.0,
                timing.readback_us as f64 / 1000.0,
                timing.srgb_encode_us as f64 / 1000.0,
            ),
            egui::FontId::monospace(12.0),
            egui::Color32::from_gray(205),
        );
        painter.text(
            egui::pos2(panel.left() + 12.0, panel.bottom() - 8.0),
            egui::Align2::LEFT_BOTTOM,
            "F3 hides this HUD",
            egui::FontId::monospace(10.0),
            egui::Color32::from_gray(140),
        );
    }

    /// Drop the frame when a session ends, so the next launch cannot open on the
    /// previous title's last frame.
    pub(crate) fn clear(&mut self) {
        self.texture = None;
        // Stop DRAWING the old title's frame immediately. The native wgpu
        // texture itself is kept: its registration survives, its contents are
        // fully overwritten before `displayed` points at it again, and
        // freeing it here would need the renderer lock this method doesn't
        // have. `upload_native` frees + recreates it on any size change.
        self.displayed = None;
        self.shown_at_epoch = 0;
        self.frame_rate.reset();
        self.frame_stats.reset();
        self.present_timing = None;
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

    #[test]
    fn first_remote_frame_cannot_alias_the_local_splash_epoch() {
        let remote = raeen_gpu::frame_ipc::RemoteFrame {
            epoch: 2,
            timing: raeen_gpu::PresentTiming::default(),
            image: std::sync::Arc::new(raeen_gpu::RenderedImage {
                width: 1,
                height: 1,
                pixels: vec![4, 3, 2, 255],
                bytes_per_pixel: 4,
            }),
        };
        assert_ne!(display_epoch(2, Some(&remote)), display_epoch(2, None));
        assert_eq!(
            display_epoch(2, Some(&remote)),
            REMOTE_EPOCH_BIT | remote.epoch
        );
    }

    #[test]
    fn native_frame_is_not_reuploaded_until_its_epoch_advances() {
        let epoch = REMOTE_EPOCH_BIT | 16;

        // No frame has been uploaded yet: the first observation must upload.
        assert!(needs_frame_refresh(epoch, 0, false));
        // The native-wgpu path owns `displayed` but deliberately leaves the
        // legacy managed `texture` empty. Repainting the same completed frame
        // must not write all 4K pixels to the GPU again.
        assert!(!needs_frame_refresh(epoch, epoch, true));
        // A newly published frame still refreshes exactly once.
        assert!(needs_frame_refresh(epoch + 2, epoch, true));
    }

    #[test]
    fn frame_time_stats_measure_avg_and_worst_per_published_frame() {
        let start = Instant::now();
        let mut stats = FrameTimeStats::default();
        stats.observe(start, 10);
        // Baseline only — no sample yet.
        assert_eq!(stats.avg_ms(), None);
        assert_eq!(stats.fps(), None);
        // One frame published 16 ms later.
        stats.observe(start + Duration::from_millis(16), 11);
        // Then a hitch: one frame that took 48 ms.
        stats.observe(start + Duration::from_millis(64), 12);
        let avg = stats.avg_ms().expect("two samples");
        let worst = stats.worst_ms().expect("two samples");
        assert!((avg - 32.0).abs() < 0.5, "avg was {avg}");
        assert!((worst - 48.0).abs() < 0.5, "worst was {worst}");
        let fps = stats.fps().expect("fps from avg");
        assert!((fps - 1000.0 / 32.0).abs() < 0.5, "fps was {fps}");
    }

    #[test]
    fn frame_time_stats_share_elapsed_time_across_skipped_epochs() {
        // The Shell repaints slower than the title publishes: 4 frames landed
        // in one 64 ms observation → four 16 ms samples, not one 64 ms one.
        let start = Instant::now();
        let mut stats = FrameTimeStats::default();
        stats.observe(start, 100);
        stats.observe(start + Duration::from_millis(64), 104);
        assert_eq!(stats.samples.len(), 4);
        let avg = stats.avg_ms().expect("samples");
        assert!((avg - 16.0).abs() < 0.5, "avg was {avg}");
    }

    #[test]
    fn frame_time_stats_window_is_capped_and_rolls() {
        let start = Instant::now();
        let mut stats = FrameTimeStats::default();
        stats.observe(start, 0);
        // A slow frame first, then far more fast frames than the window holds:
        // the slow sample must roll out, taking "worst" down with it.
        stats.observe(start + Duration::from_millis(100), 1);
        let mut now = start + Duration::from_millis(100);
        for i in 0..(FRAME_STAT_WINDOW as u64 + 8) {
            now += Duration::from_millis(10);
            stats.observe(now, 2 + i);
        }
        assert_eq!(stats.samples.len(), FRAME_STAT_WINDOW);
        let worst = stats.worst_ms().expect("full window");
        assert!(worst < 11.0, "slow sample should have rolled out: {worst}");
    }

    #[test]
    fn frame_time_stats_rebaseline_on_source_flip_and_counter_reset() {
        let start = Instant::now();
        let mut stats = FrameTimeStats::default();
        stats.observe(start, 5);
        stats.observe(start + Duration::from_millis(16), 6);
        assert!(stats.avg_ms().is_some());
        // Local → remote source flip (the high bit): not a frame delta.
        stats.observe(start + Duration::from_millis(32), REMOTE_EPOCH_BIT | 2);
        assert_eq!(stats.avg_ms(), None);
        stats.observe(start + Duration::from_millis(48), REMOTE_EPOCH_BIT | 3);
        assert!(stats.avg_ms().is_some());
        // A lower epoch (new/reset process) also rebaselines instead of
        // wrapping into a huge bogus delta.
        stats.observe(start + Duration::from_millis(64), REMOTE_EPOCH_BIT | 1);
        assert_eq!(stats.avg_ms(), None);
    }

    #[test]
    fn frame_time_stats_reset_clears_everything() {
        let start = Instant::now();
        let mut stats = FrameTimeStats::default();
        stats.observe(start, 1);
        stats.observe(start + Duration::from_millis(16), 2);
        assert!(stats.avg_ms().is_some());
        stats.reset();
        assert_eq!(stats.avg_ms(), None);
        assert_eq!(stats.worst_ms(), None);
        assert!(stats.last.is_none());
    }

    #[test]
    fn remote_sequence_exports_the_child_present_count_for_fps() {
        let start = Instant::now();
        let mut rate = PresentedFrameRate::default();
        rate.observe(start, Some(10 / 2));
        rate.observe(start + Duration::from_secs(1), Some(130 / 2));
        assert_eq!(rate.label(), "60 FPS");
    }
}
