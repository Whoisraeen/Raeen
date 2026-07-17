//! Getting rendered guest frames onto the screen.
//!
//! Until this existed the GPU's only output was `.ppm` files on disk
//! (`XPS5X_DUMP_FRAMES`): a title could render perfectly and the window would
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

/// Presents the newest rendered guest frame, if there is one.
#[derive(Default)]
pub(crate) struct GameFrameView {
    texture: Option<egui::TextureHandle>,
    /// `draw_count` when `texture` was last refreshed. Fetching a frame clones
    /// a full render target out of the GPU session (~8 MB at 1080p), so it is
    /// only worth doing when a draw has actually landed since the last one —
    /// the UI repaints far faster than a title renders, and cloning 8 MB per UI
    /// frame would make the viewer cost more than the renderer.
    shown_at_draw: u64,
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
    pub(crate) fn paint(&mut self, ui: &egui::Ui, screen: egui::Rect) -> Presented {
        let session = xps5x_gpu::AgcGpuSession::global();
        let drawn = session.draw_count();
        if drawn != self.shown_at_draw || self.texture.is_none() {
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
                    self.shown_at_draw = drawn;
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

    /// Drop the frame when a session ends, so the next launch cannot open on the
    /// previous title's last frame.
    pub(crate) fn clear(&mut self) {
        self.texture = None;
        self.shown_at_draw = 0;
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
