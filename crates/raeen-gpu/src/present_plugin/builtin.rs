//! Built-in, vendor-neutral reference plugins.
//!
//! These are the plugins Raeen itself ships: all original Rust, no proprietary
//! dependencies, GPL-2.0-clean. They exist to (a) exercise and document the
//! [`PresentPlugin`](super::PresentPlugin) trait, (b) give the present path a
//! working default, and (c) prove the extension point is a *general* upscaler
//! ABI rather than a socket for one proprietary product.
//!
//! Real upscalers (a FidelityFX/FSR pass, which is MIT and can live in-tree)
//! are drop-in replacements for [`NearestUpscale`]; they implement the same
//! trait and read the same [`PresentContext`](super::PresentContext).

use super::{Capabilities, PluginOutput, PresentContext, PresentFrame, PresentPlugin};

/// Identity: presents the source frame unchanged. Semantically what `active ==
/// None` already does, but registered by name so it can be selected explicitly
/// (e.g. to A/B the plugin path against the fast path, or as the neutral choice
/// in a Settings dropdown).
#[derive(Debug, Default, Clone, Copy)]
pub struct Passthrough;

impl Passthrough {
    pub const NAME: &'static str = "passthrough";
}

impl PresentPlugin for Passthrough {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn process(&mut self, frame: &PresentFrame<'_>, _ctx: &PresentContext) -> PluginOutput {
        PluginOutput::identity(frame)
    }
}

/// Nearest-neighbour spatial upscaler — a minimal but *real* reference that
/// proves the boundary handles a resolution change end-to-end. Scales the frame
/// by [`PresentContext::output_scale`], falling back to identity when no scale
/// is requested or the source is malformed.
///
/// Nearest-neighbour is deliberately the simplest correct resampler: no new
/// dependencies, format-agnostic (it copies whole `bytes_per_pixel` texels, so
/// it works for both the 4-byte display formats and 8-byte HDR). A quality
/// upscaler (FSR) slots in here unchanged from the caller's point of view.
#[derive(Debug, Default, Clone, Copy)]
pub struct NearestUpscale;

impl NearestUpscale {
    pub const NAME: &'static str = "nearest";
}

impl PresentPlugin for NearestUpscale {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            upscale: true,
            ..Default::default()
        }
    }

    fn process(&mut self, frame: &PresentFrame<'_>, ctx: &PresentContext) -> PluginOutput {
        let scale = ctx.output_scale.clamp(1.0, 8.0);
        let bpp = frame.bytes_per_pixel as usize;
        let dst_w = ((frame.width as f32 * scale).round() as u32).max(1);
        let dst_h = ((frame.height as f32 * scale).round() as u32).max(1);

        let src_texels = (frame.width as usize).saturating_mul(frame.height as usize);
        let src_ok = bpp != 0 && frame.color.len() >= src_texels.saturating_mul(bpp);

        // Nothing to do (or can't safely do it) → identity.
        if (dst_w == frame.width && dst_h == frame.height) || !src_ok {
            return PluginOutput::identity(frame);
        }

        let mut pixels = vec![0u8; dst_w as usize * dst_h as usize * bpp];
        for y in 0..dst_h {
            // Map destination row to nearest source row.
            let sy = ((y as u64 * frame.height as u64) / dst_h as u64).min(frame.height as u64 - 1)
                as u32;
            for x in 0..dst_w {
                let sx = ((x as u64 * frame.width as u64) / dst_w as u64)
                    .min(frame.width as u64 - 1) as u32;
                let src = ((sy * frame.width + sx) as usize) * bpp;
                let dst = ((y * dst_w + x) as usize) * bpp;
                pixels[dst..dst + bpp].copy_from_slice(&frame.color[src..src + bpp]);
            }
        }

        PluginOutput {
            primary: super::PluginFrame {
                width: dst_w,
                height: dst_h,
                bytes_per_pixel: frame.bytes_per_pixel,
                pixels,
            },
            generated: Vec::new(),
        }
    }
}
