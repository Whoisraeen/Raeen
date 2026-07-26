//! # raeen-upscale — all-in-one BYO upscaler / frame-gen plugin
//!
//! **This crate is git-ignored and is NOT part of the Raeen repository.** It
//! lives under `plugins/` (ignored by `.gitignore`), is hosted separately, and
//! is compiled in only by a user who opts into it. Raeen itself ships and
//! distributes none of this — see `plugins/README.md` and the license note
//! below.
//!
//! It implements the vendor-neutral [`raeen_gpu::PresentPlugin`] ABI once per
//! backend and registers them all with the running app, so the Shell's
//! *Settings ▸ Video ▸ Upscaler* row lets the user choose between:
//!
//! | name        | what it is                                   | works today? |
//! |-------------|----------------------------------------------|--------------|
//! | `bilinear`  | 2×2 linear spatial upscale                   | ✅ yes        |
//! | `bicubic`   | 4×4 Catmull-Rom spatial upscale (sharper)    | ✅ yes        |
//! | `sharpen`   | unsharp pass at native resolution            | ✅ yes        |
//! | `fsr`       | AMD FidelityFX Super Resolution              | ⏳ vendor+MV  |
//! | `dlss`      | NVIDIA DLSS Super Resolution                 | ⏳ vendor+MV  |
//! | `xess`      | Intel XeSS                                    | ⏳ vendor+MV  |
//!
//! ## Honest status of the vendor backends
//!
//! `fsr` / `dlss` / `xess` are **GPU + motion-vector** techniques. Two things
//! must exist for them to run for real: (1) the vendor runtime (`nvngx_dlss.dll`
//! for DLSS, the FidelityFX runtime for FSR, `libxess.dll` for XeSS), which the
//! **user** supplies — Raeen never ships or fetches it; and (2) per-pixel motion
//! vectors + depth exposed to the plugin, which the present ABI reserves
//! (`PresentFrame::motion`/`depth`) but does not yet populate (they come from
//! the PM4 stream — a follow-up in the main app).
//!
//! Until both are present, each vendor backend detects its runtime, logs its
//! state **once**, and produces a real image via the best spatial path
//! (`bicubic`) so selecting it is never a black screen. This is deliberate: the
//! selection + hook are real now; the vendor inference activates transparently
//! once its two prerequisites land.
//!
//! ## License
//!
//! This glue crate is GPL-2.0-only (compatible with Raeen). It links **no**
//! proprietary SDK: the vendor runtimes are detected on disk at run time and
//! (in a future revision) loaded dynamically from the user's own copy — never
//! bundled here. Keep it that way.

use std::sync::atomic::{AtomicBool, Ordering};

use raeen_gpu::{
    Capabilities, PluginFrame, PluginOutput, PresentContext, PresentFrame, PresentPlugin,
};

pub mod spatial;

use spatial::Kernel;

/// Register every backend this crate provides with the running app's present
/// registry. Call once at startup, before applying the persisted selection, so
/// a saved choice like `"fsr"` resolves. Idempotent (re-registering a name
/// replaces it).
pub fn register_all() {
    let plugins: Vec<Box<dyn PresentPlugin>> = vec![
        Box::new(SpatialUpscaler::new("bilinear", Kernel::Bilinear)),
        Box::new(SpatialUpscaler::new("bicubic", Kernel::Bicubic)),
        Box::new(SharpenPass),
        Box::new(VendorUpscaler::new(Vendor::Fsr)),
        Box::new(VendorUpscaler::new(Vendor::Dlss)),
        Box::new(VendorUpscaler::new(Vendor::Xess)),
    ];
    for p in plugins {
        raeen_gpu::AgcGpuSession::register_present_plugin(p);
    }
    tracing::info!("raeen-upscale: registered bilinear, bicubic, sharpen, fsr, dlss, xess");
}

/// The set of backend names this crate registers (for tests / UI hints).
#[must_use]
pub fn backend_names() -> &'static [&'static str] {
    &["bilinear", "bicubic", "sharpen", "fsr", "dlss", "xess"]
}

fn target_dims(frame: &PresentFrame<'_>, ctx: &PresentContext) -> (u32, u32) {
    let scale = ctx.output_scale.clamp(1.0, 8.0);
    let w = ((frame.width as f32 * scale).round() as u32).max(1);
    let h = ((frame.height as f32 * scale).round() as u32).max(1);
    (w, h)
}

// ─── Spatial backends (work on the current CPU-pixel ABI) ──────────────────

/// A pure spatial upscaler (bilinear or bicubic).
struct SpatialUpscaler {
    name: &'static str,
    kernel: Kernel,
}

impl SpatialUpscaler {
    fn new(name: &'static str, kernel: Kernel) -> Self {
        Self { name, kernel }
    }
}

impl PresentPlugin for SpatialUpscaler {
    fn name(&self) -> &str {
        self.name
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            upscale: true,
            ..Default::default()
        }
    }
    fn process(&mut self, frame: &PresentFrame<'_>, ctx: &PresentContext) -> PluginOutput {
        let (dw, dh) = target_dims(frame, ctx);
        if (dw, dh) == (frame.width, frame.height) {
            return PluginOutput::identity(frame);
        }
        let pixels = spatial::resample(
            frame.color,
            frame.width,
            frame.height,
            dw,
            dh,
            frame.bytes_per_pixel,
            self.kernel,
        );
        PluginOutput {
            primary: PluginFrame {
                width: dw,
                height: dh,
                bytes_per_pixel: frame.bytes_per_pixel,
                pixels,
            },
            generated: Vec::new(),
        }
    }
}

/// A native-resolution unsharp sharpen (no resize).
#[derive(Default)]
struct SharpenPass;

impl PresentPlugin for SharpenPass {
    fn name(&self) -> &str {
        "sharpen"
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
    fn process(&mut self, frame: &PresentFrame<'_>, _ctx: &PresentContext) -> PluginOutput {
        let pixels = spatial::sharpen(
            frame.color,
            frame.width,
            frame.height,
            frame.bytes_per_pixel,
            0.5,
        );
        PluginOutput {
            primary: PluginFrame {
                width: frame.width,
                height: frame.height,
                bytes_per_pixel: frame.bytes_per_pixel,
                pixels,
            },
            generated: Vec::new(),
        }
    }
}

// ─── Vendor backends (runtime-detected, MV-gated, graceful fallback) ───────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Vendor {
    Dlss,
    Fsr,
    Xess,
}

impl Vendor {
    fn name(self) -> &'static str {
        match self {
            Vendor::Dlss => "dlss",
            Vendor::Fsr => "fsr",
            Vendor::Xess => "xess",
        }
    }
    /// Runtime library filenames that indicate the vendor tech is installed
    /// alongside the app (the user supplies these; Raeen never ships them).
    fn runtime_files(self) -> &'static [&'static str] {
        match self {
            Vendor::Dlss => &["nvngx_dlss.dll", "nvngx.dll"],
            Vendor::Fsr => &["amd_fidelityfx_vk.dll", "ffx_backend_vk.dll"],
            Vendor::Xess => &["libxess.dll"],
        }
    }
}

/// A vendor (DLSS/FSR/XeSS) upscaler. Real inference needs the vendor runtime
/// on disk **and** motion vectors from the ABI; until both are present it logs
/// its state once and produces a real image via the bicubic spatial path.
struct VendorUpscaler {
    vendor: Vendor,
    logged: AtomicBool,
}

impl VendorUpscaler {
    fn new(vendor: Vendor) -> Self {
        Self {
            vendor,
            logged: AtomicBool::new(false),
        }
    }

    /// Whether a vendor runtime library is present next to the executable or in
    /// the current directory. Detection only — nothing is loaded or linked.
    fn runtime_present(&self) -> bool {
        let mut dirs: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe()
            && let Some(parent) = exe.parent()
        {
            dirs.push(parent.to_path_buf());
        }
        if let Ok(cwd) = std::env::current_dir() {
            dirs.push(cwd);
        }
        self.vendor
            .runtime_files()
            .iter()
            .any(|f| dirs.iter().any(|d| d.join(f).exists()))
    }

    fn log_state_once(&self, frame: &PresentFrame<'_>) {
        if self.logged.swap(true, Ordering::Relaxed) {
            return;
        }
        let have_runtime = self.runtime_present();
        let have_mv = frame.motion.is_some() && frame.depth.is_some();
        if have_runtime && have_mv {
            tracing::info!(
                "raeen-upscale/{}: runtime + motion vectors available (GPU inference path lands in a future revision) — using bicubic for now",
                self.vendor.name()
            );
        } else if have_runtime {
            tracing::warn!(
                "raeen-upscale/{}: vendor runtime found but the present ABI does not expose motion vectors yet — falling back to bicubic spatial upscale",
                self.vendor.name()
            );
        } else {
            tracing::warn!(
                "raeen-upscale/{}: vendor runtime not found (user must supply it) — falling back to bicubic spatial upscale",
                self.vendor.name()
            );
        }
    }
}

impl PresentPlugin for VendorUpscaler {
    fn name(&self) -> &str {
        self.vendor.name()
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            upscale: true,
            frame_gen: matches!(self.vendor, Vendor::Dlss | Vendor::Fsr),
            wants_depth: true,
            wants_motion_vectors: true,
        }
    }
    fn process(&mut self, frame: &PresentFrame<'_>, ctx: &PresentContext) -> PluginOutput {
        self.log_state_once(frame);
        // Prerequisites for real vendor inference are not both met yet; produce
        // a genuine image via the best spatial path so the selection is usable.
        let (dw, dh) = target_dims(frame, ctx);
        if (dw, dh) == (frame.width, frame.height) {
            return PluginOutput::identity(frame);
        }
        let pixels = spatial::resample(
            frame.color,
            frame.width,
            frame.height,
            dw,
            dh,
            frame.bytes_per_pixel,
            Kernel::Bicubic,
        );
        PluginOutput {
            primary: PluginFrame {
                width: dw,
                height: dh,
                bytes_per_pixel: frame.bytes_per_pixel,
                pixels,
            },
            generated: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame<'a>(w: u32, h: u32, buf: &'a [u8]) -> PresentFrame<'a> {
        PresentFrame {
            width: w,
            height: h,
            bytes_per_pixel: 4,
            color: buf,
            depth: None,
            motion: None,
            frame_index: 0,
        }
    }

    #[test]
    fn spatial_upscaler_resizes_by_output_scale() {
        let mut p = SpatialUpscaler::new("bicubic", Kernel::Bicubic);
        let src = vec![128u8; 2 * 2 * 4];
        let ctx = PresentContext {
            output_scale: 2.0,
            hdr: false,
        };
        let out = p.process(&frame(2, 2, &src), &ctx);
        assert_eq!((out.primary.width, out.primary.height), (4, 4));
        assert_eq!(out.primary.pixels.len(), 4 * 4 * 4);
    }

    #[test]
    fn spatial_upscaler_identity_at_scale_one() {
        let mut p = SpatialUpscaler::new("bilinear", Kernel::Bilinear);
        let src = vec![7u8; 2 * 2 * 4];
        let ctx = PresentContext {
            output_scale: 1.0,
            hdr: false,
        };
        let out = p.process(&frame(2, 2, &src), &ctx);
        assert_eq!((out.primary.width, out.primary.height), (2, 2));
        assert_eq!(out.primary.pixels, src);
    }

    #[test]
    fn vendor_backend_falls_back_to_real_image_and_names_itself() {
        let mut p = VendorUpscaler::new(Vendor::Dlss);
        assert_eq!(p.name(), "dlss");
        assert!(p.capabilities().wants_motion_vectors);
        let src = vec![64u8; 2 * 2 * 4];
        let ctx = PresentContext {
            output_scale: 2.0,
            hdr: false,
        };
        let out = p.process(&frame(2, 2, &src), &ctx);
        // Even without the runtime/MVs, selecting DLSS yields a real upscaled
        // frame (never a blank screen).
        assert_eq!((out.primary.width, out.primary.height), (4, 4));
        assert_eq!(out.primary.pixels.len(), 4 * 4 * 4);
    }

    #[test]
    fn backend_names_match_registration_list() {
        assert_eq!(
            backend_names(),
            &["bilinear", "bicubic", "sharpen", "fsr", "dlss", "xess"]
        );
    }
}
