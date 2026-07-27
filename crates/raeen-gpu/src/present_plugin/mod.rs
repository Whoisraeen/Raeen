//! # Present-path plugin ABI — upscalers and frame generators
//!
//! A generic, **vendor-neutral** boundary at the present chokepoint
//! ([`AgcGpuSession::publish_frame`](crate::AgcGpuSession)): every *complete*
//! frame about to be shown is offered to the currently-active
//! [`PresentPlugin`], which may upscale it and/or (later) generate interpolated
//! frames. FSR, XeSS, community experiments, and an out-of-tree DLSS shim all
//! implement the **same** trait — nothing in this module is specific to any one
//! product.
//!
//! ## Why this boundary exists here
//!
//! `publish_frame` is the single path through which a title frame ever reaches
//! the Shell, so hooking it once means every presented frame — and only
//! complete frames — passes the plugin. The default is a zero-cost identity: if
//! no plugin is selected, [`apply_to_image`] returns the original `Arc`
//! untouched (no copy, no behaviour change).
//!
//! ## Clean-room / license boundary (important)
//!
//! Raeen is **GPL-2.0-only**. It ships only the vendor-neutral built-in plugins
//! in [`builtin`] (all original Rust). Proprietary plugins — e.g. an NVIDIA
//! DLSS/Streamline shim — are **never** vendored, fetched, named as
//! "supported", or bundled here. They live in the git-ignored `plugins/` tree,
//! are hosted separately, and are loaded only if a *user* supplies them and
//! registers them via [`register`]. This keeps Raeen's distributed artifact
//! 100% GPL-2.0-compatible (it contains no proprietary code and never fetches
//! any) while still letting a user assemble a DLSS-enabled build on their own
//! machine for private use. See `plugins/README.md` for the full rationale.
//!
//! The API is deliberately generic — it is *not* a DLSS socket. That an FSR
//! reference path fills the same trait is what proves the extension point is a
//! legitimate architecture rather than a copyleft-evasion device.

pub mod builtin;
pub mod cabi;

use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use crate::vulkan::RenderedImage;

/// What a plugin can do, so the Shell can label it and the present path can
/// decide what auxiliary inputs to gather. All-false is a pure passthrough.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Capabilities {
    /// Produces an output frame at a different (larger) resolution.
    pub upscale: bool,
    /// Produces interpolated frames between real frames (frame generation).
    pub frame_gen: bool,
    /// Wants the guest depth buffer for a temporally-stable result.
    pub wants_depth: bool,
    /// Wants per-pixel motion vectors (the quality moat only an emulator that
    /// owns the command stream can supply). Extraction from the PM4 stream is a
    /// follow-up; the flag lets an MV-aware plugin advertise the need now.
    pub wants_motion_vectors: bool,
}

/// An optional auxiliary input plane (depth or motion vectors) accompanying the
/// color frame. Present only when Raeen can extract it from the PM4 stream (the
/// motion-vector-aware path). `None` today at the CPU-readback present site —
/// the fields exist so the ABI is stable when that extraction lands.
#[derive(Debug, Clone, Copy)]
pub struct AuxPlane<'a> {
    pub width: u32,
    pub height: u32,
    pub bytes_per_texel: u32,
    pub data: &'a [u8],
}

/// A complete, ready-to-present source frame handed to a plugin (borrowed).
///
/// `color` is `width * height * bytes_per_pixel` bytes, row-major, no padding —
/// the exact [`RenderedImage`] contract. `bytes_per_pixel` is 4 for the 8-bit
/// display formats and packed B10G11R11, 8 for R16G16B16A16 (HDR).
#[derive(Debug, Clone, Copy)]
pub struct PresentFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub color: &'a [u8],
    /// Guest depth, when available (see [`AuxPlane`]). `None` for now.
    pub depth: Option<AuxPlane<'a>>,
    /// Per-pixel motion vectors, when available. `None` for now.
    pub motion: Option<AuxPlane<'a>>,
    /// Monotonic present index — lets a temporal plugin detect discontinuities.
    pub frame_index: u64,
}

/// An owned frame produced by a plugin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginFrame {
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    /// `width * height * bytes_per_pixel` bytes, row-major, no padding.
    pub pixels: Vec<u8>,
}

/// Per-frame context the present path supplies to a plugin.
#[derive(Debug, Clone, Copy)]
pub struct PresentContext {
    /// Present-time upscale factor requested by the user (`1.0` = native).
    /// Distinct from the *render* resolution scale in `GpuRuntimeConfig`: this
    /// scales the finished frame at present, that scales the draws.
    pub output_scale: f32,
    /// The source color is HDR (`bytes_per_pixel == 8`, R16G16B16A16), so a
    /// plugin can select an HDR-correct path.
    pub hdr: bool,
}

/// A plugin's output for one source frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginOutput {
    /// The frame to present in place of the source (upscaled, or unchanged).
    pub primary: PluginFrame,
    /// Frame-generation output: interpolated frames to present *before*
    /// `primary`. Empty for pure upscalers. **Reserved** — the present path
    /// does not yet schedule these (frame-gen pacing is a follow-up); the field
    /// exists so the ABI is already stable for frame-gen plugins.
    pub generated: Vec<PluginFrame>,
}

impl PluginOutput {
    /// The trivial "unchanged" output: the source pixels, no generated frames.
    /// The base every plugin can fall back to when it declines a frame.
    #[must_use]
    pub fn identity(frame: &PresentFrame<'_>) -> Self {
        Self {
            primary: PluginFrame {
                width: frame.width,
                height: frame.height,
                bytes_per_pixel: frame.bytes_per_pixel,
                pixels: frame.color.to_vec(),
            },
            generated: Vec::new(),
        }
    }
}

/// The upscaler / frame-generator plugin ABI.
///
/// Implemented by the built-in reference plugins ([`builtin`]) and by
/// out-of-tree, user-supplied plugins (FSR/XeSS/DLSS shims, community work).
/// `process` takes `&mut self` so a temporal plugin can retain prior-frame
/// state. Implementations must be cheap enough to run once per presented frame.
pub trait PresentPlugin: Send + Sync {
    /// A short, stable, unique identifier (also the selection key and the label
    /// the Shell shows). Vendor-neutral built-ins use plain names
    /// (`"passthrough"`, `"nearest"`).
    fn name(&self) -> &str;

    /// What this plugin does, for UI labelling and input gathering.
    fn capabilities(&self) -> Capabilities;

    /// Where this plugin was loaded from: `Some(path)` for an out-of-tree
    /// binary (`cabi::DynamicPlugin`), `None` for a plugin compiled into the
    /// artifact (built-ins, in-tree Rust plugins). Shown by the Shell's
    /// Plugins UI so a user can tell built-in from BYO at a glance.
    fn source_path(&self) -> Option<&std::path::Path> {
        None
    }

    /// Transform one complete source frame into the frame(s) to present.
    fn process(&mut self, frame: &PresentFrame<'_>, ctx: &PresentContext) -> PluginOutput;
}

/// Everything the Shell's Plugins UI needs to describe one registered plugin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginInfo {
    pub name: String,
    pub capabilities: Capabilities,
    /// `Some(path)` for a user-supplied out-of-tree binary, `None` for a
    /// built-in / in-tree plugin.
    pub source: Option<std::path::PathBuf>,
}

/// The process-wide plugin registry: the set of registered plugins, which one
/// (if any) is active, and the present-time upscale factor. `active == None`
/// is the identity fast path.
struct Registry {
    plugins: BTreeMap<String, Box<dyn PresentPlugin>>,
    active: Option<String>,
    output_scale: f32,
    frame_index: u64,
}

impl Registry {
    /// A fresh registry pre-populated with the vendor-neutral built-ins and no
    /// active plugin (identity). Used for the global singleton and directly by
    /// unit tests (so tests never touch global state).
    fn new() -> Self {
        let mut reg = Self {
            plugins: BTreeMap::new(),
            active: None,
            output_scale: 1.0,
            frame_index: 0,
        };
        reg.insert(Box::new(builtin::Passthrough));
        reg.insert(Box::new(builtin::NearestUpscale));
        reg
    }

    fn insert(&mut self, plugin: Box<dyn PresentPlugin>) {
        self.plugins.insert(plugin.name().to_string(), plugin);
    }

    fn select(&mut self, name: &str) -> bool {
        if self.plugins.contains_key(name) {
            self.active = Some(name.to_string());
            true
        } else {
            false
        }
    }

    fn list(&self) -> Vec<(String, Capabilities)> {
        self.plugins
            .iter()
            .map(|(name, plugin)| (name.clone(), plugin.capabilities()))
            .collect()
    }

    fn list_info(&self) -> Vec<PluginInfo> {
        self.plugins
            .iter()
            .map(|(name, plugin)| PluginInfo {
                name: name.clone(),
                capabilities: plugin.capabilities(),
                source: plugin.source_path().map(std::path::Path::to_path_buf),
            })
            .collect()
    }

    /// Run the active plugin, or `None` for the identity fast path.
    fn apply_frame(
        &mut self,
        frame: &PresentFrame<'_>,
        ctx: &PresentContext,
    ) -> Option<PluginOutput> {
        let name = self.active.clone()?;
        let plugin = self.plugins.get_mut(&name)?;
        Some(plugin.process(frame, ctx))
    }
}

fn registry() -> &'static Mutex<Registry> {
    static REG: OnceLock<Mutex<Registry>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(Registry::new()))
}

// ─── Public API (Shell + out-of-tree BYO plugins) ──────────────────────────

/// Register a plugin. Out-of-tree, user-supplied plugin crates (e.g. an FSR or
/// a BYO DLSS shim) call this to add themselves; the name replaces any existing
/// plugin with the same name. Registering does **not** activate it — call
/// [`select`].
pub fn register(plugin: Box<dyn PresentPlugin>) {
    registry().lock().insert(plugin);
}

/// Make the named plugin active. Returns `false` (and changes nothing) if no
/// plugin with that name is registered.
pub fn select(name: &str) -> bool {
    registry().lock().select(name)
}

/// Deactivate any plugin — restore the zero-cost identity present path.
pub fn select_none() {
    registry().lock().active = None;
}

/// The active plugin's name, or `None` when the identity path is in effect.
#[must_use]
pub fn active() -> Option<String> {
    registry().lock().active.clone()
}

/// `(name, capabilities)` for every registered plugin — for a Settings dropdown.
#[must_use]
pub fn list() -> Vec<(String, Capabilities)> {
    registry().lock().list()
}

/// Full [`PluginInfo`] (name, capabilities, source path) for every registered
/// plugin — for the Shell's Plugins UI.
#[must_use]
pub fn list_info() -> Vec<PluginInfo> {
    registry().lock().list_info()
}

/// Refusals from the most recent out-of-tree plugin scan, one human-readable
/// line per refused candidate. Replaced (not appended) on every scan so the UI
/// always shows the current state of `plugins/`, and empty when the last scan
/// refused nothing.
fn load_failures_store() -> &'static Mutex<Vec<String>> {
    static FAILURES: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    FAILURES.get_or_init(|| Mutex::new(Vec::new()))
}

/// Replace the recorded refusals from the latest plugin-directory scan.
pub(crate) fn set_load_failures(failures: Vec<String>) {
    *load_failures_store().lock() = failures;
}

/// The latest scan's refusals — why each candidate in `plugins/` was not
/// loaded. For the Shell's Plugins UI; the same information is also logged.
#[must_use]
pub fn load_failures() -> Vec<String> {
    load_failures_store().lock().clone()
}

/// Set the present-time upscale factor an upscaler plugin should target
/// (`1.0` = native). Clamped to a sane range.
pub fn set_output_scale(scale: f32) {
    registry().lock().output_scale = scale.clamp(0.5, 8.0);
}

/// Offer a finished frame to the active plugin, returning the frame to actually
/// present. The identity fast path returns the **same** `Arc` (no copy) when no
/// plugin is active, so the default present cost is unchanged.
///
/// Called from `AgcGpuSession::publish_frame`. Auxiliary planes (depth/motion)
/// are `None` today; they will be populated when PM4-side extraction lands.
#[must_use]
pub(crate) fn apply_to_image(image: Arc<RenderedImage>) -> Arc<RenderedImage> {
    let mut reg = registry().lock();
    let ctx = PresentContext {
        output_scale: reg.output_scale,
        hdr: image.bytes_per_pixel == 8,
    };
    reg.frame_index = reg.frame_index.wrapping_add(1);
    let frame = PresentFrame {
        width: image.width,
        height: image.height,
        bytes_per_pixel: image.bytes_per_pixel,
        color: &image.pixels,
        depth: None,
        motion: None,
        frame_index: reg.frame_index,
    };
    match reg.apply_frame(&frame, &ctx) {
        // Identity fast path: no active plugin → present the original, no copy.
        None => {
            drop(reg);
            image
        }
        Some(output) => Arc::new(RenderedImage {
            width: output.primary.width,
            height: output.primary.height,
            pixels: output.primary.pixels,
            bytes_per_pixel: output.primary.bytes_per_pixel,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal test plugin: reports a fixed name/capabilities and tints the
    /// first byte of every pixel so we can prove `process` ran.
    struct TintPlugin;
    impl PresentPlugin for TintPlugin {
        fn name(&self) -> &str {
            "test-tint"
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                upscale: false,
                frame_gen: true,
                ..Default::default()
            }
        }
        fn process(&mut self, frame: &PresentFrame<'_>, _ctx: &PresentContext) -> PluginOutput {
            let mut out = PluginOutput::identity(frame);
            for px in out
                .primary
                .pixels
                .chunks_mut(frame.bytes_per_pixel as usize)
            {
                if let Some(first) = px.first_mut() {
                    *first = 0xAB;
                }
            }
            out
        }
    }

    fn frame<'a>(w: u32, h: u32, bpp: u32, buf: &'a [u8]) -> PresentFrame<'a> {
        PresentFrame {
            width: w,
            height: h,
            bytes_per_pixel: bpp,
            color: buf,
            depth: None,
            motion: None,
            frame_index: 0,
        }
    }

    #[test]
    fn list_info_reports_builtins_with_no_source_path() {
        let reg = Registry::new();
        let infos = reg.list_info();
        let passthrough = infos
            .iter()
            .find(|i| i.name == "passthrough")
            .expect("passthrough is a built-in");
        assert_eq!(
            passthrough.source, None,
            "built-ins are compiled in, not loaded from a file"
        );
        assert_eq!(passthrough.capabilities, Capabilities::default());
        let nearest = infos
            .iter()
            .find(|i| i.name == "nearest")
            .expect("nearest is a built-in");
        assert!(nearest.capabilities.upscale);
    }

    #[test]
    fn load_failures_replace_rather_than_accumulate() {
        set_load_failures(vec!["first scan: bad plugin".to_string()]);
        assert_eq!(load_failures(), vec!["first scan: bad plugin".to_string()]);
        // A clean rescan clears the record — refusals never linger after the
        // user removes the offending file.
        set_load_failures(Vec::new());
        assert!(load_failures().is_empty());
    }

    #[test]
    fn builtins_registered_and_neutral_by_default() {
        // A fresh registry ships the vendor-neutral built-ins and is identity
        // (no active plugin) — the zero-behaviour-change default.
        let reg = Registry::new();
        let names: Vec<_> = reg.list().into_iter().map(|(n, _)| n).collect();
        assert!(names.contains(&"passthrough".to_string()));
        assert!(names.contains(&"nearest".to_string()));
        assert!(reg.active.is_none(), "default must be identity");
    }

    #[test]
    fn passthrough_is_pixel_identity() {
        let mut reg = Registry::new();
        assert!(reg.select("passthrough"));
        let src = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let ctx = PresentContext {
            output_scale: 1.0,
            hdr: false,
        };
        let out = reg.apply_frame(&frame(2, 1, 4, &src), &ctx).unwrap();
        assert_eq!(out.primary.width, 2);
        assert_eq!(out.primary.height, 1);
        assert_eq!(out.primary.pixels, src, "passthrough must not alter pixels");
        assert!(out.generated.is_empty());
    }

    #[test]
    fn no_active_plugin_is_none() {
        // Identity fast path signalled by `None` (caller keeps the original).
        let mut reg = Registry::new();
        let src = vec![0u8; 16];
        let ctx = PresentContext {
            output_scale: 2.0,
            hdr: false,
        };
        assert!(reg.apply_frame(&frame(2, 2, 4, &src), &ctx).is_none());
    }

    #[test]
    fn nearest_upscales_dimensions_and_replicates() {
        let mut reg = Registry::new();
        assert!(reg.select("nearest"));
        // 2x1, two distinct RGBA pixels.
        let src = vec![10, 11, 12, 13, 20, 21, 22, 23];
        let ctx = PresentContext {
            output_scale: 2.0,
            hdr: false,
        };
        let out = reg.apply_frame(&frame(2, 1, 4, &src), &ctx).unwrap();
        assert_eq!((out.primary.width, out.primary.height), (4, 2));
        assert_eq!(
            out.primary.pixels.len(),
            4 * 2 * 4,
            "output buffer sized for the upscaled frame"
        );
        // Nearest: top-left of the upscaled frame is the first source pixel.
        assert_eq!(&out.primary.pixels[0..4], &[10, 11, 12, 13]);
    }

    #[test]
    fn nearest_is_identity_at_scale_one() {
        let mut reg = Registry::new();
        assert!(reg.select("nearest"));
        let src = vec![9u8; 16];
        let ctx = PresentContext {
            output_scale: 1.0,
            hdr: false,
        };
        let out = reg.apply_frame(&frame(2, 2, 4, &src), &ctx).unwrap();
        assert_eq!((out.primary.width, out.primary.height), (2, 2));
        assert_eq!(out.primary.pixels, src);
    }

    #[test]
    fn register_select_roundtrip_and_unknown_rejected() {
        let mut reg = Registry::new();
        reg.insert(Box::new(TintPlugin));
        let listed: Vec<_> = reg.list().into_iter().map(|(n, _)| n).collect();
        assert!(listed.contains(&"test-tint".to_string()));
        assert!(reg.select("test-tint"));
        assert!(!reg.select("does-not-exist"), "unknown selection rejected");
        // The prior valid selection stays put after a rejected one.
        let src = vec![0u8; 8];
        let ctx = PresentContext {
            output_scale: 1.0,
            hdr: false,
        };
        let out = reg.apply_frame(&frame(2, 1, 4, &src), &ctx).unwrap();
        assert_eq!(out.primary.pixels[0], 0xAB, "active plugin actually ran");
        assert_eq!(out.primary.pixels[4], 0xAB);
    }
}
