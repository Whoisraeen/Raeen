//! # Stable C ABI + runtime loading for out-of-tree present plugins
//!
//! [`super::PresentPlugin`] is a Rust trait, so a plugin implementing it must be
//! *compiled into* `raeen-gpu`. That is exactly the arrangement
//! `plugins/README.md` forbids for a proprietary plugin: linking it into the
//! distributed artifact. This module closes that gap with a `#[repr(C)]` vtable
//! and `dlopen`/`LoadLibrary` loading, so a plugin is a **separate binary the
//! user supplies**, loaded at runtime, never linked at build time.
//!
//! ## The contract a plugin binary must satisfy
//!
//! Export one symbol with C linkage:
//!
//! ```c
//! const RaeenPluginV1 *raeen_plugin_v1(void);
//! ```
//!
//! returning a pointer to a statically-lived [`RaeenPluginV1`] whose
//! `abi_version` equals [`RAEEN_PLUGIN_ABI_VERSION`]. Raeen then calls
//! `create()` once, `name()`/`capabilities()` to describe it, `process()` per
//! presented frame, `release_output()` after copying each result, and
//! `destroy()` at teardown.
//!
//! ## Ownership across the boundary
//!
//! The **plugin** allocates its output pixels and the **plugin** frees them:
//! Raeen copies the bytes it needs, then calls `release_output`. Neither side
//! ever frees the other's allocation, so the two may use different allocators,
//! languages, and CRTs. `release_output` is called whenever `process` returned
//! success — including when Raeen then rejects the output as malformed.
//!
//! ## Trust boundary — read this
//!
//! Loading a plugin executes **arbitrary native code inside the Raeen process**.
//! No validation here can make that safe; a hostile or buggy plugin can corrupt
//! memory or crash the process outright. What this module *does* guarantee is
//! that Raeen never *itself* commits undefined behaviour on well-formed-but-
//! wrong plugin output: every pointer is null-checked, every length is validated
//! against the dimensions that describe it, and any inconsistency degrades to
//! the identity frame with a named, rate-limited warning rather than a bad
//! index. Plugins are opt-in, user-supplied, and never fetched by Raeen.
//!
//! ## License boundary
//!
//! This ABI is vendor-neutral and carries no product-specific concept. Raeen
//! ships no plugin binaries and downloads none. See `plugins/README.md`.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use super::{Capabilities, PluginFrame, PluginOutput, PresentContext, PresentFrame, PresentPlugin};

/// ABI generation understood by this build. A plugin reporting anything else is
/// refused — the vtable layout is only meaningful for a matching version.
pub const RAEEN_PLUGIN_ABI_VERSION: u32 = 1;

/// The one symbol a plugin binary must export (NUL-terminated for `dlsym`).
pub const RAEEN_PLUGIN_ENTRY: &[u8] = b"raeen_plugin_v1\0";

// ─── Capability bits (mirrors `Capabilities`, stable across the boundary) ────

/// Produces an output frame at a different (larger) resolution.
pub const RAEEN_CAP_UPSCALE: u32 = 1 << 0;
/// Produces interpolated frames between real frames.
pub const RAEEN_CAP_FRAME_GEN: u32 = 1 << 1;
/// Wants the guest depth buffer.
pub const RAEEN_CAP_WANTS_DEPTH: u32 = 1 << 2;
/// Wants per-pixel motion vectors.
pub const RAEEN_CAP_WANTS_MOTION_VECTORS: u32 = 1 << 3;

/// `process` succeeded and populated the output.
pub const RAEEN_OK: i32 = 0;

// ─── Sanity bounds ──────────────────────────────────────────────────────────
//
// A plugin is allowed to change the resolution, so the output dimensions cannot
// simply be compared to the input. These bounds keep a garbage descriptor from
// becoming a multi-gigabyte allocation or an overflowing index computation.

/// Largest accepted output edge, in texels. Comfortably past 8K.
const MAX_EDGE: u32 = 16_384;
/// Largest accepted output buffer. 16384x16384x8 would be 2 GiB; cap well under.
const MAX_OUTPUT_BYTES: usize = 1 << 30;
/// Largest accepted plugin name, in bytes.
const MAX_NAME_BYTES: usize = 128;
/// Largest accepted count of generated (frame-gen) frames for one source frame.
const MAX_GENERATED: usize = 8;

// ─── `#[repr(C)]` mirror types ──────────────────────────────────────────────

/// An optional auxiliary input plane (depth or motion vectors).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenAuxPlane {
    pub width: u32,
    pub height: u32,
    pub bytes_per_texel: u32,
    pub _reserved: u32,
    pub data: *const u8,
    pub len: usize,
}

/// The source frame handed to a plugin. `depth`/`motion` are null when Raeen
/// cannot supply them (both are null today — PM4-side extraction is a follow-up).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPresentFrame {
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub _reserved: u32,
    pub color: *const u8,
    pub color_len: usize,
    pub depth: *const RaeenAuxPlane,
    pub motion: *const RaeenAuxPlane,
    pub frame_index: u64,
}

/// Per-frame context. `hdr` is `0`/`1` — C `bool` has no guaranteed width.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPresentContext {
    pub output_scale: f32,
    pub hdr: u32,
}

/// One frame produced by a plugin. The plugin owns `pixels` until
/// `release_output`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPluginFrame {
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub _reserved: u32,
    pub pixels: *const u8,
    pub pixels_len: usize,
}

impl RaeenPluginFrame {
    /// A zeroed frame, for a plugin to fill or for Raeen to hand to `process`.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            bytes_per_pixel: 0,
            _reserved: 0,
            pixels: std::ptr::null(),
            pixels_len: 0,
        }
    }
}

/// A plugin's output for one source frame.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPluginOutput {
    pub primary: RaeenPluginFrame,
    /// Interpolated frames to present *before* `primary`. Null/0 for pure
    /// upscalers. Reserved: the present path does not yet schedule these.
    pub generated: *const RaeenPluginFrame,
    pub generated_count: usize,
}

impl RaeenPluginOutput {
    /// A zeroed output, the buffer Raeen passes into `process`.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            primary: RaeenPluginFrame::empty(),
            generated: std::ptr::null(),
            generated_count: 0,
        }
    }
}

/// Create a plugin instance. Returns null on failure.
pub type RaeenCreateFn = unsafe extern "C" fn() -> *mut c_void;
/// Destroy an instance previously returned by `create`.
pub type RaeenDestroyFn = unsafe extern "C" fn(*mut c_void);
/// Write the plugin's name into `buf` (no NUL required), returning the number of
/// bytes written. Returning more than `cap` is an error and refuses the plugin.
///
/// Length-bounded on purpose: a returned `const char *` would force Raeen to
/// scan for a NUL that a buggy plugin might never place.
pub type RaeenNameFn = unsafe extern "C" fn(*mut c_void, buf: *mut u8, cap: usize) -> usize;
/// Return the capability bits (`RAEEN_CAP_*`).
pub type RaeenCapabilitiesFn = unsafe extern "C" fn(*mut c_void) -> u32;
/// Transform one frame. Returns [`RAEEN_OK`] on success; any other value means
/// "declined", and Raeen presents the source frame unchanged.
pub type RaeenProcessFn = unsafe extern "C" fn(
    *mut c_void,
    *const RaeenPresentFrame,
    *const RaeenPresentContext,
    *mut RaeenPluginOutput,
) -> i32;
/// Free everything a successful `process` allocated.
pub type RaeenReleaseOutputFn = unsafe extern "C" fn(*mut c_void, *mut RaeenPluginOutput);

/// The vtable a plugin binary returns from `raeen_plugin_v1()`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPluginV1 {
    pub abi_version: u32,
    pub _reserved: u32,
    pub create: RaeenCreateFn,
    pub destroy: RaeenDestroyFn,
    pub name: RaeenNameFn,
    pub capabilities: RaeenCapabilitiesFn,
    pub process: RaeenProcessFn,
    pub release_output: RaeenReleaseOutputFn,
}

/// The entry point signature resolved from the loaded binary.
pub type RaeenPluginEntryFn = unsafe extern "C" fn() -> *const RaeenPluginV1;

// ─── ABI v2: GPU-capable frames ─────────────────────────────────────────────
//
// v2 exists so a plugin written TODAY keeps working unchanged when the
// GPU-resident present path lands. Every frame carries a `kind`
// discriminator: `CPU` (what Raeen delivers now — identical semantics to v1)
// or `VULKAN` (live Vulkan handles, delivered once frames stay GPU-resident).
// The host announces what it will deliver in [`RaeenHostContextV2`] at
// `create`; a plugin that only supports one kind declines the other from
// `process` (a declined frame is presented unchanged — never an error).
//
// The ABI stays strictly vendor-neutral: opaque Vulkan handles and a loader
// hook, nothing specific to any upscaler product. The license boundary is
// unchanged from v1 (see the module docs and `plugins/README.md`): Raeen
// ships no proprietary plugin, fetches none, and the out-of-tree `.dll` is
// the only shape closed code may take.

/// ABI generation for [`RaeenPluginV2`].
pub const RAEEN_PLUGIN_ABI_V2: u32 = 2;

/// The v2 entry point (NUL-terminated for `dlsym`). A binary may export both
/// `raeen_plugin_v1` and `raeen_plugin_v2`; when v2 is present it is
/// authoritative and must be valid.
pub const RAEEN_PLUGIN_ENTRY_V2: &[u8] = b"raeen_plugin_v2\0";

/// Capability bit: the plugin can consume `RAEEN_FRAME_KIND_VULKAN` frames.
pub const RAEEN_CAP_GPU_FRAMES: u32 = 1 << 4;

/// Host flag: this Raeen will deliver `RAEEN_FRAME_KIND_VULKAN` frames.
/// Currently never set — frames are CPU buffers until the GPU-resident
/// present path lands; the flag is how that lands without an ABI break.
pub const RAEEN_HOST_GPU_FRAMES: u32 = 1 << 0;

/// `RaeenPresentFrameV2::kind`: `color`/`color_len` point at CPU pixels.
pub const RAEEN_FRAME_KIND_CPU: u32 = 0;
/// `RaeenPresentFrameV2::kind`: `color_image` carries live Vulkan handles.
pub const RAEEN_FRAME_KIND_VULKAN: u32 = 1;

/// Opaque Vulkan dispatch context handed to a v2 plugin at `create`. All
/// fields are zero until the host sets [`RAEEN_HOST_GPU_FRAMES`]. Handles are
/// `u64` so the header needs no Vulkan types; `get_instance_proc_addr` is a
/// `PFN_vkGetInstanceProcAddr` the plugin may use to load everything else.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenVulkanContext {
    pub instance: u64,
    pub physical_device: u64,
    pub device: u64,
    pub queue: u64,
    pub queue_family: u32,
    pub _reserved: u32,
    pub get_instance_proc_addr: *const c_void,
}

impl RaeenVulkanContext {
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            instance: 0,
            physical_device: 0,
            device: 0,
            queue: 0,
            queue_family: 0,
            _reserved: 0,
            get_instance_proc_addr: std::ptr::null(),
        }
    }
}

/// One GPU-resident color image (valid only while `kind == VULKAN`).
/// `vk_format`/`layout` are the raw `VkFormat`/`VkImageLayout` values.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenVulkanImage {
    pub image: u64,
    pub image_view: u64,
    pub vk_format: u32,
    pub layout: u32,
}

impl RaeenVulkanImage {
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            image: 0,
            image_view: 0,
            vk_format: 0,
            layout: 0,
        }
    }
}

/// What the host tells a v2 plugin at `create`. Valid only for the duration
/// of the call — copy what you need. `struct_size` lets a plugin detect a
/// newer host extending this struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenHostContextV2 {
    pub struct_size: u32,
    /// `RAEEN_HOST_*` bits — what this host will deliver.
    pub host_flags: u32,
    pub vulkan: RaeenVulkanContext,
}

/// The v2 source frame. With `kind == CPU`, `color`/`color_len`/`depth`/
/// `motion` have exactly the v1 semantics and `color_image` is zeroed; with
/// `kind == VULKAN` the pointers are null and `color_image` is live.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPresentFrameV2 {
    pub kind: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub color: *const u8,
    pub color_len: usize,
    pub color_image: RaeenVulkanImage,
    pub depth: *const RaeenAuxPlane,
    pub motion: *const RaeenAuxPlane,
    pub frame_index: u64,
}

/// The v2 output: the v1 CPU output plus a reserved GPU-produced image.
/// `produced_kind` mirrors the frame kinds; the host reads `produced_image`
/// only when it advertised [`RAEEN_HOST_GPU_FRAMES`] (never today).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPluginOutputV2 {
    pub base: RaeenPluginOutput,
    pub produced_kind: u32,
    pub _reserved: u32,
    pub produced_image: RaeenVulkanImage,
}

impl RaeenPluginOutputV2 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            base: RaeenPluginOutput::empty(),
            produced_kind: RAEEN_FRAME_KIND_CPU,
            _reserved: 0,
            produced_image: RaeenVulkanImage::zeroed(),
        }
    }
}

/// Create a v2 plugin instance. `host` is valid only during the call.
pub type RaeenCreateV2Fn = unsafe extern "C" fn(*const RaeenHostContextV2) -> *mut c_void;
/// Transform one v2 frame. [`RAEEN_OK`] on success; anything else declines.
pub type RaeenProcessV2Fn = unsafe extern "C" fn(
    *mut c_void,
    *const RaeenPresentFrameV2,
    *const RaeenPresentContext,
    *mut RaeenPluginOutputV2,
) -> i32;
/// Free everything a successful v2 `process` allocated.
pub type RaeenReleaseOutputV2Fn = unsafe extern "C" fn(*mut c_void, *mut RaeenPluginOutputV2);

/// The vtable a plugin binary returns from `raeen_plugin_v2()`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RaeenPluginV2 {
    pub abi_version: u32,
    pub _reserved: u32,
    pub create: RaeenCreateV2Fn,
    pub destroy: RaeenDestroyFn,
    pub name: RaeenNameFn,
    pub capabilities: RaeenCapabilitiesFn,
    pub process: RaeenProcessV2Fn,
    pub release_output: RaeenReleaseOutputV2Fn,
}

/// The v2 entry point signature resolved from the loaded binary.
pub type RaeenPluginEntryV2Fn = unsafe extern "C" fn() -> *const RaeenPluginV2;

// ─── Load errors ────────────────────────────────────────────────────────────

/// Why a candidate plugin binary was refused. Every variant names the file, so
/// a refusal is actionable rather than a silent skip.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("could not open plugin library {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error(
        "plugin {path} exports no `raeen_plugin_v1` entry point (not a Raeen plugin?): {source}"
    )]
    MissingEntry {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("plugin {path} returned a null vtable from `raeen_plugin_v1`")]
    NullVtable { path: PathBuf },
    #[error(
        "plugin {path} reports ABI version {found}, this build understands {RAEEN_PLUGIN_ABI_VERSION}"
    )]
    AbiMismatch { path: PathBuf, found: u32 },
    #[error("plugin {path} `create` returned null")]
    CreateFailed { path: PathBuf },
    #[error("plugin {path} reported an unusable name ({reason})")]
    BadName { path: PathBuf, reason: &'static str },
}

// ─── The adapter ────────────────────────────────────────────────────────────

/// A loaded out-of-tree plugin, adapted to the in-tree [`PresentPlugin`] trait.
///
/// Field order is load-bearing for teardown: [`Drop`] calls `destroy` on the
/// instance while `_library` is still loaded (a `Drop` impl body runs before any
/// field is dropped), and `_library` is declared last so it unloads only after
/// everything borrowed from it is gone.
pub struct DynamicPlugin {
    name: String,
    capabilities: Capabilities,
    /// Points into the loaded image; valid only while `_library` is alive.
    vtable: PluginVtable,
    instance: *mut c_void,
    source: PathBuf,
    /// `None` for a vtable supplied directly (tests, or a statically-known
    /// plugin); `Some` keeps a `dlopen`ed image alive.
    _library: Option<libloading::Library>,
}

/// Which ABI generation this plugin speaks. Both pointers alias into the
/// loaded image and share `DynamicPlugin`'s lifetime rules.
#[derive(Clone, Copy)]
enum PluginVtable {
    V1(*const RaeenPluginV1),
    V2(*const RaeenPluginV2),
}

// SAFETY: every call into the vtable happens through `&mut self` (`process`) or
// `&self` (`name`/`capabilities`), and the process-wide registry that owns every
// plugin serializes all access behind a `Mutex`. So Raeen never calls into one
// plugin instance from two threads at once, which is the contract stated for
// plugin authors. A plugin that spawns its own threads and mutates shared state
// behind Raeen's back violates that contract; loading arbitrary native code is a
// user-accepted trust boundary (see the module docs) and no `unsafe impl` here
// can extend a guarantee the foreign code declines to keep.
unsafe impl Send for DynamicPlugin {}
// SAFETY: as above — shared access is read-only (`name`, `capabilities`) and the
// registry mutex serializes the `&mut self` path.
unsafe impl Sync for DynamicPlugin {}

impl std::fmt::Debug for DynamicPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicPlugin")
            .field("name", &self.name)
            .field("capabilities", &self.capabilities)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl DynamicPlugin {
    /// Where this plugin was loaded from — shown in diagnostics so a
    /// misbehaving plugin is attributable to a file.
    #[must_use]
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Adapt an already-resolved vtable, optionally keeping its library alive.
    ///
    /// Validates the ABI version, creates the instance, and reads the name and
    /// capabilities. Used by [`load_from_path`] and directly by tests (which
    /// supply a vtable of ordinary `extern "C"` functions and no library).
    ///
    /// # Safety
    ///
    /// `vtable` must be non-null, point to a valid [`RaeenPluginV1`] that
    /// outlives the returned plugin, and contain function pointers that honour
    /// the contract in the module docs. When `library` is `Some`, `vtable` must
    /// point into that library's image.
    pub unsafe fn from_vtable(
        vtable: *const RaeenPluginV1,
        library: Option<libloading::Library>,
        source: PathBuf,
    ) -> Result<Self, LoadError> {
        if vtable.is_null() {
            return Err(LoadError::NullVtable { path: source });
        }
        // SAFETY: caller guarantees `vtable` points at a live `RaeenPluginV1`.
        let vt = unsafe { *vtable };

        // Check the ABI *before* calling anything else through the vtable: on a
        // mismatch the layout we just read may not be the layout the plugin
        // wrote, so no other field is trustworthy.
        if vt.abi_version != RAEEN_PLUGIN_ABI_VERSION {
            return Err(LoadError::AbiMismatch {
                path: source,
                found: vt.abi_version,
            });
        }

        // SAFETY: ABI version matches, so `create` is a valid function pointer.
        let instance = unsafe { (vt.create)() };
        if instance.is_null() {
            return Err(LoadError::CreateFailed { path: source });
        }

        // Read the name into a bounded buffer we own.
        let mut buf = [0u8; MAX_NAME_BYTES];
        // SAFETY: `buf` is a live, writable `MAX_NAME_BYTES` buffer and we pass
        // its true capacity; the plugin may write at most that many bytes.
        let written = unsafe { (vt.name)(instance, buf.as_mut_ptr(), buf.len()) };

        let bad = |reason: &'static str, instance: *mut c_void| {
            // SAFETY: `instance` came from `create` and has not been destroyed.
            unsafe { (vt.destroy)(instance) };
            LoadError::BadName {
                path: source.clone(),
                reason,
            }
        };
        if written == 0 {
            return Err(bad("empty", instance));
        }
        if written > buf.len() {
            return Err(bad("longer than the 128-byte limit", instance));
        }
        let name = match std::str::from_utf8(&buf[..written]) {
            Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
            Ok(_) => return Err(bad("blank", instance)),
            Err(_) => return Err(bad("not valid UTF-8", instance)),
        };

        // SAFETY: instance is live; `capabilities` is a plain value return.
        let bits = unsafe { (vt.capabilities)(instance) };

        Ok(Self {
            name,
            capabilities: capabilities_from_bits(bits),
            vtable: PluginVtable::V1(vtable),
            instance,
            source,
            _library: library,
        })
    }

    /// Adapt an already-resolved **v2** vtable, optionally keeping its
    /// library alive. Same contract as [`Self::from_vtable`], plus: `create`
    /// receives a [`RaeenHostContextV2`] valid only during the call. Today it
    /// advertises no GPU frames (`host_flags == 0`, zeroed Vulkan context),
    /// so a v2 plugin sees exactly v1's CPU-frame world through v2 types.
    ///
    /// # Safety
    ///
    /// As [`Self::from_vtable`], with `vtable` pointing at a live
    /// [`RaeenPluginV2`].
    pub unsafe fn from_vtable_v2(
        vtable: *const RaeenPluginV2,
        library: Option<libloading::Library>,
        source: PathBuf,
    ) -> Result<Self, LoadError> {
        if vtable.is_null() {
            return Err(LoadError::NullVtable { path: source });
        }
        // SAFETY: caller guarantees `vtable` points at a live `RaeenPluginV2`.
        let vt = unsafe { *vtable };
        if vt.abi_version != RAEEN_PLUGIN_ABI_V2 {
            return Err(LoadError::AbiMismatch {
                path: source,
                found: vt.abi_version,
            });
        }

        let host = RaeenHostContextV2 {
            struct_size: std::mem::size_of::<RaeenHostContextV2>() as u32,
            host_flags: 0, // no GPU-resident frames yet
            vulkan: RaeenVulkanContext::zeroed(),
        };
        // SAFETY: ABI version matches; `host` is a live local for the call.
        let instance = unsafe { (vt.create)(&host) };
        if instance.is_null() {
            return Err(LoadError::CreateFailed { path: source });
        }

        let mut buf = [0u8; MAX_NAME_BYTES];
        // SAFETY: bounded buffer with its true capacity, as in v1.
        let written = unsafe { (vt.name)(instance, buf.as_mut_ptr(), buf.len()) };
        let bad = |reason: &'static str, instance: *mut c_void| {
            // SAFETY: `instance` came from `create` and has not been destroyed.
            unsafe { (vt.destroy)(instance) };
            LoadError::BadName {
                path: source.clone(),
                reason,
            }
        };
        if written == 0 {
            return Err(bad("empty", instance));
        }
        if written > buf.len() {
            return Err(bad("longer than the 128-byte limit", instance));
        }
        let name = match std::str::from_utf8(&buf[..written]) {
            Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
            Ok(_) => return Err(bad("blank", instance)),
            Err(_) => return Err(bad("not valid UTF-8", instance)),
        };

        // SAFETY: instance is live; `capabilities` is a plain value return.
        let bits = unsafe { (vt.capabilities)(instance) };

        Ok(Self {
            name,
            capabilities: capabilities_from_bits(bits),
            vtable: PluginVtable::V2(vtable),
            instance,
            source,
            _library: library,
        })
    }
}

impl Drop for DynamicPlugin {
    fn drop(&mut self) {
        if self.instance.is_null() {
            return;
        }
        // SAFETY: the vtable was validated at construction and the library
        // that backs it is still loaded — `_library` is a field, and fields
        // drop only after this body returns.
        unsafe {
            match self.vtable {
                PluginVtable::V1(vt) if !vt.is_null() => ((*vt).destroy)(self.instance),
                PluginVtable::V2(vt) if !vt.is_null() => ((*vt).destroy)(self.instance),
                _ => {}
            }
        }
        self.instance = std::ptr::null_mut();
    }
}

/// Translate ABI capability bits into the in-tree [`Capabilities`]. Unknown bits
/// are ignored so a newer plugin degrades rather than being refused.
#[must_use]
fn capabilities_from_bits(bits: u32) -> Capabilities {
    Capabilities {
        upscale: bits & RAEEN_CAP_UPSCALE != 0,
        frame_gen: bits & RAEEN_CAP_FRAME_GEN != 0,
        wants_depth: bits & RAEEN_CAP_WANTS_DEPTH != 0,
        wants_motion_vectors: bits & RAEEN_CAP_WANTS_MOTION_VECTORS != 0,
        gpu_frames: bits & RAEEN_CAP_GPU_FRAMES != 0,
    }
}

/// The inverse — used by plugin authors and by the tests.
#[must_use]
pub fn capabilities_to_bits(caps: Capabilities) -> u32 {
    let mut bits = 0;
    if caps.upscale {
        bits |= RAEEN_CAP_UPSCALE;
    }
    if caps.frame_gen {
        bits |= RAEEN_CAP_FRAME_GEN;
    }
    if caps.wants_depth {
        bits |= RAEEN_CAP_WANTS_DEPTH;
    }
    if caps.wants_motion_vectors {
        bits |= RAEEN_CAP_WANTS_MOTION_VECTORS;
    }
    if caps.gpu_frames {
        bits |= RAEEN_CAP_GPU_FRAMES;
    }
    bits
}

/// Validate one plugin-produced frame and copy it into an owned [`PluginFrame`].
///
/// Returns `None` — never a partial or unchecked read — if anything about the
/// descriptor is inconsistent. A plugin may legitimately change the resolution,
/// so the dimensions cannot be compared against the source; they are instead
/// bounded and checked for self-consistency against `pixels_len`.
///
/// # What this cannot catch
///
/// Both `pixels_len` and the dimensions are the **plugin's own claims**. This
/// function verifies they agree with each other; it cannot verify either against
/// the allocation's true size, because no portable mechanism exists to ask an
/// arbitrary foreign allocator how large a pointer's block is. A plugin that
/// under-allocates and then reports a *consistent* pair — say a 4-byte buffer
/// described as 2x2x4 — will be read for the full 16 bytes and over-read its
/// own allocation.
///
/// This residual risk is intrinsic to the boundary, not an oversight: it is the
/// same exposure every C plugin ABI carries, and it is why loading a plugin is
/// gated on the user supplying the binary (see the module-level trust boundary).
/// The checks here exist to make *inconsistent* descriptors — the far more
/// common shape, produced by ordinary plugin bugs and version skew — refusals
/// instead of over-reads.
fn copy_frame(frame: &RaeenPluginFrame) -> Option<PluginFrame> {
    if frame.pixels.is_null() {
        return None;
    }
    if frame.width == 0 || frame.height == 0 {
        return None;
    }
    if frame.width > MAX_EDGE || frame.height > MAX_EDGE {
        return None;
    }
    // `RenderedImage` is 4 bytes/px for the display formats and 8 for HDR
    // R16G16B16A16. Anything else cannot be presented.
    if frame.bytes_per_pixel != 4 && frame.bytes_per_pixel != 8 {
        return None;
    }

    let expected = (frame.width as usize)
        .checked_mul(frame.height as usize)?
        .checked_mul(frame.bytes_per_pixel as usize)?;
    if expected == 0 || expected > MAX_OUTPUT_BYTES {
        return None;
    }
    // Exact equality, not `>=`: a length that disagrees with the dimensions
    // means the two describe different buffers, and guessing which is right is
    // how a heap over-read happens.
    if frame.pixels_len != expected {
        return None;
    }

    // SAFETY: `pixels` is non-null and the plugin declares `pixels_len` bytes
    // readable there; `pixels_len` was just proven equal to the size implied by
    // the validated dimensions. The slice is copied immediately and never
    // outlives this call (the plugin frees the original in `release_output`).
    let pixels = unsafe { std::slice::from_raw_parts(frame.pixels, frame.pixels_len) }.to_vec();

    Some(PluginFrame {
        width: frame.width,
        height: frame.height,
        bytes_per_pixel: frame.bytes_per_pixel,
        pixels,
    })
}

impl PresentPlugin for DynamicPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    fn source_path(&self) -> Option<&Path> {
        Some(&self.source)
    }

    fn process(&mut self, frame: &PresentFrame<'_>, ctx: &PresentContext) -> PluginOutput {
        let depth = frame.depth.map(|p| RaeenAuxPlane {
            width: p.width,
            height: p.height,
            bytes_per_texel: p.bytes_per_texel,
            _reserved: 0,
            data: p.data.as_ptr(),
            len: p.data.len(),
        });
        let motion = frame.motion.map(|p| RaeenAuxPlane {
            width: p.width,
            height: p.height,
            bytes_per_texel: p.bytes_per_texel,
            _reserved: 0,
            data: p.data.as_ptr(),
            len: p.data.len(),
        });

        let c_ctx = RaeenPresentContext {
            output_scale: ctx.output_scale,
            hdr: u32::from(ctx.hdr),
        };

        // Raw versioned output, kept alive until the matching `release_output`
        // below. Reading is done on `Copy` snapshots; ownership of every
        // pointer inside stays with the plugin throughout.
        enum RawOut {
            V1(RaeenPluginOutput),
            V2(RaeenPluginOutputV2),
        }
        let (rc, mut raw) = match self.vtable {
            PluginVtable::V1(vtable) => {
                // SAFETY: validated at construction; the library is alive.
                let vt = unsafe { *vtable };
                let c_frame = RaeenPresentFrame {
                    width: frame.width,
                    height: frame.height,
                    bytes_per_pixel: frame.bytes_per_pixel,
                    _reserved: 0,
                    color: frame.color.as_ptr(),
                    color_len: frame.color.len(),
                    // `as_ref` borrows the local `Option`s, which outlive the call.
                    depth: depth.as_ref().map_or(std::ptr::null(), |p| p),
                    motion: motion.as_ref().map_or(std::ptr::null(), |p| p),
                    frame_index: frame.frame_index,
                };
                let mut out = RaeenPluginOutput::empty();
                // SAFETY: all pointers are live locals outliving the call;
                // `instance` came from `create` and is not destroyed.
                let rc = unsafe { (vt.process)(self.instance, &c_frame, &c_ctx, &mut out) };
                (rc, RawOut::V1(out))
            }
            PluginVtable::V2(vtable) => {
                // SAFETY: validated at construction; the library is alive.
                let vt = unsafe { *vtable };
                // CPU-kind v2 frame: v1 semantics through v2 types (the host
                // has not advertised GPU frames).
                let c_frame = RaeenPresentFrameV2 {
                    kind: RAEEN_FRAME_KIND_CPU,
                    width: frame.width,
                    height: frame.height,
                    bytes_per_pixel: frame.bytes_per_pixel,
                    color: frame.color.as_ptr(),
                    color_len: frame.color.len(),
                    color_image: RaeenVulkanImage::zeroed(),
                    depth: depth.as_ref().map_or(std::ptr::null(), |p| p),
                    motion: motion.as_ref().map_or(std::ptr::null(), |p| p),
                    frame_index: frame.frame_index,
                };
                let mut out = RaeenPluginOutputV2::empty();
                // SAFETY: as the v1 arm.
                let rc = unsafe { (vt.process)(self.instance, &c_frame, &c_ctx, &mut out) };
                (rc, RawOut::V2(out))
            }
        };

        if rc != RAEEN_OK {
            // A declined frame is normal (a temporal plugin warming up, an
            // unsupported format); present the source unchanged, no warning.
            return PluginOutput::identity(frame);
        }

        // The CPU-output view both versions share. A v2 plugin claiming a
        // GPU-produced frame while the host never advertised GPU frames is a
        // contract violation — treat as malformed (identity + warning below).
        let out = match &raw {
            RawOut::V1(out) => *out,
            RawOut::V2(out) => {
                if out.produced_kind != RAEEN_FRAME_KIND_CPU {
                    tracing::warn!(
                        plugin = %self.name,
                        source = %self.source.display(),
                        produced_kind = out.produced_kind,
                        "v2 plugin produced a GPU frame but this host never \
                         advertised RAEEN_HOST_GPU_FRAMES — presenting the \
                         source frame unchanged"
                    );
                    RaeenPluginOutput::empty()
                } else {
                    out.base
                }
            }
        };

        let primary = copy_frame(&out.primary);

        // Generated frames are reserved — the present path does not schedule
        // them yet — but validate and copy them so the ABI is exercised now and
        // a frame-gen plugin can be developed against it.
        let generated = if out.generated.is_null() || out.generated_count == 0 {
            // The ordinary case: a pure upscaler generates nothing.
            Vec::new()
        } else if out.generated_count > MAX_GENERATED {
            // Name the miss rather than silently dropping: an implausible count
            // is a plugin bug (or a struct-layout mismatch this build cannot
            // see), and quietly returning nothing would look like a plugin that
            // simply declined to generate.
            tracing::warn!(
                plugin = %self.name,
                source = %self.source.display(),
                generated_count = out.generated_count,
                limit = MAX_GENERATED,
                "present plugin reported more generated frames than the ABI \
                 permits for one source frame — dropping all of them"
            );
            Vec::new()
        } else {
            // SAFETY: non-null, and the plugin declares `generated_count`
            // contiguous `RaeenPluginFrame` there; the count is bounded above.
            let frames = unsafe { std::slice::from_raw_parts(out.generated, out.generated_count) };
            frames.iter().filter_map(copy_frame).collect()
        };

        // The plugin owns its allocations regardless of whether we accepted
        // them, so release before deciding what to return.
        // SAFETY: `process` returned success, so the raw output holds
        // plugin-owned allocations it is responsible for freeing; `instance`
        // is live, and the vtable/output versions match by construction.
        unsafe {
            match (self.vtable, &mut raw) {
                (PluginVtable::V1(vtable), RawOut::V1(out)) => {
                    ((*vtable).release_output)(self.instance, out);
                }
                (PluginVtable::V2(vtable), RawOut::V2(out)) => {
                    ((*vtable).release_output)(self.instance, out);
                }
                _ => unreachable!("vtable and output versions always match"),
            }
        }

        match primary {
            Some(primary) => PluginOutput { primary, generated },
            None => {
                // Name the miss: a plugin returning a malformed frame is a bug
                // in the plugin, and silently presenting the source would make
                // it look like the plugin simply did nothing.
                tracing::warn!(
                    plugin = %self.name,
                    source = %self.source.display(),
                    width = out.primary.width,
                    height = out.primary.height,
                    bytes_per_pixel = out.primary.bytes_per_pixel,
                    pixels_len = out.primary.pixels_len,
                    "present plugin returned a malformed primary frame — presenting the \
                     source frame unchanged (dimensions, bytes-per-pixel and buffer length \
                     must agree, and bytes-per-pixel must be 4 or 8)"
                );
                PluginOutput::identity(frame)
            }
        }
    }
}

// ─── Loading ────────────────────────────────────────────────────────────────

/// The native shared-library extension for this platform.
#[must_use]
pub const fn plugin_extension() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}

/// Load one plugin binary.
///
/// # Safety
///
/// Executes arbitrary native code from `path` inside this process — see the
/// module-level trust boundary. Only call this on a path the *user* supplied.
pub unsafe fn load_from_path(path: &Path) -> Result<DynamicPlugin, LoadError> {
    // SAFETY: delegated to the caller — this is exactly the trust boundary the
    // `unsafe` on this function marks. Loading runs the library's initializers.
    let library = unsafe { libloading::Library::new(path) }.map_err(|source| LoadError::Open {
        path: path.to_path_buf(),
        source,
    })?;

    // Version negotiation: a `raeen_plugin_v2` export is authoritative when
    // present (a binary may export both for older Raeens); v1 remains fully
    // supported for plugins that never need GPU frames.
    //
    // Scoped so each `Symbol`'s borrow of `library` ends before `library`
    // moves into the plugin below. (`Symbol` has no `Drop`, so the borrow is
    // purely a lifetime; a block is the way to end it, not a `drop` call.)
    let vtable_v2 = {
        // SAFETY: looked up by its documented name, used only in this scope
        // while `library` is alive.
        let entry: Result<libloading::Symbol<'_, RaeenPluginEntryV2Fn>, _> =
            unsafe { library.get(RAEEN_PLUGIN_ENTRY_V2) };
        match entry {
            // SAFETY: `entry` resolved to the documented v2 entry point.
            Ok(entry) => Some(unsafe { entry() }),
            Err(_) => None,
        }
    };
    if let Some(vtable) = vtable_v2 {
        // SAFETY: `vtable` is whatever the plugin returned (null-checked
        // inside); `library` moves in so the image outlives every use.
        return unsafe { DynamicPlugin::from_vtable_v2(vtable, Some(library), path.to_path_buf()) };
    }

    let vtable = {
        // SAFETY: the symbol is looked up by its documented name and used only
        // within this scope, while `library` is alive.
        let entry: libloading::Symbol<'_, RaeenPluginEntryFn> = unsafe {
            library.get(RAEEN_PLUGIN_ENTRY)
        }
        .map_err(|source| LoadError::MissingEntry {
            path: path.to_path_buf(),
            source,
        })?;
        // SAFETY: `entry` resolved to the plugin's exported entry point;
        // calling it is the documented contract.
        unsafe { entry() }
    };

    // SAFETY: `vtable` is whatever the plugin returned (null-checked inside),
    // and `library` is moved in so the image outlives every use of it.
    unsafe { DynamicPlugin::from_vtable(vtable, Some(library), path.to_path_buf()) }
}

/// Scan a directory for plugin binaries and attempt to load each.
///
/// Non-recursive and extension-filtered. Returns one entry per candidate — a
/// refusal is reported, never silently skipped, so a user who drops in a wrong
/// or stale plugin learns why. A missing directory yields an empty list (having
/// no `plugins/` directory is the normal case, not an error).
///
/// # Safety
///
/// Loads and executes every matching binary in `dir` — see [`load_from_path`].
pub unsafe fn scan_dir(dir: &Path) -> Vec<Result<DynamicPlugin, LoadError>> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let want = plugin_extension();

    let mut candidates: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case(want))
        })
        .collect();
    // Deterministic order: the load sequence must not depend on filesystem
    // enumeration order, or two machines register plugins differently.
    candidates.sort();

    candidates
        .iter()
        // SAFETY: delegated to this function's caller.
        .map(|p| unsafe { load_from_path(p) })
        .collect()
}

/// Scan `dir`, register every plugin that loads, and log every refusal.
///
/// Returns the names registered, in registration order. This is the call the
/// Shell makes at startup; nothing is *activated* — selection stays a user
/// choice via [`super::select`].
///
/// # Safety
///
/// Loads and executes every matching binary in `dir` — see [`load_from_path`].
pub unsafe fn load_and_register_dir(dir: &Path) -> Vec<String> {
    // SAFETY: delegated to this function's caller.
    let results = unsafe { scan_dir(dir) };
    let mut registered = Vec::new();
    let mut failures = Vec::new();

    for result in results {
        match result {
            Ok(plugin) => {
                let name = plugin.name().to_string();
                tracing::info!(
                    plugin = %name,
                    source = %plugin.source().display(),
                    capabilities = ?plugin.capabilities(),
                    "registered out-of-tree present plugin"
                );
                super::register(Box::new(plugin));
                registered.push(name);
            }
            Err(err) => {
                tracing::warn!(error = %err, "refused a present plugin candidate");
                failures.push(err.to_string());
            }
        }
    }
    // Replace, don't append: a rescan reports the directory as it is now, and
    // a clean rescan clears refusals from a plugin the user since removed.
    super::set_load_failures(failures);
    registered
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ─── A well-behaved fake plugin, built from ordinary `extern "C"` fns ────
    //
    // This exercises the whole adapter without needing a compiled .dll: the
    // vtable is the only thing `from_vtable` ever sees, so a vtable of local
    // functions is indistinguishable from one returned by `dlopen`.

    /// Live instance count, so teardown can be asserted.
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    /// How many buffers the fake plugin allocated but has not yet freed.
    static UNFREED: AtomicUsize = AtomicUsize::new(0);

    struct FakeState {
        /// Value written into the first byte of every output pixel.
        tint: u8,
    }

    unsafe extern "C" fn fake_create() -> *mut c_void {
        LIVE.fetch_add(1, Ordering::SeqCst);
        Box::into_raw(Box::new(FakeState { tint: 0xAB })).cast()
    }

    unsafe extern "C" fn fake_destroy(instance: *mut c_void) {
        if instance.is_null() {
            return;
        }
        // SAFETY: `instance` came from `fake_create`'s `Box::into_raw`.
        drop(unsafe { Box::from_raw(instance.cast::<FakeState>()) });
        LIVE.fetch_sub(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn fake_name(_i: *mut c_void, buf: *mut u8, cap: usize) -> usize {
        let name = b"fake-tint";
        if cap < name.len() {
            return usize::MAX; // signal "too long"
        }
        // SAFETY: `buf` has `cap >= name.len()` writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(name.as_ptr(), buf, name.len()) };
        name.len()
    }

    unsafe extern "C" fn fake_capabilities(_i: *mut c_void) -> u32 {
        RAEEN_CAP_UPSCALE | RAEEN_CAP_WANTS_MOTION_VECTORS
    }

    /// Copies the source and tints byte 0 of each pixel — a visible, checkable
    /// transform that proves `process` really ran across the boundary.
    unsafe extern "C" fn fake_process(
        instance: *mut c_void,
        frame: *const RaeenPresentFrame,
        _ctx: *const RaeenPresentContext,
        out: *mut RaeenPluginOutput,
    ) -> i32 {
        // SAFETY: Raeen passes live pointers for the duration of the call.
        let (state, frame, out) = unsafe { (&*instance.cast::<FakeState>(), &*frame, &mut *out) };
        // SAFETY: `color`/`color_len` describe a live buffer supplied by Raeen.
        let src = unsafe { std::slice::from_raw_parts(frame.color, frame.color_len) };
        let mut pixels = src.to_vec();
        for px in pixels.chunks_mut(frame.bytes_per_pixel as usize) {
            if let Some(first) = px.first_mut() {
                *first = state.tint;
            }
        }
        let len = pixels.len();
        let ptr = Box::into_raw(pixels.into_boxed_slice()).cast::<u8>();
        UNFREED.fetch_add(1, Ordering::SeqCst);

        out.primary = RaeenPluginFrame {
            width: frame.width,
            height: frame.height,
            bytes_per_pixel: frame.bytes_per_pixel,
            _reserved: 0,
            pixels: ptr,
            pixels_len: len,
        };
        out.generated = std::ptr::null();
        out.generated_count = 0;
        RAEEN_OK
    }

    unsafe extern "C" fn fake_release(_i: *mut c_void, out: *mut RaeenPluginOutput) {
        // SAFETY: Raeen passes back exactly the output `fake_process` filled.
        let out = unsafe { &mut *out };
        if out.primary.pixels.is_null() {
            return;
        }
        // SAFETY: reconstitutes the boxed slice `fake_process` leaked.
        drop(unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                out.primary.pixels.cast_mut(),
                out.primary.pixels_len,
            ))
        });
        out.primary = RaeenPluginFrame::empty();
        UNFREED.fetch_sub(1, Ordering::SeqCst);
    }

    /// Returns a descriptor whose length disagrees with its dimensions — the
    /// exact shape that would cause a heap over-read if trusted.
    unsafe extern "C" fn lying_process(
        _i: *mut c_void,
        frame: *const RaeenPresentFrame,
        _c: *const RaeenPresentContext,
        out: *mut RaeenPluginOutput,
    ) -> i32 {
        // SAFETY: live for the call.
        let (frame, out) = unsafe { (&*frame, &mut *out) };
        // Allocate ONE pixel and report that honestly, while declaring
        // full-frame dimensions. This is the DETECTABLE inconsistency: the
        // length and the dimensions disagree, so Raeen refuses the frame
        // without ever indexing past the allocation.
        //
        // The *consistent* lie — reporting `pixels_len == w*h*bpp` for a buffer
        // that small — is deliberately NOT modelled here, because it is
        // undetectable by construction; see `copy_frame`'s docs.
        let pixels = vec![0u8; frame.bytes_per_pixel as usize];
        let ptr = Box::into_raw(pixels.into_boxed_slice()).cast::<u8>();
        UNFREED.fetch_add(1, Ordering::SeqCst);
        out.primary = RaeenPluginFrame {
            width: frame.width,
            height: frame.height,
            bytes_per_pixel: frame.bytes_per_pixel,
            _reserved: 0,
            pixels: ptr,
            pixels_len: frame.bytes_per_pixel as usize,
        };
        RAEEN_OK
    }

    unsafe extern "C" fn lying_release(_i: *mut c_void, out: *mut RaeenPluginOutput) {
        // SAFETY: frees the single pixel actually allocated above.
        let out = unsafe { &mut *out };
        if out.primary.pixels.is_null() {
            return;
        }
        // SAFETY: only `bytes_per_pixel` bytes were really allocated.
        drop(unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                out.primary.pixels.cast_mut(),
                out.primary.bytes_per_pixel as usize,
            ))
        });
        out.primary = RaeenPluginFrame::empty();
        UNFREED.fetch_sub(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn declining_process(
        _i: *mut c_void,
        _f: *const RaeenPresentFrame,
        _c: *const RaeenPresentContext,
        _o: *mut RaeenPluginOutput,
    ) -> i32 {
        -1
    }

    unsafe extern "C" fn null_create() -> *mut c_void {
        std::ptr::null_mut()
    }

    unsafe extern "C" fn empty_name(_i: *mut c_void, _b: *mut u8, _c: usize) -> usize {
        0
    }

    unsafe extern "C" fn noop_release(_i: *mut c_void, _o: *mut RaeenPluginOutput) {}

    fn vtable() -> RaeenPluginV1 {
        RaeenPluginV1 {
            abi_version: RAEEN_PLUGIN_ABI_VERSION,
            _reserved: 0,
            create: fake_create,
            destroy: fake_destroy,
            name: fake_name,
            capabilities: fake_capabilities,
            process: fake_process,
            release_output: fake_release,
        }
    }

    fn load(vt: &RaeenPluginV1) -> Result<DynamicPlugin, LoadError> {
        // SAFETY: `vt` is a live local that outlives the returned plugin in
        // every caller below, and its functions honour the ABI contract.
        unsafe { DynamicPlugin::from_vtable(vt, None, PathBuf::from("test://vtable")) }
    }

    fn source_frame(buf: &[u8]) -> PresentFrame<'_> {
        PresentFrame {
            width: 2,
            height: 2,
            bytes_per_pixel: 4,
            color: buf,
            depth: None,
            motion: None,
            frame_index: 7,
        }
    }

    const CTX: PresentContext = PresentContext {
        output_scale: 1.0,
        hdr: false,
    };

    // ─── ABI v2 fake plugin ─────────────────────────────────────────────────
    //
    // A v2 tinter built from ordinary `extern "C"` fns, exactly like the v1
    // fake above. `create` verifies the host context contract (non-null,
    // sane size, no GPU frames advertised today).

    unsafe extern "C" fn fake_create_v2(host: *const RaeenHostContextV2) -> *mut c_void {
        if host.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: `host` is valid for the duration of this call.
        let host = unsafe { &*host };
        if (host.struct_size as usize) < std::mem::size_of::<RaeenHostContextV2>()
            || host.host_flags & RAEEN_HOST_GPU_FRAMES != 0
        {
            return std::ptr::null_mut();
        }
        LIVE.fetch_add(1, Ordering::SeqCst);
        Box::into_raw(Box::new(FakeState { tint: 0xCD })).cast()
    }

    unsafe extern "C" fn fake_name_v2(_i: *mut c_void, buf: *mut u8, cap: usize) -> usize {
        let name = b"fake-tint-v2";
        if cap < name.len() {
            return usize::MAX;
        }
        // SAFETY: `buf` has `cap >= name.len()` writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(name.as_ptr(), buf, name.len()) };
        name.len()
    }

    unsafe extern "C" fn fake_capabilities_v2(_i: *mut c_void) -> u32 {
        RAEEN_CAP_UPSCALE | RAEEN_CAP_GPU_FRAMES
    }

    unsafe extern "C" fn fake_process_v2(
        instance: *mut c_void,
        frame: *const RaeenPresentFrameV2,
        _ctx: *const RaeenPresentContext,
        out: *mut RaeenPluginOutputV2,
    ) -> i32 {
        // SAFETY: Raeen passes live pointers for the duration of the call.
        let (state, frame, out) = unsafe { (&*instance.cast::<FakeState>(), &*frame, &mut *out) };
        // Today's host must deliver CPU frames with a zeroed image.
        if frame.kind != RAEEN_FRAME_KIND_CPU || frame.color.is_null() {
            return -1;
        }
        assert_eq!(frame.color_image.image, 0, "CPU frames carry no image");
        // SAFETY: `color`/`color_len` describe a live buffer supplied by Raeen.
        let src = unsafe { std::slice::from_raw_parts(frame.color, frame.color_len) };
        let mut pixels = src.to_vec();
        for px in pixels.chunks_mut(frame.bytes_per_pixel as usize) {
            if let Some(first) = px.first_mut() {
                *first = state.tint;
            }
        }
        let len = pixels.len();
        let ptr = Box::into_raw(pixels.into_boxed_slice()).cast::<u8>();
        UNFREED.fetch_add(1, Ordering::SeqCst);
        out.base.primary = RaeenPluginFrame {
            width: frame.width,
            height: frame.height,
            bytes_per_pixel: frame.bytes_per_pixel,
            _reserved: 0,
            pixels: ptr,
            pixels_len: len,
        };
        out.base.generated = std::ptr::null();
        out.base.generated_count = 0;
        out.produced_kind = RAEEN_FRAME_KIND_CPU;
        RAEEN_OK
    }

    unsafe extern "C" fn fake_release_v2(_i: *mut c_void, out: *mut RaeenPluginOutputV2) {
        // SAFETY: Raeen passes back exactly the output `fake_process_v2` filled.
        let out = unsafe { &mut *out };
        if out.base.primary.pixels.is_null() {
            return;
        }
        // SAFETY: reconstitutes the boxed slice `fake_process_v2` leaked.
        drop(unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                out.base.primary.pixels.cast_mut(),
                out.base.primary.pixels_len,
            ))
        });
        out.base.primary.pixels = std::ptr::null();
        UNFREED.fetch_sub(1, Ordering::SeqCst);
    }

    fn vtable_v2() -> RaeenPluginV2 {
        RaeenPluginV2 {
            abi_version: RAEEN_PLUGIN_ABI_V2,
            _reserved: 0,
            create: fake_create_v2,
            destroy: fake_destroy,
            name: fake_name_v2,
            capabilities: fake_capabilities_v2,
            process: fake_process_v2,
            release_output: fake_release_v2,
        }
    }

    fn load_v2(vt: &RaeenPluginV2) -> Result<DynamicPlugin, LoadError> {
        // SAFETY: `vt` is a live local outliving the plugin in every caller,
        // and its functions honour the v2 ABI contract.
        unsafe { DynamicPlugin::from_vtable_v2(vt, None, PathBuf::from("test://vtable-v2")) }
    }

    #[test]
    fn a_conforming_v2_plugin_round_trips_with_cpu_kind_frames() {
        let vt = vtable_v2();
        let mut plugin = load_v2(&vt).expect("a conforming v2 vtable must load");

        assert_eq!(plugin.name(), "fake-tint-v2");
        assert_eq!(
            plugin.capabilities(),
            Capabilities {
                upscale: true,
                gpu_frames: true,
                ..Default::default()
            },
            "the v2 GPU-frames capability bit must survive the round trip"
        );

        let buf = vec![0x22u8; 2 * 2 * 4];
        let before = UNFREED.load(Ordering::SeqCst);
        let out = plugin.process(&source_frame(&buf), &CTX);
        assert_eq!(out.primary.pixels.len(), 16);
        for px in out.primary.pixels.chunks(4) {
            assert_eq!(px[0], 0xCD, "the v2 transform must be visible");
            assert_eq!(px[1], 0x22, "untouched bytes must survive the copy");
        }
        assert_eq!(
            UNFREED.load(Ordering::SeqCst),
            before,
            "v2 output must be released back to the plugin"
        );
    }

    #[test]
    fn a_v2_abi_version_mismatch_is_refused_before_create() {
        let mut vt = vtable_v2();
        vt.abi_version = 7;
        let live = LIVE.load(Ordering::SeqCst);
        let err = load_v2(&vt).expect_err("wrong v2 version must refuse");
        assert!(matches!(err, LoadError::AbiMismatch { found: 7, .. }));
        assert_eq!(LIVE.load(Ordering::SeqCst), live, "no instance may leak");
    }

    #[test]
    fn a_conforming_plugin_round_trips_through_the_c_abi() {
        let vt = vtable();
        let mut plugin = load(&vt).expect("a conforming vtable must load");

        assert_eq!(plugin.name(), "fake-tint");
        assert_eq!(
            plugin.capabilities(),
            Capabilities {
                upscale: true,
                wants_motion_vectors: true,
                ..Default::default()
            },
            "capability bits must survive the round trip"
        );

        let buf = vec![0x11u8; 2 * 2 * 4];
        let out = plugin.process(&source_frame(&buf), &CTX);

        assert_eq!(out.primary.width, 2);
        assert_eq!(out.primary.height, 2);
        assert_eq!(out.primary.pixels.len(), 16);
        // Byte 0 of each pixel tinted by the plugin; the rest untouched.
        for px in out.primary.pixels.chunks(4) {
            assert_eq!(px[0], 0xAB, "the plugin's transform must be visible");
            assert_eq!(px[1], 0x11, "untouched bytes must survive the copy");
        }
        assert!(out.generated.is_empty());
    }

    #[test]
    fn output_is_released_back_to_the_plugin_after_every_frame() {
        let vt = vtable();
        let mut plugin = load(&vt).unwrap();
        let buf = vec![0u8; 2 * 2 * 4];

        UNFREED.store(0, Ordering::SeqCst);
        for _ in 0..8 {
            let _ = plugin.process(&source_frame(&buf), &CTX);
        }
        assert_eq!(
            UNFREED.load(Ordering::SeqCst),
            0,
            "every plugin allocation must be handed back via release_output — \
             a leak here is a leak per presented frame"
        );
    }

    #[test]
    fn a_length_that_disagrees_with_the_dimensions_is_refused_not_read() {
        // The whole point of the validation layer: this descriptor claims a full
        // frame but only one pixel exists. Trusting it is a heap over-read.
        let mut vt = vtable();
        vt.process = lying_process;
        vt.release_output = lying_release;
        let mut plugin = load(&vt).unwrap();

        let buf = vec![0x22u8; 2 * 2 * 4];
        UNFREED.store(0, Ordering::SeqCst);
        let out = plugin.process(&source_frame(&buf), &CTX);

        assert_eq!(
            out.primary.pixels, buf,
            "a malformed output must degrade to the SOURCE frame, unchanged"
        );
        assert_eq!(
            UNFREED.load(Ordering::SeqCst),
            0,
            "release_output must still run for a rejected frame — the plugin \
             allocated it and only the plugin can free it"
        );
    }

    #[test]
    fn a_declining_plugin_presents_the_source_frame() {
        let mut vt = vtable();
        vt.process = declining_process;
        vt.release_output = noop_release;
        let mut plugin = load(&vt).unwrap();

        let buf = vec![0x33u8; 2 * 2 * 4];
        let out = plugin.process(&source_frame(&buf), &CTX);
        assert_eq!(out.primary.pixels, buf);
        assert!(out.generated.is_empty());
    }

    #[test]
    fn an_abi_mismatch_is_refused_before_any_other_call() {
        let mut vt = vtable();
        vt.abi_version = RAEEN_PLUGIN_ABI_VERSION + 1;
        // `create` must never run: on a version mismatch the struct we read may
        // not be the struct the plugin wrote.
        vt.create = null_create;

        let before = LIVE.load(Ordering::SeqCst);
        let err = load(&vt).expect_err("a mismatched ABI must be refused");
        assert!(matches!(err, LoadError::AbiMismatch { found, .. }
            if found == RAEEN_PLUGIN_ABI_VERSION + 1));
        assert_eq!(
            LIVE.load(Ordering::SeqCst),
            before,
            "no instance may be created for a mismatched ABI"
        );
    }

    #[test]
    fn a_null_vtable_is_refused() {
        // SAFETY: passing null is exactly the case under test; `from_vtable`
        // null-checks before any dereference.
        let err = unsafe {
            DynamicPlugin::from_vtable(std::ptr::null(), None, PathBuf::from("test://null"))
        }
        .expect_err("a null vtable must be refused");
        assert!(matches!(err, LoadError::NullVtable { .. }));
    }

    #[test]
    fn a_failing_create_is_refused() {
        let mut vt = vtable();
        vt.create = null_create;
        let err = load(&vt).expect_err("a null instance must be refused");
        assert!(matches!(err, LoadError::CreateFailed { .. }));
    }

    #[test]
    fn an_unusable_name_is_refused_and_the_instance_destroyed() {
        let mut vt = vtable();
        vt.name = empty_name;

        let before = LIVE.load(Ordering::SeqCst);
        let err = load(&vt).expect_err("an empty name must be refused");
        assert!(matches!(err, LoadError::BadName { .. }));
        assert_eq!(
            LIVE.load(Ordering::SeqCst),
            before,
            "the instance created before the name check must be destroyed on \
             the refusal path, not leaked"
        );
    }

    #[test]
    fn dropping_a_plugin_destroys_its_instance() {
        let vt = vtable();
        let before = LIVE.load(Ordering::SeqCst);
        {
            let plugin = load(&vt).unwrap();
            assert_eq!(LIVE.load(Ordering::SeqCst), before + 1);
            drop(plugin);
        }
        assert_eq!(
            LIVE.load(Ordering::SeqCst),
            before,
            "Drop must call the plugin's destroy"
        );
    }

    #[test]
    fn capability_bits_round_trip_both_ways() {
        for caps in [
            Capabilities::default(),
            Capabilities {
                upscale: true,
                ..Default::default()
            },
            Capabilities {
                upscale: true,
                frame_gen: true,
                wants_depth: true,
                wants_motion_vectors: true,
                gpu_frames: true,
            },
        ] {
            assert_eq!(capabilities_from_bits(capabilities_to_bits(caps)), caps);
        }
        // Unknown future bits are ignored rather than refused.
        assert_eq!(
            capabilities_from_bits(RAEEN_CAP_UPSCALE | 0x8000_0000),
            Capabilities {
                upscale: true,
                ..Default::default()
            }
        );
    }

    #[test]
    fn copy_frame_rejects_every_malformed_descriptor() {
        let pixels = [0u8; 64];
        let ok = RaeenPluginFrame {
            width: 4,
            height: 4,
            bytes_per_pixel: 4,
            _reserved: 0,
            pixels: pixels.as_ptr(),
            pixels_len: 64,
        };
        assert!(copy_frame(&ok).is_some(), "the valid case must be accepted");

        let cases: [(&str, RaeenPluginFrame); 6] = [
            (
                "null pixels",
                RaeenPluginFrame {
                    pixels: std::ptr::null(),
                    ..ok
                },
            ),
            ("zero width", RaeenPluginFrame { width: 0, ..ok }),
            (
                "edge over the limit",
                RaeenPluginFrame {
                    width: MAX_EDGE + 1,
                    ..ok
                },
            ),
            (
                "unpresentable bytes-per-pixel",
                RaeenPluginFrame {
                    bytes_per_pixel: 3,
                    ..ok
                },
            ),
            (
                "length shorter than dimensions",
                RaeenPluginFrame {
                    pixels_len: 32,
                    ..ok
                },
            ),
            (
                "length longer than dimensions",
                RaeenPluginFrame {
                    pixels_len: 128,
                    ..ok
                },
            ),
        ];
        for (why, frame) in cases {
            assert!(copy_frame(&frame).is_none(), "must reject: {why}");
        }
    }

    #[test]
    fn scanning_a_missing_directory_is_empty_not_an_error() {
        // Having no `plugins/` directory is the normal case.
        // SAFETY: the directory does not exist, so nothing is loaded.
        let found = unsafe { scan_dir(Path::new("definitely/not/a/real/dir")) };
        assert!(found.is_empty());
    }

    #[test]
    fn scanning_ignores_files_that_are_not_plugin_binaries() {
        let dir = std::env::temp_dir().join("raeen-plugin-scan-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), b"not a plugin").unwrap();
        std::fs::write(dir.join("README.md"), b"# nope").unwrap();

        // SAFETY: no file in the directory has the plugin extension, so nothing
        // is loaded or executed.
        let found = unsafe { scan_dir(&dir) };
        assert!(
            found.is_empty(),
            "only files with the platform plugin extension are candidates"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_garbage_file_with_the_plugin_extension_is_refused_by_name() {
        // A user dropping a wrong/corrupt file in `plugins/` must get a named
        // refusal, not a silent skip.
        let dir = std::env::temp_dir().join("raeen-plugin-garbage-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let bogus = dir.join(format!("bogus.{}", plugin_extension()));
        std::fs::write(&bogus, b"this is not a shared library").unwrap();

        // SAFETY: the file is not a loadable image, so `Library::new` fails
        // before any foreign code can run.
        let found = unsafe { scan_dir(&dir) };
        assert_eq!(
            found.len(),
            1,
            "the candidate must be reported, not skipped"
        );
        let err = found.into_iter().next().unwrap().unwrap_err();
        assert!(
            matches!(err, LoadError::Open { .. }),
            "expected an open failure naming the file, got {err:?}"
        );
        assert!(
            err.to_string().contains("bogus"),
            "the refusal must name the file: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
