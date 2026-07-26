//! A complete, working Raeen present plugin — nearest-neighbour upscaler.
//!
//! This is the reference implementation of the C ABI documented in
//! `plugins/README.md` and `crates/raeen-gpu/src/present_plugin/cabi.rs`. It is
//! deliberately dependency-free and single-file so it compiles with a bare
//! `rustc` invocation, and so the integration test
//! `crates/raeen-gpu/tests/present_plugin_dylib.rs` can compile **this exact
//! file** and load it — the shipped example is therefore the verified one.
//!
//! # Build
//!
//! ```text
//! rustc --edition 2024 --crate-type cdylib --crate-name raeen_example_plugin \
//!       -O --out-dir plugins docs/examples/present-plugin-example.rs
//! ```
//!
//! That produces `plugins/raeen_example_plugin.dll` (Windows) or
//! `plugins/libraeen_example_plugin.so` (Linux). Restart Raeen; the plugin
//! appears in Settings ▸ Video as `example-nearest`.
//!
//! # License
//!
//! Original work, part of Raeen (GPL-2.0-only). Nothing here is derived from
//! any vendor SDK — this is a general upscaler ABI, not a socket for one
//! product.

#![allow(clippy::missing_safety_doc)]

use std::ffi::c_void;

// ─── ABI mirror — must match `RaeenPluginV1` exactly ────────────────────────

const RAEEN_PLUGIN_ABI_VERSION: u32 = 1;
const RAEEN_CAP_UPSCALE: u32 = 1 << 0;
const RAEEN_OK: i32 = 0;
/// Any non-`RAEEN_OK` return means "declined"; Raeen presents the source frame.
const RAEEN_DECLINED: i32 = -1;

#[repr(C)]
pub struct RaeenAuxPlane {
    pub width: u32,
    pub height: u32,
    pub bytes_per_texel: u32,
    pub _reserved: u32,
    pub data: *const u8,
    pub len: usize,
}

#[repr(C)]
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

#[repr(C)]
pub struct RaeenPresentContext {
    pub output_scale: f32,
    pub hdr: u32,
}

#[repr(C)]
pub struct RaeenPluginFrame {
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub _reserved: u32,
    pub pixels: *const u8,
    pub pixels_len: usize,
}

#[repr(C)]
pub struct RaeenPluginOutput {
    pub primary: RaeenPluginFrame,
    pub generated: *const RaeenPluginFrame,
    pub generated_count: usize,
}

#[repr(C)]
pub struct RaeenPluginV1 {
    pub abi_version: u32,
    pub _reserved: u32,
    pub create: unsafe extern "C" fn() -> *mut c_void,
    pub destroy: unsafe extern "C" fn(*mut c_void),
    pub name: unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> usize,
    pub capabilities: unsafe extern "C" fn(*mut c_void) -> u32,
    pub process: unsafe extern "C" fn(
        *mut c_void,
        *const RaeenPresentFrame,
        *const RaeenPresentContext,
        *mut RaeenPluginOutput,
    ) -> i32,
    pub release_output: unsafe extern "C" fn(*mut c_void, *mut RaeenPluginOutput),
}

// ─── Plugin state ───────────────────────────────────────────────────────────

/// Per-instance state. A real temporal upscaler would keep prior frames here;
/// this one only counts, to show the instance is genuinely threaded through
/// every call rather than being incidental.
struct State {
    frames_processed: u64,
}

const PLUGIN_NAME: &[u8] = b"example-nearest";

// ─── Entry points ───────────────────────────────────────────────────────────

unsafe extern "C" fn create() -> *mut c_void {
    Box::into_raw(Box::new(State {
        frames_processed: 0,
    }))
    .cast()
}

unsafe extern "C" fn destroy(instance: *mut c_void) {
    if instance.is_null() {
        return;
    }
    // SAFETY: `instance` is what `create` returned via `Box::into_raw`, and
    // Raeen calls `destroy` exactly once per instance.
    drop(unsafe { Box::from_raw(instance.cast::<State>()) });
}

/// Write the name into Raeen's buffer. Returning more than `cap` signals
/// "too long" and refuses the plugin, so check first.
unsafe extern "C" fn name(_instance: *mut c_void, buf: *mut u8, cap: usize) -> usize {
    if buf.is_null() || cap < PLUGIN_NAME.len() {
        return usize::MAX;
    }
    // SAFETY: `buf` is non-null with at least `PLUGIN_NAME.len()` writable
    // bytes, just checked against `cap`.
    unsafe { std::ptr::copy_nonoverlapping(PLUGIN_NAME.as_ptr(), buf, PLUGIN_NAME.len()) };
    PLUGIN_NAME.len()
}

unsafe extern "C" fn capabilities(_instance: *mut c_void) -> u32 {
    RAEEN_CAP_UPSCALE
}

unsafe extern "C" fn process(
    instance: *mut c_void,
    frame: *const RaeenPresentFrame,
    ctx: *const RaeenPresentContext,
    out: *mut RaeenPluginOutput,
) -> i32 {
    if instance.is_null() || frame.is_null() || ctx.is_null() || out.is_null() {
        return RAEEN_DECLINED;
    }
    // SAFETY: all four pointers are non-null (just checked) and Raeen keeps
    // them valid for the duration of this call.
    let (state, frame, ctx, out) =
        unsafe { (&mut *instance.cast::<State>(), &*frame, &*ctx, &mut *out) };

    let bpp = frame.bytes_per_pixel as usize;
    // Raeen only presents 4-byte display formats and 8-byte HDR.
    if bpp != 4 && bpp != 8 {
        return RAEEN_DECLINED;
    }

    let src_texels = (frame.width as usize).saturating_mul(frame.height as usize);
    let needed = src_texels.saturating_mul(bpp);
    if frame.color.is_null() || needed == 0 || frame.color_len < needed {
        return RAEEN_DECLINED;
    }

    let scale = ctx.output_scale.clamp(1.0, 8.0);
    let dst_w = (((frame.width as f32) * scale).round() as u32).max(1);
    let dst_h = (((frame.height as f32) * scale).round() as u32).max(1);

    // Nothing to do at native scale — decline so Raeen takes its zero-copy
    // path instead of us allocating a byte-identical duplicate.
    if dst_w == frame.width && dst_h == frame.height {
        return RAEEN_DECLINED;
    }

    // SAFETY: `color` is non-null with at least `needed` readable bytes.
    let src = unsafe { std::slice::from_raw_parts(frame.color, needed) };

    let mut pixels = vec![0u8; (dst_w as usize) * (dst_h as usize) * bpp];
    for y in 0..dst_h {
        let sy = (((y as u64) * (frame.height as u64)) / (dst_h as u64))
            .min((frame.height - 1) as u64) as u32;
        for x in 0..dst_w {
            let sx = (((x as u64) * (frame.width as u64)) / (dst_w as u64))
                .min((frame.width - 1) as u64) as u32;
            let s = ((sy * frame.width + sx) as usize) * bpp;
            let d = ((y * dst_w + x) as usize) * bpp;
            pixels[d..d + bpp].copy_from_slice(&src[s..s + bpp]);
        }
    }

    let pixels_len = pixels.len();
    // Hand ownership to Raeen's *caller* contract: we keep owning the memory,
    // and free it in `release_output`.
    let ptr = Box::into_raw(pixels.into_boxed_slice()).cast::<u8>();

    out.primary = RaeenPluginFrame {
        width: dst_w,
        height: dst_h,
        bytes_per_pixel: frame.bytes_per_pixel,
        _reserved: 0,
        pixels: ptr,
        pixels_len,
    };
    // Pure upscaler: no generated (interpolated) frames.
    out.generated = std::ptr::null();
    out.generated_count = 0;

    state.frames_processed = state.frames_processed.wrapping_add(1);
    RAEEN_OK
}

/// Free what `process` allocated. Raeen calls this after copying the pixels —
/// always, including when it rejected the output as malformed.
unsafe extern "C" fn release_output(_instance: *mut c_void, out: *mut RaeenPluginOutput) {
    if out.is_null() {
        return;
    }
    // SAFETY: Raeen hands back exactly the output `process` filled in.
    let out = unsafe { &mut *out };
    if out.primary.pixels.is_null() {
        return;
    }
    // SAFETY: reconstitutes the boxed slice `process` leaked, with the same
    // length it recorded.
    drop(unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            out.primary.pixels.cast_mut(),
            out.primary.pixels_len,
        ))
    });
    out.primary.pixels = std::ptr::null();
    out.primary.pixels_len = 0;
}

static VTABLE: RaeenPluginV1 = RaeenPluginV1 {
    abi_version: RAEEN_PLUGIN_ABI_VERSION,
    _reserved: 0,
    create,
    destroy,
    name,
    capabilities,
    process,
    release_output,
};

/// The one symbol Raeen looks up. Must keep this exact name and C linkage.
#[unsafe(no_mangle)]
pub extern "C" fn raeen_plugin_v1() -> *const RaeenPluginV1 {
    &raw const VTABLE
}
