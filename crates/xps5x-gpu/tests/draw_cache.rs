//! Stage A draw-resource caching acceptance.
//!
//! Two properties are pinned here:
//!
//! 1. **Cache hits**: a second draw with identical state must reuse the first
//!    draw's `VkPipeline` and shader modules (counted via
//!    [`VulkanDevice::draw_cache_stats`]) instead of recreating them.
//! 2. **Byte-identical semantics**: the persistent-render-target fast path
//!    (`DrawState::target_base` set, seed upload skipped, attachment LOADs
//!    from the GPU-resident image) must produce exactly the bytes of the old
//!    per-draw path (fresh image seeded by uploading `initial`).
//!
//! Machines without Vulkan 1.3 skip (unless `XPS5X_REQUIRE_VULKAN=1`).

use xps5x_gpu::backend::GpuBackend;
use xps5x_gpu::vulkan::offscreen::{DrawState, render_draw};
use xps5x_gpu::vulkan::{
    VulkanBackend,
    shaders::{triangle_fragment_spirv, triangle_vertex_spirv},
    validation_error_count,
};

const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

fn backend_or_skip() -> Option<VulkanBackend> {
    // XPS5X_TEST_LOG=1 surfaces tracing output (e.g. the XPS5X_TIME_DRAW
    // per-draw phase timing this crate's stage A work is measured with).
    if std::env::var_os("XPS5X_TEST_LOG").is_some() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    }
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => Some(backend),
        Err(e) => {
            assert!(
                std::env::var_os("XPS5X_REQUIRE_VULKAN").is_none(),
                "XPS5X_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("draw_cache: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

/// Two draws with identical state: the second must hit the pipeline cache and
/// the shader-module cache, and must render the same pixels.
#[test]
fn identical_draws_hit_the_pipeline_and_module_caches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();
    let state = DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        ..DrawState::new(64, 64, &vs, &ps)
    };

    let first = render_draw(dev, &state)
        .expect("first draw renders")
        .color
        .expect("colour draw produces a colour image");
    let after_first = dev.draw_cache_stats();
    assert_eq!(
        after_first.pipeline_misses, 1,
        "first draw builds the pipeline"
    );
    assert_eq!(after_first.pipeline_hits, 0);
    assert_eq!(
        after_first.shader_module_misses, 2,
        "first draw creates the VS and FS modules"
    );

    let second = render_draw(dev, &state)
        .expect("second draw renders")
        .color
        .expect("colour draw produces a colour image");
    let after_second = dev.draw_cache_stats();
    assert_eq!(
        after_second.pipeline_hits, 1,
        "identical state must reuse the cached pipeline"
    );
    assert_eq!(
        after_second.pipeline_misses, 1,
        "no second pipeline may be built"
    );
    assert_eq!(
        after_second.shader_module_hits, 2,
        "identical SPIR-V must reuse the cached modules"
    );
    assert_eq!(after_second.shader_module_misses, 2);

    assert_eq!(
        second.pixels, first.pixels,
        "cached-resource draw must render identical bytes"
    );
    assert_eq!(validation_error_count(), 0, "no Vulkan validation errors");
}

/// The persistent-target fast path against the per-draw reference path, on the
/// exact compose pattern the title path uses: draw 1 clears + draws, draw 2
/// LOADs draw 1's output as `initial` and draws with a viewport that puts the
/// triangle somewhere else. The persistent path must (a) skip the seed upload
/// on draw 2 and (b) produce byte-identical output to the reference path that
/// re-uploads `initial` into a fresh image.
#[test]
fn persistent_target_composes_byte_identically_and_skips_the_seed_upload() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    const W: u32 = 64;
    const H: u32 = 64;
    const BASE: u64 = 0xC0DE_0000;

    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();
    let draw1 = |target_base: Option<u64>| DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        target_base,
        ..DrawState::new(W, H, &vs, &ps)
    };
    // Draw 2 (built inline below): the same triangle squeezed into the
    // top-left quadrant via the viewport, over the LOADed frame — the
    // composition is only correct if draw 1's pixels survive the LOAD.

    // Persistent-target path.
    let target1 = render_draw(dev, &draw1(Some(BASE)))
        .expect("target draw 1 renders")
        .color
        .expect("colour image");
    let stats = dev.draw_cache_stats();
    assert_eq!(
        stats.target_misses, 1,
        "first draw creates the persistent target"
    );
    assert_eq!(stats.seed_uploads_skipped, 0);

    let mut state = draw1(Some(BASE));
    state.viewport = [0.0, 0.0, (W / 2) as f32, (H / 2) as f32];
    state.initial = Some(&target1.pixels);
    let target2 = render_draw(dev, &state)
        .expect("target draw 2 renders")
        .color
        .expect("colour image");
    let stats = dev.draw_cache_stats();
    assert_eq!(
        stats.target_hits, 1,
        "second draw reuses the persistent image"
    );
    assert_eq!(
        stats.seed_uploads_skipped, 1,
        "the synced persistent image must satisfy the LOAD without an upload"
    );

    // Reference path: per-draw images, seed re-uploaded from the CPU bytes.
    let reference1 = render_draw(dev, &draw1(None))
        .expect("reference draw 1 renders")
        .color
        .expect("colour image");
    assert_eq!(
        reference1.pixels, target1.pixels,
        "draw 1 must not depend on the target path at all"
    );
    let mut state = draw1(None);
    state.viewport = [0.0, 0.0, (W / 2) as f32, (H / 2) as f32];
    state.initial = Some(&reference1.pixels);
    let reference2 = render_draw(dev, &state)
        .expect("reference draw 2 renders")
        .color
        .expect("colour image");

    assert_eq!(
        target2.pixels, reference2.pixels,
        "the persistent-target fast path must be byte-identical to seeding by upload"
    );
    assert_eq!(validation_error_count(), 0, "no Vulkan validation errors");
}
