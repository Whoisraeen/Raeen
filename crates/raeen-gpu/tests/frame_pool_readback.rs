//! Full-frame readback destinations are recycled, and recycling changes no pixel.
//!
//! The present path's last CPU crossing is
//! `vkCmdCopyImageToBuffer` → `vkMapMemory` → copy into an owned `Vec<u8>`.
//! That `Vec` was freshly allocated on every frame, so the copy also paid a
//! soft page fault per 4 KiB of a multi-megabyte mapping — measured at ~870 us
//! per 1080p frame on the dev machine, and it scales with resolution.
//!
//! Two properties are pinned here, and they are the two that matter:
//!
//! 1. **Recycling happens**: once a frame is dropped, the next readback of the
//!    same size reuses that exact allocation (`frame_pool` hit, same pointer)
//!    instead of allocating.
//! 2. **Recycling is invisible**: a readback into a recycled buffer is
//!    byte-identical to the same readback into a fresh one. This is the
//!    guard-rail for the whole change — a performance win that alters output
//!    is a regression, not a win.
//!
//! One `#[test]` on purpose: the pool is process-global, so two tests asserting
//! on its occupancy would race inside one test binary.
//!
//! Machines without Vulkan 1.3 skip (unless `RAEEN_REQUIRE_VULKAN=1`).

use raeen_gpu::backend::GpuBackend;
use raeen_gpu::frame_pool;
use raeen_gpu::vulkan::offscreen::{DrawState, flush_deferred_draws, render_draw_deferred};
use raeen_gpu::vulkan::{
    VulkanBackend,
    shaders::{triangle_fragment_spirv, triangle_vertex_spirv},
    validation_error_count,
};

const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

/// 640x480 RGBA = 1.17 MB, over the pool's 1 MiB recycling threshold. A 64x64
/// target (16 KiB, what the other GPU tests use) is deliberately below it.
const W: u32 = 640;
const H: u32 = 480;

fn backend_or_skip() -> Option<VulkanBackend> {
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => Some(backend),
        Err(e) => {
            assert!(
                std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
                "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("frame_pool_readback: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

#[test]
fn a_recycled_readback_destination_is_the_same_allocation_and_the_same_pixels() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    const BASE: u64 = 0xF00D_0000;
    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();
    let draw = || DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        target_base: Some(BASE),
        ..DrawState::new(W, H, &vs, &ps)
    };

    // Frame 1 — the pool starts empty, so this readback allocates.
    frame_pool::clear();
    let baseline = frame_pool::stats();
    assert_eq!(baseline.buffers, 0, "the pool must start empty");

    render_draw_deferred(dev, &draw()).expect("deferred draw 1 submits");
    let mut first = flush_deferred_draws(dev).expect("flush 1 succeeds");
    assert_eq!(first.len(), 1, "one touched target");
    let (_, first_image) = first.pop().expect("one flushed image");

    let after_first = frame_pool::stats();
    assert_eq!(
        after_first.hits - baseline.hits,
        0,
        "with an empty pool the first readback must allocate, not hit"
    );
    assert_eq!(
        after_first.misses - baseline.misses,
        1,
        "the first readback must record exactly one miss"
    );

    // Keep the pixels and the address, then drop the frame so its buffer is
    // offered back. `expected` is an independent copy for the comparison
    // below, so dropping the original cannot affect it.
    let expected = first_image.pixels.clone();
    let recycled_address = first_image.pixels.as_ptr() as usize;
    let recycled_size = first_image.pixels.len();
    assert_eq!(
        recycled_size,
        (W * H * 4) as usize,
        "the target is 640x480 RGBA"
    );
    drop(first_image);

    assert_eq!(
        frame_pool::stats().buffers,
        1,
        "dropping a full-frame image must return its buffer to the pool"
    );

    // Frame 2 — identical draw. Its readback must reuse frame 1's allocation.
    render_draw_deferred(dev, &draw()).expect("deferred draw 2 submits");
    let mut second = flush_deferred_draws(dev).expect("flush 2 succeeds");
    let (_, second_image) = second.pop().expect("one flushed image");

    let after_second = frame_pool::stats();
    assert_eq!(
        after_second.hits - after_first.hits,
        1,
        "the second readback must be served from the pool"
    );
    assert_eq!(
        after_second.misses - after_first.misses,
        0,
        "a served request must not also count as a miss"
    );
    assert_eq!(
        second_image.pixels.as_ptr() as usize,
        recycled_address,
        "the recycled readback must land in the SAME allocation, not a fresh one"
    );

    // The property the whole change rests on.
    assert_eq!(
        second_image.pixels, expected,
        "a readback into a recycled buffer must be byte-identical to one into \
         a fresh buffer"
    );
    assert_eq!(
        second_image.pixels.len(),
        recycled_size,
        "a recycled buffer must not change the frame's length — every consumer \
         of RenderedImage::pixels reads len(), not capacity()"
    );

    // A sub-threshold image must never be retained: the test suite builds these
    // by the thousand and the pool exists for full frames only.
    drop(second_image);
    frame_pool::clear();
    let small = raeen_gpu::RenderedImage {
        width: 4,
        height: 4,
        pixels: vec![0x5a; 4 * 4 * 4],
        bytes_per_pixel: 4,
    };
    drop(small);
    assert_eq!(
        frame_pool::stats().buffers,
        0,
        "a 64-byte image must not be retained by the frame pool"
    );

    frame_pool::clear();
    assert_eq!(validation_error_count(), 0, "no Vulkan validation errors");
}
