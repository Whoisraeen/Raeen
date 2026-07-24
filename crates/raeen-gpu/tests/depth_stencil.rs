//! Depth/stencil attachment acceptance: a draw with a [`DepthState`] on its
//! [`DrawState`] creates a real Vulkan depth (and, for a stencil-bearing
//! format, stencil) attachment, clears or loads it, runs the draw, and reads
//! the planes back into a [`DepthImage`]. Exercises the full path added for the
//! depth/stencil pipeline: `create_depth_target` → pipeline depth-stencil state
//! → dynamic-rendering depth/stencil attachments → per-aspect barriers →
//! copy-out → `read_back_depth`.
//!
//! The asserts use depth WRITE disabled so the readback is exactly the
//! clear/loaded value, independent of the vertex shader's interpolated z — the
//! attachment mechanics are what's under test, not shader depth output.
//!
//! Machines without Vulkan 1.3 skip (unless `RAEEN_REQUIRE_VULKAN=1`).

use ash::vk;
use raeen_gpu::backend::GpuBackend;
use raeen_gpu::vulkan::offscreen::{CLEAR_COLOR, DepthState, DrawState, render_draw, unorm8};
use raeen_gpu::vulkan::shaders::{triangle_fragment_spirv, triangle_vertex_spirv};
use raeen_gpu::vulkan::{VulkanBackend, validation_error_count};

const W: u32 = 32;
const H: u32 = 32;
const DEPTH_TOLERANCE: f32 = 1.0 / 1024.0;

/// The triangle `triangle_vertex_spirv` reads from vertex input Location 0.
const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

fn backend_or_skip(name: &str) -> Option<VulkanBackend> {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => Some(backend),
        Err(e) => {
            assert!(
                std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
                "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("{name}: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

/// A depth-only stencil-less [`DepthState`] with test/write disabled: the
/// attachment is CLEAR-ed to `clear` and never written, so the whole plane
/// reads back `clear`.
fn cleared_depth_state(format: vk::Format, clear: f32) -> DepthState<'static> {
    DepthState {
        target_base: None,
        format,
        test_enable: false,
        write_enable: false,
        compare_op: vk::CompareOp::ALWAYS,
        stencil_test_enable: false,
        stencil_front: vk::StencilOpState::default(),
        stencil_back: vk::StencilOpState::default(),
        clear_depth: true,
        clear_stencil: false,
        clear_depth_value: clear,
        clear_stencil_value: 0,
        viewport_depth: [0.0, 1.0],
        initial: None,
        initial_stencil: None,
    }
}

#[test]
fn persistent_depth_target_reuses_image_for_same_guest_surface() {
    let Some(backend) =
        backend_or_skip("persistent_depth_target_reuses_image_for_same_guest_surface")
    else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();

    for clear in [0.25, 0.75] {
        let state = DrawState {
            vertices: Some(&TRIANGLE_VERTICES),
            vertex_count: TRIANGLE_VERTICES.len() as u32,
            depth: Some(DepthState {
                target_base: Some(0x1234_0000),
                ..cleared_depth_state(vk::Format::D32_SFLOAT, clear)
            }),
            ..DrawState::new(W, H, &vs, &ps)
        };
        let output = render_draw(dev, &state).expect("persistent depth draw");
        let image = output.depth.expect("depth readback");
        let actual = image.depth_at(0, 0).expect("depth texel");
        assert!((actual - clear).abs() <= DEPTH_TOLERANCE);
    }

    let stats = dev.draw_cache_stats();
    assert_eq!(stats.depth_target_misses, 1, "first bind creates the image");
    assert_eq!(
        stats.depth_target_hits, 1,
        "second bind reuses the same image/view/allocation"
    );
}

#[test]
fn depth_attachment_clears_and_reads_back() {
    let Some(backend) = backend_or_skip("depth_attachment_clears_and_reads_back") else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();
    let state = DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        // D32_SFLOAT is a Vulkan-required depth-attachment format, so this runs
        // on any conformant device.
        depth: Some(cleared_depth_state(vk::Format::D32_SFLOAT, 0.75)),
        ..DrawState::new(W, H, &vs, &ps)
    };

    let output = render_draw(dev, &state).expect("depth draw must render");

    // The colour attachment is still produced alongside the depth one.
    assert!(
        output.color.is_some(),
        "a colour+depth draw must still produce a colour image"
    );

    let depth = output.depth.expect("a depth attachment must read back");
    assert_eq!(depth.format, vk::Format::D32_SFLOAT);
    assert_eq!((depth.width, depth.height), (W, H));
    assert!(
        depth.stencil.is_none(),
        "D32_SFLOAT carries no stencil plane"
    );

    // Write was disabled, so every texel is the clear value 0.75.
    for (x, y) in [(0, 0), (W / 2, H / 2), (W - 1, H - 1)] {
        let d = depth.depth_at(x, y).expect("depth texel in bounds");
        assert!(
            (d - 0.75).abs() <= DEPTH_TOLERANCE,
            "depth at ({x},{y}) = {d}, expected ~0.75"
        );
    }

    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the depth draw"
    );
}

#[test]
fn depth_attachment_loads_prior_contents() {
    let Some(backend) = backend_or_skip("depth_attachment_loads_prior_contents") else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    // Seed the whole depth plane with 0.25 and LOAD it (no clear). Write stays
    // disabled, so the readback is exactly the seeded value — proving the depth
    // upload/seed path, not just CLEAR.
    let seed: Vec<u8> = (0..(W * H)).flat_map(|_| 0.25f32.to_le_bytes()).collect();
    let depth_state = DepthState {
        clear_depth: false,
        initial: Some(&seed),
        ..cleared_depth_state(vk::Format::D32_SFLOAT, 1.0)
    };

    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();
    let state = DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        depth: Some(depth_state),
        ..DrawState::new(W, H, &vs, &ps)
    };

    let output = render_draw(dev, &state).expect("depth LOAD draw must render");
    let depth = output.depth.expect("a depth attachment must read back");
    for (x, y) in [(0, 0), (W / 2, H / 2), (W - 1, H - 1)] {
        let d = depth.depth_at(x, y).expect("depth texel in bounds");
        assert!(
            (d - 0.25).abs() <= DEPTH_TOLERANCE,
            "loaded depth at ({x},{y}) = {d}, expected ~0.25"
        );
    }

    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the depth LOAD draw"
    );
}

#[test]
fn stencil_attachment_clears_and_reads_back() {
    let Some(backend) = backend_or_skip("stencil_attachment_clears_and_reads_back") else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    // Keep every stencil op (no writes) so the plane stays at its clear value;
    // compare ALWAYS so nothing is discarded on the stencil test.
    let stencil_op = vk::StencilOpState::default()
        .fail_op(vk::StencilOp::KEEP)
        .pass_op(vk::StencilOp::KEEP)
        .depth_fail_op(vk::StencilOp::KEEP)
        .compare_op(vk::CompareOp::ALWAYS)
        .compare_mask(0xFF)
        .write_mask(0xFF)
        .reference(0);

    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();

    // Vulkan guarantees at least one of these depth+stencil formats; try both
    // so the test runs regardless of which the device supports.
    let mut ran = false;
    for &format in &[
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D32_SFLOAT_S8_UINT,
    ] {
        let depth_state = DepthState {
            format,
            stencil_test_enable: true,
            stencil_front: stencil_op,
            stencil_back: stencil_op,
            clear_stencil: true,
            clear_stencil_value: 0x2A,
            ..cleared_depth_state(format, 1.0)
        };
        let state = DrawState {
            vertices: Some(&TRIANGLE_VERTICES),
            vertex_count: TRIANGLE_VERTICES.len() as u32,
            depth: Some(depth_state),
            ..DrawState::new(W, H, &vs, &ps)
        };

        let output = match render_draw(dev, &state) {
            Ok(output) => output,
            // The format is optional per device; try the next candidate.
            Err(e) => {
                eprintln!("stencil: {format:?} unsupported ({e}); trying next format");
                continue;
            }
        };
        ran = true;

        let depth = output
            .depth
            .expect("a depth/stencil attachment must read back");
        assert_eq!(depth.format, format);
        let stencil = depth.stencil.as_ref().expect("stencil plane present");
        assert_eq!(stencil.len() as u32, W * H);

        for (x, y) in [(0, 0), (W / 2, H / 2), (W - 1, H - 1)] {
            assert_eq!(
                depth.stencil_at(x, y),
                Some(0x2A),
                "stencil at ({x},{y}) must be the clear value"
            );
            let d = depth.depth_at(x, y).expect("depth texel in bounds");
            assert!(
                (d - 1.0).abs() <= DEPTH_TOLERANCE,
                "depth at ({x},{y}) = {d}, expected ~1.0"
            );
        }

        assert_eq!(
            validation_error_count(),
            0,
            "Vulkan validation reported errors during the stencil draw ({format:?})"
        );
        break;
    }

    if !ran {
        assert!(
            std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
            "RAEEN_REQUIRE_VULKAN is set but no depth+stencil format was supported"
        );
        eprintln!("stencil_attachment_clears_and_reads_back: SKIP — no D/S format supported");
    }
}

#[test]
fn persistent_zero_stencil_equal_test_keeps_rasterizing() {
    let Some(backend) = backend_or_skip("persistent_zero_stencil_equal_test_keeps_rasterizing")
    else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let stencil_op = vk::StencilOpState::default()
        .fail_op(vk::StencilOp::KEEP)
        .pass_op(vk::StencilOp::KEEP)
        .depth_fail_op(vk::StencilOp::KEEP)
        .compare_op(vk::CompareOp::EQUAL)
        .compare_mask(0xF0)
        .write_mask(0xFF)
        .reference(0);
    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();

    for clear_stencil in [true, false] {
        let depth_state = DepthState {
            target_base: Some(0x5678_0000),
            format: vk::Format::D32_SFLOAT_S8_UINT,
            stencil_test_enable: true,
            stencil_front: stencil_op,
            stencil_back: stencil_op,
            clear_stencil,
            clear_stencil_value: 0,
            ..cleared_depth_state(vk::Format::D32_SFLOAT_S8_UINT, 1.0)
        };
        let state = DrawState {
            vertices: Some(&TRIANGLE_VERTICES),
            vertex_count: TRIANGLE_VERTICES.len() as u32,
            depth: Some(depth_state),
            ..DrawState::new(W, H, &vs, &ps)
        };
        let output = render_draw(dev, &state).expect("zero-stencil EQUAL draw");
        let image = output.color.expect("colour attachment");
        let clear = unorm8(CLEAR_COLOR);
        assert!(
            image
                .pixels
                .chunks_exact(4)
                .any(|pixel| pixel != clear.as_slice()),
            "stencil EQUAL(ref=0, mask=0xf0) rejected every fragment \
             (clear_stencil={clear_stencil})"
        );
        assert_eq!(
            output
                .depth
                .expect("depth/stencil readback")
                .stencil_at(W / 2, H / 2),
            Some(0)
        );
    }

    assert_eq!(validation_error_count(), 0);
}
