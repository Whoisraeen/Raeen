//! Acceptance test for the `RAEEN_PIPELINE_STATS` instrument.
//!
//! GTA V executes 1,144 draws with healthy state — full-screen viewport and
//! scissor, `CB_TARGET_MASK` 0xf, blending off, seven shaders translated with
//! zero skips — and every one of its 69 render targets comes back holding
//! nothing but the attachment clear. Nothing left on the CPU side can say
//! whether the hardware rasterized anything, so the answer has to come from the
//! hardware: a per-draw `VK_QUERY_TYPE_PIPELINE_STATISTICS` query.
//!
//! A counter that has never been validated is worse than no counter, so this
//! test proves the instrument SEPARATES the three outcomes it exists to
//! distinguish, using three draws that differ only in their vertex positions:
//!
//! | draw                        | expected reading                          |
//! |-----------------------------|-------------------------------------------|
//! | ordinary triangle           | primitives survive clipping, fragments run |
//! | three identical vertices    | primitive survives, ZERO fragments (no area) |
//! | all positions `w = 0`       | ZERO primitives leave clipping             |
//!
//! The third row is the signature of a vertex shader that never writes
//! `gl_Position`: the builtin keeps its undefined all-zero value, so every
//! vertex has `w == 0` and no primitive survives clipping. That is exactly what
//! a GTA V run must be compared against, and it is why the reading is worth
//! trusting.
//!
//! Machines without a Vulkan 1.3 device skip (unless `RAEEN_REQUIRE_VULKAN=1`).
//! A device without `pipelineStatisticsQuery` also skips — the query is
//! optional in Vulkan, and this test must not fail a conformant device that
//! lacks it.

use ash::vk;
use raeen_gpu::backend::GpuBackend;
use raeen_gpu::vulkan::offscreen::{DrawState, render_draw};
use raeen_gpu::vulkan::shaders::{triangle_fragment_spirv, triangle_vertex_spirv};
use raeen_gpu::vulkan::{
    PipelineStatisticsCensus, VulkanBackend, pipeline_statistics_census, validation_error_count,
};

const W: u32 = 64;
const H: u32 = 64;

/// A triangle covering the middle of the target, in Vulkan NDC.
const COVERING: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

/// Three identical vertices: input assembly builds one primitive, the clipper
/// keeps it, and it covers no pixel. Zero AREA, not zero geometry.
const ZERO_AREA: [[f32; 4]; 3] = [[0.0, 0.0, 0.0, 1.0]; 3];

/// Every position at the origin with `w = 0` — the value an unwritten
/// `gl_Position` holds. No primitive can leave clipping.
const UNWRITTEN_POSITION: [[f32; 4]; 3] = [[0.0, 0.0, 0.0, 0.0]; 3];

/// The delta one draw contributed to the cumulative census.
struct Delta {
    input_vertices: u64,
    clip_invocations: u64,
    clip_primitives: u64,
    fs_invocations: u64,
}

fn delta(before: &PipelineStatisticsCensus, after: &PipelineStatisticsCensus) -> Delta {
    assert_eq!(
        after.measured_draws - before.measured_draws,
        1,
        "exactly one draw must have been measured; the query pool was not recorded or not read"
    );
    Delta {
        input_vertices: after.input_vertices - before.input_vertices,
        clip_invocations: after.clip_invocations - before.clip_invocations,
        clip_primitives: after.clip_primitives - before.clip_primitives,
        fs_invocations: after.fs_invocations - before.fs_invocations,
    }
}

#[test]
fn pipeline_statistics_separate_a_rasterizing_draw_from_a_clipped_and_a_zero_area_one() {
    // Must precede every GPU touch in this process: the switch is snapshotted
    // once into a `OnceLock`, and the device feature it needs is requested at
    // device creation. This is the only test in this binary for that reason.
    //
    // SAFETY: single-threaded, before any other thread exists in this test
    // binary and before any code reads the environment.
    unsafe { std::env::set_var("RAEEN_PIPELINE_STATS", "1") };
    // The skip reasons this test can hit are all reported through `tracing`
    // (unsupported feature, query-pool creation failure); without a subscriber a
    // skip is indistinguishable from a silent bug.
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    let mut backend = VulkanBackend::new(true);
    if let Err(e) = backend.init() {
        assert!(
            std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
            "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
        );
        eprintln!("pipeline_statistics: SKIP — no usable Vulkan 1.3 device ({e})");
        return;
    }
    let dev = backend.device().expect("backend is initialized");
    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();

    // Probe with a real draw: if the device refused `pipelineStatisticsQuery`,
    // no query was recorded and the census stays empty.
    let probe_before = pipeline_statistics_census();
    let state = DrawState {
        vertices: Some(&COVERING),
        vertex_count: COVERING.len() as u32,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..DrawState::new(W, H, &vs, &ps)
    };
    render_draw(dev, &state).expect("the covering triangle must render");
    let probe_after = pipeline_statistics_census();
    if probe_after.measured_draws == probe_before.measured_draws {
        // A skip must mean "this device cannot do it", never "our recording is
        // wrong". An invalid query pool/reset/read shows up as validation
        // errors, so a skip is only honest when there are none. (This caught a
        // real bug: `ash` derives the query count from `data.len()`, so a flat
        // `[u64; 6]` asked for six queries out of a one-query pool and the read
        // failed on a device that fully supports the feature.)
        assert_eq!(
            validation_error_count(),
            0,
            "no statistics were recorded AND Vulkan reported validation errors — \
             the query recording is broken, not the device"
        );
        assert!(
            std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
            "RAEEN_REQUIRE_VULKAN is set but no pipeline statistics were recorded — \
             this device does not support pipelineStatisticsQuery"
        );
        eprintln!(
            "pipeline_statistics: SKIP — device on which no statistics query was recorded \
             (pipelineStatisticsQuery unsupported)"
        );
        return;
    }

    // 1. The covering triangle: primitives survive clipping and fragments run.
    let covering = delta(&probe_before, &probe_after);
    assert_eq!(
        covering.input_vertices, 3,
        "input assembly must see the three submitted vertices"
    );
    assert!(
        covering.clip_primitives >= 1,
        "a triangle inside the frustum must leave clipping, got {}",
        covering.clip_primitives
    );
    assert!(
        covering.fs_invocations > 0,
        "a triangle covering the target centre must invoke the fragment shader"
    );

    // 2. Zero area: the primitive reaches and leaves the clipper, and covers
    //    nothing. This is what a real draw looks like when its geometry
    //    collapses rather than disappears.
    let before = pipeline_statistics_census();
    let state = DrawState {
        vertices: Some(&ZERO_AREA),
        vertex_count: ZERO_AREA.len() as u32,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..DrawState::new(W, H, &vs, &ps)
    };
    render_draw(dev, &state).expect("the zero-area triangle must still submit");
    let zero_area = delta(&before, &pipeline_statistics_census());
    assert_eq!(
        zero_area.input_vertices, 3,
        "the vertices are submitted; it is the AREA that is zero"
    );
    assert_eq!(
        zero_area.fs_invocations, 0,
        "three identical vertices cover no pixel"
    );

    // 3. Unwritten gl_Position (`w == 0`): nothing leaves clipping at all. The
    //    distinction from row 2 is the whole value of the instrument.
    let before = pipeline_statistics_census();
    let state = DrawState {
        vertices: Some(&UNWRITTEN_POSITION),
        vertex_count: UNWRITTEN_POSITION.len() as u32,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..DrawState::new(W, H, &vs, &ps)
    };
    render_draw(dev, &state).expect("a fully clipped draw must still submit");
    let clipped = delta(&before, &pipeline_statistics_census());
    assert!(
        clipped.clip_invocations >= 1,
        "the primitive must reach the clipper before being rejected by it"
    );
    assert_eq!(
        clipped.clip_primitives, 0,
        "a w == 0 triangle must not survive clipping — this is the signature of a \
         vertex shader that never writes gl_Position"
    );
    assert_eq!(clipped.fs_invocations, 0, "nothing survived to rasterize");

    // The three readings must not be the same reading. Without this the three
    // assertions above could all pass on a device that reports constants.
    assert_ne!(
        (covering.clip_primitives, covering.fs_invocations),
        (zero_area.clip_primitives, zero_area.fs_invocations),
        "a covering and a zero-area triangle must read differently"
    );
    assert_ne!(
        zero_area.clip_primitives, clipped.clip_primitives,
        "a zero-AREA primitive and a fully CLIPPED one must read differently"
    );

    assert_eq!(
        validation_error_count(),
        0,
        "the statistics queries must be valid Vulkan (reset outside the render pass \
         instance, begin/end inside the same one)"
    );
}
