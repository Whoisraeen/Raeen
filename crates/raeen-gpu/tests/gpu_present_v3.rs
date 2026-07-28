//! ABI-v3 GPU present bridge acceptance test.
//!
//! Uses Raeen's GPL-compatible `gpu-blit` plugin: no vendor SDK is involved.

use raeen_gpu::backend::GpuBackend;
use raeen_gpu::present_plugin;
use raeen_gpu::vulkan::offscreen::{
    DrawState, flush_deferred_draws_with_gpu_plugins, render_draw_deferred,
};
use raeen_gpu::vulkan::{
    VulkanBackend,
    shaders::{triangle_fragment_spirv, triangle_vertex_spirv},
    validation_error_count,
};

const VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

#[test]
fn gpu_plugin_records_before_readback_and_keeps_native_seed_separate() {
    assert!(present_plugin::select("gpu-blit"));
    present_plugin::set_output_scale(2.0);

    let mut backend = VulkanBackend::new(true);
    if let Err(error) = backend.init() {
        assert!(
            std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
            "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {error}"
        );
        eprintln!("gpu_present_v3: SKIP — no usable Vulkan device ({error})");
        return;
    }
    let device = backend.device().expect("backend initialized");
    let vertex = triangle_vertex_spirv();
    let fragment = triangle_fragment_spirv();
    let state = DrawState {
        vertices: Some(&VERTICES),
        vertex_count: VERTICES.len() as u32,
        target_base: Some(0xD155_0000),
        ..DrawState::new(32, 24, &vertex, &fragment)
    };
    assert!(
        render_draw_deferred(device, &state)
            .expect("deferred draw records")
            .is_none()
    );

    let (native, plugin) =
        flush_deferred_draws_with_gpu_plugins(device).expect("GPU plugin flush succeeds");
    assert_eq!(native.len(), 1);
    assert_eq!(native[0].1.width, 32);
    assert_eq!(native[0].1.height, 24);
    assert_eq!(plugin.len(), 1);
    assert_eq!(plugin[0].0, native[0].0);
    assert_eq!(plugin[0].1.width, 64);
    assert_eq!(plugin[0].1.height, 48);
    assert_eq!(plugin[0].1.bytes_per_pixel, native[0].1.bytes_per_pixel);
    assert!(
        plugin[0].1.pixels.iter().any(|byte| *byte != 0),
        "the plugin output must contain recorded GPU work"
    );
    assert_eq!(validation_error_count(), 0);
}
