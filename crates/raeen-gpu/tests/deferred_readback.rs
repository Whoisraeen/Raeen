//! Stage B acceptance: deferred readback + direct render-target sampling.
//!
//! Two properties are pinned here:
//!
//! 1. **Readback at most once per flush**: a batch of deferred draws into one
//!    guest target performs ZERO readbacks until [`flush_deferred_draws`],
//!    which reads the target back exactly once — and the flushed bytes are
//!    byte-identical to the old per-draw immediate path composing the same
//!    two draws.
//! 2. **Render-target-as-texture binds the GPU image**: a draw whose T# names
//!    a live persistent target (`TextureUpload::render_target`) samples that
//!    target's rendered pixels straight from the `VkImage` — no CPU round
//!    trip — even when the sampled target's own draw is still pending in the
//!    same deferred batch (queue order + barriers make it visible).
//!
//! Machines without Vulkan 1.3 skip (unless `RAEEN_REQUIRE_VULKAN=1`).

use ash::vk;
use kyty_graphics::spirv_asm::assemble;
use raeen_gpu::backend::GpuBackend;
use raeen_gpu::vulkan::offscreen::{
    CLEAR_COLOR, DepthState, DrawState, ShaderStageBinding, TextureBinding, TextureUpload,
    flush_deferred_draws, render_draw, render_draw_deferred, unorm8,
};
use raeen_gpu::vulkan::{
    TRIANGLE_COLOR, VulkanBackend,
    shaders::{triangle_fragment_spirv, triangle_vertex_spirv},
    validation_error_count,
};

const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

const W: u32 = 64;
const H: u32 = 64;
const TOLERANCE: u8 = 2;

/// Fragment shader in the recompiler's dialect (same shape as
/// `texture_upload.rs`): samples the bound texture at (0.75, 0.75) — a point
/// inside the reference triangle when the texture is a 64x64 render target —
/// and writes the sampled color.
const PS_SAMPLE_TEXTURE: &str = r#"
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %textures2D_S %samplers %push %outColor
               OpExecutionMode %main OriginUpperLeft

               OpDecorate %outColor Location 0
               OpDecorate %textures2D_S DescriptorSet 0
               OpDecorate %textures2D_S Binding 0
               OpDecorate %samplers DescriptorSet 0
               OpDecorate %samplers Binding 1
               OpMemberDecorate %PushBlock 0 Offset 0
               OpMemberDecorate %PushBlock 1 Offset 4
               OpDecorate %PushBlock Block

       %void = OpTypeVoid
        %fty = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
    %v2float = OpTypeVector %float 2
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %outColor = OpVariable %_ptr_Output_v4float Output
     %ImageS = OpTypeImage %float 2D 0 0 0 1 Unknown
   %SampledImage = OpTypeSampledImage %ImageS
     %Sampler = OpTypeSampler
      %uint_1 = OpConstant %uint 1
  %arr_ImageS = OpTypeArray %ImageS %uint_1
  %arr_Sampler = OpTypeArray %Sampler %uint_1
%_ptr_UniformConstant_arr_ImageS = OpTypePointer UniformConstant %arr_ImageS
%_ptr_UniformConstant_arr_Sampler = OpTypePointer UniformConstant %arr_Sampler
%_ptr_UniformConstant_ImageS = OpTypePointer UniformConstant %ImageS
%_ptr_UniformConstant_Sampler = OpTypePointer UniformConstant %Sampler
%textures2D_S = OpVariable %_ptr_UniformConstant_arr_ImageS UniformConstant
  %samplers = OpVariable %_ptr_UniformConstant_arr_Sampler UniformConstant
  %PushBlock = OpTypeStruct %uint %uint
%_ptr_PushConstant_PushBlock = OpTypePointer PushConstant %PushBlock
       %push = OpVariable %_ptr_PushConstant_PushBlock PushConstant
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
      %uint_0 = OpConstant %uint 0
    %float_0_75 = OpConstant %float 0.75

       %main = OpFunction %void None %fty
        %lbl = OpLabel
   %p_tex_idx = OpAccessChain %_ptr_PushConstant_uint %push %uint_0
    %tex_idx = OpLoad %uint %p_tex_idx
      %p_img = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %tex_idx
        %img = OpLoad %ImageS %p_img
   %p_smp_idx = OpAccessChain %_ptr_PushConstant_uint %push %uint_1
    %smp_idx = OpLoad %uint %p_smp_idx
      %p_smp = OpAccessChain %_ptr_UniformConstant_Sampler %samplers %smp_idx
        %smp = OpLoad %Sampler %p_smp
   %sampled = OpSampledImage %SampledImage %img %smp
      %coord = OpCompositeConstruct %v2float %float_0_75 %float_0_75
      %color = OpImageSampleImplicitLod %v4float %sampled %coord
               OpStore %outColor %color
               OpReturn
               OpFunctionEnd
"#;

fn assert_pixel_eq(actual: [u8; 4], expected: [u8; 4], label: &str) {
    let close = actual
        .iter()
        .zip(expected.iter())
        .all(|(a, e)| a.abs_diff(*e) <= TOLERANCE);
    assert!(
        close,
        "{label}: expected RGBA {expected:?} (+/-{TOLERANCE}), read back {actual:?}"
    );
}

fn backend_or_skip() -> Option<VulkanBackend> {
    if std::env::var_os("RAEEN_TEST_LOG").is_some() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    }
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => Some(backend),
        Err(e) => {
            assert!(
                std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
                "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("deferred_readback: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

/// Two deferred draws compose into one target with a single readback at the
/// flush, byte-identical to the immediate per-draw path.
#[test]
fn deferred_batch_reads_back_once_and_composes_byte_identically() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    const BASE: u64 = 0xDEFE_0000;
    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();
    let full = |target_base: Option<u64>| DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        target_base,
        ..DrawState::new(W, H, &vs, &ps)
    };

    // Reference: the old immediate path, per-draw readback, CPU-seeded LOAD.
    let ref1 = render_draw(dev, &full(None))
        .expect("reference draw 1 renders")
        .color
        .expect("colour image");
    let mut ref_state = full(None);
    ref_state.viewport = [0.0, 0.0, (W / 2) as f32, (H / 2) as f32];
    ref_state.initial = Some(&ref1.pixels);
    let ref2 = render_draw(dev, &ref_state)
        .expect("reference draw 2 renders")
        .color
        .expect("colour image");

    // Deferred: same two draws into a persistent target, no readback until
    // the flush. Draw 2 passes NO initial — the GPU copy is the authority
    // (TargetContent::GpuNewer) and the attachment must LOAD from it.
    let before = dev.draw_cache_stats();
    assert!(
        render_draw_deferred(dev, &full(Some(BASE)))
            .expect("deferred draw 1 submits")
            .is_none(),
        "a target-named colour draw must defer, not fall back"
    );
    let mut second = full(Some(BASE));
    second.viewport = [0.0, 0.0, (W / 2) as f32, (H / 2) as f32];
    assert!(
        render_draw_deferred(dev, &second)
            .expect("deferred draw 2 submits")
            .is_none()
    );
    let mid = dev.draw_cache_stats();
    assert_eq!(
        mid.deferred_draws - before.deferred_draws,
        2,
        "both draws must take the deferred path"
    );
    assert_eq!(
        mid.target_readbacks, before.target_readbacks,
        "no readback may happen before the flush"
    );

    let flushed = flush_deferred_draws(dev).expect("flush succeeds");
    let after = dev.draw_cache_stats();
    assert_eq!(after.batch_flushes - before.batch_flushes, 1);
    assert_eq!(
        after.target_readbacks - before.target_readbacks,
        1,
        "two draws into one target must cost exactly one readback"
    );
    assert_eq!(flushed.len(), 1, "one touched target, one flushed image");
    let (base, image) = &flushed[0];
    assert_eq!(*base, BASE);
    assert_eq!(
        image.pixels, ref2.pixels,
        "deferred composition must be byte-identical to the immediate path"
    );

    // A second flush with nothing pending is a no-op.
    assert!(flush_deferred_draws(dev).expect("empty flush").is_empty());
    assert_eq!(validation_error_count(), 0, "no Vulkan validation errors");
}

/// Minecraft's hot path: a named depth target must stay GPU-resident across
/// deferred draws. The batch performs one colour readback at flush and no
/// per-draw fence/readback merely because depth testing is present.
#[test]
fn persistent_depth_target_stays_deferred_across_draws() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let vs = triangle_vertex_spirv();
    let ps = triangle_fragment_spirv();
    const COLOR_BASE: u64 = 0xDEFE_3000;
    const DEPTH_BASE: u64 = 0xDEFE_4000;

    let make_state = |clear_depth| DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        target_base: Some(COLOR_BASE),
        depth: Some(DepthState {
            target_base: Some(DEPTH_BASE),
            format: vk::Format::D32_SFLOAT,
            test_enable: true,
            write_enable: true,
            compare_op: vk::CompareOp::LESS_OR_EQUAL,
            stencil_test_enable: false,
            stencil_front: vk::StencilOpState::default(),
            stencil_back: vk::StencilOpState::default(),
            clear_depth,
            clear_stencil: false,
            clear_depth_value: 1.0,
            clear_stencil_value: 0,
            viewport_depth: [0.0, 1.0],
            initial: None,
            initial_stencil: None,
        }),
        ..DrawState::new(W, H, &vs, &ps)
    };

    let before = dev.draw_cache_stats();
    assert!(
        render_draw_deferred(dev, &make_state(true))
            .expect("first depth draw submits")
            .is_none()
    );
    assert!(
        render_draw_deferred(dev, &make_state(false))
            .expect("second depth draw submits")
            .is_none()
    );
    let mid = dev.draw_cache_stats();
    assert_eq!(mid.deferred_draws - before.deferred_draws, 2);
    assert_eq!(mid.depth_target_misses - before.depth_target_misses, 1);
    assert_eq!(mid.depth_target_hits - before.depth_target_hits, 1);

    let flushed = flush_deferred_draws(dev).expect("depth batch flushes");
    assert_eq!(flushed.len(), 1, "one colour target is read once");
    assert_eq!(flushed[0].0, COLOR_BASE);
    assert_eq!(validation_error_count(), 0, "no Vulkan validation errors");
}

/// A deferred draw samples another target still pending in the same batch by
/// binding its persistent `VkImage` directly — the sampled pixels are the
/// pending draw's output, with zero CPU round trips.
#[test]
fn sampled_render_target_binds_the_gpu_image_within_a_batch() {
    let ps_sample = assemble(PS_SAMPLE_TEXTURE).expect("test PS must assemble");
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    const SCENE: u64 = 0xDEFE_1000;
    const COMPOSITE: u64 = 0xDEFE_2000;
    let vs = triangle_vertex_spirv();
    let fs = triangle_fragment_spirv();

    // Draw A (deferred): the green triangle into the SCENE target.
    let scene = DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        target_base: Some(SCENE),
        ..DrawState::new(W, H, &vs, &fs)
    };
    assert!(
        render_draw_deferred(dev, &scene)
            .expect("scene draw submits")
            .is_none()
    );

    // Draw B (deferred): a triangle into COMPOSITE whose PS samples SCENE at
    // (0.75, 0.75) — inside A's triangle, so the sampled color is A's green.
    let before = dev.draw_cache_stats();
    let composite = DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        target_base: Some(COMPOSITE),
        stage_bindings: vec![ShaderStageBinding {
            stage: ash::vk::ShaderStageFlags::FRAGMENT,
            descriptor_set_slot: 0,
            push_constant_offset: 0,
            push_constants: vec![0u8; 8], // texture index 0, sampler index 0
            push_uniform_binding: None,
            storage_buffers: None,
            textures: Some(TextureBinding {
                sampled_binding: 0,
                sampler_binding: 1,
                textures: vec![TextureUpload {
                    width: W,
                    height: H,
                    format: ash::vk::Format::R8G8B8A8_UNORM,
                    pixels: Vec::new(),
                    layers: 1,
                    cube: false,
                    array: false,
                    volume: false,
                    depth: 1,
                    render_target: Some(SCENE),
                    guest_base: 0,
                    sample_hash: 0,
                    cached: false,
                }],
                samplers: vec![raeen_gpu::vulkan::offscreen::SamplerState::nearest_repeat()],
                sampled_groups: Vec::new(),
            }),
            storage_images: None,
            gds_binding: None,
            eud_raw: None,
            global_mem: None,
        }],
        ..DrawState::new(W, H, &vs, &ps_sample)
    };
    assert!(
        render_draw_deferred(dev, &composite)
            .expect("composite draw submits")
            .is_none()
    );
    let mid = dev.draw_cache_stats();
    assert!(
        mid.sampled_target_binds > before.sampled_target_binds,
        "the composite's T# must bind the persistent image, not upload pixels"
    );

    let flushed = flush_deferred_draws(dev).expect("flush succeeds");
    assert_eq!(flushed.len(), 2, "both touched targets read back once each");
    let composite_image = flushed
        .iter()
        .find(|(base, _)| *base == COMPOSITE)
        .map(|(_, img)| img)
        .expect("composite target flushed");

    // Sampled center pixel: A's triangle color, proving the pending SCENE
    // draw's output was visible to B through the direct GPU binding.
    let center = composite_image
        .pixel(W / 2, H / 2)
        .expect("center pixel in bounds");
    assert_pixel_eq(
        center,
        unorm8(TRIANGLE_COLOR),
        "composite center should be the sampled scene color",
    );
    for (x, y) in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)] {
        let corner = composite_image.pixel(x, y).expect("corner in bounds");
        assert_pixel_eq(corner, unorm8(CLEAR_COLOR), &format!("corner ({x}, {y})"));
    }
    assert_eq!(validation_error_count(), 0, "no Vulkan validation errors");
}
