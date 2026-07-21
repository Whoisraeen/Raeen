//! Texture chain acceptance: a pixel shader that samples a guest texture
//! through the **recompiler's resource ABI** — a `%textures2D_S` sampled-image
//! array and a `%samplers` array, indexed by descriptor dword 0 carried in
//! push constants, exactly as `image_sample_channels` emits it — produces the
//! texture's pixels in the readback, not the clear color and not black.
//!
//! The shader text is assembled with the same `spirv_asm` the recompiler's
//! own tests use, and the draw runs the real `render_draw` path: image
//! creation, staging upload, layout transitions, descriptor arrays, sampler.
//!
//! Machines without Vulkan 1.3 skip (unless `XPS5X_REQUIRE_VULKAN=1`).

use kyty_graphics::spirv_asm::assemble;
use xps5x_gpu::backend::GpuBackend;
use xps5x_gpu::vulkan::offscreen::{
    CLEAR_COLOR, DrawState, ShaderStageBinding, TextureBinding, TextureUpload, render_draw, unorm8,
};
use xps5x_gpu::vulkan::{VulkanBackend, shaders::triangle_vertex_spirv, validation_error_count};

/// R8G8B8A8_UNORM quantization slack, same rationale as vulkan_triangle.
const TOLERANCE: u8 = 2;

/// The sampled texel: nothing else in the test is magenta, so this pixel can
/// only have come from the texture.
const TEXEL: [u8; 4] = [255, 0, 255, 255];

/// Same triangle the fixture draws, repeated here because the fixture's
/// constant is private to `offscreen`.
const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

/// Fragment shader in the recompiler's dialect: loads the texture index from
/// push-constant dword 0 and the sampler index from dword 4 (the recompiler
/// loads the rewritten T#/S# dword 0 the same way), samples texel (1,1) of a
/// 2x2 texture at constant coordinate (0.75, 0.75), writes the color.
const PS_SAMPLE_TEXTURE: &str = r#"
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %textures2D_S %samplers %push %outColor
               OpExecutionMode %main OriginUpperLeft

               ; Annotations
               OpDecorate %outColor Location 0
               OpDecorate %textures2D_S DescriptorSet 0
               OpDecorate %textures2D_S Binding 0
               OpDecorate %samplers DescriptorSet 0
               OpDecorate %samplers Binding 1
               OpMemberDecorate %PushBlock 0 Offset 0
               OpMemberDecorate %PushBlock 1 Offset 4
               OpDecorate %PushBlock Block

               ; Types, variables and constants
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

               ; Function main
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

/// Bring up the backend, or return `None` when this machine has no usable
/// Vulkan device (unless `XPS5X_REQUIRE_VULKAN` demands one).
fn backend_or_skip() -> Option<VulkanBackend> {
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => {
            let name = backend
                .device()
                .map(|d| d.device_name().to_owned())
                .unwrap_or_default();
            eprintln!("texture_upload: running on {name}");
            Some(backend)
        }
        Err(e) => {
            assert!(
                std::env::var_os("XPS5X_REQUIRE_VULKAN").is_none(),
                "XPS5X_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("texture_upload: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

#[test]
fn sampled_texture_pixels_reach_the_readback() {
    // Opt-in log surface for the validation callback (counts stay assertable
    // without it): XPS5X_TEST_LOG=1 prints the layer's messages.
    if std::env::var_os("XPS5X_TEST_LOG").is_some() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    }
    let ps = assemble(PS_SAMPLE_TEXTURE).expect("test PS must assemble");
    rspirv::dr::load_words(&ps).expect("test PS must parse as SPIR-V");

    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    const W: u32 = 64;
    const H: u32 = 64;

    // 2x2 R8G8B8A8_UNORM; texel (1,1) is the magenta the shader samples.
    let mut pixels = vec![0u8; 16];
    pixels[12..16].copy_from_slice(&TEXEL);
    let texture = TextureUpload {
        width: 2,
        height: 2,
        format: ash::vk::Format::R8G8B8A8_UNORM,
        pixels,
        layers: 1,
        cube: false,
        depth: 1,
        render_target: None,
        guest_base: 0,
        sample_hash: 0,
        cached: false,
    };

    let vs = triangle_vertex_spirv();
    let state = DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
        stage_bindings: vec![ShaderStageBinding {
            stage: ash::vk::ShaderStageFlags::FRAGMENT,
            descriptor_set_slot: 0,
            push_constant_offset: 0,
            push_constants: vec![0u8; 8], // texture index 0, sampler index 0
            storage_buffers: None,
            textures: Some(TextureBinding {
                sampled_binding: 0,
                sampler_binding: 1,
                textures: vec![texture],
                linear_filter: vec![false],
                sampled_groups: Vec::new(),
            }),
            storage_images: None,
            gds_binding: None,
            eud_raw: None,
        }],
        ..DrawState::new(W, H, &vs, &ps)
    };

    let image = render_draw(dev, &state)
        .expect("textured draw must render")
        .color
        .expect("a colour draw produces a colour image");

    // The triangle's center carries the sampled texel; the corners stay on
    // the clear color, so the texture cannot be confused with a full-fill.
    let center = image
        .pixel(W / 2, H / 2)
        .expect("center pixel is in bounds");
    assert_pixel_eq(center, TEXEL, "center should be the sampled texel");
    for (x, y) in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)] {
        let corner = image.pixel(x, y).expect("corner pixel is in bounds");
        assert_pixel_eq(corner, unorm8(CLEAR_COLOR), &format!("corner ({x}, {y})"));
    }

    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the textured draw"
    );
}
