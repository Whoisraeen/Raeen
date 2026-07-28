//! Persistent-texture cache acceptance (perf stage D item 1).
//!
//! Contract under test, end to end on the real `render_draw` path:
//!
//! 1. A cacheable upload (non-zero `guest_base` + `sample_hash`) renders
//!    byte-identically to the same draw with caching disabled (`guest_base
//!    == 0`) — the cache changes WHERE the texels come from, never what they
//!    are.
//! 2. A second draw whose `TextureUpload` arrives as a cache hit (`cached:
//!    true`, empty pixels, matching hash) binds the cached image and renders
//!    byte-identically to the first draw.
//! 3. After the guest content changes in a way the sample-hash catches (a
//!    new hash + new pixels — exactly what the decode path produces on a
//!    hash mismatch), the draw shows the NEW content and the stale entry is
//!    evicted; a subsequent hit on the new hash returns the new content.
//!
//! Machines without Vulkan 1.3 skip (unless `RAEEN_REQUIRE_VULKAN=1`).

use kyty_graphics::spirv_asm::assemble;
use raeen_gpu::backend::GpuBackend;
use raeen_gpu::vulkan::offscreen::{
    DrawState, ShaderStageBinding, TextureBinding, TextureUpload, render_draw,
};
use raeen_gpu::vulkan::{VulkanBackend, shaders::triangle_vertex_spirv, validation_error_count};

/// R8G8B8A8_UNORM quantization slack, same rationale as vulkan_triangle.
const TOLERANCE: u8 = 2;

const MAGENTA: [u8; 4] = [255, 0, 255, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];

/// A guest base address for the cached texture. Arbitrary but non-zero — the
/// offscreen path only compares identities, it never dereferences this.
const GUEST_BASE: u64 = 0x4000_0000;

const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0],
    [0.7, 0.7, 0.0, 1.0],
    [-0.7, 0.7, 0.0, 1.0],
];

/// Same recompiler-dialect sampling PS as tests/texture_upload.rs: texture
/// index from push-constant dword 0, sampler index from dword 4, samples
/// texel (1,1) of a 2x2 texture at (0.75, 0.75).
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

fn backend_or_skip() -> Option<VulkanBackend> {
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => {
            let name = backend
                .device()
                .map(|d| d.device_name().to_owned())
                .unwrap_or_default();
            eprintln!("texture_cache: running on {name}");
            Some(backend)
        }
        Err(e) => {
            assert!(
                std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
                "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("texture_cache: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

/// A 2x2 RGBA8 texture whose texel (1,1) is `texel`.
fn texture_pixels(texel: [u8; 4]) -> Vec<u8> {
    let mut pixels = vec![0u8; 16];
    pixels[12..16].copy_from_slice(&texel);
    pixels
}

fn draw_state<'a>(vs: &'a [u32], ps: &'a [u32], texture: TextureUpload) -> DrawState<'a> {
    const W: u32 = 64;
    const H: u32 = 64;
    DrawState {
        vertices: Some(&TRIANGLE_VERTICES),
        vertex_count: TRIANGLE_VERTICES.len() as u32,
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
                textures: vec![texture],
                samplers: vec![raeen_gpu::vulkan::offscreen::SamplerState::nearest_repeat()],
                sampled_groups: Vec::new(),
            }),
            storage_images: None,
            gds_binding: None,
            eud_raw: None,
        }],
        ..DrawState::new(W, H, vs, ps)
    }
}

fn assert_center(pixels: &raeen_gpu::vulkan::RenderedImage, expected: [u8; 4], label: &str) {
    let center = pixels.pixel(32, 32).expect("center pixel is in bounds");
    let close = center
        .iter()
        .zip(expected.iter())
        .all(|(a, e)| a.abs_diff(*e) <= TOLERANCE);
    assert!(
        close,
        "{label}: expected center RGBA {expected:?} (+/-{TOLERANCE}), read back {center:?}"
    );
}

#[test]
fn cached_texture_reuse_is_byte_identical_and_invalidates_on_content_change() {
    if std::env::var_os("RAEEN_NO_TEX_CACHE").is_some() {
        eprintln!("texture_cache: SKIP — RAEEN_NO_TEX_CACHE disables the cache under test");
        return;
    }
    let ps = assemble(PS_SAMPLE_TEXTURE).expect("test PS must assemble");
    let vs = triangle_vertex_spirv();
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    // Arbitrary distinct content hashes: the backend only ever compares them
    // for equality (the real values come from `guest_sample_hash` over guest
    // memory, which the fixture path has none of).
    const HASH_V1: u64 = 0x1111_2222_3333_4444;
    const HASH_V2: u64 = 0x5555_6666_7777_8888;

    // Control: identical draw with the cache disabled for this upload.
    let control = render_draw(
        dev,
        &draw_state(
            &vs,
            &ps,
            TextureUpload {
                width: 2,
                height: 2,
                format: ash::vk::Format::R8G8B8A8_UNORM,
                pixels: texture_pixels(MAGENTA),
                layers: 1,
                cube: false,
                array: false,
                volume: false,
                depth: 1,
                render_target: None,
                guest_base: 0, // cache disabled
                sample_hash: 0,
                cached: false,
            },
        ),
    )
    .expect("uncached control draw renders")
    .color
    .expect("colour draw produces an image");
    assert_center(&control, MAGENTA, "uncached control");

    // Draw 1: cacheable miss — uploads and donates the image to the cache.
    let stats0 = dev.draw_cache_stats();
    let first = render_draw(
        dev,
        &draw_state(
            &vs,
            &ps,
            TextureUpload {
                width: 2,
                height: 2,
                format: ash::vk::Format::R8G8B8A8_UNORM,
                pixels: texture_pixels(MAGENTA),
                layers: 1,
                cube: false,
                array: false,
                volume: false,
                depth: 1,
                render_target: None,
                guest_base: GUEST_BASE,
                sample_hash: HASH_V1,
                cached: false,
            },
        ),
    )
    .expect("cacheable miss draw renders")
    .color
    .expect("colour draw produces an image");
    let stats1 = dev.draw_cache_stats();
    assert_eq!(
        stats1.texture_cache_misses,
        stats0.texture_cache_misses + 1,
        "the first cacheable upload must count one texture-cache miss"
    );
    assert_eq!(
        first.pixels, control.pixels,
        "a cacheable upload must render byte-identically to the uncached control"
    );

    // Draw 2: cache hit — empty pixels, matching hash; binds the cached image.
    let second = render_draw(
        dev,
        &draw_state(
            &vs,
            &ps,
            TextureUpload {
                width: 2,
                height: 2,
                format: ash::vk::Format::R8G8B8A8_UNORM,
                pixels: Vec::new(),
                layers: 1,
                cube: false,
                array: false,
                volume: false,
                depth: 1,
                render_target: None,
                guest_base: GUEST_BASE,
                sample_hash: HASH_V1,
                cached: true,
            },
        ),
    )
    .expect("cache-hit draw renders")
    .color
    .expect("colour draw produces an image");
    let stats2 = dev.draw_cache_stats();
    assert_eq!(
        stats2.texture_cache_hits,
        stats1.texture_cache_hits + 1,
        "the second draw must be served by the texture cache"
    );
    assert_eq!(
        stats2.texture_cache_misses, stats1.texture_cache_misses,
        "a cache hit must not re-upload"
    );
    assert_eq!(
        second.pixels, first.pixels,
        "a cache-hit draw must render byte-identically to the upload draw"
    );

    // Draw 3: guest content changed in a way the sample-hash catches — the
    // decode path re-reads and produces a fresh upload with a new hash; the
    // stale cached entry is evicted at donation.
    let third = render_draw(
        dev,
        &draw_state(
            &vs,
            &ps,
            TextureUpload {
                width: 2,
                height: 2,
                format: ash::vk::Format::R8G8B8A8_UNORM,
                pixels: texture_pixels(GREEN),
                layers: 1,
                cube: false,
                array: false,
                volume: false,
                depth: 1,
                render_target: None,
                guest_base: GUEST_BASE,
                sample_hash: HASH_V2,
                cached: false,
            },
        ),
    )
    .expect("post-mutation draw renders")
    .color
    .expect("colour draw produces an image");
    let stats3 = dev.draw_cache_stats();
    assert_center(&third, GREEN, "post-mutation draw sees the new content");
    assert_eq!(
        stats3.texture_cache_evictions,
        stats2.texture_cache_evictions + 1,
        "replacing the same key with new content must evict the stale entry"
    );

    // Draw 4: hit on the NEW content.
    let fourth = render_draw(
        dev,
        &draw_state(
            &vs,
            &ps,
            TextureUpload {
                width: 2,
                height: 2,
                format: ash::vk::Format::R8G8B8A8_UNORM,
                pixels: Vec::new(),
                layers: 1,
                cube: false,
                array: false,
                volume: false,
                depth: 1,
                render_target: None,
                guest_base: GUEST_BASE,
                sample_hash: HASH_V2,
                cached: true,
            },
        ),
    )
    .expect("cache-hit draw on new content renders")
    .color
    .expect("colour draw produces an image");
    assert_eq!(
        fourth.pixels, third.pixels,
        "a cache hit after invalidation must serve the NEW content byte-identically"
    );

    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the texture-cache draws"
    );
}
