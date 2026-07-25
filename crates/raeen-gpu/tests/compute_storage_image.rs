//! Storage-image (UAV) compute round-trip on a real Vulkan device.
//!
//! ASTRO.BOT renders its 3D scene from compute shaders into storage images
//! (`%textures2D_L`, `OpTypeImage ... 2 Rgba8`), so the dispatch path must be
//! able to bind a STORAGE_IMAGE descriptor array, run the shader, and read the
//! written pixels back for the guest-memory writeback. This test dispatches a
//! tiny shader that writes a known per-pixel pattern into a 2x2 RGBA8 storage
//! image and asserts on the exact bytes that came back from the device.
//!
//! Machines without a Vulkan 1.3 device **skip** (like `vulkan_triangle`);
//! `RAEEN_REQUIRE_VULKAN=1` turns the skip into a failure.

use ash::vk;
use raeen_gpu::backend::GpuBackend;
use raeen_gpu::vulkan::compute::{ComputeState, dispatch_compute};
use raeen_gpu::vulkan::offscreen::{ShaderStageBinding, StorageImageBinding, StorageImageUpload};
use raeen_gpu::vulkan::{VulkanBackend, validation_error_count};

/// UNORM8 conversion is allowed a little slack (0.6 ULP), same as the other
/// pixel-readback tests.
const TOLERANCE: u8 = 1;

/// `LocalSize 2 2 1` compute shader writing, per invocation:
/// `rgba = ((gid.x*100+10)/255, (gid.y*100+20)/255, 1.0, 200/255)`
/// into element 0 of a one-entry storage-image array at set 0, binding 0 —
/// the same declaration shape the recompiler emits for `%textures2D_L`.
const STORE_PATTERN_CS: &str = "\
    OpCapability Shader\n\
    OpMemoryModel Logical GLSL450\n\
    OpEntryPoint GLCompute %main \"main\" %gid %images\n\
    OpExecutionMode %main LocalSize 2 2 1\n\
    OpDecorate %gid BuiltIn GlobalInvocationId\n\
    OpDecorate %images DescriptorSet 0\n\
    OpDecorate %images Binding 0\n\
    %void = OpTypeVoid\n\
    %fnty = OpTypeFunction %void\n\
    %uint = OpTypeInt 32 0\n\
    %float = OpTypeFloat 32\n\
    %v2uint = OpTypeVector %uint 2\n\
    %v3uint = OpTypeVector %uint 3\n\
    %v4float = OpTypeVector %float 4\n\
    %imgL = OpTypeImage %float 2D 0 0 0 2 Rgba8\n\
    %uint_1 = OpConstant %uint 1\n\
    %arr = OpTypeArray %imgL %uint_1\n\
    %ptr_arr = OpTypePointer UniformConstant %arr\n\
    %ptr_img = OpTypePointer UniformConstant %imgL\n\
    %ptr_gid = OpTypePointer Input %v3uint\n\
    %uint_0 = OpConstant %uint 0\n\
    %f100 = OpConstant %float 100.000000\n\
    %f10 = OpConstant %float 10.000000\n\
    %f20 = OpConstant %float 20.000000\n\
    %f200 = OpConstant %float 200.000000\n\
    %f255 = OpConstant %float 255.000000\n\
    %f1 = OpConstant %float 1.000000\n\
    %images = OpVariable %ptr_arr UniformConstant\n\
    %gid = OpVariable %ptr_gid Input\n\
    %main = OpFunction %void None %fnty\n\
    %entry = OpLabel\n\
    %g = OpLoad %v3uint %gid\n\
    %gx = OpCompositeExtract %uint %g 0\n\
    %gy = OpCompositeExtract %uint %g 1\n\
    %fx = OpConvertUToF %float %gx\n\
    %fy = OpConvertUToF %float %gy\n\
    %rx = OpFMul %float %fx %f100\n\
    %r10 = OpFAdd %float %rx %f10\n\
    %r = OpFDiv %float %r10 %f255\n\
    %gyx = OpFMul %float %fy %f100\n\
    %g20 = OpFAdd %float %gyx %f20\n\
    %gc = OpFDiv %float %g20 %f255\n\
    %a = OpFDiv %float %f200 %f255\n\
    %color = OpCompositeConstruct %v4float %r %gc %f1 %a\n\
    %coord = OpCompositeConstruct %v2uint %gx %gy\n\
    %pimg = OpAccessChain %ptr_img %images %uint_0\n\
    %img = OpLoad %imgL %pimg\n\
    OpImageWrite %img %coord %color\n\
    OpReturn\n\
    OpFunctionEnd\n";

/// Array-storage acceptance shader: dispatch Z selects the destination layer.
/// Layer 0 becomes green and layer 1 red, proving the third image coordinate,
/// 2D-array view, barriers, copies, and full-layer readback all agree.
const STORE_ARRAY_CS: &str = "\
    OpCapability Shader\n\
    OpMemoryModel Logical GLSL450\n\
    OpEntryPoint GLCompute %main \"main\" %gid %images\n\
    OpExecutionMode %main LocalSize 2 2 1\n\
    OpDecorate %gid BuiltIn GlobalInvocationId\n\
    OpDecorate %images DescriptorSet 0\n\
    OpDecorate %images Binding 0\n\
    %void = OpTypeVoid\n\
    %fnty = OpTypeFunction %void\n\
    %uint = OpTypeInt 32 0\n\
    %float = OpTypeFloat 32\n\
    %v3uint = OpTypeVector %uint 3\n\
    %v4float = OpTypeVector %float 4\n\
    %imgL = OpTypeImage %float 2D 0 1 0 2 Rgba8\n\
    %uint_1 = OpConstant %uint 1\n\
    %uint_0 = OpConstant %uint 0\n\
    %float_0 = OpConstant %float 0.000000\n\
    %float_1 = OpConstant %float 1.000000\n\
    %arr = OpTypeArray %imgL %uint_1\n\
    %ptr_arr = OpTypePointer UniformConstant %arr\n\
    %ptr_img = OpTypePointer UniformConstant %imgL\n\
    %ptr_gid = OpTypePointer Input %v3uint\n\
    %images = OpVariable %ptr_arr UniformConstant\n\
    %gid = OpVariable %ptr_gid Input\n\
    %main = OpFunction %void None %fnty\n\
    %entry = OpLabel\n\
    %coord = OpLoad %v3uint %gid\n\
    %z = OpCompositeExtract %uint %coord 2\n\
    %red = OpConvertUToF %float %z\n\
    %green = OpFSub %float %float_1 %red\n\
    %color = OpCompositeConstruct %v4float %red %green %float_0 %float_1\n\
    %pimg = OpAccessChain %ptr_img %images %uint_0\n\
    %img = OpLoad %imgL %pimg\n\
    OpImageWrite %img %coord %color\n\
    OpReturn\n\
    OpFunctionEnd\n";

fn backend_or_skip() -> Option<VulkanBackend> {
    let mut backend = VulkanBackend::new(true);
    match backend.init() {
        Ok(()) => {
            let name = backend
                .device()
                .map(|d| d.device_name().to_owned())
                .unwrap_or_default();
            eprintln!("compute_storage_image: running on {name}");
            Some(backend)
        }
        Err(e) => {
            assert!(
                std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
                "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("compute_storage_image: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

/// The dispatch writes every pixel of the 2x2 UAV; the readback bytes must be
/// the shader's pattern, proving upload -> GENERAL -> dispatch -> readback all
/// wired through the STORAGE_IMAGE descriptor array.
#[test]
fn compute_shader_writes_are_visible_in_storage_image_readback() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    let spirv = kyty_graphics::spirv_asm::assemble(STORE_PATTERN_CS)
        .expect("storage-image compute shader assembles");

    // Seed with a sentinel so a dispatch that silently never ran cannot pass
    // (all-zero would be indistinguishable from a failed write of zeros).
    let binding = ShaderStageBinding {
        stage: vk::ShaderStageFlags::COMPUTE,
        descriptor_set_slot: 0,
        push_constant_offset: 0,
        push_constants: Vec::new(),
        push_uniform_binding: None,
        storage_buffers: None,
        textures: None,
        storage_images: Some(StorageImageBinding {
            binding: 0,
            images: vec![StorageImageUpload {
                width: 2,
                height: 2,
                depth: 1,
                layers: 1,
                array: false,
                tile_mode: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                pixels: vec![0xEE; 16].into(),
                guest_base: 0,
            }],
        }),
        gds_binding: None,
        eud_raw: None,
    };

    let outputs = dispatch_compute(
        dev,
        &ComputeState {
            groups: [1, 1, 1],
            spirv: &spirv,
            binding: Some(&binding),
        },
    )
    .expect("storage-image compute dispatch");

    assert!(outputs.buffers.is_empty(), "no storage buffers were bound");
    assert_eq!(outputs.images.len(), 1, "one UAV readback");
    let pixels = &outputs.images[0];
    assert_eq!(pixels.len(), 16, "2x2 RGBA8");

    for y in 0u32..2 {
        for x in 0u32..2 {
            let at = ((y * 2 + x) * 4) as usize;
            let got: [u8; 4] = pixels[at..at + 4].try_into().unwrap();
            let want = [(x * 100 + 10) as u8, (y * 100 + 20) as u8, 255, 200];
            let close = got
                .iter()
                .zip(want.iter())
                .all(|(g, w)| g.abs_diff(*w) <= TOLERANCE);
            assert!(
                close,
                "pixel ({x},{y}): expected RGBA {want:?} (+/-{TOLERANCE}), read back {got:?}"
            );
        }
    }

    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the storage-image dispatch"
    );
}

/// Minecraft dispatches the same panorama UAVs repeatedly. Reusing the exact
/// image shape must avoid a second image/allocation/view/readback creation
/// while preserving the complete upload -> dispatch -> guest-visible
/// readback contract.
#[test]
fn compute_storage_image_allocations_are_reused_across_dispatches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let spirv = kyty_graphics::spirv_asm::assemble(STORE_PATTERN_CS)
        .expect("storage-image compute shader assembles");
    let binding = ShaderStageBinding {
        stage: vk::ShaderStageFlags::COMPUTE,
        descriptor_set_slot: 0,
        push_constant_offset: 0,
        push_constants: Vec::new(),
        push_uniform_binding: None,
        storage_buffers: None,
        textures: None,
        storage_images: Some(StorageImageBinding {
            binding: 0,
            images: vec![StorageImageUpload {
                width: 2,
                height: 2,
                depth: 1,
                layers: 1,
                array: false,
                tile_mode: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                pixels: vec![0xEE; 16].into(),
                guest_base: 0x3368_0000,
            }],
        }),
        gds_binding: None,
        eud_raw: None,
    };
    let state = ComputeState {
        groups: [1, 1, 1],
        spirv: &spirv,
        binding: Some(&binding),
    };

    let before = dev.draw_cache_stats();
    let first = dispatch_compute(dev, &state).expect("first storage-image dispatch");
    let after_first = dev.draw_cache_stats();
    let second = dispatch_compute(dev, &state).expect("second storage-image dispatch");
    let after_second = dev.draw_cache_stats();

    assert_eq!(
        first.images, second.images,
        "reuse cannot change output bytes"
    );
    assert_eq!(
        after_first.compute_image_misses,
        before.compute_image_misses + 1,
        "first shape allocates one persistent UAV"
    );
    assert_eq!(
        after_second.compute_image_hits,
        after_first.compute_image_hits + 1,
        "second dispatch reuses its persistent UAV"
    );
    assert_eq!(
        after_second.compute_image_uploads_skipped,
        after_first.compute_image_uploads_skipped + 1,
        "the same live UAV snapshot should skip a redundant staging upload"
    );
    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors while reusing the compute UAV"
    );
}

#[test]
fn compute_shader_writes_every_2d_array_layer() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");
    let spirv = kyty_graphics::spirv_asm::assemble(STORE_ARRAY_CS)
        .expect("array storage-image compute shader assembles");
    let binding = ShaderStageBinding {
        stage: vk::ShaderStageFlags::COMPUTE,
        descriptor_set_slot: 0,
        push_constant_offset: 0,
        push_constants: Vec::new(),
        push_uniform_binding: None,
        storage_buffers: None,
        textures: None,
        storage_images: Some(StorageImageBinding {
            binding: 0,
            images: vec![StorageImageUpload {
                width: 2,
                height: 2,
                depth: 1,
                layers: 2,
                array: true,
                tile_mode: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                pixels: vec![0xEE; 32].into(),
                guest_base: 0,
            }],
        }),
        gds_binding: None,
        eud_raw: None,
    };
    let outputs = dispatch_compute(
        dev,
        &ComputeState {
            groups: [1, 1, 2],
            spirv: &spirv,
            binding: Some(&binding),
        },
    )
    .expect("array storage-image compute dispatch");
    let pixels = &outputs.images[0];
    assert_eq!(pixels.len(), 32, "two tightly packed 2x2 RGBA8 layers");
    for pixel in pixels[..16].chunks_exact(4) {
        assert_eq!(pixel, [0, 255, 0, 255], "layer 0 is green");
    }
    for pixel in pixels[16..].chunks_exact(4) {
        assert_eq!(pixel, [255, 0, 0, 255], "layer 1 is red");
    }

    // Arrayed is part of the SPIR-V image type, not a consequence of
    // layer_count > 1. Minecraft creates the panorama output incrementally,
    // beginning with a one-layer type-13 descriptor. Exercise that exact
    // boundary so the bind path cannot regress to a TYPE_2D view.
    let one_layer_binding = ShaderStageBinding {
        stage: vk::ShaderStageFlags::COMPUTE,
        descriptor_set_slot: 0,
        push_constant_offset: 0,
        push_constants: Vec::new(),
        push_uniform_binding: None,
        storage_buffers: None,
        textures: None,
        storage_images: Some(StorageImageBinding {
            binding: 0,
            images: vec![StorageImageUpload {
                width: 2,
                height: 2,
                depth: 1,
                layers: 1,
                array: true,
                tile_mode: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                pixels: vec![0xEE; 16].into(),
                guest_base: 0,
            }],
        }),
        gds_binding: None,
        eud_raw: None,
    };
    let one_layer = dispatch_compute(
        dev,
        &ComputeState {
            groups: [1, 1, 1],
            spirv: &spirv,
            binding: Some(&one_layer_binding),
        },
    )
    .expect("one-layer array storage-image compute dispatch");
    for pixel in one_layer.images[0].chunks_exact(4) {
        assert_eq!(pixel, [0, 255, 0, 255], "the sole array layer is green");
    }
    assert_eq!(
        validation_error_count(),
        0,
        "Vulkan validation reported errors during the array-storage dispatch"
    );
}
