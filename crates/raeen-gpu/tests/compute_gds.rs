//! Round-10 compute bind-time walls, on a real Vulkan device.
//!
//! 1. **GDS persistence** — ASTRO.BOT's `ds_append` counters live in GDS and
//!    accumulate ACROSS dispatches (they feed indirect-draw arguments), so the
//!    GDS arena must be one device-persistent buffer, not a per-dispatch
//!    allocation. The test dispatches an atomic-increment shader twice and
//!    asserts the second dispatch observes the first one's counter.
//! 2. **Sampled textures without samplers** — a CS that only texel-fetches
//!    (`OpImageFetch`) binds textures but zero S#s; the descriptor arrays are
//!    independent and the dispatch must not refuse the empty sampler array.
//!
//! Machines without a Vulkan 1.3 device **skip** (like `vulkan_triangle`);
//! `RAEEN_REQUIRE_VULKAN=1` turns the skip into a failure.

use ash::vk;
use raeen_gpu::backend::GpuBackend;
use raeen_gpu::vulkan::compute::{ComputeState, dispatch_compute};
use raeen_gpu::vulkan::offscreen::{
    ShaderStageBinding, StorageBufferBinding, TextureBinding, TextureUpload,
};
use raeen_gpu::vulkan::{VulkanBackend, validation_error_count};

/// `LocalSize 1 1 1`: `out[0] = atomicAdd(gds[0], 1)` — the same `%gds`
/// declaration shape the recompiler emits (StorageBuffer block over a runtime
/// uint array, set 0 at the binding after the storage buffers).
const GDS_COUNTER_CS: &str = "\
    OpCapability Shader\n\
    OpMemoryModel Logical GLSL450\n\
    OpEntryPoint GLCompute %main \"main\" %gds %out\n\
    OpExecutionMode %main LocalSize 1 1 1\n\
    OpDecorate %gds_arr ArrayStride 4\n\
    OpMemberDecorate %GDS 0 Coherent\n\
    OpMemberDecorate %GDS 0 Offset 0\n\
    OpDecorate %GDS Block\n\
    OpDecorate %gds DescriptorSet 0\n\
    OpDecorate %gds Binding 1\n\
    OpDecorate %out_arr ArrayStride 4\n\
    OpMemberDecorate %Out 0 Offset 0\n\
    OpDecorate %Out Block\n\
    OpDecorate %out DescriptorSet 0\n\
    OpDecorate %out Binding 0\n\
    %void = OpTypeVoid\n\
    %fnty = OpTypeFunction %void\n\
    %uint = OpTypeInt 32 0\n\
    %uint_0 = OpConstant %uint 0\n\
    %uint_1 = OpConstant %uint 1\n\
    %gds_arr = OpTypeRuntimeArray %uint\n\
    %GDS = OpTypeStruct %gds_arr\n\
    %ptr_gds = OpTypePointer StorageBuffer %GDS\n\
    %gds = OpVariable %ptr_gds StorageBuffer\n\
    %out_arr = OpTypeRuntimeArray %uint\n\
    %Out = OpTypeStruct %out_arr\n\
    %ptr_out = OpTypePointer StorageBuffer %Out\n\
    %out = OpVariable %ptr_out StorageBuffer\n\
    %ptr_uint = OpTypePointer StorageBuffer %uint\n\
    %main = OpFunction %void None %fnty\n\
    %entry = OpLabel\n\
    %pg = OpAccessChain %ptr_uint %gds %uint_0 %uint_0\n\
    %old = OpAtomicIAdd %uint %pg %uint_1 %uint_0 %uint_1\n\
    %po = OpAccessChain %ptr_uint %out %uint_0 %uint_0\n\
    OpStore %po %old\n\
    OpReturn\n\
    OpFunctionEnd\n";

/// `LocalSize 1 1 1`: `out[0] = uint(texelFetch(tex[0], ivec2(0,0), 0).r *
/// 255 + 0.5)` — a sampled-image array with NO sampler array, the shape a
/// texel-fetch-only CS produces (`textures2d_sampled_num > 0`,
/// `samplers_num == 0`).
const FETCH_NO_SAMPLER_CS: &str = "\
    OpCapability Shader\n\
    OpMemoryModel Logical GLSL450\n\
    OpEntryPoint GLCompute %main \"main\" %tex %out\n\
    OpExecutionMode %main LocalSize 1 1 1\n\
    OpDecorate %tex DescriptorSet 0\n\
    OpDecorate %tex Binding 1\n\
    OpDecorate %out_arr ArrayStride 4\n\
    OpMemberDecorate %Out 0 Offset 0\n\
    OpDecorate %Out Block\n\
    OpDecorate %out DescriptorSet 0\n\
    OpDecorate %out Binding 0\n\
    %void = OpTypeVoid\n\
    %fnty = OpTypeFunction %void\n\
    %uint = OpTypeInt 32 0\n\
    %int = OpTypeInt 32 1\n\
    %float = OpTypeFloat 32\n\
    %v2int = OpTypeVector %int 2\n\
    %v4float = OpTypeVector %float 4\n\
    %img = OpTypeImage %float 2D 0 0 0 1 Unknown\n\
    %uint_0 = OpConstant %uint 0\n\
    %uint_1 = OpConstant %uint 1\n\
    %int_0 = OpConstant %int 0\n\
    %f255 = OpConstant %float 255.000000\n\
    %f05 = OpConstant %float 0.500000\n\
    %arr = OpTypeArray %img %uint_1\n\
    %ptr_arr = OpTypePointer UniformConstant %arr\n\
    %ptr_img = OpTypePointer UniformConstant %img\n\
    %tex = OpVariable %ptr_arr UniformConstant\n\
    %out_arr = OpTypeRuntimeArray %uint\n\
    %Out = OpTypeStruct %out_arr\n\
    %ptr_out = OpTypePointer StorageBuffer %Out\n\
    %out = OpVariable %ptr_out StorageBuffer\n\
    %ptr_uint = OpTypePointer StorageBuffer %uint\n\
    %main = OpFunction %void None %fnty\n\
    %entry = OpLabel\n\
    %pimg = OpAccessChain %ptr_img %tex %uint_0\n\
    %image = OpLoad %img %pimg\n\
    %coord = OpCompositeConstruct %v2int %int_0 %int_0\n\
    %texel = OpImageFetch %v4float %image %coord Lod %int_0\n\
    %r = OpCompositeExtract %float %texel 0\n\
    %scaled = OpFMul %float %r %f255\n\
    %rounded = OpFAdd %float %scaled %f05\n\
    %value = OpConvertFToU %uint %rounded\n\
    %po = OpAccessChain %ptr_uint %out %uint_0 %uint_0\n\
    OpStore %po %value\n\
    OpReturn\n\
    OpFunctionEnd\n";

/// `LocalSize 1 1 1`: `out[0] = uint(texture(sampler2D(tex[0], smp[0]),
/// vec2(0.5)).r * 255 + 0.5)` — the recompiled sample-family shape: SEPARATE
/// sampled-image and sampler descriptor arrays. With an all-zero synthesized
/// S# the host binds the cached default nearest/wrap sampler
/// (`SamplerState::nearest_repeat()`), the default-sampler port.
const SAMPLE_DEFAULT_SAMPLER_CS: &str = "\
    OpCapability Shader\n\
    OpMemoryModel Logical GLSL450\n\
    OpEntryPoint GLCompute %main \"main\" %tex %smp %out\n\
    OpExecutionMode %main LocalSize 1 1 1\n\
    OpDecorate %tex DescriptorSet 0\n\
    OpDecorate %tex Binding 1\n\
    OpDecorate %smp DescriptorSet 0\n\
    OpDecorate %smp Binding 2\n\
    OpDecorate %out_arr ArrayStride 4\n\
    OpMemberDecorate %Out 0 Offset 0\n\
    OpDecorate %Out Block\n\
    OpDecorate %out DescriptorSet 0\n\
    OpDecorate %out Binding 0\n\
    %void = OpTypeVoid\n\
    %fnty = OpTypeFunction %void\n\
    %uint = OpTypeInt 32 0\n\
    %float = OpTypeFloat 32\n\
    %v2float = OpTypeVector %float 2\n\
    %v4float = OpTypeVector %float 4\n\
    %img = OpTypeImage %float 2D 0 0 0 1 Unknown\n\
    %simg = OpTypeSampledImage %img\n\
    %sampler = OpTypeSampler\n\
    %uint_0 = OpConstant %uint 0\n\
    %uint_1 = OpConstant %uint 1\n\
    %f0 = OpConstant %float 0.000000\n\
    %f05 = OpConstant %float 0.500000\n\
    %f255 = OpConstant %float 255.000000\n\
    %arr_img = OpTypeArray %img %uint_1\n\
    %ptr_arr_img = OpTypePointer UniformConstant %arr_img\n\
    %ptr_img = OpTypePointer UniformConstant %img\n\
    %tex = OpVariable %ptr_arr_img UniformConstant\n\
    %arr_smp = OpTypeArray %sampler %uint_1\n\
    %ptr_arr_smp = OpTypePointer UniformConstant %arr_smp\n\
    %ptr_smp = OpTypePointer UniformConstant %sampler\n\
    %smp = OpVariable %ptr_arr_smp UniformConstant\n\
    %out_arr = OpTypeRuntimeArray %uint\n\
    %Out = OpTypeStruct %out_arr\n\
    %ptr_out = OpTypePointer StorageBuffer %Out\n\
    %out = OpVariable %ptr_out StorageBuffer\n\
    %ptr_uint = OpTypePointer StorageBuffer %uint\n\
    %main = OpFunction %void None %fnty\n\
    %entry = OpLabel\n\
    %pimg = OpAccessChain %ptr_img %tex %uint_0\n\
    %image = OpLoad %img %pimg\n\
    %psmp = OpAccessChain %ptr_smp %smp %uint_0\n\
    %samp = OpLoad %sampler %psmp\n\
    %si = OpSampledImage %simg %image %samp\n\
    %coord = OpCompositeConstruct %v2float %f05 %f05\n\
    %texel = OpImageSampleExplicitLod %v4float %si %coord Lod %f0\n\
    %r = OpCompositeExtract %float %texel 0\n\
    %scaled = OpFMul %float %r %f255\n\
    %rounded = OpFAdd %float %scaled %f05\n\
    %value = OpConvertFToU %uint %rounded\n\
    %po = OpAccessChain %ptr_uint %out %uint_0 %uint_0\n\
    OpStore %po %value\n\
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
            eprintln!("compute_gds: running on {name}");
            Some(backend)
        }
        Err(e) => {
            assert!(
                std::env::var_os("RAEEN_REQUIRE_VULKAN").is_none(),
                "RAEEN_REQUIRE_VULKAN is set but Vulkan init failed: {e}"
            );
            eprintln!("compute_gds: SKIP — no usable Vulkan 1.3 device ({e})");
            None
        }
    }
}

/// Two dispatches of `out[0] = atomicAdd(gds[0], 1)`: the first must read the
/// zero-initialized counter (0), the second must observe the first's
/// increment (1) — proving the GDS buffer persists on the device across
/// dispatch boundaries instead of being recreated per dispatch.
#[test]
fn gds_counter_persists_across_dispatches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    let spirv =
        kyty_graphics::spirv_asm::assemble(GDS_COUNTER_CS).expect("GDS compute shader assembles");

    let binding = ShaderStageBinding {
        stage: vk::ShaderStageFlags::COMPUTE,
        descriptor_set_slot: 0,
        push_constant_offset: 0,
        push_constants: Vec::new(),
        push_uniform_binding: None,
        storage_buffers: Some(StorageBufferBinding {
            binding: 0,
            buffers: vec![std::sync::Arc::new(vec![0u8; 4])],
            guest_bases: vec![0x1234_0000],
            guest_sizes: vec![4],
            writable: vec![true],
        }),
        textures: None,
        storage_images: None,
        gds_binding: Some(1),
        eud_raw: None,
        global_mem: None,
    };
    let state = ComputeState {
        groups: [1, 1, 1],
        spirv: &spirv,
        binding: Some(&binding),
    };

    let before = dev.draw_cache_stats();
    let first = dispatch_compute(dev, &state).expect("first GDS dispatch");
    let after_first = dev.draw_cache_stats();
    assert_eq!(
        first.buffers[0].materialize(&[0u8; 4]),
        0u32.to_le_bytes().to_vec(),
        "first dispatch must read the zero-initialized GDS counter"
    );
    let second = dispatch_compute(dev, &state).expect("second GDS dispatch");
    let after_second = dev.draw_cache_stats();
    assert_eq!(
        second.buffers[0].materialize(&[0u8; 4]),
        1u32.to_le_bytes().to_vec(),
        "second dispatch must observe the first dispatch's GDS increment — \
        the arena must persist across dispatches"
    );
    assert_eq!(
        after_first.compute_buffer_misses,
        before.compute_buffer_misses + 1,
        "first guest identity allocates one persistent compute buffer"
    );
    assert_eq!(
        after_second.compute_buffer_hits,
        after_first.compute_buffer_hits + 1,
        "second dispatch reuses the same guest-addressed buffer"
    );
    assert_eq!(
        after_second.compute_buffer_uploads_skipped,
        after_first.compute_buffer_uploads_skipped + 1,
        "an unchanged complete guest snapshot skips the second upload"
    );
    assert_eq!(validation_error_count(), 0, "validation must stay clean");
}

/// A texel-fetch CS binds one sampled texture and ZERO samplers; the dispatch
/// must create only the sampled-image descriptor array (matching the SPIR-V,
/// which declares no sampler binding) instead of refusing the empty sampler
/// array.
#[test]
fn sampled_texture_without_sampler_dispatches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    let spirv = kyty_graphics::spirv_asm::assemble(FETCH_NO_SAMPLER_CS)
        .expect("texel-fetch compute shader assembles");

    let binding = ShaderStageBinding {
        stage: vk::ShaderStageFlags::COMPUTE,
        descriptor_set_slot: 0,
        push_constant_offset: 0,
        push_constants: Vec::new(),
        push_uniform_binding: None,
        storage_buffers: Some(StorageBufferBinding {
            binding: 0,
            buffers: vec![std::sync::Arc::new(vec![0u8; 4])],
            guest_bases: vec![0],
            guest_sizes: vec![4],
            writable: vec![true],
        }),
        textures: Some(TextureBinding {
            sampled_binding: 1,
            // No S# bound: the recompiled SPIR-V declares no %samplers, so
            // this index is never used.
            sampler_binding: 2,
            textures: vec![TextureUpload {
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                pixels: vec![0x40, 0x00, 0x00, 0xFF],
                layers: 1,
                cube: false,
                array: false,
                volume: false,
                depth: 1,
                render_target: None,
                guest_base: 0,
                sample_hash: 0,
                cached: false,
            }],
            samplers: Vec::new(),
            sampled_groups: Vec::new(),
        }),
        storage_images: None,
        gds_binding: None,
        eud_raw: None,
        global_mem: None,
    };
    let outputs = dispatch_compute(
        dev,
        &ComputeState {
            groups: [1, 1, 1],
            spirv: &spirv,
            binding: Some(&binding),
        },
    )
    .expect("texel-fetch dispatch with zero samplers");
    assert_eq!(
        outputs.buffers[0].materialize(&[0u8; 4]),
        0x40u32.to_le_bytes().to_vec(),
        "the fetched red texel must round-trip through the sampled-image array"
    );
    assert_eq!(validation_error_count(), 0, "validation must stay clean");
}

/// Device-loss defusal sub-fix (i), Vulkan side (SharpEmu port —
/// `reference/sharpemu/src/SharpEmu.Libs/VideoOut/VulkanVideoPresenter.cs`
/// L6314-6322 binds an on-the-fly sampler when none was captured; L8121-8156
/// caches one `VkSampler` per S# state, all-zero decoding to nearest/wrap):
/// a sample-family CS with a synthesized all-zero S# binds the cached
/// default nearest/wrap sampler (`SamplerState::nearest_repeat()`). Two dispatches
/// prove creation AND the per-device cache-hit path, with clean validation.
#[test]
fn default_nearest_sampler_binds_and_caches_across_dispatches() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let dev = backend.device().expect("backend is initialized");

    let spirv = kyty_graphics::spirv_asm::assemble(SAMPLE_DEFAULT_SAMPLER_CS)
        .expect("default-sampler compute shader assembles");

    let binding = ShaderStageBinding {
        stage: vk::ShaderStageFlags::COMPUTE,
        descriptor_set_slot: 0,
        push_constant_offset: 0,
        push_constants: Vec::new(),
        push_uniform_binding: None,
        storage_buffers: Some(StorageBufferBinding {
            binding: 0,
            buffers: vec![std::sync::Arc::new(vec![0u8; 4])],
            guest_bases: vec![0],
            guest_sizes: vec![4],
            writable: vec![true],
        }),
        textures: Some(TextureBinding {
            sampled_binding: 1,
            sampler_binding: 2,
            textures: vec![TextureUpload {
                width: 1,
                height: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                pixels: vec![0x40, 0x00, 0x00, 0xFF],
                layers: 1,
                cube: false,
                array: false,
                volume: false,
                depth: 1,
                render_target: None,
                guest_base: 0,
                sample_hash: 0,
                cached: false,
            }],
            // The synthesized all-zero S#: xy_mag_filter == 0 -> nearest.
            samplers: vec![raeen_gpu::vulkan::offscreen::SamplerState::nearest_repeat()],
            sampled_groups: Vec::new(),
        }),
        storage_images: None,
        gds_binding: None,
        eud_raw: None,
        global_mem: None,
    };
    let state = ComputeState {
        groups: [1, 1, 1],
        spirv: &spirv,
        binding: Some(&binding),
    };

    let first = dispatch_compute(dev, &state).expect("first sampled dispatch (sampler creation)");
    assert_eq!(
        first.buffers[0].materialize(&[0u8; 4]),
        0x40u32.to_le_bytes().to_vec(),
        "the sampled red texel must round-trip through the default nearest sampler"
    );
    let second = dispatch_compute(dev, &state).expect("second sampled dispatch (cached sampler)");
    assert_eq!(
        second.buffers[0].materialize(&[0u8; 4]),
        0x40u32.to_le_bytes().to_vec(),
        "the cached default sampler must serve repeat dispatches identically"
    );
    assert_eq!(validation_error_count(), 0, "validation must stay clean");
}
