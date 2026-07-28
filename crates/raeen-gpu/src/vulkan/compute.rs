//! One-shot Vulkan compute dispatch for translated guest shaders.
//!
//! Guest storage buffers are uploaded into host-visible coherent Vulkan
//! buffers, and guest storage images (UAVs) into device-local images via
//! staging buffers. After the queue fence, only buffers whose guest usage is
//! `ReadWrite` plus all storage images are read back; the caller owns copying
//! those bytes back into identity-mapped guest memory.

use super::cache::{
    ComputeBufferKey, ComputeImageKey, ComputeImageWriteback, DrawCaches, PendingDrawResources,
    PersistentComputeImage,
};
use super::instance::VulkanDevice;
use super::offscreen::{ShaderStageBinding, StorageImageUpload, TextureUpload};
use ash::vk::Handle;
use ash::{Device, vk};
use raeen_core::error::GpuError;
use std::sync::Arc;

pub struct ComputeState<'a> {
    pub groups: [u32; 3],
    pub spirv: &'a [u32],
    pub binding: Option<&'a ShaderStageBinding>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchSlice {
    base: [u32; 3],
    groups: [u32; 3],
}

/// Conservative translated-work budget for one Vulkan dispatch command.
///
/// This is not a performance score. It is a TDR containment boundary: a
/// 64x64x1 dispatch of ASTRO.BOT's 8.7K-word, 16x16 shader reproducibly reset
/// the Radeon 760M, while the same program divided into workgroup-base slices
/// stays preemptible. The estimate intentionally includes declaration words;
/// overestimating creates a few extra commands but preserves guest work.
const DISPATCH_TDR_BUDGET: u128 = 1_000_000_000;

fn compute_local_size(spirv: &[u32]) -> [u32; 3] {
    // SPIR-V: five-word header, OpExecutionMode = 16, LocalSize = 17.
    let mut at = 5usize;
    while at < spirv.len() {
        let word_count = (spirv[at] >> 16) as usize;
        let opcode = spirv[at] & 0xffff;
        if word_count == 0 || at.saturating_add(word_count) > spirv.len() {
            break;
        }
        if opcode == 16 && word_count >= 6 && spirv[at + 2] == 17 {
            return [spirv[at + 3], spirv[at + 4], spirv[at + 5]].map(|n| n.max(1));
        }
        at += word_count;
    }
    [1, 1, 1]
}

fn dispatch_slices(state: &ComputeState<'_>) -> Vec<DispatchSlice> {
    let local = compute_local_size(state.spirv);
    let words = state.spirv.len().max(1) as u128;
    let local_invocations = local.into_iter().map(u128::from).product::<u128>().max(1);
    let total_groups = state.groups.into_iter().map(u128::from).product::<u128>();
    let total_work = words
        .saturating_mul(local_invocations)
        .saturating_mul(total_groups);
    if total_work <= DISPATCH_TDR_BUDGET || total_groups == 0 {
        return vec![DispatchSlice {
            base: [0; 3],
            groups: state.groups,
        }];
    }

    // Slice the largest group dimension. The two untouched dimensions stay
    // whole, while vkCmdDispatchBase preserves the original WorkGroupID.
    let axis = (0..3)
        .max_by_key(|&axis| state.groups[axis])
        .unwrap_or_default();
    let axis_groups = state.groups[axis];
    if axis_groups <= 1 {
        return vec![DispatchSlice {
            base: [0; 3],
            groups: state.groups,
        }];
    }
    let other_groups = state
        .groups
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &n)| u128::from(n))
        .product::<u128>()
        .max(1);
    let work_per_axis_group = words
        .saturating_mul(local_invocations)
        .saturating_mul(other_groups)
        .max(1);
    let groups_per_slice = u32::try_from((DISPATCH_TDR_BUDGET / work_per_axis_group).max(1))
        .unwrap_or(u32::MAX)
        .min(axis_groups);

    let mut slices = Vec::new();
    let mut base_axis = 0u32;
    while base_axis < axis_groups {
        let count = groups_per_slice.min(axis_groups - base_axis);
        let mut base = [0; 3];
        let mut groups = state.groups;
        base[axis] = base_axis;
        groups[axis] = count;
        slices.push(DispatchSlice { base, groups });
        base_axis += count;
    }
    slices
}

/// Whether this dispatch must cross queue-submission boundaries to stay below
/// the conservative Windows TDR work budget.
///
/// Heavy dispatches cannot join the deferred frame batch: dividing one command
/// buffer into several `vkCmdDispatchBase` calls does not give the Windows
/// scheduler a fence-completed preemption boundary.
pub(crate) fn compute_requires_slicing(state: &ComputeState<'_>) -> bool {
    dispatch_slices(state).len() > 1
}

/// Post-dispatch device content, in the binding's declaration order.
pub struct ComputeOutputs {
    /// One entry per writable storage buffer, preserving
    /// `StorageBufferBinding.buffers` order after read-only entries are
    /// filtered out.
    pub buffers: Vec<ComputeBufferOutput>,
    /// One entry per storage image (`StorageImageBinding.images` order),
    /// RGBA8 tightly packed rows.
    pub images: Vec<Vec<u8>>,
}

/// One changed byte range in a writable compute storage buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeDirtySpan {
    pub offset: usize,
    pub bytes: Vec<u8>,
}

/// Sparse post-dispatch storage-buffer content.
///
/// The compute path uploaded `initial` before dispatch. Returning only pages
/// that differ avoids allocating/copying/writing a complete multi-megabyte V#
/// when a small workgroup touched a handful of dwords.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeBufferOutput {
    pub size: usize,
    pub dirty: Vec<ComputeDirtySpan>,
}

impl ComputeBufferOutput {
    /// Reconstruct the complete buffer for Vulkan integration tests and
    /// diagnostics. The production guest writeback consumes `dirty` directly.
    #[must_use]
    pub fn materialize(&self, initial: &[u8]) -> Vec<u8> {
        let mut bytes = initial[..initial.len().min(self.size)].to_vec();
        bytes.resize(self.size, 0);
        for span in &self.dirty {
            let end = span
                .offset
                .saturating_add(span.bytes.len())
                .min(bytes.len());
            if span.offset < end {
                bytes[span.offset..end].copy_from_slice(&span.bytes[..end - span.offset]);
            }
        }
        bytes
    }
}

/// Choose the Vulkan inline push range for a translated resource table.
///
/// A spill binding means the shader declares the table as a descriptor-backed
/// storage buffer. Descriptor creation uploads those bytes, so the pipeline
/// must not also declare an invalid over-cap push range.
fn inline_push_constant_range(
    offset: u32,
    size: usize,
    spill_binding: Option<u32>,
    cap: u32,
) -> Result<Option<(u32, u32)>, GpuError> {
    if size == 0 || spill_binding.is_some() {
        return Ok(None);
    }
    let size = u32::try_from(size).map_err(|_| {
        GpuError::PipelineCreationFailed("push-constant resource table exceeds u32".to_owned())
    })?;
    let need = offset.checked_add(size).ok_or_else(|| {
        GpuError::PipelineCreationFailed("push-constant range overflow".to_owned())
    })?;
    if need > cap {
        return Err(GpuError::PipelineCreationFailed(format!(
            "push constants {need} B exceed the device maxPushConstantsSize {cap} B \
             and the translated shader declares no spill SSBO"
        )));
    }
    Ok(Some((offset, size)))
}

pub fn dispatch_compute(
    dev: &VulkanDevice,
    state: &ComputeState<'_>,
) -> Result<ComputeOutputs, GpuError> {
    if state.spirv.is_empty() {
        return Err(GpuError::ShaderCompilationFailed(
            "compute SPIR-V must be non-empty".to_owned(),
        ));
    }
    if let Some(binding) = state.binding
        && binding.stage != vk::ShaderStageFlags::COMPUTE
    {
        return Err(GpuError::PipelineCreationFailed(format!(
            "compute binding has non-compute stage {:?}",
            binding.stage
        )));
    }

    // A synchronous dispatch (including every TDR-sliced dispatch) cannot
    // reuse persistent resources from an open deferred batch until that batch
    // has actually executed. Cache insertion happens while recording, so a
    // cache hit alone does not prove the image has left UNDEFINED or the
    // buffer contains its preceding writer. Submit/fence/publish the ordered
    // batch before acquiring the cache for this dispatch. The empty target
    // filter avoids unrelated colour-target readback.
    let open_batch = {
        let caches = dev.draw_caches();
        caches.batch_open()
    };
    if open_batch {
        super::offscreen::flush_deferred_draws_filtered(dev, Some(&[]))?;
    }

    let timing = crate::diagnostics::gpu_env().time_compute;
    let total_at = timing.then(std::time::Instant::now);
    // Same locking contract as `render_draw`: the cache lock spans the whole
    // synchronous dispatch, including the fence wait, so the cached pipeline,
    // command buffer, fence, and descriptor pool are reused soundly.
    let mut caches = dev.draw_caches();
    let mut resources = ComputeResources::new(dev, &mut caches);
    let phase_at = timing.then(std::time::Instant::now);
    resources.build(state)?;
    let build = phase_at.map_or(std::time::Duration::ZERO, |at| at.elapsed());
    let phase_at = timing.then(std::time::Instant::now);
    resources.record_and_submit(state)?;
    let submit_wait = phase_at.map_or(std::time::Duration::ZERO, |at| at.elapsed());
    let phase_at = timing.then(std::time::Instant::now);
    let outputs = ComputeOutputs {
        buffers: resources.read_storage(
            state
                .binding
                .and_then(|binding| binding.storage_buffers.as_ref()),
        )?,
        images: resources.read_images()?,
    };
    let map_copy = phase_at.map_or(std::time::Duration::ZERO, |at| at.elapsed());
    let storage_bytes = state
        .binding
        .and_then(|binding| binding.storage_buffers.as_ref())
        .map_or(0usize, |storage| {
            storage.buffers.iter().map(|bytes| bytes.len()).sum()
        });
    let image_bytes = state
        .binding
        .and_then(|binding| binding.storage_images.as_ref())
        .map_or(0usize, |images| {
            images.images.iter().map(|image| image.pixels.len()).sum()
        });
    let sampled_bytes = state
        .binding
        .and_then(|binding| binding.textures.as_ref())
        .map_or(0usize, |textures| {
            textures
                .textures
                .iter()
                .map(|texture| texture.pixels.len())
                .sum()
        });
    let image_count = resources.images.len();
    let sampled_count = resources.sampled.len();
    let dirty_storage_bytes: usize = outputs
        .buffers
        .iter()
        .flat_map(|output| &output.dirty)
        .map(|span| span.bytes.len())
        .sum();
    let dirty_storage_spans: usize = outputs
        .buffers
        .iter()
        .map(|output| output.dirty.len())
        .sum();
    let phase_at = timing.then(std::time::Instant::now);
    drop(resources);
    let retire = phase_at.map_or(std::time::Duration::ZERO, |at| at.elapsed());
    if let Some(total_at) = total_at {
        use std::sync::atomic::{AtomicU64, Ordering};
        static DISPATCHES: AtomicU64 = AtomicU64::new(0);
        static SLOW: AtomicU64 = AtomicU64::new(0);
        let n = DISPATCHES.fetch_add(1, Ordering::Relaxed) + 1;
        let total = total_at.elapsed();
        let slow_n = (total >= std::time::Duration::from_millis(10))
            .then(|| SLOW.fetch_add(1, Ordering::Relaxed) + 1);
        if n.is_multiple_of(512) || slow_n.is_some_and(|slow| slow <= 32 || slow.is_power_of_two())
        {
            tracing::warn!(
                dispatch = n,
                slow_dispatch = slow_n,
                groups = format_args!(
                    "{}x{}x{}",
                    state.groups[0], state.groups[1], state.groups[2]
                ),
                storage_bytes,
                dirty_storage_bytes,
                dirty_storage_spans,
                image_count,
                image_bytes,
                sampled_count,
                sampled_bytes,
                build_us = build.as_micros(),
                submit_wait_us = submit_wait.as_micros(),
                map_copy_us = map_copy.as_micros(),
                retire_us = retire.as_micros(),
                total_us = total.as_micros(),
                "TIME_COMPUTE: synchronous dispatch phase split"
            );
        }
    }
    Ok(outputs)
}

/// Submit a guest-addressed compute packet without a per-dispatch fence.
///
/// Command buffers and descriptor/upload resources join the shared deferred
/// batch. Persistent guest-addressed SSBOs preserve visibility between queued
/// dispatches; the next flip/submission flush fences them once.
pub fn dispatch_compute_deferred(
    dev: &VulkanDevice,
    state: &ComputeState<'_>,
) -> Result<(), GpuError> {
    if compute_requires_slicing(state) {
        return Err(GpuError::PipelineCreationFailed(
            "TDR-sliced compute dispatch cannot join the deferred submission batch".to_owned(),
        ));
    }
    let timing = crate::diagnostics::gpu_env().time_compute;
    let total_at = timing.then(std::time::Instant::now);
    let binding = state.binding.ok_or_else(|| {
        GpuError::PipelineCreationFailed(
            "deferred compute requires an explicit resource binding".to_owned(),
        )
    })?;
    let storage = binding.storage_buffers.as_ref();
    let images = binding.storage_images.as_ref();
    if storage.is_none() && images.is_none() {
        return Err(GpuError::PipelineCreationFailed(
            "deferred compute requires guest-addressed storage buffers or images".to_owned(),
        ));
    }
    if storage.is_some_and(|storage| storage.guest_bases.contains(&0))
        || images.is_some_and(|images| images.images.iter().any(|image| image.guest_base == 0))
    {
        return Err(GpuError::PipelineCreationFailed(
            "deferred compute cannot retain synthetic/null guest resources".to_owned(),
        ));
    }

    let mut caches = dev.draw_caches();
    let mut resources = ComputeResources::new(dev, &mut caches);
    resources.batched = true;
    let build_at = timing.then(std::time::Instant::now);
    resources.build(state)?;
    let build = build_at.map_or(std::time::Duration::ZERO, |at| at.elapsed());
    let record_at = timing.then(std::time::Instant::now);
    resources.record_and_submit(state)?;
    let record = record_at.map_or(std::time::Duration::ZERO, |at| at.elapsed());
    let commit_at = timing.then(std::time::Instant::now);
    resources.commit_to_batch()?;
    let commit = commit_at.map_or(std::time::Duration::ZERO, |at| at.elapsed());
    drop(resources);
    if let Some(total_at) = total_at {
        use std::sync::atomic::{AtomicU64, Ordering};
        static DEFERRED_DISPATCHES: AtomicU64 = AtomicU64::new(0);
        let dispatch = DEFERRED_DISPATCHES.fetch_add(1, Ordering::Relaxed) + 1;
        if dispatch.is_multiple_of(512) {
            tracing::warn!(
                dispatch,
                groups = format_args!(
                    "{}x{}x{}",
                    state.groups[0], state.groups[1], state.groups[2]
                ),
                build_us = build.as_micros(),
                record_us = record.as_micros(),
                commit_us = commit.as_micros(),
                total_us = total_at.elapsed().as_micros(),
                compute_buffer_hits = caches.stats.compute_buffer_hits,
                compute_buffer_misses = caches.stats.compute_buffer_misses,
                compute_buffer_uploads_skipped = caches.stats.compute_buffer_uploads_skipped,
                compute_image_hits = caches.stats.compute_image_hits,
                compute_image_misses = caches.stats.compute_image_misses,
                compute_image_uploads_skipped = caches.stats.compute_image_uploads_skipped,
                "TIME_COMPUTE: deferred dispatch phase split"
            );
        }
    }
    Ok(())
}

struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
    persistent_key: Option<ComputeBufferKey>,
}

/// One storage image plus its upload staging and readback buffers.
struct ImageAllocation {
    key: ComputeImageKey,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    width: u32,
    height: u32,
    /// Volume depth (1 for a 2D UAV).
    depth: u32,
    /// Array layers (1 for a 2D/3D UAV).
    layers: u32,
    /// Bytes per texel (4 = RGBA8, 8 = RGBA16F).
    texel: u32,
    /// The image/view/allocation/readback pair lives in `DrawCaches`; this
    /// dispatch owns only its staging upload.
    persistent: bool,
    /// A freshly-created cache entry starts in UNDEFINED. Cache hits rest in
    /// TRANSFER_SRC_OPTIMAL after the preceding dispatch's readback copy.
    fresh: bool,
    /// Whether this dispatch must seed the image from `staging_buffer`.
    /// False for a repeated descriptor in one PM4 submission: the persistent
    /// image already holds the preceding ordered dispatch's newer result.
    upload_seed: bool,
    /// Duplicate descriptor slots may alias the same guest UAV. They still
    /// need a descriptor entry, but only the first slot records layout
    /// transitions and one readback copy for the shared Vulkan image.
    records_commands: bool,
}

/// One sampled (read-only) texture: staging buffer + device-local image +
/// view. Uploaded once before the dispatch; never read back.
struct SampledAllocation {
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    width: u32,
    height: u32,
    depth: u32,
    layers: u32,
}

/// Per-dispatch resources (owned, destroyed in `Drop`) plus handles borrowed
/// from [`DrawCaches`] — shader module, descriptor layout, descriptor pool,
/// pipeline layout, pipeline, samplers, command buffer, and fence are cached
/// on the device and must NOT be destroyed here.
struct ComputeResources<'a> {
    dev: &'a VulkanDevice,
    caches: &'a mut DrawCaches,
    storage: Vec<BufferAllocation>,
    images: Vec<ImageAllocation>,
    sampled: Vec<SampledAllocation>,
    samplers: Vec<vk::Sampler>,
    /// The persistent GDS arena (cache-owned, NOT destroyed here); null when
    /// the dispatch binds no GDS.
    gds: vk::Buffer,
    /// The raw EUD-window snapshot (SharpEmu port); per-dispatch, owned,
    /// never read back.
    eud_raw: Option<BufferAllocation>,
    /// Oversized translated resource table, uploaded as a UBO instead of
    /// exceeding the device's push-constant limit.
    push_uniform: Option<BufferAllocation>,
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    batched: bool,
    deferred_write_keys: Vec<ComputeBufferKey>,
    deferred_image_writes: Vec<ComputeImageWriteback>,
}

impl<'a> ComputeResources<'a> {
    fn new(dev: &'a VulkanDevice, caches: &'a mut DrawCaches) -> Self {
        Self {
            dev,
            caches,
            storage: Vec::new(),
            images: Vec::new(),
            sampled: Vec::new(),
            samplers: Vec::new(),
            gds: vk::Buffer::null(),
            eud_raw: None,
            push_uniform: None,
            shader: vk::ShaderModule::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            batched: false,
            deferred_write_keys: Vec::new(),
            deferred_image_writes: Vec::new(),
        }
    }

    fn device(&self) -> &Device {
        self.dev.device()
    }

    fn build(&mut self, state: &ComputeState<'_>) -> Result<(), GpuError> {
        // Cached by SPIR-V content: a title re-dispatches the same compute
        // shaders thousands of times per frame.
        self.shader = self.caches.shader_module(self.dev, state.spirv)?;

        let storage = state.binding.and_then(|b| b.storage_buffers.as_ref());
        let storage_images = state.binding.and_then(|b| b.storage_images.as_ref());
        let textures = state.binding.and_then(|b| b.textures.as_ref());
        let gds_binding = state.binding.and_then(|b| b.gds_binding);
        let eud_raw = state.binding.and_then(|b| b.eud_raw.as_ref());
        let push_uniform_binding = state.binding.and_then(|b| b.push_uniform_binding);
        if storage.is_some()
            || storage_images.is_some()
            || textures.is_some()
            || gds_binding.is_some()
            || eud_raw.is_some()
            || push_uniform_binding.is_some()
        {
            let binding = state.binding.expect("resource groups come from a binding");
            if binding.descriptor_set_slot != 0 {
                return Err(GpuError::PipelineCreationFailed(format!(
                    "compute descriptor set slot {} is not supported yet",
                    binding.descriptor_set_slot
                )));
            }
            let mut layout_bindings = Vec::new();
            if let Some(storage) = storage {
                if storage.buffers.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "compute storage descriptor array is empty".to_owned(),
                    ));
                }
                if storage.guest_bases.len() != storage.buffers.len()
                    || storage.guest_sizes.len() != storage.buffers.len()
                    || storage.writable.len() != storage.buffers.len()
                {
                    return Err(GpuError::PipelineCreationFailed(format!(
                        "compute storage buffers ({}) / guest bases ({}) / guest sizes ({}) / \
                         writeback flags ({}) must have identical counts",
                        storage.buffers.len(),
                        storage.guest_bases.len(),
                        storage.guest_sizes.len(),
                        storage.writable.len()
                    )));
                }
                for (index, ((bytes, &base), &guest_size)) in storage
                    .buffers
                    .iter()
                    .zip(&storage.guest_bases)
                    .zip(&storage.guest_sizes)
                    .enumerate()
                {
                    let allocation =
                        self.create_storage_buffer(bytes, base, guest_size, Some(bytes))?;
                    if self.batched
                        && storage.writable[index]
                        && let Some(key) = allocation.persistent_key
                    {
                        self.deferred_write_keys.push(key);
                    }
                    self.storage.push(allocation);
                }
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(storage.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(self.storage.len() as u32)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            if let Some(images) = storage_images {
                if images.images.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "compute storage-image descriptor array is empty".to_owned(),
                    ));
                }
                for upload in &images.images {
                    let records_commands = self.create_storage_image(upload)?;
                    if self.batched && records_commands {
                        self.deferred_image_writes.push(ComputeImageWriteback {
                            key: ComputeImageKey {
                                base: upload.guest_base,
                                width: upload.width,
                                height: upload.height,
                                depth: upload.depth.max(1),
                                layers: upload.layers.max(1),
                                array: upload.array,
                                volume: upload.volume,
                                format: upload.format.as_raw(),
                            },
                            tile_mode: upload.tile_mode,
                            texel: upload.texel_bytes(),
                        });
                    }
                }
                if images.groups.is_empty() {
                    // Homogeneous: one array of every storage view.
                    layout_bindings.push(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(images.binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(self.images.len() as u32)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    );
                } else {
                    // Mixed (Dim, format): one `%textures2D_L<key>` array per
                    // key, at its own binding — matching the recompiled
                    // SPIR-V.
                    for group in &images.groups {
                        layout_bindings.push(
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(group.binding)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(group.view_indices.len() as u32)
                                .stage_flags(vk::ShaderStageFlags::COMPUTE),
                        );
                    }
                }
            }
            // Sampled textures + samplers (the recompiled SPIR-V declares
            // %textures2D_S and %samplers as separate bindings) — first
            // consumed by ASTRO.BOT's froxel/LUT-volume compute shaders.
            //
            // The two arrays are INDEPENDENT: the SPIR-V declares
            // `%textures2D_S` only when `textures2d_sampled_num > 0` and
            // `%samplers` only when `samplers_num > 0`, and a CS legitimately
            // carries one without the other (texel-fetch/image-load shaders
            // bind textures but zero samplers). Each descriptor array is
            // created only when non-empty, mirroring the SPIR-V exactly.
            if let Some(textures) = textures {
                if textures.textures.is_empty() && textures.samplers.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "compute sampled-texture and sampler descriptor arrays are both empty"
                            .to_owned(),
                    ));
                }
                if !textures.textures.is_empty() {
                    for upload in &textures.textures {
                        self.create_sampled_image(upload)?;
                    }
                    if textures.sampled_groups.is_empty() {
                        // Homogeneous: one array of every sampled view.
                        layout_bindings.push(
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(textures.sampled_binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(self.sampled.len() as u32)
                                .stage_flags(vk::ShaderStageFlags::COMPUTE),
                        );
                    } else {
                        // Mixed-dim: one `%textures2D_S<dim>` array per Dim, at
                        // its own binding — matching the recompiled SPIR-V.
                        for group in &textures.sampled_groups {
                            layout_bindings.push(
                                vk::DescriptorSetLayoutBinding::default()
                                    .binding(group.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .descriptor_count(group.view_indices.len() as u32)
                                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                            );
                        }
                    }
                }
                if !textures.samplers.is_empty() {
                    for &sampler_state in &textures.samplers {
                        self.samplers
                            .push(self.caches.sampler(self.dev, sampler_state)?);
                    }
                    layout_bindings.push(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(textures.sampler_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(self.samplers.len() as u32)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    );
                }
            }
            // The persistent GDS arena: one storage buffer whose contents
            // persist across dispatches (device lifetime). Bound at the
            // binding index `shader_calc_binding_indices` assigned to
            // `%gds`; never read back (GDS is on-chip, not guest memory).
            if let Some(binding) = gds_binding {
                self.gds = self.caches.gds_buffer(self.dev)?;
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            // The raw EUD-window snapshot (SharpEmu port): one per-dispatch
            // SSBO at the binding index the recompiled `%eud_raw` declares.
            // Uploaded once, never read back (the shader only s_loads it).
            if let Some(window) = eud_raw {
                if window.bytes.len() < 4 || !window.bytes.len().is_multiple_of(4) {
                    return Err(GpuError::PipelineCreationFailed(format!(
                        "raw EUD-window snapshot carries {} bytes — must be a non-zero \
                         dword multiple",
                        window.bytes.len()
                    )));
                }
                self.eud_raw =
                    Some(self.create_storage_buffer(&window.bytes, 0, window.bytes.len(), None)?);
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(window.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            if let Some(uniform_binding) = push_uniform_binding {
                if binding.push_constants.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "push-uniform binding has an empty resource table".to_owned(),
                    ));
                }
                self.push_uniform = Some(self.create_storage_buffer(
                    &binding.push_constants,
                    0,
                    binding.push_constants.len(),
                    None,
                )?);
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(uniform_binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }

            // Cached by binding signature, shared with the graphics path.
            self.descriptor_layout = self.caches.set_layout(self.dev, &layout_bindings)?;

            let pool_sizes: Vec<_> = [
                (
                    vk::DescriptorType::STORAGE_BUFFER,
                    self.storage.len() as u32
                        + u32::from(!self.gds.is_null())
                        + u32::from(self.eud_raw.is_some())
                        + u32::from(self.push_uniform.is_some()),
                ),
                (vk::DescriptorType::STORAGE_IMAGE, self.images.len() as u32),
                (vk::DescriptorType::SAMPLED_IMAGE, self.sampled.len() as u32),
                (vk::DescriptorType::SAMPLER, self.samplers.len() as u32),
            ]
            .into_iter()
            .filter(|&(_, count)| count != 0)
            .map(|(ty, count)| {
                vk::DescriptorPoolSize::default()
                    .ty(ty)
                    .descriptor_count(count)
            })
            .collect();
            self.descriptor_pool = if self.batched {
                self.caches
                    .batch_descriptor_pool(self.dev, 1, &pool_sizes)?
            } else {
                // The persistent pool is safe only for synchronous dispatches;
                // its acquisition resets every previously allocated set.
                self.caches.descriptor_pool(self.dev, 1, &pool_sizes)?
            };
            let layouts = [self.descriptor_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);
            // SAFETY: pool and layout are live handles from this device.
            self.descriptor_set = unsafe { self.device().allocate_descriptor_sets(&alloc_info) }
                .map_err(|e| {
                    GpuError::PipelineCreationFailed(format!("vkAllocateDescriptorSets: {e}"))
                })?[0];

            let buffer_infos: Vec<_> = self
                .storage
                .iter()
                .map(|allocation| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(allocation.buffer)
                        .range(allocation.size as u64)
                })
                .collect();
            let image_infos: Vec<_> = self
                .images
                .iter()
                .map(|allocation| {
                    vk::DescriptorImageInfo::default()
                        .image_view(allocation.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                })
                .collect();
            let mut writes = Vec::new();
            if let Some(storage) = storage {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(storage.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buffer_infos),
                );
            }
            // Mixed (Dim, format): split the storage-view pool into one
            // descriptor array per key, in SPIR-V array order.
            // `self.images[i]` corresponds to `images.images[i]`, so a
            // group's `view_indices` select its views directly. Kept alive
            // alongside `image_infos`.
            let storage_group_infos: Vec<Vec<vk::DescriptorImageInfo>> = storage_images
                .map(|imgs| {
                    imgs.groups
                        .iter()
                        .map(|group| group.view_indices.iter().map(|&i| image_infos[i]).collect())
                        .collect()
                })
                .unwrap_or_default();
            if let Some(images) = storage_images {
                if images.groups.is_empty() {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_set)
                            .dst_binding(images.binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&image_infos),
                    );
                } else {
                    for (group, infos) in images.groups.iter().zip(&storage_group_infos) {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(self.descriptor_set)
                                .dst_binding(group.binding)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                .image_info(infos),
                        );
                    }
                }
            }
            let sampled_infos: Vec<_> = self
                .sampled
                .iter()
                .map(|allocation| {
                    vk::DescriptorImageInfo::default()
                        .image_view(allocation.view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();
            let sampler_infos: Vec<_> = self
                .samplers
                .iter()
                .map(|&sampler| vk::DescriptorImageInfo::default().sampler(sampler))
                .collect();
            // Mixed-dim: split the sampled-view pool into one descriptor array
            // per Dim, in SPIR-V array order. `self.sampled[i]` corresponds to
            // `textures.textures[i]`, so a group's `view_indices` select its
            // views directly. Kept alive alongside `sampled_infos`.
            let group_infos: Vec<Vec<vk::DescriptorImageInfo>> = textures
                .map(|t| {
                    t.sampled_groups
                        .iter()
                        .map(|group| {
                            group
                                .view_indices
                                .iter()
                                .map(|&i| sampled_infos[i])
                                .collect()
                        })
                        .collect()
                })
                .unwrap_or_default();
            if let Some(textures) = textures {
                if !sampled_infos.is_empty() {
                    if textures.sampled_groups.is_empty() {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(self.descriptor_set)
                                .dst_binding(textures.sampled_binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(&sampled_infos),
                        );
                    } else {
                        for (group, infos) in textures.sampled_groups.iter().zip(&group_infos) {
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(self.descriptor_set)
                                    .dst_binding(group.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(infos),
                            );
                        }
                    }
                }
                if !sampler_infos.is_empty() {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_set)
                            .dst_binding(textures.sampler_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&sampler_infos),
                    );
                }
            }
            let gds_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.gds)
                .range(super::cache::GDS_SIZE as u64)];
            if let Some(binding) = gds_binding {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&gds_info),
                );
            }
            let eud_raw_info = self
                .eud_raw
                .as_ref()
                .map(|allocation| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(allocation.buffer)
                        .range(allocation.size as u64)]
                })
                .unwrap_or_default();
            if let Some(window) = eud_raw {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(window.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&eud_raw_info),
                );
            }
            // The push-constant spill SSBO: created above when the kernel
            // declares a spill binding, so `push_uniform_binding` being set implies
            // `self.push_uniform` is populated (single-element info).
            let push_uniform_info = self
                .push_uniform
                .as_ref()
                .map(|allocation| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(allocation.buffer)
                        .range(allocation.size as u64)]
                })
                .unwrap_or_default();
            if let Some(uniform_binding) = push_uniform_binding {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(uniform_binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&push_uniform_info),
                );
            }
            // SAFETY: descriptor set and every named buffer/view are live.
            unsafe { self.device().update_descriptor_sets(&writes, &[]) };
        }

        let set_layouts: Vec<_> = (!self.descriptor_layout.is_null())
            .then_some(self.descriptor_layout)
            .into_iter()
            .collect();
        // Exceeding maxPushConstantsSize is invalid usage — UB without
        // validation layers (measured: AMD driver access violation). Refuse
        // the dispatch by name until the SSBO spill path exists; iGPUs
        // commonly report the 256-byte spec minimum.
        let push_ranges: Vec<_> = state
            .binding
            .map(|binding| {
                inline_push_constant_range(
                    binding.push_constant_offset,
                    binding.push_constants.len(),
                    binding.push_uniform_binding,
                    self.dev.max_push_constants_size(),
                )
            })
            .transpose()?
            .flatten()
            .map(|(offset, size)| {
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(offset)
                    .size(size)
            })
            .into_iter()
            .collect();
        self.pipeline_layout = self
            .caches
            .pipeline_layout(self.dev, &set_layouts, &push_ranges)?;

        // Cached by (canonical module, canonical layout): identical dispatch
        // programs reuse one VkPipeline instead of recompiling per dispatch.
        self.pipeline =
            self.caches
                .compute_pipeline(self.dev, self.shader, self.pipeline_layout)?;

        if self.batched {
            self.command_buffer = self.caches.batch_command_buffer(self.dev)?;
        } else {
            let (command_buffer, fence) = self.caches.submit_resources(self.dev)?;
            self.command_buffer = command_buffer;
            self.fence = fence;
        }
        Ok(())
    }

    /// Host-visible coherent buffer, optionally filled with `fill`.
    fn create_host_buffer(
        &mut self,
        size: usize,
        usage: vk::BufferUsageFlags,
        fill: Option<&[u8]>,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        if size == 0 {
            return Err(GpuError::VulkanInitFailed(
                "zero-sized compute host buffer".to_owned(),
            ));
        }
        debug_assert!(
            super::cache::HOST_POOL_USAGE.contains(usage),
            "compute host-buffer usage must be covered by the shared pool"
        );
        // The draw and compute paths share one size-classed host-visible
        // upload/readback pool. Every pooled buffer carries the union of the
        // usages above, so a staging allocation can later serve storage or
        // transfer readback without recreation.
        let (buffer, memory, mapped) = self.caches.acquire_host_buffer(self.dev, size as u64)?;
        if let Some(bytes) = fill {
            debug_assert_eq!(bytes.len(), size);
            // SAFETY: pooled allocations stay coherently mapped for their
            // lifetime, cover at least `size`, and are fence-idle at checkout.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
            }
        }
        Ok((buffer, memory))
    }

    fn create_storage_buffer(
        &mut self,
        bytes: &[u8],
        guest_base: u64,
        guest_size: usize,
        snapshot: Option<&Arc<Vec<u8>>>,
    ) -> Result<BufferAllocation, GpuError> {
        if bytes.is_empty() {
            return Err(GpuError::VulkanInitFailed(
                "zero-sized compute storage buffer".to_owned(),
            ));
        }
        let persistent_key = (guest_base != 0).then_some(ComputeBufferKey {
            base: guest_base,
            size: bytes.len(),
        });
        let (buffer, memory) = if let Some(key) = persistent_key {
            let snapshot = snapshot.expect("guest-addressed buffers have submission snapshots");
            self.caches
                .acquire_compute_buffer(self.dev, key, snapshot, guest_size)?
        } else {
            self.create_host_buffer(
                bytes.len(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                Some(bytes),
            )?
        };
        Ok(BufferAllocation {
            buffer,
            memory,
            size: bytes.len(),
            persistent_key,
        })
    }

    /// One UAV: staging buffer + device-local image + view + readback
    /// buffer, in the upload's own format; `depth > 1` builds a
    /// `VK_IMAGE_TYPE_3D` volume (measured: ASTRO.BOT's 240x135x64 RGBA16F
    /// UAVs), while `array` builds a `TYPE_2D_ARRAY` view even for one layer.
    /// Pushed with null handles up front so `Drop` cleans up any
    /// partially-built entry on the error paths.
    fn create_storage_image(&mut self, upload: &StorageImageUpload) -> Result<bool, GpuError> {
        let depth = upload.depth.max(1);
        let layers = upload.layers.max(1);
        let texel = upload.texel_bytes();
        let size = (upload.width as usize)
            * (upload.height as usize)
            * (depth as usize)
            * (layers as usize)
            * (texel as usize);
        if size == 0 || upload.pixels.len() != size {
            return Err(GpuError::VulkanInitFailed(format!(
                "storage image {}x{}x{depth} ({texel} B/texel) carries {} initial bytes \
                 (want {size})",
                upload.width,
                upload.height,
                upload.pixels.len()
            )));
        }
        let key = ComputeImageKey {
            base: upload.guest_base,
            width: upload.width,
            height: upload.height,
            depth,
            layers,
            array: upload.array,
            volume: upload.volume,
            format: upload.format.as_raw(),
        };
        if let Some(owner) = self.images.iter().find(|allocation| allocation.key == key) {
            self.images.push(ImageAllocation {
                key,
                staging_buffer: vk::Buffer::null(),
                staging_memory: vk::DeviceMemory::null(),
                image: owner.image,
                memory: owner.memory,
                view: owner.view,
                readback_buffer: owner.readback_buffer,
                readback_memory: owner.readback_memory,
                width: owner.width,
                height: owner.height,
                depth: owner.depth,
                layers: owner.layers,
                texel: owner.texel,
                persistent: true,
                fresh: false,
                upload_seed: false,
                records_commands: false,
            });
            return Ok(false);
        }
        if let Some((cached, upload_seed)) = self.caches.compute_image_entry(&key, &upload.pixels) {
            let (staging_buffer, staging_memory) = if upload_seed {
                self.create_host_buffer(
                    size,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    Some(&upload.pixels),
                )?
            } else {
                (vk::Buffer::null(), vk::DeviceMemory::null())
            };
            self.images.push(ImageAllocation {
                key,
                staging_buffer,
                staging_memory,
                image: cached.image,
                memory: cached.memory,
                view: cached.view,
                readback_buffer: cached.readback_buffer,
                readback_memory: cached.readback_memory,
                width: upload.width,
                height: upload.height,
                depth,
                layers,
                texel,
                persistent: true,
                fresh: false,
                upload_seed,
                records_commands: true,
            });
            return Ok(true);
        }
        self.images.push(ImageAllocation {
            key,
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            readback_buffer: vk::Buffer::null(),
            readback_memory: vk::DeviceMemory::null(),
            width: upload.width,
            height: upload.height,
            depth,
            layers,
            texel,
            persistent: false,
            fresh: true,
            upload_seed: true,
            records_commands: true,
        });
        let slot = self.images.len() - 1;

        let (staging_buffer, staging_memory) = self.create_host_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some(&upload.pixels),
        )?;
        self.images[slot].staging_buffer = staging_buffer;
        self.images[slot].staging_memory = staging_memory;

        let info = vk::ImageCreateInfo::default()
            // Type-driven, not `depth > 1`: see `StorageImageUpload::volume`.
            .image_type(if upload.volume {
                vk::ImageType::TYPE_3D
            } else {
                vk::ImageType::TYPE_2D
            })
            .format(upload.format)
            .extent(vk::Extent3D {
                width: upload.width,
                height: upload.height,
                depth,
            })
            .mip_levels(1)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `info` is fully initialized and borrows nothing beyond this
        // call; the device is live.
        let image = unsafe { self.device().create_image(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("storage vkCreateImage: {e}")))?;
        self.images[slot].image = image;

        // SAFETY: `image` was just created from this device.
        let reqs = unsafe { self.device().get_image_memory_requirements(image) };
        let type_index = self
            .dev
            .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);
        // SAFETY: allocation size/type come from this image's own requirements.
        let memory = unsafe { self.device().allocate_memory(&alloc, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("storage image allocation: {e}")))?;
        self.images[slot].memory = memory;

        // SAFETY: memory was allocated for exactly this image; offset 0 is
        // within it and satisfies the alignment requirement by construction.
        unsafe { self.device().bind_image_memory(image, memory, 0) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("storage image bind memory: {e}")))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(if upload.volume {
                vk::ImageViewType::TYPE_3D
            } else if upload.array {
                vk::ImageViewType::TYPE_2D_ARRAY
            } else {
                vk::ImageViewType::TYPE_2D
            })
            .format(upload.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: layers,
            });
        // SAFETY: the view's image is live and its format/range match the
        // image's creation parameters.
        let view = unsafe { self.device().create_image_view(&view_info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("storage image view: {e}")))?;
        self.images[slot].view = view;

        let (readback_buffer, readback_memory) =
            self.create_host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST, None)?;
        self.images[slot].readback_buffer = readback_buffer;
        self.images[slot].readback_memory = readback_memory;
        self.caches.insert_compute_image(
            self.dev,
            key,
            PersistentComputeImage {
                image: self.images[slot].image,
                memory: self.images[slot].memory,
                view: self.images[slot].view,
                readback_buffer: self.images[slot].readback_buffer,
                readback_memory: self.images[slot].readback_memory,
                bytes: size as u64,
                last_use: 0,
                last_snapshot: Arc::downgrade(&upload.pixels),
            },
        );
        self.images[slot].persistent = true;
        Ok(true)
    }

    /// One sampled texture: staging buffer + device-local image + view, in
    /// the upload's own decoded format. `depth > 1` builds a
    /// `VK_IMAGE_TYPE_3D` volume with a `3D` view (measured: ASTRO.BOT's
    /// 240x135x64 froxel/LUT volumes); `cube` a CUBE view over 6 layers.
    fn create_sampled_image(&mut self, upload: &TextureUpload) -> Result<(), GpuError> {
        if upload.pixels.is_empty() {
            return Err(GpuError::VulkanInitFailed(
                "compute sampled texture with no pixels".to_owned(),
            ));
        }
        // Cube `arrayLayers` must be a valid multiple of 6 (see
        // `TextureUpload::cube_safe_layers`); pad the staging pixels to match so
        // the copy of `img_layers` faces never overruns the buffer.
        let img_layers = upload.cube_safe_layers();
        let staging = upload.staging_pixels(img_layers)?;
        self.sampled.push(SampledAllocation {
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: upload.width,
            height: upload.height,
            depth: upload.depth.max(1),
            layers: img_layers,
        });
        let slot = self.sampled.len() - 1;

        let (staging_buffer, staging_memory) = self.create_host_buffer(
            staging.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some(&*staging),
        )?;
        self.sampled[slot].staging_buffer = staging_buffer;
        self.sampled[slot].staging_memory = staging_memory;

        let volume = upload.depth > 1;
        let info = vk::ImageCreateInfo::default()
            .image_type(if volume {
                vk::ImageType::TYPE_3D
            } else {
                vk::ImageType::TYPE_2D
            })
            .format(upload.format)
            .extent(vk::Extent3D {
                width: upload.width,
                height: upload.height,
                depth: upload.depth.max(1),
            })
            .mip_levels(1)
            .array_layers(img_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .flags(if upload.cube {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            } else {
                vk::ImageCreateFlags::empty()
            });
        // SAFETY: `info` is fully initialized and borrows nothing beyond this
        // call; the device is live.
        let image = unsafe { self.device().create_image(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("sampled vkCreateImage: {e}")))?;
        self.sampled[slot].image = image;

        // SAFETY: `image` was just created from this device.
        let reqs = unsafe { self.device().get_image_memory_requirements(image) };
        let type_index = self
            .dev
            .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);
        // SAFETY: allocation size/type come from this image's own requirements.
        let memory = unsafe { self.device().allocate_memory(&alloc, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("sampled image allocation: {e}")))?;
        self.sampled[slot].memory = memory;

        // SAFETY: memory was allocated for exactly this image; offset 0 is
        // within it and satisfies the alignment requirement by construction.
        unsafe { self.device().bind_image_memory(image, memory, 0) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("sampled image bind memory: {e}")))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            // View type is decided from the T#-TYPE-driven upload flags, NOT
            // from the layer count, so it always matches the recompiled SPIR-V's
            // `OpTypeImage` Arrayed/Dim (both come from `from_texture_type`). A
            // 2DArray (type 13) with a single layer stays `TYPE_2D_ARRAY`
            // (`layer_count == 1`) — binding `TYPE_2D` there was the measured
            // ASTRO.BOT `vkCmdDispatch` device-loss (view type 2D under an
            // `Arrayed = 1` sampled image).
            .view_type(if upload.cube {
                vk::ImageViewType::CUBE
            } else if volume {
                vk::ImageViewType::TYPE_3D
            } else if upload.array {
                // 2DArray (T# type 13) — the recompiled SPIR-V samples an
                // arrayed 2D image (measured: ASTRO.BOT's 1536x1536x3).
                vk::ImageViewType::TYPE_2D_ARRAY
            } else {
                vk::ImageViewType::TYPE_2D
            })
            .format(upload.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: img_layers,
            });
        // SAFETY: the view's image is live and its format/range match the
        // image's creation parameters.
        let view = unsafe { self.device().create_image_view(&view_info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("sampled image view: {e}")))?;
        self.sampled[slot].view = view;
        Ok(())
    }

    fn record_and_submit(&self, state: &ComputeState<'_>) -> Result<(), GpuError> {
        let full_color = |allocation: &ImageAllocation| vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: allocation.layers,
        };
        let copy_region = |allocation: &ImageAllocation| {
            vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: allocation.layers,
                })
                .image_extent(vk::Extent3D {
                    width: allocation.width,
                    height: allocation.height,
                    depth: allocation.depth,
                })
        };
        if !self.batched {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // SAFETY: the cached command buffer is not pending — its previous
            // submission fence-completed under the cache lock — and the pool was
            // created RESET_COMMAND_BUFFER, so begin implicitly resets it.
            unsafe {
                self.device()
                    .begin_command_buffer(self.command_buffer, &begin)
            }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkBeginCommandBuffer: {e}")))?;
        }
        // SAFETY: pipeline/layout/sets/images are live and the command buffer
        // is recording; barriers and copies name handles this bundle retains
        // until after the fence wait.
        unsafe {
            // Upload every UAV's initial content and move it to GENERAL, the
            // layout the STORAGE_IMAGE descriptor promised.
            for allocation in self
                .images
                .iter()
                .filter(|allocation| allocation.records_commands)
            {
                let (old_layout, src_access, src_stage) = if allocation.upload_seed {
                    let old_layout = if allocation.fresh {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    };
                    let src_access = if allocation.fresh {
                        vk::AccessFlags::empty()
                    } else {
                        vk::AccessFlags::TRANSFER_READ
                    };
                    let src_stage = if allocation.fresh {
                        vk::PipelineStageFlags::TOP_OF_PIPE
                    } else {
                        vk::PipelineStageFlags::TRANSFER
                    };
                    let to_transfer = vk::ImageMemoryBarrier::default()
                        .src_access_mask(src_access)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(old_layout)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(allocation.image)
                        .subresource_range(full_color(allocation));
                    self.device().cmd_pipeline_barrier(
                        self.command_buffer,
                        src_stage,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_transfer],
                    );
                    self.device().cmd_copy_buffer_to_image(
                        self.command_buffer,
                        allocation.staging_buffer,
                        allocation.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy_region(allocation)],
                    );
                    (
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::PipelineStageFlags::TRANSFER,
                    )
                } else {
                    // The prior dispatch copied its result to the persistent
                    // readback buffer and left the image in TRANSFER_SRC.
                    // Keep that GPU-newer result; only transition it back to
                    // GENERAL for this ordered dispatch.
                    (
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                    )
                };
                let to_general = vk::ImageMemoryBarrier::default()
                    .src_access_mask(src_access)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(allocation.image)
                    .subresource_range(full_color(allocation));
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    src_stage,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }

            // Upload every sampled texture and move it to SHADER_READ_ONLY,
            // the layout its SAMPLED_IMAGE descriptor promised.
            for allocation in &self.sampled {
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: allocation.layers,
                };
                let to_transfer = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(allocation.image)
                    .subresource_range(range);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_transfer],
                );
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: allocation.layers,
                    })
                    .image_extent(vk::Extent3D {
                        width: allocation.width,
                        height: allocation.height,
                        depth: allocation.depth,
                    });
                self.device().cmd_copy_buffer_to_image(
                    self.command_buffer,
                    allocation.staging_buffer,
                    allocation.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                let to_sampled = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(allocation.image)
                    .subresource_range(range);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_sampled],
                );
            }

            self.device().cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            if !self.descriptor_set.is_null() {
                self.device().cmd_bind_descriptor_sets(
                    self.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
            }
            if let Some(binding) = state.binding
                && !binding.push_constants.is_empty()
                && binding.push_uniform_binding.is_none()
            {
                self.device().cmd_push_constants(
                    self.command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    binding.push_constant_offset,
                    &binding.push_constants,
                );
            }
            let slices = dispatch_slices(state);
            if slices.len() > 1 {
                use std::sync::atomic::{AtomicU64, Ordering};
                static SLICED_DISPATCHES: AtomicU64 = AtomicU64::new(0);
                let sliced = SLICED_DISPATCHES.fetch_add(1, Ordering::Relaxed) + 1;
                if sliced <= 16 || sliced.is_power_of_two() {
                    tracing::warn!(
                        sliced_dispatch = sliced,
                        groups = format_args!(
                            "{}x{}x{}",
                            state.groups[0], state.groups[1], state.groups[2]
                        ),
                        local_size = ?compute_local_size(state.spirv),
                        spirv_words = state.spirv.len(),
                        slices = slices.len(),
                        "compute dispatch divided into TDR-safe WorkGroupID slices"
                    );
                }
            }
            for (index, slice) in slices.iter().enumerate() {
                if slice.base == [0; 3] && slice.groups == state.groups {
                    self.device().cmd_dispatch(
                        self.command_buffer,
                        slice.groups[0],
                        slice.groups[1],
                        slice.groups[2],
                    );
                } else {
                    self.device().cmd_dispatch_base(
                        self.command_buffer,
                        slice.base[0],
                        slice.base[1],
                        slice.base[2],
                        slice.groups[0],
                        slice.groups[1],
                        slice.groups[2],
                    );
                }
                if index + 1 != slices.len() {
                    // The original dispatch provides no ordering between
                    // workgroups. Serial slices are necessarily stronger; make
                    // any guest storage effects visible before the next slice.
                    let between_slices = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        );
                    self.device().cmd_pipeline_barrier(
                        self.command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[between_slices],
                        &[],
                        &[],
                    );
                    if !self.batched {
                        // A second vkCmdDispatch in the same command buffer did
                        // not contain ASTRO.BOT's measured Windows TDR. End,
                        // submit, and fence-complete every slice so the next
                        // slice is a genuinely separate schedulable unit.
                        self.device()
                            .end_command_buffer(self.command_buffer)
                            .map_err(|e| {
                                self.dev.note_vk_error(e);
                                GpuError::VulkanInitFailed(format!(
                                    "vkEndCommandBuffer (compute slice): {e}"
                                ))
                            })?;
                        let command_buffers = [self.command_buffer];
                        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
                        self.device()
                            .queue_submit(self.dev.queue(), &submits, self.fence)
                            .map_err(|e| {
                                self.dev.note_vk_error(e);
                                GpuError::VulkanInitFailed(format!(
                                    "vkQueueSubmit (compute slice): {e}"
                                ))
                            })?;
                        self.device()
                            .wait_for_fences(&[self.fence], true, u64::MAX)
                            .map_err(|e| {
                                self.dev.note_vk_error(e);
                                GpuError::VulkanInitFailed(format!(
                                    "vkWaitForFences (compute slice): {e}"
                                ))
                            })?;
                        self.device().reset_fences(&[self.fence]).map_err(|e| {
                            self.dev.note_vk_error(e);
                            GpuError::VulkanInitFailed(format!(
                                "vkResetFences (compute slice): {e}"
                            ))
                        })?;
                        let begin = vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                        self.device()
                            .begin_command_buffer(self.command_buffer, &begin)
                            .map_err(|e| {
                                self.dev.note_vk_error(e);
                                GpuError::VulkanInitFailed(format!(
                                    "vkBeginCommandBuffer (compute slice): {e}"
                                ))
                            })?;
                        self.device().cmd_bind_pipeline(
                            self.command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            self.pipeline,
                        );
                        if !self.descriptor_set.is_null() {
                            self.device().cmd_bind_descriptor_sets(
                                self.command_buffer,
                                vk::PipelineBindPoint::COMPUTE,
                                self.pipeline_layout,
                                0,
                                &[self.descriptor_set],
                                &[],
                            );
                        }
                        if let Some(binding) = state.binding
                            && !binding.push_constants.is_empty()
                            && binding.push_uniform_binding.is_none()
                        {
                            self.device().cmd_push_constants(
                                self.command_buffer,
                                self.pipeline_layout,
                                vk::ShaderStageFlags::COMPUTE,
                                binding.push_constant_offset,
                                &binding.push_constants,
                            );
                        }
                    }
                }
            }

            // GDS contents persist across dispatches: make this dispatch's
            // GDS writes available to LATER dispatches' shader reads/writes
            // (a pipeline barrier's second scope covers subsequent
            // submissions on this queue; the fence alone only orders
            // device-to-host visibility).
            if !self.gds.is_null() || !self.storage.is_empty() {
                let gds_flush = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[gds_flush],
                    &[],
                    &[],
                );
            }

            // Copy every UAV back out for the guest-memory writeback.
            for allocation in self
                .images
                .iter()
                .filter(|allocation| allocation.records_commands)
            {
                let to_readback = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(allocation.image)
                    .subresource_range(full_color(allocation));
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_readback],
                );
                self.device().cmd_copy_image_to_buffer(
                    self.command_buffer,
                    allocation.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    allocation.readback_buffer,
                    &[copy_region(allocation)],
                );
            }
            if !self.images.is_empty() {
                // Make the transfer writes visible to the host map after the
                // fence wait.
                let host_read = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[host_read],
                    &[],
                    &[],
                );
            }
            if !self.batched {
                self.device()
                    .end_command_buffer(self.command_buffer)
                    .map_err(|e| {
                        self.dev.note_vk_error(e);
                        GpuError::VulkanInitFailed(format!("vkEndCommandBuffer: {e}"))
                    })?;
            }
        }
        if self.batched {
            // Do not submit one command buffer at a time. The batch flush
            // submits every recorded draw/dispatch command buffer in PM4
            // order with one vkQueueSubmit and one fence. Merely omitting the
            // per-dispatch fence still paid the driver's queue-submit cost
            // hundreds of times per frame.
            Ok(())
        } else {
            let command_buffers = [self.command_buffer];
            let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            // SAFETY: every handle belongs to this device, the command buffer
            // is executable and the reusable fence is currently unsignaled.
            unsafe {
                self.device()
                    .queue_submit(self.dev.queue(), &submits, self.fence)
            }
            .map_err(|e| {
                self.dev.note_vk_error(e);
                GpuError::VulkanInitFailed(format!("vkQueueSubmit: {e}"))
            })?;
            // SAFETY: waiting on this submission's live fence.
            unsafe { self.device().wait_for_fences(&[self.fence], true, u64::MAX) }.map_err(|e| {
                self.dev.note_vk_error(e);
                GpuError::VulkanInitFailed(format!("vkWaitForFences: {e}"))
            })
        }
    }

    fn commit_to_batch(&mut self) -> Result<(), GpuError> {
        debug_assert!(self.batched);
        // The cache owns the shared recording handle until the flip closes it.
        self.command_buffer = vk::CommandBuffer::null();
        let mut pending = PendingDrawResources {
            command_buffer: vk::CommandBuffer::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            buffers: Vec::new(),
            images: Vec::new(),
        };
        for allocation in [&mut self.eud_raw, &mut self.push_uniform] {
            if let Some(allocation) = allocation.take() {
                debug_assert!(allocation.persistent_key.is_none());
                pending.buffers.push((allocation.buffer, allocation.memory));
            }
        }
        while let Some(allocation) = self.images.pop() {
            if !allocation.staging_buffer.is_null() || !allocation.staging_memory.is_null() {
                pending
                    .buffers
                    .push((allocation.staging_buffer, allocation.staging_memory));
            }
            debug_assert!(
                allocation.persistent,
                "deferred storage images must live in the persistent cache"
            );
        }
        while let Some(allocation) = self.sampled.pop() {
            pending
                .buffers
                .push((allocation.staging_buffer, allocation.staging_memory));
            pending
                .images
                .push((allocation.image, allocation.memory, allocation.view));
        }
        self.caches.commit_deferred_resources(
            pending,
            std::mem::take(&mut self.deferred_write_keys),
            std::mem::take(&mut self.deferred_image_writes),
        );
        Ok(())
    }

    /// Return a pooled persistent mapping when available, otherwise establish
    /// a transient mapping. The boolean reports whether the caller must unmap.
    fn host_memory_address(
        &self,
        memory: vk::DeviceMemory,
        size: usize,
    ) -> Result<(usize, bool), GpuError> {
        if let Some(mapped) = self.caches.mapped_host_memory(memory) {
            return Ok((mapped, false));
        }
        // SAFETY: callers invoke this only after the queue fence signals, and
        // the allocation is HOST_VISIBLE and covers `size`.
        let ptr = unsafe {
            self.device()
                .map_memory(memory, 0, size as u64, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkMapMemory: {e}")))?;
        Ok((ptr as usize, true))
    }

    /// Copy `size` bytes out of one fence-idle host-coherent allocation.
    fn read_host_memory(&self, memory: vk::DeviceMemory, size: usize) -> Result<Vec<u8>, GpuError> {
        let (mapped, must_unmap) = self.host_memory_address(memory, size)?;
        // Fallible host allocation: a large compute readback under host memory
        // pressure must DEGRADE (return an error the dispatch path skips on),
        // never abort the whole process via the infallible allocator. Same
        // "degrade, not abort" policy as `draw_translate::alloc_zeroed`.
        let mut bytes: Vec<u8> = Vec::new();
        bytes.try_reserve_exact(size).map_err(|_| {
            if must_unmap {
                // SAFETY: balances the transient mapping opened above.
                unsafe { self.device().unmap_memory(memory) };
            }
            GpuError::VulkanInitFailed(format!(
                "compute readback: {size} B host allocation failed (out of memory) — \
                skipping the dispatch instead of aborting"
            ))
        })?;
        // Do not `resize(size, 0)` before the copy: that wrote the entire
        // multi-megabyte destination once with zeroes and immediately wrote
        // it again with the GPU result. Measured Minecraft dispatches return
        // a 4 MiB V# every call; the redundant first pass was inside the
        // 18-22 ms `map_copy` wall. Capacity is allocated but length remains
        // zero until `copy_nonoverlapping` initializes every byte.
        //
        // SAFETY: `try_reserve_exact` made at least `size` bytes writable at
        // `as_mut_ptr`; the source mapping covers exactly `size`, cannot
        // overlap this host allocation, and the copy initializes the whole
        // future slice before `set_len` exposes it.
        unsafe {
            std::ptr::copy_nonoverlapping(mapped as *const u8, bytes.as_mut_ptr(), size);
            bytes.set_len(size);
            if must_unmap {
                self.device().unmap_memory(memory);
            }
        }
        Ok(bytes)
    }

    fn read_storage(
        &mut self,
        storage: Option<&super::offscreen::StorageBufferBinding>,
    ) -> Result<Vec<ComputeBufferOutput>, GpuError> {
        let Some(storage_binding) = storage else {
            return Ok(Vec::new());
        };
        if storage_binding.writable.len() != self.storage.len()
            || storage_binding.buffers.len() != self.storage.len()
            || storage_binding.guest_bases.len() != self.storage.len()
            || storage_binding.guest_sizes.len() != self.storage.len()
        {
            return Err(GpuError::PipelineCreationFailed(format!(
                "compute storage inputs ({}) / guest bases ({}) / guest sizes ({}) / writeback \
                 flags ({}) do not match buffer count ({})",
                storage_binding.buffers.len(),
                storage_binding.guest_bases.len(),
                storage_binding.guest_sizes.len(),
                storage_binding.writable.len(),
                self.storage.len()
            )));
        }
        let mut outputs = Vec::new();
        for (index, initial) in storage_binding.buffers.iter().enumerate() {
            if !storage_binding.writable[index] {
                continue;
            }
            let allocation = &self.storage[index];
            let key = allocation.persistent_key;
            let output = self.read_storage_dirty(allocation.memory, allocation.size, initial)?;
            if let Some(key) = key {
                self.caches.update_compute_buffer_shadow(key, &output.dirty);
            }
            outputs.push(output);
        }
        Ok(outputs)
    }

    /// Map a cached coherent storage allocation and retain only changed
    /// 4 KiB pages. Page granularity keeps the comparison linear and cheap
    /// while coalescing adjacent writes into one guest-memory copy.
    fn read_storage_dirty(
        &self,
        memory: vk::DeviceMemory,
        size: usize,
        initial: &[u8],
    ) -> Result<ComputeBufferOutput, GpuError> {
        if initial.len() != size {
            return Err(GpuError::PipelineCreationFailed(format!(
                "compute storage initial bytes ({}) do not match allocation size ({size})",
                initial.len()
            )));
        }
        let (mapped, must_unmap) = self.host_memory_address(memory, size)?;
        // SAFETY: the fence completed before this call, the coherent mapping
        // covers `size`, and it remains live through the comparison below.
        let result = unsafe { std::slice::from_raw_parts(mapped as *const u8, size) };
        const PAGE: usize = 4096;
        let mut dirty = Vec::new();
        let mut at = 0usize;
        while at < size {
            let end = at.saturating_add(PAGE).min(size);
            if result[at..end] == initial[at..end] {
                at = end;
                continue;
            }
            let start = at;
            at = end;
            while at < size {
                let next = at.saturating_add(PAGE).min(size);
                if result[at..next] == initial[at..next] {
                    break;
                }
                at = next;
            }
            let mut bytes = Vec::new();
            if bytes.try_reserve_exact(at - start).is_err() {
                if must_unmap {
                    // SAFETY: balances the transient mapping opened above.
                    unsafe { self.device().unmap_memory(memory) };
                }
                return Err(GpuError::VulkanInitFailed(format!(
                    "compute dirty writeback: {} B host allocation failed",
                    at - start
                )));
            }
            bytes.extend_from_slice(&result[start..at]);
            dirty.push(ComputeDirtySpan {
                offset: start,
                bytes,
            });
        }
        if must_unmap {
            // SAFETY: balances the transient mapping after borrowed slices'
            // final use. Persistent pooled mappings deliberately stay live.
            unsafe { self.device().unmap_memory(memory) };
        }
        Ok(ComputeBufferOutput { size, dirty })
    }

    fn read_images(&self) -> Result<Vec<Vec<u8>>, GpuError> {
        self.images
            .iter()
            .map(|allocation| {
                let size = (allocation.width as usize)
                    * (allocation.height as usize)
                    * (allocation.depth as usize)
                    * (allocation.layers as usize)
                    * (allocation.texel as usize);
                self.read_host_memory(allocation.readback_memory, size)
            })
            .collect()
    }
}

impl Drop for ComputeResources<'_> {
    fn drop(&mut self) {
        // NOT destroyed here — owned by the device's DrawCaches and reused by
        // the next draw/dispatch: fence, command buffer, pipeline, pipeline
        // layout, descriptor pool, descriptor layout, shader module, samplers.
        //
        // Every host buffer came from the shared pool. The dispatch fence was
        // waited before a success reaches here; a pre-submit build failure
        // also leaves them unreferenced, so returning them is safe.
        while let Some(allocation) = self.storage.pop() {
            if allocation.persistent_key.is_none() {
                self.caches
                    .release_host_buffer(self.dev, allocation.buffer, allocation.memory);
            }
        }
        if self.batched && self.command_buffer != vk::CommandBuffer::null() {
            self.command_buffer = vk::CommandBuffer::null();
        }
        if let Some(allocation) = self.eud_raw.take() {
            self.caches
                .release_host_buffer(self.dev, allocation.buffer, allocation.memory);
        }
        if let Some(allocation) = self.push_uniform.take() {
            self.caches
                .release_host_buffer(self.dev, allocation.buffer, allocation.memory);
        }
        self.samplers.clear();
        while let Some(allocation) = self.sampled.pop() {
            // SAFETY: sampled images are still per-dispatch and the fence has
            // completed (or submission never happened); children are
            // destroyed before parents.
            unsafe {
                let device = self.dev.device();
                if !allocation.view.is_null() {
                    device.destroy_image_view(allocation.view, None);
                }
                if !allocation.image.is_null() {
                    device.destroy_image(allocation.image, None);
                }
                if !allocation.memory.is_null() {
                    device.free_memory(allocation.memory, None);
                }
            }
            self.caches.release_host_buffer(
                self.dev,
                allocation.staging_buffer,
                allocation.staging_memory,
            );
        }
        while let Some(allocation) = self.images.pop() {
            if !allocation.staging_buffer.is_null() || !allocation.staging_memory.is_null() {
                self.caches.release_host_buffer(
                    self.dev,
                    allocation.staging_buffer,
                    allocation.staging_memory,
                );
            }
            if !allocation.persistent {
                // A partial creation error before cache insertion still owns
                // whatever non-null handles it reached.
                unsafe {
                    let device = self.dev.device();
                    if !allocation.view.is_null() {
                        device.destroy_image_view(allocation.view, None);
                    }
                    if !allocation.image.is_null() {
                        device.destroy_image(allocation.image, None);
                    }
                    if !allocation.memory.is_null() {
                        device.free_memory(allocation.memory, None);
                    }
                }
                self.caches.release_host_buffer(
                    self.dev,
                    allocation.readback_buffer,
                    allocation.readback_memory,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ComputeState, compute_local_size, compute_requires_slicing, dispatch_slices,
        inline_push_constant_range,
    };

    fn compute_module(local: [u32; 3], words: usize) -> Vec<u32> {
        let mut module = vec![
            0x0723_0203,
            0x0001_0300,
            0,
            10,
            0,
            (6 << 16) | 16, // OpExecutionMode %1 LocalSize x y z
            1,
            17,
            local[0],
            local[1],
            local[2],
        ];
        // OpNop (opcode 0, word count 1) padding.
        module.resize(words.max(module.len()), 1 << 16);
        module
    }

    #[test]
    fn minecraft_304_byte_resource_table_uses_spill_ssbo() {
        assert_eq!(
            inline_push_constant_range(0, 304, Some(4), 256).expect("spill is valid"),
            None
        );
    }

    #[test]
    fn oversized_inline_resource_table_is_refused() {
        let error = inline_push_constant_range(0, 304, None, 256)
            .expect_err("missing spill must remain invalid");
        assert!(error.to_string().contains("declares no spill SSBO"));
    }

    #[test]
    fn small_resource_table_stays_inline() {
        assert_eq!(
            inline_push_constant_range(16, 128, None, 256).expect("inline range is valid"),
            Some((16, 128))
        );
    }

    #[test]
    fn reads_compute_local_size_from_execution_mode() {
        assert_eq!(
            compute_local_size(&compute_module([16, 8, 2], 32)),
            [16, 8, 2]
        );
    }

    #[test]
    fn astro_heavy_compute_is_divided_without_changing_workgroup_ids() {
        let spirv = compute_module([16, 16, 1], 8_700);
        let state = ComputeState {
            groups: [64, 64, 1],
            spirv: &spirv,
            binding: None,
        };
        let slices = dispatch_slices(&state);
        assert!(slices.len() > 1, "{slices:?}");

        // Equal X/Y dimensions choose one axis, preserve the other two, and
        // cover every original workgroup exactly once through dispatch-base.
        let axis = (0..3)
            .find(|&axis| slices.iter().any(|slice| slice.base[axis] != 0))
            .expect("one dimension must be sliced");
        let mut next = 0;
        for slice in &slices {
            assert_eq!(slice.base[axis], next);
            next += slice.groups[axis];
            for other in (0..3).filter(|&candidate| candidate != axis) {
                assert_eq!(slice.base[other], 0);
                assert_eq!(slice.groups[other], state.groups[other]);
            }
        }
        assert_eq!(next, state.groups[axis]);
        assert!(compute_requires_slicing(&state));
    }

    #[test]
    fn ordinary_compute_stays_one_plain_dispatch() {
        let spirv = compute_module([8, 8, 1], 256);
        let state = ComputeState {
            groups: [8, 8, 1],
            spirv: &spirv,
            binding: None,
        };
        let slices = dispatch_slices(&state);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].base, [0; 3]);
        assert_eq!(slices[0].groups, state.groups);
        assert!(!compute_requires_slicing(&state));
    }
}
