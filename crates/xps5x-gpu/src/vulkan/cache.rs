//! Long-lived Vulkan resource caches shared by every draw and dispatch on one
//! device (performance stage A).
//!
//! Before this cache existed, `render_draw` rebuilt **every** Vulkan resource
//! per draw — shader modules, descriptor set layouts, pipeline layout, the
//! pipeline itself, command buffer, fence, descriptor pool, and the render
//! target images — then destroyed them all. Measured on ASTRO.BOT
//! (`XPS5X_TIME_DRAW=1`, release): 1.6–16 ms of pure resource construction per
//! draw, ~170 draws per frame — under 1 FPS before a single pixel moves.
//!
//! What lives here, keyed by what:
//!
//! | Resource | Key |
//! |---|---|
//! | `VkShaderModule` | full SPIR-V words (exact content) |
//! | `VkDescriptorSetLayout` | (binding, type, count, stages) list |
//! | `VkPipelineLayout` | canonical set-layout handles + push ranges |
//! | graphics `VkPipeline` | every state that feeds creation (see [`GraphicsPipelineKey`]) |
//! | compute `VkPipeline` | (canonical module, canonical layout) |
//! | `VkSampler` | linear-vs-nearest flag |
//! | render target image + readback buffer | (guest base, width, height, format) |
//! | command buffer + fence | singleton, reset per submission |
//! | descriptor pool | singleton, grown on demand, reset per acquisition |
//!
//! Viewport, scissor, and blend constants are **dynamic** pipeline state, so
//! they deliberately do not key the pipeline.
//!
//! ## Thread ownership / locking
//!
//! The cache sits behind a `Mutex` on [`VulkanDevice`] and the lock is held for
//! the **entire** draw or dispatch (`render_draw` / `dispatch_compute`), which
//! also serializes queue submission. In production exactly one thread renders:
//! the per-process `xps5x-gpu` worker consumes the submit queue single-file and
//! the session's `backend` mutex covers the inline-fallback path, so this lock
//! is uncontended; it exists so tests that drive a device from several threads
//! stay sound. Every use of a cached resource completes synchronously (the
//! fence is waited before the lock is released), which is what makes resetting
//! the fence, command buffer, and descriptor pool at the next acquisition
//! legal.

use super::instance::VulkanDevice;
use ash::vk::{self, Handle};
use std::collections::HashMap;
use xps5x_core::error::GpuError;

/// Cache-effectiveness counters, cumulative since device creation.
///
/// `seed_uploads_skipped` counts draws whose attachment LOAD was satisfied
/// from the persistent GPU image instead of re-uploading the CPU-side
/// framebuffer — the stage A fast path for composited frames.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DrawCacheStats {
    pub pipeline_hits: u64,
    pub pipeline_misses: u64,
    pub compute_pipeline_hits: u64,
    pub compute_pipeline_misses: u64,
    pub shader_module_hits: u64,
    pub shader_module_misses: u64,
    pub target_hits: u64,
    pub target_misses: u64,
    pub seed_uploads_skipped: u64,
}

/// One descriptor-set-layout binding, reduced to the fields that feed
/// `VkDescriptorSetLayoutBinding` (immutable samplers are never used here).
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
struct LayoutBindingKey {
    binding: u32,
    ty: i32,
    count: u32,
    stages: u32,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
struct PipelineLayoutKey {
    /// Canonical `VkDescriptorSetLayout` handles — canonical because they come
    /// from this cache's own set-layout map, so equal signatures yield equal
    /// handles.
    set_layouts: Vec<u64>,
    /// (stage flags, offset, size) per push-constant range.
    push_ranges: Vec<(u32, u32, u32)>,
}

/// `VkStencilOpState`, reduced to hashable raw values.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub(crate) struct StencilKey {
    fail: i32,
    pass: i32,
    depth_fail: i32,
    compare: i32,
    compare_mask: u32,
    write_mask: u32,
    reference: u32,
}

impl StencilKey {
    pub(crate) fn from_vk(s: &vk::StencilOpState) -> Self {
        Self {
            fail: s.fail_op.as_raw(),
            pass: s.pass_op.as_raw(),
            depth_fail: s.depth_fail_op.as_raw(),
            compare: s.compare_op.as_raw(),
            compare_mask: s.compare_mask,
            write_mask: s.write_mask,
            reference: s.reference,
        }
    }
}

/// The depth/stencil half of a graphics pipeline key.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub(crate) struct DepthPipelineKey {
    pub format: i32,
    pub test: bool,
    pub write: bool,
    pub compare: i32,
    pub stencil_test: bool,
    pub front: StencilKey,
    pub back: StencilKey,
    /// Whether the pipeline declares a stencil attachment format.
    pub stencil_attachment: bool,
}

/// The blend half of a graphics pipeline key. Blend **constants** are absent
/// on purpose: they are dynamic state (`vkCmdSetBlendConstants`).
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub(crate) struct BlendKey {
    pub enable: bool,
    pub src_color: i32,
    pub dst_color: i32,
    pub color_op: i32,
    pub src_alpha: i32,
    pub dst_alpha: i32,
    pub alpha_op: i32,
}

/// Everything that feeds `vkCreateGraphicsPipelines` for the offscreen draw
/// path. Two draws with equal keys are served by one `VkPipeline`; viewport,
/// scissor, and blend constants are dynamic and set during recording.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub(crate) struct GraphicsPipelineKey {
    /// Canonical shader-module handles (from [`DrawCaches::shader_module`],
    /// so identical SPIR-V yields identical handles).
    pub vs: u64,
    pub fs: u64,
    /// Canonical pipeline-layout handle — covers descriptor-set signatures and
    /// push-constant ranges transitively.
    pub layout: u64,
    /// `None` for a depth-only pipeline (zero colour attachments).
    pub color_format: Option<i32>,
    pub depth: Option<DepthPipelineKey>,
    pub topology: i32,
    pub cull: u32,
    pub front_face: i32,
    pub color_write_mask: u32,
    pub blend: BlendKey,
    /// (binding, stride) — input rate is always per-vertex here.
    pub vertex_bindings: Vec<(u32, u32)>,
    /// (location, binding, format, offset).
    pub vertex_attributes: Vec<(u32, u32, i32, u32)>,
}

/// A guest render target kept alive on the device across draws.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) struct TargetKey {
    pub base: u64,
    pub width: u32,
    pub height: u32,
    pub format: i32,
}

/// The device-side half of one guest render target: the attachment image and
/// the host-visible buffer its pixels are read back through.
///
/// `synced` is the honesty bit for the seed-skip fast path: it is true exactly
/// when the GPU image's contents are byte-identical to the last colour
/// readback handed to the caller (which is what the CPU-side framebuffer map
/// stores). It is cleared when a draw acquires the target and set again only
/// after that draw's readback completes, so a failed draw can never leave a
/// stale image masquerading as the composed frame.
#[derive(Clone, Copy)]
pub(crate) struct PersistentTarget {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub readback_buffer: vk::Buffer,
    pub readback_memory: vk::DeviceMemory,
    pub synced: bool,
}

/// The grown-on-demand descriptor pool (see [`DrawCaches::descriptor_pool`]).
struct PoolState {
    pool: vk::DescriptorPool,
    max_sets: u32,
    /// Capacity per raw `VkDescriptorType`.
    capacity: HashMap<i32, u32>,
}

/// See the module docs for inventory, keys, and the locking contract.
#[derive(Default)]
pub(crate) struct DrawCaches {
    shader_modules: HashMap<Vec<u32>, vk::ShaderModule>,
    set_layouts: HashMap<Vec<LayoutBindingKey>, vk::DescriptorSetLayout>,
    pipeline_layouts: HashMap<PipelineLayoutKey, vk::PipelineLayout>,
    graphics_pipelines: HashMap<GraphicsPipelineKey, vk::Pipeline>,
    /// Keyed by (canonical module handle, canonical layout handle).
    compute_pipelines: HashMap<(u64, u64), vk::Pipeline>,
    /// Keyed by linear-vs-nearest — the only sampler state decoded today.
    samplers: HashMap<bool, vk::Sampler>,
    targets: HashMap<TargetKey, PersistentTarget>,
    /// One command buffer + fence, reused for every synchronous submission.
    submit: Option<(vk::CommandBuffer, vk::Fence)>,
    pool: Option<PoolState>,
    /// The device-persistent GDS arena (see [`DrawCaches::gds_buffer`]).
    gds: Option<(vk::Buffer, vk::DeviceMemory)>,
    pub stats: DrawCacheStats,
}

/// Byte size of the emulated GDS arena — the real chip's Global Data Share is
/// 64 KiB.
pub(crate) const GDS_SIZE: usize = 64 * 1024;

impl DrawCaches {
    /// Get or create the `VkShaderModule` for exactly these SPIR-V words.
    pub(crate) fn shader_module(
        &mut self,
        dev: &VulkanDevice,
        code: &[u32],
    ) -> Result<vk::ShaderModule, GpuError> {
        if let Some(&module) = self.shader_modules.get(code) {
            self.stats.shader_module_hits += 1;
            return Ok(module);
        }
        let info = vk::ShaderModuleCreateInfo::default().code(code);
        // SAFETY: `code` is a `&[u32]` — 4-byte aligned, whole words — alive
        // for the call. The module is retained in this cache and destroyed
        // exactly once in `destroy`.
        let module = unsafe { dev.device().create_shader_module(&info, None) }
            .map_err(|e| GpuError::ShaderCompilationFailed(format!("vkCreateShaderModule: {e}")))?;
        self.stats.shader_module_misses += 1;
        self.shader_modules.insert(code.to_vec(), module);
        Ok(module)
    }

    /// Get or create a descriptor set layout for these bindings. Immutable
    /// samplers are not supported (no caller uses them).
    pub(crate) fn set_layout(
        &mut self,
        dev: &VulkanDevice,
        bindings: &[vk::DescriptorSetLayoutBinding<'_>],
    ) -> Result<vk::DescriptorSetLayout, GpuError> {
        let key: Vec<LayoutBindingKey> = bindings
            .iter()
            .map(|b| {
                debug_assert!(
                    b.p_immutable_samplers.is_null(),
                    "immutable samplers are not part of the layout key"
                );
                LayoutBindingKey {
                    binding: b.binding,
                    ty: b.descriptor_type.as_raw(),
                    count: b.descriptor_count,
                    stages: b.stage_flags.as_raw(),
                }
            })
            .collect();
        if let Some(&layout) = self.set_layouts.get(&key) {
            return Ok(layout);
        }
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
        // SAFETY: `bindings` is alive for the call; the layout is retained in
        // this cache and destroyed exactly once in `destroy`.
        let layout =
            unsafe { dev.device().create_descriptor_set_layout(&info, None) }.map_err(|e| {
                GpuError::PipelineCreationFailed(format!("vkCreateDescriptorSetLayout: {e}"))
            })?;
        self.set_layouts.insert(key, layout);
        Ok(layout)
    }

    /// Get or create a pipeline layout for these (canonical) set layouts and
    /// push ranges.
    pub(crate) fn pipeline_layout(
        &mut self,
        dev: &VulkanDevice,
        set_layouts: &[vk::DescriptorSetLayout],
        push_ranges: &[vk::PushConstantRange],
    ) -> Result<vk::PipelineLayout, GpuError> {
        let key = PipelineLayoutKey {
            set_layouts: set_layouts.iter().map(|l| l.as_raw()).collect(),
            push_ranges: push_ranges
                .iter()
                .map(|r| (r.stage_flags.as_raw(), r.offset, r.size))
                .collect(),
        };
        if let Some(&layout) = self.pipeline_layouts.get(&key) {
            return Ok(layout);
        }
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_ranges);
        // SAFETY: both slices are alive for the call and every set layout is a
        // live handle from this cache. Retained here; destroyed in `destroy`.
        let layout = unsafe { dev.device().create_pipeline_layout(&info, None) }.map_err(|e| {
            GpuError::PipelineCreationFailed(format!("vkCreatePipelineLayout: {e}"))
        })?;
        self.pipeline_layouts.insert(key, layout);
        Ok(layout)
    }

    /// A cached graphics pipeline, if one exists for `key` (counts a hit).
    pub(crate) fn lookup_graphics_pipeline(
        &mut self,
        key: &GraphicsPipelineKey,
    ) -> Option<vk::Pipeline> {
        let hit = self.graphics_pipelines.get(key).copied();
        if hit.is_some() {
            self.stats.pipeline_hits += 1;
        }
        hit
    }

    /// Retain a freshly created graphics pipeline (counts a miss).
    pub(crate) fn store_graphics_pipeline(
        &mut self,
        key: GraphicsPipelineKey,
        pipeline: vk::Pipeline,
    ) {
        self.stats.pipeline_misses += 1;
        self.graphics_pipelines.insert(key, pipeline);
    }

    /// Get or create the compute pipeline for a (canonical module, canonical
    /// layout) pair — the only two inputs `vkCreateComputePipelines` takes
    /// here.
    pub(crate) fn compute_pipeline(
        &mut self,
        dev: &VulkanDevice,
        module: vk::ShaderModule,
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, GpuError> {
        let key = (module.as_raw(), layout.as_raw());
        if let Some(&pipeline) = self.compute_pipelines.get(&key) {
            self.stats.compute_pipeline_hits += 1;
            return Ok(pipeline);
        }
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: module and layout are live cached handles from this device;
        // the create info is local. Retained here; destroyed in `destroy`.
        let pipeline = unsafe {
            dev.device()
                .create_compute_pipelines(dev.pipeline_cache(), &[info], None)
        }
        .map_err(|(_, e)| {
            GpuError::PipelineCreationFailed(format!("vkCreateComputePipelines: {e}"))
        })?[0];
        self.stats.compute_pipeline_misses += 1;
        self.compute_pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    /// Get or create the sampler for `linear` filtering. Only linear-vs-nearest
    /// is decoded from the guest S# today, so two samplers cover every draw.
    pub(crate) fn sampler(
        &mut self,
        dev: &VulkanDevice,
        linear: bool,
    ) -> Result<vk::Sampler, GpuError> {
        if let Some(&sampler) = self.samplers.get(&linear) {
            return Ok(sampler);
        }
        let filter = if linear {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(filter)
            .min_filter(filter)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .max_lod(0.0);
        // SAFETY: plain sampler on a live device; retained in this cache and
        // destroyed exactly once in `destroy`.
        let sampler = unsafe { dev.device().create_sampler(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateSampler: {e}")))?;
        self.samplers.insert(linear, sampler);
        Ok(sampler)
    }

    /// The device-persistent GDS arena: one 64 KiB storage buffer, zeroed at
    /// creation, whose contents persist across dispatches for the lifetime of
    /// the device — real GDS counters (`ds_append`/`ds_consume`) accumulate
    /// across dispatches to feed indirect-draw arguments (measured on
    /// ASTRO.BOT). GDS is on-chip memory, so nothing is ever written back to
    /// guest memory.
    pub(crate) fn gds_buffer(&mut self, dev: &VulkanDevice) -> Result<vk::Buffer, GpuError> {
        if let Some((buffer, _)) = self.gds {
            return Ok(buffer);
        }
        let info = vk::BufferCreateInfo::default()
            .size(GDS_SIZE as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: local create info on a live device; the buffer is retained
        // in this cache and destroyed exactly once in `destroy`.
        let buffer = unsafe { dev.device().create_buffer(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("GDS vkCreateBuffer: {e}")))?;
        // SAFETY: `buffer` is a live handle from this device.
        let req = unsafe { dev.device().get_buffer_memory_requirements(buffer) };
        let cleanup_buffer = |e| {
            // SAFETY: destroying the just-created, never-bound buffer.
            unsafe { dev.device().destroy_buffer(buffer, None) };
            e
        };
        let memory_type = dev
            .find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .map_err(cleanup_buffer)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(memory_type);
        // SAFETY: allocation size/type come from this buffer's requirements.
        let memory = unsafe { dev.device().allocate_memory(&alloc, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("GDS vkAllocateMemory: {e}")))
            .map_err(cleanup_buffer)?;
        let cleanup_both = |e| {
            // SAFETY: destroying the never-bound buffer and its allocation.
            unsafe {
                dev.device().destroy_buffer(buffer, None);
                dev.device().free_memory(memory, None);
            }
            e
        };
        // SAFETY: buffer and allocation are compatible live handles.
        unsafe { dev.device().bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("GDS vkBindBufferMemory: {e}")))
            .map_err(cleanup_both)?;
        // Zero the arena once — hardware GDS starts each session cold and the
        // shaders themselves initialize the counters they use.
        // SAFETY: host-visible coherent allocation, not in use by the GPU
        // (never yet bound to a descriptor), mapped for its full size.
        unsafe {
            let ptr = dev
                .device()
                .map_memory(memory, 0, GDS_SIZE as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| GpuError::VulkanInitFailed(format!("GDS vkMapMemory: {e}")))
                .map_err(cleanup_both)?;
            std::ptr::write_bytes(ptr.cast::<u8>(), 0, GDS_SIZE);
            dev.device().unmap_memory(memory);
        }
        self.gds = Some((buffer, memory));
        Ok(buffer)
    }

    /// The reusable command buffer + fence, with the fence reset for a new
    /// submission. Legal because every submission through it is synchronous:
    /// the caller waits the fence before the cache lock is released.
    pub(crate) fn submit_resources(
        &mut self,
        dev: &VulkanDevice,
    ) -> Result<(vk::CommandBuffer, vk::Fence), GpuError> {
        if self.submit.is_none() {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(dev.command_pool())
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the pool belongs to this device and access is serialized
            // by the cache lock (see module docs).
            let buffers = unsafe { dev.device().allocate_command_buffers(&alloc_info) }
                .map_err(|e| GpuError::VulkanInitFailed(format!("command buffer alloc: {e}")))?;
            let command_buffer = *buffers.first().ok_or_else(|| {
                GpuError::VulkanInitFailed("no command buffer returned".to_owned())
            })?;
            // SAFETY: plain unsignaled fence on a live device.
            let fence = unsafe {
                dev.device()
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateFence: {e}")))?;
            self.submit = Some((command_buffer, fence));
        }
        let (command_buffer, fence) = self.submit.expect("submit resources just ensured");
        // SAFETY: the previous submission that used this fence was waited to
        // completion before the cache lock was released (synchronous-draw
        // contract); resetting an unsignaled fence is also legal.
        unsafe { dev.device().reset_fences(&[fence]) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkResetFences: {e}")))?;
        Ok((command_buffer, fence))
    }

    /// The reusable descriptor pool, reset for a new draw/dispatch and grown
    /// when `max_sets`/`sizes` exceed its capacity. Growth merges the previous
    /// capacity so alternating draws with different descriptor mixes cannot
    /// thrash recreate.
    pub(crate) fn descriptor_pool(
        &mut self,
        dev: &VulkanDevice,
        max_sets: u32,
        sizes: &[vk::DescriptorPoolSize],
    ) -> Result<vk::DescriptorPool, GpuError> {
        let fits = self.pool.as_ref().is_some_and(|p| {
            p.max_sets >= max_sets
                && sizes.iter().all(|s| {
                    p.capacity.get(&s.ty.as_raw()).copied().unwrap_or(0) >= s.descriptor_count
                })
        });
        if fits {
            let pool = self.pool.as_ref().expect("fits implies a pool").pool;
            // SAFETY: no set allocated from this pool is referenced by pending
            // GPU work — the draw/dispatch that allocated them completed
            // synchronously before this acquisition (module-docs contract).
            unsafe {
                dev.device()
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
            }
            .map_err(|e| GpuError::PipelineCreationFailed(format!("vkResetDescriptorPool: {e}")))?;
            return Ok(pool);
        }

        // Grow: merge old per-type capacity with double the new requirement.
        let mut capacity = self
            .pool
            .as_ref()
            .map(|p| p.capacity.clone())
            .unwrap_or_default();
        for s in sizes {
            let entry = capacity.entry(s.ty.as_raw()).or_insert(0);
            *entry = (*entry).max((s.descriptor_count * 2).max(16));
        }
        let grown_max_sets = self
            .pool
            .as_ref()
            .map_or(0, |p| p.max_sets)
            .max((max_sets * 2).max(8));
        if let Some(old) = self.pool.take() {
            // SAFETY: same completed-work argument as the reset above.
            unsafe { dev.device().destroy_descriptor_pool(old.pool, None) };
        }
        let pool_sizes: Vec<_> = capacity
            .iter()
            .map(|(&ty, &count)| {
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::from_raw(ty))
                    .descriptor_count(count)
            })
            .collect();
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(grown_max_sets)
            .pool_sizes(&pool_sizes);
        // SAFETY: the pool-size slice is alive for the call; the pool is
        // retained in this cache and destroyed exactly once in `destroy`.
        let pool = unsafe { dev.device().create_descriptor_pool(&info, None) }.map_err(|e| {
            GpuError::PipelineCreationFailed(format!("vkCreateDescriptorPool: {e}"))
        })?;
        self.pool = Some(PoolState {
            pool,
            max_sets: grown_max_sets,
            capacity,
        });
        Ok(pool)
    }

    /// Acquire the persistent target for `key`, if one exists. The returned
    /// copy carries the entry's previous `synced` value; the entry itself is
    /// marked un-synced until [`Self::mark_target_synced`] confirms the draw's
    /// readback landed.
    pub(crate) fn acquire_target(&mut self, key: &TargetKey) -> Option<PersistentTarget> {
        let entry = self.targets.get_mut(key)?;
        let copy = *entry;
        entry.synced = false;
        self.stats.target_hits += 1;
        Some(copy)
    }

    /// Retain a freshly created persistent target (counts a miss).
    pub(crate) fn insert_target(&mut self, key: TargetKey, target: PersistentTarget) {
        self.stats.target_misses += 1;
        self.targets.insert(key, target);
    }

    /// Record that the draw into `key`'s target read its pixels back, so the
    /// GPU image and the CPU-side framebuffer entry are byte-identical again.
    pub(crate) fn mark_target_synced(&mut self, key: &TargetKey) {
        if let Some(target) = self.targets.get_mut(key) {
            target.synced = true;
        }
    }

    /// Destroy every persistent target at `base` whose extent/format differs
    /// from `keep` — the guest re-programmed the target, so the old-size image
    /// can never be drawn again.
    pub(crate) fn evict_targets_for_base(
        &mut self,
        dev: &VulkanDevice,
        base: u64,
        keep: &TargetKey,
    ) {
        let stale: Vec<TargetKey> = self
            .targets
            .keys()
            .filter(|k| k.base == base && *k != keep)
            .copied()
            .collect();
        for key in stale {
            if let Some(target) = self.targets.remove(&key) {
                // SAFETY: every draw that referenced this target completed
                // synchronously (fence waited) before this call — nothing on
                // the GPU can still name these handles.
                unsafe { destroy_target(dev.device(), &target) };
            }
        }
    }

    /// Destroy everything. Called from `VulkanDevice::drop` after
    /// `device_wait_idle`, before the command pool and device go away.
    pub(crate) fn destroy(&mut self, device: &ash::Device, command_pool: vk::CommandPool) {
        // SAFETY: the caller waited the device idle; every handle below was
        // created from `device` and is destroyed exactly once, children before
        // parents.
        unsafe {
            if let Some((command_buffer, fence)) = self.submit.take() {
                device.destroy_fence(fence, None);
                device.free_command_buffers(command_pool, &[command_buffer]);
            }
            if let Some(pool) = self.pool.take() {
                device.destroy_descriptor_pool(pool.pool, None);
            }
            if let Some((buffer, memory)) = self.gds.take() {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            for (_, pipeline) in self.graphics_pipelines.drain() {
                device.destroy_pipeline(pipeline, None);
            }
            for (_, pipeline) in self.compute_pipelines.drain() {
                device.destroy_pipeline(pipeline, None);
            }
            for (_, layout) in self.pipeline_layouts.drain() {
                device.destroy_pipeline_layout(layout, None);
            }
            for (_, layout) in self.set_layouts.drain() {
                device.destroy_descriptor_set_layout(layout, None);
            }
            for (_, module) in self.shader_modules.drain() {
                device.destroy_shader_module(module, None);
            }
            for (_, sampler) in self.samplers.drain() {
                device.destroy_sampler(sampler, None);
            }
            for (_, target) in self.targets.drain() {
                destroy_target(device, &target);
            }
        }
    }
}

/// Destroy one persistent target's handles.
///
/// # Safety
///
/// No submitted GPU work may still reference the target, and each handle must
/// be destroyed exactly once (the entry must already be out of the map).
unsafe fn destroy_target(device: &ash::Device, target: &PersistentTarget) {
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        device.destroy_image_view(target.view, None);
        device.destroy_image(target.image, None);
        device.free_memory(target.memory, None);
        device.destroy_buffer(target.readback_buffer, None);
        device.free_memory(target.readback_memory, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pipeline key must distinguish state that really feeds pipeline
    /// creation and must NOT include dynamic state (viewport/scissor/blend
    /// constants have no fields here at all — this is a compile-time property,
    /// the test documents the equality behavior).
    #[test]
    fn pipeline_keys_compare_by_creation_state() {
        let base = GraphicsPipelineKey {
            vs: 1,
            fs: 2,
            layout: 3,
            color_format: Some(vk::Format::R8G8B8A8_UNORM.as_raw()),
            depth: None,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST.as_raw(),
            cull: vk::CullModeFlags::NONE.as_raw(),
            front_face: vk::FrontFace::COUNTER_CLOCKWISE.as_raw(),
            color_write_mask: vk::ColorComponentFlags::RGBA.as_raw(),
            blend: BlendKey {
                enable: false,
                src_color: vk::BlendFactor::ONE.as_raw(),
                dst_color: vk::BlendFactor::ZERO.as_raw(),
                color_op: vk::BlendOp::ADD.as_raw(),
                src_alpha: vk::BlendFactor::ONE.as_raw(),
                dst_alpha: vk::BlendFactor::ZERO.as_raw(),
                alpha_op: vk::BlendOp::ADD.as_raw(),
            },
            vertex_bindings: vec![(0, 16)],
            vertex_attributes: vec![(0, 0, vk::Format::R32G32B32A32_SFLOAT.as_raw(), 0)],
        };
        assert_eq!(base, base.clone());

        let different_topology = GraphicsPipelineKey {
            topology: vk::PrimitiveTopology::TRIANGLE_STRIP.as_raw(),
            ..base.clone()
        };
        assert_ne!(base, different_topology);

        let different_shader = GraphicsPipelineKey {
            fs: 99,
            ..base.clone()
        };
        assert_ne!(base, different_shader);
    }

    #[test]
    fn target_keys_include_extent_and_format() {
        let a = TargetKey {
            base: 0x1000,
            width: 64,
            height: 64,
            format: vk::Format::R8G8B8A8_UNORM.as_raw(),
        };
        let b = TargetKey { width: 128, ..a };
        let c = TargetKey {
            format: vk::Format::B8G8R8A8_UNORM.as_raw(),
            ..a
        };
        assert_ne!(a, b);
        assert_ne!(a, c);
    }
}
