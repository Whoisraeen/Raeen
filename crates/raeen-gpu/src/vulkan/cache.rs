//! Long-lived Vulkan resource caches shared by every draw and dispatch on one
//! device (performance stage A).
//!
//! Before this cache existed, `render_draw` rebuilt **every** Vulkan resource
//! per draw — shader modules, descriptor set layouts, pipeline layout, the
//! pipeline itself, command buffer, fence, descriptor pool, and the render
//! target images — then destroyed them all. Measured on ASTRO.BOT
//! (`RAEEN_TIME_DRAW=1`, release): 1.6–16 ms of pure resource construction per
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
//! | guest texture image + view (stage D) | (guest base, extent, layers, depth, cube, format), content-validated per bind by a sparse sample-hash |
//! | command buffer + fence | singleton, reset per submission |
//! | descriptor pool | singleton, grown on demand, reset per acquisition |
//! | batch descriptor pools (stage D) | shared by a whole deferred batch, reset together at retire |
//!
//! Viewport, scissor, and blend constants are **dynamic** pipeline state, so
//! they deliberately do not key the pipeline.
//!
//! ## Thread ownership / locking
//!
//! The cache sits behind a `Mutex` on [`VulkanDevice`] and the lock is held for
//! the **entire** draw or dispatch (`render_draw` / `dispatch_compute`), which
//! also serializes queue submission. In production exactly one thread renders:
//! the per-process `raeen-gpu` worker consumes the submit queue single-file and
//! the session's `backend` mutex covers the inline-fallback path, so this lock
//! is uncontended; it exists so tests that drive a device from several threads
//! stay sound. Every use of a cached resource completes synchronously (the
//! fence is waited before the lock is released), which is what makes resetting
//! the fence, command buffer, and descriptor pool at the next acquisition
//! legal.

use super::instance::VulkanDevice;
use ash::vk::{self, Handle};
use raeen_core::error::GpuError;
use std::collections::HashMap;

/// Cache-effectiveness counters, cumulative since device creation.
///
/// `seed_uploads_skipped` counts draws whose attachment LOAD was satisfied
/// from the persistent GPU image instead of re-uploading the CPU-side
/// framebuffer — the stage A fast path for composited frames.
///
/// Stage B (deferred readback) adds:
/// - `deferred_draws`: draws submitted without a fence wait or readback.
/// - `batch_flushes`: how many times the deferred batch was flushed.
/// - `target_readbacks`: persistent-target readbacks performed at flushes —
///   the number stage B exists to shrink (was one per draw, now at most one
///   per touched target per flush).
/// - `sampled_target_binds`: sampled T#s satisfied by binding the persistent
///   `VkImage` directly instead of uploading CPU bytes.
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
    pub deferred_draws: u64,
    pub batch_flushes: u64,
    pub target_readbacks: u64,
    pub sampled_target_binds: u64,
    /// Upload-ring effectiveness (stage C item 3): a hit recycles a pooled
    /// host buffer; a miss pays vkCreateBuffer + vkAllocateMemory.
    pub host_pool_hits: u64,
    pub host_pool_misses: u64,
    /// Persistent-texture cache (stage D item 1): a hit binds a cached
    /// device-local image directly — no guest read, no detile, no
    /// vkCreateImage/vkAllocateMemory, no staging copy. A miss is a cacheable
    /// upload that created + donated a fresh image. An eviction is a cached
    /// image destroyed because its content hash changed, its key was
    /// re-programmed, or the byte cap forced out the least-recently-used.
    pub texture_cache_hits: u64,
    pub texture_cache_misses: u64,
    pub texture_cache_evictions: u64,
    /// Batch descriptor pools created (stage D item 2): deferred draws
    /// allocate from shared per-batch pools reset at retire, so this stops
    /// growing once a steady frame's worth of capacity exists.
    pub batch_pool_creates: u64,
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

/// What the persistent GPU image holds relative to the CPU-side framebuffer
/// entry for the same guest base. This is stage A's `synced` bit generalized
/// for deferred readback (stage B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetContent {
    /// GPU image and the last readback handed to the caller are
    /// byte-identical: the attachment may LOAD from the GPU copy and the CPU
    /// pixels are equally authoritative.
    Synced,
    /// The GPU image holds newer content than the CPU-side entry — draws were
    /// submitted whose readback was deferred. The GPU copy is the only
    /// authority: the attachment must LOAD from it and the stale CPU seed
    /// must NOT be uploaded over it.
    GpuNewer,
    /// Nothing is known (a draw is mid-flight or failed). The image must not
    /// be LOADed from; the next draw seeds from the CPU pixels or clears.
    Unknown,
}

/// The device-side half of one guest render target: the attachment image and
/// the host-visible buffer its pixels are read back through.
///
/// `content` is the honesty state for the seed-skip / deferred-readback fast
/// paths (see [`TargetContent`]). It is set to `Unknown` when a draw acquires
/// the target, `GpuNewer` when a deferred draw's commands were submitted, and
/// `Synced` only after a readback of the image lands, so a failed draw can
/// never leave a stale image masquerading as the composed frame.
#[derive(Clone, Copy)]
pub(crate) struct PersistentTarget {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub readback_buffer: vk::Buffer,
    pub readback_memory: vk::DeviceMemory,
    pub content: TargetContent,
}

/// A guest texture kept alive on the device across draws (stage D item 1).
///
/// The key is the SOURCE identity: the T#'s guest base address plus every
/// creation parameter of the Vulkan image (extent, layers, volume depth, cube
/// flag, format). Content freshness is NOT part of the key — it is validated
/// per bind by comparing [`PersistentTexture::sample_hash`] against a fresh
/// sparse hash of the guest bytes (see `draw_translate::guest_sample_hash`).
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) struct TextureKey {
    pub base: u64,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub depth: u32,
    pub cube: bool,
    pub format: i32,
}

/// The device half of one cached guest texture. The image rests in
/// `SHADER_READ_ONLY_OPTIMAL` between draws (the upload's tail barrier put it
/// there with visibility to both graphics shader stages), so a cache hit
/// binds the view with **no** barrier at all.
///
/// ## Invalidation contract (documented, deliberate)
///
/// - Every bind re-hashes a sparse ~4 KiB sample of the guest source bytes;
///   a mismatch is a miss (the entry is evicted and re-uploaded). CPU guest
///   writes that leave every sampled chunk byte-identical are therefore NOT
///   detected until any sampled byte changes — that is the staleness window,
///   bounded by the sample coverage.
/// - Writeback paths we control (compute storage writeback, DMA copies,
///   render-target readbacks) do NOT proactively invalidate: no cheap range
///   index exists over this cache today, so they too are covered by the
///   per-bind rehash above.
/// - `RAEEN_NO_TEX_CACHE=1` is the full-honesty escape hatch: every draw
///   decodes and uploads its textures per draw, the pre-stage-D behaviour.
#[derive(Clone, Copy)]
pub(crate) struct PersistentTexture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    /// Sparse sample-hash of the guest source bytes at upload time.
    pub sample_hash: u64,
    /// Decoded byte size (cap accounting).
    pub bytes: u64,
    /// LRU stamp from [`DrawCaches::texture_entry`] / insertion.
    pub last_use: u64,
}

/// The grown-on-demand descriptor pool (see [`DrawCaches::descriptor_pool`]).
struct PoolState {
    pool: vk::DescriptorPool,
    max_sets: u32,
    /// Capacity per raw `VkDescriptorType`.
    capacity: HashMap<i32, u32>,
}

/// One shared descriptor pool for deferred-batch draws (stage D item 2).
/// Sets allocated from it stay live until the batch flush, so the pool is
/// never reset per draw — capacity is accounted here and every batch pool is
/// reset together in [`DrawCaches::retire_batch`], after the flush fence.
struct BatchPoolState {
    pool: vk::DescriptorPool,
    total_sets: u32,
    free_sets: u32,
    /// Full capacity per raw `VkDescriptorType`.
    capacity: HashMap<i32, u32>,
    /// Remaining capacity per raw `VkDescriptorType`.
    free: HashMap<i32, u32>,
}

/// Per-draw resources whose destruction is deferred until the batch fence:
/// a deferred draw's GPU work is still pending when the draw call returns, so
/// everything its command buffer references must stay alive until the flush
/// waits the fence.
#[derive(Default)]
pub(crate) struct PendingDrawResources {
    /// The draw's own primary command buffer (recycled at flush).
    pub command_buffer: vk::CommandBuffer,
    /// The draw's own descriptor pool (deferred draws never touch the shared
    /// resettable pool — its per-draw reset is illegal while sets are live).
    pub descriptor_pool: vk::DescriptorPool,
    /// Buffers: vertex/index/storage/upload/staging.
    pub buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// Images with their view: texture uploads, depth attachments.
    pub images: Vec<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
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
    /// Persistent guest textures (stage D item 1) — see [`PersistentTexture`]
    /// for the invalidation contract.
    textures: HashMap<TextureKey, PersistentTexture>,
    /// Total decoded bytes held by `textures`, for the cap.
    texture_bytes: u64,
    /// Monotonic LRU clock for `textures`.
    texture_clock: u64,
    /// Cached texture images evicted while pending command buffers may still
    /// reference them; destroyed at the batch retire (post-fence).
    deferred_image_destroys: Vec<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
    /// Shared descriptor pools for deferred-batch draws (stage D item 2).
    batch_pools: Vec<BatchPoolState>,
    /// One command buffer + fence, reused for every synchronous submission.
    submit: Option<(vk::CommandBuffer, vk::Fence)>,
    pool: Option<PoolState>,
    /// The device-persistent GDS arena (see [`DrawCaches::gds_buffer`]).
    gds: Option<(vk::Buffer, vk::DeviceMemory)>,
    /// Deferred draws whose GPU work is submitted but not yet fence-waited.
    pending: Vec<PendingDrawResources>,
    /// Targets drawn by the pending deferred draws, in draw order (last draw
    /// of a target keeps it at the back). Flush reads each back exactly once.
    touched: Vec<TargetKey>,
    /// Recycled primary command buffers for deferred draws.
    free_command_buffers: Vec<vk::CommandBuffer>,
    /// Targets evicted while a batch was open: their images may still be
    /// referenced by pending command buffers, so destruction waits for the
    /// flush's fence.
    deferred_target_destroys: Vec<PersistentTarget>,
    /// Upload ring (stage C item 3): recycled HOST_VISIBLE|HOST_COHERENT
    /// buffers for per-draw guest data (vertex/index/storage/seed/texture
    /// staging). Free entries keyed by capacity (a power of two); a draw takes
    /// the smallest entry that fits and creation only happens on a pool miss.
    /// Reuse is fence-tracked by construction: buffers return here only after
    /// the immediate draw's fence wait (`Resources::Drop`) or the batch flush
    /// fence (`retire_batch`).
    host_pool_free: std::collections::BTreeMap<u64, Vec<(vk::Buffer, vk::DeviceMemory)>>,
    /// Capacity registry for every live pooled buffer (in use or free), keyed
    /// by the raw buffer handle — how `release_host_buffer` distinguishes a
    /// pooled buffer (recycle) from an ad-hoc one (destroy) and knows which
    /// size class it returns to.
    host_pool_capacity: HashMap<u64, u64>,
    /// Total bytes sitting FREE in the pool, for the recycle cap.
    host_pool_free_bytes: u64,
    pub stats: DrawCacheStats,
}

/// Free-pool cap: pooled buffers beyond this many bytes are destroyed on
/// release instead of recycled, so a burst of huge uploads cannot pin
/// host memory forever.
const HOST_POOL_FREE_CAP: u64 = 256 * 1024 * 1024;

/// Persistent-texture cache cap (decoded bytes). Over it, least-recently-used
/// entries are evicted at insertion, so a title streaming many unique
/// textures cannot pin device memory without bound.
const TEXTURE_CACHE_CAP: u64 = 256 * 1024 * 1024;

/// Usage union for pooled host buffers. One pool serves every per-draw guest
/// upload, so each buffer carries the union of the usages those uploads need;
/// extra usage bits on a buffer are free on desktop drivers.
pub(crate) const HOST_POOL_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw(),
);

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

    /// Take a pooled host-visible buffer with capacity >= `size` (smallest
    /// fitting size class), or create one (capacity = next power of two).
    /// The returned buffer is registered; hand it back through
    /// [`Self::release_host_buffer`] once no submitted GPU work references it.
    pub(crate) fn acquire_host_buffer(
        &mut self,
        dev: &VulkanDevice,
        size: u64,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        debug_assert!(size > 0, "zero-sized host buffer request");
        // Smallest free size class that fits. Don't hand a small request a
        // giant buffer (>= 8x) — that starves the big classes and bloats
        // descriptor ranges' backing for nothing.
        let fitting = if std::env::var_os("RAEEN_NO_HOST_POOL").is_none() {
            self.host_pool_free
                .range(size..)
                .find(|(cap, list)| **cap < size.saturating_mul(8) && !list.is_empty())
                .map(|(cap, _)| *cap)
        } else {
            None
        };
        if let Some(cap) = fitting {
            let list = self
                .host_pool_free
                .get_mut(&cap)
                .expect("size class just found");
            let entry = list.pop().expect("size class was non-empty");
            if list.is_empty() {
                self.host_pool_free.remove(&cap);
            }
            self.host_pool_free_bytes -= cap;
            self.stats.host_pool_hits += 1;
            return Ok(entry);
        }
        let capacity = size.next_power_of_two().max(256);
        let (buffer, memory) = create_host_buffer(dev, capacity)?;
        self.host_pool_capacity.insert(buffer.as_raw(), capacity);
        self.stats.host_pool_misses += 1;
        Ok((buffer, memory))
    }

    /// Return a buffer to the pool (if it came from [`Self::acquire_host_buffer`])
    /// or destroy it (ad-hoc buffers — readback/depth buffers — route through
    /// here too so call sites need not distinguish).
    ///
    /// # Safety contract (checked by the caller)
    ///
    /// No submitted GPU work may still reference the buffer: immediate draws
    /// waited their fence, batched draws are released only by
    /// [`Self::retire_batch`] after the flush fence.
    pub(crate) fn release_host_buffer(
        &mut self,
        dev: &VulkanDevice,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
    ) {
        if buffer == vk::Buffer::null() && memory == vk::DeviceMemory::null() {
            return;
        }
        if let Some(&capacity) = self.host_pool_capacity.get(&buffer.as_raw()) {
            if std::env::var_os("RAEEN_NO_HOST_POOL").is_none()
                && self.host_pool_free_bytes + capacity <= HOST_POOL_FREE_CAP
            {
                self.host_pool_free
                    .entry(capacity)
                    .or_default()
                    .push((buffer, memory));
                self.host_pool_free_bytes += capacity;
                return;
            }
            // Over the cap: fall through to destruction.
            self.host_pool_capacity.remove(&buffer.as_raw());
        }
        // SAFETY: caller guarantees no pending GPU work references the pair;
        // both handles were created from this device and are destroyed once.
        unsafe {
            if buffer != vk::Buffer::null() {
                dev.device().destroy_buffer(buffer, None);
            }
            if memory != vk::DeviceMemory::null() {
                dev.device().free_memory(memory, None);
            }
        }
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
    /// copy carries the entry's previous `content` value; the entry itself is
    /// marked [`TargetContent::Unknown`] until either
    /// [`Self::mark_target_synced`] (a readback landed) or
    /// [`Self::commit_deferred_draw`] (commands submitted, readback deferred)
    /// records what really happened.
    pub(crate) fn acquire_target(&mut self, key: &TargetKey) -> Option<PersistentTarget> {
        let entry = self.targets.get_mut(key)?;
        let copy = *entry;
        entry.content = TargetContent::Unknown;
        self.stats.target_hits += 1;
        Some(copy)
    }

    /// Retain a freshly created persistent target (counts a miss).
    pub(crate) fn insert_target(&mut self, key: TargetKey, target: PersistentTarget) {
        self.stats.target_misses += 1;
        self.targets.insert(key, target);
    }

    /// The cached texture for `key`, if one is live: its view (bindable with
    /// no barrier — the image rests in `SHADER_READ_ONLY_OPTIMAL`) and the
    /// sample-hash its content was uploaded under. Stamps the LRU clock.
    pub(crate) fn texture_entry(&mut self, key: &TextureKey) -> Option<(vk::ImageView, u64)> {
        self.texture_clock += 1;
        let clock = self.texture_clock;
        let entry = self.textures.get_mut(key)?;
        entry.last_use = clock;
        Some((entry.view, entry.sample_hash))
    }

    /// Snapshot of every cached texture's (key, sample-hash) — published to
    /// the texture-decode path so it can skip the guest read + detile for a
    /// texture whose fresh sample-hash matches the cached one.
    pub(crate) fn cached_texture_hashes(&self) -> Vec<(TextureKey, u64)> {
        self.textures
            .iter()
            .map(|(k, t)| (*k, t.sample_hash))
            .collect()
    }

    /// Retain a freshly uploaded texture. Replaces (and destroys, deferred if
    /// a batch is open) any previous entry at the same key, then enforces the
    /// byte cap by evicting least-recently-used entries.
    ///
    /// Call only after the upload's command buffer was SUBMITTED (batched) or
    /// fence-completed (immediate): eviction safety here relies on
    /// [`Self::batch_open`] seeing every command buffer that references a
    /// cached image, and insert-on-success keeps images whose upload never
    /// ran out of the cache entirely.
    pub(crate) fn insert_texture(
        &mut self,
        dev: &VulkanDevice,
        key: TextureKey,
        mut texture: PersistentTexture,
    ) {
        if let Some(old) = self.textures.remove(&key) {
            self.texture_bytes -= old.bytes;
            self.destroy_texture_when_safe(dev, old);
            self.stats.texture_cache_evictions += 1;
        }
        while self.texture_bytes.saturating_add(texture.bytes) > TEXTURE_CACHE_CAP {
            let Some((&lru_key, _)) = self.textures.iter().min_by_key(|(_, t)| t.last_use) else {
                break;
            };
            let old = self
                .textures
                .remove(&lru_key)
                .expect("LRU key was just found in the map");
            self.texture_bytes -= old.bytes;
            self.destroy_texture_when_safe(dev, old);
            self.stats.texture_cache_evictions += 1;
        }
        self.texture_clock += 1;
        texture.last_use = self.texture_clock;
        self.texture_bytes += texture.bytes;
        self.textures.insert(key, texture);
    }

    /// Destroy an evicted texture's handles now if no deferred batch is open
    /// (every command buffer that referenced them fence-completed), otherwise
    /// park them for the batch retire.
    fn destroy_texture_when_safe(&mut self, dev: &VulkanDevice, texture: PersistentTexture) {
        if self.batch_open() {
            self.deferred_image_destroys
                .push((texture.image, texture.memory, texture.view));
        } else {
            // SAFETY: no batch is open, so every submitted command buffer that
            // could reference these handles completed under its own fence;
            // the entry already left the map, so this is the sole handle set.
            unsafe {
                let d = dev.device();
                d.destroy_image_view(texture.view, None);
                d.destroy_image(texture.image, None);
                d.free_memory(texture.memory, None);
            }
        }
    }

    /// A descriptor pool for one deferred draw's sets (stage D item 2): the
    /// first existing batch pool with enough remaining capacity, or a new
    /// generously sized pool. Sets stay live until the flush; every batch
    /// pool is reset together in [`Self::retire_batch`].
    ///
    /// A draw that fails after allocating leaks its sets' capacity until the
    /// next retire reset — conservative (capacity, not memory) and safe.
    pub(crate) fn batch_descriptor_pool(
        &mut self,
        dev: &VulkanDevice,
        max_sets: u32,
        sizes: &[vk::DescriptorPoolSize],
    ) -> Result<vk::DescriptorPool, GpuError> {
        'pools: for pool in &mut self.batch_pools {
            if pool.free_sets < max_sets {
                continue;
            }
            for s in sizes {
                if pool.free.get(&s.ty.as_raw()).copied().unwrap_or(0) < s.descriptor_count {
                    continue 'pools;
                }
            }
            pool.free_sets -= max_sets;
            for s in sizes {
                *pool
                    .free
                    .get_mut(&s.ty.as_raw())
                    .expect("capacity was just checked") -= s.descriptor_count;
            }
            return Ok(pool.pool);
        }
        // No pool fits: create one sized for a whole batch of draws like this
        // (32x the request), so a steady frame settles on one pool.
        let mut capacity: HashMap<i32, u32> = HashMap::new();
        for s in sizes {
            let entry = capacity.entry(s.ty.as_raw()).or_insert(0);
            *entry = entry.saturating_add(s.descriptor_count);
        }
        for count in capacity.values_mut() {
            *count = count.saturating_mul(32).max(64);
        }
        let total_sets = max_sets.saturating_mul(32).max(64);
        let pool_sizes: Vec<_> = capacity
            .iter()
            .map(|(&ty, &count)| {
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::from_raw(ty))
                    .descriptor_count(count)
            })
            .collect();
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(total_sets)
            .pool_sizes(&pool_sizes);
        // SAFETY: the pool-size slice is alive for the call; the pool is
        // retained in `batch_pools` and destroyed exactly once in `destroy`.
        let pool = unsafe { dev.device().create_descriptor_pool(&info, None) }.map_err(|e| {
            GpuError::PipelineCreationFailed(format!("batch vkCreateDescriptorPool: {e}"))
        })?;
        let mut free = capacity.clone();
        for s in sizes {
            *free
                .get_mut(&s.ty.as_raw())
                .expect("capacity covers every requested type") -= s.descriptor_count;
        }
        self.batch_pools.push(BatchPoolState {
            pool,
            total_sets,
            free_sets: total_sets - max_sets,
            capacity,
            free,
        });
        self.stats.batch_pool_creates += 1;
        Ok(pool)
    }

    /// Record that the draw into `key`'s target read its pixels back, so the
    /// GPU image and the CPU-side framebuffer entry are byte-identical again.
    pub(crate) fn mark_target_synced(&mut self, key: &TargetKey) {
        if let Some(target) = self.targets.get_mut(key) {
            target.content = TargetContent::Synced;
        }
    }

    /// The persistent image + view for `key`, if one is live — used to bind a
    /// render target directly as a sampled descriptor (stage B).
    pub(crate) fn target_image(&self, key: &TargetKey) -> Option<(vk::Image, vk::ImageView)> {
        self.targets.get(key).map(|t| (t.image, t.view))
    }

    /// A copy of the whole persistent-target entry for `key` (flush readback).
    pub(crate) fn target_entry(&self, key: &TargetKey) -> Option<PersistentTarget> {
        self.targets.get(key).copied()
    }

    /// Degrade `key`'s content to [`TargetContent::Unknown`] (a flush failed:
    /// the image may or may not hold the batch's draws).
    pub(crate) fn mark_target_unknown(&mut self, key: &TargetKey) {
        if let Some(target) = self.targets.get_mut(key) {
            target.content = TargetContent::Unknown;
        }
    }

    /// Every live persistent-target key whose image content is trustworthy
    /// (not mid-draw / not post-failure). `draw_translate` snapshots this to
    /// decide which sampled T#s can bind the GPU image directly — an
    /// `Unknown` target's image may still be layout-UNDEFINED (its creating
    /// draw failed), so it must not be bound.
    pub(crate) fn live_target_keys(&self) -> Vec<TargetKey> {
        self.targets
            .iter()
            .filter(|(_, t)| t.content != TargetContent::Unknown)
            .map(|(k, _)| *k)
            .collect()
    }

    /// Whether any deferred draw is awaiting its flush, or any touched target
    /// still holds GPU-side content that was never read back (a scanout-
    /// filtered flush read only the flipped target and re-queued the rest).
    pub(crate) fn batch_open(&self) -> bool {
        !self.pending.is_empty() || !self.touched.is_empty()
    }

    /// Put back touched targets a scanout-filtered flush chose NOT to read.
    /// Their GPU images remain the sole content authority
    /// ([`TargetContent::GpuNewer`]); a later full flush (flip miss, frame
    /// dump, feedback fallback, wait_idle) reads them then. Runs under the
    /// same cache lock as the `take_batch` that emptied the list, so no draw
    /// can interleave.
    pub(crate) fn requeue_touched(&mut self, keys: Vec<TargetKey>) {
        for key in keys {
            if !self.touched.contains(&key) {
                self.touched.push(key);
            }
        }
    }

    /// Whether `base` has deferred (not yet read back) draws pending.
    pub(crate) fn base_is_batch_dirty(&self, base: u64) -> bool {
        self.touched.iter().any(|k| k.base == base)
    }

    /// A primary command buffer for one deferred draw: recycled from the free
    /// list, or freshly allocated. Not yet in the recording state.
    pub(crate) fn batch_command_buffer(
        &mut self,
        dev: &VulkanDevice,
    ) -> Result<vk::CommandBuffer, GpuError> {
        if let Some(cb) = self.free_command_buffers.pop() {
            return Ok(cb);
        }
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(dev.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool belongs to this device and access is serialized by
        // the cache lock (see module docs).
        let buffers = unsafe { dev.device().allocate_command_buffers(&alloc_info) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("batch command buffer alloc: {e}")))?;
        buffers.first().copied().ok_or_else(|| {
            GpuError::VulkanInitFailed("no batch command buffer returned".to_owned())
        })
    }

    /// Return a command buffer acquired via [`Self::batch_command_buffer`]
    /// that was never submitted (the draw failed before submission).
    pub(crate) fn recycle_command_buffer(&mut self, cb: vk::CommandBuffer) {
        self.free_command_buffers.push(cb);
    }

    /// Record a successfully submitted deferred draw: its per-draw resources
    /// join the pending list, the target joins the touched list (moved to the
    /// back so flush order follows last-draw order), and the target's GPU
    /// image becomes the sole content authority.
    pub(crate) fn commit_deferred_draw(&mut self, res: PendingDrawResources, key: TargetKey) {
        self.pending.push(res);
        self.touched.retain(|k| *k != key);
        self.touched.push(key);
        if let Some(target) = self.targets.get_mut(&key) {
            target.content = TargetContent::GpuNewer;
        }
        self.stats.deferred_draws += 1;
    }

    /// Take the whole pending batch for a flush: per-draw resources, touched
    /// targets (draw order), and targets whose destruction was deferred.
    pub(crate) fn take_batch(
        &mut self,
    ) -> (
        Vec<PendingDrawResources>,
        Vec<TargetKey>,
        Vec<PersistentTarget>,
    ) {
        (
            std::mem::take(&mut self.pending),
            std::mem::take(&mut self.touched),
            std::mem::take(&mut self.deferred_target_destroys),
        )
    }

    /// Destroy (or recycle) one flushed batch's resources. The caller must
    /// have waited the flush fence — nothing on the GPU references them.
    pub(crate) fn retire_batch(
        &mut self,
        dev: &VulkanDevice,
        pending: Vec<PendingDrawResources>,
        evicted: Vec<PersistentTarget>,
    ) {
        for res in pending {
            if res.command_buffer != vk::CommandBuffer::null() {
                self.free_command_buffers.push(res.command_buffer);
            }
            // Pooled host buffers go back to the upload ring (the flush fence
            // was waited, so nothing on the GPU references them); ad-hoc
            // buffers are destroyed inside release_host_buffer.
            for (buffer, memory) in res.buffers {
                self.release_host_buffer(dev, buffer, memory);
            }
            // SAFETY: the flush fence was waited; every handle below was
            // created for the retired draw alone and is destroyed exactly
            // once, children before parents.
            unsafe {
                let d = dev.device();
                if res.descriptor_pool != vk::DescriptorPool::null() {
                    d.destroy_descriptor_pool(res.descriptor_pool, None);
                }
                for (image, memory, view) in res.images {
                    if view != vk::ImageView::null() {
                        d.destroy_image_view(view, None);
                    }
                    if image != vk::Image::null() {
                        d.destroy_image(image, None);
                    }
                    if memory != vk::DeviceMemory::null() {
                        d.free_memory(memory, None);
                    }
                }
            }
        }
        for target in evicted {
            // SAFETY: fence waited (above); the entry left the map when it
            // was evicted, so this is the sole remaining handle set.
            unsafe { destroy_target(dev.device(), &target) };
        }
        // Evicted cached textures parked while the batch was open: the flush
        // fence covered every command buffer that referenced them.
        for (image, memory, view) in std::mem::take(&mut self.deferred_image_destroys) {
            // SAFETY: fence waited; the entries left the texture map at
            // eviction, so these are the sole remaining handles.
            unsafe {
                let d = dev.device();
                d.destroy_image_view(view, None);
                d.destroy_image(image, None);
                d.free_memory(memory, None);
            }
        }
        // Batch descriptor pools: every set allocated from them belonged to
        // the just-retired draws, whose fence was waited — reset for the next
        // batch and restore full capacity.
        for pool in &mut self.batch_pools {
            // SAFETY: same fence-waited argument; resetting frees the sets.
            if let Err(e) = unsafe {
                dev.device()
                    .reset_descriptor_pool(pool.pool, vk::DescriptorPoolResetFlags::empty())
            } {
                // Reset cannot fail per spec except device loss; keep the
                // accounting honest (capacity stays consumed) and say so.
                tracing::warn!("batch vkResetDescriptorPool failed: {e}");
                continue;
            }
            pool.free_sets = pool.total_sets;
            pool.free = pool.capacity.clone();
        }
    }

    /// Destroy every persistent target at `base` whose extent/format differs
    /// from `keep` — the guest re-programmed the target, so the old-size image
    /// can never be drawn again. With a deferred batch open the handles may
    /// still be referenced by pending command buffers, so their destruction
    /// waits for the flush.
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
                if self.batch_open() {
                    self.deferred_target_destroys.push(target);
                } else {
                    // SAFETY: every draw that referenced this target completed
                    // synchronously (fence waited) before this call — nothing
                    // on the GPU can still name these handles.
                    unsafe { destroy_target(dev.device(), &target) };
                }
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
            // A pending deferred batch at device teardown (e.g. a session shut
            // down mid-submission): device_wait_idle already ran, so the
            // pending work completed and the handles can go.
            for res in self.pending.drain(..) {
                if res.command_buffer != vk::CommandBuffer::null() {
                    device.free_command_buffers(command_pool, &[res.command_buffer]);
                }
                if res.descriptor_pool != vk::DescriptorPool::null() {
                    device.destroy_descriptor_pool(res.descriptor_pool, None);
                }
                for (buffer, memory) in res.buffers {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                for (image, memory, view) in res.images {
                    if view != vk::ImageView::null() {
                        device.destroy_image_view(view, None);
                    }
                    if image != vk::Image::null() {
                        device.destroy_image(image, None);
                    }
                    if memory != vk::DeviceMemory::null() {
                        device.free_memory(memory, None);
                    }
                }
            }
            self.touched.clear();
            for target in self.deferred_target_destroys.drain(..) {
                destroy_target(device, &target);
            }
            if !self.free_command_buffers.is_empty() {
                device.free_command_buffers(command_pool, &self.free_command_buffers);
                self.free_command_buffers.clear();
            }
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
            for (_, texture) in self.textures.drain() {
                device.destroy_image_view(texture.view, None);
                device.destroy_image(texture.image, None);
                device.free_memory(texture.memory, None);
            }
            self.texture_bytes = 0;
            for (image, memory, view) in self.deferred_image_destroys.drain(..) {
                device.destroy_image_view(view, None);
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            for pool in self.batch_pools.drain(..) {
                device.destroy_descriptor_pool(pool.pool, None);
            }
            // Upload ring: free entries are owned solely by the pool. In-use
            // pooled buffers were already destroyed above through the pending
            // list (or by `Resources::Drop` before teardown).
            for (_, list) in std::mem::take(&mut self.host_pool_free) {
                for (buffer, memory) in list {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
            }
            self.host_pool_capacity.clear();
            self.host_pool_free_bytes = 0;
        }
    }
}

/// Create one pooled host-visible|coherent buffer of `capacity` bytes with the
/// pool's usage union.
fn create_host_buffer(
    dev: &VulkanDevice,
    capacity: u64,
) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
    let info = vk::BufferCreateInfo::default()
        .size(capacity)
        .usage(HOST_POOL_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `info` is fully initialized; the device is live. The handle is
    // registered in the pool and destroyed exactly once (release over cap,
    // teardown, or its owner's Drop via release_host_buffer).
    let buffer = unsafe { dev.device().create_buffer(&info, None) }
        .map_err(|e| GpuError::VulkanInitFailed(format!("pool vkCreateBuffer: {e}")))?;
    // SAFETY: `buffer` was just created from this device.
    let reqs = unsafe { dev.device().get_buffer_memory_requirements(buffer) };
    let cleanup = |e| {
        // SAFETY: destroying the just-created, never-bound buffer.
        unsafe { dev.device().destroy_buffer(buffer, None) };
        e
    };
    let type_index = dev
        .find_memory_type(
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .map_err(cleanup)?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(type_index);
    // SAFETY: allocation size/type come from this buffer's own requirements.
    let memory = unsafe { dev.device().allocate_memory(&alloc, None) }
        .map_err(|e| GpuError::VulkanInitFailed(format!("pool vkAllocateMemory: {e}")))
        .map_err(cleanup)?;
    // SAFETY: memory was allocated for exactly this buffer; offset 0 is in
    // range and aligned by construction.
    if let Err(e) = unsafe { dev.device().bind_buffer_memory(buffer, memory, 0) } {
        // SAFETY: unwinding our own two handles, neither yet in use.
        unsafe {
            dev.device().free_memory(memory, None);
            dev.device().destroy_buffer(buffer, None);
        }
        return Err(GpuError::VulkanInitFailed(format!(
            "pool vkBindBufferMemory: {e}"
        )));
    }
    Ok((buffer, memory))
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
