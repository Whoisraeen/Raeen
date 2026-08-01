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

use super::{instance::VulkanDevice, offscreen::SamplerState};
use ash::vk::{self, Handle};
use raeen_core::error::GpuError;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Weak},
};

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
    /// Persistent depth/stencil attachment reuse. A hit avoids
    /// `vkCreateImage` + allocation + view creation for a draw that reuses
    /// the same guest `DB_Z_WRITE_BASE`, extent, and format.
    pub depth_target_hits: u64,
    pub depth_target_misses: u64,
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
    /// Persistent compute-UAV effectiveness. A hit reuses the device image,
    /// view, allocation, and readback buffer. A new per-submission snapshot
    /// still uploads current guest bytes; repeated ordered dispatches sharing
    /// that exact snapshot retain the GPU-newer contents in-place.
    pub compute_image_hits: u64,
    pub compute_image_misses: u64,
    pub compute_image_evictions: u64,
    /// Persistent UAV binds that kept the GPU-newer ordered contents instead
    /// of staging the same per-submission seed again.
    pub compute_image_uploads_skipped: u64,
    /// Guest-addressed compute SSBOs retained across dispatches. A hit avoids
    /// allocation; `compute_buffer_uploads_skipped` additionally means the
    /// complete guest snapshot matched the cache's authoritative shadow, so
    /// no map/copy upload was needed.
    pub compute_buffer_hits: u64,
    pub compute_buffer_misses: u64,
    pub compute_buffer_evictions: u64,
    pub compute_buffer_uploads_skipped: u64,
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

/// Successful cross-stage interface checks.  The check walks both SPIR-V
/// modules and builds temporary location maps, so doing it before every
/// pipeline-cache lookup made a cache hit surprisingly expensive.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
struct GraphicsInterfaceKey {
    vs: u64,
    fs: u64,
    topology: i32,
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
    /// (binding, stride, input rate).
    pub vertex_bindings: Vec<(u32, u32, i32)>,
    /// (location, binding, format, offset).
    pub vertex_attributes: Vec<(u32, u32, i32, u32)>,
    /// Extra MRT attachments: (format, write mask, blend), in attachment
    /// order after the primary. Empty for a single-target pipeline.
    pub mrt: Vec<(i32, u32, BlendKey)>,
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

/// Last successfully recorded layout of a persistent colour target.
///
/// Deferred draws keep an image attachment-resident until it is actually
/// sampled or copied to the host. `Undefined` means the prior contents/layout
/// are not trustworthy and the next writer must discard them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetLayout {
    Undefined,
    TransferSrc,
    ColorAttachment,
}

/// Last successfully recorded layout of a persistent depth/stencil target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DepthTargetLayout {
    Undefined,
    TransferSrc,
    DepthStencilAttachment,
    DepthStencilReadOnly,
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
    pub layout: TargetLayout,
}

/// Cached host-owned destination for an ABI-v3 GPU present pass.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) struct GpuPresentKey {
    pub source_base: u64,
    pub width: u32,
    pub height: u32,
    pub format: i32,
}

#[derive(Clone, Copy)]
pub(crate) struct GpuPresentTarget {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub readback_buffer: vk::Buffer,
    pub readback_memory: vk::DeviceMemory,
    pub layout: vk::ImageLayout,
}

/// Identity of one guest depth/stencil surface. Like [`TargetKey`], creation
/// parameters are part of the key so reprogramming an address cannot serve an
/// incompatible Vulkan image.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) struct DepthTargetKey {
    pub base: u64,
    pub width: u32,
    pub height: u32,
    pub format: i32,
}

/// Device-side depth/stencil attachment retained across draws.
#[derive(Clone, Copy)]
pub(crate) struct PersistentDepthTarget {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub readback_buffer: vk::Buffer,
    pub readback_memory: vk::DeviceMemory,
    pub layout: DepthTargetLayout,
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
    /// 2DArray (T# type 13) view intent — part of the key so a cached
    /// `VK_IMAGE_VIEW_TYPE_2D_ARRAY` view is never served under a plain-2D
    /// descriptor (or vice-versa) when both share the same base/extent/format
    /// with a single layer. See `TextureUpload::array`.
    pub array: bool,
    /// 3D volume (T# type 10) view intent — part of the key for exactly the
    /// [`Self::array`] reason: a one-slice volume has `depth == 1`, so without
    /// this a `VK_IMAGE_TYPE_3D` image would be served under a plain-2D
    /// descriptor sharing the same base/extent/format. See
    /// `TextureUpload::volume`.
    pub volume: bool,
    pub format: i32,
}

fn texture_aliases_compute_image(key: &TextureKey, base: u64) -> bool {
    base != 0 && key.base == base
}

/// The device half of one cached guest texture. The image rests in
/// `SHADER_READ_ONLY_OPTIMAL` between draws (the upload's tail barrier put it
/// there with visibility to both graphics shader stages), so a cache hit
/// binds the view with **no** barrier at all.
///
/// ## Invalidation contract (documented, deliberate)
///
/// - Every bind re-hashes a sparse sample of the guest source bytes; a mismatch
///   is a miss (the entry is evicted and re-uploaded). Atlas-sized resources
///   additionally rotate an exact contiguous audit across the whole source,
///   bounding sparse-probe misses to 64 submissions. See
///   `draw_translate::GuestTextureHashAuditor`.
/// - A completed compute storage-image write invalidates cached sampled views
///   at the same guest base. This covers format-reinterpreted aliases (for
///   example an `R32_UINT` UAV later sampled as `R8_UNORM`) without a range
///   index. Other writes still use the per-bind rehash above.
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

/// Identity of a compute storage image whose Vulkan allocation can be reused.
///
/// `base` is included because the guest address is the resource's stable
/// identity across dispatches. Shape and format stay in the key so a title
/// reprogramming one address with a different view can never receive an
/// incompatible image.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) struct ComputeImageKey {
    pub base: u64,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layers: u32,
    pub array: bool,
    /// 3D volume (T# type 10) view intent — see `TextureKey::volume`.
    pub volume: bool,
    pub format: i32,
}

/// Device-lifetime half of one compute UAV. The dispatch owns only its upload
/// staging buffer; these handles remain in [`DrawCaches`] until eviction.
#[derive(Clone)]
pub(crate) struct PersistentComputeImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub readback_buffer: vk::Buffer,
    pub readback_memory: vk::DeviceMemory,
    pub bytes: u64,
    pub last_use: u64,
    /// Weak identity of the decoded per-submission seed last uploaded. While
    /// it upgrades and matches, later ordered dispatches must consume the
    /// GPU-newer image rather than overwrite it with stale guest bytes.
    pub last_snapshot: Weak<Vec<u8>>,
}

/// Guest publication metadata for a storage image written by one or more
/// deferred compute dispatches. The persistent image's readback buffer is
/// overwritten in queue order, so one entry per key publishes the final
/// dispatch result after the shared batch fence.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ComputeImageWriteback {
    pub key: ComputeImageKey,
    pub tile_mode: u8,
    pub texel: u32,
}

/// Stable identity of a guest compute SSBO. The padded descriptor byte size is
/// part of the key so reprogramming an address with a different range cannot
/// alias an incompatible Vulkan buffer.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) struct ComputeBufferKey {
    pub base: u64,
    pub size: usize,
}

/// Host-visible compute SSBO plus the exact bytes it currently contains.
///
/// The complete shadow is the cache's invalidation contract: every bind
/// compares the freshly captured guest bytes against it. This deliberately
/// avoids the sampled-hash shortcut used by read-only textures, which would
/// be unsound for writable/global-memory shader resources.
pub(crate) struct PersistentComputeBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub shadow: Vec<u8>,
    /// Weak identity of the submission-owned snapshot last validated against
    /// `shadow`. It can only match while that exact Arc is still alive, so
    /// allocator address reuse across submissions cannot produce a false hit.
    pub last_snapshot: Weak<Vec<u8>>,
    pub guest_size: usize,
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
    validated_graphics_interfaces: HashSet<GraphicsInterfaceKey>,
    graphics_pipelines: HashMap<GraphicsPipelineKey, vk::Pipeline>,
    /// Keyed by (canonical module handle, canonical layout handle).
    compute_pipelines: HashMap<(u64, u64), vk::Pipeline>,
    /// Keyed by linear-vs-nearest — the only sampler state decoded today.
    samplers: HashMap<SamplerState, vk::Sampler>,
    targets: HashMap<TargetKey, PersistentTarget>,
    /// Most recently submitted depth attachment associated with each colour
    /// target, used by the GPU presentation bridge.
    target_depth: HashMap<TargetKey, DepthTargetKey>,
    gpu_present_targets: HashMap<GpuPresentKey, GpuPresentTarget>,
    depth_targets: HashMap<DepthTargetKey, PersistentDepthTarget>,
    /// Persistent guest textures (stage D item 1) — see [`PersistentTexture`]
    /// for the invalidation contract.
    textures: HashMap<TextureKey, PersistentTexture>,
    /// Total decoded bytes held by `textures`, for the cap.
    texture_bytes: u64,
    /// Monotonic LRU clock for `textures`.
    texture_clock: u64,
    /// Compute storage images persist across dispatches. Their complete input
    /// is still uploaded and their output read back on every dispatch; this
    /// cache removes only Vulkan create/allocate/view/free churn.
    compute_images: HashMap<ComputeImageKey, PersistentComputeImage>,
    compute_image_bytes: u64,
    compute_image_clock: u64,
    compute_buffers: HashMap<ComputeBufferKey, PersistentComputeBuffer>,
    compute_buffer_bytes: u64,
    compute_buffer_clock: u64,
    /// Compute SSBOs evicted while recorded/submitted batch command buffers
    /// may still reference them. Destroyed only after the shared batch fence.
    deferred_compute_buffer_destroys: Vec<PersistentComputeBuffer>,
    /// Cached texture images evicted while pending command buffers may still
    /// reference them; destroyed at the batch retire (post-fence).
    deferred_image_destroys: Vec<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
    /// Shared descriptor pools for deferred-batch draws (stage D item 2).
    batch_pools: Vec<BatchPoolState>,
    /// One command buffer + fence, reused for every synchronous submission.
    submit: Option<(vk::CommandBuffer, vk::Fence)>,
    pool: Option<PoolState>,
    /// The device-persistent GDS arena (see [`DrawCaches::gds_buffer`]).
    gds: Option<(vk::Buffer, GdsBacking)>,
    // (GdsBacking is defined below the struct.)
    /// Deferred draws whose GPU work is submitted but not yet fence-waited.
    pending: Vec<PendingDrawResources>,
    /// One primary command buffer kept in RECORDING state for the whole PM4
    /// batch. Draws and compute dispatches append to it in guest order; the
    /// flip closes it and submits it once before the readback command.
    batch_recording: Option<vk::CommandBuffer>,
    /// Guest-addressed SSBOs writable by at least one pending compute command.
    /// Read-only cache entries never need a post-fence host scan.
    pending_compute_writes: HashSet<ComputeBufferKey>,
    /// Persistent storage images written by pending compute commands. Values
    /// carry the guest swizzle needed to publish the final linear readback.
    pending_compute_image_writes: HashMap<ComputeImageKey, ComputeImageWriteback>,
    /// Targets drawn by the pending deferred draws, in draw order (last draw
    /// of a target keeps it at the back). Flush reads each back exactly once.
    touched: Vec<TargetKey>,
    /// Recycled primary command buffers for deferred draws.
    free_command_buffers: Vec<vk::CommandBuffer>,
    /// Targets evicted while a batch was open: their images may still be
    /// referenced by pending command buffers, so destruction waits for the
    /// flush's fence.
    deferred_target_destroys: Vec<PersistentTarget>,
    /// Same retirement rule for persistent depth/stencil targets.
    deferred_depth_target_destroys: Vec<PersistentDepthTarget>,
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
    /// Persistently-mapped address of every live pooled allocation, keyed by
    /// raw memory handle. Stored as an integer so the mutex-owned cache remains
    /// `Send`; it is converted back to a pointer only while the allocation is
    /// fence-idle.
    host_pool_mapped: HashMap<u64, usize>,
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

/// Compute UAVs can be substantially larger than sampled textures (for
/// example full-resolution RGBA16F intermediates). Keep a separate bounded
/// budget so a streaming title cannot turn allocation reuse into a leak.
const COMPUTE_IMAGE_CACHE_CAP: u64 = 512 * 1024 * 1024;

/// Writable/global-memory buffers are commonly several MiB each. Keep their
/// host-visible allocations and CPU shadows under an independent bounded LRU.
const COMPUTE_BUFFER_CACHE_CAP: u64 = 512 * 1024 * 1024;

/// Usage union for pooled host buffers. One pool serves every per-draw guest
/// upload, so each buffer carries the union of the usages those uploads need;
/// extra usage bits on a buffer are free on desktop drivers.
pub(crate) const HOST_POOL_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw(),
);

/// Byte size of the emulated GDS arena — the real chip's Global Data Share is
/// 64 KiB.
pub(crate) const GDS_SIZE: usize = 64 * 1024;

/// Backing memory for the GDS arena: sub-allocated through the device's
/// `gpu-allocator` when available (the preferred path — see
/// `VulkanDevice::allocator`), or one raw dedicated `vkAllocateMemory`
/// otherwise. Freed by the matching arm in [`DrawCaches::destroy`].
pub(crate) enum GdsBacking {
    Managed(gpu_allocator::vulkan::Allocation),
    Raw(vk::DeviceMemory),
}

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

    /// Run the SPIR-V cross-stage interface gate once for a canonical
    /// (VS, FS, topology) tuple.
    ///
    /// Shader modules are device-cache handles, so equal SPIR-V has equal
    /// handles for this cache's lifetime.  A failed validation is deliberately
    /// not cached: diagnostics and future recovery retain the old behavior.
    pub(crate) fn validate_graphics_interface_once(
        &mut self,
        vs: vk::ShaderModule,
        fs: vk::ShaderModule,
        topology: vk::PrimitiveTopology,
        validate: impl FnOnce() -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        let key = GraphicsInterfaceKey {
            vs: vs.as_raw(),
            fs: fs.as_raw(),
            topology: topology.as_raw(),
        };
        if self.validated_graphics_interfaces.contains(&key) {
            return Ok(());
        }
        validate()?;
        self.validated_graphics_interfaces.insert(key);
        Ok(())
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
            // Heavy translated shaders are divided with vkCmdDispatchBase so
            // each command remains preemptible under Windows TDR. Vulkan
            // requires this flag even when only later uses have non-zero base
            // groups; enabling it uniformly keeps the canonical pipeline key
            // independent of a particular dispatch's dimensions.
            .flags(vk::PipelineCreateFlags::DISPATCH_BASE)
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
        dev.note_pipeline_compiled();
        Ok(pipeline)
    }

    /// Get or create the sampler for `linear` filtering. Only linear-vs-nearest
    /// is decoded from the guest S# today, so two samplers cover every draw.
    pub(crate) fn sampler(
        &mut self,
        dev: &VulkanDevice,
        mut state: SamplerState,
    ) -> Result<vk::Sampler, GpuError> {
        if !dev.supports_sampler_mirror_clamp_to_edge() {
            // Mirror-once is optional Vulkan functionality. Preserve the
            // mirror direction on unsupported devices without passing an
            // enum whose feature was not enabled (invalid Vulkan).
            for mode in [
                &mut state.address_mode_u,
                &mut state.address_mode_v,
                &mut state.address_mode_w,
            ] {
                if *mode == vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE {
                    *mode = vk::SamplerAddressMode::MIRRORED_REPEAT;
                }
            }
        }
        if let Some(&sampler) = self.samplers.get(&state) {
            return Ok(sampler);
        }
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(state.mag_filter)
            .min_filter(state.min_filter)
            .mipmap_mode(state.mipmap_mode)
            .address_mode_u(state.address_mode_u)
            .address_mode_v(state.address_mode_v)
            .address_mode_w(state.address_mode_w)
            .max_lod(0.0);
        // SAFETY: plain sampler on a live device; retained in this cache and
        // destroyed exactly once in `destroy`.
        let sampler = unsafe { dev.device().create_sampler(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateSampler: {e}")))?;
        self.samplers.insert(state, sampler);
        Ok(sampler)
    }

    /// The device-persistent GDS arena: one 64 KiB storage buffer, zeroed at
    /// creation, whose contents persist across dispatches for the lifetime of
    /// the device — real GDS counters (`ds_append`/`ds_consume`) accumulate
    /// across dispatches to feed indirect-draw arguments (measured on
    /// ASTRO.BOT). GDS is on-chip memory, so nothing is ever written back to
    /// guest memory.
    pub(crate) fn gds_buffer(&mut self, dev: &VulkanDevice) -> Result<vk::Buffer, GpuError> {
        if let Some((buffer, _)) = &self.gds {
            return Ok(*buffer);
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

        // Preferred path: sub-allocate through the device's gpu-allocator
        // (first adopted site). CpuToGpu is host-visible|coherent and comes
        // back persistently mapped, so zeroing needs no map/unmap round trip.
        let managed = {
            let mut guard = dev.allocator().lock();
            guard.as_mut().and_then(|allocator| {
                match allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "raeen-gds",
                    requirements: req,
                    location: gpu_allocator::MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                }) {
                    Ok(allocation) => Some(allocation),
                    Err(e) => {
                        tracing::warn!("GDS sub-allocation failed ({e}); using a raw allocation");
                        None
                    }
                }
            })
        };
        let backing = if let Some(allocation) = managed {
            let cleanup_managed = |allocation, e| {
                if let Some(allocator) = dev.allocator().lock().as_mut() {
                    let _ = allocator.free(allocation);
                }
                // SAFETY: destroying the just-created, never-bound buffer.
                unsafe { dev.device().destroy_buffer(buffer, None) };
                e
            };
            // SAFETY: buffer and the sub-allocation are compatible live
            // handles; offset comes from the allocator for these exact
            // requirements.
            if let Err(e) = unsafe {
                dev.device()
                    .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            } {
                return Err(cleanup_managed(
                    allocation,
                    GpuError::VulkanInitFailed(format!("GDS vkBindBufferMemory: {e}")),
                ));
            }
            let Some(ptr) = allocation.mapped_ptr() else {
                return Err(cleanup_managed(
                    allocation,
                    GpuError::VulkanInitFailed("GDS CpuToGpu allocation is unmapped".to_string()),
                ));
            };
            // Zero the arena once — hardware GDS starts each session cold and
            // the shaders themselves initialize the counters they use.
            // SAFETY: persistently-mapped host-visible allocation of at least
            // GDS_SIZE bytes, not yet visible to the GPU.
            unsafe { std::ptr::write_bytes(ptr.as_ptr().cast::<u8>(), 0, GDS_SIZE) };
            GdsBacking::Managed(allocation)
        } else {
            // Fallback: dedicated raw allocation (allocator unavailable).
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
            GdsBacking::Raw(memory)
        };
        self.gds = Some((buffer, backing));
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
    ) -> Result<(vk::Buffer, vk::DeviceMemory, usize), GpuError> {
        debug_assert!(size > 0, "zero-sized host buffer request");
        // Smallest free size class that fits. Don't hand a small request a
        // giant buffer (>= 8x) — that starves the big classes and bloats
        // descriptor ranges' backing for nothing.
        let requested_class = size.next_power_of_two().max(256);
        let fitting = if std::env::var_os("RAEEN_NO_HOST_POOL").is_none() {
            self.host_pool_free
                .range(size..)
                .find(|(cap, list)| **cap < requested_class.saturating_mul(8) && !list.is_empty())
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
            let mapped = *self
                .host_pool_mapped
                .get(&entry.1.as_raw())
                .expect("pooled host buffer must remain persistently mapped");
            return Ok((entry.0, entry.1, mapped));
        }
        let capacity = requested_class;
        let (buffer, memory) = create_host_buffer(dev, capacity)?;
        let mapped = match unsafe {
            dev.device()
                .map_memory(memory, 0, capacity, vk::MemoryMapFlags::empty())
        } {
            Ok(ptr) => ptr as usize,
            Err(e) => {
                // SAFETY: neither handle has escaped or been referenced by a
                // GPU submission, so creation can be rolled back immediately.
                unsafe {
                    dev.device().destroy_buffer(buffer, None);
                    dev.device().free_memory(memory, None);
                }
                return Err(GpuError::VulkanInitFailed(format!(
                    "persistent vkMapMemory failed: {e}"
                )));
            }
        };
        self.host_pool_capacity.insert(buffer.as_raw(), capacity);
        self.host_pool_mapped.insert(memory.as_raw(), mapped);
        self.stats.host_pool_misses += 1;
        Ok((buffer, memory, mapped))
    }

    /// Return the persistent mapping for a pooled allocation. The caller must
    /// have fence ownership before reading or writing through this address.
    pub(crate) fn mapped_host_memory(&self, memory: vk::DeviceMemory) -> Option<usize> {
        self.host_pool_mapped.get(&memory.as_raw()).copied()
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
        let was_mapped = self.host_pool_mapped.remove(&memory.as_raw()).is_some();
        // SAFETY: caller guarantees no pending GPU work references the pair;
        // both handles were created from this device and are destroyed once.
        unsafe {
            if was_mapped && memory != vk::DeviceMemory::null() {
                dev.device().unmap_memory(memory);
            }
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
        dev.ensure_device_usable()?;
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
        unsafe { dev.device().reset_fences(&[fence]) }.map_err(|e| {
            dev.note_vk_error(e);
            GpuError::VulkanInitFailed(format!("vkResetFences: {e}"))
        })?;
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

    pub(crate) fn acquire_depth_target(
        &mut self,
        key: &DepthTargetKey,
    ) -> Option<PersistentDepthTarget> {
        let entry = self.depth_targets.get(key).copied()?;
        self.stats.depth_target_hits += 1;
        Some(entry)
    }

    pub(crate) fn insert_depth_target(
        &mut self,
        key: DepthTargetKey,
        target: PersistentDepthTarget,
    ) {
        self.stats.depth_target_misses += 1;
        self.depth_targets.insert(key, target);
    }

    /// Remove stale images when the guest reprograms one depth base with a
    /// different extent/format. During a deferred batch, park them until the
    /// shared fence; otherwise no submitted work can still name them.
    pub(crate) fn evict_depth_targets_for_base(
        &mut self,
        device: &ash::Device,
        base: u64,
        keep: &DepthTargetKey,
    ) {
        let stale: Vec<_> = self
            .depth_targets
            .keys()
            .filter(|key| key.base == base && *key != keep)
            .copied()
            .collect();
        for key in stale {
            self.target_depth
                .retain(|_, associated_depth| *associated_depth != key);
            if let Some(target) = self.depth_targets.remove(&key) {
                if self.batch_open() {
                    self.deferred_depth_target_destroys.push(target);
                } else {
                    // SAFETY: no deferred batch is open and synchronous work
                    // fence-waits before releasing this cache lock.
                    unsafe { destroy_depth_target(device, &target) };
                }
            }
        }
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

    /// Drop sampled views that alias a completed compute storage-image write.
    ///
    /// The write owns the complete image at `base`, while a title may consume
    /// the same bytes through another format and extent. Sparse guest hashing
    /// can miss those changed texels; exact base identity is the stronger
    /// coherence boundary. Destruction stays batch-safe because an older draw
    /// may still reference the evicted view.
    pub(crate) fn invalidate_textures_at_base(&mut self, dev: &VulkanDevice, base: u64) -> usize {
        let stale: Vec<_> = self
            .textures
            .keys()
            .filter(|key| texture_aliases_compute_image(key, base))
            .copied()
            .collect();
        let count = stale.len();
        for key in stale {
            if let Some(old) = self.textures.remove(&key) {
                self.texture_bytes = self.texture_bytes.saturating_sub(old.bytes);
                self.destroy_texture_when_safe(dev, old);
                self.stats.texture_cache_evictions += 1;
            }
        }
        count
    }

    /// Reuse or create a guest-addressed compute SSBO and make its content
    /// exactly `bytes`.
    ///
    /// A complete byte comparison is intentional. Guest CPU writes have no
    /// generation counter yet, and a sparse hash is not a valid invalidation
    /// contract for writable shader resources.
    pub(crate) fn acquire_compute_buffer(
        &mut self,
        dev: &VulkanDevice,
        key: ComputeBufferKey,
        snapshot: &Arc<Vec<u8>>,
        guest_size: usize,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        let bytes = snapshot.as_slice();
        debug_assert_ne!(key.base, 0);
        debug_assert_eq!(key.size, bytes.len());
        debug_assert!(guest_size <= key.size);
        self.compute_buffer_clock += 1;
        let clock = self.compute_buffer_clock;
        if let Some(entry) = self.compute_buffers.get_mut(&key) {
            entry.last_use = clock;
            entry.guest_size = guest_size;
            self.stats.compute_buffer_hits += 1;
            if entry
                .last_snapshot
                .upgrade()
                .is_some_and(|previous| Arc::ptr_eq(&previous, snapshot))
            {
                // The command processor captured this guest allocation once
                // for the whole PM4 submission. Later dispatches sharing that
                // exact Arc cannot observe intervening guest CPU writes, and
                // must preserve any GPU writes already queued into the
                // persistent buffer. Avoid re-scanning up to 4 MiB here.
                self.stats.compute_buffer_uploads_skipped += 1;
                return Ok((entry.buffer, entry.memory));
            }
            if entry.shadow == bytes {
                entry.last_snapshot = Arc::downgrade(snapshot);
                self.stats.compute_buffer_uploads_skipped += 1;
                return Ok((entry.buffer, entry.memory));
            }
            map_copy_compute_buffer(dev, entry.memory, bytes)?;
            entry.shadow.copy_from_slice(bytes);
            entry.last_snapshot = Arc::downgrade(snapshot);
            return Ok((entry.buffer, entry.memory));
        }

        // Never evict an SSBO while the active batch may still reference it:
        // besides invalidating a recorded command buffer, eviction before the
        // fence would discard shader output before guest writeback. A batch may
        // temporarily exceed the steady-state budget; post-fence pruning below
        // restores the cap safely.
        while !self.batch_open()
            && self.compute_buffer_bytes.saturating_add(key.size as u64) > COMPUTE_BUFFER_CACHE_CAP
        {
            let Some((&lru_key, _)) = self
                .compute_buffers
                .iter()
                .min_by_key(|(_, entry)| entry.last_use)
            else {
                break;
            };
            let old = self
                .compute_buffers
                .remove(&lru_key)
                .expect("compute buffer LRU key was just found");
            self.compute_buffer_bytes = self
                .compute_buffer_bytes
                .saturating_sub(old.shadow.len() as u64);
            if self.batch_open() {
                self.deferred_compute_buffer_destroys.push(old);
            } else {
                destroy_compute_buffer(dev.device(), old);
            }
            self.stats.compute_buffer_evictions += 1;
        }

        let (buffer, memory) = create_host_buffer(dev, key.size as u64)?;
        if let Err(error) = map_copy_compute_buffer(dev, memory, bytes) {
            // SAFETY: creation succeeded but the allocation has never been
            // submitted or published into the cache.
            unsafe {
                dev.device().destroy_buffer(buffer, None);
                dev.device().free_memory(memory, None);
            }
            return Err(error);
        }
        let shadow = bytes.to_vec();
        self.compute_buffer_bytes = self
            .compute_buffer_bytes
            .saturating_add(shadow.len() as u64);
        self.compute_buffers.insert(
            key,
            PersistentComputeBuffer {
                buffer,
                memory,
                shadow,
                last_snapshot: Arc::downgrade(snapshot),
                guest_size,
                last_use: clock,
            },
        );
        self.stats.compute_buffer_misses += 1;
        Ok((buffer, memory))
    }

    /// Restore the steady-state SSBO cache budget after the batch fence and
    /// guest writeback. At this point no command buffer references an evicted
    /// entry and its GPU output has already been published.
    pub(crate) fn prune_compute_buffers(&mut self, dev: &VulkanDevice) {
        while self.compute_buffer_bytes > COMPUTE_BUFFER_CACHE_CAP {
            let Some((&lru_key, _)) = self
                .compute_buffers
                .iter()
                .min_by_key(|(_, entry)| entry.last_use)
            else {
                break;
            };
            let old = self
                .compute_buffers
                .remove(&lru_key)
                .expect("compute buffer LRU key was just found");
            self.compute_buffer_bytes = self
                .compute_buffer_bytes
                .saturating_sub(old.shadow.len() as u64);
            destroy_compute_buffer(dev.device(), old);
            self.stats.compute_buffer_evictions += 1;
        }
    }

    /// Apply the sparse post-dispatch delta to the authoritative CPU shadow.
    /// The bind path already made the shadow equal to the initial guest bytes,
    /// so only shader-written spans need copying.
    pub(crate) fn update_compute_buffer_shadow(
        &mut self,
        key: ComputeBufferKey,
        dirty: &[super::compute::ComputeDirtySpan],
    ) {
        let Some(entry) = self.compute_buffers.get_mut(&key) else {
            return;
        };
        for span in dirty {
            let end = span
                .offset
                .saturating_add(span.bytes.len())
                .min(entry.shadow.len());
            if span.offset < end {
                entry.shadow[span.offset..end].copy_from_slice(&span.bytes[..end - span.offset]);
            }
        }
    }

    /// Publish GPU-written persistent compute buffers after the shared batch
    /// fence. Only changed 4 KiB pages reach guest memory; the cache shadow is
    /// advanced in lockstep so the next submission can validate CPU writes
    /// against exact bytes.
    pub(crate) fn flush_compute_buffers_to_guest(
        &mut self,
        dev: &VulkanDevice,
    ) -> Result<(usize, usize), GpuError> {
        const PAGE: usize = 4096;
        let mut dirty_bytes = 0usize;
        let mut dirty_spans = 0usize;
        let writable = std::mem::take(&mut self.pending_compute_writes);
        for key in writable {
            let Some(entry) = self.compute_buffers.get_mut(&key) else {
                continue;
            };
            // SAFETY: the caller's shared queue fence completed and this
            // allocation is HOST_VISIBLE|COHERENT for its full shadow size.
            let ptr = unsafe {
                dev.device().map_memory(
                    entry.memory,
                    0,
                    entry.shadow.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|e| GpuError::VulkanInitFailed(format!("compute batch vkMapMemory: {e}")))?;
            let result =
                unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), entry.shadow.len()) };
            let mut at = 0usize;
            while at < entry.guest_size {
                let end = at.saturating_add(PAGE).min(entry.guest_size);
                if result[at..end] == entry.shadow[at..end] {
                    at = end;
                    continue;
                }
                let start = at;
                at = end;
                while at < entry.guest_size {
                    let next = at.saturating_add(PAGE).min(entry.guest_size);
                    if result[at..next] == entry.shadow[at..next] {
                        break;
                    }
                    at = next;
                }
                let address = key.base.saturating_add(start as u64);
                crate::guest_mem::trace_scanout_fill(address, at - start, "compute-batch-storage");
                if !crate::guest_mem::write_bytes_checked(address, &result[start..at]) {
                    unsafe { dev.device().unmap_memory(entry.memory) };
                    return Err(GpuError::VulkanInitFailed(format!(
                        "deferred compute writeback {address:#x}..{:#x} is not writable guest \
                         memory",
                        address.saturating_add((at - start) as u64)
                    )));
                }
                entry.shadow[start..at].copy_from_slice(&result[start..at]);
                dirty_bytes += at - start;
                dirty_spans += 1;
            }
            unsafe { dev.device().unmap_memory(entry.memory) };
        }
        Ok((dirty_bytes, dirty_spans))
    }

    /// Publish the final result of every storage image written by the pending
    /// batch. Each dispatch copied its result into the persistent readback
    /// buffer in queue order; after the shared fence the buffer therefore
    /// contains the last writer's complete linear image. A CPU guest-memory
    /// mirror is best-effort because GPU-only images remain valid inputs for
    /// later draws and valid presentation candidates.
    pub(crate) fn flush_compute_images_to_guest(
        &mut self,
        dev: &VulkanDevice,
    ) -> Result<(usize, Vec<(u64, crate::RenderedImage)>), GpuError> {
        let pending = std::mem::take(&mut self.pending_compute_image_writes);
        let mut written = 0usize;
        let mut presentable = Vec::new();
        for (key, writeback) in pending {
            let Some(entry) = self.compute_images.get(&key).cloned() else {
                continue;
            };
            let linear_size = key
                .width
                .checked_mul(key.height)
                .and_then(|n| n.checked_mul(key.depth))
                .and_then(|n| n.checked_mul(key.layers))
                .and_then(|n| n.checked_mul(writeback.texel))
                .map(|n| n as usize)
                .ok_or_else(|| {
                    GpuError::VulkanInitFailed(format!(
                        "deferred compute image {:#x} extent overflow",
                        key.base
                    ))
                })?;
            // SAFETY: the shared batch fence completed. The persistent
            // readback allocation is HOST_VISIBLE|COHERENT and was created for
            // exactly this image's linear byte size.
            let (ptr, must_unmap) =
                if let Some(mapped) = self.mapped_host_memory(entry.readback_memory) {
                    (mapped as *mut std::ffi::c_void, false)
                } else {
                    let ptr = unsafe {
                        dev.device().map_memory(
                            entry.readback_memory,
                            0,
                            linear_size as u64,
                            vk::MemoryMapFlags::empty(),
                        )
                    }
                    .map_err(|e| {
                        GpuError::VulkanInitFailed(format!("compute image batch vkMapMemory: {e}"))
                    })?;
                    (ptr, true)
                };
            let linear = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), linear_size) };
            let nonzero = linear.iter().any(|&byte| byte != 0);
            if crate::diagnostics::gpu_env().trace_draws {
                tracing::warn!(
                    base = format_args!("{:#x}", key.base),
                    len = linear_size,
                    nonzero,
                    width = key.width,
                    height = key.height,
                    format = ?vk::Format::from_raw(key.format),
                    tile_mode = writeback.tile_mode,
                    "TRACE_DRAWS: deferred compute image writeback"
                );
            }
            if nonzero
                && matches!(
                    vk::Format::from_raw(key.format),
                    vk::Format::R8G8B8A8_UNORM | vk::Format::R16G16B16A16_SFLOAT
                )
                && key.depth == 1
                && key.layers == 1
                && crate::guest_mem::is_scanout_candidate(key.base, linear_size)
            {
                let mut pixels = Vec::new();
                if pixels.try_reserve_exact(linear_size).is_err() {
                    if must_unmap {
                        unsafe { dev.device().unmap_memory(entry.readback_memory) };
                    }
                    return Err(GpuError::VulkanInitFailed(format!(
                        "deferred compute image {:#x} present copy allocation failed",
                        key.base
                    )));
                }
                pixels.extend_from_slice(linear);
                presentable.push((
                    key.base,
                    crate::RenderedImage {
                        width: key.width,
                        height: key.height,
                        pixels,
                        bytes_per_pixel: writeback.texel,
                    },
                ));
            }
            let guest = encode_compute_image_writeback(writeback, linear);
            if must_unmap {
                unsafe { dev.device().unmap_memory(entry.readback_memory) };
            }
            let guest = guest?;
            let guest_len = guest.len();
            if crate::guest_mem::mirror_compute_image_to_guest(
                key.base,
                guest,
                "compute-batch-image",
            ) == crate::guest_mem::ComputeImageGuestMirror::Written
            {
                written = written.saturating_add(guest_len);
            }
            self.invalidate_textures_at_base(dev, key.base);
        }
        Ok((written, presentable))
    }

    /// Reuse a compute UAV allocation with exactly the requested guest
    /// identity and Vulkan shape. Returns whether the caller must upload its
    /// seed. Rebinding the exact same submission snapshot preserves GPU-newer
    /// ordered contents and skips the redundant staging copy.
    pub(crate) fn compute_image_entry(
        &mut self,
        key: &ComputeImageKey,
        snapshot: &Arc<Vec<u8>>,
    ) -> Option<(PersistentComputeImage, bool)> {
        self.compute_image_clock += 1;
        let clock = self.compute_image_clock;
        let entry = self.compute_images.get_mut(key)?;
        entry.last_use = clock;
        self.stats.compute_image_hits += 1;
        let upload = !entry
            .last_snapshot
            .upgrade()
            .is_some_and(|previous| Arc::ptr_eq(&previous, snapshot));
        if upload {
            entry.last_snapshot = Arc::downgrade(snapshot);
        } else {
            self.stats.compute_image_uploads_skipped += 1;
        }
        Some((entry.clone(), upload))
    }

    /// Retain a newly-created compute UAV and enforce a bounded LRU budget.
    ///
    /// An open deferred batch may still reference any retained UAV. Like the
    /// SSBO cache, it may temporarily exceed its steady-state cap; pruning
    /// runs only after the shared fence and guest publication.
    pub(crate) fn insert_compute_image(
        &mut self,
        dev: &VulkanDevice,
        key: ComputeImageKey,
        mut image: PersistentComputeImage,
    ) {
        if let Some(old) = self.compute_images.remove(&key) {
            self.compute_image_bytes = self.compute_image_bytes.saturating_sub(old.bytes);
            self.destroy_compute_image(dev, old);
            self.stats.compute_image_evictions += 1;
        }
        while !self.batch_open()
            && self.compute_image_bytes.saturating_add(image.bytes) > COMPUTE_IMAGE_CACHE_CAP
        {
            let Some((&lru_key, _)) = self
                .compute_images
                .iter()
                .min_by_key(|(_, entry)| entry.last_use)
            else {
                break;
            };
            let old = self
                .compute_images
                .remove(&lru_key)
                .expect("compute image LRU key was just found");
            self.compute_image_bytes = self.compute_image_bytes.saturating_sub(old.bytes);
            self.destroy_compute_image(dev, old);
            self.stats.compute_image_evictions += 1;
        }
        self.compute_image_clock += 1;
        image.last_use = self.compute_image_clock;
        self.compute_image_bytes = self.compute_image_bytes.saturating_add(image.bytes);
        self.compute_images.insert(key, image);
        self.stats.compute_image_misses += 1;
    }

    /// Restore the steady-state UAV budget after every pending storage-image
    /// result has been copied to guest memory and no queued command buffer can
    /// reference an evicted entry.
    pub(crate) fn prune_compute_images(&mut self, dev: &VulkanDevice) {
        while self.compute_image_bytes > COMPUTE_IMAGE_CACHE_CAP {
            let Some((&lru_key, _)) = self
                .compute_images
                .iter()
                .min_by_key(|(_, entry)| entry.last_use)
            else {
                break;
            };
            let old = self
                .compute_images
                .remove(&lru_key)
                .expect("compute image LRU key was just found");
            self.compute_image_bytes = self.compute_image_bytes.saturating_sub(old.bytes);
            self.destroy_compute_image(dev, old);
            self.stats.compute_image_evictions += 1;
        }
    }

    fn destroy_compute_image(&mut self, dev: &VulkanDevice, image: PersistentComputeImage) {
        // SAFETY: synchronous compute guarantees no command buffer references
        // these handles; the cache entry has already been removed.
        unsafe {
            let device = dev.device();
            device.destroy_image_view(image.view, None);
            device.destroy_image(image.image, None);
            device.free_memory(image.memory, None);
        }
        self.release_host_buffer(dev, image.readback_buffer, image.readback_memory);
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
            target.layout = TargetLayout::TransferSrc;
        }
    }

    /// The persistent image + view for `key`, if one is live — used to bind a
    /// render target directly as a sampled descriptor (stage B).
    pub(crate) fn target_image(
        &self,
        key: &TargetKey,
    ) -> Option<(vk::Image, vk::ImageView, TargetLayout)> {
        self.targets
            .get(key)
            .filter(|target| target.content != TargetContent::Unknown)
            .map(|target| (target.image, target.view, target.layout))
    }

    /// A copy of the whole persistent-target entry for `key` (flush readback).
    pub(crate) fn target_entry(&self, key: &TargetKey) -> Option<PersistentTarget> {
        self.targets.get(key).copied()
    }

    pub(crate) fn gpu_present_target(
        &mut self,
        dev: &VulkanDevice,
        key: GpuPresentKey,
        bytes_per_pixel: u32,
    ) -> Result<GpuPresentTarget, GpuError> {
        let stale: Vec<_> = self
            .gpu_present_targets
            .keys()
            .filter(|candidate| candidate.source_base == key.source_base && **candidate != key)
            .copied()
            .collect();
        for stale_key in stale {
            if let Some(target) = self.gpu_present_targets.remove(&stale_key) {
                // SAFETY: GPU present targets are used only by the synchronous
                // flush, whose fence is waited before another flush can resize.
                unsafe { destroy_gpu_present_target(dev.device(), &target) };
            }
        }
        if let Some(target) = self.gpu_present_targets.get(&key) {
            return Ok(*target);
        }
        let target = create_gpu_present_target(dev, key, bytes_per_pixel)?;
        self.gpu_present_targets.insert(key, target);
        Ok(target)
    }

    pub(crate) fn mark_gpu_present_layout(&mut self, key: &GpuPresentKey, layout: vk::ImageLayout) {
        if let Some(target) = self.gpu_present_targets.get_mut(key) {
            target.layout = layout;
        }
    }

    /// The depth image submitted with the most recent draw into `color`.
    pub(crate) fn depth_target_for_color(
        &self,
        color: &TargetKey,
    ) -> Option<(DepthTargetKey, PersistentDepthTarget)> {
        let key = *self.target_depth.get(color)?;
        self.depth_targets
            .get(&key)
            .copied()
            .map(|target| (key, target))
    }

    /// Degrade `key`'s content to [`TargetContent::Unknown`] (a flush failed:
    /// the image may or may not hold the batch's draws).
    pub(crate) fn mark_target_unknown(&mut self, key: &TargetKey) {
        if let Some(target) = self.targets.get_mut(key) {
            target.content = TargetContent::Unknown;
            target.layout = TargetLayout::Undefined;
        }
    }

    pub(crate) fn mark_target_layout(&mut self, key: &TargetKey, layout: TargetLayout) {
        if let Some(target) = self.targets.get_mut(key) {
            target.layout = layout;
        }
    }

    pub(crate) fn mark_depth_target_layout(
        &mut self,
        key: &DepthTargetKey,
        layout: DepthTargetLayout,
    ) {
        if let Some(target) = self.depth_targets.get_mut(key) {
            target.layout = layout;
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

    /// The primary command buffer shared by every deferred draw/dispatch in the
    /// current PM4 batch. It is begun once here and remains in RECORDING state
    /// until [`Self::finish_batch_recording`] at the flip boundary.
    pub(crate) fn batch_command_buffer(
        &mut self,
        dev: &VulkanDevice,
    ) -> Result<vk::CommandBuffer, GpuError> {
        if let Some(cb) = self.batch_recording {
            return Ok(cb);
        }
        let cb = if let Some(cb) = self.free_command_buffers.pop() {
            cb
        } else {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(dev.command_pool())
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the pool belongs to this device and access is serialized
            // by the cache lock (see module docs).
            let buffers =
                unsafe { dev.device().allocate_command_buffers(&alloc_info) }.map_err(|e| {
                    GpuError::VulkanInitFailed(format!("batch command buffer alloc: {e}"))
                })?;
            buffers.first().copied().ok_or_else(|| {
                GpuError::VulkanInitFailed("no batch command buffer returned".to_owned())
            })?
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cb` is fence-retired or newly allocated from a
        // RESET_COMMAND_BUFFER pool, and the cache lock serializes recording.
        if let Err(error) = unsafe { dev.device().begin_command_buffer(cb, &begin) } {
            self.free_command_buffers.push(cb);
            return Err(GpuError::VulkanInitFailed(format!(
                "batch vkBeginCommandBuffer: {error}"
            )));
        }
        self.batch_recording = Some(cb);
        Ok(cb)
    }

    /// Close the current shared recording and attach its sole command-buffer
    /// handle to the pending resource batch for submit/recycle at the fence.
    pub(crate) fn finish_batch_recording(&mut self, dev: &VulkanDevice) -> Result<(), GpuError> {
        let Some(cb) = self.batch_recording.take() else {
            return Ok(());
        };
        // SAFETY: `batch_command_buffer` began `cb`, and every append happened
        // under this same cache lock. It has not been submitted.
        if let Err(error) = unsafe { dev.device().end_command_buffer(cb) } {
            self.batch_recording = Some(cb);
            return Err(GpuError::VulkanInitFailed(format!(
                "batch vkEndCommandBuffer: {error}"
            )));
        }
        if let Some(resources) = self.pending.first_mut() {
            debug_assert_eq!(resources.command_buffer, vk::CommandBuffer::null());
            resources.command_buffer = cb;
        } else {
            self.pending.push(PendingDrawResources {
                command_buffer: cb,
                ..PendingDrawResources::default()
            });
        }
        Ok(())
    }

    /// Record a successfully submitted deferred draw: its per-draw resources
    /// join the pending list, the target joins the touched list (moved to the
    /// back so flush order follows last-draw order), and the target's GPU
    /// image becomes the sole content authority.
    pub(crate) fn commit_deferred_draw(
        &mut self,
        res: PendingDrawResources,
        key: TargetKey,
        depth_key: Option<DepthTargetKey>,
        layout: TargetLayout,
    ) {
        self.pending.push(res);
        self.touched.retain(|k| *k != key);
        self.touched.push(key);
        if let Some(depth_key) = depth_key {
            self.target_depth.insert(key, depth_key);
        } else {
            self.target_depth.remove(&key);
        }
        if let Some(target) = self.targets.get_mut(&key) {
            target.content = TargetContent::GpuNewer;
            target.layout = layout;
        }
        self.stats.deferred_draws += 1;
    }

    /// Retain resources for deferred compute work that has no render-target
    /// identity. The next graphics/flip flush fences these command buffers on
    /// the same queue and retires their descriptor/upload resources together.
    pub(crate) fn commit_deferred_resources(
        &mut self,
        res: PendingDrawResources,
        writable: impl IntoIterator<Item = ComputeBufferKey>,
        image_writes: impl IntoIterator<Item = ComputeImageWriteback>,
    ) {
        self.pending.push(res);
        self.pending_compute_writes.extend(writable);
        self.pending_compute_image_writes
            .extend(image_writes.into_iter().map(|write| (write.key, write)));
    }

    pub(crate) fn discard_pending_compute_writes(&mut self) {
        self.pending_compute_writes.clear();
        self.pending_compute_image_writes.clear();
    }

    /// Whether deferred compute has guest-addressed outputs that a following
    /// graphics draw may consume as an index/vertex/resource buffer.
    ///
    /// Draw translation currently fetches those inputs from the identity-mapped
    /// guest bytes. The ordered Vulkan batch may keep unrelated work GPU-side,
    /// but it must publish these outputs before the CPU-side fetch occurs.
    pub(crate) fn has_pending_compute_writebacks(&self) -> bool {
        !self.pending_compute_writes.is_empty() || !self.pending_compute_image_writes.is_empty()
    }

    /// Take the whole pending batch for a flush: per-draw resources, touched
    /// targets (draw order), and targets whose destruction was deferred.
    pub(crate) fn take_batch(
        &mut self,
    ) -> (
        Vec<PendingDrawResources>,
        usize,
        Vec<TargetKey>,
        Vec<PersistentTarget>,
        Vec<PersistentDepthTarget>,
    ) {
        debug_assert!(self.batch_recording.is_none());
        (
            std::mem::take(&mut self.pending),
            0,
            std::mem::take(&mut self.touched),
            std::mem::take(&mut self.deferred_target_destroys),
            std::mem::take(&mut self.deferred_depth_target_destroys),
        )
    }

    /// Destroy (or recycle) one flushed batch's resources. The caller must
    /// have waited the flush fence — nothing on the GPU references them.
    pub(crate) fn retire_batch(
        &mut self,
        dev: &VulkanDevice,
        pending: Vec<PendingDrawResources>,
        evicted: Vec<PersistentTarget>,
        evicted_depth: Vec<PersistentDepthTarget>,
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
        for target in evicted_depth {
            // SAFETY: the same shared fence covers the command buffers that
            // last referenced this evicted persistent depth target.
            unsafe { destroy_depth_target(dev.device(), &target) };
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
        for buffer in std::mem::take(&mut self.deferred_compute_buffer_destroys) {
            // SAFETY: the shared fence covered every command buffer that
            // referenced this cache entry before its eviction.
            destroy_compute_buffer(dev.device(), buffer);
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
            self.target_depth.remove(&key);
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
    /// `device_wait_idle`, before the command pool, the memory allocator,
    /// and the device go away. `allocator` takes back sub-allocated backings
    /// (currently the GDS arena).
    pub(crate) fn destroy(
        &mut self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        allocator: &mut Option<gpu_allocator::vulkan::Allocator>,
    ) {
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
                    if self.host_pool_mapped.remove(&memory.as_raw()).is_some() {
                        device.unmap_memory(memory);
                    }
                    self.host_pool_capacity.remove(&buffer.as_raw());
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
            if let Some(command_buffer) = self.batch_recording.take() {
                device.free_command_buffers(command_pool, &[command_buffer]);
            }
            self.touched.clear();
            self.target_depth.clear();
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
            if let Some((buffer, backing)) = self.gds.take() {
                device.destroy_buffer(buffer, None);
                match backing {
                    GdsBacking::Managed(allocation) => {
                        if let Some(allocator) = allocator.as_mut() {
                            let _ = allocator.free(allocation);
                        }
                    }
                    GdsBacking::Raw(memory) => device.free_memory(memory, None),
                }
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
            for (_, target) in self.gpu_present_targets.drain() {
                destroy_gpu_present_target(device, &target);
            }
            for (_, target) in self.depth_targets.drain() {
                destroy_depth_target(device, &target);
            }
            for (_, texture) in self.textures.drain() {
                device.destroy_image_view(texture.view, None);
                device.destroy_image(texture.image, None);
                device.free_memory(texture.memory, None);
            }
            self.texture_bytes = 0;
            for (_, image) in self.compute_images.drain() {
                device.destroy_image_view(image.view, None);
                device.destroy_image(image.image, None);
                device.free_memory(image.memory, None);
                if self
                    .host_pool_mapped
                    .remove(&image.readback_memory.as_raw())
                    .is_some()
                {
                    device.unmap_memory(image.readback_memory);
                }
                self.host_pool_capacity
                    .remove(&image.readback_buffer.as_raw());
                device.destroy_buffer(image.readback_buffer, None);
                device.free_memory(image.readback_memory, None);
            }
            self.compute_image_bytes = 0;
            for (_, buffer) in self.compute_buffers.drain() {
                destroy_compute_buffer(device, buffer);
            }
            self.compute_buffer_bytes = 0;
            for buffer in self.deferred_compute_buffer_destroys.drain(..) {
                destroy_compute_buffer(device, buffer);
            }
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
                    if self.host_pool_mapped.remove(&memory.as_raw()).is_some() {
                        device.unmap_memory(memory);
                    }
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
            }
            self.host_pool_capacity.clear();
            self.host_pool_mapped.clear();
            self.host_pool_free_bytes = 0;
        }
    }
}

/// Create one pooled host-visible|coherent buffer of `capacity` bytes with the
/// pool's usage union.
///
/// Prefer `HOST_CACHED`: compute storage buffers are both upload sources and
/// guest writeback targets. Mapping an uncached host-visible heap and copying
/// a multi-megabyte writable V# can be tens of milliseconds on an iGPU
/// (measured: Minecraft's 4 MiB menu compute output). Cached coherent memory
/// keeps direct descriptor access while making the post-fence CPU readback a
/// normal cached memcpy. Devices without that combination retain the previous
/// host-visible/coherent fallback.
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
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let type_index = dev
        .find_memory_type(
            reqs.memory_type_bits,
            host | vk::MemoryPropertyFlags::HOST_CACHED,
        )
        .or_else(|_| dev.find_memory_type(reqs.memory_type_bits, host))
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

fn encode_compute_image_writeback(
    writeback: ComputeImageWriteback,
    linear: &[u8],
) -> Result<Vec<u8>, GpuError> {
    let key = writeback.key;
    if key.depth > 1 || writeback.tile_mode == 0 {
        let mut guest = Vec::new();
        guest.try_reserve_exact(linear.len()).map_err(|_| {
            GpuError::VulkanInitFailed(format!(
                "deferred compute image {:#x} linear writeback allocation failed",
                key.base
            ))
        })?;
        guest.extend_from_slice(linear);
        return Ok(guest);
    }
    let bpp_log2 = writeback.texel.trailing_zeros();
    let face_linear = key.width as usize * key.height as usize * writeback.texel as usize;
    let expected = face_linear.saturating_mul(key.layers as usize);
    if linear.len() < expected {
        return Err(GpuError::VulkanInitFailed(format!(
            "deferred compute image {:#x} readback is {} B; expected at least {expected} B",
            key.base,
            linear.len()
        )));
    }
    let face_tiled = crate::texture::tiling::tiled_byte_count_for_mode(
        writeback.tile_mode,
        key.width,
        key.height,
        bpp_log2,
    )
    .ok_or_else(|| {
        GpuError::VulkanInitFailed(format!(
            "deferred compute image tile mode {} is not implemented",
            writeback.tile_mode
        ))
    })? as usize;
    let total = face_tiled.saturating_mul(key.layers as usize);
    let mut tiled = Vec::new();
    tiled.try_reserve_exact(total).map_err(|_| {
        GpuError::VulkanInitFailed(format!(
            "deferred compute image {:#x} tiled writeback allocation failed",
            key.base
        ))
    })?;
    tiled.resize(total, 0);
    for layer in 0..key.layers as usize {
        let face = &linear[layer * face_linear..(layer + 1) * face_linear];
        let output = &mut tiled[layer * face_tiled..(layer + 1) * face_tiled];
        if !crate::texture::tiling::tile_64kb_into(
            writeback.tile_mode,
            face,
            output,
            key.width,
            key.height,
            bpp_log2,
        ) {
            return Err(GpuError::VulkanInitFailed(format!(
                "deferred compute image tile mode {} could not encode {}x{} layer {layer}",
                writeback.tile_mode, key.width, key.height
            )));
        }
    }
    Ok(tiled)
}

fn map_copy_compute_buffer(
    dev: &VulkanDevice,
    memory: vk::DeviceMemory,
    bytes: &[u8],
) -> Result<(), GpuError> {
    // SAFETY: compute-buffer cache allocations are HOST_VISIBLE|COHERENT,
    // cache access is serialized, and callers only update them after the
    // preceding synchronous dispatch fence completed.
    let ptr = unsafe {
        dev.device()
            .map_memory(memory, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())
    }
    .map_err(|e| GpuError::VulkanInitFailed(format!("compute buffer vkMapMemory: {e}")))?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
        dev.device().unmap_memory(memory);
    }
    Ok(())
}

fn destroy_compute_buffer(device: &ash::Device, buffer: PersistentComputeBuffer) {
    // SAFETY: caller guarantees no pending queue work references this cache
    // entry and removes it from the sole owning map before calling.
    unsafe {
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }
}

fn create_gpu_present_target(
    dev: &VulkanDevice,
    key: GpuPresentKey,
    bytes_per_pixel: u32,
) -> Result<GpuPresentTarget, GpuError> {
    let device = dev.device();
    let format = vk::Format::from_raw(key.format);
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: key.width,
            height: key.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: create info is complete and the device is live.
    let image = unsafe { device.create_image(&image_info, None) }.map_err(|error| {
        GpuError::VulkanInitFailed(format!("GPU plugin output vkCreateImage: {error}"))
    })?;
    // SAFETY: image belongs to this device.
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = match dev.find_memory_type(
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    ) {
        Ok(index) => index,
        Err(error) => {
            // SAFETY: image is not bound or in use.
            unsafe { device.destroy_image(image, None) };
            return Err(error);
        }
    };
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    // SAFETY: allocation matches the image requirements.
    let memory = match unsafe { device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(error) => {
            // SAFETY: image is not bound or in use.
            unsafe { device.destroy_image(image, None) };
            return Err(GpuError::VulkanInitFailed(format!(
                "GPU plugin output vkAllocateMemory: {error}"
            )));
        }
    };
    // SAFETY: memory was allocated for this image at offset zero.
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        // SAFETY: neither handle is in use.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
        }
        return Err(GpuError::VulkanInitFailed(format!(
            "GPU plugin output vkBindImageMemory: {error}"
        )));
    }
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: image is live and the view describes its sole color subresource.
    let view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(view) => view,
        Err(error) => {
            // SAFETY: handles are not in use.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
            }
            return Err(GpuError::VulkanInitFailed(format!(
                "GPU plugin output vkCreateImageView: {error}"
            )));
        }
    };
    let readback_size = u64::from(key.width)
        .checked_mul(u64::from(key.height))
        .and_then(|pixels| pixels.checked_mul(u64::from(bytes_per_pixel)))
        .ok_or_else(|| {
            GpuError::VulkanInitFailed("GPU plugin output readback size overflow".to_owned())
        });
    let readback_size = match readback_size {
        Ok(size) => size,
        Err(error) => {
            // SAFETY: handles are not in use.
            unsafe {
                device.destroy_image_view(view, None);
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(error);
        }
    };
    let (readback_buffer, readback_memory) = match create_host_buffer(dev, readback_size) {
        Ok(pair) => pair,
        Err(error) => {
            // SAFETY: handles are not in use.
            unsafe {
                device.destroy_image_view(view, None);
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(error);
        }
    };
    Ok(GpuPresentTarget {
        image,
        memory,
        view,
        readback_buffer,
        readback_memory,
        layout: vk::ImageLayout::UNDEFINED,
    })
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

unsafe fn destroy_gpu_present_target(device: &ash::Device, target: &GpuPresentTarget) {
    // SAFETY: forwarded from the caller's no-live-work contract.
    unsafe {
        device.destroy_image_view(target.view, None);
        device.destroy_image(target.image, None);
        device.free_memory(target.memory, None);
        device.destroy_buffer(target.readback_buffer, None);
        device.free_memory(target.readback_memory, None);
    }
}

unsafe fn destroy_depth_target(device: &ash::Device, target: &PersistentDepthTarget) {
    // SAFETY: forwarded from the caller's no-live-work contract.
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

    #[test]
    fn compute_image_write_invalidates_every_format_alias_at_the_same_base() {
        let key = TextureKey {
            base: 0x1614_73000,
            width: 2048,
            height: 4096,
            layers: 1,
            depth: 1,
            cube: false,
            array: false,
            volume: false,
            format: vk::Format::R8_UNORM.as_raw(),
        };

        assert!(texture_aliases_compute_image(&key, 0x1614_73000));
        assert!(!texture_aliases_compute_image(&key, 0x1614_74000));
        assert!(!texture_aliases_compute_image(&key, 0));
    }

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
            vertex_bindings: vec![(0, 16, vk::VertexInputRate::VERTEX.as_raw())],
            vertex_attributes: vec![(0, 0, vk::Format::R32G32B32A32_SFLOAT.as_raw(), 0)],
            mrt: Vec::new(),
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

        let different_input_rate = GraphicsPipelineKey {
            vertex_bindings: vec![(0, 16, vk::VertexInputRate::INSTANCE.as_raw())],
            ..base.clone()
        };
        assert_ne!(base, different_input_rate);
    }

    #[test]
    fn pending_compute_writebacks_form_a_graphics_resource_boundary() {
        let mut caches = DrawCaches::default();
        assert!(!caches.has_pending_compute_writebacks());

        caches.commit_deferred_resources(
            PendingDrawResources::default(),
            [ComputeBufferKey {
                base: 0x4000,
                size: 0x100,
            }],
            std::iter::empty(),
        );
        assert!(
            caches.has_pending_compute_writebacks(),
            "a GPU-newer guest SSBO must be published before a following draw reads it"
        );

        caches.discard_pending_compute_writes();
        assert!(!caches.has_pending_compute_writebacks());
    }

    #[test]
    fn graphics_interface_validation_is_once_per_canonical_pair_and_topology() {
        use std::cell::Cell;

        let mut caches = DrawCaches::default();
        let validations = Cell::new(0_u32);
        let vs = vk::ShaderModule::from_raw(0x100);
        let fs = vk::ShaderModule::from_raw(0x200);

        for _ in 0..3 {
            caches
                .validate_graphics_interface_once(
                    vs,
                    fs,
                    vk::PrimitiveTopology::TRIANGLE_LIST,
                    || {
                        validations.set(validations.get() + 1);
                        Ok(())
                    },
                )
                .expect("valid interface");
        }
        assert_eq!(validations.get(), 1);

        caches
            .validate_graphics_interface_once(vs, fs, vk::PrimitiveTopology::POINT_LIST, || {
                validations.set(validations.get() + 1);
                Ok(())
            })
            .expect("topology-specific validation");
        assert_eq!(validations.get(), 2);
    }

    #[test]
    fn compute_image_reuses_only_the_same_live_submission_snapshot() {
        let mut caches = DrawCaches::default();
        let key = ComputeImageKey {
            base: 0x8000,
            width: 4,
            height: 4,
            depth: 1,
            layers: 1,
            array: false,
            volume: false,
            format: vk::Format::R8G8B8A8_UNORM.as_raw(),
        };
        let snapshot = Arc::new(vec![0x11; 64]);
        caches.compute_images.insert(
            key,
            PersistentComputeImage {
                image: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
                view: vk::ImageView::null(),
                readback_buffer: vk::Buffer::null(),
                readback_memory: vk::DeviceMemory::null(),
                bytes: 64,
                last_use: 0,
                last_snapshot: Arc::downgrade(&snapshot),
            },
        );

        let (_, upload) = caches
            .compute_image_entry(&key, &snapshot)
            .expect("cached image");
        assert!(!upload, "the same live snapshot keeps GPU-newer content");
        assert_eq!(caches.stats.compute_image_uploads_skipped, 1);

        let next_submission = Arc::new(vec![0x11; 64]);
        let (_, upload) = caches
            .compute_image_entry(&key, &next_submission)
            .expect("cached image");
        assert!(
            upload,
            "equal bytes from a different submission need a fresh upload"
        );
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

    #[test]
    fn depth_target_keys_include_guest_identity_extent_and_format() {
        let a = DepthTargetKey {
            base: 0x4000,
            width: 1280,
            height: 720,
            format: vk::Format::D32_SFLOAT.as_raw(),
        };
        assert_ne!(a, DepthTargetKey { base: 0x8000, ..a });
        assert_ne!(a, DepthTargetKey { width: 1920, ..a });
        assert_ne!(
            a,
            DepthTargetKey {
                format: vk::Format::D16_UNORM.as_raw(),
                ..a
            }
        );
    }
}
