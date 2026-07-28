//! Offscreen triangle draw with pixel readback.
//!
//! Renders one triangle into a device-local `R8G8B8A8_UNORM` image using
//! Vulkan 1.3 dynamic rendering (no render-pass/framebuffer objects), copies
//! the image into a host-visible buffer, and hands the pixels back to the
//! caller.
//!
//! This exists so the GPU draw path is verifiable **headlessly**: no window, no
//! swapchain, no surface extensions. A test can assert on actual rasterized
//! pixels, which is the only honest proof that the pipeline drew anything.
//! Presentation to a real swapchain is a separate, later concern.

use super::cache::{
    BlendKey, DepthPipelineKey, DepthTargetKey, DepthTargetLayout, DrawCaches, GpuPresentKey,
    GraphicsPipelineKey, PendingDrawResources, PersistentDepthTarget, PersistentTarget,
    PersistentTexture, StencilKey, TargetContent, TargetKey, TargetLayout, TextureKey,
};
use super::instance::VulkanDevice;
use super::shaders::{triangle_fragment_spirv, triangle_vertex_spirv};
use ash::vk::Handle;
use ash::{Device, util::Align, vk};
use raeen_core::error::GpuError;
use std::{
    mem,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tracing::debug;

/// The color the attachment is cleared to before the draw, as linear RGBA.
///
/// Deliberately not black and not fully-saturated: a readback buffer that was
/// never written (all zeroes) or a mis-sized copy cannot masquerade as a
/// correct clear.
pub const CLEAR_COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
static GPU_PLUGIN_FRAME_INDEX: AtomicU64 = AtomicU64::new(0);

/// Triangle vertices in Vulkan normalized device coordinates (`x, y, z, w`).
///
/// Vulkan's NDC has +Y pointing **down**, so the `-0.7` vertex is the top one.
/// The triangle is sized to cover the image center while leaving every corner
/// outside it — that gap is exactly what the acceptance test asserts on.
const TRIANGLE_VERTICES: [[f32; 4]; 3] = [
    [0.0, -0.7, 0.0, 1.0], // top
    [0.7, 0.7, 0.0, 1.0],  // bottom right
    [-0.7, 0.7, 0.0, 1.0], // bottom left
];

/// A rendered image read back from the GPU: tightly-packed rows.
#[derive(Debug, Clone)]
pub struct RenderedImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * bytes_per_pixel` bytes, row-major, no padding.
    pub pixels: Vec<u8>,
    /// Bytes per pixel of the render target: 4 for the 8-bit RGBA/BGRA and
    /// packed B10G11R11 formats, 8 for R16G16B16A16 (HDR). Everything that
    /// slices `pixels` per pixel must use this rather than assuming 4.
    pub bytes_per_pixel: u32,
}

impl RenderedImage {
    /// The first-4-bytes-as-RGBA at `(x, y)`, or `None` if out of bounds. Only
    /// meaningful for the 4-byte display formats; HDR targets are re-sampled as
    /// textures, not read pixel-by-pixel for display.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = ((y * self.width + x) * self.bytes_per_pixel) as usize;
        self.pixels
            .get(offset..offset + 4)
            .map(|p| [p[0], p[1], p[2], p[3]])
    }
}

/// Bytes per pixel the offscreen readback must copy for a render-target format,
/// or an error naming an unsupported one. The readback is a raw byte copy, so
/// only the size matters here (the format's meaning is handled where it is
/// sampled). Packed 32-bit HDR (B10G11R11) is 4 bytes like RGBA8; R16G16B16A16
/// is 8.
pub(crate) fn readback_bpp(format: vk::Format) -> Result<u32, GpuError> {
    match format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::R32_SFLOAT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::B10G11R11_UFLOAT_PACK32
        | vk::Format::A2B10G10R10_UNORM_PACK32 => Ok(4),
        vk::Format::R16G16B16A16_SFLOAT => Ok(8),
        other => Err(GpuError::VulkanInitFailed(format!(
            "render target format {other:?} has no readback byte size mapping"
        ))),
    }
}

/// Convert a linear float color to the `R8G8B8A8_UNORM` bytes it should
/// produce, for comparing against readback.
pub fn unorm8(color: [f32; 4]) -> [u8; 4] {
    color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// One guest vertex-buffer binding uploaded for a draw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexBufferData {
    pub bytes: Vec<u8>,
    pub stride: u32,
    /// `true` when Gen5 `fetch_index` selects the instance index.
    pub per_instance: bool,
}

/// Vulkan's interpretation of one analyzed guest vertex attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttributeData {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}

/// One descriptor binding containing an array of storage buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageBufferBinding {
    pub binding: u32,
    /// Submission-owned snapshots. `Arc` makes repeated descriptors for the
    /// same guest allocation cheap: the command processor captures the bytes
    /// once and every dispatch in that PM4 submission shares them.
    pub buffers: Vec<std::sync::Arc<Vec<u8>>>,
    /// Stable guest identity for each descriptor. A zero address denotes a
    /// synthetic/null/test buffer and deliberately bypasses the persistent
    /// compute-buffer cache.
    pub guest_bases: Vec<u64>,
    /// Unpadded guest byte lengths, parallel to `buffers`. Vulkan descriptor
    /// storage is dword-padded; deferred writeback must never publish that pad.
    pub guest_sizes: Vec<usize>,
    /// One flag per `buffers` entry. Only `ReadWrite` guest V# descriptors
    /// need a post-dispatch host readback; `ReadOnly` and `Constant` entries
    /// are inputs and must never be copied back over guest memory.
    pub writable: Vec<bool>,
}

/// The raw EUD-window fallback SSBO (SharpEmu port): a dispatch-time
/// snapshot of the guest memory behind the shader's EUD base pointer, bound
/// as the `%eud_raw` uint array the recompiled `s_load` fallback reads
/// (see `kyty_graphics::shader::ShaderEudRawResources`; SharpEmu binds its
/// pooled window the same way —
/// `reference/sharpemu/src/SharpEmu.ShaderCompiler/`
/// `Gen5ShaderScalarEvaluator.cs:1939-1980`). Read-only: never written by
/// the shader, never written back to guest memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EudRawBinding {
    pub binding: u32,
    /// Snapshot bytes (dword multiple, at least 4). May be shorter than the
    /// shader's furthest constant offset — the recompiled reads clamp
    /// against the bound size and yield 0 beyond it.
    pub bytes: Vec<u8>,
}

/// One storage image (UAV) a translated compute shader reads and writes.
///
/// The pixels are the guest's initial content (tightly packed rows, `depth`
/// slices back to back for a 3D UAV); `guest_base` is where the caller
/// writes the post-dispatch content back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageImageUpload {
    pub width: u32,
    pub height: u32,
    /// Volume depth: the T# DEPTH field + 1 for a 3D UAV, 1 otherwise
    /// (measured: ASTRO.BOT's 240x135x64 UAV volumes).
    pub depth: u32,
    /// The guest descriptor is a 3D volume (type 10) — create a
    /// `VK_IMAGE_TYPE_3D` image with a `TYPE_3D` view.
    ///
    /// Type-driven, exactly like [`Self::array`], and deliberately NOT
    /// `depth > 1`: SPIR-V's `Dim` operand is part of the image type, and the
    /// recompiler declares `Dim3D` from the TYPE nibble alone
    /// (`SampledDim::from_texture_type(10) == Three`). A type-10 T# whose
    /// DEPTH field is 0 is a legal one-slice volume with `depth == 1`, so the
    /// count-derived test bound a `TYPE_2D` view under a `Dim3D` image type —
    /// the same emit/bind divergence class as the arrayed case. Measured
    /// shape: GTA V's tile-5 single-voxel type-10 descriptor.
    pub volume: bool,
    /// Array layers: 1 for 2D/3D, or the selected
    /// `T#.BASE_ARRAY..=T#.LAST_ARRAY` span for a type-13 2D-array storage
    /// image. Type-11 writable cubes use the same 2D-array storage
    /// representation; sampled cube views remain a separate path.
    pub layers: u32,
    /// The guest descriptor is arrayed (type 11 or 13). This remains true
    /// even when `layers == 1`: SPIR-V's `Arrayed` operand is part of the
    /// image type, so Vulkan must bind a `TYPE_2D_ARRAY` view for a one-layer
    /// array instead of silently changing it to `TYPE_2D`.
    pub array: bool,
    /// Guest swizzle mode. Compute works on a linear host image; writeback
    /// retiles each array layer to this layout before publishing guest bytes.
    pub tile_mode: u8,
    /// The Vulkan texel format matching the recompiled SPIR-V's `%ImageL`
    /// declaration: `R8G8B8A8_UNORM` (Rgba8), `R16G16B16A16_SFLOAT`
    /// (Rgba16f, guest T# format 71), or `R32G32B32A32_SFLOAT`
    /// (Rgba32f, guest T# format 77).
    pub format: vk::Format,
    /// Initial linear content,
    /// `width * height * depth * layers * texel_bytes()` bytes.
    /// Shared per-submission snapshot. Repeated dispatches binding the same
    /// descriptor retain this allocation and can prove that the persistent
    /// GPU image already carries newer ordered contents.
    pub pixels: Arc<Vec<u8>>,
    /// Guest address of the selected base array layer for post-dispatch
    /// writeback (not necessarily the allocation's layer-zero address).
    pub guest_base: u64,
}

impl StorageImageUpload {
    /// Bytes per texel of [`Self::format`].
    #[must_use]
    pub fn texel_bytes(&self) -> u32 {
        match self.format {
            vk::Format::R16G16B16A16_SFLOAT => 8,
            vk::Format::R32G32B32A32_SFLOAT => 16,
            _ => 4,
        }
    }
}

/// One or more descriptor bindings containing arrays of storage images.
///
/// The recompiled SPIR-V declares `%textures2D_L` as
/// `OpTypeImage %float <dim> 0 0 0 2 <format>` — a STORAGE_IMAGE array in
/// the upload's format — indexed by dword 0 of the rewritten T# it reads
/// from the push constants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageImageBinding {
    pub binding: u32,
    pub images: Vec<StorageImageUpload>,
    /// Per-(Dim, format) storage-image descriptor groups for a MIXED shader
    /// (measured: ASTRO.BOT compute writes a 3D Rgba16f volume next to 2D
    /// Rgba16f targets). The recompiled SPIR-V declares one
    /// `%textures2D_L<key>` array per present key, each at its own binding;
    /// a group's `view_indices` select which `images` entries (in per-key
    /// order) fill that array. Empty for a homogeneous shader, which binds
    /// the whole `images` list as one array at `binding` (unchanged path).
    pub groups: Vec<SampledGroup>,
}

/// A guest texture decoded to linear pixels, ready to upload as a sampled
/// image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureUpload {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    /// Linear (de-tiled) pixel data, tightly packed rows, `layers` images
    /// (or `depth` slices for a volume) back to back.
    pub pixels: Vec<u8>,
    /// Array layers: 1 for a plain 2D texture, 6 for a cube map, the T#
    /// depth field + 1 for a 2DArray (type 13). Always 1 for a 3D volume
    /// (`depth > 1`).
    pub layers: u32,
    /// Create the view as `CUBE` (requires `layers == 6`).
    pub cube: bool,
    /// Create the view as `VK_IMAGE_VIEW_TYPE_2D_ARRAY` (T# type 13, 2DArray).
    ///
    /// This is the single carrier of the emit/bind agreement for the arrayed
    /// case: the recompiled SPIR-V declares the sampled `OpTypeImage` with
    /// `Arrayed = 1` whenever `SampledDim::from_texture_type(t.type_())` is
    /// `TwoArray` — a decision made purely from the T# TYPE nibble, independent
    /// of the layer count. The bound view type MUST match, so it is decided from
    /// this same flag (set in `draw_translate::texture_view_kind`) and never
    /// from `layers > 1`: a 2DArray descriptor whose depth field is 0 has
    /// `layers == 1` yet is still `Arrayed = 1` in SPIR-V. Binding a plain
    /// `VK_IMAGE_VIEW_TYPE_2D` there is the ASTRO.BOT array/cube device-loss
    /// (`VUID-vkCmdDispatch`: view type 2D under an `Arrayed = 1` image). A
    /// `TYPE_2D_ARRAY` view with `layer_count == 1` is valid and matches.
    pub array: bool,
    /// Volume depth: the T# DEPTH field + 1 for a 3D volume (T# type 10,
    /// measured: ASTRO.BOT's 240x135x64 froxel/LUT volumes), 1 otherwise.
    pub depth: u32,
    /// Create the image as `VK_IMAGE_TYPE_3D` with a `VK_IMAGE_VIEW_TYPE_3D`
    /// view (T# type 10, 3D volume).
    ///
    /// The [`Self::array`] argument applies verbatim to `Dim`: the recompiled
    /// SPIR-V declares `Dim3D` from the TYPE nibble alone, so the bound view
    /// must be decided from this type-driven flag and never from `depth > 1`.
    /// A type-10 descriptor whose DEPTH field is 0 is a one-slice volume with
    /// `depth == 1` — still `Dim3D` in SPIR-V, and binding a `TYPE_2D` view
    /// there is the same emit/bind divergence as a one-layer array bound as
    /// plain 2D. Measured shape: GTA V's tile-5 single-voxel type-10 T#.
    pub volume: bool,
    /// When `Some(base)`, this T# names a live persistent render target
    /// (`CB_COLOR0_BASE == base`, matching extent and format): the draw binds
    /// that target's `VkImage` directly as the sampled descriptor instead of
    /// uploading `pixels` (which are then empty and ignored). This is how a
    /// composite samples its scene targets without a GPU→CPU→GPU round trip
    /// (stage B). Must be `None` for the draw's own attachment (feedback
    /// loop) — the CPU-upload path handles that case.
    pub render_target: Option<u64>,
    /// Guest source identity for the persistent-texture cache (stage D): the
    /// T#'s 40-bit base address. `0` disables caching for this upload
    /// (fixture/test uploads, the compute path, `RAEEN_NO_TEX_CACHE=1`) —
    /// the upload then behaves exactly as before the cache existed.
    pub guest_base: u64,
    /// Sparse sample-hash of the guest SOURCE bytes (computed by the decode
    /// path — see `draw_translate::guest_sample_hash` and the invalidation
    /// contract on [`super::cache::PersistentTexture`]). `0` = no hash, not
    /// cacheable.
    pub sample_hash: u64,
    /// The decode was skipped because the cache holds this texture with an
    /// equal sample-hash: `pixels` is empty and the backend binds the cached
    /// image's view directly.
    pub cached: bool,
}

/// Rate-limited diagnostic (first 8 occurrences) for a cube upload that reaches
/// a create site with an invalid layer count. Names the upload's identity so
/// the upstream path that produced it — the one bypassing `decode_texture`'s
/// cube clamp — can be found and fixed at the source.
fn warn_bad_cube_layers_once(upload: &TextureUpload, safe: u32) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
        tracing::warn!(
            layers = upload.layers,
            safe,
            extent = format_args!("{}x{}", upload.width, upload.height),
            depth = upload.depth,
            guest_base = format_args!("{:#x}", upload.guest_base),
            render_target = ?upload.render_target,
            cached = upload.cached,
            pixels = upload.pixels.len(),
            "cube texture upload has an invalid (<6 / non-multiple-of-6) layer \
             count; clamping arrayLayers to keep the CUBE image spec-valid \
             (prevents VK_ERROR_DEVICE_LOST). This upstream path bypassed \
             decode_texture's cube clamp — investigate the source."
        );
    }
}

/// Width/height, in texels, of one addressable element of `format`.
///
/// 1 for every uncompressed format — one element is one texel. 4 for the
/// block-compressed (BC) family, whose smallest addressable unit is a 4x4
/// texel block: a 1024x1024 BC7 image is 256x256 *elements* of 16 bytes, not
/// 1024x1024 of anything. Every size, tiling, and staging computation has to
/// run in elements; only the `VkImage` extent stays in texels.
///
/// Ports SharpEmu's block/element split (`Agc/AgcExports.cs`
/// `TryGetTextureElementLayout`, `elementsWide = (width + 3) / 4`).
pub(crate) const fn format_block_extent(format: vk::Format) -> u32 {
    match format {
        vk::Format::BC1_RGBA_UNORM_BLOCK
        | vk::Format::BC1_RGBA_SRGB_BLOCK
        | vk::Format::BC2_UNORM_BLOCK
        | vk::Format::BC2_SRGB_BLOCK
        | vk::Format::BC3_UNORM_BLOCK
        | vk::Format::BC3_SRGB_BLOCK
        | vk::Format::BC4_UNORM_BLOCK
        | vk::Format::BC4_SNORM_BLOCK
        | vk::Format::BC5_UNORM_BLOCK
        | vk::Format::BC5_SNORM_BLOCK
        | vk::Format::BC6H_UFLOAT_BLOCK
        | vk::Format::BC6H_SFLOAT_BLOCK
        | vk::Format::BC7_UNORM_BLOCK
        | vk::Format::BC7_SRGB_BLOCK => 4,
        _ => 1,
    }
}

/// Bytes per addressable element for sampled texture uploads accepted by
/// `draw_translate::texture_vk_format` — per texel for uncompressed formats,
/// per 4x4 block for the BC family.
fn texture_texel_bytes(format: vk::Format) -> Result<u32, GpuError> {
    match format {
        vk::Format::R8_UNORM | vk::Format::R8_UINT => Ok(1),
        vk::Format::R16_UNORM | vk::Format::R8G8_UNORM => Ok(2),
        vk::Format::B10G11R11_UFLOAT_PACK32
        | vk::Format::R8G8B8A8_UNORM
        | vk::Format::R32_SFLOAT
        | vk::Format::R16G16_SFLOAT => Ok(4),
        vk::Format::R16G16B16A16_UNORM | vk::Format::R16G16B16A16_SFLOAT => Ok(8),
        vk::Format::R32G32B32A32_SFLOAT => Ok(16),
        // BC blocks: 8 bytes for the two-colour-endpoint families (BC1/BC4),
        // 16 for everything else. Matches SharpEmu's
        // `GetBlockCompressedBlockBytes` (`Agc/AgcExports.cs:8226-8231`).
        vk::Format::BC1_RGBA_UNORM_BLOCK
        | vk::Format::BC1_RGBA_SRGB_BLOCK
        | vk::Format::BC4_UNORM_BLOCK
        | vk::Format::BC4_SNORM_BLOCK => Ok(8),
        vk::Format::BC2_UNORM_BLOCK
        | vk::Format::BC2_SRGB_BLOCK
        | vk::Format::BC3_UNORM_BLOCK
        | vk::Format::BC3_SRGB_BLOCK
        | vk::Format::BC5_UNORM_BLOCK
        | vk::Format::BC5_SNORM_BLOCK
        | vk::Format::BC6H_UFLOAT_BLOCK
        | vk::Format::BC6H_SFLOAT_BLOCK
        | vk::Format::BC7_UNORM_BLOCK
        | vk::Format::BC7_SRGB_BLOCK => Ok(16),
        other => Err(GpuError::VulkanInitFailed(format!(
            "sampled texture format {other:?} has no texel byte size mapping"
        ))),
    }
}

fn warn_short_texture_upload_once(upload: &TextureUpload, required: usize) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
        tracing::warn!(
            supplied = upload.pixels.len(),
            required,
            extent = format_args!("{}x{}x{}", upload.width, upload.height, upload.depth),
            layers = upload.layers,
            cube = upload.cube,
            array = upload.array,
            guest_base = format_args!("{:#x}", upload.guest_base),
            "sampled texture upload is shorter than its declared image; padding \
             missing texels with zero to prevent a staging-buffer overrun"
        );
    }
}

impl TextureUpload {
    /// Vulkan requires a CUBE image's `arrayLayers` to be a positive multiple of
    /// 6 (`VUID-VkImageCreateInfo-flags-08866`) and a CUBE view's `layerCount`
    /// to be 6. `draw_translate::decode_texture` already clamps cube layers to a
    /// whole number of cubes, but this is the AUTHORITATIVE last line of defence
    /// at the create site: any path that hands a cube upload with fewer layers
    /// would otherwise make the driver accept a `<6`-layer CUBE image and then
    /// lose the device on the first sample (measured: a Minecraft draw whose
    /// cube T# reached `vkCreateImage` with `layers == 1`). Returns the
    /// spec-valid layer count, warning (rate-limited) when it has to bump.
    pub(crate) fn cube_safe_layers(&self) -> u32 {
        if self.cube && (self.layers == 0 || !self.layers.is_multiple_of(6)) {
            let safe = self.layers.max(1).next_multiple_of(6);
            warn_bad_cube_layers_once(self, safe);
            safe
        } else {
            self.layers
        }
    }

    /// The staging pixels sized for the image dimensions Vulkan will copy.
    /// Undersized inputs are zero-padded whether they came from a clamped cube
    /// layer count or from an upstream pixel-vector mismatch. This is the
    /// authoritative last line of defence against a copy region exceeding its
    /// staging buffer (`VUID-vkCmdCopyBufferToImage-pRegions-00171`, measured
    /// on Minecraft's title-panorama cube).
    pub(crate) fn staging_pixels(
        &self,
        img_layers: u32,
    ) -> Result<std::borrow::Cow<'_, [u8]>, GpuError> {
        let texel_bytes = texture_texel_bytes(self.format)? as usize;
        // Sizes run in ELEMENTS: for a BC format one element is a 4x4 texel
        // block, so a 1024x1024 BC7 surface stages 256*256*16 bytes, not
        // 1024*1024*16. Sizing it in texels would demand (and zero-pad to) 16x
        // the real data and make every BC upload look catastrophically short.
        let block = format_block_extent(self.format) as usize;
        let elements_wide = (self.width as usize).div_ceil(block);
        let elements_high = (self.height as usize).div_ceil(block);
        let required = elements_wide
            .checked_mul(elements_high)
            .and_then(|n| n.checked_mul(self.depth.max(1) as usize))
            .and_then(|n| n.checked_mul(img_layers.max(1) as usize))
            .and_then(|n| n.checked_mul(texel_bytes))
            .ok_or_else(|| {
                GpuError::VulkanInitFailed(format!(
                    "sampled texture staging size overflow for {}x{}x{} layers={} format={:?}",
                    self.width, self.height, self.depth, img_layers, self.format
                ))
            })?;
        if self.pixels.len() >= required || self.pixels.is_empty() {
            return Ok(std::borrow::Cow::Borrowed(&self.pixels));
        }
        warn_short_texture_upload_once(self, required);
        let mut padded = vec![0u8; required];
        padded[..self.pixels.len()].copy_from_slice(&self.pixels);
        Ok(std::borrow::Cow::Owned(padded))
    }
}

/// The sampled-image + sampler descriptor arrays one translated stage binds.
///
/// The recompiled SPIR-V declares `%textures2D_S` (an array of sampled images)
/// and `%samplers` (an array of samplers) and indexes them with the values the
/// push constants carry, so the arrays here must match the analyzer's counts
/// exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerState {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
}

impl SamplerState {
    #[must_use]
    pub const fn nearest_repeat() -> Self {
        Self {
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureBinding {
    pub sampled_binding: u32,
    pub sampler_binding: u32,
    pub textures: Vec<TextureUpload>,
    /// One fully-decoded host state per guest S#.
    pub samplers: Vec<SamplerState>,
    /// Per-Dim sampled-image descriptor groups for a MIXED-dim shader. The
    /// recompiled SPIR-V declares one `%textures2D_S<dim>` array per present
    /// Dim (2D, 3D, Cube, 2DArray), each at its own binding; a group's
    /// `view_indices` select which `textures` entries (in per-Dim order) fill
    /// that array. Empty for a homogeneous shader, which binds the whole
    /// `textures` list as one array at `sampled_binding` (unchanged path).
    pub sampled_groups: Vec<SampledGroup>,
}

/// One per-Dim sampled-image descriptor array of a mixed-dim shader:
/// `binding` is the Vulkan binding of its `%textures2D_S<dim>` array,
/// `view_indices` name the `TextureBinding::textures` entries that fill it in
/// SPIR-V array order (the T# index the shader seeds for a descriptor equals
/// its position in this list).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampledGroup {
    pub binding: u32,
    pub view_indices: Vec<usize>,
}

/// Per-stage resource ABI used by translated SPIR-V.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderStageBinding {
    pub stage: vk::ShaderStageFlags,
    pub descriptor_set_slot: u32,
    pub push_constant_offset: u32,
    pub push_constants: Vec<u8>,
    /// When present, `push_constants` is uploaded to this STORAGE_BUFFER
    /// descriptor instead of passed to `vkCmdPushConstants`. The translated
    /// SPIR-V declares the same tightly packed resource table in
    /// StorageBuffer storage class (Uniform/std140 cannot represent its
    /// four-byte inner array stride).
    pub push_uniform_binding: Option<u32>,
    pub storage_buffers: Option<StorageBufferBinding>,
    pub textures: Option<TextureBinding>,
    /// Storage images (UAVs). Compute-only today: the graphics draw path
    /// rejects a stage that carries these rather than silently ignoring them.
    pub storage_images: Option<StorageImageBinding>,
    /// Descriptor binding index of the `%gds` SSBO when the recompiled shader
    /// uses GDS (`ds_append`/`ds_consume` with the gds bit). The buffer itself
    /// is the device-persistent 64 KiB GDS arena (`DrawCaches::gds_buffer`) —
    /// counters must persist across dispatches (measured: ASTRO.BOT feeds
    /// indirect-draw args through them). Compute-only today.
    pub gds_binding: Option<u32>,
    /// Raw EUD-window fallback SSBO (SharpEmu port). Compute-only today —
    /// detection is wired on the CS translate path; the graphics draw path
    /// rejects a stage that carries it rather than binding nothing.
    pub eud_raw: Option<EudRawBinding>,
}

/// Register-derived alpha-blend state for the single color attachment.
///
/// `Default` is blending off with the conventional ONE/ZERO factors, so a
/// fixture preset (`DrawState::new`) behaves exactly like the old hardcoded
/// `blend_enable(false)` pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendState {
    pub enable: bool,
    pub src_color: vk::BlendFactor,
    pub dst_color: vk::BlendFactor,
    pub color_op: vk::BlendOp,
    pub src_alpha: vk::BlendFactor,
    pub dst_alpha: vk::BlendFactor,
    pub alpha_op: vk::BlendOp,
    /// `CB_BLEND_{RED,GREEN,BLUE,ALPHA}` — feeds CONSTANT_* factors.
    pub constants: [f32; 4],
}

impl Default for BlendState {
    fn default() -> Self {
        Self {
            enable: false,
            src_color: vk::BlendFactor::ONE,
            dst_color: vk::BlendFactor::ZERO,
            color_op: vk::BlendOp::ADD,
            src_alpha: vk::BlendFactor::ONE,
            dst_alpha: vk::BlendFactor::ZERO,
            alpha_op: vk::BlendOp::ADD,
            constants: [0.0; 4],
        }
    }
}

/// One additional colour attachment (MRT slot 1–7) of a multi-render-target
/// draw. The primary attachment is `DrawState`'s own width/height/format/
/// blend/write-mask; extras share the primary extent — the hardware requires
/// matching CB extents — and carry their own format, blend, write mask, and
/// guest identity.
///
/// Extra targets are per-draw resources (no persistent-image cache): an MRT
/// draw always takes the immediate path, seeds each extra from `initial`
/// (the framebuffer map's prior readback of that guest base) or CLEARs, and
/// reads every attachment back so the sink can land each in the map.
#[derive(Debug, Clone)]
pub struct MrtAttachment {
    /// Guest CB slot (1..=7), diagnostic only.
    pub slot: u8,
    pub format: vk::Format,
    /// This slot's nibble of `CB_TARGET_MASK`.
    pub write_mask: vk::ColorComponentFlags,
    /// This slot's `CB_BLEND{n}_CONTROL` (blend constants stay shared —
    /// `CB_BLEND_RED..ALPHA` are per-context, not per-target).
    pub blend: BlendState,
    /// `CB_COLOR{n}_BASE` — where the sink files the readback.
    pub target_base: u64,
    /// Prior contents (readback-format bytes, primary extent) for a LOAD
    /// seed; `None` clears to transparent black.
    pub initial: Option<Vec<u8>>,
}

/// Everything one offscreen draw needs, with nothing hardcoded.
///
/// This is the parameter object [`render_draw`] takes so a caller can drive a
/// draw from decoded PM4 register state instead of the fixture constants.
/// `render_triangle_with_spirv` is now just a preset of this.
#[derive(Debug, Clone)]
pub struct DrawState<'a> {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub clear_color: [f32; 4],
    /// `[left, top, right, bottom]` in pixels.
    pub scissor: [i32; 4],
    /// `[x, y, width, height]` in pixels.
    pub viewport: [f32; 4],
    pub topology: vk::PrimitiveTopology,
    pub cull_mode: vk::CullModeFlags,
    /// Winding that counts as FRONT, from `PA_SU_SC_MODE_CNTL.FACE`
    /// (0 = counter-clockwise is front, 1 = clockwise is front).
    /// This was hardcoded COUNTER_CLOCKWISE, which was harmless only while
    /// `cull_mode` was permanently NONE. Decoding PA_SU_SC_MODE_CNTL turned
    /// culling ON, so an unwired winding would cull exactly the wrong faces.
    pub front_face: vk::FrontFace,
    /// Which colour channels the draw may write, from `CB_TARGET_MASK`.
    ///
    /// Vulkan expresses this natively, so a guest mask maps straight through
    /// rather than being a limitation: a title that writes RGB and leaves alpha
    /// alone (mask 0x7) is doing something completely ordinary.
    pub color_write_mask: vk::ColorComponentFlags,
    /// Alpha blending, from `CB_BLEND0_CONTROL` + `CB_BLEND_*`.
    pub blend: BlendState,
    /// Host vertex data, or `None` for a shader that synthesizes its own
    /// geometry from `gl_VertexIndex` (Kyty's embedded VS does exactly this and
    /// declares no input attributes, so binding one would be invalid).
    pub vertices: Option<&'a [[f32; 4]]>,
    /// Guest-backed vertex buffers and translated Vulkan attributes.
    pub vertex_buffers: Vec<VertexBufferData>,
    pub vertex_attributes: Vec<VertexAttributeData>,
    /// Descriptor sets and push constants required by translated stages.
    pub stage_bindings: Vec<ShaderStageBinding>,
    pub vertex_count: u32,
    pub vs_spirv: &'a [u32],
    pub fs_spirv: &'a [u32],
    /// Prior contents of the render target (tightly-packed RGBA,
    /// `width * height * 4` bytes). When present the attachment is seeded
    /// with these pixels and `load_op` is LOAD instead of CLEAR — this is
    /// how successive draws into the same guest render target compose into
    /// one frame across otherwise independent one-shot draw submissions.
    pub initial: Option<&'a [u8]>,
    /// Guest identity of the render target (`CB_COLOR0_BASE`), enabling the
    /// persistent-target fast path: the backend keeps one `VkImage` per
    /// (base, extent, format) alive across draws, and when that image still
    /// holds exactly the pixels of the last readback the seed upload of
    /// `initial` is skipped — the attachment LOADs straight from the GPU copy.
    ///
    /// Invariant the caller owns: with `target_base` set, `initial` must be
    /// either `None` or byte-identical to the previous draw's readback of this
    /// target (which is what `OffscreenDrawSink`'s framebuffer map stores).
    /// A caller that substitutes other pixels for a target it names here must
    /// leave `target_base` as `None` or the substituted seed may be ignored.
    pub target_base: Option<u64>,
    /// The bound index buffer for an indexed draw, or `None` for a
    /// vertex-order (auto) draw. When present the draw is `vkCmdDrawIndexed`
    /// and `vertex_count` is the index count; the vertices are pulled through
    /// this buffer instead of straight from the vertex stream.
    pub index: Option<IndexBinding<'a>>,
    /// Whether the draw produces a colour attachment. False for a depth-only
    /// draw (`CB_TARGET_MASK == 0` with depth active — a z-prepass); `format`
    /// and `initial` are then unused.
    pub color_output: bool,
    /// Depth/stencil state; `None` = no depth attachment (colour-only draw).
    pub depth: Option<DepthState<'a>>,
    /// Additional colour attachments (MRT slots 1–7). Non-empty only when
    /// `color_output` — the primary attachment is always slot 0. Forces the
    /// immediate (non-deferred) path.
    pub mrt: Vec<MrtAttachment>,
}

/// A guest index buffer fetched into host memory, ready to upload.
#[derive(Debug, Clone)]
pub struct IndexBinding<'a> {
    /// Tightly-packed index data, `index_count * element_bytes` long.
    pub bytes: &'a [u8],
    /// 8-, 16-, or 32-bit indices.
    pub index_type: vk::IndexType,
}

/// Register-derived depth/stencil state for one draw — the depth counterpart
/// of [`BlendState`]. `draw_translate::depth_state_from_regs` builds it from
/// the PM4 register model; [`render_draw`] turns it into the depth attachment,
/// the pipeline's depth-stencil state, and the depth readback.
#[derive(Debug, Clone)]
pub struct DepthState<'a> {
    /// Guest `DB_Z_WRITE_BASE`. Nonzero enables the persistent depth-target
    /// cache, keyed together with extent and format.
    pub target_base: Option<u64>,
    /// The depth attachment's format, from `DB_Z_INFO.format * 2 +
    /// DB_STENCIL_INFO.format` (Kyty GraphicsRender.cpp L3829): D16_UNORM (2),
    /// D24_UNORM_S8_UINT (3), D32_SFLOAT (6), D32_SFLOAT_S8_UINT (7).
    pub format: vk::Format,
    /// `DB_DEPTH_CONTROL.z_enable`.
    pub test_enable: bool,
    /// `DB_DEPTH_CONTROL.z_write_enable`.
    pub write_enable: bool,
    /// `DB_DEPTH_CONTROL.zfunc` — the Gen5 compare codes match `vk::CompareOp`.
    pub compare_op: vk::CompareOp,
    /// `DB_DEPTH_CONTROL.stencil_enable` (only reachable with a stencil plane).
    pub stencil_test_enable: bool,
    /// Front-face stencil state (`DB_STENCIL_CONTROL` + `DB_STENCILREFMASK`).
    pub stencil_front: vk::StencilOpState,
    /// Back-face state — the `_BF` registers when `backface_enable`, else a
    /// copy of the front (Kyty GraphicsRender.cpp L3916).
    pub stencil_back: vk::StencilOpState,
    /// `DB_RENDER_CONTROL.depth_clear_enable`: attachment load-op CLEAR instead
    /// of LOAD (Kyty GraphicsRender.cpp L1508).
    pub clear_depth: bool,
    /// `DB_RENDER_CONTROL.stencil_clear_enable`.
    pub clear_stencil: bool,
    /// `DB_DEPTH_CLEAR`.
    pub clear_depth_value: f32,
    /// `DB_STENCIL_CLEAR`.
    pub clear_stencil_value: u32,
    /// `[min, max]` viewport depth from `PA_CL_VPORT_ZOFFSET`/`_ZSCALE` (Kyty
    /// GraphicsRender.cpp L2124: min = zoffset, max = zoffset + zscale).
    pub viewport_depth: [f32; 2],
    /// Prior depth-plane contents for a LOAD, mirroring [`DrawState::initial`]:
    /// `width * height * depth_texel_bytes(format)` bytes, tightly packed.
    pub initial: Option<&'a [u8]>,
    /// Prior stencil-plane contents (`width * height` bytes) for a LOAD.
    pub initial_stencil: Option<&'a [u8]>,
}

/// A depth/stencil attachment read back from the GPU: the depth plane, plus
/// the stencil plane when the format has one. This is both the persistence
/// unit across draws (keyed by guest `DB_Z_WRITE_BASE`, mirroring the colour
/// framebuffer map) and what tests assert on.
#[derive(Debug, Clone)]
pub struct DepthImage {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    /// Depth aspect, tightly packed rows, `width * height *
    /// depth_texel_bytes(format)` bytes.
    pub depth: Vec<u8>,
    /// Stencil aspect, tightly packed rows, `width * height` bytes.
    pub stencil: Option<Vec<u8>>,
}

impl DepthImage {
    /// The depth value at `(x, y)` as a float in [0, 1] (UNORM formats are
    /// normalized; D24 uses the low 24 bits of each 4-byte texel, matching the
    /// Vulkan buffer-copy layout of `D24_UNORM_S8_UINT`'s depth aspect).
    #[must_use]
    pub fn depth_at(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let texel = depth_texel_bytes(self.format).ok()? as usize;
        let offset = ((y * self.width + x) as usize) * texel;
        let bytes = self.depth.get(offset..offset + texel)?;
        Some(match self.format {
            vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => {
                f32::from_le_bytes(bytes[..4].try_into().expect("4-byte depth texel"))
            }
            vk::Format::D16_UNORM => {
                f32::from(u16::from_le_bytes(
                    bytes[..2].try_into().expect("2-byte depth texel"),
                )) / f32::from(u16::MAX)
            }
            vk::Format::D24_UNORM_S8_UINT => {
                let raw =
                    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
                raw as f32 / 16_777_215.0
            }
            _ => return None,
        })
    }

    /// The stencil value at `(x, y)`, when the format has a stencil plane.
    #[must_use]
    pub fn stencil_at(&self, x: u32, y: u32) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.stencil
            .as_ref()?
            .get((y * self.width + x) as usize)
            .copied()
    }
}

/// What one offscreen draw produced. `color` is `None` for a depth-only draw
/// (`CB_TARGET_MASK == 0` with depth active — a z-prepass); `depth` is `Some`
/// whenever a depth attachment was bound.
#[derive(Debug, Clone)]
pub struct DrawOutput {
    pub color: Option<RenderedImage>,
    pub depth: Option<DepthImage>,
    /// Readbacks of the extra MRT attachments, `(guest base, image)` in
    /// `DrawState::mrt` order. Empty for a single-target draw.
    pub mrt_colors: Vec<(u64, RenderedImage)>,
}

/// Copy `size` bytes from a just-mapped host-visible allocation into an owned
/// `Vec`, FALLIBLY, unmapping `memory` in every case. A large render-target
/// readback under host memory pressure must DEGRADE (return an error the
/// draw/flush path skips on) rather than abort the whole process via the
/// infallible global allocator — the same "degrade, not abort" policy as
/// `draw_translate::alloc_zeroed` and the compute-readback path. Measured on
/// ASTRO.BOT: enabling more of the scene composite pushed a 4K render-target
/// readback past this memory-constrained host's page file.
///
/// SAFETY: `ptr` must be a valid, initialized mapping of at least `size` bytes
/// backed by `memory`, with no other live reference into it.
unsafe fn readback_to_vec_fallible(
    device: &Device,
    memory: vk::DeviceMemory,
    ptr: *mut std::ffi::c_void,
    size: usize,
    what: &str,
) -> Result<Vec<u8>, GpuError> {
    let mut pixels: Vec<u8> = Vec::new();
    if pixels.try_reserve_exact(size).is_err() {
        // SAFETY: the caller mapped `memory`; unmap it before bailing.
        unsafe { device.unmap_memory(memory) };
        return Err(GpuError::VulkanInitFailed(format!(
            "{what}: {size} B host allocation failed (out of memory) — \
             skipping instead of aborting"
        )));
    }
    // SAFETY: `ptr` covers `size` initialized bytes (caller's contract); the
    // reservation above guarantees the copy does not reallocate.
    unsafe {
        pixels.extend_from_slice(std::slice::from_raw_parts(ptr.cast::<u8>(), size));
        device.unmap_memory(memory);
    }
    Ok(pixels)
}

/// Bytes per depth-aspect texel in the readback/upload of `format`.
fn depth_texel_bytes(format: vk::Format) -> Result<u32, GpuError> {
    match format {
        vk::Format::D16_UNORM => Ok(2),
        vk::Format::D32_SFLOAT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => {
            Ok(4)
        }
        other => Err(GpuError::VulkanInitFailed(format!(
            "depth format {other:?} has no texel byte size mapping"
        ))),
    }
}

/// Whether `format` carries a stencil plane.
fn has_stencil_plane(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
    )
}

/// The image aspect mask for a depth format (DEPTH, plus STENCIL when present).
fn depth_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    let mut aspects = vk::ImageAspectFlags::DEPTH;
    if has_stencil_plane(format) {
        aspects |= vk::ImageAspectFlags::STENCIL;
    }
    aspects
}

/// Byte size of the depth plane of a `width`x`height` surface, which is also
/// the offset the stencil plane sits at in the upload/readback buffers. Always
/// a multiple of 4 for the stencil-bearing formats (their depth texel is 4
/// bytes), satisfying the D/S copy offset alignment rule.
fn depth_plane_bytes(width: u32, height: u32, format: vk::Format) -> Result<u64, GpuError> {
    Ok(u64::from(width) * u64::from(height) * u64::from(depth_texel_bytes(format)?))
}

/// Which attachment planes LOAD prior contents (vs CLEAR), from the register
/// clear flags and the availability of prior contents. Mirrors Kyty's
/// `loadOp = clear_enable ? CLEAR : LOAD` (GraphicsRender.cpp L1508-1511); a
/// LOAD with no prior contents falls back to CLEAR — the colour path's
/// no-`initial` behaviour.
fn depth_loads(depth: &DepthState) -> (bool, bool) {
    let depth_load = !depth.clear_depth && depth.initial.is_some();
    let stencil_load =
        has_stencil_plane(depth.format) && !depth.clear_stencil && depth.initial_stencil.is_some();
    (depth_load, stencil_load)
}

/// One depth/stencil plane copy region for the upload/readback transfers.
fn depth_copy_region(
    width: u32,
    height: u32,
    aspect: vk::ImageAspectFlags,
    buffer_offset: u64,
) -> vk::BufferImageCopy {
    vk::BufferImageCopy::default()
        .buffer_offset(buffer_offset)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
}

/// One dynamic-rendering depth/stencil attachment referencing `view`: LOAD when
/// `load` (prior contents were seeded), else CLEAR to the register clear value;
/// STORE always, so the result is available for readback. Depth and stencil
/// share the same clear-value struct — the driver applies the plane matching
/// the attachment slot it is bound to.
fn depth_stencil_attachment(
    view: vk::ImageView,
    load: bool,
    depth: &DepthState,
) -> vk::RenderingAttachmentInfo<'static> {
    vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .load_op(if load {
            vk::AttachmentLoadOp::LOAD
        } else {
            vk::AttachmentLoadOp::CLEAR
        })
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: depth.clear_depth_value,
                stencil: depth.clear_stencil_value,
            },
        })
}

impl<'a> DrawState<'a> {
    /// A full-target viewport/scissor at `width` x `height`, no vertex buffer.
    #[must_use]
    pub fn new(width: u32, height: u32, vs_spirv: &'a [u32], fs_spirv: &'a [u32]) -> Self {
        Self {
            width,
            height,
            format: vk::Format::R8G8B8A8_UNORM,
            clear_color: CLEAR_COLOR,
            scissor: [0, 0, width as i32, height as i32],
            viewport: [0.0, 0.0, width as f32, height as f32],
            // The fixture preset writes every channel; a guest draw overrides
            // this from CB_TARGET_MASK.
            color_write_mask: vk::ColorComponentFlags::RGBA,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            blend: BlendState::default(),
            vertices: None,
            vertex_buffers: Vec::new(),
            vertex_attributes: Vec::new(),
            stage_bindings: Vec::new(),
            vertex_count: 0,
            vs_spirv,
            fs_spirv,
            initial: None,
            target_base: None,
            index: None,
            color_output: true,
            depth: None,
            mrt: Vec::new(),
        }
    }

    /// Internal-resolution scaling (Settings ▸ Video ▸ Resolution Scale):
    /// supersample by rendering into a target `factor`× larger with a
    /// proportionally scaled viewport and scissor. Guest vertices are in
    /// resolution-independent NDC, so this only changes the sample count, not
    /// the image — the color/depth targets both derive from `width`/`height`,
    /// so scaling those keeps them matched. `factor` is clamped to a sane range;
    /// **`1.0` is an exact no-op** (the default, so it changes nothing).
    pub fn scale_resolution(&mut self, factor: f32) {
        let factor = if factor.is_finite() {
            factor.clamp(0.5, 4.0)
        } else {
            1.0
        };
        if (factor - 1.0).abs() < f32::EPSILON {
            return;
        }
        let scale_u = |v: u32| ((v as f32 * factor).round() as u32).max(1);
        self.width = scale_u(self.width);
        self.height = scale_u(self.height);
        for v in &mut self.viewport {
            *v *= factor;
        }
        for s in &mut self.scissor {
            *s = (*s as f32 * factor).round() as i32;
        }
    }
}

/// Draw once offscreen from an explicit [`DrawState`] and read back the pixels.
///
/// # Errors
///
/// [`GpuError::VulkanInitFailed`] on a zero-sized or non-32bpp target or any
/// resource/submission failure, [`GpuError::ShaderCompilationFailed`] on empty
/// SPIR-V, [`GpuError::PipelineCreationFailed`] if the pipeline is rejected.
/// Draw once offscreen from an explicit [`DrawState`] and read back the pixels.
///
/// Returns a [`DrawOutput`]: the colour readback when the draw has colour
/// output, and the depth/stencil readback when a depth attachment was bound.
///
/// # Errors
///
/// [`GpuError::VulkanInitFailed`] on a zero-sized or non-32bpp target or any
/// resource/submission failure, [`GpuError::ShaderCompilationFailed`] on empty
/// SPIR-V, [`GpuError::PipelineCreationFailed`] if the pipeline is rejected.
pub fn render_draw(dev: &VulkanDevice, state: &DrawState) -> Result<DrawOutput, GpuError> {
    dump_draw_state_resources(state);
    // Full pre-submit draw identity for device-loss forensics (RAEEN_NO_DEFER=1
    // makes each draw synchronous, so the LAST line here before a device-lost is
    // the faulting draw). Gated to keep the hot path quiet.
    if crate::diagnostics::gpu_env().trace_draw_state {
        let tex: Vec<String> = state
            .stage_bindings
            .iter()
            .filter_map(|s| s.textures.as_ref())
            .flat_map(|t| t.textures.iter())
            .map(|u| {
                format!(
                    "T#{{base:{:#x} {}x{}x{} l{} cube{} rt:{:?} fmt:{:?} bytes:{}}}",
                    u.guest_base,
                    u.width,
                    u.height,
                    u.depth,
                    u.layers,
                    u.cube,
                    u.render_target,
                    u.format,
                    u.pixels.len()
                )
            })
            .collect();
        let attrs: Vec<String> = state
            .vertex_attributes
            .iter()
            .map(|a| format!("loc{}@{}:{:?}", a.location, a.offset, a.format))
            .collect();
        let vbs: Vec<String> = state
            .vertex_buffers
            .iter()
            .map(|b| format!("stride{} len{}", b.stride, b.bytes.len()))
            .collect();
        let binds: Vec<String> = state
            .stage_bindings
            .iter()
            .map(|s| {
                let push_head = s
                    .push_constants
                    .chunks_exact(4)
                    .take(16)
                    .map(|w| u32::from_le_bytes(w.try_into().expect("four-byte chunk")))
                    .map(|w| format!("{w:08x}"))
                    .collect::<Vec<_>>();
                let storage_lens = s
                    .storage_buffers
                    .as_ref()
                    .map(|b| {
                        b.buffers
                            .iter()
                            .map(|bytes| bytes.len())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                format!(
                    "{:?}:push{}={push_head:?}:sbuf={storage_lens:?}",
                    s.stage,
                    s.push_constants.len()
                )
            })
            .collect();
        tracing::warn!(
            rt = format_args!(
                "{}x{} {:?} base:{:?}",
                state.width, state.height, state.format, state.target_base
            ),
            topo = format_args!("{:?}", state.topology),
            cull = format_args!("{:?}/{:?}", state.cull_mode, state.front_face),
            viewport = format_args!("{:?}", state.viewport),
            scissor = format_args!("{:?}", state.scissor),
            color_write_mask = format_args!("{:?}", state.color_write_mask),
            blend = format_args!("{:?}", state.blend),
            depth = state.depth.is_some(),
            vcount = state.vertex_count,
            indexed = state.index.is_some(),
            vs_words = state.vs_spirv.len(),
            fs_words = state.fs_spirv.len(),
            binds = format_args!("{binds:?}"),
            vbs = format_args!("{vbs:?}"),
            attrs = format_args!("{attrs:?}"),
            tex = format_args!("{tex:?}"),
            "TRACE_DRAW_STATE"
        );
    }
    if state.width == 0 || state.height == 0 {
        return Err(GpuError::VulkanInitFailed(format!(
            "invalid render target size {}x{}",
            state.width, state.height
        )));
    }
    if state.vs_spirv.is_empty() || state.fs_spirv.is_empty() {
        return Err(GpuError::ShaderCompilationFailed(
            "vertex and fragment SPIR-V must be non-empty".to_owned(),
        ));
    }
    if !state.color_output && state.depth.is_none() {
        return Err(GpuError::VulkanInitFailed(
            "draw with neither colour nor depth output".to_owned(),
        ));
    }
    let bpp = if state.color_output {
        Some(readback_bpp(state.format)?)
    } else {
        None
    };

    // The cache lock spans the whole draw, including the synchronous fence
    // wait — that is the contract that makes the cached command buffer,
    // fence, and descriptor pool reusable (see `super::cache` module docs).
    let mut caches = dev.draw_caches();
    let timing = crate::diagnostics::gpu_env().time_draw;
    let mut res = Resources::new(dev, &mut caches);
    let t0 = std::time::Instant::now();
    res.build(state)?;
    let t_build = t0.elapsed();
    let t1 = std::time::Instant::now();
    res.record_and_submit(state)?;
    res.commit_auxiliary_layouts(false);
    // The fence was waited (immediate mode): cache-eligible texture uploads
    // are complete on the device and can join the persistent-texture cache.
    res.donate_textures_to_cache();
    let t_submit = t1.elapsed();
    let t2 = std::time::Instant::now();
    let color = res.read_back_color(state, bpp)?;
    let depth = res.read_back_depth(state)?;
    let mrt_colors = res.read_back_mrt(state)?;
    // The colour readback landed, so the persistent GPU image and the pixels
    // handed back to the caller are byte-identical again: the next draw into
    // this target may LOAD straight from the GPU copy.
    res.mark_target_synced();
    let t_readback = t2.elapsed();
    drop(res);
    if timing {
        let stats = caches.stats;
        tracing::warn!(
            build_us = t_build.as_micros(),
            submit_us = t_submit.as_micros(),
            readback_us = t_readback.as_micros(),
            pipeline_hits = stats.pipeline_hits,
            pipeline_misses = stats.pipeline_misses,
            target_hits = stats.target_hits,
            target_misses = stats.target_misses,
            seed_uploads_skipped = stats.seed_uploads_skipped,
            "TIME_DRAW: per-draw phase timing (cache counters are cumulative)"
        );
        return Ok(DrawOutput {
            color,
            depth,
            mrt_colors,
        });
    }

    debug!(
        width = state.width,
        height = state.height,
        vertices = state.vertex_count,
        "offscreen draw rendered on {}",
        dev.device_name()
    );
    Ok(DrawOutput {
        color,
        depth,
        mrt_colors,
    })
}

/// Dump one selected draw's fully translated host resources for offline
/// replay. `RAEEN_DUMP_DRAW_TARGET` is the guest target base (hex, with or
/// without `0x`) and `RAEEN_DUMP_DRAW_STATE` is a local output directory.
/// This is deliberately generic: title bytes stay in the caller-selected,
/// gitignored diagnostic directory and never become repository fixtures.
fn dump_draw_state_resources(state: &DrawState) {
    let env = crate::diagnostics::gpu_env();
    let (Some(target), Some(dir)) = (
        env.dump_draw_target.as_deref(),
        env.dump_draw_state.as_deref(),
    ) else {
        return;
    };
    let target = target.strip_prefix("0x").unwrap_or(target);
    let Ok(target) = u64::from_str_radix(target, 16) else {
        return;
    };
    if state.target_base != Some(target) || dir.is_empty() {
        return;
    }
    let dir = std::path::Path::new(dir);
    let write = |name: String, bytes: &[u8]| {
        let path = dir.join(name);
        if let Err(error) = std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, bytes))
        {
            debug!(%error, path = %path.display(), "translated draw resource dump failed");
        }
    };
    write(
        "vs.spv".to_owned(),
        &state
            .vs_spirv
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    write(
        "ps.spv".to_owned(),
        &state
            .fs_spirv
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    for (index, buffer) in state.vertex_buffers.iter().enumerate() {
        write(format!("vertex_{index}.bin"), &buffer.bytes);
    }
    for (stage_index, stage) in state.stage_bindings.iter().enumerate() {
        write(
            format!("stage_{stage_index}_push.bin"),
            &stage.push_constants,
        );
        if let Some(storage) = &stage.storage_buffers {
            for (index, buffer) in storage.buffers.iter().enumerate() {
                write(format!("stage_{stage_index}_storage_{index}.bin"), buffer);
            }
        }
        if let Some(textures) = &stage.textures {
            for (index, texture) in textures.textures.iter().enumerate() {
                write(
                    format!("stage_{stage_index}_texture_{index}.bin"),
                    &texture.pixels,
                );
            }
        }
    }
}

/// Deferred-readback (stage B) variant of [`render_draw`] for the title path.
///
/// The draw's commands are recorded into a per-draw command buffer without
/// submitting or reading the target back. The persistent target is marked
/// GPU-newer; the flush submits every recorded draw/dispatch in order with one
/// queue call and fence, then performs one batch readback. This is what turns
/// per-draw fence+readback cost (measured 11–12 ms/draw on ASTRO.BOT) into a
/// per-flush cost.
///
/// Returns `Ok(None)` when the draw was deferred. Returns `Ok(Some(image))`
/// when the draw fell back to the immediate path — an unnamed output or
/// `RAEEN_NO_DEFER=1` (the A/B switch) — in which case the readback happened
/// now, exactly as [`render_draw`]. Persistent depth-only passes are deferred:
/// their cache-owned image remains the content authority and no caller uses
/// the immediate CPU depth readback.
///
/// # Errors
///
/// Same as [`render_draw`].
fn has_only_named_persistent_outputs(
    color_output: bool,
    named_color: bool,
    depth_output: bool,
    named_depth: bool,
) -> bool {
    (!color_output || named_color) && (!depth_output || named_depth) && (named_color || named_depth)
}

pub fn render_draw_deferred(
    dev: &VulkanDevice,
    state: &DrawState,
) -> Result<Option<RenderedImage>, GpuError> {
    let force_immediate = crate::diagnostics::gpu_env().no_defer;
    let named_color = state.color_output && state.target_base.is_some();
    let named_depth = state
        .depth
        .as_ref()
        .and_then(|depth| depth.target_base)
        .is_some();
    if force_immediate
        // MRT draws are immediate-only: the extra attachments are per-draw
        // resources whose readbacks the caller files by guest base, which the
        // deferred batch cannot express. The caller that populates `mrt`
        // (draw_common) calls `render_draw` directly so those readbacks are
        // not lost; this route is the safety net for any other caller.
        || !state.mrt.is_empty()
        || !has_only_named_persistent_outputs(
            state.color_output,
            named_color,
            state.depth.is_some(),
            named_depth,
        )
    {
        return Ok(render_draw(dev, state)?.color);
    }
    if state.width == 0 || state.height == 0 {
        return Err(GpuError::VulkanInitFailed(format!(
            "invalid render target size {}x{}",
            state.width, state.height
        )));
    }
    if state.vs_spirv.is_empty() || state.fs_spirv.is_empty() {
        return Err(GpuError::ShaderCompilationFailed(
            "vertex and fragment SPIR-V must be non-empty".to_owned(),
        ));
    }
    let timing = crate::diagnostics::gpu_env().time_draw;
    let mut caches = dev.draw_caches();
    let mut res = Resources::new(dev, &mut caches);
    res.batched = true;
    let t0 = std::time::Instant::now();
    res.build(state)?;
    let t_build = t0.elapsed();
    let t1 = std::time::Instant::now();
    res.record_and_submit(state)?;
    res.commit_to_batch()?;
    let t_submit = t1.elapsed();
    drop(res);
    if timing {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TIMED_DRAWS: AtomicU64 = AtomicU64::new(0);
        let draw = TIMED_DRAWS.fetch_add(1, Ordering::Relaxed) + 1;
        // Per-draw warnings materially perturb a title that records dozens of
        // draws per frame. Sample sparsely so this diagnostic can identify the
        // build-vs-record wall without becoming that wall itself.
        if draw.is_multiple_of(512) {
            let stats = caches.stats;
            tracing::warn!(
                draw,
                build_us = t_build.as_micros(),
                submit_us = t_submit.as_micros(),
                deferred_draws = stats.deferred_draws,
                sampled_target_binds = stats.sampled_target_binds,
                texture_cache_hits = stats.texture_cache_hits,
                texture_cache_misses = stats.texture_cache_misses,
                batch_pool_creates = stats.batch_pool_creates,
                "TIME_DRAW: deferred draw (no fence wait, no readback — the flush pays those once)"
            );
        }
    }
    Ok(None)
}

/// Flush the pending deferred-draw batch: fence the queue once, read every
/// touched persistent target back once, retire the batch's per-draw
/// resources, and return the readbacks in draw order (`(guest base, image)`),
/// ready to merge into the CPU-side framebuffer map.
///
/// No-op (empty vec) when nothing is pending. This is the ONLY readback point
/// for deferred draws; callers invoke it when the pixels are actually needed
/// — end of a draw-bearing submission, presentation, frame dumps, or a
/// feedback-loop CPU fallback.
///
/// # Errors
///
/// [`GpuError::VulkanInitFailed`] on any submission/wait/map failure. The
/// batch's resources are still retired safely (best-effort device wait) and
/// every touched target degrades to [`TargetContent::Unknown`] — the next
/// draw seeds from the (stale) CPU pixels rather than trusting an image in an
/// unknown state.
pub fn flush_deferred_draws(dev: &VulkanDevice) -> Result<Vec<(u64, RenderedImage)>, GpuError> {
    flush_deferred_draws_filtered(dev, None)
}

/// Render-target readbacks keyed by their guest base address.
pub type RenderedTargets = Vec<(u64, RenderedImage)>;

/// Flush native targets plus any successfully produced ABI-v3 presentation
/// images. Native pixels remain the guest framebuffer authority; plugin images
/// are display-only.
pub fn flush_deferred_draws_with_gpu_plugins(
    dev: &VulkanDevice,
) -> Result<(RenderedTargets, RenderedTargets), GpuError> {
    flush_deferred_draws_filtered_timed(dev, None).map(|flush| (flush.images, flush.plugin_images))
}

/// [`flush_deferred_draws`] with the readback optionally restricted to a
/// small set of guest base addresses (stage C item 2: at a flip, only the
/// flipped/presented target's pixels are needed on the CPU).
///
/// With `only_bases = Some(bases)`, the flush still fences and retires EVERY
/// pending draw (their command buffers completed under the same fence), but
/// copies back only the touched targets whose base is in `bases`. The other
/// touched targets stay GPU-side — content state [`TargetContent::GpuNewer`],
/// re-queued as touched — until a later flush whose filter (or lack of one)
/// selects them. `None` reads every touched target back, exactly the old
/// behaviour.
///
/// # Errors
///
/// Same as [`flush_deferred_draws`].
pub fn flush_deferred_draws_filtered(
    dev: &VulkanDevice,
    only_bases: Option<&[u64]>,
) -> Result<Vec<(u64, RenderedImage)>, GpuError> {
    flush_deferred_draws_filtered_timed(dev, only_bases).map(|flush| flush.images)
}

/// Cheap always-on timing for the two blocking host portions of a deferred
/// present flush. Queue-drain and colour conversion are measured by the AGC
/// session; this layer owns the Vulkan fence and host readback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DeferredFlushTiming {
    pub fence_wait_us: u64,
    pub readback_us: u64,
}

/// Timed form used by the presentation HUD. Public callers keep the legacy
/// image-only wrapper above so diagnostics do not leak into the rendering API.
pub(crate) struct DeferredFlush {
    pub images: RenderedTargets,
    /// GPU-plugin results keyed by the native target base. These are for
    /// presentation only and must never replace the native framebuffer seed.
    pub plugin_images: RenderedTargets,
    pub timing: DeferredFlushTiming,
}

pub(crate) fn flush_deferred_draws_filtered_timed(
    dev: &VulkanDevice,
    only_bases: Option<&[u64]>,
) -> Result<DeferredFlush, GpuError> {
    let timing =
        crate::diagnostics::gpu_env().time_draw || crate::diagnostics::gpu_env().time_compute;
    let t0 = std::time::Instant::now();
    let mut caches = dev.draw_caches();
    caches.finish_batch_recording(dev)?;
    if !caches.batch_open() {
        return Ok(DeferredFlush {
            images: Vec::new(),
            plugin_images: Vec::new(),
            timing: DeferredFlushTiming::default(),
        });
    }
    let (pending, already_submitted, touched, evicted, evicted_depth) = caches.take_batch();
    let (read_now, keep): (Vec<TargetKey>, Vec<TargetKey>) = match only_bases {
        Some(bases) => touched.into_iter().partition(|k| bases.contains(&k.base)),
        None => (touched, Vec::new()),
    };
    // Nothing pending to fence and the filter selected nothing: put the
    // unread targets back and do no GPU work at all.
    if pending.is_empty() && read_now.is_empty() && evicted.is_empty() && evicted_depth.is_empty() {
        caches.requeue_touched(keep);
        return Ok(DeferredFlush {
            images: Vec::new(),
            plugin_images: Vec::new(),
            timing: DeferredFlushTiming::default(),
        });
    }
    let pending_draws = pending.len();
    let gpu_at = std::time::Instant::now();
    match record_and_read_flush(dev, &mut caches, &pending, already_submitted, &read_now) {
        Ok((mut images, plugin_images, flush_timing)) => {
            let gpu_flush = gpu_at.elapsed();
            let publish_at = std::time::Instant::now();
            let (compute_dirty_bytes, compute_dirty_spans) =
                match caches.flush_compute_buffers_to_guest(dev) {
                    Ok(stats) => stats,
                    Err(error) => {
                        for key in read_now.iter().chain(keep.iter()) {
                            caches.mark_target_unknown(key);
                        }
                        caches.retire_batch(dev, pending, evicted, evicted_depth);
                        return Err(error);
                    }
                };
            let (compute_image_bytes, compute_images) =
                match caches.flush_compute_images_to_guest(dev) {
                    Ok(outputs) => outputs,
                    Err(error) => {
                        for key in read_now.iter().chain(keep.iter()) {
                            caches.mark_target_unknown(key);
                        }
                        caches.retire_batch(dev, pending, evicted, evicted_depth);
                        return Err(error);
                    }
                };
            let compute_image_outputs = compute_images.len();
            images.extend(compute_images);
            let compute_publish = publish_at.elapsed();
            caches.prune_compute_buffers(dev);
            caches.prune_compute_images(dev);
            let retire_at = std::time::Instant::now();
            caches.retire_batch(dev, pending, evicted, evicted_depth);
            let retire = retire_at.elapsed();
            let kept_gpu_side = keep.len();
            caches.requeue_touched(keep);
            caches.stats.batch_flushes += 1;
            caches.stats.target_readbacks += images.len() as u64;
            if timing {
                use std::sync::atomic::{AtomicU64, Ordering};
                static FLUSHES: AtomicU64 = AtomicU64::new(0);
                let flush = FLUSHES.fetch_add(1, Ordering::Relaxed) + 1;
                if flush.is_multiple_of(64) {
                    tracing::warn!(
                        flush,
                        flush_us = t0.elapsed().as_micros(),
                        gpu_flush_us = gpu_flush.as_micros(),
                        fence_wait_us = flush_timing.fence_wait_us,
                        readback_us = flush_timing.readback_us,
                        compute_publish_us = compute_publish.as_micros(),
                        retire_us = retire.as_micros(),
                        pending_draws,
                        targets_read = images.len(),
                        targets_kept_gpu_side = kept_gpu_side,
                        compute_dirty_bytes,
                        compute_dirty_spans,
                        compute_image_bytes,
                        compute_image_outputs,
                        "TIME_DRAW: deferred-batch flush (one fence + one readback per selected target)"
                    );
                }
            }
            Ok(DeferredFlush {
                images,
                plugin_images,
                timing: flush_timing,
            })
        }
        Err(e) => {
            // The batch's command buffers may still be executing: wait the
            // device (best effort — a lost device fails this too) so retiring
            // their resources cannot free memory the GPU is reading.
            // SAFETY: waiting the device idle takes no handles.
            let _ = unsafe { dev.device().device_wait_idle() };
            // Unread (`keep`) targets degrade too: the device just faulted, so
            // no GPU image is trustworthy enough to LOAD from or defer again.
            for key in read_now.iter().chain(keep.iter()) {
                caches.mark_target_unknown(key);
            }
            caches.discard_pending_compute_writes();
            caches.retire_batch(dev, pending, evicted, evicted_depth);
            Err(e)
        }
    }
}

/// The flush's GPU half: record one readback copy per touched live target,
/// submit with the shared fence, wait, map. Marks each read target
/// [`TargetContent::Synced`]. Touched keys whose target was evicted mid-batch
/// are skipped (the guest re-programmed them; their pixels are unreachable
/// and unwanted).
fn record_and_read_flush(
    dev: &VulkanDevice,
    caches: &mut DrawCaches,
    pending: &[PendingDrawResources],
    already_submitted: usize,
    touched: &[TargetKey],
) -> Result<(RenderedTargets, RenderedTargets, DeferredFlushTiming), GpuError> {
    let device = dev.device();
    let live: Vec<(TargetKey, PersistentTarget, u32)> = touched
        .iter()
        .filter_map(|key| caches.target_entry(key).map(|t| (*key, t)))
        .map(|(key, t)| readback_bpp(vk::Format::from_raw(key.format)).map(|bpp| (key, t, bpp)))
        .collect::<Result<_, _>>()?;
    if let Some((key, _, _)) = live
        .iter()
        .find(|(_, target, _)| target.layout == TargetLayout::Undefined)
    {
        return Err(GpuError::VulkanInitFailed(format!(
            "flush target {:#x} has no trustworthy Vulkan layout",
            key.base
        )));
    }

    let (command_buffer, fence) = caches.submit_resources(dev)?;
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: the cached command buffer's previous submission fence-completed
    // under this same lock (submit_resources contract); begin implicitly
    // resets it.
    unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
        .map_err(|e| GpuError::VulkanInitFailed(format!("flush vkBeginCommandBuffer: {e}")))?;
    let gpu_request = crate::present_plugin::active_gpu_v3_request().filter(|request| {
        // Motion-vector extraction is not implemented yet. Keep temporal
        // plugins fail-closed instead of fabricating zero vectors/history.
        !request.capabilities.wants_motion_vectors
    });
    let plugin_sync = gpu_request.and_then(|_| dev.next_plugin_timeline());
    let mut plugin_readbacks = Vec::new();
    let mut transitioned_plugin_depths = std::collections::HashSet::new();
    for (key, target, _) in &live {
        let plugin_depth = caches.depth_target_for_color(key);
        let target_gpu_request = gpu_request
            .filter(|request| !request.capabilities.wants_depth || plugin_depth.is_some());
        let plugin_target = target_gpu_request.and_then(|request| {
            let scale = request.output_scale.clamp(0.5, 8.0);
            let width = ((key.width as f32 * scale).round() as u32).clamp(1, 16_384);
            let height = ((key.height as f32 * scale).round() as u32).clamp(1, 16_384);
            let plugin_key = GpuPresentKey {
                source_base: key.base,
                width,
                height,
                format: key.format,
            };
            match caches.gpu_present_target(
                dev,
                plugin_key,
                readback_bpp(vk::Format::from_raw(key.format)).ok()?,
            ) {
                Ok(output) => Some((plugin_key, output)),
                Err(error) => {
                    tracing::warn!(
                        target = format_args!("{:#x}", key.base),
                        %error,
                        "GPU plugin output allocation failed; using native readback"
                    );
                    None
                }
            }
        });

        let target_layout = if plugin_target.is_some() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        };
        if let Some(transition) = colour_target_flush_transition(target.layout, target_layout) {
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(transition.old_layout)
                .new_layout(transition.new_layout)
                .src_access_mask(transition.src_access)
                .dst_access_mask(transition.dst_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(target.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            // SAFETY: the flush command buffer is recording; the transition
            // starts from the cache-tracked layout produced by the last
            // successful deferred draw.
            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    transition.src_stage,
                    transition.dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }

        if let (Some((plugin_key, output_target)), Some((semaphore, wait_value, signal_value))) =
            (plugin_target, plugin_sync)
        {
            if output_target.layout != vk::ImageLayout::GENERAL {
                let (src_access, src_stage) = match output_target.layout {
                    vk::ImageLayout::UNDEFINED => (
                        vk::AccessFlags::empty(),
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                    ),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                        vk::AccessFlags::TRANSFER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    _ => (
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                    ),
                };
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(output_target.layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(src_access)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(output_target.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                // SAFETY: output image is host-owned, live, and not referenced
                // by another in-flight flush (the prior flush fence completed).
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        src_stage,
                        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
            }
            if let Some((depth_key, depth_target)) = plugin_depth
                && target_gpu_request.is_some_and(|request| request.capabilities.wants_depth)
                && transitioned_plugin_depths.insert(depth_key)
            {
                let (old_layout, src_access, src_stage) = match depth_target.layout {
                    DepthTargetLayout::Undefined => (
                        vk::ImageLayout::UNDEFINED,
                        vk::AccessFlags::empty(),
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                    ),
                    DepthTargetLayout::TransferSrc => (
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    DepthTargetLayout::DepthStencilAttachment => (
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    ),
                    DepthTargetLayout::DepthStencilReadOnly => (
                        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                    ),
                };
                if old_layout != vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL {
                    let barrier = vk::ImageMemoryBarrier::default()
                        .old_layout(old_layout)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                        .src_access_mask(src_access)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(depth_target.image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: depth_aspect_mask(vk::Format::from_raw(depth_key.format)),
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    // SAFETY: the depth image is persistent and the tracked
                    // layout was committed by its most recent draw.
                    unsafe {
                        device.cmd_pipeline_barrier(
                            command_buffer,
                            src_stage,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &[barrier],
                        );
                    }
                }
            }
            let resource_flags = crate::present_plugin::cabi_v3::RAEEN_V3_RESOURCE_BORROWED
                | crate::present_plugin::cabi_v3::RAEEN_V3_RESOURCE_HOST_OWNS_LAYOUT;
            let frame_index = GPU_PLUGIN_FRAME_INDEX.fetch_add(1, Ordering::Relaxed) + 1;
            let identity = [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ];
            let frame = crate::present_plugin::cabi_v3::RaeenPresentFrameV3 {
                struct_size: std::mem::size_of::<crate::present_plugin::cabi_v3::RaeenPresentFrameV3>(
                ) as u32,
                _reserved: 0,
                frame_index,
                command_buffer: command_buffer.as_raw(),
                color: crate::present_plugin::cabi_v3::RaeenVulkanResourceV3 {
                    image: target.image.as_raw(),
                    image_view: target.view.as_raw(),
                    device_memory: target.memory.as_raw(),
                    vk_format: key.format as u32,
                    layout: vk::ImageLayout::GENERAL.as_raw() as u32,
                    width: key.width,
                    height: key.height,
                    queue_family: dev.queue_family_index(),
                    flags: resource_flags,
                },
                depth: if target_gpu_request.is_some_and(|request| request.capabilities.wants_depth)
                {
                    let (depth_key, depth_target) =
                        plugin_depth.expect("depth capability was gated on a live target");
                    crate::present_plugin::cabi_v3::RaeenVulkanResourceV3 {
                        image: depth_target.image.as_raw(),
                        image_view: depth_target.view.as_raw(),
                        device_memory: depth_target.memory.as_raw(),
                        vk_format: depth_key.format as u32,
                        layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL.as_raw() as u32,
                        width: depth_key.width,
                        height: depth_key.height,
                        queue_family: dev.queue_family_index(),
                        flags: resource_flags,
                    }
                } else {
                    crate::present_plugin::cabi_v3::RaeenVulkanResourceV3::absent()
                },
                motion_vectors: crate::present_plugin::cabi_v3::RaeenVulkanResourceV3::absent(),
                exposure: crate::present_plugin::cabi_v3::RaeenVulkanResourceV3::absent(),
                output: crate::present_plugin::cabi_v3::RaeenVulkanResourceV3 {
                    image: output_target.image.as_raw(),
                    image_view: output_target.view.as_raw(),
                    device_memory: output_target.memory.as_raw(),
                    vk_format: key.format as u32,
                    layout: vk::ImageLayout::GENERAL.as_raw() as u32,
                    width: plugin_key.width,
                    height: plugin_key.height,
                    queue_family: dev.queue_family_index(),
                    flags: resource_flags,
                },
                render_rect: crate::present_plugin::cabi_v3::RaeenRectV3 {
                    x: 0,
                    y: 0,
                    width: key.width,
                    height: key.height,
                },
                output_rect: crate::present_plugin::cabi_v3::RaeenRectV3 {
                    x: 0,
                    y: 0,
                    width: plugin_key.width,
                    height: plugin_key.height,
                },
                temporal: crate::present_plugin::cabi_v3::RaeenTemporalDataV3 {
                    flags: 0,
                    _reserved: 0,
                    jitter_x: 0.0,
                    jitter_y: 0.0,
                    motion_vector_scale_x: 1.0,
                    motion_vector_scale_y: 1.0,
                    exposure_scale: 1.0,
                    pre_exposure: 1.0,
                    near_plane: 0.1,
                    far_plane: 1_000.0,
                    frame_time_ms: 0.0,
                    camera_view_to_clip: identity,
                    camera_clip_to_view: identity,
                    camera_clip_to_previous_clip: identity,
                    camera_previous_clip_to_clip: identity,
                },
                sync: crate::present_plugin::cabi_v3::RaeenFrameSyncV3 {
                    wait_semaphore: semaphore.as_raw(),
                    wait_value,
                    signal_semaphore: semaphore.as_raw(),
                    signal_value,
                },
            };
            let mut plugin_output = crate::present_plugin::cabi_v3::RaeenPluginOutputV3::empty();
            let status = crate::present_plugin::process_active_gpu_v3(&frame, &mut plugin_output);
            if status == crate::present_plugin::cabi_v3::RAEEN_V3_OK {
                if plugin_output.output_layout != vk::ImageLayout::GENERAL.as_raw() as u32 {
                    return Err(GpuError::VulkanInitFailed(format!(
                        "GPU plugin returned unsupported output layout {} (must remain GENERAL)",
                        plugin_output.output_layout
                    )));
                }
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(output_target.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                // SAFETY: plugin returned success after recording into this
                // command buffer and promised to leave output in GENERAL.
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: plugin_key.width,
                        height: plugin_key.height,
                        depth: 1,
                    });
                // SAFETY: output is TRANSFER_SRC and its readback buffer was
                // allocated for the complete tightly-packed image.
                unsafe {
                    device.cmd_copy_image_to_buffer(
                        command_buffer,
                        output_target.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        output_target.readback_buffer,
                        &[region],
                    );
                }
                caches.mark_gpu_present_layout(&plugin_key, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                plugin_readbacks.push((
                    *key,
                    plugin_key,
                    output_target,
                    readback_bpp(vk::Format::from_raw(key.format))?,
                ));
            } else {
                caches.mark_gpu_present_layout(&plugin_key, vk::ImageLayout::GENERAL);
            }
        }

        if plugin_target.is_some() {
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(target.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            // SAFETY: target is in GENERAL after the transition above.
            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: 1,
            });
        // SAFETY: the target either already rested in TRANSFER_SRC or the
        // barrier above moved its last attachment write there; the readback
        // buffer was sized width*height*bpp at target creation.
        unsafe {
            device.cmd_copy_image_to_buffer(
                command_buffer,
                target.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target.readback_buffer,
                &[region],
            );
        }
    }
    let host_barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    // Preserve exact PM4 order by putting every recorded deferred
    // draw/dispatch command buffer before the flush command buffer in ONE
    // VkSubmitInfo. This removes the per-packet vkQueueSubmit driver cost while
    // keeping the existing resource lifetime and one-fence contract.
    let mut command_buffers = Vec::with_capacity(pending.len().saturating_add(1));
    debug_assert!(already_submitted <= pending.len());
    command_buffers.extend(
        pending
            .iter()
            .skip(already_submitted)
            .map(|resources| resources.command_buffer)
            .filter(|buffer| *buffer != vk::CommandBuffer::null()),
    );
    command_buffers.push(command_buffer);

    // SAFETY: recording; then end/submit/wait with live handles from this
    // device. Every pending resource remains owned by the batch until this
    // fence signals, and the command buffers are ordered as they were recorded.
    let fence_wait_at = std::time::Instant::now();
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[host_barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(command_buffer)
            .map_err(|e| GpuError::VulkanInitFailed(format!("flush vkEndCommandBuffer: {e}")))?;
        if let Some((semaphore, wait_value, signal_value)) = plugin_sync {
            let wait_semaphores = [semaphore];
            let signal_semaphores = [semaphore];
            let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let wait_values = [wait_value];
            let signal_values = [signal_value];
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let submit = vk::SubmitInfo::default()
                .command_buffers(&command_buffers)
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&signal_semaphores)
                .push_next(&mut timeline);
            device
                .queue_submit(dev.queue(), &[submit], fence)
                .map_err(|e| {
                    dev.note_vk_error(e);
                    GpuError::VulkanInitFailed(format!("flush vkQueueSubmit: {e}"))
                })?;
        } else {
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .queue_submit(dev.queue(), &[submit], fence)
                .map_err(|e| {
                    dev.note_vk_error(e);
                    GpuError::VulkanInitFailed(format!("flush vkQueueSubmit: {e}"))
                })?;
        }
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| {
                dev.note_vk_error(e);
                GpuError::VulkanInitFailed(format!("flush vkWaitForFences: {e}"))
            })?;
    }
    let fence_wait_us = fence_wait_at.elapsed().as_micros() as u64;
    for key in transitioned_plugin_depths {
        caches.mark_depth_target_layout(&key, DepthTargetLayout::DepthStencilReadOnly);
    }

    let readback_at = std::time::Instant::now();
    let mut images = Vec::with_capacity(live.len());
    for (key, target, bpp) in &live {
        let size = (key.width as usize) * (key.height as usize) * (*bpp as usize);
        // SAFETY: the readback memory is HOST_VISIBLE, sized exactly `size`,
        // not currently mapped, its copy completed (fence waited) and was
        // made host-visible by the barrier above. The bytes are copied into
        // an owned Vec before unmapping.
        let ptr = unsafe {
            device.map_memory(
                target.readback_memory,
                0,
                size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("flush readback map: {e}")))?;
        // SAFETY: `ptr` maps exactly `size` initialized host-visible bytes
        // (readback_memory is sized to `size`, its copy fence-waited above); the
        // helper copies them into an owned Vec fallibly and unmaps.
        let pixels = unsafe {
            readback_to_vec_fallible(
                device,
                target.readback_memory,
                ptr,
                size,
                "render-target flush readback",
            )?
        };
        caches.mark_target_synced(key);
        images.push((
            key.base,
            RenderedImage {
                width: key.width,
                height: key.height,
                pixels,
                bytes_per_pixel: *bpp,
            },
        ));
    }
    let mut plugin_images = Vec::with_capacity(plugin_readbacks.len());
    for (source_key, plugin_key, target, bpp) in plugin_readbacks {
        let size = (plugin_key.width as usize)
            .saturating_mul(plugin_key.height as usize)
            .saturating_mul(bpp as usize);
        // SAFETY: the flush fence covers the plugin output copy and this
        // host-visible allocation was sized for the complete output.
        let ptr = unsafe {
            device.map_memory(
                target.readback_memory,
                0,
                size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("GPU plugin readback map: {e}")))?;
        // SAFETY: same fence-complete mapping contract as the native readback.
        let pixels = unsafe {
            readback_to_vec_fallible(
                device,
                target.readback_memory,
                ptr,
                size,
                "GPU plugin output readback",
            )?
        };
        plugin_images.push((
            source_key.base,
            RenderedImage {
                width: plugin_key.width,
                height: plugin_key.height,
                pixels,
                bytes_per_pixel: bpp,
            },
        ));
    }
    Ok((
        images,
        plugin_images,
        DeferredFlushTiming {
            fence_wait_us,
            readback_us: readback_at.elapsed().as_micros() as u64,
        },
    ))
}

/// Draw one triangle offscreen at `width` x `height` and read back the pixels.
///
/// # Errors
///
/// [`GpuError::VulkanInitFailed`] if any resource creation or submission fails,
/// [`GpuError::PipelineCreationFailed`] if the graphics pipeline is rejected.
pub fn render_triangle(
    dev: &VulkanDevice,
    width: u32,
    height: u32,
) -> Result<RenderedImage, GpuError> {
    render_triangle_with_spirv(
        dev,
        width,
        height,
        &triangle_vertex_spirv(),
        &triangle_fragment_spirv(),
    )
}

/// Draw one triangle offscreen using caller-supplied SPIR-V modules.
///
/// Used by the M2 path (`kyty-graphics` SPIR-V) and by the hand-built smoke
/// path via [`render_triangle`].
pub fn render_triangle_with_spirv(
    dev: &VulkanDevice,
    width: u32,
    height: u32,
    vs_spirv: &[u32],
    fs_spirv: &[u32],
) -> Result<RenderedImage, GpuError> {
    render_draw(
        dev,
        &DrawState {
            vertices: Some(&TRIANGLE_VERTICES),
            vertex_count: TRIANGLE_VERTICES.len() as u32,
            ..DrawState::new(width, height, vs_spirv, fs_spirv)
        },
    )?
    .color
    .ok_or_else(|| {
        GpuError::VulkanInitFailed(
            "triangle draw produced no colour image (colour output is on by default)".to_owned(),
        )
    })
}

/// One guest texture uploaded to the device: the staging source, the
/// device-local image, and the view the descriptor array names. `stage` picks
/// the pipeline stage the final read-transition makes the image visible to.
struct TextureGpu {
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    width: u32,
    height: u32,
    layers: u32,
    /// Volume depth (1 for 2D/cube) — the staging copy's extent depth.
    depth: u32,
    stage: vk::ShaderStageFlags,
    /// When `Some`, this upload is donated to the persistent-texture cache on
    /// draw success (batched: after the batch join; immediate: after the
    /// fence wait); the image/memory/view handles then leave this struct.
    cache_key: Option<TextureKey>,
    /// The guest-source sample-hash the donated entry is stored under.
    sample_hash: u64,
    /// Decoded byte size (cache cap accounting).
    byte_size: u64,
}

/// One image layout transition, bundled so `image_barrier` stays readable.
struct ImageTransition {
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
}

fn colour_target_attachment_transition(layout: TargetLayout) -> ImageTransition {
    match layout {
        TargetLayout::Undefined => ImageTransition {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access: vk::AccessFlags::empty(),
            dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_READ,
            src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        },
        TargetLayout::TransferSrc => ImageTransition {
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access: vk::AccessFlags::TRANSFER_READ,
            dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_READ,
            src_stage: vk::PipelineStageFlags::TRANSFER,
            dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        },
        TargetLayout::ColorAttachment => ImageTransition {
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_READ,
            src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        },
    }
}

fn colour_target_readback_transition(layout: TargetLayout) -> Option<ImageTransition> {
    match layout {
        TargetLayout::ColorAttachment => Some(ImageTransition {
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access: vk::AccessFlags::TRANSFER_READ,
            src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage: vk::PipelineStageFlags::TRANSFER,
        }),
        TargetLayout::TransferSrc | TargetLayout::Undefined => None,
    }
}

fn colour_target_flush_transition(
    layout: TargetLayout,
    destination: vk::ImageLayout,
) -> Option<ImageTransition> {
    if destination == vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        return colour_target_readback_transition(layout);
    }
    debug_assert_eq!(destination, vk::ImageLayout::GENERAL);
    let (old_layout, src_access, src_stage) = match layout {
        TargetLayout::Undefined => return None,
        TargetLayout::TransferSrc => (
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
        ),
        TargetLayout::ColorAttachment => (
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
    };
    Some(ImageTransition {
        old_layout,
        new_layout: vk::ImageLayout::GENERAL,
        src_access,
        dst_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
        src_stage,
        dst_stage: vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
    })
}

/// The pipeline stage a sampled image must become visible to, for the shader
/// stage that samples it.
fn shader_stage_to_pipeline(stage: vk::ShaderStageFlags) -> vk::PipelineStageFlags {
    if stage.contains(vk::ShaderStageFlags::VERTEX) {
        vk::PipelineStageFlags::VERTEX_SHADER
    } else if stage.contains(vk::ShaderStageFlags::FRAGMENT) {
        vk::PipelineStageFlags::FRAGMENT_SHADER
    } else if stage.contains(vk::ShaderStageFlags::COMPUTE) {
        vk::PipelineStageFlags::COMPUTE_SHADER
    } else {
        vk::PipelineStageFlags::ALL_COMMANDS
    }
}

/// Every Vulkan handle one draw uses — a mix of per-draw resources this
/// struct owns and long-lived ones borrowed from [`DrawCaches`].
///
/// Owned (destroyed in `Drop`): guest data buffers (vertex/index/storage/
/// upload), texture uploads, the whole depth/stencil set, and the colour
/// target + readback buffer **only when** `owns_target` (no `target_base`,
/// i.e. fixture/test draws). Everything else — pipeline, layouts, shader
/// modules, samplers, command buffer, fence, descriptor pool, persistent
/// colour target — belongs to the cache and must NOT be destroyed here.
///
/// Handles start null and are filled in by `build`; `Drop` destroys whatever
/// owned handle is non-null, so an error at any step during `build` cleans up
/// correctly rather than leaking GPU memory — `?` early-returns are safe here.
struct Resources<'a> {
    dev: &'a VulkanDevice,
    caches: &'a mut DrawCaches,
    /// Stage B deferred mode: the draw's commands are submitted without a
    /// fence wait or readback; per-draw resources (own command buffer, own
    /// descriptor pool, buffers, images) transfer to the caches' pending
    /// batch on success and are destroyed only after the flush fence.
    batched: bool,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    /// True when the colour target/readback pair is per-draw (owned); false
    /// when they came from (or were donated to) the persistent-target cache.
    owns_target: bool,
    /// The persistent-target identity of this draw, when `target_base` named
    /// one — used to mark the entry synced after a successful readback.
    target_key: Option<TargetKey>,
    /// The stage A seed-skip: the persistent image already holds the
    /// authoritative frame (last readback, or newer deferred draws), so the
    /// attachment LOADs from the GPU copy and the upload staging never
    /// happens.
    load_from_gpu: bool,
    /// Layout captured when the persistent colour target was acquired.
    target_layout: TargetLayout,
    /// Persistent-target images this draw samples as textures (deduplicated).
    /// Transitioned TRANSFER_SRC → SHADER_READ_ONLY before rendering and back
    /// after, preserving the between-draws layout invariant.
    sampled_targets: Vec<(TargetKey, vk::Image, TargetLayout)>,
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    /// Uploaded index buffer for an indexed draw; null for an auto draw.
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    guest_vertex_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    storage_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// Uploaded guest textures and their samplers, in stage-binding order.
    texture_uploads: Vec<TextureGpu>,
    samplers: Vec<vk::Sampler>,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    /// Always borrowed from the cache (never destroyed here): the shared
    /// resettable pool for immediate draws, or a shared batch pool — reset at
    /// the batch retire, after the flush fence — for deferred draws
    /// (stage D item 2).
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<(u32, vk::DescriptorSet)>,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    upload_buffer: vk::Buffer,
    upload_memory: vk::DeviceMemory,
    /// Depth attachment + its view (null when the draw has no depth state).
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    /// False when the depth image/readback pair belongs to `DrawCaches`.
    owns_depth_target: bool,
    depth_target_key: Option<DepthTargetKey>,
    /// A cache hit means the image rests in TRANSFER_SRC layout from the
    /// previous draw's readback, rather than UNDEFINED like a fresh image.
    depth_target_cached: bool,
    /// Layout captured when the persistent depth/stencil target was acquired.
    depth_target_layout: DepthTargetLayout,
    /// Prior depth/stencil contents for a LOAD seed (null when both planes
    /// CLEAR — the attachment then starts undefined by design).
    depth_upload_buffer: vk::Buffer,
    depth_upload_memory: vk::DeviceMemory,
    /// Depth/stencil readback — the result persists into the depth map.
    depth_readback_buffer: vk::Buffer,
    depth_readback_memory: vk::DeviceMemory,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Extra MRT colour attachments (slots 1–7), in `DrawState::mrt` order.
    /// Always per-draw owned (no persistent cache) and immediate-only.
    mrt_targets: Vec<MrtTargetRes>,
}

/// Per-draw Vulkan resources of one extra MRT attachment. Owned by
/// [`Resources`] and destroyed in its `Drop` (the seed staging buffer returns
/// to the upload ring like every other host upload).
struct MrtTargetRes {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    upload_buffer: vk::Buffer,
    upload_memory: vk::DeviceMemory,
    bpp: u32,
}

// RAEEN_TIME_DRAW: per-draw `build()` stage timers (nanoseconds, process-
// global), summarized every 512 draws. Splits the setup cost into
// image/seed/depth/vbuf/stage-resources/pipeline/command so the dominant call
// is named from evidence rather than guessed.
static DRAW_STAGE_TARGET_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAW_STAGE_SEED_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAW_STAGE_DEPTH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAW_STAGE_VBUF_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAW_STAGE_STAGE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAW_STAGE_PIPELINE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAW_STAGE_CMD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAW_STAGE_DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// Upstream worker_submit costs, OUTSIDE the offscreen `build()`: whole-draw
// translate (PM4 → DrawState, incl. vertex fetch), guest texture decode
// (detile), and compute dispatch. `build` is only ~5% of the worker, so these
// name where the rest goes. `drawcommon` is the per-draw total (translate +
// render); `decode` and the build stages are subsets of it.
pub(crate) static DRAW_STAGE_DRAWCOMMON_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_DECODE_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_RESOLVE_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_RESOLVE_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_RESOLVE_MISSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_PARSE_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_PARSE_MISSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_SETUP_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_CENSUS_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_BIND_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_BACKEND_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_DISPATCH_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_DISPATCH_N: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_CS_TRANSLATE_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_CS_PREPARE_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static DRAW_STAGE_CS_BACKEND_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn draw_stage_timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::diagnostics::gpu_env().time_draw)
}

/// Scope timer: adds elapsed ns to `counter` on drop when `RAEEN_TIME_DRAW`
/// is set. Captures every early-return path of the instrumented function.
pub(crate) struct StageTimer(
    Option<std::time::Instant>,
    &'static std::sync::atomic::AtomicU64,
);
impl StageTimer {
    pub(crate) fn start(counter: &'static std::sync::atomic::AtomicU64) -> Self {
        Self(
            draw_stage_timing_enabled().then(std::time::Instant::now),
            counter,
        )
    }
}
impl Drop for StageTimer {
    fn drop(&mut self) {
        if let Some(start) = self.0 {
            self.1.fetch_add(
                start.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }
}

fn draw_stage_add(counter: &std::sync::atomic::AtomicU64, start: Option<std::time::Instant>) {
    if let Some(start) = start {
        counter.fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

fn draw_stage_tick() {
    use std::sync::atomic::Ordering::Relaxed;
    let n = DRAW_STAGE_DRAWS.fetch_add(1, Relaxed) + 1;
    if !n.is_multiple_of(512) {
        return;
    }
    let target = DRAW_STAGE_TARGET_NS.swap(0, Relaxed);
    let seed = DRAW_STAGE_SEED_NS.swap(0, Relaxed);
    let depth = DRAW_STAGE_DEPTH_NS.swap(0, Relaxed);
    let vbuf = DRAW_STAGE_VBUF_NS.swap(0, Relaxed);
    let stage = DRAW_STAGE_STAGE_NS.swap(0, Relaxed);
    let pipeline = DRAW_STAGE_PIPELINE_NS.swap(0, Relaxed);
    let cmd = DRAW_STAGE_CMD_NS.swap(0, Relaxed);
    let build = target + seed + depth + vbuf + stage + pipeline + cmd;
    let drawcommon = DRAW_STAGE_DRAWCOMMON_NS.swap(0, Relaxed);
    let decode = DRAW_STAGE_DECODE_NS.swap(0, Relaxed);
    let resolve = DRAW_STAGE_RESOLVE_NS.swap(0, Relaxed);
    let resolve_hits = DRAW_STAGE_RESOLVE_HITS.swap(0, Relaxed);
    let resolve_misses = DRAW_STAGE_RESOLVE_MISSES.swap(0, Relaxed);
    let parse_hits = DRAW_STAGE_PARSE_HITS.swap(0, Relaxed);
    let parse_misses = DRAW_STAGE_PARSE_MISSES.swap(0, Relaxed);
    let setup = DRAW_STAGE_SETUP_NS.swap(0, Relaxed);
    let census = DRAW_STAGE_CENSUS_NS.swap(0, Relaxed);
    let bind = DRAW_STAGE_BIND_NS.swap(0, Relaxed);
    let backend = DRAW_STAGE_BACKEND_NS.swap(0, Relaxed);
    let dispatch = DRAW_STAGE_DISPATCH_NS.swap(0, Relaxed);
    let dispatches = DRAW_STAGE_DISPATCH_N.swap(0, Relaxed);
    let cs_translate = DRAW_STAGE_CS_TRANSLATE_NS.swap(0, Relaxed);
    let cs_prepare = DRAW_STAGE_CS_PREPARE_NS.swap(0, Relaxed);
    let cs_backend = DRAW_STAGE_CS_BACKEND_NS.swap(0, Relaxed);
    let bpct = |x: u64| x.saturating_mul(100).checked_div(build).unwrap_or(0);
    let dpct = |x: u64| x.saturating_mul(100).checked_div(drawcommon).unwrap_or(0);
    let per_draw_us = |x: u64| x / 1000 / 512;
    let per_disp_us = |x: u64| (x / 1000).checked_div(dispatches).unwrap_or(0);
    tracing::warn!(
        draws = 512,
        drawcommon_us = per_draw_us(drawcommon),
        decode_us = per_draw_us(decode),
        decode_of_draw_pct = dpct(decode),
        build_us = per_draw_us(build),
        build_of_draw_pct = dpct(build),
        dispatches = dispatches,
        dispatch_us = per_disp_us(dispatch),
        dispatch_total_ms = dispatch / 1_000_000,
        drawcommon_total_ms = drawcommon / 1_000_000,
        "DRAW COST: per-draw draw_common vs its texture-decode + build subsets, and compute dispatch (per 512 draws; RAEEN_TIME_DRAW)"
    );
    tracing::warn!(
        resolve_us = per_draw_us(resolve),
        resolve_hits,
        resolve_misses,
        parse_hits,
        parse_misses,
        setup_us = per_draw_us(setup),
        census_us = per_draw_us(census),
        bind_us = per_draw_us(bind),
        backend_us = per_draw_us(backend),
        "DRAW COMMON STAGES: shader resolve, state/vertex setup, cache census, resource binding, and Vulkan backend (per draw)"
    );
    tracing::warn!(
        dispatches,
        translate_us = per_disp_us(cs_translate),
        prepare_us = per_disp_us(cs_prepare),
        backend_us = per_disp_us(cs_backend),
        "COMPUTE STAGES: shader analysis/cache, guest-resource preparation, and Vulkan backend (per dispatch)"
    );
    tracing::warn!(
        pipeline_pct = bpct(pipeline),
        pipeline_us = per_draw_us(pipeline),
        stage_pct = bpct(stage),
        vbuf_pct = bpct(vbuf),
        target_pct = bpct(target),
        depth_pct = bpct(depth),
        "DRAW STAGES: build() internal split (% of build)"
    );
}

impl<'a> Resources<'a> {
    fn new(dev: &'a VulkanDevice, caches: &'a mut DrawCaches) -> Self {
        Self {
            dev,
            caches,
            batched: false,
            image: vk::Image::null(),
            image_memory: vk::DeviceMemory::null(),
            image_view: vk::ImageView::null(),
            owns_target: false,
            target_key: None,
            load_from_gpu: false,
            target_layout: TargetLayout::Undefined,
            sampled_targets: Vec::new(),
            vertex_buffer: vk::Buffer::null(),
            vertex_memory: vk::DeviceMemory::null(),
            index_buffer: vk::Buffer::null(),
            index_memory: vk::DeviceMemory::null(),
            guest_vertex_buffers: Vec::new(),
            storage_buffers: Vec::new(),
            texture_uploads: Vec::new(),
            samplers: Vec::new(),
            descriptor_set_layouts: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
            readback_buffer: vk::Buffer::null(),
            readback_memory: vk::DeviceMemory::null(),
            upload_buffer: vk::Buffer::null(),
            upload_memory: vk::DeviceMemory::null(),
            depth_image: vk::Image::null(),
            depth_memory: vk::DeviceMemory::null(),
            depth_view: vk::ImageView::null(),
            owns_depth_target: false,
            depth_target_key: None,
            depth_target_cached: false,
            depth_target_layout: DepthTargetLayout::Undefined,
            depth_upload_buffer: vk::Buffer::null(),
            depth_upload_memory: vk::DeviceMemory::null(),
            depth_readback_buffer: vk::Buffer::null(),
            depth_readback_memory: vk::DeviceMemory::null(),
            vertex_module: vk::ShaderModule::null(),
            fragment_module: vk::ShaderModule::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            mrt_targets: Vec::new(),
        }
    }

    fn device(&self) -> &Device {
        self.dev.device()
    }

    fn effective_depth_loads(&self, depth: &DepthState) -> (bool, bool) {
        let (seed_depth, seed_stencil) = depth_loads(depth);
        (
            seed_depth || (self.depth_target_cached && !depth.clear_depth),
            seed_stencil
                || (self.depth_target_cached
                    && has_stencil_plane(depth.format)
                    && !depth.clear_stencil),
        )
    }

    fn build(&mut self, state: &DrawState) -> Result<(), GpuError> {
        let t = draw_stage_timing_enabled();
        if state.color_output {
            let bpp = readback_bpp(state.format)? as usize;
            if let Some(initial) = state.initial {
                let expected = state.width as usize * state.height as usize * bpp;
                if initial.len() != expected {
                    return Err(GpuError::VulkanInitFailed(format!(
                        "initial render-target contents are {} bytes; {}x{} needs {expected}",
                        initial.len(),
                        state.width,
                        state.height
                    )));
                }
            }
            let s = t.then(std::time::Instant::now);
            self.create_render_target(state, bpp as u32)?;
            draw_stage_add(&DRAW_STAGE_TARGET_NS, s);
            if let Some(initial) = state.initial
                && !self.load_from_gpu
            {
                let s = t.then(std::time::Instant::now);
                let (buffer, memory) = self.create_buffer_with_bytes(initial)?;
                self.upload_buffer = buffer;
                self.upload_memory = memory;
                draw_stage_add(&DRAW_STAGE_SEED_NS, s);
            }
            self.create_mrt_targets(state)?;
        }
        if let Some(depth) = &state.depth {
            let s = t.then(std::time::Instant::now);
            self.create_depth_resources(state.width, state.height, depth)?;
            draw_stage_add(&DRAW_STAGE_DEPTH_NS, s);
        }
        {
            let s = t.then(std::time::Instant::now);
            if let Some(vertices) = state.vertices {
                self.create_vertex_buffer(vertices)?;
            }
            if let Some(index) = &state.index {
                let (buffer, memory) = self.create_buffer_with_bytes(index.bytes)?;
                self.index_buffer = buffer;
                self.index_memory = memory;
            }
            self.create_guest_vertex_buffers(state)?;
            draw_stage_add(&DRAW_STAGE_VBUF_NS, s);
        }
        let s = t.then(std::time::Instant::now);
        self.create_stage_resources(state)?;
        draw_stage_add(&DRAW_STAGE_STAGE_NS, s);
        let s = t.then(std::time::Instant::now);
        self.create_pipeline(state)?;
        draw_stage_add(&DRAW_STAGE_PIPELINE_NS, s);
        let s = t.then(std::time::Instant::now);
        self.create_command_resources()?;
        draw_stage_add(&DRAW_STAGE_CMD_NS, s);
        if t {
            draw_stage_tick();
        }
        Ok(())
    }

    /// Per-draw images + readback buffers (+ seed staging) for the extra MRT
    /// attachments. Extras share the primary extent; each has its own format
    /// and bpp. MRT draws are immediate-only — the deferred batch cannot file
    /// their readbacks — so a batched build with extras is refused loudly.
    fn create_mrt_targets(&mut self, state: &DrawState) -> Result<(), GpuError> {
        if state.mrt.is_empty() {
            return Ok(());
        }
        if self.batched {
            return Err(GpuError::VulkanInitFailed(
                "MRT draw reached the deferred batch — extras are immediate-only".to_owned(),
            ));
        }
        for extra in &state.mrt {
            let bpp = readback_bpp(extra.format)?;
            if let Some(initial) = &extra.initial {
                let expected = state.width as usize * state.height as usize * bpp as usize;
                if initial.len() != expected {
                    return Err(GpuError::VulkanInitFailed(format!(
                        "initial MRT{} contents are {} bytes; {}x{} needs {expected}",
                        extra.slot,
                        initial.len(),
                        state.width,
                        state.height
                    )));
                }
            }
            let (image, memory, view) =
                self.create_color_image_raw(state.width, state.height, extra.format)?;
            // Push a partially-filled record FIRST so Drop cleans the image up
            // if a later allocation in this loop fails.
            self.mrt_targets.push(MrtTargetRes {
                image,
                memory,
                view,
                readback_buffer: vk::Buffer::null(),
                readback_memory: vk::DeviceMemory::null(),
                upload_buffer: vk::Buffer::null(),
                upload_memory: vk::DeviceMemory::null(),
                bpp,
            });
            let (buffer, memory) =
                self.create_readback_buffer_raw(state.width, state.height, bpp)?;
            let record = self.mrt_targets.last_mut().expect("pushed above");
            record.readback_buffer = buffer;
            record.readback_memory = memory;
            if let Some(initial) = &extra.initial {
                let (buffer, memory) = self.create_buffer_with_bytes(initial)?;
                let record = self.mrt_targets.last_mut().expect("pushed above");
                record.upload_buffer = buffer;
                record.upload_memory = memory;
            }
        }
        Ok(())
    }

    /// Create the depth attachment image and its view. Usage covers the draw
    /// itself, the post-draw readback (TRANSFER_SRC), and seeding prior
    /// contents (TRANSFER_DST).
    fn create_depth_resources(
        &mut self,
        width: u32,
        height: u32,
        depth: &DepthState,
    ) -> Result<(), GpuError> {
        if let Some(base) = depth.target_base.filter(|base| *base != 0) {
            let key = DepthTargetKey {
                base,
                width,
                height,
                format: depth.format.as_raw(),
            };
            self.caches
                .evict_depth_targets_for_base(self.dev.device(), base, &key);
            if let Some(entry) = self.caches.acquire_depth_target(&key) {
                self.depth_image = entry.image;
                self.depth_memory = entry.memory;
                self.depth_view = entry.view;
                self.depth_readback_buffer = entry.readback_buffer;
                self.depth_readback_memory = entry.readback_memory;
                self.depth_target_key = Some(key);
                self.depth_target_layout = entry.layout;
                self.depth_target_cached = entry.layout != DepthTargetLayout::Undefined;
                self.owns_depth_target = false;
                self.create_depth_buffers(width, height, depth)?;
                return Ok(());
            }
            self.owns_depth_target = true;
            self.create_depth_target(width, height, depth.format)?;
            self.create_depth_buffers(width, height, depth)?;
            self.caches.insert_depth_target(
                key,
                PersistentDepthTarget {
                    image: self.depth_image,
                    memory: self.depth_memory,
                    view: self.depth_view,
                    readback_buffer: self.depth_readback_buffer,
                    readback_memory: self.depth_readback_memory,
                    layout: DepthTargetLayout::Undefined,
                },
            );
            self.owns_depth_target = false;
            self.depth_target_key = Some(key);
            return Ok(());
        }
        self.owns_depth_target = true;
        self.create_depth_target(width, height, depth.format)?;
        self.create_depth_buffers(width, height, depth)
    }

    fn create_depth_target(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<(), GpuError> {
        // Fail cleanly before touching Vulkan: an unsupported depth/stencil
        // format is device-specific (AMD has no D24_UNORM_S8_UINT), and some
        // drivers accept the create anyway and then error on every use.
        if !self.dev.supports_depth_stencil_attachment(format) {
            return Err(GpuError::VulkanInitFailed(format!(
                "depth/stencil format {format:?} is not supported for an OPTIMAL-tiling \
                 attachment on {}",
                self.dev.device_name()
            )));
        }
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: `info` is fully initialized and borrows nothing beyond this
        // call; the device is live. The handle is stored and destroyed in Drop.
        self.depth_image = unsafe { self.device().create_image(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("depth vkCreateImage failed: {e}")))?;

        // SAFETY: `self.depth_image` was just created from this device.
        let reqs = unsafe {
            self.device()
                .get_image_memory_requirements(self.depth_image)
        };
        let type_index = self
            .dev
            .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);

        // SAFETY: allocation size/type come from this image's own requirements.
        self.depth_memory = unsafe { self.device().allocate_memory(&alloc, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("depth allocation failed: {e}")))?;

        // SAFETY: memory was allocated for exactly this image; offset 0 is
        // within it and satisfies the alignment requirement by construction.
        unsafe {
            self.device()
                .bind_image_memory(self.depth_image, self.depth_memory, 0)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("depth bind memory failed: {e}")))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: depth_aspect_mask(format),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // SAFETY: the view's image is live and its format/range match the
        // image's creation parameters.
        self.depth_view = unsafe { self.device().create_image_view(&view_info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("depth view failed: {e}")))?;
        Ok(())
    }

    /// The depth upload buffer (only when a plane LOADs prior contents) and
    /// the readback buffer (always — the result persists into the depth map).
    fn create_depth_buffers(
        &mut self,
        width: u32,
        height: u32,
        depth: &DepthState,
    ) -> Result<(), GpuError> {
        let px = (width * height) as usize;
        let depth_bytes = px * depth_texel_bytes(depth.format)? as usize;
        let stencil_bytes = if has_stencil_plane(depth.format) {
            px
        } else {
            0
        };
        let total = (depth_bytes + stencil_bytes) as u64;

        let (depth_load, stencil_load) = depth_loads(depth);
        if depth_load || stencil_load {
            // Plane layout: depth at 0, stencil right after. A plane that
            // CLEARs is not copied, so its bytes here stay zero — harmless.
            // Fallible: a large depth-target seed under host memory pressure
            // must DEGRADE (error the draw path skips on) rather than abort the
            // process — same "degrade, not abort" policy as the readbacks.
            let mut bytes: Vec<u8> = Vec::new();
            bytes.try_reserve_exact(total as usize).map_err(|_| {
                GpuError::VulkanInitFailed(format!(
                    "depth-target seed: {total} B host allocation failed (out of memory) — \
                     skipping the draw instead of aborting"
                ))
            })?;
            bytes.resize(total as usize, 0);
            if depth_load {
                let initial = depth.initial.expect("depth LOAD implies initial");
                if initial.len() != depth_bytes {
                    return Err(GpuError::VulkanInitFailed(format!(
                        "initial depth contents are {} bytes; {}x{} needs {depth_bytes}",
                        initial.len(),
                        width,
                        height
                    )));
                }
                bytes[..depth_bytes].copy_from_slice(initial);
            }
            if stencil_load {
                let initial = depth.initial_stencil.expect("stencil LOAD implies initial");
                if initial.len() != stencil_bytes {
                    return Err(GpuError::VulkanInitFailed(format!(
                        "initial stencil contents are {} bytes; {}x{} needs {stencil_bytes}",
                        initial.len(),
                        width,
                        height
                    )));
                }
                bytes[depth_bytes..].copy_from_slice(initial);
            }
            let (buffer, memory) = self.create_buffer_with_bytes(&bytes)?;
            self.depth_upload_buffer = buffer;
            self.depth_upload_memory = memory;
        }

        if self.depth_readback_buffer != vk::Buffer::null() {
            return Ok(());
        }
        // Same cached-host preference as the colour readback (`create_readback_buffer`).
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let (buffer, memory) = self
            .create_buffer(
                total,
                vk::BufferUsageFlags::TRANSFER_DST,
                host | vk::MemoryPropertyFlags::HOST_CACHED,
            )
            .or_else(|_| self.create_buffer(total, vk::BufferUsageFlags::TRANSFER_DST, host))?;
        self.depth_readback_buffer = buffer;
        self.depth_readback_memory = memory;
        Ok(())
    }

    /// The colour target + its readback buffer, persistent when the draw
    /// names a guest render target (`target_base`), per-draw otherwise.
    ///
    /// Persistent path: one `VkImage` per (base, extent, format) lives in the
    /// cache across draws. On a hit whose entry is still synced with the last
    /// readback, the seed upload of `state.initial` is skipped entirely and
    /// the attachment LOADs from the GPU copy (`load_from_gpu`). A miss
    /// creates the image/readback pair through the owned path, then donates
    /// them to the cache.
    fn create_render_target(&mut self, state: &DrawState, bpp: u32) -> Result<(), GpuError> {
        let (width, height, format) = (state.width, state.height, state.format);
        if let Some(base) = state.target_base {
            let key = TargetKey {
                base,
                width,
                height,
                format: format.as_raw(),
            };
            // The guest re-programmed this target's extent/format: the old
            // image can never be drawn again, so drop it now.
            self.caches.evict_targets_for_base(self.dev, base, &key);
            if let Some(entry) = self.caches.acquire_target(&key) {
                self.image = entry.image;
                self.image_memory = entry.memory;
                self.image_view = entry.view;
                self.readback_buffer = entry.readback_buffer;
                self.readback_memory = entry.readback_memory;
                self.owns_target = false;
                self.target_key = Some(key);
                self.target_layout = entry.layout;
                // `entry.content` carries the pre-acquisition value:
                // - Synced: the GPU image equals the last readback, which is
                //   exactly what the caller passes as `initial` (contract on
                //   `target_base`) — LOAD from the GPU copy when a seed exists.
                // - GpuNewer: deferred draws made the GPU image the ONLY
                //   authority; LOAD from it unconditionally and never upload
                //   the (stale) CPU seed over it.
                // - Unknown: a prior draw failed mid-flight; fall back to the
                //   CPU seed (or a clear).
                self.load_from_gpu = match entry.content {
                    TargetContent::GpuNewer => true,
                    TargetContent::Synced => state.initial.is_some(),
                    TargetContent::Unknown => false,
                };
                if self.load_from_gpu {
                    self.caches.stats.seed_uploads_skipped += 1;
                }
                return Ok(());
            }
            // Own the new image/readback pair until the donation to the cache
            // below — an error in between must destroy them in Drop, not leak.
            self.owns_target = true;
            self.create_color_image(width, height, format)?;
            self.create_readback_buffer(width, height, bpp)?;
            self.caches.insert_target(
                key,
                PersistentTarget {
                    image: self.image,
                    memory: self.image_memory,
                    view: self.image_view,
                    readback_buffer: self.readback_buffer,
                    readback_memory: self.readback_memory,
                    content: TargetContent::Unknown,
                    layout: TargetLayout::Undefined,
                },
            );
            self.owns_target = false;
            self.target_key = Some(key);
            return Ok(());
        }
        self.create_color_image(width, height, format)?;
        self.create_readback_buffer(width, height, bpp)?;
        self.owns_target = true;
        Ok(())
    }

    /// Mark this draw's persistent target as synced with the CPU-side pixels.
    /// Called only after the colour readback succeeded.
    fn mark_target_synced(&mut self) {
        if let Some(key) = self.target_key {
            self.caches.mark_target_synced(&key);
        }
    }

    fn create_color_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<(), GpuError> {
        let (image, memory, view) = self.create_color_image_raw(width, height, format)?;
        self.image = image;
        self.image_memory = memory;
        self.image_view = view;
        Ok(())
    }

    /// Create a colour-attachment image + memory + view without storing it on
    /// `self` — shared by the primary target and the per-draw MRT extras. On
    /// error every partially-created handle is destroyed here, so the caller
    /// only ever owns a complete triple.
    fn create_color_image_raw(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), GpuError> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            // COLOR_ATTACHMENT to draw into it, TRANSFER_SRC to copy it out,
            // TRANSFER_DST to seed it with the target's prior contents,
            // SAMPLED so a later draw can bind it as a texture directly
            // (render-target-as-texture, stage B).
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: `info` is fully initialized and borrows nothing beyond this
        // call; the device is live. On every error path below the handles
        // created so far are destroyed before returning.
        let image = unsafe { self.device().create_image(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateImage failed: {e}")))?;

        // SAFETY: `image` was just created from this device.
        let reqs = unsafe { self.device().get_image_memory_requirements(image) };
        // SAFETY (cleanup closures): each destroys only handles created above
        // in this call, exactly once, with nothing referencing them yet.
        let type_index = match self
            .dev
            .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            Ok(index) => index,
            Err(e) => {
                unsafe { self.device().destroy_image(image, None) };
                return Err(e);
            }
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);

        // SAFETY: allocation size/type come from this image's own requirements.
        let memory = match unsafe { self.device().allocate_memory(&alloc, None) } {
            Ok(memory) => memory,
            Err(e) => {
                unsafe { self.device().destroy_image(image, None) };
                return Err(GpuError::VulkanInitFailed(format!(
                    "image allocation failed: {e}"
                )));
            }
        };

        // SAFETY: memory was allocated for exactly this image, offset 0 is
        // within it and satisfies the alignment requirement by construction.
        if let Err(e) = unsafe { self.device().bind_image_memory(image, memory, 0) } {
            unsafe {
                self.device().free_memory(memory, None);
                self.device().destroy_image(image, None);
            }
            return Err(GpuError::VulkanInitFailed(format!(
                "vkBindImageMemory failed: {e}"
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

        // SAFETY: the view's image is live and its format/range match the
        // image's creation parameters.
        let view = match unsafe { self.device().create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(e) => {
                unsafe {
                    self.device().free_memory(memory, None);
                    self.device().destroy_image(image, None);
                }
                return Err(GpuError::VulkanInitFailed(format!(
                    "vkCreateImageView failed: {e}"
                )));
            }
        };
        Ok((image, memory, view))
    }

    /// Create a buffer plus memory satisfying `properties`.
    fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: `info` is fully initialized; the device is live.
        let buffer = unsafe { self.device().create_buffer(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateBuffer failed: {e}")))?;

        // SAFETY: `buffer` was just created from this device.
        let reqs = unsafe { self.device().get_buffer_memory_requirements(buffer) };
        let type_index = match self.dev.find_memory_type(reqs.memory_type_bits, properties) {
            Ok(i) => i,
            Err(e) => {
                // SAFETY: destroying the buffer we just created and are about
                // to drop the only handle to; nothing references it yet.
                unsafe { self.device().destroy_buffer(buffer, None) };
                return Err(e);
            }
        };

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);

        // SAFETY: size/type come from this buffer's own requirements.
        let memory = match unsafe { self.device().allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: same as above — unbound buffer, sole handle.
                unsafe { self.device().destroy_buffer(buffer, None) };
                return Err(GpuError::VulkanInitFailed(format!(
                    "buffer allocation failed: {e}"
                )));
            }
        };

        // SAFETY: memory was allocated for exactly this buffer; offset 0 is in
        // range and correctly aligned by construction.
        if let Err(e) = unsafe { self.device().bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: unwinding our own two handles, neither yet in use.
            unsafe {
                self.device().free_memory(memory, None);
                self.device().destroy_buffer(buffer, None);
            }
            return Err(GpuError::VulkanInitFailed(format!(
                "vkBindBufferMemory failed: {e}"
            )));
        }
        Ok((buffer, memory))
    }

    fn create_vertex_buffer(&mut self, vertices: &[[f32; 4]]) -> Result<(), GpuError> {
        let size = mem::size_of_val(vertices) as vk::DeviceSize;
        if size == 0 {
            return Err(GpuError::VulkanInitFailed(
                "vertex buffer requested with no vertices".to_owned(),
            ));
        }
        let (buffer, memory) = self.create_buffer(
            size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            // HOST_COHERENT so the write is visible to the GPU without an
            // explicit flush.
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.vertex_buffer = buffer;
        self.vertex_memory = memory;

        // SAFETY: the memory is HOST_VISIBLE, not currently mapped, and the
        // whole allocation is requested. The GPU is not using it yet — nothing
        // has been submitted.
        let ptr = unsafe {
            self.device()
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkMapMemory failed: {e}")))?;

        // SAFETY: `ptr` maps `size` bytes allocated for this buffer, and
        // `Align` writes exactly the slice we pass within that range. The
        // alignment of `[f32; 4]` is 4, well under the mapped base alignment
        // Vulkan guarantees.
        unsafe {
            let mut align = Align::new(ptr, mem::align_of::<[f32; 4]>() as u64, size);
            align.copy_from_slice(vertices);
            self.device().unmap_memory(memory);
        }
        Ok(())
    }

    /// A host-visible buffer holding `bytes`, taken from the device's upload
    /// ring (stage C item 3) — recycled fence-tracked buffers instead of a
    /// vkCreateBuffer + vkAllocateMemory per guest upload. The pool's usage
    /// union covers every caller (vertex/index/storage/transfer-src), so no
    /// usage parameter exists any more.
    fn create_buffer_with_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        if bytes.is_empty() {
            return Err(GpuError::VulkanInitFailed(
                "buffer upload requested with no bytes".to_owned(),
            ));
        }
        let size = bytes.len() as vk::DeviceSize;
        let (buffer, memory, mapped) = self.caches.acquire_host_buffer(self.dev, size)?;

        // SAFETY: the pooled allocation is persistently mapped, coherent, and
        // at least `bytes.len()` bytes. Fence-tracked checkout guarantees that
        // no submitted GPU work can reference this buffer while it is written.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
        }
        Ok((buffer, memory))
    }

    fn create_guest_vertex_buffers(&mut self, state: &DrawState) -> Result<(), GpuError> {
        for vertex in &state.vertex_buffers {
            if vertex.stride == 0 || vertex.bytes.len() < vertex.stride as usize {
                return Err(GpuError::VulkanInitFailed(format!(
                    "invalid guest vertex buffer: {} bytes, stride {}",
                    vertex.bytes.len(),
                    vertex.stride
                )));
            }
            let allocation = self.create_buffer_with_bytes(&vertex.bytes)?;
            self.guest_vertex_buffers.push(allocation);
        }
        Ok(())
    }

    fn create_stage_resources(&mut self, state: &DrawState) -> Result<(), GpuError> {
        // Storage images are implemented only on the one-shot compute path
        // (`vulkan::compute`); a graphics stage carrying them would bind
        // nothing at the shader's STORAGE_IMAGE array and fault on the GPU.
        if let Some(stage) = state
            .stage_bindings
            .iter()
            .find(|stage| stage.storage_images.is_some())
        {
            return Err(GpuError::PipelineCreationFailed(format!(
                "{:?} stage binds storage image(s) — storage images are implemented \
                 for COMPUTE dispatches only",
                stage.stage
            )));
        }
        // Same policy for GDS: the persistent GDS arena is bound only on the
        // one-shot compute path (`vulkan::compute`).
        if let Some(stage) = state
            .stage_bindings
            .iter()
            .find(|stage| stage.gds_binding.is_some())
        {
            return Err(GpuError::PipelineCreationFailed(format!(
                "{:?} stage binds GDS — GDS is implemented for COMPUTE dispatches only",
                stage.stage
            )));
        }
        // Same policy for the raw EUD-window fallback: detection is wired on
        // the compute translate path only, so a graphics stage carrying it is
        // a named refusal rather than a shader reading an unbound SSBO.
        if let Some(stage) = state
            .stage_bindings
            .iter()
            .find(|stage| stage.eud_raw.is_some())
        {
            return Err(GpuError::PipelineCreationFailed(format!(
                "{:?} stage binds a raw EUD window — implemented for COMPUTE dispatches only",
                stage.stage
            )));
        }
        let resource_stages: Vec<_> = state
            .stage_bindings
            .iter()
            .filter(|stage| stage.storage_buffers.is_some() || stage.textures.is_some())
            .collect();
        if resource_stages.is_empty() {
            return Ok(());
        }

        // One set layout per stage, holding that stage's descriptor arrays —
        // storage buffers, sampled images, samplers — at the exact bindings
        // the recompiled SPIR-V declares (`shader_calc_binding_indices`).
        //
        // Each stage's SPIR-V decorates its resources with `DescriptorSet
        // <bind.descriptor_set_slot>`, and that slot is honoured as-is: a
        // translated PS can carry set 1 while no stage claims set 0 (the slot
        // was assigned against the VS the title bound at analysis time), so
        // gaps are filled with empty set layouts rather than refused. Two
        // stages claiming one slot IS refused: per-stage binding indices both
        // start at 0, so their bindings would collide.
        let max_slot = resource_stages
            .iter()
            .map(|stage| stage.descriptor_set_slot)
            .max()
            .expect("resource_stages is non-empty") as usize;
        // A wild slot value would come from a mis-decoded bind ABI; refuse it
        // rather than building an absurd pipeline layout.
        if max_slot >= 8 {
            return Err(GpuError::PipelineCreationFailed(format!(
                "descriptor set slot {max_slot} is out of range (max 7)"
            )));
        }
        let mut slot_layouts: Vec<Option<vk::DescriptorSetLayout>> = vec![None; max_slot + 1];
        for stage in &resource_stages {
            let slot = stage.descriptor_set_slot as usize;
            if slot_layouts[slot].is_some() {
                return Err(GpuError::PipelineCreationFailed(format!(
                    "descriptor set slot {slot} is claimed by two shader stages — \
                     per-stage binding indices both start at 0 and would collide"
                )));
            }
            let mut bindings = Vec::new();
            if let Some(storage) = &stage.storage_buffers {
                if storage.buffers.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "storage-buffer descriptor array is empty".to_owned(),
                    ));
                }
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(storage.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(storage.buffers.len() as u32)
                        .stage_flags(stage.stage),
                );
            }
            if let Some(textures) = &stage.textures {
                // The two arrays are independent: the recompiled SPIR-V
                // declares `%textures2D_S` only when the stage samples
                // textures and `%samplers` only when it binds S#s, and a
                // shader legitimately uses one without the other
                // (texel-fetch needs no sampler). Each descriptor array is
                // created only when non-empty, mirroring exactly what the
                // SPIR-V declared.
                if textures.textures.is_empty() && textures.samplers.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "sampled-image and sampler descriptor arrays are both empty".to_owned(),
                    ));
                }
                if !textures.textures.is_empty() {
                    if textures.sampled_groups.is_empty() {
                        // Homogeneous: one array of every sampled view.
                        bindings.push(
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(textures.sampled_binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(textures.textures.len() as u32)
                                .stage_flags(stage.stage),
                        );
                    } else {
                        // Mixed-dim: one `%textures2D_S<dim>` array per Dim,
                        // each at its own binding — matching the SPIR-V.
                        for group in &textures.sampled_groups {
                            bindings.push(
                                vk::DescriptorSetLayoutBinding::default()
                                    .binding(group.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .descriptor_count(group.view_indices.len() as u32)
                                    .stage_flags(stage.stage),
                            );
                        }
                    }
                }
                if !textures.samplers.is_empty() {
                    bindings.push(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(textures.sampler_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(textures.samplers.len() as u32)
                            .stage_flags(stage.stage),
                    );
                }
            }
            // Cached by binding signature: layouts live on the device caches,
            // not in this per-draw bundle.
            slot_layouts[slot] = Some(self.caches.set_layout(self.dev, &bindings)?);
        }
        for layout in slot_layouts {
            let layout = match layout {
                Some(layout) => layout,
                // Gap slot: an empty layout keeps set indices aligned with the
                // SPIR-V's `DescriptorSet` decorations.
                None => self.caches.set_layout(self.dev, &[])?,
            };
            self.descriptor_set_layouts.push(layout);
        }

        let count_of = |pick: &dyn Fn(&ShaderStageBinding) -> u32| -> u32 {
            resource_stages.iter().map(|stage| pick(stage)).sum()
        };
        let pool_sizes: Vec<_> = [
            (
                vk::DescriptorType::STORAGE_BUFFER,
                count_of(&|stage| {
                    stage
                        .storage_buffers
                        .as_ref()
                        .map_or(0, |storage| storage.buffers.len() as u32)
                }),
            ),
            (
                vk::DescriptorType::SAMPLED_IMAGE,
                count_of(&|stage| {
                    stage
                        .textures
                        .as_ref()
                        .map_or(0, |textures| textures.textures.len() as u32)
                }),
            ),
            (
                vk::DescriptorType::SAMPLER,
                count_of(&|stage| {
                    stage
                        .textures
                        .as_ref()
                        .map_or(0, |textures| textures.samplers.len() as u32)
                }),
            ),
        ]
        .into_iter()
        .filter(|(_, count)| *count != 0)
        .map(|(ty, count)| {
            vk::DescriptorPoolSize::default()
                .ty(ty)
                .descriptor_count(count)
        })
        .collect();
        if self.batched {
            // A deferred draw's descriptor sets stay live until the batch
            // fence, so it cannot use the shared resettable pool (whose
            // per-draw reset would free them). Stage D item 2: instead of a
            // fresh exact-size pool per draw (a vkCreateDescriptorPool +
            // vkDestroyDescriptorPool per deferred draw), all deferred draws
            // allocate from shared capacity-accounted batch pools that are
            // reset together at the batch retire, after the flush fence.
            self.descriptor_pool = self.caches.batch_descriptor_pool(
                self.dev,
                resource_stages.len() as u32,
                &pool_sizes,
            )?;
        } else {
            // Persistent pool from the cache, reset for this draw (the
            // previous draw's sets completed with its fence) and grown when
            // this draw needs more than it holds.
            self.descriptor_pool =
                self.caches
                    .descriptor_pool(self.dev, resource_stages.len() as u32, &pool_sizes)?;
        }

        // Allocate a set per RESOURCE stage (gap slots need no set — nothing
        // is ever bound at them), each from its own slot's layout.
        let stage_layouts: Vec<_> = resource_stages
            .iter()
            .map(|stage| self.descriptor_set_layouts[stage.descriptor_set_slot as usize])
            .collect();
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&stage_layouts);
        // SAFETY: the pool and every layout are live handles from this device.
        let sets = unsafe { self.device().allocate_descriptor_sets(&alloc_info) }.map_err(|e| {
            GpuError::PipelineCreationFailed(format!("vkAllocateDescriptorSets: {e}"))
        })?;

        for (stage, set) in resource_stages.into_iter().zip(sets) {
            // Upload resources and collect descriptor infos. The info vectors
            // must outlive `update_descriptor_sets`, so they live at stage
            // scope rather than inside each branch.
            let mut buffer_infos = Vec::new();
            let mut image_infos = Vec::new();
            // Per-Dim group image infos for a mixed-dim shader; each inner Vec
            // is one `%textures2D_S<dim>` array's descriptors. Kept at stage
            // scope so it outlives `update_descriptor_sets`.
            let mut group_infos: Vec<Vec<vk::DescriptorImageInfo>> = Vec::new();
            let mut sampler_infos = Vec::new();
            if let Some(storage) = &stage.storage_buffers {
                for bytes in &storage.buffers {
                    let allocation = self.create_buffer_with_bytes(bytes)?;
                    self.storage_buffers.push(allocation);
                }
                let first_buffer = self.storage_buffers.len() - storage.buffers.len();
                // Range = the guest data's exact size, NOT WHOLE_SIZE: pooled
                // buffers (upload ring) are usually larger than the data, and
                // WHOLE_SIZE would expose a previous draw's stale tail bytes
                // to the shader (and change OpArrayLength results).
                buffer_infos = self.storage_buffers[first_buffer..]
                    .iter()
                    .zip(&storage.buffers)
                    .map(|((buffer, _), bytes)| {
                        vk::DescriptorBufferInfo::default()
                            .buffer(*buffer)
                            .offset(0)
                            .range(bytes.len() as vk::DeviceSize)
                    })
                    .collect();
            }
            if let Some(textures) = &stage.textures {
                // Each T# resolves to a view in array order: a live persistent
                // render target binds its GPU image directly (stage B — no
                // CPU round trip); everything else uploads through staging.
                let mut views = Vec::with_capacity(textures.textures.len());
                for upload in &textures.textures {
                    if let Some(base) = upload.render_target {
                        let key = TargetKey {
                            base,
                            width: upload.width,
                            height: upload.height,
                            format: upload.format.as_raw(),
                        };
                        let (image, view, layout) =
                            self.caches.target_image(&key).ok_or_else(|| {
                                GpuError::PipelineCreationFailed(format!(
                                    "sampled render target {base:#x} ({}x{}) is no longer a \
                                     live persistent target",
                                    upload.width, upload.height
                                ))
                            })?;
                        if image == self.image {
                            return Err(GpuError::PipelineCreationFailed(format!(
                                "draw samples its own render target {base:#x} (feedback \
                                 loop) — the caller must use the CPU-pixels fallback"
                            )));
                        }
                        if !self
                            .sampled_targets
                            .iter()
                            .any(|(_, candidate, _)| *candidate == image)
                        {
                            self.sampled_targets.push((key, image, layout));
                        }
                        self.caches.stats.sampled_target_binds += 1;
                        views.push(view);
                    } else if upload.cached {
                        // Persistent-texture cache hit (stage D): the decode
                        // path verified the guest content's sample-hash
                        // matches the cached entry, so bind the cached view
                        // directly. No barrier: the image rests in
                        // SHADER_READ_ONLY_OPTIMAL with visibility to both
                        // graphics shader stages (the upload's tail barrier).
                        let key = TextureKey {
                            base: upload.guest_base,
                            width: upload.width,
                            height: upload.height,
                            layers: upload.layers,
                            depth: upload.depth.max(1),
                            cube: upload.cube,
                            array: upload.array,
                            volume: upload.volume,
                            format: upload.format.as_raw(),
                        };
                        let (view, hash) = self.caches.texture_entry(&key).ok_or_else(|| {
                            GpuError::PipelineCreationFailed(format!(
                                "cached texture {:#x} ({}x{}) predicted by the decode \
                                     snapshot is no longer in the texture cache",
                                upload.guest_base, upload.width, upload.height
                            ))
                        })?;
                        if hash != upload.sample_hash {
                            return Err(GpuError::PipelineCreationFailed(format!(
                                "cached texture {:#x} content hash changed between decode \
                                 and bind ({hash:#x} != {:#x})",
                                upload.guest_base, upload.sample_hash
                            )));
                        }
                        self.caches.stats.texture_cache_hits += 1;
                        views.push(view);
                    } else {
                        self.create_texture_image(upload, stage.stage)?;
                        views.push(
                            self.texture_uploads
                                .last()
                                .expect("create_texture_image pushed an entry")
                                .view,
                        );
                    }
                }
                let info_of = |view: vk::ImageView| {
                    vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(view)
                };
                if textures.sampled_groups.is_empty() {
                    image_infos = views.iter().map(|&v| info_of(v)).collect();
                } else {
                    // Mixed-dim: split the view pool into one descriptor array
                    // per Dim, in SPIR-V array order (the seeded T# index is
                    // the position within its group).
                    group_infos = textures
                        .sampled_groups
                        .iter()
                        .map(|group| {
                            group
                                .view_indices
                                .iter()
                                .map(|&i| info_of(views[i]))
                                .collect()
                        })
                        .collect();
                }
                for &sampler_state in &textures.samplers {
                    self.samplers
                        .push(self.caches.sampler(self.dev, sampler_state)?);
                }
                let first_sampler = self.samplers.len() - textures.samplers.len();
                sampler_infos = self.samplers[first_sampler..]
                    .iter()
                    .map(|&sampler| vk::DescriptorImageInfo::default().sampler(sampler))
                    .collect();
            }
            let mut writes = Vec::new();
            if let Some(storage) = &stage.storage_buffers {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(storage.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buffer_infos),
                );
            }
            if let Some(textures) = &stage.textures {
                if !textures.textures.is_empty() {
                    if textures.sampled_groups.is_empty() {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(textures.sampled_binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(&image_infos),
                        );
                    } else {
                        for (group, infos) in textures.sampled_groups.iter().zip(&group_infos) {
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(set)
                                    .dst_binding(group.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(infos),
                            );
                        }
                    }
                }
                if !textures.samplers.is_empty() {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(textures.sampler_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&sampler_infos),
                    );
                }
            }
            // SAFETY: `set` came from our pool/layout and every info names
            // live buffers, images, and samplers retained by this bundle.
            unsafe { self.device().update_descriptor_sets(&writes, &[]) };
            self.descriptor_sets.push((stage.descriptor_set_slot, set));
        }
        Ok(())
    }

    /// Upload one decoded guest texture: staging buffer plus device-local
    /// image and view. The staging-to-image copy is recorded in
    /// `record_and_submit`, before any rendering samples the image.
    fn create_texture_image(
        &mut self,
        upload: &TextureUpload,
        stage: vk::ShaderStageFlags,
    ) -> Result<(), GpuError> {
        if upload.pixels.is_empty() {
            return Err(GpuError::VulkanInitFailed(
                "texture upload requested with no pixels".to_owned(),
            ));
        }
        // Cube `arrayLayers` must be a valid multiple of 6 (see `cube_safe_layers`);
        // pad the staging pixels to match so the copy of `img_layers` faces never
        // overruns the buffer. Non-cube uploads borrow `pixels` unchanged.
        let img_layers = upload.cube_safe_layers();
        let staging = upload.staging_pixels(img_layers)?;
        let (staging_buffer, staging_memory) = self.create_buffer_with_bytes(&staging)?;
        // Persistent-texture cache (stage D): a cacheable upload donates its
        // image to the cache on draw success, so the next bind of the same
        // guest texture (same key, same content sample-hash) skips the whole
        // decode + create + upload. `RAEEN_NO_TEX_CACHE=1` disables donation.
        // A clamped-cube anomaly (`img_layers != upload.layers`) is never donated:
        // its key would not match a well-formed later decode of the same base.
        let cache_key = (img_layers == upload.layers
            && upload.guest_base != 0
            && upload.sample_hash != 0
            && !crate::diagnostics::gpu_env().no_tex_cache)
            .then_some(TextureKey {
                base: upload.guest_base,
                width: upload.width,
                height: upload.height,
                layers: img_layers,
                depth: upload.depth.max(1),
                cube: upload.cube,
                array: upload.array,
                volume: upload.volume,
                format: upload.format.as_raw(),
            });
        if cache_key.is_some() {
            self.caches.stats.texture_cache_misses += 1;
        }
        // Pushed with null image handles up front: `Drop` destroys whatever is
        // non-null, so every error path below cleans up the partial upload.
        self.texture_uploads.push(TextureGpu {
            staging_buffer,
            staging_memory,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: upload.width,
            height: upload.height,
            layers: img_layers,
            depth: upload.depth.max(1),
            stage,
            cache_key,
            sample_hash: upload.sample_hash,
            byte_size: staging.len() as u64,
        });
        let slot = self.texture_uploads.len() - 1;

        // A 3D volume (T# type 10): one layer, `depth` slices. Type-driven —
        // see `TextureUpload::volume` for why `depth > 1` is the wrong test.
        let volume = upload.volume;
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
            // A cube view requires the CUBE_CREATE flag and exactly 6 layers.
            .flags(if upload.cube {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            } else {
                vk::ImageCreateFlags::empty()
            });
        // SAFETY: `info` is fully initialized and borrows nothing beyond this
        // call; the device is live.
        let image = unsafe { self.device().create_image(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("texture vkCreateImage: {e}")))?;
        self.texture_uploads[slot].image = image;

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
            .map_err(|e| GpuError::VulkanInitFailed(format!("texture allocation: {e}")))?;
        self.texture_uploads[slot].memory = memory;

        // SAFETY: memory was allocated for exactly this image; offset 0 is
        // within it and satisfies the alignment requirement by construction.
        unsafe { self.device().bind_image_memory(image, memory, 0) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("texture bind memory: {e}")))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            // View type is decided from the T#-TYPE-driven upload flags, NOT
            // from the layer count, so it always matches the recompiled SPIR-V's
            // `OpTypeImage` Arrayed/Dim (both come from `from_texture_type`). A
            // 2DArray (type 13) with a single layer stays `TYPE_2D_ARRAY`
            // (`layer_count == 1`) — binding `TYPE_2D` there was the ASTRO.BOT
            // array/cube device-loss.
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
            .map_err(|e| GpuError::VulkanInitFailed(format!("texture view: {e}")))?;
        self.texture_uploads[slot].view = view;
        Ok(())
    }

    fn create_readback_buffer(
        &mut self,
        width: u32,
        height: u32,
        bpp: u32,
    ) -> Result<(), GpuError> {
        let (buffer, memory) = self.create_readback_buffer_raw(width, height, bpp)?;
        self.readback_buffer = buffer;
        self.readback_memory = memory;
        Ok(())
    }

    /// A host-readable readback buffer for a `width`x`height`x`bpp` colour
    /// target, without storing it on `self` — shared by the primary target
    /// and the per-draw MRT extras.
    fn create_readback_buffer_raw(
        &self,
        width: u32,
        height: u32,
        bpp: u32,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        let size =
            vk::DeviceSize::from(width) * vk::DeviceSize::from(height) * vk::DeviceSize::from(bpp);
        // The whole frame is copied out of this buffer on the CPU. Without
        // HOST_CACHED that copy reads uncached memory, which is ~50x slower:
        // measured 32 ms to read back one 1080p frame, dwarfing the ~1 ms of
        // actual GPU work. Prefer a cached+coherent type (fast reads, no manual
        // invalidate); fall back to coherent-only where the device has no such
        // type.
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            host | vk::MemoryPropertyFlags::HOST_CACHED,
        )
        .or_else(|_| self.create_buffer(size, vk::BufferUsageFlags::TRANSFER_DST, host))
    }

    fn create_pipeline(&mut self, state: &DrawState) -> Result<(), GpuError> {
        // Cached by SPIR-V content: the translate cache upstream already
        // dedups shaders, so repeated binds of one shader resolve to one
        // canonical VkShaderModule instead of a fresh module per draw.
        self.vertex_module = self.caches.shader_module(self.dev, state.vs_spirv)?;
        self.fragment_module = self.caches.shader_module(self.dev, state.fs_spirv)?;
        // Parsing both modules to compare stage interfaces allocates several
        // maps/sets.  Canonical module handles let the device cache prove that
        // an identical pair/topology was already accepted, removing that
        // repeated CPU work from every graphics-pipeline cache hit.
        self.caches.validate_graphics_interface_once(
            self.vertex_module,
            self.fragment_module,
            state.topology,
            || validate_graphics_interface(state),
        )?;

        // Same guard as the compute path: over-cap push constants are invalid
        // usage (UB without validation). Refuse the draw by name until the
        // SSBO spill path exists.
        for stage in &state.stage_bindings {
            let need = stage.push_constant_offset + stage.push_constants.len() as u32;
            let cap = self.dev.max_push_constants_size();
            if need > cap {
                return Err(GpuError::PipelineCreationFailed(format!(
                    "push constants {need} B exceed the device maxPushConstantsSize {cap} B \
                     (SSBO spill not implemented)"
                )));
            }
        }
        let push_ranges: Vec<_> = state
            .stage_bindings
            .iter()
            .filter(|stage| !stage.push_constants.is_empty())
            .map(|stage| {
                vk::PushConstantRange::default()
                    .stage_flags(stage.stage)
                    .offset(stage.push_constant_offset)
                    .size(stage.push_constants.len() as u32)
            })
            .collect();
        self.pipeline_layout =
            self.caches
                .pipeline_layout(self.dev, &self.descriptor_set_layouts, &push_ranges)?;

        // The effective vertex input layout: guest buffers when present; else
        // one vec4 attribute at location 0 for the fixture vertex buffer; else
        // nothing (a shader that builds its geometry from `gl_VertexIndex`
        // declares no inputs, and binding an attribute it never consumes is
        // invalid).
        let (vertex_bindings, vertex_attributes): (
            Vec<vk::VertexInputBindingDescription>,
            Vec<vk::VertexInputAttributeDescription>,
        ) = if !state.vertex_buffers.is_empty() {
            (
                state
                    .vertex_buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, data)| {
                        vk::VertexInputBindingDescription::default()
                            .binding(binding as u32)
                            .stride(data.stride)
                            .input_rate(if data.per_instance {
                                vk::VertexInputRate::INSTANCE
                            } else {
                                vk::VertexInputRate::VERTEX
                            })
                    })
                    .collect(),
                state
                    .vertex_attributes
                    .iter()
                    .map(|attr| {
                        vk::VertexInputAttributeDescription::default()
                            .location(attr.location)
                            .binding(attr.binding)
                            .format(attr.format)
                            .offset(attr.offset)
                    })
                    .collect(),
            )
        } else if state.vertices.is_some() {
            (
                vec![
                    vk::VertexInputBindingDescription::default()
                        .binding(0)
                        .stride(mem::size_of::<[f32; 4]>() as u32)
                        .input_rate(vk::VertexInputRate::VERTEX),
                ],
                vec![
                    vk::VertexInputAttributeDescription::default()
                        .location(0)
                        .binding(0)
                        .format(vk::Format::R32G32B32A32_SFLOAT)
                        .offset(0),
                ],
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // Everything that feeds pipeline creation is in the key; viewport,
        // scissor, and blend constants are dynamic state and deliberately
        // absent, so they cannot fragment the cache.
        let stencil_attachment = state
            .depth
            .as_ref()
            .is_some_and(|depth| has_stencil_plane(depth.format) && depth.stencil_test_enable);
        let key = GraphicsPipelineKey {
            vs: self.vertex_module.as_raw(),
            fs: self.fragment_module.as_raw(),
            layout: self.pipeline_layout.as_raw(),
            color_format: state.color_output.then(|| state.format.as_raw()),
            depth: state.depth.as_ref().map(|depth| DepthPipelineKey {
                format: depth.format.as_raw(),
                test: depth.test_enable,
                write: depth.write_enable,
                compare: depth.compare_op.as_raw(),
                stencil_test: depth.stencil_test_enable,
                front: StencilKey::from_vk(&depth.stencil_front),
                back: StencilKey::from_vk(&depth.stencil_back),
                stencil_attachment,
            }),
            topology: state.topology.as_raw(),
            cull: state.cull_mode.as_raw(),
            front_face: state.front_face.as_raw(),
            color_write_mask: state.color_write_mask.as_raw(),
            blend: BlendKey {
                enable: state.blend.enable,
                src_color: state.blend.src_color.as_raw(),
                dst_color: state.blend.dst_color.as_raw(),
                color_op: state.blend.color_op.as_raw(),
                src_alpha: state.blend.src_alpha.as_raw(),
                dst_alpha: state.blend.dst_alpha.as_raw(),
                alpha_op: state.blend.alpha_op.as_raw(),
            },
            vertex_bindings: vertex_bindings
                .iter()
                .map(|b| (b.binding, b.stride, b.input_rate.as_raw()))
                .collect(),
            vertex_attributes: vertex_attributes
                .iter()
                .map(|a| (a.location, a.binding, a.format.as_raw(), a.offset))
                .collect(),
            mrt: state
                .mrt
                .iter()
                .map(|extra| {
                    (
                        extra.format.as_raw(),
                        extra.write_mask.as_raw(),
                        BlendKey {
                            enable: extra.blend.enable,
                            src_color: extra.blend.src_color.as_raw(),
                            dst_color: extra.blend.dst_color.as_raw(),
                            color_op: extra.blend.color_op.as_raw(),
                            src_alpha: extra.blend.src_alpha.as_raw(),
                            dst_alpha: extra.blend.dst_alpha.as_raw(),
                            alpha_op: extra.blend.alpha_op.as_raw(),
                        },
                    )
                })
                .collect(),
        };
        if let Some(pipeline) = self.caches.lookup_graphics_pipeline(&key) {
            self.pipeline = pipeline;
            return Ok(());
        }

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vertex_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.fragment_module)
                .name(c"main"),
        ];

        // Empty slices produce zero-count vertex input state, matching the
        // old no-input default.
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_bindings)
            .vertex_attribute_descriptions(&vertex_attributes);

        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(state.topology);

        // Viewport and scissor are dynamic, set during recording.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(state.cull_mode)
            .front_face(state.front_face)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Depth/stencil state, only when a depth attachment is bound.
        let depth_stencil = state.depth.as_ref().map(|depth| {
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(depth.test_enable)
                .depth_write_enable(depth.write_enable)
                .depth_compare_op(depth.compare_op)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(depth.stencil_test_enable)
                .front(depth.stencil_front)
                .back(depth.stencil_back)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0)
        });

        // One blend-attachment state per colour attachment: the primary from
        // DrawState's own blend/write-mask fields, then one per MRT extra
        // (per-slot CB_BLEND{n}_CONTROL + CB_TARGET_MASK nibble). A depth-only
        // draw declares zero colour attachments; the blend attachment count
        // must match the pipeline's colour attachment count.
        let blend_attachment = |blend: &BlendState, write_mask: vk::ColorComponentFlags| {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(write_mask)
                .blend_enable(blend.enable)
                .src_color_blend_factor(blend.src_color)
                .dst_color_blend_factor(blend.dst_color)
                .color_blend_op(blend.color_op)
                .src_alpha_blend_factor(blend.src_alpha)
                .dst_alpha_blend_factor(blend.dst_alpha)
                .alpha_blend_op(blend.alpha_op)
        };
        let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = if state
            .color_output
        {
            if state.mrt.is_empty() || self.dev.supports_independent_blend() {
                std::iter::once(blend_attachment(&state.blend, state.color_write_mask))
                    .chain(
                        state
                            .mrt
                            .iter()
                            .map(|extra| blend_attachment(&extra.blend, extra.write_mask)),
                    )
                    .collect()
            } else {
                // Without `independentBlend` every element of pAttachments
                // must be IDENTICAL (VUID-VkPipelineColorBlendStateCreateInfo-
                // pAttachments-00605): degrade to the primary's state for
                // all targets — a named per-slot-blend loss, not a failed
                // draw.
                static NOTED: std::sync::Once = std::sync::Once::new();
                NOTED.call_once(|| {
                    tracing::warn!(
                        "device lacks independentBlend — MRT slots 1-7 use the \
                             primary attachment's blend/write-mask state"
                    );
                });
                vec![blend_attachment(&state.blend, state.color_write_mask); 1 + state.mrt.len()]
            }
        } else {
            Vec::new()
        };
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        // Blend constants join viewport/scissor as dynamic state so that a
        // register write to CB_BLEND_* cannot force a new pipeline.
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
        ];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Vulkan 1.3 dynamic rendering: the pipeline declares the attachment
        // formats directly instead of referencing a VkRenderPass.
        let color_formats = if state.color_output {
            std::iter::once(state.format)
                .chain(state.mrt.iter().map(|extra| extra.format))
                .collect()
        } else {
            Vec::new()
        };
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if let Some(depth) = &state.depth {
            rendering_info = rendering_info.depth_attachment_format(depth.format);
            if has_stencil_plane(depth.format) && depth.stencil_test_enable {
                rendering_info = rendering_info.stencil_attachment_format(depth.format);
            }
        }

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(self.pipeline_layout)
            .push_next(&mut rendering_info);
        if let Some(depth_stencil) = &depth_stencil {
            pipeline_info = pipeline_info.depth_stencil_state(depth_stencil);
        }

        // SAFETY: every struct chained into `pipeline_info` is a local alive
        // for this call; the shader modules and layout are live handles from
        // this device. A null pipeline cache is valid.
        let pipelines = unsafe {
            self.device().create_graphics_pipelines(
                self.dev.pipeline_cache(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, e)| {
            GpuError::PipelineCreationFailed(format!("vkCreateGraphicsPipelines: {e}"))
        })?;

        let pipeline = *pipelines.first().ok_or_else(|| {
            GpuError::PipelineCreationFailed("driver returned no pipeline".to_owned())
        })?;
        self.caches.store_graphics_pipeline(key, pipeline);
        self.dev.note_pipeline_compiled();
        self.pipeline = pipeline;
        Ok(())
    }

    /// Immediate mode: the persistent command buffer + fence from the cache
    /// (fence reset for this submission); the command buffer is implicitly
    /// reset by `vkBeginCommandBuffer` (the pool is RESET_COMMAND_BUFFER).
    ///
    /// Batched mode: a per-draw command buffer (recycled through the caches'
    /// free list) and NO fence — the draw is submitted without a wait and the
    /// flush fences the whole batch once.
    fn create_command_resources(&mut self) -> Result<(), GpuError> {
        if self.batched {
            self.command_buffer = self.caches.batch_command_buffer(self.dev)?;
            self.fence = vk::Fence::null();
            return Ok(());
        }
        let (command_buffer, fence) = self.caches.submit_resources(self.dev)?;
        self.command_buffer = command_buffer;
        self.fence = fence;
        Ok(())
    }

    /// Barrier helper: transition `image` (render target, depth target, or
    /// texture) between layouts, across `layers` array layers and the given
    /// aspect mask.
    fn image_barrier_layers(
        &self,
        aspect: vk::ImageAspectFlags,
        image: vk::Image,
        layers: u32,
        transition: ImageTransition,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(transition.old_layout)
            .new_layout(transition.new_layout)
            .src_access_mask(transition.src_access)
            .dst_access_mask(transition.dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: layers,
            });

        // SAFETY: called only between begin/end of `self.command_buffer`, which
        // is in the recording state. The barrier names this struct's own live
        // image and a subresource range within its creation parameters.
        unsafe {
            self.device().cmd_pipeline_barrier(
                self.command_buffer,
                transition.src_stage,
                transition.dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    fn record_and_submit(&mut self, state: &DrawState) -> Result<(), GpuError> {
        let (width, height) = (state.width, state.height);
        // Resolve the only fallible recording-time calculation before a
        // deferred draw appends anything to the shared batch command buffer.
        let stencil_plane_offset = state
            .depth
            .as_ref()
            .filter(|depth| has_stencil_plane(depth.format))
            .map(|depth| depth_plane_bytes(width, height, depth.format))
            .transpose()?;
        if !self.batched {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            // SAFETY: the command buffer is not pending — the previous submission
            // through it fence-completed under the cache lock — and the pool was
            // created RESET_COMMAND_BUFFER, so begin implicitly resets it. It is
            // recorded only under the cache lock (one recorder at a time).
            unsafe {
                self.device()
                    .begin_command_buffer(self.command_buffer, &begin_info)
            }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkBeginCommandBuffer: {e}")))?;
        }

        // Upload guest textures into their device-local images before any
        // rendering samples them: UNDEFINED -> TRANSFER_DST, staging copy,
        // then SHADER_READ_ONLY for the stage that samples.
        for texture in &self.texture_uploads {
            self.image_barrier_layers(
                vk::ImageAspectFlags::COLOR,
                texture.image,
                texture.layers,
                ImageTransition {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_access: vk::AccessFlags::empty(),
                    dst_access: vk::AccessFlags::TRANSFER_WRITE,
                    src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                    dst_stage: vk::PipelineStageFlags::TRANSFER,
                },
            );
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: texture.layers,
                })
                .image_extent(vk::Extent3D {
                    width: texture.width,
                    height: texture.height,
                    depth: texture.depth,
                });
            // SAFETY: the staging buffer holds exactly the upload's bytes
            // (create_texture_image sized it) and the image was created with
            // TRANSFER_DST usage; both belong to this device and the command
            // buffer is recording.
            unsafe {
                self.device().cmd_copy_buffer_to_image(
                    self.command_buffer,
                    texture.staging_buffer,
                    texture.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
            self.image_barrier_layers(
                vk::ImageAspectFlags::COLOR,
                texture.image,
                texture.layers,
                ImageTransition {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_access: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access: vk::AccessFlags::SHADER_READ,
                    src_stage: vk::PipelineStageFlags::TRANSFER,
                    // A cache-donated texture (stage D) may be sampled by
                    // EITHER graphics stage in later draws with no further
                    // barrier, so its writes must be made visible to both
                    // shader stages here, not just the binding stage.
                    dst_stage: if texture.cache_key.is_some() {
                        vk::PipelineStageFlags::VERTEX_SHADER
                            | vk::PipelineStageFlags::FRAGMENT_SHADER
                    } else {
                        shader_stage_to_pipeline(texture.stage)
                    },
                },
            );
        }

        // Persistent render targets this draw samples as textures: between
        // draws every persistent image sits in TRANSFER_SRC_OPTIMAL (each
        // draw's tail transition and the flush's copy both preserve that), so
        // transition to SHADER_READ_ONLY for the shader stages, and back
        // after rendering (below) to keep the invariant. The layout
        // transition itself publishes the prior draw's attachment writes
        // (already made available by that draw's tail barrier).
        for &(_, image, layout) in &self.sampled_targets {
            let (old_layout, src_access, src_stage) = match layout {
                TargetLayout::TransferSrc => (
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                ),
                TargetLayout::ColorAttachment => (
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ),
                TargetLayout::Undefined => {
                    return Err(GpuError::PipelineCreationFailed(
                        "sampled persistent target has an undefined layout".to_owned(),
                    ));
                }
            };
            self.image_barrier_layers(
                vk::ImageAspectFlags::COLOR,
                image,
                1,
                ImageTransition {
                    old_layout,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_access,
                    dst_access: vk::AccessFlags::SHADER_READ,
                    src_stage,
                    dst_stage: vk::PipelineStageFlags::VERTEX_SHADER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                },
            );
        }

        // Colour attachment: seed/transition only when this draw writes colour.
        // A depth-only z-prepass (`color_output == false`) has no colour image.
        if state.color_output {
            if state.initial.is_some() && !self.load_from_gpu {
                // Seed the attachment with the target's prior contents:
                // UNDEFINED -> TRANSFER_DST, copy in, then hand off to the
                // attachment stage so LOAD sees the composed frame so far.
                self.image_barrier_layers(
                    vk::ImageAspectFlags::COLOR,
                    self.image,
                    1,
                    ImageTransition {
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        src_access: vk::AccessFlags::empty(),
                        dst_access: vk::AccessFlags::TRANSFER_WRITE,
                        src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                        dst_stage: vk::PipelineStageFlags::TRANSFER,
                    },
                );
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                // SAFETY: the upload buffer holds exactly width*height*4 bytes
                // (validated in build) and the image was created TRANSFER_DST;
                // both belong to this device and the command buffer is recording.
                unsafe {
                    self.device().cmd_copy_buffer_to_image(
                        self.command_buffer,
                        self.upload_buffer,
                        self.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    );
                }
                self.image_barrier_layers(
                    vk::ImageAspectFlags::COLOR,
                    self.image,
                    1,
                    ImageTransition {
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        src_access: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags::COLOR_ATTACHMENT_READ,
                        src_stage: vk::PipelineStageFlags::TRANSFER,
                        dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    },
                );
            } else {
                // No seed upload. Two cases share this barrier:
                // - `load_from_gpu`: the persistent image still holds exactly
                //   the last readback, so transition TRANSFER_SRC_OPTIMAL (its
                //   layout after that readback copy) -> COLOR_ATTACHMENT,
                //   PRESERVING contents for the LOAD. The prior submission's
                //   only access was a transfer read, so no source writes need
                //   making available.
                // - otherwise: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL, which
                //   discards existing contents — fine, the pass clears anyway.
                let transition = if self.load_from_gpu {
                    colour_target_attachment_transition(self.target_layout)
                } else {
                    colour_target_attachment_transition(TargetLayout::Undefined)
                };
                self.image_barrier_layers(vk::ImageAspectFlags::COLOR, self.image, 1, transition);
            }

            // Extra MRT attachments: fresh per-draw images, so each either
            // seeds from its upload staging (prior framebuffer-map contents)
            // exactly like the primary, or starts UNDEFINED and CLEARs.
            for extra in &self.mrt_targets {
                if extra.upload_buffer != vk::Buffer::null() {
                    self.image_barrier_layers(
                        vk::ImageAspectFlags::COLOR,
                        extra.image,
                        1,
                        ImageTransition {
                            old_layout: vk::ImageLayout::UNDEFINED,
                            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            src_access: vk::AccessFlags::empty(),
                            dst_access: vk::AccessFlags::TRANSFER_WRITE,
                            src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                            dst_stage: vk::PipelineStageFlags::TRANSFER,
                        },
                    );
                    let region = vk::BufferImageCopy::default()
                        .buffer_offset(0)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        });
                    // SAFETY: the staging buffer holds exactly
                    // width*height*bpp bytes (validated in `create_mrt_targets`)
                    // and the image was created TRANSFER_DST; both belong to
                    // this device and the command buffer is recording.
                    unsafe {
                        self.device().cmd_copy_buffer_to_image(
                            self.command_buffer,
                            extra.upload_buffer,
                            extra.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[region],
                        );
                    }
                    self.image_barrier_layers(
                        vk::ImageAspectFlags::COLOR,
                        extra.image,
                        1,
                        ImageTransition {
                            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            src_access: vk::AccessFlags::TRANSFER_WRITE,
                            dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags::COLOR_ATTACHMENT_READ,
                            src_stage: vk::PipelineStageFlags::TRANSFER,
                            dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        },
                    );
                } else {
                    self.image_barrier_layers(
                        vk::ImageAspectFlags::COLOR,
                        extra.image,
                        1,
                        colour_target_attachment_transition(TargetLayout::Undefined),
                    );
                }
            }
        }

        // Depth/stencil attachment: transition it to the attachment layout,
        // seeding a plane that LOADs prior contents. A CLEAR plane starts
        // undefined and is cleared by the render pass, so it needs no seed.
        if let Some(depth) = &state.depth {
            let aspect = depth_aspect_mask(depth.format);
            let (depth_load, stencil_load) = self.effective_depth_loads(depth);
            if self.depth_upload_buffer != vk::Buffer::null() {
                let (old_layout, src_access, src_stage) = match self.depth_target_layout {
                    DepthTargetLayout::Undefined => (
                        vk::ImageLayout::UNDEFINED,
                        vk::AccessFlags::empty(),
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                    ),
                    DepthTargetLayout::TransferSrc => (
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    DepthTargetLayout::DepthStencilAttachment => (
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    ),
                    DepthTargetLayout::DepthStencilReadOnly => (
                        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                    ),
                };
                self.image_barrier_layers(
                    aspect,
                    self.depth_image,
                    1,
                    ImageTransition {
                        old_layout,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        src_access,
                        dst_access: vk::AccessFlags::TRANSFER_WRITE,
                        src_stage,
                        dst_stage: vk::PipelineStageFlags::TRANSFER,
                    },
                );
                let mut regions = Vec::new();
                if depth_load {
                    regions.push(depth_copy_region(
                        width,
                        height,
                        vk::ImageAspectFlags::DEPTH,
                        0,
                    ));
                }
                if stencil_load {
                    let offset = stencil_plane_offset.expect("stencil format has an offset");
                    regions.push(depth_copy_region(
                        width,
                        height,
                        vk::ImageAspectFlags::STENCIL,
                        offset,
                    ));
                }
                // SAFETY: the upload buffer holds the loaded planes' bytes
                // (`create_depth_buffers` sized and filled it), the depth image
                // was created TRANSFER_DST, and both belong to this device
                // while the command buffer records.
                if !regions.is_empty() {
                    unsafe {
                        self.device().cmd_copy_buffer_to_image(
                            self.command_buffer,
                            self.depth_upload_buffer,
                            self.depth_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &regions,
                        );
                    }
                }
                self.image_barrier_layers(
                    aspect,
                    self.depth_image,
                    1,
                    ImageTransition {
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        src_access: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        src_stage: vk::PipelineStageFlags::TRANSFER,
                        dst_stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    },
                );
            } else {
                let (old_layout, src_access, src_stage) = match self.depth_target_layout {
                    DepthTargetLayout::Undefined => (
                        vk::ImageLayout::UNDEFINED,
                        vk::AccessFlags::empty(),
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                    ),
                    DepthTargetLayout::TransferSrc => (
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    DepthTargetLayout::DepthStencilAttachment => (
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    ),
                    DepthTargetLayout::DepthStencilReadOnly => (
                        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                    ),
                };
                self.image_barrier_layers(
                    aspect,
                    self.depth_image,
                    1,
                    ImageTransition {
                        old_layout,
                        new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        src_access,
                        dst_access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        src_stage,
                        dst_stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    },
                );
            }
        }

        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: state.clear_color,
            },
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            // `RAEEN_FORCE_CLEAR=1` is an attachment/readback probe: it forces
            // every draw to CLEAR instead of LOAD. A uniform final clear proves
            // that the clear and readback path work and that no draw changed
            // the colour attachment. It does NOT isolate the vertex shader:
            // culling, depth, or stencil can reject otherwise-valid geometry.
            .load_op(if crate::diagnostics::gpu_env().force_clear {
                vk::AttachmentLoadOp::CLEAR
            } else if self.load_from_gpu || state.initial.is_some() {
                // `load_from_gpu` alone must force a LOAD: with deferred
                // readback the GPU copy can be the only authority
                // (TargetContent::GpuNewer) with no CPU seed to fall back on.
                vk::AttachmentLoadOp::LOAD
            } else {
                vk::AttachmentLoadOp::CLEAR
            })
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear);
        // A depth-only draw declares zero colour attachments, matching the
        // pipeline built with no colour formats (`create_pipeline`). MRT
        // extras follow the primary in `DrawState::mrt` order: LOAD when
        // seeded from prior contents, CLEAR (transparent black) otherwise.
        let color_attachments: Vec<vk::RenderingAttachmentInfo> = if state.color_output {
            std::iter::once(color_attachment)
                .chain(self.mrt_targets.iter().map(|extra| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(extra.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(if extra.upload_buffer != vk::Buffer::null() {
                            vk::AttachmentLoadOp::LOAD
                        } else {
                            vk::AttachmentLoadOp::CLEAR
                        })
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { float32: [0.0; 4] },
                        })
                }))
                .collect()
        } else {
            Vec::new()
        };

        // Vulkan 1.3 dynamic-rendering depth/stencil attachments. `depth_view`
        // carries both planes (aspect from `depth_aspect_mask`), so depth and
        // stencil reference the same view.
        let depth_attachment = state.depth.as_ref().map(|depth| {
            let (depth_load, _) = self.effective_depth_loads(depth);
            depth_stencil_attachment(self.depth_view, depth_load, depth)
        });
        let stencil_attachment = state
            .depth
            .as_ref()
            .filter(|depth| has_stencil_plane(depth.format) && depth.stencil_test_enable)
            .map(|depth| {
                let (_, stencil_load) = self.effective_depth_loads(depth);
                depth_stencil_attachment(self.depth_view, stencil_load, depth)
            });

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments);
        if let Some(depth_attachment) = &depth_attachment {
            rendering_info = rendering_info.depth_attachment(depth_attachment);
        }
        if let Some(stencil_attachment) = &stencil_attachment {
            rendering_info = rendering_info.stencil_attachment(stencil_attachment);
        }

        let viewports = [vk::Viewport {
            x: state.viewport[0],
            y: state.viewport[1],
            width: state.viewport[2],
            height: state.viewport[3],
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        // Clamp the register-supplied scissor into the attachment: Vulkan
        // rejects a scissor that leaves the render area, and a guest is free to
        // program one that does.
        let [sl, st, sr, sb] = state.scissor;
        let sl = sl.clamp(0, width as i32);
        let st = st.clamp(0, height as i32);
        let sr = sr.clamp(sl, width as i32);
        let sb = sb.clamp(st, height as i32);
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: sl, y: st },
            extent: vk::Extent2D {
                width: (sr - sl) as u32,
                height: (sb - st) as u32,
            },
        }];

        // SAFETY: all handles below belong to this device and are live; the
        // command buffer is recording; the vertex buffer (when bound) holds
        // exactly `state.vertex_count` vertices, matching the draw; the
        // pipeline's attachment format matches the image view's.
        // `cmd_begin_rendering` is core in Vulkan 1.3 and `dynamicRendering`
        // was required at device selection.
        unsafe {
            let d = self.device();
            d.cmd_begin_rendering(self.command_buffer, &rendering_info);
            d.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            d.cmd_set_viewport(self.command_buffer, 0, &viewports);
            d.cmd_set_scissor(self.command_buffer, 0, &scissors);
            // Dynamic since the pipeline cache landed: constants no longer
            // key (or fragment) the pipeline.
            d.cmd_set_blend_constants(self.command_buffer, &state.blend.constants);
            if !self.guest_vertex_buffers.is_empty() {
                let buffers: Vec<_> = self
                    .guest_vertex_buffers
                    .iter()
                    .map(|(buffer, _)| *buffer)
                    .collect();
                let offsets = vec![0; buffers.len()];
                d.cmd_bind_vertex_buffers(self.command_buffer, 0, &buffers, &offsets);
            } else if state.vertices.is_some() {
                d.cmd_bind_vertex_buffers(self.command_buffer, 0, &[self.vertex_buffer], &[0]);
            }
            for stage in &state.stage_bindings {
                if let Some((_, set)) = self
                    .descriptor_sets
                    .iter()
                    .find(|(slot, _)| *slot == stage.descriptor_set_slot)
                {
                    d.cmd_bind_descriptor_sets(
                        self.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        stage.descriptor_set_slot,
                        &[*set],
                        &[],
                    );
                }
                if !stage.push_constants.is_empty() {
                    d.cmd_push_constants(
                        self.command_buffer,
                        self.pipeline_layout,
                        stage.stage,
                        stage.push_constant_offset,
                        &stage.push_constants,
                    );
                }
            }
            if let Some(index) = &state.index {
                // vertex_count carries the index count for an indexed draw
                // (draw_state_from_regs stores the count it was given).
                d.cmd_bind_index_buffer(
                    self.command_buffer,
                    self.index_buffer,
                    0,
                    index.index_type,
                );
                d.cmd_draw_indexed(self.command_buffer, state.vertex_count, 1, 0, 0, 0);
            } else {
                d.cmd_draw(self.command_buffer, state.vertex_count, 1, 0, 0);
            }
            d.cmd_end_rendering(self.command_buffer);
        }

        // Return sampled persistent targets to the between-draws layout
        // invariant (TRANSFER_SRC_OPTIMAL). Reads need no availability
        // operation; the execution dependency alone orders the transition
        // after the shader reads.
        for &(_, image, _) in &self.sampled_targets {
            self.image_barrier_layers(
                vk::ImageAspectFlags::COLOR,
                image,
                1,
                ImageTransition {
                    old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_access: vk::AccessFlags::empty(),
                    dst_access: vk::AccessFlags::TRANSFER_READ,
                    src_stage: vk::PipelineStageFlags::VERTEX_SHADER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                    dst_stage: vk::PipelineStageFlags::TRANSFER,
                },
            );
        }

        // Immediate mode transitions for its readback now. Deferred persistent
        // targets remain attachment-resident; the next draw uses a same-layout
        // dependency and only the eventual filtered flush pays this transition.
        if state.color_output && !self.batched {
            self.image_barrier_layers(
                vk::ImageAspectFlags::COLOR,
                self.image,
                1,
                ImageTransition {
                    old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access: vk::AccessFlags::TRANSFER_READ,
                    src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage: vk::PipelineStageFlags::TRANSFER,
                },
            );
        }
        if state.color_output && !self.batched {
            // buffer_row_length/image_height = 0 means "tightly packed", which is
            // what `RenderedImage::pixel` assumes.
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });

            // SAFETY: the image is in TRANSFER_SRC_OPTIMAL per the barrier above,
            // and the readback buffer was sized `width * height * 4` — exactly the
            // region copied.
            unsafe {
                self.device().cmd_copy_image_to_buffer(
                    self.command_buffer,
                    self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.readback_buffer,
                    &[region],
                );
            }
            // Extra MRT attachments: same transition + copy-out, one per
            // attachment (MRT draws are immediate-only, so `!self.batched`
            // always holds when extras exist).
            for extra in &self.mrt_targets {
                self.image_barrier_layers(
                    vk::ImageAspectFlags::COLOR,
                    extra.image,
                    1,
                    ImageTransition {
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_access: vk::AccessFlags::TRANSFER_READ,
                        src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        dst_stage: vk::PipelineStageFlags::TRANSFER,
                    },
                );
                // SAFETY: the extra image is in TRANSFER_SRC per the barrier
                // above and its readback buffer was sized width*height*bpp —
                // exactly the region copied.
                unsafe {
                    self.device().cmd_copy_image_to_buffer(
                        self.command_buffer,
                        extra.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        extra.readback_buffer,
                        &[region],
                    );
                }
            }
        }

        // Depth/stencil copy out: DEPTH_STENCIL_ATTACHMENT_OPTIMAL ->
        // TRANSFER_SRC, then copy the depth plane (and stencil plane, if any)
        // into the readback buffer at the layout `read_back_depth` expects.
        if let Some(depth) = &state.depth
            && !self.batched
        {
            let aspect = depth_aspect_mask(depth.format);
            self.image_barrier_layers(
                aspect,
                self.depth_image,
                1,
                ImageTransition {
                    old_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dst_access: vk::AccessFlags::TRANSFER_READ,
                    src_stage: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    dst_stage: vk::PipelineStageFlags::TRANSFER,
                },
            );
            if !self.batched {
                let mut regions = vec![depth_copy_region(
                    width,
                    height,
                    vk::ImageAspectFlags::DEPTH,
                    0,
                )];
                if has_stencil_plane(depth.format) {
                    let offset = stencil_plane_offset.expect("stencil format has an offset");
                    regions.push(depth_copy_region(
                        width,
                        height,
                        vk::ImageAspectFlags::STENCIL,
                        offset,
                    ));
                }
                // SAFETY: the depth image is in TRANSFER_SRC per the barrier
                // above, and `depth_readback_buffer` was sized for exactly
                // these planes.
                unsafe {
                    self.device().cmd_copy_image_to_buffer(
                        self.command_buffer,
                        self.depth_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.depth_readback_buffer,
                        &regions,
                    );
                }
            }
        }

        if !self.batched {
            // Make the transfer writes visible to host reads of the mapped
            // memory. Batched draws record no host-read copies, so they skip
            // this (the flush's own command buffer carries one).
            let host_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            // SAFETY: still recording; a global memory barrier names no handles.
            unsafe {
                self.device().cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[host_barrier],
                    &[],
                    &[],
                );
            }
        }
        if self.batched {
            // Deferred: the executable command buffer is retained but not
            // submitted yet. The flush submits every pending draw/dispatch in
            // PM4 order with one queue call and one fence.
            return Ok(());
        }

        // SAFETY: the immediate command buffer is in the recording state.
        unsafe {
            self.device()
                .end_command_buffer(self.command_buffer)
                .map_err(|e| GpuError::VulkanInitFailed(format!("vkEndCommandBuffer: {e}")))?;
        }
        let command_buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);

        // SAFETY: the command buffer is recorded and not already pending; the
        // fence is unsignaled and belongs to this device.
        unsafe {
            self.device()
                .queue_submit(self.dev.queue(), &[submit], self.fence)
        }
        .map_err(|e| {
            self.dev.note_vk_error(e);
            GpuError::VulkanInitFailed(format!("vkQueueSubmit failed: {e}"))
        })?;

        // Wait for the GPU. u64::MAX = no timeout; a hang here means a driver
        // fault, which surfaces as a hung test rather than silent bad pixels.
        // SAFETY: the fence was just submitted with and belongs to this device.
        unsafe { self.device().wait_for_fences(&[self.fence], true, u64::MAX) }.map_err(|e| {
            self.dev.note_vk_error(e);
            GpuError::VulkanInitFailed(format!("vkWaitForFences failed: {e}"))
        })?;
        Ok(())
    }

    /// Transfer every per-draw handle into the caches' pending batch after a
    /// successful deferred submission. Colour draws record their target as
    /// [`TargetContent::GpuNewer`]; persistent depth-only passes retain only
    /// ordered resources because their cache-owned image is already the
    /// authority. After this, `Drop` finds only null handles and destroys
    /// nothing.
    fn commit_to_batch(&mut self) -> Result<(), GpuError> {
        debug_assert!(self.batched, "only batched draws defer their resources");
        if self.target_key.is_none() && self.depth_target_key.is_none() {
            return Err(GpuError::VulkanInitFailed(
                "deferred draw has neither a persistent colour nor depth target".to_owned(),
            ));
        }
        // The cache owns the one shared recording handle. This draw contributes
        // only resources; `finish_batch_recording` attaches that handle to the
        // first pending entry at the flip boundary.
        self.command_buffer = vk::CommandBuffer::null();
        let mut res = PendingDrawResources {
            command_buffer: vk::CommandBuffer::null(),
            // The draw's sets live in a shared batch pool (stage D item 2),
            // reset by the cache at the batch retire — nothing to transfer.
            descriptor_pool: vk::DescriptorPool::null(),
            buffers: Vec::new(),
            images: Vec::new(),
        };
        let mut take_buffer = |buffer: &mut vk::Buffer, memory: &mut vk::DeviceMemory| {
            let buffer = mem::replace(buffer, vk::Buffer::null());
            let memory = mem::replace(memory, vk::DeviceMemory::null());
            if buffer != vk::Buffer::null() || memory != vk::DeviceMemory::null() {
                res.buffers.push((buffer, memory));
            }
        };
        take_buffer(&mut self.upload_buffer, &mut self.upload_memory);
        take_buffer(&mut self.vertex_buffer, &mut self.vertex_memory);
        take_buffer(&mut self.index_buffer, &mut self.index_memory);
        take_buffer(&mut self.depth_upload_buffer, &mut self.depth_upload_memory);
        if self.owns_depth_target {
            take_buffer(
                &mut self.depth_readback_buffer,
                &mut self.depth_readback_memory,
            );
        }
        for (buffer, memory) in self.guest_vertex_buffers.drain(..) {
            res.buffers.push((buffer, memory));
        }
        for (buffer, memory) in self.storage_buffers.drain(..) {
            res.buffers.push((buffer, memory));
        }
        // Cache-eligible textures are donated to the persistent-texture cache
        // instead of being destroyed with the batch; their staging buffers
        // still retire with the batch. The donation itself happens AFTER
        // `commit_deferred_draw` (below) so `batch_open()` is true and any
        // eviction it triggers defers destruction past the flush fence —
        // this draw's just-submitted command buffer may reference the
        // evicted image.
        let mut donations: Vec<(TextureKey, PersistentTexture)> = Vec::new();
        for mut texture in self.texture_uploads.drain(..) {
            res.buffers
                .push((texture.staging_buffer, texture.staging_memory));
            if let Some(cache_key) = texture.cache_key.take() {
                donations.push((
                    cache_key,
                    PersistentTexture {
                        image: texture.image,
                        memory: texture.memory,
                        view: texture.view,
                        sample_hash: texture.sample_hash,
                        bytes: texture.byte_size,
                        last_use: 0,
                    },
                ));
            } else {
                res.images
                    .push((texture.image, texture.memory, texture.view));
            }
        }
        if self.owns_depth_target
            && (self.depth_image != vk::Image::null() || self.depth_view != vk::ImageView::null())
        {
            res.images.push((
                mem::replace(&mut self.depth_image, vk::Image::null()),
                mem::replace(&mut self.depth_memory, vk::DeviceMemory::null()),
                mem::replace(&mut self.depth_view, vk::ImageView::null()),
            ));
        }
        self.commit_auxiliary_layouts(true);
        if let Some(key) = self.target_key {
            self.caches.commit_deferred_draw(
                res,
                key,
                self.depth_target_key,
                TargetLayout::ColorAttachment,
            );
        } else {
            // Depth-only work still belongs to the ordered batch and its
            // resources must live through the shared fence. No colour target
            // needs joining the touched/readback list.
            self.caches.commit_deferred_resources(res, [], []);
        }
        // Batch is now open: evictions inside insert_texture defer safely.
        for (cache_key, entry) in donations {
            self.caches.insert_texture(self.dev, cache_key, entry);
        }
        Ok(())
    }

    /// Publish image layouts only after recording succeeded. A failed draw
    /// leaves the cache's prior layout intact because its shared command
    /// buffer is never committed as an executable batch entry.
    fn commit_auxiliary_layouts(&mut self, batched: bool) {
        for (key, _, _) in &self.sampled_targets {
            self.caches
                .mark_target_layout(key, TargetLayout::TransferSrc);
        }
        if let Some(key) = self.depth_target_key {
            self.caches.mark_depth_target_layout(
                &key,
                if batched {
                    DepthTargetLayout::DepthStencilAttachment
                } else {
                    DepthTargetLayout::TransferSrc
                },
            );
        }
    }

    /// Immediate-path counterpart of the donation in [`Self::commit_to_batch`]:
    /// called after `record_and_submit` fence-waited the upload, so the cached
    /// image is complete and any eviction can only touch images referenced by
    /// fence-completed work (or defers, if a deferred batch happens to be
    /// open). Handles leave this struct so `Drop` no longer destroys them.
    fn donate_textures_to_cache(&mut self) {
        for texture in &mut self.texture_uploads {
            let Some(cache_key) = texture.cache_key.take() else {
                continue;
            };
            let entry = PersistentTexture {
                image: mem::replace(&mut texture.image, vk::Image::null()),
                memory: mem::replace(&mut texture.memory, vk::DeviceMemory::null()),
                view: mem::replace(&mut texture.view, vk::ImageView::null()),
                sample_hash: texture.sample_hash,
                bytes: texture.byte_size,
                last_use: 0,
            };
            self.caches.insert_texture(self.dev, cache_key, entry);
        }
    }

    fn read_back(&self, width: u32, height: u32, bpp: u32) -> Result<Vec<u8>, GpuError> {
        let size = (width as usize) * (height as usize) * (bpp as usize);

        // SAFETY: the memory is HOST_VISIBLE and not currently mapped. The GPU
        // work that writes it has completed — `record_and_submit` waited on the
        // fence — and a barrier made those writes available to the host.
        let ptr = unsafe {
            self.device().map_memory(
                self.readback_memory,
                0,
                size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("readback map failed: {e}")))?;

        // SAFETY: `ptr` is a valid mapping of `size` initialized bytes (the
        // buffer was allocated at exactly this size, its copy completed); the
        // helper copies them into an owned Vec fallibly (degrade, not abort) and
        // unmaps.
        let pixels = unsafe {
            readback_to_vec_fallible(self.device(), self.readback_memory, ptr, size, "readback")?
        };
        Ok(pixels)
    }

    /// The colour readback as a [`RenderedImage`], or `None` for a depth-only
    /// draw. `bpp` is `Some` exactly when `state.color_output` — the caller
    /// computed it once from the format so a `None` here means "no colour
    /// attachment", not an error.
    fn read_back_color(
        &self,
        state: &DrawState,
        bpp: Option<u32>,
    ) -> Result<Option<RenderedImage>, GpuError> {
        let Some(bpp) = bpp else {
            return Ok(None);
        };
        let pixels = self.read_back(state.width, state.height, bpp)?;
        Ok(Some(RenderedImage {
            width: state.width,
            height: state.height,
            pixels,
            bytes_per_pixel: bpp,
        }))
    }

    /// Read back every extra MRT attachment, `(guest base, image)` in
    /// `DrawState::mrt` order. Empty when the draw had no extras.
    fn read_back_mrt(&self, state: &DrawState) -> Result<Vec<(u64, RenderedImage)>, GpuError> {
        let mut out = Vec::with_capacity(self.mrt_targets.len());
        for (res, extra) in self.mrt_targets.iter().zip(&state.mrt) {
            let size = (state.width as usize) * (state.height as usize) * (res.bpp as usize);
            // SAFETY: the memory is HOST_VISIBLE and not currently mapped; the
            // copy that fills it completed (the fence was waited) and the host
            // barrier made those writes visible.
            let ptr = unsafe {
                self.device().map_memory(
                    res.readback_memory,
                    0,
                    size as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|e| GpuError::VulkanInitFailed(format!("MRT readback map failed: {e}")))?;
            // SAFETY: `ptr` maps exactly `size` initialized bytes; the helper
            // copies them into an owned Vec fallibly and unmaps in every case.
            let pixels = unsafe {
                readback_to_vec_fallible(self.device(), res.readback_memory, ptr, size, "MRT")?
            };
            out.push((
                extra.target_base,
                RenderedImage {
                    width: state.width,
                    height: state.height,
                    pixels,
                    bytes_per_pixel: res.bpp,
                },
            ));
        }
        Ok(out)
    }

    /// The depth/stencil readback as a [`DepthImage`], or `None` when the draw
    /// bound no depth attachment. The depth plane occupies the first
    /// `depth_plane_bytes` of `depth_readback_buffer`; the stencil plane (one
    /// byte per texel, present only for a stencil-bearing format) follows — the
    /// exact layout `create_depth_buffers` sized and the copy-out in
    /// `record_and_submit` wrote.
    fn read_back_depth(&self, state: &DrawState) -> Result<Option<DepthImage>, GpuError> {
        let Some(depth) = &state.depth else {
            return Ok(None);
        };
        let (width, height) = (state.width, state.height);
        let depth_bytes = depth_plane_bytes(width, height, depth.format)? as usize;
        let stencil_bytes = if has_stencil_plane(depth.format) {
            (width as usize) * (height as usize)
        } else {
            0
        };
        let total = depth_bytes + stencil_bytes;

        // SAFETY: `depth_readback_memory` is HOST_VISIBLE|HOST_COHERENT (or the
        // HOST_CACHED fallback) memory sized `total` by `create_depth_buffers`
        // and not currently mapped. The copy that filled it completed
        // (`record_and_submit` waited on the fence) and a host barrier made
        // those writes visible.
        let ptr = unsafe {
            self.device().map_memory(
                self.depth_readback_memory,
                0,
                total as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("depth readback map failed: {e}")))?;

        // SAFETY: `ptr` maps exactly `total` bytes, initialized by the completed
        // copy; the bytes are copied into owned Vecs before unmapping, so no
        // reference outlives the mapping.
        let all = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), total) };
        let depth_plane = all[..depth_bytes].to_vec();
        let stencil = (stencil_bytes > 0).then(|| all[depth_bytes..].to_vec());

        // SAFETY: mapped by the call above; no references into it remain.
        unsafe { self.device().unmap_memory(self.depth_readback_memory) };

        Ok(Some(DepthImage {
            width,
            height,
            format: depth.format,
            depth: depth_plane,
            stencil,
        }))
    }
}

/// Refuse translated stage pairs that are invalid Vulkan before they can be
/// submitted. A bad guest shader must cost one draw, never the whole logical
/// device. The location scan intentionally handles the direct interface
/// variables emitted by both Raeen and Kyty's current Gen5 recompiler.
fn validate_graphics_interface(state: &DrawState) -> Result<(), GpuError> {
    fn writes_point_size(words: &[u32]) -> bool {
        let mut point_members = std::collections::HashSet::new();
        let mut pointer_pointees = std::collections::HashMap::new();
        let mut output_variables = std::collections::HashMap::new();
        let mut constants = std::collections::HashMap::new();
        let mut point_pointers = std::collections::HashSet::new();
        let mut stores = std::collections::HashSet::new();
        let mut at = 5;
        while at < words.len() {
            let header = words[at];
            let len = (header >> 16) as usize;
            let op = (header & 0xffff) as u16;
            if len == 0 || at.saturating_add(len) > words.len() {
                break;
            }
            let inst = &words[at..at + len];
            match op {
                // OpMemberDecorate %struct member BuiltIn PointSize.
                72 if len >= 5 && inst[3] == 11 && inst[4] == 1 => {
                    point_members.insert((inst[1], inst[2]));
                }
                // OpTypePointer %result Output %pointee.
                32 if len >= 4 && inst[2] == 3 => {
                    pointer_pointees.insert(inst[1], inst[3]);
                }
                // OpVariable %ptr_type %result Output.
                59 if len >= 4 && inst[3] == 3 => {
                    output_variables.insert(inst[2], inst[1]);
                }
                // OpConstant %type %result literal.
                43 if len >= 4 => {
                    constants.insert(inst[2], inst[3]);
                }
                // OpAccessChain / OpInBoundsAccessChain. The current Gen5
                // translator accesses PointSize directly from outPerVertex.
                65 | 66 if len >= 5 => {
                    let base = inst[3];
                    let Some(&pointer_type) = output_variables.get(&base) else {
                        at += len;
                        continue;
                    };
                    let Some(&pointee) = pointer_pointees.get(&pointer_type) else {
                        at += len;
                        continue;
                    };
                    let Some(&member) = constants.get(&inst[4]) else {
                        at += len;
                        continue;
                    };
                    if point_members.contains(&(pointee, member)) {
                        point_pointers.insert(inst[2]);
                    }
                }
                // OpStore %pointer %object.
                62 if len >= 3 => {
                    stores.insert(inst[1]);
                }
                _ => {}
            }
            at += len;
        }
        point_pointers
            .iter()
            .any(|pointer| stores.contains(pointer))
    }

    if state.topology == vk::PrimitiveTopology::POINT_LIST && !writes_point_size(state.vs_spirv) {
        return Err(GpuError::PipelineCreationFailed(
            "point-list draw skipped: translated vertex shader does not write gl_PointSize"
                .to_owned(),
        ));
    }

    fn locations(words: &[u32], storage_class: u32) -> std::collections::BTreeSet<u32> {
        let mut storage = std::collections::HashMap::new();
        let mut decorated = Vec::new();
        let mut at = 5;
        while at < words.len() {
            let header = words[at];
            let len = (header >> 16) as usize;
            let op = (header & 0xffff) as u16;
            if len == 0 || at.saturating_add(len) > words.len() {
                break;
            }
            let inst = &words[at..at + len];
            match op {
                // OpVariable: result type, result id, storage class.
                59 if len >= 4 => {
                    storage.insert(inst[2], inst[3]);
                }
                // OpDecorate %id Location N (Decoration::Location == 30).
                71 if len >= 4 && inst[2] == 30 => decorated.push((inst[1], inst[3])),
                _ => {}
            }
            at += len;
        }
        decorated
            .into_iter()
            .filter_map(|(id, location)| {
                (storage.get(&id) == Some(&storage_class)).then_some(location)
            })
            .collect()
    }

    // SPIR-V StorageClass: Input=1, Output=3.
    let vertex_outputs = locations(state.vs_spirv, 3);
    let fragment_inputs = locations(state.fs_spirv, 1);
    let missing = fragment_inputs
        .difference(&vertex_outputs)
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        // Vulkan does NOT permit the GCN behavior of feeding undefined values
        // to unmatched fragment inputs. Letting this pair link produced
        // VUID-RuntimeSpirv-OpEntryPoint-08743 and an AMD device loss on
        // ASTRO.BOT. Until the GS-copy/primitive stage is translated and can
        // supply these params, skip this draw by name rather than submitting
        // invalid SPIR-V to the host.
        return Err(GpuError::PipelineCreationFailed(format!(
            "fragment input locations {missing:?} have no vertex-stage outputs \
             (GS-copy/primitive-stage params are not translated)"
        )));
    }
    Ok(())
}

impl Drop for Resources<'_> {
    fn drop(&mut self) {
        let guest_vertex_buffers = mem::take(&mut self.guest_vertex_buffers);
        let storage_buffers = mem::take(&mut self.storage_buffers);
        let texture_uploads = mem::take(&mut self.texture_uploads);
        // NOT destroyed here — owned by the device's DrawCaches and reused by
        // the next draw: pipeline, pipeline layout, descriptor set layouts,
        // shader modules, samplers, command buffer + fence (immediate mode),
        // the shared descriptor pool, and (when `owns_target` is false) the
        // colour image + its readback buffer. `DrawCaches::destroy` releases
        // them with the device. A successful batched draw reaches this Drop
        // with every per-draw handle already moved into the pending batch
        // (`commit_to_batch`), so nothing is destroyed early.
        //
        // A batched draw borrows the cache-owned shared recording handle. Clear
        // this local alias on an error path; the cache closes/recycles the
        // handle at the batch boundary.
        if self.batched && self.command_buffer != vk::CommandBuffer::null() {
            self.command_buffer = vk::CommandBuffer::null();
        }
        // Extra MRT attachments are always per-draw owned: destroy their
        // images/views/readbacks here and return their seed staging buffers
        // to the upload ring. The same fence/error argument as the primary
        // target applies (immediate-only, fence waited before any success).
        {
            let mrt_targets = mem::take(&mut self.mrt_targets);
            for extra in mrt_targets {
                self.caches
                    .release_host_buffer(self.dev, extra.upload_buffer, extra.upload_memory);
                // SAFETY: per-draw handles created from this device for this
                // draw alone, destroyed exactly once, children before parents;
                // null handles are skipped (partial build cleanup).
                unsafe {
                    let d = self.dev.device();
                    if extra.readback_buffer != vk::Buffer::null() {
                        d.destroy_buffer(extra.readback_buffer, None);
                    }
                    if extra.readback_memory != vk::DeviceMemory::null() {
                        d.free_memory(extra.readback_memory, None);
                    }
                    if extra.view != vk::ImageView::null() {
                        d.destroy_image_view(extra.view, None);
                    }
                    if extra.image != vk::Image::null() {
                        d.destroy_image(extra.image, None);
                    }
                    if extra.memory != vk::DeviceMemory::null() {
                        d.free_memory(extra.memory, None);
                    }
                }
            }
        }
        // Guest-data buffers came from the upload ring: return them (no
        // submitted GPU work references them — see the safety argument below;
        // ad-hoc/unpooled pairs are destroyed inside release_host_buffer).
        {
            let upload = (
                mem::replace(&mut self.upload_buffer, vk::Buffer::null()),
                mem::replace(&mut self.upload_memory, vk::DeviceMemory::null()),
            );
            let vertex = (
                mem::replace(&mut self.vertex_buffer, vk::Buffer::null()),
                mem::replace(&mut self.vertex_memory, vk::DeviceMemory::null()),
            );
            let index = (
                mem::replace(&mut self.index_buffer, vk::Buffer::null()),
                mem::replace(&mut self.index_memory, vk::DeviceMemory::null()),
            );
            let depth_upload = (
                mem::replace(&mut self.depth_upload_buffer, vk::Buffer::null()),
                mem::replace(&mut self.depth_upload_memory, vk::DeviceMemory::null()),
            );
            for (buffer, memory) in [upload, vertex, index, depth_upload]
                .into_iter()
                .chain(guest_vertex_buffers)
                .chain(storage_buffers)
            {
                self.caches.release_host_buffer(self.dev, buffer, memory);
            }
        }
        // SAFETY: every handle destroyed below was created from `self.dev`'s
        // device for this draw alone and is destroyed exactly once, children
        // before parents. No submitted GPU work references them: an immediate
        // draw's fence was waited in `record_and_submit` before any success
        // path, an error before submission leaves nothing pending, and a
        // failed fence wait means a lost device (destruction is the only
        // option either way). The old per-draw `device_wait_idle` here was
        // redundant with the fence wait and is gone (stage B). Null handles
        // are skipped, so a partially-built `Resources` (an error during
        // `build`) cleans up correctly.
        unsafe {
            let d = self.dev.device();

            // `descriptor_pool` is always cache-owned (shared resettable pool
            // or a shared batch pool) — never destroyed here.
            for texture in texture_uploads {
                if texture.view != vk::ImageView::null() {
                    d.destroy_image_view(texture.view, None);
                }
                if texture.image != vk::Image::null() {
                    d.destroy_image(texture.image, None);
                }
                if texture.memory != vk::DeviceMemory::null() {
                    d.free_memory(texture.memory, None);
                }
                self.caches.release_host_buffer(
                    self.dev,
                    texture.staging_buffer,
                    texture.staging_memory,
                );
            }
            if self.owns_target {
                if self.readback_buffer != vk::Buffer::null() {
                    d.destroy_buffer(self.readback_buffer, None);
                }
                if self.readback_memory != vk::DeviceMemory::null() {
                    d.free_memory(self.readback_memory, None);
                }
                if self.image_view != vk::ImageView::null() {
                    d.destroy_image_view(self.image_view, None);
                }
                if self.image != vk::Image::null() {
                    d.destroy_image(self.image, None);
                }
                if self.image_memory != vk::DeviceMemory::null() {
                    d.free_memory(self.image_memory, None);
                }
            }
            if self.owns_depth_target {
                if self.depth_view != vk::ImageView::null() {
                    d.destroy_image_view(self.depth_view, None);
                }
                if self.depth_image != vk::Image::null() {
                    d.destroy_image(self.depth_image, None);
                }
                if self.depth_memory != vk::DeviceMemory::null() {
                    d.free_memory(self.depth_memory, None);
                }
            }
            if self.depth_upload_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.depth_upload_buffer, None);
            }
            if self.depth_upload_memory != vk::DeviceMemory::null() {
                d.free_memory(self.depth_upload_memory, None);
            }
            if self.owns_depth_target {
                if self.depth_readback_buffer != vk::Buffer::null() {
                    d.destroy_buffer(self.depth_readback_buffer, None);
                }
                if self.depth_readback_memory != vk::DeviceMemory::null() {
                    d.free_memory(self.depth_readback_memory, None);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::shaders::TRIANGLE_COLOR;
    use super::*;

    #[test]
    fn persistent_depth_only_output_is_batch_eligible() {
        assert!(has_only_named_persistent_outputs(false, false, true, true));
        assert!(has_only_named_persistent_outputs(true, true, false, false));
        assert!(has_only_named_persistent_outputs(true, true, true, true));

        assert!(!has_only_named_persistent_outputs(
            false, false, true, false
        ));
        assert!(!has_only_named_persistent_outputs(true, false, true, true));
        assert!(!has_only_named_persistent_outputs(
            false, false, false, false
        ));
    }

    fn location_module(storage_class: u32, location: u32) -> Vec<u32> {
        vec![
            0x0723_0203,
            0x0001_0000,
            0,
            100,
            0,
            (4 << 16) | 71, // OpDecorate %1 Location N
            1,
            30,
            location,
            (4 << 16) | 59, // OpVariable %type %1 StorageClass
            99,
            1,
            storage_class,
        ]
    }

    fn sampled_upload(pixels: Vec<u8>, layers: u32, cube: bool) -> TextureUpload {
        TextureUpload {
            width: 2,
            height: 2,
            format: vk::Format::R8G8B8A8_UNORM,
            pixels,
            layers,
            cube,
            array: false,
            volume: false,
            depth: 1,
            render_target: None,
            guest_base: 0x3368_0000,
            sample_hash: 1,
            cached: false,
        }
    }

    /// Vulkan copies all declared faces even if an upstream fallback supplied
    /// only one. The staging guard must make that copy memory-safe.
    #[test]
    fn cube_staging_pixels_pad_one_face_to_all_six_faces() {
        let face = vec![0x5a; 2 * 2 * 4];
        let upload = sampled_upload(face.clone(), 6, true);
        let staging = upload.staging_pixels(upload.cube_safe_layers()).unwrap();
        assert_eq!(staging.len(), face.len() * 6);
        assert_eq!(&staging[..face.len()], face);
        assert!(staging[face.len()..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn complete_texture_staging_borrows_without_reallocation() {
        let pixels = vec![0x31; 2 * 2 * 4 * 6];
        let upload = sampled_upload(pixels, 6, true);
        assert!(matches!(
            upload.staging_pixels(6).unwrap(),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn graphics_interface_gate_rejects_missing_vertex_output() {
        // Vulkan requires every fragment input Location to be supplied by the
        // previous host stage. The guest may source it from a GS-copy stage,
        // but linking the incomplete VS/FS pair is invalid and can device-loss.
        let vs = location_module(3, 0);
        let fs = location_module(1, 1);
        let state = DrawState::new(16, 16, &vs, &fs);
        let error = validate_graphics_interface(&state).unwrap_err().to_string();
        assert!(error.contains("locations [1]"), "{error}");
        assert!(error.contains("GS-copy"), "{error}");
    }

    #[test]
    fn graphics_interface_gate_accepts_matching_locations() {
        let vs = location_module(3, 1);
        let fs = location_module(1, 1);
        let state = DrawState::new(16, 16, &vs, &fs);
        validate_graphics_interface(&state).expect("matching stage interface is valid");
    }

    #[test]
    fn graphics_interface_gate_rejects_unsafe_point_list() {
        let vs = location_module(3, 0);
        let fs = location_module(1, 0);
        let mut state = DrawState::new(16, 16, &vs, &fs);
        state.topology = vk::PrimitiveTopology::POINT_LIST;
        let error = validate_graphics_interface(&state).unwrap_err().to_string();
        assert!(error.contains("gl_PointSize"), "{error}");
    }

    #[test]
    fn graphics_interface_gate_accepts_point_list_with_point_size_store() {
        let mut vs = location_module(3, 0);
        vs.extend_from_slice(&[
            (5 << 16) | 72, // OpMemberDecorate %struct 1 BuiltIn PointSize
            10,
            1,
            11,
            1,
            (4 << 16) | 32, // OpTypePointer %ptr Output %struct
            20,
            3,
            10,
            (4 << 16) | 59, // OpVariable %ptr %out Output
            20,
            30,
            3,
            (4 << 16) | 43, // OpConstant %uint %one 1
            90,
            40,
            1,
            (5 << 16) | 65, // OpAccessChain %ptr_float %point %out %one
            91,
            50,
            30,
            40,
            (3 << 16) | 62, // OpStore %point %value
            50,
            60,
        ]);
        let fs = location_module(1, 0);
        let mut state = DrawState::new(16, 16, &vs, &fs);
        state.topology = vk::PrimitiveTopology::POINT_LIST;
        validate_graphics_interface(&state).expect("PointSize store makes point-list valid");
    }

    #[test]
    fn unorm8_maps_endpoints_exactly() {
        assert_eq!(unorm8([0.0, 1.0, 0.0, 1.0]), [0, 255, 0, 255]);
        assert_eq!(unorm8(TRIANGLE_COLOR), [0, 255, 0, 255]);
    }

    #[test]
    fn scale_resolution_supersamples_target_viewport_and_scissor_together() {
        let mut state = DrawState::new(96, 48, &[], &[]);
        state.scale_resolution(2.0);
        assert_eq!((state.width, state.height), (192, 96));
        assert_eq!(state.viewport, [0.0, 0.0, 192.0, 96.0]);
        assert_eq!(state.scissor, [0, 0, 192, 96]);
    }

    #[test]
    fn scale_resolution_of_one_is_an_exact_no_op() {
        let mut state = DrawState::new(96, 48, &[], &[]);
        let (w, h, vp, sc) = (state.width, state.height, state.viewport, state.scissor);
        state.scale_resolution(1.0);
        assert_eq!((state.width, state.height), (w, h));
        assert_eq!(state.viewport, vp);
        assert_eq!(state.scissor, sc);
    }

    #[test]
    fn scale_resolution_clamps_wild_and_non_finite_factors() {
        // Above the cap → 4x, not 100x.
        let mut big = DrawState::new(100, 100, &[], &[]);
        big.scale_resolution(100.0);
        assert_eq!((big.width, big.height), (400, 400));
        // NaN / inf fall back to a no-op rather than producing a zero-sized target.
        let mut nan = DrawState::new(100, 100, &[], &[]);
        nan.scale_resolution(f32::NAN);
        assert_eq!((nan.width, nan.height), (100, 100));
    }

    #[test]
    fn unorm8_clamps_out_of_range() {
        assert_eq!(unorm8([-1.0, 2.0, 0.5, 1.0]), [0, 255, 128, 255]);
    }

    /// The test asserts corners are clear-colored, which only holds if the
    /// triangle stays away from them.
    #[test]
    fn triangle_covers_center_but_no_corner() {
        let inside = |px: f32, py: f32| {
            let [a, b, c] = TRIANGLE_VERTICES;
            let sign = |p: [f32; 4], q: [f32; 4]| {
                (px - q[0]) * (p[1] - q[1]) - (p[0] - q[0]) * (py - q[1])
            };
            let d1 = sign(a, b);
            let d2 = sign(b, c);
            let d3 = sign(c, a);
            let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
            let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
            !(neg && pos)
        };
        assert!(inside(0.0, 0.0), "NDC center must be covered");
        for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            assert!(!inside(x, y), "NDC corner ({x}, {y}) must stay uncovered");
        }
    }

    #[test]
    fn rendered_image_indexes_rows_major() {
        let img = RenderedImage {
            width: 2,
            height: 2,
            pixels: vec![
                1, 1, 1, 1, 2, 2, 2, 2, // row 0
                3, 3, 3, 3, 4, 4, 4, 4, // row 1
            ],
            bytes_per_pixel: 4,
        };
        assert_eq!(img.pixel(0, 0), Some([1, 1, 1, 1]));
        assert_eq!(img.pixel(1, 0), Some([2, 2, 2, 2]));
        assert_eq!(img.pixel(0, 1), Some([3, 3, 3, 3]));
        assert_eq!(img.pixel(1, 1), Some([4, 4, 4, 4]));
        assert_eq!(img.pixel(2, 0), None);
        assert_eq!(img.pixel(0, 2), None);
    }

    #[test]
    fn batched_colour_target_stays_attachment_resident_between_draws() {
        let transition = colour_target_attachment_transition(TargetLayout::ColorAttachment);
        assert_eq!(
            transition.old_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            transition.new_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            transition.src_access,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        );
        assert!(
            transition
                .dst_access
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_READ)
        );
    }

    #[test]
    fn flushed_colour_target_returns_to_transfer_source_layout() {
        let transition = colour_target_readback_transition(TargetLayout::ColorAttachment)
            .expect("attachment-resident target needs one readback transition");
        assert_eq!(
            transition.old_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(transition.new_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert!(
            colour_target_readback_transition(TargetLayout::TransferSrc).is_none(),
            "an already-readable target must not receive a redundant barrier"
        );
    }
}
