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
    BlendKey, DepthPipelineKey, DrawCaches, GraphicsPipelineKey, PersistentTarget, StencilKey,
    TargetKey,
};
use super::instance::VulkanDevice;
use super::shaders::{triangle_fragment_spirv, triangle_vertex_spirv};
use ash::vk::Handle;
use ash::{Device, util::Align, vk};
use std::mem;
use tracing::debug;
use xps5x_core::error::GpuError;

/// The color the attachment is cleared to before the draw, as linear RGBA.
///
/// Deliberately not black and not fully-saturated: a readback buffer that was
/// never written (all zeroes) or a mis-sized copy cannot masquerade as a
/// correct clear.
pub const CLEAR_COLOR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

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
        | vk::Format::B10G11R11_UFLOAT_PACK32 => Ok(4),
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
    pub buffers: Vec<Vec<u8>>,
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
    /// Volume depth: 1 for a 2D UAV; > 1 creates a `VK_IMAGE_TYPE_3D`
    /// storage image (measured: ASTRO.BOT's 240x135x64 UAV volumes).
    pub depth: u32,
    /// The Vulkan texel format matching the recompiled SPIR-V's `%ImageL`
    /// declaration: `R8G8B8A8_UNORM` (Rgba8) or `R16G16B16A16_SFLOAT`
    /// (Rgba16f, guest T# format 71).
    pub format: vk::Format,
    /// Initial content, `width * height * depth * texel_bytes()` bytes.
    pub pixels: Vec<u8>,
    /// Guest base address for the post-dispatch writeback.
    pub guest_base: u64,
}

impl StorageImageUpload {
    /// Bytes per texel of [`Self::format`].
    #[must_use]
    pub fn texel_bytes(&self) -> u32 {
        if self.format == vk::Format::R16G16B16A16_SFLOAT {
            8
        } else {
            4
        }
    }
}

/// One descriptor binding containing an array of storage images.
///
/// The recompiled SPIR-V declares `%textures2D_L` as
/// `OpTypeImage %float <dim> 0 0 0 2 <format>` — a STORAGE_IMAGE array in
/// the upload's format — indexed by dword 0 of the rewritten T# it reads
/// from the push constants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageImageBinding {
    pub binding: u32,
    pub images: Vec<StorageImageUpload>,
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
    /// Volume depth: 1 for 2D/cube; > 1 creates a `VK_IMAGE_TYPE_3D` image
    /// with a `3D` view (measured: ASTRO.BOT's 240x135x64 froxel/LUT
    /// volumes, T# type 10).
    pub depth: u32,
}

/// The sampled-image + sampler descriptor arrays one translated stage binds.
///
/// The recompiled SPIR-V declares `%textures2D_S` (an array of sampled images)
/// and `%samplers` (an array of samplers) and indexes them with the values the
/// push constants carry, so the arrays here must match the analyzer's counts
/// exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureBinding {
    pub sampled_binding: u32,
    pub sampler_binding: u32,
    pub textures: Vec<TextureUpload>,
    /// One entry per S#; only linear-vs-nearest is honoured today.
    pub linear_filter: Vec<bool>,
}

/// Per-stage resource ABI used by translated SPIR-V.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderStageBinding {
    pub stage: vk::ShaderStageFlags,
    pub descriptor_set_slot: u32,
    pub push_constant_offset: u32,
    pub push_constants: Vec<u8>,
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
    let timing = std::env::var_os("XPS5X_TIME_DRAW").is_some();
    let mut res = Resources::new(dev, &mut caches);
    let t0 = std::time::Instant::now();
    res.build(state)?;
    let t_build = t0.elapsed();
    let t1 = std::time::Instant::now();
    res.record_and_submit(state)?;
    let t_submit = t1.elapsed();
    let t2 = std::time::Instant::now();
    let color = res.read_back_color(state, bpp)?;
    let depth = res.read_back_depth(state)?;
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
        return Ok(DrawOutput { color, depth });
    }

    debug!(
        width = state.width,
        height = state.height,
        vertices = state.vertex_count,
        "offscreen draw rendered on {}",
        dev.device_name()
    );
    Ok(DrawOutput { color, depth })
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
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    /// True when the colour target/readback pair is per-draw (owned); false
    /// when they came from (or were donated to) the persistent-target cache.
    owns_target: bool,
    /// The persistent-target identity of this draw, when `target_base` named
    /// one — used to mark the entry synced after a successful readback.
    target_key: Option<TargetKey>,
    /// The stage A seed-skip: the persistent image still holds exactly the
    /// last readback (== `state.initial`), so the attachment LOADs from the
    /// GPU copy and the upload staging never happens.
    load_from_gpu: bool,
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
}

impl<'a> Resources<'a> {
    fn new(dev: &'a VulkanDevice, caches: &'a mut DrawCaches) -> Self {
        Self {
            dev,
            caches,
            image: vk::Image::null(),
            image_memory: vk::DeviceMemory::null(),
            image_view: vk::ImageView::null(),
            owns_target: false,
            target_key: None,
            load_from_gpu: false,
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
        }
    }

    fn device(&self) -> &Device {
        self.dev.device()
    }

    fn build(&mut self, state: &DrawState) -> Result<(), GpuError> {
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
            self.create_render_target(state, bpp as u32)?;
            if let Some(initial) = state.initial
                && !self.load_from_gpu
            {
                let (buffer, memory) =
                    self.create_buffer_with_bytes(initial, vk::BufferUsageFlags::TRANSFER_SRC)?;
                self.upload_buffer = buffer;
                self.upload_memory = memory;
            }
        }
        if let Some(depth) = &state.depth {
            self.create_depth_target(state.width, state.height, depth.format)?;
            self.create_depth_buffers(state.width, state.height, depth)?;
        }
        if let Some(vertices) = state.vertices {
            self.create_vertex_buffer(vertices)?;
        }
        if let Some(index) = &state.index {
            let (buffer, memory) =
                self.create_buffer_with_bytes(index.bytes, vk::BufferUsageFlags::INDEX_BUFFER)?;
            self.index_buffer = buffer;
            self.index_memory = memory;
        }
        self.create_guest_vertex_buffers(state)?;
        self.create_stage_resources(state)?;
        self.create_pipeline(state)?;
        self.create_command_resources()?;
        Ok(())
    }

    /// Create the depth attachment image and its view. Usage covers the draw
    /// itself, the post-draw readback (TRANSFER_SRC), and seeding prior
    /// contents (TRANSFER_DST).
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
                    | vk::ImageUsageFlags::TRANSFER_DST,
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
            let mut bytes = vec![0u8; total as usize];
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
            let (buffer, memory) =
                self.create_buffer_with_bytes(&bytes, vk::BufferUsageFlags::TRANSFER_SRC)?;
            self.depth_upload_buffer = buffer;
            self.depth_upload_memory = memory;
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
                // `entry.synced` carries the pre-acquisition value: the GPU
                // image equals the last readback, which is exactly what the
                // caller passes as `initial` (contract on `target_base`).
                self.load_from_gpu = entry.synced && state.initial.is_some();
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
                    synced: false,
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
            // TRANSFER_DST to seed it with the target's prior contents.
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: `info` is fully initialized and borrows nothing beyond this
        // call; the device is live. The handle is stored and destroyed in Drop.
        self.image = unsafe { self.device().create_image(&info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateImage failed: {e}")))?;

        // SAFETY: `self.image` was just created from this device.
        let reqs = unsafe { self.device().get_image_memory_requirements(self.image) };
        let type_index = self
            .dev
            .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);

        // SAFETY: allocation size/type come from this image's own requirements.
        self.image_memory = unsafe { self.device().allocate_memory(&alloc, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("image allocation failed: {e}")))?;

        // SAFETY: memory was allocated for exactly this image, offset 0 is
        // within it and satisfies the alignment requirement by construction.
        unsafe {
            self.device()
                .bind_image_memory(self.image, self.image_memory, 0)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkBindImageMemory failed: {e}")))?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.image)
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
        self.image_view = unsafe { self.device().create_image_view(&view_info, None) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkCreateImageView failed: {e}")))?;
        Ok(())
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

    fn create_buffer_with_bytes(
        &self,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), GpuError> {
        if bytes.is_empty() {
            return Err(GpuError::VulkanInitFailed(
                "buffer upload requested with no bytes".to_owned(),
            ));
        }
        let size = bytes.len() as vk::DeviceSize;
        let (buffer, memory) = self.create_buffer(
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let ptr = unsafe {
            self.device()
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkMapMemory failed: {e}")))?;

        // SAFETY: `memory` is HOST_VISIBLE and mapped for exactly
        // `bytes.len()` bytes. No GPU submission can reference it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            self.device().unmap_memory(memory);
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
            let allocation =
                self.create_buffer_with_bytes(&vertex.bytes, vk::BufferUsageFlags::VERTEX_BUFFER)?;
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
                if textures.textures.is_empty() && textures.linear_filter.is_empty() {
                    return Err(GpuError::PipelineCreationFailed(
                        "sampled-image and sampler descriptor arrays are both empty".to_owned(),
                    ));
                }
                if !textures.textures.is_empty() {
                    bindings.push(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(textures.sampled_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(textures.textures.len() as u32)
                            .stage_flags(stage.stage),
                    );
                }
                if !textures.linear_filter.is_empty() {
                    bindings.push(
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(textures.sampler_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(textures.linear_filter.len() as u32)
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
                        .map_or(0, |textures| textures.linear_filter.len() as u32)
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
        // Persistent pool from the cache, reset for this draw (the previous
        // draw's sets completed with its fence) and grown when this draw
        // needs more than it holds.
        self.descriptor_pool =
            self.caches
                .descriptor_pool(self.dev, resource_stages.len() as u32, &pool_sizes)?;

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
            let mut sampler_infos = Vec::new();
            if let Some(storage) = &stage.storage_buffers {
                for bytes in &storage.buffers {
                    let allocation =
                        self.create_buffer_with_bytes(bytes, vk::BufferUsageFlags::STORAGE_BUFFER)?;
                    self.storage_buffers.push(allocation);
                }
                let first_buffer = self.storage_buffers.len() - storage.buffers.len();
                buffer_infos = self.storage_buffers[first_buffer..]
                    .iter()
                    .map(|(buffer, _)| {
                        vk::DescriptorBufferInfo::default()
                            .buffer(*buffer)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)
                    })
                    .collect();
            }
            if let Some(textures) = &stage.textures {
                for upload in &textures.textures {
                    self.create_texture_image(upload, stage.stage)?;
                }
                let first_texture = self.texture_uploads.len() - textures.textures.len();
                image_infos = self.texture_uploads[first_texture..]
                    .iter()
                    .map(|texture| {
                        vk::DescriptorImageInfo::default()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(texture.view)
                    })
                    .collect();
                for &linear in &textures.linear_filter {
                    self.samplers.push(self.caches.sampler(self.dev, linear)?);
                }
                let first_sampler = self.samplers.len() - textures.linear_filter.len();
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
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(textures.sampled_binding)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&image_infos),
                    );
                }
                if !textures.linear_filter.is_empty() {
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
        let (staging_buffer, staging_memory) =
            self.create_buffer_with_bytes(&upload.pixels, vk::BufferUsageFlags::TRANSFER_SRC)?;
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
            layers: upload.layers,
            depth: upload.depth.max(1),
            stage,
        });
        let slot = self.texture_uploads.len() - 1;

        // depth > 1 is a 3D volume (T# type 10): one layer, `depth` slices.
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
            .array_layers(upload.layers)
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
            .view_type(if upload.cube {
                vk::ImageViewType::CUBE
            } else if volume {
                vk::ImageViewType::TYPE_3D
            } else if upload.layers > 1 {
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
                layer_count: upload.layers,
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
        let size =
            vk::DeviceSize::from(width) * vk::DeviceSize::from(height) * vk::DeviceSize::from(bpp);
        // The whole frame is copied out of this buffer on the CPU. Without
        // HOST_CACHED that copy reads uncached memory, which is ~50x slower:
        // measured 32 ms to read back one 1080p frame, dwarfing the ~1 ms of
        // actual GPU work. Prefer a cached+coherent type (fast reads, no manual
        // invalidate); fall back to coherent-only where the device has no such
        // type.
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let (buffer, memory) = self
            .create_buffer(
                size,
                vk::BufferUsageFlags::TRANSFER_DST,
                host | vk::MemoryPropertyFlags::HOST_CACHED,
            )
            .or_else(|_| self.create_buffer(size, vk::BufferUsageFlags::TRANSFER_DST, host))?;
        self.readback_buffer = buffer;
        self.readback_memory = memory;
        Ok(())
    }

    fn create_pipeline(&mut self, state: &DrawState) -> Result<(), GpuError> {
        // Cached by SPIR-V content: the translate cache upstream already
        // dedups shaders, so repeated binds of one shader resolve to one
        // canonical VkShaderModule instead of a fresh module per draw.
        self.vertex_module = self.caches.shader_module(self.dev, state.vs_spirv)?;
        self.fragment_module = self.caches.shader_module(self.dev, state.fs_spirv)?;

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
                            .input_rate(vk::VertexInputRate::VERTEX)
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
                .map(|b| (b.binding, b.stride))
                .collect(),
            vertex_attributes: vertex_attributes
                .iter()
                .map(|a| (a.location, a.binding, a.format.as_raw(), a.offset))
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

        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(state.color_write_mask)
            .blend_enable(state.blend.enable)
            .src_color_blend_factor(state.blend.src_color)
            .dst_color_blend_factor(state.blend.dst_color)
            .color_blend_op(state.blend.color_op)
            .src_alpha_blend_factor(state.blend.src_alpha)
            .dst_alpha_blend_factor(state.blend.dst_alpha)
            .alpha_blend_op(state.blend.alpha_op)];
        // A depth-only draw declares zero colour attachments; the blend
        // attachment count must match the pipeline's colour attachment count.
        let color_blend_attachments: &[vk::PipelineColorBlendAttachmentState] =
            if state.color_output {
                &blend_attachments
            } else {
                &[]
            };
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(color_blend_attachments);

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
            vec![state.format]
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
        self.pipeline = pipeline;
        Ok(())
    }

    /// The persistent command buffer + fence from the cache (fence reset for
    /// this submission). The command buffer is implicitly reset by
    /// `vkBeginCommandBuffer` — the pool was created RESET_COMMAND_BUFFER.
    fn create_command_resources(&mut self) -> Result<(), GpuError> {
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
                    dst_stage: shader_stage_to_pipeline(texture.stage),
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
                let (old_layout, src_stage) = if self.load_from_gpu {
                    (
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::TRANSFER,
                    )
                } else {
                    (
                        vk::ImageLayout::UNDEFINED,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                    )
                };
                self.image_barrier_layers(
                    vk::ImageAspectFlags::COLOR,
                    self.image,
                    1,
                    ImageTransition {
                        old_layout,
                        new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        src_access: vk::AccessFlags::empty(),
                        dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags::COLOR_ATTACHMENT_READ,
                        src_stage,
                        dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    },
                );
            }
        }

        // Depth/stencil attachment: transition it to the attachment layout,
        // seeding a plane that LOADs prior contents. A CLEAR plane starts
        // undefined and is cleared by the render pass, so it needs no seed.
        if let Some(depth) = &state.depth {
            let aspect = depth_aspect_mask(depth.format);
            let (depth_load, stencil_load) = depth_loads(depth);
            if self.depth_upload_buffer != vk::Buffer::null() {
                self.image_barrier_layers(
                    aspect,
                    self.depth_image,
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
                    let offset = depth_plane_bytes(width, height, depth.format)?;
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
                self.image_barrier_layers(
                    aspect,
                    self.depth_image,
                    1,
                    ImageTransition {
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        src_access: vk::AccessFlags::empty(),
                        dst_access: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
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
            // `XPS5X_FORCE_CLEAR=1` is the COVERAGE PROBE: it forces every draw
            // to CLEAR instead of LOAD, so the target ends as pure CLEAR_COLOR
            // unless some draw actually produced a fragment. It answers the one
            // question a black frame cannot — "is nothing being drawn, or is
            // something being drawn in black?" — without shader surgery.
            // MEASURED on Minecraft: 12,083 draws, final frame 100% uniform
            // CLEAR_COLOR (2,073,600/2,073,600 pixels) => ZERO coverage; not a
            // single fragment from any draw. It also proves the clear reaches
            // the image and the RGBA8 readback/dump is faithful (blue came back
            // 0xBF exactly).
            .load_op(if std::env::var_os("XPS5X_FORCE_CLEAR").is_some() {
                vk::AttachmentLoadOp::CLEAR
            } else if state.initial.is_some() {
                vk::AttachmentLoadOp::LOAD
            } else {
                vk::AttachmentLoadOp::CLEAR
            })
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear);
        let color_attachment_slice = [color_attachment];
        // A depth-only draw declares zero colour attachments, matching the
        // pipeline built with no colour formats (`create_pipeline`).
        let color_attachments: &[vk::RenderingAttachmentInfo] = if state.color_output {
            &color_attachment_slice
        } else {
            &[]
        };

        // Vulkan 1.3 dynamic-rendering depth/stencil attachments. `depth_view`
        // carries both planes (aspect from `depth_aspect_mask`), so depth and
        // stencil reference the same view.
        let depth_attachment = state.depth.as_ref().map(|depth| {
            let (depth_load, _) = depth_loads(depth);
            depth_stencil_attachment(self.depth_view, depth_load, depth)
        });
        let stencil_attachment = state
            .depth
            .as_ref()
            .filter(|depth| has_stencil_plane(depth.format) && depth.stencil_test_enable)
            .map(|depth| {
                let (_, stencil_load) = depth_loads(depth);
                depth_stencil_attachment(self.depth_view, stencil_load, depth)
            });

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(color_attachments);
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

        // COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL for the copy out.
        if state.color_output {
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
        }

        // Depth/stencil copy out: DEPTH_STENCIL_ATTACHMENT_OPTIMAL ->
        // TRANSFER_SRC, then copy the depth plane (and stencil plane, if any)
        // into the readback buffer at the layout `read_back_depth` expects.
        if let Some(depth) = &state.depth {
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
            let mut regions = vec![depth_copy_region(
                width,
                height,
                vk::ImageAspectFlags::DEPTH,
                0,
            )];
            if has_stencil_plane(depth.format) {
                let offset = depth_plane_bytes(width, height, depth.format)?;
                regions.push(depth_copy_region(
                    width,
                    height,
                    vk::ImageAspectFlags::STENCIL,
                    offset,
                ));
            }
            // SAFETY: the depth image is in TRANSFER_SRC per the barrier above,
            // and `depth_readback_buffer` was sized for exactly these planes.
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

        // Make the transfer writes visible to host reads of the mapped memory.
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
            self.device()
                .end_command_buffer(self.command_buffer)
                .map_err(|e| GpuError::VulkanInitFailed(format!("vkEndCommandBuffer: {e}")))?;
        }

        let command_buffers = [self.command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);

        // SAFETY: the command buffer is recorded and not already pending; the
        // fence is unsignaled and unused; the queue came from this device.
        unsafe {
            self.device()
                .queue_submit(self.dev.queue(), &[submit], self.fence)
        }
        .map_err(|e| GpuError::VulkanInitFailed(format!("vkQueueSubmit failed: {e}")))?;

        // Wait for the GPU. u64::MAX = no timeout; a hang here means a driver
        // fault, which surfaces as a hung test rather than silent bad pixels.
        // SAFETY: the fence was just submitted with and belongs to this device.
        unsafe { self.device().wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| GpuError::VulkanInitFailed(format!("vkWaitForFences failed: {e}")))?;
        Ok(())
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

        // SAFETY: `ptr` is a valid mapping of `size` bytes (the buffer was
        // allocated at exactly this size), initialized by the completed copy.
        // The bytes are read into an owned Vec before unmapping, so no
        // reference outlives the mapping.
        let pixels = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), size).to_vec() };

        // SAFETY: the memory is currently mapped by the call above and no
        // references into it remain.
        unsafe { self.device().unmap_memory(self.readback_memory) };
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

impl Drop for Resources<'_> {
    fn drop(&mut self) {
        let guest_vertex_buffers = mem::take(&mut self.guest_vertex_buffers);
        let storage_buffers = mem::take(&mut self.storage_buffers);
        let texture_uploads = mem::take(&mut self.texture_uploads);
        // NOT destroyed here — owned by the device's DrawCaches and reused by
        // the next draw: pipeline, pipeline layout, descriptor set layouts,
        // shader modules, samplers, command buffer, fence, descriptor pool,
        // and (when `owns_target` is false) the colour image + its readback
        // buffer. `DrawCaches::destroy` releases them with the device.
        //
        // SAFETY: every handle destroyed below was created from `self.dev`'s
        // device for this draw alone and is destroyed exactly once, children
        // before parents. `device_wait_idle` ensures no submitted work still
        // references them; its error is ignored because drop must not panic
        // and a lost device cannot be recovered. Null handles are skipped, so
        // a partially-built `Resources` (an error during `build`) cleans up
        // correctly.
        unsafe {
            let d = self.device();
            let _ = d.device_wait_idle();

            if self.upload_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.upload_buffer, None);
            }
            if self.upload_memory != vk::DeviceMemory::null() {
                d.free_memory(self.upload_memory, None);
            }
            if self.vertex_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.vertex_buffer, None);
            }
            if self.vertex_memory != vk::DeviceMemory::null() {
                d.free_memory(self.vertex_memory, None);
            }
            if self.index_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.index_buffer, None);
            }
            if self.index_memory != vk::DeviceMemory::null() {
                d.free_memory(self.index_memory, None);
            }
            for (buffer, memory) in guest_vertex_buffers {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
            }
            for (buffer, memory) in storage_buffers {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
            }
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
                d.destroy_buffer(texture.staging_buffer, None);
                d.free_memory(texture.staging_memory, None);
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
            if self.depth_view != vk::ImageView::null() {
                d.destroy_image_view(self.depth_view, None);
            }
            if self.depth_image != vk::Image::null() {
                d.destroy_image(self.depth_image, None);
            }
            if self.depth_memory != vk::DeviceMemory::null() {
                d.free_memory(self.depth_memory, None);
            }
            if self.depth_upload_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.depth_upload_buffer, None);
            }
            if self.depth_upload_memory != vk::DeviceMemory::null() {
                d.free_memory(self.depth_upload_memory, None);
            }
            if self.depth_readback_buffer != vk::Buffer::null() {
                d.destroy_buffer(self.depth_readback_buffer, None);
            }
            if self.depth_readback_memory != vk::DeviceMemory::null() {
                d.free_memory(self.depth_readback_memory, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::shaders::TRIANGLE_COLOR;
    use super::*;

    #[test]
    fn unorm8_maps_endpoints_exactly() {
        assert_eq!(unorm8([0.0, 1.0, 0.0, 1.0]), [0, 255, 0, 255]);
        assert_eq!(unorm8(TRIANGLE_COLOR), [0, 255, 0, 255]);
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
}
