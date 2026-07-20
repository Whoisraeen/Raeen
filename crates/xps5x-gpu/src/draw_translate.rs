//! Translate decoded PM4 register state into a Vulkan draw.
//!
//! This is the seam between `kyty-graphics` (which has no Vulkan dependency and
//! so terminates its command-processor walk at [`DrawSink`]) and the offscreen
//! Vulkan path. It replaces Kyty's `GraphicsRender` layer for the Phase 1
//! subset.
//!
//! # What it does not do
//!
//! Everything here is driven by registers — extent, format, viewport, scissor,
//! topology, shaders. Nothing is a fixture. In exchange, a draw whose registers
//! this slice cannot honour is a **named error**, never a fallback. Shader
//! binds split two ways:
//!
//! - **embedded** (Kyty's clear/blit shaders) — assembled from the embedded
//!   SPIR-V table, exactly as in Phase 1;
//! - **non-embedded** — the real title path: the code is fetched from guest
//!   memory and recompiled through [`crate::shader_fetch`] (ShaderMemory,
//!   Phase 2). A stage that fails translation warns once (negative-cached,
//!   named reason) and the **draw is skipped, not the whole DCB** — a title
//!   frame mixes translatable and untranslatable shaders and one bad shader
//!   must not hide every other draw.

use crate::shader_fetch::{ShaderTranslateCache, TranslatedShader};
use crate::vulkan::compute::{ComputeState, dispatch_compute};
use crate::vulkan::instance::VulkanDevice;
use crate::vulkan::offscreen::{
    BlendState, CLEAR_COLOR, DrawState, RenderedImage, ShaderStageBinding, StorageBufferBinding,
    StorageImageBinding, StorageImageUpload, TextureBinding, TextureUpload, VertexAttributeData,
    VertexBufferData, render_draw,
};
use ash::vk;
use kyty_graphics::hw_regs::{ComputeShaderInfo, Context, Shader, UserConfig};
use kyty_graphics::run::{DrawError, DrawSink, IndexedDraw};
use kyty_graphics::shader::resources::{
    ShaderBindResources, ShaderPixelInputInfo, ShaderTextureUsage, ShaderVertexInputInfo,
};
use kyty_graphics::shader::{spirv_get_embedded_ps, spirv_get_embedded_vs};
use kyty_graphics::spirv_asm;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// `VGT_PRIMITIVE_TYPE` values Kyty's Gen5 path emits.
mod prim {
    /// NONE: a draw issued with no primitive type draws nothing on hardware.
    pub const NONE: u32 = 0;
    pub const TRIANGLE_LIST: u32 = 4;
    pub const TRIANGLE_FAN: u32 = 5;
    pub const TRIANGLE_STRIP: u32 = 6;
    /// AMD `DI_PT_POLYGON`: one convex polygon per draw. Measured: ASTRO.BOT
    /// issues its early-boot draws with this type (226 skips before support).
    /// A convex polygon rasterizes exactly as a triangle fan of its vertices
    /// (identical to a list for the 3-vertex case). SharpEmu ships the same
    /// draws through its TriangleList catch-all; the fan is the faithful
    /// mapping.
    pub const POLYGON: u32 = 7;
    /// Kyty's clear/blit primitive. Rasterized as a 4-vertex strip quad.
    pub const RECT_LIST: u32 = 17;
}

fn err(msg: impl Into<String>) -> DrawError {
    DrawError(msg.into())
}

fn color_output_disabled(ctx: &Context) -> bool {
    ctx.render_target_mask == 0
}

/// Map `CB_TARGET_MASK`'s MRT0 nibble to Vulkan's colour write mask.
///
/// Bit per channel, R in bit 0 through A in bit 3 — the same shape Vulkan uses,
/// so this is a rename rather than an approximation.
///
/// This used to reject anything but `0xF` as "a partial write mask; only 0xF is
/// supported", inherited from Kyty accepting only all-or-nothing targets. Two
/// things were wrong with that. Partial masks are ordinary — `0x7` is RGB with
/// alpha left alone, which Minecraft issues — and the rejection was an `Err` out
/// of `draw_state_from_regs`, which propagates through `run?` in `execute_dcb_cp`
/// and **abandons every remaining draw in the command buffer**. One unremarkable
/// mask killed a whole DCB.
fn vulkan_color_write_mask(target_mask: u32) -> vk::ColorComponentFlags {
    let mut flags = vk::ColorComponentFlags::empty();
    for (bit, flag) in [
        (0, vk::ColorComponentFlags::R),
        (1, vk::ColorComponentFlags::G),
        (2, vk::ColorComponentFlags::B),
        (3, vk::ColorComponentFlags::A),
    ] {
        if target_mask & (1 << bit) != 0 {
            flags |= flag;
        }
    }
    flags
}

/// Map a `CB_BLEND*_CONTROL` 5-bit blend factor to Vulkan.
///
/// The encoding is Kyty's `GraphicsRender.cpp` blend switch. Dual-source and
/// BOTH_SRC_ALPHA factors have no single-source Vulkan equivalent — a named
/// error, not a silently-wrong ZERO.
fn gen5_blend_factor(code: u8) -> Result<vk::BlendFactor, DrawError> {
    match code {
        0x00 => Ok(vk::BlendFactor::ZERO),
        0x01 => Ok(vk::BlendFactor::ONE),
        0x02 => Ok(vk::BlendFactor::SRC_COLOR),
        0x03 => Ok(vk::BlendFactor::ONE_MINUS_SRC_COLOR),
        0x04 => Ok(vk::BlendFactor::SRC_ALPHA),
        0x05 => Ok(vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
        0x06 => Ok(vk::BlendFactor::DST_ALPHA),
        0x07 => Ok(vk::BlendFactor::ONE_MINUS_DST_ALPHA),
        0x08 => Ok(vk::BlendFactor::DST_COLOR),
        0x09 => Ok(vk::BlendFactor::ONE_MINUS_DST_COLOR),
        0x0a => Ok(vk::BlendFactor::SRC_ALPHA_SATURATE),
        0x0d => Ok(vk::BlendFactor::CONSTANT_COLOR),
        0x0e => Ok(vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR),
        0x13 => Ok(vk::BlendFactor::CONSTANT_ALPHA),
        0x14 => Ok(vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA),
        other => Err(err(format!(
            "blend factor {other:#04x} not implemented (dual-source/BOTH_SRC_ALPHA \
             factors need VK dual-source, which this pipeline does not use)"
        ))),
    }
}

/// Map a `CB_BLEND*_CONTROL` 3-bit combine function to Vulkan.
fn gen5_blend_op(code: u8) -> Result<vk::BlendOp, DrawError> {
    match code {
        0 => Ok(vk::BlendOp::ADD),
        1 => Ok(vk::BlendOp::SUBTRACT),
        2 => Ok(vk::BlendOp::MIN),
        3 => Ok(vk::BlendOp::MAX),
        4 => Ok(vk::BlendOp::REVERSE_SUBTRACT),
        other => Err(err(format!(
            "blend combine function {other} not implemented"
        ))),
    }
}

/// Build the attachment's blend state from `CB_BLEND0_CONTROL` +
/// `CB_BLEND_{RED,GREEN,BLUE,ALPHA}`. When the register clears
/// `separate_alpha_blend`, the alpha channel uses the *colour* factors — that
/// is what the hardware does, not a shortcut.
fn blend_state_from_regs(ctx: &Context) -> Result<BlendState, DrawError> {
    let bc = &ctx.blend_control[0];
    let color_src = gen5_blend_factor(bc.color_srcblend)?;
    let color_dst = gen5_blend_factor(bc.color_destblend)?;
    let color_op = gen5_blend_op(bc.color_comb_fcn)?;
    let (alpha_src, alpha_dst, alpha_op) = if bc.separate_alpha_blend {
        (
            gen5_blend_factor(bc.alpha_srcblend)?,
            gen5_blend_factor(bc.alpha_destblend)?,
            gen5_blend_op(bc.alpha_comb_fcn)?,
        )
    } else {
        (color_src, color_dst, color_op)
    };
    Ok(BlendState {
        enable: bc.enable,
        src_color: color_src,
        dst_color: color_dst,
        color_op,
        src_alpha: alpha_src,
        dst_alpha: alpha_dst,
        alpha_op,
        constants: [
            ctx.blend_color.red,
            ctx.blend_color.green,
            ctx.blend_color.blue,
            ctx.blend_color.alpha,
        ],
    })
}

/// Map `CB_COLOR0_INFO`'s format/channel_type/channel_order triple to Vulkan.
///
/// Only the combinations the Phase 1 path can honour are accepted; anything
/// else is named rather than approximated.
fn vulkan_format(
    format: u32,
    channel_type: u32,
    channel_order: u32,
) -> Result<vk::Format, DrawError> {
    match (format, channel_type, channel_order) {
        (0xa, 0, 0) => Ok(vk::Format::R8G8B8A8_UNORM),
        (0xa, 6, 0) => Ok(vk::Format::R8G8B8A8_SRGB),
        (0xa, 0, 1) => Ok(vk::Format::B8G8R8A8_UNORM),
        (0xa, 6, 1) => Ok(vk::Format::B8G8R8A8_SRGB),
        // 10_11_11 / 11_11_10 FLOAT (channel_type 7): the packed HDR
        // intermediate render target ASTRO.BOT draws into. SharpEmu maps both
        // CB formats 6 and 7 with channel_type 7 to B10G11R11_UFLOAT_PACK32.
        (0x6 | 0x7, 7, 0) => Ok(vk::Format::B10G11R11_UFLOAT_PACK32),
        // 16_16_16_16 FLOAT (CB format 0xc, channel_type 7): the 64bpp HDR main
        // scene target ASTRO.BOT renders into before tone-mapping. The offscreen
        // readback is bpp-aware (8 bytes/pixel for this one).
        (0xc, 7, 0) => Ok(vk::Format::R16G16B16A16_SFLOAT),
        // CB format 0x3 (8_8 UNORM) is NOT accepted, deliberately. ASTRO.BOT
        // asks for it (16 draws, once the vertex-fetch fix let them reach this
        // check) and the enum numbering says it is R8G8_UNORM — but MEASURED,
        // mapping it to R8G8_UNORM makes those draws lose the Vulkan device
        // (vkQueueSubmit -> VK_ERROR_DEVICE_LOST) and takes the whole run from
        // 10 presented frames to 0. Something else in the pipeline (attachment
        // usage, the fragment shader's 4-component export, or blend state)
        // cannot honour a 2-channel target yet. A named error costs 16 draws;
        // a device loss costs every frame. Re-enable only together with
        // whatever makes a 2-channel attachment actually work.
        _ => Err(err(format!(
            "unsupported CB_COLOR0_INFO format={format:#x} channel_type={channel_type} \
             channel_order={channel_order} — no Vulkan format mapping"
        ))),
    }
}

/// Assemble the SPIR-V for an embedded shader stage.
fn assemble_embedded(id: u32, stage: &str) -> Result<Vec<u32>, DrawError> {
    let source = match stage {
        "vs" => spirv_get_embedded_vs(id),
        _ => spirv_get_embedded_ps(id),
    }
    .map_err(|e| err(format!("embedded {stage} id {id}: {e}")))?;

    spirv_asm::assemble(source).map_err(|e| err(format!("assembling embedded {stage}: {e}")))
}

/// Both stages' SPIR-V, each either embedded or fetched from guest memory.
#[derive(Debug)]
struct ResolvedShaders {
    vs: Arc<Vec<u32>>,
    ps: Arc<Vec<u32>>,
    vs_info: ShaderVertexInputInfo,
    ps_info: ShaderPixelInputInfo,
}

/// Resolve the bound VS/PS to SPIR-V through the embedded table or the
/// guest-memory fetch+translate cache.
///
/// The error carries the named reason; for guest shaders the loud warn
/// already happened (once) inside the cache, so the caller can degrade a
/// repeat failure quietly.
fn resolve_shaders(
    cache: &mut ShaderTranslateCache,
    ctx: &Context,
    sh: &Shader,
) -> Result<ResolvedShaders, DrawError> {
    let (vs, vs_info) = if sh.vs.vs_embedded {
        let vs = Arc::new(assemble_embedded(sh.vs.vs_embedded_id, "vs")?);
        // An embedded VS exports exactly its position+param set; the PS
        // input-info builder only needs the export count.
        let vs_info = ShaderVertexInputInfo {
            export_count: ctx.sh_regs.get_export_count() as i32,
            ..Default::default()
        };
        (vs, vs_info)
    } else {
        let t: TranslatedShader = cache
            .translate_vs(&sh.vs, &ctx.sh_regs)
            .map_err(|e| err(e.to_string()))?;
        (t.spirv, t.vs_info)
    };

    let (ps, ps_info) = if sh.ps.ps_embedded {
        (
            Arc::new(assemble_embedded(sh.ps.ps_embedded_id, "ps")?),
            ShaderPixelInputInfo::default(),
        )
    } else {
        let translated = cache
            .translate_ps(&sh.ps, &ctx.sh_regs, &vs_info)
            .map_err(|e| err(e.to_string()))?;
        (translated.spirv, translated.ps_info)
    };

    debug!(
        vs_resources = vs_info.resources_num,
        vs_buffers = vs_info.buffers_num,
        vs_push_bytes = vs_info.bind.push_constant_size,
        vs_storage_buffers = vs_info.bind.storage_buffers.buffers_num,
        ps_push_bytes = ps_info.bind.push_constant_size,
        ps_storage_buffers = ps_info.bind.storage_buffers.buffers_num,
        ps_textures = ps_info.bind.textures2d.textures_num,
        ps_samplers = ps_info.bind.samplers.samplers_num,
        "resolved guest shader resource ABI"
    );

    Ok(ResolvedShaders {
        vs,
        ps,
        vs_info,
        ps_info,
    })
}

/// Fetch an indexed draw's index buffer from guest memory, in a form Vulkan can
/// bind directly.
///
/// `VGT_INDEX_TYPE` (bits 1:0 of `index_type_and_size`) gives the guest element
/// size: 0 = 16-bit, 1 = 32-bit, 2 = 8-bit. Vulkan only guarantees UINT16 and
/// UINT32; UINT8 needs `VK_EXT_index_type_uint8`, which this device does not
/// enable, so 8-bit indices are widened to 16-bit on the CPU rather than taking
/// a dependency on an extension for a rare case.
fn fetch_index_buffer(draw: &IndexedDraw) -> Result<(Vec<u8>, vk::IndexType), DrawError> {
    if draw.index_addr == 0 || draw.index_count == 0 {
        return Err(err(format!(
            "indexed draw with no index buffer: addr={:#x} count={}",
            draw.index_addr, draw.index_count
        )));
    }
    // An index buffer is `index_base + index_offset * element_bytes`, which
    // routinely lands off a dword boundary — so this needs a byte-granular
    // read, not the dword-aligned `read_guest_bytes`.
    let read = |bytes_per: u64| {
        read_guest_bytes_unaligned(
            draw.index_addr,
            u64::from(draw.index_count) * bytes_per,
            "index buffer",
        )
    };
    match draw.index_type_and_size & 0x3 {
        0 => Ok((read(2)?, vk::IndexType::UINT16)),
        2 => {
            let widened = read(1)?
                .iter()
                .take(draw.index_count as usize)
                .flat_map(|&b| u16::from(b).to_le_bytes())
                .collect();
            Ok((widened, vk::IndexType::UINT16))
        }
        // 32-bit (1), and the reserved 3 as the widest sane element.
        _ => Ok((read(4)?, vk::IndexType::UINT32)),
    }
}

/// Read `size` guest bytes starting at an arbitrary (possibly unaligned)
/// address.
///
/// The underlying identity-map reader is dword-granular and rejects a
/// non-4-aligned address, but an index buffer legitimately starts at any byte
/// offset (`index_base + index_offset * element_bytes`). Read the aligned dword
/// window that covers `[addr, addr + size)` and slice out the requested bytes.
fn read_guest_bytes_unaligned(addr: u64, size: u64, kind: &str) -> Result<Vec<u8>, DrawError> {
    if size == 0 {
        return Err(err(format!("{kind} at {addr:#x} has zero size")));
    }
    let head = addr & 0x3; // bytes to skip inside the first dword
    let aligned = addr - head;
    let span = (head + size).next_multiple_of(4);
    let window = read_guest_bytes(aligned, span, kind)?;
    let start = head as usize;
    let end = start + size as usize;
    window.get(start..end).map(<[u8]>::to_vec).ok_or_else(|| {
        err(format!(
            "{kind} at {addr:#x}: slice {start}..{end} outside read window"
        ))
    })
}

fn read_guest_bytes(addr: u64, size: u64, kind: &str) -> Result<Vec<u8>, DrawError> {
    if size == 0 || !size.is_multiple_of(4) {
        return Err(err(format!(
            "{kind} at {addr:#x} has invalid byte size {size}"
        )));
    }
    let count = u32::try_from(size / 4)
        .map_err(|_| err(format!("{kind} at {addr:#x} is too large: {size} bytes")))?;
    let words = crate::guest_mem::read_dwords_validated(addr, count).ok_or_else(|| {
        // A zero-ish prefix means the base itself is wild (mis-decoded
        // pointer); a page-aligned interior cut means the tail is
        // reserved-but-uncommitted (lazy guest memory); a prefix equal to the
        // size means the read was refused by the resource cap, not by memory.
        let good = crate::guest_mem::readable_prefix(addr, size);
        err(format!(
            "{kind} guest range {addr:#x}..{:#x} is not fully readable \
             (readable prefix {good:#x} of {size:#x})",
            addr.saturating_add(size)
        ))
    })?;
    let bytes: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    if let Ok(dir) = std::env::var("XPS5X_DUMP_GPU_RESOURCES")
        && !dir.is_empty()
    {
        let safe_kind = kind.replace(' ', "_");
        let path = std::path::Path::new(&dir).join(format!("{safe_kind}_{addr:012x}_{size}.bin"));
        if !path.exists()
            && let Err(error) =
                std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, &bytes))
        {
            debug!(%error, path = %path.display(), "guest GPU resource dump failed");
        }
    }
    Ok(bytes)
}

fn gen5_vertex_format(format: u8) -> Result<vk::Format, DrawError> {
    // Gen5 unified-format code → Vulkan, per SharpEmu's Gfx10UnifiedFormat
    // table (the RDNA2 authority): 64 → (11,7) = 32_32_FLOAT,
    // 74 → (13,7) = 32_32_32_FLOAT, 77 → (14,7) = 32_32_32_32_FLOAT,
    // 56 → (10,0) = 8_8_8_8 UNORM, 71 → (12,7) = 16_16_16_16_FLOAT,
    // 11 → (2,4) = 16 UINT (measured: Minecraft's packed per-vertex value),
    // 57 → (10,1) = 8_8_8_8 SNORM (same UI draw, next attribute).
    match format {
        74 => Ok(vk::Format::R32G32B32_SFLOAT),
        64 => Ok(vk::Format::R32G32_SFLOAT),
        77 => Ok(vk::Format::R32G32B32A32_SFLOAT),
        56 => Ok(vk::Format::R8G8B8A8_UNORM),
        57 => Ok(vk::Format::R8G8B8A8_SNORM),
        71 => Ok(vk::Format::R16G16B16A16_SFLOAT),
        23 => Ok(vk::Format::R16G16_UNORM),
        // MUST be a float-convertible format, NOT an integer one. The SPIR-V
        // we generate declares EVERY vertex input as float/vecN-float
        // (`Spirv::WriteGlobalVariables` only ever emits %_ptr_Input_float /
        // v2float / v3float / v4float), and Vulkan requires the attribute
        // format's numeric type to match the shader input's. R16_UINT against
        // a float32 input is an INVALID pipeline — measured on Minecraft, the
        // validation layer reports "pVertexAttributeDescriptions[1].format
        // (VK_FORMAT_R16_UINT) at Location 1 does not match [Input variable,
        // Location 1] type of (float32)" and the draw contributes NO fragment,
        // which is why every target stayed byte-exactly zero.
        // R16_USCALED delivers the same integer VALUE converted to float,
        // which is what a float input expects.
        11 => Ok(vk::Format::R16_USCALED),
        other => Err(err(format!(
            "unsupported Gen5 vertex-buffer format {other}"
        ))),
    }
}

fn prepare_vertex_inputs(
    info: &ShaderVertexInputInfo,
) -> Result<(Vec<VertexBufferData>, Vec<VertexAttributeData>), DrawError> {
    let buffers_num = usize::try_from(info.buffers_num)
        .map_err(|_| err(format!("negative vertex buffer count {}", info.buffers_num)))?;
    let resources_num = usize::try_from(info.resources_num).map_err(|_| {
        err(format!(
            "negative vertex resource count {}",
            info.resources_num
        ))
    })?;
    if buffers_num > info.buffers.len() || resources_num > info.resources.len() {
        return Err(err("vertex input metadata exceeds fixed resource arrays"));
    }

    let mut buffers = Vec::with_capacity(buffers_num);
    let mut attributes = Vec::with_capacity(resources_num);
    for (binding, guest) in info.buffers[..buffers_num].iter().enumerate() {
        let size = u64::from(guest.stride)
            .checked_mul(u64::from(guest.num_records))
            .ok_or_else(|| err("vertex buffer size overflow"))?;
        let bytes = read_guest_bytes(guest.addr, size, "vertex buffer")?;
        // Coverage probe (XPS5X_TRACE_DRAWS): XPS5X_FORCE_CLEAR proved well-
        // formed quads rasterize ZERO fragments, so the suspect is the vertex
        // DATA behind the V#. Report the descriptor and whether the guest bytes
        // are actually non-zero — an all-zero buffer collapses every vertex to
        // the origin and would explain zero coverage exactly.
        if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static VB_SEEN: AtomicU32 = AtomicU32::new(0);
            if VB_SEEN.fetch_add(1, Ordering::Relaxed) < 12 {
                let nz = bytes.iter().filter(|&&b| b != 0).count();
                tracing::warn!(
                    binding,
                    addr = format_args!("{:#x}", guest.addr),
                    stride = guest.stride,
                    num_records = guest.num_records,
                    size,
                    non_zero_bytes = nz,
                    head = format_args!("{:02x?}", &bytes[..bytes.len().min(32)]),
                    "TRACE_DRAWS: vertex buffer content"
                );
            }
        }
        buffers.push(VertexBufferData {
            bytes,
            stride: guest.stride,
        });

        let attr_num = usize::try_from(guest.attr_num).map_err(|_| {
            err(format!(
                "negative vertex attribute count {}",
                guest.attr_num
            ))
        })?;
        if attr_num > guest.attr_indices.len() {
            return Err(err("vertex attribute count exceeds fixed array"));
        }
        for ai in 0..attr_num {
            let location = usize::try_from(guest.attr_indices[ai]).map_err(|_| {
                err(format!(
                    "negative vertex attribute index {}",
                    guest.attr_indices[ai]
                ))
            })?;
            if location >= resources_num {
                return Err(err(format!(
                    "vertex attribute {location} exceeds resource count {resources_num}"
                )));
            }
            let attr = VertexAttributeData {
                location: location as u32,
                binding: binding as u32,
                format: gen5_vertex_format(info.resources[location].format())?,
                offset: guest.attr_offsets[ai],
            };
            // Coverage probe (XPS5X_TRACE_DRAWS). Data + gl_Position + draw
            // state are all confirmed correct yet coverage is ZERO, so the
            // remaining link is this binding. `location` must be the SAME index
            // space the shader declares (`OpDecorate %attr{i} Location {i}` over
            // 0..resources_num) and the format/offset must match the V#.
            if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static ATTR_SEEN: AtomicU32 = AtomicU32::new(0);
                if ATTR_SEEN.fetch_add(1, Ordering::Relaxed) < 12 {
                    tracing::warn!(
                        ai,
                        location = attr.location,
                        binding = attr.binding,
                        format = format_args!("{:?}", attr.format),
                        offset = attr.offset,
                        gen5_format = info.resources[location].format(),
                        resources_num,
                        "TRACE_DRAWS: vertex attribute binding"
                    );
                }
            }
            attributes.push(attr);
        }
    }
    if attributes.len() != resources_num {
        return Err(err(format!(
            "vertex metadata describes {} resources but {} bound attributes",
            resources_num,
            attributes.len()
        )));
    }
    Ok((buffers, attributes))
}

/// Decode one T# into linear pixels a Vulkan sampled image can hold.
///
/// Formats and tile modes are added strictly from measurement: an unhandled
/// value is a named error carrying every raw field, so a run against the title
/// states exactly what to implement next — a guessed format number would render
/// silently-wrong colours, which is worse than an honest skip.
fn decode_texture(
    t: &kyty_graphics::shader::ShaderTextureResource,
) -> Result<TextureUpload, DrawError> {
    let width = u32::from(t.width5()) + 1;
    let height = u32::from(t.height5()) + 1;
    if !(1..=16384).contains(&width) || !(1..=16384).contains(&height) {
        return Err(err(format!("texture extent {width}x{height} out of range")));
    }
    // 9 = Texture2D, 11 = Cube (measured: Minecraft's 1024x1024x6 skybox),
    // 10 = 3D volume (measured: ASTRO.BOT's 240x135x64 froxel/LUT volumes).
    let (cube, volume) = match t.type_() {
        // 8 = Texture1D. A 1D image is a 2D image one row tall, and the T#
        // already reports height5 = 0 => height 1, so the existing 2D decode
        // path handles it unchanged (measured on ASTRO.BOT: a 1x1 format-71
        // texture, tile mode 27). Kept a distinct arm rather than folding into
        // 9 so the disagreement is visible if a >1-row "1D" texture ever shows.
        8 => (false, false),
        9 => (false, false),
        10 => (false, true),
        11 => (true, false),
        other => {
            return Err(err(format!(
                "texture type {other} is not Texture2D (9), 3D (10) or Cube (11)"
            )));
        }
    };
    let layers = if cube { u32::from(t.depth()) + 1 } else { 1 };
    let depth = if volume { u32::from(t.depth()) + 1 } else { 1 };
    if volume && !(1..=2048).contains(&depth) {
        return Err(err(format!("volume depth {depth} out of range")));
    }

    // Gen5 unified T# format -> (Vulkan format, bytes per pixel), decoded via
    // SharpEmu's Gfx10UnifiedFormat table (the RDNA2 authority). Filled from
    // measured titles only; an unhandled value names itself rather than
    // guessing. `bpp` typed because the arms are added incrementally.
    let (format, bpp): (vk::Format, u32) = match t.format() {
        // 1 = single 8-bit channel, UNORM (measured on ASTRO.BOT's 480x270
        // coverage/mask texture, tile mode 27). SharpEmu's Gfx10UnifiedFormat
        // maps unified 1 -> (dataFormat 1 = FMT_8, numFormat 0 = UNORM).
        1 => (vk::Format::R8_UNORM, 1),
        // 36 = 10_11_11 FLOAT (packed 32-bit HDR) — the title samples its HDR
        // render target as a texture. SharpEmu Gfx10UnifiedFormat maps unified
        // 36 -> (dataFormat 6 = 10_11_11, numFormat 7 = FLOAT).
        36 => (vk::Format::B10G11R11_UFLOAT_PACK32, 4),
        // 0x0a = 8_8_8_8; channel type UNORM (measured on Minecraft's UI T#s).
        // NOTE: SharpEmu's table maps unified 10 -> (2,3) = 16_SSCALED, which
        // contradicts this arm. No 0x0a texture has appeared in a measured
        // run since; the first one that does must settle the table.
        0x0a => (vk::Format::R8G8B8A8_UNORM, 4),
        // 56 -> (10,0) = 8_8_8_8 UNORM (measured: Minecraft's 1920x1080 UI
        // texture, tile mode 27).
        56 => (vk::Format::R8G8B8A8_UNORM, 4),
        // 22 -> (4,7) = 32 FLOAT (measured: ASTRO.BOT's 1920x1080 R32F buffer
        // — a linear-depth/scalar target sampled back as a texture). SharpEmu
        // Gfx10UnifiedFormat.cs:48 maps unified 22 -> (dataFormat 4,
        // numFormat 7); dataFormat 4 is the single 32-bit channel per its Gen5
        // layout table ("SetLayout(4, 0, 0, 32); // 32") and numFormat 7 is
        // FLOAT, the same numFormat as the 36 and 71 arms.
        22 => (vk::Format::R32_SFLOAT, 4),
        // 71 -> (12,7) = 16_16_16_16 FLOAT (measured: ASTRO.BOT's 2432x1368
        // HDR scene buffer sampled back as a texture, tile mode 27). SharpEmu's
        // Gfx10UnifiedFormat maps unified 71 -> (dataFormat 12, numFormat 7);
        // dataFormat 12 is 16_16_16_16 per its Gen5 layout table, and numFormat
        // 7 is FLOAT (same numFormat as the 36 arm above). Every draw in the
        // title's 7966-dword DCB failed on this one format.
        71 => (vk::Format::R16G16B16A16_SFLOAT, 8),
        other => {
            return Err(err(format!(
                "texture format {other} not implemented \
                 (base={:#x} {width}x{height} pitch={} tile={} levels={})",
                t.base40(),
                t.pitch(),
                t.tile_mode(),
                t.last_level()
            )));
        }
    };

    let pixels = match t.tile_mode() {
        0 if cube => {
            return Err(err(
                "cube texture with linear tile mode not implemented (only tiled measured)",
            ));
        }
        0 => {
            // Linear: row-major at `pitch`, trimmed to tight rows below. A
            // volume is `depth` such slices back to back (slice pitch =
            // pitch * height for a linear T# — the measured ASTRO.BOT
            // volumes are tile 0).
            let pitch = u32::from(t.pitch()).max(width);
            let tiled = read_guest_bytes_unaligned(
                t.base40(),
                u64::from(pitch) * u64::from(height) * u64::from(depth) * u64::from(bpp),
                "texture",
            )?;
            let row = (width * bpp) as usize;
            let src_row = (pitch * bpp) as usize;
            let src_slice = src_row * height as usize;
            let dst_slice = row * height as usize;
            let mut pixels = vec![0u8; dst_slice * depth as usize];
            for z in 0..depth as usize {
                for y in 0..height as usize {
                    let src = z * src_slice + y * src_row;
                    let dst = z * dst_slice + y * row;
                    pixels[dst..dst + row].copy_from_slice(&tiled[src..src + row]);
                }
            }
            pixels
        }
        other if volume => {
            return Err(err(format!(
                "3D texture tile mode {other} not implemented (only linear measured; \
                 base={:#x} {width}x{height}x{depth} format={})",
                t.base40(),
                t.format()
            )));
        }
        // 64 KiB-block GFX10 swizzles with a ported exact equation
        // (SW_64KB_S = 9 measured on the 1937x333 atlas; SW_64KB_R_X = 27 on
        // the 1920x1080 UI texture). Fetch whole 64 KiB blocks — a tiled
        // surface owns its padding — and deswizzle, per array layer (a cube's
        // six faces are six block grids back to back).
        mode if crate::texture::tiling::swizzle_64kb_table(mode).is_some() => {
            let bpp_log2 = bpp.trailing_zeros();
            let face_tiled =
                crate::texture::tiling::tiled_byte_count_64kb(width, height, bpp_log2) as usize;
            let face_linear = (width * height * bpp) as usize;
            let tiled = read_guest_bytes_unaligned(
                t.base40(),
                face_tiled as u64 * u64::from(layers),
                "texture",
            )?;
            let mut pixels = vec![0u8; face_linear * layers as usize];
            for layer in 0..layers as usize {
                let src = &tiled[layer * face_tiled..(layer + 1) * face_tiled];
                let face = crate::texture::tiling::detile_64kb(mode, src, width, height, bpp_log2)
                    .expect("table-checked above");
                pixels[layer * face_linear..(layer + 1) * face_linear].copy_from_slice(&face);
            }
            pixels
        }
        other => {
            return Err(err(format!(
                "texture tile mode {other} not implemented \
                 (base={:#x} {width}x{height} format={})",
                t.base40(),
                t.format()
            )));
        }
    };
    // Content probe (XPS5X_TRACE_DRAWS): is the texture the PS samples
    // actually EMPTY? Thousands of title draws rasterize fine in-tree, and
    // most title draws alpha-blend (SRC_ALPHA/ONE_MINUS_SRC_ALPHA) — a PS
    // sampling an all-zero texture emits alpha 0, which is byte-identical to
    // "no coverage" in every frame probe. If these log all-zero, the GPU is
    // correctly rendering an EMPTY UI and the blocker is upstream (Gameface
    // never paints the menu), not in the GPU at all.
    if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static TEX_SEEN: AtomicU32 = AtomicU32::new(0);
        if TEX_SEEN.fetch_add(1, Ordering::Relaxed) < 12 {
            let nz = pixels.iter().filter(|&&b| b != 0).count();
            tracing::warn!(
                base = format_args!("{:#x}", t.base40()),
                extent = format_args!("{width}x{height}"),
                format = ?format,
                bytes = pixels.len(),
                non_zero = nz,
                "TRACE_DRAWS: sampled texture content"
            );
        }
    }
    Ok(TextureUpload {
        width,
        height,
        format,
        pixels,
        layers,
        cube,
        depth,
    })
}

/// Gen5 unified T# formats that are 32 bits per pixel — the only ones whose
/// guest bytes can seed an `R8G8B8A8_UNORM` storage image directly (the
/// recompiled SPIR-V declares every `%textures2D_L` entry as `Rgba8`).
fn storage_image_format_is_32bpp(format: u16) -> bool {
    // 0x0a/56 = 8_8_8_8, 22 = 32 FLOAT, 36 = 10_11_11 FLOAT — see
    // `decode_texture`'s table for the measurements behind each.
    matches!(format, 0x0a | 22 | 36 | 56)
}

/// Read one storage-image (UAV) T#'s extent and initial guest content.
///
/// The content is a best-effort seed: a UAV is typically fully overwritten by
/// the dispatch, so a T# whose format is not 32-bpp (RGBA8-sized) or whose
/// guest range is unreadable zero-fills with a once-per-process warning
/// instead of failing the dispatch.
fn read_storage_image(
    t: &kyty_graphics::shader::ShaderTextureResource,
) -> Result<StorageImageUpload, DrawError> {
    let width = u32::from(t.width5()) + 1;
    let height = u32::from(t.height5()) + 1;
    if !(1..=16384).contains(&width) || !(1..=16384).contains(&height) {
        return Err(err(format!(
            "storage image extent {width}x{height} out of range"
        )));
    }
    let size = u64::from(width) * u64::from(height) * 4;
    let base = t.base40();
    let readable = storage_image_format_is_32bpp(t.format());
    let pixels = if readable {
        // Linear read: UAV surfaces the title dispatches into are addressed
        // by the shader itself, so no de-tiling is applied to the seed.
        read_guest_bytes(base, size, "storage image").ok()
    } else {
        None
    };
    let pixels = pixels.unwrap_or_else(|| {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                base = format_args!("{base:#x}"),
                extent = format_args!("{width}x{height}"),
                format = t.format(),
                readable_as_32bpp = readable,
                "storage image initial content unavailable (non-32-bpp format \
                 or unreadable guest range) — zero-filling; the compute shader \
                 typically overwrites the whole UAV"
            );
        }
        vec![0u8; size as usize]
    });
    Ok(StorageImageUpload {
        width,
        height,
        pixels,
        guest_base: base,
    })
}

thread_local! {
    /// Raw pointer to the live render-target map for the draw currently being
    /// translated. `OffscreenDrawSink` sets it to its `framebuffers` immediately
    /// before `render_draw` and clears it right after, so the texture path can
    /// source a render-target-as-texture from its actual rendered pixels instead
    /// of the guest memory it was never written back to (that read black, which
    /// is why composited scenes stayed black). Null outside a draw.
    static RENDER_TARGETS: std::cell::Cell<*const HashMap<u64, RenderedImage>> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

/// Publish the live render-target map for the duration of `f` (one draw's
/// translation), then restore the previous value. See [`RENDER_TARGETS`].
pub fn with_render_targets<R>(map: &HashMap<u64, RenderedImage>, f: impl FnOnce() -> R) -> R {
    let prev = RENDER_TARGETS.with(|c| c.replace(std::ptr::from_ref(map)));
    let r = f();
    RENDER_TARGETS.with(|c| c.set(prev));
    r
}

/// The rendered pixels of the render target at guest `base` (matching `width` x
/// `height`), if one is live for the current draw. Used to sample a
/// render-target-as-texture — its content lives in the framebuffer map, not the
/// guest memory `decode_texture` reads.
fn render_target_pixels(base: u64, width: u32, height: u32) -> Option<Vec<u8>> {
    RENDER_TARGETS.with(|c| {
        let ptr = c.get();
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `with_render_targets` sets this to a live `&HashMap` for
        // exactly the synchronous, same-thread span of `render_draw` (which
        // drives this translation) and clears it after; the map is not mutated
        // during that span, and access here is read-only.
        let map = unsafe { &*ptr };
        map.get(&base)
            .filter(|img| img.width == width && img.height == height)
            .map(|img| img.pixels.clone())
    })
}

fn prepare_stage_binding(
    bind: &ShaderBindResources,
    stage: vk::ShaderStageFlags,
) -> Result<ShaderStageBinding, DrawError> {
    // Textures and samplers: decode every bound T#/S# and carry them to the
    // Vulkan layer. The push constants must carry the REWRITTEN descriptors
    // (base replaced by the descriptor-array index) in the exact
    // `shader_calc_binding_indices` order: storage V#s, then T#s (8 dwords),
    // then S#s (4 dwords), then direct SGPRs.
    let texture_num =
        usize::try_from(bind.textures2d.textures_num).map_err(|_| err("negative texture count"))?;
    let sampler_num =
        usize::try_from(bind.samplers.samplers_num).map_err(|_| err("negative sampler count"))?;
    if texture_num > bind.textures2d.desc.len() || sampler_num > bind.samplers.samplers.len() {
        return Err(err("texture/sampler count exceeds fixed array"));
    }
    if bind.textures2d.textures2d_storage_num != 0 && stage != vk::ShaderStageFlags::COMPUTE {
        return Err(err(format!(
            "translated {stage:?} shader uses {} STORAGE image(s) — storage images \
             are implemented for COMPUTE dispatches only",
            bind.textures2d.textures2d_storage_num
        )));
    }
    if bind.gds_pointers.pointers_num != 0 || bind.extended.used {
        return Err(err(format!(
            "translated {stage:?} shader needs unsupported GDS/EUD resources"
        )));
    }

    let storage_num = usize::try_from(bind.storage_buffers.buffers_num).map_err(|_| {
        err(format!(
            "negative storage-buffer count {}",
            bind.storage_buffers.buffers_num
        ))
    })?;
    if storage_num > bind.storage_buffers.buffers.len() {
        return Err(err("storage-buffer count exceeds fixed array"));
    }

    let mut push_constants = Vec::with_capacity(bind.push_constant_size as usize);
    let mut storage_bytes = Vec::with_capacity(storage_num);
    for (index, resource) in bind.storage_buffers.buffers[..storage_num]
        .iter()
        .enumerate()
    {
        if resource.add_tid() || resource.swizzle_enabled() || resource.out_of_bounds() != 0 {
            return Err(err(format!(
                "storage buffer {index} uses unsupported add-tid/swizzle/out-of-bounds mode"
            )));
        }
        let size = u64::from(resource.stride())
            .checked_mul(u64::from(resource.num_records()))
            .ok_or_else(|| err("storage buffer size overflow"))?;
        let bytes = read_guest_bytes(resource.base48(), size, "storage buffer")?;
        let all_zero = bytes.iter().all(|&b| b == 0);
        debug!(
            stage = ?stage,
            index,
            addr = format_args!("{:#x}", resource.base48()),
            len = bytes.len(),
            head = format_args!("{:02x?}", &bytes[..bytes.len().min(16)]),
            "stage storage buffer read"
        );
        if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEEN: AtomicU32 = AtomicU32::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                tracing::warn!(
                    stage = ?stage,
                    addr = format_args!("{:#x}", resource.base48()),
                    stride = resource.stride(),
                    records = resource.num_records(),
                    len = bytes.len(),
                    all_zero,
                    head = format_args!("{:02x?}", &bytes[..bytes.len().min(32)]),
                    "TRACE_DRAWS: storage buffer content"
                );
            }
        }
        storage_bytes.push(bytes);

        // Kyty rewrites the descriptor's guest base to the Vulkan descriptor
        // array index before exposing the four dwords as push constants.
        let mut rewritten = *resource;
        rewritten.update_address48(index as u64);
        for field in rewritten.fields {
            push_constants.extend_from_slice(&field.to_le_bytes());
        }
    }

    // T#s: decode + upload lists, and the rewritten 8-dword descriptor in the
    // push constants — the recompiled shader loads dword 0 at runtime as its
    // index into the %textures2D_S (sampled) or %textures2D_L (storage) array.
    // The push constants stay in analysis order over ALL T#s, but each
    // rewritten dword 0 is the index WITHIN its own descriptor array: sampled
    // T#s count 0..sampled_num and storage (usage == ReadWrite) T#s count
    // 0..storage_num, because the two SPIR-V arrays are separate bindings.
    let mut textures = Vec::with_capacity(texture_num);
    let mut storage_images = Vec::new();
    for (index, desc) in bind.textures2d.desc[..texture_num].iter().enumerate() {
        let mut rewritten = desc.texture;
        if desc.usage == ShaderTextureUsage::ReadWrite {
            let upload = read_storage_image(&desc.texture)?;
            if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static SEEN: AtomicU32 = AtomicU32::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                    tracing::warn!(
                        stage = ?stage,
                        index,
                        base = format_args!("{:#x}", upload.guest_base),
                        width = upload.width,
                        height = upload.height,
                        "TRACE_DRAWS: storage image bound"
                    );
                }
            }
            rewritten.update_address38(storage_images.len() as u64);
            storage_images.push(upload);
        } else {
            let mut decoded = decode_texture(&desc.texture)?;
            // If this T# points at a live render target, sample its actual
            // rendered pixels — they live in the framebuffer map, not the
            // guest memory decode_texture reads (render targets are never
            // written back). Without this, a title's final composite (a
            // fullscreen quad sampling its scene targets) reads black, so
            // nothing shows on screen.
            if let Some(px) =
                render_target_pixels(desc.texture.base40(), decoded.width, decoded.height)
            {
                decoded.pixels = px;
            }
            if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static SEEN: AtomicU32 = AtomicU32::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                    tracing::warn!(
                        stage = ?stage,
                        index,
                        base = format_args!("{:#x}", desc.texture.base40()),
                        width = decoded.width,
                        height = decoded.height,
                        vk_format = ?decoded.format,
                        "TRACE_DRAWS: texture decoded"
                    );
                }
            }
            rewritten.update_address38(textures.len() as u64);
            textures.push(decoded);
        }
        for field in rewritten.fields {
            push_constants.extend_from_slice(&field.to_le_bytes());
        }
    }
    if storage_images.len() != bind.textures2d.textures2d_storage_num as usize
        || textures.len() != bind.textures2d.textures2d_sampled_num as usize
    {
        return Err(err(format!(
            "translated {stage:?} T# usage split gives {} sampled + {} storage but the \
             analyzer counted {} + {}",
            textures.len(),
            storage_images.len(),
            bind.textures2d.textures2d_sampled_num,
            bind.textures2d.textures2d_storage_num
        )));
    }

    // S#s: only the mag-filter bit is honoured today; the rewritten descriptor
    // carries the sampler-array index in dword 0.
    let mut linear_filter = Vec::with_capacity(sampler_num);
    for (index, sampler) in bind.samplers.samplers[..sampler_num].iter().enumerate() {
        linear_filter.push(sampler.xy_mag_filter() != 0);
        let mut rewritten = *sampler;
        rewritten.update_index(index as u32);
        for field in rewritten.fields {
            push_constants.extend_from_slice(&field.to_le_bytes());
        }
    }

    let direct_num = usize::try_from(bind.direct_sgprs.sgprs_num).map_err(|_| {
        err(format!(
            "negative direct-SGPR count {}",
            bind.direct_sgprs.sgprs_num
        ))
    })?;
    if direct_num > bind.direct_sgprs.sgprs.len() {
        return Err(err("direct-SGPR count exceeds fixed array"));
    }
    for sgpr in &bind.direct_sgprs.sgprs[..direct_num] {
        push_constants.extend_from_slice(&sgpr.field.to_le_bytes());
    }
    if direct_num != 0 {
        let padded = push_constants.len().next_multiple_of(16);
        push_constants.resize(padded, 0);
    }

    if push_constants.len() != bind.push_constant_size as usize {
        return Err(err(format!(
            "translated {stage:?} push-constant ABI says {} bytes but preparation produced {}",
            bind.push_constant_size,
            push_constants.len()
        )));
    }

    Ok(ShaderStageBinding {
        stage,
        descriptor_set_slot: bind.descriptor_set_slot,
        push_constant_offset: bind.push_constant_offset,
        push_constants,
        storage_buffers: (storage_num != 0).then_some(StorageBufferBinding {
            binding: bind.storage_buffers.binding_index as u32,
            buffers: storage_bytes,
        }),
        textures: (!textures.is_empty()).then_some(TextureBinding {
            sampled_binding: bind.textures2d.binding_sampled_index as u32,
            sampler_binding: bind.samplers.binding_index as u32,
            textures,
            linear_filter,
        }),
        storage_images: (!storage_images.is_empty()).then_some(StorageImageBinding {
            binding: bind.textures2d.binding_storage_index as u32,
            images: storage_images,
        }),
    })
}

/// Build a [`DrawState`] from decoded register state.
///
/// Every field is register-derived. Returns a named [`DrawError`] rather than
/// substituting a default whenever the state cannot describe a real draw.
pub fn draw_state_from_regs<'a>(
    ctx: &Context,
    ucfg: &UserConfig,
    index_count: u32,
    vs_spirv: &'a [u32],
    fs_spirv: &'a [u32],
) -> Result<DrawState<'a>, DrawError> {
    let rt = &ctx.render_targets[0];

    if rt.base.addr == 0 {
        return Err(err(
            "no bound render target: CB_COLOR0_BASE is 0 (NoColorOutput)",
        ));
    }
    // A fully-disabled target is handled by the caller (`color_output_disabled`)
    // before this point; reaching here with 0 would silently draw nothing.
    if ctx.render_target_mask == 0 {
        return Err(err("CB_TARGET_MASK is 0 — colour output disabled"));
    }
    let color_write_mask = vulkan_color_write_mask(ctx.render_target_mask);

    // The PS5 extent lives in ATTRIB2 and stores width/height minus one.
    let width = rt.attrib2.width + 1;
    let height = rt.attrib2.height + 1;
    if rt.attrib2.width == 0 || rt.attrib2.height == 0 {
        return Err(err(format!(
            "CB_COLOR0_ATTRIB2 gives a degenerate extent {width}x{height} — \
             the render target extent was never programmed"
        )));
    }

    let format = vulkan_format(rt.info.format, rt.info.channel_type, rt.info.channel_order)?;

    // Kyty: CreatePipelineInternal — viewport from scale/offset.
    let vp = &ctx.screen_viewport.viewports[0];
    let viewport = [
        vp.xoffset - vp.xscale,
        vp.yoffset - vp.yscale,
        vp.xscale * 2.0,
        vp.yscale * 2.0,
    ];
    if viewport[2] == 0.0 || viewport[3] == 0.0 {
        return Err(err(
            "PA_CL_VPORT_XSCALE/YSCALE give a zero-area viewport — nothing would \
             rasterize and the frame would be silently empty",
        ));
    }

    // Kyty: viewport scissor -> generic scissor -> screen scissor.
    let sv = &ctx.screen_viewport;
    let scissor = if ctx.scan_mode_control.vport_scissor_enable
        && (vp.viewport_scissor_right > vp.viewport_scissor_left)
    {
        [
            vp.viewport_scissor_left,
            vp.viewport_scissor_top,
            vp.viewport_scissor_right,
            vp.viewport_scissor_bottom,
        ]
    } else if sv.generic_scissor_right > sv.generic_scissor_left {
        [
            sv.generic_scissor_left,
            sv.generic_scissor_top,
            sv.generic_scissor_right,
            sv.generic_scissor_bottom,
        ]
    } else {
        [
            sv.screen_scissor_left,
            sv.screen_scissor_top,
            sv.screen_scissor_right,
            sv.screen_scissor_bottom,
        ]
    };
    if scissor[2] <= scissor[0] || scissor[3] <= scissor[1] {
        return Err(err(format!(
            "scissor {scissor:?} is empty — nothing would rasterize"
        )));
    }

    // RectList is Kyty's clear primitive: the embedded VS emits a 4-vertex
    // strip quad from gl_VertexIndex, so the draw issues 4 despite index_count
    // being 3. That mismatch is Kyty's real behaviour, not a bug.
    let (topology, vertex_count) = match ucfg.prim_type {
        prim::RECT_LIST => (vk::PrimitiveTopology::TRIANGLE_STRIP, 4),
        prim::TRIANGLE_LIST => (vk::PrimitiveTopology::TRIANGLE_LIST, index_count),
        prim::TRIANGLE_FAN | prim::POLYGON => (vk::PrimitiveTopology::TRIANGLE_FAN, index_count),
        prim::TRIANGLE_STRIP => (vk::PrimitiveTopology::TRIANGLE_STRIP, index_count),
        other => {
            return Err(err(format!(
                "unsupported VGT_PRIMITIVE_TYPE {other} (supported: 4 TriList, \
                 5 TriFan, 6 TriStrip, 7 Polygon, 17 RectList)"
            )));
        }
    };

    let mut cull_mode = vk::CullModeFlags::NONE;
    if ctx.mode_control.cull_front {
        cull_mode |= vk::CullModeFlags::FRONT;
    }
    if ctx.mode_control.cull_back {
        cull_mode |= vk::CullModeFlags::BACK;
    }
    // Diagnostic (XPS5X_NO_CULL=1): disable culling entirely. Measured on
    // Minecraft: its draws run cull=FRONT face=CLOCKWISE, and under our
    // Y-flipped viewport the measured quad rasterizes clockwise — a FRONT
    // face — so every primitive is culled. The full title VS+PS replayed
    // in-tree with cull NONE covers 4096/4096 (tests/coverage_bisect.rs), so
    // culling is the LAST field separating the in-tree render from the
    // title's black frame. This switch is the yes/no for that mechanism.
    if std::env::var_os("XPS5X_NO_CULL").is_some() {
        cull_mode = vk::CullModeFlags::NONE;
    }
    // PA_SU_SC_MODE_CNTL.FACE: 0 = counter-clockwise is the front face,
    // 1 = clockwise is. Must travel with cull_mode — culling against the wrong
    // winding removes exactly the geometry it should keep.
    let front_face = if ctx.mode_control.face {
        vk::FrontFace::CLOCKWISE
    } else {
        vk::FrontFace::COUNTER_CLOCKWISE
    };

    let blend = blend_state_from_regs(ctx)?;

    // Why-is-it-black diagnostic. Every GEOMETRIC degeneracy above (zero target
    // mask, degenerate extent, zero-area viewport, empty scissor) is already a
    // loud error, and titles hit none of them — yet Minecraft's targets stay
    // byte-exactly zero across ~11,878 draws that reach here. So the cause is
    // one of: coverage (viewport/scissor vs extent), blend collapsing the
    // result, or the write mask. Log them together, rate-limited.
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static DRAW_STATE_LOGS: AtomicU64 = AtomicU64::new(0);
        let n = DRAW_STATE_LOGS.fetch_add(1, Ordering::Relaxed);
        if n < 8 || n.is_power_of_two() {
            debug!(
                n,
                extent = format_args!("{width}x{height}"),
                rt_base = format_args!("{:#x}", rt.base.addr),
                viewport = format_args!("{viewport:?}"),
                scissor = format_args!("{scissor:?}"),
                target_mask = format_args!("{:#x}", ctx.render_target_mask),
                color_write_mask = format_args!("{color_write_mask:?}"),
                blend = format_args!("{blend:?}"),
                topology = format_args!("{topology:?}"),
                // Cull was NEVER logged here, and the in-tree replay (which
                // covers 4096/4096 with the title's own VS+PS and both zero
                // and non-zero resources) uses CullModeFlags::NONE. With the
                // Y-flipped viewport the effective winding inverts, so a
                // decoded cull_back + wrong FACE culls EVERY primitive —
                // silently and validation-clean. The last unreplicated field.
                cull = format_args!("{cull_mode:?}"),
                face = format_args!("{front_face:?}"),
                vertex_count,
                "draw state (black-frame diagnostic)"
            );
        }
    }

    Ok(DrawState {
        width,
        height,
        format,
        clear_color: CLEAR_COLOR,
        scissor,
        viewport,
        topology,
        cull_mode,
        front_face,
        color_write_mask,
        blend,
        // The embedded VS declares no input attributes and builds its own quad.
        vertices: None,
        vertex_buffers: Vec::new(),
        vertex_attributes: Vec::new(),
        stage_bindings: Vec::new(),
        vertex_count,
        vs_spirv,
        fs_spirv,
        initial: None,
        // The caller (draw_common) fills this in for an indexed draw.
        index: None,
        // This register-driven composite path is colour-only for now; the
        // depth/stencil attachment is wired in `render_draw` but not yet fed
        // from the PM4 DB_* registers (a future `depth_state_from_regs`).
        color_output: true,
        depth: None,
    })
}

/// A [`DrawSink`] that renders each draw offscreen and keeps the last image.
///
/// # Indexed-draw degradation (documented, deliberate)
///
/// This sink does not override [`DrawSink::draw_index`], so an indexed draw
/// takes the trait's default degradation: the index buffer is **not**
/// fetched, and the draw runs through the same register-driven path as an
/// auto draw with `index_count` vertices. Right vertex *count*, wrong vertex
/// *order* for anything but sequential indices — enough for first light, and
/// the command processor logs the degradation (rate-limited). A real indexed
/// path needs the index-buffer fetch from Kyty's `GraphicsRender` (Phase 2).
pub struct OffscreenDrawSink<'a> {
    dev: &'a VulkanDevice,
    cache: &'a mut ShaderTranslateCache,
    /// Persistent per-render-target contents, keyed by `CB_COLOR0_BASE`.
    /// Each draw seeds its attachment with the target's prior pixels and
    /// stores the result back, so draws compose into a frame instead of each
    /// one starting from a cleared target.
    framebuffers: &'a mut HashMap<u64, RenderedImage>,
    pub last: Option<RenderedImage>,
    pub draws: u64,
    /// Draws and compute dispatches skipped because a bound guest shader failed
    /// translation. The named reason was warned once by the cache; each skip
    /// here is quiet (debug) so 1600 re-binds of one bad shader stay one loud
    /// line. This is the SUM of `draw_skips` + `dispatch_skips`.
    pub shader_skips: u64,
    /// Skips attributable to the graphics-draw path (`draw_index_auto`).
    pub draw_skips: u64,
    /// Skips attributable to the compute-dispatch path (`dispatch_direct`).
    pub dispatch_skips: u64,
    /// Compute dispatches completed and written back to guest storage.
    pub dispatches: u64,
    /// Most recent named skip reason from the DRAW path, surfaced by the
    /// session with a process-wide rate limit. Kept separate from the compute
    /// reason: a title issues far more dispatches than draws, so a single
    /// shared field almost always reports a compute failure and silently
    /// masks why a *draw* skipped.
    pub last_draw_skip_reason: Option<String>,
    /// Most recent named skip reason from the compute-DISPATCH path.
    pub last_dispatch_skip_reason: Option<String>,
    /// True when this sink is draining the ACB (async-compute) queue rather than
    /// the graphics DCB. Diagnostic only — surfaced in the dispatch skip trace so
    /// a zeroed-shader dispatch can be attributed to the right queue.
    pub queue_is_compute: bool,
    /// The last compute shader bound on EITHER queue, carried across submissions.
    /// The title binds its compute shader on the graphics DCB and dispatches it
    /// on the async-compute ACB, whose submissions are dispatch-only (no bind);
    /// a dispatch that arrives with a null `cs` falls back to this. Seeded from
    /// the session before a submission runs and read back after, so it persists
    /// across the per-submission sink lifetime.
    pub current_compute: Option<ComputeShaderInfo>,
}

impl<'a> OffscreenDrawSink<'a> {
    #[must_use]
    pub fn new(
        dev: &'a VulkanDevice,
        cache: &'a mut ShaderTranslateCache,
        framebuffers: &'a mut HashMap<u64, RenderedImage>,
    ) -> Self {
        Self {
            dev,
            cache,
            framebuffers,
            last: None,
            draws: 0,
            shader_skips: 0,
            draw_skips: 0,
            dispatch_skips: 0,
            dispatches: 0,
            last_draw_skip_reason: None,
            last_dispatch_skip_reason: None,
            queue_is_compute: false,
            current_compute: None,
        }
    }
}

impl OffscreenDrawSink<'_> {
    /// Choose the compute shader for a dispatch, applying cross-queue seeding.
    ///
    /// The title binds its compute shader on the graphics DCB and dispatches it
    /// on the asynchronous-compute ACB ring, whose command buffers are
    /// dispatch-only — so an ACB dispatch reaches us with a null `cs`. A dispatch
    /// that carries a real shader (`data_addr != 0`) is used as-is and recorded
    /// in `current`; a dispatch with a null shader falls back to `current` (the
    /// last shader bound on either queue). With nothing recorded yet, the null
    /// shader passes through unchanged and the dispatch skips as before.
    fn seed_compute(
        sh_cs: &ComputeShaderInfo,
        current: &mut Option<ComputeShaderInfo>,
    ) -> ComputeShaderInfo {
        if sh_cs.cs_regs.data_addr != 0 {
            *current = Some(*sh_cs);
            *sh_cs
        } else if let Some(seeded) = *current {
            seeded
        } else {
            *sh_cs
        }
    }

    /// The body shared by the auto and indexed draw paths.
    ///
    /// `index` is `None` for a vertex-order draw and `Some((bytes, type))` for
    /// an indexed one — the only difference between the two is whether an index
    /// buffer is bound. `count` is the vertex count (auto) or index count
    /// (indexed); either way it is what the draw call is told to draw.
    fn draw_common(
        &mut self,
        ctx: &Context,
        ucfg: &UserConfig,
        sh: &Shader,
        count: u32,
        index: Option<(&[u8], vk::IndexType)>,
    ) -> Result<(), DrawError> {
        // A zero colour mask is a legitimate depth-only/no-colour draw, not a
        // malformed DCB. Until the depth backend is wired, consume it without
        // aborting later colour draws in the same submission.
        if color_output_disabled(ctx) {
            debug!("draw consumed without colour output (depth path pending)");
            return Ok(());
        }
        // VGT_PRIMITIVE_TYPE 0 (NONE) draws nothing on hardware — the packet
        // is a state carrier, not a malformed draw (measured: Minecraft issues
        // one per DCB preamble). Consume it quietly rather than failing the
        // draw pipeline creation.
        if ucfg.prim_type == prim::NONE {
            debug!("draw consumed: VGT_PRIMITIVE_TYPE NONE");
            return Ok(());
        }
        let shaders = if sh.vs.vs_embedded && sh.ps.ps_embedded {
            // The embedded pair is the Phase 1 / M2 invariant: a failure here
            // is a broken fixture and must abort loudly.
            resolve_shaders(self.cache, ctx, sh)?
        } else {
            match resolve_shaders(self.cache, ctx, sh) {
                Ok(s) => s,
                Err(e) => {
                    // Named degradation: skip this draw, keep the DCB going.
                    self.shader_skips += 1;
                    self.draw_skips += 1;
                    self.last_draw_skip_reason = Some(e.to_string());
                    debug!(reason = %e, "draw skipped: bound guest shader is untranslatable");
                    return Ok(());
                }
            }
        };

        let mut state = draw_state_from_regs(ctx, ucfg, count, &shaders.vs, &shaders.ps)?;
        let (vertex_buffers, vertex_attributes) = prepare_vertex_inputs(&shaders.vs_info)?;
        state.vertex_buffers = vertex_buffers;
        state.vertex_attributes = vertex_attributes;
        // Coverage probe (XPS5X_TRACE_DRAWS). Vertex data, attribute bindings,
        // gl_Position and draw state are all confirmed correct yet coverage is
        // ZERO. An all-zero index buffer collapses every primitive to a single
        // vertex — degenerate triangles cover no pixel — and would look exactly
        // like this. Report whether the draw is indexed and what the indices are.
        if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static IDX_SEEN: AtomicU32 = AtomicU32::new(0);
            if IDX_SEEN.fetch_add(1, Ordering::Relaxed) < 12 {
                match index {
                    Some((bytes, index_type)) => tracing::warn!(
                        indexed = true,
                        index_type = format_args!("{index_type:?}"),
                        len = bytes.len(),
                        count,
                        non_zero = bytes.iter().filter(|&&b| b != 0).count(),
                        head = format_args!("{:02x?}", &bytes[..bytes.len().min(24)]),
                        "TRACE_DRAWS: index buffer"
                    ),
                    None => tracing::warn!(indexed = false, count, "TRACE_DRAWS: non-indexed draw"),
                }
            }
        }
        if let Some((bytes, index_type)) = index {
            state.index = Some(crate::vulkan::IndexBinding { bytes, index_type });
        }
        for (bind, stage) in [
            (&shaders.vs_info.bind, vk::ShaderStageFlags::VERTEX),
            (&shaders.ps_info.bind, vk::ShaderStageFlags::FRAGMENT),
        ] {
            if bind.push_constant_size != 0
                || bind.storage_buffers.buffers_num != 0
                || bind.textures2d.textures_num != 0
                || bind.samplers.samplers_num != 0
                || bind.gds_pointers.pointers_num != 0
                || bind.direct_sgprs.sgprs_num != 0
                || bind.extended.used
            {
                state
                    .stage_bindings
                    .push(prepare_stage_binding(bind, stage)?);
            }
        }

        // One-shot forensic: what does a real Minecraft draw actually bind?
        // CORRECTED 2026-07-20: this used to claim "geometry covers the screen
        // and the PS shades black". That is REFUTED — with XPS5X_FORCE_CLEAR=1
        // (see offscreen.rs) Minecraft's final frame is 100% uniform
        // CLEAR_COLOR, so NOT ONE draw produces a fragment. There is no
        // coverage, and the PS is therefore never invoked; chasing the PS
        // inputs from here is the wrong wall. What this trace is still good
        // for: it shows the draws ARE well-formed (measured: textured/storage
        // -fed quads, prim=4 verts=6 and prim=6 verts=4, 1 vertex buffer,
        // 1-2 attributes), which is what pins the failure on VERTEX POSITIONS.
        // Gated to the first few draws (XPS5X_TRACE_DRAWS) so it never floods.
        if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEEN: AtomicU32 = AtomicU32::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 12 {
                let ps = &shaders.ps_info.bind;
                let vs = &shaders.vs_info.bind;
                tracing::warn!(
                    prim = ucfg.prim_type,
                    verts = state.vertex_count,
                    guest_vbufs = state.vertex_buffers.len(),
                    vattrs = state.vertex_attributes.len(),
                    ps_tex = ps.textures2d.textures_num,
                    ps_samp = ps.samplers.samplers_num,
                    ps_sbuf = ps.storage_buffers.buffers_num,
                    ps_pushc = ps.push_constant_size,
                    ps_ext = ps.extended.used,
                    vs_tex = vs.textures2d.textures_num,
                    vs_sbuf = vs.storage_buffers.buffers_num,
                    vs_pushc = vs.push_constant_size,
                    "TRACE_DRAWS: real draw bind profile"
                );
            }
        }

        // Compose into the guest render target: seed with its prior pixels
        // (taken from the framebuffer map) so this draw adds to the frame
        // instead of starting over on a cleared attachment.
        let rt_base = ctx.render_targets[0].base.addr;
        let prior = self
            .framebuffers
            .remove(&rt_base)
            .filter(|p| p.width == state.width && p.height == state.height);
        let image = {
            if let Some(p) = &prior {
                state.initial = Some(&p.pixels);
            }
            // Publish the other live render targets (this one was `remove`d
            // above) so this draw's texture path can sample any of them as a
            // render-target-as-texture — the composite that produces the visible
            // frame samples its scene targets this way.
            let dev = self.dev;
            let fbs: &HashMap<u64, RenderedImage> = self.framebuffers;
            let output = with_render_targets(fbs, || {
                render_draw(dev, &state).map_err(|e| err(format!("offscreen draw failed: {e}")))
            })?;
            // This path is colour-only (`color_output: true`, `depth: None`),
            // so the draw always produces a colour image; a depth-only draw
            // would land here only once the register decode emits one.
            output
                .color
                .ok_or_else(|| err("offscreen draw produced no colour image".to_string()))?
        };

        self.framebuffers.insert(rt_base, image.clone());
        self.last = Some(image);
        self.draws += 1;
        Ok(())
    }
}

impl DrawSink for OffscreenDrawSink<'_> {
    fn draw_index_auto(
        &mut self,
        ctx: &Context,
        ucfg: &UserConfig,
        sh: &Shader,
        index_count: u32,
        _flags: u32,
    ) -> Result<(), DrawError> {
        self.draw_common(ctx, ucfg, sh, index_count, None)
    }

    /// A real indexed draw — the vertices are pulled through the bound index
    /// buffer instead of straight from the vertex stream.
    ///
    /// Without this the trait default runs, which throws the index buffer away
    /// and issues a vertex-order auto draw. That is why Minecraft's fullscreen
    /// tri-strip QUAD (`prim=6 verts=4`, two triangles sharing an edge) rendered
    /// as one triangle covering exactly half the target: the four vertices came
    /// out in submission order rather than index order.
    fn draw_index(
        &mut self,
        ctx: &Context,
        ucfg: &UserConfig,
        sh: &Shader,
        draw: &IndexedDraw,
    ) -> Result<(), DrawError> {
        let (index_bytes, index_type) = fetch_index_buffer(draw)?;
        self.draw_common(
            ctx,
            ucfg,
            sh,
            draw.index_count,
            Some((&index_bytes, index_type)),
        )
    }

    fn dispatch_direct(
        &mut self,
        ctx: &Context,
        _ucfg: &UserConfig,
        sh: &Shader,
        groups: [u32; 3],
        mode: u32,
    ) -> Result<(), DrawError> {
        // The legacy Kyty AGC wrapper emits 0. Retail RDNA2 command streams
        // also carry COMPUTE_SHADER_EN (bit 0) and CS_W32_EN (bit 6), yielding
        // the measured 0x41; ASTRO.BOT additionally sets USE_THREAD_DIMENSIONS
        // (bit 5, 0x20) for 0x61. All three describe execution already
        // represented by the translated Vulkan compute stage — the group counts
        // carry the thread dimensions bit 5 selects — so accept them; other
        // initiator bits need explicit semantics before they can be accepted.
        if mode & !0x61 != 0 {
            return Err(err(format!(
                "unsupported compute dispatch initiator {mode:#x}"
            )));
        }
        // Cross-queue compute-shader seeding (see `seed_compute`): substitute the
        // last shader bound on either queue when this dispatch carries none, so
        // dispatch-only ACB buffers translate against the shader the title bound
        // on the DCB instead of skipping on a null address.
        let cs = Self::seed_compute(&sh.cs, &mut self.current_compute);
        let translated = match self.cache.translate_cs(&cs, &ctx.sh_regs) {
            Ok(shader) => shader,
            Err(error) => {
                self.shader_skips += 1;
                self.dispatch_skips += 1;
                self.last_dispatch_skip_reason = Some(error.to_string());
                let r = &cs.cs_regs;
                debug!(
                    reason = %error,
                    queue = if self.queue_is_compute { "ACB" } else { "DCB" },
                    cs_addr = format_args!("{:#x}", r.data_addr),
                    groups = format_args!("{}x{}x{}", groups[0], groups[1], groups[2]),
                    threads = format_args!("{}x{}x{}", r.num_thread_x, r.num_thread_y, r.num_thread_z),
                    user_sgpr = r.user_sgpr,
                    mode = format_args!("{mode:#x}"),
                    "compute dispatch skipped: bound shader is untranslatable"
                );
                return Ok(());
            }
        };
        let bind = &translated.cs_info.bind;
        let has_binding = bind.push_constant_size != 0
            || bind.storage_buffers.buffers_num != 0
            || bind.textures2d.textures_num != 0
            || bind.samplers.samplers_num != 0
            || bind.gds_pointers.pointers_num != 0
            || bind.direct_sgprs.sgprs_num != 0
            || bind.extended.used;
        let prepared = has_binding
            .then(|| prepare_stage_binding(bind, vk::ShaderStageFlags::COMPUTE))
            .transpose()?;
        let storage_num = usize::try_from(bind.storage_buffers.buffers_num)
            .map_err(|_| err("negative compute storage-buffer count"))?;
        if storage_num > bind.storage_buffers.buffers.len() {
            return Err(err("compute storage-buffer count exceeds fixed array"));
        }
        let guest_outputs: Vec<_> = bind.storage_buffers.buffers[..storage_num]
            .iter()
            .map(|resource| resource.base48())
            .collect();
        // Storage-image guest bases, collected pre-dispatch in the same order
        // `ComputeOutputs::images` returns them.
        let guest_image_outputs: Vec<u64> = prepared
            .as_ref()
            .and_then(|binding| binding.storage_images.as_ref())
            .map(|images| images.images.iter().map(|img| img.guest_base).collect())
            .unwrap_or_default();
        let outputs = dispatch_compute(
            self.dev,
            &ComputeState {
                groups,
                spirv: &translated.spirv,
                binding: prepared.as_ref(),
            },
        )
        .map_err(|error| err(format!("Vulkan compute dispatch failed: {error}")))?;
        if outputs.buffers.len() != guest_outputs.len() {
            return Err(err(format!(
                "compute writeback returned {} buffers for {} guest outputs",
                outputs.buffers.len(),
                guest_outputs.len()
            )));
        }
        if outputs.images.len() != guest_image_outputs.len() {
            return Err(err(format!(
                "compute writeback returned {} images for {} guest image outputs",
                outputs.images.len(),
                guest_image_outputs.len()
            )));
        }
        for (addr, bytes) in guest_outputs.into_iter().zip(outputs.buffers) {
            debug!(
                addr = format_args!("{addr:#x}"),
                len = bytes.len(),
                head = format_args!("{:02x?}", &bytes[..bytes.len().min(16)]),
                "compute storage writeback"
            );
            if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
                let nonzero = bytes.iter().any(|&b| b != 0);
                tracing::warn!(
                    addr = format_args!("{addr:#x}"),
                    len = bytes.len(),
                    nonzero,
                    "TRACE_DRAWS: compute writeback"
                );
            }
            if !crate::guest_mem::write_bytes_checked(addr, &bytes) {
                return Err(err(format!(
                    "compute storage writeback range {addr:#x}..{:#x} is not writable guest memory",
                    addr.saturating_add(bytes.len() as u64)
                )));
            }
        }
        for (addr, bytes) in guest_image_outputs.into_iter().zip(outputs.images) {
            let nonzero = bytes.iter().any(|&b| b != 0);
            debug!(
                addr = format_args!("{addr:#x}"),
                len = bytes.len(),
                nonzero,
                "compute storage-image writeback"
            );
            if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
                tracing::warn!(
                    addr = format_args!("{addr:#x}"),
                    len = bytes.len(),
                    nonzero,
                    "TRACE_DRAWS: compute image writeback"
                );
            }
            if !crate::guest_mem::write_bytes_checked(addr, &bytes) {
                return Err(err(format!(
                    "compute storage-image writeback range {addr:#x}..{:#x} is not writable \
                     guest memory",
                    addr.saturating_add(bytes.len() as u64)
                )));
            }
        }
        self.dispatches += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kyty_graphics::hw_regs::{ColorAttrib2, ComputeShaderInfo};

    /// Cross-queue compute-shader seeding: a dispatch-only ACB dispatch (null
    /// shader) must fall back to the last shader bound on either queue, so the
    /// title's compute work — bound on the DCB, dispatched on the ACB — reaches
    /// translation instead of skipping on a null address. Recovered ASTRO.BOT's
    /// zeroed ACB dispatches (measured 516 → 9).
    #[test]
    fn cross_queue_compute_seeding_falls_back_to_last_bound() {
        let cs = |addr: u64| {
            let mut c = ComputeShaderInfo::default();
            c.cs_regs.data_addr = addr;
            c
        };
        let mut current = None;

        // A bound (non-null) dispatch is used as-is and recorded.
        let chosen = OffscreenDrawSink::seed_compute(&cs(0x500a00), &mut current);
        assert_eq!(chosen.cs_regs.data_addr, 0x500a00);
        assert_eq!(current.unwrap().cs_regs.data_addr, 0x500a00);

        // A null (dispatch-only ACB) dispatch falls back to the recorded shader.
        let chosen = OffscreenDrawSink::seed_compute(&cs(0), &mut current);
        assert_eq!(
            chosen.cs_regs.data_addr, 0x500a00,
            "null dispatch must reuse the last bound compute shader"
        );

        // A later bind updates what a subsequent null dispatch falls back to.
        OffscreenDrawSink::seed_compute(&cs(0x500b00), &mut current);
        let chosen = OffscreenDrawSink::seed_compute(&cs(0), &mut current);
        assert_eq!(chosen.cs_regs.data_addr, 0x500b00);

        // With nothing recorded, a null dispatch stays null (skips as before).
        let mut empty = None;
        let chosen = OffscreenDrawSink::seed_compute(&cs(0), &mut empty);
        assert_eq!(chosen.cs_regs.data_addr, 0);
        assert!(empty.is_none());
    }

    /// Registers describing a valid 96x48 RGBA target, left-half scissor.
    fn ctx_96x48() -> Context {
        let mut ctx = Context::default();
        let rt = &mut ctx.render_targets[0];
        rt.base.addr = 0x1_0000;
        rt.info.format = 0xa;
        rt.attrib2 = ColorAttrib2 {
            width: 95,
            height: 47,
            num_mip_levels: 0,
        };
        ctx.render_target_mask = 0xF;
        let vp = &mut ctx.screen_viewport.viewports[0];
        vp.xscale = 48.0;
        vp.xoffset = 48.0;
        vp.yscale = 24.0;
        vp.yoffset = 24.0;
        ctx.screen_viewport.screen_scissor_right = 48;
        ctx.screen_viewport.screen_scissor_bottom = 48;
        ctx
    }

    fn ucfg_rect() -> UserConfig {
        UserConfig {
            prim_type: prim::RECT_LIST,
            ..UserConfig::default()
        }
    }

    const SPIRV: &[u32] = &[0x0723_0203];

    #[test]
    fn attrib2_drives_extent_not_m2_constants() {
        let state = draw_state_from_regs(&ctx_96x48(), &ucfg_rect(), 3, SPIRV, SPIRV)
            .expect("valid register state");
        assert_eq!((state.width, state.height), (96, 48));
        assert_ne!(
            (state.width, state.height),
            (
                crate::agc_exec::M2_DRAW_WIDTH,
                crate::agc_exec::M2_DRAW_HEIGHT
            ),
            "the extent must come from ATTRIB2, not the fixture constants"
        );
    }

    #[test]
    fn viewport_derives_from_scale_and_offset() {
        let state =
            draw_state_from_regs(&ctx_96x48(), &ucfg_rect(), 3, SPIRV, SPIRV).expect("valid");
        // x = xoffset - xscale, w = xscale * 2
        assert_eq!(state.viewport, [0.0, 0.0, 96.0, 48.0]);
    }

    /// 16-bit indices — the common case — are read straight from guest memory
    /// and bound as UINT16.
    #[test]
    fn fetch_index_buffer_reads_16bit_indices_verbatim() {
        // A quad as two triangles. Heap-backed: read_guest_bytes VirtualQuery-
        // validates the pointer, and a live Vec is committed-readable.
        let indices: Vec<u16> = vec![0, 1, 2, 2, 1, 3];
        let draw = IndexedDraw {
            index_type_and_size: 0,
            index_count: 6,
            index_addr: indices.as_ptr() as u64,
            flags: 0,
            index_type: 1,
        };
        let (bytes, ty) = crate::guest_mem::with_test_ranges(
            &[(
                indices.as_ptr() as u64,
                std::mem::size_of_val(indices.as_slice()),
            )],
            || fetch_index_buffer(&draw),
        )
        .expect("readable index buffer");
        assert_eq!(ty, vk::IndexType::UINT16);
        assert_eq!(&bytes[..12], bytemuck_le(&indices).as_slice());
    }

    /// 8-bit indices are widened to 16-bit — Vulkan has no guaranteed UINT8.
    #[test]
    fn fetch_index_buffer_widens_8bit_to_16bit() {
        let indices: Vec<u8> = vec![0, 1, 2, 3];
        let draw = IndexedDraw {
            index_type_and_size: 2,
            index_count: 4,
            index_addr: indices.as_ptr() as u64,
            flags: 0,
            index_type: 1,
        };
        let (bytes, ty) = crate::guest_mem::with_test_ranges(
            &[(
                indices.as_ptr() as u64,
                std::mem::size_of_val(indices.as_slice()),
            )],
            || fetch_index_buffer(&draw),
        )
        .expect("readable index buffer");
        assert_eq!(ty, vk::IndexType::UINT16);
        // Each u8 becomes a little-endian u16: 0,1,2,3 -> 00 00 01 00 02 00 03 00.
        assert_eq!(bytes, vec![0, 0, 1, 0, 2, 0, 3, 0]);
    }

    /// Index buffers start at `base + offset*element`, which routinely lands
    /// off a dword boundary — the earlier version rejected that as unreadable
    /// and Minecraft's draws died on it.
    #[test]
    fn fetch_index_buffer_reads_from_an_unaligned_address() {
        // 8 u16s in a heap buffer; point the draw two bytes in (index 1).
        let backing: Vec<u16> = vec![99, 10, 11, 12, 13, 14, 15, 16];
        let unaligned = backing.as_ptr() as u64 + 2;
        assert_ne!(unaligned & 0x3, 0, "the test address must be unaligned");
        let draw = IndexedDraw {
            index_type_and_size: 0,
            index_count: 4,
            index_addr: unaligned,
            flags: 0,
            index_type: 1,
        };
        let (bytes, _) = crate::guest_mem::with_test_ranges(
            &[(
                backing.as_ptr() as u64,
                std::mem::size_of_val(backing.as_slice()),
            )],
            || fetch_index_buffer(&draw),
        )
        .expect("unaligned read must work");
        assert_eq!(bytes, bytemuck_le(&[10u16, 11, 12, 13]));
    }

    /// A null address or zero count is a malformed indexed draw, not a hang.
    #[test]
    fn fetch_index_buffer_rejects_an_empty_draw() {
        let draw = IndexedDraw {
            index_type_and_size: 0,
            index_count: 0,
            index_addr: 0x1000,
            flags: 0,
            index_type: 1,
        };
        assert!(fetch_index_buffer(&draw).is_err());
    }

    /// Little-endian bytes of a u16 slice, for comparing against fetched data.
    fn bytemuck_le(indices: &[u16]) -> Vec<u8> {
        indices.iter().flat_map(|i| i.to_le_bytes()).collect()
    }

    #[test]
    fn screen_scissor_register_reaches_the_draw_state() {
        let state =
            draw_state_from_regs(&ctx_96x48(), &ucfg_rect(), 3, SPIRV, SPIRV).expect("valid");
        assert_eq!(state.scissor, [0, 0, 48, 48]);
    }

    #[test]
    fn unbound_render_target_is_a_named_error() {
        let mut ctx = ctx_96x48();
        ctx.render_targets[0].base.addr = 0;
        let e = draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV).expect_err("no RT");
        assert!(e.0.contains("render target"), "got {e}");
    }

    /// A partial `CB_TARGET_MASK` is an ordinary draw, not a broken DCB.
    ///
    /// This used to be `partial_render_target_mask_is_a_named_error` and
    /// asserted the opposite. That rejection was an `Err` out of
    /// `draw_state_from_regs`, which propagates through `run?` in
    /// `execute_dcb_cp` and abandons every remaining draw in the command
    /// buffer — so a mask of 0x7 (RGB, alpha untouched, which Minecraft issues)
    /// destroyed a whole DCB. Vulkan expresses the mask natively; it maps
    /// straight through.
    #[test]
    fn partial_render_target_mask_maps_to_vulkans_write_mask() {
        let mut ctx = ctx_96x48();
        ctx.render_target_mask = 0x7;
        let state = draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV)
            .expect("a partial mask is a normal draw");
        assert_eq!(
            state.color_write_mask,
            vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B,
            "0x7 is RGB with alpha writes disabled"
        );
    }

    /// Bit order is R,G,B,A from bit 0 — getting it reversed would silently
    /// write the wrong channels rather than fail.
    #[test]
    fn target_mask_bits_map_to_the_right_channels() {
        for (mask, expected) in [
            (0xF, vk::ColorComponentFlags::RGBA),
            (0x1, vk::ColorComponentFlags::R),
            (0x2, vk::ColorComponentFlags::G),
            (0x4, vk::ColorComponentFlags::B),
            (0x8, vk::ColorComponentFlags::A),
            (0x9, vk::ColorComponentFlags::R | vk::ColorComponentFlags::A),
        ] {
            assert_eq!(
                vulkan_color_write_mask(mask),
                expected,
                "CB_TARGET_MASK {mask:#x}"
            );
        }
    }

    #[test]
    fn zero_target_mask_is_a_colorless_draw_policy_not_a_broken_dcb() {
        let mut ctx = ctx_96x48();
        ctx.render_target_mask = 0;
        assert!(color_output_disabled(&ctx));
        assert!(
            draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV).is_err(),
            "the colour renderer must still reject it if called directly"
        );
    }

    /// A zero viewport rasterizes nothing and reports no error anywhere in
    /// Vulkan — the likeliest way a structurally-correct CP yields a blank
    /// image. It must be a fault, not an empty frame.
    #[test]
    fn zero_area_viewport_is_a_named_error() {
        let mut ctx = ctx_96x48();
        ctx.screen_viewport.viewports[0].xscale = 0.0;
        let e = draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV).expect_err("zero vp");
        assert!(e.0.contains("viewport"), "got {e}");
    }

    #[test]
    fn degenerate_extent_is_a_named_error() {
        let mut ctx = ctx_96x48();
        ctx.render_targets[0].attrib2 = ColorAttrib2::default();
        let e = draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV).expect_err("no extent");
        assert!(e.0.contains("ATTRIB2"), "got {e}");
    }

    #[test]
    fn unsupported_format_is_a_named_error() {
        let mut ctx = ctx_96x48();
        ctx.render_targets[0].info.format = 0x1;
        let e = draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV).expect_err("bad format");
        assert!(e.0.contains("format"), "got {e}");
    }

    #[test]
    fn unsupported_primitive_type_is_a_named_error() {
        let ucfg = UserConfig {
            prim_type: 99,
            ..UserConfig::default()
        };
        let e = draw_state_from_regs(&ctx_96x48(), &ucfg, 3, SPIRV, SPIRV).expect_err("bad prim");
        assert!(e.0.contains("VGT_PRIMITIVE_TYPE"), "got {e}");
    }

    /// RectList issues 4 vertices even though the guest asks for 3 — Kyty's
    /// own behaviour, and the embedded VS's quad depends on it.
    #[test]
    fn rect_list_becomes_a_four_vertex_strip() {
        let state =
            draw_state_from_regs(&ctx_96x48(), &ucfg_rect(), 3, SPIRV, SPIRV).expect("valid");
        assert_eq!(state.topology, vk::PrimitiveTopology::TRIANGLE_STRIP);
        assert_eq!(state.vertex_count, 4, "RectList draws a strip quad");
        assert!(
            state.vertices.is_none(),
            "the embedded VS declares no inputs"
        );
    }

    #[test]
    fn triangle_list_keeps_the_guest_index_count() {
        let ucfg = UserConfig {
            prim_type: prim::TRIANGLE_LIST,
            ..UserConfig::default()
        };
        let state = draw_state_from_regs(&ctx_96x48(), &ucfg, 3, SPIRV, SPIRV).expect("valid");
        assert_eq!(state.topology, vk::PrimitiveTopology::TRIANGLE_LIST);
        assert_eq!(state.vertex_count, 3);
    }

    #[test]
    fn gen5_triangle_fan_and_strip_match_kytys_vulkan_topologies() {
        for (prim_type, expected) in [
            (5, vk::PrimitiveTopology::TRIANGLE_FAN),
            (6, vk::PrimitiveTopology::TRIANGLE_STRIP),
        ] {
            let ucfg = UserConfig {
                prim_type,
                ..UserConfig::default()
            };
            let state = draw_state_from_regs(&ctx_96x48(), &ucfg, 6, SPIRV, SPIRV)
                .unwrap_or_else(|e| panic!("Gen5 primitive {prim_type}: {e}"));
            assert_eq!(state.topology, expected, "Gen5 primitive {prim_type}");
            assert_eq!(state.vertex_count, 6);
        }
    }

    /// A non-embedded bind with no readable code behind it must resolve to a
    /// **named** error (which `draw_index_auto` degrades to a skipped draw,
    /// never a crash and never a silently wrong image).
    #[test]
    fn non_embedded_shader_without_code_is_a_named_error() {
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        let sh = Shader::default(); // vs_embedded=false, data_addr=0
        let e = resolve_shaders(&mut cache, &ctx_96x48(), &sh).expect_err("no code to fetch");
        assert!(e.0.contains("null or unaligned"), "got {e}");
    }

    /// The embedded PS is `outColor = vec4(0)`. Alpha 0 is unreachable from the
    /// fixture, which is what makes the acceptance assertion decisive.
    #[test]
    fn embedded_shaders_assemble_to_real_spirv() {
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        let mut sh = Shader::default();
        sh.set_vs_embedded(0, 0);
        sh.set_ps_embedded(0);
        let r = resolve_shaders(&mut cache, &ctx_96x48(), &sh).expect("embedded pair");
        assert_eq!(r.vs[0], 0x0723_0203, "VS SPIR-V magic");
        assert_eq!(r.ps[0], 0x0723_0203, "PS SPIR-V magic");
    }

    #[test]
    fn format_mapping_covers_the_supported_orders() {
        assert_eq!(
            vulkan_format(0xa, 0, 0).unwrap(),
            vk::Format::R8G8B8A8_UNORM
        );
        assert_eq!(vulkan_format(0xa, 6, 0).unwrap(), vk::Format::R8G8B8A8_SRGB);
        assert_eq!(
            vulkan_format(0xa, 0, 1).unwrap(),
            vk::Format::B8G8R8A8_UNORM
        );
        assert!(vulkan_format(0xb, 0, 0).is_err());
    }

    #[cfg(windows)]
    #[test]
    // Nested array/struct field setup (`vs.resources[0].fields[3]`, `vs.buffers[0]`)
    // can't use struct-init syntax, so the default-then-assign form stays.
    #[allow(clippy::field_reassign_with_default)]
    fn measured_gen5_vertex_and_storage_resources_become_vulkan_bindings() {
        let vertex_words: Vec<u32> = vec![
            0xbf19_999a,
            0xbf19_999a,
            0, // -0.6, -0.6, 0
            0x3f19_999a,
            0xbf19_999a,
            0, //  0.6, -0.6, 0
            0xbf19_999a,
            0x3f19_999a,
            0, // -0.6,  0.6, 0
            0x3f19_999a,
            0x3f19_999a,
            0, //  0.6,  0.6, 0
        ];
        let storage_words: Vec<u32> = (0..32).collect();

        let mut vs = ShaderVertexInputInfo::default();
        vs.resources_num = 1;
        vs.buffers_num = 1;
        vs.resources[0].fields[3] = 74 << 12; // Gen5 float3
        vs.buffers[0].addr = vertex_words.as_ptr() as u64;
        vs.buffers[0].stride = 12;
        vs.buffers[0].num_records = 4;
        vs.buffers[0].attr_num = 1;
        vs.buffers[0].attr_indices[0] = 0;

        let ranges = [
            (
                vertex_words.as_ptr() as u64,
                std::mem::size_of_val(vertex_words.as_slice()),
            ),
            (
                storage_words.as_ptr() as u64,
                std::mem::size_of_val(storage_words.as_slice()),
            ),
        ];
        let (buffers, attributes) =
            crate::guest_mem::with_test_ranges(&ranges, || prepare_vertex_inputs(&vs))
                .expect("measured vertex ABI");
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].bytes.len(), 48);
        assert_eq!(buffers[0].stride, 12);
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0].format, vk::Format::R32G32B32_SFLOAT);

        let mut ps = ShaderPixelInputInfo::default();
        ps.bind.push_constant_size = 16;
        ps.bind.storage_buffers.buffers_num = 1;
        ps.bind.storage_buffers.binding_index = 0;
        let resource = &mut ps.bind.storage_buffers.buffers[0];
        resource.update_address48(storage_words.as_ptr() as u64);
        resource.fields[1] |= 16 << 16;
        resource.fields[2] = 8;
        resource.fields[3] = 77 << 12;

        let binding = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_stage_binding(&ps.bind, vk::ShaderStageFlags::FRAGMENT)
        })
        .expect("measured storage ABI");
        let storage = binding.storage_buffers.expect("descriptor set");
        assert_eq!(storage.binding, 0);
        assert_eq!(
            storage.buffers,
            vec![
                storage_words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>()
            ]
        );
        assert_eq!(binding.push_constants.len(), 16);
        assert_eq!(&binding.push_constants[0..4], &[0, 0, 0, 0]);
        assert_eq!(
            u32::from_le_bytes(binding.push_constants[4..8].try_into().unwrap()) >> 16,
            16,
            "rewritten descriptor must preserve the guest stride"
        );
    }

    /// A linear (tile mode 0) format-56 RGBA8 T# at `base`, `w`x`h`.
    fn rgba8_linear_tsharp(
        base: u64,
        w: u32,
        h: u32,
    ) -> kyty_graphics::shader::ShaderTextureResource {
        assert_eq!(base & 0xff, 0, "base40 drops the low 8 bits");
        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 56 << 20; // unified format 8_8_8_8 UNORM
        t.fields[1] |= ((w - 1) & 3) << 30; // width5 low bits
        t.fields[2] = (w - 1) >> 2; // width5 high bits
        t.fields[2] |= (h - 1) << 14; // height5
        t.fields[3] |= 9 << 28; // type = Texture2D (tile mode stays 0 = linear)
        t
    }

    /// The ASTRO.BOT front-#3 shape: a COMPUTE stage whose T#s mix sampled
    /// textures and storage images (usage == ReadWrite). The push constants
    /// must keep ALL T#s in analysis order, but each rewritten dword 0 is the
    /// index WITHIN its own descriptor array — sampled T#s count through
    /// `%textures2D_S` and storage T#s through `%textures2D_L` — and the
    /// storage list must carry extent, initial content, and guest base.
    #[test]
    // Nested array/struct field setup (`bind.textures2d.desc[0].texture`)
    // can't use struct-init syntax, so the default-then-assign form stays.
    #[allow(clippy::field_reassign_with_default)]
    fn compute_binding_splits_sampled_and_storage_tsharps() {
        const W: u32 = 4;
        const H: u32 = 4;
        const BYTES: usize = (W * H * 4) as usize;

        // One 256-aligned arena: sampled A at +0, storage at +256, sampled B
        // at +512 (base40 requires 256-aligned bases).
        let mut arena = vec![0u8; 1024 + 255];
        let base = (arena.as_ptr() as u64 + 255) & !255;
        let off = (base - arena.as_ptr() as u64) as usize;
        for i in 0..BYTES {
            arena[off + i] = (i % 251) as u8; // sampled A
            arena[off + 256 + i] = ((i * 3 + 7) % 251) as u8; // storage seed
            arena[off + 512 + i] = ((i * 5 + 11) % 251) as u8; // sampled B
        }
        let content = |o: usize| arena[off + o..off + o + BYTES].to_vec();
        let (tex_a, uav, tex_b) = (content(0), content(256), content(512));

        let mut bind = ShaderBindResources::default();
        bind.push_constant_size = 3 * 32; // three 8-dword T#s, nothing else
        bind.textures2d.textures_num = 3;
        bind.textures2d.textures2d_sampled_num = 2;
        bind.textures2d.textures2d_storage_num = 1;
        bind.textures2d.binding_sampled_index = 0;
        bind.textures2d.binding_storage_index = 1;
        bind.textures2d.desc[0].texture = rgba8_linear_tsharp(base, W, H);
        bind.textures2d.desc[1].texture = rgba8_linear_tsharp(base + 256, W, H);
        bind.textures2d.desc[1].usage = ShaderTextureUsage::ReadWrite;
        bind.textures2d.desc[2].texture = rgba8_linear_tsharp(base + 512, W, H);

        let ranges = [(arena.as_ptr() as u64, arena.len())];
        let binding = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_stage_binding(&bind, vk::ShaderStageFlags::COMPUTE)
        })
        .expect("mixed sampled + storage compute binding");

        // Sampled array: the two non-ReadWrite T#s, in analysis order.
        let textures = binding.textures.expect("sampled textures");
        assert_eq!(textures.sampled_binding, 0);
        assert_eq!(textures.textures.len(), 2);
        assert_eq!(textures.textures[0].pixels, tex_a);
        assert_eq!(textures.textures[1].pixels, tex_b);

        // Storage array: extent, guest seed content, and writeback base.
        let storage = binding.storage_images.expect("storage images");
        assert_eq!(storage.binding, 1);
        assert_eq!(storage.images.len(), 1);
        assert_eq!(storage.images[0].width, W);
        assert_eq!(storage.images[0].height, H);
        assert_eq!(storage.images[0].guest_base, base + 256);
        assert_eq!(storage.images[0].pixels, uav);

        // Push constants: 3 x 32 bytes in desc[] order; dword 0 of each is
        // the PER-ARRAY index — sampled 0, storage 0, sampled 1.
        assert_eq!(binding.push_constants.len(), 96);
        let dword0 = |group: usize| {
            u32::from_le_bytes(
                binding.push_constants[group * 32..group * 32 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        assert_eq!(dword0(0), 0, "first sampled T# indexes %textures2D_S[0]");
        assert_eq!(dword0(1), 0, "storage T# indexes %textures2D_L[0]");
        assert_eq!(dword0(2), 1, "second sampled T# indexes %textures2D_S[1]");
        // The rewrite must not clobber the descriptor's format bits.
        assert_eq!(
            (u32::from_le_bytes(binding.push_constants[36..40].try_into().unwrap()) >> 20) & 0x1ff,
            56,
            "storage T# dword 1 keeps its unified format"
        );

        // Graphics stages still reject storage images by name.
        let e = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_stage_binding(&bind, vk::ShaderStageFlags::FRAGMENT)
        })
        .expect_err("storage images are compute-only");
        assert!(e.0.contains("STORAGE"), "names the storage rejection: {e}");
    }

    /// A storage T# whose format is not 32-bpp must zero-fill its seed (warn
    /// once) instead of failing the dispatch — the UAV is typically fully
    /// overwritten by the shader anyway.
    #[test]
    fn storage_image_with_non_32bpp_format_zero_fills() {
        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(0x100); // base 0x10000, unreadable — must not matter
        t.fields[1] |= 1 << 20; // unified format 1 = R8 UNORM (1 byte per pixel)
        t.fields[1] |= 3 << 30; // width5 low bits (w-1 = 3)
        t.fields[2] |= 3 << 14; // height5 (h-1 = 3)
        t.fields[3] |= 9 << 28;
        let upload = read_storage_image(&t).expect("zero-filled storage image");
        assert_eq!((upload.width, upload.height), (4, 4));
        assert_eq!(upload.guest_base, 0x10000);
        assert_eq!(upload.pixels, vec![0u8; 64]);
    }

    /// The classic alpha-over blend: SRC_ALPHA / ONE_MINUS_SRC_ALPHA / ADD.
    #[test]
    fn blend_state_maps_alpha_over() {
        let mut ctx = ctx_96x48();
        ctx.blend_control[0] = kyty_graphics::hw_regs::BlendControl {
            color_srcblend: 0x04,  // SrcAlpha
            color_comb_fcn: 0,     // ADD
            color_destblend: 0x05, // OneMinusSrcAlpha
            alpha_srcblend: 0x01,  // One
            alpha_comb_fcn: 0,
            alpha_destblend: 0x00, // Zero
            separate_alpha_blend: true,
            enable: true,
        };
        ctx.blend_color.red = 0.25;
        let blend = blend_state_from_regs(&ctx).expect("supported blend state");
        assert!(blend.enable);
        assert_eq!(blend.src_color, vk::BlendFactor::SRC_ALPHA);
        assert_eq!(blend.dst_color, vk::BlendFactor::ONE_MINUS_SRC_ALPHA);
        assert_eq!(blend.color_op, vk::BlendOp::ADD);
        assert_eq!(blend.src_alpha, vk::BlendFactor::ONE);
        assert_eq!(blend.dst_alpha, vk::BlendFactor::ZERO);
        assert_eq!(blend.constants[0], 0.25);
    }

    /// Without `separate_alpha_blend` the alpha channel reuses the *colour*
    /// factors — that is the hardware behaviour, not an approximation.
    #[test]
    fn blend_state_without_separate_alpha_reuses_color_factors() {
        let mut ctx = ctx_96x48();
        ctx.blend_control[0] = kyty_graphics::hw_regs::BlendControl {
            color_srcblend: 0x02,  // SrcColor
            color_comb_fcn: 1,     // SUBTRACT
            color_destblend: 0x08, // DestColor
            alpha_srcblend: 0x1f,  // junk on purpose — must be ignored
            alpha_comb_fcn: 0x7,
            alpha_destblend: 0x1f,
            separate_alpha_blend: false,
            enable: true,
        };
        let blend = blend_state_from_regs(&ctx).expect("supported");
        assert_eq!(blend.src_alpha, vk::BlendFactor::SRC_COLOR);
        assert_eq!(blend.dst_alpha, vk::BlendFactor::DST_COLOR);
        assert_eq!(blend.alpha_op, vk::BlendOp::SUBTRACT);
    }

    /// Dual-source factors have no single-source Vulkan equivalent — a named
    /// error, never a silent ZERO.
    #[test]
    fn blend_factor_dual_source_is_a_named_error() {
        for code in [0x0b_u8, 0x0c, 0x0f, 0x10, 0x11, 0x12, 0x15, 0xff] {
            let e = gen5_blend_factor(code).expect_err("must be named");
            let msg = format!("{e}");
            assert!(
                msg.contains(&format!("{code:#04x}")),
                "error names the code: {msg}"
            );
        }
    }

    #[test]
    fn blend_op_reserved_is_a_named_error() {
        let e = gen5_blend_op(5).expect_err("must be named");
        assert!(format!("{e}").contains('5'));
    }

    /// Gen5 unified vertex formats per SharpEmu's Gfx10UnifiedFormat table:
    /// 64 → (11,7) 32_32_FLOAT, 74 → (13,7) 32_32_32_FLOAT,
    /// 77 → (14,7) 32_32_32_32_FLOAT (Minecraft's float4 vertex arena).
    #[test]
    fn gen5_vertex_formats_map_per_sharpemu_table() {
        assert_eq!(gen5_vertex_format(64).unwrap(), vk::Format::R32G32_SFLOAT);
        assert_eq!(
            gen5_vertex_format(74).unwrap(),
            vk::Format::R32G32B32_SFLOAT
        );
        assert_eq!(
            gen5_vertex_format(77).unwrap(),
            vk::Format::R32G32B32A32_SFLOAT
        );
        assert_eq!(gen5_vertex_format(56).unwrap(), vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            gen5_vertex_format(71).unwrap(),
            vk::Format::R16G16B16A16_SFLOAT
        );
        assert_eq!(gen5_vertex_format(23).unwrap(), vk::Format::R16G16_UNORM);
        // 11 → (2,4) = 16 UINT (Minecraft's packed per-vertex value).
        // Format 11 is a 16-bit INTEGER on hardware, but every vertex input we
        // generate is declared float, and Vulkan requires the numeric types to
        // match. USCALED delivers the same value as a float; UINT makes the
        // pipeline invalid and the draw silently contributes no fragment
        // (measured on Minecraft via the validation layer).
        assert_eq!(gen5_vertex_format(11).unwrap(), vk::Format::R16_USCALED);
        for f in [64u8, 74, 77, 56, 57, 71, 23, 11] {
            let vf = gen5_vertex_format(f).unwrap();
            assert!(
                !format!("{vf:?}").contains("UINT") && !format!("{vf:?}").contains("SINT"),
                "vertex format {f} maps to integer {vf:?}, which cannot feed a float shader input"
            );
        }
        // 57 → (10,1) = 8_8_8_8 SNORM (same UI draw, next attribute).
        assert_eq!(gen5_vertex_format(57).unwrap(), vk::Format::R8G8B8A8_SNORM);
        let e = gen5_vertex_format(0).expect_err("unknown formats stay named");
        assert!(format!("{e}").contains('0'));
    }

    /// A tiled T# (SWIZZLE_MODE 27 = SW_64KB_R_X, format 56 = 8_8_8_8 UNORM —
    /// the pair Minecraft's UI binds) decodes back to the original pixels.
    #[test]
    fn decode_texture_detiles_sw_64kb_r_x() {
        let (w, h, bpp_log2) = (8u32, 8u32, 2u32);
        let linear: Vec<u8> = (0..(w * h) as usize * 4)
            .map(|i| ((i * 7 + 3) % 251) as u8)
            .collect();
        let tiled = crate::texture::tiling::tile_64kb_r_x(&linear, w, h, bpp_log2);
        // Fake a 256-aligned guest base (base40 drops the low 8 bits).
        let mut blob = vec![0u8; tiled.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + tiled.len()].copy_from_slice(&tiled);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 56 << 20; // unified format 8_8_8_8 UNORM
        t.fields[1] |= ((w - 1) & 3) << 30; // width5 low bits
        t.fields[2] = (w - 1) >> 2; // width5 high bits
        t.fields[2] |= (h - 1) << 14; // height5
        t.fields[3] |= 27 << 20; // SWIZZLE_MODE = SW_64KB_R_X
        t.fields[3] |= 9 << 28; // type = Texture2D

        let tex = crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
            decode_texture(&t)
        })
        .expect("mode-27 texture decodes");
        assert_eq!((tex.width, tex.height), (w, h));
        assert_eq!(tex.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(tex.pixels, linear, "detiled pixels must match the original");
    }

    /// The COLOUR-BUFFER (CB_COLOR_INFO) format table is a different table from
    /// the texture (T#) one — same data-format numbering, different consumer.
    /// Every accepted entry must also have a `readback_bpp` size or the
    /// offscreen readback fails at run time, so they are asserted together.
    #[test]
    fn cb_colour_formats_map_and_have_readback_sizes() {
        for (fmt, ty, order, want, bpp) in [
            (0xa, 0, 0, vk::Format::R8G8B8A8_UNORM, 4u32),
            (0x6, 7, 0, vk::Format::B10G11R11_UFLOAT_PACK32, 4),
            (0xc, 7, 0, vk::Format::R16G16B16A16_SFLOAT, 8),
        ] {
            let got = vulkan_format(fmt, ty, order)
                .unwrap_or_else(|e| panic!("CB format {fmt:#x}/{ty}/{order}: {e}"));
            assert_eq!(got, want, "CB format {fmt:#x}");
            assert_eq!(
                crate::vulkan::offscreen::readback_bpp(got).ok(),
                Some(bpp),
                "{want:?} needs a readback size"
            );
        }
        // CB format 0x3 (8_8) must STAY rejected: mapping it to R8G8_UNORM was
        // measured to lose the Vulkan device and drop ASTRO.BOT from 10 frames
        // to 0. This asserts the regression cannot be reintroduced without
        // also making 2-channel attachments work.
        assert!(
            vulkan_format(0x3, 0, 0).is_err(),
            "CB format 0x3 causes VK_ERROR_DEVICE_LOST — see the arm's comment"
        );
    }

    /// Unified format 71 = (dataFormat 12 = 16_16_16_16, numFormat 7 = FLOAT):
    /// the 8-byte-per-pixel FP16 HDR surface ASTRO.BOT samples back as a
    /// texture. Exercises the 8-bpp (`bpp_log2` 3) row of the swizzle table,
    /// which no 1/4-byte format reaches.
    #[test]
    fn decode_texture_accepts_rgba16f_at_eight_bytes_per_pixel() {
        let (w, h, bpp_log2) = (8u32, 8u32, 3u32);
        let linear: Vec<u8> = (0..(w * h) as usize * 8)
            .map(|i| ((i * 11 + 5) % 251) as u8)
            .collect();
        let tiled = crate::texture::tiling::tile_64kb_r_x(&linear, w, h, bpp_log2);
        let mut blob = vec![0u8; tiled.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + tiled.len()].copy_from_slice(&tiled);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 71 << 20; // unified format 16_16_16_16 FLOAT
        t.fields[1] |= ((w - 1) & 3) << 30;
        t.fields[2] = (w - 1) >> 2;
        t.fields[2] |= (h - 1) << 14;
        t.fields[3] |= 27 << 20; // SWIZZLE_MODE = SW_64KB_R_X
        t.fields[3] |= 9 << 28; // type = Texture2D

        let tex = crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
            decode_texture(&t)
        })
        .expect("format-71 texture decodes");
        assert_eq!((tex.width, tex.height), (w, h));
        assert_eq!(tex.format, vk::Format::R16G16B16A16_SFLOAT);
        assert_eq!(tex.pixels, linear, "detiled FP16 pixels must match");
    }

    /// A cube T# (type 11, six faces, SWIZZLE_MODE 9 = SW_64KB_S — the
    /// measured skybox shape) decodes every face and marks the upload CUBE.
    #[test]
    fn decode_texture_cube_decodes_all_six_faces() {
        let (w, h, bpp_log2) = (8u32, 8u32, 2u32);
        let bpp = 1usize << bpp_log2;
        // Six faces with distinct first-byte-per-face content.
        let faces: Vec<Vec<u8>> = (0..6u8)
            .map(|f| {
                (0..(w * h) as usize * bpp)
                    .map(|i| (f * 40 + (i % 37) as u8) % 251)
                    .collect()
            })
            .collect();
        let tiled: Vec<u8> = faces
            .iter()
            .flat_map(|f| crate::texture::tiling::tile_64kb_s(f, w, h, bpp_log2))
            .collect();
        let mut blob = vec![0u8; tiled.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + tiled.len()].copy_from_slice(&tiled);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 56 << 20;
        t.fields[1] |= ((w - 1) & 3) << 30;
        t.fields[2] = (w - 1) >> 2;
        t.fields[2] |= (h - 1) << 14;
        t.fields[3] |= 9 << 20; // SWIZZLE_MODE 9 = SW_64KB_S
        t.fields[3] |= 11 << 28; // type = Cube
        t.fields[4] = 5; // depth 5 + 1 = 6 faces

        let tex = crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
            decode_texture(&t)
        })
        .expect("cube texture decodes");
        assert!(tex.cube);
        assert_eq!(tex.layers, 6);
        assert_eq!((tex.width, tex.height), (w, h));
        let face_bytes = (w * h) as usize * bpp;
        for (l, face) in faces.iter().enumerate() {
            assert_eq!(
                &tex.pixels[l * face_bytes..(l + 1) * face_bytes],
                face.as_slice(),
                "face {l} must detile to its original pixels"
            );
        }
    }

    /// A 3D T# (type 10, linear tile 0 — the measured ASTRO.BOT froxel/LUT
    /// volume shape, format 1 = R8_UNORM) decodes every slice and carries the
    /// depth so the Vulkan layer builds a `VK_IMAGE_TYPE_3D` image.
    #[test]
    fn decode_texture_3d_volume_decodes_all_slices() {
        let (w, h, d) = (8u32, 4u32, 4u32);
        let voxels: Vec<u8> = (0..(w * h * d) as usize)
            .map(|i| ((i * 7 + 3) % 251) as u8)
            .collect();
        let mut blob = vec![0u8; voxels.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + voxels.len()].copy_from_slice(&voxels);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 1 << 20; // unified format 1 = 8 UNORM
        t.fields[1] |= ((w - 1) & 3) << 30;
        t.fields[2] = (w - 1) >> 2;
        t.fields[2] |= (h - 1) << 14;
        t.fields[3] |= 10 << 28; // type = 3D volume (tile mode 0 = linear)
        t.fields[4] = (d - 1) & 0x1FFF; // depth

        let tex = crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
            decode_texture(&t)
        })
        .expect("3D volume decodes");
        assert!(!tex.cube);
        assert_eq!(tex.layers, 1);
        assert_eq!((tex.width, tex.height, tex.depth), (w, h, d));
        assert_eq!(tex.format, vk::Format::R8_UNORM);
        assert_eq!(tex.pixels, voxels, "all slices must decode in order");
    }
}
