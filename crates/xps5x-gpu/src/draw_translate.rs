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
    BlendState, CLEAR_COLOR, DrawState, EudRawBinding, RenderedImage, SampledGroup,
    ShaderStageBinding, StorageBufferBinding, StorageImageBinding, StorageImageUpload,
    TextureBinding, TextureUpload, VertexAttributeData, VertexBufferData,
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
    /// AMD `DI_PT_POINTLIST`: one point per vertex. Measured: ASTRO.BOT issues
    /// point-list draws in its render loop (78 skips before support). Maps 1:1
    /// to Vulkan's point-list topology.
    pub const POINT_LIST: u32 = 1;
    /// AMD `DI_PT_LINELIST` / `DI_PT_LINESTRIP`: the standard line primitives,
    /// each a direct Vulkan topology. Included alongside point-list so a debug
    /// or UI line draw does not skip the way point-list did.
    pub const LINE_LIST: u32 = 2;
    pub const LINE_STRIP: u32 = 3;
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
        // CB format 0x3 (8_8 UNORM). MEASURED: mapping it to the exact
        // R8G8_UNORM (a 2-channel attachment) device-loses on vkQueueSubmit —
        // the recompiled PS exports 4 components (mrt0 = vec4) and the pipeline
        // cannot honour a 2-channel target. The code's own note pointed at the
        // fix: keep the PS's 4-component export and give it a 4-CHANNEL target.
        // R8G8B8A8_UNORM is the same attachment class as the working 0xa/0x6/0xc
        // formats, so it does not device-lose; the two live channels (R,G) land
        // correct and B,A are extra. The stride widens 2->4 B/texel, which is
        // safe here because render targets stay GPU-side (no guest writeback) and
        // `readback_bpp(R8G8B8A8_UNORM)`=4 matches. A wide/4-channel
        // approximation of a 2-channel target is a glitch, never a device loss —
        // it lets the 16 composite draws SUCCEED instead of being skipped.
        (0x3, 0, 0) => Ok(vk::Format::R8G8B8A8_UNORM),
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
    // Fallible: a full-resolution texture is tens of MiB; under host memory
    // pressure the byte buffer must degrade to a named skip, not abort. The
    // dword read above (`read_dwords_validated`) already reserves fallibly.
    let mut bytes: Vec<u8> = Vec::new();
    bytes.try_reserve_exact(size as usize).map_err(|_| {
        err(format!(
            "{kind} at {addr:#x}: {size} B host allocation failed (out of memory) — skipping"
        ))
    })?;
    bytes.extend(words.into_iter().flat_map(u32::to_le_bytes));
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

/// Zero-filled host buffer of `len` bytes that returns a named [`DrawError`]
/// instead of ABORTING the process when the allocation fails.
///
/// The final composite decodes several full-resolution scene targets from guest
/// memory in one pass; under host memory pressure an infallible `vec![0u8; len]`
/// aborts the whole process (the measured crash: "memory allocation of 26615808
/// bytes failed" = one 2432x1368 RGBA16F target). A fallible reservation
/// degrades that into a skipped draw/dispatch, so the process survives and a
/// later frame — with more headroom — can retry. Paired with the per-stage
/// decode cap (`stage_texture_byte_cap`), which preemptively refuses a composite
/// whose cumulative decode is clearly too large.
fn alloc_zeroed(len: usize, kind: &str) -> Result<Vec<u8>, DrawError> {
    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve_exact(len).map_err(|_| {
        err(format!(
            "{kind}: {len} B host allocation failed (out of memory) — skipping the draw/dispatch \
             instead of aborting the process"
        ))
    })?;
    buf.resize(len, 0);
    Ok(buf)
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

/// Gen5 unified T# format -> (Vulkan format, bytes per pixel), decoded via
/// SharpEmu's Gfx10UnifiedFormat table (the RDNA2 authority). Filled from
/// measured titles only; an unhandled value names itself rather than guessing.
fn texture_vk_format(
    t: &kyty_graphics::shader::ShaderTextureResource,
) -> Result<(vk::Format, u32), DrawError> {
    match t.format() {
        // 1 = single 8-bit channel, UNORM (measured on ASTRO.BOT's 480x270
        // coverage/mask texture, tile mode 27). SharpEmu's Gfx10UnifiedFormat
        // maps unified 1 -> (dataFormat 1 = FMT_8, numFormat 0 = UNORM).
        1 => Ok((vk::Format::R8_UNORM, 1)),
        // 7 -> (2,0) = 16 UNORM: a single 16-bit normalized channel
        // (measured: ASTRO.BOT's 1536x1536x3 2DArray, tile mode 24).
        // SharpEmu Gfx10UnifiedFormat maps unified 7 -> (dataFormat 2 = 16,
        // numFormat 0 = UNORM).
        7 => Ok((vk::Format::R16_UNORM, 2)),
        // 14 -> (3,0) = 8_8 UNORM (measured: ASTRO.BOT samples a 1920x1080
        // format-14 texture, tile mode 27 — 64 draws/run failed on it). SharpEmu
        // Gfx10UnifiedFormat.cs:40 maps unified 14 -> (dataFormat 3 = 8_8,
        // numFormat 0 = UNORM) per the standard GCN table (df1=8, df2=16,
        // df3=8_8), so R8G8_UNORM at 2 B/texel.
        14 => Ok((vk::Format::R8G8_UNORM, 2)),
        // 36 = 10_11_11 FLOAT (packed 32-bit HDR) — the title samples its HDR
        // render target as a texture. SharpEmu Gfx10UnifiedFormat maps unified
        // 36 -> (dataFormat 6 = 10_11_11, numFormat 7 = FLOAT).
        36 => Ok((vk::Format::B10G11R11_UFLOAT_PACK32, 4)),
        // 0x0a = 8_8_8_8; channel type UNORM (measured on Minecraft's UI T#s).
        // NOTE: SharpEmu's table maps unified 10 -> (2,3) = 16_SSCALED, which
        // contradicts this arm. No 0x0a texture has appeared in a measured
        // run since; the first one that does must settle the table.
        0x0a => Ok((vk::Format::R8G8B8A8_UNORM, 4)),
        // 56 -> (10,0) = 8_8_8_8 UNORM (measured: Minecraft's 1920x1080 UI
        // texture, tile mode 27).
        56 => Ok((vk::Format::R8G8B8A8_UNORM, 4)),
        // 22 -> (4,7) = 32 FLOAT (measured: ASTRO.BOT's 1920x1080 R32F buffer
        // — a linear-depth/scalar target sampled back as a texture). SharpEmu
        // Gfx10UnifiedFormat.cs:48 maps unified 22 -> (dataFormat 4,
        // numFormat 7); dataFormat 4 is the single 32-bit channel per its Gen5
        // layout table ("SetLayout(4, 0, 0, 32); // 32") and numFormat 7 is
        // FLOAT, the same numFormat as the 36 and 71 arms.
        22 => Ok((vk::Format::R32_SFLOAT, 4)),
        // 29 -> (5,7) = 16_16 FLOAT (user log flagged unified format 29 as
        // unimplemented). SharpEmu Gfx10UnifiedFormat.cs:55 maps unified 29 ->
        // (dataFormat 5, numFormat 7); dataFormat 5 is the two-channel 16_16
        // per the standard GCN IMG_DATA_FORMAT table (the same table that
        // gives dataFormat 4 = 32, 12 = 16_16_16_16), and numFormat 7 is
        // FLOAT — so R16G16_SFLOAT at 4 B/texel.
        29 => Ok((vk::Format::R16G16_SFLOAT, 4)),
        // 65 -> (12,0) = 16_16_16_16 UNORM (user log flagged unified format 65
        // as unimplemented). SharpEmu Gfx10UnifiedFormat.cs:77 maps unified 65
        // -> (dataFormat 12, numFormat 0); dataFormat 12 is 16_16_16_16 (same
        // channel layout as the FLOAT arm 71 below) and numFormat 0 is UNORM —
        // so R16G16B16A16_UNORM at 8 B/texel.
        65 => Ok((vk::Format::R16G16B16A16_UNORM, 8)),
        // 71 -> (12,7) = 16_16_16_16 FLOAT (measured: ASTRO.BOT's 2432x1368
        // HDR scene buffer sampled back as a texture, tile mode 27). SharpEmu's
        // Gfx10UnifiedFormat maps unified 71 -> (dataFormat 12, numFormat 7);
        // dataFormat 12 is 16_16_16_16 per its Gen5 layout table, and numFormat
        // 7 is FLOAT (same numFormat as the 36 arm above). Every draw in the
        // title's 7966-dword DCB failed on this one format.
        71 => Ok((vk::Format::R16G16B16A16_SFLOAT, 8)),
        // 5 -> (dataFormat 1, numFormat 4) = 8-bit UINT (SharpEmu
        // Gfx10UnifiedFormat unified 5 -> (1u, 4u)); R8_UINT at 1 B/texel.
        // Measured on ASTRO.BOT's 1920x1080 tile=24 target sampled as a texture.
        5 => Ok((vk::Format::R8_UINT, 1)),
        other => Err(err(format!(
            "texture format {other} not implemented \
             (base={:#x} {}x{} pitch={} tile={} levels={})",
            t.base40(),
            u32::from(t.width5()) + 1,
            u32::from(t.height5()) + 1,
            t.pitch(),
            t.tile_mode(),
            t.last_level()
        ))),
    }
}

/// Sparse sample-hash of a guest byte range for the persistent-texture cache
/// (stage D): FNV-1a over the range length plus 64 evenly-strided 64-byte
/// chunks and the final 64 bytes — ~4 KiB of guest reads regardless of
/// texture size (a whole-range hash for ranges up to 4 KiB). Never returns 0
/// (0 is the "no hash / not cacheable" sentinel), and returns `None` when the
/// guest range is not readable — the caller then decodes uncached and produces
/// its own named error if the range is truly bad.
///
/// ## Staleness window (documented, deliberate)
///
/// A CPU guest write that leaves every sampled chunk byte-identical is NOT
/// detected: the cached image keeps being bound until any sampled byte
/// changes. The window is bounded by the sample coverage — the hash is
/// recomputed from guest memory on EVERY bind, so any write that touches a
/// sampled chunk is picked up at the next draw. Writeback paths we control
/// (compute storage writeback, DMA copies) do not proactively invalidate the
/// texture cache — no cheap range index over it exists today — so they are
/// covered by the same per-bind rehash. `XPS5X_NO_TEX_CACHE=1` restores
/// per-draw decode + upload wholesale.
fn guest_sample_hash(base: u64, len: u64) -> Option<u64> {
    const CHUNKS: u64 = 64;
    const CHUNK_BYTES: u64 = 64;
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    fn mix(mut h: u64, bytes: &[u8]) -> u64 {
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(FNV_PRIME);
        }
        h
    }
    if len == 0 {
        return None;
    }
    let mut h = mix(FNV_OFFSET, &len.to_le_bytes());
    if len <= CHUNKS * CHUNK_BYTES {
        let bytes = read_guest_bytes_unaligned(base, len, "texture sample-hash").ok()?;
        h = mix(h, &bytes);
    } else {
        let stride = len / CHUNKS;
        for i in 0..CHUNKS {
            let offset = i * stride;
            let size = CHUNK_BYTES.min(len - offset);
            let bytes =
                read_guest_bytes_unaligned(base + offset, size, "texture sample-hash").ok()?;
            h = mix(h, &bytes);
        }
        // The strided chunks can miss the very end of the range; the tail is
        // where partial updates (e.g. an atlas row append) often land.
        let tail = read_guest_bytes_unaligned(
            base + (len - CHUNK_BYTES),
            CHUNK_BYTES,
            "texture sample-hash",
        )
        .ok()?;
        h = mix(h, &tail);
    }
    Some(h.max(1))
}

/// Consult the persistent-texture cache before decoding a T# (stage D).
///
/// Returns the fresh sample-hash of the guest source range (0 when caching is
/// disabled, no sampling scope is published — the compute path — or the range
/// is unreadable) and, when the published cache snapshot holds this exact
/// texture with an equal hash, the ready [`TextureUpload`] that binds the
/// cached image (empty pixels, `cached: true`) so the caller skips the guest
/// read and detile entirely.
#[allow(clippy::too_many_arguments)]
fn texture_cache_probe(
    base: u64,
    src_len: u64,
    width: u32,
    height: u32,
    layers: u32,
    depth: u32,
    cube: bool,
    format: vk::Format,
) -> (u64, Option<TextureUpload>) {
    if std::env::var_os("XPS5X_NO_TEX_CACHE").is_some() {
        return (0, None);
    }
    let Some(hash) = sampling_scope(|_| guest_sample_hash(base, src_len)) else {
        return (0, None);
    };
    let hit = sampling_scope(|scope| {
        scope
            .cached_textures
            .iter()
            .find(|(k, cached_hash)| {
                k.base == base
                    && k.width == width
                    && k.height == height
                    && k.layers == layers
                    && k.depth == depth
                    && k.cube == cube
                    && k.format == format.as_raw()
                    && *cached_hash == hash
            })
            .map(|_| TextureUpload {
                width,
                height,
                format,
                pixels: Vec::new(),
                layers,
                cube,
                depth,
                render_target: None,
                guest_base: base,
                sample_hash: hash,
                cached: true,
            })
    });
    (hash, hit)
}

/// Addresses whose multi-layer array upload once overran its real allocation.
///
/// SharpEmu #476 (224a36e): a 2D/1D-array texture whose `Depth * per-slice
/// stride` runs past its allocation fails a slice read partway through the
/// upload and, without a memo, re-detiles the whole thing on every draw
/// (measured 568-879 ms of every second in Demon's Souls). An allocation that
/// is too short stays too short, so remembering the address and never retrying
/// the array read costs nothing and repairs the cache key as a side effect:
/// the fall-back reads a single base layer, which keys under `layers == 1`,
/// exactly what later draws look it up with.
static ARRAY_UPLOAD_UNSUPPORTED: std::sync::RwLock<Vec<u64>> = std::sync::RwLock::new(Vec::new());

fn array_upload_unsupported(base: u64) -> bool {
    ARRAY_UPLOAD_UNSUPPORTED
        .read()
        .map(|set| set.contains(&base))
        .unwrap_or(false)
}

fn mark_array_upload_unsupported(base: u64) {
    if let Ok(mut set) = ARRAY_UPLOAD_UNSUPPORTED.write()
        && !set.contains(&base)
    {
        set.push(base);
    }
}

/// The number of distinct sampled-array keys — 4 Dims x 3 numeric classes
/// (`kyty_graphics::shader::sampled_key_ordinal` is Dim-major).
const SAMPLED_KEYS: usize = 12;

/// The canonical (Dim, numeric class) key ordinal of a sampled T#, matching
/// `kyty_graphics::shader::sampled_key_ordinal`. A mixed shader assigns
/// per-key descriptor bindings in this order, so the host and the SPIR-V
/// generator agree on which array each T# lands at
/// (`binding_sampled_index + position-among-present-keys`).
fn sampled_key_ordinal(t: &kyty_graphics::shader::ShaderTextureResource) -> usize {
    // Delegates to the SPIR-V generator's own classifiers so the host and the
    // shader can never disagree on a T#'s array (classifier drift here would
    // bind a texture into an array the shader never reads it from — or, for
    // the class axis, put an R8_UINT view under a `%float` image type:
    // the measured VUID-vkCmdDispatch-format-07753).
    kyty_graphics::shader::sampled_key_ordinal(
        kyty_graphics::shader::SampledDim::from_texture_type(t.type_()),
        kyty_graphics::shader::SampledClass::from_unified_format(t.format()),
    ) as usize
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
    // 10 = 3D volume (measured: ASTRO.BOT's 240x135x64 froxel/LUT volumes),
    // 13 = 2DArray (measured: ASTRO.BOT's 1536x1536x3 array, tile 24 —
    // the T# depth field carries the layer count).
    let (cube, volume, array) = match t.type_() {
        // 8 = Texture1D. A 1D image is a 2D image one row tall, and the T#
        // already reports height5 = 0 => height 1, so the existing 2D decode
        // path handles it unchanged (measured on ASTRO.BOT: a 1x1 format-71
        // texture, tile mode 27). Kept a distinct arm rather than folding into
        // 9 so the disagreement is visible if a >1-row "1D" texture ever shows.
        8 => (false, false, false),
        9 => (false, false, false),
        10 => (false, true, false),
        11 => (true, false, false),
        13 => (false, false, true),
        other => {
            return Err(err(format!(
                "texture type {other} is not Texture2D (9), 3D (10), Cube (11) or 2DArray (13)"
            )));
        }
    };
    let mut layers = if cube || array {
        u32::from(t.depth()) + 1
    } else {
        1
    };
    let depth = if volume { u32::from(t.depth()) + 1 } else { 1 };
    if volume && !(1..=2048).contains(&depth) {
        return Err(err(format!("volume depth {depth} out of range")));
    }

    let (format, bpp) = texture_vk_format(t)?;

    // Persistent-texture cache probe (stage D): hashed against the SOURCE
    // bytes (pitch-padded / tiled, exactly what each branch would read), so a
    // cache hit skips the guest read AND the detile AND the upload. Each
    // decoding branch yields (pixels, source sample-hash).
    let (pixels, sample_hash) = match t.tile_mode() {
        0 if cube || array => {
            return Err(err(
                "cube/2DArray texture with linear tile mode not implemented (only tiled measured)",
            ));
        }
        0 => {
            // Linear: row-major at `pitch`, trimmed to tight rows below. A
            // volume is `depth` such slices back to back (slice pitch =
            // pitch * height for a linear T# — the measured ASTRO.BOT
            // volumes are tile 0).
            let pitch = u32::from(t.pitch()).max(width);
            let src_len = u64::from(pitch) * u64::from(height) * u64::from(depth) * u64::from(bpp);
            let (hash, hit) = texture_cache_probe(
                t.base40(),
                src_len,
                width,
                height,
                layers,
                depth,
                cube,
                format,
            );
            if let Some(upload) = hit {
                return Ok(upload);
            }
            let tiled = read_guest_bytes_unaligned(t.base40(), src_len, "texture")?;
            let row = (width * bpp) as usize;
            let src_row = (pitch * bpp) as usize;
            let src_slice = src_row * height as usize;
            let dst_slice = row * height as usize;
            let mut pixels = alloc_zeroed(dst_slice * depth as usize, "texture decode")?;
            for z in 0..depth as usize {
                for y in 0..height as usize {
                    let src = z * src_slice + y * src_row;
                    let dst = z * dst_slice + y * row;
                    pixels[dst..dst + row].copy_from_slice(&tiled[src..src + row]);
                }
            }
            (pixels, hash)
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
        mode if crate::texture::tiling::swizzle_table(mode).is_some() => {
            let bpp_log2 = bpp.trailing_zeros();
            let face_tiled = crate::texture::tiling::tiled_byte_count_for_mode(
                mode, width, height, bpp_log2,
            )
            .expect("guarded by swizzle_table above") as usize;
            let face_linear = (width * height * bpp) as usize;
            // Array-upload OOM guard (SharpEmu #476 / 224a36e): if a previous
            // draw's multi-layer read of this address overran its allocation,
            // never retry — drop to a single base layer up front. layers == 1
            // keys the cache the same way later single-layer draws look it up.
            if layers > 1 && array_upload_unsupported(t.base40()) {
                layers = 1;
            }
            let probe = |layers: u32| {
                texture_cache_probe(
                    t.base40(),
                    face_tiled as u64 * u64::from(layers),
                    width,
                    height,
                    layers,
                    depth,
                    cube,
                    format,
                )
            };
            let (hash, hit) = probe(layers);
            if let Some(upload) = hit {
                return Ok(upload);
            }
            let src_len = face_tiled as u64 * u64::from(layers);
            let tiled = match read_guest_bytes_unaligned(t.base40(), src_len, "texture") {
                Ok(tiled) => tiled,
                // The multi-layer array read overran the allocation: remember it
                // so no draw retries the overrun, fall back to a single base
                // layer, and re-probe under layers == 1 for a possible hit.
                Err(_) if layers > 1 => {
                    mark_array_upload_unsupported(t.base40());
                    layers = 1;
                    let (single_hash, single_hit) = probe(layers);
                    if let Some(upload) = single_hit {
                        return Ok(upload);
                    }
                    let single =
                        read_guest_bytes_unaligned(t.base40(), face_tiled as u64, "texture")?;
                    let face =
                        crate::texture::tiling::detile_64kb(mode, &single, width, height, bpp_log2)
                            .expect("table-checked above");
                    let mut pixels = alloc_zeroed(face_linear, "texture decode")?;
                    pixels[..face_linear].copy_from_slice(&face);
                    return Ok(texture_upload_from(
                        t,
                        width,
                        height,
                        format,
                        pixels,
                        layers,
                        cube,
                        depth,
                        single_hash,
                    ));
                }
                Err(e) => return Err(e),
            };
            let mut pixels = alloc_zeroed(face_linear * layers as usize, "texture decode")?;
            for layer in 0..layers as usize {
                let src = &tiled[layer * face_tiled..(layer + 1) * face_tiled];
                let face = crate::texture::tiling::detile_64kb(mode, src, width, height, bpp_log2)
                    .expect("table-checked above");
                pixels[layer * face_linear..(layer + 1) * face_linear].copy_from_slice(&face);
            }
            (pixels, hash)
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
        // This path has already read and detiled the guest bytes above, so it is
        // the CPU-staging upload — not the Stage-B direct-bind of a live
        // persistent target (that would early-return before the detile with
        // `render_target: Some(base)`).
        render_target: None,
        // Cache identity (stage D): with a non-zero hash the backend donates
        // the uploaded image to the persistent-texture cache on draw success.
        guest_base: t.base40(),
        sample_hash,
        cached: false,
    })
}

/// Build a CPU-staged [`TextureUpload`] from already-decoded linear pixels.
/// Shared by the array-upload OOM fall-back (SharpEmu #476), which returns a
/// single base layer, and any other path that has finished the guest read and
/// detile itself. `render_target` is always `None` (this is a staging upload,
/// not a live-target direct bind) and `cached` is `false` (the backend donates
/// the image to the persistent-texture cache on draw success when the hash is
/// non-zero).
#[allow(clippy::too_many_arguments)]
fn texture_upload_from(
    t: &kyty_graphics::shader::ShaderTextureResource,
    width: u32,
    height: u32,
    format: vk::Format,
    pixels: Vec<u8>,
    layers: u32,
    cube: bool,
    depth: u32,
    sample_hash: u64,
) -> TextureUpload {
    TextureUpload {
        width,
        height,
        format,
        pixels,
        layers,
        cube,
        depth,
        render_target: None,
        guest_base: t.base40(),
        sample_hash,
        cached: false,
    }
}

/// Gen5 unified T# formats that are 32 bits per pixel — their guest bytes
/// seed an `R8G8B8A8_UNORM` storage image directly (the recompiled SPIR-V's
/// `%textures2D_L` uses `Rgba8` for them).
fn storage_image_format_is_32bpp(format: u16) -> bool {
    // 0x0a/56 = 8_8_8_8, 22 = 32 FLOAT, 36 = 10_11_11 FLOAT — see
    // `decode_texture`'s table for the measurements behind each.
    matches!(format, 0x0a | 22 | 36 | 56)
}

/// Read one storage-image (UAV) T#'s extent and initial guest content.
///
/// A type-10 T# is a 3D UAV (`depth` slices back to back — measured:
/// ASTRO.BOT's 240x135x64 format-71 volumes); format 71 uploads as
/// `R16G16B16A16_SFLOAT` (8 B/texel) matching the recompiled `Rgba16f`
/// image, everything else keeps the RGBA8 view. The content is a
/// best-effort seed: a UAV is typically fully overwritten by the dispatch,
/// so an unknown format or unreadable guest range zero-fills with a
/// once-per-process warning instead of failing the dispatch.
fn read_storage_image(
    t: &kyty_graphics::shader::ShaderTextureResource,
) -> Result<StorageImageUpload, DrawError> {
    let width = u32::from(t.width5()) + 1;
    let height = u32::from(t.height5()) + 1;
    let depth = if t.type_() == 10 {
        u32::from(t.depth()) + 1
    } else {
        1
    };
    if !(1..=16384).contains(&width)
        || !(1..=16384).contains(&height)
        || !(1..=2048).contains(&depth)
    {
        return Err(err(format!(
            "storage image extent {width}x{height}x{depth} out of range"
        )));
    }
    // Must agree with `kyty-graphics` `storage_texture_dim_format` (which
    // declares `%ImageL` as Rgba16f exactly for guest format 71).
    let (format, texel) = if t.format() == 71 {
        (vk::Format::R16G16B16A16_SFLOAT, 8u64)
    } else {
        (vk::Format::R8G8B8A8_UNORM, 4u64)
    };
    let size = u64::from(width) * u64::from(height) * u64::from(depth) * texel;
    let base = t.base40();
    let readable = t.format() == 71 || storage_image_format_is_32bpp(t.format());
    let pixels = if readable {
        // Linear read: UAV surfaces the title dispatches into are addressed
        // by the shader itself, so no de-tiling is applied to the seed.
        read_guest_bytes(base, size, "storage image").ok()
    } else {
        None
    };
    let pixels = match pixels {
        Some(p) => p,
        None => {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    base = format_args!("{base:#x}"),
                    extent = format_args!("{width}x{height}x{depth}"),
                    format = t.format(),
                    readable,
                    "storage image initial content unavailable (unknown format \
                     or unreadable guest range) — zero-filling; the compute shader \
                     typically overwrites the whole UAV"
                );
            }
            // Fallible: the zero seed is full-resolution too; degrade to a named
            // skip under host memory pressure rather than aborting.
            alloc_zeroed(size as usize, "storage image seed")?
        }
    };
    Ok(StorageImageUpload {
        width,
        height,
        depth,
        format,
        pixels,
        guest_base: base,
    })
}

/// What the texture-decode path may consult about live render targets while
/// one draw's stage bindings are prepared:
///
/// - `map`: the CPU-side framebuffer map (per-target pixels of the last
///   readback) — the fallback for the feedback loop (a draw sampling its own
///   attachment) and for extent/format mismatches.
/// - `live`: `(base, width, height, vk format raw)` of every persistent GPU
///   target whose image content is trustworthy — a matching sampled T# binds
///   that `VkImage` directly (stage B) instead of round-tripping pixels.
/// - `self_base`: the draw's own `CB_COLOR0_BASE`. Never GPU-bound (binding
///   the current attachment as a texture is a feedback loop); takes the CPU
///   fallback instead.
struct SamplingScope {
    map: *const HashMap<u64, RenderedImage>,
    live: Vec<(u64, u32, u32, i32)>,
    self_base: u64,
    /// Snapshot of the persistent-texture cache (stage D): every cached
    /// texture's key and content sample-hash. `decode_texture` consults it to
    /// skip the guest read + detile + upload for a texture whose fresh
    /// sample-hash matches; empty when the cache is empty or disabled.
    cached_textures: Vec<(crate::vulkan::cache::TextureKey, u64)>,
}

thread_local! {
    /// The scope for the draw currently preparing its stage bindings.
    /// `OffscreenDrawSink::draw_common` publishes it around the
    /// `prepare_stage_binding` loop (where textures are decoded — NOT around
    /// `render_draw`, which decodes nothing) and restores the previous value
    /// after. Null outside that span.
    ///
    /// (Its predecessor, `RENDER_TARGETS`, was published around `render_draw`
    /// instead — after every texture had already been decoded — so the
    /// render-target-as-texture substitution it existed for never actually
    /// fired.)
    static SAMPLING_SCOPE: std::cell::Cell<*const SamplingScope> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

/// Publish `scope` for the duration of `f` (one draw's stage-binding
/// preparation), then restore the previous value. See [`SAMPLING_SCOPE`].
fn with_sampling_scope<R>(scope: &SamplingScope, f: impl FnOnce() -> R) -> R {
    let prev = SAMPLING_SCOPE.with(|c| c.replace(std::ptr::from_ref(scope)));
    let r = f();
    SAMPLING_SCOPE.with(|c| c.set(prev));
    r
}

/// Run `f` with the current sampling scope, if one is published.
fn sampling_scope<R>(f: impl FnOnce(&SamplingScope) -> Option<R>) -> Option<R> {
    SAMPLING_SCOPE.with(|c| {
        let ptr = c.get();
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `with_sampling_scope` sets this to a live `&SamplingScope`
        // for exactly the synchronous, same-thread span of the stage-binding
        // preparation and restores it after; the scope (and the framebuffer
        // map it points to) is not mutated during that span, and access here
        // is read-only.
        f(unsafe { &*ptr })
    })
}

/// A [`TextureUpload`] that binds the persistent GPU image of a live render
/// target directly, when this T# names one (stage B). `None` falls through to
/// the guest-memory decode / CPU-pixels fallback.
fn sampled_render_target(
    t: &kyty_graphics::shader::ShaderTextureResource,
) -> Option<TextureUpload> {
    // Only a plain 2D T# can alias a colour attachment.
    if !matches!(t.type_(), 8 | 9) {
        return None;
    }
    let width = u32::from(t.width5()) + 1;
    let height = u32::from(t.height5()) + 1;
    let base = t.base40();
    let format = texture_vk_format(t).ok()?.0;
    sampling_scope(|scope| {
        if base == scope.self_base {
            // Feedback loop: the CPU-pixels fallback handles it.
            return None;
        }
        scope
            .live
            .iter()
            .find(|(b, w, h, f)| *b == base && *w == width && *h == height && *f == format.as_raw())
            .map(|_| TextureUpload {
                width,
                height,
                format,
                pixels: Vec::new(),
                layers: 1,
                cube: false,
                depth: 1,
                render_target: Some(base),
                // Render-target binds are served by the persistent-TARGET
                // machinery; the texture cache plays no part.
                guest_base: 0,
                sample_hash: 0,
                cached: false,
            })
    })
}

/// The rendered pixels of the render target at guest `base` (matching `width`
/// x `height`), from the CPU-side framebuffer map, if one is live for the
/// current draw. This is the fallback for cases the direct GPU binding cannot
/// serve: the draw's own target (feedback loop) and extent/format mismatches.
/// The content lives in the framebuffer map, not the guest memory
/// `decode_texture` reads (render targets are never written back).
fn render_target_pixels(base: u64, width: u32, height: u32) -> Option<Vec<u8>> {
    sampling_scope(|scope| {
        // SAFETY: `scope.map` points at the sink's framebuffer map, alive and
        // unmutated for the published span (see `SAMPLING_SCOPE`).
        let map = unsafe { &*scope.map };
        map.get(&base)
            .filter(|img| img.width == width && img.height == height)
            .map(|img| img.pixels.clone())
    })
}

/// The byte size a V# addresses, per GNM/RDNA V# semantics: `stride == 0`
/// means a RAW buffer whose `num_records` IS the size in bytes; otherwise
/// the size is records × stride (shadPS4 `video_core/amdgpu/resource.h`
/// `Buffer::GetSize`, the RDNA2 authority).
fn buffer_byte_size(resource: &kyty_graphics::shader::ShaderBufferResource) -> Option<u64> {
    if resource.stride() == 0 {
        Some(u64::from(resource.num_records()))
    } else {
        u64::from(resource.stride()).checked_mul(u64::from(resource.num_records()))
    }
}

/// Cumulative decoded-pixel budget (bytes) for one stage's sampled textures
/// plus storage-image seeds, before the draw/dispatch is refused as oversized.
///
/// A title's final composite samples several full-resolution scene targets in a
/// SINGLE pass (measured: ASTRO.BOT decodes ~5 render-resolution RGBA16F / RGBA8
/// / R32F targets in one compute dispatch). Each target is uploaded from guest
/// memory — the compute path WRITES them as storage images to guest memory and
/// the composite RE-READS the raw bytes, often at a DIFFERENT extent/format than
/// they were written (measured: base `0x53a500000` written 1920x1080 R8, sampled
/// 960x540 R32F — the same bytes reinterpreted), so a single persistent VkImage
/// cannot alias them and the guest-memory decode is semantically required. Their
/// decoded buffers plus the matching Vulkan staging copies all coexist as host
/// allocations; at 2432x1368 the peak reaches a few hundred MiB, and under host
/// commit pressure a single 26 MiB (2432x1368x8) RGBA16F allocation can fail and
/// abort the process. Bounding the per-stage decoded total refuses the
/// pathological composite (a named, counted skip) BEFORE the allocation, instead
/// of letting the process abort. Composites whose samples fit run unchanged.
///
/// Tunable via `XPS5X_MAX_STAGE_TEXTURE_MIB`; default 96 MiB.
fn stage_texture_byte_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("XPS5X_MAX_STAGE_TEXTURE_MIB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&mib| mib > 0)
            .unwrap_or(96)
            .saturating_mul(1024 * 1024)
    })
}

/// Process-wide count of draws/dispatches refused by [`stage_texture_byte_cap`],
/// surfaced in the run summary so a skipped composite is visible, not silent.
static STAGE_TEXTURE_CAP_SKIPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Total draws/dispatches skipped for exceeding the per-stage sampled-texture
/// byte budget (see [`stage_texture_byte_cap`]).
#[must_use]
pub fn stage_texture_cap_skips() -> u64 {
    STAGE_TEXTURE_CAP_SKIPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Expected decoded (linear) byte size of one sampled T#, computed from the
/// descriptor WITHOUT decoding it: `width * height * bpp`, times the array-layer
/// count (cube / 2DArray) or the volume depth (3D). Used to bound a stage's
/// cumulative decode before the allocation happens (matches `decode_texture`'s
/// output size).
fn expected_sampled_bytes(t: &kyty_graphics::shader::ShaderTextureResource) -> u64 {
    let width = u64::from(u32::from(t.width5()) + 1);
    let height = u64::from(u32::from(t.height5()) + 1);
    let bpp = texture_vk_format(t).map_or(4, |(_, b)| u64::from(b));
    // 3D volume (type 10) is `depth` slices; cube (11) / 2DArray (13) is
    // `depth + 1` layers; everything else is a single 2D image.
    let extent = match t.type_() {
        10 | 11 | 13 => u64::from(u32::from(t.depth()) + 1),
        _ => 1,
    };
    width
        .saturating_mul(height)
        .saturating_mul(bpp)
        .saturating_mul(extent)
}

/// Expected initial-content byte size of one storage-image (UAV) T#: the linear
/// volume `width * height * depth * texel` (`texel` = 8 for format 71 RGBA16F,
/// else 4), matching `read_storage_image`.
fn expected_storage_image_bytes(t: &kyty_graphics::shader::ShaderTextureResource) -> u64 {
    let width = u64::from(u32::from(t.width5()) + 1);
    let height = u64::from(u32::from(t.height5()) + 1);
    let depth = if t.type_() == 10 {
        u64::from(u32::from(t.depth()) + 1)
    } else {
        1
    };
    let texel = if t.format() == 71 { 8 } else { 4 };
    width
        .saturating_mul(height)
        .saturating_mul(depth)
        .saturating_mul(texel)
}

/// Desired raw EUD-window snapshot size in bytes (SharpEmu port): at least
/// the shader's required prefix, floored at 256 KiB, page-rounded up, capped
/// at 16 MiB (`reference/sharpemu/src/SharpEmu.ShaderCompiler/`
/// `Gen5ShaderScalarEvaluator.cs:1952-1960` — the 256 KiB/page-round window —
/// and `:69` — `MaxGlobalMemoryBindingBytes` = 16 MiB).
fn eud_raw_window_want_bytes(required_dwords: u32) -> u64 {
    const MIN_BYTES: u64 = 256 * 1024;
    const MAX_BYTES: u64 = 16 * 1024 * 1024;
    u64::from(required_dwords)
        .saturating_mul(4)
        .max(MIN_BYTES)
        .next_multiple_of(4096)
        .min(MAX_BYTES)
}

/// Snapshot the guest window behind the EUD base pointer for the `%eud_raw`
/// SSBO. Ports SharpEmu's halving probe (`TryReadGlobalMemory`,
/// `Gen5ShaderScalarEvaluator.cs:997-1005`: try the full window, halve down
/// to one page) with its degrade-to-zero contract (`:14-35`): an unreadable
/// pointer yields a zero-filled buffer and `false` — never a refusal. The
/// returned bytes are a non-zero dword multiple; a snapshot shorter than the
/// shader's furthest offset is fine — the recompiled reads clamp against the
/// bound size and yield 0 beyond it.
///
/// `read` is injected (addr, byte length → bytes) so the sizing/degrade
/// logic is testable without live guest memory.
fn snapshot_eud_raw_window(
    base: u64,
    required_dwords: u32,
    read: impl Fn(u64, u64) -> Option<Vec<u8>>,
) -> (Vec<u8>, bool) {
    let required_bytes = usize::try_from(u64::from(required_dwords.max(1)) * 4)
        .unwrap_or(4)
        .max(4);
    if base == 0 || !base.is_multiple_of(4) {
        return (vec![0u8; required_bytes], false);
    }
    let want = eud_raw_window_want_bytes(required_dwords);
    let mut size = want;
    while size >= 4096 {
        if let Some(bytes) = read(base, size) {
            return (bytes, true);
        }
        size /= 2;
    }
    // Below one page: try the exact required prefix before degrading —
    // a tiny readable EUD (a few dwords) is still better than zeros.
    if let Some(bytes) = read(base, required_bytes as u64) {
        return (bytes, true);
    }
    (vec![0u8; required_bytes], false)
}

/// Build the [`EudRawBinding`] for a stage whose recompiled shader declares
/// the raw EUD-window fallback. Unreadable windows degrade to zeros with a
/// once-per-EUD-base warning (SharpEmu's one-shot trace gate,
/// `Gen5ShaderScalarEvaluator.cs:36-37`) — never a skipped dispatch.
fn prepare_eud_raw_binding(bind: &ShaderBindResources) -> EudRawBinding {
    let base = bind.extended.data.base();
    let required_dwords = bind.eud_raw.required_dwords;
    let (bytes, readable) = snapshot_eud_raw_window(base, required_dwords, |addr, len| {
        debug_assert!(len.is_multiple_of(4));
        // Validate before reading: a failed probe must not charge the
        // per-submission guest-byte budget (the halving ladder tries several
        // sizes per dispatch).
        if crate::guest_mem::readable_prefix(addr, len) != len {
            return None;
        }
        let count = u32::try_from(len / 4).ok()?;
        crate::guest_mem::read_dwords_validated(addr, count)
            .map(|words| words.into_iter().flat_map(u32::to_le_bytes).collect())
    });
    if !readable {
        use std::collections::HashSet;
        use std::sync::Mutex;
        static WARNED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
        let first = WARNED
            .lock()
            .map(|mut set| set.get_or_insert_with(HashSet::new).insert(base))
            .unwrap_or(false);
        if first {
            tracing::warn!(
                eud_base = format_args!("{base:#x}"),
                required_dwords,
                "raw EUD window is unreadable — binding zeros (degrade, not refuse); \
                 further hits on this base are silent"
            );
        }
    }
    EudRawBinding {
        binding: bind.eud_raw.binding_index.max(0) as u32,
        bytes,
    }
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
    // `bind.extended.used` (an EUD was recovered) is NOT a refusal: every
    // extended sharp's descriptor content was captured at analysis time
    // into the same resource tables as a direct sharp, so the loops below
    // bind it like any other — the recompiled shader reads it back through
    // the push-constant table via the `s_load_dwordx*` EUD translation.
    let gds_num = usize::try_from(bind.gds_pointers.pointers_num).map_err(|_| {
        err(format!(
            "negative gds pointer count {}",
            bind.gds_pointers.pointers_num
        ))
    })?;
    if gds_num > bind.gds_pointers.pointers.len() {
        return Err(err("gds pointer count exceeds fixed array"));
    }
    if gds_num != 0 && stage != vk::ShaderStageFlags::COMPUTE {
        return Err(err(format!(
            "translated {stage:?} shader uses GDS — GDS is implemented for \
             COMPUTE dispatches only"
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
        let size = buffer_byte_size(resource).ok_or_else(|| err("storage buffer size overflow"))?;
        let mut bytes = if resource.base48() == 0 || size == 0 {
            // A null V# (base 0 or zero byte size): RDNA out-of-bounds
            // semantics make every read return 0 and drop every write, and
            // titles legitimately dispatch shaders whose analysis bound such
            // a V# (measured: ASTRO.BOT compute at capture time). Bind a
            // 4-byte zero dummy so the recompiled SPIR-V's descriptor array
            // stays fully populated; the writeback skips it (`dispatch_direct`
            // checks addr/size, mirroring the dropped-write semantics).
            debug!(
                stage = ?stage,
                index,
                addr = format_args!("{:#x}", resource.base48()),
                size,
                "storage buffer V# is null — binding 4-byte zero dummy (RDNA OOB semantics)"
            );
            vec![0u8; 4]
        } else {
            read_guest_bytes(resource.base48(), size, "storage buffer")?
        };
        // The SSBO view is an array of 32-bit elements, so pad the upload to
        // a dword multiple (a V# byte size need not be one — the recompiler
        // dropped Kyty's alignment EXIT). The writeback truncates back to
        // `size` so the pad bytes never reach guest memory.
        let padded = bytes.len().div_ceil(4) * 4;
        bytes.resize(padded, 0);
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
    // Mixed-key sampled routing: the recompiled SPIR-V declares one
    // `%textures2D_S<key>` array per present sampled (Dim, numeric class)
    // key, each at its own binding. Each sampled T#'s seeded index is its
    // position WITHIN its own key's array (0..count-of-that-key), and
    // `sampled_key_views[ord]` records which `textures` entry fills each
    // slot. Indexed by the canonical key ordinal `sampled_key_ordinal` — the
    // same order the SPIR-V generator and `shader_calc_binding_indices` use
    // to assign per-key bindings.
    let mut sampled_key_count = [0u64; SAMPLED_KEYS];
    let mut sampled_key_views: [Vec<usize>; SAMPLED_KEYS] =
        std::array::from_fn(|_| Vec::new());
    // Per-stage decoded-byte budget (see `stage_texture_byte_cap`): a composite
    // sampling several full-resolution scene targets decodes them all from guest
    // memory at once, and the peak host allocation can abort the process. Refuse
    // (skip) BEFORE the over-budget allocation; targets served by direct
    // persistent-target binds (`render_target`, empty pixels) cost nothing.
    let texture_byte_cap = stage_texture_byte_cap();
    let mut stage_decoded_bytes: u64 = 0;
    let refuse_over_cap = |bytes: u64, stage: vk::ShaderStageFlags| -> DrawError {
        STAGE_TEXTURE_CAP_SKIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        err(format!(
            "translated {stage:?} stage decodes {bytes} B of sampled/storage textures, over the \
             {texture_byte_cap} B per-stage cap (XPS5X_MAX_STAGE_TEXTURE_MIB) — refusing the \
             draw/dispatch so the composite's simultaneous full-res uploads cannot exhaust host \
             memory"
        ))
    };
    for (index, desc) in bind.textures2d.desc[..texture_num].iter().enumerate() {
        let mut rewritten = desc.texture;
        if desc.usage == ShaderTextureUsage::ReadWrite {
            stage_decoded_bytes =
                stage_decoded_bytes.saturating_add(expected_storage_image_bytes(&desc.texture));
            if stage_decoded_bytes > texture_byte_cap {
                return Err(refuse_over_cap(stage_decoded_bytes, stage));
            }
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
            // A T# naming a live persistent render target binds that target's
            // GPU image directly (stage B) — its content lives on the device,
            // not in the guest memory decode_texture reads (render targets are
            // never written back). Without this, a title's final composite (a
            // fullscreen quad sampling its scene targets) reads black, so
            // nothing shows on screen.
            let mut decoded = match sampled_render_target(&desc.texture) {
                Some(upload) => upload,
                None => {
                    // Guest-memory decode: count it against the per-stage budget
                    // and refuse before allocating if the composite's samples
                    // would exceed the cap (a direct persistent-target bind
                    // above costs nothing and never reaches here).
                    stage_decoded_bytes =
                        stage_decoded_bytes.saturating_add(expected_sampled_bytes(&desc.texture));
                    if stage_decoded_bytes > texture_byte_cap {
                        return Err(refuse_over_cap(stage_decoded_bytes, stage));
                    }
                    decode_texture(&desc.texture)?
                }
            };
            // CPU fallback for what the direct binding cannot serve (the
            // draw's own target — a feedback loop — or an extent/format
            // mismatch): substitute the framebuffer map's rendered pixels.
            if decoded.render_target.is_none()
                && let Some(px) =
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
            let ord = sampled_key_ordinal(&desc.texture);
            let view_index = textures.len();
            // Seed the T# index WITHIN its key's array (homogeneous shaders
            // have one key, so this counts 0,1,2… exactly as the old
            // `textures.len()` did).
            rewritten.update_address38(sampled_key_count[ord]);
            sampled_key_count[ord] += 1;
            sampled_key_views[ord].push(view_index);
            textures.push(decoded);
        }
        for field in rewritten.fields {
            push_constants.extend_from_slice(&field.to_le_bytes());
        }
    }
    // A shader binds more than one sampled (Dim, class) key => build the
    // per-key groups the host descriptor path uses (one array per key).
    // Present keys are taken in canonical ordinal order and assigned
    // consecutive bindings starting at `binding_sampled_index`, matching the
    // SPIR-V generator exactly. A homogeneous shader leaves `sampled_groups`
    // empty (single-array path).
    let present_keys: Vec<usize> = (0..SAMPLED_KEYS)
        .filter(|&o| !sampled_key_views[o].is_empty())
        .collect();
    let sampled_groups: Vec<SampledGroup> = if present_keys.len() > 1 {
        present_keys
            .iter()
            .enumerate()
            .map(|(pos, &ord)| SampledGroup {
                binding: bind.textures2d.binding_sampled_index as u32 + pos as u32,
                view_indices: std::mem::take(&mut sampled_key_views[ord]),
            })
            .collect()
    } else {
        Vec::new()
    };
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

    // GDS pointers: one raw dword each (the base/size field the guest loaded
    // into the pointer SGPR), packed 4 per 16-byte push-constant granule in
    // the exact `WriteLocalVariables` order — after the S#s, before the
    // direct SGPRs. The GDS arena itself is the device-persistent buffer the
    // Vulkan layer binds at `gds_pointers.binding_index`.
    if gds_num != 0 {
        let base = push_constants.len();
        let granules = (gds_num - 1) / 4 + 1;
        push_constants.resize(base + granules * 16, 0);
        for (i, pointer) in bind.gds_pointers.pointers[..gds_num].iter().enumerate() {
            let at = base + i * 4;
            push_constants[at..at + 4].copy_from_slice(&pointer.field.to_le_bytes());
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
        // Present when EITHER array is non-empty: a shader legitimately
        // binds textures without samplers (texel fetch) or samplers without
        // sampled textures — the Vulkan layer creates each descriptor array
        // independently, exactly as the SPIR-V declared them.
        textures: (!textures.is_empty() || !linear_filter.is_empty()).then_some(TextureBinding {
            sampled_binding: bind.textures2d.binding_sampled_index as u32,
            sampler_binding: bind.samplers.binding_index as u32,
            textures,
            linear_filter,
            sampled_groups,
        }),
        storage_images: (!storage_images.is_empty()).then_some(StorageImageBinding {
            binding: bind.textures2d.binding_storage_index as u32,
            images: storage_images,
        }),
        gds_binding: (gds_num != 0).then_some(bind.gds_pointers.binding_index as u32),
        // The raw EUD-window snapshot (SharpEmu port): read at dispatch time
        // from the captured EUD base pointer; unreadable degrades to zeros.
        eud_raw: bind.eud_raw.used.then(|| prepare_eud_raw_binding(bind)),
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
        prim::POINT_LIST => (vk::PrimitiveTopology::POINT_LIST, index_count),
        prim::LINE_LIST => (vk::PrimitiveTopology::LINE_LIST, index_count),
        prim::LINE_STRIP => (vk::PrimitiveTopology::LINE_STRIP, index_count),
        other => {
            return Err(err(format!(
                "unsupported VGT_PRIMITIVE_TYPE {other} (supported: 1 PointList, \
                 2 LineList, 3 LineStrip, 4 TriList, 5 TriFan, 6 TriStrip, \
                 7 Polygon, 17 RectList)"
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
        // The caller (draw_common) names the guest render target so the
        // backend can keep its VkImage alive across draws.
        target_base: None,
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
    /// `CB_COLOR0_BASE` of the last draw's render target. With deferred
    /// readback (stage B) `last` is only populated by immediate-fallback
    /// draws; the session resolves the presented frame by looking this base
    /// up in the framebuffer map AFTER the flush lands the batch's pixels.
    pub last_target: Option<u64>,
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
            last_target: None,
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
        // Internal-resolution scaling (Settings ▸ Video ▸ Resolution Scale).
        // Supersamples the whole draw (target + viewport + scissor together);
        // a factor of 1.0 — the default — is an exact no-op.
        state.scale_resolution(crate::agc_exec::AgcGpuSession::runtime_config().resolution_scale);
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

        let rt_base = ctx.render_targets[0].base.addr;
        let stage_binds = [
            (&shaders.vs_info.bind, vk::ShaderStageFlags::VERTEX),
            (&shaders.ps_info.bind, vk::ShaderStageFlags::FRAGMENT),
        ];
        // Feedback loop: a stage samples the very target this draw renders
        // into. That T# takes the CPU-pixels fallback, and with deferred
        // readback (stage B) those pixels can be stale — flush the pending
        // batch first so the framebuffer map is current. Named, counted via
        // the flush stats; measured composites sample OTHER targets, so this
        // stays off the hot path.
        let samples_own_target = stage_binds.iter().any(|(bind, _)| {
            let n = usize::try_from(bind.textures2d.textures_num).unwrap_or(0);
            bind.textures2d.desc[..n.min(bind.textures2d.desc.len())]
                .iter()
                .any(|d| d.usage != ShaderTextureUsage::ReadWrite && d.texture.base40() == rt_base)
        });
        if samples_own_target && self.dev.draw_caches().base_is_batch_dirty(rt_base) {
            self.flush_deferred_into_framebuffers()?;
        }
        // Publish what the texture decode may consult: live persistent GPU
        // targets (bindable directly) and the CPU framebuffer map (feedback /
        // mismatch fallback). The draw's own target is excluded from direct
        // binding — sampling the current attachment is the feedback loop.
        // ONE lock acquisition for both snapshots: two `draw_caches()`
        // temporaries in a single expression would deadlock — the first
        // guard lives to the end of the statement while the second lock()
        // waits on it.
        let (live, cached_textures) = {
            let caches = self.dev.draw_caches();
            (
                caches
                    .live_target_keys()
                    .into_iter()
                    .filter(|k| k.base != rt_base)
                    .map(|k| (k.base, k.width, k.height, k.format))
                    .collect(),
                // Persistent-texture cache snapshot (stage D): lets the
                // texture decode skip the guest read + detile + upload for
                // any texture whose content sample-hash still matches the
                // cached image.
                caches.cached_texture_hashes(),
            )
        };
        let scope = SamplingScope {
            map: std::ptr::from_ref(self.framebuffers),
            live,
            self_base: rt_base,
            cached_textures,
        };
        with_sampling_scope(&scope, || -> Result<(), DrawError> {
            for (bind, stage) in stage_binds {
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
            Ok(())
        })?;

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
        let prior = self
            .framebuffers
            .remove(&rt_base)
            .filter(|p| p.width == state.width && p.height == state.height);
        if let Some(p) = &prior {
            state.initial = Some(&p.pixels);
        }
        // Name the guest target so the backend keeps one VkImage per
        // (base, extent, format) across draws. The `target_base` contract
        // holds here by construction: `prior` is exactly the previous
        // readback of this target (this map is only ever written with
        // readbacks), so the backend may LOAD the persistent GPU copy
        // instead of re-uploading these bytes.
        state.target_base = Some(rt_base);
        // Stage B: the draw is submitted with its readback DEFERRED —
        // `Ok(None)` means the pixels land in the framebuffer map at the next
        // flush (end of submission, presentation, or a feedback fallback).
        // `Ok(Some(image))` is the immediate-fallback path (readback now),
        // preserving the old per-draw behaviour.
        let immediate = crate::vulkan::offscreen::render_draw_deferred(self.dev, &state)
            .map_err(|e| err(format!("offscreen draw failed: {e}")))?;
        drop(state);
        match immediate {
            Some(image) => {
                self.framebuffers.insert(rt_base, image.clone());
                self.last = Some(image);
            }
            None => {
                // Deferred: keep the (now-stale) prior entry so the map's key
                // census stays complete until the flush replaces it. The
                // target itself is marked GPU-newer, so no path can mistake
                // these bytes for the current frame's authority.
                if let Some(p) = prior {
                    self.framebuffers.insert(rt_base, p);
                }
            }
        }
        self.last_target = Some(rt_base);
        self.draws += 1;
        Ok(())
    }

    /// Flush the pending deferred-draw batch and land every readback in the
    /// framebuffer map (the feedback-loop fallback path).
    fn flush_deferred_into_framebuffers(&mut self) -> Result<(), DrawError> {
        let flushed = crate::vulkan::offscreen::flush_deferred_draws(self.dev)
            .map_err(|e| err(format!("deferred-draw flush failed: {e}")))?;
        for (base, image) in flushed {
            self.framebuffers.insert(base, image);
        }
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
        // Forensic kill switch: `XPS5X_SKIP_CS=0xADDR[,0xADDR...]` skips the
        // named compute programs as a counted, named degradation. Used to
        // bisect device-loss culprits: a lethal dispatch resets the whole
        // device and every later draw in the session fails, so isolating one
        // program by address is the fastest way to pin the killer.
        if let Ok(list) = std::env::var("XPS5X_SKIP_CS") {
            let addr = format!("{:#x}", cs.cs_regs.data_addr);
            if list
                .split(',')
                .any(|s| s.trim().eq_ignore_ascii_case(&addr))
            {
                self.dispatch_skips += 1;
                self.last_dispatch_skip_reason = Some(format!("XPS5X_SKIP_CS: {addr}"));
                debug!(cs_addr = %addr, "compute dispatch skipped by XPS5X_SKIP_CS");
                return Ok(());
            }
        }
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
        // The round-10 tex-no-sampler quarantine is GONE. The measured
        // device loss (0x5006c5f00: sampled textures + zero samplers +
        // runtime-loaded T# via `s_load_dwordx8`) was descriptor-array OOB
        // indexing, now structurally defused at translate time:
        // - `mimg_descriptor_guard` (kyty-graphics) refuses any MIMG whose
        //   T#/S# registers are not a captured descriptor, or are
        //   overwritten by a raw (uncovered-EUD) `s_load`, with the named
        //   `dynamic-image-descriptor` skip (SharpEmu parity);
        // - `shader_synthesize_default_sampler` rescues sample-family
        //   shaders with zero captured S#s via a cached nearest/wrap
        //   default sampler instead of a refusal;
        // - every LDS access is index-clamped in the emitted SPIR-V.
        // Texel-fetch shaders with zero samplers are legitimate and run.
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
        // (base address, real V# byte size) — the Vulkan buffer may carry up
        // to 3 pad bytes (dword-aligned upload), which must never be written
        // back over guest memory beyond the V#.
        let guest_outputs: Vec<(u64, usize)> = bind.storage_buffers.buffers[..storage_num]
            .iter()
            .map(|resource| {
                (
                    resource.base48(),
                    buffer_byte_size(resource).unwrap_or(0) as usize,
                )
            })
            .collect();
        // Storage-image guest bases, collected pre-dispatch in the same order
        // `ComputeOutputs::images` returns them.
        // (guest base, width, height, vk format) per output image, in the same
        // order `ComputeOutputs::images` returns them — carries enough to
        // register a content-bearing writeback as a presentable census entry.
        let guest_image_outputs: Vec<(u64, u32, u32, vk::Format)> = prepared
            .as_ref()
            .and_then(|binding| binding.storage_images.as_ref())
            .map(|images| {
                images
                    .images
                    .iter()
                    .map(|img| (img.guest_base, img.width, img.height, img.format))
                    .collect()
            })
            .unwrap_or_default();
        // Forensic breadcrumb: device loss surfaces LAZILY (the next
        // vkQueueSubmit reports it), so identifying a lethal dispatch needs
        // the pre-submit identity of every dispatch in the log.
        debug!(
            cs_addr = format_args!("{:#x}", cs.cs_regs.data_addr),
            groups = format_args!("{}x{}x{}", groups[0], groups[1], groups[2]),
            storage_num,
            null_vsharps = guest_outputs
                .iter()
                .filter(|(addr, len)| *addr == 0 || *len == 0)
                .count(),
            gds = prepared.as_ref().is_some_and(|p| p.gds_binding.is_some()),
            textures = prepared
                .as_ref()
                .and_then(|p| p.textures.as_ref())
                .map_or(0, |t| t.textures.len()),
            samplers = prepared
                .as_ref()
                .and_then(|p| p.textures.as_ref())
                .map_or(0, |t| t.linear_filter.len()),
            images = prepared
                .as_ref()
                .and_then(|p| p.storage_images.as_ref())
                .map_or(0, |i| i.images.len()),
            "compute dispatch submitting"
        );
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
        for ((addr, real_len), bytes) in guest_outputs.into_iter().zip(outputs.buffers) {
            // A null V# (base 0 or zero size) was bound as a zero dummy;
            // hardware drops its writes (RDNA OOB semantics), so skip the
            // writeback explicitly — `write_bytes_checked` would refuse
            // address 0 anyway and fail the whole dispatch.
            if addr == 0 || real_len == 0 {
                debug!(
                    addr = format_args!("{addr:#x}"),
                    real_len, "compute storage writeback skipped: null V# (writes dropped)"
                );
                continue;
            }
            // Truncate off the dword-alignment pad (see `guest_outputs`).
            let bytes = &bytes[..bytes.len().min(real_len)];
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
            crate::guest_mem::trace_scanout_fill(addr, bytes.len(), "compute-storage");
            if !crate::guest_mem::write_bytes_checked(addr, bytes) {
                return Err(err(format!(
                    "compute storage writeback range {addr:#x}..{:#x} is not writable guest memory",
                    addr.saturating_add(bytes.len() as u64)
                )));
            }
        }
        for ((addr, img_w, img_h, img_format), bytes) in
            guest_image_outputs.into_iter().zip(outputs.images)
        {
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
            crate::guest_mem::trace_scanout_fill(addr, bytes.len(), "compute-image");
            // Promote a content-bearing 8-bit UAV writeback into the present
            // census: ASTRO-class titles compose their scene with compute
            // dispatches into guest memory — never a GPU render pass we capture
            // — so without this the frame the census elects is always some flat
            // cleared draw target and the real pixels stay invisible. Only
            // R8G8B8A8 is promoted (already the Shell's RGBA byte order); HDR
            // (R16F) intermediates are left to the draw/scanout paths. Keyed by
            // guest base so a re-dispatch to the same UAV replaces in place.
            if nonzero {
                // R8G8B8A8 is already the Shell's RGBA byte order; the HDR
                // R16G16B16A16_SFLOAT scene/composite buffers are sRGB-encoded to
                // RGBA8 by `to_presentable` at present time (bytes_per_pixel == 8
                // is its float-target signal). Other formats are left alone.
                let bpp = match img_format {
                    vk::Format::R8G8B8A8_UNORM => 4usize,
                    vk::Format::R16G16B16A16_SFLOAT => 8usize,
                    _ => 0,
                };
                let want = img_w as usize * img_h as usize * bpp;
                // Cap at 128 MiB (4K RGBA16F = 66 MiB) so a mis-sized image never
                // triggers an absurd copy.
                if bpp != 0 && want > 0 && want <= (128 << 20) && bytes.len() >= want {
                    self.framebuffers.insert(
                        addr,
                        RenderedImage {
                            width: img_w,
                            height: img_h,
                            pixels: bytes[..want].to_vec(),
                            bytes_per_pixel: bpp as u32,
                        },
                    );
                }
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

    /// GNM V# byte-size semantics: `stride == 0` marks a RAW buffer whose
    /// `num_records` is the size in BYTES (shadPS4 `Buffer::GetSize`); any
    /// other stride multiplies. An odd total is legitimate — the upload pads
    /// and the writeback truncates.
    #[test]
    fn buffer_byte_size_follows_gnm_v_sharp_semantics() {
        use kyty_graphics::shader::ShaderBufferResource;
        // stride 12, 10 records -> 120 bytes.
        let mut typed = ShaderBufferResource::default();
        typed.fields[1] = 12 << 16;
        typed.fields[2] = 10;
        assert_eq!(buffer_byte_size(&typed), Some(120));
        // stride 0 = raw buffer: num_records IS the byte size (the old
        // `stride * num_records` computed 0 here and bound nothing).
        let mut raw = ShaderBufferResource::default();
        raw.fields[2] = 123;
        assert_eq!(buffer_byte_size(&raw), Some(123));
        // stride 2 with 7 records: 14 bytes — NOT a dword multiple, still a
        // valid V# (previously refused by the recompiler's alignment gate).
        let mut odd = ShaderBufferResource::default();
        odd.fields[1] = 2 << 16;
        odd.fields[2] = 7;
        assert_eq!(buffer_byte_size(&odd), Some(14));
    }

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

    /// Point and line primitives map 1:1 to their Vulkan topologies and keep the
    /// guest index count. ASTRO.BOT issues point-list draws in its render loop;
    /// before support they skipped as "unsupported VGT_PRIMITIVE_TYPE 1".
    #[test]
    fn point_and_line_primitives_map_to_vulkan_topologies() {
        for (prim_type, expected) in [
            (prim::POINT_LIST, vk::PrimitiveTopology::POINT_LIST),
            (prim::LINE_LIST, vk::PrimitiveTopology::LINE_LIST),
            (prim::LINE_STRIP, vk::PrimitiveTopology::LINE_STRIP),
        ] {
            let ucfg = UserConfig {
                prim_type,
                ..UserConfig::default()
            };
            let state = draw_state_from_regs(&ctx_96x48(), &ucfg, 6, SPIRV, SPIRV)
                .unwrap_or_else(|e| panic!("primitive {prim_type}: {e}"));
            assert_eq!(state.topology, expected, "primitive {prim_type}");
            assert_eq!(
                state.vertex_count, 6,
                "primitive {prim_type} keeps the guest index count"
            );
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

    /// The pre-decode size estimates must match what `decode_texture` /
    /// `read_storage_image` actually allocate, so the per-stage budget refuses
    /// at the right threshold.
    #[test]
    fn expected_texture_bytes_match_decoder_output() {
        // 2D RGBA8 4x4 = 64 B, both as a sampled texture and a storage seed.
        let t2d = rgba8_linear_tsharp(0x1000, 4, 4);
        assert_eq!(expected_sampled_bytes(&t2d), 4 * 4 * 4);
        assert_eq!(expected_storage_image_bytes(&t2d), 4 * 4 * 4);
        // A larger extent scales linearly: 1024x1024 RGBA8 = 4 MiB.
        let big = rgba8_linear_tsharp(0x1000, 1024, 1024);
        assert_eq!(expected_sampled_bytes(&big), 1024 * 1024 * 4);
    }

    /// A composite whose single sampled T# alone exceeds the per-stage byte cap
    /// is refused as a named, counted skip BEFORE any guest read or allocation —
    /// the guard that keeps a full-resolution multi-target composite from
    /// exhausting host memory and aborting the process.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn stage_texture_byte_cap_refuses_oversized_composite() {
        // 8192x8192 RGBA8 decodes to 256 MiB, over the 96 MiB default cap.
        let mut bind = ShaderBindResources::default();
        bind.push_constant_size = 32;
        bind.textures2d.textures_num = 1;
        bind.textures2d.textures2d_sampled_num = 1;
        bind.textures2d.binding_sampled_index = 0;
        bind.textures2d.desc[0].texture = rgba8_linear_tsharp(0x1000, 8192, 8192);

        assert!(
            expected_sampled_bytes(&bind.textures2d.desc[0].texture) > stage_texture_byte_cap(),
            "test texture must exceed the default cap"
        );
        let before = stage_texture_cap_skips();
        // No guest ranges installed: the refusal must fire before decode_texture
        // reads guest memory.
        let e = prepare_stage_binding(&bind, vk::ShaderStageFlags::FRAGMENT)
            .expect_err("oversized composite is refused");
        assert!(e.0.contains("per-stage cap"), "names the cap: {e}");
        assert_eq!(
            stage_texture_cap_skips(),
            before + 1,
            "the refusal is counted"
        );
    }

    /// A host allocation failure in the decode path degrades to a named
    /// `DrawError` (a skip) instead of aborting the process — the safety net
    /// that turns the measured "memory allocation of N bytes failed" crash into
    /// a recoverable skipped draw/dispatch.
    #[test]
    fn alloc_zeroed_degrades_on_failure_not_abort() {
        // A normal allocation succeeds and is zero-filled.
        let ok = alloc_zeroed(64, "test").expect("small alloc succeeds");
        assert_eq!(ok.len(), 64);
        assert!(ok.iter().all(|&b| b == 0));
        // An impossible reservation returns Err (a skip), never aborts.
        let e = alloc_zeroed(usize::MAX, "test").expect_err("huge alloc is refused");
        assert!(e.0.contains("out of memory"), "names the failure: {e}");
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

    /// The ASTRO.BOT composite/read-pass shape: a stage sampling one 2D
    /// texture AND one 3D volume. The recompiled SPIR-V declares one image
    /// array per Dim (2D at `binding_sampled_index`, 3D at the next binding),
    /// so `prepare_stage_binding` must split the sampled views into per-Dim
    /// groups and seed each T#'s index WITHIN its own Dim's array (both 0
    /// here — one texture per Dim).
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn compute_binding_mixed_dims_split_into_per_dim_groups() {
        const W: u32 = 4;
        const H: u32 = 4;
        const D: u32 = 2;
        const BYTES2D: usize = (W * H * 4) as usize;
        const BYTES3D: usize = (W * H * D * 4) as usize;

        let mut arena = vec![0u8; 1024 + 255];
        let base = (arena.as_ptr() as u64 + 255) & !255;
        let off = (base - arena.as_ptr() as u64) as usize;
        for i in 0..BYTES3D {
            arena[off + i] = (i % 251) as u8; // 2D at +0
            arena[off + 512 + i] = ((i * 5 + 11) % 251) as u8; // 3D volume at +512
        }
        let content = |o: usize, n: usize| arena[off + o..off + o + n].to_vec();

        // A 3D volume T# (type 10) with `depth = D` slices, linear tile.
        let mut vol = rgba8_linear_tsharp(base + 512, W, H);
        vol.fields[3] &= !(0xF << 28);
        vol.fields[3] |= 10 << 28; // type = 3D
        vol.fields[4] = (vol.fields[4] & !0x1FFF) | (D - 1); // depth field = D - 1

        let mut bind = ShaderBindResources::default();
        bind.push_constant_size = 2 * 32; // two 8-dword T#s
        bind.textures2d.textures_num = 2;
        bind.textures2d.textures2d_sampled_num = 2;
        bind.textures2d.binding_sampled_index = 0;
        // Two present Dims => two sampled bindings (0, 1); storage at 2.
        bind.textures2d.binding_storage_index = 2;
        bind.textures2d.desc[0].texture = rgba8_linear_tsharp(base, W, H); // 2D
        bind.textures2d.desc[1].texture = vol; // 3D

        let ranges = [(arena.as_ptr() as u64, arena.len())];
        let binding = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_stage_binding(&bind, vk::ShaderStageFlags::COMPUTE)
        })
        .expect("mixed-dim sampled compute binding");

        let textures = binding.textures.expect("sampled textures");
        assert_eq!(textures.textures.len(), 2);
        assert_eq!(textures.textures[0].pixels, content(0, BYTES2D));
        assert_eq!(textures.textures[1].pixels, content(512, BYTES3D));

        // Two per-Dim groups: 2D (ordinal 0) at binding 0, 3D (ordinal 2) at
        // binding 1, each holding its single view.
        assert_eq!(textures.sampled_groups.len(), 2);
        assert_eq!(textures.sampled_groups[0].binding, 0);
        assert_eq!(textures.sampled_groups[0].view_indices, vec![0]);
        assert_eq!(textures.sampled_groups[1].binding, 1);
        assert_eq!(textures.sampled_groups[1].view_indices, vec![1]);

        // Each T#'s seeded index is its position within its own Dim's array.
        let dword0 = |group: usize| {
            u32::from_le_bytes(
                binding.push_constants[group * 32..group * 32 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        assert_eq!(dword0(0), 0, "the 2D T# indexes %textures2D_S_2D[0]");
        assert_eq!(dword0(1), 0, "the 3D T# indexes %textures2D_S_3D[0]");
    }

    /// INTERLEAVED mixed dims — [2D, 3D, 2D] in analysis order — pin the
    /// per-dim RUNNING index: the second 2D T#'s seeded dword 0 must be its
    /// position within the 2D array (1), never its global sampled position
    /// (2), and each group's `view_indices` keep analysis order within the
    /// dim ([0, 2] for 2D, [1] for 3D). This is the shape where a global
    /// index would run past the smaller per-dim array (descriptor OOB, the
    /// measured `VK_ERROR_DEVICE_LOST` class).
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn compute_binding_mixed_dims_use_per_dim_running_indices() {
        const W: u32 = 4;
        const H: u32 = 4;
        const D: u32 = 2;
        const BYTES2D: usize = (W * H * 4) as usize;
        const BYTES3D: usize = (W * H * D * 4) as usize;

        // 256-aligned arena: 2D A at +0, 3D volume at +512, 2D B at +1024.
        let mut arena = vec![0u8; 1536 + 255];
        let base = (arena.as_ptr() as u64 + 255) & !255;
        let off = (base - arena.as_ptr() as u64) as usize;
        for i in 0..BYTES3D {
            arena[off + 512 + i] = ((i * 5 + 11) % 251) as u8; // 3D volume
        }
        for i in 0..BYTES2D {
            arena[off + i] = (i % 251) as u8; // 2D A
            arena[off + 1024 + i] = ((i * 3 + 7) % 251) as u8; // 2D B
        }
        let content = |o: usize, n: usize| arena[off + o..off + o + n].to_vec();

        let mut vol = rgba8_linear_tsharp(base + 512, W, H);
        vol.fields[3] &= !(0xF << 28);
        vol.fields[3] |= 10 << 28; // type = 3D
        vol.fields[4] = (vol.fields[4] & !0x1FFF) | (D - 1); // depth = D - 1

        let mut bind = ShaderBindResources::default();
        bind.push_constant_size = 3 * 32;
        bind.textures2d.textures_num = 3;
        bind.textures2d.textures2d_sampled_num = 3;
        bind.textures2d.binding_sampled_index = 0;
        // Two present dims => storage would sit past both (binding 2).
        bind.textures2d.binding_storage_index = 2;
        bind.textures2d.desc[0].texture = rgba8_linear_tsharp(base, W, H); // 2D A
        bind.textures2d.desc[1].texture = vol; // 3D
        bind.textures2d.desc[2].texture = rgba8_linear_tsharp(base + 1024, W, H); // 2D B

        let ranges = [(arena.as_ptr() as u64, arena.len())];
        let binding = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_stage_binding(&bind, vk::ShaderStageFlags::COMPUTE)
        })
        .expect("interleaved mixed-dim sampled compute binding");

        let textures = binding.textures.expect("sampled textures");
        assert_eq!(textures.textures.len(), 3, "flat pool keeps analysis order");
        assert_eq!(textures.textures[0].pixels, content(0, BYTES2D));
        assert_eq!(textures.textures[1].pixels, content(512, BYTES3D));
        assert_eq!(textures.textures[2].pixels, content(1024, BYTES2D));

        // Groups in dim-ordinal order; view_indices keep analysis order
        // WITHIN each dim.
        assert_eq!(textures.sampled_groups.len(), 2);
        assert_eq!(textures.sampled_groups[0].binding, 0);
        assert_eq!(
            textures.sampled_groups[0].view_indices,
            vec![0, 2],
            "the 2D array holds both 2D T#s in analysis order"
        );
        assert_eq!(textures.sampled_groups[1].binding, 1);
        assert_eq!(textures.sampled_groups[1].view_indices, vec![1]);

        // Seeded indices are per-dim running counters.
        let dword0 = |t: usize| {
            u32::from_le_bytes(
                binding.push_constants[t * 32..t * 32 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        assert_eq!(dword0(0), 0, "2D A indexes %textures2D_S_2D[0]");
        assert_eq!(dword0(1), 0, "the 3D T# indexes %textures2D_S_3D[0]");
        assert_eq!(
            dword0(2),
            1,
            "2D B indexes %textures2D_S_2D[1] — its dim-local position, not its global 2"
        );
    }

    /// The numeric class the SHADER translator declares its sampled image
    /// type with must agree with the VkFormat the VIEW is created in, for
    /// every unified format `texture_vk_format` implements — their divergence
    /// IS the measured VUID-vkCmdDispatch-format-07753 (unified 5 view =
    /// `R8_UINT`, shader type = `%float`). Sweeps the whole 9-bit unified
    /// format space so any future `texture_vk_format` arm is covered the day
    /// it lands.
    #[test]
    fn texture_vk_format_numeric_class_matches_shader_sampled_class() {
        for fmt in 0u16..512 {
            let mut t = kyty_graphics::shader::ShaderTextureResource::default();
            t.fields[1] |= u32::from(fmt) << 20;
            let Ok((vk_format, _)) = texture_vk_format(&t) else {
                continue; // unimplemented format — named error, nothing bound
            };
            let name = format!("{vk_format:?}");
            let view_class = if name.contains("UINT") {
                kyty_graphics::shader::SampledClass::Uint
            } else if name.contains("SINT") {
                kyty_graphics::shader::SampledClass::Sint
            } else {
                kyty_graphics::shader::SampledClass::Float
            };
            assert_eq!(
                kyty_graphics::shader::SampledClass::from_unified_format(fmt),
                view_class,
                "unified format {fmt} maps to {name} but the shader would \
                 declare a different sampled component class"
            );
        }
    }

    /// A stage sampling one FLOAT-class 2D texture AND one UINT-class 2D
    /// texture (the ASTRO.BOT R8_UINT shape): `prepare_stage_binding` must
    /// split the views into per-(Dim, class) groups — float at
    /// `binding_sampled_index`, uint at the next binding — and seed each T#'s
    /// index WITHIN its own class's array, mirroring the recompiled SPIR-V's
    /// `%textures2D_S_2D` / `%textures2D_S_2D_U` split.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn compute_binding_mixed_classes_split_into_per_class_groups() {
        const W: u32 = 4;
        const H: u32 = 4;
        const BYTES_RGBA8: usize = (W * H * 4) as usize;
        const BYTES_R8: usize = (W * H) as usize;

        let mut arena = vec![0u8; 1024 + 255];
        let base = (arena.as_ptr() as u64 + 255) & !255;
        let off = (base - arena.as_ptr() as u64) as usize;
        for i in 0..BYTES_RGBA8 {
            arena[off + i] = (i % 251) as u8; // RGBA8 float-class at +0
        }
        for i in 0..BYTES_R8 {
            arena[off + 512 + i] = ((i * 7 + 3) % 251) as u8; // R8_UINT at +512
        }
        let content = |o: usize, n: usize| arena[off + o..off + o + n].to_vec();

        // An R8_UINT T# (unified format 5), 2D, linear tile.
        let mut r8 = rgba8_linear_tsharp(base + 512, W, H);
        r8.fields[1] &= !(0x1FF << 20);
        r8.fields[1] |= 5 << 20; // unified 5 = R8 UINT

        let mut bind = ShaderBindResources::default();
        bind.push_constant_size = 2 * 32;
        bind.textures2d.textures_num = 2;
        bind.textures2d.textures2d_sampled_num = 2;
        bind.textures2d.binding_sampled_index = 0;
        // Two present keys => two sampled bindings (0, 1); storage at 2.
        bind.textures2d.binding_storage_index = 2;
        bind.textures2d.desc[0].texture = rgba8_linear_tsharp(base, W, H); // float
        bind.textures2d.desc[1].texture = r8; // uint

        let ranges = [(arena.as_ptr() as u64, arena.len())];
        let binding = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_stage_binding(&bind, vk::ShaderStageFlags::COMPUTE)
        })
        .expect("mixed-class sampled compute binding");

        let textures = binding.textures.expect("sampled textures");
        assert_eq!(textures.textures.len(), 2);
        assert_eq!(textures.textures[0].pixels, content(0, BYTES_RGBA8));
        assert_eq!(textures.textures[0].format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(textures.textures[1].pixels, content(512, BYTES_R8));
        assert_eq!(textures.textures[1].format, vk::Format::R8_UINT);

        // Two per-class groups: (2D, Float) at binding 0, (2D, Uint) at
        // binding 1, each holding its single view.
        assert_eq!(textures.sampled_groups.len(), 2);
        assert_eq!(textures.sampled_groups[0].binding, 0);
        assert_eq!(textures.sampled_groups[0].view_indices, vec![0]);
        assert_eq!(textures.sampled_groups[1].binding, 1);
        assert_eq!(textures.sampled_groups[1].view_indices, vec![1]);

        // Each T#'s seeded index is its position within its own class array.
        let dword0 = |group: usize| {
            u32::from_le_bytes(
                binding.push_constants[group * 32..group * 32 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        assert_eq!(dword0(0), 0, "the float T# indexes %textures2D_S_2D[0]");
        assert_eq!(dword0(1), 0, "the uint T# indexes %textures2D_S_2D_U[0]");
    }

    /// SharpEmu-parity window sizing (Gen5ShaderScalarEvaluator.cs:1952-1960
    /// and :69): max(need, 256 KiB), page-rounded up, capped at 16 MiB.
    #[test]
    fn eud_raw_window_sizing_min_round_cap() {
        // Tiny requirement floors at the 256 KiB minimum (already page-round).
        assert_eq!(eud_raw_window_want_bytes(1), 256 * 1024);
        assert_eq!(eud_raw_window_want_bytes(0), 256 * 1024);
        // Above the minimum: page-rounded up.
        assert_eq!(
            eud_raw_window_want_bytes(100_000), // 400_000 B
            400_000u64.next_multiple_of(4096)
        );
        // Enormous requirement caps at 16 MiB.
        assert_eq!(eud_raw_window_want_bytes(5_000_000), 16 * 1024 * 1024);
    }

    /// SharpEmu-parity snapshot probe (Gen5ShaderScalarEvaluator.cs:997-1005):
    /// halve from the wanted window down to one page, then the exact
    /// required prefix; an unreadable pointer degrades to zeros, never fails.
    #[test]
    fn eud_raw_snapshot_halves_and_degrades_to_zero() {
        // A "guest" that can serve at most 8 KiB from base 0x1000.
        let read = |addr: u64, len: u64| {
            (addr == 0x1000 && len <= 8192).then(|| vec![0xABu8; len as usize])
        };
        let (bytes, ok) = snapshot_eud_raw_window(0x1000, 16, read);
        assert!(ok);
        assert_eq!(
            bytes.len(),
            8192,
            "halved from 256 KiB to the first readable rung"
        );
        assert!(bytes.iter().all(|&b| b == 0xAB));

        // Sub-page readable prefix: the halving floor misses, the exact
        // required prefix still lands.
        let read_small =
            |addr: u64, len: u64| (addr == 0x1000 && len <= 64).then(|| vec![0xCDu8; len as usize]);
        let (bytes, ok) = snapshot_eud_raw_window(0x1000, 16, read_small);
        assert!(ok);
        assert_eq!(bytes.len(), 64, "the 16-dword required prefix");

        // Unreadable pointer: zero-filled required prefix, flagged degraded.
        let (bytes, ok) = snapshot_eud_raw_window(0x1000, 16, |_, _| None);
        assert!(!ok);
        assert_eq!(bytes.len(), 64);
        assert!(bytes.iter().all(|&b| b == 0));

        // Null / unaligned base short-circuits to the same degrade.
        let (bytes, ok) = snapshot_eud_raw_window(0, 2, |_, _| Some(vec![1]));
        assert!(!ok);
        assert_eq!(bytes.len(), 8);
        let (_, ok) = snapshot_eud_raw_window(0x1002, 2, |_, _| Some(vec![1]));
        assert!(!ok);
    }

    /// End-to-end binding-layer shape: a compute bind whose shader declared
    /// the raw EUD window snapshots live guest memory into `eud_raw` at the
    /// detected binding index, alongside the usual groups.
    #[test]
    fn compute_binding_carries_eud_raw_snapshot() {
        // 8 KiB of "EUD" guest memory with a recognizable head.
        let mut arena = vec![0u8; 8192 + 255];
        let base = (arena.as_ptr() as u64 + 255) & !255;
        let off = (base - arena.as_ptr() as u64) as usize;
        for (i, b) in arena[off..off + 64].iter_mut().enumerate() {
            *b = i as u8;
        }

        let mut bind = ShaderBindResources::default();
        bind.extended.used = true;
        bind.extended.start_register = 12;
        bind.extended.data.update_address(base);
        bind.eud_raw.used = true;
        bind.eud_raw.binding_index = 0;
        bind.eud_raw.required_dwords = 8;

        let ranges = [(arena.as_ptr() as u64, arena.len())];
        let binding = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_stage_binding(&bind, vk::ShaderStageFlags::COMPUTE)
        })
        .expect("eud_raw-only compute binding");
        let raw = binding.eud_raw.expect("raw window bound");
        assert_eq!(raw.binding, 0);
        assert_eq!(
            raw.bytes.len(),
            8192,
            "snapshot halves from 256 KiB down into the 8 KiB test range"
        );
        assert_eq!(
            &raw.bytes[..64],
            &(0..64).map(|i| i as u8).collect::<Vec<_>>()[..],
            "snapshot carries the live guest bytes"
        );

        // An unreadable EUD base still binds — zero-filled, not refused.
        bind.extended.data.update_address(0xDEAD_0000);
        let binding = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_stage_binding(&bind, vk::ShaderStageFlags::COMPUTE)
        })
        .expect("unreadable EUD window degrades, never refuses");
        let raw = binding.eud_raw.expect("raw window bound");
        assert_eq!(raw.bytes.len(), 32, "required 8 dwords of zeros");
        assert!(raw.bytes.iter().all(|&b| b == 0));
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
        // CB format 0x3 (8_8) maps to a 4-CHANNEL R8G8B8A8_UNORM target, NOT the
        // exact 2-channel R8G8_UNORM: the exact form was measured to device-lose
        // (the PS exports 4 components a 2-channel attachment can't honour), so
        // this widens to a same-class 4-channel target that does not device-lose
        // and lets the composite draws succeed (R,G correct, B,A extra). Never
        // R8G8_UNORM here — that is the regression.
        assert_eq!(
            vulkan_format(0x3, 0, 0).unwrap(),
            vk::Format::R8G8B8A8_UNORM,
            "CB 0x3 must widen to a 4-channel target, never the device-losing R8G8"
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

    /// Unified formats 14, 29 and 65 (all flagged unimplemented in ASTRO.BOT
    /// runs) map through SharpEmu's Gfx10UnifiedFormat table: 14 -> (3,0) =
    /// R8G8_UNORM (2 B), 29 -> (5,7) = R16G16_SFLOAT (4 B), 65 -> (12,0) =
    /// R16G16B16A16_UNORM (8 B).
    #[test]
    fn texture_vk_format_maps_unified_14_29_and_65() {
        let case = |unified: u32| {
            let mut t = kyty_graphics::shader::ShaderTextureResource::default();
            t.fields[1] |= unified << 20;
            texture_vk_format(&t)
        };
        assert_eq!(case(14).expect("format 14"), (vk::Format::R8G8_UNORM, 2));
        assert_eq!(case(29).expect("format 29"), (vk::Format::R16G16_SFLOAT, 4));
        assert_eq!(
            case(65).expect("format 65"),
            (vk::Format::R16G16B16A16_UNORM, 8)
        );
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

    /// A 2DArray T# (type 13, SWIZZLE_MODE 24 = SW_64KB_Z_X, format 7 =
    /// R16_UNORM — the measured ASTRO.BOT 1536x1536x3 shape) decodes every
    /// layer per-layer and reports `layers` so the Vulkan layer builds a
    /// `TYPE_2D_ARRAY` view.
    #[test]
    fn decode_texture_2darray_decodes_all_layers() {
        let (w, h, bpp_log2) = (8u32, 8u32, 1u32);
        let bpp = 1usize << bpp_log2;
        let layers: Vec<Vec<u8>> = (0..3u8)
            .map(|l| {
                (0..(w * h) as usize * bpp)
                    .map(|i| (l * 60 + (i % 41) as u8) % 251)
                    .collect()
            })
            .collect();
        let tiled: Vec<u8> = layers
            .iter()
            .flat_map(|l| crate::texture::tiling::tile_64kb_z_x(l, w, h, bpp_log2))
            .collect();
        let mut blob = vec![0u8; tiled.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + tiled.len()].copy_from_slice(&tiled);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 7 << 20; // unified format 7 = 16 UNORM
        t.fields[1] |= ((w - 1) & 3) << 30;
        t.fields[2] = (w - 1) >> 2;
        t.fields[2] |= (h - 1) << 14;
        t.fields[3] |= 24 << 20; // SWIZZLE_MODE 24 = SW_64KB_Z_X
        t.fields[3] |= 13 << 28; // type = 2DArray
        t.fields[4] = 2; // depth 2 + 1 = 3 layers

        let tex = crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
            decode_texture(&t)
        })
        .expect("2DArray texture decodes");
        assert!(!tex.cube);
        assert_eq!(tex.layers, 3);
        assert_eq!(tex.depth, 1);
        assert_eq!((tex.width, tex.height), (w, h));
        assert_eq!(tex.format, vk::Format::R16_UNORM);
        let layer_bytes = (w * h) as usize * bpp;
        for (l, layer) in layers.iter().enumerate() {
            assert_eq!(
                &tex.pixels[l * layer_bytes..(l + 1) * layer_bytes],
                layer.as_slice(),
                "layer {l} must detile to its original pixels"
            );
        }
    }

    /// A 3D storage-image (UAV) T# — type 10, format 71 (the measured
    /// ASTRO.BOT 240x135x64 RGBA16F volume shape) — reads its whole volume
    /// as an `R16G16B16A16_SFLOAT` upload with 8 B/texel.
    #[test]
    fn read_storage_image_3d_rgba16f_reads_the_whole_volume() {
        let (w, h, d) = (4u32, 2u32, 3u32);
        let bytes: Vec<u8> = (0..(w * h * d) as usize * 8)
            .map(|i| ((i * 13 + 1) % 251) as u8)
            .collect();
        let mut blob = vec![0u8; bytes.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + bytes.len()].copy_from_slice(&bytes);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 71 << 20; // unified format 16_16_16_16 FLOAT
        t.fields[1] |= ((w - 1) & 3) << 30;
        t.fields[2] = (w - 1) >> 2;
        t.fields[2] |= (h - 1) << 14;
        t.fields[3] |= 10 << 28; // type = 3D volume
        t.fields[4] = (d - 1) & 0x1FFF; // depth

        let upload =
            crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
                read_storage_image(&t)
            })
            .expect("3D RGBA16F UAV reads");
        assert_eq!((upload.width, upload.height, upload.depth), (w, h, d));
        assert_eq!(upload.format, vk::Format::R16G16B16A16_SFLOAT);
        assert_eq!(upload.texel_bytes(), 8);
        assert_eq!(upload.pixels, bytes, "the whole volume seeds the upload");
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
