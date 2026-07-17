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
    CLEAR_COLOR, DrawState, RenderedImage, ShaderStageBinding, StorageBufferBinding,
    VertexAttributeData, VertexBufferData, render_draw,
};
use ash::vk;
use kyty_graphics::hw_regs::{Context, Shader, UserConfig};
use kyty_graphics::run::{DrawError, DrawSink, IndexedDraw};
use kyty_graphics::shader::resources::{
    ShaderBindResources, ShaderPixelInputInfo, ShaderVertexInputInfo,
};
use kyty_graphics::shader::{spirv_get_embedded_ps, spirv_get_embedded_vs};
use kyty_graphics::spirv_asm;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// `VGT_PRIMITIVE_TYPE` values Kyty's Gen5 path emits.
mod prim {
    pub const TRIANGLE_LIST: u32 = 4;
    pub const TRIANGLE_FAN: u32 = 5;
    pub const TRIANGLE_STRIP: u32 = 6;
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
    window
        .get(start..end)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| err(format!("{kind} at {addr:#x}: slice {start}..{end} outside read window")))
}

fn read_guest_bytes(addr: u64, size: u64, kind: &str) -> Result<Vec<u8>, DrawError> {
    if size == 0 || !size.is_multiple_of(4) {
        return Err(err(format!(
            "{kind} at {addr:#x} has invalid byte size {size}"
        )));
    }
    let count = u32::try_from(size / 4)
        .map_err(|_| err(format!("{kind} at {addr:#x} is too large: {size} bytes")))?;
    let words = crate::guest_mem::read_dwords_checked(addr, count).ok_or_else(|| {
        err(format!(
            "{kind} guest range {addr:#x}..{:#x} is not fully readable",
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
    match format {
        74 => Ok(vk::Format::R32G32B32_SFLOAT),
        64 => Ok(vk::Format::R32G32_SFLOAT),
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
        buffers.push(VertexBufferData {
            bytes: read_guest_bytes(guest.addr, size, "vertex buffer")?,
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
            attributes.push(VertexAttributeData {
                location: location as u32,
                binding: binding as u32,
                format: gen5_vertex_format(info.resources[location].format())?,
                offset: guest.attr_offsets[ai],
            });
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

fn prepare_stage_binding(
    bind: &ShaderBindResources,
    stage: vk::ShaderStageFlags,
) -> Result<ShaderStageBinding, DrawError> {
    if bind.textures2d.textures_num != 0 {
        return Err(err(format!(
            "translated {stage:?} shader needs {} texture descriptors",
            bind.textures2d.textures_num
        )));
    }
    if bind.samplers.samplers_num != 0 {
        return Err(err(format!(
            "translated {stage:?} shader needs {} sampler descriptors",
            bind.samplers.samplers_num
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
        prim::TRIANGLE_FAN => (vk::PrimitiveTopology::TRIANGLE_FAN, index_count),
        prim::TRIANGLE_STRIP => (vk::PrimitiveTopology::TRIANGLE_STRIP, index_count),
        other => {
            return Err(err(format!(
                "unsupported VGT_PRIMITIVE_TYPE {other} (supported: 4 TriList, \
                 5 TriFan, 6 TriStrip, 17 RectList)"
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

    Ok(DrawState {
        width,
        height,
        format,
        clear_color: CLEAR_COLOR,
        scissor,
        viewport,
        topology,
        cull_mode,
        color_write_mask,
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
        }
    }
}

impl OffscreenDrawSink<'_> {
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
        // Frames are pure black despite the light-blue CLEAR working (M2 test),
        // so geometry covers the screen and the PS shades black. This names
        // whether the PS samples textures (→ texture-upload wall) or computes
        // black from missing storage/push-constant inputs. Gated to the first
        // few draws (XPS5X_TRACE_DRAWS) so it never floods a normal run.
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
            render_draw(self.dev, &state)
                .map_err(|e| err(format!("offscreen draw failed: {e}")))?
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
        self.draw_common(ctx, ucfg, sh, draw.index_count, Some((&index_bytes, index_type)))
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
        // the measured 0x41. Both flags describe execution already represented
        // by the translated Vulkan compute stage; other initiator bits need
        // explicit semantics before they can be accepted.
        if mode & !0x41 != 0 {
            return Err(err(format!(
                "unsupported compute dispatch initiator {mode:#x}"
            )));
        }
        let translated = match self.cache.translate_cs(&sh.cs, &ctx.sh_regs) {
            Ok(shader) => shader,
            Err(error) => {
                self.shader_skips += 1;
                self.dispatch_skips += 1;
                self.last_dispatch_skip_reason = Some(error.to_string());
                debug!(reason = %error, "compute dispatch skipped: bound shader is untranslatable");
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
        let outputs = dispatch_compute(
            self.dev,
            &ComputeState {
                groups,
                spirv: &translated.spirv,
                binding: prepared.as_ref(),
            },
        )
        .map_err(|error| err(format!("Vulkan compute dispatch failed: {error}")))?;
        if outputs.len() != guest_outputs.len() {
            return Err(err(format!(
                "compute writeback returned {} buffers for {} guest outputs",
                outputs.len(),
                guest_outputs.len()
            )));
        }
        for (addr, bytes) in guest_outputs.into_iter().zip(outputs) {
            debug!(
                addr = format_args!("{addr:#x}"),
                len = bytes.len(),
                head = format_args!(
                    "{:02x?}",
                    &bytes[..bytes.len().min(16)]
                ),
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
        self.dispatches += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kyty_graphics::hw_regs::ColorAttrib2;

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
        let (bytes, ty) = fetch_index_buffer(&draw).expect("readable index buffer");
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
        let (bytes, ty) = fetch_index_buffer(&draw).expect("readable index buffer");
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
        let (bytes, _) = fetch_index_buffer(&draw).expect("unaligned read must work");
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
            (
                0x9,
                vk::ColorComponentFlags::R | vk::ColorComponentFlags::A,
            ),
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

        let (buffers, attributes) = prepare_vertex_inputs(&vs).expect("measured vertex ABI");
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

        let binding = prepare_stage_binding(&ps.bind, vk::ShaderStageFlags::FRAGMENT)
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
}
