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
use crate::vulkan::compute::{
    ComputeState, compute_requires_slicing, dispatch_compute, dispatch_compute_deferred,
};
use crate::vulkan::instance::VulkanDevice;
use crate::vulkan::offscreen::{
    BlendState, CLEAR_COLOR, DepthState, DrawState, EudRawBinding, RenderedImage, SampledGroup,
    SamplerState, ShaderStageBinding, StorageBufferBinding, StorageImageBinding,
    StorageImageUpload, TextureBinding, TextureUpload, VertexAttributeData, VertexBufferData,
};
use ash::vk;
use kyty_graphics::hw_regs::{
    ComputeShaderInfo, Context, PixelShaderInfo, Shader, ShaderRegisters, UserConfig,
    VertexShaderInfo,
};
use kyty_graphics::run::{DrawError, DrawSink, IndexedDraw};
use kyty_graphics::shader::resources::{
    ShaderBindResources, ShaderPixelInputInfo, ShaderSamplerResource, ShaderStorageUsage,
    ShaderTextureUsage, ShaderVertexInputInfo,
};
use kyty_graphics::shader::{
    shader_push_constant_spill_binding, spirv_get_embedded_ps, spirv_get_embedded_vs,
};
use kyty_graphics::spirv_asm;

/// One captured ASTRO.BOT compute program is validation-clean but
/// reproducibly poisons the Windows Vulkan device after submission. Keep the
/// production path alive with an exact, named quarantine while its
/// exec-mask/reloop semantics are audited. The address + dispatch shape is
/// deliberately narrower than a generic "large compute" heuristic.
const fn is_known_device_loss_compute(cs_addr: u64, groups: [u32; 3]) -> bool {
    cs_addr == 0x5005_3c700 && groups[0] == 64 && groups[1] == 64 && groups[2] == 1
}
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

/// Opt-in shader-address filter for late-title draw forensics.
///
/// `RAEEN_TRACE_SHADER_ADDR=0x1234,0x5678` selects a draw when either its VS
/// or PS address matches. Keeping this separate from `RAEEN_TRACE_DRAWS`
/// avoids consuming every rate limiter on boot clears before a UI shader is
/// first bound.
fn shader_addr_selected(variable: &str, vs: u64, ps: u64) -> bool {
    let env = crate::diagnostics::gpu_env();
    let raw = match variable {
        "RAEEN_TRACE_SHADER_ADDR" => env.trace_shader_addr.as_deref(),
        "RAEEN_SOLID_PS_ADDR" => env.solid_ps_addr.as_deref(),
        _ => None,
    };
    let Some(raw) = raw else {
        return false;
    };
    raw.split(',').any(|part| {
        let part = part.trim();
        let digits = part
            .strip_prefix("0x")
            .or_else(|| part.strip_prefix("0X"))
            .unwrap_or(part);
        u64::from_str_radix(digits, 16).is_ok_and(|addr| addr == vs || addr == ps)
    })
}

fn trace_selected_shader(vs: u64, ps: u64) -> bool {
    shader_addr_selected("RAEEN_TRACE_SHADER_ADDR", vs, ps)
}

/// Address-independent companion to [`trace_selected_shader`].
///
/// Guest shader addresses move when a title is relaunched after another
/// process, while the translated program sizes remain stable. Accept
/// `RAEEN_TRACE_SHADER_WORDS=vs:4103,ps:1142` so a late-screen shader can be
/// selected without first consuming all of the boot-time trace limiters just
/// to discover its address.
fn trace_shader_words_selected(vs_words: usize, ps_words: usize) -> bool {
    crate::diagnostics::gpu_env()
        .trace_shader_words
        .as_deref()
        .is_some_and(|raw| {
            raw.split(',').any(|part| {
                let Some((stage, words)) = part.trim().split_once(':') else {
                    return false;
                };
                let Ok(words) = words.trim().parse::<usize>() else {
                    return false;
                };
                match stage.trim().to_ascii_lowercase().as_str() {
                    "vs" => words == vs_words,
                    "ps" => words == ps_words,
                    _ => false,
                }
            })
        })
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
    blend_state_for_slot(ctx, 0)
}

/// Per-slot variant of [`blend_state_from_regs`]: MRT slot `n` blends by
/// `CB_BLEND{n}_CONTROL`. The blend CONSTANTS (`CB_BLEND_RED..ALPHA`) are
/// per-context and shared by every slot.
fn blend_state_for_slot(ctx: &Context, slot: usize) -> Result<BlendState, DrawError> {
    let bc = &ctx.blend_control[slot];
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

/// Convert one Gen5 stencil operation to Vulkan and report the reference value
/// needed by a REPLACE-class operation.
///
/// The two enums are not layout-compatible. In particular, AMD 2/3/4 are
/// Ones/ReplaceTest/ReplaceOp, while Vulkan 2 is the single REPLACE opcode;
/// AMD wrap operations are 8/9 while Vulkan's are 6/7. Casting the raw value
/// silently turned Minecraft's stencil setup into different operations.
fn gen5_stencil_op(
    op: u8,
    test_value: u8,
    operation_value: u8,
) -> Result<(vk::StencilOp, Option<u32>), DrawError> {
    let mapped = match op {
        0 => (vk::StencilOp::KEEP, None),
        1 => (vk::StencilOp::ZERO, None),
        2 => (vk::StencilOp::REPLACE, Some(u32::from(u8::MAX))),
        3 => (vk::StencilOp::REPLACE, Some(u32::from(test_value))),
        4 => (vk::StencilOp::REPLACE, Some(u32::from(operation_value))),
        5 => (vk::StencilOp::INCREMENT_AND_CLAMP, None),
        6 => (vk::StencilOp::DECREMENT_AND_CLAMP, None),
        7 => (vk::StencilOp::INVERT, None),
        8 => (vk::StencilOp::INCREMENT_AND_WRAP, None),
        9 => (vk::StencilOp::DECREMENT_AND_WRAP, None),
        other => {
            return Err(err(format!(
                "unsupported Gen5 stencil operation {other} (supported: 0 Keep, 1 Zero, \
                 2 Ones, 3 ReplaceTest, 4 ReplaceOp, 5/6 clamp, 7 Invert, 8/9 wrap)"
            )));
        }
    };
    Ok(mapped)
}

fn gen5_stencil_state(
    operations: [u8; 3],
    compare: u8,
    values: [u8; 4],
) -> Result<vk::StencilOpState, DrawError> {
    let [fail, pass, depth_fail] = operations;
    let [test_value, compare_mask, write_mask, operation_value] = values;
    let (fail_op, fail_reference) = gen5_stencil_op(fail, test_value, operation_value)?;
    let (pass_op, pass_reference) = gen5_stencil_op(pass, test_value, operation_value)?;
    let (depth_fail_op, depth_fail_reference) =
        gen5_stencil_op(depth_fail, test_value, operation_value)?;
    let replace_reference = [fail_reference, pass_reference, depth_fail_reference]
        .into_iter()
        .flatten()
        .next();
    if let Some(reference) = replace_reference
        && [fail_reference, pass_reference, depth_fail_reference]
            .into_iter()
            .flatten()
            .any(|candidate| candidate != reference)
    {
        return Err(err(
            "stencil operations require conflicting Vulkan reference values",
        ));
    }

    // Vulkan has one reference for both comparison and REPLACE. When no
    // REPLACE-class op consumes it, retain the guest comparison value.
    let reference = replace_reference.unwrap_or(u32::from(test_value));
    Ok(vk::StencilOpState::default()
        .fail_op(fail_op)
        .pass_op(pass_op)
        .depth_fail_op(depth_fail_op)
        .compare_op(vk::CompareOp::from_raw(i32::from(compare)))
        .compare_mask(u32::from(compare_mask))
        .write_mask(u32::from(write_mask))
        .reference(reference))
}

fn depth_state_from_regs(ctx: &Context) -> Result<Option<DepthState<'static>>, DrawError> {
    let control = &ctx.depth_control;
    if !control.z_enable && !control.stencil_enable {
        return Ok(None);
    }
    let target = &ctx.depth_render_target;
    if target.z_write_base_addr == 0 {
        return Err(err(
            "depth/stencil enabled but DB_Z_WRITE_BASE is 0 — refusing an unbacked attachment",
        ));
    }
    let combined = target.z_info.format * 2 + target.stencil_info.format;
    let format = match combined {
        2 => vk::Format::D16_UNORM,
        3 => vk::Format::D24_UNORM_S8_UINT,
        6 => vk::Format::D32_SFLOAT,
        7 => vk::Format::D32_SFLOAT_S8_UINT,
        other => {
            return Err(err(format!(
                "unsupported depth/stencil format code {other} \
                 (DB_Z_INFO={} DB_STENCIL_INFO={})",
                target.z_info.format, target.stencil_info.format
            )));
        }
    };
    let stencil = &ctx.stencil_control;
    let mask = &ctx.stencil_mask;
    let front = gen5_stencil_state(
        [
            stencil.stencil_fail,
            stencil.stencil_zpass,
            stencil.stencil_zfail,
        ],
        control.stencilfunc,
        [
            mask.stencil_testval,
            mask.stencil_mask,
            mask.stencil_writemask,
            mask.stencil_opval,
        ],
    )?;
    let back = if control.backface_enable {
        gen5_stencil_state(
            [
                stencil.stencil_fail_bf,
                stencil.stencil_zpass_bf,
                stencil.stencil_zfail_bf,
            ],
            control.stencilfunc_bf,
            [
                mask.stencil_testval_bf,
                mask.stencil_mask_bf,
                mask.stencil_writemask_bf,
                mask.stencil_opval_bf,
            ],
        )?
    } else {
        front
    };
    let vp = &ctx.screen_viewport.viewports[0];
    // Diagnostic bisection only: if missing geometry reappears with both depth
    // reads and writes disabled, the draw reached rasterization and the defect
    // is in attachment lifetime/clear/compare state rather than vertex
    // translation. Never changes production behaviour when unset.
    let disable_depth = crate::diagnostics::gpu_env().no_depth;
    Ok(Some(DepthState {
        target_base: Some(target.z_write_base_addr),
        format,
        test_enable: control.z_enable && !disable_depth,
        write_enable: control.z_write_enable
            && !target.depth_view.depth_write_disable
            && !disable_depth,
        compare_op: vk::CompareOp::from_raw(i32::from(control.zfunc)),
        // Diagnostic bisection only: lets a real-title run distinguish an
        // empty frame caused by stencil rejection from shader/geometry faults.
        // Production behaviour remains register-derived when unset.
        stencil_test_enable: control.stencil_enable && !crate::diagnostics::gpu_env().no_stencil,
        stencil_front: front,
        stencil_back: back,
        clear_depth: ctx.render_control.depth_clear_enable,
        clear_stencil: ctx.render_control.stencil_clear_enable,
        clear_depth_value: ctx.depth_clear_value,
        clear_stencil_value: u32::from(ctx.stencil_clear_value),
        viewport_depth: [vp.zoffset, vp.zoffset + vp.zscale],
        initial: None,
        initial_stencil: None,
    }))
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
        // 32 FLOAT (CB format 4, number type 7). GTA V programs this scalar
        // target immediately after its packed 10_10_10_2 pass.
        (0x4, 7, 0) => Ok(vk::Format::R32_SFLOAT),
        // 16_16 FLOAT (CB format 5, number type 7). GTA V programs this
        // two-channel intermediate in its first live render sequence.
        // KytyPS5 names layout 5 `k16_16`; SharpEmu maps the exact pair
        // `(5, 7)` to `R16G16Sfloat`.
        (0x5, 7, 0) => Ok(vk::Format::R16G16_SFLOAT),
        // 10_11_11 / 11_11_10 FLOAT (channel_type 7): the packed HDR
        // intermediate render target ASTRO.BOT draws into. SharpEmu maps both
        // CB formats 6 and 7 with channel_type 7 to B10G11R11_UFLOAT_PACK32.
        (0x6 | 0x7, 7, 0) => Ok(vk::Format::B10G11R11_UFLOAT_PACK32),
        // 10_10_10_2 UNORM (CB format 9, number type 0). GTA V uses this
        // packed 32-bit target in its first live DCB. SharpEmu's
        // GetRenderTargetFormat maps CB format 9 to the same Vulkan packed
        // layout (R in bits 0..9, A in 30..31).
        (0x9, 0, 0) => Ok(vk::Format::A2B10G10R10_UNORM_PACK32),
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

/// `CB_COLOR_CONTROL.MODE` operation modes (shadPS4 `regs_color.h`
/// `OperationMode`; Kyty accepts only Disable/Normal — `bc_check`,
/// GraphicsRender.cpp L938). Anything above Normal is a special CB pass, not
/// an ordinary draw.
pub(crate) mod cb_mode {
    pub const ELIMINATE_FAST_CLEAR: u8 = 2;
    pub const RESOLVE: u8 = 3;
    pub const FMASK_DECOMPRESS: u8 = 5;
    pub const DCC_DECOMPRESS: u8 = 6;
}

/// Build the direct-clear image for an eliminate-fast-clear pass on `rt`.
///
/// Hardware fast clear stores the clear colour PACKED in the surface's own
/// format in `CB_COLOR{n}_CLEAR_WORD0/1`; the FCE pass rewrites CMASK-cleared
/// tiles with exactly those bytes. Raeen keeps no CMASK, so the honest
/// equivalent (shadPS4 `Rasterizer::EliminateFastClear`) is a full-target
/// clear to the packed words — a raw byte splat, no per-format unpack needed,
/// because the framebuffer map holds raw target-format bytes.
///
/// Returns `Ok(None)` when there is nothing to clear (no bound target, fast
/// clear not enabled on it, or a degenerate extent — the register state does
/// not describe a real FCE). `resolution_scale` mirrors
/// [`DrawState::scale_resolution`] so the cleared image matches the extent
/// later draws render at.
pub(crate) fn fast_clear_image(
    rt: &kyty_graphics::hw_regs::RenderTarget,
    resolution_scale: f32,
) -> Result<Option<RenderedImage>, DrawError> {
    if rt.base.addr == 0 || !rt.info.cmask_fast_clear_enable {
        return Ok(None);
    }
    if rt.attrib2.width == 0 || rt.attrib2.height == 0 {
        return Ok(None);
    }
    let format = vulkan_format(rt.info.format, rt.info.channel_type, rt.info.channel_order)?;
    let bpp = crate::vulkan::offscreen::readback_bpp(format)
        .map_err(|e| err(format!("eliminate-fast-clear: {e}")))?;
    let factor = if resolution_scale.is_finite() {
        resolution_scale.clamp(0.5, 4.0)
    } else {
        1.0
    };
    let scale_u = |v: u32| ((v as f32 * factor).round() as u32).max(1);
    let (width, height) = (
        scale_u(rt.attrib2.width + 1),
        scale_u(rt.attrib2.height + 1),
    );
    let texel: [u8; 8] = {
        let mut t = [0u8; 8];
        t[..4].copy_from_slice(&rt.clear_word0.word0.to_le_bytes());
        t[4..].copy_from_slice(&rt.clear_word1.word1.to_le_bytes());
        t
    };
    let mut pixels = vec![0u8; (width * height * bpp) as usize];
    for chunk in pixels.chunks_exact_mut(bpp as usize) {
        chunk.copy_from_slice(&texel[..bpp as usize]);
    }
    Ok(Some(RenderedImage {
        width,
        height,
        pixels,
        bytes_per_pixel: bpp,
    }))
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

/// Diagnostic-only opaque fragment shader for proving whether selected
/// geometry reaches rasterization independently of the title PS's sampling and
/// discard path. `RAEEN_SOLID_PS_ADDR=<hex>[,...]` selects exact guest PS
/// addresses; production never calls this when the variable is unset.
fn assemble_solid_diagnostic_ps() -> Result<Vec<u32>, DrawError> {
    const SOURCE: &str = r#"
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %9
               OpExecutionMode %4 OriginUpperLeft
               OpDecorate %9 Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %9 = OpVariable %_ptr_Output_v4float Output
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
         %11 = OpConstantComposite %v4float %float_1 %float_0 %float_1 %float_1
          %4 = OpFunction %void None %3
          %5 = OpLabel
               OpStore %9 %11
               OpReturn
               OpFunctionEnd
"#;
    spirv_asm::assemble(SOURCE)
        .map_err(|e| err(format!("assembling diagnostic solid fragment shader: {e}")))
}

/// Both stages' SPIR-V, each either embedded or fetched from guest memory.
#[derive(Debug, Clone)]
struct ResolvedShaders {
    vs: Arc<Vec<u32>>,
    ps: Arc<Vec<u32>>,
    vs_info: ShaderVertexInputInfo,
    ps_info: ShaderPixelInputInfo,
}

/// Everything [`resolve_shaders`] reads from the mutable PM4 register files.
///
/// The shader translation cache already avoids recompilation, but resolving a
/// cache hit still walks both stage analyses and clones their metadata on every
/// draw. Minecraft repeats the same few exact bindings hundreds of times in a
/// submission, so retain the completed pair for that submission only. Keeping
/// this key exact avoids guessing which register fields a future translator
/// revision may begin consulting.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ResolvedShaderKey {
    vs: VertexShaderInfo,
    ps: PixelShaderInfo,
    regs: ShaderRegisters,
}

impl ResolvedShaderKey {
    fn new(ctx: &Context, sh: &Shader) -> Self {
        Self {
            vs: sh.vs,
            ps: sh.ps,
            regs: ctx.sh_regs,
        }
    }
}

/// A Minecraft gameplay frame resolves roughly 120 distinct exact VS/PS
/// register states before repeating them on the next frame. A 32-entry LRU
/// therefore thrashed completely: the measured in-world profile reported
/// `resolve_hits=0` while both parsed shader caches were 100% hot, spending
/// 80-95 us per draw re-running resource analysis. Keep two full frames'
/// working set so the exact-state safety contract remains unchanged while
/// repeated frame bindings can hit.
const RESOLVED_SHADER_MEMO_CAPACITY: usize = 256;

/// Small submission-local LRU of successful shader resolutions.
///
/// Guest writes and compute dispatches clear it before a later draw can
/// observe modified shader code or embedded resource metadata. Failed
/// translations deliberately stay in [`ShaderTranslateCache`]'s named
/// negative cache rather than becoming an uninspectable entry here.
#[derive(Default)]
struct ResolvedShaderMemo {
    entries: Vec<(ResolvedShaderKey, ResolvedShaders)>,
    hits: u64,
    misses: u64,
}

/// The fetcher never reads more than 256 KiB for one guest shader. Treat that
/// complete bounded window as a dependency so a compute write can invalidate
/// code safely without throwing away unrelated shader analyses.
const SHADER_CODE_DEPENDENCY_BYTES: u64 = 256 * 1024;

fn ranges_overlap(a_base: u64, a_len: u64, b_base: u64, b_len: u64) -> bool {
    a_len != 0
        && b_len != 0
        && a_base < b_base.saturating_add(b_len)
        && b_base < a_base.saturating_add(a_len)
}

fn eud_descriptor_end_dword(start_register: i32, width: u64, eud_base: i32) -> Option<u64> {
    let start = u64::try_from(start_register).ok()?;
    let relative = if start >= kyty_graphics::hw_regs::UserSgprInfo::SGPRS_MAX as u64 {
        start - kyty_graphics::hw_regs::UserSgprInfo::SGPRS_MAX as u64
    } else {
        start.checked_sub(u64::try_from(eud_base).ok()?)?
    };
    Some(relative.saturating_add(width))
}

/// Exact guest-memory descriptor window captured into one analyzed bind.
///
/// Direct user-SGPR descriptors are already part of [`ResolvedShaderKey`].
/// Only EUD-resident descriptors depend on mutable guest memory; retaining
/// their precise span lets compute writes to texture/vertex *contents* keep
/// the analysis while writes to the descriptor table still invalidate it.
fn bind_eud_dependency(bind: &ShaderBindResources) -> Option<(u64, u64)> {
    if !bind.extended.used {
        return None;
    }
    let base = bind.extended.data.base();
    if base == 0 {
        return None;
    }
    let eud_base = bind.extended.start_register;
    let mut end_dwords = 2u64;
    for index in 0..bind.storage_buffers.buffers_num.max(0) as usize {
        if bind.storage_buffers.extended[index] {
            let Some(end) =
                eud_descriptor_end_dword(bind.storage_buffers.start_register[index], 4, eud_base)
            else {
                return Some((base, 16 * 1024 * 1024));
            };
            end_dwords = end_dwords.max(end);
        }
    }
    for index in 0..bind.textures2d.textures_num.max(0) as usize {
        let descriptor = &bind.textures2d.desc[index];
        if descriptor.extended {
            let Some(end) = eud_descriptor_end_dword(descriptor.start_register, 8, eud_base) else {
                return Some((base, 16 * 1024 * 1024));
            };
            end_dwords = end_dwords.max(end);
        }
    }
    for index in 0..bind.samplers.samplers_num.max(0) as usize {
        if bind.samplers.extended[index] {
            let Some(end) =
                eud_descriptor_end_dword(bind.samplers.start_register[index], 4, eud_base)
            else {
                return Some((base, 16 * 1024 * 1024));
            };
            end_dwords = end_dwords.max(end);
        }
    }
    for index in 0..bind.gds_pointers.pointers_num.max(0) as usize {
        if bind.gds_pointers.extended[index] {
            let Some(end) =
                eud_descriptor_end_dword(bind.gds_pointers.start_register[index], 1, eud_base)
            else {
                return Some((base, 16 * 1024 * 1024));
            };
            end_dwords = end_dwords.max(end);
        }
    }
    if bind.eud_raw.used {
        end_dwords = end_dwords
            .max(eud_raw_window_want_bytes(bind.eud_raw.required_dwords).saturating_add(3) / 4);
    }
    Some((base, end_dwords.saturating_mul(4)))
}

fn resolved_shader_depends_on_ranges(
    key: &ResolvedShaderKey,
    shaders: &ResolvedShaders,
    writes: &[(u64, u64)],
) -> bool {
    let mut code = [0u64; 4];
    let code_count = if key.vs.vs_embedded {
        0
    } else {
        code[0] = key.vs.vs_regs.data_addr;
        code[1] = key.vs.es_regs.data_addr;
        code[2] = key.vs.gs_regs.data_addr;
        3
    };
    let ps_index = code_count;
    if !key.ps.ps_embedded {
        code[ps_index] = key.ps.ps_regs.data_addr;
    }
    let code_count = code_count + usize::from(!key.ps.ps_embedded);
    let vs_eud = bind_eud_dependency(&shaders.vs_info.bind);
    let ps_eud = bind_eud_dependency(&shaders.ps_info.bind);

    writes.iter().any(|&(write_base, write_len)| {
        code[..code_count].iter().any(|&program| {
            program != 0
                && ranges_overlap(program, SHADER_CODE_DEPENDENCY_BYTES, write_base, write_len)
        }) || vs_eud.is_some_and(|(base, len)| ranges_overlap(base, len, write_base, write_len))
            || ps_eud.is_some_and(|(base, len)| ranges_overlap(base, len, write_base, write_len))
    })
}

impl ResolvedShaderMemo {
    fn get(&mut self, key: ResolvedShaderKey) -> Option<ResolvedShaders> {
        let Some(index) = self
            .entries
            .iter()
            .position(|(candidate, _)| *candidate == key)
        else {
            self.misses += 1;
            crate::vulkan::offscreen::DRAW_STAGE_RESOLVE_MISSES
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        };
        self.hits += 1;
        crate::vulkan::offscreen::DRAW_STAGE_RESOLVE_HITS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let entry = self.entries.remove(index);
        let shaders = entry.1.clone();
        self.entries.push(entry);
        Some(shaders)
    }

    fn insert(&mut self, key: ResolvedShaderKey, shaders: ResolvedShaders) {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(candidate, _)| *candidate == key)
        {
            self.entries.remove(index);
        } else if self.entries.len() == RESOLVED_SHADER_MEMO_CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push((key, shaders));
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn invalidate_ranges(&mut self, writes: &[(u64, u64)]) {
        if writes.is_empty() {
            return;
        }
        self.entries
            .retain(|(key, shaders)| !resolved_shader_depends_on_ranges(key, shaders, writes));
    }
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
    } else if shader_addr_selected("RAEEN_SOLID_PS_ADDR", 0, sh.ps.ps_regs.data_addr) {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SOLID_PS_SEEN: AtomicU32 = AtomicU32::new(0);
        if SOLID_PS_SEEN.fetch_add(1, Ordering::Relaxed) < 4 {
            tracing::warn!(
                ps_addr = format_args!("{:#x}", sh.ps.ps_regs.data_addr),
                "using diagnostic solid fragment shader for selected guest PS"
            );
        }
        (
            Arc::new(assemble_solid_diagnostic_ps()?),
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

/// Number of vertex records addressable by this draw.
///
/// Uploading the V# descriptor's full `num_records` is catastrophically
/// wasteful for UI ring buffers: Minecraft exposes roughly 4 MiB but a quad
/// indexes only records 0..=3. Vulkan cannot read beyond the largest submitted
/// index, so the exact safe upload is `(max_index + 1) * stride`.
fn required_vertex_records(
    index: Option<(&[u8], vk::IndexType)>,
    vertex_count: u32,
) -> Result<u32, DrawError> {
    let Some((bytes, index_type)) = index else {
        return Ok(vertex_count);
    };
    let max = match index_type {
        vk::IndexType::UINT16 => bytes
            .chunks_exact(2)
            .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
            .max(),
        vk::IndexType::UINT32 => bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .max(),
        other => {
            return Err(err(format!(
                "unsupported Vulkan index type {other:?} while sizing vertex upload"
            )));
        }
    };
    // An empty index buffer addresses no record at all; fall back to the draw's
    // vertex count rather than refusing it (the pre-limit behaviour).
    Ok(max
        .and_then(|index| index.checked_add(1))
        .unwrap_or(vertex_count))
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
    let bytes = crate::guest_mem::read_bytes_validated(addr, size).ok_or_else(|| {
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
    if let Some(dir) = crate::diagnostics::gpu_env().dump_gpu_resources.as_deref()
        && !dir.is_empty()
    {
        let safe_kind = kind.replace(' ', "_");
        let path = std::path::Path::new(dir).join(format!("{safe_kind}_{addr:012x}_{size}.bin"));
        if !path.exists()
            && let Err(error) =
                std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, &bytes))
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

fn gen5_vertex_format_and_size(format: u8) -> Result<(vk::Format, u64), DrawError> {
    // Gen5 unified-format code → Vulkan, per SharpEmu's Gfx10UnifiedFormat
    // table (the RDNA2 authority): 64 → (11,7) = 32_32_FLOAT,
    // 74 → (13,7) = 32_32_32_FLOAT, 77 → (14,7) = 32_32_32_32_FLOAT,
    // 56 → (10,0) = 8_8_8_8 UNORM, 71 → (12,7) = 16_16_16_16_FLOAT,
    // 5 → (1,4) = 8 UINT (measured: GTA V's first submitted DCB),
    // 11 → (2,4) = 16 UINT (measured: Minecraft's packed per-vertex value),
    // 57 → (10,1) = 8_8_8_8 SNORM (same UI draw, next attribute).
    match format {
        74 => Ok((vk::Format::R32G32B32_SFLOAT, 12)),
        64 => Ok((vk::Format::R32G32_SFLOAT, 8)),
        77 => Ok((vk::Format::R32G32B32A32_SFLOAT, 16)),
        56 => Ok((vk::Format::R8G8B8A8_UNORM, 4)),
        57 => Ok((vk::Format::R8G8B8A8_SNORM, 4)),
        71 => Ok((vk::Format::R16G16B16A16_SFLOAT, 8)),
        23 => Ok((vk::Format::R16G16_UNORM, 4)),
        // Unified 5 is (FMT_8, UINT). GTA's shader consumes the raw integer
        // bits, so the Vulkan interface must be R8_UINT rather than a
        // normalized/float approximation.
        5 => Ok((vk::Format::R8_UINT, 1)),
        // Unified 11 is (FMT_16, UINT), not USCALED. The guest fetch writes
        // the raw integer bits into its VGPR; Minecraft immediately uses
        // integer bit operations on this value to form its skinning-matrix
        // address. The shader translator declares this attribute as uint and
        // bitcasts it into the float-backed VGPR representation.
        11 => Ok((vk::Format::R16_UINT, 2)),
        other => Err(err(format!(
            "unsupported Gen5 vertex-buffer format {other}"
        ))),
    }
}

fn gen5_vertex_format(format: u8) -> Result<vk::Format, DrawError> {
    Ok(gen5_vertex_format_and_size(format)?.0)
}

#[cfg(test)]
fn prepare_vertex_inputs(
    info: &ShaderVertexInputInfo,
) -> Result<(Vec<VertexBufferData>, Vec<VertexAttributeData>), DrawError> {
    prepare_vertex_inputs_limited(info, None)
}

fn prepare_vertex_inputs_limited(
    info: &ShaderVertexInputInfo,
    record_limit: Option<u32>,
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
        let attr_num = usize::try_from(guest.attr_num).map_err(|_| {
            err(format!(
                "negative vertex attribute count {}",
                guest.attr_num
            ))
        })?;
        if attr_num > guest.attr_indices.len() {
            return Err(err("vertex attribute count exceeds fixed array"));
        }

        let stride = u64::from(guest.stride);
        // Upload only the records this draw can address — but an index that runs
        // PAST the V# is clamped, never refused. Hardware tolerates it (the
        // fetch reads zero under the `robustBufferAccess` this device enables),
        // and refusing dropped the entire draw: a primitive-restart sentinel
        // (0xFFFF / 0xFFFFFFFF — this pipeline does not enable restart, so the
        // sentinel is walked as a real index) made `required` 65536 and silently
        // erased every restart-using draw, counted only as a refusal.
        let records = u64::from(match record_limit {
            Some(required) if required > guest.num_records => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static OVER_RANGE: AtomicU64 = AtomicU64::new(0);
                let occurrence = OVER_RANGE.fetch_add(1, Ordering::Relaxed) + 1;
                if occurrence <= 4 || occurrence.is_power_of_two() {
                    debug!(
                        occurrence,
                        required,
                        exposed = guest.num_records,
                        binding,
                        "indexed draw addresses more vertex records than its V# exposes — \
                         clamping the upload (out-of-range fetches read zero)"
                    );
                }
                guest.num_records
            }
            Some(required) => required,
            None => guest.num_records,
        });
        let mut size = stride
            .checked_mul(records)
            .ok_or_else(|| err("vertex buffer size overflow"))?;
        // A merged input buffer may contain a format wider than the bytes left
        // in its stride (ASTRO.BOT: float4 at offset 16, stride 24). GCN fetch
        // descriptors address each attribute independently, so the final fetch
        // extends past `stride * records`. Size the host upload to the union of
        // every descriptor instead of exposing an out-of-bounds Vulkan read.
        if records != 0 {
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
                let (_, format_bytes) =
                    gen5_vertex_format_and_size(info.resources[location].format())?;
                let extent = u64::from(guest.attr_offsets[ai])
                    .checked_add(
                        stride
                            .checked_mul(records - 1)
                            .ok_or_else(|| err("vertex attribute extent overflow"))?,
                    )
                    .and_then(|n| n.checked_add(format_bytes))
                    .ok_or_else(|| err("vertex attribute extent overflow"))?;
                size = size.max(extent);
            }
        }
        let bytes = read_guest_bytes(guest.addr, size, "vertex buffer")?;
        // Vertex-input probe (RAEEN_TRACE_DRAWS). Report the descriptor and
        // whether the guest bytes are actually non-zero. This can identify an
        // all-zero input buffer, but does not by itself prove whether later
        // raster state accepted or rejected the transformed primitives.
        if crate::diagnostics::gpu_env().trace_draws {
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
            per_instance: guest.fetch_index != 0,
        });

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
            // Vertex-binding probe (RAEEN_TRACE_DRAWS). `location` must use the
            // same index space the shader declares (`OpDecorate %attr{i}
            // Location {i}` over 0..resources_num), and the format/offset must
            // match the V#. This reports that link without treating it as a
            // complete fragment-coverage verdict.
            if crate::diagnostics::gpu_env().trace_draws {
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
        // 77 -> (14,7) = 32_32_32_32 FLOAT. The same unified row is already
        // used by Gen5 vertex attributes; live ASTRO.BOT now exposes it as a
        // 1x1 read-write T# and requires the full 16-byte texel.
        77 => Ok((vk::Format::R32G32B32A32_SFLOAT, 16)),
        // 5 -> (dataFormat 1, numFormat 4) = 8-bit UINT (SharpEmu
        // Gfx10UnifiedFormat unified 5 -> (1u, 4u)); R8_UINT at 1 B/texel.
        // Measured on ASTRO.BOT's 1920x1080 tile=24 target sampled as a texture.
        5 => Ok((vk::Format::R8_UINT, 1)),
        // ---- Block-compressed (BC) family ----
        //
        // The unified codes 169-182 are the GFX10 image-only BC encodings; they
        // have no legacy DATA_FORMAT equivalent, so SharpEmu's
        // `Gfx10UnifiedFormat` maps each to itself and the BC identity lives in
        // its guest-format table (`Gpu/Metal/MetalGuestFormats.cs:157-170`) with
        // block sizes from `Agc/AgcExports.cs:8226-8231` (169/170/175/176 = 8
        // bytes, the rest 16). The sRGB pairs differ only in the numeric class,
        // which the shader's sampled type never sees.
        //
        // The second element of the returned tuple is bytes per ADDRESSABLE
        // ELEMENT, and for BC an element is a 4x4 texel block — so these are
        // block bytes, and every size/tiling computation downstream runs in
        // block units (see `format_block_extent`). That is also why the 8- and
        // 16-byte rows of the swizzle tables already carry the comment "also
        // BC1/BC4 blocks": a tiled BC surface swizzles its blocks with exactly
        // the same equations, at an element size the tables already cover.
        169 => Ok((vk::Format::BC1_RGBA_UNORM_BLOCK, 8)),
        170 => Ok((vk::Format::BC1_RGBA_SRGB_BLOCK, 8)),
        171 => Ok((vk::Format::BC2_UNORM_BLOCK, 16)),
        172 => Ok((vk::Format::BC2_SRGB_BLOCK, 16)),
        173 => Ok((vk::Format::BC3_UNORM_BLOCK, 16)),
        174 => Ok((vk::Format::BC3_SRGB_BLOCK, 16)),
        175 => Ok((vk::Format::BC4_UNORM_BLOCK, 8)),
        176 => Ok((vk::Format::BC4_SNORM_BLOCK, 8)),
        177 => Ok((vk::Format::BC5_UNORM_BLOCK, 16)),
        178 => Ok((vk::Format::BC5_SNORM_BLOCK, 16)),
        179 => Ok((vk::Format::BC6H_UFLOAT_BLOCK, 16)),
        180 => Ok((vk::Format::BC6H_SFLOAT_BLOCK, 16)),
        181 => Ok((vk::Format::BC7_UNORM_BLOCK, 16)),
        182 => Ok((vk::Format::BC7_SRGB_BLOCK, 16)),
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
/// (stage D): FNV-1a over the range length, tile mode, 64 evenly-strided
/// 64-byte chunks, and the final 64 bytes — ~4 KiB of guest reads regardless of
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
/// covered by the same per-bind rehash. `RAEEN_NO_TEX_CACHE=1` restores
/// per-draw decode + upload wholesale.
fn guest_sample_hash(base: u64, len: u64, tile_mode: u8) -> Option<u64> {
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
    // Source bytes are not enough to identify decoded content: the same
    // allocation produces different pixels under linear, SW_64KB_S, and
    // SW_64KB_R_X addressing.
    h = mix(h, &[tile_mode]);
    if len <= CHUNKS * CHUNK_BYTES {
        let bytes = read_guest_bytes_unaligned(base, len, "texture sample-hash").ok()?;
        h = mix(h, &bytes);
    } else {
        // Scale the probe with the surface instead of always sampling a flat
        // 4 KiB. A fixed 64x64-byte probe covers 0.1% of a 4 MiB font atlas,
        // so a title that rasterizes a few new glyphs into an existing atlas
        // almost never perturbs a sampled window: the hash is unchanged, the
        // cache reports a hit, and the STALE atlas stays bound. That is
        // exactly what "missing glyphs" looks like on screen — characters
        // uploaded early render, ones added later never appear (measured on
        // Minecraft's world-creation page: W/S/E/D/P/N/H/Y/b/f/g/w/x all
        // absent while C/G/M and most lowercase rendered).
        //
        // Sample ~1/64th of the surface, bounded so the cost stays far below
        // a full re-decode: a 4 MiB atlas now probes 64 KiB across 1024
        // windows (16x the linear density, 64x the byte coverage) and a
        // 16 MiB one is capped at 512 KiB. Still probabilistic — the exact
        // fix is write-tracking the guest pages (Tier 5 page-dirty tracking)
        // — but it moves a routine miss to an unlikely one.
        const MIN_SAMPLED: u64 = CHUNKS * CHUNK_BYTES; // 4 KiB floor
        const MAX_SAMPLED: u64 = 512 * 1024;
        let sampled = (len / 64).clamp(MIN_SAMPLED, MAX_SAMPLED);
        let chunks = (sampled / CHUNK_BYTES).max(CHUNKS);
        let stride = (len / chunks).max(CHUNK_BYTES);
        for i in 0..chunks {
            let offset = i * stride;
            if offset >= len {
                break;
            }
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

/// Guest texture probes already computed in the current PM4 submission.
///
/// A cache hit used to re-read and hash up to 512 KiB on every draw even
/// though almost all draws in a submission bind the same handful of immutable
/// resources. The command stream is the ownership boundary: CPU-side resource
/// updates require a later submit, while compute and PM4 memory writes call
/// [`GuestSampleHashMemo::clear`] before a later draw can reuse an entry.
#[derive(Default)]
struct GuestSampleHashMemo {
    values: std::cell::RefCell<HashMap<(u64, u64, u8), u64>>,
}

impl GuestSampleHashMemo {
    fn get_or_compute(&self, base: u64, len: u64, tile_mode: u8) -> Option<u64> {
        let key = (base, len, tile_mode);
        if let Some(hash) = self.values.borrow().get(&key).copied() {
            return Some(hash);
        }
        let hash = guest_sample_hash(base, len, tile_mode)?;
        self.values.borrow_mut().insert(key, hash);
        Some(hash)
    }

    fn clear(&self) {
        self.values.borrow_mut().clear();
    }

    /// Drop only probes whose guest byte range can be changed by a compute
    /// output. Read-only dispatches and writes into unrelated allocations do
    /// not invalidate immutable texture hashes for the whole submission.
    fn invalidate_ranges(&self, writes: &[(u64, u64)]) {
        if writes.is_empty() {
            return;
        }
        self.values.borrow_mut().retain(|&(base, len, _), _| {
            let end = base.saturating_add(len);
            !writes.iter().any(|&(write_base, write_len)| {
                if base == 0 || len == 0 || write_base == 0 || write_len == 0 {
                    return false;
                }
                let write_end = write_base.saturating_add(write_len);
                base < write_end && write_base < end
            })
        });
    }
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
    array: bool,
    volume: bool,
    tile_mode: u8,
    format: vk::Format,
) -> (u64, Option<TextureUpload>) {
    if crate::diagnostics::gpu_env().no_tex_cache {
        return (0, None);
    }
    let Some(hash) = sampling_scope(|scope| {
        // SAFETY: `draw_common` publishes a pointer to its sink-owned memo for
        // exactly the synchronous lifetime of this scope. See
        // `sampling_scope` for the matching same-thread lifetime invariant.
        let memo = unsafe { &*scope.sample_hash_memo };
        memo.get_or_compute(base, src_len, tile_mode)
    }) else {
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
                    && k.array == array
                    && k.volume == volume
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
                array,
                volume,
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

/// The number of distinct storage-array keys — 4 Dims x 3 storage formats
/// (`kyty_graphics::shader::storage_key_ordinal` is Dim-major).
const STORAGE_KEYS: usize = 12;

/// The canonical (Dim, storage format) key ordinal of a RW (storage) T#,
/// matching `kyty_graphics::shader::storage_key_ordinal`. A mixed shader
/// assigns per-key storage bindings in this order, so the host and the
/// SPIR-V generator agree on which `%textures2D_L<key>` array each RW T#
/// lands at (`binding_storage_index + position-among-present-keys`).
fn storage_key_ordinal(t: &kyty_graphics::shader::ShaderTextureResource) -> usize {
    // Delegates to the SPIR-V generator's own classifiers so the host and
    // the shader can never disagree on a RW T#'s array (drift here would
    // write a 3D volume's descriptor into the 2D array's binding — the
    // shader would then image-write through the wrong image).
    kyty_graphics::shader::storage_key_ordinal(
        kyty_graphics::shader::SampledDim::from_texture_type(t.type_()),
        kyty_graphics::shader::StorageFormat::from_unified_format(t.format()),
    ) as usize
}

/// The `(cube, volume, array)` view intent of a sampled T#, decided SOLELY
/// from its TYPE nibble — the single source of truth the bound `VkImageView`
/// type is built from so it can never disagree with the recompiled SPIR-V's
/// `OpTypeImage` (which the emitter derives from the SAME nibble via
/// `kyty_graphics::shader::SampledDim::from_texture_type`):
///
/// | T# type | `SampledDim` | view intent            | `VkImageViewType` |
/// |---------|--------------|------------------------|-------------------|
/// | 8, 9    | `Two`        | `(false,false,false)`  | `TYPE_2D`         |
/// | 10      | `Three`      | `(false,true ,false)`  | `TYPE_3D`         |
/// | 11      | `TwoArray`   | `(false,false,true )`  | `TYPE_2D_ARRAY`   |
/// | 13      | `TwoArray`   | `(false,false,true )`  | `TYPE_2D_ARRAY`   |
///
/// The array/volume flags are TYPE-driven, NOT layer-count-driven: a 2DArray
/// (type 11 or 13) whose depth field is 0 has one layer yet still declares
/// `Arrayed = 1` in SPIR-V, so it MUST bind a `TYPE_2D_ARRAY` view (with
/// `layer_count == 1`). Deriving the view from `layers > 1` instead was the
/// ASTRO.BOT array/cube device-loss (`VUID-vkCmdDispatch`: view type 2D under
/// an `Arrayed = 1` sampled image). The same holds for the SharpEmu #476
/// array-upload OOM fall-back, which drops to one layer but keeps type 13.
///
/// Types 12/14/15 (1DArray, 2DMsaa, 2DMsaaArray) never reach here — analysis
/// (`check_read_only_texture_type`) rewrites a plausible one to 2D and replaces
/// the poison with a 2D placeholder before decode, so this stays a named
/// refusal only for a genuinely-unhandled nibble.
fn texture_view_kind(ty: u8) -> Result<(bool, bool, bool), DrawError> {
    // Cross-check against the emitter's classifier so the two can never drift:
    // this table and `SampledDim::from_texture_type` are the ONE decision.
    use kyty_graphics::shader::SampledDim;
    let kind = match ty {
        // 8 = Texture1D. A 1D image is a 2D image one row tall, and the T#
        // already reports height5 = 0 => height 1, so the 2D decode path
        // handles it unchanged (measured on ASTRO.BOT: a 1x1 format-71
        // texture, tile mode 27). Kept a distinct arm rather than folding into
        // 9 so the disagreement is visible if a >1-row "1D" texture ever shows.
        8 => (false, false, false),
        // 9 = Texture2D.
        9 => (false, false, false),
        // 10 = 3D volume (measured: ASTRO.BOT's 240x135x64 froxel/LUT volumes).
        10 => (false, true, false),
        // 11 = guest Cube, sampled as a 2D array. RDNA's V_CUBE* sequence
        // already turns the direction into (s,t,face) before image_sample;
        // a Vulkan CUBE view would interpret those values as a direction a
        // second time and smear Minecraft's panorama radially.
        11 => (false, false, true),
        // 13 = 2DArray (measured: ASTRO.BOT's 1536x1536x3 array, tile 24 — the
        // T# depth field carries the layer count).
        13 => (false, false, true),
        other => {
            return Err(err(format!(
                "texture type {other} is not Texture2D (9), 3D (10), Cube (11) or 2DArray (13)"
            )));
        }
    };
    debug_assert_eq!(
        (
            SampledDim::from_texture_type(ty) == SampledDim::Cube,
            SampledDim::from_texture_type(ty) == SampledDim::Three,
            SampledDim::from_texture_type(ty) == SampledDim::TwoArray,
        ),
        kind,
        "bind-side view kind must equal the emitter's SampledDim for type {ty}",
    );
    Ok(kind)
}

/// Count of `decode_texture` calls whose T# selected a non-zero mip as the base
/// of its view (`base_level > 0`) — i.e. draws currently sampling the WRONG LOD.
/// See [`note_mip_view_base_level`]. Read by diagnostics; never resets.
pub static MIP_VIEW_BASE_LEVEL_IGNORED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Count of `decode_texture` calls whose T# carries a mip chain (`MAX_MIP > 0`).
///
/// Before SharpEmu #470 this was the SILENT failure: `base_level == 0` with
/// `MAX_MIP > 0` took the bytes at the descriptor base, which on GFX10 is the
/// mip TAIL, not mip 0 — no warning fired at all. Every one of these now goes
/// through [`crate::texture::tiling::base_mip_placement`].
pub static MIP_CHAIN_TEXTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Count of mip-chain textures whose mip 0 could NOT be located — an unsupported
/// swizzle mode/element size, or an in-tail sub-rectangle that did not fit its
/// block. These still read at the descriptor base (the pre-#470 bytes) and are
/// the ones to look at when a mipped texture renders scrambled.
pub static MIP_CHAIN_PLACEMENT_UNKNOWN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Mip levels the ALLOCATION carries, from `MAX_MIP` (RDNA2 table 45) clamped to
/// the most an extent of this size can hold.
///
/// `MAX_MIP` describes the resource; `BASE_LEVEL`/`LAST_LEVEL` describe one
/// view of it. Sizing the chain from a view is wrong — another descriptor for
/// the same allocation may expose a different subset. The clamp is what keeps a
/// malformed or stale `MAX_MIP` (all-ones descriptors do exist) from computing a
/// chain offset far past the real allocation.
///
/// Ported from SharpEmu `TextureDescriptor.ResourceMipLevels` /
/// `GetMaximumMipLevels` (#470, commit 6ee445f, GPL-2.0-or-later).
fn resource_mip_levels(max_mip: u8, width: u32, height: u32, depth: u32) -> u32 {
    let largest = width.max(height).max(depth).max(1);
    let maximum = 1 + largest.ilog2();
    (u32::from(max_mip) + 1).clamp(1, maximum)
}

/// Name the miss when a T# asks for a view starting at a non-zero mip.
///
/// `ShaderTextureResource::base_level()` is decoded but **consumed nowhere**:
/// [`decode_texture`] always reads from `base40()` and always uses the mip-0
/// extent (`width5()+1` x `height5()+1`). So a descriptor selecting
/// `base_level = N` is served mip 0's bytes at mip 0's dimensions — silently the
/// wrong content, which is exactly the failure mode this module's doc comment
/// forbids for formats and tile modes.
///
/// ## Why this counts instead of correcting
///
/// Correcting it means locating mip N inside the guest mip chain. Since SharpEmu
/// #470 this tree *can* locate mip **0** in a chain
/// ([`crate::texture::tiling::base_mip_placement`]), but an arbitrary level N
/// also needs the per-level tail slot, which that port does not carry. Emitting a
/// guessed offset would sample unrelated memory: strictly worse than a
/// wrong-but-consistent mip 0, and worse than saying so.
///
/// It is also **unmeasured**: no tracked title is known to set `base_level > 0`.
/// This counter is how that gets decided — run a title, read the count, and
/// implement the addressing only if it fires. Same discipline the tile-mode
/// refusal diagnostic established (port gated on measurement, not assumption).
///
/// ## Also fires for `MAX_MIP > 0`
///
/// `base_level == 0` with `MAX_MIP > 0` — an ordinary view of a mipped texture —
/// is the COMMON case and used to be completely silent while taking the mip
/// tail's bytes as if they were mip 0. It now warns and counts
/// ([`MIP_CHAIN_TEXTURES`]) whether or not the relocation succeeds, so a title
/// log says how many mipped textures a frame binds; the tiled decode then counts
/// the ones it could not place ([`MIP_CHAIN_PLACEMENT_UNKNOWN`]).
///
/// Rate-limited per `(base_level, last_level, max_mip)` triple so a hot draw loop
/// cannot flood the log; the counters still increment on every call.
fn note_mip_view_base_level(t: &kyty_graphics::shader::ShaderTextureResource) {
    let base_level = t.base_level();
    let max_mip = t.max_mip();
    if base_level == 0 && max_mip == 0 {
        return;
    }
    if base_level > 0 {
        MIP_VIEW_BASE_LEVEL_IGNORED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if max_mip > 0 {
        MIP_CHAIN_TEXTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    use std::collections::HashSet;
    use std::sync::Mutex;
    static WARNED: Mutex<Option<HashSet<(u8, u8, u8)>>> = Mutex::new(None);
    let key = (base_level, t.last_level(), max_mip);
    let first = WARNED
        .lock()
        .map(|mut set| set.get_or_insert_with(HashSet::new).insert(key))
        .unwrap_or(false);
    if !first {
        return;
    }
    let width = u32::from(t.width5()) + 1;
    let height = u32::from(t.height5()) + 1;
    if base_level > 0 {
        tracing::warn!(
            base_level,
            last_level = t.last_level(),
            max_mip,
            base = format_args!("{:#x}", t.base40()),
            width,
            height,
            tile_mode = t.tile_mode(),
            "T# selects a non-zero mip as its view base, but per-level mip \
             addressing is NOT implemented — serving mip 0 at mip 0's extent, so \
             this draw samples the WRONG LOD (degrade, not refuse). Further hits \
             on this (base_level, last_level, max_mip) triple are silent; see \
             MIP_VIEW_BASE_LEVEL_IGNORED for the total."
        );
    } else {
        tracing::warn!(
            max_mip,
            last_level = t.last_level(),
            base = format_args!("{:#x}", t.base40()),
            width,
            height,
            tile_mode = t.tile_mode(),
            resource_levels = resource_mip_levels(max_mip, width, height, 1),
            "T# carries a mip chain (MAX_MIP > 0). A GFX10 chain is stored \
             SMALLEST-FIRST, so mip 0 is at the END of the allocation, not at the \
             descriptor base — the tiled decode relocates the read (SharpEmu \
             #470). Further hits on this (base_level, last_level, max_mip) triple \
             are silent; see MIP_CHAIN_TEXTURES and \
             MIP_CHAIN_PLACEMENT_UNKNOWN for the totals."
        );
    }
}

/// Decode one T# into linear pixels a Vulkan sampled image can hold.
///
/// Formats and tile modes are added strictly from measurement: an unhandled
/// value is a named error carrying every raw field, so a run against the title
/// states exactly what to implement next — a guessed format number would render
/// silently-wrong colours, which is worse than an honest skip.
///
/// **Known gap, deliberately not guessed:** a T# whose `base_level > 0` is
/// served mip 0 — see [`note_mip_view_base_level`].
fn decode_texture(
    t: &kyty_graphics::shader::ShaderTextureResource,
) -> Result<TextureUpload, DrawError> {
    let _decode_timer = crate::vulkan::offscreen::StageTimer::start(
        &crate::vulkan::offscreen::DRAW_STAGE_DECODE_NS,
    );
    // A placeholder T# (base 0) stands in for a descriptor shader analysis
    // could not resolve — the all-ones "type 15 / format 511" poison a
    // runtime-/SRT-bound descriptor reads back as, replaced upstream by
    // `kyty_graphics` `check_read_only_texture_type`. Serve a 1x1
    // transparent-black dummy (no guest read) so the draw/dispatch proceeds
    // untextured instead of the whole shader being skipped. Mirrors the
    // null-V#-as-4-byte-zero-dummy storage-buffer path.
    if t.base40() == 0 {
        return Ok(placeholder_texture_dummy());
    }
    note_mip_view_base_level(t);
    let width = u32::from(t.width5()) + 1;
    let height = u32::from(t.height5()) + 1;
    if !(1..=16384).contains(&width) || !(1..=16384).contains(&height) {
        return Err(err(format!("texture extent {width}x{height} out of range")));
    }
    let (cube, volume, array) = texture_view_kind(t.type_())?;
    let mut layers = if cube {
        // A CUBE image's arrayLayers MUST be a positive multiple of 6 (Vulkan:
        // VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT ⇒ arrayLayers ≥ 6, and a CUBE view
        // ⇒ layer_count == 6). AMD `sq_img_rsrc_t` stores DEPTH = 6·cubes − 1,
        // so a well-formed single cube is depth=5 → 6 (Minecraft's 1024² skybox).
        // A malformed/misclassified cube T# with depth < 5 (measured: a Minecraft
        // draw at DWORD 1105 carries depth=0 → 1) would otherwise create a
        // 1-layer CUBE the driver accepts then loses the device sampling. Round
        // up to whole cubes so the image/view are always spec-valid.
        (u32::from(t.depth()) + 1).div_ceil(6) * 6
    } else if array {
        u32::from(t.depth()) + 1
    } else {
        1
    };
    let depth = if volume { u32::from(t.depth()) + 1 } else { 1 };
    if volume && !(1..=2048).contains(&depth) {
        return Err(err(format!("volume depth {depth} out of range")));
    }

    let (format, bpp) = texture_vk_format(t)?;
    // Guest layout, tiling and staging all address ELEMENTS, which for a
    // block-compressed format is a 4x4 texel block rather than a texel. The
    // `VkImage` keeps the texel extent (`width`/`height`); everything that
    // touches bytes below uses these. For every uncompressed format the block
    // extent is 1 and these are exactly `width`/`height`, so the arithmetic is
    // unchanged for the formats that worked before.
    let block_extent = crate::vulkan::offscreen::format_block_extent(format);
    let elements_wide = width.div_ceil(block_extent);
    let elements_high = height.div_ceil(block_extent);

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
            // A BC T#'s pitch is in texels like its width, so it converts to
            // elements the same way.
            let pitch = u32::from(t.pitch()).max(width).div_ceil(block_extent);
            let src_len =
                u64::from(pitch) * u64::from(elements_high) * u64::from(depth) * u64::from(bpp);
            let (hash, hit) = texture_cache_probe(
                t.base40(),
                src_len,
                width,
                height,
                layers,
                depth,
                cube,
                array,
                volume,
                t.tile_mode(),
                format,
            );
            if let Some(upload) = hit {
                return Ok(upload);
            }
            let tiled = read_guest_bytes_unaligned(t.base40(), src_len, "texture")?;
            let row = (elements_wide * bpp) as usize;
            let src_row = (pitch * bpp) as usize;
            let src_slice = src_row * elements_high as usize;
            let dst_slice = row * elements_high as usize;
            let mut pixels = alloc_zeroed(dst_slice * depth as usize, "texture decode")?;
            for z in 0..depth as usize {
                for y in 0..elements_high as usize {
                    let src = z * src_slice + y * src_row;
                    let dst = z * dst_slice + y * row;
                    pixels[dst..dst + row].copy_from_slice(&tiled[src..src + row]);
                }
            }
            (pixels, hash)
        }
        // GTA V's first tiled volume is the degenerate 1x1x1 RGBA8 case.
        // Coordinate (0, 0, 0) is the first element of every block layout, so
        // reading that one element is exact without guessing the still-open
        // general GFX10 3D tile-mode-5 address equation. Keep the format and
        // extent guard deliberately narrow: larger/non-RGBA8 volumes continue
        // to name the unsupported layout below instead of silently detiling
        // with a 2D equation.
        5 if volume && width == 1 && height == 1 && depth == 1 && t.format() == 56 => {
            let src_len = u64::from(bpp);
            let (hash, hit) = texture_cache_probe(
                t.base40(),
                src_len,
                width,
                height,
                layers,
                depth,
                cube,
                array,
                volume,
                t.tile_mode(),
                format,
            );
            if let Some(upload) = hit {
                return Ok(upload);
            }
            let pixels = read_guest_bytes_unaligned(t.base40(), src_len, "texture")?;
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
            // `trailing_zeros` is a log2 only for a power of two: a 3-byte
            // element would read as 1 byte per texel and a 32-byte one would
            // index past the last swizzle-table row. Refuse by name.
            // A tiled BC surface swizzles its 4x4 BLOCKS with the same
            // equations at an element size the tables already cover (8/16 B),
            // so the whole detile runs on element dimensions.
            let Some(face_tiled) = crate::texture::tiling::tiled_byte_count_for_mode(
                mode,
                elements_wide,
                elements_high,
                bpp_log2,
            ) else {
                return Err(err(format!(
                    "texture tile mode {mode} with {bpp}-byte elements is not a supported \
                     swizzle element size (base={:#x} {width}x{height} format={})",
                    t.base40(),
                    t.format()
                )));
            };
            let face_tiled = face_tiled as usize;
            let face_linear = (elements_wide * elements_high * bpp) as usize;
            // Mip 0 is NOT at the descriptor base for a mipped surface: a GFX10
            // AddrLib chain is stored SMALLEST-FIRST — the small levels pack into
            // the first swizzle block (the mip tail), the rest follow in
            // decreasing size, and mip 0 lands at the END of the allocation. So
            // reading `base40()` for a MAX_MIP > 0 texture decodes the TAIL at
            // mip 0's extent (SharpEmu #470 saw scrambled menu text and repeated
            // icons). `None` = nothing to relocate (single level, or a placement
            // this port cannot compute) — then keep the base and count it.
            let mip_placement = if crate::diagnostics::gpu_env().no_mip_chain {
                None
            } else {
                crate::texture::tiling::base_mip_placement(
                    mode,
                    elements_wide,
                    elements_high,
                    bpp_log2,
                    resource_mip_levels(t.max_mip(), width, height, depth),
                )
            };
            if t.max_mip() > 0 && mip_placement.is_none() {
                MIP_CHAIN_PLACEMENT_UNKNOWN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let mip0_offset = mip_placement.map_or(0, |p| p.byte_offset);
            // A mipped array layer strides by its WHOLE chain, not by mip 0's
            // block grid.
            let layer_stride = mip_placement
                .and_then(|p| usize::try_from(p.chain_slice_bytes).ok())
                .filter(|stride| *stride >= face_tiled)
                .unwrap_or(face_tiled);
            let mip_tail_element = mip_placement.and_then(|p| p.tail_element);
            // Bytes this decode depends on, measured from the descriptor base —
            // the cache key stays the allocation base while the content hash
            // covers everything up to and including mip 0. Identical to
            // `face_tiled * layers` whenever there is no chain to relocate.
            let source_span = |layers: u32| {
                mip0_offset + layer_stride as u64 * u64::from(layers - 1) + face_tiled as u64
            };
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
                    source_span(layers),
                    width,
                    height,
                    layers,
                    depth,
                    cube,
                    array,
                    volume,
                    t.tile_mode(),
                    format,
                )
            };
            let (hash, hit) = probe(layers);
            if let Some(upload) = hit {
                return Ok(upload);
            }
            // One layer's linear elements from that layer's tiled bytes. Mip 0 is
            // either a block grid of its own, or — when the whole chain fits the
            // tail block — a sub-rectangle of that one detiled block.
            let detile_face = |src: &[u8]| match mip_tail_element {
                Some((tail_x, tail_y)) => crate::texture::tiling::detile_mip_tail_base(
                    mode,
                    src,
                    elements_wide,
                    elements_high,
                    bpp_log2,
                    tail_x,
                    tail_y,
                ),
                None => crate::texture::tiling::detile_64kb(
                    mode,
                    src,
                    elements_wide,
                    elements_high,
                    bpp_log2,
                ),
            };
            // Mip 0 is `mip0_offset` past the base, and each array layer strides
            // by its whole chain. Both are 0 / `face_tiled` for an unmipped
            // surface, so this is the previous single contiguous read verbatim.
            let read_base = t.base40() + mip0_offset;
            let src_len = layer_stride as u64 * u64::from(layers - 1) + face_tiled as u64;
            let tiled = match read_guest_bytes_unaligned(read_base, src_len, "texture") {
                Ok(tiled) => tiled,
                // A CUBE view REQUIRES a multiple-of-6 layer_count; the array
                // "drop to one base layer" path below would recreate the very
                // <6-layer CUBE image the driver accepts then loses the device
                // sampling. Keep the cube's layer count and read face by face,
                // zero-filling any face whose guest bytes are unmapped.
                Err(_) if cube => {
                    let mut pixels = alloc_zeroed(face_linear * layers as usize, "texture decode")?;
                    for layer in 0..layers as usize {
                        let face_addr = read_base + (layer * layer_stride) as u64;
                        if let Ok(single) =
                            read_guest_bytes_unaligned(face_addr, face_tiled as u64, "texture")
                        {
                            // Element extents, not texel ones: a BC face's linear
                            // size is `elements_wide * elements_high * bpp`, and
                            // detiling at texel extents would produce a 16x
                            // longer buffer than the destination slice.
                            if let Some(face) = detile_face(&single) {
                                pixels[layer * face_linear..(layer + 1) * face_linear]
                                    .copy_from_slice(&face);
                            }
                        }
                    }
                    return Ok(texture_upload_from(
                        t, width, height, format, pixels, layers, cube, array, volume, depth, hash,
                    ));
                }
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
                        read_guest_bytes_unaligned(read_base, face_tiled as u64, "texture")?;
                    let Some(face) = detile_face(&single) else {
                        return Err(err(format!(
                            "texture tile mode {mode} mip-0 detile failed for a single layer \
                             (base={:#x} {width}x{height} format={})",
                            t.base40(),
                            t.format()
                        )));
                    };
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
                        // The array read overran and dropped to one layer, but
                        // the T# is still type 13: the SPIR-V stays Arrayed = 1,
                        // so the view must stay TYPE_2D_ARRAY (layer_count 1).
                        array,
                        volume,
                        depth,
                        single_hash,
                    ));
                }
                Err(e) => return Err(e),
            };
            let mut pixels = alloc_zeroed(face_linear * layers as usize, "texture decode")?;
            for layer in 0..layers as usize {
                let start = layer * layer_stride;
                let src = &tiled[start..start + face_tiled];
                let Some(face) = detile_face(src) else {
                    return Err(err(format!(
                        "texture tile mode {mode} mip-0 detile failed for layer {layer} \
                         (base={:#x} {width}x{height} format={})",
                        t.base40(),
                        t.format()
                    )));
                };
                pixels[layer * face_linear..(layer + 1) * face_linear].copy_from_slice(&face);
            }
            (pixels, hash)
        }
        other => {
            note_unsupported_tile_mode(other, t.format());
            return Err(err(format!(
                "texture tile mode {other} not implemented \
                 (base={:#x} {width}x{height} format={})",
                t.base40(),
                t.format()
            )));
        }
    };
    // Content probe (RAEEN_TRACE_DRAWS): is the texture the PS samples
    // actually EMPTY? Thousands of title draws rasterize fine in-tree, and
    // most title draws alpha-blend (SRC_ALPHA/ONE_MINUS_SRC_ALPHA) — a PS
    // sampling an all-zero texture emits alpha 0, which is byte-identical to
    // "no coverage" in every frame probe. If these log all-zero, the GPU is
    // correctly rendering an EMPTY UI and the blocker is upstream (Gameface
    // never paints the menu), not in the GPU at all.
    if crate::diagnostics::gpu_env().trace_draws {
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
    if (cube || array) && crate::diagnostics::gpu_env().trace_textures {
        use std::sync::atomic::{AtomicU32, Ordering};
        static LAYERED_SEEN: AtomicU32 = AtomicU32::new(0);
        if width >= 512 || height >= 512 || LAYERED_SEEN.fetch_add(1, Ordering::Relaxed) < 24 {
            let non_zero = pixels.iter().filter(|&&byte| byte != 0).count();
            let draw = sampling_scope(|scope| {
                Some((
                    scope.vs_addr,
                    scope.ps_addr,
                    scope.primitive,
                    scope.vertex_count,
                    scope.indexed,
                    scope.first_attribute,
                    scope.first_stride,
                    scope.index_type,
                    scope.vertex_head.clone(),
                    scope.index_head.clone(),
                ))
            });
            tracing::info!(
                base = format_args!("{:#x}", t.base40()),
                texture_type = t.type_(),
                extent = format_args!("{width}x{height}x{layers}"),
                format = t.format(),
                tile_mode = t.tile_mode(),
                bytes = pixels.len(),
                non_zero,
                cube,
                array,
                draw = ?draw,
                "layered sampled texture decoded"
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
        array,
        volume,
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
    array: bool,
    volume: bool,
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
        array,
        volume,
        depth,
        render_target: None,
        guest_base: t.base40(),
        sample_hash,
        cached: false,
    }
}

/// A 1x1 transparent-black sampled image standing in for a placeholder (base 0)
/// T# — see the early return in [`decode_texture`]. No guest read; the sample
/// result is a deterministic transparent black, so a draw/dispatch whose
/// texture descriptor could not be statically resolved renders untextured
/// instead of being skipped (M5: maximize geometry on screen, glitches OK).
fn placeholder_texture_dummy() -> TextureUpload {
    TextureUpload {
        width: 1,
        height: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        pixels: vec![0u8; 4],
        layers: 1,
        cube: false,
        array: false,
        volume: false,
        depth: 1,
        render_target: None,
        guest_base: 0,
        sample_hash: 0,
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
/// ASTRO.BOT's 240x135x64 format-71 volumes). Types 11/13 are writable
/// 2D-array views spanning `BASE_ARRAY..=LAST_ARRAY` (the latter is exposed
/// by `depth()`). Minecraft builds its panorama with six one-layer views:
/// base/last 0/0 through 5/5.
/// Format 71 uploads as
/// `R16G16B16A16_SFLOAT` (8 B/texel), format 77 as
/// `R32G32B32A32_SFLOAT` (16 B/texel), matching the recompiled storage-image
/// format. Everything else keeps the RGBA8 view. The content is a
/// best-effort seed: a UAV is typically fully overwritten by the dispatch,
/// so an unknown format or unreadable guest range zero-fills with a
/// once-per-process warning instead of failing the dispatch.
fn read_storage_image(
    t: &kyty_graphics::shader::ShaderTextureResource,
) -> Result<StorageImageUpload, DrawError> {
    let width = u32::from(t.width5()) + 1;
    let height = u32::from(t.height5()) + 1;
    // Type-driven, exactly like `array` below: a type-10 T# is a `Dim3D`
    // storage image in the recompiled SPIR-V no matter how many slices its
    // DEPTH field names, so the flag — not `depth > 1` — decides the host
    // image/view type. See `StorageImageUpload::volume`.
    let volume = t.type_() == 10;
    let depth = if volume { u32::from(t.depth()) + 1 } else { 1 };
    let array = matches!(t.type_(), 11 | 13);
    let base_array = if array { u32::from(t.base_array5()) } else { 0 };
    let last_array = u32::from(t.depth());
    let layers = if array {
        last_array
            .checked_sub(base_array)
            .map_or(0, |last_from_base| last_from_base + 1)
    } else {
        1
    };
    if !(1..=16384).contains(&width)
        || !(1..=16384).contains(&height)
        || !(1..=2048).contains(&depth)
        || !(1..=2048).contains(&layers)
    {
        return Err(err(format!(
            "storage image extent {width}x{height}x{depth}x{layers} out of range"
        )));
    }
    // Must agree with `kyty-graphics` `storage_texture_dim_format` (which
    // declares `%ImageL` as Rgba16f for 71 and Rgba32f for 77).
    let (format, texel) = match t.format() {
        71 => (vk::Format::R16G16B16A16_SFLOAT, 8u64),
        77 => (vk::Format::R32G32B32A32_SFLOAT, 16u64),
        _ => (vk::Format::R8G8B8A8_UNORM, 4u64),
    };
    let size = u64::from(width) * u64::from(height) * u64::from(depth) * u64::from(layers) * texel;
    let allocation_base = t.base40();
    let linear_layer_bytes = u64::from(width) * u64::from(height) * texel;
    let guest_layer_bytes = if array && t.tile_mode() != 0 {
        crate::texture::tiling::tiled_byte_count_for_mode(
            t.tile_mode(),
            width,
            height,
            (texel as u32).trailing_zeros(),
        )
        .map_or(linear_layer_bytes, u64::from)
    } else {
        linear_layer_bytes
    };
    let base =
        allocation_base.saturating_add(guest_layer_bytes.saturating_mul(u64::from(base_array)));
    let readable = matches!(t.format(), 71 | 77) || storage_image_format_is_32bpp(t.format());
    let pixels = if readable {
        if depth > 1 || t.tile_mode() == 0 {
            read_guest_bytes(base, size, "storage image").ok()
        } else if crate::texture::tiling::swizzle_table(t.tile_mode()).is_some() {
            // Storage arrays are guest-visible tiled surfaces just like the
            // sampled view that consumes them later. Detile every layer before
            // uploading it to the host storage image; writeback performs the
            // exact inverse. This is the path Minecraft uses to assemble its
            // six 1024x1024 panorama faces.
            let bpp_log2 = (texel as u32).trailing_zeros();
            let face_tiled = crate::texture::tiling::tiled_byte_count_for_mode(
                t.tile_mode(),
                width,
                height,
                bpp_log2,
            )
            .expect("guarded by swizzle_table") as usize;
            let face_linear = (u64::from(width) * u64::from(height) * texel) as usize;
            read_guest_bytes(
                base,
                (face_tiled as u64).saturating_mul(u64::from(layers)),
                "storage image",
            )
            .ok()
            .and_then(|tiled| {
                let mut linear = alloc_zeroed(
                    face_linear.saturating_mul(layers as usize),
                    "linear storage-image array",
                )
                .ok()?;
                for layer in 0..layers as usize {
                    let face = crate::texture::tiling::detile_64kb(
                        t.tile_mode(),
                        &tiled[layer * face_tiled..(layer + 1) * face_tiled],
                        width,
                        height,
                        bpp_log2,
                    )
                    .expect("table-checked above");
                    linear[layer * face_linear..(layer + 1) * face_linear].copy_from_slice(&face);
                }
                Some(linear)
            })
        } else {
            None
        }
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
                    extent = format_args!("{width}x{height}x{depth}x{layers}"),
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
    if matches!(t.type_(), 11 | 13) && crate::diagnostics::gpu_env().trace_textures {
        use std::sync::atomic::{AtomicU32, Ordering};
        static LAYERED_STORAGE_SEEN: AtomicU32 = AtomicU32::new(0);
        if width >= 512
            || height >= 512
            || LAYERED_STORAGE_SEEN.fetch_add(1, Ordering::Relaxed) < 24
        {
            tracing::info!(
                base = format_args!("{base:#x}"),
                allocation_base = format_args!("{allocation_base:#x}"),
                texture_type = t.type_(),
                base_array,
                last_array,
                descriptor_depth = u32::from(t.depth()) + 1,
                extent = format_args!("{width}x{height}x{depth}x{layers}"),
                format = t.format(),
                tile_mode = t.tile_mode(),
                bytes = pixels.len(),
                non_zero = pixels.iter().any(|&byte| byte != 0),
                "layered storage texture decoded for a 2D-array UAV"
            );
        }
    }
    Ok(StorageImageUpload {
        width,
        height,
        depth,
        volume,
        layers,
        array,
        tile_mode: t.tile_mode(),
        format,
        pixels: Arc::new(pixels),
        guest_base: base,
    })
}

/// Convert a linear host storage-image readback back to the guest descriptor's
/// swizzled layout. The inverse is performed in [`read_storage_image`].
fn encode_storage_image_writeback(
    width: u32,
    height: u32,
    depth: u32,
    layers: u32,
    tile_mode: u8,
    texel: u32,
    linear: &[u8],
) -> Result<Vec<u8>, DrawError> {
    if depth > 1 || tile_mode == 0 {
        let mut guest = alloc_zeroed(linear.len(), "linear storage-image writeback")?;
        guest.copy_from_slice(linear);
        return Ok(guest);
    }
    let bpp_log2 = texel.trailing_zeros();
    let face_linear = width as usize * height as usize * texel as usize;
    let expected = face_linear.saturating_mul(layers as usize);
    if linear.len() < expected {
        return Err(err(format!(
            "storage image readback is {} B, smaller than {width}x{height}x{layers}x{texel} \
             ({expected} B)",
            linear.len()
        )));
    }
    let face_tiled =
        crate::texture::tiling::tiled_byte_count_for_mode(tile_mode, width, height, bpp_log2)
            .ok_or_else(|| {
                err(format!(
                    "storage-image writeback tile mode {tile_mode} not implemented"
                ))
            })? as usize;
    let mut tiled = alloc_zeroed(
        face_tiled.saturating_mul(layers as usize),
        "storage image tiled writeback",
    )?;
    for layer in 0..layers as usize {
        let face = &linear[layer * face_linear..(layer + 1) * face_linear];
        let output = &mut tiled[layer * face_tiled..(layer + 1) * face_tiled];
        if !crate::texture::tiling::tile_64kb_into(tile_mode, face, output, width, height, bpp_log2)
        {
            return Err(err(format!(
                "storage-image writeback tile mode {tile_mode} could not encode \
                 {width}x{height} layer {layer}"
            )));
        }
    }
    Ok(tiled)
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
    map: *const HashMap<u64, Arc<RenderedImage>>,
    live: Vec<(u64, u32, u32, i32)>,
    self_base: u64,
    resolution_scale: f32,
    vs_addr: u64,
    ps_addr: u64,
    primitive: u32,
    vertex_count: u32,
    indexed: bool,
    first_attribute: Option<VertexAttributeData>,
    first_stride: Option<u32>,
    index_type: Option<vk::IndexType>,
    vertex_head: Vec<u8>,
    index_head: Vec<u8>,
    /// Snapshot of the persistent-texture cache (stage D): every cached
    /// texture's key and content sample-hash. `decode_texture` consults it to
    /// skip the guest read + detile + upload for a texture whose fresh
    /// sample-hash matches; empty when the cache is empty or disabled.
    cached_textures: Vec<(crate::vulkan::cache::TextureKey, u64)>,
    /// Submission-local sample-hash memo owned by the active draw sink.
    sample_hash_memo: *const GuestSampleHashMemo,
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

fn scaled_sampling_extent(width: u32, height: u32, factor: f32) -> (u32, u32) {
    let factor = if factor.is_finite() {
        factor.clamp(0.5, 4.0)
    } else {
        1.0
    };
    let scale = |value: u32| ((value as f32 * factor).round() as u32).max(1);
    (scale(width), scale(height))
}

fn matching_live_target(
    live: &[(u64, u32, u32, i32)],
    base: u64,
    width: u32,
    height: u32,
    format: i32,
    resolution_scale: f32,
) -> Option<(u32, u32)> {
    let scaled = scaled_sampling_extent(width, height, resolution_scale);
    live.iter()
        .find(|(b, w, h, f)| *b == base && *w == width && *h == height && *f == format)
        .or_else(|| {
            live.iter()
                .find(|(b, w, h, f)| *b == base && (*w, *h) == scaled && *f == format)
        })
        .map(|(_, width, height, _)| (*width, *height))
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
        matching_live_target(
            &scope.live,
            base,
            width,
            height,
            format.as_raw(),
            scope.resolution_scale,
        )
        .map(|(target_width, target_height)| TextureUpload {
            width: target_width,
            height: target_height,
            format,
            pixels: Vec::new(),
            layers: 1,
            cube: false,
            array: false,
            volume: false,
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
fn render_target_pixels(base: u64, width: u32, height: u32) -> Option<(u32, u32, Vec<u8>)> {
    sampling_scope(|scope| {
        // SAFETY: `scope.map` points at the sink's framebuffer map, alive and
        // unmutated for the published span (see `SAMPLING_SCOPE`).
        let map = unsafe { &*scope.map };
        let scaled = scaled_sampling_extent(width, height, scope.resolution_scale);
        map.get(&base)
            .filter(|img| {
                (img.width, img.height) == (width, height) || (img.width, img.height) == scaled
            })
            .map(|img| (img.width, img.height, img.pixels.clone()))
    })
}

/// Whether a decoded sampled texture can be replaced by the CPU framebuffer
/// snapshot at the same guest base.
///
/// The framebuffer map contains one plain 2D attachment. It cannot stand in for
/// a cube, array, or volume even when the first face happens to share the same
/// base and extent. Minecraft exposed the consequence: a six-face 1024x1024
/// cube (24 MiB) was replaced with one 4 MiB render-target snapshot while
/// retaining `layers == 6`, so `vkCmdCopyBufferToImage` read past the staging
/// buffer and reset the device.
fn can_replace_with_render_target_pixels(upload: &TextureUpload) -> bool {
    upload.render_target.is_none()
        && !upload.cube
        && !upload.array
        && upload.layers == 1
        && upload.depth == 1
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
/// Tunable via `RAEEN_MAX_STAGE_TEXTURE_MIB`; default 96 MiB.
fn stage_texture_byte_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("RAEEN_MAX_STAGE_TEXTURE_MIB")
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

/// Process-wide count of draws/dispatches skipped because a bound storage-buffer
/// V# used an element-addressing modifier (`add_tid` / `swizzle`) the recompiled
/// SSBO access does not model yet. Distinct from [`stage_texture_cap_skips`] and
/// from an OOB mode (which is now admitted): a growing count is the honest,
/// title-specific measure of what an add-tid/swizzle SPIR-V follow-up would
/// recover. See the split guard in [`prepare_stage_binding`].
static STORAGE_ADDRESSING_SKIPS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Total draws/dispatches skipped for an unsupported storage-buffer element
/// addressing modifier (`add_tid` / `swizzle`).
#[must_use]
pub fn storage_addressing_skips() -> u64 {
    STORAGE_ADDRESSING_SKIPS.load(std::sync::atomic::Ordering::Relaxed)
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

/// Expected linear byte size of one storage-image (UAV) T#:
/// `width * height * depth * layers * texel` (`texel` = 8 for format 71
/// RGBA16F, 16 for format 77 RGBA32F, else 4), matching
/// `read_storage_image`.
fn expected_storage_image_bytes(t: &kyty_graphics::shader::ShaderTextureResource) -> u64 {
    let width = u64::from(u32::from(t.width5()) + 1);
    let height = u64::from(u32::from(t.height5()) + 1);
    let depth = if t.type_() == 10 {
        u64::from(u32::from(t.depth()) + 1)
    } else {
        1
    };
    let layers = if matches!(t.type_(), 11 | 13) {
        u64::from(
            u32::from(t.depth())
                .checked_sub(u32::from(t.base_array5()))
                .map_or(0, |last_from_base| last_from_base + 1),
        )
    } else {
        1
    };
    let texel = match t.format() {
        71 => 8,
        77 => 16,
        _ => 4,
    };
    width
        .saturating_mul(height)
        .saturating_mul(depth)
        .saturating_mul(layers)
        .saturating_mul(texel)
}

/// Desired raw EUD-window snapshot size in bytes (SharpEmu port): at least
/// the shader's required prefix, floored at 256 KiB, page-rounded up, capped
/// at 16 MiB (`reference/sharpemu/src/SharpEmu.ShaderCompiler/`
/// `Gen5ShaderScalarEvaluator.cs:1952-1960` — the 256 KiB/page-round window —
/// and `:69` — `MaxGlobalMemoryBindingBytes` = 16 MiB).
/// Report each distinct unsupported `(tile_mode, format)` pair exactly once.
///
/// A texture whose swizzle mode has no ported equation makes [`decode_texture`]
/// refuse, the draw drop, and the frame come back as the BLACK FRAME warning in
/// `agc_exec.rs` — with nothing naming *which* mode was responsible. Raeen
/// implements swizzle modes 5/9/24/27; SharpEmu additionally models 1/4/8 via a
/// block table (`reference/sharpemu/src/SharpEmu.Libs/Agc/GnmTiling.cs:821-861`,
/// whose own comment concedes it is a model rather than a transcribed AddrLib
/// PATINFO table). Porting that is only worth its inexactness if a real title
/// actually binds those modes — this line is what turns that into a measurement
/// instead of a guess. Rate-limited per pair, so a per-draw miss cannot flood.
fn note_unsupported_tile_mode(mode: u8, format: u16) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u8, u16)>>> = Mutex::new(None);
    let first = SEEN
        .lock()
        .map(|mut set| set.get_or_insert_with(HashSet::new).insert((mode, format)))
        .unwrap_or(false);
    if first {
        tracing::warn!(
            tile_mode = mode,
            format,
            "unsupported texture swizzle mode — this draw is DROPPED (a likely BLACK FRAME \
             contributor); Raeen implements modes 5/9/24/27. Further textures with this \
             (mode, format) are silent."
        );
    }
}

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

fn sampler_address_mode(clamp: u8) -> vk::SamplerAddressMode {
    match clamp {
        // SQ_TEX_WRAP / SQ_TEX_MIRROR.
        0 => vk::SamplerAddressMode::REPEAT,
        1 => vk::SamplerAddressMode::MIRRORED_REPEAT,
        // SQ_TEX_CLAMP_LAST_TEXEL. Minecraft uses this for its 64x64 skin
        // atlas; treating it as REPEAT samples the opposite atlas edge.
        2 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        // Vulkan has no separate "mirror once + last texel/half border/border"
        // modes. MIRROR_CLAMP_TO_EDGE preserves the one-mirror coordinate rule
        // and is the closest representable behaviour.
        3 | 5 | 7 => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        // HALF_BORDER and BORDER both use the descriptor's border colour.
        4 | 6 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        _ => vk::SamplerAddressMode::REPEAT,
    }
}

fn sampler_filter(filter: u8) -> vk::Filter {
    match filter {
        // Point and anisotropic-point retain nearest texel selection. Full
        // anisotropy needs the guest ratio and device feature wired together;
        // it must not silently turn point sampling into bilinear filtering.
        0 | 2 => vk::Filter::NEAREST,
        1 | 3 => vk::Filter::LINEAR,
        _ => vk::Filter::NEAREST,
    }
}

fn sampler_state(sampler: &ShaderSamplerResource) -> SamplerState {
    SamplerState {
        mag_filter: sampler_filter(sampler.xy_mag_filter()),
        min_filter: sampler_filter(sampler.xy_min_filter()),
        mipmap_mode: if sampler.mip_filter() == 2 {
            vk::SamplerMipmapMode::LINEAR
        } else {
            vk::SamplerMipmapMode::NEAREST
        },
        address_mode_u: sampler_address_mode(sampler.clamp_x()),
        address_mode_v: sampler_address_mode(sampler.clamp_y()),
        address_mode_w: sampler_address_mode(sampler.clamp_z()),
    }
}

fn prepare_stage_binding(
    bind: &ShaderBindResources,
    stage: vk::ShaderStageFlags,
) -> Result<ShaderStageBinding, DrawError> {
    prepare_stage_binding_inner(bind, stage, None, None)
}

type ComputeStorageSnapshots = HashMap<(u64, usize), Arc<Vec<u8>>>;
type ComputeImageSnapshots = HashMap<[u32; 8], StorageImageUpload>;

fn prepare_compute_stage_binding(
    bind: &ShaderBindResources,
    storage_snapshots: &mut ComputeStorageSnapshots,
    image_snapshots: &mut ComputeImageSnapshots,
) -> Result<ShaderStageBinding, DrawError> {
    prepare_stage_binding_inner(
        bind,
        vk::ShaderStageFlags::COMPUTE,
        Some(storage_snapshots),
        Some(image_snapshots),
    )
}

fn prepare_stage_binding_inner(
    bind: &ShaderBindResources,
    stage: vk::ShaderStageFlags,
    mut compute_snapshots: Option<&mut ComputeStorageSnapshots>,
    mut compute_image_snapshots: Option<&mut ComputeImageSnapshots>,
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
    let mut storage_bases = Vec::with_capacity(storage_num);
    let mut storage_sizes = Vec::with_capacity(storage_num);
    for (index, resource) in bind.storage_buffers.buffers[..storage_num]
        .iter()
        .enumerate()
    {
        // Split the old blanket refusal (add_tid || swizzle || OOB) into the two
        // classes it conflated — they are NOT the same risk:
        //
        // * `out_of_bounds` (2-bit OOB_SELECT) only defines what an out-of-RANGE
        //   element returns; on RDNA that is zero (and out-of-range writes are
        //   dropped). It does NOT change in-range element addressing. We already
        //   read exactly `stride*num_records` bytes, bind that as the SSBO, and
        //   zero-pad to a dword multiple — so an out-of-range read already yields
        //   0 and an out-of-range write already lands in pad we truncate off the
        //   writeback. Admitting any OOB mode is therefore a no-op relative to what
        //   we already do, and matches hardware's clamp-to-zero. This was the
        //   decisive over-conservatism: Minecraft's async-compute dispatches set a
        //   NON-ZERO OOB mode on their storage V#s and were refused outright
        //   (~48 skips/run), and — before Fix 1 — that refusal aborted the ACB
        //   walk and deadlocked the title's submit worker. Admit it.
        //
        // * `add_tid` (per-lane TID byte offset) and `swizzle` (interleaved
        //   sub-element addressing) genuinely change WHICH bytes each invocation
        //   touches, and the recompiled SSBO access does not model that yet.
        //   Keep refusing them — but as a COUNTED, per-flag-LOGGED skip so a
        //   verify run reveals exactly which modifier a title actually needs.
        //   Fix 1 guarantees this refusal degrades to a skipped dispatch, never a
        //   hang, so admitting OOB while deferring add_tid/swizzle is safe to ship.
        let usage = bind.storage_buffers.usages[index];
        if usage != ShaderStorageUsage::Constant
            && (resource.add_tid() || resource.swizzle_enabled())
        {
            let n = STORAGE_ADDRESSING_SKIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 8 || (n + 1).is_power_of_two() {
                tracing::warn!(
                    stage = ?stage,
                    index,
                    add_tid = resource.add_tid(),
                    swizzle = resource.swizzle_enabled(),
                    out_of_bounds = resource.out_of_bounds(),
                    stride = resource.stride(),
                    num_records = resource.num_records(),
                    addr = format_args!("{:#x}", resource.base48()),
                    total_addressing_skips = n + 1,
                    "storage buffer uses element-addressing modifier (add_tid/swizzle) — \
                     not yet modeled in the SSBO access; skipping the draw/dispatch \
                     (degrades to a glitch, never a hang)"
                );
            }
            return Err(err(format!(
                "storage buffer {index} uses unsupported element addressing \
                 (add_tid={}, swizzle={})",
                resource.add_tid(),
                resource.swizzle_enabled()
            )));
        }
        if resource.out_of_bounds() != 0 {
            // Rate-limited breadcrumb so a verify run can confirm the OOB
            // relaxation is what admitted these dispatches (and that no add_tid/
            // swizzle slipped through as a different mode).
            debug!(
                stage = ?stage,
                index,
                out_of_bounds = resource.out_of_bounds(),
                stride = resource.stride(),
                num_records = resource.num_records(),
                addr = format_args!("{:#x}", resource.base48()),
                "storage buffer OOB_SELECT admitted (clamp-to-zero already modeled)"
            );
        }
        let descriptor_size =
            buffer_byte_size(resource).ok_or_else(|| err("storage buffer size overflow"))?;
        let measured_prefix = u64::from(bind.storage_buffers.required_bytes[index]);
        let size = if usage == ShaderStorageUsage::Constant && measured_prefix != 0 {
            measured_prefix
        } else {
            descriptor_size
        };
        if usage == ShaderStorageUsage::Constant
            && (resource.add_tid() || resource.swizzle_enabled())
        {
            debug!(
                stage = ?stage,
                index,
                add_tid = resource.add_tid(),
                swizzle = resource.swizzle_enabled(),
                descriptor_size,
                required_bytes = size,
                "scalar constant-buffer load ignores MUBUF element-addressing modifiers"
            );
        }
        let base = resource.base48();
        let padded_size = if base == 0 || size == 0 {
            4
        } else {
            (size as usize).div_ceil(4) * 4
        };
        let snapshot_key = (base, padded_size);
        let cached = compute_snapshots
            .as_deref()
            .and_then(|snapshots| snapshots.get(&snapshot_key))
            .cloned();
        let cache_hit = cached.is_some();
        let mut bytes = if cache_hit {
            Vec::new()
        } else if resource.base48() == 0 || size == 0 {
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
            read_guest_bytes(resource.base48(), size, "storage buffer").map_err(|source| {
                err(format!(
                    "storage buffer {index} descriptor rejected \
                     (stage={stage:?}, start_register={}, slot={}, usage={:?}, \
                     extended={}, fields=[{:#010x}, {:#010x}, {:#010x}, {:#010x}], \
                     base={base:#x}, stride={}, num_records={}, size={size:#x}): {source}",
                    bind.storage_buffers.start_register[index],
                    bind.storage_buffers.slots[index],
                    bind.storage_buffers.usages[index],
                    bind.storage_buffers.extended[index],
                    resource.fields[0],
                    resource.fields[1],
                    resource.fields[2],
                    resource.fields[3],
                    resource.stride(),
                    resource.num_records(),
                ))
            })?
        };
        // The SSBO view is an array of 32-bit elements, so pad the upload to
        // a dword multiple (a V# byte size need not be one — the recompiler
        // dropped Kyty's alignment EXIT). The writeback truncates back to
        // `size` so the pad bytes never reach guest memory.
        if !cache_hit {
            bytes.resize(padded_size, 0);
        }
        let bytes = cached.unwrap_or_else(|| {
            let bytes = Arc::new(bytes);
            if base != 0
                && let Some(snapshots) = compute_snapshots.as_deref_mut()
            {
                snapshots.insert(snapshot_key, Arc::clone(&bytes));
            }
            bytes
        });
        debug!(
            stage = ?stage,
            index,
            addr = format_args!("{:#x}", resource.base48()),
            len = bytes.len(),
            head = format_args!("{:02x?}", &bytes[..bytes.len().min(16)]),
            "stage storage buffer read"
        );
        if crate::diagnostics::gpu_env().trace_draws {
            // This is a diagnostic-only O(n) scan. Minecraft binds multi-MiB
            // V# resources many times per frame; computing it unconditionally
            // consumed several milliseconds even though the trace was off.
            let all_zero = bytes.iter().all(|&b| b == 0);
            use std::sync::atomic::{AtomicU32, Ordering};
            static COMPUTE_SEEN: AtomicU32 = AtomicU32::new(0);
            static GRAPHICS_SEEN: AtomicU32 = AtomicU32::new(0);
            let (seen, limit) = if stage == vk::ShaderStageFlags::COMPUTE {
                (&COMPUTE_SEEN, 8)
            } else {
                (&GRAPHICS_SEEN, 32)
            };
            if seen.fetch_add(1, Ordering::Relaxed) < limit {
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
        storage_bases.push(resource.base48());
        storage_sizes.push(size as usize);

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
    let mut sampled_key_views: [Vec<usize>; SAMPLED_KEYS] = std::array::from_fn(|_| Vec::new());
    // Mixed-key STORAGE routing, the exact same contract for RW T#s: one
    // `%textures2D_L<key>` array per present storage (Dim, format) key
    // (measured: ASTRO.BOT compute writes a 3D Rgba16f volume next to 2D
    // Rgba16f targets). Each storage T#'s seeded index is its position
    // WITHIN its own key's array, and `storage_key_images[ord]` records
    // which `storage_images` entry fills each slot.
    let mut storage_key_count = [0u64; STORAGE_KEYS];
    let mut storage_key_images: [Vec<usize>; STORAGE_KEYS] = std::array::from_fn(|_| Vec::new());
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
             {texture_byte_cap} B per-stage cap (RAEEN_MAX_STAGE_TEXTURE_MIB) — refusing the \
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
            let upload = if let Some(cached) = compute_image_snapshots
                .as_deref()
                .and_then(|snapshots| snapshots.get(&desc.texture.fields))
            {
                cached.clone()
            } else {
                let upload = read_storage_image(&desc.texture)?;
                if let Some(snapshots) = compute_image_snapshots.as_deref_mut() {
                    snapshots.insert(desc.texture.fields, upload.clone());
                }
                upload
            };
            if crate::diagnostics::gpu_env().trace_draws {
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
            let ord = storage_key_ordinal(&desc.texture);
            let image_index = storage_images.len();
            // Seed the RW T# index WITHIN its key's array (homogeneous
            // shaders have one key, so this counts 0,1,2… exactly as the
            // old `storage_images.len()` did).
            rewritten.update_address38(storage_key_count[ord]);
            storage_key_count[ord] += 1;
            storage_key_images[ord].push(image_index);
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
            if can_replace_with_render_target_pixels(&decoded)
                && let Some((width, height, px)) =
                    render_target_pixels(desc.texture.base40(), decoded.width, decoded.height)
            {
                decoded.width = width;
                decoded.height = height;
                decoded.pixels = px;
            }
            if crate::diagnostics::gpu_env().trace_draws {
                use std::sync::atomic::{AtomicU32, Ordering};
                static SEEN: AtomicU32 = AtomicU32::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                    tracing::warn!(
                        stage = ?stage,
                        index,
                        base = format_args!("{:#x}", desc.texture.base40()),
                        raw = format_args!("{:08x?}", desc.texture.fields),
                        width = decoded.width,
                        height = decoded.height,
                        vk_format = ?decoded.format,
                        direct_target = ?decoded.render_target,
                        decoded_bytes = decoded.pixels.len(),
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
    // Same split for the storage arrays: present keys in canonical ordinal
    // order, consecutive bindings starting at `binding_storage_index` —
    // matching the SPIR-V generator and `shader_calc_binding_indices`
    // exactly. A homogeneous shader leaves `storage_groups` empty
    // (single-array path).
    let storage_present_keys: Vec<usize> = (0..STORAGE_KEYS)
        .filter(|&o| !storage_key_images[o].is_empty())
        .collect();
    let storage_groups: Vec<SampledGroup> = if storage_present_keys.len() > 1 {
        storage_present_keys
            .iter()
            .enumerate()
            .map(|(pos, &ord)| SampledGroup {
                binding: bind.textures2d.binding_storage_index as u32 + pos as u32,
                view_indices: std::mem::take(&mut storage_key_images[ord]),
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

    // S#s: preserve each axis' address mode and the independent min/mag/mip
    // filters. The rewritten descriptor carries only the sampler-array index
    // in dword 0; Vulkan receives the decoded state out-of-band.
    let mut samplers = Vec::with_capacity(sampler_num);
    for (index, sampler) in bind.samplers.samplers[..sampler_num].iter().enumerate() {
        samplers.push(sampler_state(sampler));
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
        push_uniform_binding: shader_push_constant_spill_binding(bind),
        storage_buffers: (storage_num != 0).then_some(StorageBufferBinding {
            binding: bind.storage_buffers.binding_index as u32,
            buffers: storage_bytes,
            guest_bases: storage_bases,
            guest_sizes: storage_sizes,
            writable: bind.storage_buffers.usages[..storage_num]
                .iter()
                .map(|usage| {
                    *usage == kyty_graphics::shader::resources::ShaderStorageUsage::ReadWrite
                })
                .collect(),
        }),
        // Present when EITHER array is non-empty: a shader legitimately
        // binds textures without samplers (texel fetch) or samplers without
        // sampled textures — the Vulkan layer creates each descriptor array
        // independently, exactly as the SPIR-V declared them.
        textures: (!textures.is_empty() || !samplers.is_empty()).then_some(TextureBinding {
            sampled_binding: bind.textures2d.binding_sampled_index as u32,
            sampler_binding: bind.samplers.binding_index as u32,
            textures,
            samplers,
            sampled_groups,
        }),
        storage_images: (!storage_images.is_empty()).then_some(StorageImageBinding {
            binding: bind.textures2d.binding_storage_index as u32,
            images: storage_images,
            groups: storage_groups,
        }),
        gds_binding: (gds_num != 0).then_some(bind.gds_pointers.binding_index as u32),
        // The raw EUD-window snapshot (SharpEmu port): read at dispatch time
        // from the captured EUD base pointer; unreadable degrades to zeros.
        eud_raw: bind.eud_raw.used.then(|| prepare_eud_raw_binding(bind)),
    })
}

/// One WARN per distinct `CB_COLOR_CONTROL.MODE` value for the whole process
/// — the special-pass skip stays visible without per-draw spam.
fn warn_once_per_mode(mode: u8, message: &str) {
    use std::sync::Mutex;
    static SEEN: Mutex<[bool; 8]> = Mutex::new([false; 8]);
    let first = SEEN
        .lock()
        .map(|mut seen| {
            let slot = &mut seen[usize::from(mode.min(7))];
            !std::mem::replace(slot, true)
        })
        .unwrap_or(false);
    if first {
        tracing::warn!(mode, "{message}");
    }
}

/// Report, once per distinct set, which `CB_COLOR` slots a draw has bound.
/// Purely diagnostic now that MRT slots 1–7 are attached for real
/// (`mrt_attachments_from_regs`); the once-per-set line remains so a title's
/// MRT usage stays visible in the logs.
fn note_active_color_slots(ctx: &Context) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u8>>> = Mutex::new(None);

    let mut mask = 0u8;
    for (slot, rt) in ctx.render_targets.iter().enumerate() {
        if rt.base.addr != 0 {
            mask |= 1 << slot;
        }
    }
    // Slot 0 alone is the ordinary case and says nothing; only report a draw
    // that binds anything above it.
    if mask & !1 == 0 {
        return;
    }
    let first = SEEN
        .lock()
        .map(|mut set| set.get_or_insert_with(HashSet::new).insert(mask))
        .unwrap_or(false);
    if first {
        let slots: Vec<usize> = (0..8).filter(|s| mask & (1 << s) != 0).collect();
        tracing::info!(
            slot_mask = format_args!("{mask:#010b}"),
            ?slots,
            target_mask = format_args!("{:#x}", ctx.render_target_mask),
            "draw binds MULTIPLE colour render targets (attached as Vulkan MRT)"
        );
    }
}

/// Build the extra-attachment list (MRT slots 1–7) from decoded register
/// state. A slot joins when it has a base address AND its `CB_TARGET_MASK`
/// nibble writes anything; a slot the pipeline cannot honour (extent mismatch
/// with slot 0, unmapped format, untranslatable blend) is dropped with a
/// rate-limited WARN naming the reason — a localized loss, never a silent
/// one and never a failed draw.
fn mrt_attachments_from_regs(ctx: &Context) -> Vec<crate::vulkan::MrtAttachment> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static WARNED: AtomicU32 = AtomicU32::new(0);
    let warn_slot = |slot: usize, reason: &str| {
        if WARNED.fetch_add(1, Ordering::Relaxed) < 16 {
            tracing::warn!(slot, reason, "MRT attachment dropped");
        }
    };

    let rt0 = &ctx.render_targets[0];
    let mut extras = Vec::new();
    for slot in 1..ctx.render_targets.len() {
        let rt = &ctx.render_targets[slot];
        if rt.base.addr == 0 {
            continue;
        }
        let nibble = (ctx.render_target_mask >> (slot * 4)) & 0xF;
        if nibble == 0 {
            // Bound but fully write-masked: attaching it would only risk an
            // undefined-content write; hardware writes nothing either.
            continue;
        }
        if (rt.attrib2.width, rt.attrib2.height) != (rt0.attrib2.width, rt0.attrib2.height) {
            warn_slot(slot, "extent differs from slot 0");
            continue;
        }
        let format =
            match vulkan_format(rt.info.format, rt.info.channel_type, rt.info.channel_order) {
                Ok(format) => format,
                Err(e) => {
                    warn_slot(slot, &e.to_string());
                    continue;
                }
            };
        let blend = match blend_state_for_slot(ctx, slot) {
            Ok(blend) => blend,
            Err(e) => {
                warn_slot(slot, &e.to_string());
                continue;
            }
        };
        extras.push(crate::vulkan::MrtAttachment {
            slot: slot as u8,
            format,
            write_mask: vulkan_color_write_mask(nibble),
            blend,
            target_base: rt.base.addr,
            initial: None,
        });
    }
    extras
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
    note_active_color_slots(ctx);
    let color_output = !color_output_disabled(ctx);
    let depth = depth_state_from_regs(ctx)?;
    if !color_output && depth.is_none() {
        return Err(err(
            "draw has neither colour nor depth/stencil output enabled",
        ));
    }

    let (width, height, format, color_write_mask) = if color_output {
        if rt.base.addr == 0 {
            return Err(err(
                "no bound render target: CB_COLOR0_BASE is 0 (NoColorOutput)",
            ));
        }
        // The PS5 colour extent lives in ATTRIB2 and stores width/height minus
        // one.
        let width = rt.attrib2.width + 1;
        let height = rt.attrib2.height + 1;
        if rt.attrib2.width == 0 || rt.attrib2.height == 0 {
            return Err(err(format!(
                "CB_COLOR0_ATTRIB2 gives a degenerate extent {width}x{height} — \
                 the render target extent was never programmed"
            )));
        }
        (
            width,
            height,
            vulkan_format(rt.info.format, rt.info.channel_type, rt.info.channel_order)?,
            vulkan_color_write_mask(ctx.render_target_mask),
        )
    } else {
        // A depth-only prepass/clear has no meaningful CB_COLOR0 state. Size
        // the render area from DB_DEPTH_SIZE_XY, which stores max X/Y just like
        // ColorAttrib2. `format` is unused because the Vulkan pipeline declares
        // zero colour attachments when `color_output` is false.
        let size = ctx.depth_render_target.size;
        let width = u32::from(size.x_max) + 1;
        let height = u32::from(size.y_max) + 1;
        if size.x_max == 0 || size.y_max == 0 {
            return Err(err(format!(
                "depth-only draw has degenerate DB_DEPTH_SIZE_XY {width}x{height}"
            )));
        }
        (
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ColorComponentFlags::empty(),
        )
    };

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
    // Diagnostic (RAEEN_NO_CULL=1): disable culling entirely. Measured on
    // Minecraft: its draws run cull=FRONT face=CLOCKWISE, and under our
    // Y-flipped viewport the measured quad rasterizes clockwise — a FRONT
    // face — so every primitive is culled. The full title VS+PS replayed
    // in-tree with cull NONE covers 4096/4096 (tests/coverage_bisect.rs), so
    // culling is the LAST field separating the in-tree render from the
    // title's black frame. This switch is the yes/no for that mechanism.
    if crate::diagnostics::gpu_env().no_cull {
        cull_mode = vk::CullModeFlags::NONE;
    }
    // PA_SU_SC_MODE_CNTL.FACE: 0 = counter-clockwise is the front face,
    // 1 = clockwise is. Kyty, SharpEmu, and shadPS4 all map this directly to
    // Vulkan; negative viewport height is already part of Vulkan's
    // framebuffer-space winding calculation and must not be applied twice.
    let front_face = if ctx.mode_control.face {
        vk::FrontFace::CLOCKWISE
    } else {
        vk::FrontFace::COUNTER_CLOCKWISE
    };

    let blend = if color_output {
        blend_state_from_regs(ctx)?
    } else {
        BlendState::default()
    };

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
        color_output,
        depth,
        // MRT slots 1-7, decoded per-slot; the caller (draw_common) seeds
        // each extra's `initial` from the framebuffer map.
        mrt: if color_output {
            mrt_attachments_from_regs(ctx)
        } else {
            Vec::new()
        },
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
    framebuffers: &'a mut HashMap<u64, Arc<RenderedImage>>,
    pub last: Option<RenderedImage>,
    /// `CB_COLOR0_BASE` of the last draw's render target. With deferred
    /// readback (stage B) `last` is only populated by immediate-fallback
    /// draws; the session resolves the presented frame by looking this base
    /// up in the framebuffer map AFTER the flush lands the batch's pixels.
    pub last_target: Option<u64>,
    pub draws: u64,
    /// Eliminate-fast-clear passes applied as direct clears (real feature
    /// work, not skips — see `eliminate_fast_clear`).
    pub fast_clears: u64,
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
    /// Complete guest SSBO snapshots captured once per PM4 submission. Later
    /// dispatches share the same allocation and sparse shader writebacks are
    /// folded into it, matching GPU-visible resource lifetime without copying
    /// multi-megabyte buffers for every packet.
    compute_storage_snapshots: ComputeStorageSnapshots,
    /// Storage-image counterparts of `compute_storage_snapshots`. The decoded
    /// linear pixels are shared by every matching descriptor in this PM4
    /// submission; the Vulkan cache uses Arc identity to preserve GPU-newer
    /// contents between ordered dispatches instead of re-uploading the seed.
    compute_image_snapshots: ComputeImageSnapshots,
    /// Texture source hashes already sampled in this PM4 submission. Compute
    /// and packet memory-write boundaries invalidate it before a later draw.
    texture_sample_hashes: GuestSampleHashMemo,
    /// Completed VS+PS resolutions for exact register states in this PM4
    /// submission. This avoids repeating stage analysis for every draw while
    /// retaining write-boundary correctness.
    resolved_shaders: ResolvedShaderMemo,
    /// At least one storage-only compute packet joined the deferred queue.
    /// The session uses this to fence/write back once at the end of this PM4
    /// submission, before transient guest allocations may be released.
    pub queued_compute: bool,
}

impl<'a> OffscreenDrawSink<'a> {
    #[must_use]
    pub fn new(
        dev: &'a VulkanDevice,
        cache: &'a mut ShaderTranslateCache,
        framebuffers: &'a mut HashMap<u64, Arc<RenderedImage>>,
    ) -> Self {
        Self {
            dev,
            cache,
            framebuffers,
            last: None,
            last_target: None,
            draws: 0,
            fast_clears: 0,
            shader_skips: 0,
            draw_skips: 0,
            dispatch_skips: 0,
            dispatches: 0,
            last_draw_skip_reason: None,
            last_dispatch_skip_reason: None,
            queue_is_compute: false,
            current_compute: None,
            compute_storage_snapshots: HashMap::new(),
            compute_image_snapshots: HashMap::new(),
            texture_sample_hashes: GuestSampleHashMemo::default(),
            resolved_shaders: ResolvedShaderMemo::default(),
            queued_compute: false,
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

    /// Fence and publish only when deferred compute produced guest-addressed
    /// data that the next graphics draw can read.
    ///
    /// The batch stays open across compute-only submissions for throughput,
    /// but index and vertex fetch still happen on the CPU from guest memory.
    /// Without this dependency boundary Minecraft's second-screen UI saw
    /// all-zero generated indices/vertices and collapsed every triangle.
    fn flush_compute_for_graphics_read(&mut self) -> Result<(), DrawError> {
        let pending = {
            let caches = self.dev.draw_caches();
            caches.has_pending_compute_writebacks()
        };
        if !pending {
            return Ok(());
        }
        self.texture_sample_hashes.clear();

        // An empty render-target filter fences the ordered batch and publishes
        // compute SSBO/UAV outputs without paying to read unrelated colour
        // targets back. Storage-image outputs are returned and become valid
        // framebuffer/texture sources for this draw.
        let flushed = crate::vulkan::offscreen::flush_deferred_draws_filtered(self.dev, Some(&[]))
            .map_err(|e| err(format!("compute-to-graphics resource barrier failed: {e}")))?;
        for (base, image) in flushed {
            self.framebuffers.insert(base, Arc::new(image));
        }
        Ok(())
    }

    fn resolve_shaders_cached(
        &mut self,
        ctx: &Context,
        sh: &Shader,
    ) -> Result<ResolvedShaders, DrawError> {
        let key = ResolvedShaderKey::new(ctx, sh);
        if let Some(shaders) = self.resolved_shaders.get(key) {
            return Ok(shaders);
        }
        let shaders = resolve_shaders(self.cache, ctx, sh)?;
        self.resolved_shaders.insert(key, shaders.clone());
        Ok(shaders)
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
        let _draw_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_DRAWCOMMON_NS,
        );
        // Special CB passes (shadPS4 `FilterDraw`): a draw issued with
        // CB_COLOR_CONTROL.MODE above Normal is not ordinary geometry.
        // Eliminate-fast-clear becomes a real direct clear of the bound
        // target; the resolve/decompress passes are named, counted skips —
        // never silently rasterized as if they were scene draws.
        match ctx.color_control.mode {
            cb_mode::ELIMINATE_FAST_CLEAR => return self.eliminate_fast_clear(ctx),
            cb_mode::RESOLVE => {
                warn_once_per_mode(ctx.color_control.mode, "CB resolve pass skipped");
                return Ok(());
            }
            cb_mode::FMASK_DECOMPRESS | cb_mode::DCC_DECOMPRESS => {
                warn_once_per_mode(
                    ctx.color_control.mode,
                    "CB FMASK/DCC decompress pass skipped (compression metadata not emulated)",
                );
                return Ok(());
            }
            _ => {}
        }
        // A zero colour mask with depth/stencil disabled is a state-carrying
        // no-op. With depth/stencil enabled it is a real z-prepass/clear and
        // must reach the now-wired depth backend; dropping it leaves a stale
        // persistent depth surface that can reject later colour geometry.
        if color_output_disabled(ctx)
            && !ctx.depth_control.z_enable
            && !ctx.depth_control.stencil_enable
        {
            debug!("draw consumed with neither colour nor depth/stencil output");
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
        let resolve_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_RESOLVE_NS,
        );
        let resolved = self.resolve_shaders_cached(ctx, sh);
        let shaders = if sh.vs.vs_embedded && sh.ps.ps_embedded {
            // The embedded pair is the Phase 1 / M2 invariant: a failure here
            // is a broken fixture and must abort loudly.
            resolved?
        } else {
            match resolved {
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
        drop(resolve_timer);

        let setup_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_SETUP_NS,
        );
        let mut state = draw_state_from_regs(ctx, ucfg, count, &shaders.vs, &shaders.ps)?;
        // Internal-resolution scaling (Settings ▸ Video ▸ Resolution Scale).
        // Supersamples the whole draw (target + viewport + scissor together);
        // a factor of 1.0 — the default — is an exact no-op.
        state.scale_resolution(crate::agc_exec::AgcGpuSession::runtime_config().resolution_scale);
        let vertex_records = required_vertex_records(index, state.vertex_count)?;
        let (vertex_buffers, vertex_attributes) =
            prepare_vertex_inputs_limited(&shaders.vs_info, Some(vertex_records))?;
        state.vertex_buffers = vertex_buffers;
        state.vertex_attributes = vertex_attributes;
        drop(setup_timer);
        // Coverage probe (RAEEN_TRACE_DRAWS). Vertex data, attribute bindings,
        // gl_Position and draw state are all confirmed correct yet coverage is
        // ZERO. An all-zero index buffer collapses every primitive to a single
        // vertex — degenerate triangles cover no pixel — and would look exactly
        // like this. Report whether the draw is indexed and what the indices are.
        if crate::diagnostics::gpu_env().trace_draws {
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

        let rt_base = if state.color_output {
            ctx.render_targets[0].base.addr
        } else {
            0
        };
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
        let samples_own_target = state.color_output
            && stage_binds.iter().any(|(bind, _)| {
                let n = usize::try_from(bind.textures2d.textures_num).unwrap_or(0);
                bind.textures2d.desc[..n.min(bind.textures2d.desc.len())]
                    .iter()
                    .any(|d| {
                        d.usage != ShaderTextureUsage::ReadWrite && d.texture.base40() == rt_base
                    })
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
        let census_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_CENSUS_NS,
        );
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
        drop(census_timer);
        // NGG exposes its vertex program through the ES register block; the
        // legacy VS block remains zero. Report the effective shader address so
        // a layered-texture trace can be matched to the correct SPIR-V dump.
        let vs_addr = if sh.vs.vs_regs.data_addr != 0 {
            sh.vs.vs_regs.data_addr
        } else {
            sh.vs.es_regs.data_addr
        };
        let ps_addr = sh.ps.ps_regs.data_addr;
        let trace_textures = crate::diagnostics::gpu_env().trace_textures;
        let vertex_head = if trace_textures {
            state
                .vertex_buffers
                .first()
                .map(|buffer| buffer.bytes[..buffer.bytes.len().min(96)].to_vec())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let index_head = if trace_textures {
            index
                .map(|(bytes, _)| bytes[..bytes.len().min(96)].to_vec())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let scope = SamplingScope {
            map: std::ptr::from_ref(self.framebuffers),
            live,
            self_base: rt_base,
            resolution_scale: crate::agc_exec::AgcGpuSession::runtime_config().resolution_scale,
            vs_addr,
            ps_addr,
            primitive: ucfg.prim_type,
            vertex_count: state.vertex_count,
            indexed: index.is_some(),
            first_attribute: state.vertex_attributes.first().copied(),
            first_stride: state.vertex_buffers.first().map(|buffer| buffer.stride),
            index_type: index.map(|(_, index_type)| index_type),
            vertex_head,
            index_head,
            cached_textures,
            sample_hash_memo: std::ptr::from_ref(&self.texture_sample_hashes),
        };
        let bind_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_BIND_NS,
        );
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
        drop(bind_timer);

        // One-shot forensic: what does a real Minecraft draw actually bind?
        // A force-clear experiment alone cannot prove zero fragment coverage:
        // a fragment shader or blend state may legitimately write that same
        // colour. Treat this as a state census, not a coverage verdict.
        // Gated to the first few draws (RAEEN_TRACE_DRAWS) so it never floods.
        if crate::diagnostics::gpu_env().trace_draws {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEEN: AtomicU32 = AtomicU32::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 12 {
                let ps = &shaders.ps_info.bind;
                let vs = &shaders.vs_info.bind;
                tracing::warn!(
                    prim = ucfg.prim_type,
                    verts = state.vertex_count,
                    target_base = format_args!("{rt_base:#x}"),
                    target_extent = format_args!("{}x{}", state.width, state.height),
                    target_format = ?state.format,
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

        let trace_minecraft_model = crate::diagnostics::gpu_env().trace_model
            && matches!(shaders.vs.len(), 16_848 | 16_852)
            && matches!(shaders.ps.len(), 5_184 | 5_187);
        if trace_selected_shader(vs_addr, ps_addr)
            || trace_shader_words_selected(shaders.vs.len(), shaders.ps.len())
            || trace_minecraft_model
        {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SELECTED_SEEN: AtomicU32 = AtomicU32::new(0);
            let selected = SELECTED_SEEN.fetch_add(1, Ordering::Relaxed);
            if selected < 32 {
                let vertex_heads: Vec<_> = state
                    .vertex_buffers
                    .iter()
                    .map(|buffer| {
                        (
                            buffer.stride,
                            buffer.bytes.len(),
                            buffer.bytes[..buffer.bytes.len().min(64)].to_vec(),
                        )
                    })
                    .collect();
                let stage_resources: Vec<_> = state
                    .stage_bindings
                    .iter()
                    .map(|binding| {
                        let storage_heads: Vec<_> = binding
                            .storage_buffers
                            .iter()
                            .flat_map(|storage| storage.buffers.iter())
                            .map(|bytes| (bytes.len(), bytes[..bytes.len().min(96)].to_vec()))
                            .collect();
                        (
                            binding.stage,
                            binding.descriptor_set_slot,
                            binding.push_constant_offset,
                            binding.push_constants.len(),
                            binding.push_constants[..binding.push_constants.len().min(96)].to_vec(),
                            storage_heads,
                            binding
                                .textures
                                .as_ref()
                                .map_or(0, |textures| textures.textures.len()),
                        )
                    })
                    .collect();
                let texture_summaries: Vec<_> = state
                    .stage_bindings
                    .iter()
                    .filter_map(|binding| binding.textures.as_ref())
                    .flat_map(|binding| binding.textures.iter())
                    .map(|texture| {
                        let (alpha_nonzero, alpha_opaque) =
                            if texture.format == vk::Format::R8G8B8A8_UNORM {
                                (
                                    texture
                                        .pixels
                                        .chunks_exact(4)
                                        .filter(|pixel| pixel[3] != 0)
                                        .count(),
                                    texture
                                        .pixels
                                        .chunks_exact(4)
                                        .filter(|pixel| pixel[3] == u8::MAX)
                                        .count(),
                                )
                            } else {
                                (0, 0)
                            };
                        (
                            texture.guest_base,
                            texture.width,
                            texture.height,
                            texture.format,
                            texture.layers,
                            texture.pixels.len(),
                            alpha_nonzero,
                            alpha_opaque,
                            texture.cached,
                            texture.pixels[..texture.pixels.len().min(96)].to_vec(),
                        )
                    })
                    .collect();
                let sampler_summaries: Vec<_> =
                    [("vs", &shaders.vs_info.bind), ("ps", &shaders.ps_info.bind)]
                        .into_iter()
                        .flat_map(|(stage, bind)| {
                            let count = usize::try_from(bind.samplers.samplers_num)
                                .unwrap_or_default()
                                .min(bind.samplers.samplers.len());
                            bind.samplers.samplers[..count].iter().enumerate().map(
                                move |(index, sampler)| {
                                    (
                                        stage,
                                        index,
                                        sampler.fields,
                                        sampler.clamp_x(),
                                        sampler.clamp_y(),
                                        sampler.clamp_z(),
                                        sampler.force_unorm_coords(),
                                        sampler.xy_mag_filter(),
                                        sampler.xy_min_filter(),
                                        sampler.mip_filter(),
                                    )
                                },
                            )
                        })
                        .collect();
                let uv_alpha_samples: Vec<_> = state
                    .vertex_attributes
                    .iter()
                    .find(|attribute| {
                        attribute.location == 4 && attribute.format == vk::Format::R16G16_UNORM
                    })
                    .and_then(|attribute| {
                        let vertex = state.vertex_buffers.get(attribute.binding as usize)?;
                        let texture = state
                            .stage_bindings
                            .iter()
                            .filter_map(|binding| binding.textures.as_ref())
                            .flat_map(|binding| binding.textures.iter())
                            .find(|texture| {
                                texture.format == vk::Format::R8G8B8A8_UNORM
                                    && !texture.pixels.is_empty()
                            })?;
                        let stride = vertex.stride as usize;
                        let offset = attribute.offset as usize;
                        (stride >= offset + 4 && texture.width != 0 && texture.height != 0).then(
                            || {
                                vertex
                                    .bytes
                                    .chunks(stride)
                                    .take(64)
                                    .filter_map(|record| {
                                        let uv = record.get(offset..offset + 4)?;
                                        let u = u16::from_le_bytes([uv[0], uv[1]]);
                                        let v = u16::from_le_bytes([uv[2], uv[3]]);
                                        let x = (u64::from(u) * u64::from(texture.width - 1)
                                            / u64::from(u16::MAX))
                                            as u32;
                                        let y = (u64::from(v) * u64::from(texture.height - 1)
                                            / u64::from(u16::MAX))
                                            as u32;
                                        let at = (u64::from(y) * u64::from(texture.width)
                                            + u64::from(x))
                                            as usize
                                            * 4;
                                        texture
                                            .pixels
                                            .get(at + 3)
                                            .copied()
                                            .map(|alpha| (u, v, x, y, alpha))
                                    })
                                    .collect()
                            },
                        )
                    })
                    .unwrap_or_default();
                let depth_summary = state.depth.as_ref().map(|depth| {
                    (
                        depth.target_base,
                        depth.format,
                        depth.test_enable,
                        depth.write_enable,
                        depth.compare_op,
                        depth.stencil_test_enable,
                        depth.stencil_front,
                        depth.stencil_back,
                        depth.clear_depth,
                        depth.clear_stencil,
                        depth.clear_stencil_value,
                        depth.viewport_depth,
                    )
                });
                let index_head = index.map(|(bytes, index_type)| {
                    (
                        index_type,
                        bytes.len(),
                        bytes[..bytes.len().min(32)].to_vec(),
                    )
                });
                tracing::warn!(
                    selected,
                    vs_addr = format_args!("{vs_addr:#x}"),
                    es_addr = format_args!("{:#x}", sh.vs.es_regs.data_addr),
                    gs_addr = format_args!("{:#x}", sh.vs.gs_regs.data_addr),
                    ps_addr = format_args!("{ps_addr:#x}"),
                    prim = ucfg.prim_type,
                    count = state.vertex_count,
                    indexed = index.is_some(),
                    index_head = ?index_head,
                    target_base = format_args!("{rt_base:#x}"),
                    extent = format_args!("{}x{}", state.width, state.height),
                    viewport = ?state.viewport,
                    scissor = ?state.scissor,
                    topology = ?state.topology,
                    cull = ?state.cull_mode,
                    face = ?state.front_face,
                    write_mask = ?state.color_write_mask,
                    blend = ?state.blend,
                    depth = ?depth_summary,
                    attributes = ?state.vertex_attributes,
                    vertex_heads = ?vertex_heads,
                    stage_resources = ?stage_resources,
                    texture_summaries = ?texture_summaries,
                    sampler_summaries = ?sampler_summaries,
                    uv_alpha_samples = ?uv_alpha_samples,
                    "TRACE_SHADER_ADDR: selected draw state"
                );
            }
        }

        // Compose into the guest render target: seed with its prior pixels
        // (taken from the framebuffer map) so this draw adds to the frame
        // instead of starting over on a cleared attachment.
        let backend_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_BACKEND_NS,
        );
        let prior = state
            .color_output
            .then(|| self.framebuffers.remove(&rt_base))
            .flatten()
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
        state.target_base = state.color_output.then_some(rt_base);
        // MRT extras: PM4-ordered flush first (a pending deferred draw into
        // the primary or any extra base must land before this immediate
        // draw), then seed each extra's LOAD from the framebuffer map when
        // the prior readback matches this draw's extent and byte size.
        if !state.mrt.is_empty() {
            let needs_flush = {
                let caches = self.dev.draw_caches();
                caches.base_is_batch_dirty(rt_base)
                    || state
                        .mrt
                        .iter()
                        .any(|extra| caches.base_is_batch_dirty(extra.target_base))
            };
            if needs_flush {
                self.flush_deferred_into_framebuffers()?;
            }
            for extra in &mut state.mrt {
                let expected = crate::vulkan::offscreen::readback_bpp(extra.format)
                    .ok()
                    .map(|bpp| state.width as usize * state.height as usize * bpp as usize);
                if let Some(prior) = self
                    .framebuffers
                    .get(&extra.target_base)
                    .filter(|p| p.width == state.width && p.height == state.height)
                    .filter(|p| Some(p.pixels.len()) == expected)
                {
                    extra.initial = Some(prior.pixels.clone());
                }
            }
        }
        // Stage B: the draw is submitted with its readback DEFERRED —
        // `Ok(None)` means the pixels land in the framebuffer map at the next
        // flush (end of submission, presentation, or a feedback fallback).
        // `Ok(Some(image))` is the immediate-fallback path (readback now),
        // preserving the old per-draw behaviour. MRT draws are immediate by
        // construction and additionally land every extra attachment's
        // readback in the framebuffer map by its own guest base.
        let color_output = state.color_output;
        let has_mrt = !state.mrt.is_empty();
        let backend_error = |e: raeen_core::error::GpuError, state: &DrawState| {
            let depth = state.depth.as_ref().map(|d| {
                (
                    d.target_base,
                    d.format,
                    d.test_enable,
                    d.write_enable,
                    d.stencil_test_enable,
                )
            });
            err(format!(
                "offscreen draw failed: {e}; vs={vs_addr:#x} ps={ps_addr:#x} \
                 target={rt_base:#x} {}x{} format={:?} prim={} vertices={} indexed={} \
                 depth={depth:?} mrt={} stage_bindings={}",
                state.width,
                state.height,
                state.format,
                ucfg.prim_type,
                state.vertex_count,
                index.is_some(),
                state.mrt.len(),
                state.stage_bindings.len()
            ))
        };
        let immediate = if has_mrt {
            let output = crate::vulkan::offscreen::render_draw(self.dev, &state)
                .map_err(|e| backend_error(e, &state))?;
            for (base, image) in output.mrt_colors {
                self.framebuffers.insert(base, Arc::new(image));
            }
            output.color
        } else {
            crate::vulkan::offscreen::render_draw_deferred(self.dev, &state)
                .map_err(|e| backend_error(e, &state))?
        };
        drop(backend_timer);
        drop(state);
        match immediate {
            Some(image) => {
                self.framebuffers.insert(rt_base, Arc::new(image.clone()));
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
        if color_output {
            self.last_target = Some(rt_base);
        }
        self.draws += 1;
        Ok(())
    }

    /// Eliminate-fast-clear as a direct clear (shadPS4
    /// `Rasterizer::EliminateFastClear`). The FCE draw itself is consumed;
    /// its effect — the CMASK-cleared tiles materializing as the packed
    /// `CLEAR_WORD` colour — is applied to the WHOLE target, because Raeen
    /// keeps no CMASK to know which tiles were fast-cleared. Titles issue FCE
    /// right after a full-surface fast clear, where whole-target == exact;
    /// a partial fast clear would be over-cleared, which the once-log names.
    fn eliminate_fast_clear(&mut self, ctx: &Context) -> Result<(), DrawError> {
        let rt = &ctx.render_targets[0];
        let scale = crate::agc_exec::AgcGpuSession::runtime_config().resolution_scale;
        let image = match fast_clear_image(rt, scale) {
            Ok(Some(image)) => image,
            // No bound target / fast clear not armed: an FCE with nothing to
            // eliminate is a quiet no-op (shadPS4 returns the same way).
            Ok(None) => return Ok(()),
            Err(e) => {
                // Named degradation: the clear colour cannot be materialized
                // (unmapped format). Skipping is visible, not silent.
                self.draw_skips += 1;
                self.last_draw_skip_reason = Some(e.to_string());
                debug!(reason = %e, "eliminate-fast-clear skipped");
                return Ok(());
            }
        };
        let base = rt.base.addr;
        // PM4 order: deferred draws recorded before this FCE must land their
        // readbacks first, or the flush would overwrite the clear.
        if self.dev.draw_caches().base_is_batch_dirty(base) {
            self.flush_deferred_into_framebuffers()?;
        }
        // The persistent GPU image (if any) now holds stale pre-clear pixels:
        // evict every image at this base so the next draw seeds from the
        // cleared CPU pixels. The sentinel key matches nothing, so ALL
        // extents/formats at this base go.
        {
            let sentinel = crate::vulkan::cache::TargetKey {
                base,
                width: 0,
                height: 0,
                format: 0,
            };
            let mut caches = self.dev.draw_caches();
            caches.evict_targets_for_base(self.dev, base, &sentinel);
        }
        static NOTED: std::sync::Once = std::sync::Once::new();
        NOTED.call_once(|| {
            tracing::info!(
                target_base = format_args!("{base:#x}"),
                "eliminate-fast-clear implemented as a whole-target direct clear \
                 (no CMASK: partial fast clears would be over-cleared)"
            );
        });
        debug!(
            target_base = format_args!("{base:#x}"),
            width = image.width,
            height = image.height,
            clear_word0 = format_args!("{:#010x}", rt.clear_word0.word0),
            clear_word1 = format_args!("{:#010x}", rt.clear_word1.word1),
            "eliminate-fast-clear: direct clear"
        );
        self.framebuffers.insert(base, Arc::new(image));
        self.last_target = Some(base);
        self.fast_clears += 1;
        Ok(())
    }

    /// Flush the pending deferred-draw batch and land every readback in the
    /// framebuffer map (the feedback-loop fallback path).
    fn flush_deferred_into_framebuffers(&mut self) -> Result<(), DrawError> {
        let flushed = crate::vulkan::offscreen::flush_deferred_draws(self.dev)
            .map_err(|e| err(format!("deferred-draw flush failed: {e}")))?;
        for (base, image) in flushed {
            self.framebuffers.insert(base, Arc::new(image));
        }
        Ok(())
    }
}

impl DrawSink for OffscreenDrawSink<'_> {
    fn guest_memory_write_boundary(&mut self) {
        self.texture_sample_hashes.clear();
        self.resolved_shaders.clear();
    }

    fn draw_index_auto(
        &mut self,
        ctx: &Context,
        ucfg: &UserConfig,
        sh: &Shader,
        index_count: u32,
        _flags: u32,
    ) -> Result<(), DrawError> {
        self.flush_compute_for_graphics_read()?;
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
        // This must precede `fetch_index_buffer`: generated indices live in
        // guest memory only after the deferred compute writeback is published.
        self.flush_compute_for_graphics_read()?;
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
        let _dispatch_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_DISPATCH_NS,
        );
        crate::vulkan::offscreen::DRAW_STAGE_DISPATCH_N
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        let translate_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_CS_TRANSLATE_NS,
        );
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
        drop(translate_timer);
        // Forensic kill switch: translate (and therefore dump, when
        // RAEEN_DUMP_SHADERS is set) the named compute program, but never
        // submit it. Keeping the cut after translation makes a confirmed
        // device-loss shader inspectable without risking another GPU reset.
        // Translation failures already return through the named refusal above.
        if let Some(list) = crate::diagnostics::gpu_env().skip_cs.as_deref() {
            let addr = format!("{:#x}", cs.cs_regs.data_addr);
            if list
                .split(',')
                .any(|s| s.trim().eq_ignore_ascii_case(&addr))
            {
                self.dispatch_skips += 1;
                self.last_dispatch_skip_reason = Some(format!("RAEEN_SKIP_CS: {addr}"));
                debug!(cs_addr = %addr, "translated compute dispatch skipped by RAEEN_SKIP_CS");
                return Ok(());
            }
        }
        if is_known_device_loss_compute(cs.cs_regs.data_addr, groups)
            && !crate::diagnostics::gpu_env().allow_known_device_loss_cs
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static QUARANTINED: AtomicU64 = AtomicU64::new(0);
            let occurrence = QUARANTINED.fetch_add(1, Ordering::Relaxed) + 1;
            let reason = format!(
                "known-device-loss compute quarantined at {:#x} (64x64x1); \
                 RAEEN_ALLOW_KNOWN_DEVICE_LOSS_CS=1 restores it for diagnostics",
                cs.cs_regs.data_addr
            );
            self.dispatch_skips += 1;
            self.last_dispatch_skip_reason = Some(reason.clone());
            if occurrence <= 4 || occurrence.is_power_of_two() {
                tracing::warn!(
                    occurrence,
                    cs_addr = format_args!("{:#x}", cs.cs_regs.data_addr),
                    groups = ?groups,
                    reason,
                    "compute dispatch safely refused before the known Windows driver reset"
                );
            }
            return Ok(());
        }
        let prepare_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_CS_PREPARE_NS,
        );
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
            .then(|| {
                prepare_compute_stage_binding(
                    bind,
                    &mut self.compute_storage_snapshots,
                    &mut self.compute_image_snapshots,
                )
            })
            .transpose()?;
        if crate::diagnostics::gpu_env().trace_draws && bind.textures2d.textures_num != 0 {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEEN: AtomicU32 = AtomicU32::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 24 {
                let push_dwords = prepared
                    .as_ref()
                    .map(|binding| {
                        binding
                            .push_constants
                            .chunks_exact(4)
                            .map(|field| u32::from_le_bytes(field.try_into().unwrap()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                tracing::warn!(
                    cs_addr = format_args!("{:#x}", cs.cs_regs.data_addr),
                    groups = format_args!("{}x{}x{}", groups[0], groups[1], groups[2]),
                    threads = format_args!(
                        "{}x{}x{}",
                        cs.cs_regs.num_thread_x,
                        cs.cs_regs.num_thread_y,
                        cs.cs_regs.num_thread_z
                    ),
                    push_constant_offset = bind.push_constant_offset,
                    push_constant_size = bind.push_constant_size,
                    storage_buffers = bind.storage_buffers.buffers_num,
                    textures = bind.textures2d.textures_num,
                    sampled_images = bind.textures2d.textures2d_sampled_num,
                    storage_images = bind.textures2d.textures2d_storage_num,
                    samplers = bind.samplers.samplers_num,
                    direct_sgprs = bind.direct_sgprs.sgprs_num,
                    push_dwords = ?push_dwords,
                    "TRACE_DRAWS: compute binding ABI"
                );
            }
        }
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
            .zip(&bind.storage_buffers.usages[..storage_num])
            .filter(|(_, usage)| {
                **usage == kyty_graphics::shader::resources::ShaderStorageUsage::ReadWrite
            })
            .map(|(resource, _)| {
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
        let guest_image_outputs: Vec<(u64, u32, u32, u32, u32, u8, vk::Format)> = prepared
            .as_ref()
            .and_then(|binding| binding.storage_images.as_ref())
            .map(|images| {
                images
                    .images
                    .iter()
                    .map(|img| {
                        (
                            img.guest_base,
                            img.width,
                            img.height,
                            img.depth,
                            img.layers,
                            img.tile_mode,
                            img.format,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Invalidate exactly what this dispatch can write. The old blanket
        // clear ran for every dispatch, including read-only compute, and made
        // the submission-local texture hash memo ineffective in Minecraft
        // (hundreds of dispatches interleave the same immutable texture
        // binds). Storage-buffer ranges are known exactly; storage-image
        // swizzle padding is not carried by `StorageImageUpload`, so retain
        // the conservative full texture clear for any writable image.
        let writable_buffer_ranges: Vec<(u64, u64)> = guest_outputs
            .iter()
            .filter_map(|&(base, len)| (base != 0 && len != 0).then_some((base, len as u64)))
            .collect();
        let writes_guest_memory =
            !writable_buffer_ranges.is_empty() || !guest_image_outputs.is_empty();
        if writes_guest_memory {
            if guest_image_outputs.is_empty() {
                self.resolved_shaders
                    .invalidate_ranges(&writable_buffer_ranges);
            } else {
                // Storage-image swizzle padding is not carried by
                // `StorageImageUpload`, so its complete guest write range is
                // not yet known precisely.
                self.resolved_shaders.clear();
            }
            if guest_image_outputs.is_empty() {
                self.texture_sample_hashes
                    .invalidate_ranges(&writable_buffer_ranges);
            } else {
                self.texture_sample_hashes.clear();
            }
        }
        // Forensic breadcrumb: device loss surfaces LAZILY (the next
        // vkQueueSubmit reports it), so identifying a lethal dispatch needs
        // the pre-submit identity of every dispatch in the log.
        let log_submit = |trace: bool| {
            let cs_addr = format_args!("{:#x}", cs.cs_regs.data_addr);
            let groups = format_args!("{}x{}x{}", groups[0], groups[1], groups[2]);
            let null_vsharps = guest_outputs
                .iter()
                .filter(|(addr, len)| *addr == 0 || *len == 0)
                .count();
            let gds = prepared.as_ref().is_some_and(|p| p.gds_binding.is_some());
            let textures = prepared
                .as_ref()
                .and_then(|p| p.textures.as_ref())
                .map_or(0, |t| t.textures.len());
            let samplers = prepared
                .as_ref()
                .and_then(|p| p.textures.as_ref())
                .map_or(0, |t| t.samplers.len());
            let images = prepared
                .as_ref()
                .and_then(|p| p.storage_images.as_ref())
                .map_or(0, |i| i.images.len());
            if trace {
                tracing::warn!(
                    cs_addr,
                    groups,
                    storage_num,
                    null_vsharps,
                    gds,
                    textures,
                    samplers,
                    images,
                    "TRACE_DRAWS: compute dispatch submitting"
                );
            } else {
                debug!(
                    cs_addr,
                    groups,
                    storage_num,
                    null_vsharps,
                    gds,
                    textures,
                    samplers,
                    images,
                    "compute dispatch submitting"
                );
            }
        };
        log_submit(crate::diagnostics::gpu_env().trace_draws);
        let compute_state = ComputeState {
            groups,
            spirv: &translated.spirv,
            binding: prepared.as_ref(),
        };
        let deferred = !crate::diagnostics::gpu_env().no_defer_compute
            && !compute_requires_slicing(&compute_state)
            && prepared.as_ref().is_some_and(|binding| {
                let storage_ok = binding
                    .storage_buffers
                    .as_ref()
                    .is_none_or(|storage| storage.guest_bases.iter().all(|&base| base != 0));
                let images_ok = binding
                    .storage_images
                    .as_ref()
                    .is_none_or(|images| images.images.iter().all(|image| image.guest_base != 0));
                let has_guest_resource = binding
                    .storage_buffers
                    .as_ref()
                    .is_some_and(|storage| !storage.buffers.is_empty())
                    || binding
                        .storage_images
                        .as_ref()
                        .is_some_and(|images| !images.images.is_empty());
                storage_ok && images_ok && has_guest_resource
            });
        drop(prepare_timer);
        let backend_timer = crate::vulkan::offscreen::StageTimer::start(
            &crate::vulkan::offscreen::DRAW_STAGE_CS_BACKEND_NS,
        );
        if deferred {
            dispatch_compute_deferred(self.dev, &compute_state).map_err(|error| {
                err(format!("deferred Vulkan compute dispatch failed: {error}"))
            })?;
            self.queued_compute = true;
            self.dispatches += 1;
            return Ok(());
        }
        let dispatch_at = crate::diagnostics::gpu_env()
            .time_compute
            .then(std::time::Instant::now);
        let outputs = dispatch_compute(self.dev, &compute_state)
            .map_err(|error| err(format!("Vulkan compute dispatch failed: {error}")))?;
        drop(backend_timer);
        if let Some(dispatch_at) = dispatch_at {
            let elapsed = dispatch_at.elapsed();
            if elapsed >= std::time::Duration::from_millis(10) {
                use std::sync::atomic::{AtomicU64, Ordering};
                static SLOW: AtomicU64 = AtomicU64::new(0);
                let n = SLOW.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 16 || n.is_power_of_two() {
                    tracing::warn!(
                        slow_dispatch = n,
                        cs_addr = format_args!("{:#x}", cs.cs_regs.data_addr),
                        groups = format_args!("{}x{}x{}", groups[0], groups[1], groups[2]),
                        elapsed_us = elapsed.as_micros(),
                        guest_outputs = ?guest_outputs,
                        guest_image_outputs = ?guest_image_outputs,
                        "TIME_COMPUTE: slow guest resource identity"
                    );
                }
            }
        }
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
        // The Vulkan dispatch is fence-complete and all output identities were
        // copied above. Release binding Arcs before folding sparse deltas into
        // the submission cache so `Arc::make_mut` stays allocation-free.
        drop(prepared);
        for ((addr, real_len), output) in guest_outputs.into_iter().zip(outputs.buffers) {
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
            // Keep the submission snapshot authoritative for the next
            // dispatch that binds this guest allocation. `Arc::make_mut`
            // copies only if an earlier binding still holds a live reference;
            // the synchronous dispatch returned before this point, so in the
            // steady path the snapshot is uniquely owned by the cache.
            if let Some(snapshot) = self.compute_storage_snapshots.get_mut(&(addr, output.size)) {
                let snapshot = Arc::make_mut(snapshot);
                for span in &output.dirty {
                    let end = span
                        .offset
                        .saturating_add(span.bytes.len())
                        .min(snapshot.len());
                    if span.offset < end {
                        snapshot[span.offset..end]
                            .copy_from_slice(&span.bytes[..end - span.offset]);
                    }
                }
            }
            let dirty_bytes: usize = output
                .dirty
                .iter()
                .map(|span| span.bytes.len().min(real_len.saturating_sub(span.offset)))
                .sum();
            debug!(
                addr = format_args!("{addr:#x}"),
                real_len,
                dirty_spans = output.dirty.len(),
                dirty_bytes,
                "compute sparse storage writeback"
            );
            if crate::diagnostics::gpu_env().trace_draws {
                tracing::warn!(
                    addr = format_args!("{addr:#x}"),
                    real_len,
                    dirty_spans = output.dirty.len(),
                    dirty_bytes,
                    "TRACE_DRAWS: sparse compute writeback"
                );
            }
            for span in output.dirty {
                if span.offset >= real_len {
                    continue;
                }
                // Truncate the final dirty page at the V#'s real byte length,
                // excluding Vulkan's dword-alignment pad.
                let bytes =
                    &span.bytes[..span.bytes.len().min(real_len.saturating_sub(span.offset))];
                let span_addr = addr.saturating_add(span.offset as u64);
                crate::guest_mem::trace_scanout_fill(span_addr, bytes.len(), "compute-storage");
                if !crate::guest_mem::write_bytes_checked(span_addr, bytes) {
                    return Err(err(format!(
                        "compute storage writeback range {span_addr:#x}..{:#x} is not writable \
                         guest memory",
                        span_addr.saturating_add(bytes.len() as u64)
                    )));
                }
            }
        }
        for ((addr, img_w, img_h, img_depth, img_layers, tile_mode, img_format), bytes) in
            guest_image_outputs.into_iter().zip(outputs.images)
        {
            let nonzero = bytes.iter().any(|&b| b != 0);
            debug!(
                addr = format_args!("{addr:#x}"),
                len = bytes.len(),
                nonzero,
                "compute storage-image writeback"
            );
            if crate::diagnostics::gpu_env().trace_draws {
                tracing::warn!(
                    addr = format_args!("{addr:#x}"),
                    len = bytes.len(),
                    nonzero,
                    "TRACE_DRAWS: compute image writeback"
                );
            }
            // Promote a content-bearing 8-bit UAV writeback into the present
            // census only when it matches a known scanout address/size:
            // ASTRO-class titles compose their scene with compute
            // dispatches into guest memory — never a GPU render pass we capture
            // — so without this the frame the census elects is always some flat
            // cleared draw target and the real pixels stay invisible. Only
            // R8G8B8A8 is promoted (already the Shell's RGBA byte order); HDR
            // (R16F) intermediates are left to the draw/scanout paths. Keyed by
            // guest base so a re-dispatch to the same UAV replaces in place.
            // Square texture atlases and mip targets stay GPU-resident but are
            // never allowed to replace the displayed frame.
            if nonzero && crate::guest_mem::is_scanout_candidate(addr, bytes.len()) {
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
                        Arc::new(RenderedImage {
                            width: img_w,
                            height: img_h,
                            pixels: bytes[..want].to_vec(),
                            bytes_per_pixel: bpp as u32,
                        }),
                    );
                }
            }
            let texel = match img_format {
                vk::Format::R16G16B16A16_SFLOAT => 8,
                vk::Format::R32G32B32A32_SFLOAT => 16,
                _ => 4,
            };
            let guest_bytes = encode_storage_image_writeback(
                img_w, img_h, img_depth, img_layers, tile_mode, texel, &bytes,
            )?;
            // A storage image can be GPU-only. The content-bearing result was
            // retained above for later sampling/presentation, so an unavailable
            // CPU guest mirror must not discard the dispatch.
            crate::guest_mem::mirror_compute_image_to_guest(addr, &guest_bytes, "compute-image");
        }
        self.dispatches += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kyty_graphics::hw_regs::{ColorAttrib2, ComputeShaderInfo};

    #[test]
    fn astrobot_device_loss_quarantine_is_exact() {
        assert!(is_known_device_loss_compute(0x5005_3c700, [64, 64, 1]));
        assert!(!is_known_device_loss_compute(0x5005_3c700, [64, 63, 1]));
        assert!(!is_known_device_loss_compute(0x5005_3c701, [64, 64, 1]));
    }

    #[test]
    fn gen5_stencil_operations_map_explicitly_to_vulkan() {
        let expected = [
            vk::StencilOp::KEEP,
            vk::StencilOp::ZERO,
            vk::StencilOp::REPLACE,
            vk::StencilOp::REPLACE,
            vk::StencilOp::REPLACE,
            vk::StencilOp::INCREMENT_AND_CLAMP,
            vk::StencilOp::DECREMENT_AND_CLAMP,
            vk::StencilOp::INVERT,
            vk::StencilOp::INCREMENT_AND_WRAP,
            vk::StencilOp::DECREMENT_AND_WRAP,
        ];
        for (guest, expected) in expected.into_iter().enumerate() {
            let (actual, reference) =
                gen5_stencil_op(guest as u8, 0x2A, 0x55).expect("known Gen5 op");
            assert_eq!(actual, expected, "guest stencil op {guest}");
            assert_eq!(
                reference,
                match guest {
                    2 => Some(0xFF),
                    3 => Some(0x2A),
                    4 => Some(0x55),
                    _ => None,
                },
                "guest stencil op {guest} reference"
            );
        }
        assert!(gen5_stencil_op(10, 0, 0).is_err());
    }

    #[test]
    fn keep_only_stencil_state_preserves_comparison_reference() {
        let state = gen5_stencil_state([0, 0, 0], 2, [0x2A, 0xF0, 0xFF, 0x55]).expect("KEEP state");
        assert_eq!(state.compare_op, vk::CompareOp::EQUAL);
        assert_eq!(state.reference, 0x2A);
        assert_eq!(state.compare_mask, 0xF0);
        assert_eq!(state.write_mask, 0xFF);
    }

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
    fn depth_registers_reach_the_live_draw_state_and_name_the_guest_surface() {
        let mut ctx = ctx_96x48();
        ctx.depth_control.z_enable = true;
        ctx.depth_control.z_write_enable = true;
        ctx.depth_control.zfunc = vk::CompareOp::LESS.as_raw() as u8;
        ctx.depth_render_target.z_info.format = 3; // 3 * 2 + 0 = D32_SFLOAT.
        ctx.depth_render_target.z_write_base_addr = 0x9000_0000;
        ctx.render_control.depth_clear_enable = true;
        ctx.depth_clear_value = 0.625;

        let state = draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV)
            .expect("depth-bearing register state");
        let depth = state.depth.expect("depth attachment is wired");
        assert_eq!(depth.target_base, Some(0x9000_0000));
        assert_eq!(depth.format, vk::Format::D32_SFLOAT);
        assert!(depth.test_enable);
        assert!(depth.write_enable);
        assert_eq!(depth.compare_op, vk::CompareOp::LESS);
        assert!(depth.clear_depth);
        assert_eq!(depth.clear_depth_value, 0.625);
    }

    #[test]
    fn viewport_derives_from_scale_and_offset() {
        let state =
            draw_state_from_regs(&ctx_96x48(), &ucfg_rect(), 3, SPIRV, SPIRV).expect("valid");
        // x = xoffset - xscale, w = xscale * 2
        assert_eq!(state.viewport, [0.0, 0.0, 96.0, 48.0]);
    }

    /// A second bound slot with a live `CB_TARGET_MASK` nibble becomes a real
    /// extra attachment carrying its own base, format, write mask, and
    /// per-slot blend.
    #[test]
    fn mrt_slots_reach_the_draw_state_with_per_slot_state() {
        let mut ctx = ctx_96x48();
        let rt2 = &mut ctx.render_targets[2];
        rt2.base.addr = 0x2_0000;
        rt2.info.format = 0xa; // RGBA8 UNORM
        rt2.attrib2 = ColorAttrib2 {
            width: 95,
            height: 47,
            num_mip_levels: 0,
        };
        // Slot 0 writes RGBA, slot 2 writes only R+A.
        ctx.render_target_mask = 0xF | (0x9 << 8);
        // Slot-2 blend differs from slot 0 (which stays disabled).
        ctx.blend_control[2] = kyty_graphics::hw_regs::BlendControl {
            color_srcblend: 0x04,  // SrcAlpha
            color_comb_fcn: 0x00,  // Add
            color_destblend: 0x05, // OneMinusSrcAlpha
            alpha_srcblend: 0x01,
            alpha_comb_fcn: 0x00,
            alpha_destblend: 0x00,
            separate_alpha_blend: true,
            enable: true,
        };

        let state =
            draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV).expect("valid MRT state");
        assert_eq!(state.mrt.len(), 1, "one extra attachment");
        let extra = &state.mrt[0];
        assert_eq!(extra.slot, 2);
        assert_eq!(extra.target_base, 0x2_0000);
        assert_eq!(extra.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            extra.write_mask,
            vk::ColorComponentFlags::R | vk::ColorComponentFlags::A
        );
        assert!(extra.blend.enable);
        assert_eq!(extra.blend.src_color, vk::BlendFactor::SRC_ALPHA);
        assert_eq!(extra.blend.dst_color, vk::BlendFactor::ONE_MINUS_SRC_ALPHA);
        // The primary attachment is untouched by slot-2 state.
        assert!(!state.blend.enable);
        assert_eq!(state.color_write_mask, vk::ColorComponentFlags::RGBA);
    }

    /// A bound slot whose `CB_TARGET_MASK` nibble is zero writes nothing on
    /// hardware — it must not become an attachment. A slot whose extent
    /// differs from slot 0 cannot share the render area — dropped (named).
    #[test]
    fn mrt_masked_or_mismatched_slots_are_not_attached() {
        let mut ctx = ctx_96x48();
        // Slot 1: bound, but mask nibble zero.
        ctx.render_targets[1].base.addr = 0x3_0000;
        ctx.render_targets[1].info.format = 0xa;
        ctx.render_targets[1].attrib2 = ctx.render_targets[0].attrib2;
        // Slot 3: bound + written, but a different extent.
        let rt3 = &mut ctx.render_targets[3];
        rt3.base.addr = 0x4_0000;
        rt3.info.format = 0xa;
        rt3.attrib2 = ColorAttrib2 {
            width: 31,
            height: 31,
            num_mip_levels: 0,
        };
        ctx.render_target_mask = 0xF | (0xF << 12);

        let state = draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV).expect("valid");
        assert!(
            state.mrt.is_empty(),
            "masked and extent-mismatched slots must not attach: {:?}",
            state.mrt
        );
    }

    /// The eliminate-fast-clear image is the packed `CLEAR_WORD` splatted in
    /// the target's own byte layout — no per-format unpack.
    #[test]
    fn fast_clear_image_splats_the_packed_clear_words() {
        let mut rt = kyty_graphics::hw_regs::RenderTarget::default();
        rt.base.addr = 0x1_0000;
        rt.info.format = 0xa; // RGBA8 UNORM
        rt.info.cmask_fast_clear_enable = true;
        rt.attrib2 = ColorAttrib2 {
            width: 3,
            height: 1,
            num_mip_levels: 0,
        };
        rt.clear_word0.word0 = 0x8040_20FF;

        let image = fast_clear_image(&rt, 1.0)
            .expect("mapped format")
            .expect("armed fast clear");
        assert_eq!(
            (image.width, image.height, image.bytes_per_pixel),
            (4, 2, 4)
        );
        assert_eq!(image.pixels.len(), 4 * 2 * 4);
        for px in image.pixels.chunks_exact(4) {
            assert_eq!(px, 0x8040_20FFu32.to_le_bytes());
        }
        // Resolution scale supersamples the clear like every draw.
        let scaled = fast_clear_image(&rt, 2.0).expect("mapped").expect("armed");
        assert_eq!((scaled.width, scaled.height), (8, 4));
    }

    /// FCE with nothing to eliminate (no base / fast clear unarmed /
    /// degenerate extent) is a quiet no-op; an unmapped format is a named
    /// error, never a wrong-coloured clear.
    #[test]
    fn fast_clear_image_refuses_unarmed_or_unmapped_state() {
        let mut rt = kyty_graphics::hw_regs::RenderTarget::default();
        assert!(matches!(fast_clear_image(&rt, 1.0), Ok(None)), "no base");
        rt.base.addr = 0x1_0000;
        rt.attrib2 = ColorAttrib2 {
            width: 3,
            height: 1,
            num_mip_levels: 0,
        };
        assert!(
            matches!(fast_clear_image(&rt, 1.0), Ok(None)),
            "fast clear not armed"
        );
        rt.info.cmask_fast_clear_enable = true;
        rt.attrib2.width = 0;
        assert!(
            matches!(fast_clear_image(&rt, 1.0), Ok(None)),
            "degenerate extent"
        );
        rt.attrib2.width = 3;
        rt.info.format = 0x1F; // unmapped CB format
        assert!(fast_clear_image(&rt, 1.0).is_err(), "unmapped format");
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

    #[test]
    fn indexed_quad_limits_vertex_upload_to_largest_referenced_record() {
        let indices = [1u16, 2, 0, 0, 2, 3];
        let bytes = indices
            .iter()
            .flat_map(|index| index.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            required_vertex_records(Some((&bytes, vk::IndexType::UINT16)), 6)
                .expect("valid quad indices"),
            4
        );
    }

    /// An empty index buffer addresses no record; sizing falls back to the
    /// draw's vertex count rather than refusing the draw outright.
    #[test]
    fn empty_index_buffer_falls_back_to_the_vertex_count() {
        assert_eq!(
            required_vertex_records(Some((&[], vk::IndexType::UINT16)), 6)
                .expect("an empty index buffer is not a refusal"),
            6
        );
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
        ctx.depth_control.z_enable = true;
        ctx.depth_control.z_write_enable = true;
        ctx.depth_control.zfunc = vk::CompareOp::LESS.as_raw() as u8;
        ctx.depth_render_target.z_write_base_addr = 0x2_0000;
        ctx.depth_render_target.z_info.format = 3;
        ctx.depth_render_target.stencil_info.format = 1;
        ctx.depth_render_target.size.x_max = 95;
        ctx.depth_render_target.size.y_max = 47;
        assert!(color_output_disabled(&ctx));
        let state = draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV)
            .expect("depth-only draw reaches the wired depth backend");
        assert!(!state.color_output);
        assert_eq!((state.width, state.height), (96, 48));
        assert!(state.color_write_mask.is_empty());
        assert_eq!(
            state.depth.expect("depth state").target_base,
            Some(0x2_0000)
        );
    }

    #[test]
    fn zero_target_mask_without_depth_or_stencil_is_a_named_no_output_error() {
        let mut ctx = ctx_96x48();
        ctx.render_target_mask = 0;
        let error = draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV)
            .expect_err("no attachment can receive the draw");
        assert!(error.0.contains("neither colour nor depth/stencil"));
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

    #[test]
    fn minecraft_skin_sampler_preserves_point_and_clamp_last_texel() {
        // Captured from Minecraft's live 64x64 player-skin draw. Clamp mode 2
        // is CLAMP_LAST_TEXEL on every axis; the old host path discarded it
        // and always created REPEAT samplers.
        let guest = ShaderSamplerResource {
            fields: [1682, 16_773_120, 100_663_296, 1_073_741_824],
        };
        let host = sampler_state(&guest);
        assert_eq!(host.mag_filter, vk::Filter::NEAREST);
        assert_eq!(host.min_filter, vk::Filter::NEAREST);
        assert_eq!(host.mipmap_mode, vk::SamplerMipmapMode::NEAREST);
        assert_eq!(host.address_mode_u, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(host.address_mode_v, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(host.address_mode_w, vk::SamplerAddressMode::CLAMP_TO_EDGE);
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
        vs.buffers[0].fetch_index = 1;
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
        assert!(
            buffers[0].per_instance,
            "Gen5 fetch_index=1 must reach a Vulkan per-instance binding"
        );
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
            storage.writable,
            vec![false],
            "an unspecified/read-only V# is an input, not a guest writeback"
        );
        assert_eq!(
            storage.buffers,
            vec![Arc::new(
                storage_words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>()
            )]
        );
        assert_eq!(binding.push_constants.len(), 16);
        assert_eq!(&binding.push_constants[0..4], &[0, 0, 0, 0]);
        assert_eq!(
            u32::from_le_bytes(binding.push_constants[4..8].try_into().unwrap()) >> 16,
            16,
            "rewritten descriptor must preserve the guest stride"
        );
    }

    /// Minecraft's in-world PS scalar-loads one dword at byte offset 8 from a
    /// constant V#. Its descriptor also carries ADD_TID + swizzle and an
    /// enormous stride/record product. Those flags affect MUBUF element
    /// addressing, not SBUFFER scalar loads; binding only the measured
    /// 12-byte prefix must proceed without attempting a multi-gigabyte upload.
    #[cfg(windows)]
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn scalar_constant_buffer_ignores_mubuf_flags_and_uploads_touched_prefix() {
        let guest = [
            0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
        ];
        let base = guest.as_ptr() as u64;

        let mut bind = ShaderBindResources::default();
        bind.push_constant_size = 16;
        bind.storage_buffers.buffers_num = 1;
        bind.storage_buffers.binding_index = 0;
        bind.storage_buffers.usages[0] = ShaderStorageUsage::Constant;
        bind.storage_buffers.required_bytes[0] = 12;
        let resource = &mut bind.storage_buffers.buffers[0];
        resource.update_address48(base);
        resource.fields[1] |= (896 << 16) | (1 << 31); // stride + swizzle
        resource.fields[2] = 5_226_499;
        resource.fields[3] |= 1 << 23; // ADD_TID

        let ranges = [(base, guest.len())];
        let binding = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_stage_binding(&bind, vk::ShaderStageFlags::FRAGMENT)
        })
        .expect("scalar constant V# must ignore MUBUF-only addressing flags");

        let storage = binding.storage_buffers.expect("constant-buffer binding");
        assert_eq!(storage.buffers, vec![Arc::new(guest.to_vec())]);
        assert_eq!(storage.guest_sizes, vec![guest.len()]);
        assert_eq!(storage.writable, vec![false]);
    }

    #[cfg(windows)]
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn vertex_upload_honors_draw_record_limit_instead_of_full_ring_buffer() {
        let guest = [0x5au8; 160];
        let base = guest.as_ptr() as u64;
        let mut vs = ShaderVertexInputInfo::default();
        vs.resources_num = 1;
        vs.buffers_num = 1;
        vs.resources[0].update_address48(base);
        vs.resources[0].fields[1] |= 20 << 16;
        vs.resources[0].fields[2] = 8;
        vs.resources[0].fields[3] = 71 << 12;
        vs.buffers[0].addr = base;
        vs.buffers[0].stride = 20;
        vs.buffers[0].num_records = 8;
        vs.buffers[0].attr_num = 1;
        vs.buffers[0].attr_indices[0] = 0;

        let ranges = [(base, guest.len())];
        let (buffers, _) = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_vertex_inputs_limited(&vs, Some(4))
        })
        .expect("four-record UI quad");
        assert_eq!(buffers[0].bytes.len(), 80);
    }

    /// A primitive-restart sentinel (`0xFFFF`) is walked as a REAL index — this
    /// pipeline never enables restart — so sizing by it yields 65536 records.
    /// That must CLAMP to what the V# exposes, not refuse: refusing dropped the
    /// whole draw and silently erased every restart-using primitive.
    #[cfg(windows)]
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn over_range_index_clamps_the_upload_instead_of_refusing_the_draw() {
        let guest = [0x5au8; 160];
        let base = guest.as_ptr() as u64;
        let mut vs = ShaderVertexInputInfo::default();
        vs.resources_num = 1;
        vs.buffers_num = 1;
        vs.resources[0].update_address48(base);
        vs.resources[0].fields[1] |= 20 << 16;
        vs.resources[0].fields[2] = 8;
        vs.resources[0].fields[3] = 71 << 12;
        vs.buffers[0].addr = base;
        vs.buffers[0].stride = 20;
        vs.buffers[0].num_records = 8;
        vs.buffers[0].attr_num = 1;
        vs.buffers[0].attr_indices[0] = 0;

        let ranges = [(base, guest.len())];
        let (buffers, _) = crate::guest_mem::with_test_ranges(&ranges, || {
            prepare_vertex_inputs_limited(&vs, Some(65536))
        })
        .expect("an over-range index clamps rather than refusing the draw");
        assert_eq!(
            buffers[0].bytes.len(),
            160,
            "clamped to the V#'s 8 records x 20-byte stride"
        );
    }

    /// ASTRO.BOT's measured HDR composite interleaves a float4 at offset 0 and
    /// another float4 at offset 16 in a 24-byte stride.  The second descriptor
    /// legally overlaps the next record, so the merged host upload must cover
    /// the final attribute fetch (80 bytes), not merely stride * records (72).
    #[cfg(windows)]
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn interleaved_vertex_upload_covers_final_attribute_extent() {
        let guest = [0x5au8; 80];
        let base = guest.as_ptr() as u64;

        let mut vs = ShaderVertexInputInfo::default();
        vs.resources_num = 2;
        vs.buffers_num = 1;
        for (index, offset) in [(0usize, 0u64), (1, 16)] {
            vs.resources[index].update_address48(base + offset);
            vs.resources[index].fields[1] |= 24 << 16;
            vs.resources[index].fields[2] = 3;
            vs.resources[index].fields[3] = 77 << 12; // float4 = 16 bytes
        }
        vs.buffers[0].addr = base;
        vs.buffers[0].stride = 24;
        vs.buffers[0].num_records = 3;
        vs.buffers[0].attr_num = 2;
        vs.buffers[0].attr_indices[0] = 0;
        vs.buffers[0].attr_indices[1] = 1;
        vs.buffers[0].attr_offsets[0] = 0;
        vs.buffers[0].attr_offsets[1] = 16;

        let ranges = [(base, guest.len())];
        let (buffers, attributes) =
            crate::guest_mem::with_test_ranges(&ranges, || prepare_vertex_inputs(&vs))
                .expect("measured ASTRO interleaved vertex ABI");
        assert_eq!(buffers[0].bytes.len(), 80);
        assert_eq!(attributes[1].format, vk::Format::R32G32B32A32_SFLOAT);
        assert_eq!(attributes[1].offset, 16);
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
        assert_eq!(storage.images[0].pixels.as_ref(), &uav);

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
        assert_eq!(upload.pixels.as_ref(), &vec![0u8; 64]);
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
        // 11 → (2,4) = 16 UINT (Minecraft's packed bone index). Its shader
        // input is declared uint by kyty-graphics and bitcast into the
        // float-backed guest VGPR, preserving the raw integer value.
        assert_eq!(gen5_vertex_format(11).unwrap(), vk::Format::R16_UINT);
        // GTA V's first live DCB uses unified 5 -> (1,4) = 8 UINT.
        assert_eq!(gen5_vertex_format(5).unwrap(), vk::Format::R8_UINT);
        assert_eq!(gen5_vertex_format_and_size(5).unwrap().1, 1);
        for f in [64u8, 74, 77, 56, 57, 71, 23] {
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

    #[test]
    fn texture_cache_hash_includes_the_tile_layout() {
        let bytes: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let base = bytes.as_ptr() as u64;
        let (linear, swizzle_s, swizzle_rx, repeat) =
            crate::guest_mem::with_test_ranges(&[(base, bytes.len())], || {
                (
                    guest_sample_hash(base, bytes.len() as u64, 0).unwrap(),
                    guest_sample_hash(base, bytes.len() as u64, 9).unwrap(),
                    guest_sample_hash(base, bytes.len() as u64, 27).unwrap(),
                    guest_sample_hash(base, bytes.len() as u64, 27).unwrap(),
                )
            });
        assert_ne!(
            linear, swizzle_s,
            "linear bytes must not reuse a tiled image"
        );
        assert_ne!(
            swizzle_s, swizzle_rx,
            "distinct GFX10 swizzles must not share a cached decode"
        );
        assert_eq!(swizzle_rx, repeat, "the same layout stays cacheable");
    }

    #[test]
    fn submission_texture_hash_memo_reuses_then_invalidates_a_probe() {
        let mut bytes: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let base = bytes.as_ptr() as u64;
        let memo = GuestSampleHashMemo::default();
        let first = crate::guest_mem::with_test_ranges(&[(base, bytes.len())], || {
            memo.get_or_compute(base, bytes.len() as u64, 27)
                .expect("first probe")
        });

        bytes[0] ^= 0xff;
        let reused = crate::guest_mem::with_test_ranges(&[(base, bytes.len())], || {
            memo.get_or_compute(base, bytes.len() as u64, 27)
                .expect("memo hit")
        });
        assert_eq!(first, reused, "the submission-local probe must be reused");

        memo.clear();
        let refreshed = crate::guest_mem::with_test_ranges(&[(base, bytes.len())], || {
            memo.get_or_compute(base, bytes.len() as u64, 27)
                .expect("probe after invalidation")
        });
        assert_ne!(
            first, refreshed,
            "a write boundary must force a fresh guest-memory probe"
        );
    }

    #[test]
    fn texture_hash_memo_invalidates_only_overlapping_compute_writes() {
        let mut a: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let mut b: Vec<u8> = (0..1024).map(|i| (i % 239) as u8).collect();
        let a_base = a.as_ptr() as u64;
        let b_base = b.as_ptr() as u64;
        let memo = GuestSampleHashMemo::default();
        let ranges = [(a_base, a.len()), (b_base, b.len())];
        let (a_first, b_first) = crate::guest_mem::with_test_ranges(&ranges, || {
            (
                memo.get_or_compute(a_base, a.len() as u64, 27).unwrap(),
                memo.get_or_compute(b_base, b.len() as u64, 27).unwrap(),
            )
        });

        a[0] ^= 0xff;
        b[0] ^= 0xff;
        memo.invalidate_ranges(&[(a_base, a.len() as u64)]);
        let (a_after, b_after) = crate::guest_mem::with_test_ranges(&ranges, || {
            (
                memo.get_or_compute(a_base, a.len() as u64, 27).unwrap(),
                memo.get_or_compute(b_base, b.len() as u64, 27).unwrap(),
            )
        });

        assert_ne!(a_first, a_after, "overlapping output must rehash");
        assert_eq!(b_first, b_after, "disjoint sampled texture stays memoized");
    }

    #[test]
    fn submission_shader_memo_reuses_only_exact_state_and_clears_at_boundaries() {
        let ctx = Context::default();
        let sh = Shader::default();
        let key = ResolvedShaderKey::new(&ctx, &sh);
        let shaders = ResolvedShaders {
            vs: Arc::new(vec![1, 2, 3]),
            ps: Arc::new(vec![4, 5, 6]),
            vs_info: ShaderVertexInputInfo::default(),
            ps_info: ShaderPixelInputInfo::default(),
        };
        let mut memo = ResolvedShaderMemo::default();

        assert!(memo.get(key).is_none());
        memo.insert(key, shaders.clone());
        let reused = memo.get(key).expect("exact state must be memoized");
        assert!(Arc::ptr_eq(&reused.vs, &shaders.vs));
        assert!(Arc::ptr_eq(&reused.ps, &shaders.ps));
        assert_eq!((memo.hits, memo.misses), (1, 1));

        let mut changed = sh.clone();
        changed.ps.ps_embedded = true;
        assert!(
            memo.get(ResolvedShaderKey::new(&ctx, &changed)).is_none(),
            "a stage-register change must not reuse the old resolution"
        );

        memo.clear();
        assert!(
            memo.get(key).is_none(),
            "a guest/compute write boundary must invalidate the resolution"
        );
    }

    #[test]
    fn submission_shader_memo_holds_a_full_commercial_frame_working_set() {
        let ctx = Context::default();
        let shaders = ResolvedShaders {
            vs: Arc::new(vec![1, 2, 3]),
            ps: Arc::new(vec![4, 5, 6]),
            vs_info: ShaderVertexInputInfo::default(),
            ps_info: ShaderPixelInputInfo::default(),
        };
        let mut memo = ResolvedShaderMemo::default();
        let working_set = 128u32;

        for value in 0..working_set {
            let mut sh = Shader::default();
            sh.ps.ps_user_sgpr.count = 1;
            sh.ps.ps_user_sgpr.value[0] = value;
            memo.insert(ResolvedShaderKey::new(&ctx, &sh), shaders.clone());
        }
        for value in 0..working_set {
            let mut sh = Shader::default();
            sh.ps.ps_user_sgpr.count = 1;
            sh.ps.ps_user_sgpr.value[0] = value;
            assert!(
                memo.get(ResolvedShaderKey::new(&ctx, &sh)).is_some(),
                "frame binding {value} was evicted before the next frame"
            );
        }
        assert_eq!(memo.hits, u64::from(working_set));
        assert_eq!(memo.misses, 0);
    }

    #[test]
    fn submission_shader_memo_invalidates_only_overlapping_compute_writes() {
        let ctx = Context::default();
        let mut sh = Shader::default();
        sh.vs.vs_regs.data_addr = 0x10_0000;
        sh.ps.ps_regs.data_addr = 0x20_0000;
        let key = ResolvedShaderKey::new(&ctx, &sh);
        let mut shaders = ResolvedShaders {
            vs: Arc::new(vec![1, 2, 3]),
            ps: Arc::new(vec![4, 5, 6]),
            vs_info: ShaderVertexInputInfo::default(),
            ps_info: ShaderPixelInputInfo::default(),
        };
        shaders.vs_info.bind.extended.used = true;
        shaders.vs_info.bind.extended.start_register = 12;
        shaders.vs_info.bind.extended.data.update_address(0x80_0000);
        shaders.vs_info.bind.storage_buffers.buffers_num = 1;
        shaders.vs_info.bind.storage_buffers.extended[0] = true;
        shaders.vs_info.bind.storage_buffers.start_register[0] = 12;

        let mut memo = ResolvedShaderMemo::default();
        memo.insert(key, shaders.clone());
        memo.invalidate_ranges(&[(0x40_0000, 0x1000)]);
        assert!(
            memo.get(key).is_some(),
            "an unrelated compute output must retain shader analysis"
        );

        memo.invalidate_ranges(&[(0x80_0004, 4)]);
        assert!(
            memo.get(key).is_none(),
            "an EUD descriptor write must invalidate shader analysis"
        );

        memo.insert(key, shaders);
        memo.invalidate_ranges(&[(0x10_0100, 4)]);
        assert!(
            memo.get(key).is_none(),
            "a guest shader-code write must invalidate shader analysis"
        );
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

    /// A placeholder T# (base 0) — the stand-in `kyty_graphics`
    /// `check_read_only_texture_type` installs for an unresolvable descriptor —
    /// decodes to a 1x1 transparent-black dummy without any guest read, so the
    /// draw/dispatch proceeds untextured instead of the whole shader being
    /// skipped.
    #[test]
    fn decode_texture_serves_placeholder_base_zero_as_1x1_dummy() {
        // The placeholder `kyty_graphics` installs for an unresolvable T# is a
        // base-0 Texture2D (type 9, identity dst_sel); its exact shape is the
        // gate's concern — here we only assert base 0 => 1x1 dummy.
        let mut t = kyty_graphics::shader::ShaderTextureResource {
            fields: [0, 0, 0, (9u32 << 28) | 0xFAC, 0, 0, 0, 0],
        };
        assert_eq!(t.base40(), 0, "placeholder is base 0");
        assert_eq!(t.type_(), 9);
        // No test guest ranges installed at all: proves the dummy needs no read.
        let tex = decode_texture(&t).expect("placeholder decodes without a guest read");
        assert_eq!((tex.width, tex.height), (1, 1));
        assert_eq!(tex.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(tex.pixels, vec![0u8; 4], "transparent black");
        assert_eq!(tex.layers, 1);
        assert_eq!(
            tex.guest_base, 0,
            "base 0 disables the persistent-texture cache"
        );
        // A base-0 T# with a garbage type still yields the dummy (the base-0
        // check precedes any type/format decode).
        t.fields[3] = 15 << 28;
        let tex = decode_texture(&t).expect("base-0 dummy is type-agnostic");
        assert_eq!((tex.width, tex.height), (1, 1));
    }

    #[test]
    fn sampled_render_target_matches_the_scaled_live_extent() {
        let format = vk::Format::R8G8B8A8_UNORM.as_raw();
        let live = [
            (0x31c0_0000, 960, 540, format),
            (
                0x31c0_0000,
                960,
                540,
                vk::Format::R16G16B16A16_SFLOAT.as_raw(),
            ),
            (0x9999_0000, 960, 540, format),
        ];
        assert_eq!(
            matching_live_target(&live, 0x31c0_0000, 1920, 1080, format, 0.5),
            Some((960, 540)),
            "a native T# must bind the resolution-scaled persistent target"
        );
        assert_eq!(
            matching_live_target(&live, 0x31c0_0000, 1920, 1080, format, 1.0),
            None,
            "a mismatched extent is not an alias when scaling is disabled"
        );
    }

    #[test]
    fn sampled_render_target_prefers_an_exact_extent() {
        let format = vk::Format::R8G8B8A8_UNORM.as_raw();
        let live = [
            (0x31c0_0000, 960, 540, format),
            (0x31c0_0000, 1920, 1080, format),
        ];
        assert_eq!(
            matching_live_target(&live, 0x31c0_0000, 1920, 1080, format, 0.5),
            Some((1920, 1080))
        );
    }

    fn replacement_candidate(cube: bool, array: bool, layers: u32, depth: u32) -> TextureUpload {
        TextureUpload {
            width: 8,
            height: 8,
            format: vk::Format::R8G8B8A8_UNORM,
            pixels: vec![0; 8 * 8 * 4],
            layers,
            cube,
            array,
            volume: depth > 1,
            depth,
            render_target: None,
            guest_base: 0x1000,
            sample_hash: 1,
            cached: false,
        }
    }

    /// A framebuffer entry is one 2D attachment, never an alias for every face
    /// of a cube/array or every slice of a volume.
    #[test]
    fn render_target_pixel_fallback_only_accepts_plain_2d_uploads() {
        assert!(can_replace_with_render_target_pixels(
            &replacement_candidate(false, false, 1, 1)
        ));
        assert!(!can_replace_with_render_target_pixels(
            &replacement_candidate(true, false, 6, 1)
        ));
        assert!(!can_replace_with_render_target_pixels(
            &replacement_candidate(false, true, 1, 1)
        ));
        assert!(!can_replace_with_render_target_pixels(
            &replacement_candidate(false, false, 1, 4)
        ));
    }

    /// The COLOUR-BUFFER (CB_COLOR_INFO) format table is a different table from
    /// the texture (T#) one — same data-format numbering, different consumer.
    /// Every accepted entry must also have a `readback_bpp` size or the
    /// offscreen readback fails at run time, so they are asserted together.
    #[test]
    fn cb_colour_formats_map_and_have_readback_sizes() {
        for (fmt, ty, order, want, bpp) in [
            (0xa, 0, 0, vk::Format::R8G8B8A8_UNORM, 4u32),
            (0x4, 7, 0, vk::Format::R32_SFLOAT, 4),
            (0x5, 7, 0, vk::Format::R16G16_SFLOAT, 4),
            (0x6, 7, 0, vk::Format::B10G11R11_UFLOAT_PACK32, 4),
            (0x9, 0, 0, vk::Format::A2B10G10R10_UNORM_PACK32, 4),
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
    fn texture_vk_format_maps_unified_14_29_65_and_77() {
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
        assert_eq!(
            case(77).expect("format 77"),
            (vk::Format::R32G32B32A32_SFLOAT, 16)
        );
    }

    /// A guest cube T# (type 11, six faces, SWIZZLE_MODE 9 = SW_64KB_S)
    /// decodes every face and exposes the guest-computed `(s,t,face)` through
    /// a six-layer 2D-array upload.
    #[test]
    fn decode_guest_cube_as_six_layer_2d_array() {
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
        assert!(!tex.cube);
        assert!(tex.array);
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

    /// Recon rank 9: a T# selecting a non-zero mip as its view base is served
    /// mip 0 at mip 0's extent, because `base_level()` is consumed nowhere.
    ///
    /// This pins the CURRENT, KNOWN-WRONG behaviour together with the counter
    /// that makes it visible. It is deliberately not an assertion that mip
    /// addressing works — it does not. Locating mip N needs the GFX10 mip-tail
    /// layout, which is not derivable from anything in this tree, and a guessed
    /// offset would sample unrelated memory (worse than a consistent mip 0).
    ///
    /// When mip addressing IS implemented, this test should fail and be
    /// rewritten to assert the mip-N extent and contents. Until then the counter
    /// is how a real title decides whether that work is needed at all.
    #[test]
    fn nonzero_base_level_is_counted_and_still_serves_mip_zero() {
        use std::sync::atomic::Ordering;

        let (w, h, bpp) = (8u32, 8u32, 4usize);
        let pixels: Vec<u8> = (0..(w * h) as usize * bpp)
            .map(|i| (i % 251) as u8)
            .collect();
        // `base40()` truncates to 256 bytes, so the guest base must be
        // 256-aligned or the decode reads BELOW the registered test range.
        let tiled = crate::texture::tiling::tile_64kb_s(&pixels, w, h, 2);
        let mut blob = vec![0u8; tiled.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + tiled.len()].copy_from_slice(&tiled);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 10 << 20; // unified format 10 = 8_8_8_8 UNORM
        t.fields[1] |= ((w - 1) & 3) << 30;
        t.fields[2] = (w - 1) >> 2;
        t.fields[2] |= (h - 1) << 14;
        t.fields[3] |= 9 << 20; // SWIZZLE_MODE 9 = SW_64KB_S
        t.fields[3] |= 9 << 28; // type = Texture2D
        // The whole point: view starts at mip 2 of a 4-level chain.
        t.fields[3] |= 2 << 12; // base_level = 2
        t.fields[3] |= 3 << 16; // last_level = 3
        assert_eq!(t.base_level(), 2, "the fixture must really set base_level");
        assert_eq!(t.last_level(), 3);

        let before = MIP_VIEW_BASE_LEVEL_IGNORED.load(Ordering::Relaxed);
        let tex = crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
            decode_texture(&t)
        })
        .expect("a mipped T# must still decode (degrade, not refuse)");
        let after = MIP_VIEW_BASE_LEVEL_IGNORED.load(Ordering::Relaxed);

        assert_eq!(
            after,
            before + 1,
            "a non-zero base_level must be COUNTED, so a title run reveals whether \
             mip addressing is worth implementing"
        );
        // Documents the defect precisely: mip 2 of an 8x8 chain is 2x2, but the
        // decode still produces the mip-0 extent from the mip-0 bytes.
        assert_eq!(
            (tex.width, tex.height),
            (w, h),
            "KNOWN WRONG: mip 2 of 8x8 is 2x2; base_level is ignored so the \
             mip-0 extent is served. Rewrite this assertion when mip addressing lands."
        );
        assert_eq!(tex.pixels, pixels, "the bytes served are mip 0's");

        // And a base_level of 0 must not touch the counter.
        let mut plain = kyty_graphics::shader::ShaderTextureResource::default();
        plain.update_address40(base >> 8);
        plain.fields[1] |= 10 << 20;
        plain.fields[1] |= ((w - 1) & 3) << 30;
        plain.fields[2] = (w - 1) >> 2;
        plain.fields[2] |= (h - 1) << 14;
        plain.fields[3] |= 9 << 20;
        plain.fields[3] |= 9 << 28;
        let before = MIP_VIEW_BASE_LEVEL_IGNORED.load(Ordering::Relaxed);
        let _ =
            crate::guest_mem::with_test_ranges(&[(base, blob.len())], || decode_texture(&plain));
        assert_eq!(
            MIP_VIEW_BASE_LEVEL_IGNORED.load(Ordering::Relaxed),
            before,
            "base_level 0 is the ordinary case and must not be counted"
        );
    }

    /// `MAX_MIP + 1` is the ALLOCATION's level count, clamped to the most an
    /// extent of that size can carry — never taken from a view's LAST_LEVEL.
    #[test]
    fn resource_mip_levels_clamps_max_mip_to_the_extent() {
        assert_eq!(
            resource_mip_levels(0, 512, 512, 1),
            1,
            "MAX_MIP 0 = 1 level"
        );
        assert_eq!(resource_mip_levels(9, 512, 512, 1), 10);
        assert_eq!(
            resource_mip_levels(15, 64, 64, 1),
            7,
            "a 64x64 surface holds 7 levels; a stale/all-ones MAX_MIP must not \
             compute a chain offset past the real allocation"
        );
        assert_eq!(
            resource_mip_levels(9, 1, 1, 1),
            1,
            "a 1x1 surface has exactly one level"
        );
        assert_eq!(
            resource_mip_levels(9, 8, 4, 32),
            6,
            "a volume's largest dimension can be its depth"
        );
    }

    /// SharpEmu #470: a GFX10 mip chain is stored SMALLEST-FIRST, so mip 0 lives
    /// at the END of the allocation. Reading the descriptor base for a mipped
    /// texture decodes the mip TAIL at mip 0's extent — measured by SharpEmu as
    /// scrambled menu text and repeated icons.
    ///
    /// The fixture is exact: a 128x128 RGBA8 `SW_64KB_S` surface with 4 levels
    /// has a 128x64 tail extent, so mip 0 (128x128) and nothing else sits outside
    /// the tail. Its chain is therefore `[tail 64 KiB][mip 0 64 KiB]` and mip 0
    /// begins 65536 B past the base. The tail half of the blob is filled with a
    /// DIFFERENT pattern, so serving it would fail loudly.
    #[test]
    fn mip_chain_reads_mip_zero_from_the_end_of_the_allocation() {
        use std::sync::atomic::Ordering;

        let (w, h, bpp_log2) = (128u32, 128u32, 2u32);
        let mip0: Vec<u8> = (0..(w * h) as usize * 4).map(|i| (i % 251) as u8).collect();
        let mip0_tiled = crate::texture::tiling::tile_64kb_s(&mip0, w, h, bpp_log2);
        assert_eq!(mip0_tiled.len(), 65_536, "one 64 KiB swizzle block");

        let placement = crate::texture::tiling::base_mip_placement(9, w, h, bpp_log2, 4)
            .expect("128x128 with 4 levels places");
        assert_eq!(placement.byte_offset, 65_536);
        assert_eq!(placement.tail_element, None);

        // [mip tail 64 KiB, filled 0xA5][mip 0, tiled], 256-aligned for base40().
        let chain_len = placement.byte_offset as usize + mip0_tiled.len();
        let mut blob = vec![0xA5u8; chain_len + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off + placement.byte_offset as usize..off + chain_len].copy_from_slice(&mip0_tiled);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 10 << 20; // unified format 10 = 8_8_8_8 UNORM
        t.fields[1] |= ((w - 1) & 3) << 30;
        t.fields[2] = (w - 1) >> 2;
        t.fields[2] |= (h - 1) << 14;
        t.fields[3] |= 9 << 20; // SWIZZLE_MODE 9 = SW_64KB_S
        t.fields[3] |= 9 << 28; // type = Texture2D
        t.fields[3] |= 3 << 16; // last_level = 3 (a view of the whole chain)
        t.fields[5] |= 3 << 4; // MAX_MIP = 3 -> a 4-level ALLOCATION
        assert_eq!(t.max_mip(), 3, "the fixture must really set MAX_MIP");
        assert_eq!(t.base_level(), 0, "the COMMON case: an ordinary mip-0 view");

        let ranges = [(blob.as_ptr() as u64, blob.len())];
        let before_chain = MIP_CHAIN_TEXTURES.load(Ordering::Relaxed);
        let before_unknown = MIP_CHAIN_PLACEMENT_UNKNOWN.load(Ordering::Relaxed);
        let before_view = MIP_VIEW_BASE_LEVEL_IGNORED.load(Ordering::Relaxed);
        let tex = crate::guest_mem::with_test_ranges(&ranges, || decode_texture(&t))
            .expect("a mipped T# must decode");
        assert_eq!(
            tex.pixels, mip0,
            "mip 0 must come from the END of the chain, not the descriptor base"
        );
        assert_eq!(
            MIP_CHAIN_TEXTURES.load(Ordering::Relaxed),
            before_chain + 1,
            "MAX_MIP > 0 must be counted even with base_level == 0 — that silent \
             case is exactly what took the tail's bytes before #470"
        );
        assert_eq!(
            MIP_CHAIN_PLACEMENT_UNKNOWN.load(Ordering::Relaxed),
            before_unknown,
            "a placed chain is not an unknown placement"
        );
        assert_eq!(
            MIP_VIEW_BASE_LEVEL_IGNORED.load(Ordering::Relaxed),
            before_view,
            "base_level is still 0: the view counter must not move"
        );

        // The same bytes with MAX_MIP = 0 is the pre-#470 read: it serves the
        // 0xA5 tail block, which must NOT be what a mipped descriptor gets.
        let mut unmipped = t;
        unmipped.fields[5] = 0;
        let tail = crate::guest_mem::with_test_ranges(&ranges, || decode_texture(&unmipped))
            .expect("an unmipped T# still decodes at the base");
        assert_ne!(
            tail.pixels, mip0,
            "reading at the descriptor base must be the WRONG bytes — otherwise \
             the fixture does not reproduce the bug being fixed"
        );
        assert!(
            tail.pixels.iter().all(|&byte| byte == 0xA5),
            "the pre-fix read is the tail block verbatim"
        );
    }

    /// The bind-side sampled view kind is a pure function of the T# TYPE and
    /// agrees, arm for arm, with the emitter's `SampledDim::from_texture_type`.
    /// This is the invariant that keeps the bound `VkImageViewType` matching the
    /// recompiled SPIR-V's `OpTypeImage` Arrayed/Dim — its violation was the
    /// ASTRO.BOT array/cube `vkCmdDispatch` device-loss (view type 2D under an
    /// `Arrayed = 1` sampled image).
    #[test]
    fn texture_view_kind_matches_emitter_sampled_dim() {
        use kyty_graphics::shader::SampledDim;
        for ty in [8u8, 9, 10, 11, 13] {
            let (cube, volume, array) = texture_view_kind(ty).expect("accepted sampled type");
            let dim = SampledDim::from_texture_type(ty);
            assert_eq!(cube, dim == SampledDim::Cube, "cube flag for type {ty}");
            assert_eq!(
                volume,
                dim == SampledDim::Three,
                "volume flag for type {ty}"
            );
            assert_eq!(
                array,
                dim == SampledDim::TwoArray,
                "array flag for type {ty}"
            );
        }
        // Guest cube type 11 and explicit array type 13 are both arrayed.
        // The array flag is TYPE-driven, so a single-layer descriptor still
        // binds TYPE_2D_ARRAY, matching SPIR-V Arrayed = 1.
        assert_eq!(texture_view_kind(13).unwrap(), (false, false, true));
        assert_eq!(texture_view_kind(9).unwrap(), (false, false, false));
        assert_eq!(texture_view_kind(11).unwrap(), (false, false, true));
        assert_eq!(texture_view_kind(10).unwrap(), (false, true, false));
        // 12/14/15 never reach decode (analysis rewrites/replaces them); an
        // unhandled nibble stays a named refusal rather than a silent 2D guess.
        for ty in [12u8, 14, 15] {
            assert!(
                texture_view_kind(ty).is_err(),
                "type {ty} is a named refusal"
            );
        }
    }

    /// Minecraft's panorama compute writes a type-13 `SW_64KB_S` storage
    /// array and later samples the same guest bytes as six faces. The host
    /// dispatch therefore needs a lossless two-way layout conversion for
    /// every layer, not a linear write into tiled guest memory.
    #[test]
    fn storage_image_2darray_detile_and_writeback_round_trip_every_layer() {
        let (w, h, bpp_log2, layer_count) = (8u32, 8u32, 2u32, 3u32);
        let bpp = 1usize << bpp_log2;
        let layers: Vec<Vec<u8>> = (0..layer_count as u8)
            .map(|layer| {
                (0..(w * h) as usize * bpp)
                    .map(|i| layer.wrapping_mul(73).wrapping_add((i % 67) as u8))
                    .collect()
            })
            .collect();
        let tiled: Vec<u8> = layers
            .iter()
            .flat_map(|layer| crate::texture::tiling::tile_64kb_s(layer, w, h, bpp_log2))
            .collect();
        let mut blob = vec![0u8; tiled.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + tiled.len()].copy_from_slice(&tiled);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 56 << 20; // unified format 8_8_8_8 UNORM
        t.fields[1] |= ((w - 1) & 3) << 30;
        t.fields[2] = (w - 1) >> 2;
        t.fields[2] |= (h - 1) << 14;
        t.fields[3] |= 9 << 20; // SWIZZLE_MODE 9 = SW_64KB_S
        t.fields[3] |= 13 << 28; // type = 2DArray
        t.fields[4] = layer_count - 1;

        let upload =
            crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
                read_storage_image(&t)
            })
            .expect("2D-array storage image detiles");
        assert!(upload.array, "the Vulkan view must remain arrayed");
        assert_eq!(upload.layers, layer_count);
        assert_eq!(upload.depth, 1);
        assert_eq!(
            upload.pixels.as_ref(),
            &layers.into_iter().flatten().collect::<Vec<_>>(),
            "all guest layers detile to tightly-packed host rows"
        );

        let guest = encode_storage_image_writeback(
            upload.width,
            upload.height,
            upload.depth,
            upload.layers,
            upload.tile_mode,
            upload.texel_bytes(),
            &upload.pixels,
        )
        .expect("host readback retiles");
        assert_eq!(
            guest, tiled,
            "writeback must reconstruct every original guest swizzle block"
        );
    }

    #[test]
    fn storage_image_base_array_selects_one_guest_layer() {
        let (w, h, bpp_log2) = (8u32, 8u32, 2u32);
        let bpp = 1usize << bpp_log2;
        let layers: Vec<Vec<u8>> = (0..3u8)
            .map(|layer| {
                (0..(w * h) as usize * bpp)
                    .map(|i| layer.wrapping_mul(73).wrapping_add((i % 67) as u8))
                    .collect()
            })
            .collect();
        let tiled_layers: Vec<Vec<u8>> = layers
            .iter()
            .map(|layer| crate::texture::tiling::tile_64kb_s(layer, w, h, bpp_log2))
            .collect();
        let tiled: Vec<u8> = tiled_layers.iter().flatten().copied().collect();
        let mut blob = vec![0u8; tiled.len() + 255];
        let allocation_base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (allocation_base - blob.as_ptr() as u64) as usize;
        blob[off..off + tiled.len()].copy_from_slice(&tiled);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(allocation_base >> 8);
        t.fields[1] |= 56 << 20;
        t.fields[1] |= ((w - 1) & 3) << 30;
        t.fields[2] = (w - 1) >> 2;
        t.fields[2] |= (h - 1) << 14;
        t.fields[3] |= 9 << 20;
        t.fields[3] |= 13 << 28;
        // Exactly Minecraft's per-face descriptor shape: BASE_ARRAY and
        // LAST_ARRAY both name the one face this dispatch updates.
        t.fields[4] = 2 | (2 << 16);

        let upload =
            crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
                read_storage_image(&t)
            })
            .expect("selected array layer detiles");
        assert_eq!(upload.layers, 1);
        assert_eq!(upload.pixels.as_ref(), &layers[2]);
        assert_eq!(
            upload.guest_base,
            allocation_base + (2 * tiled_layers[0].len()) as u64,
            "writeback starts at BASE_ARRAY, preserving earlier faces"
        );

        let guest = encode_storage_image_writeback(
            upload.width,
            upload.height,
            upload.depth,
            upload.layers,
            upload.tile_mode,
            upload.texel_bytes(),
            &upload.pixels,
        )
        .expect("selected layer retiles");
        assert_eq!(guest, tiled_layers[2]);
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
        assert_eq!(
            upload.pixels.as_ref(),
            &bytes,
            "the whole volume seeds the upload"
        );
    }

    /// Live ASTRO.BOT UAV descriptor: type 8 is a 1D image represented by a
    /// height-1 Vulkan 2D image, and format 77 is RGBA32F (16 B/texel).
    #[test]
    fn read_storage_image_type8_rgba32f_reads_sixteen_byte_texel() {
        let bytes: Vec<u8> = (0..16u8).map(|x| x.wrapping_mul(7)).collect();
        let mut blob = vec![0u8; bytes.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + bytes.len()].copy_from_slice(&bytes);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 77 << 20;
        t.fields[3] |= 8 << 28;

        let upload =
            crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
                read_storage_image(&t)
            })
            .expect("type-8 RGBA32F UAV reads");
        assert_eq!((upload.width, upload.height, upload.depth), (1, 1, 1));
        assert_eq!(upload.format, vk::Format::R32G32B32A32_SFLOAT);
        assert_eq!(upload.texel_bytes(), 16);
        assert_eq!(upload.pixels.as_ref(), &bytes);
        assert_eq!(expected_storage_image_bytes(&t), 16);
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

    /// GTA V's first non-linear 3D T# is a one-voxel RGBA8 texture in tile
    /// mode 5. Every block layout places coordinate (0, 0, 0) at the source
    /// base, so this trivial extent does not need (and must not pretend to
    /// implement) the still-unknown general 3D tile-mode-5 equation.
    #[test]
    fn gta_tile5_single_voxel_volume_reads_the_origin_texel() {
        let rgba = [0x12, 0x34, 0x56, 0x78];
        let mut blob = vec![0u8; rgba.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + rgba.len()].copy_from_slice(&rgba);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 56 << 20; // unified format 8_8_8_8 UNORM
        t.fields[3] |= 5 << 20; // measured GTA tile mode
        t.fields[3] |= 10 << 28; // type = 3D volume

        let tex = crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
            decode_texture(&t)
        })
        .expect("the single origin voxel needs no general 3D detile equation");
        assert!(!tex.cube);
        assert_eq!(tex.layers, 1);
        assert_eq!((tex.width, tex.height, tex.depth), (1, 1, 1));
        assert_eq!(tex.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(tex.pixels, rgba);

        t.fields[4] = 1; // depth = 2: no longer the measured trivial extent
        let refused =
            crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
                decode_texture(&t)
            })
            .expect_err("a nontrivial tiled volume still needs the real 3D equation");
        assert!(
            refused.0.contains("3D texture tile mode 5 not implemented"),
            "the refusal must name the missing layout: {refused:?}"
        );
    }

    /// SharpEmu PR #587 (Gen5 3D images): the DEPTH the guest transports must
    /// not decide the host image TYPE. A type-10 T# whose DEPTH field is 0 is a
    /// one-slice volume — `depth == 1` — yet the recompiler still declares
    /// `Dim3D` for it (`SampledDim::from_texture_type(10) == Three`), so the
    /// upload must ask for a `VK_IMAGE_TYPE_3D` image and a `TYPE_3D` view.
    ///
    /// RED before the `volume` flag: both the sampled and the storage create
    /// sites derived the volume branch from `depth > 1`, so exactly this
    /// measured GTA V shape built a 2D image/view under a `Dim3D` image type —
    /// the emit/bind divergence class that already cost a device loss for the
    /// arrayed case.
    #[test]
    fn one_slice_type10_volume_stays_a_3d_image_not_a_2d_one() {
        let rgba = [0x12, 0x34, 0x56, 0x78];
        let mut blob = vec![0u8; rgba.len() + 255];
        let base = (blob.as_ptr() as u64 + 255) & !255;
        let off = (base - blob.as_ptr() as u64) as usize;
        blob[off..off + rgba.len()].copy_from_slice(&rgba);

        let mut t = kyty_graphics::shader::ShaderTextureResource::default();
        t.update_address40(base >> 8);
        t.fields[1] |= 56 << 20; // unified format 8_8_8_8 UNORM
        t.fields[3] |= 5 << 20; // measured GTA tile mode
        t.fields[3] |= 10 << 28; // type = 3D volume, DEPTH field left at 0

        let tex = crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
            decode_texture(&t)
        })
        .expect("one-slice volume decodes");
        assert_eq!(tex.depth, 1, "DEPTH 0 is a single slice");
        assert!(
            tex.volume,
            "a type-10 T# is a volume regardless of its slice count"
        );
        assert!(!tex.array && !tex.cube);
        // The classifier the recompiler reads must agree — the two sides of the
        // emit/bind contract.
        assert_eq!(
            kyty_graphics::shader::SampledDim::from_texture_type(t.type_()),
            kyty_graphics::shader::SampledDim::Three,
            "SPIR-V declares Dim3D for this descriptor"
        );

        // The storage (UAV) side of the same descriptor, same contract.
        let uav = read_storage_image(&t).expect("one-slice storage volume decodes");
        assert_eq!(uav.depth, 1);
        assert!(uav.volume, "a type-10 UAV is a 3D storage image");
        assert!(!uav.array);

        // A plain 2D descriptor of the same extent must NOT claim to be one —
        // otherwise the assertion above would pass for the wrong reason.
        let mut flat = t;
        flat.fields[3] &= !(0xF << 28);
        flat.fields[3] |= 9 << 28;
        flat.fields[3] &= !(0x1F << 20); // linear: the 2D path reads 4 bytes, not a tile
        let flat_tex =
            crate::guest_mem::with_test_ranges(&[(blob.as_ptr() as u64, blob.len())], || {
                decode_texture(&flat)
            })
            .expect("2D texture decodes");
        assert!(!flat_tex.volume, "a type-9 T# is never a volume");
    }
}
