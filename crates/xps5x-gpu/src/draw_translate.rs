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
//! this slice cannot honour is a **named error**, never a fallback: notably a
//! non-embedded shader bind needs a guest-memory reader (`ShaderMemory`) that
//! Phase 1 does not have, so such a draw is logged and skipped rather than
//! quietly rendered as something else.

use crate::vulkan::instance::VulkanDevice;
use crate::vulkan::offscreen::{CLEAR_COLOR, DrawState, RenderedImage, render_draw};
use ash::vk;
use kyty_graphics::hw_regs::{Context, Shader, UserConfig};
use kyty_graphics::run::{DrawError, DrawSink};
use kyty_graphics::shader::{spirv_get_embedded_ps, spirv_get_embedded_vs};
use kyty_graphics::spirv_asm;
use tracing::warn;

/// `VGT_PRIMITIVE_TYPE` values Kyty's Gen5 path emits.
mod prim {
    pub const TRIANGLE_LIST: u32 = 4;
    pub const TRIANGLE_STRIP: u32 = 5;
    /// Kyty's clear/blit primitive. Rasterized as a 4-vertex strip quad.
    pub const RECT_LIST: u32 = 17;
}

fn err(msg: impl Into<String>) -> DrawError {
    DrawError(msg.into())
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

/// Assemble the SPIR-V for a stage, or explain why we cannot.
fn embedded_or_skip(
    embedded: bool,
    id: u32,
    data_addr: u64,
    stage: &str,
) -> Result<Vec<u32>, DrawError> {
    if !embedded {
        // The real path needs ShaderMemory over the guest arena (Phase 2).
        warn!(
            stage,
            data_addr = format_args!("{data_addr:#x}"),
            "non-embedded shader bind requires ShaderMemory (Phase 2) — draw skipped"
        );
        return Err(err(format!(
            "{stage} shader at {data_addr:#x} is not embedded; a guest-memory \
             shader bind requires ShaderMemory (Phase 2)"
        )));
    }
    let source = match stage {
        "vs" => spirv_get_embedded_vs(id),
        _ => spirv_get_embedded_ps(id),
    }
    .map_err(|e| err(format!("embedded {stage} id {id}: {e}")))?;

    spirv_asm::assemble(source).map_err(|e| err(format!("assembling embedded {stage}: {e}")))
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
    // Kyty accepts only a fully-enabled or fully-disabled colour target.
    match ctx.render_target_mask {
        0xF => {}
        0 => return Err(err("CB_TARGET_MASK is 0 — colour output disabled")),
        other => {
            return Err(err(format!(
                "CB_TARGET_MASK {other:#x} is a partial write mask; only 0xF is supported"
            )));
        }
    }

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
        prim::TRIANGLE_STRIP => (vk::PrimitiveTopology::TRIANGLE_STRIP, index_count),
        other => {
            return Err(err(format!(
                "unsupported VGT_PRIMITIVE_TYPE {other} (supported: 4 TriList, \
                 5 TriStrip, 17 RectList)"
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
        // The embedded VS declares no input attributes and builds its own quad.
        vertices: None,
        vertex_count,
        vs_spirv,
        fs_spirv,
    })
}

/// A [`DrawSink`] that renders each draw offscreen and keeps the last image.
pub struct OffscreenDrawSink<'a> {
    dev: &'a VulkanDevice,
    pub last: Option<RenderedImage>,
    pub draws: u64,
}

impl<'a> OffscreenDrawSink<'a> {
    #[must_use]
    pub fn new(dev: &'a VulkanDevice) -> Self {
        Self {
            dev,
            last: None,
            draws: 0,
        }
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
        let vs = embedded_or_skip(
            sh.vs.vs_embedded,
            sh.vs.vs_embedded_id,
            sh.vs.vs_regs.data_addr,
            "vs",
        )?;
        let fs = embedded_or_skip(
            sh.ps.ps_embedded,
            sh.ps.ps_embedded_id,
            sh.ps.ps_regs.data_addr,
            "ps",
        )?;

        let state = draw_state_from_regs(ctx, ucfg, index_count, &vs, &fs)?;
        let image = render_draw(self.dev, &state)
            .map_err(|e| err(format!("offscreen draw failed: {e}")))?;

        self.last = Some(image);
        self.draws += 1;
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

    #[test]
    fn partial_render_target_mask_is_a_named_error() {
        let mut ctx = ctx_96x48();
        ctx.render_target_mask = 0x3;
        let e =
            draw_state_from_regs(&ctx, &ucfg_rect(), 3, SPIRV, SPIRV).expect_err("partial mask");
        assert!(e.0.contains("CB_TARGET_MASK"), "got {e}");
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
    fn non_embedded_shader_is_skipped_with_a_named_error() {
        let e = embedded_or_skip(false, 0, 0x1000, "vs").expect_err("needs ShaderMemory");
        assert!(e.0.contains("ShaderMemory"), "got {e}");
    }

    /// The embedded PS is `outColor = vec4(0)`. Alpha 0 is unreachable from the
    /// fixture, which is what makes the acceptance assertion decisive.
    #[test]
    fn embedded_shaders_assemble_to_real_spirv() {
        let vs = embedded_or_skip(true, 0, 0, "vs").expect("embedded VS");
        let ps = embedded_or_skip(true, 0, 0, "ps").expect("embedded PS");
        assert_eq!(vs[0], 0x0723_0203, "VS SPIR-V magic");
        assert_eq!(ps[0], 0x0723_0203, "PS SPIR-V magic");
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
}
