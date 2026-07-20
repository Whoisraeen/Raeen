//! Kyty's hardware-context register model. Ported from Kyty
//! (MIT (c) InoriRus).
//!
//! Kyty source: `emulator/include/Emulator/Graphics/HardwareContext.h`
//! (namespace `HW`). This is a **partial** port: every struct carries its Kyty
//! anchor and keeps the members that either `Shader.cpp` analysis reads or the
//! PM4 command processor ([`crate::run`]) writes on the minimal draw path.
//! DCC/CMASK/FMASK and the Gen4 pitch/slice decode are not ported yet.
//!
//! `HardwareContext.h` is generation-agnostic, so this lives at crate level
//! rather than under `shader::` — both the recompiler and the command processor
//! read it. `shader::hw_regs` remains as an alias.
//!
//! # Defaults are not all zero
//!
//! Four fields have non-zero Kyty defaults and a `#[derive(Default)]` would
//! silently get them wrong: [`Context::line_width`] (1.0),
//! [`ColorControl::mode`] (1) / [`ColorControl::op`] (0xCC),
//! [`ScanModeControl::vport_scissor_enable`] (true), and
//! [`ScreenViewport::transform_control`] (1087). Their `Default` impls are
//! hand-written and pinned by tests.

/// Kyty: HardwareContext.h `UserSgprType` (L579).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum UserSgprType {
    #[default]
    Unknown,
    Region,
    Vsharp,
}

/// Kyty: HardwareContext.h `UserSgprInfo` (L586) — the user-SGPR values
/// latched from PM4 `SET_SH_REG` writes, plus what kind of data each holds.
/// Sized by [`Self::SGPRS_MAX`], widened to Gen5's 32 (see there).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct UserSgprInfo {
    pub value: [u32; Self::SGPRS_MAX],
    pub type_: [UserSgprType; Self::SGPRS_MAX],
    pub count: u32,
}

impl UserSgprInfo {
    /// Gen5 graphics stages expose 32 user-SGPR registers. Kyty's PS4-era code
    /// used 16, which silently DROPPED writes to slots 16..31 and capped
    /// `count` at 16 — so every ASTRO.BOT pixel shader declaring 20/24/30/32
    /// user SGPRs was rejected by the `user_sgpr > count` gate and no draw
    /// could translate. The SH register map leaves room for 32 per graphics
    /// stage (PS 0x0C, VS 0x4C, GS 0x8C, ES 0xC8); SharpEmu's Gen5 scalar
    /// evaluator likewise carries up to 32. Compute keeps its hardware 16
    /// (COMPUTE_USER_DATA_0..15) and spills to EUD instead.
    pub const SGPRS_MAX: usize = 32;
}

/// Kyty: HardwareContext.h `VsShaderResource2` (L436). Minimal slice:
/// analysis reads only `user_sgpr`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VsShaderResource2 {
    pub user_sgpr: u8,
}

/// Kyty: HardwareContext.h `VsStageRegisters` (L445). Minimal slice: `rsrc1`
/// omitted (unused by analysis).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VsStageRegisters {
    pub data_addr: u64,
    pub rsrc2: VsShaderResource2,
}

/// Kyty: HardwareContext.h `PsShaderResource2` (L466). Minimal slice.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PsShaderResource2 {
    pub user_sgpr: u8,
}

/// Kyty: HardwareContext.h `PsStageRegisters` (L476). Minimal slice: `rsrc1`
/// omitted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PsStageRegisters {
    pub data_addr: u64,
    pub rsrc2: PsShaderResource2,
    pub chksum: u64,
}

/// Kyty: HardwareContext.h `CsStageRegisters` (L484). Minimal slice: analysis
/// reads `data_addr`, thread counts, `user_sgpr`, `tgid_*_en`,
/// `tidig_comp_cnt`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CsStageRegisters {
    pub data_addr: u64,
    pub chksum: u64,
    pub num_thread_x: u32,
    pub num_thread_y: u32,
    pub num_thread_z: u32,
    pub vgprs: u8,
    pub sgprs: u8,
    pub bulky: u8,
    pub scratch_en: u8,
    pub user_sgpr: u8,
    pub tgid_x_en: u8,
    pub tgid_y_en: u8,
    pub tgid_z_en: u8,
    pub tg_size_en: u8,
    pub tidig_comp_cnt: u8,
    pub lds_size: u8,
}

/// Kyty: HardwareContext.h `EsStageRegisters` (L504).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct EsStageRegisters {
    pub data_addr: u64,
}

/// Kyty: HardwareContext.h `GsShaderResource2` (L525). Minimal slice.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GsShaderResource2 {
    pub user_sgpr: u8,
}

/// Kyty: HardwareContext.h `GsStageRegisters` (L535). Minimal slice: `rsrc1`
/// omitted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GsStageRegisters {
    pub data_addr: u64,
    pub rsrc2: GsShaderResource2,
    pub chksum: u64,
}

/// Kyty: HardwareContext.h `DepthShaderControl` (L366) — DB_SHADER_CONTROL
/// decode (full struct; it is small and PS analysis reads three fields).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthShaderControl {
    pub other_bits: u32,
    pub conservative_z_export_value: u8,
    pub shader_z_behavior: u8,
    pub shader_kill_enable: bool,
    pub shader_z_export_enable: bool,
    pub shader_execute_on_noop: bool,
}

/// Kyty: HardwareContext.h `ShaderRegisters` (L543). Minimal slice: analysis
/// reads the export count (from `m_spiVsOutConfig`, renamed
/// `spi_vs_out_config`), the PS interpolator/input state, and
/// `db_shader_control`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderRegisters {
    pub spi_vs_out_config: u32,
    pub ps_interpolator_settings: [u32; 32],
    pub target_output_mode: [u8; 8],
    pub ps_input_ena: u32,
    pub ps_input_addr: u32,
    pub ps_in_control: u32,
    pub db_shader_control: DepthShaderControl,
}

impl ShaderRegisters {
    /// Kyty: `GetExportCount` (HardwareContext.h L573).
    #[must_use]
    pub const fn get_export_count(&self) -> u32 {
        1 + ((self.spi_vs_out_config >> 1) & 0x1F)
    }
}

/// Kyty: HardwareContext.h `VertexShaderInfo` (L595). Minimal slice:
/// `vs_shader_modifier` omitted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexShaderInfo {
    pub vs_regs: VsStageRegisters,
    pub es_regs: EsStageRegisters,
    pub gs_regs: GsStageRegisters,
    pub vs_embedded_id: u32,
    pub vs_user_sgpr: UserSgprInfo,
    pub gs_user_sgpr: UserSgprInfo,
    pub vs_embedded: bool,
}

/// Kyty: HardwareContext.h `PixelShaderInfo` (L607).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PixelShaderInfo {
    pub ps_regs: PsStageRegisters,
    pub ps_user_sgpr: UserSgprInfo,
    pub ps_embedded_id: u32,
    pub ps_embedded: bool,
}

/// Kyty: HardwareContext.h `ComputeShaderInfo` (L615). Minimal slice:
/// `cs_shader_modifier` omitted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputeShaderInfo {
    pub cs_regs: CsStageRegisters,
    pub cs_user_sgpr: UserSgprInfo,
}

// =========================================================================
// Render-target registers (HardwareContext.h L13-120)
// =========================================================================

/// Kyty: `ColorBase` (L13). `addr` is the register value shifted left by 8.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ColorBase {
    pub addr: u64,
}

/// Kyty: `ColorInfo` (L34). Format/type/order drive the Vulkan format choice.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ColorInfo {
    pub fmask_compression_enable: bool,
    pub fmask_data_compression_disable: bool,
    pub fmask_one_frag_mode: bool,
    pub cmask_fast_clear_enable: bool,
    pub dcc_compression_enable: bool,
    pub neo_mode: bool,
    pub blend_clamp: bool,
    pub blend_bypass: bool,
    pub round_mode: bool,
    pub cmask_tile_mode: u32,
    pub cmask_tile_mode_neo: u32,
    pub format: u32,
    pub channel_type: u32,
    pub channel_order: u32,
}

/// Kyty: `ColorAttrib2` (L62) — **the PS5 render-target extent**.
///
/// Stores width/height minus one; the consumer adds 1. Field order here matches
/// Kyty (`height` before `width`), which is a transposition trap for positional
/// construction — always initialize by name.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ColorAttrib2 {
    pub height: u32,
    pub width: u32,
    pub num_mip_levels: u32,
}

/// Kyty: `ColorAttrib3` (L69).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ColorAttrib3 {
    pub depth: u32,
    pub tile_mode: u32,
    pub dimension: u32,
    pub cmask_pipe_aligned: bool,
    pub dcc_pipe_aligned: bool,
}

/// Kyty: `RenderTarget` (L162). Phase-1 subset — the Gen4 pitch/slice/view
/// members and the DCC/CMASK/FMASK block are not ported.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderTarget {
    pub base: ColorBase,
    pub info: ColorInfo,
    pub attrib2: ColorAttrib2,
    pub attrib3: ColorAttrib3,
}

/// Kyty: `BlendControl` (L200).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BlendControl {
    pub color_srcblend: u8,
    pub color_comb_fcn: u8,
    pub color_destblend: u8,
    pub alpha_srcblend: u8,
    pub alpha_comb_fcn: u8,
    pub alpha_destblend: u8,
    pub separate_alpha_blend: bool,
    pub enable: bool,
}

/// Kyty: `BlendColor` (L211).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct BlendColor {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub alpha: f32,
}

/// Kyty: `ColorControl` (L219). **Non-zero defaults.**
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ColorControl {
    pub mode: u8,
    pub op: u8,
}

impl Default for ColorControl {
    fn default() -> Self {
        Self { mode: 1, op: 0xCC }
    }
}

/// Kyty: `ModeControl` (L225).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ModeControl {
    pub cull_front: bool,
    pub cull_back: bool,
    pub face: bool,
    pub poly_mode: u8,
    pub polymode_front_ptype: u8,
    pub polymode_back_ptype: u8,
    pub poly_offset_front_enable: bool,
    pub poly_offset_back_enable: bool,
    pub vtx_window_offset_enable: bool,
    pub provoking_vtx_last: bool,
    pub persp_corr_dis: bool,
}

/// Kyty: `ScanModeControl` (L239). **`vport_scissor_enable` defaults true.**
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ScanModeControl {
    pub msaa_enable: bool,
    pub vport_scissor_enable: bool,
    pub line_stipple_enable: bool,
}

impl Default for ScanModeControl {
    fn default() -> Self {
        Self {
            msaa_enable: false,
            vport_scissor_enable: true,
            line_stipple_enable: false,
        }
    }
}

/// Kyty: `DepthControl` (L246) — `DB_DEPTH_CONTROL` decode.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthControl {
    pub stencil_enable: bool,
    pub z_enable: bool,
    pub z_write_enable: bool,
    pub depth_bounds_enable: bool,
    pub zfunc: u8,
    pub backface_enable: bool,
    pub stencilfunc: u8,
    pub stencilfunc_bf: u8,
    pub color_writes_on_depth_fail_enable: bool,
    pub color_writes_on_depth_pass_disable: bool,
}

/// Kyty: `DepthZInfo` (HardwareContext.h L154) — `DB_Z_INFO` decode.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthZInfo {
    pub format: u32,
    pub tile_mode_index: u32,
    pub num_samples: u32,
    pub zrange_precision: u32,
    pub tile_surface_enable: bool,
    pub expclear_enabled: bool,
    pub embedded_sample_locations: bool,
    pub partially_resident: bool,
    pub num_mip_levels: u8,
    pub plane_compression: u8,
}

/// Kyty: `DepthStencilInfo` (L168) — `DB_STENCIL_INFO` decode.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilInfo {
    pub format: u32,
    pub tile_mode_index: u32,
    pub tile_split: u32,
    pub expclear_enabled: bool,
    pub tile_stencil_disable: bool,
    pub texture_compatible_stencil: bool,
    pub partially_resident: bool,
}

/// Kyty: `DepthRenderTargetDepthInfo` (L179) — `DB_DEPTH_INFO` decode. Tiling
/// metadata; the offscreen path never reads guest depth memory, so this is
/// tracked for completeness only.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthRenderTargetDepthInfo {
    pub addr5_swizzle_mask: u32,
    pub array_mode: u32,
    pub pipe_config: u32,
    pub bank_width: u32,
    pub bank_height: u32,
    pub macro_tile_aspect: u32,
    pub num_banks: u32,
}

/// Kyty: `DepthDepthView` (L190) — `DB_DEPTH_VIEW` decode.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthDepthView {
    pub slice_start: u32,
    pub slice_max: u32,
    pub current_mip_level: u8,
    pub depth_write_disable: bool,
    pub stencil_write_disable: bool,
}

/// Kyty: `DepthDepthSizeXY` (L199) — `DB_DEPTH_SIZE_XY`, the PS5 depth-surface
/// extent (stores max X/Y, i.e. width/height minus one, like `ColorAttrib2`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthDepthSizeXy {
    pub x_max: u16,
    pub y_max: u16,
}

/// Kyty: `DepthRenderTargetHTileSurface` (L205) — `DB_HTILE_SURFACE` decode.
/// Tracked so the register write is not an "unknown" warn; HTile (depth
/// compression) itself is not implemented.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthRenderTargetHTileSurface {
    pub linear: u32,
    pub full_cache: u32,
    pub htile_uses_preload_win: u32,
    pub preload: u32,
    pub prefetch_width: u32,
    pub prefetch_height: u32,
    pub dst_outside_zero_to_one: u32,
}

/// Kyty: `DepthRenderTarget` (L216). The depth/stencil surface: format, base
/// addresses, and extent. Kyty's `width`/`height` pair (from its fused Gen4
/// packet) is not ported — the PS5 path sizes the surface from `size`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthRenderTarget {
    pub z_info: DepthZInfo,
    pub stencil_info: DepthStencilInfo,
    pub depth_view: DepthDepthView,
    pub size: DepthDepthSizeXy,
    pub depth_info: DepthRenderTargetDepthInfo,
    pub htile_surface: DepthRenderTargetHTileSurface,
    pub z_read_base_addr: u64,
    pub stencil_read_base_addr: u64,
    pub z_write_base_addr: u64,
    pub stencil_write_base_addr: u64,
    pub htile_data_base_addr: u64,
    pub pitch_div8_minus1: u32,
    pub height_div8_minus1: u32,
    pub slice_div64_minus1: u32,
}

/// Kyty: `RenderControl` (L237) — `DB_RENDER_CONTROL` decode.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderControl {
    pub depth_clear_enable: bool,
    pub stencil_clear_enable: bool,
    pub resummarize_enable: bool,
    pub stencil_compress_disable: bool,
    pub depth_compress_disable: bool,
    pub copy_centroid: bool,
    pub copy_sample: u8,
}

/// Kyty: `StencilControl` (L278) — `DB_STENCIL_CONTROL` decode.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct StencilControl {
    pub stencil_fail: u8,
    pub stencil_zpass: u8,
    pub stencil_zfail: u8,
    pub stencil_fail_bf: u8,
    pub stencil_zpass_bf: u8,
    pub stencil_zfail_bf: u8,
}

/// Kyty: `StencilMask` (L288) — `DB_STENCILREFMASK` / `_BF` decode.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct StencilMask {
    pub stencil_testval: u8,
    pub stencil_mask: u8,
    pub stencil_writemask: u8,
    pub stencil_opval: u8,
    pub stencil_testval_bf: u8,
    pub stencil_mask_bf: u8,
    pub stencil_writemask_bf: u8,
    pub stencil_opval_bf: u8,
}

/// Kyty: `Viewport` (L262).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Viewport {
    pub zmin: f32,
    pub zmax: f32,
    pub xscale: f32,
    pub xoffset: f32,
    pub yscale: f32,
    pub yoffset: f32,
    pub zscale: f32,
    pub zoffset: f32,
    pub viewport_scissor_left: i32,
    pub viewport_scissor_top: i32,
    pub viewport_scissor_right: i32,
    pub viewport_scissor_bottom: i32,
    pub viewport_scissor_window_offset_enable: bool,
}

/// Kyty: `ScreenViewport` (L278). Note `viewports` is **15** entries, not 16,
/// and `transform_control` defaults to 1087.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScreenViewport {
    pub viewports: [Viewport; Self::VIEWPORTS_MAX],
    pub transform_control: u32,
    pub screen_scissor_left: i32,
    pub screen_scissor_top: i32,
    pub screen_scissor_right: i32,
    pub screen_scissor_bottom: i32,
    pub generic_scissor_left: i32,
    pub generic_scissor_top: i32,
    pub generic_scissor_right: i32,
    pub generic_scissor_bottom: i32,
    pub generic_scissor_window_offset_enable: bool,
    pub hw_offset_x: u32,
    pub hw_offset_y: u32,
    pub guard_band_horz_clip: f32,
    pub guard_band_vert_clip: f32,
    pub guard_band_horz_discard: f32,
    pub guard_band_vert_discard: f32,
}

impl ScreenViewport {
    pub const VIEWPORTS_MAX: usize = 15;
}

impl Default for ScreenViewport {
    fn default() -> Self {
        Self {
            viewports: [Viewport::default(); Self::VIEWPORTS_MAX],
            transform_control: 1087,
            screen_scissor_left: 0,
            screen_scissor_top: 0,
            screen_scissor_right: 0,
            screen_scissor_bottom: 0,
            generic_scissor_left: 0,
            generic_scissor_top: 0,
            generic_scissor_right: 0,
            generic_scissor_bottom: 0,
            generic_scissor_window_offset_enable: false,
            hw_offset_x: 0,
            hw_offset_y: 0,
            guard_band_horz_clip: 0.0,
            guard_band_vert_clip: 0.0,
            guard_band_horz_discard: 0.0,
            guard_band_vert_discard: 0.0,
        }
    }
}

// =========================================================================
// Register files (HardwareContext.h: class Context / UserConfig / Shader)
// =========================================================================

/// Kyty: `class Context` (L640). The context (CX) register file.
#[derive(Clone, Debug, PartialEq)]
pub struct Context {
    pub render_targets: [RenderTarget; Self::RENDER_TARGETS_MAX],
    pub render_target_mask: u32,
    pub shader_stages: u32,
    pub blend_control: [BlendControl; Self::RENDER_TARGETS_MAX],
    pub blend_color: BlendColor,
    pub color_control: ColorControl,
    pub mode_control: ModeControl,
    pub scan_mode_control: ScanModeControl,
    pub depth_control: DepthControl,
    pub depth_render_target: DepthRenderTarget,
    pub render_control: RenderControl,
    pub stencil_control: StencilControl,
    pub stencil_mask: StencilMask,
    /// `DB_DEPTH_CLEAR` (Kyty: `m_depth_clear_value`, HardwareContext.h L842).
    pub depth_clear_value: f32,
    /// `DB_STENCIL_CLEAR` (Kyty: `m_stencil_clear_value`).
    pub stencil_clear_value: u8,
    /// `DB_DEPTH_BOUNDS_MIN/MAX` — tracked; the bounds test is not implemented.
    pub depth_bounds_min: f32,
    pub depth_bounds_max: f32,
    pub screen_viewport: ScreenViewport,
    pub line_width: f32,
    pub sh_regs: ShaderRegisters,
}

impl Context {
    pub const RENDER_TARGETS_MAX: usize = 8;

    /// Kyty: `Context::Reset` (L648).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            render_targets: [RenderTarget::default(); Self::RENDER_TARGETS_MAX],
            render_target_mask: 0,
            shader_stages: 0,
            blend_control: [BlendControl::default(); Self::RENDER_TARGETS_MAX],
            blend_color: BlendColor::default(),
            color_control: ColorControl::default(),
            mode_control: ModeControl::default(),
            scan_mode_control: ScanModeControl::default(),
            depth_control: DepthControl::default(),
            depth_render_target: DepthRenderTarget::default(),
            render_control: RenderControl::default(),
            stencil_control: StencilControl::default(),
            stencil_mask: StencilMask::default(),
            depth_clear_value: 0.0,
            stencil_clear_value: 0,
            depth_bounds_min: 0.0,
            depth_bounds_max: 0.0,
            screen_viewport: ScreenViewport::default(),
            line_width: 1.0,
            sh_regs: ShaderRegisters::default(),
        }
    }
}

/// Kyty: `GeControl` (HardwareContext.h).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GeControl {
    pub prim_grp_size: u32,
    pub verts_per_subgrp: u32,
    pub prims_per_subgrp: u32,
    pub break_wave_at_eoi: bool,
}

/// Kyty: `GeUserVgprEn` (HardwareContext.h).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GeUserVgprEn {
    pub en: u32,
}

/// Kyty: `class UserConfig` (L960). The user-config (UC) register file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UserConfig {
    pub prim_type: u32,
    pub ge_cntl: GeControl,
    pub ge_user_vgpr_en: GeUserVgprEn,
}

impl UserConfig {
    /// Kyty: `UserConfig::Reset`.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Kyty: `class Shader` (L700). The shader (SH) register file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Shader {
    pub vs: VertexShaderInfo,
    pub ps: PixelShaderInfo,
    pub cs: ComputeShaderInfo,
}

impl Shader {
    /// Kyty: `Shader::Reset`.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Kyty: `Shader::SetVsShaderBase` (L711). Writing a real shader base
    /// **clears** the embedded flag — the two are mutually exclusive.
    pub fn set_vs_shader_base(&mut self, addr: u64) {
        self.vs.vs_regs.data_addr = addr;
        self.vs.vs_embedded = false;
    }

    /// Kyty: `Shader::SetVsEmbedded` (L727).
    pub fn set_vs_embedded(&mut self, id: u32, _shader_modifier: u32) {
        self.vs.vs_embedded_id = id;
        self.vs.vs_embedded = true;
    }

    /// Kyty: `Shader::SetPsShaderBase` (L754). Clears the embedded flag.
    pub fn set_ps_shader_base(&mut self, addr: u64) {
        self.ps.ps_regs.data_addr = addr;
        self.ps.ps_embedded = false;
    }

    /// Kyty: `Shader::SetPsEmbedded` (L769).
    pub fn set_ps_embedded(&mut self, id: u32) {
        self.ps.ps_embedded_id = id;
        self.ps.ps_embedded = true;
    }

    /// Kyty: `Shader::SetCsShader` (HardwareContext.h L960).
    pub fn set_cs_shader(&mut self, regs: CsStageRegisters) {
        self.cs.cs_regs = regs;
    }

    /// Kyty: `Shader::SetEsShaderBase` (L913). Gen5's "gs instead of vs" wave
    /// puts the vertex-stage code behind the ES base; a real bind clears the
    /// embedded flag.
    pub fn set_es_shader_base(&mut self, addr: u64) {
        self.vs.es_regs.data_addr = addr;
        self.vs.vs_embedded = false;
    }

    /// Kyty: `SetGsShaderChksum` (L928) — accumulates across two writes.
    pub fn push_gs_chksum(&mut self, value: u32) {
        self.vs.gs_regs.chksum = (self.vs.gs_regs.chksum << 32) | u64::from(value);
    }

    /// Kyty: `SetGsShaderResource2` (L923). Clears the embedded flag.
    pub fn set_gs_rsrc2_user_sgpr(&mut self, user_sgpr: u8) {
        self.vs.gs_regs.rsrc2.user_sgpr = user_sgpr;
        self.vs.vs_embedded = false;
    }

    /// Kyty: `SetPsShaderResource2` (L944). Clears the embedded flag.
    pub fn set_ps_rsrc2_user_sgpr(&mut self, user_sgpr: u8) {
        self.ps.ps_regs.rsrc2.user_sgpr = user_sgpr;
        self.ps.ps_embedded = false;
    }
}

impl PsStageRegisters {
    /// Kyty: `PsStageRegisters` checksum writes **accumulate** into a u64
    /// across two register writes rather than assigning.
    pub fn push_chksum(&mut self, value: u32) {
        self.chksum = (self.chksum << 32) | u64::from(value);
    }
}

impl CsStageRegisters {
    /// Gen5 checksum writes accumulate across the low/high dwords.
    pub fn push_chksum(&mut self, value: u32) {
        self.chksum = (self.chksum << 32) | u64::from(value);
    }
}

impl UserSgprInfo {
    /// Kyty: user-SGPR setters keep a high-water mark rather than counting
    /// writes — slot 3 alone implies a count of 4.
    pub fn set(&mut self, id: u32, value: u32, type_: UserSgprType) {
        let idx = id as usize;
        if idx >= Self::SGPRS_MAX {
            return;
        }
        self.value[idx] = value;
        self.type_[idx] = type_;
        self.count = self.count.max(id + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gen5 graphics stages have THIRTY-TWO user-SGPR registers, not the 16
    /// Kyty's PS4-era code assumed. Measured on ASTRO.BOT: pixel shaders
    /// declare `rsrc2.user_sgpr` = 20/24/30/32 while only 16 were ever
    /// recorded, so `shader_parse_ps` rejected every one of them. The SH
    /// register map has room: PS user data starts at 0x0C and the next SH
    /// register (VS_0) is 0x4C, GS_0 at 0x8C runs to ES at 0xC8. Compute is
    /// genuinely 16 (COMPUTE_USER_DATA_0..15 = 0x240..0x24F) — its overflow
    /// goes to EUD instead.
    #[test]
    fn user_sgpr_records_gen5_slots_above_sixteen() {
        let mut info = UserSgprInfo::default();
        info.set(20, 0xdead_beef, UserSgprType::Region);
        assert_eq!(
            info.value[20], 0xdead_beef,
            "slot 20 must be recorded, not silently dropped"
        );
        assert_eq!(
            info.count, 21,
            "count is a high-water mark of written slots"
        );

        info.set(31, 0x1234_5678, UserSgprType::Region);
        assert_eq!(info.value[31], 0x1234_5678, "slot 31 is the last Gen5 slot");
        assert_eq!(info.count, 32);
    }

    /// A `#[derive(Default)]` would zero these. Kyty does not.
    #[test]
    fn defaults_match_kyty_nonzero_defaults() {
        let ctx = Context::default();
        assert_eq!(ctx.line_width, 1.0);
        assert_eq!(ctx.color_control.mode, 1);
        assert_eq!(ctx.color_control.op, 0xCC);
        assert!(ctx.scan_mode_control.vport_scissor_enable);
        assert_eq!(ctx.screen_viewport.transform_control, 1087);
    }

    #[test]
    fn shader_base_write_clears_embedded_flag() {
        let mut sh = Shader::default();
        sh.set_vs_embedded(0, 0);
        assert!(sh.vs.vs_embedded);
        sh.set_vs_shader_base(0x1000);
        assert!(
            !sh.vs.vs_embedded,
            "a real base write retires the embedded VS"
        );

        sh.set_ps_embedded(0);
        assert!(sh.ps.ps_embedded);
        sh.set_ps_shader_base(0x2000);
        assert!(!sh.ps.ps_embedded);
    }

    #[test]
    fn chksum_accumulates_two_writes() {
        let mut ps = PsStageRegisters::default();
        ps.push_chksum(0x0000_AAAA);
        ps.push_chksum(0x0000_BBBB);
        assert_eq!(ps.chksum, 0x0000_AAAA_0000_BBBB);
    }

    #[test]
    fn user_sgpr_count_is_a_high_water_mark() {
        let mut sgpr = UserSgprInfo::default();
        sgpr.set(3, 0xDEAD, UserSgprType::Vsharp);
        assert_eq!(sgpr.count, 4, "slot 3 written => count 4");
        assert_eq!(sgpr.value[3], 0xDEAD);
        assert_eq!(sgpr.value[0], 0, "lower slots untouched");
        sgpr.set(1, 0xBEEF, UserSgprType::Region);
        assert_eq!(sgpr.count, 4, "a lower write must not shrink the mark");
    }

    #[test]
    fn reset_restores_nonzero_defaults() {
        let mut ctx = Context::default();
        ctx.line_width = 8.0;
        ctx.render_target_mask = 0xF;
        ctx.reset();
        assert_eq!(ctx.line_width, 1.0);
        assert_eq!(ctx.render_target_mask, 0);
    }

    /// `ColorAttrib2` lists height before width; a positional literal would
    /// transpose the render target's dimensions.
    #[test]
    fn color_attrib2_field_order_is_height_then_width() {
        let a = ColorAttrib2 {
            width: 95,
            height: 47,
            num_mip_levels: 0,
        };
        assert_eq!(a.width, 95);
        assert_eq!(a.height, 47);
    }
}
