//! Minimal slice of Kyty's hardware-context register model — only the fields
//! read by the shader analysis layer (`analysis.rs`). Ported from Kyty
//! (MIT (c) InoriRus).
//!
//! Kyty source: `emulator/include/Emulator/Graphics/HardwareContext.h`
//! (namespace `HW`). This is NOT a full `HardwareContext` port (that is a
//! later batch); every struct below carries its Kyty anchor and keeps only
//! the members that `Shader.cpp` analysis functions actually touch, plus the
//! `rsrc1`-style siblings are omitted entirely.

/// Kyty: HardwareContext.h `UserSgprType` (L579).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum UserSgprType {
    #[default]
    Unknown,
    Region,
    Vsharp,
}

/// Kyty: HardwareContext.h `UserSgprInfo` (L586) — the 16 user-SGPR values
/// latched from PM4 `SET_SH_REG` writes, plus what kind of data each holds.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct UserSgprInfo {
    pub value: [u32; Self::SGPRS_MAX],
    pub type_: [UserSgprType; Self::SGPRS_MAX],
    pub count: u32,
}

impl UserSgprInfo {
    pub const SGPRS_MAX: usize = 16;
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
    pub num_thread_x: u32,
    pub num_thread_y: u32,
    pub num_thread_z: u32,
    pub user_sgpr: u8,
    pub tgid_x_en: u8,
    pub tgid_y_en: u8,
    pub tgid_z_en: u8,
    pub tidig_comp_cnt: u8,
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
