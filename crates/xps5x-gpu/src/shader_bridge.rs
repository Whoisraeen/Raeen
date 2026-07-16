//! Bridge from `kyty-graphics` into the Vulkan offscreen draw path.
//!
//! M2 acceptance requires SPIR-V that did **not** come from the hand-built
//! modules in [`crate::vulkan::shaders`]. This module produces:
//!
//! - **VS** — Kyty-format SPIR-V assembly assembled by `spirv_asm` (attribute
//!   passthrough for the host NDC vertex buffer).
//! - **FS** — real GCN bytes → parse → analysis → recompile → assemble, with
//!   a solid green MRT0 write matching [`crate::vulkan::shaders::TRIANGLE_COLOR`].

use std::borrow::Cow;

use kyty_graphics::shader::analysis::SHADER_BINARY_INFO_SENTINEL;
use kyty_graphics::shader::hw_regs::PsStageRegisters;
use kyty_graphics::shader::{
    PixelShaderInfo, ShaderMap, ShaderMemory, ShaderPixelInputInfo, ShaderRegisters,
    ShaderVertexInputInfo, shader_get_input_info_ps, shader_parse_ps, shader_recompile_ps,
};
use kyty_graphics::spirv_asm;
use thiserror::Error;
use xps5x_core::error::GpuError;

/// Guest address used for the fixture pixel-shader blob.
const PS_ADDR: u64 = 0x2_0000;

/// `s_endpgm`.
const S_ENDPGM: u32 = 0xBF81_0000;

/// Solid green GCN pixel shader body:
///
/// ```text
/// v_mov_b32 v0, 0              7E000280
/// v_mov_b32 v1, lit(1.0)       7E0202FF 3F800000
/// v_mov_b32 v2, 0              7E040280
/// v_mov_b32 v3, lit(1.0)       7E0602FF 3F800000
/// exp mrt0 v0..v3 vm done      F800180F 03020100
/// s_endpgm                     BF810000
/// ```
const PS_BODY_SOLID_GREEN: &[u32] = &[
    0x7E00_0280,
    0x7E02_02FF,
    0x3F80_0000,
    0x7E04_0280,
    0x7E06_02FF,
    0x3F80_0000,
    0xF800_180F,
    0x0302_0100,
    S_ENDPGM,
];

/// Attribute-passthrough vertex shader in Kyty SPIR-V assembly text.
///
/// Matches the interface of the previous hand-built triangle VS: `location=0`
/// `vec4` in → `BuiltIn Position` out.
const VS_PASSTHROUGH_ASM: &str = "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main \"main\" %inPos %outPosition
OpDecorate %outPosition BuiltIn Position
OpDecorate %inPos Location 0
%void = OpTypeVoid
%fn_type = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%inPos = OpVariable %_ptr_Input_v4float Input
%outPosition = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %fn_type
%entry = OpLabel
%loaded = OpLoad %v4float %inPos
OpStore %outPosition %loaded
OpReturn
OpFunctionEnd
";

#[derive(Debug, Error)]
pub enum ShaderBridgeError {
    #[error("spirv_asm failed: {0}")]
    Assemble(String),
    #[error("GCN→SPIR-V recompile failed: {0}")]
    Recompile(String),
}

impl From<ShaderBridgeError> for GpuError {
    fn from(e: ShaderBridgeError) -> Self {
        GpuError::ShaderCompilationFailed(e.to_string())
    }
}

struct FixtureMem {
    regions: Vec<(u64, Vec<u32>)>,
}

impl ShaderMemory for FixtureMem {
    fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
        if addr == 0 {
            return None;
        }
        for (base, data) in &self.regions {
            let end = base + data.len() as u64 * 4;
            if addr >= *base && addr < end && (addr - base).is_multiple_of(4) {
                return Some(Cow::Borrowed(&data[((addr - base) / 4) as usize..]));
            }
        }
        None
    }
}

/// Build a shader blob with the real Kyty `0xBEEB03FF` binary-info trailer.
fn build_shader_blob(body: &[u32], hash0: u32, crc32: u32) -> Vec<u32> {
    let mut v = vec![SHADER_BINARY_INFO_SENTINEL, 0];
    v.extend_from_slice(body);
    if (v.len() + 1) % 2 != 0 {
        v.push(0);
    }
    v.push(0); // usage masks
    let info_dw = v.len();
    v[1] = (info_dw / 2 - 1) as u32;
    v.push(u32::from_le_bytes(*b"OrbS"));
    v.push(u32::from_le_bytes([b'h', b'd', b'r', 0x42]));
    v.push((body.len() as u32 * 4) << 8);
    v.push(1); // chunk_usage_base_offset_dw = 1, num_slots = 0
    v.push(hash0);
    v.push(0x1111_2222);
    v.push(crc32);
    v
}

/// Assemble the attribute-passthrough VS through `kyty-graphics::spirv_asm`.
pub fn m2_vertex_spirv() -> Result<Vec<u32>, ShaderBridgeError> {
    spirv_asm::assemble(VS_PASSTHROUGH_ASM).map_err(|e| ShaderBridgeError::Assemble(e.to_string()))
}

/// Recompile the solid-green GCN PS through the full Kyty chain.
pub fn m2_fragment_spirv() -> Result<Vec<u32>, ShaderBridgeError> {
    let mem = FixtureMem {
        regions: vec![(
            PS_ADDR,
            build_shader_blob(PS_BODY_SOLID_GREEN, 0xAAAA_00E2, 0xBBBB_00E2),
        )],
    };
    let shader_map = ShaderMap::new();
    let mut sh = ShaderRegisters {
        ps_in_control: 0,
        ..Default::default()
    };
    // Non-compressed MRT0 (`Mrt0Vsrc0Vsrc1Vsrc2Vsrc3VmDone`) requires mode 9.
    sh.target_output_mode[0] = 9;

    let regs = PixelShaderInfo {
        ps_regs: PsStageRegisters {
            data_addr: PS_ADDR,
            ..Default::default()
        },
        ..Default::default()
    };

    let code = shader_parse_ps(&regs, &sh, &mem, false)
        .map_err(|e| ShaderBridgeError::Recompile(format!("shader_parse_ps: {e}")))?;

    let vs_input = ShaderVertexInputInfo {
        export_count: 1,
        ..Default::default()
    };
    let mut ps_input = ShaderPixelInputInfo::default();
    shader_get_input_info_ps(
        &regs,
        &sh,
        &vs_input,
        &mem,
        &shader_map,
        false,
        &mut ps_input,
    )
    .map_err(|e| ShaderBridgeError::Recompile(format!("shader_get_input_info_ps: {e}")))?;
    // Analysis may leave mode 0; force the mode the EXP encoding needs.
    ps_input.target_output_mode[0] = 9;

    shader_recompile_ps(&code, &ps_input)
        .map_err(|e| ShaderBridgeError::Recompile(format!("shader_recompile_ps: {e}")))
}

/// VS + FS SPIR-V words for the M2 triangle path.
pub fn m2_triangle_spirv() -> Result<(Vec<u32>, Vec<u32>), ShaderBridgeError> {
    Ok((m2_vertex_spirv()?, m2_fragment_spirv()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m2_spirv_modules_have_magic_and_body() {
        let (vs, fs) = m2_triangle_spirv().expect("bridge must produce SPIR-V");
        assert_eq!(vs[0], 0x0723_0203, "VS SPIR-V magic");
        assert_eq!(fs[0], 0x0723_0203, "FS SPIR-V magic");
        assert!(vs.len() > 5, "VS must have a body");
        assert!(fs.len() > 5, "FS must have a body");
    }
}
