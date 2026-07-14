//! Shader recompiler: GCN ISA → SPIR-V, ported from Kyty.
//!
//! Kyty sources: `Graphics/ShaderParse.cpp` (decode), `Graphics/Shader.cpp`
//! (binary info / input usages / resources), `Graphics/ShaderSpirv.cpp`
//! (SPIR-V generation). Module layout mirrors those files 1:1.
//!
//! - [`types`]: data model from `Shader.h` (+ `ShaderCode` debug/block
//!   helpers from `Shader.cpp`).
//! - [`parse`]: `ShaderParse.cpp` — `operand_parse`, the 17 per-family
//!   instruction parsers, and the top-level `shader_parse` walker.
//! - [`resources`]: `Shader.h` L532-1028 — sharps (V#/T#/S#), bind
//!   resources, per-stage input infos, PS5 shader-file header.
//! - [`hw_regs`]: minimal `HardwareContext.h` slice consumed by analysis.
//! - [`analysis`]: `Shader.cpp` — binary info, usage slots, resource
//!   extraction, fetch recovery, input infos, cache ids, parse wrappers.

pub mod analysis;
pub mod hw_regs;
pub mod parse;
pub mod resources;
pub mod types;

pub use analysis::{
    ShaderAnalysisError, ShaderBinaryInfo, ShaderMap, ShaderMemory, ShaderParsedUsage,
    ShaderUsageInfo, ShaderUsageSlot, get_binary_info, get_usage_slots,
    shader_calc_binding_indices, shader_detect_buffers, shader_get_id_cs, shader_get_id_ps,
    shader_get_id_vs, shader_get_input_info_cs, shader_get_input_info_ps, shader_get_input_info_vs,
    shader_parse_attrib, shader_parse_cs, shader_parse_fetch, shader_parse_ps, shader_parse_usage,
    shader_parse_usage2, shader_parse_vs,
};
pub use hw_regs::{
    ComputeShaderInfo, PixelShaderInfo, ShaderRegisters, UserSgprInfo, UserSgprType,
    VertexShaderInfo,
};
pub use parse::{ShaderParseError, operand_parse, shader_parse};
pub use resources::{
    ShaderBindResources, ShaderBufferResource, ShaderComputeInputInfo, ShaderId, ShaderMappedData,
    ShaderPixelInputInfo, ShaderSamplerResource, ShaderSemantic, ShaderSharp,
    ShaderTextureResource, ShaderUserData, ShaderVertexInputBuffer, ShaderVertexInputInfo,
};
pub use types::{
    ShaderCode, ShaderConstant, ShaderControlFlowBlock, ShaderDebugPrintf, ShaderInstruction,
    ShaderInstructionType, ShaderInstructionTypeFormat, ShaderLabel, ShaderOperand,
    ShaderOperandType, ShaderType, shader_instruction_format,
};
