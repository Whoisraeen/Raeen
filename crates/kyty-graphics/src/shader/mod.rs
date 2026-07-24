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
//! - [`hw_regs`]: re-export of [`crate::hw_regs`]. `HardwareContext.h` is
//!   generation-agnostic and is also read by the PM4 command processor
//!   ([`crate::run`]), so the register model lives at crate level; this alias
//!   keeps the historical `shader::hw_regs::` paths resolving.
//! - [`analysis`]: `Shader.cpp` — binary info, usage slots, resource
//!   extraction, fetch recovery, input infos, cache ids, parse wrappers.

pub mod analysis;
pub mod parse;
pub mod recompile;
pub mod resources;
pub mod spirv;
pub mod types;

pub use crate::hw_regs;
pub use crate::hw_regs::{
    ComputeShaderInfo, CsStageRegisters, DepthShaderControl, EsStageRegisters, GsShaderResource2,
    GsStageRegisters, PixelShaderInfo, PsShaderResource2, PsStageRegisters, ShaderRegisters,
    UserSgprInfo, UserSgprType, VertexShaderInfo, VsShaderResource2, VsStageRegisters,
};
pub use analysis::{
    EudView, ShaderAnalysisError, ShaderBinaryInfo, ShaderMap, ShaderMemory, ShaderParsedUsage,
    ShaderUsageInfo, ShaderUsageSlot, get_binary_info, get_usage_slots,
    shader_calc_binding_indices, shader_detect_buffers, shader_detect_embedded_buffer_fetch,
    shader_detect_embedded_constant_loads, shader_get_id_cs, shader_get_id_ps, shader_get_id_vs,
    shader_get_input_info_cs, shader_get_input_info_ps, shader_get_input_info_vs,
    shader_parse_attrib, shader_parse_cs, shader_parse_fetch, shader_parse_ps, shader_parse_usage,
    shader_parse_usage2, shader_parse_vs, shader_synthesize_default_sampler,
    shader_synthesize_gds_pointer, shader_synthesize_placeholder_sampled_texture,
    shader_synthesize_placeholder_storage_texture,
};
pub use parse::{ShaderParseError, operand_parse, shader_parse};
pub use recompile::{
    InstRecompileFn, RecompileFn, RecompilerFunc, SccCheck, get_scc_check, recomp_func,
    recomp_func_table, shader_recompile_cs, shader_recompile_ps, shader_recompile_vs, spirv_run,
};
pub use resources::{
    ShaderBindResources, ShaderBufferResource, ShaderComputeInputInfo, ShaderEmbeddedBufferFetch,
    ShaderEmbeddedBufferFetches, ShaderEmbeddedConstantLoad, ShaderEmbeddedConstantLoads,
    ShaderEudRawResources, ShaderGlobalMemResources, ShaderId, ShaderMappedData,
    ShaderPixelInputInfo, ShaderSamplerResource, ShaderSemantic, ShaderSharp,
    ShaderTextureResource, ShaderUserData, ShaderVertexInputBuffer, ShaderVertexInputInfo,
};
pub use spirv::{
    PUSH_CONSTANT_SPILL_THRESHOLD, SampledClass, SampledDim, ShaderRecompileError, Spirv,
    SpirvType, SpirvValue, sampled_key_ordinal, shader_detect_eud_raw_window,
    shader_detect_flat_global_window, shader_push_constant_spill_binding, spirv_generate_source,
    spirv_get_embedded_ps, spirv_get_embedded_vs,
};
pub use types::{
    DppCtrl, DppMode, ShaderCode, ShaderConstant, ShaderControlFlowBlock, ShaderDebugPrintf,
    ShaderInstruction, ShaderInstructionType, ShaderInstructionTypeFormat, ShaderLabel,
    ShaderOperand, ShaderOperandType, ShaderType, shader_instruction_format,
};
