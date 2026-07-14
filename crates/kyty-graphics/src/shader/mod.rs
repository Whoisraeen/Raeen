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

pub mod parse;
pub mod types;

pub use parse::{ShaderParseError, operand_parse, shader_parse};
pub use types::{
    ShaderCode, ShaderConstant, ShaderControlFlowBlock, ShaderInstruction, ShaderInstructionType,
    ShaderInstructionTypeFormat, ShaderLabel, ShaderOperand, ShaderOperandType, ShaderType,
    shader_instruction_format,
};
