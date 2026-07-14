//! Faithful Rust port of Kyty `emulator/src/Graphics` (MIT © 2021 InoriRus).
//!
//! Phase 5 of the Kyty port plan. The shader recompiler lands first:
//! `shader_parse` (GCN instruction decode), `shader` (binary info / usages),
//! `shader_spirv` (SPIR-V generation). PM4 / render / VideoOut follow.

pub mod shader;
pub mod spirv_asm;
