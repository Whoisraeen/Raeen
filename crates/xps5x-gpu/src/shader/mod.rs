//! Shader recompiler — RDNA2 ISA → SPIR-V.
//!
//! PS5 games ship with precompiled GPU shaders in AMD's RDNA2 ISA
//! (Instruction Set Architecture). This module decodes those binary
//! shaders and recompiles them to SPIR-V for the host Vulkan driver.

pub mod cache;
pub mod gcn_decoder;
pub mod ir;
pub mod spirv_emitter;

use tracing::{debug, info};

/// Shader type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Vertex,
    Pixel, // Fragment
    Compute,
    Geometry,
    Hull,   // Tessellation Control
    Domain, // Tessellation Evaluation
}

/// A recompiled shader ready for Vulkan pipeline creation.
#[derive(Debug, Clone)]
pub struct RecompiledShader {
    /// Shader type.
    pub shader_type: ShaderType,
    /// SPIR-V bytecode.
    pub spirv: Vec<u32>,
    /// Number of input attributes.
    pub input_count: u32,
    /// Number of output attributes.
    pub output_count: u32,
    /// Number of uniform buffer bindings.
    pub ubo_count: u32,
    /// Number of texture/sampler bindings.
    pub texture_count: u32,
    /// Hash of the original ISA binary (for caching).
    pub isa_hash: u64,
}

/// Entry point for shader recompilation.
pub fn recompile_shader(
    isa_binary: &[u8],
    shader_type: ShaderType,
) -> Result<RecompiledShader, xps5x_core::error::GpuError> {
    info!(
        "Recompiling {:?} shader ({} bytes ISA)",
        shader_type,
        isa_binary.len()
    );

    // Step 1: Decode RDNA2 ISA instructions.
    let instructions = gcn_decoder::decode(isa_binary)?;
    debug!("Decoded {} ISA instructions", instructions.len());

    // Step 2: Lift to IR.
    let ir_program = ir::lift_to_ir(&instructions, shader_type);
    debug!("IR: {} nodes", ir_program.nodes.len());

    // Step 3: Emit SPIR-V.
    let spirv = spirv_emitter::emit_spirv(&ir_program)?;
    debug!("Emitted {} SPIR-V words", spirv.len());

    // Compute hash of original ISA for cache key.
    let isa_hash = compute_hash(isa_binary);

    Ok(RecompiledShader {
        shader_type,
        spirv,
        input_count: ir_program.input_count,
        output_count: ir_program.output_count,
        ubo_count: ir_program.ubo_count,
        texture_count: ir_program.texture_count,
        isa_hash,
    })
}

/// Simple FNV-1a hash for cache keys.
fn compute_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
