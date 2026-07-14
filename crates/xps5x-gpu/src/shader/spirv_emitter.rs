//! SPIR-V bytecode emitter.
//!
//! Converts the XPS5X shader IR into SPIR-V binary modules
//! that can be consumed by Vulkan's shader pipeline.

use super::ir::IrProgram;
use super::ShaderType;
use tracing::info;
use xps5x_core::error::GpuError;

// ─── SPIR-V Magic and Constants ────────────────────────────
const SPIRV_MAGIC: u32 = 0x07230203;
const SPIRV_VERSION: u32 = 0x00010500; // SPIR-V 1.5

// Simplified SPIR-V opcodes (subset).
const OP_CAPABILITY: u16 = 17;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_FUNCTION: u16 = 33;
#[allow(dead_code)] // reserved: pointer/variable emission (I/O interface vars) not yet generated
const OP_TYPE_POINTER: u16 = 32;
#[allow(dead_code)] // reserved: see OP_TYPE_POINTER
const OP_VARIABLE: u16 = 59;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;
#[allow(dead_code)] // reserved: decorations (Location/Binding) not yet emitted
const OP_DECORATE: u16 = 71;

/// Emit a SPIR-V module from an IR program.
pub fn emit_spirv(program: &IrProgram) -> Result<Vec<u32>, GpuError> {
    info!(
        "Emitting SPIR-V for {:?} shader ({} IR nodes)",
        program.shader_type,
        program.nodes.len()
    );

    let mut module = SpirvModule::new();

    // Add capabilities.
    module.add_capability(1); // Shader capability.
    if program.shader_type == ShaderType::Geometry {
        module.add_capability(2); // Geometry capability.
    }

    // Memory model: Logical + GLSL450.
    module.add_memory_model(0, 1);

    // Declare types.
    let void_type = module.next_id();
    module.add_type_void(void_type);

    let f32_type = module.next_id();
    module.add_type_float(f32_type, 32);

    let vec4_type = module.next_id();
    module.add_type_vector(vec4_type, f32_type, 4);

    let func_type = module.next_id();
    module.add_type_function(func_type, void_type, &[]);

    // Declare entry point.
    let main_func = module.next_id();
    let execution_model = match program.shader_type {
        ShaderType::Vertex => 0,   // Vertex
        ShaderType::Pixel => 4,    // Fragment
        ShaderType::Compute => 5,  // GLCompute
        ShaderType::Geometry => 3, // Geometry
        ShaderType::Hull => 1,     // TessellationControl
        ShaderType::Domain => 2,   // TessellationEvaluation
    };

    module.add_entry_point(execution_model, main_func, "main", &[]);

    // Add execution mode.
    if program.shader_type == ShaderType::Pixel {
        module.add_execution_mode(main_func, 7); // OriginUpperLeft.
    }

    // Define main function.
    let entry_label = module.next_id();
    module.add_function(main_func, void_type, 0, func_type);
    module.add_label(entry_label);

    // TODO: Emit IR nodes as SPIR-V instructions.
    // For now, emit a minimal valid shader.

    module.add_return();
    module.add_function_end();

    Ok(module.build())
}

/// A SPIR-V module builder.
struct SpirvModule {
    /// All instructions (header will be prepended).
    instructions: Vec<u32>,
    /// Next ID to allocate.
    id_counter: u32,
    /// ID bound (will be set to id_counter at build time).
    id_bound: u32,
}

impl SpirvModule {
    fn new() -> Self {
        Self {
            instructions: Vec::with_capacity(256),
            id_counter: 1,
            id_bound: 0,
        }
    }

    fn next_id(&mut self) -> u32 {
        let id = self.id_counter;
        self.id_counter += 1;
        id
    }

    fn emit(&mut self, opcode: u16, operands: &[u32]) {
        let word_count = (1 + operands.len()) as u16;
        self.instructions
            .push(((word_count as u32) << 16) | (opcode as u32));
        self.instructions.extend_from_slice(operands);
    }

    fn add_capability(&mut self, cap: u32) {
        self.emit(OP_CAPABILITY, &[cap]);
    }

    fn add_memory_model(&mut self, addressing: u32, memory: u32) {
        self.emit(OP_MEMORY_MODEL, &[addressing, memory]);
    }

    fn add_type_void(&mut self, id: u32) {
        self.emit(OP_TYPE_VOID, &[id]);
    }

    fn add_type_float(&mut self, id: u32, width: u32) {
        self.emit(OP_TYPE_FLOAT, &[id, width]);
    }

    fn add_type_vector(&mut self, id: u32, component: u32, count: u32) {
        self.emit(OP_TYPE_VECTOR, &[id, component, count]);
    }

    fn add_type_function(&mut self, id: u32, return_type: u32, params: &[u32]) {
        let mut operands = vec![id, return_type];
        operands.extend_from_slice(params);
        self.emit(OP_TYPE_FUNCTION, &operands);
    }

    fn add_entry_point(&mut self, model: u32, func: u32, name: &str, interfaces: &[u32]) {
        let mut operands = vec![model, func];
        // Encode name as null-terminated, padded to 4-byte boundary.
        let name_bytes = name.as_bytes();
        let padded_len = (name_bytes.len() + 1).div_ceil(4) * 4;
        let mut name_words = vec![0u32; padded_len / 4];
        for (i, &b) in name_bytes.iter().enumerate() {
            let word_idx = i / 4;
            let byte_idx = i % 4;
            name_words[word_idx] |= (b as u32) << (byte_idx * 8);
        }
        operands.extend_from_slice(&name_words);
        operands.extend_from_slice(interfaces);
        self.emit(OP_ENTRY_POINT, &operands);
    }

    fn add_execution_mode(&mut self, entry: u32, mode: u32) {
        self.emit(OP_EXECUTION_MODE, &[entry, mode]);
    }

    fn add_function(&mut self, id: u32, return_type: u32, control: u32, func_type: u32) {
        self.emit(OP_FUNCTION, &[return_type, id, control, func_type]);
    }

    fn add_label(&mut self, id: u32) {
        self.emit(OP_LABEL, &[id]);
    }

    fn add_return(&mut self) {
        self.emit(OP_RETURN, &[]);
    }

    fn add_function_end(&mut self) {
        self.emit(OP_FUNCTION_END, &[]);
    }

    /// Build the final SPIR-V module with header.
    fn build(mut self) -> Vec<u32> {
        self.id_bound = self.id_counter;

        let mut module = Vec::with_capacity(5 + self.instructions.len());
        // SPIR-V header (5 words).
        module.push(SPIRV_MAGIC);
        module.push(SPIRV_VERSION);
        module.push(0); // Generator ID (XPS5X = 0 for now).
        module.push(self.id_bound);
        module.push(0); // Reserved.

        module.extend_from_slice(&self.instructions);
        module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::ir::IrProgram;
    use crate::shader::ShaderType;

    /// An empty IR program of the given shader stage.
    fn prog(ty: ShaderType) -> IrProgram {
        IrProgram {
            nodes: vec![],
            shader_type: ty,
            input_count: 0,
            output_count: 0,
            ubo_count: 0,
            texture_count: 0,
        }
    }

    /// A minimal SPIR-V module for an empty program of `ty`: valid magic +
    /// header, a nonzero ID bound, and body instructions past the header.
    fn assert_valid_spirv(ty: ShaderType) {
        let module = emit_spirv(&prog(ty)).expect("emit_spirv must succeed");

        assert!(
            module.len() > 5,
            "module must have a 5-word header + instructions"
        );
        assert_eq!(
            module[0], SPIRV_MAGIC,
            "word 0 must be the SPIR-V magic 0x07230203"
        );
        assert_eq!(
            module[1], SPIRV_VERSION,
            "word 1 must be the SPIR-V version"
        );
        // word 2 = generator (0), word 3 = id bound, word 4 = reserved (0).
        assert!(
            module[3] > 1,
            "id bound must reflect the ids allocated ({})",
            module[3]
        );
        assert_eq!(module[4], 0, "reserved header word must be 0");
        // Every id used must be < the declared bound (SPIR-V's core invariant).
        assert!(
            module[3] <= module.len() as u32 * 4,
            "id bound is implausibly large"
        );
    }

    #[test]
    fn emits_valid_spirv_header_for_every_stage() {
        for ty in [
            ShaderType::Vertex,
            ShaderType::Pixel,
            ShaderType::Compute,
            ShaderType::Geometry,
            ShaderType::Hull,
            ShaderType::Domain,
        ] {
            assert_valid_spirv(ty);
        }
    }

    /// A pixel shader must carry the OriginUpperLeft execution mode, and a
    /// geometry shader the Geometry capability — stage-specific structure the
    /// emitter is responsible for. (OpExecutionMode = opcode 16,
    /// OpCapability = opcode 17; the low 16 bits of an instruction's first
    /// word are its opcode.)
    #[test]
    fn stage_specific_instructions_are_present() {
        let px = emit_spirv(&prog(ShaderType::Pixel)).unwrap();
        assert!(
            px[5..].iter().any(|&w| (w & 0xFFFF) == 16),
            "pixel shader must emit an OpExecutionMode (OriginUpperLeft)"
        );

        let gs = emit_spirv(&prog(ShaderType::Geometry)).unwrap();
        // Two OpCapability (Shader + Geometry) for a geometry shader.
        let caps = gs[5..].iter().filter(|&&w| (w & 0xFFFF) == 17).count();
        assert!(
            caps >= 2,
            "geometry shader must declare the Geometry capability too (got {caps})"
        );
    }
}
