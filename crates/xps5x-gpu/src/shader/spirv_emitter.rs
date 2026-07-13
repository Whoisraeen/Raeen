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
        ShaderType::Vertex => 0,    // Vertex
        ShaderType::Pixel => 4,     // Fragment
        ShaderType::Compute => 5,   // GLCompute
        ShaderType::Geometry => 3,  // Geometry
        ShaderType::Hull => 1,      // TessellationControl
        ShaderType::Domain => 2,    // TessellationEvaluation
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
        self.instructions.push(((word_count as u32) << 16) | (opcode as u32));
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
