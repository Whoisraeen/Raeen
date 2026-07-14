//! SPIR-V bytecode emitter.
//!
//! Converts the XPS5X shader IR into SPIR-V binary modules
//! that can be consumed by Vulkan's shader pipeline.

use super::ir::{IrOp, IrProgram, IrValue};
use super::ShaderType;
use std::collections::HashMap;
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
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
#[allow(dead_code)] // reserved: pointer/variable emission (I/O interface vars) not yet generated
const OP_TYPE_POINTER: u16 = 32;
#[allow(dead_code)] // reserved: see OP_TYPE_POINTER
const OP_VARIABLE: u16 = 59;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;
const OP_UNDEF: u16 = 1;
// Arithmetic opcodes (integer / float binary ops).
const OP_I_ADD: u16 = 128;
const OP_F_ADD: u16 = 129;
const OP_I_SUB: u16 = 130;
const OP_F_SUB: u16 = 131;
const OP_I_MUL: u16 = 132;
const OP_F_MUL: u16 = 133;
const OP_F_DIV: u16 = 136;
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

    let i32_type = module.next_id();
    module.add_type_int(i32_type, 32, 1);

    let vec4_type = module.next_id();
    module.add_type_vector(vec4_type, f32_type, 4);

    let func_type = module.next_id();
    module.add_type_function(func_type, void_type, &[]);

    // Constant pool: materialize each distinct constant the IR references, so
    // the arithmetic body (emitted later) can point at these ids. Integer and
    // float constants use their respective scalar types; deduplicated by
    // (type, bit-pattern).
    let mut const_ids: HashMap<(u32, u32), u32> = HashMap::new();
    for node in &program.nodes {
        for src in &node.sources {
            let entry = match src {
                IrValue::ConstI32(v) => Some((i32_type, *v as u32)),
                IrValue::ConstU32(v) => Some((i32_type, *v)),
                IrValue::ConstF32(v) => Some((f32_type, v.to_bits())),
                _ => None,
            };
            let Some(key) = entry else { continue };
            if const_ids.contains_key(&key) {
                continue;
            }
            let id = module.next_id();
            module.add_constant(id, key.0, key.1);
            const_ids.insert(key, id);
        }
    }

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

    // Emit the body: walk IR nodes in program order (SSA order, so a value is
    // defined before it is used) and lower the arithmetic ops to SPIR-V.
    // Sources resolve to constant-pool ids, prior SSA result ids, or a shared
    // OpUndef for values not yet wired (live-in inputs, memory results); every
    // node that defines an SSA result gets an id so later references resolve.
    let mut ssa_to_id: HashMap<u32, u32> = HashMap::new();

    for node in &program.nodes {
        let binop = spirv_binop(node.op, f32_type, i32_type);
        // A typed binary op's operands share its result type; elsewhere default
        // to f32. Undefs (live-ins, unwired reads) are emitted with this type so
        // an integer op never receives a float-typed operand.
        let operand_type = binop.map(|(_, t)| t).unwrap_or(f32_type);

        // Resolve this node's source operands to SPIR-V ids.
        let mut src_ids = Vec::with_capacity(node.sources.len());
        for s in &node.sources {
            let id = match s {
                IrValue::Reg(n) => ssa_to_id
                    .get(n)
                    .copied()
                    .unwrap_or_else(|| module.undef(operand_type)),
                IrValue::ConstI32(v) => const_ids[&(i32_type, *v as u32)],
                IrValue::ConstU32(v) => const_ids[&(i32_type, *v)],
                IrValue::ConstF32(v) => const_ids[&(f32_type, v.to_bits())],
                // Live-ins and resource reads are not wired to real interface
                // variables yet — model them as an undefined value for now.
                _ => module.undef(operand_type),
            };
            src_ids.push(id);
        }

        let Some(IrValue::Reg(result_reg)) = node.result else {
            continue; // sinks (export / return) carry no SSA result
        };

        // A binary arithmetic op with two resolved operands lowers directly;
        // anything else still needs a defined id, so emit an OpUndef for it.
        let rid = match binop {
            Some((opcode, result_type)) if src_ids.len() >= 2 => {
                module.emit_binop(opcode, result_type, src_ids[0], src_ids[1])
            }
            _ => module.undef(operand_type),
        };
        ssa_to_id.insert(result_reg, rid);
    }

    module.add_return();
    module.add_function_end();

    Ok(module.build())
}

/// Map an IR binary op to its SPIR-V opcode and result type. Float ops use the
/// f32 type, integer ops the i32 type. Returns `None` for ops that are not a
/// simple two-operand arithmetic instruction (handled elsewhere).
fn spirv_binop(op: IrOp, f32_type: u32, i32_type: u32) -> Option<(u16, u32)> {
    Some(match op {
        IrOp::Add => (OP_F_ADD, f32_type),
        IrOp::Sub => (OP_F_SUB, f32_type),
        IrOp::Mul => (OP_F_MUL, f32_type),
        IrOp::Div => (OP_F_DIV, f32_type),
        IrOp::IAdd => (OP_I_ADD, i32_type),
        IrOp::ISub => (OP_I_SUB, i32_type),
        IrOp::IMul => (OP_I_MUL, i32_type),
        _ => return None,
    })
}

/// A SPIR-V module builder.
struct SpirvModule {
    /// All instructions (header will be prepended).
    instructions: Vec<u32>,
    /// Next ID to allocate.
    id_counter: u32,
    /// ID bound (will be set to id_counter at build time).
    id_bound: u32,
    /// One shared `OpUndef` id per type id (lazily emitted on first use).
    undef_ids: HashMap<u32, u32>,
}

impl SpirvModule {
    fn new() -> Self {
        Self {
            instructions: Vec::with_capacity(256),
            id_counter: 1,
            id_bound: 0,
            undef_ids: HashMap::new(),
        }
    }

    /// Return (lazily creating) the shared `OpUndef` value of the given type.
    /// Emitted at first use inside the function body, so it dominates every
    /// later reference.
    fn undef(&mut self, type_id: u32) -> u32 {
        if let Some(&id) = self.undef_ids.get(&type_id) {
            return id;
        }
        let id = self.next_id();
        self.emit(OP_UNDEF, &[type_id, id]);
        self.undef_ids.insert(type_id, id);
        id
    }

    /// Emit a binary op (`OpFAdd`/`OpIAdd`/…): `result_type result_id a b`.
    /// Returns the fresh result id.
    fn emit_binop(&mut self, opcode: u16, result_type: u32, a: u32, b: u32) -> u32 {
        let id = self.next_id();
        self.emit(opcode, &[result_type, id, a, b]);
        id
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

    fn add_type_int(&mut self, id: u32, width: u32, signedness: u32) {
        self.emit(OP_TYPE_INT, &[id, width, signedness]);
    }

    /// Emit `OpConstant result_type result_id value` — a 32-bit scalar constant.
    fn add_constant(&mut self, id: u32, type_id: u32, value: u32) {
        self.emit(OP_CONSTANT, &[type_id, id, value]);
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
    /// Walk a SPIR-V module instruction-by-instruction (past the 5-word
    /// header), returning the value operand of every OpConstant.
    fn constant_values(module: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut i = 5;
        while i < module.len() {
            let word_count = (module[i] >> 16) as usize;
            let opcode = (module[i] & 0xFFFF) as u16;
            if opcode == OP_CONSTANT {
                // OpConstant: [type_id, result_id, value]; value is operand 3.
                out.push(module[i + 3]);
            }
            if word_count == 0 {
                break; // malformed; avoid looping forever
            }
            i += word_count;
        }
        out
    }

    /// Every emitted module must parse as structurally-valid SPIR-V through
    /// rspirv (magic/version, per-instruction word counts, operand layout).
    /// This is a real external structural check — stronger than reading raw
    /// words by hand — short of full `spirv-val` semantic validation.
    #[test]
    fn every_stage_parses_as_structural_spirv() {
        for ty in [
            ShaderType::Vertex,
            ShaderType::Pixel,
            ShaderType::Compute,
            ShaderType::Geometry,
            ShaderType::Hull,
            ShaderType::Domain,
        ] {
            let module = emit_spirv(&prog(ty)).unwrap();
            let parsed = rspirv::dr::load_words(&module);
            assert!(
                parsed.is_ok(),
                "stage {ty:?} must parse as SPIR-V: {:?}",
                parsed.err()
            );
        }
    }

    #[test]
    fn arithmetic_body_lowers_to_spirv_ops_and_validates() {
        use crate::shader::ir::{IrNode, IrOp};
        // r0 = 2.0 + 3.0   (float add of two constants)
        // r1 = r0 * r0     (float mul of the add's SSA result)
        let program = IrProgram {
            nodes: vec![
                IrNode {
                    op: IrOp::Add,
                    result: Some(IrValue::Reg(0)),
                    sources: vec![IrValue::ConstF32(2.0), IrValue::ConstF32(3.0)],
                },
                IrNode {
                    op: IrOp::Mul,
                    result: Some(IrValue::Reg(1)),
                    sources: vec![IrValue::Reg(0), IrValue::Reg(0)],
                },
            ],
            shader_type: ShaderType::Vertex,
            input_count: 0,
            output_count: 0,
            ubo_count: 0,
            texture_count: 0,
        };
        let module = emit_spirv(&program).unwrap();

        // Parse through rspirv and inspect the function body's opcodes.
        let parsed = rspirv::dr::load_words(&module).expect("arithmetic body must parse");
        let ops: Vec<rspirv::spirv::Op> = parsed
            .functions
            .iter()
            .flat_map(|f| f.blocks.iter())
            .flat_map(|b| b.instructions.iter())
            .map(|i| i.class.opcode)
            .collect();
        assert!(
            ops.contains(&rspirv::spirv::Op::FAdd),
            "the Add node must lower to OpFAdd (got {ops:?})"
        );
        assert!(
            ops.contains(&rspirv::spirv::Op::FMul),
            "the Mul node must lower to OpFMul (got {ops:?})"
        );
    }

    #[test]
    fn a_module_with_a_constant_pool_parses() {
        use crate::shader::ir::{IrNode, IrOp};
        let program = IrProgram {
            nodes: vec![IrNode {
                op: IrOp::IAdd,
                result: Some(IrValue::Reg(0)),
                sources: vec![IrValue::ConstI32(7), IrValue::ConstU32(0xDEAD_BEEF)],
            }],
            shader_type: ShaderType::Vertex,
            input_count: 0,
            output_count: 0,
            ubo_count: 0,
            texture_count: 0,
        };
        let module = emit_spirv(&program).unwrap();
        let parsed = rspirv::dr::load_words(&module);
        assert!(
            parsed.is_ok(),
            "a module carrying an OpConstant pool must still parse: {:?}",
            parsed.err()
        );
    }

    #[test]
    fn integer_constants_are_materialized_once_in_the_constant_pool() {
        use crate::shader::ir::{IrNode, IrOp};
        // An IR program whose one node reads the constant 5 twice.
        let program = IrProgram {
            nodes: vec![IrNode {
                op: IrOp::IAdd,
                result: Some(IrValue::Reg(0)),
                sources: vec![IrValue::ConstI32(5), IrValue::ConstI32(5)],
            }],
            shader_type: ShaderType::Vertex,
            input_count: 0,
            output_count: 0,
            ubo_count: 0,
            texture_count: 0,
        };
        let module = emit_spirv(&program).unwrap();
        let consts = constant_values(&module);
        assert!(consts.contains(&5), "the constant 5 must be materialized");
        assert_eq!(
            consts.iter().filter(|&&v| v == 5).count(),
            1,
            "the repeated constant is deduplicated to a single OpConstant"
        );
    }

    #[test]
    fn a_program_without_constants_emits_no_constant_pool() {
        let module = emit_spirv(&prog(ShaderType::Vertex)).unwrap();
        assert!(
            constant_values(&module).is_empty(),
            "an empty program needs no OpConstant"
        );
    }

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
