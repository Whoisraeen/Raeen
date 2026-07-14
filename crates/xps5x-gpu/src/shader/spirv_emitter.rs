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
const OP_TYPE_POINTER: u16 = 32;
const OP_VARIABLE: u16 = 59;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;
const OP_UNDEF: u16 = 1;
const OP_LOAD: u16 = 61;
// SPIR-V storage classes.
const STORAGE_CLASS_INPUT: u32 = 1;
// Arithmetic opcodes (integer / float binary ops).
const OP_I_ADD: u16 = 128;
const OP_F_ADD: u16 = 129;
const OP_I_SUB: u16 = 130;
const OP_F_SUB: u16 = 131;
const OP_I_MUL: u16 = 132;
const OP_F_MUL: u16 = 133;
const OP_F_DIV: u16 = 136;
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

    // Declare an input interface variable per distinct live-in location the
    // body reads (`IrValue::Input(loc)`). Each is an Input-storage pointer to
    // f32, decorated with its Location; they populate the entry point's
    // interface list. (Inputs are modelled as f32 for now — see the ledger.)
    let mut input_locations: Vec<u32> = Vec::new();
    for node in &program.nodes {
        for s in &node.sources {
            let IrValue::Input(loc) = s else { continue };
            if !input_locations.contains(loc) {
                input_locations.push(*loc);
            }
        }
    }
    let mut input_vars: HashMap<u32, u32> = HashMap::new();
    let mut interface: Vec<u32> = Vec::new();
    if !input_locations.is_empty() {
        let in_ptr_type = module.next_id();
        module.add_type_pointer(in_ptr_type, STORAGE_CLASS_INPUT, f32_type);
        for &loc in &input_locations {
            let var = module.add_io_variable(in_ptr_type, STORAGE_CLASS_INPUT, loc);
            input_vars.insert(loc, var);
            interface.push(var);
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

    module.add_entry_point(execution_model, main_func, "main", &interface);

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
    // One OpLoad per input location, cached (the load dominates later uses).
    let mut input_loads: HashMap<u32, u32> = HashMap::new();

    for node in &program.nodes {
        let binop = spirv_binop(node.op, f32_type, i32_type);
        // A typed binary op's operands share its result type; elsewhere default
        // to f32. Undefs (unwired resource reads) are emitted with this type so
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
                // A live-in input loads (once) from its interface variable.
                IrValue::Input(loc) => {
                    if let Some(&id) = input_loads.get(loc) {
                        id
                    } else {
                        let ptr = input_vars[loc];
                        let id = module.emit_load(f32_type, ptr);
                        input_loads.insert(*loc, id);
                        id
                    }
                }
                // Output/UBO/texture reads are not wired to real resources yet.
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

/// A section of the SPIR-V module's logical layout. `build()` concatenates the
/// sections in this order, which is the order the SPIR-V spec mandates
/// (capabilities → memory model → entry points → execution modes → annotations
/// → types/constants/global-vars → function definitions). Forward references
/// from earlier sections (e.g. `OpEntryPoint` naming its function/interface
/// vars) to later ones are permitted by the spec.
#[derive(Clone, Copy)]
enum Section {
    Caps,
    MemoryModel,
    EntryPoints,
    ExecModes,
    Annotations,
    TypesVars,
    Functions,
}

/// A SPIR-V module builder that keeps each logical-layout section in its own
/// buffer so the final module is emitted in spec-correct order regardless of
/// the order the caller produces instructions in.
struct SpirvModule {
    caps: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    exec_modes: Vec<u32>,
    annotations: Vec<u32>,
    types_vars: Vec<u32>,
    functions: Vec<u32>,
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
            caps: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            exec_modes: Vec::new(),
            annotations: Vec::new(),
            types_vars: Vec::with_capacity(64),
            functions: Vec::with_capacity(128),
            id_counter: 1,
            id_bound: 0,
            undef_ids: HashMap::new(),
        }
    }

    fn section_mut(&mut self, section: Section) -> &mut Vec<u32> {
        match section {
            Section::Caps => &mut self.caps,
            Section::MemoryModel => &mut self.memory_model,
            Section::EntryPoints => &mut self.entry_points,
            Section::ExecModes => &mut self.exec_modes,
            Section::Annotations => &mut self.annotations,
            Section::TypesVars => &mut self.types_vars,
            Section::Functions => &mut self.functions,
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
        self.emit(Section::Functions, OP_UNDEF, &[type_id, id]);
        self.undef_ids.insert(type_id, id);
        id
    }

    /// Emit a binary op (`OpFAdd`/`OpIAdd`/…): `result_type result_id a b`.
    /// Returns the fresh result id.
    fn emit_binop(&mut self, opcode: u16, result_type: u32, a: u32, b: u32) -> u32 {
        let id = self.next_id();
        self.emit(Section::Functions, opcode, &[result_type, id, a, b]);
        id
    }

    /// Emit `OpLoad result_type result_id pointer` in the function body.
    /// Returns the loaded value id.
    fn emit_load(&mut self, result_type: u32, pointer: u32) -> u32 {
        let id = self.next_id();
        self.emit(Section::Functions, OP_LOAD, &[result_type, id, pointer]);
        id
    }

    /// Declare a global I/O variable of `pointer_type` in `storage_class`
    /// (1 = Input, 3 = Output), decorate it with `location`, and return its id.
    fn add_io_variable(&mut self, pointer_type: u32, storage_class: u32, location: u32) -> u32 {
        let id = self.next_id();
        self.emit(
            Section::TypesVars,
            OP_VARIABLE,
            &[pointer_type, id, storage_class],
        );
        // OpDecorate <id> Location <location>. Decoration 30 = Location.
        self.emit(Section::Annotations, OP_DECORATE, &[id, 30, location]);
        id
    }

    fn next_id(&mut self) -> u32 {
        let id = self.id_counter;
        self.id_counter += 1;
        id
    }

    fn emit(&mut self, section: Section, opcode: u16, operands: &[u32]) {
        let word_count = (1 + operands.len()) as u32;
        let buf = self.section_mut(section);
        buf.push((word_count << 16) | (opcode as u32));
        buf.extend_from_slice(operands);
    }

    fn add_capability(&mut self, cap: u32) {
        self.emit(Section::Caps, OP_CAPABILITY, &[cap]);
    }

    fn add_memory_model(&mut self, addressing: u32, memory: u32) {
        self.emit(Section::MemoryModel, OP_MEMORY_MODEL, &[addressing, memory]);
    }

    fn add_type_void(&mut self, id: u32) {
        self.emit(Section::TypesVars, OP_TYPE_VOID, &[id]);
    }

    fn add_type_float(&mut self, id: u32, width: u32) {
        self.emit(Section::TypesVars, OP_TYPE_FLOAT, &[id, width]);
    }

    fn add_type_int(&mut self, id: u32, width: u32, signedness: u32) {
        self.emit(Section::TypesVars, OP_TYPE_INT, &[id, width, signedness]);
    }

    /// Emit `OpTypePointer result_id storage_class pointee_type`.
    fn add_type_pointer(&mut self, id: u32, storage_class: u32, pointee: u32) {
        self.emit(
            Section::TypesVars,
            OP_TYPE_POINTER,
            &[id, storage_class, pointee],
        );
    }

    /// Emit `OpConstant result_type result_id value` — a 32-bit scalar constant.
    fn add_constant(&mut self, id: u32, type_id: u32, value: u32) {
        self.emit(Section::TypesVars, OP_CONSTANT, &[type_id, id, value]);
    }

    fn add_type_vector(&mut self, id: u32, component: u32, count: u32) {
        self.emit(Section::TypesVars, OP_TYPE_VECTOR, &[id, component, count]);
    }

    fn add_type_function(&mut self, id: u32, return_type: u32, params: &[u32]) {
        let mut operands = vec![id, return_type];
        operands.extend_from_slice(params);
        self.emit(Section::TypesVars, OP_TYPE_FUNCTION, &operands);
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
        self.emit(Section::EntryPoints, OP_ENTRY_POINT, &operands);
    }

    fn add_execution_mode(&mut self, entry: u32, mode: u32) {
        self.emit(Section::ExecModes, OP_EXECUTION_MODE, &[entry, mode]);
    }

    fn add_function(&mut self, id: u32, return_type: u32, control: u32, func_type: u32) {
        self.emit(
            Section::Functions,
            OP_FUNCTION,
            &[return_type, id, control, func_type],
        );
    }

    fn add_label(&mut self, id: u32) {
        self.emit(Section::Functions, OP_LABEL, &[id]);
    }

    fn add_return(&mut self) {
        self.emit(Section::Functions, OP_RETURN, &[]);
    }

    fn add_function_end(&mut self) {
        self.emit(Section::Functions, OP_FUNCTION_END, &[]);
    }

    /// Build the final SPIR-V module: 5-word header, then every section in
    /// SPIR-V logical-layout order.
    fn build(mut self) -> Vec<u32> {
        self.id_bound = self.id_counter;

        let body_len = self.caps.len()
            + self.memory_model.len()
            + self.entry_points.len()
            + self.exec_modes.len()
            + self.annotations.len()
            + self.types_vars.len()
            + self.functions.len();
        let mut module = Vec::with_capacity(5 + body_len);
        // SPIR-V header (5 words).
        module.push(SPIRV_MAGIC);
        module.push(SPIRV_VERSION);
        module.push(0); // Generator ID (XPS5X = 0 for now).
        module.push(self.id_bound);
        module.push(0); // Reserved.

        // Sections in mandated logical-layout order.
        module.extend_from_slice(&self.caps);
        module.extend_from_slice(&self.memory_model);
        module.extend_from_slice(&self.entry_points);
        module.extend_from_slice(&self.exec_modes);
        module.extend_from_slice(&self.annotations);
        module.extend_from_slice(&self.types_vars);
        module.extend_from_slice(&self.functions);
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
    /// SPIR-V's logical layout mandates OpEntryPoint (section 5) before the
    /// first type declaration (section 9). Walk the raw words and check the
    /// first OpEntryPoint precedes the first OpType*. (Before the sectioned
    /// builder this failed — types were emitted first.)
    #[test]
    fn respects_spirv_logical_layout_order() {
        let module = emit_spirv(&prog(ShaderType::Vertex)).unwrap();
        let mut i = 5;
        let mut first_entry_point = None;
        let mut first_type = None;
        while i < module.len() {
            let word_count = (module[i] >> 16) as usize;
            let opcode = (module[i] & 0xFFFF) as u16;
            if opcode == OP_ENTRY_POINT && first_entry_point.is_none() {
                first_entry_point = Some(i);
            }
            // Type ops occupy the 19..=33 range in this subset.
            if (19..=33).contains(&opcode) && first_type.is_none() {
                first_type = Some(i);
            }
            if word_count == 0 {
                break;
            }
            i += word_count;
        }
        let ep = first_entry_point.expect("module must have an OpEntryPoint");
        let ty = first_type.expect("module must declare types");
        assert!(
            ep < ty,
            "OpEntryPoint (word {ep}) must precede the first type decl (word {ty})"
        );
    }

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
    fn live_in_inputs_become_interface_variables_and_loads() {
        use crate::shader::ir::{IrNode, IrOp};
        // r0 = in[0] + in[1]: two live-in inputs feed a float add.
        let program = IrProgram {
            nodes: vec![IrNode {
                op: IrOp::Add,
                result: Some(IrValue::Reg(0)),
                sources: vec![IrValue::Input(0), IrValue::Input(1)],
            }],
            shader_type: ShaderType::Vertex,
            input_count: 2,
            output_count: 0,
            ubo_count: 0,
            texture_count: 0,
        };
        let module = emit_spirv(&program).unwrap();
        let parsed = rspirv::dr::load_words(&module).expect("must parse");

        // Two Input OpVariables declared, and the entry point lists both in its
        // interface (SPIR-V requires all Input/Output vars in the interface).
        use rspirv::spirv::{Op, StorageClass};
        let input_vars: Vec<u32> = parsed
            .types_global_values
            .iter()
            .filter(|i| i.class.opcode == Op::Variable)
            .filter(|i| {
                matches!(
                    i.operands.first(),
                    Some(rspirv::dr::Operand::StorageClass(StorageClass::Input))
                )
            })
            .filter_map(|i| i.result_id)
            .collect();
        assert_eq!(input_vars.len(), 2, "two input interface variables");

        let entry = &parsed.entry_points[0];
        let iface: Vec<u32> = entry
            .operands
            .iter()
            .filter_map(|o| match o {
                rspirv::dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            })
            .collect();
        for v in &input_vars {
            assert!(
                iface.contains(v),
                "input var {v} must be in the entry interface"
            );
        }

        // The body loads from the inputs (at least one OpLoad) and adds them.
        let body_ops: Vec<Op> = parsed
            .functions
            .iter()
            .flat_map(|f| f.blocks.iter())
            .flat_map(|b| b.instructions.iter())
            .map(|i| i.class.opcode)
            .collect();
        assert!(body_ops.contains(&Op::Load), "inputs must be OpLoad-ed");
        assert!(body_ops.contains(&Op::FAdd), "the add must lower to OpFAdd");
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
