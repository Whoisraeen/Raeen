//! Shader intermediate representation (IR).
//!
//! After decoding RDNA2 ISA, instructions are lifted into this IR
//! which is closer to SPIR-V's SSA form. This allows optimization
//! passes before final SPIR-V emission.

use super::gcn_decoder::{Encoding, Instruction, Operand};
use super::ShaderType;
use std::collections::HashMap;

/// An IR program ready for SPIR-V emission.
#[derive(Debug)]
pub struct IrProgram {
    /// IR nodes (instructions in SSA form).
    pub nodes: Vec<IrNode>,
    /// Shader type.
    pub shader_type: ShaderType,
    /// Number of input attributes.
    pub input_count: u32,
    /// Number of output attributes.
    pub output_count: u32,
    /// Number of UBO bindings.
    pub ubo_count: u32,
    /// Number of texture bindings.
    pub texture_count: u32,
}

/// A single IR node.
#[derive(Debug, Clone)]
pub struct IrNode {
    /// IR operation.
    pub op: IrOp,
    /// Result register (SSA value).
    pub result: Option<IrValue>,
    /// Source operands.
    pub sources: Vec<IrValue>,
}

/// IR operations — a simplified instruction set that maps well to SPIR-V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrOp {
    // ─── Arithmetic ────────────────────────────────
    Add,
    Sub,
    Mul,
    Div,
    Fma, // Fused multiply-add
    Min,
    Max,
    Abs,
    Neg,
    Sqrt,
    Rsqrt, // Reciprocal square root
    Rcp,   // Reciprocal
    Floor,
    Ceil,
    Fract,
    // ─── Integer ───────────────────────────────────
    IAdd,
    ISub,
    IMul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    // ─── Comparison ────────────────────────────────
    CmpEq,
    CmpNe,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,
    // ─── Conversion ────────────────────────────────
    F32ToI32,
    I32ToF32,
    F32ToF16,
    F16ToF32,
    // ─── Memory ────────────────────────────────────
    Load,
    Store,
    BufferLoad,
    BufferStore,
    ImageSample,
    ImageLoad,
    ImageStore,
    // ─── Interpolation ─────────────────────────────
    Interp,
    // ─── Flow control ──────────────────────────────
    Branch,
    BranchCond,
    Phi, // SSA phi node
    Return,
    // ─── Export ─────────────────────────────────────
    ExportPosition,
    ExportParam,
    ExportColor,
    // ─── Misc ──────────────────────────────────────
    Mov,
    Select, // Ternary select (condition ? a : b)
    Nop,
}

/// An IR value (SSA register or constant).
#[derive(Debug, Clone)]
pub enum IrValue {
    /// SSA register.
    Reg(u32),
    /// 32-bit float constant.
    ConstF32(f32),
    /// 32-bit integer constant.
    ConstI32(i32),
    /// 32-bit unsigned constant.
    ConstU32(u32),
    /// Input attribute (vertex input or varying).
    Input(u32),
    /// Output attribute (vertex output or pixel color).
    Output(u32),
    /// Uniform buffer binding.
    Ubo(u32, u32), // (binding, offset)
    /// Texture/sampler binding.
    Texture(u32),
}

/// Lift decoded RDNA2 instructions to IR.
pub fn lift_to_ir(instructions: &[Instruction], shader_type: ShaderType) -> IrProgram {
    let mut nodes = Vec::with_capacity(instructions.len());
    let mut next_ssa = 0u32;
    let mut input_count = 0u32;
    let mut output_count = 0u32;
    let mut ubo_count = 0u32;
    let mut texture_count = 0u32;
    // Local value numbering: which SSA value currently lives in each VGPR.
    // A read of a VGPR no prior instruction wrote is a shader live-in.
    let mut vgpr_def: HashMap<u32, u32> = HashMap::new();

    for inst in instructions {
        let node = match inst.encoding {
            Encoding::Vop2 | Encoding::Vop1 | Encoding::Vop3 => {
                // Vector ALU → arithmetic IR ops. Thread each decoded source
                // operand to the SSA value it refers to.
                let sources = inst
                    .src
                    .iter()
                    .map(|op| resolve_source(op, &vgpr_def))
                    .collect();

                let result_reg = next_ssa;
                next_ssa += 1;
                // The destination VGPR now holds this SSA value.
                if let Some(Operand::Vgpr(n)) = &inst.dst {
                    vgpr_def.insert(*n, result_reg);
                }

                IrNode {
                    op: map_vop_to_ir(inst.opcode, inst.encoding),
                    result: Some(IrValue::Reg(result_reg)),
                    sources,
                }
            }
            Encoding::Sop2 | Encoding::Sop1 | Encoding::Sopk => {
                // Scalar ALU → integer IR ops.
                let result_reg = next_ssa;
                next_ssa += 1;

                IrNode {
                    op: map_sop_to_ir(inst.opcode, inst.encoding),
                    result: Some(IrValue::Reg(result_reg)),
                    sources: vec![],
                }
            }
            Encoding::Exp => {
                // Export instruction → write output.
                output_count += 1;
                IrNode {
                    op: IrOp::ExportColor, // Refined based on target in detailed decode.
                    result: None,
                    sources: vec![],
                }
            }
            Encoding::Smem => {
                // Scalar memory load → buffer load.
                ubo_count = ubo_count.max(1);
                let result_reg = next_ssa;
                next_ssa += 1;

                IrNode {
                    op: IrOp::BufferLoad,
                    result: Some(IrValue::Reg(result_reg)),
                    sources: vec![],
                }
            }
            Encoding::Mimg => {
                // Image instruction → texture sample.
                texture_count += 1;
                let result_reg = next_ssa;
                next_ssa += 1;

                IrNode {
                    op: IrOp::ImageSample,
                    result: Some(IrValue::Reg(result_reg)),
                    sources: vec![],
                }
            }
            Encoding::Vintrp => {
                // Vertex interpolation → input attribute.
                input_count += 1;
                let result_reg = next_ssa;
                next_ssa += 1;

                IrNode {
                    op: IrOp::Interp,
                    result: Some(IrValue::Reg(result_reg)),
                    sources: vec![],
                }
            }
            Encoding::Sopp if inst.opcode == 0x01 => {
                // S_ENDPGM → Return.
                IrNode {
                    op: IrOp::Return,
                    result: None,
                    sources: vec![],
                }
            }
            _ => {
                // Default: NOP.
                IrNode {
                    op: IrOp::Nop,
                    result: None,
                    sources: vec![],
                }
            }
        };

        nodes.push(node);
    }

    IrProgram {
        nodes,
        shader_type,
        input_count,
        output_count,
        ubo_count,
        texture_count,
    }
}

/// Resolve a decoded GCN operand to the IR value it references.
///
/// A VGPR read resolves to the SSA value a prior instruction wrote into it
/// (`vgpr_def`); a VGPR with no prior definition is a shader live-in
/// (`Input`). Inline integer constants and literals fold to IR constants.
/// SGPR/special sources are modelled as live-ins for now (a full port needs a
/// parallel scalar SSA map — see the ledger's remaining-work note).
fn resolve_source(op: &Operand, vgpr_def: &HashMap<u32, u32>) -> IrValue {
    match op {
        Operand::Vgpr(n) => match vgpr_def.get(n) {
            Some(&ssa) => IrValue::Reg(ssa),
            None => IrValue::Input(*n),
        },
        Operand::Sgpr(n) => IrValue::Input(*n),
        Operand::InlineConst(c) => IrValue::ConstI32(*c),
        Operand::Literal(bits) => IrValue::ConstU32(*bits),
        Operand::Special(_) => IrValue::Input(0),
    }
}

/// Map VOP opcodes to IR operations (simplified).
fn map_vop_to_ir(opcode: u32, encoding: Encoding) -> IrOp {
    match encoding {
        Encoding::Vop2 => match opcode {
            0x01 => IrOp::Add,  // V_ADD_F32
            0x02 => IrOp::Sub,  // V_SUB_F32
            0x04 => IrOp::Mul,  // V_MUL_F32
            0x05 => IrOp::Min,  // V_MIN_F32
            0x06 => IrOp::Max,  // V_MAX_F32
            0x08 => IrOp::Fma,  // V_MAC_F32
            0x19 => IrOp::IAdd, // V_ADD_U32
            0x1A => IrOp::ISub, // V_SUB_U32
            0x1B => IrOp::And,  // V_AND_B32
            0x1C => IrOp::Or,   // V_OR_B32
            0x1D => IrOp::Xor,  // V_XOR_B32
            _ => IrOp::Nop,
        },
        Encoding::Vop1 => match opcode {
            0x01 => IrOp::Mov,      // V_MOV_B32
            0x20 => IrOp::Sqrt,     // V_SQRT_F32
            0x21 => IrOp::Rsqrt,    // V_RSQ_F32
            0x22 => IrOp::Rcp,      // V_RCP_F32
            0x23 => IrOp::Floor,    // V_FLOOR_F32
            0x24 => IrOp::Ceil,     // V_CEIL_F32
            0x25 => IrOp::Fract,    // V_FRACT_F32
            0x33 => IrOp::F32ToI32, // V_CVT_I32_F32
            0x34 => IrOp::I32ToF32, // V_CVT_F32_I32
            _ => IrOp::Nop,
        },
        _ => IrOp::Nop,
    }
}

/// Map SOP opcodes to IR operations (simplified).
fn map_sop_to_ir(opcode: u32, encoding: Encoding) -> IrOp {
    match encoding {
        Encoding::Sop2 => match opcode {
            0x00 => IrOp::IAdd, // S_ADD_U32
            0x01 => IrOp::ISub, // S_SUB_U32
            0x0E => IrOp::And,  // S_AND_B32
            0x0F => IrOp::Or,   // S_OR_B32
            0x10 => IrOp::Xor,  // S_XOR_B32
            _ => IrOp::Nop,
        },
        Encoding::Sop1 => match opcode {
            0x03 => IrOp::Mov, // S_MOV_B32
            _ => IrOp::Nop,
        },
        _ => IrOp::Nop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::gcn_decoder::{Encoding, Instruction};

    use crate::shader::gcn_decoder::Operand;

    /// Build a bare decoded instruction of the given encoding/opcode. Operand
    /// fields stay empty — `lift_to_ir` classifies by encoding+opcode, which is
    /// exactly the surface under test here.
    fn inst(encoding: Encoding, opcode: u32) -> Instruction {
        Instruction {
            encoding,
            opcode,
            raw: 0,
            src: vec![],
            dst: None,
            size: 4,
            offset: 0,
        }
    }

    /// Build a VOP2 instruction with explicit operands.
    fn vop2(opcode: u32, src: Vec<Operand>, dst: Operand) -> Instruction {
        Instruction {
            encoding: Encoding::Vop2,
            opcode,
            raw: 0,
            src,
            dst: Some(dst),
            size: 4,
            offset: 0,
        }
    }

    #[test]
    fn vop_and_sop_opcodes_map_to_the_right_ir_ops() {
        // VOP2 arithmetic + integer, VOP1 unary, SOP2 scalar, SOP1 move.
        let stream = [
            inst(Encoding::Vop2, 0x01), // V_ADD_F32  -> Add
            inst(Encoding::Vop2, 0x04), // V_MUL_F32  -> Mul
            inst(Encoding::Vop2, 0x19), // V_ADD_U32  -> IAdd
            inst(Encoding::Vop1, 0x01), // V_MOV_B32  -> Mov
            inst(Encoding::Vop1, 0x20), // V_SQRT_F32 -> Sqrt
            inst(Encoding::Sop2, 0x00), // S_ADD_U32  -> IAdd
            inst(Encoding::Sop1, 0x03), // S_MOV_B32  -> Mov
        ];
        let prog = lift_to_ir(&stream, ShaderType::Vertex);
        let ops: Vec<IrOp> = prog.nodes.iter().map(|n| n.op).collect();
        assert_eq!(
            ops,
            vec![
                IrOp::Add,
                IrOp::Mul,
                IrOp::IAdd,
                IrOp::Mov,
                IrOp::Sqrt,
                IrOp::IAdd,
                IrOp::Mov,
            ]
        );
        // An unrecognized opcode within a known ALU encoding lowers to Nop, not a panic.
        let junk = lift_to_ir(&[inst(Encoding::Vop2, 0xFF)], ShaderType::Vertex);
        assert_eq!(junk.nodes[0].op, IrOp::Nop);
    }

    #[test]
    fn alu_results_get_sequential_ssa_registers() {
        // Every ALU/memory op allocates one fresh SSA result, numbered 0,1,2,…
        let stream = [
            inst(Encoding::Vop2, 0x01),
            inst(Encoding::Sop2, 0x00),
            inst(Encoding::Vop1, 0x01),
        ];
        let prog = lift_to_ir(&stream, ShaderType::Vertex);
        let regs: Vec<u32> = prog
            .nodes
            .iter()
            .filter_map(|n| match n.result {
                Some(IrValue::Reg(r)) => Some(r),
                _ => None,
            })
            .collect();
        assert_eq!(
            regs,
            vec![0, 1, 2],
            "SSA results are numbered in program order"
        );
    }

    #[test]
    fn export_and_endpgm_produce_sinks_without_ssa_results() {
        // EXP writes an output (no SSA result); S_ENDPGM is a Return sink.
        let stream = [inst(Encoding::Exp, 0), inst(Encoding::Sopp, 0x01)];
        let prog = lift_to_ir(&stream, ShaderType::Pixel);
        assert_eq!(prog.nodes[0].op, IrOp::ExportColor);
        assert!(
            prog.nodes[0].result.is_none(),
            "an export has no SSA result"
        );
        assert_eq!(prog.nodes[1].op, IrOp::Return, "S_ENDPGM -> Return");
        assert!(prog.nodes[1].result.is_none());
        assert_eq!(prog.output_count, 1, "the EXP bumped the output count");
    }

    #[test]
    fn sources_thread_vgpr_defs_into_an_ssa_def_use_chain() {
        // v2 = v0 + v1        (node 0, SSA reg 0; v0/v1 are live-ins)
        // v3 = v2 * v2        (node 1, SSA reg 1; both operands are node 0's def)
        let stream = [
            vop2(
                0x01,
                vec![Operand::Vgpr(0), Operand::Vgpr(1)],
                Operand::Vgpr(2),
            ),
            vop2(
                0x04,
                vec![Operand::Vgpr(2), Operand::Vgpr(2)],
                Operand::Vgpr(3),
            ),
        ];
        let prog = lift_to_ir(&stream, ShaderType::Vertex);

        // Node 0: reads two undefined VGPRs → live-in Inputs; defines SSA 0.
        assert_eq!(prog.nodes[0].op, IrOp::Add);
        assert!(matches!(prog.nodes[0].result, Some(IrValue::Reg(0))));
        assert!(matches!(prog.nodes[0].sources[0], IrValue::Input(0)));
        assert!(matches!(prog.nodes[0].sources[1], IrValue::Input(1)));

        // Node 1: both operands read v2, which node 0 wrote → SSA reg 0.
        assert_eq!(prog.nodes[1].op, IrOp::Mul);
        assert!(matches!(prog.nodes[1].result, Some(IrValue::Reg(1))));
        assert!(matches!(prog.nodes[1].sources[0], IrValue::Reg(0)));
        assert!(matches!(prog.nodes[1].sources[1], IrValue::Reg(0)));
    }

    #[test]
    fn inline_constants_fold_into_ir_source_values() {
        // v1 = v0 + (inline 5)
        let stream = [vop2(
            0x01,
            vec![Operand::Vgpr(0), Operand::InlineConst(5)],
            Operand::Vgpr(1),
        )];
        let prog = lift_to_ir(&stream, ShaderType::Vertex);
        assert!(matches!(prog.nodes[0].sources[1], IrValue::ConstI32(5)));
    }

    #[test]
    fn resource_counts_reflect_memory_and_interp_instructions() {
        // SMEM -> ubo, MIMG -> texture, VINTRP -> input, EXP -> output.
        let stream = [
            inst(Encoding::Smem, 0x00),
            inst(Encoding::Mimg, 0x00),
            inst(Encoding::Mimg, 0x00),
            inst(Encoding::Vintrp, 0x00),
            inst(Encoding::Exp, 0x00),
        ];
        let prog = lift_to_ir(&stream, ShaderType::Pixel);
        assert_eq!(
            prog.ubo_count, 1,
            "SMEM load implies at least one UBO binding"
        );
        assert_eq!(prog.texture_count, 2, "two image samples -> two textures");
        assert_eq!(prog.input_count, 1, "one interpolated varying");
        assert_eq!(prog.output_count, 1, "one export");
        // The SMEM/MIMG/VINTRP results are still SSA-numbered without collision.
        assert_eq!(prog.nodes[0].op, IrOp::BufferLoad);
        assert_eq!(prog.nodes[1].op, IrOp::ImageSample);
        assert_eq!(prog.nodes[3].op, IrOp::Interp);
    }
}
