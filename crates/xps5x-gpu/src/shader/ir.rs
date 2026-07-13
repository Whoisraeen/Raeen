//! Shader intermediate representation (IR).
//!
//! After decoding RDNA2 ISA, instructions are lifted into this IR
//! which is closer to SPIR-V's SSA form. This allows optimization
//! passes before final SPIR-V emission.

use super::gcn_decoder::{Encoding, Instruction};
use super::ShaderType;

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
    Fma,        // Fused multiply-add
    Min,
    Max,
    Abs,
    Neg,
    Sqrt,
    Rsqrt,      // Reciprocal square root
    Rcp,        // Reciprocal
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
    Phi,        // SSA phi node
    Return,
    // ─── Export ─────────────────────────────────────
    ExportPosition,
    ExportParam,
    ExportColor,
    // ─── Misc ──────────────────────────────────────
    Mov,
    Select,     // Ternary select (condition ? a : b)
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

    for inst in instructions {
        let node = match inst.encoding {
            Encoding::Vop2 | Encoding::Vop1 | Encoding::Vop3 => {
                // Vector ALU → arithmetic IR ops.
                let result_reg = next_ssa;
                next_ssa += 1;

                IrNode {
                    op: map_vop_to_ir(inst.opcode, inst.encoding),
                    result: Some(IrValue::Reg(result_reg)),
                    sources: vec![], // Populated by detailed operand decoding.
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

/// Map VOP opcodes to IR operations (simplified).
fn map_vop_to_ir(opcode: u32, encoding: Encoding) -> IrOp {
    match encoding {
        Encoding::Vop2 => match opcode {
            0x01 => IrOp::Add,    // V_ADD_F32
            0x02 => IrOp::Sub,    // V_SUB_F32
            0x04 => IrOp::Mul,    // V_MUL_F32
            0x05 => IrOp::Min,    // V_MIN_F32
            0x06 => IrOp::Max,    // V_MAX_F32
            0x08 => IrOp::Fma,    // V_MAC_F32
            0x19 => IrOp::IAdd,   // V_ADD_U32
            0x1A => IrOp::ISub,   // V_SUB_U32
            0x1B => IrOp::And,    // V_AND_B32
            0x1C => IrOp::Or,     // V_OR_B32
            0x1D => IrOp::Xor,    // V_XOR_B32
            _ => IrOp::Nop,
        },
        Encoding::Vop1 => match opcode {
            0x01 => IrOp::Mov,    // V_MOV_B32
            0x20 => IrOp::Sqrt,   // V_SQRT_F32
            0x21 => IrOp::Rsqrt,  // V_RSQ_F32
            0x22 => IrOp::Rcp,    // V_RCP_F32
            0x23 => IrOp::Floor,  // V_FLOOR_F32
            0x24 => IrOp::Ceil,   // V_CEIL_F32
            0x25 => IrOp::Fract,  // V_FRACT_F32
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
            0x00 => IrOp::IAdd,   // S_ADD_U32
            0x01 => IrOp::ISub,   // S_SUB_U32
            0x0E => IrOp::And,    // S_AND_B32
            0x0F => IrOp::Or,     // S_OR_B32
            0x10 => IrOp::Xor,    // S_XOR_B32
            _ => IrOp::Nop,
        },
        Encoding::Sop1 => match opcode {
            0x03 => IrOp::Mov,    // S_MOV_B32
            _ => IrOp::Nop,
        },
        _ => IrOp::Nop,
    }
}
