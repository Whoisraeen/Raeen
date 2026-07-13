//! GCN/RDNA2 ISA instruction decoder.
//!
//! Decodes binary shader instructions from AMD's RDNA2 ISA into
//! a structured representation. RDNA2 instructions are variable-length
//! (32-bit or 64-bit) and divided into categories:
//!
//! - **Scalar ALU** (SOP1, SOP2, SOPC, SOPK, SOPP) — Lane-uniform operations
//! - **Vector ALU** (VOP1, VOP2, VOP3, VOPC) — Per-lane SIMD operations
//! - **Memory** (SMEM, MUBUF, MTBUF, MIMG, FLAT) — Load/store/texture
//! - **Export** (EXP) — Write outputs (vertex attributes, pixel colors)
//! - **Flow Control** (S_BRANCH, S_CBRANCH, S_ENDPGM, etc.)

use tracing::{debug, warn};
use xps5x_core::error::GpuError;

/// A decoded RDNA2 instruction.
#[derive(Debug, Clone)]
pub struct Instruction {
    /// Instruction encoding type.
    pub encoding: Encoding,
    /// Opcode within the encoding.
    pub opcode: u32,
    /// Raw instruction word(s).
    pub raw: u64,
    /// Source operands.
    pub src: Vec<Operand>,
    /// Destination operand.
    pub dst: Option<Operand>,
    /// Instruction size in bytes (4 or 8).
    pub size: u32,
    /// Offset in the shader binary.
    pub offset: u32,
}

/// Instruction encoding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    // Scalar ALU
    Sop1,
    Sop2,
    Sopc,
    Sopk,
    Sopp,
    // Vector ALU
    Vop1,
    Vop2,
    Vop3,
    Vopc,
    Vop3p,  // Packed math (FP16)
    // Memory
    Smem,   // Scalar memory
    Mubuf,  // Untyped buffer
    Mtbuf,  // Typed buffer
    Mimg,   // Image (texture)
    Flat,   // Flat/global/scratch memory
    // Export
    Exp,
    // Special
    Vintrp, // Vertex interpolation
    Ds,     // Local data share
}

/// Instruction operand.
#[derive(Debug, Clone)]
pub enum Operand {
    /// Scalar general-purpose register (s0-s103).
    Sgpr(u32),
    /// Vector general-purpose register (v0-v255).
    Vgpr(u32),
    /// Inline constant (0-64, negatives, 0.5, 1.0, etc.).
    InlineConst(i32),
    /// Literal constant (32-bit immediate).
    Literal(u32),
    /// Special register (VCC, EXEC, SCC, M0, etc.).
    Special(SpecialReg),
}

/// Special GPU registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialReg {
    /// Vector Condition Code.
    Vcc,
    /// Execution mask.
    Exec,
    /// Scalar Condition Code.
    Scc,
    /// Misc register (used for LDS, memory addressing).
    M0,
    /// NULL (discard result).
    Null,
}

/// Decode a binary RDNA2 shader into a list of instructions.
pub fn decode(binary: &[u8]) -> Result<Vec<Instruction>, GpuError> {
    if binary.len() < 4 {
        return Err(GpuError::ShaderCompilationFailed(
            "Shader binary too small".to_string(),
        ));
    }

    let mut instructions = Vec::new();
    let mut offset = 0u32;

    while (offset as usize + 4) <= binary.len() {
        let pos = offset as usize;
        let word0 = u32::from_le_bytes([
            binary[pos],
            binary[pos + 1],
            binary[pos + 2],
            binary[pos + 3],
        ]);

        // Determine encoding from the high bits.
        let (encoding, opcode, size) = classify_instruction(word0);

        let raw = if size == 8 && (pos + 8) <= binary.len() {
            let word1 = u32::from_le_bytes([
                binary[pos + 4],
                binary[pos + 5],
                binary[pos + 6],
                binary[pos + 7],
            ]);
            ((word1 as u64) << 32) | (word0 as u64)
        } else {
            word0 as u64
        };

        let instruction = Instruction {
            encoding,
            opcode,
            raw,
            src: Vec::new(), // Operand decoding is encoding-specific.
            dst: None,
            size,
            offset,
        };

        // Check for S_ENDPGM (end of shader).
        if encoding == Encoding::Sopp && opcode == 0x01 {
            instructions.push(instruction);
            debug!("S_ENDPGM at offset {:#x} — shader decode complete", offset);
            break;
        }

        instructions.push(instruction);
        offset += size;
    }

    Ok(instructions)
}

/// Classify an instruction by examining the high bits of the first word.
fn classify_instruction(word: u32) -> (Encoding, u32, u32) {
    let top9 = word >> 23;
    let top7 = word >> 25;
    let top6 = word >> 26;
    let _top5 = word >> 27;
    let top4 = word >> 28;
    let top2 = word >> 30;
    let top1 = word >> 31;

    // Ordered from most-specific (more bits) to least-specific.
    match top9 {
        0b101111101 => return (Encoding::Sopp, word & 0x7F, 4),
        0b101111110 => return (Encoding::Sopc, (word >> 16) & 0x7F, 4),
        0b101111111 => return (Encoding::Sop1, (word >> 8) & 0xFF, 4),
        _ => {}
    };

    match top7 {
        0b1111110 => return (Encoding::Exp, 0, 8),
        0b1100100 => return (Encoding::Smem, (word >> 18) & 0x3F, 8),
        _ => {}
    };

    match top6 {
        0b110100 => return (Encoding::Vop3, (word >> 16) & 0x3FF, 8),
        0b111000 => return (Encoding::Mubuf, (word >> 18) & 0x7F, 8),
        0b111010 => return (Encoding::Mimg, (word >> 18) & 0x7F, 8),
        0b110110 => return (Encoding::Ds, (word >> 17) & 0xFF, 8),
        0b110111 => return (Encoding::Flat, (word >> 18) & 0x7F, 8),
        _ => {}
    };

    if top4 == 0b1011 { return (Encoding::Sopk, (word >> 23) & 0x1F, 4) };

    if top2 == 0b10 { return (Encoding::Sop2, (word >> 23) & 0x7F, 4) };

    if top1 == 0 {
        // VOP1, VOP2, VOPC based on further bits.
        let vop_bits = (word >> 25) & 0x3F;
        if vop_bits == 0b111111 {
            return (Encoding::Vop1, (word >> 9) & 0xFF, 4);
        } else if (word >> 25) & 0x1F == 0b11110 {
            return (Encoding::Vopc, (word >> 17) & 0xFF, 4);
        } else {
            return (Encoding::Vop2, (word >> 25) & 0x3F, 4);
        }
    };

    // Fallback: treat as SOPP NOP.
    warn!("Unknown instruction encoding: {:#010x}", word);
    (Encoding::Sopp, 0, 4)
}
