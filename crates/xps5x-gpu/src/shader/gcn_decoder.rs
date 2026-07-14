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
    Vop3p, // Packed math (FP16)
    // Memory
    Smem,  // Scalar memory
    Mubuf, // Untyped buffer
    Mtbuf, // Typed buffer
    Mimg,  // Image (texture)
    Flat,  // Flat/global/scratch memory
    // Export
    Exp,
    // Special
    Vintrp, // Vertex interpolation
    Ds,     // Local data share
}

/// Instruction operand.
#[derive(Debug, Clone, PartialEq, Eq)]
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

        // Populate operands for the encodings we model precisely; other
        // encodings keep empty operand lists until their layout is decoded.
        let (src, dst) = decode_operands(encoding, word0);

        let instruction = Instruction {
            encoding,
            opcode,
            raw,
            src,
            dst,
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

/// Decode the source/destination operands for the encodings whose layout we
/// model. VOP1/VOP2 share the 9-bit SRC0 field + 8-bit VGPR fields; other
/// encodings return empty operands until their layout is added.
fn decode_operands(encoding: Encoding, word: u32) -> (Vec<Operand>, Option<Operand>) {
    match encoding {
        // VOP2: SRC0[8:0] (any source), VSRC1[16:9] (VGPR), VDST[24:17] (VGPR).
        Encoding::Vop2 => {
            let src0 = decode_src9(word & 0x1FF);
            let src1 = Operand::Vgpr((word >> 9) & 0xFF);
            let vdst = Operand::Vgpr((word >> 17) & 0xFF);
            (vec![src0, src1], Some(vdst))
        }
        // VOP1: SRC0[8:0] (any source), OP[16:9], VDST[24:17] (VGPR).
        Encoding::Vop1 => {
            let src0 = decode_src9(word & 0x1FF);
            let vdst = Operand::Vgpr((word >> 17) & 0xFF);
            (vec![src0], Some(vdst))
        }
        // SOP2: SSRC0[7:0], SSRC1[15:8] (8-bit scalar sources — never VGPRs),
        // SDST[22:16] (SGPR/special).
        Encoding::Sop2 => {
            let ssrc0 = decode_src9(word & 0xFF);
            let ssrc1 = decode_src9((word >> 8) & 0xFF);
            let sdst = decode_src9((word >> 16) & 0x7F);
            (vec![ssrc0, ssrc1], Some(sdst))
        }
        // SOP1: SSRC0[7:0], OP[15:8], SDST[22:16].
        Encoding::Sop1 => {
            let ssrc0 = decode_src9(word & 0xFF);
            let sdst = decode_src9((word >> 16) & 0x7F);
            (vec![ssrc0], Some(sdst))
        }
        _ => (Vec::new(), None),
    }
}

/// Decode a 9-bit VOP source field (SRC0 / SSRC) into an operand: SGPRs,
/// VGPRs, inline integer/float constants, the common special registers, and
/// the literal-follows marker. Values are per the RDNA2 ISA source encoding.
fn decode_src9(field: u32) -> Operand {
    match field {
        0..=101 => Operand::Sgpr(field),
        106 => Operand::Special(SpecialReg::Vcc), // VCC_LO
        124 => Operand::Special(SpecialReg::M0),
        125 => Operand::Special(SpecialReg::Null),
        126 => Operand::Special(SpecialReg::Exec), // EXEC_LO
        128 => Operand::InlineConst(0),
        129..=192 => Operand::InlineConst((field - 128) as i32), // +1..+64
        193..=208 => Operand::InlineConst(-((field - 192) as i32)), // -1..-16
        // Inline floating-point constants — carried as their IEEE-754 bit pattern.
        240 => Operand::Literal(0.5f32.to_bits()),
        241 => Operand::Literal((-0.5f32).to_bits()),
        242 => Operand::Literal(1.0f32.to_bits()),
        243 => Operand::Literal((-1.0f32).to_bits()),
        244 => Operand::Literal(2.0f32.to_bits()),
        245 => Operand::Literal((-2.0f32).to_bits()),
        246 => Operand::Literal(4.0f32.to_bits()),
        247 => Operand::Literal((-4.0f32).to_bits()),
        255 => Operand::Literal(0), // literal dword follows the instruction
        256..=511 => Operand::Vgpr(field - 256),
        // Remaining special registers (VCC_HI/EXEC_HI/TMA/TTMP/…) not yet modeled.
        _ => Operand::Special(SpecialReg::Null),
    }
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
        // EXP: the export target (MRT/POS/PARAM) is TGT[9:4]; carry it as the opcode.
        0b1111110 => return (Encoding::Exp, (word >> 4) & 0x3F, 8),
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

    if top4 == 0b1011 {
        return (Encoding::Sopk, (word >> 23) & 0x1F, 4);
    };

    if top2 == 0b10 {
        return (Encoding::Sop2, (word >> 23) & 0x7F, 4);
    };

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Flatten `words` into a little-endian byte stream (as a real shader
    /// binary is laid out).
    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn too_small_binary_errors() {
        assert!(
            decode(&[0u8; 3]).is_err(),
            "a <4-byte binary can't hold an instruction"
        );
    }

    #[test]
    fn classifies_encodings_and_widths() {
        // VOP2 (bit31=0, not the VOP1/VOPC sub-patterns) → 4 bytes.
        assert_eq!(classify_instruction(0x0000_0000), (Encoding::Vop2, 0, 4));
        // VOP3 (top6 == 0b110100 == 0x34) → 8 bytes.
        assert_eq!(classify_instruction(0xD000_0000).0, Encoding::Vop3);
        assert_eq!(classify_instruction(0xD000_0000).2, 8);
        // SOPP (top9 == 0b101111101); opcode 1 == S_ENDPGM.
        let endpgm = (0b1_0111_1101u32 << 23) | 0x01;
        assert_eq!(classify_instruction(endpgm), (Encoding::Sopp, 1, 4));
    }

    #[test]
    fn decodes_a_stream_and_stops_at_s_endpgm() {
        let endpgm = (0b1_0111_1101u32 << 23) | 0x01;
        // VOP2 (4B), VOP3 (8B: word0 + word1), S_ENDPGM (4B), then a trailing
        // word that must NOT be decoded (decode stops at S_ENDPGM).
        let binary = bytes(&[0x0000_0000, 0xD000_0000, 0x0000_0000, endpgm, 0xDEAD_BEEF]);
        let insns = decode(&binary).expect("decodes");

        assert_eq!(
            insns.len(),
            3,
            "VOP2 + VOP3 + S_ENDPGM (trailing word not reached)"
        );
        assert_eq!(insns[0].encoding, Encoding::Vop2);
        assert_eq!(insns[0].size, 4);
        assert_eq!(insns[1].encoding, Encoding::Vop3);
        assert_eq!(insns[1].size, 8);
        assert_eq!(insns[1].raw, 0xD000_0000, "VOP3 raw is (word1<<32)|word0");
        assert_eq!(insns[2].encoding, Encoding::Sopp);
        assert_eq!(insns[2].opcode, 1, "S_ENDPGM");
        // Byte offsets advance by each instruction's width.
        assert_eq!(insns[0].offset, 0);
        assert_eq!(insns[1].offset, 4);
        assert_eq!(insns[2].offset, 12);
    }

    #[test]
    fn src9_decodes_each_operand_class() {
        assert_eq!(decode_src9(0), Operand::Sgpr(0));
        assert_eq!(decode_src9(101), Operand::Sgpr(101));
        assert_eq!(decode_src9(256), Operand::Vgpr(0), "256 is VGPR 0");
        assert_eq!(decode_src9(300), Operand::Vgpr(44));
        assert_eq!(decode_src9(128), Operand::InlineConst(0));
        assert_eq!(decode_src9(129), Operand::InlineConst(1), "129 -> +1");
        assert_eq!(decode_src9(192), Operand::InlineConst(64), "192 -> +64");
        assert_eq!(decode_src9(193), Operand::InlineConst(-1), "193 -> -1");
        assert_eq!(decode_src9(208), Operand::InlineConst(-16));
        assert_eq!(decode_src9(106), Operand::Special(SpecialReg::Vcc));
        assert_eq!(decode_src9(126), Operand::Special(SpecialReg::Exec));
        // Inline float 1.0 (encoding 242) carries the IEEE-754 bit pattern.
        assert_eq!(decode_src9(242), Operand::Literal(1.0f32.to_bits()));
        assert_eq!(decode_src9(243), Operand::Literal((-1.0f32).to_bits()));
        // 255 is the "literal dword follows" marker.
        assert_eq!(decode_src9(255), Operand::Literal(0));
    }

    #[test]
    fn vop2_operands_decode_src0_vsrc1_vdst() {
        // Fields: SRC0[8:0]=v3 (256+3=259), VSRC1[16:9]=v5, VDST[24:17]=v7.
        let word = 259 | (5 << 9) | (7 << 17);
        let (src, dst) = decode_operands(Encoding::Vop2, word);
        assert_eq!(src, vec![Operand::Vgpr(3), Operand::Vgpr(5)]);
        assert_eq!(dst, Some(Operand::Vgpr(7)));
    }

    #[test]
    fn vop1_operands_decode_src0_and_vdst() {
        // SRC0[8:0]=s10, VDST[24:17]=v2 (OP field [16:9] is irrelevant here).
        let word = 10 | (2 << 17);
        let (src, dst) = decode_operands(Encoding::Vop1, word);
        assert_eq!(src, vec![Operand::Sgpr(10)]);
        assert_eq!(dst, Some(Operand::Vgpr(2)));
    }

    #[test]
    fn sop2_operands_decode_ssrc0_ssrc1_sdst() {
        // S_ADD_U32 s2, s0, s1: SSRC0[7:0]=0, SSRC1[15:8]=1, SDST[22:16]=2.
        let word = (1 << 8) | (2 << 16);
        let (src, dst) = decode_operands(Encoding::Sop2, word);
        assert_eq!(src, vec![Operand::Sgpr(0), Operand::Sgpr(1)]);
        assert_eq!(dst, Some(Operand::Sgpr(2)));
    }

    #[test]
    fn sop1_operands_decode_ssrc0_and_sdst() {
        // S_MOV_B32 s5, s3: SSRC0[7:0]=3, SDST[22:16]=5.
        let word = 3 | (5 << 16);
        let (src, dst) = decode_operands(Encoding::Sop1, word);
        assert_eq!(src, vec![Operand::Sgpr(3)]);
        assert_eq!(dst, Some(Operand::Sgpr(5)));
    }

    #[test]
    fn unmodeled_encodings_have_empty_operands() {
        // We don't yet decode SMEM operand layout — it stays empty, not wrong.
        let (src, dst) = decode_operands(Encoding::Smem, 0xFFFF_FFFF);
        assert!(src.is_empty() && dst.is_none());
    }
}
