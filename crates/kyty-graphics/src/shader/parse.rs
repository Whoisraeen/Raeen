//! GCN instruction decoder, ported from Kyty (MIT (c) InoriRus).
//!
//! Kyty source: `emulator/src/Graphics/ShaderParse.cpp` (3424 lines).
//! Per-family parsers keep Kyty's line anchors in their doc comments.
//!
//! Deviations from Kyty (error handling only — project rule: never panic,
//! never hard-exit in library code; parsing is total over arbitrary bytes):
//! - `KYTY_NI` / `KYTY_UNKNOWN_OP` / `EXIT_NOT_IMPLEMENTED` / `EXIT` become
//!   typed [`ShaderParseError`] values with a loud `tracing::error!` that
//!   includes the Kyty-style `DbgDump` of everything decoded so far.
//! - All dword reads are bounds-checked ([`ShaderParseError::Truncated`]).
//! - The instruction walk is bounded by the buffer length.

use super::types::{
    DppCtrl, DppMode, ShaderCode, ShaderConstant, ShaderInstruction, ShaderInstructionType as T,
    ShaderLabel, ShaderOperand, ShaderOperandType as O, ShaderType,
    shader_instruction_format::Format as F,
};

/// GFX10 `VCMPX` instructions write EXEC only. Older generations also exposed
/// a scalar compare destination, so the generation-aware decoders use this
/// predicate after selecting the opcode.
const fn is_vcmpx_instruction(type_: T) -> bool {
    matches!(
        type_,
        T::VCmpxLtF32
            | T::VCmpxEqF32
            | T::VCmpxLeF32
            | T::VCmpxGtF32
            | T::VCmpxGeF32
            | T::VCmpxNgeF32
            | T::VCmpxNleF32
            | T::VCmpxNeqF32
            | T::VCmpxNltF32
            | T::VCmpxLtI32
            | T::VCmpxEqI32
            | T::VCmpxLeI32
            | T::VCmpxGtI32
            | T::VCmpxNeI32
            | T::VCmpxGeI32
            | T::VCmpxLtU32
            | T::VCmpxEqU32
            | T::VCmpxGtU32
            | T::VCmpxNeU32
            | T::VCmpxGeU32
    )
}

/// Typed replacement for Kyty's hard exits (ShaderParse.cpp macros L21-28).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderParseError {
    /// `KYTY_UNKNOWN_OP` (L25): opcode not in Kyty's table at all.
    UnknownOpcode {
        family: &'static str,
        opcode: u32,
        pc: u32,
        raw: u32,
    },
    /// `KYTY_NI` (L21): opcode known to Kyty by name but not implemented.
    NotImplemented {
        family: &'static str,
        instruction: &'static str,
        opcode: u32,
        pc: u32,
        raw: u32,
    },
    /// `EXIT_NOT_IMPLEMENTED(cond)`: an encoding feature Kyty rejects
    /// (e.g. SDWA modifiers, MUBUF offen, SMEM glc).
    NotImplementedFeature {
        family: &'static str,
        feature: &'static str,
        pc: u32,
    },
    /// `operand_parse` default branch (L77): unknown operand code.
    UnknownOperand { code: u32 },
    /// Buffer ended mid-instruction (Kyty reads unchecked; we bound it).
    Truncated { pc: u32 },
    /// `shader_parse` default branch (L3401): unknown top-level encoding.
    UnknownEncoding { pc: u32, raw: u32 },
    /// `shader_parse_exp` (L2345): unknown exp target/en/compr combination.
    UnknownExpTarget { target: u32, pc: u32 },
    /// `shader_parse_mimg` (L3201): no format for this opcode/dmask pair.
    UnknownMimgFormat { opcode: u32, dmask: u32, pc: u32 },
    /// `shader_parse_mtbuf` (L3244): unsupported dfmt/nfmt combination.
    UnknownMtbufFormat { dfmt: u32, nfmt: u32, pc: u32 },
}

impl std::fmt::Display for ShaderParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownOpcode {
                family,
                opcode,
                pc,
                raw,
            } => write!(
                f,
                "unknown {family} opcode: 0x{opcode:x} at addr 0x{pc:08x} (raw 0x{raw:08x})"
            ),
            Self::NotImplemented {
                family,
                instruction,
                opcode,
                pc,
                raw,
            } => write!(
                f,
                "unknown {family} instruction {instruction}, opcode = 0x{opcode:x} at addr 0x{pc:08x} (raw 0x{raw:08x})"
            ),
            Self::NotImplementedFeature {
                family,
                feature,
                pc,
            } => write!(
                f,
                "not implemented {family} feature: {feature} at addr 0x{pc:08x}"
            ),
            Self::UnknownOperand { code } => write!(f, "unknown operand: {code}"),
            Self::Truncated { pc } => write!(f, "truncated instruction at addr 0x{pc:08x}"),
            Self::UnknownEncoding { pc, raw } => {
                write!(f, "unknown code 0x{raw:08x} at addr 0x{pc:08x}")
            }
            Self::UnknownExpTarget { target, pc } => {
                write!(f, "unknown exp target: 0x{target:02x} at addr 0x{pc:08x}")
            }
            Self::UnknownMimgFormat { opcode, dmask, pc } => write!(
                f,
                "unknown mimg format for opcode: 0x{opcode:02x} at addr 0x{pc:08x}, dmask: 0x{dmask:x}"
            ),
            Self::UnknownMtbufFormat { dfmt, nfmt, pc } => write!(
                f,
                "unknown format: dfmt = {dfmt}, nfmt = {nfmt} at addr 0x{pc:08x}"
            ),
        }
    }
}

impl std::error::Error for ShaderParseError {}

/// `KYTY_NI` (ShaderParse.cpp L21): log Kyty-style (incl. DbgDump) and build
/// the typed error instead of exiting.
fn ni(
    dst: &ShaderCode,
    family: &'static str,
    instruction: &'static str,
    opcode: u32,
    pc: u32,
    raw: u32,
) -> ShaderParseError {
    tracing::error!(
        "unknown {family} instruction {instruction}, opcode = 0x{opcode:x} at addr 0x{pc:08x} \
         (hash0 = 0x{:08x}, crc32 = 0x{:08x})\n{}",
        dst.get_hash0(),
        dst.get_crc32(),
        dst.dbg_dump()
    );
    ShaderParseError::NotImplemented {
        family,
        instruction,
        opcode,
        pc,
        raw,
    }
}

/// `KYTY_UNKNOWN_OP` (ShaderParse.cpp L25).
fn unknown_op(
    dst: &ShaderCode,
    family: &'static str,
    opcode: u32,
    pc: u32,
    raw: u32,
) -> ShaderParseError {
    tracing::error!(
        "unknown {family} opcode: 0x{opcode:x} at addr 0x{pc:08x} \
         (hash0 = 0x{:08x}, crc32 = 0x{:08x})\n{}",
        dst.get_hash0(),
        dst.get_crc32(),
        dst.dbg_dump()
    );
    ShaderParseError::UnknownOpcode {
        family,
        opcode,
        pc,
        raw,
    }
}

/// `EXIT_NOT_IMPLEMENTED(cond)` equivalent for encoding features.
fn feature(family: &'static str, feat: &'static str, pc: u32) -> ShaderParseError {
    tracing::error!("not implemented {family} feature: {feat} at addr 0x{pc:08x}");
    ShaderParseError::NotImplementedFeature {
        family,
        feature: feat,
        pc,
    }
}

/// Bounds-checked dword read (Kyty indexes `buffer[i]` unchecked).
fn dw(buffer: &[u32], index: u32, pc: u32) -> Result<u32, ShaderParseError> {
    buffer
        .get(index as usize)
        .copied()
        .ok_or(ShaderParseError::Truncated { pc })
}

/// Kyty: ShaderParse.cpp `operand_parse` (L32) — the canonical operand-code
/// map: 0-103 SGPR, 106/107 VCC_LO/HI, 124 M0, 125 NULL, 126/127 EXEC_LO/HI,
/// 128-208 inline ints, 240-247 inline floats, 252 EXECZ, 255 literal (in
/// next dword), >=256 VGPR. Unknown codes return an error (Kyty EXITs).
pub fn operand_parse(code: u32) -> Result<ShaderOperand, ShaderParseError> {
    let mut ret = ShaderOperand {
        size: 1,
        ..Default::default()
    };

    if code <= 103 {
        ret.type_ = O::Sgpr;
        ret.register_id = code as i32;
    } else if (128..=192).contains(&code) {
        ret.type_ = O::IntegerInlineConstant;
        ret.constant.u = (code as i32 - 128) as u32;
        ret.size = 0;
    } else if (193..=208).contains(&code) {
        ret.type_ = O::IntegerInlineConstant;
        ret.constant.u = (192 - code as i32) as u32;
        ret.size = 0;
    } else if (240..=248).contains(&code) {
        // 248 = 1/(2*pi), the GCN3+/RDNA2 ninth inline float (SharpEmu
        // `Gen5InlineConstants` maps 248 => 1/(2*PI); Kyty predates it and
        // errors). It cannot appear in a legacy SI stream, so accepting it
        // unconditionally is safe. Measured: 58 ASTRO.BOT CS failures
        // ("unknown operand: 248").
        const FV: [f32; 9] = [
            0.5,
            -0.5,
            1.0,
            -1.0,
            2.0,
            -2.0,
            4.0,
            -4.0,
            1.0 / (2.0 * std::f32::consts::PI),
        ];
        ret.type_ = O::FloatInlineConstant;
        ret.constant.u = FV[(code - 240) as usize].to_bits();
        ret.size = 0;
    } else if code >= 256 {
        ret.type_ = O::Vgpr;
        ret.register_id = (code - 256) as i32;
    } else {
        match code {
            106 => ret.type_ = O::VccLo,
            107 => ret.type_ = O::VccHi,
            124 => ret.type_ = O::M0,
            125 => ret.type_ = O::Null,
            126 => ret.type_ = O::ExecLo,
            127 => ret.type_ = O::ExecHi,
            252 => ret.type_ = O::ExecZ,
            255 => {
                ret.type_ = O::LiteralConstant;
                ret.size = 0;
            }
            _ => return Err(ShaderParseError::UnknownOperand { code }),
        }
    }

    Ok(ret)
}

/// Beyond Kyty: is `src0` a DPP (Data-Parallel Primitives) marker? `0xfa`
/// (250) = DPP16; `0xe9`/`0xea` (233/234) = DPP8/DPP8FI. Like SDWA (`0xf9`),
/// a DPP marker means the instruction carries a second control dword and the
/// real src0 (always a VGPR) lives in that dword's low byte. Kyty predates
/// DPP and hands the marker straight to `operand_parse`, which errors and
/// fails the whole shader.
fn is_dpp_marker(src0: u32) -> bool {
    matches!(src0, 0xfa | 0xe9 | 0xea)
}

/// Beyond Kyty: the modifier and control fields pulled from a DPP control
/// dword. `src0` is *not* here — it is `b1 & 0xff` at every call site.
struct DppDecoded {
    ctrl: DppCtrl,
    src0_neg: bool,
    src0_abs: bool,
    src1_neg: bool,
    src1_abs: bool,
}

/// Beyond Kyty: decode a DPP control dword (`b1`) given its src0 `marker`.
/// Layout mirrors shadPS4's `struct Dpp` for DPP16 and its DPP8 lane-select
/// packing (GPL-2.0 — studied, not copied):
///
/// * DPP16 (`marker == 0xfa`): `src0[7:0]`, `dpp_ctrl[16:8]`, `fi[18]`,
///   `bound_ctrl[19]`, `src0_neg[20]`, `src0_abs[21]`, `src1_neg[22]`,
///   `src1_abs[23]`, `bank_mask[27:24]`, `row_mask[31:28]`.
/// * DPP8/DPP8FI (`marker == 0xe9`/`0xea`): `src0[7:0]` then eight 3-bit lane
///   selects filling `[31:8]`; no abs/neg/masks. `0xea` is fetch-inactive.
fn decode_dpp(marker: u32, b1: u32) -> DppDecoded {
    if marker == 0xfa {
        DppDecoded {
            ctrl: DppCtrl {
                mode: DppMode::Dpp16 {
                    ctrl: ((b1 >> 8) & 0x1ff) as u16,
                },
                row_mask: ((b1 >> 28) & 0xf) as u8,
                bank_mask: ((b1 >> 24) & 0xf) as u8,
                bound_ctrl: (b1 >> 19) & 0x1 != 0,
                fetch_inactive: (b1 >> 18) & 0x1 != 0,
            },
            src0_neg: (b1 >> 20) & 0x1 != 0,
            src0_abs: (b1 >> 21) & 0x1 != 0,
            src1_neg: (b1 >> 22) & 0x1 != 0,
            src1_abs: (b1 >> 23) & 0x1 != 0,
        }
    } else {
        let mut lane_sel = [0u8; 8];
        for (i, s) in lane_sel.iter_mut().enumerate() {
            *s = ((b1 >> (8 + i * 3)) & 0x7) as u8;
        }
        DppDecoded {
            ctrl: DppCtrl {
                mode: DppMode::Dpp8 { lane_sel },
                row_mask: 0,
                bank_mask: 0,
                bound_ctrl: false,
                fetch_inactive: marker == 0xea,
            },
            src0_neg: false,
            src0_abs: false,
            src1_neg: false,
            src1_abs: false,
        }
    }
}

/// Kyty: ShaderParse.cpp `shader_parse_sopc` (L84).
fn shader_parse_sopc(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "sopc";
    let b0 = buffer[0];

    let ssrc1 = (b0 >> 8) & 0xff;
    let ssrc0 = b0 & 0xff;
    let opcode = (b0 >> 16) & 0x7f;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.src[0] = operand_parse(ssrc0)?;
    inst.src[1] = operand_parse(ssrc1)?;
    inst.src_num = 2;

    let mut size: u32 = 1;

    if inst.src[0].type_ == O::LiteralConstant {
        inst.src[0].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    if inst.src[1].type_ == O::LiteralConstant {
        inst.src[1].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    inst.format = F::Ssrc0Ssrc1;

    match opcode {
        0x00 => inst.type_ = T::SCmpEqI32,
        0x01 => inst.type_ = T::SCmpLgI32,
        0x02 => inst.type_ = T::SCmpGtI32,
        0x03 => inst.type_ = T::SCmpGeI32,
        0x04 => inst.type_ = T::SCmpLtI32,
        0x05 => inst.type_ = T::SCmpLeI32,
        0x06 => inst.type_ = T::SCmpEqU32,
        0x07 => inst.type_ = T::SCmpLgU32,
        0x08 => inst.type_ = T::SCmpGtU32,
        0x09 => inst.type_ = T::SCmpGeU32,
        0x0a => inst.type_ = T::SCmpLtU32,
        0x0b => inst.type_ = T::SCmpLeU32,
        0x0c => return Err(ni(dst, S, "s_bitcmp0_b32", opcode, pc, b0)),
        0x0d => return Err(ni(dst, S, "s_bitcmp1_b32", opcode, pc, b0)),
        0x0e => return Err(ni(dst, S, "s_bitcmp0_b64", opcode, pc, b0)),
        0x0f => return Err(ni(dst, S, "s_bitcmp1_b64", opcode, pc, b0)),
        0x10 => return Err(ni(dst, S, "s_setvskip", opcode, pc, b0)),
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_sopk` (L147).
fn shader_parse_sopk(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "sopk";
    let b0 = buffer[0];

    let opcode = (b0 >> 23) & 0x1f;
    let imm = (b0 & 0xffff) as u16 as i16;
    let sdst = (b0 >> 16) & 0x7f;

    // RDNA2 reassigns SOPK opcodes 1 and 0x17. Handle them before parsing the
    // reserved `sdst` field as a register. Both still emit an instruction so
    // branch destinations at their PC remain visible to the relooper.
    if next_gen && matches!(opcode, 0x01 | 0x17) {
        let mut inst = ShaderInstruction {
            pc,
            ..Default::default()
        };
        // GFX10 ISA: opcode 1 = s_version (metadata/no execution effect),
        // opcode 0x17 = s_waitcnt_vscnt (vector-store completion wait).
        inst.type_ = if opcode == 0x01 {
            T::SVersion
        } else {
            T::SWaitcnt
        };
        inst.format = F::Imm;
        inst.src[0].type_ = O::LiteralConstant;
        inst.src[0].constant.u = i32::from(imm) as u32;
        inst.src_num = 1;
        dst.get_instructions_mut().push(inst);
        return Ok(1);
    }

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.dst = operand_parse(sdst)?;

    inst.format = F::SVdstSVsrc0;
    inst.src[0].type_ = O::IntegerInlineConstant;
    inst.src[0].constant.u = i32::from(imm) as u32;
    inst.src_num = 1;

    match opcode {
        0x00 => inst.type_ = T::SMovkI32,
        0x02 => return Err(ni(dst, S, "s_cmovk_i32", opcode, pc, b0)),
        0x03 => return Err(ni(dst, S, "s_cmpk_eq_i32", opcode, pc, b0)),
        0x04 => return Err(ni(dst, S, "s_cmpk_lg_i32", opcode, pc, b0)),
        0x05 => return Err(ni(dst, S, "s_cmpk_gt_i32", opcode, pc, b0)),
        0x06 => return Err(ni(dst, S, "s_cmpk_ge_i32", opcode, pc, b0)),
        0x07 => return Err(ni(dst, S, "s_cmpk_lt_i32", opcode, pc, b0)),
        0x08 => return Err(ni(dst, S, "s_cmpk_le_i32", opcode, pc, b0)),
        0x09 => return Err(ni(dst, S, "s_cmpk_eq_u32", opcode, pc, b0)),
        0x0a => return Err(ni(dst, S, "s_cmpk_lg_u32", opcode, pc, b0)),
        0x0b => return Err(ni(dst, S, "s_cmpk_gt_u32", opcode, pc, b0)),
        0x0c => return Err(ni(dst, S, "s_cmpk_ge_u32", opcode, pc, b0)),
        0x0d => return Err(ni(dst, S, "s_cmpk_lt_u32", opcode, pc, b0)),
        0x0e => return Err(ni(dst, S, "s_cmpk_le_u32", opcode, pc, b0)),
        0x0f => return Err(ni(dst, S, "s_addk_i32", opcode, pc, b0)),
        0x10 => inst.type_ = T::SMulkI32,
        0x11 => return Err(ni(dst, S, "s_cbranch_i_fork", opcode, pc, b0)),
        0x12 => return Err(ni(dst, S, "s_getreg_b32", opcode, pc, b0)),
        0x13 => return Err(ni(dst, S, "s_setreg_b32", opcode, pc, b0)),
        0x14 => return Err(ni(dst, S, "s_getreg_regrd_b32", opcode, pc, b0)),
        0x15 => return Err(ni(dst, S, "s_setreg_imm32_b32", opcode, pc, b0)),
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(1)
}

/// Kyty: ShaderParse.cpp `shader_parse_sopp` (L202).
fn shader_parse_sopp(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "sopp";
    let b0 = buffer[0];

    let opcode = (b0 >> 16) & 0x7f;
    let simm = b0 & 0xffff;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };

    inst.format = F::Label;
    inst.src[0].type_ = O::LiteralConstant;
    inst.src[0].constant.u = (i32::from(simm as u16 as i16) * 4) as u32;
    inst.src_num = 1;

    match opcode {
        0x01 => {
            inst.type_ = T::SEndpgm;
            inst.format = F::Empty;
            inst.src_num = 0;
        }
        0x02 => inst.type_ = T::SBranch,
        0x04 => inst.type_ = T::SCbranchScc0,
        0x05 => inst.type_ = T::SCbranchScc1,
        0x06 => inst.type_ = T::SCbranchVccz,
        0x07 => inst.type_ = T::SCbranchVccnz,
        0x08 => inst.type_ = T::SCbranchExecz,
        0x0c => {
            inst.type_ = T::SWaitcnt;
            inst.format = F::Imm;
            inst.src[0].type_ = O::LiteralConstant;
            inst.src[0].constant.u = simm;
            inst.src_num = 1;
        }
        0x10 => {
            inst.type_ = T::SSendmsg;
            inst.format = F::Imm;
            inst.src[0].type_ = O::LiteralConstant;
            inst.src[0].constant.u = simm;
            inst.src_num = 1;
        }
        0x20 => {
            inst.type_ = T::SInstPrefetch;
            inst.format = F::Imm;
            inst.src[0].type_ = O::LiteralConstant;
            inst.src[0].constant.u = simm;
            inst.src_num = 1;
        }
        0x00 => {
            inst.type_ = T::SNop;
            inst.format = F::Imm;
            inst.src[0].type_ = O::LiteralConstant;
            inst.src[0].constant.u = simm;
            inst.src_num = 1;
        }
        0x1f => {
            // RDNA2 `s_code_end` — the padding terminator compilers emit AFTER
            // a shader's real code (measured: ASTRO.BOT, raw 0xbf9f0000). A
            // parser that walks a whole fetched buffer rather than stopping at
            // s_endpgm runs into it; treat it as an end-of-code marker, which
            // is what it is. Without this, parsing a full shader buffer fails
            // and any analysis built on that parse (e.g. the EUD scalar-load
            // base scan) silently gets nothing.
            inst.type_ = T::SEndpgm;
            inst.format = F::Empty;
            inst.src_num = 0;
        }
        0x09 => return Err(ni(dst, S, "s_cbranch_execnz", opcode, pc, b0)),
        // Beyond Kyty (KYTY_NI upstream): workgroup barrier — required by the
        // LDS `ds_write_b32`/`ds_read_b32` pairs (ASTRO.BOT scene compute).
        0x0a => {
            inst.type_ = T::SBarrier;
            inst.format = F::Empty;
            inst.src_num = 0;
        }
        0x0b => return Err(ni(dst, S, "s_setkill", opcode, pc, b0)),
        0x0d => return Err(ni(dst, S, "s_sethalt", opcode, pc, b0)),
        0x0e => return Err(ni(dst, S, "s_sleep", opcode, pc, b0)),
        0x0f => return Err(ni(dst, S, "s_setprio", opcode, pc, b0)),
        0x11 => return Err(ni(dst, S, "s_sendmsghalt", opcode, pc, b0)),
        0x12 => return Err(ni(dst, S, "s_trap", opcode, pc, b0)),
        0x13 => return Err(ni(dst, S, "s_icache_inv", opcode, pc, b0)),
        0x14 => return Err(ni(dst, S, "s_incperflevel", opcode, pc, b0)),
        0x15 => return Err(ni(dst, S, "s_decperflevel", opcode, pc, b0)),
        0x16 => return Err(ni(dst, S, "s_ttracedata", opcode, pc, b0)),
        0x17 => return Err(ni(dst, S, "s_cbranch_cdbgsys", opcode, pc, b0)),
        0x18 => return Err(ni(dst, S, "s_cbranch_cdbguser", opcode, pc, b0)),
        0x19 => return Err(ni(dst, S, "s_cbranch_cdbgsys_or_user", opcode, pc, b0)),
        0x1a => return Err(ni(dst, S, "s_cbranch_cdbgsys_and_user", opcode, pc, b0)),
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    if matches!(
        inst.type_,
        T::SCbranchScc0
            | T::SCbranchScc1
            | T::SCbranchVccz
            | T::SCbranchVccnz
            | T::SCbranchExecz
            | T::SBranch
    ) {
        dst.get_labels_mut()
            .push(ShaderLabel::from_instruction(&inst));

        if inst.type_ != T::SBranch {
            dst.get_indirect_labels_mut()
                .push(ShaderLabel::new(inst.pc + 4, inst.pc));
        }
    }

    Ok(1)
}

/// Kyty: ShaderParse.cpp `shader_parse_sop1` (L295).
fn shader_parse_sop1(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "sop1";
    let b0 = buffer[0];

    let opcode = (b0 >> 8) & 0xff;
    let ssrc0 = b0 & 0xff;
    let sdst = (b0 >> 16) & 0x7f;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.src[0] = operand_parse(ssrc0)?;
    inst.src_num = 1;
    inst.dst = operand_parse(sdst)?;

    let mut size: u32 = 1;

    if inst.src[0].type_ == O::LiteralConstant {
        inst.src[0].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    match opcode {
        0x03 => {
            inst.type_ = T::SMovB32;
            inst.format = F::SVdstSVsrc0;
        }
        0x04 => {
            inst.type_ = T::SMovB64;
            inst.format = F::Sdst2Ssrc02;
            inst.dst.size = 2;
            inst.src[0].size = 2;
        }
        0x05 => return Err(ni(dst, S, "s_cmov_b32", opcode, pc, b0)),
        0x06 => return Err(ni(dst, S, "s_cmov_b64", opcode, pc, b0)),
        0x07 => return Err(ni(dst, S, "s_not_b32", opcode, pc, b0)),
        0x08 => {
            // GCN: D.u64 = ~S0.u64; SCC = (D != 0). Measured in ASTRO.BOT's
            // compute shaders (exec-mask manipulation).
            inst.type_ = T::SNotB64;
            inst.format = F::Sdst2Ssrc02;
            inst.dst.size = 2;
            inst.src[0].size = 2;
        }
        0x09 => return Err(ni(dst, S, "s_wqm_b32", opcode, pc, b0)),
        0x0a => {
            inst.type_ = T::SWqmB64;
            inst.format = F::Sdst2Ssrc02;
            inst.dst.size = 2;
            inst.src[0].size = 2;
        }
        0x0b => {
            // GCN: D.u = bitreverse(S0.u). Does NOT write SCC (unlike the
            // s_not/s_and family) — see the S::None in its recompile entry.
            inst.type_ = T::SBrevB32;
            inst.format = F::SVdstSVsrc0;
        }
        0x0c => return Err(ni(dst, S, "s_brev_b64", opcode, pc, b0)),
        0x0d => return Err(ni(dst, S, "s_bcnt0_i32_b32", opcode, pc, b0)),
        0x0e => return Err(ni(dst, S, "s_bcnt0_i32_b64", opcode, pc, b0)),
        0x0f => return Err(ni(dst, S, "s_bcnt1_i32_b32", opcode, pc, b0)),
        0x10 => return Err(ni(dst, S, "s_bcnt1_i32_b64", opcode, pc, b0)),
        0x11 => return Err(ni(dst, S, "s_ff0_i32_b32", opcode, pc, b0)),
        0x12 => return Err(ni(dst, S, "s_ff0_i32_b64", opcode, pc, b0)),
        0x13 => return Err(ni(dst, S, "s_ff1_i32_b32", opcode, pc, b0)),
        0x14 => return Err(ni(dst, S, "s_ff1_i32_b64", opcode, pc, b0)),
        0x15 => return Err(ni(dst, S, "s_flbit_i32_b32", opcode, pc, b0)),
        0x16 => return Err(ni(dst, S, "s_flbit_i32_b64", opcode, pc, b0)),
        0x17 => return Err(ni(dst, S, "s_flbit_i32", opcode, pc, b0)),
        0x18 => return Err(ni(dst, S, "s_flbit_i32_i64", opcode, pc, b0)),
        0x19 => return Err(ni(dst, S, "s_sext_i32_i8", opcode, pc, b0)),
        0x1a => return Err(ni(dst, S, "s_sext_i32_i16", opcode, pc, b0)),
        0x1b => return Err(ni(dst, S, "s_bitset0_b32", opcode, pc, b0)),
        0x1c => return Err(ni(dst, S, "s_bitset0_b64", opcode, pc, b0)),
        0x1d => return Err(ni(dst, S, "s_bitset1_b32", opcode, pc, b0)),
        0x1e => return Err(ni(dst, S, "s_bitset1_b64", opcode, pc, b0)),
        0x1f => {
            // The hardware returns the absolute address of the instruction
            // following S_GETPC_B64. The C++ path implicitly retained this in
            // its raw code pointer; the Rust model carries the guest base on
            // ShaderCode so high address bits are not lost.
            let following = dst
                .get_base_address()
                .wrapping_add(u64::from(pc))
                .wrapping_add(4);
            inst.type_ = T::SGetpcB64;
            inst.format = F::Sdst2;
            inst.dst.size = 2;
            inst.src[0] = ShaderOperand {
                type_: O::LiteralConstant,
                constant: ShaderConstant::from_u(following as u32),
                size: 1,
                ..Default::default()
            };
            inst.src[1] = ShaderOperand {
                type_: O::LiteralConstant,
                constant: ShaderConstant::from_u((following >> 32) as u32),
                size: 1,
                ..Default::default()
            };
            inst.src_num = 2;
        }
        0x20 => {
            inst.type_ = T::SSetpcB64;
            inst.format = F::Saddr;
            inst.src[0].size = 2;
        }
        0x21 => {
            inst.type_ = T::SSwappcB64;
            inst.format = F::Sdst2Ssrc02;
            inst.src[0].size = 2;
            inst.dst.size = 2;
        }
        0x22 => return Err(ni(dst, S, "s_rfe_b64", opcode, pc, b0)),
        0x24 => {
            inst.type_ = T::SAndSaveexecB64;
            inst.format = F::Sdst2Ssrc02;
            inst.dst.size = 2;
            inst.src[0].size = 2;
        }
        0x25 => return Err(ni(dst, S, "s_or_saveexec_b64", opcode, pc, b0)),
        0x26 => return Err(ni(dst, S, "s_xor_saveexec_b64", opcode, pc, b0)),
        0x27 => return Err(ni(dst, S, "s_andn2_saveexec_b64", opcode, pc, b0)),
        0x28 => {
            inst.type_ = T::SOrn2SaveexecB64;
            inst.format = F::Sdst2Ssrc02;
            inst.dst.size = 2;
            inst.src[0].size = 2;
        }
        0x29 => return Err(ni(dst, S, "s_nand_saveexec_b64", opcode, pc, b0)),
        0x2a => return Err(ni(dst, S, "s_nor_saveexec_b64", opcode, pc, b0)),
        0x2b => return Err(ni(dst, S, "s_xnor_saveexec_b64", opcode, pc, b0)),
        0x2c => return Err(ni(dst, S, "s_quadmask_b32", opcode, pc, b0)),
        0x2d => return Err(ni(dst, S, "s_quadmask_b64", opcode, pc, b0)),
        0x2e => return Err(ni(dst, S, "s_movrels_b32", opcode, pc, b0)),
        0x2f => return Err(ni(dst, S, "s_movrels_b64", opcode, pc, b0)),
        0x30 => return Err(ni(dst, S, "s_movreld_b32", opcode, pc, b0)),
        0x31 => return Err(ni(dst, S, "s_movreld_b64", opcode, pc, b0)),
        0x32 => return Err(ni(dst, S, "s_cbranch_join", opcode, pc, b0)),
        0x33 => return Err(ni(dst, S, "s_mov_regrd_b32", opcode, pc, b0)),
        0x34 => return Err(ni(dst, S, "s_abs_i32", opcode, pc, b0)),
        0x35 => return Err(ni(dst, S, "s_mov_fed_b32", opcode, pc, b0)),
        // RDNA2 (`next_gen`) SOP1 0x37: s_andn1_saveexec_b64 (SharpEmu Gen5
        // L710). `sdst = exec; exec = ~ssrc0 & exec`. Same 64-bit save-exec
        // shape as 0x24/0x28.
        0x37 => {
            inst.type_ = T::SAndn1SaveexecB64;
            inst.format = F::Sdst2Ssrc02;
            inst.dst.size = 2;
            inst.src[0].size = 2;
        }
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_sop2` (L410). SOP1/SOPC/SOPP/SOPK are
/// nested inside this decoder (opcode 0x7d/0x7e/0x7f/>=0x60).
fn shader_parse_sop2(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "sop2";
    let b0 = buffer[0];

    let opcode = (b0 >> 23) & 0x7f;

    match opcode {
        0x7d => return shader_parse_sop1(pc, buffer, dst, next_gen),
        0x7e => return shader_parse_sopc(pc, buffer, dst, next_gen),
        0x7f => return shader_parse_sopp(pc, buffer, dst, next_gen),
        _ => {}
    }

    if opcode >= 0x60 {
        return shader_parse_sopk(pc, buffer, dst, next_gen);
    }

    let ssrc1 = (b0 >> 8) & 0xff;
    let ssrc0 = b0 & 0xff;
    let sdst = (b0 >> 16) & 0x7f;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.src[0] = operand_parse(ssrc0)?;
    inst.src[1] = operand_parse(ssrc1)?;
    inst.src_num = 2;
    inst.dst = operand_parse(sdst)?;

    let mut size: u32 = 1;

    if inst.src[0].type_ == O::LiteralConstant {
        inst.src[0].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    if inst.src[1].type_ == O::LiteralConstant {
        inst.src[1].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    inst.format = F::SVdstSVsrc0SVsrc1;

    // Sdst2Ssrc02Ssrc12 helper (Kyty repeats this block per 64-bit opcode).
    let b64_full = |inst: &mut ShaderInstruction, t: T| {
        inst.type_ = t;
        inst.format = F::Sdst2Ssrc02Ssrc12;
        inst.dst.size = 2;
        inst.src[0].size = 2;
        inst.src[1].size = 2;
    };
    // Sdst2Ssrc02Ssrc1 helper (shift-style 64-bit ops).
    let b64_shift = |inst: &mut ShaderInstruction, t: T| {
        inst.type_ = t;
        inst.format = F::Sdst2Ssrc02Ssrc1;
        inst.dst.size = 2;
        inst.src[0].size = 2;
    };

    match opcode {
        0x00 => inst.type_ = T::SAddU32,
        0x01 => inst.type_ = T::SSubU32,
        0x02 => inst.type_ = T::SAddI32,
        0x03 => inst.type_ = T::SSubI32,
        0x04 => inst.type_ = T::SAddcU32,
        0x05 => return Err(ni(dst, S, "s_subb_u32", opcode, pc, b0)),
        0x06 => return Err(ni(dst, S, "s_min_i32", opcode, pc, b0)),
        0x07 => return Err(ni(dst, S, "s_min_u32", opcode, pc, b0)),
        0x08 => return Err(ni(dst, S, "s_max_i32", opcode, pc, b0)),
        0x09 => return Err(ni(dst, S, "s_max_u32", opcode, pc, b0)),
        0x0a => inst.type_ = T::SCselectB32,
        0x0b => b64_full(&mut inst, T::SCselectB64),
        0x0e => inst.type_ = T::SAndB32,
        0x0f => b64_full(&mut inst, T::SAndB64),
        0x10 => inst.type_ = T::SOrB32,
        0x11 => b64_full(&mut inst, T::SOrB64),
        0x12 => return Err(ni(dst, S, "s_xor_b32", opcode, pc, b0)),
        0x13 => b64_full(&mut inst, T::SXorB64),
        0x14 => return Err(ni(dst, S, "s_andn2_b32", opcode, pc, b0)),
        0x15 => b64_full(&mut inst, T::SAndn2B64),
        0x16 => return Err(ni(dst, S, "s_orn2_b32", opcode, pc, b0)),
        0x17 => b64_full(&mut inst, T::SOrn2B64),
        0x18 => return Err(ni(dst, S, "s_nand_b32", opcode, pc, b0)),
        0x19 => b64_full(&mut inst, T::SNandB64),
        0x1a => return Err(ni(dst, S, "s_nor_b32", opcode, pc, b0)),
        0x1b => b64_full(&mut inst, T::SNorB64),
        0x1c => return Err(ni(dst, S, "s_xnor_b32", opcode, pc, b0)),
        0x1d => b64_full(&mut inst, T::SXnorB64),
        0x1e => inst.type_ = T::SLshlB32,
        0x1f => b64_shift(&mut inst, T::SLshlB64),
        0x20 => inst.type_ = T::SLshrB32,
        0x21 => b64_shift(&mut inst, T::SLshrB64),
        0x22 => return Err(ni(dst, S, "s_ashr_i32", opcode, pc, b0)),
        0x23 => return Err(ni(dst, S, "s_ashr_i64", opcode, pc, b0)),
        0x24 => inst.type_ = T::SBfmB32,
        0x25 => return Err(ni(dst, S, "s_bfm_b64", opcode, pc, b0)),
        0x26 => inst.type_ = T::SMulI32,
        0x27 => inst.type_ = T::SBfeU32,
        0x28 => return Err(ni(dst, S, "s_bfe_i32", opcode, pc, b0)),
        0x29 => b64_shift(&mut inst, T::SBfeU64),
        0x2a => return Err(ni(dst, S, "s_bfe_i64", opcode, pc, b0)),
        0x2b => return Err(ni(dst, S, "s_cbranch_g_fork", opcode, pc, b0)),
        0x2c => return Err(ni(dst, S, "s_absdiff_i32", opcode, pc, b0)),
        0x31 => {
            // Kyty L575: EXIT_NOT_IMPLEMENTED(!next_gen).
            if !next_gen {
                return Err(feature(S, "s_lshl4_add_u32 requires next_gen", pc));
            }
            inst.type_ = T::SLshl4AddU32;
        }
        0x32 => inst.type_ = T::SPackLlB32B16,
        0x33 => return Err(ni(dst, S, "s_pack_lh_b32_b16", opcode, pc, b0)),
        0x34 => return Err(ni(dst, S, "s_pack_hh_b32_b16", opcode, pc, b0)),
        0x35 => inst.type_ = T::SMulHiU32,
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Names of VOPC-family compare opcodes that Kyty knows by name but does not
/// implement. Kyty repeats this 256-entry table twice — in `shader_parse_vopc`
/// (ShaderParse.cpp L655-908) and again for VOPC-via-VOP3 (L1472-1725); the
/// port shares one lookup. 0x11/0x14 are here because the VOP3 copy NIs them
/// (L1489/L1492) while plain VOPC implements them — implemented match arms
/// take precedence over this table in both callers.
fn vopc_ni_name(opcode: u32) -> Option<&'static str> {
    Some(match opcode {
        0x10 => "v_cmpx_f_f32",
        0x11 => "v_cmpx_lt_f32",
        0x12 => "v_cmpx_eq_f32",
        0x13 => "v_cmpx_le_f32",
        0x14 => "v_cmpx_gt_f32",
        0x15 => "v_cmpx_lg_f32",
        0x16 => "v_cmpx_ge_f32",
        0x17 => "v_cmpx_o_f32",
        0x18 => "v_cmpx_u_f32",
        0x19 => "v_cmpx_nge_f32",
        0x1a => "v_cmpx_nlg_f32",
        0x1b => "v_cmpx_ngt_f32",
        0x1c => "v_cmpx_nle_f32",
        0x1e => "v_cmpx_nlt_f32",
        0x1f => "v_cmpx_tru_f32",
        0x20 => "v_cmp_f_f64",
        0x21 => "v_cmp_lt_f64",
        0x22 => "v_cmp_eq_f64",
        0x23 => "v_cmp_le_f64",
        0x24 => "v_cmp_gt_f64",
        0x25 => "v_cmp_lg_f64",
        0x26 => "v_cmp_ge_f64",
        0x27 => "v_cmp_o_f64",
        0x28 => "v_cmp_u_f64",
        0x29 => "v_cmp_nge_f64",
        0x2a => "v_cmp_nlg_f64",
        0x2b => "v_cmp_ngt_f64",
        0x2c => "v_cmp_nle_f64",
        0x2d => "v_cmp_neq_f64",
        0x2e => "v_cmp_nlt_f64",
        0x2f => "v_cmp_tru_f64",
        0x30 => "v_cmpx_f_f64",
        0x31 => "v_cmpx_lt_f64",
        0x32 => "v_cmpx_eq_f64",
        0x33 => "v_cmpx_le_f64",
        0x34 => "v_cmpx_gt_f64",
        0x35 => "v_cmpx_lg_f64",
        0x36 => "v_cmpx_ge_f64",
        0x37 => "v_cmpx_o_f64",
        0x38 => "v_cmpx_u_f64",
        0x39 => "v_cmpx_nge_f64",
        0x3a => "v_cmpx_nlg_f64",
        0x3b => "v_cmpx_ngt_f64",
        0x3c => "v_cmpx_nle_f64",
        0x3d => "v_cmpx_neq_f64",
        0x3e => "v_cmpx_nlt_f64",
        0x3f => "v_cmpx_tru_f64",
        0x40 => "v_cmps_f_f32",
        0x41 => "v_cmps_lt_f32",
        0x42 => "v_cmps_eq_f32",
        0x43 => "v_cmps_le_f32",
        0x44 => "v_cmps_gt_f32",
        0x45 => "v_cmps_lg_f32",
        0x46 => "v_cmps_ge_f32",
        0x47 => "v_cmps_o_f32",
        0x48 => "v_cmps_u_f32",
        0x49 => "v_cmps_nge_f32",
        0x4a => "v_cmps_nlg_f32",
        0x4b => "v_cmps_ngt_f32",
        0x4c => "v_cmps_nle_f32",
        0x4d => "v_cmps_neq_f32",
        0x4e => "v_cmps_nlt_f32",
        0x4f => "v_cmps_tru_f32",
        0x50 => "v_cmpsx_f_f32",
        0x51 => "v_cmpsx_lt_f32",
        0x52 => "v_cmpsx_eq_f32",
        0x53 => "v_cmpsx_le_f32",
        0x54 => "v_cmpsx_gt_f32",
        0x55 => "v_cmpsx_lg_f32",
        0x56 => "v_cmpsx_ge_f32",
        0x57 => "v_cmpsx_o_f32",
        0x58 => "v_cmpsx_u_f32",
        0x59 => "v_cmpsx_nge_f32",
        0x5a => "v_cmpsx_nlg_f32",
        0x5b => "v_cmpsx_ngt_f32",
        0x5c => "v_cmpsx_nle_f32",
        0x5d => "v_cmpsx_neq_f32",
        0x5e => "v_cmpsx_nlt_f32",
        0x5f => "v_cmpsx_tru_f32",
        0x60 => "v_cmps_f_f64",
        0x61 => "v_cmps_lt_f64",
        0x62 => "v_cmps_eq_f64",
        0x63 => "v_cmps_le_f64",
        0x64 => "v_cmps_gt_f64",
        0x65 => "v_cmps_lg_f64",
        0x66 => "v_cmps_ge_f64",
        0x67 => "v_cmps_o_f64",
        0x68 => "v_cmps_u_f64",
        0x69 => "v_cmps_nge_f64",
        0x6a => "v_cmps_nlg_f64",
        0x6b => "v_cmps_ngt_f64",
        0x6c => "v_cmps_nle_f64",
        0x6d => "v_cmps_neq_f64",
        0x6e => "v_cmps_nlt_f64",
        0x6f => "v_cmps_tru_f64",
        0x70 => "v_cmpsx_f_f64",
        0x71 => "v_cmpsx_lt_f64",
        0x72 => "v_cmpsx_eq_f64",
        0x73 => "v_cmpsx_le_f64",
        0x74 => "v_cmpsx_gt_f64",
        0x75 => "v_cmpsx_lg_f64",
        0x76 => "v_cmpsx_ge_f64",
        0x77 => "v_cmpsx_o_f64",
        0x78 => "v_cmpsx_u_f64",
        0x79 => "v_cmpsx_nge_f64",
        0x7a => "v_cmpsx_nlg_f64",
        0x7b => "v_cmpsx_ngt_f64",
        0x7c => "v_cmpsx_nle_f64",
        0x7d => "v_cmpsx_neq_f64",
        0x7e => "v_cmpsx_nlt_f64",
        0x7f => "v_cmpsx_tru_f64",
        0x88 => "v_cmp_class_f32",
        0x89 => "v_cmp_lt_i16",
        0x8a => "v_cmp_eq_i16",
        0x8b => "v_cmp_le_i16",
        0x8c => "v_cmp_gt_i16",
        0x8d => "v_cmp_ne_i16",
        0x8e => "v_cmp_ge_i16",
        0x8f => "v_cmp_class_f16",
        0x90 => "v_cmpx_f_i32",
        0x91 => "v_cmpx_lt_i32",
        0x92 => "v_cmpx_eq_i32",
        0x93 => "v_cmpx_le_i32",
        0x94 => "v_cmpx_gt_i32",
        0x95 => "v_cmpx_ne_i32",
        0x96 => "v_cmpx_ge_i32",
        0x97 => "v_cmpx_t_i32",
        0x98 => "v_cmpx_class_f32",
        0x99 => "v_cmpx_lt_i16",
        0x9a => "v_cmpx_eq_i16",
        0x9b => "v_cmpx_le_i16",
        0x9c => "v_cmpx_gt_i16",
        0x9d => "v_cmpx_ne_i16",
        0x9e => "v_cmpx_ge_i16",
        0x9f => "v_cmpx_class_f16",
        0xa0 => "v_cmp_f_i64",
        0xa1 => "v_cmp_lt_i64",
        0xa2 => "v_cmp_eq_i64",
        0xa3 => "v_cmp_le_i64",
        0xa4 => "v_cmp_gt_i64",
        0xa5 => "v_cmp_ne_i64",
        0xa6 => "v_cmp_ge_i64",
        0xa7 => "v_cmp_t_i64",
        0xa8 => "v_cmp_class_f64",
        0xa9 => "v_cmp_lt_u16",
        0xaa => "v_cmp_eq_u16",
        0xab => "v_cmp_le_u16",
        0xac => "v_cmp_gt_u16",
        0xad => "v_cmp_ne_u16",
        0xae => "v_cmp_ge_u16",
        0xb0 => "v_cmpx_f_i64",
        0xb1 => "v_cmpx_lt_i64",
        0xb2 => "v_cmpx_eq_i64",
        0xb3 => "v_cmpx_le_i64",
        0xb4 => "v_cmpx_gt_i64",
        0xb5 => "v_cmpx_ne_i64",
        0xb6 => "v_cmpx_ge_i64",
        0xb7 => "v_cmpx_t_i64",
        0xb8 => "v_cmpx_class_f64",
        0xb9 => "v_cmpx_lt_u16",
        0xba => "v_cmpx_eq_u16",
        0xbb => "v_cmpx_le_u16",
        0xbc => "v_cmpx_gt_u16",
        0xbd => "v_cmpx_ne_u16",
        0xbe => "v_cmpx_ge_u16",
        0xc8 => "v_cmp_f_f16",
        0xc9 => "v_cmp_lt_f16",
        0xca => "v_cmp_eq_f16",
        0xcb => "v_cmp_le_f16",
        0xcc => "v_cmp_gt_f16",
        0xcd => "v_cmp_lg_f16",
        0xce => "v_cmp_ge_f16",
        0xcf => "v_cmp_o_f16",
        0xd0 => "v_cmpx_f_u32",
        0xd1 => "v_cmpx_lt_u32",
        0xd3 => "v_cmpx_le_u32",
        0xd7 => "v_cmpx_t_u32",
        0xd8 => "v_cmpx_f_f16",
        0xd9 => "v_cmpx_lt_f16",
        0xda => "v_cmpx_eq_f16",
        0xdb => "v_cmpx_le_f16",
        0xdc => "v_cmpx_gt_f16",
        0xdd => "v_cmpx_lg_f16",
        0xde => "v_cmpx_ge_f16",
        0xdf => "v_cmpx_o_f16",
        0xe0 => "v_cmp_f_u64",
        0xe1 => "v_cmp_lt_u64",
        0xe2 => "v_cmp_eq_u64",
        0xe3 => "v_cmp_le_u64",
        0xe4 => "v_cmp_gt_u64",
        0xe5 => "v_cmp_ne_u64",
        0xe6 => "v_cmp_ge_u64",
        0xe7 => "v_cmp_t_u64",
        0xe8 => "v_cmp_u_f16",
        0xe9 => "v_cmp_nge_f16",
        0xea => "v_cmp_nlg_f16",
        0xeb => "v_cmp_ngt_f16",
        0xec => "v_cmp_nle_f16",
        0xed => "v_cmp_neq_f16",
        0xee => "v_cmp_nlt_f16",
        0xef => "v_cmp_tru_f16",
        0xf0 => "v_cmpx_f_u64",
        0xf1 => "v_cmpx_lt_u64",
        0xf2 => "v_cmpx_eq_u64",
        0xf3 => "v_cmpx_le_u64",
        0xf4 => "v_cmpx_gt_u64",
        0xf5 => "v_cmpx_ne_u64",
        0xf6 => "v_cmpx_ge_u64",
        0xf7 => "v_cmpx_t_u64",
        0xf8 => "v_cmpx_u_f16",
        0xf9 => "v_cmpx_nge_f16",
        0xfa => "v_cmpx_nlg_f16",
        0xfb => "v_cmpx_ngt_f16",
        0xfc => "v_cmpx_nle_f16",
        0xfd => "v_cmpx_neq_f16",
        0xfe => "v_cmpx_nlt_f16",
        0xff => "v_cmpx_tru_f16",
        _ => return None,
    })
}

/// Kyty: ShaderParse.cpp `shader_parse_vopc` (L592).
fn shader_parse_vopc(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "vopc";
    let b0 = buffer[0];

    let opcode = (b0 >> 17) & 0xff;
    let mut src0 = b0 & 0x1ff;
    let vsrc1 = (b0 >> 9) & 0xff;

    let sdwa = src0 == 249;
    // DPP (src0 == 0xfa/0xe9/0xea): mutually exclusive with SDWA; also a
    // two-dword form whose real src0 (a VGPR) is `b1 & 0xff`. See `decode_dpp`.
    let dpp = is_dpp_marker(src0);

    let mut size: u32 = if sdwa || dpp { 2 } else { 1 };
    let b1 = if sdwa || dpp { dw(buffer, 1, pc)? } else { 0 };
    let dpp_dec = if dpp {
        Some(decode_dpp(src0, b1))
    } else {
        None
    };

    src0 = if sdwa || dpp { b1 & 0xff } else { src0 };
    let sdst = if sdwa { (b1 >> 8) & 0x7f } else { 0 };
    let sd = if sdwa { (b1 >> 15) & 0x1 } else { 0 };
    let src0_sel = if sdwa { (b1 >> 16) & 0x7 } else { 6 };
    let src0_sext = if sdwa { (b1 >> 19) & 0x1 } else { 0 };
    let src0_neg = if sdwa { (b1 >> 20) & 0x1 } else { 0 };
    let src0_abs = if sdwa { (b1 >> 21) & 0x1 } else { 0 };
    let s0 = if sdwa { (b1 >> 23) & 0x1 } else { 1 };
    let src1_sel = if sdwa { (b1 >> 24) & 0x7 } else { 6 };
    let src1_sext = if sdwa { (b1 >> 27) & 0x1 } else { 0 };
    let src1_neg = if sdwa { (b1 >> 28) & 0x1 } else { 0 };
    let src1_abs = if sdwa { (b1 >> 29) & 0x1 } else { 0 };
    let s1 = if sdwa { (b1 >> 31) & 0x1 } else { 0 };

    // Kyty L622-629: EXIT_NOT_IMPLEMENTED on any SDWA modifier. Beyond Kyty:
    // float abs/neg ride the operand modifiers into `operand_load_float`
    // (measured: Minecraft's menu VS does `v_cmp_lt_f32 s2, |v2|, c` and the
    // UI PS does `v_mul_f32 v2, v4, -v3`), and sub-dword source selects
    // (0-3 = BYTE_0..BYTE_3, 4-5 = WORD_0..WORD_1) ride
    // `ShaderOperand::lane_sel` into the operand loaders, which extract the
    // lane with shift + mask (measured: ASTRO.BOT scene compute, vopc
    // src1_sel). Sign extension stays a named refusal — the loaders
    // zero-extend.
    if src0_sel > 6 || src1_sel > 6 {
        return Err(feature(S, "sdwa src_sel == 7 (reserved)", pc));
    }
    if src0_sext != 0 {
        return Err(feature(S, "sdwa src0_sext != 0", pc));
    }
    if src1_sext != 0 {
        return Err(feature(S, "sdwa src1_sext != 0", pc));
    }

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    // DPP src0 is always a VGPR (`s0` is an SDWA-only field, 1 here).
    inst.src[0] = operand_parse(src0 + if s0 == 0 || dpp { 256 } else { 0 })?;
    inst.src[1] = operand_parse(vsrc1 + if s1 == 0 { 256 } else { 0 })?;
    inst.src_num = 2;

    inst.src[0].absolute = src0_abs != 0;
    inst.src[1].absolute = src1_abs != 0;
    inst.src[0].negate = src0_neg != 0;
    inst.src[1].negate = src1_neg != 0;
    inst.src[0].lane_sel = src0_sel as u8;
    inst.src[1].lane_sel = src1_sel as u8;

    // DPP overrides: the control dword carries its own src0/src1 abs/neg and the
    // cross-lane pattern (attached to src0). DPP8 has none, so this is a no-op
    // for its modifiers.
    if let Some(d) = &dpp_dec {
        inst.src[0].absolute = d.src0_abs;
        inst.src[0].negate = d.src0_neg;
        inst.src[1].absolute = d.src1_abs;
        inst.src[1].negate = d.src1_neg;
        inst.src[0].dpp = Some(d.ctrl);
    }

    if inst.src[0].type_ == O::LiteralConstant {
        inst.src[0].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    inst.format = F::SmaskVsrc0Vsrc1;
    if sd == 0 {
        inst.dst.type_ = O::VccLo;
    } else {
        inst.dst = operand_parse(sdst)?;
    }
    inst.dst.size = 2;

    match opcode {
        0x00 => inst.type_ = T::VCmpFF32,
        0x01 => inst.type_ = T::VCmpLtF32,
        0x02 => inst.type_ = T::VCmpEqF32,
        0x03 => inst.type_ = T::VCmpLeF32,
        0x04 => inst.type_ = T::VCmpGtF32,
        0x05 => inst.type_ = T::VCmpLgF32,
        0x06 => inst.type_ = T::VCmpGeF32,
        0x07 => inst.type_ = T::VCmpOF32,
        0x08 => inst.type_ = T::VCmpUF32,
        0x09 => inst.type_ = T::VCmpNgeF32,
        0x0a => inst.type_ = T::VCmpNlgF32,
        0x0b => inst.type_ = T::VCmpNgtF32,
        0x0c => inst.type_ = T::VCmpNleF32,
        0x0d => inst.type_ = T::VCmpNeqF32,
        0x0e => inst.type_ = T::VCmpNltF32,
        0x0f => inst.type_ = T::VCmpTruF32,
        0x11 => inst.type_ = T::VCmpxLtF32,
        // VOPC cmpx block mirrors cmp at +0x10, so 0x12 is v_cmpx_eq_f32
        // (measured: ASTRO.BOT compute).
        0x12 => inst.type_ = T::VCmpxEqF32,
        0x13 => inst.type_ = T::VCmpxLeF32,
        0x14 => inst.type_ = T::VCmpxGtF32,
        0x16 => inst.type_ = T::VCmpxGeF32,
        0x19 => inst.type_ = T::VCmpxNgeF32,
        0x1c => inst.type_ = T::VCmpxNleF32,
        0x1d => inst.type_ = T::VCmpxNeqF32,
        0x1e => inst.type_ = T::VCmpxNltF32,
        0x80 => inst.type_ = T::VCmpFI32,
        0x81 => inst.type_ = T::VCmpLtI32,
        0x82 => inst.type_ = T::VCmpEqI32,
        0x83 => inst.type_ = T::VCmpLeI32,
        0x84 => inst.type_ = T::VCmpGtI32,
        0x85 => inst.type_ = T::VCmpNeI32,
        0x86 => inst.type_ = T::VCmpGeI32,
        0x87 => inst.type_ = T::VCmpTI32,
        // 0x9x is the `v_cmpx_*_i32` block — the signed twin of the 0xdx
        // (`v_cmpx_*_u32`) block below. The whole block was missing, so each
        // instruction decoded as unknown and every draw binding that shader was
        // skipped. Measured in Minecraft, which reaches shaders using
        // `v_cmpx_lt_i32` (0x91) and `v_cmpx_ge_i32` (0x96) once boot gets far
        // enough; the rest of the block is wired at the same time because it is
        // the same lowering and the title decides one opcode at a time.
        0x91 => inst.type_ = T::VCmpxLtI32,
        0x92 => inst.type_ = T::VCmpxEqI32,
        // 0x93 has no unsigned twin (there is no 0xd3), so it is easy to miss
        // when mirroring that block — Minecraft emits it.
        0x93 => inst.type_ = T::VCmpxLeI32,
        0x94 => inst.type_ = T::VCmpxGtI32,
        0x95 => inst.type_ = T::VCmpxNeI32,
        0x96 => inst.type_ = T::VCmpxGeI32,
        0xc0 => inst.type_ = T::VCmpFU32,
        0xc1 => inst.type_ = T::VCmpLtU32,
        0xc2 => inst.type_ = T::VCmpEqU32,
        0xc3 => inst.type_ = T::VCmpLeU32,
        0xc4 => inst.type_ = T::VCmpGtU32,
        0xc5 => inst.type_ = T::VCmpNeU32,
        0xc6 => inst.type_ = T::VCmpGeU32,
        0xc7 => inst.type_ = T::VCmpTU32,
        0xd1 => inst.type_ = T::VCmpxLtU32,
        0xd2 => inst.type_ = T::VCmpxEqU32,
        0xd4 => inst.type_ = T::VCmpxGtU32,
        0xd5 => inst.type_ = T::VCmpxNeU32,
        0xd6 => inst.type_ = T::VCmpxGeU32,
        _ => {
            if let Some(name) = vopc_ni_name(opcode) {
                return Err(ni(dst, S, name, opcode, pc, b0));
            }
            return Err(unknown_op(dst, S, opcode, pc, b0));
        }
    }

    // RDNA/GFX10 VCMPX is EXEC-only. Retaining the legacy VCC destination
    // corrupts live scalar data: Minecraft deliberately loads width/height
    // into VCC and performs two consecutive VCMPX bounds checks, reading
    // VCC_HI and then VCC_LO. SharpEmu and KytyPS5's native Gen5 decoder both
    // route these opcodes to EXEC_LO.
    if next_gen && is_vcmpx_instruction(inst.type_) {
        inst.dst.type_ = O::ExecLo;
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_vop1` (L919).
fn shader_parse_vop1(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "vop1";
    let b0 = buffer[0];

    let vdst = (b0 >> 17) & 0xff;
    let mut src0 = b0 & 0x1ff;
    let opcode = (b0 >> 9) & 0xff;

    // SDWA (src0 == 249): a second dword carries the real src0 plus sub-dword
    // select and abs/neg/clamp/omod modifiers. VOP2 and VOPC decode this, but
    // VOP1 did not — so `operand_parse` was handed the 249 marker directly and
    // failed the whole shader. Measured: Minecraft's UI PS emits
    // `v_rcp_f32 v1, |v5|` (SDWA abs) this way. Mirrors the VOP2 SDWA block
    // (single source: no src1 fields).
    let sdwa = src0 == 249;
    // DPP (src0 == 0xfa/0xe9/0xea): a two-dword form like SDWA, but the second
    // dword is a cross-lane control (single-source here — no src1). See
    // `decode_dpp`; mirrors the VOP2 DPP block below.
    let dpp = is_dpp_marker(src0);
    let mut size: u32 = if sdwa || dpp { 2 } else { 1 };
    let b1 = if sdwa || dpp { dw(buffer, 1, pc)? } else { 0 };
    let dpp_dec = if dpp {
        Some(decode_dpp(src0, b1))
    } else {
        None
    };
    src0 = if sdwa || dpp { b1 & 0xff } else { src0 };
    let dst_sel = if sdwa { (b1 >> 8) & 0x7 } else { 6 };
    let dst_u = if sdwa { (b1 >> 11) & 0x3 } else { 2 };
    let clmp = if sdwa { (b1 >> 13) & 0x1 } else { 0 };
    let omod = if sdwa { (b1 >> 14) & 0x3 } else { 0 };
    let src0_sel = if sdwa { (b1 >> 16) & 0x7 } else { 6 };
    let src0_sext = if sdwa { (b1 >> 19) & 0x1 } else { 0 };
    let src0_neg = if sdwa { (b1 >> 20) & 0x1 } else { 0 };
    let src0_abs = if sdwa { (b1 >> 21) & 0x1 } else { 0 };
    let s0 = if sdwa { (b1 >> 23) & 0x1 } else { 1 };

    if dst_sel != 6 {
        return Err(feature(S, "sdwa dst_sel != 6", pc));
    }
    if sdwa && dst_sel == 6 && dst_u != 0 {
        return Err(feature(S, "sdwa dst_u != 0", pc));
    }
    // Beyond Kyty: sub-dword src0 selects ride `ShaderOperand::lane_sel`
    // into the operand loaders (measured: ASTRO.BOT scene compute, vop1
    // src0_sel). Sign extension stays named — the loaders zero-extend.
    if src0_sel > 6 {
        return Err(feature(S, "sdwa src0_sel == 7 (reserved)", pc));
    }
    if src0_sext != 0 {
        return Err(feature(S, "sdwa src0_sext != 0", pc));
    }

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    // DPP src0 is always a VGPR (`s0` is an SDWA-only field, 1 here).
    inst.src[0] = operand_parse(src0 + if s0 == 0 || dpp { 256 } else { 0 })?;
    inst.dst = operand_parse(vdst + 256)?;
    inst.src_num = 1;

    if inst.src[0].type_ == O::LiteralConstant {
        inst.src[0].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    inst.src[0].absolute = src0_abs != 0;
    inst.src[0].negate = src0_neg != 0;
    inst.src[0].lane_sel = src0_sel as u8;
    // DPP overrides: control-dword src0 abs/neg (DPP8 has none) plus the
    // cross-lane pattern on src0. VOP1 is single-source, so src1 is untouched.
    if let Some(d) = &dpp_dec {
        inst.src[0].absolute = d.src0_abs;
        inst.src[0].negate = d.src0_neg;
        inst.src[0].dpp = Some(d.ctrl);
    }
    inst.dst.clamp = clmp != 0;
    inst.dst.multiplier = match omod {
        0 => 1.0,
        1 => 2.0,
        2 => 4.0,
        3 => 0.5,
        _ => unreachable!(),
    };

    inst.format = F::SVdstSVsrc0;

    match opcode {
        0x00 => return Err(ni(dst, S, "v_nop", opcode, pc, b0)),
        0x01 => inst.type_ = T::VMovB32,
        0x02 => return Err(ni(dst, S, "v_readfirstlane_b32", opcode, pc, b0)),
        0x03 => return Err(ni(dst, S, "v_cvt_i32_f64", opcode, pc, b0)),
        0x04 => return Err(ni(dst, S, "v_cvt_f64_i32", opcode, pc, b0)),
        0x05 => inst.type_ = T::VCvtF32I32,
        0x06 => inst.type_ = T::VCvtF32U32,
        0x07 => inst.type_ = T::VCvtU32F32,
        0x08 => inst.type_ = T::VCvtI32F32,
        0x09 => return Err(ni(dst, S, "v_mov_fed_b32", opcode, pc, b0)),
        0x0a => return Err(ni(dst, S, "v_cvt_f16_f32", opcode, pc, b0)),
        0x0b => inst.type_ = T::VCvtF32F16,
        0x0c => return Err(ni(dst, S, "v_cvt_rpi_i32_f32", opcode, pc, b0)),
        0x0d => inst.type_ = T::VCvtFlrI32F32,
        0x0e => return Err(ni(dst, S, "v_cvt_off_f32_i4", opcode, pc, b0)),
        0x0f => return Err(ni(dst, S, "v_cvt_f32_f64", opcode, pc, b0)),
        0x10 => return Err(ni(dst, S, "v_cvt_f64_f32", opcode, pc, b0)),
        0x11 => inst.type_ = T::VCvtF32Ubyte0,
        0x12 => inst.type_ = T::VCvtF32Ubyte1,
        0x13 => inst.type_ = T::VCvtF32Ubyte2,
        0x14 => inst.type_ = T::VCvtF32Ubyte3,
        0x15 => return Err(ni(dst, S, "v_cvt_u32_f64", opcode, pc, b0)),
        0x16 => return Err(ni(dst, S, "v_cvt_f64_u32", opcode, pc, b0)),
        0x17 => return Err(ni(dst, S, "v_trunc_f64", opcode, pc, b0)),
        0x18 => return Err(ni(dst, S, "v_ceil_f64", opcode, pc, b0)),
        0x19 => return Err(ni(dst, S, "v_rndne_f64", opcode, pc, b0)),
        0x1a => return Err(ni(dst, S, "v_floor_f64", opcode, pc, b0)),
        0x20 => inst.type_ = T::VFractF32,
        0x21 => inst.type_ = T::VTruncF32,
        0x22 => inst.type_ = T::VCeilF32,
        0x23 => inst.type_ = T::VRndneF32,
        0x24 => inst.type_ = T::VFloorF32,
        0x25 => inst.type_ = T::VExpF32,
        0x26 => return Err(ni(dst, S, "v_log_clamp_f32", opcode, pc, b0)),
        0x27 => inst.type_ = T::VLogF32,
        0x28 => return Err(ni(dst, S, "v_rcp_clamp_f32", opcode, pc, b0)),
        0x29 => return Err(ni(dst, S, "v_rcp_legacy_f32", opcode, pc, b0)),
        0x2a => inst.type_ = T::VRcpF32,
        // Beyond Kyty (KYTY_NI upstream): measured on ASTRO.BOT compute
        // (58 skips/run). Same arithmetic as v_rcp_f32; the iflag TRAP
        // status it would raise is not modelled (see `VRcpIflagF32`).
        0x2b => inst.type_ = T::VRcpIflagF32,
        0x2c => return Err(ni(dst, S, "v_rsq_clamp_f32", opcode, pc, b0)),
        0x2d => return Err(ni(dst, S, "v_rsq_legacy_f32", opcode, pc, b0)),
        0x2e => inst.type_ = T::VRsqF32,
        0x2f => return Err(ni(dst, S, "v_rcp_f64", opcode, pc, b0)),
        0x30 => return Err(ni(dst, S, "v_rcp_clamp_f64", opcode, pc, b0)),
        0x31 => return Err(ni(dst, S, "v_rsq_f64", opcode, pc, b0)),
        0x32 => return Err(ni(dst, S, "v_rsq_clamp_f64", opcode, pc, b0)),
        0x33 => inst.type_ = T::VSqrtF32,
        0x34 => return Err(ni(dst, S, "v_sqrt_f64", opcode, pc, b0)),
        0x35 => inst.type_ = T::VSinF32,
        0x36 => inst.type_ = T::VCosF32,
        0x37 => inst.type_ = T::VNotB32,
        0x38 => inst.type_ = T::VBfrevB32,
        0x39 => return Err(ni(dst, S, "v_ffbh_u32", opcode, pc, b0)),
        0x3a => return Err(ni(dst, S, "v_ffbl_b32", opcode, pc, b0)),
        0x3b => return Err(ni(dst, S, "v_ffbh_i32", opcode, pc, b0)),
        0x3c => return Err(ni(dst, S, "v_frexp_exp_i32_f64", opcode, pc, b0)),
        0x3d => return Err(ni(dst, S, "v_frexp_mant_f64", opcode, pc, b0)),
        0x3e => return Err(ni(dst, S, "v_fract_f64", opcode, pc, b0)),
        0x3f => return Err(ni(dst, S, "v_frexp_exp_i32_f32", opcode, pc, b0)),
        0x40 => return Err(ni(dst, S, "v_frexp_mant_f32", opcode, pc, b0)),
        0x41 => return Err(ni(dst, S, "v_clrexcp", opcode, pc, b0)),
        0x42 => return Err(ni(dst, S, "v_movreld_b32", opcode, pc, b0)),
        0x43 => return Err(ni(dst, S, "v_movrels_b32", opcode, pc, b0)),
        0x44 => return Err(ni(dst, S, "v_movrelsd_b32", opcode, pc, b0)),
        0x45 => return Err(ni(dst, S, "v_log_legacy_f32", opcode, pc, b0)),
        0x46 => return Err(ni(dst, S, "v_exp_legacy_f32", opcode, pc, b0)),
        0x50 => return Err(ni(dst, S, "v_cvt_f16_u16", opcode, pc, b0)),
        0x51 => return Err(ni(dst, S, "v_cvt_f16_i16", opcode, pc, b0)),
        0x52 => return Err(ni(dst, S, "v_cvt_u16_f16", opcode, pc, b0)),
        0x53 => return Err(ni(dst, S, "v_cvt_i16_f16", opcode, pc, b0)),
        0x54 => return Err(ni(dst, S, "v_rcp_f16", opcode, pc, b0)),
        0x55 => return Err(ni(dst, S, "v_sqrt_f16", opcode, pc, b0)),
        0x56 => return Err(ni(dst, S, "v_rsq_f16", opcode, pc, b0)),
        0x57 => return Err(ni(dst, S, "v_log_f16", opcode, pc, b0)),
        0x58 => return Err(ni(dst, S, "v_exp_f16", opcode, pc, b0)),
        0x59 => return Err(ni(dst, S, "v_frexp_mant_f16", opcode, pc, b0)),
        0x5a => return Err(ni(dst, S, "v_frexp_exp_i16_f16", opcode, pc, b0)),
        0x5b => return Err(ni(dst, S, "v_floor_f16", opcode, pc, b0)),
        0x5c => return Err(ni(dst, S, "v_ceil_f16", opcode, pc, b0)),
        0x5d => return Err(ni(dst, S, "v_trunc_f16", opcode, pc, b0)),
        0x5e => return Err(ni(dst, S, "v_rndne_f16", opcode, pc, b0)),
        0x5f => return Err(ni(dst, S, "v_fract_f16", opcode, pc, b0)),
        0x60 => return Err(ni(dst, S, "v_sin_f16", opcode, pc, b0)),
        0x61 => return Err(ni(dst, S, "v_cos_f16", opcode, pc, b0)),
        0x62 => return Err(ni(dst, S, "v_sat_pk_u8_i16", opcode, pc, b0)),
        0x63 => return Err(ni(dst, S, "v_cvt_norm_i16_f16", opcode, pc, b0)),
        0x64 => return Err(ni(dst, S, "v_cvt_norm_u16_f16", opcode, pc, b0)),
        0x65 => return Err(ni(dst, S, "v_swap_b32", opcode, pc, b0)),
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_vop2` (L1047). VOPC/VOP1 are nested
/// inside this decoder (opcode 0x3e/0x3f).
fn shader_parse_vop2(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "vop2";
    let b0 = buffer[0];

    let opcode = (b0 >> 25) & 0x3f;

    match opcode {
        0x3e => return shader_parse_vopc(pc, buffer, dst, next_gen),
        0x3f => return shader_parse_vop1(pc, buffer, dst, next_gen),
        _ => {}
    }

    let vdst = (b0 >> 17) & 0xff;
    let mut src0 = b0 & 0x1ff;
    let vsrc1 = (b0 >> 9) & 0xff;

    let sdwa = src0 == 249;
    // DPP (src0 == 0xfa/0xe9/0xea): two-dword cross-lane form, mutually
    // exclusive with SDWA. Real src0 (VGPR) is `b1 & 0xff`; see `decode_dpp`.
    let dpp = is_dpp_marker(src0);

    let mut size: u32 = if sdwa || dpp { 2 } else { 1 };
    let b1 = if sdwa || dpp { dw(buffer, 1, pc)? } else { 0 };
    let dpp_dec = if dpp {
        Some(decode_dpp(src0, b1))
    } else {
        None
    };

    src0 = if sdwa || dpp { b1 & 0xff } else { src0 };
    let dst_sel = if sdwa { (b1 >> 8) & 0x7 } else { 6 };
    let dst_u = if sdwa { (b1 >> 11) & 0x3 } else { 2 };
    let clmp = if sdwa { (b1 >> 13) & 0x1 } else { 0 };
    let omod = if sdwa { (b1 >> 14) & 0x3 } else { 0 };
    let src0_sel = if sdwa { (b1 >> 16) & 0x7 } else { 6 };
    let src0_sext = if sdwa { (b1 >> 19) & 0x1 } else { 0 };
    let src0_neg = if sdwa { (b1 >> 20) & 0x1 } else { 0 };
    let src0_abs = if sdwa { (b1 >> 21) & 0x1 } else { 0 };
    let s0 = if sdwa { (b1 >> 23) & 0x1 } else { 1 };
    let src1_sel = if sdwa { (b1 >> 24) & 0x7 } else { 6 };
    let src1_sext = if sdwa { (b1 >> 27) & 0x1 } else { 0 };
    let src1_neg = if sdwa { (b1 >> 28) & 0x1 } else { 0 };
    let src1_abs = if sdwa { (b1 >> 29) & 0x1 } else { 0 };
    let s1 = if sdwa { (b1 >> 31) & 0x1 } else { 0 };

    // Kyty L1088-1096: EXIT_NOT_IMPLEMENTED on unsupported SDWA modifiers
    // (abs is supported and applied below). Beyond Kyty: omod is NOT refused
    // — it is a plain output multiply (x2/x4/x0.5) carried in
    // `dst.multiplier`, exactly as the VOP1 SDWA path already does (see
    // `astro_vop1_sdwa_omod_recompiles_as_float_multiply`); the float
    // recompile bodies apply it via the MULTIPLY template. 58 measured
    // ASTRO.BOT CS failures ("vop2 feature: sdwa omod != 0").
    if dst_sel != 6 {
        return Err(feature(S, "sdwa dst_sel != 6", pc));
    }
    if sdwa && dst_sel == 6 && dst_u != 0 {
        return Err(feature(S, "sdwa dst_u != 0", pc));
    }
    // Beyond Kyty: sub-dword source selects ride `ShaderOperand::lane_sel`
    // into the operand loaders (same model as the vopc/vop1 SDWA paths).
    // Sign extension stays named — the loaders zero-extend.
    if src0_sel > 6 || src1_sel > 6 {
        return Err(feature(S, "sdwa src_sel == 7 (reserved)", pc));
    }
    if src0_sext != 0 {
        return Err(feature(S, "sdwa src0_sext != 0", pc));
    }
    if src1_sext != 0 {
        return Err(feature(S, "sdwa src1_sext != 0", pc));
    }

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    // DPP src0 is always a VGPR (`s0` is an SDWA-only field, 1 here).
    inst.src[0] = operand_parse(src0 + if s0 == 0 || dpp { 256 } else { 0 })?;
    inst.src[1] = operand_parse(vsrc1 + if s1 == 0 { 256 } else { 0 })?;
    inst.dst = operand_parse(vdst + 256)?;
    inst.src_num = 2;

    match omod {
        0 => inst.dst.multiplier = 1.0,
        1 => inst.dst.multiplier = 2.0,
        2 => inst.dst.multiplier = 4.0,
        3 => inst.dst.multiplier = 0.5,
        _ => {}
    }

    if inst.src[0].type_ == O::LiteralConstant {
        inst.src[0].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    inst.src[0].absolute = src0_abs != 0;
    inst.src[1].absolute = src1_abs != 0;
    inst.src[0].negate = src0_neg != 0;
    inst.src[1].negate = src1_neg != 0;
    inst.src[0].lane_sel = src0_sel as u8;
    inst.src[1].lane_sel = src1_sel as u8;

    // DPP overrides: the control dword carries its own src0/src1 abs/neg and the
    // cross-lane pattern (attached to src0). DPP8 has no modifiers.
    if let Some(d) = &dpp_dec {
        inst.src[0].absolute = d.src0_abs;
        inst.src[0].negate = d.src0_neg;
        inst.src[1].absolute = d.src1_abs;
        inst.src[1].negate = d.src1_neg;
        inst.src[0].dpp = Some(d.ctrl);
    }

    inst.dst.clamp = clmp != 0;

    inst.format = F::SVdstSVsrc0SVsrc1;

    match opcode {
        0x00 => {
            // Kyty L1130 punts next_gen 0x00, but the measured RDNA2 menu CS
            // emits v_cndmask_b32 there with the plain VOP2 layout (dst,
            // vsrc0, vsrc1, implicit VCC) — identical to the legacy form and
            // to Kyty's own next_gen 0x01 handler below.
            inst.type_ = T::VCndmaskB32;
            inst.format = F::VdstVsrc0Vsrc1Smask2;
            inst.src[2].type_ = O::VccLo;
            inst.src[2].size = 2;
            inst.src_num = 3;
        }
        0x01 => {
            if next_gen {
                inst.type_ = T::VCndmaskB32;
                inst.format = F::VdstVsrc0Vsrc1Smask2;
                inst.src[2].type_ = O::VccLo;
                inst.src[2].size = 2;
                inst.src_num = 3;
            } else {
                return Err(ni(dst, S, "v_readlane_b32", opcode, pc, b0));
            }
        }
        0x02 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_writelane_b32", opcode, pc, b0));
        }
        0x03 => inst.type_ = T::VAddF32,
        0x04 => inst.type_ = T::VSubF32,
        0x05 => inst.type_ = T::VSubrevF32,
        0x06 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_mac_legacy_f32", opcode, pc, b0));
        }
        0x07 => return Err(ni(dst, S, "v_mul_legacy_f32", opcode, pc, b0)),
        0x08 => inst.type_ = T::VMulF32,
        0x09 => return Err(ni(dst, S, "v_mul_i32_i24", opcode, pc, b0)),
        0x0a => return Err(ni(dst, S, "v_mul_hi_i32_i24", opcode, pc, b0)),
        0x0c => return Err(ni(dst, S, "v_mul_hi_u32_u24", opcode, pc, b0)),
        0x0d => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_min_legacy_f32", opcode, pc, b0));
        }
        0x0e => return Err(ni(dst, S, "v_max_legacy_f32", opcode, pc, b0)),
        0x0b => inst.type_ = T::VMulU32U24,
        0x0f => inst.type_ = T::VMinF32,
        0x10 => inst.type_ = T::VMaxF32,
        // Integer min/max share the VOP2 slot in both GCN and RDNA2 (like the
        // float 0x0f/0x10 above); carry-less two-source ALU ops.
        0x11 => inst.type_ = T::VMinI32,
        0x12 => inst.type_ = T::VMaxI32,
        0x13 => inst.type_ = T::VMinU32,
        0x14 => inst.type_ = T::VMaxU32,
        0x15 => inst.type_ = T::VLshrB32,
        0x16 => inst.type_ = T::VLshrrevB32,
        0x17 => inst.type_ = T::VAshrI32,
        0x18 => inst.type_ = T::VAshrrevI32,
        0x19 => inst.type_ = T::VLshlB32,
        0x1a => inst.type_ = T::VLshlrevB32,
        0x1b => inst.type_ = T::VAndB32,
        0x1c => inst.type_ = T::VOrB32,
        0x1d => inst.type_ = T::VXorB32,
        0x1e => {
            if next_gen {
                // RDNA2 reuses this VOP2 slot for v_xnor_b32 = ~(src0 ^ src1).
                inst.type_ = T::VXnorB32;
            } else {
                inst.type_ = T::VBfmB32;
            }
        }
        0x1f => inst.type_ = T::VMacF32,
        0x20 => {
            inst.type_ = T::VMadmkF32;
            inst.format = F::VdstVsrc0Vsrc1Vsrc2;
            inst.src_num = 3;
            inst.src[2] = inst.src[1];
            inst.src[1].type_ = O::LiteralConstant;
            inst.src[1].constant.u = dw(buffer, size, pc)?;
            inst.src[1].size = 0;
            size += 1;
        }
        0x21 => {
            inst.type_ = T::VMadakF32;
            inst.format = F::VdstVsrc0Vsrc1Vsrc2;
            inst.src_num = 3;
            inst.src[2].type_ = O::LiteralConstant;
            inst.src[2].constant.u = dw(buffer, size, pc)?;
            inst.src[2].size = 0;
            size += 1;
        }
        0x22 => inst.type_ = T::VBcntU32B32,
        0x23 => inst.type_ = T::VMbcntLoU32B32,
        0x24 => inst.type_ = T::VMbcntHiU32B32,
        0x25 => {
            if next_gen {
                // RDNA2 reuses the slot for the carry-less v_add_nc_u32.
                inst.type_ = T::VAddNcU32;
            } else {
                inst.type_ = T::VAddI32;
                inst.format = F::VdstSdst2Vsrc0Vsrc1;
                inst.dst2.type_ = O::VccLo;
                inst.dst2.size = 2;
            }
        }
        0x26 => {
            if next_gen {
                inst.type_ = T::VSubNcU32;
            } else {
                inst.type_ = T::VSubI32;
                inst.format = F::VdstSdst2Vsrc0Vsrc1;
                inst.dst2.type_ = O::VccLo;
                inst.dst2.size = 2;
            }
        }
        0x27 => {
            if next_gen {
                inst.type_ = T::VSubrevNcU32;
            } else {
                inst.type_ = T::VSubrevI32;
                inst.format = F::VdstSdst2Vsrc0Vsrc1;
                inst.dst2.type_ = O::VccLo;
                inst.dst2.size = 2;
            }
        }
        0x28 => {
            if next_gen {
                // RDNA2 v_add_co_ci_u32 (VOP2): vdst = src0 + vsrc1 + vcc_in;
                // carry in and out both flow through VCC.
                inst.type_ = T::VAddCoCiU32;
                inst.format = F::VdstSdst2Vsrc0Vsrc1Smask2;
                inst.src_num = 3;
                inst.dst2.type_ = O::VccLo;
                inst.dst2.size = 2;
                inst.src[2].type_ = O::VccLo;
                inst.src[2].size = 2;
            } else {
                return Err(ni(dst, S, "v_addc_u32", opcode, pc, b0));
            }
        }
        0x29 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_subb_u32", opcode, pc, b0));
        }
        0x2a => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_subbrev_u32", opcode, pc, b0));
        }
        0x2b => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_ldexp_f32", opcode, pc, b0));
        }
        0x2c => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_cvt_pkaccum_u8_f32", opcode, pc, b0));
        }
        0x2d => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_cvt_pknorm_i16_f32", opcode, pc, b0));
        }
        0x2e => return Err(ni(dst, S, "v_cvt_pknorm_u16_f32", opcode, pc, b0)),
        0x2f => inst.type_ = T::VCvtPkrtzF16F32,
        0x30 => return Err(ni(dst, S, "v_cvt_pk_u16_u32", opcode, pc, b0)),
        0x31 => return Err(ni(dst, S, "v_cvt_pk_i16_i32", opcode, pc, b0)),
        0x32 => return Err(ni(dst, S, "v_add_f16", opcode, pc, b0)),
        0x33 => return Err(ni(dst, S, "v_sub_f16", opcode, pc, b0)),
        0x34 => return Err(ni(dst, S, "v_subrev_f16", opcode, pc, b0)),
        0x35 => return Err(ni(dst, S, "v_mul_f16", opcode, pc, b0)),
        0x36 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_mac_f16", opcode, pc, b0));
        }
        0x37 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_madmk_f16", opcode, pc, b0));
        }
        0x38 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_madak_f16", opcode, pc, b0));
        }
        0x39 => return Err(ni(dst, S, "v_max_f16", opcode, pc, b0)),
        0x3a => return Err(ni(dst, S, "v_min_f16", opcode, pc, b0)),
        0x3b => return Err(ni(dst, S, "v_ldexp_f16", opcode, pc, b0)),
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// RDNA2 VOP3B opcodes: VOP3 ops that carry a scalar carry/borrow/mask
/// destination (`sdst`, bits [14:8]) instead of the VOP3A `op_sel`/`abs` fields.
/// SharpEmu Gen5 keeps the identical set (`IsVop3BOpcode`,
/// Gen5ShaderTranslator.cs L1163-1164): 0x128 `v_add_co_ci_u32`, 0x16d/0x16e
/// `v_div_scale_f32/f64`, 0x176/0x177 `v_mad_u64_u32`/`v_mad_i64_i32`, 0x30f
/// `v_add_co_u32`, 0x310 `v_sub_co_u32`, 0x319 `v_subrev_co_u32`. Only 0x128 is
/// implemented so far; the rest are named refusals but must still be recognised
/// here so their sdst is not misread as op_sel.
const fn is_vop3b_opcode(opcode: u32) -> bool {
    matches!(
        opcode,
        0x128 | 0x16d | 0x16e | 0x176 | 0x177 | 0x30f | 0x310 | 0x319
    )
}

/// Kyty: ShaderParse.cpp `shader_parse_vop3` (L1372). Handles the VOP3
/// encoding, which also carries VOPC (opcode 0x00-0xff), VOP2 (0x100-0x13d)
/// and VOP1 (0x180-0x1e8) operations. Legacy vs next-gen differ in the
/// opcode/clamp/op_sel bit layout (L1380-1382) and in which opcodes exist.
#[allow(clippy::too_many_lines)]
fn shader_parse_vop3(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "vop3";
    let b0 = buffer[0];
    let b1 = dw(buffer, 1, pc)?;

    let opcode = if next_gen {
        (b0 >> 16) & 0x3ff
    } else {
        (b0 >> 17) & 0x1ff
    };
    let clamp = if next_gen {
        (b0 >> 15) & 0x1
    } else {
        (b0 >> 11) & 0x1
    };
    // op_sel (bits [14:11]) is a VOP3A-only field. VOP3B opcodes reuse those
    // same bits as the top of their sdst field (bits [14:8]), so reading op_sel
    // there is a misdecode — a carry-out SGPR reads back as a non-zero op_sel
    // and the shader is wrongly refused. Gate it out for the VOP3B opcodes.
    // SharpEmu Gen5 does the same (Gen5ShaderTranslator.cs L1916-1923: op_sel is
    // forced to 0 and sdst is read instead when `IsVop3BOpcode`).
    let op_sel = if next_gen && !is_vop3b_opcode(opcode) {
        (b0 >> 11) & 0xf
    } else {
        0
    };
    let abs = (b0 >> 8) & 0x7;
    let vdst = b0 & 0xff;
    let sdst = (b0 >> 8) & 0x7f;
    let neg = (b1 >> 29) & 0x7;
    let omod = (b1 >> 27) & 0x3;
    let src0 = b1 & 0x1ff;
    let src1 = (b1 >> 9) & 0x1ff;
    let src2 = (b1 >> 18) & 0x1ff;

    // Kyty L1392: EXIT_NOT_IMPLEMENTED(op_sel != 0). Genuine VOP3A op_sel
    // (packed-16-bit half select on true VOP3A ops) stays a named refusal — no
    // shader measured to date needs it. The former false positives were all the
    // VOP3B carry op above, now gated out.
    if op_sel != 0 {
        return Err(feature(S, "op_sel != 0", pc));
    }

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.src[0] = operand_parse(src0)?;
    inst.src[1] = operand_parse(src1)?;
    inst.src[2] = operand_parse(src2)?;
    inst.src_num = 3;
    inst.dst = operand_parse(vdst + 256)?;

    match omod {
        0 => inst.dst.multiplier = 1.0,
        1 => inst.dst.multiplier = 2.0,
        2 => inst.dst.multiplier = 4.0,
        3 => inst.dst.multiplier = 0.5,
        _ => {}
    }

    if (neg & 0x1) != 0 {
        inst.src[0].negate = true;
    }
    if (neg & 0x2) != 0 {
        inst.src[1].negate = true;
    }
    if (neg & 0x4) != 0 {
        inst.src[2].negate = true;
    }

    let mut size: u32 = 2;

    if inst.src[0].type_ == O::LiteralConstant {
        inst.src[0].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    if inst.src[1].type_ == O::LiteralConstant {
        inst.src[1].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    if inst.src[2].type_ == O::LiteralConstant {
        inst.src[2].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    inst.format = F::VdstVsrc0Vsrc1Vsrc2;

    if opcode <= 0xff {
        // VOPC using VOP3 encoding (Kyty L1446).
        inst.format = F::SmaskVsrc0Vsrc1;
        inst.src_num = 2;
        inst.dst = operand_parse(vdst)?;
        inst.dst.size = 2;
    }

    if (0x100..=0x13d).contains(&opcode) {
        // VOP2 using VOP3 encoding (Kyty L1455).
        inst.format = F::SVdstSVsrc0SVsrc1;
        inst.src_num = 2;
    }

    if (0x180..=0x1e8).contains(&opcode) {
        // VOP1 using VOP3 encoding (Kyty L1462).
        inst.format = F::SVdstSVsrc0;
        inst.src_num = 1;
    }

    match opcode {
        // VOPC using VOP3 encoding (Kyty L1471). Note: unlike plain VOPC,
        // 0x11/0x14 (v_cmpx_lt/gt_f32) are NI here (L1489/L1492).
        0x00 => inst.type_ = T::VCmpFF32,
        0x01 => inst.type_ = T::VCmpLtF32,
        0x02 => inst.type_ = T::VCmpEqF32,
        0x03 => inst.type_ = T::VCmpLeF32,
        0x04 => inst.type_ = T::VCmpGtF32,
        0x05 => inst.type_ = T::VCmpLgF32,
        0x06 => inst.type_ = T::VCmpGeF32,
        0x07 => inst.type_ = T::VCmpOF32,
        0x08 => inst.type_ = T::VCmpUF32,
        0x09 => inst.type_ = T::VCmpNgeF32,
        0x0a => inst.type_ = T::VCmpNlgF32,
        0x0b => inst.type_ = T::VCmpNgtF32,
        0x0c => inst.type_ = T::VCmpNleF32,
        0x0d => inst.type_ = T::VCmpNeqF32,
        0x0e => inst.type_ = T::VCmpNltF32,
        0x0f => inst.type_ = T::VCmpTruF32,
        0x12 => inst.type_ = T::VCmpxEqF32,
        0x13 => inst.type_ = T::VCmpxLeF32,
        0x16 => inst.type_ = T::VCmpxGeF32,
        0x19 => inst.type_ = T::VCmpxNgeF32,
        0x1c => inst.type_ = T::VCmpxNleF32,
        0x1d => inst.type_ = T::VCmpxNeqF32,
        0x1e => inst.type_ = T::VCmpxNltF32,
        0x80 => inst.type_ = T::VCmpFI32,
        0x81 => inst.type_ = T::VCmpLtI32,
        0x82 => inst.type_ = T::VCmpEqI32,
        0x83 => inst.type_ = T::VCmpLeI32,
        0x84 => inst.type_ = T::VCmpGtI32,
        0x85 => inst.type_ = T::VCmpNeI32,
        0x86 => inst.type_ = T::VCmpGeI32,
        0x87 => inst.type_ = T::VCmpTI32,
        // 0x9x is the `v_cmpx_*_i32` block — the signed twin of the 0xdx
        // (`v_cmpx_*_u32`) block below. The whole block was missing, so each
        // instruction decoded as unknown and every draw binding that shader was
        // skipped. Measured in Minecraft, which reaches shaders using
        // `v_cmpx_lt_i32` (0x91) and `v_cmpx_ge_i32` (0x96) once boot gets far
        // enough; the rest of the block is wired at the same time because it is
        // the same lowering and the title decides one opcode at a time.
        0x91 => inst.type_ = T::VCmpxLtI32,
        0x92 => inst.type_ = T::VCmpxEqI32,
        // 0x93 has no unsigned twin (there is no 0xd3), so it is easy to miss
        // when mirroring that block — Minecraft emits it.
        0x93 => inst.type_ = T::VCmpxLeI32,
        0x94 => inst.type_ = T::VCmpxGtI32,
        0x95 => inst.type_ = T::VCmpxNeI32,
        0x96 => inst.type_ = T::VCmpxGeI32,
        0xc0 => inst.type_ = T::VCmpFU32,
        0xc1 => inst.type_ = T::VCmpLtU32,
        0xc2 => inst.type_ = T::VCmpEqU32,
        0xc3 => inst.type_ = T::VCmpLeU32,
        0xc4 => inst.type_ = T::VCmpGtU32,
        0xc5 => inst.type_ = T::VCmpNeU32,
        0xc6 => inst.type_ = T::VCmpGeU32,
        0xc7 => inst.type_ = T::VCmpTU32,
        0xd1 => inst.type_ = T::VCmpxLtU32,
        0xd2 => inst.type_ = T::VCmpxEqU32,
        0xd4 => inst.type_ = T::VCmpxGtU32,
        0xd5 => inst.type_ = T::VCmpxNeU32,
        0xd6 => inst.type_ = T::VCmpxGeU32,

        // VOP2 using VOP3 encoding (Kyty L1727).
        0x100 => {
            // Kyty L1729: EXIT_NOT_IMPLEMENTED(next_gen).
            if next_gen {
                return Err(feature(S, "v_cndmask_b32 (op 0x100) on next_gen", pc));
            }
            inst.type_ = T::VCndmaskB32;
            inst.format = F::VdstVsrc0Vsrc1Smask2;
            inst.src_num = 3;
            inst.src[2].size = 2;
        }
        0x101 => {
            if next_gen {
                inst.type_ = T::VCndmaskB32;
                inst.format = F::VdstVsrc0Vsrc1Smask2;
                inst.src_num = 3;
                inst.src[2].size = 2;
            } else {
                return Err(ni(dst, S, "v_readlane_b32", opcode, pc, b0));
            }
        }
        0x102 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_writelane_b32", opcode, pc, b0));
        }
        0x103 => inst.type_ = T::VAddF32,
        0x104 => inst.type_ = T::VSubF32,
        0x105 => inst.type_ = T::VSubrevF32,
        0x106 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_mac_legacy_f32", opcode, pc, b0));
        }
        0x107 => return Err(ni(dst, S, "v_mul_legacy_f32", opcode, pc, b0)),
        0x108 => inst.type_ = T::VMulF32,
        0x109 => return Err(ni(dst, S, "v_mul_i32_i24", opcode, pc, b0)),
        0x10a => return Err(ni(dst, S, "v_mul_hi_i32_i24", opcode, pc, b0)),
        0x10b => inst.type_ = T::VMulU32U24,
        0x10c => return Err(ni(dst, S, "v_mul_hi_u32_u24", opcode, pc, b0)),
        0x10d => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_min_legacy_f32", opcode, pc, b0));
        }
        0x10e => return Err(ni(dst, S, "v_max_legacy_f32", opcode, pc, b0)),
        0x10f => inst.type_ = T::VMinF32,
        0x110 => inst.type_ = T::VMaxF32,
        0x111 => return Err(ni(dst, S, "v_min_i32", opcode, pc, b0)),
        0x112 => return Err(ni(dst, S, "v_max_i32", opcode, pc, b0)),
        0x113 => return Err(ni(dst, S, "v_min_u32", opcode, pc, b0)),
        0x114 => return Err(ni(dst, S, "v_max_u32", opcode, pc, b0)),
        0x115 => inst.type_ = T::VLshrB32,
        0x116 => inst.type_ = T::VLshrrevB32,
        0x117 => inst.type_ = T::VAshrI32,
        0x118 => inst.type_ = T::VAshrrevI32,
        0x119 => inst.type_ = T::VLshlB32,
        0x11a => inst.type_ = T::VLshlrevB32,
        0x11b => inst.type_ = T::VAndB32,
        0x11c => inst.type_ = T::VOrB32,
        0x11d => inst.type_ = T::VXorB32,
        0x11e => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            inst.type_ = T::VBfmB32;
        }
        0x11f => inst.type_ = T::VMacF32,
        0x120 => return Err(ni(dst, S, "v_madmk_f32", opcode, pc, b0)),
        0x121 => return Err(ni(dst, S, "v_madak_f32", opcode, pc, b0)),
        0x122 => inst.type_ = T::VBcntU32B32,
        0x123 => inst.type_ = T::VMbcntLoU32B32,
        0x124 => inst.type_ = T::VMbcntHiU32B32,
        0x125 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            inst.type_ = T::VAddI32;
            inst.format = F::VdstSdst2Vsrc0Vsrc1;
            inst.dst2 = operand_parse(sdst)?;
            inst.dst2.size = 2;
        }
        0x126 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            inst.type_ = T::VSubI32;
            inst.format = F::VdstSdst2Vsrc0Vsrc1;
            inst.dst2 = operand_parse(sdst)?;
            inst.dst2.size = 2;
        }
        0x127 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            inst.type_ = T::VSubrevI32;
            inst.format = F::VdstSdst2Vsrc0Vsrc1;
            inst.dst2 = operand_parse(sdst)?;
            inst.dst2.size = 2;
        }
        0x128 => {
            if next_gen {
                // RDNA2 v_add_co_ci_u32 (VOP3B): vdst = src0 + src1 + carry_in;
                // carry_out -> sdst. src2 (already parsed) is the carry-in mask;
                // sdst (bits [14:8]) is the carry-out mask. See is_vop3b_opcode.
                inst.type_ = T::VAddCoCiU32;
                inst.format = F::VdstSdst2Vsrc0Vsrc1Smask2;
                inst.src_num = 3;
                inst.src[2].size = 2;
                inst.dst2 = operand_parse(sdst)?;
                inst.dst2.size = 2;
            } else {
                return Err(ni(dst, S, "v_addc_u32", opcode, pc, b0));
            }
        }
        0x129 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_subb_u32", opcode, pc, b0));
        }
        0x12a => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_subbrev_u32", opcode, pc, b0));
        }
        0x12b => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_ldexp_f32", opcode, pc, b0));
        }
        0x12c => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_cvt_pkaccum_u8_f32", opcode, pc, b0));
        }
        0x12d => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_cvt_pknorm_i16_f32", opcode, pc, b0));
        }
        0x12e => return Err(ni(dst, S, "v_cvt_pknorm_u16_f32", opcode, pc, b0)),
        0x12f => inst.type_ = T::VCvtPkrtzF16F32,
        0x130 => return Err(ni(dst, S, "v_cvt_pk_u16_u32", opcode, pc, b0)),
        0x131 => return Err(ni(dst, S, "v_cvt_pk_i16_i32", opcode, pc, b0)),
        0x132 => return Err(ni(dst, S, "v_add_f16", opcode, pc, b0)),
        0x133 => return Err(ni(dst, S, "v_sub_f16", opcode, pc, b0)),
        0x134 => return Err(ni(dst, S, "v_subrev_f16", opcode, pc, b0)),
        0x135 => return Err(ni(dst, S, "v_mul_f16", opcode, pc, b0)),
        0x136 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_mac_f16", opcode, pc, b0));
        }
        0x137 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_madmk_f16", opcode, pc, b0));
        }
        0x138 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_madak_f16", opcode, pc, b0));
        }
        0x139 => return Err(ni(dst, S, "v_max_f16", opcode, pc, b0)),
        0x13a => return Err(ni(dst, S, "v_min_f16", opcode, pc, b0)),
        0x13b => return Err(ni(dst, S, "v_ldexp_f16", opcode, pc, b0)),

        // VOP3 instructions (Kyty L1943).
        0x140 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_mad_legacy_f32", opcode, pc, b0));
        }
        0x141 => inst.type_ = T::VMadF32,
        0x142 => return Err(ni(dst, S, "v_mad_i32_i24", opcode, pc, b0)),
        0x143 => inst.type_ = T::VMadU32U24,
        0x144 => inst.type_ = T::VCubeIdF32,
        0x145 => inst.type_ = T::VCubeScF32,
        0x146 => inst.type_ = T::VCubeTcF32,
        0x147 => inst.type_ = T::VCubeMaF32,
        0x148 => inst.type_ = T::VBfeU32,
        0x149 => inst.type_ = T::VBfeI32,
        0x14a => inst.type_ = T::VBfiB32,
        0x14b => inst.type_ = T::VFmaF32,
        0x14c => return Err(ni(dst, S, "v_fma_f64", opcode, pc, b0)),
        0x14d => return Err(ni(dst, S, "v_lerp_u8", opcode, pc, b0)),
        0x14e => return Err(ni(dst, S, "v_alignbit_b32", opcode, pc, b0)),
        0x14f => return Err(ni(dst, S, "v_alignbyte_b32", opcode, pc, b0)),
        0x150 => return Err(ni(dst, S, "v_mullit_f32", opcode, pc, b0)),
        0x151 => inst.type_ = T::VMin3F32,
        0x152 => return Err(ni(dst, S, "v_min3_i32", opcode, pc, b0)),
        0x153 => return Err(ni(dst, S, "v_min3_u32", opcode, pc, b0)),
        0x154 => inst.type_ = T::VMax3F32,
        0x155 => return Err(ni(dst, S, "v_max3_i32", opcode, pc, b0)),
        0x156 => return Err(ni(dst, S, "v_max3_u32", opcode, pc, b0)),
        0x157 => inst.type_ = T::VMed3F32,
        0x158 => return Err(ni(dst, S, "v_med3_i32", opcode, pc, b0)),
        0x159 => return Err(ni(dst, S, "v_med3_u32", opcode, pc, b0)),
        0x15a => return Err(ni(dst, S, "v_sad_u8", opcode, pc, b0)),
        0x15b => return Err(ni(dst, S, "v_sad_hi_u8", opcode, pc, b0)),
        0x15c => return Err(ni(dst, S, "v_sad_u16", opcode, pc, b0)),
        0x15d => inst.type_ = T::VSadU32,
        0x15e => return Err(ni(dst, S, "v_cvt_pk_u8_f32", opcode, pc, b0)),
        0x15f => return Err(ni(dst, S, "v_div_fixup_f32", opcode, pc, b0)),
        0x160 => return Err(ni(dst, S, "v_div_fixup_f64", opcode, pc, b0)),
        0x161 => return Err(ni(dst, S, "v_lshl_b64", opcode, pc, b0)),
        0x162 => return Err(ni(dst, S, "v_lshr_b64", opcode, pc, b0)),
        0x163 => return Err(ni(dst, S, "v_ashr_i64", opcode, pc, b0)),
        0x164 => return Err(ni(dst, S, "v_add_f64", opcode, pc, b0)),
        0x165 => return Err(ni(dst, S, "v_mul_f64", opcode, pc, b0)),
        0x166 => return Err(ni(dst, S, "v_min_f64", opcode, pc, b0)),
        0x167 => return Err(ni(dst, S, "v_max_f64", opcode, pc, b0)),
        0x168 => return Err(ni(dst, S, "v_ldexp_f64", opcode, pc, b0)),
        0x169 => {
            inst.type_ = T::VMulLoU32;
            inst.format = F::SVdstSVsrc0SVsrc1;
            inst.src_num = 2;
        }
        0x16a => {
            inst.type_ = T::VMulHiU32;
            inst.format = F::SVdstSVsrc0SVsrc1;
            inst.src_num = 2;
        }
        0x16b => {
            inst.type_ = T::VMulLoI32;
            inst.format = F::SVdstSVsrc0SVsrc1;
            inst.src_num = 2;
        }
        0x16c => return Err(ni(dst, S, "v_mul_hi_i32", opcode, pc, b0)),
        0x16d => return Err(ni(dst, S, "v_div_scale_f32", opcode, pc, b0)),
        0x16e => return Err(ni(dst, S, "v_div_scale_f64", opcode, pc, b0)),
        0x16f => return Err(ni(dst, S, "v_div_fmas_f32", opcode, pc, b0)),
        0x170 => return Err(ni(dst, S, "v_div_fmas_f64", opcode, pc, b0)),
        0x171 => return Err(ni(dst, S, "v_msad_u8", opcode, pc, b0)),
        0x174 => return Err(ni(dst, S, "v_trig_preop_f64", opcode, pc, b0)),
        0x175 => return Err(ni(dst, S, "v_mqsad_u32_u8", opcode, pc, b0)),
        0x176 => {
            if !next_gen {
                return Err(ni(dst, S, "v_mad_u64_u32", opcode, pc, b0));
            }
            // RDNA2 v_mad_u64_u32 (VOP3B): vdst.u64 = src0.u32 * src1.u32 +
            // src2.u64; carry-out of the 64-bit add -> sdst. vdst and src2 are
            // 64-bit register pairs; src0/src1 stay 32-bit. sdst (bits [14:8])
            // is the carry-out mask — recognised as VOP3B above so it is not
            // misread as op_sel. Shares the add-with-carry format.
            inst.type_ = T::VMadU64U32;
            inst.format = F::VdstSdst2Vsrc0Vsrc1Smask2;
            inst.src_num = 3;
            inst.dst.size = 2;
            inst.src[2].size = 2;
            inst.dst2 = operand_parse(sdst)?;
            inst.dst2.size = 2;
        }
        0x177 => return Err(ni(dst, S, "v_mad_i64_i32", opcode, pc, b0)),
        0x303 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_add_u16", opcode, pc, b0));
        }
        0x304 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_sub_u16", opcode, pc, b0));
        }
        0x305 => return Err(ni(dst, S, "v_mul_lo_u16", opcode, pc, b0)),
        0x307 => return Err(ni(dst, S, "v_lshrrev_b16", opcode, pc, b0)),
        0x308 => return Err(ni(dst, S, "v_ashrrev_i16", opcode, pc, b0)),
        0x309 => return Err(ni(dst, S, "v_max_u16", opcode, pc, b0)),
        0x30a => return Err(ni(dst, S, "v_max_i16", opcode, pc, b0)),
        0x30b => return Err(ni(dst, S, "v_min_u16", opcode, pc, b0)),
        0x30c => return Err(ni(dst, S, "v_min_i16", opcode, pc, b0)),
        0x30d => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_add_i16", opcode, pc, b0));
        }
        0x30e => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_sub_i16", opcode, pc, b0));
        }
        0x30f => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_add_u32", opcode, pc, b0));
        }
        0x310 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_sub_u32", opcode, pc, b0));
        }
        0x311 => return Err(ni(dst, S, "v_pack_b32_f16", opcode, pc, b0)),
        0x312 => return Err(ni(dst, S, "v_cvt_pknorm_i16_f16", opcode, pc, b0)),
        0x313 => return Err(ni(dst, S, "v_cvt_pknorm_u16_f16", opcode, pc, b0)),
        0x314 => return Err(ni(dst, S, "v_lshlrev_b16", opcode, pc, b0)),
        0x340 => return Err(ni(dst, S, "v_mad_u16", opcode, pc, b0)),
        0x341 => return Err(ni(dst, S, "v_mad_f16", opcode, pc, b0)),
        0x342 => return Err(ni(dst, S, "v_interp_p1ll_f16", opcode, pc, b0)),
        0x343 => return Err(ni(dst, S, "v_interp_p1lv_f16", opcode, pc, b0)),
        0x344 => return Err(ni(dst, S, "v_perm_b32", opcode, pc, b0)),
        0x345 => {
            if next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            return Err(ni(dst, S, "v_xad_b32", opcode, pc, b0));
        }
        // Legacy's 9-bit VOP3 opcode space ends at 0x1ff, so 0x346 is
        // unambiguously the RDNA2 (`next_gen`) v_lshl_add_u32.
        0x346 => inst.type_ = T::VLshlAddU32,
        0x347 => return Err(ni(dst, S, "v_add_lshl_u32", opcode, pc, b0)),
        0x34b => return Err(ni(dst, S, "v_fma_f16", opcode, pc, b0)),
        0x351 => return Err(ni(dst, S, "v_min3_f16", opcode, pc, b0)),
        0x352 => return Err(ni(dst, S, "v_min3_i16", opcode, pc, b0)),
        0x353 => return Err(ni(dst, S, "v_min3_u16", opcode, pc, b0)),
        0x354 => return Err(ni(dst, S, "v_max3_f16", opcode, pc, b0)),
        0x355 => return Err(ni(dst, S, "v_max3_i16", opcode, pc, b0)),
        0x356 => return Err(ni(dst, S, "v_max3_u16", opcode, pc, b0)),
        0x357 => return Err(ni(dst, S, "v_med3_f16", opcode, pc, b0)),
        0x358 => return Err(ni(dst, S, "v_med3_i16", opcode, pc, b0)),
        0x359 => return Err(ni(dst, S, "v_med3_u16", opcode, pc, b0)),
        0x35a => return Err(ni(dst, S, "v_interp_p2_f16", opcode, pc, b0)),
        0x35e => return Err(ni(dst, S, "v_mad_i16", opcode, pc, b0)),
        0x35f => return Err(ni(dst, S, "v_div_fixup_f16", opcode, pc, b0)),
        // RDNA2 (`next_gen`) VOP3 encodings of the mbcnt pair (VOP2-native
        // 0x23/0x24 on GCN, already wired above). Verified against SharpEmu's
        // Gen5 decoder: `0x365 => VMbcntLoU32B32`, `0x366 => VMbcntHiU32B32`
        // (Gen5ShaderTranslator.cs L1144-1145) — NOT v_lshl_add_u32, which is
        // 0x346. Measured: ASTRO.BOT scene CS emits 0x366 (raw 0xd7660003) in
        // the canonical lane-index idiom. The legacy 9-bit VOP3 space ends at
        // 0x1ff, so the numbers are unambiguous.
        0x365 => {
            inst.type_ = T::VMbcntLoU32B32;
            inst.format = F::SVdstSVsrc0SVsrc1;
            inst.src_num = 2;
        }
        0x366 => {
            inst.type_ = T::VMbcntHiU32B32;
            inst.format = F::SVdstSVsrc0SVsrc1;
            inst.src_num = 2;
        }
        // `v_add3_u32`: dst = src0 + src1 + src2 (carry-less). Kyty leaves it
        // NI (ShaderParse.cpp L2112); shadPS4 `V_ADD3_U32 = 877` (== 0x36d)
        // confirms opcode + semantics. Measured: 175 ASTRO.BOT CS failures.
        0x36d => inst.type_ = T::VAdd3U32,
        // `v_lshl_or_u32`: dst = (src0 << (src1 & 31)) | src2. Same deliberate
        // deviation as 0x371 below — Kyty gates this off on next_gen, but
        // SharpEmu's Gen5 (PS5/RDNA2) decoder maps `0x36F => "VLshlOrU32"` and
        // lowers it exactly this way, and Minecraft emits it.
        0x36f => inst.type_ = T::VLshlOrU32,
        // `v_and_or_b32`: dst = (src0 & src1) | src2.
        //
        // DELIBERATE DEVIATION from Kyty, which rejects this as UNKNOWN_OP on
        // next_gen (ShaderParse.cpp L2122) and leaves it NI on legacy. That gate
        // is untested conservatism, not an RDNA2 difference — two independent
        // references agree the opcode is the same on both generations:
        //   * SharpEmu's Gen5 (PS5/RDNA2) decoder: `0x371 => "VAndOrB32"`
        //     (Gen5ShaderTranslator.cs L1137), lowered as
        //     `BitwiseOr(BitwiseAnd(s0, s1), s2)` (Gen5SpirvTranslator.Alu.cs).
        //   * shadPS4: `V_AND_OR_B32 = 881` (opcodes.h L716) — 881 == 0x371.
        // Measured: Minecraft emits it (raw 0xd7710001) in a compute shader once
        // boot reaches the menu stage; rejecting it failed the whole shader and
        // skipped every draw bound to it.
        0x371 => inst.type_ = T::VAndOrB32,
        // `v_or3_u32`: dst = (src0 | src1) | src2. Completes the gated trio
        // (0x36f/0x371/0x372); SharpEmu Gen5: `0x372 => "VOr3U32"`.
        0x372 => inst.type_ = T::VOr3U32,
        0x373 => return Err(ni(dst, S, "v_mad_u32_u16", opcode, pc, b0)),
        0x375 => return Err(ni(dst, S, "v_mad_i32_i16", opcode, pc, b0)),

        // VOP1 using VOP3 encoding (Kyty L2143).
        0x180 => return Err(ni(dst, S, "v_nop", opcode, pc, b0)),
        0x181 => return Err(ni(dst, S, "v_mov_b32", opcode, pc, b0)),
        0x182 => return Err(ni(dst, S, "v_readfirstlane_b32", opcode, pc, b0)),
        0x183 => return Err(ni(dst, S, "v_cvt_i32_f64", opcode, pc, b0)),
        0x184 => return Err(ni(dst, S, "v_cvt_f64_i32", opcode, pc, b0)),
        0x185 => return Err(ni(dst, S, "v_cvt_f32_i32", opcode, pc, b0)),
        0x186 => return Err(ni(dst, S, "v_cvt_f32_u32", opcode, pc, b0)),
        0x187 => return Err(ni(dst, S, "v_cvt_u32_f32", opcode, pc, b0)),
        0x188 => return Err(ni(dst, S, "v_cvt_i32_f32", opcode, pc, b0)),
        0x189 => return Err(ni(dst, S, "v_mov_fed_b32", opcode, pc, b0)),
        0x18a => return Err(ni(dst, S, "v_cvt_f16_f32", opcode, pc, b0)),
        0x18b => return Err(ni(dst, S, "v_cvt_f32_f16", opcode, pc, b0)),
        0x18c => return Err(ni(dst, S, "v_cvt_rpi_i32_f32", opcode, pc, b0)),
        0x18d => return Err(ni(dst, S, "v_cvt_flr_i32_f32", opcode, pc, b0)),
        0x18e => return Err(ni(dst, S, "v_cvt_off_f32_i4", opcode, pc, b0)),
        0x18f => return Err(ni(dst, S, "v_cvt_f32_f64", opcode, pc, b0)),
        0x190 => return Err(ni(dst, S, "v_cvt_f64_f32", opcode, pc, b0)),
        0x191 => return Err(ni(dst, S, "v_cvt_f32_ubyte0", opcode, pc, b0)),
        0x192 => return Err(ni(dst, S, "v_cvt_f32_ubyte1", opcode, pc, b0)),
        0x193 => return Err(ni(dst, S, "v_cvt_f32_ubyte2", opcode, pc, b0)),
        0x194 => return Err(ni(dst, S, "v_cvt_f32_ubyte3", opcode, pc, b0)),
        0x195 => return Err(ni(dst, S, "v_cvt_u32_f64", opcode, pc, b0)),
        0x196 => return Err(ni(dst, S, "v_cvt_f64_u32", opcode, pc, b0)),
        0x197 => return Err(ni(dst, S, "v_trunc_f64", opcode, pc, b0)),
        0x198 => return Err(ni(dst, S, "v_ceil_f64", opcode, pc, b0)),
        0x199 => return Err(ni(dst, S, "v_rndne_f64", opcode, pc, b0)),
        0x19a => return Err(ni(dst, S, "v_floor_f64", opcode, pc, b0)),
        0x1a0 => inst.type_ = T::VFractF32,
        0x1a1 => inst.type_ = T::VTruncF32,
        0x1a2 => inst.type_ = T::VCeilF32,
        0x1a3 => inst.type_ = T::VRndneF32,
        0x1a4 => inst.type_ = T::VFloorF32,
        0x1a5 => inst.type_ = T::VExpF32,
        0x1a6 => return Err(ni(dst, S, "v_log_clamp_f32", opcode, pc, b0)),
        0x1a7 => inst.type_ = T::VLogF32,
        0x1a8 => return Err(ni(dst, S, "v_rcp_clamp_f32", opcode, pc, b0)),
        0x1a9 => return Err(ni(dst, S, "v_rcp_legacy_f32", opcode, pc, b0)),
        0x1aa => inst.type_ = T::VRcpF32,
        // VOP3 encoding of VOP1 0x2b (see the VOP1 arm / `VRcpIflagF32`).
        0x1ab => inst.type_ = T::VRcpIflagF32,
        0x1ac => return Err(ni(dst, S, "v_rsq_clamp_f32", opcode, pc, b0)),
        0x1ad => return Err(ni(dst, S, "v_rsq_legacy_f32", opcode, pc, b0)),
        0x1ae => inst.type_ = T::VRsqF32,
        0x1af => return Err(ni(dst, S, "v_rcp_f64", opcode, pc, b0)),
        0x1b0 => return Err(ni(dst, S, "v_rcp_clamp_f64", opcode, pc, b0)),
        0x1b1 => return Err(ni(dst, S, "v_rsq_f64", opcode, pc, b0)),
        0x1b2 => return Err(ni(dst, S, "v_rsq_clamp_f64", opcode, pc, b0)),
        0x1b3 => inst.type_ = T::VSqrtF32,
        0x1b4 => return Err(ni(dst, S, "v_sqrt_f64", opcode, pc, b0)),
        0x1b5 => inst.type_ = T::VSinF32,
        0x1b6 => inst.type_ = T::VCosF32,
        0x1b7 => return Err(ni(dst, S, "v_not_b32", opcode, pc, b0)),
        0x1b8 => return Err(ni(dst, S, "v_bfrev_b32", opcode, pc, b0)),
        0x1b9 => return Err(ni(dst, S, "v_ffbh_u32", opcode, pc, b0)),
        0x1ba => return Err(ni(dst, S, "v_ffbl_b32", opcode, pc, b0)),
        0x1bb => return Err(ni(dst, S, "v_ffbh_i32", opcode, pc, b0)),
        0x1bc => return Err(ni(dst, S, "v_frexp_exp_i32_f64", opcode, pc, b0)),
        0x1bd => return Err(ni(dst, S, "v_frexp_mant_f64", opcode, pc, b0)),
        0x1be => return Err(ni(dst, S, "v_fract_f64", opcode, pc, b0)),
        0x1bf => return Err(ni(dst, S, "v_frexp_exp_i32_f32", opcode, pc, b0)),
        0x1c0 => return Err(ni(dst, S, "v_frexp_mant_f32", opcode, pc, b0)),
        0x1c1 => return Err(ni(dst, S, "v_clrexcp", opcode, pc, b0)),
        0x1c2 => return Err(ni(dst, S, "v_movreld_b32", opcode, pc, b0)),
        0x1c3 => return Err(ni(dst, S, "v_movrels_b32", opcode, pc, b0)),
        0x1c4 => return Err(ni(dst, S, "v_movrelsd_b32", opcode, pc, b0)),
        0x1c5 => return Err(ni(dst, S, "v_log_legacy_f32", opcode, pc, b0)),
        0x1c6 => return Err(ni(dst, S, "v_exp_legacy_f32", opcode, pc, b0)),
        0x1d0 => return Err(ni(dst, S, "v_cvt_f16_u16", opcode, pc, b0)),
        0x1d1 => return Err(ni(dst, S, "v_cvt_f16_i16", opcode, pc, b0)),
        0x1d2 => return Err(ni(dst, S, "v_cvt_u16_f16", opcode, pc, b0)),
        0x1d3 => return Err(ni(dst, S, "v_cvt_i16_f16", opcode, pc, b0)),
        0x1d4 => return Err(ni(dst, S, "v_rcp_f16", opcode, pc, b0)),
        0x1d5 => return Err(ni(dst, S, "v_sqrt_f16", opcode, pc, b0)),
        0x1d6 => return Err(ni(dst, S, "v_rsq_f16", opcode, pc, b0)),
        0x1d7 => return Err(ni(dst, S, "v_log_f16", opcode, pc, b0)),
        0x1d8 => return Err(ni(dst, S, "v_exp_f16", opcode, pc, b0)),
        0x1d9 => return Err(ni(dst, S, "v_frexp_mant_f16", opcode, pc, b0)),
        0x1da => return Err(ni(dst, S, "v_frexp_exp_i16_f16", opcode, pc, b0)),
        0x1db => return Err(ni(dst, S, "v_floor_f16", opcode, pc, b0)),
        0x1dc => return Err(ni(dst, S, "v_ceil_f16", opcode, pc, b0)),
        0x1dd => return Err(ni(dst, S, "v_trunc_f16", opcode, pc, b0)),
        0x1de => return Err(ni(dst, S, "v_rndne_f16", opcode, pc, b0)),
        0x1df => return Err(ni(dst, S, "v_fract_f16", opcode, pc, b0)),
        0x1e0 => return Err(ni(dst, S, "v_sin_f16", opcode, pc, b0)),
        0x1e1 => return Err(ni(dst, S, "v_cos_f16", opcode, pc, b0)),
        0x1e2 => return Err(ni(dst, S, "v_sat_pk_u8_i16", opcode, pc, b0)),
        0x1e3 => return Err(ni(dst, S, "v_cvt_norm_i16_f16", opcode, pc, b0)),
        0x1e4 => return Err(ni(dst, S, "v_cvt_norm_u16_f16", opcode, pc, b0)),
        0x1e5 => return Err(ni(dst, S, "v_swap_b32", opcode, pc, b0)),

        // RDNA2 VOP3-only re-encodings of VOP2 ops that moved out of the VOP2
        // opcode space (SharpEmu Gen5 L1139-1152). 0x364 v_bcnt_u32_b32 =
        // `vdst = bitcount(src0) + src1`; reuse the existing VOP2 lowering by
        // pinning the two-source scalar layout it expects.
        0x364 => {
            if !next_gen {
                return Err(unknown_op(dst, S, opcode, pc, b0));
            }
            inst.type_ = T::VBcntU32B32;
            inst.format = F::SVdstSVsrc0SVsrc1;
            inst.src_num = 2;
        }

        _ => {
            // VOPC-via-VOP3 not-implemented compares share the VOPC table.
            if opcode <= 0xff {
                if let Some(name) = vopc_ni_name(opcode) {
                    return Err(ni(dst, S, name, opcode, pc, b0));
                }
            }
            return Err(unknown_op(dst, S, opcode, pc, b0));
        }
    }

    // VOP3-encoded VCMPX has the same GFX10 EXEC-only destination as the
    // compact VOPC form above.
    if next_gen && is_vcmpx_instruction(inst.type_) {
        inst.dst.type_ = O::ExecLo;
        inst.dst.size = 2;
    }

    // Kyty L2236-2260: abs/clamp application depends on whether dst2 was set.
    if inst.dst2.type_ == O::Unknown {
        if (abs & 0x1) != 0 {
            inst.src[0].absolute = true;
        }
        if (abs & 0x2) != 0 {
            inst.src[1].absolute = true;
        }
        if (abs & 0x4) != 0 {
            inst.src[2].absolute = true;
        }

        if !next_gen {
            inst.dst.clamp = clamp != 0;
        }
    }

    if next_gen {
        inst.dst.clamp = clamp != 0;
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_exp` (L2267).
fn shader_parse_exp(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    let b0 = buffer[0];
    let b1 = dw(buffer, 1, pc)?;

    let vm = (b0 >> 12) & 0x1;
    let done = (b0 >> 11) & 0x1;
    let compr = (b0 >> 10) & 0x1;
    let target = (b0 >> 4) & 0x3f;
    let en = b0 & 0xf;

    let vsrc0 = b1 & 0xff;
    let vsrc1 = (b1 >> 8) & 0xff;
    let vsrc2 = (b1 >> 16) & 0xff;
    let vsrc3 = (b1 >> 24) & 0xff;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.src[0] = operand_parse(vsrc0 + 256)?;
    inst.src[1] = operand_parse(vsrc1 + 256)?;
    inst.src[2] = operand_parse(vsrc2 + 256)?;
    inst.src[3] = operand_parse(vsrc3 + 256)?;
    inst.src_num = 4;

    inst.type_ = T::Exp;
    inst.export_enable = en;

    match target {
        0x00 => {
            if done != 0 && compr != 0 && vm != 0 && en == 0x0 {
                inst.format = F::Mrt0OffOffComprVmDone;
                inst.src_num = 0;
            } else if done != 0 && compr != 0 && vm != 0 && en == 0xf {
                inst.format = F::Mrt0Vsrc0Vsrc1ComprVmDone;
                inst.src_num = 2;
            } else if done != 0 && compr == 0 && vm != 0 && en != 0 {
                // Uncompressed MRT0 export. Kyty knows only the full en==0xf
                // form (ShaderSpirv.cpp L2348); a partial mask selects which
                // full-float VGPRs are written (measured: ASTRO.BOT PS exports
                // en=0x3 — rg only, a 32_GR render target). `export_enable`
                // carries the mask; the recompiler writes the GCN default
                // (0, 0, 0, 1) to the disabled channels.
                inst.format = F::Mrt0Vsrc0Vsrc1Vsrc2Vsrc3VmDone;
            }
        }
        0x0c if done != 0 && en == 0xf => {
            inst.format = F::Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done;
        }
        // Auxiliary position exports pos1..pos3 — beyond Kyty, which knows
        // only pos0 (ShaderParse.cpp L2313-2316) and EXITs here. They carry
        // clip/cull distances or point size selected by PA_CL_VS_OUT_CNTL
        // (shadPS4 `ir/position.h` `ExportPosition`). Any channel-enable mask
        // is legal (measured: ASTRO.BOT exports pos1 with en=0x4, done=0);
        // `export_enable` records it for the recompiler.
        0x0d => inst.format = F::Pos1Vsrc0Vsrc1Vsrc2Vsrc3,
        0x0e => inst.format = F::Pos2Vsrc0Vsrc1Vsrc2Vsrc3,
        0x0f => inst.format = F::Pos3Vsrc0Vsrc1Vsrc2Vsrc3,
        0x14 if done != 0 && en == 0x1 => {
            inst.format = F::PrimVsrc0OffOffOffDone;
            inst.src_num = 1;
        }
        _ => {}
    }

    // Param exports (PARAM0..) carry a channel-enable mask: a full export is
    // en=0xf, but a vec2 texcoord is 0x3 and a vec3 normal 0x7. The mask is
    // recorded in `export_enable` and the recompiler writes 0 to the disabled
    // channels, so any `en` is accepted here — the earlier `en == 0xf` gate
    // rejected every partial param export and failed the whole vertex shader.
    if inst.format == F::Unknown && done == 0 && compr == 0 && vm == 0 && en != 0 {
        match target {
            0x20 => inst.format = F::Param0Vsrc0Vsrc1Vsrc2Vsrc3,
            0x21 => inst.format = F::Param1Vsrc0Vsrc1Vsrc2Vsrc3,
            0x22 => inst.format = F::Param2Vsrc0Vsrc1Vsrc2Vsrc3,
            0x23 => inst.format = F::Param3Vsrc0Vsrc1Vsrc2Vsrc3,
            0x24 => inst.format = F::Param4Vsrc0Vsrc1Vsrc2Vsrc3,
            _ => {}
        }
    }

    if inst.format == F::Unknown {
        // Kyty L2342-2348: dump + EXIT on unknown exp target.
        tracing::error!(
            "unknown exp target: 0x{target:02x} at addr 0x{pc:08x} \
             (en=0x{en:x} done={done} compr={compr} vm={vm}) \
             (hash0 = 0x{:08x}, crc32 = 0x{:08x})\n{}",
            dst.get_hash0(),
            dst.get_crc32(),
            dst.dbg_dump()
        );
        return Err(ShaderParseError::UnknownExpTarget { target, pc });
    }

    dst.get_instructions_mut().push(inst);

    Ok(2)
}

/// Kyty: ShaderParse.cpp `shader_parse_smem` (L2356) — next-gen scalar memory
/// encoding (top-level 0x3d).
fn shader_parse_smem(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "smem";
    let b0 = buffer[0];
    let b1 = dw(buffer, 1, pc)?;

    let opcode = (b0 >> 18) & 0xff;
    let glc = (b0 >> 16) & 0x1;
    let dlc = (b0 >> 14) & 0x1;
    let sdst = (b0 >> 6) & 0x7f;
    let sbase = b0 & 0x3f;
    let soffset = (b1 >> 25) & 0x7f;
    let offset = b1 & 0x1fffff;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.dst = operand_parse(sdst)?;
    inst.src_num = 2;
    inst.src[0] = operand_parse(sbase * 2)?;
    inst.src[1] = operand_parse(soffset)?;

    let size: u32 = 2;

    // Kyty L2381-2384: EXIT_NOT_IMPLEMENTED checks.
    if glc != 0 {
        return Err(feature(S, "glc != 0", pc));
    }
    if dlc != 0 {
        return Err(feature(S, "dlc != 0", pc));
    }
    if inst.src[0].type_ == O::LiteralConstant {
        return Err(feature(S, "sbase is a literal", pc));
    }
    if inst.src[1].type_ == O::LiteralConstant {
        return Err(feature(S, "soffset is a literal", pc));
    }

    if inst.src[1].type_ == O::Null {
        // Kyty L2386-2397: NULL soffset means the 21-bit signed immediate
        // offset is used (sign-extended via a 21-bit bitfield).
        let imm21 = ((offset << 11) as i32) >> 11;
        inst.src[1].type_ = O::IntegerInlineConstant;
        inst.src[1].constant.u = imm21 as u32;
        inst.src[1].size = 0;
    } else if offset != 0 {
        // Kyty L2400: EXIT_NOT_IMPLEMENTED(offset != 0).
        return Err(feature(S, "offset != 0 with register soffset", pc));
    }

    match opcode {
        0x00 => {
            inst.type_ = T::SLoadDword;
            inst.format = F::SdstSbaseSoffset;
            inst.src[0].size = 2;
            inst.dst.size = 1;
        }
        0x01 => {
            inst.type_ = T::SLoadDwordx2;
            inst.format = F::Sdst2Ssrc02Ssrc1;
            inst.src[0].size = 2;
            inst.dst.size = 2;
        }
        0x02 => {
            inst.type_ = T::SLoadDwordx4;
            inst.format = F::Sdst4SbaseSoffset;
            inst.src[0].size = 2;
            inst.dst.size = 4;
        }
        0x03 => {
            inst.type_ = T::SLoadDwordx8;
            inst.format = F::Sdst8SbaseSoffset;
            inst.src[0].size = 2;
            inst.dst.size = 8;
        }
        0x04 => return Err(ni(dst, S, "s_load_dwordx16", opcode, pc, b0)),
        0x08 => {
            inst.type_ = T::SBufferLoadDword;
            inst.format = F::SdstSvSoffset;
            inst.src[0].size = 4;
            inst.dst.size = 1;
        }
        0x09 => {
            inst.type_ = T::SBufferLoadDwordx2;
            inst.format = F::Sdst2SvSoffset;
            inst.src[0].size = 4;
            inst.dst.size = 2;
        }
        0x0a => {
            inst.type_ = T::SBufferLoadDwordx4;
            inst.format = F::Sdst4SvSoffset;
            inst.src[0].size = 4;
            inst.dst.size = 4;
        }
        0x0b => {
            inst.type_ = T::SBufferLoadDwordx8;
            inst.format = F::Sdst8SvSoffset;
            inst.src[0].size = 4;
            inst.dst.size = 8;
        }
        0x0c => {
            inst.type_ = T::SBufferLoadDwordx16;
            inst.format = F::Sdst16SvSoffset;
            inst.src[0].size = 4;
            inst.dst.size = 16;
        }
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_smrd` (L2450) — legacy scalar memory
/// encoding (top-level mask 0xC0000000).
fn shader_parse_smrd(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "smrd";
    let b0 = buffer[0];

    let opcode = (b0 >> 22) & 0x1f;
    let sdst = (b0 >> 15) & 0x7f;
    let sbase = (b0 >> 9) & 0x3f;
    let imm = (b0 >> 8) & 0x1;
    let offset = b0 & 0xff;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.dst = operand_parse(sdst)?;
    inst.src_num = 2;
    inst.src[0] = operand_parse(sbase * 2)?;

    let mut size: u32 = 1;

    if imm == 1 {
        inst.src[1].type_ = O::LiteralConstant;
        inst.src[1].constant.u = offset << 2;
    } else {
        inst.src[1] = operand_parse(offset)?;

        if inst.src[1].type_ == O::LiteralConstant {
            inst.src[1].constant.u = dw(buffer, size, pc)?;
            size += 1;
        }
    }

    match opcode {
        0x00 => return Err(ni(dst, S, "s_load_dword", opcode, pc, b0)),
        0x01 => return Err(ni(dst, S, "s_load_dwordx2", opcode, pc, b0)),
        0x02 => {
            inst.type_ = T::SLoadDwordx4;
            inst.format = F::Sdst4SbaseSoffset;
            inst.src[0].size = 2;
            inst.dst.size = 4;
        }
        0x03 => {
            inst.type_ = T::SLoadDwordx8;
            inst.format = F::Sdst8SbaseSoffset;
            inst.src[0].size = 2;
            inst.dst.size = 8;
        }
        0x04 => return Err(ni(dst, S, "s_load_dwordx16", opcode, pc, b0)),
        0x08 => {
            inst.type_ = T::SBufferLoadDword;
            inst.format = F::SdstSvSoffset;
            inst.src[0].size = 4;
        }
        0x09 => {
            inst.type_ = T::SBufferLoadDwordx2;
            inst.format = F::Sdst2SvSoffset;
            inst.src[0].size = 4;
            inst.dst.size = 2;
        }
        0x0a => {
            inst.type_ = T::SBufferLoadDwordx4;
            inst.format = F::Sdst4SvSoffset;
            inst.src[0].size = 4;
            inst.dst.size = 4;
        }
        0x0b => {
            inst.type_ = T::SBufferLoadDwordx8;
            inst.format = F::Sdst8SvSoffset;
            inst.src[0].size = 4;
            inst.dst.size = 8;
        }
        0x0c => {
            inst.type_ = T::SBufferLoadDwordx16;
            inst.format = F::Sdst16SvSoffset;
            inst.src[0].size = 4;
            inst.dst.size = 16;
        }
        0x1c => return Err(ni(dst, S, "s_memrealtime", opcode, pc, b0)),
        0x1d => return Err(ni(dst, S, "s_dcache_inv_vol", opcode, pc, b0)),
        0x1e => return Err(ni(dst, S, "s_memtime", opcode, pc, b0)),
        0x1f => return Err(ni(dst, S, "s_dcache_inv", opcode, pc, b0)),
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_mubuf` (L2547).
fn shader_parse_mubuf(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "mubuf";
    let b0 = buffer[0];
    let b1 = dw(buffer, 1, pc)?;

    let opcode = (b0 >> 18) & 0x1f;
    let lds = (b0 >> 16) & 0x1;
    let glc = (b0 >> 14) & 0x1;
    let idxen = (b0 >> 13) & 0x1;
    let offen = (b0 >> 12) & 0x1;
    let offset = b0 & 0xfff;

    let soffset = (b1 >> 24) & 0xff;
    let tfe = (b1 >> 23) & 0x1;
    let slc = (b1 >> 22) & 0x1;
    let srsrc = (b1 >> 16) & 0x1f;
    let vdata = (b1 >> 8) & 0xff;
    let vaddr = b1 & 0xff;

    // Kyty L2569-2575: EXIT_NOT_IMPLEMENTED checks. Beyond Kyty: idxen/offen
    // are no longer a blanket gate — the flexible opcodes below (single-dword
    // loads/stores, format x, format xyzw, dwordx4) select their format from
    // the (idxen, offen) addressing mode, the model the BufferLoadDwordX4
    // rows established: address = base + soffset + offset
    // + (idxen ? vindex * stride : 0) + (offen ? voffset : 0). The remaining
    // opcodes keep Kyty's strict gate, applied per-opcode AFTER the opcode is
    // known so a rejection names the instruction (previously 114 ASTRO.BOT
    // failures said only "idxen == 0").
    //
    // The 12-bit immediate `offset` is one addend of that same documented
    // address model. Kyty EXITs on it (L2571); here it is folded into the
    // soffset operand below once src[2] is known — both are plain byte
    // addends, and every recompile body already routes src[2] into
    // `temp_int_2` (the instruction-offset slot). 116 measured ASTRO.BOT CS
    // failures ("offset != 0").
    if glc == 1 {
        return Err(feature(S, "glc == 1", pc));
    }
    if slc == 1 {
        return Err(feature(S, "slc == 1", pc));
    }
    if lds == 1 {
        return Err(feature(S, "lds == 1", pc));
    }
    if tfe == 1 {
        return Err(feature(S, "tfe == 1", pc));
    }

    let size: u32 = 2;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.dst = operand_parse(vdata + 256)?;
    inst.src_num = 3;
    inst.src[0] = operand_parse(vaddr + 256)?;
    inst.src[1] = operand_parse(srsrc * 4)?;
    inst.src[2] = operand_parse(soffset)?;

    // MUBUF is a fixed 64-bit encoding: it cannot acquire a third SALU-style
    // literal dword. The measured PS5 stream uses the all-ones SOFFSET encoding
    // as its zero/no-extra-word form. Keep that console-specific meaning scoped
    // to the next-gen path; legacy decoding has no evidence for the value and
    // must refuse it rather than silently consuming the next instruction or
    // inventing equivalent semantics.
    if inst.src[2].type_ == O::LiteralConstant {
        if !next_gen {
            return Err(feature(
                S,
                "legacy literal soffset in fixed-width MUBUF",
                pc,
            ));
        }
        inst.src[2].type_ = O::IntegerInlineConstant;
        inst.src[2].constant.u = 0;
        inst.src[2].size = 0;
    }

    // Fold the immediate offset into a constant soffset (see the note above
    // the glc gate). A register soffset would need an extra runtime add no
    // recompile body models yet — that combination stays a named refusal.
    if offset != 0 {
        match inst.src[2].type_ {
            O::LiteralConstant | O::IntegerInlineConstant => {
                inst.src[2].constant.u = inst.src[2].constant.u.wrapping_add(offset);
            }
            _ => return Err(feature(S, "offset != 0 with register soffset", pc)),
        }
    }

    // Single-dword flexible addressing (the Vdata1 counterpart of the
    // Vdata4 quartet below).
    let format1 = match (idxen, offen) {
        (1, 1) => F::Vdata1Vaddr2SvSoffsOffenIdxen,
        (1, 0) => F::Vdata1VaddrSvSoffsIdxen,
        (0, 1) => F::Vdata1VaddrSvSoffsOffen,
        _ => F::Vdata1SvSoffs,
    };
    // Four-dword flexible addressing — measured on Minecraft's menu VS
    // (`buffer_load_dwordx4 v[8:11], v[4:5], s[8:11]` with idxen+offen:
    // vindex=v4, voffset=v5) and ASTRO.BOT's `buffer_store_format_xyzw`.
    let format4 = match (idxen, offen) {
        (1, 1) => F::Vdata4Vaddr2SvSoffsOffenIdxen,
        (1, 0) => F::Vdata4VaddrSvSoffsIdxen,
        (0, 1) => F::Vdata4VaddrSvSoffsOffen,
        _ => F::Vdata4SvSoffs,
    };
    let src0_size = (idxen + offen).max(1) as i32;
    // Kyty's per-opcode strict gate (upstream applies it globally before the
    // opcode switch, L2569-2570).
    let strict = |feat_ok: bool, feat: &'static str| -> Result<(), ShaderParseError> {
        if feat_ok {
            Ok(())
        } else {
            Err(feature(S, feat, pc))
        }
    };

    match opcode {
        0x00 => {
            inst.type_ = T::BufferLoadFormatX;
            inst.format = format1;
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
        }
        0x01 => {
            strict(idxen == 1, "idxen == 0")?;
            strict(offen == 0, "offen == 1")?;
            inst.type_ = T::BufferLoadFormatXy;
            inst.format = F::Vdata2VaddrSvSoffsIdxen;
            inst.src[1].size = 4;
            inst.dst.size = 2;
        }
        0x02 => {
            strict(idxen == 1, "idxen == 0")?;
            strict(offen == 0, "offen == 1")?;
            inst.type_ = T::BufferLoadFormatXyz;
            inst.format = F::Vdata3VaddrSvSoffsIdxen;
            inst.src[1].size = 4;
            inst.dst.size = 3;
        }
        0x03 => {
            strict(idxen == 1, "idxen == 0")?;
            strict(offen == 0, "offen == 1")?;
            inst.type_ = T::BufferLoadFormatXyzw;
            inst.format = F::Vdata4VaddrSvSoffsIdxen;
            inst.src[1].size = 4;
            inst.dst.size = 4;
        }
        0x04 => {
            inst.type_ = T::BufferStoreFormatX;
            inst.format = format1;
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
        }
        0x05 => {
            strict(idxen == 1, "idxen == 0")?;
            strict(offen == 0, "offen == 1")?;
            inst.type_ = T::BufferStoreFormatXy;
            inst.format = F::Vdata2VaddrSvSoffsIdxen;
            inst.src[1].size = 4;
            inst.dst.size = 2;
        }
        // 0x06/0x07 are KYTY_NI upstream (ShaderParse.cpp L2629-2630);
        // measured on ASTRO.BOT scene compute (raw 0xe01c2000 = xyzw store
        // with idxen).
        0x06 => {
            strict(idxen == 1, "idxen == 0")?;
            strict(offen == 0, "offen == 1")?;
            inst.type_ = T::BufferStoreFormatXyz;
            inst.format = F::Vdata3VaddrSvSoffsIdxen;
            inst.src[1].size = 4;
            inst.dst.size = 3;
        }
        0x07 => {
            inst.type_ = T::BufferStoreFormatXyzw;
            inst.format = format4;
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
            inst.dst.size = 4;
        }
        // Beyond Kyty (KYTY_NI upstream): single byte load, zero-extended.
        // Measured on ASTRO.BOT scene compute (raw 0xe02020c0: idxen with
        // immediate offset 0xc0; 58 dispatches/run). Same flexible
        // addressing quartet as buffer_load_dword; the recompiler extracts
        // the byte from the containing dword.
        0x08 => {
            inst.type_ = T::BufferLoadUbyte;
            inst.format = format1;
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
        }
        0x09 => return Err(ni(dst, S, "buffer_load_sbyte", opcode, pc, b0)),
        0x0a => return Err(ni(dst, S, "buffer_load_ushort", opcode, pc, b0)),
        0x0b => return Err(ni(dst, S, "buffer_load_sshort", opcode, pc, b0)),
        0x0c => {
            inst.type_ = T::BufferLoadDword;
            inst.format = format1;
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
        }
        // Beyond Kyty (KYTY_NI upstream): two-dword raw load, measured on
        // ASTRO.BOT scene compute (raw 0xe0342000, idxen). Same flexible
        // addressing quartet as the Vdata1/Vdata4 opcodes.
        0x0d => {
            inst.type_ = T::BufferLoadDwordX2;
            inst.format = match (idxen, offen) {
                (1, 1) => F::Vdata2Vaddr2SvSoffsOffenIdxen,
                (1, 0) => F::Vdata2VaddrSvSoffsIdxen,
                (0, 1) => F::Vdata2VaddrSvSoffsOffen,
                _ => F::Vdata2SvSoffs,
            };
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
            inst.dst.size = 2;
        }
        0x0e => {
            inst.type_ = T::BufferLoadDwordX4;
            inst.format = format4;
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
            inst.dst.size = 4;
        }
        // Beyond Kyty (KYTY_NI upstream): three-dword raw load, measured on
        // ASTRO.BOT scene compute (raws 0xe03c2074/0xe03c2034, idxen with a
        // nonzero immediate offset). Same flexible addressing quartet as the
        // Vdata1/2/4 opcodes.
        0x0f => {
            inst.type_ = T::BufferLoadDwordX3;
            inst.format = match (idxen, offen) {
                (1, 1) => F::Vdata3Vaddr2SvSoffsOffenIdxen,
                (1, 0) => F::Vdata3VaddrSvSoffsIdxen,
                (0, 1) => F::Vdata3VaddrSvSoffsOffen,
                _ => F::Vdata3SvSoffs,
            };
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
            inst.dst.size = 3;
        }
        0x18 => return Err(ni(dst, S, "buffer_store_byte", opcode, pc, b0)),
        0x1a => return Err(ni(dst, S, "buffer_store_short", opcode, pc, b0)),
        0x1c => {
            inst.type_ = T::BufferStoreDword;
            inst.format = format1;
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
        }
        // Beyond Kyty (KYTY_NI upstream): two-dword raw store, measured on
        // ASTRO.BOT scene compute (0x500757800). Same flexible addressing
        // quartet as BufferLoadDwordX2; the store data is the two-dword vdata
        // register (`inst.dst`).
        0x1d => {
            inst.type_ = T::BufferStoreDwordX2;
            inst.format = match (idxen, offen) {
                (1, 1) => F::Vdata2Vaddr2SvSoffsOffenIdxen,
                (1, 0) => F::Vdata2VaddrSvSoffsIdxen,
                (0, 1) => F::Vdata2VaddrSvSoffsOffen,
                _ => F::Vdata2SvSoffs,
            };
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
            inst.dst.size = 2;
        }
        // Beyond Kyty (KYTY_NI upstream): four-dword raw store, measured on
        // ASTRO.BOT scene compute (raw 0xe0780000). Same flexible addressing
        // quartet as BufferLoadDwordX4.
        0x1e => {
            inst.type_ = T::BufferStoreDwordX4;
            inst.format = format4;
            inst.src[0].size = src0_size;
            inst.src[1].size = 4;
            inst.dst.size = 4;
        }
        0x1f => return Err(ni(dst, S, "buffer_store_dwordx3", opcode, pc, b0)),
        // Kyty's table continues past the 5-bit opcode range (0x30-0x87
        // atomics/d16, incl. next_gen-gated 0x34/0x71 — ShaderParse.cpp
        // L2653-2711). Those arms are unreachable with the
        // (buffer[0] >> 18) & 0x1f decode Kyty itself uses, so the port
        // folds them into UnknownOpcode.
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// GFX10/RDNA2 FLAT-class decoder (FLAT / GLOBAL / SCRATCH segments, encoding
/// `0x37`). Kyty's SI/GNM parser has no FLAT class; ported from SharpEmu PR
/// #587 (`Gen5ShaderTranslator.DecodeFlat`, GPL-2.0).
///
/// Encoding (little-endian, two dwords):
/// * word0: `offset[12:0]`, `seg[15:14]` (0 = FLAT, 1 = SCRATCH, 2 = GLOBAL),
///   `glc[16]`, `slc[17]`, `op[24:18]`, `enc[31:26] = 0x37`.
/// * word1: `addr[7:0]` (VGPR), `data[15:8]` (VGPR store source),
///   `saddr[22:16]` (SGPR base; `0x7f` = NULL), `vdst[31:24]` (VGPR load dest).
///
/// The FLAT segment holds the whole 64-bit address in the VGPR pair
/// `(addr, addr+1)`; a GLOBAL op with a real SADDR uses an SGPR base pair plus
/// a 32-bit VGPR offset. `SharpEmu`'s `UsesFlatAddress` (true for the FLAT
/// segment, and for a GLOBAL segment whose SADDR is NULL) rides on
/// [`ShaderInstruction::uses_flat_address`]. `glc`/`slc`/`dlc` are ignored —
/// the recompiler serves all of guest memory from one coherent window.
fn shader_parse_flat(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "flat";
    let b0 = buffer[0];
    let b1 = dw(buffer, 1, pc)?;

    // The FLAT class is an RDNA2 (next-gen/PS5) encoding; a legacy stream
    // reaching 0x37 is a decode error, not a FLAT op.
    if !next_gen {
        return Err(feature(S, "FLAT-class encoding on legacy", pc));
    }

    let offset = b0 & 0x1fff; // [12:0]
    let seg = (b0 >> 14) & 0x3; // [15:14]
    let opcode = (b0 >> 18) & 0x7f; // [24:18]

    let addr = b1 & 0xff; // [7:0]  VGPR
    let data = (b1 >> 8) & 0xff; // [15:8] VGPR store data
    let saddr = (b1 >> 16) & 0x7f; // [22:16] SGPR base (0x7f = NULL)
    let vdst = (b1 >> 24) & 0xff; // [31:24] VGPR load dest

    // Segment selects the addressing form (SharpEmu Gen5 `segment` switch).
    let uses_flat_segment = match seg {
        0x0 => true,                                          // FLAT
        0x2 => false,                                         // GLOBAL
        0x1 => return Err(feature(S, "scratch segment", pc)), // stack spill — not hot path
        _ => return Err(feature(S, "reserved segment", pc)),
    };
    let saddr_null = saddr == 0x7f;
    // The address is a full 64-bit VGPR pair for the FLAT segment, or a GLOBAL
    // op that left SADDR NULL. Otherwise SADDR names the base pair and the VGPR
    // is a 32-bit per-lane offset (SharpEmu `UsesFlatAddress`).
    let flat_addressing = uses_flat_segment || saddr_null;

    let mut inst = ShaderInstruction {
        pc,
        uses_flat_address: flat_addressing,
        ..Default::default()
    };
    inst.format = F::FlatAddr;
    inst.src_num = 3;

    // src[0]: VGPR address (pair when flat-addressed, else a 32-bit offset).
    inst.src[0] = operand_parse(addr + 256)?;
    inst.src[0].size = if flat_addressing { 2 } else { 1 };

    // src[1]: SGPR base pair, or NULL when the VGPR carries the whole address.
    if flat_addressing {
        inst.src[1] = ShaderOperand {
            type_: O::Null,
            size: 2,
            ..Default::default()
        };
    } else {
        inst.src[1] = operand_parse(saddr)?;
        inst.src[1].size = 2;
    }

    // src[2]: immediate byte offset. GLOBAL/SCRATCH sign-extend the 13-bit
    // field; FLAT uses the unsigned low 11 bits (GFX10 reserves bits 12:11).
    let off_val = if uses_flat_segment {
        offset & 0x7ff
    } else {
        (((offset & 0x1fff) << 19) as i32 >> 19) as u32
    };
    inst.src[2] = ShaderOperand {
        type_: O::IntegerInlineConstant,
        constant: ShaderConstant::from_u(off_val),
        size: 0,
        ..Default::default()
    };

    // Opcode suffix -> operation + dword width (SharpEmu Gen5 `suffix` switch).
    let (type_, dst_size, is_store, name) = match opcode {
        0x08 => (T::FlatLoadUbyte, 1, false, "flat_load_ubyte"),
        0x0c => (T::FlatLoadDword, 1, false, "flat_load_dword"),
        0x0d => (T::FlatLoadDwordX2, 2, false, "flat_load_dwordx2"),
        0x0e => (T::FlatLoadDwordX4, 4, false, "flat_load_dwordx4"),
        0x0f => (T::FlatLoadDwordX3, 3, false, "flat_load_dwordx3"),
        0x1c => (T::FlatStoreDword, 1, true, "flat_store_dword"),
        0x1d => (T::FlatStoreDwordX2, 2, true, "flat_store_dwordx2"),
        0x1e => (T::FlatStoreDwordX4, 4, true, "flat_store_dwordx4"),
        // Named-NI for the byte/short/atomic suffixes SharpEmu also decodes but
        // that no measured PS5 hot-path shader has needed yet.
        0x09 => return Err(ni(dst, S, "flat_load_sbyte", opcode, pc, b0)),
        0x0a => return Err(ni(dst, S, "flat_load_ushort", opcode, pc, b0)),
        0x0b => return Err(ni(dst, S, "flat_load_sshort", opcode, pc, b0)),
        0x18 => return Err(ni(dst, S, "flat_store_byte", opcode, pc, b0)),
        0x1a => return Err(ni(dst, S, "flat_store_short", opcode, pc, b0)),
        0x1f => return Err(ni(dst, S, "flat_store_dwordx3", opcode, pc, b0)),
        0x32 => return Err(ni(dst, S, "flat_atomic_add", opcode, pc, b0)),
        0x38 => return Err(ni(dst, S, "flat_atomic_umax", opcode, pc, b0)),
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    };
    // Name is carried only for the (unreachable here) NI arms above; keep it
    // referenced so the match's binding is not flagged unused.
    let _ = name;

    inst.type_ = type_;
    // Load writes VDST; store reads the DATA VGPR (mirrors MUBUF's `vdata`).
    inst.dst = operand_parse(if is_store { data } else { vdst } + 256)?;
    inst.dst.size = dst_size;

    dst.get_instructions_mut().push(inst);

    Ok(2)
}

/// Kyty: ShaderParse.cpp `shader_parse_ds` (L2722). Kyty only implements the
/// GDS append/consume pair; everything else is named-NI.
#[allow(clippy::too_many_lines)]
fn shader_parse_ds(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "ds";
    let b0 = buffer[0];
    let b1 = dw(buffer, 1, pc)?;

    let opcode = (b0 >> 18) & 0xff;
    let gds = (b0 >> 17) & 0x1;
    let offset0 = b0 & 0xff;
    let offset1 = (b0 >> 8) & 0xff;

    let vdst = (b1 >> 24) & 0xff;
    let data1 = (b1 >> 16) & 0xff;
    let data0 = (b1 >> 8) & 0xff;
    // addr is a don't-care for the append/consume pair (they select the GDS
    // counter through M0) but is the LDS address VGPR for ds_write/ds_read.
    let addr = b1 & 0xff;

    // Kyty applies its EXIT_NOT_IMPLEMENTED operand checks (L2740-2745)
    // BEFORE the opcode switch, so an unimplemented LDS op with a non-zero
    // addr/data/offset field died as "addr != 0" — 173 ASTRO.BOT failures
    // with no instruction name. Deviation (diagnosis only): the opcode switch
    // runs first so every unimplemented DS op reports its own name; the
    // operand checks now guard only the implemented append/consume pair
    // below, exactly as strictly as upstream.
    let size: u32 = 2;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.dst = operand_parse(vdst + 256)?;
    inst.src_num = 0;

    if matches!(opcode, 0x3d | 0x3e) {
        // Kyty L2740-2745 for DS_CONSUME/DS_APPEND. They select the GDS
        // counter through M0 and do not read the encoded address VGPR; real
        // Gen5 shaders leave that don't-care field non-zero, so `addr` is
        // deliberately not checked.
        //
        // Beyond Kyty: the 16-bit instruction offset is a BYTE offset added
        // to the M0 counter base (shadPS4 `DS_APPEND`/`DS_CONSUME`
        // translate `gds_offset = M0 + inst_offset`,
        // data_share.cpp L323-L335; resource_tracking_pass.cpp L699-L708
        // then indexes the GDS buffer at `gds_addr >> 2`). Kyty EXITs on any
        // nonzero offset — 59 measured ASTRO.BOT CS failures. The offset
        // rides as a literal src[0] (dword-aligned only; an unaligned
        // counter address has no defined uint slot).
        if data0 != 0 {
            return Err(feature(S, "data0 != 0", pc));
        }
        if data1 != 0 {
            return Err(feature(S, "data1 != 0", pc));
        }
        let counter_offset = offset0 | (offset1 << 8);
        if counter_offset & 3 != 0 {
            return Err(feature(S, "append/consume offset not dword-aligned", pc));
        }
        if counter_offset != 0 {
            inst.src[0].type_ = O::LiteralConstant;
            inst.src[0].constant.u = counter_offset;
            inst.src_num = 1;
        }
        if gds == 0 {
            return Err(feature(S, "gds == 0", pc));
        }
    }

    match opcode {
        // Beyond Kyty (KYTY_NI upstream): LDS atomic dword add without
        // return, measured on ASTRO.BOT scene compute (raw 0xd8000514,
        // offset 0x514). Same operand shape as ds_write_b32: src0 = address
        // VGPR, src1 = data VGPR, src2 = the 16-bit instruction byte offset.
        0x00 => {
            if gds != 0 {
                return Err(feature(S, "ds_add_u32 with gds == 1", pc));
            }
            inst.type_ = T::DsAddU32;
            inst.format = F::Vsrc0Vsrc1Vsrc2;
            inst.dst = ShaderOperand::default();
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1] = operand_parse(data0 + 256)?;
            inst.src[2].type_ = O::LiteralConstant;
            inst.src[2].constant.u = offset0 | (offset1 << 8);
            inst.src_num = 3;
        }
        0x01 => return Err(ni(dst, S, "ds_sub_u32", opcode, pc, b0)),
        0x02 => return Err(ni(dst, S, "ds_rsub_u32", opcode, pc, b0)),
        0x03 => return Err(ni(dst, S, "ds_inc_u32", opcode, pc, b0)),
        0x04 => return Err(ni(dst, S, "ds_dec_u32", opcode, pc, b0)),
        0x05 => return Err(ni(dst, S, "ds_min_i32", opcode, pc, b0)),
        0x06 => return Err(ni(dst, S, "ds_max_i32", opcode, pc, b0)),
        0x07 => return Err(ni(dst, S, "ds_min_u32", opcode, pc, b0)),
        0x08 => return Err(ni(dst, S, "ds_max_u32", opcode, pc, b0)),
        0x09 => return Err(ni(dst, S, "ds_and_b32", opcode, pc, b0)),
        0x0a => return Err(ni(dst, S, "ds_or_b32", opcode, pc, b0)),
        0x0b => return Err(ni(dst, S, "ds_xor_b32", opcode, pc, b0)),
        0x0c => return Err(ni(dst, S, "ds_mskor_b32", opcode, pc, b0)),
        // Beyond Kyty (KYTY_NI upstream): LDS dword write, measured on
        // ASTRO.BOT scene compute (raw 0xd8340000). src0 = address VGPR,
        // src1 = data VGPR, src2 = the 16-bit instruction byte offset.
        0x0d => {
            if gds != 0 {
                return Err(feature(S, "ds_write_b32 with gds == 1", pc));
            }
            inst.type_ = T::DsWriteB32;
            inst.format = F::Vsrc0Vsrc1Vsrc2;
            inst.dst = ShaderOperand::default();
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1] = operand_parse(data0 + 256)?;
            inst.src[2].type_ = O::LiteralConstant;
            inst.src[2].constant.u = offset0 | (offset1 << 8);
            inst.src_num = 3;
        }
        0x0e => return Err(ni(dst, S, "ds_write2_b32", opcode, pc, b0)),
        0x0f => return Err(ni(dst, S, "ds_write2st64_b32", opcode, pc, b0)),
        0x10 => return Err(ni(dst, S, "ds_cmpst_b32", opcode, pc, b0)),
        0x11 => return Err(ni(dst, S, "ds_cmpst_f32", opcode, pc, b0)),
        0x12 => return Err(ni(dst, S, "ds_min_f32", opcode, pc, b0)),
        0x13 => return Err(ni(dst, S, "ds_max_f32", opcode, pc, b0)),
        0x14 => return Err(ni(dst, S, "ds_nop", opcode, pc, b0)),
        0x18 => return Err(ni(dst, S, "ds_gws_sema_release_all", opcode, pc, b0)),
        0x19 => return Err(ni(dst, S, "ds_gws_init", opcode, pc, b0)),
        0x1a => return Err(ni(dst, S, "ds_gws_sema_v", opcode, pc, b0)),
        0x1b => return Err(ni(dst, S, "ds_gws_sema_br", opcode, pc, b0)),
        0x1c => return Err(ni(dst, S, "ds_gws_sema_p", opcode, pc, b0)),
        0x1d => return Err(ni(dst, S, "ds_gws_barrier", opcode, pc, b0)),
        0x1e => return Err(ni(dst, S, "ds_write_b8", opcode, pc, b0)),
        0x1f => return Err(ni(dst, S, "ds_write_b16", opcode, pc, b0)),
        0x20 => return Err(ni(dst, S, "ds_add_rtn_u32", opcode, pc, b0)),
        0x21 => return Err(ni(dst, S, "ds_sub_rtn_u32", opcode, pc, b0)),
        0x22 => return Err(ni(dst, S, "ds_rsub_rtn_u32", opcode, pc, b0)),
        0x23 => return Err(ni(dst, S, "ds_inc_rtn_u32", opcode, pc, b0)),
        0x24 => return Err(ni(dst, S, "ds_dec_rtn_u32", opcode, pc, b0)),
        0x25 => return Err(ni(dst, S, "ds_min_rtn_i32", opcode, pc, b0)),
        0x26 => return Err(ni(dst, S, "ds_max_rtn_i32", opcode, pc, b0)),
        0x27 => return Err(ni(dst, S, "ds_min_rtn_u32", opcode, pc, b0)),
        0x28 => return Err(ni(dst, S, "ds_max_rtn_u32", opcode, pc, b0)),
        0x29 => return Err(ni(dst, S, "ds_and_rtn_b32", opcode, pc, b0)),
        0x2a => return Err(ni(dst, S, "ds_or_rtn_b32", opcode, pc, b0)),
        0x2b => return Err(ni(dst, S, "ds_xor_rtn_b32", opcode, pc, b0)),
        0x2c => return Err(ni(dst, S, "ds_mskor_rtn_b32", opcode, pc, b0)),
        // Beyond Kyty (KYTY_NI upstream): LDS atomic write-exchange returning
        // the old value, measured on ASTRO.BOT tiled-lighting compute (raw
        // 0xd8b40510, offset 0x510). Same operand shape as ds_write_b32 plus a
        // return VGPR: dst = vdst (old value), src0 = address VGPR, src1 = data
        // VGPR, src2 = the 16-bit instruction byte offset.
        0x2d => {
            if gds != 0 {
                return Err(feature(S, "ds_wrxchg_rtn_b32 with gds == 1", pc));
            }
            inst.type_ = T::DsWrxchgRtnB32;
            inst.format = F::VdstVsrc0Vsrc1Vsrc2;
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1] = operand_parse(data0 + 256)?;
            inst.src[2].type_ = O::LiteralConstant;
            inst.src[2].constant.u = offset0 | (offset1 << 8);
            inst.src_num = 3;
        }
        0x2e => return Err(ni(dst, S, "ds_wrxchg2_rtn_b32", opcode, pc, b0)),
        0x2f => return Err(ni(dst, S, "ds_wrxchg2st64_rtn_b32", opcode, pc, b0)),
        0x30 => return Err(ni(dst, S, "ds_cmpst_rtn_b32", opcode, pc, b0)),
        0x31 => return Err(ni(dst, S, "ds_cmpst_rtn_f32", opcode, pc, b0)),
        0x32 => return Err(ni(dst, S, "ds_min_rtn_f32", opcode, pc, b0)),
        0x33 => return Err(ni(dst, S, "ds_max_rtn_f32", opcode, pc, b0)),
        0x34 => return Err(ni(dst, S, "ds_wrap_rtn_b32", opcode, pc, b0)),
        0x35 => return Err(ni(dst, S, "ds_swizzle_b32", opcode, pc, b0)),
        // The read twin of ds_write_b32 above: dst = vdst VGPR (already
        // parsed), src0 = address VGPR, src1 = the 16-bit byte offset.
        0x36 => {
            if gds != 0 {
                return Err(feature(S, "ds_read_b32 with gds == 1", pc));
            }
            if data0 != 0 || data1 != 0 {
                return Err(feature(S, "ds_read_b32 with data operands", pc));
            }
            inst.type_ = T::DsReadB32;
            inst.format = F::SVdstSVsrc0SVsrc1;
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1].type_ = O::LiteralConstant;
            inst.src[1].constant.u = offset0 | (offset1 << 8);
            inst.src_num = 2;
        }
        // Beyond Kyty (KYTY_NI upstream): two independent LDS dword reads.
        // RDNA2 `DS_READ2_B32`: offsets are in DWORD units (scaled by 4 here
        // so the stored literals are byte offsets like every other DS form);
        // results land in vdst and vdst+1. Measured on ASTRO.BOT scene
        // compute (raw 0xd8dc0100 = offset0 0, offset1 1).
        0x37 => {
            if gds != 0 {
                return Err(feature(S, "ds_read2_b32 with gds == 1", pc));
            }
            if data0 != 0 || data1 != 0 {
                return Err(feature(S, "ds_read2_b32 with data operands", pc));
            }
            inst.type_ = T::DsRead2B32;
            inst.format = F::Vdst2Vsrc0Vsrc1Vsrc2;
            inst.dst.size = 2;
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1].type_ = O::LiteralConstant;
            inst.src[1].constant.u = offset0 * 4;
            inst.src[2].type_ = O::LiteralConstant;
            inst.src[2].constant.u = offset1 * 4;
            inst.src_num = 3;
        }
        0x38 => return Err(ni(dst, S, "ds_read2st64_b32", opcode, pc, b0)),
        0x39 => return Err(ni(dst, S, "ds_read_i8", opcode, pc, b0)),
        0x3a => return Err(ni(dst, S, "ds_read_u8", opcode, pc, b0)),
        0x3b => return Err(ni(dst, S, "ds_read_i16", opcode, pc, b0)),
        0x3c => return Err(ni(dst, S, "ds_read_u16", opcode, pc, b0)),
        0x3d => {
            inst.type_ = T::DsConsume;
            inst.format = F::VdstGds;
        }
        0x3e => {
            inst.type_ = T::DsAppend;
            inst.format = F::VdstGds;
        }
        0x3f => return Err(ni(dst, S, "ds_ordered_count", opcode, pc, b0)),
        0x40 => return Err(ni(dst, S, "ds_add_u64", opcode, pc, b0)),
        0x41 => return Err(ni(dst, S, "ds_sub_u64", opcode, pc, b0)),
        0x42 => return Err(ni(dst, S, "ds_rsub_u64", opcode, pc, b0)),
        0x43 => return Err(ni(dst, S, "ds_inc_u64", opcode, pc, b0)),
        0x44 => return Err(ni(dst, S, "ds_dec_u64", opcode, pc, b0)),
        0x45 => return Err(ni(dst, S, "ds_min_i64", opcode, pc, b0)),
        0x46 => return Err(ni(dst, S, "ds_max_i64", opcode, pc, b0)),
        0x47 => return Err(ni(dst, S, "ds_min_u64", opcode, pc, b0)),
        0x48 => return Err(ni(dst, S, "ds_max_u64", opcode, pc, b0)),
        0x49 => return Err(ni(dst, S, "ds_and_b64", opcode, pc, b0)),
        0x4a => return Err(ni(dst, S, "ds_or_b64", opcode, pc, b0)),
        0x4b => return Err(ni(dst, S, "ds_xor_b64", opcode, pc, b0)),
        0x4c => return Err(ni(dst, S, "ds_mskor_b64", opcode, pc, b0)),
        0x4d => return Err(ni(dst, S, "ds_write_b64", opcode, pc, b0)),
        0x4e => return Err(ni(dst, S, "ds_write2_b64", opcode, pc, b0)),
        0x4f => return Err(ni(dst, S, "ds_write2st64_b64", opcode, pc, b0)),
        0x50 => return Err(ni(dst, S, "ds_cmpst_b64", opcode, pc, b0)),
        0x51 => return Err(ni(dst, S, "ds_cmpst_f64", opcode, pc, b0)),
        0x52 => return Err(ni(dst, S, "ds_min_f64", opcode, pc, b0)),
        0x53 => return Err(ni(dst, S, "ds_max_f64", opcode, pc, b0)),
        0x60 => return Err(ni(dst, S, "ds_add_rtn_u64", opcode, pc, b0)),
        0x61 => return Err(ni(dst, S, "ds_sub_rtn_u64", opcode, pc, b0)),
        0x62 => return Err(ni(dst, S, "ds_rsub_rtn_u64", opcode, pc, b0)),
        0x63 => return Err(ni(dst, S, "ds_inc_rtn_u64", opcode, pc, b0)),
        0x64 => return Err(ni(dst, S, "ds_dec_rtn_u64", opcode, pc, b0)),
        0x65 => return Err(ni(dst, S, "ds_min_rtn_i64", opcode, pc, b0)),
        0x66 => return Err(ni(dst, S, "ds_max_rtn_i64", opcode, pc, b0)),
        0x67 => return Err(ni(dst, S, "ds_min_rtn_u64", opcode, pc, b0)),
        0x68 => return Err(ni(dst, S, "ds_max_rtn_u64", opcode, pc, b0)),
        0x69 => return Err(ni(dst, S, "ds_and_rtn_b64", opcode, pc, b0)),
        0x6a => return Err(ni(dst, S, "ds_or_rtn_b64", opcode, pc, b0)),
        0x6b => return Err(ni(dst, S, "ds_xor_rtn_b64", opcode, pc, b0)),
        0x6c => return Err(ni(dst, S, "ds_mskor_rtn_b64", opcode, pc, b0)),
        0x6d => return Err(ni(dst, S, "ds_wrxchg_rtn_b64", opcode, pc, b0)),
        0x6e => return Err(ni(dst, S, "ds_wrxchg2_rtn_b64", opcode, pc, b0)),
        0x6f => return Err(ni(dst, S, "ds_wrxchg2st64_rtn_b64", opcode, pc, b0)),
        0x70 => return Err(ni(dst, S, "ds_cmpst_rtn_b64", opcode, pc, b0)),
        0x71 => return Err(ni(dst, S, "ds_cmpst_rtn_f64", opcode, pc, b0)),
        0x72 => return Err(ni(dst, S, "ds_min_rtn_f64", opcode, pc, b0)),
        0x73 => return Err(ni(dst, S, "ds_max_rtn_f64", opcode, pc, b0)),
        // Beyond Kyty (KYTY_NI upstream): two consecutive LDS dwords at the
        // single 16-bit BYTE offset (RDNA2 ISA `DS_READ_B64`) — measured on
        // ASTRO.BOT scene compute (raw 0xd9d80000). Reuses the `DsRead2B32`
        // operand shape with the second offset literal at `offset + 4`, so
        // the existing two-dword recompile body covers it.
        0x76 => {
            if gds != 0 {
                return Err(feature(S, "ds_read_b64 with gds == 1", pc));
            }
            if data0 != 0 || data1 != 0 {
                return Err(feature(S, "ds_read_b64 with data operands", pc));
            }
            let offset = offset0 | (offset1 << 8);
            inst.type_ = T::DsReadB64;
            inst.format = F::Vdst2Vsrc0Vsrc1Vsrc2;
            inst.dst.size = 2;
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1].type_ = O::LiteralConstant;
            inst.src[1].constant.u = offset;
            inst.src[2].type_ = O::LiteralConstant;
            inst.src[2].constant.u = offset + 4;
            inst.src_num = 3;
        }
        0x77 => return Err(ni(dst, S, "ds_read2_b64", opcode, pc, b0)),
        0x78 => return Err(ni(dst, S, "ds_read2st64_b64", opcode, pc, b0)),
        0x7e => return Err(ni(dst, S, "ds_condxchg32_rtn_b64", opcode, pc, b0)),
        0x80 => return Err(ni(dst, S, "ds_add_src2_u32", opcode, pc, b0)),
        0x81 => return Err(ni(dst, S, "ds_sub_src2_u32", opcode, pc, b0)),
        0x82 => return Err(ni(dst, S, "ds_rsub_src2_u32", opcode, pc, b0)),
        0x83 => return Err(ni(dst, S, "ds_inc_src2_u32", opcode, pc, b0)),
        0x84 => return Err(ni(dst, S, "ds_dec_src2_u32", opcode, pc, b0)),
        0x85 => return Err(ni(dst, S, "ds_min_src2_i32", opcode, pc, b0)),
        0x86 => return Err(ni(dst, S, "ds_max_src2_i32", opcode, pc, b0)),
        0x87 => return Err(ni(dst, S, "ds_min_src2_u32", opcode, pc, b0)),
        0x88 => return Err(ni(dst, S, "ds_max_src2_u32", opcode, pc, b0)),
        0x89 => return Err(ni(dst, S, "ds_and_src2_b32", opcode, pc, b0)),
        0x8a => return Err(ni(dst, S, "ds_or_src2_b32", opcode, pc, b0)),
        0x8b => return Err(ni(dst, S, "ds_xor_src2_b32", opcode, pc, b0)),
        0x8d => return Err(ni(dst, S, "ds_write_src2_b32", opcode, pc, b0)),
        0x92 => return Err(ni(dst, S, "ds_min_src2_f32", opcode, pc, b0)),
        0x93 => return Err(ni(dst, S, "ds_max_src2_f32", opcode, pc, b0)),
        0xc0 => return Err(ni(dst, S, "ds_add_src2_u64", opcode, pc, b0)),
        0xc1 => return Err(ni(dst, S, "ds_sub_src2_u64", opcode, pc, b0)),
        0xc2 => return Err(ni(dst, S, "ds_rsub_src2_u64", opcode, pc, b0)),
        0xc3 => return Err(ni(dst, S, "ds_inc_src2_u64", opcode, pc, b0)),
        0xc4 => return Err(ni(dst, S, "ds_dec_src2_u64", opcode, pc, b0)),
        0xc5 => return Err(ni(dst, S, "ds_min_src2_i64", opcode, pc, b0)),
        0xc6 => return Err(ni(dst, S, "ds_max_src2_i64", opcode, pc, b0)),
        0xc7 => return Err(ni(dst, S, "ds_min_src2_u64", opcode, pc, b0)),
        0xc8 => return Err(ni(dst, S, "ds_max_src2_u64", opcode, pc, b0)),
        0xc9 => return Err(ni(dst, S, "ds_and_src2_b64", opcode, pc, b0)),
        0xca => return Err(ni(dst, S, "ds_or_src2_b64", opcode, pc, b0)),
        0xcb => return Err(ni(dst, S, "ds_xor_src2_b64", opcode, pc, b0)),
        0xcd => return Err(ni(dst, S, "ds_write_src2_b64", opcode, pc, b0)),
        0xd2 => return Err(ni(dst, S, "ds_min_src2_f64", opcode, pc, b0)),
        0xd3 => return Err(ni(dst, S, "ds_max_src2_f64", opcode, pc, b0)),
        // Beyond Kyty (KYTY_NI upstream): three consecutive LDS dwords
        // stored from data0..data0+2 at the 16-bit byte offset (RDNA2
        // `DS_WRITE_B96`). Measured on ASTRO.BOT scene compute.
        0xde => {
            if gds != 0 {
                return Err(feature(S, "ds_write_b96 with gds == 1", pc));
            }
            if data1 != 0 {
                return Err(feature(S, "ds_write_b96 with data1 operand", pc));
            }
            inst.type_ = T::DsWriteB96;
            inst.format = F::Vsrc0Vsrc13Vsrc2;
            inst.dst = ShaderOperand::default();
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1] = operand_parse(data0 + 256)?;
            inst.src[1].size = 3;
            inst.src[2].type_ = O::LiteralConstant;
            inst.src[2].constant.u = offset0 | (offset1 << 8);
            inst.src_num = 3;
        }
        // Beyond Kyty (KYTY_NI upstream): four consecutive LDS dwords stored
        // from data0..data0+3 at the 16-bit byte offset (RDNA2
        // `DS_WRITE_B128`). Measured on ASTRO.BOT scene compute
        // (raw 0xdb7c0000). Same model as the b96 arm above.
        0xdf => {
            if gds != 0 {
                return Err(feature(S, "ds_write_b128 with gds == 1", pc));
            }
            if data1 != 0 {
                return Err(feature(S, "ds_write_b128 with data1 operand", pc));
            }
            inst.type_ = T::DsWriteB128;
            inst.format = F::Vsrc0Vsrc14Vsrc2;
            inst.dst = ShaderOperand::default();
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1] = operand_parse(data0 + 256)?;
            inst.src[1].size = 4;
            inst.src[2].type_ = O::LiteralConstant;
            inst.src[2].constant.u = offset0 | (offset1 << 8);
            inst.src_num = 3;
        }
        0xfd => return Err(ni(dst, S, "ds_condxchg32_rtn_b128", opcode, pc, b0)),
        // Beyond Kyty (KYTY_NI upstream): three consecutive LDS dwords read
        // at the single 16-bit byte offset (RDNA2 `DS_READ_B96`) — measured
        // on ASTRO.BOT scene compute (raw 0xdbf80550, 58 dispatches/run).
        // The three-dword row of the b128 model below.
        0xfe => {
            if gds != 0 {
                return Err(feature(S, "ds_read_b96 with gds == 1", pc));
            }
            if data0 != 0 || data1 != 0 {
                return Err(feature(S, "ds_read_b96 with data operands", pc));
            }
            inst.type_ = T::DsReadB96;
            inst.format = F::Vdst3Vsrc0Vsrc1;
            inst.dst.size = 3;
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1].type_ = O::LiteralConstant;
            inst.src[1].constant.u = offset0 | (offset1 << 8);
            inst.src_num = 2;
        }
        // Beyond Kyty (KYTY_NI upstream): four consecutive LDS dwords read
        // at the single 16-bit BYTE offset (RDNA2 ISA `DS_READ_B128`) —
        // measured on ASTRO.BOT scene compute (58 dispatches/run). Extends
        // the b64 model: dword k reads at `offset + 4k` in the recompiler.
        0xff => {
            if gds != 0 {
                return Err(feature(S, "ds_read_b128 with gds == 1", pc));
            }
            if data0 != 0 || data1 != 0 {
                return Err(feature(S, "ds_read_b128 with data operands", pc));
            }
            inst.type_ = T::DsReadB128;
            inst.format = F::Vdst4Vsrc0Vsrc1;
            inst.dst.size = 4;
            inst.src[0] = operand_parse(addr + 256)?;
            inst.src[1].type_ = O::LiteralConstant;
            inst.src[1].constant.u = offset0 | (offset1 << 8);
            inst.src_num = 2;
        }
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_mimg` (L2912). The dmask switch picks
/// the Vdata width; an opcode/dmask pair without a format is an error
/// (Kyty EXITs at L3201).
#[allow(clippy::too_many_lines)]
fn shader_parse_mimg(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "mimg";
    let b0 = buffer[0];
    let b1 = dw(buffer, 1, pc)?;

    let slc = (b0 >> 25) & 0x1;
    let opcode = ((b0 >> 18) & 0x7f) | (u32::from(next_gen) * ((b0 & 0x1) << 7));
    let nsa_dwords = if next_gen { (b0 >> 1) & 0x3 } else { 0 };
    let lwe = (b0 >> 17) & 0x1;
    let tff = (b0 >> 16) & 0x1;
    let r128 = (b0 >> 15) & 0x1;
    let da = (b0 >> 14) & 0x1;
    let glc = (b0 >> 13) & 0x1;
    let unrm = (b0 >> 12) & 0x1;
    let dmask = (b0 >> 8) & 0xf;
    // GFX10/RDNA2 moves the image dimension into MIMG word 0 bits [5:3].
    // It is the source of truth for the number of address VGPRs consumed by
    // image_load/store; the T# resource type instead describes the view. In
    // particular, Minecraft uses DIM_2D with a type-13 array descriptor and
    // selects the destination face through T#.BASE_ARRAY. Treating every
    // operation as Vaddr3 reads the next, unrelated VGPR as an array layer.
    //
    // DIM 0/4 are reserved. Keep the old three-component shape for those
    // encodings so legacy GCN fixtures (which did not populate GFX10 DIM)
    // remain parseable; real GFX10 DIM values follow KytyPS5 ImageOps.cpp.
    let image_coord_components = if next_gen {
        match (b0 >> 3) & 0x7 {
            1 | 6 => 2,         // 2D / 2D MSAA
            2 | 3 | 5 | 7 => 3, // 3D / 2D array variants
            _ => 3,
        }
    } else {
        3
    };

    let ssamp = (b1 >> 21) & 0x1f; // S#
    let srsrc = (b1 >> 16) & 0x1f; // T#
    let vdata = (b1 >> 8) & 0xff;
    let vaddr = b1 & 0xff;

    // Kyty L2935-2941: EXIT_NOT_IMPLEMENTED checks.
    if da == 1 {
        return Err(feature(S, "da == 1", pc));
    }
    if r128 == 1 {
        return Err(feature(S, "r128 == 1", pc));
    }
    if tff == 1 {
        return Err(feature(S, "tff == 1", pc));
    }
    if lwe == 1 {
        return Err(feature(S, "lwe == 1", pc));
    }
    if glc == 1 {
        return Err(feature(S, "glc == 1", pc));
    }
    if slc == 1 {
        return Err(feature(S, "slc == 1", pc));
    }
    if unrm == 1 {
        return Err(feature(S, "unrm == 1", pc));
    }

    let size: u32 = 2 + nsa_dwords;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.dst = operand_parse(vdata + 256)?;
    inst.src_num = 3;
    inst.src[0] = operand_parse(vaddr + 256)?;
    inst.src[1] = operand_parse(srsrc * 4)?;
    inst.src[2] = operand_parse(ssamp * 4)?;
    inst.mimg_nsa_dwords = nsa_dwords as u8;
    for word_index in 0..nsa_dwords {
        let nsa = dw(buffer, 2 + word_index, pc)?;
        for byte_index in 0..4 {
            let component = (word_index * 4 + byte_index) as usize;
            let vgpr = (nsa >> (byte_index * 8)) & 0xff;
            inst.mimg_nsa_addr[component] = operand_parse(vgpr + 256)?;
        }
    }

    match opcode {
        0x00 => {
            inst.type_ = T::ImageLoad;
            inst.src[0].size = image_coord_components;
            inst.src[1].size = 8;
            inst.src_num = 2;
            match dmask {
                0x1 => {
                    inst.format = F::Vdata1Vaddr3StDmask1;
                    inst.dst.size = 1;
                }
                // Beyond Kyty: two-channel fetch (measured on ASTRO.BOT scene
                // compute, MIMG 0x00 dmask 0x3).
                0x3 => {
                    inst.format = F::Vdata2Vaddr3StDmask3;
                    inst.dst.size = 2;
                }
                0x7 => {
                    inst.format = F::Vdata3Vaddr3StDmask7;
                    inst.dst.size = 3;
                }
                0xf => {
                    inst.format = F::Vdata4Vaddr3StDmaskF;
                    inst.dst.size = 4;
                }
                _ => {}
            }
        }
        0x01 => return Err(ni(dst, S, "image_load_mip", opcode, pc, b0)),
        0x02 => return Err(ni(dst, S, "image_load_pck", opcode, pc, b0)),
        0x03 => return Err(ni(dst, S, "image_load_pck_sgn", opcode, pc, b0)),
        0x04 => return Err(ni(dst, S, "image_load_mip_pck", opcode, pc, b0)),
        0x05 => return Err(ni(dst, S, "image_load_mip_pck_sgn", opcode, pc, b0)),
        0x08 => {
            inst.type_ = T::ImageStore;
            inst.src[0].size = image_coord_components;
            inst.src[1].size = 8;
            inst.src_num = 2;
            match dmask {
                // Beyond Kyty: single-channel store (measured on ASTRO.BOT
                // scene compute, MIMG 0x08 dmask 0x1).
                0x1 => {
                    inst.format = F::Vdata1Vaddr3StDmask1;
                    inst.dst.size = 1;
                }
                // Beyond Kyty: two-channel store (measured on ASTRO.BOT
                // scene compute, MIMG 0x08 dmask 0x3).
                0x3 => {
                    inst.format = F::Vdata2Vaddr3StDmask3;
                    inst.dst.size = 2;
                }
                0xf => {
                    inst.format = F::Vdata4Vaddr3StDmaskF;
                    inst.dst.size = 4;
                }
                _ => {}
            }
        }
        0x09 => {
            inst.type_ = T::ImageStoreMip;
            inst.src[0].size = image_coord_components + 1;
            inst.src[1].size = 8;
            inst.src_num = 2;
            if dmask == 0xf {
                inst.format = F::Vdata4Vaddr4StDmaskF;
                inst.dst.size = 4;
            }
        }
        0x0a => return Err(ni(dst, S, "image_store_pck", opcode, pc, b0)),
        0x0b => return Err(ni(dst, S, "image_store_mip_pck", opcode, pc, b0)),
        0x0e => {
            inst.type_ = T::ImageGetResinfo;
            inst.src[0].size = 1; // mip level
            inst.src[1].size = 8; // T#
            inst.src_num = 2;
            if dmask == 0x3 {
                inst.format = F::Vdata2VaddrStDmask3;
                inst.dst.size = 2;
            }
        }
        0x0f => return Err(ni(dst, S, "image_atomic_swap", opcode, pc, b0)),
        0x10 => return Err(ni(dst, S, "image_atomic_cmpswap", opcode, pc, b0)),
        0x11 => return Err(ni(dst, S, "image_atomic_add", opcode, pc, b0)),
        0x12 => return Err(ni(dst, S, "image_atomic_sub", opcode, pc, b0)),
        0x13 => return Err(ni(dst, S, "image_atomic_rsub", opcode, pc, b0)),
        0x14 => return Err(ni(dst, S, "image_atomic_smin", opcode, pc, b0)),
        0x15 => return Err(ni(dst, S, "image_atomic_umin", opcode, pc, b0)),
        0x16 => return Err(ni(dst, S, "image_atomic_smax", opcode, pc, b0)),
        0x17 => return Err(ni(dst, S, "image_atomic_umax", opcode, pc, b0)),
        0x18 => return Err(ni(dst, S, "image_atomic_and", opcode, pc, b0)),
        0x19 => return Err(ni(dst, S, "image_atomic_or", opcode, pc, b0)),
        0x1a => return Err(ni(dst, S, "image_atomic_xor", opcode, pc, b0)),
        0x1b => return Err(ni(dst, S, "image_atomic_inc", opcode, pc, b0)),
        0x1c => return Err(ni(dst, S, "image_atomic_dec", opcode, pc, b0)),
        0x1d => return Err(ni(dst, S, "image_atomic_fcmpswap", opcode, pc, b0)),
        0x1e => return Err(ni(dst, S, "image_atomic_fmin", opcode, pc, b0)),
        0x1f => return Err(ni(dst, S, "image_atomic_fmax", opcode, pc, b0)),
        0x20 => {
            inst.type_ = T::ImageSample;
            inst.src[0].size = 3;
            inst.src[1].size = 8;
            inst.src[2].size = 4;
            match dmask {
                0x1 => {
                    inst.format = F::Vdata1Vaddr3StSsDmask1;
                    inst.dst.size = 1;
                }
                0x2 => {
                    inst.format = F::Vdata1Vaddr3StSsDmask2;
                    inst.dst.size = 1;
                }
                0x3 => {
                    inst.format = F::Vdata2Vaddr3StSsDmask3;
                    inst.dst.size = 2;
                }
                0x5 => {
                    inst.format = F::Vdata2Vaddr3StSsDmask5;
                    inst.dst.size = 2;
                }
                0x7 => {
                    inst.format = F::Vdata3Vaddr3StSsDmask7;
                    inst.dst.size = 3;
                }
                0x8 => {
                    inst.format = F::Vdata1Vaddr3StSsDmask8;
                    inst.dst.size = 1;
                }
                0x9 => {
                    inst.format = F::Vdata2Vaddr3StSsDmask9;
                    inst.dst.size = 2;
                }
                0xf => {
                    inst.format = F::Vdata4Vaddr3StSsDmaskF;
                    inst.dst.size = 4;
                }
                _ => {}
            }
        }
        0x21 => return Err(ni(dst, S, "image_sample_cl", opcode, pc, b0)),
        0x22 => return Err(ni(dst, S, "image_sample_d", opcode, pc, b0)),
        0x23 => return Err(ni(dst, S, "image_sample_d_cl", opcode, pc, b0)),
        0x24 => return Err(ni(dst, S, "image_sample_l", opcode, pc, b0)),
        0x25 => return Err(ni(dst, S, "image_sample_b", opcode, pc, b0)),
        0x26 => return Err(ni(dst, S, "image_sample_b_cl", opcode, pc, b0)),
        0x27 => {
            inst.type_ = T::ImageSampleLz;
            inst.src[0].size = 3;
            inst.src[1].size = 8;
            inst.src[2].size = 4;
            match dmask {
                // Beyond Kyty: single-channel LOD-zero samples (measured on
                // ASTRO.BOT scene compute, MIMG 0x27 dmask 0x1 and 0x2).
                0x1 => {
                    inst.format = F::Vdata1Vaddr3StSsDmask1;
                    inst.dst.size = 1;
                }
                0x2 => {
                    inst.format = F::Vdata1Vaddr3StSsDmask2;
                    inst.dst.size = 1;
                }
                // Beyond Kyty: two-channel LOD-zero sample (measured on
                // ASTRO.BOT scene compute, MIMG 0x27 dmask 0x3 — 58
                // dispatches/run).
                0x3 => {
                    inst.format = F::Vdata2Vaddr3StSsDmask3;
                    inst.dst.size = 2;
                }
                0x7 => {
                    inst.format = F::Vdata3Vaddr3StSsDmask7;
                    inst.dst.size = 3;
                }
                0xf => {
                    inst.format = F::Vdata4Vaddr3StSsDmaskF;
                    inst.dst.size = 4;
                }
                _ => {}
            }
        }
        0x28 => return Err(ni(dst, S, "image_sample_c", opcode, pc, b0)),
        0x29 => return Err(ni(dst, S, "image_sample_c_cl", opcode, pc, b0)),
        0x2a => return Err(ni(dst, S, "image_sample_c_d", opcode, pc, b0)),
        0x2b => return Err(ni(dst, S, "image_sample_c_d_cl", opcode, pc, b0)),
        0x2c => return Err(ni(dst, S, "image_sample_c_l", opcode, pc, b0)),
        0x2d => return Err(ni(dst, S, "image_sample_c_b", opcode, pc, b0)),
        0x2e => return Err(ni(dst, S, "image_sample_c_b_cl", opcode, pc, b0)),
        0x2f => {
            inst.type_ = T::ImageSampleCLz;
            // Gen5 comparison samples place the full-width depth reference
            // before the ordinary 2D coordinates: {reference, x, y}.
            inst.src[0].size = 3;
            inst.src[1].size = 8;
            inst.src[2].size = 4;
            match dmask {
                0x1 => {
                    inst.format = F::Vdata1Vaddr3StSsDmask1;
                    inst.dst.size = 1;
                }
                0x3 => {
                    inst.format = F::Vdata2Vaddr3StSsDmask3;
                    inst.dst.size = 2;
                }
                0x5 => {
                    inst.format = F::Vdata2Vaddr3StSsDmask5;
                    inst.dst.size = 2;
                }
                0x7 => {
                    inst.format = F::Vdata3Vaddr3StSsDmask7;
                    inst.dst.size = 3;
                }
                0x8 => {
                    inst.format = F::Vdata1Vaddr3StSsDmask8;
                    inst.dst.size = 1;
                }
                0x9 => {
                    inst.format = F::Vdata2Vaddr3StSsDmask9;
                    inst.dst.size = 2;
                }
                0xf => {
                    inst.format = F::Vdata4Vaddr3StSsDmaskF;
                    inst.dst.size = 4;
                }
                _ => {}
            }
        }
        0x30 => return Err(ni(dst, S, "image_sample_o", opcode, pc, b0)),
        0x31 => return Err(ni(dst, S, "image_sample_cl_o", opcode, pc, b0)),
        0x32 => return Err(ni(dst, S, "image_sample_d_o", opcode, pc, b0)),
        0x33 => return Err(ni(dst, S, "image_sample_d_cl_o", opcode, pc, b0)),
        0x34 => return Err(ni(dst, S, "image_sample_l_o", opcode, pc, b0)),
        0x35 => return Err(ni(dst, S, "image_sample_b_o", opcode, pc, b0)),
        0x36 => return Err(ni(dst, S, "image_sample_b_cl_o", opcode, pc, b0)),
        0x37 => {
            inst.type_ = T::ImageSampleLzO;
            inst.src[0].size = 4;
            inst.src[1].size = 8;
            inst.src[2].size = 4;
            match dmask {
                0x1 => {
                    inst.format = F::Vdata1Vaddr4StSsDmask1;
                    inst.dst.size = 1;
                }
                0x2 => {
                    inst.format = F::Vdata1Vaddr4StSsDmask2;
                    inst.dst.size = 1;
                }
                0x7 => {
                    inst.format = F::Vdata3Vaddr4StSsDmask7;
                    inst.dst.size = 3;
                }
                _ => {}
            }
        }
        0x38 => return Err(ni(dst, S, "image_sample_c_o", opcode, pc, b0)),
        0x39 => return Err(ni(dst, S, "image_sample_c_cl_o", opcode, pc, b0)),
        0x3a => return Err(ni(dst, S, "image_sample_c_d_o", opcode, pc, b0)),
        0x3b => return Err(ni(dst, S, "image_sample_c_d_cl_o", opcode, pc, b0)),
        0x3c => return Err(ni(dst, S, "image_sample_c_l_o", opcode, pc, b0)),
        0x3d => return Err(ni(dst, S, "image_sample_c_b_o", opcode, pc, b0)),
        0x3e => return Err(ni(dst, S, "image_sample_c_b_cl_o", opcode, pc, b0)),
        0x3f => return Err(ni(dst, S, "image_sample_c_lz_o", opcode, pc, b0)),
        0x40 => return Err(ni(dst, S, "image_gather4", opcode, pc, b0)),
        0x41 => return Err(ni(dst, S, "image_gather4_cl", opcode, pc, b0)),
        0x44 => return Err(ni(dst, S, "image_gather4_l", opcode, pc, b0)),
        0x45 => return Err(ni(dst, S, "image_gather4_b", opcode, pc, b0)),
        0x46 => return Err(ni(dst, S, "image_gather4_b_cl", opcode, pc, b0)),
        // Beyond Kyty (KYTY_NI upstream): four-texel single-channel gather at
        // an implicit zero LOD — measured on ASTRO.BOT scene compute
        // (raw 0xf11c0108, dmask 0x1). The gather dmask names the ONE channel
        // gathered (must be a single bit); vdata is always 4 dwords, one per
        // texel. Only the measured dmask is wired; others stay named with the
        // dmask evidence via the unset-format failure below.
        0x47 => {
            inst.type_ = T::ImageGather4Lz;
            inst.src[0].size = 3;
            inst.src[1].size = 8;
            inst.src[2].size = 4;
            if dmask == 0x1 {
                inst.format = F::Vdata4Vaddr3StSsDmask1;
                inst.dst.size = 4;
            }
        }
        0x48 => return Err(ni(dst, S, "image_gather4_c", opcode, pc, b0)),
        0x49 => return Err(ni(dst, S, "image_gather4_c_cl", opcode, pc, b0)),
        0x4c => return Err(ni(dst, S, "image_gather4_c_l", opcode, pc, b0)),
        0x4d => return Err(ni(dst, S, "image_gather4_c_b", opcode, pc, b0)),
        0x4e => return Err(ni(dst, S, "image_gather4_c_b_cl", opcode, pc, b0)),
        0x4f => return Err(ni(dst, S, "image_gather4_c_lz", opcode, pc, b0)),
        0x50 => return Err(ni(dst, S, "image_gather4_o", opcode, pc, b0)),
        0x51 => return Err(ni(dst, S, "image_gather4_cl_o", opcode, pc, b0)),
        0x54 => return Err(ni(dst, S, "image_gather4_l_o", opcode, pc, b0)),
        0x55 => return Err(ni(dst, S, "image_gather4_b_o", opcode, pc, b0)),
        0x56 => return Err(ni(dst, S, "image_gather4_b_cl_o", opcode, pc, b0)),
        0x57 => return Err(ni(dst, S, "image_gather4_lz_o", opcode, pc, b0)),
        0x58 => return Err(ni(dst, S, "image_gather4_c_o", opcode, pc, b0)),
        0x59 => return Err(ni(dst, S, "image_gather4_c_cl_o", opcode, pc, b0)),
        0x5c => return Err(ni(dst, S, "image_gather4_c_l_o", opcode, pc, b0)),
        0x5d => return Err(ni(dst, S, "image_gather4_c_b_o", opcode, pc, b0)),
        0x5e => return Err(ni(dst, S, "image_gather4_c_b_cl_o", opcode, pc, b0)),
        0x5f => return Err(ni(dst, S, "image_gather4_c_lz_o", opcode, pc, b0)),
        0x60 => return Err(ni(dst, S, "image_get_lod", opcode, pc, b0)),
        0x68 => return Err(ni(dst, S, "image_sample_cd", opcode, pc, b0)),
        0x69 => return Err(ni(dst, S, "image_sample_cd_cl", opcode, pc, b0)),
        0x6a => return Err(ni(dst, S, "image_sample_c_cd", opcode, pc, b0)),
        0x6b => return Err(ni(dst, S, "image_sample_c_cd_cl", opcode, pc, b0)),
        0x6c => return Err(ni(dst, S, "image_sample_cd_o", opcode, pc, b0)),
        0x6d => return Err(ni(dst, S, "image_sample_cd_cl_o", opcode, pc, b0)),
        0x6e => return Err(ni(dst, S, "image_sample_c_cd_o", opcode, pc, b0)),
        0x6f => return Err(ni(dst, S, "image_sample_c_cd_cl_o", opcode, pc, b0)),
        0x7e => return Err(ni(dst, S, "image_rsrc256", opcode, pc, b0)),
        0x7f => return Err(ni(dst, S, "image_sampler", opcode, pc, b0)),
        // Kyty's table continues with 0xA0-0xDE `_a` variants
        // (ShaderParse.cpp L3162-3193); they are unreachable with the
        // (buffer[0] >> 18) & 0x7f decode Kyty itself uses, so the port folds
        // them into UnknownOpcode.
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    if inst.format == F::Unknown {
        // Kyty L3198-3202: dump + EXIT on missing dmask format.
        tracing::error!(
            "unknown mimg format for opcode: 0x{opcode:02x} at addr 0x{pc:08x}, dmask: 0x{dmask:x}\n{}",
            dst.dbg_dump()
        );
        return Err(ShaderParseError::UnknownMimgFormat { opcode, dmask, pc });
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_mtbuf` (L3210).
fn shader_parse_mtbuf(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "mtbuf";
    let b0 = buffer[0];
    let b1 = dw(buffer, 1, pc)?;

    let opcode = (b0 >> 16) & 0x7;
    let dfmt = (b0 >> 19) & 0xf;
    let nfmt = (b0 >> 23) & 0x7;
    let glc = (b0 >> 14) & 0x1;
    let idxen = (b0 >> 13) & 0x1;
    let offen = (b0 >> 12) & 0x1;
    let offset = b0 & 0xfff;

    let soffset = (b1 >> 24) & 0xff;
    let tfe = (b1 >> 23) & 0x1;
    let slc = (b1 >> 22) & 0x1;
    let srsrc = (b1 >> 16) & 0x1f;
    let vdata = (b1 >> 8) & 0xff;
    let vaddr = b1 & 0xff;

    // Kyty L3233-3238: EXIT_NOT_IMPLEMENTED checks (offen is allowed here).
    if idxen == 0 {
        return Err(feature(S, "idxen == 0", pc));
    }
    if offset != 0 {
        return Err(feature(S, "offset != 0", pc));
    }
    if glc == 1 {
        return Err(feature(S, "glc == 1", pc));
    }
    if slc == 1 {
        return Err(feature(S, "slc == 1", pc));
    }
    if tfe == 1 {
        return Err(feature(S, "tfe == 1", pc));
    }

    if (dfmt != 14 && dfmt != 4) || nfmt != 7 {
        // Kyty L3242-3246: dump-free EXIT on unsupported buffer format.
        tracing::error!(
            "unknown format: dfmt = {dfmt}, nfmt = {nfmt} at addr 0x{pc:08x} \
             (hash0 = 0x{:08x}, crc32 = 0x{:08x})",
            dst.get_hash0(),
            dst.get_crc32()
        );
        return Err(ShaderParseError::UnknownMtbufFormat { dfmt, nfmt, pc });
    }

    let mut size: u32 = 2;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.dst = operand_parse(vdata + 256)?;
    inst.src_num = 3;
    inst.src[0] = operand_parse(vaddr + 256)?;
    inst.src[1] = operand_parse(srsrc * 4)?;
    inst.src[2] = operand_parse(soffset)?;

    if inst.src[2].type_ == O::LiteralConstant {
        inst.src[2].constant.u = dw(buffer, size, pc)?;
        size += 1;
    }

    inst.src[1].size = 4;

    match opcode {
        0x00 => {
            inst.type_ = T::TBufferLoadFormatX;
            inst.format = F::Vdata1VaddrSvSoffsIdxenFloat1;
            if offen == 1 {
                return Err(feature(S, "tbuffer_load_format_x with offen", pc));
            }
            if !(dfmt == 4 && nfmt == 7) {
                return Err(feature(S, "tbuffer_load_format_x needs dfmt=4 nfmt=7", pc));
            }
        }
        0x01 => return Err(ni(dst, S, "tbuffer_load_format_xy", opcode, pc, b0)),
        0x02 => return Err(ni(dst, S, "tbuffer_load_format_xyz", opcode, pc, b0)),
        0x03 => {
            inst.type_ = T::TBufferLoadFormatXyzw;
            inst.format = if offen == 1 {
                F::Vdata4Vaddr2SvSoffsOffenIdxenFloat4
            } else {
                F::Vdata4VaddrSvSoffsIdxenFloat4
            };
            inst.src[0].size += offen as i32;
            inst.dst.size = 4;
            if !(dfmt == 14 && nfmt == 7) {
                return Err(feature(
                    S,
                    "tbuffer_load_format_xyzw needs dfmt=14 nfmt=7",
                    pc,
                ));
            }
        }
        0x04 => return Err(ni(dst, S, "tbuffer_store_format_x", opcode, pc, b0)),
        0x05 => return Err(ni(dst, S, "tbuffer_store_format_xy", opcode, pc, b0)),
        0x06 => return Err(ni(dst, S, "tbuffer_store_format_xyz", opcode, pc, b0)),
        0x07 => return Err(ni(dst, S, "tbuffer_store_format_xyzw", opcode, pc, b0)),
        // Kyty also lists d16 variants 0x08-0x0f (ShaderParse.cpp
        // L3288-3295); unreachable with the 3-bit opcode mask Kyty uses, so
        // the port folds them into UnknownOpcode.
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(size)
}

/// Kyty: ShaderParse.cpp `shader_parse_vintrp` (L3304).
fn shader_parse_vintrp(
    pc: u32,
    buffer: &[u32],
    dst: &mut ShaderCode,
    _next_gen: bool,
) -> Result<u32, ShaderParseError> {
    const S: &str = "vintrp";
    let b0 = buffer[0];

    let opcode = (b0 >> 16) & 0x3;
    let vdst = (b0 >> 18) & 0xff;
    let attr = (b0 >> 10) & 0x3f;
    let chan = (b0 >> 8) & 0x3;
    let vsrc = b0 & 0xff;

    let mut inst = ShaderInstruction {
        pc,
        ..Default::default()
    };
    inst.src[0] = operand_parse(vsrc + 256)?;
    inst.dst = operand_parse(vdst + 256)?;
    inst.src[1].type_ = O::IntegerInlineConstant;
    inst.src[1].constant.u = attr;
    inst.src[2].type_ = O::IntegerInlineConstant;
    inst.src[2].constant.u = chan;
    inst.src_num = 3;

    inst.format = F::VdstVsrcAttrChan;

    match opcode {
        0x00 => inst.type_ = T::VInterpP1F32,
        0x01 => inst.type_ = T::VInterpP2F32,
        0x02 => {
            inst.type_ = T::VInterpMovF32;
            inst.src[0].type_ = O::IntegerInlineConstant;
            inst.src[0].constant.u = vsrc & 0x3;
            inst.src[0].size = 0;
        }
        _ => return Err(unknown_op(dst, S, opcode, pc, b0)),
    }

    dst.get_instructions_mut().push(inst);

    Ok(1)
}

/// Kyty: ShaderParse.cpp `shader_parse` (L3348) — the top-level dword walker.
///
/// Encoding classification (in Kyty's order): bit31 clear -> VOP2;
/// `& 0xF8000000 == 0xC0000000` -> SMRD (legacy only); `& 0xC0000000 ==
/// 0x80000000` -> SOP2 (which nests SOP1/SOPC/SOPP/SOPK); otherwise dispatch
/// on `instr >> 26`: 0x32 VINTRP, 0x34 VOP3 (legacy), 0x35 VOP3 (next-gen),
/// 0x36 DS, 0x38 MUBUF, 0x3a MTBUF, 0x3c MIMG, 0x3d SMEM (next-gen),
/// 0x3e EXP.
///
/// End detection (Kyty L3406): `0xBF810000` (s_endpgm) stops
/// Vertex/Pixel/Compute parsing unless the following pc is a live label
/// target; `0xBE802000` (s_setpc_b64 s[0:1]) stops Fetch shaders.
///
/// Returns the total number of dwords consumed from the start of `src`
/// (Kyty returns `ptr - src`). Deviation: the walk is bounded by
/// `src.len()` — running off the end yields [`ShaderParseError::Truncated`]
/// instead of reading unmapped memory.
pub fn shader_parse(
    pc: u32,
    src: &[u32],
    dst: &mut ShaderCode,
    next_gen: bool,
) -> Result<usize, ShaderParseError> {
    let type_ = dst.get_type();

    dst.get_instructions_mut().clear();
    dst.get_labels_mut().clear();
    dst.get_indirect_labels_mut().clear();

    let mut index = (pc / 4) as usize;
    loop {
        let Some(&instruction) = src.get(index) else {
            return Err(ShaderParseError::Truncated {
                pc: (index as u32) * 4,
            });
        };
        let pc = 4 * index as u32;
        let buffer = &src[index..];

        let advance = if (instruction & 0x8000_0000) == 0x0000_0000 {
            shader_parse_vop2(pc, buffer, dst, next_gen)?
        } else if (instruction & 0xF800_0000) == 0xC000_0000 {
            // Kyty L3371: EXIT_NOT_IMPLEMENTED(next_gen).
            if next_gen {
                return Err(feature("smrd", "legacy SMRD encoding on next_gen", pc));
            }
            shader_parse_smrd(pc, buffer, dst, next_gen)?
        } else if (instruction & 0xC000_0000) == 0x8000_0000 {
            shader_parse_sop2(pc, buffer, dst, next_gen)?
        } else {
            match instruction >> 26 {
                0x32 => shader_parse_vintrp(pc, buffer, dst, next_gen)?,
                0x34 => {
                    // Kyty L3382: EXIT_NOT_IMPLEMENTED(next_gen).
                    if next_gen {
                        return Err(feature("vop3", "legacy VOP3 encoding on next_gen", pc));
                    }
                    shader_parse_vop3(pc, buffer, dst, next_gen)?
                }
                0x35 => {
                    // Kyty L3386: EXIT_NOT_IMPLEMENTED(!next_gen).
                    if !next_gen {
                        return Err(feature("vop3", "next-gen VOP3 encoding on legacy", pc));
                    }
                    shader_parse_vop3(pc, buffer, dst, next_gen)?
                }
                0x36 => shader_parse_ds(pc, buffer, dst, next_gen)?,
                // Beyond Kyty (SharpEmu PR #587): FLAT-class (FLAT/GLOBAL)
                // direct guest-memory access, encoding 0x37.
                0x37 => shader_parse_flat(pc, buffer, dst, next_gen)?,
                0x38 => shader_parse_mubuf(pc, buffer, dst, next_gen)?,
                0x3a => shader_parse_mtbuf(pc, buffer, dst, next_gen)?,
                0x3c => shader_parse_mimg(pc, buffer, dst, next_gen)?,
                0x3d => {
                    // Kyty L3394: EXIT_NOT_IMPLEMENTED(!next_gen).
                    if !next_gen {
                        return Err(feature("smem", "next-gen SMEM encoding on legacy", pc));
                    }
                    shader_parse_smem(pc, buffer, dst, next_gen)?
                }
                0x3e => shader_parse_exp(pc, buffer, dst, next_gen)?,
                _ => {
                    // Kyty L3398-3402: dump + EXIT on unknown encoding.
                    tracing::error!(
                        "unknown code 0x{instruction:08x} at addr 0x{pc:08x}\n{}",
                        dst.dbg_dump()
                    );
                    return Err(ShaderParseError::UnknownEncoding {
                        pc,
                        raw: instruction,
                    });
                }
            }
        };

        index += advance as usize;

        // Kyty L3406-3411: end detection (with the live-label exception).
        let next_pc = 4 * index as u32;
        if (instruction == 0xBF81_0000
            && matches!(
                type_,
                ShaderType::Vertex | ShaderType::Pixel | ShaderType::Compute
            )
            && !dst
                .get_labels()
                .iter()
                .any(|label| label.get_dst() == next_pc))
            || (instruction == 0xBE80_2000 && type_ == ShaderType::Fetch)
            // RDNA2 `s_code_end` ends the code BLOCK, so unlike s_endpgm it takes
            // no live-label exception — nothing can branch past it. Measured on
            // ASTRO.BOT: shaders whose branch targets sit beyond the first
            // s_endpgm keep parsing (correctly) until they reach this, and
            // without the break they run on into padding and fail
            // ("unknown operand: 115"), which killed any analysis built on a
            // full-buffer parse.
            || instruction == 0xBF9F_0000
        {
            break;
        }
    }

    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn parse(
        src: &[u32],
        type_: ShaderType,
        next_gen: bool,
    ) -> (ShaderCode, Result<usize, ShaderParseError>) {
        let mut code = ShaderCode::new();
        code.set_type(type_);
        let result = shader_parse(0, src, &mut code, next_gen);
        (code, result)
    }

    fn parse_vs(src: &[u32]) -> (ShaderCode, usize) {
        let (code, result) = parse(src, ShaderType::Vertex, false);
        let consumed = result.expect("parse failed");
        (code, consumed)
    }

    #[test]
    fn vop1_sdwa_decodes_abs_modifier() {
        // Measured in Minecraft's UI PS (ps_253f0800): `v_rcp_f32 v1, |v5|` —
        // VOP1 with the SDWA marker (src0 == 249) whose second dword carries
        // the real src0 (v5) plus the abs modifier. VOP1 previously lacked the
        // SDWA handling that VOP2/VOPC have, so `operand_parse` was handed the
        // 249 marker and failed the whole shader with "unknown operand: 249".
        let (code, _) = parse_vs(&[0x7e02_54f9, 0x0026_0605, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VRcpF32);
        assert_eq!(inst.format, F::SVdstSVsrc0);
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 5));
        assert!(inst.src[0].absolute, "SDWA abs modifier applied to src0");
        assert!(!inst.src[0].negate);
        assert_eq!(inst.dst.register_id, 1);
    }

    /// Measured ASTRO.BOT tiled-lighting encoding `0x7c32d4f9`: VOPC op 0x19
    /// (`v_cmpx_nge_f32`) with `src0 == 0xf9` (SDWA), so the instruction is TWO
    /// dwords (b0 + the SDWA control dword). A one-dword mis-decode of the
    /// newly-wired 0x19 arm would shift every later PC and make a valid branch
    /// target look "not on an instruction boundary" (`Spirv::FindReloopBlocks`).
    /// The SDWA length comes from the shared `src0 == 249` path, independent of
    /// the opcode arm — this guards that it stays so.
    #[test]
    fn v_cmpx_nge_f32_sdwa_is_two_dwords() {
        let (code, result) = parse(
            &[0x7c32_d4f9, 0x0686_8081, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse v_cmpx_nge_f32 SDWA");
        let insts = code.get_instructions();
        assert_eq!(insts[0].type_, T::VCmpxNgeF32);
        assert_eq!(
            insts.len(),
            2,
            "SDWA v_cmpx_nge + s_endpgm = 2 instructions (not 3)"
        );
        assert_eq!(insts[1].type_, T::SEndpgm);
        assert_eq!(insts[1].pc, 8, "v_cmpx_nge SDWA spans 8 bytes (2 dwords)");
    }

    /// `ds_wrxchg_rtn_b32` is a DS (LDS) instruction, and every DS encoding is a
    /// fixed 2 dwords — the newly-wired 0x2d arm must not perturb that. A wrong
    /// length here would likewise desync `FindReloopBlocks` boundaries.
    #[test]
    fn ds_wrxchg_rtn_b32_is_two_dwords() {
        let (code, result) = parse(
            &[0xD8B4_0510, 0x0200_0100, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse ds_wrxchg_rtn_b32");
        let insts = code.get_instructions();
        assert_eq!(insts[0].type_, T::DsWrxchgRtnB32);
        assert_eq!(insts.len(), 2, "ds_wrxchg + s_endpgm = 2 instructions");
        assert_eq!(insts[1].pc, 8, "ds_wrxchg spans 8 bytes (2 dwords)");
    }

    /// VOP2 DPP16 (`src0 == 0xfa`): a second dword carries the real src0 (a
    /// VGPR, in its low byte) plus a 9-bit `dpp_ctrl` cross-lane pattern and
    /// row/bank masks. Before DPP decode, `operand_parse` was handed the 0xfa
    /// marker and failed the whole shader ("unknown operand: 250"). Mirrors the
    /// VOP2 SDWA block; shape studied from shadPS4 `decodeDataParallelPrimitive`
    /// (GPL-2.0, not copied). Encoding: `v_add_f32 v2, v5 row_shr:1, v3` — the
    /// DPP dword is `row_shr` (`dpp_ctrl == 0x111`), row/bank mask 0xf,
    /// bound_ctrl set.
    #[test]
    fn vop2_dpp16_decodes_and_is_two_dwords() {
        let (code, result) = parse(
            &[0x0604_06FA, 0xFF09_1105, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse v_add_f32 DPP16");
        let insts = code.get_instructions();
        assert_eq!(insts[0].type_, T::VAddF32);
        assert_eq!(
            insts.len(),
            2,
            "DPP add + s_endpgm = 2 instructions (not 3)"
        );
        assert_eq!(insts[1].pc, 8, "DPP16 add spans 8 bytes (2 dwords)");
        // Real src0 comes from the low byte of the DPP dword, always a VGPR.
        assert_eq!(
            (insts[0].src[0].type_, insts[0].src[0].register_id),
            (O::Vgpr, 5)
        );
        assert_eq!(
            (insts[0].src[1].type_, insts[0].src[1].register_id),
            (O::Vgpr, 3)
        );
        assert_eq!((insts[0].dst.type_, insts[0].dst.register_id), (O::Vgpr, 2));
        assert_eq!(
            insts[0].src[0].dpp,
            Some(DppCtrl {
                mode: DppMode::Dpp16 { ctrl: 0x111 },
                row_mask: 0xf,
                bank_mask: 0xf,
                bound_ctrl: true,
                fetch_inactive: false,
            }),
            "DPP16 control decoded onto src0"
        );
    }

    /// VOP1 DPP16: single-source form (no src1 modifiers). `v_mov_b32 v1,
    /// v7 quad_perm:[3,2,1,0]` — `dpp_ctrl == 0x1b`, the quad-permute op.
    #[test]
    fn vop1_dpp16_decodes() {
        let (code, result) = parse(
            &[0x7E02_02FA, 0xFF00_1B07, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse v_mov_b32 DPP16");
        let insts = code.get_instructions();
        assert_eq!(insts[0].type_, T::VMovB32);
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[1].pc, 8, "DPP16 mov spans 8 bytes");
        assert_eq!(
            (insts[0].src[0].type_, insts[0].src[0].register_id),
            (O::Vgpr, 7)
        );
        assert_eq!((insts[0].dst.type_, insts[0].dst.register_id), (O::Vgpr, 1));
        assert_eq!(
            insts[0].src[0].dpp,
            Some(DppCtrl {
                mode: DppMode::Dpp16 { ctrl: 0x1b },
                row_mask: 0xf,
                bank_mask: 0xf,
                bound_ctrl: false,
                fetch_inactive: false,
            })
        );
    }

    /// VOPC DPP16: `v_cmp_lt_f32 vcc, v6 row_mirror, v4` — `dpp_ctrl == 0x140`.
    /// The 2-dword length must come from the shared DPP marker, independent of
    /// the opcode arm, so a valid later branch target stays on an instruction
    /// boundary (the `FindReloopBlocks` invariant the SDWA guards also protect).
    #[test]
    fn vopc_dpp16_decodes() {
        let (code, result) = parse(
            &[0x7C02_08FA, 0xFF01_4006, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse v_cmp_lt_f32 DPP16");
        let insts = code.get_instructions();
        assert_eq!(insts[0].type_, T::VCmpLtF32);
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[1].pc, 8, "DPP16 vopc spans 8 bytes");
        assert_eq!(
            (insts[0].src[0].type_, insts[0].src[0].register_id),
            (O::Vgpr, 6)
        );
        assert_eq!(
            (insts[0].src[1].type_, insts[0].src[1].register_id),
            (O::Vgpr, 4)
        );
        assert_eq!(insts[0].dst.type_, O::VccLo, "sd == 0 targets VCC");
        assert_eq!(
            insts[0].src[0].dpp,
            Some(DppCtrl {
                mode: DppMode::Dpp16 { ctrl: 0x140 },
                row_mask: 0xf,
                bank_mask: 0xf,
                bound_ctrl: false,
                fetch_inactive: false,
            })
        );
    }

    /// VOP2 DPP16 abs/neg: the DPP control dword carries its own src0/src1
    /// abs/neg bits (distinct bit positions from SDWA). `v_add_f32 v2,
    /// -|v5| quad_perm, |v3|` — src0_neg+src0_abs and src1_abs set.
    #[test]
    fn vop2_dpp16_applies_src_modifiers() {
        // dpp_ctrl 0 (quad_perm identity base), src0=v5, src0_neg[20]=1,
        // src0_abs[21]=1, src1_abs[23]=1, masks 0xf.
        let b1 = 0xFF00_0000u32 | (1 << 20) | (1 << 21) | (1 << 23) | 0x05;
        let (code, result) = parse(&[0x0604_06FA, b1, S_ENDPGM], ShaderType::Compute, true);
        result.expect("parse v_add_f32 DPP16 with modifiers");
        let inst = &code.get_instructions()[0];
        assert!(inst.src[0].negate, "DPP src0_neg");
        assert!(inst.src[0].absolute, "DPP src0_abs");
        assert!(!inst.src[1].negate);
        assert!(inst.src[1].absolute, "DPP src1_abs");
    }

    /// VOP2 DPP8 (`src0 == 0xe9`): eight 3-bit lane selects fill bits [31:8];
    /// no abs/neg/masks. `v_add_f32 v2, v5 dpp8:[0,1,2,3,4,5,6,7], v3`.
    #[test]
    fn vop2_dpp8_decodes() {
        let (code, result) = parse(
            &[0x0604_06E9, 0xFAC6_8805, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse v_add_f32 DPP8");
        let insts = code.get_instructions();
        assert_eq!(insts[0].type_, T::VAddF32);
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[1].pc, 8, "DPP8 add spans 8 bytes (2 dwords)");
        assert_eq!(
            (insts[0].src[0].type_, insts[0].src[0].register_id),
            (O::Vgpr, 5)
        );
        assert_eq!(
            insts[0].src[0].dpp,
            Some(DppCtrl {
                mode: DppMode::Dpp8 {
                    lane_sel: [0, 1, 2, 3, 4, 5, 6, 7],
                },
                row_mask: 0,
                bank_mask: 0,
                bound_ctrl: false,
                fetch_inactive: false,
            })
        );
    }

    /// DPP8FI (`src0 == 0xea`) is the fetch-inactive DPP8 variant — same
    /// lane-select layout, `fetch_inactive` flagged.
    #[test]
    fn vop2_dpp8_fi_flags_fetch_inactive() {
        let (code, result) = parse(
            &[0x0604_06EA, 0xFAC6_8805, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse v_add_f32 DPP8FI");
        let inst = &code.get_instructions()[0];
        match inst.src[0].dpp {
            Some(DppCtrl {
                fetch_inactive,
                mode: DppMode::Dpp8 { .. },
                ..
            }) => assert!(fetch_inactive, "0xea is DPP8 fetch-inactive"),
            other => panic!("expected DPP8 ctrl, got {other:?}"),
        }
    }

    // ---- 1. operand_parse table (Kyty: ShaderParse.cpp L32) ----

    #[test]
    fn operand_parse_sgpr_bounds() {
        let op = operand_parse(0).unwrap();
        assert_eq!((op.type_, op.register_id, op.size), (O::Sgpr, 0, 1));
        let op = operand_parse(103).unwrap();
        assert_eq!((op.type_, op.register_id, op.size), (O::Sgpr, 103, 1));
        assert_eq!(
            operand_parse(104),
            Err(ShaderParseError::UnknownOperand { code: 104 })
        );
        assert_eq!(
            operand_parse(105),
            Err(ShaderParseError::UnknownOperand { code: 105 })
        );
    }

    #[test]
    fn operand_parse_special_registers() {
        assert_eq!(operand_parse(106).unwrap().type_, O::VccLo);
        assert_eq!(operand_parse(107).unwrap().type_, O::VccHi);
        assert_eq!(operand_parse(124).unwrap().type_, O::M0);
        assert_eq!(operand_parse(125).unwrap().type_, O::Null);
        assert_eq!(operand_parse(126).unwrap().type_, O::ExecLo);
        assert_eq!(operand_parse(127).unwrap().type_, O::ExecHi);
        assert_eq!(operand_parse(252).unwrap().type_, O::ExecZ);
        assert_eq!(operand_parse(106).unwrap().size, 1);
    }

    #[test]
    fn operand_parse_inline_integers() {
        let op = operand_parse(128).unwrap();
        assert_eq!(
            (op.type_, op.constant.i(), op.size),
            (O::IntegerInlineConstant, 0, 0)
        );
        assert_eq!(operand_parse(129).unwrap().constant.i(), 1);
        assert_eq!(operand_parse(192).unwrap().constant.i(), 64);
        assert_eq!(operand_parse(193).unwrap().constant.i(), -1);
        assert_eq!(operand_parse(208).unwrap().constant.i(), -16);
    }

    #[test]
    fn operand_parse_inline_floats() {
        // 248 = 1/(2*pi), RDNA2's ninth inline float (SharpEmu
        // Gen5InlineConstants).
        let expected = [0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 0.159_154_94];
        for (i, want) in expected.iter().enumerate() {
            let op = operand_parse(240 + i as u32).unwrap();
            assert_eq!(op.type_, O::FloatInlineConstant, "code {}", 240 + i);
            assert_eq!(op.constant.f(), *want, "code {}", 240 + i);
            assert_eq!(op.size, 0);
        }
    }

    #[test]
    fn operand_parse_literal_and_vgpr() {
        let op = operand_parse(255).unwrap();
        assert_eq!((op.type_, op.size), (O::LiteralConstant, 0));
        let op = operand_parse(256).unwrap();
        assert_eq!((op.type_, op.register_id, op.size), (O::Vgpr, 0, 1));
        let op = operand_parse(511).unwrap();
        assert_eq!((op.type_, op.register_id), (O::Vgpr, 255));
        assert!(operand_parse(209).is_err());
        assert!(operand_parse(249).is_err());
        assert!(operand_parse(254).is_err());
    }

    // ---- 2./3. per-family decodes (dwords assembled from Kyty bit layout) ----

    #[test]
    fn sop1_s_mov_b32() {
        // SOP2 opcode7=0x7d -> SOP1; sop1 opcode 0x03, sdst=s0, ssrc0=s1.
        let (code, consumed) = parse_vs(&[0xBE80_0301, S_ENDPGM]);
        assert_eq!(consumed, 2);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SMovB32);
        assert_eq!(inst.format, F::SVdstSVsrc0);
        assert_eq!(inst.src_num, 1);
        assert_eq!((inst.dst.type_, inst.dst.register_id), (O::Sgpr, 0));
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Sgpr, 1));
    }

    #[test]
    fn sop1_s_mov_b32_literal() {
        // ssrc0=255 -> literal in next dword.
        let (code, consumed) = parse_vs(&[0xBE85_03FF, 0xDEAD_BEEF, S_ENDPGM]);
        assert_eq!(consumed, 3);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.src[0].type_, O::LiteralConstant);
        assert_eq!(inst.src[0].constant.u, 0xDEAD_BEEF);
        assert_eq!(inst.src[0].size, 0);
        assert_eq!(inst.dst.register_id, 5);
        // The literal consumed a dword: next instruction pc is 8.
        assert_eq!(code.get_instructions()[1].pc, 8);
    }

    #[test]
    fn sop2_s_add_u32() {
        let (code, _) = parse_vs(&[0x8002_0100, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SAddU32);
        assert_eq!(inst.format, F::SVdstSVsrc0SVsrc1);
        assert_eq!(inst.dst.register_id, 2);
        assert_eq!(inst.src[0].register_id, 0);
        assert_eq!(inst.src[1].register_id, 1);
    }

    #[test]
    fn sop2_s_sub_u32_decodes_the_measured_next_gen_encoding() {
        // Minecraft PPSA17221: s_sub_u32 vcc_lo, 64, vcc_hi.
        let (code, result) = parse(&[0x80EA_6BC0, S_ENDPGM], ShaderType::Vertex, true);
        assert_eq!(result.unwrap(), 2);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SSubU32);
        assert_eq!(inst.format, F::SVdstSVsrc0SVsrc1);
        assert_eq!(inst.dst.type_, O::VccLo);
        assert_eq!(inst.src[1].type_, O::VccHi);
    }

    #[test]
    fn sopc_s_cmp_eq_i32() {
        // SOP2 opcode7=0x7e -> SOPC.
        let (code, _) = parse_vs(&[0xBF00_0100, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SCmpEqI32);
        assert_eq!(inst.format, F::Ssrc0Ssrc1);
        assert_eq!(inst.src_num, 2);
        assert_eq!(inst.dst.type_, O::Unknown);
    }

    #[test]
    fn sopk_s_movk_i32() {
        // SOP2 opcode7=0x60 -> SOPK opcode5=0.
        let (code, _) = parse_vs(&[0xB003_1234, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SMovkI32);
        assert_eq!(inst.format, F::SVdstSVsrc0);
        assert_eq!(inst.dst.register_id, 3);
        assert_eq!(inst.src[0].type_, O::IntegerInlineConstant);
        assert_eq!(inst.src[0].constant.i(), 0x1234);
    }

    #[test]
    fn sopp_s_waitcnt() {
        let (code, _) = parse_vs(&[0xBF8C_0070, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SWaitcnt);
        assert_eq!(inst.format, F::Imm);
        assert_eq!(inst.src[0].constant.u, 0x70);
    }

    #[test]
    fn sopp_s_cbranch_scc0_creates_labels() {
        let (code, _) = parse_vs(&[0xBF84_0002, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SCbranchScc0);
        assert_eq!(inst.format, F::Label);
        assert_eq!(inst.src[0].constant.i(), 8); // simm16 * 4
        assert_eq!(code.get_labels().len(), 1);
        assert_eq!(code.get_labels()[0].get_dst(), 12); // pc + 4 + 8
        assert_eq!(code.get_labels()[0].get_src(), 0);
        // Conditional branches also record the fall-through pc.
        assert_eq!(code.get_indirect_labels().len(), 1);
        assert_eq!(code.get_indirect_labels()[0].get_dst(), 4);
    }

    #[test]
    fn sopp_s_branch_has_no_indirect_label() {
        let (code, _) = parse_vs(&[0xBF82_0002, S_ENDPGM]);
        assert_eq!(code.get_instructions()[0].type_, T::SBranch);
        assert_eq!(code.get_labels().len(), 1);
        assert!(code.get_indirect_labels().is_empty());
    }

    #[test]
    fn vop2_v_add_f32() {
        let (code, _) = parse_vs(&[0x0600_0501, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VAddF32);
        assert_eq!(inst.format, F::SVdstSVsrc0SVsrc1);
        assert_eq!((inst.dst.type_, inst.dst.register_id), (O::Vgpr, 0));
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 1));
        assert_eq!((inst.src[1].type_, inst.src[1].register_id), (O::Vgpr, 2));
        assert_eq!(inst.dst.multiplier, 1.0);
        assert!(!inst.dst.clamp);
    }

    #[test]
    fn vop2_sdwa_clamp_and_abs() {
        // v_add_f32 with SDWA dword: src0=s2 (s0 bit set), src1_abs, clamp.
        let (code, consumed) = parse_vs(&[0x0600_02F9, 0x2686_2602, S_ENDPGM]);
        assert_eq!(consumed, 3);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VAddF32);
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Sgpr, 2));
        assert_eq!((inst.src[1].type_, inst.src[1].register_id), (O::Vgpr, 1));
        assert!(inst.src[1].absolute);
        assert!(!inst.src[0].absolute);
        assert!(inst.dst.clamp);
    }

    #[test]
    fn vop1_v_mov_b32() {
        // VOP2 opcode6=0x3f -> VOP1.
        let (code, _) = parse_vs(&[0x7E06_0200, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VMovB32);
        assert_eq!(inst.format, F::SVdstSVsrc0);
        assert_eq!((inst.dst.type_, inst.dst.register_id), (O::Vgpr, 3));
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Sgpr, 0));
    }

    #[test]
    fn vopc_v_cmp_eq_f32() {
        // VOP2 opcode6=0x3e -> VOPC; implicit VCC destination.
        let (code, _) = parse_vs(&[0x7C04_0300, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VCmpEqF32);
        assert_eq!(inst.format, F::SmaskVsrc0Vsrc1);
        assert_eq!((inst.dst.type_, inst.dst.size), (O::VccLo, 2));
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 0));
        assert_eq!((inst.src[1].type_, inst.src[1].register_id), (O::Vgpr, 1));
    }

    #[test]
    fn vop3_legacy_v_mad_f32_abs_neg() {
        // Legacy VOP3 (instr>>26 == 0x34), opcode 0x141, abs=src0, neg=src1.
        let (code, consumed) = parse_vs(&[0xD282_0100, 0x440A_0300, S_ENDPGM]);
        assert_eq!(consumed, 3);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VMadF32);
        assert_eq!(inst.format, F::VdstVsrc0Vsrc1Vsrc2);
        assert_eq!(inst.src_num, 3);
        assert_eq!((inst.dst.type_, inst.dst.register_id), (O::Vgpr, 0));
        assert!(inst.src[0].absolute);
        assert!(!inst.src[0].negate);
        assert!(inst.src[1].negate);
        assert!(!inst.src[1].absolute);
        assert_eq!(inst.src[2].register_id, 2);
    }

    #[test]
    fn vop3_next_gen_vop2_encoding() {
        // Next-gen VOP3 (instr>>26 == 0x35), opcode 0x103 = v_add_f32.
        let (code, result) = parse(
            &[0xD503_0005, 0x0002_0300, S_ENDPGM],
            ShaderType::Vertex,
            true,
        );
        assert_eq!(result.unwrap(), 3);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VAddF32);
        assert_eq!(inst.format, F::SVdstSVsrc0SVsrc1);
        assert_eq!(inst.src_num, 2);
        assert_eq!(inst.dst.register_id, 5);
    }

    #[test]
    fn vop3_next_gen_omod_multiplier() {
        // Next-gen v_mad_f32 with omod=1 (mul:2).
        let (code, result) = parse(
            &[0xD541_0001, 0x0C0A_0300, S_ENDPGM],
            ShaderType::Vertex,
            true,
        );
        assert!(result.is_ok());
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VMadF32);
        assert_eq!(inst.dst.multiplier, 2.0);
        assert_eq!(inst.dst.register_id, 1);
    }

    #[test]
    fn exp_pos0_done() {
        let (code, _) = parse_vs(&[0xF800_08CF, 0x0302_0100, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::Exp);
        assert_eq!(inst.format, F::Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done);
        assert_eq!(inst.src_num, 4);
        for (i, src) in inst.src.iter().enumerate() {
            assert_eq!((src.type_, src.register_id), (O::Vgpr, i as i32));
        }
    }

    #[test]
    fn exp_mrt0_off_compr_vm_done() {
        let (code, _) = parse_vs(&[0xF800_1C00, 0x0000_0000, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::Exp);
        assert_eq!(inst.format, F::Mrt0OffOffComprVmDone);
        assert_eq!(inst.src_num, 0);
    }

    #[test]
    fn smrd_s_load_dwordx4_imm() {
        // Legacy SMRD (mask 0xF8000000 == 0xC0000000), imm=1, offset=4.
        let (code, _) = parse_vs(&[0xC082_0304, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SLoadDwordx4);
        assert_eq!(inst.format, F::Sdst4SbaseSoffset);
        assert_eq!(
            (inst.dst.type_, inst.dst.register_id, inst.dst.size),
            (O::Sgpr, 4, 4)
        );
        assert_eq!(
            (inst.src[0].type_, inst.src[0].register_id, inst.src[0].size),
            (O::Sgpr, 2, 2)
        );
        assert_eq!(inst.src[1].type_, O::LiteralConstant);
        assert_eq!(inst.src[1].constant.u, 16); // offset << 2
    }

    #[test]
    fn smem_s_load_dword_null_soffset() {
        // Next-gen SMEM (instr>>26 == 0x3d): NULL soffset -> 21-bit imm.
        let (code, result) = parse(
            &[0xF400_0080, 0xFA00_0010, S_ENDPGM],
            ShaderType::Vertex,
            true,
        );
        assert!(result.is_ok());
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SLoadDword);
        assert_eq!(inst.format, F::SdstSbaseSoffset);
        assert_eq!((inst.dst.register_id, inst.dst.size), (2, 1));
        assert_eq!((inst.src[0].register_id, inst.src[0].size), (0, 2));
        assert_eq!(inst.src[1].type_, O::IntegerInlineConstant);
        assert_eq!(inst.src[1].constant.i(), 0x10);
    }

    #[test]
    fn smem_offset_sign_extends_21_bits() {
        // offset = 0x1FFFFF (all ones) -> -1 after 21-bit sign extension.
        let (code, result) = parse(
            &[0xF400_0080, 0xFA1F_FFFF, S_ENDPGM],
            ShaderType::Vertex,
            true,
        );
        assert!(result.is_ok());
        assert_eq!(code.get_instructions()[0].src[1].constant.i(), -1);
    }

    #[test]
    fn mubuf_buffer_load_format_xyzw() {
        let (code, _) = parse_vs(&[0xE00C_2000, 0x8001_0400, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadFormatXyzw);
        assert_eq!(inst.format, F::Vdata4VaddrSvSoffsIdxen);
        assert_eq!(
            (inst.dst.type_, inst.dst.register_id, inst.dst.size),
            (O::Vgpr, 4, 4)
        );
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 0));
        assert_eq!(
            (inst.src[1].type_, inst.src[1].register_id, inst.src[1].size),
            (O::Sgpr, 4, 4)
        );
        assert_eq!(inst.src[2].type_, O::IntegerInlineConstant);
        assert_eq!(inst.src[2].constant.i(), 0);
    }

    #[test]
    fn astro_buffer_store_format_xyzw_decodes() {
        // Measured ASTRO.BOT raw b0 = 0xE01C2000: MUBUF opcode 0x07
        // (buffer_store_format_xyzw), idxen=1, offen=0 — Kyty leaves the
        // opcode KYTY_NI (ShaderParse.cpp L2630).
        let (code, result) = parse(
            &[0xE01C_2000, 0x8001_0400, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse buffer_store_format_xyzw");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferStoreFormatXyzw);
        assert_eq!(inst.format, F::Vdata4VaddrSvSoffsIdxen);
        assert_eq!(
            (inst.dst.type_, inst.dst.register_id, inst.dst.size),
            (O::Vgpr, 4, 4)
        );
        assert_eq!(
            (inst.src[1].type_, inst.src[1].register_id, inst.src[1].size),
            (O::Sgpr, 4, 4)
        );
    }

    #[test]
    fn astro_mubuf_flexible_addressing_formats() {
        // buffer_load_dword (0x0c) across the four addressing modes. Kyty
        // EXITs on idxen==0 / offen==1 (ShaderParse.cpp L2569-2570); the port
        // selects the format instead (the BufferLoadDwordX4 model).
        // b0 base = MUBUF (0x38 << 26) | opcode 0x0c << 18.
        const BASE: u32 = 0xE030_0000;
        for (b0, format, src0_size) in [
            (BASE | (1 << 13), F::Vdata1VaddrSvSoffsIdxen, 1),
            (BASE, F::Vdata1SvSoffs, 1),
            (BASE | (1 << 12), F::Vdata1VaddrSvSoffsOffen, 1),
            (
                BASE | (1 << 13) | (1 << 12),
                F::Vdata1Vaddr2SvSoffsOffenIdxen,
                2,
            ),
        ] {
            let (code, result) = parse(&[b0, 0x8001_0400, S_ENDPGM], ShaderType::Compute, true);
            result.unwrap_or_else(|e| panic!("parse mubuf b0={b0:#010x}: {e}"));
            let inst = &code.get_instructions()[0];
            assert_eq!(inst.type_, T::BufferLoadDword, "b0={b0:#010x}");
            assert_eq!(inst.format, format, "b0={b0:#010x}");
            assert_eq!(inst.src[0].size, src0_size, "b0={b0:#010x}");
        }
    }

    /// RDNA2 MUBUF is a fixed 64-bit encoding. ASTRO.BOT emits `0xff` in the
    /// SOFFSET byte of this store; it is the no-extra-word form, not the SALU
    /// `src_literal` escape understood by the generic operand decoder. The
    /// following `s_endpgm` is also the branch target, so consuming it as a
    /// third dword reproduces the live `target is inside BufferStoreDword`
    /// relooper failure.
    #[test]
    fn next_gen_mubuf_ff_soffset_does_not_consume_branch_target() {
        let words = [
            0xBF82_0002, // s_branch +2 -> pc 0xc
            0xE070_2000, // buffer_store_dword, idxen
            0xFF01_0400, // soffset=0xff, srsrc=s4, vdata=v4, vaddr=v0
            S_ENDPGM,    // pc 0xc: branch target / next instruction
        ];
        let (code, result) = parse(&words, ShaderType::Compute, true);
        result.expect("parse fixed-width next-gen MUBUF");
        let insts = code.get_instructions();
        assert_eq!(insts.len(), 3);
        assert_eq!(insts[1].type_, T::BufferStoreDword);
        assert_eq!(insts[1].format, F::Vdata1VaddrSvSoffsIdxen);
        assert_eq!(insts[2].type_, T::SEndpgm);
        assert_eq!(insts[2].pc, 0xc, "MUBUF must span exactly two dwords");

        let (_, legacy) = parse(&words, ShaderType::Compute, false);
        let legacy = legacy.expect_err("legacy mode must not infer the PS5 0xff meaning");
        assert!(
            legacy
                .to_string()
                .contains("legacy literal soffset in fixed-width MUBUF"),
            "unexpected legacy refusal: {legacy}"
        );
    }

    #[test]
    fn astro_mubuf_strict_ops_name_the_addressing_feature() {
        // buffer_load_format_xy (0x01) keeps Kyty's strict idxen gate, now
        // applied after opcode decode.
        let b0: u32 = 0xE004_0000; // opcode 0x01, idxen=0
        let (_, result) = parse(&[b0, 0x8001_0400, S_ENDPGM], ShaderType::Compute, true);
        assert_eq!(
            result,
            Err(ShaderParseError::NotImplementedFeature {
                family: "mubuf",
                feature: "idxen == 0",
                pc: 0,
            })
        );
    }

    // ---- FLAT-class decode (SharpEmu PR #587 `Gen5FlatMemoryTests`) ----

    /// A FLAT-segment `flat_load_dword v5, v[2:3]`: the whole 64-bit address is
    /// the VGPR pair, SADDR is NULL, and `uses_flat_address` is set.
    #[test]
    fn flat_load_dword_flat_segment_decodes() {
        // word0: enc 0x37<<26 | op 0x0c<<18 | seg 0 => 0xDC300000.
        // word1: vdst v5<<24 | saddr NULL 0x7f<<16 | addr v2 => 0x057F0002.
        let (code, result) = parse(
            &[0xDC30_0000, 0x057F_0002, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse flat_load_dword");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::FlatLoadDword);
        assert_eq!(inst.format, F::FlatAddr);
        assert!(
            inst.uses_flat_address,
            "FLAT segment addresses via VGPR pair"
        );
        assert_eq!(
            (inst.src[0].type_, inst.src[0].register_id, inst.src[0].size),
            (O::Vgpr, 2, 2)
        );
        assert_eq!(inst.src[1].type_, O::Null);
        assert_eq!(
            (inst.src[2].type_, inst.src[2].constant.u),
            (O::IntegerInlineConstant, 0)
        );
        assert_eq!(
            (inst.dst.type_, inst.dst.register_id, inst.dst.size),
            (O::Vgpr, 5, 1)
        );
    }

    /// A GLOBAL-segment `global_load_dword v5, v2, s[8:9]`: the base is an SGPR
    /// pair, the VGPR is a 32-bit offset, and `uses_flat_address` is clear.
    #[test]
    fn global_load_dword_with_sgpr_base_decodes() {
        // word0: enc | op 0x0c<<18 | seg 2<<14 => 0xDC308000.
        // word1: vdst v5<<24 | saddr s8<<16 | addr v2 => 0x05080002.
        let (code, result) = parse(
            &[0xDC30_8000, 0x0508_0002, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse global_load_dword");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::FlatLoadDword);
        assert!(
            !inst.uses_flat_address,
            "GLOBAL with SADDR uses an SGPR base"
        );
        assert_eq!(
            (inst.src[0].type_, inst.src[0].register_id, inst.src[0].size),
            (O::Vgpr, 2, 1)
        );
        assert_eq!(
            (inst.src[1].type_, inst.src[1].register_id, inst.src[1].size),
            (O::Sgpr, 8, 2)
        );
        assert_eq!(
            (inst.dst.type_, inst.dst.register_id, inst.dst.size),
            (O::Vgpr, 5, 1)
        );
    }

    /// `flat_store_dword v[2:3], v6`: the store DATA VGPR lands in `dst`.
    #[test]
    fn flat_store_dword_decodes() {
        // word0: enc | op 0x1c<<18 | seg 0 => 0xDC700000.
        // word1: saddr NULL 0x7f<<16 | data v6<<8 | addr v2 => 0x007F0602.
        let (code, result) = parse(
            &[0xDC70_0000, 0x007F_0602, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse flat_store_dword");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::FlatStoreDword);
        assert!(inst.uses_flat_address);
        assert_eq!(
            (inst.src[0].type_, inst.src[0].register_id, inst.src[0].size),
            (O::Vgpr, 2, 2)
        );
        assert_eq!(
            (inst.dst.type_, inst.dst.register_id, inst.dst.size),
            (O::Vgpr, 6, 1)
        );
    }

    /// The FLAT widths x2/x3/x4 size their destinations correctly.
    #[test]
    fn flat_load_dword_widths_size_destination() {
        for (op, ty, size) in [
            (0x0du32, T::FlatLoadDwordX2, 2i32),
            (0x0f, T::FlatLoadDwordX3, 3),
            (0x0e, T::FlatLoadDwordX4, 4),
        ] {
            let word0 = 0xDC00_0000 | (op << 18);
            let (code, result) = parse(&[word0, 0x057F_0002, S_ENDPGM], ShaderType::Compute, true);
            result.unwrap_or_else(|e| panic!("parse flat width op={op:#x}: {e}"));
            let inst = &code.get_instructions()[0];
            assert_eq!(inst.type_, ty, "op={op:#x}");
            assert_eq!(inst.dst.size, size, "op={op:#x}");
        }
    }

    /// GLOBAL/SCRATCH sign-extend the 13-bit immediate offset; an all-ones
    /// field decodes to -1.
    #[test]
    fn global_load_dword_negative_offset_sign_extends() {
        // word0: enc | op 0x0c<<18 | seg 2<<14 | offset 0x1fff => 0xDC309FFF.
        let (code, result) = parse(
            &[0xDC30_9FFF, 0x0508_0002, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse global_load_dword offset");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.src[2].constant.i(), -1);
        assert_eq!(inst.src[2].constant.u, 0xFFFF_FFFF);
    }

    /// A FLAT op must not abort the whole parse — the shader keeps decoding to
    /// its terminator (the regression FLAT closed: `UnknownEncoding` at 0x37
    /// killed every downstream analysis).
    #[test]
    fn flat_load_does_not_kill_shader_parse() {
        let (code, result) = parse(
            &[0xDC30_0000, 0x057F_0002, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("flat parse");
        let insts = code.get_instructions();
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[0].type_, T::FlatLoadDword);
        assert_eq!(insts[1].type_, T::SEndpgm);
    }

    /// The FLAT class is an RDNA2 encoding; a legacy stream at 0x37 is refused.
    #[test]
    fn flat_on_legacy_is_refused() {
        let (_, result) = parse(
            &[0xDC30_0000, 0x057F_0002, S_ENDPGM],
            ShaderType::Compute,
            false,
        );
        assert_eq!(
            result,
            Err(ShaderParseError::NotImplementedFeature {
                family: "flat",
                feature: "FLAT-class encoding on legacy",
                pc: 0,
            })
        );
    }

    /// SCRATCH (stack-spill) addressing is refused by name — not on the hot path.
    #[test]
    fn flat_scratch_segment_is_refused() {
        // seg 1<<14 => 0x4000.
        let (_, result) = parse(
            &[0xDC30_4000, 0x057F_0002, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        assert_eq!(
            result,
            Err(ShaderParseError::NotImplementedFeature {
                family: "flat",
                feature: "scratch segment",
                pc: 0,
            })
        );
    }

    /// An unmodeled FLAT suffix is a named refusal, never a silent wrong decode.
    #[test]
    fn flat_unknown_opcode_is_refused() {
        // op 0x7f (not in the suffix table).
        let word0 = 0xDC00_0000 | (0x7f << 18);
        let (_, result) = parse(&[word0, 0x057F_0002, S_ENDPGM], ShaderType::Compute, true);
        assert!(matches!(
            result,
            Err(ShaderParseError::UnknownOpcode { family: "flat", .. })
        ));
    }

    #[test]
    fn mimg_image_sample_dmask_f() {
        let (code, _) = parse_vs(&[0xF080_0F00, 0x0061_0800, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageSample);
        assert_eq!(inst.format, F::Vdata4Vaddr3StSsDmaskF);
        assert_eq!((inst.dst.register_id, inst.dst.size), (8, 4));
        assert_eq!((inst.src[0].register_id, inst.src[0].size), (0, 3));
        assert_eq!(
            (inst.src[1].type_, inst.src[1].register_id, inst.src[1].size),
            (O::Sgpr, 4, 8)
        );
        assert_eq!(
            (inst.src[2].type_, inst.src[2].register_id, inst.src[2].size),
            (O::Sgpr, 12, 4)
        );
    }

    #[test]
    fn astro_mimg_new_dmask_forms_decode() {
        // The three MIMG operand-format gaps measured on ASTRO.BOT scene
        // compute: image_sample_lz (0x27) dmask 0x1/0x2, image_load (0x00)
        // dmask 0x3, image_store (0x08) dmask 0x1.
        for (b0, ty, format, dst_size) in [
            (
                0xF09C_0100u32,
                T::ImageSampleLz,
                F::Vdata1Vaddr3StSsDmask1,
                1,
            ),
            (0xF09C_0200, T::ImageSampleLz, F::Vdata1Vaddr3StSsDmask2, 1),
            (0xF000_0300, T::ImageLoad, F::Vdata2Vaddr3StDmask3, 2),
            (0xF020_0100, T::ImageStore, F::Vdata1Vaddr3StDmask1, 1),
        ] {
            let (code, result) = parse(&[b0, 0x0061_0800, S_ENDPGM], ShaderType::Compute, true);
            result.unwrap_or_else(|e| panic!("parse mimg b0={b0:#010x}: {e}"));
            let inst = &code.get_instructions()[0];
            assert_eq!(inst.type_, ty, "b0={b0:#010x}");
            assert_eq!(inst.format, format, "b0={b0:#010x}");
            assert_eq!(inst.dst.size, dst_size, "b0={b0:#010x}");
        }
    }

    #[test]
    fn minecraft_mimg_dim_controls_address_vgpr_count() {
        // Exact DIM-bearing words from Minecraft's panorama copy shader.
        // Both operations are DIM_2D (bits [5:3] == 1), even though their
        // runtime T# descriptors are type-13 arrays. BASE_ARRAY selects the
        // face; vaddr+2 is not part of either instruction.
        let (load_code, load_result) = parse(
            &[0xF000_0F0A, 0x0000_0003, 0x0000_0000, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        load_result.expect("parse Minecraft NSA image_load");
        let load = &load_code.get_instructions()[0];
        assert_eq!(load.type_, T::ImageLoad);
        assert_eq!(load.src[0].size, 2);
        assert_eq!(load.mimg_nsa_dwords, 1);
        assert_eq!(
            load.mimg_nsa_addr[0].register_id, 0,
            "the NSA dword explicitly selects v0 as Y"
        );
        assert_eq!(load_code.get_instructions().len(), 2);
        assert_eq!(load_code.get_instructions()[1].type_, T::SEndpgm);
        assert_eq!(
            load_code.get_instructions()[1].pc,
            12,
            "the end marker follows the three-dword MIMG; the NSA payload \
             must not decode as a fake VOP instruction"
        );

        let (store_code, store_result) = parse(
            &[0xF020_0F08, 0x0006_0004, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        store_result.expect("parse Minecraft image_store");
        let store = &store_code.get_instructions()[0];
        assert_eq!(store.type_, T::ImageStore);
        assert_eq!(
            store.src[0].size, 2,
            "DIM_2D consumes only x/y address VGPRs"
        );
        assert_eq!(store.mimg_nsa_dwords, 0);

        let (code, result) = parse(
            &[0xF020_0F18, 0x0006_0004, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse DIM_2D_ARRAY image_store");
        assert_eq!(
            code.get_instructions()[0].src[0].size,
            3,
            "DIM_2D_ARRAY consumes x/y/layer"
        );
    }

    #[test]
    fn astro_exp_pos1_partial_export_decodes() {
        // Measured: exp target 0x0d (pos1) with en=0x4, done=0, compr=0,
        // vm=0 — an auxiliary position export (clip/cull distance per
        // PA_CL_VS_OUT_CNTL; shadPS4 ir/position.h). 632 failures / 30s.
        let (code, result) = parse(
            &[0xF800_00D4, 0x0302_0100, S_ENDPGM],
            ShaderType::Vertex,
            true,
        );
        result.expect("parse exp pos1");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::Exp);
        assert_eq!(inst.format, F::Pos1Vsrc0Vsrc1Vsrc2Vsrc3);
        assert_eq!(inst.export_enable, 0x4);
        // pos2/pos3 ride the same path.
        let (code, result) = parse(
            &[0xF800_00E4, 0x0302_0100, S_ENDPGM],
            ShaderType::Vertex,
            true,
        );
        result.expect("parse exp pos2");
        assert_eq!(
            code.get_instructions()[0].format,
            F::Pos2Vsrc0Vsrc1Vsrc2Vsrc3
        );
    }

    #[test]
    fn astro_vop3_v_add3_u32_decodes() {
        // VOP3 0x36d (shadPS4 V_ADD3_U32 = 877): v1 = v0 + v1 + v2.
        let (code, result) = parse(
            &[0xD76D_0001, 0x040A_0300, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse v_add3_u32");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VAdd3U32);
        assert_eq!(inst.format, F::VdstVsrc0Vsrc1Vsrc2);
        assert_eq!(inst.src_num, 3);
        assert_eq!((inst.dst.type_, inst.dst.register_id), (O::Vgpr, 1));
        for (i, reg) in [0, 1, 2].into_iter().enumerate() {
            assert_eq!((inst.src[i].type_, inst.src[i].register_id), (O::Vgpr, reg));
        }
    }

    #[test]
    fn astro_ds_lds_op_reports_instruction_name() {
        // A still-unimplemented LDS op with non-zero addr/offset fields must
        // fail by NAME (previously the pre-switch `addr != 0` check hid the
        // opcode — 173 anonymous ASTRO.BOT failures). ds_write2_b32 = 0x0e.
        let (_, result) = parse(
            &[0xD838_0000, 0x0000_0005, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        match result {
            Err(ShaderParseError::NotImplemented {
                family,
                instruction,
                opcode,
                ..
            }) => {
                assert_eq!(family, "ds");
                assert_eq!(instruction, "ds_write2_b32");
                assert_eq!(opcode, 0x0e);
            }
            other => panic!("expected named ds NI, got {other:?}"),
        }
    }

    #[test]
    fn astro_ds_write_b32_parses_addr_data_offset() {
        // ds_write_b32 (0x0d) — measured raw 0xd8340000 with addr/data VGPRs
        // in b1 (here addr=v5, data0=v3) plus a 16-bit byte offset in b0.
        let (code, result) = parse(
            &[0xD834_0010, 0x0000_0305, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse ds_write_b32");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::DsWriteB32);
        assert_eq!(inst.format, F::Vsrc0Vsrc1Vsrc2);
        assert_eq!(inst.src_num, 3);
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 5));
        assert_eq!((inst.src[1].type_, inst.src[1].register_id), (O::Vgpr, 3));
        assert_eq!(inst.src[2].type_, O::LiteralConstant);
        assert_eq!(inst.src[2].constant.u, 0x10);
    }

    #[test]
    fn astro_ds_read_b32_parses_vdst_addr_offset() {
        // ds_read_b32 (0x36): vdst=v7, addr=v5, offset 4.
        let (code, result) = parse(
            &[0xD8D8_0004, 0x0700_0005, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse ds_read_b32");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::DsReadB32);
        assert_eq!(inst.format, F::SVdstSVsrc0SVsrc1);
        assert_eq!((inst.dst.type_, inst.dst.register_id), (O::Vgpr, 7));
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 5));
        assert_eq!(inst.src[1].type_, O::LiteralConstant);
        assert_eq!(inst.src[1].constant.u, 4);
    }

    #[test]
    fn astro_ds_read2_b32_parses_two_dword_offsets() {
        // ds_read2_b32 (0x37) — measured raw 0xd8dc0100: offset0=0,
        // offset1=1 (DWORD units → byte literals 0 and 4). vdst=v7 (pair
        // v7/v8), addr=v5.
        let (code, result) = parse(
            &[0xD8DC_0100, 0x0700_0005, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse ds_read2_b32");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::DsRead2B32);
        assert_eq!(inst.format, F::Vdst2Vsrc0Vsrc1Vsrc2);
        assert_eq!(
            (inst.dst.type_, inst.dst.register_id, inst.dst.size),
            (O::Vgpr, 7, 2)
        );
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 5));
        assert_eq!(inst.src[1].type_, O::LiteralConstant);
        assert_eq!(inst.src[1].constant.u, 0, "offset0 in bytes");
        assert_eq!(inst.src[2].type_, O::LiteralConstant);
        assert_eq!(inst.src[2].constant.u, 4, "offset1 scaled to bytes");
    }

    #[test]
    fn astro_ds_write_b96_parses_three_dword_data() {
        // ds_write_b96 (0xde): addr=v5, data0=v[3:5], byte offset 8.
        let (code, result) = parse(
            &[0xDB78_0008, 0x0000_0305, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse ds_write_b96");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::DsWriteB96);
        assert_eq!(inst.format, F::Vsrc0Vsrc13Vsrc2);
        assert_eq!(inst.dst.type_, O::Unknown, "no destination");
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 5));
        assert_eq!(
            (inst.src[1].type_, inst.src[1].register_id, inst.src[1].size),
            (O::Vgpr, 3, 3)
        );
        assert_eq!(inst.src[2].type_, O::LiteralConstant);
        assert_eq!(inst.src[2].constant.u, 8);
    }

    #[test]
    fn astro_ds_append_carries_byte_offset_as_literal() {
        // ds_append (0x3e, gds) with instruction offset 4: the counter is one
        // dword past the M0 base (shadPS4 DS_APPEND: gds_offset = M0 +
        // inst_offset). Previously "ds feature: offset0 != 0" (59 measured).
        let (code, result) = parse(
            &[0xD8FA_0004, 0x0700_0000, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse ds_append with offset");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::DsAppend);
        assert_eq!(inst.src_num, 1);
        assert_eq!(inst.src[0].type_, O::LiteralConstant);
        assert_eq!(inst.src[0].constant.u, 4);

        // Unaligned counter offsets stay a named refusal.
        let (_, result) = parse(
            &[0xD8FA_0002, 0x0700_0000, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        assert_eq!(
            result,
            Err(ShaderParseError::NotImplementedFeature {
                family: "ds",
                feature: "append/consume offset not dword-aligned",
                pc: 0,
            })
        );
    }

    #[test]
    fn astro_mubuf_immediate_offset_folds_into_soffset() {
        // buffer_load_dword (0x0c, idxen) with immediate offset 16 — the
        // offset is one addend of the documented flexible-address model and
        // folds into the constant soffset (inline 0 here). Previously
        // "mubuf feature: offset != 0" (116 measured).
        let (code, result) = parse(
            &[0xE030_2010, 0x8001_0400, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse buffer_load_dword with immediate offset");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadDword);
        assert_eq!(inst.format, F::Vdata1VaddrSvSoffsIdxen);
        assert_eq!(inst.src[2].type_, O::IntegerInlineConstant);
        assert_eq!(inst.src[2].constant.u, 16, "0 (inline) + 16 (immediate)");

        // A register soffset cannot absorb the immediate — named refusal.
        let (_, result) = parse(
            &[0xE030_2010, 0x0501_0400, S_ENDPGM], // soffset = s5
            ShaderType::Compute,
            true,
        );
        assert_eq!(
            result,
            Err(ShaderParseError::NotImplementedFeature {
                family: "mubuf",
                feature: "offset != 0 with register soffset",
                pc: 0,
            })
        );
    }

    #[test]
    fn astro_vop2_sdwa_omod_is_preserved() {
        // v_mul_f32 SDWA with omod=1 (mul:2) — same lowering as the VOP1
        // SDWA omod path: the multiplier rides `dst.multiplier`. Previously
        // "vop2 feature: sdwa omod != 0" (58 measured).
        let (code, result) = parse(
            &[0x1002_06F9, 0x1606_4604, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse VOP2 SDWA omod");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VMulF32);
        assert_eq!(inst.dst.multiplier, 2.0);
        assert!(inst.src[1].negate, "the other SDWA modifiers still apply");
    }

    #[test]
    fn operand_248_is_inv_2pi() {
        // RDNA2 inline float 248 = 1/(2*pi) (SharpEmu Gen5InlineConstants;
        // Kyty predates it). v_mul_f32 v1, 1/(2*pi), v3. Previously
        // "unknown operand: 248" (58 measured).
        let (code, result) = parse(&[0x1002_06F8, S_ENDPGM], ShaderType::Compute, true);
        result.expect("parse v_mul_f32 with inline 1/(2*pi)");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.src[0].type_, O::FloatInlineConstant);
        assert!((inst.src[0].constant.f() - 0.159_154_94).abs() < 1e-9);
    }

    #[test]
    fn mtbuf_tbuffer_load_format_xyzw() {
        let (code, _) = parse_vs(&[0xEBF3_2000, 0x8001_0400, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::TBufferLoadFormatXyzw);
        assert_eq!(inst.format, F::Vdata4VaddrSvSoffsIdxenFloat4);
        assert_eq!((inst.dst.register_id, inst.dst.size), (4, 4));
        assert_eq!(inst.src[0].size, 1); // no offen
        assert_eq!(inst.src[1].size, 4);
    }

    #[test]
    fn ds_append_gds() {
        let (code, _) = parse_vs(&[0xD8FA_0000, 0x0700_0000, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::DsAppend);
        assert_eq!(inst.format, F::VdstGds);
        assert_eq!((inst.dst.type_, inst.dst.register_id), (O::Vgpr, 7));
        assert_eq!(inst.src_num, 0);
    }

    #[test]
    fn astro_ds_append_accepts_nonzero_dont_care_addr() {
        // DS_APPEND/CONSUME select the GDS counter from M0. The encoded addr
        // VGPR is not consumed by these two opcodes (shadPS4 Gen5 agrees), and
        // ASTRO.BOT leaves it non-zero.
        let (code, result) = parse(
            &[0xD8FA_0000, 0x0700_0009, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse DS_APPEND with non-zero don't-care addr");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::DsAppend);
        assert_eq!(inst.format, F::VdstGds);
        assert_eq!(inst.src_num, 0);
    }

    #[test]
    fn vintrp_v_interp_p1_f32() {
        let (code, _) = parse_vs(&[0xC814_0D02, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VInterpP1F32);
        assert_eq!(inst.format, F::VdstVsrcAttrChan);
        assert_eq!((inst.dst.type_, inst.dst.register_id), (O::Vgpr, 5));
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 2));
        assert_eq!(inst.src[1].constant.u, 3); // attr
        assert_eq!(inst.src[2].constant.u, 1); // chan
        assert_eq!(inst.src_num, 3);
    }

    // ---- program-level walks + end detection (Kyty L3348/L3406) ----

    #[test]
    fn full_program_walk() {
        let (code, consumed) = parse_vs(&[
            0xBE80_0301, // s_mov_b32 s0, s1
            0x0600_0501, // v_add_f32 v0, v1, v2
            0xF800_08CF,
            0x0302_0100, // exp pos0 v0..v3 done
            S_ENDPGM,
        ]);
        assert_eq!(consumed, 5);
        let insts = code.get_instructions();
        assert_eq!(insts.len(), 4);
        assert_eq!(
            insts.iter().map(|i| i.pc).collect::<Vec<_>>(),
            vec![0, 4, 8, 16]
        );
        assert_eq!(insts[3].type_, T::SEndpgm);
    }

    #[test]
    fn endpgm_with_live_label_does_not_end_parse() {
        // s_cbranch_scc0 jumps over the first s_endpgm -> parsing continues.
        let (code, consumed) = parse_vs(&[
            0xBF84_0001, // s_cbranch_scc0 +4 (label dst = 8)
            S_ENDPGM,    // pc 4: next pc (8) is a live label target
            0xBE80_0301, // pc 8
            S_ENDPGM,    // pc 12: final
        ]);
        assert_eq!(consumed, 4);
        assert_eq!(code.get_instructions().len(), 4);
    }

    #[test]
    fn fetch_shader_ends_on_s_setpc() {
        let (code, result) = parse(&[0xBE80_2000], ShaderType::Fetch, false);
        assert_eq!(result.unwrap(), 1);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SSetpcB64);
        assert_eq!(inst.format, F::Saddr);
        assert_eq!((inst.src[0].type_, inst.src[0].size), (O::Sgpr, 2));
    }

    #[test]
    fn shader_code_helpers_on_parsed_program() {
        let (code, _) = parse_vs(&[0xBE80_0301, 0xF800_08CF, 0x0302_0100, S_ENDPGM]);
        assert!(code.has_any_of(&[T::Exp]));
        assert!(code.has_any_of(&[T::VMadF32, T::SMovB32]));
        assert!(!code.has_any_of(&[T::VMadF32]));

        let block = code.read_block(0);
        assert!(block.is_valid);
        assert!(!block.is_discard);
        assert_eq!(block.last.type_, T::SEndpgm);

        let insts = code.read_intructions(&block);
        assert_eq!(insts.len(), 3);

        let dump = code.dbg_dump();
        assert!(dump.contains("SMovB32"), "{dump}");
        assert!(dump.contains("Exp"), "{dump}");
        assert!(dump.contains("pos0"), "{dump}");
    }

    #[test]
    fn dbg_dump_prints_labels() {
        let (code, _) = parse_vs(&[0xBF84_0001, S_ENDPGM, 0xBE80_0301, S_ENDPGM]);
        let dump = code.dbg_dump();
        assert!(dump.contains("label_0008:"), "{dump}");
        assert!(dump.contains("SCbranchScc0"), "{dump}");
    }

    // ---- 4. malformed input: typed errors, no panics ----

    #[test]
    fn truncated_literal_is_error() {
        let (_, result) = parse(&[0xBE85_03FF], ShaderType::Vertex, false);
        assert_eq!(result, Err(ShaderParseError::Truncated { pc: 0 }));
    }

    #[test]
    fn truncated_vop3_is_error() {
        let (_, result) = parse(&[0xD282_0100], ShaderType::Vertex, false);
        assert_eq!(result, Err(ShaderParseError::Truncated { pc: 0 }));
    }

    #[test]
    fn empty_and_unterminated_programs_are_errors() {
        let (_, result) = parse(&[], ShaderType::Vertex, false);
        assert_eq!(result, Err(ShaderParseError::Truncated { pc: 0 }));
        // Valid instruction but no s_endpgm: the walk hits the end bound.
        let (_, result) = parse(&[0x0600_0501], ShaderType::Vertex, false);
        assert_eq!(result, Err(ShaderParseError::Truncated { pc: 4 }));
    }

    #[test]
    fn unknown_opcode_is_typed_error() {
        // SOPC opcode 0x20 is past Kyty's table.
        let (_, result) = parse(&[0xBF20_0100], ShaderType::Vertex, false);
        assert_eq!(
            result,
            Err(ShaderParseError::UnknownOpcode {
                family: "sopc",
                opcode: 0x20,
                pc: 0,
                raw: 0xBF20_0100,
            })
        );
    }

    #[test]
    fn s_nop_is_a_decoded_no_op() {
        let (code, result) = parse(&[0xBF80_0000, S_ENDPGM], ShaderType::Vertex, true);
        assert_eq!(result.unwrap(), 2);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SNop);
        assert_eq!(inst.format, F::Imm);
        assert_eq!(inst.src[0].constant.u, 0);
    }

    #[test]
    fn unknown_encoding_is_typed_error() {
        // instr>>26 == 0x33 matches no family.
        let (_, result) = parse(&[0xCC00_0000], ShaderType::Vertex, false);
        assert_eq!(
            result,
            Err(ShaderParseError::UnknownEncoding {
                pc: 0,
                raw: 0xCC00_0000
            })
        );
    }

    #[test]
    fn unknown_operand_is_typed_error() {
        // s_add_u32 with ssrc0 = 104 (hole in the operand map).
        let (_, result) = parse(&[0x8002_0168], ShaderType::Vertex, false);
        assert_eq!(result, Err(ShaderParseError::UnknownOperand { code: 104 }));
    }

    #[test]
    fn encoding_generation_mismatches_are_errors() {
        // Legacy SMRD on next_gen (Kyty L3371).
        let (_, result) = parse(&[0xC082_0304], ShaderType::Vertex, true);
        assert!(matches!(
            result,
            Err(ShaderParseError::NotImplementedFeature { family: "smrd", .. })
        ));
        // Legacy VOP3 encoding on next_gen (Kyty L3382).
        let (_, result) = parse(&[0xD282_0100, 0x440A_0300], ShaderType::Vertex, true);
        assert!(matches!(
            result,
            Err(ShaderParseError::NotImplementedFeature { family: "vop3", .. })
        ));
        // Next-gen VOP3 encoding on legacy (Kyty L3386).
        let (_, result) = parse(&[0xD503_0005, 0x0002_0300], ShaderType::Vertex, false);
        assert!(matches!(
            result,
            Err(ShaderParseError::NotImplementedFeature { family: "vop3", .. })
        ));
        // Next-gen SMEM encoding on legacy (Kyty L3394).
        let (_, result) = parse(&[0xF400_0080, 0xFA00_0010], ShaderType::Vertex, false);
        assert!(matches!(
            result,
            Err(ShaderParseError::NotImplementedFeature { family: "smem", .. })
        ));
    }

    #[test]
    fn mimg_unknown_dmask_is_error() {
        // image_sample with an as-yet-unwired single-channel Z dmask.
        let (_, result) = parse(&[0xF080_0400, 0x0061_0800], ShaderType::Vertex, false);
        assert_eq!(
            result,
            Err(ShaderParseError::UnknownMimgFormat {
                opcode: 0x20,
                dmask: 4,
                pc: 0
            })
        );
    }

    #[test]
    fn astro_pixel_mimg_dmask1_and_dmask2_variants_decode() {
        let (sample, result) = parse(
            &[0xF080_0200, 0x0061_0800, S_ENDPGM],
            ShaderType::Pixel,
            true,
        );
        result.expect("image_sample dmask 0x2");
        assert_eq!(sample.get_instructions()[0].type_, T::ImageSample);
        assert_eq!(
            sample.get_instructions()[0].format,
            F::Vdata1Vaddr3StSsDmask2
        );

        let (sample_lzo, result) = parse(
            &[0xF0DC_0100, 0x0061_0800, S_ENDPGM],
            ShaderType::Pixel,
            true,
        );
        result.expect("image_sample_lz_o dmask 0x1");
        assert_eq!(sample_lzo.get_instructions()[0].type_, T::ImageSampleLzO);
        assert_eq!(
            sample_lzo.get_instructions()[0].format,
            F::Vdata1Vaddr4StSsDmask1
        );

        let (sample_lzo_y, result) = parse(
            &[0xF0DC_0200, 0x0061_0800, S_ENDPGM],
            ShaderType::Pixel,
            true,
        );
        result.expect("image_sample_lz_o dmask 0x2");
        assert_eq!(
            sample_lzo_y.get_instructions()[0].format,
            F::Vdata1Vaddr4StSsDmask2
        );
    }

    #[test]
    fn mimg_image_load_dmask_1_and_7() {
        // image_load (opcode 0x00) with the partial dmasks ASTRO.BOT's scene
        // compute shaders use; only dmask 0xf had a format before.
        let (code, _) = parse_vs(&[0xF000_0100, 0x0061_0800, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageLoad);
        assert_eq!(inst.format, F::Vdata1Vaddr3StDmask1);
        assert_eq!(inst.dst.size, 1);
        assert_eq!(inst.src_num, 2);
        assert_eq!(inst.src[0].size, 3);
        assert_eq!(inst.src[1].size, 8);

        let (code, _) = parse_vs(&[0xF000_0700, 0x0061_0800, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageLoad);
        assert_eq!(inst.format, F::Vdata3Vaddr3StDmask7);
        assert_eq!(inst.dst.size, 3);
    }

    #[test]
    fn minecraft_nsa_copy_shader_keeps_every_instruction_boundary() {
        // The live Minecraft panorama/cubemap copy kernel. All four MIMG
        // instructions carry NSA payload dwords, so preserving their exact
        // lengths is essential: treating an NSA payload as VOP changes both
        // the coordinates and every later branch/instruction boundary.
        let words = [
            0xbfa0_0001,
            0xd746_0001,
            0x0405_060f,
            0xd746_0000,
            0x0401_060e,
            0xf424_1a84,
            0xfa00_0000,
            0x7e04_0d01,
            0x7e02_0d00,
            0xbf8c_c07f,
            0x7c28_046b,
            0x7c28_026a,
            0xbf88_0013,
            0xf42c_0404,
            0xfa00_0010,
            0xbf8c_c07f,
            0x0606_0210,
            0x0600_0411,
            0x0602_0214,
            0x0604_0415,
            0xf40c_0606,
            0xfa00_0000,
            0x7e06_1103,
            0x7e00_1100,
            0x7e08_1101,
            0x7e0a_1102,
            0xf000_0f0a,
            0x0000_0003,
            0x0000_0000,
            0xbf8c_0070,
            0xf020_0f08,
            0x0006_0004,
            S_ENDPGM,
        ];
        let (code, result) = parse(&words, ShaderType::Compute, true);
        result.expect("parse Minecraft NSA copy shader");
        let instructions = code.get_instructions();
        assert_eq!(
            instructions.iter().map(|inst| inst.pc).collect::<Vec<_>>(),
            [
                0x00, 0x04, 0x0c, 0x14, 0x1c, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x34, 0x3c, 0x40, 0x44,
                0x48, 0x4c, 0x50, 0x58, 0x5c, 0x60, 0x64, 0x68, 0x74, 0x78, 0x80,
            ]
        );
        assert_eq!(instructions[7].type_, T::VCmpxGtF32);
        assert_eq!(instructions[7].src[0].type_, O::VccHi);
        assert_eq!(instructions[7].dst.type_, O::ExecLo);
        assert_eq!(instructions[8].type_, T::VCmpxGtF32);
        assert_eq!(instructions[8].src[0].type_, O::VccLo);
        assert_eq!(instructions[8].dst.type_, O::ExecLo);
        assert_eq!(instructions[21].type_, T::ImageLoad);
        assert_eq!(instructions[23].type_, T::ImageStore);
    }

    #[test]
    fn mimg_image_sample_lz_dmask_f() {
        // image_sample_lz (opcode 0x27) rgba — ASTRO.BOT's fullscreen
        // composite samples its HDR scene buffers this way.
        let (code, _) = parse_vs(&[0xF09C_0F00, 0x0061_0800, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageSampleLz);
        assert_eq!(inst.format, F::Vdata4Vaddr3StSsDmaskF);
        assert_eq!(inst.dst.size, 4);
        assert_eq!(inst.src[2].size, 4, "S# still present on the lz form");
    }

    #[test]
    fn astro_image_sample_lz_dmask3_decodes() {
        // image_sample_lz (opcode 0x27) .xy — measured on ASTRO.BOT scene
        // compute (58 dispatches/run said "unknown mimg format for opcode:
        // 0x27, dmask: 0x3").
        let (code, result) = parse(
            &[0xF09C_0300, 0x0061_0800, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse image_sample_lz dmask 0x3");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageSampleLz);
        assert_eq!(inst.format, F::Vdata2Vaddr3StSsDmask3);
        assert_eq!(inst.dst.size, 2);
        assert_eq!(inst.src[2].size, 4, "S# still present on the lz form");
    }

    #[test]
    fn astro_buffer_load_dwordx3_decodes() {
        // Measured raw first dword (0xe03c2074): MUBUF 0x0f, idxen, immediate
        // offset 0x74 — folded into the soffset constant per the flexible
        // addressing model.
        let (code, result) = parse(
            &[0xE03C_2074, 0x8001_0400, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse buffer_load_dwordx3");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadDwordX3);
        assert_eq!(inst.format, F::Vdata3VaddrSvSoffsIdxen);
        assert_eq!(inst.dst.size, 3);
        assert_eq!(inst.src[1].size, 4, "V# quad");
        assert_eq!(
            inst.src[2].constant.u, 0x74,
            "immediate offset folded into the inline-zero soffset"
        );
    }

    #[test]
    fn sopk_s_waitcnt_vscnt_is_an_emitted_wait_boundary() {
        // RDNA2 `s_waitcnt_vscnt` (SOPK 0x17), measured raw 0xbbfd0000.
        // Its sdst field is reserved rather than a register, so it is handled
        // before operand parsing and emitted as the common wait type.
        let (code, result) = parse(&[0xBBFD_0000, S_ENDPGM], ShaderType::Compute, true);
        result.expect("s_waitcnt_vscnt must not fail the parse");
        let insts = code.get_instructions();
        assert_eq!(insts.len(), 2, "wait instruction remains a boundary");
        assert_eq!(
            (insts[0].pc, insts[0].type_, insts[0].format),
            (0, T::SWaitcnt, F::Imm)
        );
        assert_eq!((insts[1].pc, insts[1].type_), (4, T::SEndpgm));
    }

    #[test]
    fn sopk_s_version_is_next_gen_only() {
        // GFX10 SOPK opcode 1 is s_version. It is metadata-only, but remains
        // a named instruction/boundary instead of masquerading as s_movk.
        let (code, result) = parse(&[0xB080_0001, S_ENDPGM], ShaderType::Compute, true);
        result.expect("next-gen s_version parses");
        assert_eq!(code.get_instructions()[0].type_, T::SVersion);
        let (_, legacy) = parse(&[0xB080_0001, S_ENDPGM], ShaderType::Compute, false);
        assert!(legacy.is_err(), "legacy SOPK opcode 1 is not s_version");
    }

    #[test]
    fn branch_to_s_waitcnt_vscnt_after_mubuf_has_a_real_boundary() {
        // Reduced form of the live ASTRO.BOT failure: the 64-bit MUBUF ends
        // at pc 0xc, which is an s_waitcnt_vscnt and the branch target. When
        // that wait was consumed without emitting an instruction, the next
        // recorded boundary was 0x10 and the relooper blamed the MUBUF.
        let words = [
            0xBF82_0002, // s_branch +2 -> pc 0xc
            0xE070_2000, // buffer_store_dword, idxen
            0xFF01_0400, // fixed-width second MUBUF dword
            0xBBFD_0000, // s_waitcnt_vscnt at the live branch target
            S_ENDPGM,
        ];
        let (code, result) = parse(&words, ShaderType::Compute, true);
        result.expect("parse branch to s_waitcnt_vscnt boundary");
        let insts = code.get_instructions();
        assert_eq!((insts[1].pc, insts[1].type_), (4, T::BufferStoreDword));
        assert_eq!((insts[2].pc, insts[2].type_), (0xc, T::SWaitcnt));
        assert!(
            code.get_labels().iter().any(|label| label.get_dst() == 0xc),
            "the branch target must resolve to the emitted wait boundary"
        );
    }

    #[test]
    fn astro_ds_read_b64_decodes() {
        // Measured raw first dword (0xd9d80000): DS 0x76, two CONSECUTIVE
        // dwords at one byte offset — parsed into the DsRead2B32 shape with
        // the second offset literal at offset + 4.
        let (code, result) = parse(
            &[0xD9D8_0000, 0x0500_0002, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse ds_read_b64");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::DsReadB64);
        assert_eq!(inst.format, F::Vdst2Vsrc0Vsrc1Vsrc2);
        assert_eq!((inst.dst.register_id, inst.dst.size), (5, 2));
        assert_eq!(inst.src[0].register_id, 2, "LDS address VGPR");
        assert_eq!(inst.src[1].constant.u, 0);
        assert_eq!(inst.src[2].constant.u, 4);

        // A nonzero 16-bit byte offset rides into both literals.
        let (code2, result2) = parse(
            &[0xD9D8_0110, 0x0500_0002, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result2.expect("parse ds_read_b64 with offset");
        let inst2 = &code2.get_instructions()[0];
        assert_eq!(inst2.src[1].constant.u, 0x110);
        assert_eq!(inst2.src[2].constant.u, 0x114);
    }

    #[test]
    fn astro_image_sample_c_lz_dmask1_decodes_reference_and_xy() {
        // image_sample_c_lz (opcode 0x2f), dmask=x. Its three address VGPRs
        // are depth-reference, x, y in that order.
        let (code, result) = parse(
            &[0xF0BC_0100, 0x0061_0800, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse image_sample_c_lz");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageSampleCLz);
        assert_eq!(inst.format, F::Vdata1Vaddr3StSsDmask1);
        assert_eq!((inst.dst.register_id, inst.dst.size), (8, 1));
        assert_eq!(inst.src[0].size, 3);
        assert_eq!(inst.src[1].size, 8);
        assert_eq!(inst.src[2].size, 4);
    }

    #[test]
    fn sop1_s_orn2_saveexec_b64() {
        // Measured ASTRO.BOT encoding 0xBE92287E: sdst=s[18:19], opcode 0x28,
        // ssrc0=0x7e (exec). `sdst = exec; exec = ssrc0 | ~exec`.
        let (code, _) = parse_vs(&[0xBE92_287E, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SOrn2SaveexecB64);
        assert_eq!(inst.format, F::Sdst2Ssrc02);
        assert_eq!(inst.dst.size, 2);
        assert_eq!(inst.src[0].size, 2);
    }

    #[test]
    fn astro_s_getpc_b64_captures_the_absolute_following_pc() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        code.set_base_address(0x0000_0005_0074_e000);
        shader_parse(0, &[0xBE80_1F00, S_ENDPGM], &mut code, true).expect("parse s_getpc_b64");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SGetpcB64);
        assert_eq!(inst.format, F::Sdst2);
        assert_eq!((inst.dst.register_id, inst.dst.size), (0, 2));
        assert_eq!(inst.src_num, 2);
        assert_eq!(inst.src[0].constant.u, 0x0074_e004);
        assert_eq!(inst.src[1].constant.u, 0x0000_0005);
    }

    #[test]
    fn astro_s_pack_ll_b32_b16_decodes() {
        // Measured ASTRO.BOT scene-compute encoding.
        let (code, result) = parse(&[0x9935_806B, S_ENDPGM], ShaderType::Compute, true);
        result.expect("parse s_pack_ll_b32_b16");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SPackLlB32B16);
        assert_eq!(inst.format, F::SVdstSVsrc0SVsrc1);
    }

    #[test]
    fn astro_vop1_sdwa_omod_is_preserved() {
        // v_rcp_f32 v1, v5 with SDWA omod=2 (multiply result by 2.0).
        let (code, result) = parse(
            &[0x7E02_54F9, 0x0026_4605, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse VOP1 SDWA omod");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VRcpF32);
        assert_eq!(inst.dst.multiplier, 2.0);
    }

    #[test]
    fn astro_buffer_load_dwordx4_accepts_address_only_mode() {
        // idxen=0/offen=0: the buffer descriptor/soffset form has no VGPR
        // index contribution. This is emitted by ASTRO.BOT compute shaders.
        let (code, result) = parse(
            &[0xE038_0000, 0x8001_0400, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse address-only buffer_load_dwordx4");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadDwordX4);
        assert_eq!(inst.format, F::Vdata4SvSoffs);
        assert_eq!(inst.dst.size, 4);
    }

    #[test]
    fn astro_image_get_resinfo_dmask3_decodes() {
        // Raw first dword measured in ASTRO.BOT scene compute; dmask=xy.
        let (code, result) = parse(
            &[0xF038_0308, 0x0001_0400, S_ENDPGM],
            ShaderType::Compute,
            true,
        );
        result.expect("parse image_get_resinfo");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageGetResinfo);
        assert_eq!(inst.format, F::Vdata2VaddrStDmask3);
        assert_eq!((inst.dst.register_id, inst.dst.size), (4, 2));
        assert_eq!(inst.src_num, 2);
        assert_eq!(inst.src[0].size, 1);
        assert_eq!(inst.src[1].size, 8);
    }

    #[test]
    fn parse_is_total_over_arbitrary_bytes() {
        // Pseudo-random dwords: every outcome must be Ok or a typed error —
        // never a panic, never an out-of-bounds read.
        let mut seed: u32 = 0x1234_5678;
        for _ in 0..512 {
            let mut src = [0u32; 8];
            for dw in &mut src {
                // xorshift32
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *dw = seed;
            }
            for next_gen in [false, true] {
                let mut code = ShaderCode::new();
                code.set_type(ShaderType::Pixel);
                let _ = shader_parse(0, &src, &mut code, next_gen);
            }
        }
    }

    #[test]
    fn vop2_madmk_reads_literal_between_sources() {
        // v_madmk_f32 (VOP2 op 0x20): src1 = literal K, src2 = old vsrc1.
        // dword: op 0x20<<25 | vdst 0<<17 | vsrc1 3<<9 | src0 257.
        let dw0 = (0x20 << 25) | (3 << 9) | 257;
        let k = 2.5f32.to_bits();
        let (code, consumed) = parse_vs(&[dw0, k, S_ENDPGM]);
        assert_eq!(consumed, 3);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VMadmkF32);
        assert_eq!(inst.format, F::VdstVsrc0Vsrc1Vsrc2);
        assert_eq!(inst.src_num, 3);
        assert_eq!((inst.src[0].type_, inst.src[0].register_id), (O::Vgpr, 1));
        assert_eq!(inst.src[1].type_, O::LiteralConstant);
        assert_eq!(inst.src[1].constant.f(), 2.5);
        assert_eq!((inst.src[2].type_, inst.src[2].register_id), (O::Vgpr, 3));
    }

    #[test]
    fn vop2_v_add_i32_has_vcc_dst2() {
        // v_add_i32 (VOP2 op 0x25, legacy): dst2 = VCC.
        let dw0 = (0x25 << 25) | (1 << 17) | (2 << 9) | 256;
        let (code, _) = parse_vs(&[dw0, S_ENDPGM]);
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VAddI32);
        assert_eq!(inst.format, F::VdstSdst2Vsrc0Vsrc1);
        assert_eq!((inst.dst2.type_, inst.dst2.size), (O::VccLo, 2));
        assert_eq!(inst.dst.register_id, 1);
        // ...and on next_gen the same slot is RDNA2's carry-less
        // v_add_nc_u32 (measured in Minecraft's menu CS): no dst2.
        let (code, result) = parse(&[dw0, S_ENDPGM], ShaderType::Vertex, true);
        result.expect("v_add_nc_u32 parses on next_gen");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VAddNcU32);
        assert_eq!(inst.format, F::SVdstSVsrc0SVsrc1);
        assert_eq!(inst.dst2.type_, O::Unknown);
    }
}
