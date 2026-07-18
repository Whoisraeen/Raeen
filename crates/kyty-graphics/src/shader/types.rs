//! GCN shader data model, ported from Kyty (MIT (c) InoriRus).
//!
//! Kyty sources:
//! - `emulator/include/Emulator/Graphics/Shader.h` (data model)
//! - `emulator/src/Graphics/Shader.cpp` (`operand_to_str` L117,
//!   `operand_array_to_str` L170, `dbg_fmt_to_str` L222, `dbg_fmt_print` L282,
//!   `DbgInstructionToStr` L397, `DbgDump` L410, `ReadBlock` L474,
//!   `ReadIntructions` L509)
//!
//! C++ `type` fields are `type_` in Rust (`type` is a keyword). Kyty hard-EXIT
//! assertions in the debug printers are replaced by graceful `"???"` output —
//! library code must never panic on arbitrary decoded data.

use std::fmt::Write as _;

/// Kyty: Shader.h `ShaderType` (L24).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ShaderType {
    #[default]
    Unknown,
    Vertex,
    Pixel,
    Fetch,
    Compute,
}

/// Kyty: Shader.h `ShaderInstructionType` (L33-233). Complete list — the
/// SPIR-V recompiler batch dispatches on these even where the parser cannot
/// reach them yet.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShaderInstructionType {
    #[default]
    Unknown,

    BufferLoadDword,
    BufferLoadDwordX4,
    BufferLoadFormatX,
    BufferLoadFormatXy,
    BufferLoadFormatXyz,
    BufferLoadFormatXyzw,
    BufferStoreDword,
    BufferStoreFormatX,
    BufferStoreFormatXy,
    DsAppend,
    DsConsume,
    Exp,
    ImageLoad,
    ImageSample,
    ImageSampleLz,
    ImageSampleLzO,
    ImageStore,
    ImageStoreMip,
    SAddcU32,
    SAddI32,
    SAddU32,
    SAndB32,
    SAndB64,
    SAndn2B64,
    SAndSaveexecB64,
    SBfeU32,
    SBfeU64,
    SBfmB32,
    SBranch,
    SBufferLoadDword,
    SBufferLoadDwordx16,
    SBufferLoadDwordx2,
    SBufferLoadDwordx4,
    SBufferLoadDwordx8,
    SCbranchExecz,
    SCbranchScc0,
    SCbranchScc1,
    SCbranchVccz,
    SCbranchVccnz,
    SCmpEqI32,
    SCmpEqU32,
    SCmpGeI32,
    SCmpGeU32,
    SCmpGtI32,
    SCmpGtU32,
    SCmpLeI32,
    SCmpLeU32,
    SCmpLgI32,
    SCmpLgU32,
    SCmpLtI32,
    SCmpLtU32,
    SCselectB32,
    SCselectB64,
    SEndpgm,
    SInstPrefetch,
    SLoadDword,
    SLoadDwordx2,
    SLoadDwordx4,
    SLoadDwordx8,
    SLoadDwordx16,
    SLshl4AddU32,
    SLshlB32,
    SLshrB32,
    SLshlB64,
    SLshrB64,
    SMovB32,
    SMovB64,
    SMovkI32,
    SMulHiU32,
    SMulI32,
    SMulkI32,
    SNandB64,
    SNop,
    SNorB64,
    SOrB32,
    SOrB64,
    SOrn2B64,
    SSendmsg,
    SSetpcB64,
    SSwappcB64,
    SSubI32,
    SSubU32,
    SWaitcnt,
    SWqmB64,
    SXnorB64,
    SXorB64,
    TBufferLoadFormatX,
    TBufferLoadFormatXyzw,
    VAddF32,
    VAddI32,
    /// RDNA2 (`next_gen`) VOP2 0x25: carry-less `vdst = vsrc0 + vsrc1`
    /// (replaces GCN's carry-writing v_add_i32 in the same encoding slot).
    VAddNcU32,
    VAndB32,
    VAshrI32,
    VAshrrevI32,
    VBcntU32B32,
    VBfeU32,
    VBfmB32,
    VBfrevB32,
    VCeilF32,
    VCmpEqF32,
    VCmpEqI32,
    VCmpEqU32,
    VCmpFF32,
    VCmpFI32,
    VCmpFU32,
    VCmpGeF32,
    VCmpGeI32,
    VCmpGeU32,
    VCmpGtF32,
    VCmpGtI32,
    VCmpGtU32,
    VCmpLeF32,
    VCmpLeI32,
    VCmpLeU32,
    VCmpLgF32,
    VCmpLtF32,
    VCmpLtI32,
    VCmpLtU32,
    VCmpNeI32,
    VCmpNeqF32,
    VCmpNeU32,
    VCmpNgeF32,
    VCmpNgtF32,
    VCmpNleF32,
    VCmpNlgF32,
    VCmpNltF32,
    VCmpOF32,
    VCmpTI32,
    VCmpTruF32,
    VCmpTU32,
    VCmpUF32,
    VCmpxEqU32,
    VCmpxGeU32,
    VCmpxGtF32,
    VCmpxGtU32,
    VCmpxEqI32,
    VCmpxGeI32,
    VCmpxGtI32,
    VCmpxLeI32,
    VCmpxLtF32,
    VCmpxLtI32,
    VCmpxLtU32,
    VCmpxNeI32,
    VCmpxNeqF32,
    VCmpxNeU32,
    VCndmaskB32,
    VCosF32,
    VCvtF32F16,
    VCvtF32I32,
    VCvtF32U32,
    VCvtF32Ubyte0,
    VCvtF32Ubyte1,
    VCvtF32Ubyte2,
    VCvtF32Ubyte3,
    /// VOP1 0x8: `vdst = (int)vsrc0` (float→signed int). The signed sibling
    /// of `VCvtU32F32`, measured in Minecraft's menu CS.
    VCvtI32F32,
    VCvtPkrtzF16F32,
    VCvtU32F32,
    VExpF32,
    VFloorF32,
    VFmaF32,
    VFractF32,
    VInterpMovF32,
    VInterpP1F32,
    VInterpP2F32,
    VLogF32,
    /// RDNA2 (`next_gen`) VOP3 0x346: `vdst = (vsrc0 << vsrc1[4:0]) + vsrc2`.
    /// Not in Kyty's GCN table — first RDNA2-only instruction, added for the
    /// Minecraft menu CS.
    VAndOrB32,
    VLshlAddU32,
    VLshlOrU32,
    VOr3U32,
    VLshlB32,
    VLshlrevB32,
    VLshrB32,
    VLshrrevB32,
    VMacF32,
    VMadakF32,
    VMadF32,
    VMadmkF32,
    VMadU32U24,
    VMax3F32,
    VMaxF32,
    VMbcntHiU32B32,
    VMbcntLoU32B32,
    VMed3F32,
    VMin3F32,
    VMinF32,
    VMovB32,
    VMulF32,
    VMulHiU32,
    VMulLoI32,
    VMulLoU32,
    VMulU32U24,
    VNotB32,
    VOrB32,
    VRcpF32,
    VRndneF32,
    VRsqF32,
    VSadU32,
    VSinF32,
    VSqrtF32,
    VSubF32,
    VSubI32,
    /// RDNA2 (`next_gen`) VOP2 0x26: carry-less `vdst = vsrc0 - vsrc1`.
    VSubNcU32,
    VSubrevF32,
    VSubrevI32,
    /// RDNA2 (`next_gen`) VOP2 0x27: carry-less `vdst = vsrc1 - vsrc0`.
    VSubrevNcU32,
    VTruncF32,
    VXorB32,

    FetchX,
    FetchXy,
    FetchXyz,
    FetchXyzw,

    ZMax,
}

/// Kyty: Shader.h namespace `ShaderInstructionFormat` (L235-359).
///
/// A `Format` is a u64-packed string of `FormatByte` tokens (low byte =
/// last-printed operand). It is both the disassembly spec (see
/// [`ShaderCode::dbg_instruction_to_str`]) and the recompiler dispatch key —
/// keep the packed-u64 mechanism intact.
pub mod shader_instruction_format {
    // FormatByte tokens — Kyty: Shader.h `enum FormatByte` (L237-291).
    // Kyty spells these U/N/D/../DmaskF/Gds; upper-snake per Rust const style.
    pub const U: u64 = 0;
    pub const N: u64 = 1;
    /// operand_to_str(inst.dst)
    pub const D: u64 = 2;
    /// operand_to_str(inst.dst2)
    pub const D2: u64 = 3;
    /// operand_to_str(inst.src[0])
    pub const S0: u64 = 4;
    /// operand_to_str(inst.src[1])
    pub const S1: u64 = 5;
    /// operand_to_str(inst.src[2])
    pub const S2: u64 = 6;
    /// operand_to_str(inst.src[3])
    pub const S3: u64 = 7;
    pub const DA2: u64 = 8;
    pub const DA3: u64 = 9;
    pub const DA4: u64 = 10;
    pub const DA8: u64 = 11;
    pub const DA16: u64 = 12;
    pub const D2A2: u64 = 13;
    pub const D2A3: u64 = 14;
    pub const D2A4: u64 = 15;
    pub const S0A2: u64 = 16;
    pub const S0A3: u64 = 17;
    pub const S0A4: u64 = 18;
    pub const S1A2: u64 = 19;
    pub const S1A3: u64 = 20;
    pub const S1A4: u64 = 21;
    pub const S1A8: u64 = 22;
    pub const S2A2: u64 = 23;
    pub const S2A3: u64 = 24;
    pub const S2A4: u64 = 25;
    /// attr%u.%u <- inst.src[1].constant.u, inst.src[2].constant.u
    pub const ATTR: u64 = 26;
    pub const IDXEN: u64 = 27;
    pub const OFFEN: u64 = 28;
    pub const FLOAT1: u64 = 29;
    pub const FLOAT4: u64 = 30;
    pub const POS0: u64 = 31;
    pub const DONE: u64 = 32;
    pub const PARAM0: u64 = 33;
    pub const PARAM1: u64 = 34;
    pub const PARAM2: u64 = 35;
    pub const PARAM3: u64 = 36;
    pub const PARAM4: u64 = 37;
    pub const MRT0: u64 = 38;
    pub const PRIM: u64 = 39;
    pub const OFF: u64 = 40;
    pub const COMPR: u64 = 41;
    pub const VM: u64 = 42;
    /// label_%u
    pub const L: u64 = 43;
    pub const DMASK_F: u64 = 44;
    pub const DMASK_7: u64 = 45;
    pub const DMASK_1: u64 = 46;
    pub const DMASK_8: u64 = 47;
    pub const DMASK_3: u64 = 48;
    pub const DMASK_5: u64 = 49;
    pub const DMASK_9: u64 = 50;
    pub const GDS: u64 = 51;

    /// Kyty: Shader.h `FormatDefine` (L293). Packs FormatByte tokens into a
    /// u64, first token in the highest-used byte.
    #[must_use]
    pub const fn format_define(f: &[u64]) -> u64 {
        let mut r: u64 = 0;
        let mut i = 0;
        while i < f.len() {
            r = (r << 8) | f[i];
            i += 1;
        }
        r
    }

    /// Kyty: Shader.h `enum Format` (L303-357).
    #[repr(u64)]
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
    pub enum Format {
        #[default]
        Unknown = format_define(&[U]),
        Empty = format_define(&[N]),
        Imm = format_define(&[S0]),
        Label = format_define(&[L]),
        Mrt0OffOffComprVmDone = format_define(&[MRT0, OFF, OFF, COMPR, VM, DONE]),
        Mrt0Vsrc0Vsrc1ComprVmDone = format_define(&[MRT0, S0, S1, COMPR, VM, DONE]),
        Mrt0Vsrc0Vsrc1Vsrc2Vsrc3VmDone = format_define(&[MRT0, S0, S1, S2, S3, VM, DONE]),
        Param0Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM0, S0, S1, S2, S3]),
        Param1Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM1, S0, S1, S2, S3]),
        Param2Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM2, S0, S1, S2, S3]),
        Param3Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM3, S0, S1, S2, S3]),
        Param4Vsrc0Vsrc1Vsrc2Vsrc3 = format_define(&[PARAM4, S0, S1, S2, S3]),
        Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done = format_define(&[POS0, S0, S1, S2, S3, DONE]),
        PrimVsrc0OffOffOffDone = format_define(&[PRIM, S0, OFF, OFF, OFF, DONE]),
        Saddr = format_define(&[S0A2]),
        SdstSbaseSoffset = format_define(&[D, S0A2, S1]),
        Sdst16SvSoffset = format_define(&[DA16, S0A4, S1]),
        Sdst2Ssrc02 = format_define(&[DA2, S0A2]),
        Sdst2Ssrc02Ssrc1 = format_define(&[DA2, S0A2, S1]),
        Sdst2Ssrc02Ssrc12 = format_define(&[DA2, S0A2, S1A2]),
        Sdst2SvSoffset = format_define(&[DA2, S0A4, S1]),
        Sdst4SbaseSoffset = format_define(&[DA4, S0A2, S1]),
        Sdst4SvSoffset = format_define(&[DA4, S0A4, S1]),
        Sdst8SbaseSoffset = format_define(&[DA8, S0A2, S1]),
        Sdst8SvSoffset = format_define(&[DA8, S0A4, S1]),
        SdstSvSoffset = format_define(&[D, S0A4, S1]),
        SmaskVsrc0Vsrc1 = format_define(&[DA2, S0, S1]),
        Ssrc0Ssrc1 = format_define(&[S0, S1]),
        SVdstSVsrc0 = format_define(&[D, S0]),
        SVdstSVsrc0SVsrc1 = format_define(&[D, S0, S1]),
        Vdata1Vaddr3StSsDmask1 = format_define(&[D, S0A3, S1A8, S2A4, DMASK_1]),
        Vdata1Vaddr3StSsDmask8 = format_define(&[D, S0A3, S1A8, S2A4, DMASK_8]),
        Vdata1VaddrSvSoffsIdxen = format_define(&[D, S0, S1A4, S2, IDXEN]),
        Vdata1VaddrSvSoffsIdxenFloat1 = format_define(&[D, S0, S1A4, S2, IDXEN, FLOAT1]),
        Vdata2Vaddr3StSsDmask3 = format_define(&[DA2, S0A3, S1A8, S2A4, DMASK_3]),
        Vdata2Vaddr3StSsDmask5 = format_define(&[DA2, S0A3, S1A8, S2A4, DMASK_5]),
        Vdata2Vaddr3StSsDmask9 = format_define(&[DA2, S0A3, S1A8, S2A4, DMASK_9]),
        Vdata2VaddrSvSoffsIdxen = format_define(&[DA2, S0, S1A4, S2, IDXEN]),
        Vdata3Vaddr3StSsDmask7 = format_define(&[DA3, S0A3, S1A8, S2A4, DMASK_7]),
        Vdata3Vaddr4StSsDmask7 = format_define(&[DA3, S0A4, S1A8, S2A4, DMASK_7]),
        Vdata3VaddrSvSoffsIdxen = format_define(&[DA3, S0, S1A4, S2, IDXEN]),
        Vdata4Vaddr2SvSoffsOffenIdxen =
            format_define(&[DA4, S0A2, S1A4, S2, OFFEN, IDXEN]),
        Vdata4Vaddr2SvSoffsOffenIdxenFloat4 =
            format_define(&[DA4, S0A2, S1A4, S2, OFFEN, IDXEN, FLOAT4]),
        Vdata4Vaddr3StDmaskF = format_define(&[DA4, S0A3, S1A8, DMASK_F]),
        Vdata4Vaddr3StSsDmaskF = format_define(&[DA4, S0A3, S1A8, S2A4, DMASK_F]),
        Vdata4Vaddr4StDmaskF = format_define(&[DA4, S0A4, S1A8, DMASK_F]),
        Vdata4VaddrSvSoffsIdxen = format_define(&[DA4, S0, S1A4, S2, IDXEN]),
        Vdata4VaddrSvSoffsIdxenFloat4 = format_define(&[DA4, S0, S1A4, S2, IDXEN, FLOAT4]),
        VdstGds = format_define(&[D, GDS]),
        VdstSdst2Vsrc0Vsrc1 = format_define(&[D, D2A2, S0, S1]),
        VdstVsrc0Vsrc1Smask2 = format_define(&[D, S0, S1, S2A2]),
        VdstVsrc0Vsrc1Vsrc2 = format_define(&[D, S0, S1, S2]),
        VdstVsrcAttrChan = format_define(&[D, S0, ATTR]),
    }
}

/// Kyty: Shader.h `ShaderInstructionTypeFormat` (L361).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderInstructionTypeFormat {
    pub type_: ShaderInstructionType,
    pub format: shader_instruction_format::Format,
}

/// Kyty: Shader.h `ShaderOperandType` (L367).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShaderOperandType {
    #[default]
    Unknown,
    LiteralConstant,
    IntegerInlineConstant,
    FloatInlineConstant,
    VccLo,
    VccHi,
    ExecLo,
    ExecHi,
    ExecZ,
    Scc,
    Vgpr,
    Sgpr,
    M0,
    Null,
}

/// Kyty: Shader.h `ShaderConstant` union (L385). Rust stores the raw 32 bits
/// (`u`) and reinterprets on access instead of a C union.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct ShaderConstant {
    pub u: u32,
}

impl ShaderConstant {
    #[must_use]
    pub const fn from_u(u: u32) -> Self {
        Self { u }
    }

    #[must_use]
    pub const fn from_i(i: i32) -> Self {
        Self { u: i as u32 }
    }

    #[must_use]
    pub const fn from_f(f: f32) -> Self {
        Self { u: f.to_bits() }
    }

    /// The union's `.i` view.
    #[must_use]
    pub const fn i(self) -> i32 {
        self.u as i32
    }

    /// The union's `.f` view.
    #[must_use]
    pub const fn f(self) -> f32 {
        f32::from_bits(self.u)
    }
}

impl std::fmt::Debug for ShaderConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:08x}", self.u)
    }
}

/// Kyty: Shader.h `ShaderOperand` (L392).
#[derive(Copy, Clone, Debug)]
pub struct ShaderOperand {
    pub type_: ShaderOperandType,
    pub constant: ShaderConstant,
    pub register_id: i32,
    pub size: i32,
    pub multiplier: f32,
    pub absolute: bool,
    pub negate: bool,
    pub clamp: bool,
}

impl Default for ShaderOperand {
    fn default() -> Self {
        Self {
            type_: ShaderOperandType::Unknown,
            constant: ShaderConstant::default(),
            register_id: 0,
            size: 0,
            multiplier: 1.0,
            absolute: false,
            negate: false,
            clamp: false,
        }
    }
}

/// Kyty equality (Shader.h L403) deliberately ignores the modifiers
/// (`multiplier`/`absolute`/`negate`/`clamp`).
impl PartialEq for ShaderOperand {
    fn eq(&self, other: &Self) -> bool {
        self.type_ == other.type_
            && self.constant.u == other.constant.u
            && self.register_id == other.register_id
            && self.size == other.size
    }
}

/// Kyty: Shader.h `ShaderInstruction` (L409).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ShaderInstruction {
    pub pc: u32,
    pub type_: ShaderInstructionType,
    pub format: shader_instruction_format::Format,
    pub src: [ShaderOperand; 4],
    pub src_num: i32,
    pub dst: ShaderOperand,
    pub dst2: ShaderOperand,
    /// EXP channel-enable mask (`en`): which of the four `vsrc` channels this
    /// export actually writes. Only meaningful for `type_ == Exp`; a full
    /// export is `0xf`, a partial one (e.g. a `vec2` texcoord) `0x3`. Set by
    /// `shader_parse_exp`; the recompiler writes 0 to the disabled channels.
    pub export_enable: u32,
}

/// Kyty: Shader.h `ShaderLabel` (L420). `dst = pc + 4 + src[0].constant.i`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ShaderLabel {
    dst: u32,
    src: u32,
}

impl ShaderLabel {
    #[must_use]
    pub const fn new(dst: u32, src: u32) -> Self {
        Self { dst, src }
    }

    /// Kyty: `ShaderLabel(const ShaderInstruction&)` (Shader.h L424).
    #[must_use]
    pub fn from_instruction(inst: &ShaderInstruction) -> Self {
        Self {
            dst: inst
                .pc
                .wrapping_add(4)
                .wrapping_add(inst.src[0].constant.i() as u32),
            src: inst.pc,
        }
    }

    #[must_use]
    pub const fn get_dst(&self) -> u32 {
        self.dst
    }

    #[must_use]
    pub const fn get_src(&self) -> u32 {
        self.src
    }

    pub fn disable(&mut self) {
        self.dst = 0;
        self.src = 0;
    }

    #[must_use]
    pub const fn is_disabled(&self) -> bool {
        self.dst == 0 && self.src == 0
    }
}

/// Kyty: `ShaderLabel::ToString()` (Shader.h L431) — exposed as `Display`
/// (and thereby `.to_string()`) in Rust.
impl std::fmt::Display for ShaderLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "label_{:04x}_{:04x}", self.dst, self.src)
    }
}

/// Kyty: Shader.h `ShaderDebugPrintf::Type` (L448).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShaderDebugPrintfType {
    Uint,
    Int,
    Float,
}

/// Kyty: Shader.h `ShaderDebugPrintf` (L446) — a debug-printf command
/// injected at `pc`. The data model is ported; the global injection registry
/// (`g_debug_printfs`, Shader.cpp L100/L3006) is not — see `analysis.rs`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShaderDebugPrintf {
    pub pc: u32,
    pub format: String,
    pub types: Vec<ShaderDebugPrintfType>,
    pub args: Vec<ShaderOperand>,
}

/// Kyty: Shader.h `ShaderControlFlowBlock` (L460).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ShaderControlFlowBlock {
    pub pc: u32,
    pub is_discard: bool,
    pub is_valid: bool,
    pub last: ShaderInstruction,
}

/// Kyty: Shader.cpp `operand_to_str` (L117). Kyty EXITs on inconsistent
/// sizes/modifiers; the port renders what it can instead.
fn operand_to_str(op: &ShaderOperand) -> String {
    use ShaderOperandType as O;
    match op.type_ {
        O::LiteralConstant => return format!("{:.6} ({})", op.constant.f(), op.constant.u),
        O::IntegerInlineConstant => return format!("{}", op.constant.i()),
        O::FloatInlineConstant => return format!("{:.6}", op.constant.f()),
        _ => {}
    }

    let mut ret = match op.type_ {
        O::VccHi => "vcc_hi".to_string(),
        O::VccLo => "vcc_lo".to_string(),
        O::ExecHi => "exec_hi".to_string(),
        O::ExecLo => "exec_lo".to_string(),
        O::ExecZ => "execz".to_string(),
        O::Scc => "scc".to_string(),
        O::M0 => "m0".to_string(),
        O::Vgpr => format!("v{}", op.register_id),
        O::Sgpr => format!("s{}", op.register_id),
        O::Null => "null".to_string(),
        _ => "???".to_string(),
    };

    if op.absolute {
        ret = format!("abs({ret})");
    }
    if op.negate {
        return format!("-{ret}");
    }
    ret
}

/// Kyty: Shader.cpp `operand_array_to_str` (L170).
fn operand_array_to_str(op: &ShaderOperand, n: i32) -> String {
    use ShaderOperandType as O;
    let mut ret = match op.type_ {
        O::VccLo if n == 2 => "vcc".to_string(),
        O::ExecLo if n == 2 => "exec".to_string(),
        O::Sgpr => format!("s[{}:{}]", op.register_id, op.register_id + n - 1),
        O::Vgpr => format!("v[{}:{}]", op.register_id, op.register_id + n - 1),
        O::LiteralConstant if n == 2 => format!("{:.6} ({})", op.constant.f(), op.constant.u),
        O::IntegerInlineConstant if n == 2 => format!("{}", op.constant.i()),
        _ => "???".to_string(),
    };

    if op.absolute {
        ret = format!("abs({ret})");
    }
    if op.negate {
        return format!("-{ret}");
    }
    ret
}

/// Kyty: Shader.cpp `dbg_fmt_print` (L282). Walks the packed Format bytes,
/// low byte first, prepending — so the rendered operand order matches the
/// token order in `FormatDefine`.
fn dbg_fmt_print(inst: &ShaderInstruction) -> String {
    use shader_instruction_format as sif;
    use shader_instruction_format::Format;

    let mut f = inst.format as u64;
    if inst.format == Format::Unknown || inst.format == Format::Empty {
        return String::new();
    }
    let mut str = String::new();
    loop {
        let fu = f & 0xff;
        if fu == 0 {
            break;
        }
        let s = match fu {
            sif::D => operand_to_str(&inst.dst),
            sif::D2 => operand_to_str(&inst.dst2),
            sif::S0 => operand_to_str(&inst.src[0]),
            sif::S1 => operand_to_str(&inst.src[1]),
            sif::S2 => operand_to_str(&inst.src[2]),
            sif::S3 => operand_to_str(&inst.src[3]),
            sif::DA2 => operand_array_to_str(&inst.dst, 2),
            sif::DA3 => operand_array_to_str(&inst.dst, 3),
            sif::DA4 => operand_array_to_str(&inst.dst, 4),
            sif::DA8 => operand_array_to_str(&inst.dst, 8),
            sif::DA16 => operand_array_to_str(&inst.dst, 16),
            sif::D2A2 => operand_array_to_str(&inst.dst2, 2),
            sif::D2A3 => operand_array_to_str(&inst.dst2, 3),
            sif::D2A4 => operand_array_to_str(&inst.dst2, 4),
            sif::S0A2 => operand_array_to_str(&inst.src[0], 2),
            sif::S0A3 => operand_array_to_str(&inst.src[0], 3),
            sif::S0A4 => operand_array_to_str(&inst.src[0], 4),
            sif::S1A2 => operand_array_to_str(&inst.src[1], 2),
            sif::S1A3 => operand_array_to_str(&inst.src[1], 3),
            sif::S1A4 => operand_array_to_str(&inst.src[1], 4),
            sif::S1A8 => operand_array_to_str(&inst.src[1], 8),
            sif::S2A2 => operand_array_to_str(&inst.src[2], 2),
            sif::S2A3 => operand_array_to_str(&inst.src[2], 3),
            sif::S2A4 => operand_array_to_str(&inst.src[2], 4),
            sif::ATTR => format!("attr{}.{}", inst.src[1].constant.u, inst.src[2].constant.u),
            sif::IDXEN => "idxen".to_string(),
            sif::OFFEN => "offen".to_string(),
            sif::FLOAT1 => "format:float1".to_string(),
            sif::FLOAT4 => "format:float4".to_string(),
            sif::POS0 => "pos0".to_string(),
            sif::DONE => "done".to_string(),
            sif::PARAM0 => "param0".to_string(),
            sif::PARAM1 => "param1".to_string(),
            sif::PARAM2 => "param2".to_string(),
            sif::PARAM3 => "param3".to_string(),
            sif::PARAM4 => "param4".to_string(),
            sif::MRT0 => "mrt_color0".to_string(),
            sif::PRIM => "prim".to_string(),
            sif::OFF => "off".to_string(),
            sif::COMPR => "compr".to_string(),
            sif::VM => "vm".to_string(),
            sif::L => format!(
                "label_{:04x}",
                inst.pc
                    .wrapping_add(4)
                    .wrapping_add(inst.src[0].constant.i() as u32)
            ),
            sif::DMASK_1 => "dmask:0x1".to_string(),
            sif::DMASK_8 => "dmask:0x8".to_string(),
            sif::DMASK_3 => "dmask:0x3".to_string(),
            sif::DMASK_5 => "dmask:0x5".to_string(),
            sif::DMASK_7 => "dmask:0x7".to_string(),
            sif::DMASK_9 => "dmask:0x9".to_string(),
            sif::DMASK_F => "dmask:0xf".to_string(),
            sif::GDS => "gds".to_string(),
            _ => "???".to_string(),
        };
        str = if str.is_empty() {
            s
        } else {
            format!("{s}, {str}")
        };
        f >>= 8;
    }
    if inst.dst.multiplier == 2.0 {
        str += " mul:2";
    }
    if inst.dst.multiplier == 4.0 {
        str += " mul:4";
    }
    if inst.dst.multiplier == 0.5 {
        str += " div:2";
    }
    if inst.dst.clamp {
        str += " clamp";
    }
    str
}

/// Kyty: Shader.cpp `IsDiscardInstruction` (L428).
fn is_discard_instruction(code: &[ShaderInstruction], index: usize) -> bool {
    use ShaderInstructionType as T;
    use shader_instruction_format::Format;
    if index == 0 || index + 1 >= code.len() {
        return false;
    }
    let prev_inst = &code[index - 1];
    let inst = &code[index];
    let next_inst = &code[index + 1];

    inst.type_ == T::Exp
        && inst.format == Format::Mrt0OffOffComprVmDone
        && prev_inst.type_ == T::SMovB64
        && prev_inst.format == Format::Sdst2Ssrc02
        && prev_inst.dst.type_ == ShaderOperandType::ExecLo
        && prev_inst.src[0].type_ == ShaderOperandType::IntegerInlineConstant
        && prev_inst.src[0].constant.i() == 0
        && next_inst.type_ == T::SEndpgm
}

/// Kyty: Shader.h `ShaderCode` (L468).
#[derive(Clone, Debug)]
pub struct ShaderCode {
    hash0: u32,
    crc32: u32,
    instructions: Vec<ShaderInstruction>,
    labels: Vec<ShaderLabel>,
    indirect_labels: Vec<ShaderLabel>,
    type_: ShaderType,
    debug_printfs: Vec<ShaderDebugPrintf>,
    vs_embedded_id: u32,
    ps_embedded_id: u32,
    vs_embedded: bool,
    ps_embedded: bool,
}

impl Default for ShaderCode {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderCode {
    /// Kyty ctor pre-expands the instruction vector to 128 entries.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hash0: 0,
            crc32: 0,
            instructions: Vec::with_capacity(128),
            labels: Vec::new(),
            indirect_labels: Vec::new(),
            type_: ShaderType::Unknown,
            debug_printfs: Vec::new(),
            vs_embedded_id: 0,
            ps_embedded_id: 0,
            vs_embedded: false,
            ps_embedded: false,
        }
    }

    #[must_use]
    pub fn get_instructions(&self) -> &Vec<ShaderInstruction> {
        &self.instructions
    }

    pub fn get_instructions_mut(&mut self) -> &mut Vec<ShaderInstruction> {
        &mut self.instructions
    }

    #[must_use]
    pub fn get_labels(&self) -> &Vec<ShaderLabel> {
        &self.labels
    }

    pub fn get_labels_mut(&mut self) -> &mut Vec<ShaderLabel> {
        &mut self.labels
    }

    #[must_use]
    pub fn get_indirect_labels(&self) -> &Vec<ShaderLabel> {
        &self.indirect_labels
    }

    pub fn get_indirect_labels_mut(&mut self) -> &mut Vec<ShaderLabel> {
        &mut self.indirect_labels
    }

    #[must_use]
    pub const fn get_type(&self) -> ShaderType {
        self.type_
    }

    pub fn set_type(&mut self, type_: ShaderType) {
        self.type_ = type_;
    }

    /// Kyty: `GetDebugPrintfs` (Shader.h L487).
    #[must_use]
    pub fn get_debug_printfs(&self) -> &Vec<ShaderDebugPrintf> {
        &self.debug_printfs
    }

    pub fn get_debug_printfs_mut(&mut self) -> &mut Vec<ShaderDebugPrintf> {
        &mut self.debug_printfs
    }

    /// Kyty: `HasAnyOf` (Shader.h L491).
    #[must_use]
    pub fn has_any_of(&self, types: &[ShaderInstructionType]) -> bool {
        types
            .iter()
            .any(|t| self.instructions.iter().any(|inst| inst.type_ == *t))
    }

    #[must_use]
    pub const fn is_vs_embedded(&self) -> bool {
        self.vs_embedded
    }

    pub fn set_vs_embedded(&mut self, embedded: bool) {
        self.vs_embedded = embedded;
    }

    #[must_use]
    pub const fn get_vs_embedded_id(&self) -> u32 {
        self.vs_embedded_id
    }

    pub fn set_vs_embedded_id(&mut self, embedded_id: u32) {
        self.vs_embedded_id = embedded_id;
    }

    #[must_use]
    pub const fn is_ps_embedded(&self) -> bool {
        self.ps_embedded
    }

    pub fn set_ps_embedded(&mut self, embedded: bool) {
        self.ps_embedded = embedded;
    }

    #[must_use]
    pub const fn get_ps_embedded_id(&self) -> u32 {
        self.ps_embedded_id
    }

    pub fn set_ps_embedded_id(&mut self, embedded_id: u32) {
        self.ps_embedded_id = embedded_id;
    }

    #[must_use]
    pub const fn get_crc32(&self) -> u32 {
        self.crc32
    }

    pub fn set_crc32(&mut self, c: u32) {
        self.crc32 = c;
    }

    #[must_use]
    pub const fn get_hash0(&self) -> u32 {
        self.hash0
    }

    pub fn set_hash0(&mut self, h: u32) {
        self.hash0 = h;
    }

    /// Kyty: Shader.cpp `DbgInstructionToStr` (L397).
    #[must_use]
    pub fn dbg_instruction_to_str(inst: &ShaderInstruction) -> String {
        let name = format!("{:?}", inst.type_);
        let format = format!("{:?}", inst.format);
        format!("{name:<20} [{format:<30}] {}", dbg_fmt_print(inst))
    }

    /// Kyty: Shader.cpp `DbgDump` (L410).
    #[must_use]
    pub fn dbg_dump(&self) -> String {
        let mut ret = String::new();
        for inst in &self.instructions {
            if self
                .labels
                .iter()
                .any(|label| !label.is_disabled() && label.get_dst() == inst.pc)
            {
                let _ = write!(ret, "\nlabel_{:04x}:\n", inst.pc);
            }
            if self
                .indirect_labels
                .iter()
                .any(|label| !label.is_disabled() && label.get_dst() == inst.pc)
            {
                ret.push('\n');
            }
            let _ = writeln!(ret, "  {}", Self::dbg_instruction_to_str(inst));
        }
        ret
    }

    /// Kyty: Shader.cpp `ReadBlock` (L474).
    #[must_use]
    pub fn read_block(&self, pc: u32) -> ShaderControlFlowBlock {
        use ShaderInstructionType as T;
        let mut ret = ShaderControlFlowBlock::default();
        if let Some(index) = self.instructions.iter().position(|inst| inst.pc == pc) {
            ret.pc = pc;
            ret.is_valid = true;
            for i in index..self.instructions.len() {
                let inst = &self.instructions[i];
                if matches!(
                    inst.type_,
                    T::SEndpgm
                        | T::SCbranchExecz
                        | T::SCbranchScc0
                        | T::SCbranchScc1
                        | T::SCbranchVccz
                        | T::SCbranchVccnz
                        | T::SBranch
                ) {
                    ret.last = *inst;
                    break;
                }
                if is_discard_instruction(&self.instructions, i) {
                    ret.is_discard = true;
                }
            }
        }
        ret
    }

    /// Kyty: Shader.cpp `ReadIntructions` (L509) — Kyty's spelling (sic).
    #[must_use]
    pub fn read_intructions(&self, block: &ShaderControlFlowBlock) -> Vec<ShaderInstruction> {
        let mut ret = Vec::new();
        if let Some(index) = self
            .instructions
            .iter()
            .position(|inst| inst.pc == block.pc)
        {
            for inst in &self.instructions[index..] {
                ret.push(*inst);
                if inst.pc == block.last.pc {
                    break;
                }
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::shader_instruction_format::{Format, format_define};
    use super::*;

    #[test]
    fn format_define_packs_bytes_first_token_highest() {
        // Kyty: Shader.h FormatDefine (L293).
        use super::shader_instruction_format as sif;
        assert_eq!(format_define(&[sif::U]), 0);
        assert_eq!(format_define(&[sif::N]), 1);
        assert_eq!(format_define(&[sif::D, sif::S0]), (sif::D << 8) | sif::S0);
        assert_eq!(Format::SVdstSVsrc0 as u64, 0x0204);
        assert_eq!(Format::Label as u64, sif::L);
        assert_eq!(
            Format::Mrt0OffOffComprVmDone as u64,
            (sif::MRT0 << 40)
                | (sif::OFF << 32)
                | (sif::OFF << 24)
                | (sif::COMPR << 16)
                | (sif::VM << 8)
                | sif::DONE
        );
    }

    #[test]
    fn shader_operand_equality_ignores_modifiers() {
        // Kyty: Shader.h ShaderOperand::operator== (L403).
        let a = ShaderOperand {
            type_: ShaderOperandType::Vgpr,
            register_id: 3,
            size: 1,
            ..Default::default()
        };
        let mut b = a;
        b.negate = true;
        b.absolute = true;
        b.multiplier = 4.0;
        b.clamp = true;
        assert_eq!(a, b);
        b.register_id = 4;
        assert_ne!(a, b);
    }

    #[test]
    fn shader_label_from_instruction() {
        // Kyty: Shader.h ShaderLabel(const ShaderInstruction&) (L424).
        let mut inst = ShaderInstruction {
            pc: 8,
            ..Default::default()
        };
        inst.src[0].type_ = ShaderOperandType::LiteralConstant;
        inst.src[0].constant = ShaderConstant::from_i(-12);
        let label = ShaderLabel::from_instruction(&inst);
        assert_eq!(label.get_dst(), 0);
        assert_eq!(label.get_src(), 8);
        assert_eq!(label.to_string(), "label_0000_0008");
        // IsDisabled needs dst == 0 AND src == 0 (Shader.h L439); src is 8.
        assert!(!label.is_disabled());
    }

    #[test]
    fn shader_label_disable() {
        let mut label = ShaderLabel::new(0x10, 0x4);
        assert!(!label.is_disabled());
        label.disable();
        assert!(label.is_disabled());
    }

    #[test]
    fn dbg_instruction_to_str_s_mov() {
        let mut inst = ShaderInstruction {
            type_: ShaderInstructionType::SMovB32,
            format: Format::SVdstSVsrc0,
            src_num: 1,
            ..Default::default()
        };
        inst.dst.type_ = ShaderOperandType::Sgpr;
        inst.dst.register_id = 0;
        inst.dst.size = 1;
        inst.src[0].type_ = ShaderOperandType::Sgpr;
        inst.src[0].register_id = 1;
        inst.src[0].size = 1;
        let s = ShaderCode::dbg_instruction_to_str(&inst);
        assert!(s.contains("SMovB32"), "{s}");
        assert!(s.contains("[SVdstSVsrc0"), "{s}");
        assert!(s.ends_with("s0, s1"), "{s}");
    }

    #[test]
    fn dbg_operand_modifiers() {
        let mut inst = ShaderInstruction {
            type_: ShaderInstructionType::VAddF32,
            format: Format::SVdstSVsrc0SVsrc1,
            src_num: 2,
            ..Default::default()
        };
        inst.dst.type_ = ShaderOperandType::Vgpr;
        inst.dst.size = 1;
        inst.dst.multiplier = 2.0;
        inst.dst.clamp = true;
        inst.src[0].type_ = ShaderOperandType::Vgpr;
        inst.src[0].register_id = 1;
        inst.src[0].size = 1;
        inst.src[0].absolute = true;
        inst.src[1].type_ = ShaderOperandType::Vgpr;
        inst.src[1].register_id = 2;
        inst.src[1].size = 1;
        inst.src[1].negate = true;
        let s = ShaderCode::dbg_instruction_to_str(&inst);
        assert!(s.contains("v0, abs(v1), -v2"), "{s}");
        assert!(s.contains(" mul:2"), "{s}");
        assert!(s.contains(" clamp"), "{s}");
    }
}
