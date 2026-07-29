//! Pure-Rust SPIR-V assembly-text assembler for the Kyty shader recompiler.
//!
//! Kyty's `ShaderSpirv.cpp` generates SPIR-V *assembly text* and assembles it
//! with SPIRV-Tools (`SpirvRun`, Shader.cpp L845: Assemble → Validate →
//! Optimize, targeting `SPV_ENV_VULKAN_1_2`, Shader.cpp L850/L811). This
//! module is the Assemble step, clean-room re-implemented in Rust (SPIRV-Tools
//! is *not* vendored). The opcode / enum-token vocabulary is exactly what
//! Kyty's templates emit (extracted from `ShaderSpirv.cpp`), plus the obvious
//! structural opcodes (`OpSource`, `OpName`, `OpSpecConstant*`, …).
//!
//! Semantics follow `spirv-as` for this subset:
//! - one instruction per line; `;` starts a comment; blank lines ignored;
//! - `%name` ids are symbolic and assigned sequential numeric ids in
//!   first-appearance order; a purely numeric `%42` binds to that exact id
//!   (Kyty mixes both, e.g. `%4 = OpFunction …` next to `%void`), and
//!   symbolic assignment skips numbers claimed anywhere by numeric ids;
//! - forward references are allowed (`OpEntryPoint %main` before `%main`),
//!   but every referenced id must be defined somewhere in the module;
//! - `OpConstant` literals are typed by the result type (`OpTypeFloat 32` /
//!   `OpTypeInt 32 …`); Kyty only emits 32-bit scalars (ShaderSpirv.cpp
//!   L6527-L6535), wider constants are rejected;
//! - `0x…` literals are raw bit patterns (Kyty emits `0x%08x` for uints);
//! - strings are double-quoted; a backslash escapes the next character;
//! - bitmask tokens may be combined with `|` (Kyty only emits single tokens:
//!   `None`, `Lod`);
//! - `OpExtInst` accepts an extended-instruction *name* (resolved through the
//!   set imported by `OpExtInstImport`, e.g. `FMin` in `GLSL.std.450`) or a
//!   raw number (Kyty emits `%NonSemantic_DebugPrintf 1`, ShaderSpirv.cpp
//!   L6167).
//!
//! Header: magic `0x07230203`, version 1.5 (`SPV_ENV_VULKAN_1_2` assembles
//! to SPIR-V 1.5), generator 0, bound = max id + 1, schema 0.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// SPIR-V binary magic number.
pub const SPIRV_MAGIC: u32 = 0x0723_0203;
/// SPIR-V version 1.5 — Kyty assembles for `SPV_ENV_VULKAN_1_2`
/// (Shader.cpp L850), which targets SPIR-V 1.5.
pub const SPIRV_VERSION: u32 = 0x0001_0500;

/// Assembly error with a 1-based source line number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsmError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for AsmError {}

fn err(line: usize, message: impl Into<String>) -> AsmError {
    AsmError {
        line,
        message: message.into(),
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    /// `%name` (without the `%`).
    Id(String),
    /// `"…"` with escapes resolved.
    Str(String),
    /// Bare word: opcode, enum token, or literal.
    Word(String),
    /// `=`
    Eq,
}

fn tokenize_line(line_no: usize, s: &str) -> Result<Vec<Tok>, AsmError> {
    let chars: Vec<char> = s.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == ';' {
            break;
        } else if c == '"' {
            i += 1;
            let mut out = String::new();
            let mut closed = false;
            while i < chars.len() {
                match chars[i] {
                    '\\' => {
                        i += 1;
                        if i >= chars.len() {
                            return Err(err(line_no, "unterminated escape in string literal"));
                        }
                        out.push(chars[i]);
                        i += 1;
                    }
                    '"' => {
                        closed = true;
                        i += 1;
                        break;
                    }
                    ch => {
                        out.push(ch);
                        i += 1;
                    }
                }
            }
            if !closed {
                return Err(err(line_no, "unterminated string literal"));
            }
            toks.push(Tok::Str(out));
        } else if c == '%' {
            i += 1;
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            if i == start {
                return Err(err(line_no, "expected id name after '%'"));
            }
            toks.push(Tok::Id(chars[start..i].iter().collect()));
        } else if c == '=' {
            toks.push(Tok::Eq);
            i += 1;
        } else {
            let start = i;
            while i < chars.len()
                && !chars[i].is_whitespace()
                && !matches!(chars[i], ';' | '"' | '=' | '%')
            {
                i += 1;
            }
            toks.push(Tok::Word(chars[start..i].iter().collect()));
        }
    }
    Ok(toks)
}

#[derive(Debug)]
struct RawInst {
    line: usize,
    result: Option<String>,
    opcode: String,
    args: Vec<Tok>,
}

fn parse_inst(line: usize, toks: Vec<Tok>) -> Result<Option<RawInst>, AsmError> {
    if toks.is_empty() {
        return Ok(None);
    }
    let (result, rest) = match toks.as_slice() {
        [Tok::Id(r), Tok::Eq, rest @ ..] => (Some(r.clone()), rest),
        rest => (None, rest),
    };
    match rest {
        [Tok::Word(op), args @ ..] if op.starts_with("Op") => Ok(Some(RawInst {
            line,
            result,
            opcode: op.clone(),
            args: args.to_vec(),
        })),
        _ => Err(err(
            line,
            "malformed instruction: expected `[%result =] OpXxx …`",
        )),
    }
}

// ---------------------------------------------------------------------------
// Opcode / operand-kind tables
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EKind {
    Capability,
    ExecutionModel,
    AddressingModel,
    MemoryModel,
    ExecutionMode,
    StorageClass,
    Dim,
    ImageFormat,
    Decoration,
    BuiltIn,
    SelectionControl,
    LoopControl,
    FunctionControl,
    ImageOperands,
    MemoryAccess,
    SourceLanguage,
}

fn ekind_name(kind: EKind) -> &'static str {
    match kind {
        EKind::Capability => "Capability",
        EKind::ExecutionModel => "ExecutionModel",
        EKind::AddressingModel => "AddressingModel",
        EKind::MemoryModel => "MemoryModel",
        EKind::ExecutionMode => "ExecutionMode",
        EKind::StorageClass => "StorageClass",
        EKind::Dim => "Dim",
        EKind::ImageFormat => "ImageFormat",
        EKind::Decoration => "Decoration",
        EKind::BuiltIn => "BuiltIn",
        EKind::SelectionControl => "SelectionControl",
        EKind::LoopControl => "LoopControl",
        EKind::FunctionControl => "FunctionControl",
        EKind::ImageOperands => "ImageOperands",
        EKind::MemoryAccess => "MemoryAccess",
        EKind::SourceLanguage => "SourceLanguage",
    }
}

/// Single-token enum value (per the SPIR-V specification).
#[allow(clippy::too_many_lines)]
fn enum_value(kind: EKind, tok: &str) -> Option<u32> {
    use EKind as E;
    Some(match (kind, tok) {
        // Capabilities used by Kyty: `OpCapability Shader` / `ImageQuery`
        // (ShaderSpirv.cpp WriteHeader, L6689 region).
        (E::Capability, "Shader") => 1,
        (E::Capability, "ImageQuery") => 50,
        (E::Capability, "GroupNonUniform") => 61,
        (E::Capability, "GroupNonUniformBallot") => 64,
        // `OpEntryPoint <Type>` where <Type> ∈ Fragment/Vertex/GLCompute.
        (E::ExecutionModel, "Vertex") => 0,
        (E::ExecutionModel, "Fragment") => 4,
        (E::ExecutionModel, "GLCompute") => 5,
        (E::AddressingModel, "Logical") => 0,
        (E::MemoryModel, "Simple") => 0,
        (E::MemoryModel, "GLSL450") => 1,
        (E::MemoryModel, "OpenCL") => 2,
        (E::MemoryModel, "Vulkan") => 3,
        (E::ExecutionMode, "OriginUpperLeft") => 7,
        (E::ExecutionMode, "OriginLowerLeft") => 8,
        (E::ExecutionMode, "EarlyFragmentTests") => 9,
        (E::ExecutionMode, "DepthReplacing") => 12,
        (E::ExecutionMode, "LocalSize") => 17,
        (E::StorageClass, "UniformConstant") => 0,
        (E::StorageClass, "Input") => 1,
        (E::StorageClass, "Uniform") => 2,
        (E::StorageClass, "Output") => 3,
        (E::StorageClass, "Workgroup") => 4,
        (E::StorageClass, "CrossWorkgroup") => 5,
        (E::StorageClass, "Private") => 6,
        (E::StorageClass, "Function") => 7,
        (E::StorageClass, "PushConstant") => 9,
        (E::StorageClass, "StorageBuffer") => 12,
        (E::Dim, "1D") => 0,
        (E::Dim, "2D") => 1,
        (E::Dim, "3D") => 2,
        (E::Dim, "Cube") => 3,
        (E::Dim, "Rect") => 4,
        (E::Dim, "Buffer") => 5,
        (E::Dim, "SubpassData") => 6,
        // Kyty emits `Unknown` and `Rgba8` (ShaderSpirv.cpp OpTypeImage
        // templates); a few nearby formats included for completeness.
        (E::ImageFormat, "Unknown") => 0,
        (E::ImageFormat, "Rgba32f") => 1,
        (E::ImageFormat, "Rgba16f") => 2,
        (E::ImageFormat, "R32f") => 3,
        (E::ImageFormat, "Rgba8") => 4,
        (E::ImageFormat, "Rgba8Snorm") => 5,
        (E::ImageFormat, "R32i") => 24,
        (E::ImageFormat, "R32ui") => 33,
        (E::Decoration, "RelaxedPrecision") => 0,
        (E::Decoration, "SpecId") => 1,
        (E::Decoration, "Block") => 2,
        (E::Decoration, "BufferBlock") => 3,
        (E::Decoration, "ArrayStride") => 6,
        (E::Decoration, "MatrixStride") => 7,
        (E::Decoration, "BuiltIn") => 11,
        (E::Decoration, "NoPerspective") => 13,
        (E::Decoration, "Flat") => 14,
        (E::Decoration, "Coherent") => 23,
        (E::Decoration, "NonWritable") => 24,
        (E::Decoration, "NonReadable") => 25,
        (E::Decoration, "Location") => 30,
        (E::Decoration, "Component") => 31,
        (E::Decoration, "Binding") => 33,
        (E::Decoration, "DescriptorSet") => 34,
        (E::Decoration, "Offset") => 35,
        (E::BuiltIn, "Position") => 0,
        (E::BuiltIn, "PointSize") => 1,
        (E::BuiltIn, "ClipDistance") => 3,
        (E::BuiltIn, "CullDistance") => 4,
        (E::BuiltIn, "FragCoord") => 15,
        (E::BuiltIn, "FrontFacing") => 17,
        (E::BuiltIn, "FragDepth") => 22,
        (E::BuiltIn, "NumWorkgroups") => 24,
        (E::BuiltIn, "WorkgroupSize") => 25,
        (E::BuiltIn, "WorkgroupId") => 26,
        (E::BuiltIn, "LocalInvocationId") => 27,
        (E::BuiltIn, "GlobalInvocationId") => 28,
        (E::BuiltIn, "LocalInvocationIndex") => 29,
        (E::BuiltIn, "VertexIndex") => 42,
        (E::BuiltIn, "InstanceIndex") => 43,
        (E::SelectionControl, "None") => 0,
        (E::SelectionControl, "Flatten") => 1,
        (E::SelectionControl, "DontFlatten") => 2,
        (E::LoopControl, "None") => 0,
        (E::LoopControl, "Unroll") => 1,
        (E::LoopControl, "DontUnroll") => 2,
        (E::FunctionControl, "None") => 0,
        (E::FunctionControl, "Inline") => 1,
        (E::FunctionControl, "DontInline") => 2,
        (E::FunctionControl, "Pure") => 4,
        (E::FunctionControl, "Const") => 8,
        (E::ImageOperands, "None") => 0,
        (E::ImageOperands, "Bias") => 0x1,
        (E::ImageOperands, "Lod") => 0x2,
        (E::ImageOperands, "Grad") => 0x4,
        (E::ImageOperands, "ConstOffset") => 0x8,
        (E::ImageOperands, "Offset") => 0x10,
        (E::ImageOperands, "ConstOffsets") => 0x20,
        (E::ImageOperands, "Sample") => 0x40,
        (E::ImageOperands, "MinLod") => 0x80,
        (E::MemoryAccess, "None") => 0,
        (E::MemoryAccess, "Volatile") => 1,
        (E::MemoryAccess, "Aligned") => 2,
        (E::MemoryAccess, "Nontemporal") => 4,
        (E::SourceLanguage, "Unknown") => 0,
        (E::SourceLanguage, "ESSL") => 1,
        (E::SourceLanguage, "GLSL") => 2,
        (E::SourceLanguage, "OpenCL_C") => 3,
        (E::SourceLanguage, "OpenCL_CPP") => 4,
        (E::SourceLanguage, "HLSL") => 5,
        _ => return None,
    })
}

/// GLSL.std.450 extended-instruction numbers for the names Kyty emits
/// (`ShaderSpirv.cpp` recompiler templates) plus the common neighbours.
fn glsl_std_450(name: &str) -> Option<u32> {
    Some(match name {
        "Round" => 1,
        "RoundEven" => 2,
        "Trunc" => 3,
        "FAbs" => 4,
        "SAbs" => 5,
        "FSign" => 6,
        "SSign" => 7,
        "Floor" => 8,
        "Ceil" => 9,
        "Fract" => 10,
        "Sin" => 13,
        "Cos" => 14,
        "Pow" => 26,
        "Exp" => 27,
        "Log" => 28,
        "Exp2" => 29,
        "Log2" => 30,
        "Sqrt" => 31,
        "InverseSqrt" => 32,
        "FMin" => 37,
        "UMin" => 38,
        "SMin" => 39,
        "FMax" => 40,
        "UMax" => 41,
        "SMax" => 42,
        "FClamp" => 43,
        "UClamp" => 44,
        "SClamp" => 45,
        "FMix" => 46,
        "Fma" => 50,
        "PackHalf2x16" => 58,
        "UnpackHalf2x16" => 62,
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// One id operand.
    Id,
    /// Zero or more trailing id operands.
    IdRest,
    /// One 32-bit integer literal.
    Lit,
    /// Zero or more trailing integer literals.
    LitRest,
    /// One string literal (nul-terminated, word-padded UTF-8).
    Str_,
    /// Optional trailing id (e.g. `OpVariable` initializer).
    OptId,
    /// Optional trailing string (e.g. `OpSource`).
    OptStr,
    /// `OpConstant` / `OpSpecConstant` value, typed by the result type.
    TypedLit,
    /// Single enum token.
    Enum(EKind),
    /// Bitmask token(s), `|`-combinable.
    Mask(EKind),
    /// Optional trailing bitmask (image operands / memory access).
    OptMask(EKind),
    /// Extended-instruction name or number (`OpExtInst`).
    ExtName,
    /// `OpSwitch` `literal %target` pairs.
    Switch,
    /// Decoration-dependent extra operands (`OpDecorate` / `OpMemberDecorate`).
    DecoExtra,
}

#[derive(Debug, Clone, Copy)]
struct OpInfo {
    code: u16,
    rt: bool,
    res: bool,
    ops: &'static [Kind],
}

const fn op(code: u16, rt: bool, res: bool, ops: &'static [Kind]) -> OpInfo {
    OpInfo { code, rt, res, ops }
}

/// All-id-operand instruction with a result type and result.
const IDR: &[Kind] = &[Kind::IdRest];

/// Whether [`assemble`] can encode `name`.
///
/// Exists so the recompiler's dispatch table can be checked against this
/// assembler in a test: a row that emits an opcode `op_info` has no row for
/// only fails when a title happens to hit that instruction, and it fails at
/// runtime as a skipped draw rather than at build time. See
/// `every_wired_template_opcode_assembles`.
#[cfg(test)]
pub(crate) fn knows_opcode(name: &str) -> bool {
    op_info(name).is_some()
}

#[allow(clippy::too_many_lines)]
fn op_info(name: &str) -> Option<OpInfo> {
    use EKind as E;
    use Kind::{
        DecoExtra, Enum, ExtName, Id, IdRest, Lit, LitRest, Mask, OptId, OptMask, OptStr, Str_,
        Switch, TypedLit,
    };
    Some(match name {
        // --- Mode setting / debug -----------------------------------------
        "OpUndef" => op(1, true, true, &[]),
        "OpSource" => op(
            3,
            false,
            false,
            &[Enum(E::SourceLanguage), Lit, OptId, OptStr],
        ),
        "OpName" => op(5, false, false, &[Id, Str_]),
        "OpMemberName" => op(6, false, false, &[Id, Lit, Str_]),
        "OpString" => op(7, false, true, &[Str_]),
        "OpExtension" => op(10, false, false, &[Str_]),
        "OpExtInstImport" => op(11, false, true, &[Str_]),
        "OpExtInst" => op(12, true, true, &[Id, ExtName, IdRest]),
        "OpMemoryModel" => op(
            14,
            false,
            false,
            &[Enum(E::AddressingModel), Enum(E::MemoryModel)],
        ),
        "OpEntryPoint" => op(
            15,
            false,
            false,
            &[Enum(E::ExecutionModel), Id, Str_, IdRest],
        ),
        "OpExecutionMode" => op(16, false, false, &[Id, Enum(E::ExecutionMode), LitRest]),
        "OpCapability" => op(17, false, false, &[Enum(E::Capability)]),
        "OpGroupNonUniformBroadcastFirst" => op(338, true, true, &[Id, Id]),
        // --- Types ---------------------------------------------------------
        "OpTypeVoid" => op(19, false, true, &[]),
        "OpTypeBool" => op(20, false, true, &[]),
        "OpTypeInt" => op(21, false, true, &[Lit, Lit]),
        "OpTypeFloat" => op(22, false, true, &[Lit]),
        "OpTypeVector" => op(23, false, true, &[Id, Lit]),
        "OpTypeImage" => op(
            25,
            false,
            true,
            &[Id, Enum(E::Dim), Lit, Lit, Lit, Lit, Enum(E::ImageFormat)],
        ),
        "OpTypeSampler" => op(26, false, true, &[]),
        "OpTypeSampledImage" => op(27, false, true, &[Id]),
        "OpTypeArray" => op(28, false, true, &[Id, Id]),
        "OpTypeRuntimeArray" => op(29, false, true, &[Id]),
        "OpTypeStruct" => op(30, false, true, &[IdRest]),
        "OpTypePointer" => op(32, false, true, &[Enum(E::StorageClass), Id]),
        "OpTypeFunction" => op(33, false, true, &[Id, IdRest]),
        // --- Constants -------------------------------------------------------
        "OpConstantTrue" => op(41, true, true, &[]),
        "OpConstantFalse" => op(42, true, true, &[]),
        "OpConstant" => op(43, true, true, &[TypedLit]),
        "OpConstantComposite" => op(44, true, true, &[IdRest]),
        "OpSpecConstantTrue" => op(48, true, true, &[]),
        "OpSpecConstantFalse" => op(49, true, true, &[]),
        "OpSpecConstant" => op(50, true, true, &[TypedLit]),
        "OpSpecConstantComposite" => op(51, true, true, &[IdRest]),
        // --- Functions -------------------------------------------------------
        "OpFunction" => op(54, true, true, &[Mask(E::FunctionControl), Id]),
        "OpFunctionParameter" => op(55, true, true, &[]),
        "OpFunctionEnd" => op(56, false, false, &[]),
        "OpFunctionCall" => op(57, true, true, &[Id, IdRest]),
        // --- Memory ----------------------------------------------------------
        "OpVariable" => op(59, true, true, &[Enum(E::StorageClass), OptId]),
        "OpLoad" => op(61, true, true, &[Id, OptMask(E::MemoryAccess)]),
        "OpStore" => op(62, false, false, &[Id, Id, OptMask(E::MemoryAccess)]),
        "OpAccessChain" => op(65, true, true, &[Id, IdRest]),
        // Structure (id) + array-member index (literal) — the raw EUD-window
        // fallback reads its bound size through this (`sload_dword_extended`).
        "OpArrayLength" => op(68, true, true, &[Id, Lit]),
        // --- Annotations -----------------------------------------------------
        "OpDecorate" => op(71, false, false, &[Id, Enum(E::Decoration), DecoExtra]),
        "OpMemberDecorate" => op(72, false, false, &[Id, Lit, Enum(E::Decoration), DecoExtra]),
        // --- Composites ------------------------------------------------------
        "OpCompositeConstruct" => op(80, true, true, &[IdRest]),
        "OpCompositeExtract" => op(81, true, true, &[Id, LitRest]),
        // --- Images ----------------------------------------------------------
        "OpSampledImage" => op(86, true, true, &[Id, Id]),
        "OpImageSampleImplicitLod" => {
            op(87, true, true, &[Id, Id, OptMask(E::ImageOperands), IdRest])
        }
        "OpImageSampleExplicitLod" => {
            op(88, true, true, &[Id, Id, OptMask(E::ImageOperands), IdRest])
        }
        "OpImageFetch" => op(95, true, true, &[Id, Id, OptMask(E::ImageOperands), IdRest]),
        // Sampled image, coordinate, component, then optional image operands
        // (needed by the `image_gather4_lz` recompile body).
        "OpImageGather" => op(
            96,
            true,
            true,
            &[Id, Id, Id, OptMask(E::ImageOperands), IdRest],
        ),
        "OpImageWrite" => op(
            99,
            false,
            false,
            &[Id, Id, Id, OptMask(E::ImageOperands), IdRest],
        ),
        "OpImage" => op(100, true, true, &[Id]),
        "OpImageQuerySizeLod" => op(103, true, true, &[Id, Id]),
        // --- Conversions -----------------------------------------------------
        "OpConvertFToU" => op(109, true, true, IDR),
        "OpConvertFToS" => op(110, true, true, IDR),
        "OpConvertSToF" => op(111, true, true, IDR),
        "OpConvertUToF" => op(112, true, true, IDR),
        "OpBitcast" => op(124, true, true, IDR),
        // --- Arithmetic / logic / comparison (all operands are ids) ---------
        "OpFNegate" => op(127, true, true, IDR),
        "OpIAdd" => op(128, true, true, IDR),
        "OpFAdd" => op(129, true, true, IDR),
        "OpISub" => op(130, true, true, IDR),
        "OpFSub" => op(131, true, true, IDR),
        "OpIMul" => op(132, true, true, IDR),
        "OpFMul" => op(133, true, true, IDR),
        "OpSDiv" => op(135, true, true, IDR),
        "OpFDiv" => op(136, true, true, IDR),
        "OpIAddCarry" => op(149, true, true, IDR),
        "OpISubBorrow" => op(150, true, true, IDR),
        "OpUMulExtended" => op(151, true, true, IDR),
        "OpSMulExtended" => op(152, true, true, IDR),
        "OpIsNan" => op(156, true, true, IDR),
        "OpLogicalOr" => op(166, true, true, IDR),
        "OpLogicalAnd" => op(167, true, true, IDR),
        "OpLogicalNot" => op(168, true, true, IDR),
        "OpSelect" => op(169, true, true, IDR),
        "OpIEqual" => op(170, true, true, IDR),
        "OpINotEqual" => op(171, true, true, IDR),
        "OpUGreaterThan" => op(172, true, true, IDR),
        "OpSGreaterThan" => op(173, true, true, IDR),
        "OpUGreaterThanEqual" => op(174, true, true, IDR),
        "OpSGreaterThanEqual" => op(175, true, true, IDR),
        "OpULessThan" => op(176, true, true, IDR),
        "OpSLessThan" => op(177, true, true, IDR),
        "OpULessThanEqual" => op(178, true, true, IDR),
        "OpSLessThanEqual" => op(179, true, true, IDR),
        "OpFOrdEqual" => op(180, true, true, IDR),
        "OpFUnordEqual" => op(181, true, true, IDR),
        "OpFOrdNotEqual" => op(182, true, true, IDR),
        "OpFUnordNotEqual" => op(183, true, true, IDR),
        "OpFOrdLessThan" => op(184, true, true, IDR),
        "OpFUnordLessThan" => op(185, true, true, IDR),
        "OpFOrdGreaterThan" => op(186, true, true, IDR),
        "OpFUnordGreaterThan" => op(187, true, true, IDR),
        "OpFOrdLessThanEqual" => op(188, true, true, IDR),
        "OpFUnordLessThanEqual" => op(189, true, true, IDR),
        "OpFOrdGreaterThanEqual" => op(190, true, true, IDR),
        "OpFUnordGreaterThanEqual" => op(191, true, true, IDR),
        "OpShiftRightLogical" => op(194, true, true, IDR),
        "OpShiftRightArithmetic" => op(195, true, true, IDR),
        "OpShiftLeftLogical" => op(196, true, true, IDR),
        "OpBitwiseOr" => op(197, true, true, IDR),
        "OpBitwiseXor" => op(198, true, true, IDR),
        "OpBitwiseAnd" => op(199, true, true, IDR),
        "OpNot" => op(200, true, true, IDR),
        "OpBitFieldInsert" => op(201, true, true, IDR),
        "OpBitFieldSExtract" => op(202, true, true, IDR),
        "OpBitFieldUExtract" => op(203, true, true, IDR),
        "OpBitReverse" => op(204, true, true, IDR),
        "OpBitCount" => op(205, true, true, IDR),
        // --- Barriers / atomics ----------------------------------------------
        // Kyty passes scope/semantics as constant *ids* (`%uint_1 %uint_72`,
        // ShaderSpirv.cpp L2230-L2268), so these are plain id operands.
        "OpControlBarrier" => op(224, false, false, &[Id, Id, Id]),
        "OpMemoryBarrier" => op(225, false, false, &[Id, Id]),
        "OpAtomicExchange" => op(229, true, true, IDR),
        "OpAtomicIAdd" => op(234, true, true, IDR),
        "OpAtomicISub" => op(235, true, true, IDR),
        // --- Control flow ------------------------------------------------------
        "OpPhi" => op(245, true, true, IDR),
        "OpLoopMerge" => op(246, false, false, &[Id, Id, Mask(E::LoopControl)]),
        "OpSelectionMerge" => op(247, false, false, &[Id, Mask(E::SelectionControl)]),
        "OpLabel" => op(248, false, true, &[]),
        "OpBranch" => op(249, false, false, &[Id]),
        "OpBranchConditional" => op(250, false, false, &[Id, Id, Id, LitRest]),
        "OpSwitch" => op(251, false, false, &[Id, Id, Switch]),
        "OpKill" => op(252, false, false, &[]),
        "OpReturn" => op(253, false, false, &[]),
        "OpReturnValue" => op(254, false, false, &[Id]),
        "OpUnreachable" => op(255, false, false, &[]),
        "OpDemoteToHelperInvocation" => op(5380, false, false, &[]),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

fn is_numeric(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_digit())
}

fn parse_u32_lit(line: usize, tok: &str) -> Result<u32, AsmError> {
    let parsed = if let Some(h) = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X")) {
        u32::from_str_radix(h, 16).ok()
    } else {
        tok.parse::<u32>().ok()
    };
    parsed.ok_or_else(|| err(line, format!("expected integer literal, found '{tok}'")))
}

/// Signed-or-unsigned 32-bit literal (used by `OpSwitch` case values).
fn parse_i32_or_u32_lit(line: usize, tok: &str) -> Result<u32, AsmError> {
    if let Some(h) = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X")) {
        return u32::from_str_radix(h, 16)
            .map_err(|_| err(line, format!("invalid hex literal '{tok}'")));
    }
    let v: i64 = tok
        .parse()
        .map_err(|_| err(line, format!("expected integer literal, found '{tok}'")))?;
    if v >= 0 && v <= i64::from(u32::MAX) {
        Ok(v as u32)
    } else if v < 0 && v >= i64::from(i32::MIN) {
        Ok(v as i32 as u32)
    } else {
        Err(err(
            line,
            format!("integer literal '{tok}' out of 32-bit range"),
        ))
    }
}

fn push_string(out: &mut Vec<u32>, s: &str) {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    for c in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
}

/// Scalar numeric type info recorded from `OpTypeInt` / `OpTypeFloat` /
/// `OpTypeBool`, used to encode `OpConstant` literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumKind {
    Float(u32),
    Int { width: u32, signed: bool },
    Bool,
}

#[derive(Debug)]
struct NameInfo {
    id: u32,
    first_line: usize,
    defined: bool,
}

#[derive(Default)]
struct Encoder {
    names: HashMap<String, NameInfo>,
    reserved: HashSet<u32>,
    next: u32,
    max_id: u32,
    types: HashMap<u32, NumKind>,
    ext_sets: HashMap<u32, String>,
}

impl Encoder {
    fn new(reserved: HashSet<u32>) -> Self {
        Encoder {
            reserved,
            next: 1,
            ..Encoder::default()
        }
    }

    fn id_of(&mut self, name: &str, line: usize) -> Result<u32, AsmError> {
        if let Some(info) = self.names.get(name) {
            return Ok(info.id);
        }
        let id = if is_numeric(name) {
            let v: u32 = name
                .parse()
                .map_err(|_| err(line, format!("id %{name} out of range")))?;
            if v == 0 {
                return Err(err(line, format!("id %{name} must be non-zero")));
            }
            v
        } else {
            while self.reserved.contains(&self.next) {
                self.next += 1;
            }
            let v = self.next;
            self.next += 1;
            v
        };
        self.max_id = self.max_id.max(id);
        self.names.insert(
            name.to_string(),
            NameInfo {
                id,
                first_line: line,
                defined: false,
            },
        );
        Ok(id)
    }

    fn define(&mut self, name: &str, line: usize) -> Result<u32, AsmError> {
        let id = self.id_of(name, line)?;
        let info = self.names.get_mut(name).expect("id_of inserts");
        if info.defined {
            return Err(err(
                line,
                format!("duplicate definition of result id %{name}"),
            ));
        }
        info.defined = true;
        Ok(id)
    }

    fn encode_typed_literal(&self, rtype: u32, tok: &str, line: usize) -> Result<u32, AsmError> {
        let kind = self.types.get(&rtype).copied().ok_or_else(|| {
            err(
                line,
                "OpConstant result type must be an OpTypeInt/OpTypeFloat defined earlier",
            )
        })?;
        // `0x…` is a raw bit pattern for both int and float (Kyty emits
        // `0x%08x` for uint constants, ShaderSpirv.cpp L6527).
        if let Some(h) = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X")) {
            return u32::from_str_radix(h, 16)
                .map_err(|_| err(line, format!("invalid hex literal '{tok}'")));
        }
        match kind {
            NumKind::Float(32) => tok
                .parse::<f32>()
                .map(f32::to_bits)
                .map_err(|_| err(line, format!("invalid float literal '{tok}'"))),
            NumKind::Float(w) => Err(err(
                line,
                format!("{w}-bit float constants unsupported (Kyty emits 32-bit only)"),
            )),
            NumKind::Int { width: 32, signed } => {
                let v: i64 = tok
                    .parse()
                    .map_err(|_| err(line, format!("invalid integer literal '{tok}'")))?;
                let ok = if signed {
                    v >= i64::from(i32::MIN) && v <= i64::from(i32::MAX)
                } else {
                    v >= 0 && v <= i64::from(u32::MAX)
                };
                if !ok {
                    return Err(err(
                        line,
                        format!("integer literal '{tok}' out of range for result type"),
                    ));
                }
                Ok(if v < 0 { v as i32 as u32 } else { v as u32 })
            }
            NumKind::Int { width, .. } => Err(err(
                line,
                format!("{width}-bit integer constants unsupported (Kyty emits 32-bit only)"),
            )),
            NumKind::Bool => Err(err(
                line,
                "boolean constants use OpConstantTrue/OpConstantFalse",
            )),
        }
    }

    fn parse_mask(&self, kind: EKind, tok: &str, line: usize) -> Result<u32, AsmError> {
        let mut mask = 0u32;
        for part in tok.split('|') {
            mask |= enum_value(kind, part)
                .ok_or_else(|| err(line, format!("unknown {} token '{part}'", ekind_name(kind))))?;
        }
        Ok(mask)
    }

    #[allow(clippy::too_many_lines)]
    fn encode_inst(&mut self, inst: &RawInst, out: &mut Vec<u32>) -> Result<(), AsmError> {
        let line = inst.line;
        let info = op_info(&inst.opcode)
            .ok_or_else(|| err(line, format!("unknown opcode '{}'", inst.opcode)))?;

        let result_id = match (&inst.result, info.res) {
            (Some(name), true) => Some(self.define(name, line)?),
            (None, false) => None,
            (Some(_), false) => {
                return Err(err(
                    line,
                    format!("{} does not take a result id", inst.opcode),
                ));
            }
            (None, true) => {
                return Err(err(
                    line,
                    format!("{} requires a result id (`%name = …`)", inst.opcode),
                ));
            }
        };

        let args = inst.args.as_slice();
        let mut idx = 0usize;
        let rtype_id = if info.rt {
            match args.first() {
                Some(Tok::Id(n)) => {
                    idx = 1;
                    Some(self.id_of(n, line)?)
                }
                _ => {
                    return Err(err(
                        line,
                        format!("{} expects a result type id as first operand", inst.opcode),
                    ));
                }
            }
        } else {
            None
        };

        let mut ow: Vec<u32> = Vec::new();
        let mut deco: Option<u32> = None;

        for kind in info.ops {
            match kind {
                Kind::Id => match args.get(idx) {
                    Some(Tok::Id(n)) => {
                        ow.push(self.id_of(n, line)?);
                        idx += 1;
                    }
                    other => {
                        return Err(err(
                            line,
                            format!("{} expects an id operand, found {other:?}", inst.opcode),
                        ));
                    }
                },
                Kind::IdRest => {
                    while idx < args.len() {
                        match &args[idx] {
                            Tok::Id(n) => {
                                ow.push(self.id_of(n, line)?);
                                idx += 1;
                            }
                            other => {
                                return Err(err(
                                    line,
                                    format!("expected id operand, found {other:?}"),
                                ));
                            }
                        }
                    }
                }
                Kind::Lit => match args.get(idx) {
                    Some(Tok::Word(w)) => {
                        ow.push(parse_u32_lit(line, w)?);
                        idx += 1;
                    }
                    other => {
                        return Err(err(
                            line,
                            format!("expected integer literal, found {other:?}"),
                        ));
                    }
                },
                Kind::LitRest => {
                    while idx < args.len() {
                        match &args[idx] {
                            Tok::Word(w) => {
                                ow.push(parse_u32_lit(line, w)?);
                                idx += 1;
                            }
                            other => {
                                return Err(err(
                                    line,
                                    format!("expected integer literal, found {other:?}"),
                                ));
                            }
                        }
                    }
                }
                Kind::Str_ => match args.get(idx) {
                    Some(Tok::Str(s)) => {
                        push_string(&mut ow, s);
                        idx += 1;
                    }
                    other => {
                        return Err(err(
                            line,
                            format!("expected string literal, found {other:?}"),
                        ));
                    }
                },
                Kind::OptId => {
                    if let Some(Tok::Id(n)) = args.get(idx) {
                        ow.push(self.id_of(n, line)?);
                        idx += 1;
                    }
                }
                Kind::OptStr => {
                    if let Some(Tok::Str(s)) = args.get(idx) {
                        push_string(&mut ow, s);
                        idx += 1;
                    }
                }
                Kind::TypedLit => {
                    let rtype = rtype_id.expect("TypedLit opcodes have a result type");
                    match args.get(idx) {
                        Some(Tok::Word(w)) => {
                            ow.push(self.encode_typed_literal(rtype, w, line)?);
                            idx += 1;
                        }
                        other => {
                            return Err(err(
                                line,
                                format!("expected constant literal, found {other:?}"),
                            ));
                        }
                    }
                }
                Kind::Enum(k) => match args.get(idx) {
                    Some(Tok::Word(w)) => {
                        let v = enum_value(*k, w).ok_or_else(|| {
                            err(line, format!("unknown {} token '{w}'", ekind_name(*k)))
                        })?;
                        if *k == EKind::Decoration {
                            deco = Some(v);
                        }
                        ow.push(v);
                        idx += 1;
                    }
                    other => {
                        return Err(err(
                            line,
                            format!("expected {} token, found {other:?}", ekind_name(*k)),
                        ));
                    }
                },
                Kind::Mask(k) => match args.get(idx) {
                    Some(Tok::Word(w)) => {
                        ow.push(self.parse_mask(*k, w, line)?);
                        idx += 1;
                    }
                    other => {
                        return Err(err(
                            line,
                            format!("expected {} mask, found {other:?}", ekind_name(*k)),
                        ));
                    }
                },
                Kind::OptMask(k) => {
                    if let Some(Tok::Word(w)) = args.get(idx) {
                        let mask = self.parse_mask(*k, w, line)?;
                        ow.push(mask);
                        idx += 1;
                        // MemoryAccess `Aligned` carries a literal alignment.
                        if *k == EKind::MemoryAccess && mask & 0x2 != 0 {
                            match args.get(idx) {
                                Some(Tok::Word(a)) => {
                                    ow.push(parse_u32_lit(line, a)?);
                                    idx += 1;
                                }
                                other => {
                                    return Err(err(
                                        line,
                                        format!("Aligned expects a literal, found {other:?}"),
                                    ));
                                }
                            }
                        }
                    }
                }
                Kind::ExtName => {
                    let set_id = *ow.last().expect("ExtName follows the set id operand");
                    match args.get(idx) {
                        Some(Tok::Word(w)) => {
                            let num = if is_numeric(w) {
                                parse_u32_lit(line, w)?
                            } else {
                                let set = self.ext_sets.get(&set_id).ok_or_else(|| {
                                    err(line, "OpExtInst set was not imported via OpExtInstImport")
                                })?;
                                match set.as_str() {
                                    "GLSL.std.450" => glsl_std_450(w).ok_or_else(|| {
                                        err(line, format!("unknown GLSL.std.450 instruction '{w}'"))
                                    })?,
                                    "NonSemantic.DebugPrintf" if w == "DebugPrintf" => 1,
                                    other => {
                                        return Err(err(
                                            line,
                                            format!(
                                                "unknown extended instruction '{w}' in set '{other}'"
                                            ),
                                        ));
                                    }
                                }
                            };
                            ow.push(num);
                            idx += 1;
                        }
                        other => {
                            return Err(err(
                                line,
                                format!(
                                    "expected extended-instruction name/number, found {other:?}"
                                ),
                            ));
                        }
                    }
                }
                Kind::Switch => {
                    while idx < args.len() {
                        match &args[idx] {
                            Tok::Word(w) => {
                                ow.push(parse_i32_or_u32_lit(line, w)?);
                                idx += 1;
                            }
                            other => {
                                return Err(err(
                                    line,
                                    format!("OpSwitch expects a case literal, found {other:?}"),
                                ));
                            }
                        }
                        match args.get(idx) {
                            Some(Tok::Id(n)) => {
                                ow.push(self.id_of(n, line)?);
                                idx += 1;
                            }
                            other => {
                                return Err(err(
                                    line,
                                    format!("OpSwitch expects a case target id, found {other:?}"),
                                ));
                            }
                        }
                    }
                }
                Kind::DecoExtra => match deco {
                    // BuiltIn <builtin-token>
                    Some(11) => match args.get(idx) {
                        Some(Tok::Word(w)) => {
                            let v = enum_value(EKind::BuiltIn, w)
                                .ok_or_else(|| err(line, format!("unknown BuiltIn token '{w}'")))?;
                            ow.push(v);
                            idx += 1;
                        }
                        other => {
                            return Err(err(
                                line,
                                format!("BuiltIn decoration expects a token, found {other:?}"),
                            ));
                        }
                    },
                    // SpecId / ArrayStride / MatrixStride / Location /
                    // Component / Index / Binding / DescriptorSet / Offset
                    Some(1 | 6 | 7 | 29..=35) => match args.get(idx) {
                        Some(Tok::Word(w)) => {
                            ow.push(parse_u32_lit(line, w)?);
                            idx += 1;
                        }
                        other => {
                            return Err(err(
                                line,
                                format!("decoration expects an integer literal, found {other:?}"),
                            ));
                        }
                    },
                    // Block / Coherent / Flat / … take no extra operands.
                    _ => {}
                },
            }
        }

        if idx != args.len() {
            return Err(err(
                line,
                format!(
                    "unexpected extra operand {:?} for {}",
                    args[idx], inst.opcode
                ),
            ));
        }

        // Assemble: word0 = (wordcount << 16) | opcode, then result type,
        // result, operands.
        let start = out.len();
        out.push(0);
        if let Some(t) = rtype_id {
            out.push(t);
        }
        if let Some(r) = result_id {
            out.push(r);
        }
        out.extend_from_slice(&ow);
        let wc = (out.len() - start) as u32;
        out[start] = (wc << 16) | u32::from(info.code);

        // Post-actions: record scalar types and extended-instruction sets.
        match inst.opcode.as_str() {
            "OpTypeInt" => {
                self.types.insert(
                    result_id.expect("OpTypeInt has a result"),
                    NumKind::Int {
                        width: ow[0],
                        signed: ow[1] != 0,
                    },
                );
            }
            "OpTypeFloat" => {
                self.types.insert(
                    result_id.expect("OpTypeFloat has a result"),
                    NumKind::Float(ow[0]),
                );
            }
            "OpTypeBool" => {
                self.types
                    .insert(result_id.expect("OpTypeBool has a result"), NumKind::Bool);
            }
            "OpExtInstImport" => {
                if let Some(Tok::Str(s)) = inst.args.first() {
                    self.ext_sets
                        .insert(result_id.expect("OpExtInstImport has a result"), s.clone());
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Assemble SPIR-V assembly text (the dialect Kyty's recompiler emits) into
/// a SPIR-V 1.5 binary module.
pub fn assemble(text: &str) -> Result<Vec<u32>, AsmError> {
    let mut insts: Vec<RawInst> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let toks = tokenize_line(i + 1, line)?;
        if let Some(inst) = parse_inst(i + 1, toks)? {
            insts.push(inst);
        }
    }

    // Pass 1: numeric ids (`%42`) claim their exact value; collect them so
    // symbolic assignment can skip those numbers.
    let mut reserved: HashSet<u32> = HashSet::new();
    for inst in &insts {
        if let Some(r) = &inst.result {
            if is_numeric(r) {
                if let Ok(v) = r.parse::<u32>() {
                    reserved.insert(v);
                }
            }
        }
        for t in &inst.args {
            if let Tok::Id(n) = t {
                if is_numeric(n) {
                    if let Ok(v) = n.parse::<u32>() {
                        reserved.insert(v);
                    }
                }
            }
        }
    }

    // Pass 2: encode in source order.
    let mut enc = Encoder::new(reserved);
    let mut body: Vec<u32> = Vec::new();
    for inst in &insts {
        enc.encode_inst(inst, &mut body)?;
    }

    // Every referenced id must be defined somewhere (Kyty always defines its
    // ids; catching this here replaces the SPIRV-Tools validation step for
    // dangling references).
    if let Some((name, info)) = enc
        .names
        .iter()
        .filter(|(_, i)| !i.defined)
        .min_by_key(|(_, i)| i.first_line)
    {
        return Err(err(
            info.first_line,
            format!("id %{name} is used but never defined"),
        ));
    }

    let mut words = Vec::with_capacity(5 + body.len());
    words.push(SPIRV_MAGIC);
    words.push(SPIRV_VERSION);
    words.push(0); // generator
    words.push(enc.max_id + 1); // bound
    words.push(0); // schema
    words.extend_from_slice(&body);
    Ok(words)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Walk instruction word counts; the module must partition exactly.
    fn check_word_counts(words: &[u32]) {
        assert!(words.len() >= 5, "missing header");
        let mut idx = 5;
        while idx < words.len() {
            let wc = (words[idx] >> 16) as usize;
            assert!(wc >= 1, "zero word count at index {idx}");
            idx += wc;
        }
        assert_eq!(idx, words.len(), "instruction stream over/underruns");
    }

    // --- 1. Tokenizer -----------------------------------------------------

    #[test]
    fn tokenize_comments_and_whitespace() {
        let toks =
            tokenize_line(1, "   OpReturn   ; trailing comment %not_an_id \"nope\"").unwrap();
        assert_eq!(toks, vec![Tok::Word("OpReturn".into())]);
        assert!(tokenize_line(1, "; only a comment").unwrap().is_empty());
        assert!(tokenize_line(1, "\t \t").unwrap().is_empty());
    }

    #[test]
    fn tokenize_string_escapes() {
        let toks = tokenize_line(1, r#"OpString "a\"b\\c %d""#).unwrap();
        assert_eq!(
            toks,
            vec![Tok::Word("OpString".into()), Tok::Str(r#"a"b\c %d"#.into()),]
        );
    }

    #[test]
    fn tokenize_id_assignment_and_literals() {
        let toks = tokenize_line(1, "%x_1 = OpConstant %float 0x3F800000").unwrap();
        assert_eq!(
            toks,
            vec![
                Tok::Id("x_1".into()),
                Tok::Eq,
                Tok::Word("OpConstant".into()),
                Tok::Id("float".into()),
                Tok::Word("0x3F800000".into()),
            ]
        );
        let toks = tokenize_line(1, "OpExecutionMode %main LocalSize 8 8 1").unwrap();
        assert_eq!(toks.len(), 6);
        assert_eq!(toks[3], Tok::Word("8".into()));
    }

    #[test]
    fn tokenize_unterminated_string_errors() {
        let e = tokenize_line(7, r#"OpString "oops"#).unwrap_err();
        assert_eq!(e.line, 7);
        assert!(e.message.contains("unterminated"));
    }

    // --- 2. Minimal module golden words ------------------------------------

    const MINIMAL: &str = r#"
                ; Minimal fragment module
                OpCapability Shader
                OpMemoryModel Logical GLSL450
                OpEntryPoint Fragment %main "main"
                OpExecutionMode %main OriginUpperLeft
        %void = OpTypeVoid
        %fnty = OpTypeFunction %void
        %main = OpFunction %void None %fnty
         %lbl = OpLabel
                OpReturn
                OpFunctionEnd
    "#;

    #[test]
    fn minimal_module_golden_words() {
        let words = assemble(MINIMAL).unwrap();
        // Ids in first-appearance order: main=1, void=2, fnty=3, lbl=4.
        let expected = vec![
            SPIRV_MAGIC,
            SPIRV_VERSION,
            0,
            5, // bound
            0,
            (2 << 16) | 17, // OpCapability Shader
            1,
            (3 << 16) | 14, // OpMemoryModel Logical GLSL450
            0,
            1,
            (5 << 16) | 15, // OpEntryPoint Fragment %1 "main"
            4,
            1,
            0x6E69_616D, // "main"
            0,
            (3 << 16) | 16, // OpExecutionMode %1 OriginUpperLeft
            1,
            7,
            (2 << 16) | 19, // OpTypeVoid
            2,
            (3 << 16) | 33, // OpTypeFunction
            3,
            2,
            (5 << 16) | 54, // OpFunction %void None %fnty
            2,
            1,
            0,
            3,
            (2 << 16) | 248, // OpLabel
            4,
            (1 << 16) | 253, // OpReturn
            (1 << 16) | 56,  // OpFunctionEnd
        ];
        assert_eq!(words, expected);
    }

    // --- 3. Types & constants ----------------------------------------------

    #[test]
    fn float_constant_bit_patterns() {
        let words = assemble(
            "%float = OpTypeFloat 32\n\
             %a = OpConstant %float 1.5\n\
             %b = OpConstant %float -1\n\
             %c = OpConstant %float 0.000000\n\
             %d = OpConstant %float 6.283185307179586476925286766559\n",
        )
        .unwrap();
        // Header (5 words) + OpTypeFloat (3 words), then each OpConstant is
        // 4 words: word0, type, result, value.
        let base = 5 + 3;
        assert_eq!(words[base + 3], 0x3FC0_0000); // 1.5f32
        assert_eq!(words[base + 7], 0xBF80_0000); // -1.0f32
        assert_eq!(words[base + 11], 0);
        let two_pi: f32 = "6.283185307179586476925286766559".parse().unwrap();
        assert_eq!(words[base + 15], two_pi.to_bits());
    }

    #[test]
    fn int_constants_signed_unsigned_hex() {
        let words = assemble(
            "%int = OpTypeInt 32 1\n\
             %uint = OpTypeInt 32 0\n\
             %a = OpConstant %int -1\n\
             %b = OpConstant %uint 0xdeadbeef\n\
             %c = OpConstant %uint 4\n\
             %d = OpConstant %int 3\n",
        )
        .unwrap();
        let base = 5 + 4 + 4; // header + two OpTypeInt (4 words each)
        assert_eq!(words[base + 3], 0xFFFF_FFFF);
        assert_eq!(words[base + 7], 0xDEAD_BEEF);
        assert_eq!(words[base + 11], 4);
        assert_eq!(words[base + 15], 3);
    }

    #[test]
    fn unsigned_constant_rejects_negative() {
        let e = assemble("%uint = OpTypeInt 32 0\n%a = OpConstant %uint -5\n").unwrap_err();
        assert_eq!(e.line, 2);
        assert!(e.message.contains("out of range"));
    }

    #[test]
    fn constant_composite_words() {
        let words = assemble(
            "%float = OpTypeFloat 32\n\
             %v4 = OpTypeVector %float 4\n\
             %z = OpConstant %float 0\n\
             %cc = OpConstantComposite %v4 %z %z %z %z\n",
        )
        .unwrap();
        let n = words.len();
        // Ids in first-appearance order: float=1, v4=2, z=3, cc=4.
        // Last instruction: (7<<16)|44, v4, cc, z, z, z, z.
        assert_eq!(&words[n - 7..], &[(7 << 16) | 44, 2, 4, 3, 3, 3, 3][..]);
    }

    #[test]
    fn sixty_four_bit_float_constant_rejected() {
        let e = assemble("%f64 = OpTypeFloat 64\n%a = OpConstant %f64 1.0\n").unwrap_err();
        assert_eq!(e.line, 2);
        assert!(e.message.contains("64-bit float"));
    }

    // --- 4. Ids -------------------------------------------------------------

    #[test]
    fn forward_reference_and_bound() {
        // %main referenced by OpEntryPoint before its definition.
        let words = assemble(MINIMAL).unwrap();
        assert_eq!(words[3], 5); // bound = max id (4) + 1
    }

    #[test]
    fn numeric_ids_reserved_and_skipped() {
        // `%2` claims id 2; symbolic %fn takes 1, %next takes 3.
        let words = assemble(
            "%2 = OpTypeVoid\n\
             %fn = OpTypeFunction %2\n\
             %next = OpTypeBool\n",
        )
        .unwrap();
        assert_eq!(words[3], 4); // bound
        let expected = [
            (2 << 16) | 19,
            2, // OpTypeVoid  -> id 2
            (3 << 16) | 33,
            1,
            2, // OpTypeFunction -> id 1
            (2 << 16) | 20,
            3, // OpTypeBool -> id 3
        ];
        assert_eq!(&words[5..], &expected[..]);
    }

    #[test]
    fn duplicate_result_id_errors() {
        let e = assemble("%a = OpTypeVoid\n%a = OpTypeBool\n").unwrap_err();
        assert_eq!(e.line, 2);
        assert!(e.message.contains("duplicate definition"));
    }

    #[test]
    fn undefined_id_errors() {
        let e = assemble("OpBranch %nowhere\n").unwrap_err();
        assert_eq!(e.line, 1);
        assert!(e.message.contains("%nowhere"));
        assert!(e.message.contains("never defined"));
    }

    // --- 5. Structural instructions -----------------------------------------

    #[test]
    fn decorations_and_member_offsets() {
        let words = assemble(
            "OpDecorate %v DescriptorSet 0\n\
             OpDecorate %v Binding 3\n\
             OpDecorate %s Block\n\
             OpDecorate %p BuiltIn FragCoord\n\
             OpMemberDecorate %s 0 Offset 16\n\
             OpMemberDecorate %s 1 Coherent\n\
             %float = OpTypeFloat 32\n\
             %s = OpTypeStruct %float\n\
             %uc = OpTypePointer UniformConstant %float\n\
             %v = OpVariable %uc UniformConstant\n\
             %inp = OpTypePointer Input %float\n\
             %p = OpVariable %inp Input\n",
        )
        .unwrap();
        // OpDecorate %v DescriptorSet 0 => (4<<16)|71, id(v)=1, 34, 0
        assert_eq!(&words[5..9], &[(4 << 16) | 71, 1, 34, 0]);
        // OpDecorate %v Binding 3
        assert_eq!(&words[9..13], &[(4 << 16) | 71, 1, 33, 3]);
        // OpDecorate %s Block => (3<<16)|71, id(s)=2, 2
        assert_eq!(&words[13..16], &[(3 << 16) | 71, 2, 2]);
        // OpDecorate %p BuiltIn FragCoord => (4<<16)|71, id(p)=3, 11, 15
        assert_eq!(&words[16..20], &[(4 << 16) | 71, 3, 11, 15]);
        // OpMemberDecorate %s 0 Offset 16 => (5<<16)|72, 2, 0, 35, 16
        assert_eq!(&words[20..25], &[(5 << 16) | 72, 2, 0, 35, 16]);
        // OpMemberDecorate %s 1 Coherent => (4<<16)|72, 2, 1, 23
        assert_eq!(&words[25..29], &[(4 << 16) | 72, 2, 1, 23]);
        check_word_counts(&words);
    }

    #[test]
    fn variables_access_chain_load_store() {
        let words = assemble(
            "%float = OpTypeFloat 32\n\
             %uint = OpTypeInt 32 0\n\
             %u0 = OpConstant %uint 0\n\
             %v4 = OpTypeVector %float 4\n\
             %ptr_out_v4 = OpTypePointer Output %v4\n\
             %ptr_out_f = OpTypePointer Output %float\n\
             %ptr_fn_f = OpTypePointer Function %float\n\
             %out = OpVariable %ptr_out_v4 Output\n\
             %tmp = OpVariable %ptr_fn_f Function\n\
             %ac = OpAccessChain %ptr_out_f %out %u0\n\
             %ld = OpLoad %float %ac\n\
             OpStore %tmp %ld\n",
        )
        .unwrap();
        check_word_counts(&words);
        let n = words.len();
        // OpStore %tmp %ld = (3<<16)|62, tmp, ld
        assert_eq!(words[n - 3], (3 << 16) | 62);
        // OpLoad = (4<<16)|61
        assert_eq!(words[n - 7], (4 << 16) | 61);
        // OpAccessChain with 2 index-ids = (5<<16)|65
        assert_eq!(words[n - 12], (5 << 16) | 65);
    }

    #[test]
    fn image_sample_with_lod_mask() {
        let words = assemble(
            "%float = OpTypeFloat 32\n\
             %v2 = OpTypeVector %float 2\n\
             %v4 = OpTypeVector %float 4\n\
             %img_t = OpTypeImage %float 2D 0 0 0 1 Unknown\n\
             %simg_t = OpTypeSampledImage %img_t\n\
             %f0 = OpConstant %float 0.000000\n\
             %coord = OpUndef %v2\n\
             %simg = OpUndef %simg_t\n\
             %r = OpImageSampleExplicitLod %v4 %simg %coord Lod %f0\n",
        )
        .unwrap();
        check_word_counts(&words);
        let n = words.len();
        // (7<<16)|88, v4, r, simg, coord, mask=2 (Lod), f0
        assert_eq!(words[n - 7], (7 << 16) | 88);
        assert_eq!(words[n - 2], 0x2);
        // OpTypeImage: 9 words, Dim 2D = 1, sampled = 1, format Unknown = 0.
        let ti = words
            .iter()
            .position(|w| *w == (9 << 16) | 25)
            .expect("OpTypeImage");
        assert_eq!(words[ti + 3], 1); // Dim 2D
        assert_eq!(words[ti + 7], 1); // sampled
        assert_eq!(words[ti + 8], 0); // Unknown
    }

    #[test]
    fn ext_inst_glsl_named_and_numeric() {
        let words = assemble(
            "%glsl = OpExtInstImport \"GLSL.std.450\"\n\
             %dbg = OpExtInstImport \"NonSemantic.DebugPrintf\"\n\
             %float = OpTypeFloat 32\n\
             %void = OpTypeVoid\n\
             %a = OpUndef %float\n\
             %b = OpUndef %float\n\
             %s = OpString \"x = %f\"\n\
             %r = OpExtInst %float %glsl FMin %a %b\n\
             %p = OpExtInst %void %dbg 1 %s %a\n",
        )
        .unwrap();
        check_word_counts(&words);
        // Find the FMin OpExtInst: (7<<16)|12, float, r, glsl, 37, a, b
        let fmin = words
            .iter()
            .position(|w| *w == (7 << 16) | 12)
            .expect("OpExtInst FMin");
        assert_eq!(words[fmin + 4], 37); // GLSL.std.450 FMin
        // The DebugPrintf one uses the raw number 1 (ShaderSpirv.cpp L6167).
        let dbg = words[fmin + 7..]
            .iter()
            .position(|w| *w == (7 << 16) | 12)
            .map(|p| p + fmin + 7)
            .expect("OpExtInst DebugPrintf");
        assert_eq!(words[dbg + 4], 1);
    }

    #[test]
    fn ext_inst_unknown_name_errors() {
        let e = assemble(
            "%glsl = OpExtInstImport \"GLSL.std.450\"\n\
             %float = OpTypeFloat 32\n\
             %a = OpUndef %float\n\
             %r = OpExtInst %float %glsl NotAnInst %a\n",
        )
        .unwrap_err();
        assert_eq!(e.line, 4);
        assert!(e.message.contains("NotAnInst"));
    }

    #[test]
    fn branches_and_selection_merge() {
        let words = assemble(
            "%bool = OpTypeBool\n\
             %t = OpConstantTrue %bool\n\
             %void = OpTypeVoid\n\
             %fnty = OpTypeFunction %void\n\
             %f = OpFunction %void Inline|Pure %fnty\n\
             %e = OpLabel\n\
             OpSelectionMerge %m None\n\
             OpBranchConditional %t %then %m\n\
             %then = OpLabel\n\
             OpBranch %m\n\
             %m = OpLabel\n\
             OpReturn\n\
             OpFunctionEnd\n",
        )
        .unwrap();
        check_word_counts(&words);
        // OpFunction control mask Inline|Pure = 5.
        let f = words
            .iter()
            .position(|w| *w == (5 << 16) | 54)
            .expect("OpFunction");
        assert_eq!(words[f + 3], 5);
        // OpSelectionMerge %m None = (3<<16)|247, m, 0
        let sm = words
            .iter()
            .position(|w| *w == (3 << 16) | 247)
            .expect("OpSelectionMerge");
        assert_eq!(words[sm + 2], 0);
        // OpBranchConditional = (4<<16)|250
        assert!(words.contains(&((4 << 16) | 250)));
    }

    #[test]
    fn switch_phi_loop_merge_mipmap_shape() {
        // Mirrors the shape of Kyty's mipmap helper function
        // (ShaderSpirv.cpp L246-L262): OpSwitch with only a default target,
        // OpPhi merges, OpLoopMerge.
        let words = assemble(
            "%uint = OpTypeInt 32 0\n\
             %u0 = OpConstant %uint 0\n\
             %void = OpTypeVoid\n\
             %fnty = OpTypeFunction %void\n\
             %f = OpFunction %void None %fnty\n\
             %e = OpLabel\n\
             OpSelectionMerge %sm None\n\
             OpSwitch %u0 %case\n\
             %case = OpLabel\n\
             OpBranch %loop\n\
             %loop = OpLabel\n\
             %acc = OpPhi %uint %u0 %case %nxt %cont\n\
             OpLoopMerge %sm %cont None\n\
             OpBranch %body\n\
             %body = OpLabel\n\
             %nxt = OpIAdd %uint %acc %u0\n\
             OpBranch %cont\n\
             %cont = OpLabel\n\
             OpBranch %loop\n\
             %sm = OpLabel\n\
             OpReturn\n\
             OpFunctionEnd\n",
        )
        .unwrap();
        check_word_counts(&words);
        // OpSwitch with no case pairs: (3<<16)|251, selector, default.
        assert!(words.contains(&((3 << 16) | 251)));
        // OpPhi with 2 (value, parent) pairs: (7<<16)|245.
        assert!(words.contains(&((7 << 16) | 245)));
        // OpLoopMerge: (4<<16)|246 … None mask 0.
        let lm = words
            .iter()
            .position(|w| *w == (4 << 16) | 246)
            .expect("OpLoopMerge");
        assert_eq!(words[lm + 3], 0);
    }

    #[test]
    fn switch_with_case_pairs() {
        let words = assemble(
            "%uint = OpTypeInt 32 0\n\
             %sel = OpConstant %uint 2\n\
             %void = OpTypeVoid\n\
             %fnty = OpTypeFunction %void\n\
             %f = OpFunction %void None %fnty\n\
             %e = OpLabel\n\
             OpSelectionMerge %m None\n\
             OpSwitch %sel %m 0 %c0 7 %c1\n\
             %c0 = OpLabel\n\
             OpBranch %m\n\
             %c1 = OpLabel\n\
             OpBranch %m\n\
             %m = OpLabel\n\
             OpReturn\n\
             OpFunctionEnd\n",
        )
        .unwrap();
        check_word_counts(&words);
        let sw = words
            .iter()
            .position(|w| *w == (7 << 16) | 251)
            .expect("OpSwitch with two case pairs");
        assert_eq!(words[sw + 3], 0); // case literal 0
        assert_eq!(words[sw + 5], 7); // case literal 7
    }

    #[test]
    fn atomics_and_memory_barrier() {
        // Kyty passes scope/semantics as constant ids
        // (ShaderSpirv.cpp L2230-L2268).
        let words = assemble(
            "%uint = OpTypeInt 32 0\n\
             %u1 = OpConstant %uint 1\n\
             %u72 = OpConstant %uint 72\n\
             %ptr = OpTypePointer StorageBuffer %uint\n\
             %p = OpUndef %ptr\n\
             %r = OpAtomicIAdd %uint %p %u1 %u1 %u1\n\
             OpMemoryBarrier %u1 %u72\n",
        )
        .unwrap();
        check_word_counts(&words);
        assert!(words.contains(&((7 << 16) | 234))); // OpAtomicIAdd
        let n = words.len();
        assert_eq!(words[n - 3], (3 << 16) | 225); // OpMemoryBarrier
    }

    #[test]
    fn compute_entry_point_and_local_size() {
        // Kyty compute header: GLCompute + LocalSize (WriteHeader, L6779).
        let words = assemble(
            "OpCapability Shader\n\
             OpMemoryModel Logical GLSL450\n\
             OpEntryPoint GLCompute %main \"main\" %gl_LocalInvocationID\n\
             OpExecutionMode %main LocalSize 8 8 1\n\
             OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId\n\
             %uint = OpTypeInt 32 0\n\
             %v3uint = OpTypeVector %uint 3\n\
             %ptr = OpTypePointer Input %v3uint\n\
             %gl_LocalInvocationID = OpVariable %ptr Input\n\
             %void = OpTypeVoid\n\
             %fnty = OpTypeFunction %void\n\
             %main = OpFunction %void None %fnty\n\
             %e = OpLabel\n\
             OpReturn\n\
             OpFunctionEnd\n",
        )
        .unwrap();
        check_word_counts(&words);
        // OpExecutionMode %main LocalSize 8 8 1 = (6<<16)|16, main, 17, 8, 8, 1
        let em = words
            .iter()
            .position(|w| *w == (6 << 16) | 16)
            .expect("OpExecutionMode LocalSize");
        assert_eq!(&words[em + 2..em + 6], &[17, 8, 8, 1]);
        // BuiltIn LocalInvocationId = 27.
        let dec = words
            .iter()
            .position(|w| *w == (4 << 16) | 71)
            .expect("OpDecorate BuiltIn");
        assert_eq!(words[dec + 3], 27);
    }

    // --- 6. Realistic Kyty-shaped fragment shader ---------------------------

    /// Modeled on ShaderSpirv.cpp WriteHeader (L6685 region) and the
    /// pixel-shader templates (L1279+): capabilities, GLSL.std.450 import,
    /// Fragment entry point with interface list, OriginUpperLeft, mixed
    /// numeric/symbolic ids, texture sample with `Lod`, ExtInst, phi merge.
    const FRAGMENT: &str = "\
            ; Kyty-shaped fragment shader\n\
            OpCapability Shader\n\
            OpCapability ImageQuery\n\
            %GLSL_std_450 = OpExtInstImport \"GLSL.std.450\"\n\
            OpMemoryModel Logical GLSL450\n\
            OpEntryPoint Fragment %main \"main\" %outColor %attr0\n\
            OpExecutionMode %main OriginUpperLeft\n\
            OpDecorate %outColor Location 0\n\
            OpDecorate %attr0 Location 0\n\
            OpDecorate %img DescriptorSet 0\n\
            OpDecorate %img Binding 0\n\
            OpDecorate %smp DescriptorSet 0\n\
            OpDecorate %smp Binding 1\n\
            %void = OpTypeVoid\n\
            %3 = OpTypeFunction %void\n\
            %float = OpTypeFloat 32\n\
            %v2float = OpTypeVector %float 2\n\
            %v4float = OpTypeVector %float 4\n\
            %bool = OpTypeBool\n\
            %image2d = OpTypeImage %float 2D 0 0 0 1 Unknown\n\
            %sampled2d = OpTypeSampledImage %image2d\n\
            %sampler = OpTypeSampler\n\
            %_ptr_UniformConstant_image2d = OpTypePointer UniformConstant %image2d\n\
            %_ptr_UniformConstant_sampler = OpTypePointer UniformConstant %sampler\n\
            %_ptr_Output_v4float = OpTypePointer Output %v4float\n\
            %_ptr_Input_v2float = OpTypePointer Input %v2float\n\
            %float_0_000000 = OpConstant %float 0.000000\n\
            %float_1_000000 = OpConstant %float 1.000000\n\
            %img = OpVariable %_ptr_UniformConstant_image2d UniformConstant\n\
            %smp = OpVariable %_ptr_UniformConstant_sampler UniformConstant\n\
            %outColor = OpVariable %_ptr_Output_v4float Output\n\
            %attr0 = OpVariable %_ptr_Input_v2float Input\n\
            %main = OpFunction %void None %3\n\
            %5 = OpLabel\n\
            %t40_0 = OpLoad %v2float %attr0\n\
            %t41_0 = OpLoad %image2d %img\n\
            %t42_0 = OpLoad %sampler %smp\n\
            %t43_0 = OpSampledImage %sampled2d %t41_0 %t42_0\n\
            %t44_0 = OpImageSampleExplicitLod %v4float %t43_0 %t40_0 Lod %float_0_000000\n\
            %t45_0 = OpCompositeExtract %float %t44_0 0\n\
            %t46_0 = OpFOrdGreaterThan %bool %t45_0 %float_0_000000\n\
            OpSelectionMerge %merge None\n\
            OpBranchConditional %t46_0 %then %merge\n\
            %then = OpLabel\n\
            %t47_0 = OpExtInst %float %GLSL_std_450 FMin %t45_0 %float_1_000000\n\
            OpBranch %merge\n\
            %merge = OpLabel\n\
            %t48_0 = OpPhi %float %t45_0 %5 %t47_0 %then\n\
            %t49_0 = OpCompositeConstruct %v4float %t48_0 %t48_0 %t48_0 %float_1_000000\n\
            OpStore %outColor %t49_0\n\
            OpReturn\n\
            OpFunctionEnd\n";

    #[test]
    fn realistic_fragment_shader_assembles() {
        let words = assemble(FRAGMENT).unwrap();
        assert_eq!(words[0], SPIRV_MAGIC);
        assert_eq!(words[1], SPIRV_VERSION);
        assert_eq!(words[2], 0); // generator
        assert_eq!(words[4], 0); // schema
        check_word_counts(&words);
        // 32 symbolic ids assigned sequentially while skipping the numeric
        // ids %3 and %5 → the 32nd symbolic id is 34; bound = 34 + 1.
        assert_eq!(words[3], 35);
        // Every referenced id is defined — assemble() errors otherwise, so
        // reaching this point covers that check.
    }

    // --- 7. Round-trip vs hand-built words -----------------------------------

    #[test]
    fn round_trip_hand_built_words() {
        let text = "OpCapability Shader\n\
                    %u32 = OpTypeInt 32 0\n\
                    %c = OpConstant %u32 42\n";
        let words = assemble(text).unwrap();
        let expected = vec![
            SPIRV_MAGIC,
            SPIRV_VERSION,
            0,
            3, // bound: u32=1, c=2
            0,
            (2 << 16) | 17,
            1, // Shader
            (4 << 16) | 21,
            1,
            32,
            0,
            (4 << 16) | 43,
            1,
            2,
            42,
        ];
        assert_eq!(words, expected);
    }

    // --- Errors ---------------------------------------------------------------

    #[test]
    fn unknown_opcode_and_enum_errors() {
        let e = assemble("OpFrobnicate %x\n").unwrap_err();
        assert_eq!(e.line, 1);
        assert!(e.message.contains("OpFrobnicate"));

        let e = assemble("OpCapability NotACap\n").unwrap_err();
        assert!(e.message.contains("NotACap"));

        let e = assemble("%p = OpTypePointer NotAClass %p\n").unwrap_err();
        assert!(e.message.contains("NotAClass"));
    }

    #[test]
    fn malformed_instruction_errors() {
        let e = assemble("%x OpTypeVoid\n").unwrap_err();
        assert_eq!(e.line, 1);
        assert!(e.message.contains("malformed"));

        let e = assemble("= OpTypeVoid\n").unwrap_err();
        assert!(e.message.contains("malformed"));

        // Missing required result id.
        let e = assemble("OpTypeVoid\n").unwrap_err();
        assert!(e.message.contains("requires a result id"));

        // Extra operand.
        let e = assemble("%v = OpTypeVoid %v\n").unwrap_err();
        assert!(e.message.contains("unexpected extra operand"));
    }

    // --- 8. naga validity gate --------------------------------------------------

    #[test]
    fn naga_parses_minimal_module() {
        let words = assemble(MINIMAL).unwrap();
        let bytes = words_to_bytes(&words);
        let module =
            naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default());
        assert!(
            module.is_ok(),
            "naga rejected minimal module: {:?}",
            module.err()
        );
    }

    #[test]
    fn naga_parses_realistic_fragment_shader() {
        let words = assemble(FRAGMENT).unwrap();
        let bytes = words_to_bytes(&words);
        let module =
            naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default());
        assert!(
            module.is_ok(),
            "naga rejected fragment shader: {:?}",
            module.err()
        );
    }
}
