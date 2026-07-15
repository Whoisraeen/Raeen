//! SPIR-V source generator (`Spirv` class), ported from Kyty
//! (MIT (c) InoriRus).
//!
//! Kyty source: `emulator/src/Graphics/ShaderSpirv.cpp`:
//! - embedded helper-function texts (`FUNC_FETCH_*` L36-131,
//!   `BUFFER_LOAD_FLOAT1/4` L569/L598, `TBUFFER_LOAD_FORMAT_XYZW/X`
//!   L715/L772, `SBUFFER_LOAD_DWORD[_2/_4]` L912/L936/L968)
//! - embedded shaders `EMBEDDED_SHADER_VS_0` L1244 / `EMBEDDED_SHADER_PS_0`
//!   L1327
//! - `SpirvType`/`SpirvValue` L1445/L1453, class `Spirv` L1459
//! - operand helpers `operand_is_constant` L1564, `operand_is_variable`
//!   L1570, `operand_variable_to_str` L1577 (+ shift overload L1627),
//!   `operand_is_exec` L1671, `operand_load_int/uint/float`
//!   L1683/L1723/L1791
//! - `Spirv::AddConstant*`/`GetConstant*` L6467-6650, `GenerateSource` L6652,
//!   `WriteHeader` L6685, `WriteDebug` L6801, `WriteAnnotations` L6814,
//!   `WriteTypes` L6967, `WriteConstants` L7133, `WriteGlobalVariables`
//!   L7152, `WriteMainProlog` L7258, `WriteLocalVariables` L7271,
//!   `WriteLabel` L7553, `ModifyCode` L7592, `DetectFetch` L7639,
//!   `WriteInstructions` L7797, `WriteMainEpilog` L7841, `WriteFunctions`
//!   L7851, `FindConstants` L7940, `FindVariables` L8001,
//!   `SpirvGenerateSource` L8074, `SpirvGetEmbeddedVs/Ps` L8087/L8094.
//!
//! Deviations (documented per project conventions):
//! - Kyty `EXIT`/`EXIT_NOT_IMPLEMENTED` aborts become the typed
//!   [`ShaderRecompileError`]; every error is also logged loudly.
//! - `Config::SpirvDebugPrintfEnabled()` (a Kyty global) becomes the
//!   `debug_printf_enabled` field (default `false`) — no globals in the port.
//! - `WriteFunctions` only embeds the helper-function texts needed by the C1
//!   instruction subset. The remaining Kyty branches (`FUNC_ABS_DIFF` L133,
//!   `FUNC_WQM` L149, `FUNC_ADDC` L167, `FUNC_LSHL_ADD` L200, `FUNC_MIPMAP`
//!   L225, `FUNC_ORDERED` L304, `FUNC_MUL_EXTENDED` L338, `FUNC_SHIFT_RIGHT`
//!   L397, `FUNC_SHIFT_LEFT` L483, `BUFFER_STORE_FLOAT1/2` L650/L679,
//!   `TBUFFER_STORE_FORMAT_X/XY` L817/L862, `SBUFFER_LOAD_DWORD_8/16`
//!   L1016/L1097) guard instructions whose recompile entries are still
//!   `NotImplemented` in C1, so `WriteInstructions` errors before those
//!   texts could ever be needed. C2 adds them.

use std::fmt;

use crate::shader::resources::{
    ShaderBindResources, ShaderComputeInputInfo, ShaderPixelInputInfo, ShaderVertexInputInfo,
};
use crate::shader::types::{
    ShaderCode, ShaderConstant, ShaderInstruction, ShaderInstructionType, ShaderOperand,
    ShaderOperandType, ShaderType, shader_instruction_format::Format,
};

/// Typed replacement for Kyty's hard `EXIT` / `EXIT_NOT_IMPLEMENTED` aborts
/// in the recompiler (ShaderSpirv.cpp / Shader.cpp `SpirvRun` path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderRecompileError {
    /// No `g_recomp_func` table entry for this (type, format) pair —
    /// Kyty: `EXIT("Can't recompile: ...")` (ShaderSpirv.cpp L7828).
    UnknownTypeFormat {
        type_: ShaderInstructionType,
        format: Format,
        instruction: String,
    },
    /// Table entry exists but its `Recompile_*` function is not ported yet
    /// (C1 sub-batch limit). Carries the Kyty function name + line anchor.
    NotImplemented {
        kyty_func: &'static str,
        line: u32,
        instruction: String,
    },
    /// The recompile function returned `false` —
    /// Kyty: `EXIT("Can't recompile: ...")` (ShaderSpirv.cpp L7828).
    CannotRecompile { instruction: String },
    /// A Kyty `EXIT_NOT_IMPLEMENTED`/`EXIT_IF` condition fired.
    NotSupported { func: &'static str, message: String },
    /// Kyty: `EXIT("unknown type: ...")` in WriteHeader et al.
    UnknownShaderType,
    /// SPIR-V assembly failed (Kyty: `SpirvRun` Assemble failure,
    /// Shader.cpp L873).
    Asm(crate::spirv_asm::AsmError),
}

impl fmt::Display for ShaderRecompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTypeFormat {
                type_,
                format,
                instruction,
            } => write!(
                f,
                "can't recompile (no recompiler for {type_:?}/{format:?}): {instruction}"
            ),
            Self::NotImplemented {
                kyty_func,
                line,
                instruction,
            } => write!(
                f,
                "recompiler {kyty_func} (ShaderSpirv.cpp L{line}) not ported yet (C2): \
                 {instruction}"
            ),
            Self::CannotRecompile { instruction } => {
                write!(f, "can't recompile: {instruction}")
            }
            Self::NotSupported { func, message } => {
                write!(f, "{func}: not supported: {message}")
            }
            Self::UnknownShaderType => write!(f, "unknown shader type"),
            Self::Asm(e) => write!(f, "SPIR-V assembly failed: {e}"),
        }
    }
}

impl std::error::Error for ShaderRecompileError {}

impl From<crate::spirv_asm::AsmError> for ShaderRecompileError {
    fn from(e: crate::spirv_asm::AsmError) -> Self {
        Self::Asm(e)
    }
}

pub(crate) fn not_supported(
    func: &'static str,
    message: impl Into<String>,
) -> ShaderRecompileError {
    let message = message.into();
    tracing::error!("{func}: not supported: {message}");
    ShaderRecompileError::NotSupported { func, message }
}

// ---------------------------------------------------------------------------
// Embedded helper functions (SPIR-V assembly text, verbatim from Kyty)
// ---------------------------------------------------------------------------

/// Kyty: ShaderSpirv.cpp `FUNC_FETCH_4` (L36).
pub(crate) const FUNC_FETCH_4: &str = r#"
       ; Function fetch_f1_f1_f1_f1_vf4_
       ; void fetch(out float p1, out float p2, out float p3, out float p4, in vec4 attr)
       ; {
       ; p1 = attr.x;
       ; p2 = attr.y;
       ; p3 = attr.z;
       ; p4 = attr.w;
       ; }
%fetch_f1_f1_f1_f1_vf4_ = OpFunction %void None %function_fetch4
 %fetch_p1 = OpFunctionParameter %_ptr_Function_float
 %fetch_p2 = OpFunctionParameter %_ptr_Function_float
 %fetch_p3 = OpFunctionParameter %_ptr_Function_float
 %fetch_p4 = OpFunctionParameter %_ptr_Function_float
%fetch_attr = OpFunctionParameter %_ptr_Function_v4float
%fetch_label = OpLabel
 %fetch_20 = OpAccessChain %_ptr_Function_float %fetch_attr %uint_0
 %fetch_21 = OpLoad %float %fetch_20
             OpStore %fetch_p1 %fetch_21
 %fetch_23 = OpAccessChain %_ptr_Function_float %fetch_attr %uint_1
 %fetch_24 = OpLoad %float %fetch_23
             OpStore %fetch_p2 %fetch_24
 %fetch_26 = OpAccessChain %_ptr_Function_float %fetch_attr %uint_2
 %fetch_27 = OpLoad %float %fetch_26
             OpStore %fetch_p3 %fetch_27
 %fetch_29 = OpAccessChain %_ptr_Function_float %fetch_attr %uint_3
 %fetch_30 = OpLoad %float %fetch_29
             OpStore %fetch_p4 %fetch_30
             OpReturn
             OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_FETCH_3` (L68).
pub(crate) const FUNC_FETCH_3: &str = r#"
       ; Function fetch_f1_f1_f1_vf3_
       ; void fetch(out float p1, out float p2, out float p3, in vec3 attr)
       ; {
       ; p1 = attr.x;
       ; p2 = attr.y;
       ; p3 = attr.z;
       ; }
%fetch_f1_f1_f1_vf3_ = OpFunction %void None %function_fetch3
       %fetch3_p1_0 = OpFunctionParameter %_ptr_Function_float
       %fetch3_p2_0 = OpFunctionParameter %_ptr_Function_float
       %fetch3_p3_0 = OpFunctionParameter %_ptr_Function_float
     %fetch3_attr_0 = OpFunctionParameter %_ptr_Function_v3float
         %fetch3_26 = OpLabel
         %fetch3_53 = OpAccessChain %_ptr_Function_float %fetch3_attr_0 %uint_0
         %fetch3_54 = OpLoad %float %fetch3_53
               OpStore %fetch3_p1_0 %fetch3_54
         %fetch3_55 = OpAccessChain %_ptr_Function_float %fetch3_attr_0 %uint_1
         %fetch3_56 = OpLoad %float %fetch3_55
               OpStore %fetch3_p2_0 %fetch3_56
         %fetch3_57 = OpAccessChain %_ptr_Function_float %fetch3_attr_0 %uint_2
         %fetch3_58 = OpLoad %float %fetch3_57
               OpStore %fetch3_p3_0 %fetch3_58
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_FETCH_2` (L95).
pub(crate) const FUNC_FETCH_2: &str = r#"
       ; Function fetch_f1_f1_vf2_
       ; void fetch(out float p1, out float p2, in vec2 attr)
       ; {
       ; p1 = attr.x;
       ; p2 = attr.y;
       ; }
%fetch_f1_f1_vf2_ = OpFunction %void None %function_fetch2
       %fetch2_p1_1 = OpFunctionParameter %_ptr_Function_float
       %fetch2_p2_1 = OpFunctionParameter %_ptr_Function_float
     %fetch2_attr_1 = OpFunctionParameter %_ptr_Function_v2float
         %fetch2_34 = OpLabel
         %fetch2_59 = OpAccessChain %_ptr_Function_float %fetch2_attr_1 %uint_0
         %fetch2_60 = OpLoad %float %fetch2_59
               OpStore %fetch2_p1_1 %fetch2_60
         %fetch2_61 = OpAccessChain %_ptr_Function_float %fetch2_attr_1 %uint_1
         %fetch2_62 = OpLoad %float %fetch2_61
               OpStore %fetch2_p2_1 %fetch2_62
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_FETCH_1` (L117).
pub(crate) const FUNC_FETCH_1: &str = r#"
       ; Function fetch_f1_f1_
       ; void fetch(out float p1, in float attr)
       ; {
       ; p1 = attr;
       ; }
%fetch_f1_f1_ = OpFunction %void None %function_fetch1
       %fetch1_p1_2 = OpFunctionParameter %_ptr_Function_float
     %fetch1_attr_2 = OpFunctionParameter %_ptr_Function_float
         %fetch1_39 = OpLabel
         %fetch1_63 = OpLoad %float %fetch1_attr_2
               OpStore %fetch1_p1_2 %fetch1_63
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `BUFFER_LOAD_FLOAT1` (L569).
pub(crate) const BUFFER_LOAD_FLOAT1: &str = r#"
             ; void buffer_load_float1(out float p1, in int index, in int offset, in int stride, in int buffer_index)
             ; {
             ; 	int addr = (offset + index * stride)/4;
             ; 	p1 = buf[buffer_index].data[addr+0];
             ; }
%buffer_load_float1 = OpFunction %void None %function_buffer_load_store_float1
         %buf_l_f1_11 = OpFunctionParameter %_ptr_Function_float
         %buf_l_f1_12 = OpFunctionParameter %_ptr_Function_int
         %buf_l_f1_13 = OpFunctionParameter %_ptr_Function_int
         %buf_l_f1_14 = OpFunctionParameter %_ptr_Function_int
         %buf_l_f1_15 = OpFunctionParameter %_ptr_Function_int
         %buf_l_f1_17 = OpLabel
         %buf_l_f1_42 = OpVariable %_ptr_Function_int Function
         %buf_l_f1_43 = OpLoad %int %buf_l_f1_13
         %buf_l_f1_44 = OpLoad %int %buf_l_f1_12
         %buf_l_f1_45 = OpLoad %int %buf_l_f1_14
         %buf_l_f1_46 = OpIMul %int %buf_l_f1_44 %buf_l_f1_45
         %buf_l_f1_47 = OpIAdd %int %buf_l_f1_43 %buf_l_f1_46
         %buf_l_f1_49 = OpSDiv %int %buf_l_f1_47 %int_4
               OpStore %buf_l_f1_42 %buf_l_f1_49
         %buf_l_f1_57 = OpLoad %int %buf_l_f1_15
         %buf_l_f1_62 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_l_f1_57 %int_0 %buf_l_f1_49
         %buf_l_f1_63 = OpLoad %float %buf_l_f1_62
               OpStore %buf_l_f1_11 %buf_l_f1_63
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `BUFFER_LOAD_FLOAT4` (L598).
pub(crate) const BUFFER_LOAD_FLOAT4: &str = r#"
             ; Function buffer_load_float4
             ;void buffer_load_float4(out float p1, out float p2, out float p3, out float p4, in int index,
             ;                                in int offset, in int stride, in int buffer_index)
             ;{
             ;	int addr = (offset + index * stride)/4;
             ;	p1 = buf[buffer_index].data[addr+0];
             ;	p2 = buf[buffer_index].data[addr+1];
             ;	p3 = buf[buffer_index].data[addr+2];
             ;	p4 = buf[buffer_index].data[addr+3];
             ;}
%buffer_load_float4 = OpFunction %void None %function_buffer_load_float4
  %buf_l_f4_21 = OpFunctionParameter %_ptr_Function_float
  %buf_l_f4_22 = OpFunctionParameter %_ptr_Function_float
  %buf_l_f4_23 = OpFunctionParameter %_ptr_Function_float
  %buf_l_f4_24 = OpFunctionParameter %_ptr_Function_float
  %buf_l_f4_25 = OpFunctionParameter %_ptr_Function_int
  %buf_l_f4_26 = OpFunctionParameter %_ptr_Function_int
  %buf_l_f4_27 = OpFunctionParameter %_ptr_Function_int
  %buf_l_f4_28 = OpFunctionParameter %_ptr_Function_int
  %buf_l_f4_30 = OpLabel
  %buf_l_f4_44 = OpVariable %_ptr_Function_int Function
  %buf_l_f4_45 = OpLoad %int %buf_l_f4_26
  %buf_l_f4_46 = OpLoad %int %buf_l_f4_25
  %buf_l_f4_47 = OpLoad %int %buf_l_f4_27
  %buf_l_f4_48 = OpIMul %int %buf_l_f4_46 %buf_l_f4_47
  %buf_l_f4_49 = OpIAdd %int %buf_l_f4_45 %buf_l_f4_48
  %buf_l_f4_51 = OpSDiv %int %buf_l_f4_49 %int_4
        OpStore %buf_l_f4_44 %buf_l_f4_51
  %buf_l_f4_58 = OpLoad %int %buf_l_f4_28
  %buf_l_f4_63 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_l_f4_58 %int_0 %buf_l_f4_51
  %buf_l_f4_64 = OpLoad %float %buf_l_f4_63
        OpStore %buf_l_f4_21 %buf_l_f4_64
  %buf_l_f4_65 = OpLoad %int %buf_l_f4_28
  %buf_l_f4_68 = OpIAdd %int %buf_l_f4_51 %int_1
  %buf_l_f4_69 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_l_f4_65 %int_0 %buf_l_f4_68
  %buf_l_f4_70 = OpLoad %float %buf_l_f4_69
        OpStore %buf_l_f4_22 %buf_l_f4_70
  %buf_l_f4_71 = OpLoad %int %buf_l_f4_28
  %buf_l_f4_74 = OpIAdd %int %buf_l_f4_51 %int_2
  %buf_l_f4_75 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_l_f4_71 %int_0 %buf_l_f4_74
  %buf_l_f4_76 = OpLoad %float %buf_l_f4_75
        OpStore %buf_l_f4_23 %buf_l_f4_76
  %buf_l_f4_77 = OpLoad %int %buf_l_f4_28
  %buf_l_f4_80 = OpIAdd %int %buf_l_f4_51 %int_3
  %buf_l_f4_81 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_l_f4_77 %int_0 %buf_l_f4_80
  %buf_l_f4_82 = OpLoad %float %buf_l_f4_81
        OpStore %buf_l_f4_24 %buf_l_f4_82
        OpReturn
        OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `TBUFFER_LOAD_FORMAT_XYZW` (L715).
pub(crate) const TBUFFER_LOAD_FORMAT_XYZW: &str = r#"
             ; Function tbuffer_load_format_xyzw
             ; void tbuffer_load_format_xyzw(out float p1, out float p2, out float p3, out float p4,
             ;                               in int index, in int offset, in int stride, in int buffer_index, in int dfmt_nfmt)
             ; {
             ; 	if (dfmt_nfmt == 119) // dfmt = 14, nfmt = 7
             ; 	{
             ; 		buffer_load_float4(p1, p2, p3, p4, index, offset, stride, buffer_index);
             ; 	}
             ; }
%tbuffer_load_format_xyzw = OpFunction %void None %function_tbuffer_load_format_xyzw
%tbuf_l_f_xyzw_54 = OpFunctionParameter %_ptr_Function_float
%tbuf_l_f_xyzw_55 = OpFunctionParameter %_ptr_Function_float
%tbuf_l_f_xyzw_56 = OpFunctionParameter %_ptr_Function_float
%tbuf_l_f_xyzw_57 = OpFunctionParameter %_ptr_Function_float
%tbuf_l_f_xyzw_58 = OpFunctionParameter %_ptr_Function_int
%tbuf_l_f_xyzw_59 = OpFunctionParameter %_ptr_Function_int
%tbuf_l_f_xyzw_60 = OpFunctionParameter %_ptr_Function_int
%tbuf_l_f_xyzw_61 = OpFunctionParameter %_ptr_Function_int
%tbuf_l_f_xyzw_62 = OpFunctionParameter %_ptr_Function_int
%tbuf_l_f_xyzw_64 = OpLabel
%tbuf_l_f_xyzw_166 = OpVariable %_ptr_Function_float Function
%tbuf_l_f_xyzw_167 = OpVariable %_ptr_Function_float Function
%tbuf_l_f_xyzw_168 = OpVariable %_ptr_Function_float Function
%tbuf_l_f_xyzw_169 = OpVariable %_ptr_Function_float Function
%tbuf_l_f_xyzw_170 = OpVariable %_ptr_Function_int Function
%tbuf_l_f_xyzw_172 = OpVariable %_ptr_Function_int Function
%tbuf_l_f_xyzw_174 = OpVariable %_ptr_Function_int Function
%tbuf_l_f_xyzw_176 = OpVariable %_ptr_Function_int Function
%tbuf_l_f_xyzw_161 = OpLoad %int %tbuf_l_f_xyzw_62
%tbuf_l_f_xyzw_163 = OpIEqual %bool %tbuf_l_f_xyzw_161 %int_119
   OpSelectionMerge %tbuf_l_f_xyzw_165 None
   OpBranchConditional %tbuf_l_f_xyzw_163 %tbuf_l_f_xyzw_164 %tbuf_l_f_xyzw_165
%tbuf_l_f_xyzw_164 = OpLabel
%tbuf_l_f_xyzw_171 = OpLoad %int %tbuf_l_f_xyzw_58
   OpStore %tbuf_l_f_xyzw_170 %tbuf_l_f_xyzw_171
%tbuf_l_f_xyzw_173 = OpLoad %int %tbuf_l_f_xyzw_59
   OpStore %tbuf_l_f_xyzw_172 %tbuf_l_f_xyzw_173
%tbuf_l_f_xyzw_175 = OpLoad %int %tbuf_l_f_xyzw_60
   OpStore %tbuf_l_f_xyzw_174 %tbuf_l_f_xyzw_175
%tbuf_l_f_xyzw_177 = OpLoad %int %tbuf_l_f_xyzw_61
   OpStore %tbuf_l_f_xyzw_176 %tbuf_l_f_xyzw_177
%tbuf_l_f_xyzw_178 = OpFunctionCall %void %buffer_load_float4 %tbuf_l_f_xyzw_166 %tbuf_l_f_xyzw_167 %tbuf_l_f_xyzw_168 %tbuf_l_f_xyzw_169 %tbuf_l_f_xyzw_170 %tbuf_l_f_xyzw_172 %tbuf_l_f_xyzw_174 %tbuf_l_f_xyzw_176
%tbuf_l_f_xyzw_179 = OpLoad %float %tbuf_l_f_xyzw_166
   OpStore %tbuf_l_f_xyzw_54 %tbuf_l_f_xyzw_179
%tbuf_l_f_xyzw_180 = OpLoad %float %tbuf_l_f_xyzw_167
   OpStore %tbuf_l_f_xyzw_55 %tbuf_l_f_xyzw_180
%tbuf_l_f_xyzw_181 = OpLoad %float %tbuf_l_f_xyzw_168
   OpStore %tbuf_l_f_xyzw_56 %tbuf_l_f_xyzw_181
%tbuf_l_f_xyzw_182 = OpLoad %float %tbuf_l_f_xyzw_169
   OpStore %tbuf_l_f_xyzw_57 %tbuf_l_f_xyzw_182
   OpBranch %tbuf_l_f_xyzw_165
%tbuf_l_f_xyzw_165 = OpLabel
   OpReturn
   OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `TBUFFER_LOAD_FORMAT_X` (L772).
pub(crate) const TBUFFER_LOAD_FORMAT_X: &str = r#"
             ; void tbuffer_load_format_x(out float p1, in int index, in int offset, in int stride, in int buffer_index, in int dfmt_nfmt)
             ; {
             ; 	if (dfmt_nfmt == 36 || dfmt_nfmt == 39) // dmft = 4, nfmt = 4 or 7
             ; 	{
             ; 		buffer_load_float1(p1, index, offset, stride, buffer_index);
             ; 	}
             ; }
%tbuffer_load_format_x = OpFunction %void None %function_tbuffer_load_store_format_x
         %tbuf_l_f_x_26 = OpFunctionParameter %_ptr_Function_float
         %tbuf_l_f_x_27 = OpFunctionParameter %_ptr_Function_int
         %tbuf_l_f_x_28 = OpFunctionParameter %_ptr_Function_int
         %tbuf_l_f_x_29 = OpFunctionParameter %_ptr_Function_int
         %tbuf_l_f_x_30 = OpFunctionParameter %_ptr_Function_int
         %tbuf_l_f_x_31 = OpFunctionParameter %_ptr_Function_int
         %tbuf_l_f_x_33 = OpLabel
         %tbuf_l_f_x_82 = OpVariable %_ptr_Function_float Function
         %tbuf_l_f_x_83 = OpVariable %_ptr_Function_int Function
         %tbuf_l_f_x_85 = OpVariable %_ptr_Function_int Function
         %tbuf_l_f_x_87 = OpVariable %_ptr_Function_int Function
         %tbuf_l_f_x_89 = OpVariable %_ptr_Function_int Function
         %tbuf_l_f_x_76 = OpLoad %int %tbuf_l_f_x_31
         %tbuf_l_f_x_79 = OpIEqual %bool %tbuf_l_f_x_76 %int_36
         %tbuf_l_f_x_79_2 = OpIEqual %bool %tbuf_l_f_x_76 %int_39
         %tbuf_l_f_x_79_3 = OpLogicalOr %bool %tbuf_l_f_x_79 %tbuf_l_f_x_79_2
               OpSelectionMerge %tbuf_l_f_x_81 None
               OpBranchConditional %tbuf_l_f_x_79_3 %tbuf_l_f_x_80 %tbuf_l_f_x_81
         %tbuf_l_f_x_80 = OpLabel
         %tbuf_l_f_x_84 = OpLoad %int %tbuf_l_f_x_27
               OpStore %tbuf_l_f_x_83 %tbuf_l_f_x_84
         %tbuf_l_f_x_86 = OpLoad %int %tbuf_l_f_x_28
               OpStore %tbuf_l_f_x_85 %tbuf_l_f_x_86
         %tbuf_l_f_x_88 = OpLoad %int %tbuf_l_f_x_29
               OpStore %tbuf_l_f_x_87 %tbuf_l_f_x_88
         %tbuf_l_f_x_90 = OpLoad %int %tbuf_l_f_x_30
               OpStore %tbuf_l_f_x_89 %tbuf_l_f_x_90
         %tbuf_l_f_x_91 = OpFunctionCall %void %buffer_load_float1 %tbuf_l_f_x_82 %tbuf_l_f_x_83 %tbuf_l_f_x_85 %tbuf_l_f_x_87 %tbuf_l_f_x_89
         %tbuf_l_f_x_92 = OpLoad %float %tbuf_l_f_x_82
               OpStore %tbuf_l_f_x_26 %tbuf_l_f_x_92
               OpBranch %tbuf_l_f_x_81
         %tbuf_l_f_x_81 = OpLabel
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `SBUFFER_LOAD_DWORD` (L912).
pub(crate) const SBUFFER_LOAD_DWORD: &str = r#"
                     ; void sbuffer_load_dword(out uint p1, in int offset, in int buffer_index)
                     ; {
                     ; 	int addr = offset/4;
                     ; 	p1 = floatBitsToUint(buf[buffer_index].data[addr+0]);
                     ; }
%sbuffer_load_dword = OpFunction %void None %function_sbuffer_load_dword
         %sbuf_dw_45 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw_46 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw_47 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw_49 = OpLabel
        %sbuf_dw_115 = OpVariable %_ptr_Function_int Function
        %sbuf_dw_116 = OpLoad %int %sbuf_dw_46
        %sbuf_dw_117 = OpSDiv %int %sbuf_dw_116 %int_4
               OpStore %sbuf_dw_115 %sbuf_dw_117
        %sbuf_dw_118 = OpLoad %int %sbuf_dw_47
        %sbuf_dw_121 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw_118 %int_0 %sbuf_dw_117
        %sbuf_dw_122 = OpLoad %float %sbuf_dw_121
        %sbuf_dw_123 = OpBitcast %uint %sbuf_dw_122
               OpStore %sbuf_dw_45 %sbuf_dw_123
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `SBUFFER_LOAD_DWORD_2` (L936).
pub(crate) const SBUFFER_LOAD_DWORD_2: &str = r#"
                      ; void sbuffer_load_dwordx2(out uint p1, out uint p2, in int offset, in int buffer_index)
                      ; {
                      ; 	int addr = offset/4;
                      ; 	p1 = floatBitsToUint(buf[buffer_index].data[addr+0]);
                      ; 	p2 = floatBitsToUint(buf[buffer_index].data[addr+1]);
                      ; }
%sbuffer_load_dword_2 = OpFunction %void None %function_sbuffer_load_dword_2
         %sbuf_dw2_11 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw2_12 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw2_13 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw2_14 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw2_16 = OpLabel
         %sbuf_dw2_17 = OpVariable %_ptr_Function_int Function
         %sbuf_dw2_18 = OpLoad %int %sbuf_dw2_13
         %sbuf_dw2_20 = OpSDiv %int %sbuf_dw2_18 %int_4
               OpStore %sbuf_dw2_17 %sbuf_dw2_20
         %sbuf_dw2_28 = OpLoad %int %sbuf_dw2_14
         %sbuf_dw2_33 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw2_28 %int_0 %sbuf_dw2_20
         %sbuf_dw2_34 = OpLoad %float %sbuf_dw2_33
         %sbuf_dw2_35 = OpBitcast %uint %sbuf_dw2_34
               OpStore %sbuf_dw2_11 %sbuf_dw2_35
         %sbuf_dw2_36 = OpLoad %int %sbuf_dw2_14
         %sbuf_dw2_39 = OpIAdd %int %sbuf_dw2_20 %int_1
         %sbuf_dw2_40 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw2_36 %int_0 %sbuf_dw2_39
         %sbuf_dw2_41 = OpLoad %float %sbuf_dw2_40
         %sbuf_dw2_42 = OpBitcast %uint %sbuf_dw2_41
               OpStore %sbuf_dw2_12 %sbuf_dw2_42
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `SBUFFER_LOAD_DWORD_4` (L968).
pub(crate) const SBUFFER_LOAD_DWORD_4: &str = r#"
                     ; void sbuffer_load_dwordx4(out uint p1, out uint p2, out uint p3, out uint p4, in int offset, in int buffer_index)
                     ; {
                     ; 	int addr = offset/4;
                     ; 	p1 = floatBitsToUint(buf[buffer_index].data[addr+0]);
                     ; 	p2 = floatBitsToUint(buf[buffer_index].data[addr+1]);
                     ; 	p3 = floatBitsToUint(buf[buffer_index].data[addr+2]);
                     ; 	p4 = floatBitsToUint(buf[buffer_index].data[addr+3]);
                     ; }
%sbuffer_load_dword_4 = OpFunction %void None %function_sbuffer_load_dword_4
         %sbuf_dw4_51 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw4_52 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw4_53 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw4_54 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw4_55 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw4_56 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw4_58 = OpLabel
        %sbuf_dw4_133 = OpVariable %_ptr_Function_int Function
        %sbuf_dw4_134 = OpLoad %int %sbuf_dw4_55
        %sbuf_dw4_135 = OpSDiv %int %sbuf_dw4_134 %int_4
               OpStore %sbuf_dw4_133 %sbuf_dw4_135
        %sbuf_dw4_136 = OpLoad %int %sbuf_dw4_56
        %sbuf_dw4_139 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw4_136 %int_0 %sbuf_dw4_135
        %sbuf_dw4_140 = OpLoad %float %sbuf_dw4_139
        %sbuf_dw4_141 = OpBitcast %uint %sbuf_dw4_140
               OpStore %sbuf_dw4_51 %sbuf_dw4_141
        %sbuf_dw4_142 = OpLoad %int %sbuf_dw4_56
        %sbuf_dw4_145 = OpIAdd %int %sbuf_dw4_135 %int_1
        %sbuf_dw4_146 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw4_142 %int_0 %sbuf_dw4_145
        %sbuf_dw4_147 = OpLoad %float %sbuf_dw4_146
        %sbuf_dw4_148 = OpBitcast %uint %sbuf_dw4_147
               OpStore %sbuf_dw4_52 %sbuf_dw4_148
        %sbuf_dw4_149 = OpLoad %int %sbuf_dw4_56
        %sbuf_dw4_152 = OpIAdd %int %sbuf_dw4_135 %int_2
        %sbuf_dw4_153 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw4_149 %int_0 %sbuf_dw4_152
        %sbuf_dw4_154 = OpLoad %float %sbuf_dw4_153
        %sbuf_dw4_155 = OpBitcast %uint %sbuf_dw4_154
               OpStore %sbuf_dw4_53 %sbuf_dw4_155
        %sbuf_dw4_156 = OpLoad %int %sbuf_dw4_56
        %sbuf_dw4_159 = OpIAdd %int %sbuf_dw4_135 %int_3
        %sbuf_dw4_160 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw4_156 %int_0 %sbuf_dw4_159
        %sbuf_dw4_161 = OpLoad %float %sbuf_dw4_160
        %sbuf_dw4_162 = OpBitcast %uint %sbuf_dw4_161
               OpStore %sbuf_dw4_54 %sbuf_dw4_162
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `EMBEDDED_SHADER_VS_0` (L1244).
pub(crate) const EMBEDDED_SHADER_VS_0: &str = r#"
               ; #version 450
               ;
               ; void main()
               ; {
               ; 	float x = gl_VertexIndex == 0 || gl_VertexIndex == 2 ? 1.0 : -1.0;
               ; 	float y = gl_VertexIndex == 2 || gl_VertexIndex == 3 ? -1.0 : 1.0;
               ;
               ;     gl_Position = vec4(x,y, 0.0, 1.0);
               ; }

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %4 "main" %gl_VertexIndex %43

               ; Annotations
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpMemberDecorate %_struct_41 0 BuiltIn Position
               OpMemberDecorate %_struct_41 1 BuiltIn PointSize
               OpMemberDecorate %_struct_41 2 BuiltIn ClipDistance
               OpMemberDecorate %_struct_41 3 BuiltIn CullDistance
               OpDecorate %_struct_41 Block

               ; Types, variables and constants
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
       %bool = OpTypeBool
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_VertexIndex = OpVariable %_ptr_Input_int Input
      %int_0 = OpConstant %int 0
      %int_2 = OpConstant %int 2
    %float_1 = OpConstant %float 1
   %float_n1 = OpConstant %float -1
      %int_3 = OpConstant %int 3
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
%_arr_float_uint_1 = OpTypeArray %float %uint_1
 %_struct_41 = OpTypeStruct %v4float %float %_arr_float_uint_1 %_arr_float_uint_1
%_ptr_Output__struct_41 = OpTypePointer Output %_struct_41
         %43 = OpVariable %_ptr_Output__struct_41 Output
    %float_0 = OpConstant %float 0
%_ptr_Output_v4float = OpTypePointer Output %v4float

               ; Function 4
          %4 = OpFunction %void None %3
          %5 = OpLabel
          %8 = OpVariable %_ptr_Function_float Function
         %26 = OpVariable %_ptr_Function_float Function
         %13 = OpLoad %int %gl_VertexIndex
         %15 = OpIEqual %bool %13 %int_0
         %16 = OpLogicalNot %bool %15
               OpSelectionMerge %18 None
               OpBranchConditional %16 %17 %18
         %17 = OpLabel
         %21 = OpIEqual %bool %13 %int_2
               OpBranch %18
         %18 = OpLabel
         %22 = OpPhi %bool %15 %5 %21 %17
         %25 = OpSelect %float %22 %float_1 %float_n1
               OpStore %8 %25
         %28 = OpIEqual %bool %13 %int_2
         %29 = OpLogicalNot %bool %28
               OpSelectionMerge %31 None
               OpBranchConditional %29 %30 %31
         %30 = OpLabel
         %34 = OpIEqual %bool %13 %int_3
               OpBranch %31
         %31 = OpLabel
         %35 = OpPhi %bool %28 %18 %34 %30
         %36 = OpSelect %float %35 %float_n1 %float_1
               OpStore %26 %36
         %47 = OpCompositeConstruct %v4float %25 %36 %float_0 %float_1
         %49 = OpAccessChain %_ptr_Output_v4float %43 %int_0
               OpStore %49 %47
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `EMBEDDED_SHADER_PS_0` (L1327).
pub(crate) const EMBEDDED_SHADER_PS_0: &str = r#"
               ; #version 450
               ;
               ; layout(location = 0) out vec4 outColor;
               ;
               ; void main() {
               ; 	outColor = vec4(0);
               ; }

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %9
               OpExecutionMode %4 OriginUpperLeft

               ; Annotations
               OpDecorate %9 Location 0

               ; Types, variables and constants
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
          %9 = OpVariable %_ptr_Output_v4float Output
    %float_0 = OpConstant %float 0
         %11 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0

               ; Function 4
          %4 = OpFunction %void None %3
          %5 = OpLabel
               OpStore %9 %11
               OpReturn
               OpFunctionEnd
"#;

// ---------------------------------------------------------------------------
// Value model + operand helpers
// ---------------------------------------------------------------------------

/// Kyty: ShaderSpirv.cpp `SpirvType` (L1445).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum SpirvType {
    #[default]
    Unknown,
    Float,
    Int,
    Uint,
}

impl SpirvType {
    /// Kyty: `Core::EnumName(type).ToLower()`.
    #[must_use]
    pub const fn to_lower_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Float => "float",
            Self::Int => "int",
            Self::Uint => "uint",
        }
    }
}

/// Kyty: ShaderSpirv.cpp `SpirvValue` (L1453).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpirvValue {
    pub type_: SpirvType,
    pub value: String,
}

/// Kyty: ShaderSpirv.cpp `operand_is_constant` (L1564).
#[must_use]
pub(crate) fn operand_is_constant(op: ShaderOperand) -> bool {
    matches!(
        op.type_,
        ShaderOperandType::LiteralConstant
            | ShaderOperandType::IntegerInlineConstant
            | ShaderOperandType::FloatInlineConstant
    )
}

/// Kyty: ShaderSpirv.cpp `operand_is_variable` (L1570).
#[must_use]
pub(crate) fn operand_is_variable(op: ShaderOperand) -> bool {
    matches!(
        op.type_,
        ShaderOperandType::Vgpr
            | ShaderOperandType::VccLo
            | ShaderOperandType::VccHi
            | ShaderOperandType::Sgpr
            | ShaderOperandType::ExecLo
            | ShaderOperandType::ExecHi
            | ShaderOperandType::ExecZ
            | ShaderOperandType::Scc
            | ShaderOperandType::M0
    )
}

/// Kyty: ShaderSpirv.cpp `operand_variable_to_str` (L1577). Kyty `EXIT_IF`s
/// on `op.size != 1`; the port returns an `Unknown` value instead (callers
/// turn that into a typed error).
#[must_use]
pub(crate) fn operand_variable_to_str(op: ShaderOperand) -> SpirvValue {
    use ShaderOperandType as O;
    let mut ret = SpirvValue::default();
    match op.type_ {
        O::Vgpr => {
            ret.value = format!("v{}", op.register_id);
            ret.type_ = SpirvType::Float;
        }
        O::Sgpr => {
            ret.value = format!("s{}", op.register_id);
            ret.type_ = SpirvType::Uint;
        }
        O::VccLo => {
            ret.value = "vcc_lo".to_string();
            ret.type_ = SpirvType::Uint;
        }
        O::VccHi => {
            ret.value = "vcc_hi".to_string();
            ret.type_ = SpirvType::Uint;
        }
        O::ExecLo => {
            ret.value = "exec_lo".to_string();
            ret.type_ = SpirvType::Uint;
        }
        O::ExecHi => {
            ret.value = "exec_hi".to_string();
            ret.type_ = SpirvType::Uint;
        }
        O::ExecZ => {
            ret.value = "execz".to_string();
            ret.type_ = SpirvType::Uint;
        }
        O::Scc => {
            ret.value = "scc".to_string();
            ret.type_ = SpirvType::Uint;
        }
        O::M0 => {
            ret.value = "m0".to_string();
            ret.type_ = SpirvType::Uint;
        }
        _ => {}
    }
    ret
}

/// Kyty: ShaderSpirv.cpp `operand_variable_to_str` multi-dword overload
/// (L1627).
#[must_use]
pub(crate) fn operand_variable_to_str_shift(op: ShaderOperand, shift: i32) -> SpirvValue {
    use ShaderOperandType as O;
    let mut ret = SpirvValue::default();
    match op.type_ {
        O::Vgpr => {
            ret.value = format!("v{}", op.register_id + shift);
            ret.type_ = SpirvType::Float;
        }
        O::Sgpr => {
            ret.value = format!("s{}", op.register_id + shift);
            ret.type_ = SpirvType::Uint;
        }
        O::VccLo => {
            if shift == 0 {
                ret.value = "vcc_lo".to_string();
                ret.type_ = SpirvType::Uint;
            } else if shift == 1 {
                ret.value = "vcc_hi".to_string();
                ret.type_ = SpirvType::Uint;
            }
        }
        O::ExecLo => {
            if shift == 0 {
                ret.value = "exec_lo".to_string();
                ret.type_ = SpirvType::Uint;
            } else if shift == 1 {
                ret.value = "exec_hi".to_string();
                ret.type_ = SpirvType::Uint;
            }
        }
        _ => {}
    }
    ret
}

/// Kyty: ShaderSpirv.cpp `operand_is_exec` (L1671).
#[must_use]
pub(crate) fn operand_is_exec(op: ShaderOperand) -> bool {
    matches!(
        op.type_,
        ShaderOperandType::ExecLo | ShaderOperandType::ExecHi | ShaderOperandType::ExecZ
    )
}

/// Kyty: ShaderSpirv.cpp `operand_load_int` (L1683).
///
/// No C1 recompiler consumes int loads yet (first users are C2 functions
/// such as `Recompile_SMulkI32`, ShaderSpirv.cpp L4437) — kept alongside
/// its uint/float siblings so C2 stays mechanical.
#[allow(dead_code)]
pub(crate) fn operand_load_int(
    spirv: &Spirv<'_>,
    op: ShaderOperand,
    result_id: &str,
    index: &str,
    load: &mut String,
) -> Result<bool, ShaderRecompileError> {
    if op.negate || op.absolute {
        return Err(not_supported(
            "operand_load_int",
            "negate/absolute modifier",
        ));
    }

    if operand_is_constant(op) {
        let id = spirv.get_constant(op);
        *load = "%<result_id> = OpBitcast %int %<id>"
            .replace("<index>", index)
            .replace("<id>", &id)
            .replace("<result_id>", result_id);
    } else if operand_is_variable(op) {
        let value = operand_variable_to_str(op);
        if value.type_ == SpirvType::Float {
            *load = concat!(
                "%t<result_id> = OpLoad %float %<id>\n",
                "          ",
                "%<result_id> = OpBitcast %int %t<result_id>\n"
            )
            .replace("<index>", index)
            .replace("<id>", &value.value)
            .replace("<result_id>", result_id);
        } else if value.type_ == SpirvType::Uint {
            *load = concat!(
                "%t<result_id> = OpLoad %uint %<id>\n",
                "          ",
                "%<result_id> = OpBitcast %int %t<result_id>\n"
            )
            .replace("<index>", index)
            .replace("<id>", &value.value)
            .replace("<result_id>", result_id);
        }
    } else {
        return Ok(false);
    }
    Ok(true)
}

/// Kyty: ShaderSpirv.cpp `operand_load_uint` (L1723). `shift = -1` matches
/// Kyty's default argument.
pub(crate) fn operand_load_uint(
    spirv: &Spirv<'_>,
    op: ShaderOperand,
    result_id: &str,
    index: &str,
    load: &mut String,
    shift: i32,
) -> Result<bool, ShaderRecompileError> {
    if op.negate || op.absolute {
        return Err(not_supported(
            "operand_load_uint",
            "negate/absolute modifier",
        ));
    }

    if operand_is_constant(op) {
        if op.size == 2 {
            if !(0..2).contains(&shift) {
                return Err(not_supported(
                    "operand_load_uint",
                    format!("invalid shift {shift} for 64-bit constant"),
                ));
            }
            if shift == 0 {
                let id = spirv.get_constant(op);
                *load = "%<result_id> = OpBitcast %uint %<id>"
                    .replace("<index>", index)
                    .replace("<id>", &id)
                    .replace("<result_id>", result_id);
            } else if op.type_ == ShaderOperandType::IntegerInlineConstant && op.constant.i() < 0 {
                *load = "%<result_id> = OpBitcast %uint %uint_0xffffffff"
                    .replace("<index>", index)
                    .replace("<result_id>", result_id);
            } else {
                *load = "%<result_id> = OpBitcast %uint %uint_0"
                    .replace("<index>", index)
                    .replace("<result_id>", result_id);
            }
        } else {
            let id = spirv.get_constant(op);
            *load = "%<result_id> = OpBitcast %uint %<id>"
                .replace("<index>", index)
                .replace("<id>", &id)
                .replace("<result_id>", result_id);
        }
    } else if operand_is_variable(op) {
        let value = if shift >= 0 {
            operand_variable_to_str_shift(op, shift)
        } else {
            operand_variable_to_str(op)
        };
        if value.type_ == SpirvType::Float {
            *load = concat!(
                "%t<result_id> = OpLoad %float %<id>\n",
                "          ",
                "%<result_id> = OpBitcast %uint %t<result_id>\n"
            )
            .replace("<index>", index)
            .replace("<id>", &value.value)
            .replace("<result_id>", result_id);
        } else if value.type_ == SpirvType::Uint {
            *load = "%<result_id> = OpLoad %uint %<id>"
                .replace("<index>", index)
                .replace("<id>", &value.value)
                .replace("<result_id>", result_id);
        } else {
            return Ok(false);
        }
    } else {
        return Ok(false);
    }
    Ok(true)
}

/// Kyty: ShaderSpirv.cpp `operand_load_float` (L1791).
pub(crate) fn operand_load_float(
    spirv: &Spirv<'_>,
    op: ShaderOperand,
    result_id: &str,
    index: &str,
    load: &mut String,
) -> Result<bool, ShaderRecompileError> {
    let mut l: String;

    if operand_is_constant(op) {
        let id = spirv.get_constant(op);
        l = "%<result_id> = OpBitcast %float %<id>".replace("<id>", &id);
    } else if operand_is_variable(op) {
        let value = operand_variable_to_str(op);
        if value.type_ == SpirvType::Float {
            l = "%<result_id> = OpLoad %float %<id>\n".replace("<id>", &value.value);
        } else if value.type_ == SpirvType::Uint {
            l = concat!(
                "%t<result_id> = OpLoad %uint %<id>\n",
                "          ",
                "%<result_id> = OpBitcast %float %t<result_id>\n"
            )
            .replace("<id>", &value.value);
        } else {
            return Ok(false);
        }
    } else {
        return Ok(false);
    }

    if op.negate && op.absolute {
        l += concat!(
            "          ",
            "%abs_<index> = OpExtInst %float %GLSL_std_450 FAbs %<result_id>\n",
            "          ",
            "%<result> = OpFNegate %float %abs_<index>\n"
        );
        *load = l
            .replace("<index>", index)
            .replace("<result_id>", &format!("a{result_id}"))
            .replace("<result>", result_id);
        return Ok(true);
    }

    if op.absolute {
        l += concat!(
            "          ",
            "%<result> = OpExtInst %float %GLSL_std_450 FAbs %<result_id>\n"
        );
        *load = l
            .replace("<index>", index)
            .replace("<result_id>", &format!("a{result_id}"))
            .replace("<result>", result_id);
    } else if op.negate {
        l += concat!("          ", "%<result> = OpFNegate %float %<result_id>\n");
        *load = l
            .replace("<index>", index)
            .replace("<result_id>", &format!("n{result_id}"))
            .replace("<result>", result_id);
    } else {
        *load = l
            .replace("<index>", index)
            .replace("<result_id>", result_id);
    }

    Ok(true)
}

// ---------------------------------------------------------------------------
// Spirv generator
// ---------------------------------------------------------------------------

/// Kyty: ShaderSpirv.cpp `Spirv::Constant` (L1508).
#[derive(Clone, Debug, Default)]
struct Constant {
    type_: SpirvType,
    constant: ShaderConstant,
    type_str: String,
    value_str: String,
    id: String,
}

/// Kyty: ShaderSpirv.cpp `Spirv::Variable` (L1503).
#[derive(Clone, Debug)]
struct Variable {
    op: ShaderOperand,
}

/// Kyty: ShaderSpirv.cpp class `Spirv` (L1459) — generates SPIR-V assembly
/// text from a parsed [`ShaderCode`].
pub struct Spirv<'a> {
    source: String,
    code: ShaderCode,
    constants: Vec<Constant>,
    variables: Vec<Variable>,
    vs_input_info: Option<&'a ShaderVertexInputInfo>,
    cs_input_info: Option<&'a ShaderComputeInputInfo>,
    ps_input_info: Option<&'a ShaderPixelInputInfo>,
    bind: Option<&'a ShaderBindResources>,
    /// Kyty: `Core::Array2<int, 64, 2> m_extended_mapping`.
    extended_mapping: [[i32; 2]; 64],
    /// Deviation: Kyty reads the global `Config::SpirvDebugPrintfEnabled()`;
    /// the port threads it as a field (default off).
    pub debug_printf_enabled: bool,
}

impl Default for Spirv<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Spirv<'a> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            source: String::new(),
            code: ShaderCode::new(),
            constants: Vec::new(),
            variables: Vec::new(),
            vs_input_info: None,
            cs_input_info: None,
            ps_input_info: None,
            bind: None,
            extended_mapping: [[0; 2]; 64],
            debug_printf_enabled: false,
        }
    }

    #[must_use]
    pub fn get_code(&self) -> &ShaderCode {
        &self.code
    }

    pub fn get_code_mut(&mut self) -> &mut ShaderCode {
        &mut self.code
    }

    pub fn set_code(&mut self, code: &ShaderCode) {
        self.code = code.clone();
    }

    #[must_use]
    pub fn get_source(&self) -> &str {
        &self.source
    }

    pub fn set_vs_input_info(&mut self, input_info: Option<&'a ShaderVertexInputInfo>) {
        self.vs_input_info = input_info;
    }

    #[must_use]
    pub fn get_vs_input_info(&self) -> Option<&'a ShaderVertexInputInfo> {
        self.vs_input_info
    }

    pub fn set_cs_input_info(&mut self, input_info: Option<&'a ShaderComputeInputInfo>) {
        self.cs_input_info = input_info;
    }

    #[must_use]
    pub fn get_cs_input_info(&self) -> Option<&'a ShaderComputeInputInfo> {
        self.cs_input_info
    }

    pub fn set_ps_input_info(&mut self, input_info: Option<&'a ShaderPixelInputInfo>) {
        self.ps_input_info = input_info;
    }

    #[must_use]
    pub fn get_ps_input_info(&self) -> Option<&'a ShaderPixelInputInfo> {
        self.ps_input_info
    }

    #[must_use]
    pub fn get_bind_info(&self) -> Option<&'a ShaderBindResources> {
        self.bind
    }

    /// Kyty: `Spirv::GetMappedIndex` (L1495).
    pub fn get_mapped_index(&self, offset: i32) -> Result<(i32, i32), ShaderRecompileError> {
        let Some(m) = usize::try_from(offset)
            .ok()
            .and_then(|o| self.extended_mapping.get(o))
        else {
            return Err(not_supported(
                "Spirv::GetMappedIndex",
                format!("offset {offset} >= extended mapping size"),
            ));
        };
        Ok((m[0], m[1]))
    }

    /// Kyty: `Spirv::AddConstantUint` (L6467).
    pub fn add_constant_uint(&mut self, u: u32) {
        self.add_constant_typed(SpirvType::Uint, ShaderConstant::from_u(u));
    }

    /// Kyty: `Spirv::AddConstantInt` (L6474).
    pub fn add_constant_int(&mut self, i: i32) {
        self.add_constant_typed(SpirvType::Int, ShaderConstant::from_i(i));
    }

    /// Kyty: `Spirv::AddConstantFloat` (L6481).
    pub fn add_constant_float(&mut self, f: f32) {
        self.add_constant_typed(SpirvType::Float, ShaderConstant::from_f(f));
    }

    /// Kyty: `Spirv::AddConstant(ShaderOperand)` (L6488).
    pub fn add_constant(&mut self, op: ShaderOperand) -> Result<(), ShaderRecompileError> {
        let type_ = match op.type_ {
            ShaderOperandType::LiteralConstant => SpirvType::Uint,
            ShaderOperandType::IntegerInlineConstant => SpirvType::Int,
            ShaderOperandType::FloatInlineConstant => SpirvType::Float,
            _ => {
                return Err(not_supported(
                    "Spirv::AddConstant",
                    format!("operand type {:?} is not a constant", op.type_),
                ));
            }
        };
        self.add_constant_typed(type_, op.constant);
        Ok(())
    }

    /// Kyty: `Spirv::AddConstant(SpirvType, ShaderConstant)` (L6510).
    fn add_constant_typed(&mut self, type_: SpirvType, constant: ShaderConstant) {
        for c in &self.constants {
            if c.type_ == type_ && c.constant.u == constant.u {
                return;
            }
        }

        let type_str = type_.to_lower_str().to_string();
        let value_str = match type_ {
            SpirvType::Uint => {
                if constant.u < 256 {
                    format!("{}", constant.u)
                } else {
                    format!("0x{:08x}", constant.u)
                }
            }
            SpirvType::Int => format!("{}", constant.i()),
            SpirvType::Float => format!("{:.6}", constant.f()),
            SpirvType::Unknown => String::new(),
        };
        let id = format!(
            "{}_{}",
            type_str,
            value_str.replace('.', "_").replace('-', "m")
        );

        self.constants.push(Constant {
            type_,
            constant,
            type_str,
            value_str,
            id,
        });
    }

    /// Kyty: `Spirv::AddVariable(type, register_id, size)` (L6543).
    fn add_variable_parts(&mut self, type_: ShaderOperandType, register_id: i32, size: i32) {
        let op = ShaderOperand {
            type_,
            register_id,
            size,
            ..Default::default()
        };
        self.add_variable(op);
    }

    /// Kyty: `Spirv::AddVariable(ShaderOperand)` (L6552).
    fn add_variable(&mut self, op: ShaderOperand) {
        if operand_is_variable(op) {
            for i in 0..op.size {
                let mut v = Variable {
                    op: ShaderOperand {
                        type_: op.type_,
                        register_id: op.register_id + i,
                        size: 1,
                        ..Default::default()
                    },
                };

                if op.type_ == ShaderOperandType::VccLo && op.size == 2 && i == 1 {
                    v.op.type_ = ShaderOperandType::VccHi;
                    v.op.register_id = 0;
                }

                if op.type_ == ShaderOperandType::ExecLo && op.size == 2 && i == 1 {
                    v.op.type_ = ShaderOperandType::ExecHi;
                    v.op.register_id = 0;
                }

                if !self.variables.iter().any(|v1| v1.op == v.op) {
                    self.variables.push(v);
                }
            }
        }
    }

    /// Kyty: `Spirv::GetConstantUint` (L6585).
    #[must_use]
    pub fn get_constant_uint(&self, u: u32) -> String {
        for c in &self.constants {
            if c.type_ == SpirvType::Uint && c.constant.u == u {
                return c.id.clone();
            }
        }
        "unknown_uint_constant".to_string()
    }

    /// Kyty: `Spirv::GetConstantInt` (L6598).
    #[must_use]
    pub fn get_constant_int(&self, i: i32) -> String {
        for c in &self.constants {
            if c.type_ == SpirvType::Int && c.constant.i() == i {
                return c.id.clone();
            }
        }
        "unknown_int_constant".to_string()
    }

    /// Kyty: `Spirv::GetConstantFloat` (L6611).
    #[must_use]
    pub fn get_constant_float(&self, f: f32) -> String {
        for c in &self.constants {
            if c.type_ == SpirvType::Float && c.constant.f() == f {
                return c.id.clone();
            }
        }
        "unknown_float_constant".to_string()
    }

    /// Kyty: `Spirv::GetConstant(ShaderOperand)` (L6624).
    #[must_use]
    pub fn get_constant(&self, op: ShaderOperand) -> String {
        let type_ = match op.type_ {
            ShaderOperandType::LiteralConstant => SpirvType::Uint,
            ShaderOperandType::IntegerInlineConstant => SpirvType::Int,
            ShaderOperandType::FloatInlineConstant => SpirvType::Float,
            _ => SpirvType::Unknown,
        };
        for c in &self.constants {
            if c.type_ == type_ && c.constant.u == op.constant.u {
                return c.id.clone();
            }
        }
        "unknown_operand_constant".to_string()
    }

    /// Kyty: `Spirv::GenerateSource` (L6652).
    pub fn generate_source(&mut self) -> Result<(), ShaderRecompileError> {
        self.source.clear();

        self.bind = match self.code.get_type() {
            ShaderType::Pixel => self.ps_input_info.map(|i| &i.bind),
            ShaderType::Vertex => self.vs_input_info.map(|i| &i.bind),
            ShaderType::Compute => self.cs_input_info.map(|i| &i.bind),
            _ => None,
        };

        if let Some(vs) = self.vs_input_info {
            if vs.fetch_embedded || vs.fetch_inline {
                self.detect_fetch()?;
            }
        }

        self.write_header()?;
        self.write_debug();
        self.write_annotations()?;
        self.write_types()?;
        self.write_constants()?;
        self.write_global_variables()?;
        self.write_main_prolog();
        self.write_local_variables()?;
        self.write_instructions()?;
        self.write_main_epilog();
        self.write_functions();

        Ok(())
    }

    /// Kyty: `Spirv::WriteHeader` (L6685).
    fn write_header(&mut self) -> Result<(), ShaderRecompileError> {
        const HEADER: &str = r#"
                ; Header
                OpCapability Shader
                OpCapability ImageQuery
                <Extensions>
                <Imports>
                OpMemoryModel Logical GLSL450
                OpEntryPoint <Type> %main "main" <Variables>
                <ExecutionModes>
"#;

        let mut vars: Vec<String> = Vec::new();
        let mut extensions: Vec<String> = Vec::new();
        let mut imports: Vec<String> = Vec::new();
        let mut execution_modes: Vec<String> = Vec::new();

        imports.push("%GLSL_std_450 = OpExtInstImport \"GLSL.std.450\"".to_string());

        if self.debug_printf_enabled {
            extensions.push("OpExtension \"SPV_KHR_non_semantic_info\"".to_string());
            imports.push(
                "%NonSemantic_DebugPrintf = OpExtInstImport \"NonSemantic.DebugPrintf\""
                    .to_string(),
            );
        }

        if let Some(bind) = self.bind {
            if bind.storage_buffers.buffers_num > 0 {
                vars.push("%buf".to_string());
            }
            if bind.textures2d.textures2d_sampled_num > 0 {
                vars.push("%textures2D_S".to_string());
            }
            if bind.textures2d.textures2d_storage_num > 0 {
                vars.push("%textures2D_L".to_string());
            }
            if bind.samplers.samplers_num > 0 {
                vars.push("%samplers".to_string());
            }
            if bind.gds_pointers.pointers_num > 0 {
                vars.push("%gds".to_string());
            }
            if bind.push_constant_size > 0 {
                vars.push("%vsharp".to_string());
            }
        }

        let header_str = match self.code.get_type() {
            ShaderType::Pixel => {
                vars.push("%outColor".to_string());
                if let Some(info) = self.ps_input_info {
                    for i in 0..info.input_num {
                        vars.push(format!("%attr{i}"));
                    }
                    if info.ps_pos_xy {
                        vars.push("%gl_FragCoord".to_string());
                    }
                    if info.ps_early_z {
                        execution_modes
                            .push("OpExecutionMode %main EarlyFragmentTests\n".to_string());
                    }
                }
                let h = HEADER.replace("<Type>", "Fragment");
                execution_modes.push("OpExecutionMode %main OriginUpperLeft\n".to_string());
                // TODO() do we need PixelCenterInteger mode?
                h
            }
            ShaderType::Vertex => {
                if let Some(info) = self.vs_input_info {
                    for i in 0..info.resources_num {
                        vars.push(format!("%attr{i}"));
                    }
                    for i in 0..info.export_count {
                        vars.push(format!("%param{i}"));
                    }
                }
                vars.push("%gl_VertexIndex".to_string());
                vars.push("%gl_InstanceIndex".to_string());
                vars.push("%outPerVertex".to_string());
                HEADER.replace("<Type>", "Vertex")
            }
            ShaderType::Compute => {
                if let Some(info) = self.cs_input_info {
                    execution_modes.push(format!(
                        "OpExecutionMode %main LocalSize {} {} {}",
                        info.threads_num[0], info.threads_num[1], info.threads_num[2]
                    ));
                }
                vars.push("%gl_LocalInvocationID".to_string());
                vars.push("%gl_WorkGroupID".to_string());
                HEADER.replace("<Type>", "GLCompute")
            }
            _ => {
                tracing::error!("unknown shader type: {:?}", self.code.get_type());
                return Err(ShaderRecompileError::UnknownShaderType);
            }
        };

        let sep = format!("\n{}", " ".repeat(15));
        self.source += &header_str
            .replace("<Variables>", &vars.join(" "))
            .replace("<ExecutionModes>", &execution_modes.join(&sep))
            .replace("<Imports>", &imports.join(&sep))
            .replace("<Extensions>", &extensions.join(&sep));

        Ok(())
    }

    /// Kyty: `Spirv::WriteDebug` (L6801).
    fn write_debug(&mut self) {
        if self.debug_printf_enabled {
            for (index, p) in self.code.get_debug_printfs().iter().enumerate() {
                self.source += &format!("%printf_str_{index} = OpString \"{}\"", p.format);
            }
        }
    }

    /// Kyty: `Spirv::WriteAnnotations` (L6814).
    fn write_annotations(&mut self) -> Result<(), ShaderRecompileError> {
        const PIXEL_ANNOTATIONS: &str = r#"
               ; Annotations
               OpDecorate %outColor Location 0
               <Variables>
"#;
        const VERTEX_ANNOTATIONS: &str = r#"
               ; Annotations
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpDecorate %gl_InstanceIndex BuiltIn InstanceIndex
               OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
               OpMemberDecorate %gl_PerVertex 1 BuiltIn PointSize
               OpMemberDecorate %gl_PerVertex 2 BuiltIn ClipDistance
               OpMemberDecorate %gl_PerVertex 3 BuiltIn CullDistance
               OpDecorate %gl_PerVertex Block
               ; OpDecorate %param0 Location 0
               <Variables>
"#;
        const COMPUTE_ANNOTATIONS: &str = r#"
               ; Annotations
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
               <Variables>
"#;

        let mut vars: Vec<String> = Vec::new();
        let sep = format!("\n{}", " ".repeat(15));

        match self.code.get_type() {
            ShaderType::Pixel => {
                if let Some(info) = self.ps_input_info {
                    for i in 0..info.input_num as usize {
                        if (info.interpolator_settings[i] & !0x41f_u32) != 0 {
                            return Err(not_supported(
                                "Spirv::WriteAnnotations",
                                format!(
                                    "interpolator settings 0x{:08x}",
                                    info.interpolator_settings[i]
                                ),
                            ));
                        }
                        let flat = (info.interpolator_settings[i] & 0x400_u32) != 0;
                        let location = info.interpolator_settings[i] & 0x1f_u32;
                        if flat {
                            vars.push(format!("OpDecorate %attr{i} Flat"));
                        }
                        vars.push(format!("OpDecorate %attr{i} Location {location}"));
                    }
                    if info.ps_pos_xy {
                        vars.push("OpDecorate %gl_FragCoord BuiltIn FragCoord".to_string());
                    }
                }
                self.source += &PIXEL_ANNOTATIONS.replace("<Variables>", &vars.join(&sep));
            }
            ShaderType::Vertex => {
                if let Some(info) = self.vs_input_info {
                    for i in 0..info.resources_num {
                        vars.push(format!("OpDecorate %attr{i} Location {i}"));
                    }
                    for i in 0..info.export_count {
                        vars.push(format!("OpDecorate %param{i} Location {i}"));
                    }
                }
                self.source += &VERTEX_ANNOTATIONS.replace("<Variables>", &vars.join(&sep));
            }
            ShaderType::Compute => {
                self.source += &COMPUTE_ANNOTATIONS.replace("<Variables>", &vars.join(&sep));
            }
            _ => {
                tracing::error!("unknown shader type: {:?}", self.code.get_type());
                return Err(ShaderRecompileError::UnknownShaderType);
            }
        }

        const STORAGE_BUFFERS_ANNOTATIONS: &str = r#"
       OpDecorate %buffers_runtimearr_float ArrayStride 4
       OpMemberDecorate %BufferObject 0 Offset 0
       OpDecorate %BufferObject Block
       OpDecorate %buf DescriptorSet <DescriptorSet>
       OpDecorate %buf Binding <BindingIndex>
"#;
        const TEXTURES_ANNOTATIONS_S: &str = r#"
       OpDecorate %textures2D_S DescriptorSet <DescriptorSet>
       OpDecorate %textures2D_S Binding <BindingIndex>
"#;
        const TEXTURES_ANNOTATIONS_L: &str = r#"
       OpDecorate %textures2D_L DescriptorSet <DescriptorSet>
       OpDecorate %textures2D_L Binding <BindingIndex>
"#;
        const SAMPLERS_ANNOTATIONS: &str = r#"
       OpDecorate %samplers DescriptorSet <DescriptorSet>
       OpDecorate %samplers Binding <BindingIndex>
"#;
        const GDS_ANNOTATIONS: &str = r#"
               OpDecorate %gds_runtimearr_uint ArrayStride 4
               OpMemberDecorate %GDS 0 Coherent
               OpMemberDecorate %GDS 0 Offset 0
               OpDecorate %GDS Block
               OpDecorate %gds DescriptorSet <DescriptorSet>
               OpDecorate %gds Binding <BindingIndex>
"#;
        const VSHARP_ANNOTATIONS: &str = r#"
       OpDecorate %vsharp_arr_uint_uint_4 ArrayStride 4
       OpDecorate %vsharp_arr__arr_uint_uint_4_uint_<buffers_num> ArrayStride 16
	   OpMemberDecorate %BufferResource 0 Offset <Offset>
       OpDecorate %BufferResource Block
"#;

        if let Some(bind) = self.bind {
            if bind.storage_buffers.buffers_num > 0 {
                self.source += &STORAGE_BUFFERS_ANNOTATIONS
                    .replace("<DescriptorSet>", &format!("{}", bind.descriptor_set_slot))
                    .replace(
                        "<BindingIndex>",
                        &format!("{}", bind.storage_buffers.binding_index),
                    );
            }
            if bind.textures2d.textures2d_sampled_num > 0 {
                self.source += &TEXTURES_ANNOTATIONS_S
                    .replace("<DescriptorSet>", &format!("{}", bind.descriptor_set_slot))
                    .replace(
                        "<BindingIndex>",
                        &format!("{}", bind.textures2d.binding_sampled_index),
                    );
            }
            if bind.textures2d.textures2d_storage_num > 0 {
                self.source += &TEXTURES_ANNOTATIONS_L
                    .replace("<DescriptorSet>", &format!("{}", bind.descriptor_set_slot))
                    .replace(
                        "<BindingIndex>",
                        &format!("{}", bind.textures2d.binding_storage_index),
                    );
            }
            if bind.samplers.samplers_num > 0 {
                self.source += &SAMPLERS_ANNOTATIONS
                    .replace("<DescriptorSet>", &format!("{}", bind.descriptor_set_slot))
                    .replace(
                        "<BindingIndex>",
                        &format!("{}", bind.samplers.binding_index),
                    );
            }
            if bind.gds_pointers.pointers_num > 0 {
                self.source += &GDS_ANNOTATIONS
                    .replace("<DescriptorSet>", &format!("{}", bind.descriptor_set_slot))
                    .replace(
                        "<BindingIndex>",
                        &format!("{}", bind.gds_pointers.binding_index),
                    );
            }
            if bind.push_constant_size > 0 {
                self.source += &VSHARP_ANNOTATIONS
                    .replace(
                        "<buffers_num>",
                        &format!("{}", bind.push_constant_size / 16),
                    )
                    .replace("<Offset>", &format!("{}", bind.push_constant_offset));
            }
        }

        Ok(())
    }

    /// Kyty: `Spirv::WriteTypes` (L6967).
    fn write_types(&mut self) -> Result<(), ShaderRecompileError> {
        const TYPES: &str = r#"
                               ; Types
                         %void = OpTypeVoid
                        %float = OpTypeFloat 32
                          %int = OpTypeInt 32 1
                         %uint = OpTypeInt 32 0
                         %bool = OpTypeBool
                      %v2float = OpTypeVector %float 2
                      %v3float = OpTypeVector %float 3
                      %v4float = OpTypeVector %float 4
                       %v2uint = OpTypeVector %uint 2
                       %v3uint = OpTypeVector %uint 3
                       %v4uint = OpTypeVector %uint 4
                        %v2int = OpTypeVector %int 2
                 %undef_v2uint = OpUndef %v2uint
               %_ptr_Input_int = OpTypePointer Input %int
              %_ptr_Input_uint = OpTypePointer Input %uint
             %_ptr_Input_float = OpTypePointer Input %float
           %_ptr_Input_v2float = OpTypePointer Input %v2float
           %_ptr_Input_v3float = OpTypePointer Input %v3float
           %_ptr_Input_v4float = OpTypePointer Input %v4float
            %_ptr_Input_v3uint = OpTypePointer Input %v3uint
          %_ptr_Output_v4float = OpTypePointer Output %v4float
          %_ptr_Function_float = OpTypePointer Function %float
           %_ptr_Function_bool = OpTypePointer Function %bool
            %_ptr_Function_int = OpTypePointer Function %int
           %_ptr_Function_uint = OpTypePointer Function %uint
        %_ptr_Function_v2float = OpTypePointer Function %v2float
        %_ptr_Function_v3float = OpTypePointer Function %v3float
        %_ptr_Function_v4float = OpTypePointer Function %v4float
           %_ptr_Uniform_float = OpTypePointer Uniform %float
     %_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
      %_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
                     %ResTypeI = OpTypeStruct %int %int
                     %ResTypeU = OpTypeStruct %uint %uint
                %function_void = OpTypeFunction %void
              %function_fetch1 = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_float
              %function_fetch2 = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_float %_ptr_Function_v2float
              %function_fetch3 = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_float %_ptr_Function_float %_ptr_Function_v3float
              %function_fetch4 = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_float %_ptr_Function_float %_ptr_Function_float %_ptr_Function_v4float
                 %function_u_u = OpTypeFunction %uint %uint %uint
               %function_u_u_u = OpTypeFunction %uint %uint %uint %uint
            %function_u2_u_u_u = OpTypeFunction %v2uint %uint %uint %uint
               %function_b_f_f = OpTypeFunction %bool %float %float
                 %function_i_i = OpTypeFunction %int %int %int
               %function_shift = OpTypeFunction %void %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint
    %function_tbuffer_load_format_xyzw = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_float %_ptr_Function_float %_ptr_Function_float %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int
    %function_buffer_load_store_float1 = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int
    %function_buffer_load_store_float2 = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_float %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int
          %function_buffer_load_float4 = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_float %_ptr_Function_float %_ptr_Function_float %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int
 %function_tbuffer_load_store_format_x = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int
%function_tbuffer_load_store_format_xy = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_float %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int %_ptr_Function_int
          %function_sbuffer_load_dword = OpTypeFunction %void %_ptr_Function_uint %_ptr_Function_int %_ptr_Function_int
        %function_sbuffer_load_dword_2 = OpTypeFunction %void %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_int %_ptr_Function_int
        %function_sbuffer_load_dword_4 = OpTypeFunction %void %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_int %_ptr_Function_int
        %function_sbuffer_load_dword_8 = OpTypeFunction %void %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_int %_ptr_Function_int
       %function_sbuffer_load_dword_16 = OpTypeFunction %void %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_uint %_ptr_Function_int %_ptr_Function_int
"#;

        const PIXEL_TYPES: &str = "
";

        const VERTEX_TYPES: &str = r#"
            %array_length = OpConstant %uint 1
        %int_per_vertex_0 = OpConstant %int 0
       %_arr_float_uint_1 = OpTypeArray %float %array_length
            %gl_PerVertex = OpTypeStruct %v4float %float %_arr_float_uint_1 %_arr_float_uint_1
%_ptr_Output_gl_PerVertex = OpTypePointer Output %gl_PerVertex
"#;

        const COMPUTE_TYPES: &str = "
";

        self.source += TYPES;

        match self.code.get_type() {
            ShaderType::Vertex => self.source += VERTEX_TYPES,
            ShaderType::Pixel => self.source += PIXEL_TYPES,
            ShaderType::Compute => self.source += COMPUTE_TYPES,
            _ => {
                tracing::error!("unknown shader type: {:?}", self.code.get_type());
                return Err(ShaderRecompileError::UnknownShaderType);
            }
        }

        const STORAGE_BUFFERS_TYPES: &str = r#"
                               %buffers_runtimearr_float = OpTypeRuntimeArray %float
                                           %BufferObject = OpTypeStruct %buffers_runtimearr_float
                         %buffers_num_uint_<buffers_num> = OpConstant %uint <buffers_num>
                   %_arr_BufferObject_uint_<buffers_num> = OpTypeArray %BufferObject %buffers_num_uint_<buffers_num>
%_ptr_StorageBuffer__arr_BufferObject_uint_<buffers_num> = OpTypePointer StorageBuffer %_arr_BufferObject_uint_<buffers_num>
"#;

        const TEXTURES_SAMPLED_TYPES: &str = r#"
                                             %ImageS = OpTypeImage %float 2D 0 0 0 1 Unknown
                    %textures2D_S_uint_<buffers_num> = OpConstant %uint <buffers_num>
                     %_arr_ImageS_uint_<buffers_num> = OpTypeArray %ImageS %textures2D_S_uint_<buffers_num>
%_ptr_UniformConstant__arr_ImageS_uint_<buffers_num> = OpTypePointer UniformConstant %_arr_ImageS_uint_<buffers_num>
                        %_ptr_UniformConstant_ImageS = OpTypePointer UniformConstant %ImageS
                                       %SampledImage = OpTypeSampledImage %ImageS
"#;

        const TEXTURES_LOADED_TYPES: &str = r#"
                                             %ImageL = OpTypeImage %float 2D 0 0 0 2 Rgba8
                    %textures2D_L_uint_<buffers_num> = OpConstant %uint <buffers_num>
                     %_arr_ImageL_uint_<buffers_num> = OpTypeArray %ImageL %textures2D_L_uint_<buffers_num>
%_ptr_UniformConstant__arr_ImageL_uint_<buffers_num> = OpTypePointer UniformConstant %_arr_ImageL_uint_<buffers_num>
                        %_ptr_UniformConstant_ImageL = OpTypePointer UniformConstant %ImageL
"#;

        const SAMPLERS_TYPES: &str = r#"
                                             %Sampler = OpTypeSampler
                         %samplers_uint_<buffers_num> = OpConstant %uint <buffers_num>
                     %_arr_Sampler_uint_<buffers_num> = OpTypeArray %Sampler %samplers_uint_<buffers_num>
%_ptr_UniformConstant__arr_Sampler_uint_<buffers_num> = OpTypePointer UniformConstant %_arr_Sampler_uint_<buffers_num>
                        %_ptr_UniformConstant_Sampler = OpTypePointer UniformConstant %Sampler
"#;

        const GDS_TYPES: &str = r#"
            %gds_runtimearr_uint = OpTypeRuntimeArray %uint
                    %GDS = OpTypeStruct %gds_runtimearr_uint
            %_ptr_StorageBuffer_GDS = OpTypePointer StorageBuffer %GDS
"#;

        const VSHARP_TYPES: &str = r#"
         %vsharp_buffers_num_uint_<buffers_num> = OpConstant %uint <buffers_num>
                             %vsharp_num_uint_4 = OpConstant %uint 4
                        %vsharp_arr_uint_uint_4 = OpTypeArray %uint %vsharp_num_uint_4
%vsharp_arr__arr_uint_uint_4_uint_<buffers_num> = OpTypeArray %vsharp_arr_uint_uint_4 %vsharp_buffers_num_uint_<buffers_num>
                                %BufferResource = OpTypeStruct %vsharp_arr__arr_uint_uint_4_uint_<buffers_num>
              %_ptr_PushConstant_BufferResource = OpTypePointer PushConstant %BufferResource
                        %_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
"#;

        if let Some(bind) = self.bind {
            if bind.storage_buffers.buffers_num > 0 {
                self.source += &STORAGE_BUFFERS_TYPES.replace(
                    "<buffers_num>",
                    &format!("{}", bind.storage_buffers.buffers_num),
                );
            }
            if bind.textures2d.textures2d_sampled_num > 0 {
                self.source += &TEXTURES_SAMPLED_TYPES.replace(
                    "<buffers_num>",
                    &format!("{}", bind.textures2d.textures2d_sampled_num),
                );
            }
            if bind.textures2d.textures2d_storage_num > 0 {
                self.source += &TEXTURES_LOADED_TYPES.replace(
                    "<buffers_num>",
                    &format!("{}", bind.textures2d.textures2d_storage_num),
                );
            }
            if bind.samplers.samplers_num > 0 {
                self.source += &SAMPLERS_TYPES
                    .replace("<buffers_num>", &format!("{}", bind.samplers.samplers_num));
            }
            if bind.gds_pointers.pointers_num > 0 {
                self.source += GDS_TYPES;
            }
            if bind.push_constant_size > 0 {
                self.source += &VSHARP_TYPES.replace(
                    "<buffers_num>",
                    &format!("{}", bind.push_constant_size / 16),
                );
            }
        }

        Ok(())
    }

    /// Kyty: `Spirv::WriteConstants` (L7133).
    fn write_constants(&mut self) -> Result<(), ShaderRecompileError> {
        self.find_constants()?;

        const COMMENT: &str = r#"
    ; Constants
         %true = OpConstantTrue %bool
        %false = OpConstantFalse %bool
    %float_2pi = OpConstant %float 6.283185307179586476925286766559
"#;

        self.source += COMMENT;

        for c in &self.constants {
            self.source.push_str(&format!(
                "%{} = OpConstant %{} {}\n",
                c.id, c.type_str, c.value_str
            ));
        }

        Ok(())
    }

    /// Kyty: `Spirv::WriteGlobalVariables` (L7152).
    fn write_global_variables(&mut self) -> Result<(), ShaderRecompileError> {
        const PIXEL_VARIABLES: &str = r#"
              ;Variables
   %outColor = OpVariable %_ptr_Output_v4float Output
               <Variables>
"#;
        const VERTEX_VARIABLES: &str = r#"
              ;Variables
    %gl_VertexIndex = OpVariable %_ptr_Input_int Input
  %gl_InstanceIndex = OpVariable %_ptr_Input_int Input
      %outPerVertex = OpVariable %_ptr_Output_gl_PerVertex Output
            ; %param0 = OpVariable %_ptr_Output_v4float Output
               <Variables>
"#;
        const COMPUTE_VARIABLES: &str = r#"
              ;Variables
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
      %gl_WorkGroupID = OpVariable %_ptr_Input_v3uint Input
               <Variables>
"#;

        let mut vars: Vec<String> = Vec::new();

        if let Some(bind) = self.bind {
            if bind.storage_buffers.buffers_num > 0 {
                vars.push(format!(
                    "%buf = OpVariable %_ptr_StorageBuffer__arr_BufferObject_uint_{} StorageBuffer",
                    bind.storage_buffers.buffers_num
                ));
            }
            if bind.textures2d.textures2d_sampled_num > 0 {
                vars.push(format!(
                    "%textures2D_S = OpVariable %_ptr_UniformConstant__arr_ImageS_uint_{} UniformConstant",
                    bind.textures2d.textures2d_sampled_num
                ));
            }
            if bind.textures2d.textures2d_storage_num > 0 {
                vars.push(format!(
                    "%textures2D_L = OpVariable %_ptr_UniformConstant__arr_ImageL_uint_{} UniformConstant",
                    bind.textures2d.textures2d_storage_num
                ));
            }
            if bind.samplers.samplers_num > 0 {
                vars.push(format!(
                    "%samplers = OpVariable %_ptr_UniformConstant__arr_Sampler_uint_{} UniformConstant",
                    bind.samplers.samplers_num
                ));
            }
            if bind.gds_pointers.pointers_num > 0 {
                vars.push("%gds = OpVariable %_ptr_StorageBuffer_GDS StorageBuffer".to_string());
            }
            if bind.push_constant_size > 0 {
                vars.push(
                    "%vsharp = OpVariable %_ptr_PushConstant_BufferResource PushConstant"
                        .to_string(),
                );
            }
        }

        let sep = format!("\n{}", " ".repeat(15));

        match self.code.get_type() {
            ShaderType::Pixel => {
                if let Some(info) = self.ps_input_info {
                    for i in 0..info.input_num {
                        vars.push(format!("%attr{i} = OpVariable %_ptr_Input_v4float Input"));
                    }
                    if info.ps_pos_xy {
                        vars.push(
                            "%gl_FragCoord = OpVariable %_ptr_Input_v4float Input".to_string(),
                        );
                    }
                }
                self.source += &PIXEL_VARIABLES.replace("<Variables>", &vars.join(&sep));
            }
            ShaderType::Vertex => {
                if let Some(info) = self.vs_input_info {
                    for i in 0..info.resources_num as usize {
                        match info.resources_dst[i].registers_num {
                            1 => {
                                vars.push(format!("%attr{i} = OpVariable %_ptr_Input_float Input"))
                            }
                            2 => vars
                                .push(format!("%attr{i} = OpVariable %_ptr_Input_v2float Input")),
                            3 => vars
                                .push(format!("%attr{i} = OpVariable %_ptr_Input_v3float Input")),
                            4 => vars
                                .push(format!("%attr{i} = OpVariable %_ptr_Input_v4float Input")),
                            n => {
                                return Err(not_supported(
                                    "Spirv::WriteGlobalVariables",
                                    format!("invalid registers_num: {n}"),
                                ));
                            }
                        }
                    }
                    for i in 0..info.export_count {
                        vars.push(format!(
                            "%param{i} = OpVariable %_ptr_Output_v4float Output"
                        ));
                    }
                }
                self.source += &VERTEX_VARIABLES.replace("<Variables>", &vars.join(&sep));
            }
            ShaderType::Compute => {
                if let Some(info) = self.cs_input_info {
                    vars.push(format!(
                        "%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_{} %uint_{} %uint_{}",
                        info.threads_num[0], info.threads_num[1], info.threads_num[2]
                    ));
                }
                self.source += &COMPUTE_VARIABLES.replace("<Variables>", &vars.join(&sep));
            }
            _ => {
                tracing::error!("unknown shader type: {:?}", self.code.get_type());
                return Err(ShaderRecompileError::UnknownShaderType);
            }
        }

        Ok(())
    }

    /// Kyty: `Spirv::WriteMainProlog` (L7258).
    fn write_main_prolog(&mut self) {
        const TEXT: &str = r#"
                   ; Function main
                   ; Prolog
       %main       = OpFunction %void None %function_void
       %main_label = OpLabel
"#;
        self.source += TEXT;
    }

    /// Kyty: `Spirv::WriteLocalVariables` (L7271).
    fn write_local_variables(&mut self) -> Result<(), ShaderRecompileError> {
        self.find_variables();

        const COMMENT: &str = "
    ; Registers
";
        self.source += COMMENT;

        let mut decls = String::new();
        for c in &self.variables {
            let value = operand_variable_to_str(c.op);
            decls.push_str(&format!(
                "%{} = OpVariable %_ptr_Function_{} Function\n",
                value.value,
                value.type_.to_lower_str()
            ));
        }
        self.source += &decls;

        const COMMON_VARS: &str = r#"
             %temp_float = OpVariable %_ptr_Function_float Function
           %temp_v2float = OpVariable %_ptr_Function_v2float Function
           %temp_v3float = OpVariable %_ptr_Function_v3float Function
	       %temp_v4float = OpVariable %_ptr_Function_v4float Function
           %temp_int_0 = OpVariable %_ptr_Function_int Function
           %temp_int_1 = OpVariable %_ptr_Function_int Function
           %temp_int_2 = OpVariable %_ptr_Function_int Function
           %temp_int_3 = OpVariable %_ptr_Function_int Function
           %temp_int_4 = OpVariable %_ptr_Function_int Function
           %temp_int_5 = OpVariable %_ptr_Function_int Function
           %temp_uint_0 = OpVariable %_ptr_Function_uint Function
           %temp_uint_1 = OpVariable %_ptr_Function_uint Function
           %temp_uint_2 = OpVariable %_ptr_Function_uint Function
           %temp_uint_3 = OpVariable %_ptr_Function_uint Function
           %temp_uint_4 = OpVariable %_ptr_Function_uint Function
           %temp_uint_5 = OpVariable %_ptr_Function_uint Function
"#;

        self.source += COMMON_VARS;

        if self.code.get_type() == ShaderType::Vertex {
            const TEXT: &str = r#"
       %vertex_index_int = OpLoad %int %gl_VertexIndex
           %vertex_index = OpBitcast %float %vertex_index_int
                           OpStore %<v> %vertex_index
       %instance_index_int = OpLoad %int %gl_InstanceIndex
           %instance_index = OpBitcast %float %instance_index_int
                           OpStore %<i> %instance_index
"#;
            if self.vs_input_info.is_some_and(|i| i.gs_prolog) {
                self.source += &TEXT.replace("<v>", "v5").replace("<i>", "v8");

                // [7:0] - num_vertices, [15:8] - num_primitives
                const INIT_S3: &str = r#"
	               OpStore %s3 %uint_1
				"#;
                self.source += INIT_S3;
            } else {
                self.source += &TEXT.replace("<v>", "v0").replace("<i>", "v3");
            }
        }

        if self.code.get_type() == ShaderType::Pixel
            && self.ps_input_info.is_some_and(|i| i.ps_pos_xy)
        {
            const TEXT: &str = r#"
         %FragCoord_px = OpAccessChain %_ptr_Input_float %gl_FragCoord %uint_0
         %FragCoord_x = OpLoad %float %FragCoord_px
               OpStore %v2 %FragCoord_x
         %FragCoord_py = OpAccessChain %_ptr_Input_float %gl_FragCoord %uint_1
         %FragCoord_y = OpLoad %float %FragCoord_py
               OpStore %v3 %FragCoord_y
"#;
            self.source += TEXT;
        }

        if self.code.get_type() == ShaderType::Compute {
            const TEXT_THREAD_ID: &str = r#"
		%LocalInvocationID_114_<i> = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_<i>
        %LocalInvocationID_115_<i> = OpLoad %uint %LocalInvocationID_114_<i>
        %LocalInvocationID_116_<i> = OpBitcast %float %LocalInvocationID_115_<i>
               OpStore %v<i> %LocalInvocationID_116_<i>
"#;

            const TEXT_GROUP_ID: &str = r#"
        %WorkGroupID_120_<i> = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_<i>
        %WorkGroupID_121_<i> = OpLoad %uint %WorkGroupID_120_<i>
               OpStore %<WorkGroupReg> %WorkGroupID_121_<i>
"#;
            if let Some(info) = self.cs_input_info {
                for i in 0..info.thread_ids_num {
                    self.source += &TEXT_THREAD_ID.replace("<i>", &format!("{i}"));
                }

                let mut reg = 0;
                for i in 0..3 {
                    if info.group_id[i] {
                        self.source += &TEXT_GROUP_ID
                            .replace(
                                "<WorkGroupReg>",
                                &format!("s{}", info.workgroup_register + reg),
                            )
                            .replace("<i>", &format!("{i}"));
                        reg += 1;
                    }
                }
            }
        }

        if let Some(bind) = self.bind {
            const TEXT: &str = r#"
         %vsharp_<reg> = OpAccessChain %_ptr_PushConstant_uint %vsharp %int_0 %int_<buffer> %int_<field>
         %vsharp_value_<reg> = OpLoad %uint %vsharp_<reg>
               OpStore %<reg> %vsharp_value_<reg>
		"#;

            let mut buffer_index: i32 = 0;

            let shift_regs: i32 = if self.vs_input_info.is_some_and(|i| i.gs_prolog) {
                8
            } else {
                0
            };

            for m in &mut self.extended_mapping {
                m[0] = 0;
                m[1] = 0;
            }

            let push_slots = bind.push_constant_size as i32 / 16;
            let mut out = String::new();

            for i in 0..bind.storage_buffers.buffers_num {
                let start_reg = bind.storage_buffers.start_register[i as usize];
                let extended = bind.storage_buffers.extended[i as usize];

                if buffer_index + i >= push_slots {
                    return Err(not_supported(
                        "Spirv::WriteLocalVariables",
                        "storage buffer exceeds push constant window",
                    ));
                }

                let buffer = format!("{}", buffer_index + i);
                for f in 0..4 {
                    if extended {
                        if start_reg < 16 || shift_regs != 0 {
                            return Err(not_supported(
                                "Spirv::WriteLocalVariables",
                                "extended storage buffer mapping",
                            ));
                        }
                        let idx = (start_reg - 16 + f) as usize;
                        if idx >= self.extended_mapping.len() {
                            return Err(not_supported(
                                "Spirv::WriteLocalVariables",
                                "extended mapping overflow",
                            ));
                        }
                        self.extended_mapping[idx][0] = buffer_index + i;
                        self.extended_mapping[idx][1] = f;
                    } else {
                        let reg = format!("s{}", start_reg + f + shift_regs);
                        let field = format!("{f}");
                        out += &TEXT
                            .replace("<reg>", &reg)
                            .replace("<buffer>", &buffer)
                            .replace("<field>", &field);
                    }
                }
            }

            buffer_index += bind.storage_buffers.buffers_num;

            for i in 0..bind.textures2d.textures_num {
                let start_reg = bind.textures2d.desc[i as usize].start_register;
                let extended = bind.textures2d.desc[i as usize].extended;

                for ti in 0..2 {
                    if buffer_index + i * 2 + ti >= push_slots {
                        return Err(not_supported(
                            "Spirv::WriteLocalVariables",
                            "texture sharp exceeds push constant window",
                        ));
                    }

                    let buffer = format!("{}", buffer_index + i * 2 + ti);
                    for f in 0..4 {
                        if extended {
                            if start_reg < 16 || shift_regs != 0 {
                                return Err(not_supported(
                                    "Spirv::WriteLocalVariables",
                                    "extended texture mapping",
                                ));
                            }
                            let idx = (start_reg - 16 + 4 * ti + f) as usize;
                            if idx >= self.extended_mapping.len() {
                                return Err(not_supported(
                                    "Spirv::WriteLocalVariables",
                                    "extended mapping overflow",
                                ));
                            }
                            self.extended_mapping[idx][0] = buffer_index + i * 2 + ti;
                            self.extended_mapping[idx][1] = f;
                        } else {
                            let reg = format!("s{}", start_reg + 4 * ti + f + shift_regs);
                            let field = format!("{f}");
                            out += &TEXT
                                .replace("<reg>", &reg)
                                .replace("<buffer>", &buffer)
                                .replace("<field>", &field);
                        }
                    }
                }
            }

            buffer_index += bind.textures2d.textures_num * 2;

            for i in 0..bind.samplers.samplers_num {
                let start_reg = bind.samplers.start_register[i as usize];
                let extended = bind.samplers.extended[i as usize];

                if buffer_index + i >= push_slots {
                    return Err(not_supported(
                        "Spirv::WriteLocalVariables",
                        "sampler exceeds push constant window",
                    ));
                }

                let buffer = format!("{}", buffer_index + i);
                for f in 0..4 {
                    if extended {
                        if start_reg < 16 || shift_regs != 0 {
                            return Err(not_supported(
                                "Spirv::WriteLocalVariables",
                                "extended sampler mapping",
                            ));
                        }
                        let idx = (start_reg - 16 + f) as usize;
                        if idx >= self.extended_mapping.len() {
                            return Err(not_supported(
                                "Spirv::WriteLocalVariables",
                                "extended mapping overflow",
                            ));
                        }
                        self.extended_mapping[idx][0] = buffer_index + i;
                        self.extended_mapping[idx][1] = f;
                    } else {
                        let reg = format!("s{}", start_reg + f + shift_regs);
                        let field = format!("{f}");
                        out += &TEXT
                            .replace("<reg>", &reg)
                            .replace("<buffer>", &buffer)
                            .replace("<field>", &field);
                    }
                }
            }

            buffer_index += bind.samplers.samplers_num;

            for i in 0..bind.gds_pointers.pointers_num {
                let start_reg = bind.gds_pointers.start_register[i as usize];
                let extended = bind.gds_pointers.extended[i as usize];

                if buffer_index + i / 4 >= push_slots {
                    return Err(not_supported(
                        "Spirv::WriteLocalVariables",
                        "gds pointer exceeds push constant window",
                    ));
                }

                if extended {
                    if start_reg < 16 || shift_regs != 0 {
                        return Err(not_supported(
                            "Spirv::WriteLocalVariables",
                            "extended gds mapping",
                        ));
                    }
                    let idx = (start_reg - 16) as usize;
                    if idx >= self.extended_mapping.len() {
                        return Err(not_supported(
                            "Spirv::WriteLocalVariables",
                            "extended mapping overflow",
                        ));
                    }
                    self.extended_mapping[idx][0] = buffer_index + i / 4;
                    self.extended_mapping[idx][1] = i % 4;
                } else {
                    let buffer = format!("{}", buffer_index + i / 4);
                    let reg = format!("s{}", start_reg + shift_regs);
                    let field = format!("{}", i % 4);
                    out += &TEXT
                        .replace("<reg>", &reg)
                        .replace("<buffer>", &buffer)
                        .replace("<field>", &field);
                }
            }

            buffer_index += if bind.gds_pointers.pointers_num > 0 {
                (bind.gds_pointers.pointers_num - 1) / 4 + 1
            } else {
                0
            };

            for i in 0..bind.direct_sgprs.sgprs_num {
                let start_reg = bind.direct_sgprs.start_register[i as usize];

                if buffer_index + i / 4 >= push_slots {
                    return Err(not_supported(
                        "Spirv::WriteLocalVariables",
                        "direct sgpr exceeds push constant window",
                    ));
                }

                let buffer = format!("{}", buffer_index + i / 4);
                let reg = format!("s{}", start_reg + shift_regs);
                let field = format!("{}", i % 4);
                out += &TEXT
                    .replace("<reg>", &reg)
                    .replace("<buffer>", &buffer)
                    .replace("<field>", &field);
            }

            /* buffer_index += ... (direct sgprs; kept commented as in Kyty) */

            self.source += &out;

            if bind.extended.used {
                // TODO() load pointer
                tracing::debug!("Extended mapping: {:?}", self.extended_mapping);
            }
        }

        const COMMON_INIT: &str = r#"
               OpStore %exec_lo %uint_1
               OpStore %exec_hi %uint_0
               OpStore %execz %uint_0
               OpStore %scc %uint_0
	"#;

        self.source += COMMON_INIT;
        self.source += "\n";

        Ok(())
    }

    /// Kyty: `Spirv::WriteLabel` (L7553).
    fn write_label(&mut self, index: usize) {
        if index > 0 {
            const TEXT: &str = r#"
                   <branch>
                   %<label> = OpLabel
		"#;

            let inst_pc = self.code.get_instructions()[index].pc;
            let prev_type = self.code.get_instructions()[index - 1].type_;
            let mut labels_num = 0;
            let labels_len = self.code.get_labels().len();
            for i in (0..labels_len).rev() {
                let label = self.code.get_labels()[i];
                if !label.is_disabled() && label.get_dst() == inst_pc {
                    let discard = self.code.read_block(label.get_dst()).is_discard;

                    let skip_branch = discard
                        || ((prev_type == ShaderInstructionType::SEndpgm
                            || prev_type == ShaderInstructionType::SBranch)
                            && labels_num == 0);

                    self.source += &TEXT
                        .replace(
                            "<branch>",
                            if skip_branch { "" } else { "OpBranch %<label>" },
                        )
                        .replace("<label>", &label.to_string());
                    labels_num += 1;

                    if discard {
                        self.code.get_labels_mut()[i].disable();
                        break;
                    }
                }
            }
        }
    }

    /// Kyty: `Spirv::ModifyCode` (L7592) — duplicates discard blocks when
    /// several different branches target the same discard label.
    fn modify_code(&mut self) {
        struct DiscardLabel {
            block: crate::shader::types::ShaderControlFlowBlock,
            num: usize,
        }

        let labels = self.code.get_labels().clone();
        let mut dls: Vec<DiscardLabel> = Vec::new();
        for l in &labels {
            if !l.is_disabled() {
                let pc = l.get_dst();
                let num = labels.iter().filter(|l2| l2.get_dst() == pc).count();
                if num > 1 {
                    let block = self.code.read_block(pc);
                    if block.is_discard && !dls.iter().any(|d| d.block.pc == pc) {
                        dls.push(DiscardLabel {
                            block,
                            num: num - 1,
                        });
                    }
                }
            }
        }
        for dl in &dls {
            let block = self.code.read_intructions(&dl.block);
            for _ in 0..dl.num {
                // Duplicate discard block if there are different branches
                // with the same label
                self.code.get_instructions_mut().extend_from_slice(&block);
            }
        }
    }

    /// Kyty: `Spirv::DetectFetch` (L7639) — rewrites embedded fetch-shader
    /// `BufferLoadFormat*` loads into `Fetch*` pseudo-instructions.
    fn detect_fetch(&mut self) -> Result<(), ShaderRecompileError> {
        use ShaderInstructionType as T;

        let Some(vs) = self.vs_input_info else {
            return Err(not_supported("Spirv::DetectFetch", "no vs input info"));
        };
        if !vs.fetch_embedded {
            return Err(not_supported("Spirv::DetectFetch", "!fetch_embedded"));
        }
        if !vs.gs_prolog {
            return Err(not_supported("Spirv::DetectFetch", "!gs_prolog"));
        }
        if vs.fetch_inline {
            return Err(not_supported("Spirv::DetectFetch", "fetch_inline"));
        }

        #[derive(Copy, Clone, PartialEq, Eq, Default)]
        enum Type {
            #[default]
            Unknown,
            Attrib,
            Buffer,
            Index,
        }

        #[derive(Copy, Clone, Default)]
        struct VgprInfo {
            type_: Type,
        }

        #[derive(Copy, Clone, Default)]
        struct SgprInfo {
            type_: Type,
            attrib_id: i32,
        }

        let is_sgpr = |op: &ShaderOperand| {
            matches!(
                op.type_,
                ShaderOperandType::Sgpr | ShaderOperandType::VccLo | ShaderOperandType::VccHi
            )
        };
        let sgpr_reg = |op: &ShaderOperand| -> usize {
            match op.type_ {
                ShaderOperandType::VccLo => 106,
                ShaderOperandType::VccHi => 107,
                _ => op.register_id.max(0) as usize,
            }
        };
        let is_vgpr = |op: &ShaderOperand| op.type_ == ShaderOperandType::Vgpr;
        let vgpr_reg = |op: &ShaderOperand| -> usize { op.register_id.max(0) as usize };

        let shift_regs = 8;
        let attrib_reg = (vs.fetch_attrib_reg + shift_regs) as usize;
        let buffer_reg = (vs.fetch_buffer_reg + shift_regs) as usize;

        let mut blocks = vec![self.code.read_block(0)];
        for label in self.code.get_labels() {
            blocks.push(self.code.read_block(label.get_dst()));
        }
        for label in self.code.get_indirect_labels() {
            blocks.push(self.code.read_block(label.get_dst()));
        }

        let mut load_instructions: Vec<(ShaderInstruction, i32)> = Vec::new();

        for block in &blocks {
            let code = self.code.read_intructions(block);

            let mut sgprs = [SgprInfo::default(); 108];
            let mut vgprs = [VgprInfo::default(); 256];

            for inst in &code {
                match inst.type_ {
                    T::SLoadDword
                    | T::SLoadDwordx2
                    | T::SLoadDwordx4
                    | T::SLoadDwordx8
                    | T::SLoadDwordx16 => {
                        if is_sgpr(&inst.src[0]) && sgpr_reg(&inst.src[0]) == attrib_reg {
                            if !operand_is_constant(inst.src[1]) || inst.src[1].constant.i() < 0 {
                                return Err(not_supported(
                                    "Spirv::DetectFetch",
                                    "attrib load with non-constant/negative offset",
                                ));
                            }
                            let register_id = sgpr_reg(&inst.dst);
                            let index = inst.src[1].constant.i() / 4;
                            for i in 0..inst.dst.size {
                                let Some(s) = sgprs.get_mut(register_id + i as usize) else {
                                    return Err(not_supported(
                                        "Spirv::DetectFetch",
                                        "sgpr index out of range",
                                    ));
                                };
                                s.type_ = Type::Attrib;
                                s.attrib_id = i + index;
                            }
                        }
                        if is_sgpr(&inst.src[0]) && sgpr_reg(&inst.src[0]) == buffer_reg {
                            if operand_is_constant(inst.src[1]) {
                                return Err(not_supported(
                                    "Spirv::DetectFetch",
                                    "buffer load with constant offset",
                                ));
                            }
                            if is_sgpr(&inst.src[1])
                                && sgprs[sgpr_reg(&inst.src[1])].type_ != Type::Attrib
                            {
                                return Err(not_supported(
                                    "Spirv::DetectFetch",
                                    "buffer load index is not an attrib",
                                ));
                            }
                            let register_id = sgpr_reg(&inst.dst);
                            let attrib_id = sgprs[sgpr_reg(&inst.src[1])].attrib_id;
                            for i in 0..inst.dst.size {
                                let Some(s) = sgprs.get_mut(register_id + i as usize) else {
                                    return Err(not_supported(
                                        "Spirv::DetectFetch",
                                        "sgpr index out of range",
                                    ));
                                };
                                s.type_ = Type::Buffer;
                                s.attrib_id = attrib_id;
                            }
                        }
                    }

                    T::VCndmaskB32 => {
                        if is_vgpr(&inst.src[0])
                            && vgpr_reg(&inst.src[0]) == 8
                            && is_vgpr(&inst.src[1])
                            && vgpr_reg(&inst.src[1]) == 5
                        {
                            vgprs[vgpr_reg(&inst.dst)].type_ = Type::Index;
                        }
                    }

                    T::SBfeU32 | T::SAndB32 | T::SLshlB32 => {
                        if is_sgpr(&inst.src[0])
                            && sgprs[sgpr_reg(&inst.src[0])].type_ == Type::Attrib
                            && operand_is_constant(inst.src[1])
                        {
                            sgprs[sgpr_reg(&inst.dst)] = sgprs[sgpr_reg(&inst.src[0])];
                        }
                    }

                    T::BufferLoadFormatX
                    | T::BufferLoadFormatXy
                    | T::BufferLoadFormatXyz
                    | T::BufferLoadFormatXyzw => {
                        if !(vgprs[vgpr_reg(&inst.src[0])].type_ == Type::Index
                            && sgprs[sgpr_reg(&inst.src[1])].type_ == Type::Buffer
                            && sgprs[sgpr_reg(&inst.src[2])].type_ == Type::Attrib)
                        {
                            return Err(not_supported(
                                "Spirv::DetectFetch",
                                "unrecognized vertex-buffer load pattern",
                            ));
                        }
                        load_instructions.push((*inst, sgprs[sgpr_reg(&inst.src[1])].attrib_id));
                    }
                    _ => {}
                }
            }
        }

        for inst in self.code.get_instructions_mut() {
            if let Some(p) = load_instructions.iter().find(|p| p.0.pc == inst.pc) {
                tracing::debug!(
                    "load vertex: pc = 0x{:08x}, size = {}, attrib_id = {}",
                    p.0.pc,
                    p.0.dst.size,
                    p.1
                );

                match inst.type_ {
                    T::BufferLoadFormatX => inst.type_ = T::FetchX,
                    T::BufferLoadFormatXy => inst.type_ = T::FetchXy,
                    T::BufferLoadFormatXyz => inst.type_ = T::FetchXyz,
                    T::BufferLoadFormatXyzw => inst.type_ = T::FetchXyzw,
                    _ => {}
                }

                inst.src[2].type_ = ShaderOperandType::IntegerInlineConstant;
                inst.src[2].size = 0;
                inst.src[2].constant = ShaderConstant::from_i(p.1);
            }
        }

        Ok(())
    }

    /// Kyty: `Spirv::WriteInstructions` (L7797). Kyty `EXIT`s on
    /// unrecompilable instructions; the port returns the typed error.
    fn write_instructions(&mut self) -> Result<(), ShaderRecompileError> {
        use super::recompile::{RecompileFn, recomp_func};

        self.modify_code();

        for index in 0..self.code.get_instructions().len() {
            self.write_label(index);

            let inst = self.code.get_instructions()[index];
            let src = ShaderCode::dbg_instruction_to_str(&inst);
            let mut dst = String::new();

            let func = recomp_func(inst.type_, inst.format);

            let ok = match func {
                Some(rf) => match rf.func {
                    RecompileFn::Func(f) => f(
                        index as u32,
                        &self.code,
                        &mut dst,
                        self,
                        &rf.param,
                        rf.scc_check,
                    )?,
                    RecompileFn::NotImplemented { kyty_func, line } => {
                        tracing::error!(
                            "recompiler {kyty_func} (ShaderSpirv.cpp L{line}) not ported yet \
                             (C2): {src}"
                        );
                        return Err(ShaderRecompileError::NotImplemented {
                            kyty_func,
                            line,
                            instruction: src,
                        });
                    }
                },
                None => {
                    tracing::error!(
                        "can't recompile (no table entry for {:?}/{:?}): {src}",
                        inst.type_,
                        inst.format
                    );
                    return Err(ShaderRecompileError::UnknownTypeFormat {
                        type_: inst.type_,
                        format: inst.format,
                        instruction: src,
                    });
                }
            };

            if !ok {
                tracing::error!("can't recompile: {src}\n{}", self.source);
                return Err(ShaderRecompileError::CannotRecompile { instruction: src });
            }

            self.source += &format!("; {src}\n");
            self.source += &format!("{dst}\n");

            // Kyty: `Recompile_Inject_Debug` (L6131) injection point. C1
            // ports the data model only — debug printf injection lands with
            // C2 (needs `Config::SpirvDebugPrintfEnabled` + printf strings).
        }

        Ok(())
    }

    /// Kyty: `Spirv::WriteMainEpilog` (L7841).
    fn write_main_epilog(&mut self) {
        const TEXT: &str = r#"
                   ; Epilog
                   OpFunctionEnd
"#;
        self.source += TEXT;
    }

    /// Kyty: `Spirv::WriteFunctions` (L7851).
    ///
    /// C1 embeds only the helper texts reachable through implemented
    /// recompilers; the other Kyty branches (see module doc) are C2. Any
    /// shader that would need them fails earlier in `WriteInstructions`
    /// with a `NotImplemented` error.
    fn write_functions(&mut self) {
        use ShaderInstructionType as T;

        if self.code.has_any_of(&[
            T::SSwappcB64,
            T::FetchX,
            T::FetchXy,
            T::FetchXyz,
            T::FetchXyzw,
        ]) {
            self.source += FUNC_FETCH_1;
            self.source += FUNC_FETCH_2;
            self.source += FUNC_FETCH_3;
            self.source += FUNC_FETCH_4;
        }

        if self.code.has_any_of(&[
            T::BufferLoadDword,
            T::BufferLoadFormatX,
            T::BufferLoadFormatXy,
            T::BufferLoadFormatXyz,
            T::BufferLoadFormatXyzw,
            T::TBufferLoadFormatX,
            T::TBufferLoadFormatXyzw,
        ]) {
            self.source += BUFFER_LOAD_FLOAT1;
            self.source += BUFFER_LOAD_FLOAT4;
            self.source += TBUFFER_LOAD_FORMAT_X;
            self.source += TBUFFER_LOAD_FORMAT_XYZW;
        }

        if self.code.has_any_of(&[
            T::SBufferLoadDword,
            T::SBufferLoadDwordx2,
            T::SBufferLoadDwordx4,
            T::SBufferLoadDwordx8,
            T::SBufferLoadDwordx16,
        ]) {
            self.source += SBUFFER_LOAD_DWORD;
            self.source += SBUFFER_LOAD_DWORD_2;
            self.source += SBUFFER_LOAD_DWORD_4;
            // C2: SBUFFER_LOAD_DWORD_8 (L1016) / SBUFFER_LOAD_DWORD_16
            // (L1097) — only called from SBufferLoadDwordx8/x16, which are
            // NotImplemented in C1.
        }

        // C2 branches (guarded instructions all NotImplemented in C1):
        // VSadU32 -> FUNC_ABS_DIFF (L133); SWqmB64 -> FUNC_WQM (L149);
        // SAddcU32 -> FUNC_ADDC (L167); SLshl4AddU32 -> FUNC_LSHL_ADD (L200);
        // ImageStoreMip -> FUNC_MIPMAP (L225); VCmpO/UF32 -> FUNC_ORDERED
        // (L304); VMulLo*/VMulHi*/V(Mad|Mul)U32U24/SMulHiU32 ->
        // FUNC_MUL_EXTENDED (L338); SLshrB64/SBfeU64 -> FUNC_SHIFT_RIGHT
        // (L397); SLshlB64/SBfeU64 -> FUNC_SHIFT_LEFT (L483);
        // BufferStore* -> BUFFER_STORE_FLOAT1/2 + TBUFFER_STORE_FORMAT_X/XY
        // (L650/L679/L817/L862).
    }

    /// Kyty: `Spirv::FindConstants` (L7940).
    fn find_constants(&mut self) -> Result<(), ShaderRecompileError> {
        self.constants.clear();
        self.add_constant_float(0.0);
        self.add_constant_float(0.5);
        self.add_constant_float(1.0);
        self.add_constant_float(2.0);
        self.add_constant_float(4.0);
        for i in 0..=32 {
            self.add_constant_int(i);
            self.add_constant_uint(i as u32);
        }
        let operands: Vec<ShaderOperand> = self
            .code
            .get_instructions()
            .iter()
            .flat_map(|inst| inst.src[..inst.src_num.max(0) as usize].to_vec())
            .filter(|op| operand_is_constant(*op))
            .collect();
        for op in operands {
            self.add_constant(op)?;
        }
        if self.vs_input_info.is_some()
            || self.ps_input_info.is_some()
            || self.cs_input_info.is_some()
        {
            self.add_constant_int(12);
            self.add_constant_int(16);
            self.add_constant_int(31);
            self.add_constant_int(36);
            self.add_constant_int(39);
            self.add_constant_int(92);
            self.add_constant_int(95);
            self.add_constant_int(119);
            self.add_constant_uint(24);
            self.add_constant_uint(31);
            self.add_constant_uint(32);
            self.add_constant_uint(63);
            self.add_constant_uint(64);
            self.add_constant_uint(72);
            self.add_constant_uint(127);
            self.add_constant_uint(0x3fff);
            self.add_constant_uint(0x00ff_ffff);
            self.add_constant_uint(0xffff_e000);
            self.add_constant_uint(0xffff_ffff);
            self.add_constant_uint(0x0000_000f);
            self.add_constant_uint(0x0000_00f0);
            self.add_constant_uint(0x0000_0f00);
            self.add_constant_uint(0x0000_f000);
            self.add_constant_uint(0x000f_0000);
            self.add_constant_uint(0x00f0_0000);
            self.add_constant_uint(0x0f00_0000);
            self.add_constant_uint(0xf000_0000);
        }
        if let Some(info) = self.cs_input_info {
            self.add_constant_uint(info.threads_num[0]);
            self.add_constant_uint(info.threads_num[1]);
            self.add_constant_uint(info.threads_num[2]);
        }
        Ok(())
    }

    /// Kyty: `Spirv::FindVariables` (L8001).
    fn find_variables(&mut self) {
        self.variables.clear();

        self.add_variable_parts(ShaderOperandType::Vgpr, 0, 1);
        self.add_variable_parts(ShaderOperandType::ExecLo, 0, 2);
        self.add_variable_parts(ShaderOperandType::ExecZ, 0, 1);
        self.add_variable_parts(ShaderOperandType::Scc, 0, 1);

        let insts: Vec<ShaderInstruction> = self.code.get_instructions().clone();
        for inst in &insts {
            self.add_variable(inst.dst);
            self.add_variable(inst.dst2);
            for i in 0..inst.src_num.max(0) as usize {
                self.add_variable(inst.src[i]);
            }
        }

        if let Some(info) = self.vs_input_info {
            if info.gs_prolog {
                self.add_variable_parts(ShaderOperandType::Vgpr, 5, 1);
                self.add_variable_parts(ShaderOperandType::Vgpr, 8, 1);
            } else {
                self.add_variable_parts(ShaderOperandType::Vgpr, 3, 1);
            }
            for i in 0..info.resources_num as usize {
                self.add_variable_parts(
                    ShaderOperandType::Vgpr,
                    info.resources_dst[i].register_start,
                    info.resources_dst[i].registers_num,
                );
            }
        }

        if let Some(info) = self.ps_input_info {
            if info.ps_pos_xy {
                self.add_variable_parts(ShaderOperandType::Vgpr, 2, 1);
                self.add_variable_parts(ShaderOperandType::Vgpr, 3, 1);
            }
        }

        if let Some(info) = self.cs_input_info {
            self.add_variable_parts(ShaderOperandType::Vgpr, 0, 3);
            self.add_variable_parts(ShaderOperandType::Sgpr, info.workgroup_register, 3);
        }

        if let Some(bind) = self.bind {
            let shift_regs = if self.vs_input_info.is_some_and(|i| i.gs_prolog) {
                8
            } else {
                0
            };

            for i in 0..bind.storage_buffers.buffers_num as usize {
                let storage_start = bind.storage_buffers.start_register[i] + shift_regs;
                self.add_variable_parts(ShaderOperandType::Sgpr, storage_start, 4);
            }
            for i in 0..bind.textures2d.textures_num as usize {
                let storage_start = bind.textures2d.desc[i].start_register + shift_regs;
                self.add_variable_parts(ShaderOperandType::Sgpr, storage_start, 8);
            }
            for i in 0..bind.samplers.samplers_num as usize {
                let storage_start = bind.samplers.start_register[i] + shift_regs;
                self.add_variable_parts(ShaderOperandType::Sgpr, storage_start, 8);
            }
        }
    }
}

/// Kyty: ShaderSpirv.cpp `SpirvGenerateSource` (L8074).
pub fn spirv_generate_source(
    code: &ShaderCode,
    vs_input_info: Option<&ShaderVertexInputInfo>,
    ps_input_info: Option<&ShaderPixelInputInfo>,
    cs_input_info: Option<&ShaderComputeInputInfo>,
) -> Result<String, ShaderRecompileError> {
    let mut spirv = Spirv::new();
    spirv.set_code(code);
    spirv.set_vs_input_info(vs_input_info);
    spirv.set_ps_input_info(ps_input_info);
    spirv.set_cs_input_info(cs_input_info);
    spirv.generate_source()?;
    Ok(spirv.get_source().to_string())
}

/// Kyty: ShaderSpirv.cpp `SpirvGetEmbeddedVs` (L8087).
pub fn spirv_get_embedded_vs(id: u32) -> Result<&'static str, ShaderRecompileError> {
    if id != 0 {
        return Err(not_supported(
            "SpirvGetEmbeddedVs",
            format!("embedded vs id {id}"),
        ));
    }
    Ok(EMBEDDED_SHADER_VS_0)
}

/// Kyty: ShaderSpirv.cpp `SpirvGetEmbeddedPs` (L8094).
pub fn spirv_get_embedded_ps(id: u32) -> Result<&'static str, ShaderRecompileError> {
    if id != 0 {
        return Err(not_supported(
            "SpirvGetEmbeddedPs",
            format!("embedded ps id {id}"),
        ));
    }
    Ok(EMBEDDED_SHADER_PS_0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::types::ShaderOperand;

    fn vgpr(id: i32) -> ShaderOperand {
        ShaderOperand {
            type_: ShaderOperandType::Vgpr,
            register_id: id,
            size: 1,
            ..Default::default()
        }
    }

    fn sgpr_n(id: i32, size: i32) -> ShaderOperand {
        ShaderOperand {
            type_: ShaderOperandType::Sgpr,
            register_id: id,
            size,
            ..Default::default()
        }
    }

    #[test]
    fn operand_variable_names_and_types() {
        // Kyty: operand_variable_to_str (L1577).
        let v = operand_variable_to_str(vgpr(7));
        assert_eq!((v.type_, v.value.as_str()), (SpirvType::Float, "v7"));
        let s = operand_variable_to_str(sgpr_n(12, 1));
        assert_eq!((s.type_, s.value.as_str()), (SpirvType::Uint, "s12"));

        for (type_, name) in [
            (ShaderOperandType::VccLo, "vcc_lo"),
            (ShaderOperandType::VccHi, "vcc_hi"),
            (ShaderOperandType::ExecLo, "exec_lo"),
            (ShaderOperandType::ExecHi, "exec_hi"),
            (ShaderOperandType::ExecZ, "execz"),
            (ShaderOperandType::Scc, "scc"),
            (ShaderOperandType::M0, "m0"),
        ] {
            let op = ShaderOperand {
                type_,
                size: 1,
                ..Default::default()
            };
            let v = operand_variable_to_str(op);
            assert_eq!(v.value, name);
            assert_eq!(v.type_, SpirvType::Uint);
        }
    }

    #[test]
    fn operand_variable_multi_dword_shift() {
        // Kyty: operand_variable_to_str shift overload (L1627).
        let v = operand_variable_to_str_shift(
            ShaderOperand {
                type_: ShaderOperandType::Vgpr,
                register_id: 4,
                size: 4,
                ..Default::default()
            },
            3,
        );
        assert_eq!(v.value, "v7");
        let s = operand_variable_to_str_shift(sgpr_n(8, 2), 1);
        assert_eq!(s.value, "s9");

        let vcc = ShaderOperand {
            type_: ShaderOperandType::VccLo,
            size: 2,
            ..Default::default()
        };
        assert_eq!(operand_variable_to_str_shift(vcc, 0).value, "vcc_lo");
        assert_eq!(operand_variable_to_str_shift(vcc, 1).value, "vcc_hi");
        let exec = ShaderOperand {
            type_: ShaderOperandType::ExecLo,
            size: 2,
            ..Default::default()
        };
        assert_eq!(operand_variable_to_str_shift(exec, 0).value, "exec_lo");
        assert_eq!(operand_variable_to_str_shift(exec, 1).value, "exec_hi");
        // Unmapped shift -> Unknown.
        assert_eq!(
            operand_variable_to_str_shift(exec, 2).type_,
            SpirvType::Unknown
        );
    }

    #[test]
    fn constant_ids_match_kyty_formats() {
        // Kyty: Spirv::AddConstant (L6510) — three literal formats
        // (L6527-6535) + id derivation (L6538).
        let mut spirv = Spirv::new();
        spirv.add_constant_uint(15);
        spirv.add_constant_uint(0x3fff);
        spirv.add_constant_int(-1);
        spirv.add_constant_float(1.0);
        spirv.add_constant_float(-0.5);

        assert_eq!(spirv.get_constant_uint(15), "uint_15");
        assert_eq!(spirv.get_constant_uint(0x3fff), "uint_0x00003fff");
        assert_eq!(spirv.get_constant_int(-1), "int_m1");
        assert_eq!(spirv.get_constant_float(1.0), "float_1_000000");
        assert_eq!(spirv.get_constant_float(-0.5), "float_m0_500000");
        // Missing constants keep Kyty's sentinel names.
        assert_eq!(spirv.get_constant_uint(9999), "unknown_uint_constant");
        assert_eq!(spirv.get_constant_int(-77), "unknown_int_constant");
        assert_eq!(spirv.get_constant_float(3.5), "unknown_float_constant");
    }

    #[test]
    fn constants_deduplicate_by_type_and_bits() {
        let mut spirv = Spirv::new();
        spirv.add_constant_uint(1);
        spirv.add_constant_uint(1);
        spirv.add_constant_int(1);
        assert_eq!(spirv.constants.len(), 2);
    }

    #[test]
    fn generate_source_empty_vs_section_order() {
        // Kyty: Spirv::GenerateSource (L6652) ordering.
        let code = {
            let mut c = ShaderCode::new();
            c.set_type(ShaderType::Vertex);
            c
        };
        let input_info = ShaderVertexInputInfo::default();
        let source = spirv_generate_source(&code, Some(&input_info), None, None).unwrap();

        let order = [
            "; Header",
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "OpEntryPoint Vertex %main \"main\"",
            "; Annotations",
            "OpDecorate %gl_VertexIndex BuiltIn VertexIndex",
            "; Types",
            "%void = OpTypeVoid",
            "; Constants",
            "%float_2pi = OpConstant %float 6.283185307179586476925286766559",
            ";Variables",
            "%outPerVertex = OpVariable %_ptr_Output_gl_PerVertex Output",
            "; Function main",
            "%main       = OpFunction %void None %function_void",
            "; Registers",
            "%exec_lo = OpVariable %_ptr_Function_uint Function",
            "OpStore %exec_lo %uint_1",
            "; Epilog",
            "OpFunctionEnd",
        ];
        let mut last = 0;
        for needle in order {
            let pos = source[last..]
                .find(needle)
                .unwrap_or_else(|| panic!("missing/misordered {needle:?} in:\n{source}"));
            last += pos;
        }

        // VS interface variables listed on the entry point (header L6764).
        assert!(
            source.contains(
                "OpEntryPoint Vertex %main \"main\" %gl_VertexIndex %gl_InstanceIndex %outPerVertex"
            ),
            "{source}"
        );
        // Vertex-index prolog init (WriteLocalVariables L7311, non-gs_prolog).
        assert!(source.contains("OpStore %v0 %vertex_index"), "{source}");
        assert!(source.contains("OpStore %v3 %instance_index"), "{source}");
    }

    #[test]
    fn generate_source_unknown_type_is_error() {
        let code = ShaderCode::new(); // type Unknown
        let err = spirv_generate_source(&code, None, None, None).unwrap_err();
        assert_eq!(err, ShaderRecompileError::UnknownShaderType);
    }

    #[test]
    fn operand_load_float_modifiers() {
        // Kyty: operand_load_float (L1791) — abs/neg wrap the load.
        let spirv = Spirv::new();
        let mut op = vgpr(1);
        op.absolute = true;
        op.negate = true;
        let mut load = String::new();
        assert!(operand_load_float(&spirv, op, "t0_3", "3", &mut load).unwrap());
        assert!(load.contains("%at0_3 = OpLoad %float %v1"), "{load}");
        assert!(
            load.contains("%abs_3 = OpExtInst %float %GLSL_std_450 FAbs %at0_3"),
            "{load}"
        );
        assert!(load.contains("%t0_3 = OpFNegate %float %abs_3"), "{load}");
    }

    #[test]
    fn operand_load_uint_64bit_constant_high_half() {
        // Kyty: operand_load_uint (L1723) — size-2 constants split into
        // low half (the literal) and high half (0 or sign extension).
        let mut spirv = Spirv::new();
        let mut op = ShaderOperand {
            type_: ShaderOperandType::IntegerInlineConstant,
            size: 2,
            ..Default::default()
        };
        op.constant = ShaderConstant::from_i(-1);
        spirv.add_constant(op).unwrap();

        let mut load = String::new();
        assert!(operand_load_uint(&spirv, op, "t1_0", "0", &mut load, 1).unwrap());
        assert!(
            load.contains("%t1_0 = OpBitcast %uint %uint_0xffffffff"),
            "{load}"
        );

        let mut load0 = String::new();
        assert!(operand_load_uint(&spirv, op, "t0_0", "0", &mut load0, 0).unwrap());
        assert!(load0.contains("%t0_0 = OpBitcast %uint %int_m1"), "{load0}");

        // Non-negative 64-bit constant: high half is 0.
        let mut op2 = op;
        op2.constant = ShaderConstant::from_i(3);
        let mut load2 = String::new();
        assert!(operand_load_uint(&spirv, op2, "t1_1", "1", &mut load2, 1).unwrap());
        assert!(load2.contains("%t1_1 = OpBitcast %uint %uint_0"), "{load2}");
    }

    #[test]
    fn embedded_shaders_assemble() {
        // Kyty: EMBEDDED_SHADER_VS_0 (L1244) / EMBEDDED_SHADER_PS_0 (L1327).
        for (name, text) in [
            ("vs0", spirv_get_embedded_vs(0).unwrap()),
            ("ps0", spirv_get_embedded_ps(0).unwrap()),
        ] {
            let words = crate::spirv_asm::assemble(text)
                .unwrap_or_else(|e| panic!("{name} failed to assemble: {e}"));
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let module =
                naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default());
            assert!(module.is_ok(), "naga rejected {name}: {:?}", module.err());
        }
        assert!(spirv_get_embedded_vs(1).is_err());
        assert!(spirv_get_embedded_ps(1).is_err());
    }
}
