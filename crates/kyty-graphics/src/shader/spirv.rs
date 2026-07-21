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
//!
//! C2 completes `WriteFunctions` (all Kyty helper texts, in Kyty branch
//! order) and the `Recompile_Inject_Debug` injection point in
//! `WriteInstructions` (L7834).

use std::fmt;

use crate::hw_regs::UserSgprInfo;
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

/// Kyty: ShaderSpirv.cpp `FUNC_ABS_DIFF` (L133).
pub(crate) const FUNC_ABS_DIFF: &str = r#"
                    ; uint abs_diff(uint u1, uint u2)
                    ; {
                    ; 	return max(u1,u2)-min(u1,u2);	
                    ; }
%abs_diff = OpFunction %uint None %function_u_u
         %abs_diff_18 = OpFunctionParameter %uint
         %abs_diff_19 = OpFunctionParameter %uint
         %abs_diff_21 = OpLabel
         %abs_diff_50 = OpExtInst %uint %GLSL_std_450 UMax %abs_diff_18 %abs_diff_19
         %abs_diff_53 = OpExtInst %uint %GLSL_std_450 UMin %abs_diff_18 %abs_diff_19
         %abs_diff_54 = OpISub %uint %abs_diff_50 %abs_diff_53
               OpReturnValue %abs_diff_54
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_WQM` (L149).
pub(crate) const FUNC_WQM: &str = r#"
                    ; uint w(uint u, uint s, uint m)
                    ; {
                    ; 	return ((u >> s) & 0xF) != 0 ? m : 0;
                    ; }
         %wqm = OpFunction %uint None %function_u_u_u
         %wqm_155 = OpFunctionParameter %uint
         %wqm_156 = OpFunctionParameter %uint
         %wqm_161 = OpFunctionParameter %uint
         %wqm_50 = OpLabel
        %wqm_157 = OpShiftRightLogical %uint %wqm_155 %wqm_156
        %wqm_159 = OpBitwiseAnd %uint %wqm_157 %uint_15
        %wqm_160 = OpINotEqual %bool %wqm_159 %uint_0
        %wqm_162 = OpSelect %uint %wqm_160 %wqm_161 %uint_0
               OpReturnValue %wqm_162
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_ADDC` (L167).
pub(crate) const FUNC_ADDC: &str = r#"
                  ; uvec2 addc(uint a, uint b, uint c)
                  ; {
                  ; 	uint cc = 0;
                  ; 	uint sum = uaddCarry(a, b, cc) + c;
                  ; 	return uvec2(sum, (cc != 0 || (c !=0 && sum == 0)) ? 1u : 0u);
                  ; }
         %addc = OpFunction %v2uint None %function_u2_u_u_u
         %addc_47 = OpFunctionParameter %uint
         %addc_48 = OpFunctionParameter %uint
         %addc_49 = OpFunctionParameter %uint
         %addc_51 = OpLabel
        %addc_156 = OpIAddCarry %ResTypeU %addc_47 %addc_48
        %addc_157 = OpCompositeExtract %uint %addc_156 1
        %addc_158 = OpCompositeExtract %uint %addc_156 0
        %addc_160 = OpIAdd %uint %addc_158 %addc_49
        %addc_163 = OpINotEqual %bool %addc_157 %uint_0
        %addc_164 = OpLogicalNot %bool %addc_163
               OpSelectionMerge %addc_166 None
               OpBranchConditional %addc_164 %addc_165 %addc_166
        %addc_165 = OpLabel
        %addc_168 = OpINotEqual %bool %addc_49 %uint_0
        %addc_170 = OpIEqual %bool %addc_160 %uint_0
        %addc_171 = OpLogicalAnd %bool %addc_168 %addc_170
               OpBranch %addc_166
        %addc_166 = OpLabel
        %addc_172 = OpPhi %bool %addc_163 %addc_51 %addc_171 %addc_165
        %addc_173 = OpSelect %uint %addc_172 %uint_1 %uint_0
        %addc_174 = OpCompositeConstruct %v2uint %addc_160 %addc_173
               OpReturnValue %addc_174
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_LSHL_ADD` (L200).
pub(crate) const FUNC_LSHL_ADD: &str = r#"
                  ; uvec2 lshl_add(uint a, uint b, uint n)                                
                  ; {                                                                 
                  ; 	uint cc = 0;                                                  
                  ; 	uint sum = uaddCarry(a << n, b, cc);                           
                  ; 	return uvec2(sum, ((a >> (32-n)) !=0) ? 1u : cc);
                  ; }                                                                
        %lshl_add = OpFunction %v2uint None %function_u2_u_u_u
         %ladd_25 = OpFunctionParameter %uint
         %ladd_26 = OpFunctionParameter %uint
         %ladd_27 = OpFunctionParameter %uint
         %ladd_29 = OpLabel
        %ladd_124 = OpShiftLeftLogical %uint %ladd_25 %ladd_27
        %ladd_127 = OpIAddCarry %ResTypeU %ladd_124 %ladd_26
        %ladd_128 = OpCompositeExtract %uint %ladd_127 1
        %ladd_129 = OpCompositeExtract %uint %ladd_127 0
        %ladd_133 = OpISub %uint %uint_32 %ladd_27
        %ladd_134 = OpShiftRightLogical %uint %ladd_25 %ladd_133
        %ladd_135 = OpINotEqual %bool %ladd_134 %uint_0
        %ladd_138 = OpSelect %uint %ladd_135 %uint_1 %ladd_128
        %ladd_139 = OpCompositeConstruct %v2uint %ladd_129 %ladd_138
               OpReturnValue %ladd_139
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_MIPMAP` (L225).
pub(crate) const FUNC_MIPMAP: &str = r#"
                  ; uvec2 mipmap(uint lod, uint width, uint height)
                  ; {
                  ; 	uint mip_width  = width;
                  ; 	uint mip_height = height;
                  ; 	uint mip_x      = 0;
                  ; 	uint mip_y      = 0;
                  ; 	for (uint i = 0; i < 16; i++)
                  ; 	{
                  ; 		if (i == lod)
                  ; 		{
                  ; 			return uvec2(mip_x, mip_y);
                  ; 		}
                  ; 		bool odd = ((i & 1u) != 0u);
                  ; 		mip_x += (odd ? mip_width : 0u);
                  ; 		mip_y += (odd ? 0u : mip_height);
                  ; 		mip_width >>= (mip_width > 1u ? 1u : 0u);
                  ; 		mip_height >>= (mip_height > 1u ? 1u : 0u);
                  ; 	}
                  ; 	return uvec2(mip_x, mip_y);
                  ; }
         %mipmap = OpFunction %v2uint None %function_u2_u_u_u
         %mipmap_33 = OpFunctionParameter %uint
         %mipmap_16 = OpFunctionParameter %uint
         %mipmap_18 = OpFunctionParameter %uint		 
         %mipmap_14 = OpLabel
               OpSelectionMerge %mipmap_188 None
               OpSwitch %uint_0 %mipmap_191
        %mipmap_191 = OpLabel
               OpBranch %mipmap_23
         %mipmap_23 = OpLabel
        %mipmap_296 = OpPhi %uint %uint_0 %mipmap_191 %mipmap_56 %mipmap_26
        %mipmap_295 = OpPhi %uint %mipmap_18 %mipmap_191 %mipmap_66 %mipmap_26
        %mipmap_294 = OpPhi %uint %uint_0 %mipmap_191 %mipmap_51 %mipmap_26
        %mipmap_293 = OpPhi %uint %mipmap_16 %mipmap_191 %mipmap_61 %mipmap_26
        %mipmap_292 = OpPhi %uint %uint_0 %mipmap_191 %mipmap_70 %mipmap_26
               OpLoopMerge %mipmap_25 %mipmap_26 None
               OpBranch %mipmap_27
         %mipmap_27 = OpLabel
         %mipmap_31 = OpULessThan %bool %mipmap_292 %uint_16
               OpBranchConditional %mipmap_31 %mipmap_24 %mipmap_25
         %mipmap_24 = OpLabel
         %mipmap_34 = OpIEqual %bool %mipmap_292 %mipmap_33
               OpSelectionMerge %mipmap_36 None
               OpBranchConditional %mipmap_34 %mipmap_35 %mipmap_36
         %mipmap_35 = OpLabel
         %mipmap_39 = OpCompositeConstruct %v2uint %mipmap_294 %mipmap_296
               OpBranch %mipmap_25
         %mipmap_36 = OpLabel
         %mipmap_45 = OpBitwiseAnd %uint %mipmap_292 %uint_1
         %mipmap_46 = OpINotEqual %bool %mipmap_45 %uint_0
         %mipmap_49 = OpSelect %uint %mipmap_46 %mipmap_293 %uint_0
         %mipmap_51 = OpIAdd %uint %mipmap_294 %mipmap_49
         %mipmap_54 = OpSelect %uint %mipmap_46 %uint_0 %mipmap_295
         %mipmap_56 = OpIAdd %uint %mipmap_296 %mipmap_54
         %mipmap_58 = OpUGreaterThan %bool %mipmap_293 %uint_1
         %mipmap_59 = OpSelect %uint %mipmap_58 %uint_1 %uint_0
         %mipmap_61 = OpShiftRightLogical %uint %mipmap_293 %mipmap_59
         %mipmap_63 = OpUGreaterThan %bool %mipmap_295 %uint_1
         %mipmap_64 = OpSelect %uint %mipmap_63 %uint_1 %uint_0
         %mipmap_66 = OpShiftRightLogical %uint %mipmap_295 %mipmap_64
               OpBranch %mipmap_26
         %mipmap_26 = OpLabel
         %mipmap_70 = OpIAdd %uint %mipmap_292 %int_1
               OpBranch %mipmap_23
         %mipmap_25 = OpLabel
        %mipmap_302 = OpPhi %v2uint %undef_v2uint %mipmap_27 %mipmap_39 %mipmap_35
        %mipmap_297 = OpPhi %bool %false %mipmap_27 %true %mipmap_35
               OpSelectionMerge %mipmap_195 None
               OpBranchConditional %mipmap_297 %mipmap_188 %mipmap_195
        %mipmap_195 = OpLabel
         %mipmap_73 = OpCompositeConstruct %v2uint %mipmap_294 %mipmap_296
               OpBranch %mipmap_188
        %mipmap_188 = OpLabel
        %mipmap_301 = OpPhi %v2uint %mipmap_302 %mipmap_25 %mipmap_73 %mipmap_195
               OpReturnValue %mipmap_301
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_ORDERED` (L304).
pub(crate) const FUNC_ORDERED: &str = r#"
                  ; bool unordered(float f1, float f2)
                  ; {
                  ; 	return (isnan(f1) || isnan(f2));
                  ; }
                  ; bool ordered(float f1, float f2)
                  ; {
                  ; 	return !unordered(f1, f2);
                  ; }
  %unordered = OpFunction %bool None %function_b_f_f
         %ord_49 = OpFunctionParameter %float
         %ord_50 = OpFunctionParameter %float
         %ord_52 = OpLabel
        %ord_156 = OpIsNan %bool %ord_49
        %ord_157 = OpLogicalNot %bool %ord_156
               OpSelectionMerge %ord_159 None
               OpBranchConditional %ord_157 %ord_158 %ord_159
        %ord_158 = OpLabel
        %ord_161 = OpIsNan %bool %ord_50
               OpBranch %ord_159
        %ord_159 = OpLabel
        %ord_162 = OpPhi %bool %ord_156 %ord_52 %ord_161 %ord_158
               OpReturnValue %ord_162
               OpFunctionEnd
    %ordered = OpFunction %bool None %function_b_f_f
         %ord_53 = OpFunctionParameter %float
         %ord_54 = OpFunctionParameter %float
         %ord_56 = OpLabel
        %ord_169 = OpFunctionCall %bool %unordered %ord_53 %ord_54
        %ord_170 = OpLogicalNot %bool %ord_169
               OpReturnValue %ord_170
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_MUL_EXTENDED` (L338).
pub(crate) const FUNC_MUL_EXTENDED: &str = r#"
               ; uint mul_lo_uint(uint u1, uint u2)
               ; {
               ; 	uint r1, r2;
               ; 	umulExtended(u1, u2, r1, r2);
               ; 	return r2;
               ; }
               ; uint mul_hi_uint(uint u1, uint u2)
               ; {
               ; 	uint r1, r2;
               ; 	umulExtended(u1, u2, r1, r2);
               ; 	return r1;
               ; }
               ; int mul_lo_int(int i1, int i2)
               ; {
               ; 	int r1, r2;
               ; 	imulExtended(i1, i2, r1, r2);
               ; 	return r2;
               ; }
               ; int mul_hi_int(int i1, int i2)
               ; {
               ; 	int r1, r2;
               ; 	imulExtended(i1, i2, r1, r2);
               ; 	return r1;
               ; }
         %mul_lo_uint = OpFunction %uint None %function_u_u
         %22 = OpFunctionParameter %uint
         %23 = OpFunctionParameter %uint
         %25 = OpLabel
         %79 = OpUMulExtended %ResTypeU %22 %23
         %80 = OpCompositeExtract %uint %79 0
               OpReturnValue %80
               OpFunctionEnd
         %mul_hi_uint = OpFunction %uint None %function_u_u
         %26 = OpFunctionParameter %uint
         %27 = OpFunctionParameter %uint
         %29 = OpLabel
         %89 = OpUMulExtended %ResTypeU %26 %27
         %91 = OpCompositeExtract %uint %89 1
               OpReturnValue %91
               OpFunctionEnd
         %mul_lo_int = OpFunction %int None %function_i_i
         %31 = OpFunctionParameter %int
         %32 = OpFunctionParameter %int
         %34 = OpLabel
        %100 = OpSMulExtended %ResTypeI %31 %32
        %101 = OpCompositeExtract %int %100 0
               OpReturnValue %101
               OpFunctionEnd
         %mul_hi_int = OpFunction %int None %function_i_i
         %35 = OpFunctionParameter %int
         %36 = OpFunctionParameter %int
         %38 = OpLabel
        %110 = OpSMulExtended %ResTypeI %35 %36
        %112 = OpCompositeExtract %int %110 1
               OpReturnValue %112
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_SHIFT_RIGHT` (L397).
pub(crate) const FUNC_SHIFT_RIGHT: &str = r#"
                    ; void shift_r(out uint d0, out uint d1, in uint s0, in uint s1, in uint n)
                    ; {
                    ; 	d0 = n < 32 ? (s0 >> n) | (n != 0 ? (s1 << (32 - n)) : 0) : (n < 64 ? s1 >> (n - 32) : 0) ;
                    ; 	d1 = n < 32 ? s1 >> n : 0;
                    ; }
%shift_right = OpFunction %void None %function_shift
          %shr_9 = OpFunctionParameter %_ptr_Function_uint
         %shr_10 = OpFunctionParameter %_ptr_Function_uint
         %shr_11 = OpFunctionParameter %_ptr_Function_uint
         %shr_12 = OpFunctionParameter %_ptr_Function_uint
         %shr_13 = OpFunctionParameter %_ptr_Function_uint
         %shr_15 = OpLabel
         %shr_27 = OpVariable %_ptr_Function_uint Function
         %shr_36 = OpVariable %_ptr_Function_uint Function
         %shr_50 = OpVariable %_ptr_Function_uint Function
         %shr_62 = OpVariable %_ptr_Function_uint Function
         %shr_23 = OpLoad %uint %shr_13
         %shr_26 = OpULessThan %bool %shr_23 %uint_32
               OpSelectionMerge %shr_29 None
               OpBranchConditional %shr_26 %shr_28 %shr_46
         %shr_28 = OpLabel
         %shr_30 = OpLoad %uint %shr_11
         %shr_31 = OpLoad %uint %shr_13
         %shr_32 = OpShiftRightLogical %uint %shr_30 %shr_31
         %shr_33 = OpLoad %uint %shr_13
         %shr_35 = OpINotEqual %bool %shr_33 %uint_0
               OpSelectionMerge %shr_38 None
               OpBranchConditional %shr_35 %shr_37 %shr_43
         %shr_37 = OpLabel
         %shr_39 = OpLoad %uint %shr_12
         %shr_40 = OpLoad %uint %shr_13
         %shr_41 = OpISub %uint %uint_32 %shr_40
         %shr_42 = OpShiftLeftLogical %uint %shr_39 %shr_41
               OpStore %shr_36 %shr_42
               OpBranch %shr_38
         %shr_43 = OpLabel
               OpStore %shr_36 %uint_0
               OpBranch %shr_38
         %shr_38 = OpLabel
        %shr_331 = OpPhi %uint %shr_42 %shr_37 %uint_0 %shr_43
         %shr_45 = OpBitwiseOr %uint %shr_32 %shr_331
               OpStore %shr_27 %shr_45
               OpBranch %shr_29
         %shr_46 = OpLabel
         %shr_47 = OpLoad %uint %shr_13
         %shr_49 = OpULessThan %bool %shr_47 %uint_64
               OpSelectionMerge %shr_52 None
               OpBranchConditional %shr_49 %shr_51 %shr_57
         %shr_51 = OpLabel
         %shr_53 = OpLoad %uint %shr_12
         %shr_54 = OpLoad %uint %shr_13
         %shr_55 = OpISub %uint %shr_54 %uint_32
         %shr_56 = OpShiftRightLogical %uint %shr_53 %shr_55
               OpStore %shr_50 %shr_56
               OpBranch %shr_52
         %shr_57 = OpLabel
               OpStore %shr_50 %uint_0
               OpBranch %shr_52
         %shr_52 = OpLabel
        %shr_330 = OpPhi %uint %shr_56 %shr_51 %uint_0 %shr_57
               OpStore %shr_27 %shr_330
               OpBranch %shr_29
         %shr_29 = OpLabel
        %shr_332 = OpPhi %uint %shr_45 %shr_38 %shr_330 %shr_52
               OpStore %shr_9 %shr_332
         %shr_60 = OpLoad %uint %shr_13
         %shr_61 = OpULessThan %bool %shr_60 %uint_32
               OpSelectionMerge %shr_64 None
               OpBranchConditional %shr_61 %shr_63 %shr_68
         %shr_63 = OpLabel
         %shr_65 = OpLoad %uint %shr_12
         %shr_66 = OpLoad %uint %shr_13
         %shr_67 = OpShiftRightLogical %uint %shr_65 %shr_66
               OpStore %shr_62 %shr_67
               OpBranch %shr_64
         %shr_68 = OpLabel
               OpStore %shr_62 %uint_0
               OpBranch %shr_64
         %shr_64 = OpLabel
        %shr_333 = OpPhi %uint %shr_67 %shr_63 %uint_0 %shr_68
               OpStore %shr_10 %shr_333
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `FUNC_SHIFT_LEFT` (L483).
pub(crate) const FUNC_SHIFT_LEFT: &str = r#"
                    ; void shift_l(out uint d0, out uint d1, in uint s0, in uint s1, in uint n)
                    ; {
                    ; 	d0 = n < 32 ? s0 << n : 0;
                    ; 	d1 = n < 32 ? (n!=0 ? s0 >> (32-n) : 0) | (s1 << n) : (n < 64 ? s0 << (n-32) : 0);
                    ; }
%shift_left = OpFunction %void None %function_shift
         %shl_16 = OpFunctionParameter %_ptr_Function_uint
         %shl_17 = OpFunctionParameter %_ptr_Function_uint
         %shl_18 = OpFunctionParameter %_ptr_Function_uint
         %shl_19 = OpFunctionParameter %_ptr_Function_uint
         %shl_20 = OpFunctionParameter %_ptr_Function_uint
         %shl_22 = OpLabel
         %shl_72 = OpVariable %_ptr_Function_uint Function
         %shl_82 = OpVariable %_ptr_Function_uint Function
         %shl_87 = OpVariable %_ptr_Function_uint Function
        %shl_103 = OpVariable %_ptr_Function_uint Function
         %shl_70 = OpLoad %uint %shl_20
         %shl_71 = OpULessThan %bool %shl_70 %uint_32
               OpSelectionMerge %shl_74 None
               OpBranchConditional %shl_71 %shl_73 %shl_78
         %shl_73 = OpLabel
         %shl_75 = OpLoad %uint %shl_18
         %shl_76 = OpLoad %uint %shl_20
         %shl_77 = OpShiftLeftLogical %uint %shl_75 %shl_76
               OpStore %shl_72 %shl_77
               OpBranch %shl_74
         %shl_78 = OpLabel
               OpStore %shl_72 %uint_0
               OpBranch %shl_74
         %shl_74 = OpLabel
        %shl_334 = OpPhi %uint %shl_77 %shl_73 %uint_0 %shl_78
               OpStore %shl_16 %shl_334
         %shl_80 = OpLoad %uint %shl_20
         %shl_81 = OpULessThan %bool %shl_80 %uint_32
               OpSelectionMerge %shl_84 None
               OpBranchConditional %shl_81 %shl_83 %shl_100
         %shl_83 = OpLabel
         %shl_85 = OpLoad %uint %shl_20
         %shl_86 = OpINotEqual %bool %shl_85 %uint_0
               OpSelectionMerge %shl_89 None
               OpBranchConditional %shl_86 %shl_88 %shl_94
         %shl_88 = OpLabel
         %shl_90 = OpLoad %uint %shl_18
         %shl_91 = OpLoad %uint %shl_20
         %shl_92 = OpISub %uint %uint_32 %shl_91
         %shl_93 = OpShiftRightLogical %uint %shl_90 %shl_92
               OpStore %shl_87 %shl_93
               OpBranch %shl_89
         %shl_94 = OpLabel
               OpStore %shl_87 %uint_0
               OpBranch %shl_89
         %shl_89 = OpLabel
        %shl_336 = OpPhi %uint %shl_93 %shl_88 %uint_0 %shl_94
         %shl_96 = OpLoad %uint %shl_19
         %shl_97 = OpLoad %uint %shl_20
         %shl_98 = OpShiftLeftLogical %uint %shl_96 %shl_97
         %shl_99 = OpBitwiseOr %uint %shl_336 %shl_98
               OpStore %shl_82 %shl_99
               OpBranch %shl_84
        %shl_100 = OpLabel
        %shl_101 = OpLoad %uint %shl_20
        %shl_102 = OpULessThan %bool %shl_101 %uint_64
               OpSelectionMerge %shl_105 None
               OpBranchConditional %shl_102 %shl_104 %shl_110
        %shl_104 = OpLabel
        %shl_106 = OpLoad %uint %shl_18
        %shl_107 = OpLoad %uint %shl_20
        %shl_108 = OpISub %uint %shl_107 %uint_32
        %shl_109 = OpShiftLeftLogical %uint %shl_106 %shl_108
               OpStore %shl_103 %shl_109
               OpBranch %shl_105
        %shl_110 = OpLabel
               OpStore %shl_103 %uint_0
               OpBranch %shl_105
        %shl_105 = OpLabel
        %shl_335 = OpPhi %uint %shl_109 %shl_104 %uint_0 %shl_110
               OpStore %shl_82 %shl_335
               OpBranch %shl_84
         %shl_84 = OpLabel
        %shl_337 = OpPhi %uint %shl_99 %shl_89 %shl_335 %shl_105
               OpStore %shl_17 %shl_337
               OpReturn
               OpFunctionEnd
"#;

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

/// Beyond Kyty (`buffer_load_ubyte` is `KYTY_NI` upstream): single byte
/// load, zero-extended. The address model is the same BYTE address the
/// float1 helper computes (`offset + index * stride`) — it is NOT
/// pre-divided by 4: the containing dword is `byte_addr / 4` and the byte
/// within it is `byte_addr & 3`, extracted with `OpBitFieldUExtract` at bit
/// offset `(byte_addr & 3) * 8`, width 8. Measured on ASTRO.BOT scene
/// compute (raw 0xe02020c0, 58 dispatches/run).
pub(crate) const BUFFER_LOAD_UBYTE: &str = r#"
             ; void buffer_load_ubyte(out float p1, in int index, in int offset, in int stride, in int buffer_index)
             ; {
             ; 	int byte_addr = offset + index * stride;
             ; 	uint dw = floatBitsToUint(buf[buffer_index].data[byte_addr / 4]);
             ; 	p1 = uintBitsToFloat(bitfieldExtract(dw, (byte_addr & 3) * 8, 8));
             ; }
%buffer_load_ubyte = OpFunction %void None %function_buffer_load_store_float1
         %buf_l_ub_11 = OpFunctionParameter %_ptr_Function_float
         %buf_l_ub_12 = OpFunctionParameter %_ptr_Function_int
         %buf_l_ub_13 = OpFunctionParameter %_ptr_Function_int
         %buf_l_ub_14 = OpFunctionParameter %_ptr_Function_int
         %buf_l_ub_15 = OpFunctionParameter %_ptr_Function_int
         %buf_l_ub_17 = OpLabel
         %buf_l_ub_43 = OpLoad %int %buf_l_ub_13
         %buf_l_ub_44 = OpLoad %int %buf_l_ub_12
         %buf_l_ub_45 = OpLoad %int %buf_l_ub_14
         %buf_l_ub_46 = OpIMul %int %buf_l_ub_44 %buf_l_ub_45
         %buf_l_ub_47 = OpIAdd %int %buf_l_ub_43 %buf_l_ub_46
         %buf_l_ub_49 = OpSDiv %int %buf_l_ub_47 %int_4
         %buf_l_ub_50 = OpBitwiseAnd %int %buf_l_ub_47 %int_3
         %buf_l_ub_51 = OpIMul %int %buf_l_ub_50 %int_8
         %buf_l_ub_57 = OpLoad %int %buf_l_ub_15
         %buf_l_ub_62 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_l_ub_57 %int_0 %buf_l_ub_49
         %buf_l_ub_63 = OpLoad %float %buf_l_ub_62
         %buf_l_ub_64 = OpBitcast %uint %buf_l_ub_63
         %buf_l_ub_65 = OpBitFieldUExtract %uint %buf_l_ub_64 %buf_l_ub_51 %int_8
         %buf_l_ub_66 = OpBitcast %float %buf_l_ub_65
               OpStore %buf_l_ub_11 %buf_l_ub_66
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

/// Kyty: ShaderSpirv.cpp `BUFFER_STORE_FLOAT1` (L650).
pub(crate) const BUFFER_STORE_FLOAT1: &str = r#"
             ; void buffer_store_float1(in float p1, in int index, in int offset, in int stride, in int buffer_index)
             ; {
             ; 	int addr = (offset + index * stride)/4;
             ; 	buf[buffer_index].data[addr+0] = p1;
             ; }
%buffer_store_float1 = OpFunction %void None %function_buffer_load_store_float1
         %buf_s_f1_18 = OpFunctionParameter %_ptr_Function_float
         %buf_s_f1_19 = OpFunctionParameter %_ptr_Function_int 
         %buf_s_f1_20 = OpFunctionParameter %_ptr_Function_int 
         %buf_s_f1_21 = OpFunctionParameter %_ptr_Function_int 
         %buf_s_f1_22 = OpFunctionParameter %_ptr_Function_int 
         %buf_s_f1_24 = OpLabel
         %buf_s_f1_64 = OpVariable %_ptr_Function_int Function 
         %buf_s_f1_65 = OpLoad %int %buf_s_f1_20
         %buf_s_f1_66 = OpLoad %int %buf_s_f1_19
         %buf_s_f1_67 = OpLoad %int %buf_s_f1_21
         %buf_s_f1_68 = OpIMul %int %buf_s_f1_66 %buf_s_f1_67
         %buf_s_f1_69 = OpIAdd %int %buf_s_f1_65 %buf_s_f1_68
         %buf_s_f1_70 = OpSDiv %int %buf_s_f1_69 %int_4
               OpStore %buf_s_f1_64 %buf_s_f1_70
         %buf_s_f1_71 = OpLoad %int %buf_s_f1_22
         %buf_s_f1_74 = OpLoad %float %buf_s_f1_18
         %buf_s_f1_75 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_s_f1_71 %int_0 %buf_s_f1_70
               OpStore %buf_s_f1_75 %buf_s_f1_74
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `BUFFER_STORE_FLOAT2` (L679).
pub(crate) const BUFFER_STORE_FLOAT2: &str = r#"
                      ; void buffer_store_float2(in float p1, in float p2, in int index, in int offset, in int stride, in int buffer_index)
                      ; {
                      ; 	int addr = (offset + index * stride)/4;
                      ; 	buf[buffer_index].data[addr+0] = p1;
                      ; 	buf[buffer_index].data[addr+1] = p2;
                      ; }
%buffer_store_float2 = OpFunction %void None %function_buffer_load_store_float2
         %buf_s_f2_51 = OpFunctionParameter %_ptr_Function_float
         %buf_s_f2_52 = OpFunctionParameter %_ptr_Function_float
         %buf_s_f2_53 = OpFunctionParameter %_ptr_Function_int
         %buf_s_f2_54 = OpFunctionParameter %_ptr_Function_int
         %buf_s_f2_55 = OpFunctionParameter %_ptr_Function_int
         %buf_s_f2_56 = OpFunctionParameter %_ptr_Function_int
         %buf_s_f2_58 = OpLabel
        %buf_s_f2_143 = OpVariable %_ptr_Function_int Function
        %buf_s_f2_144 = OpLoad %int %buf_s_f2_54
        %buf_s_f2_145 = OpLoad %int %buf_s_f2_53
        %buf_s_f2_146 = OpLoad %int %buf_s_f2_55
        %buf_s_f2_147 = OpIMul %int %buf_s_f2_145 %buf_s_f2_146
        %buf_s_f2_148 = OpIAdd %int %buf_s_f2_144 %buf_s_f2_147
        %buf_s_f2_149 = OpSDiv %int %buf_s_f2_148 %int_4
               OpStore %buf_s_f2_143 %buf_s_f2_149
        %buf_s_f2_150 = OpLoad %int %buf_s_f2_56
        %buf_s_f2_153 = OpLoad %float %buf_s_f2_51
        %buf_s_f2_154 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_s_f2_150 %int_0 %buf_s_f2_149
               OpStore %buf_s_f2_154 %buf_s_f2_153
        %buf_s_f2_155 = OpLoad %int %buf_s_f2_56
        %buf_s_f2_158 = OpIAdd %int %buf_s_f2_149 %int_1
        %buf_s_f2_159 = OpLoad %float %buf_s_f2_52
        %buf_s_f2_160 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_s_f2_155 %int_0 %buf_s_f2_158
               OpStore %buf_s_f2_160 %buf_s_f2_159
               OpReturn
               OpFunctionEnd
"#;

/// Beyond Kyty: the 4-dword store twin of `BUFFER_LOAD_FLOAT4` (L598) —
/// upstream has no `buffer_store_float4` because `buffer_store_format_xyzw`
/// is `KYTY_NI`. Same signature (and thus the same
/// `%function_buffer_load_float4` type): addr = (offset + index * stride)/4,
/// then four consecutive dword stores.
pub(crate) const BUFFER_STORE_FLOAT4: &str = r#"
             ; void buffer_store_float4(in float p1, in float p2, in float p3, in float p4, in int index,
             ;                          in int offset, in int stride, in int buffer_index)
             ; {
             ; 	int addr = (offset + index * stride)/4;
             ; 	buf[buffer_index].data[addr+0] = p1;
             ; 	buf[buffer_index].data[addr+1] = p2;
             ; 	buf[buffer_index].data[addr+2] = p3;
             ; 	buf[buffer_index].data[addr+3] = p4;
             ; }
%buffer_store_float4 = OpFunction %void None %function_buffer_load_float4
  %buf_s_f4_21 = OpFunctionParameter %_ptr_Function_float
  %buf_s_f4_22 = OpFunctionParameter %_ptr_Function_float
  %buf_s_f4_23 = OpFunctionParameter %_ptr_Function_float
  %buf_s_f4_24 = OpFunctionParameter %_ptr_Function_float
  %buf_s_f4_25 = OpFunctionParameter %_ptr_Function_int
  %buf_s_f4_26 = OpFunctionParameter %_ptr_Function_int
  %buf_s_f4_27 = OpFunctionParameter %_ptr_Function_int
  %buf_s_f4_28 = OpFunctionParameter %_ptr_Function_int
  %buf_s_f4_30 = OpLabel
  %buf_s_f4_44 = OpVariable %_ptr_Function_int Function
  %buf_s_f4_45 = OpLoad %int %buf_s_f4_26
  %buf_s_f4_46 = OpLoad %int %buf_s_f4_25
  %buf_s_f4_47 = OpLoad %int %buf_s_f4_27
  %buf_s_f4_48 = OpIMul %int %buf_s_f4_46 %buf_s_f4_47
  %buf_s_f4_49 = OpIAdd %int %buf_s_f4_45 %buf_s_f4_48
  %buf_s_f4_51 = OpSDiv %int %buf_s_f4_49 %int_4
        OpStore %buf_s_f4_44 %buf_s_f4_51
  %buf_s_f4_58 = OpLoad %int %buf_s_f4_28
  %buf_s_f4_62 = OpLoad %float %buf_s_f4_21
  %buf_s_f4_63 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_s_f4_58 %int_0 %buf_s_f4_51
        OpStore %buf_s_f4_63 %buf_s_f4_62
  %buf_s_f4_65 = OpLoad %int %buf_s_f4_28
  %buf_s_f4_68 = OpIAdd %int %buf_s_f4_51 %int_1
  %buf_s_f4_69 = OpLoad %float %buf_s_f4_22
  %buf_s_f4_70 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_s_f4_65 %int_0 %buf_s_f4_68
        OpStore %buf_s_f4_70 %buf_s_f4_69
  %buf_s_f4_71 = OpLoad %int %buf_s_f4_28
  %buf_s_f4_74 = OpIAdd %int %buf_s_f4_51 %int_2
  %buf_s_f4_75 = OpLoad %float %buf_s_f4_23
  %buf_s_f4_76 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_s_f4_71 %int_0 %buf_s_f4_74
        OpStore %buf_s_f4_76 %buf_s_f4_75
  %buf_s_f4_77 = OpLoad %int %buf_s_f4_28
  %buf_s_f4_80 = OpIAdd %int %buf_s_f4_51 %int_3
  %buf_s_f4_81 = OpLoad %float %buf_s_f4_24
  %buf_s_f4_82 = OpAccessChain %_ptr_StorageBuffer_float %buf %buf_s_f4_77 %int_0 %buf_s_f4_80
        OpStore %buf_s_f4_82 %buf_s_f4_81
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

/// Kyty: ShaderSpirv.cpp `TBUFFER_STORE_FORMAT_X` (L817).
pub(crate) const TBUFFER_STORE_FORMAT_X: &str = r#"
             ; void tbuffer_store_format_x(in float p1, in int index, in int offset, in int stride, in int buffer_index, in int dfmt_nfmt)
             ; {
             ; 	if (dfmt_nfmt == 36 || dfmt_nfmt == 39) // dmft = 4, nfmt = 4 or 7
             ; 	{
             ; 		buffer_store_float1(p1, index, offset, stride, buffer_index);
             ; 	}
             ; }
%tbuffer_store_format_x = OpFunction %void None %function_tbuffer_load_store_format_x
         %tbuf_s_f_x_34 = OpFunctionParameter %_ptr_Function_float 
         %tbuf_s_f_x_35 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_x_36 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_x_37 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_x_38 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_x_39 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_x_41 = OpLabel
         %tbuf_s_f_x_97 = OpVariable %_ptr_Function_float Function
         %tbuf_s_f_x_99 = OpVariable %_ptr_Function_int Function
        %tbuf_s_f_x_101 = OpVariable %_ptr_Function_int Function
        %tbuf_s_f_x_103 = OpVariable %_ptr_Function_int Function
        %tbuf_s_f_x_105 = OpVariable %_ptr_Function_int Function
         %tbuf_s_f_x_93 = OpLoad %int %tbuf_s_f_x_39
         %tbuf_s_f_x_94 = OpIEqual %bool %tbuf_s_f_x_93 %int_36
         %tbuf_s_f_x_94_2 = OpIEqual %bool %tbuf_s_f_x_93 %int_39
         %tbuf_s_f_x_94_3 = OpLogicalOr %bool %tbuf_s_f_x_94 %tbuf_s_f_x_94_2
               OpSelectionMerge %tbuf_s_f_x_96 None
               OpBranchConditional %tbuf_s_f_x_94_3 %tbuf_s_f_x_95 %tbuf_s_f_x_96
         %tbuf_s_f_x_95 = OpLabel
         %tbuf_s_f_x_98 = OpLoad %float %tbuf_s_f_x_34
               OpStore %tbuf_s_f_x_97 %tbuf_s_f_x_98
        %tbuf_s_f_x_100 = OpLoad %int %tbuf_s_f_x_35
               OpStore %tbuf_s_f_x_99 %tbuf_s_f_x_100
        %tbuf_s_f_x_102 = OpLoad %int %tbuf_s_f_x_36
               OpStore %tbuf_s_f_x_101 %tbuf_s_f_x_102
        %tbuf_s_f_x_104 = OpLoad %int %tbuf_s_f_x_37
               OpStore %tbuf_s_f_x_103 %tbuf_s_f_x_104
        %tbuf_s_f_x_106 = OpLoad %int %tbuf_s_f_x_38
               OpStore %tbuf_s_f_x_105 %tbuf_s_f_x_106
        %tbuf_s_f_x_107 = OpFunctionCall %void %buffer_store_float1 %tbuf_s_f_x_97 %tbuf_s_f_x_99 %tbuf_s_f_x_101 %tbuf_s_f_x_103 %tbuf_s_f_x_105
               OpBranch %tbuf_s_f_x_96
         %tbuf_s_f_x_96 = OpLabel
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `TBUFFER_STORE_FORMAT_XY` (L862).
pub(crate) const TBUFFER_STORE_FORMAT_XY: &str = r#"
                        ; void tbuffer_store_format_xy(in float p1, in float p2, in int index, in int offset, in int stride, in int buffer_index, in int dfmt_nfmt)
                        ; {
                        ; 	if (dfmt_nfmt == 92 || dfmt_nfmt == 95) // dmft = 11, nfmt = 4 or 7
                        ; 	{
                        ; 		buffer_store_float2(p1, p2, index, offset, stride, buffer_index);
                        ; 	}
                        ; }
%tbuffer_store_format_xy = OpFunction %void None %function_tbuffer_load_store_format_xy
         %tbuf_s_f_xy_60 = OpFunctionParameter %_ptr_Function_float
         %tbuf_s_f_xy_61 = OpFunctionParameter %_ptr_Function_float
         %tbuf_s_f_xy_62 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_xy_63 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_xy_64 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_xy_65 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_xy_66 = OpFunctionParameter %_ptr_Function_int
         %tbuf_s_f_xy_68 = OpLabel
        %tbuf_s_f_xy_170 = OpVariable %_ptr_Function_float Function
        %tbuf_s_f_xy_172 = OpVariable %_ptr_Function_float Function
        %tbuf_s_f_xy_174 = OpVariable %_ptr_Function_int Function
        %tbuf_s_f_xy_176 = OpVariable %_ptr_Function_int Function
        %tbuf_s_f_xy_178 = OpVariable %_ptr_Function_int Function
        %tbuf_s_f_xy_180 = OpVariable %_ptr_Function_int Function
        %tbuf_s_f_xy_161 = OpLoad %int %tbuf_s_f_xy_66
        %tbuf_s_f_xy_163 = OpIEqual %bool %tbuf_s_f_xy_161 %int_92
        %tbuf_s_f_xy_164 = OpLoad %int %tbuf_s_f_xy_66
        %tbuf_s_f_xy_166 = OpIEqual %bool %tbuf_s_f_xy_164 %int_95
        %tbuf_s_f_xy_167 = OpLogicalOr %bool %tbuf_s_f_xy_163 %tbuf_s_f_xy_166
               OpSelectionMerge %tbuf_s_f_xy_169 None
               OpBranchConditional %tbuf_s_f_xy_167 %tbuf_s_f_xy_168 %tbuf_s_f_xy_169
        %tbuf_s_f_xy_168 = OpLabel
        %tbuf_s_f_xy_171 = OpLoad %float %tbuf_s_f_xy_60
               OpStore %tbuf_s_f_xy_170 %tbuf_s_f_xy_171
        %tbuf_s_f_xy_173 = OpLoad %float %tbuf_s_f_xy_61
               OpStore %tbuf_s_f_xy_172 %tbuf_s_f_xy_173
        %tbuf_s_f_xy_175 = OpLoad %int %tbuf_s_f_xy_62
               OpStore %tbuf_s_f_xy_174 %tbuf_s_f_xy_175
        %tbuf_s_f_xy_177 = OpLoad %int %tbuf_s_f_xy_63
               OpStore %tbuf_s_f_xy_176 %tbuf_s_f_xy_177
        %tbuf_s_f_xy_179 = OpLoad %int %tbuf_s_f_xy_64
               OpStore %tbuf_s_f_xy_178 %tbuf_s_f_xy_179
        %tbuf_s_f_xy_181 = OpLoad %int %tbuf_s_f_xy_65
               OpStore %tbuf_s_f_xy_180 %tbuf_s_f_xy_181
        %tbuf_s_f_xy_182 = OpFunctionCall %void %buffer_store_float2 %tbuf_s_f_xy_170 %tbuf_s_f_xy_172 %tbuf_s_f_xy_174 %tbuf_s_f_xy_176 %tbuf_s_f_xy_178 %tbuf_s_f_xy_180
               OpBranch %tbuf_s_f_xy_169
        %tbuf_s_f_xy_169 = OpLabel
               OpReturn
               OpFunctionEnd
"#;

/// Beyond Kyty: the store twin of `TBUFFER_LOAD_FORMAT_XYZW` (L715), for
/// `buffer_store_format_xyzw` (`KYTY_NI` upstream). dfmt_nfmt 119 = dfmt 14
/// (32_32_32_32), nfmt 7 (float) — the only combination that stores as four
/// raw dwords; every other format is left unwritten rather than corrupted.
/// Signature matches `%function_tbuffer_load_format_xyzw`.
pub(crate) const TBUFFER_STORE_FORMAT_XYZW: &str = r#"
             ; void tbuffer_store_format_xyzw(in float p1, in float p2, in float p3, in float p4,
             ;                                in int index, in int offset, in int stride, in int buffer_index, in int dfmt_nfmt)
             ; {
             ; 	if (dfmt_nfmt == 119) // dfmt = 14, nfmt = 7
             ; 	{
             ; 		buffer_store_float4(p1, p2, p3, p4, index, offset, stride, buffer_index);
             ; 	}
             ; }
%tbuffer_store_format_xyzw = OpFunction %void None %function_tbuffer_load_format_xyzw
%tbuf_s_f_xyzw_54 = OpFunctionParameter %_ptr_Function_float
%tbuf_s_f_xyzw_55 = OpFunctionParameter %_ptr_Function_float
%tbuf_s_f_xyzw_56 = OpFunctionParameter %_ptr_Function_float
%tbuf_s_f_xyzw_57 = OpFunctionParameter %_ptr_Function_float
%tbuf_s_f_xyzw_58 = OpFunctionParameter %_ptr_Function_int
%tbuf_s_f_xyzw_59 = OpFunctionParameter %_ptr_Function_int
%tbuf_s_f_xyzw_60 = OpFunctionParameter %_ptr_Function_int
%tbuf_s_f_xyzw_61 = OpFunctionParameter %_ptr_Function_int
%tbuf_s_f_xyzw_62 = OpFunctionParameter %_ptr_Function_int
%tbuf_s_f_xyzw_64 = OpLabel
%tbuf_s_f_xyzw_166 = OpVariable %_ptr_Function_float Function
%tbuf_s_f_xyzw_167 = OpVariable %_ptr_Function_float Function
%tbuf_s_f_xyzw_168 = OpVariable %_ptr_Function_float Function
%tbuf_s_f_xyzw_169 = OpVariable %_ptr_Function_float Function
%tbuf_s_f_xyzw_170 = OpVariable %_ptr_Function_int Function
%tbuf_s_f_xyzw_172 = OpVariable %_ptr_Function_int Function
%tbuf_s_f_xyzw_174 = OpVariable %_ptr_Function_int Function
%tbuf_s_f_xyzw_176 = OpVariable %_ptr_Function_int Function
%tbuf_s_f_xyzw_161 = OpLoad %int %tbuf_s_f_xyzw_62
%tbuf_s_f_xyzw_163 = OpIEqual %bool %tbuf_s_f_xyzw_161 %int_119
   OpSelectionMerge %tbuf_s_f_xyzw_165 None
   OpBranchConditional %tbuf_s_f_xyzw_163 %tbuf_s_f_xyzw_164 %tbuf_s_f_xyzw_165
%tbuf_s_f_xyzw_164 = OpLabel
%tbuf_s_f_xyzw_179 = OpLoad %float %tbuf_s_f_xyzw_54
   OpStore %tbuf_s_f_xyzw_166 %tbuf_s_f_xyzw_179
%tbuf_s_f_xyzw_180 = OpLoad %float %tbuf_s_f_xyzw_55
   OpStore %tbuf_s_f_xyzw_167 %tbuf_s_f_xyzw_180
%tbuf_s_f_xyzw_181 = OpLoad %float %tbuf_s_f_xyzw_56
   OpStore %tbuf_s_f_xyzw_168 %tbuf_s_f_xyzw_181
%tbuf_s_f_xyzw_182 = OpLoad %float %tbuf_s_f_xyzw_57
   OpStore %tbuf_s_f_xyzw_169 %tbuf_s_f_xyzw_182
%tbuf_s_f_xyzw_171 = OpLoad %int %tbuf_s_f_xyzw_58
   OpStore %tbuf_s_f_xyzw_170 %tbuf_s_f_xyzw_171
%tbuf_s_f_xyzw_173 = OpLoad %int %tbuf_s_f_xyzw_59
   OpStore %tbuf_s_f_xyzw_172 %tbuf_s_f_xyzw_173
%tbuf_s_f_xyzw_175 = OpLoad %int %tbuf_s_f_xyzw_60
   OpStore %tbuf_s_f_xyzw_174 %tbuf_s_f_xyzw_175
%tbuf_s_f_xyzw_177 = OpLoad %int %tbuf_s_f_xyzw_61
   OpStore %tbuf_s_f_xyzw_176 %tbuf_s_f_xyzw_177
%tbuf_s_f_xyzw_178 = OpFunctionCall %void %buffer_store_float4 %tbuf_s_f_xyzw_166 %tbuf_s_f_xyzw_167 %tbuf_s_f_xyzw_168 %tbuf_s_f_xyzw_169 %tbuf_s_f_xyzw_170 %tbuf_s_f_xyzw_172 %tbuf_s_f_xyzw_174 %tbuf_s_f_xyzw_176
   OpBranch %tbuf_s_f_xyzw_165
%tbuf_s_f_xyzw_165 = OpLabel
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

/// Kyty: ShaderSpirv.cpp `SBUFFER_LOAD_DWORD_8` (L1016).
pub(crate) const SBUFFER_LOAD_DWORD_8: &str = r#"
                     ; void sbuffer_load_dwordx8(out uint p1, out uint p2, out uint p3, out uint p4, 
                     ;                           out uint p5, out uint p6, out uint p7, out uint p8, in int offset, in int buffer_index)
                     ; {
                     ; 	int addr = offset/4;
                     ; 	p1 = floatBitsToUint(buf[buffer_index].data[addr+0]);
                     ; 	p2 = floatBitsToUint(buf[buffer_index].data[addr+1]);
                     ; 	p3 = floatBitsToUint(buf[buffer_index].data[addr+2]);
                     ; 	p4 = floatBitsToUint(buf[buffer_index].data[addr+3]);
                     ; 	p5 = floatBitsToUint(buf[buffer_index].data[addr+4]);
                     ; 	p6 = floatBitsToUint(buf[buffer_index].data[addr+5]);
                     ; 	p7 = floatBitsToUint(buf[buffer_index].data[addr+6]);
                     ; 	p8 = floatBitsToUint(buf[buffer_index].data[addr+7]);
                     ; }
%sbuffer_load_dword_8 = OpFunction %void None %function_sbuffer_load_dword_8
         %sbuf_dw8_60 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw8_61 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw8_62 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw8_63 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw8_64 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw8_65 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw8_66 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw8_67 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw8_68 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw8_69 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw8_71 = OpLabel
        %sbuf_dw8_197 = OpVariable %_ptr_Function_int Function
        %sbuf_dw8_198 = OpLoad %int %sbuf_dw8_68
        %sbuf_dw8_199 = OpSDiv %int %sbuf_dw8_198 %int_4
               OpStore %sbuf_dw8_197 %sbuf_dw8_199
        %sbuf_dw8_200 = OpLoad %int %sbuf_dw8_69
        %sbuf_dw8_203 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw8_200 %int_0 %sbuf_dw8_199
        %sbuf_dw8_204 = OpLoad %float %sbuf_dw8_203
        %sbuf_dw8_205 = OpBitcast %uint %sbuf_dw8_204
               OpStore %sbuf_dw8_60 %sbuf_dw8_205
        %sbuf_dw8_206 = OpLoad %int %sbuf_dw8_69
        %sbuf_dw8_208 = OpIAdd %int %sbuf_dw8_199 %int_1
        %sbuf_dw8_209 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw8_206 %int_0 %sbuf_dw8_208
        %sbuf_dw8_210 = OpLoad %float %sbuf_dw8_209
        %sbuf_dw8_211 = OpBitcast %uint %sbuf_dw8_210
               OpStore %sbuf_dw8_61 %sbuf_dw8_211
        %sbuf_dw8_212 = OpLoad %int %sbuf_dw8_69
        %sbuf_dw8_214 = OpIAdd %int %sbuf_dw8_199 %int_2
        %sbuf_dw8_215 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw8_212 %int_0 %sbuf_dw8_214
        %sbuf_dw8_216 = OpLoad %float %sbuf_dw8_215
        %sbuf_dw8_217 = OpBitcast %uint %sbuf_dw8_216
               OpStore %sbuf_dw8_62 %sbuf_dw8_217
        %sbuf_dw8_218 = OpLoad %int %sbuf_dw8_69
        %sbuf_dw8_220 = OpIAdd %int %sbuf_dw8_199 %int_3
        %sbuf_dw8_221 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw8_218 %int_0 %sbuf_dw8_220
        %sbuf_dw8_222 = OpLoad %float %sbuf_dw8_221
        %sbuf_dw8_223 = OpBitcast %uint %sbuf_dw8_222
               OpStore %sbuf_dw8_63 %sbuf_dw8_223
        %sbuf_dw8_224 = OpLoad %int %sbuf_dw8_69
        %sbuf_dw8_226 = OpIAdd %int %sbuf_dw8_199 %int_4
        %sbuf_dw8_227 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw8_224 %int_0 %sbuf_dw8_226
        %sbuf_dw8_228 = OpLoad %float %sbuf_dw8_227
        %sbuf_dw8_229 = OpBitcast %uint %sbuf_dw8_228
               OpStore %sbuf_dw8_64 %sbuf_dw8_229
        %sbuf_dw8_230 = OpLoad %int %sbuf_dw8_69
        %sbuf_dw8_233 = OpIAdd %int %sbuf_dw8_199 %int_5
        %sbuf_dw8_234 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw8_230 %int_0 %sbuf_dw8_233
        %sbuf_dw8_235 = OpLoad %float %sbuf_dw8_234
        %sbuf_dw8_236 = OpBitcast %uint %sbuf_dw8_235
               OpStore %sbuf_dw8_65 %sbuf_dw8_236
        %sbuf_dw8_237 = OpLoad %int %sbuf_dw8_69
        %sbuf_dw8_240 = OpIAdd %int %sbuf_dw8_199 %int_6
        %sbuf_dw8_241 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw8_237 %int_0 %sbuf_dw8_240
        %sbuf_dw8_242 = OpLoad %float %sbuf_dw8_241
        %sbuf_dw8_243 = OpBitcast %uint %sbuf_dw8_242
               OpStore %sbuf_dw8_66 %sbuf_dw8_243
        %sbuf_dw8_244 = OpLoad %int %sbuf_dw8_69
        %sbuf_dw8_247 = OpIAdd %int %sbuf_dw8_199 %int_7
        %sbuf_dw8_248 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw8_244 %int_0 %sbuf_dw8_247
        %sbuf_dw8_249 = OpLoad %float %sbuf_dw8_248
        %sbuf_dw8_250 = OpBitcast %uint %sbuf_dw8_249
               OpStore %sbuf_dw8_67 %sbuf_dw8_250
               OpReturn
               OpFunctionEnd
"#;

/// Kyty: ShaderSpirv.cpp `SBUFFER_LOAD_DWORD_16` (L1097).
pub(crate) const SBUFFER_LOAD_DWORD_16: &str = r#"
                     ; void sbuffer_load_dwordx16(out uint p1, out uint p2, out uint p3, out uint p4, 
                     ;                            out uint p5, out uint p6, out uint p7, out uint p8,
                     ;                            out uint p9, out uint p10, out uint p11, out uint p12,
                     ;                            out uint p13, out uint p14, out uint p15, out uint p16, in int offset, in int buffer_index)
                     ; {
                     ; 	int addr = offset/4;
                     ; 	p1 = floatBitsToUint(buf[buffer_index].data[addr+0]);
                     ; 	p2 = floatBitsToUint(buf[buffer_index].data[addr+1]);
                     ; 	p3 = floatBitsToUint(buf[buffer_index].data[addr+2]);
                     ; 	p4 = floatBitsToUint(buf[buffer_index].data[addr+3]);
                     ; 	p5 = floatBitsToUint(buf[buffer_index].data[addr+4]);
                     ; 	p6 = floatBitsToUint(buf[buffer_index].data[addr+5]);
                     ; 	p7 = floatBitsToUint(buf[buffer_index].data[addr+6]);
                     ; 	p8 = floatBitsToUint(buf[buffer_index].data[addr+7]);
                     ; 	p9 = floatBitsToUint(buf[buffer_index].data[addr+8]);
                     ; 	p10 = floatBitsToUint(buf[buffer_index].data[addr+9]);
                     ; 	p11 = floatBitsToUint(buf[buffer_index].data[addr+10]);
                     ; 	p12 = floatBitsToUint(buf[buffer_index].data[addr+11]);
                     ; 	p13 = floatBitsToUint(buf[buffer_index].data[addr+12]);
                     ; 	p14 = floatBitsToUint(buf[buffer_index].data[addr+13]);
                     ; 	p15 = floatBitsToUint(buf[buffer_index].data[addr+14]);
                     ; 	p16 = floatBitsToUint(buf[buffer_index].data[addr+15]);
                     ; }
%sbuffer_load_dword_16 = OpFunction %void None %function_sbuffer_load_dword_16
         %sbuf_dw16_60 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_61 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_62 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_63 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_64 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_65 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_66 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_67 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_68 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_69 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_70 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_71 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_72 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_73 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_74 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_75 = OpFunctionParameter %_ptr_Function_uint
         %sbuf_dw16_76 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw16_77 = OpFunctionParameter %_ptr_Function_int
         %sbuf_dw16_79 = OpLabel
        %sbuf_dw16_184 = OpVariable %_ptr_Function_int Function
        %sbuf_dw16_185 = OpLoad %int %sbuf_dw16_76
        %sbuf_dw16_186 = OpSDiv %int %sbuf_dw16_185 %int_4
               OpStore %sbuf_dw16_184 %sbuf_dw16_186
        %sbuf_dw16_187 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_190 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_187 %int_0 %sbuf_dw16_186
        %sbuf_dw16_191 = OpLoad %float %sbuf_dw16_190
        %sbuf_dw16_192 = OpBitcast %uint %sbuf_dw16_191
               OpStore %sbuf_dw16_60 %sbuf_dw16_192
        %sbuf_dw16_193 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_195 = OpIAdd %int %sbuf_dw16_186 %int_1
        %sbuf_dw16_196 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_193 %int_0 %sbuf_dw16_195
        %sbuf_dw16_197 = OpLoad %float %sbuf_dw16_196
        %sbuf_dw16_198 = OpBitcast %uint %sbuf_dw16_197
               OpStore %sbuf_dw16_61 %sbuf_dw16_198
        %sbuf_dw16_199 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_201 = OpIAdd %int %sbuf_dw16_186 %int_2
        %sbuf_dw16_202 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_199 %int_0 %sbuf_dw16_201
        %sbuf_dw16_203 = OpLoad %float %sbuf_dw16_202
        %sbuf_dw16_204 = OpBitcast %uint %sbuf_dw16_203
               OpStore %sbuf_dw16_62 %sbuf_dw16_204
        %sbuf_dw16_205 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_207 = OpIAdd %int %sbuf_dw16_186 %int_3
        %sbuf_dw16_208 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_205 %int_0 %sbuf_dw16_207
        %sbuf_dw16_209 = OpLoad %float %sbuf_dw16_208
        %sbuf_dw16_210 = OpBitcast %uint %sbuf_dw16_209
               OpStore %sbuf_dw16_63 %sbuf_dw16_210
        %sbuf_dw16_211 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_213 = OpIAdd %int %sbuf_dw16_186 %int_4
        %sbuf_dw16_214 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_211 %int_0 %sbuf_dw16_213
        %sbuf_dw16_215 = OpLoad %float %sbuf_dw16_214
        %sbuf_dw16_216 = OpBitcast %uint %sbuf_dw16_215
               OpStore %sbuf_dw16_64 %sbuf_dw16_216
        %sbuf_dw16_217 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_220 = OpIAdd %int %sbuf_dw16_186 %int_5
        %sbuf_dw16_221 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_217 %int_0 %sbuf_dw16_220
        %sbuf_dw16_222 = OpLoad %float %sbuf_dw16_221
        %sbuf_dw16_223 = OpBitcast %uint %sbuf_dw16_222
               OpStore %sbuf_dw16_65 %sbuf_dw16_223
        %sbuf_dw16_224 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_227 = OpIAdd %int %sbuf_dw16_186 %int_6
        %sbuf_dw16_228 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_224 %int_0 %sbuf_dw16_227
        %sbuf_dw16_229 = OpLoad %float %sbuf_dw16_228
        %sbuf_dw16_230 = OpBitcast %uint %sbuf_dw16_229
               OpStore %sbuf_dw16_66 %sbuf_dw16_230
        %sbuf_dw16_231 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_234 = OpIAdd %int %sbuf_dw16_186 %int_7
        %sbuf_dw16_235 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_231 %int_0 %sbuf_dw16_234
        %sbuf_dw16_236 = OpLoad %float %sbuf_dw16_235
        %sbuf_dw16_237 = OpBitcast %uint %sbuf_dw16_236
               OpStore %sbuf_dw16_67 %sbuf_dw16_237
        %sbuf_dw16_238 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_241 = OpIAdd %int %sbuf_dw16_186 %int_8
        %sbuf_dw16_242 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_238 %int_0 %sbuf_dw16_241
        %sbuf_dw16_243 = OpLoad %float %sbuf_dw16_242
        %sbuf_dw16_244 = OpBitcast %uint %sbuf_dw16_243
               OpStore %sbuf_dw16_68 %sbuf_dw16_244
        %sbuf_dw16_245 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_248 = OpIAdd %int %sbuf_dw16_186 %int_9
        %sbuf_dw16_249 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_245 %int_0 %sbuf_dw16_248
        %sbuf_dw16_250 = OpLoad %float %sbuf_dw16_249
        %sbuf_dw16_251 = OpBitcast %uint %sbuf_dw16_250
               OpStore %sbuf_dw16_69 %sbuf_dw16_251
        %sbuf_dw16_252 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_255 = OpIAdd %int %sbuf_dw16_186 %int_10
        %sbuf_dw16_256 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_252 %int_0 %sbuf_dw16_255
        %sbuf_dw16_257 = OpLoad %float %sbuf_dw16_256
        %sbuf_dw16_258 = OpBitcast %uint %sbuf_dw16_257
               OpStore %sbuf_dw16_70 %sbuf_dw16_258
        %sbuf_dw16_259 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_262 = OpIAdd %int %sbuf_dw16_186 %int_11
        %sbuf_dw16_263 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_259 %int_0 %sbuf_dw16_262
        %sbuf_dw16_264 = OpLoad %float %sbuf_dw16_263
        %sbuf_dw16_265 = OpBitcast %uint %sbuf_dw16_264
               OpStore %sbuf_dw16_71 %sbuf_dw16_265
        %sbuf_dw16_266 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_269 = OpIAdd %int %sbuf_dw16_186 %int_12
        %sbuf_dw16_270 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_266 %int_0 %sbuf_dw16_269
        %sbuf_dw16_271 = OpLoad %float %sbuf_dw16_270
        %sbuf_dw16_272 = OpBitcast %uint %sbuf_dw16_271
               OpStore %sbuf_dw16_72 %sbuf_dw16_272
        %sbuf_dw16_273 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_276 = OpIAdd %int %sbuf_dw16_186 %int_13
        %sbuf_dw16_277 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_273 %int_0 %sbuf_dw16_276
        %sbuf_dw16_278 = OpLoad %float %sbuf_dw16_277
        %sbuf_dw16_279 = OpBitcast %uint %sbuf_dw16_278
               OpStore %sbuf_dw16_73 %sbuf_dw16_279
        %sbuf_dw16_280 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_283 = OpIAdd %int %sbuf_dw16_186 %int_14
        %sbuf_dw16_284 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_280 %int_0 %sbuf_dw16_283
        %sbuf_dw16_285 = OpLoad %float %sbuf_dw16_284
        %sbuf_dw16_286 = OpBitcast %uint %sbuf_dw16_285
               OpStore %sbuf_dw16_74 %sbuf_dw16_286
        %sbuf_dw16_287 = OpLoad %int %sbuf_dw16_77
        %sbuf_dw16_290 = OpIAdd %int %sbuf_dw16_186 %int_15
        %sbuf_dw16_291 = OpAccessChain %_ptr_StorageBuffer_float %buf %sbuf_dw16_287 %int_0 %sbuf_dw16_290
        %sbuf_dw16_292 = OpLoad %float %sbuf_dw16_291
        %sbuf_dw16_293 = OpBitcast %uint %sbuf_dw16_292
               OpStore %sbuf_dw16_75 %sbuf_dw16_293
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

/// SDWA lane extraction (see `ShaderOperand::lane_sel`): the SPIR-V snippet
/// turning the raw 32-bit register value `%<src>` into the zero-extended
/// selected lane `%<dst>` — a logical shift right by the lane's bit offset
/// and a mask to its width. Returns `None` for 6 (DWORD: the whole register,
/// nothing to do). The shift (`%uint_0/8/16/24`) and mask
/// (`%uint_255`/`%uint_0x0000ffff`) constants are guaranteed by
/// `find_constants` whenever any operand carries a lane select.
fn lane_sel_snippet(lane_sel: u8, src: &str, dst: &str) -> Option<String> {
    let (shift, mask) = match lane_sel {
        0..=3 => (u32::from(lane_sel) * 8, "uint_255"),
        4 | 5 => (u32::from(lane_sel - 4) * 16, "uint_0x0000ffff"),
        _ => return None,
    };
    Some(format!(
        "%s{dst} = OpShiftRightLogical %uint %{src} %uint_{shift}\n          \
         %{dst} = OpBitwiseAnd %uint %s{dst} %{mask}\n"
    ))
}

/// The dword offset of an extended (EUD-resident) sharp inside the EUD —
/// the index the shader's `s_load_dwordx*` byte-offset literal (`>> 2`)
/// addresses, and therefore the `extended_mapping` slot `GetMappedIndex`
/// resolves at recompile time.
///
/// Kyty hardcodes `start_register - 16` (the PS4 user-SGPR count); this
/// mirrors `read_sharp_fields`' measured Gen5 residence rules instead:
///
/// * `start >= SGPRS_MAX` (32): the EUD is addressed as a continuation of
///   the register FILE — rebase by the file size.
/// * otherwise the sharp sits at `start - eud_base`, where `eud_base` is
///   the pointer-pair register the analysis recorded in
///   `bind.extended.start_register` (measured on ASTRO.BOT compute: T# at
///   start_register=12 with the EUD pair at (s12, s13) → EUD dword 0).
fn eud_rel_index(
    bind: &ShaderBindResources,
    start_reg: i32,
    shift_regs: i32,
    what: &str,
) -> Result<i32, ShaderRecompileError> {
    if shift_regs != 0 {
        return Err(not_supported(
            "Spirv::WriteLocalVariables",
            format!("extended {what} mapping with gs_prolog register shift"),
        ));
    }
    let sgprs_max = UserSgprInfo::SGPRS_MAX as i32;
    if start_reg >= sgprs_max {
        return Ok(start_reg - sgprs_max);
    }
    if !bind.extended.used || start_reg < bind.extended.start_register {
        return Err(not_supported(
            "Spirv::WriteLocalVariables",
            format!(
                "extended {what} at s{start_reg} has no EUD base to rebase on \
                 (extended.used={}, eud_base={})",
                bind.extended.used, bind.extended.start_register
            ),
        ));
    }
    Ok(start_reg - bind.extended.start_register)
}

/// The single `OpTypeImage` Dim of the sampled-texture array
/// (`%textures2D_S`), decided from the measured T# types: 9 (and the
/// height-1 "1D" 8) = 2D, 10 = 3D volume, 11 = Cube, 13 = 2DArray
/// (Dim 2D with `arrayed = 1` — ASTRO.BOT's 1536x1536x3 array). Storage
/// (read-write) descriptors are excluded — `%textures2D_L` is its own 2D
/// array.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SampledDim {
    Two,
    TwoArray,
    Three,
    Cube,
}

impl SampledDim {
    /// SPIR-V `Dim` token and the sample-coordinate component count.
    pub(crate) const fn dim_str(self) -> &'static str {
        match self {
            Self::Two | Self::TwoArray => "2D",
            Self::Three => "3D",
            Self::Cube => "Cube",
        }
    }

    /// The `arrayed` field of `OpTypeImage`.
    pub(crate) const fn arrayed_str(self) -> &'static str {
        match self {
            Self::TwoArray => "1",
            Self::Two | Self::Three | Self::Cube => "0",
        }
    }

    pub(crate) const fn coord_components(self) -> u32 {
        match self {
            Self::Two => 2,
            // An arrayed 2D sample carries the layer as the third coordinate
            // component (SPIR-V: coordinate includes the array layer last).
            Self::TwoArray | Self::Three | Self::Cube => 3,
        }
    }
}

/// Decide the sampled-texture Dim for a bind set, refusing mixed dims — a
/// single SPIR-V array type has exactly one `Dim`, so a shader mixing 2D and
/// 3D/cube sampled textures stays a named refusal until a measured shader
/// requires split arrays.
pub(crate) fn sampled_texture_dim(
    bind: &ShaderBindResources,
) -> Result<SampledDim, ShaderRecompileError> {
    let bound = usize::try_from(bind.textures2d.textures_num)
        .unwrap_or(0)
        .min(bind.textures2d.desc.len());
    let (mut has_2d, mut has_3d, mut has_cube, mut has_2darray) = (false, false, false, false);
    for d in &bind.textures2d.desc[..bound] {
        if d.textures2d_without_sampler {
            continue;
        }
        match d.texture.type_() {
            10 => has_3d = true,
            11 => has_cube = true,
            13 => has_2darray = true,
            _ => has_2d = true,
        }
    }
    match (has_2d, has_3d, has_cube, has_2darray) {
        (_, false, false, false) => Ok(SampledDim::Two),
        (false, true, false, false) => Ok(SampledDim::Three),
        (false, false, true, false) => Ok(SampledDim::Cube),
        (false, false, false, true) => Ok(SampledDim::TwoArray),
        _ => Err(not_supported(
            "sampled_texture_dim",
            format!(
                "mixed sampled texture dims in one shader \
                 (2d={has_2d} 3d={has_3d} cube={has_cube} 2darray={has_2darray})"
            ),
        )),
    }
}

/// The single `OpTypeImage` (Dim, storage format) of the STORAGE
/// (read-write) image array (`%textures2D_L`), decided from the measured RW
/// T#s: type 9 = 2D, type 10 = 3D (ASTRO.BOT: 240x135x64 UAV volumes);
/// guest format 71 (16_16_16_16 FLOAT) = `Rgba16f`, everything else keeps
/// the legacy `Rgba8` view (the 32-bpp guest formats the upload path reads,
/// or the zero-filled seed). Mixed dims/formats stay a named refusal — one
/// SPIR-V array type carries exactly one `OpTypeImage`.
pub(crate) fn storage_texture_dim_format(
    bind: &ShaderBindResources,
) -> Result<(SampledDim, &'static str), ShaderRecompileError> {
    let bound = usize::try_from(bind.textures2d.textures_num)
        .unwrap_or(0)
        .min(bind.textures2d.desc.len());
    let mut chosen: Option<(SampledDim, &'static str)> = None;
    for d in &bind.textures2d.desc[..bound] {
        if !d.textures2d_without_sampler {
            continue; // sampled — lives in %textures2D_S
        }
        let dim = match d.texture.type_() {
            10 => SampledDim::Three,
            _ => SampledDim::Two,
        };
        let format = if d.texture.format() == 71 {
            "Rgba16f"
        } else {
            "Rgba8"
        };
        match chosen {
            None => chosen = Some((dim, format)),
            Some(prev) if prev == (dim, format) => {}
            Some(prev) => {
                return Err(not_supported(
                    "storage_texture_dim_format",
                    format!(
                        "mixed storage image dims/formats in one shader \
                         ({prev:?} vs ({dim:?}, {format}))"
                    ),
                ));
            }
        }
    }
    Ok(chosen.unwrap_or((SampledDim::Two, "Rgba8")))
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
    if op.lane_sel != 6 {
        // No consumer yet (see the fn doc); wire the uint-style extraction
        // when one appears.
        return Err(not_supported("operand_load_int", "sdwa lane select"));
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
    if op.lane_sel != 6 && operand_is_constant(op) {
        return Err(not_supported(
            "operand_load_uint",
            "sdwa lane select on a constant operand",
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
        // SDWA lane select: extract the selected byte/word (zero-extended)
        // before the consumer sees the value. The non-SDWA path keeps Kyty's
        // exact load text.
        let l = if let Some(sel) = lane_sel_snippet(op.lane_sel, "r<result_id>", "<result_id>") {
            let raw = if value.type_ == SpirvType::Float {
                concat!(
                    "%t<result_id> = OpLoad %float %<id>\n",
                    "          ",
                    "%r<result_id> = OpBitcast %uint %t<result_id>\n"
                )
            } else if value.type_ == SpirvType::Uint {
                "%r<result_id> = OpLoad %uint %<id>\n"
            } else {
                return Ok(false);
            };
            format!("{raw}          {sel}")
        } else if value.type_ == SpirvType::Float {
            concat!(
                "%t<result_id> = OpLoad %float %<id>\n",
                "          ",
                "%<result_id> = OpBitcast %uint %t<result_id>\n"
            )
            .to_string()
        } else if value.type_ == SpirvType::Uint {
            "%<result_id> = OpLoad %uint %<id>".to_string()
        } else {
            return Ok(false);
        };
        *load = l
            .replace("<index>", index)
            .replace("<id>", &value.value)
            .replace("<result_id>", result_id);
    } else {
        return Ok(false);
    }
    Ok(true)
}

/// Kyty: ShaderSpirv.cpp `operand_load_float` (L1791).
/// Highest exp param index the shader body writes, from the Param0..4 format
/// of its Exp instructions. The register-derived `export_count` can under-read
/// the body (measured: a menu VS writes `%param1` while `spi_vs_out_config`
/// says 1 export) — the declarations must cover the body's ground truth or
/// the assembler dies with "id %paramN is used but never defined".
fn max_exp_param(code: &ShaderCode) -> i32 {
    use super::shader_instruction_format::Format as F;
    code.get_instructions()
        .iter()
        .filter(|inst| inst.type_ == ShaderInstructionType::Exp)
        .map(|inst| match inst.format {
            F::Param0Vsrc0Vsrc1Vsrc2Vsrc3 => 0,
            F::Param1Vsrc0Vsrc1Vsrc2Vsrc3 => 1,
            F::Param2Vsrc0Vsrc1Vsrc2Vsrc3 => 2,
            F::Param3Vsrc0Vsrc1Vsrc2Vsrc3 => 3,
            F::Param4Vsrc0Vsrc1Vsrc2Vsrc3 => 4,
            _ => -1,
        })
        .max()
        .unwrap_or(-1)
}

pub(crate) fn operand_load_float(
    spirv: &Spirv<'_>,
    op: ShaderOperand,
    result_id: &str,
    index: &str,
    load: &mut String,
) -> Result<bool, ShaderRecompileError> {
    let mut l: String;

    if op.lane_sel != 6 && operand_is_constant(op) {
        return Err(not_supported(
            "operand_load_float",
            "sdwa lane select on a constant operand",
        ));
    }

    if operand_is_constant(op) {
        let id = spirv.get_constant(op);
        l = "%<result_id> = OpBitcast %float %<id>".replace("<id>", &id);
    } else if operand_is_variable(op) {
        let value = operand_variable_to_str(op);
        // SDWA lane select: the hardware extracts the selected byte/word of
        // the raw register (zero-extended to 32 bits) and the operation then
        // consumes that dword as its operand type — so extract in uint space
        // and bitcast. The non-SDWA path keeps Kyty's exact load text.
        if let Some(sel) = lane_sel_snippet(op.lane_sel, "r<result_id>", "e<result_id>") {
            let raw = if value.type_ == SpirvType::Float {
                concat!(
                    "%f<result_id> = OpLoad %float %<id>\n",
                    "          ",
                    "%r<result_id> = OpBitcast %uint %f<result_id>\n"
                )
            } else if value.type_ == SpirvType::Uint {
                "%r<result_id> = OpLoad %uint %<id>\n"
            } else {
                return Ok(false);
            };
            l = format!(
                "{raw}          {sel}          \
                 %<result_id> = OpBitcast %float %e<result_id>\n"
            )
            .replace("<id>", &value.value);
        } else if value.type_ == SpirvType::Float {
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
            extended_mapping: [[-1; 2]; 64],
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

    /// Kyty: `Spirv::GetMappedIndex` (L1495). Deviation: an unfilled slot
    /// (the -1 sentinel `WriteLocalVariables` seeds) is a named refusal —
    /// Kyty's zero default would silently route the load to push constant
    /// (0, 0), i.e. another resource's descriptor.
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
        if m[0] < 0 || m[1] < 0 {
            return Err(not_supported(
                "Spirv::GetMappedIndex",
                format!("EUD dword {offset} is not a captured descriptor field"),
            ));
        }
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

        if self.uses_lds() {
            vars.push("%lds".to_string());
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
                    // See WriteGlobalVariables: the body's exp formats, not
                    // just the register count, decide the declared set.
                    let export_count = info.export_count.max(max_exp_param(&self.code) + 1);
                    for i in 0..export_count {
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
    ///
    /// KYTY-QUIRK: upstream appends each `OpString` without a trailing
    /// newline (ShaderSpirv.cpp L6808), so a second printf string would land
    /// on the same source line and fail to assemble. Ported verbatim; a
    /// single printf works because the next section starts with `\n`.
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
                    // See WriteGlobalVariables: the body's exp formats, not
                    // just the register count, decide the declared set.
                    let export_count = info.export_count.max(max_exp_param(&self.code) + 1);
                    for i in 0..export_count {
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
                                             %ImageS = OpTypeImage %float <dim> 0 <arrayed> 0 1 Unknown
                    %textures2D_S_uint_<buffers_num> = OpConstant %uint <buffers_num>
                     %_arr_ImageS_uint_<buffers_num> = OpTypeArray %ImageS %textures2D_S_uint_<buffers_num>
%_ptr_UniformConstant__arr_ImageS_uint_<buffers_num> = OpTypePointer UniformConstant %_arr_ImageS_uint_<buffers_num>
                        %_ptr_UniformConstant_ImageS = OpTypePointer UniformConstant %ImageS
                                       %SampledImage = OpTypeSampledImage %ImageS
"#;

        // Dim and storage format parametric (round 7 — 3D Rgba16f UAVs
        // measured on ASTRO.BOT); see `storage_texture_dim_format`.
        const TEXTURES_LOADED_TYPES: &str = r#"
                                             %ImageL = OpTypeImage %float <dim> 0 <arrayed> 0 2 <format>
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
                // The OpTypeImage Dim comes from the measured T# types: 9 =
                // 2D, 10 = 3D (ASTRO.BOT's froxel/LUT volumes), 11 = Cube
                // (Minecraft's skybox), 13 = 2DArray (`arrayed = 1`;
                // ASTRO.BOT's 1536x1536x3 array). A mixed binding set has no
                // single array type — `sampled_texture_dim` refuses it by
                // name.
                let dim = sampled_texture_dim(bind)?;
                self.source += &TEXTURES_SAMPLED_TYPES
                    .replace(
                        "<buffers_num>",
                        &format!("{}", bind.textures2d.textures2d_sampled_num),
                    )
                    .replace("<dim>", dim.dim_str())
                    .replace("<arrayed>", dim.arrayed_str());
            }
            if bind.textures2d.textures2d_storage_num > 0 {
                let (dim, format) = storage_texture_dim_format(bind)?;
                self.source += &TEXTURES_LOADED_TYPES
                    .replace(
                        "<buffers_num>",
                        &format!("{}", bind.textures2d.textures2d_storage_num),
                    )
                    .replace("<dim>", dim.dim_str())
                    .replace("<arrayed>", dim.arrayed_str())
                    .replace("<format>", format);
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

        // Beyond Kyty: LDS (workgroup-shared memory) backing for the
        // `ds_write_b32`/`ds_read_b32` pair — a fixed uint array in the
        // Workgroup storage class, sized from COMPUTE_PGM_RSRC2.LDS_SIZE
        // (64 KiB fallback; see `lds_size_dw`).
        const LDS_TYPES: &str = r#"
          %lds_num_uint_<lds_dw> = OpConstant %uint <lds_dw>
              %_arr_uint_lds_len = OpTypeArray %uint %lds_num_uint_<lds_dw>
    %_ptr_Workgroup__arr_uint_lds = OpTypePointer Workgroup %_arr_uint_lds_len
             %_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
"#;
        if self.uses_lds() {
            self.source += &LDS_TYPES.replace("<lds_dw>", &format!("{}", self.lds_size_dw()));
        }

        Ok(())
    }

    /// Whether the shader touches LDS through the implemented DS opcodes.
    fn uses_lds(&self) -> bool {
        use ShaderInstructionType as T;
        self.code.has_any_of(&[
            T::DsAddU32,
            T::DsReadB32,
            T::DsWriteB32,
            T::DsRead2B32,
            T::DsReadB64,
            T::DsReadB96,
            T::DsReadB128,
            T::DsWriteB96,
            T::DsWriteB128,
        ])
    }

    /// LDS allocation in dwords. `COMPUTE_PGM_RSRC2.LDS_SIZE` counts 128-dword
    /// granules (GFX10); a shader that uses DS ops with a zero (or missing)
    /// register still needs backing, so fall back to the full 64 KiB LDS.
    pub(crate) fn lds_size_dw(&self) -> u32 {
        let dw = self
            .cs_input_info
            .map_or(0, |i| i.lds_size_dw)
            .min(16 * 1024);
        if dw == 0 { 16 * 1024 } else { dw }
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

        if self.uses_lds() {
            vars.push("%lds = OpVariable %_ptr_Workgroup__arr_uint_lds Workgroup".to_string());
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
                    // The register-derived count can under-read the body's
                    // real exports (measured: menu VS writes param1 with a
                    // register count of 1). The exp formats in the body are
                    // the ground truth — declaring a dead export is legal,
                    // leaving a written one undeclared is not.
                    let export_count = info.export_count.max(max_exp_param(&self.code) + 1);
                    for i in 0..export_count {
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

            // Deviation from Kyty (which resets to 0): unfilled slots keep
            // the -1 sentinel so `GetMappedIndex` refuses an EUD dword no
            // captured descriptor covers instead of silently reading push
            // constant (0, 0).
            for m in &mut self.extended_mapping {
                m[0] = -1;
                m[1] = -1;
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
                        let rel = eud_rel_index(bind, start_reg, shift_regs, "storage buffer")?;
                        let idx = (rel + f) as usize;
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
                            let rel = eud_rel_index(bind, start_reg, shift_regs, "texture")?;
                            let idx = (rel + 4 * ti + f) as usize;
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
                        let rel = eud_rel_index(bind, start_reg, shift_regs, "sampler")?;
                        let idx = (rel + f) as usize;
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
                    let rel = eud_rel_index(bind, start_reg, shift_regs, "gds pointer")?;
                    let idx = rel as usize;
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
        use super::recompile::{RecompileFn, SccCheck, recomp_func, recompile_inject_debug};

        self.modify_code();

        // Kyty: `need_debug` (ShaderSpirv.cpp L7803).
        let need_debug = self.debug_printf_enabled && !self.code.get_debug_printfs().is_empty();

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

            // Kyty: `Recompile_Inject_Debug` injection point (L7834).
            if need_debug {
                let mut dst_debug = String::new();
                if recompile_inject_debug(
                    index as u32,
                    &self.code,
                    &mut dst_debug,
                    self,
                    &[None; 4],
                    SccCheck::None,
                )? {
                    self.source += &format!("{dst_debug}\n");
                }
            }
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
    fn write_functions(&mut self) {
        use ShaderInstructionType as T;

        if self.code.has_any_of(&[T::VSadU32]) {
            self.source += FUNC_ABS_DIFF;
        }

        if self.code.has_any_of(&[T::SWqmB64]) {
            self.source += FUNC_WQM;
        }

        if self.code.has_any_of(&[T::SAddcU32]) {
            self.source += FUNC_ADDC;
        }

        if self.code.has_any_of(&[T::SLshl4AddU32]) {
            self.source += FUNC_LSHL_ADD;
        }

        if self.code.has_any_of(&[T::ImageStoreMip]) {
            self.source += FUNC_MIPMAP;
        }

        if self.code.has_any_of(&[T::VCmpOF32, T::VCmpUF32]) {
            self.source += FUNC_ORDERED;
        }

        if self.code.has_any_of(&[
            T::VMulLoI32,
            T::VMulLoU32,
            T::VMulHiU32,
            T::VMadU32U24,
            T::VMulU32U24,
            T::SMulHiU32,
        ]) {
            self.source += FUNC_MUL_EXTENDED;
        }

        if self.code.has_any_of(&[T::SLshrB64, T::SBfeU64]) {
            self.source += FUNC_SHIFT_RIGHT;
        }

        if self.code.has_any_of(&[T::SLshlB64, T::SBfeU64]) {
            self.source += FUNC_SHIFT_LEFT;
        }

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

        // The buffer helper functions all index the %buf descriptor array,
        // which only exists when at least one storage buffer is bound. A
        // shader whose only MUBUF ops go through a NULL V# (recompiled as
        // dropped stores / zero loads — see `mubuf_flexible`) must not emit
        // them, or assembly fails on the undefined %buf id.
        let has_buffers = self
            .bind
            .is_some_and(|bind| bind.storage_buffers.buffers_num > 0);

        if has_buffers
            && self.code.has_any_of(&[
                T::BufferLoadDword,
                T::BufferLoadDwordX2,
                T::BufferLoadDwordX3,
                T::BufferLoadDwordX4,
                T::BufferLoadFormatX,
                T::BufferLoadFormatXy,
                T::BufferLoadFormatXyz,
                T::BufferLoadFormatXyzw,
                T::TBufferLoadFormatX,
                T::TBufferLoadFormatXyzw,
            ])
        {
            self.source += BUFFER_LOAD_FLOAT1;
            self.source += BUFFER_LOAD_FLOAT4;
            self.source += TBUFFER_LOAD_FORMAT_X;
            self.source += TBUFFER_LOAD_FORMAT_XYZW;
        }

        if has_buffers && self.code.has_any_of(&[T::BufferLoadUbyte]) {
            self.source += BUFFER_LOAD_UBYTE;
        }

        if has_buffers
            && self.code.has_any_of(&[
                T::BufferStoreDword,
                T::BufferStoreDwordX4,
                T::BufferStoreFormatX,
                T::BufferStoreFormatXy,
            ])
        {
            self.source += BUFFER_STORE_FLOAT1;
            self.source += BUFFER_STORE_FLOAT2;
            self.source += TBUFFER_STORE_FORMAT_X;
            self.source += TBUFFER_STORE_FORMAT_XY;
        }

        if has_buffers && self.code.has_any_of(&[T::BufferStoreFormatXyzw]) {
            self.source += BUFFER_STORE_FLOAT4;
            self.source += TBUFFER_STORE_FORMAT_XYZW;
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
            self.source += SBUFFER_LOAD_DWORD_8;
            self.source += SBUFFER_LOAD_DWORD_16;
        }
    }

    /// Kyty: `Spirv::FindConstants` (L7940).
    fn find_constants(&mut self) -> Result<(), ShaderRecompileError> {
        self.constants.clear();
        self.add_constant_float(0.0);
        self.add_constant_float(0.5);
        self.add_constant_float(1.0);
        self.add_constant_float(2.0);
        // 3.0 and 5.0 are the odd cube-face ids emitted by V_CUBEID_F32.
        self.add_constant_float(3.0);
        self.add_constant_float(4.0);
        self.add_constant_float(5.0);
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
            self.add_constant_uint(0x0000_ffff);
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
        // SDWA lane selects need the extraction shift/mask constants (the
        // shifts 0/8/16/24 are already in the unconditional 0..=32 block).
        if self
            .code
            .get_instructions()
            .iter()
            .any(|inst| inst.src.iter().any(|op| op.lane_sel != 6))
        {
            self.add_constant_uint(0xff);
            self.add_constant_uint(0xffff);
        }
        if self.code.has_any_of(&[ShaderInstructionType::SBarrier]) {
            // OpControlBarrier memory semantics: AcquireRelease (0x8) |
            // WorkgroupMemory (0x100).
            self.add_constant_uint(0x108);
        }
        if self.uses_lds() {
            // The LDS index clamp bound (see recompile_ds_write/read_b32).
            self.add_constant_uint(self.lds_size_dw() - 1);
        }
        // DS_READ_B96/B128 derive their 2nd..Nth dword offsets from the
        // single encoded byte-offset literal (`offset + 4k`) — materialise
        // them so the recompiler's `get_constant_uint` lookups resolve.
        let multi_dw_read_offsets: Vec<(u32, u32)> = self
            .code
            .get_instructions()
            .iter()
            .filter_map(|i| match i.type_ {
                ShaderInstructionType::DsReadB96 => Some((i.src[1].constant.u, 3)),
                ShaderInstructionType::DsReadB128 => Some((i.src[1].constant.u, 4)),
                _ => None,
            })
            .collect();
        for (off, n) in multi_dw_read_offsets {
            for k in 1..n {
                self.add_constant_uint(off + 4 * k);
            }
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
