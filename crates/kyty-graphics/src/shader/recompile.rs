//! GCN -> SPIR-V recompile functions + dispatch table + recompile entry
//! points, ported from Kyty (MIT (c) InoriRus).
//!
//! Kyty sources:
//! - `emulator/src/Graphics/ShaderSpirv.cpp`: `SccCheck` L1434, snippet
//!   templates `EXECZ`/`SCC_*`/`CLAMP`/`MULTIPLY` L1363-1430,
//!   `get_scc_check` L1849, the `KYTY_RECOMPILER_FUNC` bodies (anchors on
//!   each function below), `RecompilerFunc` L1555 and the `g_recomp_func`
//!   table inside `RecompFunc` L6182-6465.
//! - `emulator/src/Graphics/Shader.cpp`: `SpirvRun` L845,
//!   `ShaderRecompileVS` L2361, `ShaderRecompilePS` L2461,
//!   `ShaderRecompileCS` L2545.
//!
//! C1 scope: the dispatch table is complete (every Kyty row is present as
//! data), but only the minimal VS/PS subset of `Recompile_*` functions is
//! ported; the rest are [`RecompileFn::NotImplemented`] markers carrying the
//! Kyty function name + line anchor so C2 can fill them in mechanically and
//! coverage stays measurable (see `dispatch_table_counts` test).
//!
//! Deviations:
//! - Kyty `EXIT`/`EXIT_NOT_IMPLEMENTED` aborts become
//!   [`ShaderRecompileError`].
//! - `SpirvRun` runs SPIRV-Tools Assemble -> (config-gated) Validate ->
//!   Optimize; the port's [`spirv_run`] is the pure-Rust
//!   [`crate::spirv_asm::assemble`] only. Validation is exercised in tests
//!   through naga; optimization is intentionally skipped.
//! - `ShaderLogHelper` dump-to-file logging is replaced by `tracing`.

use std::collections::HashMap;
use std::sync::OnceLock;

pub use super::spirv::ShaderRecompileError;
use super::spirv::{
    Spirv, SpirvType, not_supported, operand_is_constant, operand_is_exec, operand_is_variable,
    operand_load_float, operand_load_uint, operand_variable_to_str, operand_variable_to_str_shift,
    spirv_generate_source, spirv_get_embedded_ps, spirv_get_embedded_vs,
};
use crate::shader::resources::{
    ShaderComputeInputInfo, ShaderPixelInputInfo, ShaderVertexInputInfo,
};
use crate::shader::types::{
    ShaderCode, ShaderInstruction, ShaderInstructionType, ShaderLabel, ShaderOperandType,
    shader_instruction_format::Format,
};

/// Kyty: ShaderSpirv.cpp `SccCheck` (L1434).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum SccCheck {
    #[default]
    None,
    NonZero,
    OverflowAdd,
    OverflowSub,
    CarryOut,
}

// ---------------------------------------------------------------------------
// Snippet templates (verbatim from Kyty)
// ---------------------------------------------------------------------------

/// Kyty: ShaderSpirv.cpp `EXECZ` (L1363).
pub(crate) const EXECZ: &str = r#"
        %z191_<index> = OpLoad %uint %exec_lo
        %z192_<index> = OpIEqual %bool %z191_<index> %uint_0
        %z193_<index> = OpLoad %uint %exec_hi
        %z194_<index> = OpIEqual %bool %z193_<index> %uint_0
        %z195_<index> = OpLogicalAnd %bool %z192_<index> %z194_<index>
        %z196_<index> = OpSelect %uint %z195_<index> %uint_1 %uint_0
               OpStore %execz %z196_<index>
"#;

/// Kyty: ShaderSpirv.cpp `SCC_NZ_1` (L1373).
pub(crate) const SCC_NZ_1: &str = r#"
        %snz1_118_<index> = OpLoad %uint %<dst>
        %snz1_121_<index> = OpINotEqual %bool %snz1_118_<index> %uint_0
        %snz1_123_<index> = OpSelect %uint %snz1_121_<index> %uint_1 %uint_0
               OpStore %scc %snz1_123_<index>
"#;

/// Kyty: ShaderSpirv.cpp `SCC_NZ_2` (L1380).
pub(crate) const SCC_NZ_2: &str = r#"
        %snz2_124_<index> = OpLoad %uint %<dst0>
        %snz2_125_<index> = OpINotEqual %bool %snz2_124_<index> %uint_0
        %snz2_127_<index> = OpLoad %uint %<dst1>
        %snz2_128_<index> = OpINotEqual %bool %snz2_127_<index> %uint_0
        %snz2_129_<index> = OpLogicalOr %bool %snz2_125_<index> %snz2_128_<index>
        %snz2_130_<index> = OpSelect %uint %snz2_129_<index> %uint_1 %uint_0
               OpStore %scc %snz2_130_<index>
"#;

/// Kyty: ShaderSpirv.cpp `SCC_OVERFLOW_ADD_1` (L1390).
pub(crate) const SCC_OVERFLOW_ADD_1: &str = r#"
        %so1_124_<index> = OpExtInst %int %GLSL_std_450 SSign %t0_<index>
        %so1_127_<index> = OpExtInst %int %GLSL_std_450 SSign %t1_<index>
        %so1_129_<index> = OpLoad %uint %<dst>
        %so1_130_<index> = OpBitcast %int %so1_129_<index>
        %so1_131_<index> = OpExtInst %int %GLSL_std_450 SSign %so1_130_<index>
        %so1_135_<index> = OpIEqual %bool %so1_124_<index> %so1_127_<index>
        %so1_138_<index> = OpINotEqual %bool %so1_131_<index> %so1_124_<index>
        %so1_139_<index> = OpLogicalAnd %bool %so1_135_<index> %so1_138_<index>
        %so1_142_<index> = OpSelect %uint %so1_139_<index> %uint_1 %uint_0
               OpStore %scc %so1_142_<index>
"#;

/// Kyty: ShaderSpirv.cpp `SCC_OVERFLOW_SUB_1` (L1403).
pub(crate) const SCC_OVERFLOW_SUB_1: &str = r#"
        %so1_124_<index> = OpExtInst %int %GLSL_std_450 SSign %t0_<index>
        %so1_127_<index> = OpExtInst %int %GLSL_std_450 SSign %t1_<index>
        %so1_129_<index> = OpLoad %uint %<dst>
        %so1_130_<index> = OpBitcast %int %so1_129_<index>
        %so1_131_<index> = OpExtInst %int %GLSL_std_450 SSign %so1_130_<index>
        %so1_135_<index> = OpINotEqual %bool %so1_124_<index> %so1_127_<index>
        %so1_138_<index> = OpINotEqual %bool %so1_131_<index> %so1_124_<index>
        %so1_139_<index> = OpLogicalAnd %bool %so1_135_<index> %so1_138_<index>
        %so1_142_<index> = OpSelect %uint %so1_139_<index> %uint_1 %uint_0
               OpStore %scc %so1_142_<index>
"#;

/// Kyty: ShaderSpirv.cpp `SCC_CARRY_1` (L1416).
pub(crate) const SCC_CARRY_1: &str = r#"
        OpStore %scc %carry_<index>
"#;

/// Kyty: ShaderSpirv.cpp `CLAMP` (L1420).
pub(crate) const CLAMP: &str = r#"
		%c197_<index> = OpLoad %float %<dst>
        %c200_<index> = OpExtInst %float %GLSL_std_450 FClamp %c197_<index> %float_0_000000 %float_1_000000
               OpStore %<dst> %c200_<index>
"#;

/// Kyty: ShaderSpirv.cpp `MULTIPLY` (L1426).
pub(crate) const MULTIPLY: &str = r#"
		%m197_<index> = OpLoad %float %<dst>
        %m200_<index> = OpFMul %float %m197_<index> %<mul>
               OpStore %<dst> %m200_<index>
"#;

/// Kyty: ShaderSpirv.cpp `get_scc_check` (L1849). The `dst_num == 2`
/// overflow/carry arms are `KYTY_NOT_IMPLEMENTED` upstream too — no ported
/// C1 function applies an SCC check yet (application lands with C2).
#[must_use]
pub fn get_scc_check(scc_check: SccCheck, dst_num: i32) -> &'static str {
    if dst_num == 1 {
        match scc_check {
            SccCheck::NonZero => return SCC_NZ_1,
            SccCheck::OverflowAdd => return SCC_OVERFLOW_ADD_1,
            SccCheck::OverflowSub => return SCC_OVERFLOW_SUB_1,
            SccCheck::CarryOut => return SCC_CARRY_1,
            SccCheck::None => {}
        }
    } else if dst_num == 2 {
        match scc_check {
            SccCheck::NonZero => return SCC_NZ_2,
            SccCheck::OverflowAdd | SccCheck::OverflowSub | SccCheck::CarryOut => {
                // Kyty: KYTY_NOT_IMPLEMENTED (ShaderSpirv.cpp L1868-1870).
                tracing::error!("get_scc_check: dst_num == 2 with {scc_check:?} not implemented");
            }
            SccCheck::None => {}
        }
    }
    ""
}

/// Kyty: `inst_recompile_func_t` (ShaderSpirv.cpp L1443) —
/// `KYTY_RECOMPILER_ARGS` (L16) with the abort paths turned into `Result`.
/// `Ok(false)` = Kyty `return false` ("can't recompile this instance").
pub type InstRecompileFn = fn(
    u32,
    &ShaderCode,
    &mut String,
    &Spirv<'_>,
    &[Option<&'static str>; 4],
    SccCheck,
) -> Result<bool, ShaderRecompileError>;

type Params = [Option<&'static str>; 4];

fn inst_at(
    code: &ShaderCode,
    index: u32,
    func: &'static str,
) -> Result<ShaderInstruction, ShaderRecompileError> {
    code.get_instructions()
        .get(index as usize)
        .copied()
        .ok_or_else(|| not_supported(func, format!("instruction index {index} out of range")))
}

// ---------------------------------------------------------------------------
// Recompile functions (C1 minimal VS/PS subset)
// ---------------------------------------------------------------------------

/// Kyty: `Recompile_BufferLoadDword_Vdata1VaddrSvSoffsIdxen`
/// (ShaderSpirv.cpp L1877).
fn recompile_buffer_load_dword_vdata1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_BufferLoadDword_Vdata1VaddrSvSoffsIdxen";
    let inst = inst_at(code, index, FUNC)?;
    let bind_info = spirv.get_bind_info();

    if let Some(bind_info) = bind_info {
        if bind_info.storage_buffers.buffers_num > 0 {
            if !operand_is_constant(inst.src[2]) {
                return Err(not_supported(FUNC, "src2 is not a constant"));
            }

            let dst_value = operand_variable_to_str(inst.dst);
            let src0_value = operand_variable_to_str(inst.src[0]);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
            let offset = spirv.get_constant(inst.src[2]);

            if dst_value.type_ != SpirvType::Float
                || src0_value.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src1_value1.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check EXEC

            const TEXT: &str = r#"
        %t100_<index> = OpLoad %float %<src0>
        %t101_<index> = OpBitcast %int %t100_<index>
               OpStore %temp_int_1 %t101_<index>
        %t148_<index> = OpLoad %uint %<src1_value1>
        %t150_<index> = OpShiftRightLogical %uint %t148_<index> %int_16
        %t152_<index> = OpBitwiseAnd %uint %t150_<index> %uint_0x00003fff
        %t153_<index> = OpBitcast %int %t152_<index>
               OpStore %temp_int_3 %t153_<index>
        %t155_<index> = OpLoad %uint %<src1_value0>
        %t156_<index> = OpBitcast %int %t155_<index>
               OpStore %temp_int_4 %t156_<index>
               OpStore %temp_int_2 %<offset>
        %t110_<index> = OpFunctionCall %void %buffer_load_float1 %<p0> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0>", &src0_value.value)
                .replace("<offset>", &offset)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value1>", &src1_value1.value)
                .replace("<p0>", &dst_value.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_BufferLoadFormatX_Vdata1VaddrSvSoffsIdxen`
/// (ShaderSpirv.cpp L1937).
fn recompile_buffer_load_format_x_vdata1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_BufferLoadFormatX_Vdata1VaddrSvSoffsIdxen";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.storage_buffers.buffers_num > 0 {
            if !operand_is_constant(inst.src[2]) {
                return Err(not_supported(FUNC, "src2 is not a constant"));
            }

            let dst_value = operand_variable_to_str(inst.dst);
            let src0_value = operand_variable_to_str(inst.src[0]);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
            let src1_value3 = operand_variable_to_str_shift(inst.src[1], 3);
            let offset = spirv.get_constant(inst.src[2]);

            if dst_value.type_ != SpirvType::Float
                || src0_value.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src1_value1.type_ != SpirvType::Uint
                || src1_value3.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check EXEC

            const TEXT: &str = r#"
        %t100_<index> = OpLoad %float %<src0>
        %t101_<index> = OpBitcast %int %t100_<index>
               OpStore %temp_int_1 %t101_<index>
        %t148_<index> = OpLoad %uint %<src1_value1>
        %t150_<index> = OpShiftRightLogical %uint %t148_<index> %int_16
        %t152_<index> = OpBitwiseAnd %uint %t150_<index> %uint_0x00003fff
        %t153_<index> = OpBitcast %int %t152_<index>
               OpStore %temp_int_3 %t153_<index>
        %t155_<index> = OpLoad %uint %<src1_value0>
        %t156_<index> = OpBitcast %int %t155_<index>
               OpStore %temp_int_4 %t156_<index>
               OpStore %temp_int_2 %<offset>
		%t206_<index> = OpLoad %uint %<src1_value3>
        %t208_<index> = OpShiftRightLogical %uint %t206_<index> %int_12
        %t210_<index> = OpBitwiseAnd %uint %t208_<index> %uint_127
        %t211_<index> = OpBitcast %int %t210_<index>
               OpStore %temp_int_5 %t211_<index>
        %t110_<index> = OpFunctionCall %void %tbuffer_load_format_x %<p0> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4 %temp_int_5
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0>", &src0_value.value)
                .replace("<offset>", &offset)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value1>", &src1_value1.value)
                .replace("<src1_value3>", &src1_value3.value)
                .replace("<p0>", &dst_value.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_Exp_Mrt0OffOffComprVmDone` (ShaderSpirv.cpp L2278).
fn recompile_exp_mrt0_off_off_compr_vm_done(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_Mrt0OffOffComprVmDone";
    if index == 0 || index as usize + 1 >= code.get_instructions().len() {
        return Err(not_supported(FUNC, "index at program boundary"));
    }

    let prev_inst = inst_at(code, index - 1, FUNC)?;
    let inst = inst_at(code, index, FUNC)?;
    let block = code.read_block(prev_inst.pc);

    if !block.is_discard {
        return Ok(false);
    }

    let Some(info) = spirv.get_ps_input_info() else {
        return Err(not_supported(FUNC, "no ps input info"));
    };
    if !info.ps_pixel_kill_enable {
        return Err(not_supported(FUNC, "!ps_pixel_kill_enable"));
    }
    if info.target_output_mode[0] != 4 {
        return Err(not_supported(FUNC, "target_output_mode[0] != 4"));
    }
    if inst.src_num > 0 {
        return Err(not_supported(FUNC, "src_num > 0"));
    }

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
        OpKill
"#;

    *dst_source += TEXT;

    Ok(true)
}

/// Kyty: `Recompile_Exp_Mrt0Vsrc0Vsrc1ComprVmDone` (ShaderSpirv.cpp L2310).
fn recompile_exp_mrt0_vsrc0_vsrc1_compr_vm_done(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_Mrt0Vsrc0Vsrc1ComprVmDone";
    let inst = inst_at(code, index, FUNC)?;

    if !spirv
        .get_ps_input_info()
        .is_some_and(|i| i.target_output_mode[0] == 4)
    {
        return Err(not_supported(FUNC, "target_output_mode[0] != 4"));
    }

    if !operand_is_variable(inst.src[0]) || !operand_is_variable(inst.src[1]) {
        return Err(not_supported(FUNC, "sources are not variables"));
    }

    let src0_value = operand_variable_to_str(inst.src[0]);
    let src1_value = operand_variable_to_str(inst.src[1]);

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
         %t1_<index> = OpLoad %float %<src0>
         %t2_<index> = OpBitcast %uint %t1_<index>
         %t3_<index> = OpExtInst %v2float %GLSL_std_450 UnpackHalf2x16 %t2_<index>
         %t4_<index> = OpCompositeExtract %float %t3_<index> 0
         %t5_<index> = OpCompositeExtract %float %t3_<index> 1
         %t6_<index> = OpLoad %float %<src1>
         %t7_<index> = OpBitcast %uint %t6_<index>
         %t8_<index> = OpExtInst %v2float %GLSL_std_450 UnpackHalf2x16 %t7_<index>
         %t9_<index> = OpCompositeExtract %float %t8_<index> 0
         %t10_<index> = OpCompositeExtract %float %t8_<index> 1
         %t11_<index> = OpCompositeConstruct %v4float %t4_<index> %t5_<index> %t9_<index> %t10_<index>
               OpStore %outColor %t11_<index>
"#;

    *dst_source += &TEXT
        .replace("<index>", &format!("{index}"))
        .replace("<src0>", &src0_value.value)
        .replace("<src1>", &src1_value.value);

    Ok(true)
}

/// Kyty: `Recompile_Exp_Mrt0Vsrc0Vsrc1Vsrc2Vsrc3VmDone` (ShaderSpirv.cpp
/// L2348).
fn recompile_exp_mrt0_vsrc0123_vm_done(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_Mrt0Vsrc0Vsrc1Vsrc2Vsrc3VmDone";
    let inst = inst_at(code, index, FUNC)?;

    if !spirv
        .get_ps_input_info()
        .is_some_and(|i| i.target_output_mode[0] == 9)
    {
        return Err(not_supported(FUNC, "target_output_mode[0] != 9"));
    }

    if inst.src[..4].iter().any(|s| !operand_is_variable(*s)) {
        return Err(not_supported(FUNC, "sources are not variables"));
    }

    let src0_value = operand_variable_to_str(inst.src[0]);
    let src1_value = operand_variable_to_str(inst.src[1]);
    let src2_value = operand_variable_to_str(inst.src[2]);
    let src3_value = operand_variable_to_str(inst.src[3]);

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
         %t0_<index> = OpLoad %float %<src0>
         %t1_<index> = OpLoad %float %<src1>
         %t2_<index> = OpLoad %float %<src2>
         %t3_<index> = OpLoad %float %<src3>
         %t11_<index> = OpCompositeConstruct %v4float %t0_<index> %t1_<index> %t2_<index> %t3_<index>
               OpStore %outColor %t11_<index>
"#;

    *dst_source += &TEXT
        .replace("<index>", &format!("{index}"))
        .replace("<src0>", &src0_value.value)
        .replace("<src1>", &src1_value.value)
        .replace("<src2>", &src2_value.value)
        .replace("<src3>", &src3_value.value);

    Ok(true)
}

/// Kyty: `Recompile_Exp_Param_XXX_Vsrc0Vsrc1Vsrc2Vsrc3` (ShaderSpirv.cpp
/// L2387). XXX: 0, 1, 2, 3, 4 (via `param[0]`).
fn recompile_exp_param_xxx(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    _spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_Param_XXX_Vsrc0Vsrc1Vsrc2Vsrc3";
    let inst = inst_at(code, index, FUNC)?;

    if inst.src[..4].iter().any(|s| !operand_is_variable(*s)) {
        return Err(not_supported(FUNC, "sources are not variables"));
    }

    let src0_value = operand_variable_to_str(inst.src[0]);
    let src1_value = operand_variable_to_str(inst.src[1]);
    let src2_value = operand_variable_to_str(inst.src[2]);
    let src3_value = operand_variable_to_str(inst.src[3]);

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
         %t0_<index> = OpLoad %float %<src0>
         %t1_<index> = OpLoad %float %<src1>
         %t2_<index> = OpLoad %float %<src2>
         %t3_<index> = OpLoad %float %<src3>
         %t4_<index> = OpCompositeConstruct %v4float %t0_<index> %t1_<index> %t2_<index> %t3_<index>
               OpStore %<param> %t4_<index>
"#;

    *dst_source += &TEXT
        .replace("<index>", &format!("{index}"))
        .replace("<src0>", &src0_value.value)
        .replace("<src1>", &src1_value.value)
        .replace("<src2>", &src2_value.value)
        .replace("<src3>", &src3_value.value)
        .replace("<param>", param[0].unwrap_or(""));

    Ok(true)
}

/// Kyty: `Recompile_Exp_Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done` (ShaderSpirv.cpp
/// L2424).
fn recompile_exp_pos0(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    _spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done";
    let inst = inst_at(code, index, FUNC)?;

    if inst.src[..4].iter().any(|s| !operand_is_variable(*s)) {
        return Err(not_supported(FUNC, "sources are not variables"));
    }

    let src0_value = operand_variable_to_str(inst.src[0]);
    let src1_value = operand_variable_to_str(inst.src[1]);
    let src2_value = operand_variable_to_str(inst.src[2]);
    let src3_value = operand_variable_to_str(inst.src[3]);

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
         %t0_<index> = OpLoad %float %<src0>
         %t1_<index> = OpLoad %float %<src1>
         %t2_<index> = OpLoad %float %<src2>
         %t3_<index> = OpLoad %float %<src3>
         %t4_<index> = OpCompositeConstruct %v4float %t0_<index> %t1_<index> %t2_<index> %t3_<index>
         %t5_<index> = OpAccessChain %_ptr_Output_v4float %outPerVertex %int_per_vertex_0
               OpStore %t5_<index> %t4_<index>
"#;

    *dst_source += &TEXT
        .replace("<index>", &format!("{index}"))
        .replace("<src0>", &src0_value.value)
        .replace("<src1>", &src1_value.value)
        .replace("<src2>", &src2_value.value)
        .replace("<src3>", &src3_value.value);

    Ok(true)
}

/// Kyty: `Recompile_Exp_PrimVsrc0OffOffOffDone` (ShaderSpirv.cpp L2461).
fn recompile_exp_prim(
    index: u32,
    code: &ShaderCode,
    _dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_PrimVsrc0OffOffOffDone";
    let inst = inst_at(code, index, FUNC)?;
    let vs_info = spirv.get_vs_input_info();

    if !operand_is_variable(inst.src[0]) {
        return Err(not_supported(FUNC, "src0 is not a variable"));
    }

    Ok(vs_info.is_some_and(|v| v.gs_prolog))
}

/// Kyty: `Recompile_SBufferLoadDword_SdstSvSoffset` (ShaderSpirv.cpp L3794).
fn recompile_sbuffer_load_dword(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SBufferLoadDword_SdstSvSoffset";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.storage_buffers.buffers_num > 0 {
            if !operand_is_constant(inst.src[1]) {
                return Err(not_supported(FUNC, "src1 is not a constant"));
            }

            let dst_value = operand_variable_to_str(inst.dst);
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let offset = spirv.get_constant(inst.src[1]);

            if dst_value.type_ != SpirvType::Uint || src0_value0.type_ != SpirvType::Uint {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }
            if operand_is_exec(inst.dst) {
                return Err(not_supported(FUNC, "exec destination"));
            }

            const TEXT: &str = r#"
        %t100_<index> = OpLoad %uint %<src0_value0>
        %t101_<index> = OpBitcast %int %t100_<index>
               OpStore %temp_int_2 %t101_<index>
        %t102_<index> = OpBitcast %int %<offset>
               OpStore %temp_int_1 %t102_<index>
        %t110_<index> = OpFunctionCall %void %sbuffer_load_dword %<p0> %temp_int_1 %temp_int_2
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<offset>", &offset)
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<p0>", &dst_value.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_SBufferLoadDwordx4_Sdst4SvSoffset` (ShaderSpirv.cpp
/// L3872).
fn recompile_sbuffer_load_dwordx4(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SBufferLoadDwordx4_Sdst4SvSoffset";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.storage_buffers.buffers_num > 0 {
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let dst_value3 = operand_variable_to_str_shift(inst.dst, 3);
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);

            if dst_value0.type_ != SpirvType::Uint || src0_value0.type_ != SpirvType::Uint {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }
            if operand_is_exec(inst.dst) {
                return Err(not_supported(FUNC, "exec destination"));
            }

            let index_str = format!("{index}");

            let mut load1 = String::new();
            if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
                return Ok(false);
            }

            const TEXT: &str = r#"
        <load1>
        %t100_<index> = OpLoad %uint %<src0_value0>
        %t101_<index> = OpBitcast %int %t100_<index>
               OpStore %temp_int_2 %t101_<index>
        %t102_<index> = OpBitcast %int %t1_<index>
               OpStore %temp_int_1 %t102_<index>
        %t110_<index> = OpFunctionCall %void %sbuffer_load_dword_4 %<p0> %<p1> %<p2> %<p3> %temp_int_1 %temp_int_2
"#;
            *dst_source += &TEXT
                .replace("<load1>", &load1)
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<p0>", &dst_value0.value)
                .replace("<p1>", &dst_value1.value)
                .replace("<p2>", &dst_value2.value)
                .replace("<p3>", &dst_value3.value)
                .replace("<index>", &index_str);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_SBranch_Label` (ShaderSpirv.cpp L4047).
fn recompile_sbranch_label(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    _spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SBranch_Label";
    let inst = inst_at(code, index, FUNC)?;

    if !operand_is_constant(inst.src[0]) {
        return Err(not_supported(FUNC, "src0 is not a constant"));
    }

    let label = ShaderLabel::from_instruction(&inst);

    if code.read_block(label.get_dst()).is_discard {
        return Err(not_supported(FUNC, "branch to discard block"));
    }

    const TEXT: &str = r#"
                OpBranch %<label>
"#;

    *dst_source += &TEXT
        .replace("<index>", &format!("{index}"))
        .replace("<label>", &label.to_string());

    Ok(true)
}

/// Kyty: `Recompile_SCbranch_XXX_Label` (ShaderSpirv.cpp L4067).
/// XXX: Execz, Scc0, Scc1, Vccz, Vccnz (condition load via `param[0..1]`).
fn recompile_scbranch_xxx_label(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    _spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SCbranch_XXX_Label";
    if index as usize + 1 >= code.get_instructions().len() {
        return Err(not_supported(FUNC, "branch at end of program"));
    }

    let inst = inst_at(code, index, FUNC)?;
    let next_inst = inst_at(code, index + 1, FUNC)?;

    if !operand_is_constant(inst.src[0]) {
        return Err(not_supported(FUNC, "src0 is not a constant"));
    }

    // TODO(): analyze control flow graph
    let label = ShaderLabel::from_instruction(&inst);
    let dst_block = code.read_block(label.get_dst());
    let next_block = code.read_block(next_inst.pc);
    let discard = dst_block.is_discard;
    let label_next_block = ShaderLabel::from_instruction(&next_block.last);
    let label_dst_block = ShaderLabel::from_instruction(&dst_block.last);

    let if_else = next_block.is_valid
        && !next_block.is_discard
        && dst_block.is_valid
        && !dst_block.is_discard
        && ((next_block.last.type_ == ShaderInstructionType::SBranch
            && label_next_block.get_dst() >= dst_block.pc
            && label_next_block.get_dst() <= dst_block.last.pc)
            || (dst_block.last.type_ == ShaderInstructionType::SBranch
                && label_dst_block.get_dst() >= next_block.pc
                && label_dst_block.get_dst() <= next_block.last.pc));

    let label_str = label.to_string();
    let label_merge = if if_else {
        if dst_block.last.type_ == ShaderInstructionType::SBranch {
            label_dst_block.to_string()
        } else {
            label_next_block.to_string()
        }
    } else {
        String::new()
    };

    const TEXT_VARIANT_A: &str = r#"
        <param0>
        <param1>
               OpSelectionMerge %<label> None
               OpBranchConditional %cc_b_<index> %<label> %t230_<index>
        %t230_<index> = OpLabel
"#;

    const TEXT_VARIANT_B: &str = r#"
        <param0>
        <param1>
               OpSelectionMerge %t230_<index> None
               OpBranchConditional %cc_b_<index> %<label> %t230_<index>
        %t230_<index> = OpLabel
"#;

    const TEXT_VARIANT_C: &str = r#"
        <param0>
        <param1>
               OpSelectionMerge %<merge> None
               OpBranchConditional %cc_b_<index> %<label> %t230_<index>
        %t230_<index> = OpLabel
"#;

    let text = if if_else {
        TEXT_VARIANT_C
    } else if discard {
        TEXT_VARIANT_B
    } else {
        TEXT_VARIANT_A
    };

    *dst_source += &text
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<merge>", &label_merge)
        .replace("<index>", &format!("{index}"))
        .replace("<label>", &label_str);

    Ok(true)
}

/// Kyty: `Recompile_SEndpgm_Empty` (ShaderSpirv.cpp L4258).
fn recompile_sendpgm_empty(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    _spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SEndpgm_Empty";

    const TEXT: &str = r#"
       OpReturn
"#;

    if index < 2 {
        return Err(not_supported(FUNC, "s_endpgm before instruction 2"));
    }

    let prev_prev_inst = inst_at(code, index - 2, FUNC)?;
    let prev_inst = inst_at(code, index - 1, FUNC)?;

    let after_kill = prev_prev_inst.type_ == ShaderInstructionType::SMovB64
        && prev_prev_inst.format == Format::Sdst2Ssrc02
        && prev_prev_inst.dst.type_ == ShaderOperandType::ExecLo
        && prev_prev_inst.src[0].type_ == ShaderOperandType::IntegerInlineConstant
        && prev_prev_inst.src[0].constant.i() == 0
        && prev_inst.type_ == ShaderInstructionType::Exp
        && prev_inst.format == Format::Mrt0OffOffComprVmDone;

    if !after_kill {
        *dst_source += TEXT;
    }

    Ok(true)
}

/// Kyty: `Recompile_SLoadDword_SdstSbaseSoffset` (ShaderSpirv.cpp L4287) —
/// pure gate: skip the load when it belongs to the embedded fetch shader.
fn recompile_sload_dword(
    index: u32,
    code: &ShaderCode,
    _dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SLoadDword_SdstSbaseSoffset";
    let inst = inst_at(code, index, FUNC)?;
    let vs_info = spirv.get_vs_input_info();

    let shift_regs = if vs_info.is_some_and(|v| v.gs_prolog) {
        8
    } else {
        0
    };

    Ok(vs_info.is_some_and(|v| {
        v.fetch_embedded
            && !v.fetch_external
            && !v.fetch_inline
            && (inst.src[0].register_id == v.fetch_attrib_reg + shift_regs
                || inst.src[0].register_id == v.fetch_buffer_reg + shift_regs)
    }))
}

/// Kyty: `Recompile_SLoadDwordx2_Sdst2Ssrc02Ssrc1` (ShaderSpirv.cpp L4299).
fn recompile_sload_dwordx2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    recompile_sload_dword(index, code, dst_source, spirv, param, scc_check)
}

/// Common extended (EUD) V#-from-push-constants path of
/// `Recompile_SLoadDwordx4/x8` (ShaderSpirv.cpp L4325-4369 / L4388-4432).
fn sload_dword_extended(
    index: u32,
    inst: &ShaderInstruction,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    n: i32,
    func: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let Some(bind_info) = spirv.get_bind_info() else {
        return Ok(false);
    };
    if !bind_info.extended.used {
        return Ok(false);
    }

    if inst.src[1].type_ != ShaderOperandType::LiteralConstant {
        return Err(not_supported(func, "src1 is not a literal constant"));
    }
    if inst.src[0].register_id != bind_info.extended.start_register {
        return Err(not_supported(func, "src0 is not the EUD base register"));
    }

    // TODO() check pointer

    let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
    let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
    let offset = (inst.src[1].constant.u >> 2u32) as i32;

    if src0_value0.type_ != SpirvType::Uint || src0_value1.type_ != SpirvType::Uint {
        return Err(not_supported(func, "unexpected src0 type"));
    }

    const TEXT: &str = r#"
		         %vsharp_<index>_<reg> = OpAccessChain %_ptr_PushConstant_uint %vsharp %int_0 %int_<buffer> %int_<field>
		         %vsharp_<index>_value_<reg> = OpLoad %uint %vsharp_<index>_<reg>
		               OpStore %<reg> %vsharp_<index>_value_<reg>
				"#;

    for i in 0..n {
        let dst_value = operand_variable_to_str_shift(inst.dst, i);
        if i == 0 && dst_value.type_ != SpirvType::Uint {
            return Err(not_supported(func, "unexpected dst type"));
        }
        let (buffer, field) = spirv.get_mapped_index(offset + i)?;

        *dst_source += &TEXT
            .replace("<reg>", &dst_value.value)
            .replace("<buffer>", &format!("{buffer}"))
            .replace("<field>", &format!("{field}"))
            .replace("<index>", &format!("{index}"));
    }

    Ok(true)
}

/// Kyty: `Recompile_SLoadDwordx4_Sdst4SbaseSoffset` (ShaderSpirv.cpp L4311).
fn recompile_sload_dwordx4(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SLoadDwordx4_Sdst4SbaseSoffset";
    let inst = inst_at(code, index, FUNC)?;
    let vs_info = spirv.get_vs_input_info();

    let shift_regs = if vs_info.is_some_and(|v| v.gs_prolog) {
        8
    } else {
        0
    };

    if vs_info.is_some_and(|v| {
        v.fetch_embedded
            && !v.fetch_external
            && !v.fetch_inline
            && (inst.src[0].register_id == v.fetch_attrib_reg + shift_regs
                || inst.src[0].register_id == v.fetch_buffer_reg + shift_regs)
    }) {
        return Ok(true);
    }

    if spirv.get_bind_info().is_some_and(|b| b.extended.used) && shift_regs != 0 {
        return Err(not_supported(FUNC, "extended path with gs_prolog shift"));
    }

    sload_dword_extended(index, &inst, dst_source, spirv, 4, FUNC)
}

/// Kyty: `Recompile_SLoadDwordx8_Sdst8SbaseSoffset` (ShaderSpirv.cpp L4374).
fn recompile_sload_dwordx8(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SLoadDwordx8_Sdst8SbaseSoffset";
    let inst = inst_at(code, index, FUNC)?;
    let vs_info = spirv.get_vs_input_info();

    let shift_regs = if vs_info.is_some_and(|v| v.gs_prolog) {
        8
    } else {
        0
    };

    if vs_info.is_some_and(|v| {
        v.fetch_embedded
            && !v.fetch_external
            && !v.fetch_inline
            && (inst.src[0].register_id == v.fetch_attrib_reg + shift_regs
                || inst.src[0].register_id == v.fetch_buffer_reg + shift_regs)
    }) {
        return Ok(true);
    }

    if spirv.get_bind_info().is_some_and(|b| b.extended.used) && shift_regs != 0 {
        return Err(not_supported(FUNC, "extended path with gs_prolog shift"));
    }

    sload_dword_extended(index, &inst, dst_source, spirv, 8, FUNC)
}

/// Kyty: `Recompile_SMovB32_SVdstSVsrc0` (ShaderSpirv.cpp L4480). Also
/// serves `SMovkI32` (same table row function upstream).
fn recompile_smov_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SMovB32_SVdstSVsrc0";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    if dst_value.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "dst is not uint"));
    }
    if operand_is_exec(inst.dst) {
        return Err(not_supported(FUNC, "exec destination"));
    }

    let mut load0 = String::new();
    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
    <load0>
    OpStore %<dst> %t0_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_SMovB64_Sdst2Ssrc02` (ShaderSpirv.cpp L4509).
fn recompile_smov_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SMovB64_Sdst2Ssrc02";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }

    let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
    let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);

    if dst_value0.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "dst is not uint"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, 0)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[0], "t1_<index>", &index_str, &mut load1, 1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
    <load0>
    <load1>
    OpStore %<dst0> %t0_<index>
    OpStore %<dst1> %t1_<index>
    <execz>
"#;
    *dst_source += &TEXT
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace(
            "<execz>",
            if operand_is_exec(inst.dst) { EXECZ } else { "" },
        )
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_SSwappcB64_Sdst2Ssrc02` (ShaderSpirv.cpp L4554) — the
/// external fetch-shader call: inline `fetch_*` calls per vertex attribute.
fn recompile_sswappc_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SSwappcB64_Sdst2Ssrc02";
    let inst = inst_at(code, index, FUNC)?;
    let input_info = spirv.get_vs_input_info();

    if let Some(info) = input_info {
        if !info.fetch_external {
            return Err(not_supported(FUNC, "!fetch_external"));
        }
        if info.fetch_shader_reg != 0 {
            return Err(not_supported(FUNC, "fetch_shader_reg != 0"));
        }
    }

    if let Some(info) = input_info {
        if info.fetch_external
            && inst.dst.type_ == ShaderOperandType::Sgpr
            && inst.dst.register_id == 0
            && inst.src[0].type_ == ShaderOperandType::Sgpr
            && inst.src[0].register_id == 0
            && index == 1
        {
            for i in 0..info.resources_num {
                let r = info.resources_dst[i as usize];

                let text = match r.registers_num {
                    1 => {
                        r#"
				         %t1_<index> = OpLoad %float %<attr>
				                       OpStore %temp_float %t1_<index>
				         %t2_<index> = OpFunctionCall %void %fetch_f1_f1_ %<p0> %temp_float
				"#
                    }
                    2 => {
                        r#"
				         %t1_<index> = OpLoad %v2float %<attr>
				                       OpStore %temp_v2float %t1_<index>
				         %t2_<index> = OpFunctionCall %void %fetch_f1_f1_vf2_ %<p0> %<p1> %temp_v2float
				"#
                    }
                    3 => {
                        r#"
				         %t1_<index> = OpLoad %v3float %<attr>
				                       OpStore %temp_v3float %t1_<index>
				         %t2_<index> = OpFunctionCall %void %fetch_f1_f1_f1_vf3_ %<p0> %<p1> %<p2> %temp_v3float
				"#
                    }
                    4 => {
                        r#"
				         %t1_<index> = OpLoad %v4float %<attr>
				                       OpStore %temp_v4float %t1_<index>
				         %t2_<index> = OpFunctionCall %void %fetch_f1_f1_f1_f1_vf4_ %<p0> %<p1> %<p2> %<p3> %temp_v4float
				"#
                    }
                    n => {
                        return Err(not_supported(FUNC, format!("invalid registers_num: {n}")));
                    }
                };

                *dst_source += &text
                    .replace("<index>", &format!("{i}_{index}"))
                    .replace("<p0>", &format!("v{}", r.register_start))
                    .replace("<p1>", &format!("v{}", r.register_start + 1))
                    .replace("<p2>", &format!("v{}", r.register_start + 2))
                    .replace("<p3>", &format!("v{}", r.register_start + 3))
                    .replace("<attr>", &format!("attr{i}"));
            }
            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_Skip` (ShaderSpirv.cpp L4707) — SWaitcnt / SSendmsg /
/// SInstPrefetch have no SPIR-V counterpart.
fn recompile_skip(
    _index: u32,
    _code: &ShaderCode,
    _dst_source: &mut String,
    _spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    Ok(true)
}

/// Kyty: `Recompile_TBufferLoadFormatX_Vdata1VaddrSvSoffsIdxenFloat1`
/// (ShaderSpirv.cpp L4712).
fn recompile_tbuffer_load_format_x_float1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_TBufferLoadFormatX_Vdata1VaddrSvSoffsIdxenFloat1";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.storage_buffers.buffers_num > 0 {
            if !operand_is_constant(inst.src[2]) {
                return Err(not_supported(FUNC, "src2 is not a constant"));
            }

            let dst_value0 = operand_variable_to_str(inst.dst);
            let src0_value = operand_variable_to_str(inst.src[0]);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
            let offset = spirv.get_constant(inst.src[2]);

            if dst_value0.type_ != SpirvType::Float
                || src0_value.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src1_value1.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check EXEC

            const TEXT: &str = r#"
        %t100_<index> = OpLoad %float %<src0>
        %t101_<index> = OpBitcast %int %t100_<index>
               OpStore %temp_int_1 %t101_<index>
        %t148_<index> = OpLoad %uint %<src1_value1>
        %t150_<index> = OpShiftRightLogical %uint %t148_<index> %int_16
        %t152_<index> = OpBitwiseAnd %uint %t150_<index> %uint_0x00003fff
        %t153_<index> = OpBitcast %int %t152_<index>
               OpStore %temp_int_3 %t153_<index>
        %t155_<index> = OpLoad %uint %<src1_value0>
        %t156_<index> = OpBitcast %int %t155_<index>
               OpStore %temp_int_4 %t156_<index>
               OpStore %temp_int_2 %<offset>
               OpStore %temp_int_5 %int_36
        %t110_<index> = OpFunctionCall %void %tbuffer_load_format_x %<p0> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4 %temp_int_5
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0>", &src0_value.value)
                .replace("<offset>", &offset)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value1>", &src1_value1.value)
                .replace("<p0>", &dst_value0.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_TBufferLoadFormatXyzw_Vdata4VaddrSvSoffsIdxenFloat4`
/// (ShaderSpirv.cpp L4765).
fn recompile_tbuffer_load_format_xyzw_float4(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_TBufferLoadFormatXyzw_Vdata4VaddrSvSoffsIdxenFloat4";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.storage_buffers.buffers_num > 0 {
            if !operand_is_constant(inst.src[2]) {
                return Err(not_supported(FUNC, "src2 is not a constant"));
            }

            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let dst_value3 = operand_variable_to_str_shift(inst.dst, 3);
            let src0_value = operand_variable_to_str(inst.src[0]);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
            let offset = spirv.get_constant(inst.src[2]);

            if dst_value0.type_ != SpirvType::Float
                || src0_value.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src1_value1.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check EXEC

            const TEXT: &str = r#"
        %t100_<index> = OpLoad %float %<src0>
        %t101_<index> = OpBitcast %int %t100_<index>
               OpStore %temp_int_1 %t101_<index>
        %t148_<index> = OpLoad %uint %<src1_value1>
        %t150_<index> = OpShiftRightLogical %uint %t148_<index> %int_16
        %t152_<index> = OpBitwiseAnd %uint %t150_<index> %uint_0x00003fff
        %t153_<index> = OpBitcast %int %t152_<index>
               OpStore %temp_int_3 %t153_<index>
        %t155_<index> = OpLoad %uint %<src1_value0>
        %t156_<index> = OpBitcast %int %t155_<index>
               OpStore %temp_int_4 %t156_<index>
               OpStore %temp_int_2 %<offset>
               OpStore %temp_int_5 %int_119
        %t110_<index> = OpFunctionCall %void %tbuffer_load_format_xyzw %<p0> %<p1> %<p2> %<p3> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4 %temp_int_5
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0>", &src0_value.value)
                .replace("<offset>", &offset)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value1>", &src1_value1.value)
                .replace("<p0>", &dst_value0.value)
                .replace("<p1>", &dst_value1.value)
                .replace("<p2>", &dst_value2.value)
                .replace("<p3>", &dst_value3.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_VCndmaskB32_VdstVsrc0Vsrc1Smask2` (ShaderSpirv.cpp
/// L5201).
fn recompile_vcndmask_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCndmaskB32_VdstVsrc0Vsrc1Smask2";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }
    if inst.dst.clamp {
        return Err(not_supported(FUNC, "clamp"));
    }
    if inst.dst.multiplier != 1.0 {
        return Err(not_supported(FUNC, "multiplier"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    if !operand_is_variable(inst.src[2]) {
        return Err(not_supported(FUNC, "src2 is not a variable"));
    }

    let src_bool_value0 = operand_variable_to_str_shift(inst.src[2], 0);
    let src_bool_value1 = operand_variable_to_str_shift(inst.src[2], 1);

    if src_bool_value0.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "src2 is not uint"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();
    if !operand_load_float(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_float(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }

    // TODO() check VSKIP

    const TEXT: &str = r#"
    <load0>
    <load1>
    %t22_<index> = OpLoad %uint %<src0>
    %t23_<index> = OpLoad %uint %<src1> ; unused
    %tb_<index> = OpBitwiseAnd %uint %t22_<index> %uint_1
    %t2_<index> = OpINotEqual %bool %tb_<index> %uint_0
    %t3_<index> = OpSelect %float %t2_<index> %t1_<index> %t0_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %t3_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<src0>", &src_bool_value0.value)
        .replace("<src1>", &src_bool_value1.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VInterpP1F32_VdstVsrcAttrChan` (ShaderSpirv.cpp L5315) —
/// P1 is folded into P2, so it emits nothing.
fn recompile_vinterp_p1_f32(
    _index: u32,
    _code: &ShaderCode,
    _dst_source: &mut String,
    _spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    Ok(true)
}

/// Kyty: `Recompile_VInterpP2F32_VdstVsrcAttrChan` (ShaderSpirv.cpp L5320).
fn recompile_vinterp_p2_f32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    _spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VInterpP2F32_VdstVsrcAttrChan";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst)
        || !operand_is_variable(inst.src[0])
        || !operand_is_constant(inst.src[1])
        || !operand_is_constant(inst.src[2])
    {
        return Err(not_supported(FUNC, "unexpected operand kinds"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    let load0 = format!(
        "%t0_<index> = OpAccessChain %_ptr_Input_float %attr{} %uint_{}",
        inst.src[1].constant.u, inst.src[2].constant.u
    );

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
         <load0>
         %t1_<index> = OpLoad %float %t0_<index>
                       OpStore %<dst> %t1_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VInterpMovF32_VdstVsrcAttrChan` (ShaderSpirv.cpp L5349).
fn recompile_vinterp_mov_f32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    _spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VInterpMovF32_VdstVsrcAttrChan";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst)
        || !operand_is_constant(inst.src[0])
        || !operand_is_constant(inst.src[1])
        || !operand_is_constant(inst.src[2])
    {
        return Err(not_supported(FUNC, "unexpected operand kinds"));
    }

    if inst.src[0].constant.u != 2 {
        return Err(not_supported(FUNC, "P0 select != 2"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    let load0 = format!(
        "%t0_<index> = OpAccessChain %_ptr_Input_float %attr{} %uint_{}",
        inst.src[1].constant.u, inst.src[2].constant.u
    );

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
         <load0>
         %t1_<index> = OpLoad %float %t0_<index>
                       OpStore %<dst> %t1_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_V_XXX_F32_VdstVsrc0Vsrc1Vsrc2` (ShaderSpirv.cpp L5381).
/// XXX: Mad, Madak, Madmk, Max3, Min3, Med3, Fma.
fn recompile_v_xxx_f32_vdst_vsrc012(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_V_XXX_F32_VdstVsrc0Vsrc1Vsrc2";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    if dst_value.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "dst is not float"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();
    let mut load2 = String::new();
    if !operand_load_float(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_float(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }
    if !operand_load_float(spirv, inst.src[2], "t2_<index>", &index_str, &mut load2)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check SP_ROUND
    // TODO() check DX10_CLAMP
    // TODO() check IEEE

    const TEXT: &str = r#"
              <load0>
              <load1>
              <load2>
              <param0>
              <param1>
              <param2>
              <param3>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
               OpSelectionMerge %tl2_<index> None
               OpBranchConditional %exec_lo_b_<index> %tl1_<index> %tl2_<index>
         %tl1_<index> = OpLabel
               OpStore %<dst> %t_<index>
              <multiply>
              <clamp>
               OpBranch %tl2_<index>
         %tl2_<index> = OpLabel
"#;
    *dst_source += &TEXT
        .replace(
            "<multiply>",
            &if inst.dst.multiplier != 1.0 {
                MULTIPLY.replace("<mul>", &spirv.get_constant_float(inst.dst.multiplier))
            } else {
                String::new()
            },
        )
        .replace("<clamp>", if inst.dst.clamp { CLAMP } else { "" })
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<load2>", &load2)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<param2>", param[2].unwrap_or(""))
        .replace("<param3>", param[3].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VMovB32_SVdstSVsrc0` (ShaderSpirv.cpp L5579).
fn recompile_vmov_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VMovB32_SVdstSVsrc0";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    if dst_value.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "dst is not float"));
    }

    let mut load0 = String::new();
    if !operand_load_float(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }

    // TODO() check VSKIP

    const TEXT: &str = r#"
    <load0>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %t0_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_V_XXX_F32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp L5615).
/// XXX: Mac, Max, Min, Mul, Sub, Subrev, Add.
fn recompile_v_xxx_f32_svdst_svsrc01(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_V_XXX_F32_SVdstSVsrc0SVsrc1";
    let inst = inst_at(code, index, FUNC)?;

    let param0 = param[0].unwrap_or("");
    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    if dst_value.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "dst is not float"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();
    let mut load_dst = String::new();
    if !operand_load_float(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_float(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }
    if param0.contains("tdst_<index>")
        && !operand_load_float(spirv, inst.dst, "tdst_<index>", &index_str, &mut load_dst)?
    {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check SP_DENORM
    // TODO() check SP_ROUND
    // TODO() check DX10_CLAMP
    // TODO() check IEEE

    const TEXT: &str = r#"
              <load0>
              <load1>
              <load_dst>
              <param>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
               OpSelectionMerge %tl2_<index> None
               OpBranchConditional %exec_lo_b_<index> %tl1_<index> %tl2_<index>
         %tl1_<index> = OpLabel
               OpStore %<dst> %t_<index>
              <multiply>
              <clamp>
               OpBranch %tl2_<index>
         %tl2_<index> = OpLabel
"#;
    *dst_source += &TEXT
        .replace(
            "<multiply>",
            &if inst.dst.multiplier != 1.0 {
                MULTIPLY.replace("<mul>", &spirv.get_constant_float(inst.dst.multiplier))
            } else {
                String::new()
            },
        )
        .replace("<clamp>", if inst.dst.clamp { CLAMP } else { "" })
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<load_dst>", &load_dst)
        .replace("<param>", param0)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_V_XXX_F32_SVdstSVsrc0` (ShaderSpirv.cpp L5684).
/// XXX: Rcp, Rsq, Sqrt, Ceil, Floor, Fract, Rndne, Trunc, Exp, Log, Cos,
/// Sin.
fn recompile_v_xxx_f32_svdst_svsrc0(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_V_XXX_F32_SVdstSVsrc0";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    if dst_value.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "dst is not float"));
    }

    let mut load0 = String::new();
    if !operand_load_float(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check DX10_CLAMP
    // TODO() check IEEE

    const TEXT: &str = r#"
    <load0>
    <param0>
    <param1>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
               OpSelectionMerge %tl2_<index> None
               OpBranchConditional %exec_lo_b_<index> %tl1_<index> %tl2_<index>
         %tl1_<index> = OpLabel
               OpStore %<dst> %t_<index>
              <multiply>
              <clamp>
               OpBranch %tl2_<index>
         %tl2_<index> = OpLabel
"#;
    *dst_source += &TEXT
        .replace(
            "<multiply>",
            &if inst.dst.multiplier != 1.0 {
                MULTIPLY.replace("<mul>", &spirv.get_constant_float(inst.dst.multiplier))
            } else {
                String::new()
            },
        )
        .replace("<clamp>", if inst.dst.clamp { CLAMP } else { "" })
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VCvt_XXX_F32_SVdstSVsrc0` (ShaderSpirv.cpp L5849).
/// XXX: U32.
fn recompile_vcvt_xxx_f32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCvt_XXX_F32_SVdstSVsrc0";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }
    if inst.dst.clamp {
        return Err(not_supported(FUNC, "clamp"));
    }
    if inst.dst.multiplier != 1.0 {
        return Err(not_supported(FUNC, "multiplier"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    if dst_value.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "dst is not float"));
    }

    let mut load0 = String::new();
    if !operand_load_float(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check EXEC
    // TODO() check SP_DENORM_IN

    const TEXT: &str = r#"
    <load0>
    <param0>
    <param1>
    <param2>
    %t_<index> = OpBitcast %float %t2_<index>
    OpStore %<dst> %t_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<param2>", param[2].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VCvtF32_XXX_SVdstSVsrc0` (ShaderSpirv.cpp L5894).
/// XXX: U32, I32, UbyteX, F16.
fn recompile_vcvt_f32_xxx(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCvtF32_XXX_SVdstSVsrc0";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }
    if inst.dst.clamp {
        return Err(not_supported(FUNC, "clamp"));
    }
    if inst.dst.multiplier != 1.0 {
        return Err(not_supported(FUNC, "multiplier"));
    }

    let dst_value = operand_variable_to_str(inst.dst);

    if dst_value.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "dst is not float"));
    }

    let mut load0 = String::new();
    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check SP_ROUND

    const TEXT: &str = r#"
    <load0>
    <param0>
    <param1>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %t_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_Fetch` (ShaderSpirv.cpp L6065) — the embedded
/// fetch-shader `Fetch*` pseudo-instructions produced by `DetectFetch`.
fn recompile_fetch(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Fetch";
    let inst = inst_at(code, index, FUNC)?;

    let input_info = spirv.get_vs_input_info();

    let Some(info) = input_info else {
        return Err(not_supported(FUNC, "no vs input info"));
    };
    if !info.fetch_embedded {
        return Err(not_supported(FUNC, "!fetch_embedded"));
    }

    if inst.dst.type_ == ShaderOperandType::Vgpr
        && inst.src[2].type_ == ShaderOperandType::IntegerInlineConstant
    {
        let attrib_id = inst.src[2].constant.i();

        let Some(r) = usize::try_from(attrib_id)
            .ok()
            .and_then(|a| info.resources_dst.get(a))
            .copied()
        else {
            return Err(not_supported(FUNC, format!("attrib id {attrib_id}")));
        };

        if r.registers_num != inst.dst.size {
            return Err(not_supported(FUNC, "registers_num != dst.size"));
        }

        let text = match r.registers_num {
            1 => {
                r#"
				         %t1_<index> = OpLoad %float %<attr>
				                       OpStore %temp_float %t1_<index>
				         %t2_<index> = OpFunctionCall %void %fetch_f1_f1_ %<p0> %temp_float
				"#
            }
            2 => {
                r#"
				         %t1_<index> = OpLoad %v2float %<attr>
				                       OpStore %temp_v2float %t1_<index>
				         %t2_<index> = OpFunctionCall %void %fetch_f1_f1_vf2_ %<p0> %<p1> %temp_v2float
				"#
            }
            3 => {
                r#"
				         %t1_<index> = OpLoad %v3float %<attr>
				                       OpStore %temp_v3float %t1_<index>
				         %t2_<index> = OpFunctionCall %void %fetch_f1_f1_f1_vf3_ %<p0> %<p1> %<p2> %temp_v3float
				"#
            }
            4 => {
                r#"
				         %t1_<index> = OpLoad %v4float %<attr>
				                       OpStore %temp_v4float %t1_<index>
				         %t2_<index> = OpFunctionCall %void %fetch_f1_f1_f1_f1_vf4_ %<p0> %<p1> %<p2> %<p3> %temp_v4float
				"#
            }
            n => {
                return Err(not_supported(FUNC, format!("invalid registers_num: {n}")));
            }
        };

        *dst_source += &text
            .replace("<index>", &format!("{attrib_id}_{index}"))
            .replace("<p0>", &format!("v{}", inst.dst.register_id))
            .replace("<p1>", &format!("v{}", inst.dst.register_id + 1))
            .replace("<p2>", &format!("v{}", inst.dst.register_id + 2))
            .replace("<p3>", &format!("v{}", inst.dst.register_id + 3))
            .replace("<attr>", &format!("attr{attrib_id}"));

        return Ok(true);
    }

    Ok(false)
}

// ---------------------------------------------------------------------------
// Dispatch table (Kyty: g_recomp_func, ShaderSpirv.cpp L6184-6446)
// ---------------------------------------------------------------------------

/// Either a ported recompile function or a C2 placeholder that carries the
/// Kyty function name + `ShaderSpirv.cpp` line anchor.
#[derive(Copy, Clone)]
pub enum RecompileFn {
    Func(InstRecompileFn),
    NotImplemented { kyty_func: &'static str, line: u32 },
}

/// Kyty: ShaderSpirv.cpp `RecompilerFunc` (L1555).
pub struct RecompilerFunc {
    pub func: RecompileFn,
    pub type_: ShaderInstructionType,
    pub format: Format,
    pub param: [Option<&'static str>; 4],
    pub scc_check: SccCheck,
}

const fn p1(a: &'static str) -> Params {
    [Some(a), None, None, None]
}

const fn p2(a: &'static str, b: &'static str) -> Params {
    [Some(a), Some(b), None, None]
}

const fn p3(a: &'static str, b: &'static str, c: &'static str) -> Params {
    [Some(a), Some(b), Some(c), None]
}

const fn p4(a: &'static str, b: &'static str, c: &'static str, d: &'static str) -> Params {
    [Some(a), Some(b), Some(c), Some(d)]
}

/// Table row backed by a ported function (scc_check = None).
const fn f(
    func: InstRecompileFn,
    type_: ShaderInstructionType,
    format: Format,
    param: Params,
) -> RecompilerFunc {
    RecompilerFunc {
        func: RecompileFn::Func(func),
        type_,
        format,
        param,
        scc_check: SccCheck::None,
    }
}

/// Table row whose Kyty function is not ported yet (scc_check = None).
const fn ni(
    kyty_func: &'static str,
    line: u32,
    type_: ShaderInstructionType,
    format: Format,
    param: Params,
) -> RecompilerFunc {
    RecompilerFunc {
        func: RecompileFn::NotImplemented { kyty_func, line },
        type_,
        format,
        param,
        scc_check: SccCheck::None,
    }
}

/// Table row whose Kyty function is not ported yet, with an SCC check.
const fn nis(
    kyty_func: &'static str,
    line: u32,
    type_: ShaderInstructionType,
    format: Format,
    param: Params,
    scc_check: SccCheck,
) -> RecompilerFunc {
    RecompilerFunc {
        func: RecompileFn::NotImplemented { kyty_func, line },
        type_,
        format,
        param,
        scc_check,
    }
}

use self::SccCheck as S;
use crate::shader::types::ShaderInstructionType as T;
use crate::shader::types::shader_instruction_format::Format as F;

/// Kyty: ShaderSpirv.cpp `g_recomp_func` (L6184) — all 204 rows, in Kyty
/// order. `param` strings are verbatim SPIR-V template fragments.
#[rustfmt::skip]
static G_RECOMP_FUNC: &[RecompilerFunc] = &[
    f(recompile_buffer_load_dword_vdata1,    T::BufferLoadDword,   F::Vdata1VaddrSvSoffsIdxen, p1("")),
    f(recompile_buffer_load_format_x_vdata1, T::BufferLoadFormatX, F::Vdata1VaddrSvSoffsIdxen, p1("")),
    ni("Recompile_BufferStoreDword_Vdata1VaddrSvSoffsIdxen",    1999, T::BufferStoreDword,    F::Vdata1VaddrSvSoffsIdxen, p1("")),
    ni("Recompile_BufferStoreFormatX_Vdata1VaddrSvSoffsIdxen",  2068, T::BufferStoreFormatX,  F::Vdata1VaddrSvSoffsIdxen, p1("")),
    ni("Recompile_BufferStoreFormatXy_Vdata2VaddrSvSoffsIdxen", 2137, T::BufferStoreFormatXy, F::Vdata2VaddrSvSoffsIdxen, p1("")),

    f(recompile_fetch, T::FetchX,    F::Vdata1VaddrSvSoffsIdxen, p1("")),
    f(recompile_fetch, T::FetchXy,   F::Vdata2VaddrSvSoffsIdxen, p1("")),
    f(recompile_fetch, T::FetchXyz,  F::Vdata3VaddrSvSoffsIdxen, p1("")),
    f(recompile_fetch, T::FetchXyzw, F::Vdata4VaddrSvSoffsIdxen, p1("")),

    ni("Recompile_DsAppend_VdstGds",  2208, T::DsAppend,  F::VdstGds, p1("")),
    ni("Recompile_DsConsume_VdstGds", 2243, T::DsConsume, F::VdstGds, p1("")),

    f(recompile_exp_mrt0_off_off_compr_vm_done,     T::Exp, F::Mrt0OffOffComprVmDone,          p1("")),
    f(recompile_exp_mrt0_vsrc0_vsrc1_compr_vm_done, T::Exp, F::Mrt0Vsrc0Vsrc1ComprVmDone,      p1("")),
    f(recompile_exp_mrt0_vsrc0123_vm_done,          T::Exp, F::Mrt0Vsrc0Vsrc1Vsrc2Vsrc3VmDone, p1("")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param0Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param0")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param1Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param1")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param2Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param2")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param3Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param3")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param4Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param4")),
    f(recompile_exp_pos0,                           T::Exp, F::Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done,   p1("")),
    f(recompile_exp_prim,                           T::Exp, F::PrimVsrc0OffOffOffDone,         p1("")),

    ni("Recompile_ImageLoad_Vdata4Vaddr3StDmaskF",        3038, T::ImageLoad,      F::Vdata4Vaddr3StDmaskF,   p1("")),
    ni("Recompile_ImageSample_Vdata1Vaddr3StSsDmask1",    2471, T::ImageSample,    F::Vdata1Vaddr3StSsDmask1, p1("")),
    ni("Recompile_ImageSample_Vdata1Vaddr3StSsDmask8",    2525, T::ImageSample,    F::Vdata1Vaddr3StSsDmask8, p1("")),
    ni("Recompile_ImageSample_Vdata2Vaddr3StSsDmask3",    2579, T::ImageSample,    F::Vdata2Vaddr3StSsDmask3, p1("")),
    ni("Recompile_ImageSample_Vdata2Vaddr3StSsDmask5",    2638, T::ImageSample,    F::Vdata2Vaddr3StSsDmask5, p1("")),
    ni("Recompile_ImageSample_Vdata2Vaddr3StSsDmask9",    2697, T::ImageSample,    F::Vdata2Vaddr3StSsDmask9, p1("")),
    ni("Recompile_ImageSample_Vdata3Vaddr3StSsDmask7",    2756, T::ImageSample,    F::Vdata3Vaddr3StSsDmask7, p1("")),
    ni("Recompile_ImageSample_Vdata4Vaddr3StSsDmaskF",    2968, T::ImageSample,    F::Vdata4Vaddr3StSsDmaskF, p1("")),
    ni("Recompile_ImageSampleLz_Vdata3Vaddr3StSsDmask7",  2821, T::ImageSampleLz,  F::Vdata3Vaddr3StSsDmask7, p1("")),
    ni("Recompile_ImageSampleLzO_Vdata3Vaddr4StSsDmask7", 2887, T::ImageSampleLzO, F::Vdata3Vaddr4StSsDmask7, p1("")),
    ni("Recompile_ImageStore_Vdata4Vaddr3StDmaskF",       3105, T::ImageStore,     F::Vdata4Vaddr3StDmaskF,   p1("")),
    ni("Recompile_ImageStoreMip_Vdata4Vaddr4StDmaskF",    3173, T::ImageStoreMip,  F::Vdata4Vaddr4StDmaskF,   p1("")),

    f(recompile_sbuffer_load_dword,   T::SBufferLoadDword,   F::SdstSvSoffset,  p1("")),
    ni("Recompile_SBufferLoadDwordx2_Sdst2SvSoffset",   3831, T::SBufferLoadDwordx2,  F::Sdst2SvSoffset,  p1("")),
    f(recompile_sbuffer_load_dwordx4, T::SBufferLoadDwordx4, F::Sdst4SvSoffset, p1("")),
    ni("Recompile_SBufferLoadDwordx8_Sdst8SvSoffset",   3928, T::SBufferLoadDwordx8,  F::Sdst8SvSoffset,  p1("")),
    ni("Recompile_SBufferLoadDwordx16_Sdst16SvSoffset", 3976, T::SBufferLoadDwordx16, F::Sdst16SvSoffset, p1("")),

    f(recompile_scbranch_xxx_label, T::SCbranchExecz, F::Label, p2("%cc_u_<index> = OpLoad %uint %execz",  "%cc_b_<index> = OpINotEqual %bool %cc_u_<index> %uint_0")),
    f(recompile_scbranch_xxx_label, T::SCbranchScc0,  F::Label, p2("%cc_u_<index> = OpLoad %uint %scc",    "%cc_b_<index> = OpIEqual    %bool %cc_u_<index> %uint_0")),
    f(recompile_scbranch_xxx_label, T::SCbranchScc1,  F::Label, p2("%cc_u_<index> = OpLoad %uint %scc",    "%cc_b_<index> = OpIEqual    %bool %cc_u_<index> %uint_1")),
    f(recompile_scbranch_xxx_label, T::SCbranchVccz,  F::Label, p2("%cc_u_<index> = OpLoad %uint %vcc_lo", "%cc_b_<index> = OpIEqual    %bool %cc_u_<index> %uint_0")),
    f(recompile_scbranch_xxx_label, T::SCbranchVccnz, F::Label, p2("%cc_u_<index> = OpLoad %uint %vcc_lo", "%cc_b_<index> = OpINotEqual %bool %cc_u_<index> %uint_0")),
    f(recompile_sbranch_label,      T::SBranch,       F::Label, p1("")),

    f(recompile_sendpgm_empty, T::SEndpgm, F::Empty, p1("")),

    f(recompile_sload_dword,   T::SLoadDword,   F::SdstSbaseSoffset,  p1("")),
    f(recompile_sload_dwordx2, T::SLoadDwordx2, F::Sdst2Ssrc02Ssrc1,  p1("")),
    f(recompile_sload_dwordx4, T::SLoadDwordx4, F::Sdst4SbaseSoffset, p1("")),
    f(recompile_sload_dwordx8, T::SLoadDwordx8, F::Sdst8SbaseSoffset, p1("")),

    nis("Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12", 3248, T::SAndn2B64,   F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpNot %uint %t2_<index>",
        "%tb_<index> = OpBitwiseAnd %uint %t0_<index> %ta_<index>",
        "%tc_<index> = OpNot %uint %t3_<index>",
        "%td_<index> = OpBitwiseAnd %uint %t1_<index> %tc_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12", 3248, T::SOrn2B64,    F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpNot %uint %t2_<index>",
        "%tb_<index> = OpBitwiseOr %uint %t0_<index> %ta_<index>",
        "%tc_<index> = OpNot %uint %t3_<index>",
        "%td_<index> = OpBitwiseOr %uint %t1_<index> %tc_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12", 3248, T::SAndB64,     F::Sdst2Ssrc02Ssrc12, p2("%tb_<index> = OpBitwiseAnd %uint %t0_<index> %t2_<index>",
        "%td_<index> = OpBitwiseAnd %uint %t1_<index> %t3_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12", 3248, T::SNorB64,     F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpBitwiseOr %uint %t0_<index> %t2_<index>",
        "%tb_<index> = OpNot %uint %ta_<index>",
        "%tc_<index> = OpBitwiseOr %uint %t1_<index> %t3_<index>",
        "%td_<index> = OpNot %uint %tc_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12", 3248, T::SNandB64,    F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpBitwiseAnd %uint %t0_<index> %t2_<index>",
        "%tb_<index> = OpNot %uint %ta_<index>",
        "%tc_<index> = OpBitwiseAnd %uint %t1_<index> %t3_<index>",
        "%td_<index> = OpNot %uint %tc_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12", 3248, T::SXnorB64,    F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpBitwiseXor %uint %t0_<index> %t2_<index>",
        "%tb_<index> = OpNot %uint %ta_<index>",
        "%tc_<index> = OpBitwiseXor %uint %t1_<index> %t3_<index>",
        "%td_<index> = OpNot %uint %tc_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12", 3248, T::SOrB64,      F::Sdst2Ssrc02Ssrc12, p2("%tb_<index> = OpBitwiseOr %uint %t0_<index> %t2_<index>",
        "%td_<index> = OpBitwiseOr %uint %t1_<index> %t3_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12", 3248, T::SXorB64,     F::Sdst2Ssrc02Ssrc12, p2("%tb_<index> = OpBitwiseXor %uint %t0_<index> %t2_<index>",
        "%td_<index> = OpBitwiseXor %uint %t1_<index> %t3_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12", 3248, T::SCselectB64, F::Sdst2Ssrc02Ssrc12, p4("%ts_<index> = OpLoad %uint %scc",
        "%tsb_<index> = OpINotEqual %bool %ts_<index> %uint_0",
        "%tb_<index> = OpSelect %uint %tsb_<index> %t0_<index> %t2_<index>",
        "%td_<index> = OpSelect %uint %tsb_<index> %t1_<index> %t3_<index>"), S::None),

    nis("Recompile_S_Bfe_U64_Sdst2Ssrc02Ssrc1",  3452, T::SBfeU64,  F::Sdst2Ssrc02Ssrc1, p2("", ""), S::NonZero),
    nis("Recompile_S_Lshl_B64_Sdst2Ssrc02Ssrc1", 3316, T::SLshlB64, F::Sdst2Ssrc02Ssrc1, p2("", ""), S::NonZero),
    nis("Recompile_S_Lshr_B64_Sdst2Ssrc02Ssrc1", 3384, T::SLshrB64, F::Sdst2Ssrc02Ssrc1, p2("", ""), S::NonZero),

    nis("Recompile_S_XXX_B32_SVdstSVsrc0SVsrc1", 3528, T::SAndB32,      F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseAnd %uint %t0_<index> %t1_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B32_SVdstSVsrc0SVsrc1", 3528, T::SBfmB32,      F::SVdstSVsrc0SVsrc1, p3("%tcount_<index> = OpBitwiseAnd %uint %t0_<index> %uint_31", "%toffset_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpBitFieldInsert %uint %uint_0 %uint_0xffffffff %toffset_<index> %tcount_<index>"), S::None),
    nis("Recompile_S_XXX_B32_SVdstSVsrc0SVsrc1", 3528, T::SCselectB32,  F::SVdstSVsrc0SVsrc1, p3("%t22_<index> = OpLoad %uint %scc", "%t2_<index> = OpINotEqual %bool %t22_<index> %uint_0", "%t_<index> = OpSelect %uint %t2_<index> %t0_<index> %t1_<index>"), S::None),
    nis("Recompile_S_XXX_B32_SVdstSVsrc0SVsrc1", 3528, T::SLshlB32,     F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpShiftLeftLogical %uint %t0_<index> %ts_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B32_SVdstSVsrc0SVsrc1", 3528, T::SLshrB32,     F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpShiftRightLogical %uint %t0_<index> %ts_<index>"), S::NonZero),
    nis("Recompile_S_XXX_B32_SVdstSVsrc0SVsrc1", 3528, T::SOrB32,       F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseOr %uint %t0_<index> %t1_<index>"), S::NonZero),
    nis("Recompile_S_XXX_I32_SVdstSVsrc0SVsrc1", 3576, T::SAddI32,      F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpIAdd %int %t0_<index> %t1_<index>"), S::OverflowAdd),
    nis("Recompile_S_XXX_I32_SVdstSVsrc0SVsrc1", 3576, T::SMulI32,      F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpIMul %int %t0_<index> %t1_<index>"), S::None),
    nis("Recompile_S_XXX_I32_SVdstSVsrc0SVsrc1", 3576, T::SSubI32,      F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpISub %int %t0_<index> %t1_<index>"), S::OverflowSub),
    nis("Recompile_S_XXX_U32_SVdstSVsrc0SVsrc1", 3621, T::SAddcU32,     F::SVdstSVsrc0SVsrc1, p4("%tscc_<index> = OpLoad %uint %scc", "%ts_<index> = OpFunctionCall %v2uint %addc %t0_<index> %t1_<index> %tscc_<index>", "%t_<index> = OpCompositeExtract %uint %ts_<index> 0", "%carry_<index> = OpCompositeExtract %uint %ts_<index> 1"), S::CarryOut),
    nis("Recompile_S_XXX_U32_SVdstSVsrc0SVsrc1", 3621, T::SAddU32,      F::SVdstSVsrc0SVsrc1, p3("%ts_<index> = OpIAddCarry %ResTypeU %t0_<index> %t1_<index>", "%t_<index> = OpCompositeExtract %uint %ts_<index> 0", "%carry_<index> = OpCompositeExtract %uint %ts_<index> 1"), S::CarryOut),
    nis("Recompile_S_XXX_U32_SVdstSVsrc0SVsrc1", 3621, T::SBfeU32,      F::SVdstSVsrc0SVsrc1, p3("%to_<index> = OpBitFieldUExtract %uint %t1_<index> %uint_0  %uint_5", "%ts_<index> = OpBitFieldUExtract %uint %t1_<index> %uint_16 %uint_7", "%t_<index> = OpBitFieldUExtract %uint %t0_<index> %to_<index> %ts_<index>"), S::NonZero),
    nis("Recompile_S_XXX_U32_SVdstSVsrc0SVsrc1", 3621, T::SLshl4AddU32, F::SVdstSVsrc0SVsrc1, p3("%ts_<index> = OpFunctionCall %v2uint %lshl_add %t0_<index> %t1_<index> %uint_4", "%t_<index> = OpCompositeExtract %uint %ts_<index> 0", "%carry_<index> = OpCompositeExtract %uint %ts_<index> 1"), S::CarryOut),
    nis("Recompile_S_XXX_U32_SVdstSVsrc0SVsrc1", 3621, T::SMulHiU32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFunctionCall %uint %mul_hi_uint %t0_<index> %t1_<index>"), S::None),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VAndB32,     F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseAnd %uint %t0_<index> %t1_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VBcntU32B32, F::SVdstSVsrc0SVsrc1, p3("%tb_<index> = OpBitCount %int %t0_<index>", "%tbu_<index> = OpBitcast %uint %tb_<index>", "%t_<index> = OpIAdd %uint %tbu_<index> %t1_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VBfmB32,     F::SVdstSVsrc0SVsrc1, p3("%tcount_<index> = OpBitwiseAnd %uint %t0_<index> %uint_31", "%toffset_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpBitFieldInsert %uint %uint_0 %uint_0xffffffff %toffset_<index> %tcount_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VLshlB32,    F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpShiftLeftLogical %uint %t0_<index> %ts_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VLshlrevB32, F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t0_<index> %uint_31", "%t_<index> = OpShiftLeftLogical %uint %t1_<index> %ts_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VLshrB32,    F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpShiftRightLogical %uint %t0_<index> %ts_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VLshrrevB32, F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t0_<index> %uint_31", "%t_<index> = OpShiftRightLogical %uint %t1_<index> %ts_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VMulHiU32,   F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFunctionCall %uint %mul_hi_uint %t0_<index> %t1_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VMulLoU32,   F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFunctionCall %uint %mul_lo_uint %t0_<index> %t1_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VMulU32U24,  F::SVdstSVsrc0SVsrc1, p3("%tu0_<index> = OpBitwiseAnd %uint %t0_<index> %uint_0x00ffffff", "%tu1_<index> = OpBitwiseAnd %uint %t1_<index> %uint_0x00ffffff", "%t_<index> = OpFunctionCall %uint %mul_lo_uint %tu0_<index> %tu1_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VOrB32,      F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseOr %uint %t0_<index> %t1_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1", 5740, T::VXorB32,     F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseXor %uint %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VAddF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFAdd %float %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMacF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Fma %t0_<index> %t1_<index> %tdst_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMaxF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %float %GLSL_std_450 FMax %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMinF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %float %GLSL_std_450 FMin %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMulF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFMul %float %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VSubF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFSub %float %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VSubrevF32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFSub %float %t1_<index> %t0_<index>")),
    ni("Recompile_V_XXX_I32_SVdstSVsrc0SVsrc1", 5795, T::VAshrI32,    F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %int %t1_<index> %int_31", "%t_<index> = OpShiftRightArithmetic %int %t0_<index> %ts_<index>")),
    ni("Recompile_V_XXX_I32_SVdstSVsrc0SVsrc1", 5795, T::VAshrrevI32, F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %int %t0_<index> %int_31", "%t_<index> = OpShiftRightArithmetic %int %t1_<index> %ts_<index>")),
    ni("Recompile_V_XXX_I32_SVdstSVsrc0SVsrc1", 5795, T::VMulLoI32,   F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFunctionCall %int %mul_lo_int %t0_<index> %t1_<index>")),
    ni("Recompile_VCvtPkrtzF16F32_SVdstSVsrc0SVsrc1", 5260, T::VCvtPkrtzF16F32, F::SVdstSVsrc0SVsrc1, p1("")),
    ni("Recompile_VMbcntHiU32B32_SVdstSVsrc0SVsrc1",  5455, T::VMbcntHiU32B32,  F::SVdstSVsrc0SVsrc1, p1("")),
    ni("Recompile_VMbcntLoU32B32_SVdstSVsrc0SVsrc1",  5497, T::VMbcntLoU32B32,  F::SVdstSVsrc0SVsrc1, p1("")),

    f(recompile_smov_b32, T::SMovB32,  F::SVdstSVsrc0, p1("")),
    f(recompile_smov_b32, T::SMovkI32, F::SVdstSVsrc0, p1("")),
    ni("Recompile_SMulkI32_SVdstSVsrc0", 4437, T::SMulkI32, F::SVdstSVsrc0, p1("")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0", 5538, T::VBfrevB32, F::SVdstSVsrc0, p1("%t_<index> = OpBitReverse %uint %t0_<index>")),
    ni("Recompile_V_XXX_B32_SVdstSVsrc0", 5538, T::VNotB32,   F::SVdstSVsrc0, p1("%t_<index> = OpNot %uint %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VCeilF32,  F::SVdstSVsrc0, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Ceil %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VCosF32,   F::SVdstSVsrc0, p2("%tr_<index> = OpFMul %float %t0_<index> %float_2pi", "%t_<index> = OpExtInst %float %GLSL_std_450 Cos %tr_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VExpF32,   F::SVdstSVsrc0, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Exp2 %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VFloorF32, F::SVdstSVsrc0, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Floor %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VFractF32, F::SVdstSVsrc0, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Fract %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VLogF32,   F::SVdstSVsrc0, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Log2 %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VRcpF32,   F::SVdstSVsrc0, p1("%t_<index> = OpFDiv %float %float_1_000000 %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VRndneF32, F::SVdstSVsrc0, p1("%t_<index> = OpExtInst %float %GLSL_std_450 RoundEven %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VRsqF32,   F::SVdstSVsrc0, p1("%t_<index> = OpExtInst %float %GLSL_std_450 InverseSqrt %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VSinF32,   F::SVdstSVsrc0, p2("%tr_<index> = OpFMul %float %t0_<index> %float_2pi", "%t_<index> = OpExtInst %float %GLSL_std_450 Sin %tr_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VSqrtF32,  F::SVdstSVsrc0, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Sqrt %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VTruncF32, F::SVdstSVsrc0, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Trunc %t0_<index>")),
    f(recompile_vcvt_xxx_f32, T::VCvtU32F32, F::SVdstSVsrc0, p2("%t1_<index> = OpExtInst %float %GLSL_std_450 Trunc %t0_<index>", "%t2_<index> = OpConvertFToU %uint %t1_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32F16,    F::SVdstSVsrc0, p2("%ts_<index> = OpExtInst %v2float %GLSL_std_450 UnpackHalf2x16 %t0_<index>", "%t_<index> = OpCompositeExtract %float %ts_<index> 0")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32I32,    F::SVdstSVsrc0, p2("%ti_<index> = OpBitcast %int %t0_<index>", "%t_<index> = OpConvertSToF %float %ti_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32U32,    F::SVdstSVsrc0, p1("%t_<index> = OpConvertUToF %float %t0_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32Ubyte0, F::SVdstSVsrc0, p2("%tb_<index> = OpBitFieldUExtract %uint %t0_<index> %uint_0 %uint_8", "%t_<index> = OpConvertUToF %float %tb_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32Ubyte1, F::SVdstSVsrc0, p2("%tb_<index> = OpBitFieldUExtract %uint %t0_<index> %uint_8 %uint_8", "%t_<index> = OpConvertUToF %float %tb_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32Ubyte2, F::SVdstSVsrc0, p2("%tb_<index> = OpBitFieldUExtract %uint %t0_<index> %uint_16 %uint_8", "%t_<index> = OpConvertUToF %float %tb_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32Ubyte3, F::SVdstSVsrc0, p2("%tb_<index> = OpBitFieldUExtract %uint %t0_<index> %uint_24 %uint_8", "%t_<index> = OpConvertUToF %float %tb_<index>")),
    f(recompile_vmov_b32, T::VMovB32, F::SVdstSVsrc0, p1("")),

    nis("Recompile_SAndSaveexecB64_Sdst2Ssrc02", 3670, T::SAndSaveexecB64, F::Sdst2Ssrc02, p1(""), S::NonZero),
    f(recompile_smov_b64,    T::SMovB64,    F::Sdst2Ssrc02, p1("")),
    f(recompile_sswappc_b64, T::SSwappcB64, F::Sdst2Ssrc02, p1("")),
    nis("Recompile_SWqmB64_Sdst2Ssrc02", 4621, T::SWqmB64, F::Sdst2Ssrc02, p1(""), S::NonZero),

    f(recompile_skip, T::SInstPrefetch, F::Imm, p1("")),
    f(recompile_skip, T::SSendmsg,      F::Imm, p1("")),
    f(recompile_skip, T::SWaitcnt,      F::Imm, p1("")),

    f(recompile_tbuffer_load_format_x_float1, T::TBufferLoadFormatX, F::Vdata1VaddrSvSoffsIdxenFloat1, p1("")),
    ni("Recompile_TBufferLoadFormatXyzw_Vdata4Vaddr2SvSoffsOffenIdxenFloat4", 4824, T::TBufferLoadFormatXyzw, F::Vdata4Vaddr2SvSoffsOffenIdxenFloat4, p1("")),
    f(recompile_tbuffer_load_format_xyzw_float4, T::TBufferLoadFormatXyzw, F::Vdata4VaddrSvSoffsIdxenFloat4, p1("")),

    ni("Recompile_V_XXX_U32_VdstSdst2Vsrc0Vsrc1", 6005, T::VAddI32,    F::VdstSdst2Vsrc0Vsrc1, p1("%t_<index> = OpIAddCarry %ResTypeU %t0_<index> %t1_<index>")),
    ni("Recompile_V_XXX_U32_VdstSdst2Vsrc0Vsrc1", 6005, T::VSubI32,    F::VdstSdst2Vsrc0Vsrc1, p1("%t_<index> = OpISubBorrow %ResTypeU %t0_<index> %t1_<index>")),
    ni("Recompile_V_XXX_U32_VdstSdst2Vsrc0Vsrc1", 6005, T::VSubrevI32, F::VdstSdst2Vsrc0Vsrc1, p1("%t_<index> = OpISubBorrow %ResTypeU %t1_<index> %t0_<index>")),

    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpEqF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdEqual")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpFF32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_1 ; ")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpGeF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdGreaterThanEqual")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpGtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdGreaterThan")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpLeF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdLessThanEqual")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpLgF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdNotEqual")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpLtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdLessThan")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpNeqF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordNotEqual")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpNgeF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordLessThan")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpNgtF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordLessThanEqual")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpNleF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordGreaterThan")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpNlgF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordEqual")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpNltF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordGreaterThanEqual")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpOF32,   F::SmaskVsrc0Vsrc1, p1("OpFunctionCall %bool %ordered %t0_<index> %t1_<index> ; ")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpTruF32, F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_0 ; ")),
    ni("Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1", 4890, T::VCmpUF32,   F::SmaskVsrc0Vsrc1, p1("OpFunctionCall %bool %unordered %t0_<index> %t1_<index> ; ")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpEqI32,  F::SmaskVsrc0Vsrc1, p1("OpIEqual")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpEqU32,  F::SmaskVsrc0Vsrc1, p1("OpIEqual")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpFI32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_1 ; ")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpGeI32,  F::SmaskVsrc0Vsrc1, p1("OpSGreaterThanEqual")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpGtI32,  F::SmaskVsrc0Vsrc1, p1("OpSGreaterThan")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpLeI32,  F::SmaskVsrc0Vsrc1, p1("OpSLessThanEqual")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpLtI32,  F::SmaskVsrc0Vsrc1, p1("OpSLessThan")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpNeI32,  F::SmaskVsrc0Vsrc1, p1("OpINotEqual")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpNeU32,  F::SmaskVsrc0Vsrc1, p1("OpINotEqual")),
    ni("Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1", 4940, T::VCmpTI32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_0 ; ")),
    ni("Recompile_VCmp_XXX_U32_SmaskVsrc0Vsrc1", 4990, T::VCmpFU32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_1 ; ")),
    ni("Recompile_VCmp_XXX_U32_SmaskVsrc0Vsrc1", 4990, T::VCmpGeU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThanEqual")),
    ni("Recompile_VCmp_XXX_U32_SmaskVsrc0Vsrc1", 4990, T::VCmpGtU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThan")),
    ni("Recompile_VCmp_XXX_U32_SmaskVsrc0Vsrc1", 4990, T::VCmpLeU32,  F::SmaskVsrc0Vsrc1, p1("OpULessThanEqual")),
    ni("Recompile_VCmp_XXX_U32_SmaskVsrc0Vsrc1", 4990, T::VCmpLtU32,  F::SmaskVsrc0Vsrc1, p1("OpULessThan")),
    ni("Recompile_VCmp_XXX_U32_SmaskVsrc0Vsrc1", 4990, T::VCmpTU32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_0 ; ")),
    ni("Recompile_VCmpx_XXX_F32_SmaskVsrc0Vsrc1", 5148, T::VCmpxNeqF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordNotEqual")),
    ni("Recompile_VCmpx_XXX_F32_SmaskVsrc0Vsrc1", 5148, T::VCmpxGtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdGreaterThan")),
    ni("Recompile_VCmpx_XXX_F32_SmaskVsrc0Vsrc1", 5148, T::VCmpxLtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdLessThan")),
    ni("Recompile_VCmpx_XXX_I32_SmaskVsrc0Vsrc1", 5040, T::VCmpxEqU32,  F::SmaskVsrc0Vsrc1, p1("OpIEqual")),
    ni("Recompile_VCmpx_XXX_I32_SmaskVsrc0Vsrc1", 5040, T::VCmpxNeU32,  F::SmaskVsrc0Vsrc1, p1("OpINotEqual")),
    ni("Recompile_VCmpx_XXX_U32_SmaskVsrc0Vsrc1", 5094, T::VCmpxGeU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThanEqual")),
    ni("Recompile_VCmpx_XXX_U32_SmaskVsrc0Vsrc1", 5094, T::VCmpxGtU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThan")),

    ni("Recompile_SCmp_XXX_I32_Ssrc0Ssrc1", 3725, T::SCmpEqI32, F::Ssrc0Ssrc1, p1("OpIEqual")),
    ni("Recompile_SCmp_XXX_I32_Ssrc0Ssrc1", 3725, T::SCmpGeI32, F::Ssrc0Ssrc1, p1("OpSGreaterThanEqual")),
    ni("Recompile_SCmp_XXX_I32_Ssrc0Ssrc1", 3725, T::SCmpGtI32, F::Ssrc0Ssrc1, p1("OpSGreaterThan")),
    ni("Recompile_SCmp_XXX_I32_Ssrc0Ssrc1", 3725, T::SCmpLgI32, F::Ssrc0Ssrc1, p1("OpINotEqual")),
    ni("Recompile_SCmp_XXX_I32_Ssrc0Ssrc1", 3725, T::SCmpLtI32, F::Ssrc0Ssrc1, p1("OpSLessThan")),
    ni("Recompile_SCmp_XXX_I32_Ssrc0Ssrc1", 3725, T::SCmpLeI32, F::Ssrc0Ssrc1, p1("OpSLessThanEqual")),
    ni("Recompile_SCmp_XXX_U32_Ssrc0Ssrc1", 3760, T::SCmpEqU32, F::Ssrc0Ssrc1, p1("OpIEqual")),
    ni("Recompile_SCmp_XXX_U32_Ssrc0Ssrc1", 3760, T::SCmpGeU32, F::Ssrc0Ssrc1, p1("OpUGreaterThanEqual")),
    ni("Recompile_SCmp_XXX_U32_Ssrc0Ssrc1", 3760, T::SCmpGtU32, F::Ssrc0Ssrc1, p1("OpUGreaterThan")),
    ni("Recompile_SCmp_XXX_U32_Ssrc0Ssrc1", 3760, T::SCmpLeU32, F::Ssrc0Ssrc1, p1("OpULessThanEqual")),
    ni("Recompile_SCmp_XXX_U32_Ssrc0Ssrc1", 3760, T::SCmpLtU32, F::Ssrc0Ssrc1, p1("OpULessThan")),
    ni("Recompile_SCmp_XXX_U32_Ssrc0Ssrc1", 3760, T::SCmpLgU32, F::Ssrc0Ssrc1, p1("OpINotEqual")),

    f(recompile_vcndmask_b32, T::VCndmaskB32, F::VdstVsrc0Vsrc1Smask2, p1("")),

    f(recompile_vinterp_mov_f32, T::VInterpMovF32, F::VdstVsrcAttrChan, p1("")),
    f(recompile_vinterp_p1_f32,  T::VInterpP1F32,  F::VdstVsrcAttrChan, p1("")),
    f(recompile_vinterp_p2_f32,  T::VInterpP2F32,  F::VdstVsrcAttrChan, p1("")),

    f(recompile_v_xxx_f32_vdst_vsrc012, T::VMadF32,   F::VdstVsrc0Vsrc1Vsrc2, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Fma %t0_<index> %t1_<index> %t2_<index>")),
    f(recompile_v_xxx_f32_vdst_vsrc012, T::VFmaF32,   F::VdstVsrc0Vsrc1Vsrc2, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Fma %t0_<index> %t1_<index> %t2_<index>")),
    f(recompile_v_xxx_f32_vdst_vsrc012, T::VMadakF32, F::VdstVsrc0Vsrc1Vsrc2, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Fma %t0_<index> %t1_<index> %t2_<index>")),
    f(recompile_v_xxx_f32_vdst_vsrc012, T::VMadmkF32, F::VdstVsrc0Vsrc1Vsrc2, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Fma %t0_<index> %t1_<index> %t2_<index>")),
    f(recompile_v_xxx_f32_vdst_vsrc012, T::VMax3F32,  F::VdstVsrc0Vsrc1Vsrc2, p2("%tm_<index> = OpExtInst %float %GLSL_std_450 FMax %t0_<index> %t1_<index>",
        "%t_<index> = OpExtInst %float %GLSL_std_450 FMax %tm_<index> %t2_<index>")),
    f(recompile_v_xxx_f32_vdst_vsrc012, T::VMin3F32,  F::VdstVsrc0Vsrc1Vsrc2, p2("%tm_<index> = OpExtInst %float %GLSL_std_450 FMin %t0_<index> %t1_<index>",
        "%t_<index> = OpExtInst %float %GLSL_std_450 FMin %tm_<index> %t2_<index>")),
    f(recompile_v_xxx_f32_vdst_vsrc012, T::VMed3F32,  F::VdstVsrc0Vsrc1Vsrc2, p4("%t3_<index> = OpExtInst %float %GLSL_std_450 FMin %t0_<index> %t1_<index>",
        "%t4_<index> = OpExtInst %float %GLSL_std_450 FMax %t0_<index> %t1_<index>",
        "%t5_<index> = OpExtInst %float %GLSL_std_450 FMin %t4_<index> %t2_<index>",
        "%t_<index> = OpExtInst %float %GLSL_std_450 FMax %t3_<index> %t5_<index>")),
    ni("Recompile_V_XXX_U32_VdstVsrc0Vsrc1Vsrc2", 5940, T::VSadU32,    F::VdstVsrc0Vsrc1Vsrc2, p2("%td_<index> = OpFunctionCall %uint %abs_diff %t0_<index> %t1_<index>",
        "%t_<index> = OpIAdd %uint %td_<index> %t2_<index>")),
    ni("Recompile_V_XXX_U32_VdstVsrc0Vsrc1Vsrc2", 5940, T::VBfeU32,    F::VdstVsrc0Vsrc1Vsrc2, p3("%to_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31",
        "%ts_<index> = OpBitwiseAnd %uint %t2_<index> %uint_31",
        "%t_<index> = OpBitFieldUExtract %uint %t0_<index> %to_<index> %ts_<index>")),
    ni("Recompile_V_XXX_U32_VdstVsrc0Vsrc1Vsrc2", 5940, T::VMadU32U24, F::VdstVsrc0Vsrc1Vsrc2, p4("%tu0_<index> = OpBitwiseAnd %uint %t0_<index> %uint_0x00ffffff",
        "%tu1_<index> = OpBitwiseAnd %uint %t1_<index> %uint_0x00ffffff",
        "%tm_<index> = OpFunctionCall %uint %mul_lo_uint %tu0_<index> %tu1_<index>",
        "%t_<index> = OpIAdd %uint %tm_<index> %t2_<index>")),
];

/// Kyty: ShaderSpirv.cpp `RecompFunc` (L6182) — hash-keyed
/// (type, format) -> row lookup.
pub fn recomp_func(
    type_: ShaderInstructionType,
    format: Format,
) -> Option<&'static RecompilerFunc> {
    static MAP: OnceLock<HashMap<(ShaderInstructionType, Format), &'static RecompilerFunc>> =
        OnceLock::new();

    let map = MAP.get_or_init(|| {
        let mut m = HashMap::new();
        for func in G_RECOMP_FUNC {
            let prev = m.insert((func.type_, func.format), func);
            debug_assert!(
                prev.is_none(),
                "duplicate recompiler entry {:?}/{:?}",
                func.type_,
                func.format
            );
        }
        m
    });

    map.get(&(type_, format)).copied()
}

/// The full `g_recomp_func` table (for coverage accounting / C2 planning).
#[must_use]
pub fn recomp_func_table() -> &'static [RecompilerFunc] {
    G_RECOMP_FUNC
}

// ---------------------------------------------------------------------------
// Recompile entry points (Kyty: Shader.cpp)
// ---------------------------------------------------------------------------

/// Kyty: Shader.cpp `SpirvRun` (L845). Deviation: assemble-only (see module
/// doc) — no SPIRV-Tools Validate/Optimize passes.
pub fn spirv_run(source: &str) -> Result<Vec<u32>, ShaderRecompileError> {
    Ok(crate::spirv_asm::assemble(source)?)
}

/// Kyty: Shader.cpp `ShaderRecompileVS` (L2361).
pub fn shader_recompile_vs(
    code: &ShaderCode,
    input_info: &ShaderVertexInputInfo,
) -> Result<Vec<u32>, ShaderRecompileError> {
    let source = if code.is_vs_embedded() {
        spirv_get_embedded_vs(code.get_vs_embedded_id())?.to_string()
    } else {
        for i in 0..input_info.bind.storage_buffers.buffers_num as usize {
            let r = &input_info.bind.storage_buffers.buffers[i];
            if (u32::from(r.stride()).wrapping_mul(r.num_records())) & 0x3 != 0 {
                return Err(not_supported(
                    "ShaderRecompileVS",
                    "buffer stride * num_records is not dword-aligned",
                ));
            }
        }
        spirv_generate_source(code, Some(input_info), None, None)?
    };

    tracing::trace!("recompiled vs source:\n{source}");

    spirv_run(&source)
}

/// Kyty: Shader.cpp `ShaderRecompilePS` (L2461).
pub fn shader_recompile_ps(
    code: &ShaderCode,
    input_info: &ShaderPixelInputInfo,
) -> Result<Vec<u32>, ShaderRecompileError> {
    let source = if code.is_ps_embedded() {
        spirv_get_embedded_ps(code.get_ps_embedded_id())?.to_string()
    } else {
        for i in 0..input_info.bind.storage_buffers.buffers_num as usize {
            let r = &input_info.bind.storage_buffers.buffers[i];
            if (u32::from(r.stride()).wrapping_mul(r.num_records())) & 0x3 != 0 {
                return Err(not_supported(
                    "ShaderRecompilePS",
                    "buffer stride * num_records is not dword-aligned",
                ));
            }
        }
        spirv_generate_source(code, None, Some(input_info), None)?
    };

    tracing::trace!("recompiled ps source:\n{source}");

    spirv_run(&source)
}

/// Kyty: Shader.cpp `ShaderRecompileCS` (L2545).
pub fn shader_recompile_cs(
    code: &ShaderCode,
    input_info: &ShaderComputeInputInfo,
) -> Result<Vec<u32>, ShaderRecompileError> {
    for i in 0..input_info.bind.storage_buffers.buffers_num as usize {
        let r = &input_info.bind.storage_buffers.buffers[i];
        if (u32::from(r.stride()).wrapping_mul(r.num_records())) & 0x3 != 0 {
            return Err(not_supported(
                "ShaderRecompileCS",
                "buffer stride * num_records is not dword-aligned",
            ));
        }
    }

    let source = spirv_generate_source(code, None, None, Some(input_info))?;

    tracing::trace!("recompiled cs source:\n{source}");

    spirv_run(&source)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::parse::shader_parse;
    use crate::shader::types::{ShaderInstruction, ShaderOperand, ShaderType};

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn parse(src: &[u32], type_: ShaderType) -> ShaderCode {
        let mut code = ShaderCode::new();
        code.set_type(type_);
        shader_parse(0, src, &mut code, false).expect("parse failed");
        code
    }

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn naga_parse_and_validate(words: &[u32], name: &str) {
        let bytes = words_to_bytes(words);
        let module =
            naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default())
                .unwrap_or_else(|e| panic!("naga parse of {name} failed: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("naga validate of {name} failed: {e:?}"));
    }

    // ---- 1. dispatch table ------------------------------------------------

    #[test]
    fn dispatch_table_counts() {
        // Kyty: g_recomp_func (ShaderSpirv.cpp L6184) has 204 rows. C1
        // implements the minimal VS/PS subset; C2 flips the NI count to 0.
        let table = recomp_func_table();
        let implemented = table
            .iter()
            .filter(|e| matches!(e.func, RecompileFn::Func(_)))
            .count();
        let ni = table
            .iter()
            .filter(|e| matches!(e.func, RecompileFn::NotImplemented { .. }))
            .count();
        assert_eq!(table.len(), 204, "table must mirror Kyty row-for-row");
        assert_eq!(implemented + ni, table.len());
        assert_eq!(implemented, 77, "C1 implemented subset");
        assert_eq!(ni, 127, "C2 remainder");

        // Kyty EXIT_IF(map->Contains(p)) — (type, format) keys are unique.
        let mut seen = std::collections::HashSet::new();
        for e in table {
            assert!(
                seen.insert((e.type_, e.format)),
                "duplicate entry {:?}/{:?}",
                e.type_,
                e.format
            );
        }
    }

    #[test]
    fn dispatch_lookups() {
        // Implemented entry with its Kyty param string.
        let e = recomp_func(T::VMulF32, F::SVdstSVsrc0SVsrc1).expect("VMulF32");
        assert!(matches!(e.func, RecompileFn::Func(_)));
        assert_eq!(
            e.param[0],
            Some("%t_<index> = OpFMul %float %t0_<index> %t1_<index>")
        );

        // NI entry carries the Kyty function name + line anchor.
        let e = recomp_func(T::ImageSample, F::Vdata4Vaddr3StSsDmaskF).expect("ImageSample");
        match e.func {
            RecompileFn::NotImplemented { kyty_func, line } => {
                assert_eq!(kyty_func, "Recompile_ImageSample_Vdata4Vaddr3StSsDmaskF");
                assert_eq!(line, 2968);
            }
            RecompileFn::Func(_) => panic!("ImageSample must be NI in C1"),
        }

        // SccCheck rides along as table data (application is C2).
        let e = recomp_func(T::SAddU32, F::SVdstSVsrc0SVsrc1).expect("SAddU32");
        assert_eq!(e.scc_check, SccCheck::CarryOut);

        // Unknown (type, format) pair -> None.
        assert!(recomp_func(T::VMovB32, F::Label).is_none());
    }

    #[test]
    fn scc_check_templates() {
        // Kyty: get_scc_check (ShaderSpirv.cpp L1849).
        assert_eq!(get_scc_check(SccCheck::NonZero, 1), SCC_NZ_1);
        assert_eq!(get_scc_check(SccCheck::OverflowAdd, 1), SCC_OVERFLOW_ADD_1);
        assert_eq!(get_scc_check(SccCheck::OverflowSub, 1), SCC_OVERFLOW_SUB_1);
        assert_eq!(get_scc_check(SccCheck::CarryOut, 1), SCC_CARRY_1);
        assert_eq!(get_scc_check(SccCheck::NonZero, 2), SCC_NZ_2);
        assert_eq!(get_scc_check(SccCheck::None, 1), "");
        assert_eq!(get_scc_check(SccCheck::CarryOut, 2), "");
    }

    // ---- 2. acceptance: minimal VS ----------------------------------------

    #[test]
    fn acceptance_recompile_minimal_vs() {
        // v_mov_b32 v0, lit(1.0f); v_mov_b32 v1, 0; v_mul_f32 v2, v0, v1;
        // exp pos0 v0..v3 done; exp param0 v0..v3; s_endpgm.
        let code = parse(
            &[
                0x7E00_02FF,
                0x3F80_0000,
                0x7E02_0280,
                0x1004_0300,
                0xF800_08CF,
                0x0302_0100,
                0xF800_020F,
                0x0302_0100,
                S_ENDPGM,
            ],
            ShaderType::Vertex,
        );
        let input_info = ShaderVertexInputInfo {
            export_count: 1,
            ..Default::default()
        };

        // Source-level spot checks first (GenerateSource output).
        let source = spirv_generate_source(&code, Some(&input_info), None, None).unwrap();
        assert!(
            source.contains(
                "%t5_3 = OpAccessChain %_ptr_Output_v4float %outPerVertex %int_per_vertex_0"
            ),
            "exp pos0 output missing:\n{source}"
        );
        assert!(source.contains("OpStore %param0"), "{source}");
        assert!(
            source.contains("%t0_0 = OpBitcast %float %uint_0x3f800000"),
            "literal 1.0f load missing:\n{source}"
        );
        assert!(source.contains("OpReturn"), "{source}");

        // End-to-end: recompile -> assemble -> naga parse + validate.
        let words = shader_recompile_vs(&code, &input_info).expect("recompile vs");
        naga_parse_and_validate(&words, "minimal vs");
    }

    // ---- 3. acceptance: minimal PS ----------------------------------------

    #[test]
    fn acceptance_recompile_minimal_ps() {
        // v_interp_p1_f32 v2, v0, attr0.x; v_interp_p2_f32 v2, v1, attr0.x;
        // v_mul_f32 v0, v2, v2; exp mrt0 v0, v0 compr vm done; s_endpgm.
        let code = parse(
            &[
                0xC808_0000,
                0xC809_0001,
                0x1000_0502,
                0xF800_1C0F,
                0x0000_0000,
                S_ENDPGM,
            ],
            ShaderType::Pixel,
        );
        let mut input_info = ShaderPixelInputInfo {
            input_num: 1,
            ..Default::default()
        };
        input_info.interpolator_settings[0] = 0;
        input_info.target_output_mode[0] = 4;

        let source = spirv_generate_source(&code, None, Some(&input_info), None).unwrap();
        assert!(
            source.contains("%t0_1 = OpAccessChain %_ptr_Input_float %attr0 %uint_0"),
            "v_interp_p2 attr access missing:\n{source}"
        );
        assert!(source.contains("UnpackHalf2x16"), "{source}");
        assert!(source.contains("OpStore %outColor"), "{source}");
        assert!(
            source.contains("OpExecutionMode %main OriginUpperLeft"),
            "{source}"
        );

        let words = shader_recompile_ps(&code, &input_info).expect("recompile ps");
        naga_parse_and_validate(&words, "minimal ps");
    }

    // ---- 4. embedded bypass ------------------------------------------------

    #[test]
    fn embedded_vs_ps_bypass() {
        // Kyty: ShaderRecompileVS/PS embedded path (Shader.cpp L2370/L2472).
        let mut vs = ShaderCode::new();
        vs.set_type(ShaderType::Vertex);
        vs.set_vs_embedded(true);
        vs.set_vs_embedded_id(0);
        let words = shader_recompile_vs(&vs, &ShaderVertexInputInfo::default()).unwrap();
        naga_parse_and_validate(&words, "embedded vs0");

        let mut ps = ShaderCode::new();
        ps.set_type(ShaderType::Pixel);
        ps.set_ps_embedded(true);
        ps.set_ps_embedded_id(0);
        let words = shader_recompile_ps(&ps, &ShaderPixelInputInfo::default()).unwrap();
        naga_parse_and_validate(&words, "embedded ps0");
    }

    // ---- 5. error paths ----------------------------------------------------

    #[test]
    fn not_implemented_error_names_kyty_function() {
        // image_sample (MIMG) parses but its recompiler is C2.
        let code = parse(&[0xF080_0F00, 0x0061_0800, S_ENDPGM], ShaderType::Pixel);
        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        let err = shader_recompile_ps(&code, &input_info).unwrap_err();
        match err {
            ShaderRecompileError::NotImplemented {
                kyty_func, line, ..
            } => {
                assert_eq!(kyty_func, "Recompile_ImageSample_Vdata4Vaddr3StSsDmaskF");
                assert_eq!(line, 2968);
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn unknown_type_format_error() {
        // A (type, format) pair with no table row at all.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        let mut inst = ShaderInstruction {
            type_: T::SSetpcB64,
            format: F::Saddr,
            src_num: 1,
            ..Default::default()
        };
        inst.src[0] = ShaderOperand {
            type_: crate::shader::types::ShaderOperandType::Sgpr,
            register_id: 0,
            size: 2,
            ..Default::default()
        };
        code.get_instructions_mut().push(inst);
        let err = spirv_generate_source(&code, Some(&ShaderVertexInputInfo::default()), None, None)
            .unwrap_err();
        match err {
            ShaderRecompileError::UnknownTypeFormat { type_, format, .. } => {
                assert_eq!(type_, T::SSetpcB64);
                assert_eq!(format, F::Saddr);
            }
            other => panic!("expected UnknownTypeFormat, got {other:?}"),
        }
    }

    #[test]
    fn spirv_run_reports_asm_errors() {
        let err = spirv_run("%x OpNotARealInstruction\n").unwrap_err();
        assert!(matches!(err, ShaderRecompileError::Asm(_)));
    }
}
