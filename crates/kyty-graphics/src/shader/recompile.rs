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
    operand_load_float, operand_load_int, operand_load_uint, operand_variable_to_str,
    operand_variable_to_str_shift, spirv_generate_source, spirv_get_embedded_ps,
    spirv_get_embedded_vs,
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

/// Kyty: `Recompile_BufferStoreDword_Vdata1VaddrSvSoffsIdxen`
/// (ShaderSpirv.cpp L1999).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_buffer_store_dword_vdata1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_BufferStoreDword_Vdata1VaddrSvSoffsIdxen";
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
            // auto src1_value3 = operand_variable_to_str(inst.src[1], 3);
            let offset = spirv.get_constant(inst.src[2]);

            if dst_value.type_ != SpirvType::Float
                || src0_value.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src1_value1.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP

            const TEXT: &str = r#"
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
               OpSelectionMerge %t278_<index> None
               OpBranchConditional %exec_lo_b_<index> %t277_<index> %t278_<index>
		%t277_<index> = OpLabel

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
		;%t206_<index> = OpLoad %uint %<src1_value3>
        ;%t208_<index> = OpShiftRightLogical %uint %t206_<index> %int_12
        ;%t210_<index> = OpBitwiseAnd %uint %t208_<index> %uint_127
        ;%t211_<index> = OpBitcast %int %t210_<index>
        ;       OpStore %temp_int_5 %t211_<index>
        %t110_<index> = OpFunctionCall %void %buffer_store_float1 %<p0> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4

               OpBranch %t278_<index>
        %t278_<index> = OpLabel
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0>", &src0_value.value)
                .replace("<offset>", &offset)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value1>", &src1_value1.value)
                // .replace("<src1_value3>", ...) — commented out in Kyty too.
                .replace("<p0>", &dst_value.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_BufferStoreFormatX_Vdata1VaddrSvSoffsIdxen`
/// (ShaderSpirv.cpp L2068).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_buffer_store_format_x_vdata1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_BufferStoreFormatX_Vdata1VaddrSvSoffsIdxen";
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

            const TEXT: &str = r#"
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
               OpSelectionMerge %t278_<index> None
               OpBranchConditional %exec_lo_b_<index> %t277_<index> %t278_<index>
		%t277_<index> = OpLabel

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
        %t110_<index> = OpFunctionCall %void %tbuffer_store_format_x %<p0> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4 %temp_int_5

               OpBranch %t278_<index>
        %t278_<index> = OpLabel
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

/// Kyty: `Recompile_BufferStoreFormatXy_Vdata2VaddrSvSoffsIdxen`
/// (ShaderSpirv.cpp L2137).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_buffer_store_format_xy_vdata2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_BufferStoreFormatXy_Vdata2VaddrSvSoffsIdxen";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.storage_buffers.buffers_num > 0 {
            if !operand_is_constant(inst.src[2]) {
                return Err(not_supported(FUNC, "src2 is not a constant"));
            }

            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let src0_value = operand_variable_to_str(inst.src[0]);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
            let src1_value3 = operand_variable_to_str_shift(inst.src[1], 3);
            let offset = spirv.get_constant(inst.src[2]);

            if dst_value0.type_ != SpirvType::Float
                || src0_value.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src1_value1.type_ != SpirvType::Uint
                || src1_value3.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP

            const TEXT: &str = r#"
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
               OpSelectionMerge %t278_<index> None
               OpBranchConditional %exec_lo_b_<index> %t277_<index> %t278_<index>
		%t277_<index> = OpLabel

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
        %t110_<index> = OpFunctionCall %void %tbuffer_store_format_xy %<p0> %<p1> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4 %temp_int_5

               OpBranch %t278_<index>
        %t278_<index> = OpLabel
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0>", &src0_value.value)
                .replace("<offset>", &offset)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value1>", &src1_value1.value)
                .replace("<src1_value3>", &src1_value3.value)
                .replace("<p0>", &dst_value0.value)
                .replace("<p1>", &dst_value1.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Shared body of `Recompile_DsAppend_VdstGds` / `Recompile_DsConsume_VdstGds`
/// (ShaderSpirv.cpp L2208/L2243) — the two Kyty functions are identical
/// except for the atomic opcode.
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn ds_append_consume(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    atomic_op: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.gds_pointers.pointers_num > 0 {
            let index_str = format!("{index}");

            if !operand_is_variable(inst.dst) {
                return Err(not_supported(func, "dst is not a variable"));
            }

            let dst_value = operand_variable_to_str(inst.dst);

            if dst_value.type_ != SpirvType::Float {
                return Err(not_supported(func, "dst is not float"));
            }

            // TODO() check VSKIP
            // TODO() check EXEC

            const TEXT: &str = r#"
        %t192_<index> = OpLoad %uint %m0
        %t194_<index> = OpShiftRightLogical %uint %t192_<index> %int_16
        %t196_<index> = OpAccessChain %_ptr_StorageBuffer_uint %gds %int_0 %t194_<index>
        %t198_<index> = <atomic_op> %uint %t196_<index> %uint_1 %uint_0 %uint_1
        %t199_<index> = OpBitcast %float %t198_<index>
               OpStore %<dst> %t199_<index>
               OpMemoryBarrier %uint_1 %uint_72
"#;
            *dst_source += &TEXT
                .replace("<atomic_op>", atomic_op)
                .replace("<dst>", &dst_value.value)
                .replace("<index>", &index_str);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_DsAppend_VdstGds` (ShaderSpirv.cpp L2208).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_ds_append(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    ds_append_consume(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_DsAppend_VdstGds",
        "OpAtomicIAdd",
    )
}

/// Kyty: `Recompile_DsConsume_VdstGds` (ShaderSpirv.cpp L2243).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_ds_consume(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    ds_append_consume(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_DsConsume_VdstGds",
        "OpAtomicISub",
    )
}

/// Shared body of the seven `Recompile_ImageSample_*` dmask variants
/// (ShaderSpirv.cpp L2471/L2525/L2579/L2638/L2697/L2756/L2968). Upstream
/// duplicates the whole function per dmask; the bodies differ only in which
/// sampled channels land in which consecutive dst registers, expressed here
/// as `(temp_id, channel)` pairs matching Kyty's temp numbering exactly.
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn image_sample_channels(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    channels: &[(u32, u32)],
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_sampled_num > 0 && bind_info.samplers.samplers_num > 0 {
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);

            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            if dst_value0.type_ != SpirvType::Float
                || src0_value0.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src2_value0.type_ != SpirvType::Uint
            {
                return Err(not_supported(func, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check LOD_CLAMPED

            const HEAD: &str = r#"
         %t24_<index> = OpLoad %uint %<src1_value0>
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %t24_<index>
         %t27_<index> = OpLoad %ImageS %t26_<index>
         %t33_<index> = OpLoad %uint %<src2_value0>
         %t35_<index> = OpAccessChain %_ptr_UniformConstant_Sampler %samplers %t33_<index>
         %t36_<index> = OpLoad %Sampler %t35_<index>
         %t38_<index> = OpSampledImage %SampledImage %t27_<index> %t36_<index>
         %t39_<index> = OpLoad %float %<src0_value0>
         %t40_<index> = OpLoad %float %<src0_value1>
         %t42_<index> = OpCompositeConstruct %v2float %t39_<index> %t40_<index>
         %t43_<index> = OpImageSampleImplicitLod %v4float %t38_<index> %t42_<index>
               OpStore %temp_v4float %t43_<index>
"#;
            const TAIL: &str = r#"         %t<t0>_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_<chan>
         %t<t1>_<index> = OpLoad %float %t<t0>_<index>
               OpStore %<dst_value> %t<t1>_<index>
"#;

            let mut text = HEAD.to_string();
            for (i, (t0, chan)) in channels.iter().enumerate() {
                let dst_value = operand_variable_to_str_shift(inst.dst, i as i32);
                text += &TAIL
                    .replace("<t0>", &format!("{t0}"))
                    .replace("<t1>", &format!("{}", t0 + 1))
                    .replace("<chan>", &format!("{chan}"))
                    .replace("<dst_value>", &dst_value.value);
            }

            *dst_source += &text
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value)
                .replace("<index>", &format!("{index}"));

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_ImageSample_Vdata1Vaddr3StSsDmask1` (ShaderSpirv.cpp
/// L2471).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_sample_dmask1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSample_Vdata1Vaddr3StSsDmask1";
    image_sample_channels(index, code, dst_source, spirv, FUNC, &[(46, 0)])
}

/// Kyty: `Recompile_ImageSample_Vdata1Vaddr3StSsDmask8` (ShaderSpirv.cpp
/// L2525).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_sample_dmask8(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSample_Vdata1Vaddr3StSsDmask8";
    image_sample_channels(index, code, dst_source, spirv, FUNC, &[(46, 3)])
}

/// Kyty: `Recompile_ImageSample_Vdata2Vaddr3StSsDmask3` (ShaderSpirv.cpp
/// L2579).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_sample_dmask3(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSample_Vdata2Vaddr3StSsDmask3";
    image_sample_channels(index, code, dst_source, spirv, FUNC, &[(46, 0), (54, 1)])
}

/// Kyty: `Recompile_ImageSample_Vdata2Vaddr3StSsDmask5` (ShaderSpirv.cpp
/// L2638).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_sample_dmask5(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSample_Vdata2Vaddr3StSsDmask5";
    image_sample_channels(index, code, dst_source, spirv, FUNC, &[(46, 0), (54, 2)])
}

/// Kyty: `Recompile_ImageSample_Vdata2Vaddr3StSsDmask9` (ShaderSpirv.cpp
/// L2697).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_sample_dmask9(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSample_Vdata2Vaddr3StSsDmask9";
    image_sample_channels(index, code, dst_source, spirv, FUNC, &[(46, 0), (54, 3)])
}

/// Kyty: `Recompile_ImageSample_Vdata3Vaddr3StSsDmask7` (ShaderSpirv.cpp
/// L2756).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_sample_dmask7(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSample_Vdata3Vaddr3StSsDmask7";
    image_sample_channels(
        index,
        code,
        dst_source,
        spirv,
        FUNC,
        &[(46, 0), (50, 1), (54, 2)],
    )
}

/// Kyty: `Recompile_ImageSample_Vdata4Vaddr3StSsDmaskF` (ShaderSpirv.cpp
/// L2968).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_sample_dmask_f(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSample_Vdata4Vaddr3StSsDmaskF";
    image_sample_channels(
        index,
        code,
        dst_source,
        spirv,
        FUNC,
        &[(46, 0), (50, 1), (54, 2), (57, 3)],
    )
}

/// Kyty: `Recompile_ImageSampleLz_Vdata3Vaddr3StSsDmask7` (ShaderSpirv.cpp
/// L2821).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_sample_lz_dmask7(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLz_Vdata3Vaddr3StSsDmask7";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_sampled_num > 0 && bind_info.samplers.samplers_num > 0 {
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);

            if dst_value0.type_ != SpirvType::Float
                || src0_value0.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src2_value0.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check LOD_CLAMPED

            const TEXT: &str = r#"
         %t24_<index> = OpLoad %uint %<src1_value0>
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %t24_<index>
         %t27_<index> = OpLoad %ImageS %t26_<index>
         %t33_<index> = OpLoad %uint %<src2_value0>
         %t35_<index> = OpAccessChain %_ptr_UniformConstant_Sampler %samplers %t33_<index>
         %t36_<index> = OpLoad %Sampler %t35_<index>
         %t38_<index> = OpSampledImage %SampledImage %t27_<index> %t36_<index>

         %t39_<index> = OpLoad %float %<src0_value0>
         %t40_<index> = OpLoad %float %<src0_value1>
         %t42_<index> = OpCompositeConstruct %v2float %t39_<index> %t40_<index>

         %t43_<index> = OpImageSampleExplicitLod %v4float %t38_<index> %t42_<index> Lod %float_0_000000
               OpStore %temp_v4float %t43_<index>
         %t46_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_0
         %t47_<index> = OpLoad %float %t46_<index>
               OpStore %<dst_value0> %t47_<index>
         %t50_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_1
         %t51_<index> = OpLoad %float %t50_<index>
               OpStore %<dst_value1> %t51_<index>
         %t54_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_2
         %t55_<index> = OpLoad %float %t54_<index>
               OpStore %<dst_value2> %t55_<index>
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_ImageSampleLzO_Vdata3Vaddr4StSsDmask7` (ShaderSpirv.cpp
/// L2887).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_sample_lzo_dmask7(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLzO_Vdata3Vaddr4StSsDmask7";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_sampled_num > 0 && bind_info.samplers.samplers_num > 0 {
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
            let src0_value2 = operand_variable_to_str_shift(inst.src[0], 2);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);

            if dst_value0.type_ != SpirvType::Float
                || src0_value0.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src2_value0.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check LOD_CLAMPED

            const TEXT: &str = r#"
         %t24_<index> = OpLoad %uint %<src1_value0>
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %t24_<index>
         %t27_<index> = OpLoad %ImageS %t26_<index>
         %t33_<index> = OpLoad %uint %<src2_value0>
         %t35_<index> = OpAccessChain %_ptr_UniformConstant_Sampler %samplers %t33_<index>
         %t36_<index> = OpLoad %Sampler %t35_<index>
         %t38_<index> = OpSampledImage %SampledImage %t27_<index> %t36_<index>

         %t39_<index> = OpLoad %float %<src0_value1>
         %t40_<index> = OpLoad %float %<src0_value2>
         %t42_<index> = OpCompositeConstruct %v2float %t39_<index> %t40_<index>

         %90_<index> = OpLoad %float %<src0_value0>
         %91_<index> = OpBitcast %int %90_<index>
         %98_<index> = OpBitFieldSExtract %int %91_<index> %int_0 %int_6
        %101_<index> = OpBitFieldSExtract %int %91_<index> %int_8 %int_6
        %102_<index> = OpCompositeConstruct %v2int %98_<index> %101_<index>

         %130_<index> = OpConvertSToF %v2float %102_<index>
         %138_<index> = OpImage %ImageS %t38_<index>
        %139_<index> = OpImageQuerySizeLod %v2int %138_<index> %int_0
        %140_<index> = OpConvertSToF %v2float %139_<index>
        %141_<index> = OpFDiv %v2float %130_<index> %140_<index>
        %142_<index> = OpFAdd %v2float %t42_<index> %141_<index>

         %t43_<index> = OpImageSampleExplicitLod %v4float %t38_<index> %142_<index> Lod %float_0_000000
               OpStore %temp_v4float %t43_<index>
         %t46_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_0
         %t47_<index> = OpLoad %float %t46_<index>
               OpStore %<dst_value0> %t47_<index>
         %t50_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_1
         %t51_<index> = OpLoad %float %t50_<index>
               OpStore %<dst_value1> %t51_<index>
         %t54_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_2
         %t55_<index> = OpLoad %float %t54_<index>
               OpStore %<dst_value2> %t55_<index>
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src0_value2>", &src0_value2.value)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_ImageLoad_Vdata4Vaddr3StDmaskF` (ShaderSpirv.cpp L3038).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_load_dmask_f(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageLoad_Vdata4Vaddr3StDmaskF";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_sampled_num > 0 {
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let dst_value3 = operand_variable_to_str_shift(inst.dst, 3);
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);

            if dst_value0.type_ != SpirvType::Float
                || src0_value0.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check LOD_CLAMPED
            // TODO() swizzle channels
            // TODO() convert SRGB -> LINEAR if SRGB format was replaced with UNORM

            const TEXT: &str = r#"
         %t24_<index> = OpLoad %uint %<src1_value0>
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %t24_<index>
         %t27_<index> = OpLoad %ImageS %t26_<index>
         %t67_<index> = OpLoad %float %<src0_value0>
         %t69_<index> = OpBitcast %uint %t67_<index>
         %t70_<index> = OpLoad %float %<src0_value1>
         %t71_<index> = OpBitcast %uint %t70_<index>
         %t73_<index> = OpCompositeConstruct %v2uint %t69_<index> %t71_<index>
         %t74_<index> = OpImageFetch %v4float %t27_<index> %t73_<index>
               OpStore %temp_v4float %t74_<index>
         %t46_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_0
         %t47_<index> = OpLoad %float %t46_<index>
               OpStore %<dst_value0> %t47_<index>
         %t50_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_1
         %t51_<index> = OpLoad %float %t50_<index>
               OpStore %<dst_value1> %t51_<index>
         %t54_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_2
         %t55_<index> = OpLoad %float %t54_<index>
               OpStore %<dst_value2> %t55_<index>
         %t57_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_3
         %t58_<index> = OpLoad %float %t57_<index>
               OpStore %<dst_value3> %t58_<index>
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value)
                .replace("<dst_value3>", &dst_value3.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_ImageStore_Vdata4Vaddr3StDmaskF` (ShaderSpirv.cpp L3105).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_store_dmask_f(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageStore_Vdata4Vaddr3StDmaskF";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_storage_num > 0 {
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let dst_value3 = operand_variable_to_str_shift(inst.dst, 3);

            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);

            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src1_value2 = operand_variable_to_str_shift(inst.src[1], 2);

            if dst_value0.type_ != SpirvType::Float
                || src0_value0.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check LOD_CLAMPED
            // TODO() swizzle channels
            // TODO() convert SRGB -> LINEAR if SRGB format was replaced with UNORM

            const TEXT: &str = r#"
         %t24_<index> = OpLoad %uint %<src1_value0>
         %t25_<index> = OpLoad %uint %<src1_value2>
		%t143_<index> = OpShiftRightLogical %uint %t25_<index> %uint_0
        %t145_<index> = OpBitwiseAnd %uint %t143_<index> %uint_0x00003fff
        %t146_<index> = OpIAdd %uint %t145_<index> %uint_1
        %t149_<index> = OpShiftRightLogical %uint %t25_<index> %uint_14
        %t150_<index> = OpBitwiseAnd %uint %t149_<index> %uint_0x00003fff
        %t151_<index> = OpIAdd %uint %t150_<index> %uint_1
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageL %textures2D_L %t24_<index>
         %t27_<index> = OpLoad %ImageL %t26_<index>
         %t67_<index> = OpLoad %float %<src0_value0>
         %t69_<index> = OpBitcast %uint %t67_<index>
         %t70_<index> = OpLoad %float %<src0_value1>
         %t71_<index> = OpBitcast %uint %t70_<index>
         %t73_<index> = OpCompositeConstruct %v2uint %t69_<index> %t71_<index>
         %t84_<index> = OpLoad %float %<dst_value0>
         %t85_<index> = OpLoad %float %<dst_value1>
         %t86_<index> = OpLoad %float %<dst_value2>
         %t87_<index> = OpLoad %float %<dst_value3>
         %t88_<index> = OpCompositeConstruct %v4float %t84_<index> %t85_<index> %t86_<index> %t87_<index>
               OpImageWrite %t27_<index> %t73_<index> %t88_<index>
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value2>", &src1_value2.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value)
                .replace("<dst_value3>", &dst_value3.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_ImageStoreMip_Vdata4Vaddr4StDmaskF` (ShaderSpirv.cpp
/// L3173).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_image_store_mip_dmask_f(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageStoreMip_Vdata4Vaddr4StDmaskF";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_storage_num > 0 {
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let dst_value3 = operand_variable_to_str_shift(inst.dst, 3);

            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
            let src0_value2 = operand_variable_to_str_shift(inst.src[0], 2);

            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src1_value2 = operand_variable_to_str_shift(inst.src[1], 2);

            if dst_value0.type_ != SpirvType::Float
                || src0_value0.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check LOD_CLAMPED
            // TODO() swizzle channels
            // TODO() convert SRGB -> LINEAR if SRGB format was replaced with UNORM

            const TEXT: &str = r#"
         %t24_<index> = OpLoad %uint %<src1_value0>
         %t25_<index> = OpLoad %uint %<src1_value2>
		%t143_<index> = OpShiftRightLogical %uint %t25_<index> %uint_0
        %t145_<index> = OpBitwiseAnd %uint %t143_<index> %uint_0x00003fff
        %t146_<index> = OpIAdd %uint %t145_<index> %uint_1
        %t149_<index> = OpShiftRightLogical %uint %t25_<index> %uint_14
        %t150_<index> = OpBitwiseAnd %uint %t149_<index> %uint_0x00003fff
        %t151_<index> = OpIAdd %uint %t150_<index> %uint_1
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageL %textures2D_L %t24_<index>
         %t27_<index> = OpLoad %ImageL %t26_<index>
         %t67_<index> = OpLoad %float %<src0_value0>
         %t69_<index> = OpBitcast %uint %t67_<index>
         %t70_<index> = OpLoad %float %<src0_value1>
         %t71_<index> = OpBitcast %uint %t70_<index>
         %t701_<index> = OpLoad %float %<src0_value2>
         %t711_<index> = OpBitcast %uint %t701_<index>
         %t160_<index> = OpFunctionCall %v2uint %mipmap %t711_<index> %t146_<index> %t151_<index>
         %t73_<index> = OpCompositeConstruct %v2uint %t69_<index> %t71_<index>
         %t84_<index> = OpLoad %float %<dst_value0>
         %t85_<index> = OpLoad %float %<dst_value1>
         %t86_<index> = OpLoad %float %<dst_value2>
         %t87_<index> = OpLoad %float %<dst_value3>
         %t172_<index> = OpIAdd %v2uint %t160_<index> %t73_<index>
         %t88_<index> = OpCompositeConstruct %v4float %t84_<index> %t85_<index> %t86_<index> %t87_<index>
               OpImageWrite %t27_<index> %t172_<index> %t88_<index>
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src0_value2>", &src0_value2.value)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value2>", &src1_value2.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value)
                .replace("<dst_value3>", &dst_value3.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12` (ShaderSpirv.cpp L3248).
/// XXX: Andn2, Orn2, And, Nor, Nand, Xnor, Or, Xor, Cselect (via `param`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_s_xxx_b64_sdst2_ssrc02_ssrc12(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_S_XXX_B64_Sdst2Ssrc02Ssrc12";
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
    let mut load2 = String::new();
    let mut load3 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, 0)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[0], "t1_<index>", &index_str, &mut load1, 1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t2_<index>", &index_str, &mut load2, 0)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t3_<index>", &index_str, &mut load3, 1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
    <load0>
    <load1>
    <load2>
    <load3>
    <param0>
    <param1>
    <param2>
    <param3>
    OpStore %<dst0> %tb_<index>
    OpStore %<dst1> %td_<index>
    <execz>
    <scc>
"#;

    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<load2>", &load2)
        .replace("<load3>", &load3)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<param2>", param[2].unwrap_or(""))
        .replace("<param3>", param[3].unwrap_or(""))
        .replace(
            "<execz>",
            if operand_is_exec(inst.dst) { EXECZ } else { "" },
        )
        .replace("<scc>", get_scc_check(scc_check, 2))
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Shared body of `Recompile_S_Lshl_B64_Sdst2Ssrc02Ssrc1` /
/// `Recompile_S_Lshr_B64_Sdst2Ssrc02Ssrc1` (ShaderSpirv.cpp L3316/L3384) —
/// identical upstream except for the `shift_left`/`shift_right` callee.
#[allow(dead_code)]
// C2: staged recompiler, not yet wired into G_RECOMP_FUNC
// The 6-arg recompiler signature plus the two args that parameterise the
// upstream difference; splitting it would diverge from the Kyty shape.
#[allow(clippy::too_many_arguments)]
fn s_shift_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    scc_check: SccCheck,
    func: &'static str,
    shift_func: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(func, "dst is not a variable"));
    }

    let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
    let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);

    if dst_value0.type_ != SpirvType::Uint {
        return Err(not_supported(func, "dst is not uint"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();
    let mut load2 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, 0)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[0], "t1_<index>", &index_str, &mut load1, 1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t2_<index>", &index_str, &mut load2, -1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
     <load0>
     <load1>
     <load2>
     <param0>
     <param1>
     <param2>
     <param3>
%t22_<index> = OpBitwiseAnd %uint %t2_<index> %uint_63
     OpStore %temp_uint_2 %t0_<index>
     OpStore %temp_uint_3 %t1_<index>
     OpStore %temp_uint_4 %t22_<index>
%t_<index> = OpFunctionCall %void %<shift_func> %temp_uint_0 %temp_uint_1 %temp_uint_2 %temp_uint_3 %temp_uint_4
%r0_<index> = OpLoad %uint %temp_uint_0
%r1_<index> = OpLoad %uint %temp_uint_1
     OpStore %<dst0> %r0_<index>
     OpStore %<dst1> %r1_<index>
     <execz>
     <scc>
"#;

    *dst_source += &TEXT
        .replace("<shift_func>", shift_func)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<load2>", &load2)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<param2>", param[2].unwrap_or(""))
        .replace("<param3>", param[3].unwrap_or(""))
        .replace(
            "<execz>",
            if operand_is_exec(inst.dst) { EXECZ } else { "" },
        )
        .replace("<scc>", get_scc_check(scc_check, 2))
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_S_Lshl_B64_Sdst2Ssrc02Ssrc1` (ShaderSpirv.cpp L3316).
fn recompile_s_lshl_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    s_shift_b64(
        index,
        code,
        dst_source,
        spirv,
        param,
        scc_check,
        "Recompile_S_Lshl_B64_Sdst2Ssrc02Ssrc1",
        "shift_left",
    )
}

/// Kyty: `Recompile_S_Lshr_B64_Sdst2Ssrc02Ssrc1` (ShaderSpirv.cpp L3384).
fn recompile_s_lshr_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    s_shift_b64(
        index,
        code,
        dst_source,
        spirv,
        param,
        scc_check,
        "Recompile_S_Lshr_B64_Sdst2Ssrc02Ssrc1",
        "shift_right",
    )
}

/// Kyty: `Recompile_S_Bfe_U64_Sdst2Ssrc02Ssrc1` (ShaderSpirv.cpp L3452).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_s_bfe_u64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_S_Bfe_U64_Sdst2Ssrc02Ssrc1";
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
    let mut load2 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, 0)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[0], "t1_<index>", &index_str, &mut load1, 1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t2_<index>", &index_str, &mut load2, -1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
     <load0>
     <load1>
     <load2>
     <param0>
     <param1>
     <param2>
     <param3>
 %to_<index> = OpBitFieldUExtract %uint %t2_<index> %uint_0  %uint_6
 %ts_<index> = OpBitFieldUExtract %uint %t2_<index> %uint_16 %uint_7
%tn0_<index> = OpISub %uint %uint_64 %to_<index>
%ts2_<index> = OpExtInst %uint %GLSL_std_450 UMin %ts_<index> %tn0_<index>
%tn1_<index> = OpISub %uint %uint_64 %ts2_<index>
%tn2_<index> = OpISub %uint %tn1_<index> %to_<index>
     OpStore %temp_uint_2 %t0_<index>
     OpStore %temp_uint_3 %t1_<index>
     OpStore %temp_uint_4 %tn2_<index>
%tf1_<index> = OpFunctionCall %void %shift_left %temp_uint_0 %temp_uint_1 %temp_uint_2 %temp_uint_3 %temp_uint_4
     OpStore %temp_uint_4 %tn1_<index>
%tf2_<index> = OpFunctionCall %void %shift_right %temp_uint_2 %temp_uint_3 %temp_uint_0 %temp_uint_1 %temp_uint_4
 %r0_<index> = OpLoad %uint %temp_uint_2
 %r1_<index> = OpLoad %uint %temp_uint_3
     OpStore %<dst0> %r0_<index>
     OpStore %<dst1> %r1_<index>
     <execz>
     <scc>
"#;

    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<load2>", &load2)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<param2>", param[2].unwrap_or(""))
        .replace("<param3>", param[3].unwrap_or(""))
        .replace(
            "<execz>",
            if operand_is_exec(inst.dst) { EXECZ } else { "" },
        )
        .replace("<scc>", get_scc_check(scc_check, 2))
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_S_XXX_B32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp L3528).
/// XXX: And, Bfm, Cselect, Lshl, Lshr, Or (via `param`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_s_xxx_b32_svdst_svsrc01(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_S_XXX_B32_SVdstSVsrc0SVsrc1";
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
    let mut load1 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
              <load0>
              <load1>
              <param0>
              <param1>
              <param2>
              OpStore %<dst> %t_<index>
              <scc>
"#;
    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<param2>", param[2].unwrap_or(""))
        .replace("<scc>", get_scc_check(scc_check, 1))
        .replace("<dst>", &dst_value.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_S_XXX_I32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp L3576).
/// XXX: Add, Mul, Sub (via `param`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_s_xxx_i32_svdst_svsrc01(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_S_XXX_I32_SVdstSVsrc0SVsrc1";
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
    let mut load1 = String::new();

    if !operand_load_int(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_int(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
              <load0>
              <load1>
              <param>
              %tu_<index> = OpBitcast %uint %t_<index>
              OpStore %<dst> %tu_<index>
              <scc>
"#;
    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<scc>", get_scc_check(scc_check, 1))
        .replace("<dst>", &dst_value.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_S_XXX_U32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp L3621).
/// XXX: Add, Addc, Bfe, Lshl4Add, MulHi (via `param`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_s_xxx_u32_svdst_svsrc01(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_S_XXX_U32_SVdstSVsrc0SVsrc1";
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
    let mut load1 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
              <load0>
              <load1>
              <param0>
              <param1>
              <param2>
              <param3>
              OpStore %<dst> %t_<index>
              <scc>
"#;
    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<param2>", param[2].unwrap_or(""))
        .replace("<param3>", param[3].unwrap_or(""))
        .replace("<scc>", get_scc_check(scc_check, 1))
        .replace("<dst>", &dst_value.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_SAndSaveexecB64_Sdst2Ssrc02` (ShaderSpirv.cpp L3670).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_s_and_saveexec_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SAndSaveexecB64_Sdst2Ssrc02";
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
    if operand_is_exec(inst.dst) {
        return Err(not_supported(FUNC, "exec destination"));
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
        %t190_<index> = OpLoad %uint %exec_lo
               OpStore %<dst0> %t190_<index>
        %t191_<index> = OpLoad %uint %exec_hi
               OpStore %<dst1> %t191_<index>
        %t194_<index> = OpBitwiseAnd %uint %t0_<index> %t190_<index>
               OpStore %exec_lo %t194_<index>
        %t197_<index> = OpBitwiseAnd %uint %t1_<index> %t191_<index>
               OpStore %exec_hi %t197_<index>
        <execz>
        <scc>
"#;

    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<execz>", EXECZ)
        .replace("<scc>", get_scc_check(scc_check, 2))
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_SCmp_XXX_I32_Ssrc0Ssrc1` (ShaderSpirv.cpp L3725).
/// XXX: Eq, Ge, Gt, Lg, Lt, Le (comparison opcode via `param[0]`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_scmp_xxx_i32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SCmp_XXX_I32_Ssrc0Ssrc1";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_int(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_int(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpSelect %uint %t2_<index> %uint_1 %uint_0
          OpStore %scc %t3_<index>
"#;
    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_SCmp_XXX_U32_Ssrc0Ssrc1` (ShaderSpirv.cpp L3760).
/// XXX: Eq, Ge, Gt, Le, Lt, Lg (comparison opcode via `param[0]`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_scmp_xxx_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SCmp_XXX_U32_Ssrc0Ssrc1";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpSelect %uint %t2_<index> %uint_1 %uint_0
          OpStore %scc %t3_<index>
"#;
    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_SMulkI32_SVdstSVsrc0` (ShaderSpirv.cpp L4437).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_smulk_i32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SMulkI32_SVdstSVsrc0";
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
    if !operand_load_int(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }

    let mut load_dst = String::new();
    if !operand_load_int(spirv, inst.dst, "tdst_<index>", &index_str, &mut load_dst)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
    <load0>
    <load_dst>
%t_<index> = OpIMul %int %tdst_<index> %t0_<index>
%tu_<index> = OpBitcast %uint %t_<index>
    OpStore %<dst> %tu_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<load_dst>", &load_dst)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_SWqmB64_Sdst2Ssrc02` (ShaderSpirv.cpp L4621).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_swqm_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SWqmB64_Sdst2Ssrc02";
    let inst = inst_at(code, index, FUNC)?;

    if inst.dst.type_ == ShaderOperandType::ExecLo && inst.src[0].type_ == ShaderOperandType::ExecLo
    {
        return Ok(true);
    }

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
        %t170_<index> = OpFunctionCall %uint %wqm %t0_<index> %uint_0 %uint_15
        %t172_<index> = OpBitwiseOr %uint %uint_0 %t170_<index>
        %t179_<index> = OpFunctionCall %uint %wqm %t0_<index> %uint_4 %uint_240
        %t181_<index> = OpBitwiseOr %uint %t172_<index> %t179_<index>
        %t188_<index> = OpFunctionCall %uint %wqm %t0_<index> %uint_8 %uint_0x00000f00
        %t190_<index> = OpBitwiseOr %uint %t181_<index> %t188_<index>
        %t197_<index> = OpFunctionCall %uint %wqm %t0_<index> %uint_12 %uint_0x0000f000
        %t199_<index> = OpBitwiseOr %uint %t190_<index> %t197_<index>
        %t206_<index> = OpFunctionCall %uint %wqm %t0_<index> %uint_16 %uint_0x000f0000
        %t208_<index> = OpBitwiseOr %uint %t199_<index> %t206_<index>
        %t215_<index> = OpFunctionCall %uint %wqm %t0_<index> %uint_20 %uint_0x00f00000
        %t217_<index> = OpBitwiseOr %uint %t208_<index> %t215_<index>
        %t224_<index> = OpFunctionCall %uint %wqm %t0_<index> %uint_24 %uint_0x0f000000
        %t226_<index> = OpBitwiseOr %uint %t217_<index> %t224_<index>
        %t233_<index> = OpFunctionCall %uint %wqm %t0_<index> %uint_28 %uint_0xf0000000
        %t235_<index> = OpBitwiseOr %uint %t226_<index> %t233_<index>
        %t1701_<index> = OpFunctionCall %uint %wqm %t1_<index> %uint_0 %uint_15
        %t1721_<index> = OpBitwiseOr %uint %uint_0 %t1701_<index>
        %t1791_<index> = OpFunctionCall %uint %wqm %t1_<index> %uint_4 %uint_240
        %t1811_<index> = OpBitwiseOr %uint %t1721_<index> %t1791_<index>
        %t1881_<index> = OpFunctionCall %uint %wqm %t1_<index> %uint_8 %uint_0x00000f00
        %t1901_<index> = OpBitwiseOr %uint %t1811_<index> %t1881_<index>
        %t1971_<index> = OpFunctionCall %uint %wqm %t1_<index> %uint_12 %uint_0x0000f000
        %t1991_<index> = OpBitwiseOr %uint %t1901_<index> %t1971_<index>
        %t2061_<index> = OpFunctionCall %uint %wqm %t1_<index> %uint_16 %uint_0x000f0000
        %t2081_<index> = OpBitwiseOr %uint %t1991_<index> %t2061_<index>
        %t2151_<index> = OpFunctionCall %uint %wqm %t1_<index> %uint_20 %uint_0x00f00000
        %t2171_<index> = OpBitwiseOr %uint %t2081_<index> %t2151_<index>
        %t2241_<index> = OpFunctionCall %uint %wqm %t1_<index> %uint_24 %uint_0x0f000000
        %t2261_<index> = OpBitwiseOr %uint %t2171_<index> %t2241_<index>
        %t2331_<index> = OpFunctionCall %uint %wqm %t1_<index> %uint_28 %uint_0xf0000000
        %t2351_<index> = OpBitwiseOr %uint %t2261_<index> %t2331_<index>
               OpStore %<dst0> %t235_<index>
               OpStore %<dst1> %t2351_<index>
        <execz>
        <scc>
"#;

    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace(
            "<execz>",
            if operand_is_exec(inst.dst) { EXECZ } else { "" },
        )
        .replace("<scc>", get_scc_check(scc_check, 2))
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_SBufferLoadDwordx2_Sdst2SvSoffset` (ShaderSpirv.cpp
/// L3831).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_sbuffer_load_dwordx2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SBufferLoadDwordx2_Sdst2SvSoffset";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.storage_buffers.buffers_num > 0 {
            if !operand_is_constant(inst.src[1]) {
                return Err(not_supported(FUNC, "src1 is not a constant"));
            }

            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let offset = spirv.get_constant(inst.src[1]);

            if dst_value0.type_ != SpirvType::Uint || src0_value0.type_ != SpirvType::Uint {
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
        %t110_<index> = OpFunctionCall %void %sbuffer_load_dword_2 %<p0> %<p1> %temp_int_1 %temp_int_2
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<offset>", &offset)
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<p0>", &dst_value0.value)
                .replace("<p1>", &dst_value1.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Shared body of `Recompile_SBufferLoadDwordx8/x16_Sdst*SvSoffset`
/// (ShaderSpirv.cpp L3928/L3976) — identical upstream except for N and the
/// `sbuffer_load_dword_N` callee.
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn sbuffer_load_dword_n(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    n: usize,
    callee: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.storage_buffers.buffers_num > 0 {
            if !operand_is_constant(inst.src[1]) {
                return Err(not_supported(func, "src1 is not a constant"));
            }

            let dst_value: Vec<_> = (0..n)
                .map(|i| operand_variable_to_str_shift(inst.dst, i as i32))
                .collect();

            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let offset = spirv.get_constant(inst.src[1]);

            if dst_value[0].type_ != SpirvType::Uint || src0_value0.type_ != SpirvType::Uint {
                return Err(not_supported(func, "unexpected operand types"));
            }
            if operand_is_exec(inst.dst) {
                return Err(not_supported(func, "exec destination"));
            }

            const TEXT: &str = r#"
        %t100_<index> = OpLoad %uint %<src0_value0>
        %t101_<index> = OpBitcast %int %t100_<index>
               OpStore %temp_int_2 %t101_<index>
        %t102_<index> = OpBitcast %int %<offset>
               OpStore %temp_int_1 %t102_<index>
        %t110_<index> = OpFunctionCall %void %<callee> <regs> %temp_int_1 %temp_int_2
"#;
            let regs: Vec<String> = dst_value.iter().map(|v| format!("%{}", v.value)).collect();

            *dst_source += &TEXT
                .replace("<callee>", callee)
                .replace("<regs>", &regs.join(" "))
                .replace("<index>", &format!("{index}"))
                .replace("<offset>", &offset)
                .replace("<src0_value0>", &src0_value0.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_SBufferLoadDwordx8_Sdst8SvSoffset` (ShaderSpirv.cpp
/// L3928).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_sbuffer_load_dwordx8(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    sbuffer_load_dword_n(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_SBufferLoadDwordx8_Sdst8SvSoffset",
        8,
        "sbuffer_load_dword_8",
    )
}

/// Kyty: `Recompile_SBufferLoadDwordx16_Sdst16SvSoffset` (ShaderSpirv.cpp
/// L3976).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_sbuffer_load_dwordx16(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    sbuffer_load_dword_n(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_SBufferLoadDwordx16_Sdst16SvSoffset",
        16,
        "sbuffer_load_dword_16",
    )
}

/// Kyty: `Recompile_TBufferLoadFormatXyzw_Vdata4Vaddr2SvSoffsOffenIdxenFloat4`
/// (ShaderSpirv.cpp L4824).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_tbuffer_load_format_xyzw_offen_float4(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_TBufferLoadFormatXyzw_Vdata4Vaddr2SvSoffsOffenIdxenFloat4";
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
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
            let offset = spirv.get_constant(inst.src[2]);

            if dst_value0.type_ != SpirvType::Float
                || src0_value0.type_ != SpirvType::Float
                || src0_value1.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src1_value1.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check EXEC

            const TEXT: &str = r#"
        %t100_<index> = OpLoad %float %<src0_value0>
        %t101_<index> = OpBitcast %int %t100_<index>
       %to100_<index> = OpLoad %float %<src0_value1>
       %to101_<index> = OpBitcast %int %to100_<index>
               OpStore %temp_int_1 %t101_<index>
        %t148_<index> = OpLoad %uint %<src1_value1>
        %t150_<index> = OpShiftRightLogical %uint %t148_<index> %int_16
        %t152_<index> = OpBitwiseAnd %uint %t150_<index> %uint_0x00003fff
        %t153_<index> = OpBitcast %int %t152_<index>
               OpStore %temp_int_3 %t153_<index>
        %t155_<index> = OpLoad %uint %<src1_value0>
        %t156_<index> = OpBitcast %int %t155_<index>
      %offset_<index> = OpIAdd %int %to101_<index> %<offset>
               OpStore %temp_int_4 %t156_<index>
               OpStore %temp_int_2 %offset_<index>
               OpStore %temp_int_5 %int_119
        %t110_<index> = OpFunctionCall %void %tbuffer_load_format_xyzw %<p0> %<p1> %<p2> %<p3> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4 %temp_int_5
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
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

/// Kyty: `Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1` (ShaderSpirv.cpp L4890).
/// XXX: F, Eq, Ge, Gt, Le, Lg, Lt, Neq, Nge, Ngt, Nle, Nlg, Nlt, O, Tru, U
/// (comparison via `param[0]`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_vcmp_xxx_f32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCmp_XXX_F32_SmaskVsrc0Vsrc1";
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
    if operand_is_exec(inst.dst) {
        return Err(not_supported(FUNC, "exec destination"));
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
    // TODO() check EXEC

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpSelect %uint %t2_<index> %uint_1 %uint_0
          OpStore %<dst0> %t3_<index>
          OpStore %<dst1> %uint_0
"#;
    *dst_source += &TEXT
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1` (ShaderSpirv.cpp L4940).
/// XXX: Eq, Ne, Gt, Ge, F, Le, Lt, T (comparison via `param[0]`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_vcmp_xxx_i32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCmp_XXX_I32_SmaskVsrc0Vsrc1";
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
    if operand_is_exec(inst.dst) {
        return Err(not_supported(FUNC, "exec destination"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_int(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_int(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpSelect %uint %t2_<index> %uint_1 %uint_0
          OpStore %<dst0> %t3_<index>
          OpStore %<dst1> %uint_0
"#;
    *dst_source += &TEXT
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VCmp_XXX_U32_SmaskVsrc0Vsrc1` (ShaderSpirv.cpp L4990).
/// XXX: Le, Ge, F, Gt, Lt, T (comparison via `param[0]`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_vcmp_xxx_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCmp_XXX_U32_SmaskVsrc0Vsrc1";
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
    if operand_is_exec(inst.dst) {
        return Err(not_supported(FUNC, "exec destination"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpSelect %uint %t2_<index> %uint_1 %uint_0
          OpStore %<dst0> %t3_<index>
          OpStore %<dst1> %uint_0
"#;
    *dst_source += &TEXT
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VCmpx_XXX_I32_SmaskVsrc0Vsrc1` (ShaderSpirv.cpp L5040).
/// XXX: Eq, Ne (comparison via `param[0]`; also writes EXEC).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_vcmpx_xxx_i32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCmpx_XXX_I32_SmaskVsrc0Vsrc1";
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
    if operand_is_exec(inst.dst) {
        return Err(not_supported(FUNC, "exec destination"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_int(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_int(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpSelect %uint %t2_<index> %uint_1 %uint_0
          OpStore %<dst0> %t3_<index>
          OpStore %<dst1> %uint_0
          OpStore %exec_lo %t3_<index>
          OpStore %exec_hi %uint_0
          <execz>
"#;
    *dst_source += &TEXT
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<execz>", EXECZ)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VCmpx_XXX_U32_SmaskVsrc0Vsrc1` (ShaderSpirv.cpp L5094).
/// XXX: Gt, Ge (comparison via `param[0]`; also writes EXEC).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_vcmpx_xxx_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCmpx_XXX_U32_SmaskVsrc0Vsrc1";
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
    if operand_is_exec(inst.dst) {
        return Err(not_supported(FUNC, "exec destination"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpSelect %uint %t2_<index> %uint_1 %uint_0
          OpStore %<dst0> %t3_<index>
          OpStore %<dst1> %uint_0
          OpStore %exec_lo %t3_<index>
          OpStore %exec_hi %uint_0
          <execz>
"#;
    *dst_source += &TEXT
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<execz>", EXECZ)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VCmpx_XXX_F32_SmaskVsrc0Vsrc1` (ShaderSpirv.cpp L5148).
/// XXX: Neq, Gt, Lt (comparison via `param[0]`; also writes EXEC).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_vcmpx_xxx_f32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCmpx_XXX_F32_SmaskVsrc0Vsrc1";
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
    if operand_is_exec(inst.dst) {
        return Err(not_supported(FUNC, "exec destination"));
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
    // TODO() check EXEC

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpSelect %uint %t2_<index> %uint_1 %uint_0
          OpStore %<dst0> %t3_<index>
          OpStore %<dst1> %uint_0
          OpStore %exec_lo %t3_<index>
          OpStore %exec_hi %uint_0
          <execz>
"#;
    *dst_source += &TEXT
        .replace("<dst0>", &dst_value0.value)
        .replace("<dst1>", &dst_value1.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<execz>", EXECZ)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VCvtPkrtzF16F32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp
/// L5260).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_vcvt_pkrtz_f16_f32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCvtPkrtzF16F32_SVdstSVsrc0SVsrc1";
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

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_float(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_float(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check DX10_CLAMP

    const TEXT: &str = r#"
    <load0>
    <load1>
    %t0u_<index> = OpBitcast %uint %t0_<index>
    %t0uu_<index> = OpBitwiseAnd %uint %t0u_<index> %uint_0xffffe000
    %t0f_<index> = OpBitcast %float %t0uu_<index>
    %t1u_<index> = OpBitcast %uint %t1_<index>
    %t1uu_<index> = OpBitwiseAnd %uint %t1u_<index> %uint_0xffffe000
    %t1f_<index> = OpBitcast %float %t1uu_<index>
    %t2_<index> = OpCompositeConstruct %v2float %t0f_<index> %t1f_<index>
    %t3_<index> = OpExtInst %uint %GLSL_std_450 PackHalf2x16 %t2_<index>
    %t4_<index> = OpBitcast %float %t3_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %t4_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VMbcntHiU32B32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp
/// L5455).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_vmbcnt_hi_u32_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VMbcntHiU32B32_SVdstSVsrc0SVsrc1";
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

    if !operand_load_float(spirv, inst.src[1], "t1_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }

    // TODO() check VSKIP

    const TEXT: &str = r#"
	    <load0>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %t1_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
	"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_VMbcntLoU32B32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp
/// L5497).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_vmbcnt_lo_u32_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VMbcntLoU32B32_SVdstSVsrc0SVsrc1";
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

    if !operand_load_float(spirv, inst.src[1], "t1_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }

    // TODO() check VSKIP

    const TEXT: &str = r#"
	    <load0>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %t1_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
	"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_V_XXX_B32_SVdstSVsrc0` (ShaderSpirv.cpp L5538).
/// XXX: Bfrev, Not (via `param`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_v_xxx_b32_svdst_svsrc0(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_V_XXX_B32_SVdstSVsrc0";
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

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }

    // TODO() check VSKIP

    const TEXT: &str = r#"
              <load0>
              <param0>
              %tf_<index> = OpBitcast %float %t_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %tf_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp L5740).
/// XXX: And, Or, Xor, Bcnt, Bfm, Lshr, Lshl, Lshlrev, Lshrrev, MulU32U24,
/// MulLoU32, MulHiU32 (via `param`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_v_xxx_b32_svdst_svsrc01(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_V_XXX_B32_SVdstSVsrc0SVsrc1";
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
    let mut load1 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }

    // TODO() check VSKIP

    const TEXT: &str = r#"
              <load0>
              <load1>
              <param0>
              <param1>
              <param2>
              %tf_<index> = OpBitcast %float %t_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %tf_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<param2>", param[2].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_V_XXX_I32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp L5795).
/// XXX: Ashr, Ashrrev, MulLo (via `param`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_v_xxx_i32_svdst_svsrc01(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_V_XXX_I32_SVdstSVsrc0SVsrc1";
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
    let mut load1 = String::new();

    if !operand_load_int(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_int(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }

    // TODO() check VSKIP

    const TEXT: &str = r#"
              <load0>
              <load1>
              <param0>
              <param1>
              %tf_<index> = OpBitcast %float %t_<index>
              OpStore %<dst> %tf_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %tf_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param0>", param[0].unwrap_or(""))
        .replace("<param1>", param[1].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_V_XXX_U32_VdstVsrc0Vsrc1Vsrc2` (ShaderSpirv.cpp L5940).
/// XXX: Sad, Bfe, MadU32U24 (via `param`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_v_xxx_u32_vdst_vsrc012(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_V_XXX_U32_VdstVsrc0Vsrc1Vsrc2";
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
    let mut load1 = String::new();
    let mut load2 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[2], "t2_<index>", &index_str, &mut load2, -1)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() Sad: use only lower 16 bits of Vaccum

    const TEXT: &str = r#"
               <load0>
               <load1>
               <load2>
               <param0>
               <param1>
               <param2>
               <param3>
         %tf_<index> = OpBitcast %float %t_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %tf_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
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

/// Kyty: `Recompile_V_XXX_U32_VdstSdst2Vsrc0Vsrc1` (ShaderSpirv.cpp L6005).
/// XXX: Add, Sub, Subrev (via `param`; carry-out goes to `dst2`).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn recompile_v_xxx_u32_vdst_sdst2_vsrc01(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_V_XXX_U32_VdstSdst2Vsrc0Vsrc1";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }
    if !operand_is_variable(inst.dst2) {
        return Err(not_supported(FUNC, "dst2 is not a variable"));
    }

    let dst_value = operand_variable_to_str(inst.dst);
    let dst2_value0 = operand_variable_to_str_shift(inst.dst2, 0);
    let dst2_value1 = operand_variable_to_str_shift(inst.dst2, 1);

    if operand_is_exec(inst.dst2) {
        return Err(not_supported(FUNC, "exec dst2"));
    }

    if dst_value.type_ != SpirvType::Float || dst2_value0.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "unexpected operand types"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }

    // TODO() check VSKIP
    // TODO() check EXEC

    const TEXT: &str = r#"
              <load0>
              <load1>
        <param>
        %t208_<index> = OpCompositeExtract %uint %t_<index> 1
        %t209_<index> = OpCompositeExtract %uint %t_<index> 0
        %t210_<index> = OpBitcast %float %t209_<index>
               OpStore %<dst> %t210_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_hi_u_<index> = OpLoad %uint %exec_hi ; unused
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %t213_<index> = OpSelect %uint %exec_lo_b_<index> %t208_<index> %uint_0
               OpStore %<dst2_0> %t213_<index>
               OpStore %<dst2_1> %uint_0
"#;
    *dst_source += &TEXT
        .replace("<dst>", &dst_value.value)
        .replace("<dst2_0>", &dst2_value0.value)
        .replace("<dst2_1>", &dst2_value1.value)
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<param>", param[0].unwrap_or(""))
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_Inject_Debug` (ShaderSpirv.cpp L6131) — not a table row;
/// called from `Spirv::WriteInstructions` (L7834) after each recompiled
/// instruction when debug printf is enabled.
pub(crate) fn recompile_inject_debug(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    use crate::shader::types::ShaderDebugPrintfType;

    const FUNC: &str = "Recompile_Inject_Debug";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    let mut injected = false;
    for (str_id, c) in code.get_debug_printfs().iter().enumerate() {
        if c.pc == inst.pc {
            if c.args.len() != c.types.len() {
                return Err(not_supported(FUNC, "args/types size mismatch"));
            }
            let mut loads: Vec<String> = Vec::new();
            let mut params_ids: Vec<String> = Vec::new();
            for (arg_id, a) in c.args.iter().enumerate() {
                let type_ = c.types[arg_id];
                let result_id = format!("t_{arg_id}_<index>");
                let mut load = String::new();
                let ok = match type_ {
                    ShaderDebugPrintfType::Uint => {
                        operand_load_uint(spirv, *a, &result_id, &index_str, &mut load, -1)?
                    }
                    ShaderDebugPrintfType::Int => {
                        operand_load_int(spirv, *a, &result_id, &index_str, &mut load)?
                    }
                    ShaderDebugPrintfType::Float => {
                        operand_load_float(spirv, *a, &result_id, &index_str, &mut load)?
                    }
                };
                if !ok {
                    return Err(not_supported(FUNC, "can't load printf argument"));
                }
                loads.push(load);
                params_ids.push(format!("%{result_id}"));
            }

            const TEXT: &str = r#"
                <loads>
     %tt_<index> = OpExtInst %void %NonSemantic_DebugPrintf 1 %printf_str_<str_id> <params>
		"#;
            *dst_source += &TEXT
                .replace("<loads>", &loads.join("\n"))
                .replace("<str_id>", &format!("{str_id}"))
                .replace("<params>", &params_ids.join(" "))
                .replace("<index>", &index_str);
            injected = true;
        }
    }

    Ok(injected)
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

/// Table row backed by a ported function that needs an SCC check.
///
/// The counterpart of [`f`] for the `S_*` scalar ops: Kyty's table carries a
/// `SccCheck` per row (`Recompile_S_Lshl_B64_*` is `SCC_CHECK_NONZERO`), and the
/// recompiler reads it to emit the right `scc` update. `f` hardcodes
/// `SccCheck::None`, so wiring an SCC-bearing function through it would silently
/// drop that update and produce a shader whose `scc` never changes.
const fn fs(
    func: InstRecompileFn,
    type_: ShaderInstructionType,
    format: Format,
    param: Params,
    scc_check: SccCheck,
) -> RecompilerFunc {
    RecompilerFunc {
        func: RecompileFn::Func(func),
        type_,
        format,
        param,
        scc_check,
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
    fs(recompile_s_lshl_b64, T::SLshlB64, F::Sdst2Ssrc02Ssrc1, p2("", ""), S::NonZero),
    fs(recompile_s_lshr_b64, T::SLshrB64, F::Sdst2Ssrc02Ssrc1, p2("", ""), S::NonZero),

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
        // Each row wired from the staged set must move these two numbers and
        // arrive with a per-opcode test — see `s_lshl_b64_is_wired_and_shifts`.
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
        assert_eq!(implemented, 79, "C1 implemented subset");
        assert_eq!(ni, 125, "C2 remainder");

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

    /// The two 64-bit scalar shifts are wired, and wired **with their SCC
    /// check intact**.
    ///
    /// Both were in the staged-but-unwired set (`#[allow(dead_code)]`): the
    /// bodies existed and compiled, but no `(type, format)` row pointed at
    /// them, so the recompiler reported them `NotImplemented` and any shader
    /// using `S_LSHL_B64` failed. Wiring is not just flipping the row — Kyty
    /// gives these `SCC_CHECK_NONZERO`, and routing them through `f()` (which
    /// hardcodes `SccCheck::None`) would compile, pass a "is it Func?" check,
    /// and silently never update `scc`. Hence `fs()`, and hence this test
    /// asserts the check rather than only the wiring.
    #[test]
    fn s_shift_b64_rows_are_wired_with_their_scc_check() {
        for (ty, name) in [(T::SLshlB64, "S_LSHL_B64"), (T::SLshrB64, "S_LSHR_B64")] {
            let e = recomp_func(ty, F::Sdst2Ssrc02Ssrc1)
                .unwrap_or_else(|| panic!("{name} must have a table row"));
            assert!(
                matches!(e.func, RecompileFn::Func(_)),
                "{name} is staged and must be wired, not NotImplemented"
            );
            assert_eq!(
                e.scc_check,
                SccCheck::NonZero,
                "{name} carries Kyty's SCC_CHECK_NONZERO; dropping it silently \
                 leaves scc never updated"
            );
        }
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
