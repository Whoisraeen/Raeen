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

/// `Recompile_BufferLoadDwordX4_Vdata4VaddrSvSoffsIdxen` — beyond Kyty (it
/// NIs the opcode). Four consecutive dword loads at `(offset + vindex*stride)/4
/// + i`, plus the per-thread `voffset` register when the Offen format is in
/// play — measured on Minecraft's menu VS (`v[8:11], v[4:5], s[8:11]`).
fn recompile_buffer_load_dwordx4(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_BufferLoadDwordX4_Vdata4VaddrSvSoffsIdxen";
    let inst = inst_at(code, index, FUNC)?;

    let Some(bind_info) = spirv.get_bind_info() else {
        return Ok(false);
    };
    if bind_info.storage_buffers.buffers_num == 0 {
        return Ok(false);
    }
    if !operand_is_constant(inst.src[2]) {
        return Err(not_supported(FUNC, "src2 is not a constant"));
    }

    let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
    let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
    let offset = spirv.get_constant(inst.src[2]);

    if src1_value0.type_ != SpirvType::Uint || src1_value1.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "unexpected operand types"));
    }

    let idxen = matches!(
        inst.format,
        Format::Vdata4VaddrSvSoffsIdxen | Format::Vdata4Vaddr2SvSoffsOffenIdxen
    );
    let offen = matches!(
        inst.format,
        Format::Vdata4Vaddr2SvSoffsOffenIdxen | Format::Vdata4VaddrSvSoffsOffen
    );
    let src0_index = idxen.then(|| operand_variable_to_str(inst.src[0]));
    let src0_off = offen.then(|| operand_variable_to_str_shift(inst.src[0], i32::from(idxen)));
    if src0_index
        .as_ref()
        .is_some_and(|value| value.type_ != SpirvType::Float)
    {
        return Err(not_supported(FUNC, "unexpected index register type"));
    }

    let index_str = format!("{index}");
    let index_setup = src0_index.map_or_else(
        || "               OpStore %temp_int_1 %int_0\n".to_owned(),
        |value| {
            format!(
                "        %t100_{index_str} = OpLoad %float %{src}\n        %t101_{index_str} = OpBitcast %int %t100_{index_str}\n               OpStore %temp_int_1 %t101_{index_str}\n",
                src = value.value,
            )
        },
    );
    let mut text = format!(
        r#"
{index_setup}
        %t148_{index_str} = OpLoad %uint %{src1_value1}
        %t150_{index_str} = OpShiftRightLogical %uint %t148_{index_str} %int_16
        %t152_{index_str} = OpBitwiseAnd %uint %t150_{index_str} %uint_0x00003fff
        %t153_{index_str} = OpBitcast %int %t152_{index_str}
               OpStore %temp_int_3 %t153_{index_str}
        %t155_{index_str} = OpLoad %uint %{src1_value0}
        %t156_{index_str} = OpBitcast %int %t155_{index_str}
               OpStore %temp_int_4 %t156_{index_str}
               OpStore %temp_int_2 %{offset}
"#,
        src1_value0 = src1_value0.value,
        src1_value1 = src1_value1.value,
        offset = offset,
    );

    // offen: the second vaddr register is the per-thread byte offset.
    if let Some(off) = &src0_off {
        if off.type_ != SpirvType::Float {
            return Err(not_supported(FUNC, "unexpected offen register type"));
        }
        text += &format!(
            r#"        %t160_{index_str} = OpLoad %float %{off}
        %t161_{index_str} = OpBitcast %int %t160_{index_str}
        %t162_{index_str} = OpLoad %int %temp_int_2
        %t163_{index_str} = OpIAdd %int %t162_{index_str} %t161_{index_str}
               OpStore %temp_int_2 %t163_{index_str}
"#,
            off = off.value,
        );
    }

    for i in 0..4 {
        let dst_value = operand_variable_to_str_shift(inst.dst, i);
        if dst_value.type_ != SpirvType::Float {
            return Err(not_supported(FUNC, "unexpected dst type"));
        }
        if i != 0 {
            text += &format!(
                r#"        %t164_{index_str}_{i} = OpLoad %int %temp_int_2
        %t165_{index_str}_{i} = OpIAdd %int %t164_{index_str}_{i} %int_4
               OpStore %temp_int_2 %t165_{index_str}_{i}
"#,
            );
        }
        text += &format!(
            "        %t110_{index_str}_{i} = OpFunctionCall %void %buffer_load_float1 %{p} %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4\n",
            p = dst_value.value,
        );
    }

    *dst_source += &text;
    Ok(true)
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

    // The `en` mask says which of the four channels this export writes. A full
    // export is 0xf; a vec2 texcoord is 0x3, a vec3 normal 0x7. The disabled
    // channels' `vsrc` fields are don't-care in the hardware encoding, so only
    // the enabled ones are required to be variables — the earlier version
    // demanded all four and rejected every partial export.
    let en = inst.export_enable;
    let enabled = |i: usize| en & (1 << i) != 0;

    // `find_constants` always registers 0.0, so `%float_0_000000` is available
    // to seed the disabled channels; the param slot is then fully defined and
    // the consuming PS reads only the channels it declared.
    let mut loads = String::new();
    let mut channels: [String; 4] = Default::default();
    for (i, channel) in channels.iter_mut().enumerate() {
        if enabled(i) {
            if !operand_is_variable(inst.src[i]) {
                return Err(not_supported(
                    FUNC,
                    "enabled export source is not a variable",
                ));
            }
            let value = operand_variable_to_str(inst.src[i]).value;
            loads += &format!("         %t{i}_<index> = OpLoad %float %{value}\n");
            *channel = format!("%t{i}_<index>");
        } else {
            *channel = "%float_0_000000".to_owned();
        }
    }

    // TODO() check VSKIP
    // TODO() check EXEC

    let text = format!(
        "{loads}         %t4_<index> = OpCompositeConstruct %v4float {} {} {} {}\n               OpStore %<param> %t4_<index>\n",
        channels[0], channels[1], channels[2], channels[3]
    );

    *dst_source += &text
        .replace("<index>", &format!("{index}"))
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

    // Coverage probe (XPS5X_TRACE_DRAWS). XPS5X_FORCE_CLEAR proved a correct
    // full-screen NDC quad rasterizes ZERO fragments, so the suspect is the
    // position export. If this never fires for a title's VS, the shader never
    // writes gl_Position at all and nothing can cover a pixel.
    if std::env::var_os("XPS5X_TRACE_DRAWS").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static POS0_SEEN: AtomicU32 = AtomicU32::new(0);
        let n = POS0_SEEN.fetch_add(1, Ordering::Relaxed);
        if n < 8 {
            tracing::warn!(
                n,
                srcs_are_variables = inst.src[..4].iter().all(|s| operand_is_variable(*s)),
                // The VGPRs the export READS. The fetch WRITES to the VGPR named
                // by sem.hardware_mapping() (measured: 9, 12, 13). If these
                // disagree the export reads never-written registers (zeros) and
                // every vertex collapses to the origin — zero coverage.
                src_regs = format_args!(
                    "[{}, {}, {}, {}]",
                    inst.src[0].register_id,
                    inst.src[1].register_id,
                    inst.src[2].register_id,
                    inst.src[3].register_id
                ),
                src_types = format_args!(
                    "[{:?}, {:?}, {:?}, {:?}]",
                    inst.src[0].type_, inst.src[1].type_, inst.src[2].type_, inst.src[3].type_
                ),
                "TRACE_DRAWS: POS0 export recompiled (VS writes gl_Position)"
            );
        }
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

    // DIAGNOSTIC (XPS5X_VS_PASSTHROUGH=1): bypass the VS arithmetic and export
    // input attribute 0 directly as the clip position. Every INPUT is verified
    // correct (the measured vertex buffer is a textbook NDC quad) and Vulkan
    // reports ZERO validation messages, yet no primitive covers a pixel — so
    // the suspect is the VALUES the translated VS computes. Pixels under this
    // flag => the generated VS body is at fault; still black => the fault is
    // below the shader. Assumes a vec3 attr0 (the measured stride-12 quad);
    // other shapes will fail to assemble, which is fine for a gated probe.
    if std::env::var_os("XPS5X_VS_PASSTHROUGH").is_some() {
        const PASS: &str = r#"
         %p0_<index> = OpLoad %v3float %attr0
         %px_<index> = OpCompositeExtract %float %p0_<index> 0
         %py_<index> = OpCompositeExtract %float %p0_<index> 1
         %pz_<index> = OpCompositeExtract %float %p0_<index> 2
         %pv_<index> = OpCompositeConstruct %v4float %px_<index> %py_<index> %pz_<index> %float_1_000000
         %pa_<index> = OpAccessChain %_ptr_Output_v4float %outPerVertex %int_per_vertex_0
               OpStore %pa_<index> %pv_<index>
"#;
        *dst_source += &PASS.replace("<index>", &format!("{index}"));
        return Ok(true);
    }

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
/// Kyty upstream carries only the embedded-fetch gate here; the extended
/// (EUD) path is a beyond-Kyty addition measured on Minecraft's menu VS
/// (`s_load_dwordx2 s[82:83], s[14:15], 8`), modelled on the x4/x8 path.
fn recompile_sload_dwordx2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SLoadDwordx2_Sdst2Ssrc02Ssrc1";
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

    sload_dword_extended(index, &inst, dst_source, spirv, 2, FUNC)
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

/// RDNA2 `s_getpc_b64`: write the absolute address of the following guest
/// instruction. The parser materializes the low/high dwords from
/// [`ShaderCode::get_base_address`], so this remains correct for shaders above
/// 4 GiB instead of truncating to a parser-relative PC.
fn recompile_s_getpc_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SGetpcB64_Sdst2";
    let inst = inst_at(code, index, FUNC)?;
    if !operand_is_variable(inst.dst)
        || !operand_is_constant(inst.src[0])
        || !operand_is_constant(inst.src[1])
    {
        return Err(not_supported(FUNC, "unexpected operand kinds"));
    }
    let dst_lo = operand_variable_to_str_shift(inst.dst, 0);
    let dst_hi = operand_variable_to_str_shift(inst.dst, 1);
    if dst_lo.type_ != SpirvType::Uint || dst_hi.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "destination is not an SGPR pair"));
    }
    let low = spirv.get_constant(inst.src[0]);
    let high = spirv.get_constant(inst.src[1]);
    *dst_source += &format!(
        "               OpStore %{lo} %{low}\n               OpStore %{hi} %{high}\n",
        lo = dst_lo.value,
        hi = dst_hi.value,
    );
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

/// RDNA/GCN cubemap coordinate helpers `V_CUBEID/SC/TC/MA_F32` (VOP3
/// 0x144-0x147). No Kyty upstream (`EXIT_NOT_IMPLEMENTED`); formulas ported
/// from shadPS4 `vector_alu.cpp` (`V_CUBE*_F32` + `SelectCubeResult`). Given a
/// direction (x=src0, y=src1, z=src2) the major axis is the component with the
/// largest magnitude; each op emits the value belonging to that face:
///   * ID: face index 0..5
///   * SC/TC: the S / T face coordinate (pre-divide, pre-bias)
///   * MA: 2 * major-axis component (the divisor)
fn recompile_v_cube_f32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCubeF32_VdstVsrc0Vsrc1Vsrc2";
    let inst = inst_at(code, index, FUNC)?;

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

    let index_str = format!("{index}");
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

    // Constants are pre-declared by `Spirv::find_constants`.
    let c0 = format!("%{}", spirv.get_constant_float(0.0));
    let c1 = format!("%{}", spirv.get_constant_float(1.0));
    let c2 = format!("%{}", spirv.get_constant_float(2.0));
    let c3 = format!("%{}", spirv.get_constant_float(3.0));
    let c4 = format!("%{}", spirv.get_constant_float(4.0));
    let c5 = format!("%{}", spirv.get_constant_float(5.0));

    // Per-op result triple (%xr/%yr/%zr): the value chosen when x / y / z is
    // the major axis. `%xneg/%yneg/%zneg` (component < 0) are defined by the
    // shared prologue below, before this block is spliced in.
    let per_op: &str = match inst.type_ {
        ShaderInstructionType::VCubeIdF32 => {
            "%xr_<index> = OpSelect %float %xneg_<index> <c1> <c0>\n\
             %yr_<index> = OpSelect %float %yneg_<index> <c3> <c2>\n\
             %zr_<index> = OpSelect %float %zneg_<index> <c5> <c4>"
        }
        ShaderInstructionType::VCubeScF32 => {
            "%negz_<index> = OpFNegate %float %t2_<index>\n\
             %negx_<index> = OpFNegate %float %t0_<index>\n\
             %xr_<index> = OpSelect %float %xneg_<index> %t2_<index> %negz_<index>\n\
             %yr_<index> = OpFMul %float %t0_<index> <c1>\n\
             %zr_<index> = OpSelect %float %zneg_<index> %negx_<index> %t0_<index>"
        }
        ShaderInstructionType::VCubeTcF32 => {
            "%negy_<index> = OpFNegate %float %t1_<index>\n\
             %negz_<index> = OpFNegate %float %t2_<index>\n\
             %xr_<index> = OpFMul %float %negy_<index> <c1>\n\
             %yr_<index> = OpSelect %float %yneg_<index> %negz_<index> %t2_<index>\n\
             %zr_<index> = OpFMul %float %negy_<index> <c1>"
        }
        ShaderInstructionType::VCubeMaF32 => {
            "%xr_<index> = OpFMul %float %t0_<index> <c2>\n\
             %yr_<index> = OpFMul %float %t1_<index> <c2>\n\
             %zr_<index> = OpFMul %float %t2_<index> <c2>"
        }
        _ => return Err(not_supported(FUNC, "not a cube opcode")),
    };

    // Shared: |x|,|y|,|z|; z is major when |z|>=|x| && |z|>=|y|, else y when
    // |y|>=|x|, else x. Store the chosen result under EXEC (lane 0 mask).
    const TEXT: &str = r#"
              <load0>
              <load1>
              <load2>
        %xneg_<index> = OpFOrdLessThan %bool %t0_<index> <c0>
        %yneg_<index> = OpFOrdLessThan %bool %t1_<index> <c0>
        %zneg_<index> = OpFOrdLessThan %bool %t2_<index> <c0>
              <per_op>
        %absx_<index> = OpExtInst %float %GLSL_std_450 FAbs %t0_<index>
        %absy_<index> = OpExtInst %float %GLSL_std_450 FAbs %t1_<index>
        %absz_<index> = OpExtInst %float %GLSL_std_450 FAbs %t2_<index>
        %zc1_<index> = OpFOrdGreaterThanEqual %bool %absz_<index> %absx_<index>
        %zc2_<index> = OpFOrdGreaterThanEqual %bool %absz_<index> %absy_<index>
        %zcond_<index> = OpLogicalAnd %bool %zc1_<index> %zc2_<index>
        %ycond_<index> = OpFOrdGreaterThanEqual %bool %absy_<index> %absx_<index>
        %inner_<index> = OpSelect %float %ycond_<index> %yr_<index> %xr_<index>
        %t_<index> = OpSelect %float %zcond_<index> %zr_<index> %inner_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %t_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<load2>", &load2)
        .replace("<per_op>", per_op)
        .replace("<c0>", &c0)
        .replace("<c1>", &c1)
        .replace("<c2>", &c2)
        .replace("<c3>", &c3)
        .replace("<c4>", &c4)
        .replace("<c5>", &c5)
        .replace("<dst>", &dst_value.value)
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

        // Resolve by SEMANTIC (attrib-table index), not by array position.
        // `shader_parse_attrib` appends resources in discovery order while
        // `attrib_id` here is an attrib-table dword index, so the two agree
        // only when the semantics table is identity-mapped. Minecraft's is not
        // (measured: positions 0,1,2 carry semantics 0,2,3), and the old
        // by-position read therefore returned a DIFFERENT attribute's V# for
        // one id and an unwritten slot (registers_num 0) for another — one
        // silently wrong binding and one loud failure from the same gap.
        let resolved = info.resources_dst
            [..(info.resources_num.max(0) as usize).min(info.resources_dst.len())]
            .iter()
            .position(|d| d.semantic == attrib_id);
        let Some(r) = resolved.map(|p| info.resources_dst[p]) else {
            return Err(not_supported(
                FUNC,
                format!("attrib id {attrib_id} not in the semantics table"),
            ));
        };
        let attrib_pos = resolved.expect("resolved is Some in this branch");

        let n_attr = r.registers_num;
        let n_dst = inst.dst.size;
        if !(1..=4).contains(&n_attr) {
            return Err(not_supported(
                FUNC,
                format!("invalid registers_num: {n_attr} (attrib {attrib_id})"),
            ));
        }
        if !(1..=4).contains(&n_dst) {
            return Err(not_supported(
                FUNC,
                format!("invalid fetch dst.size: {n_dst} (attrib {attrib_id})"),
            ));
        }

        // GCN vertex fetch tolerates either direction of width mismatch:
        // channels beyond the attribute read back as the (0,0,0,1) default;
        // channels beyond the fetch are dropped into a scratch. Beyond Kyty
        // (upstream EXITs on any mismatch). Measured on Minecraft's menu VS:
        // attrib 2 as 2ch feeding a vec3 (fill z=0.0) and as 4ch (drop w).
        let (temp_ty, load_ty, helper) = match n_attr {
            1 => ("%temp_float", "%float", "%fetch_f1_f1_"),
            2 => ("%temp_v2float", "%v2float", "%fetch_f1_f1_vf2_"),
            3 => ("%temp_v3float", "%v3float", "%fetch_f1_f1_f1_vf3_"),
            _ => ("%temp_v4float", "%v4float", "%fetch_f1_f1_f1_f1_vf4_"),
        };
        let mut params = String::new();
        for i in 0..n_attr {
            if i < n_dst {
                params += &format!("%v{} ", inst.dst.register_id + i);
            } else {
                // A wider attribute's dropped channels land in the scratch
                // (function-scope float var; the helper writes them in order).
                params += "%temp_float ";
            }
        }
        let mut text = format!(
            "
        %t1_<index> = OpLoad {load_ty} %<attr>
                       OpStore {temp_ty} %t1_<index>
        %t2_<index> = OpFunctionCall %void {helper} {params}{temp_ty}
",
        );
        for i in n_attr..n_dst {
            let default = if i == 3 {
                "%float_1_000000"
            } else {
                "%float_0_000000"
            };
            text += &format!(
                "               OpStore %v{} {default}
",
                inst.dst.register_id + i
            );
        }

        // `%attrN` is DECLARED and decorated by array POSITION
        // (`Spirv::WriteGlobalVariables` emits `%attr{i}` / `OpDecorate
        // %attr{i} Location {i}` over `0..resources_num`, and the host binds
        // attributes by position too). So the variable reference must use the
        // resolved position, not the attrib-table id — they differ exactly
        // when the semantics table is gapped. `<index>` keeps the id so SSA
        // temporaries stay unique per fetch.
        *dst_source += &text
            .replace("<index>", &format!("{attrib_id}_{index}"))
            .replace("<attr>", &format!("attr{attrib_pos}"));

        return Ok(true);
    }

    Ok(false)
}

/// Kyty: `Recompile_BufferStoreDword_Vdata1VaddrSvSoffsIdxen`
/// (ShaderSpirv.cpp L1999). Wired for the Minecraft menu CS's output store.
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
<coord>         %t43_<index> = OpImageSampleImplicitLod %v4float %t38_<index> %t42_<index>
               OpStore %temp_v4float %t43_<index>
"#;
            const TAIL: &str = r#"         %t<t0>_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_<chan>
         %t<t1>_<index> = OpLoad %float %t<t0>_<index>
               OpStore %<dst_value> %t<t1>_<index>
"#;

            // Cube textures sample with a 3-component direction; 2D with 2.
            // The Dim was decided from the measured T# types in WriteTypes.
            let bound = usize::try_from(bind_info.textures2d.textures_num)
                .unwrap_or(0)
                .min(bind_info.textures2d.desc.len());
            let cube = bound > 0
                && bind_info.textures2d.desc[..bound]
                    .iter()
                    .all(|d| d.texture.type_() == 11);
            let coord = if cube {
                let src0_value2 = operand_variable_to_str_shift(inst.src[0], 2);
                if src0_value2.type_ != SpirvType::Float {
                    return Err(not_supported(func, "unexpected cube coord type"));
                }
                format!(
                    "         %t39_<index> = OpLoad %float %{}
         %t40_<index> = OpLoad %float %{}
         %t41_<index> = OpLoad %float %{}
         %t42_<index> = OpCompositeConstruct %v3float %t39_<index> %t40_<index> %t41_<index>
",
                    src0_value0.value, src0_value1.value, src0_value2.value
                )
            } else {
                format!(
                    "         %t39_<index> = OpLoad %float %{}
         %t40_<index> = OpLoad %float %{}
         %t42_<index> = OpCompositeConstruct %v2float %t39_<index> %t40_<index>
",
                    src0_value0.value, src0_value1.value
                )
            };
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
                .replace("<coord>", &coord)
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

/// Lower Gen5 `image_sample_c_lz` through the existing non-depth image type.
///
/// XPS5X currently declares sampled textures with `Depth = 0`, so SPIR-V's
/// depth-reference sampling opcodes are not legal for these descriptors. The
/// equivalent lowering used by SharpEmu samples red at LOD zero, evaluates
/// `reference <= texel`, and materializes `(compare, compare, compare, 1)`
/// before applying the instruction's dmask.
fn recompile_image_sample_c_lz(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleCLz_VdataVaddr3StSs";
    let inst = inst_at(code, index, FUNC)?;

    let Some(bind_info) = spirv.get_bind_info() else {
        return Ok(false);
    };
    if bind_info.textures2d.textures2d_sampled_num == 0 || bind_info.samplers.samplers_num == 0 {
        return Ok(false);
    }

    let bound = usize::try_from(bind_info.textures2d.textures_num)
        .unwrap_or(0)
        .min(bind_info.textures2d.desc.len());
    let cube = bound > 0
        && bind_info.textures2d.desc[..bound]
            .iter()
            .all(|d| d.texture.type_() == 11);
    if cube {
        return Err(not_supported(FUNC, "comparison sampling of cube textures"));
    }

    let reference = operand_variable_to_str_shift(inst.src[0], 0);
    let coord_x = operand_variable_to_str_shift(inst.src[0], 1);
    let coord_y = operand_variable_to_str_shift(inst.src[0], 2);
    let texture_index = operand_variable_to_str_shift(inst.src[1], 0);
    let sampler_index = operand_variable_to_str_shift(inst.src[2], 0);
    let dst0 = operand_variable_to_str_shift(inst.dst, 0);
    if dst0.type_ != SpirvType::Float
        || reference.type_ != SpirvType::Float
        || coord_x.type_ != SpirvType::Float
        || coord_y.type_ != SpirvType::Float
        || texture_index.type_ != SpirvType::Uint
        || sampler_index.type_ != SpirvType::Uint
    {
        return Err(not_supported(FUNC, "unexpected operand types"));
    }

    const HEAD: &str = r#"
         %clz_texture_index_<index> = OpLoad %uint %<texture_index>
         %clz_texture_ptr_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %clz_texture_index_<index>
         %clz_texture_<index> = OpLoad %ImageS %clz_texture_ptr_<index>
         %clz_sampler_index_<index> = OpLoad %uint %<sampler_index>
         %clz_sampler_ptr_<index> = OpAccessChain %_ptr_UniformConstant_Sampler %samplers %clz_sampler_index_<index>
         %clz_sampler_<index> = OpLoad %Sampler %clz_sampler_ptr_<index>
         %clz_sampled_image_<index> = OpSampledImage %SampledImage %clz_texture_<index> %clz_sampler_<index>

         %clz_reference_<index> = OpLoad %float %<reference>
         %clz_x_<index> = OpLoad %float %<coord_x>
         %clz_y_<index> = OpLoad %float %<coord_y>
         %clz_coord_<index> = OpCompositeConstruct %v2float %clz_x_<index> %clz_y_<index>
         %clz_sample_<index> = OpImageSampleExplicitLod %v4float %clz_sampled_image_<index> %clz_coord_<index> Lod %float_0_000000
         %clz_texel_<index> = OpCompositeExtract %float %clz_sample_<index> 0
         %clz_passes_<index> = OpFOrdLessThanEqual %bool %clz_reference_<index> %clz_texel_<index>
         %clz_result_<index> = OpSelect %float %clz_passes_<index> %float_1_000000 %float_0_000000
"#;
    *dst_source += &HEAD
        .replace("<texture_index>", &texture_index.value)
        .replace("<sampler_index>", &sampler_index.value)
        .replace("<reference>", &reference.value)
        .replace("<coord_x>", &coord_x.value)
        .replace("<coord_y>", &coord_y.value)
        .replace("<index>", &format!("{index}"));

    let values: &[&str] = match inst.format {
        F::Vdata1Vaddr3StSsDmask1 => &["%clz_result_<index>"],
        F::Vdata1Vaddr3StSsDmask8 => &["%float_1_000000"],
        F::Vdata2Vaddr3StSsDmask3 | F::Vdata2Vaddr3StSsDmask5 => {
            &["%clz_result_<index>", "%clz_result_<index>"]
        }
        F::Vdata2Vaddr3StSsDmask9 => &["%clz_result_<index>", "%float_1_000000"],
        F::Vdata3Vaddr3StSsDmask7 => &[
            "%clz_result_<index>",
            "%clz_result_<index>",
            "%clz_result_<index>",
        ],
        F::Vdata4Vaddr3StSsDmaskF => &[
            "%clz_result_<index>",
            "%clz_result_<index>",
            "%clz_result_<index>",
            "%float_1_000000",
        ],
        _ => return Err(not_supported(FUNC, "unsupported dmask format")),
    };
    for (shift, value) in values.iter().enumerate() {
        let dst = operand_variable_to_str_shift(inst.dst, shift as i32);
        if dst.type_ != SpirvType::Float {
            return Err(not_supported(FUNC, "unexpected destination type"));
        }
        *dst_source += &format!(
            "               OpStore %{} {}\n",
            dst.value,
            value.replace("<index>", &format!("{index}"))
        );
    }

    Ok(true)
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

/// Beyond-Kyty: `image_sample_lz` with `dmask == 0xf` (rgba, explicit LOD 0),
/// measured in ASTRO.BOT's fullscreen composite. Same lowering as
/// [`recompile_image_sample_lz_dmask7`] plus the fourth channel store.
fn recompile_image_sample_lz_dmask_f(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLz_Vdata4Vaddr3StSsDmaskF";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_sampled_num > 0 && bind_info.samplers.samplers_num > 0 {
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let dst_value3 = operand_variable_to_str_shift(inst.dst, 3);
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
         %t57_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_3
         %t58_<index> = OpLoad %float %t57_<index>
               OpStore %<dst_value3> %t58_<index>
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value)
                .replace("<dst_value3>", &dst_value3.value);

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

/// ASTRO.BOT's `image_get_resinfo` dmask=xy form. Query the selected mip's
/// width/height and write their raw u32 values into consecutive VGPRs.
/// Reference semantics: shadPS4 `Translator::IMAGE_GET_RESINFO` (GPL-2.0).
fn recompile_image_get_resinfo_dmask3(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageGetResinfo_Vdata2VaddrStDmask3";
    let inst = inst_at(code, index, FUNC)?;
    let Some(bind_info) = spirv.get_bind_info() else {
        return Ok(false);
    };
    if bind_info.textures2d.textures2d_sampled_num == 0 {
        return Ok(false);
    }

    let dst_x = operand_variable_to_str_shift(inst.dst, 0);
    let dst_y = operand_variable_to_str_shift(inst.dst, 1);
    let lod = operand_variable_to_str(inst.src[0]);
    let texture = operand_variable_to_str_shift(inst.src[1], 0);
    if dst_x.type_ != SpirvType::Float
        || dst_y.type_ != SpirvType::Float
        || lod.type_ != SpirvType::Float
        || texture.type_ != SpirvType::Uint
    {
        return Err(not_supported(FUNC, "unexpected operand types"));
    }

    let index_str = format!("{index}");
    const TEXT: &str = r#"
         %t0_<index> = OpLoad %uint %<texture>
         %t1_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %t0_<index>
         %t2_<index> = OpLoad %ImageS %t1_<index>
         %t3_<index> = OpLoad %float %<lod>
         %t4_<index> = OpBitcast %int %t3_<index>
         %t5_<index> = OpImageQuerySizeLod %v2int %t2_<index> %t4_<index>
         %t6_<index> = OpCompositeExtract %int %t5_<index> 0
         %t7_<index> = OpBitcast %float %t6_<index>
               OpStore %<dst_x> %t7_<index>
         %t8_<index> = OpCompositeExtract %int %t5_<index> 1
         %t9_<index> = OpBitcast %float %t8_<index>
               OpStore %<dst_y> %t9_<index>
"#;
    *dst_source += &TEXT
        .replace("<index>", &index_str)
        .replace("<texture>", &texture.value)
        .replace("<lod>", &lod.value)
        .replace("<dst_x>", &dst_x.value)
        .replace("<dst_y>", &dst_y.value);
    Ok(true)
}

/// Shared body of the `ImageLoad` dmask family: fetch a texel by integer
/// coordinate into `temp_v4float`, then store the dmask-selected `channels` —
/// `(t0, chan)` pairs naming the temp id and the `%uint_<chan>` component —
/// into consecutive vdata registers. Same channel scheme as
/// [`image_sample_channels`].
fn image_load_channels(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    channels: &[(u32, u32)],
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_sampled_num > 0 {
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);

            if dst_value0.type_ != SpirvType::Float
                || src0_value0.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
            {
                return Err(not_supported(func, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check LOD_CLAMPED
            // TODO() swizzle channels
            // TODO() convert SRGB -> LINEAR if SRGB format was replaced with UNORM

            const HEAD: &str = r#"
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
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src1_value0>", &src1_value0.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_ImageLoad_Vdata4Vaddr3StDmaskF` (ShaderSpirv.cpp L3038).
fn recompile_image_load_dmask_f(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageLoad_Vdata4Vaddr3StDmaskF";
    image_load_channels(
        index,
        code,
        dst_source,
        spirv,
        FUNC,
        &[(46, 0), (50, 1), (54, 2), (57, 3)],
    )
}

/// Beyond-Kyty: `image_load` with `dmask == 0x1` (single-channel fetch),
/// measured in ASTRO.BOT scene compute shaders.
fn recompile_image_load_dmask1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageLoad_Vdata1Vaddr3StDmask1";
    image_load_channels(index, code, dst_source, spirv, FUNC, &[(46, 0)])
}

/// Beyond-Kyty: `image_load` with `dmask == 0x7` (xyz fetch), measured in
/// ASTRO.BOT scene compute shaders.
fn recompile_image_load_dmask7(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageLoad_Vdata3Vaddr3StDmask7";
    image_load_channels(
        index,
        code,
        dst_source,
        spirv,
        FUNC,
        &[(46, 0), (50, 1), (54, 2)],
    )
}

/// Kyty: `Recompile_ImageStore_Vdata4Vaddr3StDmaskF` (ShaderSpirv.cpp L3105).
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
/// XXX: Add, Addc, Bfe, Lshl4Add, MulHi, Sub (via `param`).
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

/// Beyond-Kyty: `s_orn2_saveexec_b64` — `sdst = exec; exec = ssrc0 | ~exec`.
/// The ORN2 sibling of [`recompile_s_and_saveexec_b64`]; measured in
/// ASTRO.BOT scene compute.
fn recompile_s_orn2_saveexec_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SOrn2SaveexecB64_Sdst2Ssrc02";
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
        %t192_<index> = OpNot %uint %t190_<index>
        %t193_<index> = OpNot %uint %t191_<index>
        %t194_<index> = OpBitwiseOr %uint %t0_<index> %t192_<index>
               OpStore %exec_lo %t194_<index>
        %t197_<index> = OpBitwiseOr %uint %t1_<index> %t193_<index>
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

/// `s_not_b64`: D.u64 = ~S0.u64, SCC = (D != 0). No Kyty upstream — added from
/// measurement (ASTRO.BOT compute shaders manipulating the exec mask). Structure
/// mirrors [`recompile_swqm_b64`], the other `Sdst2Ssrc02` unary: same
/// exec-passthrough shortcut, same paired dword load/store, same `<execz>`/
/// `<scc>` tail.
fn recompile_snot_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SNotB64_Sdst2Ssrc02";
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
        %t2_<index> = OpNot %uint %t0_<index>
        %t3_<index> = OpNot %uint %t1_<index>
               OpStore %<dst0> %t2_<index>
               OpStore %<dst1> %t3_<index>
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

/// `s_brev_b32`: D.u = bitreverse(S0.u), SCC untouched. No Kyty upstream —
/// added from measurement (ASTRO.BOT compute). SPIR-V has `OpBitReverse`, so
/// this is a direct single-op lowering.
fn recompile_sbrev_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SBrevB32_SVdstSVsrc0";
    let inst = inst_at(code, index, FUNC)?;

    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }

    let dst_value = operand_variable_to_str(inst.dst);
    if dst_value.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "dst is not uint"));
    }

    let mut load0 = String::new();
    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, 0)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
        <load0>
        %t1_<index> = OpBitReverse %uint %t0_<index>
               OpStore %<dst> %t1_<index>
"#;

    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<dst>", &dst_value.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// Kyty: `Recompile_SWqmB64_Sdst2Ssrc02` (ShaderSpirv.cpp L4621).
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
/// XXX: Gt, Ge, and the RDNA2-measured Lt (comparison via `param[0]`; also
/// writes EXEC).
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
/// MulLoU32, MulHiU32 (via `param`) — plus the RDNA2-only carry-less
/// add/sub family, which has no Kyty upstream rows.
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
/// XXX: Sad, Bfe, MadU32U24 (via `param`) — plus the RDNA2-only LshlAdd,
/// which has no Kyty upstream row.
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
#[allow(dead_code)] // C2: no staged SCC-bearing rows right now; kept for the next one
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
    // Beyond Kyty (it NIs the opcode) — measured on Minecraft's menu VS.
    f(recompile_buffer_load_dwordx4,         T::BufferLoadDwordX4, F::Vdata4VaddrSvSoffsIdxen, p1("")),
    f(recompile_buffer_load_dwordx4,         T::BufferLoadDwordX4, F::Vdata4Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_load_dwordx4,         T::BufferLoadDwordX4, F::Vdata4SvSoffs, p1("")),
    f(recompile_buffer_load_dwordx4,         T::BufferLoadDwordX4, F::Vdata4VaddrSvSoffsOffen, p1("")),
    f(recompile_buffer_load_format_x_vdata1, T::BufferLoadFormatX, F::Vdata1VaddrSvSoffsIdxen, p1("")),
    f(recompile_buffer_store_dword_vdata1, T::BufferStoreDword, F::Vdata1VaddrSvSoffsIdxen, p1("")),
    ni("Recompile_BufferStoreFormatX_Vdata1VaddrSvSoffsIdxen",  2068, T::BufferStoreFormatX,  F::Vdata1VaddrSvSoffsIdxen, p1("")),
    ni("Recompile_BufferStoreFormatXy_Vdata2VaddrSvSoffsIdxen", 2137, T::BufferStoreFormatXy, F::Vdata2VaddrSvSoffsIdxen, p1("")),

    f(recompile_fetch, T::FetchX,    F::Vdata1VaddrSvSoffsIdxen, p1("")),
    f(recompile_fetch, T::FetchXy,   F::Vdata2VaddrSvSoffsIdxen, p1("")),
    f(recompile_fetch, T::FetchXyz,  F::Vdata3VaddrSvSoffsIdxen, p1("")),
    f(recompile_fetch, T::FetchXyzw, F::Vdata4VaddrSvSoffsIdxen, p1("")),

    f(recompile_ds_append,  T::DsAppend,  F::VdstGds, p1("")),
    f(recompile_ds_consume, T::DsConsume, F::VdstGds, p1("")),

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

    f(recompile_image_get_resinfo_dmask3, T::ImageGetResinfo, F::Vdata2VaddrStDmask3, p1("")),
    f(recompile_image_load_dmask_f,        T::ImageLoad,      F::Vdata4Vaddr3StDmaskF,   p1("")),
    f(recompile_image_load_dmask1,         T::ImageLoad,      F::Vdata1Vaddr3StDmask1,   p1("")),
    f(recompile_image_load_dmask7,         T::ImageLoad,      F::Vdata3Vaddr3StDmask7,   p1("")),
    // Wired for the texture chain: Minecraft's content pixel shaders reach
    // ImageSample the moment their vertex partners translate. The nine
    // recompilers were already ported (shared dmask body + Lz/LzO); the
    // downstream texture upload feeds the %textures2D_S/%samplers arrays.
    f(recompile_image_sample_dmask1,     T::ImageSample,    F::Vdata1Vaddr3StSsDmask1, p1("")),
    f(recompile_image_sample_dmask8,     T::ImageSample,    F::Vdata1Vaddr3StSsDmask8, p1("")),
    f(recompile_image_sample_dmask3,     T::ImageSample,    F::Vdata2Vaddr3StSsDmask3, p1("")),
    f(recompile_image_sample_dmask5,     T::ImageSample,    F::Vdata2Vaddr3StSsDmask5, p1("")),
    f(recompile_image_sample_dmask9,     T::ImageSample,    F::Vdata2Vaddr3StSsDmask9, p1("")),
    f(recompile_image_sample_dmask7,     T::ImageSample,    F::Vdata3Vaddr3StSsDmask7, p1("")),
    f(recompile_image_sample_dmask_f,    T::ImageSample,    F::Vdata4Vaddr3StSsDmaskF, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata1Vaddr3StSsDmask1, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata1Vaddr3StSsDmask8, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata2Vaddr3StSsDmask3, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata2Vaddr3StSsDmask5, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata2Vaddr3StSsDmask9, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata3Vaddr3StSsDmask7, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata4Vaddr3StSsDmaskF, p1("")),
    f(recompile_image_sample_lz_dmask7,  T::ImageSampleLz,  F::Vdata3Vaddr3StSsDmask7, p1("")),
    f(recompile_image_sample_lz_dmask_f, T::ImageSampleLz,  F::Vdata4Vaddr3StSsDmaskF, p1("")),
    f(recompile_image_sample_lzo_dmask7, T::ImageSampleLzO, F::Vdata3Vaddr4StSsDmask7, p1("")),
    f(recompile_image_store_dmask_f,       T::ImageStore,     F::Vdata4Vaddr3StDmaskF,   p1("")),
    ni("Recompile_ImageStoreMip_Vdata4Vaddr4StDmaskF",    3173, T::ImageStoreMip,  F::Vdata4Vaddr4StDmaskF,   p1("")),

    f(recompile_sbuffer_load_dword,   T::SBufferLoadDword,   F::SdstSvSoffset,  p1("")),
    f(recompile_sbuffer_load_dwordx2, T::SBufferLoadDwordx2, F::Sdst2SvSoffset, p1("")),
    f(recompile_sbuffer_load_dwordx4, T::SBufferLoadDwordx4, F::Sdst4SvSoffset, p1("")),
    f(recompile_sbuffer_load_dwordx8, T::SBufferLoadDwordx8, F::Sdst8SvSoffset, p1("")),
    f(recompile_sbuffer_load_dwordx16, T::SBufferLoadDwordx16, F::Sdst16SvSoffset, p1("")),

    f(recompile_scbranch_xxx_label, T::SCbranchExecz, F::Label, p2("%cc_u_<index> = OpLoad %uint %execz",  "%cc_b_<index> = OpINotEqual %bool %cc_u_<index> %uint_0")),
    f(recompile_scbranch_xxx_label, T::SCbranchScc0,  F::Label, p2("%cc_u_<index> = OpLoad %uint %scc",    "%cc_b_<index> = OpIEqual    %bool %cc_u_<index> %uint_0")),
    f(recompile_scbranch_xxx_label, T::SCbranchScc1,  F::Label, p2("%cc_u_<index> = OpLoad %uint %scc",    "%cc_b_<index> = OpIEqual    %bool %cc_u_<index> %uint_1")),
    f(recompile_scbranch_xxx_label, T::SCbranchVccz,  F::Label, p2("%cc_u_<index> = OpLoad %uint %vcc_lo", "%cc_b_<index> = OpIEqual    %bool %cc_u_<index> %uint_0")),
    f(recompile_scbranch_xxx_label, T::SCbranchVccnz, F::Label, p2("%cc_u_<index> = OpLoad %uint %vcc_lo", "%cc_b_<index> = OpINotEqual %bool %cc_u_<index> %uint_0")),
    f(recompile_sbranch_label,      T::SBranch,       F::Label, p1("")),

    f(recompile_sendpgm_empty, T::SEndpgm, F::Empty, p1("")),

    f(recompile_s_getpc_b64, T::SGetpcB64, F::Sdst2, p1("")),

    f(recompile_sload_dword,   T::SLoadDword,   F::SdstSbaseSoffset,  p1("")),
    f(recompile_sload_dwordx2, T::SLoadDwordx2, F::Sdst2Ssrc02Ssrc1,  p1("")),
    f(recompile_sload_dwordx4, T::SLoadDwordx4, F::Sdst4SbaseSoffset, p1("")),
    f(recompile_sload_dwordx8, T::SLoadDwordx8, F::Sdst8SbaseSoffset, p1("")),

    fs(recompile_s_xxx_b64_sdst2_ssrc02_ssrc12, T::SAndn2B64, F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpNot %uint %t2_<index>",
        "%tb_<index> = OpBitwiseAnd %uint %t0_<index> %ta_<index>",
        "%tc_<index> = OpNot %uint %t3_<index>",
        "%td_<index> = OpBitwiseAnd %uint %t1_<index> %tc_<index>"), S::NonZero),
    fs(recompile_s_xxx_b64_sdst2_ssrc02_ssrc12, T::SOrn2B64, F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpNot %uint %t2_<index>",
        "%tb_<index> = OpBitwiseOr %uint %t0_<index> %ta_<index>",
        "%tc_<index> = OpNot %uint %t3_<index>",
        "%td_<index> = OpBitwiseOr %uint %t1_<index> %tc_<index>"), S::NonZero),
    fs(recompile_s_xxx_b64_sdst2_ssrc02_ssrc12, T::SAndB64, F::Sdst2Ssrc02Ssrc12, p2("%tb_<index> = OpBitwiseAnd %uint %t0_<index> %t2_<index>",
        "%td_<index> = OpBitwiseAnd %uint %t1_<index> %t3_<index>"), S::NonZero),
    fs(recompile_s_xxx_b64_sdst2_ssrc02_ssrc12, T::SNorB64, F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpBitwiseOr %uint %t0_<index> %t2_<index>",
        "%tb_<index> = OpNot %uint %ta_<index>",
        "%tc_<index> = OpBitwiseOr %uint %t1_<index> %t3_<index>",
        "%td_<index> = OpNot %uint %tc_<index>"), S::NonZero),
    fs(recompile_s_xxx_b64_sdst2_ssrc02_ssrc12, T::SNandB64, F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpBitwiseAnd %uint %t0_<index> %t2_<index>",
        "%tb_<index> = OpNot %uint %ta_<index>",
        "%tc_<index> = OpBitwiseAnd %uint %t1_<index> %t3_<index>",
        "%td_<index> = OpNot %uint %tc_<index>"), S::NonZero),
    fs(recompile_s_xxx_b64_sdst2_ssrc02_ssrc12, T::SXnorB64, F::Sdst2Ssrc02Ssrc12, p4("%ta_<index> = OpBitwiseXor %uint %t0_<index> %t2_<index>",
        "%tb_<index> = OpNot %uint %ta_<index>",
        "%tc_<index> = OpBitwiseXor %uint %t1_<index> %t3_<index>",
        "%td_<index> = OpNot %uint %tc_<index>"), S::NonZero),
    fs(recompile_s_xxx_b64_sdst2_ssrc02_ssrc12, T::SOrB64, F::Sdst2Ssrc02Ssrc12, p2("%tb_<index> = OpBitwiseOr %uint %t0_<index> %t2_<index>",
        "%td_<index> = OpBitwiseOr %uint %t1_<index> %t3_<index>"), S::NonZero),
    fs(recompile_s_xxx_b64_sdst2_ssrc02_ssrc12, T::SXorB64, F::Sdst2Ssrc02Ssrc12, p2("%tb_<index> = OpBitwiseXor %uint %t0_<index> %t2_<index>",
        "%td_<index> = OpBitwiseXor %uint %t1_<index> %t3_<index>"), S::NonZero),
    fs(recompile_s_xxx_b64_sdst2_ssrc02_ssrc12, T::SCselectB64, F::Sdst2Ssrc02Ssrc12, p4("%ts_<index> = OpLoad %uint %scc",
        "%tsb_<index> = OpINotEqual %bool %ts_<index> %uint_0",
        "%tb_<index> = OpSelect %uint %tsb_<index> %t0_<index> %t2_<index>",
        "%td_<index> = OpSelect %uint %tsb_<index> %t1_<index> %t3_<index>"), S::None),

    fs(recompile_s_bfe_u64,  T::SBfeU64,  F::Sdst2Ssrc02Ssrc1, p2("", ""), S::NonZero),
    fs(recompile_s_lshl_b64, T::SLshlB64, F::Sdst2Ssrc02Ssrc1, p2("", ""), S::NonZero),
    fs(recompile_s_lshr_b64, T::SLshrB64, F::Sdst2Ssrc02Ssrc1, p2("", ""), S::NonZero),

    fs(recompile_s_xxx_b32_svdst_svsrc01, T::SAndB32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseAnd %uint %t0_<index> %t1_<index>"), S::NonZero),
    fs(recompile_s_xxx_b32_svdst_svsrc01, T::SBfmB32, F::SVdstSVsrc0SVsrc1, p3("%tcount_<index> = OpBitwiseAnd %uint %t0_<index> %uint_31", "%toffset_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpBitFieldInsert %uint %uint_0 %uint_0xffffffff %toffset_<index> %tcount_<index>"), S::None),
    fs(recompile_s_xxx_b32_svdst_svsrc01, T::SCselectB32, F::SVdstSVsrc0SVsrc1, p3("%t22_<index> = OpLoad %uint %scc", "%t2_<index> = OpINotEqual %bool %t22_<index> %uint_0", "%t_<index> = OpSelect %uint %t2_<index> %t0_<index> %t1_<index>"), S::None),
    fs(recompile_s_xxx_b32_svdst_svsrc01, T::SLshlB32, F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpShiftLeftLogical %uint %t0_<index> %ts_<index>"), S::NonZero),
    fs(recompile_s_xxx_b32_svdst_svsrc01, T::SLshrB32, F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpShiftRightLogical %uint %t0_<index> %ts_<index>"), S::NonZero),
    fs(recompile_s_xxx_b32_svdst_svsrc01, T::SOrB32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseOr %uint %t0_<index> %t1_<index>"), S::NonZero),
    // Wired for Minecraft's menu vertex shaders: all three VS at the analysis
    // frontier (0x253e6200/0x253e7900/0x253f0400) stop on SSubI32 once their
    // user-SGPR layout resolves. Kyty had the recompiler staged; the SCC
    // overflow templates were already in the SCC machinery.
    fs(recompile_s_xxx_i32_svdst_svsrc01, T::SAddI32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpIAdd %int %t0_<index> %t1_<index>"), S::OverflowAdd),
    fs(recompile_s_xxx_i32_svdst_svsrc01, T::SMulI32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpIMul %int %t0_<index> %t1_<index>"), S::None),
    fs(recompile_s_xxx_i32_svdst_svsrc01, T::SSubI32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpISub %int %t0_<index> %t1_<index>"), S::OverflowSub),
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SAddcU32, F::SVdstSVsrc0SVsrc1, p4("%tscc_<index> = OpLoad %uint %scc", "%ts_<index> = OpFunctionCall %v2uint %addc %t0_<index> %t1_<index> %tscc_<index>", "%t_<index> = OpCompositeExtract %uint %ts_<index> 0", "%carry_<index> = OpCompositeExtract %uint %ts_<index> 1"), S::CarryOut),
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SAddU32, F::SVdstSVsrc0SVsrc1, p3("%ts_<index> = OpIAddCarry %ResTypeU %t0_<index> %t1_<index>", "%t_<index> = OpCompositeExtract %uint %ts_<index> 0", "%carry_<index> = OpCompositeExtract %uint %ts_<index> 1"), S::CarryOut),
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SSubU32, F::SVdstSVsrc0SVsrc1, p4("%t_<index> = OpISub %uint %t0_<index> %t1_<index>", "%nb_<index> = OpUGreaterThanEqual %bool %t0_<index> %t1_<index>", "%carry_<index> = OpSelect %uint %nb_<index> %uint_1 %uint_0", ""), S::CarryOut),
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SBfeU32, F::SVdstSVsrc0SVsrc1, p3("%to_<index> = OpBitFieldUExtract %uint %t1_<index> %uint_0  %uint_5", "%ts_<index> = OpBitFieldUExtract %uint %t1_<index> %uint_16 %uint_7", "%t_<index> = OpBitFieldUExtract %uint %t0_<index> %to_<index> %ts_<index>"), S::NonZero),
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SLshl4AddU32, F::SVdstSVsrc0SVsrc1, p3("%ts_<index> = OpFunctionCall %v2uint %lshl_add %t0_<index> %t1_<index> %uint_4", "%t_<index> = OpCompositeExtract %uint %ts_<index> 0", "%carry_<index> = OpCompositeExtract %uint %ts_<index> 1"), S::CarryOut),
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SMulHiU32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFunctionCall %uint %mul_hi_uint %t0_<index> %t1_<index>"), S::None),
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SPackLlB32B16, F::SVdstSVsrc0SVsrc1, p3("%tlo_<index> = OpBitwiseAnd %uint %t0_<index> %uint_0x0000ffff", "%thi_<index> = OpShiftLeftLogical %uint %t1_<index> %uint_16", "%t_<index> = OpBitwiseOr %uint %tlo_<index> %thi_<index>"), S::None),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VAndB32,     F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseAnd %uint %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VBcntU32B32, F::SVdstSVsrc0SVsrc1, p3("%tb_<index> = OpBitCount %int %t0_<index>", "%tbu_<index> = OpBitcast %uint %tb_<index>", "%t_<index> = OpIAdd %uint %tbu_<index> %t1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VBfmB32,     F::SVdstSVsrc0SVsrc1, p3("%tcount_<index> = OpBitwiseAnd %uint %t0_<index> %uint_31", "%toffset_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpBitFieldInsert %uint %uint_0 %uint_0xffffffff %toffset_<index> %tcount_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VLshlB32,    F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpShiftLeftLogical %uint %t0_<index> %ts_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VLshlrevB32, F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t0_<index> %uint_31", "%t_<index> = OpShiftLeftLogical %uint %t1_<index> %ts_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VLshrB32,    F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31", "%t_<index> = OpShiftRightLogical %uint %t0_<index> %ts_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VLshrrevB32, F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %uint %t0_<index> %uint_31", "%t_<index> = OpShiftRightLogical %uint %t1_<index> %ts_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VMulHiU32,   F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFunctionCall %uint %mul_hi_uint %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VMulLoU32,   F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFunctionCall %uint %mul_lo_uint %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VMulU32U24,  F::SVdstSVsrc0SVsrc1, p3("%tu0_<index> = OpBitwiseAnd %uint %t0_<index> %uint_0x00ffffff", "%tu1_<index> = OpBitwiseAnd %uint %t1_<index> %uint_0x00ffffff", "%t_<index> = OpFunctionCall %uint %mul_lo_uint %tu0_<index> %tu1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VOrB32,      F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseOr %uint %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VXorB32,     F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpBitwiseXor %uint %t0_<index> %t1_<index>")),
    // RDNA2-only (no Kyty upstream rows): the carry-less VOP2 add/sub family
    // measured in Minecraft's menu CS.
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VAddNcU32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpIAdd %uint %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VSubNcU32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpISub %uint %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VSubrevNcU32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpISub %uint %t1_<index> %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VAddF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFAdd %float %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMacF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Fma %t0_<index> %t1_<index> %tdst_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMaxF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %float %GLSL_std_450 FMax %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMinF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %float %GLSL_std_450 FMin %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMulF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFMul %float %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VSubF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFSub %float %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VSubrevF32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFSub %float %t1_<index> %t0_<index>")),
    // Wired from the staged set: `recompile_v_xxx_i32_svdst_svsrc01` was already
    // written (and these rows' SPIR-V already correct) but left unreachable.
    // Minecraft reaches `v_ashrrev_i32` in a compute shader once boot gets far
    // enough, and an NI row fails the whole shader, skipping every draw bound to
    // it. Ashr/Ashrrev differ only in which operand is the shift amount — the
    // "rev" form takes it from src0 and shifts src1. `%mul_lo_int` is emitted by
    // the SPIR-V preamble (spirv.rs), so MulLo is safe to wire alongside them.
    f(recompile_v_xxx_i32_svdst_svsrc01, T::VAshrI32,    F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %int %t1_<index> %int_31", "%t_<index> = OpShiftRightArithmetic %int %t0_<index> %ts_<index>")),
    f(recompile_v_xxx_i32_svdst_svsrc01, T::VAshrrevI32, F::SVdstSVsrc0SVsrc1, p2("%ts_<index> = OpBitwiseAnd %int %t0_<index> %int_31", "%t_<index> = OpShiftRightArithmetic %int %t1_<index> %ts_<index>")),
    f(recompile_v_xxx_i32_svdst_svsrc01, T::VMulLoI32,   F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFunctionCall %int %mul_lo_int %t0_<index> %t1_<index>")),
    f(recompile_vcvt_pkrtz_f16_f32, T::VCvtPkrtzF16F32, F::SVdstSVsrc0SVsrc1, p1("")),
    ni("Recompile_VMbcntHiU32B32_SVdstSVsrc0SVsrc1",  5455, T::VMbcntHiU32B32,  F::SVdstSVsrc0SVsrc1, p1("")),
    ni("Recompile_VMbcntLoU32B32_SVdstSVsrc0SVsrc1",  5497, T::VMbcntLoU32B32,  F::SVdstSVsrc0SVsrc1, p1("")),

    f(recompile_smov_b32, T::SMovB32,  F::SVdstSVsrc0, p1("")),
    f(recompile_smov_b32, T::SMovkI32, F::SVdstSVsrc0, p1("")),
    f(recompile_smulk_i32, T::SMulkI32, F::SVdstSVsrc0, p1("")),
    f(recompile_v_xxx_b32_svdst_svsrc0, T::VBfrevB32, F::SVdstSVsrc0, p1("%t_<index> = OpBitReverse %uint %t0_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc0, T::VNotB32,   F::SVdstSVsrc0, p1("%t_<index> = OpNot %uint %t0_<index>")),
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
    f(recompile_vcvt_xxx_f32, T::VCvtI32F32, F::SVdstSVsrc0, p2("%t1_<index> = OpExtInst %float %GLSL_std_450 Trunc %t0_<index>", "%t2_<index> = OpConvertFToS %int %t1_<index>")),
    f(recompile_vcvt_xxx_f32, T::VCvtFlrI32F32, F::SVdstSVsrc0, p2("%t1_<index> = OpExtInst %float %GLSL_std_450 Floor %t0_<index>", "%t2_<index> = OpConvertFToS %int %t1_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32F16,    F::SVdstSVsrc0, p2("%ts_<index> = OpExtInst %v2float %GLSL_std_450 UnpackHalf2x16 %t0_<index>", "%t_<index> = OpCompositeExtract %float %ts_<index> 0")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32I32,    F::SVdstSVsrc0, p2("%ti_<index> = OpBitcast %int %t0_<index>", "%t_<index> = OpConvertSToF %float %ti_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32U32,    F::SVdstSVsrc0, p1("%t_<index> = OpConvertUToF %float %t0_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32Ubyte0, F::SVdstSVsrc0, p2("%tb_<index> = OpBitFieldUExtract %uint %t0_<index> %uint_0 %uint_8", "%t_<index> = OpConvertUToF %float %tb_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32Ubyte1, F::SVdstSVsrc0, p2("%tb_<index> = OpBitFieldUExtract %uint %t0_<index> %uint_8 %uint_8", "%t_<index> = OpConvertUToF %float %tb_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32Ubyte2, F::SVdstSVsrc0, p2("%tb_<index> = OpBitFieldUExtract %uint %t0_<index> %uint_16 %uint_8", "%t_<index> = OpConvertUToF %float %tb_<index>")),
    f(recompile_vcvt_f32_xxx, T::VCvtF32Ubyte3, F::SVdstSVsrc0, p2("%tb_<index> = OpBitFieldUExtract %uint %t0_<index> %uint_24 %uint_8", "%t_<index> = OpConvertUToF %float %tb_<index>")),
    f(recompile_vmov_b32, T::VMovB32, F::SVdstSVsrc0, p1("")),

    fs(recompile_s_and_saveexec_b64, T::SAndSaveexecB64, F::Sdst2Ssrc02, p1(""), S::NonZero),
    fs(recompile_s_orn2_saveexec_b64, T::SOrn2SaveexecB64, F::Sdst2Ssrc02, p1(""), S::NonZero),
    f(recompile_smov_b64,    T::SMovB64,    F::Sdst2Ssrc02, p1("")),
    f(recompile_sswappc_b64, T::SSwappcB64, F::Sdst2Ssrc02, p1("")),
    fs(recompile_swqm_b64, T::SWqmB64, F::Sdst2Ssrc02, p1(""), S::NonZero),
    fs(recompile_snot_b64, T::SNotB64, F::Sdst2Ssrc02, p1(""), S::NonZero),
    // s_brev_b32 does not write SCC.
    fs(
        recompile_sbrev_b32,
        T::SBrevB32,
        F::SVdstSVsrc0,
        p1(""),
        S::None,
    ),

    f(recompile_skip, T::SInstPrefetch, F::Imm, p1("")),
    f(recompile_skip, T::SNop,          F::Imm, p1("")),
    f(recompile_skip, T::SSendmsg,      F::Imm, p1("")),
    f(recompile_skip, T::SWaitcnt,      F::Imm, p1("")),

    f(recompile_tbuffer_load_format_x_float1, T::TBufferLoadFormatX, F::Vdata1VaddrSvSoffsIdxenFloat1, p1("")),
    ni("Recompile_TBufferLoadFormatXyzw_Vdata4Vaddr2SvSoffsOffenIdxenFloat4", 4824, T::TBufferLoadFormatXyzw, F::Vdata4Vaddr2SvSoffsOffenIdxenFloat4, p1("")),
    f(recompile_tbuffer_load_format_xyzw_float4, T::TBufferLoadFormatXyzw, F::Vdata4VaddrSvSoffsIdxenFloat4, p1("")),

    ni("Recompile_V_XXX_U32_VdstSdst2Vsrc0Vsrc1", 6005, T::VAddI32,    F::VdstSdst2Vsrc0Vsrc1, p1("%t_<index> = OpIAddCarry %ResTypeU %t0_<index> %t1_<index>")),
    ni("Recompile_V_XXX_U32_VdstSdst2Vsrc0Vsrc1", 6005, T::VSubI32,    F::VdstSdst2Vsrc0Vsrc1, p1("%t_<index> = OpISubBorrow %ResTypeU %t0_<index> %t1_<index>")),
    ni("Recompile_V_XXX_U32_VdstSdst2Vsrc0Vsrc1", 6005, T::VSubrevI32, F::VdstSdst2Vsrc0Vsrc1, p1("%t_<index> = OpISubBorrow %ResTypeU %t1_<index> %t0_<index>")),

    // Wired for Minecraft's menu VS (`v_cmp_lt_f32 s2, |v2|, c` — SDWA
    // src0_abs now parses into the operand modifier operand_load_float
    // already applies). The recompiler was staged; O/U use the ordered/
    // unordered helpers that spirv.rs already defines.
    f(recompile_vcmp_xxx_f32, T::VCmpEqF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdEqual")),
    f(recompile_vcmp_xxx_f32, T::VCmpFF32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_1 ; ")),
    f(recompile_vcmp_xxx_f32, T::VCmpGeF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdGreaterThanEqual")),
    f(recompile_vcmp_xxx_f32, T::VCmpGtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdGreaterThan")),
    f(recompile_vcmp_xxx_f32, T::VCmpLeF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdLessThanEqual")),
    f(recompile_vcmp_xxx_f32, T::VCmpLgF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdNotEqual")),
    f(recompile_vcmp_xxx_f32, T::VCmpLtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdLessThan")),
    f(recompile_vcmp_xxx_f32, T::VCmpNeqF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordNotEqual")),
    f(recompile_vcmp_xxx_f32, T::VCmpNgeF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordLessThan")),
    f(recompile_vcmp_xxx_f32, T::VCmpNgtF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordLessThanEqual")),
    f(recompile_vcmp_xxx_f32, T::VCmpNleF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordGreaterThan")),
    f(recompile_vcmp_xxx_f32, T::VCmpNlgF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordEqual")),
    f(recompile_vcmp_xxx_f32, T::VCmpNltF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordGreaterThanEqual")),
    f(recompile_vcmp_xxx_f32, T::VCmpOF32,   F::SmaskVsrc0Vsrc1, p1("OpFunctionCall %bool %ordered %t0_<index> %t1_<index> ; ")),
    f(recompile_vcmp_xxx_f32, T::VCmpTruF32, F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_0 ; ")),
    f(recompile_vcmp_xxx_f32, T::VCmpUF32,   F::SmaskVsrc0Vsrc1, p1("OpFunctionCall %bool %unordered %t0_<index> %t1_<index> ; ")),
    // Wired for Minecraft's menu VS (VCmpEqU32 measured after the SDWA fix
    // let the shaders progress). Eq/Ne are sign-agnostic and live in the I32
    // family exactly as Kyty's table lays them out.
    f(recompile_vcmp_xxx_i32, T::VCmpEqI32,  F::SmaskVsrc0Vsrc1, p1("OpIEqual")),
    f(recompile_vcmp_xxx_i32, T::VCmpEqU32,  F::SmaskVsrc0Vsrc1, p1("OpIEqual")),
    f(recompile_vcmp_xxx_i32, T::VCmpFI32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_1 ; ")),
    f(recompile_vcmp_xxx_i32, T::VCmpGeI32,  F::SmaskVsrc0Vsrc1, p1("OpSGreaterThanEqual")),
    f(recompile_vcmp_xxx_i32, T::VCmpGtI32,  F::SmaskVsrc0Vsrc1, p1("OpSGreaterThan")),
    f(recompile_vcmp_xxx_i32, T::VCmpLeI32,  F::SmaskVsrc0Vsrc1, p1("OpSLessThanEqual")),
    f(recompile_vcmp_xxx_i32, T::VCmpLtI32,  F::SmaskVsrc0Vsrc1, p1("OpSLessThan")),
    f(recompile_vcmp_xxx_i32, T::VCmpNeI32,  F::SmaskVsrc0Vsrc1, p1("OpINotEqual")),
    f(recompile_vcmp_xxx_i32, T::VCmpNeU32,  F::SmaskVsrc0Vsrc1, p1("OpINotEqual")),
    f(recompile_vcmp_xxx_i32, T::VCmpTI32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_0 ; ")),
    f(recompile_vcmp_xxx_u32, T::VCmpFU32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_1 ; ")),
    f(recompile_vcmp_xxx_u32, T::VCmpGeU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThanEqual")),
    f(recompile_vcmp_xxx_u32, T::VCmpGtU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThan")),
    f(recompile_vcmp_xxx_u32, T::VCmpLeU32,  F::SmaskVsrc0Vsrc1, p1("OpULessThanEqual")),
    f(recompile_vcmp_xxx_u32, T::VCmpLtU32,  F::SmaskVsrc0Vsrc1, p1("OpULessThan")),
    f(recompile_vcmp_xxx_u32, T::VCmpTU32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_0 ; ")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxEqF32, F::SmaskVsrc0Vsrc1, p1("OpFOrdEqual")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxNeqF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordNotEqual")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxGtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdGreaterThan")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxLtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdLessThan")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxNltF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordGreaterThanEqual")),
    // Wired for the Minecraft menu CS (v_cmpx family; Kyty GCN semantics —
    // the comparison result lands in both the mask destination and EXEC; on
    // real RDNA2 cmpx writes EXEC only, the extra mask write is a documented
    // deviation until a title proves it matters).
    f(recompile_vcmpx_xxx_i32, T::VCmpxEqU32,  F::SmaskVsrc0Vsrc1, p1("OpIEqual")),
    f(recompile_vcmpx_xxx_i32, T::VCmpxNeU32,  F::SmaskVsrc0Vsrc1, p1("OpINotEqual")),
    f(recompile_vcmpx_xxx_u32, T::VCmpxGeU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThanEqual")),
    f(recompile_vcmpx_xxx_u32, T::VCmpxGtU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThan")),
    f(recompile_vcmpx_xxx_u32, T::VCmpxLtU32,  F::SmaskVsrc0Vsrc1, p1("OpULessThan")),
    // The `v_cmpx_*_i32` block: same lowering as its unsigned twins above, but
    // through `_i32` — that variant loads its operands with `operand_load_int`,
    // which is what a signed compare needs (an inline constant must arrive
    // sign-extended). Eq/Ne are sign-agnostic and match the unsigned rows.
    f(recompile_vcmpx_xxx_i32, T::VCmpxLtI32,  F::SmaskVsrc0Vsrc1, p1("OpSLessThan")),
    f(recompile_vcmpx_xxx_i32, T::VCmpxGeI32,  F::SmaskVsrc0Vsrc1, p1("OpSGreaterThanEqual")),
    f(recompile_vcmpx_xxx_i32, T::VCmpxLeI32,  F::SmaskVsrc0Vsrc1, p1("OpSLessThanEqual")),
    f(recompile_vcmpx_xxx_i32, T::VCmpxGtI32,  F::SmaskVsrc0Vsrc1, p1("OpSGreaterThan")),
    f(recompile_vcmpx_xxx_i32, T::VCmpxEqI32,  F::SmaskVsrc0Vsrc1, p1("OpIEqual")),
    f(recompile_vcmpx_xxx_i32, T::VCmpxNeI32,  F::SmaskVsrc0Vsrc1, p1("OpINotEqual")),

    f(recompile_scmp_xxx_i32, T::SCmpEqI32, F::Ssrc0Ssrc1, p1("OpIEqual")),
    f(recompile_scmp_xxx_i32, T::SCmpGeI32, F::Ssrc0Ssrc1, p1("OpSGreaterThanEqual")),
    f(recompile_scmp_xxx_i32, T::SCmpGtI32, F::Ssrc0Ssrc1, p1("OpSGreaterThan")),
    f(recompile_scmp_xxx_i32, T::SCmpLgI32, F::Ssrc0Ssrc1, p1("OpINotEqual")),
    f(recompile_scmp_xxx_i32, T::SCmpLtI32, F::Ssrc0Ssrc1, p1("OpSLessThan")),
    f(recompile_scmp_xxx_i32, T::SCmpLeI32, F::Ssrc0Ssrc1, p1("OpSLessThanEqual")),
    f(recompile_scmp_xxx_u32, T::SCmpEqU32, F::Ssrc0Ssrc1, p1("OpIEqual")),
    f(recompile_scmp_xxx_u32, T::SCmpGeU32, F::Ssrc0Ssrc1, p1("OpUGreaterThanEqual")),
    f(recompile_scmp_xxx_u32, T::SCmpGtU32, F::Ssrc0Ssrc1, p1("OpUGreaterThan")),
    f(recompile_scmp_xxx_u32, T::SCmpLeU32, F::Ssrc0Ssrc1, p1("OpULessThanEqual")),
    f(recompile_scmp_xxx_u32, T::SCmpLtU32, F::Ssrc0Ssrc1, p1("OpULessThan")),
    f(recompile_scmp_xxx_u32, T::SCmpLgU32, F::Ssrc0Ssrc1, p1("OpINotEqual")),

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
    // Cubemap coordinate helpers (VOP3 0x144-0x147). Shared recompiler picks
    // the major axis; the param slots are unused (its own match keys on type).
    f(recompile_v_cube_f32, T::VCubeIdF32, F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_v_cube_f32, T::VCubeScF32, F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_v_cube_f32, T::VCubeTcF32, F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_v_cube_f32, T::VCubeMaF32, F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    ni("Recompile_V_XXX_U32_VdstVsrc0Vsrc1Vsrc2", 5940, T::VSadU32,    F::VdstVsrc0Vsrc1Vsrc2, p2("%td_<index> = OpFunctionCall %uint %abs_diff %t0_<index> %t1_<index>",
        "%t_<index> = OpIAdd %uint %td_<index> %t2_<index>")),
    ni("Recompile_V_XXX_U32_VdstVsrc0Vsrc1Vsrc2", 5940, T::VBfeU32,    F::VdstVsrc0Vsrc1Vsrc2, p3("%to_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31",
        "%ts_<index> = OpBitwiseAnd %uint %t2_<index> %uint_31",
        "%t_<index> = OpBitFieldUExtract %uint %t0_<index> %to_<index> %ts_<index>")),
    ni("Recompile_V_XXX_U32_VdstVsrc0Vsrc1Vsrc2", 5940, T::VMadU32U24, F::VdstVsrc0Vsrc1Vsrc2, p4("%tu0_<index> = OpBitwiseAnd %uint %t0_<index> %uint_0x00ffffff",
        "%tu1_<index> = OpBitwiseAnd %uint %t1_<index> %uint_0x00ffffff",
        "%tm_<index> = OpFunctionCall %uint %mul_lo_uint %tu0_<index> %tu1_<index>",
        "%t_<index> = OpIAdd %uint %tm_<index> %t2_<index>")),
    // RDNA2-only (no Kyty upstream row): v_lshl_add_u32 — the first
    // next-gen instruction Minecraft's menu CS needs.
    f(recompile_v_xxx_u32_vdst_vsrc012, T::VLshlAddU32, F::VdstVsrc0Vsrc1Vsrc2, p3("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31",
        "%tsh_<index> = OpShiftLeftLogical %uint %t0_<index> %ts_<index>",
        "%t_<index> = OpIAdd %uint %tsh_<index> %t2_<index>")),
    // `v_and_or_b32`: dst = (src0 & src1) | src2 — matching SharpEmu's Gen5
    // lowering `BitwiseOr(BitwiseAnd(s0, s1), s2)` exactly.
    f(recompile_v_xxx_u32_vdst_vsrc012, T::VAndOrB32, F::VdstVsrc0Vsrc1Vsrc2, p2("%ta_<index> = OpBitwiseAnd %uint %t0_<index> %t1_<index>",
        "%t_<index> = OpBitwiseOr %uint %ta_<index> %t2_<index>")),
    // `v_lshl_or_u32`: dst = (src0 << (src1 & 31)) | src2. Identical to
    // VLshlAddU32 above except the fold is OR, not ADD — SharpEmu's Gen5
    // lowers the two that way, which also cross-checks VLshlAddU32.
    f(recompile_v_xxx_u32_vdst_vsrc012, T::VLshlOrU32, F::VdstVsrc0Vsrc1Vsrc2, p3("%ts_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31",
        "%tsh_<index> = OpShiftLeftLogical %uint %t0_<index> %ts_<index>",
        "%t_<index> = OpBitwiseOr %uint %tsh_<index> %t2_<index>")),
    // `v_or3_u32`: dst = (src0 | src1) | src2.
    f(recompile_v_xxx_u32_vdst_vsrc012, T::VOr3U32, F::VdstVsrc0Vsrc1Vsrc2, p2("%to_<index> = OpBitwiseOr %uint %t0_<index> %t1_<index>",
        "%t_<index> = OpBitwiseOr %uint %to_<index> %t2_<index>")),
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
        if let Err(e) = validator.validate(&module) {
            // naga's SPIR-V frontend rejects the V# push-constant descriptor
            // table type — an array of `[uint; 4]` (`InvalidArrayBaseType`,
            // naga has no arrays-of-arrays in its IR). That type is SPIR-V-spec
            // valid and REAL Vulkan accepts it: Minecraft's storage-buffer
            // pixel shader renders through this exact declaration. So this one
            // naga error is a known false negative; every other class still
            // fails the test. The real-driver gate is the `--run-eboot` render,
            // not naga.
            let msg = format!("{e:?}");
            assert!(
                msg.contains("InvalidArrayBaseType"),
                "naga validate of {name} failed: {msg}"
            );
        }
    }

    // ---- 1. dispatch table ------------------------------------------------

    /// Every SPIR-V opcode a wired row can emit must be one the assembler can
    /// encode.
    ///
    /// The two tables are edited independently, so a row can name an opcode
    /// `spirv_asm::op_info` has no arm for. Nothing catches that until a title
    /// executes that exact instruction, and then it surfaces a long way from
    /// the cause: the shader fails to assemble, the whole shader is dropped,
    /// and draws binding it are silently skipped — a black frame, not an error.
    /// Minecraft found `OpConvertFToS` (110) this way; the assembler had 109,
    /// 111 and 112, so the gap was invisible when reading either table alone.
    ///
    /// Scoped to `Func` rows because only those can reach the assembler today.
    /// Wiring a staged row flips it to `Func` and it is covered from then on.
    #[test]
    fn every_wired_template_opcode_assembles() {
        let mut missing: Vec<String> = Vec::new();
        for row in recomp_func_table()
            .iter()
            .filter(|e| matches!(e.func, RecompileFn::Func(_)))
        {
            for template in row.param.iter().flatten() {
                for tok in template.split_whitespace().filter(|tok| {
                    tok.starts_with("Op")
                        && tok[2..].starts_with(|c: char| c.is_ascii_uppercase())
                        && tok.chars().all(|c| c.is_ascii_alphanumeric())
                }) {
                    if !crate::spirv_asm::knows_opcode(tok) {
                        missing.push(format!("{tok} (emitted by {:?})", row.type_));
                    }
                }
            }
        }
        missing.sort_unstable();
        missing.dedup();
        assert!(
            missing.is_empty(),
            "wired recompiler rows emit opcodes spirv_asm cannot assemble; \
             add an arm to op_info for each: {missing:?}"
        );
    }

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
        assert_eq!(
            table.len(),
            248,
            "204 Kyty rows plus SSubU32, SNop, the RDNA2-only rows \
             (VLshlAddU32, VCmpxLtU32, VAddNcU32, VSubNcU32, VSubrevNcU32, VCvtI32F32, \
             VCvtFlrI32F32, VCmpxNltF32, SOrn2SaveexecB64, the ImageLoad dmask1/7 \
             and ImageSampleLz dmaskF rows, \
             the Kyty-gated trio VAndOrB32/VLshlOrU32/VOr3U32, and the v_cmpx_*_i32 \
             block: VCmpxLtI32/GeI32/GtI32/LeI32/EqI32/NeI32), the beyond-Kyty \
             BufferLoadDwordX4 (+Offen and address-only) rows, ImageGetResinfo, \
             SGetpcB64, SPackLlB32B16, the seven ImageSampleCLz dmask rows, \
             and the four cubemap helpers VCubeId/Sc/Tc/MaF32, plus SNotB64, SBrevB32 and VCmpxEqF32"
        );
        assert_eq!(implemented + ni, table.len());
        assert_eq!(
            implemented, 236,
            "C1 implemented subset plus title-driven ports (incl. the S_XXX_I32 \
             trio, VCvtFlrI32F32, VCmpxNltF32, SOrn2SaveexecB64, the ImageLoad \
             dmask1/7 + ImageSampleLz dmaskF rows, the nine ImageSample dmask recompilers, the VCmp \
              F32/I32/U32 families, address-only BufferLoadDwordX4, \
              ImageGetResinfo, SGetpcB64, SPackLlB32B16, the seven \
              ImageSampleCLz dmask rows, and the four VCube*F32 \
              cubemap-coordinate helpers)"
        );
        assert_eq!(ni, 12, "C2 remainder");

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

        // ImageSample is wired now (texture chain): the lookup lands on the
        // shared dmask recompiler.
        let e = recomp_func(T::ImageSample, F::Vdata4Vaddr3StSsDmaskF).expect("ImageSample");
        assert!(matches!(e.func, RecompileFn::Func(_)));

        // NI entry carries the Kyty function name + line anchor.
        let e = recomp_func(T::ImageStoreMip, F::Vdata4Vaddr4StDmaskF).expect("ImageStoreMip");
        match e.func {
            RecompileFn::NotImplemented { kyty_func, line } => {
                assert_eq!(kyty_func, "Recompile_ImageStoreMip_Vdata4Vaddr4StDmaskF");
                assert_eq!(line, 3173);
            }
            RecompileFn::Func(_) => panic!("ImageStoreMip must be NI in C1"),
        }

        // SccCheck rides along as table data (application is C2).
        let e = recomp_func(T::SAddU32, F::SVdstSVsrc0SVsrc1).expect("SAddU32");
        assert_eq!(e.scc_check, SccCheck::CarryOut);

        // Unknown (type, format) pair -> None.
        assert!(recomp_func(T::VMovB32, F::Label).is_none());
    }

    #[test]
    fn s_sub_u32_recompiles_with_no_borrow_scc() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                0x80EA_6BC0,
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
            &mut code,
            true,
        )
        .expect("parse measured s_sub_u32");

        let entry = recomp_func(T::SSubU32, F::SVdstSVsrc0SVsrc1).expect("SSubU32 row");
        assert!(matches!(entry.func, RecompileFn::Func(_)));
        assert_eq!(entry.scc_check, SccCheck::CarryOut);

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile measured s_sub_u32");
        assert!(source.contains("OpISub %uint"));
        assert!(source.contains("OpUGreaterThanEqual %bool"));
        assert!(source.contains("OpStore %scc %carry_0"));
        let words = spirv_run(&source).expect("assemble measured s_sub_u32");
        naga_parse_and_validate(&words, "s_sub_u32");
    }

    /// `s_sub_i32` — the signed twin, and the single instruction all three of
    /// Minecraft's menu vertex shaders stopped on once their user-SGPR layout
    /// resolved. Signed subtract sets SCC on signed OVERFLOW (not borrow), so
    /// the row must carry `SccCheck::OverflowSub` — wiring it through the
    /// carry-out template would silently compute the wrong SCC.
    #[test]
    fn s_sub_i32_is_wired_and_sets_overflow_scc() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                // s_sub_i32 s10, s3, s4 (SOP2 op 0x03).
                0x818A_0403,
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
            &mut code,
            true,
        )
        .expect("parse s_sub_i32");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SSubI32);
        assert_eq!(inst.dst.register_id, 10);
        assert_eq!(inst.src[0].register_id, 3);
        assert_eq!(inst.src[1].register_id, 4);

        let entry = recomp_func(T::SSubI32, F::SVdstSVsrc0SVsrc1).expect("SSubI32 row");
        assert!(matches!(entry.func, RecompileFn::Func(_)));
        assert_eq!(entry.scc_check, SccCheck::OverflowSub);

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile s_sub_i32");
        assert!(source.contains("OpISub %int"), "signed subtract:\n{source}");
        assert!(
            source.contains("SSign"),
            "overflow SCC path (sign compare), not carry-out:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble s_sub_i32");
        naga_parse_and_validate(&words, "s_sub_i32");
    }

    /// `v_mul_f32 v2, v4, -v3` — the measured UI-PS instruction (Minecraft
    /// PPSA17221, ps+0x38) that SDWA `src1_neg` blocked at parse. The
    /// modifier must ride the operand into `operand_load_float`'s OpFNegate.
    #[test]
    fn v_mul_f32_sdwa_src1_neg_recompiles_with_fnegate() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0x1002_06F9,
                0x1606_0604,
                0xBF80_0000, // s_nop — endpgm must sit at index >= 2
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse v_mul_f32 with sdwa src1_neg");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VMulF32);
        assert!(inst.src[1].negate, "src1 negate must be recorded");
        assert!(!inst.src[0].negate);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile v_mul_f32 sdwa");
        assert!(source.contains("OpFNegate"), "negated src1:\n{source}");
        assert!(source.contains("OpFMul"), "the multiply itself:\n{source}");
        let words = spirv_run(&source).expect("assemble v_mul_f32 sdwa");
        naga_parse_and_validate(&words, "v_mul_f32_sdwa_src1_neg");
    }

    /// `v_cmp_lt_f32 s2, |v2|, ...` — the measured menu-VS instruction
    /// (Minecraft PPSA17221, vs+0x1b0) that SDWA `src0_abs` blocked at parse.
    /// Beyond Kyty (its vopc path exits on any modifier); the float load
    /// already applies FAbs.
    #[test]
    fn v_cmp_lt_f32_sdwa_src0_abs_recompiles_with_fabs() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0x7C03_E4F9,
                0x8626_8202,
                0xBF80_0000, // s_nop — endpgm must sit at index >= 2
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse v_cmp_lt_f32 with sdwa src0_abs");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VCmpLtF32);
        assert!(inst.src[0].absolute, "src0 absolute must be recorded");
        assert!(!inst.src[1].absolute);

        let entry = recomp_func(T::VCmpLtF32, F::SmaskVsrc0Vsrc1).expect("VCmpLtF32 row");
        assert!(matches!(entry.func, RecompileFn::Func(_)));

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile v_cmp_lt_f32 sdwa");
        assert!(source.contains("FAbs"), "abs src0:\n{source}");
        assert!(source.contains("OpFOrdLessThan"), "the compare:\n{source}");
        let words = spirv_run(&source).expect("assemble v_cmp_lt_f32 sdwa");
        naga_parse_and_validate(&words, "v_cmp_lt_f32_sdwa_src0_abs");
    }

    /// `v_cmp_eq_u32 s[0:1], 1, v7` — the measured menu-VS integer compare
    /// (Minecraft PPSA17221; surfaced once SDWA stopped gating). Equality is
    /// sign-agnostic, so the I32 family's OpIEqual row covers U32 too.
    #[test]
    fn v_cmp_eq_u32_recompiles_with_iequal() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0x7D84_0EF9, // vopc marker|op 0xC2 (eq_u32), vsrc1=v7, sdwa
                0x0686_8081, // src0=inline 1 (operand 129), sdst=0, sd=1, s0=sgpr
                0xBF80_0000,
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse v_cmp_eq_u32");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VCmpEqU32);
        assert_eq!(inst.dst.register_id, 0);
        assert_eq!(inst.src[0].constant.i(), 1);

        let entry = recomp_func(T::VCmpEqU32, F::SmaskVsrc0Vsrc1).expect("VCmpEqU32 row");
        assert!(matches!(entry.func, RecompileFn::Func(_)));

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile v_cmp_eq_u32");
        assert!(source.contains("OpIEqual"), "the compare:\n{source}");
        let words = spirv_run(&source).expect("assemble v_cmp_eq_u32");
        naga_parse_and_validate(&words, "v_cmp_eq_u32");
    }

    /// The beyond-Kyty extended (EUD) `SLoadDwordx2` path, measured on
    /// Minecraft's menu VS (`s_load_dwordx2 s[82:83], s[14:15], 8`): with the
    /// EUD base live at s14, the two dwords come from the push-constant
    /// resource table — the x4/x8 machinery with n=2.
    #[test]
    fn s_load_dwordx2_extended_reads_the_resource_table() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx2,
            format: F::Sdst2Ssrc02Ssrc1,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 82,
                size: 2,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 14,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(8),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        // Three loads so s_endpgm sits at index >= 2 (it inspects the two
        // preceding instructions).
        for _ in 0..3 {
            code.get_instructions_mut().push(sload);
        }
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 48;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 14;

        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile extended s_load_dwordx2");
        assert!(
            source.contains("%vsharp"),
            "the load must come from the push-constant table:\n{source}"
        );
        assert!(
            source.contains("s82") && source.contains("s83"),
            "both destination dwords are written:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble extended s_load_dwordx2");
        naga_parse_and_validate(&words, "s_load_dwordx2_extended");
    }

    /// `buffer_load_dwordx4 v[8:11], v[4:5], s[8:11]` with idxen+offen — the
    /// measured menu-VS load (Minecraft PPSA17221, vs+0x334). vindex=v4,
    /// voffset=v5: the per-thread offset must add into the byte address.
    #[test]
    fn buffer_load_dwordx4_offen_adds_the_per_thread_offset() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0xE038_3000, // mubuf op 0x0e (load_dwordx4), idxen+offen
                0x8002_0804, // soffset=0x80, srsrc=s8, vdata=v8, vaddr=v4
                0xBF80_0000,
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse buffer_load_dwordx4 with offen");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadDwordX4);
        assert_eq!(inst.format, F::Vdata4Vaddr2SvSoffsOffenIdxen);
        assert_eq!(inst.src[0].size, 2, "offen makes vaddr a pair");
        assert_eq!(inst.dst.size, 4);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 48;
        input_info.bind.storage_buffers.buffers_num = 1;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile buffer_load_dwordx4");
        assert_eq!(
            source.matches("%t110_").count(),
            4,
            "four consecutive dword loads:\n{source}"
        );
        assert!(
            source.contains("OpIAdd %int"),
            "the per-thread offset adds into the address:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble buffer_load_dwordx4");
        // NOTE: naga's validator rejects Kyty's `%_arr_BufferObject_uint_N`
        // pattern (an array of a struct containing a runtime array), which the
        // Vulkan driver accepts — every storage-buffer module tonight ran
        // through it. This test asserts on assembly + source shape instead.
        let _ = words;
    }
    /// A cube-bound shader emits `OpTypeImage %float Cube` and samples with a
    /// 3-component direction — measured on Minecraft's skybox PS (type 11 T#,
    /// `ImageSample [Vdata4Vaddr3StSsDmaskF]`).
    #[test]
    fn cube_texture_emits_cube_image_and_vec3_coords() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        let sample = ShaderInstruction {
            type_: T::ImageSample,
            format: F::Vdata4Vaddr3StSsDmaskF,
            src_num: 3,
            dst: ShaderOperand {
                type_: ShaderOperandType::Vgpr,
                register_id: 2,
                size: 4,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Vgpr,
                    register_id: 6,
                    size: 3,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 0,
                    size: 8,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 8,
                    size: 4,
                    ..Default::default()
                },
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        for _ in 0..3 {
            code.get_instructions_mut().push(sample.clone());
        }
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 11 << 28; // Cube
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8; // S# lives at s8..s11

        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile cube sample");
        assert!(
            source.contains("OpTypeImage %float Cube"),
            "the cube image type:\n{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v3float"),
            "3-component cube direction:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble cube sample");
    }

    #[test]
    fn s_bfe_u32_recompiles_the_measured_literal_extract() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                // Minecraft PPSA17221: s_bfe_u32 vcc_hi, s3, 0x00080008
                // (extract eight bits starting at bit eight).
                0x93EB_FF03,
                0x0008_0008,
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
            &mut code,
            true,
        )
        .expect("parse measured s_bfe_u32");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SBfeU32);
        assert_eq!(inst.dst.type_, ShaderOperandType::VccHi);
        assert_eq!(inst.src[0].register_id, 3);
        assert_eq!(inst.src[1].constant.u, 0x0008_0008);

        let entry = recomp_func(T::SBfeU32, F::SVdstSVsrc0SVsrc1).expect("SBfeU32 row");
        assert!(matches!(entry.func, RecompileFn::Func(_)));
        assert_eq!(entry.scc_check, SccCheck::NonZero);

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile measured s_bfe_u32");
        assert!(source.contains("OpBitFieldUExtract %uint"));
        assert!(source.contains("OpStore %scc"));
        let words = spirv_run(&source).expect("assemble measured s_bfe_u32");
        naga_parse_and_validate(&words, "s_bfe_u32");
    }

    #[test]
    fn s_and_b32_recompiles_the_measured_literal_mask() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                // Minecraft PPSA17221: s_and_b32 s0, s3, 255.
                0x8700_FF03,
                0x0000_00ff,
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
            &mut code,
            true,
        )
        .expect("parse measured s_and_b32");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SAndB32);
        assert_eq!(inst.dst.register_id, 0);
        assert_eq!(inst.src[0].register_id, 3);
        assert_eq!(inst.src[1].constant.u, 0xff);

        let entry = recomp_func(T::SAndB32, F::SVdstSVsrc0SVsrc1).expect("SAndB32 row");
        assert!(matches!(entry.func, RecompileFn::Func(_)));
        assert_eq!(entry.scc_check, SccCheck::NonZero);

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile measured s_and_b32");
        assert!(source.contains("OpBitwiseAnd %uint"));
        assert!(source.contains("OpStore %scc"));
        let words = spirv_run(&source).expect("assemble measured s_and_b32");
        naga_parse_and_validate(&words, "s_and_b32");
    }

    #[test]
    fn scalar_b32_family_is_wired_with_kyty_scc_semantics() {
        let cases = [
            (T::SAndB32, SccCheck::NonZero),
            (T::SBfmB32, SccCheck::None),
            (T::SCselectB32, SccCheck::None),
            (T::SLshlB32, SccCheck::NonZero),
            (T::SLshrB32, SccCheck::NonZero),
            (T::SOrB32, SccCheck::NonZero),
        ];

        for (ty, scc) in cases {
            let entry =
                recomp_func(ty, F::SVdstSVsrc0SVsrc1).unwrap_or_else(|| panic!("{ty:?} row"));
            assert!(matches!(entry.func, RecompileFn::Func(_)), "{ty:?}");
            assert_eq!(entry.scc_check, scc, "{ty:?}");
        }
    }

    /// `s_not_b64` / `s_brev_b32` — both blocked ASTRO.BOT's compute shaders.
    /// The s_not_b64 word is the MEASURED encoding from the title's failure
    /// log (`raw 0xbefe087e` = `s_not_b64 exec, exec`, the exec-mask invert);
    /// s_brev_b32 is built on the same SOP1 base (0xBE80_0000 | sdst<<16 |
    /// op<<8 | ssrc0).
    #[test]
    fn s_not_b64_and_s_brev_b32_are_wired_with_gcn_scc_semantics() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0xBEFE_087E, // s_not_b64 exec, exec  (measured, ASTRO.BOT)
                0xBE80_0B00, // s_brev_b32 s0, s0
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("both SOP1 opcodes parse");

        let insts = code.get_instructions();
        assert_eq!(insts[0].type_, T::SNotB64);
        assert_eq!(insts[0].dst.type_, ShaderOperandType::ExecLo);
        assert_eq!(insts[0].dst.size, 2, "s_not_b64 is a 64-bit destination");
        assert_eq!(insts[0].src[0].size, 2);
        assert_eq!(insts[1].type_, T::SBrevB32);

        // Both must reach a real recompiler, with GCN's SCC semantics.
        let not64 = recomp_func(T::SNotB64, F::Sdst2Ssrc02).expect("SNotB64 row");
        assert!(matches!(not64.func, RecompileFn::Func(_)));
        assert_eq!(not64.scc_check, SccCheck::NonZero, "s_not_b64 sets SCC");

        let brev = recomp_func(T::SBrevB32, F::SVdstSVsrc0).expect("SBrevB32 row");
        assert!(matches!(brev.func, RecompileFn::Func(_)));
        assert_eq!(
            brev.scc_check,
            SccCheck::None,
            "s_brev_b32 must NOT write SCC"
        );
    }

    #[test]
    fn s_lshl_b32_recompiles_the_measured_vcc_shift() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                // Minecraft PPSA17221: s_lshl_b32 vcc_lo, vcc_hi, 12.
                0x8F6A_8C6B,
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
            &mut code,
            true,
        )
        .expect("parse measured s_lshl_b32");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SLshlB32);
        assert_eq!(inst.dst.type_, ShaderOperandType::VccLo);
        assert_eq!(inst.src[0].type_, ShaderOperandType::VccHi);
        assert_eq!(inst.src[1].constant.u, 12);

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile measured s_lshl_b32");
        assert!(source.contains("OpShiftLeftLogical %uint"));
        assert!(source.contains("OpStore %scc"));
        let words = spirv_run(&source).expect("assemble measured s_lshl_b32");
        naga_parse_and_validate(&words, "s_lshl_b32");
    }

    /// The `V_XXX_I32_SVdstSVsrc0SVsrc1` family (Ashr / Ashrrev / MulLo), wired
    /// from the staged set: `recompile_v_xxx_i32_svdst_svsrc01` was written but
    /// unreachable, so Minecraft's `v_ashrrev_i32` failed its whole shader.
    ///
    /// The pin that matters is Ashr vs Ashrrev: they differ ONLY in which
    /// operand is the shift amount, so transposing them still emits valid SPIR-V
    /// and silently shifts by the wrong value.
    #[test]
    fn v_xxx_i32_svdst_family_is_wired_and_ashrrev_reverses_operands() {
        for ty in [T::VAshrI32, T::VAshrrevI32, T::VMulLoI32] {
            let entry =
                recomp_func(ty, F::SVdstSVsrc0SVsrc1).unwrap_or_else(|| panic!("{ty:?} row"));
            assert!(
                matches!(entry.func, RecompileFn::Func(_)),
                "{ty:?} must be wired, not NI"
            );
        }

        // v_ashr_i32: shift amount is src1; the value shifted is src0.
        let ashr = recomp_func(T::VAshrI32, F::SVdstSVsrc0SVsrc1).expect("row");
        assert!(
            ashr.param[0].expect("mask").contains("%t1_"),
            "v_ashr_i32 takes its shift amount from src1"
        );
        assert!(
            ashr.param[1].expect("shift").contains("%t0_"),
            "v_ashr_i32 shifts src0"
        );

        // v_ashrrev_i32: reversed — shift amount is src0, value is src1.
        let rev = recomp_func(T::VAshrrevI32, F::SVdstSVsrc0SVsrc1).expect("row");
        assert!(
            rev.param[0].expect("mask").contains("%t0_"),
            "v_ashrrev_i32 takes its shift amount from src0"
        );
        let shift = rev.param[1].expect("shift");
        assert!(
            shift.contains("OpShiftRightArithmetic") && shift.contains("%t1_"),
            "v_ashrrev_i32 arithmetic-shifts src1, got: {shift}"
        );
    }

    #[test]
    fn kyty_gated_vop3_trio_is_wired_with_sharpemu_gen5_semantics() {
        for ty in [T::VAndOrB32, T::VLshlOrU32, T::VOr3U32] {
            let entry =
                recomp_func(ty, F::VdstVsrc0Vsrc1Vsrc2).unwrap_or_else(|| panic!("{ty:?} row"));
            assert!(
                matches!(entry.func, RecompileFn::Func(_)),
                "{ty:?} must be implemented, not NI"
            );
        }

        // v_and_or_b32: (s0 & s1) | s2
        let and_or = recomp_func(T::VAndOrB32, F::VdstVsrc0Vsrc1Vsrc2).expect("row");
        let and = and_or.param[0].expect("and step");
        let or = and_or.param[1].expect("or step");
        assert!(
            and.contains("OpBitwiseAnd") && and.contains("%t0_") && and.contains("%t1_"),
            "must AND src0 with src1, got: {and}"
        );
        assert!(
            or.contains("OpBitwiseOr") && or.contains("%ta_") && or.contains("%t2_"),
            "must OR that with src2, got: {or}"
        );

        // v_lshl_or_u32: (s0 << (s1 & 31)) | s2 — the shift amount MUST be
        // masked to 31, and the fold MUST be OR (v_lshl_ADD_u32 is the ADD twin;
        // confusing them silently corrupts the value).
        let lshl_or = recomp_func(T::VLshlOrU32, F::VdstVsrc0Vsrc1Vsrc2).expect("row");
        assert!(
            lshl_or.param[0].expect("mask").contains("%uint_31"),
            "shift amount must be masked to 31"
        );
        assert!(
            lshl_or.param[1]
                .expect("shift")
                .contains("OpShiftLeftLogical"),
            "must shift src0 left"
        );
        let fold = lshl_or.param[2].expect("fold");
        assert!(
            fold.contains("OpBitwiseOr") && fold.contains("%t2_"),
            "must OR (not ADD) src2, got: {fold}"
        );
        assert!(
            recomp_func(T::VLshlAddU32, F::VdstVsrc0Vsrc1Vsrc2)
                .expect("row")
                .param[2]
                .expect("fold")
                .contains("OpIAdd"),
            "the ADD twin must stay ADD"
        );

        // v_or3_u32: (s0 | s1) | s2
        let or3 = recomp_func(T::VOr3U32, F::VdstVsrc0Vsrc1Vsrc2).expect("row");
        assert!(or3.param[0].expect("or0").contains("OpBitwiseOr"));
        assert!(
            or3.param[1].expect("or1").contains("%t2_"),
            "must fold in src2"
        );
    }

    #[test]
    fn v_cmpx_i32_block_is_wired_and_compares_signed() {
        // (type, expected SPIR-V op). Ordering ops MUST be signed: these share
        // one lowering with their unsigned twins, so the op is the only thing
        // separating them — swap it and the shader silently compares as the
        // wrong signedness, which no other test would catch.
        let signed = [
            (T::VCmpxLtI32, "OpSLessThan"),
            (T::VCmpxGeI32, "OpSGreaterThanEqual"),
            (T::VCmpxGtI32, "OpSGreaterThan"),
            (T::VCmpxLeI32, "OpSLessThanEqual"),
            // Equality is sign-agnostic, hence the same ops as the U32 rows.
            (T::VCmpxEqI32, "OpIEqual"),
            (T::VCmpxNeI32, "OpINotEqual"),
        ];
        for (ty, op) in signed {
            let entry = recomp_func(ty, F::SmaskVsrc0Vsrc1).unwrap_or_else(|| panic!("{ty:?} row"));
            assert!(
                matches!(entry.func, RecompileFn::Func(_)),
                "{ty:?} must be implemented, not NI"
            );
            assert_eq!(
                entry.param[0],
                Some(op),
                "{ty:?} compares with the wrong op"
            );
        }

        // The unsigned twins must be untouched by the signed block landing.
        for (ty, op) in [
            (T::VCmpxLtU32, "OpULessThan"),
            (T::VCmpxGeU32, "OpUGreaterThanEqual"),
            (T::VCmpxGtU32, "OpUGreaterThan"),
        ] {
            let entry = recomp_func(ty, F::SmaskVsrc0Vsrc1).unwrap_or_else(|| panic!("{ty:?} row"));
            assert_eq!(entry.param[0], Some(op), "{ty:?} must stay UNSIGNED");
        }
    }

    #[test]
    fn astro_scene_mimg_and_saveexec_rows_are_wired() {
        // The ASTRO.BOT scene-compute batch: image_load dmask 1/7, the
        // fullscreen composite's image_sample_lz rgba, and s_orn2_saveexec.
        // Each row must exist and be implemented (not NI) so the shaders that
        // lead with these instructions translate instead of skipping.
        for (ty, fmt) in [
            (T::ImageLoad, F::Vdata1Vaddr3StDmask1),
            (T::ImageLoad, F::Vdata3Vaddr3StDmask7),
            (T::ImageSampleLz, F::Vdata4Vaddr3StSsDmaskF),
            (T::SOrn2SaveexecB64, F::Sdst2Ssrc02),
        ] {
            let entry = recomp_func(ty, fmt).unwrap_or_else(|| panic!("{ty:?} row missing"));
            assert!(
                matches!(entry.func, RecompileFn::Func(_)),
                "{ty:?} must be implemented, not NI"
            );
        }
    }

    #[test]
    fn astro_scalar_address_and_pack_rows_recompile() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        code.set_base_address(0x0000_0005_0074_e000);
        shader_parse(0, &[0xBE80_1F00, 0x9935_806B, S_ENDPGM], &mut code, true)
            .expect("parse scalar opcode batch");

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile scalar opcode batch");
        assert!(source.contains("OpStore %s0 %uint_0x0074e004"), "{source}");
        assert!(source.contains("OpBitwiseAnd %uint"), "{source}");
        assert!(source.contains("OpShiftLeftLogical %uint"), "{source}");
        let words = spirv_run(&source).expect("assemble scalar opcode batch");
        naga_parse_and_validate(&words, "scalar opcode batch");
    }

    #[test]
    fn astro_vop1_sdwa_omod_recompiles_as_float_multiply() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0x7E02_54F9, 0x0026_4605, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse VOP1 SDWA omod");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile VOP1 SDWA omod");
        assert!(source.contains("OpFMul %float"), "{source}");
        let words = spirv_run(&source).expect("assemble VOP1 SDWA omod");
        naga_parse_and_validate(&words, "VOP1 SDWA omod");
    }

    #[test]
    fn astro_address_only_buffer_load_uses_zero_index() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xE038_0000, 0x8001_0400, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse address-only buffer load");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.storage_buffers.buffers_num = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile address-only buffer load");
        assert!(source.contains("OpStore %temp_int_1 %int_0"), "{source}");
        let _ = spirv_run(&source).expect("assemble address-only buffer load");
    }

    #[test]
    fn astro_image_get_resinfo_queries_xy_dimensions() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF038_0308, 0x0001_0400, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_get_resinfo");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_get_resinfo");
        assert!(source.contains("OpImageQuerySizeLod %v2int"), "{source}");
        let words = spirv_run(&source).expect("assemble image_get_resinfo");
        naga_parse_and_validate(&words, "image_get_resinfo");
    }

    #[test]
    fn astro_image_sample_c_lz_manually_compares_depth_at_lod_zero() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF0BC_0100, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_sample_c_lz");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 12;
        input_info.bind.samplers.binding_index = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_sample_c_lz");
        assert!(
            source.contains("OpImageSampleExplicitLod %v4float"),
            "{source}"
        );
        assert!(source.contains("Lod %float_0_000000"), "{source}");
        assert!(source.contains("OpFOrdLessThanEqual %bool"), "{source}");
        assert!(source.contains("OpSelect %float"), "{source}");
        let words = spirv_run(&source).expect("assemble image_sample_c_lz");
        naga_parse_and_validate(&words, "image_sample_c_lz");
    }

    #[test]
    fn v_cmpx_nlt_f32_is_wired_as_exec_writing_unordered_ge() {
        // VOPC 0x1e: `exec/smask = !(vsrc0 < vsrc1)`. With NaN, "not less than"
        // is the UNORDERED ≥ (NaN → true), the same lowering as the non-exec
        // VCmpNltF32. Measured in ASTRO.BOT scene compute (raw 0x7c3c66f9). Must
        // be implemented (not NI) and lower via the exec-writing vcmpx path.
        let entry = recomp_func(T::VCmpxNltF32, F::SmaskVsrc0Vsrc1)
            .unwrap_or_else(|| panic!("VCmpxNltF32 row"));
        assert!(
            matches!(entry.func, RecompileFn::Func(_)),
            "VCmpxNltF32 must be implemented, not NI"
        );
        assert_eq!(
            entry.param[0],
            Some("OpFUnordGreaterThanEqual"),
            "v_cmpx_nlt is unordered ≥ (NaN → true), not an ordered compare"
        );
        // Its ordered exec-writing twin must stay distinct (OpFOrdLessThan), so
        // a wrong-op regression on either can't hide behind the other.
        let lt = recomp_func(T::VCmpxLtF32, F::SmaskVsrc0Vsrc1)
            .unwrap_or_else(|| panic!("VCmpxLtF32 row"));
        assert_eq!(
            lt.param[0],
            Some("OpFOrdLessThan"),
            "VCmpxLtF32 must stay ordered <"
        );
    }

    #[test]
    fn scalar_compare_family_is_wired() {
        let signed = [
            T::SCmpEqI32,
            T::SCmpGeI32,
            T::SCmpGtI32,
            T::SCmpLgI32,
            T::SCmpLtI32,
            T::SCmpLeI32,
        ];
        let unsigned = [
            T::SCmpEqU32,
            T::SCmpGeU32,
            T::SCmpGtU32,
            T::SCmpLeU32,
            T::SCmpLtU32,
            T::SCmpLgU32,
        ];

        for ty in signed.into_iter().chain(unsigned) {
            let entry = recomp_func(ty, F::Ssrc0Ssrc1).unwrap_or_else(|| panic!("{ty:?} row"));
            assert!(matches!(entry.func, RecompileFn::Func(_)), "{ty:?}");
            assert_eq!(entry.scc_check, SccCheck::None, "{ty:?}");
        }
    }

    #[test]
    fn s_cmp_eq_i32_recompiles_the_measured_vcc_comparison() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                // Minecraft PPSA17221: s_cmp_eq_i32 0, vcc_lo.
                0xBF00_6A80,
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
            &mut code,
            true,
        )
        .expect("parse measured s_cmp_eq_i32");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SCmpEqI32);
        assert_eq!(inst.src[0].constant.u, 0);
        assert_eq!(inst.src[1].type_, ShaderOperandType::VccLo);

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile measured s_cmp_eq_i32");
        assert!(source.contains("OpIEqual %bool"));
        assert!(source.contains("OpStore %scc"));
        let words = spirv_run(&source).expect("assemble measured s_cmp_eq_i32");
        naga_parse_and_validate(&words, "s_cmp_eq_i32");
    }

    #[test]
    fn v_lshl_add_u32_recompiles_the_measured_rdna2_encoding() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                // Minecraft PPSA17221 menu CS (first RDNA2-only op measured):
                // v_lshl_add_u32 v0, v1, v2, v3 — VOP3 opcode 0x346.
                0xD746_0000,
                0x040E_0501,
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
            &mut code,
            true,
        )
        .expect("parse measured v_lshl_add_u32");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VLshlAddU32);
        assert_eq!(inst.format, F::VdstVsrc0Vsrc1Vsrc2);
        assert_eq!(inst.dst.register_id, 0);
        assert_eq!(
            [
                inst.src[0].register_id,
                inst.src[1].register_id,
                inst.src[2].register_id
            ],
            [1, 2, 3]
        );

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile measured v_lshl_add_u32");
        assert!(source.contains("OpShiftLeftLogical %uint"));
        assert!(source.contains("OpIAdd %uint"));
        let words = spirv_run(&source).expect("assemble measured v_lshl_add_u32");
        naga_parse_and_validate(&words, "v_lshl_add_u32");
    }

    #[test]
    fn v_cube_f32_family_is_wired_and_validates() {
        // v_cube{id,sc,tc,ma}_f32 v0, v1, v2, v3 — VOP3 0x144-0x147, the
        // cubemap-coordinate helpers Minecraft's skybox PS uses (measured
        // encoding for cubema was 0xd547...). Each must parse, recompile via
        // the shadPS4-derived SelectCubeResult path, and assemble to
        // naga-valid SPIR-V.
        for (opcode_dw0, ty, marker) in [
            (0xD544_0000u32, T::VCubeIdF32, "v_cubeid_f32"),
            (0xD545_0000u32, T::VCubeScF32, "v_cubesc_f32"),
            (0xD546_0000u32, T::VCubeTcF32, "v_cubetc_f32"),
            (0xD547_0000u32, T::VCubeMaF32, "v_cubema_f32"),
        ] {
            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Vertex);
            shader_parse(
                0,
                &[
                    opcode_dw0,
                    0x040E_0501, // src0=v1, src1=v2, src2=v3
                    0x7E00_02FF,
                    0x3F80_0000, // v_mov_b32 v0, 1.0
                    0x7E02_0280, // v_mov_b32 v1, 0
                    0x1004_0300,
                    0xF800_08CF,
                    0x0302_0100, // exp
                    0xF800_020F,
                    0x0302_0100, // exp
                    S_ENDPGM,
                ],
                &mut code,
                true,
            )
            .unwrap_or_else(|e| panic!("parse {marker}: {e:?}"));

            let inst = &code.get_instructions()[0];
            assert_eq!(inst.type_, ty, "{marker} type");
            assert_eq!(inst.format, F::VdstVsrc0Vsrc1Vsrc2, "{marker} format");
            assert_eq!(
                [
                    inst.src[0].register_id,
                    inst.src[1].register_id,
                    inst.src[2].register_id
                ],
                [1, 2, 3],
                "{marker} sources"
            );

            let source = spirv_generate_source(
                &code,
                Some(&ShaderVertexInputInfo {
                    export_count: 1,
                    ..Default::default()
                }),
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("recompile {marker}: {e:?}"));
            // Every cube op picks the major axis with |z|>=|x|&&|z|>=|y| etc.
            assert!(
                source.contains("OpFOrdGreaterThanEqual %bool"),
                "{marker} major-axis compare"
            );
            assert!(
                source.contains("%GLSL_std_450 FAbs"),
                "{marker} component abs"
            );
            let words = spirv_run(&source).unwrap_or_else(|e| panic!("assemble {marker}: {e:?}"));
            naga_parse_and_validate(&words, marker);
        }
    }

    #[test]
    fn v_cvt_flr_i32_f32_is_wired_and_validates() {
        // v_cvt_flr_i32_f32 v0, v1 — VOP1 0xd, floor(float)→signed-int. Measured
        // in ASTRO.BOT's scene compute shaders (raw 0x7e681b01 = vdst v52); here
        // vdst=v0 for a compact test. Must parse to VCvtFlrI32F32 and recompile
        // to a GLSL Floor followed by OpConvertFToS (not Trunc — that is the
        // truncating VCvtI32F32 sibling).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                0x7E00_1B01, // v_cvt_flr_i32_f32 v0, v1
                0x7E00_02FF,
                0x3F80_0000, // v_mov_b32 v0, 1.0
                0x7E02_0280, // v_mov_b32 v1, 0
                0x1004_0300,
                0xF800_08CF,
                0x0302_0100, // exp
                0xF800_020F,
                0x0302_0100, // exp
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .unwrap_or_else(|e| panic!("parse v_cvt_flr_i32_f32: {e:?}"));

        assert_eq!(code.get_instructions()[0].type_, T::VCvtFlrI32F32);

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("recompile v_cvt_flr_i32_f32: {e:?}"));
        assert!(
            source.contains("%GLSL_std_450 Floor"),
            "v_cvt_flr must floor toward −∞ before the int conversion"
        );
        assert!(
            source.contains("OpConvertFToS %int"),
            "v_cvt_flr result is a signed int"
        );
        let words =
            spirv_run(&source).unwrap_or_else(|e| panic!("assemble v_cvt_flr_i32_f32: {e:?}"));
        naga_parse_and_validate(&words, "v_cvt_flr_i32_f32");
    }

    #[test]
    fn scalar_b64_boolean_family_is_wired_with_kyty_scc_semantics() {
        let nonzero = [
            T::SAndn2B64,
            T::SOrn2B64,
            T::SAndB64,
            T::SNorB64,
            T::SNandB64,
            T::SXnorB64,
            T::SOrB64,
            T::SXorB64,
        ];

        for ty in nonzero {
            let entry =
                recomp_func(ty, F::Sdst2Ssrc02Ssrc12).unwrap_or_else(|| panic!("{ty:?} row"));
            assert!(matches!(entry.func, RecompileFn::Func(_)), "{ty:?}");
            assert_eq!(entry.scc_check, SccCheck::NonZero, "{ty:?}");
        }

        let select = recomp_func(T::SCselectB64, F::Sdst2Ssrc02Ssrc12).expect("SCselectB64 row");
        assert!(matches!(select.func, RecompileFn::Func(_)));
        assert_eq!(select.scc_check, SccCheck::None);
    }

    #[test]
    fn s_cselect_b64_recompiles_the_measured_exec_select() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                // Minecraft PPSA17221: s_cselect_b64 s[6:7], exec, 0.
                0x8586_807E,
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
            &mut code,
            true,
        )
        .expect("parse measured s_cselect_b64");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SCselectB64);
        assert_eq!(inst.dst.register_id, 6);
        assert_eq!(inst.src[0].type_, ShaderOperandType::ExecLo);
        assert_eq!(inst.src[1].constant.u, 0);

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile measured s_cselect_b64");
        assert!(source.contains("OpSelect %uint"));
        assert!(source.contains("OpStore %s6"));
        assert!(source.contains("OpStore %s7"));
        let words = spirv_run(&source).expect("assemble measured s_cselect_b64");
        naga_parse_and_validate(&words, "s_cselect_b64");
    }

    #[test]
    fn scalar_u32_arithmetic_family_is_wired_with_kyty_scc_semantics() {
        for ty in [T::SAddcU32, T::SAddU32, T::SSubU32, T::SLshl4AddU32] {
            let entry =
                recomp_func(ty, F::SVdstSVsrc0SVsrc1).unwrap_or_else(|| panic!("{ty:?} row"));
            assert!(matches!(entry.func, RecompileFn::Func(_)), "{ty:?}");
            assert_eq!(entry.scc_check, SccCheck::CarryOut, "{ty:?}");
        }

        for (ty, scc) in [
            (T::SBfeU32, SccCheck::NonZero),
            (T::SMulHiU32, SccCheck::None),
        ] {
            let entry =
                recomp_func(ty, F::SVdstSVsrc0SVsrc1).unwrap_or_else(|| panic!("{ty:?} row"));
            assert!(matches!(entry.func, RecompileFn::Func(_)), "{ty:?}");
            assert_eq!(entry.scc_check, scc, "{ty:?}");
        }
    }

    #[test]
    fn s_lshl4_add_u32_recompiles_the_measured_address_build() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                // Minecraft PPSA17221:
                // s_lshl4_add_u32 vcc_hi, vcc_lo, 0x000c0000.
                0x98EB_FF6A,
                0x000C_0000,
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
            &mut code,
            true,
        )
        .expect("parse measured s_lshl4_add_u32");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SLshl4AddU32);
        assert_eq!(inst.dst.type_, ShaderOperandType::VccHi);
        assert_eq!(inst.src[0].type_, ShaderOperandType::VccLo);
        assert_eq!(inst.src[1].constant.u, 0x000c_0000);

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile measured s_lshl4_add_u32");
        assert!(source.contains("OpFunctionCall %v2uint %lshl_add"));
        assert!(source.contains("OpStore %scc %carry_0"));
        let words = spirv_run(&source).expect("assemble measured s_lshl4_add_u32");
        // `OpIAddCarry` is core SPIR-V and is used by Kyty's faithful
        // `lshl_add` helper. naga 24 does not parse it, so assembly plus the
        // real Vulkan title run is the appropriate gate for this row.
        assert_eq!(words.first().copied(), Some(0x0723_0203));
    }

    #[test]
    fn vcvt_pkrtz_f16_f32_recompiles_the_measured_minecraft_pixel_sequence() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                // Minecraft PPSA17221 pixel shader at guest 0x253c4d00:
                // v_cvt_pkrtz_f16_f32 v0, s4, s5
                0xD52F_0000,
                0x0000_0A04,
                // v_cvt_pkrtz_f16_f32 v1, s6, s7
                0xD52F_0001,
                0x0000_0E06,
                // exp mrt0 v0, v1 compr vm done
                0xF800_1C0F,
                0x0000_0100,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse measured next-gen pixel sequence");

        let first = &code.get_instructions()[0];
        assert_eq!(first.type_, T::VCvtPkrtzF16F32);
        assert_eq!(first.dst.register_id, 0);
        assert_eq!(first.src[0].register_id, 4);
        assert_eq!(first.src[1].register_id, 5);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile measured v_cvt_pkrtz_f16_f32 sequence");
        assert!(source.contains("PackHalf2x16"), "{source}");
        assert!(
            source.contains("OpBitwiseAnd %uint %t0u_0 %uint_0xffffe000"),
            "{source}"
        );
        let words = spirv_run(&source).expect("assemble measured v_cvt_pkrtz_f16_f32");
        naga_parse_and_validate(&words, "measured v_cvt_pkrtz_f16_f32");
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

    /// Every newly-wired row carries **exactly** the `SccCheck` Kyty's table
    /// gives it — no more, no less.
    ///
    /// The `f`/`fs` split cuts both ways, and each direction is a silent
    /// wrong-answer bug that compiles and passes an "is it wired?" check:
    ///
    /// * `f()` hardcodes `SccCheck::None`. Routing an SCC-bearing row through
    ///   it makes `get_scc_check` return `""`, so `scc` is **never updated**
    ///   after e.g. `s_wqm_b64`.
    /// * `fs()` carries whatever it is handed. Routing a `None` row through it
    ///   with a made-up check **invents** an `scc` write Kyty does not emit.
    ///
    /// The values below were each read out of Kyty's `g_recomp_func` row, not
    /// guessed: `DsAppend`/`DsConsume`/`ImageLoad`/`ImageStore`/`SMulkI32` omit
    /// the 5th initializer, so `scc_check` takes the struct's default-member
    /// initializer (`SccCheck::None`, ShaderSpirv.cpp L1555); the rest spell
    /// `SccCheck::NonZero` explicitly.
    #[test]
    fn newly_wired_rows_carry_kytys_own_scc_check() {
        // (type, format, expected SccCheck, Kyty g_recomp_func evidence)
        let cases: &[(T, F, SccCheck, &str)] = &[
            (
                T::DsAppend,
                F::VdstGds,
                SccCheck::None,
                "L6197: no 5th initializer",
            ),
            (
                T::DsConsume,
                F::VdstGds,
                SccCheck::None,
                "L6198: no 5th initializer",
            ),
            (
                T::ImageLoad,
                F::Vdata4Vaddr3StDmaskF,
                SccCheck::None,
                "no 5th initializer",
            ),
            (
                T::ImageStore,
                F::Vdata4Vaddr3StDmaskF,
                SccCheck::None,
                "no 5th initializer",
            ),
            (
                T::SMulkI32,
                F::SVdstSVsrc0,
                SccCheck::None,
                "no 5th initializer",
            ),
            (
                T::SBfeU64,
                F::Sdst2Ssrc02Ssrc1,
                SccCheck::NonZero,
                "explicit SccCheck::NonZero",
            ),
            (
                T::SAndSaveexecB64,
                F::Sdst2Ssrc02,
                SccCheck::NonZero,
                "explicit SccCheck::NonZero",
            ),
            (
                T::SWqmB64,
                F::Sdst2Ssrc02,
                SccCheck::NonZero,
                "L6350: explicit SccCheck::NonZero",
            ),
            (
                T::SNotB64,
                F::Sdst2Ssrc02,
                SccCheck::NonZero,
                "s_not_b64 sets SCC = (D != 0)",
            ),
            (
                T::SBrevB32,
                F::SVdstSVsrc0,
                SccCheck::None,
                "s_brev_b32 does NOT write SCC",
            ),
        ];

        for (ty, fmt, expected, why) in cases {
            let e = recomp_func(*ty, *fmt)
                .unwrap_or_else(|| panic!("{ty:?}/{fmt:?} must have a table row"));
            assert!(
                matches!(e.func, RecompileFn::Func(_)),
                "{ty:?} is staged and verified faithful — it must be wired, not NotImplemented"
            );
            assert_eq!(
                e.scc_check, *expected,
                "{ty:?} must carry Kyty's own SccCheck ({why}); a mismatch here silently \
                 drops or invents an scc update"
            );
        }
    }

    /// `DsAppend` and `DsConsume` share one Rust helper (`ds_append_consume`)
    /// parameterised by the atomic op, so a transposed argument at the two call
    /// sites would compile, wire, and silently turn every append into a
    /// decrement. The rows must stay distinct and both wired.
    #[test]
    fn ds_append_and_consume_are_distinct_wired_rows() {
        let a = recomp_func(T::DsAppend, F::VdstGds).expect("DsAppend row");
        let c = recomp_func(T::DsConsume, F::VdstGds).expect("DsConsume row");
        assert!(matches!(a.func, RecompileFn::Func(_)));
        assert!(matches!(c.func, RecompileFn::Func(_)));
        assert_ne!(
            a.type_, c.type_,
            "append and consume must not collapse onto one row"
        );
        // Kyty gives both `{\"\"}` — param must survive the ni -> f swap.
        assert_eq!(a.param, p1(""));
        assert_eq!(c.param, p1(""));
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
        // image_sample is wired (texture chain); ImageStoreMip is the MIMG
        // row still C2. Built by hand — guessing a store-mip encoding would
        // test the parser, not the error naming this test is about.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        let inst = ShaderInstruction {
            type_: T::ImageStoreMip,
            format: F::Vdata4Vaddr4StDmaskF,
            ..Default::default()
        };
        code.get_instructions_mut().push(inst);
        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        let err = shader_recompile_ps(&code, &input_info).unwrap_err();
        match err {
            ShaderRecompileError::NotImplemented {
                kyty_func, line, ..
            } => {
                assert_eq!(kyty_func, "Recompile_ImageStoreMip_Vdata4Vaddr4StDmaskF");
                assert_eq!(line, 3173);
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

    fn exp_param_vgpr(id: i32) -> ShaderOperand {
        ShaderOperand {
            type_: crate::shader::types::ShaderOperandType::Vgpr,
            register_id: id,
            size: 1,
            ..Default::default()
        }
    }

    fn recompile_one_param_export(export_enable: u32) -> String {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        let mut inst = ShaderInstruction {
            type_: T::Exp,
            format: F::Param0Vsrc0Vsrc1Vsrc2Vsrc3,
            export_enable,
            src_num: 4,
            ..Default::default()
        };
        inst.src = [
            exp_param_vgpr(0),
            exp_param_vgpr(1),
            exp_param_vgpr(2),
            exp_param_vgpr(3),
        ];
        code.get_instructions_mut().push(inst);

        let spirv = Spirv::new();
        let mut out = String::new();
        recompile_exp_param_xxx(0, &code, &mut out, &spirv, &p1("param0"), SccCheck::None)
            .expect("param export recompiles");
        out
    }

    /// A partial-channel param export (e.g. a `vec2` texcoord, `en=0x3`) writes
    /// its enabled channels and 0 to the rest.
    ///
    /// The earlier recompiler demanded all four sources be variables, so every
    /// partial export failed and took the whole vertex shader with it — which
    /// is why Minecraft's content shaders would not translate. `%float_0_000000`
    /// is always registered by `find_constants`, so the disabled channels get a
    /// defined zero.
    #[test]
    fn exp_param_partial_mask_writes_zero_to_disabled_channels() {
        let out = recompile_one_param_export(0x3); // channels x, y
        assert!(
            out.contains("OpLoad %float %v0"),
            "channel 0 loads v0:\n{out}"
        );
        assert!(
            out.contains("OpLoad %float %v1"),
            "channel 1 loads v1:\n{out}"
        );
        assert!(!out.contains("%v2"), "channel 2 disabled — no load:\n{out}");
        assert!(!out.contains("%v3"), "channel 3 disabled — no load:\n{out}");
        assert!(
            out.contains(
                "OpCompositeConstruct %v4float %t0_0 %t1_0 %float_0_000000 %float_0_000000"
            ),
            "disabled channels must be zero:\n{out}"
        );
    }

    /// A full export (`en=0xf`) is unchanged: all four channels load their vgpr.
    #[test]
    fn exp_param_full_mask_still_writes_all_four_channels() {
        let out = recompile_one_param_export(0xf);
        for i in 0..4 {
            assert!(
                out.contains(&format!("OpLoad %float %v{i}")),
                "channel {i}:\n{out}"
            );
        }
        assert!(
            !out.contains("%float_0_000000"),
            "a full export writes no zeros:\n{out}"
        );
    }
}
