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
    SampledClass, SampledDim, Spirv, SpirvType, SpirvValue, StorageFormat, not_supported,
    operand_is_constant, operand_is_exec, operand_is_variable, operand_load_float,
    operand_load_int, operand_load_uint, operand_variable_to_str, operand_variable_to_str_shift,
    sampled_key_of, sampled_key_suffix, sampled_keys_present, spirv_generate_source,
    spirv_get_embedded_ps, spirv_get_embedded_vs, storage_key_of, storage_key_suffix,
    storage_keys_present, vertex_input_class, vertex_input_types,
};
use crate::shader::resources::{
    ShaderBindResources, ShaderComputeInputInfo, ShaderEmbeddedBufferFetch, ShaderPixelInputInfo,
    ShaderVertexInputInfo,
};
use crate::shader::types::{
    ShaderCode, ShaderInstruction, ShaderInstructionType, ShaderLabel, ShaderOperandType,
    ShaderType, shader_instruction_format::Format,
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

/// Resolve one GFX10 MIMG address component. With no NSA payload, components
/// are consecutive from VADDR as on GCN. With NSA, component zero still uses
/// VADDR and components 1+ use the explicitly encoded VGPR byte array.
fn mimg_address_value(inst: &ShaderInstruction, component: usize) -> SpirvValue {
    if component > 0 && inst.mimg_nsa_dwords != 0 {
        operand_variable_to_str_shift(inst.mimg_nsa_addr[component - 1], 0)
    } else {
        operand_variable_to_str_shift(inst.src[0], component as i32)
    }
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
    buffer_load_dwordxn(
        index,
        code,
        dst_source,
        spirv,
        4,
        "Recompile_BufferLoadDwordX4_Vdata4VaddrSvSoffsIdxen",
    )
}

/// Beyond Kyty (`buffer_load_dwordx2` is `KYTY_NI` upstream): the two-dword
/// row of the same flexible-addressing model — measured on ASTRO.BOT scene
/// compute (raw 0xe0342000, idxen).
fn recompile_buffer_load_dwordx2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    buffer_load_dwordxn(
        index,
        code,
        dst_source,
        spirv,
        2,
        "Recompile_BufferLoadDwordX2_Vdata2VaddrSvSoffsIdxen",
    )
}

/// Beyond Kyty (`buffer_load_dwordx3` is `KYTY_NI` upstream): the three-dword
/// row of the same flexible-addressing model — measured on ASTRO.BOT scene
/// compute (raws 0xe03c2074/0xe03c2034, idxen with a nonzero immediate
/// offset; 116 dispatches in the measured window).
fn recompile_buffer_load_dwordx3(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    buffer_load_dwordxn(
        index,
        code,
        dst_source,
        spirv,
        3,
        "Recompile_BufferLoadDwordX3_Vdata3VaddrSvSoffsIdxen",
    )
}

/// Beyond Kyty: lower an `offen` MUBUF buffer load whose V# is constructed
/// **in-shader** and points at the shader's own embedded vertex data
/// (`shader_detect_embedded_buffer_fetch` snapshotted the window). The per-lane
/// byte offset (`voffset`) is runtime, so each destination dword is *selected*
/// from the captured constants by its runtime dword index — a select-chain over
/// the window. An index outside the window yields 0 (the read degrades rather
/// than faulting or reading an unbound descriptor). Reuses the embedded-constant
/// capture machinery from `sload_dword_extended`; no storage buffer is bound.
fn recompile_embedded_buffer_fetch(
    inst: &ShaderInstruction,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    fetch: &ShaderEmbeddedBufferFetch,
    func: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let voff = operand_variable_to_str(inst.src[0]);
    if voff.type_ != SpirvType::Float {
        return Err(not_supported(func, "offen voffset is not a VGPR"));
    }
    let pc = inst.pc;
    let len = fetch.window_len.min(fetch.window.len() as u32);
    let uint = |u: u32| spirv.get_constant_uint(u);

    // byte offset = voffset_bits (+ inst_offset); dword index = byte >> 2.
    let mut text = format!(
        "        %ebf_vo_{pc} = OpLoad %float %{vo}\n        \
         %ebf_vou_{pc} = OpBitcast %uint %ebf_vo_{pc}\n",
        vo = voff.value,
    );
    let byte_off = if fetch.inst_offset == 0 {
        format!("ebf_vou_{pc}")
    } else {
        text += &format!(
            "        %ebf_bo_{pc} = OpIAdd %uint %ebf_vou_{pc} %{off}\n",
            off = uint(fetch.inst_offset),
        );
        format!("ebf_bo_{pc}")
    };
    text += &format!(
        "        %ebf_di_{pc} = OpShiftRightLogical %uint %{byte_off} %{two}\n",
        two = uint(2),
    );

    for i in 0..fetch.dwords_num as i32 {
        let dst = operand_variable_to_str_shift(inst.dst, i);
        if dst.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected embedded-fetch dst type"));
        }
        let idx = if i == 0 {
            format!("ebf_di_{pc}")
        } else {
            text += &format!(
                "        %ebf_idx_{pc}_{i} = OpIAdd %uint %ebf_di_{pc} %{ic}\n",
                ic = uint(i as u32),
            );
            format!("ebf_idx_{pc}_{i}")
        };
        // acc = (idx == k) ? window[k] : acc, folded over the window (0 default).
        let mut acc = format!("%{}", uint(0));
        for k in 0..len {
            text += &format!(
                "        %ebf_eq_{pc}_{i}_{k} = OpIEqual %bool %{idx} %{kc}\n        \
                 %ebf_sel_{pc}_{i}_{k} = OpSelect %uint %ebf_eq_{pc}_{i}_{k} %{vc} {acc}\n",
                kc = uint(k),
                vc = uint(fetch.window[k as usize]),
            );
            acc = format!("%ebf_sel_{pc}_{i}_{k}");
        }
        text += &format!(
            "        %ebf_fv_{pc}_{i} = OpBitcast %float {acc}\n               \
             OpStore %{d} %ebf_fv_{pc}_{i}\n",
            d = dst.value,
        );
    }

    *dst_source += &text;
    Ok(true)
}

/// Shared body of the n-dword raw MUBUF loads: n consecutive dword loads at
/// `(offset + vindex*stride)/4 + i` (+ per-thread `voffset` when Offen).
fn buffer_load_dwordxn(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    n: i32,
    func: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    let Some(bind_info) = spirv.get_bind_info() else {
        return Ok(false);
    };

    // Beyond Kyty: an `offen` load through an in-shader-constructed V# that
    // points at the shader's own embedded vertex data (see
    // `shader_detect_embedded_buffer_fetch`). Such a V# is never a captured
    // storage-buffer descriptor, so the path below would refuse it
    // (`buffers_num == 0`); serve the runtime-indexed read from the snapshotted
    // embedded window instead.
    if let Some(fetch) = bind_info.embedded_buffer_fetches.find(inst.pc) {
        return recompile_embedded_buffer_fetch(&inst, dst_source, spirv, fetch, func);
    }

    if bind_info.storage_buffers.buffers_num == 0 {
        return Ok(false);
    }
    if !operand_is_constant(inst.src[2]) {
        return Err(not_supported(func, "src2 is not a constant"));
    }

    let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
    let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
    let offset = spirv.get_constant(inst.src[2]);

    if src1_value0.type_ != SpirvType::Uint || src1_value1.type_ != SpirvType::Uint {
        return Err(not_supported(func, "unexpected operand types"));
    }

    let idxen = matches!(
        inst.format,
        Format::Vdata4VaddrSvSoffsIdxen
            | Format::Vdata4Vaddr2SvSoffsOffenIdxen
            | Format::Vdata3VaddrSvSoffsIdxen
            | Format::Vdata3Vaddr2SvSoffsOffenIdxen
            | Format::Vdata2VaddrSvSoffsIdxen
            | Format::Vdata2Vaddr2SvSoffsOffenIdxen
    );
    let offen = matches!(
        inst.format,
        Format::Vdata4Vaddr2SvSoffsOffenIdxen
            | Format::Vdata4VaddrSvSoffsOffen
            | Format::Vdata3Vaddr2SvSoffsOffenIdxen
            | Format::Vdata3VaddrSvSoffsOffen
            | Format::Vdata2Vaddr2SvSoffsOffenIdxen
            | Format::Vdata2VaddrSvSoffsOffen
    );
    let src0_index = idxen.then(|| operand_variable_to_str(inst.src[0]));
    let src0_off = offen.then(|| operand_variable_to_str_shift(inst.src[0], i32::from(idxen)));
    if src0_index
        .as_ref()
        .is_some_and(|value| value.type_ != SpirvType::Float)
    {
        return Err(not_supported(func, "unexpected index register type"));
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
            return Err(not_supported(func, "unexpected offen register type"));
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

    for i in 0..n {
        let dst_value = operand_variable_to_str_shift(inst.dst, i);
        if dst_value.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected dst type"));
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

/// Beyond Kyty (`buffer_store_dwordx4` is `KYTY_NI` upstream): the store twin
/// of [`buffer_load_dwordxn`] — n consecutive dword stores at
/// `(offset + vindex*stride)/4 + i` (+ per-thread `voffset` when Offen),
/// wrapped in the exec_lo guard every Kyty store body uses. Measured on
/// ASTRO.BOT scene compute (raw 0xe0780000, MUBUF 0x1e).
fn recompile_buffer_store_dwordx4(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    buffer_store_dwordxn(
        index,
        code,
        dst_source,
        spirv,
        4,
        "Recompile_BufferStoreDwordX4_Vdata4VaddrSvSoffsIdxen",
    )
}

/// Beyond-Kyty: MUBUF 0x1d `buffer_store_dwordx2` — two-dword raw store,
/// measured in ASTRO.BOT scene compute (0x500757800). Same `buffer_store_dwordxn`
/// machinery as the X4 store with `n = 2`.
fn recompile_buffer_store_dwordx2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    buffer_store_dwordxn(
        index,
        code,
        dst_source,
        spirv,
        2,
        "Recompile_BufferStoreDwordX2_Vdata2VaddrSvSoffsIdxen",
    )
}

fn buffer_store_dwordxn(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    n: i32,
    func: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    let Some(bind_info) = spirv.get_bind_info() else {
        return Ok(false);
    };
    if bind_info.storage_buffers.buffers_num == 0 {
        return Ok(false);
    }
    if !operand_is_constant(inst.src[2]) {
        return Err(not_supported(func, "src2 is not a constant"));
    }

    let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
    let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
    let offset = spirv.get_constant(inst.src[2]);

    if src1_value0.type_ != SpirvType::Uint || src1_value1.type_ != SpirvType::Uint {
        return Err(not_supported(func, "unexpected operand types"));
    }

    // The helper is width-agnostic (`n` dwords), so it accepts the Vdata2 and
    // Vdata4 addressing-quartet formats alike.
    let idxen = matches!(
        inst.format,
        Format::Vdata4VaddrSvSoffsIdxen
            | Format::Vdata4Vaddr2SvSoffsOffenIdxen
            | Format::Vdata2VaddrSvSoffsIdxen
            | Format::Vdata2Vaddr2SvSoffsOffenIdxen
    );
    let offen = matches!(
        inst.format,
        Format::Vdata4Vaddr2SvSoffsOffenIdxen
            | Format::Vdata4VaddrSvSoffsOffen
            | Format::Vdata2Vaddr2SvSoffsOffenIdxen
            | Format::Vdata2VaddrSvSoffsOffen
    );
    let src0_index = idxen.then(|| operand_variable_to_str(inst.src[0]));
    let src0_off = offen.then(|| operand_variable_to_str_shift(inst.src[0], i32::from(idxen)));
    if src0_index
        .as_ref()
        .is_some_and(|value| value.type_ != SpirvType::Float)
    {
        return Err(not_supported(func, "unexpected index register type"));
    }

    let index_str = format!("{index}");
    let mut text = format!(
        "
        %sdxn_e0_{index_str} = OpLoad %uint %exec_lo
        %sdxn_e1_{index_str} = OpINotEqual %bool %sdxn_e0_{index_str} %uint_0
               OpSelectionMerge %sdxn_end_{index_str} None
               OpBranchConditional %sdxn_e1_{index_str} %sdxn_body_{index_str} %sdxn_end_{index_str}
        %sdxn_body_{index_str} = OpLabel
"
    );
    text += &src0_index.map_or_else(
        || "               OpStore %temp_int_1 %int_0\n".to_owned(),
        |value| {
            format!(
                "        %sdxn_i0_{index_str} = OpLoad %float %{src}\n        %sdxn_i1_{index_str} = OpBitcast %int %sdxn_i0_{index_str}\n               OpStore %temp_int_1 %sdxn_i1_{index_str}\n",
                src = value.value,
            )
        },
    );
    text += &format!(
        r#"        %sdxn_s0_{index_str} = OpLoad %uint %{src1_value1}
        %sdxn_s1_{index_str} = OpShiftRightLogical %uint %sdxn_s0_{index_str} %int_16
        %sdxn_s2_{index_str} = OpBitwiseAnd %uint %sdxn_s1_{index_str} %uint_0x00003fff
        %sdxn_s3_{index_str} = OpBitcast %int %sdxn_s2_{index_str}
               OpStore %temp_int_3 %sdxn_s3_{index_str}
        %sdxn_b0_{index_str} = OpLoad %uint %{src1_value0}
        %sdxn_b1_{index_str} = OpBitcast %int %sdxn_b0_{index_str}
               OpStore %temp_int_4 %sdxn_b1_{index_str}
               OpStore %temp_int_2 %{offset}
"#,
        src1_value0 = src1_value0.value,
        src1_value1 = src1_value1.value,
        offset = offset,
    );

    // offen: the vaddr register after the (optional) vindex is a per-thread
    // byte offset folded into temp_int_2.
    if let Some(off) = &src0_off {
        if off.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected offen register type"));
        }
        text += &format!(
            r#"        %sdxn_o0_{index_str} = OpLoad %float %{off}
        %sdxn_o1_{index_str} = OpBitcast %int %sdxn_o0_{index_str}
        %sdxn_o2_{index_str} = OpLoad %int %temp_int_2
        %sdxn_o3_{index_str} = OpIAdd %int %sdxn_o2_{index_str} %sdxn_o1_{index_str}
               OpStore %temp_int_2 %sdxn_o3_{index_str}
"#,
            off = off.value,
        );
    }

    for i in 0..n {
        let vdata = operand_variable_to_str_shift(inst.dst, i);
        if vdata.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected vdata type"));
        }
        if i != 0 {
            text += &format!(
                r#"        %sdxn_a0_{index_str}_{i} = OpLoad %int %temp_int_2
        %sdxn_a1_{index_str}_{i} = OpIAdd %int %sdxn_a0_{index_str}_{i} %int_4
               OpStore %temp_int_2 %sdxn_a1_{index_str}_{i}
"#,
            );
        }
        text += &format!(
            "        %sdxn_c_{index_str}_{i} = OpFunctionCall %void %buffer_store_float1 %{p} %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4\n",
            p = vdata.value,
        );
    }

    text += &format!(
        "               OpBranch %sdxn_end_{index_str}
        %sdxn_end_{index_str} = OpLabel
"
    );

    *dst_source += &text;
    Ok(true)
}

/// One typed-buffer SPIR-V helper, described by the set of element formats it
/// actually implements.
///
/// Kyty's helpers (`TBUFFER_LOAD_FORMAT_X` L772, `TBUFFER_STORE_FORMAT_X` L817,
/// `TBUFFER_STORE_FORMAT_XY` L862, `TBUFFER_LOAD_FORMAT_XYZW` L715 and its
/// beyond-Kyty store twin) each guard their body with an equality test against
/// hardcoded **legacy MTBUF packing** constants and do nothing at all when it
/// fails. For those, `accepted` is that literal set, taken from the constants
/// in [`crate::shader::spirv`], with the format each one spells.
///
/// A beyond-Kyty helper written for the MUBUF path needs no runtime guard at
/// all — the descriptor is known here, at translate time — so for those
/// `accepted` is simply the format the body unpacks
/// (`BUF_LOAD_FORMAT_XYZW_UNORM8`). Either way the meaning is the same: the
/// formats this function will decode correctly.
struct TypedBufferHelper {
    /// The SPIR-V function name, for the refusal message.
    name: &'static str,
    /// `(dfmt * 8 + nfmt, what it spells)` — every value this helper decodes.
    accepted: &'static [(u32, &'static str)],
    /// vdata registers the access covers, for the refusal message.
    channels: i32,
    /// Stores write nothing on a failed guard; loads leave vdata untouched.
    is_store: bool,
    /// Whether the SPIR-V signature ends in a `dfmt_nfmt` parameter — true for
    /// Kyty's guarded `tbuffer_*` helpers, which re-test the format at runtime,
    /// and false for a beyond-Kyty helper written for a single format because
    /// the descriptor is already known at translate time. Decides whether the
    /// emitting row stores `%temp_int_5` and passes it.
    takes_format_arg: bool,
}

/// `%tbuffer_load_format_x`: "dfmt = 4, nfmt = 4 or 7" (`TBUFFER_LOAD_FORMAT_X`).
const TBUF_LOAD_FORMAT_X: TypedBufferHelper = TypedBufferHelper {
    name: "tbuffer_load_format_x",
    accepted: &[(36, "32_UINT"), (39, "32_FLOAT")],
    channels: 1,
    is_store: false,
    takes_format_arg: true,
};

/// `%tbuffer_store_format_x`: same guard as the load (`TBUFFER_STORE_FORMAT_X`).
const TBUF_STORE_FORMAT_X: TypedBufferHelper = TypedBufferHelper {
    name: "tbuffer_store_format_x",
    accepted: &[(36, "32_UINT"), (39, "32_FLOAT")],
    channels: 1,
    is_store: true,
    takes_format_arg: true,
};

/// `%tbuffer_store_format_xy`: "dfmt = 11, nfmt = 4 or 7"
/// (`TBUFFER_STORE_FORMAT_XY`).
const TBUF_STORE_FORMAT_XY: TypedBufferHelper = TypedBufferHelper {
    name: "tbuffer_store_format_xy",
    accepted: &[(92, "32_32_UINT"), (95, "32_32_FLOAT")],
    channels: 2,
    is_store: true,
    takes_format_arg: true,
};

/// `%tbuffer_load_format_xyzw`: "dfmt = 14, nfmt = 4 or 7"
/// (`TBUFFER_LOAD_FORMAT_XYZW`).
const TBUF_LOAD_FORMAT_XYZW: TypedBufferHelper = TypedBufferHelper {
    name: "tbuffer_load_format_xyzw",
    accepted: &[(116, "32_32_32_32_UINT"), (119, "32_32_32_32_FLOAT")],
    channels: 4,
    is_store: false,
    takes_format_arg: true,
};

/// `%tbuffer_store_format_xyzw`: the beyond-Kyty store twin, same 116/119
/// guard (`TBUFFER_STORE_FORMAT_XYZW`).
///
/// The uint half must stay in step with that guard in both directions: the
/// emitting row passes the *resolved* descriptor format (`OpStore %temp_int_5
/// %int_{packed_format}`), so admitting a format here that the SPIR-V body
/// rejects would translate cleanly and then store nothing at all — a silent
/// version of the refusal this table exists to make loud.
const TBUF_STORE_FORMAT_XYZW: TypedBufferHelper = TypedBufferHelper {
    name: "tbuffer_store_format_xyzw",
    accepted: &[(116, "32_32_32_32_UINT"), (119, "32_32_32_32_FLOAT")],
    channels: 4,
    is_store: true,
    takes_format_arg: true,
};

/// `%buffer_load_format_xyzw_unorm8`: the beyond-Kyty four-byte normalized
/// unpack (`BUFFER_LOAD_FORMAT_XYZW_UNORM8`) — legacy `dfmt 10, nfmt 0`, so
/// packed `10 * 8 + 0` = **80**, RDNA2 unified **56**.
///
/// Measured second capability gap of Avatar: Frontiers of Pandora once the
/// unit conversion made the descriptor's format legible (`V# unified format 56
/// (dfmt 10, nfmt 0) is not 32_32_32_32_FLOAT`, 769 occurrences in a 180 s
/// run): the title packs vertex attributes as four normalized bytes.
///
/// Unlike the `tbuffer_*` helpers this one carries no `dfmt_nfmt` parameter and
/// no `OpIEqual` guard, because the format is resolved here rather than in the
/// shader — so `accepted` states what its body decodes, not what a runtime test
/// lets through.
const BUF_LOAD_FORMAT_XYZW_UNORM8: TypedBufferHelper = TypedBufferHelper {
    name: "buffer_load_format_xyzw_unorm8",
    accepted: &[(80, "8_8_8_8_UNORM")],
    channels: 4,
    is_store: false,
    takes_format_arg: false,
};

/// The packed `dfmt * 8 + nfmt` constant a MUBUF typed access must hand its
/// Kyty helper, resolved from the BOUND descriptor at translate time.
///
/// # Why this is not the descriptor field
///
/// MTBUF carries `dfmt`/`nfmt` in the *instruction*, so Kyty's MTBUF rows
/// hardcode the packed number (36 / 39 / 92 / 95 / 119) and its helpers compare
/// against exactly those. MUBUF takes the format from the *descriptor*, where
/// RDNA2 stores the **unified** FORMAT number — `(V#.dword3 >> 12) & 0x7f`,
/// as SharpEmu `Gen5ShaderScalarEvaluator.cs::TryDecodeBufferDescriptor`
/// (L2205) reads it. Every MUBUF body used to extract that field at *runtime*
/// and pass it straight in:
///
/// ```text
/// %t208 = OpShiftRightLogical %uint %t206 %int_12
/// %t210 = OpBitwiseAnd %uint %t208 %uint_127
///         OpStore %temp_int_5 ...
/// ```
///
/// The two numbering schemes do not overlap where it matters —
/// `32_32_32_32_FLOAT` is unified **77** and packed **119**, and 119 is not
/// even a valid unified encoding — so the guard could never pass and every
/// MUBUF typed access was a silent no-op: loads left their destination VGPRs
/// untouched, stores wrote nothing.
///
/// The descriptor is known at translate time (it is the one this shader binds
/// for the V#), so the conversion belongs here, once, as a constant — not in
/// the shader.
///
/// A format no candidate helper serves is refused **by name** and counted in
/// `UNSUPPORTED_BUFFER_FORMAT_SKIPS`: the helper's upstream behaviour there is
/// silent garbage, which is invisible in a log, and a real unpack for the other
/// formats is the follow-up this counter measures.
///
/// # Dispatching over several helpers
///
/// `candidates` is the set of helpers the calling row can emit, and the return
/// says which one the descriptor selected. Most rows have exactly one and pass
/// a single-element slice. The four-channel load has two — `32_32_32_32_FLOAT`
/// through Kyty's `tbuffer_load_format_xyzw` and `8_8_8_8_UNORM` through the
/// beyond-Kyty `buffer_load_format_xyzw_unorm8` — and this is where that choice
/// is made, once, from the bound descriptor. Resolving all candidates in one
/// call keeps the descriptor lookup and the skip counter single-shot, and lets
/// the refusal name every format the row could have served rather than only the
/// first one tried.
fn mubuf_descriptor_packed_format(
    inst: &ShaderInstruction,
    bind_info: &ShaderBindResources,
    spirv: &Spirv<'_>,
    func: &'static str,
    candidates: &[&'static TypedBufferHelper],
) -> Result<(u32, &'static TypedBufferHelper), ShaderRecompileError> {
    let base = inst.src[1].register_id;
    let base_end = base + 3;

    // A gs-prolog VS shifts every bound descriptor's start register by 8, the
    // same convention `shader_bind_vsharp_storage_buffers` binds with.
    let shift_regs = if spirv.get_vs_input_info().is_some_and(|v| v.gs_prolog) {
        8
    } else {
        0
    };
    let count = usize::try_from(bind_info.storage_buffers.buffers_num.max(0))
        .unwrap_or(0)
        .min(bind_info.storage_buffers.buffers.len());
    let Some(i) =
        (0..count).find(|&i| bind_info.storage_buffers.start_register[i] + shift_regs == base)
    else {
        return Err(not_supported(
            func,
            format!(
                "the V# is not one of this shader's bound descriptors, so its element format \
                 is unknown: {:?} [{:?}] V#=s[{base}:{base_end}], pc={:#x}",
                inst.type_, inst.format, inst.pc,
            ),
        ));
    };

    let unified = u32::from(bind_info.storage_buffers.buffers[i].format());
    let packed = crate::shader::spirv::gfx10_unified_to_packed_dfmt_nfmt(unified);
    if let Some(packed) = packed {
        for helper in candidates {
            if helper.accepted.iter().any(|&(a, _)| a == packed) {
                return Ok((packed, helper));
            }
        }
    }

    crate::shader::spirv::UNSUPPORTED_BUFFER_FORMAT_SKIPS
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let decoded = match crate::shader::spirv::gfx10_unified_to_dfmt_nfmt(unified) {
        Some((dfmt, nfmt)) => format!("dfmt {dfmt}, nfmt {nfmt}"),
        None => "no legacy dfmt/nfmt equivalent".to_owned(),
    };
    let served = candidates
        .iter()
        .flat_map(|helper| helper.accepted.iter())
        .map(|&(packed, name)| {
            match crate::shader::spirv::gfx10_packed_to_unified_dfmt_nfmt(packed) {
                Some(unified) => format!("{name} (unified {unified} / packed {packed})"),
                None => format!("{name} (packed {packed})"),
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let helper_name = candidates
        .iter()
        .map(|helper| helper.name)
        .collect::<Vec<_>>()
        .join(" / ");
    // Every candidate for one row is the same access shape, so the consequence
    // of refusing is the same whichever would have been picked; take it from
    // the first, and fall back to the store wording only if there are none.
    let consequence = match candidates.first() {
        Some(helper) if !helper.is_store => format!(
            "leave v[{d}:{d_end}] untouched",
            d = inst.dst.register_id,
            d_end = inst.dst.register_id + helper.channels - 1,
        ),
        _ => "write nothing".to_owned(),
    };
    Err(not_supported(
        func,
        format!(
            "V# unified format {unified} ({decoded}) is not one the {helper_name} helper serves \
             [{served}]: {:?} [{:?}] V#=s[{base}:{base_end}], stride={}, pc={:#x} — the typed \
             helper would {consequence}",
            inst.type_,
            inst.format,
            bind_info.storage_buffers.buffers[i].stride(),
            inst.pc,
        ),
    ))
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

            // The descriptor's unified FORMAT converted to the packed number
            // `%tbuffer_load_format_x` actually tests — see
            // [`mubuf_descriptor_packed_format`]. Kyty extracted dword3 in the
            // shader and handed the unified number straight to the guard, which
            // could never match, so this fetch was a silent no-op.
            let packed_format = mubuf_descriptor_packed_format(
                &inst,
                bind_info,
                spirv,
                FUNC,
                &[&TBUF_LOAD_FORMAT_X],
            )?
            .0;

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
               OpStore %temp_int_5 %int_<packed_format>
        %t110_<index> = OpFunctionCall %void %tbuffer_load_format_x %<p0> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4 %temp_int_5
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0>", &src0_value.value)
                .replace("<offset>", &offset)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value1>", &src1_value1.value)
                .replace("<packed_format>", &format!("{packed_format}"))
                .replace("<p0>", &dst_value.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Beyond Kyty (no upstream row): `buffer_load_format_xyzw v[d:d+3], vindex,
/// s[r:r+3], soffset idxen` — the four-channel MUBUF twin of
/// `Recompile_BufferLoadFormatX_Vdata1VaddrSvSoffsIdxen` (ShaderSpirv.cpp
/// L1937). Measured first blocker of Avatar: Frontiers of Pandora
/// (`can't recompile (no table entry for
/// BufferLoadFormatXyzw/Vdata4VaddrSvSoffsIdxen): ... v[0:3], v5, s[12:15], 0,
/// idxen` — 2398 shader errors in a 180 s window).
///
/// This is the composition of two rows that already exist, not a new rule:
///
/// * addressing and the `stride` / `buffer_index` operands come from the
///   `BufferLoadFormatX` idxen row. `idxen`-only means
///   `byte_offset = vindex * stride + inst_offset` with no `offen` voffset term
///   — KytyPS5 `spirvEmitterMemory.cpp::EmitBufferByteAddress` (L255-278) takes
///   the index from the first address source only when `inst.memory.idxen`, and
///   `EmitBufferAddressFromParts` (L212-253) forms `index * stride + offset` for
///   the non-swizzled case;
/// * the four-channel unpack comes from
///   `Recompile_TBufferLoadFormatXyzw_Vdata4VaddrSvSoffsIdxenFloat4` (L4765).
///
/// The one difference between the MUBUF and MTBUF forms is where the format
/// comes from: MTBUF carries `dfmt`/`nfmt` in the instruction (the `Float4`
/// suffix hardcodes 119), while MUBUF takes it from the descriptor —
/// `unified_format = (V#.dword3 >> 12) & 0x7f`, exactly as SharpEmu
/// `Gen5ShaderScalarEvaluator.cs::TryDecodeBufferDescriptor` (L2205) reads it,
/// and exactly what the `BufferLoadFormatX` row already does.
///
/// Two element formats are served, chosen from the bound descriptor at
/// translate time — there is no runtime format branch in the shader:
///
/// * packed **119** = `32_32_32_32_FLOAT` → Kyty's `%tbuffer_load_format_xyzw`,
///   four dwords straight out of the storage buffer;
/// * packed **80** = `8_8_8_8_UNORM` (unified 56, `dfmt 10, nfmt 0`) →
///   `%buffer_load_format_xyzw_unorm8`, the beyond-Kyty four-byte normalized
///   unpack. Measured on Avatar: Frontiers of Pandora, which packs its vertex
///   attributes this way; see `BUFFER_LOAD_FORMAT_XYZW_UNORM8` for the channel
///   order and the `/ 255.0` rule, both taken from KytyPS5 `BufferFormat.h` and
///   SharpEmu `Gen5SpirvTranslator.cs`.
///
/// Any other format is a named, counted refusal rather than a silent no-op —
/// Kyty's upstream behaviour for the typed helper is to leave the destination
/// VGPRs untouched, which is invisible in a log.
fn recompile_buffer_load_format_xyzw_vdata4(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_BufferLoadFormatXyzw_Vdata4VaddrSvSoffsIdxen";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.storage_buffers.buffers_num == 0 {
            // Kyty's bare `return false` here printed only
            // `can't recompile: <disassembly>`, which is why Avatar: Frontiers
            // of Pandora's 829-per-run blocker read as "no table entry" when
            // the row and its recompiler both exist. Name the precondition.
            return Err(not_supported(
                FUNC,
                format!(
                    "no storage buffer bound for the V#: {:?} [{:?}] V#=s[{base}:{base_end}], \
                     pc={:#x} — the descriptor could not be proved by \
                     shader_bind_vsharp_storage_buffers",
                    inst.type_,
                    inst.format,
                    inst.pc,
                    base = inst.src[1].register_id,
                    base_end = inst.src[1].register_id + 3,
                ),
            ));
        }
        // The element format comes from the DESCRIPTOR, in RDNA2's unified
        // numbering, while the helper's guard tests the legacy MTBUF packing —
        // see [`mubuf_descriptor_packed_format`] for why extracting it in the
        // shader could never match. It also picks WHICH unpack this site emits:
        // the descriptor is known here, so the choice is a constant, not a
        // branch the shader re-evaluates per invocation.
        let (packed_format, helper) = mubuf_descriptor_packed_format(
            &inst,
            bind_info,
            spirv,
            FUNC,
            &[&TBUF_LOAD_FORMAT_XYZW, &BUF_LOAD_FORMAT_XYZW_UNORM8],
        )?;
        {
            if !operand_is_constant(inst.src[2]) {
                return Err(not_supported(FUNC, "src2 is not a constant"));
            }

            let dst_value: Vec<_> = (0..4)
                .map(|i| operand_variable_to_str_shift(inst.dst, i))
                .collect();
            let src0_value = operand_variable_to_str(inst.src[0]);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
            let src1_value3 = operand_variable_to_str_shift(inst.src[1], 3);
            let offset = spirv.get_constant(inst.src[2]);

            if dst_value.iter().any(|v| v.type_ != SpirvType::Float)
                || src0_value.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src1_value1.type_ != SpirvType::Uint
                || src1_value3.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // Addressing is identical for both unpacks — index, offset, stride
            // and buffer_index into the same four temporaries. Only the callee
            // differs, and whether it takes the format argument at all.
            const PROLOG: &str = r#"
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
"#;
            let mut text = PROLOG.to_owned();
            if helper.takes_format_arg {
                text += "               OpStore %temp_int_5 %int_<packed_format>\n";
            }
            text += &format!(
                "        %t110_<index> = OpFunctionCall %void %{name} %<p0> %<p1> %<p2> %<p3> \
                 %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4{fmt_arg}\n",
                name = helper.name,
                fmt_arg = if helper.takes_format_arg {
                    " %temp_int_5"
                } else {
                    ""
                },
            );

            *dst_source += &text
                .replace("<index>", &format!("{index}"))
                .replace("<src0>", &src0_value.value)
                .replace("<offset>", &offset)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value1>", &src1_value1.value)
                .replace("<packed_format>", &format!("{packed_format}"))
                .replace("<p0>", &dst_value[0].value)
                .replace("<p1>", &dst_value[1].value)
                .replace("<p2>", &dst_value[2].value)
                .replace("<p3>", &dst_value[3].value);

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

fn mrt_output_variable(target: u8, func: &'static str) -> Result<String, ShaderRecompileError> {
    match target {
        0 => Ok("%outColor".to_owned()),
        1..=7 => Ok(format!("%outColor{target}")),
        _ => Err(not_supported(
            func,
            format!("EXP colour target {target} is outside MRT0..7"),
        )),
    }
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

    let target = usize::from(inst.export_target);
    let mode = spirv
        .get_ps_input_info()
        .and_then(|info| info.target_output_mode.get(target))
        .copied()
        .unwrap_or(0);
    if mode != 4 {
        return Err(not_supported(
            FUNC,
            format!("target_output_mode[{target}] = {mode}, expected 4"),
        ));
    }
    let output = mrt_output_variable(inst.export_target, FUNC)?;

    // `en` selects individual channels after each packed source is unpacked.
    // A disabled pair's source fields are don't-care, so requiring/loading
    // both packed VGPRs rejects valid partial exports and can read an
    // unwritten register. Disabled colour channels use the export defaults
    // (0, 0, 0, 1), matching the KytyPS5 emitter.
    let en = inst.export_enable;
    let mut text = String::new();
    let mut channels = [
        "%float_0_000000".to_owned(),
        "%float_0_000000".to_owned(),
        "%float_0_000000".to_owned(),
        "%float_1_000000".to_owned(),
    ];

    if en & 0x3 != 0 {
        if !operand_is_variable(inst.src[0]) {
            return Err(not_supported(
                FUNC,
                "enabled packed RG source is not a variable",
            ));
        }
        let src0 = operand_variable_to_str(inst.src[0]).value;
        text += &format!(
            "         %t1_<index> = OpLoad %float %{src0}\n\
             %t2_<index> = OpBitcast %uint %t1_<index>\n\
             %t3_<index> = OpExtInst %v2float %GLSL_std_450 UnpackHalf2x16 %t2_<index>\n"
        );
        if en & 0x1 != 0 {
            text += "         %t4_<index> = OpCompositeExtract %float %t3_<index> 0\n";
            channels[0] = "%t4_<index>".to_owned();
        }
        if en & 0x2 != 0 {
            text += "         %t5_<index> = OpCompositeExtract %float %t3_<index> 1\n";
            channels[1] = "%t5_<index>".to_owned();
        }
    }

    if en & 0xc != 0 {
        if !operand_is_variable(inst.src[1]) {
            return Err(not_supported(
                FUNC,
                "enabled packed BA source is not a variable",
            ));
        }
        let src1 = operand_variable_to_str(inst.src[1]).value;
        text += &format!(
            "         %t6_<index> = OpLoad %float %{src1}\n\
             %t7_<index> = OpBitcast %uint %t6_<index>\n\
             %t8_<index> = OpExtInst %v2float %GLSL_std_450 UnpackHalf2x16 %t7_<index>\n"
        );
        if en & 0x4 != 0 {
            text += "         %t9_<index> = OpCompositeExtract %float %t8_<index> 0\n";
            channels[2] = "%t9_<index>".to_owned();
        }
        if en & 0x8 != 0 {
            text += "         %t10_<index> = OpCompositeExtract %float %t8_<index> 1\n";
            channels[3] = "%t10_<index>".to_owned();
        }
    }

    // TODO() check VSKIP
    // TODO() check EXEC

    text += &format!(
        "         %t11_<index> = OpCompositeConstruct %v4float {} {} {} {}\n\
                       OpStore {output} %t11_<index>\n",
        channels[0], channels[1], channels[2], channels[3],
    );
    *dst_source += &text.replace("<index>", &format!("{index}"));

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

    // SPI_SHADER_EX_FORMAT for the selected MRT must be an uncompressed 32-bit form:
    // 1 = 32_R, 2 = 32_GR, 3 = 32_AR, 9 = 32_ABGR. Kyty accepted only 9
    // (full en==0xf); the partial-en forms measured on ASTRO.BOT (en=0x3, a
    // 32_GR target) arrive with the matching partial output mode.
    let target = usize::from(inst.export_target);
    let mode = spirv
        .get_ps_input_info()
        .and_then(|info| info.target_output_mode.get(target))
        .copied()
        .unwrap_or(0);
    if !matches!(mode, 1 | 2 | 3 | 9) {
        return Err(not_supported(
            FUNC,
            format!(
                "target_output_mode[{target}] = {mode} is not an uncompressed 32-bit form (1/2/3/9)"
            ),
        ));
    }
    let output = mrt_output_variable(inst.export_target, FUNC)?;

    // The en mask selects which of the four VGPRs this export writes; the
    // disabled channels' vsrc fields are don't-care in the hardware encoding,
    // so only the enabled ones must be variables. Disabled channels get the
    // GCN default (0, 0, 0, 1) so the v4float store is fully defined.
    let en = inst.export_enable;
    let mut loads = String::new();
    let mut channels: [String; 4] = Default::default();
    for (i, channel) in channels.iter_mut().enumerate() {
        if en & (1 << i) != 0 {
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
            *channel = if i == 3 {
                "%float_1_000000".to_owned()
            } else {
                "%float_0_000000".to_owned()
            };
        }
    }

    // TODO() check VSKIP
    // TODO() check EXEC

    let text = format!(
        "{loads}         %t11_<index> = OpCompositeConstruct %v4float {} {} {} {}\n               OpStore {output} %t11_<index>\n",
        channels[0], channels[1], channels[2], channels[3],
    );

    *dst_source += &text.replace("<index>", &format!("{index}"));

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
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done";
    let inst = inst_at(code, index, FUNC)?;

    // Coverage probe (RAEEN_TRACE_DRAWS). A force-clear run can establish that
    // a draw produced no colour, but cannot distinguish vertex coverage from
    // cull/depth/stencil rejection. This line answers only the narrower
    // question: did translation encounter a POS0 export at all?
    if std::env::var_os("RAEEN_TRACE_DRAWS").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static POS0_SEEN: AtomicU32 = AtomicU32::new(0);
        let n = POS0_SEEN.fetch_add(1, Ordering::Relaxed);
        if n < 8 {
            tracing::warn!(
                n,
                srcs_are_variables = inst.src[..4].iter().all(|s| operand_is_variable(*s)),
                // The VGPRs the export READS. These need not equal the fetch
                // destinations because ordinary shader arithmetic commonly
                // writes the final position to different VGPRs. Correlate this
                // with the gated `Fetch* recompiled` lines below before
                // diagnosing an unwritten-register collapse.
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

    // DIAGNOSTIC (RAEEN_VS_PASSTHROUGH=1): bypass the VS arithmetic and export
    // input attribute 0 directly as the clip position. Every INPUT is verified
    // correct (the measured vertex buffer is a textbook NDC quad) and Vulkan
    // reports ZERO validation messages, yet no primitive covers a pixel — so
    // the suspect is the VALUES the translated VS computes. Pixels under this
    // flag => the generated VS body is at fault; still black => the fault is
    // below the shader. Match attr0's declared scalar/vector type so this
    // diagnostic remains valid when a title switches vertex layouts.
    if std::env::var_os("RAEEN_VS_PASSTHROUGH").is_some() {
        let info = spirv
            .get_vs_input_info()
            .ok_or_else(|| not_supported(FUNC, "VS passthrough has no vertex input info"))?;
        if info.resources_num <= 0 {
            return Err(not_supported(
                FUNC,
                "VS passthrough has no input attribute 0",
            ));
        }
        *dst_source += &vs_passthrough_source(
            index,
            info.resources_dst[0].registers_num,
            vertex_input_class(&info.resources[0]),
        )?;
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

/// Build the `RAEEN_VS_PASSTHROUGH` diagnostic body for attribute 0 of any
/// supported (component count, numeric class) pair, using the same shared
/// resolver the real declaration and fetch use — the three sites used to
/// enumerate three different subsets of the pair space.
fn vs_passthrough_source(
    index: u32,
    registers_num: i32,
    class: SampledClass,
) -> Result<String, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done";
    let Some(types) = vertex_input_types(registers_num, class) else {
        return Err(not_supported(
            FUNC,
            format!(
                "VS passthrough attr0: {registers_num} components of a {class:?} vertex format \
                 — only 1..=4 components are supported"
            ),
        ));
    };
    let n = registers_num as usize;

    // A full-width float attribute needs no rebuild: load it straight into the
    // clip position.
    if n == 4 && !types.bitcast {
        return Ok(passthrough_store(
            index,
            format!("         %pv_{index} = OpLoad %v4float %attr0\n"),
        ));
    }

    let mut load = format!("         %p0_{index} = OpLoad {} %attr0\n", types.load_type);
    let value = if types.bitcast {
        load += &format!(
            "             %pf_{index} = OpBitcast {} %p0_{index}\n",
            types.float_type
        );
        format!("%pf_{index}")
    } else {
        format!("%p0_{index}")
    };

    // Widen to vec4: present components first, then (0, 0, 0, 1) defaults —
    // the same fill the hardware fetch applies to absent channels.
    const COMPONENTS: [&str; 4] = ["px", "py", "pz", "pw"];
    let mut parts: Vec<String> = Vec::with_capacity(4);
    if n == 1 {
        parts.push(value);
    } else {
        for (c, component) in COMPONENTS.iter().enumerate().take(n) {
            let name = format!("%{component}_{index}");
            load += &format!("             {name} = OpCompositeExtract %float {value} {c}\n");
            parts.push(name);
        }
    }
    while parts.len() < 4 {
        parts.push(
            if parts.len() == 3 {
                "%float_1_000000"
            } else {
                "%float_0_000000"
            }
            .to_string(),
        );
    }
    load += &format!(
        "             %pv_{index} = OpCompositeConstruct %v4float {}\n",
        parts.join(" ")
    );
    Ok(passthrough_store(index, load))
}

fn passthrough_store(index: u32, load: String) -> String {
    format!(
        "{load}         %pa_{index} = OpAccessChain %_ptr_Output_v4float %outPerVertex %int_per_vertex_0\n\
                       OpStore %pa_{index} %pv_{index}\n"
    )
}

/// Beyond Kyty: auxiliary position exports pos1..pos3 (exp targets
/// 0x0d-0x0f), which upstream EXITs on. Per shadPS4 (`ir/position.h`
/// `ExportPosition`), each enabled channel maps to a clip distance, cull
/// distance, point size, or viewport/render-target index as configured by
/// PA_CL_VS_OUT_CNTL. Raeen does not plumb VS_OUT_CNTL into the recompiler
/// yet, so the export is accepted and dropped: nothing is written and
/// gl_Position (pos0) is untouched. Dropping clip/cull distances disables
/// user clip planes for the draw — visible at worst as missing clipping,
/// never as corruption. Measured: 632 ASTRO.BOT VS failures on pos1 with
/// en=0x4.
fn recompile_exp_pos_aux(
    index: u32,
    code: &ShaderCode,
    _dst_source: &mut String,
    _spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_PosAuxVsrc0Vsrc1Vsrc2Vsrc3";
    let _ = inst_at(code, index, FUNC)?;
    Ok(true)
}

/// `exp null off,off,off,off [done] [vm]` — accept and drop.
///
/// Beyond Kyty, which EXITs on EXP target 9. The null target exports nothing,
/// so there is no SPIR-V to emit: the instruction exists only to terminate the
/// shader's export sequence on hardware. Dropping it is not a degradation — it
/// is the complete and correct translation, because SPIR-V has no equivalent
/// concept and the function's `OpReturn` already ends the shader.
///
/// Refusing it, by contrast, failed the WHOLE shader recompile and dropped
/// every draw using it — the same whole-shader-refusal failure mode as the
/// `image_get_resinfo` non-2D gate and the partial-`en` param export.
/// Measured: 83 shader-translation failures on ASTRO.BOT (2026-07-27 sweep).
fn recompile_exp_null(
    index: u32,
    code: &ShaderCode,
    _dst_source: &mut String,
    _spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Exp_NullOffOffOffOffVmDone";
    let inst = inst_at(code, index, FUNC)?;
    // The parser only builds this format with no channels enabled; assert the
    // invariant rather than silently emitting nothing for a real export.
    if inst.export_enable != 0 || inst.src_num != 0 {
        return Err(not_supported(
            FUNC,
            "null export with enabled channels or sources",
        ));
    }
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

/// Shared body of every `Recompile_SBufferLoadDword{,x2,x4,x8,x16}_Sdst*SvSoffset*`
/// row (Kyty ShaderSpirv.cpp L3794/L3831/L3872/L3928/L3976 — identical upstream
/// apart from N and the `sbuffer_load_dword_N` callee).
///
/// Three lowerings are tried in order:
///
/// 1. **Per-PC capture.** `shader_capture_vsharp_buffer_loads` resolved the V#
///    base out of live-in user data, added the immediate + register soffset, and
///    snapshotted the dwords from guest memory. Materialize them straight into
///    the destination SGPRs — the same mechanism `sload_dword_extended` uses for
///    pointer-based loads. This is what makes a V#-based `s_buffer_load` work in
///    a shader that declares no storage-buffer slot at all (measured first
///    blocker of Grand Theft Auto V: `can't recompile: SBufferLoadDwordx8
///    [Sdst8SvSoffset] s[24:31], s[20:23], 0`, where
///    `storage_buffers.buffers_num == 0` made every width return `false`).
/// 2. **Bound storage buffer** (Kyty's path). The V# was bound as a descriptor,
///    so only the byte offset is needed. All widths now accept a runtime offset
///    the way `x4` always did, plus the combined
///    `soffset_register + immediate` form — the two terms simply sum, see
///    [`Format::SdstSvSoffsetOffset`] for the rule and its sources.
/// 3. **Named refusal.** Kyty's bare `return false` produced only
///    `can't recompile: <disassembly>`; this states which of the two lowerings
///    was missing, with the width and the descriptor register, so a measured log
///    line identifies the exact gap.
fn sbuffer_load_dwords(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    n: usize,
    callee: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;
    let Some(bind_info) = spirv.get_bind_info() else {
        return Err(not_supported(
            func,
            format!(
                "no bind info: {:?} [{:?}] x{n} dwords, V#=s{}, pc={:#x}",
                inst.type_, inst.format, inst.src[0].register_id, inst.pc,
            ),
        ));
    };

    // 1. The analysis-side V# capture (see `shader_capture_vsharp_buffer_loads`).
    if let Some(load) = bind_info.embedded_constant_loads.find(inst.pc) {
        let count = (load.dwords_num as usize).min(n);
        for i in 0..count {
            // Never overwrite a descriptor-array index a bound storage buffer
            // seeded into this register — see `descriptor_seeded_register`.
            if descriptor_seeded_register(spirv, bind_info, inst.dst, i as i32) {
                continue;
            }
            let dst_value = operand_variable_to_str_shift(inst.dst, i as i32);
            if dst_value.type_ != SpirvType::Uint {
                return Err(not_supported(func, "unexpected embedded-load dst type"));
            }
            *dst_source += &format!(
                "               OpStore %{reg} %{val}\n",
                reg = dst_value.value,
                val = spirv.get_constant_uint(load.values[i]),
            );
        }
        return Ok(true);
    }

    if bind_info.storage_buffers.buffers_num > 0 {
        let dst_value: Vec<_> = (0..n)
            .map(|i| operand_variable_to_str_shift(inst.dst, i as i32))
            .collect();
        let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);

        if dst_value[0].type_ != SpirvType::Uint || src0_value0.type_ != SpirvType::Uint {
            return Err(not_supported(func, "unexpected operand types"));
        }
        if operand_is_exec(inst.dst) {
            return Err(not_supported(func, "exec destination"));
        }

        let index_str = format!("{index}");

        // The byte offset is `soffset + immediate`. In the two-operand form one
        // of the two is absent and `src[1]` alone carries it (a constant when
        // the soffset field was NULL, an SGPR otherwise); in the combined form
        // `src[1]` is the register and `src[2]` the immediate, and both are
        // live. Emitting the add here keeps ONE offset expression for the
        // callee, which only ever sees a byte offset.
        let mut offset_src = String::new();
        if !operand_load_uint(
            spirv,
            inst.src[1],
            "t1_<index>",
            &index_str,
            &mut offset_src,
            -1,
        )? {
            return Err(not_supported(
                func,
                format!(
                    "offset operand {:?} has no uint load form (pc={:#x})",
                    inst.src[1].type_, inst.pc,
                ),
            ));
        }
        let mut offset_id = format!("t1_{index_str}");
        if crate::shader::types::smem_has_combined_offset(&inst) {
            // Combined form: add the immediate to the register value.
            let imm = crate::shader::types::smem_offset_operand(&inst);
            if !operand_is_constant(imm) {
                return Err(not_supported(
                    func,
                    format!(
                        "combined-form immediate is not a constant (pc={:#x})",
                        inst.pc
                    ),
                ));
            }
            let imm_id = spirv.get_constant_uint(imm.constant.u);
            // `operand_load_uint` leaves its last line UNTERMINATED (the caller's
            // template supplies the newline), so open a new line before adding.
            offset_src +=
                &format!("\n        %t1sum_{index_str} = OpIAdd %uint %t1_{index_str} %{imm_id}");
            offset_id = format!("t1sum_{index_str}");
        }

        const TEXT: &str = r#"
        <offset_src>
        %t100_<index> = OpLoad %uint %<src0_value0>
        %t101_<index> = OpBitcast %int %t100_<index>
               OpStore %temp_int_2 %t101_<index>
        %t102_<index> = OpBitcast %int %<offset_id>
               OpStore %temp_int_1 %t102_<index>
        %t110_<index> = OpFunctionCall %void %<callee> <regs> %temp_int_1 %temp_int_2
"#;
        let regs: Vec<String> = dst_value.iter().map(|v| format!("%{}", v.value)).collect();

        *dst_source += &TEXT
            .replace("<offset_src>", &offset_src)
            .replace("<callee>", callee)
            .replace("<regs>", &regs.join(" "))
            .replace("<src0_value0>", &src0_value0.value)
            .replace("<offset_id>", &offset_id)
            .replace("<index>", &index_str);

        return Ok(true);
    }

    Err(not_supported(
        func,
        format!(
            "no storage buffer bound for the V# and no resolved capture: {:?} [{:?}] x{n} \
             dwords, V#=s[{base}:{base_end}], soffset={}, imm={:#x}, pc={:#x}",
            inst.type_,
            inst.format,
            match crate::shader::types::smem_register_soffset(&inst) {
                Some(op) if op.type_ == ShaderOperandType::Sgpr => format!("s{}", op.register_id),
                Some(op) => format!("{:?}", op.type_),
                None => "none".to_string(),
            },
            crate::shader::types::smem_offset_operand(&inst).constant.u,
            inst.pc,
            base = inst.src[0].register_id,
            base_end = inst.src[0].register_id + 3,
        ),
    ))
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
    sbuffer_load_dwords(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_SBufferLoadDword_SdstSvSoffset",
        1,
        "sbuffer_load_dword",
    )
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
    sbuffer_load_dwords(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_SBufferLoadDwordx4_Sdst4SvSoffset",
        4,
        "sbuffer_load_dword_4",
    )
}

/// Kyty: `Recompile_SBranch_Label` (ShaderSpirv.cpp L4047).
fn recompile_sbranch_label(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SBranch_Label";
    let inst = inst_at(code, index, FUNC)?;

    if !operand_is_constant(inst.src[0]) {
        return Err(not_supported(FUNC, "src0 is not a constant"));
    }

    let label = ShaderLabel::from_instruction(&inst);

    // Dispatch-loop relooper: a branch is a store of the target case id.
    // Discard targets need no special casing — the discard block is an
    // ordinary case.
    if spirv.reloop_active() {
        let Some(id) = spirv.reloop_case_id(label.get_dst()) else {
            return Err(not_supported(FUNC, "branch target has no relooper block"));
        };
        *dst_source += &format!(
            "               OpStore %reloop_bb %int_{id}\n               \
             OpBranch %reloop_continue\n"
        );
        return Ok(true);
    }

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
    spirv: &Spirv<'_>,
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

    // Dispatch-loop relooper: select the next case id by the condition and
    // hand control to the dispatch loop. Direction (forward/backward) and
    // discard targets need no analysis — every block is a case.
    if spirv.reloop_active() {
        let label = ShaderLabel::from_instruction(&inst);
        let Some(dst_id) = spirv.reloop_case_id(label.get_dst()) else {
            return Err(not_supported(FUNC, "branch target has no relooper block"));
        };
        let Some(next_id) = spirv.reloop_case_id(next_inst.pc) else {
            return Err(not_supported(FUNC, "fallthrough has no relooper block"));
        };
        const TEXT: &str = r#"
        <param0>
        <param1>
        %reloop_t_<index> = OpSelect %int %cc_b_<index> %int_<dst> %int_<next>
               OpStore %reloop_bb %reloop_t_<index>
               OpBranch %reloop_continue
"#;
        *dst_source += &TEXT
            .replace("<param0>", param[0].unwrap_or(""))
            .replace("<param1>", param[1].unwrap_or(""))
            .replace("<index>", &format!("{index}"))
            .replace("<dst>", &dst_id.to_string())
            .replace("<next>", &next_id.to_string());
        return Ok(true);
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
/// upstream is a pure gate: skip the load when it belongs to the embedded fetch
/// shader, otherwise `return false` ("can't recompile").
///
/// Beyond Kyty: a single-dword scalar load is the same addressing family as
/// x2/x4/x8/x16 and now shares their materialization ([`sload_dword_wide`]), so
/// an `s_load_dword` off the EUD base or off a captured live-in/PC-relative
/// pointer lowers instead of failing the whole shader.
fn recompile_sload_dword(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    sload_dword_wide(
        index,
        code,
        dst_source,
        spirv,
        1,
        "Recompile_SLoadDword_SdstSbaseSoffset",
    )
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

    sload_dword_extended(index, code, &inst, dst_source, spirv, 2, FUNC)
}

/// Common extended (EUD) V#-from-push-constants path of
/// `Recompile_SLoadDwordx4/x8` (ShaderSpirv.cpp L4325-4369 / L4388-4432).
/// Whether dword `i` of a captured scalar load lands in an SGPR that a
/// **non-extended storage binding** already owns.
///
/// `Spirv::WriteLocalVariables` seeds those four registers from the
/// push-constant window, in which dword 0 has been REWRITTEN from the guest
/// base address to the compact descriptor-array index (`prepare_stage_binding`),
/// and every buffer body indexes `%buf` with the value of that register.
/// Materializing the captured RAW guest dwords over that seed replaces a small
/// array index with a guest address and indexes the descriptor array out of
/// bounds — the shape that produced a measured `VK_ERROR_DEVICE_LOST` on
/// ASTRO.BOT (see [`mimg_descriptor_guard`] for the image-side twin).
///
/// The seed is the value the shader must observe, so skipping the store is the
/// CORRECT lowering, not a workaround: the descriptor content still reaches the
/// draw, live, through the bound descriptor.
fn descriptor_seeded_register(
    spirv: &Spirv<'_>,
    bind: &ShaderBindResources,
    dst: crate::shader::types::ShaderOperand,
    i: i32,
) -> bool {
    if dst.type_ != ShaderOperandType::Sgpr {
        return false;
    }
    let shift_regs = if spirv.get_vs_input_info().is_some_and(|v| v.gs_prolog) {
        8
    } else {
        0
    };
    crate::shader::analysis::storage_binding_owns_register(bind, dst.register_id + i, shift_regs)
}

fn sload_dword_extended(
    index: u32,
    code: &ShaderCode,
    inst: &ShaderInstruction,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    n: i32,
    func: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let Some(bind_info) = spirv.get_bind_info() else {
        return Ok(false);
    };

    // Beyond Kyty: a PC-relative embedded-constant load — the shader reading
    // its own baked constant table. `shader_detect_embedded_constant_loads`
    // resolved the compile-time address and captured the dwords from guest
    // memory (the recompiler has no raw shader bytes); materialize them as
    // constants straight into the destination SGPRs. This is independent of the
    // EUD, so it runs before the extended-descriptor path and its base gate.
    if let Some(load) = bind_info.embedded_constant_loads.find(inst.pc) {
        let count = (load.dwords_num as i32).min(n);
        for i in 0..count {
            if descriptor_seeded_register(spirv, bind_info, inst.dst, i) {
                continue;
            }
            let dst_value = operand_variable_to_str_shift(inst.dst, i);
            if dst_value.type_ != SpirvType::Uint {
                return Err(not_supported(func, "unexpected embedded-load dst type"));
            }
            *dst_source += &format!(
                "               OpStore %{reg} %{val}\n",
                reg = dst_value.value,
                val = spirv.get_constant_uint(load.values[i as usize]),
            );
        }
        return Ok(true);
    }

    // Beyond Kyty: an unresolved **register soffset**. RDNA2 adds
    // `SGPR[soffset]` to the address, so neither the EUD dword index below nor
    // the raw-window index is knowable at translate time. Analysis
    // (`resolve_scalar_soffset_bytes`) folds the shapes it can prove into the
    // per-PC capture handled above; reaching here means it could not, so refuse
    // by name — with the width and format, so the log line identifies the exact
    // form still missing (measured ASTRO.BOT `rendering`, three compute
    // shaders).
    if let Some(soffset) = crate::shader::types::smem_register_soffset(inst) {
        return Err(not_supported(
            func,
            format!(
                "unresolved register soffset: {:?} [{:?}] x{n} dwords, base=s{}, soffset={}, \
                 imm={:#x}, pc={:#x}",
                inst.type_,
                inst.format,
                inst.src[0].register_id,
                match soffset.type_ {
                    ShaderOperandType::Sgpr => format!("s{}", soffset.register_id),
                    other => format!("{other:?}"),
                },
                crate::shader::types::smem_offset_operand(inst).constant.u,
                inst.pc,
            ),
        ));
    }

    if !bind_info.extended.used {
        return Ok(false);
    }

    // Kyty accepts only `LiteralConstant` here, but this codebase's next-gen
    // SMEM parser (`shader_parse_smem`) materializes a NULL soffset as the
    // sign-extended 21-bit immediate in an `IntegerInlineConstant` operand.
    // Both are compile-time constants. Measured on ASTRO.BOT compute (693
    // skips/run, the whole round-9 bulk): every refused s_load_dwordx4/x8 is
    // `s[N..], s[12:13], <inline imm>` — base = the EUD pointer pair, offset
    // a non-negative inline constant (0x0..0x50) — no register soffset at all.
    if !matches!(
        inst.src[1].type_,
        ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant
    ) {
        return Err(not_supported(func, "src1 is not a constant offset"));
    }
    if inst.src[0].register_id != bind_info.extended.start_register {
        let prior = code
            .get_instructions()
            .iter()
            .filter(|candidate| candidate.pc < inst.pc)
            .rev()
            .take(6)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|candidate| format!("{:#x}:{:?}", candidate.pc, candidate.type_))
            .collect::<Vec<_>>()
            .join(" -> ");
        let offset = format!("{:#x}", inst.src[1].constant.u);
        return Err(not_supported(
            func,
            format!(
                "src0 is not the EUD base register: pc={:#x} src0=s{} eud_base=s{} \
                 offset={} nearby_producers=[{}]",
                inst.pc,
                inst.src[0].register_id,
                bind_info.extended.start_register,
                offset,
                if prior.is_empty() { "<none>" } else { &prior },
            ),
        ));
    }
    if inst.src[1].constant.i() < 0 {
        // A sign-extended imm21 can be negative; nothing below the EUD base
        // is mapped, so refuse by name instead of indexing with a huge u32.
        return Err(not_supported(func, "negative s_load offset"));
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

    // SharpEmu port (see `shader_detect_eud_raw_window`): an EUD dword no
    // captured descriptor covers reads the dispatch-time guest-memory
    // snapshot bound at `%eud_raw` instead of refusing the shader
    // (SharpEmu treats every such load as a guest-memory read —
    // reference/sharpemu/src/SharpEmu.ShaderCompiler/
    // Gen5ShaderScalarEvaluator.cs:1939-1980; its SPIR-V side reads the
    // pooled window buffer with in-bounds checks —
    // Gen5SpirvTranslator.cs:2183-2236). The read clamps against the bound
    // window's `OpArrayLength` and yields 0 beyond it, so a short (partially
    // readable) snapshot degrades instead of faulting.
    const RAW_TEXT: &str = r#"
		         %eudraw_len_<index>_<i> = OpArrayLength %uint %eud_raw 0
		         %eudraw_lenm1_<index>_<i> = OpISub %uint %eudraw_len_<index>_<i> %uint_1
		         %eudraw_idx_<index>_<i> = OpExtInst %uint %GLSL_std_450 UMin %<idxc> %eudraw_lenm1_<index>_<i>
		         %eudraw_ptr_<index>_<i> = OpAccessChain %_ptr_StorageBuffer_uint %eud_raw %int_0 %eudraw_idx_<index>_<i>
		         %eudraw_val_<index>_<i> = OpLoad %uint %eudraw_ptr_<index>_<i>
		         %eudraw_inb_<index>_<i> = OpULessThan %bool %<idxc> %eudraw_len_<index>_<i>
		         %eudraw_res_<index>_<i> = OpSelect %uint %eudraw_inb_<index>_<i> %eudraw_val_<index>_<i> %uint_0
		               OpStore %<reg> %eudraw_res_<index>_<i>
				"#;

    for i in 0..n {
        let dst_value = operand_variable_to_str_shift(inst.dst, i);
        if i == 0 && dst_value.type_ != SpirvType::Uint {
            return Err(not_supported(func, "unexpected dst type"));
        }
        match spirv.get_mapped_index(offset + i) {
            Ok((buffer, field)) => {
                *dst_source += &TEXT
                    .replace("<reg>", &dst_value.value)
                    .replace("<buffer>", &format!("{buffer}"))
                    .replace("<field>", &format!("{field}"))
                    .replace("<index>", &format!("{index}"));
            }
            Err(refusal) => {
                if !bind_info.eud_raw.used {
                    // Detection did not declare a raw window for this
                    // shader; keep the named refusal rather than reading a
                    // buffer the dispatch path will not bind.
                    return Err(refusal);
                }
                if bind_info.eud_raw.unresolved_dynamic_offset {
                    // Detection could not size the window: another s_load off
                    // the same EUD base has a runtime register soffset, so
                    // `required_dwords` is only a lower bound. Reading it would
                    // silently clamp to 0 past a window we KNOW may be short —
                    // refuse by name instead (see `shader_detect_eud_raw_window`).
                    return Err(not_supported(
                        func,
                        format!(
                            "raw EUD window is a lower bound (an s_load off s{} has an \
                             unresolved register soffset); refusing dword {} of x{n} at pc={:#x}",
                            bind_info.extended.start_register,
                            offset + i,
                            inst.pc,
                        ),
                    ));
                }
                let idx = u32::try_from(offset + i)
                    .map_err(|_| not_supported(func, "negative raw EUD dword index"))?;
                *dst_source += &RAW_TEXT
                    .replace("<reg>", &dst_value.value)
                    .replace("<idxc>", &spirv.get_constant_uint(idx))
                    .replace("<index>", &format!("{index}"))
                    .replace("<i>", &format!("{i}"));
            }
        }
    }

    Ok(true)
}

/// Shared body of the wide `s_load_dword{x4,x8,x16}` rows: skip the loads that
/// only exist to bring the embedded vertex-fetch descriptors into SGPRs, then
/// hand `n` dwords to [`sload_dword_extended`].
fn sload_dword_wide(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    n: i32,
    func: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;
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
        return Err(not_supported(func, "extended path with gs_prolog shift"));
    }

    sload_dword_extended(index, code, &inst, dst_source, spirv, n, func)
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
    sload_dword_wide(
        index,
        code,
        dst_source,
        spirv,
        4,
        "Recompile_SLoadDwordx4_Sdst4SbaseSoffset",
    )
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
    sload_dword_wide(
        index,
        code,
        dst_source,
        spirv,
        8,
        "Recompile_SLoadDwordx8_Sdst8SbaseSoffset",
    )
}

/// Beyond Kyty (upstream `KYTY_NI`s SMEM/SMRD opcode 0x04):
/// `s_load_dwordx16` — the 16-dword row of the same family, reached through
/// the identical `sload_dword_extended` materialization as x4/x8. Measured as
/// the first shader blocker of Avatar: Frontiers of Pandora.
fn recompile_sload_dwordx16(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    sload_dword_wide(
        index,
        code,
        dst_source,
        spirv,
        16,
        "Recompile_SLoadDwordx16_Sdst16SbaseSoffset",
    )
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

/// Kyty: `Recompile_Skip` (ShaderSpirv.cpp L4707) — metadata/message/prefetch
/// instructions have no SPIR-V counterpart.
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

/// Scalar wait-count instructions stall one guest wave until its own
/// outstanding VMEM/LGKM/export operations reach the encoded counter. SPIR-V
/// already carries each invocation's producer/consumer data dependencies; a
/// cross-invocation memory barrier is not equivalent.
///
/// Kyty, KytyPS5, SharpEmu, and shadPS4 therefore all lower `s_waitcnt` to an
/// IR/SPIR-V no-op. The former device-scope AcquireRelease barrier was much
/// stronger than the guest instruction and multiplied into millions of
/// device-wide barriers in ASTRO.BOT compute, reproducibly hanging the driver.
fn recompile_s_waitcnt(
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

/// Beyond Kyty (`tbuffer_load_format_xy` is `KYTY_NI` upstream): the
/// two-channel typed fetch, structured exactly like the x1 and xyzw rows.
///
/// `%temp_int_2` is the byte offset (SOFFSET plus the folded 12-bit immediate —
/// see `shader_parse_mtbuf`), `%temp_int_3` the V#'s stride, `%temp_int_4` the
/// buffer index, `%temp_int_5` the legacy packed format the guarded helper
/// re-tests. **95** = dfmt 11, nfmt 7 (`32_32_FLOAT`) — the format
/// `shader_parse_mtbuf` requires for this opcode, and one of the two the
/// helper's guard admits.
fn recompile_tbuffer_load_format_xy_float2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_TBufferLoadFormatXy_Vdata2VaddrSvSoffsIdxenFloat2";
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
            let offset = spirv.get_constant(inst.src[2]);

            if dst_value0.type_ != SpirvType::Float
                || dst_value1.type_ != SpirvType::Float
                || src0_value.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src1_value1.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

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
               OpStore %temp_int_5 %int_95
        %t110_<index> = OpFunctionCall %void %tbuffer_load_format_xy %<p0> %<p1> %temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4 %temp_int_5
"#;
            *dst_source += &TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0>", &src0_value.value)
                .replace("<offset>", &offset)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src1_value1>", &src1_value1.value)
                .replace("<p0>", &dst_value0.value)
                .replace("<p1>", &dst_value1.value);

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

/// Beyond Kyty: `v_readfirstlane_b32`.
///
/// GCN/RDNA copies the source value from the first active lane of the wave
/// into a scalar destination. A plain per-invocation move is only correct for
/// uniform sources, so use the SPIR-V subgroup primitive instead. The measured
/// GTA V shader runs on a 64-lane compute subgroup and writes VCC_HI.
fn recompile_vreadfirstlane_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VReadfirstlaneB32_SVdstSVsrc0";
    let inst = inst_at(code, index, FUNC)?;

    if code.get_type() != ShaderType::Compute {
        return Err(not_supported(
            FUNC,
            "subgroup broadcast is currently guaranteed only for compute shaders",
        ));
    }
    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }
    let dst_value = operand_variable_to_str(inst.dst);
    if dst_value.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "dst is not uint"));
    }

    let index_str = format!("{index}");
    let mut load0 = String::new();
    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }

    const TEXT: &str = r#"
    <load0>
    %t_<index> = OpGroupNonUniformBroadcastFirst %uint %uint_3 %t0_<index>
    OpStore %<dst> %t_<index>
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

        if std::env::var_os("RAEEN_TRACE_DRAWS").is_some() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static FETCH_SEEN: AtomicU32 = AtomicU32::new(0);
            let n = FETCH_SEEN.fetch_add(1, Ordering::Relaxed);
            if n < 24 {
                tracing::warn!(
                    n,
                    attrib_id,
                    attrib_position = attrib_pos,
                    fetch_dst_start = inst.dst.register_id,
                    fetch_dst_size = n_dst,
                    semantic_hw_start = r.register_start,
                    semantic_hw_size = n_attr,
                    "TRACE_DRAWS: Fetch* recompiled (vertex attribute VGPR write)"
                );
            }
        }

        // GCN vertex fetch tolerates either direction of width mismatch:
        // channels beyond the attribute read back as the (0,0,0,1) default;
        // channels beyond the fetch are dropped into a scratch. Beyond Kyty
        // (upstream EXITs on any mismatch). Measured on Minecraft's menu VS:
        // attrib 2 as 2ch feeding a vec3 (fill z=0.0) and as 4ch (drop w).
        //
        // The attribute's declared width and numeric class come from the ONE
        // shared resolver, so this load can never disagree with the
        // `%attrN` OpVariable `Spirv::WriteGlobalVariables` declared for it.
        let class = vertex_input_class(&info.resources[attrib_pos]);
        let Some(types) = vertex_input_types(n_attr, class) else {
            return Err(not_supported(
                FUNC,
                format!(
                    "vertex attribute {attrib_pos} (attrib id {attrib_id}): {n_attr} components \
                     of unified format {} ({class:?}) — only 1..=4 components are supported",
                    info.resources[attrib_pos].format()
                ),
            ));
        };
        let temp_ty = types.temp;
        let load_ty = types.load_type.as_str();
        let helper = types.helper;
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
        // A raw integer attribute keeps its BITS: the guest VGPR is
        // float-backed, so the loaded uint/int vector is bitcast (not
        // numerically converted) into the same-width float type the fetch
        // helper takes. Componentwise `OpBitcast` on equal-width vectors is
        // exactly the reinterpretation the hardware fetch performs.
        let raw_bitcast = if types.bitcast {
            format!(
                "%t1_f_<index> = OpBitcast {} %t1_<index>\n",
                types.float_type
            )
        } else {
            String::new()
        };
        let stored_value = if types.bitcast {
            "%t1_f_<index>"
        } else {
            "%t1_<index>"
        };
        let mut text = format!(
            "
        %t1_<index> = OpLoad {load_ty} %<attr>
        {raw_bitcast}       OpStore {temp_ty} {stored_value}
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

            // Packed at translate time from the bound descriptor; the runtime
            // dword3 extraction handed `%tbuffer_store_format_x` a unified
            // number its guard can never equal, so the store wrote nothing.
            // See [`mubuf_descriptor_packed_format`].
            let packed_format = mubuf_descriptor_packed_format(
                &inst,
                bind_info,
                spirv,
                FUNC,
                &[&TBUF_STORE_FORMAT_X],
            )?
            .0;

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
               OpStore %temp_int_5 %int_<packed_format>
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
                .replace("<packed_format>", &format!("{packed_format}"))
                .replace("<p0>", &dst_value.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Kyty: `Recompile_BufferStoreFormatXy_Vdata2VaddrSvSoffsIdxen`
/// (ShaderSpirv.cpp L2137).
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

            // Packed at translate time from the bound descriptor — see
            // [`mubuf_descriptor_packed_format`]. `%tbuffer_store_format_xy`
            // tests 92 / 95, which no unified FORMAT field ever holds.
            let packed_format = mubuf_descriptor_packed_format(
                &inst,
                bind_info,
                spirv,
                FUNC,
                &[&TBUF_STORE_FORMAT_XY],
            )?
            .0;

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
               OpStore %temp_int_5 %int_<packed_format>
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
                .replace("<packed_format>", &format!("{packed_format}"))
                .replace("<p0>", &dst_value0.value)
                .replace("<p1>", &dst_value1.value);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Which storage-buffer helper the flexible MUBUF addressing body calls.
#[derive(Copy, Clone, PartialEq, Eq)]
enum MubufFlexOp {
    LoadDword,
    /// Single byte load, zero-extended — the `%buffer_load_ubyte` helper
    /// extracts the byte from the containing dword (the byte address is not
    /// pre-divided by 4).
    LoadUbyte,
    StoreDword,
    LoadFormatX,
    StoreFormatX,
    StoreFormatXyzw,
}

/// Shared body of the beyond-Kyty flexible-addressing MUBUF recompilers.
///
/// Kyty's MUBUF bodies (`Recompile_BufferLoadDword_*` L1877,
/// `Recompile_BufferStoreDword_*` L1999, `Recompile_Buffer{Load,Store}FormatX_*`
/// L1937/L2068) hardcode `idxen == 1, offen == 0`. This body derives the
/// addressing mode from the parsed format instead — the model
/// [`recompile_buffer_load_dwordx4`] established:
/// `temp_int_1` = vindex (or 0 without idxen), `temp_int_2` = instruction
/// offset (+ voffset when offen), `temp_int_3` = stride from V# dword1,
/// `temp_int_4` = buffer index, `temp_int_5` = dfmt_nfmt from V# dword3
/// (format ops only) — then calls the same Kyty helper functions.
/// Stores are wrapped in the exec_lo guard the Kyty store bodies use.
fn mubuf_flexible(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    op: MubufFlexOp,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    let Some(bind_info) = spirv.get_bind_info() else {
        return Ok(false);
    };
    if bind_info.storage_buffers.buffers_num == 0 {
        // No storage buffer is bound anywhere in this shader, so this MUBUF
        // op goes through a V# whose captured descriptor was NULL (an
        // all-zero user-SGPR quad — analysis keeps those as direct
        // registers). RDNA out-of-bounds rules for a null V# (num_records
        // = 0): stores are DROPPED, loads return 0. Emit exactly that
        // (measured: 811 ASTRO.BOT CS dispatches store v[0:3] through
        // s[0:3] = a zero quad).
        let is_store = matches!(
            op,
            MubufFlexOp::StoreDword | MubufFlexOp::StoreFormatX | MubufFlexOp::StoreFormatXyzw
        );
        if !is_store {
            let n: i32 = if op == MubufFlexOp::StoreFormatXyzw {
                4
            } else {
                1
            };
            let i = format!("{index}");
            let mut text = String::new();
            for c in 0..n {
                let value = operand_variable_to_str_shift(inst.dst, c);
                if value.type_ != SpirvType::Float {
                    return Err(not_supported(func, "unexpected vdata type (null V#)"));
                }
                text += &format!(
                    "\n        %nulvz_{i}_{c} = OpBitcast %float %uint_0\n                   OpStore %{} %nulvz_{i}_{c}\n",
                    value.value
                );
            }
            *dst_source += &text;
        }
        return Ok(true);
    }
    if !operand_is_constant(inst.src[2]) {
        return Err(not_supported(func, "src2 is not a constant"));
    }

    let idxen = matches!(
        inst.format,
        Format::Vdata1VaddrSvSoffsIdxen
            | Format::Vdata1Vaddr2SvSoffsOffenIdxen
            | Format::Vdata4VaddrSvSoffsIdxen
            | Format::Vdata4Vaddr2SvSoffsOffenIdxen
    );
    let offen = matches!(
        inst.format,
        Format::Vdata1VaddrSvSoffsOffen
            | Format::Vdata1Vaddr2SvSoffsOffenIdxen
            | Format::Vdata4VaddrSvSoffsOffen
            | Format::Vdata4Vaddr2SvSoffsOffenIdxen
    );

    let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
    let src1_value1 = operand_variable_to_str_shift(inst.src[1], 1);
    let offset = spirv.get_constant(inst.src[2]);
    if src1_value0.type_ != SpirvType::Uint || src1_value1.type_ != SpirvType::Uint {
        return Err(not_supported(func, "unexpected V# operand types"));
    }

    let is_store = matches!(
        op,
        MubufFlexOp::StoreDword | MubufFlexOp::StoreFormatX | MubufFlexOp::StoreFormatXyzw
    );
    // The typed helper this op calls, for the ops that take a format argument.
    // `None` for the raw dword/byte ops, which have no format guard at all.
    let format_helper = match op {
        MubufFlexOp::LoadFormatX => Some(&TBUF_LOAD_FORMAT_X),
        MubufFlexOp::StoreFormatX => Some(&TBUF_STORE_FORMAT_X),
        MubufFlexOp::StoreFormatXyzw => Some(&TBUF_STORE_FORMAT_XYZW),
        MubufFlexOp::LoadDword | MubufFlexOp::LoadUbyte | MubufFlexOp::StoreDword => None,
    };
    let vdata_n: i32 = if op == MubufFlexOp::StoreFormatXyzw {
        4
    } else {
        1
    };

    let mut vdata = Vec::with_capacity(vdata_n as usize);
    for i in 0..vdata_n {
        let value = operand_variable_to_str_shift(inst.dst, i);
        if value.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected vdata type"));
        }
        vdata.push(value.value);
    }

    let i = format!("{index}");
    let mut text = String::new();

    // TODO() check VSKIP

    if is_store {
        text += &format!(
            "
        %exec_lo_u_{i} = OpLoad %uint %exec_lo
        %exec_lo_b_{i} = OpINotEqual %bool %exec_lo_u_{i} %uint_0
               OpSelectionMerge %mbf_end_{i} None
               OpBranchConditional %exec_lo_b_{i} %mbf_body_{i} %mbf_end_{i}
        %mbf_body_{i} = OpLabel
"
        );
    }

    // temp_int_1 = vindex (idxen) or 0.
    if idxen {
        let vindex = operand_variable_to_str_shift(inst.src[0], 0);
        if vindex.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected vindex register type"));
        }
        text += &format!(
            "        %mbf_i0_{i} = OpLoad %float %{}
        %mbf_i1_{i} = OpBitcast %int %mbf_i0_{i}
               OpStore %temp_int_1 %mbf_i1_{i}
",
            vindex.value
        );
    } else {
        text += "               OpStore %temp_int_1 %int_0\n";
    }

    // temp_int_3 = stride (V# dword1 bits 16..29), temp_int_4 = buffer index,
    // temp_int_2 = instruction offset.
    text += &format!(
        "        %mbf_s0_{i} = OpLoad %uint %{src1_value1}
        %mbf_s1_{i} = OpShiftRightLogical %uint %mbf_s0_{i} %int_16
        %mbf_s2_{i} = OpBitwiseAnd %uint %mbf_s1_{i} %uint_0x00003fff
        %mbf_s3_{i} = OpBitcast %int %mbf_s2_{i}
               OpStore %temp_int_3 %mbf_s3_{i}
        %mbf_b0_{i} = OpLoad %uint %{src1_value0}
        %mbf_b1_{i} = OpBitcast %int %mbf_b0_{i}
               OpStore %temp_int_4 %mbf_b1_{i}
               OpStore %temp_int_2 %{offset}
",
        src1_value1 = src1_value1.value,
        src1_value0 = src1_value0.value,
    );

    // offen: the vaddr register after the (optional) vindex is a per-thread
    // byte offset folded into temp_int_2.
    if offen {
        let voffset = operand_variable_to_str_shift(inst.src[0], i32::from(idxen));
        if voffset.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected voffset register type"));
        }
        text += &format!(
            "        %mbf_o0_{i} = OpLoad %float %{}
        %mbf_o1_{i} = OpBitcast %int %mbf_o0_{i}
        %mbf_o2_{i} = OpLoad %int %temp_int_2
        %mbf_o3_{i} = OpIAdd %int %mbf_o2_{i} %mbf_o1_{i}
               OpStore %temp_int_2 %mbf_o3_{i}
",
            voffset.value
        );
    }

    // temp_int_5 = the packed `dfmt * 8 + nfmt` the typed helper's guard tests,
    // converted from the BOUND descriptor's unified FORMAT at translate time.
    // This body used to extract V# dword3 bits 12..18 in the shader and pass
    // that unified number in — a comparison that can never succeed, so every
    // flexible-addressing typed access was a silent no-op. See
    // [`mubuf_descriptor_packed_format`].
    if let Some(helper) = format_helper {
        // Type-check dword3 anyway: this row still reaches the same V# quad,
        // and a non-uint there means the operand model is wrong, not the
        // format.
        let src1_value3 = operand_variable_to_str_shift(inst.src[1], 3);
        if src1_value3.type_ != SpirvType::Uint {
            return Err(not_supported(func, "unexpected V# dword3 type"));
        }
        let (packed_format, _) =
            mubuf_descriptor_packed_format(&inst, bind_info, spirv, func, &[helper])?;
        text += &format!("               OpStore %temp_int_5 %int_{packed_format}\n");
    }

    let helper = match op {
        MubufFlexOp::LoadDword => "%buffer_load_float1",
        MubufFlexOp::LoadUbyte => "%buffer_load_ubyte",
        MubufFlexOp::StoreDword => "%buffer_store_float1",
        MubufFlexOp::LoadFormatX => "%tbuffer_load_format_x",
        MubufFlexOp::StoreFormatX => "%tbuffer_store_format_x",
        MubufFlexOp::StoreFormatXyzw => "%tbuffer_store_format_xyzw",
    };
    let mut args = String::new();
    for v in &vdata {
        args += &format!("%{v} ");
    }
    args += "%temp_int_1 %temp_int_2 %temp_int_3 %temp_int_4";
    if format_helper.is_some() {
        args += " %temp_int_5";
    }
    text += &format!("        %mbf_c_{i} = OpFunctionCall %void {helper} {args}\n");

    if is_store {
        text += &format!(
            "               OpBranch %mbf_end_{i}
        %mbf_end_{i} = OpLabel
"
        );
    }

    *dst_source += &text;
    Ok(true)
}

/// Beyond Kyty: `buffer_load_dword` with idxen==0 and/or offen==1.
fn recompile_buffer_load_dword_flexible(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    mubuf_flexible(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_BufferLoadDword_FlexibleAddr",
        MubufFlexOp::LoadDword,
    )
}

/// Beyond Kyty (`buffer_load_ubyte` is `KYTY_NI` upstream): single
/// zero-extended byte load through the `%buffer_load_ubyte` helper, all four
/// addressing modes. Measured on ASTRO.BOT scene compute (raw 0xe02020c0,
/// 58 dispatches/run).
fn recompile_buffer_load_ubyte(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    mubuf_flexible(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_BufferLoadUbyte_FlexibleAddr",
        MubufFlexOp::LoadUbyte,
    )
}

/// Beyond Kyty: `buffer_store_dword` with idxen==0 and/or offen==1.
fn recompile_buffer_store_dword_flexible(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    mubuf_flexible(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_BufferStoreDword_FlexibleAddr",
        MubufFlexOp::StoreDword,
    )
}

/// Beyond Kyty: `buffer_load_format_x` with idxen==0 and/or offen==1.
fn recompile_buffer_load_format_x_flexible(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    mubuf_flexible(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_BufferLoadFormatX_FlexibleAddr",
        MubufFlexOp::LoadFormatX,
    )
}

/// Beyond Kyty: `buffer_store_format_x` with idxen==0 and/or offen==1.
fn recompile_buffer_store_format_x_flexible(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    mubuf_flexible(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_BufferStoreFormatX_FlexibleAddr",
        MubufFlexOp::StoreFormatX,
    )
}

/// Beyond Kyty (SharpEmu PR #587): lower a FLAT-class (FLAT / GLOBAL) direct
/// guest-memory load/store to a `%global_mem` window access.
///
/// The window is a `uint[]` SSBO whose first two dwords are the window's guest
/// base address (host-filled) and whose remaining dwords are the window bytes.
/// A 64-bit guest address becomes a dword index by
/// `((addr_lo - base_lo) >> 2)` — a 32-bit subtraction, exactly as SharpEmu's
/// `ISub` (`Gen5SpirvTranslator.cs`): the wrap absorbs any carry and the window
/// is < 4 GiB, so the low dword alone is exact. The address itself is
/// reconstructed per `ShaderInstruction::uses_flat_address`: a FLAT op reads
/// the whole address from the VGPR pair (`src[0]`); a GLOBAL op adds the SGPR
/// base pair (`src[1]`) to the 32-bit VGPR offset. `src[2]` is the instruction
/// immediate offset. Loads past the bound length yield 0; stores past it drop
/// (RDNA out-of-bounds behaviour).
///
/// Pre-existing `clippy::too_many_arguments` (8/7): every parameter is a
/// distinct piece of the FLAT/GLOBAL lowering and grouping them into a struct
/// would only move the same eight values behind one more indirection. Allowed
/// rather than restructured — this function's behaviour is not in question.
#[allow(clippy::too_many_arguments)]
fn flat_mem(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    dwords: i32,
    is_store: bool,
    load_ubyte: bool,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    let Some(bind) = spirv.get_bind_info() else {
        return Ok(false);
    };
    if !bind.global_mem.used {
        // Detection (`shader_detect_flat_global_window`) must have declared the
        // window whenever a FLAT-class op is present; refuse by name rather
        // than reference an undeclared `%global_mem`.
        return Err(not_supported(func, "global_mem window not declared"));
    }

    let addr = operand_variable_to_str_shift(inst.src[0], 0);
    if addr.type_ != SpirvType::Float {
        return Err(not_supported(func, "unexpected FLAT address VGPR type"));
    }

    let off_c = spirv.get_constant_uint(inst.src[2].constant.u);
    let two_c = spirv.get_constant_uint(2);
    let i = format!("{index}");
    let mut text = String::new();

    // Window base low dword (%global_mem[0]).
    text += &format!(
        "        %flat_bp_{i} = OpAccessChain %_ptr_StorageBuffer_uint %global_mem %int_0 %uint_0
        %flat_base_{i} = OpLoad %uint %flat_bp_{i}
        %flat_a0_{i} = OpLoad %float %{av}
        %flat_a1_{i} = OpBitcast %uint %flat_a0_{i}
",
        av = addr.value
    );

    // Full byte address: FLAT reads it whole from the VGPR pair; GLOBAL adds
    // the SGPR base pair to the 32-bit VGPR offset.
    if inst.uses_flat_address {
        text += &format!("        %flat_ab_{i} = OpIAdd %uint %flat_a1_{i} %uint_0\n");
    } else {
        let base = operand_variable_to_str_shift(inst.src[1], 0);
        if base.type_ != SpirvType::Uint {
            return Err(not_supported(func, "unexpected GLOBAL SGPR base type"));
        }
        text += &format!(
            "        %flat_sb_{i} = OpLoad %uint %{bv}
        %flat_ab_{i} = OpIAdd %uint %flat_sb_{i} %flat_a1_{i}
",
            bv = base.value
        );
    }

    // byte offset into the window, then dword index (+2 skips the base header).
    text += &format!(
        "        %flat_ao_{i} = OpIAdd %uint %flat_ab_{i} %{off_c}
        %flat_bo_{i} = OpISub %uint %flat_ao_{i} %flat_base_{i}
        %flat_di_{i} = OpShiftRightLogical %uint %flat_bo_{i} %{two_c}
        %flat_bi_{i} = OpIAdd %uint %flat_di_{i} %{two_c}
        %flat_len_{i} = OpArrayLength %uint %global_mem 0
        %flat_lenm1_{i} = OpISub %uint %flat_len_{i} %uint_1
"
    );

    for k in 0..dwords {
        let kc = spirv.get_constant_uint(u32::try_from(k).unwrap_or(0));
        let dv = operand_variable_to_str_shift(inst.dst, k);
        if dv.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected FLAT data VGPR type"));
        }
        text += &format!(
            "        %flat_idx_{i}_{k} = OpIAdd %uint %flat_bi_{i} %{kc}
        %flat_cidx_{i}_{k} = OpExtInst %uint %GLSL_std_450 UMin %flat_idx_{i}_{k} %flat_lenm1_{i}
        %flat_ptr_{i}_{k} = OpAccessChain %_ptr_StorageBuffer_uint %global_mem %int_0 %flat_cidx_{i}_{k}
        %flat_inb_{i}_{k} = OpULessThan %bool %flat_idx_{i}_{k} %flat_len_{i}
"
        );
        if is_store {
            // Out-of-bounds stores drop; in-bounds stores write the dword. (A
            // full exec-mask guard is a later refinement — inactive lanes are
            // not masked here.)
            text += &format!(
                "        %flat_dv0_{i}_{k} = OpLoad %float %{dv}
        %flat_dv1_{i}_{k} = OpBitcast %uint %flat_dv0_{i}_{k}
               OpSelectionMerge %flat_sm_{i}_{k} None
               OpBranchConditional %flat_inb_{i}_{k} %flat_st_{i}_{k} %flat_sm_{i}_{k}
        %flat_st_{i}_{k} = OpLabel
               OpStore %flat_ptr_{i}_{k} %flat_dv1_{i}_{k}
               OpBranch %flat_sm_{i}_{k}
        %flat_sm_{i}_{k} = OpLabel
",
                dv = dv.value
            );
        } else {
            text += &format!(
                "        %flat_lv_{i}_{k} = OpLoad %uint %flat_ptr_{i}_{k}
        %flat_raw_{i}_{k} = OpSelect %uint %flat_inb_{i}_{k} %flat_lv_{i}_{k} %uint_0
"
            );
            if load_ubyte {
                // The global window is dword-backed. Select the addressed byte
                // from that dword and zero-extend it exactly as
                // flat_load_ubyte requires.
                text += &format!(
                    "        %flat_lane_{i}_{k} = OpBitwiseAnd %uint %flat_bo_{i} %uint_3
        %flat_shift_{i}_{k} = OpShiftLeftLogical %uint %flat_lane_{i}_{k} %uint_3
        %flat_shr_{i}_{k} = OpShiftRightLogical %uint %flat_raw_{i}_{k} %flat_shift_{i}_{k}
        %flat_res_{i}_{k} = OpBitwiseAnd %uint %flat_shr_{i}_{k} %uint_255
"
                );
            } else {
                text += &format!(
                    "        %flat_res_{i}_{k} = OpIAdd %uint %flat_raw_{i}_{k} %uint_0
"
                );
            }
            text += &format!(
                "        %flat_fv_{i}_{k} = OpBitcast %float %flat_res_{i}_{k}
               OpStore %{dv} %flat_fv_{i}_{k}
",
                dv = dv.value
            );
        }
    }

    *dst_source += &text;
    Ok(true)
}

macro_rules! flat_mem_recompiler {
    ($name:ident, $func:literal, $dwords:literal, $is_store:literal) => {
        fn $name(
            index: u32,
            code: &ShaderCode,
            dst_source: &mut String,
            spirv: &Spirv<'_>,
            _param: &Params,
            _scc_check: SccCheck,
        ) -> Result<bool, ShaderRecompileError> {
            flat_mem(
                index, code, dst_source, spirv, $func, $dwords, $is_store, false,
            )
        }
    };
}

fn recompile_flat_load_ubyte(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    flat_mem(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_FlatLoadUbyte",
        1,
        false,
        true,
    )
}

flat_mem_recompiler!(
    recompile_flat_load_dword,
    "Recompile_FlatLoadDword",
    1,
    false
);
flat_mem_recompiler!(
    recompile_flat_load_dwordx2,
    "Recompile_FlatLoadDwordX2",
    2,
    false
);
flat_mem_recompiler!(
    recompile_flat_load_dwordx3,
    "Recompile_FlatLoadDwordX3",
    3,
    false
);
flat_mem_recompiler!(
    recompile_flat_load_dwordx4,
    "Recompile_FlatLoadDwordX4",
    4,
    false
);
flat_mem_recompiler!(
    recompile_flat_store_dword,
    "Recompile_FlatStoreDword",
    1,
    true
);
flat_mem_recompiler!(
    recompile_flat_store_dwordx2,
    "Recompile_FlatStoreDwordX2",
    2,
    true
);
flat_mem_recompiler!(
    recompile_flat_store_dwordx4,
    "Recompile_FlatStoreDwordX4",
    4,
    true
);

/// Beyond Kyty (`buffer_store_format_xyzw` is `KYTY_NI` upstream,
/// ShaderParse.cpp L2630): 4-channel formatted store through the
/// `tbuffer_store_format_xyzw` helper, all four addressing modes. The single
/// most frequent ASTRO.BOT shader failure (925 dispatches / 30s).
fn recompile_buffer_store_format_xyzw(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    mubuf_flexible(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_BufferStoreFormatXyzw_Vdata4VaddrSvSoffs",
        MubufFlexOp::StoreFormatXyzw,
    )
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
<offset_calc>        %t196_<index> = OpAccessChain %_ptr_StorageBuffer_uint %gds %int_0 <counter_index>
        %t198_<index> = <atomic_op> %uint %t196_<index> %uint_1 %uint_0 %uint_1
        %t199_<index> = OpBitcast %float %t198_<index>
               OpStore %<dst> %t199_<index>
               OpMemoryBarrier %uint_1 %uint_72
"#;
            // Beyond Kyty: the 16-bit instruction byte offset selects a
            // counter past the M0 base (shadPS4 `DS_APPEND`: `gds_offset =
            // M0 + inst_offset`, indexed at `>> 2` —
            // resource_tracking_pass.cpp L699-L708). Kyty's zero-offset
            // indexing (`m0 >> 16` used directly) is preserved untouched
            // when no offset rides on the instruction; a nonzero offset
            // contributes its DWORD count on top of that base.
            let (offset_calc, counter_index) =
                if inst.src_num >= 1 && operand_is_constant(inst.src[0]) {
                    let off = spirv.get_constant(inst.src[0]);
                    (
                        format!(
                            "        %t193_<index> = OpShiftRightLogical %uint %{off} %uint_2
        %t195_<index> = OpIAdd %uint %t194_<index> %t193_<index>
"
                        ),
                        "%t195_<index>",
                    )
                } else {
                    (String::new(), "%t194_<index>")
                };
            *dst_source += &TEXT
                .replace("<offset_calc>", &offset_calc)
                .replace("<counter_index>", counter_index)
                .replace("<atomic_op>", atomic_op)
                .replace("<dst>", &dst_value.value)
                .replace("<index>", &index_str);

            return Ok(true);
        }
    }

    Ok(false)
}

/// Beyond Kyty (`s_barrier` is `KYTY_NI` upstream): workgroup control
/// barrier. Scope Workgroup (2) for both execution and memory; semantics
/// AcquireRelease | WorkgroupMemory (0x108 = 264) so LDS writes are visible
/// across the group.
fn recompile_s_barrier(
    _index: u32,
    _code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    let semantics = spirv.get_constant_uint(0x108);
    *dst_source += &format!("\n               OpControlBarrier %uint_2 %uint_2 %{semantics}\n");
    Ok(true)
}

/// Beyond Kyty (`ds_write_b32` is `KYTY_NI` upstream; Kyty implements only
/// the GDS append/consume pair): LDS dword write, lowered to an `OpStore`
/// into the `%lds` Workgroup array at `(addr + offset) >> 2`. The index is
/// clamped to the array bound (hardware wraps within the allocation; a clamp
/// keeps the SPIR-V defined without inventing an aliasing model). The store
/// is wrapped in the same `exec_lo` guard the MUBUF stores use.
fn recompile_ds_write_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_DsWriteB32_Vsrc0Vsrc1Vsrc2";
    let inst = inst_at(code, index, FUNC)?;

    if !operand_is_variable(inst.src[0]) || !operand_is_variable(inst.src[1]) {
        return Err(not_supported(FUNC, "addr/data are not variables"));
    }
    if !operand_is_constant(inst.src[2]) {
        return Err(not_supported(FUNC, "offset is not a constant"));
    }

    let addr = operand_variable_to_str(inst.src[0]);
    let data = operand_variable_to_str(inst.src[1]);
    if addr.type_ != SpirvType::Float || data.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "unexpected operand types"));
    }
    let offset = spirv.get_constant(inst.src[2]);
    let clamp = spirv.get_constant_uint(spirv.lds_size_dw() - 1);

    let i = format!("{index}");
    let text = format!(
        "
        %ldsw_e0_{i} = OpLoad %uint %exec_lo
        %ldsw_e1_{i} = OpINotEqual %bool %ldsw_e0_{i} %uint_0
               OpSelectionMerge %ldsw_end_{i} None
               OpBranchConditional %ldsw_e1_{i} %ldsw_body_{i} %ldsw_end_{i}
        %ldsw_body_{i} = OpLabel
        %ldsw_a_{i} = OpLoad %float %{addr}
        %ldsw_au_{i} = OpBitcast %uint %ldsw_a_{i}
        %ldsw_ao_{i} = OpIAdd %uint %ldsw_au_{i} %{offset}
        %ldsw_ai_{i} = OpShiftRightLogical %uint %ldsw_ao_{i} %uint_2
        %ldsw_ac_{i} = OpExtInst %uint %GLSL_std_450 UMin %ldsw_ai_{i} %{clamp}
        %ldsw_p_{i} = OpAccessChain %_ptr_Workgroup_uint %lds %ldsw_ac_{i}
        %ldsw_d_{i} = OpLoad %float %{data}
        %ldsw_du_{i} = OpBitcast %uint %ldsw_d_{i}
               OpStore %ldsw_p_{i} %ldsw_du_{i}
               OpBranch %ldsw_end_{i}
        %ldsw_end_{i} = OpLabel
",
        addr = addr.value,
        data = data.value,
    );

    *dst_source += &text;
    Ok(true)
}

/// Beyond Kyty (`ds_add_u32` is `KYTY_NI` upstream): LDS atomic dword add
/// without return — `OpAtomicIAdd` on `%lds[(addr + offset) >> 2]` at
/// Workgroup scope (2), Relaxed semantics (0). Address computation, bound
/// clamp and the `exec_lo` guard mirror [`recompile_ds_write_b32`]; the
/// atomic's old-value result is discarded (the non-`_rtn` form writes no
/// VGPR).
fn recompile_ds_add_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_DsAddU32_Vsrc0Vsrc1Vsrc2";
    let inst = inst_at(code, index, FUNC)?;

    if !operand_is_variable(inst.src[0]) || !operand_is_variable(inst.src[1]) {
        return Err(not_supported(FUNC, "addr/data are not variables"));
    }
    if !operand_is_constant(inst.src[2]) {
        return Err(not_supported(FUNC, "offset is not a constant"));
    }

    let addr = operand_variable_to_str(inst.src[0]);
    let data = operand_variable_to_str(inst.src[1]);
    if addr.type_ != SpirvType::Float || data.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "unexpected operand types"));
    }
    let offset = spirv.get_constant(inst.src[2]);
    let clamp = spirv.get_constant_uint(spirv.lds_size_dw() - 1);

    let i = format!("{index}");
    let text = format!(
        "
        %ldsaa_e0_{i} = OpLoad %uint %exec_lo
        %ldsaa_e1_{i} = OpINotEqual %bool %ldsaa_e0_{i} %uint_0
               OpSelectionMerge %ldsaa_end_{i} None
               OpBranchConditional %ldsaa_e1_{i} %ldsaa_body_{i} %ldsaa_end_{i}
        %ldsaa_body_{i} = OpLabel
        %ldsaa_a_{i} = OpLoad %float %{addr}
        %ldsaa_au_{i} = OpBitcast %uint %ldsaa_a_{i}
        %ldsaa_ao_{i} = OpIAdd %uint %ldsaa_au_{i} %{offset}
        %ldsaa_ai_{i} = OpShiftRightLogical %uint %ldsaa_ao_{i} %uint_2
        %ldsaa_ac_{i} = OpExtInst %uint %GLSL_std_450 UMin %ldsaa_ai_{i} %{clamp}
        %ldsaa_p_{i} = OpAccessChain %_ptr_Workgroup_uint %lds %ldsaa_ac_{i}
        %ldsaa_d_{i} = OpLoad %float %{data}
        %ldsaa_du_{i} = OpBitcast %uint %ldsaa_d_{i}
        %ldsaa_old_{i} = OpAtomicIAdd %uint %ldsaa_p_{i} %uint_2 %uint_0 %ldsaa_du_{i}
               OpBranch %ldsaa_end_{i}
        %ldsaa_end_{i} = OpLabel
",
        addr = addr.value,
        data = data.value,
    );

    *dst_source += &text;
    Ok(true)
}

/// Return-value twin of [`recompile_ds_add_u32`]:
/// `vdst = atomic_add(lds[(addr + offset) >> 2], data)`.
fn recompile_ds_add_rtn_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_DsAddRtnU32_VdstVsrc0Vsrc1Vsrc2";
    let inst = inst_at(code, index, FUNC)?;

    if !operand_is_variable(inst.dst)
        || !operand_is_variable(inst.src[0])
        || !operand_is_variable(inst.src[1])
    {
        return Err(not_supported(FUNC, "dst/addr/data are not variables"));
    }
    if !operand_is_constant(inst.src[2]) {
        return Err(not_supported(FUNC, "offset is not a constant"));
    }

    let dst = operand_variable_to_str(inst.dst);
    let addr = operand_variable_to_str(inst.src[0]);
    let data = operand_variable_to_str(inst.src[1]);
    if dst.type_ != SpirvType::Float
        || addr.type_ != SpirvType::Float
        || data.type_ != SpirvType::Float
    {
        return Err(not_supported(FUNC, "unexpected operand types"));
    }
    let offset = spirv.get_constant(inst.src[2]);
    let clamp = spirv.get_constant_uint(spirv.lds_size_dw() - 1);

    let i = format!("{index}");
    let text = format!(
        "
        %ldsaar_e0_{i} = OpLoad %uint %exec_lo
        %ldsaar_e1_{i} = OpINotEqual %bool %ldsaar_e0_{i} %uint_0
               OpSelectionMerge %ldsaar_end_{i} None
               OpBranchConditional %ldsaar_e1_{i} %ldsaar_body_{i} %ldsaar_end_{i}
        %ldsaar_body_{i} = OpLabel
        %ldsaar_a_{i} = OpLoad %float %{addr}
        %ldsaar_au_{i} = OpBitcast %uint %ldsaar_a_{i}
        %ldsaar_ao_{i} = OpIAdd %uint %ldsaar_au_{i} %{offset}
        %ldsaar_ai_{i} = OpShiftRightLogical %uint %ldsaar_ao_{i} %uint_2
        %ldsaar_ac_{i} = OpExtInst %uint %GLSL_std_450 UMin %ldsaar_ai_{i} %{clamp}
        %ldsaar_p_{i} = OpAccessChain %_ptr_Workgroup_uint %lds %ldsaar_ac_{i}
        %ldsaar_d_{i} = OpLoad %float %{data}
        %ldsaar_du_{i} = OpBitcast %uint %ldsaar_d_{i}
        %ldsaar_old_{i} = OpAtomicIAdd %uint %ldsaar_p_{i} %uint_2 %uint_0 %ldsaar_du_{i}
        %ldsaar_of_{i} = OpBitcast %float %ldsaar_old_{i}
               OpStore %{dst} %ldsaar_of_{i}
               OpBranch %ldsaar_end_{i}
        %ldsaar_end_{i} = OpLabel
",
        addr = addr.value,
        data = data.value,
        dst = dst.value,
    );

    *dst_source += &text;
    Ok(true)
}

/// Beyond Kyty (`ds_wrxchg_rtn_b32` is `KYTY_NI` upstream): LDS atomic
/// write-exchange returning the OLD value — `vdst = lds[a]; lds[a] = data`,
/// via `OpAtomicExchange` on `%lds[(addr + offset) >> 2]` at Workgroup scope
/// (2), Relaxed semantics (0). Address computation, bound clamp and the
/// `exec_lo` guard mirror [`recompile_ds_add_u32`]; the exchange's old value is
/// written to `vdst` (masked-off lanes skip the body and keep `vdst`). Measured
/// on ASTRO.BOT tiled-lighting compute (raw 0xd8b40510).
fn recompile_ds_wrxchg_rtn_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_DsWrxchgRtnB32_VdstVsrc0Vsrc1Vsrc2";
    let inst = inst_at(code, index, FUNC)?;

    if !operand_is_variable(inst.dst)
        || !operand_is_variable(inst.src[0])
        || !operand_is_variable(inst.src[1])
    {
        return Err(not_supported(FUNC, "dst/addr/data are not variables"));
    }
    if !operand_is_constant(inst.src[2]) {
        return Err(not_supported(FUNC, "offset is not a constant"));
    }

    let dst = operand_variable_to_str(inst.dst);
    let addr = operand_variable_to_str(inst.src[0]);
    let data = operand_variable_to_str(inst.src[1]);
    if dst.type_ != SpirvType::Float
        || addr.type_ != SpirvType::Float
        || data.type_ != SpirvType::Float
    {
        return Err(not_supported(FUNC, "unexpected operand types"));
    }
    let offset = spirv.get_constant(inst.src[2]);
    let clamp = spirv.get_constant_uint(spirv.lds_size_dw() - 1);

    let i = format!("{index}");
    let text = format!(
        "
        %ldsx_e0_{i} = OpLoad %uint %exec_lo
        %ldsx_e1_{i} = OpINotEqual %bool %ldsx_e0_{i} %uint_0
               OpSelectionMerge %ldsx_end_{i} None
               OpBranchConditional %ldsx_e1_{i} %ldsx_body_{i} %ldsx_end_{i}
        %ldsx_body_{i} = OpLabel
        %ldsx_a_{i} = OpLoad %float %{addr}
        %ldsx_au_{i} = OpBitcast %uint %ldsx_a_{i}
        %ldsx_ao_{i} = OpIAdd %uint %ldsx_au_{i} %{offset}
        %ldsx_ai_{i} = OpShiftRightLogical %uint %ldsx_ao_{i} %uint_2
        %ldsx_ac_{i} = OpExtInst %uint %GLSL_std_450 UMin %ldsx_ai_{i} %{clamp}
        %ldsx_p_{i} = OpAccessChain %_ptr_Workgroup_uint %lds %ldsx_ac_{i}
        %ldsx_d_{i} = OpLoad %float %{data}
        %ldsx_du_{i} = OpBitcast %uint %ldsx_d_{i}
        %ldsx_old_{i} = OpAtomicExchange %uint %ldsx_p_{i} %uint_2 %uint_0 %ldsx_du_{i}
        %ldsx_of_{i} = OpBitcast %float %ldsx_old_{i}
               OpStore %{dst} %ldsx_of_{i}
               OpBranch %ldsx_end_{i}
        %ldsx_end_{i} = OpLabel
",
        addr = addr.value,
        data = data.value,
        dst = dst.value,
    );

    *dst_source += &text;
    Ok(true)
}

/// The read twin of [`recompile_ds_write_b32`]:
/// `vdst = lds[(addr + offset) >> 2]`. The destination keeps its old value
/// when `exec` is off (the OpSelect pattern the mbcnt pair uses).
fn recompile_ds_read_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_DsReadB32_SVdstSVsrc0SVsrc1";
    let inst = inst_at(code, index, FUNC)?;

    if !operand_is_variable(inst.dst) || !operand_is_variable(inst.src[0]) {
        return Err(not_supported(FUNC, "dst/addr are not variables"));
    }
    if !operand_is_constant(inst.src[1]) {
        return Err(not_supported(FUNC, "offset is not a constant"));
    }

    let dst_value = operand_variable_to_str(inst.dst);
    let addr = operand_variable_to_str(inst.src[0]);
    if dst_value.type_ != SpirvType::Float || addr.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "unexpected operand types"));
    }
    let offset = spirv.get_constant(inst.src[1]);
    let clamp = spirv.get_constant_uint(spirv.lds_size_dw() - 1);

    let i = format!("{index}");
    let text = format!(
        "
        %ldsr_a_{i} = OpLoad %float %{addr}
        %ldsr_au_{i} = OpBitcast %uint %ldsr_a_{i}
        %ldsr_ao_{i} = OpIAdd %uint %ldsr_au_{i} %{offset}
        %ldsr_ai_{i} = OpShiftRightLogical %uint %ldsr_ao_{i} %uint_2
        %ldsr_ac_{i} = OpExtInst %uint %GLSL_std_450 UMin %ldsr_ai_{i} %{clamp}
        %ldsr_p_{i} = OpAccessChain %_ptr_Workgroup_uint %lds %ldsr_ac_{i}
        %ldsr_v_{i} = OpLoad %uint %ldsr_p_{i}
        %ldsr_f_{i} = OpBitcast %float %ldsr_v_{i}
        %ldsr_e0_{i} = OpLoad %uint %exec_lo
        %ldsr_e1_{i} = OpINotEqual %bool %ldsr_e0_{i} %uint_0
        %ldsr_o_{i} = OpLoad %float %{dst}
        %ldsr_s_{i} = OpSelect %float %ldsr_e1_{i} %ldsr_f_{i} %ldsr_o_{i}
               OpStore %{dst} %ldsr_s_{i}
",
        addr = addr.value,
        dst = dst_value.value,
    );

    *dst_source += &text;
    Ok(true)
}

/// Beyond Kyty (`ds_read2_b32` is `KYTY_NI` upstream): two independent LDS
/// dword reads — `vdst = lds[(addr + offset0) >> 2]`,
/// `vdst+1 = lds[(addr + offset1) >> 2]` (the parser scaled the encoded
/// dword-unit offsets to bytes). Each result keeps its old value when `exec`
/// is off, exactly like [`recompile_ds_read_b32`].
fn recompile_ds_read2_b32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    ds_read_2dw(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_DsRead2B32_Vdst2Vsrc0Vsrc1Vsrc2",
    )
}

/// Beyond Kyty (`ds_read_b64` is `KYTY_NI` upstream): two CONSECUTIVE LDS
/// dwords at one byte offset — the parser materialises the second offset
/// literal as `offset + 4`, so this is exactly the [`recompile_ds_read2_b32`]
/// body. Measured on ASTRO.BOT scene compute (raw 0xd9d80000).
fn recompile_ds_read_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    ds_read_2dw(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_DsReadB64_Vdst2Vsrc0Vsrc1Vsrc2",
    )
}

/// Beyond Kyty (`ds_read_b128` is `KYTY_NI` upstream): four CONSECUTIVE LDS
/// dwords at one byte offset (RDNA2 ISA `DS_READ_B128`) — the four-dword
/// extension of [`recompile_ds_read_b64`]'s model. Dword `k` reads
/// `lds[(addr + offset + 4k) >> 2]` (the derived offsets are materialised as
/// constants in `FindConstants`); each result keeps its old value when
/// `exec` is off. Measured on ASTRO.BOT scene compute (58 dispatches/run).
fn recompile_ds_read_b128(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    ds_read_ndw(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_DsReadB128_Vdst4Vsrc0Vsrc1",
        4,
    )
}

/// Beyond Kyty (`ds_read_b96` is `KYTY_NI` upstream): the three-dword row of
/// the [`recompile_ds_read_b128`] model. Measured on ASTRO.BOT scene compute
/// (raw 0xdbf80550, 58 dispatches/run).
fn recompile_ds_read_b96(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    ds_read_ndw(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_DsReadB96_Vdst3Vsrc0Vsrc1",
        3,
    )
}

/// Shared body of the consecutive multi-dword LDS reads (`ds_read_b96` /
/// `ds_read_b128`): dword `k` reads `lds[(addr + offset + 4k) >> 2]` (the
/// derived offsets are materialised as constants in `FindConstants`); each
/// result keeps its old value when `exec` is off.
fn ds_read_ndw(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    n: i32,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    if !operand_is_variable(inst.dst) || !operand_is_variable(inst.src[0]) {
        return Err(not_supported(func, "dst/addr are not variables"));
    }
    if !operand_is_constant(inst.src[1]) {
        return Err(not_supported(func, "offset is not a constant"));
    }

    let addr = operand_variable_to_str(inst.src[0]);
    if addr.type_ != SpirvType::Float {
        return Err(not_supported(func, "unexpected addr operand type"));
    }
    let clamp = spirv.get_constant_uint(spirv.lds_size_dw() - 1);
    let base = inst.src[1].constant.u;

    let mut text = String::new();
    for k in 0..n {
        let dst_value = operand_variable_to_str_shift(inst.dst, k);
        if dst_value.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected dst operand type"));
        }
        let offset = spirv.get_constant_uint(base + 4 * k as u32);
        let i = format!("{index}_{k}");
        text += &format!(
            "
        %ldsr4_a_{i} = OpLoad %float %{addr}
        %ldsr4_au_{i} = OpBitcast %uint %ldsr4_a_{i}
        %ldsr4_ao_{i} = OpIAdd %uint %ldsr4_au_{i} %{offset}
        %ldsr4_ai_{i} = OpShiftRightLogical %uint %ldsr4_ao_{i} %uint_2
        %ldsr4_ac_{i} = OpExtInst %uint %GLSL_std_450 UMin %ldsr4_ai_{i} %{clamp}
        %ldsr4_p_{i} = OpAccessChain %_ptr_Workgroup_uint %lds %ldsr4_ac_{i}
        %ldsr4_v_{i} = OpLoad %uint %ldsr4_p_{i}
        %ldsr4_f_{i} = OpBitcast %float %ldsr4_v_{i}
        %ldsr4_e0_{i} = OpLoad %uint %exec_lo
        %ldsr4_e1_{i} = OpINotEqual %bool %ldsr4_e0_{i} %uint_0
        %ldsr4_o_{i} = OpLoad %float %{dst}
        %ldsr4_s_{i} = OpSelect %float %ldsr4_e1_{i} %ldsr4_f_{i} %ldsr4_o_{i}
               OpStore %{dst} %ldsr4_s_{i}
",
            addr = addr.value,
            dst = dst_value.value,
        );
    }

    *dst_source += &text;
    Ok(true)
}

/// Shared body of the two-dword LDS reads (`ds_read2_b32` / `ds_read_b64`):
/// `vdst = lds[(addr + src1) >> 2]`, `vdst+1 = lds[(addr + src2) >> 2]`.
fn ds_read_2dw(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    if !operand_is_variable(inst.dst) || !operand_is_variable(inst.src[0]) {
        return Err(not_supported(func, "dst/addr are not variables"));
    }
    if !operand_is_constant(inst.src[1]) || !operand_is_constant(inst.src[2]) {
        return Err(not_supported(func, "offsets are not constants"));
    }

    let addr = operand_variable_to_str(inst.src[0]);
    if addr.type_ != SpirvType::Float {
        return Err(not_supported(func, "unexpected addr operand type"));
    }
    let clamp = spirv.get_constant_uint(spirv.lds_size_dw() - 1);

    let mut text = String::new();
    for k in 0..2 {
        let dst_value = operand_variable_to_str_shift(inst.dst, k);
        if dst_value.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected dst operand type"));
        }
        let offset = spirv.get_constant(inst.src[1 + k as usize]);
        let i = format!("{index}_{k}");
        text += &format!(
            "
        %ldsr2_a_{i} = OpLoad %float %{addr}
        %ldsr2_au_{i} = OpBitcast %uint %ldsr2_a_{i}
        %ldsr2_ao_{i} = OpIAdd %uint %ldsr2_au_{i} %{offset}
        %ldsr2_ai_{i} = OpShiftRightLogical %uint %ldsr2_ao_{i} %uint_2
        %ldsr2_ac_{i} = OpExtInst %uint %GLSL_std_450 UMin %ldsr2_ai_{i} %{clamp}
        %ldsr2_p_{i} = OpAccessChain %_ptr_Workgroup_uint %lds %ldsr2_ac_{i}
        %ldsr2_v_{i} = OpLoad %uint %ldsr2_p_{i}
        %ldsr2_f_{i} = OpBitcast %float %ldsr2_v_{i}
        %ldsr2_e0_{i} = OpLoad %uint %exec_lo
        %ldsr2_e1_{i} = OpINotEqual %bool %ldsr2_e0_{i} %uint_0
        %ldsr2_o_{i} = OpLoad %float %{dst}
        %ldsr2_s_{i} = OpSelect %float %ldsr2_e1_{i} %ldsr2_f_{i} %ldsr2_o_{i}
               OpStore %{dst} %ldsr2_s_{i}
",
            addr = addr.value,
            dst = dst_value.value,
        );
    }

    *dst_source += &text;
    Ok(true)
}

/// Beyond Kyty (`ds_write_b96` is `KYTY_NI` upstream): three consecutive LDS
/// dwords — `lds[(addr + offset) >> 2 .. +3] = data0..data0+2`. One
/// exec_lo guard around all three stores (the [`recompile_ds_write_b32`]
/// pattern); each index is clamped to the array bound independently.
fn recompile_ds_write_b96(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    ds_write_bn(
        index,
        code,
        dst_source,
        spirv,
        3,
        "Recompile_DsWriteB96_Vsrc0Vsrc13Vsrc2",
    )
}

/// Beyond Kyty (`ds_write_b128` is `KYTY_NI` upstream): the four-dword row of
/// the same model — measured on ASTRO.BOT scene compute (raw 0xdb7c0000).
fn recompile_ds_write_b128(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    ds_write_bn(
        index,
        code,
        dst_source,
        spirv,
        4,
        "Recompile_DsWriteB128_Vsrc0Vsrc14Vsrc2",
    )
}

/// Shared body of the multi-dword LDS stores:
/// `lds[(addr + offset) >> 2 .. +n] = data0..data0+(n-1)`.
fn ds_write_bn(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    n: i32,
    func: &'static str,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    if !operand_is_variable(inst.src[0]) || !operand_is_variable(inst.src[1]) {
        return Err(not_supported(func, "addr/data are not variables"));
    }
    if !operand_is_constant(inst.src[2]) {
        return Err(not_supported(func, "offset is not a constant"));
    }

    let addr = operand_variable_to_str(inst.src[0]);
    if addr.type_ != SpirvType::Float {
        return Err(not_supported(func, "unexpected addr operand type"));
    }
    let offset = spirv.get_constant(inst.src[2]);
    let clamp = spirv.get_constant_uint(spirv.lds_size_dw() - 1);

    let i = format!("{index}");
    let mut text = format!(
        "
        %ldsw3_e0_{i} = OpLoad %uint %exec_lo
        %ldsw3_e1_{i} = OpINotEqual %bool %ldsw3_e0_{i} %uint_0
               OpSelectionMerge %ldsw3_end_{i} None
               OpBranchConditional %ldsw3_e1_{i} %ldsw3_body_{i} %ldsw3_end_{i}
        %ldsw3_body_{i} = OpLabel
        %ldsw3_a_{i} = OpLoad %float %{addr}
        %ldsw3_au_{i} = OpBitcast %uint %ldsw3_a_{i}
        %ldsw3_ao_{i} = OpIAdd %uint %ldsw3_au_{i} %{offset}
        %ldsw3_ai_{i} = OpShiftRightLogical %uint %ldsw3_ao_{i} %uint_2
",
        addr = addr.value,
    );
    for k in 0..n {
        let data = operand_variable_to_str_shift(inst.src[1], k);
        if data.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected data operand type"));
        }
        text += &format!(
            "        %ldsw3_ix_{i}_{k} = OpIAdd %uint %ldsw3_ai_{i} %uint_{k}
        %ldsw3_ac_{i}_{k} = OpExtInst %uint %GLSL_std_450 UMin %ldsw3_ix_{i}_{k} %{clamp}
        %ldsw3_p_{i}_{k} = OpAccessChain %_ptr_Workgroup_uint %lds %ldsw3_ac_{i}_{k}
        %ldsw3_d_{i}_{k} = OpLoad %float %{data}
        %ldsw3_du_{i}_{k} = OpBitcast %uint %ldsw3_d_{i}_{k}
               OpStore %ldsw3_p_{i}_{k} %ldsw3_du_{i}_{k}
",
            data = data.value,
        );
    }
    text += &format!(
        "               OpBranch %ldsw3_end_{i}
        %ldsw3_end_{i} = OpLabel
"
    );

    *dst_source += &text;
    Ok(true)
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

/// Which descriptor array an MIMG instruction indexes at runtime.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MimgDescriptorClass {
    /// `%textures2D_S` — sampled / texel-fetch T#s.
    Sampled,
    /// `%textures2D_L` — read-write (storage) T#s.
    Storage,
}

// `eud_load_offset_for_register` lives in `analysis` (shared with the
// raw-EUD image-descriptor capture pass there).
use crate::shader::analysis::eud_load_offset_for_register;

/// Bound on how many `s_mov_b32` register copies the EUD-alias resolver walks
/// through before giving up. Gen5 descriptor delivery is a covered `s_load`
/// optionally forwarded through a short mov chain; a deeper chain is treated as
/// unresolvable (named refusal) rather than chased indefinitely. SharpEmu
/// evaluates the FULL scalar program (`Gen5ShaderScalarEvaluator.cs:599-668`);
/// this is a bounded static form.
const EUD_ALIAS_MOV_DEPTH: u32 = 4;

/// Outcome of resolving a descriptor register's dword-0 provenance by scanning
/// the scalar program BACKWARD from the MIMG (SharpEmu's `TryCopyRegisters`,
/// `Gen5ShaderScalarEvaluator.cs:599-668`, does the dynamic version).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EudResolve {
    /// The nearest writer is a covered EUD `s_load` (optionally reached through
    /// `s_mov_b32` copies): the load's EUD dword offset.
    Offset(i32),
    /// The nearest writer redefines the register some OTHER way — arithmetic, a
    /// buffer/SRT load, an immediate move, a partial/mid-descriptor load, or a
    /// chain deeper than [`EUD_ALIAS_MOV_DEPTH`]. The value is NOT an EUD-load
    /// alias; refuse. Never chase past a real redefinition to a now-dead load —
    /// that is the device-loss guard's whole purpose.
    Blocked,
    /// No instruction writes the register before the MIMG (a persistent user
    /// SGPR, or only a textually-later load across a loop back-edge). Program
    /// order is silent; the caller may fall back to the order-independent scan.
    NoProducer,
}

/// The scalar-register dword span an instruction's destination writes, or
/// `None` when the destination is not the SGPR file (VGPR/EXEC/VCC/M0/…). The
/// width is the max of the opcode's fixed width and the parser-recorded operand
/// size, so a wide `s_load`/`s_buffer_load` covering `reg` as a MIDDLE dword is
/// still detected as a redefinition (never skipped past to a dead earlier load).
fn sgpr_dst_span(inst: &ShaderInstruction) -> Option<(i32, i32)> {
    use ShaderInstructionType as T;
    if inst.dst.type_ != ShaderOperandType::Sgpr {
        return None;
    }
    let opcode_width = match inst.type_ {
        T::SLoadDwordx16 | T::SBufferLoadDwordx16 => 16,
        T::SLoadDwordx8 | T::SBufferLoadDwordx8 => 8,
        T::SLoadDwordx4 | T::SBufferLoadDwordx4 => 4,
        T::SLoadDwordx2 | T::SBufferLoadDwordx2 | T::SMovB64 => 2,
        _ => 1,
    };
    Some((inst.dst.register_id, opcode_width.max(inst.dst.size.max(1))))
}

/// The EUD dword offset a covered `s_load{,x2,x4,x8,x16} s[reg..],
/// s[eud_base..], <const>` loads `reg`'s dword 0 from, or `None` when `inst` is
/// not exactly that shape (the same acceptance as [`sload_dword_extended`]).
fn eud_load_into_reg_offset(inst: &ShaderInstruction, eud_base: i32, reg: i32) -> Option<i32> {
    use ShaderInstructionType as T;
    match inst.type_ {
        T::SLoadDword | T::SLoadDwordx2 | T::SLoadDwordx4 | T::SLoadDwordx8 | T::SLoadDwordx16 => {}
        _ => return None,
    }
    if inst.src[0].type_ == ShaderOperandType::Sgpr
        && inst.src[0].register_id == eud_base
        && inst.dst.type_ == ShaderOperandType::Sgpr
        && inst.dst.register_id == reg
        && matches!(
            inst.src[1].type_,
            ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant
        )
        && inst.src[1].constant.i() >= 0
    {
        Some((inst.src[1].constant.u >> 2) as i32)
    } else {
        None
    }
}

/// Resolve the EUD dword offset that fills `reg`'s dword 0 as seen at
/// instruction index `at`, walking the scalar program BACKWARD through a
/// bounded chain of `s_mov_b32` register copies.
///
/// This is the program-order (SharpEmu scalar-evaluation) form of the alias
/// resolution: it binds `reg` to the descriptor the NEAREST preceding covered
/// load delivered, which disambiguates a register reused for several
/// descriptors at different EUD offsets — measured on ASTRO.BOT sampled compute
/// (e.g. `0x5006fff00`): `s16` is loaded from EUD dwords 0 / 40 / 44 / 52 at
/// different program points, and the order-independent scan would pick the
/// wrong one. Only SALU register copies (`s_mov_b32 reg, <sgpr>`) are followed;
/// an immediate move, arithmetic, or a non-EUD load is an opaque redefinition
/// that stops the walk with [`EudResolve::Blocked`].
fn resolve_eud_offset_before(
    code: &ShaderCode,
    eud_base: i32,
    reg: i32,
    at: usize,
    depth: u32,
) -> EudResolve {
    use ShaderInstructionType as T;
    let insts = code.get_instructions();
    let start = at.min(insts.len());
    for i in (0..start).rev() {
        let inst = &insts[i];
        // A covered EUD load into reg's dword 0 — the resolved alias.
        if let Some(off) = eud_load_into_reg_offset(inst, eud_base, reg) {
            return EudResolve::Offset(off);
        }
        // `s_mov_b32 reg, <sgpr>` forwards another register's dword 0; follow
        // the source as of BEFORE this move.
        if inst.type_ == T::SMovB32
            && inst.dst.type_ == ShaderOperandType::Sgpr
            && inst.dst.register_id == reg
            && inst.src[0].type_ == ShaderOperandType::Sgpr
        {
            if depth >= EUD_ALIAS_MOV_DEPTH {
                return EudResolve::Blocked;
            }
            return resolve_eud_offset_before(
                code,
                eud_base,
                inst.src[0].register_id,
                i,
                depth + 1,
            );
        }
        // Any other write covering reg redefines its value: stop here rather
        // than chasing past it to a now-dead earlier load.
        if let Some((base, width)) = sgpr_dst_span(inst) {
            if reg >= base && reg < base + width {
                return EudResolve::Blocked;
            }
        }
    }
    EudResolve::NoProducer
}

/// Program-order EUD-offset resolution with the order-independent scan as a
/// fallback only when program order is silent (`NoProducer`). A [`Blocked`]
/// producer (arithmetic / raw load / immediate move in the chain) never falls
/// back — the refusal must stand.
///
/// [`Blocked`]: EudResolve::Blocked
fn eud_alias_offset(code: &ShaderCode, eud_base: i32, reg: i32, at: usize) -> Option<i32> {
    match resolve_eud_offset_before(code, eud_base, reg, at, 0) {
        EudResolve::Offset(k) => Some(k),
        EudResolve::Blocked => None,
        EudResolve::NoProducer => eud_load_offset_for_register(code, eud_base, reg),
    }
}

/// SharpEmu-parity EUD-alias resolution for an MIMG descriptor register that
/// matched no captured descriptor DIRECTLY.
///
/// SharpEmu resolves every image descriptor by evaluating the shader's scalar
/// program — the register file holds the real descriptor at the MIMG, copied
/// straight out (`reference/sharpemu/src/SharpEmu.ShaderCompiler/`
/// `Gen5ShaderScalarEvaluator.cs:599-668`, `TryCopyRegisters`). This port's
/// captured-descriptor table is the equivalent, but a Gen5 CS commonly reads a
/// T#/S# from a REAL SGPR that a covered `s_load` off the EUD pair fills at
/// runtime, while the same descriptor is captured at its EUD-*virtual* start
/// register (`start >= SGPRS_MAX`, or rebased on the EUD base — measured on
/// ASTRO.BOT scene compute `0x5006c5f00`: `image_store` reads `s0`, but the
/// storage T# is captured at `s32` = EUD dword 0, loaded into `s0` by
/// `s_load_dwordx8 s[0:7], s[12:13], 0x0`).
///
/// `WriteLocalVariables` maps the descriptor's EUD dwords to push-constant
/// fields keyed by EUD dword (`eud_rel_index`), and `sload_dword_extended`
/// lowers the covered load to seed the load's DEST register with the rewritten
/// dword 0 (`GetMappedIndex`) — so `reg` already holds the descriptor's
/// array index at the MIMG regardless of `reg != start_register`. The refusal
/// is therefore spurious whenever: `reg` is filled by such a covered load off
/// the EUD base at dword `k`, and a captured EUD-resident descriptor of the
/// requested class has its dword 0 mapped at exactly `k`
/// (`eud_rel_index(start_register) == k`). This resolves the alias without
/// inventing a descriptor or binding any uncaptured guest bytes — the bound
/// descriptor is the already-validated captured one.
pub(crate) fn mimg_register_eud_alias_index(
    bind: &ShaderBindResources,
    code: &ShaderCode,
    reg: i32,
    at: usize,
    want_storage: bool,
) -> Option<usize> {
    if !bind.extended.used {
        return None;
    }
    let k = eud_alias_offset(code, bind.extended.start_register, reg, at)?;
    let tex_num = usize::try_from(bind.textures2d.textures_num.max(0))
        .unwrap_or(0)
        .min(bind.textures2d.desc.len());
    bind.textures2d.desc[..tex_num].iter().position(|d| {
        d.extended
            && d.textures2d_without_sampler == want_storage
            && super::spirv::eud_rel_index(bind, d.start_register, 0, "mimg alias")
                .is_ok_and(|rel| rel == k)
    })
}

/// EUD-alias resolution for a sampler register (see
/// [`mimg_register_eud_alias_index`]): the S# is captured at its EUD-virtual
/// start register but read by the MIMG from a real SGPR a covered EUD `s_load`
/// fills.
pub(crate) fn sampler_register_is_eud_alias(
    bind: &ShaderBindResources,
    code: &ShaderCode,
    reg: i32,
    at: usize,
) -> bool {
    if !bind.extended.used {
        return false;
    }
    let Some(k) = eud_alias_offset(code, bind.extended.start_register, reg, at) else {
        return false;
    };
    let samp_num = usize::try_from(bind.samplers.samplers_num.max(0))
        .unwrap_or(0)
        .min(bind.samplers.start_register.len());
    (0..samp_num).any(|i| {
        bind.samplers.extended[i]
            && super::spirv::eud_rel_index(bind, bind.samplers.start_register[i], 0, "mimg alias")
                .is_ok_and(|rel| rel == k)
    })
}

/// SharpEmu-parity translate-time guard against RUNTIME image descriptors.
///
/// The recompiled MIMG bodies index `%textures2D_S` / `%textures2D_L` /
/// `%samplers` with the VALUE of the instruction's T#/S# start SGPR, which is
/// only safe because `WriteLocalVariables` seeds exactly the CAPTURED
/// descriptors' start registers with their rewritten dword 0 (the
/// descriptor-array index, `prepare_stage_binding`). Two shapes break that
/// contract and produced a measured `VK_ERROR_DEVICE_LOST` on ASTRO.BOT
/// (CS `0x5006c5f00`, round 10 — descriptor-array OOB indexing, which
/// robustness features do not cover):
///
/// 1. the MIMG's T#/S# registers match NO captured descriptor (the sharp is
///    resolved at runtime, e.g. loaded through the raw EUD window), so the
///    index SGPR is undefined or raw guest data;
/// 2. the registers DO match a captured descriptor, but a raw
///    (uncovered-EUD) `s_load` overwrites them with raw guest dwords before
///    use.
///
/// Both are refused by name — the error string carries
/// `dynamic-image-descriptor`, matching SharpEmu, whose scalar evaluator
/// errors `dynamic-image-descriptor` and SKIPS the dispatch whenever an
/// image's descriptor cannot be pinned to one translate-time value
/// (`reference/sharpemu/src/SharpEmu.ShaderCompiler/`
/// `Gen5ShaderScalarEvaluator.cs` L654-662). The refusal surfaces as a named
/// counted dispatch/draw skip (`translate_cs` error path), never a submit.
///
/// Ordering is deliberately ignored in shape 2 (a raw load AFTER the MIMG
/// also refuses): linear order is unreliable across branches, and the
/// conservative refusal is a named skip, not a wrong render.
/// The per-descriptor sampled-image route for a MIMG body: the SPIR-V id
/// suffix that points its `%textures2D_S`/`%ImageS`/`%SampledImage`/pointer
/// references at the array matching the sampled T#'s own (Dim, numeric
/// class) key, plus that Dim (for coordinate-component selection) and class
/// (for result typing — [`route_sampled_class`]). In a homogeneous shader the
/// suffix is empty and the key is the shader-wide one, so output is
/// byte-identical to the single-array path for the Float class.
///
/// `matched` is [`mimg_descriptor_guard`]'s resolution of WHICH captured
/// descriptor the instruction's T# names — through a direct start-register
/// match or the program-order EUD-alias walk. Routing by that same resolution
/// (instead of re-matching registers here) is what lets an EUD-aliased T# in
/// a mixed shader reach its own key's array; a bare register re-match would
/// re-refuse exactly the aliased descriptors the walk was built to accept
/// (ASTRO.BOT, refusals 642->90).
fn sampled_site_route(
    spirv: &Spirv<'_>,
    matched: Option<usize>,
    func: &'static str,
) -> Result<(&'static str, SampledDim, SampledClass), ShaderRecompileError> {
    let bind = spirv
        .get_bind_info()
        .expect("sampled MIMG bodies check textures2d_sampled_num > 0 first");
    let present = sampled_keys_present(bind);
    if present.len() <= 1 {
        // Homogeneous (or a fixture with no captured T# dwords): the single
        // present key, defaulting to the legacy 2D-float shape.
        let (dim, class) = present
            .first()
            .copied()
            .unwrap_or((SampledDim::Two, SampledClass::Float));
        return Ok(("", dim, class));
    }
    // `mimg_descriptor_guard` runs before every route call and refuses an
    // unmatched T#, so a mixed shader reaching here without a resolution is
    // a defensive named refusal, not a reachable path.
    let Some(i) = matched else {
        return Err(not_supported(
            func,
            "mixed-key sample T# matches no captured sampled descriptor",
        ));
    };
    // Bounded: the guard produced `i` from an iterator over
    // `desc[..tex_num]` with `tex_num` clamped to `desc.len()`.
    let (dim, class) = sampled_key_of(&bind.textures2d.desc[i].texture);
    Ok((sampled_key_suffix(dim, class), dim, class))
}

/// Whether the sampled descriptor resolved for this MIMG is a guest cube
/// (T# type 11). Guest cube samples are represented as Vulkan 2D-array views:
/// the GCN `V_CUBE*` sequence has already selected the face, but the guest
/// hardware convention leaves S/T in [1, 2]. Vulkan's normalized 2D-array
/// coordinates are [0, 1], so sample-coordinate builders use this bit to
/// subtract one from S/T while preserving the third (face) component.
fn sampled_site_is_guest_cube(spirv: &Spirv<'_>, matched: Option<usize>) -> bool {
    let Some(i) = matched else {
        return false;
    };
    spirv
        .get_bind_info()
        .and_then(|bind| bind.textures2d.desc.get(i))
        .is_some_and(|desc| desc.texture.type_() == 11)
}

/// Rewrite the four sampled-image identifiers in a freshly-built MIMG body to
/// the per-Dim suffixed names, routing the sample to the array matching the
/// T#'s Dim in a mixed shader. `suffix == ""` (homogeneous) is a no-op.
///
/// Each identifier is always followed by a space in every MIMG body template,
/// so matching the trailing space keeps the `%_ptr_UniformConstant_ImageS` /
/// `%ImageS` overlap safe: the pointer's "ImageS" is preceded by `_`, never
/// `%`, so `%ImageS ` can never match inside `%_ptr_UniformConstant_ImageS `.
fn route_sampled_ids(body: &mut String, suffix: &str) {
    if suffix.is_empty() {
        return;
    }
    *body = body
        .replace(
            "%_ptr_UniformConstant_ImageS ",
            &format!("%_ptr_UniformConstant_ImageS{suffix} "),
        )
        .replace("%textures2D_S ", &format!("%textures2D_S{suffix} "))
        .replace("%SampledImage ", &format!("%SampledImage{suffix} "))
        .replace("%ImageS ", &format!("%ImageS{suffix} "));
}

/// The per-descriptor STORAGE-image route for a MIMG store body: the SPIR-V
/// id suffix that points its `%textures2D_L`/`%ImageL`/pointer references at
/// the array matching the RW T#'s own (Dim, storage format) key, plus that
/// Dim (for texel-coordinate width) and format. In a homogeneous shader the
/// suffix is empty and the key is the shader-wide one, so output is
/// byte-identical to the single-array path. The exact storage analogue of
/// [`sampled_site_route`] — first needed when ASTRO.BOT's ACB Phase B
/// dispatches bound a 3D Rgba16f volume and 2D Rgba16f targets in one
/// compute shader (the historical shader-global `%ImageL` refused it).
fn storage_site_route(
    spirv: &Spirv<'_>,
    matched: Option<usize>,
    func: &'static str,
) -> Result<(&'static str, SampledDim, StorageFormat), ShaderRecompileError> {
    let bind = spirv
        .get_bind_info()
        .expect("storage MIMG bodies check textures2d_storage_num > 0 first");
    let present = storage_keys_present(bind);
    if present.len() <= 1 {
        // Homogeneous (or a fixture with no captured RW T# dwords): the
        // single present key, defaulting to the legacy 2D-Rgba8 shape.
        let (dim, format) = present
            .first()
            .copied()
            .unwrap_or((SampledDim::Two, StorageFormat::Rgba8));
        return Ok(("", dim, format));
    }
    // `mimg_descriptor_guard` runs before every route call and refuses an
    // unmatched T#, so a mixed shader reaching here without a resolution is
    // a defensive named refusal, not a reachable path.
    let Some(i) = matched else {
        return Err(not_supported(
            func,
            "mixed-key storage T# matches no captured storage descriptor",
        ));
    };
    // Bounded: the guard produced `i` from an iterator over
    // `desc[..tex_num]` with `tex_num` clamped to `desc.len()`.
    let (dim, format) = storage_key_of(&bind.textures2d.desc[i].texture);
    Ok((storage_key_suffix(dim, format), dim, format))
}

/// Rewrite the three storage-image identifiers in a freshly-built MIMG store
/// body to the per-key suffixed names, routing the write to the array
/// matching the RW T#'s (Dim, format) key in a mixed shader. `suffix == ""`
/// (homogeneous) is a no-op.
///
/// Each identifier is always followed by a space in every store body
/// template, so matching the trailing space keeps the
/// `%_ptr_UniformConstant_ImageL` / `%ImageL` overlap safe: the pointer's
/// "ImageL" is preceded by `_`, never `%`, so `%ImageL ` can never match
/// inside `%_ptr_UniformConstant_ImageL `.
fn route_storage_ids(body: &mut String, suffix: &str) {
    if suffix.is_empty() {
        return;
    }
    *body = body
        .replace(
            "%_ptr_UniformConstant_ImageL ",
            &format!("%_ptr_UniformConstant_ImageL{suffix} "),
        )
        .replace("%textures2D_L ", &format!("%textures2D_L{suffix} "))
        .replace("%ImageL ", &format!("%ImageL{suffix} "));
}

/// Retype a freshly-built MIMG body's texel result for an INTEGER-class
/// sampled image: SPIR-V requires the sample/gather/fetch result to be a
/// vec4 of the image's sampled type, so each `... %v4float` result line is
/// rewritten to sample into `%v4uint`/`%v4int` and `OpBitcast` the whole
/// vector back into the float-typed register model — RAW BITS, exactly
/// SharpEmu's Uint handling (Gen5SpirvTranslator keeps Uint sample results
/// unconverted and bitcasts everything into u32 registers; the recompiler's
/// registers are float-typed bitcast equivalents), never `OpConvertUToF`.
/// `Float` (and bodies with no texel result, e.g. resinfo) is a no-op, so
/// float shaders stay byte-identical.
fn route_sampled_class(body: &mut String, class: SampledClass) {
    if class == SampledClass::Float {
        return;
    }
    const RESULT_OPS: [&str; 4] = [
        "= OpImageSampleImplicitLod %v4float ",
        "= OpImageSampleExplicitLod %v4float ",
        "= OpImageGather %v4float ",
        "= OpImageFetch %v4float ",
    ];
    let v4 = class.v4_type_str();
    let mut out = String::with_capacity(body.len() + 160);
    for line in body.split_inclusive('\n') {
        if !RESULT_OPS.iter().any(|op| line.contains(op)) {
            out.push_str(line);
            continue;
        }
        // "<indent>%id = Op... %v4float <operands>\n" becomes the same
        // instruction into "%id_raw" with the class's vec4 type, then a
        // bitcast back to the float vec4 every downstream `%id` use expects.
        let Some(eq) = line.find('=') else {
            out.push_str(line);
            continue;
        };
        let (lhs, rhs) = line.split_at(eq);
        let id = lhs.trim();
        let indent = &lhs[..lhs.len() - lhs.trim_start().len()];
        out.push_str(&format!(
            "{indent}{id}_raw {}",
            rhs.replacen("%v4float", v4, 1)
        ));
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&format!("{indent}{id} = OpBitcast %v4float {id}_raw\n"));
    }
    *body = out;
}

fn mimg_descriptor_guard(
    index: u32,
    spirv: &Spirv<'_>,
    code: &ShaderCode,
    inst: &ShaderInstruction,
    func: &'static str,
    class: MimgDescriptorClass,
    uses_sampler: bool,
) -> Result<Option<usize>, ShaderRecompileError> {
    let Some(bind) = spirv.get_bind_info() else {
        return Ok(None);
    };
    // The MIMG's own position in program order — descriptor registers are
    // resolved by walking the scalar program BACKWARD from here.
    let at = index as usize;
    let shift_regs = if spirv.get_vs_input_info().is_some_and(|v| v.gs_prolog) {
        8
    } else {
        0
    };

    // The T# operand (MIMG srsrc * 4): 8 consecutive SGPRs.
    let t_op = inst.src[1];
    if t_op.type_ != ShaderOperandType::Sgpr {
        return Err(not_supported(
            func,
            "dynamic-image-descriptor: T# operand is not an SGPR range",
        ));
    }
    let t_reg = t_op.register_id;
    let tex_num = usize::try_from(bind.textures2d.textures_num.max(0))
        .unwrap_or(0)
        .min(bind.textures2d.desc.len());
    // `textures2d_without_sampler` is the recompiler-native storage/sampled
    // discriminator (`sampled_keys_present` / `storage_keys_present`
    // split the two SPIR-V arrays on it). The resolution keeps WHICH
    // descriptor matched (its `desc` index): `sampled_site_route` routes the
    // MIMG body to that descriptor's own Dim array in a mixed-dim shader —
    // for the direct match AND for the EUD-alias walk equally.
    let t_matched = bind.textures2d.desc[..tex_num]
        .iter()
        .position(|d| {
            d.start_register + shift_regs == t_reg
                && d.textures2d_without_sampler == (class == MimgDescriptorClass::Storage)
        })
        .or_else(|| {
            mimg_register_eud_alias_index(
                bind,
                code,
                t_reg,
                at,
                class == MimgDescriptorClass::Storage,
            )
        });
    let Some(t_index) = t_matched else {
        return Err(not_supported(
            func,
            format!(
                "dynamic-image-descriptor: {class:?} T# at s{t_reg} matches no captured descriptor"
            ),
        ));
    };

    // The S# operand (MIMG ssamp * 4): 4 consecutive SGPRs. A sample-family
    // instruction with zero captured S#s is normally rescued upstream by
    // `shader_synthesize_default_sampler`; reaching here unmatched means the
    // sampler is runtime-resolved.
    let s_range: Option<(i32, i32)> = if uses_sampler {
        let s_op = inst.src[2];
        if s_op.type_ != ShaderOperandType::Sgpr {
            return Err(not_supported(
                func,
                "dynamic-image-descriptor: S# operand is not an SGPR range",
            ));
        }
        let s_reg = s_op.register_id;
        let samp_num = usize::try_from(bind.samplers.samplers_num.max(0))
            .unwrap_or(0)
            .min(bind.samplers.start_register.len());
        let s_matched = bind.samplers.start_register[..samp_num]
            .iter()
            .any(|&start| start + shift_regs == s_reg);
        if !s_matched && !sampler_register_is_eud_alias(bind, code, s_reg, at) {
            return Err(not_supported(
                func,
                format!("dynamic-image-descriptor: S# at s{s_reg} matches no captured sampler"),
            ));
        }
        Some((s_reg, 4))
    } else {
        None
    };

    // Shape 2: a raw (uncovered-EUD) s_load re-writing the matched sharp's
    // registers replaces the seeded descriptor-array index with raw guest
    // dwords. Per-dword: only an UNCOVERED loaded dword landing INSIDE the
    // sharp's register range refuses.
    if bind.eud_raw.used {
        use ShaderInstructionType as T;
        let covered = super::spirv::eud_covered_map(bind);
        let ranges: &[(i32, i32)] = match s_range {
            Some(s) => &[(t_reg, 8), s],
            None => &[(t_reg, 8)],
        };
        for load in code.get_instructions() {
            let dwords = match load.type_ {
                T::SLoadDwordx2 => 2i32,
                T::SLoadDwordx4 => 4,
                T::SLoadDwordx8 => 8,
                T::SLoadDwordx16 => 16,
                _ => continue,
            };
            if load.src[0].type_ != ShaderOperandType::Sgpr
                || load.src[0].register_id != bind.extended.start_register
                || !matches!(
                    load.src[1].type_,
                    ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant
                )
                || load.src[1].constant.i() < 0
                || load.dst.type_ != ShaderOperandType::Sgpr
            {
                continue;
            }
            let base_dw = load.src[1].constant.u >> 2;
            let dst0 = load.dst.register_id;
            for k in 0..dwords {
                let idx = base_dw as i64 + i64::from(k);
                let is_covered = usize::try_from(idx)
                    .ok()
                    .and_then(|x| covered.get(x))
                    .copied()
                    .unwrap_or(false);
                let dst_reg = dst0 + k;
                let hits = ranges
                    .iter()
                    .any(|&(r0, len)| dst_reg >= r0 && dst_reg < r0 + len);
                if !is_covered && hits {
                    return Err(not_supported(
                        func,
                        format!(
                            "dynamic-image-descriptor: raw EUD s_load (dword {idx}) overwrites \
                             descriptor register s{dst_reg}"
                        ),
                    ));
                }
            }
        }
    }

    Ok(Some(t_index))
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
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                func,
                MimgDescriptorClass::Sampled,
                true,
            )?;
            let src0_value0 = mimg_address_value(&inst, 0);
            let src0_value1 = mimg_address_value(&inst, 1);
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

            // Cube and 3D textures sample with a 3-component coordinate; 2D
            // with 2. In a mixed shader the Dim/class (and the image-array
            // route) come from THIS sample's own T#, not a shader-wide
            // decision.
            let (suffix, dim, class) = sampled_site_route(spirv, matched, func)?;
            let guest_cube = sampled_site_is_guest_cube(spirv, matched);
            let coord = if dim.coord_components() == 3 {
                let src0_value2 = operand_variable_to_str_shift(inst.src[0], 2);
                if src0_value2.type_ != SpirvType::Float {
                    return Err(not_supported(func, "unexpected cube/3d coord type"));
                }
                if guest_cube {
                    format!(
                        "         %t39_<index> = OpLoad %float %{}
         %t40_<index> = OpLoad %float %{}
         %t41_<index> = OpLoad %float %{}
         %t39_cube_<index> = OpFSub %float %t39_<index> %float_1_000000
         %t40_cube_<index> = OpFSub %float %t40_<index> %float_1_000000
         %t42_<index> = OpCompositeConstruct %v3float %t39_cube_<index> %t40_cube_<index> %t41_<index>
",
                        src0_value0.value, src0_value1.value, src0_value2.value
                    )
                } else {
                    format!(
                        "         %t39_<index> = OpLoad %float %{}
         %t40_<index> = OpLoad %float %{}
         %t41_<index> = OpLoad %float %{}
         %t42_<index> = OpCompositeConstruct %v3float %t39_<index> %t40_<index> %t41_<index>
",
                        src0_value0.value, src0_value1.value, src0_value2.value
                    )
                }
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

            let mut body = text
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<coord>", &coord)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value)
                .replace("<index>", &format!("{index}"));
            route_sampled_ids(&mut body, suffix);
            route_sampled_class(&mut body, class);
            *dst_source += &body;

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

/// Beyond Kyty: ordinary implicit-LOD sample selecting only channel Y.
fn recompile_image_sample_dmask2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSample_Vdata1Vaddr3StSsDmask2";
    image_sample_channels(index, code, dst_source, spirv, FUNC, &[(46, 1)])
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

/// Beyond Kyty: three-channel sample with the non-contiguous mask 0xb —
/// components X, Y and W into three consecutive vdata registers.
///
/// Channel *sources* are 0, 1 and 3 (the RGBA components DMASK selects);
/// destinations are the first three vdata registers, which is what
/// `image_sample_channels` does with the index of each pair. The temp-id bases
/// match the dmask7 row (46/50/54) — only the third source component differs.
fn recompile_image_sample_dmask_b(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSample_Vdata3Vaddr3StSsDmaskB";
    image_sample_channels(
        index,
        code,
        dst_source,
        spirv,
        FUNC,
        &[(46, 0), (50, 1), (54, 3)],
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
/// Raeen currently declares sampled textures with `Depth = 0`, so SPIR-V's
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
    let matched = mimg_descriptor_guard(
        index,
        spirv,
        code,
        &inst,
        FUNC,
        MimgDescriptorClass::Sampled,
        true,
    )?;

    let (clz_suffix, clz_dim, clz_class) = sampled_site_route(spirv, matched, FUNC)?;
    // 2D uses a v2 coordinate (s, t). A 2DArray depth texture sampled through a
    // Vaddr3 MIMG carries only {dref, s, t} — there is no fourth address VGPR
    // for the array slice — so the shader is sampling array layer 0; its
    // coordinate is v3 (s, t, 0.0), routed (via `clz_suffix`) to the 2DArray
    // sampled-image array. 3D comparison sampling is invalid in SPIR-V (Dim 3D
    // forbids Dref), and a Vaddr3 Cube sample cannot carry a 3-component
    // direction plus dref, so both stay named refusals — with the concrete Dim
    // and MIMG format so the shape is measurable, not guessed.
    let clz_coord = match clz_dim {
        SampledDim::Two => concat!(
            "         %clz_coord_<index> = ",
            "OpCompositeConstruct %v2float %clz_x_<index> %clz_y_<index>"
        )
        .to_string(),
        SampledDim::TwoArray => concat!(
            "         %clz_coord_<index> = OpCompositeConstruct %v3float ",
            "%clz_x_<index> %clz_y_<index> %float_0_000000"
        )
        .to_string(),
        SampledDim::Three | SampledDim::Cube => {
            return Err(not_supported(
                FUNC,
                format!(
                    "comparison sampling of a non-2D texture (dim {clz_dim:?}, format {:?})",
                    inst.format
                ),
            ));
        }
    };

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
<clz_coord>
         %clz_sample_<index> = OpImageSampleExplicitLod %v4float %clz_sampled_image_<index> %clz_coord_<index> Lod %float_0_000000
         %clz_texel_<index> = OpCompositeExtract %float %clz_sample_<index> 0
         %clz_passes_<index> = OpFOrdLessThanEqual %bool %clz_reference_<index> %clz_texel_<index>
         %clz_result_<index> = OpSelect %float %clz_passes_<index> %float_1_000000 %float_0_000000
"#;
    let mut head = HEAD
        .replace("<clz_coord>", &clz_coord)
        .replace("<texture_index>", &texture_index.value)
        .replace("<sampler_index>", &sampler_index.value)
        .replace("<reference>", &reference.value)
        .replace("<coord_x>", &coord_x.value)
        .replace("<coord_y>", &coord_y.value)
        .replace("<index>", &format!("{index}"));
    route_sampled_ids(&mut head, clz_suffix);
    // Integer-class depth-compare: the texel is sampled in its own class and
    // bitcast to the float register model before the FOrd compare — raw-bits
    // parity with SharpEmu (a UINT depth source is degenerate either way).
    route_sampled_class(&mut head, clz_class);
    *dst_source += &head;

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

/// Dim-aware sample-coordinate construction shared by the `image_sample_lz`
/// bodies: loads 2 (2D) or 3 (Cube/3D volume) consecutive coordinate floats
/// from `src0` and builds `%t42_<index>` — the Dim comes from the same
/// `sampled_site_route` decision `Spirv::write_types` declared the image
/// array with, so coordinates and image type can never disagree.
fn sample_coord_snippet(
    dim: SampledDim,
    src0: crate::shader::types::ShaderOperand,
    func: &'static str,
    guest_cube: bool,
) -> Result<String, ShaderRecompileError> {
    let c0 = operand_variable_to_str_shift(src0, 0);
    let c1 = operand_variable_to_str_shift(src0, 1);
    if c0.type_ != SpirvType::Float || c1.type_ != SpirvType::Float {
        return Err(not_supported(func, "unexpected coord operand type"));
    }
    Ok(if dim.coord_components() == 3 {
        let c2 = operand_variable_to_str_shift(src0, 2);
        if c2.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected cube/3d coord type"));
        }
        if guest_cube {
            format!(
                "         %t39_<index> = OpLoad %float %{}
         %t40_<index> = OpLoad %float %{}
         %t41_<index> = OpLoad %float %{}
         %t39_cube_<index> = OpFSub %float %t39_<index> %float_1_000000
         %t40_cube_<index> = OpFSub %float %t40_<index> %float_1_000000
         %t42_<index> = OpCompositeConstruct %v3float %t39_cube_<index> %t40_cube_<index> %t41_<index>
",
                c0.value, c1.value, c2.value
            )
        } else {
            format!(
                "         %t39_<index> = OpLoad %float %{}
         %t40_<index> = OpLoad %float %{}
         %t41_<index> = OpLoad %float %{}
         %t42_<index> = OpCompositeConstruct %v3float %t39_<index> %t40_<index> %t41_<index>
",
                c0.value, c1.value, c2.value
            )
        }
    } else {
        format!(
            "         %t39_<index> = OpLoad %float %{}
         %t40_<index> = OpLoad %float %{}
         %t42_<index> = OpCompositeConstruct %v2float %t39_<index> %t40_<index>
",
            c0.value, c1.value
        )
    })
}

/// Beyond Kyty: `image_sample_l` with `dmask == 0x7`, measured in
/// ASTRO.BOT scene compute. VADDR contains the dimensional coordinate tuple
/// followed immediately by the explicit floating-point LOD.
fn recompile_image_sample_l_dmask7(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleL_Vdata3Vaddr4StSsDmask7";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info()
        && bind_info.textures2d.textures2d_sampled_num > 0
        && bind_info.samplers.samplers_num > 0
    {
        let matched = mimg_descriptor_guard(
            index,
            spirv,
            code,
            &inst,
            FUNC,
            MimgDescriptorClass::Sampled,
            true,
        )?;
        let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
        let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
        let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
        let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
        let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);
        let (suffix, dim, class) = sampled_site_route(spirv, matched, FUNC)?;
        let lod_value = operand_variable_to_str_shift(inst.src[0], dim.coord_components() as i32);

        if dst_value0.type_ != SpirvType::Float
            || src1_value0.type_ != SpirvType::Uint
            || src2_value0.type_ != SpirvType::Uint
            || lod_value.type_ != SpirvType::Float
        {
            return Err(not_supported(FUNC, "unexpected operand types"));
        }

        let coord = sample_coord_snippet(
            dim,
            inst.src[0],
            FUNC,
            sampled_site_is_guest_cube(spirv, matched),
        )?;
        let mut body = r#"
         %isl_t24_<index> = OpLoad %uint %<src1_value0>
         %isl_t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %isl_t24_<index>
         %isl_t27_<index> = OpLoad %ImageS %isl_t26_<index>
         %isl_t33_<index> = OpLoad %uint %<src2_value0>
         %isl_t35_<index> = OpAccessChain %_ptr_UniformConstant_Sampler %samplers %isl_t33_<index>
         %isl_t36_<index> = OpLoad %Sampler %isl_t35_<index>
         %isl_t38_<index> = OpSampledImage %SampledImage %isl_t27_<index> %isl_t36_<index>

<coord>
         %isl_lod_<index> = OpLoad %float %<lod_value>
         %isl_sample_<index> = OpImageSampleExplicitLod %v4float %isl_t38_<index> %t42_<index> Lod %isl_lod_<index>
               OpStore %temp_v4float %isl_sample_<index>
         %isl_c0p_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_0
         %isl_c0_<index> = OpLoad %float %isl_c0p_<index>
               OpStore %<dst_value0> %isl_c0_<index>
         %isl_c1p_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_1
         %isl_c1_<index> = OpLoad %float %isl_c1p_<index>
               OpStore %<dst_value1> %isl_c1_<index>
         %isl_c2p_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_2
         %isl_c2_<index> = OpLoad %float %isl_c2p_<index>
               OpStore %<dst_value2> %isl_c2_<index>
"#
        .replace("<coord>", &coord)
        .replace("<index>", &format!("{index}"))
        .replace("<src1_value0>", &src1_value0.value)
        .replace("<src2_value0>", &src2_value0.value)
        .replace("<lod_value>", &lod_value.value)
        .replace("<dst_value0>", &dst_value0.value)
        .replace("<dst_value1>", &dst_value1.value)
        .replace("<dst_value2>", &dst_value2.value);
        route_sampled_ids(&mut body, suffix);
        route_sampled_class(&mut body, class);
        *dst_source += &body;
        return Ok(true);
    }

    Ok(false)
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
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                FUNC,
                MimgDescriptorClass::Sampled,
                true,
            )?;
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);

            if dst_value0.type_ != SpirvType::Float
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

<coord>
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
            let (suffix, dim, class) = sampled_site_route(spirv, matched, FUNC)?;
            let coord = sample_coord_snippet(
                dim,
                inst.src[0],
                FUNC,
                sampled_site_is_guest_cube(spirv, matched),
            )?;
            let mut body = TEXT
                .replace("<coord>", &coord)
                .replace("<index>", &format!("{index}"))
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value);
            route_sampled_ids(&mut body, suffix);
            route_sampled_class(&mut body, class);
            *dst_source += &body;

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
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                FUNC,
                MimgDescriptorClass::Sampled,
                true,
            )?;
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let dst_value3 = operand_variable_to_str_shift(inst.dst, 3);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);

            if dst_value0.type_ != SpirvType::Float
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

<coord>
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
            let (suffix, dim, class) = sampled_site_route(spirv, matched, FUNC)?;
            let coord = sample_coord_snippet(
                dim,
                inst.src[0],
                FUNC,
                sampled_site_is_guest_cube(spirv, matched),
            )?;
            let mut body = TEXT
                .replace("<coord>", &coord)
                .replace("<index>", &format!("{index}"))
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value)
                .replace("<dst_value3>", &dst_value3.value);
            route_sampled_ids(&mut body, suffix);
            route_sampled_class(&mut body, class);
            *dst_source += &body;

            return Ok(true);
        }
    }

    Ok(false)
}

/// Beyond-Kyty shared body: `image_sample_lz` writing a single channel
/// (dmask 0x1 selects .x, dmask 0x2 selects .y) — measured in ASTRO.BOT scene
/// compute. Same explicit-LOD-zero 2D lowering as
/// [`recompile_image_sample_lz_dmask7`], storing only `chan`.
fn image_sample_lz_single_channel(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    chan: u32,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_sampled_num > 0 && bind_info.samplers.samplers_num > 0 {
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                func,
                MimgDescriptorClass::Sampled,
                true,
            )?;
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);

            if dst_value0.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src2_value0.type_ != SpirvType::Uint
            {
                return Err(not_supported(func, "unexpected operand types"));
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

<coord>
         %t43_<index> = OpImageSampleExplicitLod %v4float %t38_<index> %t42_<index> Lod %float_0_000000
               OpStore %temp_v4float %t43_<index>
         %t46_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_<chan>
         %t47_<index> = OpLoad %float %t46_<index>
               OpStore %<dst_value0> %t47_<index>
"#;
            let (suffix, dim, class) = sampled_site_route(spirv, matched, func)?;
            let coord = sample_coord_snippet(
                dim,
                inst.src[0],
                func,
                sampled_site_is_guest_cube(spirv, matched),
            )?;
            let mut body = TEXT
                .replace("<coord>", &coord)
                .replace("<index>", &format!("{index}"))
                .replace("<chan>", &format!("{chan}"))
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value)
                .replace("<dst_value0>", &dst_value0.value);
            route_sampled_ids(&mut body, suffix);
            route_sampled_class(&mut body, class);
            *dst_source += &body;

            return Ok(true);
        }
    }

    Ok(false)
}

/// Beyond-Kyty: `image_sample_lz` with `dmask == 0x3` (.xy, explicit LOD 0)
/// — measured on ASTRO.BOT scene compute (58 dispatches/run). Same lowering
/// as [`recompile_image_sample_lz_dmask7`], storing the first two channels.
fn recompile_image_sample_lz_dmask3(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLz_Vdata2Vaddr3StSsDmask3";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_sampled_num > 0 && bind_info.samplers.samplers_num > 0 {
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                FUNC,
                MimgDescriptorClass::Sampled,
                true,
            )?;
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);

            if dst_value0.type_ != SpirvType::Float
                || src1_value0.type_ != SpirvType::Uint
                || src2_value0.type_ != SpirvType::Uint
            {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            const TEXT: &str = r#"
         %t24_<index> = OpLoad %uint %<src1_value0>
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %t24_<index>
         %t27_<index> = OpLoad %ImageS %t26_<index>
         %t33_<index> = OpLoad %uint %<src2_value0>
         %t35_<index> = OpAccessChain %_ptr_UniformConstant_Sampler %samplers %t33_<index>
         %t36_<index> = OpLoad %Sampler %t35_<index>
         %t38_<index> = OpSampledImage %SampledImage %t27_<index> %t36_<index>

<coord>
         %t43_<index> = OpImageSampleExplicitLod %v4float %t38_<index> %t42_<index> Lod %float_0_000000
               OpStore %temp_v4float %t43_<index>
         %t46_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_0
         %t47_<index> = OpLoad %float %t46_<index>
               OpStore %<dst_value0> %t47_<index>
         %t50_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_1
         %t51_<index> = OpLoad %float %t50_<index>
               OpStore %<dst_value1> %t51_<index>
"#;
            let (suffix, dim, class) = sampled_site_route(spirv, matched, FUNC)?;
            let coord = sample_coord_snippet(
                dim,
                inst.src[0],
                FUNC,
                sampled_site_is_guest_cube(spirv, matched),
            )?;
            let mut body = TEXT
                .replace("<coord>", &coord)
                .replace("<index>", &format!("{index}"))
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value);
            route_sampled_ids(&mut body, suffix);
            route_sampled_class(&mut body, class);
            *dst_source += &body;

            return Ok(true);
        }
    }

    Ok(false)
}

/// Beyond-Kyty: `image_sample_lz` with `dmask == 0x1` (.x only).
fn recompile_image_sample_lz_dmask1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLz_Vdata1Vaddr3StSsDmask1";
    image_sample_lz_single_channel(index, code, dst_source, spirv, FUNC, 0)
}

/// Beyond-Kyty: `image_sample_lz` with `dmask == 0x2` (.y only).
fn recompile_image_sample_lz_dmask2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLz_Vdata1Vaddr3StSsDmask2";
    image_sample_lz_single_channel(index, code, dst_source, spirv, FUNC, 1)
}

/// Beyond Kyty: `image_sample_lz` with `dmask == 0x8` (.w only).
fn recompile_image_sample_lz_dmask8(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLz_Vdata1Vaddr3StSsDmask8";
    image_sample_lz_single_channel(index, code, dst_source, spirv, FUNC, 3)
}

/// Beyond-Kyty (`image_gather4_lz` is `KYTY_NI` upstream): four-texel gather
/// of one channel — measured on ASTRO.BOT scene compute (raw 0xf11c0108,
/// dmask 0x1; dmask 0x2 on the later `rendering`-stage run).
/// `OpImageGather` samples the base level, and every uploaded image carries
/// exactly one mip, so the plain gather IS the LZ semantic. The four gathered
/// texels land in `vdata..vdata+3` in the hardware's (i0j1, i1j1, i1j0, i0j0)
/// order, which `OpImageGather` shares.
///
/// `component` is the dmask's single set bit's index, which is exactly
/// `OpImageGather`'s `Component` operand — the gather dmask selects the
/// channel sampled, not a subset of the destination registers (the
/// destination is always 4 dwords; see KytyPS5 `ImageOps.cpp`).
fn image_gather4_lz_component(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    component: u32,
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    let Some(bind_info) = spirv.get_bind_info() else {
        return Ok(false);
    };
    if bind_info.textures2d.textures2d_sampled_num == 0 || bind_info.samplers.samplers_num == 0 {
        return Ok(false);
    }
    let matched = mimg_descriptor_guard(
        index,
        spirv,
        code,
        &inst,
        func,
        MimgDescriptorClass::Sampled,
        true,
    )?;
    // Gathers are 2D-only in Vulkan (no 3D gather; cube gathers unmeasured).
    let (g4_suffix, g4_dim, g4_class) = sampled_site_route(spirv, matched, func)?;
    if g4_dim != SampledDim::Two {
        return Err(not_supported(func, "gather from a non-2D texture"));
    }

    let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
    let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
    let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
    let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);
    if src0_value0.type_ != SpirvType::Float
        || src0_value1.type_ != SpirvType::Float
        || src1_value0.type_ != SpirvType::Uint
        || src2_value0.type_ != SpirvType::Uint
    {
        return Err(not_supported(func, "unexpected operand types"));
    }

    const HEAD: &str = r#"
         %g4_t_<index> = OpLoad %uint %<src1_value0>
         %g4_ip_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %g4_t_<index>
         %g4_i_<index> = OpLoad %ImageS %g4_ip_<index>
         %g4_s_<index> = OpLoad %uint %<src2_value0>
         %g4_sp_<index> = OpAccessChain %_ptr_UniformConstant_Sampler %samplers %g4_s_<index>
         %g4_sa_<index> = OpLoad %Sampler %g4_sp_<index>
         %g4_si_<index> = OpSampledImage %SampledImage %g4_i_<index> %g4_sa_<index>
         %g4_c0_<index> = OpLoad %float %<src0_value0>
         %g4_c1_<index> = OpLoad %float %<src0_value1>
         %g4_c_<index> = OpCompositeConstruct %v2float %g4_c0_<index> %g4_c1_<index>
         %g4_r_<index> = OpImageGather %v4float %g4_si_<index> %g4_c_<index> %uint_<component>
               OpStore %temp_v4float %g4_r_<index>
"#;
    const TAIL: &str = r#"         %g4_p_<index>_<k> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_<k>
         %g4_v_<index>_<k> = OpLoad %float %g4_p_<index>_<k>
               OpStore %<dst_value> %g4_v_<index>_<k>
"#;

    let mut text = HEAD.replace("<component>", &format!("{component}"));
    for k in 0..4i32 {
        let dst_value = operand_variable_to_str_shift(inst.dst, k);
        if dst_value.type_ != SpirvType::Float {
            return Err(not_supported(func, "unexpected vdata type"));
        }
        text += &TAIL
            .replace("<k>", &format!("{k}"))
            .replace("<dst_value>", &dst_value.value);
    }

    let mut body = text
        .replace("<src0_value0>", &src0_value0.value)
        .replace("<src0_value1>", &src0_value1.value)
        .replace("<src1_value0>", &src1_value0.value)
        .replace("<src2_value0>", &src2_value0.value)
        .replace("<component>", &format!("{component}"))
        .replace("<index>", &format!("{index}"));
    route_sampled_ids(&mut body, g4_suffix);
    route_sampled_class(&mut body, g4_class);
    *dst_source += &body;

    Ok(true)
}

/// `image_gather4_lz` gathering channel X (dmask 0x1).
fn recompile_image_gather4_lz_dmask1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    image_gather4_lz_component(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_ImageGather4Lz_Vdata4Vaddr3StSsDmask1",
        0,
    )
}

/// `image_gather4_lz` gathering channel Y (dmask 0x2) — the measured
/// ASTRO.BOT blocker.
fn recompile_image_gather4_lz_dmask2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    image_gather4_lz_component(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_ImageGather4Lz_Vdata4Vaddr3StSsDmask2",
        1,
    )
}

/// `image_gather4_lz` gathering channel Z (dmask 0x4).
fn recompile_image_gather4_lz_dmask4(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    image_gather4_lz_component(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_ImageGather4Lz_Vdata4Vaddr3StSsDmask4",
        2,
    )
}

/// `image_gather4_lz` gathering channel W (dmask 0x8).
fn recompile_image_gather4_lz_dmask8(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    image_gather4_lz_component(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_ImageGather4Lz_Vdata4Vaddr3StSsDmask8",
        3,
    )
}

/// Kyty: `Recompile_ImageSampleLzO_Vdata3Vaddr4StSsDmask7` (ShaderSpirv.cpp
/// L2887).
#[allow(dead_code)] // C2: staged recompiler, not yet wired into G_RECOMP_FUNC
fn image_sample_lzo_channels(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    func: &'static str,
    channels: &[u32],
) -> Result<bool, ShaderRecompileError> {
    let inst = inst_at(code, index, func)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_sampled_num > 0 && bind_info.samplers.samplers_num > 0 {
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                func,
                MimgDescriptorClass::Sampled,
                true,
            )?;
            let src0_value0 = mimg_address_value(&inst, 0);
            let src0_value1 = mimg_address_value(&inst, 1);
            let src0_value2 = mimg_address_value(&inst, 2);
            let src1_value0 = operand_variable_to_str_shift(inst.src[1], 0);
            let src2_value0 = operand_variable_to_str_shift(inst.src[2], 0);

            if src0_value0.type_ != SpirvType::Float
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
"#;
            const TAIL: &str = r#"         %lzo_component_<index>_<slot> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_<channel>
         %lzo_value_<index>_<slot> = OpLoad %float %lzo_component_<index>_<slot>
               OpStore %<dst_value> %lzo_value_<index>_<slot>
"#;
            // The offset math builds a 2D coordinate and queries a v2int size,
            // so this body is 2D-only; a mixed shader whose LzO T# is 3D/Cube
            // is a named refusal rather than a wrong-Dim sample.
            let (suffix, dim, class) = sampled_site_route(spirv, matched, func)?;
            if dim != SampledDim::Two {
                return Err(not_supported(func, "offset sample of a non-2D texture"));
            }
            let mut text = HEAD.to_string();
            for (slot, channel) in channels.iter().copied().enumerate() {
                let dst_value = operand_variable_to_str_shift(inst.dst, slot as i32);
                if dst_value.type_ != SpirvType::Float {
                    return Err(not_supported(func, "unexpected destination type"));
                }
                text += &TAIL
                    .replace("<slot>", &slot.to_string())
                    .replace("<channel>", &channel.to_string())
                    .replace("<dst_value>", &dst_value.value);
            }
            let mut body = text
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src0_value2>", &src0_value2.value)
                .replace("<src1_value0>", &src1_value0.value)
                .replace("<src2_value0>", &src2_value0.value);
            route_sampled_ids(&mut body, suffix);
            route_sampled_class(&mut body, class);
            *dst_source += &body;

            return Ok(true);
        }
    }

    Ok(false)
}

fn recompile_image_sample_lzo_dmask1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLzO_Vdata1Vaddr4StSsDmask1";
    image_sample_lzo_channels(index, code, dst_source, spirv, FUNC, &[0])
}

fn recompile_image_sample_lzo_dmask2(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLzO_Vdata1Vaddr4StSsDmask2";
    image_sample_lzo_channels(index, code, dst_source, spirv, FUNC, &[1])
}

fn recompile_image_sample_lzo_dmask7(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageSampleLzO_Vdata3Vaddr4StSsDmask7";
    image_sample_lzo_channels(index, code, dst_source, spirv, FUNC, &[0, 1, 2])
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

    let matched = mimg_descriptor_guard(
        index,
        spirv,
        code,
        &inst,
        FUNC,
        MimgDescriptorClass::Sampled,
        false,
    )?;

    // `OpImageQuerySizeLod`'s result width is fixed by the image's dim — 2D
    // yields `%v2int`, while 2DArray and 3D yield `%v3int` (layer count or
    // depth in the third component). This dmask form only writes x and y, so
    // the extracts are the same for every dim; only the query's result TYPE
    // has to follow the descriptor, and emitting the wrong one is an invalid
    // module rather than a wrong number.
    //
    // This used to refuse anything but plain 2D, which failed the WHOLE shader
    // recompile (dropping the draw), not just the query — and it caught
    // 2DArray as well as 3D, i.e. every cube descriptor, since type 11/13
    // lower to 2DArray (`SampledDim::from_texture_type`). SharpEmu sizes the
    // query the same way (`Gen5SpirvTranslator.cs` @5228335:3296,3307,3320).
    // The numeric class routes only the array ids.
    let (resinfo_suffix, resinfo_dim, _) = sampled_site_route(spirv, matched, FUNC)?;

    let index_str = format!("{index}");
    const TEXT: &str = r#"
         %t0_<index> = OpLoad %uint %<texture>
         %t1_<index> = OpAccessChain %_ptr_UniformConstant_ImageS %textures2D_S %t0_<index>
         %t2_<index> = OpLoad %ImageS %t1_<index>
         %t3_<index> = OpLoad %float %<lod>
         %t4_<index> = OpBitcast %int %t3_<index>
         %t5_<index> = OpImageQuerySizeLod <size_ty> %t2_<index> %t4_<index>
         %t6_<index> = OpCompositeExtract %int %t5_<index> 0
         %t7_<index> = OpBitcast %float %t6_<index>
               OpStore %<dst_x> %t7_<index>
         %t8_<index> = OpCompositeExtract %int %t5_<index> 1
         %t9_<index> = OpBitcast %float %t8_<index>
               OpStore %<dst_y> %t9_<index>
"#;
    let mut body = TEXT
        .replace("<index>", &index_str)
        .replace("<size_ty>", resinfo_dim.query_size_type())
        .replace("<texture>", &texture.value)
        .replace("<lod>", &lod.value)
        .replace("<dst_x>", &dst_x.value)
        .replace("<dst_y>", &dst_y.value);
    route_sampled_ids(&mut body, resinfo_suffix);
    *dst_source += &body;
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
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                func,
                MimgDescriptorClass::Sampled,
                false,
            )?;
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let src0_value0 = mimg_address_value(&inst, 0);
            let src0_value1 = mimg_address_value(&inst, 1);
            // The third address component carries z (3D) or the array layer
            // (2DArray); NSA may name a non-consecutive VGPR.
            let src0_value2 = mimg_address_value(&inst, 2);
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
<coord>
         %t74_<index> = OpImageFetch %v4float %t27_<index> %t73_<index>
               OpStore %temp_v4float %t74_<index>
"#;
            const TAIL: &str = r#"         %t<t0>_<index> = OpAccessChain %_ptr_Function_float %temp_v4float %uint_<chan>
         %t<t1>_<index> = OpLoad %float %t<t0>_<index>
               OpStore %<dst_value> %t<t1>_<index>
"#;

            // OpImageFetch takes a 2-component integer coordinate for a 2D
            // texture, and a 3-component one for a 2DArray (x, y, layer) or a
            // 3D (x, y, z) texture — the third component from the MIMG
            // address's third VGPR. A Cube texture cannot be texel-fetched in
            // SPIR-V (it must be sampled), so that stays a named refusal.
            let (suffix, dim, class) = sampled_site_route(spirv, matched, func)?;
            let coord = match dim {
                SampledDim::Two => concat!(
                    "         %t73_<index> = ",
                    "OpCompositeConstruct %v2uint %t69_<index> %t71_<index>"
                )
                .to_string(),
                SampledDim::TwoArray | SampledDim::Three if inst.src[0].size >= 3 => {
                    format!(
                        "         %t72_<index> = OpLoad %float %{z}\n         \
                         %t72u_<index> = OpBitcast %uint %t72_<index>\n         \
                         %t73_<index> = OpCompositeConstruct %v3uint \
                         %t69_<index> %t71_<index> %t72u_<index>",
                        z = src0_value2.value
                    )
                }
                SampledDim::TwoArray | SampledDim::Three => concat!(
                    "         %t73_<index> = OpCompositeConstruct %v3uint ",
                    "%t69_<index> %t71_<index> %uint_0"
                )
                .to_string(),
                SampledDim::Cube => {
                    return Err(not_supported(func, "texel fetch of a cube texture"));
                }
            };
            let mut text = HEAD.replace("<coord>", &coord);
            for (i, (t0, chan)) in channels.iter().enumerate() {
                let dst_value = operand_variable_to_str_shift(inst.dst, i as i32);
                text += &TAIL
                    .replace("<t0>", &format!("{t0}"))
                    .replace("<t1>", &format!("{}", t0 + 1))
                    .replace("<chan>", &format!("{chan}"))
                    .replace("<dst_value>", &dst_value.value);
            }

            let mut body = text
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src1_value0>", &src1_value0.value);
            route_sampled_ids(&mut body, suffix);
            route_sampled_class(&mut body, class);
            *dst_source += &body;

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

/// Beyond-Kyty: `image_load` with `dmask == 0x3` (xy fetch), measured in
/// ASTRO.BOT scene compute shaders (MIMG 0x00 dmask 0x3).
fn recompile_image_load_dmask3(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageLoad_Vdata2Vaddr3StDmask3";
    image_load_channels(index, code, dst_source, spirv, FUNC, &[(46, 0), (50, 1)])
}

/// Beyond-Kyty: `image_load` with `dmask == 0xc` (ZW fetch), measured in
/// ASTRO.BOT scene compute. Enabled channels are packed into consecutive
/// destination VGPRs, so vdata receives texel.z followed by texel.w.
fn recompile_image_load_dmask_c(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageLoad_Vdata2Vaddr3StDmaskC";
    image_load_channels(index, code, dst_source, spirv, FUNC, &[(54, 2), (57, 3)])
}

/// The integer texel-coordinate construction (`%t73_<index>`) for a
/// storage-image (UAV) write: `v2uint (x, y)` for a 2D UAV, `v3uint
/// (x, y, z)` for a 3D one, and `v3uint (x, y, layer)` for a 2D array.
/// When the instruction's GFX10 DIM supplies only x/y but the descriptor view
/// is arrayed, z/layer is zero. The view's BASE_ARRAY selects the actual
/// subresource (Minecraft's panorama-copy shader). A genuine DIM_2D_ARRAY or
/// DIM_3D instruction carries the third address VGPR. The returned text still
/// carries `<index>` placeholders; substitute `<coord>` before `<index>`.
fn storage_image_coord_text(
    spirv: &Spirv<'_>,
    inst: &ShaderInstruction,
    matched: Option<usize>,
) -> Result<String, ShaderRecompileError> {
    let bind_info = spirv
        .get_bind_info()
        .expect("callers check textures2d_storage_num > 0");
    // The coordinate width follows the RESOLVED descriptor's own Dim (a
    // mixed shader writes 2D and 3D UAVs from one body shape), falling back
    // to the shader-wide key for fixtures without captured RW T# dwords.
    let (_, dim, _) = storage_site_route(spirv, matched, "storage_image_coord_text")?;
    let texture_type =
        matched.and_then(|i| bind_info.textures2d.desc.get(i).map(|d| d.texture.type_()));
    Ok(match (dim, texture_type) {
        // Type 8 is a true 1D image represented by a height-1 Vulkan 2D
        // image. Its MIMG address has only x semantics: synthesize y=0 rather
        // than reading the otherwise unrelated second Vaddr register.
        (SampledDim::Two, Some(8)) => {
            "         %t73_<index> = OpCompositeConstruct %v2uint %t69_<index> %uint_0".to_owned()
        }
        (SampledDim::TwoArray | SampledDim::Three, _) if inst.src[0].size >= 3 => {
            let src0_value1 = mimg_address_value(inst, 1);
            let src0_value2 = mimg_address_value(inst, 2);
            format!(
                "         %t70_<index> = OpLoad %float %{y}\n         \
                 %t71_<index> = OpBitcast %uint %t70_<index>\n         \
                 %t75_<index> = OpLoad %float %{z}\n         \
                 %t76_<index> = OpBitcast %uint %t75_<index>\n         \
                 %t73_<index> = OpCompositeConstruct %v3uint %t69_<index> %t71_<index> %t76_<index>",
                y = src0_value1.value,
                z = src0_value2.value
            )
        }
        (SampledDim::TwoArray | SampledDim::Three, _) => {
            let src0_value1 = mimg_address_value(inst, 1);
            format!(
                "         %t70_<index> = OpLoad %float %{y}\n         \
                 %t71_<index> = OpBitcast %uint %t70_<index>\n         \
                 %t73_<index> = OpCompositeConstruct %v3uint %t69_<index> %t71_<index> %uint_0",
                y = src0_value1.value
            )
        }
        _ => {
            let src0_value1 = mimg_address_value(inst, 1);
            format!(
                "         %t70_<index> = OpLoad %float %{y}\n         \
                 %t71_<index> = OpBitcast %uint %t70_<index>\n         \
                 %t73_<index> = OpCompositeConstruct %v2uint %t69_<index> %t71_<index>",
                y = src0_value1.value
            )
        }
    })
}

/// Return the descriptor-array index for the storage T# already resolved by
/// [`mimg_descriptor_guard`].
///
/// The guest SGPR still carries the rewritten descriptor dword 0 in the usual
/// case, but it is not an unambiguous source when a sampled and a storage T#
/// occupy the same register range. `WriteLocalVariables` must seed that shared
/// SGPR twice and the later descriptor wins. Minecraft's panorama copy shader
/// has exactly that shape at s24: the storage T# is followed in binding order
/// by a sampled T#, so loading s24 indexed a one-element `%textures2D_L` with
/// 1 and every image write was discarded. The guard has already proved which
/// captured descriptor this instruction uses, so select its class-local host
/// array slot directly.
fn storage_descriptor_index_constant(
    spirv: &Spirv<'_>,
    matched: Option<usize>,
    func: &'static str,
) -> Result<String, ShaderRecompileError> {
    let bind = spirv
        .get_bind_info()
        .ok_or_else(|| not_supported(func, "storage descriptor has no binding table"))?;
    let matched =
        matched.ok_or_else(|| not_supported(func, "storage descriptor was not resolved"))?;
    let texture_num = usize::try_from(bind.textures2d.textures_num.max(0))
        .unwrap_or(0)
        .min(bind.textures2d.desc.len());
    let Some(desc) = bind
        .textures2d
        .desc
        .get(matched)
        .filter(|_| matched < texture_num)
    else {
        return Err(not_supported(
            func,
            format!("storage descriptor index {matched} is outside the binding table"),
        ));
    };
    if !desc.textures2d_without_sampler {
        return Err(not_supported(
            func,
            format!("descriptor {matched} is sampled, not storage"),
        ));
    }
    // KEY-local, not class-local: each per-(Dim, format) SPIR-V array is
    // packed tight (`storage_key_layout` sizes it to its own key's count),
    // so the index counts only earlier storage descriptors of the SAME key.
    // A homogeneous shader has one key, making this identical to the old
    // storage-class-wide count.
    let key = storage_key_of(&desc.texture);
    let key_index = bind.textures2d.desc[..matched]
        .iter()
        .filter(|d| d.textures2d_without_sampler && storage_key_of(&d.texture) == key)
        .count() as u32;
    Ok(spirv.get_constant_uint(key_index))
}

/// Beyond-Kyty: `image_store` with `dmask == 0x1` (single channel), measured
/// in ASTRO.BOT scene compute (MIMG 0x08 dmask 0x1). `OpImageWrite` always
/// takes a 4-component texel; the storage image's own format decides which
/// components land. Disabled channels are zero; the dmask suppresses them and
/// does not synthesize alpha=1.
fn recompile_image_store_dmask1(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageStore_Vdata1Vaddr3StDmask1";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_storage_num > 0 {
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                FUNC,
                MimgDescriptorClass::Storage,
                false,
            )?;
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);

            if dst_value0.type_ != SpirvType::Float || src0_value0.type_ != SpirvType::Float {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() swizzle channels

            const TEXT: &str = r#"
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageL %textures2D_L %<storage_index>
         %t27_<index> = OpLoad %ImageL %t26_<index>
         %t67_<index> = OpLoad %float %<src0_value0>
         %t69_<index> = OpBitcast %uint %t67_<index>
<coord>
         %t84_<index> = OpLoad %float %<dst_value0>
         %t88_<index> = OpCompositeConstruct %v4float %t84_<index> %float_0_000000 %float_0_000000 %float_0_000000
               OpImageWrite %t27_<index> %t73_<index> %t88_<index>
"#;
            let coord = storage_image_coord_text(spirv, &inst, matched)?;
            let storage_index = storage_descriptor_index_constant(spirv, matched, FUNC)?;
            let (suffix, _, _) = storage_site_route(spirv, matched, FUNC)?;
            let mut body = TEXT
                .replace("<coord>", &coord)
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<storage_index>", &storage_index)
                .replace("<dst_value0>", &dst_value0.value);
            route_storage_ids(&mut body, suffix);
            *dst_source += &body;

            return Ok(true);
        }
    }

    Ok(false)
}

/// Beyond-Kyty: `image_store` with `dmask == 0x3` (two channels), measured in
/// ASTRO.BOT scene compute (MIMG 0x08 dmask 0x3). Same shape as the dmask1
/// body: `OpImageWrite` takes a full 4-component texel, and the disabled
/// channels are zero-filled.
fn recompile_image_store_dmask3(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_ImageStore_Vdata2Vaddr3StDmask3";
    let inst = inst_at(code, index, FUNC)?;

    if let Some(bind_info) = spirv.get_bind_info() {
        if bind_info.textures2d.textures2d_storage_num > 0 {
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                FUNC,
                MimgDescriptorClass::Storage,
                false,
            )?;
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);

            if dst_value0.type_ != SpirvType::Float || src0_value0.type_ != SpirvType::Float {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() swizzle channels

            const TEXT: &str = r#"
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageL %textures2D_L %<storage_index>
         %t27_<index> = OpLoad %ImageL %t26_<index>
         %t67_<index> = OpLoad %float %<src0_value0>
         %t69_<index> = OpBitcast %uint %t67_<index>
<coord>
         %t84_<index> = OpLoad %float %<dst_value0>
         %t85_<index> = OpLoad %float %<dst_value1>
         %t88_<index> = OpCompositeConstruct %v4float %t84_<index> %t85_<index> %float_0_000000 %float_0_000000
               OpImageWrite %t27_<index> %t73_<index> %t88_<index>
"#;
            let coord = storage_image_coord_text(spirv, &inst, matched)?;
            let storage_index = storage_descriptor_index_constant(spirv, matched, FUNC)?;
            let (suffix, _, _) = storage_site_route(spirv, matched, FUNC)?;
            let mut body = TEXT
                .replace("<coord>", &coord)
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<storage_index>", &storage_index)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value);
            route_storage_ids(&mut body, suffix);
            *dst_source += &body;

            return Ok(true);
        }
    }

    Ok(false)
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
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                FUNC,
                MimgDescriptorClass::Storage,
                false,
            )?;
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let dst_value3 = operand_variable_to_str_shift(inst.dst, 3);

            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);

            let src1_value2 = operand_variable_to_str_shift(inst.src[1], 2);

            if dst_value0.type_ != SpirvType::Float || src0_value0.type_ != SpirvType::Float {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check LOD_CLAMPED
            // TODO() swizzle channels
            // TODO() convert SRGB -> LINEAR if SRGB format was replaced with UNORM

            const TEXT: &str = r#"
         %t25_<index> = OpLoad %uint %<src1_value2>
		%t143_<index> = OpShiftRightLogical %uint %t25_<index> %uint_0
        %t145_<index> = OpBitwiseAnd %uint %t143_<index> %uint_0x00003fff
        %t146_<index> = OpIAdd %uint %t145_<index> %uint_1
        %t149_<index> = OpShiftRightLogical %uint %t25_<index> %uint_14
        %t150_<index> = OpBitwiseAnd %uint %t149_<index> %uint_0x00003fff
        %t151_<index> = OpIAdd %uint %t150_<index> %uint_1
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageL %textures2D_L %<storage_index>
         %t27_<index> = OpLoad %ImageL %t26_<index>
         %t67_<index> = OpLoad %float %<src0_value0>
         %t69_<index> = OpBitcast %uint %t67_<index>
<coord>
         %t84_<index> = OpLoad %float %<dst_value0>
         %t85_<index> = OpLoad %float %<dst_value1>
         %t86_<index> = OpLoad %float %<dst_value2>
         %t87_<index> = OpLoad %float %<dst_value3>
         %t88_<index> = OpCompositeConstruct %v4float %t84_<index> %t85_<index> %t86_<index> %t87_<index>
               OpImageWrite %t27_<index> %t73_<index> %t88_<index>
"#;
            let coord = storage_image_coord_text(spirv, &inst, matched)?;
            let storage_index = storage_descriptor_index_constant(spirv, matched, FUNC)?;
            let (suffix, _, _) = storage_site_route(spirv, matched, FUNC)?;
            let mut body = TEXT
                .replace("<coord>", &coord)
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<storage_index>", &storage_index)
                .replace("<src1_value2>", &src1_value2.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value)
                .replace("<dst_value3>", &dst_value3.value);
            route_storage_ids(&mut body, suffix);
            *dst_source += &body;

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
            let matched = mimg_descriptor_guard(
                index,
                spirv,
                code,
                &inst,
                FUNC,
                MimgDescriptorClass::Storage,
                false,
            )?;
            let dst_value0 = operand_variable_to_str_shift(inst.dst, 0);
            let dst_value1 = operand_variable_to_str_shift(inst.dst, 1);
            let dst_value2 = operand_variable_to_str_shift(inst.dst, 2);
            let dst_value3 = operand_variable_to_str_shift(inst.dst, 3);

            let src0_value0 = operand_variable_to_str_shift(inst.src[0], 0);
            let src0_value1 = operand_variable_to_str_shift(inst.src[0], 1);
            let src0_value2 = operand_variable_to_str_shift(inst.src[0], 2);

            let src1_value2 = operand_variable_to_str_shift(inst.src[1], 2);

            if dst_value0.type_ != SpirvType::Float || src0_value0.type_ != SpirvType::Float {
                return Err(not_supported(FUNC, "unexpected operand types"));
            }

            // TODO() check VSKIP
            // TODO() check LOD_CLAMPED
            // TODO() swizzle channels
            // TODO() convert SRGB -> LINEAR if SRGB format was replaced with UNORM

            const TEXT: &str = r#"
         %t25_<index> = OpLoad %uint %<src1_value2>
		%t143_<index> = OpShiftRightLogical %uint %t25_<index> %uint_0
        %t145_<index> = OpBitwiseAnd %uint %t143_<index> %uint_0x00003fff
        %t146_<index> = OpIAdd %uint %t145_<index> %uint_1
        %t149_<index> = OpShiftRightLogical %uint %t25_<index> %uint_14
        %t150_<index> = OpBitwiseAnd %uint %t149_<index> %uint_0x00003fff
        %t151_<index> = OpIAdd %uint %t150_<index> %uint_1
         %t26_<index> = OpAccessChain %_ptr_UniformConstant_ImageL %textures2D_L %<storage_index>
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
            let storage_index = storage_descriptor_index_constant(spirv, matched, FUNC)?;
            let (suffix, _, _) = storage_site_route(spirv, matched, FUNC)?;
            let mut body = TEXT
                .replace("<index>", &format!("{index}"))
                .replace("<src0_value0>", &src0_value0.value)
                .replace("<src0_value1>", &src0_value1.value)
                .replace("<src0_value2>", &src0_value2.value)
                .replace("<storage_index>", &storage_index)
                .replace("<src1_value2>", &src1_value2.value)
                .replace("<dst_value0>", &dst_value0.value)
                .replace("<dst_value1>", &dst_value1.value)
                .replace("<dst_value2>", &dst_value2.value)
                .replace("<dst_value3>", &dst_value3.value);
            route_storage_ids(&mut body, suffix);
            *dst_source += &body;

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

/// Beyond-Kyty: `s_andn1_saveexec_b64` — `sdst = exec; exec = ~ssrc0 & exec`.
/// The ANDN1 sibling of [`recompile_s_orn2_saveexec_b64`] (negates the first
/// operand instead of the second, ANDing rather than ORing). Measured in
/// ASTRO.BOT's scene-composite compute shader.
fn recompile_s_andn1_saveexec_b64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_SAndn1SaveexecB64_Sdst2Ssrc02";
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
        %t192_<index> = OpNot %uint %t0_<index>
        %t193_<index> = OpNot %uint %t1_<index>
        %t194_<index> = OpBitwiseAnd %uint %t192_<index> %t190_<index>
               OpStore %exec_lo %t194_<index>
        %t197_<index> = OpBitwiseAnd %uint %t193_<index> %t191_<index>
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

/// Beyond-Kyty: `v_add_co_ci_u32` — add with carry-in and carry-out.
/// `vdst = src0 + src1 + carry_in; carry_out -> sdst`. Both the plain VOP2 form
/// (carry in/out via VCC) and the VOP3B form (carry-in = src2, carry-out =
/// sdst) decode to this one recompiler; the parser wires `dst2`/`src[2]` to VCC
/// or to the encoded SGPR pair respectively. Follows SharpEmu's `EmitAddWithCarry`
/// (Gen5SpirvTranslator.Alu.cs L3396): the carry-out is the unsigned overflow of
/// the two-step add, `(partial <u src0) || (result <u partial)`. In this
/// single-lane model the carry mask holds the lane bit in bit 0 (matching the
/// compare recompilers), so carry-in is `src2 & 1` and carry-out is written as
/// 1/0 to the low mask dword with 0 in the high dword.
fn recompile_v_add_co_ci_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VAddCoCiU32_VdstSdst2Vsrc0Vsrc1Smask2";
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

    if !operand_is_variable(inst.dst2) {
        return Err(not_supported(FUNC, "dst2 (carry-out) is not a variable"));
    }
    let carry_out0 = operand_variable_to_str_shift(inst.dst2, 0);
    let carry_out1 = operand_variable_to_str_shift(inst.dst2, 1);
    if carry_out0.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "carry-out is not uint"));
    }
    if operand_is_exec(inst.dst2) {
        return Err(not_supported(FUNC, "exec carry-out"));
    }

    if !operand_is_variable(inst.src[2]) {
        return Err(not_supported(FUNC, "carry-in is not a variable"));
    }
    let carry_in = operand_variable_to_str_shift(inst.src[2], 0);
    if carry_in.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "carry-in is not uint"));
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
        %tci_raw_<index> = OpLoad %uint %<carryin>
        %tci_bit_<index> = OpBitwiseAnd %uint %tci_raw_<index> %uint_1
        %tpartial_<index> = OpIAdd %uint %t0_<index> %t1_<index>
        %tsum_<index> = OpIAdd %uint %tpartial_<index> %tci_bit_<index>
        %tc0_<index> = OpULessThan %bool %tpartial_<index> %t0_<index>
        %tc1_<index> = OpULessThan %bool %tsum_<index> %tpartial_<index>
        %tcarry_<index> = OpLogicalOr %bool %tc0_<index> %tc1_<index>
        %tcarry_u_<index> = OpSelect %uint %tcarry_<index> %uint_1 %uint_0
               OpStore %<carryout0> %tcarry_u_<index>
               OpStore %<carryout1> %uint_0
        %tsumf_<index> = OpBitcast %float %tsum_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %tsumf_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;

    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<carryin>", &carry_in.value)
        .replace("<carryout0>", &carry_out0.value)
        .replace("<carryout1>", &carry_out1.value)
        .replace("<dst>", &dst_value.value)
        .replace("<index>", &index_str);

    Ok(true)
}

/// RDNA2 compact VOP2 subtract-with-borrow family. The parser supplies VCC as
/// both `src[2]` (borrow in) and `dst2` (borrow out). This follows SharpEmu's
/// clean-room `EmitSubtractWithBorrow`: subtract in two steps and report a
/// borrow when either step underflows.
fn recompile_v_sub_borrow_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    reverse: bool,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VSubBorrowU32_VdstSdst2Vsrc0Vsrc1Smask2";
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

    if !operand_is_variable(inst.dst2) {
        return Err(not_supported(FUNC, "dst2 (borrow-out) is not a variable"));
    }
    let borrow_out0 = operand_variable_to_str_shift(inst.dst2, 0);
    let borrow_out1 = operand_variable_to_str_shift(inst.dst2, 1);
    if borrow_out0.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "borrow-out is not uint"));
    }
    if operand_is_exec(inst.dst2) {
        return Err(not_supported(FUNC, "exec borrow-out"));
    }

    if !operand_is_variable(inst.src[2]) {
        return Err(not_supported(FUNC, "borrow-in is not a variable"));
    }
    let borrow_in = operand_variable_to_str_shift(inst.src[2], 0);
    if borrow_in.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "borrow-in is not uint"));
    }

    let (left, right) = if reverse {
        (inst.src[1], inst.src[0])
    } else {
        (inst.src[0], inst.src[1])
    };
    let mut load_left = String::new();
    let mut load_right = String::new();
    if !operand_load_uint(spirv, left, "tleft_<index>", &index_str, &mut load_left, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(
        spirv,
        right,
        "tright_<index>",
        &index_str,
        &mut load_right,
        -1,
    )? {
        return Ok(false);
    }

    const TEXT: &str = r#"
        <load_left>
        <load_right>
        %tbin_raw_<index> = OpLoad %uint %<borrowin>
        %tbin_<index> = OpBitwiseAnd %uint %tbin_raw_<index> %uint_1
        %tpartial_<index> = OpISub %uint %tleft_<index> %tright_<index>
        %tresult_<index> = OpISub %uint %tpartial_<index> %tbin_<index>
        %tb0_<index> = OpULessThan %bool %tleft_<index> %tright_<index>
        %tb1_<index> = OpULessThan %bool %tpartial_<index> %tbin_<index>
        %tborrow_<index> = OpLogicalOr %bool %tb0_<index> %tb1_<index>
        %tborrow_u_<index> = OpSelect %uint %tborrow_<index> %uint_1 %uint_0
               OpStore %<borrowout0> %tborrow_u_<index>
               OpStore %<borrowout1> %uint_0
        %tresult_f_<index> = OpBitcast %float %tresult_<index>
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %tresult_f_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;

    *dst_source += &TEXT
        .replace("<load_left>", &load_left)
        .replace("<load_right>", &load_right)
        .replace("<borrowin>", &borrow_in.value)
        .replace("<borrowout0>", &borrow_out0.value)
        .replace("<borrowout1>", &borrow_out1.value)
        .replace("<dst>", &dst_value.value)
        .replace("<index>", &index_str);

    Ok(true)
}

fn recompile_v_subb_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    recompile_v_sub_borrow_u32(index, code, dst_source, spirv, false)
}

fn recompile_v_subbrev_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    recompile_v_sub_borrow_u32(index, code, dst_source, spirv, true)
}

/// Beyond-Kyty: `v_mad_u64_u32` — widening multiply-accumulate
/// `vdst.u64 = src0.u32 * src1.u32 + src2.u64`, carry-out of the 64-bit add ->
/// sdst. Built from the existing `mul_lo_uint`/`mul_hi_uint` helpers (the 64-bit
/// product) and two `addc` calls (the 64-bit add with inter-dword carry), the
/// same primitives `s_addc_u32` uses. The destination and src2 are 32-bit VGPR
/// pairs (float-typed dwords, so the uint result is bitcast before storing); the
/// carry-out sdst is written only when it decodes to a real SGPR pair (most
/// callers leave it unused). Measured in ASTRO.BOT's scene-composite compute
/// shader (0x555f4f500).
fn recompile_v_mad_u64_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VMadU64U32_VdstSdst2Vsrc0Vsrc1Smask2";
    let inst = inst_at(code, index, FUNC)?;
    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }
    // The 64-bit result lands in two VGPR dwords, each float-typed in this
    // model (so the uint sum is bitcast to float before the store).
    let dst0 = operand_variable_to_str_shift(inst.dst, 0);
    let dst1 = operand_variable_to_str_shift(inst.dst, 1);
    if dst0.type_ != SpirvType::Float || dst1.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "dst is not a VGPR pair"));
    }

    let mut load0 = String::new();
    let mut load1 = String::new();
    let mut load2lo = String::new();
    let mut load2hi = String::new();
    // src0/src1 are 32-bit (shift -1 = whole operand); src2 is the 64-bit
    // addend (dword 0 and dword 1).
    if !operand_load_uint(spirv, inst.src[0], "s0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "s1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(
        spirv,
        inst.src[2],
        "s2lo_<index>",
        &index_str,
        &mut load2lo,
        0,
    )? {
        return Ok(false);
    }
    if !operand_load_uint(
        spirv,
        inst.src[2],
        "s2hi_<index>",
        &index_str,
        &mut load2hi,
        1,
    )? {
        return Ok(false);
    }

    // Write the carry-out only when sdst is a real, uint-typed SGPR pair. A
    // v_mad_u64_u32 whose sdst is left off (the common case) simply drops it.
    let carry_store = if operand_is_variable(inst.dst2) {
        let co0 = operand_variable_to_str_shift(inst.dst2, 0);
        let co1 = operand_variable_to_str_shift(inst.dst2, 1);
        if co0.type_ == SpirvType::Uint && !operand_is_exec(inst.dst2) {
            format!(
                "               OpStore %{co0} %tc1_<index>\n               OpStore %{co1} %uint_0",
                co0 = co0.value,
                co1 = co1.value,
            )
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // The 64-bit add is done dword-by-dword with an explicit unsigned-overflow
    // carry (`OpULessThan`, as `v_add_co_ci_u32` does) rather than the `%addc`
    // helper — self-contained and free of `OpIAddCarry`, which the runtime
    // driver accepts but the test-time validator does not. High-dword overflow
    // (the carry-out) can come from either the phi+s2hi add or the +carry add.
    const TEXT: &str = r#"
        <load0>
        <load1>
        <load2lo>
        <load2hi>
        %tplo_<index> = OpFunctionCall %uint %mul_lo_uint %s0_<index> %s1_<index>
        %tphi_<index> = OpFunctionCall %uint %mul_hi_uint %s0_<index> %s1_<index>
        %trlo_<index> = OpIAdd %uint %tplo_<index> %s2lo_<index>
        %tc0b_<index> = OpULessThan %bool %trlo_<index> %tplo_<index>
        %tc0_<index> = OpSelect %uint %tc0b_<index> %uint_1 %uint_0
        %tmid_<index> = OpIAdd %uint %tphi_<index> %s2hi_<index>
        %tcmid_<index> = OpULessThan %bool %tmid_<index> %tphi_<index>
        %trhi_<index> = OpIAdd %uint %tmid_<index> %tc0_<index>
        %tctop_<index> = OpULessThan %bool %trhi_<index> %tmid_<index>
        %tc1b_<index> = OpLogicalOr %bool %tcmid_<index> %tctop_<index>
        %tc1_<index> = OpSelect %uint %tc1b_<index> %uint_1 %uint_0
        %trlof_<index> = OpBitcast %float %trlo_<index>
               OpStore %<dst0> %trlof_<index>
        %trhif_<index> = OpBitcast %float %trhi_<index>
               OpStore %<dst1> %trhif_<index>
<carry>
"#;

    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<load2lo>", &load2lo)
        .replace("<load2hi>", &load2hi)
        .replace("<carry>", &carry_store)
        .replace("<dst0>", &dst0.value)
        .replace("<dst1>", &dst1.value)
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
    sbuffer_load_dwords(
        index,
        code,
        dst_source,
        spirv,
        "Recompile_SBufferLoadDwordx2_Sdst2SvSoffset",
        2,
        "sbuffer_load_dword_2",
    )
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
    sbuffer_load_dwords(
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
    sbuffer_load_dwords(
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

/// GFX10 `v_cmp_gt_u64`: compare the high dwords first, then the low dwords
/// when the high halves are equal. SPIR-V Int64 is intentionally unnecessary
/// here, keeping the generated module valid on devices without shaderInt64.
fn recompile_vcmp_gt_u64(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VCmpGtU64_SmaskVsrc02Vsrc12";
    let inst = inst_at(code, index, FUNC)?;
    if !operand_is_variable(inst.dst) || operand_is_exec(inst.dst) {
        return Err(not_supported(FUNC, "non-scalar-mask destination"));
    }

    let dst_lo = operand_variable_to_str_shift(inst.dst, 0);
    let dst_hi = operand_variable_to_str_shift(inst.dst, 1);
    if dst_lo.type_ != SpirvType::Uint || dst_hi.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "destination is not a uint pair"));
    }

    let index_str = format!("{index}");
    let mut lhs_lo = String::new();
    let mut lhs_hi = String::new();
    let mut rhs_lo = String::new();
    let mut rhs_hi = String::new();
    if !operand_load_uint(
        spirv,
        inst.src[0],
        "u64_lhs_lo_<index>",
        &index_str,
        &mut lhs_lo,
        0,
    )? || !operand_load_uint(
        spirv,
        inst.src[0],
        "u64_lhs_hi_<index>",
        &index_str,
        &mut lhs_hi,
        1,
    )? || !operand_load_uint(
        spirv,
        inst.src[1],
        "u64_rhs_lo_<index>",
        &index_str,
        &mut rhs_lo,
        0,
    )? || !operand_load_uint(
        spirv,
        inst.src[1],
        "u64_rhs_hi_<index>",
        &index_str,
        &mut rhs_hi,
        1,
    )? {
        return Ok(false);
    }

    const TEXT: &str = r#"
          <lhs_lo>
          <lhs_hi>
          <rhs_lo>
          <rhs_hi>
          %u64_hi_gt_<index> = OpUGreaterThan %bool %u64_lhs_hi_<index> %u64_rhs_hi_<index>
          %u64_hi_eq_<index> = OpIEqual %bool %u64_lhs_hi_<index> %u64_rhs_hi_<index>
          %u64_lo_gt_<index> = OpUGreaterThan %bool %u64_lhs_lo_<index> %u64_rhs_lo_<index>
          %u64_eq_and_lo_gt_<index> = OpLogicalAnd %bool %u64_hi_eq_<index> %u64_lo_gt_<index>
          %u64_gt_<index> = OpLogicalOr %bool %u64_hi_gt_<index> %u64_eq_and_lo_gt_<index>
          %u64_mask_<index> = OpSelect %uint %u64_gt_<index> %uint_1 %uint_0
          OpStore %<dst_lo> %u64_mask_<index>
          OpStore %<dst_hi> %uint_0
"#;
    *dst_source += &TEXT
        .replace("<lhs_lo>", &lhs_lo)
        .replace("<lhs_hi>", &lhs_hi)
        .replace("<rhs_lo>", &rhs_lo)
        .replace("<rhs_hi>", &rhs_hi)
        .replace("<dst_lo>", &dst_lo.value)
        .replace("<dst_hi>", &dst_hi.value)
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
    let exec_only = operand_is_exec(inst.dst);

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_int(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_int(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }

    // GFX10 VCMPX is EXEC-only; legacy encodings also write their scalar
    // destination. In either case an already-disabled lane must stay disabled,
    // so intersect the comparison with the current EXEC value.
    let destination = if exec_only {
        String::new()
    } else {
        format!(
            "          OpStore %{} %t6_<index>\n          OpStore %{} %uint_0",
            dst_value0.value, dst_value1.value
        )
    };

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpLoad %uint %exec_lo
          %t4_<index> = OpINotEqual %bool %t3_<index> %uint_0
          %t5_<index> = OpLogicalAnd %bool %t2_<index> %t4_<index>
          %t6_<index> = OpSelect %uint %t5_<index> %uint_1 %uint_0
          <destination>
          OpStore %exec_lo %t6_<index>
          OpStore %exec_hi %uint_0
          <execz>
"#;
    *dst_source += &TEXT
        .replace("<destination>", &destination)
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
    let exec_only = operand_is_exec(inst.dst);

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_uint(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0, -1)? {
        return Ok(false);
    }
    if !operand_load_uint(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1, -1)? {
        return Ok(false);
    }

    // GFX10 VCMPX is EXEC-only; legacy encodings also write their scalar
    // destination. Preserve inactive lanes across consecutive comparisons.
    let destination = if exec_only {
        String::new()
    } else {
        format!(
            "          OpStore %{} %t6_<index>\n          OpStore %{} %uint_0",
            dst_value0.value, dst_value1.value
        )
    };

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpLoad %uint %exec_lo
          %t4_<index> = OpINotEqual %bool %t3_<index> %uint_0
          %t5_<index> = OpLogicalAnd %bool %t2_<index> %t4_<index>
          %t6_<index> = OpSelect %uint %t5_<index> %uint_1 %uint_0
          <destination>
          OpStore %exec_lo %t6_<index>
          OpStore %exec_hi %uint_0
          <execz>
"#;
    *dst_source += &TEXT
        .replace("<destination>", &destination)
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
    let exec_only = operand_is_exec(inst.dst);

    let mut load0 = String::new();
    let mut load1 = String::new();

    if !operand_load_float(spirv, inst.src[0], "t0_<index>", &index_str, &mut load0)? {
        return Ok(false);
    }
    if !operand_load_float(spirv, inst.src[1], "t1_<index>", &index_str, &mut load1)? {
        return Ok(false);
    }

    // GFX10 VCMPX is EXEC-only; legacy encodings also write their scalar
    // destination. Preserve inactive lanes across consecutive comparisons.
    let destination = if exec_only {
        String::new()
    } else {
        format!(
            "          OpStore %{} %t6_<index>\n          OpStore %{} %uint_0",
            dst_value0.value, dst_value1.value
        )
    };

    const TEXT: &str = r#"
          <load0>
          <load1>
          %t2_<index> = <param> %bool %t0_<index> %t1_<index>
          %t3_<index> = OpLoad %uint %exec_lo
          %t4_<index> = OpINotEqual %bool %t3_<index> %uint_0
          %t5_<index> = OpLogicalAnd %bool %t2_<index> %t4_<index>
          %t6_<index> = OpSelect %uint %t5_<index> %uint_1 %uint_0
          <destination>
          OpStore %exec_lo %t6_<index>
          OpStore %exec_hi %uint_0
          <execz>
"#;
    *dst_source += &TEXT
        .replace("<destination>", &destination)
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

// ---------------------------------------------------------------------------
// VOP3P: packed 16-bit math and the mix ops
//
// Beyond Kyty; ported from SharpEmu's `Gen5SpirvTranslator.Alu.cs`
// (`TryEmitPackedF16` / `TryEmitFmaMix`, PRs #466 `3574a3b`, #460 `472fc96`,
// #420 `3005bab`). See `shader_parse_vop3p` for the decode side.
//
// A guest VGPR holds two f16 values, one per result lane. Each lane is
// computed independently: the selected source half is widened exactly to f32,
// `neg_lo`/`neg_hi` negates it, the op runs in f32, and the f32 result is
// narrowed back to f16. For add/mul/min/max that is bit-exact to a true f16
// op (an f16 sum or product rounds losslessly through f32 by the
// double-rounding theorem, and min/max carry no rounding at all).
//
// DIVERGENCE from SharpEmu, deliberate and named:
//
//  * The f16<->f32 conversions use GLSL `UnpackHalf2x16` / `PackHalf2x16`,
//    matching this crate's existing `VCvtF32F16` / `VCvtPkrtzF16F32` bodies,
//    rather than SharpEmu's explicit branchless integer sequences. Those exist
//    to pin subnormal/rounding behaviour without float-controls execution
//    modes; the GLSL ops leave that to the driver.
//  * `v_pk_fma_f16` lowers to a single f32 `Fma` followed by the f16 narrowing,
//    NOT to SharpEmu's round-to-odd 2Sum sequence (PR #420 "exact single
//    rounding"). That sequence is only error-free when every op in the chain
//    carries `OpDecoration NoContraction`, and this generator emits
//    decorations in a separate `write_annotations` phase with no per-body
//    injection point — an uncorrected 2Sum measurably decays to the
//    double-rounded answer anyway (SharpEmu observed exactly that on RDNA3).
//    So the result can differ from hardware in the last f16 bit on midpoint
//    inputs. Getting the shader to TRANSLATE at all is the win here: it was
//    previously dropped whole.
// ---------------------------------------------------------------------------

/// Read the VOP3P control block, or refuse by name if the parser did not
/// attach one (a table row wired to the wrong instruction type).
fn vop3p_control(
    inst: &ShaderInstruction,
    func: &'static str,
) -> Result<crate::shader::types::Vop3pControl, ShaderRecompileError> {
    inst.vop3p
        .ok_or_else(|| not_supported(func, "missing vop3p control"))
}

/// Emit the load of source `i`'s raw 32 bits as `%<name>`.
fn vop3p_load_raw(
    spirv: &Spirv<'_>,
    inst: &ShaderInstruction,
    i: usize,
    name: &str,
    index_str: &str,
    out: &mut String,
) -> Result<bool, ShaderRecompileError> {
    let mut load = String::new();
    if !operand_load_uint(spirv, inst.src[i], name, index_str, &mut load, 0)? {
        return Ok(false);
    }
    out.push_str("         ");
    out.push_str(&load);
    out.push('\n');
    Ok(true)
}

/// Saturate an f32 to `[0, 1]` the way the VOP3P `clamp` modifier does: below
/// 0 becomes 0, above 1 becomes 1, and NaN becomes 0 (the ORDERED compares are
/// false for it, which is the hardware's NaN-to-zero behaviour without a
/// separate `IsNan` test). SharpEmu: `EmitClampToUnitInterval`.
fn vop3p_clamp(value: &str, out_id: &str, body: &mut String) {
    body.push_str(&format!(
        "         %{out_id}_gt0 = OpFOrdGreaterThan %bool %{value} %float_0_000000\n\
         \x20        %{out_id}_lo = OpSelect %float %{out_id}_gt0 %{value} %float_0_000000\n\
         \x20        %{out_id}_lt1 = OpFOrdLessThan %bool %{out_id}_lo %float_1_000000\n\
         \x20        %{out_id} = OpSelect %float %{out_id}_lt1 %{out_id}_lo %float_1_000000\n"
    ));
}

/// Widen one 16-bit half of `raw` to f32 and apply this lane's negate bit.
/// `hi` selects the half (`op_sel` / `op_sel_hi` bit `i`), `neg` this lane's
/// negate bit (`neg_lo` / `neg_hi` bit `i`). SharpEmu:
/// `EmitPackedF16Operand`.
fn vop3p_half_operand(raw: &str, hi: bool, neg: bool, out_id: &str, body: &mut String) {
    let component = u32::from(hi);
    body.push_str(&format!(
        "         %{out_id}_pk = OpExtInst %v2float %GLSL_std_450 UnpackHalf2x16 %{raw}\n\
         \x20        %{out_id}{suffix} = OpCompositeExtract %float %{out_id}_pk {component}\n",
        suffix = if neg { "_p" } else { "" }
    ));
    if neg {
        body.push_str(&format!(
            "         %{out_id} = OpFNegate %float %{out_id}_p\n"
        ));
    }
}

/// `fminnum_like` / `fmaxnum_like`: a NaN operand yields the other; two NaNs
/// yield a NaN. SharpEmu: `EmitPackedF16MinMax`.
fn vop3p_min_max(left: &str, right: &str, is_max: bool, out_id: &str, body: &mut String) {
    let cmp = if is_max {
        "OpFOrdGreaterThan"
    } else {
        "OpFOrdLessThan"
    };
    body.push_str(&format!(
        "         %{out_id}_c = {cmp} %bool %{left} %{right}\n\
         \x20        %{out_id}_n = OpSelect %float %{out_id}_c %{left} %{right}\n\
         \x20        %{out_id}_ln = OpIsNan %bool %{left}\n\
         \x20        %{out_id}_rn = OpIsNan %bool %{right}\n\
         \x20        %{out_id}_wr = OpSelect %float %{out_id}_rn %{left} %{out_id}_n\n\
         \x20        %{out_id} = OpSelect %float %{out_id}_ln %{right} %{out_id}_wr\n"
    ));
}

/// The exec-predicated write of an f32-valued result into a VGPR, matching
/// every other VOP body's tail (`Recompile_VCvtPkrtzF16F32_*`).
fn vop3p_store(dst: &str, value: &str, index_str: &str, body: &mut String) {
    body.push_str(
        &"         %exec_lo_u_<index> = OpLoad %uint %exec_lo\n\
          \x20        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0\n\
          \x20        %tdst_<index> = OpLoad %float %<dst>\n\
          \x20        %tval_<index> = OpSelect %float %exec_lo_b_<index> %<value> %tdst_<index>\n\
          \x20              OpStore %<dst> %tval_<index>\n"
            .replace("<dst>", dst)
            .replace("<value>", value)
            .replace("<index>", index_str),
    );
}

/// `v_pk_add_f16` / `v_pk_mul_f16` / `v_pk_min_f16` / `v_pk_max_f16` /
/// `v_pk_fma_f16`. SharpEmu: `TryEmitPackedF16`.
fn recompile_vop3p_packed_f16(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Vop3pPackedF16_VdstVsrc0Vsrc1Vsrc2";
    let inst = inst_at(code, index, FUNC)?;
    let ctrl = vop3p_control(&inst, FUNC)?;
    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }
    let dst_value = operand_variable_to_str(inst.dst);
    if dst_value.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "dst is not a float VGPR"));
    }

    let fused = inst.type_ == ShaderInstructionType::VPkFmaF16;
    let src_count = if fused { 3 } else { 2 };

    let mut body = String::new();
    for i in 0..src_count {
        if !vop3p_load_raw(
            spirv,
            &inst,
            i,
            &format!("r{i}_{index}"),
            &index_str,
            &mut body,
        )? {
            return Ok(false);
        }
    }

    // One result lane at a time (low, then high); `lanes` collects the id
    // holding each lane's final f32.
    let mut lanes: Vec<String> = Vec::with_capacity(2);
    for (lane, hi) in [("lo", false), ("hi", true)] {
        let sel = if hi { ctrl.op_sel_hi } else { ctrl.op_sel };
        let neg = if hi { ctrl.neg_hi } else { ctrl.neg_lo };
        for i in 0..src_count {
            vop3p_half_operand(
                &format!("r{i}_{index}"),
                (sel >> i) & 1 != 0,
                (neg >> i) & 1 != 0,
                &format!("h{lane}{i}_{index}"),
                &mut body,
            );
        }
        let a = format!("h{lane}0_{index}");
        let b = format!("h{lane}1_{index}");
        let raw = format!("v{lane}raw_{index}");
        match inst.type_ {
            ShaderInstructionType::VPkAddF16 => {
                body.push_str(&format!("         %{raw} = OpFAdd %float %{a} %{b}\n"))
            }
            ShaderInstructionType::VPkMulF16 => {
                body.push_str(&format!("         %{raw} = OpFMul %float %{a} %{b}\n"))
            }
            ShaderInstructionType::VPkMinF16 => vop3p_min_max(&a, &b, false, &raw, &mut body),
            ShaderInstructionType::VPkMaxF16 => vop3p_min_max(&a, &b, true, &raw, &mut body),
            ShaderInstructionType::VPkFmaF16 => body.push_str(&format!(
                "         %{raw} = OpExtInst %float %GLSL_std_450 Fma %{a} %{b} %h{lane}2_{index}\n"
            )),
            other => {
                return Err(not_supported(
                    FUNC,
                    format!("instruction type {other:?} is not a packed f16 op"),
                ));
            }
        }
        // No `OpCopyObject` (this crate's assembler has no such opcode): the
        // clamped and unclamped ids are tracked here instead of aliased in
        // SPIR-V.
        if ctrl.clamp {
            vop3p_clamp(&raw, &format!("v{lane}_{index}"), &mut body);
            lanes.push(format!("v{lane}_{index}"));
        } else {
            lanes.push(raw);
        }
    }

    body.push_str(&format!(
        "         %vpk_{index} = OpCompositeConstruct %v2float %{lo} %{hi}\n\
         \x20        %vpu_{index} = OpExtInst %uint %GLSL_std_450 PackHalf2x16 %vpk_{index}\n\
         \x20        %vpf_{index} = OpBitcast %float %vpu_{index}\n",
        lo = lanes[0],
        hi = lanes[1]
    ));
    vop3p_store(
        &dst_value.value,
        &format!("vpf_{index}"),
        &index_str,
        &mut body,
    );

    *dst_source += "\n";
    *dst_source += &body;
    Ok(true)
}

/// `v_fma_mix_f32` / `v_fma_mixlo_f16` / `v_fma_mixhi_f16`. SharpEmu:
/// `TryEmitFmaMix` + `EmitFmaMixOperand`.
///
/// Each source is read independently as either a full f32 or one f16 half
/// widened to f32: `op_sel_hi` bit `i` set means "read as f16" (with `op_sel`
/// bit `i` picking the half). For these ops `neg_hi` is the ABSOLUTE-value
/// modifier and `neg_lo` negates, applied abs-then-neg.
fn recompile_vop3p_fma_mix(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_Vop3pFmaMix_VdstVsrc0Vsrc1Vsrc2";
    let inst = inst_at(code, index, FUNC)?;
    let ctrl = vop3p_control(&inst, FUNC)?;
    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) {
        return Err(not_supported(FUNC, "dst is not a variable"));
    }
    let dst_value = operand_variable_to_str(inst.dst);
    if dst_value.type_ != SpirvType::Float {
        return Err(not_supported(FUNC, "dst is not a float VGPR"));
    }

    let mut body = String::new();
    // The id holding each source's post-modifier f32.
    let mut srcs: Vec<String> = Vec::with_capacity(3);
    for i in 0..3usize {
        if !vop3p_load_raw(
            spirv,
            &inst,
            i,
            &format!("r{i}_{index}"),
            &index_str,
            &mut body,
        )? {
            return Ok(false);
        }
        let raw = format!("r{i}_{index}");
        let val = format!("m{i}_{index}");
        // A register/SGPR source with op_sel_hi set is read as an f16 half;
        // an inline/literal constant is always the full f32 (SharpEmu gates
        // `readAsHalf` on the operand kind for exactly this reason).
        let read_as_half = (ctrl.op_sel_hi >> i) & 1 != 0 && operand_is_variable(inst.src[i]);
        if read_as_half {
            let component = u32::from((ctrl.op_sel >> i) & 1 != 0);
            body.push_str(&format!(
                "         %{val}_pk = OpExtInst %v2float %GLSL_std_450 UnpackHalf2x16 %{raw}\n\
                 \x20        %{val}_b = OpCompositeExtract %float %{val}_pk {component}\n"
            ));
        } else {
            body.push_str(&format!("         %{val}_b = OpBitcast %float %{raw}\n"));
        }
        let mut cur = format!("{val}_b");
        if (ctrl.neg_hi >> i) & 1 != 0 {
            body.push_str(&format!(
                "         %{val}_a = OpExtInst %float %GLSL_std_450 FAbs %{cur}\n"
            ));
            cur = format!("{val}_a");
        }
        if (ctrl.neg_lo >> i) & 1 != 0 {
            body.push_str(&format!("         %{val}_n = OpFNegate %float %{cur}\n"));
            cur = format!("{val}_n");
        }
        // No `OpCopyObject` in this crate's assembler — track the final id.
        srcs.push(cur);
    }

    body.push_str(&format!(
        "         %fmix_{index} = OpExtInst %float %GLSL_std_450 Fma %{a} %{b} %{c}\n",
        a = srcs[0],
        b = srcs[1],
        c = srcs[2]
    ));
    let product = if ctrl.clamp {
        vop3p_clamp(
            &format!("fmix_{index}"),
            &format!("fmixc_{index}"),
            &mut body,
        );
        format!("fmixc_{index}")
    } else {
        format!("fmix_{index}")
    };

    let value = match inst.type_ {
        ShaderInstructionType::VFmaMixF32 => product,
        // MIXLO / MIXHI: narrow to f16 and merge into one half of vdst,
        // leaving the other half intact. `PackHalf2x16` of `(v, 0)` puts
        // f16(v) in the low bits; of `(0, v)` in the high bits.
        ShaderInstructionType::VFmaMixloF16 | ShaderInstructionType::VFmaMixhiF16 => {
            let hi = inst.type_ == ShaderInstructionType::VFmaMixhiF16;
            let pair = if hi {
                format!("%float_0_000000 %{product}")
            } else {
                format!("%{product} %float_0_000000")
            };
            let (half_mask, keep_mask) = if hi {
                ("%uint_0xffff0000", "%uint_0x0000ffff")
            } else {
                ("%uint_0x0000ffff", "%uint_0xffff0000")
            };
            body.push_str(&format!(
                "         %mixv_{index} = OpCompositeConstruct %v2float {pair}\n\
                 \x20        %mixp_{index} = OpExtInst %uint %GLSL_std_450 PackHalf2x16 %mixv_{index}\n\
                 \x20        %mixh_{index} = OpBitwiseAnd %uint %mixp_{index} {half_mask}\n\
                 \x20        %mixd_{index} = OpLoad %float %{dst}\n\
                 \x20        %mixdu_{index} = OpBitcast %uint %mixd_{index}\n\
                 \x20        %mixk_{index} = OpBitwiseAnd %uint %mixdu_{index} {keep_mask}\n\
                 \x20        %mixo_{index} = OpBitwiseOr %uint %mixk_{index} %mixh_{index}\n\
                 \x20        %mixf_{index} = OpBitcast %float %mixo_{index}\n",
                dst = dst_value.value
            ));
            format!("mixf_{index}")
        }
        other => {
            return Err(not_supported(
                FUNC,
                format!("instruction type {other:?} is not a mix op"),
            ));
        }
    };

    vop3p_store(&dst_value.value, &value, &index_str, &mut body);
    *dst_source += "\n";
    *dst_source += &body;
    Ok(true)
}

/// Kyty: `Recompile_VMbcntHiU32B32_SVdstSVsrc0SVsrc1` (ShaderSpirv.cpp
/// L5455).
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
#[allow(dead_code)] // Sub/Subrev remain staged; Add uses the validator-clean body below.
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

/// Validator-clean RDNA2 `v_add_co_u32`: add two dwords, write the sum to the
/// VGPR and the unsigned overflow bit to the scalar carry mask.
///
/// Kyty's staged body uses `OpIAddCarry`; naga cannot validate that opcode, so
/// use the equivalent `sum < src0` overflow test already used by the working
/// carry-in sibling. This also keeps the generated module consumable by both
/// Vulkan and the repository's independent validator.
fn recompile_v_add_co_u32(
    index: u32,
    code: &ShaderCode,
    dst_source: &mut String,
    spirv: &Spirv<'_>,
    _param: &Params,
    _scc_check: SccCheck,
) -> Result<bool, ShaderRecompileError> {
    const FUNC: &str = "Recompile_VAddCoU32_VdstSdst2Vsrc0Vsrc1";
    let inst = inst_at(code, index, FUNC)?;
    let index_str = format!("{index}");

    if !operand_is_variable(inst.dst) || !operand_is_variable(inst.dst2) {
        return Err(not_supported(FUNC, "destination is not a variable"));
    }
    let dst_value = operand_variable_to_str(inst.dst);
    let carry_out0 = operand_variable_to_str_shift(inst.dst2, 0);
    let carry_out1 = operand_variable_to_str_shift(inst.dst2, 1);
    if dst_value.type_ != SpirvType::Float || carry_out0.type_ != SpirvType::Uint {
        return Err(not_supported(FUNC, "unexpected destination type"));
    }
    if operand_is_exec(inst.dst2) {
        return Err(not_supported(FUNC, "exec carry-out"));
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
        %tsum_<index> = OpIAdd %uint %t0_<index> %t1_<index>
        %tcarry_<index> = OpULessThan %bool %tsum_<index> %t0_<index>
        %tcarry_u_<index> = OpSelect %uint %tcarry_<index> %uint_1 %uint_0
        %exec_lo_u_<index> = OpLoad %uint %exec_lo
        %exec_lo_b_<index> = OpINotEqual %bool %exec_lo_u_<index> %uint_0
        %tcarry_mask_<index> = OpSelect %uint %exec_lo_b_<index> %tcarry_u_<index> %uint_0
               OpStore %<carryout0> %tcarry_mask_<index>
               OpStore %<carryout1> %uint_0
        %tsumf_<index> = OpBitcast %float %tsum_<index>
        %tdst_<index> = OpLoad %float %<dst>
        %tval_<index> = OpSelect %float %exec_lo_b_<index> %tsumf_<index> %tdst_<index>
               OpStore %<dst> %tval_<index>
"#;
    *dst_source += &TEXT
        .replace("<load0>", &load0)
        .replace("<load1>", &load1)
        .replace("<carryout0>", &carry_out0.value)
        .replace("<carryout1>", &carry_out1.value)
        .replace("<dst>", &dst_value.value)
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
    // Beyond Kyty (SharpEmu PR #587): FLAT-class direct guest-memory access via
    // the `%global_mem` window. One row per width; FLAT vs GLOBAL addressing is
    // selected inside the body by `ShaderInstruction::uses_flat_address`.
    f(recompile_flat_load_ubyte,   T::FlatLoadUbyte,    F::FlatAddr, p1("")),
    f(recompile_flat_load_dword,   T::FlatLoadDword,    F::FlatAddr, p1("")),
    f(recompile_flat_load_dwordx2, T::FlatLoadDwordX2,  F::FlatAddr, p1("")),
    f(recompile_flat_load_dwordx3, T::FlatLoadDwordX3,  F::FlatAddr, p1("")),
    f(recompile_flat_load_dwordx4, T::FlatLoadDwordX4,  F::FlatAddr, p1("")),
    f(recompile_flat_store_dword,  T::FlatStoreDword,   F::FlatAddr, p1("")),
    f(recompile_flat_store_dwordx2,T::FlatStoreDwordX2, F::FlatAddr, p1("")),
    f(recompile_flat_store_dwordx4,T::FlatStoreDwordX4, F::FlatAddr, p1("")),
    f(recompile_buffer_load_dword_vdata1,    T::BufferLoadDword,   F::Vdata1VaddrSvSoffsIdxen, p1("")),
    // Beyond Kyty (it NIs the opcode) — measured on Minecraft's menu VS.
    f(recompile_buffer_load_dwordx4,         T::BufferLoadDwordX4, F::Vdata4VaddrSvSoffsIdxen, p1("")),
    f(recompile_buffer_load_dwordx4,         T::BufferLoadDwordX4, F::Vdata4Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_load_dwordx4,         T::BufferLoadDwordX4, F::Vdata4SvSoffs, p1("")),
    f(recompile_buffer_load_dwordx4,         T::BufferLoadDwordX4, F::Vdata4VaddrSvSoffsOffen, p1("")),
    // Beyond Kyty: the two-dword raw-load row (ASTRO.BOT scene compute).
    f(recompile_buffer_load_dwordx2,         T::BufferLoadDwordX2, F::Vdata2VaddrSvSoffsIdxen, p1("")),
    f(recompile_buffer_load_dwordx2,         T::BufferLoadDwordX2, F::Vdata2Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_load_dwordx2,         T::BufferLoadDwordX2, F::Vdata2SvSoffs, p1("")),
    f(recompile_buffer_load_dwordx2,         T::BufferLoadDwordX2, F::Vdata2VaddrSvSoffsOffen, p1("")),

    f(recompile_buffer_load_dwordx3,         T::BufferLoadDwordX3, F::Vdata3VaddrSvSoffsIdxen, p1("")),
    f(recompile_buffer_load_dwordx3,         T::BufferLoadDwordX3, F::Vdata3Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_load_dwordx3,         T::BufferLoadDwordX3, F::Vdata3SvSoffs, p1("")),
    f(recompile_buffer_load_dwordx3,         T::BufferLoadDwordX3, F::Vdata3VaddrSvSoffsOffen, p1("")),
    f(recompile_buffer_load_format_x_vdata1, T::BufferLoadFormatX, F::Vdata1VaddrSvSoffsIdxen, p1("")),
    // Beyond Kyty: four-channel MUBUF typed fetch with index-enable addressing.
    // Measured first blocker of Avatar: Frontiers of Pandora.
    f(recompile_buffer_load_format_xyzw_vdata4, T::BufferLoadFormatXyzw, F::Vdata4VaddrSvSoffsIdxen, p1("")),
    f(recompile_buffer_store_dword_vdata1, T::BufferStoreDword, F::Vdata1VaddrSvSoffsIdxen, p1("")),
    // Wired from the staged set for ASTRO.BOT's formatted stores (the parse
    // gate no longer blanket-rejects MUBUF addressing modes).
    f(recompile_buffer_store_format_x_vdata1,  T::BufferStoreFormatX,  F::Vdata1VaddrSvSoffsIdxen, p1("")),
    f(recompile_buffer_store_format_xy_vdata2, T::BufferStoreFormatXy, F::Vdata2VaddrSvSoffsIdxen, p1("")),
    // Beyond Kyty: flexible MUBUF addressing (idxen==0 and/or offen==1) for
    // the single-dword loads/stores — the Vdata1 counterpart of the
    // BufferLoadDwordX4 quartet. Shared body `mubuf_flexible`.
    f(recompile_buffer_load_dword_flexible,  T::BufferLoadDword,  F::Vdata1SvSoffs,                 p1("")),
    f(recompile_buffer_load_dword_flexible,  T::BufferLoadDword,  F::Vdata1VaddrSvSoffsOffen,       p1("")),
    f(recompile_buffer_load_dword_flexible,  T::BufferLoadDword,  F::Vdata1Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_store_dword_flexible, T::BufferStoreDword, F::Vdata1SvSoffs,                 p1("")),
    f(recompile_buffer_store_dword_flexible, T::BufferStoreDword, F::Vdata1VaddrSvSoffsOffen,       p1("")),
    f(recompile_buffer_store_dword_flexible, T::BufferStoreDword, F::Vdata1Vaddr2SvSoffsOffenIdxen, p1("")),
    // Beyond Kyty: single zero-extended byte load (ASTRO.BOT scene compute,
    // raw 0xe02020c0), all four addressing modes through `mubuf_flexible`.
    f(recompile_buffer_load_ubyte,           T::BufferLoadUbyte,  F::Vdata1VaddrSvSoffsIdxen,       p1("")),
    f(recompile_buffer_load_ubyte,           T::BufferLoadUbyte,  F::Vdata1SvSoffs,                 p1("")),
    f(recompile_buffer_load_ubyte,           T::BufferLoadUbyte,  F::Vdata1VaddrSvSoffsOffen,       p1("")),
    f(recompile_buffer_load_ubyte,           T::BufferLoadUbyte,  F::Vdata1Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_load_format_x_flexible,  T::BufferLoadFormatX,  F::Vdata1SvSoffs,                 p1("")),
    f(recompile_buffer_load_format_x_flexible,  T::BufferLoadFormatX,  F::Vdata1VaddrSvSoffsOffen,       p1("")),
    f(recompile_buffer_load_format_x_flexible,  T::BufferLoadFormatX,  F::Vdata1Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_store_format_x_flexible, T::BufferStoreFormatX, F::Vdata1SvSoffs,                 p1("")),
    f(recompile_buffer_store_format_x_flexible, T::BufferStoreFormatX, F::Vdata1VaddrSvSoffsOffen,       p1("")),
    f(recompile_buffer_store_format_x_flexible, T::BufferStoreFormatX, F::Vdata1Vaddr2SvSoffsOffenIdxen, p1("")),
    // Beyond Kyty (`buffer_store_format_xyz(w)` are KYTY_NI upstream):
    // measured on ASTRO.BOT scene compute — 925 dispatches / 30s on xyzw.
    f(recompile_buffer_store_format_xyzw, T::BufferStoreFormatXyzw, F::Vdata4VaddrSvSoffsIdxen,       p1("")),
    f(recompile_buffer_store_format_xyzw, T::BufferStoreFormatXyzw, F::Vdata4Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_store_format_xyzw, T::BufferStoreFormatXyzw, F::Vdata4SvSoffs,                 p1("")),
    f(recompile_buffer_store_format_xyzw, T::BufferStoreFormatXyzw, F::Vdata4VaddrSvSoffsOffen,       p1("")),
    ni("Recompile_BufferStoreFormatXyz_Vdata3VaddrSvSoffsIdxen (no Kyty upstream; needs a float3 store helper)", 0, T::BufferStoreFormatXyz, F::Vdata3VaddrSvSoffsIdxen, p1("")),
    // Beyond Kyty (`buffer_store_dwordx4` is KYTY_NI upstream): raw
    // four-dword store, measured on ASTRO.BOT scene compute (MUBUF 0x1e,
    // raw 0xe0780000). Shared body `buffer_store_dwordxn`.
    f(recompile_buffer_store_dwordx4, T::BufferStoreDwordX4, F::Vdata4VaddrSvSoffsIdxen,       p1("")),
    f(recompile_buffer_store_dwordx4, T::BufferStoreDwordX4, F::Vdata4Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_store_dwordx4, T::BufferStoreDwordX4, F::Vdata4SvSoffs,                 p1("")),
    f(recompile_buffer_store_dwordx4, T::BufferStoreDwordX4, F::Vdata4VaddrSvSoffsOffen,       p1("")),
    f(recompile_buffer_store_dwordx2, T::BufferStoreDwordX2, F::Vdata2VaddrSvSoffsIdxen,       p1("")),
    f(recompile_buffer_store_dwordx2, T::BufferStoreDwordX2, F::Vdata2Vaddr2SvSoffsOffenIdxen, p1("")),
    f(recompile_buffer_store_dwordx2, T::BufferStoreDwordX2, F::Vdata2SvSoffs,                 p1("")),
    f(recompile_buffer_store_dwordx2, T::BufferStoreDwordX2, F::Vdata2VaddrSvSoffsOffen,       p1("")),

    f(recompile_fetch, T::FetchX,    F::Vdata1VaddrSvSoffsIdxen, p1("")),
    f(recompile_fetch, T::FetchXy,   F::Vdata2VaddrSvSoffsIdxen, p1("")),
    f(recompile_fetch, T::FetchXyz,  F::Vdata3VaddrSvSoffsIdxen, p1("")),
    f(recompile_fetch, T::FetchXyzw, F::Vdata4VaddrSvSoffsIdxen, p1("")),

    f(recompile_ds_append,  T::DsAppend,  F::VdstGds, p1("")),
    f(recompile_ds_consume, T::DsConsume, F::VdstGds, p1("")),
    // Beyond Kyty: LDS write/read + the workgroup barrier gluing them
    // (ASTRO.BOT scene compute).
    f(recompile_ds_add_u32,   T::DsAddU32,   F::Vsrc0Vsrc1Vsrc2,    p1("")),
    f(recompile_ds_add_rtn_u32, T::DsAddRtnU32, F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_ds_wrxchg_rtn_b32, T::DsWrxchgRtnB32, F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_ds_write_b32, T::DsWriteB32, F::Vsrc0Vsrc1Vsrc2,    p1("")),
    f(recompile_ds_read_b32,  T::DsReadB32,  F::SVdstSVsrc0SVsrc1,  p1("")),
    f(recompile_ds_read2_b32, T::DsRead2B32, F::Vdst2Vsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_ds_read_b64,  T::DsReadB64,  F::Vdst2Vsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_ds_read_b96,  T::DsReadB96,  F::Vdst3Vsrc0Vsrc1,      p1("")),
    f(recompile_ds_read_b128, T::DsReadB128, F::Vdst4Vsrc0Vsrc1,      p1("")),
    f(recompile_ds_write_b96, T::DsWriteB96, F::Vsrc0Vsrc13Vsrc2,   p1("")),
    f(recompile_ds_write_b128, T::DsWriteB128, F::Vsrc0Vsrc14Vsrc2, p1("")),
    f(recompile_s_barrier,    T::SBarrier,   F::Empty,              p1("")),

    f(recompile_exp_null,                           T::Exp, F::NullOffOffOffOffVmDone,         p1("")),
    f(recompile_exp_mrt0_off_off_compr_vm_done,     T::Exp, F::Mrt0OffOffComprVmDone,          p1("")),
    f(recompile_exp_mrt0_vsrc0_vsrc1_compr_vm_done, T::Exp, F::Mrt0Vsrc0Vsrc1ComprVmDone,      p1("")),
    f(recompile_exp_mrt0_vsrc0123_vm_done,          T::Exp, F::Mrt0Vsrc0Vsrc1Vsrc2Vsrc3VmDone, p1("")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param0Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param0")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param1Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param1")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param2Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param2")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param3Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param3")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param4Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param4")),
    // Beyond Kyty: param5..param31. Same body — only the destination slot
    // differs, and `%paramN` is declared for every slot the body writes (see
    // `max_exp_param`). RDNA 2 ISA doc 70648: EXP targets 32..63 are
    // PARAM0..PARAM31.
    f(recompile_exp_param_xxx,                      T::Exp, F::Param5Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param5")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param6Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param6")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param7Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param7")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param8Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param8")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param9Vsrc0Vsrc1Vsrc2Vsrc3,     p1("param9")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param10Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param10")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param11Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param11")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param12Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param12")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param13Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param13")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param14Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param14")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param15Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param15")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param16Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param16")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param17Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param17")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param18Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param18")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param19Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param19")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param20Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param20")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param21Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param21")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param22Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param22")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param23Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param23")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param24Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param24")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param25Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param25")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param26Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param26")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param27Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param27")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param28Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param28")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param29Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param29")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param30Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param30")),
    f(recompile_exp_param_xxx,                      T::Exp, F::Param31Vsrc0Vsrc1Vsrc2Vsrc3,    p1("param31")),
    f(recompile_exp_pos0,                           T::Exp, F::Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done,   p1("")),
    // Auxiliary position exports (clip/cull distances via PA_CL_VS_OUT_CNTL);
    // accepted and dropped until VS_OUT_CNTL is plumbed — see
    // recompile_exp_pos_aux.
    f(recompile_exp_pos_aux,                        T::Exp, F::Pos1Vsrc0Vsrc1Vsrc2Vsrc3,       p1("")),
    f(recompile_exp_pos_aux,                        T::Exp, F::Pos2Vsrc0Vsrc1Vsrc2Vsrc3,       p1("")),
    f(recompile_exp_pos_aux,                        T::Exp, F::Pos3Vsrc0Vsrc1Vsrc2Vsrc3,       p1("")),
    f(recompile_exp_prim,                           T::Exp, F::PrimVsrc0OffOffOffDone,         p1("")),

    f(recompile_image_get_resinfo_dmask3, T::ImageGetResinfo, F::Vdata2VaddrStDmask3, p1("")),
    f(recompile_image_load_dmask_f,        T::ImageLoad,      F::Vdata4Vaddr3StDmaskF,   p1("")),
    f(recompile_image_load_dmask1,         T::ImageLoad,      F::Vdata1Vaddr3StDmask1,   p1("")),
    f(recompile_image_load_dmask3,         T::ImageLoad,      F::Vdata2Vaddr3StDmask3,   p1("")),
    f(recompile_image_load_dmask_c,        T::ImageLoad,      F::Vdata2Vaddr3StDmaskC,   p1("")),
    f(recompile_image_load_dmask7,         T::ImageLoad,      F::Vdata3Vaddr3StDmask7,   p1("")),
    // Wired for the texture chain: Minecraft's content pixel shaders reach
    // ImageSample the moment their vertex partners translate. The nine
    // recompilers were already ported (shared dmask body + Lz/LzO); the
    // downstream texture upload feeds the %textures2D_S/%samplers arrays.
    f(recompile_image_sample_dmask1,     T::ImageSample,    F::Vdata1Vaddr3StSsDmask1, p1("")),
    f(recompile_image_sample_dmask2,     T::ImageSample,    F::Vdata1Vaddr3StSsDmask2, p1("")),
    f(recompile_image_sample_dmask8,     T::ImageSample,    F::Vdata1Vaddr3StSsDmask8, p1("")),
    f(recompile_image_sample_dmask3,     T::ImageSample,    F::Vdata2Vaddr3StSsDmask3, p1("")),
    f(recompile_image_sample_dmask5,     T::ImageSample,    F::Vdata2Vaddr3StSsDmask5, p1("")),
    f(recompile_image_sample_dmask9,     T::ImageSample,    F::Vdata2Vaddr3StSsDmask9, p1("")),
    f(recompile_image_sample_dmask7,     T::ImageSample,    F::Vdata3Vaddr3StSsDmask7, p1("")),
    f(recompile_image_sample_dmask_b,    T::ImageSample,    F::Vdata3Vaddr3StSsDmaskB, p1("")),
    f(recompile_image_sample_dmask_f,    T::ImageSample,    F::Vdata4Vaddr3StSsDmaskF, p1("")),
    f(recompile_image_sample_l_dmask7,   T::ImageSampleL,   F::Vdata3Vaddr4StSsDmask7, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata1Vaddr3StSsDmask1, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata1Vaddr3StSsDmask8, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata2Vaddr3StSsDmask3, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata2Vaddr3StSsDmask5, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata2Vaddr3StSsDmask9, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata3Vaddr3StSsDmask7, p1("")),
    f(recompile_image_sample_c_lz,       T::ImageSampleCLz, F::Vdata4Vaddr3StSsDmaskF, p1("")),
    f(recompile_image_sample_lz_dmask1,  T::ImageSampleLz,  F::Vdata1Vaddr3StSsDmask1, p1("")),
    f(recompile_image_sample_lz_dmask2,  T::ImageSampleLz,  F::Vdata1Vaddr3StSsDmask2, p1("")),
    f(recompile_image_sample_lz_dmask8,  T::ImageSampleLz,  F::Vdata1Vaddr3StSsDmask8, p1("")),
    f(recompile_image_sample_lz_dmask3,  T::ImageSampleLz,  F::Vdata2Vaddr3StSsDmask3, p1("")),
    f(recompile_image_sample_lz_dmask7,  T::ImageSampleLz,  F::Vdata3Vaddr3StSsDmask7, p1("")),
    f(recompile_image_sample_lz_dmask_f, T::ImageSampleLz,  F::Vdata4Vaddr3StSsDmaskF, p1("")),
    f(recompile_image_sample_lzo_dmask1, T::ImageSampleLzO, F::Vdata1Vaddr4StSsDmask1, p1("")),
    f(recompile_image_sample_lzo_dmask2, T::ImageSampleLzO, F::Vdata1Vaddr4StSsDmask2, p1("")),
    f(recompile_image_sample_lzo_dmask7, T::ImageSampleLzO, F::Vdata3Vaddr4StSsDmask7, p1("")),
    // Beyond Kyty: four-texel single-channel gather at LOD 0 (ASTRO.BOT
    // scene compute, MIMG 0x47 dmask 0x1).
    f(recompile_image_gather4_lz_dmask1, T::ImageGather4Lz, F::Vdata4Vaddr3StSsDmask1, p1("")),
    f(recompile_image_gather4_lz_dmask2, T::ImageGather4Lz, F::Vdata4Vaddr3StSsDmask2, p1("")),
    f(recompile_image_gather4_lz_dmask4, T::ImageGather4Lz, F::Vdata4Vaddr3StSsDmask4, p1("")),
    f(recompile_image_gather4_lz_dmask8, T::ImageGather4Lz, F::Vdata4Vaddr3StSsDmask8, p1("")),
    f(recompile_image_store_dmask1,        T::ImageStore,     F::Vdata1Vaddr3StDmask1,   p1("")),
    f(recompile_image_store_dmask3,        T::ImageStore,     F::Vdata2Vaddr3StDmask3,   p1("")),
    f(recompile_image_store_dmask_f,       T::ImageStore,     F::Vdata4Vaddr3StDmaskF,   p1("")),
    ni("Recompile_ImageStoreMip_Vdata4Vaddr4StDmaskF",    3173, T::ImageStoreMip,  F::Vdata4Vaddr4StDmaskF,   p1("")),

    f(recompile_sbuffer_load_dword,   T::SBufferLoadDword,   F::SdstSvSoffset,  p1("")),
    f(recompile_sbuffer_load_dwordx2, T::SBufferLoadDwordx2, F::Sdst2SvSoffset, p1("")),
    f(recompile_sbuffer_load_dwordx4, T::SBufferLoadDwordx4, F::Sdst4SvSoffset, p1("")),
    f(recompile_sbuffer_load_dwordx8, T::SBufferLoadDwordx8, F::Sdst8SvSoffset, p1("")),
    f(recompile_sbuffer_load_dwordx16, T::SBufferLoadDwordx16, F::Sdst16SvSoffset, p1("")),
    // Beyond Kyty: the combined addressing mode (register soffset AND a
    // non-zero immediate offset both live). Same lowering — the two byte
    // offsets sum, see `Format::SdstSvSoffsetOffset`. Measured first blocker of
    // ASTRO.BOT, which died in the PARSER before reaching a table row at all.
    f(recompile_sbuffer_load_dword,   T::SBufferLoadDword,   F::SdstSvSoffsetOffset,  p1("")),
    f(recompile_sbuffer_load_dwordx2, T::SBufferLoadDwordx2, F::Sdst2SvSoffsetOffset, p1("")),
    f(recompile_sbuffer_load_dwordx4, T::SBufferLoadDwordx4, F::Sdst4SvSoffsetOffset, p1("")),
    f(recompile_sbuffer_load_dwordx8, T::SBufferLoadDwordx8, F::Sdst8SvSoffsetOffset, p1("")),
    f(recompile_sbuffer_load_dwordx16, T::SBufferLoadDwordx16, F::Sdst16SvSoffsetOffset, p1("")),

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
    f(recompile_sload_dwordx16, T::SLoadDwordx16, F::Sdst16SbaseSoffset, p1("")),
    // Beyond Kyty: the same five widths in the three-operand addressing form
    // (`src[1]` register soffset + `src[2]` immediate — RDNA2 adds both). They
    // share the x1..x16 lowering: a load whose soffset analysis proved is served
    // from its per-PC capture, anything else refuses by name inside
    // `sload_dword_extended`. Without these rows the dispatch table had NO entry
    // for the format and the shader died at "no table entry" instead.
    f(recompile_sload_dword,    T::SLoadDword,    F::SdstSbaseSoffsetOffset,   p1("")),
    f(recompile_sload_dwordx2,  T::SLoadDwordx2,  F::Sdst2SbaseSoffsetOffset,  p1("")),
    f(recompile_sload_dwordx4,  T::SLoadDwordx4,  F::Sdst4SbaseSoffsetOffset,  p1("")),
    f(recompile_sload_dwordx8,  T::SLoadDwordx8,  F::Sdst8SbaseSoffsetOffset,  p1("")),
    f(recompile_sload_dwordx16, T::SLoadDwordx16, F::Sdst16SbaseSoffsetOffset, p1("")),

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
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SLshl1AddU32, F::SVdstSVsrc0SVsrc1, p3("%ts_<index> = OpFunctionCall %v2uint %lshl_add %t0_<index> %t1_<index> %uint_1", "%t_<index> = OpCompositeExtract %uint %ts_<index> 0", "%carry_<index> = OpCompositeExtract %uint %ts_<index> 1"), S::CarryOut),
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SLshl2AddU32, F::SVdstSVsrc0SVsrc1, p3("%ts_<index> = OpFunctionCall %v2uint %lshl_add %t0_<index> %t1_<index> %uint_2", "%t_<index> = OpCompositeExtract %uint %ts_<index> 0", "%carry_<index> = OpCompositeExtract %uint %ts_<index> 1"), S::CarryOut),
    fs(recompile_s_xxx_u32_svdst_svsrc01, T::SLshl3AddU32, F::SVdstSVsrc0SVsrc1, p3("%ts_<index> = OpFunctionCall %v2uint %lshl_add %t0_<index> %t1_<index> %uint_3", "%t_<index> = OpCompositeExtract %uint %ts_<index> 0", "%carry_<index> = OpCompositeExtract %uint %ts_<index> 1"), S::CarryOut),
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
    // RDNA2 VOP2 0x1e v_xnor_b32 = ~(src0 ^ src1). Measured in ASTRO.BOT scene
    // composite CS.
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VXnorB32,    F::SVdstSVsrc0SVsrc1, p2("%txor_<index> = OpBitwiseXor %uint %t0_<index> %t1_<index>", "%t_<index> = OpNot %uint %txor_<index>")),
    // RDNA2-only (no Kyty upstream rows): the carry-less VOP2 add/sub family
    // measured in Minecraft's menu CS.
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VAddNcU32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpIAdd %uint %t0_<index> %t1_<index>")),
    // RDNA2 v_add_co_ci_u32 (VOP2 0x28 + VOP3B 0x128): add with carry in/out.
    // Measured in ASTRO.BOT scene composite CS.
    f(recompile_v_add_co_ci_u32, T::VAddCoCiU32, F::VdstSdst2Vsrc0Vsrc1Smask2, p1("")),
    f(recompile_v_subb_u32, T::VSubbU32, F::VdstSdst2Vsrc0Vsrc1Smask2, p1("")),
    f(recompile_v_subbrev_u32, T::VSubbrevU32, F::VdstSdst2Vsrc0Vsrc1Smask2, p1("")),
    // RDNA2 v_mad_u64_u32 (VOP3B 0x176): u64 = u32*u32 + u64. Shares the
    // add-with-carry format key (distinct type). Measured in ASTRO.BOT scene
    // composite CS 0x555f4f500.
    f(recompile_v_mad_u64_u32, T::VMadU64U32, F::VdstSdst2Vsrc0Vsrc1Smask2, p1("")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VSubNcU32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpISub %uint %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VSubrevNcU32, F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpISub %uint %t1_<index> %t0_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VAddF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpFAdd %float %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMacF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %float %GLSL_std_450 Fma %t0_<index> %t1_<index> %tdst_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMaxF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %float %GLSL_std_450 FMax %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_f32_svdst_svsrc01, T::VMinF32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %float %GLSL_std_450 FMin %t0_<index> %t1_<index>")),
    // RDNA2 integer min/max (VOP2 0x11-0x14). Unsigned via the uint family +
    // GLSL UMin/UMax; signed via the int family + SMin/SMax. v_min_u32 measured
    // in ASTRO.BOT scene CS 0x555f4f500; the other three are its direct siblings.
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VMinU32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %uint %GLSL_std_450 UMin %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_b32_svdst_svsrc01, T::VMaxU32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %uint %GLSL_std_450 UMax %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_i32_svdst_svsrc01, T::VMinI32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %int %GLSL_std_450 SMin %t0_<index> %t1_<index>")),
    f(recompile_v_xxx_i32_svdst_svsrc01, T::VMaxI32,    F::SVdstSVsrc0SVsrc1, p1("%t_<index> = OpExtInst %int %GLSL_std_450 SMax %t0_<index> %t1_<index>")),
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
    // Wired for ASTRO.BOT scene CS (VOP3 0x365/0x366 lane-index idiom).
    // Kyty's single-lane model: mbcnt(mask, src1) = src1 when exec is on
    // (zero active lanes below the current one).
    f(recompile_vmbcnt_hi_u32_b32, T::VMbcntHiU32B32,  F::SVdstSVsrc0SVsrc1, p1("")),
    f(recompile_vmbcnt_lo_u32_b32, T::VMbcntLoU32B32,  F::SVdstSVsrc0SVsrc1, p1("")),

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
    // v_rcp_iflag_f32: identical arithmetic; the iflag TRAP status is not
    // modelled (see `VRcpIflagF32` in types.rs).
    f(recompile_v_xxx_f32_svdst_svsrc0, T::VRcpIflagF32, F::SVdstSVsrc0, p1("%t_<index> = OpFDiv %float %float_1_000000 %t0_<index>")),
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
    f(recompile_vreadfirstlane_b32, T::VReadfirstlaneB32, F::SVdstSVsrc0, p1("")),

    fs(recompile_s_and_saveexec_b64, T::SAndSaveexecB64, F::Sdst2Ssrc02, p1(""), S::NonZero),
    fs(recompile_s_orn2_saveexec_b64, T::SOrn2SaveexecB64, F::Sdst2Ssrc02, p1(""), S::NonZero),
    fs(recompile_s_andn1_saveexec_b64, T::SAndn1SaveexecB64, F::Sdst2Ssrc02, p1(""), S::NonZero),
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
    f(recompile_s_waitcnt, T::SWaitcnt, F::Imm, p1("")),
    f(recompile_skip, T::SVersion,      F::Imm, p1("")),

    f(recompile_tbuffer_load_format_x_float1, T::TBufferLoadFormatX, F::Vdata1VaddrSvSoffsIdxenFloat1, p1("")),
    ni("Recompile_TBufferLoadFormatXyzw_Vdata4Vaddr2SvSoffsOffenIdxenFloat4", 4824, T::TBufferLoadFormatXyzw, F::Vdata4Vaddr2SvSoffsOffenIdxenFloat4, p1("")),
    f(recompile_tbuffer_load_format_xyzw_float4, T::TBufferLoadFormatXyzw, F::Vdata4VaddrSvSoffsIdxenFloat4, p1("")),
    // Beyond Kyty: the two-channel typed fetch. The offen variant carries a
    // per-thread voffset the float2 body does not add, so it is a named
    // refusal rather than a silently dropped address term (same shape as the
    // xyzw offen row above).
    f(recompile_tbuffer_load_format_xy_float2, T::TBufferLoadFormatXy, F::Vdata2VaddrSvSoffsIdxenFloat2, p1("")),
    ni("Recompile_TBufferLoadFormatXy_Vdata2Vaddr2SvSoffsOffenIdxenFloat2 (no Kyty upstream; the float2 body has no voffset term)", 0, T::TBufferLoadFormatXy, F::Vdata2Vaddr2SvSoffsOffenIdxenFloat2, p1("")),

    f(recompile_v_add_co_u32, T::VAddI32, F::VdstSdst2Vsrc0Vsrc1, p1("")),
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
    f(recompile_vcmp_gt_u64,  T::VCmpGtU64,  F::SmaskVsrc0Vsrc1, p1("")),
    f(recompile_vcmp_xxx_u32, T::VCmpLeU32,  F::SmaskVsrc0Vsrc1, p1("OpULessThanEqual")),
    f(recompile_vcmp_xxx_u32, T::VCmpLtU32,  F::SmaskVsrc0Vsrc1, p1("OpULessThan")),
    f(recompile_vcmp_xxx_u32, T::VCmpTU32,   F::SmaskVsrc0Vsrc1, p1("OpIEqual %bool %uint_0 %uint_0 ; ")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxEqF32, F::SmaskVsrc0Vsrc1, p1("OpFOrdEqual")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxNeqF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordNotEqual")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxGtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdGreaterThan")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxLtF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdLessThan")),
    // `v_cmpx_le_f32` (VOPC 0x13): exec-writing ordered <= — sibling of the
    // Lt/Gt rows above. Measured in ASTRO.BOT scene CS (58 dispatches/run).
    f(recompile_vcmpx_xxx_f32, T::VCmpxLeF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdLessThanEqual")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxNltF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordGreaterThanEqual")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxNgeF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordLessThan")),
    // Siblings measured on ASTRO.BOT scene CS: ge = ordered >=,
    // nle = !(a <= b) = unordered > (NaN → true).
    f(recompile_vcmpx_xxx_f32, T::VCmpxGeF32,  F::SmaskVsrc0Vsrc1, p1("OpFOrdGreaterThanEqual")),
    f(recompile_vcmpx_xxx_f32, T::VCmpxNleF32, F::SmaskVsrc0Vsrc1, p1("OpFUnordGreaterThan")),
    // Wired for the Minecraft menu CS (v_cmpx family; Kyty GCN semantics —
    // the comparison result lands in both the mask destination and EXEC; on
    // real RDNA2 cmpx writes EXEC only, the extra mask write is a documented
    // deviation until a title proves it matters).
    f(recompile_vcmpx_xxx_i32, T::VCmpxEqU32,  F::SmaskVsrc0Vsrc1, p1("OpIEqual")),
    f(recompile_vcmpx_xxx_i32, T::VCmpxNeU32,  F::SmaskVsrc0Vsrc1, p1("OpINotEqual")),
    f(recompile_vcmpx_xxx_u32, T::VCmpxGeU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThanEqual")),
    f(recompile_vcmpx_xxx_u32, T::VCmpxGtU32,  F::SmaskVsrc0Vsrc1, p1("OpUGreaterThan")),
    f(recompile_vcmpx_xxx_u32, T::VCmpxLeU32,  F::SmaskVsrc0Vsrc1, p1("OpULessThanEqual")),
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

    // VOP3P (beyond Kyty, SharpEmu PRs #466/#460/#420): the whole encoding was
    // undecoded, so any shader containing one packed instruction was dropped.
    f(recompile_vop3p_packed_f16, T::VPkFmaF16,   F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_vop3p_packed_f16, T::VPkAddF16,   F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_vop3p_packed_f16, T::VPkMulF16,   F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_vop3p_packed_f16, T::VPkMinF16,   F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_vop3p_packed_f16, T::VPkMaxF16,   F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_vop3p_fma_mix,    T::VFmaMixF32,    F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_vop3p_fma_mix,    T::VFmaMixloF16,  F::VdstVsrc0Vsrc1Vsrc2, p1("")),
    f(recompile_vop3p_fma_mix,    T::VFmaMixhiF16,  F::VdstVsrc0Vsrc1Vsrc2, p1("")),

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
    // `v_bfe_u32` (VOP3 0x148): unsigned bitfield extract —
    // dst = (src0 >> src1[4:0]) & ((1 << src2[4:0]) - 1). Offset = src1[4:0],
    // count = src2[4:0]; OpBitFieldUExtract does the shift+mask. Measured in
    // ASTRO.BOT scene-composite CS 0x500690400.
    f(recompile_v_xxx_u32_vdst_vsrc012, T::VBfeU32, F::VdstVsrc0Vsrc1Vsrc2, p3("%to_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31",
        "%ts_<index> = OpBitwiseAnd %uint %t2_<index> %uint_31",
        "%t_<index> = OpBitFieldUExtract %uint %t0_<index> %to_<index> %ts_<index>")),
    // `v_bfe_i32` (VOP3 0x149): signed bitfield extract — mask offset/count
    // to 5 bits like the hardware, then OpBitFieldSExtract through %int
    // (SPIR-V's sign-extension is carried by the INSTRUCTION, but naga's IR
    // keys signedness on the operand type, so the bitcasts keep both
    // consumers honest). Measured in ASTRO.BOT scene CS (58 dispatches/run).
    f(recompile_v_xxx_u32_vdst_vsrc012, T::VBfeI32, F::VdstVsrc0Vsrc1Vsrc2, p4("%to_<index> = OpBitwiseAnd %uint %t1_<index> %uint_31",
        "%ts_<index> = OpBitwiseAnd %uint %t2_<index> %uint_31",
        "%tb_<index> = OpBitcast %int %t0_<index>",
        "%te_<index> = OpBitFieldSExtract %int %tb_<index> %to_<index> %ts_<index>\n         %t_<index> = OpBitcast %uint %te_<index>")),
    // `v_bfi_b32` (VOP3 0x14a): bitfield insert —
    // dst = (src0 & src1) | (~src0 & src2). Measured in ASTRO.BOT scene CS
    // (58 dispatches/run).
    f(recompile_v_xxx_u32_vdst_vsrc012, T::VBfiB32, F::VdstVsrc0Vsrc1Vsrc2, p3("%ta_<index> = OpBitwiseAnd %uint %t0_<index> %t1_<index>",
        "%tn_<index> = OpNot %uint %t0_<index>\n         %tc_<index> = OpBitwiseAnd %uint %tn_<index> %t2_<index>",
        "%t_<index> = OpBitwiseOr %uint %ta_<index> %tc_<index>")),
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
    // `v_add3_u32`: dst = src0 + src1 + src2 (carry-less, RDNA2 VOP3 0x36d;
    // shadPS4 `V_ADD3_U32 = 877`). Measured in ASTRO.BOT scene compute.
    f(recompile_v_xxx_u32_vdst_vsrc012, T::VAdd3U32, F::VdstVsrc0Vsrc1Vsrc2, p2("%ta_<index> = OpIAdd %uint %t0_<index> %t1_<index>",
        "%t_<index> = OpIAdd %uint %ta_<index> %t2_<index>")),
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
        // Kyty EXITs when `stride * num_records & 3 != 0` (Shader.cpp
        // L2361 block) — see `shader_recompile_cs` for why that gate is
        // dropped (the host pads the upload; the shader only ever addresses
        // whole dwords).
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
        // Same dropped Kyty alignment gate as `shader_recompile_cs` — see
        // the comment there.
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
    // Kyty EXITs when `stride * num_records & 3 != 0` (Shader.cpp L2545
    // block). That gate is over-conservative: the recompiled SPIR-V only
    // ever addresses the buffer in whole dwords (the SSBO is a runtime
    // array of 32-bit elements and every load/store helper indexes it by
    // `byte_offset >> 2`), so a V# whose byte size is not a dword multiple
    // simply has an unaddressable tail. The host upload pads the byte
    // buffer to a dword multiple and the writeback truncates back to the
    // real size (`raeen-gpu` `prepare_stage_binding`), which preserves the
    // guest bytes beyond the V# exactly. Measured on ASTRO.BOT scene
    // compute (58 dispatches/run refused on this gate).
    let source = spirv_generate_source(code, None, None, Some(input_info))?;

    tracing::trace!("recompiled cs source:\n{source}");

    spirv_run(&source)
}
#[cfg(test)]
// Test fixtures build shader input structs field-by-field (direct, nested, and
// indexed fields); the struct-literal rewrite cannot express the nested/indexed
// assignments, so it would leave a mixed style across ~80 fixtures for no
// behavior change.
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::shader::parse::shader_parse;
    use crate::shader::types::{ShaderInstruction, ShaderOperand, ShaderType};

    const S_ENDPGM: u32 = 0xBF81_0000;
    ///  — filler so a one-instruction body clears the
    /// generator's "s_endpgm before instruction 2" floor.
    const V_MOV_V0_0: u32 = 0x7E00_0280;

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

    /// Real spirv-val (the Khronos validator, same invocation as the
    /// raeen-gpu runtime gate): the structurizer's acceptance bar. naga
    /// cannot serve here — its SPIR-V front end structurizes and ACCEPTS
    /// back-edge modules spirv-val (and drivers) reject.
    fn spirv_val_ok(words: &[u32], name: &str) {
        use spirv_tools::val::Validator;
        let validator = spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_3));
        let options = spirv_tools::val::ValidatorOptions {
            relax_block_layout: Some(true),
            ..Default::default()
        };
        if let Err(e) = validator.validate(words, Some(options)) {
            panic!("spirv-val of {name} failed: {e}");
        }
    }

    // ---- 0. control-flow structurization (dispatch-loop relooper) ---------

    /// A backward conditional branch (a guest loop). The measured ASTRO.BOT
    /// crash class: with no OpLoopMerge anywhere in guest codegen, every loop
    /// used to emit an illegal back-edge ("Back-edges can only be formed
    /// between a block and a loop header") — VK_SUCCESS at module creation,
    /// undefined behavior (AMD driver access violation) at dispatch.
    #[test]
    fn backward_loop_translates_to_valid_spirv() {
        let code = parse(
            &[
                0x7E00_0280, // pc 0: v_mov_b32 v0, 0
                0xBF85_FFFE, // pc 4: s_cbranch_scc1 -2  -> dst pc 0 (back-edge)
                S_ENDPGM,    // pc 8
            ],
            ShaderType::Compute,
        );
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let words = shader_recompile_cs(&code, &input_info).expect("backward loop must translate");
        spirv_val_ok(&words, "backward_loop");
    }

    /// A forward conditional skip (a guest `if`) stays valid.
    #[test]
    fn forward_conditional_skip_translates_to_valid_spirv() {
        let code = parse(
            &[
                0xBF88_0001, // pc 0: s_cbranch_execz +1 -> dst pc 8
                0x7E00_0280, // pc 4: v_mov_b32 v0, 0
                S_ENDPGM,    // pc 8
            ],
            ShaderType::Compute,
        );
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let words = shader_recompile_cs(&code, &input_info).expect("forward skip must translate");
        spirv_val_ok(&words, "forward_skip");
    }

    /// An unconditional forward branch (skipping dead code) stays valid.
    #[test]
    fn unconditional_forward_branch_translates_to_valid_spirv() {
        let code = parse(
            &[
                0xBF82_0001, // pc 0: s_branch +1 -> dst pc 8
                0x7E00_0280, // pc 4: v_mov_b32 v0, 0 (unreachable)
                S_ENDPGM,    // pc 8
            ],
            ShaderType::Compute,
        );
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let words =
            shader_recompile_cs(&code, &input_info).expect("forward s_branch must translate");
        spirv_val_ok(&words, "forward_branch");
    }

    /// A nested loop: outer loop body contains an inner backward branch.
    /// The dispatch loop flattens both into one switch — both back-edges
    /// become stores to the block variable.
    #[test]
    fn nested_backward_loops_translate_to_valid_spirv() {
        let code = parse(
            &[
                0x7E00_0280, // pc 0:  v_mov_b32 v0, 0        (outer head)
                0x7E02_0280, // pc 4:  v_mov_b32 v1, 0        (inner head)
                0xBF85_FFFE, // pc 8:  s_cbranch_scc1 -2  -> pc 4 (inner latch)
                0xBF84_FFFC, // pc 12: s_cbranch_scc0 -4  -> pc 0 (outer latch)
                S_ENDPGM,    // pc 16
            ],
            ShaderType::Compute,
        );
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let words = shader_recompile_cs(&code, &input_info).expect("nested loops must translate");
        spirv_val_ok(&words, "nested_loops");
    }

    /// Straight-line shaders (no labels) keep the legacy linear emission —
    /// no dispatch-loop scaffolding in the source at all.
    #[test]
    fn zero_label_shader_keeps_the_linear_fast_path() {
        let code = parse(&[0x7E00_0280, 0x7E02_0280, S_ENDPGM], ShaderType::Compute);
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("straight-line CS translates");
        assert!(
            !source.contains("reloop"),
            "zero-label shaders must not pay for the dispatch loop:\n{source}"
        );
        let words = spirv_run(&source).expect("assembles");
        spirv_val_ok(&words, "straight_line");
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
            400,
            "the three beyond-Kyty s_lshl1/2/3_add_u32 rows (SOP2 0x2e/0x2f/0x30; \
             0x30 is ASTRO.BOT's measured `unknown sop2 opcode`), \
             the eight beyond-Kyty VOP3P rows (SharpEmu PRs #466/#460/#420: \
             VPkFma/Add/Mul/Min/MaxF16 + VFmaMixF32/loF16/hiF16), \
             the beyond-Kyty exp-null row (EXP target 9, ASTRO.BOT),            the seven beyond-Kyty FLAT-class rows (SharpEmu PR #587: \
             FlatLoadDword/X2/X3/X4 + FlatStoreDword/X2/X4), and \
             204 Kyty rows plus the compute batch DsWrxchgRtnB32 and VCmpxNgeF32, and \
             SSubU32, SNop, SVersion, the RDNA2-only rows \
             (VLshlAddU32, VCmpxLtU32, VAddNcU32, VSubNcU32, VSubrevNcU32, VCvtI32F32, \
             VCvtFlrI32F32, VCmpxNltF32, SOrn2SaveexecB64, the ImageLoad dmask1/3/7 \
             and ImageSampleLz dmask1/2/F rows, ImageStore dmask1, \
             the Kyty-gated trio VAndOrB32/VLshlOrU32/VOr3U32, VAdd3U32, and the \
             v_cmpx_*_i32 block: VCmpxLtI32/GeI32/GtI32/LeI32/EqI32/NeI32), the \
             beyond-Kyty BufferLoadDwordX4 (+Offen and address-only) rows, the \
             twelve flexible-addressing MUBUF Vdata1 rows, the four \
             BufferStoreFormatXyzw rows (+1 staged Xyz), the exp pos1..pos3 \
             rows, ImageGetResinfo, SGetpcB64, SPackLlB32B16, the seven \
             ImageSampleCLz dmask rows, \
             and the four cubemap helpers VCubeId/Sc/Tc/MaF32, plus SNotB64, SBrevB32 and \
             VCmpxEqF32, the four BufferLoadDwordX2 rows, ImageStore dmask3, \
             VCmpxGeF32/VCmpxNleF32, the LDS family DsWriteB32/DsReadB32/\
             DsRead2B32/DsWriteB96/SBarrier, the round-4 batch: the four \
             BufferStoreDwordX4 rows, DsWriteB128, and ImageGather4Lz dmask1, \
             and the convergence batch: the four BufferLoadDwordX3 rows, \
             DsReadB64, and ImageSampleLz dmask3, and the round-7 batch: \
             VCmpxLeF32, VBfeI32, VBfiB32, and DsReadB128, and the round-8 \
             batch: the four BufferLoadUbyte rows and DsReadB96, and the \
             round-9 batch: VRcpIflagF32 and DsAddU32, and the RDNA2 \
             scene-composite batch: VXnorB32, VAddCoCiU32, SAndn1SaveexecB64 \
             and VMadU64U32, and the composite-frontier batch: the integer \
             min/max quartet (VMinU32/VMaxU32/VMinI32/VMaxI32), the four \
             BufferStoreDwordX2 rows, VBfeU32 wired from staged, and the \
             ASTRO.BOT pixel ImageSample dmask2 + ImageSampleLzO dmask1/2 rows, \
             and the measured VCmpxLeU32 + DsAddRtnU32 rows, and the \
             instruction-coverage batch: ImageGather4Lz dmask2/4/8 (the dmask \
             bit index is the SPIR-V gather Component operand) and \
             SLoadDwordx16 (SMEM/SMRD opcode 0x04), and the SMEM              register-soffset batch: the five Sdst/2/4/8/16-SbaseSoffsetOffset              rows (RDNA2 `base + soffset + imm`), \
             and the V#-buffer-load batch: the five \
             Sdst/2/4/8/16-SvSoffsetOffset rows (the same combined addressing \
             through a V# base, ASTRO.BOT) plus BufferLoadFormatXyzw \
             [Vdata4VaddrSvSoffsIdxen] (Avatar: Frontiers of Pandora), \n             and the Blasphemous II decoder-gap batch (+30): the 27 exp \n             param5..param31 rows (RDNA2 EXP targets 0x25..0x3f), the two \n             TBufferLoadFormatXy rows (one wired, one refused offen variant), \n             and ImageSample dmask 0xb, plus ImageSampleLz dmask 0x8"
        );
        assert_eq!(implemented + ni, table.len());
        assert_eq!(
            implemented, 392,
            "the three s_lshl1/2/3_add_u32 rows (SOP2 0x2e/0x2f/0x30), \
             the eight VOP3P rows (SharpEmu PRs #466/#460/#420), \
             the seven FLAT-class rows (SharpEmu PR #587), and the \
             C1 implemented subset plus title-driven ports (incl. DsWrxchgRtnB32, \
             VCmpxNgeF32, SVersion, the S_XXX_I32 \
             trio, VCvtFlrI32F32, VCmpxNltF32, SOrn2SaveexecB64, the ImageLoad \
             dmask1/3/7 + ImageSampleLz dmask1/2/F rows, ImageStore dmask1, the \
             nine ImageSample dmask recompilers, the VCmp \
              F32/I32/U32 families, address-only BufferLoadDwordX4, the wired \
              BufferStoreFormatX/Xy rows, the twelve flexible-addressing MUBUF \
              rows, the four BufferStoreFormatXyzw rows, the exp pos1..pos3 \
              rows, VAdd3U32, ImageGetResinfo, SGetpcB64, SPackLlB32B16, the seven \
              ImageSampleCLz dmask rows, and the four VCube*F32 \
              cubemap-coordinate helpers, the four BufferLoadDwordX2 rows, ImageStore \
              dmask3, VCmpxGeF32/VCmpxNleF32, the LDS family DsWriteB32/DsReadB32/\
              DsRead2B32/DsWriteB96/SBarrier, \
              the mbcnt pair wired from the staged set, the round-4 batch: \
              four BufferStoreDwordX4 rows, DsWriteB128, ImageGather4Lz dmask1, \
              and the convergence batch: four BufferLoadDwordX3 rows, DsReadB64, \
              ImageSampleLz dmask3, and the round-7 batch: VCmpxLeF32, VBfeI32, \
              VBfiB32, DsReadB128, and the round-8 batch: four BufferLoadUbyte \
              rows, DsReadB96, and the round-9 batch: VRcpIflagF32, DsAddU32, \
              and the RDNA2 scene-composite batch: VXnorB32, VAddCoCiU32, \
              SAndn1SaveexecB64, VMadU64U32, and the composite-frontier batch: \
              the integer min/max quartet, four BufferStoreDwordX2 rows, \
              VBfeU32, ImageSample dmask2, ImageSampleLzO dmask1/2, VCmpxLeU32, \
              DsAddRtnU32, \
              and the instruction-coverage batch: ImageGather4Lz dmask2/4/8 \
              and SLoadDwordx16, \
              and the SMEM register-soffset batch: five \
              Sdst/2/4/8/16-SbaseSoffsetOffset rows, \
              and the V#-buffer-load batch: five Sdst/2/4/8/16-SvSoffsetOffset \
              rows and BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen], \
              and the exp-null row: EXP target 9 accepted and dropped, \n              and the Blasphemous II decoder-gap batch: 27 exp param5..param31 \n              rows, TBufferLoadFormatXy, ImageSample dmask 0xb, and ImageSampleLz \n              dmask 0x8)"
        );
        assert_eq!(
            ni, 8,
            "C2 remainder (BufferStoreFormatX/Xy wired out; BufferStoreFormatXyz staged in; \
             the mbcnt pair wired for the RDNA2 VOP3 lane-index idiom; VBfeU32 wired out \
             of the staged VdstVsrc0Vsrc1Vsrc2 U32 set; plus TBufferLoadFormatXy's offen variant, \n             refused because the float2 body has no voffset term)"
        );

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

    /// Build one VOP3P dword pair. Field layout per `shader_parse_vop3p`.
    #[allow(clippy::too_many_arguments)]
    fn vop3p_words(
        opcode: u32,
        vdst: u32,
        src: [u32; 3],
        op_sel: u32,
        op_sel_hi: u32,
        neg_lo: u32,
        neg_hi: u32,
        clamp: bool,
    ) -> [u32; 2] {
        let w0 = (0x33 << 26)
            | (opcode << 16)
            | (u32::from(clamp) << 15)
            | (((op_sel_hi >> 2) & 1) << 14)
            | ((op_sel & 0x7) << 11)
            | ((neg_hi & 0x7) << 8)
            | (vdst & 0xff);
        let w1 = ((neg_lo & 0x7) << 29)
            | ((op_sel_hi & 0x3) << 27)
            | ((src[2] & 0x1ff) << 18)
            | ((src[1] & 0x1ff) << 9)
            | (src[0] & 0x1ff);
        [w0, w1]
    }

    /// SharpEmu PR #466 `3574a3b` ("was dropping Unity HDR shaders"): the whole
    /// VOP3P encoding used to fall into `shader_parse`'s catch-all, so ONE
    /// packed instruction killed the entire shader with `UnknownEncoding`.
    ///
    /// RED before `shader_parse_vop3p`: this parse returned
    /// `UnknownEncoding { raw: 0xcc.. }` and nothing downstream ran at all.
    #[test]
    fn vop3p_encoding_no_longer_drops_the_whole_shader() {
        // v_pk_mul_f16 v2, v0, v1 (opcode 0x10). op_sel_hi = 0b011 is the
        // assembler default (each source's HIGH lane reads its high half).
        let words = vop3p_words(0x10, 2, [256, 257, 0], 0, 0b011, 0, 0, false);
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(0, &[words[0], words[1], S_ENDPGM], &mut code, true)
            .expect("VOP3P must decode, not drop the shader");

        let inst = code.get_instructions()[0];
        assert_eq!(inst.type_, T::VPkMulF16);
        assert_eq!(inst.src_num, 2, "the packed two-source ops read src0/src1");
        assert_eq!(inst.dst.register_id, 2);
        assert_eq!(inst.src[0].register_id, 0);
        assert_eq!(inst.src[1].register_id, 1);
        let ctrl = inst.vop3p.expect("VOP3P control attached");
        assert_eq!(ctrl.op_sel, 0);
        assert_eq!(ctrl.op_sel_hi, 0b011);
        assert!(!ctrl.clamp);

        // A packed opcode outside the ported table is a NAMED refusal, never a
        // silent mis-decode: v_pk_add_u16 (0x0a, integer) has no lowering.
        let unknown = vop3p_words(0x0a, 2, [256, 257, 0], 0, 0b011, 0, 0, false);
        let mut other = ShaderCode::new();
        other.set_type(ShaderType::Compute);
        let e = shader_parse(0, &[unknown[0], unknown[1], S_ENDPGM], &mut other, true)
            .expect_err("an unported packed opcode must refuse by name");
        assert!(
            format!("{e:?}").contains("vop3p"),
            "the refusal names the vop3p family: {e:?}"
        );
    }

    /// The full VOP3P lowering: every ported opcode translates to assembled,
    /// spirv-val-clean SPIR-V, and the modifier bits reach the emitted body.
    #[test]
    fn vop3p_packed_and_mix_ops_lower_to_valid_spirv() {
        for (name, opcode, type_) in [
            ("v_pk_add_f16", 0x0f, T::VPkAddF16),
            ("v_pk_mul_f16", 0x10, T::VPkMulF16),
            ("v_pk_min_f16", 0x11, T::VPkMinF16),
            ("v_pk_max_f16", 0x12, T::VPkMaxF16),
            ("v_pk_fma_f16", 0x0e, T::VPkFmaF16),
            ("v_fma_mix_f32", 0x20, T::VFmaMixF32),
            ("v_fma_mixlo_f16", 0x21, T::VFmaMixloF16),
            ("v_fma_mixhi_f16", 0x22, T::VFmaMixhiF16),
        ] {
            // Exercise clamp AND a non-default op_sel/neg on every opcode.
            let words = vop3p_words(opcode, 4, [256, 257, 258], 0b001, 0b011, 0b010, 0b100, true);
            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Compute);
            // V_MOV_B32 v0, 0 tail: the generator requires a body of at least
            // two instructions before s_endpgm.
            shader_parse(
                0,
                &[words[0], words[1], V_MOV_V0_0, S_ENDPGM],
                &mut code,
                true,
            )
            .unwrap_or_else(|e| panic!("{name} must parse: {e:?}"));
            assert_eq!(code.get_instructions()[0].type_, type_, "{name}");
            let ctrl = code.get_instructions()[0].vop3p.expect("control");
            assert!(ctrl.clamp, "{name}: clamp bit decoded");
            assert_eq!(ctrl.op_sel, 0b001, "{name}");
            assert_eq!(ctrl.neg_lo, 0b010, "{name}");
            assert_eq!(ctrl.neg_hi, 0b100, "{name}");

            let mut info = ShaderComputeInputInfo::default();
            info.threads_num = [1, 1, 1];
            let source = spirv_generate_source(&code, None, None, Some(&info))
                .unwrap_or_else(|e| panic!("{name} must translate: {e}"));

            // The clamp modifier is applied, not silently dropped (the
            // pre-existing VOP3 bodies refuse clamp by name — a packed op MUST
            // implement it instead: SharpEmu PR #460 `472fc96`).
            assert!(
                source.contains("OpFOrdGreaterThan %bool") && source.contains("OpFOrdLessThan"),
                "{name}: clamp saturation emitted:\n{source}"
            );
            // The negate modifier reached the body.
            assert!(
                source.contains("OpFNegate %float"),
                "{name}: negate modifier emitted:\n{source}"
            );

            let words = spirv_run(&source).unwrap_or_else(|e| panic!("{name} must assemble: {e}"));
            spirv_val_ok(&words, name);
        }
    }

    /// The packed ops compute TWO independent lanes and repack them; the mix
    /// ops compute ONE f32 and (for lo/hi) merge it into one half of vdst,
    /// preserving the other. These shapes are what distinguish a real packed
    /// lowering from a scalar one that happens to assemble.
    #[test]
    fn vop3p_lane_shapes_are_packed_not_scalar() {
        let emit = |opcode: u32, op_sel: u32, op_sel_hi: u32| {
            let words = vop3p_words(opcode, 4, [256, 257, 258], op_sel, op_sel_hi, 0, 0, false);
            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Compute);
            shader_parse(
                0,
                &[words[0], words[1], V_MOV_V0_0, S_ENDPGM],
                &mut code,
                true,
            )
            .expect("parse");
            let mut info = ShaderComputeInputInfo::default();
            info.threads_num = [1, 1, 1];
            spirv_generate_source(&code, None, None, Some(&info)).expect("translate")
        };

        // v_pk_add_f16 with op_sel = 0 (both low halves feed the LOW lane) and
        // op_sel_hi = 0b011 (both high halves feed the HIGH lane): the two
        // lanes must extract DIFFERENT components and repack into one dword.
        let packed = emit(0x0f, 0b000, 0b011);
        assert!(
            packed.contains("OpCompositeExtract %float %hlo0_0_pk 0")
                && packed.contains("OpCompositeExtract %float %hhi0_0_pk 1"),
            "the low lane reads half 0 and the high lane half 1:\n{packed}"
        );
        assert_eq!(
            packed.matches("OpFAdd %float").count(),
            2,
            "one f16 add per packed lane:\n{packed}"
        );
        assert!(
            packed.contains("OpCompositeConstruct %v2float %vloraw_0 %vhiraw_0")
                && packed.contains("%vpu_0 = OpExtInst %uint %GLSL_std_450 PackHalf2x16 %vpk_0"),
            "the two lanes repack into a single dword:\n{packed}"
        );

        // v_fma_mixhi_f16: ONE fma, then the result merges into the HIGH half
        // while the low half of vdst survives.
        let mixhi = emit(0x22, 0b000, 0b000);
        assert_eq!(
            mixhi.matches("GLSL_std_450 Fma").count(),
            1,
            "the mix ops are a single f32 fma, not two lanes:\n{mixhi}"
        );
        assert!(
            mixhi.contains("OpCompositeConstruct %v2float %float_0_000000 %fmix_0"),
            "mixhi packs the result into the HIGH half:\n{mixhi}"
        );
        assert!(
            mixhi.contains("OpBitwiseAnd %uint %mixdu_0 %uint_0x0000ffff"),
            "mixhi preserves the LOW half of vdst:\n{mixhi}"
        );
        // op_sel_hi = 0 means every mix source is read as a full f32, so no
        // half is unpacked at all.
        assert!(
            !mixhi.contains("UnpackHalf2x16"),
            "op_sel_hi = 0 reads full f32 sources:\n{mixhi}"
        );

        // The mirror case: op_sel_hi = 0b101 reads src0 and src2 as f16 halves,
        // src1 as a full f32.
        let mixlo = emit(0x21, 0b100, 0b101);
        assert!(
            mixlo.contains("OpCompositeExtract %float %m0_0_pk 0")
                && mixlo.contains("OpCompositeExtract %float %m2_0_pk 1"),
            "op_sel picks each mix source's half independently:\n{mixlo}"
        );
        assert!(
            mixlo.contains("%m1_0_b = OpBitcast %float %r1_0"),
            "a source with op_sel_hi clear stays a full f32:\n{mixlo}"
        );
        assert!(
            mixlo.contains("OpBitwiseAnd %uint %mixdu_0 %uint_0xffff0000"),
            "mixlo preserves the HIGH half of vdst:\n{mixlo}"
        );
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
    fn gta_readfirstlane_uses_a_real_subgroup_broadcast() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7ED6_0500, // v_readfirstlane_b32 vcc_hi, v0
                0xBF80_0000, // s_nop 0
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse measured GTA readfirstlane");

        let input_info = ShaderComputeInputInfo {
            threads_num: [64, 1, 1],
            ..Default::default()
        };
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile measured GTA readfirstlane");
        assert!(source.contains("OpCapability GroupNonUniformBallot"));
        assert!(source.contains("OpGroupNonUniformBroadcastFirst %uint %uint_3 %t0_0"));
        assert!(source.contains("OpStore %vcc_hi %t_0"));

        let words = spirv_run(&source).expect("assemble measured GTA readfirstlane");
        naga_parse_and_validate(&words, "gta_readfirstlane");
    }

    #[test]
    fn gta_add_co_u32_emits_sum_and_carry() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0xD70F_6A01, // v_add_co_u32 v1, vcc, s12, v6
                0x0002_0C0C,
                0xBF80_0000, // s_nop 0
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse measured GTA carry add");

        let input_info = ShaderComputeInputInfo {
            threads_num: [64, 1, 1],
            ..Default::default()
        };
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile measured GTA carry add");
        assert!(source.contains("OpIAdd %uint"));
        assert!(source.contains("OpULessThan %bool"));
        assert!(source.contains("OpStore %v1"));
        assert!(source.contains("OpStore %vcc_lo"));
        assert!(source.contains("OpStore %vcc_hi %uint_0"));

        let words = spirv_run(&source).expect("assemble measured GTA carry add");
        naga_parse_and_validate(&words, "gta_add_co_u32");
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

    /// ASTRO.BOT's scene-composite compute shader (0x555f4f500) stopped
    /// translating on what the log reported as `op_sel != 0`, but that was the
    /// VOP3B `sdst` field (the carry-out SGPR, VCC) of `v_add_co_ci_u32` being
    /// misread as op_sel. This asserts the real op set that shader carries all
    /// parse and recompile to valid SPIR-V: the VOP2 (0x28) and VOP3B (0x128)
    /// carry-adds collapse to one `VAddCoCiU32`, plus `s_andn1_saveexec_b64`
    /// (SOP1 0x37), `v_xnor_b32` (VOP2 0x1e), and the VOP3-form `v_bcnt_u32_b32`
    /// (0x364). Each was a hard wall on that title before this change.
    #[test]
    fn astro_composite_carry_and_exec_ops_recompile() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x5000_0302, // v_add_co_ci_u32 v0, vcc, v2, v1, vcc     (VOP2 0x28)
                0xD528_6A00,
                0x01AA_0501, // v_add_co_ci_u32 v0, vcc, v1, v2, s? (VOP3B 0x128)
                0xBE8A_376A, // s_andn1_saveexec_b64 s[10:11], vcc       (SOP1 0x37)
                0x3C00_0302, // v_xnor_b32 v0, v2, v1                    (VOP2 0x1e)
                0xD764_0000,
                0x0002_0501, // v_bcnt_u32_b32 v0, v1, v2   (VOP3 0x364)
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse composite carry/exec ops (no phantom op_sel refusal)");

        // The parser must recognise every op, not refuse on the phantom op_sel.
        let types: Vec<_> = code.get_instructions().iter().map(|i| i.type_).collect();
        assert!(types.contains(&T::SAndn1SaveexecB64));
        assert!(types.contains(&T::VXnorB32));
        assert!(types.contains(&T::VBcntU32B32));
        assert_eq!(
            types.iter().filter(|t| **t == T::VAddCoCiU32).count(),
            2,
            "both the VOP2 and VOP3B carry-add encodings resolve to one type"
        );

        let input_info = ShaderComputeInputInfo {
            threads_num: [1, 1, 1],
            ..Default::default()
        };

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile composite carry/exec ops");
        // Carry-out is the unsigned overflow of the two-step add; carry-in and
        // the andn1 exec update both mask/AND. Assert the shapes are present.
        assert!(
            source.contains("OpULessThan %bool"),
            "carry-out overflow test:\n{source}"
        );
        assert!(
            source.contains("OpBitwiseAnd %uint"),
            "carry-in bit extract / andn1 exec mask:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble composite carry/exec ops");
        naga_parse_and_validate(&words, "astro composite carry/exec ops");
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

    /// ASTRO.BOT scene compute uses the VOP3 form of `v_cmp_gt_u64`. The
    /// comparison must consume adjacent low/high dwords and must not require
    /// the optional SPIR-V Int64 capability.
    #[test]
    fn astro_vop3_cmp_gt_u64_compares_high_then_low_dwords() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0xD4E4_006A, // measured VOP3 opcode 0xe4, scalar pair s[106:107]
                0x0002_0501, // v[1:2], v[2:3]
                0xBF80_0000,
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse v_cmp_gt_u64");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VCmpGtU64);
        assert_eq!(inst.src[0].size, 2);
        assert_eq!(inst.src[1].size, 2);
        assert_eq!(inst.dst.size, 2);

        let entry = recomp_func(T::VCmpGtU64, F::SmaskVsrc0Vsrc1).expect("VCmpGtU64 row");
        assert!(matches!(entry.func, RecompileFn::Func(_)));

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile v_cmp_gt_u64");
        assert!(
            source.contains("%u64_hi_gt_0 = OpUGreaterThan"),
            "high compare:\n{source}"
        );
        assert!(
            source.contains("%u64_hi_eq_0 = OpIEqual"),
            "high equality:\n{source}"
        );
        assert!(
            source.contains("%u64_lo_gt_0 = OpUGreaterThan"),
            "low compare:\n{source}"
        );
        assert!(!source.contains("OpTypeInt 64"), "no Int64:\n{source}");
        let words = spirv_run(&source).expect("assemble v_cmp_gt_u64");
        naga_parse_and_validate(&words, "v_cmp_gt_u64");
    }

    /// GFX10 compact VOP2 opcode 0x2b is `v_fmac_f32`, not the legacy
    /// `v_ldexp_f32`. This measured ASTRO.BOT word previously stopped four
    /// compute shaders at their first fused accumulate.
    #[test]
    fn astro_vop2_fmac_f32_uses_existing_fused_accumulate_lowering() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0x5626_26F4, // measured compact v_fmac_f32
                0xBF80_0000,
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse v_fmac_f32");
        assert_eq!(code.get_instructions()[0].type_, T::VMacF32);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile v_fmac_f32");
        assert!(
            source.contains("GLSL_std_450 Fma"),
            "fused accumulate:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble v_fmac_f32");
        naga_parse_and_validate(&words, "v_fmac_f32");
    }

    /// GFX10 compact VOP2 opcode 0x2a is reverse subtract-with-borrow. Four
    /// ASTRO.BOT compute shaders previously stopped at the measured family,
    /// whose raw words begin `0x5400....`.
    #[test]
    fn astro_vop2_subbrev_u32_reverses_operands_and_updates_vcc() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        let subbrev = (0x2au32 << 25) | (3 << 17) | (2 << 9) | 257;
        shader_parse(
            0,
            &[subbrev, 0xBF80_0000, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse v_subbrev_u32");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VSubbrevU32);
        assert_eq!(inst.src[0].register_id, 1);
        assert_eq!(inst.src[1].register_id, 2);
        assert_eq!(inst.src[2].type_, ShaderOperandType::VccLo);
        assert_eq!(inst.dst2.type_, ShaderOperandType::VccLo);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile v_subbrev_u32");
        assert!(
            source.contains("%ttleft_0 = OpLoad %float %v2"),
            "reverse form must load vsrc1 as the left operand:\n{source}"
        );
        assert!(
            source.contains("%ttright_0 = OpLoad %float %v1"),
            "reverse form must load src0 as the right operand:\n{source}"
        );
        assert!(
            source.contains("OpLogicalOr")
                && source.contains("OpStore %vcc_lo")
                && source.contains("OpStore %vcc_hi"),
            "borrow-out must update VCC:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble v_subbrev_u32");
        naga_parse_and_validate(&words, "v_subbrev_u32");

        // Captured ASTRO.BOT VOP2-in-VOP3 form:
        //   v_subbrev_u32 v0, 0, v26, s[4:5] -> borrow-out VCC
        // Bits [14:11] read as op_sel=0xd unless opcode 0x12a is treated as
        // VOP3B, which previously refused all four shaders at this word.
        let mut captured = ShaderCode::new();
        captured.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[0xD52A_6A00, 0x0012_3480, 0xBF80_0000, S_ENDPGM],
            &mut captured,
            true,
        )
        .expect("parse captured VOP3B v_subbrev_u32");
        let inst = &captured.get_instructions()[0];
        assert_eq!(inst.type_, T::VSubbrevU32);
        assert_eq!(inst.dst2.type_, ShaderOperandType::VccLo);
        assert_eq!(inst.src[2].register_id, 4);
        let source = spirv_generate_source(&captured, None, Some(&input_info), None)
            .expect("recompile captured VOP3B v_subbrev_u32");
        let words = spirv_run(&source).expect("assemble captured VOP3B v_subbrev_u32");
        naga_parse_and_validate(&words, "captured_vop3b_v_subbrev_u32");
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
        // The load at byte offset 8 reads EUD dword 2 — declare the
        // extended storage buffer whose descriptor covers it (rebased on
        // the EUD pair: start_register 16 - eud_base 14 = dword 2), so the
        // extended mapping is grounded in a captured descriptor rather
        // than the pre-sentinel silent (0, 0) default.
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 16;
        input_info.bind.storage_buffers.extended[0] = true;

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

    /// Astro Bot vertex shader gap 1: MUBUF `buffer_load_dwordx4` with **offen
    /// only** (idxen=0) — `Vdata4VaddrSvSoffsOffen`. The single vaddr register
    /// is the per-thread byte offset (no vindex), which must add into the byte
    /// address just like the idxen+offen twin.
    #[test]
    fn buffer_load_dwordx4_offen_only_recompiles() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0xE038_1000, // mubuf op 0x0e (load_dwordx4), offen only (bit12=1, bit13=0)
                0x8001_0400, // soffset=0x80(=const 0), srsrc=s4, vdata=v4, vaddr=v0
                0xBF80_0000,
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse offen-only buffer_load_dwordx4");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadDwordX4);
        assert_eq!(inst.format, F::Vdata4VaddrSvSoffsOffen);
        assert_eq!(
            inst.src[0].size, 1,
            "offen-only: single vaddr (voffset) reg"
        );
        assert_eq!(inst.dst.size, 4);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 48;
        input_info.bind.storage_buffers.buffers_num = 1;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile offen-only buffer_load_dwordx4");
        assert_eq!(
            source.matches("%t110_").count(),
            4,
            "four consecutive dword loads:\n{source}"
        );
        assert!(
            source.contains("OpIAdd %int"),
            "the per-thread voffset adds into the address:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble offen-only buffer_load_dwordx4");
    }

    /// Astro Bot vertex shader gap 2 (baseline): a `s_load_dwordx8` whose base
    /// is built PC-relative (`s_getpc_b64 s[0:1]; s_add_u32 s0, 96, s0`) — the
    /// shader loading its own embedded constant table. The base register is NOT
    /// the EUD base, so the recompiler refuses it by name today.
    #[test]
    fn s_load_dwordx8_pc_relative_baseline_refuses() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        code.set_base_address(0x1000);
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SGetpcB64,
            format: F::Sdst2,
            pc: 0,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 0,
                size: 2,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0x1004),
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SAddU32,
            format: F::SVdstSVsrc0SVsrc1,
            pc: 4,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 0,
                size: 1,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(96),
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 0,
                    size: 1,
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            pc: 12,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 8,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 0,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::IntegerInlineConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            pc: 16,
            ..Default::default()
        });

        // EUD present but at a DIFFERENT base (s16), so the PC-relative s0 base
        // is not the EUD base and hits the refusal.
        let mut input_info = ShaderVertexInputInfo::default();
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 16;

        let err = spirv_generate_source(&code, Some(&input_info), None, None)
            .expect_err("PC-relative s_load_dwordx8 must refuse today");
        let msg = format!("{err}");
        assert!(
            msg.contains("src0 is not the EUD base register"),
            "baseline refusal is the EUD-base gate, got: {msg}"
        );
        for detail in [
            "pc=0xc",
            "src0=s0",
            "eud_base=s16",
            "offset=0x0",
            "SGetpcB64",
            "SAddU32",
        ] {
            assert!(msg.contains(detail), "missing {detail} in: {msg}");
        }
    }

    /// Build the measured Astro Bot pattern: `s_getpc_b64 s[0:1];
    /// s_add_u32 s0, 96, s0; s_load_dwordx8 s[8:15], s[0:1], 0; s_endpgm`, with
    /// the getpc following-address materialized at `0x1004` (base 0x1000). The
    /// PC-relative load target is therefore `0x1004 + 96 = 0x1064`.
    fn pc_relative_x8_shader() -> ShaderCode {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        code.set_base_address(0x1000);
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SGetpcB64,
            format: F::Sdst2,
            pc: 0,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 0,
                size: 2,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0x1004),
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SAddU32,
            format: F::SVdstSVsrc0SVsrc1,
            pc: 4,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 0,
                size: 1,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(96),
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 0,
                    size: 1,
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            pc: 12,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 8,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 0,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::IntegerInlineConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            pc: 16,
            ..Default::default()
        });
        code
    }

    /// Astro Bot vertex shader gap 2 (fix): once
    /// `shader_detect_embedded_constant_loads` has captured the shader's own
    /// embedded constant table from guest memory, the PC-relative
    /// `s_load_dwordx8` materializes those eight dwords as SPIR-V constants
    /// straight into `s[8:15]` — no EUD, no refusal.
    #[test]
    fn s_load_dwordx8_pc_relative_reads_embedded_constants() {
        use std::borrow::Cow;

        struct EmbeddedMem {
            base: u64,
            data: Vec<u32>,
        }
        impl crate::shader::analysis::ShaderMemory for EmbeddedMem {
            fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
                let end = self.base + self.data.len() as u64 * 4;
                if addr >= self.base && addr < end && (addr - self.base) % 4 == 0 {
                    return Some(Cow::Borrowed(
                        &self.data[((addr - self.base) / 4) as usize..],
                    ));
                }
                None
            }
        }

        let code = pc_relative_x8_shader();

        // The eight embedded constant dwords live at the PC-relative target
        // (0x1004 + 96 = 0x1064).
        let values: Vec<u32> = (0..8).map(|i| 0xC0DE_0000 + i).collect();
        let mem = EmbeddedMem {
            base: 0x1064,
            data: values.clone(),
        };

        let mut input_info = ShaderVertexInputInfo::default();
        crate::shader::analysis::shader_detect_embedded_constant_loads(
            &code,
            &mem,
            &mut input_info.bind,
        );

        // The capture pass resolved exactly one PC-relative load of 8 dwords.
        let ecl = &input_info.bind.embedded_constant_loads;
        assert_eq!(ecl.loads_num, 1, "one PC-relative embedded load captured");
        assert_eq!(ecl.loads[0].pc, 12);
        assert_eq!(ecl.loads[0].dwords_num, 8);
        assert_eq!(&ecl.loads[0].values[..8], &values[..]);

        let source = spirv_generate_source(&code, Some(&input_info), None, None)
            .expect("recompile PC-relative s_load_dwordx8");

        // Each captured dword is stored straight into its destination SGPR.
        for i in 0..8u32 {
            assert!(
                source.contains(&format!("OpStore %s{} %uint", 8 + i)),
                "s{} must receive an embedded constant:\n{source}",
                8 + i
            );
        }

        // Assembling proves the referenced `%uint_*` constants were declared
        // (an unregistered value would leave `unknown_uint_constant` and fail).
        let words = spirv_run(&source).expect("assemble PC-relative s_load_dwordx8");
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    }

    /// The measured ASTRO.BOT 64-bit PC-relative idiom pairs the low-dword add
    /// with an `s_addc_u32 s(hi), 0, s(hi)` carry. `pc_relative_base_address`
    /// must fold that pair to the right absolute address (here forcing a real
    /// carry across the 32-bit boundary), not bail on the high-dword write.
    #[test]
    fn s_load_dwordx8_pc_relative_handles_64bit_addc_carry() {
        use std::borrow::Cow;

        struct Mem64 {
            base: u64,
            data: Vec<u32>,
        }
        impl crate::shader::analysis::ShaderMemory for Mem64 {
            fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
                let end = self.base + self.data.len() as u64 * 4;
                if addr >= self.base && addr < end && (addr - self.base) % 4 == 0 {
                    return Some(Cow::Borrowed(
                        &self.data[((addr - self.base) / 4) as usize..],
                    ));
                }
                None
            }
        }

        let sgpr = |id, size| ShaderOperand {
            type_: ShaderOperandType::Sgpr,
            register_id: id,
            size,
            ..Default::default()
        };
        let lit = |u| ShaderOperand {
            type_: ShaderOperandType::LiteralConstant,
            constant: crate::shader::types::ShaderConstant::from_u(u),
            ..Default::default()
        };

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        // getpc following-address = 0x5_FFFF_FFF0.
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SGetpcB64,
            format: F::Sdst2,
            pc: 0,
            src_num: 2,
            dst: sgpr(0, 2),
            src: [
                lit(0xFFFF_FFF0),
                lit(0x5),
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        // s_add_u32 s0, 0x20, s0  → low wraps 0xFFFFFFF0+0x20 = 0x10, carry out.
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SAddU32,
            format: F::SVdstSVsrc0SVsrc1,
            pc: 4,
            src_num: 2,
            dst: sgpr(0, 1),
            src: [
                lit(0x20),
                sgpr(0, 1),
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        // s_addc_u32 s1, 0, s1 → hi 5 + carry = 6.
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SAddcU32,
            format: F::SVdstSVsrc0SVsrc1,
            pc: 8,
            src_num: 2,
            dst: sgpr(1, 1),
            src: [
                lit(0),
                sgpr(1, 1),
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            pc: 12,
            src_num: 2,
            dst: sgpr(8, 8),
            src: [
                sgpr(0, 2),
                ShaderOperand {
                    type_: ShaderOperandType::IntegerInlineConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            pc: 16,
            ..Default::default()
        });

        // 0x5_FFFF_FFF0 + 0x20 = 0x6_0000_0010 (carry propagated into the high
        // dword by the u64 add, then the addc s1,0 adds nothing).
        let values: Vec<u32> = (0..8).map(|i| 0xBEEF_0000 + i).collect();
        let mem = Mem64 {
            base: 0x6_0000_0010,
            data: values.clone(),
        };

        let mut input_info = ShaderVertexInputInfo::default();
        crate::shader::analysis::shader_detect_embedded_constant_loads(
            &code,
            &mem,
            &mut input_info.bind,
        );

        let ecl = &input_info.bind.embedded_constant_loads;
        assert_eq!(
            ecl.loads_num, 1,
            "the s_add/s_addc 64-bit pair must resolve, not bail"
        );
        assert_eq!(
            &ecl.loads[0].values[..8],
            &values[..],
            "carry across the 32-bit boundary must land at 0x6_0000_0010"
        );

        let source = spirv_generate_source(&code, Some(&input_info), None, None)
            .expect("recompile 64-bit PC-relative s_load_dwordx8");
        let words = spirv_run(&source).expect("assemble 64-bit PC-relative s_load_dwordx8");
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    }

    /// Astro Bot gap 3 (the real live geometry blocker): the full-screen-triangle
    /// vertex shader builds a V# in `s[0:3]` — base `s[0:1]` PC-relative, words
    /// `s2`/`s3` from immediates — pointing at its own embedded clip-space
    /// vertices, then reads them with `buffer_load_dwordx4 v[0:3], v4, s[0:3], 0
    /// offen` and `buffer_load_dwordx2 v[4:5], v4, s[0:3], 16 offen`. There is
    /// no captured storage buffer, so the recompiler refused (measured 116×
    /// `can't recompile: BufferLoadDwordX4 [Vdata4VaddrSvSoffsOffen]` per live
    /// frame). After `shader_detect_embedded_buffer_fetch` snapshots the window,
    /// both loads recompile as a select over the baked vertex data.
    #[test]
    fn offen_buffer_load_through_in_shader_vsharp_reads_embedded_verts() {
        use std::borrow::Cow;

        struct Mem {
            base: u64,
            data: Vec<u32>,
        }
        impl crate::shader::analysis::ShaderMemory for Mem {
            fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
                let end = self.base + self.data.len() as u64 * 4;
                if addr >= self.base && addr < end && (addr - self.base) % 4 == 0 {
                    return Some(Cow::Borrowed(
                        &self.data[((addr - self.base) / 4) as usize..],
                    ));
                }
                None
            }
        }

        let sgpr = |id, size| ShaderOperand {
            type_: ShaderOperandType::Sgpr,
            register_id: id,
            size,
            ..Default::default()
        };
        let vgpr = |id, size| ShaderOperand {
            type_: ShaderOperandType::Vgpr,
            register_id: id,
            size,
            ..Default::default()
        };
        let lit = |u| ShaderOperand {
            type_: ShaderOperandType::LiteralConstant,
            constant: crate::shader::types::ShaderConstant::from_u(u),
            ..Default::default()
        };
        let inl = |u| ShaderOperand {
            type_: ShaderOperandType::IntegerInlineConstant,
            constant: crate::shader::types::ShaderConstant::from_u(u),
            ..Default::default()
        };

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        code.set_base_address(0x1000);
        // s_getpc_b64 s[0:1] — following-address materialized at 0x1004.
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SGetpcB64,
            format: F::Sdst2,
            pc: 0,
            src_num: 2,
            dst: sgpr(0, 2),
            src: [
                lit(0x1004),
                lit(0),
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        // s_add_u32 s0, 96, s0 ; s_addc_u32 s1, 0, s1  → base = 0x1064.
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SAddU32,
            format: F::SVdstSVsrc0SVsrc1,
            pc: 4,
            src_num: 2,
            dst: sgpr(0, 1),
            src: [
                lit(96),
                sgpr(0, 1),
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SAddcU32,
            format: F::SVdstSVsrc0SVsrc1,
            pc: 8,
            src_num: 2,
            dst: sgpr(1, 1),
            src: [
                lit(0),
                sgpr(1, 1),
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        });
        // buffer_load_dwordx4 v[0:3], v4, s[0:3], 0 offen   (position)
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::BufferLoadDwordX4,
            format: F::Vdata4VaddrSvSoffsOffen,
            pc: 12,
            src_num: 3,
            dst: vgpr(0, 4),
            src: [vgpr(4, 1), sgpr(0, 4), inl(0), ShaderOperand::default()],
            ..Default::default()
        });
        // buffer_load_dwordx2 v[4:5], v4, s[0:3], 16 offen  (uv)
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::BufferLoadDwordX2,
            format: F::Vdata2VaddrSvSoffsOffen,
            pc: 16,
            src_num: 3,
            dst: vgpr(4, 2),
            src: [vgpr(4, 1), sgpr(0, 4), inl(16), ShaderOperand::default()],
            ..Default::default()
        });
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            pc: 20,
            ..Default::default()
        });

        // The embedded full-screen-triangle vertex table at base 0x1064:
        // three verts (pos.xyzw, uv.xy) — the measured clip-space triangle
        // (-1,-1),(3,-1),(-1,3).
        let f = |x: f32| x.to_bits();
        let verts: Vec<u32> = vec![
            f(-1.0),
            f(-1.0),
            f(0.0),
            f(1.0),
            f(0.0),
            f(0.0), // v0
            f(3.0),
            f(-1.0),
            f(0.0),
            f(1.0),
            f(2.0),
            f(0.0), // v1
            f(-1.0),
            f(3.0),
            f(0.0),
            f(1.0),
            f(0.0),
            f(2.0), // v2
        ];
        let mem = Mem {
            base: 0x1064,
            data: verts.clone(),
        };

        let mut input_info = ShaderVertexInputInfo::default();
        crate::shader::analysis::shader_detect_embedded_buffer_fetch(
            &code,
            &mem,
            &mut input_info.bind,
        );

        // Both offen loads through the in-shader V# were captured.
        let ebf = &input_info.bind.embedded_buffer_fetches;
        assert_eq!(ebf.loads_num, 2, "both offen buffer loads captured");
        let x4 = ebf.find(12).expect("x4 load captured");
        assert_eq!((x4.dwords_num, x4.inst_offset), (4, 0));
        assert_eq!(&x4.window[..6], &verts[..6], "vert0 pos+uv snapshot");
        let x2 = ebf.find(16).expect("x2 load captured");
        assert_eq!((x2.dwords_num, x2.inst_offset), (2, 16));

        // No storage buffer is bound — the pre-fix path would refuse here.
        assert_eq!(input_info.bind.storage_buffers.buffers_num, 0);

        let source = spirv_generate_source(&code, Some(&input_info), None, None)
            .expect("recompile offen buffer load through in-shader V#");

        // The loads lower to a select over the baked window, not a %buf read.
        assert!(
            source.contains("OpSelect %uint"),
            "the embedded window is selected by the runtime offset:\n{source}"
        );
        assert!(
            !source.contains("%buffer_load_float1"),
            "must NOT go through the storage-buffer helper (no buffer bound)"
        );
        for d in 0..4u32 {
            assert!(
                source.contains(&format!("OpStore %v{d} ")),
                "position dword v{d} written:\n{source}"
            );
        }

        let words = spirv_run(&source).expect("assemble in-shader-V# offen loads");
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
    }
    /// A guest cube descriptor emits an arrayed 2D image and samples with the
    /// 3-component `(s,t,face)` coordinate — measured on Minecraft's skybox
    /// PS (type 11 T#, `ImageSample [Vdata4Vaddr3StSsDmaskF]`).
    #[test]
    fn guest_cube_texture_emits_2d_array_image_and_face_coords() {
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
            code.get_instructions_mut().push(sample);
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
            source.contains("OpTypeImage %float 2D 0 1"),
            "guest cube faces must bind as an arrayed 2D image:\n{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v3float"),
            "3-component (s,t,face) coordinate:\n{source}"
        );
        assert!(
            source.contains("OpFSub %float %t39_0 %float_1_000000")
                && source.contains("OpFSub %float %t40_0 %float_1_000000"),
            "guest cube S/T must be rebased from the PS5 [1,2] convention \
             to Vulkan 2D-array normalized [0,1] coordinates:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble cube sample");
    }

    /// A 3D-volume-bound CS emits `OpTypeImage %float 3D` and samples with a
    /// 3-component coordinate — measured on ASTRO.BOT's froxel/LUT volumes
    /// (type 10 T#, 240x135x64, format 71) via `image_sample_lz`.
    #[test]
    fn volume_texture_emits_3d_image_and_vec3_coords() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let sample = ShaderInstruction {
            type_: T::ImageSampleLz,
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
            code.get_instructions_mut().push(sample);
        }
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 10 << 28; // 3D
        input_info.bind.textures2d.binding_sampled_index = 0;
        input_info.bind.textures2d.binding_storage_index = 1;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        input_info.bind.samplers.binding_index = 2;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile 3D volume sample");
        assert!(
            source.contains("OpTypeImage %float 3D"),
            "the 3D image type:\n{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v3float"),
            "3-component volume coordinate:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble 3D volume sample");
        naga_parse_and_validate(&words, "3D volume sample");
    }

    /// A 2DArray-bound CS emits `OpTypeImage %float 2D 0 1 0 1` (arrayed) and
    /// samples with a 3-component (u, v, layer) coordinate — measured on
    /// ASTRO.BOT's 1536x1536x3 array (type 13 T#, format 7, tile 24; 57
    /// dispatches/run). Uses the new `image_sample_lz` dmask 0x3 body, so
    /// this also covers the two-channel LOD-zero sample (58 dispatches/run).
    #[test]
    fn array_texture_emits_arrayed_2d_image_and_vec3_coords() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let sample = ShaderInstruction {
            type_: T::ImageSampleLz,
            format: F::Vdata2Vaddr3StSsDmask3,
            src_num: 3,
            dst: ShaderOperand {
                type_: ShaderOperandType::Vgpr,
                register_id: 2,
                size: 2,
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
            code.get_instructions_mut().push(sample);
        }
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 13 << 28; // 2DArray
        input_info.bind.textures2d.binding_sampled_index = 0;
        input_info.bind.textures2d.binding_storage_index = 1;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        input_info.bind.samplers.binding_index = 2;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile 2DArray sample");
        assert!(
            source.contains("OpTypeImage %float 2D 0 1 0 1 Unknown"),
            "the arrayed 2D image type:\n{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v3float"),
            "3-component (u, v, layer) coordinate:\n{source}"
        );
        // The dmask3 body stores exactly channels 0 and 1.
        assert!(source.contains("%uint_0\n"), "{source}");
        assert!(
            !source.contains("%temp_v4float %uint_2"),
            "a dmask3 sample must stop at two channels:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble 2DArray sample");
        naga_parse_and_validate(&words, "2DArray sample");
    }

    /// A shader mixing 2D and 3D sampled textures declares one image array per
    /// Dim (`%textures2D_S_2D` at binding 0, `%textures2D_S_3D` at binding 1)
    /// and routes each sample to the array matching its own T#'s Dim. Measured
    /// on ASTRO.BOT's fullscreen composite/read pass, which samples the scene
    /// HDR 2D targets alongside a 3D LUT/froxel volume — the single-array path
    /// refused it, gating the presented frame.
    #[test]
    fn mixed_2d_and_3d_sampled_textures_emit_two_arrays_and_both_sample() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // Sample T#0 (2D, s0..s7) and T#1 (3D, s16..s23), both through S#0
        // (s8..s11) — an `image_sample_lz` dmask 0x3 each, the body that
        // adapts its coordinate width to the descriptor's Dim.
        let sample = |t_reg: i32| ShaderInstruction {
            type_: T::ImageSampleLz,
            format: F::Vdata2Vaddr3StSsDmask3,
            src_num: 3,
            dst: ShaderOperand {
                type_: ShaderOperandType::Vgpr,
                register_id: 2,
                size: 2,
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
                    register_id: t_reg,
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
        code.get_instructions_mut().push(sample(0));
        code.get_instructions_mut().push(sample(16));
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 128;
        input_info.bind.textures2d.textures_num = 2;
        input_info.bind.textures2d.textures2d_sampled_num = 2;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28; // 2D
        input_info.bind.textures2d.desc[0].start_register = 0;
        input_info.bind.textures2d.desc[1].texture.fields[3] |= 10 << 28; // 3D
        input_info.bind.textures2d.desc[1].start_register = 16;
        // Two sampled Dims => two sampled bindings (0, 1); storage 2; sampler 3.
        input_info.bind.textures2d.binding_sampled_index = 0;
        input_info.bind.textures2d.binding_storage_index = 2;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        input_info.bind.samplers.binding_index = 3;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile mixed 2D+3D sample");

        // One image type + array + variable per Dim.
        assert!(
            source.contains("%ImageS_2D = OpTypeImage %float 2D 0 0 0 1 Unknown"),
            "2D image type:\n{source}"
        );
        assert!(
            source.contains("%ImageS_3D = OpTypeImage %float 3D 0 0 0 1 Unknown"),
            "3D image type:\n{source}"
        );
        assert!(
            source.contains("%textures2D_S_2D = OpVariable"),
            "2D array variable:\n{source}"
        );
        assert!(
            source.contains("%textures2D_S_3D = OpVariable"),
            "3D array variable:\n{source}"
        );
        // Each array lands at its own binding (2D=0, 3D=1).
        assert!(
            source.contains("OpDecorate %textures2D_S_2D Binding 0"),
            "2D binding:\n{source}"
        );
        assert!(
            source.contains("OpDecorate %textures2D_S_3D Binding 1"),
            "3D binding:\n{source}"
        );
        // The 2D sample routes to `%SampledImage_2D` with a 2-component coord,
        // the 3D sample to `%SampledImage_3D` with a 3-component coord.
        assert!(
            source.contains("%SampledImage_2D") && source.contains("%SampledImage_3D"),
            "both sampled-image types referenced:\n{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v2float")
                && source.contains("OpCompositeConstruct %v3float"),
            "2D sample uses v2 coords and 3D sample uses v3 coords:\n{source}"
        );

        let words = spirv_run(&source).expect("assemble mixed 2D+3D sample");
        naga_parse_and_validate(&words, "mixed 2D+3D sample");
    }

    /// The full mixed-dim pipeline through the REAL binding allocator: a 2D +
    /// 3D sampled pair PLUS a storage image, with `shader_calc_binding_indices`
    /// (not hand-set indices) reserving one binding per present sampled dim
    /// and shifting storage/samplers past them. RED on the single-array path:
    /// the allocator gave storage `sampled + 1` (aliasing the 3D array) and
    /// translation refused "mixed sampled texture dims" outright.
    #[test]
    fn mixed_dim_bind_calc_shifts_storage_and_routes_by_descriptor() {
        use crate::shader::analysis::shader_calc_binding_indices;
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let sample = |t_reg: i32, dst_reg: i32| ShaderInstruction {
            type_: T::ImageSampleLz,
            format: F::Vdata4Vaddr3StSsDmaskF,
            src_num: 3,
            dst: ShaderOperand {
                type_: ShaderOperandType::Vgpr,
                register_id: dst_reg,
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
                    register_id: t_reg,
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
        code.get_instructions_mut().push(sample(0, 2));
        code.get_instructions_mut().push(sample(12, 10));
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.textures2d.textures_num = 3;
        input_info.bind.textures2d.textures2d_sampled_num = 2;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        // T# 0: 2D (type 9) at s0..s7; T# 1: 3D (type 10) at s12..s19.
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28;
        input_info.bind.textures2d.desc[0].start_register = 0;
        input_info.bind.textures2d.desc[1].texture.fields[3] |= 10 << 28;
        input_info.bind.textures2d.desc[1].start_register = 12;
        // T# 2: a storage (RW) image — proves the storage binding shifts
        // past BOTH sampled dims instead of aliasing the second one.
        input_info.bind.textures2d.desc[2].texture.fields[3] |= 9 << 28;
        input_info.bind.textures2d.desc[2].start_register = 20;
        input_info.bind.textures2d.desc[2].textures2d_without_sampler = true;
        input_info.bind.textures2d.desc[2].usage =
            crate::shader::resources::ShaderTextureUsage::ReadWrite;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        shader_calc_binding_indices(&mut input_info.bind);

        // One binding per PRESENT sampled dim; storage/samplers shift.
        assert_eq!(
            input_info.bind.textures2d.binding_sampled_index, 0,
            "first sampled dim keeps the base binding"
        );
        assert_eq!(
            input_info.bind.textures2d.binding_storage_index, 2,
            "storage images shift past BOTH sampled dims"
        );
        assert_eq!(
            input_info.bind.samplers.binding_index, 3,
            "samplers follow the shifted storage binding"
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("mixed 2D+3D sampled bind translates (per-dim arrays)");

        // Every array lands at the allocator's binding, no aliasing.
        assert!(
            source.contains("OpDecorate %textures2D_S_2D Binding 0"),
            "2D array binding decoration:\n{source}"
        );
        assert!(
            source.contains("OpDecorate %textures2D_S_3D Binding 1"),
            "3D array binding decoration:\n{source}"
        );
        assert!(
            source.contains("OpDecorate %textures2D_L Binding 2"),
            "shifted storage binding decoration:\n{source}"
        );
        assert!(
            source.contains("OpDecorate %samplers Binding 3"),
            "shifted sampler binding decoration:\n{source}"
        );
        // Each sample's OpAccessChain routes into ITS T#'s own array — an
        // AccessChain match cannot be satisfied by the declaration, the
        // decorations, or the OpEntryPoint interface list.
        assert!(
            source.contains("OpAccessChain %_ptr_UniformConstant_ImageS_2D %textures2D_S_2D"),
            "a body indexes the 2D array:\n{source}"
        );
        assert!(
            source.contains("OpAccessChain %_ptr_UniformConstant_ImageS_3D %textures2D_S_3D"),
            "a body indexes the 3D array:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble mixed-dim sample");
        naga_parse_and_validate(&words, "mixed-dim sample with storage shift");
    }

    /// ASTRO.BOT acceptance (SharpEmu PR #587 "support Gen5 flat memory and 3D
    /// images"): ONE compute shader that writes a 3D `Rgba16f` froxel volume
    /// AND a 2D `Rgba16f` target. The historical shader-wide single storage
    /// key refused this by name —
    /// `storage_texture_dim_format: not supported: mixed storage image
    /// dims/formats in one shader ((Three, "Rgba16f") vs (Two, Rgba16f))` —
    /// 20 refusals per measured run, each followed by a host crash.
    ///
    /// The bar is structural, not textual: the module must declare TWO
    /// DISTINCT `OpTypeImage` storage types (`Dim3D` and `Dim2D`, both
    /// `Rgba16f`), each at its own binding, each indexed by the body that
    /// writes it — and it must pass real spirv-val, not just assemble.
    #[test]
    fn astro_mixed_2d_and_3d_rgba16f_storage_images_translate_to_two_image_types() {
        use crate::shader::analysis::shader_calc_binding_indices;

        // Two `image_store` MIMG instructions, one per storage T#.
        let store_inst = || {
            let mut store = ShaderCode::new();
            store.set_type(ShaderType::Compute);
            shader_parse(0, &[0xF020_0100, 0x0060_0800, S_ENDPGM], &mut store, true)
                .expect("parse image_store");
            let inst = store.get_instructions()[0];
            assert_eq!(inst.type_, T::ImageStore);
            inst
        };
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let mut to_2d = store_inst();
        to_2d.src[1].register_id = 0;
        let mut to_3d = store_inst();
        to_3d.src[1].register_id = 12;
        code.get_instructions_mut().push(to_2d);
        code.get_instructions_mut().push(to_3d);
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut info = ShaderComputeInputInfo::default();
        info.threads_num = [1, 1, 1];
        info.bind.push_constant_size = 128;
        info.bind.textures2d.textures_num = 2;
        info.bind.textures2d.textures2d_storage_num = 2;
        // Both RW T#s carry FORMAT 71 (16_16_16_16 FLOAT = Rgba16f):
        // `format()` is `fields[1] >> 20 & 0x1ff`, `type_()` is
        // `fields[3] >> 28 & 0xf`.
        for (slot, type_, start) in [(0usize, 9u32, 0i32), (1, 10, 12)] {
            let d = &mut info.bind.textures2d.desc[slot];
            d.texture.fields[1] |= 71 << 20;
            d.texture.fields[3] |= type_ << 28;
            d.start_register = start;
            d.textures2d_without_sampler = true;
            d.usage = crate::shader::resources::ShaderTextureUsage::ReadWrite;
        }
        // Real allocator, not hand-set indices: one binding per present key.
        shader_calc_binding_indices(&mut info.bind);

        let keys = storage_keys_present(&info.bind);
        assert_eq!(
            keys,
            vec![
                (SampledDim::Two, StorageFormat::Rgba16f),
                (SampledDim::Three, StorageFormat::Rgba16f),
            ],
            "the two present storage keys, in canonical Dim-major order"
        );

        let source = spirv_generate_source(&code, None, None, Some(&info))
            .expect("mixed 2D+3D Rgba16f storage images translate (per-key arrays)");

        // Two distinct storage image TYPES, not one shader-wide type.
        assert!(
            source.contains("%ImageL_2D_16F = OpTypeImage %float 2D 0 0 0 2 Rgba16f"),
            "2D Rgba16f storage image type:\n{source}"
        );
        assert!(
            source.contains("%ImageL_3D_16F = OpTypeImage %float 3D 0 0 0 2 Rgba16f"),
            "3D Rgba16f storage image type:\n{source}"
        );
        // Distinct bindings, one per present key, in `storage_key_ordinal`
        // order and starting at the allocator's `binding_storage_index`.
        let base = info.bind.textures2d.binding_storage_index;
        assert!(
            source.contains(&format!("OpDecorate %textures2D_L_2D_16F Binding {base}"))
                && source.contains(&format!(
                    "OpDecorate %textures2D_L_3D_16F Binding {}",
                    base + 1
                )),
            "one binding per present storage key from {base}:\n{source}"
        );
        // Each write indexes ITS OWN key's array — a declaration alone would
        // satisfy the two asserts above while every store still aliased one
        // array.
        assert!(
            source
                .contains("OpAccessChain %_ptr_UniformConstant_ImageL_2D_16F %textures2D_L_2D_16F"),
            "the 2D store indexes the 2D array:\n{source}"
        );
        assert!(
            source
                .contains("OpAccessChain %_ptr_UniformConstant_ImageL_3D_16F %textures2D_L_3D_16F"),
            "the 3D store indexes the 3D array:\n{source}"
        );
        // The 3D write must carry a THREE-component integer coordinate, or
        // every Z slice collapses onto one plane (the #587 symptom).
        assert!(
            source.contains("OpCompositeConstruct %v3uint"),
            "the 3D store builds a v3uint coordinate:\n{source}"
        );

        // Gate on REAL spirv-val (Khronos, Vulkan 1.3 — the same validator
        // the raeen-gpu runtime uses). naga cannot serve as the gate here: its
        // SPIR-V front end rejects *any* `OpImageWrite` storage module this
        // generator produces with `InvalidImage`, homogeneous 2D-only
        // included, so it is a known false negative for this whole path (the
        // sibling of the `InvalidArrayBaseType` carve-out in
        // `naga_parse_and_validate`) and would hide rather than prove the fix.
        let words = spirv_run(&source).expect("assemble mixed 2D+3D storage images");
        spirv_val_ok(&words, "mixed 2D+3D Rgba16f storage images");
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
    fn v_cmpx_integer_blocks_use_their_signedness() {
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
            (T::VCmpxLeU32, "OpULessThanEqual"),
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
            // The RDNA2 scene-composite batch (measured ASTRO.BOT
            // 0x555f4f500 / 0x500564500 divergent-flow prologues): XNOR, the
            // add-with-carry VOP2/VOP3B pair, and the ANDN1 save-exec sibling.
            (T::VXnorB32, F::SVdstSVsrc0SVsrc1),
            (T::VAddCoCiU32, F::VdstSdst2Vsrc0Vsrc1Smask2),
            (T::SAndn1SaveexecB64, F::Sdst2Ssrc02),
            (T::VMadU64U32, F::VdstSdst2Vsrc0Vsrc1Smask2),
        ] {
            let entry = recomp_func(ty, fmt).unwrap_or_else(|| panic!("{ty:?} row missing"));
            assert!(
                matches!(entry.func, RecompileFn::Func(_)),
                "{ty:?} must be implemented, not NI"
            );
        }
    }

    /// Regression guard for the VOP3B `op_sel` misdecode. A VOP3B opcode's
    /// carry-out SGPR occupies bits [14:8]; whenever that SGPR index is >= 8 it
    /// sets bit 11, which overlaps the VOP3A `op_sel` field [14:11]. Before the
    /// `is_vop3b_opcode` gate the decoder read that bit as `op_sel != 0` and
    /// refused the whole shader (the "VOP3 op_sel != 0" wall). Here
    /// `v_add_co_ci_u32` (opcode 0x128) with carry-out s8 must parse to
    /// `VAddCoCiU32`, not error.
    #[test]
    fn vop3b_carry_out_sgpr_is_not_misread_as_op_sel() {
        // VOP3 prefix 0b110101<<26 | opcode 0x128<<16 | sdst s8<<8 | vdst v2.
        // s8 (=0b0001000) sets bit 11 of b0 -> op_sel would read 1 if ungated.
        let b0 = (0b110101u32 << 26) | (0x128 << 16) | (8 << 8) | 2;
        // src0 = v3 (256+3), src1 = v4 (256+4)<<9, src2 (carry-in) = s0.
        let b1 = 259u32 | (260u32 << 9);
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(0, &[b0, b1, S_ENDPGM], &mut code, true)
            .expect("VOP3B v_add_co_ci_u32 must parse, not be refused as op_sel != 0");
        let inst = &code.get_instructions()[0];
        assert!(
            matches!(inst.type_, ShaderInstructionType::VAddCoCiU32),
            "expected VAddCoCiU32, got {:?}",
            inst.type_
        );
        // Carry-out landed in dst2 as the SGPR pair s8, not swallowed by op_sel.
        assert_eq!(inst.dst2.register_id, 8, "carry-out must decode to s8");
    }

    /// `v_mad_u64_u32` (VOP3B 0x176) must parse and recompile to the
    /// mul-hi/lo + add-with-carry idiom, and the emitted SPIR-V must assemble
    /// and pass naga validation (guards the hand-written 64-bit-add body).
    #[test]
    fn v_mad_u64_u32_recompiles_to_mul_add_with_carry() {
        // VOP3 prefix | opcode 0x176 | sdst s10 (carry-out) | vdst v0.
        let b0 = (0b110101u32 << 26) | (0x176 << 16) | (10 << 8);
        // src0 = v2, src1 = v3<<9, src2 = v4<<18 (the 64-bit addend pair).
        let b1 = 258u32 | (259u32 << 9) | (260u32 << 18);
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // `s_endpgm` must sit at instruction index >= 2, so the mad is followed
        // by two harmless scalar fillers (reused from the passing scalar test).
        shader_parse(
            0,
            &[b0, b1, 0xBE80_1F00, 0x9935_806B, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse v_mad_u64_u32");
        let inst = &code.get_instructions()[0];
        assert!(
            matches!(inst.type_, ShaderInstructionType::VMadU64U32),
            "expected VMadU64U32, got {:?}",
            inst.type_
        );

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile v_mad_u64_u32");
        assert!(source.contains("%mul_lo_uint"), "{source}");
        assert!(source.contains("%mul_hi_uint"), "{source}");
        // The 64-bit accumulate uses explicit unsigned-overflow carries.
        assert!(source.contains("OpULessThan %bool"), "{source}");
        // Assemble-validate only: the module goes through the `mul_lo_uint`/
        // `mul_hi_uint` helpers, whose `OpUMulExtended` the Vulkan driver
        // accepts (it backs every shipping VMulLo/Hi shader) but the naga
        // test-validator rejects — so naga is the wrong bar for this opcode,
        // exactly as for the `%addc` family.
        let _words = spirv_run(&source).expect("assemble v_mad_u64_u32");
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
        // srsrc reg 4 in the encoding — the captured T# must match it or the
        // descriptor guard refuses (dynamic-image-descriptor).
        input_info.bind.textures2d.desc[0].start_register = 4;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_get_resinfo");
        assert!(source.contains("OpImageQuerySizeLod %v2int"), "{source}");
        let words = spirv_run(&source).expect("assemble image_get_resinfo");
        naga_parse_and_validate(&words, "image_get_resinfo");
    }

    /// `image_get_resinfo` against a non-2D descriptor used to refuse, which
    /// failed the WHOLE shader recompile and dropped the draw — and it caught
    /// 2DArray as well as 3D, i.e. every cube T# (type 11/13 lower to
    /// 2DArray). `OpImageQuerySizeLod`'s result width is fixed by the image's
    /// dim, so the query type must follow the descriptor: 3D and 2DArray yield
    /// `%v3int`, plain 2D `%v2int`. Only x and y are stored either way.
    #[test]
    fn image_get_resinfo_sizes_the_query_from_the_descriptor_dim() {
        // (texture type nibble, expected OpImageQuerySizeLod result type)
        for (texture_type, want_ty) in [(9u32, "%v2int"), (10, "%v3int"), (13, "%v3int")] {
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
            input_info.bind.textures2d.desc[0].start_register = 4;
            input_info.bind.textures2d.desc[0].texture.fields[3] |= texture_type << 28;

            let source = spirv_generate_source(&code, None, None, Some(&input_info))
                .unwrap_or_else(|e| {
                    panic!("texture type {texture_type} must recompile, got {e:?}")
                });
            assert!(
                source.contains(&format!("OpImageQuerySizeLod {want_ty}")),
                "texture type {texture_type} must query {want_ty}\n{source}"
            );
            // Still an xy query: the third component is never stored.
            assert!(
                source.contains("OpCompositeExtract %int %t5_0 0")
                    && source.contains("OpCompositeExtract %int %t5_0 1"),
                "texture type {texture_type} must still store x and y\n{source}"
            );
            let words = spirv_run(&source)
                .unwrap_or_else(|e| panic!("texture type {texture_type} must assemble: {e:?}"));
            naga_parse_and_validate(&words, "image_get_resinfo_non_2d");
        }
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

    /// image_sample_c_lz on a 2DArray depth texture — the Vaddr3 MIMG carries
    /// only {dref, s, t} (no slice VGPR), so the sample is at array layer 0 and
    /// the coordinate is v3 (s, t, 0.0). Measured on ASTRO.BOT composite read
    /// shader 0x500564500 (dim TwoArray, Vdata1Vaddr3StSsDmask1). 2D output is
    /// unchanged (v2 coord); 3D/Cube stay refused.
    #[test]
    fn image_sample_c_lz_2darray_samples_layer_zero_with_v3_coord() {
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
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 13 << 28; // type = 2DArray
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 12;
        input_info.bind.samplers.binding_index = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile 2DArray image_sample_c_lz");
        assert!(
            source.contains("OpCompositeConstruct %v3float"),
            "2DArray sample needs a v3 (s, t, layer) coordinate:\n{source}"
        );
        assert!(
            source.contains("%float_0_000000"),
            "array layer 0:\n{source}"
        );
        assert!(
            source.contains("OpImageSampleExplicitLod %v4float"),
            "{source}"
        );
        let words = spirv_run(&source).expect("assemble 2DArray image_sample_c_lz");
        naga_parse_and_validate(&words, "2DArray image_sample_c_lz");
    }

    #[test]
    fn astro_buffer_store_format_xyzw_recompiles() {
        // Measured raw 0xE01C2000: buffer_store_format_xyzw with idxen.
        // 925 dispatches / 30s failed on this single opcode.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xE01C_2000, 0x8001_0400, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse buffer_store_format_xyzw");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.storage_buffers.buffers_num = 1;
        // The instruction's V# is s[4:7], and the store's element format now
        // comes from that descriptor at translate time — so the fixture has to
        // say which register the binding covers and what it holds. Unified 77 =
        // (dfmt 14, nfmt 7) = 32_32_32_32_FLOAT, the one format
        // `%tbuffer_store_format_xyzw` serves.
        input_info.bind.storage_buffers.start_register[0] = 4;
        input_info.bind.storage_buffers.buffers[0].fields = [0, 16 << 16, 256, 77 << 12];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile buffer_store_format_xyzw");
        assert!(source.contains("%tbuffer_store_format_xyzw"), "{source}");
        assert!(source.contains("%buffer_store_float4"), "{source}");
        assert!(
            source.contains("OpStore %temp_int_5 %int_119"),
            "the descriptor's unified 77 must reach the helper as PACKED 119, \
             not as the raw field:\n{source}"
        );
        assert!(
            !source.contains("%mbf_f0_"),
            "the runtime `(dword3 >> 12) & 0x7f` extraction must be gone:\n{source}"
        );
        // Stores are exec-guarded like the Kyty store bodies.
        assert!(source.contains("%exec_lo_u_0"), "{source}");
        let words = spirv_run(&source).expect("assemble buffer_store_format_xyzw");
        naga_parse_and_validate(&words, "buffer_store_format_xyzw");
    }

    #[test]
    fn astro_mubuf_store_dword_without_index_uses_zero_index() {
        // buffer_store_dword (0x1c) with idxen==0/offen==0 — previously a
        // parse-level "idxen == 0" rejection.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xE070_0000, 0x8001_0400, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse address-only buffer store");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.storage_buffers.buffers_num = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile address-only buffer store");
        assert!(source.contains("OpStore %temp_int_1 %int_0"), "{source}");
        assert!(source.contains("%buffer_store_float1"), "{source}");
        let words = spirv_run(&source).expect("assemble address-only buffer store");
        naga_parse_and_validate(&words, "address-only buffer store");
    }

    #[test]
    fn astro_v_add3_u32_recompiles_as_two_adds() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xD76D_0001, 0x040A_0300, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse v_add3_u32");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source =
            spirv_generate_source(&code, None, None, Some(&input_info)).expect("recompile v_add3");
        assert!(source.contains("%ta_0 = OpIAdd %uint"), "{source}");
        assert!(source.contains("%t_0 = OpIAdd %uint %ta_0"), "{source}");
        let words = spirv_run(&source).expect("assemble v_add3_u32");
        naga_parse_and_validate(&words, "v_add3_u32");
    }

    #[test]
    fn astro_image_sample_lz_dmask2_selects_channel_y() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF09C_0200, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_sample_lz dmask2");
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
            .expect("recompile image_sample_lz dmask2");
        assert!(
            source.contains("OpImageSampleExplicitLod %v4float"),
            "{source}"
        );
        assert!(
            source.contains("OpAccessChain %_ptr_Function_float %temp_v4float %uint_1"),
            "{source}"
        );
        let words = spirv_run(&source).expect("assemble image_sample_lz dmask2");
        naga_parse_and_validate(&words, "image_sample_lz dmask2");
    }

    #[test]
    fn astro_pixel_sample_dmask2_and_lzo_dmask1_recompile() {
        for (raw, expected_op, expected_channel, label) in [
            (0xF080_0200, "OpImageSampleImplicitLod", 1, "sample dmask2"),
            (
                0xF080_0109,
                "OpImageSampleImplicitLod",
                0,
                "GTA image_sample_a dmask1",
            ),
            (
                0xF0DC_0100,
                "OpImageSampleExplicitLod",
                0,
                "sample_lz_o dmask1",
            ),
            (
                0xF0DC_0200,
                "OpImageSampleExplicitLod",
                1,
                "sample_lz_o dmask2",
            ),
        ] {
            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Pixel);
            shader_parse(
                0,
                &[raw, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
                &mut code,
                true,
            )
            .unwrap_or_else(|e| panic!("parse {label}: {e}"));
            let mut input_info = ShaderPixelInputInfo::default();
            input_info.target_output_mode[0] = 4;
            input_info.bind.push_constant_size = 64;
            input_info.bind.textures2d.textures_num = 1;
            input_info.bind.textures2d.textures2d_sampled_num = 1;
            input_info.bind.textures2d.desc[0].start_register = 4;
            input_info.bind.samplers.samplers_num = 1;
            input_info.bind.samplers.start_register[0] = 12;
            input_info.bind.samplers.binding_index = 1;
            let source = spirv_generate_source(&code, None, Some(&input_info), None)
                .unwrap_or_else(|e| panic!("recompile {label}: {e}"));
            assert!(source.contains(expected_op), "{label}:\n{source}");
            assert!(
                source.contains(&format!(
                    "OpAccessChain %_ptr_Function_float %temp_v4float %uint_{expected_channel}"
                )),
                "{label} must select the measured output channel:\n{source}"
            );
            let words = spirv_run(&source).unwrap_or_else(|e| panic!("assemble {label}: {e}"));
            naga_parse_and_validate(&words, label);
        }
    }

    #[test]
    fn astro_image_load_dmask3_fetches_two_channels() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF000_0300, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_load dmask3");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_load dmask3");
        assert!(source.contains("OpImageFetch %v4float"), "{source}");
        assert!(
            source.contains("OpAccessChain %_ptr_Function_float %temp_v4float %uint_1"),
            "{source}"
        );
        // Assemble-only: naga's SPIR-V frontend rejects OpImageFetch through
        // the Kyty sampled-image declaration (`InvalidImage`) — the same
        // false-negative class its dmask1/7/F siblings ship under; real
        // Vulkan accepts it. The real-driver gate is the --run-eboot render.
        let _ = spirv_run(&source).expect("assemble image_load dmask3");
    }

    #[test]
    fn astro_image_load_dmask_c_fetches_z_then_w() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF000_0C00, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_load dmask0xc");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_load dmask0xc");
        assert!(source.contains("OpImageFetch %v4float"), "{source}");
        assert!(
            source.contains("OpAccessChain %_ptr_Function_float %temp_v4float %uint_2"),
            "first packed destination must receive Z:\n{source}"
        );
        assert!(
            source.contains("OpAccessChain %_ptr_Function_float %temp_v4float %uint_3"),
            "second packed destination must receive W:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble image_load dmask0xc");
    }

    /// image_load of a 3D texture must build a 3-component (x, y, z) integer
    /// coordinate and OpImageFetch it — the non-2D texel-fetch path measured
    /// on ASTRO.BOT's composite read shader (0x500566b00). 2D output is
    /// unchanged; Cube stays a named refusal (SPIR-V forbids cube fetch).
    #[test]
    fn image_load_3d_texture_fetches_with_v3uint_coords() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF000_0300, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_load");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 10 << 28; // type = 3D
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile 3D image_load");
        assert!(source.contains("OpImageFetch %v4float"), "{source}");
        assert!(
            source.contains("OpCompositeConstruct %v3uint"),
            "the 3D fetch coordinate must carry z:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble 3D image_load");
    }

    /// A T# whose unified format is an INTEGER class (5 = R8 UINT — SharpEmu
    /// Gfx10UnifiedFormat: dataFormat 1, numFormat 4) must be sampled through
    /// a UINT-typed `OpTypeImage`, with the raw texel bits bitcast into the
    /// float-typed register model (SharpEmu parity: Gen5SpirvTranslator keeps
    /// Uint sample results as raw bits, never ConvertUToF). The measured
    /// failure without this: VUID-vkCmdDispatch-format-07753 on ASTRO.BOT —
    /// the view is `VK_FORMAT_R8_UINT` (draw_translate, unified 5) while the
    /// shader declared `OpTypeImage %float`, undefined behavior with
    /// validation off.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn r8_uint_sampled_texture_samples_raw_bits_via_uint_image() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF09C_0200, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_sample_lz dmask2");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.textures2d.desc[0].texture.fields[1] |= 5 << 20; // unified 5 = R8 UINT
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 12;
        input_info.bind.samplers.binding_index = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile R8_UINT image_sample_lz");
        assert!(
            source.contains("%ImageS = OpTypeImage %uint 2D 0 0 0 1 Unknown"),
            "a UINT-class T# needs a UINT sampled image type:\n{source}"
        );
        assert!(
            source.contains("OpImageSampleExplicitLod %v4uint"),
            "the sample result type must match the image's sampled type:\n{source}"
        );
        assert!(
            source.contains("OpBitcast %v4float"),
            "raw bits must be bitcast into the float register model:\n{source}"
        );
        assert!(
            !source.contains("OpImageSampleExplicitLod %v4float"),
            "no float-typed sample of a UINT image may remain:\n{source}"
        );
        assert!(
            !source.contains("OpConvertUToF"),
            "SharpEmu parity: raw bits, never a numeric conversion:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble R8_UINT image_sample_lz");
        spirv_val_ok(&words, "R8_UINT image_sample_lz");
    }

    /// FLOAT-class regression pin: a T# with unified format 71 (16_16_16_16
    /// FLOAT — numFormat 7) keeps the legacy float-typed image and sample,
    /// with no bitcast inserted.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn float_class_sampled_texture_still_emits_float_image_type() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF09C_0200, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_sample_lz dmask2");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.textures2d.desc[0].texture.fields[1] |= 71 << 20; // 16x4 FLOAT
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 12;
        input_info.bind.samplers.binding_index = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile float image_sample_lz");
        assert!(
            source.contains("%ImageS = OpTypeImage %float 2D 0 0 0 1 Unknown"),
            "{source}"
        );
        assert!(
            source.contains("OpImageSampleExplicitLod %v4float"),
            "{source}"
        );
        assert!(
            !source.contains("OpBitcast %v4float"),
            "a float-class sample needs no result bitcast:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble float image_sample_lz");
        spirv_val_ok(&words, "float image_sample_lz");
    }

    /// A shader sampling a FLOAT-class 2D texture AND a UINT-class 2D texture
    /// splits them into per-(Dim, class) arrays — `%textures2D_S_2D` (float)
    /// and `%textures2D_S_2D_U` (uint) at consecutive bindings — and routes
    /// each sample to its own T#'s array with the matching result typing.
    /// Same grouping machinery as the mixed-Dim split, keyed on numeric class.
    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn mixed_float_and_uint_2d_textures_split_into_per_class_arrays() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // Sample T#0 (2D float, s0..s7) and T#1 (2D uint, s16..s23), both
        // through S#0 (s8..s11) — an `image_sample_lz` dmask 0x3 each.
        let sample = |t_reg: i32| ShaderInstruction {
            type_: T::ImageSampleLz,
            format: F::Vdata2Vaddr3StSsDmask3,
            src_num: 3,
            dst: ShaderOperand {
                type_: ShaderOperandType::Vgpr,
                register_id: 2,
                size: 2,
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
                    register_id: t_reg,
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
        code.get_instructions_mut().push(sample(0));
        code.get_instructions_mut().push(sample(16));
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 128;
        input_info.bind.textures2d.textures_num = 2;
        input_info.bind.textures2d.textures2d_sampled_num = 2;
        // T#0: 2D float (unified 56 = 8_8_8_8 UNORM); T#1: 2D uint (unified 5).
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28;
        input_info.bind.textures2d.desc[0].texture.fields[1] |= 56 << 20;
        input_info.bind.textures2d.desc[0].start_register = 0;
        input_info.bind.textures2d.desc[1].texture.fields[3] |= 9 << 28;
        input_info.bind.textures2d.desc[1].texture.fields[1] |= 5 << 20;
        input_info.bind.textures2d.desc[1].start_register = 16;
        // Two sampled classes => two sampled bindings (0, 1); storage 2;
        // sampler 3 — the same reservation rule as mixed Dims.
        input_info.bind.textures2d.binding_sampled_index = 0;
        input_info.bind.textures2d.binding_storage_index = 2;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        input_info.bind.samplers.binding_index = 3;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile mixed float+uint sample");

        assert!(
            source.contains("%ImageS_2D = OpTypeImage %float 2D 0 0 0 1 Unknown"),
            "float image type:\n{source}"
        );
        assert!(
            source.contains("%ImageS_2D_U = OpTypeImage %uint 2D 0 0 0 1 Unknown"),
            "uint image type:\n{source}"
        );
        assert!(
            source.contains("OpDecorate %textures2D_S_2D Binding 0"),
            "float array binding:\n{source}"
        );
        assert!(
            source.contains("OpDecorate %textures2D_S_2D_U Binding 1"),
            "uint array binding:\n{source}"
        );
        assert!(
            source.contains("OpAccessChain %_ptr_UniformConstant_ImageS_2D %textures2D_S_2D"),
            "a body indexes the float array:\n{source}"
        );
        assert!(
            source.contains("OpAccessChain %_ptr_UniformConstant_ImageS_2D_U %textures2D_S_2D_U"),
            "a body indexes the uint array:\n{source}"
        );
        assert!(
            source.contains("OpImageSampleExplicitLod %v4float"),
            "the float T#'s sample keeps its float result:\n{source}"
        );
        assert!(
            source.contains("OpImageSampleExplicitLod %v4uint")
                && source.contains("OpBitcast %v4float"),
            "the uint T#'s sample retypes and bitcasts:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble mixed float+uint sample");
        spirv_val_ok(&words, "mixed float+uint sample");
    }

    /// `v_min_u32` (VOP2 0x13) must recompile to a GLSL `UMin` and validate —
    /// the measured wall on ASTRO.BOT scene CS 0x555f4f500 after the v_mad fix.
    #[test]
    fn v_min_u32_recompiles_to_umin() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let min = ShaderInstruction {
            type_: T::VMinU32,
            format: F::SVdstSVsrc0SVsrc1,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Vgpr,
                register_id: 2,
                size: 1,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Vgpr,
                    register_id: 0,
                    size: 1,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::Vgpr,
                    register_id: 1,
                    size: 1,
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        // Two ops so s_endpgm lands at instruction index >= 2.
        for _ in 0..2 {
            code.get_instructions_mut().push(min);
        }
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile v_min_u32");
        assert!(
            source.contains("OpExtInst %uint %GLSL_std_450 UMin"),
            "{source}"
        );
        let words = spirv_run(&source).expect("assemble v_min_u32");
        naga_parse_and_validate(&words, "v_min_u32");
    }

    /// The composite-frontier batch rows must all be wired (implemented, not
    /// NI): the integer min/max quartet, the four flexible-addressing
    /// `buffer_store_dwordx2` formats, and `v_bfe_u32`. Measured on ASTRO.BOT
    /// scene shaders 0x555f4f500 / 0x500757800 / 0x500690400.
    #[test]
    fn composite_frontier_rows_are_wired() {
        let rows: &[(T, F)] = &[
            (T::VMinU32, F::SVdstSVsrc0SVsrc1),
            (T::VMaxU32, F::SVdstSVsrc0SVsrc1),
            (T::VMinI32, F::SVdstSVsrc0SVsrc1),
            (T::VMaxI32, F::SVdstSVsrc0SVsrc1),
            (T::VBfeU32, F::VdstVsrc0Vsrc1Vsrc2),
            (T::BufferStoreDwordX2, F::Vdata2VaddrSvSoffsIdxen),
            (T::BufferStoreDwordX2, F::Vdata2Vaddr2SvSoffsOffenIdxen),
            (T::BufferStoreDwordX2, F::Vdata2SvSoffs),
            (T::BufferStoreDwordX2, F::Vdata2VaddrSvSoffsOffen),
        ];
        for (ty, fmt) in rows {
            let entry = recomp_func(*ty, *fmt).unwrap_or_else(|| panic!("{ty:?} row missing"));
            assert!(
                matches!(entry.func, RecompileFn::Func(_)),
                "{ty:?} must be implemented, not NI"
            );
        }
    }

    #[test]
    fn astro_image_store_dmask1_writes_single_channel() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF020_0100, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_store dmask1");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.textures2d.desc[0].textures2d_without_sampler = true;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_store dmask1");
        assert!(source.contains("OpImageWrite"), "{source}");
        assert!(
            source.contains("%float_0_000000 %float_0_000000 %float_0_000000"),
            "{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v2uint %t69_0 %t71_0"),
            "type-9 storage images keep x/y coordinates:\n{source}"
        );
        // Assemble-only: naga rejects the format-less storage-image write
        // (`InvalidImage`) that real Vulkan accepts via
        // shaderStorageImageWriteWithoutFormat — the ImageStore dmaskF
        // sibling ships under the same false-negative class.
        let _ = spirv_run(&source).expect("assemble image_store dmask1");
    }

    /// The measured ASTRO.BOT UAV volume (round 7, 58 dispatches/run): a T#
    /// of type 10 (3D) and format 71 (16_16_16_16 FLOAT). `%ImageL` must
    /// declare `3D ... Rgba16f` and the store must build a v3uint texel
    /// coordinate from the three Vaddr VGPRs.
    #[test]
    fn astro_image_store_3d_rgba16f_uses_v3uint_coords() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF020_0100, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_store dmask1");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.textures2d.desc[0].textures2d_without_sampler = true;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 10 << 28; // type = 3D
        input_info.bind.textures2d.desc[0].texture.fields[1] |= 71 << 20; // 16_16_16_16 FLOAT
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile 3D storage image_store");
        assert!(
            source.contains("OpTypeImage %float 3D 0 0 0 2 Rgba16f"),
            "{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v3uint"),
            "the texel coordinate must carry z:\n{source}"
        );
        // Assemble-only, matching the 2D dmask1 sibling (naga rejects the
        // storage-image write shape real Vulkan accepts).
        let _ = spirv_run(&source).expect("assemble 3D storage image_store");
    }

    /// Minecraft's panorama builder uses DIM_2D and selects one face through
    /// T#.BASE_ARRAY. The type-13 view remains arrayed, but the SPIR-V layer
    /// coordinate must be zero instead of loading the unrelated vaddr+2.
    #[test]
    fn minecraft_dim2d_store_to_array_view_synthesizes_zero_layer() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF020_0108, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_store dmask1");
        // Minecraft binds sampled s0, storage s24, then sampled s24. The last
        // descriptor therefore overwrites the shared s24 local during the
        // prolog; the store must use the storage descriptor resolved by the
        // guard, not reload that ambiguous SGPR.
        code.get_instructions_mut()[0].src[1].register_id = 24;
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.textures2d.textures_num = 3;
        input_info.bind.textures2d.textures2d_sampled_num = 2;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 0;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28;
        input_info.bind.textures2d.desc[1].start_register = 24;
        input_info.bind.textures2d.desc[1].textures2d_without_sampler = true;
        input_info.bind.textures2d.desc[1].texture.fields[3] |= 13 << 28;
        input_info.bind.textures2d.desc[2].start_register = 24;
        input_info.bind.textures2d.desc[2].texture.fields[3] |= 9 << 28;
        crate::shader::analysis::shader_calc_binding_indices(&mut input_info.bind);
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile array storage image_store");
        assert!(
            source.contains("OpTypeImage %float 2D 0 1 0 2 Rgba8"),
            "{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v3uint %t69_0 %t71_0 %uint_0"),
            "DIM_2D must address layer zero within the BASE_ARRAY view:\n{source}"
        );
        assert!(
            !source.contains("OpLoad %float %v2"),
            "vaddr+2 is not part of DIM_2D:\n{source}"
        );
        assert!(
            source.contains("OpAccessChain %_ptr_UniformConstant_ImageL %textures2D_L %uint_0"),
            "the storage T# is %textures2D_L[0], even though sampled s24 seeds \
             the shared local last:\n{source}"
        );
        assert!(
            !source.contains("OpLoad %uint %s24"),
            "the store must not reload the class-ambiguous s24 local:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble array storage image_store");
    }

    #[test]
    fn minecraft_nsa_image_load_uses_explicit_y_vgpr() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF000_0F0A, 0x0000_0003, 0x0000_0000, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse Minecraft NSA image_load");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 32;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 0;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 13 << 28;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile Minecraft NSA image_load");
        assert!(
            source.contains("OpLoad %float %v0"),
            "NSA byte zero explicitly names v0 as the Y coordinate:\n{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v3uint %t69_0 %t71_0 %uint_0"),
            "DIM_2D uses x/y plus layer zero in the array view:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble NSA image_load");
    }

    #[test]
    fn image_store_dim2darray_keeps_vgpr_layer_coordinate() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF020_0118, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse DIM_2D_ARRAY image_store");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.textures2d.desc[0].textures2d_without_sampler = true;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 13 << 28;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile DIM_2D_ARRAY image_store");
        assert!(
            source.contains("OpLoad %float %v2"),
            "DIM_2D_ARRAY must load vaddr+2 as the layer:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble array storage image_store");
    }

    /// Live ASTRO.BOT table-1 UAV: type 8 (1D, represented as height-1 2D)
    /// with unified format 77 (32_32_32_32 FLOAT). Its SPIR-V declaration
    /// must be Rgba32f, not the old four-byte Rgba8 fallback.
    #[test]
    fn astro_image_store_type8_rgba32f_uses_2d_rgba32f() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF020_0100, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_store dmask1");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.textures2d.desc[0].textures2d_without_sampler = true;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 8 << 28;
        input_info.bind.textures2d.desc[0].texture.fields[1] |= 77 << 20;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile type-8 RGBA32F storage image_store");
        assert!(
            source.contains("OpTypeImage %float 2D 0 0 0 2 Rgba32f"),
            "{source}"
        );
        assert!(
            source.contains("OpCompositeConstruct %v2uint %t69_0 %uint_0"),
            "type-8 storage images must synthesize y=0:\n{source}"
        );
        assert!(
            !source.contains("OpCompositeConstruct %v2uint %t69_0 %t71_0"),
            "type-8 coordinates must not consume vaddr.y:\n{source}"
        );
        assert!(
            !source.contains("%t70_0 = OpLoad"),
            "type-8 coordinates must not even read the unrelated vaddr.y register:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble type-8 RGBA32F image_store");
    }

    #[test]
    fn astro_exp_pos1_is_accepted_and_dropped() {
        // A VS whose pos1 export (clip distance per PA_CL_VS_OUT_CNTL) used
        // to fail the whole shader — 632 failures / 30s. The export parses,
        // the recompile row drops it, and gl_Position (pos0) still lands.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        shader_parse(
            0,
            &[
                0x7E00_02FF,
                0x3F80_0000,
                0x7E02_0280,
                0x1004_0300,
                0xF800_00D4, // exp pos1 en=0x4 (measured shape)
                0x0302_0100,
                0xF800_08CF, // exp pos0 done
                0x0302_0100,
                0xF800_020F, // exp param0
                0x0302_0100,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse VS with pos1 export");

        let entry = recomp_func(T::Exp, F::Pos1Vsrc0Vsrc1Vsrc2Vsrc3).expect("pos1 row");
        assert!(matches!(entry.func, RecompileFn::Func(_)));

        let source = spirv_generate_source(
            &code,
            Some(&ShaderVertexInputInfo {
                export_count: 1,
                ..Default::default()
            }),
            None,
            None,
        )
        .expect("recompile VS with pos1 export");
        assert!(source.contains("%int_per_vertex_0"), "{source}");
        let words = spirv_run(&source).expect("assemble VS with pos1 export");
        naga_parse_and_validate(&words, "vs with pos1 export");
    }

    #[test]
    fn astro_mubuf_flexible_rows_are_wired() {
        for ty in [
            T::BufferLoadDword,
            T::BufferStoreDword,
            T::BufferLoadFormatX,
            T::BufferStoreFormatX,
        ] {
            for fmt in [
                F::Vdata1SvSoffs,
                F::Vdata1VaddrSvSoffsOffen,
                F::Vdata1Vaddr2SvSoffsOffenIdxen,
            ] {
                let entry =
                    recomp_func(ty, fmt).unwrap_or_else(|| panic!("{ty:?}/{fmt:?} row missing"));
                assert!(
                    matches!(entry.func, RecompileFn::Func(_)),
                    "{ty:?}/{fmt:?} must be implemented"
                );
            }
        }
        for fmt in [
            F::Vdata4VaddrSvSoffsIdxen,
            F::Vdata4Vaddr2SvSoffsOffenIdxen,
            F::Vdata4SvSoffs,
            F::Vdata4VaddrSvSoffsOffen,
        ] {
            let entry = recomp_func(T::BufferStoreFormatXyzw, fmt)
                .unwrap_or_else(|| panic!("store xyzw {fmt:?} row missing"));
            assert!(matches!(entry.func, RecompileFn::Func(_)));
        }
        // The staged (1,0) store-format rows are wired now too.
        for (ty, fmt) in [
            (T::BufferStoreFormatX, F::Vdata1VaddrSvSoffsIdxen),
            (T::BufferStoreFormatXy, F::Vdata2VaddrSvSoffsIdxen),
        ] {
            let entry = recomp_func(ty, fmt).expect("row");
            assert!(matches!(entry.func, RecompileFn::Func(_)));
        }
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
    fn gfx10_vcmpx_preserves_vcc_and_intersects_current_exec() {
        // Minecraft's panorama/cubemap copy shader loads {width,height} into
        // VCC, then uses two consecutive GFX10 VCMPX comparisons. The first
        // reads VCC_HI and the second reads VCC_LO. GFX10 VCMPX is EXEC-only:
        // writing VCC after the first comparison turns the second bound into
        // the float bit-pattern 0/1 and leaves no useful lanes.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7c28_046b, // v_cmpx_gt_f32 exec, vcc_hi, v2
                0x7c28_026a, // v_cmpx_gt_f32 exec, vcc_lo, v1
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse consecutive GFX10 VCMPX");
        assert!(
            code.get_instructions()[..2]
                .iter()
                .all(|inst| inst.dst.type_ == ShaderOperandType::ExecLo),
            "the Gen5 decoder must route VCMPX to EXEC, not VCC"
        );

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile consecutive GFX10 VCMPX");
        assert!(
            source.contains("OpLoad %uint %vcc_hi")
                && source.contains("OpLoad %uint %vcc_lo")
                && source.matches("OpBitcast %float").count() >= 2,
            "both preserved bounds must remain readable:\n{source}"
        );
        assert!(
            !source.contains("OpStore %vcc_lo %t6_") && !source.contains("OpStore %vcc_hi %uint_0"),
            "GFX10 VCMPX must not overwrite VCC:\n{source}"
        );
        assert!(
            source.matches("OpLogicalAnd %bool").count() >= 2,
            "each comparison must intersect its predicate with current EXEC:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble GFX10 VCMPX");
        naga_parse_and_validate(&words, "gfx10_vcmpx_exec_only");
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

    /// SOP2 `0x2e`/`0x2f`/`0x30` = `s_lshl{1,2,3}_add_u32`.
    ///
    /// Identity from two independent references that agree row-for-row:
    /// KytyPS5 `src/graphics/shader/recompiler/ScalarAluOps.cpp` L26-28 and
    /// SharpEmu `src/SharpEmu.ShaderCompiler/Gen5ShaderTranslator.cs` L804-807.
    /// `0x30` is ASTRO.BOT's measured `parse: unknown sop2 opcode: 0x30`
    /// (92 occurrences, baseline `1acd114`, stage `rendering`, 36 flips).
    ///
    /// Encoding (SOP2): `10` | opcode[29:23] | sdst[22:16] | ssrc1[15:8] |
    /// ssrc0[7:0]; `ssrc1 == 0xff` selects the following literal dword. The
    /// words below are hand-encoded from that layout, cross-checked against the
    /// already-wired `0x31` word `0x98eb_ff6a` measured in Minecraft.
    #[test]
    fn s_lshl_n_add_u32_decodes_every_shift_and_lowers_through_lshl_add() {
        for (opcode, expect, shift_id) in [
            (0x2eu32, T::SLshl1AddU32, "%uint_1"),
            (0x2fu32, T::SLshl2AddU32, "%uint_2"),
            (0x30u32, T::SLshl3AddU32, "%uint_3"),
            (0x31u32, T::SLshl4AddU32, "%uint_4"),
        ] {
            // s_lshl<N>_add_u32 s8, s4, 0x00000040
            let word = 0x8000_0000 | (opcode << 23) | (8 << 16) | (0xff << 8) | 4;

            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Vertex);
            shader_parse(
                0,
                &[
                    word,
                    0x0000_0040,
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
            .unwrap_or_else(|e| panic!("parse sop2 {opcode:#04x}: {e:?}"));

            let inst = &code.get_instructions()[0];
            assert_eq!(inst.type_, expect, "sop2 {opcode:#04x}");
            assert_eq!(inst.format, F::SVdstSVsrc0SVsrc1, "sop2 {opcode:#04x}");
            assert_eq!(inst.dst.type_, ShaderOperandType::Sgpr);
            assert_eq!(inst.dst.register_id, 8);
            assert_eq!(inst.src[0].type_, ShaderOperandType::Sgpr);
            assert_eq!(inst.src[0].register_id, 4);
            assert_eq!(inst.src[1].constant.u, 0x40);

            // The row exists, carries the 33-bit carry-out SCC rule, and is
            // implemented (not a `NotImplemented` placeholder).
            let entry = recomp_func(expect, F::SVdstSVsrc0SVsrc1)
                .unwrap_or_else(|| panic!("{expect:?} row"));
            assert!(matches!(entry.func, RecompileFn::Func(_)), "{expect:?}");
            assert_eq!(entry.scc_check, SccCheck::CarryOut, "{expect:?}");

            let source = spirv_generate_source(
                &code,
                Some(&ShaderVertexInputInfo {
                    export_count: 1,
                    ..Default::default()
                }),
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("recompile sop2 {opcode:#04x}: {e:?}"));
            // The shared helper is emitted, called with THIS shift, and the
            // carry reaches SCC.
            assert!(
                source.contains(&format!(
                    "OpFunctionCall %v2uint %lshl_add %t0_0 %t1_0 {shift_id}"
                )),
                "sop2 {opcode:#04x} must lower through %lshl_add with {shift_id}"
            );
            assert!(source.contains("OpStore %scc %carry_0"));

            let words =
                spirv_run(&source).unwrap_or_else(|e| panic!("assemble sop2 {opcode:#04x}: {e:?}"));
            spirv_val_ok(&words, "s_lshl_n_add_u32");
        }
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

    #[test]
    fn vertex_passthrough_matches_declared_attribute_width() {
        let vec3 = vs_passthrough_source(7, 3, SampledClass::Float).expect("vec3 passthrough");
        assert!(vec3.contains("%p0_7 = OpLoad %v3float %attr0"), "{vec3}");
        assert!(
            vec3.contains("OpCompositeConstruct %v4float %px_7 %py_7 %pz_7 %float_1_000000"),
            "{vec3}"
        );

        let vec4 = vs_passthrough_source(9, 4, SampledClass::Float).expect("vec4 passthrough");
        assert!(vec4.contains("%pv_9 = OpLoad %v4float %attr0"), "{vec4}");
        assert!(
            !vec4.contains("OpLoad %v3float"),
            "a vec4 input must not be loaded through a vec3 pointer:\n{vec4}"
        );
        assert!(vec4.contains("OpStore %pa_9 %pv_9"), "{vec4}");
    }

    #[test]
    fn uint16_vertex_fetch_preserves_raw_guest_vgpr_bits() {
        // Minecraft's model V# uses unified format 11 = (FMT_16, UINT) for
        // its bone index. The guest shader consumes that VGPR with integer
        // bit operations, so a numeric float conversion (5 -> 5.0f bits)
        // selects a bogus matrix. Vulkan also requires the interface type to
        // agree with R16_UINT. Load uint, then bitcast once into Raeen's
        // float-backed VGPR representation.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        let mut fetch = ShaderInstruction {
            type_: T::FetchX,
            format: F::Vdata1VaddrSvSoffsIdxen,
            src_num: 3,
            ..Default::default()
        };
        fetch.dst = ShaderOperand {
            type_: ShaderOperandType::Vgpr,
            register_id: 4,
            size: 1,
            ..Default::default()
        };
        fetch.src[2] = ShaderOperand {
            type_: ShaderOperandType::IntegerInlineConstant,
            constant: crate::shader::types::ShaderConstant::from_i(1),
            size: 1,
            ..Default::default()
        };
        code.get_instructions_mut().push(fetch);

        let mut info = ShaderVertexInputInfo {
            resources_num: 1,
            fetch_embedded: true,
            gs_prolog: true,
            ..Default::default()
        };
        info.resources[0].fields[3] = 11 << 12;
        info.resources_dst[0].semantic = 1;
        info.resources_dst[0].register_start = 13;
        info.resources_dst[0].registers_num = 1;

        let source = spirv_generate_source(&code, Some(&info), None, None).unwrap();
        assert!(
            source.contains("%attr0 = OpVariable %_ptr_Input_uint Input"),
            "format 11 must expose a uint Vulkan interface:\n{source}"
        );
        assert!(
            source.contains("OpLoad %uint %attr0") && source.contains("OpBitcast %float %t1_1_0"),
            "the uint fetch must preserve raw bits in the float-backed VGPR:\n{source}"
        );

        // GTA V uses the same raw integer semantics with unified format 5 =
        // (FMT_8, UINT). The Vulkan input is R8_UINT, but the guest VGPR still
        // receives the raw integer bits rather than a numeric float convert.
        info.resources[0].fields[3] = 5 << 12;
        let source = spirv_generate_source(&code, Some(&info), None, None).unwrap();
        assert!(
            source.contains("%attr0 = OpVariable %_ptr_Input_uint Input"),
            "format 5 must expose a uint Vulkan interface:\n{source}"
        );
        assert!(
            source.contains("OpLoad %uint %attr0") && source.contains("OpBitcast %float %t1_1_0"),
            "the GTA R8_UINT fetch must preserve raw bits:\n{source}"
        );

        // This synthetic Fetch* omits the scalar prolog registers a complete
        // shader carries. Source-level assertions are intentional here; the
        // live Minecraft verification exercises the assembled Vulkan module.
    }

    /// GTA V's measured blocker: `registers_num = 2` over unified format 5 =
    /// `(FMT_8, UINT)`. Two RAW INTEGER components — a width the raw-integer
    /// path did not cover ("invalid registers_num/input format: 2/5"), which
    /// refused the whole vertex shader.
    ///
    /// The declaration, the fetch load and the bitcast must all agree on the
    /// same width: a `%v2uint` variable read through a `%v2float` pointer, or
    /// a scalar `OpBitcast` of a vector, are both invalid SPIR-V. The
    /// assembled-and-validated module lives in
    /// `tests/gcn_to_spirv.rs::gta_two_channel_uint_vertex_attribute_translates_to_validated_spirv`.
    #[test]
    fn two_channel_uint_vertex_fetch_loads_a_uint_vector() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        let mut fetch = ShaderInstruction {
            type_: T::FetchXy,
            format: F::Vdata2VaddrSvSoffsIdxen,
            src_num: 3,
            ..Default::default()
        };
        fetch.dst = ShaderOperand {
            type_: ShaderOperandType::Vgpr,
            register_id: 2,
            size: 2,
            ..Default::default()
        };
        fetch.src[2] = ShaderOperand {
            type_: ShaderOperandType::IntegerInlineConstant,
            constant: crate::shader::types::ShaderConstant::from_i(0),
            size: 1,
            ..Default::default()
        };
        code.get_instructions_mut().push(fetch);

        let mut info = ShaderVertexInputInfo {
            resources_num: 1,
            fetch_embedded: true,
            gs_prolog: true,
            ..Default::default()
        };
        info.resources[0].fields[3] = 5 << 12; // unified format 5 = (FMT_8, UINT)
        info.resources_dst[0].semantic = 0;
        info.resources_dst[0].register_start = 2;
        info.resources_dst[0].registers_num = 2;

        let source = spirv_generate_source(&code, Some(&info), None, None)
            .expect("2 raw-integer components must translate");
        assert!(
            source.contains("%attr0 = OpVariable %_ptr_Input_v2uint Input"),
            "two uint components must declare a v2uint interface:\n{source}"
        );
        assert!(
            source.contains("OpLoad %v2uint %attr0"),
            "the load must use the declared width:\n{source}"
        );
        assert!(
            source.contains("OpBitcast %v2float %t1_0_0"),
            "the bitcast must be componentwise over the same width:\n{source}"
        );
        assert!(
            source.contains("OpStore %temp_v2float %t1_f_0_0"),
            "the reinterpreted vector feeds the vec2 fetch helper:\n{source}"
        );
        assert!(
            source.contains("OpFunctionCall %void %fetch_f1_f1_vf2_ %v2 %v3 %temp_v2float"),
            "both channels must reach the guest VGPRs:\n{source}"
        );
    }

    /// Every (component count, numeric class) pair the shared resolver claims
    /// support for must name a type the SPIR-V prelude actually declares — the
    /// declaration, the load and the pointer are three separate strings, and a
    /// typo in any of them assembles to "id is used but never defined" only at
    /// runtime, on one title's one attribute layout.
    #[test]
    fn every_supported_vertex_input_pair_names_declared_types() {
        let types_block = super::super::spirv::spirv_generate_source(
            &{
                let mut c = ShaderCode::new();
                c.set_type(ShaderType::Vertex);
                c
            },
            Some(&ShaderVertexInputInfo::default()),
            None,
            None,
        )
        .expect("empty vs");
        for n in 1..=4 {
            for class in [SampledClass::Float, SampledClass::Uint, SampledClass::Sint] {
                let t = vertex_input_types(n, class)
                    .unwrap_or_else(|| panic!("{n} components of {class:?} must be supported"));
                assert!(
                    types_block.contains(&format!("{} = OpTypePointer Input", t.ptr_type)),
                    "{n}/{class:?} declares {} but the types block has no such pointer",
                    t.ptr_type
                );
                assert_eq!(
                    t.bitcast,
                    class != SampledClass::Float,
                    "only the raw integer classes are reinterpreted, never converted"
                );
            }
        }
    }

    /// An unsupported width must name the exact pair (not just the count) and
    /// be counted, so a log line identifies which attribute of which title is
    /// still refused instead of reporting an anonymous whole-shader drop.
    #[test]
    fn unsupported_vertex_input_pair_is_named_and_counted() {
        let before = super::super::spirv::vertex_input_pair_skips();
        assert!(vertex_input_types(5, SampledClass::Uint).is_none());
        assert!(vertex_input_types(0, SampledClass::Float).is_none());
        assert!(
            super::super::spirv::vertex_input_pair_skips() >= before + 2,
            "each refused pair must be counted in the shader diagnostics"
        );

        let err = vs_passthrough_source(0, 5, SampledClass::Uint).expect_err("5 components");
        let msg = format!("{err}");
        assert!(
            msg.contains("5 components") && msg.contains("Uint"),
            "the refusal must name the pair, got: {msg}"
        );
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

    /// The measured ASTRO.BOT PS blocker (463 skipped draws): an UNCOMPRESSED
    /// MRT0 export with a partial channel mask (`en=0x3`, done=1, compr=0,
    /// vm=1). The enabled channels load their VGPRs; the disabled ones carry
    /// the GCN default (0, 0, 1) for (b, a).
    #[test]
    fn astro_exp_mrt0_uncompressed_partial_en_recompiles() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0
                0x7E02_0280, // v_mov_b32 v1, 0
                0xF800_1803, // exp mrt0 v0, v1 (en=0x3 compr=0 vm=1 done=1)
                0x0000_0100,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse uncompressed partial-en mrt0 export");

        let inst = &code.get_instructions()[2];
        assert_eq!(inst.type_, T::Exp);
        assert_eq!(inst.format, F::Mrt0Vsrc0Vsrc1Vsrc2Vsrc3VmDone);
        assert_eq!(inst.export_enable, 0x3);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 2; // SPI_SHADER_32_GR
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile uncompressed partial-en mrt0 export");
        assert!(
            source.contains(
                "OpCompositeConstruct %v4float %t0_2 %t1_2 %float_0_000000 %float_1_000000"
            ),
            "disabled b/a channels default to (0, 1):\n{source}"
        );
        assert!(source.contains("OpStore %outColor"), "{source}");
        let words = spirv_run(&source).expect("assemble uncompressed mrt0 export");
        naga_parse_and_validate(&words, "exp mrt0 compr=0 en=3");
    }

    /// GTA V emits a compressed FP16 MRT0 export with only RG enabled. The
    /// second packed source is a don't-care operand; BA must use the GCN
    /// defaults (0, 1) without loading an unwritten VGPR.
    #[test]
    fn gta_exp_mrt0_compressed_partial_en_recompiles() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0
                0xF800_1C03, // exp mrt0 v0, v1 (en=0x3 compr=1 vm=1 done=1)
                0x0000_0100,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse GTA compressed partial-en MRT0 export");

        let inst = &code.get_instructions()[1];
        assert_eq!(inst.type_, T::Exp);
        assert_eq!(inst.format, F::Mrt0Vsrc0Vsrc1ComprVmDone);
        assert_eq!(inst.export_enable, 0x3);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4; // SPI_SHADER_FP16_ABGR
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile GTA compressed partial-en MRT0 export");
        assert!(
            source.contains(
                "OpCompositeConstruct %v4float %t4_1 %t5_1 %float_0_000000 %float_1_000000"
            ),
            "disabled BA channels must default to (0, 1):\n{source}"
        );
        assert!(
            !source.contains("OpLoad %float %v1"),
            "disabled packed source must not read unwritten v1:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble compressed partial MRT0 export");
        naga_parse_and_validate(&words, "GTA exp mrt0 compr=1 en=3");
    }

    /// The full uncompressed form (`en=0xf`, mode 9) keeps its old shape:
    /// all four channels load their VGPRs, no defaults.
    #[test]
    fn exp_mrt0_uncompressed_full_en_writes_all_four_channels() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0
                0x7E02_0280, // v_mov_b32 v1, 0
                0x7E04_0280, // v_mov_b32 v2, 0
                0x7E06_0280, // v_mov_b32 v3, 0
                0xF800_180F, // exp mrt0 v0..v3 (en=0xf compr=0 vm=1 done=1)
                0x0302_0100,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse uncompressed full-en mrt0 export");

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 9; // SPI_SHADER_32_ABGR
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile uncompressed full-en mrt0 export");
        assert!(
            source.contains("OpCompositeConstruct %v4float %t0_4 %t1_4 %t2_4 %t3_4"),
            "{source}"
        );
        let words = spirv_run(&source).expect("assemble full-en mrt0 export");
        naga_parse_and_validate(&words, "exp mrt0 compr=0 en=f");
    }

    /// The pipeline already carries MRT1..7 attachments; pin the missing
    /// shader half with a real two-export sequence. MRT0 is not `done` because
    /// MRT1 follows, while MRT1 terminates the export sequence.
    #[test]
    fn fragment_mrt0_and_mrt1_exports_validate_with_distinct_locations() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0
                0x7E02_0280, // v_mov_b32 v1, 0
                0x7E04_0280, // v_mov_b32 v2, 0
                0x7E06_0280, // v_mov_b32 v3, 0
                0xF800_100F, // exp mrt0 v0..v3, vm, done=0
                0x0302_0100,
                0xF800_181F, // exp mrt1 v0..v3, vm, done=1
                0x0302_0100,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse MRT0+MRT1 fragment exports");

        let exports: Vec<_> = code
            .get_instructions()
            .iter()
            .filter(|inst| inst.type_ == T::Exp)
            .collect();
        assert_eq!(exports.len(), 2);
        assert_eq!(exports[0].export_target, 0);
        assert_eq!(exports[1].export_target, 1);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 9;
        input_info.target_output_mode[1] = 9;
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile MRT0+MRT1 fragment exports");

        assert!(
            source.contains("OpDecorate %outColor Location 0")
                && source.contains("OpDecorate %outColor1 Location 1"),
            "both fragment output locations must be declared:\n{source}"
        );
        assert!(
            source.contains("OpStore %outColor %t11_4")
                && source.contains("OpStore %outColor1 %t11_5"),
            "each EXP target must store to its own output:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble MRT0+MRT1 fragment exports");
        naga_parse_and_validate(&words, "fragment MRT0+MRT1 exports");
    }

    /// `v_cmpx_ge_f32` / `v_cmpx_nle_f32` (VOPC 0x16 / 0x1c) — the ASTRO.BOT
    /// siblings of the shipped cmpx rows. ge is ordered >=; nle is the
    /// negation of <=, i.e. unordered > (NaN → true).
    #[test]
    fn v_cmpx_ge_and_nle_f32_are_wired_with_correct_comparisons() {
        let ge = recomp_func(T::VCmpxGeF32, F::SmaskVsrc0Vsrc1).expect("VCmpxGeF32 row");
        assert!(matches!(ge.func, RecompileFn::Func(_)));
        assert_eq!(ge.param[0], Some("OpFOrdGreaterThanEqual"));

        let nle = recomp_func(T::VCmpxNleF32, F::SmaskVsrc0Vsrc1).expect("VCmpxNleF32 row");
        assert!(matches!(nle.func, RecompileFn::Func(_)));
        assert_eq!(
            nle.param[0],
            Some("OpFUnordGreaterThan"),
            "v_cmpx_nle is unordered > (NaN → true), not an ordered compare"
        );
    }

    #[test]
    fn astro_buffer_load_dwordx2_loads_two_consecutive_dwords() {
        // Measured raw 0xE0342000: buffer_load_dwordx2 with idxen (58
        // dispatches / 5 min failed on this opcode).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xE034_2000, 0x8001_0400, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse buffer_load_dwordx2");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadDwordX2);
        assert_eq!(inst.format, F::Vdata2VaddrSvSoffsIdxen);
        assert_eq!(inst.dst.size, 2);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.storage_buffers.buffers_num = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile buffer_load_dwordx2");
        assert!(source.contains("%t110_0_0 = OpFunctionCall %void %buffer_load_float1"));
        assert!(source.contains("%t110_0_1 = OpFunctionCall %void %buffer_load_float1"));
        assert!(
            !source.contains("%t110_0_2"),
            "an x2 load must stop at two dwords:\n{source}"
        );
        assert!(
            source.contains("OpIAdd %int %t164_0_1 %int_4"),
            "second dword sits 4 bytes on:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble buffer_load_dwordx2");
        naga_parse_and_validate(&words, "buffer_load_dwordx2");
    }

    #[test]
    fn astro_buffer_load_dwordx3_loads_three_consecutive_dwords() {
        // Measured raw 0xE03C2074: buffer_load_dwordx3 with idxen and an
        // immediate offset (116 dispatches/run failed on this opcode).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xE03C_2074, 0x8001_0400, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse buffer_load_dwordx3");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadDwordX3);
        assert_eq!(inst.format, F::Vdata3VaddrSvSoffsIdxen);
        assert_eq!(inst.dst.size, 3);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.storage_buffers.buffers_num = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile buffer_load_dwordx3");
        for k in 0..3 {
            assert!(
                source.contains(&format!(
                    "%t110_0_{k} = OpFunctionCall %void %buffer_load_float1"
                )),
                "dword {k}:\n{source}"
            );
        }
        assert!(
            !source.contains("%t110_0_3"),
            "an x3 load must stop at three dwords:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble buffer_load_dwordx3");
        naga_parse_and_validate(&words, "buffer_load_dwordx3");
    }

    #[test]
    fn astro_buffer_load_ubyte_extracts_the_byte_from_the_containing_dword() {
        // Measured raw 0xE02020C0: buffer_load_ubyte with idxen and an
        // immediate offset 0xc0 (58 dispatches/run failed on this opcode).
        // The byte address is offset + index * stride — NOT pre-divided by
        // 4: the helper divides for the dword index and extracts the byte at
        // bit offset (byte_addr & 3) * 8.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xE020_20C0, 0x8001_0400, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse buffer_load_ubyte");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadUbyte);
        assert_eq!(inst.format, F::Vdata1VaddrSvSoffsIdxen);
        assert_eq!(inst.dst.size, 1);
        // The 12-bit immediate byte offset is folded into the soffset
        // constant unscaled.
        assert_eq!(inst.src[2].constant.u, 0xC0);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.storage_buffers.buffers_num = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile buffer_load_ubyte");
        assert!(
            source.contains("OpFunctionCall %void %buffer_load_ubyte"),
            "ubyte helper call:\n{source}"
        );
        // The helper computes the dword index by dividing the BYTE address...
        assert!(
            source.contains("%buf_l_ub_49 = OpSDiv %int %buf_l_ub_47 %int_4"),
            "byte address divided for the dword index:\n{source}"
        );
        // ...and zero-extends the byte at (byte_addr & 3) * 8.
        assert!(
            source.contains("OpBitFieldUExtract %uint %buf_l_ub_64 %buf_l_ub_51 %int_8"),
            "byte extraction:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble buffer_load_ubyte");
        naga_parse_and_validate(&words, "buffer_load_ubyte");
    }

    #[test]
    fn astro_image_store_dmask3_writes_two_channels() {
        // MIMG 0x08 with dmask 0x3 — the last remaining image_store form
        // (58 dispatches / 5 min).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF020_0300, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_store dmask3");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageStore);
        assert_eq!(inst.format, F::Vdata2Vaddr3StDmask3);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.textures2d.desc[0].textures2d_without_sampler = true;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_store dmask3");
        assert!(source.contains("OpImageWrite"), "{source}");
        assert!(
            source.contains("%t85_0 %float_0_000000 %float_0_000000"),
            "disabled b/a channels must both be zero:\n{source}"
        );
        // Assemble-only: naga rejects the format-less storage-image write
        // (`InvalidImage`) that real Vulkan accepts via
        // shaderStorageImageWriteWithoutFormat — same class as dmask1/F.
        let _ = spirv_run(&source).expect("assemble image_store dmask3");
    }

    #[test]
    fn astro_vop3_mbcnt_hi_lane_index_idiom_recompiles() {
        // Measured raw 0xd7660003: RDNA2 VOP3 0x366 = v_mbcnt_hi_u32_b32
        // (SharpEmu Gen5 decoder agrees; NOT v_lshl_add_u32, which is 0x346).
        // b1: src0 = inline -1 (0xc1), src1 = v3 (0x103).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E06_0280, // v_mov_b32 v3, 0
                0xD766_0003,
                0x0002_06C1,
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse VOP3 v_mbcnt_hi_u32_b32");
        let inst = &code.get_instructions()[1];
        assert_eq!(inst.type_, T::VMbcntHiU32B32);
        assert_eq!(inst.format, F::SVdstSVsrc0SVsrc1);
        assert_eq!(
            (inst.dst.type_, inst.dst.register_id),
            (ShaderOperandType::Vgpr, 3)
        );

        let entry = recomp_func(T::VMbcntHiU32B32, F::SVdstSVsrc0SVsrc1).expect("mbcnt hi row");
        assert!(
            matches!(entry.func, RecompileFn::Func(_)),
            "the mbcnt pair must be wired, not staged NI"
        );

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile v_mbcnt_hi_u32_b32");
        // Kyty single-lane model: dst = src1 when exec is on.
        assert!(source.contains("OpSelect %float"), "{source}");
        let words = spirv_run(&source).expect("assemble v_mbcnt_hi_u32_b32");
        naga_parse_and_validate(&words, "vop3 v_mbcnt_hi_u32_b32");
    }

    /// A next-gen CS with direct SGPRs (the SRT/immediate path) seeds the
    /// registers from the push-constant window at shader entry — the same
    /// mechanism the other stages use, now reachable on compute since the
    /// "cs: direct sgprs" gate is relaxed for Gen5.
    #[test]
    fn cs_direct_sgprs_seed_from_push_constants() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xBE82_0300, 0xBF80_0000, S_ENDPGM], // s_mov_b32 s2, s0
            &mut code,
            true,
        )
        .expect("parse s_mov_b32 s2, s0");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.direct_sgprs.sgprs_num = 2;
        input_info.bind.direct_sgprs.start_register[0] = 0;
        input_info.bind.direct_sgprs.start_register[1] = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile CS with direct sgprs");
        assert!(
            source.contains("OpStore %s0 %vsharp_value_b0_f0"),
            "s0 seeded from push constants:\n{source}"
        );
        assert!(
            source.contains("OpStore %s1 %vsharp_value_b0_f1"),
            "s1 seeded from push constants:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble CS with direct sgprs");
        naga_parse_and_validate(&words, "cs direct sgprs");
    }

    #[test]
    fn astro_lds_write_barrier_read_round_trip() {
        // The measured LDS trio: ds_write_b32 (raw 0xd8340000 family),
        // s_barrier, ds_read_b32 — lowered onto a Workgroup-storage uint
        // array sized from COMPUTE_PGM_RSRC2.LDS_SIZE (64 KiB fallback).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0 (addr)
                0x7E02_0280, // v_mov_b32 v1, 0 (data)
                0xD834_0000, // ds_write_b32 v0, v1
                0x0000_0100,
                0xBF8A_0000, // s_barrier
                0xD8D8_0000, // ds_read_b32 v2, v0
                0x0200_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse LDS write/barrier/read");

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile LDS write/barrier/read");
        assert!(
            source.contains("%lds = OpVariable %_ptr_Workgroup__arr_uint_lds Workgroup"),
            "{source}"
        );
        assert!(
            source.contains("%lds_num_uint_16384"),
            "64 KiB fallback when LDS_SIZE is 0:\n{source}"
        );
        assert!(
            source.contains("OpControlBarrier %uint_2 %uint_2 %uint_0x00000108"),
            "{source}"
        );
        assert!(
            source.contains("OpAccessChain %_ptr_Workgroup_uint %lds"),
            "{source}"
        );
        let words = spirv_run(&source).expect("assemble LDS write/barrier/read");
        naga_parse_and_validate(&words, "lds write/barrier/read");
    }

    #[test]
    fn astro_ds_write_b96_read2_round_trip() {
        // ds_write_b96 v0, v[1:3] offset:8 then ds_read2_b32 v[4:5], v0
        // offset1:1 — the two remaining ASTRO.BOT LDS family members
        // (58 + 115 measured failures). Write lowers to three exec-guarded
        // stores; read2 to two independent clamped loads.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0 (addr)
                0x7E02_0280, // v_mov_b32 v1, 0
                0x7E04_0280, // v_mov_b32 v2, 0
                0x7E06_0280, // v_mov_b32 v3, 0
                0xDB78_0008, // ds_write_b96 addr=v0, data=v[1:3], offset 8
                0x0000_0100,
                0xD8DC_0100, // ds_read2_b32 v[4:5], v0, offset0 0, offset1 1
                0x0400_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_write_b96 + ds_read2_b32");

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_write_b96 + ds_read2_b32");
        assert!(
            source.contains("%lds = OpVariable %_ptr_Workgroup__arr_uint_lds Workgroup"),
            "{source}"
        );
        // Three write stores...
        for k in 0..3 {
            assert!(
                source.contains(&format!("OpStore %ldsw3_p_4_{k} %ldsw3_du_4_{k}")),
                "store {k}:\n{source}"
            );
        }
        // ...and two read results.
        for k in 0..2 {
            assert!(
                source.contains(&format!("OpStore %v{} %ldsr2_s_5_{k}", 4 + k)),
                "read2 result {k}:\n{source}"
            );
        }
        let words = spirv_run(&source).expect("assemble ds_write_b96 + ds_read2_b32");
        naga_parse_and_validate(&words, "ds_write_b96 + ds_read2_b32");
    }

    #[test]
    fn astro_ds_read_b64_reads_two_consecutive_lds_dwords() {
        // ds_read_b64 v[2:3], v0 offset:16 (raw 0xd9d80000 family, measured
        // on ASTRO.BOT scene compute — 58 dispatches/run). One byte offset,
        // two consecutive dwords: the parser materialises offset and
        // offset + 4 so the read2 body serves it.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0 (addr)
                0xD9D8_0010, // ds_read_b64 v[2:3], v0, offset 16
                0x0200_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_read_b64");
        let inst = &code.get_instructions()[1];
        assert_eq!(inst.type_, T::DsReadB64);
        assert_eq!(inst.src[1].constant.u, 16);
        assert_eq!(inst.src[2].constant.u, 20);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_read_b64");
        for k in 0..2 {
            assert!(
                source.contains(&format!("OpStore %v{} %ldsr2_s_1_{k}", 2 + k)),
                "result dword {k}:\n{source}"
            );
        }
        let words = spirv_run(&source).expect("assemble ds_read_b64");
        naga_parse_and_validate(&words, "ds_read_b64");
    }

    #[test]
    fn astro_ds_write_b128_stores_four_dwords() {
        // ds_write_b128 v0, v[1:4] offset:8 (raw 0xdb7c0000 family, measured
        // on ASTRO.BOT scene compute) — the four-dword row of the b96 model.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0 (addr)
                0x7E02_0280, // v_mov_b32 v1, 0
                0x7E04_0280, // v_mov_b32 v2, 0
                0x7E06_0280, // v_mov_b32 v3, 0
                0x7E08_0280, // v_mov_b32 v4, 0
                0xDB7C_0008, // ds_write_b128 addr=v0, data=v[1:4], offset 8
                0x0000_0100,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_write_b128");
        let inst = &code.get_instructions()[5];
        assert_eq!(inst.type_, T::DsWriteB128);
        assert_eq!(inst.format, F::Vsrc0Vsrc14Vsrc2);
        assert_eq!(inst.src[1].size, 4);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_write_b128");
        for k in 0..4 {
            assert!(
                source.contains(&format!("OpStore %ldsw3_p_5_{k} %ldsw3_du_5_{k}")),
                "store {k}:\n{source}"
            );
        }
        let words = spirv_run(&source).expect("assemble ds_write_b128");
        naga_parse_and_validate(&words, "ds_write_b128");
    }

    /// The measured ASTRO.BOT extended-CS shape (461 skips/run on "extended
    /// storage buffer mapping"): a V# declared at start_register=12 with
    /// the EUD pointer pair at (s12, s13). The mapping rebases on the EUD
    /// base (12 - 12 = dword 0), and the shader's
    /// `s_load_dwordx4 s[16:19], s[12:13], 0` becomes push-constant reads
    /// of the captured descriptor.
    #[test]
    fn astro_extended_storage_buffer_below_s16_recompiles() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx4,
            format: F::Sdst4SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 16,
                size: 4,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        for _ in 0..3 {
            code.get_instructions_mut().push(sload);
        }
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 12;
        input_info.bind.storage_buffers.extended[0] = true;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("extended V# below s16 must recompile (measured ASTRO.BOT shape)");
        assert!(
            source.contains("%vsharp"),
            "the descriptor loads must come from the push-constant table:\n{source}"
        );
        for reg in 16..20 {
            assert!(
                source.contains(&format!("OpStore %s{reg}")),
                "descriptor dword must land in s{reg}:\n{source}"
            );
        }
        let words = spirv_run(&source).expect("assemble extended CS s_load_dwordx4");
        naga_parse_and_validate(&words, "s_load_dwordx4_extended_cs");
    }

    /// Round 9: `v_rcp_iflag_f32` (VOP1 0x2b, 58 measured ASTRO.BOT skips)
    /// lowers exactly like `v_rcp_f32` — 1.0/x; the integer-div-by-zero TRAP
    /// flag it would raise is not modelled.
    #[test]
    fn astro_v_rcp_iflag_f32_lowers_as_reciprocal() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // v_mov_b32 v0, 0; v_rcp_iflag_f32 v1, v0.
        shader_parse(0, &[0x7E00_0280, 0x7E02_5700, S_ENDPGM], &mut code, true)
            .expect("parse v_rcp_iflag_f32");
        assert_eq!(code.get_instructions()[1].type_, T::VRcpIflagF32);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile v_rcp_iflag_f32");
        assert!(
            source.contains("OpFDiv %float %float_1_000000"),
            "reciprocal expected:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble v_rcp_iflag_f32");
        naga_parse_and_validate(&words, "v_rcp_iflag_f32");
    }

    /// Round 9: `ds_add_u32` (DS 0x00, 58 measured ASTRO.BOT skips) is an
    /// exec-guarded `OpAtomicIAdd` on the `%lds` Workgroup array, Workgroup
    /// scope, Relaxed semantics. Raw shape measured on ASTRO.BOT scene
    /// compute: 0xd8000514 (byte offset 0x514).
    #[test]
    fn astro_ds_add_u32_lowers_as_lds_atomic_add() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0 (addr)
                0x7E02_0280, // v_mov_b32 v1, 0 (data)
                0xD800_0514, // ds_add_u32 addr=v0, data=v1, offset 0x514
                0x0000_0100,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_add_u32");
        let inst = &code.get_instructions()[2];
        assert_eq!(inst.type_, T::DsAddU32);
        assert_eq!(inst.format, F::Vsrc0Vsrc1Vsrc2);
        assert_eq!(inst.src[2].constant.u, 0x514);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_add_u32");
        assert!(
            source.contains("OpAtomicIAdd %uint %ldsaa_p_2 %uint_2 %uint_0 %ldsaa_du_2"),
            "LDS atomic add expected:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble ds_add_u32");
        naga_parse_and_validate(&words, "ds_add_u32");
    }

    #[test]
    fn astro_ds_add_rtn_u32_returns_the_old_lds_value() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0 (addr)
                0x7E02_0280, // v_mov_b32 v1, 0 (data)
                0xD880_0080, // ds_add_rtn_u32, offset 0x80
                0x0200_0100, // vdst=v2, data0=v1, addr=v0
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_add_rtn_u32");
        let inst = &code.get_instructions()[2];
        assert_eq!(inst.type_, T::DsAddRtnU32);
        assert_eq!(inst.format, F::VdstVsrc0Vsrc1Vsrc2);
        assert_eq!(inst.dst.register_id, 2);
        assert_eq!(inst.src[2].constant.u, 0x80);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_add_rtn_u32");
        assert!(
            source.contains("OpAtomicIAdd %uint %ldsaar_p_2 %uint_2 %uint_0 %ldsaar_du_2"),
            "LDS atomic add expected:\n{source}"
        );
        assert!(
            source.contains("OpStore %v2 %ldsaar_of_2"),
            "the pre-add value must be returned to vdst:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble ds_add_rtn_u32");
        naga_parse_and_validate(&words, "ds_add_rtn_u32");
    }

    /// ASTRO.BOT tiled-lighting compute (`ds_wrxchg_rtn_b32`, DS 0x2d, raw
    /// 0xd8b40510, byte offset 0x510): an LDS write-exchange that returns the
    /// old value — `vdst = lds[a]; lds[a] = data`. Lowers to an exec-guarded
    /// `OpAtomicExchange` on the `%lds` Workgroup array, old value into `vdst`.
    #[test]
    fn astro_ds_wrxchg_rtn_b32_lowers_as_lds_atomic_exchange() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0 (addr)
                0x7E02_0280, // v_mov_b32 v1, 0 (data)
                0xD8B4_0510, // ds_wrxchg_rtn_b32, offset 0x510
                0x0200_0100, // vdst=v2, data0=v1, addr=v0
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_wrxchg_rtn_b32");
        let inst = &code.get_instructions()[2];
        assert_eq!(inst.type_, T::DsWrxchgRtnB32);
        assert_eq!(inst.format, F::VdstVsrc0Vsrc1Vsrc2);
        assert_eq!(inst.dst.register_id, 2, "vdst = v2 (old value)");
        assert_eq!(inst.src[0].register_id, 0, "addr = v0");
        assert_eq!(inst.src[1].register_id, 1, "data = v1");
        assert_eq!(inst.src[2].constant.u, 0x510);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_wrxchg_rtn_b32");
        assert!(
            source.contains("OpAtomicExchange %uint %ldsx_p_2 %uint_2 %uint_0 %ldsx_du_2"),
            "LDS atomic exchange expected:\n{source}"
        );
        assert!(
            source.contains("OpStore %v2 %ldsx_of_2"),
            "the old value is written to vdst:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble ds_wrxchg_rtn_b32");
        naga_parse_and_validate(&words, "ds_wrxchg_rtn_b32");
    }

    /// A branch whose target lands mid-instruction is the exact symptom a
    /// MIS-SIZED decode produces (a wrong instruction length shifts every later
    /// PC, so a valid branch target stops matching a boundary). The relooper's
    /// boundary error must name the *straddling* instruction — the mis-sized
    /// opcode — not just the target, so the culprit is actionable. (Live symptom:
    /// ASTRO.BOT tiled-lighting `branch target 0x1150 is not an instruction
    /// boundary`.)
    #[test]
    fn reloop_nonboundary_branch_target_names_straddling_instruction() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0xBF82_0001, // s_branch +1 -> target 0x8 (mid the next instruction)
                0x7E00_02FF, // v_mov_b32 v0, lit  (2 dwords, 0x4..0xc)
                0x3F80_0000, // literal 1.0
                0x7E02_0280, // v_mov_b32 v1, 0    (0xc)
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse branch-into-instruction");

        let mut cs = ShaderComputeInputInfo::default();
        cs.threads_num = [1, 1, 1];
        let err = spirv_generate_source(&code, None, None, Some(&cs))
            .expect_err("non-boundary branch target must be refused by the relooper");
        let msg = format!("{err}");
        assert!(msg.contains("0x8"), "names the target: {msg}");
        assert!(
            msg.contains("0x4") && msg.contains("VMovB32"),
            "names the straddling (mis-sized) instruction at 0x4: {msg}"
        );
        assert!(msg.contains("0xc"), "names the next boundary: {msg}");
    }

    #[test]
    fn reloop_accepts_branch_to_s_waitcnt_vscnt_after_mubuf() {
        // Reduced live ASTRO.BOT sequence. MUBUF is 8 bytes (pc 0x4..0xc)
        // and s_waitcnt_vscnt at 0xc is the branch target. Recording the wait
        // preserves that boundary; the wait itself is a SPIR-V no-op because
        // it waits on the same wave's scoreboard, not cross-invocation memory.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0xBF82_0002, // s_branch +2 -> pc 0xc
                0xE070_2000, // buffer_store_dword, idxen
                0xFF01_0400, // fixed-width second MUBUF dword
                0xBBFD_0000, // s_waitcnt_vscnt at target
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse branch to s_waitcnt_vscnt");
        let mut cs = ShaderComputeInputInfo::default();
        cs.threads_num = [1, 1, 1];
        cs.bind.push_constant_size = 16;
        cs.bind.storage_buffers.buffers_num = 1;
        cs.bind.storage_buffers.start_register[0] = 4;
        let source = spirv_generate_source(&code, None, None, Some(&cs))
            .expect("wait target is a real relooper boundary");
        assert!(source.contains("OpSwitch"), "relooper is active:\n{source}");
        assert!(
            !source.contains("OpMemoryBarrier"),
            "waitcnt must not become a device-wide barrier:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble branch-to-wait shader");
        spirv_val_ok(&words, "branch_to_waitcnt_vscnt_after_mubuf");
    }

    // ---- FLAT-class recompile (SharpEmu PR #587 `Gen5FlatMemoryTests`) ----

    /// A FLAT-segment `flat_load_dword v5, v[2:3]` lowers to a `%global_mem`
    /// window access: the VGPR pair is the whole address, converted to a dword
    /// index and read with an out-of-bounds clamp.
    #[test]
    fn flat_load_dword_recompiles_to_global_window() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // Two v_mov padding ops keep s_endpgm at instruction index >= 2 (the
        // endpgm handler looks back two instructions); the FLAT op stays at 0.
        shader_parse(
            0,
            &[0xDC30_0000, 0x057F_0002, 0x7E02_0280, 0x7E02_0280, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse flat_load_dword");
        assert!(code.get_instructions()[0].uses_flat_address);

        let mut cs = ShaderComputeInputInfo::default();
        cs.threads_num = [1, 1, 1];
        crate::shader::spirv::shader_detect_flat_global_window(&code, &mut cs.bind);
        assert!(cs.bind.global_mem.used, "detection declares the window");

        let source =
            spirv_generate_source(&code, None, None, Some(&cs)).expect("recompile flat load");
        assert!(
            source.contains("%global_mem = OpVariable %_ptr_StorageBuffer_GlobalMem StorageBuffer"),
            "declares the window SSBO:\n{source}"
        );
        assert!(
            source.contains("OpArrayLength %uint %global_mem 0"),
            "clamps against the window length:\n{source}"
        );
        assert!(
            source.contains("OpAccessChain %_ptr_StorageBuffer_uint %global_mem %int_0"),
            "reads the window at a computed dword index:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble flat load module");
        spirv_val_ok(&words, "flat_load_dword_global_window");
    }

    #[test]
    fn gta_flat_load_ubyte_selects_and_zero_extends_the_addressed_byte() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0xDC20_0000,
                0x007D_0001, // flat_load_ubyte v0, v[1:2]
                0x7E02_0280,
                0x7E02_0280,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse measured GTA flat byte load");
        assert_eq!(code.get_instructions()[0].type_, T::FlatLoadUbyte);

        let mut cs = ShaderComputeInputInfo::default();
        cs.threads_num = [64, 1, 1];
        crate::shader::spirv::shader_detect_flat_global_window(&code, &mut cs.bind);

        let source =
            spirv_generate_source(&code, None, None, Some(&cs)).expect("recompile flat byte load");
        assert!(source.contains("OpBitwiseAnd %uint %flat_bo_0 %uint_3"));
        assert!(source.contains("OpShiftLeftLogical %uint %flat_lane_0_0 %uint_3"));
        assert!(source.contains("OpShiftRightLogical %uint %flat_raw_0_0"));
        assert!(source.contains("OpBitwiseAnd %uint %flat_shr_0_0 %uint_255"));

        let words = spirv_run(&source).expect("assemble GTA flat byte load");
        spirv_val_ok(&words, "gta_flat_load_ubyte");
    }

    /// A GLOBAL-segment `global_load_dword v5, v2, s[8:9]` adds the SGPR base
    /// pair to the 32-bit VGPR offset (SharpEmu `UsesFlatAddress == false`).
    #[test]
    fn global_load_dword_recompiles_with_sgpr_base() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xDC30_8000, 0x0508_0002, 0x7E02_0280, 0x7E02_0280, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse global_load_dword");
        assert!(!code.get_instructions()[0].uses_flat_address);

        let mut cs = ShaderComputeInputInfo::default();
        cs.threads_num = [1, 1, 1];
        crate::shader::spirv::shader_detect_flat_global_window(&code, &mut cs.bind);

        let source =
            spirv_generate_source(&code, None, None, Some(&cs)).expect("recompile global load");
        // The SGPR base pair dword is loaded and added to the VGPR offset.
        assert!(
            source.contains("%flat_sb_0 = OpLoad %uint")
                && source.contains("%flat_ab_0 = OpIAdd %uint %flat_sb_0"),
            "adds the SGPR base to the VGPR offset:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble global load module");
        spirv_val_ok(&words, "global_load_dword_sgpr_base");
    }

    /// A FLAT-segment `flat_store_dword v[2:3], v6` drops out-of-bounds writes
    /// and stores in-bounds ones into the window.
    #[test]
    fn flat_store_dword_recompiles_to_global_window() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xDC70_0000, 0x007F_0602, 0x7E02_0280, 0x7E02_0280, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse flat_store_dword");

        let mut cs = ShaderComputeInputInfo::default();
        cs.threads_num = [1, 1, 1];
        crate::shader::spirv::shader_detect_flat_global_window(&code, &mut cs.bind);

        let source =
            spirv_generate_source(&code, None, None, Some(&cs)).expect("recompile flat store");
        assert!(
            source.contains("OpStore %flat_ptr_0_0"),
            "stores the dword into the window:\n{source}"
        );
        assert!(
            source.contains("OpBranchConditional %flat_inb_0_0"),
            "drops out-of-bounds stores:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble flat store module");
        spirv_val_ok(&words, "flat_store_dword_global_window");
    }

    /// Round 9 — the whole measured 693-skip bulk: the next-gen SMEM parser
    /// materializes a NULL soffset as a sign-extended 21-bit immediate in an
    /// `IntegerInlineConstant` operand, which the extended s_load path must
    /// accept exactly like the legacy parser's `LiteralConstant`. Raw dwords
    /// measured from ASTRO.BOT CS 0x50740a700:
    /// `s_load_dwordx4 s[16:19], s[12:13], 0x0` = `0xf4080406 0xfa000000`.
    #[test]
    fn astro_sload_inline_constant_soffset_recompiles() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xf408_0406, 0xfa00_0000, 0xf408_0406, 0xfa00_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse s_load_dwordx4 with NULL soffset");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SLoadDwordx4);
        assert_eq!(
            inst.src[1].type_,
            ShaderOperandType::IntegerInlineConstant,
            "NULL soffset parses as the inline-constant immediate"
        );

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 12;
        input_info.bind.storage_buffers.extended[0] = true;

        let source = spirv_generate_source(&code, None, None, Some(&input_info)).expect(
            "inline-constant soffset must recompile like a literal (measured ASTRO.BOT shape)",
        );
        for reg in 16..20 {
            assert!(
                source.contains(&format!("OpStore %s{reg}")),
                "descriptor dword must land in s{reg}:\n{source}"
            );
        }
        let words = spirv_run(&source).expect("assemble inline-soffset s_load");
        naga_parse_and_validate(&words, "s_load_dwordx4_inline_soffset");
    }

    /// A sign-extended imm21 soffset can be negative; nothing below the EUD
    /// base is mapped, so the refusal is by name (never a huge u32 index).
    #[test]
    fn sload_negative_inline_soffset_refuses_by_name() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // s_load_dwordx4 s[16:19], s[12:13], -4 (imm21 = 0x1ffffc).
        shader_parse(0, &[0xf408_0406, 0xfa1f_fffc, S_ENDPGM], &mut code, true)
            .expect("parse s_load_dwordx4 with negative imm21");
        assert_eq!(code.get_instructions()[0].src[1].constant.i(), -4);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 12;
        input_info.bind.storage_buffers.extended[0] = true;

        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("negative offset has no mapping");
        assert!(
            err.to_string().contains("negative s_load offset"),
            "named refusal expected, got: {err}"
        );
    }

    /// A sharp declared at/above the register-file size rebases by
    /// SGPRS_MAX (32) — the "EUD continues the file" shape — while an
    /// s_load of an EUD dword no descriptor covers takes the raw
    /// EUD-window fallback (SharpEmu port): detection records the window
    /// in `bind.eud_raw` and the recompiled load reads the `%eud_raw`
    /// SSBO instead of refusing (the pre-port behavior this test used to
    /// pin was the named "not a captured descriptor field" refusal — 195
    /// ASTRO.BOT compute dispatches/run).
    #[test]
    fn extended_mapping_rebases_by_file_size_and_raw_window_covers_unmapped_dwords() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let mut sload = ShaderInstruction {
            type_: T::SLoadDwordx4,
            format: F::Sdst4SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 16,
                size: 4,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    // EUD dword 4 — the sharp below covers dwords 0..4.
                    constant: crate::shader::types::ShaderConstant::from_u(16),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        // The sharp is declared at start 36 => rel = 36 - 32 = EUD dword 4.
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 36;
        input_info.bind.storage_buffers.extended[0] = true;

        for _ in 0..3 {
            code.get_instructions_mut().push(sload);
        }
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("start_register >= 32 rebases by the file size");
        assert!(source.contains("%vsharp"), "{source}");
        // Detection over the covered load: dwords 4..8 are all captured by
        // the sharp, so no raw window is declared.
        let mut bind = input_info.bind;
        crate::shader::spirv::shader_detect_eud_raw_window(&code, &mut bind);
        assert!(
            !bind.eud_raw.used,
            "a fully-captured s_load needs no raw window: {:?}",
            bind.eud_raw
        );

        // An s_load of EUD dword 0 (no descriptor there): the pipeline runs
        // detection first, which declares the raw window; the recompiled
        // load then reads `%eud_raw` instead of refusing.
        sload.src[1].constant = crate::shader::types::ShaderConstant::from_u(0);
        let mut code2 = ShaderCode::new();
        code2.set_type(ShaderType::Compute);
        for _ in 0..3 {
            code2.get_instructions_mut().push(sload);
        }
        code2.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });
        crate::shader::spirv::shader_detect_eud_raw_window(&code2, &mut input_info.bind);
        let raw = input_info.bind.eud_raw;
        assert!(raw.used, "uncaptured EUD dwords declare the raw window");
        assert_eq!(
            raw.required_dwords, 4,
            "x4 load at offset 0 needs dwords 0..4"
        );
        assert_eq!(
            raw.binding_index, 1,
            "the raw window binds after the storage-buffer array (index 0)"
        );
        let source = spirv_generate_source(&code2, None, None, Some(&input_info))
            .expect("unmapped EUD dword lowers to a raw-window read");
        assert!(
            source.contains("%eud_raw = OpVariable %_ptr_StorageBuffer_EudRaw StorageBuffer"),
            "the raw window SSBO is declared:\n{source}"
        );
        assert!(
            source.contains("OpDecorate %eud_raw Binding 1"),
            "bound at the detected index:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble raw EUD-window compute shader");
        naga_parse_and_validate(&words, "eud_raw_window_x4");

        // WITHOUT detection (bind.eud_raw cleared) the named refusal stays:
        // the dispatch path would not bind a window the SPIR-V reads.
        input_info.bind.eud_raw = Default::default();
        let err = spirv_generate_source(&code2, None, None, Some(&input_info))
            .expect_err("no declared raw window keeps the named refusal");
        assert!(
            err.to_string().contains("not a captured descriptor field"),
            "{err}"
        );
    }

    /// Recompiler + binding metadata for an s_load entirely beyond the
    /// captured descriptors — including offsets past the 64-dword extended
    /// mapping itself (`get_mapped_index`'s other refusal arm). Every read
    /// clamps against the bound window size (`OpArrayLength` +
    /// `OpULessThan`/`OpSelect`) and yields 0 beyond it, so a short
    /// snapshot degrades instead of faulting — the bounds-clamp contract.
    #[test]
    fn sload_beyond_extended_mapping_reads_clamped_raw_window() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // s_load_dwordx8 s[16:23], s[12:13], 0x148 — dwords 82..90, beyond
        // the 64-dword extended mapping (would refuse "offset >= extended
        // mapping size" without the raw window).
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 16,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0x148),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        for _ in 0..3 {
            code.get_instructions_mut().push(sload);
        }
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 12;
        input_info.bind.storage_buffers.extended[0] = true;

        crate::shader::spirv::shader_detect_eud_raw_window(&code, &mut input_info.bind);
        let raw = input_info.bind.eud_raw;
        assert!(raw.used);
        assert_eq!(
            raw.required_dwords,
            0x148 / 4 + 8,
            "window covers through the load's last dword"
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("raw window covers loads beyond the extended mapping");
        // The bounds clamp: index min'd against len-1, result selected
        // against in-bounds, out-of-window reads produce %uint_0.
        for needle in [
            "OpArrayLength %uint %eud_raw 0",
            "UMin",
            "OpULessThan %bool",
            "OpSelect %uint",
        ] {
            assert!(source.contains(needle), "missing {needle}:\n{source}");
        }
        // All 8 destination dwords store through the raw path.
        assert_eq!(
            source.matches("OpArrayLength %uint %eud_raw 0").count(),
            8 * 3,
            "one clamped read per loaded dword per instruction:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble beyond-mapping raw-window shader");
        naga_parse_and_validate(&words, "eud_raw_window_beyond_mapping");
    }

    /// A mixed x8 load whose first half is a captured V# and second half is
    /// uncaptured: the captured dwords MUST keep reading the REWRITTEN
    /// descriptor from the push constants (guest memory holds the guest
    /// base, not the descriptor-array index) while only the uncaptured
    /// dwords read the raw window.
    #[test]
    fn mixed_sload_keeps_push_constants_for_captured_dwords() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // s_load_dwordx8 s[16:23], s[12:13], 0 — dwords 0..4 covered by the
        // sharp below, dwords 4..8 uncaptured.
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 16,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        for _ in 0..3 {
            code.get_instructions_mut().push(sload);
        }
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        // V# at the EUD base pair itself: rel 0 → covers EUD dwords 0..4.
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 12;
        input_info.bind.storage_buffers.extended[0] = true;

        crate::shader::spirv::shader_detect_eud_raw_window(&code, &mut input_info.bind);
        let raw = input_info.bind.eud_raw;
        assert!(raw.used);
        assert_eq!(raw.required_dwords, 8, "uncaptured dwords 4..8");

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("mixed captured + raw s_load recompiles");
        assert_eq!(
            source.matches("OpArrayLength %uint %eud_raw 0").count(),
            4 * 3,
            "exactly the four uncaptured dwords per instruction read raw:\n{source}"
        );
        assert!(
            source.contains("%_ptr_PushConstant_uint %vsharp"),
            "captured dwords still read the rewritten push-constant table:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble mixed captured/raw shader");
        naga_parse_and_validate(&words, "eud_raw_window_mixed");
    }

    /// A V# whose `stride * num_records` is not a dword multiple must
    /// recompile (Kyty EXITs there): the SPIR-V only addresses whole
    /// dwords, and the host pads the upload / truncates the writeback.
    /// Measured on ASTRO.BOT scene compute (58 dispatches/run refused).
    #[test]
    fn unaligned_v_sharp_byte_size_recompiles() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0x7E00_0280, 0x7E00_0280, 0x7E00_0280, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse");

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.buffers[0].fields[1] = 2 << 16; // stride 2
        input_info.bind.storage_buffers.buffers[0].fields[2] = 7; // 14 bytes
        input_info.bind.push_constant_size = 16;
        shader_recompile_cs(&code, &input_info)
            .expect("an unaligned V# byte size must not refuse the recompile");
    }

    #[test]
    fn astro_ds_read_b128_reads_four_consecutive_lds_dwords() {
        // ds_read_b128 v[2:5], v0 offset:16 (DS opcode 0xff, measured on
        // ASTRO.BOT scene compute — 58 dispatches/run). One byte offset,
        // four consecutive dwords at offset + 4k.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0 (addr)
                0xDBFC_0010, // ds_read_b128 v[2:5], v0, offset 16
                0x0200_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_read_b128");
        let inst = &code.get_instructions()[1];
        assert_eq!(inst.type_, T::DsReadB128);
        assert_eq!(inst.format, F::Vdst4Vsrc0Vsrc1);
        assert_eq!(inst.dst.size, 4);
        assert_eq!(inst.src[1].constant.u, 16);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_read_b128");
        for k in 0..4 {
            assert!(
                source.contains(&format!("OpStore %v{} %ldsr4_s_1_{k}", 2 + k)),
                "result dword {k}:\n{source}"
            );
        }
        // Dword k indexes through the derived offset constant 16 + 4k.
        for off in [16, 20, 24, 28] {
            assert!(
                source.contains(&format!("%uint_{off}")),
                "derived offset constant {off}:\n{source}"
            );
        }
        let words = spirv_run(&source).expect("assemble ds_read_b128");
        naga_parse_and_validate(&words, "ds_read_b128");
    }

    #[test]
    fn astro_ds_read_b96_reads_three_consecutive_lds_dwords() {
        // ds_read_b96 v[2:4], v0 offset:16 (DS opcode 0xfe, measured on
        // ASTRO.BOT scene compute as raw 0xdbf80550 — 58 dispatches/run).
        // The three-dword row of the b128 model.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E00_0280, // v_mov_b32 v0, 0 (addr)
                0xDBF8_0010, // ds_read_b96 v[2:4], v0, offset 16
                0x0200_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_read_b96");
        let inst = &code.get_instructions()[1];
        assert_eq!(inst.type_, T::DsReadB96);
        assert_eq!(inst.format, F::Vdst3Vsrc0Vsrc1);
        assert_eq!(inst.dst.size, 3);
        assert_eq!(inst.src[1].constant.u, 16);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_read_b96");
        for k in 0..3 {
            assert!(
                source.contains(&format!("OpStore %v{} %ldsr4_s_1_{k}", 2 + k)),
                "result dword {k}:\n{source}"
            );
        }
        assert!(
            !source.contains("%ldsr4_s_1_3"),
            "a b96 read must stop at three dwords:\n{source}"
        );
        // Dword k indexes through the derived offset constant 16 + 4k.
        for off in [16, 20, 24] {
            assert!(
                source.contains(&format!("%uint_{off}")),
                "derived offset constant {off}:\n{source}"
            );
        }
        let words = spirv_run(&source).expect("assemble ds_read_b96");
        naga_parse_and_validate(&words, "ds_read_b96");
    }

    #[test]
    fn astro_v_bfe_i32_sign_extends_through_int() {
        // v_bfe_i32 v1, v0, v1, v2 (VOP3 0x149, measured on ASTRO.BOT scene
        // compute — 58 dispatches/run): signed bitfield extract.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0xD549_0001, // v_bfe_i32 v1, v0, v1, v2
                0x040A_0300,
                0x7E06_0280, // v_mov_b32 v3, 0 (pad: s_endpgm looks back 2)
                0x7E06_0280, // v_mov_b32 v3, 0
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse v_bfe_i32");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VBfeI32);
        assert_eq!(inst.format, F::VdstVsrc0Vsrc1Vsrc2);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile v_bfe_i32");
        assert!(
            source.contains("OpBitFieldSExtract %int"),
            "signed extract:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble v_bfe_i32");
        naga_parse_and_validate(&words, "v_bfe_i32");
    }

    #[test]
    fn astro_v_bfi_b32_selects_by_mask() {
        // v_bfi_b32 v1, v0, v1, v2 (VOP3 0x14a, measured on ASTRO.BOT scene
        // compute — 58 dispatches/run): dst = (v0 & v1) | (~v0 & v2).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0xD54A_0001, // v_bfi_b32 v1, v0, v1, v2
                0x040A_0300,
                0x7E06_0280, // v_mov_b32 v3, 0 (pad: s_endpgm looks back 2)
                0x7E06_0280, // v_mov_b32 v3, 0
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse v_bfi_b32");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::VBfiB32);
        assert_eq!(inst.format, F::VdstVsrc0Vsrc1Vsrc2);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile v_bfi_b32");
        assert!(source.contains("OpNot %uint"), "mask inversion:\n{source}");
        let words = spirv_run(&source).expect("assemble v_bfi_b32");
        naga_parse_and_validate(&words, "v_bfi_b32");
    }

    /// `v_cmpx_le_f32` (VOPC 0x13) — the ordered <= exec-writing compare —
    /// must be implemented and stay distinct from its Lt sibling.
    #[test]
    fn vcmpx_le_f32_is_implemented_and_ordered() {
        let le = recomp_func(T::VCmpxLeF32, F::SmaskVsrc0Vsrc1).expect("VCmpxLeF32 row");
        assert!(
            matches!(le.func, RecompileFn::Func(_)),
            "VCmpxLeF32 must be implemented, not NI"
        );
        assert_eq!(le.param[0], Some("OpFOrdLessThanEqual"));
    }

    #[test]
    fn astro_buffer_store_dwordx4_stores_four_consecutive_dwords() {
        // buffer_store_dwordx4 v[1:4], s[4:7] (raw 0xe0780000, MUBUF 0x1e,
        // measured on ASTRO.BOT scene compute; the measured form carries no
        // idxen/offen — address-only, like the load twin's Vdata4SvSoffs row).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xE078_0000, 0x8001_0100, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse buffer_store_dwordx4");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferStoreDwordX4);
        assert_eq!(inst.format, F::Vdata4SvSoffs);
        assert_eq!(inst.dst.size, 4);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.storage_buffers.buffers_num = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile buffer_store_dwordx4");
        // Exec-guarded, address-only (vindex 0), four store helper calls.
        assert!(source.contains("OpStore %temp_int_1 %int_0"), "{source}");
        assert!(
            source.contains("%sdxn_e1_0 = OpINotEqual %bool"),
            "exec guard:\n{source}"
        );
        for k in 0..4 {
            assert!(
                source.contains(&format!(
                    "%sdxn_c_0_{k} = OpFunctionCall %void %buffer_store_float1 %v{}",
                    1 + k
                )),
                "store call {k}:\n{source}"
            );
        }
        // NOTE: assemble-only — naga rejects Kyty's array-of-struct-with-
        // runtime-array storage-buffer pattern the Vulkan driver accepts.
        let _ = spirv_run(&source).expect("assemble buffer_store_dwordx4");
    }

    #[test]
    fn astro_image_gather4_lz_gathers_four_texels() {
        // image_gather4_lz v[2:5], v[6:8], s[0:7], s[8:11] dmask:1 (raw
        // 0xf11c0108, measured on ASTRO.BOT scene compute). Four texels of
        // channel 0 via OpImageGather (single-mip images make the plain
        // gather the LZ semantic).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF11C_0108, 0x0040_0206, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_gather4_lz");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageGather4Lz);
        assert_eq!(inst.format, F::Vdata4Vaddr3StSsDmask1);
        assert_eq!(inst.dst.size, 4);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28; // 2D
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_gather4_lz");
        assert!(
            source.contains("OpImageGather %v4float"),
            "gather op:\n{source}"
        );
        assert!(
            source.contains("%g4_c_0 %uint_0"),
            "component 0 from dmask 1:\n{source}"
        );
        for k in 0..4 {
            assert!(
                source.contains(&format!("OpStore %v{} %g4_v_0_{k}", 2 + k)),
                "texel {k} store:\n{source}"
            );
        }
        // NOTE: assemble-only — naga's SPIR-V front end does not support
        // OpImageGather (UnsupportedInstruction), which real Vulkan accepts.
        let _ = spirv_run(&source).expect("assemble image_gather4_lz");
    }

    /// `image_gather4_lz` with each single-bit dmask must translate all the
    /// way to spirv-val-clean SPIR-V, and the dmask bit index MUST land in
    /// `OpImageGather`'s `Component` operand — dmask 0x2 is the measured
    /// ASTRO.BOT blocker ("unknown mimg format for opcode: 0x47 ... dmask:
    /// 0x2").
    #[test]
    fn image_gather4_lz_dmask_selects_the_gather_component() {
        for (word0, dmask, component) in [
            (0xF11C_0108u32, 0x1, 0u32),
            (0xF11C_0208, 0x2, 1),
            (0xF11C_0408, 0x4, 2),
            (0xF11C_0808, 0x8, 3),
        ] {
            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Compute);
            shader_parse(
                0,
                &[word0, 0x0040_0206, V_MOV_V0_0, S_ENDPGM],
                &mut code,
                true,
            )
            .unwrap_or_else(|e| panic!("parse gather4_lz dmask {dmask:#x}: {e:?}"));
            let inst = &code.get_instructions()[0];
            assert_eq!(inst.type_, T::ImageGather4Lz);
            // Always four destination dwords: one per gathered texel.
            assert_eq!(inst.dst.size, 4, "dmask {dmask:#x}");

            let mut input_info = ShaderComputeInputInfo::default();
            input_info.threads_num = [1, 1, 1];
            input_info.bind.push_constant_size = 64;
            input_info.bind.textures2d.textures_num = 1;
            input_info.bind.textures2d.textures2d_sampled_num = 1;
            input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28; // 2D
            input_info.bind.samplers.samplers_num = 1;
            input_info.bind.samplers.start_register[0] = 8;
            let source = spirv_generate_source(&code, None, None, Some(&input_info))
                .unwrap_or_else(|e| panic!("recompile gather4_lz dmask {dmask:#x}: {e:?}"));
            assert!(
                source.contains(&format!("%g4_c_0 %uint_{component}")),
                "dmask {dmask:#x} must gather component {component}:\n{source}"
            );
            // Four texel stores into v[2:5], independent of the dmask.
            for k in 0..4 {
                assert!(
                    source.contains(&format!("OpStore %v{} %g4_v_0_{k}", 2 + k)),
                    "dmask {dmask:#x} texel {k} store:\n{source}"
                );
            }
            let words = spirv_run(&source)
                .unwrap_or_else(|e| panic!("assemble gather4_lz dmask {dmask:#x}: {e:?}"));
            // naga cannot gate this module (its SPIR-V front end rejects
            // OpImageGather outright); spirv-val is the real bar.
            spirv_val_ok(&words, &format!("image_gather4_lz_dmask{dmask:#x}"));
        }
    }

    /// `s_load_dwordx16` (SMEM opcode 0x04) — Avatar: Frontiers of Pandora's
    /// first shader blocker. All SIXTEEN destination SGPRs must materialize;
    /// a width bug would silently drop dwords 8..16.
    #[test]
    fn s_load_dwordx16_materializes_all_sixteen_dwords() {
        // s_load_dwordx16 s[16:31], s[12:13], 0x100 — the EUD base pair with a
        // constant byte offset (the shape `sload_dword_extended` accepts).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF410_0406, 0xFA00_0100, V_MOV_V0_0, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse s_load_dwordx16");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::SLoadDwordx16);
        assert_eq!(inst.format, F::Sdst16SbaseSoffset);
        assert_eq!((inst.dst.register_id, inst.dst.size), (16, 16));
        assert_eq!((inst.src[0].register_id, inst.src[0].size), (12, 2));

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 12;
        input_info.bind.storage_buffers.extended[0] = true;

        // No captured descriptor covers EUD dword 0x40..0x50, so the raw
        // window must be sized to the load's LAST dword — evidence the x16
        // width reached the detection pass, not just the parser.
        crate::shader::spirv::shader_detect_eud_raw_window(&code, &mut input_info.bind);
        let raw = input_info.bind.eud_raw;
        assert!(raw.used);
        assert_eq!(
            raw.required_dwords,
            0x100 / 4 + 16,
            "raw window must cover all 16 loaded dwords"
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile s_load_dwordx16");
        assert_eq!(
            source.matches("OpArrayLength %uint %eud_raw 0").count(),
            16,
            "one clamped read per loaded dword:\n{source}"
        );
        // Destinations are s16..s31 — the last one proves the width.
        for i in 0..16 {
            assert!(
                source.contains(&format!("OpStore %s{} %eudraw_res_0_{i}", 16 + i)),
                "dword {i} store:\n{source}"
            );
        }
        let words = spirv_run(&source).expect("assemble s_load_dwordx16");
        spirv_val_ok(&words, "s_load_dwordx16");
    }

    #[test]
    fn astro_image_sample_l_uses_explicit_vgpr_lod() {
        // Captured ASTRO.BOT opcode word 0xf0900718: image_sample_l dmask:7.
        // The compact fixture keeps the known descriptor tuple and supplies
        // XY followed by the explicit LOD in the third VADDR register.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF090_0718, 0x0040_0206, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_sample_l");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageSampleL);
        assert_eq!(inst.format, F::Vdata3Vaddr4StSsDmask7);
        assert_eq!(inst.src[0].size, 4);
        assert_eq!(inst.dst.size, 3);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28; // 2D
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        input_info.bind.samplers.binding_index = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_sample_l");
        assert!(
            source.contains("OpImageSampleExplicitLod %v4float"),
            "explicit sample:\n{source}"
        );
        assert!(source.contains("Lod %isl_lod_0"), "LOD operand:\n{source}");
        assert!(
            source.contains("%isl_lod_0 = OpLoad %float"),
            "VGPR LOD load:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble image_sample_l");
        naga_parse_and_validate(&words, "image_sample_l");
    }

    #[test]
    fn astro_image_gather4_lz_dmask2_gathers_green_channel() {
        // ASTRO.BOT pairs raw 0xf11c0108 and 0xf11c0208 gathers in four
        // captured scene-compute shaders. Keep the compact fixture's known
        // descriptor registers while exercising the exact dmask-2 opcode
        // word: four green-channel texels must land in four VGPRs.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF11C_0208, 0x0040_0206, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_gather4_lz dmask2");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageGather4Lz);
        assert_eq!(inst.format, F::Vdata4Vaddr3StSsDmask2);
        assert_eq!(inst.dst.size, 4);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28; // 2D
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile image_gather4_lz dmask2");
        assert!(
            source.contains("OpImageGather %v4float"),
            "gather op:\n{source}"
        );
        assert!(
            source.contains("%g4_c_0 %uint_1"),
            "component 1 from dmask 2:\n{source}"
        );
        for k in 0..4 {
            assert!(
                source.contains(&format!("OpStore %v{} %g4_v_0_{k}", 2 + k)),
                "texel {k} store:\n{source}"
            );
        }
        let _ = spirv_run(&source).expect("assemble image_gather4_lz dmask2");
    }

    #[test]
    fn astro_sdwa_lane_selects_extract_bytes_and_words() {
        // vop1 v_mov_b32 v1, v5 src0_sel:BYTE_1 and vopc v_cmp_lt_f32
        // vcc, v1, v2 src1_sel:BYTE_0 — the measured SDWA sub-dword lane
        // selects (ASTRO.BOT scene compute: "vop1 sdwa src0_sel != 6" /
        // "vopc sdwa src1_sel != 6"). Each lane is extracted with
        // shift + mask, zero-extended.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0x7E02_02F9, // v_mov_b32 v1, sdwa
                0x0001_0605, // src0=v5, dst_sel=DWORD, src0_sel=BYTE_1
                0x7C02_04F9, // v_cmp_lt_f32 vcc, sdwa, v2
                0x0006_0001, // src0=v1 (sel DWORD), src1_sel=BYTE_0
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse SDWA lane selects");
        assert_eq!(code.get_instructions()[0].src[0].lane_sel, 1);
        assert_eq!(code.get_instructions()[1].src[1].lane_sel, 0);
        assert_eq!(code.get_instructions()[1].src[0].lane_sel, 6);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile SDWA lane selects");
        // BYTE_1: shift 8 then mask 0xff; BYTE_0: shift 0 then mask 0xff.
        assert!(
            source.contains("OpShiftRightLogical %uint %rt0_0 %uint_8"),
            "byte_1 shift:\n{source}"
        );
        assert!(
            source.contains("OpShiftRightLogical %uint %rt1_1 %uint_0"),
            "byte_0 shift:\n{source}"
        );
        assert!(source.contains("%uint_255"), "byte mask:\n{source}");
        let words = spirv_run(&source).expect("assemble SDWA lane selects");
        naga_parse_and_validate(&words, "sdwa lane selects");
    }

    #[test]
    fn astro_ds_append_with_offset_adds_counter_dwords() {
        // ds_append v7 offset:4 (gds) — the counter one dword past the M0
        // base (shadPS4 DS_APPEND: gds_offset = M0 + inst_offset, indexed at
        // >> 2). The zero-offset indexing (m0 >> 16 used directly) must stay
        // byte-identical for existing shaders.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0xBEFC_0380, // s_mov_b32 m0, 0
                0xD8FA_0004, // ds_append v7, offset 4, gds
                0x0700_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_append with offset");

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.gds_pointers.pointers_num = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_append with offset");
        assert!(
            source.contains("OpShiftRightLogical %uint %uint_4 %uint_2"),
            "byte offset -> dword count:\n{source}"
        );
        assert!(
            source.contains("%t195_1 = OpIAdd %uint %t194_1 %t193_1"),
            "offset added to the m0 base index:\n{source}"
        );
        assert!(source.contains("OpAtomicIAdd"), "{source}");
        // Assemble-only: naga's SPIR-V frontend rejects the free-standing
        // `OpMemoryBarrier` Kyty's append/consume body has always emitted
        // (UnsupportedInstruction) — a pre-existing upstream construct, not
        // part of this offset extension. spirv-tools accepts it.
        let _ = spirv_run(&source).expect("assemble ds_append with offset");
    }

    /// Gap 3: a `ds_append` with NO captured GDS descriptor (the real ASTRO.BOT
    /// tiled-lighting case — the counter is addressed through M0, so no usage
    /// slot produces a GDS pointer). Without a pointer, `Recompile_DsAppend`
    /// returns false → "can't recompile: DsAppend". `shader_synthesize_gds_pointer`
    /// adds one so `%gds` is declared/bound and the append lowers.
    #[test]
    fn ds_append_synthesizes_gds_pointer_when_uncaptured() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                0xBEFC_0380, // s_mov_b32 m0, 0
                0xD8FA_0004, // ds_append v7, offset 4, gds
                0x0700_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse ds_append");

        // No GDS pointer captured — the pre-fix path refuses here.
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        assert_eq!(input_info.bind.gds_pointers.pointers_num, 0);
        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("append without a GDS pointer must refuse");
        assert!(
            format!("{err}").contains("can't recompile"),
            "baseline refusal, got: {err}"
        );

        // Synthesis adds the pointer; the append now lowers to the GDS atomic.
        crate::shader::analysis::shader_synthesize_gds_pointer(&code, &mut input_info.bind);
        assert_eq!(input_info.bind.gds_pointers.pointers_num, 1);
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile ds_append after GDS-pointer synthesis");
        assert!(
            source.contains("OpAtomicIAdd"),
            "GDS append atomic:\n{source}"
        );
        assert!(source.contains("%gds"), "%gds must be declared:\n{source}");
        let _ = spirv_run(&source).expect("assemble ds_append after synthesis");
    }

    #[test]
    fn astro_mubuf_folded_offset_lands_in_temp_int_2() {
        // buffer_load_dword v4, v0, s[4:7] idxen offset:16 — the immediate
        // offset folded into the constant soffset must reach the address
        // model's instruction-offset slot (temp_int_2).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xE030_2010, 0x8001_0400, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse buffer_load_dword with immediate offset");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.storage_buffers.buffers_num = 1;
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile buffer_load_dword with immediate offset");
        assert!(
            source.contains("OpStore %temp_int_2 %int_16"),
            "folded offset in the instruction-offset slot:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble buffer_load_dword with offset");
        naga_parse_and_validate(&words, "buffer_load_dword with immediate offset");
    }

    #[test]
    fn astro_vop2_sdwa_omod_recompiles_as_float_multiply() {
        // v_mul_f32 SDWA omod=1 (mul:2) — same MULTIPLY lowering the VOP1
        // SDWA path already validates.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0x1002_06F9, 0x1606_4604, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse VOP2 SDWA omod");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("recompile VOP2 SDWA omod");
        assert!(source.contains("OpFMul %float %m197_0"), "{source}");
        let words = spirv_run(&source).expect("assemble VOP2 SDWA omod");
        naga_parse_and_validate(&words, "VOP2 SDWA omod");
    }

    /// Every emitted `%lds` access chain must take a `UMin`-clamped index —
    /// device-loss defusal sub-fix (ii): SharpEmu masks every LDS dword
    /// address into the array bound
    /// (`reference/sharpemu/src/SharpEmu.ShaderCompiler.Vulkan/`
    /// `Gen5SpirvTranslator.cs` L2007-2024, `LdsPointer`); this port clamps
    /// with `UMin` because `lds_size_dw` (128-dword granules) need not be a
    /// power of two, so a bitmask would be incorrect without padding the
    /// array. One fixture per wired DS family.
    #[test]
    fn every_ds_family_lds_access_is_umin_clamped() {
        fn assert_lds_clamped(name: &str, words: &[u32]) {
            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Compute);
            shader_parse(0, words, &mut code, true).unwrap_or_else(|e| panic!("parse {name}: {e}"));
            let mut input_info = ShaderComputeInputInfo::default();
            input_info.threads_num = [1, 1, 1];
            let source = spirv_generate_source(&code, None, None, Some(&input_info))
                .unwrap_or_else(|e| panic!("recompile {name}: {e}"));
            let mut accesses = 0usize;
            for line in source.lines() {
                let Some(pos) = line.find("OpAccessChain %_ptr_Workgroup_uint %lds ") else {
                    continue;
                };
                accesses += 1;
                let index_id = line[pos..]
                    .split_whitespace()
                    .last()
                    .unwrap_or_else(|| panic!("{name}: malformed LDS access line: {line}"));
                let def = format!("{index_id} = OpExtInst %uint %GLSL_std_450 UMin ");
                assert!(
                    source.contains(&def),
                    "{name}: LDS access index {index_id} has no UMin clamp:\n{source}"
                );
            }
            assert!(
                accesses > 0,
                "{name}: fixture emitted no LDS access:\n{source}"
            );
        }

        // (v_mov preambles seed the address/data VGPRs; every DS word pair is
        // a measured ASTRO.BOT encoding reused from the per-op tests above.)
        assert_lds_clamped(
            "ds_write_b32 + ds_read_b32",
            &[
                0x7E00_0280,
                0x7E02_0280,
                0xD834_0000,
                0x0000_0100,
                0xBF8A_0000,
                0xD8D8_0000,
                0x0200_0000,
                S_ENDPGM,
            ],
        );
        assert_lds_clamped(
            "ds_add_u32",
            &[0x7E00_0280, 0x7E02_0280, 0xD800_0514, 0x0000_0100, S_ENDPGM],
        );
        assert_lds_clamped(
            "ds_write_b96 + ds_read2_b32",
            &[
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                0xDB78_0008,
                0x0000_0100,
                0xD8DC_0100,
                0x0400_0000,
                S_ENDPGM,
            ],
        );
        assert_lds_clamped(
            "ds_read_b64",
            &[0x7E00_0280, 0xD9D8_0010, 0x0200_0000, S_ENDPGM],
        );
        assert_lds_clamped(
            "ds_write_b128",
            &[
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                0x7E08_0280,
                0xDB7C_0008,
                0x0000_0100,
                S_ENDPGM,
            ],
        );
        assert_lds_clamped(
            "ds_read_b128",
            &[0x7E00_0280, 0xDBFC_0010, 0x0200_0000, S_ENDPGM],
        );
        assert_lds_clamped(
            "ds_read_b96",
            &[0x7E00_0280, 0xDBF8_0010, 0x0200_0000, S_ENDPGM],
        );
    }

    /// Device-loss defusal sub-fix (i): a shader that SAMPLES with zero
    /// captured S#s gets a synthesized all-zero (nearest/wrap) default
    /// sampler at the sample's S# register instead of a whole-shader
    /// refusal (SharpEmu VulkanVideoPresenter.cs L6314-6322 + L8121-8156).
    #[test]
    fn sampler_less_image_sample_synthesizes_default_nearest_sampler() {
        use crate::shader::analysis::shader_synthesize_default_sampler;

        // image_sample_c_lz with T# at s[4:11] and S# at s[12:15].
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
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        crate::shader::analysis::shader_calc_binding_indices(&mut input_info.bind);

        // Without synthesis: sample-family bodies bail on samplers_num == 0
        // and the shader refuses; with it, one all-zero S# lands at s12.
        shader_synthesize_default_sampler(&code, &mut input_info.bind);
        assert_eq!(input_info.bind.samplers.samplers_num, 1);
        assert_eq!(input_info.bind.samplers.start_register[0], 12);
        assert_eq!(input_info.bind.samplers.samplers[0].fields, [0; 4]);
        assert!(!input_info.bind.samplers.extended[0]);
        // Binding indices/push constants recomputed: 1 T# (32 B) + 1 S# (16 B).
        assert_eq!(input_info.bind.push_constant_size, 48);

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("sample-family shader recompiles with the synthesized default S#");
        assert!(source.contains("%samplers = OpVariable"), "{source}");

        // A texel-fetch-only shader (image_load) must stay untouched:
        // OpImageFetch needs no sampler.
        let mut fetch_code = ShaderCode::new();
        fetch_code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF000_0300, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut fetch_code,
            true,
        )
        .expect("parse image_load");
        let mut fetch_bind = input_info.bind;
        fetch_bind.samplers = Default::default();
        shader_synthesize_default_sampler(&fetch_code, &mut fetch_bind);
        assert_eq!(
            fetch_bind.samplers.samplers_num, 0,
            "image_load must not synthesize a sampler"
        );
    }

    /// Device-loss defusal sub-fix (iii): an MIMG whose T# registers match no
    /// captured descriptor is a runtime-resolved descriptor — refused with
    /// the named `dynamic-image-descriptor` error (SharpEmu
    /// Gen5ShaderScalarEvaluator.cs L654-662), which the dispatch path turns
    /// into a counted skip instead of submitting an OOB descriptor index.
    #[test]
    fn unmatched_tsharp_refuses_as_dynamic_image_descriptor() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // image_load dmask3 with T# at s[4:11]...
        shader_parse(
            0,
            &[0xF000_0300, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_load dmask3");
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        // ...but the captured T# lives at s[8:15] — mismatch.
        input_info.bind.textures2d.desc[0].start_register = 8;
        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("unmatched T# must refuse");
        assert!(
            err.to_string().contains("dynamic-image-descriptor"),
            "named refusal expected, got: {err}"
        );
    }

    /// Safe-degradation counterpart of the refusal above: after
    /// `shader_synthesize_placeholder_sampled_texture` pre-registers a 1x1
    /// placeholder T# at the MIMG's unmatched register, the guard resolves it
    /// and the shader recompiles instead of skipping every dispatch. Measured
    /// on ASTRO.BOT scene compute 0x500566b00 (image_load T# at s16, 13 skips
    /// per level transition; the register/dmask are cosmetic — the fix is
    /// independent of both).
    #[test]
    fn synthesized_placeholder_texture_resolves_unmatched_sampled_tsharp() {
        use crate::shader::analysis::{
            placeholder_texture_fields, shader_synthesize_placeholder_sampled_texture,
        };

        // Same image_load the refusal test uses; its T# operand matches no
        // captured descriptor.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF000_0300, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_load");

        // One captured 2D sampled texture at s16 makes image_load take the
        // sampled path so the descriptor guard runs; the MIMG's own T# register
        // (s4..s11) has nothing captured there and does not overlap s16..s23.
        let mut base = ShaderComputeInputInfo::default();
        base.threads_num = [1, 1, 1];
        base.bind.push_constant_size = 128;
        base.bind.textures2d.textures_num = 1;
        base.bind.textures2d.textures2d_sampled_num = 1;
        base.bind.textures2d.desc[0].start_register = 16;
        base.bind.textures2d.desc[0].texture.fields = placeholder_texture_fields();

        // Baseline: no synthesis => the whole shader refuses by name.
        let err = spirv_generate_source(&code, None, None, Some(&base))
            .expect_err("unmatched sampled T# refuses without synthesis");
        assert!(
            err.to_string().contains("dynamic-image-descriptor"),
            "named refusal expected, got: {err}"
        );

        // With synthesis: a placeholder is captured at the MIMG's T# register,
        // the guard resolves it, and the shader recompiles.
        let t_reg = code.get_instructions()[0].src[1].register_id;
        let mut fixed = base;
        shader_synthesize_placeholder_sampled_texture(&code, &mut fixed.bind);
        assert_eq!(
            fixed.bind.textures2d.textures_num, 2,
            "captured s8 texture plus one synthesized placeholder"
        );
        assert!(
            fixed.bind.textures2d.desc[..2]
                .iter()
                .any(|d| d.start_register == t_reg && !d.textures2d_without_sampler),
            "a sampled placeholder is captured at the MIMG's T# register s{t_reg}"
        );
        let source = spirv_generate_source(&code, None, None, Some(&fixed))
            .expect("synthesized placeholder lets the previously-skipped shader recompile");
        // And the produced module assembles (valid SPIR-V, not just a
        // non-refusal): the placeholder's seeded `%vsharp_s{t_reg}` indexes the
        // grown, bound `%textures2D_S` array in-bounds.
        spirv_run(&source).expect("placeholder image_load assembles to valid SPIR-V");
    }

    /// The synthesis must NOT shadow a real descriptor: when the MIMG's T#
    /// register IS already captured, no placeholder is added (idempotent) and
    /// the real descriptor stands.
    #[test]
    fn synthesized_placeholder_texture_leaves_matched_tsharp_alone() {
        use crate::shader::analysis::shader_synthesize_placeholder_sampled_texture;

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF000_0300, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_load");
        let t_reg = code.get_instructions()[0].src[1].register_id;

        let mut info = ShaderComputeInputInfo::default();
        info.threads_num = [1, 1, 1];
        info.bind.textures2d.textures_num = 1;
        info.bind.textures2d.textures2d_sampled_num = 1;
        info.bind.textures2d.desc[0].start_register = t_reg;
        crate::shader::analysis::shader_calc_binding_indices(&mut info.bind);

        shader_synthesize_placeholder_sampled_texture(&code, &mut info.bind);
        assert_eq!(
            info.bind.textures2d.textures_num, 1,
            "the already-captured T# gets no shadowing placeholder"
        );
    }

    /// Regression guard for the REVERTED rank-8 storage-placeholder fallback
    /// (unwired in `raeen-gpu` `shader_fetch.rs::translate_cs`).
    ///
    /// A compute shader that BOTH samples (`image_load`) and stores
    /// (`image_store`) through the SAME T# register is the shape that regressed
    /// ASTRO.BOT compute from 0 translation failures to 30. When
    /// `shader_synthesize_placeholder_storage_texture` runs after the sampled
    /// pass, its coverage check (`without_sampler` direct match + EUD alias) does
    /// NOT see the sampled placeholder already parked at that register, so it
    /// parks a SECOND `texture2D` there. `WriteLocalVariables` must use
    /// descriptor-slot-qualified temporary ids while storing both snapshots
    /// into the shared SGPR. Register-qualified ids used to define
    /// `%vsharp_s0` twice and made ASTRO.BOT 0x5006e7a00 / 0x5006ea100
    /// assembly-fatal.
    ///
    /// This pins BOTH facts: stacking the two passes targets one SGPR twice,
    /// and descriptor-slot-qualified ids keep that legal. The storage pass
    /// remains unwired because choosing which aliased descriptor wins is a
    /// semantic question, not an assembly one.
    #[test]
    fn storage_placeholder_at_occupied_register_uses_unique_seed_ids() {
        use crate::shader::analysis::{
            shader_synthesize_placeholder_sampled_texture,
            shader_synthesize_placeholder_storage_texture,
        };

        // image_load (sampled) + image_store (storage), both reading their T#
        // from the SAME register s0.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF000_0300, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_load");
        let mut store = ShaderCode::new();
        store.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[0xF020_0100, 0x0060_0800, 0xBF80_0000, S_ENDPGM],
            &mut store,
            true,
        )
        .expect("parse image_store");
        let store_inst = store.get_instructions()[0];
        assert_eq!(store_inst.type_, T::ImageStore);
        // Insert the store just before the load's s_endpgm.
        code.get_instructions_mut().insert(1, store_inst);
        // Force both MIMG T# operands to s0 to reproduce the collision.
        code.get_instructions_mut()[0].src[1].register_id = 0;
        code.get_instructions_mut()[1].src[1].register_id = 0;
        assert_eq!(code.get_instructions()[0].type_, T::ImageLoad);
        assert_eq!(code.get_instructions()[1].type_, T::ImageStore);

        let base = || {
            let mut info = ShaderComputeInputInfo::default();
            info.threads_num = [1, 1, 1];
            info.bind.push_constant_size = 128;
            info
        };

        // Two-pass path (the reverted wiring): sampled placeholder then storage
        // placeholder both park a texture2D at s0.
        let mut both = base();
        shader_synthesize_placeholder_sampled_texture(&code, &mut both.bind);
        shader_synthesize_placeholder_storage_texture(&code, &mut both.bind);
        let tex_num = both.bind.textures2d.textures_num as usize;
        let at_s0 = both.bind.textures2d.desc[..tex_num]
            .iter()
            .filter(|d| d.start_register == 0)
            .count();
        assert_eq!(
            at_s0, 2,
            "both passes park a descriptor at s0 (the collision)"
        );
        let dup = spirv_generate_source(&code, None, None, Some(&both))
            .expect("both descriptors resolve");
        assert_eq!(
            dup.matches("OpStore %s0 %vsharp_value_").count(),
            2,
            "both aliased descriptors seed s0 in descriptor order: {dup}"
        );
        spirv_run(&dup).expect("descriptor-slot-qualified seeding must assemble");

        // Retained sampled-only path (shipped wiring after the revert): the
        // image_store's unresolved T# refuses by name — never an invalid module,
        // never a duplicate id.
        let mut sampled_only = base();
        shader_synthesize_placeholder_sampled_texture(&code, &mut sampled_only.bind);
        match spirv_generate_source(&code, None, None, Some(&sampled_only)) {
            Ok(src) => {
                assert_eq!(
                    src.matches("OpStore %s0 %vsharp_value_").count(),
                    1,
                    "the single sampled placeholder seeds %vsharp_s0 exactly once: {src}"
                );
                spirv_run(&src).expect("sampled-only module assembles to valid SPIR-V");
            }
            Err(e) => {
                // `spirv_generate_source` refuses BEFORE assembly, so any Err
                // here is a clean named refusal (guard `dynamic-image-descriptor`
                // or recompiler `can't recompile: ImageStore`) — never the
                // duplicate-id / invalid-module the storage placeholder produced.
                let msg = e.to_string();
                assert!(
                    !msg.contains("duplicate"),
                    "sampled-only path must refuse cleanly, never a duplicate id: {msg}"
                );
            }
        }
    }

    /// Device-loss defusal sub-fix (iii), the measured 0x5006c5f00 shape: the
    /// MIMG's T# registers ARE captured, but a raw (uncovered-EUD)
    /// `s_load_dwordx8` overwrites them with raw guest dwords — the seeded
    /// descriptor-array index is destroyed, so the shader refuses by name
    /// instead of device-lossing on an OOB descriptor index.
    #[test]
    fn raw_eud_overwrite_of_tsharp_regs_refuses_as_dynamic_image_descriptor() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // image_load dmask3 with T# at s[4:11].
        shader_parse(
            0,
            &[0xF000_0300, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_load dmask3");
        // s_load_dwordx8 s[4:11], s[12:13], 0x148 — EUD dwords 82..90, beyond
        // every captured descriptor (raw-window reads), landing exactly on
        // the T# registers.
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 4,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0x148),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().insert(0, sload);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 4;
        crate::shader::spirv::shader_detect_eud_raw_window(&code, &mut input_info.bind);
        assert!(
            input_info.bind.eud_raw.used,
            "the s_load must be detected as a raw-window read"
        );

        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("raw overwrite of T# registers must refuse");
        assert!(
            err.to_string().contains("dynamic-image-descriptor"),
            "named refusal expected, got: {err}"
        );
    }

    /// EUD-alias resolution (the measured 0x5006c5f00 headline): the storage
    /// T# is captured at its EUD-VIRTUAL start register (`s32` = EUD dword 0),
    /// but the `image_store` reads its T# from `s0`, which a COVERED
    /// `s_load_dwordx8 s[0:7], s[12:13], 0x0` fills at runtime. The register's
    /// dword 0 is the rewritten storage-array index (`sload_dword_extended` +
    /// `GetMappedIndex`), so the descriptor resolves instead of refusing.
    #[test]
    fn eud_alias_storage_tsharp_resolves_via_covered_load() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // image_store dmask1 reading its storage T# from s[0:7] (srsrc = 0).
        shader_parse(
            0,
            &[0xF020_0100, 0x0060_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_store dmask1 (T# at s0)");
        assert_eq!(code.get_instructions()[0].type_, T::ImageStore);
        assert_eq!(code.get_instructions()[0].src[1].register_id, 0);
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 0,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().insert(0, sload);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        // Captured at the EUD-virtual start register (s32 = EUD dword 0), NOT s0.
        input_info.bind.textures2d.desc[0].start_register = 32;
        input_info.bind.textures2d.desc[0].extended = true;
        input_info.bind.textures2d.desc[0].textures2d_without_sampler = true;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("EUD-alias storage T# must resolve, not refuse");
        assert!(source.contains("OpImageWrite"), "{source}");
        let _ = spirv_run(&source).expect("assemble EUD-alias image_store");
    }

    /// MIXED-dim routing through the EUD-alias walk: the 3D T# is captured at
    /// its EUD-VIRTUAL start register (`s32` = EUD dword 0) while the sample
    /// reads it from `s16`, filled by a covered `s_load_dwordx8`. The route
    /// must use the guard's alias resolution — a bare start-register re-match
    /// would re-refuse exactly the aliased descriptors the walk was built to
    /// accept (ASTRO.BOT, refusals 642->90) and the mixed-dims failure class
    /// would survive under a new name. The direct-register 2D sample and the
    /// aliased 3D sample each land in their own dim's array.
    #[test]
    fn eud_alias_sampled_tsharp_routes_to_its_dim_in_mixed_shader() {
        use crate::shader::analysis::shader_calc_binding_indices;
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // Covered EUD load: s_load_dwordx8 s[16:23], s[12:13], 0x0 — fills
        // s16.. with EUD dwords 0..7 (the 3D T#'s rewritten descriptor).
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 16,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        let sample = |t_reg: i32, dst_reg: i32| ShaderInstruction {
            type_: T::ImageSampleLz,
            format: F::Vdata4Vaddr3StSsDmaskF,
            src_num: 3,
            dst: ShaderOperand {
                type_: ShaderOperandType::Vgpr,
                register_id: dst_reg,
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
                    register_id: t_reg,
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
        code.get_instructions_mut().push(sload);
        code.get_instructions_mut().push(sample(0, 2)); // 2D, direct T# at s0
        code.get_instructions_mut().push(sample(16, 10)); // 3D, EUD-aliased
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.textures2d.textures_num = 2;
        input_info.bind.textures2d.textures2d_sampled_num = 2;
        // T# 0: 2D (type 9) captured directly at s0..s7.
        input_info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28;
        input_info.bind.textures2d.desc[0].start_register = 0;
        // T# 1: 3D (type 10) captured at the EUD-virtual s32 (EUD dword 0).
        input_info.bind.textures2d.desc[1].texture.fields[3] |= 10 << 28;
        input_info.bind.textures2d.desc[1].start_register = 32;
        input_info.bind.textures2d.desc[1].extended = true;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        shader_calc_binding_indices(&mut input_info.bind);

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("EUD-aliased 3D T# in a mixed shader must route, not refuse");
        assert!(
            source.contains("OpAccessChain %_ptr_UniformConstant_ImageS_2D %textures2D_S_2D"),
            "the direct 2D sample routes to the 2D array:\n{source}"
        );
        assert!(
            source.contains("OpAccessChain %_ptr_UniformConstant_ImageS_3D %textures2D_S_3D"),
            "the EUD-aliased 3D sample routes to the 3D array:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble mixed EUD-alias sample");
        naga_parse_and_validate(&words, "mixed EUD-alias sample");
    }

    /// EUD-alias resolution is offset-exact — the device-loss guard: the
    /// `image_store` reads its storage T# from `s0`, but the covered load
    /// fills `s0` from EUD dword 0 (a RAW-window read — no captured descriptor
    /// covers it), while the only storage descriptor maps at EUD dword 8
    /// (`s40`). No captured descriptor's rewritten index lands in `s0`, so
    /// `s0` holds a RAW guest dword; the alias must NOT match and the refusal
    /// stands rather than submitting an out-of-bounds descriptor index.
    #[test]
    fn eud_alias_rejects_offset_mismatch() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // image_store dmask1 reading its storage T# from s[0:7] (srsrc = 0).
        shader_parse(
            0,
            &[0xF020_0100, 0x0060_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_store dmask1 (T# at s0)");
        // s_load_dwordx8 s[0:7], s[12:13], 0x0 — EUD dword 0, uncovered.
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 0,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().insert(0, sload);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        // The only storage descriptor maps at EUD dword 8 (s40), not the
        // loaded dword 0 — the load reads the raw window into s0 instead.
        input_info.bind.textures2d.desc[0].start_register = 40;
        input_info.bind.textures2d.desc[0].extended = true;
        input_info.bind.textures2d.desc[0].textures2d_without_sampler = true;
        // Declare the raw window so the uncovered load recompiles (reaching the
        // MIMG guard) instead of refusing at the load itself.
        crate::shader::spirv::shader_detect_eud_raw_window(&code, &mut input_info.bind);
        assert!(
            input_info.bind.eud_raw.used,
            "the uncovered load must be detected as a raw-window read"
        );

        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("offset-mismatched EUD alias must refuse");
        assert!(
            err.to_string().contains("dynamic-image-descriptor"),
            "named refusal expected, got: {err}"
        );
    }

    /// EUD-alias resolution through a SALU mov chain (the measured sampled shape
    /// `0x5006fff00`): the MIMG's T# register is NOT the direct `s_load` dest —
    /// a covered `s_load_dwordx8 s[8:15], s[12:13], 0x0` fills `s8`, then
    /// `s_mov_b32 s0, s8` forwards its dword 0, and the `image_load` reads its
    /// sampled T# from `s0`. The mov copies the rewritten descriptor-array index
    /// (`recompile_smov_b32` is a straight `OpStore`), so the descriptor
    /// resolves via program-order backtracking instead of refusing.
    #[test]
    fn sampled_tsharp_resolves_via_mov_chain() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // image_load dmask3 reading its sampled T# from s[0:7] (srsrc = 0).
        shader_parse(
            0,
            &[0xF000_0300, 0x0060_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_load dmask3 (T# at s0)");
        assert_eq!(code.get_instructions()[0].type_, T::ImageLoad);
        assert_eq!(code.get_instructions()[0].src[1].register_id, 0);

        // Insert BEFORE the MIMG (program order): s_mov_b32 s0, s8.
        let smov = ShaderInstruction {
            type_: T::SMovB32,
            format: F::SVdstSVsrc0,
            src_num: 1,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 0,
                size: 1,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 8,
                    size: 1,
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().insert(0, smov);
        // Insert BEFORE the mov: s_load_dwordx8 s[8:15], s[12:13], 0x0.
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 8,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().insert(0, sload);
        // Order is now [sload s8, smov s0<-s8, image_load T#=s0].

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        // Captured SAMPLED texture at the EUD-virtual start register s32 (rel 0).
        input_info.bind.textures2d.desc[0].start_register = 32;
        input_info.bind.textures2d.desc[0].extended = true;
        input_info.bind.textures2d.desc[0].textures2d_without_sampler = false;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("mov-chain sampled T# must resolve, not refuse");
        assert!(source.contains("%textures2D_S"), "{source}");
        let _ = spirv_run(&source).expect("assemble mov-chain image_load");
    }

    /// Non-coupled sampler EUD-alias resolution (the measured `0x5006fff00`
    /// gather shape): the T# is captured directly, but the S# is delivered
    /// separately — the `image_sample`'s S# register is filled by a covered
    /// `s_load_dwordx4 s[12:15], s[20:21], 0x10` (EUD dword 4) whose offset
    /// matches a captured EXTENDED sampler at its EUD-virtual start register
    /// s36 (rel 4). The sampler resolves via the covered load instead of the
    /// direct start-register match, so the guard passes rather than refusing.
    #[test]
    fn sampler_resolves_via_covered_load() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // image_sample_lz dmask2: T# at s4, S# at s12 (same parse the channel
        // test uses).
        shader_parse(
            0,
            &[0xF09C_0200, 0x0061_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_sample_lz dmask2");
        assert_eq!(code.get_instructions()[0].src[2].register_id, 12);
        // Insert BEFORE the MIMG: s_load_dwordx4 s[12:15], s[20:21], 0x10.
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx4,
            format: F::Sdst4SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 12,
                size: 4,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 20,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0x10),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().insert(0, sload);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 20;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        // T# captured directly at s4 (no alias needed for it).
        input_info.bind.textures2d.desc[0].start_register = 4;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.binding_index = 1;
        // The sampler is captured at its EUD-virtual start register s36 (rel 4),
        // NOT the s12 the MIMG reads it from — resolved via the covered load.
        input_info.bind.samplers.start_register[0] = 36;
        input_info.bind.samplers.extended[0] = true;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("non-coupled sampler must resolve via covered load, not refuse");
        assert!(source.contains("%samplers"), "{source}");
        let _ = spirv_run(&source).expect("assemble non-coupled sampler image_sample");
    }

    /// Device-loss guard: an EUD-alias chain that runs through a NON-copy
    /// redefinition (here an immediate `s_mov_b32 s0, 0x1234`, standing in for
    /// any arithmetic) STILL refuses — even though a covered load into `s8`
    /// exists earlier, `s0` was redefined with a value that is NOT the rewritten
    /// descriptor-array index, so binding it would be an out-of-bounds
    /// descriptor read. The program-order walk stops at the immediate move
    /// (`Blocked`) and never falls back to the covered load.
    #[test]
    fn eud_alias_arithmetic_in_chain_still_refuses() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // image_store dmask1 reading its storage T# from s[0:7] (srsrc = 0).
        shader_parse(
            0,
            &[0xF020_0100, 0x0060_0800, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse image_store dmask1 (T# at s0)");
        // Insert BEFORE the MIMG: s_mov_b32 s0, 0x1234 (immediate redefinition).
        let smov_imm = ShaderInstruction {
            type_: T::SMovB32,
            format: F::SVdstSVsrc0,
            src_num: 1,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 0,
                size: 1,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0x1234),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().insert(0, smov_imm);
        // Insert BEFORE the mov: a COVERED s_load_dwordx8 s[8:15], s[12:13], 0x0
        // — a valid descriptor load, but into s8, NOT s0. It must NOT rescue the
        // immediate-redefined s0.
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 8,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::LiteralConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(0),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().insert(0, sload);
        // Order is now [sload s8, smov s0<-imm, image_store T#=s0].

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_storage_num = 1;
        // The storage descriptor maps at EUD dword 0 (s32) — so the covered
        // s8 load recompiles, but s0's value is the immediate, not this index.
        input_info.bind.textures2d.desc[0].start_register = 32;
        input_info.bind.textures2d.desc[0].extended = true;
        input_info.bind.textures2d.desc[0].textures2d_without_sampler = true;

        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("immediate redefinition in the chain must refuse");
        assert!(
            err.to_string().contains("dynamic-image-descriptor"),
            "named refusal expected, got: {err}"
        );
    }

    /// Device-loss defusal sub-fix (iii), parity case: a register-soffset
    /// `s_load_dwordx8` (a genuinely dynamic descriptor index) is a NAMED
    /// refusal — the dispatch path logs and skips; no panic, no submit.
    #[test]
    fn register_soffset_sload_dwordx8_is_named_refusal_not_panic() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let sload = ShaderInstruction {
            type_: T::SLoadDwordx8,
            format: F::Sdst8SbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 16,
                size: 8,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                // The dynamic part: the offset comes from a register, so the
                // loaded T# cannot be resolved at translate time.
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 20,
                    size: 1,
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().push(sload);
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 12;
        input_info.bind.storage_buffers.extended[0] = true;

        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("register-soffset s_load_dwordx8 must refuse");
        // The refusal now names the width, the format, the base and the
        // soffset register (it used to be the generic "src1 is not a constant
        // offset"), so a live log line identifies the exact unsupported form.
        let text = err.to_string();
        for want in [
            "unresolved register soffset",
            "SLoadDwordx8",
            "Sdst8SbaseSoffset",
            "x8 dwords",
            "base=s12",
            "soffset=s20",
        ] {
            assert!(text.contains(want), "refusal must name {want}, got: {text}");
        }
    }

    // ---- SMEM register soffset (RDNA2 `base + soffset + imm`) -------------

    /// A three-operand register-soffset load (`src[1]` soffset register,
    /// `src[2]` immediate) whose address analysis already resolved: the per-PC
    /// capture materializes the dwords, so the shader translates instead of
    /// dying at "no table entry for .../Sdst4SbaseSoffsetOffset".
    ///
    /// This is ASTRO.BOT's blocked shape (`offset != 0 with register soffset`,
    /// three compute shaders) taken all the way to validated SPIR-V.
    #[test]
    fn register_soffset_with_offset_materializes_a_resolved_capture() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // Twice: `Recompile_SEndpgm_Empty` inspects the two preceding
        // instructions, so a one-instruction body cannot reach s_endpgm.
        let load = ShaderInstruction {
            pc: 0x20,
            type_: T::SLoadDwordx4,
            format: F::Sdst4SbaseSoffsetOffset,
            src_num: 3,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 16,
                size: 4,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 12,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 4,
                    size: 1,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::IntegerInlineConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(16),
                    ..Default::default()
                },
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().push(load);
        code.get_instructions_mut().push(load);
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let values = [0x1111_1111u32, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        let loads = &mut input_info.bind.embedded_constant_loads;
        loads.loads_num = 1;
        loads.loads[0].pc = 0x20;
        loads.loads[0].dwords_num = 4;
        loads.loads[0].values[..4].copy_from_slice(&values);

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("a resolved register-soffset load materializes");
        for v in values {
            assert!(
                source.contains(&format!("{v}")) || source.contains(&format!("{v:#x}")),
                "each captured dword is stored as a constant:\n{source}"
            );
        }
        assert!(
            source.matches("OpStore").count() >= 4,
            "four destination SGPRs are written:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble register-soffset compute shader");
        spirv_val_ok(&words, "register_soffset_resolved_x4");
    }

    /// The same three-operand form with NOTHING resolved: a named refusal that
    /// carries the width, the format, the base and the soffset register, so the
    /// live log line identifies the exact addressing form still missing (and it
    /// is counted like every other shader error, being logged through the same
    /// recompile-failure path).
    #[test]
    fn unresolved_register_soffset_with_offset_refuses_naming_format_and_width() {
        for (type_, format, width) in [
            (T::SLoadDword, F::SdstSbaseSoffsetOffset, "x1"),
            (T::SLoadDwordx2, F::Sdst2SbaseSoffsetOffset, "x2"),
            (T::SLoadDwordx4, F::Sdst4SbaseSoffsetOffset, "x4"),
            (T::SLoadDwordx8, F::Sdst8SbaseSoffsetOffset, "x8"),
            (T::SLoadDwordx16, F::Sdst16SbaseSoffsetOffset, "x16"),
        ] {
            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Compute);
            code.get_instructions_mut().push(ShaderInstruction {
                pc: 0x24,
                type_,
                format,
                src_num: 3,
                dst: ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 40,
                    size: 1,
                    ..Default::default()
                },
                src: [
                    ShaderOperand {
                        type_: ShaderOperandType::Sgpr,
                        register_id: 12,
                        size: 2,
                        ..Default::default()
                    },
                    ShaderOperand {
                        type_: ShaderOperandType::Sgpr,
                        register_id: 7,
                        size: 1,
                        ..Default::default()
                    },
                    ShaderOperand {
                        type_: ShaderOperandType::IntegerInlineConstant,
                        constant: crate::shader::types::ShaderConstant::from_u(0x50),
                        ..Default::default()
                    },
                    ShaderOperand::default(),
                ],
                ..Default::default()
            });
            code.get_instructions_mut().push(ShaderInstruction {
                type_: T::SEndpgm,
                format: F::Empty,
                ..Default::default()
            });

            let mut input_info = ShaderComputeInputInfo::default();
            input_info.threads_num = [1, 1, 1];
            input_info.bind.push_constant_size = 16;
            input_info.bind.extended.used = true;
            input_info.bind.extended.start_register = 12;

            let err = spirv_generate_source(&code, None, None, Some(&input_info))
                .expect_err("an unresolved register soffset must refuse");
            let text = err.to_string();
            for want in [
                "unresolved register soffset".to_string(),
                format!("{format:?}"),
                format!("{width} dwords"),
                "soffset=s7".to_string(),
                "imm=0x50".to_string(),
            ] {
                assert!(
                    text.contains(&want),
                    "refusal must name {want}, got: {text}"
                );
            }
        }
    }

    /// `s_load_dword` (x1) used to be an embedded-fetch-only gate: every other
    /// single-dword scalar load returned `false` -> "can't recompile", failing
    /// the whole shader. It now shares the x2..x16 materialization.
    #[test]
    fn sload_dword_x1_materializes_instead_of_failing_the_shader() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let load = ShaderInstruction {
            pc: 0x08,
            type_: T::SLoadDword,
            format: F::SdstSbaseSoffset,
            src_num: 2,
            dst: ShaderOperand {
                type_: ShaderOperandType::Sgpr,
                register_id: 30,
                size: 1,
                ..Default::default()
            },
            src: [
                ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 10,
                    size: 2,
                    ..Default::default()
                },
                ShaderOperand {
                    type_: ShaderOperandType::IntegerInlineConstant,
                    constant: crate::shader::types::ShaderConstant::from_u(4),
                    ..Default::default()
                },
                ShaderOperand::default(),
                ShaderOperand::default(),
            ],
            ..Default::default()
        };
        code.get_instructions_mut().push(load);
        code.get_instructions_mut().push(load);
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        // Without a capture the load has no resolvable source: `false` ->
        // "can't recompile" (the pre-change behavior for EVERY x1 load).
        let mut bare = ShaderComputeInputInfo::default();
        bare.threads_num = [1, 1, 1];
        let err = spirv_generate_source(&code, None, None, Some(&bare))
            .expect_err("an unresolvable x1 load still fails");
        assert!(err.to_string().contains("can't recompile"), "{err}");

        // With the capture analysis now produces for x1, it lowers.
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let loads = &mut input_info.bind.embedded_constant_loads;
        loads.loads_num = 1;
        loads.loads[0].pc = 0x08;
        loads.loads[0].dwords_num = 1;
        loads.loads[0].values[0] = 0xfeed_face;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("a captured x1 load materializes");
        assert!(source.contains("OpStore"), "{source}");
        let words = spirv_run(&source).expect("assemble x1 scalar-load compute shader");
        spirv_val_ok(&words, "sload_dword_x1_capture");
    }

    /// The raw EUD window must never look authoritative when it is not. An
    /// `s_load` off the EUD base with a runtime register soffset cannot
    /// contribute a dword index, so detection records
    /// `unresolved_dynamic_offset` (rather than silently skipping the load) and
    /// the recompiler refuses the raw read by name instead of clamping to 0
    /// past a window it knows may be short.
    #[test]
    fn eud_raw_window_records_and_refuses_an_unresolved_dynamic_offset() {
        let sload = |type_, format, src1: ShaderOperand, src2: ShaderOperand, pc, src_num| {
            ShaderInstruction {
                pc,
                type_,
                format,
                src_num,
                dst: ShaderOperand {
                    type_: ShaderOperandType::Sgpr,
                    register_id: 16,
                    size: 4,
                    ..Default::default()
                },
                src: [
                    ShaderOperand {
                        type_: ShaderOperandType::Sgpr,
                        register_id: 12,
                        size: 2,
                        ..Default::default()
                    },
                    src1,
                    src2,
                    ShaderOperand::default(),
                ],
                ..Default::default()
            }
        };
        let imm = |v: u32| ShaderOperand {
            type_: ShaderOperandType::IntegerInlineConstant,
            constant: crate::shader::types::ShaderConstant::from_u(v),
            ..Default::default()
        };
        let sgpr = |r: i32| ShaderOperand {
            type_: ShaderOperandType::Sgpr,
            register_id: r,
            size: 1,
            ..Default::default()
        };

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // A constant-offset load (sizes the window) and a register-soffset one
        // off the SAME EUD base (cannot).
        code.get_instructions_mut().push(sload(
            T::SLoadDwordx4,
            F::Sdst4SbaseSoffset,
            imm(0),
            ShaderOperand::default(),
            0x10,
            2,
        ));
        code.get_instructions_mut().push(sload(
            T::SLoadDwordx4,
            F::Sdst4SbaseSoffsetOffset,
            sgpr(5),
            imm(32),
            0x18,
            3,
        ));
        code.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 12;

        crate::shader::spirv::shader_detect_eud_raw_window(&code, &mut input_info.bind);
        let raw = input_info.bind.eud_raw;
        assert!(raw.used, "the constant-offset load still declares a window");
        assert_eq!(raw.required_dwords, 4, "sized only by what it could size");
        assert!(
            raw.unresolved_dynamic_offset,
            "the register-soffset load must be RECORDED, not silently dropped"
        );

        // With the doubt recorded, the raw read refuses by name.
        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("a lower-bound window must not be read");
        let text = err.to_string();
        assert!(
            text.contains("lower bound") || text.contains("unresolved register soffset"),
            "named refusal expected, got: {text}"
        );

        // Same shader without the dynamic load: the window is authoritative and
        // the raw read is emitted as before (no regression).
        let mut clean = ShaderCode::new();
        clean.set_type(ShaderType::Compute);
        for _ in 0..2 {
            clean.get_instructions_mut().push(sload(
                T::SLoadDwordx4,
                F::Sdst4SbaseSoffset,
                imm(0),
                ShaderOperand::default(),
                0x10,
                2,
            ));
        }
        clean.get_instructions_mut().push(ShaderInstruction {
            type_: T::SEndpgm,
            format: F::Empty,
            ..Default::default()
        });
        crate::shader::spirv::shader_detect_eud_raw_window(&clean, &mut input_info.bind);
        assert!(!input_info.bind.eud_raw.unresolved_dynamic_offset);
        let source = spirv_generate_source(&clean, None, None, Some(&input_info))
            .expect("an authoritative window still lowers");
        assert!(
            source.contains("OpArrayLength %uint %eud_raw 0"),
            "{source}"
        );
        let words = spirv_run(&source).expect("assemble raw-window compute shader");
        spirv_val_ok(&words, "eud_raw_window_authoritative");
    }

    /// The whole chain for the combined addressing mode, starting from real
    /// encoded SMEM bytes: **ISA words -> parse -> analysis capture -> recompile
    /// -> assemble -> spirv-val (Vulkan 1.3)**. On build 2741d21 this program
    /// died in step 2 with `not implemented smem feature: offset != 0 with
    /// register soffset` and never produced a module at all.
    #[test]
    fn full_chain_register_soffset_bytes_to_validated_spirv() {
        use std::borrow::Cow;

        struct Mem(u64, Vec<u32>);
        impl crate::shader::analysis::ShaderMemory for Mem {
            fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
                (addr >= self.0 && (addr - self.0) % 4 == 0)
                    .then(|| ((addr - self.0) / 4) as usize)
                    .filter(|start| *start < self.1.len())
                    .map(|start| Cow::Borrowed(&self.1[start..]))
            }
        }

        // s_load_dwordx4 s[16:19], s[12:13], s4 offset:16
        //   b0 = 0x3d << 26 | opcode 0x02 << 18 | sdst 16 << 6 | sbase 6
        //   b1 = soffset s4 << 25 | offset 16
        const SLOAD: [u32; 2] = [0xF408_0406, 0x0800_0010];
        let mut words = Vec::new();
        words.extend_from_slice(&SLOAD); // pc 0x00
        words.extend_from_slice(&SLOAD); // pc 0x08
        words.push(S_ENDPGM);

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // next_gen = true: the SMEM (0x3d) encoding is next-gen only.
        crate::shader::parse::shader_parse(0, &words, &mut code, true)
            .expect("combined-addressing SMEM must parse");
        assert_eq!(
            code.get_instructions()[0].format,
            F::Sdst4SbaseSoffsetOffset
        );

        // ptr(s12:s13) = 0x0080_0000, soffset s4 = 0x20, imm = 16.
        let payload = vec![0x0a0a_0a0au32, 0x0b0b_0b0b, 0x0c0c_0c0c, 0x0d0d_0d0d];
        let mem = Mem(0x0080_0030, payload.clone());
        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        user_sgpr.set(4, 0x20, crate::shader::hw_regs::UserSgprType::Unknown);
        user_sgpr.set(
            12,
            0x0080_0000,
            crate::shader::hw_regs::UserSgprType::Unknown,
        );
        user_sgpr.set(13, 0, crate::shader::hw_regs::UserSgprType::Unknown);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        crate::shader::analysis::shader_capture_runtime_scalar_loads(
            &code,
            &mem,
            &user_sgpr,
            &mut input_info.bind,
        );
        assert_eq!(
            input_info
                .bind
                .embedded_constant_loads
                .find(0)
                .expect("the parsed load resolves through base + soffset + imm")
                .values[..4],
            payload[..]
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("full chain recompiles");
        let module = spirv_run(&source).expect("assemble");
        spirv_val_ok(&module, "full_chain_register_soffset");
    }

    /// A guest-memory stub for the V#-based `s_buffer_load` tests: dwords are
    /// readable from `base` onwards.
    struct VsharpMem(u64, Vec<u32>);
    impl crate::shader::analysis::ShaderMemory for VsharpMem {
        fn dwords_at(&self, addr: u64) -> Option<std::borrow::Cow<'_, [u32]>> {
            (addr >= self.0 && (addr - self.0) % 4 == 0)
                .then(|| ((addr - self.0) / 4) as usize)
                .filter(|start| *start < self.1.len())
                .map(|start| std::borrow::Cow::Borrowed(&self.1[start..]))
        }
    }

    /// A buffer resource descriptor (V#): base48 in dwords 0..1, `stride` in
    /// dword1 bits 16..29, `num_records` in dword2, and a buffer-typed dword3
    /// (`(dw3 >> 28) & 0xF < 8`, `sharp_dword3_is_buffer`). Field layout from
    /// SharpEmu `Gen5ShaderScalarEvaluator.cs::TryDecodeBufferDescriptor`
    /// (L2163-2216).
    fn vsharp(base: u64, stride: u32, num_records: u32) -> [u32; 4] {
        [
            (base & 0xffff_ffff) as u32,
            (((base >> 32) as u32) & 0xffff) | ((stride & 0x3fff) << 16),
            num_records,
            // unified format 7 in bits 12..18, type 0 (buffer) in bits 30..31.
            7 << 12,
        ]
    }

    /// The whole chain for a **V#-based** scalar buffer load, from real encoded
    /// SMEM bytes: **ISA words -> parse -> `shader_capture_vsharp_buffer_loads`
    /// -> recompile -> assemble -> spirv-val (Vulkan 1.3)**.
    ///
    /// This is Grand Theft Auto V's measured first blocker on build 36a9b18:
    /// `can't recompile: SBufferLoadDwordx8 [Sdst8SvSoffset] s[24:31],
    /// s[20:23], 0`. The shader declares **no storage buffer at all**
    /// (`storage_buffers.buffers_num == 0`), so Kyty's descriptor path returned
    /// `false` and the whole shader died. The V# is plain live-in user data, so
    /// analysis can decode `base48` and snapshot the dwords instead.
    #[test]
    fn full_chain_vsharp_sbuffer_load_x8_bytes_to_validated_spirv() {
        // s_buffer_load_dwordx8 s[24:31], s[20:23], 0
        //   b0 = 0x3d << 26 | opcode 0x0b << 18 | sdst 24 << 6 | sbase 10
        //        (sbase is in SGPR *pairs*: 10 * 2 = s20)
        //   b1 = soffset NULL (125, i.e. 0x7d) << 25 | offset 0
        const SBUFFER: [u32; 2] = [0xF42C_060A, 0xFA00_0000];
        // Twice, so `s_endpgm` is not instruction 0 or 1 (Kyty
        // `Recompile_SEndpgm_Empty` refuses those).
        let mut words = Vec::new();
        words.extend_from_slice(&SBUFFER);
        words.extend_from_slice(&SBUFFER);
        words.push(S_ENDPGM);

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        crate::shader::parse::shader_parse(0, &words, &mut code, true)
            .expect("V#-based s_buffer_load must parse");
        let inst = code.get_instructions()[0];
        assert_eq!(inst.type_, T::SBufferLoadDwordx8);
        assert_eq!(inst.format, F::Sdst8SvSoffset);
        assert_eq!(
            (inst.src[0].register_id, inst.src[0].size),
            (20, 4),
            "the base is the four-SGPR V# quad s[20:23]"
        );

        const BASE: u64 = 0x0090_0000;
        let payload: Vec<u32> = (0..8).map(|i| 0xc0de_0000 + i).collect();
        let mem = VsharpMem(BASE, payload.clone());
        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        for (i, field) in vsharp(BASE, 0, 1024).into_iter().enumerate() {
            user_sgpr.set(
                20 + i as u32,
                field,
                crate::shader::hw_regs::UserSgprType::Vsharp,
            );
        }

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        // No storage buffer is bound — the whole point of the measured failure.
        assert_eq!(input_info.bind.storage_buffers.buffers_num, 0);

        crate::shader::analysis::shader_capture_vsharp_buffer_loads(
            &code,
            &mem,
            &user_sgpr,
            0,
            &mut input_info.bind,
        );
        assert_eq!(
            input_info
                .bind
                .embedded_constant_loads
                .find(0)
                .expect("the V# base48 resolves and the eight dwords are captured")
                .values[..8],
            payload[..]
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("full chain recompiles with no storage buffer bound");
        let module = spirv_run(&source).expect("assemble");
        spirv_val_ok(&module, "full_chain_vsharp_sbuffer_load_x8");
    }

    /// The V# the shader **loads for itself** out of an SRT descriptor table,
    /// which is the only shape Grand Theft Auto V's vertex shaders use and which
    /// the live-in-only guard refused outright:
    ///
    /// ```text
    /// s_load_dwordx4        s[20:23], s[12:13], 64   ; fetch the V#
    /// s_buffer_load_dwordx8 s[24:31], s[20:23], 0    ; then use it
    /// ```
    ///
    /// Measured 2026-07-28
    /// (`artifacts/compat/raw/baseline-1785273714952/PPSA04264-*.stdout.log`):
    /// `Recompile_SBufferLoadDwordx8_Sdst8SvSoffset: not supported: no storage
    /// buffer bound for the V# and no resolved capture: ... V#=s[20:23],
    /// soffset=none, imm=0x0, pc=0xd4` — 998 of that run's 999 shader errors,
    /// across three vertex shaders, every one of them this shape. The V# quad is
    /// not live-in user data (the `s_load_dwordx4` writes it), so
    /// `shader_capture_vsharp_buffer_loads` skipped the load and the whole
    /// shader died.
    ///
    /// The descriptor is knowable without guessing: the producing `s_load` was
    /// itself already captured from guest memory by the pointer-load pass that
    /// runs first, so its four dwords ARE the V#. This drives the real
    /// production entry point (`shader_capture_runtime_scalar_loads_shifted`,
    /// which calls the V# pass at its tail) so the ordering the fix depends on
    /// is under test too.
    #[test]
    fn full_chain_vsharp_loaded_from_an_srt_table_to_validated_spirv() {
        // s_load_dwordx4 s[20:23], s[12:13], 0x40
        //   b0 = 0x3d << 26 | opcode 0x02 << 18 | sdst 20 << 6 | sbase 6
        //        (sbase is in SGPR *pairs*: 6 * 2 = s12)
        //   b1 = soffset NULL (0x7d) << 25 | offset 0x40
        const SLOAD: [u32; 2] = [0xF408_0506, 0xFA00_0040];
        // s_buffer_load_dwordx8 s[24:31], s[20:23], 0 — as in the test above.
        const SBUFFER: [u32; 2] = [0xF42C_060A, 0xFA00_0000];
        let mut words = Vec::new();
        words.extend_from_slice(&SLOAD);
        words.extend_from_slice(&SBUFFER);
        words.push(S_ENDPGM);

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        crate::shader::parse::shader_parse(0, &words, &mut code, true)
            .expect("the SRT load and the V# buffer load must parse");
        let instructions = code.get_instructions();
        assert_eq!(instructions[0].type_, T::SLoadDwordx4);
        assert_eq!(
            (instructions[0].dst.register_id, instructions[0].dst.size),
            (20, 4),
            "the SRT load defines the whole V# quad s[20:23]"
        );
        assert_eq!(instructions[1].type_, T::SBufferLoadDwordx8);
        assert_eq!(
            (
                instructions[1].src[0].register_id,
                instructions[1].src[0].size
            ),
            (20, 4),
            "and the buffer load reads its V# from that quad"
        );
        let (sload_pc, sbuffer_pc) = (instructions[0].pc, instructions[1].pc);

        // One guest region holding both indirections: the SRT table at `TABLE`
        // with the V# at byte offset 64, and the payload the V# points at.
        const TABLE: u64 = 0x0090_0000;
        const PAYLOAD_AT: u64 = TABLE + 128;
        let payload: Vec<u32> = (0..8).map(|i| 0xc0de_0000 + i).collect();
        let mut region = vec![0u32; 40];
        region[16..20].copy_from_slice(&vsharp(PAYLOAD_AT, 0, 1024));
        region[32..40].copy_from_slice(&payload);
        let mem = VsharpMem(TABLE, region);

        // The only live-in user data is the SRT POINTER — never the V# itself.
        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        for (i, half) in [TABLE as u32, (TABLE >> 32) as u32].into_iter().enumerate() {
            user_sgpr.set(
                12 + i as u32,
                half,
                crate::shader::hw_regs::UserSgprType::Region,
            );
        }

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        assert_eq!(
            input_info.bind.storage_buffers.buffers_num, 0,
            "no storage buffer is bound — the measured failure"
        );

        crate::shader::analysis::shader_capture_runtime_scalar_loads_shifted(
            &code,
            &mem,
            &user_sgpr,
            0,
            &mut input_info.bind,
        );

        // The SRT load is captured (it is what makes the V# knowable) ...
        assert_eq!(
            input_info
                .bind
                .embedded_constant_loads
                .find(sload_pc)
                .expect("the SRT descriptor load resolves through the live-in pointer")
                .values[..4],
            vsharp(PAYLOAD_AT, 0, 1024)
        );
        // ... and the buffer load that consumes it now resolves too.
        assert_eq!(
            input_info
                .bind
                .embedded_constant_loads
                .find(sbuffer_pc)
                .expect("the V# from the SRT table must resolve the buffer load")
                .values[..8],
            payload[..]
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("full chain recompiles with the V# taken from the SRT table");
        let module = spirv_run(&source).expect("assemble");
        spirv_val_ok(&module, "full_chain_vsharp_loaded_from_an_srt_table");
    }

    /// The refusal must SURVIVE where the producer is not proved. A V# quad
    /// assembled by moves has no captured producer, so nothing names its fields
    /// and the named refusal is the honest answer — never a guessed address.
    #[test]
    fn vsharp_written_by_moves_is_still_refused() {
        // s_mov_b32 s20, 0x1234 — SOP1 (`0xBE` prefix), opcode 0x03 in bits
        // 8..16, sdst 20 in bits 16..23, ssrc0 = literal (255), then the literal.
        const S_MOV_S20: [u32; 2] = [0xBE94_03FF, 0x0000_1234];
        const SBUFFER: [u32; 2] = [0xF42C_060A, 0xFA00_0000]; // x8 s[24:31], s[20:23], 0
        let mut words = Vec::new();
        words.extend_from_slice(&S_MOV_S20);
        words.extend_from_slice(&SBUFFER);
        words.push(S_ENDPGM);

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        crate::shader::parse::shader_parse(0, &words, &mut code, true).expect("parse");
        let instructions = code.get_instructions();
        assert_eq!(
            instructions[0].dst.register_id, 20,
            "the move writes into the V# quad"
        );
        let sbuffer_pc = instructions[1].pc;

        const BASE: u64 = 0x0090_0000;
        let mem = VsharpMem(BASE, (0..8).map(|i| 0xc0de_0000 + i).collect());
        // A live-in quad IS present in user data — and must be ignored, because
        // the shader overwrote part of it.
        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        for (i, field) in vsharp(BASE, 0, 1024).into_iter().enumerate() {
            user_sgpr.set(
                20 + i as u32,
                field,
                crate::shader::hw_regs::UserSgprType::Vsharp,
            );
        }

        let mut bind = ShaderBindResources::default();
        crate::shader::analysis::shader_capture_vsharp_buffer_loads(
            &code, &mem, &user_sgpr, 0, &mut bind,
        );
        assert!(
            bind.embedded_constant_loads.find(sbuffer_pc).is_none(),
            "an unproved V# producer must leave the load to the named refusal"
        );
    }

    /// The combined V# addressing mode end to end: **register soffset AND a
    /// non-zero immediate offset**, which build 36a9b18 refused in the PARSER
    /// (`not implemented smem feature: offset != 0 with register soffset on an
    /// s_buffer_load (V# base)`) — ASTRO.BOT's measured first blocker, so no
    /// module was produced at all.
    ///
    /// The two byte offsets simply sum on top of `V#.base48`; the read here
    /// lands at `base + 0x20 (soffset s4) + 0x10 (imm)`.
    #[test]
    fn full_chain_vsharp_sbuffer_load_combined_offset_to_validated_spirv() {
        // s_buffer_load_dwordx4 s[24:27], s[20:23], s4 offset:16
        //   b0 = 0x3d << 26 | opcode 0x0a << 18 | sdst 24 << 6 | sbase 10
        //   b1 = soffset s4 << 25 | offset 16
        const SBUFFER: [u32; 2] = [0xF428_060A, 0x0800_0010];
        let mut words = Vec::new();
        words.extend_from_slice(&SBUFFER);
        words.extend_from_slice(&SBUFFER);
        words.push(S_ENDPGM);

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        crate::shader::parse::shader_parse(0, &words, &mut code, true)
            .expect("combined V# addressing must parse");
        assert_eq!(code.get_instructions()[0].format, F::Sdst4SvSoffsetOffset);

        const BASE: u64 = 0x00a0_0000;
        // Readable window starts at base + 0x30 = the exact resolved address.
        let payload = vec![0x1111_1111u32, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        let mem = VsharpMem(BASE + 0x30, payload.clone());
        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        user_sgpr.set(4, 0x20, crate::shader::hw_regs::UserSgprType::Unknown);
        for (i, field) in vsharp(BASE, 16, 64).into_iter().enumerate() {
            user_sgpr.set(
                20 + i as u32,
                field,
                crate::shader::hw_regs::UserSgprType::Vsharp,
            );
        }

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        crate::shader::analysis::shader_capture_vsharp_buffer_loads(
            &code,
            &mem,
            &user_sgpr,
            0,
            &mut input_info.bind,
        );
        assert_eq!(
            input_info
                .bind
                .embedded_constant_loads
                .find(0)
                .expect("base48 + soffset + imm resolves")
                .values[..4],
            payload[..],
            "the soffset register and the immediate both contribute bytes"
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("full chain recompiles");
        let module = spirv_run(&source).expect("assemble");
        spirv_val_ok(&module, "full_chain_vsharp_sbuffer_combined");
    }

    /// The V# `num_records`/`stride` fields bound the read; they never enter the
    /// address (SharpEmu `TryDecodeBufferDescriptor`:
    /// `sizeBytes = stride == 0 ? word2 : stride * word2`). A load past that
    /// bound is NOT captured — the recompiler keeps its named refusal rather
    /// than snapshotting whatever follows the buffer in guest memory.
    #[test]
    fn vsharp_sbuffer_load_past_num_records_is_not_captured() {
        const SBUFFER: [u32; 2] = [0xF42C_060A, 0xFA00_0000]; // x8 s[24:31], s[20:23], 0
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        crate::shader::parse::shader_parse(0, &[SBUFFER[0], SBUFFER[1], S_ENDPGM], &mut code, true)
            .expect("parse");

        const BASE: u64 = 0x00b0_0000;
        let mem = VsharpMem(BASE, vec![0xdead_beef; 64]);
        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        // stride 0, num_records 16 => size 16 bytes; the x8 load wants 32.
        for (i, field) in vsharp(BASE, 0, 16).into_iter().enumerate() {
            user_sgpr.set(
                20 + i as u32,
                field,
                crate::shader::hw_regs::UserSgprType::Vsharp,
            );
        }

        let mut bind = crate::shader::resources::ShaderBindResources::default();
        crate::shader::analysis::shader_capture_vsharp_buffer_loads(
            &code, &mem, &user_sgpr, 0, &mut bind,
        );
        assert!(
            bind.embedded_constant_loads.find(0).is_none(),
            "a read past the descriptor's own bound must not be captured"
        );

        // A V# with a BOUND storage buffer keeps the live descriptor path: this
        // capture must never shadow it with a translate-time snapshot, or a
        // per-draw constant buffer would freeze at frame 1.
        let mut bound = crate::shader::resources::ShaderBindResources::default();
        bound.storage_buffers.buffers_num = 1;
        bound.storage_buffers.start_register[0] = 20;
        let ok_mem = VsharpMem(BASE, vec![0x1234_5678; 64]);
        let mut ok_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        for (i, field) in vsharp(BASE, 0, 4096).into_iter().enumerate() {
            ok_sgpr.set(
                20 + i as u32,
                field,
                crate::shader::hw_regs::UserSgprType::Vsharp,
            );
        }
        crate::shader::analysis::shader_capture_vsharp_buffer_loads(
            &code, &ok_mem, &ok_sgpr, 0, &mut bound,
        );
        assert!(
            bound.embedded_constant_loads.find(0).is_none(),
            "a bound descriptor must not be shadowed by a baked snapshot"
        );

        // And the refusal names the instruction, format, width and registers.
        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("an unresolved V# load still fails the shader");
        let text = err.to_string();
        for want in [
            "SBufferLoadDwordx8",
            "Sdst8SvSoffset",
            "x8 dwords",
            "s[20:23]",
        ] {
            assert!(text.contains(want), "refusal must name {want}, got: {text}");
        }
    }

    /// The combined V# addressing mode through the **bound storage buffer**
    /// path (the other half of `sbuffer_load_dwords`): the descriptor is live,
    /// so the module must compute `soffset_register + immediate` at runtime
    /// rather than bake anything. The immediate here (0x50) is outside the
    /// seeded 0..=32 uint-constant range, which is exactly the case that needs
    /// `find_constants` to register combined-form immediates as UINT — the
    /// parser files an `IntegerInlineConstant` as Int only, so without that the
    /// operand resolves to `unknown_uint_constant` and assembly fails.
    #[test]
    fn vsharp_sbuffer_load_combined_offset_adds_at_runtime_through_bound_buffer() {
        // s_buffer_load_dwordx2 s[24:25], s[8:11], s4 offset:0x50
        //   b0 = 0x3d << 26 | opcode 0x09 << 18 | sdst 24 << 6 | sbase 4
        //   b1 = soffset s4 << 25 | offset 0x50
        const SBUFFER: [u32; 2] = [0xF424_0604, 0x0800_0050];
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        crate::shader::parse::shader_parse(
            0,
            &[SBUFFER[0], SBUFFER[1], SBUFFER[0], SBUFFER[1], S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse");
        assert_eq!(code.get_instructions()[0].format, F::Sdst2SvSoffsetOffset);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 48;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 8;

        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("the bound-descriptor path lowers the combined form");
        assert!(
            source.contains("OpIAdd %uint %t1_0 %uint_"),
            "the soffset register and the immediate must be summed at runtime, \
             not folded: {source}"
        );
        assert!(
            !source.contains("unknown_uint_constant"),
            "the combined immediate must be a registered uint constant: {source}"
        );
        assert!(
            source.contains("%sbuffer_load_dword_2"),
            "the width-2 callee is used: {source}"
        );
        let module = spirv_run(&source).expect("assemble");
        spirv_val_ok(&module, "vsharp_sbuffer_combined_bound_buffer");
    }

    /// Every `s_buffer_load` width lowers through the bound-descriptor path
    /// with a plain constant offset, and each reaches its own
    /// `sbuffer_load_dword_N` callee. Before this batch, x1/x2/x8/x16 each had
    /// their own copy of the body (only x4 accepted a runtime offset); they now
    /// share `sbuffer_load_dwords`, so one test covers the set.
    #[test]
    fn every_sbuffer_load_width_lowers_through_a_bound_descriptor() {
        for (opcode, callee) in [
            (0x08u32, "%sbuffer_load_dword "),
            (0x09, "%sbuffer_load_dword_2 "),
            (0x0a, "%sbuffer_load_dword_4 "),
            (0x0b, "%sbuffer_load_dword_8 "),
            (0x0c, "%sbuffer_load_dword_16 "),
        ] {
            // sdst = s32, sbase field 4 => s[8:11], soffset NULL (125), imm 16.
            let b0 = 0xF400_0000 | (opcode << 18) | (32 << 6) | 4;
            let b1 = 0xFA00_0000 | 16;
            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Pixel);
            crate::shader::parse::shader_parse(0, &[b0, b1, b0, b1, S_ENDPGM], &mut code, true)
                .unwrap_or_else(|e| panic!("opcode {opcode:#04x}: {e}"));

            let mut input_info = ShaderPixelInputInfo::default();
            input_info.target_output_mode[0] = 4;
            input_info.bind.push_constant_size = 48;
            input_info.bind.storage_buffers.buffers_num = 1;
            input_info.bind.storage_buffers.start_register[0] = 8;

            let source = spirv_generate_source(&code, None, Some(&input_info), None)
                .unwrap_or_else(|e| panic!("opcode {opcode:#04x} must lower: {e}"));
            assert!(
                source.contains(callee),
                "opcode {opcode:#04x} must call {callee}: {source}"
            );
            let module =
                spirv_run(&source).unwrap_or_else(|e| panic!("opcode {opcode:#04x} assemble: {e}"));
            spirv_val_ok(&module, "every_sbuffer_load_width");
        }
    }

    /// Avatar: Frontiers of Pandora's measured first blocker — a four-channel
    /// MUBUF typed fetch with index-enable addressing had **no dispatch row at
    /// all** (`can't recompile (no table entry for
    /// BufferLoadFormatXyzw/Vdata4VaddrSvSoffsIdxen)`, 2398 shader errors per
    /// 180 s window). Real MUBUF bytes -> parse -> recompile -> assemble ->
    /// spirv-val.
    #[test]
    fn full_chain_buffer_load_format_xyzw_idxen_to_validated_spirv() {
        // buffer_load_format_xyzw v[4:7], v0, s[4:7], 0 idxen
        // (the exact encoding `mubuf_buffer_load_format_xyzw` in parse.rs
        // asserts decodes to BufferLoadFormatXyzw/Vdata4VaddrSvSoffsIdxen)
        let words = [
            0xE00C_2000u32,
            0x8001_0400,
            0xE00C_2000,
            0x8001_0400,
            S_ENDPGM,
        ];
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        crate::shader::parse::shader_parse(0, &words, &mut code, true).expect("parse");
        let inst = code.get_instructions()[0];
        assert_eq!(inst.type_, T::BufferLoadFormatXyzw);
        assert_eq!(inst.format, F::Vdata4VaddrSvSoffsIdxen);
        // A dispatch row now exists (build 36a9b18 had none).
        assert!(
            recomp_func(inst.type_, inst.format).is_some(),
            "BufferLoadFormatXyzw/Vdata4VaddrSvSoffsIdxen needs a table row"
        );

        // The typed helpers index %buf, so one storage buffer must be bound —
        // and MUBUF reads its element format from that descriptor, so the
        // binding has to be the instruction's own V# (s[4:7]) carrying unified
        // 77 = (dfmt 14, nfmt 7) = 32_32_32_32_FLOAT.
        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 48;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 4;
        input_info.bind.storage_buffers.buffers[0].fields = [0, 16 << 16, 256, 77 << 12];

        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("full chain recompiles");
        assert!(
            source.contains("%tbuffer_load_format_xyzw"),
            "the four-channel typed helper must be called: {source}"
        );
        assert!(
            source.contains("OpStore %temp_int_5 %int_119"),
            "unified 77 must reach the helper as the packed 119 it tests: {source}"
        );
        let module = spirv_run(&source).expect("assemble");
        spirv_val_ok(&module, "full_chain_buffer_load_format_xyzw_idxen");
    }

    /// A V# that sits at a **nonzero offset inside** the descriptor table a
    /// single wide `s_load` fetched.
    ///
    /// One `s_load_dwordx8 s[16:23]` delivers TWO V#s; one
    /// `s_load_dwordx16 s[12:27]` delivers four. The provenance proof used to
    /// require `producer.dst.register_id == base_reg`, so only the FIRST quad
    /// of such a table was ever provable and every later one kept the named
    /// refusal — even though its dwords sit in the same already-captured
    /// snapshot at a known index. Measured shapes: Avatar: Frontiers of
    /// Pandora's vertex stage (`s_load_dwordx16 s[12:27], s[8:9], 0` feeding
    /// five `buffer_load_format_xyzw` through s[12:15], s[16:19], s[20:23],
    /// s[24:27]) and GTA V's `s_load_dwordx8 s[8:15], s[0:1], 0`.
    ///
    /// Nothing is guessed: the same proved bytes, indexed at the right dword.
    #[test]
    fn full_chain_vsharp_at_an_offset_inside_a_loaded_table() {
        // s_load_dwordx8 s[16:23], s[12:13], 0  — a table of two V#s.
        //   b0 = 0x3d << 26 | opcode 0x03 << 18 | sdst 16 << 6 | sbase 6
        //   b1 = soffset NULL (0x7d) << 25 | offset 0
        const SLOAD: [u32; 2] = [0xF40C_0406, 0xFA00_0000];
        // s_buffer_load_dwordx4 s[28:31], s[20:23], 0 — the SECOND quad.
        //   b0 = 0x3d << 26 | opcode 0x0a << 18 | sdst 28 << 6 | sbase 10
        const SBUFFER: [u32; 2] = [0xF428_070A, 0xFA00_0000];
        let mut words = Vec::new();
        words.extend_from_slice(&SLOAD);
        words.extend_from_slice(&SBUFFER);
        words.push(S_ENDPGM);

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        crate::shader::parse::shader_parse(0, &words, &mut code, true).expect("parse");
        assert_eq!(code.get_instructions()[0].type_, T::SLoadDwordx8);
        assert_eq!(code.get_instructions()[0].dst.register_id, 16);
        assert_eq!(code.get_instructions()[0].dst.size, 8);
        let use_ = code.get_instructions()[1];
        assert_eq!(use_.type_, T::SBufferLoadDwordx4);
        assert_eq!(
            (use_.src[0].register_id, use_.src[0].size),
            (20, 4),
            "the V# is the SECOND quad of the loaded table"
        );

        const TABLE: u64 = 0x0070_0000;
        const DATA: u64 = 0x0090_0000;
        // Two V#s back to back; only the second one is used.
        let mut table = Vec::new();
        table.extend_from_slice(&vsharp(0x0060_0000, 0, 64));
        table.extend_from_slice(&vsharp(DATA, 0, 1024));
        let payload: Vec<u32> = (0..4).map(|i| 0xfeed_0000 + i).collect();

        struct TwoRegion(u64, Vec<u32>, u64, Vec<u32>);
        impl crate::shader::analysis::ShaderMemory for TwoRegion {
            fn dwords_at(&self, addr: u64) -> Option<std::borrow::Cow<'_, [u32]>> {
                for (base, data) in [(self.0, &self.1), (self.2, &self.3)] {
                    if addr >= base && (addr - base) % 4 == 0 {
                        let start = ((addr - base) / 4) as usize;
                        if start < data.len() {
                            return Some(std::borrow::Cow::Borrowed(&data[start..]));
                        }
                    }
                }
                None
            }
        }
        let mem = TwoRegion(TABLE, table, DATA, payload.clone());

        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        user_sgpr.set(
            12,
            TABLE as u32,
            crate::shader::hw_regs::UserSgprType::Unknown,
        );
        user_sgpr.set(13, 0, crate::shader::hw_regs::UserSgprType::Unknown);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        assert_eq!(input_info.bind.storage_buffers.buffers_num, 0);

        // The production entry point: the pointer-load capture runs first and
        // the V# pass reads its snapshot at dword offset 4.
        crate::shader::analysis::shader_capture_runtime_scalar_loads(
            &code,
            &mem,
            &user_sgpr,
            &mut input_info.bind,
        );
        assert_eq!(
            input_info
                .bind
                .embedded_constant_loads
                .find(0x08)
                .expect("the offset quad's V# resolves and its dwords are captured")
                .values[..4],
            payload[..]
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("full chain recompiles from the offset quad");
        let module = spirv_run(&source).expect("assemble");
        spirv_val_ok(&module, "vsharp_at_an_offset_inside_a_loaded_table");
    }

    /// Avatar: Frontiers of Pandora's real, dumped vertex-fetch shape, and the
    /// three separate defects it exposed.
    ///
    /// Measured VS `0x4020024d00` (dumped 2026-07-29 from the retail title):
    ///
    /// ```text
    /// s_load_dwordx16         s[12:27], s[8:9], 0        ; a table of FOUR V#s
    /// buffer_load_format_xyzw v[0:3],  v5, s[12:15], 0 idxen
    /// buffer_load_format_xyzw v[10:13], v5, s[24:27], 0 idxen
    /// ...
    /// ```
    ///
    /// 1. Every MUBUF body is gated on `storage_buffers.buffers_num > 0`, and
    ///    the usage-table walk binds nothing for an SRT-delivered V#, so
    ///    `recompile_buffer_load_format_xyzw_vdata4` returned `Ok(false)` —
    ///    printed as the bare `can't recompile: BufferLoadFormatXyzw
    ///    [Vdata4VaddrSvSoffsIdxen] v[0:3], v5, s[12:15], 0, idxen`, 853
    ///    occurrences in a 180 s run. The descriptor-format branch the previous
    ///    note suspected is INSIDE `buffers_num > 0` and was never reached.
    /// 2. Three of the four V#s sit at nonzero offsets inside one captured
    ///    `s_load_dwordx16`.
    /// 3. Binding the descriptor makes `WriteLocalVariables` seed the quad's
    ///    SGPRs with the rewritten push-constant index — which the captured
    ///    `s_load` would then overwrite with the raw guest base address,
    ///    indexing the descriptor array out of bounds (the ASTRO.BOT
    ///    `VK_ERROR_DEVICE_LOST` class). The seed must survive.
    #[test]
    fn avatar_srt_vsharp_binds_and_its_descriptor_seed_survives() {
        // s_load_dwordx16 s[12:27], s[8:9], 0
        //   b0 = 0x3d << 26 | opcode 0x04 << 18 | sdst 12 << 6 | sbase 4
        const SLOAD: [u32; 2] = [0xF410_0304, 0xFA00_0000];
        // buffer_load_format_xyzw v[0:3], v5, s[16:19], 0 idxen — the SECOND
        // quad of the table (srsrc field = 16 / 4 = 4).
        const MUBUF: [u32; 2] = [0xE00C_2000, 0x8004_0005];
        let words = [
            SLOAD[0], SLOAD[1], MUBUF[0], MUBUF[1], MUBUF[0], MUBUF[1], S_ENDPGM,
        ];

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        crate::shader::parse::shader_parse(0, &words, &mut code, true).expect("parse");
        let fetch = code.get_instructions()[1];
        assert_eq!(fetch.type_, T::BufferLoadFormatXyzw);
        assert_eq!(fetch.format, F::Vdata4VaddrSvSoffsIdxen);
        assert_eq!(
            (fetch.src[1].register_id, fetch.src[1].size),
            (16, 4),
            "the V# is the second quad of the s_load_dwordx16 table"
        );

        const TABLE: u64 = 0x0050_0000;
        const VERTS: u64 = 0x0080_0000;
        let mut table = Vec::new();
        table.extend_from_slice(&vsharp(0x0040_0000, 16, 8)); // s[12:15]
        let mut used = vsharp(VERTS, 16, 256); // s[16:19]
        // Unified format **77** = (dfmt 14, nfmt 7) = 32_32_32_32_FLOAT, in
        // bits 12..18 of dword 3. Kyty's helper compares against the LEGACY
        // packing `dfmt * 8 + nfmt` = 119, which is what the row must convert
        // to — 119 is not a valid unified encoding at all.
        used[3] = 77 << 12;
        table.extend_from_slice(&used);
        table.extend_from_slice(&vsharp(0x0041_0000, 16, 8)); // s[20:23]
        table.extend_from_slice(&vsharp(0x0042_0000, 16, 8)); // s[24:27]

        struct Mem(u64, Vec<u32>);
        impl crate::shader::analysis::ShaderMemory for Mem {
            fn dwords_at(&self, addr: u64) -> Option<std::borrow::Cow<'_, [u32]>> {
                (addr >= self.0 && (addr - self.0) % 4 == 0)
                    .then(|| ((addr - self.0) / 4) as usize)
                    .filter(|start| *start < self.1.len())
                    .map(|start| std::borrow::Cow::Borrowed(&self.1[start..]))
            }
        }
        let mem = Mem(TABLE, table);

        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        user_sgpr.set(
            8,
            TABLE as u32,
            crate::shader::hw_regs::UserSgprType::Unknown,
        );
        user_sgpr.set(9, 0, crate::shader::hw_regs::UserSgprType::Unknown);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        crate::shader::analysis::shader_capture_runtime_scalar_loads(
            &code,
            &mem,
            &user_sgpr,
            &mut input_info.bind,
        );

        // (1) + (2): the offset quad is proved and BOUND as a real descriptor.
        assert_eq!(
            input_info.bind.storage_buffers.buffers_num, 1,
            "the proved MUBUF V# must be bound as a storage buffer"
        );
        assert_eq!(input_info.bind.storage_buffers.start_register[0], 16);
        assert!(!input_info.bind.storage_buffers.extended[0]);
        assert_eq!(input_info.bind.storage_buffers.buffers[0].base48(), VERTS);
        assert_eq!(input_info.bind.storage_buffers.buffers[0].format(), 77);

        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("Avatar's measured fetch must recompile once its V# is bound");
        assert!(
            source.contains("%tbuffer_load_format_xyzw"),
            "the four-channel typed helper must be called: {source}"
        );
        // The unified 77 in the descriptor must reach the helper as the LEGACY
        // packed 119 it actually tests. Passing the unified number straight
        // through — what the row used to do — could never match, so the fetch
        // silently did nothing for every descriptor.
        assert!(
            source.contains("OpStore %temp_int_5 %int_119"),
            "unified 77 must be converted to the packed 119 the helper tests: {source}"
        );
        assert!(
            !source.contains("OpBitwiseAnd %uint %t208_1"),
            "the broken runtime unified-format extraction must be gone: {source}"
        );

        // (3) The push-constant seed of the bound quad must be the ONLY writer
        // of s16..s19 before the fetch: the captured `s_load_dwordx16` must not
        // materialize the raw guest dwords over the descriptor-array index.
        for reg in 16..20 {
            let stores = source
                .lines()
                .filter(|line| line.trim_start().starts_with(&format!("OpStore %s{reg} ")))
                .count();
            assert_eq!(
                stores, 1,
                "s{reg} must keep its descriptor seed (found {stores} stores):\n{source}"
            );
        }
        // The dwords the binding does NOT own are still materialized.
        assert!(
            source.contains("OpStore %s12 "),
            "unowned table dwords keep the snapshot: {source}"
        );

        let module = spirv_run(&source).expect("assemble");
        spirv_val_ok(&module, "avatar_srt_vsharp_binds");
    }

    /// The unified → legacy-packed conversion, checked against the five packed
    /// constants Kyty's own SPIR-V helpers document in their comments. Those
    /// comments are the ground truth for the packing (`dfmt * 8 + nfmt`) and
    /// SharpEmu's `Gfx10UnifiedFormat.cs` (RDNA2 ISA table 47) for the unified
    /// numbering; agreement across the two is what makes the conversion safe.
    #[test]
    fn unified_format_converts_to_the_packed_number_kyty_helpers_test() {
        use crate::shader::spirv::{gfx10_unified_to_dfmt_nfmt, gfx10_unified_to_packed_dfmt_nfmt};

        // (unified, dfmt, nfmt, the packed constant a Kyty helper compares to)
        for (unified, dfmt, nfmt, packed) in [
            // `tbuffer_load_format_x`: "dfmt = 4, nfmt = 4 or 7" -> 36 / 39.
            (20u32, 4u32, 4u32, 36u32),
            (22, 4, 7, 39),
            // `tbuffer_store_format_xy`: "dfmt = 11, nfmt = 4 or 7" -> 92 / 95.
            (62, 11, 4, 92),
            (64, 11, 7, 95),
            // `tbuffer_load_format_xyzw`: "dfmt = 14, nfmt = 7" -> 119.
            (77, 14, 7, 119),
        ] {
            assert_eq!(
                gfx10_unified_to_dfmt_nfmt(unified),
                Some((dfmt, nfmt)),
                "unified {unified}"
            );
            assert_eq!(
                gfx10_unified_to_packed_dfmt_nfmt(unified),
                Some(packed),
                "unified {unified} must pack to {packed}"
            );
            assert_eq!(dfmt * 8 + nfmt, packed, "the packing rule itself");
        }

        // 119 is a PACKED number, not a unified one — RDNA2 table 47 has no
        // entry there. Feeding the descriptor's unified field straight into the
        // helper (what the MUBUF row used to do) could therefore never match.
        assert_eq!(gfx10_unified_to_dfmt_nfmt(119), None);
        // Reserved holes stay refused rather than being derived into existence.
        for reserved in [30u32, 35, 46, 47, 127] {
            assert_eq!(
                gfx10_unified_to_dfmt_nfmt(reserved),
                None,
                "unified {reserved} is reserved"
            );
        }
        // Image-only encodings have no legacy dfmt and must not pack.
        assert_eq!(gfx10_unified_to_packed_dfmt_nfmt(131), None);
    }

    /// A shader must never end up PARTIALLY bound.
    ///
    /// `mubuf_flexible` has two lowerings: with no storage buffer bound
    /// anywhere it treats every MUBUF as a null V# (loads return 0, stores
    /// drop) and the shader still compiles; with at least one bound it
    /// switches every site to the descriptor path, which indexes `%buf` with
    /// the VALUE of that site's V# dword-0 register. A site whose V# was not
    /// proved has no seeded register, so it would index the descriptor array
    /// with raw guest data — the ASTRO.BOT `VK_ERROR_DEVICE_LOST` class.
    /// Binding some-but-not-all would therefore turn a compiling shader into a
    /// device-loss risk, so the pass binds all or none.
    #[test]
    fn a_shader_with_one_unprovable_mubuf_vsharp_binds_none_of_them() {
        // s_load_dwordx8 s[16:23], s[12:13], 0 — proves s[16:19] and s[20:23].
        const SLOAD: [u32; 2] = [0xF40C_0406, 0xFA00_0000];
        // buffer_load_format_xyzw v[0:3], v5, s[16:19], 0 idxen  (provable)
        const PROVABLE: [u32; 2] = [0xE00C_2000, 0x8004_0005];
        // buffer_load_format_xyzw v[0:3], v5, s[4:7], 0 idxen
        // s[4:7] is neither live-in user data nor written by any load here.
        const UNPROVABLE: [u32; 2] = [0xE00C_2000, 0x8001_0005];
        let words = [
            SLOAD[0],
            SLOAD[1],
            PROVABLE[0],
            PROVABLE[1],
            UNPROVABLE[0],
            UNPROVABLE[1],
            S_ENDPGM,
        ];

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        crate::shader::parse::shader_parse(0, &words, &mut code, true).expect("parse");

        const TABLE: u64 = 0x0050_0000;
        let mut table = Vec::new();
        table.extend_from_slice(&vsharp(0x0080_0000, 16, 256));
        table.extend_from_slice(&vsharp(0x0081_0000, 16, 256));

        struct Mem(u64, Vec<u32>);
        impl crate::shader::analysis::ShaderMemory for Mem {
            fn dwords_at(&self, addr: u64) -> Option<std::borrow::Cow<'_, [u32]>> {
                (addr >= self.0 && (addr - self.0) % 4 == 0)
                    .then(|| ((addr - self.0) / 4) as usize)
                    .filter(|start| *start < self.1.len())
                    .map(|start| std::borrow::Cow::Borrowed(&self.1[start..]))
            }
        }
        let mem = Mem(TABLE, table);

        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        user_sgpr.set(
            12,
            TABLE as u32,
            crate::shader::hw_regs::UserSgprType::Unknown,
        );
        user_sgpr.set(13, 0, crate::shader::hw_regs::UserSgprType::Unknown);

        let mut bind = crate::shader::resources::ShaderBindResources::default();
        crate::shader::analysis::shader_capture_runtime_scalar_loads(
            &code, &mem, &user_sgpr, &mut bind,
        );
        assert_eq!(
            bind.storage_buffers.buffers_num, 0,
            "one unprovable MUBUF V# must suppress ALL new bindings"
        );
    }

    /// The second Avatar-class defect, now that its V# binds: the MUBUF form
    /// takes its element format from the DESCRIPTOR, and Kyty's
    /// `tbuffer_load_format_xyzw` helper serves only unified 119 — for any
    /// other format it silently leaves the destination VGPRs untouched. Silent
    /// garbage is invisible in a log, so a known non-119 descriptor is refused
    /// by name and counted.
    #[test]
    fn non_119_buffer_format_is_refused_by_name_not_silently_dropped() {
        const MUBUF: [u32; 2] = [0xE00C_2000, 0x8001_0400];
        let words = [MUBUF[0], MUBUF[1], MUBUF[0], MUBUF[1], S_ENDPGM];
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        crate::shader::parse::shader_parse(0, &words, &mut code, true).expect("parse");

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 48;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 4;
        // Unified 13 = (dfmt 2, nfmt 7) — a real format, but not the
        // 32_32_32_32_FLOAT the four-channel helper serves.
        input_info.bind.storage_buffers.buffers[0].fields = [0, 16 << 16, 256, 13 << 12];

        let before = crate::shader::spirv::unsupported_buffer_format_skips();
        let err = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect_err("a format the typed helper cannot serve must not translate silently");
        let text = err.to_string();
        for want in [
            "unified format 13",
            "dfmt 2, nfmt 7",
            "unified 77 / packed 119",
            "s[4:7]",
        ] {
            assert!(text.contains(want), "refusal must name {want}, got: {text}");
        }
        assert!(
            crate::shader::spirv::unsupported_buffer_format_skips() > before,
            "the refusal must be counted"
        );
    }

    /// One MUBUF fixture: a single typed access through a V# in `s[4:7]`,
    /// against a descriptor whose unified FORMAT is `unified`.
    ///
    /// `word0` is the MUBUF first dword — `0xE000_0000 | (opcode << 18) |
    /// (idxen << 13) | (offen << 12)`, the field layout `shader_parse_mubuf`
    /// decodes. `word1 = 0x8001_0400` throughout: soffset `0x80` (inline zero),
    /// srsrc 1 (= `s[4:7]`), vdata `v4`, vaddr `v0`.
    fn mubuf_typed_source(
        word0: u32,
        unified: u32,
    ) -> Result<(String, ShaderCode), ShaderRecompileError> {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        // The trailing `s_nop` is what the other MUBUF fixtures use: the
        // recompiler refuses an `s_endpgm` that is the last instruction of an
        // otherwise one-instruction program.
        shader_parse(
            0,
            &[word0, 0x8001_0400, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .unwrap_or_else(|e| panic!("parse MUBUF {word0:#010x}: {e}"));

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 64;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 4;
        input_info.bind.storage_buffers.buffers[0].fields = [0, 16 << 16, 256, unified << 12];

        spirv_generate_source(&code, None, None, Some(&input_info)).map(|s| (s, code))
    }

    /// EVERY MUBUF row that passes a format to a typed helper must pass the
    /// **packed** `dfmt * 8 + nfmt` constant, not the descriptor's raw unified
    /// field — through to spirv-val-clean SPIR-V.
    ///
    /// The `BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen]` row was converted
    /// on `fix-aaa-shader-gaps`; the identical defect survived in the
    /// `BufferLoadFormatX` / `BufferStoreFormatX` / `BufferStoreFormatXy`
    /// bodies and in `mubuf_flexible`, which ten dispatch rows share. All of
    /// them emitted
    ///
    /// ```text
    /// OpShiftRightLogical %uint <dword3> %int_12
    /// OpBitwiseAnd %uint <..> %uint_127
    /// ```
    ///
    /// and handed the result to a guard that only ever accepts 36 / 39 / 92 /
    /// 95 / 119 — so the guard could never pass and the access was a silent
    /// no-op. Each row is checked separately because they share no code path
    /// at all above `mubuf_flexible`, and the flexible rows differ by
    /// addressing mode.
    #[test]
    fn every_mubuf_typed_row_passes_the_packed_format_not_the_unified_one() {
        // (word0, what it decodes to, unified FORMAT of the descriptor, the
        //  packed constant that must reach the helper, the helper called)
        const ROWS: &[(u32, T, F, u32, u32, &str)] = &[
            // --- the three standalone Kyty bodies (idxen, no offen) ---
            // opcode 0x00 buffer_load_format_x
            (
                0xE000_2000,
                T::BufferLoadFormatX,
                F::Vdata1VaddrSvSoffsIdxen,
                22,
                39,
                "%tbuffer_load_format_x",
            ),
            // opcode 0x04 buffer_store_format_x
            (
                0xE010_2000,
                T::BufferStoreFormatX,
                F::Vdata1VaddrSvSoffsIdxen,
                20,
                36,
                "%tbuffer_store_format_x",
            ),
            // opcode 0x05 buffer_store_format_xy
            (
                0xE014_2000,
                T::BufferStoreFormatXy,
                F::Vdata2VaddrSvSoffsIdxen,
                64,
                95,
                "%tbuffer_store_format_xy",
            ),
            // --- mubuf_flexible: buffer_load_format_x, all three modes ---
            (
                0xE000_0000,
                T::BufferLoadFormatX,
                F::Vdata1SvSoffs,
                20,
                36,
                "%tbuffer_load_format_x",
            ),
            (
                0xE000_1000,
                T::BufferLoadFormatX,
                F::Vdata1VaddrSvSoffsOffen,
                22,
                39,
                "%tbuffer_load_format_x",
            ),
            (
                0xE000_3000,
                T::BufferLoadFormatX,
                F::Vdata1Vaddr2SvSoffsOffenIdxen,
                22,
                39,
                "%tbuffer_load_format_x",
            ),
            // --- mubuf_flexible: buffer_store_format_x, all three modes ---
            (
                0xE010_0000,
                T::BufferStoreFormatX,
                F::Vdata1SvSoffs,
                20,
                36,
                "%tbuffer_store_format_x",
            ),
            (
                0xE010_1000,
                T::BufferStoreFormatX,
                F::Vdata1VaddrSvSoffsOffen,
                22,
                39,
                "%tbuffer_store_format_x",
            ),
            (
                0xE010_3000,
                T::BufferStoreFormatX,
                F::Vdata1Vaddr2SvSoffsOffenIdxen,
                20,
                36,
                "%tbuffer_store_format_x",
            ),
            // --- mubuf_flexible: buffer_store_format_xyzw, all four modes ---
            (
                0xE01C_2000,
                T::BufferStoreFormatXyzw,
                F::Vdata4VaddrSvSoffsIdxen,
                77,
                119,
                "%tbuffer_store_format_xyzw",
            ),
            (
                0xE01C_3000,
                T::BufferStoreFormatXyzw,
                F::Vdata4Vaddr2SvSoffsOffenIdxen,
                77,
                119,
                "%tbuffer_store_format_xyzw",
            ),
            (
                0xE01C_0000,
                T::BufferStoreFormatXyzw,
                F::Vdata4SvSoffs,
                77,
                119,
                "%tbuffer_store_format_xyzw",
            ),
            (
                0xE01C_1000,
                T::BufferStoreFormatXyzw,
                F::Vdata4VaddrSvSoffsOffen,
                77,
                119,
                "%tbuffer_store_format_xyzw",
            ),
            // --- the four-channel load row fixed on fix-aaa-shader-gaps,
            //     kept here so the whole set is one table ---
            (
                0xE00C_2000,
                T::BufferLoadFormatXyzw,
                F::Vdata4VaddrSvSoffsIdxen,
                77,
                119,
                "%tbuffer_load_format_xyzw",
            ),
        ];

        for &(word0, type_, format, unified, packed, helper) in ROWS {
            let label = format!("{type_:?} [{format:?}] ({word0:#010x})");
            let (source, code) = mubuf_typed_source(word0, unified)
                .unwrap_or_else(|e| panic!("{label} must recompile: {e}"));

            // The fixture really is the row under test.
            let inst = &code.get_instructions()[0];
            assert_eq!(inst.type_, type_, "{label}: decoded type");
            assert_eq!(inst.format, format, "{label}: decoded format");

            assert!(
                source.contains(&format!("OpFunctionCall %void {helper} ")),
                "{label} must call {helper}:\n{source}"
            );
            assert!(
                source.contains(&format!("OpStore %temp_int_5 %int_{packed}")),
                "{label}: unified {unified} must reach {helper} as PACKED {packed}:\n{source}"
            );
            // The runtime `(dword3 >> 12) & 0x7f` extraction is what could
            // never match. Both spellings of it must be gone: `%mbf_f0_*` is
            // `mubuf_flexible`'s, `%t206_*`/`%t208_*` the standalone bodies'.
            for gone in ["%mbf_f0_", "%t206_0", "%t208_0"] {
                assert!(
                    !source.contains(gone),
                    "{label}: runtime format extraction `{gone}` must be gone:\n{source}"
                );
            }

            let words = spirv_run(&source).unwrap_or_else(|e| panic!("{label} assembles: {e}"));
            spirv_val_ok(&words, &label);
        }
    }

    /// Dead Cells' only shader is a compute dispatch that stores through a `V#`
    /// whose format is unified **75** — `dfmt 14, nfmt 4`, `32_32_32_32_UINT`.
    /// While the x4 guard admitted packed 119 (nfmt 7, float) alone, that
    /// dispatch was refused every frame (`dispatch_skips=6`,
    /// `translate_failed: 1`) and the title published 406 frames with nothing
    /// drawn into any of them.
    ///
    /// At 32 bits per channel the RDNA2 ISA (doc 70648) defines no numeric
    /// conversion for either nfmt — both move four raw dwords — so the float
    /// body is bit-exact for uint data. This is the same reason the x1 (36/39)
    /// and x2 (92/95) helpers have always admitted their UINT twin; only x4 was
    /// left narrow.
    #[test]
    fn a_32_32_32_32_uint_descriptor_reaches_both_x4_helpers() {
        // (word0, the row it decodes to, the helper it must reach)
        const ROWS: &[(u32, T, F, &str)] = &[
            // Dead Cells' measured row, verbatim from the refusal it produced.
            (
                0xE01C_2000,
                T::BufferStoreFormatXyzw,
                F::Vdata4VaddrSvSoffsIdxen,
                "%tbuffer_store_format_xyzw",
            ),
            // The load twin, widened by the same argument.
            (
                0xE00C_2000,
                T::BufferLoadFormatXyzw,
                F::Vdata4VaddrSvSoffsIdxen,
                "%tbuffer_load_format_xyzw",
            ),
        ];

        for &(word0, type_, format, helper) in ROWS {
            let label = format!("{type_:?} [{format:?}] unified 75 ({word0:#010x})");
            let (source, code) = mubuf_typed_source(word0, 75)
                .unwrap_or_else(|e| panic!("{label} must recompile: {e}"));

            // The fixture really is the row under test.
            let inst = &code.get_instructions()[0];
            assert_eq!(inst.type_, type_, "{label}: decoded type");
            assert_eq!(inst.format, format, "{label}: decoded format");

            assert!(
                source.contains(&format!("OpFunctionCall %void {helper} ")),
                "{label} must call {helper}:\n{source}"
            );
            // Packed, not unified: `14 * 8 + 4` = 116, never the raw 75.
            assert!(
                source.contains("OpStore %temp_int_5 %int_116"),
                "{label}: unified 75 must reach {helper} as PACKED 116:\n{source}"
            );
            // The guard constant has to be declared too — an undeclared
            // `%int_116` is a forward reference that fails assembly outright.
            assert!(
                source.contains("OpConstant %int 116"),
                "{label}: %int_116 must be declared:\n{source}"
            );

            let words = spirv_run(&source).unwrap_or_else(|e| panic!("{label} assembles: {e}"));
            spirv_val_ok(&words, &label);
        }
    }

    /// Every format a typed helper *claims* in `accepted` must be one its
    /// SPIR-V body's runtime guard actually tests — and vice versa.
    ///
    /// The two halves live in different files and fail in opposite directions,
    /// which is why this is checked as an equality rather than a subset:
    ///
    /// * a format in `accepted` that the guard rejects translates cleanly and
    ///   then does nothing at all — the silent no-op this whole table exists to
    ///   prevent, and not hypothetical: the emitting row passes the *resolved*
    ///   descriptor format (`OpStore %temp_int_5 %int_{packed_format}`), so
    ///   widening `accepted` alone reintroduces it;
    /// * a format the guard tests but `accepted` omits is refused with a
    ///   message claiming the helper cannot serve it, when it can — Dead Cells'
    ///   x4 store spent every frame on the wrong side of exactly that.
    ///
    /// `BUF_LOAD_FORMAT_XYZW_UNORM8` is excluded by `takes_format_arg`: it
    /// resolves the format at translate time and carries no guard to agree with.
    #[test]
    fn every_format_the_typed_helpers_accept_is_one_their_spirv_guard_tests() {
        use crate::shader::spirv::{
            TBUFFER_LOAD_FORMAT_X, TBUFFER_LOAD_FORMAT_XYZW, TBUFFER_STORE_FORMAT_X,
            TBUFFER_STORE_FORMAT_XY, TBUFFER_STORE_FORMAT_XYZW,
        };

        let pairs: [(&TypedBufferHelper, &str); 5] = [
            (&TBUF_LOAD_FORMAT_X, TBUFFER_LOAD_FORMAT_X),
            (&TBUF_STORE_FORMAT_X, TBUFFER_STORE_FORMAT_X),
            (&TBUF_STORE_FORMAT_XY, TBUFFER_STORE_FORMAT_XY),
            (&TBUF_LOAD_FORMAT_XYZW, TBUFFER_LOAD_FORMAT_XYZW),
            (&TBUF_STORE_FORMAT_XYZW, TBUFFER_STORE_FORMAT_XYZW),
        ];

        for (helper, body) in pairs {
            assert!(
                helper.takes_format_arg,
                "{}: only a runtime-guarded helper can be paired this way",
                helper.name
            );

            // What the body's guard really tests: the `%int_N` operand of every
            // equality in it.
            let mut guarded: Vec<u32> = body
                .lines()
                .filter(|line| line.contains("OpIEqual %bool"))
                .filter_map(|line| line.rsplit("%int_").next())
                .filter_map(|n| n.trim().parse::<u32>().ok())
                .collect();
            guarded.sort_unstable();
            assert!(
                !guarded.is_empty(),
                "{}: the body must guard on at least one format, or this test \
                 is vacuous — check the OpIEqual spelling",
                helper.name
            );

            let mut claimed: Vec<u32> = helper.accepted.iter().map(|&(p, _)| p).collect();
            claimed.sort_unstable();

            assert_eq!(
                claimed, guarded,
                "{}: `accepted` claims {claimed:?} but its SPIR-V guard tests \
                 {guarded:?} — the refusal table and the body must agree, or a \
                 format is either silently dropped or needlessly refused",
                helper.name
            );
        }
    }

    /// The other half of the same gate, per row: a descriptor format the
    /// helper's guard does not accept is refused **by name** and counted,
    /// rather than translating into an access that quietly does nothing.
    ///
    /// The refusal names both numbering schemes, because the fix is a
    /// descriptor-side fact: unified 56 is `(dfmt 10, nfmt 0)`, a real RDNA2
    /// format with no raw-dword spelling any of these helpers implements.
    #[test]
    fn a_format_the_helper_cannot_serve_is_refused_per_row_and_counted() {
        // (word0, the helper named in the refusal, what it does serve)
        const ROWS: &[(u32, &str, &str)] = &[
            (
                0xE000_2000,
                "tbuffer_load_format_x",
                "unified 20 / packed 36",
            ),
            (
                0xE010_2000,
                "tbuffer_store_format_x",
                "unified 20 / packed 36",
            ),
            (
                0xE014_2000,
                "tbuffer_store_format_xy",
                "unified 62 / packed 92",
            ),
            (
                0xE000_0000,
                "tbuffer_load_format_x",
                "unified 22 / packed 39",
            ),
            (
                0xE010_1000,
                "tbuffer_store_format_x",
                "unified 22 / packed 39",
            ),
            (
                0xE01C_3000,
                "tbuffer_store_format_xyzw",
                "unified 77 / packed 119",
            ),
            // `0xE00C_2000` — `BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen]` —
            // is deliberately NOT in this table: it is the one row that now
            // serves unified 56 through `%buffer_load_format_xyzw_unorm8`. Its
            // own refusal is covered by
            // `the_four_channel_load_still_refuses_a_format_neither_helper_serves`,
            // against a format neither of its two helpers implements.
        ];

        for &(word0, helper, served) in ROWS {
            let before = crate::shader::spirv::unsupported_buffer_format_skips();
            // Unified 56 = (dfmt 10, nfmt 0) — 8_8_8_8_UNORM, the format the
            // retail measurement actually hits once unified-77 streams
            // translate. None of the helpers in this table serves it.
            let err = mubuf_typed_source(word0, 56)
                .expect_err("a format no helper serves must not translate silently");
            let text = err.to_string();
            for want in [
                "unified format 56",
                "dfmt 10, nfmt 0",
                helper,
                served,
                "s[4:7]",
            ] {
                assert!(
                    text.contains(want),
                    "{word0:#010x}: refusal must name {want}, got: {text}"
                );
            }
            assert!(
                crate::shader::spirv::unsupported_buffer_format_skips() > before,
                "{word0:#010x}: the refusal must be counted"
            );
        }
    }

    /// Avatar: Frontiers of Pandora's measured shader blocker, end to end.
    ///
    /// After the unit conversion made the descriptor's format legible, the
    /// title's remaining refusal named itself precisely — `V# unified format 56
    /// (dfmt 10, nfmt 0) is not 32_32_32_32_FLOAT`, 769 occurrences in a 180 s
    /// run. `dfmt 10, nfmt 0` is `8_8_8_8_UNORM`: four normalized bytes, which
    /// Kyty's 32-bit-only `tbuffer_load_format_xyzw` does not implement.
    ///
    /// The dispatch is a translate-time choice from the bound descriptor, so
    /// the assertion is that the unified-56 site calls the UNORM helper and
    /// passes it NO format argument (there is nothing left to test at runtime),
    /// while the unified-77 site is unchanged.
    #[test]
    fn a_unified_56_descriptor_selects_the_8_8_8_8_unorm_unpack() {
        // `BufferLoadFormatXyzw [Vdata4VaddrSvSoffsIdxen]`, Avatar's row.
        const WORD0: u32 = 0xE00C_2000;

        // Not asserted against `unsupported_buffer_format_skips()`: that
        // counter is a process-global atomic and the refusal tests run
        // concurrently, so only its monotonic `>` direction is testable. That
        // this format is no longer counted as a skip is established by the
        // translation succeeding at all — the counter is only ever bumped on
        // the refusal path.
        let (source, code) = mubuf_typed_source(WORD0, 56)
            .expect("unified 56 (8_8_8_8_UNORM) must now translate, not refuse");
        assert_eq!(
            code.get_instructions()[0].type_,
            T::BufferLoadFormatXyzw,
            "fixture must be the four-channel load row"
        );

        assert!(
            source.contains("OpFunctionCall %void %buffer_load_format_xyzw_unorm8 "),
            "unified 56 must select the UNORM unpack:\n{source}"
        );
        assert!(
            !source.contains("OpFunctionCall %void %tbuffer_load_format_xyzw "),
            "unified 56 must NOT go through the 119-only float4 helper:\n{source}"
        );
        // No runtime format test survives on this path: the helper has no
        // `dfmt_nfmt` parameter at all, so nothing stores or passes temp_int_5.
        assert!(
            !source.contains("OpStore %temp_int_5"),
            "the UNORM path must carry no runtime format argument:\n{source}"
        );
        for gone in ["%mbf_f0_", "%t206_0", "%t208_0"] {
            assert!(
                !source.contains(gone),
                "runtime format extraction `{gone}` must be gone:\n{source}"
            );
        }
        // The channel order and the normalization, as emitted: component `c` is
        // the byte at `base + c` (KytyPS5 `BufferFormat.h` `component_bit_offset
        // {0,8,16,24}` with `packed_bitfield = false`; SharpEmu
        // `SetLayout(10, c, 0, 8)`), scaled by 1/255 (`(1 << 8) - 1`).
        for (component, offset_const) in
            [(0, "%int_0"), (1, "%int_1"), (2, "%int_2"), (3, "%int_3")]
        {
            assert!(
                source.contains(&format!(
                    "%buf_l_u8_{n}0 = OpIAdd %int %buf_l_u8_49 {offset_const}",
                    n = component + 6
                )),
                "component {component} must read the byte at base + {component}:\n{source}"
            );
        }
        assert_eq!(
            source.matches("OpBitFieldUExtract %uint").count(),
            4,
            "exactly four byte extractions, one per channel:\n{source}"
        );
        assert_eq!(
            source.matches("OpFDiv %float").count(),
            4,
            "exactly four normalizations:\n{source}"
        );
        assert_eq!(
            source.matches("%float_255_000000").count(),
            5, // four uses + the OpConstant declaration
            "each channel divides by 255.0:\n{source}"
        );

        let words = spirv_run(&source).expect("the UNORM path assembles");
        spirv_val_ok(&words, "buffer_load_format_xyzw unified 56 (8_8_8_8_UNORM)");

        // The float4 sibling is untouched by the dispatch.
        let (float4, _) = mubuf_typed_source(WORD0, 77).expect("unified 77 must still translate");
        assert!(
            float4.contains("OpFunctionCall %void %tbuffer_load_format_xyzw ")
                && float4.contains("OpStore %temp_int_5 %int_119"),
            "unified 77 must still take the packed-119 float4 path:\n{float4}"
        );
    }

    /// The four-channel load now serves two formats, so its refusal must list
    /// both — and must still fire for a third. Unified 13 = `(dfmt 2, nfmt 7)`
    /// is a real RDNA2 format that neither helper unpacks.
    #[test]
    fn the_four_channel_load_still_refuses_a_format_neither_helper_serves() {
        const WORD0: u32 = 0xE00C_2000;

        let before = crate::shader::spirv::unsupported_buffer_format_skips();
        let err = mubuf_typed_source(WORD0, 13)
            .expect_err("a format neither helper serves must not translate silently");
        let text = err.to_string();
        for want in [
            "unified format 13",
            "dfmt 2, nfmt 7",
            // both candidates are named, and both of their formats listed
            "tbuffer_load_format_xyzw / buffer_load_format_xyzw_unorm8",
            "32_32_32_32_FLOAT (unified 77 / packed 119)",
            "8_8_8_8_UNORM (unified 56 / packed 80)",
            "s[4:7]",
            // the consequence is still the load wording, not the store one
            "untouched",
        ] {
            assert!(text.contains(want), "refusal must name {want}, got: {text}");
        }
        assert!(
            crate::shader::spirv::unsupported_buffer_format_skips() > before,
            "the refusal must be counted"
        );
    }

    /// `8_8_8_8_UNORM` is `dfmt 10, nfmt 0`, so it packs to `10 * 8 + 0` = 80
    /// and lives at RDNA2 unified 56. Both directions must agree, because the
    /// dispatch keys on the packed number while the descriptor and every log
    /// line speak the unified one.
    #[test]
    fn the_8_8_8_8_unorm_encoding_round_trips_between_both_numberings() {
        use crate::shader::spirv::{
            gfx10_packed_to_unified_dfmt_nfmt, gfx10_unified_to_dfmt_nfmt,
            gfx10_unified_to_packed_dfmt_nfmt,
        };

        assert_eq!(gfx10_unified_to_dfmt_nfmt(56), Some((10, 0)));
        assert_eq!(gfx10_unified_to_packed_dfmt_nfmt(56), Some(80));
        assert_eq!(gfx10_packed_to_unified_dfmt_nfmt(80), Some(56));
        assert_eq!(
            BUF_LOAD_FORMAT_XYZW_UNORM8.accepted,
            &[(80, "8_8_8_8_UNORM")],
            "the helper must advertise exactly the packed number the dispatch keys on"
        );
        // 80 is not the unified spelling of anything else — the two schemes
        // must not be conflated the way 77/119 were.
        assert_ne!(
            gfx10_unified_to_packed_dfmt_nfmt(80),
            Some(80),
            "unified 80 is a different format from packed 80"
        );
    }

    /// `mubuf_flexible` is shared with the RAW dword/byte rows, which have no
    /// format guard at all. They must be untouched: no `temp_int_5` argument,
    /// no descriptor-format lookup, and therefore no new refusal — a
    /// `buffer_load_dword` through a descriptor whose format is meaningless
    /// still has to compile.
    #[test]
    fn the_raw_mubuf_rows_keep_no_format_argument() {
        // (word0, what it decodes to, the helper it calls)
        const ROWS: &[(u32, T, &str)] = &[
            (0xE030_2000, T::BufferLoadDword, "%buffer_load_float1"),
            (0xE030_0000, T::BufferLoadDword, "%buffer_load_float1"),
            (0xE030_3000, T::BufferLoadDword, "%buffer_load_float1"),
            (0xE070_2000, T::BufferStoreDword, "%buffer_store_float1"),
            (0xE070_1000, T::BufferStoreDword, "%buffer_store_float1"),
            (0xE020_2000, T::BufferLoadUbyte, "%buffer_load_ubyte"),
            (0xE020_3000, T::BufferLoadUbyte, "%buffer_load_ubyte"),
        ];

        for &(word0, type_, helper) in ROWS {
            // Unified 56 is the format the typed rows refuse; a raw row must
            // not even look at it.
            let (source, code) = mubuf_typed_source(word0, 56)
                .unwrap_or_else(|e| panic!("{type_:?} ({word0:#010x}) must recompile: {e}"));
            let label = format!("{type_:?} ({word0:#010x})");
            assert_eq!(code.get_instructions()[0].type_, type_, "{label}");
            assert!(
                source.contains(&format!("OpFunctionCall %void {helper} ")),
                "{label} must call {helper}:\n{source}"
            );
            assert!(
                !source.contains("OpFunctionCall %void %tbuffer_"),
                "{label} must not reach a typed helper:\n{source}"
            );
            assert!(
                !source.contains("%mbf_f0_"),
                "{label} must not read the descriptor format at all:\n{source}"
            );
            let words = spirv_run(&source).unwrap_or_else(|e| panic!("{label} assembles: {e}"));
            spirv_val_ok(&words, &label);
        }
    }

    /// The scalar-evaluator acceptance case: real GCN bytes whose scalar load
    /// soffset is **computed** — not a live-in and not a single constant move —
    /// must now resolve all the way to spirv-val-clean SPIR-V.
    ///
    /// ```text
    /// s_lshl_b32     s4, s5, 2                    8F048205
    /// s_add_u32      s4, s4, 16                   80049004
    /// s_load_dwordx4 s[16:19], s[12:13], s4 off:16 F4080406 08000010
    /// s_endpgm
    /// ```
    ///
    /// With the live-in `s5 = 6` the address is `ptr + (6 << 2) + 16 + 16`.
    /// Before `shader::scalar_eval` the analysis pass could not prove `s4` (its
    /// last writer was `s_add_u32`, not `s_mov_b32`), so nothing was captured
    /// and `sload_dword_extended` refused the load by name — "unresolved
    /// register soffset" — dropping the whole dispatch.
    #[test]
    fn full_chain_computed_soffset_bytes_to_validated_spirv() {
        use std::borrow::Cow;

        struct Mem(u64, Vec<u32>);
        impl crate::shader::analysis::ShaderMemory for Mem {
            fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
                (addr >= self.0 && (addr - self.0) % 4 == 0)
                    .then(|| ((addr - self.0) / 4) as usize)
                    .filter(|start| *start < self.1.len())
                    .map(|start| Cow::Borrowed(&self.1[start..]))
            }
        }

        let words = vec![
            0x8F04_8205, // pc 0x00: s_lshl_b32 s4, s5, 2
            0x8004_9004, // pc 0x04: s_add_u32  s4, s4, 16
            0xF408_0406, // pc 0x08: s_load_dwordx4 s[16:19], s[12:13], s4
            0x0800_0010, //          offset:16
            S_ENDPGM,    // pc 0x10
        ];

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        crate::shader::parse::shader_parse(0, &words, &mut code, true)
            .expect("computed-soffset fixture must parse");
        assert_eq!(
            code.get_instructions()[0].type_,
            ShaderInstructionType::SLshlB32,
            "the fixture really contains scalar arithmetic, not a constant move"
        );
        assert_eq!(
            code.get_instructions()[2].format,
            F::Sdst4SbaseSoffsetOffset
        );

        // ptr(s12:s13) = 0x0080_0000; s5 = 6 -> s4 = 6<<2 = 0x18, +16 = 0x28;
        // plus the 16-byte immediate -> 0x0080_0038.
        let payload = vec![0x1111_2222u32, 0x3333_4444, 0x5555_6666, 0x7777_8888];
        let mem = Mem(0x0080_0038, payload.clone());
        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        user_sgpr.set(5, 6, crate::shader::hw_regs::UserSgprType::Unknown);
        user_sgpr.set(
            12,
            0x0080_0000,
            crate::shader::hw_regs::UserSgprType::Unknown,
        );
        user_sgpr.set(13, 0, crate::shader::hw_regs::UserSgprType::Unknown);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        crate::shader::analysis::shader_capture_runtime_scalar_loads(
            &code,
            &mem,
            &user_sgpr,
            &mut input_info.bind,
        );
        assert_eq!(
            input_info
                .bind
                .embedded_constant_loads
                .find(0x08)
                .expect("a computed soffset must now resolve")
                .values[..4],
            payload[..],
            "the evaluator must fold s_lshl_b32 + s_add_u32 into the address"
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("full chain recompiles");
        let module = spirv_run(&source).expect("assemble");
        spirv_val_ok(&module, "full_chain_computed_soffset");
    }

    /// The other side of the same gate: when the soffset is genuinely
    /// unprovable the recompiler must still refuse **by name**, not emit a
    /// module built on a guessed address.
    ///
    /// ```text
    /// s_load_dword   s5, s[12:13], 0x100          F400_0146 0000_0100
    /// s_lshl_b32     s4, s5, 2                    8F04_8205
    /// s_load_dwordx4 s[16:19], s[12:13], s4 off:16 F408_0406 0800_0010
    /// s_endpgm
    /// ```
    ///
    /// `s5` comes out of guest memory, so `s4` has no dispatch-time value; the
    /// evaluator says unknown and nothing is captured for the second load.
    #[test]
    fn a_memory_dependent_soffset_keeps_the_named_refusal() {
        use std::borrow::Cow;

        struct Mem(u64, Vec<u32>);
        impl crate::shader::analysis::ShaderMemory for Mem {
            fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
                (addr >= self.0 && (addr - self.0) % 4 == 0)
                    .then(|| ((addr - self.0) / 4) as usize)
                    .filter(|start| *start < self.1.len())
                    .map(|start| Cow::Borrowed(&self.1[start..]))
            }
        }

        let words = vec![
            0xF400_0146, // pc 0x00: s_load_dword s5, s[12:13], ...
            0x0000_0100, //          offset:0x100 (NULL soffset)
            0x8F04_8205, // pc 0x08: s_lshl_b32 s4, s5, 2
            0xF408_0406, // pc 0x0c: s_load_dwordx4 s[16:19], s[12:13], s4
            0x0800_0010, //          offset:16
            S_ENDPGM,    // pc 0x14
        ];

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        crate::shader::parse::shader_parse(0, &words, &mut code, true)
            .expect("memory-dependent fixture must parse");

        // Every address the pass could wrongly land on is mapped, so a silent
        // fold would show up as a capture instead of a refusal.
        let mem = Mem(0x0080_0000, vec![0x2020_2020u32; 1024]);
        let mut user_sgpr = crate::shader::hw_regs::UserSgprInfo::default();
        user_sgpr.set(5, 0, crate::shader::hw_regs::UserSgprType::Unknown);
        user_sgpr.set(
            12,
            0x0080_0000,
            crate::shader::hw_regs::UserSgprType::Unknown,
        );
        user_sgpr.set(13, 0, crate::shader::hw_regs::UserSgprType::Unknown);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        crate::shader::analysis::shader_capture_runtime_scalar_loads(
            &code,
            &mem,
            &user_sgpr,
            &mut input_info.bind,
        );
        assert!(
            input_info.bind.embedded_constant_loads.find(0x0c).is_none(),
            "a memory-dependent soffset must not be captured"
        );

        let err = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect_err("an unprovable soffset must refuse, not guess");
        let msg = format!("{err}");
        assert!(
            msg.contains("unresolved register soffset"),
            "the refusal must stay identifiable by name; got: {msg}"
        );
    }

    // ---- Blasphemous II shader-decoder gaps -------------------------------
    //
    // Six register-independent decoder gaps, each reproduced with hand-built
    // GCN words rather than dumped bytes. The encodings were recovered from a
    // live run's `RAEEN_DUMP_SHADERS` output and re-spelled here from the ISA
    // fields, so no guest bytes enter the tree.

    /// EXP targets 0x25 and 0x26 — `param5` / `param6`.
    ///
    /// Pre-fix the parser matched only targets 0x20..=0x24, so the whole
    /// vertex shader was refused with
    /// `unknown exp target: 0x26 at addr 0x00000008 (en=0xf done=0 compr=0 vm=0)`
    /// — six of thirteen "guest shader analysis failed" refusals in one
    /// measured run, from just this gap.
    ///
    /// ISA fact (AMD RDNA 2 ISA reference, doc 70648, EXP target encoding):
    /// 0..7 = MRT0-7, 8 = Z, 9 = NULL, 12..15 = POS0-3, 20 = PRIM, and
    /// **32..63 = PARAM0..PARAM31**. 0x25/0x26 are 37/38, i.e. PARAM5/PARAM6:
    /// ordinary vertex parameter exports with the same four-vsrc operand
    /// shape as param0..4.
    #[test]
    fn exp_param5_and_param6_decode_and_recompile() {
        // exp pos0 v0..v3 done; exp param5 v0..v3; exp param6 v0..v3; endpgm.
        // EXP word0 = 0x3e << 26 | vm << 12 | done << 11 | compr << 10
        //             | target << 4 | en.
        let code = parse(
            &[
                0xF800_08CF,
                0x0302_0100, // exp pos0 v[0:3] done
                0xF800_025F,
                0x0302_0100, // exp param5 v[0:3]  (target 0x25, en=0xf)
                0xF800_026F,
                0x0302_0100, // exp param6 v[0:3]  (target 0x26, en=0xf)
                S_ENDPGM,
            ],
            ShaderType::Vertex,
        );

        let params: Vec<(F, u32)> = code
            .get_instructions()
            .iter()
            .filter(|i| i.type_ == T::Exp)
            .map(|i| (i.format, i.export_enable))
            .collect();
        assert_eq!(
            params,
            vec![
                (F::Pos0Vsrc0Vsrc1Vsrc2Vsrc3Done, 0xf),
                (F::Param5Vsrc0Vsrc1Vsrc2Vsrc3, 0xf),
                (F::Param6Vsrc0Vsrc1Vsrc2Vsrc3, 0xf),
            ],
        );

        // `export_count` deliberately under-reads the body (1), the way the
        // measured `spi_vs_out_config` does: the declarations must follow the
        // body or assembly dies on an undefined %param6.
        let input_info = ShaderVertexInputInfo {
            export_count: 1,
            ..Default::default()
        };
        let source = spirv_generate_source(&code, Some(&input_info), None, None)
            .expect("param5/param6 exports must recompile");
        for slot in 0..=6 {
            assert!(
                source.contains(&format!(
                    "%param{slot} = OpVariable %_ptr_Output_v4float Output"
                )),
                "param{slot} must be declared:\n{source}"
            );
            assert!(
                source.contains(&format!("OpDecorate %param{slot} Location {slot}")),
                "param{slot} needs its Location:\n{source}"
            );
        }
        assert!(source.contains("OpStore %param5"), "{source}");
        assert!(source.contains("OpStore %param6"), "{source}");

        let words = shader_recompile_vs(&code, &input_info).expect("assemble param5/param6 VS");
        naga_parse_and_validate(&words, "exp param5/param6");
    }

    /// A partial-mask high param export (`exp param5 ... en=0x7`, the measured
    /// three-channel form) rides the same path: the disabled channel is
    /// written as 0.0, matching the param0..4 behaviour.
    #[test]
    fn exp_param5_partial_channel_mask_decodes() {
        let code = parse(
            &[
                0xF800_08CF,
                0x0302_0100, // exp pos0 v[0:3] done
                0xF800_0257,
                0x0302_0100, // exp param5 v[0:2] (en=0x7)
                S_ENDPGM,
            ],
            ShaderType::Vertex,
        );
        let inst = code
            .get_instructions()
            .iter()
            .find(|i| i.format == F::Param5Vsrc0Vsrc1Vsrc2Vsrc3)
            .expect("param5 with a partial mask must decode");
        assert_eq!(inst.export_enable, 0x7);

        let input_info = ShaderVertexInputInfo {
            export_count: 1,
            ..Default::default()
        };
        let source = spirv_generate_source(&code, Some(&input_info), None, None)
            .expect("partial-mask param5 must recompile");
        assert!(
            source.contains("%float_0_000000"),
            "the disabled channel must be a defined zero:\n{source}"
        );
        let words = shader_recompile_vs(&code, &input_info).expect("assemble partial param5");
        naga_parse_and_validate(&words, "exp param5 en=0x7");
    }

    /// The parameter range ends at PARAM31 (target 0x3f). Target 0x40 does not
    /// exist — the EXP target field is 6 bits — but the arms either side of the
    /// range must behave: 0x3f decodes, and a target below 0x20 that no other
    /// arm claims still refuses by name rather than being folded into param0.
    #[test]
    fn exp_param_range_ends_at_param31_and_refuses_outside_it() {
        let code = parse(
            &[
                0xF800_08CF,
                0x0302_0100, // exp pos0 v[0:3] done
                0xF800_03FF,
                0x0302_0100, // exp param31 v[0:3] (target 0x3f, en=0xf)
                S_ENDPGM,
            ],
            ShaderType::Vertex,
        );
        assert!(
            code.get_instructions()
                .iter()
                .any(|i| i.format == F::Param31Vsrc0Vsrc1Vsrc2Vsrc3),
            "target 0x3f is PARAM31"
        );

        // Target 0x1f: below the parameter range and claimed by no other arm.
        let mut bad = ShaderCode::new();
        bad.set_type(ShaderType::Vertex);
        let err = crate::shader::parse::shader_parse(
            0,
            &[0xF800_01FF, 0x0302_0100, S_ENDPGM],
            &mut bad,
            true,
        )
        .expect_err("target 0x1f must not be accepted as a param export");
        assert!(
            format!("{err}").contains("unknown exp target: 0x1f"),
            "the refusal must name the target; got: {err}"
        );
    }

    /// One MTBUF word, built from ISA fields.
    ///
    /// GFX10/RDNA2 layout (AMD RDNA 2 ISA reference, doc 70648): encoding
    /// `0b111010` at `[31:26]`, `FORMAT[25:19]` (one **unified** 7-bit field —
    /// GCN's split `DATA_FORMAT[22:19]` + `NUM_FORMAT[25:23]` is gone),
    /// `OP[18:16]`, DLC`[15]`, `IDXEN[13]`, `OFFEN[12]`, `OFFSET[11:0]`.
    fn mtbuf_word0(unified_format: u32, op: u32, idxen: u32, offen: u32, offset: u32) -> u32 {
        (0x3a << 26)
            | ((unified_format & 0x7f) << 19)
            | ((op & 0x7) << 16)
            | (idxen << 13)
            | (offen << 12)
            | (offset & 0xfff)
    }

    /// One MTBUF second word: `SOFFSET[31:24]`, `TFE[23]`, `SLC[22]`,
    /// OP{3}`[21]`, `SRSRC[20:16]` (the T#/V# start register / 4),
    /// `VDATA[15:8]`, `VADDR[7:0]`.
    fn mtbuf_word1(soffset: u32, srsrc: u32, vdata: u32, vaddr: u32) -> u32 {
        (soffset << 24) | ((srsrc & 0x1f) << 16) | ((vdata & 0xff) << 8) | (vaddr & 0xff)
    }

    /// Pixel-shader input info with one storage buffer bound, so the
    /// `tbuffer_*` bodies (which index `%buf`) are reachable.
    fn ps_info_with_one_buffer() -> ShaderPixelInputInfo {
        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 48;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info
    }

    /// The GFX10 MTBUF element format is ONE 7-bit unified field, not GCN's
    /// split `dfmt`/`nfmt`.
    ///
    /// Pre-fix, reading the split fields produced
    /// `unknown format: dfmt = 6, nfmt = 1 at addr 0x00000000` on four pixel
    /// shaders and `dfmt = 13, nfmt = 4` on one vertex shader — five of
    /// thirteen refusals in one measured Blasphemous II run.
    ///
    /// The two numbers are the same bits read two ways: `6 | (1 << 4)` = 22 and
    /// `13 | (4 << 4)` = 77. Unified 22 is `32_FLOAT` and 77 is
    /// `32_32_32_32_FLOAT` (RDNA2 ISA table 47, already transcribed here as
    /// [`crate::shader::spirv::gfx10_unified_to_dfmt_nfmt`]) — which are
    /// exactly the formats their opcodes require. Two independently encoded
    /// fields agreeing on the channel count, in both samples, is what
    /// establishes the reading.
    #[test]
    fn mtbuf_gfx10_format_is_one_unified_field_not_split_dfmt_nfmt() {
        // op 0 = tbuffer_load_format_x, unified 22 = 32_FLOAT (1 channel).
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                mtbuf_word0(22, 0, 1, 0, 0),
                mtbuf_word1(0x80, 1, 6, 2),
                0xBF80_0000, // s_nop x2: Recompile_SEndpgm_Empty needs index >= 2
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("unified format 22 is 32_FLOAT and must parse");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::TBufferLoadFormatX);
        assert_eq!(inst.format, F::Vdata1VaddrSvSoffsIdxenFloat1);
        assert_eq!(inst.dst.size, 1);

        let input_info = ps_info_with_one_buffer();
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile unified-format tbuffer_load_format_x");
        assert!(
            source.contains("%tbuffer_load_format_x"),
            "the typed load must be emitted:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble unified-format tbuffer_load_format_x");

        // op 3 = tbuffer_load_format_xyzw, unified 77 = 32_32_32_32_FLOAT
        // (4 channels). The pair below is the whole argument: the opcode's
        // channel count and the format's agree only under the unified reading.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                mtbuf_word0(77, 3, 1, 0, 0),
                mtbuf_word1(0x80, 1, 8, 4),
                0xBF80_0000,
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("unified format 77 is 32_32_32_32_FLOAT and must parse");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::TBufferLoadFormatXyzw);
        assert_eq!(inst.format, F::Vdata4VaddrSvSoffsIdxenFloat4);
        assert_eq!(inst.dst.size, 4);

        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile unified-format tbuffer_load_format_xyzw");
        assert!(source.contains("%tbuffer_load_format_xyzw"), "{source}");
        let _ = spirv_run(&source).expect("assemble unified-format tbuffer_load_format_xyzw");
    }

    /// The two readings are gated by generation, and the SAME word decodes
    /// differently under each — which is the precise statement of the fix.
    ///
    /// The runtime tries next-gen first and falls back to legacy
    /// (`raeen_gpu::shader_fetch::attempt_generations`), so a stream that
    /// genuinely spells a legacy `dfmt`/`nfmt` pair still decodes through the
    /// second attempt. That is why gating rather than replacing is safe.
    #[test]
    fn mtbuf_legacy_split_format_still_decodes_on_the_legacy_path() {
        let parse_gen = |words: &[u32], next_gen: bool| {
            let mut code = ShaderCode::new();
            code.set_type(ShaderType::Vertex);
            crate::shader::parse::shader_parse(0, words, &mut code, next_gen).map(|_| code)
        };

        // Bits spelling legacy dfmt 14 / nfmt 7 (`32_32_32_32_FLOAT` on GCN):
        // FORMAT[22:19] = 14, FORMAT[25:23] = 7, i.e. unified 14 | (7 << 4) =
        // 126 — reserved in RDNA2 table 47.
        let legacy = [
            (0x3a << 26) | (14 << 19) | (7 << 23) | (3 << 16) | (1 << 13),
            mtbuf_word1(0x80, 1, 4, 0),
            S_ENDPGM,
        ];
        let code = parse_gen(&legacy, false).expect("the legacy split reading still applies");
        assert_eq!(code.get_instructions()[0].type_, T::TBufferLoadFormatXyzw);

        let err = parse_gen(&legacy, true)
            .expect_err("unified 126 is reserved and must not be guessed at");
        assert!(
            format!("{err}").contains("unknown mtbuf unified format: 126"),
            "the refusal must name the unified field that actually exists; got: {err}"
        );

        // And the converse: a real RDNA2 word (unified 77) is nonsense under
        // the legacy split, where it reads as dfmt 13 / nfmt 4.
        let next = [
            mtbuf_word0(77, 3, 1, 0, 0),
            mtbuf_word1(0x80, 1, 4, 0),
            S_ENDPGM,
        ];
        parse_gen(&next, true).expect("unified 77 decodes on the next-gen path");
        let err = parse_gen(&next, false).expect_err("the legacy split cannot serve unified 77");
        assert!(
            format!("{err}").contains("dfmt = 13, nfmt = 4"),
            "the legacy refusal keeps naming legacy fields; got: {err}"
        );
    }

    /// A unified FORMAT the RDNA2 table reserves is refused by name, not
    /// silently mapped to something plausible. 30 is one such hole (the table
    /// jumps 29 -> 36).
    #[test]
    fn mtbuf_reserved_unified_format_refuses_by_name() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        let err = crate::shader::parse::shader_parse(
            0,
            &[
                mtbuf_word0(30, 3, 1, 0, 0),
                mtbuf_word1(0x80, 1, 4, 0),
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect_err("a reserved unified format must refuse");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown mtbuf unified format: 30") && msg.contains("table 47"),
            "the refusal must name the encoding and where it is reserved; got: {msg}"
        );
    }

    /// GFX10 moves MTBUF `OP{3}` to word1 bit 21, so the 3-bit `[18:16]` mask
    /// cannot distinguish `tbuffer_load_format_x` from the D16 sibling that
    /// moves half the bytes per channel. Decoding one as the other would read
    /// the wrong element stride and return wrong vertex data silently, so it is
    /// a named refusal. Measured Blasphemous II MTBUFs all have this bit clear.
    #[test]
    fn mtbuf_d16_opcode_high_bit_refuses_rather_than_aliasing_the_full_width_form() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        let err = crate::shader::parse::shader_parse(
            0,
            &[
                mtbuf_word0(22, 0, 1, 0, 0),
                mtbuf_word1(0x80, 1, 6, 2) | (1 << 21),
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect_err("op{3} set is a d16 opcode and must not alias the full-width form");
        assert!(
            format!("{err}").contains("d16 opcode (op{3} set)"),
            "the refusal must name the family; got: {err}"
        );
    }

    /// MTBUF's 12-bit immediate `OFFSET` is a plain byte addend of the same
    /// address term as SOFFSET, so it folds into the constant soffset operand
    /// the recompile bodies already route into `%temp_int_2`.
    ///
    /// Pre-fix this was `not implemented mtbuf feature: offset != 0 at addr
    /// 0x00000000` — one Blasphemous II vertex shader per run. Behind it sat a
    /// second gap: the opcode is 1, `tbuffer_load_format_xy`, which upstream
    /// leaves `KYTY_NI`. Both are closed here, so this test also covers the
    /// two-channel typed fetch end to end.
    #[test]
    fn mtbuf_immediate_offset_folds_and_format_xy_recompiles() {
        // op 1 = tbuffer_load_format_xy, unified 64 = 32_32_FLOAT (2 channels),
        // immediate offset 4, idxen, soffset = inline constant 0.
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[
                mtbuf_word0(64, 1, 1, 0, 4),
                mtbuf_word1(0x80, 1, 6, 4),
                0xBF80_0000,
                0xBF80_0000,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("an immediate offset must fold, not refuse the shader");

        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::TBufferLoadFormatXy);
        assert_eq!(inst.format, F::Vdata2VaddrSvSoffsIdxenFloat2);
        assert_eq!(inst.dst.size, 2, "two channels, two vdata registers");
        assert_eq!(
            inst.src[2].type_,
            crate::shader::types::ShaderOperandType::IntegerInlineConstant,
        );
        assert_eq!(
            inst.src[2].constant.u, 4,
            "soffset 0 + immediate 4 is a byte offset of 4"
        );

        let input_info = ps_info_with_one_buffer();
        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile tbuffer_load_format_xy");
        assert!(
            source.contains("%tbuffer_load_format_xy = OpFunction")
                && source.contains("%buffer_load_float2 = OpFunction"),
            "both new helpers must be emitted:\n{source}"
        );
        assert!(
            source.contains("OpStore %temp_int_2 %int_4"),
            "the folded byte offset must reach the address slot:\n{source}"
        );
        assert!(
            source.contains("OpStore %temp_int_5 %int_95"),
            "packed 95 = dfmt 11, nfmt 7 is what the helper's guard admits:\n{source}"
        );
        let _ = spirv_run(&source).expect("assemble tbuffer_load_format_xy");
    }

    /// A register SOFFSET plus a non-zero immediate needs a runtime add that no
    /// `tbuffer_*` body models, so it refuses by name rather than dropping one
    /// of the two address terms. Same rule `shader_parse_mubuf` already applies.
    #[test]
    fn mtbuf_immediate_offset_with_register_soffset_refuses_by_name() {
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Vertex);
        let err = crate::shader::parse::shader_parse(
            0,
            &[
                mtbuf_word0(64, 1, 1, 0, 4),
                // soffset = s3 (a register, not an inline constant).
                mtbuf_word1(3, 1, 6, 4),
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect_err("an unmodelled address term must refuse, not be dropped");
        assert!(
            format!("{err}").contains("offset != 0 with register soffset"),
            "the refusal must say which combination; got: {err}"
        );
    }

    /// `image_sample` with DMASK 0xb — components X, Y and W.
    ///
    /// Pre-fix: `unknown mimg format for opcode: 0x20 at addr 0x00000000,
    /// dmask: 0xb`, which refused one whole Blasphemous II pixel shader.
    ///
    /// ISA fact (doc 70648, MIMG DMASK): the mask names which texel components
    /// are RETURNED, and they are packed into consecutive VGPRs in ascending
    /// component order. A gapped mask is therefore an ordinary
    /// destination-component subset — the already-supported 0x5 (XZ) and 0x9
    /// (XW) masks are the two-channel precedent, and 0xb is their three-channel
    /// sibling: three vdata registers holding (R, G, A).
    #[test]
    fn image_sample_dmask_b_returns_x_y_and_w() {
        // MIMG word0: encoding 0b111100 at [31:26], OP[24:18] = 0x20
        // (image_sample), DMASK[11:8] = 0xb, DIM[5:3] = 1 (2D).
        let w0 = (0x3c << 26) | (0x20 << 18) | (0xb << 8) | (1 << 3);
        // word1: SSAMP[25:21] = 2, SRSRC[20:16] = 4, VDATA[15:8] = v8,
        // VADDR[7:0] = v2.
        let w1 = (2 << 21) | (4 << 16) | (8 << 8) | 2;

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[w0, w1, 0xBF80_0000, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("dmask 0xb must decode, not refuse the shader");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageSample);
        assert_eq!(inst.format, F::Vdata3Vaddr3StSsDmaskB);
        assert_eq!(inst.dst.size, 3, "three enabled channels, three vdata regs");

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 16;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        input_info.bind.samplers.binding_index = 1;

        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile image_sample dmask 0xb");
        // The three reads are components 0, 1 and 3 — W, not Z.
        for chan in [0, 1, 3] {
            assert!(
                source.contains(&format!("%temp_v4float %uint_{chan}")),
                "component {chan} must be read:\n{source}"
            );
        }
        assert!(
            !source.contains("%temp_v4float %uint_2"),
            "component 2 (Z) is masked off and must not be read:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble image_sample dmask 0xb");
        naga_parse_and_validate(&words, "image_sample dmask 0xb");
    }

    /// `image_sample_lz` with DMASK 0x8 — component W only, at LOD zero.
    ///
    /// This gap was *hidden* behind the MTBUF immediate-offset refusal: the one
    /// vertex shader that contains it never reached this instruction, so the
    /// live log never named it. Closing the earlier gap exposed
    /// `unknown mimg format for opcode: 0x27 at addr 0x00000998, dmask: 0x8`.
    ///
    /// The plain `image_sample` (0x20) path has served dmask 0x8 since the
    /// ASTRO.BOT batch and `image_sample_c_lz` (0x2f) serves it too; DMASK is
    /// orthogonal to the LOD mode, so 0x27 takes the same single-channel body
    /// with component index 3.
    #[test]
    fn image_sample_lz_dmask8_returns_w_at_lod_zero() {
        // MIMG word0: OP[24:18] = 0x27 (image_sample_lz), DMASK[11:8] = 0x8,
        // DIM[5:3] = 1 (2D).
        let w0 = (0x3c << 26) | (0x27 << 18) | (0x8 << 8) | (1 << 3);
        let w1 = (2 << 21) | (4 << 16) | (8 << 8) | 2;

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Pixel);
        shader_parse(
            0,
            &[w0, w1, 0xBF80_0000, 0xBF80_0000, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("image_sample_lz dmask 0x8 must decode");
        let inst = &code.get_instructions()[0];
        assert_eq!(inst.type_, T::ImageSampleLz);
        assert_eq!(inst.format, F::Vdata1Vaddr3StSsDmask8);
        assert_eq!(inst.dst.size, 1);

        let mut input_info = ShaderPixelInputInfo::default();
        input_info.target_output_mode[0] = 4;
        input_info.bind.push_constant_size = 64;
        input_info.bind.textures2d.textures_num = 1;
        input_info.bind.textures2d.textures2d_sampled_num = 1;
        input_info.bind.textures2d.desc[0].start_register = 16;
        input_info.bind.samplers.samplers_num = 1;
        input_info.bind.samplers.start_register[0] = 8;
        input_info.bind.samplers.binding_index = 1;

        let source = spirv_generate_source(&code, None, Some(&input_info), None)
            .expect("recompile image_sample_lz dmask 0x8");
        assert!(
            source.contains("Lod %float_0_000000"),
            "LZ means an explicit LOD of zero:\n{source}"
        );
        assert!(
            source.contains("%temp_v4float %uint_3"),
            "component 3 (W) is the one enabled channel:\n{source}"
        );
        let words = spirv_run(&source).expect("assemble image_sample_lz dmask 0x8");
        naga_parse_and_validate(&words, "image_sample_lz dmask 0x8");
    }

    /// RDNA2 names VCC's two halves as INDEPENDENT scalar destinations (ISA
    /// 70648, "Scalar ALU Operands": the flat operand encoding puts 106 =
    /// `VCC_LO` and 107 = `VCC_HI` beside the SGPRs, and SMEM's `SDATA` field
    /// is that same encoding). So a ONE-dword `s_buffer_load` may target
    /// `vcc_hi` directly rather than reaching it as dword 1 of a `vcc_lo`
    /// pair.
    ///
    /// Measured as Blasphemous II's `s_buffer_load` translation blocker (PS
    /// `0x10001d65c00` / `0x10001d68600` / `0x10001d6de00`, which emit all
    /// three widths below):
    ///
    /// ```text
    /// s_buffer_load_dword   vcc_lo, s[12:15], 0x10
    /// s_buffer_load_dword   vcc_hi, s[12:15], 0x18   <-- refused the shader
    /// s_buffer_load_dwordx2 vcc_lo, s[12:15], 0x8
    /// ```
    ///
    /// `operand_variable_to_str_shift` had no `VccHi` arm, so shift 0 of a
    /// `VccHi` destination fell through to `Unknown` and
    /// `Recompile_SBufferLoadDword_SdstSvSoffset` refused with "unexpected
    /// operand types". The pair is kept in the fixture so the lo->hi walk that
    /// already worked is pinned alongside the new direct-hi form. Encodings are
    /// hand-built from the SMEM field layout, not copied from the title.
    #[test]
    fn sbuffer_load_dword_into_vcc_hi_recompiles() {
        // SMEM: [31:26]=0x3d, [25:18]=opcode, [12:6]=sdst, [5:0]=sbase (SGPR/2);
        // word 1: [31:25]=soffset (125 = NULL -> use the imm21), [20:0]=offset.
        const LOAD_VCC_LO_0X10: [u32; 2] = [0xF420_1A86, 0xFA00_0010];
        const LOAD_VCC_HI_0X18: [u32; 2] = [0xF420_1AC6, 0xFA00_0018];
        const LOAD_X2_VCC_0X08: [u32; 2] = [0xF424_1A86, 0xFA00_0008];

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[
                LOAD_VCC_LO_0X10[0],
                LOAD_VCC_LO_0X10[1],
                LOAD_VCC_HI_0X18[0],
                LOAD_VCC_HI_0X18[1],
                LOAD_X2_VCC_0X08[0],
                LOAD_X2_VCC_0X08[1],
                V_MOV_V0_0,
                S_ENDPGM,
            ],
            &mut code,
            true,
        )
        .expect("parse s_buffer_load with VCC destinations");

        // The encodings really do carry the destinations this test is about.
        let insts = code.get_instructions();
        assert_eq!(insts[0].type_, T::SBufferLoadDword);
        assert_eq!(insts[0].format, F::SdstSvSoffset);
        assert_eq!(insts[0].dst.type_, ShaderOperandType::VccLo);
        assert_eq!(insts[0].src[0].register_id, 12, "V# is s[12:15]");
        assert_eq!(insts[1].type_, T::SBufferLoadDword);
        assert_eq!(
            insts[1].dst.type_,
            ShaderOperandType::VccHi,
            "a one-dword load may name the HIGH half directly"
        );
        assert_eq!(insts[2].type_, T::SBufferLoadDwordx2);
        assert_eq!(insts[2].dst.type_, ShaderOperandType::VccLo);
        assert_eq!(insts[2].dst.size, 2);

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.storage_buffers.buffers_num = 1;
        input_info.bind.storage_buffers.start_register[0] = 12;

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("a VccHi scalar destination must recompile, not refuse");

        assert!(
            source.contains("%vcc_hi = OpVariable %_ptr_Function_uint Function"),
            "vcc_hi is declared as a uint register variable:\n{source}"
        );
        assert!(
            source.contains("%sbuffer_load_dword %vcc_hi "),
            "the one-dword load writes vcc_hi directly:\n{source}"
        );
        assert!(
            source.contains("%sbuffer_load_dword %vcc_lo "),
            "the one-dword lo load is unchanged:\n{source}"
        );
        assert!(
            source.contains("%sbuffer_load_dword_2 %vcc_lo %vcc_hi "),
            "the x2 load still walks vcc_lo then vcc_hi:\n{source}"
        );

        let words = spirv_run(&source).expect("assemble s_buffer_load into VCC halves");
        naga_parse_and_validate(&words, "sbuffer_load_vcc_hi");
    }

    /// A descriptor table deeper than Kyty's fixed 64-entry extended mapping.
    ///
    /// Measured as Blasphemous II's other translation blocker (PS
    /// `0x10001d00300`): it addresses ONE descriptor-table pointer pair
    /// (`s[28:29]`) with `s_load_dwordx8` at byte offsets
    /// `0x00/0x20/0x40/0x60/0x80` (the T#s) and `s_load_dwordx4` at
    /// `0xa0/0xb0/0xc0/0xd0/0xe0/0xf0/0x100` (the S#s). `0x100 >> 2 = 64`, so
    /// the last sampler occupies EUD dwords 64..67 — one past Kyty's window.
    /// The usage table declares it at `start_register = SGPRS_MAX + 64 = 96`,
    /// `eud_rel_index` rebases that to 64, and `Spirv::WriteLocalVariables`
    /// refused the whole shader ("extended mapping overflow").
    ///
    /// The second half pins the FLOOR: a table that fits Kyty's 64 keeps a
    /// 64-long mapping and coverage map, so nothing that translates today moves.
    #[test]
    fn extended_mapping_grows_for_a_descriptor_table_deeper_than_kytys_window() {
        use crate::shader::spirv::{eud_covered_map, extended_mapping_len};

        // s_load_dwordx4 s[24:27], s[28:29], 0x100 (NULL soffset).
        const SLOAD_X4_0X100: [u32; 2] = [0xF408_060E, 0xFA00_0100];

        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        shader_parse(
            0,
            &[SLOAD_X4_0X100[0], SLOAD_X4_0X100[1], V_MOV_V0_0, S_ENDPGM],
            &mut code,
            true,
        )
        .expect("parse s_load_dwordx4 at EUD byte 0x100");
        let sload = &code.get_instructions()[0];
        assert_eq!(sload.type_, T::SLoadDwordx4);
        assert_eq!(sload.src[0].register_id, 28, "base pair is s[28:29]");
        assert_eq!(
            sload.src[1].constant.u >> 2,
            64,
            "byte 0x100 is EUD dword 64 — one past Kyty's fixed window"
        );

        let mut input_info = ShaderComputeInputInfo::default();
        input_info.threads_num = [1, 1, 1];
        input_info.bind.push_constant_size = 16;
        input_info.bind.extended.used = true;
        input_info.bind.extended.start_register = 28;
        input_info.bind.samplers.samplers_num = 1;
        // SGPRS_MAX + 64: the "EUD continues the register file" declaration.
        input_info.bind.samplers.start_register[0] = 96;
        input_info.bind.samplers.extended[0] = true;

        assert_eq!(
            extended_mapping_len(&input_info.bind),
            68,
            "the window must span the declared S# at EUD dwords 64..67"
        );
        let covered = eud_covered_map(&input_info.bind);
        assert_eq!(covered.len(), 68, "coverage map matches the mapping length");
        assert!(
            !covered[63],
            "dword 63 belongs to no declared descriptor: {covered:?}"
        );
        assert!(
            covered[64..68].iter().all(|c| *c),
            "EUD dwords 64..67 are the S#'s captured descriptor fields: {covered:?}"
        );

        let source = spirv_generate_source(&code, None, None, Some(&input_info))
            .expect("a descriptor past EUD dword 63 must map, not overflow");
        for reg in 24..28 {
            assert!(
                source.contains(&format!("OpStore %s{reg}")),
                "S# dword must land in s{reg}:\n{source}"
            );
        }
        let words = spirv_run(&source).expect("assemble deep-EUD descriptor load");
        naga_parse_and_validate(&words, "extended_mapping_dword_64");

        // FLOOR: the same shape one descriptor lower (byte 0xf0 => dwords
        // 60..63) still fits Kyty's window, and the window stays exactly 64 —
        // so every shader that translates today keeps an identical mapping.
        let mut shallow = input_info.bind;
        shallow.samplers.start_register[0] = 92;
        assert_eq!(
            extended_mapping_len(&shallow),
            crate::shader::spirv::EXTENDED_MAPPING_DWORDS,
            "a table that fits Kyty's 64 keeps Kyty's 64"
        );
        let covered = eud_covered_map(&shallow);
        assert_eq!(covered.len(), 64);
        assert!(
            covered[60..64].iter().all(|c| *c),
            "EUD dwords 60..63 stay covered under the floor: {covered:?}"
        );
        assert!(!covered[59], "nothing below the S# is covered");
    }
}
