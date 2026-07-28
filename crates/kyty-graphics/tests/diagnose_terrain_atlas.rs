//! DIAGNOSIS-ONLY offline analyzer for the Minecraft terrain-atlas T#
//! resolution failure (docs/diagnosis/minecraft-terrain-atlas-descriptor.md).
//!
//! Replays dumped pixel shaders (`RAEEN_DUMP_SHADERS` `.bin` files) through the
//! SAME in-tree parse the live path uses, then for every sampled MIMG traces
//! the def-use chain of its T# register: which SMEM instruction loaded it,
//! where that load's base pair came from, and what the 0ea66d0 scalar
//! evaluator can prove about each address term.
//!
//! Run with:
//! ```text
//! RAEEN_SHADER_DUMP_DIR=<dir> [RAEEN_DIAGNOSE_FILE=<substr>] \
//!     cargo test -p kyty-graphics --test diagnose_terrain_atlas -- --nocapture
//! ```
//! Without `RAEEN_SHADER_DUMP_DIR` the test is a no-op (CI has no dumps).

use kyty_graphics::hw_regs::{PixelShaderInfo, ShaderRegisters};
use kyty_graphics::shader::analysis::{ShaderMemory, shader_parse_ps};
use kyty_graphics::shader::types::{
    ShaderCode, ShaderInstruction, ShaderInstructionType as T, ShaderOperand, ShaderOperandType,
    smem_offset_operand, smem_register_soffset,
};
use kyty_graphics::shader::{get_binary_info, get_usage_slots, scalar_eval};
use std::borrow::Cow;

struct DumpMem {
    base: u64,
    data: Vec<u32>,
}

impl ShaderMemory for DumpMem {
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

fn smem_width(t: T) -> Option<usize> {
    match t {
        T::SLoadDword => Some(1),
        T::SLoadDwordx2 => Some(2),
        T::SLoadDwordx4 => Some(4),
        T::SLoadDwordx8 => Some(8),
        T::SLoadDwordx16 => Some(16),
        _ => None,
    }
}

const fn sampled_mimg(t: T) -> bool {
    matches!(
        t,
        T::ImageLoad
            | T::ImageSample
            | T::ImageSampleCLz
            | T::ImageSampleLz
            | T::ImageSampleLzO
            | T::ImageGather4Lz
    )
}

fn writes_sgpr(inst: &ShaderInstruction, reg: i32) -> bool {
    let covers = |op: &ShaderOperand| {
        op.type_ == ShaderOperandType::Sgpr
            && reg >= op.register_id
            && reg < op.register_id + op.size.max(1)
    };
    covers(&inst.dst) || covers(&inst.dst2)
}

fn op_str(op: &ShaderOperand) -> String {
    match op.type_ {
        ShaderOperandType::Sgpr => {
            if op.size > 1 {
                format!("s[{}:{}]", op.register_id, op.register_id + op.size - 1)
            } else {
                format!("s{}", op.register_id)
            }
        }
        ShaderOperandType::Vgpr => {
            if op.size > 1 {
                format!("v[{}:{}]", op.register_id, op.register_id + op.size - 1)
            } else {
                format!("v{}", op.register_id)
            }
        }
        ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant => {
            format!("{:#x}", op.constant.u)
        }
        other => format!("{other:?}"),
    }
}

fn inst_str(inst: &ShaderInstruction) -> String {
    let srcs: Vec<String> = inst.src[..inst.src_num.max(0) as usize]
        .iter()
        .map(op_str)
        .collect();
    format!(
        "pc={:#06x} {:?} [{:?}] dst={} src=[{}]",
        inst.pc,
        inst.type_,
        inst.format,
        op_str(&inst.dst),
        srcs.join(", ")
    )
}

/// Print the full producer chain for a scalar register pair, walking backwards.
fn trace_pair(insts: &[ShaderInstruction], upto: usize, lo: i32, depth: usize) {
    if depth > 6 {
        eprintln!("        ... (trace depth cap)");
        return;
    }
    let indent = "    ".repeat(depth + 2);
    let mut found = false;
    for prior in insts[..upto].iter().rev() {
        if writes_sgpr(prior, lo) || writes_sgpr(prior, lo + 1) {
            found = true;
            eprintln!("{indent}producer of s{lo}/s{}: {}", lo + 1, inst_str(prior));
            // If the producer is itself an SMEM load, trace ITS base.
            if smem_width(prior.type_).is_some() && prior.src[0].type_ == ShaderOperandType::Sgpr {
                let at = insts.iter().position(|i| i.pc == prior.pc).unwrap_or(0);
                trace_pair(insts, at, prior.src[0].register_id, depth + 1);
            }
            break; // nearest producer only; deeper writes shown recursively
        }
    }
    if !found {
        eprintln!(
            "{indent}s{lo}/s{} are LIVE-IN (user-data SGPRs, never written in-shader)",
            lo + 1
        );
    }
}

fn analyze(name: &str, code: &ShaderCode, raw: &[u32]) {
    let insts = code.get_instructions();
    eprintln!("\n==== {name}: {} instructions ====", insts.len());

    if let Some(bi) = get_binary_info(raw) {
        eprintln!(
            "  legacy binary-info trailer: is_srt={} srt_used_valid={} extended_usage={} slots={}",
            bi.is_srt,
            bi.is_srt_used_info_valid,
            bi.is_extended_usage_info,
            bi.num_input_usage_slots
        );
        if let Ok(usage) = get_usage_slots(raw) {
            for s in &usage.slots {
                eprintln!(
                    "    usage slot: type={:#x} slot={} start_register={} flags={:#x}",
                    s.type_, s.slot, s.start_register, s.flags
                );
            }
        }
    } else {
        eprintln!(
            "  no legacy binary-info sentinel (Gen5/AGC metadata lives in ShaderMap, not the dump)"
        );
    }

    // Every SMEM scalar load, with the evaluator's verdict on its address terms.
    eprintln!("  -- scalar memory loads --");
    for (at, inst) in insts.iter().enumerate() {
        let Some(width) = smem_width(inst.type_) else {
            continue;
        };
        let base = inst.src[0];
        let imm = smem_offset_operand(inst);
        let soffset = smem_register_soffset(inst);
        eprintln!("    {}", inst_str(inst));
        eprintln!(
            "      width={width} base={} imm_off={:#x} reg_soffset={}",
            op_str(&base),
            imm.constant.u,
            soffset.map_or("none".to_string(), |s| op_str(&s)),
        );
        if base.type_ == ShaderOperandType::Sgpr {
            let base_reg = base.register_id;
            let written_before = insts[..at]
                .iter()
                .any(|p| writes_sgpr(p, base_reg) || writes_sgpr(p, base_reg + 1));
            eprintln!(
                "      base pair written earlier in-shader: {written_before} \
                 (capture pass requires FALSE)"
            );
            trace_pair(insts, at, base_reg, 0);
            // What can the 0ea66d0 evaluator prove about the base pair with no
            // live user data (dump replay has none)?
            match scalar_eval::evaluate_before(insts, at, None, 0) {
                Ok(state) => {
                    let lo = state.sgpr(base_reg);
                    let hi = state.sgpr(base_reg + 1);
                    eprintln!(
                        "      scalar_eval (no user data): base_lo={:?} base_hi={:?}",
                        lo, hi
                    );
                }
                Err(refusal) => {
                    eprintln!("      scalar_eval refused the walk: {refusal:?}");
                }
            }
        }
    }

    // Every sampled MIMG and the def chain of its T#.
    eprintln!("  -- sampled image ops --");
    for (at, inst) in insts.iter().enumerate() {
        if !sampled_mimg(inst.type_) {
            continue;
        }
        eprintln!("    {}", inst_str(inst));
        let t_op = inst.src[1];
        if t_op.type_ != ShaderOperandType::Sgpr {
            continue;
        }
        let t_reg = t_op.register_id;
        // Nearest SMEM (or any) writer of the T# register before this MIMG.
        let mut writer = None;
        for (i, prior) in insts[..at].iter().enumerate().rev() {
            if writes_sgpr(prior, t_reg) {
                writer = Some((i, prior));
                break;
            }
        }
        match writer {
            Some((wi, w)) => {
                eprintln!("      T# s{t_reg} defined by: {}", inst_str(w));
                if w.src[0].type_ == ShaderOperandType::Sgpr {
                    trace_pair(insts, wi, w.src[0].register_id, 0);
                }
            }
            None => eprintln!("      T# s{t_reg} is LIVE-IN user data (no in-shader writer)"),
        }
    }
}

#[test]
fn diagnose_dumped_ps_atlas_chains() {
    let Ok(dir) = std::env::var("RAEEN_SHADER_DUMP_DIR") else {
        eprintln!("RAEEN_SHADER_DUMP_DIR not set — nothing to diagnose");
        return;
    };
    let filter = std::env::var("RAEEN_DIAGNOSE_FILE").unwrap_or_default();
    let mut dumps: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read dump dir {dir}: {e}"))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "bin")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("ps_"))
                && (filter.is_empty()
                    || p.file_name()
                        .is_some_and(|n| n.to_string_lossy().contains(&filter)))
        })
        .collect();
    dumps.sort();
    assert!(!dumps.is_empty(), "no matching ps_*.bin dumps in {dir}");

    for path in &dumps {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = std::fs::read(path).expect("readable dump");
        let data: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mem = DumpMem {
            base: 0x10000,
            data,
        };

        let attempt = |next_gen: bool| {
            let sh_regs = ShaderRegisters::default();
            let mut ps = PixelShaderInfo::default();
            ps.ps_regs.data_addr = mem.base;
            shader_parse_ps(&ps, &sh_regs, &mem, next_gen).map_err(|e| e.to_string())
        };
        match attempt(true).or_else(|e| attempt(false).map_err(|e2| format!("{e}; {e2}"))) {
            Ok(code) => analyze(&name, &code, &mem.data),
            Err(e) => eprintln!("\n==== {name}: PARSE FAILED: {e} ===="),
        }
    }
}
