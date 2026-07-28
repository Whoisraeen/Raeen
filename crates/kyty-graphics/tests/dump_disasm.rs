//! Offline full-disassembly printer for dumped shaders (any stage).
//!
//! Complements `diagnose_terrain_atlas` (which traces T#/S# chains): this one
//! prints EVERY decoded instruction of a dump so VS attribute fetches, param
//! exports, and PS interpolant reads can be read end-to-end.
//!
//! Run with:
//! ```text
//! RAEEN_SHADER_DUMP_DIR=<dir> [RAEEN_DIAGNOSE_FILE=<substr>] \
//!     cargo test -p kyty-graphics --test dump_disasm -- --nocapture
//! ```
//! Without `RAEEN_SHADER_DUMP_DIR` the test is a no-op (CI has no dumps).

use kyty_graphics::hw_regs::{
    ComputeShaderInfo, PixelShaderInfo, ShaderRegisters, VertexShaderInfo,
};
use kyty_graphics::shader::analysis::{
    ShaderMemory, shader_parse_cs, shader_parse_ps, shader_parse_vs,
};
use kyty_graphics::shader::types::{
    ShaderCode, ShaderInstruction, ShaderOperand, ShaderOperandType,
};
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
        ShaderOperandType::FloatInlineConstant => format!("{}", op.constant.f()),
        other => format!("{other:?}"),
    }
}

fn inst_str(inst: &ShaderInstruction) -> String {
    let srcs: Vec<String> = inst.src[..inst.src_num.max(0) as usize]
        .iter()
        .map(op_str)
        .collect();
    let dst2 = if inst.dst2.type_ == ShaderOperandType::Unknown {
        String::new()
    } else {
        format!(" dst2={}", op_str(&inst.dst2))
    };
    format!(
        "pc={:#06x} {:?} [{:?}] dst={}{dst2} src=[{}] en={:#x}",
        inst.pc,
        inst.type_,
        inst.format,
        op_str(&inst.dst),
        srcs.join(", "),
        inst.export_enable,
    )
}

fn parse_dump(stage: &str, mem: &DumpMem) -> Result<ShaderCode, String> {
    let attempt = |next_gen: bool| -> Result<ShaderCode, String> {
        let sh_regs = ShaderRegisters::default();
        match stage {
            "vs" => {
                let mut vs = VertexShaderInfo::default();
                vs.es_regs.data_addr = mem.base;
                vs.gs_regs.chksum = 1;
                shader_parse_vs(&vs, &sh_regs, mem, next_gen).map_err(|e| e.to_string())
            }
            "ps" => {
                let mut ps = PixelShaderInfo::default();
                ps.ps_regs.data_addr = mem.base;
                shader_parse_ps(&ps, &sh_regs, mem, next_gen).map_err(|e| e.to_string())
            }
            "cs" => {
                let mut cs = ComputeShaderInfo::default();
                cs.cs_regs.data_addr = mem.base;
                shader_parse_cs(&cs, &sh_regs, mem, next_gen).map_err(|e| e.to_string())
            }
            other => Err(format!("unknown stage prefix {other:?}")),
        }
    };
    attempt(true).or_else(|e_next| {
        attempt(false).map_err(|e_legacy| format!("next_gen: {e_next}; legacy: {e_legacy}"))
    })
}

#[test]
fn disassemble_dumps() {
    let Ok(dir) = std::env::var("RAEEN_SHADER_DUMP_DIR") else {
        eprintln!("RAEEN_SHADER_DUMP_DIR not set — nothing to disassemble");
        return;
    };
    let filter = std::env::var("RAEEN_DIAGNOSE_FILE").unwrap_or_default();
    let mut dumps: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read dump dir {dir}: {e}"))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "bin")
                && (filter.is_empty()
                    || p.file_name()
                        .is_some_and(|n| n.to_string_lossy().contains(&filter)))
        })
        .collect();
    dumps.sort();
    assert!(!dumps.is_empty(), "no matching .bin dumps in {dir}");

    for path in &dumps {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let stage = name.split('_').next().unwrap_or("").to_owned();
        let bytes = std::fs::read(path).expect("readable dump");
        let data: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mem = DumpMem {
            base: 0x10000,
            data,
        };
        match parse_dump(&stage, &mem) {
            Ok(code) => {
                let insts = code.get_instructions();
                eprintln!("\n==== {name}: {} instructions ====", insts.len());
                for inst in insts {
                    eprintln!("  {}", inst_str(inst));
                }
            }
            Err(e) => eprintln!("\n==== {name}: PARSE FAILED: {e} ===="),
        }
    }
}
