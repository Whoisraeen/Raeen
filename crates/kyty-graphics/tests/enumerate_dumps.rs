//! Offline coverage enumerator for real dumped shaders.
//!
//! `XPS5X_DUMP_SHADERS` (see `xps5x-gpu::shader_fetch`) writes every distinct
//! shader a title binds to `<stage>_<addr>_<len>.bin`. This test replays those
//! dumps through parse and then classifies EVERY instruction against the
//! recompiler table in one pass — instead of discovering missing recompilers
//! one per run→rebuild cycle.
//!
//! Run with:
//! ```text
//! XPS5X_SHADER_DUMP_DIR=path/to/dumps cargo test -p kyty-graphics \
//!     --test enumerate_dumps -- --nocapture
//! ```
//! Without the env var the test is a no-op (CI has no dumps).

use kyty_graphics::hw_regs::{ComputeShaderInfo, PixelShaderInfo, ShaderRegisters, VertexShaderInfo};
use kyty_graphics::shader::analysis::{
    ShaderMemory, shader_parse_cs, shader_parse_ps, shader_parse_vs,
};
use kyty_graphics::shader::recompile::{RecompileFn, recomp_func};
use kyty_graphics::shader::types::ShaderCode;
use std::borrow::Cow;
use std::collections::BTreeMap;

/// A dumped shader replayed at a synthetic guest address.
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

fn parse_dump(stage: &str, mem: &DumpMem) -> Result<ShaderCode, String> {
    // Mirror xps5x-gpu's attempt_generations: next-gen first, legacy second,
    // both reasons on failure.
    let attempt = |next_gen: bool| -> Result<ShaderCode, String> {
        let sh_regs = ShaderRegisters::default();
        match stage {
            "vs" => {
                // Real PS5 titles bind the VS through the ES slot with a GS
                // checksum (`gs_instead_of_vs` — see xps5x-gpu translate_vs);
                // replay the dump the same way.
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
fn enumerate_dumped_shader_coverage() {
    let Ok(dir) = std::env::var("XPS5X_SHADER_DUMP_DIR") else {
        eprintln!("XPS5X_SHADER_DUMP_DIR not set — nothing to enumerate");
        return;
    };
    let mut dumps: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read dump dir {dir}: {e}"))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    dumps.sort();
    assert!(!dumps.is_empty(), "no .bin dumps in {dir}");

    // (kyty_func, type/format) -> instruction texts, aggregated across dumps.
    let mut not_wired: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut no_entry: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut parse_failures = Vec::new();

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

        let code = match parse_dump(&stage, &mem) {
            Ok(code) => code,
            Err(e) => {
                parse_failures.push(format!("{name}: {e}"));
                continue;
            }
        };

        let mut ported = 0usize;
        for inst in code.get_instructions() {
            let text = format!("{:?} [{:?}]", inst.type_, inst.format);
            match recomp_func(inst.type_, inst.format) {
                Some(f) => match f.func {
                    RecompileFn::Func(_) => ported += 1,
                    RecompileFn::NotImplemented { kyty_func, line } => {
                        not_wired
                            .entry(format!("{kyty_func} (ShaderSpirv.cpp L{line})"))
                            .or_default()
                            .push(format!("{name}: {text}"));
                    }
                },
                None => {
                    no_entry.entry(text.clone()).or_default().push(name.clone());
                }
            }
        }
        eprintln!(
            "{name}: {} instruction(s), {ported} ported",
            code.get_instructions().len()
        );
    }

    eprintln!(
        "\n== recompilers needed but NOT IMPLEMENTED ({}) ==",
        not_wired.len()
    );
    for (func, uses) in &not_wired {
        eprintln!("  {func}  — {} use(s)", uses.len());
        if let Some(first) = uses.first() {
            eprintln!("      e.g. {first}");
        }
    }
    eprintln!(
        "\n== instructions with NO TABLE ENTRY ({}) ==",
        no_entry.len()
    );
    for (text, files) in &no_entry {
        eprintln!("  {text}  — in {:?}", files);
    }
    eprintln!("\n== parse failures ({}) ==", parse_failures.len());
    for f in &parse_failures {
        eprintln!("  {f}");
    }
}
