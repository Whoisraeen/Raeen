//! ShaderMemory Phase 2: fetch a title's real shader code out of guest
//! memory, recompile it to SPIR-V through `kyty-graphics`, and cache the
//! result — with honest, named degradation when translation fails.
//!
//! # How the byte length is determined
//!
//! There is no length register: Kyty's `shader_parse` walks the instruction
//! stream until `s_endpgm` and treats running off the end of the buffer as
//! [`ShaderParseError::Truncated`]. This module mirrors that: it reads the
//! guest window in bounded 4 KiB chunks (capped at 256 KiB), hands the window
//! to the parser, and grows the window only when the parser itself says it
//! ran out of bytes. A window that stops growing because the guest mapping
//! ends is a named failure, never a fault.
//!
//! # Caching (positive and negative)
//!
//! Titles re-bind the same shaders every frame. Code is first parsed/analyzed,
//! then translated modules are cached by `(stage, guest_addr, full fetched
//! window digest, analyzed stage ABI)`. The analyzed ABI includes descriptor
//! types, exact formats, and embedded constants, so reusing one code address
//! with different code or bindings cannot return stale SPIR-V. The cache is
//! FIFO-bounded; analysis failures are never cached before binding identity is
//! known.
//!
//! # Forensics
//!
//! PS5 titles ship RDNA2 ISA while the ported parser speaks GCN, so most real
//! shaders are *expected* to fail translation for now. When the environment
//! variable `RAEEN_DUMP_SHADERS` names a directory, every distinct fetched
//! shader's raw bytes are written there once —
//! `<stage>_<guestaddr>_<len>.bin` — succeeding **even when translation
//! fails**; the dumps are how the GCN→RDNA2 gap gets studied.

use kyty_graphics::hw_regs::{
    ComputeShaderInfo, PixelShaderInfo, ShaderRegisters, VertexShaderInfo,
};
use kyty_graphics::shader::analysis::{
    SHADER_BINARY_INFO_SENTINEL, ShaderAnalysisError, ShaderMap, ShaderMemory,
    shader_get_input_info_cs, shader_get_input_info_ps, shader_get_input_info_vs, shader_parse_cs,
    shader_parse_ps, shader_parse_vs,
};
use kyty_graphics::shader::parse::ShaderParseError;
use kyty_graphics::shader::recompile::{
    shader_recompile_cs, shader_recompile_ps, shader_recompile_vs,
};
use kyty_graphics::shader::resources::{
    ShaderComputeInputInfo, ShaderMappedData, ShaderPixelInputInfo, ShaderVertexInputInfo,
};
use kyty_graphics::shader::types::ShaderCode;
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// One fetch step: 4 KiB.
const CHUNK_DWORDS: usize = 1024;
/// Fetch cap: 256 KiB. A shader bigger than this is a mis-decode.
const MAX_WINDOW_DWORDS: usize = 0x1_0000;
/// Hard cap on translated modules and binding-aware failures.
const MAX_CACHE_ENTRIES: usize = 256;
/// `s_endpgm` — identical encoding on GCN and RDNA2 (SOPP op 1).
const S_ENDPGM: u32 = 0xBF81_0000;

/// Which pipeline stage a fetched shader binds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Vs,
    Ps,
    Cs,
}

impl Stage {
    const fn as_str(self) -> &'static str {
        match self {
            Stage::Vs => "vs",
            Stage::Ps => "ps",
            Stage::Cs => "cs",
        }
    }
}

/// Code identity after the bounded fetch/analyze loop. The digest covers the
/// complete fetched window so a title rewriting bytes beyond the first four
/// dwords cannot reuse stale SPIR-V.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct CodeKey {
    stage: Stage,
    addr: u64,
    fetched_dwords: u32,
    digest: u64,
}

impl CodeKey {
    fn new(stage: Stage, addr: u64, fetched: &[u32]) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        fetched.hash(&mut hasher);
        Self {
            stage,
            addr,
            fetched_dwords: u32::try_from(fetched.len()).unwrap_or(u32::MAX),
            digest: hasher.finish(),
        }
    }
}

/// Active translated-module key: code identity plus the complete analyzed
/// stage metadata that can shape SPIR-V or its host binding ABI.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    code: CodeKey,
    binding: Box<str>,
}

/// A translated shader plus the stage resource ABI recovered during analysis.
///
/// Both fields are retained because Vulkan pipeline creation and descriptor
/// binding need the same metadata that shaped the generated SPIR-V. The
/// irrelevant stage field remains `Default`, which keeps cache entries a
/// single concrete type without erasing either ABI.
#[derive(Clone, Debug)]
pub struct TranslatedShader {
    pub spirv: Arc<Vec<u32>>,
    pub vs_info: ShaderVertexInputInfo,
    pub ps_info: ShaderPixelInputInfo,
    pub cs_info: ShaderComputeInputInfo,
}

/// Parse/analysis result held just long enough to consult the binding-aware
/// module cache before running the expensive SPIR-V recompiler.
enum PreparedShader {
    Vs {
        code: ShaderCode,
        info: Box<ShaderVertexInputInfo>,
    },
    Ps {
        code: ShaderCode,
        vs_info: Box<ShaderVertexInputInfo>,
        info: Box<ShaderPixelInputInfo>,
    },
    Cs {
        code: ShaderCode,
        info: Box<ShaderComputeInputInfo>,
    },
}

impl PreparedShader {
    fn binding_identity(&self) -> Box<str> {
        match self {
            Self::Vs { info, .. } => format!("{info:?}").into_boxed_str(),
            Self::Ps { vs_info, info, .. } => {
                format!("vs={vs_info:?}; ps={info:?}").into_boxed_str()
            }
            Self::Cs { info, .. } => format!("{info:?}").into_boxed_str(),
        }
    }

    fn recompile(&self) -> Result<Vec<u32>, AttemptError> {
        match self {
            Self::Vs { code, info } => {
                let spirv = shader_recompile_vs(code, info)
                    .map_err(|e| AttemptError::named(format!("shader_recompile_vs: {e}")))?;
                Ok(spirv)
            }
            Self::Ps { code, info, .. } => {
                let spirv = shader_recompile_ps(code, info)
                    .map_err(|e| AttemptError::named(format!("shader_recompile_ps: {e}")))?;
                Ok(spirv)
            }
            Self::Cs { code, info } => {
                let spirv = shader_recompile_cs(code, info)
                    .map_err(|e| AttemptError::named(format!("shader_recompile_cs: {e}")))?;
                Ok(spirv)
            }
        }
    }

    /// Pair a cached module with this bind's freshly analyzed metadata. Cache
    /// values deliberately contain no resource bases or writeback state.
    fn into_translated(self, spirv: Arc<Vec<u32>>) -> TranslatedShader {
        match self {
            Self::Vs { info, .. } => TranslatedShader {
                spirv,
                vs_info: *info,
                ps_info: ShaderPixelInputInfo::default(),
                cs_info: ShaderComputeInputInfo::default(),
            },
            Self::Ps { vs_info, info, .. } => TranslatedShader {
                spirv,
                vs_info: *vs_info,
                ps_info: *info,
                cs_info: ShaderComputeInputInfo::default(),
            },
            Self::Cs { info, .. } => TranslatedShader {
                spirv,
                vs_info: ShaderVertexInputInfo::default(),
                ps_info: ShaderPixelInputInfo::default(),
                cs_info: *info,
            },
        }
    }
}

/// Counters for the measurement report.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderCacheStats {
    /// Shader variants fetched for translation/analysis.
    pub distinct_fetched: u64,
    /// Distinct shaders that translated to SPIR-V.
    pub translated_ok: u64,
    /// Translation/analysis attempts that failed.
    pub translate_failed: u64,
    /// Binding-aware module/error hits after fresh analysis.
    pub hits: u64,
}

/// Fetch + translate + cache for guest shader code.
pub struct ShaderTranslateCache {
    entries: HashMap<CacheKey, Result<Arc<Vec<u32>>, Arc<str>>>,
    insertion_order: VecDeque<CacheKey>,
    shader_map: ShaderMap,
    dump_dir: Option<PathBuf>,
    stats: ShaderCacheStats,
}

impl Default for ShaderTranslateCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderTranslateCache {
    /// Cache with forensic dumps controlled by `RAEEN_DUMP_SHADERS`.
    #[must_use]
    pub fn new() -> Self {
        let dump_dir = std::env::var("RAEEN_DUMP_SHADERS")
            .ok()
            .filter(|d| !d.is_empty())
            .map(PathBuf::from);
        Self::with_dump_dir(dump_dir)
    }

    /// Cache with an explicit dump directory (tests; `None` disables dumps).
    #[must_use]
    pub fn with_dump_dir(dump_dir: Option<PathBuf>) -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            shader_map: ShaderMap::new(),
            dump_dir,
            stats: ShaderCacheStats::default(),
        }
    }

    #[must_use]
    pub fn stats(&self) -> ShaderCacheStats {
        self.stats
    }

    /// Register metadata relocated by `sceAgcCreateShader` for later
    /// next-generation resource analysis.
    pub fn map_shader_metadata(&mut self, addr: u64, data: ShaderMappedData) {
        self.shader_map.map_user_data(addr, data);
        // A create call can replace the analyzed ABI at this address. Remove
        // prior binding-aware modules eagerly; analysis failures are not cached.
        self.entries.retain(|key, _| key.code.addr != addr);
        self.insertion_order.retain(|key| key.code.addr != addr);
    }

    fn insert_entry(&mut self, key: CacheKey, value: Result<Arc<Vec<u32>>, Arc<str>>) {
        while self.entries.len() >= MAX_CACHE_ENTRIES {
            let Some(oldest) = self.insertion_order.pop_front() else {
                self.entries.clear();
                break;
            };
            self.entries.remove(&oldest);
        }
        self.insertion_order.push_back(key.clone());
        self.entries.insert(key, value);
    }

    /// Fetch + translate the bound vertex-stage shader.
    ///
    /// # Errors
    ///
    /// A named reason (bad address, unreadable memory, parse/recompile
    /// failure). Post-analysis translation failures are binding-aware cached;
    /// analysis failures are retried because descriptors/EUD can change.
    pub fn translate_vs(
        &mut self,
        vs: &VertexShaderInfo,
        sh_regs: &ShaderRegisters,
    ) -> Result<TranslatedShader, Arc<str>> {
        // Mirror shader_parse_vs's address selection (gs-instead-of-vs).
        let gs_instead_of_vs = vs.vs_regs.data_addr == 0
            && vs.gs_regs.data_addr == 0
            && vs.es_regs.data_addr != 0
            && vs.gs_regs.chksum != 0;
        let addr = if gs_instead_of_vs {
            vs.es_regs.data_addr
        } else {
            vs.vs_regs.data_addr
        };
        let shader_map = self.shader_map.clone();
        let vs = *vs;
        let sh_regs = *sh_regs;
        self.translate(Stage::Vs, addr, move |mem| {
            attempt_generations(|next_gen| {
                let code = shader_parse_vs(&vs, &sh_regs, mem, next_gen)
                    .map_err(|e| AttemptError::from_analysis("shader_parse_vs", &e))?;
                let mut vs_info = ShaderVertexInputInfo::default();
                shader_get_input_info_vs(&vs, &sh_regs, mem, &shader_map, next_gen, &mut vs_info)
                    .map_err(|e| AttemptError::from_analysis("shader_get_input_info_vs", &e))?;
                // Beyond Kyty: capture PC-relative embedded-constant scalar
                // loads (the shader reading its own baked constant table) so the
                // recompiler materializes them as SPIR-V constants instead of
                // refusing the non-EUD base. Measured on ASTRO.BOT vertex
                // shaders (`s_getpc_b64` + `s_add_u32` + `s_load_dwordx8`).
                kyty_graphics::shader::shader_detect_embedded_constant_loads(
                    &code,
                    mem,
                    &mut vs_info.bind,
                );
                // Beyond Kyty: capture `offen` buffer loads through an
                // in-shader-constructed V# that points at the shader's own
                // embedded vertex data (the ASTRO.BOT full-screen-triangle VS),
                // so the recompiler serves them from the baked window instead of
                // refusing an unbound descriptor.
                kyty_graphics::shader::shader_detect_embedded_buffer_fetch(
                    &code,
                    mem,
                    &mut vs_info.bind,
                );
                Ok(PreparedShader::Vs {
                    code,
                    info: Box::new(vs_info),
                })
            })
        })
    }

    /// Fetch + translate the bound pixel shader. `vs_info` is the vertex
    /// stage's input info (defaulted when the VS was embedded).
    ///
    /// # Errors
    ///
    /// As [`Self::translate_vs`].
    pub fn translate_ps(
        &mut self,
        ps: &PixelShaderInfo,
        sh_regs: &ShaderRegisters,
        vs_info: &ShaderVertexInputInfo,
    ) -> Result<TranslatedShader, Arc<str>> {
        let addr = ps.ps_regs.data_addr;
        let shader_map = self.shader_map.clone();
        let ps = *ps;
        let sh_regs = *sh_regs;
        let vs_info = *vs_info;
        self.translate(Stage::Ps, addr, move |mem| {
            attempt_generations(|next_gen| {
                let code = shader_parse_ps(&ps, &sh_regs, mem, next_gen)
                    .map_err(|e| AttemptError::from_analysis("shader_parse_ps", &e))?;
                let mut ps_info = ShaderPixelInputInfo::default();
                shader_get_input_info_ps(
                    &ps,
                    &sh_regs,
                    &vs_info,
                    mem,
                    &shader_map,
                    next_gen,
                    &mut ps_info,
                )
                .map_err(|e| AttemptError::from_analysis("shader_get_input_info_ps", &e))?;
                // PC-relative scalar constant tables are stage-agnostic. VS
                // and CS already run this capture; omitting it here left PS
                // `s_load_dwordx8` instructions to the EUD-only fallback,
                // which correctly refused their non-EUD base register.
                kyty_graphics::shader::shader_detect_embedded_constant_loads(
                    &code,
                    mem,
                    &mut ps_info.bind,
                );
                // SharpEmu port (see `translate_cs`): default nearest/wrap S#
                // for a PS that samples with zero captured samplers.
                kyty_graphics::shader::shader_synthesize_default_sampler(&code, &mut ps_info.bind);
                Ok(PreparedShader::Ps {
                    code,
                    vs_info: Box::new(vs_info),
                    info: Box::new(ps_info),
                })
            })
        })
    }

    /// Fetch + translate the bound compute shader.
    pub fn translate_cs(
        &mut self,
        cs: &ComputeShaderInfo,
        sh_regs: &ShaderRegisters,
    ) -> Result<TranslatedShader, Arc<str>> {
        let addr = cs.cs_regs.data_addr;
        let shader_map = self.shader_map.clone();
        let cs = *cs;
        let sh_regs = *sh_regs;
        self.translate(Stage::Cs, addr, move |mem| {
            attempt_generations(|next_gen| {
                let code = shader_parse_cs(&cs, &sh_regs, mem, next_gen)
                    .map_err(|e| AttemptError::from_analysis("shader_parse_cs", &e))?;
                let mut cs_info = ShaderComputeInputInfo::default();
                shader_get_input_info_cs(&cs, &sh_regs, mem, &shader_map, next_gen, &mut cs_info)
                    .map_err(|e| AttemptError::from_analysis("shader_get_input_info_cs", &e))?;
                // SharpEmu port: a CS that SAMPLES textures with zero captured
                // S#s gets one synthesized all-zero (nearest/wrap) S# per
                // sampler operand register instead of a whole-shader refusal;
                // the Vulkan layer then binds its cached default sampler.
                kyty_graphics::shader::shader_synthesize_default_sampler(&code, &mut cs_info.bind);
                // Beyond Kyty: a CS that appends/consumes through the GDS
                // counter without a captured GDS descriptor gets one synthesized
                // so `%gds` is declared and bound (measured on ASTRO.BOT
                // tiled-lighting's light-list append counter).
                kyty_graphics::shader::shader_synthesize_gds_pointer(&code, &mut cs_info.bind);
                // SharpEmu port: s_loads of EUD dwords no captured descriptor
                // covers become clamped reads of a dispatch-time guest-memory
                // window (`%eud_raw`) instead of named refusals. The detected
                // metadata rides in `cs_info.bind.eud_raw`, which
                // `prepare_stage_binding` uses to snapshot + bind the window.
                kyty_graphics::shader::shader_detect_eud_raw_window(&code, &mut cs_info.bind);
                // Beyond Kyty: the same in-shader embedded-data captures the
                // vertex path uses — a compute shader can equally build a
                // PC-relative scalar-load base or an in-shader V# pointing at
                // its own baked constants (measured in ASTRO.BOT's tiled-lighting
                // compute). Mirror `translate_vs`.
                kyty_graphics::shader::shader_detect_embedded_constant_loads(
                    &code,
                    mem,
                    &mut cs_info.bind,
                );
                kyty_graphics::shader::shader_detect_embedded_buffer_fetch(
                    &code,
                    mem,
                    &mut cs_info.bind,
                );
                Ok(PreparedShader::Cs {
                    code,
                    info: Box::new(cs_info),
                })
            })
        })
    }

    /// Shared fetch → (grow → translate) → cache path.
    fn translate(
        &mut self,
        stage: Stage,
        addr: u64,
        run: impl Fn(&WindowMem) -> Result<PreparedShader, AttemptError>,
    ) -> Result<TranslatedShader, Arc<str>> {
        if addr == 0 || !addr.is_multiple_of(4) {
            // Unkeyable (no head bytes to read) — not cached, but the command
            // processor's draw path rate-limits per draw batch upstream.
            return Err(Arc::from(format!(
                "{} shader bind address {addr:#x} is null or unaligned",
                stage.as_str()
            )));
        }
        let Some(head) = crate::guest_mem::read_dwords_checked(addr, 4) else {
            return Err(Arc::from(format!(
                "{} shader code at {addr:#x} is not readable guest memory",
                stage.as_str()
            )));
        };
        // Analyze on every bind before the positive-cache lookup. Descriptor
        // type/format and embedded metadata can change while code bytes stay
        // identical, and those fields shape generated SPIR-V.
        let mut window = WindowMem {
            base: addr,
            data: head,
        };
        let mut want = CHUNK_DWORDS;
        let prepared = loop {
            let grew = window.grow_to(want);
            match run(&window) {
                Ok(prepared) => break Ok(prepared),
                Err(e) if e.truncated && grew && want < MAX_WINDOW_DWORDS => {
                    // The parser ran off the end and more guest memory may
                    // exist — read another bounded slice and retry.
                    want = (want * 2).min(MAX_WINDOW_DWORDS);
                }
                Err(e) => break Err(e),
            }
        };

        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(e) => {
                self.stats.distinct_fetched += 1;
                self.stats.translate_failed += 1;
                self.dump_shader(stage, addr, &window.data);
                let reason: Arc<str> = Arc::from(format!(
                    "{} shader at {addr:#x} ({} bytes fetched): {}",
                    stage.as_str(),
                    window.data.len() * 4,
                    e.msg
                ));
                warn!(
                    stage = stage.as_str(),
                    addr = format_args!("{addr:#x}"),
                    reason = %reason,
                    "guest shader analysis failed — draws binding it will be skipped"
                );
                return Err(reason);
            }
        };

        let code_key = CodeKey::new(stage, addr, &window.data);
        let key = CacheKey {
            code: code_key,
            binding: prepared.binding_identity(),
        };
        if let Some(cached) = self.entries.get(&key).cloned() {
            self.stats.hits += 1;
            return match cached {
                Ok(spirv) => Ok(prepared.into_translated(spirv)),
                Err(reason) => Err(reason),
            };
        }

        self.stats.distinct_fetched += 1;
        self.dump_shader(stage, addr, &window.data);

        let result = prepared.recompile();

        // Validity gate: an invalid module passes vkCreateShaderModule but
        // dispatching it is UB (measured: AMD driver access violation that
        // kills the process). Refuse it here so it becomes a named, cached
        // translate failure instead of ever reaching the driver.
        let result = match result {
            Ok(spirv) if crate::spirv_gate::gate_enabled() => {
                match crate::spirv_gate::validate_spirv(&spirv) {
                    Ok(()) => Ok(spirv),
                    Err(reason) => Err(AttemptError {
                        msg: format!(
                            "translated ({} SPIR-V words) but the module is invalid — {reason}",
                            spirv.len()
                        ),
                        truncated: false,
                    }),
                }
            }
            other => other,
        };

        match result {
            Ok(spirv) => {
                self.stats.translated_ok += 1;
                let spirv = Arc::new(spirv);
                info!(
                    stage = stage.as_str(),
                    addr = format_args!("{addr:#x}"),
                    spirv_words = spirv.len(),
                    "guest shader fetched and translated to SPIR-V"
                );
                self.dump_spirv(stage, addr, &spirv);
                self.insert_entry(key, Ok(spirv.clone()));
                Ok(prepared.into_translated(spirv))
            }
            Err(e) => {
                self.stats.translate_failed += 1;
                let reason: Arc<str> = Arc::from(format!(
                    "{} shader at {addr:#x} ({} bytes fetched): {}",
                    stage.as_str(),
                    window.data.len() * 4,
                    e.msg
                ));
                // The one loud line per distinct failing shader. Re-binds hit
                // the negative cache and stay quiet.
                warn!(
                    stage = stage.as_str(),
                    addr = format_args!("{addr:#x}"),
                    reason = %reason,
                    "guest shader translation failed — draws binding it will be skipped"
                );
                self.insert_entry(key, Err(reason.clone()));
                Err(reason)
            }
        }
    }

    /// Forensics: write a distinct shader's raw bytes once. Never a reason to
    /// fail the draw path — errors are logged and dropped.
    fn dump_shader(&self, stage: Stage, addr: u64, data: &[u32]) {
        let Some(dir) = &self.dump_dir else {
            return;
        };
        let len_bytes = dump_len_heuristic(data) * 4;
        let path = dir.join(format!("{}_{addr:x}_{len_bytes}.bin", stage.as_str()));
        let bytes: Vec<u8> = data[..len_bytes / 4]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        match std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, &bytes)) {
            Ok(()) => debug!(path = %path.display(), "dumped fetched shader"),
            Err(e) => warn!(error = %e, path = %path.display(), "shader dump failed"),
        }
    }

    /// Forensics: write a successfully-translated shader's SPIR-V once, next
    /// to its raw `.bin`. Enables the in-tree coverage-bisect harness
    /// (`tests/coverage_bisect.rs`) to replay a TITLE's actual translated
    /// VS+PS against a known-covering draw without a title run — the one
    /// component that harness cannot fabricate. Same policy as `dump_shader`:
    /// never a reason to fail the draw path.
    fn dump_spirv(&self, stage: Stage, addr: u64, spirv: &[u32]) {
        let Some(dir) = &self.dump_dir else {
            return;
        };
        let path = dir.join(format!("{}_{addr:x}.spv", stage.as_str()));
        let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        match std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, &bytes)) {
            Ok(()) => debug!(path = %path.display(), "dumped translated SPIR-V"),
            Err(e) => warn!(error = %e, path = %path.display(), "SPIR-V dump failed"),
        }
    }
}

/// How many leading dwords of the fetched window to dump.
///
/// Heuristic, for forensics only (translation never depends on it):
/// - a PS4-style blob starting with the `0xBEEB03FF` sentinel declares its
///   own size (body + usage masks + 7-dword `OrbShdr` trailer);
/// - otherwise, code through the first `s_endpgm` plus a 16-dword margin
///   (possible trailer) — the encoding is shared by GCN and RDNA2;
/// - otherwise (no end found — likely not code at all) the first 4 KiB.
fn dump_len_heuristic(data: &[u32]) -> usize {
    if data.len() >= 2 && data[0] == SHADER_BINARY_INFO_SENTINEL {
        let blob = (data[1] as usize + 1) * 2 + 7;
        if blob <= data.len() {
            return blob;
        }
    }
    if let Some(pos) = data.iter().position(|&w| w == S_ENDPGM) {
        return (pos + 1 + 16).min(data.len());
    }
    data.len().min(CHUNK_DWORDS)
}

/// A bounded window of guest memory serving Kyty's [`ShaderMemory`] reads.
/// Addresses outside the window (e.g. a fetch-shader pointer elsewhere in
/// guest memory) come back `None` → a named `BadAddress` upstream.
struct WindowMem {
    base: u64,
    data: Vec<u32>,
}

impl WindowMem {
    /// Extend the window to `want` dwords in 4 KiB steps. Returns whether any
    /// growth happened (readable memory may end before `want`).
    fn grow_to(&mut self, want: usize) -> bool {
        let mut grew = false;
        while self.data.len() < want {
            let at = self.base + (self.data.len() as u64) * 4;
            let step = CHUNK_DWORDS.min(want - self.data.len()) as u32;
            let Some(chunk) = crate::guest_mem::read_dwords_checked(at, step) else {
                break;
            };
            self.data.extend(chunk);
            grew = true;
        }
        grew
    }
}

impl ShaderMemory for WindowMem {
    fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
        let end = self.base + self.data.len() as u64 * 4;
        if addr >= self.base && addr < end && (addr - self.base).is_multiple_of(4) {
            return Some(Cow::Borrowed(
                &self.data[((addr - self.base) / 4) as usize..],
            ));
        }
        crate::guest_mem::read_dwords_checked(addr, CHUNK_DWORDS as u32).map(Cow::Owned)
    }
}

/// A translation attempt failure, with the bit the grow loop needs: did the
/// parser run off the end of the window?
struct AttemptError {
    truncated: bool,
    msg: String,
}

impl AttemptError {
    fn named(msg: String) -> Self {
        Self {
            truncated: false,
            msg,
        }
    }

    fn from_analysis(what: &str, e: &ShaderAnalysisError) -> Self {
        Self {
            truncated: matches!(
                e,
                ShaderAnalysisError::Parse(ShaderParseError::Truncated { .. })
            ),
            msg: format!("{what}: {e}"),
        }
    }
}

/// PS5 titles are next-gen (`chksum` registers, no `OrbShdr` trailer), but
/// nothing in the DCB says so explicitly; run the **whole** parse → input
/// info → recompile pipeline as next-gen first and fall back to the legacy
/// trailer pipeline, reporting **both** named reasons on failure.
fn attempt_generations<T>(
    run: impl Fn(bool) -> Result<T, AttemptError>,
) -> Result<T, AttemptError> {
    let e_next = match run(true) {
        Ok(v) => return Ok(v),
        Err(e) => e,
    };
    let e_legacy = match run(false) {
        Ok(v) => return Ok(v),
        Err(e) => e,
    };
    Err(AttemptError {
        truncated: e_next.truncated || e_legacy.truncated,
        msg: format!("next_gen: {}; legacy: {}", e_next.msg, e_legacy.msg),
    })
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use kyty_graphics::hw_regs::PsStageRegisters;

    /// Minimal GCN vertex shader (the `kyty-graphics` recompile fixture):
    /// v_mov v0, 1.0; v_mov v1, 0; v_mul v2, v0, v1; exp pos0; exp param0;
    /// s_endpgm.
    const VS_BODY: &[u32] = &[
        0x7E00_02FF,
        0x3F80_0000,
        0x7E02_0280,
        0x1004_0300,
        0xF800_08CF,
        0x0302_0100,
        0xF800_020F,
        0x0302_0100,
        S_ENDPGM,
    ];

    #[test]
    fn window_mem_reads_validated_out_of_band_guest_data() {
        let mut external = vec![0u32; CHUNK_DWORDS];
        external[..4].copy_from_slice(&[0xAABB_CCDD, 1, 2, 3]);
        let mem = WindowMem {
            base: 0x1000,
            data: vec![S_ENDPGM],
        };
        crate::guest_mem::with_test_ranges(
            &[(
                external.as_ptr() as u64,
                std::mem::size_of_val(external.as_slice()),
            )],
            || {
                let got = mem
                    .dwords_at(external.as_ptr() as u64)
                    .expect("authorized test allocation is identity-readable");
                assert!(matches!(got, Cow::Owned(_)));
                assert_eq!(&got[..4], &external[..4]);
            },
        );
    }

    /// Solid-green GCN pixel shader (the `shader_bridge` fixture body).
    const PS_BODY: &[u32] = &[
        0x7E00_0280,
        0x7E02_02FF,
        0x3F80_0000,
        0x7E04_0280,
        0x7E06_02FF,
        0x3F80_0000,
        0xF800_180F,
        0x0302_0100,
        S_ENDPGM,
    ];

    /// PS4-style blob with the `0xBEEB03FF` binary-info trailer (mirrors
    /// `shader_bridge::build_shader_blob`).
    fn build_blob(body: &[u32], hash0: u32, crc32: u32) -> Vec<u32> {
        let mut v = vec![SHADER_BINARY_INFO_SENTINEL, 0];
        v.extend_from_slice(body);
        if (v.len() + 1) % 2 != 0 {
            v.push(0);
        }
        v.push(0); // usage masks
        let info_dw = v.len();
        v[1] = (info_dw / 2 - 1) as u32;
        v.push(u32::from_le_bytes(*b"OrbS"));
        v.push(u32::from_le_bytes([b'h', b'd', b'r', 0x42]));
        v.push((body.len() as u32 * 4) << 8);
        v.push(1);
        v.push(hash0);
        v.push(0x1111_2222);
        v.push(crc32);
        // Production shader addresses live in page-backed guest mappings, so
        // the bounded fetcher can safely read its first 4 KiB window. Model
        // that ownership honestly in the test instead of granting authority
        // beyond a short Vec allocation.
        v.resize(CHUNK_DWORDS, 0);
        v
    }

    fn vs_regs_at(addr: u64) -> VertexShaderInfo {
        VertexShaderInfo {
            vs_regs: kyty_graphics::hw_regs::VsStageRegisters {
                data_addr: addr,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn ps_regs_at(addr: u64) -> PixelShaderInfo {
        PixelShaderInfo {
            ps_regs: PsStageRegisters {
                data_addr: addr,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn cs_regs_at(addr: u64) -> kyty_graphics::hw_regs::ComputeShaderInfo {
        kyty_graphics::hw_regs::ComputeShaderInfo {
            cs_regs: kyty_graphics::hw_regs::CsStageRegisters {
                data_addr: addr,
                num_thread_x: 8,
                num_thread_y: 4,
                num_thread_z: 1,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// A synthetic in-memory VS round-trips through fetch → parse →
    /// recompile → SPIR-V, and a per-frame re-bind is a cache hit, not a
    /// re-translation.
    #[test]
    fn guest_vs_round_trips_to_spirv_and_caches() {
        let blob = build_blob(VS_BODY, 0xAAAA_0001, 0xBBBB_0001);
        let addr = blob.as_ptr() as u64;
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        let sh_regs = ShaderRegisters::default(); // export_count = 1

        crate::guest_mem::with_test_ranges(
            &[(addr, std::mem::size_of_val(blob.as_slice()))],
            || {
                let t = cache
                    .translate_vs(&vs_regs_at(addr), &sh_regs)
                    .expect("fixture VS must translate");
                assert_eq!(t.spirv[0], 0x0723_0203, "SPIR-V magic");

                let t2 = cache
                    .translate_vs(&vs_regs_at(addr), &sh_regs)
                    .expect("second bind");
                assert_eq!(t.spirv, t2.spirv);
                let s = cache.stats();
                assert_eq!((s.distinct_fetched, s.translated_ok, s.hits), (1, 1, 1));
            },
        );
    }

    #[test]
    fn guest_ps_round_trips_to_spirv() {
        let blob = build_blob(PS_BODY, 0xAAAA_00E2, 0xBBBB_00E2);
        let addr = blob.as_ptr() as u64;
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        let mut sh_regs = ShaderRegisters::default();
        // Non-compressed MRT0 export needs output mode 9 (as in shader_bridge).
        sh_regs.target_output_mode[0] = 9;

        crate::guest_mem::with_test_ranges(
            &[(addr, std::mem::size_of_val(blob.as_slice()))],
            || {
                let t = cache
                    .translate_ps(
                        &ps_regs_at(addr),
                        &sh_regs,
                        &ShaderVertexInputInfo::default(),
                    )
                    .expect("fixture PS must translate");
                assert_eq!(t.spirv[0], 0x0723_0203, "SPIR-V magic");
            },
        );
    }

    /// The embedded-constant pass used to run for VS/CS only. A pixel shader
    /// using the same `s_getpc_b64` + `s_load_dwordx8` idiom consequently fell
    /// through to the EUD-only recompiler path and was refused even though its
    /// source bytes were compile-time readable.
    #[test]
    fn guest_ps_captures_pc_relative_sload_constants() {
        let constants = [
            0xC0DE_0000,
            0xC0DE_0001,
            0xC0DE_0002,
            0xC0DE_0003,
            0xC0DE_0004,
            0xC0DE_0005,
            0xC0DE_0006,
            0xC0DE_0007,
        ];
        let mut shader = vec![
            0xBE80_1F00, // s_getpc_b64 s[0:1] (base = shader + 4)
            0xF40C_0200, // s_load_dwordx8 s[8:15], s[0:1], 0x2c
            0xFA00_002C, // NULL soffset + 44 bytes => constants at shader + 48
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_02FF, // v_mov_b32 v1, 1.0
            0x3F80_0000,
            0x7E04_0280, // v_mov_b32 v2, 0
            0x7E06_02FF, // v_mov_b32 v3, 1.0
            0x3F80_0000,
            0xF800_180F, // exp mrt0 v0..v3
            0x0302_0100,
            S_ENDPGM,
        ];
        assert_eq!(shader.len() * 4, 48);
        shader.extend_from_slice(&constants);
        shader.resize(CHUNK_DWORDS, 0);

        let addr = shader.as_ptr() as u64;
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        cache.map_shader_metadata(
            addr,
            ShaderMappedData {
                user_data: Some(Default::default()),
                ..Default::default()
            },
        );
        let mut sh_regs = ShaderRegisters::default();
        sh_regs.target_output_mode[0] = 9;

        crate::guest_mem::with_test_ranges(
            &[(addr, std::mem::size_of_val(shader.as_slice()))],
            || {
                let translated = cache
                    .translate_ps(
                        &ps_regs_at(addr),
                        &sh_regs,
                        &ShaderVertexInputInfo::default(),
                    )
                    .expect("PC-relative PS scalar load must translate");
                let captured = translated
                    .ps_info
                    .bind
                    .embedded_constant_loads
                    .find(4)
                    .expect("PS path records the load at pc 4");
                assert_eq!(captured.dwords_num, 8);
                assert_eq!(&captured.values[..8], &constants);
            },
        );
    }

    #[test]
    fn guest_cs_round_trips_to_spirv_and_retains_workgroup_abi() {
        let blob = build_blob(
            &[0xBF80_0000, 0xBF80_0000, S_ENDPGM],
            0xAAAA_00C5,
            0xBBBB_00C5,
        );
        let addr = blob.as_ptr() as u64;
        let mut cache = ShaderTranslateCache::with_dump_dir(None);

        crate::guest_mem::with_test_ranges(
            &[(addr, std::mem::size_of_val(blob.as_slice()))],
            || {
                let t = cache
                    .translate_cs(&cs_regs_at(addr), &ShaderRegisters::default())
                    .expect("fixture CS must translate");
                assert_eq!(t.spirv[0], 0x0723_0203, "SPIR-V magic");
                assert_eq!(t.cs_info.threads_num, [8, 4, 1]);
            },
        );
    }

    #[test]
    fn active_cache_distinguishes_storage_formats_71_and_77() {
        let prepared = |format: u32| {
            let mut info = ShaderComputeInputInfo::default();
            info.bind.textures2d.textures_num = 1;
            info.bind.textures2d.textures2d_storage_num = 1;
            info.bind.textures2d.desc[0].textures2d_without_sampler = true;
            info.bind.textures2d.desc[0].texture.fields[1] |= format << 20;
            info.bind.textures2d.desc[0].texture.fields[3] |= 8 << 28;
            PreparedShader::Cs {
                code: ShaderCode::new(),
                info: Box::new(info),
            }
        };
        let code = CodeKey {
            stage: Stage::Cs,
            addr: 0x5000,
            fetched_dwords: 4,
            digest: 0x1234,
        };
        let key71 = CacheKey {
            code,
            binding: prepared(71).binding_identity(),
        };
        let key77 = CacheKey {
            code,
            binding: prepared(77).binding_identity(),
        };
        assert_ne!(key71, key77, "Rgba16f and Rgba32f modules cannot alias");

        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        cache.insert_entry(key71, Err(Arc::from("sentinel format-71 cache entry")));
        assert!(
            !cache.entries.contains_key(&key77),
            "the active module cache must miss when only the T# format changes"
        );
    }

    /// Garbage bytes fail with a named reason and never panic or poison a
    /// pre-binding cache.
    #[test]
    fn analysis_failures_are_not_cached_before_binding_identity() {
        // 0xFFFF_FFFF decodes as an unknown encoding immediately. The legacy
        // fallback can also fail in header analysis, so the combined failure
        // is intentionally retried instead of poisoning a code-only cache.
        let garbage: Vec<u32> = vec![0xFFFF_FFFF; 64];
        let addr = garbage.as_ptr() as u64;
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        let sh_regs = ShaderRegisters::default();

        crate::guest_mem::with_test_ranges(
            &[(addr, std::mem::size_of_val(garbage.as_slice()))],
            || {
                let e1 = cache
                    .translate_ps(
                        &ps_regs_at(addr),
                        &sh_regs,
                        &ShaderVertexInputInfo::default(),
                    )
                    .expect_err("garbage must not translate");
                assert!(
                    e1.contains("next_gen:") && e1.contains("legacy:"),
                    "both generation attempts must be named: {e1}"
                );

                let e2 = cache
                    .translate_ps(
                        &ps_regs_at(addr),
                        &sh_regs,
                        &ShaderVertexInputInfo::default(),
                    )
                    .expect_err("still failing");
                assert_eq!(e1, e2);
                let s = cache.stats();
                assert_eq!(
                    (s.distinct_fetched, s.translate_failed, s.hits),
                    (2, 2, 0),
                    "pre-binding analysis failures must be retried"
                );
            },
        );
    }

    #[test]
    fn cache_key_hashes_the_fetched_tail_not_just_the_head() {
        let mut blob = build_blob(VS_BODY, 0xAAAA_00F1, 0xBBBB_00F1);
        let addr = blob.as_ptr() as u64;
        let byte_len = std::mem::size_of_val(blob.as_slice());
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        let sh_regs = ShaderRegisters::default();

        crate::guest_mem::with_test_ranges(&[(addr, byte_len)], || {
            let first = cache
                .translate_vs(&vs_regs_at(addr), &sh_regs)
                .expect("initial fixture VS");
            blob[CHUNK_DWORDS - 1] = 0xCAFE_BABE;
            let second = cache
                .translate_vs(&vs_regs_at(addr), &sh_regs)
                .expect("tail-mutated fixture VS");
            assert!(
                !Arc::ptr_eq(&first.spirv, &second.spirv),
                "a fetched-tail rewrite must miss the module cache"
            );
            let stats = cache.stats();
            assert_eq!(
                (stats.distinct_fetched, stats.translated_ok, stats.hits),
                (2, 2, 0)
            );
        });
    }

    #[test]
    fn active_module_cache_is_fifo_bounded() {
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        let key = |i: usize| CacheKey {
            code: CodeKey {
                stage: Stage::Cs,
                addr: 0x1000 + i as u64 * 4,
                fetched_dwords: 4,
                digest: i as u64,
            },
            binding: format!("binding-{i}").into_boxed_str(),
        };
        for i in 0..=MAX_CACHE_ENTRIES {
            cache.insert_entry(key(i), Ok(Arc::new(vec![i as u32])));
        }
        assert_eq!(cache.entries.len(), MAX_CACHE_ENTRIES);
        assert_eq!(cache.insertion_order.len(), MAX_CACHE_ENTRIES);
        assert!(!cache.entries.contains_key(&key(0)), "oldest entry evicted");
        assert!(
            cache.entries.contains_key(&key(MAX_CACHE_ENTRIES)),
            "newest entry retained"
        );
    }

    #[test]
    fn null_and_unaligned_addresses_are_named_errors() {
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        let sh_regs = ShaderRegisters::default();
        let e = cache
            .translate_ps(&ps_regs_at(0), &sh_regs, &ShaderVertexInputInfo::default())
            .expect_err("null");
        assert!(e.contains("null or unaligned"), "{e}");
        let e = cache
            .translate_ps(
                &ps_regs_at(0x1002),
                &sh_regs,
                &ShaderVertexInputInfo::default(),
            )
            .expect_err("unaligned");
        assert!(e.contains("null or unaligned"), "{e}");
    }

    /// Dumps are written once per distinct shader — and written even when
    /// translation fails, because failed shaders are exactly the ones the
    /// GCN→RDNA2 gap study needs.
    #[test]
    fn dump_writes_distinct_shaders_even_on_translation_failure() {
        let dir =
            std::env::temp_dir().join(format!("raeen_shader_dump_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let blob = build_blob(PS_BODY, 1, 2);
        let good_addr = blob.as_ptr() as u64;
        let garbage: Vec<u32> = vec![0xFFFF_FFFF; 64];
        let bad_addr = garbage.as_ptr() as u64;

        let mut cache = ShaderTranslateCache::with_dump_dir(Some(dir.clone()));
        let mut sh_regs = ShaderRegisters::default();
        sh_regs.target_output_mode[0] = 9;
        let vs_info = ShaderVertexInputInfo::default();

        crate::guest_mem::with_test_ranges(
            &[
                (good_addr, std::mem::size_of_val(blob.as_slice())),
                (bad_addr, std::mem::size_of_val(garbage.as_slice())),
            ],
            || {
                cache
                    .translate_ps(&ps_regs_at(good_addr), &sh_regs, &vs_info)
                    .expect("fixture PS");
                let _ = cache.translate_ps(&ps_regs_at(bad_addr), &sh_regs, &vs_info);
                // Re-binds must not duplicate dumps.
                let _ = cache.translate_ps(&ps_regs_at(bad_addr), &sh_regs, &vs_info);
            },
        );

        let files: Vec<_> = std::fs::read_dir(&dir)
            .expect("dump dir exists")
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        // One raw `.bin` per distinct shader, plus one `.spv` for the shader
        // that translated (`dump_spirv` — the coverage-bisect harness input).
        // The failed shader has no SPIR-V to dump.
        assert_eq!(
            files.len(),
            3,
            "raw dumps per distinct shader + SPIR-V for the translated one: {files:?}"
        );
        assert!(
            files
                .iter()
                .any(|f| f.starts_with(&format!("ps_{good_addr:x}_"))),
            "{files:?}"
        );
        assert!(
            files.contains(&format!("ps_{good_addr:x}.spv")),
            "translated shader must dump its SPIR-V: {files:?}"
        );
        assert!(
            files
                .iter()
                .any(|f| f.starts_with(&format!("ps_{bad_addr:x}_"))),
            "translation failure must still dump: {files:?}"
        );
        assert!(
            !files.contains(&format!("ps_{bad_addr:x}.spv")),
            "a shader that failed translation has no SPIR-V: {files:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The dump-length heuristic: sentinel blobs use their declared size,
    /// bare code cuts after `s_endpgm` (+margin), garbage caps at 4 KiB.
    #[test]
    fn dump_len_heuristic_bounds() {
        let blob = build_blob(PS_BODY, 1, 2);
        let declared = (blob[1] as usize + 1) * 2 + 7;
        assert_eq!(dump_len_heuristic(&blob), declared);

        let mut bare = PS_BODY.to_vec();
        bare.extend(std::iter::repeat_n(0u32, 100));
        assert_eq!(dump_len_heuristic(&bare), PS_BODY.len() + 16);

        let garbage = vec![0u32; 3000];
        assert_eq!(dump_len_heuristic(&garbage), CHUNK_DWORDS);
    }
}
