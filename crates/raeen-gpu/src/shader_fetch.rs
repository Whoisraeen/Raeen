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
//! then translated modules are cached by `(stage, guest_addr, parsed-code
//! digest, analyzed stage ABI)`. The analyzed ABI includes descriptor
//! types, exact formats, and embedded constants, so reusing one code address
//! with different code or bindings cannot return stale SPIR-V. The cache is
//! FIFO-bounded. Pre-binding analysis failures use a short bounded backoff:
//! metadata changes invalidate them immediately and stable binds retry
//! periodically, avoiding a per-draw parser/log hot loop without permanently
//! poisoning a shader whose descriptors arrive later.
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
    shader_get_input_info_cs_decoded, shader_get_input_info_ps_decoded,
    shader_get_input_info_vs_decoded, shader_parse_cs, shader_parse_ps, shader_parse_vs,
};
use kyty_graphics::shader::parse::ShaderParseError;
use kyty_graphics::shader::recompile::{
    shader_recompile_cs, shader_recompile_ps, shader_recompile_vs,
};
use kyty_graphics::shader::resources::{
    ShaderBindResources, ShaderComputeInputInfo, ShaderMappedData, ShaderPixelInputInfo,
    ShaderVertexInputInfo,
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
/// Stable binds skipped after a pre-binding analysis failure before retrying.
///
/// A title can submit the same unsupported shader tens of thousands of times
/// while loading. Retrying every bind spent nine CPU cores and emitted 30,734
/// duplicate failures in one measured Minecraft run. This remains deliberately
/// finite because descriptor/EUD state can change without a shader-create call.
const ANALYSIS_FAILURE_RETRY_BINDS: u16 = 255;
/// Bump whenever generated SPIR-V or the binding-identity contract changes.
// v2: guest Cube descriptors remain Vulkan 2D arrays, but their V_CUBE*
// generated S/T coordinates are rebased from the guest [1, 2] convention to
// Vulkan's [0, 1].  Reusing v1 modules silently restores Minecraft's flat
// green panorama, so this codegen change must get a fresh namespace.
const DISK_CACHE_VERSION: u32 = 2;
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
/// parsed instruction stream, not the entire 4-KiB fetch window: Minecraft
/// places transient shader/resource allocations next to one another and was
/// retranslating the same shaders every frame when adjacent bytes changed.
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
    binding: Box<[u32]>,
}

#[derive(Clone, Debug)]
struct AnalysisFailure {
    reason: Arc<str>,
    skips_remaining: u16,
}

#[derive(Clone, Debug)]
struct ParsedCodeEntry {
    code_key: CodeKey,
    next_gen: bool,
    register_tag: u64,
    code: ShaderCode,
}

/// Decoded instruction streams validated against their exact guest-byte
/// prefix on every reuse. Resource/EUD analysis still runs per bind; only the
/// stage-static ISA decode is retained.
#[derive(Default)]
struct ParsedCodeCache {
    entries: VecDeque<ParsedCodeEntry>,
    hits: u64,
    misses: u64,
}

impl ParsedCodeCache {
    fn get_or_parse(
        &mut self,
        stage: Stage,
        addr: u64,
        next_gen: bool,
        register_tag: u64,
        mem: &impl ShaderMemory,
        parse: impl FnOnce() -> Result<ShaderCode, ShaderAnalysisError>,
    ) -> Result<ShaderCode, ShaderAnalysisError> {
        let source = mem.dwords_at(addr);
        if let Some(source) = source.as_deref() {
            for entry in self.entries.iter().rev().filter(|entry| {
                entry.code_key.stage == stage
                    && entry.code_key.addr == addr
                    && entry.next_gen == next_gen
                    && entry.register_tag == register_tag
            }) {
                let dwords = entry.code_key.fetched_dwords as usize;
                if let Some(prefix) = source.get(..dwords)
                    && CodeKey::new(stage, addr, prefix) == entry.code_key
                {
                    self.hits += 1;
                    if crate::diagnostics::gpu_env().time_draw {
                        crate::vulkan::offscreen::DRAW_STAGE_PARSE_HITS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    return Ok(entry.code.clone());
                }
            }
        }

        self.misses += 1;
        if crate::diagnostics::gpu_env().time_draw {
            crate::vulkan::offscreen::DRAW_STAGE_PARSE_MISSES
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let code = parse()?;
        let parsed_dwords = shader_code_dwords(&code);
        if parsed_dwords != 0
            && let Some(source) = mem.dwords_at(addr)
            && let Some(prefix) = source.get(..parsed_dwords)
        {
            while self.entries.len() >= MAX_CACHE_ENTRIES {
                self.entries.pop_front();
            }
            self.entries.push_back(ParsedCodeEntry {
                code_key: CodeKey::new(stage, addr, prefix),
                next_gen,
                register_tag,
                code: code.clone(),
            });
        }
        Ok(code)
    }

    fn parse_vs(
        &mut self,
        regs: &VertexShaderInfo,
        sh_regs: &ShaderRegisters,
        mem: &impl ShaderMemory,
        next_gen: bool,
    ) -> Result<ShaderCode, ShaderAnalysisError> {
        let gs_instead_of_vs = regs.vs_regs.data_addr == 0
            && regs.gs_regs.data_addr == 0
            && regs.es_regs.data_addr != 0
            && regs.gs_regs.chksum != 0;
        let addr = if gs_instead_of_vs {
            regs.es_regs.data_addr
        } else {
            regs.vs_regs.data_addr
        };
        let register_tag = parse_register_tag(&[
            u64::from(regs.vs_embedded),
            u64::from(regs.vs_embedded_id),
            u64::from(regs.vs_regs.rsrc2.user_sgpr),
            u64::from(regs.vs_user_sgpr.count),
            u64::from(regs.gs_regs.rsrc2.user_sgpr),
            u64::from(regs.gs_user_sgpr.count),
            regs.gs_regs.chksum,
        ]);
        self.get_or_parse(Stage::Vs, addr, next_gen, register_tag, mem, || {
            shader_parse_vs(regs, sh_regs, mem, next_gen)
        })
    }

    fn parse_ps(
        &mut self,
        regs: &PixelShaderInfo,
        sh_regs: &ShaderRegisters,
        mem: &impl ShaderMemory,
        next_gen: bool,
    ) -> Result<ShaderCode, ShaderAnalysisError> {
        let register_tag = parse_register_tag(&[
            u64::from(regs.ps_embedded),
            u64::from(regs.ps_embedded_id),
            u64::from(regs.ps_regs.rsrc2.user_sgpr),
            u64::from(regs.ps_user_sgpr.count),
            regs.ps_regs.chksum,
        ]);
        self.get_or_parse(
            Stage::Ps,
            regs.ps_regs.data_addr,
            next_gen,
            register_tag,
            mem,
            || shader_parse_ps(regs, sh_regs, mem, next_gen),
        )
    }

    fn parse_cs(
        &mut self,
        regs: &ComputeShaderInfo,
        sh_regs: &ShaderRegisters,
        mem: &impl ShaderMemory,
        next_gen: bool,
    ) -> Result<ShaderCode, ShaderAnalysisError> {
        let register_tag = parse_register_tag(&[
            u64::from(regs.cs_regs.user_sgpr),
            u64::from(regs.cs_user_sgpr.count),
            regs.cs_regs.chksum,
        ]);
        self.get_or_parse(
            Stage::Cs,
            regs.cs_regs.data_addr,
            next_gen,
            register_tag,
            mem,
            || shader_parse_cs(regs, sh_regs, mem, next_gen),
        )
    }
}

fn shader_code_dwords(code: &ShaderCode) -> usize {
    code.get_instructions()
        .last()
        .map_or(0, |instruction| instruction.pc as usize / 4 + 1)
}

fn parse_register_tag(values: &[u64]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    values.hash(&mut hasher);
    hasher.finish()
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
    /// Number of source dwords that actually belong to the parsed instruction
    /// stream. The parser's final instruction is the terminating `s_endpgm`
    /// (or fetch-shader `s_setpc_b64`), so its PC identifies the exact stable
    /// prefix. Resource metadata and embedded values outside this prefix are
    /// already represented by [`Self::binding_identity`].
    fn parsed_code_dwords(&self) -> usize {
        let code = match self {
            Self::Vs { code, .. } | Self::Ps { code, .. } | Self::Cs { code, .. } => code,
        };
        shader_code_dwords(code)
    }

    fn binding_identity(&self) -> Box<[u32]> {
        let mut id = Vec::with_capacity(192);
        match self {
            Self::Vs { info, .. } => {
                id.extend([
                    info.fetch_external.into(),
                    info.fetch_embedded.into(),
                    info.fetch_inline.into(),
                    info.gs_prolog.into(),
                    info.resources_num as u32,
                    info.export_count as u32,
                    info.fetch_shader_reg as u32,
                    info.fetch_attrib_reg as u32,
                    info.fetch_buffer_reg as u32,
                ]);
                for i in bounded_count(info.resources_num, info.resources.len()) {
                    let resource = info.resources[i];
                    let dst = info.resources_dst[i];
                    id.extend([
                        dst.register_start as u32,
                        dst.registers_num as u32,
                        dst.fetch_index,
                        dst.semantic as u32,
                        u32::from(resource.stride()),
                        resource.swizzle_enabled().into(),
                        u32::from(resource.dst_sel_x()),
                        u32::from(resource.dst_sel_y()),
                        u32::from(resource.dst_sel_z()),
                        u32::from(resource.dst_sel_w()),
                        u32::from(resource.format()),
                        u32::from(resource.out_of_bounds()),
                        resource.add_tid().into(),
                    ]);
                }
                id.push(info.buffers_num as u32);
                for i in bounded_count(info.buffers_num, info.buffers.len()) {
                    let buffer = info.buffers[i];
                    id.extend([buffer.attr_num as u32, buffer.stride, buffer.fetch_index]);
                    for j in bounded_count(buffer.attr_num, buffer.attr_indices.len()) {
                        id.extend([buffer.attr_indices[j] as u32, buffer.attr_offsets[j]]);
                    }
                }
                append_bind_identity(&mut id, &info.bind);
            }
            Self::Ps { info, .. } => {
                id.extend([
                    info.input_num,
                    info.ps_pos_xy.into(),
                    info.ps_pixel_kill_enable.into(),
                    info.ps_early_z.into(),
                    info.ps_execute_on_noop.into(),
                ]);
                let inputs = usize::try_from(info.input_num)
                    .unwrap_or(usize::MAX)
                    .min(info.interpolator_settings.len());
                id.extend_from_slice(&info.interpolator_settings[..inputs]);
                id.extend(info.target_output_mode.map(u32::from));
                append_bind_identity(&mut id, &info.bind);
            }
            Self::Cs { info, .. } => {
                id.extend([
                    info.workgroup_register as u32,
                    info.thread_ids_num as u32,
                    info.lds_size_dw,
                ]);
                for i in 0..3 {
                    id.extend([info.threads_num[i], info.group_id[i].into()]);
                }
                append_bind_identity(&mut id, &info.bind);
            }
        }
        id.into_boxed_slice()
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

fn bounded_count(count: i32, capacity: usize) -> std::ops::Range<usize> {
    0..usize::try_from(count).unwrap_or(0).min(capacity)
}

/// Append only metadata that can change generated SPIR-V or its descriptor
/// ABI. Guest addresses, resource extents, record counts, sampler state, and
/// direct-SGPR values are bind-time data and deliberately excluded.
///
/// This follows Kyty's `ShaderGetBindIds` rule. Raeen adds the fields its
/// expanded recompiler introduced after that upstream function: texture
/// dimension/format, sampler-less storage classification, EUD/global-memory
/// declarations, and embedded shader constants.
fn append_bind_identity(id: &mut Vec<u32>, bind: &ShaderBindResources) {
    id.extend([
        bind.push_constant_offset,
        bind.push_constant_size,
        bind.descriptor_set_slot,
        bind.storage_buffers.buffers_num as u32,
        bind.storage_buffers.binding_index as u32,
    ]);
    for i in bounded_count(
        bind.storage_buffers.buffers_num,
        bind.storage_buffers.buffers.len(),
    ) {
        id.extend([
            bind.storage_buffers.slots[i] as u32,
            bind.storage_buffers.start_register[i] as u32,
            bind.storage_buffers.extended[i].into(),
            bind.storage_buffers.usages[i] as u32,
        ]);
    }

    id.extend([
        bind.textures2d.textures_num as u32,
        bind.textures2d.textures2d_sampled_num as u32,
        bind.textures2d.textures2d_storage_num as u32,
        bind.textures2d.binding_sampled_index as u32,
        bind.textures2d.binding_storage_index as u32,
    ]);
    for desc in &bind.textures2d.desc
        [..bounded_count(bind.textures2d.textures_num, bind.textures2d.desc.len()).end]
    {
        id.extend([
            desc.slot as u32,
            desc.start_register as u32,
            desc.extended.into(),
            desc.usage as u32,
            desc.textures2d_without_sampler.into(),
            u32::from(desc.texture.type_()),
            u32::from(desc.texture.format()),
        ]);
    }

    id.extend([
        bind.samplers.samplers_num as u32,
        bind.samplers.binding_index as u32,
    ]);
    for i in bounded_count(bind.samplers.samplers_num, bind.samplers.samplers.len()) {
        id.extend([
            bind.samplers.slots[i] as u32,
            bind.samplers.start_register[i] as u32,
            bind.samplers.extended[i].into(),
        ]);
    }

    id.extend([
        bind.gds_pointers.pointers_num as u32,
        bind.gds_pointers.binding_index as u32,
    ]);
    for i in bounded_count(
        bind.gds_pointers.pointers_num,
        bind.gds_pointers.pointers.len(),
    ) {
        id.extend([
            bind.gds_pointers.slots[i] as u32,
            bind.gds_pointers.start_register[i] as u32,
            bind.gds_pointers.extended[i].into(),
        ]);
    }

    id.push(bind.direct_sgprs.sgprs_num as u32);
    for i in bounded_count(bind.direct_sgprs.sgprs_num, bind.direct_sgprs.sgprs.len()) {
        id.push(bind.direct_sgprs.start_register[i] as u32);
    }
    id.extend([
        bind.extended.used.into(),
        bind.extended.slot as u32,
        bind.extended.start_register as u32,
        bind.eud_raw.used.into(),
        bind.eud_raw.binding_index as u32,
        bind.eud_raw.required_dwords,
        bind.global_mem.used.into(),
        bind.global_mem.binding_index as u32,
    ]);

    id.push(bind.embedded_constant_loads.loads_num as u32);
    for load in &bind.embedded_constant_loads.loads[..bounded_count(
        bind.embedded_constant_loads.loads_num,
        bind.embedded_constant_loads.loads.len(),
    )
    .end]
    {
        id.extend([load.pc, load.dwords_num]);
        let count = usize::try_from(load.dwords_num)
            .unwrap_or(usize::MAX)
            .min(load.values.len());
        id.extend_from_slice(&load.values[..count]);
    }

    id.push(bind.embedded_buffer_fetches.loads_num as u32);
    for load in &bind.embedded_buffer_fetches.loads[..bounded_count(
        bind.embedded_buffer_fetches.loads_num,
        bind.embedded_buffer_fetches.loads.len(),
    )
    .end]
    {
        id.extend([load.pc, load.inst_offset, load.dwords_num, load.window_len]);
        let count = usize::try_from(load.window_len)
            .unwrap_or(usize::MAX)
            .min(load.window.len());
        id.extend_from_slice(&load.window[..count]);
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
    /// Binding-aware module/error hits plus provisional analysis backoff hits.
    pub hits: u64,
    /// Positive modules restored from the versioned on-disk cache.
    pub disk_hits: u64,
    /// Positive modules committed to the versioned on-disk cache.
    pub disk_writes: u64,
    /// Exact-byte-validated decoded instruction stream reuses.
    pub parse_hits: u64,
    /// ISA decode attempts (including bounded-window growth retries).
    pub parse_misses: u64,
}

/// Fetch + translate + cache for guest shader code.
pub struct ShaderTranslateCache {
    entries: HashMap<CacheKey, Result<Arc<Vec<u32>>, Arc<str>>>,
    insertion_order: VecDeque<CacheKey>,
    analysis_failures: HashMap<CodeKey, AnalysisFailure>,
    analysis_failure_order: VecDeque<CodeKey>,
    parsed_code: ParsedCodeCache,
    /// Shader metadata relocated by `sceAgcCreateShader`, SHARED with every
    /// in-flight analysis closure.
    ///
    /// `Arc` because [`Self::translate_vs`] / `_ps` / `_cs` each need an owned
    /// handle (the closure outlives the `&mut self` borrow that `parsed_code`
    /// takes), and `ShaderMap` is a `HashMap<u64, ShaderMappedData>` whose values
    /// own a `Vec<ShaderSemantic>` — deep-cloning it handed out one heap
    /// allocation per registered shader, twice per DRAW. Registration
    /// ([`Self::map_shader_metadata`]) is rare and copies on write.
    shader_map: Arc<ShaderMap>,
    dump_dir: Option<PathBuf>,
    persistent_dir: Option<PathBuf>,
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
        let config = crate::agc_exec::AgcGpuSession::runtime_config();
        // Draw tracing instruments the recompiler itself (POS0 exports,
        // embedded Fetch* VGPR writes). A persistent SPIR-V hit bypasses those
        // probes and used to make a traced run falsely report that neither path
        // executed. Keep the in-process cache, but force the first bind through
        // translation whenever tracing is explicitly requested.
        let persistent_dir = persistent_cache_enabled(
            config.shader_cache,
            std::env::var_os("RAEEN_TRACE_DRAWS").is_some(),
        )
        .then(|| {
            config
                .shader_cache_dir
                .join(format!("spirv-v{DISK_CACHE_VERSION}"))
        });
        Self::with_dirs(dump_dir, persistent_dir)
    }

    /// Cache with an explicit dump directory (tests; `None` disables dumps).
    #[must_use]
    #[cfg(test)]
    pub fn with_dump_dir(dump_dir: Option<PathBuf>) -> Self {
        Self::with_dirs(dump_dir, None)
    }

    #[must_use]
    fn with_dirs(dump_dir: Option<PathBuf>, persistent_dir: Option<PathBuf>) -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            analysis_failures: HashMap::new(),
            analysis_failure_order: VecDeque::new(),
            parsed_code: ParsedCodeCache::default(),
            shader_map: Arc::new(ShaderMap::new()),
            dump_dir,
            persistent_dir,
            stats: ShaderCacheStats::default(),
        }
    }

    #[must_use]
    pub fn stats(&self) -> ShaderCacheStats {
        ShaderCacheStats {
            parse_hits: self.parsed_code.hits,
            parse_misses: self.parsed_code.misses,
            ..self.stats
        }
    }

    /// Shared handle to the shader metadata map for one analysis closure.
    ///
    /// A refcount bump, not a deep copy of every registered shader's
    /// input-semantics `Vec` — this runs twice per draw (VS + PS) and once per
    /// compute dispatch. `shader_map_handle_is_shared_until_registration` pins
    /// the shape.
    fn shader_map_handle(&self) -> Arc<ShaderMap> {
        Arc::clone(&self.shader_map)
    }

    /// Register metadata relocated by `sceAgcCreateShader` for later
    /// next-generation resource analysis.
    pub fn map_shader_metadata(&mut self, addr: u64, data: ShaderMappedData) {
        Arc::make_mut(&mut self.shader_map).map_user_data(addr, data);
        // A create call can replace the analyzed ABI at this address. Remove
        // prior binding-aware modules eagerly. Mapped user data can also make a
        // shader at another address analyzable, so invalidate all provisional
        // pre-binding failures.
        self.entries.retain(|key, _| key.code.addr != addr);
        self.insertion_order.retain(|key| key.code.addr != addr);
        self.analysis_failures.clear();
        self.analysis_failure_order.clear();
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

    fn insert_analysis_failure(&mut self, key: CodeKey, reason: Arc<str>) {
        if !self.analysis_failures.contains_key(&key) {
            while self.analysis_failures.len() >= MAX_CACHE_ENTRIES {
                let Some(oldest) = self.analysis_failure_order.pop_front() else {
                    self.analysis_failures.clear();
                    break;
                };
                self.analysis_failures.remove(&oldest);
            }
            self.analysis_failure_order.push_back(key);
        }
        self.analysis_failures.insert(
            key,
            AnalysisFailure {
                reason,
                skips_remaining: ANALYSIS_FAILURE_RETRY_BINDS,
            },
        );
    }

    /// Fetch + translate the bound vertex-stage shader.
    ///
    /// # Errors
    ///
    /// A named reason (bad address, unreadable memory, parse/recompile
    /// failure). Post-analysis translation failures are binding-aware cached;
    /// analysis failures are retried after a bounded backoff (or immediately
    /// when shader metadata changes) because descriptors/EUD can change.
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
        let shader_map = self.shader_map_handle();
        let vs = *vs;
        let sh_regs = *sh_regs;
        self.translate(Stage::Vs, addr, move |mem, parsed_code| {
            attempt_generations(|next_gen| {
                let code = parsed_code
                    .parse_vs(&vs, &sh_regs, mem, next_gen)
                    .map_err(|e| AttemptError::from_analysis("shader_parse_vs", &e))?;
                let mut vs_info = ShaderVertexInputInfo::default();
                shader_get_input_info_vs_decoded(
                    &vs,
                    &sh_regs,
                    mem,
                    &shader_map,
                    next_gen,
                    Some(&code),
                    &mut vs_info,
                )
                .map_err(|e| AttemptError::from_analysis("shader_get_input_info_vs", &e))?;
                // The vertex stage was the ONLY stage that never ran this pass
                // (the pixel-stage call site even claimed it did). Measured
                // consequence on build 2741d21: every Avatar: Frontiers of
                // Pandora and GTA V shader error was a VERTEX shader failing at
                // `can't recompile: SLoadDwordx16 [Sdst16SbaseSoffset] s[12:27],
                // s[8:9], 0` / `SLoadDwordx4 ... s[12:13], 64` — a plain
                // constant-offset scalar load through a live-in user-data
                // pointer that no pass had ever resolved for VS, so
                // `sload_dword_extended` fell through its EUD-only path and
                // returned "can't recompile".
                //
                // Next-gen vertex programs run as a gs-prolog, where shader
                // register N is hardware user-data slot N - 8 (the
                // `NGG_SCALAR_BASE` rebase `rebase_ngg_constant_sharps`,
                // `shader_measure_constant_buffer_accesses_shifted` and the
                // recompiler's `shift_regs` all apply), hence the shifted entry
                // point and the gs/vs user-SGPR file selection below.
                kyty_graphics::shader::shader_capture_runtime_scalar_loads_shifted(
                    &code,
                    mem,
                    if gs_instead_of_vs {
                        &vs.gs_user_sgpr
                    } else {
                        &vs.vs_user_sgpr
                    },
                    if vs_info.gs_prolog { 8 } else { 0 },
                    &mut vs_info.bind,
                );
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
                kyty_graphics::shader::shader_measure_constant_buffer_accesses_shifted(
                    &code,
                    &mut vs_info.bind,
                    if vs_info.gs_prolog { 8 } else { 0 },
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
        let shader_map = self.shader_map_handle();
        let ps = *ps;
        let sh_regs = *sh_regs;
        let vs_info = *vs_info;
        self.translate(Stage::Ps, addr, move |mem, parsed_code| {
            attempt_generations(|next_gen| {
                let code = parsed_code
                    .parse_ps(&ps, &sh_regs, mem, next_gen)
                    .map_err(|e| AttemptError::from_analysis("shader_parse_ps", &e))?;
                let mut ps_info = ShaderPixelInputInfo::default();
                shader_get_input_info_ps_decoded(
                    &ps,
                    &sh_regs,
                    &vs_info,
                    mem,
                    &shader_map,
                    next_gen,
                    Some(&code),
                    &mut ps_info,
                )
                .map_err(|e| AttemptError::from_analysis("shader_get_input_info_ps", &e))?;
                // Minecraft gameplay resolves its material T# with
                // `s_load_dwordx8 s[14:21], s[12:13], 0` while declaring no
                // EUD window. Evaluate bounded constant-offset loads through
                // live user-data pointers before the generic placeholder pass
                // so the real texture descriptor reaches both codegen and the
                // Vulkan binding table. (Pixel user SGPRs are not rebased, so
                // this is the unshifted entry point — cf. `translate_vs`.)
                kyty_graphics::shader::shader_capture_runtime_scalar_loads(
                    &code,
                    mem,
                    &ps.ps_user_sgpr,
                    &mut ps_info.bind,
                );
                // PC-relative scalar constant tables are stage-agnostic. VS
                // and CS already run this capture; omitting it here left PS
                // `s_load_dwordx8` instructions to the EUD-only fallback,
                // which correctly refused their non-EUD base register.
                kyty_graphics::shader::shader_detect_embedded_constant_loads(
                    &code,
                    mem,
                    &mut ps_info.bind,
                );
                // A title can supply a sampled T# through a runtime/bindless
                // path that static usage-table analysis cannot capture. The
                // compute stage already degrades that shape to a real bound
                // 1x1 descriptor; pixel shaders need the same guard-safe
                // fallback or one missing material texture drops the entire
                // draw (measured on Minecraft world PS 0x16ff8c00).
                kyty_graphics::shader::shader_synthesize_placeholder_sampled_texture(
                    &code,
                    &mut ps_info.bind,
                );
                // SharpEmu port (see `translate_cs`): default nearest/wrap S#
                // for a PS that samples with zero captured samplers.
                kyty_graphics::shader::shader_synthesize_default_sampler(&code, &mut ps_info.bind);
                kyty_graphics::shader::shader_measure_constant_buffer_accesses(
                    &code,
                    &mut ps_info.bind,
                );
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
        let shader_map = self.shader_map_handle();
        let cs = *cs;
        let sh_regs = *sh_regs;
        self.translate(Stage::Cs, addr, move |mem, parsed_code| {
            attempt_generations(|next_gen| {
                let code = parsed_code
                    .parse_cs(&cs, &sh_regs, mem, next_gen)
                    .map_err(|e| AttemptError::from_analysis("shader_parse_cs", &e))?;
                let mut cs_info = ShaderComputeInputInfo::default();
                shader_get_input_info_cs_decoded(
                    &cs,
                    &sh_regs,
                    mem,
                    &shader_map,
                    next_gen,
                    Some(&code),
                    &mut cs_info,
                )
                .map_err(|e| AttemptError::from_analysis("shader_get_input_info_cs", &e))?;
                // A Gen5 resource table can declare a large V# in its
                // read-write table even when this particular shader only
                // loads it. Prove direct load-only descriptors from the
                // decoded MUBUF operands before binding identity/codegen so
                // the compute backend does not copy an untouched heap back
                // to guest memory after every dispatch.
                kyty_graphics::shader::shader_refine_compute_storage_usage(
                    &code,
                    &mut cs_info.bind,
                );
                // SharpEmu-parity safe degradation: a sampled MIMG whose T#
                // register matches no captured descriptor (a runtime/bindless
                // texture the static capture missed) gets a 1x1 placeholder T#
                // at that register instead of the whole shader refusing
                // (`dynamic-image-descriptor`). Measured on ASTRO.BOT scene
                // compute 0x500566b00 (image_load T# at s16, 13 dispatch skips
                // per level transition). Runs BEFORE the default-sampler synth
                // so a synthesized sampled texture can also get a default S#.
                kyty_graphics::shader::shader_synthesize_placeholder_sampled_texture(
                    &code,
                    &mut cs_info.bind,
                );
                // Rank 8 (draw-time null-descriptor fallback), the STORAGE
                // counterpart, is REVERTED (unwired) — it regressed ASTRO.BOT
                // compute from 0 shader-translation failures to 30. Its
                // descriptor-resolution check (direct start-register match +
                // `mimg_register_eud_alias_index`) is NARROWER than the
                // recompiler's own `mimg_descriptor_guard`, so on registers the
                // guard WOULD resolve via a covered EUD alias it spuriously
                // synthesized a second 1x1 storage T#. That injected descriptor
                //   (1) collided with the already-present descriptor at the same
                //       register — `WriteLocalVariables` then emitted `%vsharp_sN`
                //       twice → "duplicate definition of result id %vsharp_s0"
                //       (0x5006e7a00, 0x5006ea100); and
                //   (2) grew the push-constant table past
                //       `PUSH_CONSTANT_SPILL_THRESHOLD`, spilling `%vsharp` into a
                //       `Uniform` Block whose inner `uint[4]` array has ArrayStride
                //       4 — invalid UBO layout (needs 16) → spirv-val reject
                //       (0x5006fff00).
                // Restoring the iter2 behavior (unresolved non-sampled storage
                // descriptors refuse via the guard's `not_supported()`, keeping
                // the working sampled-placeholder pass above) returns compute to
                // 0 failures. A correct re-wire needs the synthesis pass's alias
                // check to match the guard's resolution exactly AND the spill
                // path to emit StorageBuffer (relaxed layout) instead of Uniform;
                // both are follow-ups. See
                // kyty-graphics recompile.rs test
                // `storage_placeholder_at_occupied_register_duplicates_vsharp`.
                // SharpEmu port + rank-8 broadening: a CS that SAMPLES textures
                // gets an all-zero (nearest/wrap) S# synthesized for every
                // sampler operand register that resolves to no captured sampler
                // — the zero-sampler case AND an unmatched register alongside
                // captured ones — instead of a whole-shader refusal; the Vulkan
                // layer then binds its cached default sampler.
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
                // Beyond Kyty: resolve `s_buffer_load*` through a V# that lives
                // in live-in user data. The VS/PS stages get this from
                // `shader_capture_runtime_scalar_loads*` (which chains it); the
                // compute stage never called that entry point, so the pass has
                // to be invoked directly here. ASTRO.BOT's measured first
                // blocker is exactly this shape in three COMPUTE shaders
                // (`offset != 0 with register soffset on an s_buffer_load
                // (V# base)`); compute user SGPRs are not NGG-rebased, so
                // `shift = 0`.
                kyty_graphics::shader::shader_capture_vsharp_buffer_loads(
                    &code,
                    mem,
                    &cs.cs_user_sgpr,
                    0,
                    &mut cs_info.bind,
                );
                // Same reason, same stage gap: bind a MUBUF V# this analysis
                // can prove when the usage-table walk bound nothing for it.
                // VS/PS reach this through the chained entry point above.
                kyty_graphics::shader::shader_bind_vsharp_storage_buffers(
                    &code,
                    &cs.cs_user_sgpr,
                    0,
                    &mut cs_info.bind,
                );
                kyty_graphics::shader::shader_measure_constant_buffer_accesses(
                    &code,
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
        run: impl Fn(&WindowMem, &mut ParsedCodeCache) -> Result<PreparedShader, AttemptError>,
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
        let analysis_key = CodeKey::new(stage, addr, &head);
        if let Some(failure) = self.analysis_failures.get_mut(&analysis_key) {
            if failure.skips_remaining != 0 {
                failure.skips_remaining -= 1;
                self.stats.hits += 1;
                return Err(Arc::clone(&failure.reason));
            }
            self.analysis_failures.remove(&analysis_key);
            self.analysis_failure_order
                .retain(|key| *key != analysis_key);
        }
        // Analyze on every bind before the positive-cache lookup. Descriptor
        // type/format and embedded metadata can change while code bytes stay
        // identical, and those fields shape generated SPIR-V.
        //
        // COST: this is the whole pass chain (parse + `get_input_info_*` + every
        // capture pass), and it is why a draw that hits the SPIR-V cache still
        // costs tens to hundreds of microseconds. `OffscreenDrawSink`'s
        // `resolved_shaders` memo is what keeps it off the per-draw path;
        // anything that empties that memo puts the whole chain back on every
        // draw. shadPS4 avoids the question by caching the analyzed ABI WITH the
        // module (`Program { Shader::Info info; ModuleList modules; }`,
        // vk_pipeline_cache.h:40, up to 8 specialization permutations) rather
        // than re-deriving it per bind — the model to move to if the memo ever
        // stops being enough. Design reference only; nothing is ported here.
        let mut window = WindowMem {
            base: addr,
            data: head,
        };
        let mut want = CHUNK_DWORDS;
        let prepared = loop {
            let grew = window.grow_to(want);
            match run(&window, &mut self.parsed_code) {
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
                self.insert_analysis_failure(analysis_key, Arc::clone(&reason));
                return Err(reason);
            }
        };
        self.analysis_failures.remove(&analysis_key);
        self.analysis_failure_order
            .retain(|key| *key != analysis_key);

        let parsed_dwords = prepared.parsed_code_dwords();
        // Embedded shaders have no fetched instruction list; retain the small
        // head identity for them. Normal title shaders hash only the exact
        // parsed prefix, eliminating false misses from adjacent guest writes.
        let identity_dwords = if parsed_dwords == 0 {
            window.data.len().min(4)
        } else {
            parsed_dwords.min(window.data.len())
        };
        let code_key = CodeKey::new(stage, addr, &window.data[..identity_dwords]);
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
        if let Some(spirv) = self.load_persistent(&key) {
            self.stats.hits += 1;
            self.stats.disk_hits += 1;
            self.insert_entry(key, Ok(Arc::clone(&spirv)));
            return Ok(prepared.into_translated(spirv));
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
                if self.store_persistent(&key, &spirv) {
                    self.stats.disk_writes += 1;
                }
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

    fn persistent_path(&self, key: &CacheKey) -> Option<PathBuf> {
        let dir = self.persistent_dir.as_ref()?;
        // The guest address is deliberately absent: a future relocated arena
        // must still reuse byte-identical shaders. The two digests cover the
        // parsed source and every ABI field that can shape generated SPIR-V.
        let mut binding_hasher = std::collections::hash_map::DefaultHasher::new();
        key.binding.hash(&mut binding_hasher);
        Some(dir.join(format!(
            "{}-{:08x}-{:016x}-{:016x}.spv",
            key.code.stage.as_str(),
            key.code.fetched_dwords,
            key.code.digest,
            binding_hasher.finish()
        )))
    }

    fn load_persistent(&self, key: &CacheKey) -> Option<Arc<Vec<u32>>> {
        let path = self.persistent_path(key)?;
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
            Err(error) => {
                debug!(%error, path = %path.display(), "persistent shader-cache read failed");
                return None;
            }
        };
        if bytes.len() < 20 || !bytes.len().is_multiple_of(4) {
            warn!(
                path = %path.display(),
                bytes = bytes.len(),
                "discarding malformed persistent shader cache entry"
            );
            let _ = std::fs::remove_file(path);
            return None;
        }
        // Native-endian POD cast (alignment-safe copy). Identical bytes to
        // the old per-word `from_le_bytes` loop on every supported host —
        // the guest ABI is x86-64 (Zen 2), so the host is little-endian.
        let spirv: Vec<u32> = bytemuck::pod_collect_to_vec(&bytes);
        if spirv.first().copied() != Some(0x0723_0203) {
            warn!(
                path = %path.display(),
                "discarding persistent shader cache entry with bad SPIR-V magic"
            );
            let _ = std::fs::remove_file(path);
            return None;
        }
        if crate::spirv_gate::gate_enabled()
            && let Err(reason) = crate::spirv_gate::validate_spirv(&spirv)
        {
            warn!(
                path = %path.display(),
                %reason,
                "discarding invalid persistent shader cache entry"
            );
            let _ = std::fs::remove_file(path);
            return None;
        }
        debug!(
            path = %path.display(),
            words = spirv.len(),
            "persistent shader cache hit"
        );
        Some(Arc::new(spirv))
    }

    fn store_persistent(&self, key: &CacheKey, spirv: &[u32]) -> bool {
        let Some(path) = self.persistent_path(key) else {
            return false;
        };
        if path.exists() {
            return false;
        }
        let Some(dir) = path.parent() else {
            return false;
        };
        if let Err(error) = std::fs::create_dir_all(dir) {
            warn!(
                %error,
                path = %dir.display(),
                "persistent shader-cache directory creation failed"
            );
            return false;
        }
        // Zero-copy view of the words as bytes (see the load-side endianness
        // note); replaces a per-word copy loop.
        let bytes: &[u8] = bytemuck::cast_slice(spirv);
        let temp = path.with_extension(format!("{}.tmp", std::process::id()));
        if let Err(error) = std::fs::write(&temp, bytes) {
            warn!(
                %error,
                path = %temp.display(),
                "persistent shader-cache write failed"
            );
            return false;
        }
        match std::fs::rename(&temp, &path) {
            Ok(()) => {
                debug!(
                    path = %path.display(),
                    words = spirv.len(),
                    "persistent shader cache stored"
                );
                true
            }
            Err(error) if path.exists() => {
                // Another process won the content-addressed cache race.
                let _ = std::fs::remove_file(temp);
                debug!(
                    %error,
                    path = %path.display(),
                    "persistent shader cache already stored"
                );
                false
            }
            Err(error) => {
                let _ = std::fs::remove_file(temp);
                warn!(
                    %error,
                    path = %path.display(),
                    "persistent shader-cache commit failed"
                );
                false
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

const fn persistent_cache_enabled(config_enabled: bool, tracing_draws: bool) -> bool {
    config_enabled && !tracing_draws
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
    mut run: impl FnMut(bool) -> Result<T, AttemptError>,
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

    /// Every bind takes a handle on the shader metadata map. That handle must be
    /// SHARED, not a deep copy of every registered shader's input-semantics
    /// `Vec` — `translate_vs` + `translate_ps` take one each, so a deep clone was
    /// two full map copies per DRAW. Registration copies on write, so a handle
    /// taken earlier keeps observing the map it was taken from.
    #[test]
    fn shader_map_handle_is_shared_until_registration() {
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        cache.map_shader_metadata(
            0x1000,
            ShaderMappedData {
                user_data: None,
                input_semantics: vec![Default::default(); 8],
            },
        );

        let vs_handle = cache.shader_map_handle();
        let ps_handle = cache.shader_map_handle();
        assert!(
            Arc::ptr_eq(&vs_handle, &ps_handle),
            "both stages of one draw must share the metadata map"
        );
        assert!(vs_handle.find(0x1000).is_some());

        // Copy-on-write: registering does not mutate a handle already out.
        cache.map_shader_metadata(
            0x2000,
            ShaderMappedData {
                user_data: None,
                input_semantics: Vec::new(),
            },
        );
        assert!(
            vs_handle.find(0x2000).is_none(),
            "an outstanding handle is a snapshot, not a live view"
        );
        assert!(
            cache.shader_map_handle().find(0x2000).is_some(),
            "the next bind sees the newly registered shader"
        );
    }

    #[test]
    fn draw_trace_bypasses_persistent_spirv_but_not_normal_configuration() {
        assert!(persistent_cache_enabled(true, false));
        assert!(!persistent_cache_enabled(true, true));
        assert!(!persistent_cache_enabled(false, false));
    }

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
                assert_eq!(
                    (s.parse_hits, s.parse_misses),
                    (1, 3),
                    "the second legacy bind must reuse its validated decoded instruction stream; \
                     the cheap next-gen rejection is retried"
                );
            },
        );
    }

    #[test]
    fn guest_vs_round_trips_through_the_persistent_cache() {
        let blob = build_blob(VS_BODY, 0xAAAA_00D1, 0xBBBB_00D1);
        let addr = blob.as_ptr() as u64;
        let dir = std::env::temp_dir().join(format!(
            "raeen-shader-cache-test-{}-{:x}",
            std::process::id(),
            addr
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let sh_regs = ShaderRegisters::default();

        crate::guest_mem::with_test_ranges(
            &[(addr, std::mem::size_of_val(blob.as_slice()))],
            || {
                let mut writer = ShaderTranslateCache::with_dirs(None, Some(dir.clone()));
                let first = writer
                    .translate_vs(&vs_regs_at(addr), &sh_regs)
                    .expect("initial translation");
                assert_eq!(writer.stats().disk_writes, 1);

                let mut reader = ShaderTranslateCache::with_dirs(None, Some(dir.clone()));
                let second = reader
                    .translate_vs(&vs_regs_at(addr), &sh_regs)
                    .expect("persistent hit");
                assert_eq!(first.spirv, second.spirv);
                assert_eq!(reader.stats().disk_hits, 1);
                assert_eq!(reader.stats().translated_ok, 0);
            },
        );
        let _ = std::fs::remove_dir_all(dir);
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

    #[test]
    fn module_identity_excludes_runtime_resource_payloads_but_bind_keeps_them_fresh() {
        let prepared = |buffer_addr: u64, texture_addr: u64, records: u32| {
            let mut info = ShaderComputeInputInfo::default();
            info.bind.storage_buffers.buffers_num = 1;
            info.bind.storage_buffers.buffers[0].update_address48(buffer_addr);
            info.bind.storage_buffers.buffers[0].fields[2] = records;
            info.bind.textures2d.textures_num = 1;
            info.bind.textures2d.textures2d_sampled_num = 1;
            info.bind.textures2d.desc[0]
                .texture
                .update_address40(texture_addr);
            info.bind.textures2d.desc[0].texture.fields[2] = records << 14;
            info.bind.textures2d.desc[0].texture.fields[3] |= 9 << 28;
            info.bind.samplers.samplers_num = 1;
            info.bind.samplers.samplers[0].fields = [records; 4];
            PreparedShader::Cs {
                code: ShaderCode::new(),
                info: Box::new(info),
            }
        };

        let first = prepared(0x1111_2222_3000, 0x2222_3333_4000, 64);
        let second = prepared(0xAAAA_BBBB_C000, 0xBBBB_CCCC_D000, 4096);
        assert_eq!(
            first.binding_identity(),
            second.binding_identity(),
            "guest addresses, extents, record counts, and sampler payloads are bind-time state"
        );

        let translated = second.into_translated(Arc::new(vec![0x0723_0203]));
        assert_eq!(
            translated.cs_info.bind.storage_buffers.buffers[0].base48(),
            0xAAAA_BBBB_C000,
            "a cache hit must still return the current bind's descriptor metadata"
        );
        assert_eq!(
            translated.cs_info.bind.storage_buffers.buffers[0].num_records(),
            4096
        );
    }

    /// Garbage bytes fail with a named reason and use a finite pre-binding
    /// backoff rather than re-running analysis for every draw.
    #[test]
    fn analysis_failures_back_off_then_retry() {
        // 0xFFFF_FFFF decodes as an unknown encoding immediately. The legacy
        // fallback can also fail in header analysis, so the combined failure is
        // provisional rather than a permanent code-only cache entry.
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
                    .expect_err("backoff returns the same named failure");
                assert_eq!(e1, e2);
                let s = cache.stats();
                assert_eq!(
                    (s.distinct_fetched, s.translate_failed, s.hits),
                    (1, 1, 1),
                    "the first stable rebind must not rerun analysis"
                );

                for _ in 1..ANALYSIS_FAILURE_RETRY_BINDS {
                    cache
                        .translate_ps(
                            &ps_regs_at(addr),
                            &sh_regs,
                            &ShaderVertexInputInfo::default(),
                        )
                        .expect_err("bounded backoff remains a named failure");
                }
                cache
                    .translate_ps(
                        &ps_regs_at(addr),
                        &sh_regs,
                        &ShaderVertexInputInfo::default(),
                    )
                    .expect_err("analysis must retry after the bounded backoff");
                let s = cache.stats();
                assert_eq!(
                    (s.distinct_fetched, s.translate_failed, s.hits),
                    (2, 2, u64::from(ANALYSIS_FAILURE_RETRY_BINDS)),
                    "pre-binding failures must still retry periodically"
                );
            },
        );
    }

    #[test]
    fn shader_metadata_invalidates_analysis_failure_backoff() {
        let garbage: Vec<u32> = vec![0xFFFF_FFFF; 64];
        let addr = garbage.as_ptr() as u64;
        let mut cache = ShaderTranslateCache::with_dump_dir(None);
        let sh_regs = ShaderRegisters::default();

        crate::guest_mem::with_test_ranges(
            &[(addr, std::mem::size_of_val(garbage.as_slice()))],
            || {
                cache
                    .translate_ps(
                        &ps_regs_at(addr),
                        &sh_regs,
                        &ShaderVertexInputInfo::default(),
                    )
                    .expect_err("garbage must not translate");
                cache.map_shader_metadata(addr + 0x1000, ShaderMappedData::default());
                cache
                    .translate_ps(
                        &ps_regs_at(addr),
                        &sh_regs,
                        &ShaderVertexInputInfo::default(),
                    )
                    .expect_err("metadata invalidation must force a fresh attempt");
                let s = cache.stats();
                assert_eq!((s.distinct_fetched, s.translate_failed, s.hits), (2, 2, 0));
            },
        );
    }

    #[test]
    fn cache_key_ignores_adjacent_tail_but_detects_parsed_code_rewrites() {
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
                Arc::ptr_eq(&first.spirv, &second.spirv),
                "bytes beyond the parsed shader must not invalidate its module"
            );
            let body_dword = 3;
            blob[body_dword] ^= 1;
            let third = cache
                .translate_vs(&vs_regs_at(addr), &sh_regs)
                .expect("code-mutated fixture VS");
            assert!(
                !Arc::ptr_eq(&second.spirv, &third.spirv),
                "a parsed instruction rewrite must miss the module cache"
            );
            let stats = cache.stats();
            assert_eq!(
                (stats.distinct_fetched, stats.translated_ok, stats.hits),
                (2, 2, 1)
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
            binding: vec![i as u32].into_boxed_slice(),
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
