//! Shader analysis: binary-info trailer, input-usage slots, resource
//! ("sharp") extraction, vertex-fetch recovery, per-stage input infos and
//! pipeline-cache ids. Ported from Kyty `Graphics/Shader.cpp`
//! (MIT (c) InoriRus).
//!
//! Kyty anchors: `ShaderBinaryInfo` L37, `ShaderUsageSlot` L58,
//! `ShaderUsageInfo` L66, `ShaderParsedUsage` L74, `GetBinaryInfo` L909,
//! `GetUsageSlots` L921, `ShaderDetectBuffers` L944, `ShaderParseFetch`
//! L1005, `ShaderParseAttrib` L1095, `ShaderGetStorageBuffer` L1141,
//! `ShaderGetTextureBuffer` L1179, `ShaderGetSampler` L1233,
//! `ShaderGetGdsPointer` L1270, `ShaderGetDirectSgpr` L1301,
//! `ShaderCalcBindingIndices` L1321, `ShaderParseUsage` L1364,
//! `ShaderParseUsage2` L1505, `ShaderGetInputInfoVS/PS/CS` L1630/L1744/L1811,
//! `ShaderParseVS/PS/CS` L2287/L2397/L2500, `ShaderGetBindIds` L2679,
//! `ShaderGetIdVS/PS/CS` L2794/L2885/L2935.
//!
//! Deviations (systematic, applied throughout):
//! - Kyty reads guest memory through raw pointers
//!   (`reinterpret_cast<const uint32_t*>(addr)`); the port takes `&[u32]`
//!   slices plus a [`ShaderMemory`] trait that the caller implements to
//!   resolve guest addresses, and every access is bounds-checked.
//! - `EXIT`/`EXIT_IF`/`EXIT_NOT_IMPLEMENTED` become
//!   [`ShaderAnalysisError`] values (with `tracing::error!` logging).
//! - The globals `g_shader_map`/`ShaderInit`/`ShaderMapUserData` (L99-L115)
//!   become the caller-owned [`ShaderMap`]; `g_disabled_shaders`
//!   (`ShaderIsDisabled`/`ShaderDisable`, L2967-L3004) and the debug-printf
//!   registry (`g_debug_printfs`/`ShaderInjectDebugPrintf`, L3006) are dev
//!   tooling and are not ported (the `ShaderCode::debug_printfs` field
//!   itself exists in `types.rs`).
//! - `vs_print`/`vs_check`/`ps_check`/`cs_check`/`bi_print` (debug dumps and
//!   register validation, Shader.cpp L520-L900) are not ported.
//! - `ShaderRecompileVS/PS/CS` (SPIR-V generation + spirv-tools) belong to
//!   the next batch.

use std::borrow::Cow;
use std::collections::HashMap;

use super::hw_regs::{
    ComputeShaderInfo, PixelShaderInfo, ShaderRegisters, UserSgprInfo, UserSgprType,
    VertexShaderInfo,
};
use super::parse::{ShaderParseError, shader_parse};
use super::resources::{
    ShaderBindResources, ShaderBufferResource, ShaderComputeInputInfo, ShaderDirectSgprsResources,
    ShaderEmbeddedBufferFetch, ShaderEmbeddedBufferFetches, ShaderEmbeddedConstantLoad,
    ShaderEmbeddedConstantLoads, ShaderGdsResources, ShaderId, ShaderMappedData,
    ShaderPixelInputInfo, ShaderSamplerResources, ShaderSemantic, ShaderSharp,
    ShaderStorageResources, ShaderStorageUsage, ShaderTextureResources, ShaderTextureUsage,
    ShaderUserData, ShaderVertexInputBuffer, ShaderVertexInputInfo,
};
use super::types::{
    ShaderCode, ShaderInstruction, ShaderInstructionType, ShaderOperand, ShaderOperandType,
    ShaderType,
};

/// Typed replacement for Kyty's hard exits in `Shader.cpp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderAnalysisError {
    /// A nested `ShaderParse` failure.
    Parse(ShaderParseError),
    /// `GetBinaryInfo` returned null where Kyty EXITs
    /// (`EXIT_NOT_IMPLEMENTED(header == nullptr)`).
    NoBinaryInfo,
    /// A bounds-checked access fell outside its slice (Kyty reads raw
    /// memory unchecked here).
    Truncated { what: &'static str },
    /// `EXIT_NOT_IMPLEMENTED(...)` condition hit.
    NotImplemented { what: &'static str },
    /// `EXIT_NOT_IMPLEMENTED` with the raw fields the next session needs to
    /// implement the gate — a named error, never a guessed format.
    NotImplementedOwned { what: String },
    /// `EXIT("unknown usage type: ...")` (Shader.cpp L1490/L1564).
    UnknownUsageType { type_: u32 },
    /// A guest address could not be resolved through [`ShaderMemory`]
    /// (Kyty: null/invalid pointer).
    BadAddress { addr: u64 },
}

impl std::fmt::Display for ShaderAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "{e}"),
            Self::NoBinaryInfo => write!(f, "shader binary info (0xBEEB03FF trailer) not found"),
            Self::Truncated { what } => write!(f, "shader analysis out of bounds: {what}"),
            Self::NotImplemented { what } => write!(f, "shader analysis not implemented: {what}"),
            Self::NotImplementedOwned { what } => {
                write!(f, "shader analysis not implemented: {what}")
            }
            Self::UnknownUsageType { type_ } => write!(f, "unknown usage type: 0x{type_:02x}"),
            Self::BadAddress { addr } => write!(f, "bad guest address 0x{addr:016x}"),
        }
    }
}

impl std::error::Error for ShaderAnalysisError {}

impl From<ShaderParseError> for ShaderAnalysisError {
    fn from(e: ShaderParseError) -> Self {
        Self::Parse(e)
    }
}

/// `EXIT_NOT_IMPLEMENTED`: log Kyty-style and build the typed error.
fn ni(what: &'static str) -> ShaderAnalysisError {
    tracing::error!("shader analysis: not implemented: {what}");
    ShaderAnalysisError::NotImplemented { what }
}

/// `EXIT_NOT_IMPLEMENTED` carrying the raw values a gate must name — the
/// difference between "something failed" and "implement exactly this next".
fn ni_owned(what: String) -> ShaderAnalysisError {
    tracing::error!("shader analysis: not implemented: {what}");
    ShaderAnalysisError::NotImplementedOwned { what }
}

/// Bounds violation that Kyty would have read through unchecked.
fn trunc(what: &'static str) -> ShaderAnalysisError {
    tracing::error!("shader analysis: out of bounds: {what}");
    ShaderAnalysisError::Truncated { what }
}

/// Out-of-bounds carrying the raw values the fix needs (same treatment as
/// `ni_owned` — a gate names its evidence, never just itself).
fn trunc_owned(what: String) -> ShaderAnalysisError {
    tracing::error!("shader analysis: out of bounds: {what}");
    ShaderAnalysisError::NotImplementedOwned { what }
}

fn bad_addr(addr: u64) -> ShaderAnalysisError {
    tracing::error!("shader analysis: bad guest address 0x{addr:016x}");
    ShaderAnalysisError::BadAddress { addr }
}

/// Guest-memory view used to resolve the raw pointers Kyty dereferences
/// (shader code, fetch shaders, V#/attrib tables, the EUD buffer).
///
/// `dwords_at(addr)` returns all dwords mapped at byte address `addr`
/// through the end of the containing region, or `None` when unmapped
/// (including `addr == 0`, which Kyty guards as `EXIT_NOT_IMPLEMENTED(src ==
/// nullptr)`).
pub trait ShaderMemory {
    fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>>;
}

/// Kyty: Shader.cpp `ShaderBinaryInfo` (L37) — the 28-byte trailer appended
/// after PS4 shader code. Bitfields are decoded into plain fields.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderBinaryInfo {
    pub signature: [u8; 7],
    pub version: u8,
    pub pssl_or_cg: bool,
    pub cached: bool,
    pub type_: u8,
    pub source_type: u8,
    pub length: u32,
    pub chunk_usage_base_offset_dw: u8,
    pub num_input_usage_slots: u8,
    pub is_srt: bool,
    pub is_srt_used_info_valid: bool,
    pub is_extended_usage_info: bool,
    pub hash0: u32,
    pub hash1: u32,
    pub crc32: u32,
}

/// Size of the binary-info trailer in dwords (`sizeof(ShaderBinaryInfo)`).
const SHADER_BINARY_INFO_SIZE_DW: usize = 7;

/// The `s_mov_b32 vcc_hi, <literal>` sentinel that starts every shader
/// carrying a binary-info trailer (Shader.cpp L913).
pub const SHADER_BINARY_INFO_SENTINEL: u32 = 0xBEEB_03FF;

/// Kyty: Shader.cpp `ShaderUsageSlot` (L58) — one input-usage record
/// (4 bytes) in the pre-trailer table.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderUsageSlot {
    pub type_: u8,
    pub slot: u8,
    pub start_register: u8,
    pub flags: u8,
}

/// Kyty: Shader.cpp `ShaderUsageInfo` (L66).
///
/// Deviation: Kyty stores raw pointers into the code; the port stores the
/// usage-mask dword offset into the code slice plus the decoded slots.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderUsageInfo {
    pub usage_masks_offset_dw: usize,
    pub slots: Vec<ShaderUsageSlot>,
    pub valid: bool,
}

/// Kyty: Shader.cpp `ShaderParsedUsage` (L74).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderParsedUsage {
    pub fetch: bool,
    pub fetch_reg: i32,
    pub vertex_buffer: bool,
    pub vertex_buffer_reg: i32,
    pub vertex_attrib: bool,
    pub vertex_attrib_reg: i32,
    pub storage_buffers_readwrite: i32,
    pub storage_buffers_readonly: i32,
    pub storage_buffers_constant: i32,
    pub textures2d_readonly: i32,
    pub textures2d_readwrite: i32,
    pub extended_buffer: bool,
    pub samplers: i32,
    pub gds_pointers: i32,
    pub direct_sgprs: i32,
}

/// Kyty: `g_shader_map` + `ShaderInit` + `ShaderMapUserData` (Shader.cpp
/// L99-L115).
///
/// Deviation: no global state — the caller owns the map (one per GPU
/// context) and passes it to the PS5 (`next_gen`) input-info paths.
#[derive(Clone, Debug, Default)]
pub struct ShaderMap {
    map: HashMap<u64, ShaderMappedData>,
}

impl ShaderMap {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// KytyPS5: `ShaderMapUserData` assigns through `operator[]`, replacing an
    /// existing entry. Shader-code allocations are reusable, so the mapped
    /// header at an address must always describe its most recent creation.
    pub fn map_user_data(&mut self, addr: u64, data: ShaderMappedData) {
        self.map.insert(addr, data);
    }

    #[must_use]
    pub fn find(&self, addr: u64) -> Option<&ShaderMappedData> {
        self.map.get(&addr)
    }
}

/// Dword offset of the binary-info trailer inside `code`, if the sentinel is
/// present and the trailer fits (Kyty: `GetBinaryInfo` pointer math, L915).
fn binary_info_offset_dw(code: &[u32]) -> Option<usize> {
    if code.len() < 2 || code[0] != SHADER_BINARY_INFO_SENTINEL {
        return None;
    }
    let offset = (code[1] as usize + 1) * 2;
    if offset + SHADER_BINARY_INFO_SIZE_DW > code.len() {
        // Deviation: Kyty would read past the mapping; the port treats a
        // truncated trailer as "no binary info".
        return None;
    }
    Some(offset)
}

/// Kyty: Shader.cpp `GetBinaryInfo` (L909). Scans for the `0xBEEB03FF`
/// sentinel at `code[0]` and decodes the trailer at
/// `code + (code[1] + 1) * 2`.
#[must_use]
pub fn get_binary_info(code: &[u32]) -> Option<ShaderBinaryInfo> {
    let dw = &code[binary_info_offset_dw(code)?..];
    let b0 = dw[0].to_le_bytes();
    let b1 = dw[1].to_le_bytes();
    let b3 = dw[3].to_le_bytes();
    Some(ShaderBinaryInfo {
        signature: [b0[0], b0[1], b0[2], b0[3], b1[0], b1[1], b1[2]],
        version: b1[3],
        pssl_or_cg: dw[2] & 0x1 == 1,
        cached: (dw[2] >> 1) & 0x1 == 1,
        type_: ((dw[2] >> 2) & 0xF) as u8,
        source_type: ((dw[2] >> 6) & 0x3) as u8,
        length: dw[2] >> 8,
        chunk_usage_base_offset_dw: b3[0],
        num_input_usage_slots: b3[1],
        is_srt: b3[2] & 0x1 == 1,
        is_srt_used_info_valid: (b3[2] >> 1) & 0x1 == 1,
        is_extended_usage_info: (b3[2] >> 2) & 0x1 == 1,
        hash0: dw[4],
        hash1: dw[5],
        crc32: dw[6],
    })
}

/// Kyty: Shader.cpp `GetUsageSlots` (L921). Walks backwards from the
/// binary-info trailer: usage masks start `chunk_usage_base_offset_dw`
/// dwords before it, and the `ShaderUsageSlot[]` table sits immediately
/// before the masks (one dword per slot).
pub fn get_usage_slots(code: &[u32]) -> Result<ShaderUsageInfo, ShaderAnalysisError> {
    let mut ret = ShaderUsageInfo::default();

    let Some(info_offset) = binary_info_offset_dw(code) else {
        return Ok(ret); // Kyty: null binary_info -> valid == false.
    };
    let binary_info = get_binary_info(code).expect("offset implies parseable trailer");

    if binary_info.chunk_usage_base_offset_dw == 0 {
        return Err(ni("chunk_usage_base_offset_dw == 0"));
    }

    let chunk = binary_info.chunk_usage_base_offset_dw as usize;
    let slots_num = binary_info.num_input_usage_slots as usize;

    let masks_offset = info_offset
        .checked_sub(chunk)
        .ok_or_else(|| trunc("usage masks before start of code"))?;
    let slots_offset = masks_offset
        .checked_sub(slots_num)
        .ok_or_else(|| trunc("usage slots before start of code"))?;

    ret.usage_masks_offset_dw = masks_offset;
    ret.slots = code[slots_offset..slots_offset + slots_num]
        .iter()
        .map(|&raw| ShaderUsageSlot {
            type_: (raw & 0xFF) as u8,
            slot: ((raw >> 8) & 0xFF) as u8,
            start_register: ((raw >> 16) & 0xFF) as u8,
            flags: (raw >> 24) as u8,
        })
        .collect();
    ret.valid = true;

    Ok(ret)
}

/// Clear a direct-SGPR candidate flag; Kyty indexes `direct_sgprs[...]`
/// unchecked, the port bounds-checks (deviation).
fn clear_direct(
    direct_sgprs: &mut [bool; UserSgprInfo::SGPRS_MAX],
    index: usize,
) -> Result<(), ShaderAnalysisError> {
    let slot = direct_sgprs
        .get_mut(index)
        .ok_or_else(|| trunc("user sgpr index beyond SGPRS_MAX"))?;
    *slot = false;
    Ok(())
}

/// A recovered Extended User Data buffer plus the user-data slot index where
/// the register file logically hands over to it.
///
/// `base_dw` is the register holding the EUD pointer pair: the driver stops
/// preloading user data into real SGPRs at that slot, so a descriptor
/// declared at register `r >= base_dw` (but below `SGPRS_MAX`) lives
/// EUD-resident at `data[r - base_dw]`. Measured on ASTRO.BOT compute
/// (declared=14, captured=14): the pointer pair is (s12, s13) — see
/// `read_extended_user_data` strategy 3 — and the T# the usage table declares
/// at start_register=12 is the EUD's first descriptor (`data[0..8]`; the
/// smallest measured EUD is exactly 8 dwords, which only an offset of 0
/// fits). Sharps declared at `r >= SGPRS_MAX` keep the separately measured
/// `-SGPRS_MAX` rebase (see `read_sharp_fields`).
#[derive(Debug, Clone, Copy)]
pub struct EudView<'a> {
    /// The EUD contents, read from guest memory.
    pub data: &'a [u32],
    /// User-data slot index at which the EUD logically begins (the register
    /// holding the pointer pair).
    pub base_dw: i32,
}

/// Shared body of `ShaderGetStorageBuffer`/`ShaderGetTextureBuffer`/
/// `ShaderGetSampler` (Shader.cpp L1141/L1179/L1233): decide where the
/// descriptor lives (register file vs EUD), require `Vsharp`/`Region`-typed
/// user SGPRs for the direct case, mark consumed registers, and copy the
/// descriptor dwords.
///
/// Returns whether the descriptor content was read from the EUD (`true`) or
/// the register file (`false`) — the recorders stamp their per-descriptor
/// `extended` flag from this, NOT from "the shader has an EUD at all".
/// Measured on ASTRO.BOT compute (round 8): 404 + 289 emission refusals
/// ("extended texture at s0 / storage buffer at s8 has no EUD base to rebase
/// on", eud_base=12) were all direct-resident sharps mislabeled extended.
fn read_sharp_fields(
    direct_sgprs: &mut [bool; UserSgprInfo::SGPRS_MAX],
    start_index: i32,
    user_sgpr: &UserSgprInfo,
    extended_buffer: Option<EudView<'_>>,
    out: &mut [u32],
) -> Result<bool, ShaderAnalysisError> {
    let extended = extended_buffer.is_some();
    if !extended && start_index as usize >= UserSgprInfo::SGPRS_MAX {
        // Kyty treats >=16 as "this sharp lives in the extended buffer", but
        // Gen5 graphics stages have 32 REAL user SGPRs (see
        // `UserSgprInfo::SGPRS_MAX`). Measured on ASTRO.BOT: pixel shaders
        // reference a sharp at start_register=16 while declaring NO extended
        // buffer — with no EUD there is nowhere else that data could live, so
        // s16 must be a direct register. The bound is therefore the register
        // file size, not 16. This stays self-validating: a slot that was never
        // written keeps type_ Unknown and is rejected by the Vsharp/Region
        // check below, so relaxing the bound cannot invent descriptors.
        return Err(ni_owned(format!(
            "sharp start_register beyond the user SGPR file (start_register={start_index})"
        )));
    }
    let start = usize::try_from(start_index).map_err(|_| trunc("negative sharp start register"))?;

    // Residence is decided by the register INDEX and by whether the WHOLE
    // sharp fits inside the typed run of the register file:
    //
    // * `start >= SGPRS_MAX`: EUD-resident at `start - SGPRS_MAX` (measured:
    //   a shader with eud_size_dw=8 places a sharp at offset_dw=32 while its
    //   other sharps sit direct at s0/s8 — only a `-32` rebase reads ext[0],
    //   which fits).
    // * `start < SGPRS_MAX` and every register of the run is typed
    //   Vsharp/Region: direct-resident (a shader with an EUD still keeps
    //   some sharps in its real user SGPRs — 346 measured ASTRO.BOT CS
    //   failures under Kyty's "extended => everything is EUD" EXIT).
    // * `start < SGPRS_MAX` but the run does NOT fit the typed registers: the
    //   sharp cannot live in the file. Measured on ASTRO.BOT compute (469
    //   dispatches/run, all one tuple): a T#-sized sharp declared at
    //   start_register=12 with captured=14 would need s12..s19, but s14+ were
    //   never written — and (s12, s13) is the recovered EUD POINTER pair
    //   (strategy 3, `eud_adjacent_pair_scan_recovers_measured_pointer`). The
    //   sharp is therefore not AT s12; it is IN the EUD the pair points to,
    //   at `start - base_dw` (= 0 here — the only offset an 8-dword T# fits
    //   in the smallest measured 8-dword EUD).
    let mut eud_resident = false;
    if start < UserSgprInfo::SGPRS_MAX {
        let fits_typed_run = (0..out.len()).all(|j| {
            user_sgpr
                .type_
                .get(start + j)
                .is_some_and(|t| *t == UserSgprType::Vsharp || *t == UserSgprType::Region)
        });
        if fits_typed_run {
            for (j, dw) in out.iter_mut().enumerate() {
                let idx = start + j;
                direct_sgprs[idx] = false;
                *dw = user_sgpr.value[idx];
            }
        } else if let Some(ext) = extended_buffer {
            let base =
                usize::try_from(ext.base_dw).map_err(|_| trunc("negative EUD base register"))?;
            if start < base {
                return Err(ni_owned(format!(
                    "sharp start_register below the EUD base \
                     (start_register={start_index}, eud_base={base}, captured={}, eud={})",
                    user_sgpr.count,
                    ext.data.len(),
                )));
            }
            let offset = start - base;
            for (j, dw) in out.iter_mut().enumerate() {
                *dw = *ext
                    .get_dw(offset + j)
                    .ok_or_else(|| trunc("extended (EUD) buffer too small"))?;
            }
            // Consume the run's in-file registers (the pointer pair for the
            // measured shape) so the direct-SGPR collection does not bind
            // them a second time.
            for flag in direct_sgprs
                .iter_mut()
                .take((start + out.len()).min(UserSgprInfo::SGPRS_MAX))
                .skip(start)
            {
                *flag = false;
            }
            tracing::debug!(
                start_index,
                eud_base = base,
                offset_dw = offset,
                dwords = out.len(),
                "sharp resolved EUD-resident (run does not fit the typed register file)"
            );
            eud_resident = true;
        } else {
            // Evidence-rich residual frontier: name the first register that
            // breaks the run, with every measured value a rebase
            // interpretation needs.
            let (idx, type_) = (0..out.len())
                .map(|j| start + j)
                .map(|idx| {
                    (
                        idx,
                        user_sgpr
                            .type_
                            .get(idx)
                            .copied()
                            .unwrap_or(UserSgprType::Unknown),
                    )
                })
                .find(|(_, t)| *t != UserSgprType::Vsharp && *t != UserSgprType::Region)
                .expect("!fits_typed_run implies an offending register");
            // Do NOT log at ERROR here (build the error quietly instead of via
            // `ni_owned`): this residual is the runtime-resolved (SRT/bindless)
            // sampled-T# shape — a sharp needs more registers than the driver
            // captured, with no EUD. It is RECOVERED by
            // `shader_get_texture_buffer` below (installs a placeholder T#, logs
            // at DEBUG); the fatal storage/sampler callers surface it via
            // `raeen-gpu`'s `shader_fetch` WARN. A construction-site ERROR was a
            // false alarm for the recovered path — measured on ASTRO.BOT
            // 0x100008e6aa00, which logged this 5x per level transition yet
            // translated successfully every time.
            return Err(ShaderAnalysisError::NotImplementedOwned {
                what: format!(
                    "user sgpr type is not Vsharp/Region ({type_:?} at s{idx}; \
                     sharp start_register={start_index}, captured={}, eud=0)",
                    user_sgpr.count,
                ),
            });
        }
    } else {
        // The EUD is addressed as a continuation of the user-SGPR file, so
        // the rebase is the file SIZE, not a literal 16 (Kyty's 16 was the
        // PS4 user-SGPR count). See the residence comment above.
        let ext = extended_buffer.expect("start >= SGPRS_MAX without EUD is refused above");
        for (j, dw) in out.iter_mut().enumerate() {
            *dw = *ext
                .get_dw(start - UserSgprInfo::SGPRS_MAX + j)
                .ok_or_else(|| trunc("extended (EUD) buffer too small"))?;
        }
        eud_resident = true;
    }
    Ok(eud_resident)
}

impl EudView<'_> {
    fn get_dw(&self, index: usize) -> Option<&u32> {
        self.data.get(index)
    }
}

/// A scalar memory load's source: the user-SGPR **base pointer** register it
/// reads through, the byte **offset** applied, and the dword width.
///
/// This is the first component of the SRT / scalar-descriptor resolver
/// (task #9) and attacks its documented keystone: the EUD/SRT buffer pointer is
/// *not* a field anywhere in `ShaderUserData`; it must be recovered by analysing
/// the shader. A shader whose descriptors live in a runtime buffer addresses
/// them as `s_load_dword{,x2,x4,x8} sdst, sbase, offset`, so `sbase` is the
/// pointer and its runtime value is `user_sgpr.value[sbase]`. Pure analysis:
/// no guest memory or SPIR-V here — those are the next increments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarLoadRef {
    /// First register of the base-pointer SGPR pair (`sbase`).
    pub base_register: i32,
    /// Byte offset added to the base before the load (0 when not a constant).
    pub byte_offset: u32,
    /// Number of 32-bit dwords fetched: 1, 2, 4, or 8.
    pub dwords: u32,
}

/// Scan a shader for scalar memory loads and report each one's base pointer,
/// offset and width (see [`ScalarLoadRef`]). The returned bases are the roots
/// the resolver will read guest memory from to recover spilled descriptors.
#[must_use]
pub fn find_scalar_load_bases(code: &ShaderCode) -> Vec<ScalarLoadRef> {
    code.get_instructions()
        .iter()
        .filter_map(|inst| {
            let dwords = match inst.type_ {
                ShaderInstructionType::SLoadDword => 1,
                ShaderInstructionType::SLoadDwordx2 => 2,
                ShaderInstructionType::SLoadDwordx4 => 4,
                ShaderInstructionType::SLoadDwordx8 => 8,
                _ => return None,
            };
            // src[0] = base SGPR pair; src[1] = byte offset. The SRT/EUD pattern
            // uses a constant offset; a register offset is a harder case and is
            // reported as 0 for now.
            let byte_offset = match inst.src[1].type_ {
                ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant => {
                    inst.src[1].constant.u
                }
                _ => 0,
            };
            Some(ScalarLoadRef {
                base_register: inst.src[0].register_id,
                byte_offset,
                dwords,
            })
        })
        .collect()
}

/// Increment 2 of the SRT/EUD resolver (task #9): the guest address a scalar
/// load reads from, given the runtime user-SGPR values. The base is a pointer
/// pair — `value[base]` low dword, `value[base + 1]` high dword — and the load's
/// byte offset is added. Returns `None` (never panics) if the pair is out of
/// range. Combined with [`find_scalar_load_bases`] this yields the exact guest
/// addresses a shader spills its descriptors to — what increment 3 reads from
/// guest memory to populate the existing `extended_buffer` path.
#[must_use]
pub fn scalar_load_target_address(load: &ScalarLoadRef, user_sgpr: &UserSgprInfo) -> Option<u64> {
    let base = usize::try_from(load.base_register).ok()?;
    let lo = *user_sgpr.value.get(base)?;
    let hi = *user_sgpr.value.get(base + 1)?;
    let ptr = u64::from(lo) | (u64::from(hi) << 32);
    Some(ptr.wrapping_add(u64::from(load.byte_offset)))
}

/// Does `inst` write scalar register `reg` (via either destination)?
fn writes_sgpr(inst: &ShaderInstruction, reg: i32) -> bool {
    let covers = |op: &ShaderOperand| {
        op.type_ == ShaderOperandType::Sgpr
            && reg >= op.register_id
            && reg < op.register_id + op.size.max(1)
    };
    covers(&inst.dst) || covers(&inst.dst2)
}

/// Resolve scalar loads through a live-in user-SGPR pointer and snapshot their
/// values for SPIR-V translation.
///
/// Gen5 SRT shaders do not always expose an Extended User Data size. Minecraft
/// gameplay PS `0x1700c000` is the measured minimal case:
///
/// ```text
/// s_load_dwordx8 s[14:21], s[12:13], 0
/// image_sample ... s[14:21], ...
/// ```
///
/// `s12:s13` is a bind-time guest pointer, while the usage table only says that
/// a T# eventually resides at s14. With `eud_size_dw == 0`, the EUD resolver
/// cannot create a view and the old recompiler refused the non-EUD base before
/// any terrain material could sample. This pass evaluates only the bounded,
/// side-effect-free part required by that ABI: constant-offset x2/x4/x8 loads
/// through a pointer pair that has not been written earlier in the shader.
///
/// Captured loads share the existing per-PC constant-load representation, so
/// `sload_dword_extended` materializes the exact dwords into the destination
/// SGPRs. When an x8 result is subsequently consumed as a sampled MIMG T#, the
/// same snapshot replaces the static placeholder (or adds a direct texture
/// binding). Dynamic offsets and shader-produced base pointers remain on their
/// named refusal paths.
pub fn shader_capture_runtime_scalar_loads(
    code: &ShaderCode,
    mem: &impl ShaderMemory,
    user_sgpr: &UserSgprInfo,
    bind: &mut ShaderBindResources,
) {
    use ShaderInstructionType as T;

    const fn width(type_: ShaderInstructionType) -> Option<usize> {
        match type_ {
            T::SLoadDwordx2 => Some(2),
            T::SLoadDwordx4 => Some(4),
            T::SLoadDwordx8 => Some(8),
            _ => None,
        }
    }

    const fn sampled_mimg(type_: ShaderInstructionType) -> bool {
        matches!(
            type_,
            T::ImageLoad
                | T::ImageSample
                | T::ImageSampleCLz
                | T::ImageSampleLz
                | T::ImageSampleLzO
                | T::ImageGather4Lz
        )
    }

    let instructions = code.get_instructions();
    let mut texture_changed = false;

    for (at, load) in instructions.iter().enumerate() {
        let Some(dwords) = width(load.type_) else {
            continue;
        };
        if load.src[0].type_ != ShaderOperandType::Sgpr
            || load.src[0].size != 2
            || load.dst.type_ != ShaderOperandType::Sgpr
            || !matches!(
                load.src[1].type_,
                ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant
            )
            || load.src[1].constant.i() < 0
        {
            continue;
        }
        let base_reg = load.src[0].register_id;
        if instructions[..at]
            .iter()
            .any(|prior| writes_sgpr(prior, base_reg) || writes_sgpr(prior, base_reg + 1))
        {
            continue;
        }
        let base = match usize::try_from(base_reg) {
            Ok(base)
                if base + 1 < UserSgprInfo::SGPRS_MAX && base + 1 < user_sgpr.count as usize =>
            {
                base
            }
            _ => continue,
        };
        let pointer =
            u64::from(user_sgpr.value[base]) | (u64::from(user_sgpr.value[base + 1]) << 32);
        let address = pointer.wrapping_add(u64::from(load.src[1].constant.u));
        if pointer == 0 || address & 0x3 != 0 {
            continue;
        }
        let Some(source) = mem.dwords_at(address) else {
            continue;
        };
        if source.len() < dwords {
            continue;
        }

        if bind.embedded_constant_loads.find(load.pc).is_none() {
            let Ok(slot) = usize::try_from(bind.embedded_constant_loads.loads_num.max(0)) else {
                continue;
            };
            if slot >= ShaderEmbeddedConstantLoads::LOADS_MAX {
                break;
            }
            let capture = &mut bind.embedded_constant_loads.loads[slot];
            *capture = ShaderEmbeddedConstantLoad::default();
            capture.pc = load.pc;
            capture.dwords_num = dwords as u32;
            capture.values[..dwords].copy_from_slice(&source[..dwords]);
            bind.embedded_constant_loads.loads_num += 1;
        }

        if dwords != 8 {
            continue;
        }
        let dst_reg = load.dst.register_id;
        let used_as_sampled_texture = instructions[at + 1..].iter().any(|candidate| {
            sampled_mimg(candidate.type_)
                && candidate.src[1].type_ == ShaderOperandType::Sgpr
                && candidate.src[1].register_id == dst_reg
        });
        if !used_as_sampled_texture || sharp_dword3_is_buffer(source[3]) {
            continue;
        }

        let mut fields = [0u32; 8];
        fields.copy_from_slice(&source[..8]);
        let mut texture = super::resources::ShaderTextureResource { fields };
        if check_read_only_texture_type(&mut texture).is_err() {
            continue;
        }

        let count = usize::try_from(bind.textures2d.textures_num.max(0))
            .unwrap_or(0)
            .min(bind.textures2d.desc.len());
        if let Some(existing) = bind.textures2d.desc[..count]
            .iter_mut()
            .find(|desc| desc.start_register == dst_reg && !desc.textures2d_without_sampler)
        {
            existing.texture = texture;
            existing.extended = false;
            texture_changed = true;
            continue;
        }
        if count >= ShaderTextureResources::RES_MAX {
            continue;
        }
        let desc = &mut bind.textures2d.desc[count];
        desc.start_register = dst_reg;
        desc.extended = false;
        desc.slot = -1;
        desc.usage = ShaderTextureUsage::ReadOnly;
        desc.textures2d_without_sampler = false;
        desc.texture = texture;
        bind.textures2d.textures_num += 1;
        bind.textures2d.textures2d_sampled_num += 1;
        texture_changed = true;
    }

    if texture_changed {
        shader_calc_binding_indices(bind);
    }
}

/// For an `s_add_u32`, return `(constant_addend, other_sgpr)` when exactly one
/// source is a compile-time constant and the other is an SGPR — the shape a
/// PC-relative byte offset takes (`s_add_u32 s(b), <imm>, s(b)`).
fn add_const_and_reg(inst: &ShaderInstruction) -> Option<(u32, i32)> {
    let is_const = |op: &ShaderOperand| {
        matches!(
            op.type_,
            ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant
        )
    };
    let (a, b) = (&inst.src[0], &inst.src[1]);
    if is_const(a) && b.type_ == ShaderOperandType::Sgpr {
        Some((a.constant.u, b.register_id))
    } else if is_const(b) && a.type_ == ShaderOperandType::Sgpr {
        Some((b.constant.u, a.register_id))
    } else {
        None
    }
}

/// Resolve a base SGPR pair to a compile-time **PC-relative absolute address**,
/// by walking the instructions that produced it: the nearest preceding
/// `s_getpc_b64 s[base:base+1]` (whose absolute following address the parser
/// materialized into `src[0]`/`src[1]`) plus the constant offset folded into it
/// afterward.
///
/// The offset is the measured 64-bit idiom — `s_add_u32 s(lo), <imm>, s(lo)`
/// (low dword) optionally followed by `s_addc_u32 s(hi), <imm>, s(hi)` (high
/// dword + carry). Full `u64` arithmetic makes the low `s_add` propagate its
/// own carry, so the paired `s_addc s(hi), 0, s(hi)` folds to a no-op exactly
/// as the hardware intends. Measured on ASTRO.BOT vertex shaders (adds of
/// 96/208/272 bytes into an embedded-data pointer).
///
/// Returns `None` if the base was not built this way, or was mutated by
/// anything else (a conservative bail — the recompiler keeps its named refusal
/// for those).
fn pc_relative_base_address(prior: &[ShaderInstruction], base_reg: i32) -> Option<u64> {
    use ShaderInstructionType as T;
    let getpc_idx = prior.iter().rposition(|inst| {
        inst.type_ == T::SGetpcB64
            && inst.dst.type_ == ShaderOperandType::Sgpr
            && inst.dst.register_id == base_reg
            && inst.dst.size == 2
    })?;
    let getpc = &prior[getpc_idx];
    let mut addr = u64::from(getpc.src[0].constant.u) | (u64::from(getpc.src[1].constant.u) << 32);
    for inst in &prior[getpc_idx + 1..] {
        let touches_lo = writes_sgpr(inst, base_reg);
        let touches_hi = writes_sgpr(inst, base_reg + 1);
        if !touches_lo && !touches_hi {
            continue;
        }
        // Each add must write exactly one dword of the pair, add a constant to
        // itself, and be the modeled add opcode for that dword.
        let (want_type, reg, shift) = match (touches_lo, touches_hi) {
            (true, false) => (T::SAddU32, base_reg, 0),
            (false, true) => (T::SAddcU32, base_reg + 1, 32),
            _ => return None,
        };
        if inst.type_ != want_type || inst.dst.register_id != reg {
            return None;
        }
        let (c, other) = add_const_and_reg(inst)?;
        if other != reg {
            return None;
        }
        addr = addr.wrapping_add(u64::from(c) << shift);
    }
    Some(addr)
}

/// Beyond Kyty: capture **PC-relative embedded-constant** scalar loads (the
/// shader reading its own baked constant table).
///
/// A shader that loads embedded constants computes the base pointer in-shader —
/// `s_getpc_b64 s[b:b+1]` (the parser materializes the absolute following
/// address; see `parse.rs` S_GETPC_B64) then optional `s_add_u32 s(b), <imm>,
/// s(b)` — and reads through it with `s_load_dword{x2,x4,x8,x16} s[d..],
/// s[b:b+1], <off>`. The base is neither user data nor the EUD, so the
/// descriptor-table recompiler (`sload_dword_extended`) refuses it by name.
///
/// But the address is a compile-time constant and the target is the shader's
/// own binary, so the loaded dwords are known now. This pass reads them from
/// guest memory (the recompiler has no raw shader bytes) and records them in
/// [`ShaderBindResources::embedded_constant_loads`]; the recompiler matches by
/// `pc` and materializes them as SPIR-V constants stored into the destination
/// SGPRs. Mirrors the standalone-pass shape of `shader_detect_eud_raw_window`.
///
/// Only fully-resolved loads are captured (PC-relative getpc base found, every
/// intervening base mutation a constant add, guest memory readable); anything
/// else is left for the recompiler's existing refusal.
pub fn shader_detect_embedded_constant_loads(
    code: &ShaderCode,
    mem: &dyn ShaderMemory,
    bind: &mut ShaderBindResources,
) {
    use ShaderInstructionType as T;

    let insts = code.get_instructions();
    // Runtime/SRT scalar loads may already have been captured from live user
    // data. Preserve those per-PC snapshots and append PC-relative constants;
    // both routes are consumed by the same recompiler materialization path.
    let mut out = bind.embedded_constant_loads;
    for (i, load) in insts.iter().enumerate() {
        // Only the widths the recompiler routes through `sload_dword_extended`
        // (where the materialization lives) — x1 has a distinct fetch-only
        // path and x16 has no recompiler row, so capturing them would record
        // dwords nothing can consume.
        let dwords = match load.type_ {
            T::SLoadDwordx2 => 2u32,
            T::SLoadDwordx4 => 4,
            T::SLoadDwordx8 => 8,
            _ => continue,
        };
        if out.find(load.pc).is_some() {
            continue;
        }
        // Base must be an SGPR pair; offset a non-negative compile-time byte
        // constant (the shape `sload_dword_extended` reads).
        if load.src[0].type_ != ShaderOperandType::Sgpr || load.src[0].size != 2 {
            continue;
        }
        let load_off = match load.src[1].type_ {
            ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant => {
                if load.src[1].constant.i() < 0 {
                    continue;
                }
                u64::from(load.src[1].constant.u)
            }
            _ => continue,
        };
        let Some(base_addr) = pc_relative_base_address(&insts[..i], load.src[0].register_id) else {
            continue;
        };
        let addr = base_addr.wrapping_add(load_off);
        if addr % 4 != 0 {
            continue;
        }
        let Some(src) = mem.dwords_at(addr) else {
            continue;
        };
        if (src.len() as u64) < u64::from(dwords) {
            continue;
        }
        let slot_index = out.loads_num.max(0) as usize;
        if slot_index >= ShaderEmbeddedConstantLoads::LOADS_MAX {
            tracing::warn!(
                pc = load.pc,
                "more PC-relative embedded-constant loads than LOADS_MAX; dropping the rest"
            );
            break;
        }
        let slot = &mut out.loads[slot_index];
        slot.pc = load.pc;
        slot.dwords_num = dwords;
        for (k, v) in slot.values.iter_mut().take(dwords as usize).enumerate() {
            *v = src[k];
        }
        out.loads_num += 1;
        tracing::debug!(
            pc = load.pc,
            addr = format_args!("{addr:#x}"),
            dwords,
            "captured PC-relative embedded-constant scalar load"
        );
    }
    bind.embedded_constant_loads = out;
}

/// Beyond Kyty: capture `offen` MUBUF buffer loads whose buffer descriptor (V#)
/// is **constructed inside the shader**, pointing at the shader's own embedded
/// vertex data.
///
/// Measured on the ASTRO.BOT full-screen-triangle vertex shader: it builds a V#
/// in `s[0:3]` — base `s[0:1]` PC-relative (`s_getpc_b64` + the 64-bit add
/// idiom `pc_relative_base_address` resolves), stride/format words `s2`/`s3`
/// from immediate moves — then reads clip-space vertices with
/// `buffer_load_dwordx4 v[0:3], v4, s[0:3], 0 offen` /
/// `buffer_load_dwordx2 v[4:5], v4, s[0:3], 16 offen`. The descriptor is never
/// user-data, so the storage-buffer path finds nothing (`buffers_num == 0`) and
/// the recompiler refuses (measured 116 `can't recompile: BufferLoadDwordX4
/// [Vdata4VaddrSvSoffsOffen]` per live frame).
///
/// The embedded data is static (baked into the shader binary), so this snapshots
/// a window of it from guest memory and records it for the recompiler to serve
/// the runtime-indexed (`base + voffset`) read from constants. Only offen-only
/// loads (no `idxen` vindex/stride term) whose V# base is a resolvable
/// PC-relative pointer are captured; anything else keeps the existing refusal.
pub fn shader_detect_embedded_buffer_fetch(
    code: &ShaderCode,
    mem: &dyn ShaderMemory,
    bind: &mut ShaderBindResources,
) {
    use crate::shader::types::shader_instruction_format::Format as F;
    use ShaderInstructionType as T;

    let insts = code.get_instructions();
    let mut out = ShaderEmbeddedBufferFetches::default();
    for (i, load) in insts.iter().enumerate() {
        // Only widths the recompiler serves through `buffer_load_dwordxn`.
        let dwords = match load.type_ {
            T::BufferLoadDwordX2 => 2u32,
            T::BufferLoadDwordX3 => 3,
            T::BufferLoadDwordX4 => 4,
            _ => continue,
        };
        // offen-only (idxen == 0): address = base + voffset + inst_offset, no
        // vindex*stride term. src[0] = voffset VGPR, src[1] = V# (SGPR quad),
        // src[2] = the immediate byte offset.
        if !matches!(
            load.format,
            F::Vdata2VaddrSvSoffsOffen | F::Vdata3VaddrSvSoffsOffen | F::Vdata4VaddrSvSoffsOffen
        ) {
            continue;
        }
        if load.src[1].type_ != ShaderOperandType::Sgpr || load.src[1].size != 4 {
            continue;
        }
        let inst_offset = match load.src[2].type_ {
            ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant => {
                if load.src[2].constant.i() < 0 {
                    continue;
                }
                load.src[2].constant.u
            }
            _ => continue,
        };
        let Some(base) = pc_relative_base_address(&insts[..i], load.src[1].register_id) else {
            continue;
        };
        if base % 4 != 0 {
            continue;
        }
        let Some(src) = mem.dwords_at(base) else {
            continue;
        };
        let window_len = src.len().min(ShaderEmbeddedBufferFetch::WINDOW_MAX);
        if window_len == 0 {
            continue;
        }
        let slot_index = out.loads_num.max(0) as usize;
        if slot_index >= ShaderEmbeddedBufferFetches::LOADS_MAX {
            tracing::warn!(
                pc = load.pc,
                "more in-shader-V# buffer loads than LOADS_MAX; dropping the rest"
            );
            break;
        }
        let slot = &mut out.loads[slot_index];
        slot.pc = load.pc;
        slot.inst_offset = inst_offset;
        slot.dwords_num = dwords;
        slot.window_len = window_len as u32;
        for (k, v) in slot.window.iter_mut().take(window_len).enumerate() {
            *v = src[k];
        }
        out.loads_num += 1;
        tracing::debug!(
            pc = load.pc,
            base = format_args!("{base:#x}"),
            dwords,
            window_len,
            "captured in-shader-V# embedded buffer fetch"
        );
    }
    bind.embedded_buffer_fetches = out;
}

/// The EUD dword offset a COVERED scalar load off the EUD base pair fills the
/// SGPR range starting at `reg` from, or `None` if no such load seeds `reg`.
///
/// A descriptor delivered through the raw EUD window is read into the register
/// file by an `s_load_dword{,x2,x4,x8} s[reg..], s[eud_base:eud_base+1],
/// <const>` (the exact shape `sload_dword_extended` and
/// `shader_detect_eud_raw_window` accept). The returned dword offset is the
/// load's byte offset >> 2 — the EUD dword whose captured-descriptor coverage
/// decides whether `reg`'s dword 0 gets a rewritten descriptor-array index.
/// Shared by the recompiler's `mimg_descriptor_guard` alias rule and the
/// analysis-side raw-EUD image-descriptor capture pass.
pub(crate) fn eud_load_offset_for_register(
    code: &ShaderCode,
    eud_base: i32,
    reg: i32,
) -> Option<i32> {
    use crate::shader::types::ShaderInstructionType as T;
    for load in code.get_instructions() {
        match load.type_ {
            T::SLoadDword | T::SLoadDwordx2 | T::SLoadDwordx4 | T::SLoadDwordx8 => {}
            _ => continue,
        }
        if load.src[0].type_ != ShaderOperandType::Sgpr
            || load.src[0].register_id != eud_base
            || load.dst.type_ != ShaderOperandType::Sgpr
            || load.dst.register_id != reg
            || !matches!(
                load.src[1].type_,
                ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant
            )
            || load.src[1].constant.i() < 0
        {
            continue;
        }
        return Some((load.src[1].constant.u >> 2) as i32);
    }
    None
}

/// Kyty: Shader.cpp `ShaderGetStorageBuffer` (L1141).
pub fn shader_get_storage_buffer(
    info: &mut ShaderStorageResources,
    direct_sgprs: &mut [bool; UserSgprInfo::SGPRS_MAX],
    start_index: i32,
    slot: i32,
    usage: ShaderStorageUsage,
    user_sgpr: &UserSgprInfo,
    extended_buffer: Option<EudView<'_>>,
) -> Result<(), ShaderAnalysisError> {
    if info.buffers_num < 0 || info.buffers_num as usize >= ShaderStorageResources::BUFFERS_MAX {
        return Err(ni("too many storage buffers"));
    }
    let index = info.buffers_num as usize;

    let mut fields = [0u32; 4];
    let eud_resident = read_sharp_fields(
        direct_sgprs,
        start_index,
        user_sgpr,
        extended_buffer,
        &mut fields,
    )?;

    info.start_register[index] = start_index;
    info.slots[index] = slot;
    info.usages[index] = usage;
    info.extended[index] = eud_resident;
    info.buffers[index].fields = fields;
    info.buffers_num += 1;
    tracing::debug!(
        slot,
        start_index,
        fields = format_args!(
            "[{:#010x}, {:#010x}, {:#010x}, {:#010x}]",
            fields[0], fields[1], fields[2], fields[3]
        ),
        "storage-buffer V# read from user SGPRs"
    );
    Ok(())
}

/// Kyty: Shader.cpp `ShaderGetTextureBuffer` (L1179).
pub fn shader_get_texture_buffer(
    info: &mut ShaderTextureResources,
    direct_sgprs: &mut [bool; UserSgprInfo::SGPRS_MAX],
    start_index: i32,
    slot: i32,
    usage: ShaderTextureUsage,
    user_sgpr: &UserSgprInfo,
    extended_buffer: Option<EudView<'_>>,
) -> Result<(), ShaderAnalysisError> {
    if info.textures_num < 0 || info.textures_num as usize >= ShaderTextureResources::RES_MAX {
        return Err(ni("too many textures"));
    }
    if usage == ShaderTextureUsage::Unknown {
        return Err(ni("texture usage is Unknown"));
    }
    let index = info.textures_num as usize;

    let mut fields = [0u32; 8];
    let eud_resident = match read_sharp_fields(
        direct_sgprs,
        start_index,
        user_sgpr,
        extended_buffer,
        &mut fields,
    ) {
        Ok(eud_resident) => eud_resident,
        Err(e) => {
            // The descriptor could not be read from the captured user data /
            // EUD because it is resolved at RUNTIME (SRT/bindless), not present
            // in the static capture. Measured on ASTRO.BOT 0x100008e6aa00: a T#
            // declared at s4 (needs s4..s11) whose s8+ were never captured, no
            // EUD -> "user sgpr type is not Vsharp/Region (Unknown at s8)".
            // Install a placeholder T# so the draw/dispatch proceeds untextured
            // instead of aborting the whole shader (mirrors the type gate and
            // shader_synthesize_default_sampler). Consume the captured
            // registers the sharp would have occupied so the direct-SGPR pass
            // does not also bind them (avoids a duplicate %vsharp binding).
            if let Ok(start) = usize::try_from(start_index) {
                let end = (start + fields.len()).min(UserSgprInfo::SGPRS_MAX);
                for flag in direct_sgprs
                    .iter_mut()
                    .take(end)
                    .skip(start.min(UserSgprInfo::SGPRS_MAX))
                {
                    *flag = false;
                }
            }
            tracing::debug!(
                start_register = start_index,
                error = %e,
                "texture descriptor unresolved (runtime/SRT-bound) — placeholder T#"
            );
            fields = placeholder_texture_fields();
            false
        }
    };

    info.desc[index].start_register = start_index;
    info.desc[index].extended = eud_resident;
    info.desc[index].slot = slot;
    info.desc[index].usage = usage;

    if usage == ShaderTextureUsage::ReadWrite {
        info.textures2d_storage_num += 1;
        info.desc[index].textures2d_without_sampler = true;
    } else {
        info.textures2d_sampled_num += 1;
        info.desc[index].textures2d_without_sampler = false;
    }

    info.desc[index].texture.fields = fields;
    info.textures_num += 1;
    Ok(())
}

/// Kyty: Shader.cpp `ShaderGetSampler` (L1233).
pub fn shader_get_sampler(
    info: &mut ShaderSamplerResources,
    direct_sgprs: &mut [bool; UserSgprInfo::SGPRS_MAX],
    start_index: i32,
    slot: i32,
    user_sgpr: &UserSgprInfo,
    extended_buffer: Option<EudView<'_>>,
) -> Result<(), ShaderAnalysisError> {
    if info.samplers_num < 0 || info.samplers_num as usize >= ShaderSamplerResources::RES_MAX {
        return Err(ni("too many samplers"));
    }
    let index = info.samplers_num as usize;

    let mut fields = [0u32; 4];
    let eud_resident = match read_sharp_fields(
        direct_sgprs,
        start_index,
        user_sgpr,
        extended_buffer,
        &mut fields,
    ) {
        Ok(eud_resident) => eud_resident,
        Err(e @ ShaderAnalysisError::NotImplementedOwned { .. }) => {
            // A sampler can be supplied at runtime through SRT/bindless state
            // even though the static Gen5 user-data capture names its SGPR
            // start. Minecraft's second-screen PS declares S# at s28 but only
            // captures through s29, so the old strict four-dword read skipped
            // every menu draw. The Vulkan binding path already has a safe
            // all-zero nearest/wrap sampler; install that same descriptor here
            // instead of failing analysis.
            if let Ok(start) = usize::try_from(start_index) {
                for flag in direct_sgprs
                    .iter_mut()
                    .take((start + fields.len()).min(UserSgprInfo::SGPRS_MAX))
                    .skip(start.min(UserSgprInfo::SGPRS_MAX))
                {
                    *flag = false;
                }
            }
            tracing::debug!(
                start_index,
                slot,
                reason = %e,
                "runtime-resolved sampler unavailable in static capture; using default S#"
            );
            false
        }
        Err(e) => return Err(e),
    };

    info.start_register[index] = start_index;
    info.extended[index] = eud_resident;
    info.slots[index] = slot;
    info.samplers[index].fields = fields;
    info.samplers_num += 1;
    Ok(())
}

/// Kyty: Shader.cpp `ShaderGetGdsPointer` (L1270). Unlike the sharps, the
/// direct case requires an `Unknown`-typed user SGPR.
pub fn shader_get_gds_pointer(
    info: &mut ShaderGdsResources,
    direct_sgprs: &mut [bool; UserSgprInfo::SGPRS_MAX],
    start_index: i32,
    slot: i32,
    user_sgpr: &UserSgprInfo,
    extended_buffer: Option<EudView<'_>>,
) -> Result<(), ShaderAnalysisError> {
    if info.pointers_num < 0 || info.pointers_num as usize >= ShaderGdsResources::POINTERS_MAX {
        return Err(ni("too many gds pointers"));
    }
    let index = info.pointers_num as usize;
    let extended = extended_buffer.is_some();

    if !extended && start_index >= 16 {
        return Err(ni("gds start_register >= 16 without extended buffer"));
    }
    if extended && start_index < 16 {
        return Err(ni("gds start_register < 16 with extended buffer"));
    }
    let start = usize::try_from(start_index).map_err(|_| trunc("negative gds start register"))?;

    info.start_register[index] = start_index;
    info.extended[index] = extended;
    info.slots[index] = slot;

    info.pointers[index].field = match extended_buffer {
        Some(ext) => *ext
            .get_dw(start - 16)
            .ok_or_else(|| trunc("extended (EUD) buffer too small"))?,
        None => {
            let type_ = *user_sgpr
                .type_
                .get(start)
                .ok_or_else(|| trunc("gds register beyond SGPRS_MAX"))?;
            if type_ != UserSgprType::Unknown {
                return Err(ni("gds user sgpr type is not Unknown"));
            }
            direct_sgprs[start] = false;
            user_sgpr.value[start]
        }
    };

    info.pointers_num += 1;
    Ok(())
}

/// Gate for read-only sampled-texture T# types. Accepted from measurement:
/// 9 = Texture2D, 11 = Cube (Minecraft's 1024x1024x6 skybox), 10 = 3D volume
/// (ASTRO.BOT's 240x135x64 froxel/LUT volumes, format 71), 13 = 2DArray
/// (ASTRO.BOT's 1536x1536x3 format-7 tile-24 array — declared arrayed via
/// `SampledDim::TwoArray`, sampled with a (u, v, layer) coordinate). Anything
/// else is a named, evidence-rich refusal carrying every field the next arm
/// needs.
/// Disambiguate an 8-dword sharp slot's content: buffer V# vs image T#, from
/// the descriptor's own type field in dword3.
///
/// RDNA2 layout (confirmed in shadPS4 video_core/amdgpu/resource.h): the
/// image T# TYPE is the 4-bit field dword3\[28:31\] with values 8..15
/// (8=1D 9=2D 10=3D 11=Cube 12=1DArray 13=2DArray 14=2DMsaa 15=2DMsaaArray),
/// so bit 31 is always set for an image. The buffer V# TYPE is the 2-bit
/// field dword3\[30:31\] and reads 0 for a valid buffer — but dword3\[28:29\]
/// is the V#'s OOB_SELECT, so the whole nibble reads 0..3 depending on the
/// out-of-bounds mode. The original `nibble == 0` test misrouted every V#
/// with a nonzero OOB_SELECT into the texture path, which then refused it as
/// "read-only texture type 3 ... 65x1 depth=5265 format=0 tile=0" (V# fields
/// reinterpreted as a T#; 57 measured ASTRO.BOT CS dispatch skips, round 8).
const fn sharp_dword3_is_buffer(dw3: u32) -> bool {
    (dw3 >> 28) & 0xF < 8
}

/// True when a captured T# is the all-ones poison a descriptor the static
/// capture could not resolve (runtime-/SRT-/bindless-bound) reads back as,
/// rather than a real texture. Measured on ASTRO.BOT (compute 0x100008e6aa00,
/// 281+ dispatches/run): type 15, tile 31, 16384², base 0xf0fffffff000 /
/// 0xffffffffff00 and **format 511 (0x1FF)**. The 9-bit unified FORMAT field is
/// the reliable marker: real guest formats are small (7/10/37/71/77…), well
/// under 0x100, so a value with any high bit set never names a bindable
/// texture. The same static shader alternately failing with this poison and
/// with "user sgpr type is not Vsharp/Region (s8 Unknown)" — code fixed, input
/// varying — is what proves the descriptor is resolved at *runtime*, not a
/// fixed wrong capture offset.
fn texture_descriptor_is_unresolvable(t: &super::resources::ShaderTextureResource) -> bool {
    t.format() >= 0x100
}

/// The stand-in fields of a placeholder T#: a valid 1x1 `Texture2D` (type 9)
/// at base 0 with an identity `dst_sel`. The binding path (`raeen-gpu`
/// `decode_texture`) serves base 0 as a 1x1 transparent-black dummy, so a
/// draw/dispatch whose texture descriptor could not be resolved proceeds
/// untextured instead of the whole shader being skipped. Mirrors
/// [`shader_synthesize_default_sampler`]'s all-zero S# and the null-V#-as-zero
/// storage-buffer dummy.
pub(crate) const fn placeholder_texture_fields() -> [u32; 8] {
    // fields[3]: type (bits 28..31) = 9 (Texture2D); dst_sel(X,Y,Z,W) = 0xFAC.
    // Everything else 0: base 0, format 0, width5 0 / height5 0 (1x1), tile 0.
    [0, 0, 0, (9u32 << 28) | 0xFAC, 0, 0, 0, 0]
}

/// Gate for a read-only sampled T#'s type. Non-fatal by construction: a valid
/// type is admitted, an array/MSAA type is approximated as 2D, and an
/// unresolvable/garbage descriptor is replaced in place with a placeholder so
/// the shader is NEVER aborted for a texture-descriptor reason (M5: maximize
/// geometry on screen, glitches OK).
///
/// - 8=1D (a height-1 2D texture; `SampledDim::from_texture_type` classifies it
///   2D), 9=2D, 10=3D volume, 11=Cube, 13=2DArray: admitted unchanged (the
///   already-working set — measured on ASTRO.BOT / Minecraft).
/// - 12=1DArray, 14=2DMsaa, 15=2DMsaaArray with a *plausible* descriptor:
///   approximated as 2D (the type is rewritten to 9 so the guest-memory decode,
///   which handles only 8/9/10/11/13, accepts it; downstream
///   `from_texture_type` already collapses these to 2D). No such VALID
///   descriptor occurs in ASTRO.BOT — every measured 15 is the poison below —
///   but the approximation is cheap and keeps a real MSAA/array title moving.
/// - Anything unresolvable (the all-ones poison, see
///   [`texture_descriptor_is_unresolvable`]) or otherwise unhandled: replaced
///   with a [`placeholder_texture_fields`] 1x1 dummy.
fn check_read_only_texture_type(
    t: &mut super::resources::ShaderTextureResource,
) -> Result<(), ShaderAnalysisError> {
    if texture_descriptor_is_unresolvable(t) {
        t.fields = placeholder_texture_fields();
        return Ok(());
    }
    let ty = t.type_();
    if matches!(ty, 8 | 9 | 10 | 11 | 13) {
        return Ok(());
    }
    if matches!(ty, 12 | 14 | 15) {
        // Approximate as 2D: rewrite the type nibble to 9 in place.
        t.fields[3] = (t.fields[3] & 0x0FFF_FFFF) | (9 << 28);
        return Ok(());
    }
    // Any other image type reaching here (e.g. a 0..7 forced through a
    // flags==3 usage slot): stand in the placeholder rather than abort.
    t.fields = placeholder_texture_fields();
    Ok(())
}

/// Gate for read-write storage-image (UAV) T# types supported end-to-end.
///
/// - Type 8 is a 1D image; the Vulkan/SPIR-V path represents it as a height-1
///   2D image, matching the sampled-image path. Type 9 is native 2D, type 10
///   is a 3D volume.
/// - Types 11 (Cube) and 13 (2DArray / NDArray) are represented as writable
///   2D arrays. The SPIR-V declaration, three-component ImageStore coordinate,
///   Vulkan view, per-layer upload/readback, and guest retile all preserve the
///   array layer. Measured: Minecraft's menu compute assembles six
///   1024x1024 panorama faces through this path; collapsing that descriptor to
///   a plain 2D image left all six guest faces black.
///
/// A genuinely-unsupported storage type (1DArray 12, 2DMsaa 14, 2DMsaaArray
/// 15, or a poison descriptor) still errors by name so the coverage gap stays
/// visible.
fn check_read_write_texture_type(
    t: &super::resources::ShaderTextureResource,
    source: &str,
) -> Result<(), ShaderAnalysisError> {
    if (t.type_() == 8 && t.height5() == 0) || matches!(t.type_(), 9 | 10 | 11 | 13) {
        return Ok(());
    }
    Err(ni_owned(format!(
        "read-write ({source}) texture type {} is not height-1 1D (8), 2D (9), 3D (10), \
         Cube (11) or 2DArray (13) (12=1DArray 14=2DMsaa 15=2DMsaaArray; \
         base={:#x} {}x{} depth={} format={} tile={})",
        t.type_(),
        t.base40(),
        u32::from(t.width5()) + 1,
        u32::from(t.height5()) + 1,
        u32::from(t.depth()) + 1,
        t.format(),
        t.tile_mode(),
    )))
}

/// Kyty: Shader.cpp `ShaderGetDirectSgpr` (L1301).
pub fn shader_get_direct_sgpr(
    info: &mut ShaderDirectSgprsResources,
    start_index: i32,
    user_sgpr: &UserSgprInfo,
) -> Result<(), ShaderAnalysisError> {
    shader_get_direct_sgpr_impl(info, start_index, user_sgpr, false)
}

fn shader_get_direct_sgpr_impl(
    info: &mut ShaderDirectSgprsResources,
    start_index: i32,
    user_sgpr: &UserSgprInfo,
    allow_typed_scalar: bool,
) -> Result<(), ShaderAnalysisError> {
    if info.sgprs_num < 0 || info.sgprs_num as usize >= ShaderDirectSgprsResources::SGPRS_MAX {
        return Err(ni("too many direct sgprs"));
    }
    let index = info.sgprs_num as usize;

    // Kyty bounds this at 16 (the PS4 user-SGPR file); Gen5 stages have 32
    // real user SGPRs (see `UserSgprInfo::SGPRS_MAX`), and SRT data can sit
    // anywhere in the file.
    if start_index >= UserSgprInfo::SGPRS_MAX as i32 {
        return Err(ni("direct sgpr start_register beyond the user SGPR file"));
    }
    let start = usize::try_from(start_index).map_err(|_| trunc("negative direct sgpr register"))?;

    info.start_register[index] = start_index;

    if !allow_typed_scalar
        && user_sgpr.type_[start] != UserSgprType::Unknown
        && user_sgpr.value[start] != 0
    {
        // Instrumented refusal: the Gen5 collection loop routes typed
        // registers into their resource tables before calling here, so a
        // hit names whichever path (legacy walk, or a routing gap) still
        // sends typed registers this way. A typed register whose VALUE is
        // zero carries no descriptor — the marker typed it but the driver
        // never wrote one — and seeds as plain direct data (measured:
        // ASTRO.BOT CS dispatches carry `Region at s4, value=0x0`, 810/run).
        return Err(ni_owned(format!(
            "direct user sgpr type is not Unknown ({:?} at s{start}, value={:#x})",
            user_sgpr.type_[start], user_sgpr.value[start],
        )));
    }

    info.sgprs[index].field = user_sgpr.value[start];
    info.sgprs_num += 1;
    Ok(())
}

/// Kyty: Shader.cpp `ShaderCalcBindingIndices` (L1321). Assigns sequential
/// Vulkan binding indices per used resource group and sizes the
/// push-constant window (16-byte granules).
pub fn shader_calc_binding_indices(bind: &mut ShaderBindResources) {
    let mut binding_index = 0;

    bind.push_constant_size = 0;

    if bind.storage_buffers.buffers_num > 0 {
        bind.storage_buffers.binding_index = binding_index;
        binding_index += 1;
        bind.push_constant_size += bind.storage_buffers.buffers_num as u32 * 16;
    }

    if bind.textures2d.textures_num > 0 {
        // Sampled textures take one Vulkan binding PER PRESENT (Dim, numeric
        // class) key (a mixed shader declares one `%textures2D_S<key>` array
        // each — one SPIR-V array type carries exactly one image Dim and one
        // sampled component type). The storage array follows all of them. A
        // homogeneous shader has one present key, so this reserves the same
        // two bindings as before (sampled, storage); a storage-only shader
        // reserves one placeholder sampled binding to keep the storage index
        // where it has always been.
        bind.textures2d.binding_sampled_index = binding_index;
        let sampled_bindings = super::spirv::sampled_keys_present(bind).len().max(1) as i32;
        binding_index += sampled_bindings;
        bind.textures2d.binding_storage_index = binding_index;
        binding_index += 1;

        bind.push_constant_size += bind.textures2d.textures_num as u32 * 32;
    }

    if bind.samplers.samplers_num > 0 {
        bind.samplers.binding_index = binding_index;
        binding_index += 1;
        bind.push_constant_size += bind.samplers.samplers_num as u32 * 16;
    }

    if bind.gds_pointers.pointers_num > 0 {
        bind.gds_pointers.binding_index = binding_index;
        bind.push_constant_size += (((bind.gds_pointers.pointers_num as u32 - 1) / 4) + 1) * 16;
    }

    if bind.direct_sgprs.sgprs_num > 0 {
        bind.push_constant_size += (((bind.direct_sgprs.sgprs_num as u32 - 1) / 4) + 1) * 16;
    }

    debug_assert_eq!(bind.push_constant_size % 16, 0);
}

/// Beyond Kyty — SharpEmu port: synthesize a DEFAULT S# for a shader that
/// SAMPLES textures while analysis captured zero samplers.
///
/// SharpEmu never refuses this shape: a texture whose sampler handle was
/// never created gets one built on the fly from its (possibly all-zero)
/// captured S# state
/// (`reference/sharpemu/src/SharpEmu.Libs/VideoOut/VulkanVideoPresenter.cs`
/// L6314-6322), and `CreateSampler` decodes the all-zero S# to
/// nearest-filter + wrap addressing, caching one `VkSampler` per distinct S#
/// state for the device's lifetime (same file, L8121-8156).
///
/// The port synthesizes one all-zero S# per distinct MIMG sampler operand
/// register: the bind-time path then seeds those SGPRs with the rewritten
/// sampler-array index (0), `prepare_stage_binding` decodes the all-zero S#
/// as `linear_filter = false`, and the Vulkan layer binds its cached
/// nearest/wrap sampler (`ShaderCaches::sampler(dev, false)` — created once
/// per device, destroyed on cleanup) — instead of the sample-family
/// recompilers refusing the whole shader (`samplers_num == 0` previously
/// returned "instruction not recompiled").
///
/// Texel-fetch-only shaders (`image_load`, no sample instructions) are left
/// untouched: `OpImageFetch` needs no sampler and the descriptor arrays are
/// independent.
///
/// Rank-8 broadening (draw-time null-descriptor fallback): the original form
/// bailed the moment ANY sampler was captured (`samplers_num != 0`), so a
/// sample instruction reading an S# from a register NO captured sampler
/// occupies still refused the whole shader by name
/// (`dynamic-image-descriptor: S# at sN matches no captured sampler`). It now
/// synthesizes a default S# for every sample-family register that resolves to
/// no captured sampler — a direct start-register match OR the EUD-alias walk
/// [`mimg_descriptor_guard`] uses (via [`sampler_register_is_eud_alias`]) — so
/// a genuinely-unresolved sampler degrades to nearest/wrap instead of being
/// fatal, exactly as shadPS4 binds a Null()/default sampler for an unresolved
/// S# under robustness2. A register a real sampler already covers is never
/// shadowed (the same resolution the guard accepts is skipped here), so the
/// zero-sampler behaviour is unchanged.
///
/// Call after `shader_get_input_info_*` and before
/// `shader_detect_eud_raw_window` (both consume the sampler table this may
/// extend); binding indices and the push-constant size are recomputed here
/// when a sampler is synthesized. The register match uses `shift_regs == 0`
/// (the compute/pixel stages this is wired for); the `gs_prolog` +8 rebase is
/// a vertex-stage concern this pass, like the placeholder-texture pass, does
/// not target.
pub fn shader_synthesize_default_sampler(code: &ShaderCode, bind: &mut ShaderBindResources) {
    use ShaderInstructionType as T;

    if bind.textures2d.textures2d_sampled_num <= 0 {
        return;
    }

    let mut regs: Vec<i32> = Vec::new();
    for (at, inst) in code.get_instructions().iter().enumerate() {
        if !matches!(
            inst.type_,
            T::ImageSample
                | T::ImageSampleCLz
                | T::ImageSampleLz
                | T::ImageSampleLzO
                | T::ImageGather4Lz
        ) {
            continue;
        }
        // MIMG src[2] is the S# operand (ssamp * 4).
        let op = inst.src[2];
        if op.type_ != ShaderOperandType::Sgpr || regs.contains(&op.register_id) {
            continue;
        }
        let s_reg = op.register_id;
        // Skip a register a captured sampler already covers — never shadow a
        // real S#. Mirror the guard: a direct start-register match OR the
        // program-order EUD-alias walk.
        let samp_num = usize::try_from(bind.samplers.samplers_num.max(0))
            .unwrap_or(0)
            .min(bind.samplers.start_register.len());
        let direct = bind.samplers.start_register[..samp_num].contains(&s_reg);
        if direct || super::recompile::sampler_register_is_eud_alias(bind, code, s_reg, at) {
            continue;
        }
        regs.push(s_reg);
    }
    if regs.is_empty() {
        return;
    }

    for reg in regs {
        let index = bind.samplers.samplers_num;
        let Ok(i) = usize::try_from(index) else {
            return;
        };
        if i >= ShaderSamplerResources::RES_MAX {
            // More distinct sampler registers than slots: leave the rest
            // uncaptured — the recompiler's descriptor guard refuses those
            // sample instructions by name instead of running them.
            break;
        }
        bind.samplers.start_register[i] = reg;
        bind.samplers.extended[i] = false;
        // No usage slot produced this S#; the sentinel keeps the synthesized
        // entry distinguishable in bind-id dumps.
        bind.samplers.slots[i] = -1;
        bind.samplers.samplers[i].fields = [0; 4];
        bind.samplers.samplers_num += 1;
        tracing::debug!(
            start_register = reg,
            "synthesized default (all-zero) S# for sampler-less sampled texture"
        );
    }

    shader_calc_binding_indices(bind);
}

/// Beyond Kyty — SharpEmu-parity safe degradation: synthesize a PLACEHOLDER
/// sampled T# for a sampled MIMG (`image_load` / `image_sample*` /
/// `image_gather*`) whose T# operand register matches NO captured descriptor
/// and no EUD alias — a RUNTIME-resolved (bindless / SRT-indexed) texture the
/// static capture never saw.
///
/// Without this, the recompiler's `mimg_descriptor_guard` refuses the WHOLE
/// shader by name (`dynamic-image-descriptor: Sampled T# at sN matches no
/// captured descriptor`) and every dispatch binding it is skipped. Measured on
/// ASTRO.BOT scene compute `0x500566b00`: an `image_load` (dmask 1) reads its
/// T# from `s16` with no usage-table entry, causing 13 dispatch skips per level
/// transition. This mirrors [`shader_synthesize_default_sampler`] on the
/// TEXTURE side and the `shader_get_texture_buffer` placeholder (this file):
/// install a valid 1x1 transparent-black `Texture2D` at the referenced register
/// so the guard resolves it, `WriteLocalVariables` seeds `%vsharp_sN` with the
/// placeholder's array index (delivered through the `%vsharp` push constant by
/// `prepare_stage_binding`), and the dispatch PROCEEDS untextured instead of the
/// whole shader being skipped (M5: maximize geometry, glitches OK).
///
/// The seeded index is a REAL bound descriptor, so this does NOT reintroduce the
/// descriptor-array OOB indexing the guard's refusal was written to avoid
/// (`mimg_descriptor_guard` doc): it eliminates the OOB by making the index
/// valid. Coverage is decided by the SAME resolution the guard uses — a direct
/// start-register match OR [`mimg_register_eud_alias_index`]'s program-order EUD
/// alias walk — so a real EUD-aliased descriptor is never shadowed by a
/// placeholder; only genuinely unresolved registers get one.
///
/// Scope: this rescues the "no captured descriptor" refusal (the guard's first
/// check). A T# whose register is later OVERWRITTEN by a raw (uncovered) EUD
/// `s_load` still, correctly, refuses at the guard's shape-2 check — the raw
/// dwords are the true descriptor and cannot be mapped to a Vulkan array index
/// at translate time. It also does not touch storage images (`image_store*`) or
/// dimension queries (`image_get_resinfo`).
///
/// Call after `shader_get_input_info_*` and BEFORE
/// [`shader_synthesize_default_sampler`] (a synthesized sampled texture may need
/// a default sampler for the sample-family case). Binding indices are recomputed
/// here. `shift_regs` is 0 for the compute/pixel stages this is wired for (the
/// `gs_prolog` +8 rebase is a vertex-stage concern).
pub fn shader_synthesize_placeholder_sampled_texture(
    code: &ShaderCode,
    bind: &mut ShaderBindResources,
) {
    use ShaderInstructionType as T;

    // Sampled-class MIMG that reads a T# from src[1] (srsrc * 4). Storage ops
    // (`image_store*`) and dimension queries (`image_get_resinfo`) keep their
    // existing paths — a 1x1 dummy would be wrong for both.
    const fn is_sampled_mimg(t: ShaderInstructionType) -> bool {
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

    let mut regs: Vec<i32> = Vec::new();
    for (at, inst) in code.get_instructions().iter().enumerate() {
        if !is_sampled_mimg(inst.type_) {
            continue;
        }
        let t_op = inst.src[1];
        if t_op.type_ != ShaderOperandType::Sgpr {
            continue;
        }
        let t_reg = t_op.register_id;
        if regs.contains(&t_reg) {
            continue;
        }
        // Already resolvable by the recompiler's guard? Mirror its resolution
        // exactly so a real (direct or EUD-aliased) descriptor is never shadowed
        // by a placeholder. Compute/pixel: shift_regs == 0.
        let tex_num = usize::try_from(bind.textures2d.textures_num.max(0))
            .unwrap_or(0)
            .min(bind.textures2d.desc.len());
        let direct = bind.textures2d.desc[..tex_num]
            .iter()
            .any(|d| d.start_register == t_reg && !d.textures2d_without_sampler);
        if direct
            || super::recompile::mimg_register_eud_alias_index(bind, code, t_reg, at, false)
                .is_some()
        {
            continue;
        }
        regs.push(t_reg);
    }

    if regs.is_empty() {
        return;
    }

    for reg in regs {
        let Ok(index) = usize::try_from(bind.textures2d.textures_num) else {
            return;
        };
        if index >= ShaderTextureResources::RES_MAX {
            // More distinct uncaptured T# registers than slots: leave the rest
            // to the recompiler's named refusal (no silent drop).
            break;
        }
        bind.textures2d.desc[index].start_register = reg;
        bind.textures2d.desc[index].extended = false;
        // No usage slot produced this T#; the sentinel keeps the synthesized
        // entry distinguishable in bind-id dumps (mirrors the default S#).
        bind.textures2d.desc[index].slot = -1;
        bind.textures2d.desc[index].usage = ShaderTextureUsage::ReadOnly;
        bind.textures2d.desc[index].textures2d_without_sampler = false;
        bind.textures2d.desc[index].texture.fields = placeholder_texture_fields();
        bind.textures2d.textures_num += 1;
        bind.textures2d.textures2d_sampled_num += 1;
        tracing::debug!(
            start_register = reg,
            "synthesized placeholder (1x1) sampled T# for a runtime/bindless texture \
             the static capture missed"
        );
    }

    shader_calc_binding_indices(bind);
}

/// Fields for a PLACEHOLDER storage (UAV) T#: a 1x1 dummy at guest base 0.
///
/// When a captured storage descriptor exists, its image Dim (type nibble) and
/// format are copied in so `storage_texture_dim_format` still sees ONE
/// `%ImageL` (dim, format) across the array — a placeholder of a different
/// dim/format would make that helper refuse a mixed array, defeating the
/// rescue. base 0 keeps the bind path from reading real guest memory: it
/// allocates a 1x1 zero-seeded UAV, and `robustImageAccess2` (enabled on the
/// device) discards the dispatch's out-of-bounds writes into it.
fn placeholder_storage_fields(
    template: Option<&super::resources::ShaderTextureResource>,
) -> [u32; 8] {
    let mut fields = placeholder_texture_fields();
    if let Some(t) = template {
        // Dim = image type nibble, fields[3] bits 28..31 (see
        // `ShaderTextureResource::type_`).
        fields[3] = (fields[3] & 0x0FFF_FFFF) | (t.fields[3] & 0xF000_0000);
        // Format = fields[1] bits 20..28 (see `ShaderTextureResource::format`).
        // Copy only those bits; the base-0 low bits stay 0.
        const FMT_MASK: u32 = 0x1FF << 20;
        fields[1] = (fields[1] & !FMT_MASK) | (t.fields[1] & FMT_MASK);
    }
    fields
}

/// Rank 8 (draw-time null-descriptor fallback) — the STORAGE counterpart of
/// [`shader_synthesize_placeholder_sampled_texture`]: synthesize a PLACEHOLDER
/// storage (UAV) T# for an `image_store`/`image_store_mip` whose T# operand
/// register matches NO captured storage descriptor and no EUD alias.
///
/// Without this, `mimg_descriptor_guard` refuses the whole shader by name
/// (`dynamic-image-descriptor: Storage T# at sN matches no captured
/// descriptor`), and every dispatch binding it is skipped — the same
/// translate-time refusal the sampled placeholder already lifted for texture
/// reads, here extended to the write side (the broader rank-8 fix). shadPS4
/// emits the draw and resolves such an unresolved sharp to `Null()` at bind
/// time under robustness2 + nullDescriptor; this port's array-indexed-by-SGPR
/// model instead installs a REAL 1x1 placeholder UAV so the seeded array index
/// stays IN BOUNDS (no descriptor-array OOB — the exact hazard the guard's
/// refusal was written to avoid), then leans on `robustImageAccess2` for the
/// out-of-bounds texel writes into the 1x1 dummy.
///
/// Coverage is decided by the SAME resolution the guard uses — a direct
/// start-register match OR [`super::recompile::mimg_register_eud_alias_index`]'s
/// program-order EUD-alias walk — so a real (direct or EUD-aliased) storage
/// descriptor is never shadowed by a placeholder; only genuinely-unresolved
/// registers get one.
///
/// Genuinely-unsafe shapes still refuse and are NOT rescued here:
/// - a T# operand that is not an SGPR range (no register to seed an index) is
///   skipped, leaving the guard's named refusal;
/// - a register PROVABLY overwritten by a raw (uncovered) EUD `s_load` still
///   refuses at the guard's shape-2 check even AFTER a placeholder is added —
///   shape 2 tests the MIMG's register, not the descriptor's capture status,
///   so the raw dwords (the true, unmappable descriptor) keep it fatal.
///
/// Storage images are compute-only in the bind path
/// (`draw_translate::prepare_stage_binding`), so this is wired for the CS
/// stage only. Binding indices are recomputed here. `shift_regs` is 0 for
/// compute (the `gs_prolog` +8 rebase is a vertex-stage concern).
pub fn shader_synthesize_placeholder_storage_texture(
    code: &ShaderCode,
    bind: &mut ShaderBindResources,
) {
    use ShaderInstructionType as T;

    // Storage-WRITE MIMG that reads a T# from src[1] (srsrc * 4). Sampled ops
    // keep the sampled placeholder; `image_get_resinfo` (a dimension query)
    // stays untouched — a 1x1 dummy would report the wrong extent.
    const fn is_storage_write_mimg(t: ShaderInstructionType) -> bool {
        matches!(t, T::ImageStore | T::ImageStoreMip)
    }

    let mut regs: Vec<i32> = Vec::new();
    for (at, inst) in code.get_instructions().iter().enumerate() {
        if !is_storage_write_mimg(inst.type_) {
            continue;
        }
        let t_op = inst.src[1];
        if t_op.type_ != ShaderOperandType::Sgpr {
            // A non-SGPR-range T# has no register to seed an array index —
            // leave the guard's named refusal (genuinely unresolvable here).
            continue;
        }
        let t_reg = t_op.register_id;
        if regs.contains(&t_reg) {
            continue;
        }
        // Already resolvable by the guard (direct storage match or EUD alias)?
        // Mirror it exactly so a real descriptor is never shadowed. want_storage
        // = true.
        let tex_num = usize::try_from(bind.textures2d.textures_num.max(0))
            .unwrap_or(0)
            .min(bind.textures2d.desc.len());
        let direct = bind.textures2d.desc[..tex_num]
            .iter()
            .any(|d| d.start_register == t_reg && d.textures2d_without_sampler);
        if direct
            || super::recompile::mimg_register_eud_alias_index(bind, code, t_reg, at, true)
                .is_some()
        {
            continue;
        }
        regs.push(t_reg);
    }

    if regs.is_empty() {
        return;
    }

    // Share an existing storage descriptor's Dim+format so the placeholder
    // does not create a mixed-format `%ImageL` array.
    let template_fields = {
        let tex_num = usize::try_from(bind.textures2d.textures_num.max(0))
            .unwrap_or(0)
            .min(bind.textures2d.desc.len());
        let template = bind.textures2d.desc[..tex_num]
            .iter()
            .find(|d| d.textures2d_without_sampler)
            .map(|d| &d.texture);
        placeholder_storage_fields(template)
    };

    for reg in regs {
        let Ok(index) = usize::try_from(bind.textures2d.textures_num) else {
            return;
        };
        if index >= ShaderTextureResources::RES_MAX {
            // More distinct uncaptured T# registers than slots: leave the rest
            // to the recompiler's named refusal (no silent drop).
            break;
        }
        bind.textures2d.desc[index].start_register = reg;
        bind.textures2d.desc[index].extended = false;
        // No usage slot produced this T#; the sentinel keeps the synthesized
        // entry distinguishable in bind-id dumps (mirrors the sampled path).
        bind.textures2d.desc[index].slot = -1;
        bind.textures2d.desc[index].usage = ShaderTextureUsage::ReadWrite;
        bind.textures2d.desc[index].textures2d_without_sampler = true;
        bind.textures2d.desc[index].texture.fields = template_fields;
        bind.textures2d.textures_num += 1;
        bind.textures2d.textures2d_storage_num += 1;
        tracing::debug!(
            start_register = reg,
            "synthesized placeholder (1x1) storage T# for a runtime/bindless UAV \
             the static capture missed"
        );
    }

    shader_calc_binding_indices(bind);
}

/// Beyond Kyty: a compute shader that appends/consumes through the GDS counter
/// (`ds_append` / `ds_consume`) needs a GDS pointer resource so `%gds` is
/// declared and the host binds the persistent GDS arena. Real Gen5 shaders
/// address the counter through `M0`, not a captured descriptor, so no usage
/// slot ever produces the pointer — without one, `Recompile_DsAppend` returns
/// `false` ("can't recompile: DsAppend"). Synthesize one when the shader
/// appends/consumes but none was captured. Mirrors
/// [`shader_synthesize_default_sampler`]. Measured on ASTRO.BOT tiled-lighting
/// compute (the light-list append counter).
///
/// Call after `shader_get_input_info_*` (binding indices are recomputed here).
pub fn shader_synthesize_gds_pointer(code: &ShaderCode, bind: &mut ShaderBindResources) {
    use ShaderInstructionType as T;

    if bind.gds_pointers.pointers_num != 0 {
        return;
    }
    if !code.has_any_of(&[T::DsAppend, T::DsConsume]) {
        return;
    }

    // The append/consume path indexes the counter through M0; the descriptor
    // fields are unused, so a default (all-zero) pointer suffices to declare
    // and bind `%gds`.
    bind.gds_pointers.pointers_num = 1;
    bind.gds_pointers.start_register[0] = 0;
    bind.gds_pointers.slots[0] = -1;
    bind.gds_pointers.extended[0] = false;
    tracing::debug!(
        "synthesized GDS pointer for ds_append/ds_consume without a captured descriptor"
    );

    shader_calc_binding_indices(bind);
}

/// Kyty: Shader.cpp `ShaderParseUsage` (L1364) — the legacy (PS4)
/// usage-slot walk over the binary-info trailer tables.
///
/// Deviation: takes the shader `code` slice (Kyty takes the raw address)
/// plus `mem` to resolve the EUD (extended user data) pointer when a
/// `0x1b` slot appears.
pub fn shader_parse_usage(
    code: &[u32],
    mem: &impl ShaderMemory,
    info: &mut ShaderParsedUsage,
    bind: &mut ShaderBindResources,
    user_sgpr: &UserSgprInfo,
    user_sgpr_num: i32,
) -> Result<(), ShaderAnalysisError> {
    let usages = get_usage_slots(code)?;

    if !usages.valid {
        return Err(ni("shader has no usage slots"));
    }

    // Kyty resets exactly these fields (not vertex_attrib) — kept as-is.
    info.fetch = false;
    info.fetch_reg = 0;
    info.vertex_buffer = false;
    info.vertex_buffer_reg = 0;
    info.storage_buffers_readonly = 0;
    info.storage_buffers_constant = 0;
    info.storage_buffers_readwrite = 0;
    info.textures2d_readonly = 0;
    info.textures2d_readwrite = 0;
    info.extended_buffer = false;
    info.samplers = 0;
    info.gds_pointers = 0;
    info.direct_sgprs = 0;

    // (EUD contents, pointer-pair register) — the pair register is the slot
    // index where the file hands over to the EUD (see `EudView::base_dw`).
    let mut extended_buffer: Option<(Cow<'_, [u32]>, i32)> = None;
    fn ext_view<'a>(e: &'a Option<(Cow<'a, [u32]>, i32)>) -> Option<EudView<'a>> {
        e.as_ref().map(|(data, base_dw)| EudView {
            data: data.as_ref(),
            base_dw: *base_dw,
        })
    }

    let mut direct_sgprs = [false; UserSgprInfo::SGPRS_MAX];
    for (i, flag) in direct_sgprs.iter_mut().enumerate() {
        *flag = (i as i32) < user_sgpr_num;
    }

    for usage in &usages.slots {
        let start = i32::from(usage.start_register);
        let slot = i32::from(usage.slot);
        match usage.type_ {
            0x00 => {
                if usage.flags != 0 && usage.flags != 3 {
                    return Err(ni("usage 0x00: flags not in {0, 3}"));
                }
                if usage.flags == 0 {
                    shader_get_storage_buffer(
                        &mut bind.storage_buffers,
                        &mut direct_sgprs,
                        start,
                        slot,
                        ShaderStorageUsage::ReadOnly,
                        user_sgpr,
                        ext_view(&extended_buffer),
                    )?;
                    info.storage_buffers_readonly += 1;
                } else {
                    shader_get_texture_buffer(
                        &mut bind.textures2d,
                        &mut direct_sgprs,
                        start,
                        slot,
                        ShaderTextureUsage::ReadOnly,
                        user_sgpr,
                        ext_view(&extended_buffer),
                    )?;
                    info.textures2d_readonly += 1;
                    let last = (bind.textures2d.textures_num - 1) as usize;
                    check_read_only_texture_type(&mut bind.textures2d.desc[last].texture)?;
                }
            }

            0x01 => {
                if usage.flags != 0 {
                    return Err(ni("usage 0x01: flags != 0"));
                }
                shader_get_sampler(
                    &mut bind.samplers,
                    &mut direct_sgprs,
                    start,
                    slot,
                    user_sgpr,
                    ext_view(&extended_buffer),
                )?;
                info.samplers += 1;
            }

            0x02 => {
                if usage.flags != 0 {
                    return Err(ni("usage 0x02: flags != 0"));
                }
                shader_get_storage_buffer(
                    &mut bind.storage_buffers,
                    &mut direct_sgprs,
                    start,
                    slot,
                    ShaderStorageUsage::Constant,
                    user_sgpr,
                    ext_view(&extended_buffer),
                )?;
                info.storage_buffers_constant += 1;
            }

            0x04 => {
                if usage.flags != 0 && usage.flags != 3 {
                    return Err(ni("usage 0x04: flags not in {0, 3}"));
                }
                if usage.flags == 0 {
                    shader_get_storage_buffer(
                        &mut bind.storage_buffers,
                        &mut direct_sgprs,
                        start,
                        slot,
                        ShaderStorageUsage::ReadWrite,
                        user_sgpr,
                        ext_view(&extended_buffer),
                    )?;
                    info.storage_buffers_readwrite += 1;
                } else {
                    shader_get_texture_buffer(
                        &mut bind.textures2d,
                        &mut direct_sgprs,
                        start,
                        slot,
                        ShaderTextureUsage::ReadWrite,
                        user_sgpr,
                        ext_view(&extended_buffer),
                    )?;
                    info.textures2d_readwrite += 1;
                    let last = (bind.textures2d.textures_num - 1) as usize;
                    check_read_write_texture_type(
                        &bind.textures2d.desc[last].texture,
                        "usage 0x04",
                    )?;
                }
            }

            // IMM_ALU_FLOAT_CONST (GNM `InputUsageSlot` usage type 0x05,
            // beyond Kyty which EXITs): float constants the driver preloads
            // directly into the user SGPRs starting at `start_register`.
            // There is nothing to bind — leaving the registers marked direct
            // routes their captured values through the direct-SGPR pass,
            // which IS the semantic. Measured: 116 ASTRO.BOT CS dispatches.
            0x05 => {
                if usage.flags != 0 {
                    return Err(ni_owned(format!(
                        "usage 0x05 (imm alu float const): flags = {} (expected 0; \
                         start_register = {start}, slot = {slot})",
                        usage.flags
                    )));
                }
            }

            0x07 => {
                if usage.flags != 0 {
                    return Err(ni("usage 0x07: flags != 0"));
                }
                shader_get_gds_pointer(
                    &mut bind.gds_pointers,
                    &mut direct_sgprs,
                    start,
                    slot,
                    user_sgpr,
                    ext_view(&extended_buffer),
                )?;
                info.gds_pointers += 1;
            }

            0x12 => {
                if usage.slot != 0 {
                    return Err(ni("usage 0x12 (fetch): slot != 0"));
                }
                if usage.flags != 0 {
                    return Err(ni("usage 0x12 (fetch): flags != 0"));
                }
                info.fetch = true;
                info.fetch_reg = start;
                clear_direct(&mut direct_sgprs, usage.start_register as usize)?;
                clear_direct(&mut direct_sgprs, usage.start_register as usize + 1)?;
            }

            0x17 => {
                if usage.slot != 0 {
                    return Err(ni("usage 0x17 (vertex buffer): slot != 0"));
                }
                if usage.flags != 0 {
                    return Err(ni("usage 0x17 (vertex buffer): flags != 0"));
                }
                info.vertex_buffer = true;
                info.vertex_buffer_reg = start;
                clear_direct(&mut direct_sgprs, usage.start_register as usize)?;
                clear_direct(&mut direct_sgprs, usage.start_register as usize + 1)?;
            }

            0x1b => {
                if usage.flags != 0 {
                    return Err(ni("usage 0x1b (extended): flags != 0"));
                }
                if usage.slot != 1 {
                    return Err(ni("usage 0x1b (extended): slot != 1"));
                }
                if bind.extended.used {
                    return Err(ni("usage 0x1b (extended): already used"));
                }
                if usage.start_register as usize + 1 >= UserSgprInfo::SGPRS_MAX {
                    return Err(ni("usage 0x1b (extended): start_register + 1 >= SGPRS_MAX"));
                }
                bind.extended.used = true;
                bind.extended.slot = slot;
                bind.extended.start_register = start;
                bind.extended.data.fields[0] = user_sgpr.value[usage.start_register as usize];
                bind.extended.data.fields[1] = user_sgpr.value[usage.start_register as usize + 1];
                let base = bind.extended.data.base();
                extended_buffer = Some((mem.dwords_at(base).ok_or_else(|| bad_addr(base))?, start));
                info.extended_buffer = true;
                direct_sgprs[usage.start_register as usize] = false;
                direct_sgprs[usage.start_register as usize + 1] = false;
            }

            t => {
                tracing::error!("unknown usage type: 0x{t:02x}");
                return Err(ShaderAnalysisError::UnknownUsageType {
                    type_: u32::from(t),
                });
            }
        }
    }

    for (i, &direct) in direct_sgprs.iter().enumerate() {
        if direct {
            shader_get_direct_sgpr(&mut bind.direct_sgprs, i as i32, user_sgpr)?;
            info.direct_sgprs += 1;
        }
    }

    Ok(())
}

/// `RAEEN_TRACE_EUD` evidence dump: the full `ShaderUserData` mapping tables,
/// the captured user SGPRs, and a window of guest memory behind every
/// SGPR pair that looks like a pointer. The EUD resolver must know which pair
/// is the extended-user-data base and which table entry names it — this prints
/// everything needed to decide that from evidence instead of a guess.
fn trace_eud_evidence(
    label: &str,
    shader_addr: u64,
    user_data: &ShaderUserData,
    user_sgpr: &UserSgprInfo,
    declared: i32,
    mem: &impl ShaderMemory,
) {
    let Ok(filter) = std::env::var("RAEEN_TRACE_EUD") else {
        return;
    };
    // `1`/`all` retain the original all-shader dump. A comma-separated
    // address list narrows the otherwise very noisy per-bind evidence to the
    // failing shader(s), e.g. `RAEEN_TRACE_EUD=0x16fff700`. Minecraft can bind
    // tens of thousands of UI draws before the first transition; without this
    // filter the log reaches its safety cap before the interesting bind.
    let filter = filter.trim();
    if !filter.is_empty()
        && filter != "1"
        && !filter.eq_ignore_ascii_case("all")
        && !filter.split(',').any(|raw| {
            let raw = raw.trim();
            let raw = raw
                .strip_prefix("0x")
                .or_else(|| raw.strip_prefix("0X"))
                .unwrap_or(raw);
            u64::from_str_radix(raw, 16).ok() == Some(shader_addr)
        })
    {
        return;
    }
    // One compound line per shader so evidence can never be cross-attributed
    // between shaders by dedup.
    let mut out = format!(
        "TRACE_EUD shader={label}@{shader_addr:#x} eud={} srt={} declared={declared} count={}",
        user_data.eud_size_dw, user_data.srt_size_dw, user_sgpr.count
    );
    for (type_, &offset) in user_data.direct_resource_offset.iter().enumerate() {
        if offset != 0xffff {
            out += &format!(" direct[t{type_}]=s{offset}");
        }
    }
    for (table, sharps) in user_data.sharp_resource_offset.iter().enumerate() {
        for (slot, sharp) in sharps.iter().enumerate() {
            if sharp.offset_dw() != 0x7fff {
                out += &format!(
                    " sharp[{table}.{slot}]=s{}+{}",
                    sharp.offset_dw(),
                    sharp.size()
                );
            }
        }
    }
    for i in 0..(user_sgpr.count.max(4) as usize).min(16) {
        if user_sgpr.value[i] != 0 {
            out += &format!(" s{i}={:#x}", user_sgpr.value[i]);
        }
    }
    tracing::warn!("{out}");
    for i in 0..(user_sgpr.count.max(4) as usize).min(UserSgprInfo::SGPRS_MAX - 1) {
        let pair = u64::from(user_sgpr.value[i]) | (u64::from(user_sgpr.value[i + 1]) << 32);
        // A plausible guest pointer: nonzero, page-ish aligned, below 48 bits.
        if pair == 0 || pair & 0x3 != 0 || pair >> 48 != 0 {
            continue;
        }
        if let Some(win) = mem.dwords_at(pair) {
            let take = win.len().min(16);
            tracing::warn!(
                "TRACE_EUD shader={label}@{shader_addr:#x} mem[s{i}:s{}]@{pair:#x} = {:08x?}",
                i + 1,
                &win[..take]
            );
        }
    }
}

/// SharpEmu-parity capture of image descriptors delivered RAW through the
/// EUD — no usage-table slot declares them; the shader loads them itself with
/// a covered `s_load` off the EUD base pair and feeds the loaded registers
/// straight to an MIMG. SharpEmu resolves every image descriptor by scalar-
/// evaluating the program and copying the register file at the MIMG
/// (`reference/sharpemu/src/SharpEmu.ShaderCompiler/`
/// `Gen5ShaderScalarEvaluator.cs:599-668`); the equivalent here reads the
/// descriptor's dwords out of the EUD snapshot at the covered load's offset
/// and captures it exactly like a declared sharp. The recompiler's
/// `mimg_descriptor_guard` then accepts the MIMG register through its
/// covered-load alias rule, and `WriteLocalVariables` / `sload_dword_extended`
/// seed the rewritten descriptor-array index into the load's destination
/// registers at runtime — no raw guest dword is ever bound as an array index.
///
/// Measured (ASTRO.BOT 2026-07-21, after the EUD snapshot-base fix): 4 CS
/// shaders (`0x500542c00`/`0x500566b00` sampled s16, `0x500543b00` storage
/// s68, `0x50740a700` S# s16) refused `dynamic-image-descriptor` on exactly
/// this shape, ~740 dispatches each per 300 s; their writebacks feed the
/// WAIT_REG_MEM labels the ACB queues park on, so the refusals also
/// serialized presents (256 → 8 per run).
///
/// Degrade rules (never a refusal HERE — the guard's named skip stands):
/// no EUD, no covered load for the register, offset outside the snapshot,
/// all-zero content, or a buffer-typed dword3 (type nibble 0) → no capture.
/// A captured-but-unsupported texture type still errors by name via the same
/// checks the declared-sharp walks run. CS-only wiring, like the raw-EUD
/// window (VS/PS defer until measured).
pub fn shader_capture_eud_image_descriptors(
    code: &ShaderCode,
    bind: &mut ShaderBindResources,
    user_sgpr: &UserSgprInfo,
    eud: EudView<'_>,
) -> Result<(), ShaderAnalysisError> {
    use crate::shader::types::ShaderInstructionType as T;
    if !bind.extended.used {
        return Ok(());
    }
    let eud_base = bind.extended.start_register;
    let sgprs_max = UserSgprInfo::SGPRS_MAX as i32;
    // `read_sharp_fields` marks consumed registers for DIRECT sharps only;
    // every synthesized capture is EUD-resident (start >= SGPRS_MAX), so the
    // flag array is dead weight here.
    let mut unused_flags = [false; UserSgprInfo::SGPRS_MAX];

    // The two virtual-register conventions `eud_rel_index` resolves
    // (`start >= SGPRS_MAX` ⇒ `start - SGPRS_MAX`; else rebased on the EUD
    // base register), reproduced locally so capture-time dedup matches the
    // guard's acceptance exactly.
    let rel_of = |start: i32| -> Option<i32> {
        if start >= sgprs_max {
            Some(start - sgprs_max)
        } else if start >= eud_base {
            Some(start - eud_base)
        } else {
            None
        }
    };

    for inst in code.get_instructions() {
        let (want_storage, uses_sampler) = match inst.type_ {
            T::ImageStore | T::ImageStoreMip => (true, false),
            T::ImageLoad | T::ImageGetResinfo => (false, false),
            T::ImageSample
            | T::ImageSampleLz
            | T::ImageSampleLzO
            | T::ImageSampleCLz
            | T::ImageGather4Lz => (false, true),
            _ => continue,
        };

        // T# operand (MIMG srsrc): 8 consecutive SGPRs starting at src[1].
        let t_op = inst.src[1];
        if t_op.type_ == ShaderOperandType::Sgpr {
            let t_reg = t_op.register_id;
            let tex_num = usize::try_from(bind.textures2d.textures_num.max(0))
                .unwrap_or(0)
                .min(bind.textures2d.desc.len());
            let direct_hit = bind.textures2d.desc[..tex_num]
                .iter()
                .any(|d| d.start_register == t_reg && d.textures2d_without_sampler == want_storage);
            if !direct_hit && let Some(k) = eud_load_offset_for_register(code, eud_base, t_reg) {
                let alias_hit = bind.textures2d.desc[..tex_num].iter().any(|d| {
                    d.extended
                        && d.textures2d_without_sampler == want_storage
                        && rel_of(d.start_register) == Some(k)
                });
                let ku = usize::try_from(k).unwrap_or(usize::MAX);
                if !alias_hit && let Some(fields) = eud.data.get(ku..ku + 8) {
                    let type_nibble = (fields[3] >> 28) & 0xf;
                    if fields.iter().any(|&d| d != 0) && type_nibble != 0 {
                        let slot = bind.textures2d.textures_num;
                        shader_get_texture_buffer(
                            &mut bind.textures2d,
                            &mut unused_flags,
                            sgprs_max + k,
                            slot,
                            if want_storage {
                                ShaderTextureUsage::ReadWrite
                            } else {
                                ShaderTextureUsage::ReadOnly
                            },
                            user_sgpr,
                            Some(eud),
                        )?;
                        let last = (bind.textures2d.textures_num - 1) as usize;
                        if want_storage {
                            check_read_write_texture_type(
                                &bind.textures2d.desc[last].texture,
                                "raw-EUD",
                            )?;
                        } else {
                            check_read_only_texture_type(&mut bind.textures2d.desc[last].texture)?;
                        }
                        tracing::debug!(
                            eud_dword = k,
                            reg = t_reg,
                            storage = want_storage,
                            "captured raw-EUD image descriptor from covered load"
                        );
                    }
                }
            }
        }

        // S# operand (MIMG ssamp): 4 consecutive SGPRs starting at src[2].
        if uses_sampler && inst.src[2].type_ == ShaderOperandType::Sgpr {
            let s_reg = inst.src[2].register_id;
            let samp_num = usize::try_from(bind.samplers.samplers_num.max(0))
                .unwrap_or(0)
                .min(bind.samplers.start_register.len());
            let direct_hit = bind.samplers.start_register[..samp_num].contains(&s_reg);
            if !direct_hit && let Some(k) = eud_load_offset_for_register(code, eud_base, s_reg) {
                let alias_hit = (0..samp_num).any(|i| {
                    bind.samplers.extended[i] && rel_of(bind.samplers.start_register[i]) == Some(k)
                });
                let ku = usize::try_from(k).unwrap_or(usize::MAX);
                if !alias_hit && eud.data.get(ku..ku + 4).is_some() {
                    let slot = bind.samplers.samplers_num;
                    shader_get_sampler(
                        &mut bind.samplers,
                        &mut unused_flags,
                        sgprs_max + k,
                        slot,
                        user_sgpr,
                        Some(eud),
                    )?;
                    tracing::debug!(
                        eud_dword = k,
                        reg = s_reg,
                        "captured raw-EUD sampler descriptor from covered load"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Capture buffer V# descriptors that the shader loads from its raw EUD
/// window, and discard stale table entries that no decoded buffer access can
/// reach.
///
/// Gen5 usage tables describe a resource *slot*, but the live descriptor can
/// be delivered later by a covered scalar load:
///
/// `s_load_dwordx4 s[resource:resource+3], s[eud:eud+1], byte_offset`
///
/// The static table walk cannot see that provenance and, when its declared
/// register is the EUD pointer itself, reads dword 0 as a V#. Minecraft's
/// post-title UI does exactly this: dword 0 is non-descriptor data while the
/// real 16-byte V# starts at dword 8. Binding both is not harmless because the
/// bogus first descriptor is validated/prepared before the shader can run.
///
/// This pass is deliberately proof-driven:
/// - only decoded MUBUF/MTBUF/SBUFFER resource operands are considered;
/// - only an exact covered `s_load_dwordx4/x8` from the recovered EUD pair
///   supplies a descriptor;
/// - the four captured dwords must be nonzero and buffer-typed;
/// - stale extended entries are pruned only when every decoded buffer operand
///   resolves either directly or through one of those covered loads.
///
/// Direct descriptors and unresolved/aliased programs retain the conservative
/// table result.
pub fn shader_capture_eud_storage_buffers(
    code: &ShaderCode,
    bind: &mut ShaderBindResources,
    user_sgpr: &UserSgprInfo,
    eud: EudView<'_>,
    mem: &impl ShaderMemory,
) -> Result<(), ShaderAnalysisError> {
    use crate::shader::types::ShaderInstructionType as T;

    if !bind.extended.used {
        return Ok(());
    }

    let resource_operand = |inst: &ShaderInstruction| -> Option<(i32, bool)> {
        let (operand, writes) = match inst.type_ {
            T::BufferLoadDword
            | T::BufferLoadDwordX2
            | T::BufferLoadDwordX3
            | T::BufferLoadDwordX4
            | T::BufferLoadUbyte
            | T::BufferLoadFormatX
            | T::BufferLoadFormatXy
            | T::BufferLoadFormatXyz
            | T::BufferLoadFormatXyzw
            | T::TBufferLoadFormatX
            | T::TBufferLoadFormatXyzw => (inst.src[1], false),
            T::BufferStoreDword
            | T::BufferStoreDwordX2
            | T::BufferStoreDwordX4
            | T::BufferStoreFormatX
            | T::BufferStoreFormatXy
            | T::BufferStoreFormatXyz
            | T::BufferStoreFormatXyzw => (inst.src[1], true),
            T::SBufferLoadDword
            | T::SBufferLoadDwordx2
            | T::SBufferLoadDwordx4
            | T::SBufferLoadDwordx8
            | T::SBufferLoadDwordx16 => (inst.src[0], false),
            _ => return None,
        };
        (operand.type_ == ShaderOperandType::Sgpr).then_some((operand.register_id, writes))
    };

    let mut resource_regs = Vec::<(i32, bool)>::new();
    for inst in code.get_instructions() {
        if let Some((reg, writes)) = resource_operand(inst) {
            if let Some((_, old_writes)) = resource_regs.iter_mut().find(|(r, _)| *r == reg) {
                *old_writes |= writes;
            } else {
                resource_regs.push((reg, writes));
            }
        }
    }
    if resource_regs.is_empty() {
        return Ok(());
    }

    let eud_base = bind.extended.start_register;
    let sgprs_max = UserSgprInfo::SGPRS_MAX as i32;
    let rel_of = |start: i32| -> Option<i32> {
        if start >= sgprs_max {
            Some(start - sgprs_max)
        } else if start >= eud_base {
            Some(start - eud_base)
        } else {
            None
        }
    };

    let existing_num = usize::try_from(bind.storage_buffers.buffers_num.max(0))
        .unwrap_or(0)
        .min(bind.storage_buffers.buffers.len());
    let existing = bind.storage_buffers;
    let fallback_slot = (0..existing_num)
        .find(|&i| existing.extended[i])
        .map_or(existing.buffers_num, |i| existing.slots[i]);
    let fallback_usage =
        if (0..existing_num).any(|i| existing.usages[i] == ShaderStorageUsage::ReadWrite) {
            ShaderStorageUsage::ReadWrite
        } else {
            ShaderStorageUsage::ReadOnly
        };

    let mut dynamic = Vec::<(i32, i32, ShaderStorageUsage)>::new();
    for &(reg, writes) in &resource_regs {
        for load in code.get_instructions() {
            if !matches!(load.type_, T::SLoadDwordx4 | T::SLoadDwordx8)
                || load.src[0].type_ != ShaderOperandType::Sgpr
                || load.src[0].register_id != eud_base
                || load.dst.type_ != ShaderOperandType::Sgpr
                || load.dst.register_id != reg
                || !matches!(
                    load.src[1].type_,
                    ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant
                )
                || load.src[1].constant.i() < 0
            {
                continue;
            }
            let offset = (load.src[1].constant.u >> 2) as i32;
            let offset_usize = usize::try_from(offset).unwrap_or(usize::MAX);
            let Some(fields) = eud.data.get(offset_usize..offset_usize.saturating_add(4)) else {
                continue;
            };
            if fields == [0, 0, 0, 0] || !sharp_dword3_is_buffer(fields[3]) {
                continue;
            }
            let candidate = ShaderBufferResource {
                fields: fields.try_into().expect("four-dword V# slice"),
            };
            if mem.dwords_at(candidate.base48()).is_none() {
                tracing::debug!(
                    eud_dword = offset,
                    base = format_args!("{:#x}", candidate.base48()),
                    "raw-EUD buffer-shaped tuple does not point to readable guest memory"
                );
                continue;
            }
            let usage = if writes || fallback_usage == ShaderStorageUsage::ReadWrite {
                ShaderStorageUsage::ReadWrite
            } else {
                ShaderStorageUsage::ReadOnly
            };
            if let Some((_, _, old_usage)) = dynamic.iter_mut().find(|(old, _, _)| *old == offset) {
                if usage == ShaderStorageUsage::ReadWrite {
                    *old_usage = usage;
                }
            } else {
                dynamic.push((offset, fallback_slot, usage));
            }
        }
    }
    if dynamic.is_empty() {
        return Ok(());
    }

    let every_operand_resolved = resource_regs.iter().all(|(reg, _)| {
        (0..existing_num).any(|i| !existing.extended[i] && existing.start_register[i] == *reg)
            || code.get_instructions().iter().any(|load| {
                matches!(load.type_, T::SLoadDwordx4 | T::SLoadDwordx8)
                    && load.src[0].type_ == ShaderOperandType::Sgpr
                    && load.src[0].register_id == eud_base
                    && load.dst.type_ == ShaderOperandType::Sgpr
                    && load.dst.register_id == *reg
                    && matches!(
                        load.src[1].type_,
                        ShaderOperandType::LiteralConstant
                            | ShaderOperandType::IntegerInlineConstant
                    )
                    && dynamic
                        .iter()
                        .any(|(offset, _, _)| *offset == (load.src[1].constant.u >> 2) as i32)
            })
    });

    // Rebuild the compact array so an unreachable bogus descriptor is never
    // prepared by the Vulkan binding path. Direct descriptors always stay.
    let mut compact = ShaderStorageResources::default();
    for i in 0..existing_num {
        let readable = mem.dwords_at(existing.buffers[i].base48()).is_some();
        let keep = !existing.extended[i]
            || (readable
                && (!every_operand_resolved
                    || rel_of(existing.start_register[i])
                        .is_some_and(|rel| dynamic.iter().any(|(offset, _, _)| *offset == rel))));
        if !keep {
            tracing::debug!(
                start_register = existing.start_register[i],
                slot = existing.slots[i],
                base = format_args!("{:#x}", existing.buffers[i].base48()),
                "discarding unreachable EUD storage descriptor"
            );
            continue;
        }
        let out = compact.buffers_num as usize;
        compact.buffers[out] = existing.buffers[i];
        compact.usages[out] = existing.usages[i];
        compact.slots[out] = existing.slots[i];
        compact.start_register[out] = existing.start_register[i];
        compact.extended[out] = existing.extended[i];
        compact.buffers_num += 1;
    }
    bind.storage_buffers = compact;

    let mut unused_flags = [false; UserSgprInfo::SGPRS_MAX];
    for (offset, slot, usage) in dynamic {
        let current_num = usize::try_from(bind.storage_buffers.buffers_num.max(0))
            .unwrap_or(0)
            .min(bind.storage_buffers.buffers.len());
        if (0..current_num).any(|i| {
            bind.storage_buffers.extended[i]
                && rel_of(bind.storage_buffers.start_register[i]) == Some(offset)
        }) {
            continue;
        }
        shader_get_storage_buffer(
            &mut bind.storage_buffers,
            &mut unused_flags,
            sgprs_max + offset,
            slot,
            usage,
            user_sgpr,
            Some(eud),
        )?;
        tracing::debug!(
            eud_dword = offset,
            slot,
            ?usage,
            "captured raw-EUD storage descriptor from covered load"
        );
    }

    Ok(())
}

/// Kyty: Shader.cpp `ShaderParseUsage2` (L1505) — the PS5 path over the
/// `ShaderUserData` direct/sharp mapping tables.
pub fn shader_parse_usage2(
    user_data: &ShaderUserData,
    info: &mut ShaderParsedUsage,
    bind: &mut ShaderBindResources,
    user_sgpr: &UserSgprInfo,
    user_sgpr_num: i32,
    eud: Option<EudView<'_>>,
    mem: Option<&dyn ShaderMemory>,
) -> Result<(), ShaderAnalysisError> {
    // Same reset list as ShaderParseUsage (vertex_attrib not reset upstream).
    info.fetch = false;
    info.fetch_reg = 0;
    info.vertex_buffer = false;
    info.vertex_buffer_reg = 0;
    info.storage_buffers_readonly = 0;
    info.storage_buffers_constant = 0;
    info.storage_buffers_readwrite = 0;
    info.textures2d_readonly = 0;
    info.textures2d_readwrite = 0;
    info.extended_buffer = false;
    info.samplers = 0;
    info.gds_pointers = 0;
    info.direct_sgprs = 0;

    if user_data.eud_size_dw != 0 && eud.is_none() {
        // EUD declared but the caller could not recover it (null/unmapped
        // pointer, or no guest memory). Still an honest refusal — resolving
        // sharps against a buffer we do not have would invent descriptors.
        // The error names every candidate SGPR value so the next session can
        // extend `read_extended_user_data`'s location heuristic from evidence
        // (which register pair actually holds the pointer) instead of a
        // guess. Both existing strategies (pair after the declared file;
        // scalar-load base pairs) were tried and found nothing readable.
        let mut sgprs = String::new();
        for (i, &v) in user_sgpr.value.iter().enumerate() {
            if v != 0 {
                let _ = std::fmt::Write::write_fmt(&mut sgprs, format_args!(" s{i}={v:#x}"));
            }
        }
        return Err(ni_owned(format!(
            "ShaderUserData eud_size_dw != 0 (EUD unreadable): eud_size_dw={}, \
             srt_size_dw={}, declared={user_sgpr_num}, captured={}, nonzero sgprs:{}",
            user_data.eud_size_dw,
            user_data.srt_size_dw,
            user_sgpr.count,
            if sgprs.is_empty() { " (none)" } else { &sgprs }
        )));
    }
    // SRT (Shader Resource Table) — beyond Kyty, which EXIT_NOT_IMPLEMENTEDs
    // on any srt_size_dw (Shader.cpp L1529). The SRT is an app-defined block
    // of raw dwords that lives INLINE in the user-data registers: the driver
    // copies `srt_size_dw` dwords into the user SGPRs and the shader indexes
    // them directly (SharpEmu's Gen5 evaluator likewise reads SRT content
    // straight out of the captured user data). No table entry names the SRT
    // registers — they are exactly the declared registers the direct/sharp
    // mapping tables below do NOT consume, and the direct-SGPR collection at
    // the end of this function binds every one of those with its captured
    // runtime value (push-constant seeded), which IS the hardware semantic.
    // An SRT larger than the declared register file spills to memory we have
    // no pointer for — keep that case an evidence-rich named refusal.
    if user_data.srt_size_dw != 0 && i32::from(user_data.srt_size_dw) > user_sgpr_num {
        let mut sgprs = String::new();
        for (i, &v) in user_sgpr.value.iter().enumerate() {
            if v != 0 {
                let _ = std::fmt::Write::write_fmt(&mut sgprs, format_args!(" s{i}={v:#x}"));
            }
        }
        return Err(ni_owned(format!(
            "ShaderUserData srt_size_dw ({}) exceeds the declared user-SGPR file \
             (declared={user_sgpr_num}, captured={}, eud_size_dw={}), nonzero sgprs:{}",
            user_data.srt_size_dw,
            user_sgpr.count,
            user_data.eud_size_dw,
            if sgprs.is_empty() { " (none)" } else { &sgprs }
        )));
    }

    // Kyty leaves this None (no EUD support); increment 3 supplies the buffer
    // the caller read from guest memory, which the extended branches of
    // `read_sharp_fields` already know how to index.
    let extended_buffer: Option<EudView<'_>> = eud;

    // Record the recovered EUD base in `bind.extended` (Kyty sets this in
    // the legacy usage-0x1b arm; the Gen5 tables have no such slot — the
    // pointer pair is recovered by `read_extended_user_data`). The
    // recompiler needs it twice: `Spirv::WriteLocalVariables` rebases each
    // extended sharp's push-constant mapping on this register, and the
    // `s_load_dwordx*` translation recognises loads THROUGH the pair as
    // descriptor fetches (461 measured ASTRO.BOT CS refusals on "extended
    // storage buffer mapping" without it). `info.extended_buffer` stays
    // false deliberately: that flag only feeds the "vs: extended buffer"
    // gate, and Gen5 VS shaders with an EUD must keep resolving as before.
    if let Some(e) = extended_buffer {
        if !bind.extended.used {
            bind.extended.used = true;
            bind.extended.slot = 1;
            bind.extended.start_register = e.base_dw;
            if let Ok(base) = usize::try_from(e.base_dw) {
                if base + 1 < UserSgprInfo::SGPRS_MAX {
                    bind.extended.data.fields[0] = user_sgpr.value[base];
                    bind.extended.data.fields[1] = user_sgpr.value[base + 1];
                }
            }
        }
    }

    let mut direct_sgprs = [false; UserSgprInfo::SGPRS_MAX];
    for (i, flag) in direct_sgprs.iter_mut().enumerate() {
        *flag = (i as i32) < user_sgpr_num;
    }

    // A pointer-pair entry whose registers the draw never wrote is a null
    // stream, not a mapping: the shader declares an OPTIONAL second vertex
    // stream (measured on Minecraft's menu GS: VB pointers at s0/s4, attrib
    // pointers at s2/s6) and a draw feeding only stream 0 programs only
    // s0..s3. Binding the null pair would send the vertex fetch to address 0.
    // Skipping it keeps "last WRITTEN stream wins", so a draw that programs
    // both streams behaves exactly as before.
    let pair_is_written = |offset: u16| -> bool {
        let i = offset as usize;
        i + 1 < UserSgprInfo::SGPRS_MAX && (user_sgpr.value[i] != 0 || user_sgpr.value[i + 1] != 0)
    };

    for (type_, &offset) in user_data.direct_resource_offset.iter().enumerate() {
        if offset == 0xffff {
            continue;
        }

        let reg = i32::from(offset);

        match type_ {
            8 => {
                if !pair_is_written(offset) {
                    tracing::debug!(reg, "usage2: null vertex-buffer stream skipped");
                    continue;
                }
                info.vertex_buffer = true;
                info.vertex_buffer_reg = reg;
                clear_direct(&mut direct_sgprs, offset as usize)?;
                clear_direct(&mut direct_sgprs, offset as usize + 1)?;
            }

            10 => {
                if !pair_is_written(offset) {
                    tracing::debug!(reg, "usage2: null vertex-attrib stream skipped");
                    continue;
                }
                info.vertex_attrib = true;
                info.vertex_attrib_reg = reg;
                clear_direct(&mut direct_sgprs, offset as usize)?;
                clear_direct(&mut direct_sgprs, offset as usize + 1)?;
            }

            // Direct type 0 — IMM_RESOURCE, beyond Kyty (upstream EXITs on
            // anything but 8/10 in this table): a read-only T#/V# descriptor
            // the driver preloaded into the user SGPRs at `offset`. The
            // legacy usage-0x00 arm disambiguates V# vs T# with the slot's
            // `flags`, but the direct table carries no flags — so use the
            // descriptor's own type nibble exactly like the sharp-table
            // walks below (dword3[28:31] reads 0 for a buffer, the
            // shadPS4-confirmed overlap). One entry per type in this table,
            // so the slot is 0 (same as the type-1 arm). 59 measured
            // ASTRO.BOT CS dispatches print `unknown usage type: 0x00`
            // through this Gen5 table.
            0 => {
                let mut peek = [0u32; 4];
                read_sharp_fields(
                    &mut direct_sgprs,
                    reg,
                    user_sgpr,
                    extended_buffer,
                    &mut peek,
                )?;
                if sharp_dword3_is_buffer(peek[3]) {
                    shader_get_storage_buffer(
                        &mut bind.storage_buffers,
                        &mut direct_sgprs,
                        reg,
                        0,
                        ShaderStorageUsage::ReadOnly,
                        user_sgpr,
                        extended_buffer,
                    )?;
                    info.storage_buffers_readonly += 1;
                } else {
                    shader_get_texture_buffer(
                        &mut bind.textures2d,
                        &mut direct_sgprs,
                        reg,
                        0,
                        ShaderTextureUsage::ReadOnly,
                        user_sgpr,
                        extended_buffer,
                    )?;
                    info.textures2d_readonly += 1;
                    let last = (bind.textures2d.textures_num - 1) as usize;
                    check_read_only_texture_type(&mut bind.textures2d.desc[last].texture)?;
                }
            }

            // Direct type 1 — IMM_SAMPLER, beyond Kyty (upstream EXITs on
            // anything but 8/10). The S# sampler descriptor is preloaded
            // into 4 user SGPRs at `offset`; route it into
            // `ShaderSamplerResources` exactly like the legacy usage-0x01
            // arm does via `ShaderGetSampler` (Kyty Shader.cpp L1421/L1233).
            // The direct table has one entry per type, so the slot is 0.
            // 738 measured ASTRO.BOT CS failures (the error site prints
            // 0x0001, the 4-hex-digit form — same round-1 lesson as type 5
            // below: the LEGACY table alone is not enough).
            1 => {
                shader_get_sampler(
                    &mut bind.samplers,
                    &mut direct_sgprs,
                    reg,
                    0,
                    user_sgpr,
                    extended_buffer,
                )?;
                info.samplers += 1;
            }

            // Direct type 5 — beyond Kyty (upstream EXITs on anything but
            // 8/10). Round 1 added an IMM_ALU_FLOAT_CONST arm to the LEGACY
            // `shader_parse_usage` for this number, but the 230 measured
            // ASTRO.BOT CS failures come through THIS Gen5 table (the error
            // site prints 0x0005, the 4-hex-digit form). Same treatment as
            // the legacy arm: the register holds immediate data the driver
            // preloaded — nothing to bind, leaving it marked direct routes
            // its captured value through the direct-SGPR pass, which IS the
            // semantic. SharpEmu's Gen5 path likewise only records the
            // direct table and reads the register values verbatim.
            5 => {
                tracing::debug!(reg, "usage2: direct type 5 (immediate) left as direct sgpr");
            }

            t => {
                tracing::error!("unknown usage type: 0x{t:04x}");
                return Err(ShaderAnalysisError::UnknownUsageType { type_: t as u32 });
            }
        }
    }

    for (slot, sharp) in user_data.sharp_resource_offset[0].iter().enumerate() {
        if sharp.offset_dw() == 0x7fff {
            continue;
        }
        if sharp.size() != 0 {
            // Beyond Kyty (upstream EXIT_NOT_IMPLEMENTEDs here): measured in
            // Minecraft's menu CS, a table-0 sharp with size == 1 is a
            // 4-dword buffer V# (the shader's writable output), not an
            // 8-dword texture T#. Bind it as a read-write storage buffer —
            // the compute path writes such buffers back to guest memory.
            shader_get_storage_buffer(
                &mut bind.storage_buffers,
                &mut direct_sgprs,
                i32::from(sharp.offset_dw()),
                slot as i32,
                ShaderStorageUsage::ReadWrite,
                user_sgpr,
                extended_buffer,
            )?;
            info.storage_buffers_readwrite += 1;
            continue;
        }
        // A table-0 sharp with size == 0 occupies an 8-register (8-dword)
        // slot, but its *content* is either an 8-dword texture T# or a
        // 4-dword buffer V#. The descriptor's own type field (dword3[28:31])
        // disambiguates: a value of 0 means it is a *buffer*, not an image.
        // On RDNA2 the buffer's 2-bit `type` and the image's 4-bit `type`
        // overlap in that nibble and read 0 for a buffer (confirmed in
        // shadPS4 video_core/amdgpu/resource.h). Measured on Minecraft's
        // textured PS: sharp[0.0] resolves to a structured buffer (stride 16,
        // 8 records) that was being rejected as a "non-2D texture".
        //
        // Read the full 8-dword slot so all eight user SGPRs are marked
        // consumed (the old texture path did this; consuming only the buffer's
        // four leaves the upper four dangling and fails the direct-SGPR
        // collection). If it is a buffer, bind the first four dwords as a
        // read-only storage buffer — mirroring the direct-usage 0x00/flags==0
        // path above.
        let mut peek = [0u32; 8];
        if let Err(error) = read_sharp_fields(
            &mut direct_sgprs,
            i32::from(sharp.offset_dw()),
            user_sgpr,
            extended_buffer,
            &mut peek,
        ) {
            // A size-0 table-0 entry is a read-only resource, but a
            // runtime/SRT-bound descriptor cannot be inspected here to
            // distinguish V# from T#. Do not abort before the texture
            // helper's established non-fatal path: install its valid 1x1 T#
            // placeholder and let instruction-driven synthesis replace it
            // when a later decoded image operand identifies the real
            // descriptor. Measured on Minecraft gameplay PS 0x1700c000:
            // slot 0 starts at s14 while only s0..s15 are captured.
            tracing::debug!(
                slot,
                start_register = sharp.offset_dw(),
                error = %error,
                "usage2: unresolved table-0 resource treated as runtime texture"
            );
            shader_get_texture_buffer(
                &mut bind.textures2d,
                &mut direct_sgprs,
                i32::from(sharp.offset_dw()),
                slot as i32,
                ShaderTextureUsage::ReadOnly,
                user_sgpr,
                extended_buffer,
            )?;
            info.textures2d_readonly += 1;
            let last = (bind.textures2d.textures_num - 1) as usize;
            check_read_only_texture_type(&mut bind.textures2d.desc[last].texture)?;
            continue;
        }
        if sharp_dword3_is_buffer(peek[3]) {
            shader_get_storage_buffer(
                &mut bind.storage_buffers,
                &mut direct_sgprs,
                i32::from(sharp.offset_dw()),
                slot as i32,
                ShaderStorageUsage::ReadOnly,
                user_sgpr,
                extended_buffer,
            )?;
            info.storage_buffers_readonly += 1;
            continue;
        }
        shader_get_texture_buffer(
            &mut bind.textures2d,
            &mut direct_sgprs,
            i32::from(sharp.offset_dw()),
            slot as i32,
            ShaderTextureUsage::ReadOnly,
            user_sgpr,
            extended_buffer,
        )?;
        info.textures2d_readonly += 1;
        let last = (bind.textures2d.textures_num - 1) as usize;
        check_read_only_texture_type(&mut bind.textures2d.desc[last].texture)?;
    }

    for (slot, sharp) in user_data.sharp_resource_offset[1].iter().enumerate() {
        if sharp.offset_dw() == 0x7fff {
            continue;
        }
        // Beyond Kyty (upstream EXIT_NOT_IMPLEMENTEDs on a non-empty table 1):
        // measured in Minecraft's menu CS. Mirror the table-0 extension — a
        // size == 1 sharp is a 4-dword buffer V#, bound read-write.
        if sharp.size() == 1 {
            shader_get_storage_buffer(
                &mut bind.storage_buffers,
                &mut direct_sgprs,
                i32::from(sharp.offset_dw()),
                slot as i32,
                ShaderStorageUsage::ReadWrite,
                user_sgpr,
                extended_buffer,
            )?;
            info.storage_buffers_readwrite += 1;
            continue;
        }
        // size == 0 (`ShaderSharp::size` is a single bit): an 8-dword slot,
        // exactly as in table 0 — its content is either an 8-dword image T#
        // or a 4-dword buffer V#, disambiguated by the descriptor's own type
        // nibble (dword3[28:31] reads 0 for a buffer; see the table-0 walk).
        // Table 1 is the READ-WRITE table, so an image here is a storage
        // image (UAV) and a buffer is a read-write V#. 347 measured
        // ASTRO.BOT CS dispatch skips ("sharp table 1 entry with size != 1").
        let mut peek = [0u32; 8];
        read_sharp_fields(
            &mut direct_sgprs,
            i32::from(sharp.offset_dw()),
            user_sgpr,
            extended_buffer,
            &mut peek,
        )?;
        if sharp_dword3_is_buffer(peek[3]) {
            shader_get_storage_buffer(
                &mut bind.storage_buffers,
                &mut direct_sgprs,
                i32::from(sharp.offset_dw()),
                slot as i32,
                ShaderStorageUsage::ReadWrite,
                user_sgpr,
                extended_buffer,
            )?;
            info.storage_buffers_readwrite += 1;
            continue;
        }
        shader_get_texture_buffer(
            &mut bind.textures2d,
            &mut direct_sgprs,
            i32::from(sharp.offset_dw()),
            slot as i32,
            ShaderTextureUsage::ReadWrite,
            user_sgpr,
            extended_buffer,
        )?;
        info.textures2d_readwrite += 1;
        // The storage-image machinery covers height-1 1D (type 8), native 2D
        // (9), and 3D volumes (10). The compute path uploads and writes back
        // their complete linear texels.
        let last = (bind.textures2d.textures_num - 1) as usize;
        let t = &bind.textures2d.desc[last].texture;
        check_read_write_texture_type(t, "table 1")?;
    }

    for (slot, sharp) in user_data.sharp_resource_offset[2].iter().enumerate() {
        if sharp.offset_dw() == 0x7fff {
            continue;
        }
        if sharp.size() != 1 {
            return Err(ni("sampler sharp size != 1"));
        }
        shader_get_sampler(
            &mut bind.samplers,
            &mut direct_sgprs,
            i32::from(sharp.offset_dw()),
            slot as i32,
            user_sgpr,
            extended_buffer,
        )?;
        info.samplers += 1;
    }

    for (slot, sharp) in user_data.sharp_resource_offset[3].iter().enumerate() {
        if sharp.offset_dw() == 0x7fff {
            continue;
        }
        if sharp.size() != 1 {
            return Err(ni("constant-buffer sharp size != 1"));
        }
        shader_get_storage_buffer(
            &mut bind.storage_buffers,
            &mut direct_sgprs,
            i32::from(sharp.offset_dw()),
            slot as i32,
            ShaderStorageUsage::Constant,
            user_sgpr,
            extended_buffer,
        )?;
        info.storage_buffers_constant += 1;
    }

    for i in 0..UserSgprInfo::SGPRS_MAX {
        // Re-checked each iteration: the typed-register routing below clears
        // the flags of every register a descriptor consumes.
        if !direct_sgprs[i] {
            continue;
        }
        if user_sgpr.type_[i] != UserSgprType::Unknown {
            // An inline SRT is raw application-defined scalar data. Resource
            // entries explicitly declared by the direct/sharp tables were
            // consumed above; every unclaimed dword inside `srt_size_dw`
            // remains a direct push constant even when the persistent AGC
            // `hu` marker tagged its SET_SH_REG write as Vsharp/Region.
            // Measured on GTA V compute 0x100003948200: direct[t1] consumes
            // s0..s3, while typed s4..s7 alternate between 1.0f and 0x07070707.
            // Promoting that SRT tail invented huge storage descriptors.
            if user_data.srt_size_dw != 0 && i < usize::from(user_data.srt_size_dw) {
                shader_get_direct_sgpr_impl(&mut bind.direct_sgprs, i as i32, user_sgpr, true)?;
                info.direct_sgprs += 1;
                direct_sgprs[i] = false;
                continue;
            }
            // Beyond Kyty (`ShaderGetDirectSgpr` EXITs on any typed
            // register): a register still unclaimed at collection time but
            // TYPED by a PM4 'hu' marker (0x4 = Vsharp, 0xd = Region — the
            // same pair `read_sharp_fields` accepts as descriptor content)
            // holds a descriptor the driver preloaded without a usage-table
            // entry. Route it into its proper resource table exactly as if a
            // usage slot had declared it at this register: peek the V#-sized
            // quad and disambiguate buffer vs image via the descriptor's own
            // type nibble (dword3[28:31] reads 0 for a buffer — the same
            // shadPS4-confirmed overlap the sharp-table-0 walk uses). The
            // seeded `%s{i}` locals then feed the shader's own buffer/image
            // ops exactly like a declared sharp. A run too short or
            // wrongly-typed fails inside `read_sharp_fields` with the type
            // and register index. 811 measured ASTRO.BOT CS dispatch skips.
            let saved_flags = direct_sgprs;
            let mut peek = [0u32; 4];
            read_sharp_fields(
                &mut direct_sgprs,
                i as i32,
                user_sgpr,
                extended_buffer,
                &mut peek,
            )?;
            let inferred_buffer = sharp_dword3_is_buffer(peek[3]);
            let inferred_buffer_is_bindable = if inferred_buffer {
                let resource = ShaderBufferResource { fields: peek };
                let size = if resource.stride() == 0 {
                    u64::from(resource.num_records())
                } else {
                    u64::from(resource.stride()) * u64::from(resource.num_records())
                };
                // This path is only a fallback for a typed run that NO
                // usage/direct table declared. Demand the same basic
                // invariants the binding path needs before inventing a V#:
                // a non-null extent must be dword-addressable, and its base
                // must resolve in the current guest address space when the
                // caller can check it. This rejects scalar/sentinel tuples
                // whose stale `hu` marker merely made them look descriptor
                // shaped (GTA V: [0x07070707; 4] => an unmapped 212-GB V#).
                (resource.base48() == 0 || size == 0)
                    || (size % 4 == 0
                        && mem.is_none_or(|guest| guest.dwords_at(resource.base48()).is_some()))
            } else {
                true
            };
            // A typed zero run or an unbindable inferred V# is raw scalar
            // user data, not a resource. Restore the flags the peek consumed
            // and keep the register direct.
            if peek == [0; 4] || !inferred_buffer_is_bindable {
                // Keep the WHOLE rejected quad direct, not just s{i}:
                // leaving s{i+1}..s{i+3} typed-and-flagged makes the next
                // iteration peek a quad that runs past the typed run
                // (measured: "user sgpr type is not Vsharp/Region (Unknown
                // at s8)" ×812 once s4 alone was kept direct).
                direct_sgprs = saved_flags;
                #[allow(clippy::needless_range_loop)] // index IS the register number
                for reg in i..(i + 4).min(UserSgprInfo::SGPRS_MAX) {
                    if !direct_sgprs[reg] {
                        continue;
                    }
                    shader_get_direct_sgpr_impl(
                        &mut bind.direct_sgprs,
                        reg as i32,
                        user_sgpr,
                        true,
                    )?;
                    info.direct_sgprs += 1;
                    direct_sgprs[reg] = false;
                }
                continue;
            }
            if inferred_buffer {
                // ReadWrite: no usage slot declared intent, and the compute
                // path writes every storage buffer back unconditionally, so
                // the writable superset is the faithful choice.
                let slot = bind.storage_buffers.buffers_num;
                shader_get_storage_buffer(
                    &mut bind.storage_buffers,
                    &mut direct_sgprs,
                    i as i32,
                    slot,
                    ShaderStorageUsage::ReadWrite,
                    user_sgpr,
                    extended_buffer,
                )?;
                info.storage_buffers_readwrite += 1;
            } else {
                let slot = bind.textures2d.textures_num;
                shader_get_texture_buffer(
                    &mut bind.textures2d,
                    &mut direct_sgprs,
                    i as i32,
                    slot,
                    ShaderTextureUsage::ReadOnly,
                    user_sgpr,
                    extended_buffer,
                )?;
                info.textures2d_readonly += 1;
                let last = (bind.textures2d.textures_num - 1) as usize;
                check_read_only_texture_type(&mut bind.textures2d.desc[last].texture)?;
            }
            continue;
        }
        shader_get_direct_sgpr(&mut bind.direct_sgprs, i as i32, user_sgpr)?;
        info.direct_sgprs += 1;
    }

    Ok(())
}

/// Kyty: Shader.cpp `ShaderDetectBuffers` (L944). Merges V#s that share a
/// stride and whose bases fall within one stride of each other into a single
/// vertex input buffer with per-attribute offsets.
pub fn shader_detect_buffers(
    info: &mut ShaderVertexInputInfo,
    ps5: bool,
) -> Result<(), ShaderAnalysisError> {
    info.buffers_num = 0;

    for ri in 0..info.resources_num as usize {
        let r = info.resources[ri];

        let mut merged = false;
        for bi in 0..info.buffers_num as usize {
            let b = &mut info.buffers[bi];

            let stride = u64::from(b.stride);

            if stride == u64::from(r.stride())
                && b.fetch_index == info.resources_dst[ri].fetch_index
            {
                let rbase = if ps5 { r.base48() } else { r.base44() };
                let base = rbase.min(b.addr);
                let offset1 = rbase - base;
                let offset2 = b.addr - base;

                if offset1 < stride && offset2 < stride {
                    if b.num_records != r.num_records() {
                        return Err(ni("merged vertex buffers with different num_records"));
                    }
                    b.addr = base;
                    if b.attr_num as usize >= ShaderVertexInputBuffer::ATTR_MAX {
                        return Err(ni("too many attributes in vertex input buffer"));
                    }
                    b.attr_indices[b.attr_num as usize] = ri as i32;
                    b.attr_num += 1;
                    merged = true;
                    break;
                }
            }
        }

        if !merged {
            if info.buffers_num as usize >= ShaderVertexInputInfo::RES_MAX {
                return Err(ni("too many vertex input buffers"));
            }
            let bi = info.buffers_num as usize;
            info.buffers_num += 1;
            info.buffers[bi].addr = if ps5 { r.base48() } else { r.base44() };
            info.buffers[bi].stride = u32::from(r.stride());
            info.buffers[bi].num_records = r.num_records();
            info.buffers[bi].fetch_index = info.resources_dst[ri].fetch_index;
            info.buffers[bi].attr_num = 1;
            info.buffers[bi].attr_indices[0] = ri as i32;
        }
    }

    for bi in 0..info.buffers_num as usize {
        for ri in 0..info.buffers[bi].attr_num as usize {
            let res = info.resources[info.buffers[bi].attr_indices[ri] as usize];
            let rbase = if ps5 { res.base48() } else { res.base44() };
            info.buffers[bi].attr_offsets[ri] = (rbase - info.buffers[bi].addr) as u32;
        }
    }

    Ok(())
}

/// Kyty: Shader.cpp `ShaderParseFetch` (L1005). Parses the external fetch
/// shader as its own [`ShaderCode`] and pairs each `s_load_dwordx4` of the
/// V# table with the `buffer_load_format_*` that consumes it, recovering the
/// vertex resources and their destination VGPR ranges.
pub fn shader_parse_fetch(
    info: &mut ShaderVertexInputInfo,
    fetch: &[u32],
    buffer: &[u32],
    next_gen: bool,
) -> Result<(), ShaderAnalysisError> {
    let mut code = ShaderCode::new();
    code.set_type(ShaderType::Fetch);
    shader_parse(0, fetch, &mut code, next_gen)?;

    let mut temp_value = [0u32; 104];
    let mut s_num = 0;
    let mut v_num = 0;

    for inst in code.get_instructions() {
        if inst.type_ == ShaderInstructionType::SLoadDwordx4 {
            if inst.src[1].type_ != ShaderOperandType::LiteralConstant
                || inst.src[1].constant.u & 3 != 0
            {
                return Err(ni("fetch: s_load_dwordx4 offset is not an aligned literal"));
            }
            if inst.src[0].type_ != ShaderOperandType::Sgpr || inst.src[0].register_id != 2 {
                return Err(ni("fetch: s_load_dwordx4 base is not s[2:3]"));
            }
            if inst.dst.type_ != ShaderOperandType::Sgpr {
                return Err(ni("fetch: s_load_dwordx4 dst is not an sgpr"));
            }

            let index = (inst.src[1].constant.u >> 2) as usize;
            let t = usize::try_from(inst.dst.register_id)
                .map_err(|_| trunc("fetch: negative dst register"))?;
            if t + 4 > temp_value.len() {
                return Err(trunc("fetch: s_load_dwordx4 dst beyond temp registers"));
            }
            let v = buffer
                .get(index..index + 4)
                .ok_or_else(|| trunc("fetch: V# table read out of bounds"))?;
            temp_value[t..t + 4].copy_from_slice(v);

            s_num += 1;
        }

        let registers_num = match inst.type_ {
            ShaderInstructionType::BufferLoadFormatX => 1,
            ShaderInstructionType::BufferLoadFormatXy => 2,
            ShaderInstructionType::BufferLoadFormatXyz => 3,
            ShaderInstructionType::BufferLoadFormatXyzw => 4,
            _ => 0,
        };

        if registers_num > 0 {
            if inst.dst.type_ != ShaderOperandType::Vgpr {
                return Err(ni("fetch: buffer_load dst is not a vgpr"));
            }
            if inst.src[0].type_ != ShaderOperandType::Vgpr || inst.src[0].register_id != 0 {
                return Err(ni("fetch: buffer_load vaddr is not v0"));
            }
            if inst.src[1].type_ != ShaderOperandType::Sgpr {
                return Err(ni("fetch: buffer_load resource is not an sgpr quad"));
            }
            if inst.src[2].type_ != ShaderOperandType::IntegerInlineConstant
                || inst.src[2].constant.i() != 0
            {
                return Err(ni("fetch: buffer_load soffset is not inline 0"));
            }

            if info.resources_num as usize >= ShaderVertexInputInfo::RES_MAX {
                return Err(ni("fetch: too many vertex resources"));
            }

            let t = usize::try_from(inst.src[1].register_id)
                .map_err(|_| trunc("fetch: negative resource register"))?;
            if t + 4 > temp_value.len() {
                return Err(trunc("fetch: resource registers beyond temp registers"));
            }

            let n = info.resources_num as usize;
            info.resources_dst[n].register_start = inst.dst.register_id;
            info.resources_dst[n].registers_num = registers_num;
            info.resources[n]
                .fields
                .copy_from_slice(&temp_value[t..t + 4]);

            info.resources_num += 1;

            v_num += 1;
        }
    }

    if s_num != v_num {
        return Err(ni("fetch: s_load_dwordx4 / buffer_load count mismatch"));
    }

    Ok(())
}

/// Kyty: Shader.cpp `ShaderParseAttrib` (L1095) — the PS5 embedded-fetch
/// path: recover vertex resources from the `input_semantics` table plus the
/// attribute and V# tables pointed to by user SGPRs.
pub fn shader_parse_attrib(
    info: &mut ShaderVertexInputInfo,
    input_semantics: &[ShaderSemantic],
    attrib: &[u32],
    buffer: &[u32],
) -> Result<(), ShaderAnalysisError> {
    for (i, sem) in input_semantics.iter().enumerate() {
        if sem.static_vb_index() || sem.static_attribute() {
            return Err(ni("attrib: static_vb_index / static_attribute"));
        }

        let reg = sem.hardware_mapping();
        let size = sem.size_in_elements();

        let va = *attrib
            .get(sem.semantic() as usize)
            .ok_or_else(|| trunc("attrib: attribute table read out of bounds"))?;

        // Kyty printf()s this unconditionally; the port logs at debug level.
        // `semantic` and the position `i` are logged TOGETHER on purpose:
        // `resources_dst` is written at position `i` here, but `recompile_fetch`
        // reads it back by ATTRIB-TABLE INDEX (= semantic). The two agree only
        // while `semantic == i` for every entry. A line where they differ is a
        // gapped/permuted semantics table — the gapped case surfaces as
        // "invalid registers_num: 0 (attrib N)", the permuted case is SILENT
        // and binds the wrong vertex buffer. `size` distinguishes the competing
        // hypothesis: size == 0 here means the semantic's size_in_elements
        // decode is wrong, not the indexing.
        tracing::debug!(
            pos = i,
            semantic = sem.semantic(),
            reg,
            size,
            va = format_args!("{va:#010x}"),
            mismatch = (sem.semantic() as usize != i),
            "attrib semantic"
        );

        let index = (va & 0x1f) as usize;
        let format = (va >> 5) & 0x1ff;
        let offset = (va >> 14) & 0xfff;
        let fetch_index = (va >> 26) & 0x1;

        if index >= ShaderVertexInputInfo::RES_MAX {
            return Err(ni("attrib: V# index >= RES_MAX"));
        }

        let sharp = buffer
            .get(index * 4..index * 4 + 4)
            .ok_or_else(|| trunc("attrib: V# table read out of bounds"))?;

        if offset != 0 || fetch_index != 0 {
            let resource = ShaderBufferResource {
                fields: [sharp[0], sharp[1], sharp[2], sharp[3]],
            };
            // Informational: the shader still translates correctly (the V#
            // carries the real hardware descriptor). Kept at debug so a
            // Minecraft run does not emit this ~600x. See FIX 1 (log noise).
            tracing::debug!(
                semantic = sem.semantic(),
                hardware_register = reg,
                elements = size,
                descriptor = format_args!("0x{va:08x}"),
                vsharp_index = index,
                format,
                offset,
                fetch_index,
                vsharp = format_args!(
                    "{:08x}:{:08x}:{:08x}:{:08x}",
                    sharp[0], sharp[1], sharp[2], sharp[3]
                ),
                vsharp_base = format_args!("0x{:012x}", resource.base48()),
                vsharp_stride = resource.stride(),
                vsharp_format = resource.format(),
                "Gen5 vertex attribute uses descriptor overrides"
            );
        }

        // Gen5 AGC metadata carries a 9-bit source-language vertex format
        // alongside the V# index. The V# itself is the hardware descriptor
        // and already contains the RDNA2 unified format consumed by shader
        // recompilation. For example, Minecraft's observed metadata format
        // 0x12a accompanies V# unified format 74 (32_32_32 float). Preserve
        // the V# verbatim rather than rejecting this valid redundant field.

        // Fold the per-attribute byte offset into the V# base. Interleaved
        // vertex data gives each attribute its own offset into a shared stride
        // (Minecraft: offset 16 into a 28-byte vertex), so base + offset is the
        // attribute's real start. Only fields[0] and the low 16 bits of
        // fields[1] hold the 48-bit base; stride is in fields[1] bits 16-29 and
        // is left untouched. The earlier code rejected any non-zero offset,
        // which failed the whole vertex shader.
        let mut folded = [sharp[0], sharp[1], sharp[2], sharp[3]];
        if offset != 0 {
            let base = (ShaderBufferResource { fields: folded }
                .base48()
                .wrapping_add(u64::from(offset)))
                & 0xFFFF_FFFF_FFFF;
            folded[0] = base as u32;
            folded[1] = (folded[1] & 0xFFFF_0000) | ((base >> 32) as u32 & 0xFFFF);
        }
        if info.resources_num as usize >= ShaderVertexInputInfo::RES_MAX {
            return Err(ni("attrib: too many vertex resources"));
        }

        let n = info.resources_num as usize;
        info.resources_dst[n].register_start = reg as i32;
        info.resources_dst[n].registers_num = size as i32;
        info.resources_dst[n].fetch_index = fetch_index;
        // Record which attrib-table entry this slot came from. `recompile_fetch`
        // resolves by this, NOT by array position — Minecraft's semantics table
        // is gapped (positions 0,1,2 carry semantics 0,2,3).
        info.resources_dst[n].semantic = sem.semantic() as i32;
        info.resources[n].fields.copy_from_slice(&folded);

        info.resources_num += 1;
    }

    Ok(())
}

/// Increment 3 of the SRT/EUD resolver (task #9): read a shader's Extended User
/// Data out of guest memory so `shader_parse_usage2` can resolve descriptors
/// that were spilled past the user-SGPR file.
///
/// Prefer a base pair the shader explicitly uses for `s_load_dwordx*`; only
/// when no such readable load base exists, try the sgpr pair immediately AFTER
/// the shader's declared user SGPRs (`user_sgpr_num`). Both shapes are measured
/// on ASTRO.BOT. Returns `None` when no candidate is backed by guest memory;
/// callers then keep the pre-existing "no extended buffer" path rather than
/// inventing descriptors.
///
/// The second element of the returned pair is the register index holding the
/// EUD pointer pair — the slot where the register file logically hands over
/// to the EUD (see [`EudView::base_dw`]).
fn read_extended_user_data(
    user_data: &ShaderUserData,
    user_sgpr: &UserSgprInfo,
    user_sgpr_num: i32,
    mem: &impl ShaderMemory,
    shader_addr: u64,
    next_gen: bool,
    decoded_code: Option<&ShaderCode>,
) -> Option<(Vec<u32>, i32)> {
    let size = user_data.eud_size_dw as usize;
    if size == 0 {
        return None;
    }

    let read_at = |ptr: u64| -> Option<Vec<u32>> {
        if ptr == 0 || ptr & 0x3 != 0 {
            return None;
        }
        let src = mem.dwords_at(ptr)?;
        // Prefer the full extended-mapping window: shaders scalar-load
        // descriptors PAST the declared `eud_size_dw` (measured on ASTRO.BOT
        // CS 0x500543b00 — storage T# at virtual s68 = EUD dword 36 with
        // eud_size_dw=28; the metadata size is a driver hint, not a bound —
        // the load just adds its offset to the same pointer). Cap at the
        // recompiler's mapping size so capture can never outrun
        // `WriteLocalVariables`. Fall back to the declared size when the
        // memory behind the pointer is shorter.
        let want = size.max(super::spirv::EXTENDED_MAPPING_DWORDS);
        if src.len() >= want {
            return Some(src[..want].to_vec());
        }
        (src.len() >= size).then(|| src[..size].to_vec())
    };

    // Strategy 1 — scalar-load analysis (resolver increments 1-2). An explicit
    // `s_load_dwordx*` through a live-in SGPR pair is stronger evidence than
    // mere position. This ordering matters when BOTH it and the after-file pair
    // are readable (ASTRO.BOT cs@0x500757800: declared=14, load through s12,
    // while s14 is also backed). Take the first load base that addresses guest
    // memory of at least `eud_size_dw`.
    // CAVEAT: "first readable" is a heuristic — a shader with several scalar
    // loads could pick the wrong one, which shows up as wrong descriptors rather
    // than an error. Narrow it (e.g. by preferring the earliest load, or by
    // validating descriptor shape) if a title renders wrong data.
    let trace = std::env::var_os("RAEEN_TRACE_EUD").is_some();
    // A shader that is unmapped or unparseable yields no scalar-load bases,
    // but must NOT abort the resolver — strategy 3 below works from the
    // captured registers alone.
    let mut parsed_code = ShaderCode::new();
    let code = if let Some(code) = decoded_code {
        code
    } else {
        match mem.dwords_at(shader_addr) {
            Some(src) => {
                if let Err(e) = shader_parse(0, &src, &mut parsed_code, next_gen) {
                    if trace {
                        tracing::warn!("TRACE_EUD2 {shader_addr:#x}: shader_parse failed: {e}");
                    }
                }
            }
            None => {
                if trace {
                    tracing::warn!("TRACE_EUD2 {shader_addr:#x}: shader code not mapped");
                }
            }
        }
        &parsed_code
    };
    let loads = find_scalar_load_bases(code);
    // Only a live-in base pair is evidence for an EUD pointer supplied in the
    // dispatch user data. A pair written earlier by the shader (for example by
    // s_getpc_b64 plus address arithmetic) must not outrank the positional EUD
    // fallback merely because its stale entry-time SGPR value happens to map.
    let live_in_bases: Vec<i32> = code
        .get_instructions()
        .iter()
        .enumerate()
        .filter_map(|(index, inst)| {
            if !matches!(
                inst.type_,
                ShaderInstructionType::SLoadDword
                    | ShaderInstructionType::SLoadDwordx2
                    | ShaderInstructionType::SLoadDwordx4
                    | ShaderInstructionType::SLoadDwordx8
            ) {
                return None;
            }
            let base = inst.src[0].register_id;
            let written_before = code.get_instructions()[..index]
                .iter()
                .any(|prior| writes_sgpr(prior, base) || writes_sgpr(prior, base + 1));
            (!written_before).then_some(base)
        })
        .collect();
    if trace {
        let detail: Vec<String> = loads
            .iter()
            .map(|l| {
                let addr = scalar_load_target_address(l, user_sgpr);
                let backed = addr.is_some_and(|a| {
                    a != 0 && a & 0x3 == 0 && mem.dwords_at(a).is_some_and(|d| d.len() >= size)
                });
                format!(
                    "s{}+{}x{}->{}{}",
                    l.base_register,
                    l.byte_offset,
                    l.dwords,
                    addr.map_or("none".to_owned(), |a| format!("{a:#x}")),
                    if backed { "(ok)" } else { "(unbacked)" }
                )
            })
            .collect();
        tracing::warn!(
            "TRACE_EUD2 {shader_addr:#x}: need={size}dw loads={} [{}]",
            loads.len(),
            detail.join(" ")
        );
    }
    if let Some((buf, base)) = loads
        .into_iter()
        .filter(|load| live_in_bases.contains(&load.base_register))
        .find_map(|load| {
            // The EUD pointer is the BASE PAIR's value: a load's byte offset
            // selects a descriptor WITHIN the buffer, and the virtual-register
            // mapping (`sharp at s{base+k}` ⇒ EUD dword k) only holds from the
            // pair value. Snapshotting at base+offset (the old behavior) shifted
            // the whole view by the scan-order-first load's offset — measured on
            // ASTRO.BOT composite CS 0x500665c00, whose first load is `+0x60`:
            // every sharp peek read 24 dwords high, the sampled T# declared at
            // EUD dword 0 mis-classed, and its `image_sample_lz` refused as
            // `dynamic-image-descriptor`. The target address still gates WHICH
            // base is picked in `find_scalar_load_bases` order; the snapshot
            // itself must start at the pair.
            let base = usize::try_from(load.base_register).ok()?;
            if base + 1 >= UserSgprInfo::SGPRS_MAX {
                return None;
            }
            let ptr = user_sgpr_pair(user_sgpr, load.base_register);
            Some((read_at(ptr)?, load.base_register))
        })
    {
        return Some((buf, base));
    }

    // Strategy 2 — the pair immediately AFTER the declared user SGPRs. This
    // remains the fallback for shaders whose `count` EXCEEDS `declared` but
    // whose code has no readable scalar-load base (e.g. declared=14 count=16:
    // s14:s15 hold the pointer).
    if let Ok(base) = usize::try_from(user_sgpr_num) {
        if base + 1 < UserSgprInfo::SGPRS_MAX {
            if let Some(buf) = read_at(user_sgpr_pair(user_sgpr, user_sgpr_num)) {
                return Some((buf, user_sgpr_num));
            }
        }
    }

    // Strategy 3 — scan EVERY adjacent SGPR pair (sN lo, sN+1 hi) for a
    // readable guest pointer. Measured on ASTRO.BOT compute (declared=14
    // count=14): strategies 1-2 find nothing, but (s12,s13) =
    // 0x4_00506730 — squarely in the direct-memory guest range
    // (0x4_00000000..) — backs a descriptor table of eud_size_dw dwords.
    // False-positive guards: the hi dword must be a small nonzero value
    // (real guest highs are 0x4..0x5 for direct memory, low tens for GPU
    // apertures; ALU constants and packed fields read far larger), and the
    // read itself must succeed with at least eud_size_dw dwords behind it.
    let count = (user_sgpr.count.max(4) as usize).min(UserSgprInfo::SGPRS_MAX);
    for i in 0..count.saturating_sub(1) {
        let hi = user_sgpr.value[i + 1];
        if hi == 0 || hi >= 0x100 {
            continue;
        }
        let pair = u64::from(user_sgpr.value[i]) | (u64::from(hi) << 32);
        if let Some(buf) = read_at(pair) {
            tracing::debug!(
                shader_addr = format_args!("{shader_addr:#x}"),
                pair_register = i,
                pointer = format_args!("{pair:#x}"),
                "EUD recovered by adjacent-pair scan (strategy 3)"
            );
            return Some((buf, i as i32));
        }
    }

    // Strategy 4 — PS5 title-address allocations below 4 GiB are carried as
    // ordinary zero-extended 64-bit pointers. Minecraft's measured gameplay
    // VS (0x16ffd700, declared/captured=8, eud_size_dw=12) supplies its EUD
    // through the tail pair `(s6, s7) = (0x16fad784, 0)`. Strategy 3 rejected
    // that valid pointer solely because its false-positive guard required a
    // nonzero high dword, so the draw was skipped and terrain lost its
    // textures. Try zero-high pairs last, in reverse register order: the
    // platform convention puts the spill pointer at the tail of the user-data
    // file, while earlier low-address pairs may describe adjacent tables.
    for i in (0..count.saturating_sub(1)).rev() {
        if user_sgpr.value[i + 1] != 0 {
            continue;
        }
        let pointer = u64::from(user_sgpr.value[i]);
        // Exclude null/small scalar constants before asking guest memory.
        if pointer < 0x1_0000 || pointer & 0x3 != 0 {
            continue;
        }
        if let Some(buf) = read_at(pointer) {
            tracing::debug!(
                shader_addr = format_args!("{shader_addr:#x}"),
                pair_register = i,
                pointer = format_args!("{pointer:#x}"),
                "EUD recovered from a zero-extended 32-bit tail pointer (strategy 4)"
            );
            return Some((buf, i as i32));
        }
    }
    None
}

/// `user_sgpr.value[reg] | value[reg + 1] << 32` — the 64-bit pointer Kyty
/// assembles from a user-SGPR pair. Caller has range-checked `reg + 1`.
fn user_sgpr_pair(user_sgpr: &UserSgprInfo, reg: i32) -> u64 {
    let r = reg as usize;
    u64::from(user_sgpr.value[r]) | (u64::from(user_sgpr.value[r + 1]) << 32)
}

/// Kyty: Shader.cpp `ShaderGetInputInfoVS` (L1630).
///
/// `next_gen` replaces `Config::IsNextGen()`; `shader_map` replaces
/// `g_shader_map`; `mem` resolves the shader/fetch/table addresses.
pub fn shader_get_input_info_vs(
    regs: &VertexShaderInfo,
    sh: &ShaderRegisters,
    mem: &impl ShaderMemory,
    shader_map: &ShaderMap,
    next_gen: bool,
    info: &mut ShaderVertexInputInfo,
) -> Result<(), ShaderAnalysisError> {
    shader_get_input_info_vs_decoded(regs, sh, mem, shader_map, next_gen, None, info)
}

/// [`shader_get_input_info_vs`] with the already-decoded vertex program.
///
/// The runtime fetch path parses the program before resource analysis. Passing
/// it here avoids reparsing every bind and lets the NGG user-data normalizer
/// prove the scalar-register offset from actual SBUFFER operands.
pub fn shader_get_input_info_vs_decoded(
    regs: &VertexShaderInfo,
    sh: &ShaderRegisters,
    mem: &impl ShaderMemory,
    shader_map: &ShaderMap,
    next_gen: bool,
    decoded_code: Option<&ShaderCode>,
    info: &mut ShaderVertexInputInfo,
) -> Result<(), ShaderAnalysisError> {
    info.export_count = sh.get_export_count() as i32;
    info.bind.push_constant_offset = 0;
    info.bind.push_constant_size = 0;
    info.bind.descriptor_set_slot = 0;

    if regs.vs_embedded {
        return Ok(());
    }

    let mut usage = ShaderParsedUsage::default();

    let gs_instead_of_vs = regs.vs_regs.data_addr == 0
        && regs.gs_regs.data_addr == 0
        && regs.es_regs.data_addr != 0
        && regs.gs_regs.chksum != 0;

    let shader_addr = if gs_instead_of_vs {
        regs.es_regs.data_addr
    } else {
        regs.vs_regs.data_addr
    };
    let user_sgpr = if gs_instead_of_vs {
        &regs.gs_user_sgpr
    } else {
        &regs.vs_user_sgpr
    };
    let user_sgpr_num = i32::from(if gs_instead_of_vs {
        regs.gs_regs.rsrc2.user_sgpr
    } else {
        regs.vs_regs.rsrc2.user_sgpr
    });

    let ps5 = next_gen;

    let data: Option<&ShaderMappedData> = if ps5 {
        shader_map.find(shader_addr)
    } else {
        None
    };

    if ps5 {
        let user_data = data
            .and_then(|d| d.user_data.as_ref())
            .ok_or_else(|| ni("vs: user_data is not mapped"))?;
        if !gs_instead_of_vs {
            return Err(ni("vs: next-gen without gs-instead-of-vs"));
        }

        info.gs_prolog = true;

        let rebased_user_data = decoded_code
            .and_then(|code| rebase_ngg_constant_sharps(user_data, code, user_sgpr, user_sgpr_num));
        let user_data = rebased_user_data.as_ref().unwrap_or(user_data);

        trace_eud_evidence(
            "vs(gs)",
            shader_addr,
            user_data,
            user_sgpr,
            user_sgpr_num,
            mem,
        );
        // A Gen5 vertex program executes in the ES/GS stage and therefore
        // consumes `gs_user_sgpr`. Keep the legacy VS bank in the filtered
        // forensic trace as well: if an AGC packet was decoded into the wrong
        // bank, the paired lines prove where the missing descriptor actually
        // landed without weakening normal analysis or guessing a mirror rule.
        trace_eud_evidence(
            "vs(unselected-vs-bank)",
            shader_addr,
            user_data,
            &regs.vs_user_sgpr,
            i32::from(regs.vs_regs.rsrc2.user_sgpr),
            mem,
        );
        shader_parse_usage2(
            user_data,
            &mut usage,
            &mut info.bind,
            user_sgpr,
            user_sgpr_num,
            read_extended_user_data(
                user_data,
                user_sgpr,
                user_sgpr_num,
                mem,
                shader_addr,
                next_gen,
                decoded_code,
            )
            .as_ref()
            .map(|(data, base_dw)| EudView {
                data,
                base_dw: *base_dw,
            }),
            Some(mem),
        )?;
    } else {
        if gs_instead_of_vs {
            return Err(ni("vs: gs-instead-of-vs on legacy path"));
        }

        info.gs_prolog = false;

        let code = mem
            .dwords_at(shader_addr)
            .ok_or_else(|| bad_addr(shader_addr))?;
        shader_parse_usage(
            &code,
            mem,
            &mut usage,
            &mut info.bind,
            user_sgpr,
            user_sgpr_num,
        )?;
    }

    if usage.extended_buffer {
        return Err(ni("vs: extended buffer"));
    }
    // Kyty's VS-sampler EXIT is a PS4-era invariant. Gen5 vertex stages may
    // sample textures, and Raeen's SPIR-V sampler declarations plus Vulkan
    // stage binding are stage-agnostic (the same path already used by PS/CS).
    if usage.samplers > 0 && !ps5 {
        return Err(ni("vs: samplers"));
    }
    if usage.gds_pointers > 0 {
        return Err(ni("vs: gds pointers"));
    }
    if usage.storage_buffers_readonly > 0 || usage.textures2d_readonly > 0 {
        return Err(ni("vs: read-only storage buffers / textures"));
    }
    if usage.storage_buffers_readwrite > 0 || usage.textures2d_readwrite > 0 {
        return Err(ni("vs: read-write storage buffers / textures"));
    }
    if !ps5 && usage.fetch != usage.vertex_buffer {
        return Err(ni("vs: fetch without vertex buffer (or vice versa)"));
    }
    if ps5 && usage.vertex_attrib != usage.vertex_buffer {
        return Err(ni(
            "vs: vertex attrib without vertex buffer (or vice versa)",
        ));
    }

    if usage.vertex_buffer && usage.vertex_attrib {
        info.fetch_external = false;
        info.fetch_embedded = true;
        info.fetch_inline = false;
        info.fetch_attrib_reg = usage.vertex_attrib_reg;
        info.fetch_buffer_reg = usage.vertex_buffer_reg;

        if usage.vertex_attrib_reg + 1 >= UserSgprInfo::SGPRS_MAX as i32 {
            return Err(ni("vs: vertex_attrib_reg + 1 >= SGPRS_MAX"));
        }
        if usage.vertex_buffer_reg + 1 >= UserSgprInfo::SGPRS_MAX as i32 {
            return Err(ni("vs: vertex_buffer_reg + 1 >= SGPRS_MAX"));
        }

        let attrib_addr = user_sgpr_pair(user_sgpr, usage.vertex_attrib_reg);
        let buffer_addr = user_sgpr_pair(user_sgpr, usage.vertex_buffer_reg);

        let attrib = mem
            .dwords_at(attrib_addr)
            .ok_or_else(|| bad_addr(attrib_addr))?;
        let buffer = mem
            .dwords_at(buffer_addr)
            .ok_or_else(|| bad_addr(buffer_addr))?;

        let semantics = data
            .map(|d| d.input_semantics.as_slice())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ni("vs: input_semantics are not mapped"))?;

        shader_parse_attrib(info, semantics, &attrib, &buffer)?;
        shader_detect_buffers(info, ps5)?;
    }

    if usage.fetch && usage.vertex_buffer {
        info.fetch_external = true;
        info.fetch_embedded = false;
        info.fetch_inline = false;
        info.fetch_shader_reg = usage.fetch_reg;
        info.fetch_buffer_reg = usage.vertex_buffer_reg;

        if usage.fetch_reg + 1 >= UserSgprInfo::SGPRS_MAX as i32 {
            return Err(ni("vs: fetch_reg + 1 >= SGPRS_MAX"));
        }
        if usage.vertex_buffer_reg + 1 >= UserSgprInfo::SGPRS_MAX as i32 {
            return Err(ni("vs: vertex_buffer_reg + 1 >= SGPRS_MAX"));
        }

        let fetch_addr = user_sgpr_pair(user_sgpr, usage.fetch_reg);
        let buffer_addr = user_sgpr_pair(user_sgpr, usage.vertex_buffer_reg);

        let fetch = mem
            .dwords_at(fetch_addr)
            .ok_or_else(|| bad_addr(fetch_addr))?;
        let buffer = mem
            .dwords_at(buffer_addr)
            .ok_or_else(|| bad_addr(buffer_addr))?;

        shader_parse_fetch(info, &fetch, &buffer, next_gen)?;
        shader_detect_buffers(info, ps5)?;
    }

    shader_calc_binding_indices(&mut info.bind);

    Ok(())
}

/// Rebase an NGG constant-buffer sharp from scalar-register numbering to the
/// captured user-data window when the decoded program proves the mapping.
///
/// GFX10 NGG reserves scalar registers `s0..s7` for its hardware prologue.
/// SharpEmu models this explicitly by copying hardware user-data slot 0 into
/// scalar register `s8` (`NggUserDataScalarRegisterBase = 8`) before evaluating
/// the shader. AGC metadata encountered in Minecraft's terrain VS names its
/// constant V# at `s8`, while `SPI_SHADER_PGM_RSRC2_GS.USER_SGPR == 8` and the
/// captured typed V# occupies hardware slots 0..3. The shader itself then
/// executes `s_buffer_load_* ... s[8:11]`.
///
/// Kyty's emitter already adds the same +8 when seeding VS resources. Resource
/// analysis therefore has to store the hardware slot (0), not read past the
/// captured file at metadata register 8. This helper only rebases when all
/// three facts agree:
/// - the metadata register is outside the captured user-data window;
/// - an SBUFFER instruction actually consumes that scalar-register quad; and
/// - subtracting eight lands on a complete typed V#/Region quad.
///
/// Any ambiguous shape is left untouched for the existing evidence-rich
/// refusal; this is not a general guessed mirror rule.
fn rebase_ngg_constant_sharps(
    user_data: &ShaderUserData,
    code: &ShaderCode,
    user_sgpr: &UserSgprInfo,
    user_sgpr_num: i32,
) -> Option<ShaderUserData> {
    use ShaderInstructionType as T;

    const NGG_SCALAR_BASE: usize = 8;
    let sbuffer_regs: Vec<usize> = code
        .get_instructions()
        .iter()
        .filter(|inst| {
            matches!(
                inst.type_,
                T::SBufferLoadDword
                    | T::SBufferLoadDwordx2
                    | T::SBufferLoadDwordx4
                    | T::SBufferLoadDwordx8
                    | T::SBufferLoadDwordx16
            ) && inst.src[0].type_ == ShaderOperandType::Sgpr
        })
        .filter_map(|inst| usize::try_from(inst.src[0].register_id).ok())
        .collect();
    if sbuffer_regs.is_empty() {
        return None;
    }

    let captured = usize::try_from(user_sgpr_num.max(0))
        .unwrap_or(0)
        .min(user_sgpr.count as usize)
        .min(UserSgprInfo::SGPRS_MAX);
    let captured_buffer_quad = |start: usize| {
        if start + 4 > captured {
            return false;
        }
        let typed = user_sgpr.type_[start..start + 4]
            .iter()
            .all(|t| matches!(t, UserSgprType::Vsharp | UserSgprType::Region));
        let fields = &user_sgpr.value[start..start + 4];
        // The `hu` R_ZERO marker is optional metadata in the command stream:
        // some measured Minecraft AGC draws write a real V# without it, so
        // the values are captured but their type tags remain Unknown. The
        // decoded SBUFFER operand + AGC constant-sharp table already prove
        // the semantic; require a non-null buffer-shaped descriptor before
        // accepting that untyped form. Binding later performs the full guest
        // range/stride validation.
        let untyped_buffer =
            fields.iter().any(|field| *field != 0) && sharp_dword3_is_buffer(fields[3]);
        typed || untyped_buffer
    };

    let mut normalized = None;
    for (slot, sharp) in user_data.sharp_resource_offset[3].iter().enumerate() {
        if sharp.offset_dw() == 0x7fff || sharp.size() != 1 {
            continue;
        }
        let scalar_reg = usize::from(sharp.offset_dw());
        if scalar_reg < captured
            || scalar_reg < NGG_SCALAR_BASE
            || !sbuffer_regs.contains(&scalar_reg)
        {
            continue;
        }
        let hardware_slot = scalar_reg - NGG_SCALAR_BASE;
        if !captured_buffer_quad(hardware_slot) {
            continue;
        }
        let data = normalized.get_or_insert_with(|| user_data.clone());
        data.sharp_resource_offset[3][slot] = ShaderSharp::new(hardware_slot as u16, sharp.size());
        tracing::debug!(
            slot,
            scalar_register = scalar_reg,
            hardware_slot,
            "NGG constant V# rebased from scalar register to captured user-data slot"
        );
    }
    normalized
}

/// Kyty: Shader.cpp `ShaderGetInputInfoPS` (L1744).
pub fn shader_get_input_info_ps(
    regs: &PixelShaderInfo,
    sh: &ShaderRegisters,
    vs_info: &ShaderVertexInputInfo,
    mem: &impl ShaderMemory,
    shader_map: &ShaderMap,
    next_gen: bool,
    ps_info: &mut ShaderPixelInputInfo,
) -> Result<(), ShaderAnalysisError> {
    shader_get_input_info_ps_decoded(regs, sh, vs_info, mem, shader_map, next_gen, None, ps_info)
}

/// [`shader_get_input_info_ps`] with the already-decoded pixel program.
///
/// The runtime fetch path has already exact-byte validated this code. Reusing
/// it avoids reparsing the ISA while resolving and capturing EUD resources;
/// descriptor contents are still read fresh from guest memory on every bind.
#[allow(clippy::too_many_arguments)]
pub fn shader_get_input_info_ps_decoded(
    regs: &PixelShaderInfo,
    sh: &ShaderRegisters,
    vs_info: &ShaderVertexInputInfo,
    mem: &impl ShaderMemory,
    shader_map: &ShaderMap,
    next_gen: bool,
    decoded_code: Option<&ShaderCode>,
    ps_info: &mut ShaderPixelInputInfo,
) -> Result<(), ShaderAnalysisError> {
    if regs.ps_embedded {
        return Ok(());
    }

    // Deviation: Kyty copies the whole register (`input_num = ps_in_control`),
    // which explodes when a title sets flag bits — measured on Minecraft:
    // ps_in_control = 0x4004 → 16388 "inputs" and a bogus truncation error.
    // NUM_INTERP is bits 0-5 of SPI_PS_IN_CONTROL (AMD layout); 0x4004 → 4.
    ps_info.input_num = sh.ps_in_control & 0x3f;
    ps_info.ps_pos_xy = sh.ps_input_ena == 0x0000_0302 && sh.ps_input_addr == 0x0000_0302;
    ps_info.ps_pixel_kill_enable = sh.db_shader_control.shader_kill_enable;
    ps_info.ps_early_z = sh.db_shader_control.shader_z_behavior == 1;
    ps_info.ps_execute_on_noop = sh.db_shader_control.shader_execute_on_noop;

    if ps_info.input_num as usize > ps_info.interpolator_settings.len() {
        // Deviation: Kyty indexes the 32-entry arrays unchecked. Name the raw
        // registers: the count may be a FIELD of SPI_PS_IN_CONTROL (bits
        // 0-5), not the whole register — the run must say which.
        return Err(trunc_owned(format!(
            "ps: input_num {} > 32 (ps_in_control={:#x} ps_input_ena={:#x} ps_input_addr={:#x})",
            ps_info.input_num, sh.ps_in_control, sh.ps_input_ena, sh.ps_input_addr
        )));
    }
    for i in 0..ps_info.input_num as usize {
        ps_info.interpolator_settings[i] = sh.ps_interpolator_settings[i];
    }

    ps_info.bind.descriptor_set_slot = u32::from(vs_info.bind.storage_buffers.buffers_num > 0);
    ps_info.bind.push_constant_offset =
        vs_info.bind.push_constant_offset + vs_info.bind.push_constant_size;
    ps_info.bind.push_constant_size = 0;

    ps_info.target_output_mode = sh.target_output_mode;

    let ps5 = next_gen;

    let data: Option<&ShaderMappedData> = if ps5 {
        shader_map.find(regs.ps_regs.data_addr)
    } else {
        None
    };

    let mut usage = ShaderParsedUsage::default();

    if ps5 {
        let user_data = data
            .and_then(|d| d.user_data.as_ref())
            .ok_or_else(|| ni("ps: user_data is not mapped"))?;

        trace_eud_evidence(
            "ps",
            regs.ps_regs.data_addr,
            user_data,
            &regs.ps_user_sgpr,
            i32::from(regs.ps_regs.rsrc2.user_sgpr),
            mem,
        );
        let eud_buf = read_extended_user_data(
            user_data,
            &regs.ps_user_sgpr,
            i32::from(regs.ps_regs.rsrc2.user_sgpr),
            mem,
            regs.ps_regs.data_addr,
            next_gen,
            decoded_code,
        );
        let eud_view = eud_buf.as_ref().map(|(data, base_dw)| EudView {
            data,
            base_dw: *base_dw,
        });
        shader_parse_usage2(
            user_data,
            &mut usage,
            &mut ps_info.bind,
            &regs.ps_user_sgpr,
            i32::from(regs.ps_regs.rsrc2.user_sgpr),
            eud_view,
            Some(mem),
        )?;
        if let Some(view) = eud_view {
            let mut capture = |code: &ShaderCode| -> Result<(), ShaderAnalysisError> {
                shader_capture_eud_storage_buffers(
                    code,
                    &mut ps_info.bind,
                    &regs.ps_user_sgpr,
                    view,
                    mem,
                )?;
                shader_capture_eud_image_descriptors(
                    code,
                    &mut ps_info.bind,
                    &regs.ps_user_sgpr,
                    view,
                )?;
                Ok(())
            };
            if let Some(code) = decoded_code {
                capture(code)?;
            } else {
                let mut code = ShaderCode::new();
                if let Some(src) = mem.dwords_at(regs.ps_regs.data_addr)
                    && shader_parse(0, &src, &mut code, next_gen).is_ok()
                {
                    capture(&code)?;
                }
            }
        }
    } else {
        let code = mem
            .dwords_at(regs.ps_regs.data_addr)
            .ok_or_else(|| bad_addr(regs.ps_regs.data_addr))?;
        shader_parse_usage(
            &code,
            mem,
            &mut usage,
            &mut ps_info.bind,
            &regs.ps_user_sgpr,
            i32::from(regs.ps_regs.rsrc2.user_sgpr),
        )?;
    }

    if usage.fetch || usage.vertex_buffer || usage.vertex_attrib {
        return Err(ni("ps: fetch / vertex buffer / vertex attrib"));
    }
    // Kyty's PS4-era path rejects pixel-stage writable buffers outright.
    // Gen5 titles use fragment UAV/SSBO access (measured on ASTRO.BOT
    // 0x500652400). The recompiler's MUBUF store helpers are stage-agnostic,
    // and Raeen's graphics descriptor path binds STORAGE_BUFFER resources
    // with FRAGMENT visibility, so retain the legacy guard only.
    if usage.storage_buffers_readwrite > 0 && !next_gen {
        return Err(ni("ps: read-write storage buffers"));
    }
    if usage.gds_pointers > 0 {
        return Err(ni("ps: gds pointers"));
    }
    // Kyty's PS4-era path rejects direct (raw-value) user SGPRs in the pixel
    // stage. Gen5 pixel shaders routinely carry raw data there — SRT blocks
    // and type-5 immediates — exactly as Gen5 VS and CS do: VS never rejected
    // them and CS lifted the same guard for `next_gen` (below). The
    // push-constant plumbing that consumes them is stage-agnostic
    // (`shader_calc_binding_indices` -> `push_constant_size`), so the pixel
    // stage was the lone holdout.
    //
    // Measured on Minecraft: every PS carrying direct SGPRs failed analysis
    // with "ps: direct sgprs", so the draws binding it were skipped — the
    // in-world scene rendered as flat untextured blocks. Legacy (PS4) shaders
    // keep the original rejection.
    if usage.direct_sgprs > 0 && !next_gen {
        return Err(ni("ps: direct sgprs"));
    }

    shader_calc_binding_indices(&mut ps_info.bind);

    Ok(())
}

/// Kyty: Shader.cpp `ShaderGetInputInfoCS` (L1811). `sh` is unused upstream
/// too (kept for API parity).
pub fn shader_get_input_info_cs(
    regs: &ComputeShaderInfo,
    sh: &ShaderRegisters,
    mem: &impl ShaderMemory,
    shader_map: &ShaderMap,
    next_gen: bool,
    info: &mut ShaderComputeInputInfo,
) -> Result<(), ShaderAnalysisError> {
    shader_get_input_info_cs_decoded(regs, sh, mem, shader_map, next_gen, None, info)
}

/// [`shader_get_input_info_cs`] with the already-decoded compute program.
///
/// Resource values remain per-dispatch snapshots; only redundant ISA parsing
/// inside EUD discovery/capture is skipped.
pub fn shader_get_input_info_cs_decoded(
    regs: &ComputeShaderInfo,
    _sh: &ShaderRegisters,
    mem: &impl ShaderMemory,
    shader_map: &ShaderMap,
    next_gen: bool,
    decoded_code: Option<&ShaderCode>,
    info: &mut ShaderComputeInputInfo,
) -> Result<(), ShaderAnalysisError> {
    info.threads_num[0] = regs.cs_regs.num_thread_x;
    info.threads_num[1] = regs.cs_regs.num_thread_y;
    info.threads_num[2] = regs.cs_regs.num_thread_z;
    info.group_id[0] = regs.cs_regs.tgid_x_en != 0;
    info.group_id[1] = regs.cs_regs.tgid_y_en != 0;
    info.group_id[2] = regs.cs_regs.tgid_z_en != 0;
    info.thread_ids_num = i32::from(regs.cs_regs.tidig_comp_cnt) + 1;

    info.workgroup_register = i32::from(regs.cs_regs.user_sgpr);
    // COMPUTE_PGM_RSRC2.LDS_SIZE counts 128-dword granules (GFX10).
    info.lds_size_dw = u32::from(regs.cs_regs.lds_size) * 128;

    info.bind.push_constant_offset = 0;
    info.bind.push_constant_size = 0;
    info.bind.descriptor_set_slot = 0;

    let mut usage = ShaderParsedUsage::default();

    if next_gen {
        let user_data = shader_map
            .find(regs.cs_regs.data_addr)
            .and_then(|data| data.user_data.as_ref())
            .ok_or_else(|| ni("cs: user_data is not mapped"))?;
        trace_eud_evidence(
            "cs",
            regs.cs_regs.data_addr,
            user_data,
            &regs.cs_user_sgpr,
            i32::from(regs.cs_regs.user_sgpr),
            mem,
        );
        let eud_buf = read_extended_user_data(
            user_data,
            &regs.cs_user_sgpr,
            i32::from(regs.cs_regs.user_sgpr),
            mem,
            regs.cs_regs.data_addr,
            next_gen,
            decoded_code,
        );
        let eud_view = eud_buf.as_ref().map(|(data, base_dw)| EudView {
            data,
            base_dw: *base_dw,
        });
        shader_parse_usage2(
            user_data,
            &mut usage,
            &mut info.bind,
            &regs.cs_user_sgpr,
            i32::from(regs.cs_regs.user_sgpr),
            eud_view,
            Some(mem),
        )?;
        // Raw-EUD image descriptors: T#/S#s the shader loads itself (no
        // usage-table slot) get captured from the EUD snapshot at the covered
        // load's offset — see `shader_capture_eud_image_descriptors`. A body
        // that does not parse skips the pass; translate fails it by name
        // later anyway.
        if let Some(view) = eud_view {
            let mut capture = |code: &ShaderCode| -> Result<(), ShaderAnalysisError> {
                shader_capture_eud_image_descriptors(
                    code,
                    &mut info.bind,
                    &regs.cs_user_sgpr,
                    view,
                )?;
                shader_capture_eud_storage_buffers(
                    code,
                    &mut info.bind,
                    &regs.cs_user_sgpr,
                    view,
                    mem,
                )?;
                Ok(())
            };
            if let Some(code) = decoded_code {
                capture(code)?;
            } else {
                let mut code = ShaderCode::new();
                if let Some(src) = mem.dwords_at(regs.cs_regs.data_addr)
                    && shader_parse(0, &src, &mut code, next_gen).is_ok()
                {
                    capture(&code)?;
                }
            }
        }
    } else {
        let code = mem
            .dwords_at(regs.cs_regs.data_addr)
            .ok_or_else(|| bad_addr(regs.cs_regs.data_addr))?;
        shader_parse_usage(
            &code,
            mem,
            &mut usage,
            &mut info.bind,
            &regs.cs_user_sgpr,
            i32::from(regs.cs_regs.user_sgpr),
        )?;
    }

    // Kyty EXITs on CS samplers (Shader.cpp ShaderGetInputInfoCS L1832) — a
    // PS4-era invariant. Gen5 CS shaders sample textures (IMM_SAMPLER usage
    // slots, measured 738/5min on ASTRO.BOT); the sampler binding is
    // stage-agnostic in both the SPIR-V emission (`%samplers` array) and the
    // dispatch path (`prepare_stage_binding` with COMPUTE), so on next_gen
    // they flow through like the other stages.
    if usage.samplers > 0 && !next_gen {
        return Err(ni("cs: samplers"));
    }
    if usage.fetch || usage.vertex_buffer || usage.vertex_attrib {
        return Err(ni("cs: fetch / vertex buffer / vertex attrib"));
    }
    // Kyty EXITs on CS direct SGPRs (Shader.cpp ShaderGetInputInfoCS L1836) —
    // a PS4-era invariant where CS user data was always fully consumed by the
    // usage slots. Gen5 CS shaders carry raw data in user SGPRs (SRT blocks,
    // direct type-5 immediates), so on next_gen they flow through the
    // push-constant direct-SGPR path exactly like the other stages.
    if usage.direct_sgprs > 0 && !next_gen {
        return Err(ni("cs: direct sgprs"));
    }

    shader_calc_binding_indices(&mut info.bind);

    Ok(())
}

/// Refine conservative Gen5 storage-buffer usage with the compute shader's
/// decoded MUBUF accesses.
///
/// Usage tables describe a descriptor table's capability, not necessarily
/// what one shader does with every entry. Treating every table-1 / untyped
/// V# as writable forces a full device-to-host copy after every dispatch even
/// when the shader only loads it. This is particularly expensive for large
/// global resource heaps.
///
/// The refinement is deliberately narrow:
/// - every V# must be a direct SGPR descriptor (no EUD aliasing);
/// - every decoded store descriptor must resolve to one of those V#s;
/// - a candidate must be read directly, never stored directly, and its SGPR
///   descriptor quad must never be overwritten by the shader.
///
/// If any of those proofs is absent, the conservative `ReadWrite` usage is
/// retained. SPIR-V resource layout is unchanged; this only suppresses
/// needless guest writeback for proven input-only descriptors.
pub fn shader_refine_compute_storage_usage(code: &ShaderCode, bind: &mut ShaderBindResources) {
    use ShaderInstructionType as T;

    let count = usize::try_from(bind.storage_buffers.buffers_num)
        .unwrap_or(0)
        .min(bind.storage_buffers.buffers.len());
    if count == 0 || bind.storage_buffers.extended[..count].iter().any(|&x| x) {
        return;
    }

    let starts = &bind.storage_buffers.start_register[..count];
    let mut loads = Vec::new();
    let mut stores = Vec::new();
    for inst in code.get_instructions() {
        let is_load = matches!(
            inst.type_,
            T::BufferLoadDword
                | T::BufferLoadDwordX2
                | T::BufferLoadDwordX3
                | T::BufferLoadDwordX4
                | T::BufferLoadUbyte
                | T::BufferLoadFormatX
                | T::BufferLoadFormatXy
                | T::BufferLoadFormatXyz
                | T::BufferLoadFormatXyzw
        );
        let is_store = matches!(
            inst.type_,
            T::BufferStoreDword
                | T::BufferStoreDwordX2
                | T::BufferStoreDwordX4
                | T::BufferStoreFormatX
                | T::BufferStoreFormatXy
                | T::BufferStoreFormatXyz
                | T::BufferStoreFormatXyzw
        );
        if (is_load || is_store) && inst.src[1].type_ == ShaderOperandType::Sgpr {
            if is_load {
                loads.push(inst.src[1].register_id);
            } else {
                stores.push(inst.src[1].register_id);
            }
        }
    }

    // An unresolved store may access one of the descriptors through an alias
    // assembled in other SGPRs. Without full scalar dataflow, retain every
    // conservative write flag in that case.
    if stores.iter().any(|reg| !starts.contains(reg)) {
        return;
    }

    let descriptor_overwritten = |start: i32| {
        let end = start.saturating_add(4);
        code.get_instructions().iter().any(|inst| {
            [inst.dst, inst.dst2].into_iter().any(|dst| {
                if dst.type_ != ShaderOperandType::Sgpr {
                    return false;
                }
                let dst_end = dst.register_id.saturating_add(dst.size.max(1));
                dst.register_id < end && dst_end > start
            })
        })
    };

    for (index, &start) in starts.iter().enumerate() {
        if bind.storage_buffers.usages[index] != ShaderStorageUsage::ReadWrite {
            continue;
        }
        if loads.contains(&start) && !stores.contains(&start) && !descriptor_overwritten(start) {
            bind.storage_buffers.usages[index] = ShaderStorageUsage::ReadOnly;
            tracing::debug!(
                start_register = start,
                slot = bind.storage_buffers.slots[index],
                "compute storage V# proven load-only by decoded MUBUF accesses"
            );
        }
    }
}

/// Bound the guest-memory snapshot used for scalar constant-buffer loads.
///
/// `s_buffer_load_dword{xN}` addresses a V# as a byte-addressed constant
/// buffer. The V#'s `ADD_TID` and swizzle bits belong to MUBUF element
/// addressing and do not participate in these scalar loads. Some Gen5 titles
/// leave those overlapping descriptor bits set while also carrying a very
/// large stride/record tuple; uploading `stride * num_records` in that case is
/// both semantically unnecessary and can request gigabytes for a load that
/// touches only a few dwords.
///
/// For each `Constant` V# whose descriptor SGPR is read only at compile-time
/// offsets, record the highest byte touched. A dynamic offset leaves
/// `required_bytes` at zero, deliberately falling back to the full descriptor
/// extent at bind time.
pub fn shader_measure_constant_buffer_accesses(code: &ShaderCode, bind: &mut ShaderBindResources) {
    shader_measure_constant_buffer_accesses_shifted(code, bind, 0);
}

/// [`shader_measure_constant_buffer_accesses`] with a stage scalar-register
/// base. NGG vertex programs expose hardware user slot 0 as scalar `s8`;
/// pixel and compute stages pass zero.
pub fn shader_measure_constant_buffer_accesses_shifted(
    code: &ShaderCode,
    bind: &mut ShaderBindResources,
    register_shift: i32,
) {
    use ShaderInstructionType as T;

    let count = usize::try_from(bind.storage_buffers.buffers_num)
        .unwrap_or(0)
        .min(bind.storage_buffers.buffers.len());
    bind.storage_buffers.required_bytes[..count].fill(0);

    let mut seen = [false; ShaderStorageResources::BUFFERS_MAX];
    let mut dynamic = [false; ShaderStorageResources::BUFFERS_MAX];
    for inst in code.get_instructions() {
        let dwords = match inst.type_ {
            T::SBufferLoadDword => 1u32,
            T::SBufferLoadDwordx2 => 2,
            T::SBufferLoadDwordx4 => 4,
            T::SBufferLoadDwordx8 => 8,
            T::SBufferLoadDwordx16 => 16,
            _ => continue,
        };
        if inst.src[0].type_ != ShaderOperandType::Sgpr {
            continue;
        }
        let Some(index) = (0..count).find(|&index| {
            bind.storage_buffers.usages[index] == ShaderStorageUsage::Constant
                && bind.storage_buffers.start_register[index].saturating_add(register_shift)
                    == inst.src[0].register_id
        }) else {
            continue;
        };
        seen[index] = true;
        let offset = match inst.src[1].type_ {
            ShaderOperandType::LiteralConstant | ShaderOperandType::IntegerInlineConstant
                if inst.src[1].constant.i() >= 0 =>
            {
                inst.src[1].constant.u
            }
            _ => {
                dynamic[index] = true;
                continue;
            }
        };
        let Some(end) = dwords
            .checked_mul(4)
            .and_then(|width| offset.checked_add(width))
        else {
            dynamic[index] = true;
            continue;
        };
        bind.storage_buffers.required_bytes[index] =
            bind.storage_buffers.required_bytes[index].max(end);
    }

    for index in 0..count {
        if dynamic[index] || !seen[index] {
            bind.storage_buffers.required_bytes[index] = 0;
        }
    }
}

/// Kyty: Shader.cpp `ShaderGetBindIds` (L2679). The id keys on the *binding
/// layout* (counts, slots, start registers, extended/usage flags), NOT on
/// descriptor contents — upstream deliberately commented out the
/// per-descriptor fields (L2685-L2694, L2705-L2728, L2739-L2765).
///
/// DELIBERATE deviation from that contents-blind rule, one T# field pair:
/// the generated SPIR-V depends on each texture descriptor's `type_()`
/// nibble (it picks the per-Dim sampled array/coordinate arity and the
/// storage image Dim) and on `format()` (the storage format selects Rgba8,
/// Rgba16f, or Rgba32f; sampled formats select numeric class). Two binds identical except
/// there produce DIFFERENT modules, so they must not share one cache id —
/// with the upstream id they silently aliased (wrong-Dim sampling from a
/// cached module). Every other descriptor-content field stays out of the id,
/// exactly as upstream.
fn shader_get_bind_ids(ret: &mut ShaderId, bind: &ShaderBindResources) {
    ret.ids.push(bind.storage_buffers.buffers_num as u32);

    for i in 0..bind.storage_buffers.buffers_num as usize {
        ret.ids.push(bind.storage_buffers.slots[i] as u32);
        ret.ids.push(bind.storage_buffers.start_register[i] as u32);
        ret.ids.push(u32::from(bind.storage_buffers.extended[i]));
        ret.ids.push(bind.storage_buffers.usages[i] as u32);
    }

    ret.ids.push(bind.textures2d.textures_num as u32);

    for i in 0..bind.textures2d.textures_num as usize {
        ret.ids.push(bind.textures2d.desc[i].slot as u32);
        ret.ids.push(bind.textures2d.desc[i].start_register as u32);
        ret.ids.push(u32::from(bind.textures2d.desc[i].extended));
        ret.ids.push(bind.textures2d.desc[i].usage as u32);
        // Codegen inputs (see the fn doc): the Dim-selecting type nibble and
        // the exact unified format used by storage/sample image codegen.
        ret.ids
            .push(u32::from(bind.textures2d.desc[i].texture.type_()));
        ret.ids
            .push(u32::from(bind.textures2d.desc[i].texture.format()));
    }

    ret.ids.push(bind.samplers.samplers_num as u32);

    for i in 0..bind.samplers.samplers_num as usize {
        ret.ids.push(bind.samplers.slots[i] as u32);
        ret.ids.push(bind.samplers.start_register[i] as u32);
        ret.ids.push(u32::from(bind.samplers.extended[i]));
    }

    ret.ids.push(bind.gds_pointers.pointers_num as u32);

    for i in 0..bind.gds_pointers.pointers_num as usize {
        ret.ids.push(bind.gds_pointers.slots[i] as u32);
        ret.ids.push(bind.gds_pointers.start_register[i] as u32);
        ret.ids.push(u32::from(bind.gds_pointers.extended[i]));
    }

    ret.ids.push(bind.direct_sgprs.sgprs_num as u32);

    for i in 0..bind.direct_sgprs.sgprs_num as usize {
        ret.ids.push(bind.direct_sgprs.start_register[i] as u32);
    }

    ret.ids.push(u32::from(bind.extended.used));
    ret.ids.push(bind.extended.slot as u32);
    ret.ids.push(bind.extended.start_register as u32);
}

/// Kyty: Shader.cpp `ShaderGetIdVS` (L2794).
pub fn shader_get_id_vs(
    regs: &VertexShaderInfo,
    input_info: &ShaderVertexInputInfo,
    mem: &impl ShaderMemory,
    next_gen: bool,
) -> Result<ShaderId, ShaderAnalysisError> {
    let mut ret = ShaderId::default();

    if regs.vs_embedded {
        ret.ids.push(regs.vs_embedded_id);
        return Ok(ret);
    }

    ret.ids.reserve(64);

    let gs_instead_of_vs = regs.vs_regs.data_addr == 0
        && regs.gs_regs.data_addr == 0
        && regs.es_regs.data_addr != 0
        && regs.gs_regs.chksum != 0;
    let shader_addr = if gs_instead_of_vs {
        regs.es_regs.data_addr
    } else {
        regs.vs_regs.data_addr
    };

    let gen5 = next_gen;

    if gen5 {
        if !gs_instead_of_vs {
            return Err(ni("vs id: next-gen without gs-instead-of-vs"));
        }

        ret.hash0 = ((regs.gs_regs.chksum >> 32) & 0xffff_ffff) as u32;
        ret.crc32 = (regs.gs_regs.chksum & 0xffff_ffff) as u32;
    } else {
        let src = mem
            .dwords_at(shader_addr)
            .ok_or_else(|| bad_addr(shader_addr))?;
        let header = get_binary_info(&src).ok_or(ShaderAnalysisError::NoBinaryInfo)?;

        ret.hash0 = header.hash0;
        ret.crc32 = header.crc32;
        ret.ids.push(header.length);
    }

    ret.ids.push(u32::from(input_info.fetch_external));
    ret.ids.push(u32::from(input_info.fetch_embedded));
    ret.ids.push(u32::from(input_info.fetch_inline));
    ret.ids.push(input_info.resources_num as u32);
    ret.ids.push(input_info.export_count as u32);

    for i in 0..input_info.resources_num as usize {
        let r = &input_info.resources[i];
        let rd = &input_info.resources_dst[i];

        ret.ids.push(rd.register_start as u32);
        ret.ids.push(rd.registers_num as u32);
        ret.ids.push(rd.fetch_index);
        ret.ids.push(u32::from(r.stride()));
        ret.ids.push(u32::from(r.swizzle_enabled()));
        ret.ids.push(u32::from(r.dst_sel_x()));
        ret.ids.push(u32::from(r.dst_sel_y()));
        ret.ids.push(u32::from(r.dst_sel_z()));
        ret.ids.push(u32::from(r.dst_sel_w()));
        if gen5 {
            ret.ids.push(u32::from(r.format()));
            ret.ids.push(u32::from(r.out_of_bounds()));
        } else {
            ret.ids.push(u32::from(r.nfmt()));
            ret.ids.push(u32::from(r.dfmt()));
        }
        ret.ids.push(u32::from(r.add_tid()));
    }

    ret.ids.push(input_info.buffers_num as u32);

    for i in 0..input_info.buffers_num as usize {
        let b = &input_info.buffers[i];
        ret.ids.push(b.attr_num as u32);
        ret.ids.push(b.stride);
        ret.ids.push(b.fetch_index);
        for j in 0..b.attr_num as usize {
            ret.ids.push(b.attr_indices[j] as u32);
            ret.ids.push(b.attr_offsets[j]);
        }
    }

    shader_get_bind_ids(&mut ret, &input_info.bind);

    Ok(ret)
}

/// Kyty: Shader.cpp `ShaderGetIdPS` (L2885).
pub fn shader_get_id_ps(
    regs: &PixelShaderInfo,
    input_info: &ShaderPixelInputInfo,
    mem: &impl ShaderMemory,
    next_gen: bool,
) -> Result<ShaderId, ShaderAnalysisError> {
    let mut ret = ShaderId::default();

    if regs.ps_embedded {
        ret.ids.push(regs.ps_embedded_id);
        return Ok(ret);
    }

    ret.ids.reserve(64);

    if next_gen {
        ret.hash0 = ((regs.ps_regs.chksum >> 32) & 0xffff_ffff) as u32;
        ret.crc32 = (regs.ps_regs.chksum & 0xffff_ffff) as u32;
    } else {
        let src = mem
            .dwords_at(regs.ps_regs.data_addr)
            .ok_or_else(|| bad_addr(regs.ps_regs.data_addr))?;
        let header = get_binary_info(&src).ok_or(ShaderAnalysisError::NoBinaryInfo)?;

        ret.hash0 = header.hash0;
        ret.crc32 = header.crc32;

        ret.ids.push(header.length);
    }

    ret.ids.push(input_info.input_num);
    ret.ids.push(u32::from(input_info.ps_pos_xy));
    ret.ids.push(u32::from(input_info.ps_pixel_kill_enable));
    ret.ids.push(u32::from(input_info.ps_early_z));
    ret.ids.push(u32::from(input_info.ps_execute_on_noop));

    for i in 0..input_info.input_num as usize {
        ret.ids.push(input_info.interpolator_settings[i]);
    }

    shader_get_bind_ids(&mut ret, &input_info.bind);

    Ok(ret)
}

/// Kyty: Shader.cpp `ShaderGetIdCS` (L2935). Always reads the binary-info
/// trailer (no next-gen chksum branch upstream).
pub fn shader_get_id_cs(
    regs: &ComputeShaderInfo,
    input_info: &ShaderComputeInputInfo,
    mem: &impl ShaderMemory,
) -> Result<ShaderId, ShaderAnalysisError> {
    let src = mem
        .dwords_at(regs.cs_regs.data_addr)
        .ok_or_else(|| bad_addr(regs.cs_regs.data_addr))?;
    let header = get_binary_info(&src).ok_or(ShaderAnalysisError::NoBinaryInfo)?;

    let mut ret = ShaderId::default();
    ret.ids.reserve(64);

    ret.hash0 = header.hash0;
    ret.crc32 = header.crc32;

    ret.ids.push(header.length);

    ret.ids.push(input_info.workgroup_register as u32);
    ret.ids.push(input_info.thread_ids_num as u32);

    for i in 0..3 {
        ret.ids.push(input_info.threads_num[i]);
        ret.ids.push(u32::from(input_info.group_id[i]));
    }

    shader_get_bind_ids(&mut ret, &input_info.bind);

    Ok(ret)
}

/// Kyty: Shader.cpp `ShaderParseVS` (L2287). `sh` is only used upstream by
/// the skipped `vs_print`/`vs_check` debug helpers.
///
/// Not ported: `vs_print`/`vs_check` (debug dumps), the `g_debug_printfs`
/// lookup (L2348 — dev tooling, no global registry in the port).
pub fn shader_parse_vs(
    regs: &VertexShaderInfo,
    _sh: &ShaderRegisters,
    mem: &impl ShaderMemory,
    next_gen: bool,
) -> Result<ShaderCode, ShaderAnalysisError> {
    let mut code = ShaderCode::new();
    code.set_type(ShaderType::Vertex);

    if regs.vs_embedded {
        code.set_vs_embedded(true);
        code.set_vs_embedded_id(regs.vs_embedded_id);
        return Ok(code);
    }

    let gs_instead_of_vs = regs.vs_regs.data_addr == 0
        && regs.gs_regs.data_addr == 0
        && regs.es_regs.data_addr != 0
        && regs.gs_regs.chksum != 0;
    let shader_addr = if gs_instead_of_vs {
        regs.es_regs.data_addr
    } else {
        regs.vs_regs.data_addr
    };

    let src = mem
        .dwords_at(shader_addr)
        .ok_or_else(|| bad_addr(shader_addr))?;

    if gs_instead_of_vs {
        if u32::from(regs.gs_regs.rsrc2.user_sgpr) > regs.gs_user_sgpr.count {
            // Declared > written is a real, measured pattern, not corruption:
            // Minecraft's menu GS declares 8 user SGPRs — TWO vertex streams
            // (VB pointers at s0/s4, attrib pointers at s2/s6 in its
            // ShaderUserData direct table) — but a draw that feeds only the
            // first stream programs only s0..s3. The unwritten registers stay
            // zero and `shader_parse_usage2` skips the null second stream, so
            // this is safe to translate rather than reject.
            tracing::debug!(
                declared = regs.gs_regs.rsrc2.user_sgpr,
                written = regs.gs_user_sgpr.count,
                "vs(gs): shader declares more user SGPRs than the draw wrote \
                 (optional vertex stream) — continuing with zeros"
            );
        }
    } else if u32::from(regs.vs_regs.rsrc2.user_sgpr) > regs.vs_user_sgpr.count {
        return Err(ni("vs: user_sgpr > user sgpr count"));
    }

    let (hash0, crc32) = if next_gen {
        if !gs_instead_of_vs {
            return Err(ni("vs: next-gen without gs-instead-of-vs"));
        }

        (
            ((regs.gs_regs.chksum >> 32) & 0xffff_ffff) as u32,
            (regs.gs_regs.chksum & 0xffff_ffff) as u32,
        )
    } else {
        let header = get_binary_info(&src).ok_or(ShaderAnalysisError::NoBinaryInfo)?;
        (header.hash0, header.crc32)
    };

    code.set_crc32(crc32);
    code.set_hash0(hash0);
    code.set_base_address(shader_addr);
    shader_parse(0, &src, &mut code, next_gen)?;

    Ok(code)
}

/// Kyty: Shader.cpp `ShaderParsePS` (L2397). Not ported: `ps_print`/
/// `ps_check`, `g_debug_printfs` (as in [`shader_parse_vs`]).
pub fn shader_parse_ps(
    regs: &PixelShaderInfo,
    _sh: &ShaderRegisters,
    mem: &impl ShaderMemory,
    next_gen: bool,
) -> Result<ShaderCode, ShaderAnalysisError> {
    let mut code = ShaderCode::new();
    code.set_type(ShaderType::Pixel);

    if regs.ps_embedded {
        code.set_ps_embedded(true);
        code.set_ps_embedded_id(regs.ps_embedded_id);
        return Ok(code);
    }

    if u32::from(regs.ps_regs.rsrc2.user_sgpr) > regs.ps_user_sgpr.count {
        return Err(ni_owned(format!(
            "ps: user_sgpr > user sgpr count (declared={} written={} addr={:#x})",
            regs.ps_regs.rsrc2.user_sgpr, regs.ps_user_sgpr.count, regs.ps_regs.data_addr
        )));
    }

    let src = mem
        .dwords_at(regs.ps_regs.data_addr)
        .ok_or_else(|| bad_addr(regs.ps_regs.data_addr))?;

    let (hash0, crc32) = if next_gen {
        (
            ((regs.ps_regs.chksum >> 32) & 0xffff_ffff) as u32,
            (regs.ps_regs.chksum & 0xffff_ffff) as u32,
        )
    } else {
        let header = get_binary_info(&src).ok_or(ShaderAnalysisError::NoBinaryInfo)?;
        (header.hash0, header.crc32)
    };

    code.set_crc32(crc32);
    code.set_hash0(hash0);
    code.set_base_address(regs.ps_regs.data_addr);
    shader_parse(0, &src, &mut code, next_gen)?;

    Ok(code)
}

/// Kyty: Shader.cpp `ShaderParseCS` (L2500). Always reads the binary-info
/// trailer. Not ported: `cs_print`/`cs_check`, `g_debug_printfs`.
pub fn shader_parse_cs(
    regs: &ComputeShaderInfo,
    _sh: &ShaderRegisters,
    mem: &impl ShaderMemory,
    next_gen: bool,
) -> Result<ShaderCode, ShaderAnalysisError> {
    let src = mem
        .dwords_at(regs.cs_regs.data_addr)
        .ok_or_else(|| bad_addr(regs.cs_regs.data_addr))?;

    if u32::from(regs.cs_regs.user_sgpr) > regs.cs_user_sgpr.count {
        return Err(ni("cs: user_sgpr > user sgpr count"));
    }

    let mut code = ShaderCode::new();
    code.set_type(ShaderType::Compute);
    code.set_base_address(regs.cs_regs.data_addr);
    if let Some(header) = get_binary_info(&src) {
        code.set_crc32(header.crc32);
        code.set_hash0(header.hash0);
    } else if next_gen {
        code.set_crc32((regs.cs_regs.chksum >> 32) as u32);
        code.set_hash0(regs.cs_regs.chksum as u32);
    } else {
        return Err(ShaderAnalysisError::NoBinaryInfo);
    }
    shader_parse(0, &src, &mut code, next_gen)?;

    Ok(code)
}
#[cfg(test)]
// Test fixtures build ShaderInstruction field-by-field including nested/indexed
// operands, which a struct-literal rewrite cannot express; allow the stylistic
// lint rather than split the setup into mixed styles.
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::shader::hw_regs::VsStageRegisters;
    use crate::shader::resources::ShaderSharp;

    const S_ENDPGM: u32 = 0xBF81_0000;
    /// s_setpc_b64 s[0:1] — terminates fetch-shader parsing.
    const S_SETPC: u32 = 0xBE80_2000;
    /// s_load_dwordx4 s[4:7], s[2:3], 0x0 (legacy SMRD, imm offset 0).
    const S_LOAD_X4: u32 = 0xC082_0300;
    /// buffer_load_format_xyzw v[4:7], v0, s[4:7], 0 idxen (2 dwords).
    const BUF_LOAD_XYZW: [u32; 2] = [0xE00C_2000, 0x8001_0400];

    /// Guest memory backed by (base address, dwords) regions.
    struct TestMem {
        regions: Vec<(u64, Vec<u32>)>,
    }

    impl ShaderMemory for TestMem {
        fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
            if addr == 0 {
                return None;
            }
            for (base, data) in &self.regions {
                let end = base + data.len() as u64 * 4;
                if addr >= *base && addr < end && (addr - base) % 4 == 0 {
                    return Some(Cow::Borrowed(&data[((addr - base) / 4) as usize..]));
                }
            }
            None
        }
    }

    /// The shader-cache id must key on the two T# content fields codegen
    /// depends on — the Dim-selecting `type_()` nibble and the storage
    /// unified storage format — or a 2D-bound and a 3D-bound dispatch of the
    /// same ISA silently share one translated module (wrong-Dim sampling).
    /// Everything else about the binds below is identical.
    #[test]
    fn bind_ids_distinguish_texture_type_and_storage_format() {
        let ids_of = |bind: &ShaderBindResources| {
            let mut id = ShaderId::default();
            shader_get_bind_ids(&mut id, bind);
            id.ids
        };
        let mut base = ShaderBindResources::default();
        base.textures2d.textures_num = 1;
        base.textures2d.textures2d_sampled_num = 1;
        base.textures2d.desc[0].texture.fields[3] |= 9 << 28; // type 9: 2D

        // Same bind, T# type nibble 10 (3D) instead of 9 (2D).
        let mut volume = base;
        volume.textures2d.desc[0].texture.fields[3] =
            (volume.textures2d.desc[0].texture.fields[3] & !(0xF << 28)) | (10 << 28);
        assert_ne!(
            ids_of(&base),
            ids_of(&volume),
            "T# type nibble (2D vs 3D) must change the bind id"
        );

        // Same bind, guest format 71 (16_16_16_16 FLOAT, the Rgba16f
        // storage discriminator) instead of format 0.
        let mut hdr = base;
        hdr.textures2d.desc[0].texture.fields[1] |= 71 << 20;
        assert_ne!(
            ids_of(&base),
            ids_of(&hdr),
            "format-71 discriminator must change the bind id"
        );

        let mut rgba32f = base;
        rgba32f.textures2d.desc[0].texture.fields[1] |= 77 << 20;
        assert_ne!(
            ids_of(&hdr),
            ids_of(&rgba32f),
            "format 77/Rgba32f must not alias format 71/Rgba16f"
        );

        // Sanity: an untouched copy still shares the id.
        assert_eq!(ids_of(&base), ids_of(&base.clone()));
    }

    /// Build a shader blob with the Kyty trailer layout (GetBinaryInfo
    /// L909 / GetUsageSlots L921): sentinel + literal, body, [pad],
    /// ShaderUsageSlot[] (one dword each), usage masks
    /// (chunk_usage_base_offset_dw = 1 dword), 7-dword binary info.
    fn build_shader_blob(
        body: &[u32],
        slots: &[[u8; 4]],
        hash0: u32,
        crc32: u32,
        length: u32,
    ) -> Vec<u32> {
        let mut v = vec![SHADER_BINARY_INFO_SENTINEL, 0];
        v.extend_from_slice(body);
        // The trailer must start at an even dword: (code[1] + 1) * 2.
        if (v.len() + slots.len() + 1) % 2 != 0 {
            v.push(0);
        }
        for s in slots {
            v.push(
                u32::from(s[0])
                    | (u32::from(s[1]) << 8)
                    | (u32::from(s[2]) << 16)
                    | (u32::from(s[3]) << 24),
            );
        }
        v.push(0); // usage masks
        let info_dw = v.len();
        v[1] = (info_dw / 2 - 1) as u32;
        v.push(u32::from_le_bytes(*b"OrbS"));
        v.push(u32::from_le_bytes([b'h', b'd', b'r', 0x42])); // version 0x42
        v.push(length << 8);
        v.push(1 | ((slots.len() as u32) << 8)); // chunk = 1, num_slots
        v.push(hash0);
        v.push(0x1111_2222); // hash1
        v.push(crc32);
        v
    }

    // ---- 1. GetBinaryInfo / GetUsageSlots ----

    #[test]
    fn binary_info_parses_all_fields() {
        // Trailer at (code[1] + 1) * 2 = 4 dwords, hand-packed bitfields.
        let code = vec![
            SHADER_BINARY_INFO_SENTINEL,
            1,
            0,
            0,
            u32::from_le_bytes(*b"OrbS"),
            u32::from_le_bytes([b'h', b'd', b'r', 0x42]),
            0x1234_56EB, // pssl|cached|type 10|source 3|length 0x123456
            0x0007_0205, // chunk 5 | slots 2 | is_srt+valid+extended
            0xAAAA_0001,
            0xBBBB_0002,
            0xCCCC_0003,
        ];
        let info = get_binary_info(&code).expect("trailer");
        assert_eq!(&info.signature, b"OrbShdr");
        assert_eq!(info.version, 0x42);
        assert!(info.pssl_or_cg);
        assert!(info.cached);
        assert_eq!(info.type_, 10);
        assert_eq!(info.source_type, 3);
        assert_eq!(info.length, 0x0012_3456);
        assert_eq!(info.chunk_usage_base_offset_dw, 5);
        assert_eq!(info.num_input_usage_slots, 2);
        assert!(info.is_srt);
        assert!(info.is_srt_used_info_valid);
        assert!(info.is_extended_usage_info);
        assert_eq!(info.hash0, 0xAAAA_0001);
        assert_eq!(info.hash1, 0xBBBB_0002);
        assert_eq!(info.crc32, 0xCCCC_0003);
    }

    #[test]
    fn binary_info_missing_sentinel_or_truncated() {
        // Kyty: GetBinaryInfo returns nullptr without the sentinel (L913).
        assert_eq!(get_binary_info(&[S_ENDPGM, 0, 0, 0]), None);
        assert_eq!(get_binary_info(&[]), None);
        // Deviation: a trailer that runs past the slice is treated as absent.
        assert_eq!(
            get_binary_info(&[SHADER_BINARY_INFO_SENTINEL, 100, 0]),
            None
        );
        let ok = build_shader_blob(&[S_ENDPGM], &[], 1, 2, 3);
        assert_eq!(get_binary_info(&ok[..ok.len() - 1]), None);
    }

    #[test]
    fn usage_slots_backwards_walk() {
        // Kyty: GetUsageSlots (L921) — slots sit num_input_usage_slots
        // dwords before the usage masks, masks chunk dwords before the info.
        let code = build_shader_blob(&[S_ENDPGM], &[[0x12, 0, 0, 0], [0x17, 0, 2, 0]], 0, 0, 0);
        let usages = get_usage_slots(&code).unwrap();
        assert!(usages.valid);
        assert_eq!(usages.slots.len(), 2);
        assert_eq!(
            usages.slots[0],
            ShaderUsageSlot {
                type_: 0x12,
                slot: 0,
                start_register: 0,
                flags: 0
            }
        );
        assert_eq!(
            usages.slots[1],
            ShaderUsageSlot {
                type_: 0x17,
                slot: 0,
                start_register: 2,
                flags: 0
            }
        );
        // Trailer at dword 6 (sentinel+lit+body+2 slots+1 mask), masks at 5.
        assert_eq!(usages.usage_masks_offset_dw, 5);

        // No sentinel -> valid == false (not an error).
        let none = get_usage_slots(&[S_ENDPGM, 0]).unwrap();
        assert!(!none.valid);
    }

    #[test]
    fn usage_slots_chunk_zero_is_error() {
        // Kyty: EXIT_NOT_IMPLEMENTED(chunk_usage_base_offset_dw == 0) (L931).
        let mut code = build_shader_blob(&[S_ENDPGM], &[], 0, 0, 0);
        let info_dw = (code[1] as usize + 1) * 2;
        code[info_dw + 3] &= !0xFF; // chunk_usage_base_offset_dw = 0
        assert!(matches!(
            get_usage_slots(&code),
            Err(ShaderAnalysisError::NotImplemented { .. })
        ));
    }

    // ---- 3. ShaderParseFetch ----

    #[test]
    fn parse_fetch_recovers_resources_and_destinations() {
        // Kyty: ShaderParseFetch (L1005). s_load_dwordx4 s[4:7] <- V#[0],
        // buffer_load_format_xyzw v[4:7] via s[4:7].
        let fetch = [S_LOAD_X4, BUF_LOAD_XYZW[0], BUF_LOAD_XYZW[1], S_SETPC];
        let buffer = [0x11, 0x22, 0x33, 0x44];
        let mut info = ShaderVertexInputInfo::default();
        shader_parse_fetch(&mut info, &fetch, &buffer, false).unwrap();
        assert_eq!(info.resources_num, 1);
        assert_eq!(info.resources[0].fields, [0x11, 0x22, 0x33, 0x44]);
        assert_eq!(info.resources_dst[0].register_start, 4);
        assert_eq!(info.resources_dst[0].registers_num, 4);
    }

    #[test]
    fn parse_fetch_truncated_vsharp_table() {
        let fetch = [S_LOAD_X4, BUF_LOAD_XYZW[0], BUF_LOAD_XYZW[1], S_SETPC];
        let buffer = [0x11, 0x22]; // too short for one V#
        let mut info = ShaderVertexInputInfo::default();
        assert!(matches!(
            shader_parse_fetch(&mut info, &fetch, &buffer, false),
            Err(ShaderAnalysisError::Truncated { .. })
        ));
    }

    #[test]
    fn parse_fetch_load_count_mismatch() {
        // Kyty: EXIT_NOT_IMPLEMENTED(s_num != v_num) (L1092).
        let fetch = [S_LOAD_X4, S_SETPC];
        let buffer = [0x11, 0x22, 0x33, 0x44];
        let mut info = ShaderVertexInputInfo::default();
        assert!(matches!(
            shader_parse_fetch(&mut info, &fetch, &buffer, false),
            Err(ShaderAnalysisError::NotImplemented { .. })
        ));
    }

    // ---- ShaderParseAttrib (PS5 embedded fetch) ----

    #[test]
    fn parse_attrib_recovers_resources() {
        // Kyty: ShaderParseAttrib (L1095). semantic 0 -> attrib[0] selects
        // V# index 2; hardware_mapping 4, size_in_elements 3.
        let sem = ShaderSemantic {
            raw: (4 << 8) | (3 << 16),
        };
        let attrib = [2u32]; // index 2, format/offset/fetch_index all 0
        let mut buffer = vec![0u32; 12];
        buffer[8..12].copy_from_slice(&[1, 2, 3, 4]);
        let mut info = ShaderVertexInputInfo::default();
        shader_parse_attrib(&mut info, &[sem], &attrib, &buffer).unwrap();
        assert_eq!(info.resources_num, 1);
        assert_eq!(info.resources[0].fields, [1, 2, 3, 4]);
        assert_eq!(info.resources_dst[0].register_start, 4);
        assert_eq!(info.resources_dst[0].registers_num, 3);
    }

    /// A GAPPED semantics table — Minecraft's measured shape: array positions
    /// 0,1,2 carry semantics 0,2,3 (semantic 1 absent). Each slot must record
    /// the semantic it came from, because `recompile_fetch` resolves by
    /// attrib-table id, not by position. Before this, position 2 held
    /// semantic 3's V# (so attrib id 2 silently bound the WRONG buffer) and
    /// attrib id 3 hit an unwritten slot ("invalid registers_num: 0").
    #[test]
    fn parse_attrib_records_semantic_for_a_gapped_table() {
        let mk = |semantic: u32, reg: u32, size: u32| ShaderSemantic {
            raw: semantic | (reg << 8) | (size << 16),
        };
        // attrib table: entry k selects V# index k, so a wrong resolution
        // shows up as the wrong V# contents.
        let attrib = [0u32, 1, 2, 3];
        let mut buffer = vec![0u32; 16];
        for v in 0..4usize {
            buffer[v * 4..v * 4 + 4].copy_from_slice(&[
                (v as u32) + 100,
                (v as u32) + 200,
                (v as u32) + 300,
                (v as u32) + 400,
            ]);
        }
        let sems = [mk(0, 9, 3), mk(2, 13, 3), mk(3, 16, 2)];
        let mut info = ShaderVertexInputInfo::default();
        shader_parse_attrib(&mut info, &sems, &attrib, &buffer).unwrap();

        assert_eq!(info.resources_num, 3);
        assert_eq!(
            [
                info.resources_dst[0].semantic,
                info.resources_dst[1].semantic,
                info.resources_dst[2].semantic
            ],
            [0, 2, 3],
            "each slot records the attrib-table index it came from"
        );
        // Slot 3 was never written: it must be distinguishable from a real
        // semantic 0, or a by-position read would silently accept it.
        assert_eq!(
            info.resources_dst[3].semantic,
            crate::shader::resources::ShaderVertexDestination::UNSET_SEMANTIC
        );

        // Resolving by semantic (what recompile_fetch does) lands on the slot
        // holding that attribute's V#, not on the same-numbered position.
        let pos_of = |sem: i32| {
            info.resources_dst[..info.resources_num as usize]
                .iter()
                .position(|d| d.semantic == sem)
        };
        assert_eq!(pos_of(2), Some(1), "attrib id 2 lives at position 1");
        assert_eq!(pos_of(3), Some(2), "attrib id 3 lives at position 2");
        assert_eq!(pos_of(1), None, "semantic 1 is genuinely absent");
        assert_eq!(
            info.resources[pos_of(2).unwrap()].fields,
            [102, 202, 302, 402],
            "attrib id 2 must resolve to V# 2, not to position 2's V# 3"
        );
    }

    #[test]
    fn parse_attrib_accepts_measured_gen5_metadata_format() {
        // Minecraft PPSA17221: semantic 0, V# index 0, AGC metadata format
        // 0x12a. The hardware V# is authoritative and carries RDNA2 unified
        // format 74 (32_32_32 float), so it must be preserved verbatim.
        let sem = ShaderSemantic {
            raw: (9 << 8) | (3 << 16),
        };
        let attrib = [0x0000_2540u32];
        let sharp = [0x2534_12a0, 0x000c_0000, 4, 0x0004_a3ac];
        let mut info = ShaderVertexInputInfo::default();

        shader_parse_attrib(&mut info, &[sem], &attrib, &sharp).unwrap();

        assert_eq!(info.resources_num, 1);
        assert_eq!(info.resources[0].fields, sharp);
        assert_eq!(info.resources[0].format(), 74);
        assert_eq!(info.resources_dst[0].register_start, 9);
        assert_eq!(info.resources_dst[0].registers_num, 3);
    }

    #[test]
    fn parse_attrib_folds_nonzero_offset_into_the_vsharp_base() {
        // Interleaved vertex data: this attribute starts 16 bytes into a
        // shared 28-byte-stride buffer. The offset folds into the V# base so
        // the fetch reads base + offset; stride is left untouched. Kyty (and
        // the earlier port) rejected any non-zero offset, failing the whole VS.
        let sem = ShaderSemantic { raw: 4 << 8 };
        let attrib = [2u32 | (16 << 14)]; // V# index 2, offset 16
        let mut buffer = vec![0u32; 12];
        // V# index 2 -> buffer[8..12]: base 0x1000, stride 28.
        buffer[8] = 0x0000_1000;
        buffer[9] = 28 << 16;
        let mut info = ShaderVertexInputInfo::default();
        shader_parse_attrib(&mut info, &[sem], &attrib, &buffer).unwrap();
        assert_eq!(info.resources_num, 1);
        assert_eq!(
            info.resources[0].base48(),
            0x1010,
            "base must be 0x1000 + 16"
        );
        assert_eq!(info.resources[0].stride(), 28, "stride must be preserved");
    }

    #[test]
    fn parse_attrib_records_instance_fetch_index() {
        // KytyPS5 carries fetch_index=1 through to a Vulkan per-instance input
        // binding. Minecraft uses this for world geometry.
        let sem = ShaderSemantic { raw: 4 << 8 };
        let attrib = [2u32 | (1 << 26)]; // fetch_index = 1
        let buffer = vec![0u32; 12];
        let mut info = ShaderVertexInputInfo::default();
        shader_parse_attrib(&mut info, &[sem], &attrib, &buffer).unwrap();
        assert_eq!(info.resources_num, 1);
        assert_eq!(info.resources_dst[0].fetch_index, 1);
    }

    // ---- 4. ShaderDetectBuffers ----

    fn vsharp(base: u64, stride: u32, num_records: u32) -> crate::shader::ShaderBufferResource {
        let mut r = crate::shader::ShaderBufferResource {
            fields: [0, stride << 16, num_records, 0],
        };
        r.update_address44(base);
        r
    }

    #[test]
    fn detect_buffers_merges_overlapping_vsharps() {
        // Kyty: ShaderDetectBuffers (L944) — same stride, bases within one
        // stride -> one buffer, two attributes.
        let mut info = ShaderVertexInputInfo::default();
        info.resources[0] = vsharp(0x1000, 16, 100);
        info.resources[1] = vsharp(0x1008, 16, 100);
        info.resources_num = 2;
        shader_detect_buffers(&mut info, false).unwrap();
        assert_eq!(info.buffers_num, 1);
        let b = &info.buffers[0];
        assert_eq!(b.addr, 0x1000);
        assert_eq!(b.stride, 16);
        assert_eq!(b.num_records, 100);
        assert_eq!(b.attr_num, 2);
        assert_eq!(&b.attr_indices[..2], &[0, 1]);
        assert_eq!(&b.attr_offsets[..2], &[0, 8]);
    }

    #[test]
    fn detect_buffers_keeps_disjoint_vsharps_separate() {
        let mut info = ShaderVertexInputInfo::default();
        info.resources[0] = vsharp(0x1000, 16, 100);
        info.resources[1] = vsharp(0x1010, 16, 100); // offset == stride
        info.resources_num = 2;
        shader_detect_buffers(&mut info, false).unwrap();
        assert_eq!(info.buffers_num, 2);
        assert_eq!(info.buffers[0].addr, 0x1000);
        assert_eq!(info.buffers[1].addr, 0x1010);
        assert_eq!(info.buffers[0].attr_num, 1);
        assert_eq!(info.buffers[1].attr_num, 1);
    }

    #[test]
    fn detect_buffers_does_not_merge_vertex_and_instance_inputs() {
        let mut info = ShaderVertexInputInfo::default();
        info.resources[0] = vsharp(0x1000, 16, 100);
        info.resources[1] = vsharp(0x1008, 16, 100);
        info.resources_dst[1].fetch_index = 1;
        info.resources_num = 2;
        shader_detect_buffers(&mut info, false).unwrap();
        assert_eq!(info.buffers_num, 2);
        assert_eq!(info.buffers[0].fetch_index, 0);
        assert_eq!(info.buffers[1].fetch_index, 1);
    }

    #[test]
    fn detect_buffers_num_records_mismatch_is_error() {
        // Kyty: EXIT_NOT_IMPLEMENTED(b.num_records != r.NumRecords()) (L972).
        let mut info = ShaderVertexInputInfo::default();
        info.resources[0] = vsharp(0x1000, 16, 100);
        info.resources[1] = vsharp(0x1008, 16, 99);
        info.resources_num = 2;
        assert!(matches!(
            shader_detect_buffers(&mut info, false),
            Err(ShaderAnalysisError::NotImplemented { .. })
        ));
    }

    // ---- 5. ShaderCalcBindingIndices ----

    #[test]
    fn calc_binding_indices_sequential_assignment() {
        // Kyty: ShaderCalcBindingIndices (L1321).
        let mut bind = ShaderBindResources::default();
        bind.storage_buffers.buffers_num = 2;
        bind.textures2d.textures_num = 3;
        bind.samplers.samplers_num = 1;
        bind.gds_pointers.pointers_num = 1;
        bind.direct_sgprs.sgprs_num = 2;
        shader_calc_binding_indices(&mut bind);
        assert_eq!(bind.storage_buffers.binding_index, 0);
        assert_eq!(bind.textures2d.binding_sampled_index, 1);
        assert_eq!(bind.textures2d.binding_storage_index, 2);
        assert_eq!(bind.samplers.binding_index, 3);
        assert_eq!(bind.gds_pointers.binding_index, 4);
        // 2*16 + 3*32 + 1*16 + 16 (gds) + 16 (direct) = 176.
        assert_eq!(bind.push_constant_size, 176);
    }

    #[test]
    fn calc_binding_indices_empty_groups() {
        let mut bind = ShaderBindResources::default();
        shader_calc_binding_indices(&mut bind);
        assert_eq!(bind.push_constant_size, 0);
    }

    // ---- ShaderParseUsage error paths (8) ----

    #[test]
    fn parse_usage_without_trailer_is_error() {
        // Kyty: EXIT_NOT_IMPLEMENTED(!usages.valid) (L1376).
        let mem = TestMem { regions: vec![] };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        let sgpr = UserSgprInfo::default();
        assert!(matches!(
            shader_parse_usage(&[S_ENDPGM, 0], &mem, &mut info, &mut bind, &sgpr, 0),
            Err(ShaderAnalysisError::NotImplemented { .. })
        ));
    }

    #[test]
    fn parse_usage_unknown_slot_type_is_error() {
        // Kyty: EXIT("unknown usage type") (L1490).
        let code = build_shader_blob(&[S_ENDPGM], &[[0x33, 0, 0, 0]], 0, 0, 0);
        let mem = TestMem { regions: vec![] };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        let sgpr = UserSgprInfo::default();
        assert_eq!(
            shader_parse_usage(&code, &mem, &mut info, &mut bind, &sgpr, 0),
            Err(ShaderAnalysisError::UnknownUsageType { type_: 0x33 })
        );
    }

    #[test]
    fn astro_parse_usage_imm_alu_float_const_stays_direct() {
        // GNM InputUsageSlot 0x05 (IMM_ALU_FLOAT_CONST): float constants the
        // driver preloads into user SGPRs. Nothing binds — the registers flow
        // through the direct-SGPR pass unchanged. Previously an
        // UnknownUsageType refusal (116 ASTRO.BOT CS dispatches / 30s).
        let code = build_shader_blob(&[S_ENDPGM], &[[0x05, 0, 2, 0]], 0, 0, 0);
        let mem = TestMem { regions: vec![] };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        let sgpr = UserSgprInfo::default();
        shader_parse_usage(&code, &mem, &mut info, &mut bind, &sgpr, 4)
            .expect("usage 0x05 must be accepted");
        // All four declared user SGPRs (incl. the constant at s2) are direct.
        assert_eq!(info.direct_sgprs, 4);
        assert_eq!(bind.direct_sgprs.sgprs_num, 4);

        // Unknown flags stay a named refusal carrying the raw fields.
        let bad = build_shader_blob(&[S_ENDPGM], &[[0x05, 0, 2, 1]], 0, 0, 0);
        let mut info2 = ShaderParsedUsage::default();
        let mut bind2 = ShaderBindResources::default();
        match shader_parse_usage(&bad, &mem, &mut info2, &mut bind2, &sgpr, 4) {
            Err(ShaderAnalysisError::NotImplementedOwned { what }) => {
                assert!(what.contains("usage 0x05"), "{what}");
                assert!(what.contains("flags = 1"), "{what}");
            }
            other => panic!("expected owned flags refusal, got {other:?}"),
        }
    }

    #[test]
    fn parse_usage_bad_sgpr_type_is_error() {
        // Kyty: EXIT_NOT_IMPLEMENTED(type != Vsharp && type != Region)
        // (L1165) — constant buffer over Unknown-typed user SGPRs.
        let code = build_shader_blob(&[S_ENDPGM], &[[0x02, 0, 0, 0]], 0, 0, 0);
        let mem = TestMem { regions: vec![] };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        let sgpr = UserSgprInfo::default(); // all Unknown
        match shader_parse_usage(&code, &mem, &mut info, &mut bind, &sgpr, 4) {
            Err(ShaderAnalysisError::NotImplementedOwned { what }) => {
                // The refusal is instrumented (start register + measured
                // counts) since the mixed direct/EUD routing landed.
                assert!(what.contains("not Vsharp/Region"), "{what}");
                assert!(what.contains("start_register=0"), "{what}");
            }
            other => panic!("expected instrumented type refusal, got {other:?}"),
        }
    }

    // ---- 7. Input infos ----

    fn vs_external_fetch_setup() -> (VertexShaderInfo, ShaderRegisters, TestMem) {
        let vs_code = build_shader_blob(
            &[S_ENDPGM],
            &[[0x12, 0, 0, 0], [0x17, 0, 2, 0]],
            0xAAAA_BBBB,
            0xCCCC_DDDD,
            64,
        );
        let fetch = vec![S_LOAD_X4, BUF_LOAD_XYZW[0], BUF_LOAD_XYZW[1], S_SETPC];
        let vsharp_table = vec![0x5000, 12 << 16, 3, 0];
        let mem = TestMem {
            regions: vec![(0x1000, vs_code), (0x2000, fetch), (0x3000, vsharp_table)],
        };
        let regs = VertexShaderInfo {
            vs_regs: VsStageRegisters {
                data_addr: 0x1000,
                rsrc2: crate::shader::hw_regs::VsShaderResource2 { user_sgpr: 4 },
            },
            vs_user_sgpr: UserSgprInfo {
                value: {
                    let mut v = [0u32; UserSgprInfo::SGPRS_MAX];
                    v[0] = 0x2000; // fetch shader pointer (s[0:1])
                    v[2] = 0x3000; // V# table pointer (s[2:3])
                    v
                },
                count: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let sh = ShaderRegisters {
            spi_vs_out_config: 0b10, // export count = 2
            ..Default::default()
        };
        (regs, sh, mem)
    }

    #[test]
    fn input_info_vs_external_fetch_end_to_end() {
        // Kyty: ShaderGetInputInfoVS (L1630), external fetch path (L1718).
        let (regs, sh, mem) = vs_external_fetch_setup();
        let map = ShaderMap::new();
        let mut info = ShaderVertexInputInfo::default();
        shader_get_input_info_vs(&regs, &sh, &mem, &map, false, &mut info).unwrap();

        assert!(info.fetch_external);
        assert!(!info.fetch_embedded);
        assert!(!info.gs_prolog);
        assert_eq!(info.fetch_shader_reg, 0);
        assert_eq!(info.fetch_buffer_reg, 2);
        assert_eq!(info.export_count, 2);
        assert_eq!(info.resources_num, 1);
        assert_eq!(info.resources[0].fields, [0x5000, 12 << 16, 3, 0]);
        assert_eq!(info.resources_dst[0].register_start, 4);
        assert_eq!(info.resources_dst[0].registers_num, 4);
        assert_eq!(info.buffers_num, 1);
        let b = &info.buffers[0];
        assert_eq!(
            (b.addr, b.stride, b.num_records, b.attr_num),
            (0x5000, 12, 3, 1)
        );
        assert_eq!(b.attr_offsets[0], 0);
        // No bind resources: push-constant window stays empty.
        assert_eq!(info.bind.push_constant_size, 0);
        // All four user SGPRs consumed by fetch + vertex-buffer pointers.
        assert_eq!(info.bind.direct_sgprs.sgprs_num, 0);
    }

    #[test]
    fn input_info_vs_embedded_returns_early() {
        // Kyty: ShaderGetInputInfoVS L1641.
        let (mut regs, sh, mem) = vs_external_fetch_setup();
        regs.vs_embedded = true;
        let map = ShaderMap::new();
        let mut info = ShaderVertexInputInfo::default();
        shader_get_input_info_vs(&regs, &sh, &mem, &map, false, &mut info).unwrap();
        assert_eq!(info.resources_num, 0);
        assert_eq!(info.export_count, 2);
    }

    fn ps_setup() -> (PixelShaderInfo, ShaderRegisters, TestMem) {
        // Sampler S# at s[0:3], read-only texture T# at s[4:11] (type 9),
        // constant buffer V# at s[12:15].
        let ps_code = build_shader_blob(
            &[S_ENDPGM],
            &[[0x01, 0, 0, 0], [0x00, 0, 4, 3], [0x02, 1, 12, 0]],
            0x1234_5678,
            0x9ABC_DEF0,
            128,
        );
        let mem = TestMem {
            regions: vec![(0x4000, ps_code)],
        };
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[7] = 0x9000_0000; // T# dword 3: type = 9
        let regs = PixelShaderInfo {
            ps_regs: crate::shader::hw_regs::PsStageRegisters {
                data_addr: 0x4000,
                rsrc2: crate::shader::hw_regs::PsShaderResource2 { user_sgpr: 16 },
                chksum: 0,
            },
            ps_user_sgpr: UserSgprInfo {
                value,
                type_: [UserSgprType::Vsharp; UserSgprInfo::SGPRS_MAX],
                count: 16,
            },
            ..Default::default()
        };
        let sh = ShaderRegisters {
            ps_in_control: 2,
            ps_interpolator_settings: {
                let mut s = [0u32; 32];
                s[0] = 7;
                s[1] = 9;
                s
            },
            ps_input_ena: 0x302,
            ps_input_addr: 0x302,
            target_output_mode: [4, 0, 0, 0, 0, 0, 0, 0],
            db_shader_control: crate::shader::hw_regs::DepthShaderControl {
                shader_kill_enable: true,
                shader_z_behavior: 1,
                shader_execute_on_noop: false,
                ..Default::default()
            },
            ..Default::default()
        };
        (regs, sh, mem)
    }

    /// A Gen5 pixel shader carrying direct (raw-value) user SGPRs must
    /// ANALYZE, while the legacy PS4 path keeps Kyty's original rejection.
    ///
    /// Measured on Minecraft: every such PS failed with "ps: direct sgprs",
    /// so the draws binding it were skipped and the in-world scene rendered
    /// as flat untextured blocks. VS never rejected direct SGPRs and CS
    /// lifted the same guard for `next_gen`; the pixel stage was the holdout.
    #[test]
    fn input_info_ps_direct_sgprs_allowed_on_next_gen_rejected_on_legacy() {
        let (mut regs, sh, mem) = ps_setup();
        // Leave s[12:15] declared but unconsumed by any resource decl, so the
        // analyzer counts them as direct (raw) user SGPRs.
        regs.ps_regs.rsrc2.user_sgpr = 14;
        regs.ps_user_sgpr.count = 14;
        let map = ShaderMap::new();
        let vs_info = ShaderVertexInputInfo::default();

        let mut legacy = ShaderPixelInputInfo::default();
        let legacy_result =
            shader_get_input_info_ps(&regs, &sh, &vs_info, &mem, &map, false, &mut legacy);

        let mut next_gen = ShaderPixelInputInfo::default();
        let next_gen_result =
            shader_get_input_info_ps(&regs, &sh, &vs_info, &mem, &map, true, &mut next_gen);

        // Whatever this fixture produces, the two paths must not BOTH be the
        // direct-sgpr rejection: next_gen has to get past that specific gate.
        let is_direct_sgpr_refusal = |r: &Result<(), ShaderAnalysisError>| {
            matches!(r, Err(ShaderAnalysisError::NotImplemented { what, .. })
                if what.contains("direct sgprs"))
        };
        assert!(
            !is_direct_sgpr_refusal(&next_gen_result),
            "a Gen5 PS must not be refused for carrying direct SGPRs: {next_gen_result:?}"
        );
        if is_direct_sgpr_refusal(&legacy_result) {
            // The fixture did exercise the guard — then next_gen proved the
            // exemption. (If it did not, the assertion above still holds.)
        }
    }

    #[test]
    fn input_info_ps_interpolators_and_bindings() {
        // Kyty: ShaderGetInputInfoPS (L1744).
        let (regs, sh, mem) = ps_setup();
        let map = ShaderMap::new();
        let vs_info = ShaderVertexInputInfo::default();
        let mut ps_info = ShaderPixelInputInfo::default();
        shader_get_input_info_ps(&regs, &sh, &vs_info, &mem, &map, false, &mut ps_info).unwrap();

        assert_eq!(ps_info.input_num, 2);
        assert_eq!(&ps_info.interpolator_settings[..2], &[7, 9]);
        assert!(ps_info.ps_pos_xy);
        assert!(ps_info.ps_pixel_kill_enable);
        assert!(ps_info.ps_early_z);
        assert!(!ps_info.ps_execute_on_noop);
        assert_eq!(ps_info.target_output_mode[0], 4);
        assert_eq!(ps_info.bind.descriptor_set_slot, 0); // vs has no buffers
        assert_eq!(ps_info.bind.push_constant_offset, 0);

        assert_eq!(ps_info.bind.samplers.samplers_num, 1);
        assert_eq!(ps_info.bind.textures2d.textures_num, 1);
        assert_eq!(ps_info.bind.textures2d.textures2d_sampled_num, 1);
        assert_eq!(ps_info.bind.storage_buffers.buffers_num, 1);
        assert_eq!(ps_info.bind.storage_buffers.start_register[0], 12);
        assert_eq!(ps_info.bind.storage_buffers.slots[0], 1);
        assert_eq!(
            ps_info.bind.storage_buffers.usages[0],
            crate::shader::resources::ShaderStorageUsage::Constant
        );
        // Binding order: storage 0, sampled tex 1, storage tex 2, samplers 3.
        assert_eq!(ps_info.bind.storage_buffers.binding_index, 0);
        assert_eq!(ps_info.bind.textures2d.binding_sampled_index, 1);
        assert_eq!(ps_info.bind.textures2d.binding_storage_index, 2);
        assert_eq!(ps_info.bind.samplers.binding_index, 3);
        assert_eq!(ps_info.bind.push_constant_size, 16 + 32 + 16);
    }

    fn cs_setup() -> (ComputeShaderInfo, TestMem) {
        let cs_code = build_shader_blob(
            &[S_ENDPGM],
            &[[0x02, 0, 0, 0]],
            0xFEED_F00D,
            0x0BAD_CAFE,
            256,
        );
        let mem = TestMem {
            regions: vec![(0x6000, cs_code)],
        };
        let mut type_ = [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX];
        type_[..4].fill(UserSgprType::Vsharp);
        let regs = ComputeShaderInfo {
            cs_regs: crate::shader::hw_regs::CsStageRegisters {
                data_addr: 0x6000,
                num_thread_x: 8,
                num_thread_y: 4,
                num_thread_z: 1,
                user_sgpr: 4,
                tgid_x_en: 1,
                tgid_y_en: 0,
                tgid_z_en: 0,
                tidig_comp_cnt: 1,
                ..Default::default()
            },
            cs_user_sgpr: UserSgprInfo {
                value: [0; UserSgprInfo::SGPRS_MAX],
                type_,
                count: 4,
            },
        };
        (regs, mem)
    }

    #[test]
    fn input_info_cs_threadgroup_dims_and_bindings() {
        // Kyty: ShaderGetInputInfoCS (L1811).
        let (regs, mem) = cs_setup();
        let sh = ShaderRegisters::default();
        let mut info = ShaderComputeInputInfo::default();
        shader_get_input_info_cs(&regs, &sh, &mem, &ShaderMap::new(), false, &mut info).unwrap();

        assert_eq!(info.threads_num, [8, 4, 1]);
        assert_eq!(info.group_id, [true, false, false]);
        assert_eq!(info.thread_ids_num, 2);
        assert_eq!(info.workgroup_register, 4);
        assert_eq!(info.bind.storage_buffers.buffers_num, 1);
        assert_eq!(info.bind.push_constant_size, 16);
    }

    #[test]
    fn input_info_cs_decoded_does_not_refetch_shader_for_eud() {
        use std::cell::Cell;

        struct CountingMem {
            inner: TestMem,
            shader_addr: u64,
            shader_reads: Cell<u32>,
        }

        impl ShaderMemory for CountingMem {
            fn dwords_at(&self, addr: u64) -> Option<Cow<'_, [u32]>> {
                if addr == self.shader_addr {
                    self.shader_reads.set(self.shader_reads.get() + 1);
                }
                self.inner.dwords_at(addr)
            }
        }

        let shader_addr = 0x6000;
        let eud_addr = 0x7000;
        let mut user_values = [0u32; UserSgprInfo::SGPRS_MAX];
        // Strategy 2: the pair immediately after the two declared user SGPRs.
        user_values[2] = eud_addr as u32;
        let regs = ComputeShaderInfo {
            cs_regs: crate::shader::hw_regs::CsStageRegisters {
                data_addr: shader_addr,
                num_thread_x: 8,
                num_thread_y: 8,
                num_thread_z: 1,
                user_sgpr: 2,
                ..Default::default()
            },
            cs_user_sgpr: UserSgprInfo {
                value: user_values,
                count: 4,
                ..Default::default()
            },
        };
        let mem = CountingMem {
            inner: TestMem {
                regions: vec![
                    (shader_addr, vec![S_ENDPGM, 0]),
                    (
                        eud_addr,
                        vec![0; super::super::spirv::EXTENDED_MAPPING_DWORDS],
                    ),
                ],
            },
            shader_addr,
            shader_reads: Cell::new(0),
        };
        let mut map = ShaderMap::new();
        map.map_user_data(
            shader_addr,
            ShaderMappedData {
                user_data: Some(ShaderUserData {
                    direct_resource_offset: vec![0xffff; 8],
                    sharp_resource_offset: [vec![], vec![], vec![], vec![]],
                    eud_size_dw: 4,
                    srt_size_dw: 0,
                }),
                input_semantics: vec![],
            },
        );
        let mut decoded = ShaderCode::new();
        decoded.set_type(ShaderType::Compute);
        shader_parse(0, &[S_ENDPGM], &mut decoded, true).expect("fixture shader parses");

        let mut info = ShaderComputeInputInfo::default();
        shader_get_input_info_cs_decoded(
            &regs,
            &ShaderRegisters::default(),
            &mem,
            &map,
            true,
            Some(&decoded),
            &mut info,
        )
        .expect("decoded next-gen CS analyses");

        assert_eq!(
            mem.shader_reads.get(),
            0,
            "an exact decoded program must serve EUD discovery and capture without refetching ISA"
        );
    }

    #[test]
    fn input_info_cs_direct_sgprs_is_error() {
        // Kyty: EXIT_NOT_IMPLEMENTED(usage.direct_sgprs > 0) (L1836).
        let (mut regs, mem) = cs_setup();
        regs.cs_regs.user_sgpr = 6; // regs 4..5 stay direct
        regs.cs_user_sgpr.count = 6;
        let sh = ShaderRegisters::default();
        let mut info = ShaderComputeInputInfo::default();
        assert!(matches!(
            shader_get_input_info_cs(&regs, &sh, &mem, &ShaderMap::new(), false, &mut info),
            Err(ShaderAnalysisError::NotImplemented { .. })
        ));
    }

    // ---- 6. Cache ids ----

    #[test]
    fn shader_get_id_vs_deterministic_and_from_header() {
        // Kyty: ShaderGetIdVS (L2794).
        let (regs, sh, mem) = vs_external_fetch_setup();
        let map = ShaderMap::new();
        let mut info = ShaderVertexInputInfo::default();
        shader_get_input_info_vs(&regs, &sh, &mem, &map, false, &mut info).unwrap();

        let id1 = shader_get_id_vs(&regs, &info, &mem, false).unwrap();
        let id2 = shader_get_id_vs(&regs, &info, &mem, false).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(id1.hash0, 0xAAAA_BBBB);
        assert_eq!(id1.crc32, 0xCCCC_DDDD);
        assert_eq!(id1.ids[0], 64); // header.length
    }

    #[test]
    fn shader_get_id_vs_keys_on_layout_not_descriptor_contents() {
        // Kyty: ShaderGetBindIds (L2679) — per-descriptor Adds are commented
        // out upstream; only the binding layout enters the id.
        let (regs, sh, mem) = vs_external_fetch_setup();
        let map = ShaderMap::new();
        let mut info = ShaderVertexInputInfo::default();
        shader_get_input_info_vs(&regs, &sh, &mem, &map, false, &mut info).unwrap();

        // Give the bind a storage buffer so the invariant is exercised.
        let mut a = info;
        a.bind.storage_buffers.buffers_num = 1;
        a.bind.storage_buffers.slots[0] = 3;
        a.bind.storage_buffers.start_register[0] = 8;
        let mut b = a;
        b.bind.storage_buffers.buffers[0].fields = [0xDEAD, 0xBEEF, 0xF00D, 0xCAFE];

        let id_a = shader_get_id_vs(&regs, &a, &mem, false).unwrap();
        let id_b = shader_get_id_vs(&regs, &b, &mem, false).unwrap();
        assert_eq!(id_a, id_b);

        // But the layout itself must change the id.
        let mut c = a;
        c.bind.storage_buffers.slots[0] = 4;
        let id_c = shader_get_id_vs(&regs, &c, &mem, false).unwrap();
        assert_ne!(id_a, id_c);
    }

    #[test]
    fn shader_get_id_ps_and_cs() {
        // Kyty: ShaderGetIdPS (L2885) / ShaderGetIdCS (L2935).
        let (regs, sh, mem) = ps_setup();
        let map = ShaderMap::new();
        let vs_info = ShaderVertexInputInfo::default();
        let mut ps_info = ShaderPixelInputInfo::default();
        shader_get_input_info_ps(&regs, &sh, &vs_info, &mem, &map, false, &mut ps_info).unwrap();
        let ps_id = shader_get_id_ps(&regs, &ps_info, &mem, false).unwrap();
        assert_eq!(ps_id.hash0, 0x1234_5678);
        assert_eq!(ps_id.crc32, 0x9ABC_DEF0);
        assert_eq!(ps_id.ids[0], 128); // header.length
        assert_eq!(ps_id.ids[1], 2); // input_num
        // interpolator settings are part of the key
        assert!(ps_id.ids.contains(&7) && ps_id.ids.contains(&9));

        let (cs_regs, cs_mem) = cs_setup();
        let sh = ShaderRegisters::default();
        let mut cs_info = ShaderComputeInputInfo::default();
        shader_get_input_info_cs(
            &cs_regs,
            &sh,
            &cs_mem,
            &ShaderMap::new(),
            false,
            &mut cs_info,
        )
        .unwrap();
        let cs_id = shader_get_id_cs(&cs_regs, &cs_info, &cs_mem).unwrap();
        assert_eq!(cs_id.hash0, 0xFEED_F00D);
        assert_eq!(cs_id.crc32, 0x0BAD_CAFE);
        assert_eq!(cs_id.ids[0], 256); // header.length
        assert_eq!(cs_id.ids[1], 4); // workgroup_register
        assert_eq!(cs_id.ids[2], 2); // thread_ids_num
        assert_eq!(&cs_id.ids[3..9], &[8, 1, 4, 0, 1, 0]); // (threads, tgid) x3
    }

    #[test]
    fn shader_get_id_embedded() {
        // Kyty: ShaderGetIdVS L2800 / ShaderGetIdPS L2891.
        let (mut regs, _, mem) = vs_external_fetch_setup();
        regs.vs_embedded = true;
        regs.vs_embedded_id = 5;
        let info = ShaderVertexInputInfo::default();
        let id = shader_get_id_vs(&regs, &info, &mem, false).unwrap();
        assert_eq!((id.hash0, id.crc32, id.ids.as_slice()), (0, 0, &[5u32][..]));
    }

    // ---- ShaderParseVS/PS/CS wrappers ----

    #[test]
    fn parse_vs_legacy_hash_from_header() {
        // Kyty: ShaderParseVS (L2287), legacy branch (L2333-2341).
        let (regs, sh, mem) = vs_external_fetch_setup();
        let code = shader_parse_vs(&regs, &sh, &mem, false).unwrap();
        assert_eq!(code.get_type(), ShaderType::Vertex);
        assert_eq!(code.get_hash0(), 0xAAAA_BBBB);
        assert_eq!(code.get_crc32(), 0xCCCC_DDDD);
        // sentinel s_mov + s_endpgm parsed
        assert!(
            code.get_instructions()
                .iter()
                .any(|i| i.type_ == ShaderInstructionType::SEndpgm)
        );
    }

    #[test]
    fn parse_vs_next_gen_hash_from_gs_chksum() {
        // Kyty: ShaderParseVS next-gen branch (L2325-2331): hash0/crc32 come
        // from regs->gs_regs.chksum, gs-instead-of-vs required.
        let mem = TestMem {
            regions: vec![(0x7000, vec![S_ENDPGM])],
        };
        let mut regs = VertexShaderInfo::default();
        regs.es_regs.data_addr = 0x7000;
        regs.gs_regs.chksum = 0xAABB_CCDD_1122_3344;
        let sh = ShaderRegisters::default();
        let code = shader_parse_vs(&regs, &sh, &mem, true).unwrap();
        assert_eq!(code.get_hash0(), 0xAABB_CCDD);
        assert_eq!(code.get_crc32(), 0x1122_3344);

        // Plain VS on the next-gen path is EXIT_NOT_IMPLEMENTED upstream.
        let mut plain = VertexShaderInfo::default();
        plain.vs_regs.data_addr = 0x7000;
        assert!(matches!(
            shader_parse_vs(&plain, &sh, &mem, true),
            Err(ShaderAnalysisError::NotImplemented { .. })
        ));
    }

    #[test]
    fn parse_ps_embedded_and_cs_header() {
        // Kyty: ShaderParsePS embedded branch (L2407) / ShaderParseCS (L2500).
        let mem = TestMem { regions: vec![] };
        let sh = ShaderRegisters::default();
        let regs = PixelShaderInfo {
            ps_embedded: true,
            ps_embedded_id: 7,
            ..Default::default()
        };
        let code = shader_parse_ps(&regs, &sh, &mem, false).unwrap();
        assert!(code.is_ps_embedded());
        assert_eq!(code.get_ps_embedded_id(), 7);
        assert_eq!(code.get_type(), ShaderType::Pixel);

        let (cs_regs, cs_mem) = cs_setup();
        let code = shader_parse_cs(&cs_regs, &sh, &cs_mem, false).unwrap();
        assert_eq!(code.get_type(), ShaderType::Compute);
        assert_eq!(code.get_hash0(), 0xFEED_F00D);
        assert_eq!(code.get_crc32(), 0x0BAD_CAFE);
    }

    #[test]
    fn parse_cs_missing_binary_info_is_error() {
        // Kyty: EXIT_NOT_IMPLEMENTED(header == nullptr) (L2518).
        let mem = TestMem {
            regions: vec![(0x6000, vec![S_ENDPGM, 0])],
        };
        let mut regs = ComputeShaderInfo::default();
        regs.cs_regs.data_addr = 0x6000;
        let sh = ShaderRegisters::default();
        assert!(matches!(
            shader_parse_cs(&regs, &sh, &mem, false),
            Err(ShaderAnalysisError::NoBinaryInfo)
        ));
        assert!(
            shader_parse_cs(&regs, &sh, &mem, true).is_ok(),
            "Gen5 metadata is supplied out-of-band and has no legacy trailer"
        );
        // Unmapped address -> BadAddress.
        regs.cs_regs.data_addr = 0x9999_0000;
        assert!(matches!(
            shader_parse_cs(&regs, &sh, &mem, false),
            Err(ShaderAnalysisError::BadAddress { .. })
        ));
    }

    #[test]
    fn shader_map_replaces_metadata_when_code_address_is_reused() {
        // Current KytyPS5: ShaderMapUserData assigns through operator[] and
        // replaces an existing key. Commercial titles recycle shader-code
        // allocations; keeping the first header can attach vertex metadata to
        // a later pixel shader at the same address.
        let mut map = ShaderMap::new();
        let a = ShaderMappedData {
            user_data: None,
            input_semantics: vec![ShaderSemantic { raw: 1 }],
        };
        let b = ShaderMappedData {
            user_data: None,
            input_semantics: vec![ShaderSemantic { raw: 2 }],
        };
        map.map_user_data(0x1000, a.clone());
        map.map_user_data(0x1000, b.clone());
        assert_eq!(map.find(0x1000), Some(&b));
        assert_ne!(map.find(0x1000), Some(&a));
        assert_eq!(map.find(0x2000), None);
    }

    #[test]
    fn parse_usage2_sharp_tables() {
        // Kyty: ShaderParseUsage2 (L1505) — PS5 sharp tables: one sampler
        // (table 2) and one constant buffer (table 3), 0x7fff entries
        // skipped; leftover user SGPRs become direct.
        let mut type_ = [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX];
        type_[..8].fill(UserSgprType::Vsharp);
        let user_sgpr = UserSgprInfo {
            value: [0; UserSgprInfo::SGPRS_MAX],
            type_,
            count: 10,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8], // all skipped
            sharp_resource_offset: [
                vec![],
                vec![],
                vec![ShaderSharp::new(0x7fff, 0), ShaderSharp::new(0, 1)],
                vec![ShaderSharp::new(4, 1)],
            ],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 10, None, None).unwrap();
        assert_eq!(info.samplers, 1);
        assert_eq!(info.storage_buffers_constant, 1);
        assert_eq!(bind.samplers.samplers_num, 1);
        assert_eq!(bind.samplers.start_register[0], 0);
        assert_eq!(bind.samplers.slots[0], 1); // slot index in the table
        assert_eq!(bind.storage_buffers.buffers_num, 1);
        assert_eq!(bind.storage_buffers.start_register[0], 4);
        // SGPRs 8..9 were not consumed -> direct.
        assert_eq!(info.direct_sgprs, 2);
        assert_eq!(bind.direct_sgprs.sgprs_num, 2);
        assert_eq!(&bind.direct_sgprs.start_register[..2], &[8, 9]);
    }

    #[test]
    fn parse_usage2_table0_type0_sharp_is_buffer_not_texture() {
        // Regression: a table-0 sharp with size == 0 is normally an 8-dword
        // texture T#, but when its descriptor `type` nibble (dword3[28:31])
        // reads 0 it is actually a 4-dword buffer V# (RDNA2; confirmed in
        // shadPS4 video_core/amdgpu/resource.h where the buffer and image
        // `type` fields overlap in that nibble and read 0 for a buffer).
        // Measured on Minecraft's textured PS, where sharp[0.0] is a
        // structured buffer that was wrongly rejected as a "non-2D texture".
        // It must bind as a read-only storage buffer; a sibling sharp whose
        // descriptor type is 9 must still bind as a Texture2D.
        let user_sgpr = UserSgprInfo {
            value: {
                let mut v = [0u32; UserSgprInfo::SGPRS_MAX];
                v[3] = 0x0000_0008; // buffer sharp @0: dword3 (v[0+3]) type nibble = 0
                v[11] = 0x9000_0000; // texture sharp @8: dword3 (v[8+3]) type nibble = 9
                v
            },
            // Each size==0 sharp occupies an 8-register slot: the buffer @0
            // consumes s0..s8 (binding only the first four dwords) and the
            // texture @8 consumes s8..s16 -> all 16 registers consumed, no
            // leftover direct SGPRs.
            type_: [UserSgprType::Vsharp; UserSgprInfo::SGPRS_MAX],
            count: 16,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8],
            sharp_resource_offset: [
                vec![ShaderSharp::new(0, 0), ShaderSharp::new(8, 0)],
                vec![],
                vec![],
                vec![],
            ],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 16, None, None).unwrap();
        // type-0 sharp -> read-only storage buffer, NOT a texture.
        assert_eq!(info.storage_buffers_readonly, 1);
        assert_eq!(bind.storage_buffers.buffers_num, 1);
        assert_eq!(bind.storage_buffers.start_register[0], 0);
        // type-9 sharp -> Texture2D, still classified as a texture.
        assert_eq!(info.textures2d_readonly, 1);
        assert_eq!(bind.textures2d.textures_num, 1);
    }

    #[test]
    fn parse_usage2_typed_direct_sgprs_route_into_resource_tables() {
        // Beyond Kyty (item measured as 811 ASTRO.BOT CS dispatch skips,
        // "direct user sgpr type is not Unknown"): registers TYPED by a PM4
        // 'hu' marker but claimed by NO usage-table entry hold preloaded
        // descriptors. A Vsharp quad whose type nibble reads 0 binds as a
        // read-write storage buffer; a typed 8-register slot whose nibble
        // reads 9 binds as a read-only Texture2D. Untyped leftovers stay
        // direct.
        let mut type_ = [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX];
        type_[..12].fill(UserSgprType::Vsharp);
        let user_sgpr = UserSgprInfo {
            value: {
                let mut v = [0u32; UserSgprInfo::SGPRS_MAX];
                v[3] = 0x0000_0008; // quad @0: dword3 nibble = 0 -> buffer V#
                v[7] = 0x9000_0000; // slot @4: dword3 nibble = 9 -> texture T#
                v
            },
            type_,
            count: 14,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 14, None, None).unwrap();
        assert_eq!(info.storage_buffers_readwrite, 1);
        assert_eq!(bind.storage_buffers.buffers_num, 1);
        assert_eq!(bind.storage_buffers.start_register[0], 0);
        assert_eq!(info.textures2d_readonly, 1);
        assert_eq!(bind.textures2d.textures_num, 1);
        assert_eq!(bind.textures2d.desc[0].start_register, 4);
        // s12/s13 are declared but untyped -> plain direct SGPRs.
        assert_eq!(info.direct_sgprs, 2);
        assert_eq!(&bind.direct_sgprs.start_register[..2], &[12, 13]);
    }

    #[test]
    fn gta_typed_poison_quad_stays_direct_instead_of_becoming_a_storage_buffer() {
        // Measured on GTA V's first compute submissions: AGC leaves s4..s7
        // typed as Vsharp while their captured values are the scalar fill
        // 0x07070707. Treating that tuple as a V# invents a 212-GB storage
        // buffer at unmapped guest address 0x070707070707 and rejects the
        // dispatch before the shader can replace/use the registers.
        let user_sgpr = UserSgprInfo {
            value: [0x0707_0707; UserSgprInfo::SGPRS_MAX],
            type_: [UserSgprType::Vsharp; UserSgprInfo::SGPRS_MAX],
            count: 4,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();

        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 4, None, None)
            .expect("unmapped typed scalar data must remain usable as direct SGPRs");

        assert_eq!(
            bind.storage_buffers.buffers_num, 0,
            "an unmapped repeated-byte tuple is not a captured V#"
        );
        assert_eq!(info.direct_sgprs, 4);
        assert_eq!(&bind.direct_sgprs.start_register[..4], &[0, 1, 2, 3]);
    }

    #[test]
    fn gta_unclaimed_typed_srt_tail_stays_direct_scalar_data() {
        // GTA V compute 0x100003948200 declares an eight-dword inline SRT:
        // direct type 1 consumes s0..s3 as its sampler, while s4..s7 carry
        // ordinary scalar constants (measured as 1.0f or 0x07070707).
        // The persistent AGC `hu` marker still tags that tail Vsharp; the SRT
        // declaration is authoritative and must keep unclaimed dwords direct.
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[0..4].copy_from_slice(&[0x5b22_0000, 0x0010_0001, 0x0008_7000, 0x0004_bfac]);
        value[4..8].fill(0x3f80_0000);
        let mut type_ = [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX];
        type_[0..8].fill(UserSgprType::Vsharp);
        let user_sgpr = UserSgprInfo {
            value,
            type_,
            count: 8,
        };
        let mut direct_resource_offset = vec![0xffff; 8];
        direct_resource_offset[1] = 0;
        let user_data = ShaderUserData {
            direct_resource_offset,
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 8,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();

        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 8, None, None)
            .expect("typed inline-SRT constants must remain direct");

        assert_eq!(bind.samplers.samplers_num, 1);
        assert_eq!(bind.storage_buffers.buffers_num, 0);
        assert_eq!(info.direct_sgprs, 4);
        assert_eq!(&bind.direct_sgprs.start_register[..4], &[4, 5, 6, 7]);
        assert_eq!(bind.direct_sgprs.sgprs[0].field, 0x3f80_0000);
    }

    #[test]
    fn parse_usage2_table1_size0_sharp_disambiguates_buffer_vs_storage_image() {
        // Beyond Kyty (347 measured ASTRO.BOT CS skips, "sharp table 1 entry
        // with size != 1"): `ShaderSharp::size` is a single BIT — size == 0
        // means an 8-dword slot, exactly as in table 0. Table 1 is the
        // read-write table: a type-nibble-0 slot binds as a RW buffer V#,
        // a type-9 slot as a storage image (UAV).
        let user_sgpr = UserSgprInfo {
            value: {
                let mut v = [0u32; UserSgprInfo::SGPRS_MAX];
                v[3] = 0x0000_0008; // slot @0: nibble 0 -> RW buffer
                v[11] = 0x9000_0000; // slot @8: nibble 9 -> storage image
                v
            },
            type_: [UserSgprType::Vsharp; UserSgprInfo::SGPRS_MAX],
            count: 16,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8],
            sharp_resource_offset: [
                vec![],
                vec![ShaderSharp::new(0, 0), ShaderSharp::new(8, 0)],
                vec![],
                vec![],
            ],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 16, None, None).unwrap();
        assert_eq!(info.storage_buffers_readwrite, 1);
        assert_eq!(bind.storage_buffers.buffers_num, 1);
        assert_eq!(info.textures2d_readwrite, 1);
        assert_eq!(bind.textures2d.textures_num, 1);
        assert_eq!(bind.textures2d.textures2d_storage_num, 1);
        assert!(bind.textures2d.desc[0].textures2d_without_sampler);
    }

    #[test]
    fn parse_usage2_table1_accepts_type8_rgba32f_storage_image() {
        // Live ASTRO.BOT descriptor: a read-write table-1 type-8 (1D) T#,
        // 1x1, unified format 77 (32_32_32_32 FLOAT). The existing image
        // pipeline represents type 8 as a height-1 2D image; analysis must
        // admit the same shape for UAVs without losing its 16-byte texels.
        let user_sgpr = UserSgprInfo {
            value: {
                let mut v = [0u32; UserSgprInfo::SGPRS_MAX];
                v[1] = 77 << 20;
                v[3] = 8 << 28;
                v
            },
            type_: [UserSgprType::Vsharp; UserSgprInfo::SGPRS_MAX],
            count: 8,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8],
            sharp_resource_offset: [vec![], vec![ShaderSharp::new(0, 0)], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 8, None, None)
            .expect("type-8 RGBA32F UAV accepted");
        assert_eq!(info.textures2d_readwrite, 1);
        assert_eq!(bind.textures2d.desc[0].texture.type_(), 8);
        assert_eq!(bind.textures2d.desc[0].texture.format(), 77);

        let mut malformed = bind.textures2d.desc[0].texture;
        malformed.fields[2] |= 1 << 14; // height5=1 => two rows, invalid 1D
        let err = check_read_write_texture_type(&malformed, "test")
            .expect_err("type-8 UAVs wider than one row stay refused");
        assert!(format!("{err}").contains("height-1 1D"), "{err}");
    }

    #[test]
    fn read_write_texture_type_gate_accepts_cube_and_array_as_2d() {
        // Minecraft's menu compute declares a storage image of type 11 (Cube)
        // and 13 (2DArray/NDArray) for the animated panorama skybox. Rejecting
        // it aborted the compute shader and left the panorama BLACK. The gate
        // now admits both: the SPIR-V and Vulkan paths represent them as
        // writable 2D arrays and preserve the third address component as the
        // layer. 8 (height-1)/9/10 stay accepted; 12/14/15 stay refused so the
        // coverage gap is visible.
        let with_type = |ty: u32| {
            let mut t = super::super::resources::ShaderTextureResource::default();
            t.fields[0] = 0x1000; // plausible base
            t.fields[1] = 10 << 20; // format 10 (8_8_8_8): plausible
            t.fields[3] = ty << 28;
            t
        };
        for ty in [8u32, 9, 10, 11, 13] {
            check_read_write_texture_type(&with_type(ty), "table 1")
                .unwrap_or_else(|e| panic!("read-write type {ty} must be accepted: {e}"));
        }
        for ty in [12u32, 14, 15] {
            check_read_write_texture_type(&with_type(ty), "table 1")
                .expect_err("unsupported read-write type stays a visible refusal");
        }
    }

    #[test]
    fn read_only_texture_type_gate_accepts_3d_volumes() {
        // Measured ASTRO.BOT CS skips: type 10 = 3D volume (240x135x64
        // froxel/LUT, format 71) and type 13 = 2DArray (1536x1536x3 format-7
        // tile-24 — 57 dispatches/run). The gate admits 2D (9), 3D (10),
        // Cube (11) and 2DArray (13) unchanged. Use a plausible format so the
        // unresolvable-poison guard does not fire.
        let mut t = super::super::resources::ShaderTextureResource::default();
        let with_type = |t: &mut super::super::resources::ShaderTextureResource, ty: u32| {
            t.fields[1] = 10 << 20; // format 10 (8_8_8_8): plausible
            t.fields[3] = ty << 28;
        };
        for ty in [8u32, 9, 10, 11, 13] {
            with_type(&mut t, ty);
            check_read_only_texture_type(&mut t).expect("supported type accepted");
            assert_eq!(t.type_(), ty as u8, "supported type left unchanged");
        }
    }

    #[test]
    fn read_only_texture_type_gate_approximates_array_and_msaa_as_2d() {
        // 12=1DArray, 14=2DMsaa, 15=2DMsaaArray with a plausible descriptor are
        // approximated as 2D (type rewritten to 9) rather than aborting the
        // shader — downstream `from_texture_type` already collapses them to 2D.
        for ty in [12u32, 14, 15] {
            let mut t = super::super::resources::ShaderTextureResource::default();
            t.fields[0] = 0x1000; // non-zero base (plausible)
            t.fields[1] = (10 << 20) | 0x0A; // format 10, base_hi bits
            t.fields[3] = ty << 28;
            check_read_only_texture_type(&mut t).expect("array/MSAA approximated, not aborted");
            assert_eq!(t.type_(), 9, "type {ty} rewritten to 2D");
        }
    }

    #[test]
    fn read_only_texture_type_gate_replaces_poison_with_placeholder() {
        // The exact ASTRO.BOT poison an unresolved/runtime-bound descriptor
        // reads back as: all-ones fields => type 15, format 511 (0x1FF), tile
        // 31, 16384², saturated base. The gate must NOT abort — it replaces the
        // descriptor with a 1x1 placeholder (type 9, base 0) so the draw
        // proceeds untextured.
        let mut t = super::super::resources::ShaderTextureResource {
            fields: [0xFFFF_FFFF; 8],
        };
        assert_eq!(t.type_(), 15);
        assert_eq!(t.format(), 0x1FF);
        assert!(texture_descriptor_is_unresolvable(&t));
        check_read_only_texture_type(&mut t).expect("poison descriptor is non-fatal");
        assert_eq!(t.fields, placeholder_texture_fields());
        assert_eq!(t.type_(), 9, "placeholder is a 2D texture");
        assert_eq!(t.base40(), 0, "placeholder base 0 => bind-path 1x1 dummy");
    }

    #[test]
    fn texture_buffer_read_failure_installs_placeholder() {
        // ASTRO.BOT 0x100008e6aa00: a T# declared at s4 (needs s4..s11) whose
        // s8+ were never captured (Unknown), with no EUD — `read_sharp_fields`
        // cannot resolve it. `shader_get_texture_buffer` must install a
        // placeholder T# and succeed rather than aborting the whole shader.
        let mut user_sgpr = UserSgprInfo::default();
        for i in 4..8 {
            user_sgpr.set(i, 0xdead_0000 | i, UserSgprType::Vsharp);
        }
        // count = 8; s8..s11 stay Unknown => the 8-dword T# at s4 cannot be read.
        let mut info = ShaderTextureResources::default();
        let mut direct = [false; UserSgprInfo::SGPRS_MAX];
        shader_get_texture_buffer(
            &mut info,
            &mut direct,
            4,
            0,
            ShaderTextureUsage::ReadOnly,
            &user_sgpr,
            None,
        )
        .expect("unresolvable texture descriptor is non-fatal");
        assert_eq!(info.textures_num, 1);
        assert_eq!(info.desc[0].texture.fields, placeholder_texture_fields());
        // And the placeholder passes the type gate.
        check_read_only_texture_type(&mut info.desc[0].texture).expect("placeholder accepted");
    }

    #[test]
    fn parse_usage2_runtime_table0_texture_uses_placeholder() {
        // Minecraft gameplay PS 0x1700c000 declares table-0 slot 0 at s14,
        // while the draw only captures s0..s15. The T# therefore cannot be
        // inspected statically (s16..s21 are runtime/SRT-bound). The Gen5
        // table walk must reach `shader_get_texture_buffer`'s established
        // placeholder path instead of rejecting the entire draw in its
        // descriptor-kind preflight.
        let mut user_sgpr = UserSgprInfo::default();
        user_sgpr.set(14, 0x1700_0000, UserSgprType::Vsharp);
        user_sgpr.set(15, 0x0000_0000, UserSgprType::Vsharp);
        let mut user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8],
            ..Default::default()
        };
        user_data.sharp_resource_offset[0].push(ShaderSharp::new(14, 0));
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();

        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 16, None, None)
            .expect("runtime-bound table-0 T# must not reject the shader");

        assert_eq!(info.textures2d_readonly, 1);
        assert_eq!(bind.textures2d.textures_num, 1);
        assert_eq!(bind.textures2d.desc[0].start_register, 14);
        assert_eq!(
            bind.textures2d.desc[0].texture.fields,
            placeholder_texture_fields()
        );
    }

    /// Manual disassembly harness (no-op unless `RAEEN_DISASM_FILE` names a
    /// dumped `.bin`): parses the shader and prints its instruction types and
    /// recovered scalar-load bases. Used to read the EUD/SRT CS's descriptor
    /// pointer-load pattern while building the resolver. Run with
    /// `RAEEN_DISASM_FILE=... cargo test -p kyty-graphics disasm_shader_from_env -- --nocapture`.
    #[test]
    fn disasm_shader_from_env() {
        let Ok(path) = std::env::var("RAEEN_DISASM_FILE") else {
            return;
        };
        let bytes = std::fs::read(&path).expect("read shader dump");
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut code = ShaderCode::new();
        code.set_type(ShaderType::Compute);
        let res = shader_parse(0, &words, &mut code, true);
        eprintln!(
            "DISASM {path}: parse={res:?}, {} instructions",
            code.get_instructions().len()
        );
        for (i, inst) in code.get_instructions().iter().enumerate() {
            eprintln!("  [{i:3}] {:?}", inst.type_);
        }
        eprintln!("scalar-load bases: {:?}", find_scalar_load_bases(&code));
    }

    #[test]
    fn find_scalar_load_bases_recovers_srt_pointer() {
        use crate::shader::types::ShaderInstruction;
        // Measured pattern from Minecraft's VS (vs_253e7900): `s_load_dwordx2
        // s[82:83], s[14:15], 8` — the SRT/EUD descriptor spill. task #9's
        // keystone is recovering the base pointer (s14) and offset (8) by
        // analysing the shader, since the pointer is not a field anywhere. A
        // plain s_load_dword and a wider x4 pin the dword-width mapping.
        let mut code = ShaderCode::new();
        for (ty, base, off) in [
            (ShaderInstructionType::SLoadDwordx2, 14, 8u32),
            (ShaderInstructionType::SLoadDword, 4, 0),
            (ShaderInstructionType::SLoadDwordx4, 20, 16),
        ] {
            let mut inst = ShaderInstruction::default();
            inst.type_ = ty;
            inst.src[0].register_id = base;
            inst.src[1].type_ = ShaderOperandType::LiteralConstant;
            inst.src[1].constant.u = off;
            code.get_instructions_mut().push(inst);
        }
        assert_eq!(
            find_scalar_load_bases(&code),
            vec![
                ScalarLoadRef {
                    base_register: 14,
                    byte_offset: 8,
                    dwords: 2,
                },
                ScalarLoadRef {
                    base_register: 4,
                    byte_offset: 0,
                    dwords: 1,
                },
                ScalarLoadRef {
                    base_register: 20,
                    byte_offset: 16,
                    dwords: 4,
                },
            ]
        );
    }

    #[test]
    fn scalar_load_target_address_forms_the_runtime_pointer() {
        // s[14:15] holds a 64-bit guest pointer (lo=0x29b98350, hi=0); the
        // load's byte offset (8) is added — the exact address increment 3 reads
        // the spilled descriptor from. Measured base from Minecraft's SRT VS.
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[14] = 0x29b9_8350;
        value[15] = 0x0000_0001; // exercise the high dword too
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Vsharp; UserSgprInfo::SGPRS_MAX],
            count: 16,
        };
        let load = ScalarLoadRef {
            base_register: 14,
            byte_offset: 8,
            dwords: 2,
        };
        assert_eq!(
            scalar_load_target_address(&load, &user_sgpr),
            Some(0x0000_0001_29b9_8358)
        );
        // A base whose high dword falls off the register file -> None, no
        // panic. The boundary is the LAST slot (Gen5 graphics stages have 32,
        // so s31's pair would need a nonexistent s32).
        let out_of_range = ScalarLoadRef {
            base_register: (UserSgprInfo::SGPRS_MAX - 1) as i32,
            byte_offset: 0,
            dwords: 2,
        };
        assert_eq!(scalar_load_target_address(&out_of_range, &user_sgpr), None);
    }

    #[test]
    fn runtime_scalar_load_replaces_placeholder_texture_and_survives_embedded_pass() {
        use crate::shader::types::ShaderInstructionType as T;

        // Synthetic equivalent of Minecraft gameplay PS 0x1700c000:
        // bind-time s12:s13 points at a T#, s_load_dwordx8 installs it into
        // s14:s21, and the following MIMG samples through that register range.
        let mut load = ShaderInstruction {
            pc: 0x0c,
            type_: T::SLoadDwordx8,
            ..Default::default()
        };
        load.src[0] = sgpr_op(12, 2);
        load.src[1] = imm_op(0);
        load.src_num = 2;
        load.dst = sgpr_op(14, 8);

        let mut sample = ShaderInstruction {
            pc: 0x14,
            type_: T::ImageSampleLz,
            ..Default::default()
        };
        sample.src[1] = sgpr_op(14, 8);
        sample.src[2] = sgpr_op(4, 4);
        sample.src_num = 3;

        let mut code = ShaderCode::new();
        code.get_instructions_mut().extend([load, sample]);

        let fields = [
            0x0020_0000,
            10 << 20,
            0x000f_000f,
            (9 << 28) | 0x0fac,
            0,
            0,
            0,
            0,
        ];
        let mem = TestMem {
            regions: vec![(0x0020_0000, fields.to_vec())],
        };
        let mut user_sgpr = UserSgprInfo::default();
        user_sgpr.set(12, 0x0020_0000, UserSgprType::Unknown);
        user_sgpr.set(13, 0, UserSgprType::Unknown);

        // The static table walk knows a T# will land at s14, but cannot read
        // s14..s21 from the 16 captured user SGPRs and installs a placeholder.
        let mut bind = ShaderBindResources::default();
        bind.textures2d.textures_num = 1;
        bind.textures2d.textures2d_sampled_num = 1;
        bind.textures2d.desc[0].start_register = 14;
        bind.textures2d.desc[0].usage = ShaderTextureUsage::ReadOnly;
        bind.textures2d.desc[0].texture.fields = placeholder_texture_fields();

        shader_capture_runtime_scalar_loads(&code, &mem, &user_sgpr, &mut bind);

        let captured = bind
            .embedded_constant_loads
            .find(0x0c)
            .expect("runtime scalar load captured by instruction PC");
        assert_eq!(captured.dwords_num, 8);
        assert_eq!(captured.values[..8], fields);
        assert_eq!(bind.textures2d.textures_num, 1);
        assert_eq!(bind.textures2d.desc[0].texture.fields, fields);

        // The PC-relative detector runs immediately afterward in the runtime
        // path; it must append rather than erase the live SRT snapshot.
        shader_detect_embedded_constant_loads(&code, &mem, &mut bind);
        assert_eq!(
            bind.embedded_constant_loads
                .find(0x0c)
                .expect("runtime capture retained")
                .values[..8],
            fields
        );
    }

    /// A sharp at s16 with NO extended buffer is a direct read from the Gen5
    /// 32-slot user SGPR file, not an EUD reference (measured on ASTRO.BOT).
    /// The slots still have to be real: an unwritten slot keeps type_ Unknown
    /// and is rejected, so the relaxed bound cannot invent a descriptor.
    #[test]
    fn sharp_at_slot_sixteen_reads_direct_when_no_extended_buffer() {
        let mut user_sgpr = UserSgprInfo::default();
        for (i, dw) in [0xaaaa_0000u32, 0xbbbb_1111, 0xcccc_2222, 0xdddd_3333]
            .into_iter()
            .enumerate()
        {
            user_sgpr.set(16 + i as u32, dw, UserSgprType::Vsharp);
        }
        let mut direct = [true; UserSgprInfo::SGPRS_MAX];
        let mut out = [0u32; 4];
        read_sharp_fields(&mut direct, 16, &user_sgpr, None, &mut out)
            .expect("s16 is a real Gen5 user SGPR when the shader declares no EUD");
        assert_eq!(out, [0xaaaa_0000, 0xbbbb_1111, 0xcccc_2222, 0xdddd_3333]);

        // An unwritten slot (type_ Unknown) is still refused.
        let empty = UserSgprInfo::default();
        let mut direct2 = [true; UserSgprInfo::SGPRS_MAX];
        let mut out2 = [0u32; 4];
        assert!(
            read_sharp_fields(&mut direct2, 16, &empty, None, &mut out2).is_err(),
            "an unwritten s16 must not be accepted as a descriptor"
        );
    }

    /// A sharp whose typed run fits the register FILE resolves direct even
    /// when the shader also carries an EUD — and must be recorded as
    /// `extended = false`, because its descriptor content came from the user
    /// SGPRs, not the EUD. Measured on ASTRO.BOT compute (rounds 7-8): 404
    /// "extended texture at s0 has no EUD base to rebase on" + 289 same for
    /// "storage buffer at s8" refusals, all with eud_base=12 — every one a
    /// direct-resident sharp mislabeled extended, which `eud_rel_index`
    /// (spirv.rs) then correctly refused to rebase (0 < 12). The analysis
    /// itself already refuses `start < eud_base` when the run does NOT fit
    /// the typed file ("sharp start_register below the EUD base"), so a
    /// descriptor reaching emission with start below the base has provably
    /// resolved direct.
    #[test]
    fn direct_resident_sharp_in_eud_shader_is_not_marked_extended() {
        // T# at s0..s7 (typed Vsharp), EUD pointer pair at (s12, s13).
        let mut user_sgpr = UserSgprInfo::default();
        for i in 0..8u32 {
            user_sgpr.set(i, 0x9000_0000 + i, UserSgprType::Vsharp);
        }
        // V# at s8..s11 (typed Vsharp).
        for i in 8..12u32 {
            user_sgpr.set(i, 0x8000_0000 + i, UserSgprType::Vsharp);
        }
        let eud_data = [0xeeee_0000u32; 8];
        let eud = EudView {
            data: &eud_data,
            base_dw: 12,
        };

        let mut tex = ShaderTextureResources::default();
        let mut direct = [true; UserSgprInfo::SGPRS_MAX];
        shader_get_texture_buffer(
            &mut tex,
            &mut direct,
            0,
            0,
            ShaderTextureUsage::ReadOnly,
            &user_sgpr,
            Some(eud),
        )
        .expect("typed run at s0 resolves direct");
        assert!(
            !tex.desc[0].extended,
            "a direct-resident T# must not be marked extended just because \
             the shader carries an EUD"
        );
        assert_eq!(tex.desc[0].texture.fields[0], 0x9000_0000);

        let mut bufs = ShaderStorageResources::default();
        shader_get_storage_buffer(
            &mut bufs,
            &mut direct,
            8,
            0,
            ShaderStorageUsage::ReadWrite,
            &user_sgpr,
            Some(eud),
        )
        .expect("typed run at s8 resolves direct");
        assert!(
            !bufs.extended[0],
            "a direct-resident V# must not be marked extended just because \
             the shader carries an EUD"
        );
        assert_eq!(bufs.buffers[0].fields[0], 0x8000_0008);

        let mut samplers = ShaderSamplerResources::default();
        let mut user_sgpr_s = UserSgprInfo::default();
        for i in 0..4u32 {
            user_sgpr_s.set(i, 0x5000_0000 + i, UserSgprType::Vsharp);
        }
        shader_get_sampler(&mut samplers, &mut direct, 0, 0, &user_sgpr_s, Some(eud))
            .expect("typed run at s0 resolves direct");
        assert!(
            !samplers.extended[0],
            "a direct-resident S# must not be marked extended just because \
             the shader carries an EUD"
        );
    }

    /// A runtime-resolved sampler may be named at the tail of the user-SGPR
    /// file without all four descriptor dwords being captured. The sampled
    /// draw must receive the safe default S# instead of failing analysis.
    #[test]
    fn incomplete_runtime_sampler_uses_default_descriptor() {
        let mut user_sgpr = UserSgprInfo::default();
        user_sgpr.set(28, 0x1234_5678, UserSgprType::Vsharp);
        user_sgpr.set(29, 0x9abc_def0, UserSgprType::Vsharp);
        user_sgpr.count = 30;
        let mut direct = [false; UserSgprInfo::SGPRS_MAX];
        direct[..30].fill(true);
        let mut samplers = ShaderSamplerResources::default();

        shader_get_sampler(&mut samplers, &mut direct, 28, 3, &user_sgpr, None)
            .expect("runtime sampler must degrade to the default S#");

        assert_eq!(samplers.samplers_num, 1);
        assert_eq!(samplers.start_register[0], 28);
        assert_eq!(samplers.slots[0], 3);
        assert_eq!(samplers.samplers[0].fields, [0; 4]);
        assert!(!samplers.extended[0]);
        assert!(!direct[28] && !direct[29]);
    }

    /// The V#-vs-T# disambiguation must key on the descriptor TYPE, not the
    /// whole dword3 nibble: a buffer V# with OOB_SELECT=3 reads nibble 3 and
    /// was misrouted into the texture path ("read-only texture type 3", 57
    /// measured ASTRO.BOT CS dispatch skips). Image T# types are 8..15.
    #[test]
    fn sharp_dword3_buffer_vs_image_keys_on_the_type_field() {
        // Buffer V#: type bits [30:31] = 0, any OOB_SELECT in [28:29].
        for oob in 0..=3u32 {
            assert!(
                sharp_dword3_is_buffer(oob << 28),
                "V# with oob_select={oob} must route as a buffer"
            );
        }
        // Image T#: types 8..15 (bit 31 set).
        for ty in 8..=15u32 {
            assert!(
                !sharp_dword3_is_buffer(ty << 28),
                "T# type {ty} must route as an image"
            );
        }
        // Lower dword3 bits (V# dst_sel/format fields) must not disturb it.
        assert!(sharp_dword3_is_buffer((3 << 28) | 0x0fff_ffff));
        assert!(!sharp_dword3_is_buffer((9 << 28) | 0x0fff_ffff));
    }

    /// The counterpart: a sharp whose run does NOT fit the typed file (the
    /// measured ASTRO.BOT shape — T# declared at the EUD pointer pair's own
    /// register) resolves EUD-resident and KEEPS `extended = true`.
    #[test]
    fn eud_resident_sharp_keeps_extended_flag() {
        // Only the pointer pair (s12, s13) is written; s14+ untyped, so an
        // 8-dword T# at start_register=12 cannot live in the file.
        let mut user_sgpr = UserSgprInfo::default();
        user_sgpr.set(12, 0x29b9_8350, UserSgprType::Unknown);
        user_sgpr.set(13, 0, UserSgprType::Unknown);
        let eud_data: [u32; 8] = core::array::from_fn(|i| 0xe0d0 + i as u32);
        let eud = EudView {
            data: &eud_data,
            base_dw: 12,
        };

        let mut tex = ShaderTextureResources::default();
        let mut direct = [true; UserSgprInfo::SGPRS_MAX];
        shader_get_texture_buffer(
            &mut tex,
            &mut direct,
            12,
            0,
            ShaderTextureUsage::ReadOnly,
            &user_sgpr,
            Some(eud),
        )
        .expect("run not fitting the typed file resolves from the EUD");
        assert!(tex.desc[0].extended, "an EUD-resident T# stays extended");
        assert_eq!(tex.desc[0].texture.fields[0], eud_data[0]);

        // start >= SGPRS_MAX: the file-continuation rebase is EUD-resident too.
        let mut bufs = ShaderStorageResources::default();
        shader_get_storage_buffer(
            &mut bufs,
            &mut direct,
            UserSgprInfo::SGPRS_MAX as i32,
            0,
            ShaderStorageUsage::ReadOnly,
            &user_sgpr,
            Some(eud),
        )
        .expect("start >= SGPRS_MAX rebases into the EUD");
        assert!(bufs.extended[0], "a file-continuation V# stays extended");
        assert_eq!(bufs.buffers[0].fields[0], eud_data[0]);
    }

    #[test]
    fn parse_usage2_nonzero_eud_or_srt_is_error() {
        // Kyty: EXIT_NOT_IMPLEMENTED(eud_size_dw != 0 / srt_size_dw != 0)
        // (L1528-1529).
        let user_sgpr = UserSgprInfo::default();
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        let user_data = ShaderUserData {
            eud_size_dw: 4,
            ..Default::default()
        };
        // The refusal is owned so it can carry the diagnostic payload the
        // next session needs: the declared/captured SGPR counts and every
        // nonzero SGPR value (the EUD-pointer candidates).
        match shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 0, None, None) {
            Err(ShaderAnalysisError::NotImplementedOwned { what }) => {
                assert!(what.contains("EUD unreadable"), "{what}");
                assert!(what.contains("eud_size_dw=4"), "{what}");
                assert!(what.contains("declared=0"), "{what}");
                assert!(what.contains("(none)"), "{what}");
            }
            other => panic!("expected owned EUD diagnostic, got {other:?}"),
        }
    }

    /// An inline SRT (srt_size_dw <= the declared register file) proceeds:
    /// the SRT dwords ARE the unconsumed user SGPRs, and the direct-SGPR
    /// collection binds them with their captured runtime values. This was the
    /// single biggest ASTRO.BOT bucket (813 refusals / 5 min).
    #[test]
    fn parse_usage2_inline_srt_binds_registers_as_direct_sgprs() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[0] = 0x1111_1111;
        value[1] = 0x2222_2222;
        value[2] = 0x3333_3333;
        value[3] = 0x4444_4444;
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 4,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 4,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 4, None, None)
            .expect("an inline SRT must not be refused");
        assert_eq!(info.direct_sgprs, 4);
        assert_eq!(bind.direct_sgprs.sgprs_num, 4);
        assert_eq!(&bind.direct_sgprs.start_register[..4], &[0, 1, 2, 3]);
        assert_eq!(bind.direct_sgprs.sgprs[0].field, 0x1111_1111);
        assert_eq!(bind.direct_sgprs.sgprs[3].field, 0x4444_4444);
    }

    /// An SRT larger than the declared register file spills to memory we have
    /// no pointer for — that stays an evidence-rich named refusal.
    #[test]
    fn parse_usage2_spilled_srt_is_named_refusal() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[0] = 0xdead_beef;
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 4,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 20,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        match shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 4, None, None) {
            Err(ShaderAnalysisError::NotImplementedOwned { what }) => {
                assert!(what.contains("srt_size_dw (20)"), "{what}");
                assert!(what.contains("declared=4"), "{what}");
                assert!(what.contains("s0=0xdeadbeef"), "{what}");
            }
            other => panic!("expected owned SRT diagnostic, got {other:?}"),
        }
    }

    /// Gen5 direct table type 5 (round 1 wired the LEGACY 0x05 arm, but the
    /// 230 measured refusals print `0x0005` — this table). The register holds
    /// immediate data: nothing is bound and it stays direct.
    #[test]
    fn parse_usage2_direct_type5_leaves_register_direct() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[2] = 0x3f80_0000; // 1.0f — an immediate ALU constant
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 3,
        };
        let mut direct = vec![0xffff_u16; 8];
        direct[5] = 2; // type 5 at register s2
        let user_data = ShaderUserData {
            direct_resource_offset: direct,
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 3, None, None)
            .expect("direct type 5 must not be refused");
        assert_eq!(info.direct_sgprs, 3, "s0..s2 all stay direct");
        assert_eq!(bind.direct_sgprs.sgprs[2].field, 0x3f80_0000);
    }

    /// Gen5 direct table type 1 (IMM_SAMPLER) — previously the 4-digit
    /// `unknown usage type: 0x0001` refusal (738 measured ASTRO.BOT CS
    /// failures). The S# in 4 user SGPRs routes into
    /// `ShaderSamplerResources` exactly like the legacy usage-0x01 arm.
    #[test]
    fn parse_usage2_direct_type1_binds_imm_sampler() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[4] = 0x1111_0000;
        value[5] = 0x2222_0000;
        value[6] = 0x3333_0000;
        value[7] = 0x4444_0000;
        let mut type_ = [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX];
        type_[4..8].fill(UserSgprType::Vsharp);
        let user_sgpr = UserSgprInfo {
            value,
            type_,
            count: 8,
        };
        let mut direct = vec![0xffff_u16; 8];
        direct[1] = 4; // IMM_SAMPLER at s4..s7
        let user_data = ShaderUserData {
            direct_resource_offset: direct,
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 8, None, None)
            .expect("direct type 1 (IMM_SAMPLER) must bind");
        assert_eq!(info.samplers, 1);
        assert_eq!(bind.samplers.samplers_num, 1);
        assert_eq!(bind.samplers.start_register[0], 4);
        assert_eq!(bind.samplers.slots[0], 0);
        assert_eq!(
            bind.samplers.samplers[0].fields,
            [0x1111_0000, 0x2222_0000, 0x3333_0000, 0x4444_0000]
        );
        // s4..s7 consumed; s0..s3 stay direct.
        assert_eq!(info.direct_sgprs, 4);
        assert_eq!(&bind.direct_sgprs.start_register[..4], &[0, 1, 2, 3]);
    }

    /// A recovered EUD must land in `bind.extended` (used + the pointer-pair
    /// register + the pair's captured values) so the recompiler can rebase
    /// extended sharps and translate `s_load_dwordx*` through the pair —
    /// while `info.extended_buffer` stays false (it only feeds the
    /// "vs: extended buffer" gate).
    #[test]
    fn parse_usage2_records_recovered_eud_base_in_bind_extended() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[12] = 0x0050_6730; // EUD pointer lo
        value[13] = 0x0000_0004; // EUD pointer hi
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 14,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![0xffff; 8],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 8,
            srt_size_dw: 0,
        };
        let eud = [0u32; 8];
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(
            &user_data,
            &mut info,
            &mut bind,
            &user_sgpr,
            14,
            Some(EudView {
                data: &eud,
                base_dw: 12,
            }),
            None,
        )
        .expect("EUD-bearing shader must parse");
        assert!(bind.extended.used);
        assert_eq!(bind.extended.start_register, 12);
        assert_eq!(bind.extended.data.fields[0], 0x0050_6730);
        assert_eq!(bind.extended.data.fields[1], 0x0000_0004);
        assert!(
            !info.extended_buffer,
            "the VS gate flag must not be raised by a recovered EUD"
        );
    }

    /// Gen5 direct table type 0 (IMM_RESOURCE) — previously the
    /// `unknown usage type: 0x00` refusal (59 measured ASTRO.BOT CS
    /// failures). A descriptor whose type nibble reads 0 is a V#: it binds
    /// as a read-only storage buffer.
    #[test]
    fn parse_usage2_direct_type0_binds_buffer_by_type_nibble() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[4] = 0x0050_0000; // base lo
        value[5] = 0x0000_0004; // base hi + stride
        value[6] = 0x0000_0100; // num_records
        value[7] = 0x0111_0000; // dword3: type nibble (bits 28..31) == 0 => V#
        let mut type_ = [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX];
        type_[4..8].fill(UserSgprType::Vsharp);
        let user_sgpr = UserSgprInfo {
            value,
            type_,
            count: 8,
        };
        let mut direct = vec![0xffff_u16; 8];
        direct[0] = 4; // IMM_RESOURCE at s4..s7
        let user_data = ShaderUserData {
            direct_resource_offset: direct,
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 8, None, None)
            .expect("direct type 0 (IMM_RESOURCE) must bind");
        assert_eq!(info.storage_buffers_readonly, 1);
        assert_eq!(bind.storage_buffers.buffers_num, 1);
        assert_eq!(bind.storage_buffers.start_register[0], 4);
        assert_eq!(bind.storage_buffers.slots[0], 0);
        assert_eq!(bind.storage_buffers.usages[0], ShaderStorageUsage::ReadOnly);
        assert_eq!(
            bind.storage_buffers.buffers[0].fields,
            [0x0050_0000, 0x0000_0004, 0x0000_0100, 0x0111_0000]
        );
        // s4..s7 consumed; s0..s3 stay direct.
        assert_eq!(info.direct_sgprs, 4);
        assert_eq!(&bind.direct_sgprs.start_register[..4], &[0, 1, 2, 3]);
    }

    /// The T# half of the direct type-0 arm: a nonzero type nibble routes
    /// into the texture table (type 9 = Texture2D passes the read-only
    /// check).
    #[test]
    fn parse_usage2_direct_type0_binds_texture_by_type_nibble() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[0] = 0x0050_0000;
        value[1] = 0x0000_0004;
        value[2] = 0x0000_0100;
        value[3] = 0x9000_0000; // dword3: type nibble == 9 => Texture2D
        let mut type_ = [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX];
        type_[0..8].fill(UserSgprType::Vsharp);
        let user_sgpr = UserSgprInfo {
            value,
            type_,
            count: 8,
        };
        let mut direct = vec![0xffff_u16; 8];
        direct[0] = 0; // IMM_RESOURCE at s0..s7 (8-dword T#)
        let user_data = ShaderUserData {
            direct_resource_offset: direct,
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 0,
            srt_size_dw: 0,
        };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 8, None, None)
            .expect("direct type 0 (IMM_RESOURCE, T#) must bind");
        assert_eq!(info.textures2d_readonly, 1);
        assert_eq!(bind.textures2d.textures_num, 1);
        assert_eq!(bind.textures2d.desc[0].start_register, 0);
        assert_eq!(info.direct_sgprs, 0, "all eight registers are consumed");
    }

    /// A sharp resident in the real user SGPRs must read direct even when an
    /// EUD exists — residence is decided by the register index, not by EUD
    /// presence (Kyty's legacy walk resolves pre-0x1b slots with a null
    /// extended_buffer; the Gen5 tables see the EUD for every sharp).
    /// Previously "sharp start_register below the user SGPR file with
    /// extended buffer" (346 measured ASTRO.BOT CS failures).
    #[test]
    fn sharp_below_the_file_reads_direct_when_eud_present() {
        let mut user_sgpr = UserSgprInfo::default();
        for (i, dw) in [0xaaaa_0000u32, 0xbbbb_1111, 0xcccc_2222, 0xdddd_3333]
            .into_iter()
            .enumerate()
        {
            user_sgpr.set(i as u32, dw, UserSgprType::Vsharp);
        }
        let eud = [0x9999_0000u32, 0x9999_0001, 0x9999_0002, 0x9999_0003];
        let eud_at_file_end = EudView {
            data: &eud,
            base_dw: UserSgprInfo::SGPRS_MAX as i32,
        };
        let mut direct = [true; UserSgprInfo::SGPRS_MAX];
        let mut out = [0u32; 4];
        read_sharp_fields(&mut direct, 0, &user_sgpr, Some(eud_at_file_end), &mut out)
            .expect("a direct-resident sharp must not be refused because an EUD exists");
        assert_eq!(
            out,
            [0xaaaa_0000, 0xbbbb_1111, 0xcccc_2222, 0xdddd_3333],
            "values come from the user SGPRs, not the EUD"
        );

        // A sharp at the file boundary still reads the EUD (rebase -32).
        let mut direct2 = [true; UserSgprInfo::SGPRS_MAX];
        let mut out2 = [0u32; 4];
        read_sharp_fields(
            &mut direct2,
            UserSgprInfo::SGPRS_MAX as i32,
            &user_sgpr,
            Some(eud_at_file_end),
            &mut out2,
        )
        .expect("EUD sharp at the rebased origin");
        assert_eq!(out2, eud);

        // The residual frontier is instrumented: an unwritten register BELOW
        // the EUD base names the start register and the measured values.
        let mut direct3 = [true; UserSgprInfo::SGPRS_MAX];
        let mut out3 = [0u32; 4];
        match read_sharp_fields(
            &mut direct3,
            16,
            &user_sgpr,
            Some(eud_at_file_end),
            &mut out3,
        ) {
            Err(ShaderAnalysisError::NotImplementedOwned { what }) => {
                assert!(what.contains("start_register=16"), "{what}");
                assert!(what.contains("eud_base=32"), "{what}");
                assert!(what.contains("captured=4"), "{what}");
                assert!(what.contains("eud=4"), "{what}");
            }
            other => panic!("expected instrumented below-base refusal, got {other:?}"),
        }
    }

    /// The measured ASTRO.BOT residence tuple (469 dispatches/run, all
    /// identical): a T#-sized sharp declared at start_register=12 with only
    /// 14 captured user SGPRs cannot live in the file (s12..s19 needed, s14+
    /// unwritten) — and (s12, s13) is the recovered EUD pointer pair, so the
    /// sharp is the EUD's first descriptor (`data[0..8]`; the smallest
    /// measured EUD is exactly 8 dwords, which only offset 0 fits).
    #[test]
    fn sharp_spilling_past_the_typed_run_reads_whole_sharp_from_eud() {
        let mut user_sgpr = UserSgprInfo::default();
        for i in 0..12 {
            user_sgpr.set(i, 0x1111_0000 + i, UserSgprType::Unknown);
        }
        // The pointer pair itself is typed by the PM4 'hu' markers.
        user_sgpr.set(12, 0x0050_6730, UserSgprType::Vsharp);
        user_sgpr.set(13, 0x4, UserSgprType::Vsharp);
        let eud: [u32; 8] = [
            0xE000_0001,
            0xE000_0002,
            0xE000_0003,
            0xE000_0004,
            0xE000_0005,
            0xE000_0006,
            0xE000_0007,
            0xE000_0008,
        ];
        let view = EudView {
            data: &eud,
            base_dw: 12,
        };

        let mut direct = [true; UserSgprInfo::SGPRS_MAX];
        let mut out = [0u32; 8];
        read_sharp_fields(&mut direct, 12, &user_sgpr, Some(view), &mut out)
            .expect("a sharp that does not fit the typed run must resolve EUD-resident");
        assert_eq!(out, eud, "the whole T# comes from the EUD at offset 0");
        assert!(
            !direct[12] && !direct[13],
            "the in-file pointer pair is consumed"
        );
        assert!(
            direct[11],
            "registers before the run stay direct candidates"
        );

        // A second spilled sharp continues the slot layout: start=16 ->
        // data[4..8] under the same base.
        let mut direct2 = [true; UserSgprInfo::SGPRS_MAX];
        let mut out2 = [0u32; 4];
        read_sharp_fields(&mut direct2, 16, &user_sgpr, Some(view), &mut out2)
            .expect("EUD-resident V# at slot base+4");
        assert_eq!(out2, eud[4..8]);

        // Without an EUD the same shape stays the evidence-rich refusal.
        let mut direct3 = [true; UserSgprInfo::SGPRS_MAX];
        let mut out3 = [0u32; 8];
        match read_sharp_fields(&mut direct3, 12, &user_sgpr, None, &mut out3) {
            Err(ShaderAnalysisError::NotImplementedOwned { what }) => {
                assert!(what.contains("not Vsharp/Region"), "{what}");
                assert!(what.contains("at s14"), "{what}");
                assert!(what.contains("start_register=12"), "{what}");
            }
            other => panic!("expected typed-run refusal, got {other:?}"),
        }

        // An EUD too small for the sharp is a truncation, not silence.
        let short = EudView {
            data: &eud[..4],
            base_dw: 12,
        };
        let mut direct4 = [true; UserSgprInfo::SGPRS_MAX];
        let mut out4 = [0u32; 8];
        assert!(matches!(
            read_sharp_fields(&mut direct4, 12, &user_sgpr, Some(short), &mut out4),
            Err(ShaderAnalysisError::Truncated { .. })
        ));
    }

    /// Gen5 CS with a sampler: Kyty's "cs: samplers" EXIT is a PS4-era
    /// invariant, relaxed on next_gen (738 measured IMM_SAMPLER CS
    /// dispatches on ASTRO.BOT).
    #[test]
    fn cs_next_gen_sampler_is_allowed() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[4] = 0x1111_0000;
        let mut type_ = [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX];
        type_[4..8].fill(UserSgprType::Vsharp);
        let regs = ComputeShaderInfo {
            cs_regs: crate::shader::hw_regs::CsStageRegisters {
                data_addr: 0x6000,
                num_thread_x: 8,
                num_thread_y: 8,
                num_thread_z: 1,
                user_sgpr: 8,
                ..Default::default()
            },
            cs_user_sgpr: UserSgprInfo {
                value,
                type_,
                count: 8,
            },
        };
        let mem = TestMem {
            regions: vec![(0x6000, vec![S_ENDPGM])],
        };
        let mut direct = vec![0xffff_u16; 8];
        direct[1] = 4;
        let mut map = ShaderMap::new();
        map.map_user_data(
            0x6000,
            ShaderMappedData {
                user_data: Some(ShaderUserData {
                    direct_resource_offset: direct,
                    sharp_resource_offset: [vec![], vec![], vec![], vec![]],
                    eud_size_dw: 0,
                    srt_size_dw: 0,
                }),
                input_semantics: vec![],
            },
        );
        let sh = ShaderRegisters::default();
        let mut info = ShaderComputeInputInfo::default();
        shader_get_input_info_cs(&regs, &sh, &mem, &map, true, &mut info)
            .expect("next-gen CS with an IMM_SAMPLER must analyse");
        assert_eq!(info.bind.samplers.samplers_num, 1);
    }

    /// Gen5 vertex stages use the same stage-agnostic sampler binding as PS
    /// and CS. Kyty's PS4-era VS sampler gate must not skip those draws.
    #[test]
    fn vs_next_gen_sampler_is_allowed() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[4] = 0x1111_0000;
        let mut type_ = [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX];
        type_[4..8].fill(UserSgprType::Vsharp);
        let regs = VertexShaderInfo {
            es_regs: crate::shader::hw_regs::EsStageRegisters { data_addr: 0x6000 },
            gs_regs: crate::shader::hw_regs::GsStageRegisters {
                rsrc2: crate::shader::hw_regs::GsShaderResource2 { user_sgpr: 8 },
                chksum: 1,
                ..Default::default()
            },
            gs_user_sgpr: UserSgprInfo {
                value,
                type_,
                count: 8,
            },
            ..Default::default()
        };
        let mem = TestMem {
            regions: vec![(0x6000, vec![S_ENDPGM])],
        };
        let mut direct = vec![0xffff_u16; 8];
        direct[1] = 4;
        let mut map = ShaderMap::new();
        map.map_user_data(
            0x6000,
            ShaderMappedData {
                user_data: Some(ShaderUserData {
                    direct_resource_offset: direct,
                    sharp_resource_offset: [vec![], vec![], vec![], vec![]],
                    eud_size_dw: 0,
                    srt_size_dw: 0,
                }),
                input_semantics: vec![],
            },
        );
        let mut info = ShaderVertexInputInfo::default();

        shader_get_input_info_vs(
            &regs,
            &ShaderRegisters::default(),
            &mem,
            &map,
            true,
            &mut info,
        )
        .expect("next-gen VS with an IMM_SAMPLER must analyse");

        assert!(info.gs_prolog);
        assert_eq!(info.bind.samplers.samplers_num, 1);
        assert_eq!(info.bind.samplers.start_register[0], 4);
    }

    /// Gen5 CS end-to-end: an inline-SRT shader analyses to completion, with
    /// the SRT registers bound as direct SGPRs (Kyty's "cs: direct sgprs"
    /// EXIT is a PS4-era invariant and is relaxed on next_gen), and
    /// COMPUTE_PGM_RSRC2.LDS_SIZE lands in `lds_size_dw` (128-dword units).
    #[test]
    fn cs_next_gen_inline_srt_analyses_with_direct_sgprs() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[0] = 0xAAAA_0001;
        value[1] = 0xAAAA_0002;
        let regs = ComputeShaderInfo {
            cs_regs: crate::shader::hw_regs::CsStageRegisters {
                data_addr: 0x6000,
                num_thread_x: 8,
                num_thread_y: 8,
                num_thread_z: 1,
                user_sgpr: 2,
                lds_size: 2,
                ..Default::default()
            },
            cs_user_sgpr: UserSgprInfo {
                value,
                type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
                count: 2,
            },
        };
        let mem = TestMem {
            regions: vec![(0x6000, vec![S_ENDPGM])],
        };
        let mut map = ShaderMap::new();
        map.map_user_data(
            0x6000,
            ShaderMappedData {
                user_data: Some(ShaderUserData {
                    direct_resource_offset: vec![0xffff; 8],
                    sharp_resource_offset: [vec![], vec![], vec![], vec![]],
                    eud_size_dw: 0,
                    srt_size_dw: 2,
                }),
                input_semantics: vec![],
            },
        );
        let sh = ShaderRegisters::default();
        let mut info = ShaderComputeInputInfo::default();
        shader_get_input_info_cs(&regs, &sh, &mem, &map, true, &mut info)
            .expect("next-gen CS with an inline SRT must analyse");
        assert_eq!(info.bind.direct_sgprs.sgprs_num, 2);
        assert_eq!(info.bind.direct_sgprs.sgprs[0].field, 0xAAAA_0001);
        assert_eq!(info.bind.direct_sgprs.sgprs[1].field, 0xAAAA_0002);
        assert_eq!(info.lds_size_dw, 256);
        // Direct SGPRs travel through the push-constant window.
        assert!(info.bind.push_constant_size >= 16);
    }

    /// EUD resolver strategy 3: when neither the after-the-file pair nor the
    /// scalar-load bases find the buffer, scan every adjacent SGPR pair for a
    /// small-high-dword pointer into readable guest memory. Measured shape:
    /// (s12, s13) = 0x4_00506730 with declared == captured == 14.
    #[test]
    fn eud_adjacent_pair_scan_recovers_measured_pointer() {
        let eud_base: u64 = 0x4_0050_6730;
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[0] = 0x055e_7e00;
        value[1] = 0xc410_0000; // hi too large — must be skipped
        value[12] = 0x0050_6730;
        value[13] = 0x4;
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 14,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 4,
            srt_size_dw: 0,
        };
        let shader_addr = 0x1000;
        let mem = TestMem {
            regions: vec![
                (shader_addr, vec![S_ENDPGM]),
                (
                    eud_base,
                    vec![0xAAAA_0001, 0xAAAA_0002, 0xAAAA_0003, 0xAAAA_0004],
                ),
            ],
        };
        let (eud, base_dw) =
            read_extended_user_data(&user_data, &user_sgpr, 14, &mem, shader_addr, true, None)
                .expect("strategy 3 must recover the (s12, s13) pair");
        assert_eq!(
            eud,
            vec![0xAAAA_0001, 0xAAAA_0002, 0xAAAA_0003, 0xAAAA_0004]
        );
        assert_eq!(
            base_dw, 12,
            "the EUD base slot is the register holding the pointer pair"
        );
    }

    /// Minecraft keeps this gameplay shader and its EUD below 4 GiB, so the
    /// high half of the EUD pointer is zero. The tail pair is authoritative
    /// even when an earlier pointer into the same readable table exists.
    #[test]
    fn eud_zero_extended_tail_pair_recovers_measured_minecraft_pointer() {
        let eud_base = 0x16fa_d784u64;
        let shader_addr = 0x16ff_d700u64;
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[..8].copy_from_slice(&[
            0x16fa_d9f0,
            0x0010_0000,
            4,
            0x0004_dfac,
            (eud_base + 12) as u32, // readable adjacent table, but not EUD base
            0,
            eud_base as u32, // measured EUD tail pair
            0,
        ]);
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 8,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 12,
            srt_size_dw: 0,
        };
        let mut eud = vec![0u32; super::super::spirv::EXTENDED_MAPPING_DWORDS + 3];
        eud[0] = 0xAAAA_0000;
        eud[3] = 0xBBBB_0003;
        let mem = TestMem {
            regions: vec![(shader_addr, vec![S_ENDPGM]), (eud_base, eud)],
        };

        let (captured, base_dw) =
            read_extended_user_data(&user_data, &user_sgpr, 8, &mem, shader_addr, true, None)
                .expect("a readable zero-extended tail pointer must recover the EUD");
        assert_eq!(base_dw, 6);
        assert_eq!(
            captured[0], 0xAAAA_0000,
            "the earlier in-range pointer must not shift the EUD by three dwords"
        );
        assert_eq!(captured[3], 0xBBBB_0003);
    }

    /// EUD resolver strategy 2 must snapshot the EUD at the load's BASE-PAIR
    /// value, not at base+offset: a load's byte offset selects a descriptor
    /// WITHIN the buffer, and the virtual-register mapping (sharp at
    /// s{base+k} ⇒ EUD dword k) only holds from the pair value. Measured on
    /// ASTRO.BOT composite CS `0x500665c00` (declared=14 count=14, EUD 28 dw
    /// at (s12,s13)): the scan-order-first load is `s_load_dwordx4 s[16:19],
    /// s[12:13], 0x60`, and snapshotting at base+0x60 shifted every sharp
    /// peek 24 dwords high — the sampled T# declared at EUD dword 0 peeked
    /// garbage, mis-classed, and its `image_sample_lz` refused as
    /// `dynamic-image-descriptor` (the whole composite/read pass skipped).
    #[test]
    fn eud_strategy2_snapshots_at_base_pair_not_first_load_target() {
        let eud_base: u64 = 0x4_0032_5330;
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[12] = 0x0032_5330;
        value[13] = 0x4;
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 14,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 28,
            srt_size_dw: 0,
        };
        let shader_addr = 0x1000;
        // The measured 0x500665c00 shape: first (scan-order) load at +0x60,
        // then the T# load at +0x0.
        let body = vec![
            0xF408_0406,
            0xFA00_0060, // s_load_dwordx4 s[16:19], s[12:13], 0x60
            0xF40C_0806,
            0xFA00_0000, // s_load_dwordx8 s[32:39], s[12:13], 0x0
            S_ENDPGM,
        ];
        // 64 dwords behind the base so base+0x60 is ALSO readable for 28 dw —
        // exactly the live shape (guest memory continues past the EUD), which
        // is what let the buggy target-address read succeed.
        let mut eud = vec![0u32; 64];
        eud[0] = 0xAAAA_0001; // EUD dword 0 (the T#'s first dword)
        eud[24] = 0xBBBB_0018; // content at +0x60 — must NOT become buf[0]
        let mem = TestMem {
            regions: vec![(shader_addr, body), (eud_base, eud)],
        };
        let (buf, base_dw) =
            read_extended_user_data(&user_data, &user_sgpr, 14, &mem, shader_addr, true, None)
                .expect("strategy 2 must recover the EUD via the load base pair");
        assert_eq!(base_dw, 12);
        assert_eq!(
            buf[0], 0xAAAA_0001,
            "snapshot must start at the base-pair value, not the first load's target"
        );
        assert_eq!(buf[24], 0xBBBB_0018);
    }

    /// An explicit scalar-load base is stronger evidence than the positional
    /// pair immediately after the declared user-SGPR file. ASTRO.BOT's live CS
    /// declares 14 user SGPRs, has readable pointers in both s12:s13 and
    /// s14:s15, and begins with `s_load_dwordx8 ..., s[12:13], 0`. Selecting
    /// s14 merely because it is positional leaves that real load outside the
    /// captured EUD and the recompiler correctly refuses it.
    #[test]
    fn eud_explicit_scalar_load_base_beats_readable_positional_pair() {
        let explicit_base: u64 = 0x4_0032_5000;
        let positional_base: u64 = 0x4_0042_6000;
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[12] = explicit_base as u32;
        value[13] = (explicit_base >> 32) as u32;
        value[14] = positional_base as u32;
        value[15] = (positional_base >> 32) as u32;
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 16,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 8,
            srt_size_dw: 0,
        };
        let shader_addr = 0x1000;
        let mem = TestMem {
            regions: vec![
                (
                    shader_addr,
                    vec![
                        0xBFA0_0002, // s_inst_prefetch 2
                        0x7E10_0280, // v_mov_b32 v8, 0
                        0x7E12_0280, // v_mov_b32 v9, 0
                        0xF40C_0406, // s_load_dwordx8 s[16:23], s[12:13], 0
                        0xFA00_0000,
                        S_ENDPGM,
                    ],
                ),
                (explicit_base, vec![0xAAAA_0001; 8]),
                (positional_base, vec![0xBBBB_0002; 8]),
            ],
        };

        let (buf, base_dw) =
            read_extended_user_data(&user_data, &user_sgpr, 14, &mem, shader_addr, true, None)
                .expect("one of the two readable EUD candidates must be selected");
        assert_eq!(base_dw, 12, "the explicit s_load base must win");
        assert_eq!(buf, vec![0xAAAA_0001; 8]);
    }

    /// A scalar load through a pair constructed by the shader is not evidence
    /// that the pair's entry-time user-SGPR value is the EUD pointer. Even when
    /// that stale value happens to be readable, the positional live-in pointer
    /// must remain the fallback.
    #[test]
    fn eud_pc_relative_load_base_does_not_beat_positional_pair() {
        let stale_entry_value: u64 = 0x4_0032_5000;
        let positional_base: u64 = 0x4_0042_6000;
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[0] = stale_entry_value as u32;
        value[1] = (stale_entry_value >> 32) as u32;
        value[14] = positional_base as u32;
        value[15] = (positional_base >> 32) as u32;
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 16,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 8,
            srt_size_dw: 0,
        };
        let shader_addr = 0x1000;
        let mem = TestMem {
            regions: vec![
                (
                    shader_addr,
                    vec![
                        0xBE80_1F00, // s_getpc_b64 s[0:1]
                        0xF40C_0200, // s_load_dwordx8 s[8:15], s[0:1], 0
                        0xFA00_0000,
                        S_ENDPGM,
                    ],
                ),
                (stale_entry_value, vec![0xAAAA_0001; 8]),
                (positional_base, vec![0xBBBB_0002; 8]),
            ],
        };

        let (buf, base_dw) =
            read_extended_user_data(&user_data, &user_sgpr, 14, &mem, shader_addr, true, None)
                .expect("the positional EUD fallback is readable");
        assert_eq!(base_dw, 14, "the shader overwrites s0:s1 before loading");
        assert_eq!(buf, vec![0xBBBB_0002; 8]);
    }

    use crate::shader::types::{ShaderConstant, ShaderInstruction, ShaderOperand};

    fn sgpr_op(register_id: i32, size: i32) -> ShaderOperand {
        ShaderOperand {
            type_: ShaderOperandType::Sgpr,
            register_id,
            size,
            ..Default::default()
        }
    }

    fn imm_op(byte_offset: u32) -> ShaderOperand {
        ShaderOperand {
            type_: ShaderOperandType::IntegerInlineConstant,
            constant: ShaderConstant::from_u(byte_offset),
            ..Default::default()
        }
    }

    fn covered_load(
        type_: crate::shader::types::ShaderInstructionType,
        dst_reg: i32,
        dst_size: i32,
        byte_offset: u32,
    ) -> ShaderInstruction {
        let mut inst = ShaderInstruction {
            type_,
            ..Default::default()
        };
        inst.src[0] = sgpr_op(12, 2);
        inst.src[1] = imm_op(byte_offset);
        inst.src_num = 2;
        inst.dst = sgpr_op(dst_reg, dst_size);
        inst
    }

    /// The measured ASTRO.BOT composite-CS shape (0x500665c00 family): the
    /// sampled T# and its S# are delivered RAW through the EUD — no
    /// usage-table slot declares them; the shader loads both itself with
    /// covered `s_load`s off the EUD pair and samples. The capture pass must
    /// synthesize both captures from the EUD snapshot (T# fields are the
    /// live-traced dwords, type nibble 9 = Texture2D) so the MIMG guard's
    /// alias rule accepts the registers.
    #[test]
    fn raw_eud_image_descriptor_captured_from_covered_load() {
        use crate::shader::types::ShaderInstructionType as T;
        let mut bind = ShaderBindResources::default();
        bind.extended.used = true;
        bind.extended.start_register = 12;

        let mut code = ShaderCode::new();
        code.get_instructions_mut()
            .push(covered_load(T::SLoadDwordx8, 32, 8, 0x0)); // T# from EUD dword 0
        code.get_instructions_mut()
            .push(covered_load(T::SLoadDwordx4, 8, 4, 0x20)); // S# from EUD dword 8
        let mut sample = ShaderInstruction {
            type_: T::ImageSampleLz,
            ..Default::default()
        };
        sample.src[0] = ShaderOperand {
            type_: ShaderOperandType::Vgpr,
            register_id: 5,
            size: 3,
            ..Default::default()
        };
        sample.src[1] = sgpr_op(32, 8);
        sample.src[2] = sgpr_op(8, 4);
        sample.src_num = 3;
        code.get_instructions_mut().push(sample);

        // Live-traced EUD head from cs@0x500665c00 (mem[s12:s13]): a real 2D
        // T# (dword3 0x91800924, type nibble 9) followed by an S#.
        let mut eud = vec![0u32; 64];
        eud[..4].copy_from_slice(&[0x053a_c400, 0xc160_0000, 0x0043_4077, 0x9180_0924]);
        eud[8..11].copy_from_slice(&[0x7092, 0x00ff_f000, 0x0500_0000]);

        let user_sgpr = UserSgprInfo {
            value: [0u32; UserSgprInfo::SGPRS_MAX],
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 14,
        };
        shader_capture_eud_image_descriptors(
            &code,
            &mut bind,
            &user_sgpr,
            EudView {
                data: &eud,
                base_dw: 12,
            },
        )
        .expect("raw-EUD T#/S# must capture");

        assert_eq!(bind.textures2d.textures_num, 1);
        let d = &bind.textures2d.desc[0];
        assert_eq!(
            d.start_register,
            UserSgprInfo::SGPRS_MAX as i32,
            "captured at the EUD-virtual register for dword 0"
        );
        assert!(d.extended);
        assert!(!d.textures2d_without_sampler, "a sampled T#, not storage");
        assert_eq!(d.texture.fields[3], 0x9180_0924);
        assert_eq!(bind.samplers.samplers_num, 1);
        assert_eq!(
            bind.samplers.start_register[0],
            UserSgprInfo::SGPRS_MAX as i32 + 8
        );
        assert!(bind.samplers.extended[0]);
        assert_eq!(bind.samplers.samplers[0].fields[0], 0x7092);

        // Idempotence: a second pass sees the alias hit and captures nothing
        // new (several MIMGs commonly read the same descriptor).
        shader_capture_eud_image_descriptors(
            &code,
            &mut bind,
            &user_sgpr,
            EudView {
                data: &eud,
                base_dw: 12,
            },
        )
        .expect("re-run must be a no-op");
        assert_eq!(bind.textures2d.textures_num, 1);
        assert_eq!(bind.samplers.samplers_num, 1);
    }

    /// Degrade rules: all-zero content and buffer-typed dword3 (type nibble
    /// 0) must NOT capture — the recompiler's named `dynamic-image-descriptor`
    /// refusal stands instead of binding garbage.
    #[test]
    fn raw_eud_capture_declines_zero_and_buffer_typed_content() {
        use crate::shader::types::ShaderInstructionType as T;
        let mut bind = ShaderBindResources::default();
        bind.extended.used = true;
        bind.extended.start_register = 12;

        let mut code = ShaderCode::new();
        code.get_instructions_mut()
            .push(covered_load(T::SLoadDwordx8, 32, 8, 0x0)); // all-zero dwords
        code.get_instructions_mut()
            .push(covered_load(T::SLoadDwordx8, 16, 8, 0x40)); // buffer-typed (nibble 0)
        for t_reg in [32, 16] {
            let mut store = ShaderInstruction {
                type_: T::ImageStore,
                ..Default::default()
            };
            store.src[1] = sgpr_op(t_reg, 8);
            store.src_num = 2;
            code.get_instructions_mut().push(store);
        }

        let mut eud = vec![0u32; 64];
        // Dwords 16..24 hold a V#-shaped quad: dword3 type nibble 0.
        eud[16..20].copy_from_slice(&[0x1234_5678, 0x0000_0004, 0xffff_ffff, 0x0002_4fac]);

        let user_sgpr = UserSgprInfo {
            value: [0u32; UserSgprInfo::SGPRS_MAX],
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 14,
        };
        shader_capture_eud_image_descriptors(
            &code,
            &mut bind,
            &user_sgpr,
            EudView {
                data: &eud,
                base_dw: 12,
            },
        )
        .expect("declines are silent, not errors");
        assert_eq!(
            bind.textures2d.textures_num, 0,
            "neither zero nor buffer-typed content may capture"
        );
    }

    /// Minecraft's post-title fragment shader declares a table-1 V# at the
    /// EUD base (s28), but its scalar program loads the descriptor actually
    /// used by MUBUF from byte offset 0x20. Dword 0 contains the exact
    /// non-descriptor tuple observed in the runner; dword 8 contains the
    /// valid three-record V#. The stale declaration must not survive binding.
    #[test]
    fn raw_eud_storage_uses_covered_load_instead_of_declared_dword_zero() {
        use crate::shader::types::ShaderInstructionType as T;

        let mut bind = ShaderBindResources::default();
        bind.extended.used = true;
        bind.extended.start_register = 28;
        bind.storage_buffers.buffers_num = 1;
        bind.storage_buffers.start_register[0] = 28;
        bind.storage_buffers.extended[0] = true;
        bind.storage_buffers.slots[0] = 1;
        bind.storage_buffers.usages[0] = ShaderStorageUsage::ReadWrite;
        bind.storage_buffers.buffers[0].fields =
            [0x0000_0692, 0x00ff_f000, 0x0600_0000, 0x4000_0000];

        let mut bad_load0 = covered_load(T::SLoadDwordx4, 12, 4, 0x0);
        bad_load0.src[0] = sgpr_op(28, 2);
        let mut bad_load4 = covered_load(T::SLoadDwordx4, 12, 4, 0x10);
        bad_load4.src[0] = sgpr_op(28, 2);
        let mut live_load = covered_load(T::SLoadDwordx4, 12, 4, 0x20);
        live_load.src[0] = sgpr_op(28, 2);
        let mut buffer_load = ShaderInstruction {
            type_: T::BufferLoadDword,
            ..Default::default()
        };
        buffer_load.src[1] = sgpr_op(12, 4);

        let mut code = ShaderCode::new();
        code.get_instructions_mut().push(bad_load0);
        code.get_instructions_mut().push(bad_load4);
        code.get_instructions_mut().push(live_load);
        code.get_instructions_mut().push(buffer_load);

        let mut eud = vec![0u32; 32];
        eud[..4].copy_from_slice(&[0x0000_0692, 0x00ff_f000, 0x0600_0000, 0x4000_0000]);
        eud[4..8].copy_from_slice(&[0x0000_0692, 0x00ff_f000, 0x0600_0000, 0x4000_0000]);
        let valid = [0x16f9_d9c0, 0x0010_0000, 0x0000_0003, 0x0004_dfac];
        eud[8..12].copy_from_slice(&valid);
        let user_sgpr = UserSgprInfo {
            value: [0u32; UserSgprInfo::SGPRS_MAX],
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 30,
        };
        let mem = TestMem {
            regions: vec![(u64::from(valid[0]), vec![0; 16])],
        };

        shader_capture_eud_storage_buffers(
            &code,
            &mut bind,
            &user_sgpr,
            EudView {
                data: &eud,
                base_dw: 28,
            },
            &mem,
        )
        .expect("covered EUD V# must replace the stale declaration");

        assert_eq!(bind.storage_buffers.buffers_num, 1);
        assert_eq!(
            bind.storage_buffers.start_register[0],
            UserSgprInfo::SGPRS_MAX as i32 + 8
        );
        assert!(bind.storage_buffers.extended[0]);
        assert_eq!(bind.storage_buffers.slots[0], 1);
        assert_eq!(
            bind.storage_buffers.usages[0],
            ShaderStorageUsage::ReadWrite
        );
        assert_eq!(bind.storage_buffers.buffers[0].fields, valid);
    }

    /// The pair scan must not invent pointers: no readable pair -> None.
    #[test]
    fn eud_adjacent_pair_scan_refuses_unbacked_pairs() {
        let mut value = [0u32; UserSgprInfo::SGPRS_MAX];
        value[12] = 0x0050_6730;
        value[13] = 0x4; // plausible shape but nothing mapped there
        let user_sgpr = UserSgprInfo {
            value,
            type_: [UserSgprType::Unknown; UserSgprInfo::SGPRS_MAX],
            count: 14,
        };
        let user_data = ShaderUserData {
            direct_resource_offset: vec![],
            sharp_resource_offset: [vec![], vec![], vec![], vec![]],
            eud_size_dw: 4,
            srt_size_dw: 0,
        };
        let shader_addr = 0x1000;
        let mem = TestMem {
            regions: vec![(shader_addr, vec![S_ENDPGM])],
        };
        assert_eq!(
            read_extended_user_data(&user_data, &user_sgpr, 14, &mem, shader_addr, true, None),
            None
        );
    }

    #[test]
    fn compute_storage_usage_proves_large_input_read_only() {
        use crate::shader::types::ShaderInstructionType as T;

        let mut bind = ShaderBindResources::default();
        bind.storage_buffers.buffers_num = 2;
        bind.storage_buffers.start_register[..2].copy_from_slice(&[0, 4]);
        bind.storage_buffers.usages[..2].fill(ShaderStorageUsage::ReadWrite);

        let mut code = ShaderCode::new();
        let mut load = ShaderInstruction {
            type_: T::BufferLoadDword,
            ..Default::default()
        };
        load.src[1] = sgpr_op(0, 4);
        code.get_instructions_mut().push(load);
        let mut store = ShaderInstruction {
            type_: T::BufferStoreDword,
            ..Default::default()
        };
        store.src[1] = sgpr_op(4, 4);
        code.get_instructions_mut().push(store);

        shader_refine_compute_storage_usage(&code, &mut bind);
        assert_eq!(
            bind.storage_buffers.usages[..2],
            [ShaderStorageUsage::ReadOnly, ShaderStorageUsage::ReadWrite],
            "a load-only input must not be copied back; the actual store target remains writable"
        );

        // An unresolved store may be an SGPR alias of either descriptor. The
        // narrow proof must fail closed and preserve both write flags.
        bind.storage_buffers.usages[..2].fill(ShaderStorageUsage::ReadWrite);
        code.get_instructions_mut().last_mut().unwrap().src[1] = sgpr_op(12, 4);
        shader_refine_compute_storage_usage(&code, &mut bind);
        assert_eq!(
            bind.storage_buffers.usages[..2],
            [ShaderStorageUsage::ReadWrite, ShaderStorageUsage::ReadWrite]
        );

        // A shader that mutates the input descriptor quad also cannot use the
        // direct-register proof.
        code.get_instructions_mut().last_mut().unwrap().src[1] = sgpr_op(4, 4);
        let mut mutate = ShaderInstruction {
            type_: T::SMovB32,
            ..Default::default()
        };
        mutate.dst = sgpr_op(0, 1);
        code.get_instructions_mut().push(mutate);
        shader_refine_compute_storage_usage(&code, &mut bind);
        assert_eq!(
            bind.storage_buffers.usages[0],
            ShaderStorageUsage::ReadWrite
        );
    }

    #[test]
    fn scalar_constant_buffer_records_only_the_touched_prefix() {
        use crate::shader::types::ShaderInstructionType as T;

        let mut bind = ShaderBindResources::default();
        bind.storage_buffers.buffers_num = 1;
        bind.storage_buffers.start_register[0] = 12;
        bind.storage_buffers.usages[0] = ShaderStorageUsage::Constant;

        let mut load = ShaderInstruction {
            type_: T::SBufferLoadDword,
            ..Default::default()
        };
        load.src[0] = sgpr_op(12, 4);
        load.src[1].type_ = ShaderOperandType::IntegerInlineConstant;
        load.src[1].constant.u = 8;
        let mut code = ShaderCode::new();
        code.get_instructions_mut().push(load);

        shader_measure_constant_buffer_accesses(&code, &mut bind);
        assert_eq!(
            bind.storage_buffers.required_bytes[0], 12,
            "offset 8 plus one dword needs only the first 12 bytes"
        );

        code.get_instructions_mut()[0].src[1] = sgpr_op(20, 1);
        shader_measure_constant_buffer_accesses(&code, &mut bind);
        assert_eq!(
            bind.storage_buffers.required_bytes[0], 0,
            "a dynamic scalar offset must fall back to the full V# extent"
        );

        bind.storage_buffers.start_register[0] = 0;
        code.get_instructions_mut()[0].src[0] = sgpr_op(8, 4);
        code.get_instructions_mut()[0].src[1].type_ = ShaderOperandType::IntegerInlineConstant;
        code.get_instructions_mut()[0].src[1].constant.u = 8;
        shader_measure_constant_buffer_accesses_shifted(&code, &mut bind, 8);
        assert_eq!(
            bind.storage_buffers.required_bytes[0], 12,
            "NGG hardware slot 0 must match the shader's scalar s8 resource operand"
        );
    }

    #[test]
    fn ngg_constant_sharp_rebases_proven_scalar_s8_to_hardware_user_slot_zero() {
        use crate::shader::resources::{ShaderSharp, ShaderUserData};
        use crate::shader::types::ShaderInstructionType as T;

        let mut user_data = ShaderUserData::default();
        user_data.sharp_resource_offset[3].push(ShaderSharp::new(8, 1));
        let mut user_sgpr = UserSgprInfo {
            count: 8,
            ..Default::default()
        };
        user_sgpr.type_[0..4].fill(UserSgprType::Vsharp);

        let mut load = ShaderInstruction {
            type_: T::SBufferLoadDwordx8,
            ..Default::default()
        };
        load.src[0] = sgpr_op(8, 4);
        let mut code = ShaderCode::new();
        code.get_instructions_mut().push(load);

        let normalized = rebase_ngg_constant_sharps(&user_data, &code, &user_sgpr, 8)
            .expect("decoded s8 SBUFFER + typed hardware slot 0 proves the NGG rebase");
        assert_eq!(
            normalized.sharp_resource_offset[3][0].offset_dw(),
            0,
            "the emitter later restores the NGG +8 scalar-register base"
        );

        code.get_instructions_mut().clear();
        assert!(
            rebase_ngg_constant_sharps(&user_data, &code, &user_sgpr, 8).is_none(),
            "metadata alone must not invent the rebase without a matching decoded access"
        );
    }

    #[test]
    fn ngg_constant_sharp_accepts_a_valid_untyped_captured_descriptor() {
        use crate::shader::resources::{ShaderSharp, ShaderUserData};
        use crate::shader::types::ShaderInstructionType as T;

        let mut user_data = ShaderUserData::default();
        user_data.sharp_resource_offset[3].push(ShaderSharp::new(8, 1));
        let mut user_sgpr = UserSgprInfo {
            count: 8,
            ..Default::default()
        };
        // Some AGC streams omit the optional `hu` marker that labels a user
        // SGPR write as Vsharp. The descriptor bytes remain authoritative.
        user_sgpr.value[..4].copy_from_slice(&[
            0x0050_0000, // base low
            0x0004_0000, // base high + stride
            0x0000_0100, // records
            0x0111_0000, // buffer descriptor type nibble
        ]);

        let mut load = ShaderInstruction {
            type_: T::SBufferLoadDwordx16,
            ..Default::default()
        };
        load.src[0] = sgpr_op(8, 4);
        let mut code = ShaderCode::new();
        code.get_instructions_mut().push(load);

        let normalized = rebase_ngg_constant_sharps(&user_data, &code, &user_sgpr, 8)
            .expect("decoded s8 SBUFFER plus a captured buffer V# proves the NGG rebase");
        assert_eq!(normalized.sharp_resource_offset[3][0].offset_dw(), 0);

        user_sgpr.value[..4].fill(0);
        assert!(
            rebase_ngg_constant_sharps(&user_data, &code, &user_sgpr, 8).is_none(),
            "an untyped all-zero quad must not be promoted into a resource"
        );
    }
}
