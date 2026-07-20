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
    ShaderGdsResources, ShaderId, ShaderMappedData, ShaderPixelInputInfo, ShaderSamplerResources,
    ShaderSemantic, ShaderStorageResources, ShaderStorageUsage, ShaderTextureResources,
    ShaderTextureUsage, ShaderUserData, ShaderVertexInputBuffer, ShaderVertexInputInfo,
};
use super::types::{ShaderCode, ShaderInstructionType, ShaderOperandType, ShaderType};

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

    /// Kyty: `ShaderMapUserData` (L110). Like `unordered_map::insert`, an
    /// existing entry for `addr` is kept.
    pub fn map_user_data(&mut self, addr: u64, data: ShaderMappedData) {
        self.map.entry(addr).or_insert(data);
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

/// Shared body of `ShaderGetStorageBuffer`/`ShaderGetTextureBuffer`/
/// `ShaderGetSampler` (Shader.cpp L1141/L1179/L1233): validate
/// `start_index` against the extended (EUD) mode, require
/// `Vsharp`/`Region`-typed user SGPRs for the direct case, mark them
/// consumed, and copy the descriptor dwords.
fn read_sharp_fields(
    direct_sgprs: &mut [bool; UserSgprInfo::SGPRS_MAX],
    start_index: i32,
    user_sgpr: &UserSgprInfo,
    extended_buffer: Option<&[u32]>,
    out: &mut [u32],
) -> Result<(), ShaderAnalysisError> {
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
    if extended && (start_index as usize) < UserSgprInfo::SGPRS_MAX {
        return Err(ni(
            "sharp start_register below the user SGPR file with extended buffer",
        ));
    }
    let start = usize::try_from(start_index).map_err(|_| trunc("negative sharp start register"))?;

    match extended_buffer {
        None => {
            for (j, dw) in out.iter_mut().enumerate() {
                let idx = start + j;
                let type_ = *user_sgpr
                    .type_
                    .get(idx)
                    .ok_or_else(|| trunc("sharp registers beyond SGPRS_MAX"))?;
                if type_ != UserSgprType::Vsharp && type_ != UserSgprType::Region {
                    return Err(ni("user sgpr type is not Vsharp/Region"));
                }
                direct_sgprs[idx] = false;
                *dw = user_sgpr.value[idx];
            }
        }
        Some(ext) => {
            // The EUD is addressed as a continuation of the user-SGPR file, so
            // the rebase is the file SIZE, not a literal 16 (Kyty's 16 was the
            // PS4 user-SGPR count). Measured on ASTRO.BOT: a shader with
            // eud_size_dw=8 places a sharp at offset_dw=32 while its other
            // sharps sit direct at s0/s8 — under a `-16` rebase that reads
            // ext[16], past the end of an 8-dword EUD; under `-32` it reads
            // ext[0], the buffer's first descriptor, which fits.
            for (j, dw) in out.iter_mut().enumerate() {
                *dw = *ext
                    .get(start - UserSgprInfo::SGPRS_MAX + j)
                    .ok_or_else(|| trunc("extended (EUD) buffer too small"))?;
            }
        }
    }
    Ok(())
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

/// Kyty: Shader.cpp `ShaderGetStorageBuffer` (L1141).
pub fn shader_get_storage_buffer(
    info: &mut ShaderStorageResources,
    direct_sgprs: &mut [bool; UserSgprInfo::SGPRS_MAX],
    start_index: i32,
    slot: i32,
    usage: ShaderStorageUsage,
    user_sgpr: &UserSgprInfo,
    extended_buffer: Option<&[u32]>,
) -> Result<(), ShaderAnalysisError> {
    if info.buffers_num < 0 || info.buffers_num as usize >= ShaderStorageResources::BUFFERS_MAX {
        return Err(ni("too many storage buffers"));
    }
    let index = info.buffers_num as usize;

    let mut fields = [0u32; 4];
    read_sharp_fields(
        direct_sgprs,
        start_index,
        user_sgpr,
        extended_buffer,
        &mut fields,
    )?;

    info.start_register[index] = start_index;
    info.slots[index] = slot;
    info.usages[index] = usage;
    info.extended[index] = extended_buffer.is_some();
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
    extended_buffer: Option<&[u32]>,
) -> Result<(), ShaderAnalysisError> {
    if info.textures_num < 0 || info.textures_num as usize >= ShaderTextureResources::RES_MAX {
        return Err(ni("too many textures"));
    }
    if usage == ShaderTextureUsage::Unknown {
        return Err(ni("texture usage is Unknown"));
    }
    let index = info.textures_num as usize;

    let mut fields = [0u32; 8];
    read_sharp_fields(
        direct_sgprs,
        start_index,
        user_sgpr,
        extended_buffer,
        &mut fields,
    )?;

    info.desc[index].start_register = start_index;
    info.desc[index].extended = extended_buffer.is_some();
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
    extended_buffer: Option<&[u32]>,
) -> Result<(), ShaderAnalysisError> {
    if info.samplers_num < 0 || info.samplers_num as usize >= ShaderSamplerResources::RES_MAX {
        return Err(ni("too many samplers"));
    }
    let index = info.samplers_num as usize;

    let mut fields = [0u32; 4];
    read_sharp_fields(
        direct_sgprs,
        start_index,
        user_sgpr,
        extended_buffer,
        &mut fields,
    )?;

    info.start_register[index] = start_index;
    info.extended[index] = extended_buffer.is_some();
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
    extended_buffer: Option<&[u32]>,
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
            .get(start - 16)
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

/// Kyty: Shader.cpp `ShaderGetDirectSgpr` (L1301).
pub fn shader_get_direct_sgpr(
    info: &mut ShaderDirectSgprsResources,
    start_index: i32,
    user_sgpr: &UserSgprInfo,
) -> Result<(), ShaderAnalysisError> {
    if info.sgprs_num < 0 || info.sgprs_num as usize >= ShaderDirectSgprsResources::SGPRS_MAX {
        return Err(ni("too many direct sgprs"));
    }
    let index = info.sgprs_num as usize;

    if start_index >= 16 {
        return Err(ni("direct sgpr start_register >= 16"));
    }
    let start = usize::try_from(start_index).map_err(|_| trunc("negative direct sgpr register"))?;

    info.start_register[index] = start_index;

    if user_sgpr.type_[start] != UserSgprType::Unknown {
        return Err(ni("direct user sgpr type is not Unknown"));
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
        bind.textures2d.binding_sampled_index = binding_index;
        bind.textures2d.binding_storage_index = binding_index + 1;
        binding_index += 2;

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

    let mut extended_buffer: Option<Cow<'_, [u32]>> = None;

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
                        extended_buffer.as_deref(),
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
                        extended_buffer.as_deref(),
                    )?;
                    info.textures2d_readonly += 1;
                    let last = (bind.textures2d.textures_num - 1) as usize;
                    let ty = bind.textures2d.desc[last].texture.type_();
                    // 9 = Texture2D, 11 = Cube (measured: Minecraft's
                    // 1024x1024x6 skybox). Everything else stays named.
                    if ty != 9 && ty != 11 {
                        return Err(ni_owned(format!(
                            "read-only texture type {ty} is not Texture2D (9) \
                             (base={:#x} {}x{} format={} tile={})",
                            bind.textures2d.desc[last].texture.base40(),
                            u32::from(bind.textures2d.desc[last].texture.width5()) + 1,
                            u32::from(bind.textures2d.desc[last].texture.height5()) + 1,
                            bind.textures2d.desc[last].texture.format(),
                            bind.textures2d.desc[last].texture.tile_mode(),
                        )));
                    }
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
                    extended_buffer.as_deref(),
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
                    extended_buffer.as_deref(),
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
                        extended_buffer.as_deref(),
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
                        extended_buffer.as_deref(),
                    )?;
                    info.textures2d_readwrite += 1;
                    let last = (bind.textures2d.textures_num - 1) as usize;
                    if bind.textures2d.desc[last].texture.type_() != 9 {
                        return Err(ni("read-write texture type != 9 (Texture2D)"));
                    }
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
                    extended_buffer.as_deref(),
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
                extended_buffer = Some(mem.dwords_at(base).ok_or_else(|| bad_addr(base))?);
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

/// `XPS5X_TRACE_EUD` evidence dump: the full `ShaderUserData` mapping tables,
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
    if std::env::var_os("XPS5X_TRACE_EUD").is_none() {
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

/// Kyty: Shader.cpp `ShaderParseUsage2` (L1505) — the PS5 path over the
/// `ShaderUserData` direct/sharp mapping tables (no EUD/SRT yet, exactly as
/// upstream).
pub fn shader_parse_usage2(
    user_data: &ShaderUserData,
    info: &mut ShaderParsedUsage,
    bind: &mut ShaderBindResources,
    user_sgpr: &UserSgprInfo,
    user_sgpr_num: i32,
    eud: Option<&[u32]>,
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
    if user_data.srt_size_dw != 0 {
        return Err(ni("ShaderUserData srt_size_dw != 0"));
    }

    // Kyty leaves this None (no EUD support); increment 3 supplies the buffer
    // the caller read from guest memory, which the extended branch of
    // `read_sharp_fields` already knows how to index.
    let extended_buffer: Option<&[u32]> = eud;

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
        read_sharp_fields(
            &mut direct_sgprs,
            i32::from(sharp.offset_dw()),
            user_sgpr,
            extended_buffer,
            &mut peek,
        )?;
        if (peek[3] >> 28) & 0xF == 0 {
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
        let t = &bind.textures2d.desc[last].texture;
        // 9 = Texture2D, 11 = Cube (measured: Minecraft's 1024x1024x6 skybox).
        if t.type_() != 9 && t.type_() != 11 {
            return Err(ni_owned(format!(
                "read-only texture type {} is not Texture2D (9) \
                 (10=3D 11=Cube 12=1DArray 13=2DArray; base={:#x} {}x{} depth={} format={} tile={})",
                t.type_(),
                t.base40(),
                t.width5() + 1,
                t.height5() + 1,
                t.depth() + 1,
                t.format(),
                t.tile_mode(),
            )));
        }
    }

    for (slot, sharp) in user_data.sharp_resource_offset[1].iter().enumerate() {
        if sharp.offset_dw() == 0x7fff {
            continue;
        }
        // Beyond Kyty (upstream EXIT_NOT_IMPLEMENTEDs on a non-empty table 1):
        // measured in Minecraft's menu CS. Mirror the table-0 extension — a
        // size == 1 sharp is a 4-dword buffer V#, bound read-write. Anything
        // else stays a named failure until a title shows what it means.
        if sharp.size() != 1 {
            return Err(ni("sharp table 1 entry with size != 1"));
        }
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

    for (i, &direct) in direct_sgprs.iter().enumerate() {
        if direct {
            shader_get_direct_sgpr(&mut bind.direct_sgprs, i as i32, user_sgpr)?;
            info.direct_sgprs += 1;
        }
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

            if stride == u64::from(r.stride()) {
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
            tracing::warn!(
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
        if fetch_index != 0 {
            return Err(ni("attrib: fetch_index != 0"));
        }

        if info.resources_num as usize >= ShaderVertexInputInfo::RES_MAX {
            return Err(ni("attrib: too many vertex resources"));
        }

        let n = info.resources_num as usize;
        info.resources_dst[n].register_start = reg as i32;
        info.resources_dst[n].registers_num = size as i32;
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
/// The EUD pointer is the sgpr pair immediately AFTER the shader's declared
/// user SGPRs (`user_sgpr_num`) — measured on ASTRO.BOT compute, where
/// `declared=14` and `s14:s15` points at descriptor-shaped data. Returns `None`
/// when there is no EUD, the pair is out of range, the pointer is null, or
/// guest memory does not back it; every one of those keeps the caller on the
/// pre-existing "no extended buffer" path rather than inventing descriptors.
fn read_extended_user_data(
    user_data: &ShaderUserData,
    user_sgpr: &UserSgprInfo,
    user_sgpr_num: i32,
    mem: &impl ShaderMemory,
    shader_addr: u64,
    next_gen: bool,
) -> Option<Vec<u32>> {
    let size = user_data.eud_size_dw as usize;
    if size == 0 {
        return None;
    }

    let read_at = |ptr: u64| -> Option<Vec<u32>> {
        if ptr == 0 || ptr & 0x3 != 0 {
            return None;
        }
        let src = mem.dwords_at(ptr)?;
        (src.len() >= size).then(|| src[..size].to_vec())
    };

    // Strategy 1 — the pair immediately AFTER the declared user SGPRs. Measured
    // on shaders whose `count` EXCEEDS `declared` (e.g. declared=14 count=16:
    // s14:s15 hold the pointer).
    if let Ok(base) = usize::try_from(user_sgpr_num) {
        if base + 1 < UserSgprInfo::SGPRS_MAX {
            if let Some(buf) = read_at(user_sgpr_pair(user_sgpr, user_sgpr_num)) {
                return Some(buf);
            }
        }
    }

    // Strategy 2 — scalar-load analysis (resolver increments 1-2). When
    // `count == declared` there IS no register past the declared file (measured:
    // cs@0x50053c700 declared=14 count=14), so the pointer must be one the
    // shader itself loads through: find its `s_load_dwordx*` base pairs and take
    // the first that addresses readable guest memory of at least `eud_size_dw`.
    // CAVEAT: "first readable" is a heuristic — a shader with several scalar
    // loads could pick the wrong one, which shows up as wrong descriptors rather
    // than an error. Narrow it (e.g. by preferring the earliest load, or by
    // validating descriptor shape) if a title renders wrong data.
    let trace = std::env::var_os("XPS5X_TRACE_EUD").is_some();
    let Some(src) = mem.dwords_at(shader_addr) else {
        if trace {
            tracing::warn!("TRACE_EUD2 {shader_addr:#x}: shader code not mapped");
        }
        return None;
    };
    let mut code = ShaderCode::new();
    if let Err(e) = shader_parse(0, &src, &mut code, next_gen) {
        if trace {
            tracing::warn!("TRACE_EUD2 {shader_addr:#x}: shader_parse failed: {e}");
        }
        return None;
    }
    let loads = find_scalar_load_bases(&code);
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
    loads
        .into_iter()
        .filter_map(|load| scalar_load_target_address(&load, user_sgpr))
        .find_map(read_at)
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

        trace_eud_evidence(
            "vs(gs)",
            shader_addr,
            user_data,
            user_sgpr,
            user_sgpr_num,
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
            )
            .as_deref(),
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
    if usage.samplers > 0 {
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
        shader_parse_usage2(
            user_data,
            &mut usage,
            &mut ps_info.bind,
            &regs.ps_user_sgpr,
            i32::from(regs.ps_regs.rsrc2.user_sgpr),
            read_extended_user_data(
                user_data,
                &regs.ps_user_sgpr,
                i32::from(regs.ps_regs.rsrc2.user_sgpr),
                mem,
                regs.ps_regs.data_addr,
                next_gen,
            )
            .as_deref(),
        )?;
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
    if usage.storage_buffers_readwrite > 0 {
        return Err(ni("ps: read-write storage buffers"));
    }
    if usage.gds_pointers > 0 {
        return Err(ni("ps: gds pointers"));
    }
    if usage.direct_sgprs > 0 {
        return Err(ni("ps: direct sgprs"));
    }

    shader_calc_binding_indices(&mut ps_info.bind);

    Ok(())
}

/// Kyty: Shader.cpp `ShaderGetInputInfoCS` (L1811). `sh` is unused upstream
/// too (kept for API parity).
pub fn shader_get_input_info_cs(
    regs: &ComputeShaderInfo,
    _sh: &ShaderRegisters,
    mem: &impl ShaderMemory,
    shader_map: &ShaderMap,
    next_gen: bool,
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
        shader_parse_usage2(
            user_data,
            &mut usage,
            &mut info.bind,
            &regs.cs_user_sgpr,
            i32::from(regs.cs_regs.user_sgpr),
            read_extended_user_data(
                user_data,
                &regs.cs_user_sgpr,
                i32::from(regs.cs_regs.user_sgpr),
                mem,
                regs.cs_regs.data_addr,
                next_gen,
            )
            .as_deref(),
        )?;
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

    if usage.samplers > 0 {
        return Err(ni("cs: samplers"));
    }
    if usage.fetch || usage.vertex_buffer || usage.vertex_attrib {
        return Err(ni("cs: fetch / vertex buffer / vertex attrib"));
    }
    if usage.direct_sgprs > 0 {
        return Err(ni("cs: direct sgprs"));
    }

    shader_calc_binding_indices(&mut info.bind);

    Ok(())
}

/// Kyty: Shader.cpp `ShaderGetBindIds` (L2679). The id keys on the *binding
/// layout* (counts, slots, start registers, extended/usage flags), NOT on
/// descriptor contents — upstream deliberately commented out the
/// per-descriptor fields (L2685-L2694, L2705-L2728, L2739-L2765) and that is
/// preserved exactly: changing descriptor contents must not change the id.
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
    fn parse_attrib_still_rejects_fetch_index() {
        // fetch_index changes how the vertex/instance index feeds the fetch and
        // is not yet modelled; the measured case carried a junk V# anyway.
        let sem = ShaderSemantic { raw: 4 << 8 };
        let attrib = [2u32 | (1 << 26)]; // fetch_index = 1
        let buffer = vec![0u32; 12];
        let mut info = ShaderVertexInputInfo::default();
        assert!(matches!(
            shader_parse_attrib(&mut info, &[sem], &attrib, &buffer),
            Err(ShaderAnalysisError::NotImplemented { .. })
        ));
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
    fn parse_usage_bad_sgpr_type_is_error() {
        // Kyty: EXIT_NOT_IMPLEMENTED(type != Vsharp && type != Region)
        // (L1165) — constant buffer over Unknown-typed user SGPRs.
        let code = build_shader_blob(&[S_ENDPGM], &[[0x02, 0, 0, 0]], 0, 0, 0);
        let mem = TestMem { regions: vec![] };
        let mut info = ShaderParsedUsage::default();
        let mut bind = ShaderBindResources::default();
        let sgpr = UserSgprInfo::default(); // all Unknown
        assert!(matches!(
            shader_parse_usage(&code, &mem, &mut info, &mut bind, &sgpr, 4),
            Err(ShaderAnalysisError::NotImplemented { .. })
        ));
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
    fn shader_map_insert_keeps_existing() {
        // Kyty: ShaderMapUserData uses unordered_map::insert (L114), which
        // does not overwrite an existing key.
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
        map.map_user_data(0x1000, b);
        assert_eq!(map.find(0x1000), Some(&a));
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
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 10, None).unwrap();
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
        shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 16, None).unwrap();
        // type-0 sharp -> read-only storage buffer, NOT a texture.
        assert_eq!(info.storage_buffers_readonly, 1);
        assert_eq!(bind.storage_buffers.buffers_num, 1);
        assert_eq!(bind.storage_buffers.start_register[0], 0);
        // type-9 sharp -> Texture2D, still classified as a texture.
        assert_eq!(info.textures2d_readonly, 1);
        assert_eq!(bind.textures2d.textures_num, 1);
    }

    /// Manual disassembly harness (no-op unless `XPS5X_DISASM_FILE` names a
    /// dumped `.bin`): parses the shader and prints its instruction types and
    /// recovered scalar-load bases. Used to read the EUD/SRT CS's descriptor
    /// pointer-load pattern while building the resolver. Run with
    /// `XPS5X_DISASM_FILE=... cargo test -p kyty-graphics disasm_shader_from_env -- --nocapture`.
    #[test]
    fn disasm_shader_from_env() {
        let Ok(path) = std::env::var("XPS5X_DISASM_FILE") else {
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
        match shader_parse_usage2(&user_data, &mut info, &mut bind, &user_sgpr, 0, None) {
            Err(ShaderAnalysisError::NotImplementedOwned { what }) => {
                assert!(what.contains("EUD unreadable"), "{what}");
                assert!(what.contains("eud_size_dw=4"), "{what}");
                assert!(what.contains("declared=0"), "{what}");
                assert!(what.contains("(none)"), "{what}");
            }
            other => panic!("expected owned EUD diagnostic, got {other:?}"),
        }
    }
}
