//! PM4 packet codec + opcode / register indices.
//!
//! Faithful port of Kyty `emulator/include/Emulator/Graphics/Pm4.h`
//! (MIT (c) InoriRus) — a pure constant/macro header, realized here as consts
//! plus `const fn` codecs.
//!
//! # Type-3 header layout
//!
//! ```text
//! 31:30  TYPE       (must be 0b11)
//! 29:16  COUNT      (total_dw - 2)
//! 15:8   IT opcode
//!  7:2   R code     — Kyty's custom-op id, smuggled into AMD's reserved field
//!    1   SHADER_TYPE (0 = graphics, 1 = compute)
//!    0   PREDICATE  (never set by Kyty)
//! ```
//!
//! Every AGC/Gen5-specific operation rides on [`IT_NOP`] with an [`RCode`] in
//! bits 7:2, which is why a Gen5 stream looks like a sea of NOPs to a stock PM4
//! parser. See [`crate::run`] for the dispatch.
//!
//! # Length vocabulary
//!
//! Kyty overloads the word "len": `KYTY_PM4(len, ..)` takes a **total** dword
//! count, `KYTY_PM4_LEN` returns **total**, but the local `len` inside
//! `DumpPm4PacketStream` is a **body** count. The two are split here into
//! [`total_dw`] and [`body_dw`] so a caller cannot confuse them.

/// Kyty: `Pm4::CX_NUM` (Pm4.h L728) — context-register index space.
pub const CX_NUM: usize = 0x3FF + 1;
/// Kyty: `Pm4::SH_NUM` (Pm4.h L937) — shader-register index space.
///
/// Note this is **not** a power of two, so Kyty's `& (SH_NUM - 1)` idiom would
/// be wrong here; [`crate::run`] bounds-checks instead of masking.
pub const SH_NUM: usize = 0x2FF + 1;
/// Kyty: `Pm4::UC_NUM` (Pm4.h L992) — user-config register index space.
pub const UC_NUM: usize = 0x3FFF + 1;
/// Kyty: `Pm4::R_NUM` (Pm4.h L101) — custom-op id space (header bits 7:2).
pub const R_NUM: usize = 0x3F + 1;

/// A PM4 IT (instruction type) opcode — header bits 15:8.
///
/// A distinct type from [`RCode`] because their numeric spaces collide:
/// `IT_NOP` and `R_PS_UPDATE` are both `0x10`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ItOp(pub u8);

/// A Kyty custom-op id — header bits 7:2. Only meaningful on [`IT_NOP`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RCode(pub u8);

// ---- IT_* opcodes (Pm4.h L24-71) ----------------------------------------

pub const IT_NOP: ItOp = ItOp(0x10);
pub const IT_SET_BASE: ItOp = ItOp(0x11);
pub const IT_CLEAR_STATE: ItOp = ItOp(0x12);
pub const IT_INDEX_BUFFER_SIZE: ItOp = ItOp(0x13);
pub const IT_DISPATCH_DIRECT: ItOp = ItOp(0x15);
pub const IT_DISPATCH_INDIRECT: ItOp = ItOp(0x16);
pub const IT_SET_PREDICATION: ItOp = ItOp(0x20);
pub const IT_COND_EXEC: ItOp = ItOp(0x22);
pub const IT_DRAW_INDIRECT: ItOp = ItOp(0x24);
pub const IT_DRAW_INDEX_INDIRECT: ItOp = ItOp(0x25);
pub const IT_INDEX_BASE: ItOp = ItOp(0x26);
pub const IT_DRAW_INDEX_2: ItOp = ItOp(0x27);
pub const IT_CONTEXT_CONTROL: ItOp = ItOp(0x28);
pub const IT_INDEX_TYPE: ItOp = ItOp(0x2A);
pub const IT_DRAW_INDIRECT_MULTI: ItOp = ItOp(0x2C);
pub const IT_DRAW_INDEX_AUTO: ItOp = ItOp(0x2D);
pub const IT_NUM_INSTANCES: ItOp = ItOp(0x2F);
/// AMD `PKT3_DRAW_INDEX_MULTI_AUTO` (Mesa `src/amd/common/sid.h` L70; shadPS4
/// `pm4_opcodes.h` L33). **Not in Kyty's Pm4.h** — added here so the command
/// processor can NAME it rather than dropping it in the anonymous
/// unknown-opcode arm: `raeen-gpu`'s `agc::decode_submission` counts this
/// opcode as a draw, so a packet the walker silently skipped inflated the
/// submission's draw count with no matching `draws`, `draw_skips`, or
/// `refused_draws`. See [`crate::run::CommandProcessor::dispatch`]'s
/// unimplemented-draw arm.
pub const IT_DRAW_INDEX_MULTI_AUTO: ItOp = ItOp(0x30);
pub const IT_INDIRECT_BUFFER_CNST: ItOp = ItOp(0x33);
/// KytyPS5 `pm4.h` L44 (`IT_DISPATCH_DRAW_PREAMBLE`). **Not in Kyty's Pm4.h.**
///
/// The AGC multi-instanced indexed draw, and the one opcode in this file that
/// Raeen's own HLE *emits*: `sceAgcDcbDrawIndexMultiInstanced`
/// (`raeen-hle::hle_dcb_draw_index_multi_instanced`) writes the 9-dword
/// `0xC0073A00` packet KytyPS5's `CpOpDrawIndex` decodes. See
/// [`crate::run::CommandProcessor::cp_op_draw_index_multi_instanced`].
pub const IT_DISPATCH_DRAW_PREAMBLE: ItOp = ItOp(0x3A);
pub const IT_DRAW_INDEX_OFFSET_2: ItOp = ItOp(0x35);
pub const IT_WRITE_DATA: ItOp = ItOp(0x37);
pub const IT_DRAW_INDEX_INDIRECT_MULTI: ItOp = ItOp(0x38);
pub const IT_MEM_SEMAPHORE: ItOp = ItOp(0x39);
pub const IT_WAIT_REG_MEM: ItOp = ItOp(0x3C);
pub const IT_INDIRECT_BUFFER: ItOp = ItOp(0x3F);
pub const IT_COPY_DATA: ItOp = ItOp(0x40);
pub const IT_CP_DMA: ItOp = ItOp(0x41);
pub const IT_PFP_SYNC_ME: ItOp = ItOp(0x42);
pub const IT_SURFACE_SYNC: ItOp = ItOp(0x43);
pub const IT_EVENT_WRITE: ItOp = ItOp(0x46);
pub const IT_EVENT_WRITE_EOP: ItOp = ItOp(0x47);
pub const IT_EVENT_WRITE_EOS: ItOp = ItOp(0x48);
pub const IT_RELEASE_MEM: ItOp = ItOp(0x49);
pub const IT_DMA_DATA: ItOp = ItOp(0x50);
pub const IT_ACQUIRE_MEM: ItOp = ItOp(0x58);
pub const IT_REWIND: ItOp = ItOp(0x59);
pub const IT_SET_CONFIG_REG: ItOp = ItOp(0x68);
pub const IT_SET_CONTEXT_REG: ItOp = ItOp(0x69);
pub const IT_SET_SH_REG: ItOp = ItOp(0x76);
pub const IT_SET_QUEUE_REG: ItOp = ItOp(0x78);
pub const IT_SET_UCONFIG_REG: ItOp = ItOp(0x79);
pub const IT_SET_UCONFIG_REG_INDEX: ItOp = ItOp(0x7A);
pub const IT_WRITE_CONST_RAM: ItOp = ItOp(0x81);
pub const IT_DUMP_CONST_RAM: ItOp = ItOp(0x83);
pub const IT_INCREMENT_CE_COUNTER: ItOp = ItOp(0x84);
pub const IT_INCREMENT_DE_COUNTER: ItOp = ItOp(0x85);
pub const IT_WAIT_ON_CE_COUNTER: ItOp = ItOp(0x86);
pub const IT_WAIT_ON_DE_COUNTER_DIFF: ItOp = ItOp(0x88);
/// KytyPS5 `pm4.h` L71 (`IT_DISPATCH_DRAW`). Named but never dispatched there —
/// KytyPS5's `MakeOpcodeDispatchTable` (pm4Dispatch.cpp L212) wires only the
/// *preamble* opcode `0x3A`. Present here for the same reason as
/// [`IT_DRAW_INDEX_MULTI_AUTO`]: `agc::decode_submission` counts it as a draw,
/// so the walker must account for it by name.
pub const IT_DISPATCH_DRAW: ItOp = ItOp(0x8D);

// ---- R_* custom-op ids (Pm4.h L75-99) — the AGC/Gen5 dialect ------------

pub const R_ZERO: RCode = RCode(0x00);
pub const R_VS: RCode = RCode(0x01);
pub const R_PS: RCode = RCode(0x02);
pub const R_DRAW_INDEX: RCode = RCode(0x03);
pub const R_DRAW_INDEX_AUTO: RCode = RCode(0x04);
pub const R_DRAW_RESET: RCode = RCode(0x05);
pub const R_WAIT_FLIP_DONE: RCode = RCode(0x06);
pub const R_CS: RCode = RCode(0x07);
pub const R_DISPATCH_DIRECT: RCode = RCode(0x08);
pub const R_DISPATCH_RESET: RCode = RCode(0x09);
pub const R_WAIT_MEM_32: RCode = RCode(0x0A);
pub const R_PUSH_MARKER: RCode = RCode(0x0B);
pub const R_POP_MARKER: RCode = RCode(0x0C);
pub const R_VS_EMBEDDED: RCode = RCode(0x0D);
pub const R_PS_EMBEDDED: RCode = RCode(0x0E);
pub const R_VS_UPDATE: RCode = RCode(0x0F);
pub const R_PS_UPDATE: RCode = RCode(0x10);
pub const R_SH_REGS_INDIRECT: RCode = RCode(0x11);
pub const R_CX_REGS_INDIRECT: RCode = RCode(0x12);
pub const R_UC_REGS_INDIRECT: RCode = RCode(0x13);
pub const R_ACQUIRE_MEM: RCode = RCode(0x14);
pub const R_WRITE_DATA: RCode = RCode(0x15);
pub const R_WAIT_MEM_64: RCode = RCode(0x16);
pub const R_FLIP: RCode = RCode(0x17);
pub const R_RELEASE_MEM: RCode = RCode(0x18);
/// AGC DMA_DATA payload copy — emitted by `sceAgcDcbDmaData` (8-dw form) and
/// `sceAgcAcbDmaData` (7-dw form); the two layouts are discriminated by
/// packet length.
pub const R_DMA_DATA: RCode = RCode(0x19);

// ---- Header codec (Pm4.h L14-20) ----------------------------------------

/// Kyty: `KYTY_PM4(len, op, r)` (Pm4.h L16).
///
/// `total_dw` counts the header **and** the body, matching Kyty's call sites
/// (`KYTY_PM4(7, IT_NOP, R_RELEASE_MEM)` = 1 header + 6 body).
#[must_use]
pub const fn header(total_dw: u16, op: ItOp, r: RCode) -> u32 {
    debug_assert!(
        total_dw >= 2,
        "a type-3 packet is at least header + 1 dword"
    );
    0xC000_0000
        | (((total_dw as u32) - 2) & 0x3fff) << 16
        | ((op.0 as u32) << 8)
        | (((r.0 as u32) & 0x3f) << 2)
}

/// Is this a type-3 header? Kyty's `Run` loop omits this check; we don't.
#[must_use]
pub const fn is_type3(h: u32) -> bool {
    (h & 0xC000_0000) == 0xC000_0000
}

/// Kyty: type-2 filler packet (a bare NOP, 1 dword, no body).
#[must_use]
pub const fn is_type2(h: u32) -> bool {
    (h >> 30) == 2
}

/// Header bits 15:8.
#[must_use]
pub const fn op(h: u32) -> ItOp {
    ItOp(((h >> 8) & 0xff) as u8)
}

/// Kyty: `KYTY_PM4_R` (Pm4.h L19) — header bits 7:2.
#[must_use]
pub const fn r_code(h: u32) -> RCode {
    RCode(((h >> 2) & 0x3f) as u8)
}

/// Kyty: `KYTY_PM4_LEN` (Pm4.h L20) — COUNT + 2, i.e. header + body.
#[must_use]
pub const fn total_dw(h: u32) -> u32 {
    ((h >> 16) & 0x3fff) + 2
}

/// COUNT + 1, i.e. body only — Kyty's `DumpPm4PacketStream` local `len`.
#[must_use]
pub const fn body_dw(h: u32) -> u32 {
    ((h >> 16) & 0x3fff) + 1
}

/// Strip Kyty's bit31 "fake register" marker (`0x80000000 | idx`) before
/// indexing a register file. Kyty reserves the top of each index range for
/// emulator-invented registers that do not exist on real hardware.
#[must_use]
pub const fn strip_fake(idx: u32) -> u32 {
    idx & 0x7FFF_FFFF
}

// ---- Context (CX) register indices --------------------------------------
//
// These are **flat dword indices**, not MMIO byte addresses: the guest driver
// has already subtracted the context base, so a handler does no base math.

pub const DB_RENDER_CONTROL: u32 = 0x0;
// ---- Depth/stencil surface registers (Pm4.h L123-248) ---------------------
pub const DB_DEPTH_VIEW: u32 = 0x2;
pub const DB_HTILE_DATA_BASE: u32 = 0x5;
pub const DB_DEPTH_SIZE_XY: u32 = 0x7;
pub const DB_DEPTH_BOUNDS_MIN: u32 = 0x8;
pub const DB_DEPTH_BOUNDS_MAX: u32 = 0x9;
pub const DB_STENCIL_CLEAR: u32 = 0xA;
pub const DB_DEPTH_CLEAR: u32 = 0xB;
pub const DB_DEPTH_INFO: u32 = 0xF;
pub const DB_Z_INFO: u32 = 0x10;
pub const DB_STENCIL_INFO: u32 = 0x11;
pub const DB_Z_READ_BASE: u32 = 0x12;
pub const DB_STENCIL_READ_BASE: u32 = 0x13;
pub const DB_Z_WRITE_BASE: u32 = 0x14;
pub const DB_STENCIL_WRITE_BASE: u32 = 0x15;
pub const DB_DEPTH_SIZE: u32 = 0x16;
pub const DB_DEPTH_SLICE: u32 = 0x17;
pub const DB_Z_READ_BASE_HI: u32 = 0x1A;
pub const DB_STENCIL_READ_BASE_HI: u32 = 0x1B;
pub const DB_Z_WRITE_BASE_HI: u32 = 0x1C;
pub const DB_STENCIL_WRITE_BASE_HI: u32 = 0x1D;
pub const DB_HTILE_DATA_BASE_HI: u32 = 0x1E;
pub const PA_SC_SCREEN_SCISSOR_TL: u32 = 0xC;
pub const PA_SC_SCREEN_SCISSOR_BR: u32 = 0xD;
pub const CB_TARGET_MASK: u32 = 0x8E;
/// PS interpolator settings — 32 consecutive registers.
pub const SPI_PS_INPUT_CNTL_0: u32 = 0x191;
pub const SPI_VS_OUT_CONFIG: u32 = 0x1B1;
pub const SPI_PS_INPUT_ENA: u32 = 0x1B3;
pub const SPI_PS_INPUT_ADDR: u32 = 0x1B4;
pub const SPI_PS_IN_CONTROL: u32 = 0x1B6;
/// Target output modes, 4 bits per MRT (`ShaderRegisters::target_output_mode`).
pub const SPI_SHADER_COL_FORMAT: u32 = 0x1C5;
pub const DB_SHADER_CONTROL: u32 = 0x203;
/// Stencil op/mask registers (Pm4.h L314-346).
pub const DB_STENCIL_CONTROL: u32 = 0x10B;
pub const DB_STENCILREFMASK: u32 = 0x10C;
pub const DB_STENCILREFMASK_BF: u32 = 0x10D;
/// HTile surface control (Pm4.h L543) — tracked, not implemented.
pub const DB_HTILE_SURFACE: u32 = 0x2AF;
pub const PA_SC_GENERIC_SCISSOR_TL: u32 = 0x90;
pub const PA_SC_GENERIC_SCISSOR_BR: u32 = 0x91;
pub const PA_SC_VPORT_SCISSOR_0_TL: u32 = 0x94;
pub const PA_SC_VPORT_SCISSOR_0_BR: u32 = 0x95;
pub const PA_SC_VPORT_ZMIN_0: u32 = 0xB4;
pub const PA_CL_VPORT_XSCALE: u32 = 0x10F;
pub const CB_BLEND0_CONTROL: u32 = 0x1E0;
/// `CB_BLEND0_CONTROL..CB_BLEND7_CONTROL` occupy eight consecutive dwords.
pub const CB_BLEND_CONTROL_SLOTS: u32 = 8;
pub const CB_BLEND_RED: u32 = 0x105;
pub const CB_BLEND_GREEN: u32 = 0x106;
pub const CB_BLEND_BLUE: u32 = 0x107;
pub const CB_BLEND_ALPHA: u32 = 0x108;
pub const DB_DEPTH_CONTROL: u32 = 0x200;
pub const CB_COLOR_CONTROL: u32 = 0x202;
pub const PA_SU_SC_MODE_CNTL: u32 = 0x205;
pub const PA_SC_MODE_CNTL_0: u32 = 0x292;
/// Clip/cull control. Not modelled; decoded only to report its kill bits.
pub const PA_CL_CLIP_CNTL: u32 = 0x204;
/// EQAA sample mask, pixels X0Y0/X1Y0. Not modelled; a zero value is reported.
pub const PA_SC_AA_MASK_X0Y0_X1Y0: u32 = 0x30E;
/// EQAA sample mask, pixels X0Y1/X1Y1. Not modelled; a zero value is reported.
pub const PA_SC_AA_MASK_X0Y1_X1Y1: u32 = 0x30F;

/// Slot stride for the `CB_COLOR{n}_BASE` / `_INFO` register blocks.
/// Proof: `CB_COLOR7_BASE (0x381) - CB_COLOR0_BASE (0x318) == 7 * 15`.
pub const CB_COLOR_SLOT_STRIDE: u32 = 15;
pub const CB_COLOR0_BASE: u32 = 0x318;
pub const CB_COLOR7_BASE: u32 = 0x381;
pub const CB_COLOR0_VIEW: u32 = 0x31B;
pub const CB_COLOR0_INFO: u32 = 0x31C;
pub const CB_COLOR7_INFO: u32 = 0x385;
pub const CB_COLOR0_ATTRIB: u32 = 0x31D;
pub const CB_COLOR7_VIEW: u32 = 0x384;
pub const CB_COLOR7_ATTRIB: u32 = 0x386;
/// DCC control (Pm4.h L653) — compression metadata, decoded but ignored.
pub const CB_COLOR0_DCC_CONTROL: u32 = 0x31E;
pub const CB_COLOR7_DCC_CONTROL: u32 = 0x387;
/// CMASK metadata address low bits (Pm4.h L673).
pub const CB_COLOR0_CMASK: u32 = 0x31F;
pub const CB_COLOR7_CMASK: u32 = 0x388;
/// CMASK slice size (GCN `CB_COLOR0_CMASK_SLICE`; Kyty decodes it from the
/// fused Gen4 packet's dword 8 — GraphicsRun.cpp L1969). Not named in Kyty's
/// Pm4.h, but it is the register between CMASK (0x31F) and FMASK (0x321).
pub const CB_COLOR0_CMASK_SLICE: u32 = 0x320;
pub const CB_COLOR7_CMASK_SLICE: u32 = 0x389;
/// FMASK metadata address low bits (Pm4.h L674).
pub const CB_COLOR0_FMASK: u32 = 0x321;
pub const CB_COLOR7_FMASK: u32 = 0x38A;
/// FMASK slice size (GCN `CB_COLOR0_FMASK_SLICE`; Kyty Gen4 fused packet
/// dword 10 — GraphicsRun.cpp L1973).
pub const CB_COLOR0_FMASK_SLICE: u32 = 0x322;
pub const CB_COLOR7_FMASK_SLICE: u32 = 0x38B;
/// Packed fast-clear colour, low dword (Pm4.h L675).
pub const CB_COLOR0_CLEAR_WORD0: u32 = 0x323;
pub const CB_COLOR7_CLEAR_WORD0: u32 = 0x38C;
/// Packed fast-clear colour, high dword (Pm4.h L676).
pub const CB_COLOR0_CLEAR_WORD1: u32 = 0x324;
pub const CB_COLOR7_CLEAR_WORD1: u32 = 0x38D;
/// DCC metadata address low bits (Pm4.h L677).
pub const CB_COLOR0_DCC_BASE: u32 = 0x325;
pub const CB_COLOR7_DCC_BASE: u32 = 0x38E;

/// Gen5 high-address-byte blocks (Pm4.h L688-695). Stride **1**: eight
/// consecutive dwords per family, one per colour slot, carrying bits 40..48
/// of the matching 40-bit-shifted address.
pub const CB_COLOR0_BASE_EXT: u32 = 0x390;
pub const CB_COLOR7_BASE_EXT: u32 = 0x397;
pub const CB_COLOR0_CMASK_BASE_EXT: u32 = 0x398;
pub const CB_COLOR7_CMASK_BASE_EXT: u32 = 0x39F;
pub const CB_COLOR0_FMASK_BASE_EXT: u32 = 0x3A0;
pub const CB_COLOR7_FMASK_BASE_EXT: u32 = 0x3A7;
pub const CB_COLOR0_DCC_BASE_EXT: u32 = 0x3A8;
pub const CB_COLOR7_DCC_BASE_EXT: u32 = 0x3AF;

/// PS5 extent registers. Stride **1**, not 15 — and note these are nowhere near
/// `CB_COLOR0_BASE`. Kyty registers them only in its *indirect* table.
pub const CB_COLOR0_ATTRIB2: u32 = 0x3B0;
pub const CB_COLOR7_ATTRIB2: u32 = 0x3B7;
pub const CB_COLOR0_ATTRIB3: u32 = 0x3B8;
pub const CB_COLOR7_ATTRIB3: u32 = 0x3BF;

/// Viewport slot stride for `PA_CL_VPORT_XSCALE`.
pub const PA_CL_VPORT_STRIDE: u32 = 6;

// ---- Shader (SH) register indices ---------------------------------------

/// Gen5 shader binds are plain SH-register writes (Kyty registers these in
/// `g_hw_sh_indirect_func`, GraphicsRun.cpp L3995-4100): the PS stage gets
/// `PGM_LO/HI_PS` + `PGM_CHKSUM_PS` + `PGM_RSRC2_PS`, and the VS stage rides
/// the ES/GS registers (`PGM_LO/HI_ES`, `PGM_CHKSUM_GS`, `PGM_RSRC2_GS`) —
/// the "gs instead of vs" wave layout `shader_parse_vs` detects.
pub const SPI_SHADER_PGM_CHKSUM_PS: u32 = 0x06;
pub const SPI_SHADER_PGM_LO_PS: u32 = 0x08;
pub const SPI_SHADER_PGM_HI_PS: u32 = 0x09;
pub const SPI_SHADER_PGM_RSRC1_PS: u32 = 0x0A;
pub const SPI_SHADER_PGM_RSRC2_PS: u32 = 0x0B;
pub const SPI_SHADER_USER_DATA_PS_0: u32 = 0x0C;
pub const SPI_SHADER_USER_DATA_VS_0: u32 = 0x4C;
pub const SPI_SHADER_PGM_CHKSUM_GS: u32 = 0x80;
pub const SPI_SHADER_PGM_RSRC1_GS: u32 = 0x8A;
pub const SPI_SHADER_PGM_RSRC2_GS: u32 = 0x8B;
pub const SPI_SHADER_USER_DATA_GS_0: u32 = 0x8C;
pub const SPI_SHADER_PGM_LO_ES: u32 = 0xC8;
pub const SPI_SHADER_PGM_HI_ES: u32 = 0xC9;

/// Gen5 compute register indices (Kyty Pm4.h L887-929).
pub const COMPUTE_START_X: u32 = 0x204;
pub const COMPUTE_START_Y: u32 = 0x205;
pub const COMPUTE_START_Z: u32 = 0x206;
pub const COMPUTE_NUM_THREAD_X: u32 = 0x207;
pub const COMPUTE_NUM_THREAD_Y: u32 = 0x208;
pub const COMPUTE_NUM_THREAD_Z: u32 = 0x209;
pub const COMPUTE_PGM_LO: u32 = 0x20C;
pub const COMPUTE_PGM_HI: u32 = 0x20D;
pub const COMPUTE_PGM_RSRC1: u32 = 0x212;
pub const COMPUTE_PGM_RSRC2: u32 = 0x213;
pub const COMPUTE_PGM_RSRC3: u32 = 0x228;
pub const COMPUTE_SHADER_CHKSUM: u32 = 0x22A;
pub const COMPUTE_USER_DATA_0: u32 = 0x240;
pub const COMPUTE_USER_DATA_15: u32 = 0x24F;

pub mod compute_pgm_rsrc1 {
    pub const VGPRS: (u32, u32) = (0, 0x3F);
    pub const SGPRS: (u32, u32) = (6, 0xF);
    pub const BULKY: (u32, u32) = (24, 0x1);
}

pub mod compute_pgm_rsrc2 {
    pub const SCRATCH_EN: (u32, u32) = (0, 0x1);
    pub const USER_SGPR: (u32, u32) = (1, 0x1F);
    pub const TGID_X_EN: (u32, u32) = (7, 0x1);
    pub const TGID_Y_EN: (u32, u32) = (8, 0x1);
    pub const TGID_Z_EN: (u32, u32) = (9, 0x1);
    pub const TG_SIZE_EN: (u32, u32) = (10, 0x1);
    pub const TIDIG_COMP_CNT: (u32, u32) = (11, 0x3);
    pub const LDS_SIZE: (u32, u32) = (15, 0x1FF);
}

// ---- User-config (UC) register indices ----------------------------------

pub const VGT_PRIMITIVE_TYPE: u32 = 0x242;
pub const VGT_INDEX_TYPE: u32 = 0x243;

// ---- Bitfield accessors (Kyty: KYTY_PM4_GET) ----------------------------

/// Extract `(value >> shift) & mask`. Kyty's masks are pre-shifted-down value
/// masks, not in-place masks.
#[must_use]
pub const fn get(value: u32, shift: u32, mask: u32) -> u32 {
    (value >> shift) & mask
}

macro_rules! field {
    ($name:ident, $shift:expr, $mask:expr) => {
        pub const $name: (u32, u32) = ($shift, $mask);
    };
}

/// `CB_COLOR0_INFO` fields (Pm4.h L612-639).
pub mod cb_color_info {
    field!(FORMAT, 2, 0x1F);
    field!(NUMBER_TYPE, 8, 0x7);
    field!(COMP_SWAP, 11, 0x3);
    field!(FAST_CLEAR, 13, 0x1);
    field!(COMPRESSION, 14, 0x1);
    field!(BLEND_CLAMP, 15, 0x1);
    field!(BLEND_BYPASS, 16, 0x1);
    field!(ROUND_MODE, 18, 0x1);
    field!(CMASK_IS_LINEAR, 19, 0x1);
    field!(FMASK_COMPRESSION_DISABLE, 26, 0x1);
    field!(FMASK_COMPRESS_1FRAG_ONLY, 27, 0x1);
    field!(DCC_ENABLE, 28, 0x1);
    field!(CMASK_ADDR_TYPE, 29, 0x3);
    field!(ALT_TILE_MODE, 31, 0x1);
}

/// `CB_COLOR0_VIEW` fields (Pm4.h L603-609).
pub mod cb_color_view {
    field!(SLICE_START, 0, 0x1FFF);
    field!(SLICE_MAX, 13, 0x1FFF);
    field!(MIP_LEVEL, 26, 0xF);
}

/// `CB_COLOR0_ATTRIB` fields (Pm4.h L641-651).
pub mod cb_color_attrib {
    field!(TILE_MODE_INDEX, 0, 0x1F);
    field!(FMASK_TILE_MODE_INDEX, 5, 0x1F);
    field!(NUM_SAMPLES, 12, 0x7);
    field!(NUM_FRAGMENTS, 15, 0x3);
    field!(FORCE_DST_ALPHA_1, 17, 0x1);
}

/// `CB_COLOR0_DCC_CONTROL` fields (Pm4.h L654-671).
pub mod cb_color_dcc_control {
    field!(OVERWRITE_COMBINER_DISABLE, 0, 0x1);
    field!(KEY_CLEAR_ENABLE, 1, 0x1);
    field!(MAX_UNCOMPRESSED_BLOCK_SIZE, 2, 0x3);
    field!(MIN_COMPRESSED_BLOCK_SIZE, 4, 0x1);
    field!(MAX_COMPRESSED_BLOCK_SIZE, 5, 0x3);
    field!(COLOR_TRANSFORM, 7, 0x3);
    field!(INDEPENDENT_64B_BLOCKS, 9, 0x1);
    field!(ENABLE_CONSTANT_ENCODE_REG_WRITE, 19, 0x1);
    field!(INDEPENDENT_128B_BLOCKS, 20, 0x1);
}

/// `CB_COLOR0_ATTRIB2` fields (Pm4.h L698-703). The PS5 render-target extent.
pub mod cb_color_attrib2 {
    field!(MIP0_HEIGHT, 0, 0x3FFF);
    field!(MIP0_WIDTH, 14, 0x3FFF);
    field!(MAX_MIP, 28, 0xF);
}

/// `CB_COLOR0_ATTRIB3` fields (Pm4.h L708-717).
pub mod cb_color_attrib3 {
    field!(MIP0_DEPTH, 0, 0x1FFF);
    field!(COLOR_SW_MODE, 14, 0x1F);
    field!(RESOURCE_TYPE, 24, 0x3);
    field!(CMASK_PIPE_ALIGNED, 26, 0x1);
    field!(DCC_PIPE_ALIGNED, 30, 0x1);
}

/// `DB_RENDER_CONTROL` fields (Pm4.h L105-119).
pub mod db_render_control {
    field!(DEPTH_CLEAR_ENABLE, 0, 0x1);
    field!(STENCIL_CLEAR_ENABLE, 1, 0x1);
    field!(RESUMMARIZE_ENABLE, 4, 0x1);
    field!(STENCIL_COMPRESS_DISABLE, 5, 0x1);
    field!(DEPTH_COMPRESS_DISABLE, 6, 0x1);
    field!(COPY_CENTROID, 7, 0x1);
    field!(COPY_SAMPLE, 8, 0xF);
}

/// `DB_DEPTH_VIEW` fields (Pm4.h L123-137).
pub mod db_depth_view {
    field!(SLICE_START, 0, 0x7FF);
    field!(SLICE_START_HI, 11, 0x3);
    field!(SLICE_MAX, 13, 0x7FF);
    field!(Z_READ_ONLY, 24, 0x1);
    field!(STENCIL_READ_ONLY, 25, 0x1);
    field!(MIPID, 26, 0xF);
    field!(SLICE_MAX_HI, 30, 0x3);
}

/// `DB_DEPTH_SIZE_XY` fields (Pm4.h L144-148) — the PS5 depth-surface extent.
pub mod db_depth_size_xy {
    field!(X_MAX, 0, 0x3FFF);
    field!(Y_MAX, 16, 0x3FFF);
}

/// `DB_STENCIL_CLEAR` fields (Pm4.h L153-155).
pub mod db_stencil_clear {
    field!(CLEAR, 0, 0xFF);
}

/// `DB_DEPTH_INFO` fields (Pm4.h L175-189). Tiling metadata only.
pub mod db_depth_info {
    field!(ADDR5_SWIZZLE_MASK, 0, 0xF);
    field!(ARRAY_MODE, 4, 0xF);
    field!(PIPE_CONFIG, 8, 0x1F);
    field!(BANK_WIDTH, 13, 0x3);
    field!(BANK_HEIGHT, 15, 0x3);
    field!(MACRO_TILE_ASPECT, 17, 0x3);
    field!(NUM_BANKS, 19, 0x3);
}

/// `DB_Z_INFO` fields (Pm4.h L191-211).
pub mod db_z_info {
    field!(FORMAT, 0, 0x3);
    field!(NUM_SAMPLES, 2, 0x3);
    field!(ITERATE_FLUSH, 11, 0x1);
    field!(PARTIALLY_RESIDENT, 12, 0x1);
    field!(MAXMIP, 16, 0xF);
    field!(TILE_MODE_INDEX, 20, 0x7);
    field!(DECOMPRESS_ON_N_ZPLANES, 23, 0xF);
    field!(ALLOW_EXPCLEAR, 27, 0x1);
    field!(TILE_SURFACE_ENABLE, 29, 0x1);
    field!(ZRANGE_PRECISION, 31, 0x1);
}

/// `DB_STENCIL_INFO` fields (Pm4.h L213-227).
pub mod db_stencil_info {
    field!(FORMAT, 0, 0x1);
    field!(ITERATE_FLUSH, 11, 0x1);
    field!(PARTIALLY_RESIDENT, 12, 0x1);
    field!(RESERVED_FIELD_1, 13, 0x7);
    field!(TILE_MODE_INDEX, 20, 0x7);
    field!(ALLOW_EXPCLEAR, 27, 0x1);
    field!(TILE_STENCIL_DISABLE, 29, 0x1);
}

/// `DB_DEPTH_SIZE` fields (Pm4.h L234-238).
pub mod db_depth_size {
    field!(PITCH_TILE_MAX, 0, 0x7FF);
    field!(HEIGHT_TILE_MAX, 11, 0x7FF);
}

/// `DB_DEPTH_SLICE` fields (Pm4.h L240-242).
pub mod db_depth_slice {
    field!(SLICE_TILE_MAX, 0, 0x3F_FFFF);
}

/// `DB_STENCIL_CONTROL` fields (Pm4.h L314-326) — the six stencil ops.
pub mod db_stencil_control {
    field!(STENCILFAIL, 0, 0xF);
    field!(STENCILZPASS, 4, 0xF);
    field!(STENCILZFAIL, 8, 0xF);
    field!(STENCILFAIL_BF, 12, 0xF);
    field!(STENCILZPASS_BF, 16, 0xF);
    field!(STENCILZFAIL_BF, 20, 0xF);
}

/// `DB_STENCILREFMASK` / `DB_STENCILREFMASK_BF` fields (Pm4.h L328-346). The
/// `_BF` register uses the same layout with `_BF` suffixes in Kyty.
pub mod db_stencilrefmask {
    field!(STENCILTESTVAL, 0, 0xFF);
    field!(STENCILMASK, 8, 0xFF);
    field!(STENCILWRITEMASK, 16, 0xFF);
    field!(STENCILOPVAL, 24, 0xFF);
}

/// `DB_HTILE_SURFACE` fields (Pm4.h L543-557). Tracked, not implemented.
pub mod db_htile_surface {
    field!(LINEAR, 0, 0x1);
    field!(FULL_CACHE, 1, 0x1);
    field!(HTILE_USES_PRELOAD_WIN, 2, 0x1);
    field!(PRELOAD, 3, 0x1);
    field!(PREFETCH_WIDTH, 4, 0x3F);
    field!(PREFETCH_HEIGHT, 10, 0x3F);
    field!(DST_OUTSIDE_ZERO_TO_ONE, 16, 0x1);
}

/// `SPI_SHADER_PGM_RSRC2_*` fields shared by PS/GS (Pm4.h L760-771 / 854-866):
/// `USER_SGPR` at bit 1 (5 bits) with its MSB extension at bit 27.
pub mod spi_shader_pgm_rsrc2 {
    field!(USER_SGPR, 1, 0x1F);
    field!(USER_SGPR_MSB, 27, 0x1);
}

/// `PA_SC_SCREEN_SCISSOR_TL` / `_BR` fields (Pm4.h L162-171).
pub mod pa_sc_screen_scissor {
    field!(TL_X, 0, 0xFFFF);
    field!(TL_Y, 16, 0xFFFF);
    field!(BR_X, 0, 0xFFFF);
    field!(BR_Y, 16, 0xFFFF);
}

/// `PA_SC_GENERIC_SCISSOR_*` and `PA_SC_VPORT_SCISSOR_*` fields
/// (Pm4.h L268-294). Unlike the screen scissor, coordinates use 15 bits and
/// TL bit 31 disables the window offset.
pub mod pa_sc_offset_scissor {
    field!(TL_X, 0, 0x7FFF);
    field!(TL_Y, 16, 0x7FFF);
    field!(WINDOW_OFFSET_DISABLE, 31, 0x1);
    field!(BR_X, 0, 0x7FFF);
    field!(BR_Y, 16, 0x7FFF);
}

/// `PA_CL_CLIP_CNTL` field layout (Mesa `gfx103.json`, mm `0x28810`, context
/// offset 0x0204).
///
/// Every field Mesa's gfx103 register database names is listed, so a decode of
/// this register can account for all 32 bits instead of quietly ignoring the
/// ones nobody looked up yet: a *named* refusal is the whole point of
/// [`crate::hw_regs::ClipControl`]. Bit positions cross-checked against
/// shadPS4's `AmdGpu::ClipperControl` (`src/video_core/amdgpu/regs_primitive.h`
/// L23-42), which agrees field for field.
///
/// `UCP_ENA` is the 6-bit user-clip-plane enable mask (Mesa splits it into
/// `UCP_ENA_0..5`; one field is easier to test as "any plane enabled").
pub mod pa_cl_clip_cntl {
    field!(UCP_ENA, 0, 0x3F);
    field!(PS_UCP_Y_SCALE_NEG, 13, 0x1);
    field!(PS_UCP_MODE, 14, 0x3);
    field!(CLIP_DISABLE, 16, 0x1);
    field!(UCP_CULL_ONLY_ENA, 17, 0x1);
    field!(BOUNDARY_EDGE_FLAG_ENA, 18, 0x1);
    field!(DX_CLIP_SPACE_DEF, 19, 0x1);
    field!(DIS_CLIP_ERR_DETECT, 20, 0x1);
    field!(VTX_KILL_OR, 21, 0x1);
    field!(DX_RASTERIZATION_KILL, 22, 0x1);
    field!(DX_LINEAR_ATTR_CLIP_ENA, 24, 0x1);
    field!(VTE_VPORT_PROVOKE_DISABLE, 25, 0x1);
    field!(ZCLIP_NEAR_DISABLE, 26, 0x1);
    field!(ZCLIP_FAR_DISABLE, 27, 0x1);
    field!(ZCLIP_PROG_NEAR_ENA, 28, 0x1);
}

/// Human-readable names for the context registers this decoder does not model.
///
/// A bare `reg=0x0292` in the log is a dead end: it costs an agent a Mesa
/// register-database lookup to learn whether the skip matters, and the previous
/// two investigations of a flat frame each spent that cost from scratch. Naming
/// the skip turns "unknown context register" into a *named* refusal, so a log
/// line says whether the ignored write was raster-backend tile plumbing or a
/// rasterization gate.
///
/// Names come from Mesa's `src/amd/registers/gfx103.json` (context register
/// base mm `0x28000`, so `mm = 0x28000 + offset * 4`), cross-checked against
/// the AGC Gen5 register-defaults table in `raeen-hle`. This table is
/// diagnostics only — it never changes what is or is not applied.
#[must_use]
pub const fn context_reg_name(reg: u32) -> Option<&'static str> {
    Some(match reg {
        0x008f => "CB_SHADER_MASK",
        0x00b4 => "PA_SC_VPORT_ZMIN_0",
        0x00b5 => "PA_SC_VPORT_ZMAX_0",
        0x01b8 => "SPI_BARYC_CNTL",
        0x01c2 => "SPI_SHADER_IDX_FORMAT",
        0x01c3 => "SPI_SHADER_POS_FORMAT",
        0x01c4 => "SPI_SHADER_Z_FORMAT",
        0x01ff => "GE_MAX_OUTPUT_PER_SUBGROUP",
        0x0201 => "DB_EQAA",
        0x0204 => "PA_CL_CLIP_CNTL",
        0x0207 => "PA_CL_VS_OUT_CNTL",
        0x0291 => "VGT_GS_ONCHIP_CNTL",
        0x029b => "VGT_GS_OUT_PRIM_TYPE",
        0x02ab => "VGT_ESGS_RING_ITEMSIZE",
        0x02ce => "VGT_GS_MAX_VERT_OUT",
        0x02d3 => "GE_NGG_SUBGRP_CNTL",
        0x02d5 => "VGT_SHADER_STAGES_EN",
        0x02dc => "DB_ALPHA_TO_MASK",
        0x02df => "PA_SU_POLY_OFFSET_CLAMP",
        0x02e0 => "PA_SU_POLY_OFFSET_FRONT_SCALE",
        0x02e1 => "PA_SU_POLY_OFFSET_FRONT_OFFSET",
        0x02e2 => "PA_SU_POLY_OFFSET_BACK_SCALE",
        0x02e3 => "PA_SU_POLY_OFFSET_BACK_OFFSET",
        0x02e4 => "VGT_GS_INSTANCE_CNT",
        0x02f5 => "PA_SC_CENTROID_PRIORITY_0",
        0x02f6 => "PA_SC_CENTROID_PRIORITY_1",
        0x02f8 => "PA_SC_AA_CONFIG",
        0x02fe..=0x030d => "PA_SC_AA_SAMPLE_LOCS_PIXEL_*",
        0x030e => "PA_SC_AA_MASK_X0Y0_X1Y0",
        0x030f => "PA_SC_AA_MASK_X0Y1_X1Y1",
        0x0310 => "PA_SC_SHADER_CONTROL",
        _ => return None,
    })
}

/// Human-readable names for the user-config registers this decoder skips.
///
/// Same rationale and same source as [`context_reg_name`]; the user-config
/// register base is mm `0x30000`.
#[must_use]
pub const fn user_config_reg_name(reg: u32) -> Option<&'static str> {
    Some(match reg {
        0x024a => "GE_INDX_OFFSET",
        0x025b => "GE_CNTL",
        0x0262 => "GE_USER_VGPR_EN",
        _ => return None,
    })
}

/// `PA_SC_MODE_CNTL_0` field layout, per Mesa's `gfx103.json` register database
/// (`PA_SC_MODE_CNTL_0` at mm `0x28a48`, i.e. context offset 0x0292).
///
/// Only the three bits Kyty models in [`crate::hw_regs::ScanModeControl`] are
/// named here. The rest of the register — SEND_UNLIT_STILES_TO_PKR (3),
/// ALTERNATE_RBS_PER_TILE (5), COARSE_TILE_STARTS_ON_EVEN_RB (6) — selects
/// raster-backend tile distribution on real hardware and has no Vulkan
/// analogue, so it is deliberately not decoded rather than guessed at.
pub mod pa_sc_mode_cntl_0 {
    field!(MSAA_ENABLE, 0, 0x1);
    field!(VPORT_SCISSOR_ENABLE, 1, 0x1);
    field!(LINE_STIPPLE_ENABLE, 2, 0x1);
}

/// Read a `(shift, mask)` field pair out of a register value.
#[must_use]
pub const fn field(value: u32, f: (u32, u32)) -> u32 {
    get(value, f.0, f.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The codec is pinned against the exact `cmd_id` constants Kyty's own
    /// handlers hard-compare, so a wrong header layout cannot pass.
    #[test]
    fn header_encode_matches_kyty_emitted_constants() {
        assert_eq!(header(29, IT_NOP, R_VS), 0xC01B_1004);
        assert_eq!(header(40, IT_NOP, R_PS), 0xC026_1008);
        assert_eq!(header(29, IT_NOP, R_VS_EMBEDDED), 0xC01B_1034);
        assert_eq!(header(40, IT_NOP, R_PS_EMBEDDED), 0xC026_1038);
        assert_eq!(header(7, IT_NOP, R_DRAW_INDEX_AUTO), 0xC005_1010);
        assert_eq!(header(4, IT_NOP, R_CX_REGS_INDIRECT), 0xC002_1048);
        assert_eq!(header(25, IT_NOP, R_CS), 0xC017_101C);
        assert_eq!(header(9, IT_NOP, R_DISPATCH_DIRECT), 0xC007_1020);
        // Kyty: cp_op_set_context_reg's own EXIT_NOT_IMPLEMENTED constant.
        assert_eq!(header(3, IT_SET_CONTEXT_REG, R_ZERO), 0xC001_6900);
    }

    #[test]
    fn total_dw_and_body_dw_differ_by_one() {
        let h = header(7, IT_NOP, R_DRAW_INDEX_AUTO);
        assert_eq!(total_dw(h), 7, "header + body");
        assert_eq!(body_dw(h), 6, "body only");
    }

    #[test]
    fn header_roundtrips_op_and_r_code() {
        let h = header(40, IT_NOP, R_PS_EMBEDDED);
        assert_eq!(op(h), IT_NOP);
        assert_eq!(r_code(h), R_PS_EMBEDDED);
        assert!(is_type3(h));
    }

    /// `IT_NOP` and `R_PS_UPDATE` are both `0x10`; the newtypes keep the two
    /// spaces from being confused at a call site.
    #[test]
    fn it_op_and_r_code_do_not_share_a_type() {
        assert_eq!(IT_NOP.0, R_PS_UPDATE.0);
        let h = header(2, IT_NOP, R_PS_UPDATE);
        assert_eq!(op(h), IT_NOP);
        assert_eq!(r_code(h), R_PS_UPDATE);
    }

    #[test]
    fn is_type3_rejects_type0_and_type2() {
        assert!(!is_type3(0x0000_0000), "type 0");
        assert!(!is_type3(0x8000_0000), "type 2");
        assert!(is_type2(0x8000_0000));
        assert!(is_type3(0xC000_0000));
    }

    #[test]
    fn strip_fake_masks_the_bit31_marker() {
        assert_eq!(strip_fake(0x8000_03FF), 0x3FF, "CX_NOP");
        assert_eq!(strip_fake(0x3FF), 0x3FF);
    }

    /// The color slot strides are load-bearing and easy to transpose: BASE/INFO
    /// step by 15, but the PS5 extent registers step by 1.
    #[test]
    fn color_slot_strides_match_kyty_bounds() {
        assert_eq!(CB_COLOR0_BASE + 7 * CB_COLOR_SLOT_STRIDE, CB_COLOR7_BASE);
        assert_eq!(CB_COLOR0_INFO + 7 * CB_COLOR_SLOT_STRIDE, CB_COLOR7_INFO);
        assert_eq!(CB_COLOR0_ATTRIB2 + 7, CB_COLOR7_ATTRIB2);
        assert_eq!(CB_COLOR0_ATTRIB3 + 7, CB_COLOR7_ATTRIB3);
    }

    /// Every per-slot CB sub-register family steps by the 15-dword slot
    /// stride; the Gen5 `_EXT` address-byte blocks step by 1 (Kyty Pm4.h
    /// L678-695).
    #[test]
    fn color_sub_register_strides_match_kyty_bounds() {
        const S: u32 = 7 * CB_COLOR_SLOT_STRIDE;
        assert_eq!(CB_COLOR0_VIEW + S, CB_COLOR7_VIEW);
        assert_eq!(CB_COLOR0_ATTRIB + S, CB_COLOR7_ATTRIB);
        assert_eq!(CB_COLOR0_DCC_CONTROL + S, CB_COLOR7_DCC_CONTROL);
        assert_eq!(CB_COLOR0_CMASK + S, CB_COLOR7_CMASK);
        assert_eq!(CB_COLOR0_CMASK_SLICE + S, CB_COLOR7_CMASK_SLICE);
        assert_eq!(CB_COLOR0_FMASK + S, CB_COLOR7_FMASK);
        assert_eq!(CB_COLOR0_FMASK_SLICE + S, CB_COLOR7_FMASK_SLICE);
        assert_eq!(CB_COLOR0_CLEAR_WORD0 + S, CB_COLOR7_CLEAR_WORD0);
        assert_eq!(CB_COLOR0_CLEAR_WORD1 + S, CB_COLOR7_CLEAR_WORD1);
        assert_eq!(CB_COLOR0_DCC_BASE + S, CB_COLOR7_DCC_BASE);
        assert_eq!(CB_COLOR0_BASE_EXT + 7, CB_COLOR7_BASE_EXT);
        assert_eq!(CB_COLOR0_CMASK_BASE_EXT + 7, CB_COLOR7_CMASK_BASE_EXT);
        assert_eq!(CB_COLOR0_FMASK_BASE_EXT + 7, CB_COLOR7_FMASK_BASE_EXT);
        assert_eq!(CB_COLOR0_DCC_BASE_EXT + 7, CB_COLOR7_DCC_BASE_EXT);
        // The whole 0x318..=0x3BF neighbourhood is CB-owned; the per-slot
        // families tile the stride-15 blocks without overlap.
        assert_eq!(CB_COLOR0_CMASK, CB_COLOR0_BASE + 7);
        assert_eq!(CB_COLOR0_DCC_BASE, CB_COLOR0_BASE + 13);
    }

    #[test]
    fn attrib2_field_extraction_matches_kyty_layout() {
        // MIP0_HEIGHT at 0, MIP0_WIDTH at 14 — a 96x48 target stores w-1/h-1.
        let value = (95 << 14) | 47;
        assert_eq!(field(value, cb_color_attrib2::MIP0_WIDTH), 95);
        assert_eq!(field(value, cb_color_attrib2::MIP0_HEIGHT), 47);
    }
}
