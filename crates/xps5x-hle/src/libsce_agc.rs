//! HLE libSceAgc — the guest-facing GPU **command-buffer emission** layer.
//!
//! A faithful Rust port of SharpEmu's Agc DCB (Draw Command Buffer) emitters
//! (GPL-2.0). A title builds PM4 command buffers by calling `sceAgcDcb*`, each
//! of which **reserves DWORDs from the command buffer's cursor** and writes a
//! PM4 packet there. This module ports that exactly: the command-buffer cursor
//! model (`TryAllocateCommandDwords`) and the DCB emitters, using SharpEmu's
//! Agc PM4 encoding.
//!
//! Note the Agc PM4 **dialect** differs from `gnm`'s decoder: the header is
//! `0xC000_0000 | (len-2)<<16 | op<<8 | (reg&0x3F)<<2` — with a 6-bit
//! `register` sub-discriminator and a total-length (not body-count)
//! convention. So these buffers are for the (future) Agc-aware submission path,
//! not `gnm::process_command_buffer`. What's verifiable now — and tested — is
//! that each emitter writes the exact bytes SharpEmu's Agc writes and advances
//! the cursor correctly. (Consumption by a real GPU backend is the M2 Vulkan
//! follow-up; the register-defaults tables + ACB/compute emitters are further
//! Agc rows.)

use crate::{HleContext, HleRegistry};
use tracing::{debug, error, warn};

// DrawCommandBuffer struct field offsets (bytes).
const CB_CURSOR_UP: u64 = 0x10; // u64 — current write pointer (advances up)
const CB_CURSOR_DOWN: u64 = 0x18; // u64 — end/limit
const CB_RESERVED_DW: u64 = 0x30; // u32 — reserved tail dwords

// Agc PM4 IT (instruction type) opcodes + register sub-discriminators.
const IT_INDEX_TYPE: u32 = 0x2A;
const IT_NOP: u32 = 0x10;
const IT_INDEX_BUFFER_SIZE: u32 = 0x13;
const IT_INDEX_BASE: u32 = 0x26;
const IT_DRAW_INDEX_INDIRECT: u32 = 0x25;
const IT_DRAW_INDEX_2: u32 = 0x27;
const IT_GET_LOD_STATS: u32 = 0x8E;
const IT_DRAW_INDEX_OFFSET_2: u32 = 0x35;
const IT_NUM_INSTANCES: u32 = 0x2F;
const IT_DISPATCH_DIRECT: u32 = 0x15;
const IT_DISPATCH_INDIRECT: u32 = 0x16;
const IT_SET_BASE: u32 = 0x11;
const IT_EVENT_WRITE: u32 = 0x46;
const IT_WAIT_REG_MEM: u32 = 0x3C;
const IT_SET_SH_REG: u32 = 0x76;
// Standard PM4 type-3 opcodes for packets the title emits through the newer
// Gen5 entry points (the Agc dialect reuses the standard opcode numbering —
// compare IT_DISPATCH_DIRECT/IT_INDEX_TYPE/IT_SET_BASE above).
const IT_SET_PREDICATION: u32 = 0x22;
const IT_DRAW_INDIRECT: u32 = 0x24;
const IT_DRAW_INDIRECT_MULTI: u32 = 0x2C;
/// Chain execution into another command buffer (`sceAgcDcbJump` /
/// `sceAgcCbBranch`). Standard PM4 numbering, mirrored by SharpEmu's
/// `ItIndirectBuffer` and kyty-graphics `pm4::IT_INDIRECT_BUFFER`.
const IT_INDIRECT_BUFFER: u32 = 0x3F;
const IT_DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
const IT_COPY_DATA: u32 = 0x40;
const R_WAIT_MEM32: u32 = 0x0A;
const R_WAIT_MEM64: u32 = 0x16;
const R_RELEASE_MEM: u32 = 0x18;
const R_DMA_DATA: u32 = 0x19;
const R_ACQUIRE_MEM: u32 = 0x14;
const R_WRITE_DATA: u32 = 0x15;
/// Marker DWORD preceding a `CbSetShRegisterRange` packet.
const SET_SH_RANGE_MARKER: u32 = 0x6875_000D;
const R_ZERO: u32 = 0x00;
const R_DRAW_INDEX_AUTO: u32 = 0x04;
const R_DRAW_RESET: u32 = 0x05;
const R_WAIT_FLIP_DONE: u32 = 0x06;
const R_ACB_RESET: u32 = 0x09;
const R_PUSH_MARKER: u32 = 0x0B;
const R_SH_REGS_INDIRECT: u32 = 0x11;
const R_CX_REGS_INDIRECT: u32 = 0x12;
const R_UC_REGS_INDIRECT: u32 = 0x13;
const R_POP_MARKER: u32 = 0x0C;
const R_FLIP: u32 = 0x17;

/// The `DRAW_INDEX_AUTO` modifier a valid call must pass.
const DRAW_AUTO_MODIFIER: u64 = 0x4000_0000;
/// Generic SCE "invalid argument" error (`0x8002_0000 | EINVAL`).
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
/// Generic SCE "memory fault" error (`0x8002_0000 | EFAULT`).
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;
/// Generic SCE "not found" error (`0x8002_0000 | ESRCH`).
const SCE_ERROR_NOT_FOUND: u64 = 0x8002_0003;

// Shader-header layout + magic (byte offsets/values, from SharpEmu).
const SHADER_FILE_HEADER: u32 = 0x3433_3231;
const SHADER_VERSION: u32 = 0x18;
const SHADER_USER_DATA_OFFSET: u64 = 0x08;
const SHADER_CODE_OFFSET: u64 = 0x10;
const SHADER_CX_REGISTERS_OFFSET: u64 = 0x18;
const SHADER_SH_REGISTERS_OFFSET: u64 = 0x20;
const SHADER_INPUT_SEMANTICS_OFFSET: u64 = 0x30;
const SHADER_OUTPUT_SEMANTICS_OFFSET: u64 = 0x38;
const SHADER_NUM_INPUT_SEMANTICS_OFFSET: u64 = 0x50;
const SHADER_NUM_OUTPUT_SEMANTICS_OFFSET: u64 = 0x56;
const SHADER_TYPE_OFFSET: u64 = 0x5A;
const SHADER_NUM_SH_REGISTERS_OFFSET: u64 = 0x5C;
const COMPUTE_PGM_LO: u32 = 0x20C;
const COMPUTE_PGM_HI: u32 = 0x20D;
const SPI_SHADER_PGM_LO_PS: u32 = 0x008;
const SPI_SHADER_PGM_HI_PS: u32 = 0x009;
const SPI_SHADER_PGM_LO_GS: u32 = 0x08A;
const SPI_SHADER_PGM_HI_GS: u32 = 0x08B;
const SPI_SHADER_PGM_LO_ES: u32 = 0x0C8;
const SPI_SHADER_PGM_HI_ES: u32 = 0x0C9;
const SPI_SHADER_PGM_LO_LS: u32 = 0x148;
const SPI_SHADER_PGM_HI_LS: u32 = 0x149;
/// `SPI_PS_INPUT_CNTL_0` register offset (32 interpolant slots follow).
const SPI_PS_INPUT_CNTL0: u32 = 0x191;
// PrimState shader-register layout (byte offsets, from SharpEmu).
const SHADER_SPECIALS_OFFSET: u64 = 0x28;
const SPECIAL_GE_CNTL_OFFSET: u64 = 0x00;
const SPECIAL_VGT_SHADER_STAGES_EN_OFFSET: u64 = 0x08;
const SPECIAL_VGT_GS_OUT_PRIM_TYPE_OFFSET: u64 = 0x20;
const SPECIAL_GE_USER_VGPR_EN_OFFSET: u64 = 0x28;
/// `VGT_PRIMITIVE_TYPE` register offset.
const VGT_PRIMITIVE_TYPE: u32 = 0x242;

/// Register the libSceAgc DCB command emitters.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAgc", "sceAgcDcbSetIndexSize", hle_dcb_set_index_size);
    registry.register(
        "libSceAgc",
        "sceAgcDcbDrawIndexAuto",
        hle_dcb_draw_index_auto,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbSetNumInstances",
        hle_dcb_set_num_instances,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbSetIndexBuffer",
        hle_dcb_set_index_buffer,
    );
    registry.register("libSceAgc", "sceAgcDcbDrawIndex", hle_dcb_draw_index);
    registry.register("libSceAgc", "sceAgcDcbResetQueue", hle_dcb_reset_queue);
    registry.register(
        "libSceAgc",
        "sceAgcDcbWaitUntilSafeForRendering",
        hle_dcb_wait_until_safe,
    );
    registry.register("libSceAgc", "sceAgcDcbPopMarker", hle_dcb_pop_marker);
    registry.register("libSceAgc", "sceAgcCbDispatch", hle_cb_dispatch);
    registry.register("libSceAgc", "sceAgcCbNop", hle_cb_nop);
    // Gen5 retail imports use this observed NID (`wr23dPKyWc0`). As with
    // sceAgcInit, the recovered export name does not derive to that identity.
    registry.register_nid(
        "libSceAgc",
        "sceAgcCbReleaseMem",
        0xc2bd_b774_f2b2_59cd,
        hle_cb_release_mem,
    );
    registry.register("libSceAgc", "sceAgcDcbSetFlip", hle_dcb_set_flip);
    // Measured ASTRO.BOT: the first libSceAgc import it actually calls. Base-PS5
    // (non-Trinity) is the branch that goes on to submit a DCB — see
    // `hle_get_is_trinity_mode`.
    registry.register(
        "libSceAgc",
        "sceAgcGetIsTrinityMode",
        hle_get_is_trinity_mode,
    );
    // Known ONLY by its NID `dolOmWH+huQ` — no catalogue has the name, so bind
    // the measured identity explicitly (hashing this placeholder label would
    // yield a different NID and leave the title's import unresolved).
    registry.register_nid(
        "libSceAgc",
        "sceAgcUnknownDolOmWHhuQ",
        0x7689_4e99_61fe_86e4,
        hle_unknown_agc_dol_om_wh,
    );
    registry.register_nid(
        "libSceAgc",
        "sceAgcUnknownFd5Bp5tGTgo",
        0x7dde_41a7_9b46_4e0a,
        hle_unknown_agc_fd5_bp5t,
    );
    registry.register("libSceAgc", "sceAgcAcbResetQueue", hle_acb_reset_queue);
    registry.register(
        "libSceAgc",
        "sceAgcCbSetShRegisterRangeDirect",
        hle_cb_set_sh_register_range,
    );
    // Gen5 exports this known function under the observed NID
    // `23LRUSvYu1M`. Bind that identity explicitly: deriving a NID from the
    // recovered name does not produce the Gen5 import used by retail titles.
    registry.register_nid("libSceAgc", "sceAgcInit", 0xdb72_d151_2bd8_bb53, hle_init);
    registry.register("libSceAgc", "sceAgcDcbEventWrite", hle_dcb_event_write);
    registry.register("libSceAgc", "sceAgcAcbEventWrite", hle_acb_event_write);
    // sceAgcAcbWriteData is an alias of sceAgcDcbWriteData in the reference
    // (`AcbWriteData(ctx) => DcbWriteData(ctx)`) — both emit a WRITE_DATA packet.
    registry.register("libSceAgc", "sceAgcDcbWriteData", hle_dcb_write_data);
    registry.register("libSceAgc", "sceAgcAcbWriteData", hle_dcb_write_data);
    registry.register(
        "libSceAgc",
        "sceAgcAcbDispatchIndirect",
        hle_acb_dispatch_indirect,
    );
    registry.register("libSceAgc", "sceAgcDcbWaitRegMem", hle_dcb_wait_reg_mem);
    registry.register("libSceAgc", "sceAgcDcbDmaData", hle_dcb_dma_data);
    registry.register("libSceAgc", "sceAgcDcbAcquireMem", hle_dcb_acquire_mem);
    registry.register(
        "libSceAgc",
        "sceAgcDcbSetCxRegistersIndirect",
        hle_dcb_set_cx_regs_indirect,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbSetShRegistersIndirect",
        hle_dcb_set_sh_regs_indirect,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbSetUcRegistersIndirect",
        hle_dcb_set_uc_regs_indirect,
    );
    registry.register("libSceAgc", "sceAgcCreatePrimState", hle_create_prim_state);
    registry.register("libSceAgc", "sceAgcCreateShader", hle_create_shader);
    registry.register(
        "libSceAgc",
        "sceAgcCreateInterpolantMapping",
        hle_create_interpolant_mapping,
    );
    // Minecraft (PPSA17221) imports sceAgcCreateInterpolantMapping under the
    // NID `HV4j+E0MBHE` (0x1d5e23f84d0c0471) — hashing the recovered name
    // yields a DIFFERENT NID, so the render thread died calling an
    // "unimplemented" import while the implementation sat unreachable above.
    // Bind the measured identity explicitly.
    registry.register_nid(
        "libSceAgc",
        "sceAgcCreateInterpolantMapping",
        0x1d5e_23f8_4d0c_0471,
        hle_create_interpolant_mapping,
    );
    registry.register_nid(
        "libSceAgc",
        "sceAgcGetDataPacketPayloadAddress",
        0x57ef_9480_1b50_867d,
        hle_get_data_packet_payload_address,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbDispatchIndirect",
        hle_dcb_dispatch_indirect,
    );
    registry.register("libSceAgc", "sceAgcSuspendPoint", hle_suspend_point);
    // Known ONLY by its NID (`-KRzWekV120`, measured from Minecraft PPSA17221):
    // its identity is unknown — not present in Kyty or SharpEmu. A loud stub
    // returning 0; the NID must be explicit because hashing the placeholder
    // label yields a different NID (see `register_nid`).
    registry.register_nid(
        "libSceAgc",
        "sceAgcUnknownKRzWekV120",
        0xfca4_7359_e915_d76d,
        hle_unknown_krz_wek_v120,
    );
    registry.register_nid(
        "libSceAgcDriver",
        "sceAgcDriverRegisterOwner",
        0x5ff3_66e4_a2d1_11e8,
        hle_driver_register_owner,
    );
    registry.register_nid(
        "libSceAgcDriver",
        "sceAgcDriverRegisterResource",
        0x5b9c_f879_9ae3_11ab,
        hle_driver_register_resource,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDriverAddEqEvent",
        hle_driver_add_eq_event,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbDrawIndexOffset",
        hle_dcb_draw_index_offset,
    );
    // Known ONLY by its NID: `qj7QZpgr9Uw`. The real name is unrecovered, so
    // the NID must be given explicitly — hashing this placeholder label yields
    // a different NID entirely, which left this implementation registered and
    // permanently unreachable while the measured retail title imported exactly
    // this NID and reported it missing.
    registry.register_nid(
        "libSceAgc",
        "sceAgcUnknownQj7QZpgr9Uw",
        0xaa3e_d066_982b_f54c,
        hle_unknown_filler,
    );
    registry.register("libSceAgc", "sceAgcDcbPushMarker", hle_dcb_push_marker);
    registry.register(
        "libSceAgc",
        "sceAgcDcbSetBaseIndirectArgs",
        hle_dcb_set_base_indirect_args,
    );
    // Driver submission lives in libSceAgcDriver for Gen5 retail binaries.
    // Keep the legacy libSceAgc registration for older fixtures, but bind
    // both observed retail identities explicitly.
    registry.register_nid(
        "libSceAgcDriver",
        "sceAgcDriverSubmitDcb",
        0x5209_4921_98c6_b2c3,
        hle_driver_submit_dcb,
    );
    registry.register_nid(
        "libSceAgcDriver",
        "sceAgcDriverAgrSubmitDcb",
        0x0211_afa4_84eb_7f83,
        hle_driver_submit_dcb,
    );
    // Same story for async-compute submission and GPU event registration: both
    // were implemented but registered ONLY under `libSceAgc`, while the measured
    // ASTRO.BOT title imports them from `libSceAgcDriver`. Resolution is
    // provider-aware, so the implementations were unreachable — the title
    // reported them missing at the first GPU submission. NIDs are the measured
    // retail identities, not derived from these labels.
    registry.register_nid(
        "libSceAgcDriver",
        "sceAgcDriverSubmitAcb",
        0x8124_67af_bf45_f2d4,
        hle_driver_submit_acb,
    );
    registry.register_nid(
        "libSceAgcDriver",
        "sceAgcDriverAddEqEvent",
        0xc36a_c986_60fe_76c1,
        hle_driver_add_eq_event,
    );
    registry.register("libSceAgc", "sceAgcDriverSubmitDcb", hle_driver_submit_dcb);
    registry.register("libSceAgc", "sceAgcDriverSubmitAcb", hle_driver_submit_acb);
    registry.register(
        "libSceAgc",
        "sceAgcQueueEndOfPipeActionPatchAddress",
        hle_queue_eop_patch_address,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDmaDataPatchSetDstAddressOrOffset",
        hle_dma_data_patch_address,
    );
    registry.register(
        "libSceAgc",
        "sceAgcWaitRegMemPatchAddress",
        hle_wait_reg_mem_patch_address,
    );
    // This Gen5 export patches the destination field of a WRITE_DATA packet.
    // The retail identity (`fPSCdQxgpSw`) is newer than the named tables in
    // the current references, but its ABI and field are established by the
    // title's packet-copy relocation sequence.
    registry.register_nid(
        "libSceAgc",
        "sceAgcWriteDataPatchAddress",
        0x7cf4_8275_0c60_a52c,
        hle_write_data_patch_address,
    );
    // The Cx/Sh/Uc patch-set NIDs are behaviorally identical (the register
    // space only affects tracing), so each family shares one handler.
    for space in ["Cx", "Sh", "Uc"] {
        registry.register(
            "libSceAgc",
            &format!("sceAgcSet{space}RegIndirectPatchSetAddress"),
            hle_set_indirect_patch_address,
        );
        registry.register(
            "libSceAgc",
            &format!("sceAgcSet{space}RegIndirectPatchAddRegisters"),
            hle_add_indirect_patch_registers,
        );
    }
    // Minecraft (PPSA17221) RenderDragon batch — every name below hashes to
    // the NID the title imports (verified against the measured import table),
    // so plain name registration resolves them.
    registry.register(
        "libSceAgc",
        "sceAgcGetRegisterDefaults2",
        hle_get_register_defaults2,
    );
    registry.register(
        "libSceAgc",
        "sceAgcGetRegisterDefaults2Internal",
        hle_get_register_defaults2_internal,
    );
    registry.register("libSceAgc", "sceAgcDcbCopyData", hle_dcb_copy_data);
    registry.register(
        "libSceAgc",
        "sceAgcDcbDrawIndexIndirectMulti",
        hle_dcb_draw_index_indirect_multi,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbDrawIndirectMulti",
        hle_dcb_draw_indirect_multi,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbSetIndexCount",
        hle_dcb_set_index_count,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbSetPredication",
        hle_dcb_set_predication,
    );
    registry.register(
        "libSceAgc",
        "sceAgcSetRangePredication",
        hle_set_range_predication,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDebugRaiseException",
        hle_debug_raise_exception,
    );
    registry.register(
        "libSceAgc",
        "sceAgcCbSetShRegisterRangeDirectGetSize",
        hle_cb_set_sh_register_range_get_size,
    );
    // ASTRO.BOT (PPSA21564) command-buffer builder batch — every name below
    // hashes to the NID the title imports (verified with --missing-nids).
    registry.register("libSceAgc", "sceAgcAcbDmaData", hle_acb_dma_data);
    registry.register("libSceAgc", "sceAgcAcbCopyData", hle_acb_copy_data);
    registry.register("libSceAgc", "sceAgcAcbAcquireMem", hle_acb_acquire_mem);
    registry.register("libSceAgc", "sceAgcAcbWaitRegMem", hle_acb_wait_reg_mem);
    // SharpEmu aliases the ACB marker entry points to the DCB emitters
    // (`AcbPushMarker(ctx) => DcbPushMarker(ctx)`, AgcExports.cs L2099-2104 and
    // L2125-2130) — identical packets on either queue.
    registry.register("libSceAgc", "sceAgcAcbPushMarker", hle_dcb_push_marker);
    registry.register("libSceAgc", "sceAgcAcbPopMarker", hle_dcb_pop_marker);
    registry.register(
        "libSceAgc",
        "sceAgcCbSetShRegistersDirect",
        hle_cb_set_sh_registers_direct,
    );
    registry.register(
        "libSceAgc",
        "sceAgcCbDispatchGetSize",
        hle_cb_dispatch_get_size,
    );
    registry.register("libSceAgc", "sceAgcCbNopGetSize", hle_cb_nop_get_size);
    registry.register(
        "libSceAgc",
        "sceAgcDcbDrawIndexIndirect",
        hle_dcb_draw_index_indirect,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDcbStallCommandBufferParser",
        hle_dcb_stall_command_buffer_parser,
    );
    registry.register("libSceAgc", "sceAgcDcbGetLodStats", hle_dcb_get_lod_stats);
    // UE5 pair (Until Dawn PPSA15421 + Dragon Ball Sparking Zero PPSA15210 —
    // identical libSceAgc gap set) + A Plague Tale Requiem batch. Every name
    // hashes to the NID the titles import (verified with --imports).
    registry.register("libSceAgc", "sceAgcDcbJump", hle_dcb_jump);
    registry.register("libSceAgc", "sceAgcCbBranch", hle_cb_branch);
    registry.register("libSceAgc", "sceAgcDcbDrawIndirect", hle_dcb_draw_indirect);
    registry.register(
        "libSceAgc",
        "sceAgcSetPacketPredication",
        hle_set_packet_predication,
    );
    registry.register(
        "libSceAgc",
        "sceAgcSetCxRegIndirectPatchSetNumRegisters",
        hle_set_indirect_patch_set_num_registers,
    );
    registry.register("libSceAgc", "sceAgcDcbSetMarker", hle_dcb_set_marker);
    // Driver-family functions are exported by BOTH provider names: older
    // fixtures/titles import them from `libSceAgc`, retail Gen5 binaries from
    // `libSceAgcDriver` (measured: A Plague Tale Requiem imports
    // sceAgcDriverSubmitMultiDcbs from libSceAgcDriver while it was registered
    // only under libSceAgc — provider-aware resolution left it unreachable).
    // Register every Driver-family implementation under both libraries so this
    // whole bug class is closed, not just the measured instances.
    for (name, implementation) in [
        (
            "sceAgcDriverSubmitMultiDcbs",
            hle_driver_submit_multi_dcbs as crate::HleFunction,
        ),
        (
            "sceAgcDriverAgrSubmitMultiDcbs",
            hle_driver_submit_multi_dcbs,
        ),
        ("sceAgcDriverSubmitMultiAcbs", hle_driver_submit_multi_acbs),
        ("sceAgcDriverTriggerCapture", hle_driver_trigger_capture),
        (
            "sceAgcDriverGetResourceRegistrationMaxNameLength",
            hle_get_resource_max_name_length,
        ),
        (
            "sceAgcDriverRegisterDefaultOwner",
            hle_driver_register_default_owner,
        ),
        ("sceAgcDriverGetDefaultOwner", hle_driver_get_default_owner),
        ("sceAgcDriverDeleteEqEvent", hle_driver_delete_eq_event),
        (
            "sceAgcDriverQueryResourceRegistrationUserMemoryRequirements",
            hle_driver_query_resource_memory,
        ),
        (
            "sceAgcDriverInitResourceRegistration",
            hle_driver_init_resource_registration,
        ),
    ] {
        registry.register("libSceAgc", name, implementation);
        registry.register("libSceAgcDriver", name, implementation);
    }
}

/// Resource-registration maximum name length (`sceAgcDriver...`).
const RESOURCE_REGISTRATION_MAX_NAME_LENGTH: u32 = 256;
/// Resource-registration memory sizing.
const RESOURCE_REGISTRATION_BYTES_PER_RESOURCE: u64 = 0x118;
const RESOURCE_REGISTRATION_BYTES_PER_OWNER: u64 = 0x1E0;

/// `sceAgcDcbDispatchIndirect(dcb, dataOffset, modifier)`: emit a 3-DWORD
/// indirect dispatch packet (data offset + initiator).
fn hle_dcb_dispatch_indirect(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let data_offset = args.get(1).copied().unwrap_or(0) as u32;
    let modifier = args.get(2).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 3) else {
        return 0;
    };
    let initiator = (modifier & 0xA038) | 0x41;
    let ok = ctx
        .mem
        .write(addr, &pm4(3, IT_DISPATCH_INDIRECT, R_ZERO).to_le_bytes())
        && ctx.mem.write(addr + 4, &data_offset.to_le_bytes())
        && ctx.mem.write(addr + 8, &initiator.to_le_bytes());
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDriverGetResourceRegistrationMaxNameLength(out)`: write the max
/// resource-registration name length (256).
fn hle_get_resource_max_name_length(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    if out == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !ctx
        .mem
        .write(out, &RESOURCE_REGISTRATION_MAX_NAME_LENGTH.to_le_bytes())
    {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// `sceAgcSuspendPoint()`: a no-op suspension marker; succeeds.
fn hle_suspend_point(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0
}

/// `sceAgcGetIsTrinityMode()`: is this console a PS5 **Pro** (codename
/// "Trinity")? XPS5X emulates a base PS5, so this is always false.
///
/// The flag is returned **directly in EAX**, not through an out-parameter —
/// measured at the retail ASTRO.BOT call site: the return address
/// `module+0x6f3b3f4` holds `test eax,eax; jnz +0x0e`, so the caller consumes
/// the return value. On the ZERO (base-PS5) branch it falls through to
/// `lea rdi,<BSS global>; call sceAgcDriverSubmitDcb` — i.e. the *non*-Trinity
/// path is the one that submits GPU work, which is exactly the path this
/// emulator can execute. Answering "true" would route the title down
/// PS5-Pro-only paths XPS5X does not implement.
///
/// The name for the imported NID `0x05f0436466ed8bb0` (`BfBDZGbti7A`) was
/// recovered from SharpEmu's `aerolib` symbol catalogue; no emulator in the
/// reference set implements this function.
fn hle_get_is_trinity_mode(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0
}

/// Unidentified `libSceAgc` entry point, known only by its NID `fd5Bp5tGTgo`
/// (`0x7dde_41a7_9b46_4e0a`). Like its sibling below it appears in no reference
/// catalogue, so it is bound by NID and answers with plain success.
///
/// Measured ABI at the retail ASTRO.BOT call site (`module+0x4695c2`): it is
/// the fallback the *previous* unknown's NULL branch makes —
/// `mov [rbx+0x308..0x318], 0; lea rdi,[rbx+0x320]; mov rsi,r15; mov rdx,r12;
/// call` — so the argument shape is the same `f(ctx_or_out, a, b)`. The caller
/// consumes the **return value**: `cmp eax, 0x8a6c0008; je +0x0b`. That one
/// expected error skips zeroing `[rbx+0x328]`; every other value (including
/// success) takes the branch that zeroes it and continues — no abort on either
/// side.
///
/// Returning `0` therefore lands in the caller's well-handled path and leaves
/// the object region zeroed, consistent with the "no object" state the
/// surrounding code already established. Warned once so the gap stays visible.
fn hle_unknown_agc_fd5_bp5t(_ctx: &HleContext, _args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            "libSceAgc NID 0x7dde41a79b464e0a (fd5Bp5tGTgo) is unidentified: returning \
             success, which leaves the caller's object region zeroed. If the title later \
             misbehaves around AGC setup, identify this alongside dolOmWH+huQ."
        );
    }
    0
}

/// Unidentified `libSceAgc` entry point, known only by its NID `dolOmWH+huQ`
/// (`0x7689_4e99_61fe_86e4`). The name is in **no** reference catalogue —
/// SharpEmu's `aerolib`, Kyty Gen5 and shadPS4 all lack it — so it is bound by
/// NID and reports "no object" rather than guessing a behaviour.
///
/// Measured ABI at the retail ASTRO.BOT call site (`module+0x469538`):
/// `f(void *out /* rdi */, a /* rsi */, b /* rdx */)`. The caller pre-zeroes the
/// 16-byte local it passes as `out` (`vmovups [rbp-0x130], xmm0`), **ignores the
/// return value entirely**, then reads `*out` back and branches on NULL:
/// `mov r14,[rbp-0x130]; test r14,r14; jz +0x4c`. That NULL branch is graceful —
/// it zero-fills the owning struct's fields (`[rbx+0x308/0x310/0x318]`) and
/// continues into the next call, so a missing object is a state the title
/// already handles.
///
/// Therefore: succeed, and leave `*out` exactly as the caller zeroed it. This
/// deliberately does NOT fabricate a handle — inventing one would hand the guest
/// a pointer to an object that does not exist, which is the silent-corruption
/// failure mode this project prefers a loud gap over. Warned once so the gap
/// stays visible in logs.
fn hle_unknown_agc_dol_om_wh(_ctx: &HleContext, _args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            "libSceAgc NID 0x76894e9961fe86e4 (dolOmWH+huQ) is unidentified: reporting \
             'no object' (out-param left NULL, which the caller handles). If the title \
             later misbehaves around AGC setup, this is the first thing to identify."
        );
    }
    0
}

/// `sceAgcDriverRegisterDefaultOwner(owner)`: record the default resource owner.
fn hle_driver_register_default_owner(ctx: &HleContext, args: &[u64]) -> u64 {
    let owner = args.first().copied().unwrap_or(0) as u32;
    ctx.kernel
        .agc_default_owner
        .store(owner, std::sync::atomic::Ordering::Relaxed);
    0
}

/// `sceAgcDriverGetDefaultOwner(out)`: write the registered default owner.
fn hle_driver_get_default_owner(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    if out == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let owner = ctx
        .kernel
        .agc_default_owner
        .load(std::sync::atomic::Ordering::Relaxed);
    if !ctx.mem.write(out, &owner.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// `sceAgcDriverRegisterOwner(ownerOut, name)`: allocate a process-local
/// resource-owner handle after resource registration has been initialized.
fn hle_driver_register_owner(ctx: &HleContext, args: &[u64]) -> u64 {
    use std::sync::atomic::Ordering;

    let owner_out = args.first().copied().unwrap_or(0);
    let name_address = args.get(1).copied().unwrap_or(0);
    if owner_out == 0 || name_address == 0 {
        tracing::warn!("sceAgcDriverRegisterOwner: null owner/name pointer");
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let Some(name) = read_guest_cstring(
        ctx,
        name_address,
        RESOURCE_REGISTRATION_MAX_NAME_LENGTH as usize - 1,
    ) else {
        tracing::warn!("sceAgcDriverRegisterOwner: name at {name_address:#x} is unreadable");
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    let max_owners = ctx
        .kernel
        .agc_resource_registration_max_owners
        .load(Ordering::Acquire) as usize;
    if max_owners != 0 && ctx.kernel.agc_resource_owners.len() >= max_owners {
        tracing::warn!("sceAgcDriverRegisterOwner: owner table is full ({max_owners})");
        return SCE_ERROR_INVALID_ARGUMENT;
    }

    let default_owner = ctx.kernel.agc_default_owner.load(Ordering::Relaxed);
    let owner = loop {
        let candidate = ctx.kernel.agc_next_owner.fetch_add(1, Ordering::AcqRel);
        if candidate == 0 {
            return SCE_ERROR_INVALID_ARGUMENT;
        }
        if candidate != default_owner && !ctx.kernel.agc_resource_owners.contains_key(&candidate) {
            break candidate;
        }
    };
    ctx.kernel
        .agc_resource_owners
        .insert(owner, String::from_utf8_lossy(&name).into_owned());
    if !ctx.mem.write(owner_out, &owner.to_le_bytes()) {
        ctx.kernel.agc_resource_owners.remove(&owner);
        return SCE_ERROR_MEMORY_FAULT;
    }
    debug!(
        "sceAgcDriverRegisterOwner: owner={owner}, name={}",
        String::from_utf8_lossy(&name)
    );
    0
}

/// `sceAgcDriverRegisterResource(handleOut, owner, address, size, name, type,
/// flags)`: retain the guest allocation metadata and return a process-local
/// handle. Resource tracking is also used by SDK revisions that do not call
/// the optional user-memory registration initializer first.
fn hle_driver_register_resource(ctx: &HleContext, args: &[u64]) -> u64 {
    use std::sync::atomic::Ordering;

    let handle_out = args.first().copied().unwrap_or(0);
    let owner = args.get(1).copied().unwrap_or(0) as u32;
    let address = args.get(2).copied().unwrap_or(0);
    let size = args.get(3).copied().unwrap_or(0);
    let name_address = args.get(4).copied().unwrap_or(0);
    let resource_type = args.get(5).copied().unwrap_or(0) as u32;
    let flags = args.get(6).copied().unwrap_or(0) as u32;
    let Some(name) = read_guest_cstring(
        ctx,
        name_address,
        RESOURCE_REGISTRATION_MAX_NAME_LENGTH as usize - 1,
    ) else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    if handle_out == 0 || address == 0 || size == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let default_owner = ctx.kernel.agc_default_owner.load(Ordering::Acquire);
    if owner != default_owner && !ctx.kernel.agc_resource_owners.contains_key(&owner) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }

    let handle = ctx.kernel.agc_next_resource.fetch_add(1, Ordering::AcqRel);
    if handle == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    ctx.kernel.agc_resources.insert(
        handle,
        xps5x_kernel::AgcResource {
            owner,
            address,
            size,
            name: String::from_utf8_lossy(&name).into_owned(),
            resource_type,
            flags,
        },
    );
    if !ctx.mem.write(handle_out, &handle.to_le_bytes()) {
        ctx.kernel.agc_resources.remove(&handle);
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// Kernel event filter for GPU events registered via `sceAgcDriverAddEqEvent`
/// (Orbis `SCE_KERNEL_EVFILT_GRAPHICS_CORE`). Distinguishes them from user
/// events so a RELEASE_MEM end-of-pipe interrupt can find and trigger them.
const EVFILT_GRAPHICS_CORE: i16 = -14;

/// `sceAgcDriverAddEqEvent(equeue, eventId, userData)`: register an Agc event on
/// the event queue (reuses the kernel event-queue user-event machinery).
fn hle_driver_add_eq_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let equeue = args.first().copied().unwrap_or(0);
    let event_id = args.get(1).copied().unwrap_or(0);
    let user_data = args.get(2).copied().unwrap_or(0);
    if !ctx.kernel.kernel_equeues.contains_key(&equeue) {
        return SCE_ERROR_NOT_FOUND;
    }
    ctx.kernel.kernel_equeue_events.insert(
        (equeue, event_id),
        xps5x_kernel::EqueueUserEvent {
            udata: user_data,
            filter: EVFILT_GRAPHICS_CORE,
            ..Default::default()
        },
    );
    debug!(equeue, event_id, user_data, "registered AGC event");
    0
}

/// `sceAgcDriverDeleteEqEvent(equeue, eventId)`: remove a previously-added event.
fn hle_driver_delete_eq_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let equeue = args.first().copied().unwrap_or(0);
    let event_id = args.get(1).copied().unwrap_or(0);
    if !ctx.kernel.kernel_equeues.contains_key(&equeue) {
        return SCE_ERROR_NOT_FOUND;
    }
    ctx.kernel.kernel_equeue_events.remove(&(equeue, event_id));
    0
}

/// `sceAgcDcbSetBaseIndirectArgs(dcb, baseIndex, address)`: emit a SET_BASE
/// packet (4 DWORDs) with the base index folded into the header.
fn hle_dcb_set_base_indirect_args(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let base_index = args.get(1).copied().unwrap_or(0) as u32;
    let address = args.get(2).copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 4) else {
        return 0;
    };
    let header = pm4(4, IT_SET_BASE, R_ZERO) | (base_index << 1);
    let ok = ctx.mem.write(addr, &header.to_le_bytes())
        && ctx.mem.write(addr + 4, &1u32.to_le_bytes())
        && ctx
            .mem
            .write(addr + 8, &((address as u32) & !7).to_le_bytes())
        && ctx
            .mem
            .write(addr + 12, &((address >> 32) as u32).to_le_bytes());
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDriverSubmitDcb(packet)` / `sceAgcDriverSubmitAcb(owner, packet)`:
/// capture and structurally decode the submitted Gen5 PM4 command buffer, apply
/// sync/flip side effects, and (when the DCB contains draw packets) drive the
/// M2 Vulkan offscreen path via [`xps5x_gpu::AgcGpuSession`].
/// Walk the caller's guest stack for a return-address chain (arena-relative),
/// so one probe captures the full call path from the render submit up to the
/// frame-loop function — the RE on-ramp to the UI navigate gate. Mirror of the
/// pthread_cond scanner; kept local to avoid widening that module's API.
fn agc_guest_stack_chain(ctx: &HleContext, depth: usize) -> String {
    const BASE: u64 = 0x0000_1000_0000_0000;
    const SPAN: u64 = 0x1_0000_0000;
    if ctx.caller_rsp == 0 {
        return "<no stack>".to_owned();
    }
    let mut frames: Vec<String> = Vec::new();
    let mut word = [0u8; 8];
    for slot in 0..1024u64 {
        let Some(addr) = ctx.caller_rsp.checked_add(slot * 8) else {
            break;
        };
        if !ctx.mem.read(addr, &mut word) {
            break;
        }
        let value = u64::from_le_bytes(word);
        if (BASE..BASE + SPAN).contains(&value) {
            frames.push(format!("{:#x}", value - BASE));
            if frames.len() >= depth {
                break;
            }
        }
    }
    if frames.is_empty() {
        "<none>".to_owned()
    } else {
        frames.join(" <- ")
    }
}

fn hle_driver_submit_dcb(ctx: &HleContext, args: &[u64]) -> u64 {
    // RE probe (XPS5X_TRACE_MAINLOOP): the DCB submit is the per-frame render
    // tick, called on the main thread every frame. Its caller return-addr is
    // inside that tick — the on-ramp to the UI state machine that decides
    // whether to navigate a Gameface route. Log the first few distinct callers
    // so `xps5x --disas` can be aimed at the exact per-frame function.
    if std::env::var_os("XPS5X_TRACE_MAINLOOP").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEEN: AtomicU32 = AtomicU32::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 6 {
            tracing::warn!(
                caller = format_args!("{:#x}", ctx.caller_return_addr),
                chain = %agc_guest_stack_chain(ctx, 16),
                "TRACE_MAINLOOP: sceAgcDriverSubmitDcb caller + guest-stack chain (per-frame render tick)"
            );
            // Resolve the render-dispatch singleton's vtable slot live, so its
            // runtime-determined target (call [rax+0x18] in fn 0xa8ca90) can be
            // disassembled: read guest qword[BASE+0xe39e098] (singleton), then
            // qword[singleton+0x18] (the fn ptr), and report both arena-relative.
            const BASE: u64 = 0x0000_1000_0000_0000;
            let mut rd = |a: u64| -> Option<u64> {
                let mut w = [0u8; 8];
                ctx.mem.read(a, &mut w).then(|| u64::from_le_bytes(w))
            };
            if let Some(singleton) = rd(BASE + 0xe39e098) {
                let vfn = singleton.checked_add(0x18).and_then(&mut rd).unwrap_or(0);
                tracing::warn!(
                    singleton = format_args!("{singleton:#x}"),
                    singleton_rel = format_args!("{:#x}", singleton.wrapping_sub(BASE)),
                    vtable_slot_0x18 = format_args!("{vfn:#x}"),
                    vtable_slot_rel = format_args!("{:#x}", vfn.wrapping_sub(BASE)),
                    "TRACE_MAINLOOP: render-dispatch singleton + vtable[0x18] target (--disas this)"
                );
            }
        }
    }
    submit_validate(ctx, args.first().copied().unwrap_or(0), "DCB")
}

fn hle_driver_submit_acb(ctx: &HleContext, args: &[u64]) -> u64 {
    // ACB = Asynchronous Compute Buffer (the ACE compute ring). On hardware this
    // is a SEPARATE queue from the graphics DCB, with its own shader state — a
    // graphics queue reset does not touch it. Diagnostic: count how much compute
    // work arrives here vs the DCB, to decide whether the shared-CommandProcessor
    // model (which lets a DCB reset clobber the compute shader) is the cause of
    // the zeroed-shader dispatches. `owner` is args[0]; the packet is args[1].
    submit_validate(ctx, args.get(1).copied().unwrap_or(0), "ACB")
}

/// Shared loop for the multi-buffer submission entry points: submit each
/// (address, size) pair through the REAL single-buffer path
/// ([`submit_command_buffer`] — decode, sync writes, DMA, events, flips,
/// `ctx.gpu.submit`).
///
/// ABI (SharpEmu `DriverSubmitMultiDcbs`, AgcExports.cs, reversed from Quake):
/// `rdi` = array of command-buffer base addresses (**u64** each, stride 8),
/// `rsi` = array of sizes in dwords (**u32** each, stride 4), `rdx` = count.
/// The earlier XPS5X validation stub read the size array with a u64 stride —
/// wrong per that reference; fixed here. Null/zero entries are skipped, not
/// fatal, exactly as SharpEmu `continue`s past them.
fn driver_submit_multi(ctx: &HleContext, args: &[u64], queue: &'static str) -> u64 {
    let address_array = args.first().copied().unwrap_or(0);
    let size_array = args.get(1).copied().unwrap_or(0);
    let buffer_count = args.get(2).copied().unwrap_or(0);
    if address_array == 0 || size_array == 0 || buffer_count == 0 || buffer_count > 4096 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    for i in 0..buffer_count {
        let cmd = read_u64_or_zero(ctx, address_array + i * 8);
        let dwords = read_u32_or_zero(ctx, size_array + i * 4);
        if cmd == 0 || dwords == 0 {
            continue;
        }
        let rc = submit_command_buffer(ctx, cmd, dwords, queue);
        if rc != 0 {
            warn!(
                index = i,
                count = buffer_count,
                command = format_args!("{cmd:#x}"),
                dwords,
                queue,
                "multi-buffer submission entry failed to decode — skipped"
            );
        }
    }
    0
}

/// `sceAgcDriverSubmitMultiDcbs(addressArray, sizeArray, bufferCount)`: submit
/// every graphics command buffer in the arrays through the real single-DCB
/// path. Also serves `sceAgcDriverAgrSubmitMultiDcbs` (measured A Plague Tale
/// Requiem import) — the Agr variant mirrors how `sceAgcDriverAgrSubmitDcb`
/// aliases `sceAgcDriverSubmitDcb` in this file.
fn hle_driver_submit_multi_dcbs(ctx: &HleContext, args: &[u64]) -> u64 {
    driver_submit_multi(ctx, args, "DCB")
}

/// `sceAgcDriverSubmitMultiAcbs(addressArray, sizeArray, bufferCount)`:
/// multi-buffer submission onto the **async-compute** ring (routes to
/// `GpuQueue::AsyncCompute`, see `submit_command_buffer`). No reference
/// implements this export (absent from SharpEmu/Kyty); the array ABI is
/// mirrored from `DriverSubmitMultiDcbs` above, the queue from
/// `sceAgcDriverSubmitAcb`. Measured A Plague Tale Requiem import.
fn hle_driver_submit_multi_acbs(ctx: &HleContext, args: &[u64]) -> u64 {
    driver_submit_multi(ctx, args, "ACB")
}

/// `sceAgcDriverTriggerCapture(...)`: hook for Sony's GPU-capture tooling
/// (Razor-style frame captures). XPS5X has no host capture pipeline; the call
/// is accepted as an OK no-op — capture is diagnostics, never rendering
/// progress. No reference implements it (absent from SharpEmu/Kyty); the
/// arguments are recorded at debug level for future RE. Measured A Plague
/// Tale Requiem import (NID `Xq5WmbwPTnQ`).
fn hle_driver_trigger_capture(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        arg0 = format_args!("{:#x}", args.first().copied().unwrap_or(0)),
        arg1 = format_args!("{:#x}", args.get(1).copied().unwrap_or(0)),
        "sceAgcDriverTriggerCapture: accepted (no host GPU-capture pipeline)"
    );
    0
}

/// `sceAgcDriverQueryResourceRegistrationUserMemoryRequirements(sizeOut,
/// resourceCount, ownerCount)`: write the required backing-memory size.
fn hle_driver_query_resource_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let size_out = args.first().copied().unwrap_or(0);
    let resource_count = args.get(1).copied().unwrap_or(0);
    let owner_count = args.get(2).copied().unwrap_or(0);
    if size_out == 0 || resource_count == 0 || owner_count == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let required = resource_count
        .checked_mul(RESOURCE_REGISTRATION_BYTES_PER_RESOURCE)
        .and_then(|r| {
            owner_count
                .checked_mul(RESOURCE_REGISTRATION_BYTES_PER_OWNER)
                .and_then(|o| r.checked_add(o))
        });
    let Some(required) = required else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    if !ctx.mem.write(size_out, &required.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// Count of ACB (async-compute) submissions that carried dispatch packets —
/// diagnostic for the shared-CommandProcessor question (see `hle_driver_submit_acb`).
static ACB_DISPATCH_SUBMITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn submit_validate(ctx: &HleContext, packet: u64, queue: &'static str) -> u64 {
    if packet == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    // Gen5's DCB submission descriptor is { u64 address, u32 dwords, u32 pad }.
    let mut descriptor = [0u8; 16];
    if !ctx.mem.read(packet, &mut descriptor) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let command_address = u64::from_le_bytes(descriptor[0..8].try_into().unwrap());
    let dword_count = u32::from_le_bytes(descriptor[8..12].try_into().unwrap());
    submit_command_buffer(ctx, command_address, dword_count, queue)
}

/// Submit one raw command buffer (`command_address`, `dword_count`) to the
/// decode + side-effect + GPU-handoff path. This is the shared tail of every
/// submission entry point: the single-DCB descriptor path (`submit_validate`)
/// and the multi-buffer array paths (`sceAgcDriverSubmitMulti{Dcbs,Acbs}`)
/// both land here, so array submissions get the REAL semantics — sync writes,
/// DMA, events, flips, and `ctx.gpu.submit` — not just validation.
/// Next value for a RELEASE_MEM GPU-timestamp fence: the session monotonic
/// clock in nanoseconds, forced strictly increasing across calls (and
/// therefore never zero) the way the hardware clock counter is. The state
/// lives on `OrbisKernel` so a relaunched session restarts from its own
/// clock instead of counting up from a prior session's final value (which
/// would collapse timestamp deltas to ~1 ns).
fn next_gpu_timestamp(ctx: &HleContext) -> u64 {
    use std::sync::atomic::Ordering;
    let now = u64::try_from(ctx.services.monotonic_elapsed().as_nanos()).unwrap_or(u64::MAX);
    let mut next = now.max(1);
    let _ = ctx
        .kernel
        .agc_gpu_timestamp
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |prev| {
            next = now.max(prev.saturating_add(1));
            Some(next)
        });
    next
}

fn submit_command_buffer(
    ctx: &HleContext,
    command_address: u64,
    dword_count: u32,
    queue: &'static str,
) -> u64 {
    if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let Some(byte_count) = (dword_count as usize).checked_mul(4) else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    let mut bytes = vec![0u8; byte_count];
    if !ctx.mem.read(command_address, &mut bytes) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    let Ok(decoded) = xps5x_gpu::agc::decode_submission(&words) else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };

    for write in &decoded.memory_writes {
        if !ctx.mem.write(write.address, &write.data) {
            tracing::warn!(
                address = write.address,
                bytes = write.data.len(),
                packet_offset = write.packet_offset,
                "AGC synchronization write targeted unreadable guest memory"
            );
            return SCE_ERROR_INVALID_ARGUMENT;
        }
    }
    // RELEASE_MEM `data_selection` 3 fences: hardware writes the GPU core
    // clock counter — non-zero and monotonic. Writing the packet's zero
    // immediate instead left titles polling a fence stuck at zero (measured:
    // ASTRO.BOT render thread parked forever holding its context mutex).
    for ts in &decoded.timestamp_writes {
        let value = next_gpu_timestamp(ctx);
        if !ctx.mem.write(ts.address, &value.to_le_bytes()) {
            tracing::warn!(
                address = ts.address,
                packet_offset = ts.packet_offset,
                "AGC timestamp fence targeted unreadable guest memory"
            );
            return SCE_ERROR_INVALID_ARGUMENT;
        }
    }
    // IT_DMA_DATA side effects: real guest-memory copies and pattern fills
    // (texture/buffer uploads). Applied inline at submit: the guest-visible
    // result matches hardware ordering closely enough for the titles measured.
    for copy in &decoded.memory_copies {
        let mut bytes = vec![0u8; copy.num_bytes as usize];
        if !ctx.mem.read(copy.src, &mut bytes) {
            tracing::warn!(
                src = copy.src,
                dst = copy.dst,
                bytes = copy.num_bytes,
                packet_offset = copy.packet_offset,
                "AGC DMA copy read from unreadable guest memory"
            );
            return SCE_ERROR_INVALID_ARGUMENT;
        }
        if !ctx.mem.write(copy.dst, &bytes) {
            tracing::warn!(
                src = copy.src,
                dst = copy.dst,
                bytes = copy.num_bytes,
                packet_offset = copy.packet_offset,
                "AGC DMA copy targeted unreadable guest memory"
            );
            return SCE_ERROR_INVALID_ARGUMENT;
        }
    }
    for fill in &decoded.memory_fills {
        let dwords = fill.num_bytes as usize / 4;
        let mut bytes = Vec::with_capacity(dwords * 4);
        for _ in 0..dwords {
            bytes.extend_from_slice(&fill.value.to_le_bytes());
        }
        if !ctx.mem.write(fill.address, &bytes) {
            tracing::warn!(
                address = fill.address,
                bytes = fill.num_bytes,
                packet_offset = fill.packet_offset,
                "AGC DMA fill targeted unreadable guest memory"
            );
            return SCE_ERROR_INVALID_ARGUMENT;
        }
    }
    for event_id in &decoded.events {
        for mut event in ctx.kernel.kernel_equeue_events.iter_mut() {
            if event.key().1 == u64::from(*event_id) {
                event.triggered = true;
                event.fflags = event.fflags.saturating_add(1);
                event.data = i64::from(*event_id);
            }
        }
    }
    // RELEASE_MEM end-of-pipe interrupts: hardware raises the EOP interrupt
    // and the kernel delivers it to every event registered via
    // sceAgcDriverAddEqEvent. These packets carry no memory write, so without
    // this bridge an interrupt-only completion is signaled by no component.
    // Fidelity debt: real delivery is per (equeue, event id) with the
    // interrupt selector (1-3) distinguishing pipe/type; this broadcasts to
    // every graphics-core event and folds N interrupts into one trigger.
    // Over-signaling is safe under the eager-completion model ("done" is
    // already true when submit returns), but revisit if a title registers
    // distinct events for graphics vs compute EOP and routes on data/ident.
    if !decoded.eop_interrupts.is_empty() {
        let count = decoded.eop_interrupts.len() as u32;
        let last_context = decoded
            .eop_interrupts
            .last()
            .map(|i| i.context_id)
            .unwrap_or(0);
        for mut event in ctx.kernel.kernel_equeue_events.iter_mut() {
            if event.filter == EVFILT_GRAPHICS_CORE {
                event.triggered = true;
                event.fflags = event.fflags.saturating_add(count);
                event.data = i64::from(last_context);
            }
        }
    }
    for flip in &decoded.flips {
        let _ = crate::libsce_video_out::submit_flip_from_agc(
            ctx,
            flip.video_out_handle,
            flip.display_buffer_index,
            flip.flip_mode,
            flip.flip_arg,
        );
    }

    use std::sync::atomic::Ordering;
    ctx.kernel
        .agc_last_dcb_address
        .store(command_address, Ordering::Relaxed);
    ctx.kernel
        .agc_last_dcb_dwords
        .store(dword_count, Ordering::Relaxed);
    let prior_submissions = ctx
        .kernel
        .agc_submission_count
        .fetch_add(1, Ordering::Relaxed);
    let prior_draws = ctx
        .kernel
        .agc_draw_packet_count
        .fetch_add(u64::from(decoded.draw_packets), Ordering::Relaxed);
    ctx.kernel
        .agc_dispatch_packet_count
        .fetch_add(u64::from(decoded.dispatch_packets), Ordering::Relaxed);
    let prior_flips = ctx
        .kernel
        .agc_flip_packet_count
        .fetch_add(decoded.flips.len() as u64, Ordering::Relaxed);

    // Diagnostic: how much compute lands on the ACB (async-compute ring) vs the
    // DCB. If dispatches arrive on the ACB, the shared-CommandProcessor model is
    // wrong (a graphics DCB reset clobbers the ACB's compute shader state) and
    // the fix is a separate CP / shader context per queue.
    if queue == "ACB" && decoded.dispatch_packets != 0 {
        let prior_acb = ACB_DISPATCH_SUBMITS.fetch_add(1, Ordering::Relaxed);
        if prior_acb == 0 || (prior_acb + 1).is_power_of_two() {
            tracing::warn!(
                acb_dispatch_submits = prior_acb + 1,
                dispatches = decoded.dispatch_packets,
                draws = decoded.draw_packets,
                "AGC ACB (async-compute) submission carries dispatches"
            );
        }
    }

    if prior_submissions == 0
        || (prior_draws == 0 && decoded.draw_packets != 0)
        || (prior_flips == 0 && !decoded.flips.is_empty())
    {
        tracing::info!(
            queue,
            command_address,
            dword_count,
            packets = decoded.packets.len(),
            draws = decoded.draw_packets,
            dispatches = decoded.dispatch_packets,
            flips = decoded.flips.len(),
            packet_layout = ?decoded.packets,
            flip_layout = ?decoded.flips,
            memory_writes = ?decoded.memory_writes,
            waits32 = ?decoded.waits32,
            events = ?decoded.events,
            "captured AGC submission"
        );
    } else if (prior_submissions + 1).is_power_of_two() {
        tracing::info!(
            queue,
            submissions = prior_submissions + 1,
            total_draws = ctx.kernel.agc_draw_packet_count.load(Ordering::Relaxed),
            total_dispatches = ctx.kernel.agc_dispatch_packet_count.load(Ordering::Relaxed),
            total_flips = ctx.kernel.agc_flip_packet_count.load(Ordering::Relaxed),
            "AGC submission progress"
        );
    }

    // Every DCB reaches the persistent command processor, including state-only
    // setup buffers. Register state is queue state and survives submissions;
    // dropping setup DCBs leaves later draw-only buffers with null shaders and
    // no render targets. Vulkan is initialized lazily only when a draw arrives.
    // The deprecated fixture path survives only behind the M2 regression test.
    //
    // Handed to the GPU worker rather than rendered here: hardware's submit
    // returns as soon as the GPU owns the buffer, and the title calls this from
    // its render thread while holding its own locks (see `submit_dcb_async`).
    // Everything above this line is guest-visible side effects the title expects
    // to have happened when submit returns — the sync writes, the events, the
    // flips — and they stay on this thread. Only the rendering moves.
    let gpu_queue = if queue == "ACB" {
        xps5x_core::subsystems::GpuQueue::AsyncCompute
    } else {
        xps5x_core::subsystems::GpuQueue::Graphics
    };
    ctx.kernel.diagnostics.record(
        ctx.guest_threads.current_thread(),
        xps5x_core::diagnostics::DiagnosticKind::GpuSubmit,
        queue,
        command_address,
        format!("dwords={dword_count}"),
    );
    ctx.gpu.submit(words, gpu_queue);

    0
}

/// Decode a command packet's identity: `(op, register)` from its Agc PM4
/// header (the inverse of [`pm4`]). Returns `None` if the header is unreadable.
fn packet_identity(ctx: &HleContext, command: u64) -> Option<(u32, u32)> {
    if command == 0 {
        return None;
    }
    let mut hdr = [0u8; 4];
    if !ctx.mem.read(command, &mut hdr) {
        return None;
    }
    let header = u32::from_le_bytes(hdr);
    Some(((header >> 8) & 0xFF, (header >> 2) & 0x3F))
}

/// `sceAgcQueueEndOfPipeActionPatchAddress(command, address)`: patch a
/// RELEASE_MEM packet's destination address (at `command + 12`).
fn hle_queue_eop_patch_address(ctx: &HleContext, args: &[u64]) -> u64 {
    let command = args.first().copied().unwrap_or(0);
    let address = args.get(1).copied().unwrap_or(0);
    match packet_identity(ctx, command) {
        Some((op, reg)) if op == IT_NOP && reg == R_RELEASE_MEM => {}
        _ => return SCE_ERROR_INVALID_ARGUMENT,
    }
    if !ctx.mem.write(command + 12, &address.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// `sceAgcDmaDataPatchSetDstAddressOrOffset(command, destination)`: patch a
/// DMA_DATA packet's destination address (at `command + 16`).
fn hle_dma_data_patch_address(ctx: &HleContext, args: &[u64]) -> u64 {
    let command = args.first().copied().unwrap_or(0);
    let destination = args.get(1).copied().unwrap_or(0);
    match packet_identity(ctx, command) {
        Some((op, reg)) if op == IT_NOP && reg == R_DMA_DATA => {}
        _ => return SCE_ERROR_INVALID_ARGUMENT,
    }
    if !ctx.mem.write(command + 16, &destination.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// `sceAgcWaitRegMemPatchAddress(command, address)`: patch the poll address of
/// a WAIT_REG_MEM packet — field offset 8 for the real WAIT_REG_MEM op, or 4
/// for a 32/64-bit wait-memory NOP.
fn hle_wait_reg_mem_patch_address(ctx: &HleContext, args: &[u64]) -> u64 {
    let command = args.first().copied().unwrap_or(0);
    let address = args.get(1).copied().unwrap_or(0);
    let Some((op, reg)) = packet_identity(ctx, command) else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    let field_offset = if op == IT_WAIT_REG_MEM {
        8
    } else if op == IT_NOP && (reg == R_WAIT_MEM32 || reg == R_WAIT_MEM64) {
        4
    } else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    if !ctx
        .mem
        .write(command + field_offset, &address.to_le_bytes())
    {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// Patch the destination address of a Gen5 WRITE_DATA packet. The packet
/// stores the 64-bit address in DWORDs 2 and 3 (`command + 8`).
fn hle_write_data_patch_address(ctx: &HleContext, args: &[u64]) -> u64 {
    let command = args.first().copied().unwrap_or(0);
    let address = args.get(1).copied().unwrap_or(0);
    match packet_identity(ctx, command) {
        Some((op, reg)) if op == IT_NOP && reg == R_WRITE_DATA => {}
        identity => {
            // Diagnose loudly: a mismatch here usually means the WRITE_DATA
            // packet was never emitted (e.g. an emitter rejected the call) or
            // the caller handed a payload pointer instead of the header.
            warn!(
                "sceAgcWriteDataPatchAddress: command {command:#x} is not a WRITE_DATA \
                 packet (identity {identity:?}, patch address {address:#x}) — EINVAL"
            );
            return SCE_ERROR_INVALID_ARGUMENT;
        }
    }
    if !ctx.mem.write(command + 8, &address.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// Read a NUL-terminated guest C-string (up to `max` bytes, excluding the NUL).
fn read_guest_cstring(ctx: &HleContext, addr: u64, max: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    for i in 0..max {
        let mut b = [0u8; 1];
        if !ctx.mem.read(addr + i as u64, &mut b) {
            return None;
        }
        if b[0] == 0 {
            return Some(out);
        }
        out.push(b[0]);
    }
    Some(out)
}

/// `sceAgcDcbPushMarker(dcb, markerString)`: emit a debug-marker packet whose
/// payload is the NUL-terminated marker string packed into DWORDs.
fn hle_dcb_push_marker(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let marker_addr = args.get(1).copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    let Some(marker) = read_guest_cstring(ctx, marker_addr, 4095) else {
        return 0;
    };
    let payload_dwords = (((marker.len() as u32) + 4) / 4).max(1);
    let packet_dwords = payload_dwords + 1;
    let Some(addr) = alloc_command_dwords(ctx, cb, u64::from(packet_dwords)) else {
        return 0;
    };
    if !ctx.mem.write(
        addr,
        &pm4(packet_dwords, IT_NOP, R_PUSH_MARKER).to_le_bytes(),
    ) {
        return 0;
    }
    for i in 0..payload_dwords {
        // Pack up to four marker bytes little-endian into this DWORD.
        let mut value = 0u32;
        for byte in 0..4u32 {
            let idx = (i * 4 + byte) as usize;
            if idx < marker.len() {
                value |= u32::from(marker[idx]) << (byte * 8);
            }
        }
        if !ctx
            .mem
            .write(addr + 4 + u64::from(i) * 4, &value.to_le_bytes())
        {
            return 0;
        }
    }
    addr
}

/// `sceAgcDriverInitResourceRegistration(memory, memorySize, ownerCount)`:
/// validate the resource-registration setup. (The registration state machine
/// itself is not modelled; the call succeeds so a title's init proceeds.)
fn hle_driver_init_resource_registration(ctx: &HleContext, args: &[u64]) -> u64 {
    use std::sync::atomic::Ordering;

    let memory = args.first().copied().unwrap_or(0);
    let memory_size = args.get(1).copied().unwrap_or(0);
    let owner_count = args.get(2).copied().unwrap_or(0);
    if memory == 0 || memory_size == 0 || owner_count == 0 || owner_count > u64::from(u32::MAX) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    ctx.kernel.agc_resource_owners.clear();
    ctx.kernel.agc_resources.clear();
    ctx.kernel.agc_next_owner.store(1, Ordering::Release);
    ctx.kernel.agc_next_resource.store(1, Ordering::Release);
    ctx.kernel.agc_default_owner.store(1, Ordering::Release);
    ctx.kernel
        .agc_resource_registration_max_owners
        .store(owner_count as u32, Ordering::Release);
    ctx.kernel
        .agc_resource_registration_initialized
        .store(true, Ordering::Release);
    debug!(
        "sceAgcDriverInitResourceRegistration: memory={memory:#x}, size={memory_size:#x}, owners={owner_count}"
    );
    0
}

/// `sceAgcDcbDrawIndexOffset(dcb, indexOffset, indexCount, flags)`: emit a
/// DRAW_INDEX_OFFSET_2 packet (5 DWORDs).
fn hle_dcb_draw_index_offset(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let index_offset = args.get(1).copied().unwrap_or(0) as u32;
    let index_count = args.get(2).copied().unwrap_or(0) as u32;
    let flags = args.get(3).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 5) else {
        return 0;
    };
    let ok = ctx
        .mem
        .write(addr, &pm4(5, IT_DRAW_INDEX_OFFSET_2, R_ZERO).to_le_bytes())
        && ctx.mem.write(addr + 4, &index_count.to_le_bytes())
        && ctx.mem.write(addr + 8, &index_offset.to_le_bytes())
        && ctx.mem.write(addr + 12, &index_count.to_le_bytes())
        && ctx
            .mem
            .write(addr + 16, &(flags & 0xE000_0001).to_le_bytes());
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcUnknownQj7QZpgr9Uw(dcb, ...)`: emit a single filler DWORD
/// (`0x8000_0000`, a Type-2 NOP header) into the command buffer.
fn hle_unknown_filler(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 1) else {
        return 0;
    };
    if !ctx.mem.write(addr, &0x8000_0000u32.to_le_bytes()) {
        return 0;
    }
    addr
}

/// `sceAgcGetDataPacketPayloadAddress(output, command, type)`: compute the
/// address of a command packet's payload and write it to `*output`. For
/// `type == 0` the payload follows the header (`command + 4`), unless the
/// packet is a max-length NOP (`header & 0x3FFF_0000 == 0x3FFF_0000`) → 0;
/// otherwise the payload is at `command + 8`.
fn hle_get_data_packet_payload_address(ctx: &HleContext, args: &[u64]) -> u64 {
    let output = args.first().copied().unwrap_or(0);
    let command = args.get(1).copied().unwrap_or(0);
    let ty = args.get(2).copied().unwrap_or(0) as i32;
    if output == 0 || command == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut payload = command + 8;
    if ty == 0 {
        let mut hdr = [0u8; 4];
        if !ctx.mem.read(command, &mut hdr) {
            return SCE_ERROR_MEMORY_FAULT;
        }
        let header = u32::from_le_bytes(hdr);
        payload = if header & 0x3FFF_0000 == 0x3FFF_0000 {
            0
        } else {
            command + 4
        };
    }
    if !ctx.mem.write(output, &payload.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// `sceAgcSet{Cx,Sh,Uc}RegIndirectPatchSetAddress(command, registers)`: bind the
/// register-block address into the indirect-patch command at `command + 8/12`.
fn hle_set_indirect_patch_address(ctx: &HleContext, args: &[u64]) -> u64 {
    let command = args.first().copied().unwrap_or(0);
    let registers = args.get(1).copied().unwrap_or(0);
    if command == 0 || registers == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !ctx
        .mem
        .write(command + 8, &(registers as u32).to_le_bytes())
        || !ctx
            .mem
            .write(command + 12, &((registers >> 32) as u32).to_le_bytes())
    {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// `sceAgcSet{Cx,Sh,Uc}RegIndirectPatchAddRegisters(command, registerCount)`:
/// accumulate `registerCount` into the patch command's count field (`+4`).
fn hle_add_indirect_patch_registers(ctx: &HleContext, args: &[u64]) -> u64 {
    let command = args.first().copied().unwrap_or(0);
    let register_count = args.get(1).copied().unwrap_or(0) as u32;
    if command == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut buf = [0u8; 4];
    if !ctx.mem.read(command + 4, &mut buf) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let updated = u32::from_le_bytes(buf).wrapping_add(register_count);
    if !ctx.mem.write(command + 4, &updated.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// Read a `u32` from guest memory, or 0 if unreadable.
fn read_u32_or_zero(ctx: &HleContext, addr: u64) -> u32 {
    let mut b = [0u8; 4];
    if ctx.mem.read(addr, &mut b) {
        u32::from_le_bytes(b)
    } else {
        0
    }
}

/// Read a `u64` from guest memory, or 0 if unreadable.
fn read_u64_or_zero(ctx: &HleContext, addr: u64) -> u64 {
    let mut b = [0u8; 8];
    if ctx.mem.read(addr, &mut b) {
        u64::from_le_bytes(b)
    } else {
        0
    }
}

/// `sceAgcCreateInterpolantMapping(registers, geometryShader, pixelShader)`:
/// build the 32 `SPI_PS_INPUT_CNTL` interpolant registers. Each slot below the
/// geometry shader's output-semantic count maps to interpolant `i`, with the
/// flat-shading bit (`0x400`) taken from the matching pixel-shader input
/// semantic (bit 22). Faithful to SharpEmu's layout.
fn hle_create_interpolant_mapping(ctx: &HleContext, args: &[u64]) -> u64 {
    let registers = args.first().copied().unwrap_or(0);
    let geometry = args.get(1).copied().unwrap_or(0);
    let pixel = args.get(2).copied().unwrap_or(0);
    if registers == 0 || geometry == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let output_semantics = read_u64_or_zero(ctx, geometry + SHADER_OUTPUT_SEMANTICS_OFFSET);
    let output_count = read_u32_or_zero(ctx, geometry + SHADER_NUM_OUTPUT_SEMANTICS_OFFSET);
    let input_semantics = if pixel != 0 {
        // The presence read also validates the pixel shader's semantics fields.
        let _ = read_u32_or_zero(ctx, pixel + SHADER_NUM_INPUT_SEMANTICS_OFFSET);
        read_u64_or_zero(ctx, pixel + SHADER_INPUT_SEMANTICS_OFFSET)
    } else {
        0
    };

    for i in 0..32u32 {
        let mut value = 0u32;
        if i < output_count && output_semantics != 0 {
            let mut flat = false;
            if pixel != 0 && input_semantics != 0 {
                let input_semantic = read_u32_or_zero(ctx, input_semantics + u64::from(i) * 4);
                flat = (input_semantic >> 22) & 0x1 != 0;
            }
            value = i | if flat { 0x400 } else { 0 };
        }
        let dst = registers + u64::from(i) * 8;
        if !ctx.mem.write(dst, &(SPI_PS_INPUT_CNTL0 + i).to_le_bytes())
            || !ctx.mem.write(dst + 4, &value.to_le_bytes())
        {
            return SCE_ERROR_MEMORY_FAULT;
        }
    }
    0
}

/// Relocate a self-relative pointer field: read the relative offset at
/// `field`, and (if non-zero) rewrite it as the absolute `field + rel`.
fn relocate_pointer_field(ctx: &HleContext, field: u64) -> bool {
    let mut buf = [0u8; 8];
    if !ctx.mem.read(field, &mut buf) {
        return false;
    }
    let rel = u64::from_le_bytes(buf);
    if rel == 0 {
        return true;
    }
    ctx.mem.write(field, &field.wrapping_add(rel).to_le_bytes())
}

const MAX_SHADER_METADATA_ENTRIES: usize = 16_384;

fn read_guest_u16(ctx: &HleContext, address: u64) -> Option<u16> {
    let mut bytes = [0u8; 2];
    ctx.mem
        .read(address, &mut bytes)
        .then(|| u16::from_le_bytes(bytes))
}

fn read_shader_mapped_data(ctx: &HleContext, header: u64) -> Option<xps5x_gpu::ShaderMappedData> {
    let user_data_address = read_u64_or_zero(ctx, header + SHADER_USER_DATA_OFFSET);
    let user_data = if user_data_address == 0 {
        None
    } else {
        let direct_address = read_u64_or_zero(ctx, user_data_address);
        let sharp_addresses = [
            read_u64_or_zero(ctx, user_data_address + 0x08),
            read_u64_or_zero(ctx, user_data_address + 0x10),
            read_u64_or_zero(ctx, user_data_address + 0x18),
            read_u64_or_zero(ctx, user_data_address + 0x20),
        ];
        let eud_size_dw = read_guest_u16(ctx, user_data_address + 0x28)?;
        let srt_size_dw = read_guest_u16(ctx, user_data_address + 0x2A)?;
        let direct_count = usize::from(read_guest_u16(ctx, user_data_address + 0x2C)?);
        let sharp_counts = [
            usize::from(read_guest_u16(ctx, user_data_address + 0x2E)?),
            usize::from(read_guest_u16(ctx, user_data_address + 0x30)?),
            usize::from(read_guest_u16(ctx, user_data_address + 0x32)?),
            usize::from(read_guest_u16(ctx, user_data_address + 0x34)?),
        ];
        if direct_count > MAX_SHADER_METADATA_ENTRIES
            || sharp_counts.iter().sum::<usize>() > MAX_SHADER_METADATA_ENTRIES
            || (direct_count != 0 && direct_address == 0)
            || sharp_counts
                .iter()
                .zip(sharp_addresses)
                .any(|(&count, address)| count != 0 && address == 0)
        {
            return None;
        }

        let mut direct_resource_offset = Vec::with_capacity(direct_count);
        for index in 0..direct_count {
            direct_resource_offset.push(read_guest_u16(ctx, direct_address + index as u64 * 2)?);
        }
        let mut sharp_resource_offset: [Vec<xps5x_gpu::ShaderSharp>; 4] =
            std::array::from_fn(|index| Vec::with_capacity(sharp_counts[index]));
        for table in 0..4 {
            for index in 0..sharp_counts[table] {
                let raw = read_guest_u16(ctx, sharp_addresses[table] + index as u64 * 2)?;
                sharp_resource_offset[table]
                    .push(xps5x_gpu::ShaderSharp::new(raw & 0x7fff, raw >> 15));
            }
        }
        Some(xps5x_gpu::ShaderUserData {
            direct_resource_offset,
            sharp_resource_offset,
            eud_size_dw,
            srt_size_dw,
        })
    };

    let semantics_address = read_u64_or_zero(ctx, header + SHADER_INPUT_SEMANTICS_OFFSET);
    let semantics_count =
        read_u32_or_zero(ctx, header + SHADER_NUM_INPUT_SEMANTICS_OFFSET) as usize;
    if semantics_count > MAX_SHADER_METADATA_ENTRIES
        || (semantics_count != 0 && semantics_address == 0)
    {
        return None;
    }
    let mut input_semantics = Vec::with_capacity(semantics_count);
    for index in 0..semantics_count {
        input_semantics.push(xps5x_gpu::ShaderSemantic {
            raw: read_u32_or_zero(ctx, semantics_address + index as u64 * 4),
        });
    }
    Some(xps5x_gpu::ShaderMappedData {
        user_data,
        input_semantics,
    })
}

/// Bind a shader's code allocation into the first two SH-register entries.
///
/// AGC compiler output deliberately stores zero in the program-base values.
/// `sceAgcCreateShader` is the relocation boundary that replaces those
/// placeholders. Later `SET_SH_REGS_INDIRECT` packets only point at this table,
/// so omitting the patch leaves every real draw bound to address zero.
fn patch_shader_program_registers(ctx: &HleContext, header: u64, code: u64) -> bool {
    let mut address = [0u8; 8];
    let mut shader_type = [0u8; 1];
    let mut register_count = [0u8; 1];
    if !ctx
        .mem
        .read(header + SHADER_SH_REGISTERS_OFFSET, &mut address)
        || !ctx.mem.read(header + SHADER_TYPE_OFFSET, &mut shader_type)
        || !ctx
            .mem
            .read(header + SHADER_NUM_SH_REGISTERS_OFFSET, &mut register_count)
    {
        return false;
    }
    let registers = u64::from_le_bytes(address);
    if registers == 0 || register_count[0] < 2 {
        return false;
    }

    let (expected_lo, expected_hi) = match shader_type[0] {
        0 => (COMPUTE_PGM_LO, COMPUTE_PGM_HI),
        1 => (SPI_SHADER_PGM_LO_PS, SPI_SHADER_PGM_HI_PS),
        2 | 6 => (SPI_SHADER_PGM_LO_ES, SPI_SHADER_PGM_HI_ES),
        4 => (SPI_SHADER_PGM_LO_GS, SPI_SHADER_PGM_HI_GS),
        7 => (SPI_SHADER_PGM_LO_LS, SPI_SHADER_PGM_HI_LS),
        _ => return false,
    };
    let mut lo_register = [0u8; 4];
    let mut hi_register = [0u8; 4];
    if !ctx.mem.read(registers, &mut lo_register)
        || !ctx.mem.read(registers + 8, &mut hi_register)
        || u32::from_le_bytes(lo_register) != expected_lo
        || u32::from_le_bytes(hi_register) != expected_hi
    {
        return false;
    }

    let lo_value = ((code >> 8) as u32).to_le_bytes();
    let hi_value = ((code >> 40) as u32 & 0xff).to_le_bytes();
    ctx.mem.write(registers + 4, &lo_value) && ctx.mem.write(registers + 12, &hi_value)
}

/// `sceAgcCreateShader(destination, header, code)`: validate the shader header
/// (magic + version), relocate its self-relative pointer fields to absolute,
/// bind the code address, relocate the user-data table, and publish the shader
/// object pointer to `*destination`. Faithful to SharpEmu's layout.
fn hle_create_shader(ctx: &HleContext, args: &[u64]) -> u64 {
    let destination = args.first().copied().unwrap_or(0);
    let header = args.get(1).copied().unwrap_or(0);
    let code = args.get(2).copied().unwrap_or(0);
    if header == 0 || code == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    // Validate the file header + version.
    let (mut fh, mut ver) = ([0u8; 4], [0u8; 4]);
    if !ctx.mem.read(header, &mut fh) || !ctx.mem.read(header + 4, &mut ver) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    if u32::from_le_bytes(fh) != SHADER_FILE_HEADER || u32::from_le_bytes(ver) != SHADER_VERSION {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    // Relocate the header's pointer fields, then bind the code address.
    let ok = relocate_pointer_field(ctx, header + SHADER_CX_REGISTERS_OFFSET)
        && relocate_pointer_field(ctx, header + SHADER_SH_REGISTERS_OFFSET)
        && relocate_pointer_field(ctx, header + SHADER_USER_DATA_OFFSET)
        && relocate_pointer_field(ctx, header + SHADER_SPECIALS_OFFSET)
        && relocate_pointer_field(ctx, header + SHADER_INPUT_SEMANTICS_OFFSET)
        && relocate_pointer_field(ctx, header + SHADER_OUTPUT_SEMANTICS_OFFSET)
        && ctx
            .mem
            .write(header + SHADER_CODE_OFFSET, &code.to_le_bytes());
    if !ok {
        return SCE_ERROR_MEMORY_FAULT;
    }
    // Relocate the user-data table's own pointer fields, if present.
    let mut ud = [0u8; 8];
    if !ctx.mem.read(header + SHADER_USER_DATA_OFFSET, &mut ud) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let user_data = u64::from_le_bytes(ud);
    if user_data != 0 {
        // ShaderUserData's pointer fields, per Kyty's `GraphicsCreateShader`
        // (Graphics.cpp:1460-1464, MIT © InoriRus): `direct_resource_offset`
        // at +0x00 and `sharp_resource_offset[0..4]` at +0x08..+0x20. Missing
        // +0x00 was a measured Minecraft crash: the field kept its
        // self-relative value (0x48) and the title's material system
        // dereferenced it as a pointer — a fault reading address 0x48 on the
        // render thread, with nothing connecting it back to shader creation.
        for off in [0x00u64, 0x08, 0x10, 0x18, 0x20] {
            if !relocate_pointer_field(ctx, user_data + off) {
                return SCE_ERROR_MEMORY_FAULT;
            }
        }
    }
    if !patch_shader_program_registers(ctx, header, code) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let Some(mapped_data) = read_shader_mapped_data(ctx, header) else {
        return SCE_ERROR_MEMORY_FAULT;
    };
    ctx.gpu.map_shader_metadata(code, mapped_data);
    // Publish the shader object pointer.
    if !ctx.mem.write(destination, &header.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// Copy an 8-byte shader register (offset:u32, value:u32) from `src` to `dst`.
fn copy_shader_register(ctx: &HleContext, src: u64, dst: u64) -> bool {
    let mut buf = [0u8; 8];
    ctx.mem.read(src, &mut buf) && ctx.mem.write(dst, &buf)
}

/// `sceAgcCreatePrimState(cxRegisters, ucRegisters, hullShader, geometryShader,
/// primitiveType)`: assemble the primitive-state register block by copying
/// stage-enable / prim-type / GE registers out of the geometry shader's
/// "specials" table, then writing the VGT primitive type. Faithful to
/// SharpEmu's offsets.
fn hle_create_prim_state(ctx: &HleContext, args: &[u64]) -> u64 {
    let cx = args.first().copied().unwrap_or(0);
    let uc = args.get(1).copied().unwrap_or(0);
    let hull = args.get(2).copied().unwrap_or(0);
    let geometry = args.get(3).copied().unwrap_or(0);
    let primitive_type = args.get(4).copied().unwrap_or(0) as u32;
    if cx == 0 || uc == 0 || hull != 0 || geometry == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    // The geometry shader points at a "specials" register table.
    let mut sp = [0u8; 8];
    if !ctx.mem.read(geometry + SHADER_SPECIALS_OFFSET, &mut sp) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let specials = u64::from_le_bytes(sp);
    if specials == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let ok = copy_shader_register(ctx, specials + SPECIAL_VGT_SHADER_STAGES_EN_OFFSET, cx)
        && copy_shader_register(ctx, specials + SPECIAL_VGT_GS_OUT_PRIM_TYPE_OFFSET, cx + 8)
        && copy_shader_register(ctx, specials + SPECIAL_GE_CNTL_OFFSET, uc)
        && copy_shader_register(ctx, specials + SPECIAL_GE_USER_VGPR_EN_OFFSET, uc + 8)
        && ctx.mem.write(uc + 16, &VGT_PRIMITIVE_TYPE.to_le_bytes())
        && ctx.mem.write(uc + 20, &primitive_type.to_le_bytes());
    if !ok {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// Supported Agc register-defaults versions (see `sceAgcInit`).
fn is_supported_version(version: u32) -> bool {
    // Version 12 is used by Minecraft's newer Gen5 AGC runtime. Register
    // defaults are layout-compatible across the versions served here; the
    // returned table is immutable and version-independent.
    matches!(version, 7 | 8 | 10 | 12 | 13)
}

/// `sceAgcInit(state, version)`: initialize the Agc state for a supported
/// register-defaults version.
fn hle_init(_ctx: &HleContext, args: &[u64]) -> u64 {
    let state = args.first().copied().unwrap_or(0);
    let version = args.get(1).copied().unwrap_or(0) as u32;
    if state == 0 || !is_supported_version(version) {
        warn!(
            state,
            version,
            args = ?args,
            "sceAgcInit rejected unsupported arguments"
        );
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    0
}

/// Shared body for `sceAgcDcbSet{Cx,Sh,Uc}RegistersIndirect(dcb, registers,
/// registerCount)`: emit a 4-DWORD packet (count + registers address lo/hi)
/// tagged with the register-space discriminator `packet_register`.
fn dcb_set_registers_indirect(ctx: &HleContext, args: &[u64], packet_register: u32) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let registers = args.get(1).copied().unwrap_or(0);
    let register_count = args.get(2).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 4) else {
        return 0;
    };
    let ok = ctx
        .mem
        .write(addr, &pm4(4, IT_NOP, packet_register).to_le_bytes())
        && ctx.mem.write(addr + 4, &register_count.to_le_bytes())
        && ctx.mem.write(addr + 8, &(registers as u32).to_le_bytes())
        && ctx
            .mem
            .write(addr + 12, &((registers >> 32) as u32).to_le_bytes());
    if !ok {
        return 0;
    }
    addr
}

fn hle_dcb_set_cx_regs_indirect(ctx: &HleContext, args: &[u64]) -> u64 {
    dcb_set_registers_indirect(ctx, args, R_CX_REGS_INDIRECT)
}

fn hle_dcb_set_sh_regs_indirect(ctx: &HleContext, args: &[u64]) -> u64 {
    dcb_set_registers_indirect(ctx, args, R_SH_REGS_INDIRECT)
}

fn hle_dcb_set_uc_regs_indirect(ctx: &HleContext, args: &[u64]) -> u64 {
    dcb_set_registers_indirect(ctx, args, R_UC_REGS_INDIRECT)
}

/// `sceAgcAcbDispatchIndirect(acb, argumentsAddress, modifier)`: emit an
/// indirect dispatch packet (4 DWORDs) with the split arguments address +
/// initiator. Mirrors SharpEmu's `AcbDispatchIndirect`.
fn hle_acb_dispatch_indirect(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let arguments = args.get(1).copied().unwrap_or(0);
    let modifier = args.get(2).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 4) else {
        return 0;
    };
    let initiator = (modifier & 0xA038) | 0x41;
    let ok = ctx
        .mem
        .write(addr, &pm4(4, IT_DISPATCH_INDIRECT, R_ZERO).to_le_bytes())
        && ctx.mem.write(addr + 4, &(arguments as u32).to_le_bytes())
        && ctx
            .mem
            .write(addr + 8, &((arguments >> 32) as u32).to_le_bytes())
        && ctx.mem.write(addr + 12, &initiator.to_le_bytes());
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDcbWriteData(dcb, destination, cachePolicy, destinationAddress,
/// dataAddress, dwordCount, increment, writeConfirm)`: emit a WRITE_DATA packet
/// (`dwordCount + 4` DWORDs) copying `dwordCount` inline DWORDs from
/// `dataAddress`. `increment` (arg7) and `writeConfirm` (arg8) arrive on the
/// stack — captured by the runtime dispatch into `args[6]` / `args[7]` (SysV
/// arg7 at `[Rsp+8]`, arg8 at `[Rsp+16]`). Mirrors SharpEmu's `DcbWriteData`;
/// also serves `sceAgcAcbWriteData` (an alias). Returns the command address, or
/// 0 on failure.
///
/// A **null `destinationAddress` is legal**: Minecraft (PPSA17221) builds
/// WRITE_DATA template packets with a placeholder destination and binds the
/// real address afterwards via `sceAgcWriteDataPatchAddress` (often after
/// memcpy'ing the packet into a submission ring). Rejecting the placeholder
/// meant no packet was ever emitted, and the later patch then failed its
/// header check with `EINVAL`. A null `dataAddress` zero-fills the payload,
/// matching the reference's null-values handling in the SH-range writer.
fn hle_dcb_write_data(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let destination = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let cache_policy = (args.get(2).copied().unwrap_or(0) & 0xFF) as u32;
    let destination_address = args.get(3).copied().unwrap_or(0);
    let data_address = args.get(4).copied().unwrap_or(0);
    let dword_count = args.get(5).copied().unwrap_or(0) as u32;
    let increment = (args.get(6).copied().unwrap_or(0) & 0xFF) as u32;
    let write_confirm = (args.get(7).copied().unwrap_or(0) & 0xFF) as u32;
    if cb == 0 || dword_count > 0x3FFD {
        return 0;
    }
    let packet_dwords = dword_count + 4;
    let Some(addr) = alloc_command_dwords(ctx, cb, u64::from(packet_dwords)) else {
        return 0;
    };
    let control = destination | (cache_policy << 8) | (increment << 16) | (write_confirm << 24);
    let ok = ctx.mem.write(
        addr,
        &pm4(packet_dwords, IT_NOP, R_WRITE_DATA).to_le_bytes(),
    ) && ctx.mem.write(addr + 4, &control.to_le_bytes())
        && ctx
            .mem
            .write(addr + 8, &(destination_address as u32).to_le_bytes())
        && ctx.mem.write(
            addr + 12,
            &((destination_address >> 32) as u32).to_le_bytes(),
        );
    if !ok {
        return 0;
    }
    for index in 0..u64::from(dword_count) {
        // A null data pointer zero-fills the payload (the caller will patch
        // or rewrite it); an unreadable non-null source is a real failure.
        let mut buf = [0u8; 4];
        if data_address != 0 && !ctx.mem.read(data_address + index * 4, &mut buf) {
            return 0;
        }
        if !ctx.mem.write(addr + 16 + index * 4, &buf) {
            return 0;
        }
    }
    addr
}

/// `sceAgcDcbWaitRegMem(dcb, size, compareFunction, operation, cachePolicy,
/// address, reference, mask, pollCycles)`: emit a WAIT_REG_MEM / conditional
/// poll packet. `reference` (arg7), `mask` (arg8) and `pollCycles` (arg9) are
/// stack args → `args[6]` / `args[7]` / `args[8]`. Mirrors SharpEmu's
/// `DcbWaitRegMem`. Returns the command address, or 0 on failure.
fn hle_dcb_wait_reg_mem(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let size = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let compare = (args.get(2).copied().unwrap_or(0) & 0xFF) as u32;
    let operation = (args.get(3).copied().unwrap_or(0) & 0xFF) as u32;
    let cache_policy = (args.get(4).copied().unwrap_or(0) & 0xFF) as u32;
    let address = args.get(5).copied().unwrap_or(0);
    let reference = args.get(6).copied().unwrap_or(0);
    let mask = args.get(7).copied().unwrap_or(0);
    let poll_cycles = args.get(8).copied().unwrap_or(0) as u32;
    if cb == 0 || size > 1 || compare > 7 || operation > 4 || cache_policy > 3 {
        return 0;
    }
    let standard_wait = operation == 2 || operation == 3;
    let packet_dwords = if standard_wait {
        7
    } else if size == 0 {
        6
    } else {
        9
    };
    let packet_register = if size == 0 {
        R_WAIT_MEM32
    } else {
        R_WAIT_MEM64
    };
    let Some(addr) = alloc_command_dwords(ctx, cb, u64::from(packet_dwords)) else {
        return 0;
    };
    let w = |off: u64, v: u32| ctx.mem.write(addr + off, &v.to_le_bytes());
    let ok = if standard_wait {
        w(0, pm4(packet_dwords, IT_WAIT_REG_MEM, R_ZERO))
            && w(4, compare | ((operation & 1) << 8))
            && w(8, address as u32)
            && w(12, (address >> 32) as u32)
            && w(16, reference as u32)
            && w(20, mask as u32)
            && w(24, poll_cycles / 40)
    } else {
        let head = w(0, pm4(packet_dwords, IT_NOP, packet_register))
            && w(4, address as u32)
            && w(8, (address >> 32) as u32)
            && w(12, mask as u32);
        if !head {
            false
        } else if size == 0 {
            w(16, compare | (operation << 8)) && w(20, reference as u32)
        } else {
            w(16, (mask >> 32) as u32)
                && w(20, reference as u32)
                && w(24, (reference >> 32) as u32)
                && w(28, compare | (operation << 8))
                && w(32, poll_cycles / 40)
        }
    };
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDcbDmaData(dcb, destination, destinationCachePolicy, source,
/// destinationAddress, sourceCachePolicy, control4, sourceAddress, byteCount,
/// control7, control8, control9)`: emit a DMA_DATA packet (8 DWORDs). Args 7–12
/// are on the stack → `args[6]`..`args[11]`. Mirrors SharpEmu's `DcbDmaData`.
/// Returns the command address, or 0 on failure.
fn hle_dcb_dma_data(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let destination = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let dst_cache = (args.get(2).copied().unwrap_or(0) & 0xFF) as u32;
    let source = (args.get(3).copied().unwrap_or(0) & 0xFF) as u32;
    let destination_address = args.get(4).copied().unwrap_or(0);
    let src_cache = (args.get(5).copied().unwrap_or(0) & 0xFF) as u32;
    let control4 = (args.get(6).copied().unwrap_or(0) & 0xFF) as u32;
    let source_address = args.get(7).copied().unwrap_or(0);
    let byte_count = args.get(8).copied().unwrap_or(0) as u32;
    let control7 = (args.get(9).copied().unwrap_or(0) & 0xFF) as u32;
    let control8 = (args.get(10).copied().unwrap_or(0) & 0xFF) as u32;
    let control9 = (args.get(11).copied().unwrap_or(0) & 0xFF) as u32;
    if cb == 0 || byte_count == 0 || byte_count & 3 != 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 8) else {
        return 0;
    };
    let control0 = destination | (dst_cache << 8) | (source << 16) | (src_cache << 24);
    let control_ext = control4 | (control7 << 8) | (control8 << 16) | (control9 << 24);
    let ok = ctx
        .mem
        .write(addr, &pm4(8, IT_NOP, R_DMA_DATA).to_le_bytes())
        && ctx.mem.write(addr + 4, &control0.to_le_bytes())
        && ctx.mem.write(addr + 8, &control_ext.to_le_bytes())
        && ctx.mem.write(addr + 12, &byte_count.to_le_bytes())
        && ctx.mem.write(addr + 16, &destination_address.to_le_bytes())
        && ctx.mem.write(addr + 24, &source_address.to_le_bytes());
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDcbAcquireMem(dcb, engine, cbDbOp, gcrControl, baseAddress, sizeBytes,
/// pollCycles)`: emit an ACQUIRE_MEM packet (8 DWORDs). `pollCycles` (arg7) is a
/// stack arg → `args[6]`. `sizeBytes == u64::MAX` means "no size". Mirrors
/// SharpEmu's `DcbAcquireMem`. Returns the command address, or 0 on failure.
fn hle_dcb_acquire_mem(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let engine = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let cb_db_op = args.get(2).copied().unwrap_or(0) as u32;
    let gcr_control = args.get(3).copied().unwrap_or(0) as u32;
    let base_address = args.get(4).copied().unwrap_or(0);
    let size_bytes = args.get(5).copied().unwrap_or(0);
    let poll_cycles = args.get(6).copied().unwrap_or(0) as u32;
    let no_size = size_bytes == u64::MAX;
    if cb == 0
        || engine > 1
        || (!no_size && size_bytes & 0xFF != 0)
        || (!no_size && size_bytes >> 40 != 0)
        || base_address & 0xFF != 0
        || base_address >> 40 != 0
    {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 8) else {
        return 0;
    };
    let size_field = if no_size { 0 } else { (size_bytes >> 8) as u32 };
    let w = |off: u64, v: u32| ctx.mem.write(addr + off, &v.to_le_bytes());
    let ok = w(0, pm4(8, IT_NOP, R_ACQUIRE_MEM))
        && w(4, (engine << 31) | cb_db_op)
        && w(8, size_field)
        && w(12, 0)
        && w(16, (base_address >> 8) as u32)
        && w(20, 0)
        && w(24, poll_cycles / 40)
        && w(28, gcr_control);
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcAcbAcquireMem(acb, gcrControl, baseAddress, sizeBytes, pollCycles)`:
/// emit the async-compute ACQUIRE_MEM packet (8 DWORDs). Unlike the DCB form
/// there is no engine/cbDbOp pair — DWORD1 is fixed `0x8000_0000`.
/// `sizeBytes == u64::MAX` means "no size". Ported from SharpEmu
/// `AcbAcquireMem` (AgcExports.cs L1095-1131). Returns the command address,
/// or 0 on failure.
fn hle_acb_acquire_mem(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let gcr_control = args.get(1).copied().unwrap_or(0) as u32;
    let base_address = args.get(2).copied().unwrap_or(0);
    let size_bytes = args.get(3).copied().unwrap_or(0);
    let poll_cycles = args.get(4).copied().unwrap_or(0) as u32;
    let no_size = size_bytes == u64::MAX;
    if cb == 0
        || (!no_size && size_bytes & 0xFF != 0)
        || (!no_size && size_bytes >> 40 != 0)
        || base_address & 0xFF != 0
        || base_address >> 40 != 0
    {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 8) else {
        return 0;
    };
    let size_field = if no_size { 0 } else { (size_bytes >> 8) as u32 };
    let w = |off: u64, v: u32| ctx.mem.write(addr + off, &v.to_le_bytes());
    let ok = w(0, pm4(8, IT_NOP, R_ACQUIRE_MEM))
        && w(4, 0x8000_0000)
        && w(8, size_field)
        && w(12, 0)
        && w(16, (base_address >> 8) as u32)
        && w(20, 0)
        && w(24, poll_cycles / 40)
        && w(28, gcr_control);
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcAcbWaitRegMem(acb, size, compareFunction, cachePolicy, address,
/// reference, mask, pollCycles)`: emit the async-compute wait-memory packet.
/// `mask` (arg7) and `pollCycles` (arg8) are stack args → `args[6]`/`args[7]`.
/// Unlike the DCB form there is no `operation` argument (no standard
/// WAIT_REG_MEM branch): `size == 0` → 6-DWORD 32-bit wait (`R_WAIT_MEM32`),
/// `size == 1` → 9-DWORD 64-bit wait (`R_WAIT_MEM64`). Ported from SharpEmu
/// `AcbWaitRegMem` (AgcExports.cs L1133-1186). Returns the command address,
/// or 0 on failure.
fn hle_acb_wait_reg_mem(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let size = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let compare = (args.get(2).copied().unwrap_or(0) & 0xFF) as u32;
    let cache_policy = (args.get(3).copied().unwrap_or(0) & 0xFF) as u32;
    let address = args.get(4).copied().unwrap_or(0);
    let reference = args.get(5).copied().unwrap_or(0);
    let mask = args.get(6).copied().unwrap_or(0);
    let poll_cycles = args.get(7).copied().unwrap_or(0) as u32;
    if cb == 0 || size > 1 || compare > 7 || cache_policy > 3 {
        return 0;
    }
    let (packet_dwords, packet_register) = if size == 0 {
        (6, R_WAIT_MEM32)
    } else {
        (9, R_WAIT_MEM64)
    };
    let Some(addr) = alloc_command_dwords(ctx, cb, packet_dwords) else {
        return 0;
    };
    let w = |off: u64, v: u32| ctx.mem.write(addr + off, &v.to_le_bytes());
    let head = w(0, pm4(packet_dwords as u32, IT_NOP, packet_register))
        && w(4, address as u32)
        && w(8, (address >> 32) as u32)
        && w(12, mask as u32);
    let ok = if !head {
        false
    } else if size == 0 {
        w(16, compare) && w(20, reference as u32)
    } else {
        w(16, (mask >> 32) as u32)
            && w(20, reference as u32)
            && w(24, (reference >> 32) as u32)
            && w(28, compare)
            && w(32, poll_cycles / 40)
    };
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcAcbDmaData(acb, sourceSelector, destinationSelector,
/// destinationAddress, sourceOrImmediate, byteCount)`: emit the async-compute
/// DMA_DATA packet (7 DWORDs — the DCB form is 8). `sourceOrImmediate` and
/// `byteCount` arrive on the stack → `args[6]`/`args[7]` (SharpEmu reads them
/// at `Rsp+8`/`Rsp+16`). Layout: header, destination u64, source-or-immediate
/// u64, byteCount, `sourceSelector | destinationSelector << 8`. Ported from
/// SharpEmu `AcbDmaData` (AgcExports.cs L1966-1997). Returns the command
/// address, or 0 on failure.
fn hle_acb_dma_data(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let source_selector = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let destination_selector = (args.get(2).copied().unwrap_or(0) & 0xFF) as u32;
    let destination_address = args.get(3).copied().unwrap_or(0);
    let source_or_immediate = args.get(6).copied().unwrap_or(0);
    let byte_count = args.get(7).copied().unwrap_or(0) as u32;
    if cb == 0 || byte_count == 0 || byte_count > 256 * 1024 * 1024 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 7) else {
        return 0;
    };
    let ok = ctx
        .mem
        .write(addr, &pm4(7, IT_NOP, R_DMA_DATA).to_le_bytes())
        && ctx.mem.write(addr + 4, &destination_address.to_le_bytes())
        && ctx.mem.write(addr + 12, &source_or_immediate.to_le_bytes())
        && ctx.mem.write(addr + 20, &byte_count.to_le_bytes())
        && ctx.mem.write(
            addr + 24,
            &(source_selector | (destination_selector << 8)).to_le_bytes(),
        );
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcAcbCopyData(...)`: NO reference implementation exists — SharpEmu and
/// Kyty Gen5 both lack it; aerolib only names the NID (`qzMN2XKGA4k`). Mirror
/// SharpEmu's ACB⇒DCB aliasing pattern for data packets
/// (`AcbWriteData(ctx) => DcbWriteData(ctx)`, AgcExports.cs L1188-1194) and
/// emit the same GUESSED standard PM4 COPY_DATA as [`hle_dcb_copy_data`];
/// warns once.
fn hle_acb_copy_data(ctx: &HleContext, args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if warn_once(&WARNED) {
        warn!(
            "sceAgcAcbCopyData: no reference implementation — aliased to the guessed \
             sceAgcDcbCopyData emission"
        );
    }
    hle_dcb_copy_data(ctx, args)
}

/// `sceAgcCbSetShRegistersDirect(cb, registers, count)`: read `count`
/// `(offset u32, value u32)` pairs from the guest `registers` array, sort them
/// by offset, coalesce runs of consecutive offsets, and emit one SET_SH_REG
/// packet per run (`header, offset & 0xFFFF, values…`). Returns the address of
/// the FIRST packet emitted, or 0 on failure. Ported from SharpEmu
/// `CbSetShRegistersDirect` (AgcExports.cs L970-1040).
fn hle_cb_set_sh_registers_direct(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let registers_addr = args.get(1).copied().unwrap_or(0);
    let count = args.get(2).copied().unwrap_or(0) as u32;
    if count == 0 || cb == 0 || registers_addr == 0 || count > 4096 {
        return 0;
    }
    let mut registers = Vec::with_capacity(count as usize);
    for index in 0..u64::from(count) {
        let entry = registers_addr + index * 8;
        let mut offset = [0u8; 4];
        let mut value = [0u8; 4];
        if !ctx.mem.read(entry, &mut offset) || !ctx.mem.read(entry + 4, &mut value) {
            return 0;
        }
        registers.push((u32::from_le_bytes(offset), u32::from_le_bytes(value)));
    }
    registers.sort_by_key(|&(offset, _)| offset);
    let mut first_command = 0u64;
    let mut start = 0usize;
    while start < registers.len() {
        let mut end = start + 1;
        while end < registers.len() && registers[end].0 == registers[end - 1].0 + 1 {
            end += 1;
        }
        let value_count = (end - start) as u32;
        let packet_dwords = value_count + 2;
        let Some(addr) = alloc_command_dwords(ctx, cb, u64::from(packet_dwords)) else {
            return 0;
        };
        if !ctx.mem.write(
            addr,
            &pm4(packet_dwords, IT_SET_SH_REG, R_ZERO).to_le_bytes(),
        ) || !ctx
            .mem
            .write(addr + 4, &(registers[start].0 & 0xFFFF).to_le_bytes())
        {
            return 0;
        }
        for (slot, &(_, value)) in registers[start..end].iter().enumerate() {
            if !ctx
                .mem
                .write(addr + 8 + slot as u64 * 4, &value.to_le_bytes())
            {
                return 0;
            }
        }
        if first_command == 0 {
            first_command = addr;
        }
        start = end;
    }
    first_command
}

/// `sceAgcCbDispatchGetSize()`: BYTES a [`hle_cb_dispatch`] packet occupies —
/// 5 DWORDs = 20. No reference implements this NID (aerolib names it only),
/// but SharpEmu's GetSize convention returns bytes for a fixed packet
/// (`DcbDrawIndexIndirectGetSize` returns `5 * sizeof(uint)` for its own
/// 5-DWORD packet, AgcExports.cs L1566-1575), and the writer this must match
/// is in this file.
fn hle_cb_dispatch_get_size(_ctx: &HleContext, _args: &[u64]) -> u64 {
    5 * 4
}

/// `sceAgcCbNopGetSize(dwordCount)`: BYTES a [`hle_cb_nop`] packet of
/// `dwordCount` total DWORDs occupies — `dwordCount * 4`. Signature inferred
/// from the writer (its only size input is the DWORD count); byte units follow
/// SharpEmu's GetSize convention (e.g. `DcbDmaDataGetSize`, AgcExports.cs
/// L1955-1964). Warns once about the inference.
fn hle_cb_nop_get_size(_ctx: &HleContext, args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let dword_count = args.first().copied().unwrap_or(0) as u32;
    if warn_once(&WARNED) {
        warn!(
            "sceAgcCbNopGetSize: signature inferred (dwordCount in the first argument \
             register) — returning {} bytes for dwordCount={dword_count}",
            u64::from(dword_count) * 4
        );
    }
    u64::from(dword_count) * 4
}

/// `sceAgcDcbDrawIndexIndirect(dcb, dataOffset, modifier)`: emit a 5-DWORD
/// DRAW_INDEX_INDIRECT packet — `header, dataOffset, 0, 0, modifier` — drawing
/// from the argument buffer set by `sceAgcDcbSetBaseIndirectArgs`. Ported from
/// SharpEmu `DcbDrawIndexIndirect` (AgcExports.cs L1539-1564). Returns the
/// command address, or 0 on failure.
fn hle_dcb_draw_index_indirect(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let data_offset = args.get(1).copied().unwrap_or(0) as u32;
    let modifier = args.get(2).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 5) else {
        return 0;
    };
    let w = |off: u64, v: u32| ctx.mem.write(addr + off, &v.to_le_bytes());
    let ok = w(0, pm4(5, IT_DRAW_INDEX_INDIRECT, R_ZERO))
        && w(4, data_offset)
        && w(8, 0)
        && w(12, 0)
        && w(16, modifier);
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDcbDrawIndirect(dcb, dataOffset, modifier)`: emit a 5-DWORD
/// DRAW_INDIRECT packet — `header, dataOffset, 0, 0, modifier` — the
/// non-indexed sibling of [`hle_dcb_draw_index_indirect`] (same shape, opcode
/// `IT_DRAW_INDIRECT` 0x24 instead of 0x25). SharpEmu exports no builder for
/// it, but its submitted-DCB parser consumes `ItDrawIndirect` (0x24) with the
/// argument-buffer `dataOffset` at `+4` and packet length >= 5 (AgcExports.cs
/// L5216-5227), which this emission matches. Draws from the argument buffer
/// bound by `sceAgcDcbSetBaseIndirectArgs`. Measured Until Dawn + Dragon Ball
/// Sparking Zero import (NID `1q1titRBL6o`). Returns the command address, or
/// 0 on failure.
fn hle_dcb_draw_indirect(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let data_offset = args.get(1).copied().unwrap_or(0) as u32;
    let modifier = args.get(2).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 5) else {
        return 0;
    };
    let w = |off: u64, v: u32| ctx.mem.write(addr + off, &v.to_le_bytes());
    let ok = w(0, pm4(5, IT_DRAW_INDIRECT, R_ZERO))
        && w(4, data_offset)
        && w(8, 0)
        && w(12, 0)
        && w(16, modifier);
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDcbJump(dcb, target, sizeDwords)`: chain execution into another
/// command buffer — emit a 4-DWORD INDIRECT_BUFFER packet: `header, target
/// lo32, target hi (16 bits), sizeDwords & 0xFFFFF`. Ported dword-exact from
/// SharpEmu `DcbJump` (AgcExports.cs, NID `xSAR0LTcRKM`, GPL-2.0).
///
/// CONSUMPTION GAP: the packet is emitted faithfully, but neither in-tree
/// command processor follows INDIRECT_BUFFER chains yet — kyty-graphics
/// declares `pm4::IT_INDIRECT_BUFFER` (0x3F) with no handler in `run.rs`, and
/// `xps5x_gpu::agc::decode_submission` records it as an opaque packet. A title
/// that links its frame through Jump/Branch will submit a head buffer whose
/// tail work lives in unexecuted chained buffers.
fn hle_dcb_jump(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let target = args.get(1).copied().unwrap_or(0);
    let size_dwords = args.get(2).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 4) else {
        return 0;
    };
    let w = |off: u64, v: u32| ctx.mem.write(addr + off, &v.to_le_bytes());
    let ok = w(0, pm4(4, IT_INDIRECT_BUFFER, R_ZERO))
        && w(4, target as u32)
        && w(8, ((target >> 32) & 0xFFFF) as u32)
        && w(12, size_dwords & 0xF_FFFF);
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcCbBranch(cb, target, sizeDwords)`: branch into another command
/// buffer. SharpEmu implements only `DcbJump`; Branch is modeled on it — the
/// same 4-DWORD INDIRECT_BUFFER chain packet (both transfer the command
/// processor to `target` for `sizeDwords`) — so the assumption is that any
/// Jump/Branch distinction (e.g. a return/chain flag) is not encoded in fields
/// this emission fills. Measured Until Dawn + Dragon Ball Sparking Zero import
/// (NID `w1KFAHVqpaU`). Same consumption gap as [`hle_dcb_jump`]: chained
/// buffers are not yet executed by the in-tree command processors.
fn hle_cb_branch(ctx: &HleContext, args: &[u64]) -> u64 {
    hle_dcb_jump(ctx, args)
}

/// `sceAgcSetPacketPredication(...)`: global predication toggle on a packet.
/// SharpEmu `SetPacketPredication` (AgcExports.cs, NID `w6Dj1VJt5qY`) is an
/// explicit OK no-op — "a no-op is safe for rendering" (predication only
/// culls work; never predicating draws everything). Ported as such.
fn hle_set_packet_predication(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        packet = format_args!("{:#x}", args.first().copied().unwrap_or(0)),
        "sceAgcSetPacketPredication: accepted (predication not applied — SharpEmu-parity no-op)"
    );
    0
}

/// `sceAgcSetCxRegIndirectPatchSetNumRegisters(command, numRegisters)`: SET
/// (overwrite) the register count of an indirect set-registers packet. No
/// reference implements this export (absent from SharpEmu/Kyty); the field
/// position is established by this file's own emitter — the indirect packet
/// is `header, count@+4, addrLo@+8, addrHi@+12` (`dcb_set_registers_indirect`)
/// — and by the measured siblings `...PatchSetAddress` (writes `+8/+12`) and
/// `...PatchAddRegisters` (accumulates into `+4`). Set = overwrite of the
/// same `+4` count field. Measured Until Dawn + Dragon Ball Sparking Zero
/// import (NID `whb1RL7K4Ss`).
fn hle_set_indirect_patch_set_num_registers(ctx: &HleContext, args: &[u64]) -> u64 {
    let command = args.first().copied().unwrap_or(0);
    let register_count = args.get(1).copied().unwrap_or(0) as u32;
    if command == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !ctx.mem.write(command + 4, &register_count.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    0
}

/// `sceAgcDcbSetMarker(dcb, markerString)`: one-shot debug annotation (the
/// non-nesting sibling of Push/PopMarker). No reference implements this
/// export (absent from SharpEmu/Kyty), so the exact discriminator is unknown;
/// the marker string is preserved in the stream as a plain NOP packet
/// (`R_ZERO`, which every consumer skips) with the same DWORD-packed payload
/// encoding as `hle_dcb_push_marker` — deliberately NOT `R_PUSH_MARKER`, so a
/// future marker-stack consumer never sees an unbalanced push. Measured A
/// Plague Tale Requiem import (NID `QhCbS4X9Rl8`). Returns the command
/// address, or 0 on failure.
fn hle_dcb_set_marker(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let marker_addr = args.get(1).copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    let Some(marker) = read_guest_cstring(ctx, marker_addr, 4095) else {
        return 0;
    };
    let payload_dwords = (((marker.len() as u32) + 4) / 4).max(1);
    let packet_dwords = payload_dwords + 1;
    let Some(addr) = alloc_command_dwords(ctx, cb, u64::from(packet_dwords)) else {
        return 0;
    };
    if !ctx
        .mem
        .write(addr, &pm4(packet_dwords, IT_NOP, R_ZERO).to_le_bytes())
    {
        return 0;
    }
    for i in 0..payload_dwords {
        let mut value = 0u32;
        for byte in 0..4u32 {
            let idx = (i * 4 + byte) as usize;
            if idx < marker.len() {
                value |= u32::from(marker[idx]) << (byte * 8);
            }
        }
        if !ctx
            .mem
            .write(addr + 4 + u64::from(i) * 4, &value.to_le_bytes())
        {
            return 0;
        }
    }
    addr
}

/// `sceAgcDcbStallCommandBufferParser(dcb, size, address, reference)`: SharpEmu
/// executes submissions synchronously, so there is no independent command
/// processor to stall — it emits a well-formed 2-DWORD NOP so packet addresses
/// and the cursor stay coherent (`DcbStallCommandBufferParser`, AgcExports.cs
/// L1856-1882). Same story here: our processor consumes the whole buffer per
/// submit. `size > 1` is rejected. Returns the command address, or 0 on
/// failure.
fn hle_dcb_stall_command_buffer_parser(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let size = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    if cb == 0 || size > 1 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 2) else {
        return 0;
    };
    if !ctx.mem.write(addr, &pm4(2, IT_NOP, R_ZERO).to_le_bytes())
        || !ctx.mem.write(addr + 4, &0u32.to_le_bytes())
    {
        return 0;
    }
    addr
}

/// `sceAgcDcbGetLodStats(dcb, cachePolicy, destinationAddress, control,
/// counterMask, resetCounters, enable, counterSelect)`: emit a 5-DWORD
/// GET_LOD_STATS packet — `header, control, dstLo & ~0x3F, dstHi,
/// packetControl` where `packetControl = cachePolicy << 28 | enable << 19 |
/// resetCounters << 18 | counterMask << 10 | counterSelect << 2`. `enable`
/// (arg7) and `counterSelect` (arg8) are stack args → `args[6]`/`args[7]`.
/// Ported from SharpEmu `DcbGetLodStats` (AgcExports.cs L1589-1631). Returns
/// the command address, or 0 on failure.
fn hle_dcb_get_lod_stats(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let cache_policy = (args.get(1).copied().unwrap_or(0) & 0x3) as u32;
    let destination_address = args.get(2).copied().unwrap_or(0);
    let control = args.get(3).copied().unwrap_or(0) as u32;
    let counter_mask = (args.get(4).copied().unwrap_or(0) & 0xFF) as u32;
    let reset_counters = (args.get(5).copied().unwrap_or(0) & 0x1) as u32;
    let enable = (args.get(6).copied().unwrap_or(0) & 0x1) as u32;
    let counter_select = (args.get(7).copied().unwrap_or(0) & 0xFF) as u32;
    if cb == 0 {
        return 0;
    }
    let packet_control = (cache_policy << 28)
        | (enable << 19)
        | (reset_counters << 18)
        | (counter_mask << 10)
        | (counter_select << 2);
    let Some(addr) = alloc_command_dwords(ctx, cb, 5) else {
        return 0;
    };
    let w = |off: u64, v: u32| ctx.mem.write(addr + off, &v.to_le_bytes());
    let ok = w(0, pm4(5, IT_GET_LOD_STATS, R_ZERO))
        && w(4, control)
        && w(8, (destination_address as u32) & !0x3F)
        && w(12, (destination_address >> 32) as u32)
        && w(16, packet_control);
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcAcbEventWrite(acb, eventType, eventAddress)`: emit an EVENT_WRITE
/// packet. Address-carrying event types (`eventType & ~1 == 0x38`) emit a
/// 4-DWORD packet with the address; others a 2-DWORD packet.
fn hle_acb_event_write(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let event_type = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let event_address = args.get(2).copied().unwrap_or(0);
    if cb == 0 || event_type >= 0x40 {
        return 0;
    }
    let has_address = (event_type & !1) == 0x38;
    let packet_dwords = if has_address { 4 } else { 2 };
    let Some(addr) = alloc_command_dwords(ctx, cb, packet_dwords) else {
        return 0;
    };
    let type_word = if has_address {
        event_type | 0x100
    } else {
        event_type & 0x3F
    };
    if !ctx.mem.write(
        addr,
        &pm4(packet_dwords as u32, IT_EVENT_WRITE, R_ZERO).to_le_bytes(),
    ) || !ctx.mem.write(addr + 4, &type_word.to_le_bytes())
    {
        return 0;
    }
    if has_address
        && (!ctx
            .mem
            .write(addr + 8, &((event_address as u32) & !7).to_le_bytes())
            || !ctx
                .mem
                .write(addr + 12, &((event_address >> 32) as u32).to_le_bytes()))
    {
        return 0;
    }
    addr
}

/// `sceAgcDcbEventWrite(dcb, eventType, eventAddress)`: emit an EVENT_WRITE
/// packet. `eventType` ≤ 0x3F and `eventAddress` must be 0.
fn hle_dcb_event_write(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let event_type = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let event_address = args.get(2).copied().unwrap_or(0);
    if cb == 0 || event_type > 0x3F || event_address != 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 2) else {
        return 0;
    };
    if !ctx
        .mem
        .write(addr, &pm4(2, IT_EVENT_WRITE, R_ZERO).to_le_bytes())
        || !ctx.mem.write(addr + 4, &event_type.to_le_bytes())
    {
        return 0;
    }
    addr
}

/// Agc PM4 header: type-3 (`0xC000_0000`), `len_dwords` is the **total** packet
/// size (header + body), the 6-bit `reg` is a sub-discriminator.
fn pm4(len_dwords: u32, op: u32, reg: u32) -> u32 {
    0xC000_0000
        | ((len_dwords.wrapping_sub(2) & 0x3FFF) << 16)
        | ((op & 0xFF) << 8)
        | ((reg & 0x3F) << 2)
}

/// Reserve `size_dwords` from the command buffer at `cb_addr`, advancing its
/// cursor. Returns the address to write the packet at, or `None` if the buffer
/// is full / unreadable. Mirrors SharpEmu's `TryAllocateCommandDwords`.
fn alloc_command_dwords(ctx: &HleContext, cb_addr: u64, size_dwords: u64) -> Option<u64> {
    if size_dwords == 0 {
        return None;
    }
    let mut up = [0u8; 8];
    let mut down = [0u8; 8];
    let mut reserved = [0u8; 4];
    if !ctx.mem.read(cb_addr + CB_CURSOR_UP, &mut up)
        || !ctx.mem.read(cb_addr + CB_CURSOR_DOWN, &mut down)
        || !ctx.mem.read(cb_addr + CB_RESERVED_DW, &mut reserved)
    {
        return None;
    }
    let cursor_up = u64::from_le_bytes(up);
    let cursor_down = u64::from_le_bytes(down);
    let reserved_dw = u64::from(u32::from_le_bytes(reserved));

    let available = if cursor_down >= cursor_up {
        (cursor_down - cursor_up) / 4
    } else {
        0
    };
    // remaining = max(available, reserved) - reserved  (== available - reserved, else 0)
    let remaining = available.max(reserved_dw) - reserved_dw;
    if size_dwords > remaining {
        debug!("agc: command buffer {cb_addr:#x} full (need {size_dwords}, have {remaining})");
        return None;
    }
    let next = cursor_up + size_dwords * 4;
    if !ctx.mem.write(cb_addr + CB_CURSOR_UP, &next.to_le_bytes()) {
        return None;
    }
    Some(cursor_up)
}

/// `sceAgcDcbSetIndexSize(dcb, indexSize, cachePolicy)`: emit an INDEX_TYPE
/// packet. Returns the command address, or 0 on failure.
fn hle_dcb_set_index_size(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let index_size = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let cache_policy = args.get(2).copied().unwrap_or(0) & 0xFF;
    if cb == 0 || cache_policy != 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 2) else {
        return 0;
    };
    if !ctx
        .mem
        .write(addr, &pm4(2, IT_INDEX_TYPE, R_ZERO).to_le_bytes())
        || !ctx.mem.write(addr + 4, &index_size.to_le_bytes())
    {
        return 0;
    }
    addr
}

/// `sceAgcDcbDrawIndexAuto(dcb, indexCount, modifier)`: emit a non-indexed draw
/// (7 DWORDs). Returns the command address, or 0 on failure.
fn hle_dcb_draw_index_auto(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let index_count = args.get(1).copied().unwrap_or(0) as u32;
    let modifier = args.get(2).copied().unwrap_or(0);
    if cb == 0 || modifier != DRAW_AUTO_MODIFIER {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 7) else {
        return 0;
    };
    // header + indexCount + 5 zero DWORDs.
    let ok = ctx
        .mem
        .write(addr, &pm4(7, IT_NOP, R_DRAW_INDEX_AUTO).to_le_bytes())
        && ctx.mem.write(addr + 4, &index_count.to_le_bytes())
        && (1..=5).all(|i| ctx.mem.write(addr + 4 + i * 4, &0u32.to_le_bytes()));
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDcbSetNumInstances(dcb, instanceCount)`: emit a NUM_INSTANCES packet.
fn hle_dcb_set_num_instances(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let instance_count = args.get(1).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 2) else {
        return 0;
    };
    if !ctx
        .mem
        .write(addr, &pm4(2, IT_NUM_INSTANCES, R_ZERO).to_le_bytes())
        || !ctx.mem.write(addr + 4, &instance_count.to_le_bytes())
    {
        return 0;
    }
    addr
}

/// `sceAgcDcbSetIndexBuffer(dcb, indexBufferAddress, indexCount)`: emit an
/// INDEX_BASE packet (address lo/hi) followed by an INDEX_BUFFER_SIZE packet.
fn hle_dcb_set_index_buffer(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let index_buffer = args.get(1).copied().unwrap_or(0);
    let index_count = args.get(2).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 5) else {
        return 0;
    };
    let ok = ctx
        .mem
        .write(addr, &pm4(3, IT_INDEX_BASE, R_ZERO).to_le_bytes())
        && ctx
            .mem
            .write(addr + 4, &(index_buffer as u32).to_le_bytes())
        && ctx
            .mem
            .write(addr + 8, &((index_buffer >> 32) as u32).to_le_bytes())
        && ctx.mem.write(
            addr + 12,
            &pm4(2, IT_INDEX_BUFFER_SIZE, R_ZERO).to_le_bytes(),
        )
        && ctx.mem.write(addr + 16, &index_count.to_le_bytes());
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDcbDrawIndex(dcb, indexCount, indexAddress, modifier)`: emit an
/// INDEX_BASE + INDEX_BUFFER_SIZE packet (5 DWORDs) then a DRAW_INDEX_2 packet
/// (5 DWORDs). Returns the first (base) command address.
fn hle_dcb_draw_index(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let index_count = args.get(1).copied().unwrap_or(0) as u32;
    let index_addr = args.get(2).copied().unwrap_or(0);
    let modifier = args.get(3).copied().unwrap_or(0);
    if cb == 0 || modifier != DRAW_AUTO_MODIFIER {
        return 0;
    }
    let Some(base) = alloc_command_dwords(ctx, cb, 5) else {
        return 0;
    };
    let base_ok = ctx
        .mem
        .write(base, &pm4(3, IT_INDEX_BASE, R_ZERO).to_le_bytes())
        && ctx.mem.write(base + 4, &(index_addr as u32).to_le_bytes())
        && ctx
            .mem
            .write(base + 8, &((index_addr >> 32) as u32).to_le_bytes())
        && ctx.mem.write(
            base + 12,
            &pm4(2, IT_INDEX_BUFFER_SIZE, R_ZERO).to_le_bytes(),
        )
        && ctx.mem.write(base + 16, &index_count.to_le_bytes());
    if !base_ok {
        return 0;
    }
    let Some(draw) = alloc_command_dwords(ctx, cb, 5) else {
        return 0;
    };
    let draw_ok = ctx
        .mem
        .write(draw, &pm4(5, IT_DRAW_INDEX_2, R_ZERO).to_le_bytes())
        && ctx.mem.write(draw + 4, &index_count.to_le_bytes())
        && (2..=4).all(|i| ctx.mem.write(draw + i * 4, &0u32.to_le_bytes()));
    if !draw_ok {
        return 0;
    }
    base
}

/// `sceAgcDcbResetQueue(dcb, op, state)`: emit a draw-reset packet. Requires
/// `op == 0x3FF` and `state == 0`.
fn hle_dcb_reset_queue(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let op = args.get(1).copied().unwrap_or(0);
    let state = args.get(2).copied().unwrap_or(0);
    if cb == 0 || op != 0x3FF || state != 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 2) else {
        return 0;
    };
    if !ctx
        .mem
        .write(addr, &pm4(2, IT_NOP, R_DRAW_RESET).to_le_bytes())
        || !ctx.mem.write(addr + 4, &0u32.to_le_bytes())
    {
        return 0;
    }
    addr
}

/// `sceAgcDcbWaitUntilSafeForRendering(dcb, videoOutHandle, displayBufferIndex)`:
/// emit a wait-flip-done packet (7 DWORDs).
fn hle_dcb_wait_until_safe(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let video_out = args.get(1).copied().unwrap_or(0) as u32;
    let buffer_index = args.get(2).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 7) else {
        return 0;
    };
    let ok = ctx
        .mem
        .write(addr, &pm4(7, IT_NOP, R_WAIT_FLIP_DONE).to_le_bytes())
        && ctx.mem.write(addr + 4, &video_out.to_le_bytes())
        && ctx.mem.write(addr + 8, &buffer_index.to_le_bytes())
        && (3..=6).all(|i| ctx.mem.write(addr + i * 4, &0u32.to_le_bytes()));
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcDcbPopMarker(dcb)`: emit a pop-marker packet (2 DWORDs).
fn hle_dcb_pop_marker(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 2) else {
        return 0;
    };
    if !ctx
        .mem
        .write(addr, &pm4(2, IT_NOP, R_POP_MARKER).to_le_bytes())
        || !ctx.mem.write(addr + 4, &0u32.to_le_bytes())
    {
        return 0;
    }
    addr
}

/// `sceAgcCbDispatch(cb, groupCountX, groupCountY, groupCountZ, modifier)`:
/// emit a direct compute dispatch (5 DWORDs). The last DWORD folds the
/// dispatch initiator bits (`(modifier & 0xA038) | 0x41`).
fn hle_cb_dispatch(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let x = args.get(1).copied().unwrap_or(0) as u32;
    let y = args.get(2).copied().unwrap_or(0) as u32;
    let z = args.get(3).copied().unwrap_or(0) as u32;
    let modifier = args.get(4).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 5) else {
        return 0;
    };
    let initiator = (modifier & 0xA038) | 0x41;
    let ok = ctx
        .mem
        .write(addr, &pm4(5, IT_DISPATCH_DIRECT, R_ZERO).to_le_bytes())
        && ctx.mem.write(addr + 4, &x.to_le_bytes())
        && ctx.mem.write(addr + 8, &y.to_le_bytes())
        && ctx.mem.write(addr + 12, &z.to_le_bytes())
        && ctx.mem.write(addr + 16, &initiator.to_le_bytes());
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcCbNop(cb, dwordCount)`: emit a NOP packet of `dwordCount` DWORDs
/// (header + zero-filled body). `dwordCount` must be in `[2, 0x4001]`.
fn hle_cb_nop(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let dword_count = args.get(1).copied().unwrap_or(0);
    if cb == 0 || !(2..=0x4001).contains(&dword_count) {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, dword_count) else {
        return 0;
    };
    if !ctx
        .mem
        .write(addr, &pm4(dword_count as u32, IT_NOP, R_ZERO).to_le_bytes())
    {
        return 0;
    }
    for i in 1..dword_count {
        if !ctx.mem.write(addr + i * 4, &0u32.to_le_bytes()) {
            return 0;
        }
    }
    addr
}

/// `sceAgcCbReleaseMem`: emit the Gen5 RELEASE_MEM synchronization packet.
///
/// The first six arguments arrive in registers and the remaining six on the
/// guest stack; `HleContext` presents both sets as one ABI-ordered slice.
fn hle_cb_release_mem(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let action = (args.get(1).copied().unwrap_or(0) & 0xff) as u32;
    let gcr_control = (args.get(2).copied().unwrap_or(0) & 0xffff) as u32;
    let destination = (args.get(3).copied().unwrap_or(0) & 0xff) as u32;
    let cache_policy = (args.get(4).copied().unwrap_or(0) & 0xff) as u32;
    let destination_address = args.get(5).copied().unwrap_or(0);
    let data_selection = (args.get(6).copied().unwrap_or(0) & 0xff) as u32;
    let data = args.get(7).copied().unwrap_or(0);
    let gds_offset = (args.get(8).copied().unwrap_or(0) & 0xffff) as u32;
    let gds_size = (args.get(9).copied().unwrap_or(0) & 0xffff) as u32;
    let interrupt = (args.get(10).copied().unwrap_or(0) & 0xff) as u32;
    let interrupt_context_id = args.get(11).copied().unwrap_or(0) as u32;

    if cb == 0
        || destination > 1
        || data_selection > 3
        || gds_offset != 0
        || gds_size > 2
        || interrupt > 3
    {
        return 0;
    }

    let Some(addr) = alloc_command_dwords(ctx, cb, 8) else {
        return 0;
    };
    let dwords = [
        pm4(8, IT_NOP, R_RELEASE_MEM),
        action | (cache_policy << 8),
        gcr_control | (data_selection << 16) | (interrupt << 24),
        destination_address as u32,
        (destination_address >> 32) as u32,
        data as u32,
        (data >> 32) as u32,
        interrupt_context_id,
    ];
    if !dwords
        .iter()
        .enumerate()
        .all(|(index, value)| ctx.mem.write(addr + index as u64 * 4, &value.to_le_bytes()))
    {
        return 0;
    }
    addr
}

/// `sceAgcDcbSetFlip(cb, videoOutHandle, displayBufferIndex, flipMode, flipArg)`:
/// emit a flip packet (6 DWORDs).
fn hle_dcb_set_flip(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let video_out = args.get(1).copied().unwrap_or(0) as u32;
    let buffer_index = args.get(2).copied().unwrap_or(0) as u32;
    let flip_mode = args.get(3).copied().unwrap_or(0) as u32;
    let flip_arg = args.get(4).copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 6) else {
        return 0;
    };
    let ok = ctx.mem.write(addr, &pm4(6, IT_NOP, R_FLIP).to_le_bytes())
        && ctx.mem.write(addr + 4, &video_out.to_le_bytes())
        && ctx.mem.write(addr + 8, &buffer_index.to_le_bytes())
        && ctx.mem.write(addr + 12, &flip_mode.to_le_bytes())
        && ctx.mem.write(addr + 16, &(flip_arg as u32).to_le_bytes())
        && ctx
            .mem
            .write(addr + 20, &((flip_arg >> 32) as u32).to_le_bytes());
    if !ok {
        return 0;
    }
    addr
}

/// `sceAgcAcbResetQueue(acb)`: emit an async-compute-buffer reset packet.
fn hle_acb_reset_queue(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 2) else {
        return 0;
    };
    if !ctx
        .mem
        .write(addr, &pm4(2, IT_NOP, R_ACB_RESET).to_le_bytes())
        || !ctx.mem.write(addr + 4, &0u32.to_le_bytes())
    {
        return 0;
    }
    addr
}

/// `sceAgcCbSetShRegisterRangeDirect(cb, offset, values, valueCount)`: emit a
/// marker packet then a SET_SH_REG packet writing `valueCount` values (read
/// from the guest `values` array) to SH registers starting at `offset`.
fn hle_cb_set_sh_register_range(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let offset = args.get(1).copied().unwrap_or(0) as u32;
    let values_addr = args.get(2).copied().unwrap_or(0);
    let value_count = args.get(3).copied().unwrap_or(0);
    if cb == 0 || offset == 0 || offset > 0x3FF || value_count == 0 {
        return 0;
    }
    // Marker packet.
    let Some(marker) = alloc_command_dwords(ctx, cb, 2) else {
        return 0;
    };
    if !ctx.mem.write(marker, &pm4(2, IT_NOP, R_ZERO).to_le_bytes())
        || !ctx
            .mem
            .write(marker + 4, &SET_SH_RANGE_MARKER.to_le_bytes())
    {
        return 0;
    }
    // SET_SH_REG packet: header + offset + valueCount values.
    let Some(addr) = alloc_command_dwords(ctx, cb, value_count + 2) else {
        return 0;
    };
    if !ctx.mem.write(
        addr,
        &pm4((value_count + 2) as u32, IT_SET_SH_REG, R_ZERO).to_le_bytes(),
    ) || !ctx.mem.write(addr + 4, &offset.to_le_bytes())
    {
        return 0;
    }
    for i in 0..value_count {
        // Copy each source value (0 if the source is null/unreadable).
        let mut v = [0u8; 4];
        if values_addr != 0 {
            let _ = ctx.mem.read(values_addr + i * 4, &mut v);
        }
        if !ctx.mem.write(addr + 8 + i * 4, &v) {
            return 0;
        }
    }
    addr
}

// ---------------------------------------------------------------------------
// Register defaults (sceAgcGetRegisterDefaults2 / ...2Internal)
//
// Ported from Kyty Graphics.cpp (MIT © InoriRus): Gen5's
// `GraphicsGetRegisterDefaults2[Internal]` return a pointer to a
// `RegisterDefaults` struct whose tables the GUEST walks itself, so the whole
// object graph must live in guest memory with guest addresses inside it.
// ---------------------------------------------------------------------------

use crate::libsce_agc_reg_defaults::{
    CX_REG_INFO1, CX_REG_INFO2, RegisterDefaultInfo, SH_REG_INFO1, SH_REG_INFO2, UC_REG_INFO1,
    UC_REG_INFO2,
};

/// `RegisterDefaults` header size. Layout (Kyty, `offsetof(count) == 0x38`
/// asserted against the real SDK):
/// `{ tbl0..tbl3: *[*ShaderRegister] (0x00..0x20), unknown[2]: u64 (0x20),
///    types: *u32 (0x30), count: u32 (0x38), pad (0x3C) }`.
const REG_DEFAULTS_HEADER_BYTES: u64 = 0x40;
/// `offsetof(RegisterDefaults, count)` — pinned by test against 0x38.
const REG_DEFAULTS_COUNT_OFFSET: u64 = 0x38;
/// Kyty's `RegisterDefaultInfo` carries a fixed `ShaderRegister reg[16]`.
const REG_INFO_SLOTS: u64 = 16;
/// One materialized `RegisterDefaultInfo`: `u32 type` + 16 × `(u32, u32)`.
const REG_INFO_ENTRY_BYTES: u64 = 4 + REG_INFO_SLOTS * 8;

fn put_u32(image: &mut [u8], off: u64, v: u32) {
    let off = off as usize;
    image[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(image: &mut [u8], off: u64, v: u64) {
    let off = off as usize;
    image[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// Build both `RegisterDefaults` sets in one guest allocation and return its
/// base address (set 1 at `base`, set 2 at `base + 0x40`).
///
/// Guest layout, all pointers absolute guest addresses:
///
/// ```text
/// base + 0x00   RegisterDefaults #1        (returned by GetRegisterDefaults2)
/// base + 0x40   RegisterDefaults #2        (returned by ...2Internal)
/// base + 0x80   RegisterDefaultInfo entries (cx1, sh1, uc1, cx2, sh2, uc2;
///               132 bytes each: type hash + reg[16], unused slots zeroed)
/// (8-aligned)   pointer tables — one guest pointer per entry, aimed at the
///               entry's reg[0] (i.e. entry base + 4, exactly like Kyty's
///               &g_*_reg_info[i].reg[0])
/// (then)        index triples (type_hash, id*4 + table, 0) — Kyty's
///               g_tbl_index1/2, count = number of triples
/// ```
fn materialize_register_defaults(ctx: &HleContext) -> Option<u64> {
    let sets: [[&[RegisterDefaultInfo]; 3]; 2] = [
        [CX_REG_INFO1, SH_REG_INFO1, UC_REG_INFO1],
        [CX_REG_INFO2, SH_REG_INFO2, UC_REG_INFO2],
    ];

    // First pass: compute every region's offset.
    let mut info_off = [[0u64; 3]; 2];
    let mut cursor = 2 * REG_DEFAULTS_HEADER_BYTES;
    for (s, groups) in sets.iter().enumerate() {
        for (g, table) in groups.iter().enumerate() {
            info_off[s][g] = cursor;
            cursor += table.len() as u64 * REG_INFO_ENTRY_BYTES;
        }
    }
    cursor = (cursor + 7) & !7; // pointer tables are 8-byte aligned
    let mut ptr_off = [[0u64; 3]; 2];
    for (s, groups) in sets.iter().enumerate() {
        for (g, table) in groups.iter().enumerate() {
            ptr_off[s][g] = cursor;
            cursor += table.len() as u64 * 8;
        }
    }
    let mut idx_off = [0u64; 2];
    for (s, groups) in sets.iter().enumerate() {
        idx_off[s] = cursor;
        cursor += groups.iter().map(|t| t.len() as u64).sum::<u64>() * 12;
    }
    let total = cursor;

    let base = ctx.alloc.alloc(total, 8)?;
    if base == 0 {
        return None; // 0 doubles as the "not materialized" sentinel
    }

    // Second pass: render the image host-side, then one guest write.
    let mut image = vec![0u8; total as usize];
    for (s, groups) in sets.iter().enumerate() {
        let header = s as u64 * REG_DEFAULTS_HEADER_BYTES;
        let mut triples = 0u64;
        for (g, table) in groups.iter().enumerate() {
            put_u64(&mut image, header + g as u64 * 8, base + ptr_off[s][g]);
            for (i, (type_hash, regs)) in table.iter().enumerate() {
                let entry = info_off[s][g] + i as u64 * REG_INFO_ENTRY_BYTES;
                put_u32(&mut image, entry, *type_hash);
                for (slot, (offset, value)) in regs.iter().enumerate() {
                    put_u32(&mut image, entry + 4 + slot as u64 * 8, *offset);
                    put_u32(&mut image, entry + 8 + slot as u64 * 8, *value);
                }
                put_u64(&mut image, ptr_off[s][g] + i as u64 * 8, base + entry + 4);
                let t = idx_off[s] + triples * 12;
                put_u32(&mut image, t, *type_hash);
                put_u32(&mut image, t + 4, (i * 4 + g) as u32);
                // third u32 of the triple stays 0, as in Kyty
                triples += 1;
            }
        }
        // tbl3 (0x18) and unknown[2] (0x20) stay zero.
        put_u64(&mut image, header + 0x30, base + idx_off[s]);
        put_u32(
            &mut image,
            header + REG_DEFAULTS_COUNT_OFFSET,
            triples as u32,
        );
    }
    if !ctx.mem.write(base, &image) {
        ctx.alloc.free(base);
        return None;
    }
    debug!("AGC register defaults materialized at {base:#x} ({total} bytes)");
    Some(base)
}

/// Materialize-once, then serve the cached guest address plus `offset`.
fn register_defaults_base(ctx: &HleContext, args: &[u64], offset: u64) -> u64 {
    use std::sync::atomic::Ordering;

    let version = args.first().copied().unwrap_or(0) as u32;
    if version != 8 {
        warn!(
            "sceAgcGetRegisterDefaults2: unexpected version {version} (reference \
             asserts 8) — returning the version-8 tables anyway"
        );
    }
    let mut base = ctx
        .kernel
        .agc_register_defaults_addr
        .load(Ordering::Acquire);
    if base == 0 {
        let Some(built) = materialize_register_defaults(ctx) else {
            warn!("sceAgcGetRegisterDefaults2: guest materialization failed — returning null");
            return 0;
        };
        base = match ctx.kernel.agc_register_defaults_addr.compare_exchange(
            0,
            built,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => built,
            Err(winner) => {
                // Another thread materialized first; discard ours.
                ctx.alloc.free(built);
                winner
            }
        };
    }
    base + offset
}

/// `sceAgcGetRegisterDefaults2(version)`: guest pointer to the primary
/// register-defaults set (Kyty's `g_reg_defaults1`).
fn hle_get_register_defaults2(ctx: &HleContext, args: &[u64]) -> u64 {
    register_defaults_base(ctx, args, 0)
}

/// `sceAgcGetRegisterDefaults2Internal(version)`: guest pointer to the
/// internal register-defaults set (Kyty's `g_reg_defaults2`).
fn hle_get_register_defaults2_internal(ctx: &HleContext, args: &[u64]) -> u64 {
    register_defaults_base(ctx, args, REG_DEFAULTS_HEADER_BYTES)
}

// ---------------------------------------------------------------------------
// Minecraft RenderDragon DCB batch
// ---------------------------------------------------------------------------

/// `true` exactly once per `flag` — gate for warn-once diagnostics.
fn warn_once(flag: &std::sync::atomic::AtomicBool) -> bool {
    !flag.swap(true, std::sync::atomic::Ordering::Relaxed)
}

/// `sceAgcDcbSetIndexCount(dcb, indexCount)`: emit an INDEX_BUFFER_SIZE
/// packet (2 DWORDs) — the standalone form of the size packet that
/// `sceAgcDcbSetIndexBuffer` emits after INDEX_BASE.
fn hle_dcb_set_index_count(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let index_count = args.get(1).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 2) else {
        return 0;
    };
    if !ctx
        .mem
        .write(addr, &pm4(2, IT_INDEX_BUFFER_SIZE, R_ZERO).to_le_bytes())
        || !ctx.mem.write(addr + 4, &index_count.to_le_bytes())
    {
        return 0;
    }
    addr
}

/// Shared body for `sceAgcDcbDrawIndexIndirectMulti` / `sceAgcDcbDrawIndirectMulti`.
///
/// Neither Kyty Gen5 nor SharpEmu implements these, so the argument order
/// beyond the buffer is a documented guess (loud warn on first use):
/// `(dcb, dataOffset, drawCount, stride, countAddress, modifier)` emitted in
/// the standard PM4 `DRAW_*_INDIRECT_MULTI` field order (9 DWORDs):
/// `[header, dataOffset, baseVtxLoc=0, startInstLoc=0, drawCount,
///   countAddrLo, countAddrHi, stride, modifier]`.
fn dcb_draw_indirect_multi(
    ctx: &HleContext,
    args: &[u64],
    op: u32,
    warned: &std::sync::atomic::AtomicBool,
    label: &str,
) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let data_offset = args.get(1).copied().unwrap_or(0) as u32;
    let draw_count = args.get(2).copied().unwrap_or(0) as u32;
    let stride = args.get(3).copied().unwrap_or(0) as u32;
    let count_address = args.get(4).copied().unwrap_or(0);
    let modifier = args.get(5).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    if warn_once(warned) {
        warn!(
            "{label}: no reference implementation — emitting standard PM4 layout \
             from a GUESSED argument decode (dataOffset={data_offset:#x}, \
             drawCount={draw_count}, stride={stride}, countAddress={count_address:#x}, \
             modifier={modifier:#x}); verify against a real dump before trusting draws"
        );
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 9) else {
        return 0;
    };
    let dwords = [
        pm4(9, op, R_ZERO),
        data_offset,
        0, // base vertex location
        0, // start instance location
        draw_count,
        count_address as u32,
        (count_address >> 32) as u32,
        stride,
        modifier,
    ];
    if !dwords
        .iter()
        .enumerate()
        .all(|(index, value)| ctx.mem.write(addr + index as u64 * 4, &value.to_le_bytes()))
    {
        return 0;
    }
    addr
}

fn hle_dcb_draw_index_indirect_multi(ctx: &HleContext, args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    dcb_draw_indirect_multi(
        ctx,
        args,
        IT_DRAW_INDEX_INDIRECT_MULTI,
        &WARNED,
        "sceAgcDcbDrawIndexIndirectMulti",
    )
}

fn hle_dcb_draw_indirect_multi(ctx: &HleContext, args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    dcb_draw_indirect_multi(
        ctx,
        args,
        IT_DRAW_INDIRECT_MULTI,
        &WARNED,
        "sceAgcDcbDrawIndirectMulti",
    )
}

/// `sceAgcDcbCopyData(dcb, control, srcAddress, dstAddress)`: emit a standard
/// PM4 COPY_DATA packet (6 DWORDs). No reference implementation exists
/// (Kyty Gen5 lacks it), so the argument decode is a documented guess and the
/// first use warns loudly.
fn hle_dcb_copy_data(ctx: &HleContext, args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let cb = args.first().copied().unwrap_or(0);
    let control = args.get(1).copied().unwrap_or(0) as u32;
    let source = args.get(2).copied().unwrap_or(0);
    let destination = args.get(3).copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    if warn_once(&WARNED) {
        warn!(
            "sceAgcDcbCopyData: no reference implementation — emitting standard PM4 \
             COPY_DATA from a GUESSED argument decode (control={control:#x}, \
             src={source:#x}, dst={destination:#x})"
        );
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 6) else {
        return 0;
    };
    let dwords = [
        pm4(6, IT_COPY_DATA, R_ZERO),
        control,
        source as u32,
        (source >> 32) as u32,
        destination as u32,
        (destination >> 32) as u32,
    ];
    if !dwords
        .iter()
        .enumerate()
        .all(|(index, value)| ctx.mem.write(addr + index as u64 * 4, &value.to_le_bytes()))
    {
        return 0;
    }
    addr
}

/// `sceAgcDcbSetPredication(dcb, conditionAddress, control)`: emit a standard
/// PM4 SET_PREDICATION packet (4 DWORDs: address lo/hi + control). Argument
/// decode is a documented guess (no reference implementation); warns once.
fn hle_dcb_set_predication(ctx: &HleContext, args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let cb = args.first().copied().unwrap_or(0);
    let condition_address = args.get(1).copied().unwrap_or(0);
    let control = args.get(2).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return 0;
    }
    if warn_once(&WARNED) {
        warn!(
            "sceAgcDcbSetPredication: no reference implementation — emitting standard \
             PM4 SET_PREDICATION from a GUESSED argument decode \
             (conditionAddress={condition_address:#x}, control={control:#x})"
        );
    }
    let Some(addr) = alloc_command_dwords(ctx, cb, 4) else {
        return 0;
    };
    let dwords = [
        pm4(4, IT_SET_PREDICATION, R_ZERO),
        condition_address as u32,
        (condition_address >> 32) as u32,
        control,
    ];
    if !dwords
        .iter()
        .enumerate()
        .all(|(index, value)| ctx.mem.write(addr + index as u64 * 4, &value.to_le_bytes()))
    {
        return 0;
    }
    addr
}

/// `sceAgcSetRangePredication(...)`: completely unreferenced (no Kyty/SharpEmu
/// equivalent, and — unlike the `Dcb` family — not obviously an emitter), so
/// this intentionally touches nothing: warn loudly once and report success.
fn hle_set_range_predication(_ctx: &HleContext, args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if warn_once(&WARNED) {
        warn!(
            "sceAgcSetRangePredication: unimplemented semantics (no reference) — \
             ignoring call (args={args:x?}) and returning 0"
        );
    }
    0
}

/// `sceAgcDebugRaiseException(...)`: the guest reports a fatal GPU-debug
/// condition. Log it loudly with its arguments and return — never abort the
/// host on the guest's behalf.
fn hle_debug_raise_exception(_ctx: &HleContext, args: &[u64]) -> u64 {
    error!("sceAgcDebugRaiseException({args:x?}) — guest raised a GPU debug exception; continuing");
    0
}

/// `sceAgcCbSetShRegisterRangeDirectGetSize(numValues)`: DWORDs the direct
/// SH-range write occupies — the marker packet (2), the SET_SH_REG header and
/// offset (2), and one DWORD per value (see [`hle_cb_set_sh_register_range`],
/// whose emission this must always match). The single-`numValues` signature
/// is inferred from the writer (size depends on nothing else); warns once.
fn hle_cb_set_sh_register_range_get_size(_ctx: &HleContext, args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let num_values = args.first().copied().unwrap_or(0) as u32;
    if warn_once(&WARNED) {
        warn!(
            "sceAgcCbSetShRegisterRangeDirectGetSize: signature inferred (numValues \
             in the first argument register) — returning {} DWORDs for numValues={num_values}",
            u64::from(num_values) + 4
        );
    }
    u64::from(num_values) + 4
}

/// NID `-KRzWekV120` (`0xfca47359e915d76d`, measured from Minecraft
/// PPSA17221): identity unknown — not present in Kyty, SharpEmu, or public
/// NID tables. Warns loudly once and returns 0.
fn hle_unknown_krz_wek_v120(_ctx: &HleContext, args: &[u64]) -> u64 {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if warn_once(&WARNED) {
        warn!(
            "libSceAgc NID 0xfca47359e915d76d (-KRzWekV120): UNKNOWN function called \
             (args={args:x?}) — stub returns 0; identify and implement before trusting \
             rendering"
        );
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    fn ctx_env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x2000);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    /// Set up a command-buffer object at `cb` with a data region [start, end).
    fn setup_cb(ctx: &HleContext, cb: u64, start: u64, end: u64) {
        assert!(ctx.mem.write(cb + CB_CURSOR_UP, &start.to_le_bytes()));
        assert!(ctx.mem.write(cb + CB_CURSOR_DOWN, &end.to_le_bytes()));
        assert!(ctx.mem.write(cb + CB_RESERVED_DW, &0u32.to_le_bytes()));
    }

    fn read_u32(ctx: &HleContext, addr: u64) -> u32 {
        let mut b = [0u8; 4];
        assert!(ctx.mem.read(addr, &mut b));
        u32::from_le_bytes(b)
    }

    fn read_u64(ctx: &HleContext, addr: u64) -> u64 {
        let mut b = [0u8; 8];
        assert!(ctx.mem.read(addr, &mut b));
        u64::from_le_bytes(b)
    }

    #[test]
    fn submit_signals_timestamp_fences_and_eop_interrupts() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // DCB at 0x900: one data_selection=3 (GPU timestamp) RELEASE_MEM
        // targeting the fence at 0x980, then one interrupt-only RELEASE_MEM
        // (data_selection 0, no address, interrupt=2).
        let words: [u32; 16] = [
            pm4(8, IT_NOP, R_RELEASE_MEM),
            0,
            3 << 16,
            0x980,
            0,
            0,
            0,
            0,
            pm4(8, IT_NOP, R_RELEASE_MEM),
            0,
            2 << 24,
            0,
            0,
            0,
            0,
            0x55,
        ];
        for (index, word) in words.iter().enumerate() {
            assert!(ctx.mem.write(0x900 + index as u64 * 4, &word.to_le_bytes()));
        }
        let eq = kernel.create_equeue(0);
        assert_eq!(hle_driver_add_eq_event(&ctx, &[eq, 0x84, 0xCAFE]), 0);
        // Bystanders on the same queue: a plain user event (default filter)
        // and a VideoOut event (-13). The EOP broadcast must key on the
        // graphics-core filter and leave both untouched.
        kernel
            .kernel_equeue_events
            .insert((eq, 0x99), xps5x_kernel::EqueueUserEvent::default());
        kernel.kernel_equeue_events.insert(
            (eq, 0x9a),
            xps5x_kernel::EqueueUserEvent {
                filter: -13,
                ..Default::default()
            },
        );
        assert!(ctx.mem.write(0x180, &0x900u64.to_le_bytes()));
        assert!(ctx.mem.write(0x188, &16u32.to_le_bytes()));
        assert_eq!(hle_driver_submit_dcb(&ctx, &[0x180]), 0);
        // The timestamp fence is written non-zero (the packet's immediate is
        // zero — a title polls this label for a non-zero/advancing clock).
        let first = read_u64(&ctx, 0x980);
        assert_ne!(first, 0, "timestamp fence must not stay zero");
        // The EOP interrupt triggered the registered AGC event — and ONLY it:
        // the user event and the VideoOut event on the same queue stay quiet.
        {
            let event = kernel.kernel_equeue_events.get(&(eq, 0x84)).unwrap();
            assert!(event.triggered, "EOP interrupt must trigger the AGC event");
            assert_eq!(event.filter, EVFILT_GRAPHICS_CORE);
            assert_eq!(event.data, 0x55);
            let user = kernel.kernel_equeue_events.get(&(eq, 0x99)).unwrap();
            assert!(!user.triggered, "user event must not see the EOP broadcast");
            let vout = kernel.kernel_equeue_events.get(&(eq, 0x9a)).unwrap();
            assert!(
                !vout.triggered,
                "VideoOut event must not see the EOP broadcast"
            );
        }
        // A second submission advances the fence (monotonic clock counter).
        assert_eq!(hle_driver_submit_dcb(&ctx, &[0x180]), 0);
        let second = read_u64(&ctx, 0x980);
        assert!(
            second > first,
            "timestamp fence must advance: {first} -> {second}"
        );
    }

    #[test]
    fn set_index_size_emits_packet_and_advances_cursor() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800); // 0x400 data bytes
        let ret = hle_dcb_set_index_size(&ctx, &[cb, 2, 0]);
        assert_eq!(ret, 0x400, "returns the packet address (old cursor)");
        // Exact Agc encoding: header then indexSize.
        assert_eq!(read_u32(&ctx, 0x400), pm4(2, IT_INDEX_TYPE, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 2);
        // Cursor advanced by 2 DWORDs.
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x408);
        // A non-zero cache policy is rejected.
        assert_eq!(hle_dcb_set_index_size(&ctx, &[cb, 2, 1]), 0);
    }

    #[test]
    fn draw_index_auto_emits_seven_dwords() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // A wrong modifier is rejected.
        assert_eq!(hle_dcb_draw_index_auto(&ctx, &[cb, 3, 0]), 0);
        let ret = hle_dcb_draw_index_auto(&ctx, &[cb, 3, DRAW_AUTO_MODIFIER]);
        assert_eq!(ret, 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(7, IT_NOP, R_DRAW_INDEX_AUTO));
        assert_eq!(read_u32(&ctx, 0x404), 3, "index count");
        // Cursor advanced by 7 DWORDs (28 bytes).
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x400 + 28);
    }

    #[test]
    fn set_num_instances_emits_a_two_dword_packet() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        assert_eq!(hle_dcb_set_num_instances(&ctx, &[cb, 8]), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(2, IT_NUM_INSTANCES, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 8);
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x408);
    }

    #[test]
    fn set_index_buffer_emits_base_and_size_packets() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let ib = 0x1_2345_6780u64;
        assert_eq!(hle_dcb_set_index_buffer(&ctx, &[cb, ib, 6]), 0x400);
        // INDEX_BASE packet: header + addr lo/hi.
        assert_eq!(read_u32(&ctx, 0x400), pm4(3, IT_INDEX_BASE, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), ib as u32);
        assert_eq!(read_u32(&ctx, 0x408), (ib >> 32) as u32);
        // INDEX_BUFFER_SIZE packet: header + count.
        assert_eq!(read_u32(&ctx, 0x40C), pm4(2, IT_INDEX_BUFFER_SIZE, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x410), 6);
        // 5 DWORDs total.
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x400 + 20);
    }

    #[test]
    fn draw_index_emits_base_then_draw_packets() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let ib = 0x00AB_CDEF_1234_5678u64;
        assert_eq!(
            hle_dcb_draw_index(&ctx, &[cb, 6, ib, 0]),
            0,
            "bad modifier → 0"
        );
        let ret = hle_dcb_draw_index(&ctx, &[cb, 6, ib, DRAW_AUTO_MODIFIER]);
        assert_eq!(ret, 0x400);
        // Base packet (5 dw) then draw packet (5 dw) = 10 dw total.
        assert_eq!(read_u32(&ctx, 0x400), pm4(3, IT_INDEX_BASE, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x40C), pm4(2, IT_INDEX_BUFFER_SIZE, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x414), pm4(5, IT_DRAW_INDEX_2, R_ZERO)); // draw at 0x400+20
        assert_eq!(read_u32(&ctx, 0x418), 6, "index count in draw packet");
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x400 + 40);
    }

    #[test]
    fn reset_queue_validates_and_wait_emits_flip_packet() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // ResetQueue requires op=0x3FF, state=0.
        assert_eq!(hle_dcb_reset_queue(&ctx, &[cb, 0, 0]), 0);
        assert_eq!(hle_dcb_reset_queue(&ctx, &[cb, 0x3FF, 0]), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(2, IT_NOP, R_DRAW_RESET));
        // WaitUntilSafeForRendering emits a 7-dword wait-flip packet.
        let ret = hle_dcb_wait_until_safe(&ctx, &[cb, 9, 1]);
        assert_eq!(ret, 0x408);
        assert_eq!(read_u32(&ctx, 0x408), pm4(7, IT_NOP, R_WAIT_FLIP_DONE));
        assert_eq!(read_u32(&ctx, 0x40C), 9, "video-out handle");
        assert_eq!(read_u32(&ctx, 0x410), 1, "display buffer index");
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x408 + 28);
    }

    #[test]
    fn pop_marker_and_dispatch_emit_correctly() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // Pop marker: 2-dword packet.
        assert_eq!(hle_dcb_pop_marker(&ctx, &[cb]), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(2, IT_NOP, R_POP_MARKER));
        // Dispatch: 5-dword packet with group counts + initiator.
        let ret = hle_cb_dispatch(&ctx, &[cb, 4, 2, 1, 0]);
        assert_eq!(ret, 0x408);
        assert_eq!(read_u32(&ctx, 0x408), pm4(5, IT_DISPATCH_DIRECT, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x40C), 4);
        assert_eq!(read_u32(&ctx, 0x410), 2);
        assert_eq!(read_u32(&ctx, 0x414), 1);
        assert_eq!(
            read_u32(&ctx, 0x418),
            0x41,
            "initiator = (0 & 0xA038) | 0x41"
        );
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x408 + 20);
    }

    #[test]
    fn cb_nop_setflip_acbreset_emit_correctly() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // CbNop of 3 dwords: header + 2 zeros; too-small count rejected.
        assert_eq!(hle_cb_nop(&ctx, &[cb, 1]), 0);
        assert_eq!(hle_cb_nop(&ctx, &[cb, 3]), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(3, IT_NOP, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 0);
        // SetFlip: 6-dword packet with a split 64-bit flip arg.
        let ret = hle_dcb_set_flip(&ctx, &[cb, 7, 2, 1, 0x0000_00AB_1234_5678]);
        assert_eq!(ret, 0x40C);
        assert_eq!(read_u32(&ctx, 0x40C), pm4(6, IT_NOP, R_FLIP));
        assert_eq!(read_u32(&ctx, 0x41C), 0x1234_5678, "flip arg low");
        assert_eq!(read_u32(&ctx, 0x420), 0x0000_00AB, "flip arg high");
        // AcbResetQueue: 2-dword async-compute reset.
        let a = hle_acb_reset_queue(&ctx, &[cb]);
        assert_eq!(read_u32(&ctx, a), pm4(2, IT_NOP, R_ACB_RESET));
    }

    #[test]
    fn cb_release_mem_emits_gen5_packet_and_validates_fields() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);

        let address = 0x1234_5678_9abc_def0;
        let data = 0x0fed_cba9_8765_4321;
        let args = [
            cb,
            0x5a,
            0x1234,
            1,
            0xa5,
            address,
            3,
            data,
            0,
            2,
            3,
            0xdead_beef,
        ];
        assert_eq!(hle_cb_release_mem(&ctx, &args), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(8, IT_NOP, R_RELEASE_MEM));
        assert_eq!(read_u32(&ctx, 0x404), 0x0000_a55a);
        assert_eq!(read_u32(&ctx, 0x408), 0x0303_1234);
        assert_eq!(read_u32(&ctx, 0x40c), address as u32);
        assert_eq!(read_u32(&ctx, 0x410), (address >> 32) as u32);
        assert_eq!(read_u32(&ctx, 0x414), data as u32);
        assert_eq!(read_u32(&ctx, 0x418), (data >> 32) as u32);
        assert_eq!(read_u32(&ctx, 0x41c), 0xdead_beef);
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x420);

        for invalid in [
            [cb, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [cb, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
            [cb, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [cb, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
            [cb, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
        ] {
            assert_eq!(hle_cb_release_mem(&ctx, &invalid), 0);
            assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x420);
        }

        let registry = HleRegistry::new();
        register(&registry);
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0xc2bd_b774_f2b2_59cd && key == "libSceAgc::sceAgcCbReleaseMem"
                })
        );
    }

    #[test]
    fn set_sh_register_range_emits_marker_and_copies_values() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // Source values at 0x100.
        assert!(ctx.mem.write(0x100, &0xAAAA_AAAAu32.to_le_bytes()));
        assert!(ctx.mem.write(0x104, &0xBBBB_BBBBu32.to_le_bytes()));
        let ret = hle_cb_set_sh_register_range(&ctx, &[cb, 0x2C, 0x100, 2]);
        assert_eq!(
            ret, 0x408,
            "returns the SET_SH_REG packet (after the marker)"
        );
        // Marker packet at 0x400.
        assert_eq!(read_u32(&ctx, 0x400), pm4(2, IT_NOP, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), SET_SH_RANGE_MARKER);
        // SET_SH_REG packet: header, offset, then the two copied values.
        assert_eq!(read_u32(&ctx, 0x408), pm4(4, IT_SET_SH_REG, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x40C), 0x2C, "register offset");
        assert_eq!(read_u32(&ctx, 0x410), 0xAAAA_AAAA);
        assert_eq!(read_u32(&ctx, 0x414), 0xBBBB_BBBB);
        // Invalid offset (> 0x3FF) rejected.
        assert_eq!(
            hle_cb_set_sh_register_range(&ctx, &[cb, 0x400, 0x100, 1]),
            0
        );
    }

    #[test]
    fn init_validates_version_and_event_write_emits() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Init: supported versions {7,8,10,13} with a non-null state → OK.
        assert_eq!(hle_init(&ctx, &[0x40, 8]), 0);
        assert_eq!(hle_init(&ctx, &[0x40, 12]), 0);
        assert_eq!(hle_init(&ctx, &[0x40, 13]), 0);
        assert_eq!(
            hle_init(&ctx, &[0x40, 9]),
            SCE_ERROR_INVALID_ARGUMENT,
            "bad version"
        );
        assert_eq!(
            hle_init(&ctx, &[0, 8]),
            SCE_ERROR_INVALID_ARGUMENT,
            "null state"
        );

        let registry = HleRegistry::new();
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0xdb72_d151_2bd8_bb53 && key == "libSceAgc::sceAgcInit"
                })
        );

        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // EventWrite: type ≤ 0x3F, address must be 0.
        assert_eq!(
            hle_dcb_event_write(&ctx, &[cb, 0x40, 0]),
            0,
            "type > 0x3F rejected"
        );
        assert_eq!(
            hle_dcb_event_write(&ctx, &[cb, 0x14, 1]),
            0,
            "non-zero address rejected"
        );
        assert_eq!(hle_dcb_event_write(&ctx, &[cb, 0x14, 0]), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(2, IT_EVENT_WRITE, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 0x14, "event type");

        // AcbEventWrite: address-carrying type (0x38) → 4-dword packet w/ addr.
        setup_cb(&ctx, cb, 0x500, 0x800);
        let a = hle_acb_event_write(&ctx, &[cb, 0x38, 0x1234_5678_9ABC_DEF8]);
        assert_eq!(read_u32(&ctx, a), pm4(4, IT_EVENT_WRITE, R_ZERO));
        assert_eq!(read_u32(&ctx, a + 4), 0x38 | 0x100, "address-type word");
        assert_eq!(
            read_u32(&ctx, a + 8),
            0x9ABC_DEF8 & !7,
            "addr low, 8-aligned"
        );
        assert_eq!(read_u32(&ctx, a + 12), 0x1234_5678, "addr high");
        // A non-address type → compact 2-dword packet.
        let a2 = hle_acb_event_write(&ctx, &[cb, 0x14, 0]);
        assert_eq!(read_u32(&ctx, a2), pm4(2, IT_EVENT_WRITE, R_ZERO));
        assert_eq!(read_u32(&ctx, a2 + 4), 0x14);

        // AcbDispatchIndirect: 4-dword indirect dispatch with a split args address.
        let a = hle_acb_dispatch_indirect(&ctx, &[cb, 0x00AB_1234_5678_0000, 0]);
        assert_eq!(read_u32(&ctx, a), pm4(4, IT_DISPATCH_INDIRECT, R_ZERO));
        assert_eq!(read_u32(&ctx, a + 4), 0x5678_0000, "args low");
        assert_eq!(read_u32(&ctx, a + 8), 0x00AB_1234, "args high");
        assert_eq!(read_u32(&ctx, a + 12), 0x41, "initiator");
    }

    #[test]
    fn dcb_write_data_copies_inline_dwords_with_stack_args() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // Seed two source DWORDs at 0x300.
        let src = 0x300u64;
        assert!(ctx.mem.write(src, &0xDEAD_BEEFu32.to_le_bytes()));
        assert!(ctx.mem.write(src + 4, &0xCAFE_F00Du32.to_le_bytes()));
        // args: dcb, destination=2, cachePolicy=1, destAddr, dataAddr, count=2,
        // increment=1 (arg7 → args[6]), writeConfirm=1 (arg8 → args[7]).
        let dst_addr = 0x0000_0012_3456_0000u64;
        let ret = hle_dcb_write_data(&ctx, &[cb, 2, 1, dst_addr, src, 2, 1, 1]);
        assert_eq!(ret, 0x400);
        assert_eq!(
            read_u32(&ctx, 0x400),
            pm4(2 + 4, IT_NOP, R_WRITE_DATA),
            "header"
        );
        // control = dst | cache<<8 | increment<<16 | confirm<<24.
        assert_eq!(read_u32(&ctx, 0x404), 2 | (1 << 8) | (1 << 16) | (1 << 24));
        assert_eq!(read_u32(&ctx, 0x408), dst_addr as u32, "dst low");
        assert_eq!(read_u32(&ctx, 0x40C), (dst_addr >> 32) as u32, "dst high");
        assert_eq!(read_u32(&ctx, 0x410), 0xDEAD_BEEF, "inline dword 0");
        assert_eq!(read_u32(&ctx, 0x414), 0xCAFE_F00D, "inline dword 1");
        // A count over 0x3FFD is rejected.
        assert_eq!(
            hle_dcb_write_data(&ctx, &[cb, 0, 0, dst_addr, src, 0x3FFE, 0, 0]),
            0
        );
    }

    #[test]
    fn dcb_wait_reg_mem_standard_and_wait_variants() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // operation=2 → standard wait (7 dwords, WAIT_REG_MEM op).
        // args: dcb, size=0, compare=3, op=2, cache=0, address,
        // reference (args[6]), mask (args[7]), pollCycles=80 (args[8]).
        let addr = 0x0000_0000_ABCD_0000u64;
        let ret = hle_dcb_wait_reg_mem(&ctx, &[cb, 0, 3, 2, 0, addr, 0x11, 0x22, 80]);
        assert_eq!(ret, 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(7, IT_WAIT_REG_MEM, R_ZERO));
        // standard-wait folds only op's low bit: (operation & 1) with op=2 → 0.
        assert_eq!(read_u32(&ctx, 0x404), 3, "compare | ((op & 1) << 8)");
        assert_eq!(read_u32(&ctx, 0x418), 80 / 40, "poll cycles /40");
        // size=0, op=0 → compact WAIT_MEM32 (6 dwords).
        let ret2 = hle_dcb_wait_reg_mem(&ctx, &[cb, 0, 1, 0, 0, addr, 0x33, 0x44, 40]);
        assert_eq!(read_u32(&ctx, ret2), pm4(6, IT_NOP, R_WAIT_MEM32));
        assert_eq!(read_u32(&ctx, ret2 + 12), 0x44, "mask low");
        assert_eq!(read_u32(&ctx, ret2 + 16), 1, "compare|op (op=0)");
        // Out-of-range compare is rejected.
        assert_eq!(
            hle_dcb_wait_reg_mem(&ctx, &[cb, 0, 8, 0, 0, addr, 0, 0, 0]),
            0
        );
    }

    #[test]
    fn dcb_dma_data_packs_control_and_addresses() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let dst = 0x0000_0011_1111_0000u64;
        let src = 0x0000_0022_2222_0000u64;
        // args 7-12 on stack: control4, sourceAddress, byteCount, control7..9.
        let ret = hle_dcb_dma_data(
            &ctx,
            &[cb, 1, 2, 3, dst, 0, 0xAA, src, 16, 0xBB, 0xCC, 0xDD],
        );
        assert_eq!(ret, 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(8, IT_NOP, R_DMA_DATA));
        assert_eq!(
            read_u32(&ctx, 0x404),
            1 | (2 << 8) | (3 << 16),
            "control0 (src_cache=0)"
        );
        assert_eq!(
            read_u32(&ctx, 0x408),
            0xAA | (0xBB << 8) | (0xCC << 16) | (0xDD << 24)
        );
        assert_eq!(read_u32(&ctx, 0x40C), 16, "byte count");
        assert_eq!(read_u64(&ctx, 0x410), dst, "dst address");
        assert_eq!(read_u64(&ctx, 0x418), src, "src address");
        // Unaligned byte count rejected.
        assert_eq!(
            hle_dcb_dma_data(&ctx, &[cb, 0, 0, 0, dst, 0, 0, src, 3, 0, 0, 0]),
            0
        );
    }

    #[test]
    fn dcb_acquire_mem_encodes_engine_and_size() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let base = 0x0000_0000_1234_0000u64; // low byte 0, top-40 clear.
        // pollCycles=80 is arg7 → args[6]. sizeBytes=0x100 (256).
        let ret = hle_dcb_acquire_mem(&ctx, &[cb, 1, 0xF, 0x7, base, 0x100, 80]);
        assert_eq!(ret, 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(8, IT_NOP, R_ACQUIRE_MEM));
        assert_eq!(read_u32(&ctx, 0x404), (1u32 << 31) | 0xF, "engine|cbDbOp");
        assert_eq!(read_u32(&ctx, 0x408), (0x100u64 >> 8) as u32, "size >>8");
        assert_eq!(read_u32(&ctx, 0x410), (base >> 8) as u32, "base >>8");
        assert_eq!(read_u32(&ctx, 0x418), 80 / 40, "poll /40");
        assert_eq!(read_u32(&ctx, 0x41C), 0x7, "gcr control");
        // no-size sentinel writes 0 in the size field.
        let ret2 = hle_dcb_acquire_mem(&ctx, &[cb, 0, 0, 0, base, u64::MAX, 40]);
        assert_eq!(read_u32(&ctx, ret2 + 8), 0, "no-size → 0");
        // engine > 1 rejected.
        assert_eq!(
            hle_dcb_acquire_mem(&ctx, &[cb, 2, 0, 0, base, 0x100, 40]),
            0
        );
    }

    #[test]
    fn set_registers_indirect_tags_the_right_space() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let regs = 0x00CD_1234_5678_0000u64;
        // Sh / Cx / Uc use discriminators 0x11 / 0x12 / 0x13.
        let sh = hle_dcb_set_sh_regs_indirect(&ctx, &[cb, regs, 4]);
        assert_eq!(read_u32(&ctx, sh), pm4(4, IT_NOP, R_SH_REGS_INDIRECT));
        assert_eq!(read_u32(&ctx, sh + 4), 4, "register count");
        assert_eq!(read_u32(&ctx, sh + 8), regs as u32);
        let cx = hle_dcb_set_cx_regs_indirect(&ctx, &[cb, regs, 2]);
        assert_eq!(read_u32(&ctx, cx), pm4(4, IT_NOP, R_CX_REGS_INDIRECT));
        let uc = hle_dcb_set_uc_regs_indirect(&ctx, &[cb, regs, 1]);
        assert_eq!(read_u32(&ctx, uc), pm4(4, IT_NOP, R_UC_REGS_INDIRECT));
    }

    #[test]
    fn dcb_dispatch_indirect_max_name_length_and_suspend() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // DcbDispatchIndirect: 3-dword packet (data offset + initiator).
        let ret = hle_dcb_dispatch_indirect(&ctx, &[cb, 0x20, 0]);
        assert_eq!(ret, 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(3, IT_DISPATCH_INDIRECT, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 0x20, "data offset");
        assert_eq!(read_u32(&ctx, 0x408), 0x41, "initiator");
        // Max name length getter writes 256.
        assert_eq!(hle_get_resource_max_name_length(&ctx, &[0x100]), 0);
        assert_eq!(read_u32(&ctx, 0x100), 256);
        assert_eq!(
            hle_get_resource_max_name_length(&ctx, &[0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // SuspendPoint is a no-op success.
        assert_eq!(hle_suspend_point(&ctx, &[]), 0);
        // DrawIndexOffset: 5-dword packet (count, offset, count, masked flags).
        let d = hle_dcb_draw_index_offset(&ctx, &[cb, 0x10, 6, 0xE000_0001]);
        assert_eq!(read_u32(&ctx, d), pm4(5, IT_DRAW_INDEX_OFFSET_2, R_ZERO));
        assert_eq!(read_u32(&ctx, d + 4), 6);
        assert_eq!(read_u32(&ctx, d + 8), 0x10);
        assert_eq!(read_u32(&ctx, d + 16), 0xE000_0001);
        // Unknown filler: a single 0x8000_0000 dword.
        let f = hle_unknown_filler(&ctx, &[cb]);
        assert_eq!(read_u32(&ctx, f), 0x8000_0000);
    }

    #[test]
    fn push_marker_packs_the_string_and_init_validates() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // Marker string "AB" at 0x100.
        assert!(ctx.mem.write(0x100, b"AB\0"));
        let ret = hle_dcb_push_marker(&ctx, &[cb, 0x100]);
        assert_eq!(ret, 0x400);
        // len 2 → payload = max((2+4)/4,1) = 1 dword; packet = 2 dwords.
        assert_eq!(read_u32(&ctx, 0x400), pm4(2, IT_NOP, R_PUSH_MARKER));
        // 'A'=0x41, 'B'=0x42 packed little-endian → 0x00004241.
        assert_eq!(read_u32(&ctx, 0x404), 0x0000_4241);
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x408);
        // InitResourceRegistration validates its args.
        assert_eq!(
            hle_driver_init_resource_registration(&ctx, &[0x1000, 0x100, 4]),
            0
        );
        assert_eq!(
            hle_driver_init_resource_registration(&ctx, &[0, 0x100, 4]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // SetBaseIndirectArgs: SET_BASE with the base index folded into header.
        let b = hle_dcb_set_base_indirect_args(&ctx, &[cb, 3, 0x1234_5678_9ABC_DEF8]);
        assert_eq!(read_u32(&ctx, b), pm4(4, IT_SET_BASE, R_ZERO) | (3 << 1));
        assert_eq!(read_u32(&ctx, b + 8), 0x9ABC_DEF8 & !7);
        // Driver submit captures and decodes the descriptor's complete DCB.
        assert!(
            ctx.mem
                .write(0x900, &pm4(2, IT_NOP, R_DRAW_INDEX_AUTO).to_le_bytes())
        );
        assert!(ctx.mem.write(0x904, &3u32.to_le_bytes()));
        assert!(
            ctx.mem
                .write(0x908, &pm4(8, IT_NOP, R_RELEASE_MEM).to_le_bytes())
        );
        assert!(ctx.mem.write(0x90C, &0u32.to_le_bytes()));
        assert!(ctx.mem.write(0x910, &(2u32 << 16).to_le_bytes()));
        assert!(ctx.mem.write(0x914, &0x980u32.to_le_bytes()));
        assert!(ctx.mem.write(0x918, &0u32.to_le_bytes()));
        assert!(ctx.mem.write(0x91C, &0x7654_3210u32.to_le_bytes()));
        assert!(ctx.mem.write(0x920, &0xfedc_ba98u32.to_le_bytes()));
        assert!(ctx.mem.write(0x924, &0u32.to_le_bytes()));
        assert!(ctx.mem.write(0x928, &pm4(6, IT_NOP, R_FLIP).to_le_bytes()));
        assert!(ctx.mem.write(0x92C, &1u32.to_le_bytes()));
        assert!(ctx.mem.write(0x930, &1u32.to_le_bytes()));
        assert!(ctx.mem.write(0x934, &1u32.to_le_bytes()));
        assert!(ctx.mem.write(0x938, &0x89ab_cdefu32.to_le_bytes()));
        assert!(ctx.mem.write(0x93C, &0x0123_4567u32.to_le_bytes()));
        assert!(
            ctx.mem
                .write(0x940, &pm4(2, IT_EVENT_WRITE, R_ZERO).to_le_bytes())
        );
        assert!(ctx.mem.write(0x944, &0x2au32.to_le_bytes()));
        let eq = kernel.create_equeue(0);
        assert_eq!(hle_driver_add_eq_event(&ctx, &[eq, 0x2a, 0xCAFE]), 0);
        assert!(ctx.mem.write(0x180, &0x900u64.to_le_bytes()));
        assert!(ctx.mem.write(0x188, &18u32.to_le_bytes()));
        assert_eq!(hle_driver_submit_dcb(&ctx, &[0x180]), 0);
        assert_eq!(hle_driver_submit_acb(&ctx, &[7, 0x180]), 0);
        assert_eq!(
            kernel
                .agc_submission_count
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        assert_eq!(
            kernel
                .agc_draw_packet_count
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        assert_eq!(
            kernel
                .agc_last_dcb_address
                .load(std::sync::atomic::Ordering::Relaxed),
            0x900
        );
        assert_eq!(
            kernel
                .agc_last_dcb_dwords
                .load(std::sync::atomic::Ordering::Relaxed),
            18
        );
        assert_eq!(read_u64(&ctx, 0x980), 0xfedc_ba98_7654_3210);
        assert!(
            kernel
                .kernel_equeue_events
                .get(&(eq, 0x2a))
                .unwrap()
                .triggered
        );
        assert_eq!(
            kernel
                .video_out_flip_count
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        assert_eq!(
            kernel
                .video_out_last_flip_arg
                .load(std::sync::atomic::Ordering::Relaxed),
            0x0123_4567_89ab_cdef
        );
        assert_eq!(
            hle_driver_submit_dcb(&ctx, &[0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // SubmitMultiDcbs: two buffers with valid addresses + sizes. The size
        // array is u32-per-entry (stride 4) — the SharpEmu ABI.
        assert!(ctx.mem.write(0x1A0, &0x400u64.to_le_bytes()));
        assert!(ctx.mem.write(0x1A8, &0x500u64.to_le_bytes()));
        assert!(ctx.mem.write(0x1C0, &8u32.to_le_bytes()));
        assert!(ctx.mem.write(0x1C4, &4u32.to_le_bytes()));
        assert_eq!(hle_driver_submit_multi_dcbs(&ctx, &[0x1A0, 0x1C0, 2]), 0);
        assert_eq!(
            hle_driver_submit_multi_dcbs(&ctx, &[0, 0x1C0, 2]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // Query: required = resourceCount*0x118 + ownerCount*0x1E0.
        assert_eq!(hle_driver_query_resource_memory(&ctx, &[0x1E0, 2, 3]), 0);
        assert_eq!(read_u64(&ctx, 0x1E0), 2 * 0x118 + 3 * 0x1E0);
    }

    /// M2 HLE seam: `SubmitDcb` remains a thin ABI adapter and forwards the
    /// exact command buffer to the process-owned GPU submission interface.
    #[test]
    fn submit_dcb_with_draw_drives_m2_gpu_session() {
        #[derive(Default)]
        struct RecordingGpu {
            submissions: std::sync::Mutex<Vec<(Vec<u32>, xps5x_core::subsystems::GpuQueue)>>,
        }
        impl xps5x_core::subsystems::GpuSubmissionSubsystem for RecordingGpu {
            fn submit(&self, words: Vec<u32>, queue: xps5x_core::subsystems::GpuQueue) {
                self.submissions.lock().unwrap().push((words, queue));
            }
            fn map_shader_metadata(
                &self,
                _code_address: u64,
                _data: xps5x_core::subsystems::ShaderMappedData,
            ) {
            }
            fn present_scanout(
                &self,
                _address: u64,
                _descriptor: Option<xps5x_core::subsystems::ScanoutDescriptor>,
            ) {
            }
            fn wait_idle(&self) {}
            fn stats(&self) -> xps5x_core::subsystems::GpuSubmissionStats {
                xps5x_core::subsystems::GpuSubmissionStats {
                    submitted: self.submissions.lock().unwrap().len() as u64,
                    ..Default::default()
                }
            }
        }

        let (kernel, mem, alloc) = ctx_env();
        let gpu = RecordingGpu::default();
        let ctx = crate::test_ctx_with_gpu(&kernel, &mem, &alloc, &gpu);

        let words = xps5x_gpu::build_m2_draw_dcb();
        let byte_len = words.len() * 4;
        let mut bytes = Vec::with_capacity(byte_len);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        assert!(ctx.mem.write(0xB00, &bytes));
        assert!(ctx.mem.write(0x280, &0xB00u64.to_le_bytes()));
        assert!(ctx.mem.write(0x288, &(words.len() as u32).to_le_bytes()));

        assert_eq!(hle_driver_submit_dcb(&ctx, &[0x280]), 0);
        assert_eq!(
            kernel
                .agc_draw_packet_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        let submissions = gpu.submissions.lock().unwrap();
        assert_eq!(submissions.len(), 1);
        assert_eq!(submissions[0].0, words);
        assert_eq!(submissions[0].1, xps5x_core::subsystems::GpuQueue::Graphics);
    }

    /// SharpEmu `DcbJump` parity: 4-DWORD INDIRECT_BUFFER chain packet, with
    /// `CbBranch` emitting the identical chain (modeled on Jump — see docs).
    #[test]
    fn dcb_jump_and_cb_branch_emit_the_chain_packet() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let target = 0x0001_2345_6789_ABC0u64;
        let ret = hle_dcb_jump(&ctx, &[cb, target, 0x12_3456]);
        assert_eq!(ret, 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(4, IT_INDIRECT_BUFFER, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 0x6789_ABC0, "target lo32");
        assert_eq!(read_u32(&ctx, 0x408), 0x0001_2345 & 0xFFFF, "target hi16");
        assert_eq!(
            read_u32(&ctx, 0x40C),
            0x12_3456 & 0xF_FFFF,
            "size & 0xFFFFF"
        );
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x410);
        // Branch: byte-identical emission at the next cursor position.
        assert_eq!(hle_cb_branch(&ctx, &[cb, target, 8]), 0x410);
        assert_eq!(read_u32(&ctx, 0x410), pm4(4, IT_INDIRECT_BUFFER, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x41C), 8);
        assert_eq!(hle_dcb_jump(&ctx, &[0, target, 8]), 0, "null dcb → 0");
    }

    /// DRAW_INDIRECT is the non-indexed sibling of DRAW_INDEX_INDIRECT: same
    /// 5-DWORD shape, opcode 0x24.
    #[test]
    fn draw_indirect_emits_the_five_dword_packet() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let ret = hle_dcb_draw_indirect(&ctx, &[cb, 0x120, 0xA038]);
        assert_eq!(ret, 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(5, IT_DRAW_INDIRECT, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 0x120, "argument-buffer data offset");
        assert_eq!(read_u32(&ctx, 0x408), 0);
        assert_eq!(read_u32(&ctx, 0x40C), 0);
        assert_eq!(read_u32(&ctx, 0x410), 0xA038, "modifier");
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x400 + 20);
    }

    /// SetNumRegisters overwrites the indirect packet's `+4` count field that
    /// AddRegisters accumulates into; predication is a SharpEmu-parity no-op.
    #[test]
    fn set_num_registers_overwrites_the_indirect_count_field() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // Emit an indirect Cx set-registers packet: header, count, addr lo/hi.
        let cmd = hle_dcb_set_cx_regs_indirect(&ctx, &[cb, 0x9000, 3]);
        assert_eq!(read_u32(&ctx, cmd + 4), 3);
        // Add accumulates; SetNum overwrites.
        assert_eq!(hle_add_indirect_patch_registers(&ctx, &[cmd, 5]), 0);
        assert_eq!(read_u32(&ctx, cmd + 4), 8);
        assert_eq!(
            hle_set_indirect_patch_set_num_registers(&ctx, &[cmd, 21]),
            0
        );
        assert_eq!(read_u32(&ctx, cmd + 4), 21);
        assert_eq!(
            hle_set_indirect_patch_set_num_registers(&ctx, &[0, 21]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // Predication accepts and changes nothing (SharpEmu OK no-op).
        assert_eq!(hle_set_packet_predication(&ctx, &[cmd]), 0);
        assert_eq!(read_u32(&ctx, cmd + 4), 21);
    }

    /// SetMarker preserves the string as a plain skippable NOP payload (no
    /// push/pop nesting side effects).
    #[test]
    fn set_marker_preserves_the_string_in_a_plain_nop() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        assert!(ctx.mem.write(0x100, b"frame\0"));
        let ret = hle_dcb_set_marker(&ctx, &[cb, 0x100]);
        assert_eq!(ret, 0x400);
        // "frame" = 5 bytes → (5+4)/4 = 2 payload dwords + header = 3 dwords.
        assert_eq!(read_u32(&ctx, 0x400), pm4(3, IT_NOP, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), u32::from_le_bytes(*b"fram"));
        assert_eq!(read_u32(&ctx, 0x408), u32::from(b'e'));
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x40C);
    }

    /// The multi-buffer submissions loop the REAL single-submit path: every
    /// array entry reaches `ctx.gpu.submit` on the right queue (u64 addresses
    /// stride 8, u32 dword sizes stride 4 — the SharpEmu ABI).
    #[test]
    fn multi_submit_routes_each_buffer_to_the_gpu_queue() {
        #[derive(Default)]
        struct RecordingGpu {
            submissions: std::sync::Mutex<Vec<(Vec<u32>, xps5x_core::subsystems::GpuQueue)>>,
        }
        impl xps5x_core::subsystems::GpuSubmissionSubsystem for RecordingGpu {
            fn submit(&self, words: Vec<u32>, queue: xps5x_core::subsystems::GpuQueue) {
                self.submissions.lock().unwrap().push((words, queue));
            }
            fn map_shader_metadata(
                &self,
                _code_address: u64,
                _data: xps5x_core::subsystems::ShaderMappedData,
            ) {
            }
            fn present_scanout(
                &self,
                _address: u64,
                _descriptor: Option<xps5x_core::subsystems::ScanoutDescriptor>,
            ) {
            }
            fn wait_idle(&self) {}
            fn stats(&self) -> xps5x_core::subsystems::GpuSubmissionStats {
                xps5x_core::subsystems::GpuSubmissionStats::default()
            }
        }

        let (kernel, mem, alloc) = ctx_env();
        let gpu = RecordingGpu::default();
        let ctx = crate::test_ctx_with_gpu(&kernel, &mem, &alloc, &gpu);

        // Two tiny well-formed DCBs (a 2-dword NOP each).
        for base in [0x400u64, 0x500] {
            assert!(ctx.mem.write(base, &pm4(2, IT_NOP, R_ZERO).to_le_bytes()));
            assert!(ctx.mem.write(base + 4, &0u32.to_le_bytes()));
        }
        assert!(ctx.mem.write(0x1A0, &0x400u64.to_le_bytes()));
        assert!(ctx.mem.write(0x1A8, &0x500u64.to_le_bytes()));
        assert!(ctx.mem.write(0x1C0, &2u32.to_le_bytes()));
        assert!(ctx.mem.write(0x1C4, &2u32.to_le_bytes()));

        assert_eq!(hle_driver_submit_multi_dcbs(&ctx, &[0x1A0, 0x1C0, 2]), 0);
        assert_eq!(hle_driver_submit_multi_acbs(&ctx, &[0x1A0, 0x1C0, 1]), 0);
        let submissions = gpu.submissions.lock().unwrap();
        assert_eq!(submissions.len(), 3, "2 DCBs + 1 ACB submitted");
        let expected = vec![pm4(2, IT_NOP, R_ZERO), 0];
        assert_eq!(submissions[0].0, expected);
        assert_eq!(submissions[0].1, xps5x_core::subsystems::GpuQueue::Graphics);
        assert_eq!(submissions[1].1, xps5x_core::subsystems::GpuQueue::Graphics);
        assert_eq!(
            submissions[2].1,
            xps5x_core::subsystems::GpuQueue::AsyncCompute
        );
        drop(submissions);

        // Capture trigger: OK no-op.
        assert_eq!(hle_driver_trigger_capture(&ctx, &[1, 2]), 0);

        // Provider-aware registration: the whole Driver family resolves from
        // BOTH import libraries (the measured Plague Tale gap was
        // SubmitMultiDcbs registered only under libSceAgc).
        let registry = HleRegistry::new();
        for name in [
            "sceAgcDriverSubmitMultiDcbs",
            "sceAgcDriverAgrSubmitMultiDcbs",
            "sceAgcDriverSubmitMultiAcbs",
            "sceAgcDriverTriggerCapture",
            "sceAgcDriverGetResourceRegistrationMaxNameLength",
            "sceAgcDriverRegisterDefaultOwner",
            "sceAgcDriverGetDefaultOwner",
            "sceAgcDriverDeleteEqEvent",
            "sceAgcDriverQueryResourceRegistrationUserMemoryRequirements",
            "sceAgcDriverInitResourceRegistration",
        ] {
            assert!(
                registry.is_implemented("libSceAgc", name),
                "missing libSceAgc::{name}"
            );
            assert!(
                registry.is_implemented("libSceAgcDriver", name),
                "missing libSceAgcDriver::{name}"
            );
        }
        for name in [
            "sceAgcDcbJump",
            "sceAgcCbBranch",
            "sceAgcDcbDrawIndirect",
            "sceAgcSetPacketPredication",
            "sceAgcSetCxRegIndirectPatchSetNumRegisters",
            "sceAgcDcbSetMarker",
        ] {
            assert!(
                registry.is_implemented("libSceAgc", name),
                "missing libSceAgc::{name}"
            );
        }
    }

    #[test]
    fn driver_owner_and_eq_events() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Default-owner register/get round-trips through kernel state.
        assert_eq!(hle_driver_register_default_owner(&ctx, &[42]), 0);
        assert_eq!(hle_driver_get_default_owner(&ctx, &[0x100]), 0);
        assert_eq!(read_u32(&ctx, 0x100), 42);
        assert_eq!(
            hle_driver_get_default_owner(&ctx, &[0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert!(ctx.mem.write(0x180, b"renderer\0"));
        assert_eq!(
            hle_driver_register_owner(&ctx, &[0x140, 0x180]),
            0,
            "SDKs may use driver-owned tracking without a user-memory arena"
        );
        assert_eq!(
            hle_driver_init_resource_registration(&ctx, &[0x1000, 0x800, 4]),
            0
        );
        assert_eq!(hle_driver_register_owner(&ctx, &[0x140, 0x180]), 0);
        let owner = read_u32(&ctx, 0x140);
        assert_ne!(owner, 0);
        assert_ne!(owner, 42, "default owner handle is reserved");
        assert_eq!(
            kernel
                .agc_resource_owners
                .get(&owner)
                .map(|entry| entry.value().clone()),
            Some("renderer".to_owned())
        );
        assert!(ctx.mem.write(0x1A0, b"framebuffer\0"));
        assert_eq!(
            hle_driver_register_resource(&ctx, &[0x160, owner as u64, 0x500, 0x80, 0x1A0, 3, 7]),
            0
        );
        let resource = read_u32(&ctx, 0x160);
        let metadata = kernel.agc_resources.get(&resource).unwrap();
        assert_eq!(metadata.owner, owner);
        assert_eq!(metadata.address, 0x500);
        assert_eq!(metadata.size, 0x80);
        assert_eq!(metadata.name, "framebuffer");
        assert_eq!(metadata.resource_type, 3);
        assert_eq!(metadata.flags, 7);
        let registry = HleRegistry::new();
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0x5ff3_66e4_a2d1_11e8
                        && key == "libSceAgcDriver::sceAgcDriverRegisterOwner"
                })
        );
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0x5b9c_f879_9ae3_11ab
                        && key == "libSceAgcDriver::sceAgcDriverRegisterResource"
                })
        );
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0x5209_4921_98c6_b2c3 && key == "libSceAgcDriver::sceAgcDriverSubmitDcb"
                })
        );
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0x0211_afa4_84eb_7f83
                        && key == "libSceAgcDriver::sceAgcDriverAgrSubmitDcb"
                })
        );
        // AddEqEvent on an unknown queue → NOT_FOUND; on a real one → OK.
        assert_eq!(
            hle_driver_add_eq_event(&ctx, &[0xDEAD, 5, 9]),
            SCE_ERROR_NOT_FOUND
        );
        let eq = kernel.create_equeue(0);
        assert_eq!(hle_driver_add_eq_event(&ctx, &[eq, 5, 9]), 0);
        assert!(kernel.kernel_equeue_events.contains_key(&(eq, 5)));
        assert_eq!(hle_driver_delete_eq_event(&ctx, &[eq, 5]), 0);
        assert!(!kernel.kernel_equeue_events.contains_key(&(eq, 5)));
    }

    #[test]
    fn patch_address_setters_validate_packet_identity() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // A RELEASE_MEM packet (NOP op + R_RELEASE_MEM discriminator) at 0x200.
        assert!(
            ctx.mem
                .write(0x200, &pm4(8, IT_NOP, R_RELEASE_MEM).to_le_bytes())
        );
        assert_eq!(
            hle_queue_eop_patch_address(&ctx, &[0x200, 0xDEAD_BEEF_0000]),
            0
        );
        assert_eq!(read_u64(&ctx, 0x200 + 12), 0xDEAD_BEEF_0000);
        // Wrong packet identity → rejected.
        assert!(
            ctx.mem
                .write(0x200, &pm4(8, IT_NOP, R_DMA_DATA).to_le_bytes())
        );
        assert_eq!(
            hle_queue_eop_patch_address(&ctx, &[0x200, 1]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // DMA_DATA patch → +16.
        assert_eq!(hle_dma_data_patch_address(&ctx, &[0x200, 0xC0DE_0000]), 0);
        assert_eq!(read_u64(&ctx, 0x200 + 16), 0xC0DE_0000);
        // WAIT_REG_MEM op → field offset 8; wait-mem NOP → offset 4.
        assert!(
            ctx.mem
                .write(0x300, &pm4(7, IT_WAIT_REG_MEM, R_ZERO).to_le_bytes())
        );
        assert_eq!(hle_wait_reg_mem_patch_address(&ctx, &[0x300, 0xAA00]), 0);
        assert_eq!(read_u64(&ctx, 0x300 + 8), 0xAA00);
        assert!(
            ctx.mem
                .write(0x300, &pm4(6, IT_NOP, R_WAIT_MEM32).to_le_bytes())
        );
        assert_eq!(hle_wait_reg_mem_patch_address(&ctx, &[0x300, 0xBB00]), 0);
        assert_eq!(read_u64(&ctx, 0x300 + 4), 0xBB00);

        // WRITE_DATA destination address lives at +8.
        assert!(
            ctx.mem
                .write(0x380, &pm4(6, IT_NOP, R_WRITE_DATA).to_le_bytes())
        );
        assert_eq!(
            hle_write_data_patch_address(&ctx, &[0x380, 0x1234_5678_9000]),
            0
        );
        assert_eq!(read_u64(&ctx, 0x380 + 8), 0x1234_5678_9000);
        assert!(ctx.mem.write(0x380, &pm4(6, IT_NOP, R_ZERO).to_le_bytes()));
        assert_eq!(
            hle_write_data_patch_address(&ctx, &[0x380, 1]),
            SCE_ERROR_INVALID_ARGUMENT
        );

        let registry = HleRegistry::new();
        register(&registry);
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0x7cf4_8275_0c60_a52c && key == "libSceAgc::sceAgcWriteDataPatchAddress"
                })
        );
    }

    #[test]
    fn get_data_packet_payload_address() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let out: u64 = 0x100;
        let command: u64 = 0x200;
        // type != 0 → payload at command + 8.
        assert_eq!(
            hle_get_data_packet_payload_address(&ctx, &[out, command, 1]),
            0
        );
        assert_eq!(read_u64(&ctx, out), command + 8);
        // type == 0, ordinary header → payload at command + 4.
        assert!(ctx.mem.write(command, &0xC001_0000u32.to_le_bytes()));
        assert_eq!(
            hle_get_data_packet_payload_address(&ctx, &[out, command, 0]),
            0
        );
        assert_eq!(read_u64(&ctx, out), command + 4);
        // type == 0, max-length NOP header → payload 0.
        assert!(ctx.mem.write(command, &0x3FFF_0000u32.to_le_bytes()));
        assert_eq!(
            hle_get_data_packet_payload_address(&ctx, &[out, command, 0]),
            0
        );
        assert_eq!(read_u64(&ctx, out), 0);
        assert_eq!(
            hle_get_data_packet_payload_address(&ctx, &[0, command, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );

        let registry = HleRegistry::new();
        register(&registry);
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0x57ef_9480_1b50_867d
                        && key == "libSceAgc::sceAgcGetDataPacketPayloadAddress"
                })
        );
    }

    #[test]
    fn indirect_patch_set_address_and_add_registers() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let command: u64 = 0x100;
        // SetAddress writes the register block address at command+8/+12.
        assert_eq!(
            hle_set_indirect_patch_address(&ctx, &[command, 0x00AB_1234_5678_0000]),
            0
        );
        assert_eq!(read_u32(&ctx, command + 8), 0x5678_0000);
        assert_eq!(read_u32(&ctx, command + 12), 0x00AB_1234);
        // AddRegisters accumulates the count field at command+4.
        assert!(ctx.mem.write(command + 4, &10u32.to_le_bytes()));
        assert_eq!(hle_add_indirect_patch_registers(&ctx, &[command, 5]), 0);
        assert_eq!(read_u32(&ctx, command + 4), 15);
        assert_eq!(hle_add_indirect_patch_registers(&ctx, &[command, 3]), 0);
        assert_eq!(read_u32(&ctx, command + 4), 18);
        // Validation.
        assert_eq!(
            hle_set_indirect_patch_address(&ctx, &[0, 1]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_add_indirect_patch_registers(&ctx, &[0, 1]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // All 6 NIDs resolve (Cx/Sh/Uc × SetAddress/AddRegisters).
        let reg = HleRegistry::new();
        for space in ["Cx", "Sh", "Uc"] {
            for suffix in ["SetAddress", "AddRegisters"] {
                let name = format!("sceAgcSet{space}RegIndirectPatch{suffix}");
                assert!(
                    reg.call(&ctx, "libSceAgc", &name, &[command, 1]).is_some(),
                    "{name} must be registered"
                );
            }
        }
    }

    #[test]
    fn create_interpolant_mapping_builds_input_cntl_registers() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let registers: u64 = 0x100;
        let geometry: u64 = 0x200;
        let pixel: u64 = 0x280;
        // GS: 2 output semantics at 0x300.
        assert!(ctx.mem.write(
            geometry + SHADER_OUTPUT_SEMANTICS_OFFSET,
            &0x300u64.to_le_bytes()
        ));
        assert!(ctx.mem.write(
            geometry + SHADER_NUM_OUTPUT_SEMANTICS_OFFSET,
            &2u32.to_le_bytes()
        ));
        // PS: input semantics at 0x340; slot 1 has the flat bit (bit 22) set.
        assert!(ctx.mem.write(
            pixel + SHADER_INPUT_SEMANTICS_OFFSET,
            &0x340u64.to_le_bytes()
        ));
        assert!(ctx.mem.write(0x340, &0u32.to_le_bytes())); // slot 0: not flat
        assert!(ctx.mem.write(0x344, &(1u32 << 22).to_le_bytes())); // slot 1: flat
        assert_eq!(
            hle_create_interpolant_mapping(&ctx, &[registers, geometry, pixel]),
            0
        );
        // Slot 0: cntl register 0x191, value = 0 (i=0, not flat).
        assert_eq!(read_u32(&ctx, registers), SPI_PS_INPUT_CNTL0);
        assert_eq!(read_u32(&ctx, registers + 4), 0);
        // Slot 1: cntl 0x192, value = 1 | 0x400 (flat).
        assert_eq!(read_u32(&ctx, registers + 8), SPI_PS_INPUT_CNTL0 + 1);
        assert_eq!(read_u32(&ctx, registers + 12), 1 | 0x400);
        // Slot 2 (>= count): cntl 0x193, value = 0.
        assert_eq!(read_u32(&ctx, registers + 16), SPI_PS_INPUT_CNTL0 + 2);
        assert_eq!(read_u32(&ctx, registers + 20), 0);
        assert_eq!(
            hle_create_interpolant_mapping(&ctx, &[0, geometry, pixel]),
            SCE_ERROR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn create_shader_relocates_fields_and_binds_code() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let header: u64 = 0x200;
        // Valid magic + version.
        assert!(ctx.mem.write(header, &SHADER_FILE_HEADER.to_le_bytes()));
        assert!(ctx.mem.write(header + 4, &SHADER_VERSION.to_le_bytes()));
        // A self-relative Cx-registers pointer of +0x40 (→ absolute header+0x18+0x40).
        assert!(
            ctx.mem
                .write(header + SHADER_CX_REGISTERS_OFFSET, &0x40u64.to_le_bytes())
        );
        // The SH table starts with the pixel program-base placeholders emitted
        // by the compiler. CreateShader must replace their zero values with the
        // supplied code address before a later indirect bind consumes them.
        let sh_registers = header + SHADER_SH_REGISTERS_OFFSET + 0x80;
        assert!(
            ctx.mem
                .write(header + SHADER_SH_REGISTERS_OFFSET, &0x80u64.to_le_bytes())
        );
        assert!(ctx.mem.write(sh_registers, &0x08u32.to_le_bytes()));
        assert!(ctx.mem.write(sh_registers + 4, &0u32.to_le_bytes()));
        assert!(ctx.mem.write(sh_registers + 8, &0x09u32.to_le_bytes()));
        assert!(ctx.mem.write(sh_registers + 12, &0u32.to_le_bytes()));
        assert!(ctx.mem.write(header + SHADER_TYPE_OFFSET, &[1]));
        assert!(ctx.mem.write(header + SHADER_NUM_SH_REGISTERS_OFFSET, &[2]));
        // User-data field left 0 (skips user-data relocation).
        let dest: u64 = 0x100;
        let code: u64 = 0xC0DE_0000;
        assert_eq!(hle_create_shader(&ctx, &[dest, header, code]), 0);
        // Cx field relocated to absolute.
        let mut b = [0u8; 8];
        assert!(ctx.mem.read(header + SHADER_CX_REGISTERS_OFFSET, &mut b));
        assert_eq!(
            u64::from_le_bytes(b),
            header + SHADER_CX_REGISTERS_OFFSET + 0x40
        );
        // Code bound + shader object published to *dest.
        assert!(ctx.mem.read(header + SHADER_CODE_OFFSET, &mut b));
        assert_eq!(u64::from_le_bytes(b), code);
        assert!(ctx.mem.read(dest, &mut b));
        assert_eq!(u64::from_le_bytes(b), header);
        assert_eq!(read_u32(&ctx, sh_registers + 4), (code >> 8) as u32);
        assert_eq!(read_u32(&ctx, sh_registers + 12), (code >> 40) as u32);
        // A bad magic is rejected.
        assert!(ctx.mem.write(header, &0xDEADu32.to_le_bytes()));
        assert_eq!(
            hle_create_shader(&ctx, &[dest, header, code]),
            SCE_ERROR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn shader_mapped_data_owns_relocated_guest_tables() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let header: u64 = 0x200;
        let user_data: u64 = 0x300;
        let direct: u64 = 0x400;
        let sharp0: u64 = 0x420;
        let semantics: u64 = 0x440;
        assert!(
            ctx.mem
                .write(header + SHADER_USER_DATA_OFFSET, &user_data.to_le_bytes())
        );
        assert!(ctx.mem.write(
            header + SHADER_INPUT_SEMANTICS_OFFSET,
            &semantics.to_le_bytes()
        ));
        assert!(ctx.mem.write(
            header + SHADER_NUM_INPUT_SEMANTICS_OFFSET,
            &2u32.to_le_bytes()
        ));
        assert!(ctx.mem.write(user_data, &direct.to_le_bytes()));
        assert!(ctx.mem.write(user_data + 8, &sharp0.to_le_bytes()));
        assert!(ctx.mem.write(user_data + 0x28, &3u16.to_le_bytes()));
        assert!(ctx.mem.write(user_data + 0x2A, &4u16.to_le_bytes()));
        assert!(ctx.mem.write(user_data + 0x2C, &2u16.to_le_bytes()));
        assert!(ctx.mem.write(user_data + 0x2E, &1u16.to_le_bytes()));
        assert!(ctx.mem.write(direct, &7u16.to_le_bytes()));
        assert!(ctx.mem.write(direct + 2, &0xffffu16.to_le_bytes()));
        assert!(ctx.mem.write(sharp0, &0x8123u16.to_le_bytes()));
        assert!(ctx.mem.write(semantics, &0x1234_5678u32.to_le_bytes()));
        assert!(ctx.mem.write(semantics + 4, &0x89ab_cdefu32.to_le_bytes()));

        let mapped = read_shader_mapped_data(&ctx, header).expect("mapped metadata");
        let user = mapped.user_data.expect("user data");
        assert_eq!(user.direct_resource_offset, vec![7, 0xffff]);
        assert_eq!(user.eud_size_dw, 3);
        assert_eq!(user.srt_size_dw, 4);
        assert_eq!(user.sharp_resource_offset[0][0].offset_dw(), 0x123);
        assert_eq!(user.sharp_resource_offset[0][0].size(), 1);
        assert_eq!(mapped.input_semantics[0].raw, 0x1234_5678);
        assert_eq!(mapped.input_semantics[1].raw, 0x89ab_cdef);
    }

    #[test]
    fn create_prim_state_copies_specials_and_writes_prim_type() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Geometry shader at 0x200 → specials table pointer at +0x28.
        let geometry: u64 = 0x200;
        let specials: u64 = 0x280;
        assert!(
            ctx.mem
                .write(geometry + SHADER_SPECIALS_OFFSET, &specials.to_le_bytes())
        );
        // Specials entries: (offset, value) pairs at the documented sub-offsets.
        assert!(ctx.mem.write(
            specials + SPECIAL_VGT_SHADER_STAGES_EN_OFFSET,
            &[0x11u8, 0, 0, 0, 0xAA, 0, 0, 0]
        ));
        assert!(ctx.mem.write(
            specials + SPECIAL_GE_CNTL_OFFSET,
            &[0x22u8, 0, 0, 0, 0xBB, 0, 0, 0]
        ));
        let cx = 0x100;
        let uc = 0x140;
        assert_eq!(hle_create_prim_state(&ctx, &[cx, uc, 0, geometry, 5]), 0);
        // Stage-enable pair copied to cx[0..8].
        assert_eq!(read_u32(&ctx, cx), 0x11);
        assert_eq!(read_u32(&ctx, cx + 4), 0xAA);
        // GE_CNTL pair copied to uc[0..8].
        assert_eq!(read_u32(&ctx, uc), 0x22);
        assert_eq!(read_u32(&ctx, uc + 4), 0xBB);
        // VGT primitive type + the caller's primitive type.
        assert_eq!(read_u32(&ctx, uc + 16), VGT_PRIMITIVE_TYPE);
        assert_eq!(read_u32(&ctx, uc + 20), 5);
        // Validation: a non-zero hull shader is rejected.
        assert_eq!(
            hle_create_prim_state(&ctx, &[cx, uc, 1, geometry, 5]),
            SCE_ERROR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn acb_dispatch_indirect_is_registered_as_the_same_packet() {
        // Both NIDs resolve to the same 4-dword indirect-dispatch emitter.
        let reg = HleRegistry::new();
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        setup_cb(&ctx, 0x40, 0x400, 0x800);
        assert_eq!(
            reg.call(
                &ctx,
                "libSceAgc",
                "sceAgcAcbDispatchIndirect",
                &[0x40, 0, 0]
            ),
            Some(0x400)
        );
        assert_eq!(read_u32(&ctx, 0x400), pm4(4, IT_DISPATCH_INDIRECT, R_ZERO));
    }

    #[test]
    fn allocation_fails_when_the_buffer_is_full() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        // Only 1 DWORD of space, but the packet needs 2 → fail, cursor unchanged.
        setup_cb(&ctx, cb, 0x400, 0x404);
        assert_eq!(hle_dcb_set_index_size(&ctx, &[cb, 2, 0]), 0);
        assert_eq!(
            read_u64(&ctx, cb + CB_CURSOR_UP),
            0x400,
            "cursor not advanced on failure"
        );
        // NULL command buffer → 0.
        assert_eq!(hle_dcb_set_index_size(&ctx, &[0, 2, 0]), 0);
    }

    #[test]
    fn pm4_header_matches_the_agc_encoding() {
        // 0xC0000000 | (len-2)<<16 | op<<8 | (reg&0x3F)<<2.
        // len 2 → (2-2)=0 length field, op 0x2A, reg 0.
        assert_eq!(pm4(2, IT_INDEX_TYPE, R_ZERO), 0xC000_0000 | (0x2A << 8));
        assert_eq!(
            pm4(7, IT_NOP, R_DRAW_INDEX_AUTO),
            0xC000_0000 | (5 << 16) | (0x10 << 8) | (0x04 << 2)
        );
    }

    #[test]
    fn register_defaults_materialize_in_guest_memory_with_kyty_layout() {
        // Needs a bigger arena than ctx_env's: the block is ~23 KiB.
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10000);
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let base = hle_get_register_defaults2(&ctx, &[8]);
        assert_ne!(base, 0, "defaults must materialize");
        // Cached: the same guest address on every later call — including a
        // version the reference would assert on (warn, don't fail).
        assert_eq!(hle_get_register_defaults2(&ctx, &[8]), base);
        assert_eq!(hle_get_register_defaults2(&ctx, &[7]), base);
        assert_eq!(hle_get_register_defaults2(&ctx, &[12]), base);
        let internal = hle_get_register_defaults2_internal(&ctx, &[8]);
        assert_eq!(internal, base + REG_DEFAULTS_HEADER_BYTES);

        // offsetof(RegisterDefaults, count) == 0x38 — Kyty asserts this
        // against the real SDK struct.
        assert_eq!(REG_DEFAULTS_COUNT_OFFSET, 0x38);
        assert_eq!(read_u32(&ctx, base + 0x38), 78 + 29 + 20, "set-1 triples");
        assert_eq!(read_u32(&ctx, internal + 0x38), 4 + 15 + 3, "set-2 triples");

        // tbl0..tbl2 and types are non-null guest pointers; tbl3 and
        // unknown[2] stay zero.
        let tbl0 = read_u64(&ctx, base);
        assert_ne!(tbl0, 0);
        assert_ne!(read_u64(&ctx, base + 8), 0);
        assert_ne!(read_u64(&ctx, base + 0x10), 0);
        assert_eq!(read_u64(&ctx, base + 0x18), 0);
        assert_eq!(read_u64(&ctx, base + 0x20), 0);
        assert_eq!(read_u64(&ctx, base + 0x28), 0);
        let types = read_u64(&ctx, base + 0x30);
        assert_ne!(types, 0);

        // First cx pointer aims at reg[0] of CB_COLOR_CONTROL (0x202,
        // 0x00cc0010), with the type hash 4 bytes before it — exactly
        // Kyty's &g_cx_reg_info1[0].reg[0].
        let first_cx = read_u64(&ctx, tbl0);
        assert_ne!(first_cx, 0);
        assert_eq!(read_u32(&ctx, first_cx), 0x202, "CB_COLOR_CONTROL offset");
        assert_eq!(read_u32(&ctx, first_cx + 4), 0x00cc_0010);
        assert_eq!(read_u32(&ctx, first_cx - 4), 0xE24F_806D, "type hash");

        // Index triples are (type_hash, id*4 + table, 0).
        assert_eq!(read_u32(&ctx, types), 0xE24F_806D);
        assert_eq!(read_u32(&ctx, types + 4), 0);
        assert_eq!(read_u32(&ctx, types + 8), 0);
        assert_eq!(read_u32(&ctx, types + 12), 0xF6C2_8182, "cx entry 1");
        assert_eq!(read_u32(&ctx, types + 16), 4, "id 1*4 + table 0");
        // The first sh triple follows all 78 cx triples: id 0*4 + table 1.
        assert_eq!(read_u32(&ctx, types + 78 * 12 + 4), 1);

        // Internal set: first cx2 entry is DB_DFSM_CONTROL (0x0E).
        let first_cx2 = read_u64(&ctx, read_u64(&ctx, internal));
        assert_eq!(read_u32(&ctx, first_cx2), 0x0E, "DB_DFSM_CONTROL offset");
        assert_eq!(read_u32(&ctx, first_cx2 - 4), 0x8FB4_EDB5, "type hash");
    }

    #[test]
    fn set_index_count_emits_an_index_buffer_size_packet() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        assert_eq!(hle_dcb_set_index_count(&ctx, &[cb, 9]), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(2, IT_INDEX_BUFFER_SIZE, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 9);
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x408);
        assert_eq!(hle_dcb_set_index_count(&ctx, &[0, 9]), 0, "null dcb");
    }

    #[test]
    fn write_data_accepts_placeholder_null_destination_then_patch_binds_it() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // Template packet: null destination AND null data, both bound later —
        // Minecraft's WRITE_DATA-template flow (build, memcpy, patch).
        let addr = hle_dcb_write_data(&ctx, &[cb, 5, 0, 0, 0, 2, 1, 0]);
        assert_eq!(addr, 0x400, "placeholder WRITE_DATA must be emitted");
        assert_eq!(read_u32(&ctx, addr), pm4(6, IT_NOP, R_WRITE_DATA));
        assert_eq!(read_u64(&ctx, addr + 8), 0, "placeholder destination");
        assert_eq!(read_u32(&ctx, addr + 16), 0, "zero-filled payload");
        // The later patch finds a real WRITE_DATA header and binds the
        // address — this exact flow returned 0x80020016 before placeholder
        // destinations were legal (no packet was ever emitted).
        assert_eq!(
            hle_write_data_patch_address(&ctx, &[addr, 0x00AB_CDEF_1000]),
            0
        );
        assert_eq!(read_u64(&ctx, addr + 8), 0x00AB_CDEF_1000);
    }

    #[test]
    fn indirect_multi_predication_and_copy_data_emit_and_advance() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // DrawIndexIndirectMulti: 9 DWORDs, standard PM4 opcode 0x38.
        let a = hle_dcb_draw_index_indirect_multi(&ctx, &[cb, 0x20, 3, 16, 0x9000, 0]);
        assert_eq!(a, 0x400);
        assert_eq!(
            read_u32(&ctx, a),
            pm4(9, IT_DRAW_INDEX_INDIRECT_MULTI, R_ZERO)
        );
        assert_eq!(read_u32(&ctx, a + 4), 0x20, "data offset");
        assert_eq!(read_u32(&ctx, a + 16), 3, "draw count");
        assert_eq!(read_u32(&ctx, a + 20), 0x9000, "count address low");
        assert_eq!(read_u32(&ctx, a + 28), 16, "stride");
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x400 + 36);
        // DrawIndirectMulti: same shape, opcode 0x2C.
        let b = hle_dcb_draw_indirect_multi(&ctx, &[cb, 0x40, 2, 16, 0, 0]);
        assert_eq!(b, 0x424);
        assert_eq!(read_u32(&ctx, b), pm4(9, IT_DRAW_INDIRECT_MULTI, R_ZERO));
        // SetPredication: 4 DWORDs (address lo/hi + control).
        let c = hle_dcb_set_predication(&ctx, &[cb, 0x0000_1234_5678, 1]);
        assert_ne!(c, 0);
        assert_eq!(read_u32(&ctx, c), pm4(4, IT_SET_PREDICATION, R_ZERO));
        assert_eq!(read_u32(&ctx, c + 4), 0x1234_5678);
        assert_eq!(read_u32(&ctx, c + 12), 1);
        // CopyData: 6 DWORDs (control + src lo/hi + dst lo/hi).
        let d = hle_dcb_copy_data(&ctx, &[cb, 0x100, 0xAAAA_0000, 0xBBBB_0000]);
        assert_ne!(d, 0);
        assert_eq!(read_u32(&ctx, d), pm4(6, IT_COPY_DATA, R_ZERO));
        assert_eq!(read_u32(&ctx, d + 4), 0x100);
        assert_eq!(read_u32(&ctx, d + 8), 0xAAAA_0000);
        assert_eq!(read_u32(&ctx, d + 16), 0xBBBB_0000);
        // A null buffer is rejected everywhere.
        assert_eq!(hle_dcb_draw_index_indirect_multi(&ctx, &[0]), 0);
        assert_eq!(hle_dcb_draw_indirect_multi(&ctx, &[0]), 0);
        assert_eq!(hle_dcb_set_predication(&ctx, &[0]), 0);
        assert_eq!(hle_dcb_copy_data(&ctx, &[0]), 0);
    }

    #[test]
    fn sh_register_range_get_size_matches_the_writer() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let before = read_u64(&ctx, cb + CB_CURSOR_UP);
        assert_ne!(hle_cb_set_sh_register_range(&ctx, &[cb, 0x10, 0, 3]), 0);
        let consumed = (read_u64(&ctx, cb + CB_CURSOR_UP) - before) / 4;
        assert_eq!(
            hle_cb_set_sh_register_range_get_size(&ctx, &[3]),
            consumed,
            "GetSize must always match what the writer emits"
        );
    }

    #[test]
    fn debug_raise_exception_and_unknown_stubs_log_and_return_ok() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_debug_raise_exception(&ctx, &[0xDEAD, 1, 2]), 0);
        assert_eq!(hle_set_range_predication(&ctx, &[1, 2, 3]), 0);
        assert_eq!(hle_unknown_krz_wek_v120(&ctx, &[]), 0);
    }

    #[test]
    fn minecraft_renderdragon_batch_is_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceAgcGetRegisterDefaults2",
            "sceAgcGetRegisterDefaults2Internal",
            "sceAgcDcbCopyData",
            "sceAgcDcbDrawIndexIndirectMulti",
            "sceAgcDcbDrawIndirectMulti",
            "sceAgcDcbSetIndexCount",
            "sceAgcDcbSetPredication",
            "sceAgcSetRangePredication",
            "sceAgcDebugRaiseException",
            "sceAgcCbSetShRegisterRangeDirectGetSize",
        ] {
            assert!(
                registry.is_implemented("libSceAgc", name),
                "{name} must be registered"
            );
        }
        // The two identities known only by NID resolve via explicit bindings
        // (their names do not hash to the measured import NIDs).
        let overrides = registry.registered_nid_overrides();
        assert!(overrides.iter().any(|(nid, key)| {
            *nid == 0x1d5e_23f8_4d0c_0471 && key == "libSceAgc::sceAgcCreateInterpolantMapping"
        }));
        assert!(overrides.iter().any(|(nid, key)| {
            *nid == 0xfca4_7359_e915_d76d && key == "libSceAgc::sceAgcUnknownKRzWekV120"
        }));
    }

    #[test]
    fn astrobot_builder_batch_is_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceAgcAcbDmaData",
            "sceAgcAcbCopyData",
            "sceAgcAcbAcquireMem",
            "sceAgcAcbWaitRegMem",
            "sceAgcAcbPushMarker",
            "sceAgcAcbPopMarker",
            "sceAgcCbSetShRegistersDirect",
            "sceAgcCbDispatchGetSize",
            "sceAgcCbNopGetSize",
            "sceAgcDcbDrawIndexIndirect",
            "sceAgcDcbStallCommandBufferParser",
            "sceAgcDcbGetLodStats",
        ] {
            assert!(
                registry.is_implemented("libSceAgc", name),
                "{name} must be registered"
            );
        }
    }

    #[test]
    fn acb_dma_data_emits_seven_dword_packet() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let dst = 0x0000_00AB_1234_5678u64;
        let src = 0x0000_00CD_9ABC_DEF0u64;
        // sourceOrImmediate/byteCount are stack args → args[6]/args[7].
        let args = [cb, 0x11, 0x22, dst, 0, 0, src, 0x100];
        assert_eq!(hle_acb_dma_data(&ctx, &args), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(7, IT_NOP, R_DMA_DATA));
        assert_eq!(read_u64(&ctx, 0x404), dst, "destination u64 at +4");
        assert_eq!(read_u64(&ctx, 0x40C), src, "source-or-immediate u64 at +12");
        assert_eq!(read_u32(&ctx, 0x414), 0x100, "byte count");
        assert_eq!(
            read_u32(&ctx, 0x418),
            0x11 | (0x22 << 8),
            "sourceSelector | destinationSelector << 8"
        );
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x400 + 28);
        // Zero and oversized byte counts are rejected.
        assert_eq!(hle_acb_dma_data(&ctx, &[cb, 0, 0, 0, 0, 0, 0, 0]), 0);
        let too_big = 256 * 1024 * 1024 + 1;
        assert_eq!(hle_acb_dma_data(&ctx, &[cb, 0, 0, 0, 0, 0, 0, too_big]), 0);
    }

    #[test]
    fn acb_copy_data_matches_the_dcb_emission() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let src = 0xAAAA_0000_0000_1111u64;
        let dst = 0xBBBB_0000_0000_2222u64;
        assert_eq!(hle_acb_copy_data(&ctx, &[cb, 0x205, src, dst]), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(6, IT_COPY_DATA, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 0x205, "control");
        assert_eq!(read_u32(&ctx, 0x408), src as u32);
        assert_eq!(read_u32(&ctx, 0x40C), (src >> 32) as u32);
        assert_eq!(read_u32(&ctx, 0x410), dst as u32);
        assert_eq!(read_u32(&ctx, 0x414), (dst >> 32) as u32);
        assert_eq!(hle_acb_copy_data(&ctx, &[0]), 0, "null buffer rejected");
    }

    #[test]
    fn acb_acquire_mem_emits_eight_dword_packet() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let base = 0x0000_0012_3456_7800u64; // 256-byte aligned, < 2^40
        assert_eq!(
            hle_acb_acquire_mem(&ctx, &[cb, 0xC0DE, base, 0x4000, 400]),
            0x400
        );
        assert_eq!(read_u32(&ctx, 0x400), pm4(8, IT_NOP, R_ACQUIRE_MEM));
        assert_eq!(read_u32(&ctx, 0x404), 0x8000_0000, "fixed DWORD1");
        assert_eq!(read_u32(&ctx, 0x408), 0x40, "sizeBytes >> 8");
        assert_eq!(read_u32(&ctx, 0x40C), 0);
        assert_eq!(read_u32(&ctx, 0x410), (base >> 8) as u32, "base >> 8");
        assert_eq!(read_u32(&ctx, 0x414), 0);
        assert_eq!(read_u32(&ctx, 0x418), 10, "pollCycles / 40");
        assert_eq!(read_u32(&ctx, 0x41C), 0xC0DE, "gcrControl");
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x420);
        // u64::MAX size means "no size" → size field 0.
        let a = hle_acb_acquire_mem(&ctx, &[cb, 0, base, u64::MAX, 0]);
        assert_eq!(read_u32(&ctx, a + 8), 0, "no-size form writes 0");
        // Misaligned size / base rejected.
        assert_eq!(hle_acb_acquire_mem(&ctx, &[cb, 0, base, 0x123, 0]), 0);
        assert_eq!(hle_acb_acquire_mem(&ctx, &[cb, 0, base | 1, 0x100, 0]), 0);
    }

    #[test]
    fn acb_wait_reg_mem_emits_32_and_64_bit_forms() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        let addr = 0x0000_00AB_CDEF_0120u64;
        let mask = 0x1111_2222_3333_4444u64;
        let reference = 0x5555_6666_7777_8888u64;
        // 32-bit form (size = 0): 6 DWORDs, R_WAIT_MEM32.
        let args32 = [cb, 0, 3, 1, addr, reference, mask, 400];
        assert_eq!(hle_acb_wait_reg_mem(&ctx, &args32), 0x400);
        assert_eq!(read_u32(&ctx, 0x400), pm4(6, IT_NOP, R_WAIT_MEM32));
        assert_eq!(read_u32(&ctx, 0x404), addr as u32);
        assert_eq!(read_u32(&ctx, 0x408), (addr >> 32) as u32);
        assert_eq!(read_u32(&ctx, 0x40C), mask as u32);
        assert_eq!(read_u32(&ctx, 0x410), 3, "compare function");
        assert_eq!(read_u32(&ctx, 0x414), reference as u32);
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x400 + 24);
        // 64-bit form (size = 1): 9 DWORDs, R_WAIT_MEM64.
        let args64 = [cb, 1, 5, 0, addr, reference, mask, 400];
        assert_eq!(hle_acb_wait_reg_mem(&ctx, &args64), 0x418);
        assert_eq!(read_u32(&ctx, 0x418), pm4(9, IT_NOP, R_WAIT_MEM64));
        assert_eq!(read_u32(&ctx, 0x41C), addr as u32);
        assert_eq!(read_u32(&ctx, 0x420), (addr >> 32) as u32);
        assert_eq!(read_u32(&ctx, 0x424), mask as u32);
        assert_eq!(read_u32(&ctx, 0x428), (mask >> 32) as u32);
        assert_eq!(read_u32(&ctx, 0x42C), reference as u32);
        assert_eq!(read_u32(&ctx, 0x430), (reference >> 32) as u32);
        assert_eq!(read_u32(&ctx, 0x434), 5, "compare function");
        assert_eq!(read_u32(&ctx, 0x438), 10, "pollCycles / 40");
        // Invalid size / compare / cache policy rejected.
        assert_eq!(hle_acb_wait_reg_mem(&ctx, &[cb, 2, 0, 0, 0, 0, 0, 0]), 0);
        assert_eq!(hle_acb_wait_reg_mem(&ctx, &[cb, 0, 8, 0, 0, 0, 0, 0]), 0);
        assert_eq!(hle_acb_wait_reg_mem(&ctx, &[cb, 0, 0, 4, 0, 0, 0, 0]), 0);
    }

    #[test]
    fn cb_set_sh_registers_direct_sorts_and_coalesces_runs() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // Unsorted (offset, value) pairs: run {0x0C,0x0D,0x0E} + isolated 0x240.
        let regs = 0x1000u64;
        for (i, (off, val)) in [(0x0Eu32, 30u32), (0x240, 99), (0x0C, 10), (0x0D, 20)]
            .iter()
            .enumerate()
        {
            assert!(ctx.mem.write(regs + i as u64 * 8, &off.to_le_bytes()));
            assert!(ctx.mem.write(regs + i as u64 * 8 + 4, &val.to_le_bytes()));
        }
        let first = hle_cb_set_sh_registers_direct(&ctx, &[cb, regs, 4]);
        assert_eq!(first, 0x400, "returns the FIRST packet address");
        // Packet 1: 5 DWORDs covering the 0x0C..0x0E run.
        assert_eq!(read_u32(&ctx, 0x400), pm4(5, IT_SET_SH_REG, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 0x0C, "run start offset");
        assert_eq!(read_u32(&ctx, 0x408), 10);
        assert_eq!(read_u32(&ctx, 0x40C), 20);
        assert_eq!(read_u32(&ctx, 0x410), 30);
        // Packet 2: 3 DWORDs for the isolated register.
        assert_eq!(read_u32(&ctx, 0x414), pm4(3, IT_SET_SH_REG, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x418), 0x240);
        assert_eq!(read_u32(&ctx, 0x41C), 99);
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x420);
        // Zero count / null pointers / oversized count rejected.
        assert_eq!(hle_cb_set_sh_registers_direct(&ctx, &[cb, regs, 0]), 0);
        assert_eq!(hle_cb_set_sh_registers_direct(&ctx, &[0, regs, 1]), 0);
        assert_eq!(hle_cb_set_sh_registers_direct(&ctx, &[cb, 0, 1]), 0);
        assert_eq!(hle_cb_set_sh_registers_direct(&ctx, &[cb, regs, 4097]), 0);
    }

    #[test]
    fn cb_get_size_functions_match_their_writers() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // Dispatch: the writer emits 5 DWORDs; GetSize reports 20 bytes.
        let before = read_u64(&ctx, cb + CB_CURSOR_UP);
        assert_ne!(hle_cb_dispatch(&ctx, &[cb, 1, 1, 1, 0]), 0);
        let dispatch_bytes = read_u64(&ctx, cb + CB_CURSOR_UP) - before;
        assert_eq!(hle_cb_dispatch_get_size(&ctx, &[]), dispatch_bytes);
        assert_eq!(hle_cb_dispatch_get_size(&ctx, &[]), 20);
        // Nop: the writer emits exactly dwordCount DWORDs.
        let before = read_u64(&ctx, cb + CB_CURSOR_UP);
        assert_ne!(hle_cb_nop(&ctx, &[cb, 6]), 0);
        let nop_bytes = read_u64(&ctx, cb + CB_CURSOR_UP) - before;
        assert_eq!(hle_cb_nop_get_size(&ctx, &[6]), nop_bytes);
        assert_eq!(hle_cb_nop_get_size(&ctx, &[6]), 24);
    }

    #[test]
    fn dcb_draw_index_indirect_emits_five_dword_packet() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        assert_eq!(
            hle_dcb_draw_index_indirect(&ctx, &[cb, 0x120, 0xABCD]),
            0x400
        );
        assert_eq!(
            read_u32(&ctx, 0x400),
            pm4(5, IT_DRAW_INDEX_INDIRECT, R_ZERO)
        );
        assert_eq!(read_u32(&ctx, 0x404), 0x120, "indirect-args data offset");
        assert_eq!(read_u32(&ctx, 0x408), 0);
        assert_eq!(read_u32(&ctx, 0x40C), 0);
        assert_eq!(read_u32(&ctx, 0x410), 0xABCD, "modifier");
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x400 + 20);
        assert_eq!(hle_dcb_draw_index_indirect(&ctx, &[0]), 0);
    }

    #[test]
    fn stall_parser_and_lod_stats_emit_correctly() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb = 0x40;
        setup_cb(&ctx, cb, 0x400, 0x800);
        // Stall: 2-DWORD NOP keeps the cursor coherent; size > 1 rejected.
        assert_eq!(
            hle_dcb_stall_command_buffer_parser(&ctx, &[cb, 0, 0x9000, 7]),
            0x400
        );
        assert_eq!(read_u32(&ctx, 0x400), pm4(2, IT_NOP, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x404), 0);
        assert_eq!(hle_dcb_stall_command_buffer_parser(&ctx, &[cb, 2]), 0);
        // GetLodStats: 5-DWORD packet with the packed control word. enable and
        // counterSelect are stack args → args[6]/args[7].
        let dst = 0x0000_00AB_1234_5678u64;
        let args = [cb, 2, dst, 0xFEED, 0x3C, 1, 1, 0x12];
        assert_eq!(hle_dcb_get_lod_stats(&ctx, &args), 0x408);
        assert_eq!(read_u32(&ctx, 0x408), pm4(5, IT_GET_LOD_STATS, R_ZERO));
        assert_eq!(read_u32(&ctx, 0x40C), 0xFEED, "control");
        assert_eq!(
            read_u32(&ctx, 0x410),
            (dst as u32) & !0x3F,
            "dst lo & ~0x3F"
        );
        assert_eq!(read_u32(&ctx, 0x414), (dst >> 32) as u32, "dst hi");
        assert_eq!(
            read_u32(&ctx, 0x418),
            (2 << 28) | (1 << 19) | (1 << 18) | (0x3C << 10) | (0x12 << 2),
            "cachePolicy<<28 | enable<<19 | reset<<18 | mask<<10 | select<<2"
        );
        assert_eq!(read_u64(&ctx, cb + CB_CURSOR_UP), 0x408 + 20);
    }
}
