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
use tracing::debug;

// DrawCommandBuffer struct field offsets (bytes).
const CB_CURSOR_UP: u64 = 0x10; // u64 — current write pointer (advances up)
const CB_CURSOR_DOWN: u64 = 0x18; // u64 — end/limit
const CB_RESERVED_DW: u64 = 0x30; // u32 — reserved tail dwords

// Agc PM4 IT (instruction type) opcodes + register sub-discriminators.
const IT_INDEX_TYPE: u32 = 0x2A;
const IT_NOP: u32 = 0x10;
const IT_INDEX_BUFFER_SIZE: u32 = 0x13;
const IT_INDEX_BASE: u32 = 0x26;
const IT_DRAW_INDEX_2: u32 = 0x27;
const IT_DRAW_INDEX_OFFSET_2: u32 = 0x35;
const IT_NUM_INSTANCES: u32 = 0x2F;
const IT_DISPATCH_DIRECT: u32 = 0x15;
const IT_DISPATCH_INDIRECT: u32 = 0x16;
const IT_SET_BASE: u32 = 0x11;
const IT_EVENT_WRITE: u32 = 0x46;
const IT_WAIT_REG_MEM: u32 = 0x3C;
const IT_SET_SH_REG: u32 = 0x76;
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
    registry.register(
        "libSceAgc",
        "sceAgcDriverGetResourceRegistrationMaxNameLength",
        hle_get_resource_max_name_length,
    );
    registry.register("libSceAgc", "sceAgcSuspendPoint", hle_suspend_point);
    // A Gen5 driver call whose only observable effect is a trace → benign OK.
    registry.register(
        "libSceAgc",
        "sceAgcDriverUnknown_KRzWekV120",
        hle_suspend_point,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDriverRegisterDefaultOwner",
        hle_driver_register_default_owner,
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
        "sceAgcDriverGetDefaultOwner",
        hle_driver_get_default_owner,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDriverAddEqEvent",
        hle_driver_add_eq_event,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDriverDeleteEqEvent",
        hle_driver_delete_eq_event,
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
    registry.register("libSceAgc", "sceAgcDriverSubmitDcb", hle_driver_submit_dcb);
    registry.register("libSceAgc", "sceAgcDriverSubmitAcb", hle_driver_submit_acb);
    registry.register(
        "libSceAgc",
        "sceAgcDriverSubmitMultiDcbs",
        hle_driver_submit_multi_dcbs,
    );
    registry.register(
        "libSceAgc",
        "sceAgcDriverQueryResourceRegistrationUserMemoryRequirements",
        hle_driver_query_resource_memory,
    );
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
    registry.register(
        "libSceAgc",
        "sceAgcDriverInitResourceRegistration",
        hle_driver_init_resource_registration,
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
            triggered: false,
            fflags: 0,
        },
    );
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
/// capture and structurally decode the submitted Gen5 PM4 command buffer, then
/// succeed. (No real GPU submission yet — that arrives with the Vulkan backend;
/// SharpEmu likewise only validates here.)
fn hle_driver_submit_dcb(ctx: &HleContext, args: &[u64]) -> u64 {
    submit_validate(ctx, args.first().copied().unwrap_or(0))
}

fn hle_driver_submit_acb(ctx: &HleContext, args: &[u64]) -> u64 {
    submit_validate(ctx, args.get(1).copied().unwrap_or(0))
}

/// `sceAgcDriverSubmitMultiDcbs(addressArray, sizeArray, bufferCount)`:
/// validate each command buffer in the arrays and succeed (no real
/// submission yet — the Vulkan gate).
fn hle_driver_submit_multi_dcbs(ctx: &HleContext, args: &[u64]) -> u64 {
    let address_array = args.first().copied().unwrap_or(0);
    let size_array = args.get(1).copied().unwrap_or(0);
    let buffer_count = args.get(2).copied().unwrap_or(0);
    if address_array == 0 || size_array == 0 || buffer_count == 0 || buffer_count > 4096 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    for i in 0..buffer_count {
        let cmd = read_u64_or_zero(ctx, address_array + i * 8);
        let dwords = read_u64_or_zero(ctx, size_array + i * 8);
        if cmd == 0 || dwords == 0 {
            return SCE_ERROR_INVALID_ARGUMENT;
        }
    }
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

fn submit_validate(ctx: &HleContext, packet: u64) -> u64 {
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

    if prior_submissions == 0
        || (prior_draws == 0 && decoded.draw_packets != 0)
        || (prior_flips == 0 && !decoded.flips.is_empty())
    {
        tracing::info!(
            command_address,
            dword_count,
            packets = decoded.packets.len(),
            draws = decoded.draw_packets,
            dispatches = decoded.dispatch_packets,
            flips = decoded.flips.len(),
            packet_layout = ?decoded.packets,
            flip_layout = ?decoded.flips,
            "captured AGC DCB submission"
        );
    }
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
        _ => return SCE_ERROR_INVALID_ARGUMENT,
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
        for off in [0x08u64, 0x10, 0x18, 0x20] {
            if !relocate_pointer_field(ctx, user_data + off) {
                return SCE_ERROR_MEMORY_FAULT;
            }
        }
    }
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
    matches!(version, 7 | 8 | 10 | 13)
}

/// `sceAgcInit(state, version)`: initialize the Agc state for a supported
/// register-defaults version.
fn hle_init(_ctx: &HleContext, args: &[u64]) -> u64 {
    let state = args.first().copied().unwrap_or(0);
    let version = args.get(1).copied().unwrap_or(0) as u32;
    if state == 0 || !is_supported_version(version) {
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
fn hle_dcb_write_data(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let destination = (args.get(1).copied().unwrap_or(0) & 0xFF) as u32;
    let cache_policy = (args.get(2).copied().unwrap_or(0) & 0xFF) as u32;
    let destination_address = args.get(3).copied().unwrap_or(0);
    let data_address = args.get(4).copied().unwrap_or(0);
    let dword_count = args.get(5).copied().unwrap_or(0) as u32;
    let increment = (args.get(6).copied().unwrap_or(0) & 0xFF) as u32;
    let write_confirm = (args.get(7).copied().unwrap_or(0) & 0xFF) as u32;
    if cb == 0 || destination_address == 0 || data_address == 0 || dword_count > 0x3FFD {
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
        let mut buf = [0u8; 4];
        if !ctx.mem.read(data_address + index * 4, &mut buf)
            || !ctx.mem.write(addr + 16 + index * 4, &buf)
        {
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
        assert!(ctx.mem.write(0x180, &0x900u64.to_le_bytes()));
        assert!(ctx.mem.write(0x188, &2u32.to_le_bytes()));
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
            2
        );
        assert_eq!(
            hle_driver_submit_dcb(&ctx, &[0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // SubmitMultiDcbs: two buffers with valid addresses + sizes.
        assert!(ctx.mem.write(0x1A0, &0x400u64.to_le_bytes()));
        assert!(ctx.mem.write(0x1A8, &0x500u64.to_le_bytes()));
        assert!(ctx.mem.write(0x1C0, &8u64.to_le_bytes()));
        assert!(ctx.mem.write(0x1C8, &4u64.to_le_bytes()));
        assert_eq!(hle_driver_submit_multi_dcbs(&ctx, &[0x1A0, 0x1C0, 2]), 0);
        assert_eq!(
            hle_driver_submit_multi_dcbs(&ctx, &[0, 0x1C0, 2]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // Query: required = resourceCount*0x118 + ownerCount*0x1E0.
        assert_eq!(hle_driver_query_resource_memory(&ctx, &[0x1E0, 2, 3]), 0);
        assert_eq!(read_u64(&ctx, 0x1E0), 2 * 0x118 + 3 * 0x1E0);
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
        // A bad magic is rejected.
        assert!(ctx.mem.write(header, &0xDEADu32.to_le_bytes()));
        assert_eq!(
            hle_create_shader(&ctx, &[dest, header, code]),
            SCE_ERROR_INVALID_ARGUMENT
        );
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
}
