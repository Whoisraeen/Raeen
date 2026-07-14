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
const IT_NUM_INSTANCES: u32 = 0x2F;
const IT_DISPATCH_DIRECT: u32 = 0x15;
const IT_DISPATCH_INDIRECT: u32 = 0x16;
const IT_EVENT_WRITE: u32 = 0x46;
const IT_SET_SH_REG: u32 = 0x76;
/// Marker DWORD preceding a `CbSetShRegisterRange` packet.
const SET_SH_RANGE_MARKER: u32 = 0x6875_000D;
const R_ZERO: u32 = 0x00;
const R_DRAW_INDEX_AUTO: u32 = 0x04;
const R_DRAW_RESET: u32 = 0x05;
const R_WAIT_FLIP_DONE: u32 = 0x06;
const R_ACB_RESET: u32 = 0x09;
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
    registry.register("libSceAgc", "sceAgcDcbSetFlip", hle_dcb_set_flip);
    registry.register("libSceAgc", "sceAgcAcbResetQueue", hle_acb_reset_queue);
    registry.register(
        "libSceAgc",
        "sceAgcCbSetShRegisterRangeDirect",
        hle_cb_set_sh_register_range,
    );
    registry.register("libSceAgc", "sceAgcInit", hle_init);
    registry.register("libSceAgc", "sceAgcDcbEventWrite", hle_dcb_event_write);
    registry.register("libSceAgc", "sceAgcAcbWriteData", hle_acb_write_data);
    // AcbDispatchIndirect emits the identical indirect-dispatch packet.
    registry.register("libSceAgc", "sceAgcAcbDispatchIndirect", hle_acb_write_data);
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
    registry.register(
        "libSceAgc",
        "sceAgcGetDataPacketPayloadAddress",
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

/// `sceAgcAcbWriteData` / `sceAgcAcbDispatchIndirect(acb, argumentsAddress,
/// modifier)`: emit an indirect dispatch packet (4 DWORDs) with the arguments
/// address + initiator. (Both NIDs emit the identical packet.)
fn hle_acb_write_data(ctx: &HleContext, args: &[u64]) -> u64 {
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

        // AcbWriteData: 4-dword indirect dispatch with a split args address.
        let a = hle_acb_write_data(&ctx, &[cb, 0x00AB_1234_5678_0000, 0]);
        assert_eq!(read_u32(&ctx, a), pm4(4, IT_DISPATCH_INDIRECT, R_ZERO));
        assert_eq!(read_u32(&ctx, a + 4), 0x5678_0000, "args low");
        assert_eq!(read_u32(&ctx, a + 8), 0x00AB_1234, "args high");
        assert_eq!(read_u32(&ctx, a + 12), 0x41, "initiator");
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
