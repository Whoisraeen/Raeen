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
const R_ZERO: u32 = 0x00;
const R_DRAW_INDEX_AUTO: u32 = 0x04;
const R_DRAW_RESET: u32 = 0x05;
const R_WAIT_FLIP_DONE: u32 = 0x06;

/// The `DRAW_INDEX_AUTO` modifier a valid call must pass.
const DRAW_AUTO_MODIFIER: u64 = 0x4000_0000;

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
