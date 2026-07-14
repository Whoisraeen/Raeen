//! HLE libSceAmpr — the AMPR (async processor) command-buffer lifecycle.
//!
//! A faithful Rust port of the command-buffer object management from SharpEmu's
//! `AmprExports` (GPL-2.0). An AMPR command buffer is a small guest struct
//! (self ptr @0x00, data ptr @0x08, size @0x10, aux @0x18/0x20) plus a
//! host-tracked write cursor (`OrbisKernel::ampr_write_offsets`, keyed by the
//! command-buffer address). This ports the construct/destruct/get/set/reset
//! lifecycle — enough for a title to build AMPR command buffers. The actual
//! command *content* writers (`WriteKernelEventQueue`/`WriteAddressOnCompletion`/
//! `ReadFile`) and real submission land with the compute backend (M2-adjacent).

use crate::{HleContext, HleRegistry};
use tracing::debug;

const OK: u64 = 0;
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

// AmprCommandBuffer struct field offsets (bytes).
const CB_SELF_OFFSET: u64 = 0x00;
const CB_DATA_OFFSET: u64 = 0x08;
const CB_SIZE_OFFSET: u64 = 0x10;
const CB_AUX0_OFFSET: u64 = 0x18;
const CB_AUX1_OFFSET: u64 = 0x20;

/// Register the libSceAmpr command-buffer lifecycle functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAmpr", "sceAmprCommandBufferConstructor", hle_ctor);
    registry.register("libSceAmpr", "sceAmprAprCommandBufferConstructor", hle_ctor);
    registry.register("libSceAmpr", "sceAmprCommandBufferDestructor", hle_dtor);
    registry.register("libSceAmpr", "sceAmprAprCommandBufferDestructor", hle_dtor);
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferSetBuffer",
        hle_set_buffer,
    );
    registry.register("libSceAmpr", "sceAmprCommandBufferReset", hle_reset);
    registry.register("libSceAmpr", "sceAmprCommandBufferGetSize", hle_get_size);
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferGetCurrentOffset",
        hle_get_current_offset,
    );
}

/// Write the command-buffer struct fields (self/data/size/aux) and set the
/// write cursor to `write_offset`.
fn write_cb(ctx: &HleContext, cb: u64, buffer: u64, size: u64, write_offset: u64) -> bool {
    let ok = ctx.mem.write(cb + CB_SELF_OFFSET, &cb.to_le_bytes())
        && ctx.mem.write(cb + CB_DATA_OFFSET, &buffer.to_le_bytes())
        && ctx.mem.write(cb + CB_SIZE_OFFSET, &size.to_le_bytes())
        && ctx.mem.write(cb + CB_AUX0_OFFSET, &0u64.to_le_bytes())
        && ctx.mem.write(cb + CB_AUX1_OFFSET, &0u64.to_le_bytes());
    if ok {
        ctx.kernel.ampr_write_offsets.insert(cb, write_offset);
    }
    ok
}

/// `sceAmprCommandBufferConstructor(cb, buffer, size)`: initialize the command
/// buffer over `[buffer, buffer+size)` with the cursor at 0. A NULL `cb` is a
/// benign no-op (returns 0). Returns the command-buffer pointer on success.
fn hle_ctor(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let buffer = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    if !write_cb(ctx, cb, buffer, size, 0) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    debug!("sceAmprCommandBufferConstructor(cb={cb:#x}, buffer={buffer:#x}, size={size:#x})");
    cb
}

/// `sceAmprCommandBufferDestructor(cb)`: drop the tracked write cursor.
fn hle_dtor(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb != 0 {
        ctx.kernel.ampr_write_offsets.remove(&cb);
    }
    0
}

/// `sceAmprCommandBufferSetBuffer(cb, buffer, size)`: rebind the buffer.
fn hle_set_buffer(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let buffer = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !ctx.mem.write(cb + CB_DATA_OFFSET, &buffer.to_le_bytes())
        || !ctx.mem.write(cb + CB_SIZE_OFFSET, &size.to_le_bytes())
    {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferReset(cb)`: rewind the cursor to 0 (keeping the buffer).
fn hle_reset(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let (mut data, mut size) = ([0u8; 8], [0u8; 8]);
    if !ctx.mem.read(cb + CB_DATA_OFFSET, &mut data)
        || !ctx.mem.read(cb + CB_SIZE_OFFSET, &mut size)
    {
        return SCE_ERROR_MEMORY_FAULT;
    }
    if !write_cb(
        ctx,
        cb,
        u64::from_le_bytes(data),
        u64::from_le_bytes(size),
        0,
    ) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferGetSize(cb)`: return the buffer size (in `rax`).
fn hle_get_size(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut buf = [0u8; 8];
    if !ctx.mem.read(cb + CB_SIZE_OFFSET, &mut buf) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    u64::from_le_bytes(buf)
}

/// `sceAmprCommandBufferGetCurrentOffset(cb)`: return the write cursor.
fn hle_get_current_offset(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    ctx.kernel
        .ampr_write_offsets
        .get(&cb)
        .map(|o| *o)
        .unwrap_or(0)
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
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    fn read_u64(ctx: &HleContext, addr: u64) -> u64 {
        let mut b = [0u8; 8];
        assert!(ctx.mem.read(addr, &mut b));
        u64::from_le_bytes(b)
    }

    #[test]
    fn construct_get_set_reset_lifecycle() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x100;
        // Construct binds buffer/size + zeroes the cursor, returns cb.
        assert_eq!(hle_ctor(&ctx, &[cb, 0x1000, 0x800]), cb);
        assert_eq!(read_u64(&ctx, cb + CB_SELF_OFFSET), cb);
        assert_eq!(read_u64(&ctx, cb + CB_DATA_OFFSET), 0x1000);
        assert_eq!(hle_get_size(&ctx, &[cb]), 0x800);
        assert_eq!(hle_get_current_offset(&ctx, &[cb]), 0);
        // Advance the cursor (as a writer would), then Reset rewinds it.
        kernel.ampr_write_offsets.insert(cb, 0x40);
        assert_eq!(hle_get_current_offset(&ctx, &[cb]), 0x40);
        assert_eq!(hle_reset(&ctx, &[cb]), OK);
        assert_eq!(hle_get_current_offset(&ctx, &[cb]), 0);
        // SetBuffer rebinds.
        assert_eq!(hle_set_buffer(&ctx, &[cb, 0x2000, 0x400]), OK);
        assert_eq!(hle_get_size(&ctx, &[cb]), 0x400);
        assert_eq!(read_u64(&ctx, cb + CB_DATA_OFFSET), 0x2000);
        // Destruct drops the cursor state; NULL cb constructor is a no-op.
        assert_eq!(hle_dtor(&ctx, &[cb]), 0);
        assert!(!kernel.ampr_write_offsets.contains_key(&cb));
        assert_eq!(hle_ctor(&ctx, &[0, 0, 0]), 0);
        // Getters reject a NULL command buffer.
        assert_eq!(hle_get_size(&ctx, &[0]), SCE_ERROR_INVALID_ARGUMENT);
    }
}
