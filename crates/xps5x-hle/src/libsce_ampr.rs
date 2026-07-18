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

// Per-command record sizes reported by the `MeasureCommandSize*` calls.
const READ_FILE_RECORD_SIZE: u64 = 0x30;
const KERNEL_EVENT_QUEUE_RECORD_SIZE: u64 = 0x30;
const WRITE_ADDRESS_RECORD_SIZE: u64 = 0x20;

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
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferClearBuffer",
        hle_clear_buffer,
    );
    registry.register("libSceAmpr", "sceAmprCommandBufferGetSize", hle_get_size);
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferGetCurrentOffset",
        hle_get_current_offset,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprAprCommandBufferReadFile",
        hle_apr_read_file,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferWriteKernelEventQueue_04_00",
        hle_write_equeue_record,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferWriteAddressOnCompletion",
        hle_write_address_record,
    );
    // MeasureCommandSize* report a command record's byte size.
    registry.register("libSceAmpr", "sceAmprMeasureCommandSizeReadFile", |_, _| {
        READ_FILE_RECORD_SIZE
    });
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWriteKernelEventQueue_04_00",
        |_, _| KERNEL_EVENT_QUEUE_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWriteAddressOnCompletion",
        |_, _| WRITE_ADDRESS_RECORD_SIZE,
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

/// `sceAmprCommandBufferClearBuffer(cb)`: zero the visible buffer/size
/// pointers in the struct and return the previously-bound buffer pointer.
fn hle_clear_buffer(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut data = [0u8; 8];
    if !ctx.mem.read(cb + CB_DATA_OFFSET, &mut data) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let old_buffer = u64::from_le_bytes(data);
    if !ctx.mem.write(cb + CB_DATA_OFFSET, &0u64.to_le_bytes())
        || !ctx.mem.write(cb + CB_SIZE_OFFSET, &0u64.to_le_bytes())
    {
        return SCE_ERROR_MEMORY_FAULT;
    }
    old_buffer
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

/// Append one command record to a command buffer's visible buffer at the
/// host-tracked cursor, advancing the cursor (SharpEmu's
/// `AppendCommandBufferRecord`).
fn append_record(ctx: &HleContext, cb: u64, record: &[u8]) -> bool {
    let mut data = [0u8; 8];
    let mut size = [0u8; 8];
    if !ctx.mem.read(cb + CB_DATA_OFFSET, &mut data) || !ctx.mem.read(cb + CB_SIZE_OFFSET, &mut size)
    {
        return false;
    }
    let (buffer, buf_size) = (u64::from_le_bytes(data), u64::from_le_bytes(size));
    let offset = ctx
        .kernel
        .ampr_write_offsets
        .get(&cb)
        .map(|o| *o)
        .unwrap_or(0);
    let record_len = record.len() as u64;
    if buffer == 0 || offset > buf_size || record_len > buf_size - offset {
        return false;
    }
    if !ctx.mem.write(buffer + offset, record) {
        return false;
    }
    ctx.kernel.ampr_write_offsets.insert(cb, offset + record_len);
    true
}

/// `sceAmprCommandBufferWriteKernelEventQueue_04_00(cb, equeue, ident, userData)`:
/// append a completion-event record (type 2, 0x30 bytes) that the kernel
/// fires when the buffer completes. SharpEmu `AppendKernelEventQueueRecord`.
fn hle_write_equeue_record(ctx: &HleContext, args: &[u64]) -> u64 {
    const AMPR_FILTER: i16 = -0x64; // KernelEventFilterAmpr (SharpEmu)
    let cb = args.first().copied().unwrap_or(0);
    let equeue = args.get(1).copied().unwrap_or(0);
    let ident = args.get(2).copied().unwrap_or(0);
    let user_data = args.get(3).copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut record = [0u8; 0x30];
    record[0x00..0x04].copy_from_slice(&2u32.to_le_bytes());
    record[0x04..0x06].copy_from_slice(&AMPR_FILTER.to_le_bytes());
    record[0x08..0x10].copy_from_slice(&equeue.to_le_bytes());
    record[0x10..0x18].copy_from_slice(&ident.to_le_bytes());
    record[0x18..0x20].copy_from_slice(&user_data.to_le_bytes());
    record[0x20..0x28].copy_from_slice(&user_data.to_le_bytes());
    if !append_record(ctx, cb, &record) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferWriteAddressOnCompletion(cb, address, value)`:
/// append a write-address record (type 3, 0x20 bytes) the kernel performs at
/// completion. SharpEmu `AppendWriteAddressRecord`.
fn hle_write_address_record(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let address = args.get(1).copied().unwrap_or(0);
    let value = args.get(2).copied().unwrap_or(0);
    if cb == 0 || address == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut record = [0u8; 0x20];
    record[0x00..0x04].copy_from_slice(&3u32.to_le_bytes());
    record[0x08..0x10].copy_from_slice(&address.to_le_bytes());
    record[0x10..0x18].copy_from_slice(&value.to_le_bytes());
    if !append_record(ctx, cb, &record) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprAprCommandBufferReadFile(cb, _, _, fileId, destination, size,
/// fileOffset)`: read `size` bytes of PAK file `fileId` at `fileOffset` into
/// guest `destination`. `fileOffset` is SysV arg7 (`args[6]`, on the stack at
/// `[Rsp+8]`, captured by the runtime dispatch). Mirrors SharpEmu's
/// `AprCommandBufferReadFile` for the **unregistered/missing-file** case, which
/// is the only case in-tree: XPS5X has no populated Ampr file registry, so every
/// `fileId` is missing and the guest region is zero-filled (games queue
/// speculative reads and only consume bytes on success paths — zero-fill, not
/// failure, is the documented behavior). Real file-backed reads (host-path
/// registry + PAK sequential-offset tracking) land with the I/O backend; a
/// registered read would append a command record, which is deferred with it.
fn hle_apr_read_file(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let file_id = args.get(3).copied().unwrap_or(0) as u32;
    let destination = args.get(4).copied().unwrap_or(0);
    let size = args.get(5).copied().unwrap_or(0);
    let file_offset = args.get(6).copied().unwrap_or(0);
    if cb == 0 || (destination == 0 && size != 0) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    // A registered id appends a ReadFile record the kernel completes at
    // submit (SharpEmu). The id was registered by sceKernelAprResolve*.
    if ctx.kernel.appr_host_path(file_id).is_some() {
        let mut record = [0u8; 0x30];
        record[0x00..0x04].copy_from_slice(&1u32.to_le_bytes());
        record[0x04..0x08].copy_from_slice(&file_id.to_le_bytes());
        record[0x08..0x10].copy_from_slice(&destination.to_le_bytes());
        record[0x10..0x18].copy_from_slice(&size.to_le_bytes());
        record[0x18..0x20].copy_from_slice(&file_offset.to_le_bytes());
        if !append_record(ctx, cb, &record) {
            return SCE_ERROR_MEMORY_FAULT;
        }
        return OK;
    }
    // Missing-file path: zero-fill in chunks, stopping if a write faults (a
    // partial fill still returns OK, matching the reference).
    if destination != 0 && size > 0 {
        let zeros = [0u8; 4096];
        let mut written = 0u64;
        while written < size {
            let chunk = (size - written).min(zeros.len() as u64) as usize;
            if !ctx.mem.write(destination + written, &zeros[..chunk]) {
                break;
            }
            written += chunk as u64;
        }
    }
    OK
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
        // ClearBuffer returns the bound buffer and zeroes the visible pointers.
        assert_eq!(hle_clear_buffer(&ctx, &[cb]), 0x2000);
        assert_eq!(read_u64(&ctx, cb + CB_DATA_OFFSET), 0);
        assert_eq!(read_u64(&ctx, cb + CB_SIZE_OFFSET), 0);
        // Destruct drops the cursor state; NULL cb constructor is a no-op.
        assert_eq!(hle_dtor(&ctx, &[cb]), 0);
        assert!(!kernel.ampr_write_offsets.contains_key(&cb));
        assert_eq!(hle_ctor(&ctx, &[0, 0, 0]), 0);
        // Getters reject a NULL command buffer.
        assert_eq!(hle_get_size(&ctx, &[0]), SCE_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn measure_command_sizes_report_record_bytes() {
        let reg = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            reg.call(&ctx, "libSceAmpr", "sceAmprMeasureCommandSizeReadFile", &[]),
            Some(READ_FILE_RECORD_SIZE)
        );
        assert_eq!(
            reg.call(
                &ctx,
                "libSceAmpr",
                "sceAmprMeasureCommandSizeWriteAddressOnCompletion",
                &[]
            ),
            Some(WRITE_ADDRESS_RECORD_SIZE)
        );
    }

    #[test]
    fn apr_read_file_zero_fills_missing_files() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x100;
        let dst: u64 = 0x200;
        // Pre-dirty the destination so a successful zero-fill is observable.
        assert!(ctx.mem.write(dst, &0xDEAD_BEEF_CAFE_F00Du64.to_le_bytes()));
        // args: cb, _, _, fileId=7, destination, size=16, fileOffset=0 (args[6]).
        assert_eq!(hle_apr_read_file(&ctx, &[cb, 0, 0, 7, dst, 16, 0]), OK);
        assert_eq!(read_u64(&ctx, dst), 0, "first 8 bytes zeroed");
        assert_eq!(read_u64(&ctx, dst + 8), 0, "next 8 bytes zeroed");
        // NULL command buffer and (dst==0, size!=0) are argument errors.
        assert_eq!(
            hle_apr_read_file(&ctx, &[0, 0, 0, 7, dst, 16, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_apr_read_file(&ctx, &[cb, 0, 0, 7, 0, 16, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // A zero-size read (dst==0 allowed) is a benign OK.
        assert_eq!(hle_apr_read_file(&ctx, &[cb, 0, 0, 7, 0, 0, 0]), OK);
    }
}
