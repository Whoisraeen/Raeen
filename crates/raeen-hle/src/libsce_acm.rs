//! HLE libSceAcm — the PS5 Audio Codec Module (hardware-assisted audio
//! processing contexts and batches).
//!
//! The ABI and lifecycle are independently reimplemented from the public
//! KytyPS5 `audio.h` / `audio.cpp` declarations (Kyty/MIT lineage), with the
//! context out-pointer behavior cross-checked against SharpEmu's GPL-2.0
//! `AcmExports.cs`. Context and batch creation now initialize the guest's
//! out-parameters, batch errors are cleared, and command builders advance the
//! caller's batch cursor. The DSP itself is still not emulated: batches complete
//! synchronously, while FMOD's decoded PCM continues through libSceAudioOut.
//!
//! This distinction matters for Minecraft: returning success without writing
//! the context handle left its audio engine consuming an uninitialized value,
//! so it never reached a valid output lifecycle even though the imports linked.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicU32, Ordering};

const OK: u64 = 0;
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;
const BATCH_INFO_SIZE: usize = 24;
const BATCH_ERROR_SIZE: usize = 32;

static NEXT_CONTEXT: AtomicU32 = AtomicU32::new(0);
static NEXT_BATCH: AtomicU32 = AtomicU32::new(0);

/// Register the libSceAcm functions implemented by the public references.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAcm", "sceAcmContextCreate", hle_context_create);
    registry.register("libSceAcm", "sceAcmContextDestroy", |_, _| OK);
    registry.register(
        "libSceAcm",
        "sceAcmBatchStartBuffer",
        hle_batch_start_buffer,
    );
    registry.register(
        "libSceAcm",
        "sceAcmBatchStartBuffers",
        hle_batch_start_buffers,
    );
    registry.register("libSceAcm", "sceAcmBatchWait", |_, _| OK);
    registry.register(
        "libSceAcm",
        "sceAcm_ConvReverb_SharedInput",
        hle_conv_reverb_shared_input,
    );
}

/// `sceAcmContextCreate(AcmContextId *context)`: allocate a non-zero 32-bit
/// context id and publish it to guest memory. KytyPS5 declares `AcmContextId`
/// as `uint32_t`; writing exactly four bytes also preserves a caller's adjacent
/// stack canary.
fn hle_context_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let out_context = args.first().copied().unwrap_or(0);
    if out_context == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let handle = NEXT_CONTEXT.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    if ctx.mem.write(out_context, &handle.to_le_bytes()) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// Common completion for `sceAcmBatchStartBuffer(s)`: clear the optional
/// eight-word error result and publish a fresh 32-bit batch id. DSP work is
/// currently synchronous/no-op, so `sceAcmBatchWait` can return immediately.
fn complete_batch(ctx: &HleContext, batch_error: u64, out_batch: u64) -> u64 {
    if out_batch == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if batch_error != 0 && !ctx.mem.write(batch_error, &[0; BATCH_ERROR_SIZE]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let batch = NEXT_BATCH.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    if ctx.mem.write(out_batch, &batch.to_le_bytes()) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceAcmBatchStartBuffer(context, commands, size, error, outBatch)`.
fn hle_batch_start_buffer(ctx: &HleContext, args: &[u64]) -> u64 {
    let batch_error = args.get(3).copied().unwrap_or(0);
    let out_batch = args.get(4).copied().unwrap_or(0);
    complete_batch(ctx, batch_error, out_batch)
}

/// `sceAcmBatchStartBuffers(context, count, infos, error, outBatch)`.
fn hle_batch_start_buffers(ctx: &HleContext, args: &[u64]) -> u64 {
    let count = args.get(1).copied().unwrap_or(0);
    let infos = args.get(2).copied().unwrap_or(0);
    if count != 0 && infos == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let batch_error = args.get(3).copied().unwrap_or(0);
    let out_batch = args.get(4).copied().unwrap_or(0);
    complete_batch(ctx, batch_error, out_batch)
}

/// Advance an `AcmBatchInfo { buffer, offset, buffer_size }` cursor by a
/// command's encoded byte size, clamped to the caller-provided buffer.
fn advance_batch(ctx: &HleContext, info: u64, bytes: u64) -> u64 {
    if info == 0 {
        return OK;
    }
    let mut raw = [0u8; BATCH_INFO_SIZE];
    if !ctx.mem.read(info, &mut raw) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let buffer = u64::from_le_bytes(raw[0..8].try_into().expect("fixed slice"));
    let offset = u64::from_le_bytes(raw[8..16].try_into().expect("fixed slice"));
    let buffer_size = u64::from_le_bytes(raw[16..24].try_into().expect("fixed slice"));
    if buffer == 0 || buffer_size == 0 {
        return OK;
    }
    let next = offset.saturating_add(bytes).min(buffer_size);
    if ctx.mem.write(info + 8, &next.to_le_bytes()) {
        OK
    } else {
        SCE_ERROR_MEMORY_FAULT
    }
}

/// `sceAcm_ConvReverb_SharedInput(...)` appends one 1024-byte command.
fn hle_conv_reverb_shared_input(ctx: &HleContext, args: &[u64]) -> u64 {
    advance_batch(ctx, args.first().copied().unwrap_or(0), 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn read_u32(mem: &crate::TestMemory, addr: u64) -> u32 {
        let mut bytes = [0u8; 4];
        assert!(mem.read(addr, &mut bytes));
        u32::from_le_bytes(bytes)
    }

    fn read_u64(mem: &crate::TestMemory, addr: u64) -> u64 {
        let mut bytes = [0u8; 8];
        assert!(mem.read(addr, &mut bytes));
        u64::from_le_bytes(bytes)
    }

    #[test]
    fn context_create_writes_u32_handle_without_clobbering_canary() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x14, &0xCAFE_BABEu32.to_le_bytes()));

        assert_eq!(hle_context_create(&ctx, &[0]), SCE_ERROR_INVALID_ARGUMENT);
        assert_eq!(hle_context_create(&ctx, &[0x10]), OK);
        assert_ne!(read_u32(&mem, 0x10), 0);
        assert_eq!(read_u32(&mem, 0x14), 0xCAFE_BABE);
        assert_eq!(
            hle_context_create(&ctx, &[0xFFFF_0000]),
            SCE_ERROR_MEMORY_FAULT
        );
    }

    #[test]
    fn batch_start_initializes_error_and_batch_out_parameters() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x20, &[0xCD; BATCH_ERROR_SIZE]));

        assert_eq!(
            hle_batch_start_buffers(&ctx, &[1, 1, 0, 0x20, 0x60]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_batch_start_buffers(&ctx, &[1, 0, 0, 0x20, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(hle_batch_start_buffers(&ctx, &[1, 0, 0, 0x20, 0x60]), OK);
        let mut error = [0xCD; BATCH_ERROR_SIZE];
        assert!(mem.read(0x20, &mut error));
        assert_eq!(error, [0; BATCH_ERROR_SIZE]);
        assert_ne!(read_u32(&mem, 0x60), 0);
    }

    #[test]
    fn reverb_builder_advances_and_clamps_batch_cursor() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mut info = [0u8; BATCH_INFO_SIZE];
        info[0..8].copy_from_slice(&0x80u64.to_le_bytes());
        info[8..16].copy_from_slice(&512u64.to_le_bytes());
        info[16..24].copy_from_slice(&1200u64.to_le_bytes());
        assert!(mem.write(0x20, &info));

        assert_eq!(hle_conv_reverb_shared_input(&ctx, &[0x20]), OK);
        assert_eq!(read_u64(&mem, 0x28), 1200);
        assert_eq!(hle_conv_reverb_shared_input(&ctx, &[0]), OK);
    }
}
