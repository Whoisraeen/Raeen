//! HLE media-subsystem handshakes: **libSceAjm** (audio decode),
//! **libSceNgs2** (audio synthesis), **libSceAvPlayer** (video), **libSceUlt**.
//!
//! Ported faithfully from SharpEmu's Ajm/Ngs2/AvPlayer/Ult exports (GPL-2.0),
//! which are themselves **handshake/handle-management stubs** — SharpEmu does
//! not actually decode/synthesize/demux here either. Raeen mirrors that: the
//! subsystems initialize and hand out handles so a title's setup path
//! proceeds, but **no real media is produced yet** — `Ngs2VoiceGetState`
//! reports idle, `AvPlayerIsActive` reports inactive, and the data-fetch calls
//! return no frame. This is the same "let the title run, output is a follow-up"
//! shape as `libSceAudioOut`/`libSceVideoOut`; real decode/synthesis is future
//! work, not something faked here.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::debug;

const OK: u64 = 0;
/// `ORBIS_AJM_ERROR_INVALID_PARAMETER`.
const AJM_ERROR_INVALID_PARAMETER: u64 = 0x8093_0005;
/// Ngs2 invalid out-address error.
const NGS2_ERROR_INVALID_OUT_ADDRESS: u64 = 0x8080_4002;

/// Monotonic handle source for Ngs2 systems/racks/voices (non-zero).
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn next_handle() -> u64 {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
}

/// Register the media-subsystem HLE functions.
pub fn register(registry: &HleRegistry) {
    // libSceAjm — audio decoder context handshake, plus the batch-decode
    // surface the measured title (Minecraft) imports. Handles are real,
    // batches complete instantly, and NO samples are decoded — silence, not
    // a hang. Signatures follow the documented PS4 AJM ABI where one exists;
    // the PS5-only names (`BatchInitialize`, the `BatchJob*` builders) are
    // unknown-ABI unblock stubs that log their arguments once.
    registry.register("libSceAjm", "sceAjmInitialize", hle_ajm_initialize);
    registry.register("libSceAjm", "sceAjmFinalize", hle_ok);
    registry.register("libSceAjm", "sceAjmModuleRegister", hle_ok);
    registry.register("libSceAjm", "sceAjmModuleUnregister", hle_ok);
    registry.register("libSceAjm", "sceAjmInstanceCreate", hle_ajm_instance_create);
    registry.register("libSceAjm", "sceAjmInstanceDestroy", hle_ok);
    // Batch surface, ported from SharpEmu's silence stubs (GPL-2.0,
    // AjmExports.cs commits 2272b9b + d3600c9), with BatchInitialize's
    // five-u64 descriptor layout independently reimplemented from KytyPS5.
    // The guest owns the batch storage, batches complete synchronously, and NO
    // samples are decoded — silence, not a hang. These are Bink/AJM hot-path
    // calls, so none of them WARN per call.
    registry.register(
        "libSceAjm",
        "sceAjmBatchInitialize",
        hle_ajm_batch_initialize,
    );
    registry.register(
        "libSceAjm",
        "sceAjmBatchJobDecode",
        hle_ajm_batch_job_decode,
    );
    registry.register("libSceAjm", "sceAjmBatchStart", hle_ajm_batch_start);
    registry.register(
        "libSceAjm",
        "sceAjmBatchStartBuffer",
        hle_ajm_batch_start_buffer,
    );
    registry.register("libSceAjm", "sceAjmBatchWait", hle_ajm_batch_wait);
    registry.register("libSceAjm", "sceAjmBatchCancel", hle_ok);
    registry.register("libSceAjm", "sceAjmBatchErrorDump", hle_ok);
    // The remaining PS5-only batch-*builder* names have no public ABI; keep the
    // unblock stub that logs its arguments once so a real run records the shape
    // to reverse (these are not on the per-frame decode hot path).
    registry.register("libSceAjm", "sceAjmBatchJobInitialize", hle_ajm_unknown_abi);
    registry.register(
        "libSceAjm",
        "sceAjmBatchJobClearContext",
        hle_ajm_unknown_abi,
    );
    registry.register(
        "libSceAjm",
        "sceAjmBatchJobSetGaplessDecode",
        hle_ajm_unknown_abi,
    );
    registry.register(
        "libSceAjm",
        "sceAjmBatchJobSetResampleParameters",
        hle_ajm_unknown_abi,
    );

    // libSceNgs2 — audio synthesis system/rack/voice management (no output).
    registry.register(
        "libSceNgs2",
        "sceNgs2SystemCreateWithAllocator",
        hle_ngs2_create_out2,
    );
    registry.register("libSceNgs2", "sceNgs2SystemDestroy", hle_ok);
    registry.register(
        "libSceNgs2",
        "sceNgs2RackCreateWithAllocator",
        hle_ngs2_rack_create,
    );
    registry.register("libSceNgs2", "sceNgs2RackDestroy", hle_ok);
    registry.register(
        "libSceNgs2",
        "sceNgs2RackGetVoiceHandle",
        hle_ngs2_create_out2,
    );
    registry.register("libSceNgs2", "sceNgs2VoiceControl", hle_ok);
    registry.register("libSceNgs2", "sceNgs2VoiceRunCommands", hle_ok);
    registry.register("libSceNgs2", "sceNgs2VoiceGetState", hle_ok);
    registry.register("libSceNgs2", "sceNgs2VoiceGetStateFlags", hle_ok);

    // libSceAvPlayer — video player (never becomes active; no frames).
    registry.register("libSceAvPlayer", "sceAvPlayerInit", hle_ok); // null handle → title skips FMV
    registry.register("libSceAvPlayer", "sceAvPlayerPostInit", hle_ok);
    registry.register("libSceAvPlayer", "sceAvPlayerIsActive", hle_ok); // 0 = inactive
    registry.register("libSceAvPlayer", "sceAvPlayerGetVideoDataEx", hle_ok); // no frame
    registry.register("libSceAvPlayer", "sceAvPlayerGetAudioData", hle_ok); // no frame
    registry.register("libSceAvPlayer", "sceAvPlayerClose", hle_ok);

    // libSceUlt — user-level threads library init.
    registry.register("libSceUlt", "sceUltInitialize", hle_ok);
}

/// Benign success (`rax = 0`): idle/inactive/no-frame, per the reference.
fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    OK
}

/// `sceAjmInitialize(reserved, out_context)`: hand out a context id. Requires a
/// zero `reserved` and a writable out-pointer.
fn hle_ajm_initialize(ctx: &HleContext, args: &[u64]) -> u64 {
    let reserved = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    if reserved != 0 || out == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let context_id = next_handle() as u32;
    if !ctx.mem.write(out, &context_id.to_le_bytes()) {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    debug!("sceAjmInitialize -> context {context_id}");
    OK
}

/// `sceAjmInstanceCreate(context, codec, flags, out_instance)`: hand out an
/// instance id (documented PS4 AJM ABI; the PS5 library keeps the name).
/// No decoder is created — batches against it complete with no output.
fn hle_ajm_instance_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.get(3).copied().unwrap_or(0);
    if out == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let instance = next_handle() as u32;
    if !ctx.mem.write(out, &instance.to_le_bytes()) {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    debug!("sceAjmInstanceCreate -> instance {instance}");
    OK
}

// AjmBatchInfo layout: buffer, offset, size, last_good_job, last_good_job_ra
// (5x u64). AjmBatchError: int error_code; ptr job_addr; u32 cmd_offset; ptr
// job_ra (24 bytes). AjmSidebandResult(8) + AjmSidebandStream(16) +
// AjmSidebandMFrame(8) = 32-byte decode sideband. A single decode "runs" 64
// bytes of the batch buffer. All ported from SharpEmu AjmExports.cs.
const AJM_BATCH_INFO_OFFSET_FIELD: u64 = 8;
const AJM_BATCH_INFO_SIZE_FIELD: u64 = 16;
const AJM_BATCH_INFO_LAST_GOOD_JOB_FIELD: u64 = 24;
const AJM_JOB_RUN_SIZE: u64 = 64;
const AJM_BATCH_ERROR_BYTES: usize = 24;
const AJM_DECODE_SIDEBAND_BYTES: usize = 32;
/// Cap on how much guest PCM the silence stub will zero in one decode, so a
/// bogus `outputSize` cannot make us clear an unbounded span (SharpEmu's
/// `MaxSilentPcmBytes`).
const AJM_MAX_SILENT_PCM_BYTES: u64 = 1 << 20;

/// `sceAjmBatchInitialize(buffer, size, info)`: initialize the five-u64
/// `AjmBatchInfo` descriptor (`buffer`, zero offset, capacity, and two null
/// last-good-job pointers). Returning success without this write left
/// Minecraft's per-grain codec builder reading stale descriptor fields.
fn hle_ajm_batch_initialize(ctx: &HleContext, args: &[u64]) -> u64 {
    let buffer = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    let info = args.get(2).copied().unwrap_or(0);
    if buffer == 0 || info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    let mut descriptor = [0u8; 40];
    descriptor[0..8].copy_from_slice(&buffer.to_le_bytes());
    descriptor[16..24].copy_from_slice(&size.to_le_bytes());
    if ctx.mem.write(info, &descriptor) {
        OK
    } else {
        AJM_ERROR_INVALID_PARAMETER
    }
}

/// `sceAjmBatchStart(context, info, priority, error_out, batchid_out)`
/// (SharpEmu `AjmBatchStart`, NID `5tOfnaClcqM`): the batch "completes"
/// immediately — the error struct is cleared, a fresh batch id is written so the
/// paired `sceAjmBatchWait` succeeds, and no job in the buffer is executed
/// (decode output is silence, filled at job-enqueue time). No per-call log.
fn hle_ajm_batch_start(ctx: &HleContext, args: &[u64]) -> u64 {
    let info = args.get(1).copied().unwrap_or(0);
    let error_out = args.get(3).copied().unwrap_or(0);
    let batch_out = args.get(4).copied().unwrap_or(0);
    ajm_batch_start_common(ctx, info, error_out, batch_out)
}

/// `sceAjmBatchStartBuffer(context, pBatch, batchSize, priority, pBatchError,
/// pBatchId)` — the PS4/PS5 6-argument form. Same instant-complete silence
/// semantics as [`hle_ajm_batch_start`], with the error struct and batch id one
/// register further along (`pBatchError`=r8, `pBatchId`=r9).
fn hle_ajm_batch_start_buffer(ctx: &HleContext, args: &[u64]) -> u64 {
    let batch = args.get(1).copied().unwrap_or(0);
    let error_out = args.get(4).copied().unwrap_or(0);
    let batch_out = args.get(5).copied().unwrap_or(0);
    ajm_batch_start_common(ctx, batch, error_out, batch_out)
}

/// Shared body for the batch-start family: require the batch buffer and the
/// batch-id out-pointer, clear the error sideband, and publish a fresh batch id.
fn ajm_batch_start_common(ctx: &HleContext, batch: u64, error_out: u64, batch_out: u64) -> u64 {
    if batch == 0 || batch_out == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    clear_ajm_batch_error(ctx, error_out);
    let batch_id = next_handle() as u32;
    if !ctx.mem.write(batch_out, &batch_id.to_le_bytes()) {
        return AJM_ERROR_INVALID_PARAMETER;
    }
    OK
}

/// `sceAjmBatchWait(context, batch, timeout, error_out)`: batches complete
/// synchronously in start, so wait is a no-op success that clears the error
/// sideband (SharpEmu `AjmBatchWait`). No per-call log.
fn hle_ajm_batch_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    let error_out = args.get(3).copied().unwrap_or(0);
    clear_ajm_batch_error(ctx, error_out);
    OK
}

/// `sceAjmBatchJobDecode(info, instance, input, inputSize, output, outputSize,
/// result)`: enqueue a decode job on a batch. Titles call this on the Bink/AJM
/// hot path; leaving it unresolved floods Import WARN spam. This is a **silence**
/// stub, not a codec — it accepts any codec instance id (including the Gen5
/// range), advances the batch cursor, clears the PCM output, and reports the
/// input as fully consumed with silence produced, so the title advances its
/// bitstream cursor instead of spinning on the same packet. No per-call log.
///
/// Ported from SharpEmu `AjmBatchJobDecode` (AjmExports.cs, NID `39WxhR-ePew`).
/// `result` is the 7th argument — the first stack argument (`args[6]`).
fn hle_ajm_batch_job_decode(ctx: &HleContext, args: &[u64]) -> u64 {
    let info = args.first().copied().unwrap_or(0);
    let input_size = args.get(3).copied().unwrap_or(0);
    let output = args.get(4).copied().unwrap_or(0);
    let output_size = args.get(5).copied().unwrap_or(0);
    let result = args.get(6).copied().unwrap_or(0);

    if info == 0 {
        return AJM_ERROR_INVALID_PARAMETER;
    }

    // Best-effort: bump the batch cursor when the guest filled AjmBatchInfo.
    // Still succeed without it — failing here would reintroduce hot-path spam
    // via the title's retries.
    let _ = try_append_batch_job(ctx, info, AJM_JOB_RUN_SIZE);

    // Silence: clear the PCM out region and claim full input consumed so the
    // guest advances its bitstream cursor rather than re-submitting forever.
    if output != 0 && output_size != 0 && output_size <= AJM_MAX_SILENT_PCM_BYTES {
        clear_guest_memory(ctx, output, output_size);
    }

    let input_consumed = input_size.min(i32::MAX as u64) as i32;
    let frames: u32 = u32::from(input_size != 0 || output_size != 0);
    write_decode_stream_result(ctx, result, input_consumed, frames);
    OK
}

/// Advance an AjmBatchInfo submission cursor by one job run, zeroing the job
/// slot. Returns false (and changes nothing) if the descriptor is malformed or
/// the buffer has no room — the caller succeeds regardless.
fn try_append_batch_job(ctx: &HleContext, info: u64, job_size: u64) -> bool {
    let Some(buffer) = read_u64(ctx, info) else {
        return false;
    };
    let Some(offset) = read_u64(ctx, info + AJM_BATCH_INFO_OFFSET_FIELD) else {
        return false;
    };
    let Some(size) = read_u64(ctx, info + AJM_BATCH_INFO_SIZE_FIELD) else {
        return false;
    };
    if buffer == 0 || job_size == 0 || offset > size || size - offset < job_size {
        return false;
    }
    let job_address = buffer + offset;
    clear_guest_memory(ctx, job_address, job_size);
    ctx.mem.write(
        info + AJM_BATCH_INFO_LAST_GOOD_JOB_FIELD,
        &job_address.to_le_bytes(),
    ) && ctx.mem.write(
        info + AJM_BATCH_INFO_OFFSET_FIELD,
        &(offset + job_size).to_le_bytes(),
    )
}

/// Zero an AjmBatchError sideband (`error_code`, `job_addr`, `cmd_offset`,
/// `job_ra`) so a caller that inspects it after a "completed" batch sees no
/// error. A null pointer is a no-op.
fn clear_ajm_batch_error(ctx: &HleContext, error_out: u64) {
    if error_out == 0 {
        return;
    }
    let _ = ctx.mem.write(error_out, &[0u8; AJM_BATCH_ERROR_BYTES]);
}

/// Fill the 32-byte decode sideband: result = OK (0), `input_consumed` and
/// `output_written` (0) as i32, `total_decoded_samples` (0) as u64, `frames` as
/// u32. A null `result` pointer is a no-op.
fn write_decode_stream_result(ctx: &HleContext, result: u64, input_consumed: i32, frames: u32) {
    if result == 0 {
        return;
    }
    let mut sideband = [0u8; AJM_DECODE_SIDEBAND_BYTES];
    sideband[8..12].copy_from_slice(&input_consumed.to_le_bytes());
    // output_written (12..16) and total_decoded_samples (16..24) stay zero.
    sideband[24..28].copy_from_slice(&frames.to_le_bytes());
    let _ = ctx.mem.write(result, &sideband);
}

/// Zero `byte_count` bytes of guest memory starting at `address`, in bounded
/// chunks. A null address or zero count is a no-op; a failed write stops early.
fn clear_guest_memory(ctx: &HleContext, address: u64, byte_count: u64) {
    if address == 0 || byte_count == 0 {
        return;
    }
    let zero = [0u8; 256];
    let mut remaining = byte_count;
    let mut cursor = address;
    while remaining > 0 {
        let chunk = remaining.min(zero.len() as u64) as usize;
        if !ctx.mem.write(cursor, &zero[..chunk]) {
            return;
        }
        cursor += chunk as u64;
        remaining -= chunk as u64;
    }
}

/// Read a little-endian u64 from guest memory, or `None` if unreadable.
fn read_u64(ctx: &HleContext, address: u64) -> Option<u64> {
    let mut bytes = [0u8; 8];
    if ctx.mem.read(address, &mut bytes) {
        Some(u64::from_le_bytes(bytes))
    } else {
        None
    }
}

/// The PS5-only AJM batch-builder names (`sceAjmBatchInitialize`,
/// `sceAjmBatchJob*`): no public ABI exists for them, so this succeeds
/// without touching guest memory and logs its arguments once per process so
/// a real run records the shape to reverse. See `libsce_acm.rs` for the
/// same policy and why an unresolved import here was worse (a wild jump on
/// the title's main thread).
fn hle_ajm_unknown_abi(_ctx: &HleContext, args: &[u64]) -> u64 {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            ?args,
            "sceAjmBatch* builder: UNKNOWN ABI — succeeding without side effects \
             (no audio is decoded); record these args to reverse the signature"
        );
    }
    OK
}

/// Ngs2 create-family: write a fresh out-handle to the guest's out-pointer.
///
/// `sceNgs2SystemCreateWithAllocator(option, allocator, *out)` and
/// `sceNgs2RackGetVoiceHandle(rack, voiceId, *out)` carry the out pointer in
/// arg2 (rdx), but `sceNgs2RackCreateWithAllocator(system, rackId, option,
/// allocator, *out)` carries it in arg4 (r8) — SharpEmu `Ngs2Exports.cs:143`.
/// Reading the wrong slot writes the handle into an unrelated argument and
/// leaves the guest's real out-handle uninitialized, so the guest then
/// dereferences garbage (the NULL-base fault family).
fn ngs2_write_out_handle(ctx: &HleContext, args: &[u64], out_index: usize) -> u64 {
    // TEMP-DIAG (2026-07-23, ASTRO.BOT +0xe03f1a NULL-base fault diagnosis;
    // REMOVE after the investigation): dump the full arg vector and caller so
    // we can verify which register really carries the out-handle pointer for
    // each create-family member. Gated on RAEEN_TRACE_NGS2.
    if std::env::var_os("RAEEN_TRACE_NGS2").is_some() {
        tracing::warn!(
            caller = format_args!("{:#x}", ctx.caller_return_addr),
            args = ?args,
            out_index,
            "TEMP-DIAG ngs2 create-family call"
        );
    }
    let out = args.get(out_index).copied().unwrap_or(0);
    if out == 0 {
        return NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    if !ctx.mem.write(out, &next_handle().to_le_bytes()) {
        return NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    OK
}

/// System-create and `RackGetVoiceHandle`: out-handle in arg2 (rdx).
fn hle_ngs2_create_out2(ctx: &HleContext, args: &[u64]) -> u64 {
    ngs2_write_out_handle(ctx, args, 2)
}

/// `sceNgs2RackCreateWithAllocator`: out-handle in arg4 (r8).
fn hle_ngs2_rack_create(ctx: &HleContext, args: &[u64]) -> u64 {
    ngs2_write_out_handle(ctx, args, 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn ctx_env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    #[test]
    fn ajm_initialize_writes_a_context_and_validates() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // reserved must be 0 and out non-null.
        assert_eq!(
            hle_ajm_initialize(&ctx, &[1, 0x40]),
            AJM_ERROR_INVALID_PARAMETER
        );
        assert_eq!(
            hle_ajm_initialize(&ctx, &[0, 0]),
            AJM_ERROR_INVALID_PARAMETER
        );
        assert_eq!(hle_ajm_initialize(&ctx, &[0, 0x40]), OK);
        let mut b = [0u8; 4];
        assert!(mem.read(0x40, &mut b));
        assert!(u32::from_le_bytes(b) != 0, "a context id was written");
    }

    #[test]
    fn ajm_batch_initialize_writes_the_real_descriptor() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_ajm_batch_initialize(&ctx, &[0, 0x80, 0x40]),
            AJM_ERROR_INVALID_PARAMETER
        );
        assert_eq!(
            hle_ajm_batch_initialize(&ctx, &[0x80, 0x40, 0]),
            AJM_ERROR_INVALID_PARAMETER
        );
        assert_eq!(hle_ajm_batch_initialize(&ctx, &[0x80, 0x40, 0x20]), OK);
        let mut descriptor = [0xCD; 40];
        assert!(mem.read(0x20, &mut descriptor));
        assert_eq!(
            u64::from_le_bytes(descriptor[0..8].try_into().unwrap()),
            0x80
        );
        assert_eq!(u64::from_le_bytes(descriptor[8..16].try_into().unwrap()), 0);
        assert_eq!(
            u64::from_le_bytes(descriptor[16..24].try_into().unwrap()),
            0x40
        );
        assert_eq!(descriptor[24..], [0; 16]);
    }

    #[test]
    fn ngs2_create_writes_a_handle_to_the_third_arg() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // args: allocator, spec, *out_handle.
        assert_eq!(hle_ngs2_create_out2(&ctx, &[0, 0, 0x40]), OK);
        let mut b = [0u8; 8];
        assert!(mem.read(0x40, &mut b));
        assert!(u64::from_le_bytes(b) != 0, "a system handle was written");
        // NULL out → error.
        assert_eq!(
            hle_ngs2_create_out2(&ctx, &[0, 0, 0]),
            NGS2_ERROR_INVALID_OUT_ADDRESS
        );
    }

    #[test]
    fn avplayer_reports_inactive_and_no_frames() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // IsActive / GetVideoData / GetAudioData all report the benign zero.
        assert_eq!(hle_ok(&ctx, &[1]), 0, "AvPlayerIsActive → inactive");
        // Init returns a null handle so a title cleanly skips video playback.
        assert_eq!(hle_ok(&ctx, &[0x40]), 0);
    }

    fn ctx_env_big() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x2000);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    #[test]
    fn ajm_batch_registrations_are_silence_stubs() {
        let registry = HleRegistry::new();
        for name in [
            "sceAjmBatchInitialize",
            "sceAjmBatchJobDecode",
            "sceAjmBatchStart",
            "sceAjmBatchStartBuffer",
            "sceAjmBatchWait",
            "sceAjmBatchCancel",
        ] {
            assert!(
                registry.is_implemented("libSceAjm", name),
                "{name} must be registered"
            );
        }
    }

    #[test]
    fn ajm_batch_job_decode_consumes_input_zeroes_output_and_reports_silence() {
        let (kernel, mem, alloc) = ctx_env_big();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // AjmBatchInfo at 0x100: buffer=0x400, offset=0, size=0x200.
        let info = 0x100u64;
        assert!(mem.write(info, &0x400u64.to_le_bytes())); // buffer
        assert!(mem.write(info + 8, &0u64.to_le_bytes())); // offset
        assert!(mem.write(info + 16, &0x200u64.to_le_bytes())); // size

        // Output PCM region pre-filled with 0xAB so we can see it cleared.
        let output = 0x800u64;
        let output_size = 64u64;
        assert!(mem.write(output, &[0xABu8; 64]));

        let result = 0xA00u64;
        let input_size = 0x123u64;
        // args: info, instance(Gen5 codec id), input, inputSize, output, outputSize, result
        let ret = hle_ajm_batch_job_decode(
            &ctx,
            &[info, 0x2003, 0x600, input_size, output, output_size, result],
        );
        assert_eq!(ret, OK);

        // Output PCM is silence.
        let mut pcm = [0xFFu8; 64];
        assert!(mem.read(output, &mut pcm));
        assert_eq!(pcm, [0u8; 64], "decode output must be zeroed (silence)");

        // Sideband: input fully consumed, frames = 1.
        let mut consumed = [0u8; 4];
        assert!(mem.read(result + 8, &mut consumed));
        assert_eq!(i32::from_le_bytes(consumed), input_size as i32);
        let mut frames = [0u8; 4];
        assert!(mem.read(result + 24, &mut frames));
        assert_eq!(u32::from_le_bytes(frames), 1);

        // Batch cursor advanced by one job run; last-good-job recorded.
        let mut new_offset = [0u8; 8];
        assert!(mem.read(info + 8, &mut new_offset));
        assert_eq!(u64::from_le_bytes(new_offset), AJM_JOB_RUN_SIZE);

        // A null info is rejected without touching memory.
        assert_eq!(
            hle_ajm_batch_job_decode(&ctx, &[0, 0, 0, 0, 0, 0, 0]),
            AJM_ERROR_INVALID_PARAMETER
        );
    }

    #[test]
    fn ajm_batch_start_clears_error_and_publishes_a_batch_id() {
        let (kernel, mem, alloc) = ctx_env_big();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Pre-dirty the error struct so we can confirm it is cleared.
        let error_out = 0x100u64;
        assert!(mem.write(error_out, &[0xCDu8; AJM_BATCH_ERROR_BYTES]));
        let batch_out = 0x200u64;

        // sceAjmBatchStart: (context, info, priority, error_out, batchid_out).
        let ret = hle_ajm_batch_start(&ctx, &[1, 0x400, 0, error_out, batch_out]);
        assert_eq!(ret, OK);

        let mut err = [0xFFu8; AJM_BATCH_ERROR_BYTES];
        assert!(mem.read(error_out, &mut err));
        assert_eq!(err, [0u8; AJM_BATCH_ERROR_BYTES], "error sideband cleared");

        let mut batch = [0u8; 4];
        assert!(mem.read(batch_out, &mut batch));
        assert!(u32::from_le_bytes(batch) != 0, "a batch id was published");

        // Missing batch buffer or out-pointer is rejected.
        assert_eq!(
            hle_ajm_batch_start(&ctx, &[1, 0, 0, 0, batch_out]),
            AJM_ERROR_INVALID_PARAMETER
        );
        assert_eq!(
            hle_ajm_batch_start(&ctx, &[1, 0x400, 0, 0, 0]),
            AJM_ERROR_INVALID_PARAMETER
        );

        // sceAjmBatchStartBuffer: batch id one register further along (arg 5).
        let batch_out2 = 0x240u64;
        assert_eq!(
            hle_ajm_batch_start_buffer(&ctx, &[1, 0x400, 0x100, 0, error_out, batch_out2]),
            OK
        );
        let mut batch2 = [0u8; 4];
        assert!(mem.read(batch_out2, &mut batch2));
        assert!(u32::from_le_bytes(batch2) != 0);
    }
}
