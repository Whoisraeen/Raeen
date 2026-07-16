//! HLE media-subsystem handshakes: **libSceAjm** (audio decode),
//! **libSceNgs2** (audio synthesis), **libSceAvPlayer** (video), **libSceUlt**.
//!
//! Ported faithfully from SharpEmu's Ajm/Ngs2/AvPlayer/Ult exports (GPL-2.0),
//! which are themselves **handshake/handle-management stubs** — SharpEmu does
//! not actually decode/synthesize/demux here either. XPS5X mirrors that: the
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
    registry.register("libSceAjm", "sceAjmBatchInitialize", hle_ajm_unknown_abi);
    registry.register("libSceAjm", "sceAjmBatchJobInitialize", hle_ajm_unknown_abi);
    registry.register("libSceAjm", "sceAjmBatchJobDecode", hle_ajm_unknown_abi);
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
    registry.register("libSceAjm", "sceAjmBatchStart", hle_ajm_batch_start);
    registry.register("libSceAjm", "sceAjmBatchWait", hle_ok);
    registry.register("libSceAjm", "sceAjmBatchCancel", hle_ok);
    registry.register("libSceAjm", "sceAjmBatchErrorDump", hle_ok);

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
        hle_ngs2_create_out2,
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

/// `sceAjmBatchStart(context, batch, size, priority, error_out, batchid_out)`
/// (PS4 `sceAjmBatchStartBuffer` shape): the batch "completes" immediately —
/// a fresh batch id is written so the paired `sceAjmBatchWait` succeeds, and
/// no job in the buffer is executed (decode output is silence).
fn hle_ajm_batch_start(ctx: &HleContext, args: &[u64]) -> u64 {
    let batch_out = args.get(5).copied().unwrap_or(0);
    if batch_out != 0 {
        let batch = next_handle() as u32;
        if !ctx.mem.write(batch_out, &batch.to_le_bytes()) {
            return AJM_ERROR_INVALID_PARAMETER;
        }
    }
    OK
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

/// Ngs2 create-family (`System`/`Rack` create, `RackGetVoiceHandle`): the
/// output handle lives in the **third** argument; write a fresh handle there.
fn hle_ngs2_create_out2(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.get(2).copied().unwrap_or(0);
    if out == 0 {
        return NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    if !ctx.mem.write(out, &next_handle().to_le_bytes()) {
        return NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn ctx_env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
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
}
