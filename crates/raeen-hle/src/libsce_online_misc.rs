//! HLE grab-bag of small online/social/service libraries (Tier B, 2026-07-27).
//!
//! Groups the measured single-digit-import libraries the way
//! `libsce_peripheral.rs` groups small peripheral families. Every entry here
//! has deliberate **offline semantics** — never a blind zero:
//!
//! * **libSceRemoteplay** — service up, no remote-play session:
//!   `GetConnectionStatus` reports DISCONNECT (shadPS4's model).
//! * **libSceSharePlay / libSceGameLiveStreaming** — init/term handshakes
//!   succeed; no session ever exists to stream.
//! * **libSceContentDelete** — init/term succeed; a delete-by-path reports OK
//!   (no service-managed content exists, so "it is gone" holds vacuously).
//! * **libSceContentSearch** — the media library is honestly empty: searches
//!   and metadata reads report not-found. The result-struct layouts are not
//!   publicly documented, so an empty *error* is used instead of fabricating
//!   an empty *list* into an unknown layout (uncertain codes marked below).
//! * **libSceNpUtility (bandwidth test)** — offline: starting a PSN bandwidth
//!   test reports `SIGNED_OUT`; teardown entry points succeed.
//! * **libSceNpGameIntent** — no pending game intent (activities/deep links
//!   are PSN-delivered); receive/property reads report not-found.
//! * **libSceVideoRecordingP** — the recorder is **disabled**: size query,
//!   open, start, and info entry points refuse up front (a title skips its
//!   recording feature), while stop/close teardown succeeds and the status
//!   poll reports "not recording" so no wait loop can hang.
//! * **libScePlayerInvitationDialog / libScePlayerSelectionDialog** — the
//!   measured imports are status/terminate only; the status reports the
//!   dialog was never opened (`NONE`) and terminate succeeds.
//!
//! Exact `SCE_*_ERROR_*` values for these small libraries are not publicly
//! documented; where an error must be produced, the generic kernel error
//! spellings used across this crate stand in, each marked "uncertain code".

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::debug;

const OK: u64 = 0;
/// Generic kernel EPERM — "operation refused"; uncertain code (the real
/// per-library values are undocumented).
const SCE_ERROR_OPERATION_NOT_PERMITTED: u64 = 0x8002_0001;
/// Generic kernel ENOENT — "nothing found"; uncertain code, same note.
const SCE_ERROR_NOT_FOUND: u64 = 0x8002_0002;
/// Generic kernel EFAULT for bad out-pointers.
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;
/// `ORBIS_NP_ERROR_SIGNED_OUT` (shadPS4 `np_error.h`) — the honest offline
/// answer for anything that would talk to PSN.
const NP_ERROR_SIGNED_OUT: u64 = 0x8055_0006;

/// `ORBIS_REMOTEPLAY_CONNECTION_STATUS_DISCONNECT` (shadPS4 `remote_play`).
const REMOTEPLAY_DISCONNECT: u32 = 0;

/// Register every grouped library.
///
/// The plain `hle_ok` registrations below are Initialize/Terminate/teardown
/// brackets for subsystems Raeen models as absent (no remote-play client, no
/// share-play peer, no live stream, no recorder session): their whole contract
/// is the return code, the substantive queries beside them give real honest
/// answers (disconnected / not-found / not-recording), and there is
/// legitimately nothing to set up or release — so OK is the complete behavior,
/// not a silent skip.
pub fn register(registry: &HleRegistry) {
    // --- libSceRemoteplay ---------------------------------------------------
    registry.register("libSceRemoteplay", "sceRemoteplayInitialize", hle_ok);
    registry.register("libSceRemoteplay", "sceRemoteplayTerminate", hle_ok);
    registry.register(
        "libSceRemoteplay",
        "sceRemoteplayGetConnectionStatus",
        hle_remoteplay_connection_status,
    );

    // --- libSceSharePlay / libSceGameLiveStreaming --------------------------
    registry.register("libSceSharePlay", "sceSharePlayInitialize", hle_ok);
    registry.register("libSceSharePlay", "sceSharePlayTerminate", hle_ok);
    registry.register(
        "libSceGameLiveStreaming",
        "sceGameLiveStreamingInitialize",
        hle_ok,
    );
    registry.register(
        "libSceGameLiveStreaming",
        "sceGameLiveStreamingTerminate",
        hle_ok,
    );

    // --- libSceContentDelete -------------------------------------------------
    registry.register("libSceContentDelete", "sceContentDeleteInitialize", hle_ok);
    registry.register("libSceContentDelete", "sceContentDeleteTerminate", hle_ok);
    registry.register(
        "libSceContentDelete",
        "sceContentDeleteByPath",
        hle_content_delete_by_path,
    );

    // --- libSceContentSearch --------------------------------------------------
    registry.register("libSceContentSearch", "sceContentSearchInit", hle_ok);
    registry.register("libSceContentSearch", "sceContentSearchTerm", hle_ok);
    for name in [
        "sceContentSearchSearchContent",
        "sceContentSearchOpenMetadata",
        "sceContentSearchGetMetadataValue",
        "sceContentSearchGetMetadataFieldInfo",
        "sceContentSearchCloseMetadata",
    ] {
        registry.register_incomplete(
            "libSceContentSearch",
            name,
            hle_content_search_empty,
            "media library is empty; result layout undocumented, so reports not-found",
        );
    }

    // --- libSceNpUtility (bandwidth test) ------------------------------------
    registry.register(
        "libSceNpUtility",
        "sceNpBandwidthTestInitStartDownload",
        hle_bandwidth_test_offline,
    );
    registry.register(
        "libSceNpUtility",
        "sceNpBandwidthTestInitStartUpload",
        hle_bandwidth_test_offline,
    );
    registry.register(
        "libSceNpUtility",
        "sceNpBandwidthTestGetStatus",
        hle_bandwidth_test_status,
    );
    registry.register("libSceNpUtility", "sceNpBandwidthTestAbort", hle_ok);
    registry.register("libSceNpUtility", "sceNpBandwidthTestShutdown", hle_ok);

    // --- libSceNpGameIntent ---------------------------------------------------
    registry.register(
        "libSceNpGameIntent",
        "sceNpGameIntentReceiveIntent",
        hle_game_intent_none,
    );
    registry.register(
        "libSceNpGameIntent",
        "sceNpGameIntentGetPropertyValueString",
        hle_game_intent_property,
    );
    registry.register("libSceNpGameIntent", "sceNpGameIntentTerminate", hle_ok);

    // --- libSceVideoRecordingP -----------------------------------------------
    for name in [
        "sceVideoRecordingQueryMemSize",
        "sceVideoRecordingOpen",
        "sceVideoRecordingStart",
        "sceVideoRecordingSetInfo",
        "sceVideoRecordingGetInfo",
    ] {
        registry.register_incomplete(
            "libSceVideoRecordingP",
            name,
            hle_video_recording_disabled,
            "recorder disabled: refuses up front so titles skip their recording feature",
        );
    }
    registry.register("libSceVideoRecordingP", "sceVideoRecordingStop", hle_ok);
    registry.register("libSceVideoRecordingP", "sceVideoRecordingClose", hle_ok);
    // Status poll returns "not recording" (0) so no wait loop can hang.
    registry.register(
        "libSceVideoRecordingP",
        "sceVideoRecordingGetStatus",
        hle_ok,
    );
    // Measured anonymous import (no dictionary name; see the 2026-07-25
    // ledger's anonymous-NID inventory). Refused like the rest of the
    // disabled recorder, with a one-shot log capturing the argument shape.
    registry.register_nid(
        "libSceVideoRecordingP",
        "sceVideoRecordingUnknown8904BA0D4B4BC9B1",
        0x8904_ba0d_4b4b_c9b1,
        hle_video_recording_unknown,
    );

    // --- Player invitation/selection dialogs ---------------------------------
    registry.register(
        "libScePlayerInvitationDialog",
        "scePlayerInvitationDialogUpdateStatus",
        hle_dialog_status_none,
    );
    registry.register(
        "libScePlayerInvitationDialog",
        "scePlayerInvitationDialogTerminate",
        hle_ok,
    );
    registry.register(
        "libScePlayerSelectionDialog",
        "scePlayerSelectionDialogTerminate",
        hle_ok,
    );
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    OK
}

/// `sceRemoteplayGetConnectionStatus(userId, int *status)`: no remote-play
/// client is ever connected.
fn hle_remoteplay_connection_status(ctx: &HleContext, args: &[u64]) -> u64 {
    let status = args.get(1).copied().unwrap_or(0);
    if status == 0 || !ctx.mem.write(status, &REMOTEPLAY_DISCONNECT.to_le_bytes()) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceContentDeleteByPath(path, ...)`: no service-managed downloadable
/// content exists in Raeen's model, so the requested end state ("that content
/// is gone") already holds; nothing on the host is touched.
fn hle_content_delete_by_path(_ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) == 0 {
        return SCE_ERROR_MEMORY_FAULT;
    }
    debug!("sceContentDeleteByPath -> OK (no service-managed content exists)");
    OK
}

/// libSceContentSearch queries: the media library is empty. Reported as
/// not-found (uncertain code) rather than writing an empty list into an
/// undocumented result layout.
fn hle_content_search_empty(_ctx: &HleContext, _args: &[u64]) -> u64 {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        debug!("libSceContentSearch query -> NOT_FOUND (media library empty offline)");
    }
    SCE_ERROR_NOT_FOUND
}

/// `sceNpBandwidthTestInitStart{Download,Upload}(...)`: a PSN bandwidth test
/// cannot run signed out; the title's connectivity probe fails cleanly.
fn hle_bandwidth_test_offline(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceNpBandwidthTestInitStart* -> SIGNED_OUT (offline)");
    NP_ERROR_SIGNED_OUT
}

/// `sceNpBandwidthTestGetStatus(ctxId, status*)`: no test context can exist
/// (creation refuses above).
fn hle_bandwidth_test_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_ERROR_NOT_FOUND
}

/// `sceNpGameIntentReceiveIntent(...)`: no game intent is ever pending —
/// intents (activity/invite deep links) are PSN-delivered.
fn hle_game_intent_none(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_ERROR_NOT_FOUND
}

/// `sceNpGameIntentGetPropertyValueString(intent, key, buf, bufSize)`: no
/// intent exists; when the caller passed a buffer, an empty string is placed
/// in it defensively so a caller ignoring the error never reads garbage.
fn hle_game_intent_property(ctx: &HleContext, args: &[u64]) -> u64 {
    let buf = args.get(2).copied().unwrap_or(0);
    let buf_size = args.get(3).copied().unwrap_or(0);
    if buf != 0 && buf_size > 0 {
        let _ = ctx.mem.write(buf, &[0u8]);
    }
    SCE_ERROR_NOT_FOUND
}

/// The disabled video recorder: refuse before any resources are committed.
fn hle_video_recording_disabled(_ctx: &HleContext, _args: &[u64]) -> u64 {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        debug!("libSceVideoRecordingP -> refused (recorder disabled; no capture backend)");
    }
    SCE_ERROR_OPERATION_NOT_PERMITTED
}

/// The measured anonymous libSceVideoRecordingP import: ABI unknown, so no
/// guest memory is touched; the arguments are logged once so a real run
/// records the shape, and the call is refused like the rest of the disabled
/// recorder.
fn hle_video_recording_unknown(_ctx: &HleContext, args: &[u64]) -> u64 {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            ?args,
            "libSceVideoRecordingP NID 0x8904ba0d4b4bc9b1: UNKNOWN ABI — refusing \
             without side effects (recorder disabled); record these args to reverse"
        );
    }
    SCE_ERROR_OPERATION_NOT_PERMITTED
}

/// Dialogs whose only measured imports are status/terminate: the dialog was
/// never opened, so the shared common-dialog status is `NONE` (0).
fn hle_dialog_status_none(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0 // SceCommonDialogStatus::NONE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            raeen_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x1000),
            crate::TestAllocator::new(0),
        )
    }

    #[test]
    fn remoteplay_reports_disconnected() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, &0xFFFF_FFFFu32.to_le_bytes()));
        assert_eq!(
            hle_remoteplay_connection_status(&ctx, &[0x1000_0000, 0x100]),
            OK
        );
        let mut s = [0u8; 4];
        assert!(mem.read(0x100, &mut s));
        assert_eq!(u32::from_le_bytes(s), REMOTEPLAY_DISCONNECT);
        assert_eq!(
            hle_remoteplay_connection_status(&ctx, &[0x1000_0000, 0]),
            SCE_ERROR_MEMORY_FAULT
        );
    }

    #[test]
    fn offline_refusals_use_the_documented_semantics() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // PSN bandwidth test refuses signed-out.
        assert_eq!(hle_bandwidth_test_offline(&ctx, &[]), NP_ERROR_SIGNED_OUT);
        assert_eq!(
            hle_bandwidth_test_status(&ctx, &[1, 0x100]),
            SCE_ERROR_NOT_FOUND
        );
        // No pending game intent; the property read leaves an empty string.
        assert_eq!(hle_game_intent_none(&ctx, &[0x100]), SCE_ERROR_NOT_FOUND);
        assert!(mem.write(0x200, &[0xABu8; 4]));
        assert_eq!(
            hle_game_intent_property(&ctx, &[0x100, 0x180, 0x200, 4]),
            SCE_ERROR_NOT_FOUND
        );
        let mut b = [0u8; 1];
        assert!(mem.read(0x200, &mut b));
        assert_eq!(b[0], 0, "defensive empty string");
        // Recorder refuses; content search is empty; content delete succeeds.
        assert_eq!(
            hle_video_recording_disabled(&ctx, &[]),
            SCE_ERROR_OPERATION_NOT_PERMITTED
        );
        assert_eq!(hle_content_search_empty(&ctx, &[]), SCE_ERROR_NOT_FOUND);
        assert_eq!(hle_content_delete_by_path(&ctx, &[0x100]), OK);
        assert_eq!(
            hle_content_delete_by_path(&ctx, &[0]),
            SCE_ERROR_MEMORY_FAULT
        );
        // Never-opened dialogs report status NONE.
        assert_eq!(hle_dialog_status_none(&ctx, &[]), 0);
    }

    /// Every measured import across the grouped libraries resolves, and the
    /// anonymous VideoRecording NID is bound by NID (name-hashing its
    /// placeholder would leave it unreachable).
    #[test]
    fn measured_grouped_imports_are_registered() {
        let registry = HleRegistry::new();
        for (lib, name) in [
            ("libSceRemoteplay", "sceRemoteplayInitialize"),
            ("libSceRemoteplay", "sceRemoteplayTerminate"),
            ("libSceRemoteplay", "sceRemoteplayGetConnectionStatus"),
            ("libSceSharePlay", "sceSharePlayInitialize"),
            ("libSceSharePlay", "sceSharePlayTerminate"),
            ("libSceGameLiveStreaming", "sceGameLiveStreamingInitialize"),
            ("libSceGameLiveStreaming", "sceGameLiveStreamingTerminate"),
            ("libSceContentDelete", "sceContentDeleteInitialize"),
            ("libSceContentDelete", "sceContentDeleteTerminate"),
            ("libSceContentDelete", "sceContentDeleteByPath"),
            ("libSceContentSearch", "sceContentSearchInit"),
            ("libSceContentSearch", "sceContentSearchTerm"),
            ("libSceContentSearch", "sceContentSearchSearchContent"),
            ("libSceContentSearch", "sceContentSearchOpenMetadata"),
            ("libSceContentSearch", "sceContentSearchGetMetadataValue"),
            (
                "libSceContentSearch",
                "sceContentSearchGetMetadataFieldInfo",
            ),
            ("libSceContentSearch", "sceContentSearchCloseMetadata"),
            ("libSceNpUtility", "sceNpBandwidthTestInitStartDownload"),
            ("libSceNpUtility", "sceNpBandwidthTestInitStartUpload"),
            ("libSceNpUtility", "sceNpBandwidthTestGetStatus"),
            ("libSceNpUtility", "sceNpBandwidthTestAbort"),
            ("libSceNpUtility", "sceNpBandwidthTestShutdown"),
            ("libSceNpGameIntent", "sceNpGameIntentReceiveIntent"),
            (
                "libSceNpGameIntent",
                "sceNpGameIntentGetPropertyValueString",
            ),
            ("libSceNpGameIntent", "sceNpGameIntentTerminate"),
            ("libSceVideoRecordingP", "sceVideoRecordingQueryMemSize"),
            ("libSceVideoRecordingP", "sceVideoRecordingOpen"),
            ("libSceVideoRecordingP", "sceVideoRecordingStart"),
            ("libSceVideoRecordingP", "sceVideoRecordingSetInfo"),
            ("libSceVideoRecordingP", "sceVideoRecordingGetInfo"),
            ("libSceVideoRecordingP", "sceVideoRecordingStop"),
            ("libSceVideoRecordingP", "sceVideoRecordingClose"),
            ("libSceVideoRecordingP", "sceVideoRecordingGetStatus"),
            (
                "libScePlayerInvitationDialog",
                "scePlayerInvitationDialogUpdateStatus",
            ),
            (
                "libScePlayerInvitationDialog",
                "scePlayerInvitationDialogTerminate",
            ),
            (
                "libScePlayerSelectionDialog",
                "scePlayerSelectionDialogTerminate",
            ),
        ] {
            assert!(
                registry.is_implemented(lib, name),
                "{lib}::{name} must be registered"
            );
        }
        // The anonymous NID is bound explicitly.
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, _)| *nid == 0x8904_ba0d_4b4b_c9b1),
            "anonymous VideoRecording NID must be NID-bound"
        );
    }
}
