//! HLE libScePlayGo — content install / streaming status.
//!
//! PlayGo lets a title query which "chunks" of its content are locally
//! available and how far a background install has progressed. Raeen runs
//! against fully-present local content (the whole title is on disk), so the
//! honest, correct answer to every query is "all chunks local, nothing left
//! to download, install complete" — which lets a title skip its
//! download-gating and proceed straight to gameplay. Values and the
//! all-`LOCAL_FAST` behavior are cross-checked against SharpEmu's
//! `PlayGoExports`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `ScePlayGoLocus::LOCAL_FAST` (3) — the chunk is present on fast local
/// storage. (`NOT_DOWNLOADED = 0`, `LOCAL_SLOW = 2`.)
const LOCUS_LOCAL_FAST: u8 = 3;
/// A fixed PlayGo handle handed back by `scePlayGoOpen`.
const PLAYGO_HANDLE: u32 = 1;
const DIALOG_STATUS_NONE: i32 = 0;
static PLAYGO_DIALOG_STATUS: AtomicI32 = AtomicI32::new(DIALOG_STATUS_NONE);
/// Cap on how many chunk loci one `scePlayGoGetLocus` will write, bounding
/// the host staging buffer against a wild `numberOfEntries`.
const MAX_ENTRIES: u64 = 1 << 16;

/// Register libScePlayGo HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libScePlayGo", "scePlayGoInitialize", hle_initialize);
    // Terminate/Close: the handle is a fixed constant with no backing state,
    // so there is legitimately nothing to release — OK is complete.
    registry.register("libScePlayGo", "scePlayGoTerminate", hle_ok);
    registry.register("libScePlayGo", "scePlayGoOpen", hle_open);
    registry.register("libScePlayGo", "scePlayGoClose", hle_ok);
    registry.register("libScePlayGo", "scePlayGoGetLocus", hle_get_locus);
    registry.register("libScePlayGo", "scePlayGoGetChunkId", hle_get_chunk_id);
    // PS5 spelling observed in Avatar. Its measured register shape is the
    // same four-argument count/list query (handle, list, capacity, out count).
    registry.register(
        "libScePlayGo",
        "scePlayGoGetInstallChunkId",
        hle_get_chunk_id,
    );
    registry.register(
        "libScePlayGo",
        "scePlayGoGetSupportedOptionalChunk",
        hle_get_supported_optional_chunk,
    );
    registry.register_incomplete(
        "libScePlayGo",
        "scePlayGoGetOptionalChunk",
        hle_get_supported_optional_chunk,
        "ABI inferred; reports no optional chunks because PlayGo metadata is not parsed",
    );
    registry.register("libScePlayGo", "scePlayGoGetProgress", hle_get_progress);
    registry.register("libScePlayGo", "scePlayGoGetToDoList", hle_get_todo_list);
    // `scePlayGoGetEta(handle, chunkIds, numberOfEntries, ScePlayGoEta *outEta)`
    // — fully-local content really does have a zero ETA, but this shim never
    // writes the out-parameter, so the title reads its own stack instead.
    registry.register_incomplete(
        "libScePlayGo",
        "scePlayGoGetEta",
        hle_ok,
        "reports success without writing the caller's ScePlayGoEta out-parameter",
    );
    registry.register(
        "libScePlayGo",
        "scePlayGoGetLanguageMask",
        hle_get_language_mask,
    );
    // `scePlayGoGetInstallSpeed(handle, ScePlayGoInstallSpeed *outSpeed)` — same
    // shape as GetEta above. `SetInstallSpeed` beside it has no output and stays
    // a plain registration.
    registry.register_incomplete(
        "libScePlayGo",
        "scePlayGoGetInstallSpeed",
        hle_ok,
        "reports success without writing the caller's ScePlayGoInstallSpeed out-parameter",
    );
    registry.register("libScePlayGo", "scePlayGoSetInstallSpeed", hle_ok);
    // SetTodoList/Prefetch hand download hints to an installer; with the whole
    // title already on local disk there is nothing to schedule or prefetch, so
    // accepting them is the complete fully-installed behavior.
    registry.register("libScePlayGo", "scePlayGoSetTodoList", hle_ok);
    registry.register("libScePlayGo", "scePlayGoPrefetch", hle_ok);

    // Register the complete dialog family together. Raeen has no host PlayGo
    // popup; the headless compatibility behavior matches shadPS4: calls
    // succeed, status remains NONE, and GetResult reports the proceed value.
    registry.register_incomplete(
        "libScePlayGoDialog",
        "scePlayGoDialogInitialize",
        hle_playgo_dialog_initialize,
        "headless compatibility dialog; no host PlayGo UI",
    );
    registry.register_incomplete(
        "libScePlayGoDialog",
        "scePlayGoDialogTerminate",
        hle_playgo_dialog_close,
        "headless compatibility dialog; no host PlayGo UI",
    );
    registry.register_incomplete(
        "libScePlayGoDialog",
        "scePlayGoDialogOpen",
        hle_playgo_dialog_open,
        "headless compatibility dialog; no host PlayGo UI",
    );
    registry.register_incomplete(
        "libScePlayGoDialog",
        "scePlayGoDialogClose",
        hle_playgo_dialog_close,
        "headless compatibility dialog; no host PlayGo UI",
    );
    registry.register_incomplete(
        "libScePlayGoDialog",
        "scePlayGoDialogGetStatus",
        hle_playgo_dialog_status,
        "headless compatibility dialog; no host PlayGo UI",
    );
    registry.register_incomplete(
        "libScePlayGoDialog",
        "scePlayGoDialogUpdateStatus",
        hle_playgo_dialog_status,
        "headless compatibility dialog; no host PlayGo UI",
    );
    registry.register_incomplete(
        "libScePlayGoDialog",
        "scePlayGoDialogGetResult",
        hle_playgo_dialog_get_result,
        "headless compatibility dialog; no host PlayGo UI",
    );
}

/// A generic success stub for the PlayGo calls whose only meaningful effect
/// is "acknowledged" against fully-local content.
fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

fn hle_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("scePlayGoInitialize()");
    SCE_OK
}

/// `scePlayGoOpen(ScePlayGoHandle *outHandle, const void *param)`: writes a
/// fixed handle and returns OK.
fn hle_open(ctx: &HleContext, args: &[u64]) -> u64 {
    let out_handle = args.first().copied().unwrap_or(0);
    debug!("scePlayGoOpen(outHandle={out_handle:#x})");
    if out_handle != 0 && !ctx.mem.write(out_handle, &PLAYGO_HANDLE.to_le_bytes()) {
        warn!("scePlayGoOpen: outHandle {out_handle:#x} not writable");
        return SCE_OK; // still report success; a title rarely fails on this
    }
    SCE_OK
}

/// `scePlayGoGetLocus(handle, chunkIds, numberOfEntries, ScePlayGoLocus
/// *outLoci)`: reports every requested chunk as `LOCAL_FAST` — a single byte
/// of value 3 per entry — so the title sees all content present.
fn hle_get_locus(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0);
    let num = args.get(2).copied().unwrap_or(0).min(MAX_ENTRIES);
    let out_loci = args.get(3).copied().unwrap_or(0);
    debug!("scePlayGoGetLocus(handle={handle}, num={num}, outLoci={out_loci:#x})");
    if out_loci == 0 {
        return SCE_OK;
    }
    let Ok(n) = usize::try_from(num) else {
        return SCE_OK;
    };
    let loci = vec![LOCUS_LOCAL_FAST; n];
    if !ctx.mem.write(out_loci, &loci) {
        warn!("scePlayGoGetLocus: outLoci {out_loci:#x} (+{n}) not writable");
    }
    SCE_OK
}

/// `scePlayGoGetChunkId(handle, outChunkIdList, numberOfEntries, outEntries)`
/// — SharpEmu's `PlayGoGetChunkId` ABI. Count-only mode when
/// `outChunkIdList == 0` (write the chunk count to `*outEntries`); otherwise
/// write `min(numberOfEntries, available)` uint16 chunk ids. Raeen runs
/// against fully-present content with no PlayGo metadata parsed yet, so the
/// honest answer is one present chunk (id 0) — matching SharpEmu's
/// `availableEntries = 1` fallback when there is no metadata.
fn hle_get_chunk_id(ctx: &HleContext, args: &[u64]) -> u64 {
    const PLAYGO_ERROR_BAD_POINTER: u64 = 0x80B2_000A;
    const PLAYGO_ERROR_BAD_SIZE: u64 = 0x80B2_000B;

    let handle = args.first().copied().unwrap_or(0);
    let out_chunk_id_list = args.get(1).copied().unwrap_or(0);
    let number_of_entries = args.get(2).copied().unwrap_or(0);
    let out_entries = args.get(3).copied().unwrap_or(0);
    debug!(
        "scePlayGoGetChunkId(handle={handle}, outChunkIdList={out_chunk_id_list:#x}, numberOfEntries={number_of_entries}, outEntries={out_entries:#x})"
    );
    if out_entries == 0 {
        return PLAYGO_ERROR_BAD_POINTER;
    }
    if out_chunk_id_list != 0 && number_of_entries == 0 {
        return PLAYGO_ERROR_BAD_SIZE;
    }

    let available: u64 = 1; // no PlayGo metadata parsed yet — one present chunk
    // `outEntries` is a uint32_t*. Writing a u64 here corrupts the adjacent
    // four bytes of the caller's stack (SharpEmu's implementation also writes
    // exactly UInt32).
    let write = |value: u32| ctx.mem.write(out_entries, &value.to_le_bytes());
    if out_chunk_id_list == 0 {
        if !write(available as u32) {
            warn!("scePlayGoGetChunkId: outEntries {out_entries:#x} not writable");
            return PLAYGO_ERROR_BAD_POINTER;
        }
        return SCE_OK;
    }
    let to_write = number_of_entries.min(available) as usize;
    let ids = vec![0u16; to_write];
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    if !ctx.mem.write(out_chunk_id_list, &bytes) || !write(to_write as u32) {
        warn!("scePlayGoGetChunkId: out buffers not writable");
        return PLAYGO_ERROR_BAD_POINTER;
    }
    SCE_OK
}

/// PS5 `scePlayGoGetSupportedOptionalChunk(handle, outIds, outEntries)`.
/// Avatar first calls it in count-only mode (`outIds == NULL`). Raeen has no
/// parsed optional-chunk metadata, so report an empty supported set without
/// manufacturing chunk ids.
fn hle_get_supported_optional_chunk(ctx: &HleContext, args: &[u64]) -> u64 {
    const PLAYGO_ERROR_BAD_POINTER: u64 = 0x80B2_000A;
    let handle = args.first().copied().unwrap_or(0);
    let out_ids = args.get(1).copied().unwrap_or(0);
    let out_entries = args.get(2).copied().unwrap_or(0);
    debug!(
        "scePlayGoGetSupportedOptionalChunk(handle={handle}, outIds={out_ids:#x}, outEntries={out_entries:#x})"
    );
    if out_entries == 0 {
        return PLAYGO_ERROR_BAD_POINTER;
    }
    if !ctx.mem.write(out_entries, &0u32.to_le_bytes()) {
        warn!("scePlayGoGetSupportedOptionalChunk: outEntries {out_entries:#x} not writable");
        return PLAYGO_ERROR_BAD_POINTER;
    }
    SCE_OK
}

/// `scePlayGoGetProgress(handle, chunkIds, num, ScePlayGoProgress
/// *outProgress)`: writes `progressValue == totalValue` (both 0 — nothing to
/// install), i.e. "install complete", matching SharpEmu.
fn hle_get_progress(ctx: &HleContext, args: &[u64]) -> u64 {
    let out_progress = args.get(3).copied().unwrap_or(0);
    debug!("scePlayGoGetProgress(outProgress={out_progress:#x})");
    if out_progress != 0 {
        // ScePlayGoProgress { uint64_t progressValue; uint64_t totalValue; }
        let buf = [0u8; 16];
        if !ctx.mem.write(out_progress, &buf) {
            warn!("scePlayGoGetProgress: outProgress {out_progress:#x} not writable");
        }
    }
    SCE_OK
}

/// `scePlayGoGetToDoList(handle, outTodoList, numberOfEntries, uint32_t
/// *outEntries)`: reports an empty to-do list (`*outEntries = 0`) — nothing
/// left to download.
fn hle_get_todo_list(ctx: &HleContext, args: &[u64]) -> u64 {
    let out_entries = args.get(3).copied().unwrap_or(0);
    debug!("scePlayGoGetToDoList(outEntries={out_entries:#x})");
    if out_entries != 0 && !ctx.mem.write(out_entries, &0u32.to_le_bytes()) {
        warn!("scePlayGoGetToDoList: outEntries {out_entries:#x} not writable");
    }
    SCE_OK
}

/// `scePlayGoGetLanguageMask(handle, uint64_t *outMask)`: writes `0` (no
/// language filtering) and returns OK.
fn hle_get_language_mask(ctx: &HleContext, args: &[u64]) -> u64 {
    let out_mask = args.get(1).copied().unwrap_or(0);
    debug!("scePlayGoGetLanguageMask(outMask={out_mask:#x})");
    if out_mask != 0 && !ctx.mem.write(out_mask, &0u64.to_le_bytes()) {
        warn!("scePlayGoGetLanguageMask: outMask {out_mask:#x} not writable");
    }
    SCE_OK
}

fn hle_playgo_dialog_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    PLAYGO_DIALOG_STATUS.store(DIALOG_STATUS_NONE, Ordering::Relaxed);
    SCE_OK
}

fn hle_playgo_dialog_open(_ctx: &HleContext, args: &[u64]) -> u64 {
    const COMMON_DIALOG_ERROR_ARG_NULL: u64 = 0x80B8_000D;
    if args.first().copied().unwrap_or(0) == 0 {
        return COMMON_DIALOG_ERROR_ARG_NULL;
    }
    PLAYGO_DIALOG_STATUS.store(DIALOG_STATUS_NONE, Ordering::Relaxed);
    SCE_OK
}

fn hle_playgo_dialog_close(_ctx: &HleContext, _args: &[u64]) -> u64 {
    PLAYGO_DIALOG_STATUS.store(DIALOG_STATUS_NONE, Ordering::Relaxed);
    SCE_OK
}

fn hle_playgo_dialog_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    PLAYGO_DIALOG_STATUS.load(Ordering::Relaxed) as u32 as u64
}

fn hle_playgo_dialog_get_result(ctx: &HleContext, args: &[u64]) -> u64 {
    const COMMON_DIALOG_ERROR_ARG_NULL: u64 = 0x80B8_000D;
    let result = args.first().copied().unwrap_or(0);
    if result == 0 {
        return COMMON_DIALOG_ERROR_ARG_NULL;
    }
    let mut bytes = [0u8; 0x28];
    // shadPS4's compatibility result uses value 3 to let titles proceed.
    bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
    if !ctx.mem.write(result, &bytes) {
        return COMMON_DIALOG_ERROR_ARG_NULL;
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn get_locus_reports_every_chunk_local_fast() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // handle=1, chunkIds=0x100, num=3, outLoci=0x200
        assert_eq!(hle_get_locus(&ctx, &[1, 0x100, 3, 0x200]), SCE_OK);
        let mut loci = [0u8; 3];
        assert!(mem.read(0x200, &mut loci));
        assert_eq!(loci, [LOCUS_LOCAL_FAST; 3], "all chunks must be LOCAL_FAST");
    }

    #[test]
    fn open_writes_handle_and_todo_list_is_empty() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_open(&ctx, &[0x100, 0]), SCE_OK);
        let mut h = [0u8; 4];
        assert!(mem.read(0x100, &mut h));
        assert_eq!(u32::from_le_bytes(h), PLAYGO_HANDLE);

        // to-do list empty: *outEntries (arg3) == 0
        assert!(mem.write(0x300, &0xFFFF_FFFFu32.to_le_bytes()));
        assert_eq!(hle_get_todo_list(&ctx, &[1, 0x200, 4, 0x300]), SCE_OK);
        let mut e = [0u8; 4];
        assert!(mem.read(0x300, &mut e));
        assert_eq!(u32::from_le_bytes(e), 0, "nothing left to download");
    }

    #[test]
    fn get_chunk_id_count_only_and_list_modes() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Count-only: outChunkIdList=0 → *outEntries = 1 (the present chunk).
        assert_eq!(hle_get_chunk_id(&ctx, &[1, 0, 0, 0x400]), SCE_OK);
        let mut e = [0u8; 4];
        assert!(mem.read(0x400, &mut e));
        assert_eq!(u32::from_le_bytes(e), 1);

        assert!(HleRegistry::new().is_implemented("libScePlayGo", "scePlayGoGetInstallChunkId"));

        // List mode: one uint16 id written, *outEntries = 1.
        assert_eq!(hle_get_chunk_id(&ctx, &[1, 0x500, 4, 0x400]), SCE_OK);
        let mut id = [0u8; 2];
        assert!(mem.read(0x500, &mut id));
        assert!(mem.read(0x400, &mut e));
        assert_eq!(u32::from_le_bytes(e), 1);

        // outEntries == 0 → BAD_POINTER; list with 0 entries → BAD_SIZE.
        assert_eq!(hle_get_chunk_id(&ctx, &[1, 0, 0, 0]), 0x80B2_000A);
        assert_eq!(hle_get_chunk_id(&ctx, &[1, 0x500, 0, 0x400]), 0x80B2_000B);
    }

    #[test]
    fn supported_optional_chunk_count_is_empty_and_bounded() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x400, &u32::MAX.to_le_bytes()));
        assert_eq!(
            hle_get_supported_optional_chunk(&ctx, &[1, 0, 0x400]),
            SCE_OK
        );
        let mut count = [0u8; 4];
        assert!(mem.read(0x400, &mut count));
        assert_eq!(u32::from_le_bytes(count), 0);
    }

    #[test]
    fn get_progress_reports_complete() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x40, &[0xEE; 16]));
        assert_eq!(hle_get_progress(&ctx, &[1, 0x100, 3, 0x40]), SCE_OK);
        let mut p = [0u8; 16];
        assert!(mem.read(0x40, &mut p));
        let progress = u64::from_le_bytes(p[0..8].try_into().unwrap());
        let total = u64::from_le_bytes(p[8..16].try_into().unwrap());
        assert_eq!(progress, total, "progress == total means install complete");
    }

    #[test]
    fn playgo_dialog_family_is_headless_and_returns_proceed_result() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_playgo_dialog_initialize(&ctx, &[]), SCE_OK);
        assert_eq!(hle_playgo_dialog_open(&ctx, &[0x100]), SCE_OK);
        assert_eq!(hle_playgo_dialog_status(&ctx, &[]), 0);
        assert_eq!(hle_playgo_dialog_get_result(&ctx, &[0x200]), SCE_OK);
        let mut result = [0u8; 8];
        assert!(mem.read(0x200, &mut result));
        assert_eq!(u32::from_le_bytes(result[4..8].try_into().unwrap()), 3);
        assert_eq!(hle_playgo_dialog_open(&ctx, &[0]), 0x80B8_000D);
    }
}
