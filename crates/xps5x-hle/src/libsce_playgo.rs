//! HLE libScePlayGo — content install / streaming status.
//!
//! PlayGo lets a title query which "chunks" of its content are locally
//! available and how far a background install has progressed. XPS5X runs
//! against fully-present local content (the whole title is on disk), so the
//! honest, correct answer to every query is "all chunks local, nothing left
//! to download, install complete" — which lets a title skip its
//! download-gating and proceed straight to gameplay. Values and the
//! all-`LOCAL_FAST` behavior are cross-checked against SharpEmu's
//! `PlayGoExports`.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `ScePlayGoLocus::LOCAL_FAST` (3) — the chunk is present on fast local
/// storage. (`NOT_DOWNLOADED = 0`, `LOCAL_SLOW = 2`.)
const LOCUS_LOCAL_FAST: u8 = 3;
/// A fixed PlayGo handle handed back by `scePlayGoOpen`.
const PLAYGO_HANDLE: u32 = 1;
/// Cap on how many chunk loci one `scePlayGoGetLocus` will write, bounding
/// the host staging buffer against a wild `numberOfEntries`.
const MAX_ENTRIES: u64 = 1 << 16;

/// Register libScePlayGo HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libScePlayGo", "scePlayGoInitialize", hle_initialize);
    registry.register("libScePlayGo", "scePlayGoTerminate", hle_ok);
    registry.register("libScePlayGo", "scePlayGoOpen", hle_open);
    registry.register("libScePlayGo", "scePlayGoClose", hle_ok);
    registry.register("libScePlayGo", "scePlayGoGetLocus", hle_get_locus);
    registry.register("libScePlayGo", "scePlayGoGetProgress", hle_get_progress);
    registry.register("libScePlayGo", "scePlayGoGetToDoList", hle_get_todo_list);
    registry.register("libScePlayGo", "scePlayGoGetEta", hle_ok);
    registry.register(
        "libScePlayGo",
        "scePlayGoGetLanguageMask",
        hle_get_language_mask,
    );
    registry.register("libScePlayGo", "scePlayGoGetInstallSpeed", hle_ok);
    registry.register("libScePlayGo", "scePlayGoSetInstallSpeed", hle_ok);
    registry.register("libScePlayGo", "scePlayGoSetTodoList", hle_ok);
    registry.register("libScePlayGo", "scePlayGoPrefetch", hle_ok);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_ctx, GuestMemory};

    #[test]
    fn get_locus_reports_every_chunk_local_fast() {
        let kernel = xps5x_kernel::OrbisKernel::new();
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
        let kernel = xps5x_kernel::OrbisKernel::new();
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
    fn get_progress_reports_complete() {
        let kernel = xps5x_kernel::OrbisKernel::new();
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
}
