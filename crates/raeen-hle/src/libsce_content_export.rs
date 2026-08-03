//! HLE libSceContentExport — screenshot/video export subsystem init.
//!
//! ASTRO.BOT's `ShareService` (`ShareService.cpp:102`) calls
//! `sceContentExportInit2()` and does `assert(rc == 0)` on the result. With no
//! `libSceContentExport` provider the call was skipped, `eax` held
//! garbage-nonzero, and the title hit `int 0x41` (its assert trap) and died.
//!
//! `sceContentExportInit2` validates the caller's `OrbisContentExportInitParam`
//! and returns `ORBIS_OK`. Actual export (saving a screenshot/clip) is stubbed
//! — the title only needs init to succeed to proceed past its Share setup.
//! Ported from shadPS4 (GPL-2.0) `core/libraries/content_export`.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::{HleContext, HleRegistry};
use tracing::debug;

/// `ORBIS_OK`.
const ORBIS_OK: u64 = 0;
/// `ORBIS_CONTENT_EXPORT_ERROR_NOINIT`.
const ERROR_NOINIT: u64 = 0x809D_3004;
/// `ORBIS_CONTENT_EXPORT_ERROR_MULTIPLEINIT`.
const ERROR_MULTIPLEINIT: u64 = 0x809D_3005;
/// `ORBIS_CONTENT_EXPORT_ERROR_INVALDPARAM`.
const ERROR_INVALDPARAM: u64 = 0x809D_3016;

/// Process-wide "content export initialized" flag (shadPS4's `g_is_initialized`).
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Read a little-endian `u64` from guest memory, or `None` on a bad read.
fn read_u64(ctx: &HleContext, addr: u64) -> Option<u64> {
    let mut b = [0u8; 8];
    ctx.mem.read(addr, &mut b).then(|| u64::from_le_bytes(b))
}

/// Shared body of `sceContentExportInit`/`Init2` — `version` 0 vs 1 differ only
/// in the extra reserved/bufsize checks. Faithful to shadPS4's
/// `_sceContentExportInit`.
///
/// `OrbisContentExportInitParam` layout:
/// `0x00 mallocfunc | 0x08 freefunc | 0x10 userdata | 0x18 bufsize |`
/// `0x20 reserved0 | 0x28 reserved1`.
fn content_export_init(ctx: &HleContext, init_param: u64, version: u8) -> u64 {
    if INITIALIZED.load(Ordering::Acquire) {
        return ERROR_MULTIPLEINIT;
    }
    if init_param == 0 {
        return ERROR_INVALDPARAM;
    }
    let (Some(mallocfunc), Some(freefunc)) =
        (read_u64(ctx, init_param), read_u64(ctx, init_param + 0x08))
    else {
        return ERROR_INVALDPARAM;
    };
    if mallocfunc == 0 || freefunc == 0 {
        return ERROR_INVALDPARAM;
    }
    if version == 1 {
        let bufsize = read_u64(ctx, init_param + 0x18).unwrap_or(0);
        let reserved0 = read_u64(ctx, init_param + 0x20).unwrap_or(0);
        let reserved1 = read_u64(ctx, init_param + 0x28).unwrap_or(0);
        if reserved0 != 0 || reserved1 != 0 || (bufsize != 0 && bufsize < 0x100) {
            return ERROR_INVALDPARAM;
        }
    }
    INITIALIZED.store(true, Ordering::Release);
    debug!("sceContentExportInit(v{version}): param={init_param:#x} -> OK");
    ORBIS_OK
}

fn hle_init(ctx: &HleContext, args: &[u64]) -> u64 {
    content_export_init(ctx, args.first().copied().unwrap_or(0), 0)
}

fn hle_init2(ctx: &HleContext, args: &[u64]) -> u64 {
    content_export_init(ctx, args.first().copied().unwrap_or(0), 1)
}

fn hle_term(_ctx: &HleContext, _args: &[u64]) -> u64 {
    if !INITIALIZED.swap(false, Ordering::AcqRel) {
        return ERROR_NOINIT;
    }
    ORBIS_OK
}

/// Export operations (start/finish/from-data): no local media store, so report
/// success without producing a file — the title's Share flow expects OK.
fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    ORBIS_OK
}

/// Register libSceContentExport HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceContentExport", "sceContentExportInit", hle_init);
    registry.register("libSceContentExport", "sceContentExportInit2", hle_init2);
    registry.register("libSceContentExport", "sceContentExportTerm", hle_term);
    // The export surface reports success while producing nothing: no session,
    // no file, no media-gallery entry. The title only observes the return
    // code, but the user observes the missing export — so every one of these
    // is named for the coverage report instead of passing as working.
    registry.register_incomplete(
        "libSceContentExport",
        "sceContentExportStart",
        hle_ok,
        "reports success but no export session is created",
    );
    registry.register_incomplete(
        "libSceContentExport",
        "sceContentExportFinish",
        hle_ok,
        "reports success for an export session that was never created",
    );
    registry.register_incomplete(
        "libSceContentExport",
        "sceContentExportFromData",
        hle_ok,
        "reports success but nothing is exported to any host media store",
    );
    // File-sourced exports (measured GTA V imports): same accept-and-drop
    // model as FromData above — the "export to the console media gallery"
    // side effect has no host equivalent.
    registry.register_incomplete(
        "libSceContentExport",
        "sceContentExportFromFile",
        hle_ok,
        "reports success but the source file is not exported anywhere",
    );
    registry.register_incomplete(
        "libSceContentExport",
        "sceContentExportFromFileWithThumbnail",
        hle_ok,
        "reports success but neither file nor thumbnail is exported anywhere",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn init2_validates_and_returns_ok_then_multipleinit() {
        INITIALIZED.store(false, Ordering::Release);
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let param = 0x200u64;
        // Null malloc/free -> INVALDPARAM.
        assert_eq!(hle_init2(&ctx, &[param]), ERROR_INVALDPARAM);
        // Fill mallocfunc + freefunc; reserved/bufsize zero -> OK.
        assert!(mem.write(param, &0x1234u64.to_le_bytes()));
        assert!(mem.write(param + 0x08, &0x5678u64.to_le_bytes()));
        assert_eq!(hle_init2(&ctx, &[param]), ORBIS_OK);
        // Second init without term -> MULTIPLEINIT.
        assert_eq!(hle_init2(&ctx, &[param]), ERROR_MULTIPLEINIT);
        // Term clears it; re-init succeeds.
        assert_eq!(hle_term(&ctx, &[]), ORBIS_OK);
        assert_eq!(hle_init2(&ctx, &[param]), ORBIS_OK);
        INITIALIZED.store(false, Ordering::Release);
    }
}
