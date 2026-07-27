//! HLE libSceWebBrowserDialog — the system web-browser overlay dialog.
//!
//! Raeen has no host browser overlay, so the dialog **completes immediately**:
//! `Open` jumps the shared common-dialog status to `FINISHED` and `GetResult`
//! writes a leading `s32 0` — the "dialog closed" outcome a player produces
//! by immediately backing out. A title gating on the dialog's completion
//! proceeds on its next poll; no web content is ever fetched or displayed.
//!
//! Same immediate-completion model as `libsce_signin_dialog.rs`, using the
//! shared `SceCommonDialogStatus` enum (cross-checked against shadPS4's
//! GPL-2.0 `web_browser_dialog` — re-derived, not ported).

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};
use tracing::debug;

const SCE_OK: u64 = 0;

/// Shared `SceCommonDialogStatus`.
const STATUS_NONE: i32 = 0;
const STATUS_INITIALIZED: i32 = 1;
const STATUS_FINISHED: i32 = 3;

static STATUS: AtomicI32 = AtomicI32::new(STATUS_NONE);

/// Register the libSceWebBrowserDialog functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceWebBrowserDialog",
        "sceWebBrowserDialogInitialize",
        hle_initialize,
    );
    registry.register(
        "libSceWebBrowserDialog",
        "sceWebBrowserDialogOpen",
        hle_open,
    );
    registry.register(
        "libSceWebBrowserDialog",
        "sceWebBrowserDialogUpdateStatus",
        hle_status,
    );
    registry.register(
        "libSceWebBrowserDialog",
        "sceWebBrowserDialogGetResult",
        hle_get_result,
    );
    registry.register(
        "libSceWebBrowserDialog",
        "sceWebBrowserDialogClose",
        hle_close,
    );
    registry.register(
        "libSceWebBrowserDialog",
        "sceWebBrowserDialogTerminate",
        hle_close,
    );
}

fn hle_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    STATUS.store(STATUS_INITIALIZED, Ordering::Relaxed);
    debug!("sceWebBrowserDialogInitialize()");
    SCE_OK
}

/// `sceWebBrowserDialogOpen(param)`: no host browser — complete immediately
/// so the title's poll loop finishes instead of waiting on an overlay that
/// can never appear.
fn hle_open(_ctx: &HleContext, _args: &[u64]) -> u64 {
    STATUS.store(STATUS_FINISHED, Ordering::Relaxed);
    debug!("sceWebBrowserDialogOpen() -> completes immediately (FINISHED, closed)");
    SCE_OK
}

/// `UpdateStatus`/`GetStatus`: the shared common-dialog status in `eax`.
fn hle_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    STATUS.load(Ordering::Relaxed) as u32 as u64
}

/// `sceWebBrowserDialogGetResult(result*)`: write the leading `s32 0`
/// ("closed" outcome); the rest of the undocumented struct is the caller's.
fn hle_get_result(ctx: &HleContext, args: &[u64]) -> u64 {
    let result = args.first().copied().unwrap_or(0);
    if result != 0 {
        let _ = ctx.mem.write(result, &0i32.to_le_bytes());
    }
    SCE_OK
}

fn hle_close(_ctx: &HleContext, _args: &[u64]) -> u64 {
    STATUS.store(STATUS_NONE, Ordering::Relaxed);
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    /// Initialize -> INITIALIZED, Open -> FINISHED immediately, result reads
    /// closed/success, Terminate -> NONE. A title gating on the dialog never
    /// hangs.
    #[test]
    fn web_browser_dialog_completes_immediately() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_initialize(&ctx, &[]), SCE_OK);
        assert_eq!(hle_status(&ctx, &[]) as i32, STATUS_INITIALIZED);
        assert_eq!(hle_open(&ctx, &[0x100]), SCE_OK);
        assert_eq!(hle_status(&ctx, &[]) as i32, STATUS_FINISHED);

        assert!(mem.write(0x200, &0xFFFF_FFFFu32.to_le_bytes()));
        assert_eq!(hle_get_result(&ctx, &[0x200]), SCE_OK);
        let mut r = [0u8; 4];
        assert!(mem.read(0x200, &mut r));
        assert_eq!(i32::from_le_bytes(r), 0);
        assert_eq!(hle_get_result(&ctx, &[0]), SCE_OK, "null tolerated");

        assert_eq!(hle_close(&ctx, &[]), SCE_OK);
        assert_eq!(hle_status(&ctx, &[]) as i32, STATUS_NONE);

        let registry = HleRegistry::new();
        for name in [
            "sceWebBrowserDialogInitialize",
            "sceWebBrowserDialogOpen",
            "sceWebBrowserDialogUpdateStatus",
            "sceWebBrowserDialogGetResult",
            "sceWebBrowserDialogClose",
            "sceWebBrowserDialogTerminate",
        ] {
            assert!(
                registry.is_implemented("libSceWebBrowserDialog", name),
                "{name} must be registered"
            );
        }
    }
}
