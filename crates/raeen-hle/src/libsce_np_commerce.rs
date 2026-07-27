//! HLE libSceNpCommerce — the PlayStation Store commerce dialog + store icon.
//!
//! Raeen has no store backend and no host overlay, so the commerce dialog
//! **completes immediately as canceled** — the outcome a player produces by
//! backing out of the store without buying — and the PS Store icon
//! show/hide requests are accepted and dropped (there is no overlay to draw
//! an icon on). **No purchase, entitlement, or catalog data is ever
//! fabricated.**
//!
//! Same immediate-completion model as `libsce_signin_dialog.rs`, using the
//! shared `SceCommonDialogStatus` enum. The `SceNpCommerceDialogResult`
//! leading `s32` is written as `0`; user-canceled titles read the canceled
//! disposition from their own param/result conventions and simply resume.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};
use tracing::debug;

const SCE_OK: u64 = 0;

/// Shared `SceCommonDialogStatus`.
const STATUS_NONE: i32 = 0;
const STATUS_INITIALIZED: i32 = 1;
const STATUS_FINISHED: i32 = 3;

static STATUS: AtomicI32 = AtomicI32::new(STATUS_NONE);

/// Register the libSceNpCommerce functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceNpCommerce",
        "sceNpCommerceDialogInitialize",
        hle_initialize,
    );
    registry.register("libSceNpCommerce", "sceNpCommerceDialogOpen", hle_open);
    registry.register(
        "libSceNpCommerce",
        "sceNpCommerceDialogUpdateStatus",
        hle_status,
    );
    registry.register(
        "libSceNpCommerce",
        "sceNpCommerceDialogGetResult",
        hle_get_result,
    );
    registry.register(
        "libSceNpCommerce",
        "sceNpCommerceDialogTerminate",
        hle_terminate,
    );
    // The persistent PS Store icon overlay: accepted and dropped — Raeen has
    // no system overlay plane to draw it on, and the calls carry no answer
    // the title depends on.
    registry.register("libSceNpCommerce", "sceNpCommerceShowPsStoreIcon", hle_icon);
    registry.register("libSceNpCommerce", "sceNpCommerceHidePsStoreIcon", hle_icon);
}

fn hle_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    STATUS.store(STATUS_INITIALIZED, Ordering::Relaxed);
    debug!("sceNpCommerceDialogInitialize()");
    SCE_OK
}

/// `sceNpCommerceDialogOpen(param)`: no store — the dialog completes
/// immediately (player backed out without buying), so the title resumes on
/// its next poll.
fn hle_open(_ctx: &HleContext, _args: &[u64]) -> u64 {
    STATUS.store(STATUS_FINISHED, Ordering::Relaxed);
    debug!("sceNpCommerceDialogOpen() -> completes immediately (FINISHED, canceled)");
    SCE_OK
}

fn hle_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    STATUS.load(Ordering::Relaxed) as u32 as u64
}

/// `sceNpCommerceDialogGetResult(result*)`: leading `s32 0` — the canceled
/// close; nothing was purchased and nothing is granted.
fn hle_get_result(ctx: &HleContext, args: &[u64]) -> u64 {
    let result = args.first().copied().unwrap_or(0);
    if result != 0 {
        let _ = ctx.mem.write(result, &0i32.to_le_bytes());
    }
    SCE_OK
}

fn hle_terminate(_ctx: &HleContext, _args: &[u64]) -> u64 {
    STATUS.store(STATUS_NONE, Ordering::Relaxed);
    SCE_OK
}

/// `sceNpCommerceShowPsStoreIcon(userId, pos)` / `Hide...`: no overlay.
fn hle_icon(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    /// The commerce dialog completes immediately as a canceled store visit —
    /// no purchase is fabricated and the title's poll loop terminates.
    #[test]
    fn commerce_dialog_completes_immediately_without_purchase() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_initialize(&ctx, &[]), SCE_OK);
        assert_eq!(hle_status(&ctx, &[]) as i32, STATUS_INITIALIZED);
        assert_eq!(hle_open(&ctx, &[0x100]), SCE_OK);
        assert_eq!(hle_status(&ctx, &[]) as i32, STATUS_FINISHED);
        assert_eq!(hle_get_result(&ctx, &[0x200]), SCE_OK);
        assert_eq!(hle_icon(&ctx, &[0x1000_0000, 0]), SCE_OK);
        assert_eq!(hle_terminate(&ctx, &[]), SCE_OK);
        assert_eq!(hle_status(&ctx, &[]) as i32, STATUS_NONE);

        let registry = HleRegistry::new();
        for name in [
            "sceNpCommerceDialogInitialize",
            "sceNpCommerceDialogOpen",
            "sceNpCommerceDialogUpdateStatus",
            "sceNpCommerceDialogGetResult",
            "sceNpCommerceDialogTerminate",
            "sceNpCommerceShowPsStoreIcon",
            "sceNpCommerceHidePsStoreIcon",
        ] {
            assert!(
                registry.is_implemented("libSceNpCommerce", name),
                "{name} must be registered"
            );
        }
    }
}
