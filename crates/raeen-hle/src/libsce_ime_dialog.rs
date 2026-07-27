//! HLE libSceImeDialog — the on-screen-keyboard text-entry dialog.
//!
//! Raeen has no host IME UI to display, so the dialog **completes
//! immediately as user-canceled**: `sceImeDialogInit` jumps the status
//! straight to `FINISHED` with end-status `USER_CANCELED`, the title's poll
//! loop finishes on its next frame, `GetResult` reports the cancel, and the
//! title keeps whatever default text it had — exactly what a player backing
//! out of the keyboard produces on real hardware. Nothing fabricates typed
//! input.
//!
//! Status enum (`None`=0/`Running`=1/`Finished`=2 — note this is NOT the
//! common-dialog enum), end-status (`Ok`=0/`UserCanceled`=1/`Aborted`=2) and
//! the `ORBIS_IME_ERROR_INVALID_ADDRESS` value are cross-checked against
//! shadPS4's GPL-2.0 `ime_dialog.h`/`ime_common.h` — re-derived, not ported.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};
use tracing::debug;

const OK: u64 = 0;
/// `ORBIS_IME_ERROR_INVALID_ADDRESS` (shadPS4 `ime_error.h`).
const IME_ERROR_INVALID_ADDRESS: u64 = 0x80BC_0031;

/// `OrbisImeDialogStatus`.
const STATUS_NONE: i32 = 0;
#[allow(dead_code)] // documented for parity; the dialog never lingers in RUNNING
const STATUS_RUNNING: i32 = 1;
const STATUS_FINISHED: i32 = 2;

/// `OrbisImeDialogEndStatus`.
const END_STATUS_USER_CANCELED: u32 = 1;
const END_STATUS_ABORTED: u32 = 2;

static IME_STATUS: AtomicI32 = AtomicI32::new(STATUS_NONE);
static IME_END_STATUS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(END_STATUS_USER_CANCELED);

/// Register the libSceImeDialog functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceImeDialog", "sceImeDialogInit", hle_init);
    registry.register("libSceImeDialog", "sceImeDialogGetStatus", hle_get_status);
    registry.register("libSceImeDialog", "sceImeDialogGetResult", hle_get_result);
    registry.register("libSceImeDialog", "sceImeDialogAbort", hle_abort);
    registry.register("libSceImeDialog", "sceImeDialogTerm", hle_term);
    registry.register(
        "libSceImeDialog",
        "sceImeDialogGetPanelSizeExtended",
        hle_get_panel_size_extended,
    );
}

/// `sceImeDialogInit(param, extended)`: with no host keyboard the dialog
/// finishes immediately as user-canceled, so the title's poll loop completes
/// on its next frame instead of waiting for input that can never arrive.
fn hle_init(_ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) == 0 {
        return IME_ERROR_INVALID_ADDRESS;
    }
    IME_END_STATUS.store(END_STATUS_USER_CANCELED, Ordering::Relaxed);
    IME_STATUS.store(STATUS_FINISHED, Ordering::Relaxed);
    debug!("sceImeDialogInit() -> completes immediately (FINISHED, USER_CANCELED)");
    OK
}

/// `sceImeDialogGetStatus()`: the `OrbisImeDialogStatus` in `eax`.
fn hle_get_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    IME_STATUS.load(Ordering::Relaxed) as u32 as u64
}

/// `sceImeDialogGetResult(OrbisImeDialogResult *result)`: write the end
/// status (leading `u32`) — `USER_CANCELED`, or `ABORTED` after an abort. The
/// rest of the result struct is the caller's (no typed text exists to store).
fn hle_get_result(ctx: &HleContext, args: &[u64]) -> u64 {
    let result = args.first().copied().unwrap_or(0);
    if result == 0 {
        return IME_ERROR_INVALID_ADDRESS;
    }
    let end = IME_END_STATUS.load(Ordering::Relaxed);
    if !ctx.mem.write(result, &end.to_le_bytes()) {
        return IME_ERROR_INVALID_ADDRESS;
    }
    debug!("sceImeDialogGetResult -> endstatus {end}");
    OK
}

/// `sceImeDialogAbort()`: force-finish with `ABORTED`.
fn hle_abort(_ctx: &HleContext, _args: &[u64]) -> u64 {
    IME_END_STATUS.store(END_STATUS_ABORTED, Ordering::Relaxed);
    IME_STATUS.store(STATUS_FINISHED, Ordering::Relaxed);
    OK
}

/// `sceImeDialogTerm()`: back to `NONE`.
fn hle_term(_ctx: &HleContext, _args: &[u64]) -> u64 {
    IME_STATUS.store(STATUS_NONE, Ordering::Relaxed);
    OK
}

/// Nominal panel size reported to layout code (no host panel is rendered;
/// any non-degenerate size keeps title layout math finite).
const PANEL_WIDTH: u32 = 1920;
const PANEL_HEIGHT: u32 = 480;

/// `sceImeDialogGetPanelSizeExtended(param, extended, u32 *width, u32
/// *height)`: validate the pointers (shadPS4's `INVALID_ADDRESS` rule) and
/// report a nominal full-width keyboard panel.
fn hle_get_panel_size_extended(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    let width = args.get(2).copied().unwrap_or(0);
    let height = args.get(3).copied().unwrap_or(0);
    if param == 0 || width == 0 || height == 0 {
        return IME_ERROR_INVALID_ADDRESS;
    }
    if !ctx.mem.write(width, &PANEL_WIDTH.to_le_bytes())
        || !ctx.mem.write(height, &PANEL_HEIGHT.to_le_bytes())
    {
        return IME_ERROR_INVALID_ADDRESS;
    }
    OK
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

    /// The keyboard flow completes immediately as user-canceled: a title
    /// polling GetStatus sees FINISHED on its first poll, reads the cancel
    /// from GetResult, and proceeds with its default text — no hang, no
    /// fabricated input.
    #[test]
    fn ime_dialog_finishes_immediately_as_user_canceled() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_init(&ctx, &[0]), IME_ERROR_INVALID_ADDRESS);
        assert_eq!(hle_init(&ctx, &[0x100, 0]), OK);
        assert_eq!(hle_get_status(&ctx, &[]) as i32, STATUS_FINISHED);

        assert_eq!(hle_get_result(&ctx, &[0x200]), OK);
        let mut end = [0u8; 4];
        assert!(mem.read(0x200, &mut end));
        assert_eq!(u32::from_le_bytes(end), END_STATUS_USER_CANCELED);
        assert_eq!(hle_get_result(&ctx, &[0]), IME_ERROR_INVALID_ADDRESS);

        assert_eq!(hle_term(&ctx, &[]), OK);
        assert_eq!(hle_get_status(&ctx, &[]) as i32, STATUS_NONE);

        // Abort reports ABORTED through the result.
        assert_eq!(hle_init(&ctx, &[0x100, 0]), OK);
        assert_eq!(hle_abort(&ctx, &[]), OK);
        assert_eq!(hle_get_result(&ctx, &[0x200]), OK);
        assert!(mem.read(0x200, &mut end));
        assert_eq!(u32::from_le_bytes(end), END_STATUS_ABORTED);
    }

    #[test]
    fn panel_size_is_nominal_and_pointer_checked() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_get_panel_size_extended(&ctx, &[0x100, 0, 0x200, 0x210]),
            OK
        );
        let mut v = [0u8; 4];
        assert!(mem.read(0x200, &mut v));
        assert_eq!(u32::from_le_bytes(v), PANEL_WIDTH);
        assert!(mem.read(0x210, &mut v));
        assert_eq!(u32::from_le_bytes(v), PANEL_HEIGHT);
        assert_eq!(
            hle_get_panel_size_extended(&ctx, &[0, 0, 0x200, 0x210]),
            IME_ERROR_INVALID_ADDRESS
        );
        assert_eq!(
            hle_get_panel_size_extended(&ctx, &[0x100, 0, 0, 0x210]),
            IME_ERROR_INVALID_ADDRESS
        );

        let registry = HleRegistry::new();
        for name in [
            "sceImeDialogInit",
            "sceImeDialogGetStatus",
            "sceImeDialogGetResult",
            "sceImeDialogAbort",
            "sceImeDialogTerm",
            "sceImeDialogGetPanelSizeExtended",
        ] {
            assert!(
                registry.is_implemented("libSceImeDialog", name),
                "{name} must be registered"
            );
        }
    }
}
