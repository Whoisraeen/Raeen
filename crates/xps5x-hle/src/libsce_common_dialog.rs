//! HLE libSceCommonDialog / libSceMsgDialog — system dialogs.
//!
//! Games show a message dialog (`sceMsgDialogOpen`) and then poll its status
//! (`sceMsgDialogUpdateStatus`/`GetStatus`) each frame until it reports
//! `FINISHED`, read the result, and close it. XPS5X has no host popup to
//! display, so — like SharpEmu — a dialog **completes immediately**: `Open`
//! moves the status straight to `FINISHED` with an "OK button" result, so a
//! title's dialog loop finishes on the next poll instead of hanging forever.
//! Status values and the immediate-finish behavior are cross-checked against
//! SharpEmu's `MsgDialogExports`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};
use tracing::debug;

/// `SCE_OK`.
const SCE_OK: u64 = 0;

/// `SceCommonDialogStatus` (shared by all common dialogs).
const STATUS_NONE: i32 = 0;
const STATUS_INITIALIZED: i32 = 1;
#[allow(dead_code)] // documented for parity; XPS5X never lingers in RUNNING
const STATUS_RUNNING: i32 = 2;
const STATUS_FINISHED: i32 = 3;

/// `SCE_MSG_DIALOG_BUTTON_ID_OK` — the button a completed dialog reports.
const BUTTON_ID_OK: i32 = 1;

/// The single message-dialog's status (one dialog at a time, the real API's
/// constraint). Driven by Open (→ FINISHED) / Close (→ NONE).
static MSG_STATUS: AtomicI32 = AtomicI32::new(STATUS_NONE);

/// Register libSceCommonDialog + libSceMsgDialog HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceCommonDialog", "sceCommonDialogInitialize", hle_ok);
    registry.register("libSceCommonDialog", "sceCommonDialogIsUsed", hle_is_used);

    registry.register(
        "libSceMsgDialog",
        "sceMsgDialogInitialize",
        hle_msg_initialize,
    );
    registry.register("libSceMsgDialog", "sceMsgDialogOpen", hle_msg_open);
    registry.register(
        "libSceMsgDialog",
        "sceMsgDialogUpdateStatus",
        hle_msg_status,
    );
    registry.register("libSceMsgDialog", "sceMsgDialogGetStatus", hle_msg_status);
    registry.register(
        "libSceMsgDialog",
        "sceMsgDialogGetResult",
        hle_msg_get_result,
    );
    registry.register("libSceMsgDialog", "sceMsgDialogClose", hle_msg_close);
    registry.register(
        "libSceMsgDialog",
        "sceMsgDialogTerminate",
        hle_msg_terminate,
    );
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `sceCommonDialogIsUsed()`: no dialog is currently on screen — `false`.
fn hle_is_used(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0 // false
}

fn hle_msg_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    MSG_STATUS.store(STATUS_INITIALIZED, Ordering::Relaxed);
    debug!("sceMsgDialogInitialize()");
    SCE_OK
}

/// `sceMsgDialogOpen(param)`: with no host popup to display, complete the
/// dialog immediately — status jumps to `FINISHED` so the title's next poll
/// finishes.
fn hle_msg_open(_ctx: &HleContext, _args: &[u64]) -> u64 {
    MSG_STATUS.store(STATUS_FINISHED, Ordering::Relaxed);
    debug!("sceMsgDialogOpen() -> completes immediately (FINISHED)");
    SCE_OK
}

/// `sceMsgDialogUpdateStatus()` / `sceMsgDialogGetStatus()`: return the
/// current status (`FINISHED` once opened).
fn hle_msg_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    MSG_STATUS.load(Ordering::Relaxed) as u32 as u64
}

/// `sceMsgDialogGetResult(SceMsgDialogResult *result)`: writes a completed
/// result (`buttonId = OK`). Layout: `int32 mode; int32 result; int32
/// buttonId; ...` — mode/result 0 (success), buttonId = OK.
fn hle_msg_get_result(ctx: &HleContext, args: &[u64]) -> u64 {
    let result_ptr = args.first().copied().unwrap_or(0);
    debug!("sceMsgDialogGetResult(result={result_ptr:#x})");
    if result_ptr != 0 {
        let mut buf = [0u8; 12];
        // mode@0 = 0, result@4 = 0 (SCE_OK), buttonId@8 = OK.
        buf[8..12].copy_from_slice(&BUTTON_ID_OK.to_le_bytes());
        let _ = ctx.mem.write(result_ptr, &buf);
    }
    SCE_OK
}

fn hle_msg_close(_ctx: &HleContext, _args: &[u64]) -> u64 {
    MSG_STATUS.store(STATUS_NONE, Ordering::Relaxed);
    debug!("sceMsgDialogClose()");
    SCE_OK
}

fn hle_msg_terminate(_ctx: &HleContext, _args: &[u64]) -> u64 {
    MSG_STATUS.store(STATUS_NONE, Ordering::Relaxed);
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_ctx, GuestMemory};

    #[test]
    fn msg_dialog_open_finishes_immediately_and_reports_ok() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_msg_initialize(&ctx, &[]), SCE_OK);
        assert_eq!(hle_msg_open(&ctx, &[0x100]), SCE_OK);
        // The title's poll loop sees FINISHED right away (no hang).
        assert_eq!(hle_msg_status(&ctx, &[]) as i32, STATUS_FINISHED);

        // Result reports the OK button.
        assert_eq!(hle_msg_get_result(&ctx, &[0x200]), SCE_OK);
        let mut r = [0u8; 12];
        assert!(mem.read(0x200, &mut r));
        assert_eq!(
            i32::from_le_bytes(r[8..12].try_into().unwrap()),
            BUTTON_ID_OK
        );

        // Close returns to NONE.
        assert_eq!(hle_msg_close(&ctx, &[]), SCE_OK);
        assert_eq!(hle_msg_status(&ctx, &[]) as i32, STATUS_NONE);
    }

    #[test]
    fn common_dialog_reports_not_used() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_is_used(&ctx, &[]), 0);
    }
}
