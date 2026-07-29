//! HLE libSceSigninDialog — the PSN sign-in system dialog.
//!
//! A title that requires a signed-in user opens this dialog
//! (`sceSigninDialogOpen`) and then polls its status
//! (`sceSigninDialogUpdateStatus`/`GetStatus`) each frame until it reports
//! `FINISHED`, reads the result, and terminates it. Until the flow finishes,
//! the title parks in its boot/attract screen and does **not** proceed to open
//! the pad or start gameplay — so a missing `libSceSigninDialog` (every
//! `sceSigninDialog*` NID unresolved) can be why input never reaches the game.
//!
//! Raeen models a single always-signed-in local user and has no host sign-in
//! UI to display, so — exactly like shadPS4's `Libraries::SigninDialog` — the
//! dialog **completes immediately**: `Open` moves the status straight to
//! `FINISHED` with a success result, so the title's poll loop finishes on the
//! next frame and advances to gameplay.
//!
//! Status values are the shared `SceCommonDialogStatus`
//! (`NONE`/`INITIALIZED`/`RUNNING`/`FINISHED`), cross-checked against shadPS4's
//! `enum class Status` in `signindialog.h` (GPL-2.0) — re-implemented here, not
//! ported. Behavior mirrors the same reference. Function names match the Orbis
//! exports so the NID linker (`nid_of(name)`) resolves each import.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};
use tracing::debug;

/// `SCE_OK`.
const SCE_OK: u64 = 0;

/// `SceCommonDialogStatus` (shared by all common dialogs; identical to
/// shadPS4's `enum class Status`).
const STATUS_NONE: i32 = 0;
const STATUS_INITIALIZED: i32 = 1;
#[allow(dead_code)] // documented for parity; Raeen never lingers in RUNNING
const STATUS_RUNNING: i32 = 2;
const STATUS_FINISHED: i32 = 3;

/// The single sign-in dialog's status (one dialog at a time, the real API's
/// constraint). Driven by Initialize (→ INITIALIZED) / Open (→ FINISHED) /
/// Close/Terminate (→ NONE).
static SIGNIN_STATUS: AtomicI32 = AtomicI32::new(STATUS_NONE);

/// Register libSceSigninDialog HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceSigninDialog",
        "sceSigninDialogInitialize",
        hle_initialize,
    );
    // Param initializer (zeroes the caller's SceSigninDialogParam). We ignore
    // the param on Open, so acknowledging is sufficient for the title to
    // proceed — but the caller's struct is an output this shim never touches,
    // so anything the title reads back out of it is its own leftover memory.
    registry.register_incomplete(
        "libSceSigninDialog",
        "sceSigninDialogParamInitialize",
        hle_ok,
        "reports success without initializing the caller's SceSigninDialogParam out-struct",
    );
    registry.register("libSceSigninDialog", "sceSigninDialogOpen", hle_open);
    registry.register("libSceSigninDialog", "sceSigninDialogGetStatus", hle_status);
    registry.register(
        "libSceSigninDialog",
        "sceSigninDialogUpdateStatus",
        hle_status,
    );
    registry.register(
        "libSceSigninDialog",
        "sceSigninDialogGetResult",
        hle_get_result,
    );
    registry.register("libSceSigninDialog", "sceSigninDialogClose", hle_close);
    registry.register(
        "libSceSigninDialog",
        "sceSigninDialogTerminate",
        hle_terminate,
    );
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

fn hle_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SIGNIN_STATUS.store(STATUS_INITIALIZED, Ordering::Relaxed);
    debug!("sceSigninDialogInitialize()");
    SCE_OK
}

/// `sceSigninDialogOpen(param)`: with a single always-signed-in local user and
/// no host sign-in UI, complete the dialog immediately — status jumps to
/// `FINISHED` so the title's next poll finishes and it proceeds (to open the
/// pad and start gameplay).
fn hle_open(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SIGNIN_STATUS.store(STATUS_FINISHED, Ordering::Relaxed);
    debug!("sceSigninDialogOpen() -> completes immediately (FINISHED, signed in)");
    SCE_OK
}

/// `sceSigninDialogUpdateStatus()` / `sceSigninDialogGetStatus()`: return the
/// current `SceCommonDialogStatus` (`FINISHED` once opened). Returned in
/// `eax`, so the u32 status widens into the u64 HLE return.
fn hle_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SIGNIN_STATUS.load(Ordering::Relaxed) as u32 as u64
}

/// `sceSigninDialogGetResult(SceSigninDialogResult *result)`: report success.
///
/// The `SceSigninDialogResult` layout is not publicly documented, and shadPS4
/// — which boots real titles — returns `ORBIS_OK` without touching the buffer.
/// We do the same: the status transition to `FINISHED` plus the `SCE_OK`
/// return is what advances a sign-in-gated title; the caller allocates and
/// (typically zero-)initializes its own result struct, and a zero result code
/// already reads as "signed in / OK". Writing a struct of guessed size here
/// would risk clobbering adjacent caller memory.
fn hle_get_result(_ctx: &HleContext, args: &[u64]) -> u64 {
    let result_ptr = args.first().copied().unwrap_or(0);
    debug!("sceSigninDialogGetResult(result={result_ptr:#x}) -> OK (signed in)");
    SCE_OK
}

fn hle_close(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SIGNIN_STATUS.store(STATUS_NONE, Ordering::Relaxed);
    debug!("sceSigninDialogClose()");
    SCE_OK
}

fn hle_terminate(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SIGNIN_STATUS.store(STATUS_NONE, Ordering::Relaxed);
    debug!("sceSigninDialogTerminate()");
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    /// The sign-in flow completes immediately: Initialize -> INITIALIZED, Open
    /// -> FINISHED (the title's poll sees it right away, no hang), then
    /// Terminate returns to NONE. A title gated on this proceeds to open the
    /// pad instead of parking on a sign-in screen.
    #[test]
    fn signin_dialog_completes_immediately_and_reports_finished() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_initialize(&ctx, &[]), SCE_OK);
        assert_eq!(hle_status(&ctx, &[]) as i32, STATUS_INITIALIZED);

        assert_eq!(hle_open(&ctx, &[0x100]), SCE_OK);
        // The title's poll loop sees FINISHED immediately.
        assert_eq!(hle_status(&ctx, &[]) as i32, STATUS_FINISHED);

        // Result reports success.
        assert_eq!(hle_get_result(&ctx, &[0x200]), SCE_OK);

        // Terminate returns to NONE.
        assert_eq!(hle_terminate(&ctx, &[]), SCE_OK);
        assert_eq!(hle_status(&ctx, &[]) as i32, STATUS_NONE);
    }

    /// Every entry point returns SCE_OK so a title checking return codes at
    /// each step of the sign-in flow proceeds.
    #[test]
    fn all_entry_points_return_ok() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_ok(&ctx, &[]), SCE_OK); // ParamInitialize
        assert_eq!(hle_initialize(&ctx, &[]), SCE_OK);
        assert_eq!(hle_close(&ctx, &[]), SCE_OK);
    }
}
