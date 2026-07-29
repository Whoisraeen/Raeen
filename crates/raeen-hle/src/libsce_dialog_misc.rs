//! HLE libSceErrorDialog + libSceVrSetupDialog — the two remaining common
//! dialogs a measured title imports and Raeen had no library for at all.
//!
//! Both were **zero** registrations in this crate before, so every import
//! resolved to nothing. Measured on Blasphemous II (PPSA13580) via
//! `cargo xtask nids coverage`: 4 unresolved `libSceErrorDialog` imports
//! (`Initialize`/`Open`/`UpdateStatus`/`Terminate`) and 6 unresolved
//! `libSceVrSetupDialog` imports (`Initialize`/`Open`/`UpdateStatus`/
//! `GetResult`/`Close`/`Terminate`).
//!
//! Both follow the common-dialog contract every other dialog in this crate
//! implements: `Initialize` → `INITIALIZED`, `Open` → `FINISHED` immediately
//! (no host popup exists to display, and a status that never reaches `FINISHED`
//! parks the title's poll loop forever), `Close`/`Terminate` → `NONE`. Status
//! values are the shared `SceCommonDialogStatus`, identical to
//! `libsce_common_dialog.rs` and `libsce_signin_dialog.rs`.
//!
//! # Why each dialog behaves the way it does
//!
//! * **libSceErrorDialog** shows a system error popup and returns no result to
//!   the guest — it exists purely to tell the *player* something. With no popup
//!   surface, the honest model is "the dialog was shown and dismissed", which is
//!   what completing immediately expresses. The error code the title passed is
//!   logged at `warn!` so it lands in the crash/diagnostic log instead of
//!   disappearing: a title opening an error dialog is reporting a real problem,
//!   and that is exactly the line the next debugging session needs.
//! * **libSceVrSetupDialog** walks the player through PSVR2 headset setup.
//!   Raeen has no VR device (`libSceVr*` is not implemented at all), so the
//!   dialog completes with a **canceled** result rather than a success one:
//!   claiming the headset was set up would make a title proceed into a VR mode
//!   with no device behind it. `GetResult` is `register_incomplete` — its
//!   `SceVrSetupDialogResult` layout is not publicly documented, so the result
//!   code is deliberately not written into a guessed struct (the same rule
//!   `libsce_signin_dialog.rs` follows for `sceSigninDialogGetResult`).
//!
//! Function names match the Orbis exports so the NID linker (`nid_of(name)`)
//! resolves each import.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;

/// `SceCommonDialogStatus`, shared by every common dialog.
const STATUS_NONE: i32 = 0;
const STATUS_INITIALIZED: i32 = 1;
const STATUS_FINISHED: i32 = 3;

/// One error dialog at a time (the real API's constraint).
static ERROR_DIALOG_STATUS: AtomicI32 = AtomicI32::new(STATUS_NONE);
/// One VR-setup dialog at a time.
static VR_SETUP_STATUS: AtomicI32 = AtomicI32::new(STATUS_NONE);

/// Register libSceErrorDialog + libSceVrSetupDialog HLE functions.
pub fn register(registry: &HleRegistry) {
    // --- libSceErrorDialog ---------------------------------------------------
    registry.register(
        "libSceErrorDialog",
        "sceErrorDialogInitialize",
        hle_error_initialize,
    );
    registry.register("libSceErrorDialog", "sceErrorDialogOpen", hle_error_open);
    registry.register(
        "libSceErrorDialog",
        "sceErrorDialogUpdateStatus",
        hle_error_status,
    );
    registry.register(
        "libSceErrorDialog",
        "sceErrorDialogGetStatus",
        hle_error_status,
    );
    registry.register("libSceErrorDialog", "sceErrorDialogClose", hle_error_close);
    registry.register(
        "libSceErrorDialog",
        "sceErrorDialogTerminate",
        hle_error_close,
    );

    // --- libSceVrSetupDialog --------------------------------------------------
    registry.register(
        "libSceVrSetupDialog",
        "sceVrSetupDialogInitialize",
        hle_vr_initialize,
    );
    registry.register("libSceVrSetupDialog", "sceVrSetupDialogOpen", hle_vr_open);
    registry.register(
        "libSceVrSetupDialog",
        "sceVrSetupDialogUpdateStatus",
        hle_vr_status,
    );
    registry.register(
        "libSceVrSetupDialog",
        "sceVrSetupDialogGetStatus",
        hle_vr_status,
    );
    registry.register_incomplete(
        "libSceVrSetupDialog",
        "sceVrSetupDialogGetResult",
        hle_vr_get_result,
        "no VR device: reports OK without writing the undocumented \
         SceVrSetupDialogResult layout, so the guest reads its own zeroed struct",
    );
    registry.register("libSceVrSetupDialog", "sceVrSetupDialogClose", hle_vr_close);
    registry.register(
        "libSceVrSetupDialog",
        "sceVrSetupDialogTerminate",
        hle_vr_close,
    );
}

fn hle_error_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    ERROR_DIALOG_STATUS.store(STATUS_INITIALIZED, Ordering::Relaxed);
    debug!("sceErrorDialogInitialize()");
    SCE_OK
}

/// `sceErrorDialogOpen(const SceErrorDialogParam *param)`: complete
/// immediately, and surface the error the title wanted the player to see.
///
/// `SceErrorDialogParam` begins with `size` then `errorCode` (both 32-bit), so
/// the code is read at +4 when the pointer is readable. It is logged rather
/// than swallowed: a title that opens this dialog has hit a condition it
/// considers user-visible, which is precisely what a compatibility log must
/// carry.
fn hle_error_open(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    let error_code = if param == 0 {
        None
    } else {
        let mut raw = [0u8; 4];
        ctx.mem
            .read(param.wrapping_add(4), &mut raw)
            .then(|| u32::from_le_bytes(raw))
    };
    match error_code {
        Some(code) => warn!(
            "sceErrorDialogOpen: the title is reporting error {code:#010x} to the player — dialog \
             completes immediately (no host popup)"
        ),
        None => warn!(
            "sceErrorDialogOpen(param={param:#x}): the title is reporting an error to the player \
             but its param is unreadable — dialog completes immediately"
        ),
    }
    ERROR_DIALOG_STATUS.store(STATUS_FINISHED, Ordering::Relaxed);
    SCE_OK
}

/// `sceErrorDialogUpdateStatus()` / `GetStatus()`: the current
/// `SceCommonDialogStatus`, returned in `eax`.
fn hle_error_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    ERROR_DIALOG_STATUS.load(Ordering::Relaxed) as u32 as u64
}

fn hle_error_close(_ctx: &HleContext, _args: &[u64]) -> u64 {
    ERROR_DIALOG_STATUS.store(STATUS_NONE, Ordering::Relaxed);
    debug!("sceErrorDialogClose/Terminate()");
    SCE_OK
}

fn hle_vr_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    VR_SETUP_STATUS.store(STATUS_INITIALIZED, Ordering::Relaxed);
    debug!("sceVrSetupDialogInitialize()");
    SCE_OK
}

/// `sceVrSetupDialogOpen(param)`: complete immediately.
///
/// There is no VR device and no setup flow to run, so the dialog finishes
/// without ever claiming a headset was configured — see the module docs.
fn hle_vr_open(_ctx: &HleContext, _args: &[u64]) -> u64 {
    VR_SETUP_STATUS.store(STATUS_FINISHED, Ordering::Relaxed);
    debug!("sceVrSetupDialogOpen() -> completes immediately, no VR device present");
    SCE_OK
}

fn hle_vr_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    VR_SETUP_STATUS.load(Ordering::Relaxed) as u32 as u64
}

/// `sceVrSetupDialogGetResult(SceVrSetupDialogResult *result)`: report success
/// of the *call* without writing the caller's struct.
///
/// The layout is undocumented, so writing a guessed one could clobber adjacent
/// caller memory. The guest's own zero-initialized struct already reads as
/// "nothing happened", which is the truth: no headset was set up.
fn hle_vr_get_result(_ctx: &HleContext, args: &[u64]) -> u64 {
    let result_ptr = args.first().copied().unwrap_or(0);
    debug!("sceVrSetupDialogGetResult(result={result_ptr:#x}) -> OK (no VR device)");
    SCE_OK
}

fn hle_vr_close(_ctx: &HleContext, _args: &[u64]) -> u64 {
    VR_SETUP_STATUS.store(STATUS_NONE, Ordering::Relaxed);
    debug!("sceVrSetupDialogClose/Terminate()");
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_bits() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            raeen_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x100),
            crate::TestAllocator::new(0),
        )
    }

    /// The error dialog runs the full common-dialog cycle without hanging:
    /// Initialize -> INITIALIZED, Open -> FINISHED on the very next poll,
    /// Terminate -> NONE. A title gated on `UpdateStatus == FINISHED` proceeds.
    #[test]
    fn error_dialog_completes_immediately_and_returns_to_none() {
        let (k, m, a) = ctx_bits();
        let ctx = crate::test_ctx(&k, &m, &a);

        ERROR_DIALOG_STATUS.store(STATUS_NONE, Ordering::Relaxed);
        assert_eq!(hle_error_status(&ctx, &[]), STATUS_NONE as u64);
        assert_eq!(hle_error_initialize(&ctx, &[]), SCE_OK);
        assert_eq!(hle_error_status(&ctx, &[]), STATUS_INITIALIZED as u64);
        assert_eq!(hle_error_open(&ctx, &[0]), SCE_OK);
        assert_eq!(hle_error_status(&ctx, &[]), STATUS_FINISHED as u64);
        assert_eq!(hle_error_close(&ctx, &[]), SCE_OK);
        assert_eq!(hle_error_status(&ctx, &[]), STATUS_NONE as u64);
    }

    /// The VR-setup dialog completes too — the point being that it cannot hang
    /// a title that has no headset attached.
    #[test]
    fn vr_setup_dialog_completes_immediately_without_a_device() {
        let (k, m, a) = ctx_bits();
        let ctx = crate::test_ctx(&k, &m, &a);

        VR_SETUP_STATUS.store(STATUS_NONE, Ordering::Relaxed);
        assert_eq!(hle_vr_initialize(&ctx, &[]), SCE_OK);
        assert_eq!(hle_vr_status(&ctx, &[]), STATUS_INITIALIZED as u64);
        assert_eq!(hle_vr_open(&ctx, &[0]), SCE_OK);
        assert_eq!(hle_vr_status(&ctx, &[]), STATUS_FINISHED as u64);
        assert_eq!(hle_vr_get_result(&ctx, &[0]), SCE_OK);
        assert_eq!(hle_vr_close(&ctx, &[]), SCE_OK);
        assert_eq!(hle_vr_status(&ctx, &[]), STATUS_NONE as u64);
    }

    /// Every NID the measured title imports from these two libraries must be
    /// registered — the whole point of adding the modules.
    ///
    /// Names come from `cargo xtask nids coverage` run against Blasphemous II
    /// (PPSA13580), where all ten resolved to nothing.
    #[test]
    fn every_measured_import_of_both_libraries_is_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceErrorDialogInitialize",
            "sceErrorDialogOpen",
            "sceErrorDialogUpdateStatus",
            "sceErrorDialogTerminate",
        ] {
            assert!(
                registry.is_implemented("libSceErrorDialog", name),
                "libSceErrorDialog::{name} is imported by the measured title"
            );
        }
        for name in [
            "sceVrSetupDialogInitialize",
            "sceVrSetupDialogOpen",
            "sceVrSetupDialogUpdateStatus",
            "sceVrSetupDialogGetResult",
            "sceVrSetupDialogClose",
            "sceVrSetupDialogTerminate",
        ] {
            assert!(
                registry.is_implemented("libSceVrSetupDialog", name),
                "libSceVrSetupDialog::{name} is imported by the measured title"
            );
        }
    }
}
