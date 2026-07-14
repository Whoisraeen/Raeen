//! HLE for small headless system/peripheral libraries: **libSceMouse**,
//! **libSceIme** (on-screen keyboard), and **libSceGameUpdate**.
//!
//! XPS5X runs headless with no mouse, no active IME session, and no store
//! connection, so each of these reports its benign "nothing here" state:
//! `sceMouseOpen`/`Read` succeed with zero entries (no mouse), `sceImeUpdate`/
//! keyboard calls succeed with no pending events (no IME session), and
//! `sceGameUpdateInitialize` succeeds. These NIDs are cross-checked against
//! SharpEmu; without them a title that polls a mouse or an IME (e.g. Quake KEX
//! calls `sceImeUpdate` from its main loop) hits an unresolved import and dies.

use crate::{HleContext, HleRegistry};

/// `SCE_OK`.
const OK: u64 = 0;

/// Register the small headless peripheral/system libraries.
pub fn register(registry: &HleRegistry) {
    // libSceMouse — no mouse connected.
    registry.register("libSceMouse", "sceMouseOpen", hle_ok);
    registry.register("libSceMouse", "sceMouseClose", hle_ok);
    registry.register("libSceMouse", "sceMouseRead", hle_ok); // 0 entries read

    // libSceIme — no active IME (on-screen keyboard) session.
    registry.register("libSceIme", "sceImeUpdate", hle_ok); // no pending events
    registry.register("libSceIme", "sceImeKeyboardOpen", hle_ok);
    registry.register("libSceIme", "sceImeKeyboardClose", hle_ok);
    registry.register("libSceIme", "sceImeKeyboardGetResourceId", hle_ok);
    registry.register("libSceIme", "sceImeOpen", hle_ok);
    registry.register("libSceIme", "sceImeClose", hle_ok);

    // libSceGameUpdate — no store connection; initialization succeeds.
    registry.register("libSceGameUpdate", "sceGameUpdateInitialize", hle_ok);
    registry.register("libSceGameUpdate", "sceGameUpdateTerminate", hle_ok);
}

/// Report benign success (`rax = 0`): no device, no session, no pending event.
fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_registered_functions_report_success() {
        // The registry resolves these NIDs (instead of an unresolved-import
        // crash) and each reports the benign headless state.
        let reg = HleRegistry::new();
        for (lib, func) in [
            ("libSceMouse", "sceMouseOpen"),
            ("libSceMouse", "sceMouseRead"),
            ("libSceIme", "sceImeUpdate"),
            ("libSceIme", "sceImeKeyboardOpen"),
            ("libSceGameUpdate", "sceGameUpdateInitialize"),
        ] {
            let kernel = xps5x_kernel::OrbisKernel::new();
            let mem = crate::TestMemory::new(0x10);
            let alloc = crate::TestAllocator::new(0);
            let ctx = crate::test_ctx(&kernel, &mem, &alloc);
            assert_eq!(
                reg.call(&ctx, lib, func, &[]),
                Some(OK),
                "{lib}::{func} must be registered and return OK"
            );
        }
    }
}
