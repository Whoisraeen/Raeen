//! HLE libSceSystemService — system state / events / settings.
//!
//! A running title polls `sceSystemServiceGetStatus` every frame to learn
//! about system events (a return to the home menu, an overlay, a controller
//! change) and reads system settings via `sceSystemServiceParamGetInt`.
//! XPS5X reports a quiet, steady state (no pending events, full display safe
//! area) so a title's main loop runs undisturbed. Struct sizes, the safe-area
//! `1.0` ratio, and the `ParamGetInt` value mapping are cross-checked against
//! SharpEmu's `SystemServiceExports`.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `SCE_SYSTEM_SERVICE_ERROR_PARAMETER`.
const ERROR_PARAMETER: u64 = 0x80A1_0003;
/// `SceSystemServiceStatus` is a 12-byte struct; the first `int32` is
/// `eventNum` (pending system events).
const STATUS_SIZE: usize = 0x0C;
/// `SceSystemServiceDisplaySafeAreaInfo` = `float ratio` + 128 reserved
/// bytes.
const SAFE_AREA_SIZE: usize = 4 + 128;

/// Register libSceSystemService HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceSystemService",
        "sceSystemServiceGetStatus",
        hle_get_status,
    );
    registry.register(
        "libSceSystemService",
        "sceSystemServiceParamGetInt",
        hle_param_get_int,
    );
    registry.register(
        "libSceSystemService",
        "sceSystemServiceGetDisplaySafeAreaInfo",
        hle_get_safe_area,
    );
    registry.register(
        "libSceSystemService",
        "sceSystemServiceHideSplashScreen",
        hle_ok,
    );
    registry.register(
        "libSceSystemService",
        "sceSystemServiceReportAbnormalTermination",
        hle_ok,
    );
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `sceSystemServiceGetStatus(SceSystemServiceStatus *status)`: reports a
/// quiet state — `eventNum = 0`, no overlay, not backgrounded (all-zero
/// 12-byte struct) — so a title's per-frame poll sees nothing to handle.
fn hle_get_status(ctx: &HleContext, args: &[u64]) -> u64 {
    let status_ptr = args.first().copied().unwrap_or(0);
    debug!("sceSystemServiceGetStatus(status={status_ptr:#x})");
    if status_ptr == 0 {
        return ERROR_PARAMETER;
    }
    if !ctx.mem.write(status_ptr, &[0u8; STATUS_SIZE]) {
        warn!("sceSystemServiceGetStatus: status out-ptr {status_ptr:#x} not writable");
        return ERROR_PARAMETER;
    }
    SCE_OK
}

/// `sceSystemServiceParamGetInt(int paramId, int *value)`: writes the system
/// setting for `paramId`. Values mirror SharpEmu's mapping (a stable set of
/// defaults): params 1/2/3/1000 → 1, param 4 → 180, everything else → 0.
fn hle_param_get_int(ctx: &HleContext, args: &[u64]) -> u64 {
    let param_id = args.first().copied().unwrap_or(0) as i32;
    let value_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sceSystemServiceParamGetInt(paramId={param_id}, value={value_ptr:#x})");
    if value_ptr == 0 {
        return ERROR_PARAMETER;
    }
    let value: i32 = match param_id {
        1 | 2 | 3 | 1000 => 1,
        4 => 180,
        _ => 0,
    };
    if !ctx.mem.write(value_ptr, &value.to_le_bytes()) {
        warn!("sceSystemServiceParamGetInt: value out-ptr {value_ptr:#x} not writable");
        return ERROR_PARAMETER;
    }
    SCE_OK
}

/// `sceSystemServiceGetDisplaySafeAreaInfo(SceSystemServiceDisplaySafeAreaInfo
/// *info)`: reports a full safe area (`ratio = 1.0`) so a title renders to
/// the whole display.
fn hle_get_safe_area(ctx: &HleContext, args: &[u64]) -> u64 {
    let info_ptr = args.first().copied().unwrap_or(0);
    debug!("sceSystemServiceGetDisplaySafeAreaInfo(info={info_ptr:#x})");
    if info_ptr == 0 {
        return ERROR_PARAMETER;
    }
    let mut buf = [0u8; SAFE_AREA_SIZE];
    buf[0..4].copy_from_slice(&1.0f32.to_le_bytes()); // ratio = 1.0 (full)
    if !ctx.mem.write(info_ptr, &buf) {
        warn!("sceSystemServiceGetDisplaySafeAreaInfo: info out-ptr {info_ptr:#x} not writable");
        return ERROR_PARAMETER;
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn get_status_reports_no_events() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, &[0xEE; STATUS_SIZE]));
        assert_eq!(hle_get_status(&ctx, &[0x100]), SCE_OK);
        let mut s = [0u8; STATUS_SIZE];
        assert!(mem.read(0x100, &mut s));
        assert_eq!(
            i32::from_le_bytes(s[0..4].try_into().unwrap()),
            0,
            "eventNum == 0"
        );
        assert_eq!(hle_get_status(&ctx, &[0]), ERROR_PARAMETER);
    }

    #[test]
    fn param_get_int_mirrors_the_default_mapping() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        for (pid, want) in [(1, 1), (2, 1), (3, 1), (1000, 1), (4, 180), (99, 0)] {
            assert_eq!(hle_param_get_int(&ctx, &[pid as u64, 0x200]), SCE_OK);
            let mut v = [0u8; 4];
            assert!(mem.read(0x200, &mut v));
            assert_eq!(i32::from_le_bytes(v), want, "paramId {pid}");
        }
        assert_eq!(hle_param_get_int(&ctx, &[1, 0]), ERROR_PARAMETER);
    }

    #[test]
    fn safe_area_reports_full_ratio() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_safe_area(&ctx, &[0x100]), SCE_OK);
        let mut r = [0u8; 4];
        assert!(mem.read(0x100, &mut r));
        assert_eq!(f32::from_le_bytes(r), 1.0, "full safe-area ratio");
    }
}
