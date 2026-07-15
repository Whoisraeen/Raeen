//! HLE libSceAppContent — additional content (DLC) / app params.
//!
//! A title queries this at boot to learn whether it's the full SKU (vs a
//! trial) and to enumerate installed additional content. XPS5X reports the
//! **full SKU** and **no DLC installed**, so a title's boot-time content
//! check passes and it proceeds to the main game. `AppParamGetInt`'s SKU-flag
//! value and the empty add-content list are cross-checked against SharpEmu's
//! `AppContentExports`.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `SCE_APP_CONTENT_ERROR_PARAMETER`.
const ERROR_PARAMETER: u64 = 0x8092_0002;
/// `SCE_APP_CONTENT_APPPARAM_ID_SKU_FLAG`.
const APPPARAM_ID_SKU_FLAG: u32 = 0;
/// `SCE_APP_CONTENT_APPPARAM_SKU_FLAG_FULL` — the full game (not a trial).
const SKU_FLAG_FULL: i32 = 3;

/// Register libSceAppContent HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAppContent", "sceAppContentInitialize", hle_ok);
    registry.register(
        "libSceAppContent",
        "sceAppContentAppParamGetInt",
        hle_app_param_get_int,
    );
    registry.register(
        "libSceAppContent",
        "sceAppContentGetAddcontInfoList",
        hle_get_addcont_info_list,
    );
    registry.register(
        "libSceAppContent",
        "sceAppContentGetAddcontInfo",
        hle_get_addcont_info,
    );
    registry.register(
        "libSceAppContent",
        "sceAppContentTemporaryDataMount2",
        hle_temporary_data_mount2,
    );
    registry.register(
        "libSceAppContent",
        "sceAppContentTemporaryDataGetAvailableSpaceKb",
        hle_temporary_data_available_space,
    );
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `sceAppContentAppParamGetInt(paramId, int *value)`: the SKU-flag param
/// reports the full game; anything else reports `0`.
fn hle_app_param_get_int(ctx: &HleContext, args: &[u64]) -> u64 {
    let param_id = args.first().copied().unwrap_or(0) as u32;
    let value_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sceAppContentAppParamGetInt(paramId={param_id}, value={value_ptr:#x})");
    if value_ptr == 0 {
        return ERROR_PARAMETER;
    }
    let value: i32 = if param_id == APPPARAM_ID_SKU_FLAG {
        SKU_FLAG_FULL
    } else {
        0
    };
    if !ctx.mem.write(value_ptr, &value.to_le_bytes()) {
        warn!("sceAppContentAppParamGetInt: value out-ptr {value_ptr:#x} not writable");
        return ERROR_PARAMETER;
    }
    SCE_OK
}

/// `sceAppContentGetAddcontInfoList(serviceLabel, list, listNum, uint32_t
/// *hitNum)`: reports zero installed DLC (`*hitNum = 0`).
fn hle_get_addcont_info_list(ctx: &HleContext, args: &[u64]) -> u64 {
    let hit_num_ptr = args.get(3).copied().unwrap_or(0);
    debug!("sceAppContentGetAddcontInfoList(hitNum={hit_num_ptr:#x})");
    if hit_num_ptr != 0 && !ctx.mem.write(hit_num_ptr, &0u32.to_le_bytes()) {
        warn!("sceAppContentGetAddcontInfoList: hitNum out-ptr {hit_num_ptr:#x} not writable");
    }
    SCE_OK
}

/// `sceAppContentGetAddcontInfo(...)`: a specific DLC entry — none exist, so
/// this reports "not found" via `ERROR_PARAMETER` (the entity isn't there).
fn hle_get_addcont_info(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceAppContentGetAddcontInfo() -> no DLC installed");
    ERROR_PARAMETER
}

/// Mount the process-private writable temporary-data directory and return the
/// canonical guest mount name through the second argument. ABI and behavior
/// are cross-checked against SharpEmu's GPL-2.0 implementation.
fn hle_temporary_data_mount2(ctx: &HleContext, args: &[u64]) -> u64 {
    let mount_out = args.get(1).copied().unwrap_or(0);
    if mount_out == 0 {
        return ERROR_PARAMETER;
    }
    let Some(host_root) = ctx.kernel.filesystem.resolve_path("/temp0/") else {
        return ERROR_PARAMETER;
    };
    if std::fs::create_dir_all(&host_root).is_err() || !ctx.mem.write(mount_out, b"/temp0\0") {
        return ERROR_PARAMETER;
    }
    SCE_OK
}

/// Report a stable 1 GiB process-private temporary-data quota. This avoids
/// leaking host disk capacity while giving titles a realistic nonzero budget.
fn hle_temporary_data_available_space(ctx: &HleContext, args: &[u64]) -> u64 {
    const TEMP_QUOTA_KIB: u64 = 1024 * 1024;

    let mount_ptr = args.first().copied().unwrap_or(0);
    let available_out = args.get(1).copied().unwrap_or(0);
    let Some(mount) = crate::fmt::read_cstr(ctx.mem, mount_ptr) else {
        return ERROR_PARAMETER;
    };
    if mount != b"/temp0" || available_out == 0 {
        return ERROR_PARAMETER;
    }
    if !ctx.mem.write(available_out, &TEMP_QUOTA_KIB.to_le_bytes()) {
        return ERROR_PARAMETER;
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn sku_flag_reports_full_game_and_no_dlc() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // SKU flag → full game.
        assert_eq!(
            hle_app_param_get_int(&ctx, &[APPPARAM_ID_SKU_FLAG as u64, 0x100]),
            SCE_OK
        );
        let mut v = [0u8; 4];
        assert!(mem.read(0x100, &mut v));
        assert_eq!(i32::from_le_bytes(v), SKU_FLAG_FULL);

        // Add-content list → zero installed.
        assert!(mem.write(0x200, &0xFFFF_FFFFu32.to_le_bytes()));
        assert_eq!(hle_get_addcont_info_list(&ctx, &[0, 0, 0, 0x200]), SCE_OK);
        let mut n = [0u8; 4];
        assert!(mem.read(0x200, &mut n));
        assert_eq!(u32::from_le_bytes(n), 0, "no DLC installed");

        // NULL value ptr → parameter error.
        assert_eq!(hle_app_param_get_int(&ctx, &[0, 0]), ERROR_PARAMETER);

        assert_eq!(hle_temporary_data_mount2(&ctx, &[0, 0x300]), SCE_OK);
        let mut mount = [0u8; 7];
        assert!(mem.read(0x300, &mut mount));
        assert_eq!(&mount, b"/temp0\0");
        assert_eq!(hle_temporary_data_mount2(&ctx, &[0, 0]), ERROR_PARAMETER);

        assert_eq!(
            hle_temporary_data_available_space(&ctx, &[0x300, 0x380]),
            SCE_OK
        );
        let mut available = [0u8; 8];
        assert!(mem.read(0x380, &mut available));
        assert_eq!(u64::from_le_bytes(available), 1024 * 1024);
        assert_eq!(
            hle_temporary_data_available_space(&ctx, &[0, 0x380]),
            ERROR_PARAMETER
        );
    }
}
