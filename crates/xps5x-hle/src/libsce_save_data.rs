//! HLE libSceSaveData — save-data mount management.
//!
//! A title saves progress by `sceSaveDataMount`-ing a save slot, which
//! returns a mount-point path (`/savedata0`); the title then opens/writes
//! files under that path, and unmounts. XPS5X already exposes a
//! `/savedata0/` VFS mount backed by a host `savedata` directory, and the
//! file I/O path persists writes on close (see `xps5x-kernel`'s VFS +
//! `libkernel` write). So mounting here just hands back the standard mount
//! point and the whole save→file→persist chain works end to end. The mount
//! result layout (64 bytes, mount-point string at offset 0) and the flat
//! `/savedata0` point are cross-checked against SharpEmu's `SaveDataExports`.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `SCE_SAVE_DATA_ERROR_PARAMETER`.
const ERROR_PARAMETER: u64 = 0x809F_0000 | 0x02;
/// `SceSaveDataMountResult` size in bytes.
const MOUNT_RESULT_SIZE: usize = 0x40;
/// The mount point handed back to a title (matches SharpEmu). Games open
/// their save files under this path, which the VFS maps to the host
/// `savedata` directory.
const MOUNT_POINT: &[u8] = b"/savedata0";

/// Register libSceSaveData HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceSaveData", "sceSaveDataInitialize", hle_ok);
    registry.register("libSceSaveData", "sceSaveDataInitialize2", hle_ok);
    registry.register("libSceSaveData", "sceSaveDataInitialize3", hle_ok);
    registry.register("libSceSaveData", "sceSaveDataTerminate", hle_ok);
    registry.register("libSceSaveData", "sceSaveDataMount", hle_mount);
    registry.register("libSceSaveData", "sceSaveDataMount2", hle_mount);
    registry.register("libSceSaveData", "sceSaveDataMount3", hle_mount);
    registry.register("libSceSaveData", "sceSaveDataUmount", hle_ok);
    registry.register("libSceSaveData", "sceSaveDataUmount2", hle_ok);
    registry.register("libSceSaveData", "sceSaveDataDirNameSearch", hle_ok);
    registry.register("libSceSaveData", "sceSaveDataGetMountInfo", hle_ok);
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `sceSaveDataMount{,2,3}(const SceSaveDataMount* mount,
/// SceSaveDataMountResult* result)`: writes the mount-point path
/// (`/savedata0`) into `result`'s leading 16-byte field and returns OK. The
/// title then does its file I/O under that path, which persists to the host
/// `savedata` directory via the VFS.
fn hle_mount(ctx: &HleContext, args: &[u64]) -> u64 {
    let mount_ptr = args.first().copied().unwrap_or(0);
    let result_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sceSaveDataMount(mount={mount_ptr:#x}, result={result_ptr:#x})");
    if mount_ptr == 0 || result_ptr == 0 {
        return ERROR_PARAMETER;
    }

    // SceSaveDataMountResult (64 bytes): mountPoint.data[16] at offset 0,
    // then mountStatus/requiredBlocks/... (left zero).
    let mut result = [0u8; MOUNT_RESULT_SIZE];
    result[..MOUNT_POINT.len()].copy_from_slice(MOUNT_POINT);
    // result[MOUNT_POINT.len()] stays 0 — the NUL terminator.
    if !ctx.mem.write(result_ptr, &result) {
        warn!("sceSaveDataMount: result out-ptr {result_ptr:#x} not writable");
        return ERROR_PARAMETER;
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn mount_writes_the_savedata_mount_point() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // A (mostly irrelevant here) mount request at 0x100, result at 0x200.
        assert_eq!(hle_mount(&ctx, &[0x100, 0x200]), SCE_OK);
        let mut res = [0u8; MOUNT_RESULT_SIZE];
        assert!(mem.read(0x200, &mut res));
        let nul = res.iter().position(|&b| b == 0).unwrap();
        assert_eq!(&res[..nul], MOUNT_POINT, "mount point must be /savedata0");
    }

    #[test]
    fn mount_with_null_pointers_is_parameter_error() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_mount(&ctx, &[0, 0x200]), ERROR_PARAMETER);
        assert_eq!(hle_mount(&ctx, &[0x100, 0]), ERROR_PARAMETER);
    }
}
