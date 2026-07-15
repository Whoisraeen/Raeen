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
use std::sync::atomic::Ordering;
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
    for library in [
        "libSceSaveData",
        "libSceSaveData_native",
        "libSceSaveData.native",
    ] {
        registry.register(library, "sceSaveDataInitialize", hle_ok);
        registry.register(library, "sceSaveDataInitialize2", hle_ok);
        registry.register(library, "sceSaveDataInitialize3", hle_ok);
        registry.register(library, "sceSaveDataTerminate", hle_ok);
        registry.register(library, "sceSaveDataMount", hle_mount);
        registry.register(library, "sceSaveDataMount2", hle_mount);
        registry.register(library, "sceSaveDataMount3", hle_mount);
        registry.register(library, "sceSaveDataUmount", hle_ok);
        registry.register(library, "sceSaveDataUmount2", hle_ok);
        registry.register(library, "sceSaveDataDirNameSearch", hle_ok);
        registry.register(library, "sceSaveDataDirNameSearchPs4", hle_dir_name_search);
        registry.register(library, "sceSaveDataGetMountInfo", hle_ok);
        registry.register(library, "sceSaveDataDelete", hle_delete);
        registry.register(
            library,
            "sceSaveDataCreateTransactionResource",
            hle_create_transaction_resource,
        );
        registry.register(
            library,
            "sceSaveDataDeleteTransactionResource",
            hle_delete_transaction_resource,
        );
        registry.register(library, "sceSaveDataPrepare", hle_prepare);
        registry.register(library, "sceSaveDataSetParam", hle_set_param);
    }
    registry.register_nid(
        "libSceSaveData_native",
        "sceSaveDataCreateTransactionResource",
        0x8234_5936_7c34_24f1,
        hle_create_transaction_resource,
    );
    registry.register_nid(
        "libSceSaveData_native",
        "sceSaveDataDeleteTransactionResource",
        0x9495_10b9_a2aa_a0a6,
        hle_delete_transaction_resource,
    );
    registry.register_nid(
        "libSceSaveData_native",
        "sceSaveDataPrepare",
        0xb030_81ae_673a_d575,
        hle_prepare,
    );
    registry.register_nid(
        "libSceSaveData_native",
        "sceSaveDataSetParam",
        0xf39c_ee97_ffde_197b,
        hle_set_param,
    );
}

/// Allocate a process-local transaction resource and retain the requested
/// working-memory size for lifecycle validation.
fn hle_create_transaction_resource(ctx: &HleContext, args: &[u64]) -> u64 {
    let memory_size = args.first().copied().unwrap_or(0);
    let resource = ctx
        .kernel
        .save_data_next_transaction_resource
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    ctx.kernel
        .save_data_transaction_resources
        .insert(resource, memory_size);
    resource as u32 as u64
}

/// Delete a transaction resource. Platform deletion is idempotent.
fn hle_delete_transaction_resource(ctx: &HleContext, args: &[u64]) -> u64 {
    let resource = args.first().copied().unwrap_or(0) as i32;
    ctx.kernel.save_data_transaction_resources.remove(&resource);
    SCE_OK
}

/// Prepare a save-data operation against a transaction resource. The prepare
/// descriptor begins with the resource id returned by
/// `sceSaveDataCreateTransactionResource`; the remaining policy fields are
/// consumed by later mount/search operations.
fn hle_prepare(ctx: &HleContext, args: &[u64]) -> u64 {
    let operation = args.first().copied().unwrap_or(0);
    let descriptor = args.get(1).copied().unwrap_or(0);
    if operation == 0 || descriptor == 0 {
        return ERROR_PARAMETER;
    }
    let mut resource_bytes = [0u8; 4];
    if !ctx.mem.read(descriptor, &mut resource_bytes) {
        return 0x8002_000E;
    }
    let resource = i32::from_le_bytes(resource_bytes);
    if !ctx
        .kernel
        .save_data_transaction_resources
        .contains_key(&resource)
    {
        return ERROR_PARAMETER;
    }
    SCE_OK
}

/// Store save metadata for a mounted save. Parameter type 0 is the complete
/// 0x530-byte metadata record; types 1..=4 are title, subtitle, detail and the
/// 32-bit user parameter respectively. Native PS5 callers may pass zero for
/// the redundant size argument, so the ABI-defined size is used in that case.
fn hle_set_param(ctx: &HleContext, args: &[u64]) -> u64 {
    let mount_point = args.first().copied().unwrap_or(0);
    let parameter_type = args.get(1).copied().unwrap_or(u64::MAX) as u32;
    let parameter = args.get(2).copied().unwrap_or(0);
    let supplied_size = args.get(3).copied().unwrap_or(0);
    if mount_point == 0 || parameter == 0 || parameter_type > 4 {
        return ERROR_PARAMETER;
    }

    let mut mount_bytes = [0u8; 16];
    if !ctx.mem.read(mount_point, &mut mount_bytes) {
        return 0x8002_000E;
    }
    let mount_len = mount_bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(mount_bytes.len());
    let mount = String::from_utf8_lossy(&mount_bytes[..mount_len]).into_owned();
    if mount != "/savedata0" {
        return 0x809F_000B;
    }

    let expected_size = match parameter_type {
        0 => 0x530,
        1 | 2 => 128,
        3 => 1024,
        4 => 4,
        _ => unreachable!(),
    };
    let size = if supplied_size == 0 {
        expected_size
    } else {
        usize::try_from(supplied_size)
            .unwrap_or(usize::MAX)
            .min(expected_size)
    };
    if size == 0 {
        return ERROR_PARAMETER;
    }
    let mut value = vec![0u8; size];
    if !ctx.mem.read(parameter, &mut value) {
        return 0x8002_000E;
    }
    ctx.kernel
        .save_data_params
        .insert((mount, parameter_type), value);
    SCE_OK
}

/// Enumerate immediate save-slot directories. This implements the shared
/// empty/non-empty result ABI used by the PS4-compat spelling on PS5 as well
/// as native titles, without manufacturing any title data.
fn hle_dir_name_search(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    let result = args.get(1).copied().unwrap_or(0);
    if cond == 0 || result == 0 {
        return ERROR_PARAMETER;
    }
    let mut cond_bytes = [0u8; 32];
    let mut result_bytes = [0u8; 40];
    if !ctx.mem.read(cond, &mut cond_bytes) || !ctx.mem.read(result, &mut result_bytes) {
        return 0x8002_000E;
    }
    let sort_key = u32::from_le_bytes(cond_bytes[24..28].try_into().expect("fixed slice"));
    let sort_order = u32::from_le_bytes(cond_bytes[28..32].try_into().expect("fixed slice"));
    if sort_key > 5 || sort_order > 1 {
        return ERROR_PARAMETER;
    }
    let names_out = u64::from_le_bytes(result_bytes[8..16].try_into().expect("fixed slice"));
    let capacity = u32::from_le_bytes(result_bytes[16..20].try_into().expect("fixed slice"));
    let Some(root) = ctx.kernel.filesystem.resolve_path("/savedata0") else {
        return 0x809F_000B;
    };
    let mut names: Vec<String> = match std::fs::read_dir(root) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|ty| ty.is_dir()))
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| !name.starts_with("sce_") && name.len() < 32)
            .collect(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => {
            warn!("sceSaveDataDirNameSearchPs4: {error}");
            return 0x809F_000B;
        }
    };
    names.sort();
    if sort_order == 1 {
        names.reverse();
    }
    let set_count = names.len().min(capacity as usize);
    if !ctx.mem.write(result, &(names.len() as u32).to_le_bytes())
        || !ctx
            .mem
            .write(result + 0x14, &(set_count as u32).to_le_bytes())
    {
        return 0x8002_000E;
    }
    if set_count != 0 && names_out == 0 {
        return ERROR_PARAMETER;
    }
    for (index, name) in names.into_iter().take(set_count).enumerate() {
        let mut encoded = [0u8; 32];
        encoded[..name.len()].copy_from_slice(name.as_bytes());
        if !ctx.mem.write(names_out + index as u64 * 32, &encoded) {
            return 0x8002_000E;
        }
    }
    debug!("sceSaveDataDirNameSearchPs4 -> {set_count} entr(ies)");
    SCE_OK
}

/// `sceSaveDataDelete(const SceSaveDataDelete*)`: delete one directory below
/// the process-private `/savedata0` mount. The request stores `dirName*` at
/// offset 0x10; title/user separation is already provided by launch mounts.
fn hle_delete(ctx: &HleContext, args: &[u64]) -> u64 {
    let request = args.first().copied().unwrap_or(0);
    if request == 0 {
        return ERROR_PARAMETER;
    }
    let mut dir_ptr_bytes = [0u8; 8];
    if !ctx.mem.read(request + 0x10, &mut dir_ptr_bytes) {
        return 0x8002_000E;
    }
    let dir_ptr = u64::from_le_bytes(dir_ptr_bytes);
    let Some(dir_bytes) = crate::fmt::read_cstr(ctx.mem, dir_ptr) else {
        return 0x8002_000E;
    };
    if dir_bytes.is_empty()
        || dir_bytes.len() >= 32
        || dir_bytes.iter().any(|byte| matches!(*byte, b'/' | b'\\'))
        || dir_bytes.windows(2).any(|pair| pair == b"..")
    {
        return ERROR_PARAMETER;
    }
    let dir_name = String::from_utf8_lossy(&dir_bytes);
    let guest_path = format!("/savedata0/{dir_name}");
    match ctx.kernel.filesystem.remove_dir_all(&guest_path) {
        Ok(()) => {
            debug!("sceSaveDataDelete('{dir_name}') -> OK");
            SCE_OK
        }
        Err(error) => {
            warn!("sceSaveDataDelete('{dir_name}') failed: {error}");
            0x809F_000B
        }
    }
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
    fn transaction_resources_are_process_local_and_releasable() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_create_transaction_resource(&ctx, &[0x0200_0000]), 1);
        assert_eq!(hle_create_transaction_resource(&ctx, &[0x1000]), 2);
        assert_eq!(
            kernel
                .save_data_transaction_resources
                .get(&1)
                .map(|value| *value),
            Some(0x0200_0000)
        );
        assert!(mem.write(0x08, &1i32.to_le_bytes()));
        assert_eq!(hle_prepare(&ctx, &[0x04, 0x08]), SCE_OK);
        assert!(mem.write(0x08, &99i32.to_le_bytes()));
        assert_eq!(hle_prepare(&ctx, &[0x04, 0x08]), ERROR_PARAMETER);
        assert_eq!(hle_delete_transaction_resource(&ctx, &[1]), SCE_OK);
        assert!(!kernel.save_data_transaction_resources.contains_key(&1));
        assert_eq!(hle_delete_transaction_resource(&ctx, &[1]), SCE_OK);

        let registry = HleRegistry::new();
        register(&registry);
        let overrides = registry.registered_nid_overrides();
        assert!(overrides.iter().any(|(nid, key)| {
            *nid == 0x8234_5936_7c34_24f1
                && key == "libSceSaveData_native::sceSaveDataCreateTransactionResource"
        }));
        assert!(overrides.iter().any(|(nid, key)| {
            *nid == 0x9495_10b9_a2aa_a0a6
                && key == "libSceSaveData_native::sceSaveDataDeleteTransactionResource"
        }));
        assert!(overrides.iter().any(|(nid, key)| {
            *nid == 0xb030_81ae_673a_d575 && key == "libSceSaveData_native::sceSaveDataPrepare"
        }));
    }

    #[test]
    fn set_param_records_native_save_metadata() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x800);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x10, b"/savedata0\0"));
        let mut metadata = vec![0u8; 0x530];
        metadata[..10].copy_from_slice(b"Minecraft\0");
        assert!(mem.write(0x100, &metadata));

        assert_eq!(hle_set_param(&ctx, &[0x10, 0, 0x100, 0]), SCE_OK);
        let stored = kernel
            .save_data_params
            .get(&("/savedata0".to_string(), 0))
            .expect("metadata stored");
        assert_eq!(stored.len(), 0x530);
        assert_eq!(&stored[..10], b"Minecraft\0");
        drop(stored);
        assert_eq!(hle_set_param(&ctx, &[0x10, 5, 0x100, 4]), ERROR_PARAMETER);

        let registry = HleRegistry::new();
        register(&registry);
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0xf39c_ee97_ffde_197b
                        && key == "libSceSaveData_native::sceSaveDataSetParam"
                })
        );
    }

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

    #[test]
    fn delete_removes_only_the_requested_save_directory() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root = std::env::temp_dir().join(format!("xps5x-save-delete-{}", std::process::id()));
        std::fs::create_dir_all(root.join("slot1")).unwrap();
        std::fs::create_dir_all(root.join("slot2")).unwrap();
        kernel.filesystem.set_savedata_directory(&root);
        assert!(mem.write(0x100, &0x200u64.to_le_bytes()));
        // dirName pointer is request+0x10.
        assert!(mem.write(0x110, &0x200u64.to_le_bytes()));
        assert!(mem.write(0x200, b"slot1\0"));

        assert_eq!(hle_delete(&ctx, &[0x100]), SCE_OK);
        assert!(!root.join("slot1").exists());
        assert!(root.join("slot2").exists());
        assert!(mem.write(0x200, b"../slot2\0"));
        assert_eq!(hle_delete(&ctx, &[0x100]), ERROR_PARAMETER);
        assert!(root.join("slot2").exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn dir_name_search_reports_empty_and_enumerates_slots() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root = std::env::temp_dir().join(format!("xps5x-save-search-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        kernel.filesystem.set_savedata_directory(&root);
        assert!(mem.write(0x118, &0u32.to_le_bytes())); // sort key
        assert!(mem.write(0x11c, &0u32.to_le_bytes())); // order
        assert!(mem.write(0x208, &0x300u64.to_le_bytes()));
        assert!(mem.write(0x210, &4u32.to_le_bytes()));
        assert_eq!(hle_dir_name_search(&ctx, &[0x100, 0x200]), SCE_OK);
        let mut count = [0u8; 4];
        assert!(mem.read(0x200, &mut count));
        assert_eq!(u32::from_le_bytes(count), 0);

        std::fs::create_dir_all(root.join("slotA")).unwrap();
        assert_eq!(hle_dir_name_search(&ctx, &[0x100, 0x200]), SCE_OK);
        assert!(mem.read(0x200, &mut count));
        assert_eq!(u32::from_le_bytes(count), 1);
        let mut name = [0u8; 32];
        assert!(mem.read(0x300, &mut name));
        assert_eq!(&name[..5], b"slotA");

        let _ = std::fs::remove_dir_all(root);
    }
}
