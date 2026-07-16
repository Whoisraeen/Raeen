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
const ERROR_PARAMETER: u64 = 0x809F_0000;
const ERROR_EXISTS: u64 = 0x809F_0007;
const ERROR_NOT_FOUND: u64 = 0x809F_0008;
const ERROR_INTERNAL: u64 = 0x809F_000B;
const MOUNT_MODE_CREATE: u32 = 1 << 2;
const MOUNT_MODE_CREATE2: u32 = 1 << 5;
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
        registry.register(library, "sceSaveDataUmount", hle_unmount);
        registry.register(library, "sceSaveDataUmount2", hle_unmount);
        registry.register(library, "sceSaveDataDirNameSearch", hle_dir_name_search);
        registry.register(library, "sceSaveDataDirNameSearchPs4", hle_dir_name_search);
        registry.register(library, "sceSaveDataGetMountInfo", hle_get_mount_info);
        registry.register(library, "sceSaveDataDelete", hle_delete);
        registry.register(library, "sceSaveDataCommit", hle_commit);
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
    registry.register_nid(
        "libSceSaveData_native",
        "sceSaveDataCommit",
        0x89ee_ea85_9e17_d027,
        hle_commit,
    );
}

/// Commit all buffered file writes below a mounted save-data point.
///
/// The ABI takes a pointer to the 16-byte `SceSaveDataMountPoint` returned by
/// mount. The VFS normally persists writable files on close; commit is the
/// stronger durability boundary used while those descriptors are still open.
fn hle_commit(ctx: &HleContext, args: &[u64]) -> u64 {
    let mount_point = args.first().copied().unwrap_or(0);
    if mount_point == 0 {
        return ERROR_PARAMETER;
    }
    let mut request = [0u8; 64];
    if !ctx.mem.read(mount_point, &mut request) {
        return 0x8002_000E;
    }

    // Two ABI generations are in use. The older form passes the 16-byte
    // mount-point object directly. Native PS5 titles pass an opaque commit
    // descriptor whose leading u32 is the transaction resource returned by
    // `sceSaveDataCreateTransactionResource`; the mount is implicit in the
    // preceding prepare/mount operation.
    let direct_prefix = savedata_prefix(&request);
    let resource = i32::from_le_bytes(request[..4].try_into().expect("fixed slice"));
    if direct_prefix.is_none()
        && !ctx
            .kernel
            .save_data_transaction_resources
            .contains_key(&resource)
    {
        warn!(
            commit = mount_point,
            arg1 = args.get(1).copied().unwrap_or(0),
            arg2 = args.get(2).copied().unwrap_or(0),
            resource,
            bytes = ?request,
            "sceSaveDataCommit received an unknown request layout"
        );
        return 0x809F_000B;
    }

    // Direct form: flush that one mount. Resource form: the mount is
    // implicit, so flush every active save-data container — or the boot
    // `/savedata0` mapping when no slot is mounted (files can be open under
    // it directly).
    let mounts = match direct_prefix {
        Some(prefix) => vec![prefix],
        None => {
            let prefixes = ctx.kernel.filesystem.savedata_mount_prefixes();
            if prefixes.is_empty() {
                vec!["/savedata0".to_owned()]
            } else {
                prefixes
            }
        }
    };
    let mut flushed = 0usize;
    for mount in &mounts {
        match ctx.kernel.filesystem.sync_mount(mount) {
            Ok(count) => flushed += count,
            Err(error) => {
                warn!("sceSaveDataCommit('{mount}') failed: {error}");
                return 0x809F_000B;
            }
        }
    }
    ctx.kernel.save_data_transaction_resources.clear();
    debug!("sceSaveDataCommit({mounts:?}) -> flushed {flushed} descriptor(s)");
    SCE_OK
}

/// Parse a NUL-terminated `/savedataN` mount-point string from the head of a
/// 16-byte-or-larger request block. Returns `None` when the bytes are not a
/// save-data mount point (the native opaque-descriptor form).
fn savedata_prefix(request: &[u8]) -> Option<String> {
    let head = &request[..request.len().min(16)];
    let len = head.iter().position(|byte| *byte == 0)?;
    let text = std::str::from_utf8(&head[..len]).ok()?;
    let digits = text.strip_prefix("/savedata")?;
    (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| text.to_owned())
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
    let Some(mount) = savedata_prefix(&mount_bytes) else {
        return 0x809F_000B;
    };

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

/// Report capacity for a live save-data mount. The 48-byte result is two u64
/// block counts followed by reserved zeroes; callers commonly consume the
/// counts immediately after mount, so returning success without initializing
/// the buffer feeds arbitrary stack data into their storage policy.
fn hle_get_mount_info(ctx: &HleContext, args: &[u64]) -> u64 {
    let mount_point = args.first().copied().unwrap_or(0);
    let info = args.get(1).copied().unwrap_or(0);
    if mount_point == 0 || info == 0 {
        return ERROR_PARAMETER;
    }
    let mut mount_bytes = [0u8; 16];
    if !ctx.mem.read(mount_point, &mut mount_bytes) {
        return 0x8002_000E;
    }
    let valid = savedata_prefix(&mount_bytes)
        .is_some_and(|mount| ctx.kernel.filesystem.resolve_path(&mount).is_some());
    if !valid {
        return 0x809F_000B;
    }

    // 1,048,576 32-KiB blocks = 32 GiB. This is a deterministic virtual
    // quota, not a claim about host free space; actual writes remain bounded
    // by the host filesystem and surface their real I/O errors.
    let blocks = 1_048_576u64;
    let mut result = [0u8; 48];
    result[..8].copy_from_slice(&blocks.to_le_bytes());
    result[8..16].copy_from_slice(&blocks.to_le_bytes());
    if !ctx.mem.write(info, &result) {
        return 0x8002_000E;
    }
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
    let pattern_ptr = u64::from_le_bytes(cond_bytes[16..24].try_into().expect("fixed slice"));
    let pattern = if pattern_ptr == 0 {
        String::new()
    } else {
        let mut bytes = [0u8; 32];
        if !ctx.mem.read(pattern_ptr, &mut bytes) {
            return 0x8002_000E;
        }
        let len = bytes
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(bytes.len());
        match std::str::from_utf8(&bytes[..len]) {
            Ok(value) => value.to_owned(),
            Err(_) => return ERROR_PARAMETER,
        }
    };
    let root = ctx.kernel.filesystem.savedata_root();
    let mut names: Vec<String> = match std::fs::read_dir(root) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|ty| ty.is_dir()))
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| {
                !name.starts_with("sce_")
                    && name.len() < 32
                    && (pattern.is_empty() || save_name_matches_pattern(name, &pattern))
            })
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
    let slot_path = match ctx.kernel.filesystem.savedata_slot_path(&dir_name) {
        Ok(path) => path,
        Err(_) => return ERROR_PARAMETER,
    };
    let removal = if slot_path.exists() {
        std::fs::remove_dir_all(&slot_path)
    } else {
        Ok(())
    };
    match removal {
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

/// `sceSaveDataUmount(const SceSaveDataMountPoint*)`: unmount the specific
/// point named in the 16-byte argument. Titles hold several containers
/// mounted concurrently, so unmounting "whatever is mounted" corrupts the
/// others.
fn hle_unmount(ctx: &HleContext, args: &[u64]) -> u64 {
    // Two ABI generations: PS4-compat `Umount(const SceSaveDataMountPoint*)`
    // passes the pointer first; the measured native PS5 form passes it
    // second (arg0 is 0). Take the first argument that dereferences to a
    // `/savedataN` string.
    let prefix = args.iter().take(2).find_map(|&candidate| {
        if candidate == 0 {
            return None;
        }
        let mut mount_bytes = [0u8; 16];
        if !ctx.mem.read(candidate, &mut mount_bytes) {
            return None;
        }
        savedata_prefix(&mount_bytes)
    });
    let Some(prefix) = prefix else {
        warn!(
            args = ?&args[..args.len().min(4)],
            "sceSaveDataUmount: no argument dereferences to a /savedataN mount point"
        );
        return ERROR_PARAMETER;
    };
    if ctx.kernel.filesystem.unmount_savedata_slot(&prefix) {
        debug!("sceSaveDataUmount('{prefix}') -> OK");
        SCE_OK
    } else {
        warn!("sceSaveDataUmount('{prefix}'): not a mounted save-data point");
        ERROR_NOT_FOUND
    }
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

    let mut request = [0u8; 0x2c];
    if !ctx.mem.read(mount_ptr, &mut request) {
        return 0x8002_000E;
    }
    let user_id = i32::from_le_bytes(request[..4].try_into().expect("fixed slice"));
    let dir_name_ptr = u64::from_le_bytes(request[0x08..0x10].try_into().expect("fixed slice"));
    let blocks = u64::from_le_bytes(request[0x10..0x18].try_into().expect("fixed slice"));
    let system_blocks = u64::from_le_bytes(request[0x18..0x20].try_into().expect("fixed slice"));
    let mount_mode = u32::from_le_bytes(request[0x20..0x24].try_into().expect("fixed slice"));
    let resource = u32::from_le_bytes(request[0x24..0x28].try_into().expect("fixed slice"));
    let mode = u32::from_le_bytes(request[0x28..0x2c].try_into().expect("fixed slice"));
    if user_id < 0 || dir_name_ptr == 0 {
        return ERROR_PARAMETER;
    }

    let mut dir_name_bytes = [0u8; 32];
    if !ctx.mem.read(dir_name_ptr, &mut dir_name_bytes) {
        return 0x8002_000E;
    }
    let dir_name_len = dir_name_bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(dir_name_bytes.len());
    let dir_name = match std::str::from_utf8(&dir_name_bytes[..dir_name_len]) {
        Ok(value) if !value.trim().is_empty() => value,
        _ => return ERROR_PARAMETER,
    };
    let slot_path = match ctx.kernel.filesystem.savedata_slot_path(dir_name) {
        Ok(path) => path,
        Err(_) => return ERROR_PARAMETER,
    };
    let existed = slot_path.is_dir();
    let create = mount_mode & MOUNT_MODE_CREATE != 0;
    let create_if_missing = mount_mode & MOUNT_MODE_CREATE2 != 0;
    if !existed && !create && !create_if_missing {
        return ERROR_NOT_FOUND;
    }
    if existed && create {
        return ERROR_EXISTS;
    }
    if !existed && std::fs::create_dir_all(&slot_path).is_err() {
        return ERROR_INTERNAL;
    }
    let prefix = match ctx.kernel.filesystem.mount_savedata_slot(dir_name) {
        Ok((prefix, _path)) => prefix,
        Err(error) => {
            warn!("sceSaveDataMount3('{dir_name}') failed: {error}");
            return ERROR_INTERNAL;
        }
    };

    // SceSaveDataMountResult (64 bytes): mountPoint.data[16] at offset 0,
    // then mountStatus/requiredBlocks/... (left zero).
    let mut result = [0u8; MOUNT_RESULT_SIZE];
    result[..prefix.len().min(15)].copy_from_slice(&prefix.as_bytes()[..prefix.len().min(15)]);
    result[0x1c..0x20].copy_from_slice(&u32::from(create_if_missing && !existed).to_le_bytes());
    // The byte after the prefix stays 0 — the NUL terminator.
    if !ctx.mem.write(result_ptr, &result) {
        warn!("sceSaveDataMount: result out-ptr {result_ptr:#x} not writable");
        return ERROR_PARAMETER;
    }
    debug!(
        "sceSaveDataMount3(user={user_id}, dir='{dir_name}', blocks={blocks}, system_blocks={system_blocks}, mount_mode={mount_mode:#x}, resource={resource}, mode={mode}) -> {prefix}"
    );
    SCE_OK
}

fn save_name_matches_pattern(value: &str, pattern: &str) -> bool {
    fn matches(value: &[u8], pattern: &[u8]) -> bool {
        let Some((&head, tail)) = pattern.split_first() else {
            return value.is_empty();
        };
        if head == b'%' {
            return (0..=value.len()).any(|index| matches(&value[index..], tail));
        }
        let Some((&value_head, value_tail)) = value.split_first() else {
            return false;
        };
        (head == b'_' || head.eq_ignore_ascii_case(&value_head)) && matches(value_tail, tail)
    }

    matches(value.as_bytes(), pattern.as_bytes())
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
        let root =
            std::env::temp_dir().join(format!("xps5x-save-mount-slot-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        kernel.filesystem.set_savedata_directory(&root);

        // Native Mount3: user id, dirName*, block counts and Create2 mode.
        assert!(mem.write(0x100, &1i32.to_le_bytes()));
        assert!(mem.write(0x108, &0x300u64.to_le_bytes()));
        assert!(mem.write(0x120, &MOUNT_MODE_CREATE2.to_le_bytes()));
        assert!(mem.write(0x300, b"slot00000001@world\0"));
        assert_eq!(hle_mount(&ctx, &[0x100, 0x200]), SCE_OK);
        let mut res = [0u8; MOUNT_RESULT_SIZE];
        assert!(mem.read(0x200, &mut res));
        let nul = res.iter().position(|&b| b == 0).unwrap();
        assert_eq!(&res[..nul], MOUNT_POINT, "mount point must be /savedata0");
        assert_eq!(u32::from_le_bytes(res[0x1c..0x20].try_into().unwrap()), 1);
        assert_eq!(
            kernel.filesystem.resolve_path("/savedata0/level.dat"),
            Some(root.join("slot00000001@world/level.dat"))
        );
        assert!(root.join("slot00000001@world").is_dir());
        // Umount takes the SceSaveDataMountPoint written by mount.
        assert!(mem.write(0x400, b"/savedata0\0"));
        assert_eq!(hle_unmount(&ctx, &[0x400]), SCE_OK);
        assert_eq!(
            kernel.filesystem.resolve_path("/savedata0"),
            Some(root.clone())
        );
        // A second unmount of the same point reports it isn't mounted.
        assert_eq!(hle_unmount(&ctx, &[0x400]), ERROR_NOT_FOUND);
        assert_eq!(hle_unmount(&ctx, &[0]), ERROR_PARAMETER);

        let _ = std::fs::remove_dir_all(root);
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
    fn mount_info_initializes_capacity_and_native_search_is_not_a_noop() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root =
            std::env::temp_dir().join(format!("xps5x-save-mount-info-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        kernel.filesystem.set_savedata_directory(&root);
        assert!(mem.write(0x10, b"/savedata0\0"));
        assert!(mem.write(0x100, &[0xCC; 48]));

        assert_eq!(hle_get_mount_info(&ctx, &[0x10, 0x100]), SCE_OK);
        let mut info = [0u8; 48];
        assert!(mem.read(0x100, &mut info));
        assert_eq!(u64::from_le_bytes(info[..8].try_into().unwrap()), 1_048_576);
        assert_eq!(
            u64::from_le_bytes(info[8..16].try_into().unwrap()),
            1_048_576
        );
        assert_eq!(&info[16..], &[0; 32]);

        let registry = HleRegistry::new();
        register(&registry);
        assert!(registry.registered_names().iter().any(|(library, name)| {
            library == "libSceSaveData" && name == "sceSaveDataDirNameSearch"
        }));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn commit_flushes_open_save_files_and_pins_the_native_nid() {
        use xps5x_kernel::filesystem::open_flags::{O_CREAT, O_TRUNC, O_WRONLY};

        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root = std::env::temp_dir().join(format!("xps5x-save-commit-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        kernel.filesystem.set_savedata_directory(&root);
        assert!(mem.write(0x10, b"/savedata0\0"));

        let fd = kernel
            .filesystem
            .open(
                "/savedata0/slot/level.dat",
                O_WRONLY | O_CREAT | O_TRUNC,
                0o644,
            )
            .unwrap();
        kernel.filesystem.write(fd, b"FRAME").unwrap();
        assert!(!root.join("slot/level.dat").exists());

        assert_eq!(hle_commit(&ctx, &[0x10]), SCE_OK);
        assert_eq!(
            std::fs::read(root.join("slot/level.dat")).unwrap(),
            b"FRAME"
        );

        // Native PS5 callers pass an opaque commit descriptor whose leading
        // field is the transaction-resource id, rather than a mount-point
        // string. Minecraft uses this form.
        let resource = hle_create_transaction_resource(&ctx, &[0x0200_0000]);
        assert!(mem.write(0x30, &(resource as u32).to_le_bytes()));
        kernel.filesystem.write(fd, b" TWO").unwrap();
        assert_eq!(hle_commit(&ctx, &[0x30, 0xDEAD_BEEF, 4]), SCE_OK);
        assert_eq!(
            std::fs::read(root.join("slot/level.dat")).unwrap(),
            b"FRAME TWO"
        );
        kernel.filesystem.close(fd).unwrap();

        let registry = HleRegistry::new();
        register(&registry);
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0x89ee_ea85_9e17_d027
                        && key == "libSceSaveData_native::sceSaveDataCommit"
                })
        );

        let _ = std::fs::remove_dir_all(root);
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
        std::fs::create_dir_all(root.join("games")).unwrap();
        assert!(mem.write(0x110, &0x380u64.to_le_bytes()));
        assert!(mem.write(0x380, b"slot%\0"));
        assert_eq!(hle_dir_name_search(&ctx, &[0x100, 0x200]), SCE_OK);
        assert!(mem.read(0x200, &mut count));
        assert_eq!(u32::from_le_bytes(count), 1);
        let mut name = [0u8; 32];
        assert!(mem.read(0x300, &mut name));
        assert_eq!(&name[..5], b"slotA");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn save_directory_patterns_support_percent_and_single_character_wildcards() {
        assert!(save_name_matches_pattern("slot00000001@world", "slot%"));
        assert!(save_name_matches_pattern("SLOT-A", "slot__"));
        assert!(save_name_matches_pattern(
            "000000000001@save",
            "____________@%"
        ));
        assert!(!save_name_matches_pattern("games", "____________@%"));
    }
}
