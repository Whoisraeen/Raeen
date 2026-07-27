//! HLE libSceSaveData — save-data mount management.
//!
//! A title saves progress by `sceSaveDataMount`-ing a save slot, which
//! returns a mount-point path (`/savedata0`); the title then opens/writes
//! files under that path, and unmounts. Raeen already exposes a
//! `/savedata0/` VFS mount backed by a host `savedata` directory, and the
//! file I/O path persists writes on close (see `raeen-kernel`'s VFS +
//! `libkernel` write). So mounting here just hands back the standard mount
//! point and the whole save→file→persist chain works end to end. The mount
//! result layout (64 bytes, mount-point string at offset 0) and the flat
//! `/savedata0` point are cross-checked against SharpEmu's `SaveDataExports`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::Ordering;
use tracing::{debug, info, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `SCE_SAVE_DATA_ERROR_PARAMETER`.
const ERROR_PARAMETER: u64 = 0x809F_0000;
const ERROR_EXISTS: u64 = 0x809F_0007;
const ERROR_NOT_FOUND: u64 = 0x809F_0008;
const ERROR_INTERNAL: u64 = 0x809F_000B;
/// `SCE_SAVE_DATA_ERROR_MEMORY_NOT_READY` — save-memory used before setup.
const ERROR_MEMORY_NOT_READY: u64 = 0x809F_0012;
/// Save-memory blob ceiling (SharpEmu: 64 MiB).
const SAVE_DATA_MEMORY_MAX: u64 = 64 * 1024 * 1024;
/// Deterministic virtual quota exposed through the save-data ABI.
///
/// PS5-facing Kyty uses 16,384 32-KiB blocks (512 MiB). Advertising a much
/// larger synthetic value is not harmless: Minecraft converts this value
/// through a 32-bit byte count in its storage policy, so Raeen's former
/// 32-GiB value overflowed and was reported as "almost out of storage".
const SAVE_DATA_BLOCK_SIZE: u64 = 32 * 1024;
const SAVE_DATA_TOTAL_BLOCKS: u64 = 16_384;
const SAVE_DATA_PARAM_SIZE: usize = 0x530;
const SAVE_DATA_SEARCH_INFO_SIZE: usize = 48;
const SAVE_DATA_MOUNT3_HEAD_SIZE: usize = 0x2c;
const MOUNT_MODE_CREATE: u32 = 1 << 2;
const MOUNT_MODE_CREATE2: u32 = 1 << 5;
/// `SceSaveDataMountResult` size in bytes.
const MOUNT_RESULT_SIZE: usize = 0x40;

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
        // Save-slot icon (the thumbnail shown in the system save browser).
        // Minecraft writes one while creating a world: leaving it unresolved
        // killed the save worker mid-write, and the world-load then waited
        // forever on a thread that no longer existed.
        registry.register(library, "sceSaveDataSaveIcon", hle_save_icon);
        registry.register(library, "sceSaveDataLoadIcon", hle_load_icon);
        // Mountless per-user "save data memory" blob API.
        registry.register(
            library,
            "sceSaveDataSetupSaveDataMemory2",
            hle_setup_save_data_memory2,
        );
        registry.register(
            library,
            "sceSaveDataGetSaveDataMemory2",
            hle_get_save_data_memory2,
        );
        registry.register(
            library,
            "sceSaveDataSetSaveDataMemory2",
            hle_set_save_data_memory2,
        );
        registry.register(
            library,
            "sceSaveDataSyncSaveDataMemory",
            hle_sync_save_data_memory,
        );
        registry.register(
            library,
            "sceSaveDataTransferringMount",
            hle_transferring_mount,
        );
        registry.register(
            library,
            "sceSaveDataTransferringMountPs4",
            hle_transferring_mount,
        );
    }
    registry.register(
        "libSceSaveData_native",
        "sceSaveDataCreateTransactionResource",
        hle_create_transaction_resource,
    );
    registry.register(
        "libSceSaveData_native",
        "sceSaveDataDeleteTransactionResource",
        hle_delete_transaction_resource,
    );
    registry.register("libSceSaveData_native", "sceSaveDataPrepare", hle_prepare);
    registry.register(
        "libSceSaveData_native",
        "sceSaveDataSetParam",
        hle_set_param,
    );
    registry.register("libSceSaveData_native", "sceSaveDataCommit", hle_commit);
    registry.register(
        "libSceSaveData_native",
        "sceSaveDataSetupSaveDataMemory2",
        hle_setup_save_data_memory2,
    );
}

/// Read a little-endian `u32`/`u64` from guest memory (save-memory helpers).
fn sdm_u32(ctx: &HleContext, addr: u64) -> Option<u32> {
    let mut b = [0u8; 4];
    ctx.mem.read(addr, &mut b).then(|| u32::from_le_bytes(b))
}
fn sdm_u64(ctx: &HleContext, addr: u64) -> Option<u64> {
    let mut b = [0u8; 8];
    ctx.mem.read(addr, &mut b).then(|| u64::from_le_bytes(b))
}

/// `sceSaveDataSetupSaveDataMemory2(setup*, result*)`: size the per-user save
/// blob to `setup.memorySize` (zero-filled, grown only), and report the
/// pre-existing size via `*result`. `SceSaveDataMemorySetup2` = `{ u32 option;
/// i32 userId; u64 memorySize; ... }`. Ported from SharpEmu
/// `SaveDataSetupSaveDataMemory2` (GPL-2.0).
fn hle_setup_save_data_memory2(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    let result = args.get(1).copied().unwrap_or(0);
    if param == 0 {
        return ERROR_PARAMETER;
    }
    let (Some(user_id), Some(size)) = (sdm_u32(ctx, param + 0x04), sdm_u64(ctx, param + 0x08))
    else {
        return ERROR_INTERNAL;
    };
    let user_id = user_id as i32;
    if user_id < 0 || size == 0 || size > SAVE_DATA_MEMORY_MAX {
        return ERROR_PARAMETER;
    }
    let existed = ctx
        .kernel
        .save_data_memory
        .get(&user_id)
        .map_or(0u64, |b| b.len() as u64);
    // Write the result first: a faulted result pointer must not leave grown
    // setup state behind (mirrors SharpEmu's ordering).
    if result != 0 && !ctx.mem.write(result, &existed.to_le_bytes()) {
        return ERROR_INTERNAL;
    }
    let mut blob = ctx.kernel.save_data_memory.entry(user_id).or_default();
    if (blob.len() as u64) < size {
        blob.resize(size as usize, 0);
    }
    debug!("sceSaveDataSetupSaveDataMemory2 user={user_id} size={size:#x} existed={existed:#x}");
    SCE_OK
}

/// Shared body for `Get`/`Set`SaveDataMemory2: transfer between the per-user
/// blob and a guest buffer. Request = `{ i32 userId; u8 pad[4]; SceSaveDataMemoryData* data }`;
/// `SceSaveDataMemoryData` = `{ void* buf; u64 bufSize; i64 offset }`.
fn transfer_save_data_memory(ctx: &HleContext, args: &[u64], write: bool) -> u64 {
    let request = args.first().copied().unwrap_or(0);
    if request == 0 {
        return ERROR_PARAMETER;
    }
    let (Some(user_id), Some(data)) = (sdm_u32(ctx, request), sdm_u64(ctx, request + 0x08)) else {
        return ERROR_INTERNAL;
    };
    let user_id = user_id as i32;
    if user_id < 0 {
        return ERROR_PARAMETER;
    }
    let Some(mut blob) = ctx.kernel.save_data_memory.get_mut(&user_id) else {
        return ERROR_MEMORY_NOT_READY;
    };
    if data == 0 {
        return SCE_OK; // a NULL data descriptor is a no-op success (SharpEmu)
    }
    let (Some(buf), Some(buf_size), Some(offset)) = (
        sdm_u64(ctx, data),
        sdm_u64(ctx, data + 0x08),
        sdm_u64(ctx, data + 0x10),
    ) else {
        return ERROR_INTERNAL;
    };
    let len = blob.len() as u64;
    if buf == 0 || buf_size > len || offset > len - buf_size {
        return ERROR_PARAMETER;
    }
    let (start, end) = (offset as usize, (offset + buf_size) as usize);
    if write {
        let mut tmp = vec![0u8; buf_size as usize];
        if !ctx.mem.read(buf, &mut tmp) {
            return ERROR_INTERNAL;
        }
        blob[start..end].copy_from_slice(&tmp);
    } else if !ctx.mem.write(buf, &blob[start..end]) {
        return ERROR_INTERNAL;
    }
    SCE_OK
}

/// `sceSaveDataGetSaveDataMemory2(get*)`: copy blob → guest buffer.
fn hle_get_save_data_memory2(ctx: &HleContext, args: &[u64]) -> u64 {
    transfer_save_data_memory(ctx, args, false)
}

/// `sceSaveDataSetSaveDataMemory2(set*)`: copy guest buffer → blob.
fn hle_set_save_data_memory2(ctx: &HleContext, args: &[u64]) -> u64 {
    transfer_save_data_memory(ctx, args, true)
}

/// `sceSaveDataSyncSaveDataMemory(sync*)`: writes go straight to the blob, so a
/// ready state is all sync confirms. `SceSaveDataMemorySync` starts with `i32 userId`.
fn hle_sync_save_data_memory(ctx: &HleContext, args: &[u64]) -> u64 {
    let sync = args.first().copied().unwrap_or(0);
    if sync == 0 {
        return ERROR_PARAMETER;
    }
    let Some(user_id) = sdm_u32(ctx, sync) else {
        return ERROR_INTERNAL;
    };
    if (user_id as i32) < 0 {
        return ERROR_PARAMETER;
    }
    if ctx.kernel.save_data_memory.contains_key(&(user_id as i32)) {
        SCE_OK
    } else {
        ERROR_MEMORY_NOT_READY
    }
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
        // FAIL OPEN, not closed. Refusing an unrecognized layout with
        // `ERROR_INTERNAL` fails a commit whose data the VFS has usually
        // already persisted, and a title that treats commit failure as
        // "the save did not land" then retries or waits forever. Measured on
        // Minecraft's world creation: this fired with `resource=4` (a
        // transaction id it never obtained from
        // `sceSaveDataCreateTransactionResource`, so a third ABI generation)
        // immediately before the world-load stalled. Flushing every active
        // container is exactly what the resource form below does, and the
        // worst case of doing it for an unknown layout is a redundant sync.
        // The warning stays — it names the layout so the real shape can be
        // decoded — but it is now rate-limited and non-fatal.
        static UNKNOWN_LAYOUT_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !UNKNOWN_LAYOUT_WARNED.swap(true, Ordering::Relaxed) {
            warn!(
                commit = mount_point,
                arg1 = args.get(1).copied().unwrap_or(0),
                arg2 = args.get(2).copied().unwrap_or(0),
                resource,
                bytes = ?request,
                "sceSaveDataCommit received an unknown request layout — flushing every \
                 active save container anyway (fail-open); further occurrences are silent"
            );
        }
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
/// Largest icon Raeen will move in either direction. A save-slot thumbnail is
/// a small PNG (the system browser shows it at 228x128); this only exists so a
/// wild `bufSize`/`dataSize` cannot drive a giant allocation or write.
const SAVE_DATA_ICON_MAX: u64 = 8 * 1024 * 1024;

/// Guest path of a mount's icon, relative to its mount point.
///
/// The save slot's `sce_sys` directory is the same place the package keeps its
/// metadata, and `icon0.png` is the standard name (shadPS4 resolves the same
/// file through `SaveInstance::GetIconPath`).
fn icon_guest_path(mount_prefix: &str) -> String {
    format!("{mount_prefix}/sce_sys/icon0.png")
}

/// Read the `SceSaveDataMountPoint*` argument into its `/savedataN` prefix.
fn mount_point_arg(ctx: &HleContext, ptr: u64) -> Option<String> {
    if ptr == 0 {
        return None;
    }
    let mut raw = [0u8; 16];
    ctx.mem.read(ptr, &mut raw).then_some(())?;
    savedata_prefix(&raw)
}

/// `SceSaveDataIcon` = `{ void *buf; size_t bufSize; size_t dataSize; u8 _[32] }`
/// (shadPS4 `savedata.cpp:128-133`). Returns `(buf, bufSize, dataSize)`.
fn read_icon_struct(ctx: &HleContext, ptr: u64) -> Option<(u64, u64, u64)> {
    let mut raw = [0u8; 24];
    if ptr == 0 || !ctx.mem.read(ptr, &mut raw) {
        return None;
    }
    let field = |i: usize| u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().expect("fixed"));
    Some((field(0), field(1), field(2)))
}

/// `sceSaveDataSaveIcon(const SceSaveDataMountPoint *mountPoint,
/// const SceSaveDataIcon *icon)`: write the slot's thumbnail.
///
/// Ports shadPS4's `sceSaveDataSaveIcon` (`savedata.cpp:1372-1405`,
/// GPL-2.0): validate the pointers, then write `min(bufSize, dataSize)` bytes
/// from the guest buffer to the mount's `sce_sys/icon0.png`.
///
/// MEASURED: Minecraft calls this while creating a world (`rdi -> "/savedata1"`).
/// It was unresolved, so the call landed on the stub guard page and killed the
/// save worker *while it held locks* — the runtime released them, but the
/// world-load then polled a save that could never complete (a clean 11 s
/// heartbeat on `/savedata0` forever, while the GPU kept presenting).
fn hle_save_icon(ctx: &HleContext, args: &[u64]) -> u64 {
    let mount_ptr = args.first().copied().unwrap_or(0);
    let icon_ptr = args.get(1).copied().unwrap_or(0);
    let Some((buf, buf_size, data_size)) = read_icon_struct(ctx, icon_ptr) else {
        return ERROR_PARAMETER;
    };
    if buf == 0 {
        return ERROR_PARAMETER;
    }
    let Some(prefix) = mount_point_arg(ctx, mount_ptr) else {
        return ERROR_PARAMETER;
    };
    let len = buf_size.min(data_size).min(SAVE_DATA_ICON_MAX);
    let Ok(len_usize) = usize::try_from(len) else {
        return ERROR_PARAMETER;
    };
    let mut bytes = vec![0u8; len_usize];
    if len_usize != 0 && !ctx.mem.read(buf, &mut bytes) {
        return ERROR_PARAMETER;
    }
    let guest_path = icon_guest_path(&prefix);
    let Some(host_path) = ctx.kernel.filesystem.resolve_path(&guest_path) else {
        warn!("sceSaveDataSaveIcon: cannot resolve {guest_path}");
        return ERROR_NOT_FOUND;
    };
    if let Some(parent) = host_path.parent()
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        warn!(
            "sceSaveDataSaveIcon: cannot create {}: {error}",
            parent.display()
        );
        return ERROR_INTERNAL;
    }
    match std::fs::write(&host_path, &bytes) {
        Ok(()) => {
            debug!("sceSaveDataSaveIcon({prefix}) -> {len} byte(s)");
            SCE_OK
        }
        Err(error) => {
            warn!(
                "sceSaveDataSaveIcon: write {} failed: {error}",
                host_path.display()
            );
            ERROR_INTERNAL
        }
    }
}

/// `sceSaveDataLoadIcon(const SceSaveDataMountPoint *mountPoint,
/// SceSaveDataIcon *icon)`: read the slot's thumbnail back.
///
/// Ports shadPS4's `sceSaveDataLoadIcon` (`savedata.cpp:1227-1254` +
/// `OrbisSaveDataIcon::LoadIcon`): set `dataSize` to the file's real size and
/// copy `min(bufSize, dataSize)` bytes into the guest buffer. A slot with no
/// icon yet is `NOT_FOUND`, not a fabricated image.
fn hle_load_icon(ctx: &HleContext, args: &[u64]) -> u64 {
    let mount_ptr = args.first().copied().unwrap_or(0);
    let icon_ptr = args.get(1).copied().unwrap_or(0);
    let Some((buf, buf_size, _)) = read_icon_struct(ctx, icon_ptr) else {
        return ERROR_PARAMETER;
    };
    if buf == 0 {
        return ERROR_PARAMETER;
    }
    let Some(prefix) = mount_point_arg(ctx, mount_ptr) else {
        return ERROR_PARAMETER;
    };
    let guest_path = icon_guest_path(&prefix);
    let Some(host_path) = ctx.kernel.filesystem.resolve_path(&guest_path) else {
        return ERROR_NOT_FOUND;
    };
    let Ok(bytes) = std::fs::read(&host_path) else {
        debug!("sceSaveDataLoadIcon({prefix}): no icon yet");
        return ERROR_NOT_FOUND;
    };
    // `dataSize` reports the file's REAL size even when the guest buffer is
    // shorter — that is how a caller learns the buffer it must allocate.
    let data_size = bytes.len() as u64;
    if !ctx.mem.write(icon_ptr + 16, &data_size.to_le_bytes()) {
        return ERROR_PARAMETER;
    }
    let copy = buf_size.min(data_size).min(SAVE_DATA_ICON_MAX) as usize;
    if copy != 0 && !ctx.mem.write(buf, &bytes[..copy]) {
        return ERROR_PARAMETER;
    }
    debug!("sceSaveDataLoadIcon({prefix}) -> {copy} of {data_size} byte(s)");
    SCE_OK
}

fn savedata_prefix(request: &[u8]) -> Option<String> {
    let head = &request[..request.len().min(16)];
    let len = head.iter().position(|byte| *byte == 0)?;
    let text = std::str::from_utf8(&head[..len]).ok()?;
    let digits = text.strip_prefix("/savedata")?;
    (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| text.to_owned())
}

/// Bedrock keeps its console save container mounted through `libSceSaveData`
/// while its storage adapter performs ordinary LevelDB I/O through
/// `/minecraftWorlds/<world-id>`.
///
/// MEASURED (Minecraft PPSA17221, 2026-07-26): mounting
/// `BedrockWorldxDe5FqCkuF8@P1` at `/savedata1` is followed immediately by
/// opens below `/minecraftWorlds/xDe5FqCkuF8@/db`. The service mount result
/// contains only `/savedata1`; this second path is the process-lifetime
/// application-filesystem view that the save worker populates, unmounts, and
/// then hands to LevelDB. Keep the convention narrow so arbitrary unmounted
/// guest paths remain denied by the VFS.
fn bedrock_world_guest_alias(dir_name: &str) -> Option<String> {
    let encoded = dir_name.strip_prefix("BedrockWorld")?;
    let marker = encoded.rfind('P')?;
    let (world_id, player) = encoded.split_at(marker);
    let player = player.strip_prefix('P')?;
    if world_id.is_empty()
        || !world_id.ends_with('@')
        || player.is_empty()
        || !player.bytes().all(|byte| byte.is_ascii_digit())
        || world_id.contains(['/', '\\', '\0'])
        || world_id.contains("..")
    {
        return None;
    }
    Some(format!("/minecraftWorlds/{world_id}"))
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
    let Some(host_path) =
        savedata_prefix(&mount_bytes).and_then(|mount| ctx.kernel.filesystem.resolve_path(&mount))
    else {
        return 0x809F_000B;
    };

    // This is a deterministic virtual quota, not a claim about host free
    // space; actual writes still surface their real host I/O errors.
    let Ok(result) = save_data_capacity_info(&host_path) else {
        return ERROR_INTERNAL;
    };
    if std::env::var_os("RAEEN_TRACE_SAVEDATA").is_some() {
        let blocks = u64::from_le_bytes(result[..8].try_into().expect("fixed slice"));
        let free_blocks = u64::from_le_bytes(result[8..16].try_into().expect("fixed slice"));
        info!(
            mount = %String::from_utf8_lossy(
                &mount_bytes[..mount_bytes.iter().position(|byte| *byte == 0).unwrap_or(16)]
            ),
            path = %host_path.display(),
            blocks,
            free_blocks,
            used_blocks = blocks.saturating_sub(free_blocks),
            "sceSaveDataGetMountInfo capacity"
        );
    }
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
    let params_out = u64::from_le_bytes(result_bytes[24..32].try_into().expect("fixed slice"));
    let infos_out = u64::from_le_bytes(result_bytes[32..40].try_into().expect("fixed slice"));
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
    let mut names: Vec<String> = match std::fs::read_dir(&root) {
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
    if std::env::var_os("RAEEN_TRACE_SAVEDATA").is_some() {
        info!(
            hits = names.len(),
            set_count,
            capacity,
            params = format_args!("{params_out:#x}"),
            infos = format_args!("{infos_out:#x}"),
            "sceSaveDataDirNameSearch result"
        );
    }
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
    for (index, name) in names.iter().take(set_count).enumerate() {
        let mut encoded = [0u8; 32];
        encoded[..name.len()].copy_from_slice(name.as_bytes());
        if !ctx.mem.write(names_out + index as u64 * 32, &encoded) {
            return 0x8002_000E;
        }
        if params_out != 0
            && !ctx.mem.write(
                params_out + index as u64 * SAVE_DATA_PARAM_SIZE as u64,
                &[0; SAVE_DATA_PARAM_SIZE],
            )
        {
            return 0x8002_000E;
        }
        if infos_out != 0 {
            let info = match save_data_capacity_info(&root.join(name)) {
                Ok(info) => info,
                Err(error) => {
                    warn!("sceSaveDataDirNameSearch: failed to size '{name}': {error}");
                    return ERROR_INTERNAL;
                }
            };
            if !ctx.mem.write(
                infos_out + index as u64 * SAVE_DATA_SEARCH_INFO_SIZE as u64,
                &info,
            ) {
                return 0x8002_000E;
            }
        }
    }
    debug!("sceSaveDataDirNameSearchPs4 -> {set_count} entr(ies)");
    SCE_OK
}

fn save_data_capacity_info(
    path: &std::path::Path,
) -> std::io::Result<[u8; SAVE_DATA_SEARCH_INFO_SIZE]> {
    let used_bytes = save_data_directory_bytes(path)?;
    let used_blocks = used_bytes
        .saturating_add(SAVE_DATA_BLOCK_SIZE - 1)
        .checked_div(SAVE_DATA_BLOCK_SIZE)
        .unwrap_or(0);
    let free_blocks = SAVE_DATA_TOTAL_BLOCKS.saturating_sub(used_blocks);
    let mut info = [0u8; SAVE_DATA_SEARCH_INFO_SIZE];
    info[..8].copy_from_slice(&SAVE_DATA_TOTAL_BLOCKS.to_le_bytes());
    info[8..16].copy_from_slice(&free_blocks.to_le_bytes());
    Ok(info)
}

fn save_data_directory_bytes(path: &std::path::Path) -> std::io::Result<u64> {
    let mut bytes = 0u64;
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            bytes = bytes.saturating_add(save_data_directory_bytes(&entry.path())?);
        } else if file_type.is_file() {
            bytes = bytes.saturating_add(entry.metadata()?.len());
        }
        // Symlinks and other special files are deliberately not followed:
        // save accounting must remain inside the title's sandbox.
    }
    Ok(bytes)
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
/// `sceSaveDataTransferringMount(const SceSaveDataTransferringMount *mount,
/// SceSaveDataMountResult *result)`: mount ANOTHER title's save data for a
/// cross-title transfer (shadPS4 `savedata.cpp:1685` — mount carries a
/// foreign titleId/dirName/fingerprint). Raeen's save-data host map is
/// strictly per-title, so there is never foreign save data to offer: report
/// `NOT_FOUND` (this module's error family), which titles treat as "nothing
/// to import" and fall back from. A NULL mount is a parameter error, matching
/// shadPS4's validation order.
fn hle_transferring_mount(_ctx: &HleContext, args: &[u64]) -> u64 {
    let mount_ptr = args.first().copied().unwrap_or(0);
    if mount_ptr == 0 {
        return ERROR_PARAMETER;
    }
    debug!(
        "sceSaveDataTransferringMount(mount={mount_ptr:#x}) -> NOT_FOUND (no cross-title save \
         data modeled)"
    );
    ERROR_NOT_FOUND
}

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

#[derive(Debug, PartialEq, Eq)]
struct SaveDataMount3Head {
    user_id: i32,
    dir_name_ptr: u64,
    blocks: u64,
    system_blocks: u64,
    mount_mode: u32,
    resource: i32,
}

fn parse_save_data_mount3_head(request: &[u8; SAVE_DATA_MOUNT3_HEAD_SIZE]) -> SaveDataMount3Head {
    SaveDataMount3Head {
        user_id: i32::from_le_bytes(request[..4].try_into().expect("fixed slice")),
        dir_name_ptr: u64::from_le_bytes(request[0x08..0x10].try_into().expect("fixed slice")),
        blocks: u64::from_le_bytes(request[0x10..0x18].try_into().expect("fixed slice")),
        system_blocks: u64::from_le_bytes(request[0x18..0x20].try_into().expect("fixed slice")),
        mount_mode: u32::from_le_bytes(request[0x20..0x24].try_into().expect("fixed slice")),
        // Native PS5 Mount3 has a four-byte pad at 0x24 and the signed
        // transaction resource at 0x28. Treating the pad as `resource` hid
        // the real handle from diagnostics and disagreed with the ABI used by
        // KytyPS5.
        resource: i32::from_le_bytes(request[0x28..0x2c].try_into().expect("fixed slice")),
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

    let mut request = [0u8; SAVE_DATA_MOUNT3_HEAD_SIZE];
    if !ctx.mem.read(mount_ptr, &mut request) {
        return 0x8002_000E;
    }
    let SaveDataMount3Head {
        user_id,
        dir_name_ptr,
        blocks,
        system_blocks,
        mount_mode,
        resource,
    } = parse_save_data_mount3_head(&request);
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
    let (prefix, slot_path) = match ctx.kernel.filesystem.mount_savedata_slot(dir_name) {
        Ok(mount) => mount,
        Err(error) => {
            warn!("sceSaveDataMount3('{dir_name}') failed: {error}");
            return ERROR_INTERNAL;
        }
    };
    if let Some(alias) = bedrock_world_guest_alias(dir_name) {
        // This view deliberately outlives `/savedataN`: Minecraft commits and
        // unmounts the service container before its LevelDB thread opens the
        // application path. The VFS itself is process-local, so the alias
        // cannot leak into another title.
        ctx.kernel
            .filesystem
            .mount(&alias, &slot_path.to_string_lossy());
        debug!("sceSaveDataMount3('{dir_name}') -> persistent alias {alias}");
    }

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
        "sceSaveDataMount3(user={user_id}, dir='{dir_name}', blocks={blocks}, system_blocks={system_blocks}, mount_mode={mount_mode:#x}, resource={resource}) -> {prefix}"
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

    /// Minecraft's world creation calls `sceSaveDataSaveIcon`; it must be
    /// registered under the name whose NID the title imports
    /// (`0x73cf18cb9e0cc74c` / `c88Yy54Mx0w`, confirmed against shadPS4's
    /// `savedata.cpp:1840`), or the call lands on the stub guard page and
    /// kills the save worker mid-write. `LoadIcon` is its twin
    /// (`0x7068cedf0337576f` / `cGjO3wM3V28`, `savedata.cpp:1824`).
    #[test]
    fn save_and_load_icon_are_registered_under_both_libraries() {
        let reg = HleRegistry::new();
        for library in ["libSceSaveData", "libSceSaveData_native"] {
            assert!(
                reg.is_implemented(library, "sceSaveDataSaveIcon"),
                "{library}::sceSaveDataSaveIcon must resolve"
            );
        }
        assert!(reg.is_implemented("libSceSaveData", "sceSaveDataLoadIcon"));
    }

    /// Round-trip: saving an icon writes it under the mount, and loading it
    /// back reports the file's REAL size in `dataSize` (that is how a caller
    /// with a short buffer learns what to allocate) while copying only what
    /// fits. A slot with no icon is `NOT_FOUND`, never a fabricated image.
    #[test]
    fn save_icon_round_trips_and_reports_the_real_size() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let dir = std::env::temp_dir().join(format!("raeen-icon-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp savedata root");
        kernel.filesystem.set_savedata_directory(&dir);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // mountPoint = "/savedata0"; icon = { buf, bufSize, dataSize }.
        let mount_ptr = 0x100u64;
        assert!(mem.write(mount_ptr, b"/savedata0\0\0\0\0\0\0"));
        let buf = 0x200u64;
        let payload = b"\x89PNG\r\n\x1a\nRAEEN-ICON";
        assert!(mem.write(buf, payload));
        let icon_ptr = 0x300u64;
        let mut icon = [0u8; 24];
        icon[0..8].copy_from_slice(&buf.to_le_bytes());
        icon[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        icon[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        assert!(mem.write(icon_ptr, &icon));

        // Nothing saved yet.
        assert_eq!(hle_load_icon(&ctx, &[mount_ptr, icon_ptr]), ERROR_NOT_FOUND);

        assert_eq!(hle_save_icon(&ctx, &[mount_ptr, icon_ptr]), SCE_OK);

        // Load into a SHORTER buffer: dataSize must still be the real size.
        let short_buf = 0x400u64;
        let mut short_icon = [0u8; 24];
        short_icon[0..8].copy_from_slice(&short_buf.to_le_bytes());
        short_icon[8..16].copy_from_slice(&4u64.to_le_bytes());
        assert!(mem.write(icon_ptr, &short_icon));
        assert_eq!(hle_load_icon(&ctx, &[mount_ptr, icon_ptr]), SCE_OK);

        let mut back = [0u8; 24];
        assert!(mem.read(icon_ptr, &mut back));
        assert_eq!(
            u64::from_le_bytes(back[16..24].try_into().unwrap()),
            payload.len() as u64,
            "dataSize reports the file's real size, not the copied count"
        );
        let mut copied = [0u8; 4];
        assert!(mem.read(short_buf, &mut copied));
        assert_eq!(&copied, &payload[..4], "only what fits is copied");

        // A null buffer or an unmountable point is a parameter error.
        assert_eq!(hle_save_icon(&ctx, &[mount_ptr, 0]), ERROR_PARAMETER);
        assert_eq!(hle_save_icon(&ctx, &[0, icon_ptr]), ERROR_PARAMETER);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An unrecognized commit layout must FAIL OPEN. Refusing it fails a
    /// commit whose bytes the VFS has usually already persisted, and a title
    /// that reads that as "the save did not land" stalls — measured on
    /// Minecraft's world creation (`resource=4`, a transaction id it never
    /// obtained from `sceSaveDataCreateTransactionResource`).
    #[test]
    fn commit_with_an_unknown_layout_flushes_instead_of_failing() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Neither a "/savedataN" string nor a known transaction resource —
        // exactly the shape the measured run reported.
        let request = 0x100u64;
        let mut bytes = [0u8; 64];
        bytes[0] = 4;
        assert!(mem.write(request, &bytes));

        assert_ne!(
            hle_commit(&ctx, &[request, 110, 4]),
            ERROR_INTERNAL,
            "an unknown layout must not fail the commit"
        );
    }

    #[test]
    fn transaction_resources_are_process_local_and_releasable() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        assert!(registry.is_implemented(
            "libSceSaveData_native",
            "sceSaveDataCreateTransactionResource"
        ));
        assert!(registry.is_implemented(
            "libSceSaveData_native",
            "sceSaveDataDeleteTransactionResource"
        ));
        assert!(registry.is_implemented("libSceSaveData_native", "sceSaveDataPrepare"));
    }

    #[test]
    fn set_param_records_native_save_metadata() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        assert!(registry.is_implemented("libSceSaveData_native", "sceSaveDataSetParam"));
    }

    #[test]
    fn mount_writes_the_savedata_mount_point() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root =
            std::env::temp_dir().join(format!("raeen-save-mount-slot-{}", std::process::id()));
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
        assert_eq!(&res[..nul], b"/savedata0", "mount point must be /savedata0");
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
    fn native_mount3_reads_resource_after_the_padding_field() {
        let mut request = [0u8; SAVE_DATA_MOUNT3_HEAD_SIZE];
        request[..4].copy_from_slice(&7i32.to_le_bytes());
        request[0x08..0x10].copy_from_slice(&0x1234_5678u64.to_le_bytes());
        request[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        request[0x18..0x20].copy_from_slice(&8u64.to_le_bytes());
        request[0x20..0x24].copy_from_slice(&MOUNT_MODE_CREATE2.to_le_bytes());
        request[0x24..0x28].copy_from_slice(&0x7f7f_7f7fu32.to_le_bytes());
        request[0x28..0x2c].copy_from_slice(&42i32.to_le_bytes());

        assert_eq!(
            parse_save_data_mount3_head(&request),
            SaveDataMount3Head {
                user_id: 7,
                dir_name_ptr: 0x1234_5678,
                blocks: 64,
                system_blocks: 8,
                mount_mode: MOUNT_MODE_CREATE2,
                resource: 42,
            },
            "the 0x24 padding must never be mistaken for the resource handle"
        );
    }

    #[test]
    fn bedrock_world_mount_keeps_leveldb_view_after_service_unmount() {
        assert_eq!(
            bedrock_world_guest_alias("BedrockWorldxDe5FqCkuF8@P1").as_deref(),
            Some("/minecraftWorlds/xDe5FqCkuF8@")
        );
        assert!(bedrock_world_guest_alias("BedrockUserSettingsStorage").is_none());
        assert!(bedrock_world_guest_alias("BedrockWorld../escape@P1").is_none());

        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root =
            std::env::temp_dir().join(format!("raeen-save-bedrock-alias-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        kernel.filesystem.set_savedata_directory(&root);

        assert!(mem.write(0x100, &1i32.to_le_bytes()));
        assert!(mem.write(0x108, &0x300u64.to_le_bytes()));
        assert!(mem.write(0x120, &MOUNT_MODE_CREATE2.to_le_bytes()));
        assert!(mem.write(0x300, b"BedrockWorldxDe5FqCkuF8@P1\0"));
        assert_eq!(hle_mount(&ctx, &[0x100, 0x200]), SCE_OK);
        let slot = root.join("BedrockWorldxDe5FqCkuF8@P1");
        assert_eq!(
            kernel
                .filesystem
                .resolve_path("/minecraftWorlds/xDe5FqCkuF8@/db/P/CURRENT"),
            Some(slot.join("db/P/CURRENT"))
        );

        assert!(mem.write(0x400, b"/savedata0\0"));
        assert_eq!(hle_unmount(&ctx, &[0x400]), SCE_OK);
        assert_eq!(
            kernel
                .filesystem
                .resolve_path("/minecraftWorlds/xDe5FqCkuF8@/db/P/CURRENT"),
            Some(slot.join("db/P/CURRENT")),
            "the app-storage view must outlive the transient /savedataN mount"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn mount_with_null_pointers_is_parameter_error() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_mount(&ctx, &[0, 0x200]), ERROR_PARAMETER);
        assert_eq!(hle_mount(&ctx, &[0x100, 0]), ERROR_PARAMETER);
    }

    #[test]
    fn ps4_transfer_mount_alias_reports_no_foreign_save_instead_of_faulting() {
        let registry = HleRegistry::new();
        assert!(
            registry.is_implemented("libSceSaveData_native", "sceSaveDataTransferringMountPs4")
        );

        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = crate::test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_transferring_mount(&ctx, &[0x100]), ERROR_NOT_FOUND);
    }

    #[test]
    fn mount_info_initializes_capacity_and_native_search_is_not_a_noop() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root =
            std::env::temp_dir().join(format!("raeen-save-mount-info-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        kernel.filesystem.set_savedata_directory(&root);
        assert!(mem.write(0x10, b"/savedata0\0"));
        assert!(mem.write(0x100, &[0xCC; 48]));

        assert_eq!(hle_get_mount_info(&ctx, &[0x10, 0x100]), SCE_OK);
        let mut info = [0u8; 48];
        assert!(mem.read(0x100, &mut info));
        assert_eq!(u64::from_le_bytes(info[..8].try_into().unwrap()), 16_384);
        assert_eq!(u64::from_le_bytes(info[8..16].try_into().unwrap()), 16_384);
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
        use raeen_kernel::filesystem::open_flags::{O_CREAT, O_TRUNC, O_WRONLY};

        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root = std::env::temp_dir().join(format!("raeen-save-commit-{}", std::process::id()));
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
        assert!(registry.is_implemented("libSceSaveData_native", "sceSaveDataCommit"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn delete_removes_only_the_requested_save_directory() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root = std::env::temp_dir().join(format!("raeen-save-delete-{}", std::process::id()));
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
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let root = std::env::temp_dir().join(format!("raeen-save-search-{}", std::process::id()));
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
        std::fs::write(root.join("slotA/payload.bin"), vec![0x5a; 32 * 1024 + 1]).unwrap();
        std::fs::create_dir_all(root.join("games")).unwrap();
        assert!(mem.write(0x110, &0x380u64.to_le_bytes()));
        assert!(mem.write(0x380, b"slot%\0"));
        assert!(mem.write(0x218, &0x800u64.to_le_bytes()));
        assert!(mem.write(0x220, &0x1000u64.to_le_bytes()));
        assert!(mem.write(0x800, &[0xCC; 0x530]));
        assert!(mem.write(0x1000, &[0xCC; 48]));
        assert_eq!(hle_dir_name_search(&ctx, &[0x100, 0x200]), SCE_OK);
        assert!(mem.read(0x200, &mut count));
        assert_eq!(u32::from_le_bytes(count), 1);
        let mut name = [0u8; 32];
        assert!(mem.read(0x300, &mut name));
        assert_eq!(&name[..5], b"slotA");
        let mut param = [0u8; 0x530];
        assert!(mem.read(0x800, &mut param));
        assert_eq!(
            param, [0; 0x530],
            "optional save parameters must never expose stale guest bytes"
        );
        let mut info = [0u8; 48];
        assert!(mem.read(0x1000, &mut info));
        assert_eq!(u64::from_le_bytes(info[..8].try_into().unwrap()), 16_384);
        assert_eq!(
            u64::from_le_bytes(info[8..16].try_into().unwrap()),
            16_382,
            "a 32-KiB-plus-one-byte save consumes two virtual blocks"
        );
        assert_eq!(&info[16..], &[0; 32]);

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
