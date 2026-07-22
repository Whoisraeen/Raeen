//! HLE libSceUserService — user/profile queries.
//!
//! Homebrew and games call this at startup to learn the signed-in user id,
//! which they then pass to `scePadOpen`, save-data, trophy, etc. Raeen
//! models a single local user. Constants (primary user id `1000`, the
//! `NO_EVENT` code) are cross-checked against SharpEmu's `UserServiceExports`.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// The single local user's id (matches SharpEmu's `PrimaryUserId`). Games
/// pass this to `scePadOpen`/save-data/etc.
const PRIMARY_USER_ID: i32 = 1000;
/// `SCE_USER_SERVICE_USER_ID_INVALID`.
const INVALID_USER_ID: i32 = -1;
/// `SCE_USER_SERVICE_ERROR_NO_EVENT` — returned by `GetEvent` when the queue
/// is empty (which it always is here); a title's event loop reads until it
/// sees this.
const ERROR_NO_EVENT: u64 = 0x8096_0007;
/// `SCE_USER_SERVICE_ERROR_INVALID_ARGUMENT`.
const ERROR_INVALID_ARGUMENT: u64 = 0x8096_0005;

/// Register libSceUserService HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceUserService", "sceUserServiceInitialize", hle_ok);
    registry.register("libSceUserService", "sceUserServiceTerminate", hle_ok);
    registry.register(
        "libSceUserService",
        "sceUserServiceGetInitialUser",
        hle_get_initial_user,
    );
    registry.register(
        "libSceUserService",
        "sceUserServiceGetLoginUserIdList",
        hle_get_login_user_id_list,
    );
    registry.register(
        "libSceUserService",
        "sceUserServiceGetUserName",
        hle_get_user_name,
    );
    registry.register("libSceUserService", "sceUserServiceGetEvent", hle_get_event);
    // Per-user accessibility / preset getters (measured ASTRO.BOT imports).
    // Defaults follow SharpEmu's `UserServiceExports` (GPL-2.0): trigger
    // effect 0 (no accessibility reduction — full adaptive triggers),
    // vibration 1 (enabled/normal), and a zeroed 0x28-byte presets block
    // whose leading u64 is its own size.
    registry.register(
        "libSceUserService",
        "sceUserServiceGetAccessibilityTriggerEffect",
        hle_get_accessibility_trigger_effect,
    );
    registry.register(
        "libSceUserService",
        "sceUserServiceGetAccessibilityVibration",
        hle_get_accessibility_vibration,
    );
    registry.register(
        "libSceUserService",
        "sceUserServiceGetGamePresets",
        hle_get_game_presets,
    );
    // `sceUserServiceGetPlatformPrivacySetting(parameterId, int32_t *value)`
    // — SharpEmu `UserServiceExports`: parameterId 1000 (the primary user)
    // gets value 0 ("no restriction"). SharpEmu's name is a recovered label
    // for the measured NID `D-CzAxQL0XI` (0x0ff0b303140bd172) — it does NOT
    // hash to that NID, so the binding must be explicit (`register_nid`).
    // The measured ASTRO.BOT imports it naming the wrapper library
    // `libSceUserServicePlatformPrivacyWs1`; resolution is provider-aware,
    // so both provider spellings are registered.
    for library in ["libSceUserService", "libSceUserServicePlatformPrivacyWs1"] {
        registry.register_nid(
            library,
            "sceUserServiceGetPlatformPrivacySetting",
            0x0ff0_b303_140b_d172,
            hle_get_platform_privacy_setting,
        );
    }
}

/// See the registration comment: privacy setting 0 = unrestricted.
fn hle_get_platform_privacy_setting(ctx: &HleContext, args: &[u64]) -> u64 {
    write_user_setting_i32(ctx, args, 0, "sceUserServiceGetPlatformPrivacySetting")
}

/// Shared body of the `(userId, int32_t *out)` user-setting getters
/// (SharpEmu `WriteUserSettingInt32`): only the primary user exists.
fn write_user_setting_i32(ctx: &HleContext, args: &[u64], value: i32, name: &str) -> u64 {
    let user_id = args.first().copied().unwrap_or(0) as i32;
    let out_ptr = args.get(1).copied().unwrap_or(0);
    if user_id != PRIMARY_USER_ID {
        return ERROR_INVALID_ARGUMENT;
    }
    if out_ptr == 0 || !ctx.mem.write(out_ptr, &value.to_le_bytes()) {
        warn!("{name}: out-ptr {out_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    debug!("{name}(userId={user_id}) -> {value}");
    SCE_OK
}

/// `sceUserServiceGetAccessibilityTriggerEffect(userId, int32_t *out)`:
/// 0 = the user has NOT reduced adaptive-trigger effects (default).
fn hle_get_accessibility_trigger_effect(ctx: &HleContext, args: &[u64]) -> u64 {
    write_user_setting_i32(ctx, args, 0, "sceUserServiceGetAccessibilityTriggerEffect")
}

/// `sceUserServiceGetAccessibilityVibration(userId, int32_t *out)`:
/// 1 = vibration enabled at the normal level (default).
fn hle_get_accessibility_vibration(ctx: &HleContext, args: &[u64]) -> u64 {
    write_user_setting_i32(ctx, args, 1, "sceUserServiceGetAccessibilityVibration")
}

/// `sceUserServiceGetGamePresets(userId, SceUserServiceGamePresets *out)`:
/// the user's system-level game presets (difficulty / performance-vs-quality
/// / first-person camera prefs). SharpEmu writes a zeroed 0x28-byte block
/// with the leading u64 set to the block size — "no preset expressed", which
/// every field's zero value means — and titles accept it.
fn hle_get_game_presets(ctx: &HleContext, args: &[u64]) -> u64 {
    const PRESETS_SIZE: u64 = 0x28;
    let user_id = args.first().copied().unwrap_or(0) as i32;
    let out_ptr = args.get(1).copied().unwrap_or(0);
    if user_id != PRIMARY_USER_ID {
        return ERROR_INVALID_ARGUMENT;
    }
    if out_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let mut presets = [0u8; PRESETS_SIZE as usize];
    presets[0..8].copy_from_slice(&PRESETS_SIZE.to_le_bytes());
    if !ctx.mem.write(out_ptr, &presets) {
        warn!("sceUserServiceGetGamePresets: out-ptr {out_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `sceUserServiceGetInitialUser(SceUserServiceUserId *userId)`: writes the
/// primary user id.
fn hle_get_initial_user(ctx: &HleContext, args: &[u64]) -> u64 {
    let user_id_ptr = args.first().copied().unwrap_or(0);
    debug!("sceUserServiceGetInitialUser(userId={user_id_ptr:#x})");
    if user_id_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    if !ctx.mem.write(user_id_ptr, &PRIMARY_USER_ID.to_le_bytes()) {
        warn!("sceUserServiceGetInitialUser: userId out-ptr {user_id_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

/// `sceUserServiceGetLoginUserIdList(SceUserServiceLoginUserIdList *list)`:
/// the struct is a fixed `int32_t userId[4]`; slot 0 is the primary user,
/// the rest `INVALID`.
fn hle_get_login_user_id_list(ctx: &HleContext, args: &[u64]) -> u64 {
    let list_ptr = args.first().copied().unwrap_or(0);
    debug!("sceUserServiceGetLoginUserIdList(list={list_ptr:#x})");
    if list_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let mut buf = [0u8; 16]; // int32[4]
    buf[0..4].copy_from_slice(&PRIMARY_USER_ID.to_le_bytes());
    buf[4..8].copy_from_slice(&INVALID_USER_ID.to_le_bytes());
    buf[8..12].copy_from_slice(&INVALID_USER_ID.to_le_bytes());
    buf[12..16].copy_from_slice(&INVALID_USER_ID.to_le_bytes());
    if !ctx.mem.write(list_ptr, &buf) {
        warn!("sceUserServiceGetLoginUserIdList: list out-ptr {list_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

/// `sceUserServiceGetUserName(SceUserServiceUserId userId, char *name,
/// size_t size)`: writes a NUL-terminated display name into the guest buffer.
fn hle_get_user_name(ctx: &HleContext, args: &[u64]) -> u64 {
    let user_id = args.first().copied().unwrap_or(0) as i32;
    let name_ptr = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    debug!("sceUserServiceGetUserName(userId={user_id}, name={name_ptr:#x}, size={size})");
    if user_id != PRIMARY_USER_ID {
        return ERROR_INVALID_ARGUMENT;
    }
    if name_ptr == 0 || size == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    // "Player" + NUL, truncated to the guest buffer size.
    let mut name = b"Player\0".to_vec();
    let cap = usize::try_from(size).unwrap_or(usize::MAX);
    if name.len() > cap {
        name.truncate(cap);
        *name.last_mut().unwrap() = 0; // keep it NUL-terminated
    }
    if !ctx.mem.write(name_ptr, &name) {
        warn!("sceUserServiceGetUserName: name out-ptr {name_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

/// `sceUserServiceGetEvent(SceUserServiceEvent *event)`: the event queue is
/// always empty (single, always-logged-in local user), so this reports
/// `NO_EVENT` — a title's login/logout event loop reads until it sees this.
fn hle_get_event(_ctx: &HleContext, _args: &[u64]) -> u64 {
    ERROR_NO_EVENT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn initial_user_and_login_list() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_initial_user(&ctx, &[0x100]), SCE_OK);
        let mut id = [0u8; 4];
        assert!(mem.read(0x100, &mut id));
        assert_eq!(i32::from_le_bytes(id), PRIMARY_USER_ID);

        assert_eq!(hle_get_login_user_id_list(&ctx, &[0x200]), SCE_OK);
        let mut list = [0u8; 16];
        assert!(mem.read(0x200, &mut list));
        assert_eq!(
            i32::from_le_bytes(list[0..4].try_into().unwrap()),
            PRIMARY_USER_ID
        );
        assert_eq!(
            i32::from_le_bytes(list[4..8].try_into().unwrap()),
            INVALID_USER_ID
        );
    }

    #[test]
    fn user_name_written_and_event_queue_empty() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_get_user_name(&ctx, &[PRIMARY_USER_ID as u64, 0x100, 32]),
            SCE_OK
        );
        let mut buf = [0u8; 7];
        assert!(mem.read(0x100, &mut buf));
        assert_eq!(&buf, b"Player\0");

        // Unknown user id → invalid arg.
        assert_eq!(
            hle_get_user_name(&ctx, &[42, 0x100, 32]),
            ERROR_INVALID_ARGUMENT
        );
        // Event loop terminates on NO_EVENT.
        assert_eq!(hle_get_event(&ctx, &[0x200]), ERROR_NO_EVENT);
    }

    /// Accessibility/preset getters answer the primary user with SharpEmu's
    /// defaults: trigger effect 0, vibration 1, zeroed self-sized presets.
    #[test]
    fn accessibility_and_preset_getters_report_defaults() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let uid = PRIMARY_USER_ID as u64;

        assert_eq!(
            hle_get_accessibility_trigger_effect(&ctx, &[uid, 0x100]),
            SCE_OK
        );
        let mut v = [0u8; 4];
        assert!(mem.read(0x100, &mut v));
        assert_eq!(i32::from_le_bytes(v), 0, "full trigger effects");

        assert_eq!(hle_get_accessibility_vibration(&ctx, &[uid, 0x100]), SCE_OK);
        assert!(mem.read(0x100, &mut v));
        assert_eq!(i32::from_le_bytes(v), 1, "vibration enabled");

        assert_eq!(hle_get_game_presets(&ctx, &[uid, 0x200]), SCE_OK);
        let mut presets = [0xFFu8; 0x28];
        assert!(mem.read(0x200, &mut presets));
        assert_eq!(u64::from_le_bytes(presets[0..8].try_into().unwrap()), 0x28);
        assert!(presets[8..].iter().all(|b| *b == 0), "no preset expressed");

        // Only the primary user exists.
        assert_eq!(
            hle_get_game_presets(&ctx, &[42, 0x200]),
            ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_get_accessibility_vibration(&ctx, &[uid, 0]),
            ERROR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn null_out_pointers_are_invalid_argument() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_get_initial_user(&ctx, &[0]), ERROR_INVALID_ARGUMENT);
        assert_eq!(
            hle_get_login_user_id_list(&ctx, &[0]),
            ERROR_INVALID_ARGUMENT
        );
    }
}
