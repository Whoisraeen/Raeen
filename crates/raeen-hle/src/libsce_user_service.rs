//! HLE libSceUserService — user/profile queries.
//!
//! Homebrew and games call this at startup to learn the signed-in user id,
//! which they then pass to `scePadOpen`, save-data, trophy, etc. Raeen
//! models a single local user. The retail-style primary id and one-shot login
//! event are cross-checked against SharpEmu, KytyPS5, and shadPS4.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// The single local user's retail-style id (matches current SharpEmu).
/// The high nibble encodes local slot 0; small emulator-local ids can map to
/// slot -1 in retail middleware and prevent the title from opening a pad.
const PRIMARY_USER_ID: i32 = 0x1000_0000;
/// `SCE_USER_SERVICE_USER_ID_INVALID`.
const INVALID_USER_ID: i32 = -1;
/// `SCE_USER_SERVICE_ERROR_NO_EVENT` — returned after the process's initial
/// login event has been consumed.
const ERROR_NO_EVENT: u64 = 0x8096_0007;
/// `SCE_USER_SERVICE_ERROR_INVALID_ARGUMENT`.
const ERROR_INVALID_ARGUMENT: u64 = 0x8096_0005;

/// Register libSceUserService HLE functions.
pub fn register(registry: &HleRegistry) {
    // Return-code-only lifecycle: the user queries behind them (initial user,
    // login list, names, events) are real, so acknowledging is complete.
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
    registry.register(
        "libSceUserService",
        "sceUserServiceGetAgeLevel",
        hle_get_age_level,
    );
    // The rest of the accessibility getter family, same `(userId, int32_t *out)`
    // shape and same SharpEmu defaults (`UserServiceExports.cs:227-273`): every
    // accessibility aid off. Registered as a family rather than one-per-fault
    // because a title that reads one reads several, and each miss costs a whole
    // measure/build/re-run cycle (Subnautica walked GetAgeLevel ->
    // GetAccessibilityChatTranscription one import at a time).
    registry.register(
        "libSceUserService",
        "sceUserServiceGetAccessibilityChatTranscription",
        hle_get_accessibility_chat_transcription,
    );
    registry.register(
        "libSceUserService",
        "sceUserServiceGetAccessibilityPressAndHoldDelay",
        hle_get_accessibility_press_and_hold_delay,
    );
    registry.register(
        "libSceUserService",
        "sceUserServiceGetAccessibilityZoomEnabled",
        hle_get_accessibility_zoom_enabled,
    );
    registry.register(
        "libSceUserService",
        "sceUserServiceGetAccessibilityZoomFollowFocus",
        hle_get_accessibility_zoom_follow_focus,
    );
    // `sceUserServiceGetPlatformPrivacySetting(parameterId, int32_t *value)`
    // — SharpEmu `UserServiceExports`: parameterId 1000
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
    const PRIVACY_PARAMETER_ID: i32 = 1000;
    let parameter_id = args.first().copied().unwrap_or(0) as i32;
    let out_ptr = args.get(1).copied().unwrap_or(0);
    if parameter_id != PRIVACY_PARAMETER_ID {
        return ERROR_INVALID_ARGUMENT;
    }
    if out_ptr == 0 || !ctx.mem.write(out_ptr, &0i32.to_le_bytes()) {
        warn!("sceUserServiceGetPlatformPrivacySetting: out-ptr {out_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    debug!("sceUserServiceGetPlatformPrivacySetting(parameterId={parameter_id}) -> 0");
    SCE_OK
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

/// `sceUserServiceGetAgeLevel(userId, int32_t *out)`: the account's age level,
/// which titles gate age-restricted content and online features on.
///
/// 18 = an adult account with nothing restricted, matching SharpEmu
/// (`UserServiceExports.cs:184-190`, `WriteUserSettingInt32(ctx, 18, …)`).
/// Measured as Subnautica Below Zero's blocker immediately after its Unity
/// launcher banner.
fn hle_get_age_level(ctx: &HleContext, args: &[u64]) -> u64 {
    const ADULT_AGE_LEVEL: i32 = 18;
    write_user_setting_i32(ctx, args, ADULT_AGE_LEVEL, "sceUserServiceGetAgeLevel")
}

/// `sceUserServiceGetAccessibilityChatTranscription(userId, int32_t *out)`:
/// 0 = chat transcription off (SharpEmu default).
fn hle_get_accessibility_chat_transcription(ctx: &HleContext, args: &[u64]) -> u64 {
    write_user_setting_i32(
        ctx,
        args,
        0,
        "sceUserServiceGetAccessibilityChatTranscription",
    )
}

/// `sceUserServiceGetAccessibilityPressAndHoldDelay(userId, int32_t *out)`:
/// 0 = the standard press-and-hold delay, not an extended one.
fn hle_get_accessibility_press_and_hold_delay(ctx: &HleContext, args: &[u64]) -> u64 {
    write_user_setting_i32(
        ctx,
        args,
        0,
        "sceUserServiceGetAccessibilityPressAndHoldDelay",
    )
}

/// `sceUserServiceGetAccessibilityZoomEnabled(userId, int32_t *out)`:
/// 0 = screen zoom off.
fn hle_get_accessibility_zoom_enabled(ctx: &HleContext, args: &[u64]) -> u64 {
    write_user_setting_i32(ctx, args, 0, "sceUserServiceGetAccessibilityZoomEnabled")
}

/// `sceUserServiceGetAccessibilityZoomFollowFocus(userId, int32_t *out)`:
/// 0 = zoom does not follow focus.
fn hle_get_accessibility_zoom_follow_focus(ctx: &HleContext, args: &[u64]) -> u64 {
    write_user_setting_i32(
        ctx,
        args,
        0,
        "sceUserServiceGetAccessibilityZoomFollowFocus",
    )
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

/// `sceUserServiceGetEvent(SceUserServiceEvent *event)`: deliver the initial
/// local-user login once, then report `NO_EVENT`. Retail titles commonly wait
/// for this transition before opening `libScePad`.
fn hle_get_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let event_ptr = args.first().copied().unwrap_or(0);
    if event_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    if !ctx.kernel.claim_initial_user_login_event() {
        return ERROR_NO_EVENT;
    }

    let mut event = [0u8; 8];
    // SceUserServiceEventType::Login
    event[0..4].copy_from_slice(&0i32.to_le_bytes());
    event[4..8].copy_from_slice(&PRIMARY_USER_ID.to_le_bytes());
    if !ctx.mem.write(event_ptr, &event) {
        ctx.kernel.restore_initial_user_login_event();
        warn!("sceUserServiceGetEvent: event out-ptr {event_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    tracing::info!(
        user_id = PRIMARY_USER_ID,
        "sceUserServiceGetEvent delivered the initial local-user login"
    );
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    /// Subnautica Below Zero dies on this import right after its Unity
    /// launcher banner. It must report an unrestricted adult account.
    ///
    /// The companion assertion — that name-derived registration lands on the
    /// NID the title actually imports (`0xc28369bbee3944b9`, encoded
    /// `woNpu+45RLk`) — lives in `raeen-firmware`, which owns NID hashing and
    /// provider-aware resolution.
    #[test]
    fn age_level_reports_an_adult_account() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_get_age_level(&ctx, &[PRIMARY_USER_ID as u64, 0x100]),
            SCE_OK
        );
        let mut age = [0u8; 4];
        assert!(mem.read(0x100, &mut age));
        assert_eq!(i32::from_le_bytes(age), 18);

        // A non-primary user and an unwritable out-pointer must be refused
        // rather than silently reporting an age.
        assert_ne!(hle_get_age_level(&ctx, &[0xdead, 0x100]), SCE_OK);
        assert_ne!(
            hle_get_age_level(&ctx, &[PRIMARY_USER_ID as u64, 0]),
            SCE_OK
        );
    }

    /// The remaining accessibility getters all report "aid disabled", matching
    /// SharpEmu. Registered and tested as a family so a title reading several
    /// of them does not cost one measure/build/re-run cycle per import.
    #[test]
    fn remaining_accessibility_getters_report_aids_disabled() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let family: [(&str, crate::HleFunction); 4] = [
            (
                "ChatTranscription",
                hle_get_accessibility_chat_transcription,
            ),
            (
                "PressAndHoldDelay",
                hle_get_accessibility_press_and_hold_delay,
            ),
            ("ZoomEnabled", hle_get_accessibility_zoom_enabled),
            ("ZoomFollowFocus", hle_get_accessibility_zoom_follow_focus),
        ];

        for (name, handler) in family {
            assert_eq!(
                handler(&ctx, &[PRIMARY_USER_ID as u64, 0x100]),
                SCE_OK,
                "{name} must succeed for the primary user"
            );
            let mut value = [0u8; 4];
            assert!(mem.read(0x100, &mut value));
            assert_eq!(i32::from_le_bytes(value), 0, "{name} must report disabled");

            assert_ne!(
                handler(&ctx, &[PRIMARY_USER_ID as u64, 0]),
                SCE_OK,
                "{name} must refuse a null out-pointer"
            );
        }
    }

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
    fn user_name_and_one_shot_login_event() {
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
        assert_eq!(hle_get_event(&ctx, &[0x200]), SCE_OK);
        let mut event = [0u8; 8];
        assert!(mem.read(0x200, &mut event));
        assert_eq!(i32::from_le_bytes(event[0..4].try_into().unwrap()), 0);
        assert_eq!(
            i32::from_le_bytes(event[4..8].try_into().unwrap()),
            PRIMARY_USER_ID
        );
        // Event loop terminates after consuming the initial login.
        assert_eq!(hle_get_event(&ctx, &[0x200]), ERROR_NO_EVENT);
    }

    #[test]
    fn login_event_is_process_scoped_and_failed_write_does_not_consume_it() {
        for _ in 0..2 {
            let kernel = raeen_kernel::OrbisKernel::new();
            let mem = crate::TestMemory::new(0x100);
            let alloc = crate::TestAllocator::new(0);
            let ctx = test_ctx(&kernel, &mem, &alloc);

            assert_eq!(hle_get_event(&ctx, &[0]), ERROR_INVALID_ARGUMENT);
            assert_eq!(hle_get_event(&ctx, &[0x1000]), ERROR_INVALID_ARGUMENT);
            assert_eq!(hle_get_event(&ctx, &[0x20]), SCE_OK);
            assert_eq!(hle_get_event(&ctx, &[0x20]), ERROR_NO_EVENT);
        }
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

        // Platform privacy uses parameter id 1000, not the retail user id.
        assert_eq!(
            hle_get_platform_privacy_setting(&ctx, &[1000, 0x300]),
            SCE_OK
        );
        assert!(mem.read(0x300, &mut v));
        assert_eq!(i32::from_le_bytes(v), 0, "unrestricted privacy default");
        assert_eq!(
            hle_get_platform_privacy_setting(&ctx, &[uid, 0x300]),
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
