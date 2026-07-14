//! HLE libSceUserService — user/profile queries.
//!
//! Homebrew and games call this at startup to learn the signed-in user id,
//! which they then pass to `scePadOpen`, save-data, trophy, etc. XPS5X
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
        let kernel = xps5x_kernel::OrbisKernel::new();
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
        let kernel = xps5x_kernel::OrbisKernel::new();
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

    #[test]
    fn null_out_pointers_are_invalid_argument() {
        let kernel = xps5x_kernel::OrbisKernel::new();
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
