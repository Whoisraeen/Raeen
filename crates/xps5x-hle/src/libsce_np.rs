//! HLE libSceNpManager — PSN (Np) account state.
//!
//! Titles query the signed-in PSN state at boot (even single-player ones,
//! to gate online features). XPS5X models **no PSN connection**: `GetState`
//! reports `SIGNED_OUT`, callbacks report nothing pending, and reachability
//! is `UNAVAILABLE` — so a title sees "offline", disables its online
//! features, and proceeds to gameplay rather than hanging on a PSN check.
//! The `SIGNED_OUT` state value is cross-checked against SharpEmu's
//! `NpManagerExports`.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// `SCE_NP_MANAGER_ERROR_INVALID_ARGUMENT`.
const ERROR_INVALID_ARGUMENT: u64 = 0x8055_0003;
/// `SceNpState::SIGNED_OUT` (1). (`UNKNOWN = 0`, `SIGNED_IN = 2`.)
const NP_STATE_SIGNED_OUT: u32 = 1;
/// `SceNpReachabilityState::UNAVAILABLE` (0).
const NP_REACHABILITY_UNAVAILABLE: u32 = 0;

/// Register libSceNpManager HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceNpManager", "sceNpGetState", hle_get_state);
    registry.register("libSceNpManager", "sceNpCheckCallback", hle_ok);
    registry.register("libSceNpManager", "sceNpCheckCallbackForLib", hle_ok);
    registry.register(
        "libSceNpManager",
        "sceNpRegisterStateCallback",
        hle_register_callback,
    );
    registry.register(
        "libSceNpManager",
        "sceNpRegisterStateCallbackA",
        hle_register_callback,
    );
    registry.register("libSceNpManager", "sceNpUnregisterStateCallback", hle_ok);
    registry.register("libSceNpManager", "sceNpSetNpTitleId", hle_ok);
    registry.register("libSceNpManager", "sceNpGetOnlineId", hle_get_online_id);
    registry.register(
        "libSceNpManager",
        "sceNpGetNpReachabilityState",
        hle_get_reachability,
    );
    registry.register(
        "libSceNpManager",
        "sceNpGetAccountCountryA",
        hle_get_account_country,
    );
    registry.register("libSceNpManager", "sceNpGameIntentInitialize", hle_ok);
    // libSceNpManagerForToolkit is a sibling library (same offline Np state);
    // its state callback registration behaves like the base one. Ported from
    // SharpEmu's `NpManagerExports` (GPL-2.0).
    registry.register(
        "libSceNpManagerForToolkit",
        "sceNpRegisterStateCallbackForToolkit",
        hle_register_callback,
    );
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `sceNpGetState(SceUserServiceUserId userId, SceNpState *state)`: reports
/// `SIGNED_OUT` — no PSN connection.
fn hle_get_state(ctx: &HleContext, args: &[u64]) -> u64 {
    let state_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sceNpGetState(state={state_ptr:#x}) -> SIGNED_OUT");
    if state_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    if !ctx.mem.write(state_ptr, &NP_STATE_SIGNED_OUT.to_le_bytes()) {
        warn!("sceNpGetState: state out-ptr {state_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

/// `sceNpRegisterStateCallback(callback, userdata)`: accepts the callback
/// (returns a callback id `0`); it simply never fires, since the account
/// state never changes from `SIGNED_OUT`.
fn hle_register_callback(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK // callback id 0
}

/// `sceNpGetOnlineId(SceUserServiceUserId userId, SceNpOnlineId *onlineId)`:
/// writes a placeholder online-id (`SceNpOnlineId` = `char data[16]` +
/// terminator + 3 pad = 20 bytes). A title that displays or keys on the
/// online id gets a stable value rather than garbage.
fn hle_get_online_id(ctx: &HleContext, args: &[u64]) -> u64 {
    let id_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sceNpGetOnlineId(onlineId={id_ptr:#x})");
    if id_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let mut buf = [0u8; 20];
    buf[..6].copy_from_slice(b"Player"); // data[16], NUL-padded
    if !ctx.mem.write(id_ptr, &buf) {
        warn!("sceNpGetOnlineId: onlineId out-ptr {id_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

/// `sceNpGetAccountCountryA(SceUserServiceUserId userId, SceNpCountryCode *country)`:
/// reports a fixed region (`"US"`). A title's Np/WebApi init can *require* a
/// country and assert on failure — ASTRO.BOT hard-asserts at `NpWebApi.cpp:1587`
/// when this returns an error. The country is a locale value, independent of the
/// signed-out connection state the rest of this module reports. `SceNpCountryCode`
/// is `{ char data[2]; char term; char pad[1]; }` (4 bytes). Ported from
/// SharpEmu's `NpManagerExports.NpGetAccountCountryA` (GPL-2.0).
fn hle_get_account_country(ctx: &HleContext, args: &[u64]) -> u64 {
    let user_id = args.first().copied().unwrap_or(0) as u32;
    let country_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sceNpGetAccountCountryA(country={country_ptr:#x}) -> \"US\"");
    // userId == -1 (invalid user) or a NULL out-ptr is an argument error.
    if user_id == 0xFFFF_FFFF || country_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    // data[2]="US", term=0, pad=0.
    if !ctx.mem.write(country_ptr, b"US\0\0") {
        warn!("sceNpGetAccountCountryA: out-ptr {country_ptr:#x} not writable");
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

/// `sceNpGetNpReachabilityState(...)`: PSN is unreachable (offline).
fn hle_get_reachability(ctx: &HleContext, args: &[u64]) -> u64 {
    let state_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sceNpGetNpReachabilityState(state={state_ptr:#x}) -> UNAVAILABLE");
    if state_ptr != 0
        && !ctx
            .mem
            .write(state_ptr, &NP_REACHABILITY_UNAVAILABLE.to_le_bytes())
    {
        warn!("sceNpGetNpReachabilityState: out-ptr {state_ptr:#x} not writable");
    }
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn get_state_reports_signed_out() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_state(&ctx, &[1000, 0x100]), SCE_OK);
        let mut s = [0u8; 4];
        assert!(mem.read(0x100, &mut s));
        assert_eq!(u32::from_le_bytes(s), NP_STATE_SIGNED_OUT);
        assert_eq!(hle_get_state(&ctx, &[1000, 0]), ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn np_manager_for_toolkit_state_callback_is_registered() {
        let reg = HleRegistry::new();
        assert!(reg.is_implemented(
            "libSceNpManagerForToolkit",
            "sceNpRegisterStateCallbackForToolkit"
        ));
    }

    #[test]
    fn get_account_country_writes_us() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_account_country(&ctx, &[1, 0x100]), SCE_OK);
        let mut c = [0u8; 4];
        assert!(mem.read(0x100, &mut c));
        assert_eq!(&c, b"US\0\0");
        // userId == -1 or a NULL out-ptr is an argument error.
        assert_eq!(
            hle_get_account_country(&ctx, &[0xFFFF_FFFF, 0x100]),
            ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_get_account_country(&ctx, &[1, 0]),
            ERROR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn get_online_id_writes_placeholder() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_online_id(&ctx, &[1000, 0x100]), SCE_OK);
        let mut id = [0u8; 6];
        assert!(mem.read(0x100, &mut id));
        assert_eq!(&id, b"Player");
    }
}
