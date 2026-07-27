//! HLE libSceNpManager — PSN (Np) account state.
//!
//! Titles query the signed-in PSN state at boot (even single-player ones,
//! to gate online features). Raeen models **no PSN connection**: `GetState`
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
    registry.register("libSceNpManager", "sceNpCheckCallback", hle_check_callback);
    registry.register(
        "libSceNpManager",
        "sceNpCheckCallbackForLib",
        hle_check_callback,
    );
    registry.register(
        "libSceNpManager",
        "sceNpRegisterStateCallback",
        hle_register_callback_legacy,
    );
    registry.register(
        "libSceNpManager",
        "sceNpRegisterStateCallbackA",
        hle_register_callback_a,
    );
    registry.register("libSceNpManager", "sceNpUnregisterStateCallback", hle_ok);
    // `sceNpRegisterNpReachabilityStateCallback(callback, userdata)`: accept the
    // reachability callback and never invoke it. Reachability transitions only
    // fire on a real PSN connection, which an offline session does not have, so
    // registering successfully and staying silent is the accurate emulation of a
    // signed-out console rather than a stub. SharpEmu `NpManagerExports.cs`
    // (#450, 0c467e8).
    registry.register(
        "libSceNpManager",
        "sceNpRegisterNpReachabilityStateCallback",
        hle_ok,
    );
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
    registry.register(
        "libSceNpManager",
        "sceNpGetAccountIdA",
        hle_get_account_id_a,
    );
    registry.register("libSceNpManager", "sceNpGameIntentInitialize", hle_ok);

    // Sync/async NP request lifecycle (Tier B, 2026-07-27). Model re-derived
    // from shadPS4's GPL-2.0 `np_manager.cpp`: a request is a real tracked
    // handle; a check operation *completes* it with the offline result
    // (`SIGNED_OUT`), which an async request reports as OK immediately and
    // hands the real result to `sceNpPollAsync` — so a title polling an async
    // PSN check terminates promptly with the honest offline answer instead of
    // spinning on an unresolved import. Nothing fabricates a signed-in state.
    registry.register(
        "libSceNpManager",
        "sceNpCreateAsyncRequest",
        hle_create_async_request,
    );
    registry.register("libSceNpManager", "sceNpDeleteRequest", hle_delete_request);
    registry.register("libSceNpManager", "sceNpAbortRequest", hle_abort_request);
    registry.register("libSceNpManager", "sceNpPollAsync", hle_poll_async);
    registry.register(
        "libSceNpManager",
        "sceNpCheckNpReachability",
        hle_check_offline_request,
    );
    registry.register(
        "libSceNpManager",
        "sceNpCheckPremium",
        hle_check_offline_request,
    );
    registry.register(
        "libSceNpManager",
        "sceNpGetAccountAge",
        hle_check_offline_request,
    );
    // Premium (PS Plus) events never fire on a signed-out console: notifying
    // usage is accepted and dropped; the callback registers and stays silent —
    // the same model as the reachability callback above.
    registry.register(
        "libSceNpManager",
        "sceNpNotifyPremiumFeature",
        hle_notify_premium_feature,
    );
    registry.register(
        "libSceNpManager",
        "sceNpRegisterPremiumEventCallback",
        hle_register_premium_callback,
    );
    registry.register(
        "libSceNpManager",
        "sceNpUnregisterPremiumEventCallback",
        hle_ok,
    );
    // libSceNpManagerForToolkit is a sibling library (same offline Np state);
    // its state callback registration behaves like the base one. Ported from
    // SharpEmu's `NpManagerExports` (GPL-2.0).
    registry.register(
        "libSceNpManagerForToolkit",
        "sceNpRegisterStateCallbackForToolkit",
        hle_register_callback_a,
    );

    // libSceNpAuth — process-local request lifecycle with an honest offline
    // result. Minecraft creates this request on a worker while entering its
    // local-world flow; leaving the provider unresolved kills that worker and
    // strands the main thread on its completion condition forever.
    registry.register(
        "libSceNpAuth",
        "sceNpAuthCreateRequest",
        hle_np_auth_create_request,
    );
    registry.register(
        "libSceNpAuth",
        "sceNpAuthGetAuthorizationCodeV3",
        hle_np_auth_get_authorization_code_v3,
    );
    registry.register(
        "libSceNpAuth",
        "sceNpAuthDeleteRequest",
        hle_np_auth_delete_request,
    );
    registry.register(
        "libSceNpAuthAuthorizedApp",
        "sceNpAuthGetAuthorizedAppCode",
        hle_np_auth_get_authorized_app_code,
    );
    // Async variants (measured GTA V imports). Same offline model as the
    // NpManager request family: creation succeeds with a real id, the poll
    // finishes immediately with SIGNED_OUT — every auth outcome offline.
    registry.register(
        "libSceNpAuth",
        "sceNpAuthCreateAsyncRequest",
        hle_np_auth_create_request,
    );
    registry.register(
        "libSceNpAuth",
        "sceNpAuthAbortRequest",
        hle_np_auth_abort_request,
    );
    registry.register("libSceNpAuth", "sceNpAuthPollAsync", hle_np_auth_poll_async);

    // libSceNpAuthAuthorizedAppDialog — the PSN "authorize this app" popup.
    // Names recovered via the SharpEmu catalogue merge (the whole set appeared
    // as unnamed unresolved imports in a real 2026-07-16 run). Raeen has no host
    // popup, so the dialog completes IMMEDIATELY with an authorized result —
    // the same model as `sceMsgDialog` (libsce_common_dialog.rs): Open jumps the
    // status to FINISHED, GetStatus reports it, GetResult writes success.
    registry.register(
        "libSceNpAuthAuthorizedAppDialog",
        "sceNpAuthAuthorizedAppDialogInitialize",
        hle_auth_dialog_initialize,
    );
    registry.register(
        "libSceNpAuthAuthorizedAppDialog",
        "sceNpAuthAuthorizedAppDialogOpen",
        hle_auth_dialog_open,
    );
    for f in [
        "sceNpAuthAuthorizedAppDialogUpdateStatus",
        "sceNpAuthAuthorizedAppDialogGetStatus",
    ] {
        registry.register("libSceNpAuthAuthorizedAppDialog", f, hle_auth_dialog_status);
    }
    registry.register(
        "libSceNpAuthAuthorizedAppDialog",
        "sceNpAuthAuthorizedAppDialogGetResult",
        hle_auth_dialog_get_result,
    );
    for f in [
        "sceNpAuthAuthorizedAppDialogClose",
        "sceNpAuthAuthorizedAppDialogTerminate",
    ] {
        registry.register("libSceNpAuthAuthorizedAppDialog", f, hle_ok);
    }
}

/// Shared common-dialog status enum (`NONE`=0, `INITIALIZED`=1, `RUNNING`=2,
/// `FINISHED`=3). The authorized-app dialog never lingers in `RUNNING`.
static AUTH_DIALOG_STATUS: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
const AUTH_STATUS_INITIALIZED: i32 = 1;
const AUTH_STATUS_FINISHED: i32 = 3;
const NP_ERROR_SIGNED_OUT: u64 = 0x8055_0006;
const NP_AUTH_ERROR_INVALID_ARGUMENT: u64 = 0x8055_0301;
const NP_AUTH_ERROR_REQUEST_NOT_FOUND: u64 = 0x8055_0306;

/// Allocate an offline auth request. The id range and lifetime are
/// cross-checked against shadPS4's GPL-2.0 `libSceNpAuth` implementation; the
/// state lives on `OrbisKernel`, so consecutive guest processes cannot leak
/// handles into one another.
fn hle_np_auth_create_request(ctx: &HleContext, _args: &[u64]) -> u64 {
    let id = ctx
        .kernel
        .np_auth_next_request
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        .wrapping_add(1);
    ctx.kernel.np_auth_requests.insert(id, ());
    debug!("sceNpAuthCreateRequest() -> {id:#x} (offline)");
    id as u32 as u64
}

/// Complete a synchronous authorization-code request as signed out. This is
/// the real state Raeen exposes through `sceNpGetState`; returning it here lets
/// the title disable online services and continue its local-world path.
fn hle_np_auth_get_authorization_code_v3(ctx: &HleContext, args: &[u64]) -> u64 {
    let request = args.first().copied().unwrap_or(0) as i32;
    let params = args.get(1).copied().unwrap_or(0);
    let code = args.get(2).copied().unwrap_or(0);
    if params == 0 || code == 0 {
        return NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    if !ctx.kernel.np_auth_requests.contains_key(&request) {
        return NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }
    debug!("sceNpAuthGetAuthorizationCodeV3({request:#x}) -> SIGNED_OUT");
    NP_ERROR_SIGNED_OUT
}

fn hle_np_auth_delete_request(ctx: &HleContext, args: &[u64]) -> u64 {
    let request = args.first().copied().unwrap_or(0) as i32;
    if ctx.kernel.np_auth_requests.remove(&request).is_none() {
        return NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }
    SCE_OK
}

/// Authorized-app codes are PSN-issued credentials. An offline process has no
/// code to return, so expose the same signed-out state as the base auth
/// provider. This is preferable to fabricating a credential and lets local
/// gameplay take the title's normal offline branch.
fn hle_np_auth_get_authorized_app_code(_ctx: &HleContext, _args: &[u64]) -> u64 {
    NP_ERROR_SIGNED_OUT
}

/// `sceNpAuthAbortRequest(req_id)`: acknowledged — offline auth requests
/// complete instantly, so there is never in-flight work to cancel; an unknown
/// id reports not-found.
fn hle_np_auth_abort_request(ctx: &HleContext, args: &[u64]) -> u64 {
    let request = args.first().copied().unwrap_or(0) as i32;
    if !ctx.kernel.np_auth_requests.contains_key(&request) {
        return NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }
    SCE_OK
}

/// `sceNpAuthPollAsync(req_id, s32 *result)`: finishes immediately — every
/// offline auth outcome is `SIGNED_OUT`, delivered through the result
/// pointer with an OK return so the title's poll loop terminates promptly.
fn hle_np_auth_poll_async(ctx: &HleContext, args: &[u64]) -> u64 {
    let request = args.first().copied().unwrap_or(0) as i32;
    let result_ptr = args.get(1).copied().unwrap_or(0);
    if result_ptr == 0 {
        return NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    if !ctx.kernel.np_auth_requests.contains_key(&request) {
        return NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }
    if !ctx
        .mem
        .write(result_ptr, &(NP_ERROR_SIGNED_OUT as u32).to_le_bytes())
    {
        return NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    debug!("sceNpAuthPollAsync(req={request:#x}) -> OK, result=SIGNED_OUT");
    SCE_OK
}

fn hle_auth_dialog_initialize(_ctx: &HleContext, _args: &[u64]) -> u64 {
    AUTH_DIALOG_STATUS.store(
        AUTH_STATUS_INITIALIZED,
        std::sync::atomic::Ordering::Relaxed,
    );
    SCE_OK
}

/// With no host popup, authorize immediately: status jumps to `FINISHED` so the
/// title's next poll completes.
fn hle_auth_dialog_open(_ctx: &HleContext, _args: &[u64]) -> u64 {
    AUTH_DIALOG_STATUS.store(AUTH_STATUS_FINISHED, std::sync::atomic::Ordering::Relaxed);
    SCE_OK
}

/// `UpdateStatus`/`GetStatus`: return the current status (`FINISHED` once open).
fn hle_auth_dialog_status(_ctx: &HleContext, _args: &[u64]) -> u64 {
    AUTH_DIALOG_STATUS.load(std::sync::atomic::Ordering::Relaxed) as u32 as u64
}

/// `GetResult(SceNpAuthAuthorizedAppDialogResult *result)`: report an
/// authorized (success) result. The exact struct layout is not published, so
/// only the leading `int32 result` field (offset 0) is written to `0` (success)
/// — the field a caller checks first; the rest is left as the caller allocated.
fn hle_auth_dialog_get_result(ctx: &HleContext, args: &[u64]) -> u64 {
    let result_ptr = args.first().copied().unwrap_or(0);
    if result_ptr != 0 {
        let _ = ctx.mem.write(result_ptr, &0i32.to_le_bytes());
    }
    SCE_OK
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

// --- NP request lifecycle (shadPS4-derived offline model) ------------------

/// `ORBIS_NP_ERROR_INVALID_SIZE` (shadPS4 `np_error.h`).
const NP_ERROR_INVALID_SIZE: u64 = 0x8055_0011;
/// `ORBIS_NP_ERROR_ABORTED`.
const NP_ERROR_ABORTED: u64 = 0x8055_0012;
/// `ORBIS_NP_ERROR_REQUEST_NOT_FOUND`.
const NP_ERROR_REQUEST_NOT_FOUND: u64 = 0x8055_0014;
/// `ORBIS_NP_ERROR_INVALID_ID`.
const NP_ERROR_INVALID_ID: u64 = 0x8055_0015;

#[derive(Clone, Copy, PartialEq)]
enum NpRequestState {
    /// Created; no check operation has run on it yet.
    Ready,
    /// A check operation completed it; `result` holds the outcome.
    Complete,
    /// Aborted before completion.
    Aborted,
}

#[derive(Clone, Copy)]
struct NpRequest {
    is_async: bool,
    state: NpRequestState,
    /// The i32 outcome `sceNpPollAsync` reports (`SIGNED_OUT` offline).
    result: u32,
}

/// Live NP requests. Process-global like the module's other registries; ids
/// are nonzero and monotonic.
static NP_REQUESTS: std::sync::Mutex<Option<std::collections::HashMap<i32, NpRequest>>> =
    std::sync::Mutex::new(None);
static NP_NEXT_REQUEST: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1);

/// `sceNpCreateAsyncRequest(const SceNpCreateAsyncRequestParameter *param)`:
/// a null param is an argument error and a zero leading `size` field an
/// invalid-size error (shadPS4 validates `param->size` exactly; the exact
/// struct size is SDK-dependent, so only the never-valid zero is rejected
/// here). Returns a real request id.
fn hle_create_async_request(ctx: &HleContext, args: &[u64]) -> u64 {
    let param = args.first().copied().unwrap_or(0);
    if param == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let mut size_bytes = [0u8; 8];
    if !ctx.mem.read(param, &mut size_bytes) {
        return ERROR_INVALID_ARGUMENT;
    }
    if u64::from_le_bytes(size_bytes) == 0 {
        return NP_ERROR_INVALID_SIZE;
    }
    let id = NP_NEXT_REQUEST.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    NP_REQUESTS
        .lock()
        .unwrap()
        .get_or_insert_with(std::collections::HashMap::new)
        .insert(
            id,
            NpRequest {
                is_async: true,
                state: NpRequestState::Ready,
                result: 0,
            },
        );
    debug!("sceNpCreateAsyncRequest() -> {id:#x}");
    id as u32 as u64
}

/// Complete `req_id` with the offline outcome and return what the caller
/// sees: an async request reports OK now (the result travels through
/// `sceNpPollAsync`); a sync request reports the outcome directly.
fn complete_np_request_offline(req_id: i32, outcome: u32) -> u64 {
    let mut requests = NP_REQUESTS.lock().unwrap();
    let Some(req) = requests.as_mut().and_then(|map| map.get_mut(&req_id)) else {
        return NP_ERROR_REQUEST_NOT_FOUND;
    };
    match req.state {
        NpRequestState::Aborted => NP_ERROR_ABORTED,
        NpRequestState::Complete => ERROR_INVALID_ARGUMENT,
        NpRequestState::Ready => {
            req.state = NpRequestState::Complete;
            req.result = outcome;
            if req.is_async {
                SCE_OK
            } else {
                u64::from(outcome)
            }
        }
    }
}

/// `sceNpCheckNpReachability` / `sceNpCheckPremium` / `sceNpGetAccountAge`
/// `(req_id, ...)`: request-keyed PSN checks. Offline they complete the
/// request with `SIGNED_OUT` — no reachability, no premium answer, and no
/// fabricated account age (age is PSN account data Raeen does not have).
fn hle_check_offline_request(_ctx: &HleContext, args: &[u64]) -> u64 {
    let req_id = args.first().copied().unwrap_or(0) as i32;
    let ret = complete_np_request_offline(req_id, NP_ERROR_SIGNED_OUT as u32);
    debug!("sceNp check(req={req_id:#x}) -> {ret:#x} (offline: SIGNED_OUT)");
    ret
}

/// `sceNpAbortRequest(req_id)`: an already-complete request ignores the abort
/// (OK, matching shadPS4); a pending one is marked aborted with the `ABORTED`
/// result for a subsequent poll.
fn hle_abort_request(_ctx: &HleContext, args: &[u64]) -> u64 {
    let req_id = args.first().copied().unwrap_or(0) as i32;
    let mut requests = NP_REQUESTS.lock().unwrap();
    let Some(req) = requests.as_mut().and_then(|map| map.get_mut(&req_id)) else {
        return NP_ERROR_REQUEST_NOT_FOUND;
    };
    if req.state != NpRequestState::Complete {
        req.state = NpRequestState::Aborted;
        req.result = NP_ERROR_ABORTED as u32;
    }
    SCE_OK
}

/// `sceNpDeleteRequest(req_id)`: release the request.
fn hle_delete_request(_ctx: &HleContext, args: &[u64]) -> u64 {
    let req_id = args.first().copied().unwrap_or(0) as i32;
    match NP_REQUESTS
        .lock()
        .unwrap()
        .as_mut()
        .and_then(|m| m.remove(&req_id))
    {
        Some(_) => SCE_OK,
        None => NP_ERROR_REQUEST_NOT_FOUND,
    }
}

/// `sceNpPollAsync(req_id, s32 *result)`: the request completed at
/// check-call time, so the poll finishes immediately — OK with the stored
/// offline result. A request nothing was started on (`Ready`), or a sync
/// request, is `INVALID_ID` (shadPS4's rule).
fn hle_poll_async(ctx: &HleContext, args: &[u64]) -> u64 {
    let req_id = args.first().copied().unwrap_or(0) as i32;
    let result_ptr = args.get(1).copied().unwrap_or(0);
    if result_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let requests = NP_REQUESTS.lock().unwrap();
    let Some(req) = requests.as_ref().and_then(|map| map.get(&req_id)) else {
        return NP_ERROR_REQUEST_NOT_FOUND;
    };
    if !req.is_async || req.state == NpRequestState::Ready {
        return NP_ERROR_INVALID_ID;
    }
    if !ctx.mem.write(result_ptr, &req.result.to_le_bytes()) {
        return ERROR_INVALID_ARGUMENT;
    }
    debug!(
        "sceNpPollAsync(req={req_id:#x}) -> OK, result={:#x}",
        req.result
    );
    SCE_OK
}

/// `sceNpNotifyPremiumFeature(const SceNpNotifyPremiumFeatureParameter *)`:
/// the usage notification is accepted and dropped — there is no PSN to
/// notify, and the call carries no answer the title depends on.
fn hle_notify_premium_feature(_ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    SCE_OK
}

/// `sceNpRegisterPremiumEventCallback(callback, userdata)`: accept and stay
/// silent — premium (PS Plus) events only fire on a live PSN connection.
fn hle_register_premium_callback(_ctx: &HleContext, args: &[u64]) -> u64 {
    if args.first().copied().unwrap_or(0) == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
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

/// The one local user Raeen models (matches `libsce_user_service`).
const PRIMARY_USER_ID: u64 = 0x1000_0000;

/// `sceNpRegisterStateCallback(callback, userdata)` — the legacy 4-argument
/// form, invoked `(userId, state, SceNpId *npId, void *userdata)`.
///
/// The callback is recorded so `sceNpCheckCallback` can deliver the current
/// account state to it. Titles register a state callback and then wait for
/// the initial state event rather than polling `sceNpGetState`; a callback
/// that never fires strands them (Minecraft's post-menu Ore-UI page).
fn hle_register_callback_legacy(ctx: &HleContext, args: &[u64]) -> u64 {
    register_np_state_callback(ctx, args, true)
}

/// `sceNpRegisterStateCallbackA` / `...ForToolkit` — the 3-argument form,
/// invoked `(userId, state, void *userdata)` (no `npId`).
fn hle_register_callback_a(ctx: &HleContext, args: &[u64]) -> u64 {
    register_np_state_callback(ctx, args, false)
}

fn register_np_state_callback(ctx: &HleContext, args: &[u64], legacy_np_id_arg: bool) -> u64 {
    let entry = args.first().copied().unwrap_or(0);
    let userdata = args.get(1).copied().unwrap_or(0);
    if entry == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    let mut callbacks = ctx.kernel.np_state_callbacks.lock();
    // De-dupe: a title that re-registers the same entry must not queue two
    // deliveries (the real kernel rejects a duplicate; recording it once is
    // enough for our purposes).
    if !callbacks.iter().any(|cb| cb.entry == entry) {
        callbacks.push(raeen_kernel::NpStateCallbackRegistration {
            entry,
            userdata,
            legacy_np_id_arg,
            notified: false,
        });
        debug!(
            "sceNpRegisterStateCallback(entry={entry:#x}, userdata={userdata:#x}, legacy={legacy_np_id_arg}) \
             — queued initial SIGNED_OUT delivery"
        );
    }
    SCE_OK // callback id 0
}

/// `sceNpCheckCallback()`: the title's pump for queued NP callbacks. On real
/// hardware this is where the system delivers the account-state event it
/// queued at registration, on the title's own thread (shadPS4
/// `np_manager.cpp` `DispatchPendingNpStateCallbacks`, called from
/// `sceNpCheckCallback`).
///
/// Raeen delivers exactly one un-notified callback per pump — the deferred
/// guest-call channel carries one call per dispatch, and the title pumps this
/// repeatedly, so multiple callbacks drain across successive pumps. The state
/// delivered is `SIGNED_OUT`, consistent with `sceNpGetState`: an offline
/// console genuinely reports signed-out, and the point is that the event
/// FIRES so the UI's "waiting for account state" gate opens, not that it
/// reports online.
fn hle_check_callback(ctx: &HleContext, _args: &[u64]) -> u64 {
    let pending = {
        let mut callbacks = ctx.kernel.np_state_callbacks.lock();
        match callbacks.iter_mut().find(|cb| !cb.notified) {
            Some(cb) => {
                let snapshot = *cb;
                cb.notified = true;
                Some(snapshot)
            }
            None => None,
        }
    };
    let Some(cb) = pending else {
        return SCE_OK;
    };

    // Legacy: (userId, state, npId*, userdata); A/toolkit: (userId, state,
    // userdata). npId is NULL — a signed-out account has no online id.
    let args = if cb.legacy_np_id_arg {
        [
            PRIMARY_USER_ID,
            u64::from(NP_STATE_SIGNED_OUT),
            0,
            cb.userdata,
            0,
            0,
        ]
    } else {
        [
            PRIMARY_USER_ID,
            u64::from(NP_STATE_SIGNED_OUT),
            cb.userdata,
            0,
            0,
            0,
        ]
    };
    if !ctx.guest_calls.request(crate::GuestCallRequest {
        entry: cb.entry,
        args,
        completion: None,
    }) {
        // Another deferred call is already pending this dispatch; un-mark so
        // the next pump retries. The title pumps continuously, so this is a
        // one-tick delay, not a lost event.
        if let Some(slot) = ctx
            .kernel
            .np_state_callbacks
            .lock()
            .iter_mut()
            .find(|c| c.entry == cb.entry)
        {
            slot.notified = false;
        }
        return SCE_OK;
    }
    debug!(
        "sceNpCheckCallback: delivering SIGNED_OUT to NP state callback {:#x}",
        cb.entry
    );
    SCE_OK
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

/// The stable fake PSN account id (ASCII "XPS5X" padded into a u64 — clearly
/// synthetic in logs, never zero, identical across runs).
const FAKE_ACCOUNT_ID: u64 = 0x5850_5335_5800_0001; // "XPS5X" tag + 1

/// `sceNpGetAccountIdA(SceUserServiceUserId userId, uint64_t *accountId)`:
/// report a stable, nonzero account id for the local user.
///
/// shadPS4 (`np_manager.cpp:579`, NID `rbknaUjpqWo`) validates
/// `accountId != NULL` and `userId != INVALID` then writes the account id;
/// signed-out it writes 0 and returns SIGNED_OUT. Raeen instead answers with
/// a fixed synthetic id and success — like the `sceNpGetAccountCountryA`
/// stub above, the id is identity/locale data a title keys saves and
/// telemetry buckets on, independent of the SIGNED_OUT connection state the
/// rest of this module reports, and an error here is a measured hard-assert
/// path (the NpWebApi init family).
fn hle_get_account_id_a(ctx: &HleContext, args: &[u64]) -> u64 {
    let user_id = args.first().copied().unwrap_or(0) as u32;
    let id_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sceNpGetAccountIdA(userId={user_id}, accountId={id_ptr:#x}) -> {FAKE_ACCOUNT_ID:#x}");
    if user_id == 0xFFFF_FFFF || id_ptr == 0 {
        return ERROR_INVALID_ARGUMENT;
    }
    if !ctx.mem.write(id_ptr, &FAKE_ACCOUNT_ID.to_le_bytes()) {
        warn!("sceNpGetAccountIdA: accountId out-ptr {id_ptr:#x} not writable");
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
    fn np_auth_request_lifecycle_reports_offline_without_faulting() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let request = hle_np_auth_create_request(&ctx, &[]);
        assert_eq!(request, 0x1000_0001);
        assert_eq!(
            hle_np_auth_get_authorization_code_v3(&ctx, &[request, 0x100, 0x200]),
            NP_ERROR_SIGNED_OUT
        );
        assert_eq!(hle_np_auth_delete_request(&ctx, &[request]), SCE_OK);
        assert_eq!(
            hle_np_auth_delete_request(&ctx, &[request]),
            NP_AUTH_ERROR_REQUEST_NOT_FOUND
        );

        let registry = HleRegistry::new();
        for function in [
            "sceNpAuthCreateRequest",
            "sceNpAuthGetAuthorizationCodeV3",
            "sceNpAuthDeleteRequest",
        ] {
            assert!(registry.is_implemented("libSceNpAuth", function));
        }
        assert!(
            registry.is_implemented("libSceNpAuthAuthorizedApp", "sceNpAuthGetAuthorizedAppCode")
        );
    }

    /// The async auth poll terminates immediately with SIGNED_OUT — a title
    /// polling an offline auth request never spins.
    #[test]
    fn np_auth_async_poll_finishes_immediately_signed_out() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let req = hle_np_auth_create_request(&ctx, &[0x100]);
        assert_eq!(hle_np_auth_poll_async(&ctx, &[req, 0x200]), SCE_OK);
        let mut result = [0u8; 4];
        assert!(mem.read(0x200, &mut result));
        assert_eq!(u32::from_le_bytes(result), NP_ERROR_SIGNED_OUT as u32);
        assert_eq!(hle_np_auth_abort_request(&ctx, &[req]), SCE_OK);
        assert_eq!(hle_np_auth_delete_request(&ctx, &[req]), SCE_OK);
        assert_eq!(
            hle_np_auth_poll_async(&ctx, &[req, 0x200]),
            NP_AUTH_ERROR_REQUEST_NOT_FOUND
        );
        assert_eq!(
            hle_np_auth_abort_request(&ctx, &[req]),
            NP_AUTH_ERROR_REQUEST_NOT_FOUND
        );

        let registry = HleRegistry::new();
        for function in [
            "sceNpAuthCreateAsyncRequest",
            "sceNpAuthAbortRequest",
            "sceNpAuthPollAsync",
        ] {
            assert!(registry.is_implemented("libSceNpAuth", function));
        }
    }

    /// The authorized-app dialog completes immediately: Open moves the status
    /// to FINISHED, GetStatus/UpdateStatus report it, and GetResult writes a
    /// success result — a title polling the dialog proceeds instead of hanging.
    #[test]
    fn auth_dialog_completes_immediately_with_success() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_auth_dialog_initialize(&ctx, &[]), SCE_OK);
        assert_eq!(
            hle_auth_dialog_status(&ctx, &[]) as i32,
            AUTH_STATUS_INITIALIZED
        );
        assert_eq!(hle_auth_dialog_open(&ctx, &[]), SCE_OK);
        assert_eq!(
            hle_auth_dialog_status(&ctx, &[]) as i32,
            AUTH_STATUS_FINISHED,
            "the dialog finishes as soon as it is opened"
        );

        assert_eq!(hle_auth_dialog_get_result(&ctx, &[0x200]), SCE_OK);
        let mut r = [0xFFu8; 4];
        assert!(mem.read(0x200, &mut r));
        assert_eq!(i32::from_le_bytes(r), 0, "result field reports success");
        // A null result pointer is tolerated.
        assert_eq!(hle_auth_dialog_get_result(&ctx, &[0]), SCE_OK);
    }

    /// The async request lifecycle terminates promptly with the honest
    /// offline result: create -> check completes it (returning OK because it
    /// is async) -> poll immediately reports SIGNED_OUT -> delete releases.
    /// No polling loop can hang and no signed-in state is fabricated.
    #[test]
    fn async_np_request_completes_immediately_with_signed_out() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Param block with a nonzero leading size field.
        assert!(mem.write(0x100, &16u64.to_le_bytes()));
        assert_eq!(hle_create_async_request(&ctx, &[0]), ERROR_INVALID_ARGUMENT);
        // Zero size field -> INVALID_SIZE.
        assert!(mem.write(0x200, &0u64.to_le_bytes()));
        assert_eq!(
            hle_create_async_request(&ctx, &[0x200]),
            NP_ERROR_INVALID_SIZE
        );

        let req = hle_create_async_request(&ctx, &[0x100]);
        assert!(req > 0 && req < 0x8000_0000, "a real request id: {req:#x}");

        // Polling before any check ran is INVALID_ID (shadPS4 rule).
        assert_eq!(hle_poll_async(&ctx, &[req, 0x300]), NP_ERROR_INVALID_ID);

        // The check completes the async request and returns OK.
        assert_eq!(hle_check_offline_request(&ctx, &[req, 0x1000_0000]), SCE_OK);

        // Poll now finishes immediately with the offline result.
        assert_eq!(hle_poll_async(&ctx, &[req, 0x300]), SCE_OK);
        let mut result = [0u8; 4];
        assert!(mem.read(0x300, &mut result));
        assert_eq!(
            u32::from_le_bytes(result),
            NP_ERROR_SIGNED_OUT as u32,
            "the async outcome is the honest offline error"
        );
        // Null result pointer refused.
        assert_eq!(hle_poll_async(&ctx, &[req, 0]), ERROR_INVALID_ARGUMENT);

        assert_eq!(hle_delete_request(&ctx, &[req]), SCE_OK);
        assert_eq!(hle_delete_request(&ctx, &[req]), NP_ERROR_REQUEST_NOT_FOUND);
        assert_eq!(
            hle_poll_async(&ctx, &[req, 0x300]),
            NP_ERROR_REQUEST_NOT_FOUND
        );
    }

    /// Abort marks a pending request; the poll then reports ABORTED, and an
    /// abort after completion is ignored (OK).
    #[test]
    fn abort_request_reports_aborted_through_poll() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, &16u64.to_le_bytes()));
        let req = hle_create_async_request(&ctx, &[0x100]);
        assert_eq!(hle_abort_request(&ctx, &[req]), SCE_OK);
        assert_eq!(hle_poll_async(&ctx, &[req, 0x300]), SCE_OK);
        let mut result = [0u8; 4];
        assert!(mem.read(0x300, &mut result));
        assert_eq!(u32::from_le_bytes(result), NP_ERROR_ABORTED as u32);
        // A check against an aborted request reports ABORTED.
        assert_eq!(hle_check_offline_request(&ctx, &[req, 0]), NP_ERROR_ABORTED);
        assert_eq!(hle_delete_request(&ctx, &[req]), SCE_OK);
        // Unknown ids everywhere -> REQUEST_NOT_FOUND.
        assert_eq!(
            hle_abort_request(&ctx, &[0x7FFF]),
            NP_ERROR_REQUEST_NOT_FOUND
        );
        assert_eq!(
            hle_check_offline_request(&ctx, &[0x7FFF, 0]),
            NP_ERROR_REQUEST_NOT_FOUND
        );
    }

    /// Every measured GTA V libSceNpManager import resolves.
    #[test]
    fn measured_np_manager_imports_are_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceNpCreateAsyncRequest",
            "sceNpDeleteRequest",
            "sceNpAbortRequest",
            "sceNpPollAsync",
            "sceNpCheckNpReachability",
            "sceNpCheckPremium",
            "sceNpGetAccountAge",
            "sceNpNotifyPremiumFeature",
            "sceNpRegisterPremiumEventCallback",
            "sceNpUnregisterPremiumEventCallback",
        ] {
            assert!(
                registry.is_implemented("libSceNpManager", name),
                "{name} must be registered"
            );
        }
    }

    #[test]
    fn get_state_reports_signed_out() {
        let kernel = raeen_kernel::OrbisKernel::new();
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

    /// A registered NP state callback must be DELIVERED the initial account
    /// state through `sceNpCheckCallback`, once, with the right per-form ABI —
    /// not silently accepted and never fired. This was Minecraft's post-menu
    /// wall: it registered a state callback and pumped `sceNpCheckCallback`
    /// ~10x/s forever with a blank UI, waiting for an event that never came.
    #[test]
    fn state_callback_is_delivered_the_initial_state_through_check_callback() {
        use crate::{GuestCallRequest, GuestCallScheduler};
        use std::sync::Mutex;

        #[derive(Default)]
        struct Recorder {
            calls: Mutex<Vec<GuestCallRequest>>,
        }
        impl GuestCallScheduler for Recorder {
            fn request(&self, request: GuestCallRequest) -> bool {
                self.calls.lock().unwrap().push(request);
                true
            }
        }

        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let recorder = Recorder::default();
        let mut ctx = test_ctx(&kernel, &mem, &alloc);
        ctx.guest_calls = &recorder;

        // Legacy 4-arg form: (userId, state, npId*, userdata).
        assert_eq!(
            hle_register_callback_legacy(&ctx, &[0xCAFE, 0x1234]),
            SCE_OK
        );
        // No delivery until the title pumps.
        assert!(recorder.calls.lock().unwrap().is_empty());

        assert_eq!(hle_check_callback(&ctx, &[]), SCE_OK);
        let calls = recorder.calls.lock().unwrap();
        assert_eq!(calls.len(), 1, "exactly one delivery");
        assert_eq!(calls[0].entry, 0xCAFE);
        assert_eq!(calls[0].args[0], PRIMARY_USER_ID);
        assert_eq!(calls[0].args[1], u64::from(NP_STATE_SIGNED_OUT));
        assert_eq!(calls[0].args[2], 0, "legacy npId arg is NULL");
        assert_eq!(calls[0].args[3], 0x1234, "userdata after npId");
        drop(calls);

        // A second pump must NOT re-deliver — the event is one-shot.
        assert_eq!(hle_check_callback(&ctx, &[]), SCE_OK);
        assert_eq!(recorder.calls.lock().unwrap().len(), 1, "delivered once");
    }

    /// The A/toolkit 3-arg form places userdata immediately after state, with
    /// no `npId` slot — delivering the legacy layout to it would hand the
    /// callback the NULL npId as its userdata.
    #[test]
    fn a_form_callback_omits_the_np_id_argument() {
        use crate::{GuestCallRequest, GuestCallScheduler};
        use std::sync::Mutex;

        #[derive(Default)]
        struct Recorder {
            calls: Mutex<Vec<GuestCallRequest>>,
        }
        impl GuestCallScheduler for Recorder {
            fn request(&self, request: GuestCallRequest) -> bool {
                self.calls.lock().unwrap().push(request);
                true
            }
        }

        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let recorder = Recorder::default();
        let mut ctx = test_ctx(&kernel, &mem, &alloc);
        ctx.guest_calls = &recorder;

        assert_eq!(hle_register_callback_a(&ctx, &[0xBEEF, 0x99]), SCE_OK);
        assert_eq!(hle_check_callback(&ctx, &[]), SCE_OK);
        let calls = recorder.calls.lock().unwrap();
        assert_eq!(calls[0].entry, 0xBEEF);
        assert_eq!(calls[0].args[1], u64::from(NP_STATE_SIGNED_OUT));
        assert_eq!(calls[0].args[2], 0x99, "userdata directly after state");
    }

    #[test]
    fn get_account_country_writes_us() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
    fn get_account_id_a_writes_a_stable_nonzero_id() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_account_id_a(&ctx, &[1000, 0x100]), SCE_OK);
        let mut id = [0u8; 8];
        assert!(mem.read(0x100, &mut id));
        let first = u64::from_le_bytes(id);
        assert_ne!(first, 0, "account id must be nonzero");
        // Stable: a second call reports the identical id.
        assert_eq!(hle_get_account_id_a(&ctx, &[1000, 0x108]), SCE_OK);
        assert!(mem.read(0x108, &mut id));
        assert_eq!(u64::from_le_bytes(id), first);
        // Invalid user or NULL out-ptr → argument error.
        assert_eq!(
            hle_get_account_id_a(&ctx, &[0xFFFF_FFFF, 0x100]),
            ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_get_account_id_a(&ctx, &[1000, 0]),
            ERROR_INVALID_ARGUMENT
        );
    }

    #[test]
    fn get_online_id_writes_placeholder() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_get_online_id(&ctx, &[1000, 0x100]), SCE_OK);
        let mut id = [0u8; 6];
        assert!(mem.read(0x100, &mut id));
        assert_eq!(&id, b"Player");
    }
}
