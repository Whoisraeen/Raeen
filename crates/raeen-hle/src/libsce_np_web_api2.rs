//! HLE libSceNpWebApi2 — the PSN WebAPI2 library init/term handshake.
//!
//! A faithful Rust port of SharpEmu's `NpWebApi2Exports` (GPL-2.0). WebAPI2 is
//! the PSN REST client. Raeen has no PSN backend and issues no requests, so
//! this is an honest handshake stub: `Initialize` validates its arguments and
//! records an initialized flag, `Terminate` clears it, and `CreateUserContext`
//! *refuses* (no live session) so a title backs off to offline. No HTTP request
//! is ever made (that would need the real `libSceHttp` + network backend).
//!
//! **Offline request/push-event model (Tier B, 2026-07-27):** because
//! `CreateUserContext` always refuses, no user context — and therefore no
//! request, push context, filter, or handle — can ever exist in this process.
//! Every function keyed on one of those objects reports the matching
//! `*_NOT_FOUND` error, and every push-channel *creation* entry point reports
//! `NOT_SIGNED_IN`. Nothing here fabricates a PSN response; a title's online
//! layer sees a consistent "signed out, nothing exists" world and backs off.
//! Error values cross-checked against shadPS4's GPL-2.0 `np_error.h`
//! (`ORBIS_NP_WEBAPI2_ERROR_*`, re-derived as plain zero-extended `u64`).
//!
//! The invalid-argument code `0x8055_3402` is ported verbatim from SharpEmu,
//! as a plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};

const OK: u64 = 0;
const NP_WEB_API2_ERROR_INVALID_ARGUMENT: u64 = 0x8055_3402;
/// `ORBIS_NP_WEBAPI2_ERROR_USER_CONTEXT_NOT_FOUND` (shadPS4 `np_error.h`).
const NP_WEB_API2_ERROR_USER_CONTEXT_NOT_FOUND: u64 = 0x8055_3405;
/// `ORBIS_NP_WEBAPI2_ERROR_REQUEST_NOT_FOUND`.
const NP_WEB_API2_ERROR_REQUEST_NOT_FOUND: u64 = 0x8055_3406;
/// `ORBIS_NP_WEBAPI2_ERROR_NOT_SIGNED_IN`.
const NP_WEB_API2_ERROR_NOT_SIGNED_IN: u64 = 0x8055_3407;
/// `ORBIS_NP_WEBAPI2_ERROR_PUSH_EVENT_FILTER_NOT_FOUND`.
const NP_WEB_API2_ERROR_PUSH_EVENT_FILTER_NOT_FOUND: u64 = 0x8055_340b;
/// `ORBIS_NP_WEBAPI2_ERROR_PUSH_EVENT_CALLBACK_NOT_FOUND`.
const NP_WEB_API2_ERROR_PUSH_EVENT_CALLBACK_NOT_FOUND: u64 = 0x8055_340c;
/// `ORBIS_NP_WEBAPI2_ERROR_HANDLE_NOT_FOUND`.
const NP_WEB_API2_ERROR_HANDLE_NOT_FOUND: u64 = 0x8055_340d;

// SharpEmu's `_initialized` flag (0 = down, 1 = initialized).
static INITIALIZED: AtomicI32 = AtomicI32::new(0);

/// Register the libSceNpWebApi2 functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceNpWebApi2", "sceNpWebApi2Initialize", hle_initialize);
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2InitializeForToolkit",
        hle_initialize_for_toolkit,
    );
    registry.register("libSceNpWebApi2", "sceNpWebApi2Terminate", hle_terminate);
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2CreateUserContext",
        hle_create_user_context,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2PushEventCreateHandle",
        hle_push_event_create_handle,
    );

    // Request lifecycle — no user context can exist (CreateUserContext refuses
    // above), so requests can never be created; everything keyed on a request
    // or user context reports the object as not found. GTA V imports all of
    // these (docs/gta5-blocker-analysis-2026-07-27.md).
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2CreateRequest",
        hle_create_request,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2DeleteRequest",
        hle_request_not_found,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2AbortRequest",
        hle_request_not_found,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2SendRequest",
        hle_request_not_found,
    );
    registry.register("libSceNpWebApi2", "sceNpWebApi2ReadData", hle_read_data);
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2AddHttpRequestHeader",
        hle_add_http_request_header,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2GetHttpResponseHeaderValue",
        hle_request_not_found,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2GetHttpResponseHeaderValueLength",
        hle_request_not_found,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2DeleteUserContext",
        hle_user_context_not_found,
    );

    // Push-event channel — creation refuses with NOT_SIGNED_IN (a live PSN
    // push channel cannot exist offline; a handle that "works" would strand
    // the title waiting on events that never arrive), and deletion/unregister
    // reports the object as not found, consistent with nothing having been
    // created.
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2PushEventCreatePushContext",
        hle_push_event_not_signed_in,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2PushEventCreateFilter",
        hle_push_event_not_signed_in,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2PushEventRegisterCallback",
        hle_push_event_not_signed_in,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2PushEventStartPushContextCallback",
        hle_push_event_not_signed_in,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2PushEventDeletePushContext",
        hle_push_event_handle_not_found,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2PushEventDeleteFilter",
        hle_push_event_filter_not_found,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2PushEventDeleteHandle",
        hle_push_event_handle_not_found,
    );
    registry.register(
        "libSceNpWebApi2",
        "sceNpWebApi2PushEventUnregisterCallback",
        hle_push_event_callback_not_found,
    );
}

/// `sceNpWebApi2CreateRequest(userCtxId, apiGroup, path, method,
/// contentParameter, requestId*)`: null string arguments are an argument
/// error (matching shadPS4's validation order); otherwise the user context
/// cannot exist offline, so the request is refused with
/// `USER_CONTEXT_NOT_FOUND` and no request id is written.
fn hle_create_request(_ctx: &HleContext, args: &[u64]) -> u64 {
    let api_group = args.get(1).copied().unwrap_or(0);
    let path = args.get(2).copied().unwrap_or(0);
    let method = args.get(3).copied().unwrap_or(0);
    if api_group == 0 || path == 0 || method == 0 {
        return NP_WEB_API2_ERROR_INVALID_ARGUMENT;
    }
    tracing::debug!(
        "sceNpWebApi2CreateRequest(userCtxId={}) -> USER_CONTEXT_NOT_FOUND (offline)",
        args.first().copied().unwrap_or(0) as i32
    );
    NP_WEB_API2_ERROR_USER_CONTEXT_NOT_FOUND
}

/// `sceNpWebApi2ReadData(requestId, data, size)`: null buffer / zero size is
/// an argument error; otherwise no request exists to read from.
fn hle_read_data(_ctx: &HleContext, args: &[u64]) -> u64 {
    let data = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    if data == 0 || size == 0 {
        return NP_WEB_API2_ERROR_INVALID_ARGUMENT;
    }
    tracing::debug!("sceNpWebApi2ReadData -> REQUEST_NOT_FOUND (offline)");
    NP_WEB_API2_ERROR_REQUEST_NOT_FOUND
}

/// `sceNpWebApi2AddHttpRequestHeader(requestId, fieldName, fieldValue)`:
/// null header name/value is an argument error; otherwise the request cannot
/// exist.
fn hle_add_http_request_header(_ctx: &HleContext, args: &[u64]) -> u64 {
    let field_name = args.get(1).copied().unwrap_or(0);
    let field_value = args.get(2).copied().unwrap_or(0);
    if field_name == 0 || field_value == 0 {
        return NP_WEB_API2_ERROR_INVALID_ARGUMENT;
    }
    NP_WEB_API2_ERROR_REQUEST_NOT_FOUND
}

/// Request-keyed operations (`Send`/`Abort`/`Delete`/`GetHttpResponseHeader*`):
/// no request can ever have been created, so the request is not found.
fn hle_request_not_found(_ctx: &HleContext, args: &[u64]) -> u64 {
    tracing::debug!(
        "sceNpWebApi2 request op(requestId={:#x}) -> REQUEST_NOT_FOUND (offline)",
        args.first().copied().unwrap_or(0)
    );
    NP_WEB_API2_ERROR_REQUEST_NOT_FOUND
}

/// `sceNpWebApi2DeleteUserContext(userCtxId)`: no user context exists.
fn hle_user_context_not_found(_ctx: &HleContext, _args: &[u64]) -> u64 {
    NP_WEB_API2_ERROR_USER_CONTEXT_NOT_FOUND
}

/// Push-channel creation (`CreatePushContext`/`CreateFilter`/
/// `RegisterCallback`/`StartPushContextCallback`): refuse — signed out. A
/// fabricated push channel would leave the title waiting for events that can
/// never arrive; the offline error makes it back off.
fn hle_push_event_not_signed_in(_ctx: &HleContext, _args: &[u64]) -> u64 {
    tracing::debug!("sceNpWebApi2PushEvent create/register -> NOT_SIGNED_IN (offline)");
    NP_WEB_API2_ERROR_NOT_SIGNED_IN
}

fn hle_push_event_handle_not_found(_ctx: &HleContext, _args: &[u64]) -> u64 {
    NP_WEB_API2_ERROR_HANDLE_NOT_FOUND
}

fn hle_push_event_filter_not_found(_ctx: &HleContext, _args: &[u64]) -> u64 {
    NP_WEB_API2_ERROR_PUSH_EVENT_FILTER_NOT_FOUND
}

fn hle_push_event_callback_not_found(_ctx: &HleContext, _args: &[u64]) -> u64 {
    NP_WEB_API2_ERROR_PUSH_EVENT_CALLBACK_NOT_FOUND
}

/// `sceNpWebApi2PushEventCreateHandle(...)`: refuse, for exactly the reason
/// [`hle_create_user_context`] refuses. A positive handle would tell the title a
/// live PSN push channel exists, and it would then wait on events that can never
/// arrive; an error makes its online layer back off to offline instead.
///
/// Measured: Until Dawn stops its boot on this import.
fn hle_push_event_create_handle(_ctx: &HleContext, _args: &[u64]) -> u64 {
    tracing::debug!("sceNpWebApi2PushEventCreateHandle -> INVALID_ARGUMENT (offline)");
    NP_WEB_API2_ERROR_INVALID_ARGUMENT
}

/// `sceNpWebApi2CreateUserContext(...)`: with no PSN backend, **refuse** to
/// create a user context (return `INVALID_ARGUMENT`). Handing back a positive
/// handle makes a title believe its online session is live and drive follow-up
/// WebAPI calls that can never complete — ASTRO.BOT then hard-asserts at
/// `NpWebApi.cpp:1587` (fatal: its assert handler traps the main thread, which
/// strands every worker parked on a job semaphore). Failing here makes the
/// title's online layer back off cleanly to offline. Matches SharpEmu's
/// `NpWebApi2CreateUserContext` (GPL-2.0).
fn hle_create_user_context(_ctx: &HleContext, args: &[u64]) -> u64 {
    let library_context_id = args.first().copied().unwrap_or(0) as i32;
    tracing::debug!(
        "sceNpWebApi2CreateUserContext(libraryContextId={library_context_id}) \
         -> INVALID_ARGUMENT (offline; title backs off)"
    );
    NP_WEB_API2_ERROR_INVALID_ARGUMENT
}

/// `sceNpWebApi2Initialize(httpContextId, poolSize)`: a non-positive HTTP
/// context id or a zero pool size is an invalid-argument error; otherwise the
/// library is marked initialized.
fn hle_initialize(_ctx: &HleContext, args: &[u64]) -> u64 {
    let http_context_id = args.first().copied().unwrap_or(0) as i32;
    let pool_size = args.get(1).copied().unwrap_or(0);
    if http_context_id <= 0 || pool_size == 0 {
        return NP_WEB_API2_ERROR_INVALID_ARGUMENT;
    }
    INITIALIZED.store(1, Ordering::Relaxed);
    OK
}

/// `sceNpWebApi2InitializeForToolkit(...)`: the Toolkit entry point, which
/// SharpEmu accepts unconditionally.
fn hle_initialize_for_toolkit(_ctx: &HleContext, _args: &[u64]) -> u64 {
    INITIALIZED.store(1, Ordering::Relaxed);
    OK
}

/// `sceNpWebApi2Terminate(libraryContextId)`: clears the initialized flag.
fn hle_terminate(_ctx: &HleContext, _args: &[u64]) -> u64 {
    INITIALIZED.store(0, Ordering::Relaxed);
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    fn env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            raeen_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x10),
            crate::TestAllocator::new(0),
        )
    }

    #[test]
    fn initialize_validates_and_terminate_clears() {
        INITIALIZED.store(0, Ordering::Relaxed);
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_initialize(&ctx, &[0, 0x1000]),
            NP_WEB_API2_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_initialize(&ctx, &[5, 0]),
            NP_WEB_API2_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(hle_initialize(&ctx, &[5, 0x1000]), OK);
        assert_eq!(INITIALIZED.load(Ordering::Relaxed), 1);

        assert_eq!(hle_terminate(&ctx, &[5]), OK);
        assert_eq!(INITIALIZED.load(Ordering::Relaxed), 0);

        // The Toolkit variant accepts anything.
        assert_eq!(hle_initialize_for_toolkit(&ctx, &[0, 0]), OK);
        assert_eq!(INITIALIZED.load(Ordering::Relaxed), 1);
    }

    /// Offline request model: string arguments are validated, then the user
    /// context (which can never exist) is reported not-found — never a bogus
    /// request id the title would then drive.
    #[test]
    fn request_lifecycle_reports_consistent_offline_not_found() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Null apiGroup/path/method -> INVALID_ARGUMENT.
        assert_eq!(
            hle_create_request(&ctx, &[1, 0, 0x20, 0x30, 0, 0x40]),
            NP_WEB_API2_ERROR_INVALID_ARGUMENT
        );
        // Valid pointers -> the (nonexistent) user context is not found.
        assert_eq!(
            hle_create_request(&ctx, &[1, 0x10, 0x20, 0x30, 0, 0x40]),
            NP_WEB_API2_ERROR_USER_CONTEXT_NOT_FOUND
        );
        // Everything request-keyed reports REQUEST_NOT_FOUND.
        assert_eq!(
            hle_request_not_found(&ctx, &[7]),
            NP_WEB_API2_ERROR_REQUEST_NOT_FOUND
        );
        assert_eq!(
            hle_read_data(&ctx, &[7, 0x100, 0x10]),
            NP_WEB_API2_ERROR_REQUEST_NOT_FOUND
        );
        assert_eq!(
            hle_read_data(&ctx, &[7, 0, 0x10]),
            NP_WEB_API2_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_add_http_request_header(&ctx, &[7, 0x100, 0x110]),
            NP_WEB_API2_ERROR_REQUEST_NOT_FOUND
        );
        assert_eq!(
            hle_add_http_request_header(&ctx, &[7, 0, 0x110]),
            NP_WEB_API2_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_user_context_not_found(&ctx, &[3]),
            NP_WEB_API2_ERROR_USER_CONTEXT_NOT_FOUND
        );
    }

    /// Push-channel creation refuses (signed out) and deletion reports the
    /// matching not-found code — nothing pretends a live PSN push channel
    /// exists.
    #[test]
    fn push_event_channel_refuses_offline() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_push_event_not_signed_in(&ctx, &[1]),
            NP_WEB_API2_ERROR_NOT_SIGNED_IN
        );
        assert_eq!(
            hle_push_event_handle_not_found(&ctx, &[1]),
            NP_WEB_API2_ERROR_HANDLE_NOT_FOUND
        );
        assert_eq!(
            hle_push_event_filter_not_found(&ctx, &[1]),
            NP_WEB_API2_ERROR_PUSH_EVENT_FILTER_NOT_FOUND
        );
        assert_eq!(
            hle_push_event_callback_not_found(&ctx, &[1]),
            NP_WEB_API2_ERROR_PUSH_EVENT_CALLBACK_NOT_FOUND
        );
    }

    /// Every measured GTA V libSceNpWebApi2 import resolves.
    #[test]
    fn measured_gta5_imports_are_registered() {
        let registry = HleRegistry::new();
        for name in [
            "sceNpWebApi2CreateRequest",
            "sceNpWebApi2DeleteRequest",
            "sceNpWebApi2AbortRequest",
            "sceNpWebApi2SendRequest",
            "sceNpWebApi2ReadData",
            "sceNpWebApi2AddHttpRequestHeader",
            "sceNpWebApi2GetHttpResponseHeaderValue",
            "sceNpWebApi2GetHttpResponseHeaderValueLength",
            "sceNpWebApi2DeleteUserContext",
            "sceNpWebApi2PushEventCreatePushContext",
            "sceNpWebApi2PushEventCreateFilter",
            "sceNpWebApi2PushEventRegisterCallback",
            "sceNpWebApi2PushEventStartPushContextCallback",
            "sceNpWebApi2PushEventDeletePushContext",
            "sceNpWebApi2PushEventDeleteFilter",
            "sceNpWebApi2PushEventDeleteHandle",
            "sceNpWebApi2PushEventUnregisterCallback",
        ] {
            assert!(
                registry.is_implemented("libSceNpWebApi2", name),
                "{name} must be registered"
            );
        }
    }

    #[test]
    fn create_user_context_refuses_offline() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // No PSN backend: creating a user context must fail so the title backs
        // off, rather than getting a bogus positive handle it drives online.
        assert_eq!(
            hle_create_user_context(&ctx, &[0, 1000]),
            NP_WEB_API2_ERROR_INVALID_ARGUMENT
        );
    }
}
