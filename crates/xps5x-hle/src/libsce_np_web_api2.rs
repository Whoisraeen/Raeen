//! HLE libSceNpWebApi2 — the PSN WebAPI2 library init/term handshake.
//!
//! A faithful Rust port of SharpEmu's `NpWebApi2Exports` (GPL-2.0). WebAPI2 is
//! the PSN REST client. XPS5X has no PSN backend and issues no requests, so
//! this is an honest handshake stub: `Initialize` validates its arguments and
//! records an initialized flag, `Terminate` clears it, and `CreateUserContext`
//! *refuses* (no live session) so a title backs off to offline. No HTTP request
//! is ever made (that would need the real `libSceHttp` + network backend).
//!
//! The invalid-argument code `0x8055_3402` is ported verbatim from SharpEmu,
//! as a plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::{AtomicI32, Ordering};

const OK: u64 = 0;
const NP_WEB_API2_ERROR_INVALID_ARGUMENT: u64 = 0x8055_3402;

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
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            xps5x_kernel::OrbisKernel::new(),
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
