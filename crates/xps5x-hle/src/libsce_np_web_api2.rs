//! HLE libSceNpWebApi2 — the PSN WebAPI2 library init/term handshake.
//!
//! A faithful Rust port of SharpEmu's `NpWebApi2Exports` (GPL-2.0). WebAPI2 is
//! the PSN REST client. XPS5X has no PSN backend and issues no requests, so
//! this is an honest handshake stub: `Initialize` validates its arguments and
//! records an initialized flag, and `Terminate` clears it. No HTTP request is
//! ever made (that would need the real `libSceHttp` + network backend).
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
}

/// `sceNpWebApi2CreateUserContext(...)`: hand back a fresh positive user
/// context id — the "signed-in local user" model (offline; the id carries no
/// network identity, but a title that gates on having *a* context proceeds).
static NEXT_USER_CONTEXT: AtomicI32 = AtomicI32::new(1);

fn hle_create_user_context(_ctx: &HleContext, args: &[u64]) -> u64 {
    let library_context_id = args.first().copied().unwrap_or(0) as i32;
    tracing::debug!("sceNpWebApi2CreateUserContext(libraryContextId={library_context_id})");
    NEXT_USER_CONTEXT.fetch_add(1, Ordering::Relaxed) as u64
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
}
