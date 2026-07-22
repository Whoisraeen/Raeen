//! HLE libSceSsl — the SSL/TLS context lifecycle.
//!
//! A faithful Rust port of SharpEmu's `SslExports` (GPL-2.0). `libSceSsl`
//! backs `libSceHttp`'s HTTPS support. Raeen has no TLS/network backend, so no
//! handshake is ever performed — but the SSL-context **bookkeeping** is real:
//! `Init` allocates and records a monotonic context id (returned in `rax`),
//! `Term` validates and removes it, and `Close` is an unconditional success
//! (matching SharpEmu, which does not validate the connection id there).
//!
//! Context ids live in [`raeen_kernel::OrbisKernel`] (per process). Error codes
//! are ported verbatim: `0x8095_F006` (invalid id), `0x8095_F008`
//! (out-of-size), as plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::Ordering;

const OK: u64 = 0;
const SSL_ERROR_INVALID_ID: u64 = 0x8095_F006;
const SSL_ERROR_OUT_OF_SIZE: u64 = 0x8095_F008;

/// Register the libSceSsl functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceSsl", "sceSslInit", hle_init);
    registry.register("libSceSsl", "sceSslTerm", hle_term);
    registry.register("libSceSsl", "sceSslClose", hle_close);
}

/// `sceSslInit(poolSize)`: a zero pool size is an out-of-size error; otherwise
/// a fresh context id (≥ 1) is allocated, recorded, and returned in `rax`.
fn hle_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let pool_size = args.first().copied().unwrap_or(0);
    if pool_size == 0 {
        return SSL_ERROR_OUT_OF_SIZE;
    }
    let id = ctx.kernel.ssl_next_context.fetch_add(1, Ordering::Relaxed) + 1;
    ctx.kernel.ssl_contexts.insert(id, pool_size);
    id as u32 as u64
}

/// `sceSslTerm(sslContextId)`: removes the context, or reports an invalid-id
/// error if it was not registered.
fn hle_term(ctx: &HleContext, args: &[u64]) -> u64 {
    let id = args.first().copied().unwrap_or(0) as i32;
    if ctx.kernel.ssl_contexts.remove(&id).is_some() {
        OK
    } else {
        SSL_ERROR_INVALID_ID
    }
}

/// `sceSslClose(sslConnectionId)`: an unconditional success (SharpEmu does not
/// track SSL connections separately from contexts).
fn hle_close(_ctx: &HleContext, _args: &[u64]) -> u64 {
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
    fn ssl_context_lifecycle() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_init(&ctx, &[0]), SSL_ERROR_OUT_OF_SIZE);
        assert_eq!(hle_init(&ctx, &[0x8000]), 1);
        assert_eq!(hle_init(&ctx, &[0x8000]), 2);

        // Close always succeeds, even for an unknown id.
        assert_eq!(hle_close(&ctx, &[999]), OK);

        assert_eq!(hle_term(&ctx, &[1]), OK);
        assert_eq!(hle_term(&ctx, &[1]), SSL_ERROR_INVALID_ID);
        assert!(kernel.ssl_contexts.contains_key(&2));
    }
}
