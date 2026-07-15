//! HLE libSceHttp + libSceHttp2 — the HTTP-client context/template lifecycle.
//!
//! A faithful Rust port of SharpEmu's `HttpExports` and `Http2Exports`
//! (GPL-2.0). A title creates an HTTP *context* (over a net-memory pool + SSL
//! context) and, for HTTP/1.1, one or more *templates* (per-endpoint request
//! settings). XPS5X has no host network backend, so no request is ever sent —
//! but the context/template **bookkeeping** is real: ids are allocated
//! monotonically, validated on lookup, and cascade-freed on `Term`, so a title
//! that creates and tears down HTTP contexts behaves correctly up to the point
//! it actually issues a transfer.
//!
//! Context/template registries live in [`xps5x_kernel::OrbisKernel`] (per
//! process), matching the [`crate::libsce_ampr`] pattern. `Init` returns the
//! new id in `rax`; error paths return the lib-specific codes ported verbatim
//! from SharpEmu (`libSceHttp` `0x8043_1100`/`0x8043_11FE`, `libSceHttp2`
//! `0x8043_6004`/`0x8043_6016`), all as plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::Ordering;

const OK: u64 = 0;
const HTTP_ERROR_INVALID_ID: u64 = 0x8043_1100;
const HTTP_ERROR_INVALID_VALUE: u64 = 0x8043_11FE;
const HTTP2_ERROR_INVALID_ID: u64 = 0x8043_6004;
const HTTP2_ERROR_INVALID_ARGUMENT: u64 = 0x8043_6016;

/// Register the libSceHttp and libSceHttp2 functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceHttp", "sceHttpInit", hle_http_init);
    registry.register(
        "libSceHttp",
        "sceHttpCreateTemplate",
        hle_http_create_template,
    );
    registry.register(
        "libSceHttp",
        "sceHttpDeleteTemplate",
        hle_http_delete_template,
    );
    registry.register("libSceHttp", "sceHttpTerm", hle_http_term);
    registry.register("libSceHttp2", "sceHttp2Init", hle_http2_init);
    registry.register_nid(
        "libSceHttp2",
        "sceHttp2CreateTemplate",
        0xfb00_aded_f0a2_8e09,
        hle_http2_create_template,
    );
    registry.register("libSceHttp2", "sceHttp2Term", hle_http2_term);
}

/// `sceHttpInit(netMemoryId, sslContextId, poolSize)`: a zero pool size is an
/// invalid-value error; otherwise a fresh context id (≥ 1) is allocated,
/// recorded, and returned in `rax`.
fn hle_http_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let pool_size = args.get(2).copied().unwrap_or(0);
    if pool_size == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    let id = ctx.kernel.http_next_context.fetch_add(1, Ordering::Relaxed) + 1;
    ctx.kernel.http_contexts.insert(id, pool_size);
    id as u32 as u64
}

/// `sceHttpCreateTemplate(contextId, userAgent, httpVersion, autoProxyConfig)`:
/// an unknown context id is an invalid-id error; otherwise a fresh template id
/// (≥ 0x1001) is allocated, recorded against its context, and returned.
fn hle_http_create_template(ctx: &HleContext, args: &[u64]) -> u64 {
    let context_id = args.first().copied().unwrap_or(0) as i32;
    if !ctx.kernel.http_contexts.contains_key(&context_id) {
        return HTTP_ERROR_INVALID_ID;
    }
    let id = ctx
        .kernel
        .http_next_template
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    ctx.kernel.http_templates.insert(id, context_id);
    id as u32 as u64
}

/// `sceHttpDeleteTemplate(templateId)`: removes the template, or reports an
/// invalid-id error if it was not registered.
fn hle_http_delete_template(ctx: &HleContext, args: &[u64]) -> u64 {
    let template_id = args.first().copied().unwrap_or(0) as i32;
    if ctx.kernel.http_templates.remove(&template_id).is_some() {
        OK
    } else {
        HTTP_ERROR_INVALID_ID
    }
}

/// `sceHttpTerm(contextId)`: removes the context and cascade-removes every
/// template that belonged to it; an unknown context id is an invalid-id error.
fn hle_http_term(ctx: &HleContext, args: &[u64]) -> u64 {
    let context_id = args.first().copied().unwrap_or(0) as i32;
    if ctx.kernel.http_contexts.remove(&context_id).is_none() {
        return HTTP_ERROR_INVALID_ID;
    }
    ctx.kernel
        .http_templates
        .retain(|_, owner| *owner != context_id);
    OK
}

/// `sceHttp2Init(netId, sslId, poolSize, maxRequests)`: a zero pool size or
/// non-positive max-requests is an invalid-argument error; otherwise a fresh
/// context id (≥ 1) is allocated and returned.
fn hle_http2_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let pool_size = args.get(2).copied().unwrap_or(0);
    let max_requests = args.get(3).copied().unwrap_or(0) as i32;
    if pool_size == 0 || max_requests <= 0 {
        return HTTP2_ERROR_INVALID_ARGUMENT;
    }
    let id = ctx
        .kernel
        .http2_next_context
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    ctx.kernel.http2_contexts.insert(id, pool_size);
    id as u32 as u64
}

/// `sceHttp2CreateTemplate(contextId, userAgent, httpVersion,
/// autoProxyConfig)`: allocate a template owned by a live HTTP2 context.
fn hle_http2_create_template(ctx: &HleContext, args: &[u64]) -> u64 {
    let context_id = args.first().copied().unwrap_or(0) as i32;
    if !ctx.kernel.http2_contexts.contains_key(&context_id) {
        return HTTP2_ERROR_INVALID_ID;
    }
    let id = ctx
        .kernel
        .http2_next_template
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    ctx.kernel.http2_templates.insert(id, context_id);
    id as u32 as u64
}

/// `sceHttp2Term(contextId)`: removes the context, or reports an invalid-id
/// error if it was not registered.
fn hle_http2_term(ctx: &HleContext, args: &[u64]) -> u64 {
    let context_id = args.first().copied().unwrap_or(0) as i32;
    if ctx.kernel.http2_contexts.remove(&context_id).is_none() {
        return HTTP2_ERROR_INVALID_ID;
    }
    ctx.kernel
        .http2_templates
        .retain(|_, owner| *owner != context_id);
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
    fn http_context_and_template_lifecycle() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Zero pool size rejected.
        assert_eq!(hle_http_init(&ctx, &[0, 0, 0]), HTTP_ERROR_INVALID_VALUE);
        // First context id is 1, second is 2.
        assert_eq!(hle_http_init(&ctx, &[0, 0, 0x1000]), 1);
        assert_eq!(hle_http_init(&ctx, &[0, 0, 0x1000]), 2);

        // Template against unknown context rejected; against a real one → 0x1001.
        assert_eq!(
            hle_http_create_template(&ctx, &[99, 0, 1, 0]),
            HTTP_ERROR_INVALID_ID
        );
        assert_eq!(hle_http_create_template(&ctx, &[1, 0, 1, 0]), 0x1001);
        assert_eq!(hle_http_create_template(&ctx, &[1, 0, 1, 0]), 0x1002);

        // Delete a template; deleting it again is an invalid id.
        assert_eq!(hle_http_delete_template(&ctx, &[0x1001]), OK);
        assert_eq!(
            hle_http_delete_template(&ctx, &[0x1001]),
            HTTP_ERROR_INVALID_ID
        );

        // Term of context 1 cascade-removes its remaining template (0x1002).
        assert!(kernel.http_templates.contains_key(&0x1002));
        assert_eq!(hle_http_term(&ctx, &[1]), OK);
        assert!(!kernel.http_templates.contains_key(&0x1002));
        // Second Term of the same context → invalid id.
        assert_eq!(hle_http_term(&ctx, &[1]), HTTP_ERROR_INVALID_ID);
        // Context 2 is untouched.
        assert!(kernel.http_contexts.contains_key(&2));
    }

    #[test]
    fn http2_context_lifecycle() {
        let (kernel, mem, alloc) = env();
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(
            hle_http2_init(&ctx, &[0, 0, 0, 4]),
            HTTP2_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_http2_init(&ctx, &[0, 0, 0x1000, 0]),
            HTTP2_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(hle_http2_init(&ctx, &[0, 0, 0x1000, 4]), 1);
        assert_eq!(
            hle_http2_create_template(&ctx, &[99, 0x100, 3, 1]),
            HTTP2_ERROR_INVALID_ID
        );
        assert_eq!(hle_http2_create_template(&ctx, &[1, 0x100, 3, 1]), 0x1001);
        assert!(kernel.http2_templates.contains_key(&0x1001));

        let registry = HleRegistry::new();
        register(&registry);
        assert!(
            registry
                .registered_nid_overrides()
                .iter()
                .any(|(nid, key)| {
                    *nid == 0xfb00_aded_f0a2_8e09 && key == "libSceHttp2::sceHttp2CreateTemplate"
                })
        );

        assert_eq!(hle_http2_term(&ctx, &[1]), OK);
        assert!(!kernel.http2_templates.contains_key(&0x1001));
        assert_eq!(hle_http2_term(&ctx, &[1]), HTTP2_ERROR_INVALID_ID);
    }
}
