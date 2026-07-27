//! HLE libSceHttp + libSceHttp2 — the HTTP-client context/template lifecycle.
//!
//! A faithful Rust port of SharpEmu's `HttpExports` and `Http2Exports`
//! (GPL-2.0). A title creates an HTTP *context* (over a net-memory pool + SSL
//! context) and, for HTTP/1.1, one or more *templates* (per-endpoint request
//! settings). Raeen has no host network backend, so no request is ever sent —
//! but the context/template **bookkeeping** is real: ids are allocated
//! monotonically, validated on lookup, and cascade-freed on `Term`, so a title
//! that creates and tears down HTTP contexts behaves correctly up to the point
//! it actually issues a transfer.
//!
//! Context/template registries live in [`raeen_kernel::OrbisKernel`] (per
//! process), matching the [`crate::libsce_ampr`] pattern. `Init` returns the
//! new id in `rax`; error paths return the lib-specific codes ported verbatim
//! from SharpEmu (`libSceHttp` `0x8043_1100`/`0x8043_11FE`, `libSceHttp2`
//! `0x8043_6004`/`0x8043_6016`), all as plain zero-extended `u64`.

use crate::{HleContext, HleRegistry};
use std::sync::atomic::Ordering;

const OK: u64 = 0;
const HTTP_ERROR_INVALID_ID: u64 = 0x8043_1100;
const HTTP_ERROR_INVALID_VALUE: u64 = 0x8043_11FE;
/// `ORBIS_HTTP_ERROR_OUT_OF_SIZE` (shadPS4 `http_error.h`).
const HTTP_ERROR_OUT_OF_SIZE: u64 = 0x8043_1104;
const HTTP2_ERROR_INVALID_ID: u64 = 0x8043_6004;
const HTTP2_ERROR_INVALID_ARGUMENT: u64 = 0x8043_6016;
/// `SCE_HTTP2_ERROR_CANNOT_CONNECT` — no host-network backend exists, so a
/// request can never be sent. Reported instead of a live request id.
const HTTP2_ERROR_CANNOT_CONNECT: u64 = 0x8043_6023;

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
    registry.register("libSceHttp", "sceHttpUriEscape", hle_http_uri_escape);
    registry.register("libSceHttp2", "sceHttp2Init", hle_http2_init);
    registry.register(
        "libSceHttp2",
        "sceHttp2CreateTemplate",
        hle_http2_create_template,
    );
    registry.register(
        "libSceHttp2",
        "sceHttp2CreateRequestWithURL",
        hle_http2_create_request_with_url,
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

/// Bound on the input string scanned by [`hle_http_uri_escape`], so a guest
/// string missing its NUL cannot walk the whole address space.
const URI_ESCAPE_MAX_INPUT: usize = 64 * 1024;

/// `sceHttpUriEscape(char *out, size_t *require, size_t prepare, const char
/// *in)` — a REAL implementation (RFC 3986 percent-encoding of everything but
/// the unreserved set). Purely local string work, no network. Behavior
/// cross-checked against shadPS4's GPL-2.0 `http.cpp` (re-derived): `require`
/// receives the needed size including the NUL; a null `out` only computes the
/// size; an undersized `prepare` is an out-of-size error.
fn hle_http_uri_escape(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    let require = args.get(1).copied().unwrap_or(0);
    let prepare = args.get(2).copied().unwrap_or(0);
    let input = args.get(3).copied().unwrap_or(0);
    if input == 0 {
        return HTTP_ERROR_INVALID_VALUE;
    }
    // Read the guest string, bounded.
    let mut bytes = Vec::new();
    let mut cursor = input;
    loop {
        let mut chunk = [0u8; 64];
        if !ctx.mem.read(cursor, &mut chunk) {
            return HTTP_ERROR_INVALID_VALUE;
        }
        if let Some(nul) = chunk.iter().position(|&b| b == 0) {
            bytes.extend_from_slice(&chunk[..nul]);
            break;
        }
        bytes.extend_from_slice(&chunk);
        if bytes.len() > URI_ESCAPE_MAX_INPUT {
            return HTTP_ERROR_INVALID_VALUE;
        }
        cursor += chunk.len() as u64;
    }
    let is_unreserved =
        |c: u8| c.is_ascii_alphanumeric() || c == b'-' || c == b'_' || c == b'.' || c == b'~';
    let mut escaped = Vec::with_capacity(bytes.len());
    for &c in &bytes {
        if is_unreserved(c) {
            escaped.push(c);
        } else {
            escaped.push(b'%');
            escaped.push(b"0123456789ABCDEF"[usize::from(c >> 4)]);
            escaped.push(b"0123456789ABCDEF"[usize::from(c & 0xF)]);
        }
    }
    escaped.push(0);
    if require != 0
        && !ctx
            .mem
            .write(require, &(escaped.len() as u64).to_le_bytes())
    {
        return HTTP_ERROR_INVALID_VALUE;
    }
    if out == 0 {
        return OK; // size-query mode
    }
    if (prepare as usize) < escaped.len() {
        return HTTP_ERROR_OUT_OF_SIZE;
    }
    if !ctx.mem.write(out, &escaped) {
        return HTTP_ERROR_INVALID_VALUE;
    }
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

/// `sceHttp2CreateRequestWithURL(templateId, method, url, contentLength)`.
///
/// Reports "cannot connect": Raeen has no host-network backend, so a request
/// that cannot be sent must not be handed back as a live one. A plausible
/// request id would be worse than an error — the title would then poll a
/// response that can never arrive.
///
/// Registering it AT ALL is the point. It was the one import Minecraft actually
/// called that had none, and an unresolved import is not a no-op: the call lands
/// on the unresolved-stub guard page, faults, and KILLS the calling guest thread
/// (`RuntimeError::UnimplementedImport`). Measured — the fault timestamp is
/// exactly when the title's activity stops. A named error lets the thread live
/// and take its own offline path.
fn hle_http2_create_request_with_url(ctx: &HleContext, args: &[u64]) -> u64 {
    let template_id = args.first().copied().unwrap_or(0) as i32;
    if !ctx.kernel.http2_templates.contains_key(&template_id) {
        return HTTP2_ERROR_INVALID_ID;
    }
    tracing::debug!(
        template_id,
        "sceHttp2CreateRequestWithURL -> ERROR_CANNOT_CONNECT (offline: no host-network backend)"
    );
    HTTP2_ERROR_CANNOT_CONNECT
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

    /// `sceHttpUriEscape` is a real percent-encoder: unreserved bytes pass,
    /// everything else becomes `%XX`, `require` reports the size including
    /// NUL, a null `out` is size-query mode, and an undersized `prepare` is
    /// an out-of-size error.
    #[test]
    fn http_uri_escape_percent_encodes() {
        use crate::GuestMemory;
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"a b/c~\0"));
        // Size query: "a%20b%2Fc~" + NUL = 11.
        assert_eq!(hle_http_uri_escape(&ctx, &[0, 0x200, 0, 0x100]), OK);
        let mut req = [0u8; 8];
        assert!(mem.read(0x200, &mut req));
        assert_eq!(u64::from_le_bytes(req), 11);

        // Undersized buffer refused.
        assert_eq!(
            hle_http_uri_escape(&ctx, &[0x300, 0, 10, 0x100]),
            HTTP_ERROR_OUT_OF_SIZE
        );
        // Correctly sized buffer receives the escaped NUL-terminated string.
        assert_eq!(hle_http_uri_escape(&ctx, &[0x300, 0, 11, 0x100]), OK);
        let mut out = [0u8; 11];
        assert!(mem.read(0x300, &mut out));
        assert_eq!(&out, b"a%20b%2Fc~\0");

        // Null input string refused.
        assert_eq!(
            hle_http_uri_escape(&ctx, &[0x300, 0, 11, 0]),
            HTTP_ERROR_INVALID_VALUE
        );
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
        assert!(registry.is_implemented("libSceHttp2", "sceHttp2CreateTemplate"));

        assert_eq!(hle_http2_term(&ctx, &[1]), OK);
        assert!(!kernel.http2_templates.contains_key(&0x1001));
        assert_eq!(hle_http2_term(&ctx, &[1]), HTTP2_ERROR_INVALID_ID);
    }
}
