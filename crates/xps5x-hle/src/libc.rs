//! HLE libc — Standard C library re-implementation.
//!
//! Clean-room re-implementation of the PS5 `libc.sprx` exports. Function
//! *names* below are factual, standard C API identifiers (not copyrightable);
//! every implementation is original.
//!
//! ## Stub status
//!
//! Same caveat as [`crate::libkernel`]: [`crate::HleFunction`] is a bare
//! `fn(&[u64]) -> u64` with no access to guest memory, so string/buffer
//! functions here cannot actually read or write guest bytes. Each stub logs
//! the call and returns a plausible value (documented per-function below);
//! real semantics require a dispatch signature with guest memory access,
//! deferred to a later milestone.

use crate::HleRegistry;
use tracing::debug;

/// Register libc HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libc", "malloc", hle_malloc);
    registry.register("libc", "free", hle_free);
    registry.register("libc", "calloc", hle_calloc);
    registry.register("libc", "realloc", hle_realloc);
    registry.register("libc", "memcpy", hle_memcpy);
    registry.register("libc", "memset", hle_memset);
    registry.register("libc", "memmove", hle_memmove);
    registry.register("libc", "strlen", hle_strlen);
    registry.register("libc", "strcmp", hle_strcmp);
    registry.register("libc", "strcpy", hle_strcpy);
    registry.register("libc", "strncpy", hle_strncpy);
    registry.register("libc", "snprintf", hle_snprintf);
    registry.register("libc", "printf", hle_printf);
    registry.register("libc", "puts", hle_puts);
    registry.register("libc", "abort", hle_abort);
    registry.register("libc", "exit", hle_exit);
    registry.register("libc", "__stack_chk_fail", hle_stack_chk_fail);
    registry.register("libc", "memalign", hle_memalign);
    registry.register("libc", "posix_memalign", hle_posix_memalign);
}

/// A fake, always-the-same non-null "heap" address. No real allocation
/// backs it — there is no guest address space reachable from this stub.
const FAKE_HEAP_ADDR: u64 = 0x0000_7000_0000_0000;

fn hle_malloc(args: &[u64]) -> u64 {
    debug!("malloc(size={:#x})", args.first().copied().unwrap_or(0));
    FAKE_HEAP_ADDR
}

fn hle_free(args: &[u64]) -> u64 {
    debug!("free(ptr={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_calloc(args: &[u64]) -> u64 {
    debug!(
        "calloc(nmemb={}, size={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    FAKE_HEAP_ADDR
}

fn hle_realloc(args: &[u64]) -> u64 {
    debug!(
        "realloc(ptr={:#x}, size={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    FAKE_HEAP_ADDR
}

/// Real `memcpy` returns `dst` unchanged; this stub does the same without
/// actually copying any bytes (no guest memory access available here).
fn hle_memcpy(args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    debug!(
        "memcpy(dst={:#x}, src={:#x}, n={:#x})",
        dst,
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    dst
}

fn hle_memset(args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    debug!(
        "memset(s={:#x}, c={}, n={:#x})",
        dst,
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    dst
}

fn hle_memmove(args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    debug!(
        "memmove(dst={:#x}, src={:#x}, n={:#x})",
        dst,
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    dst
}

/// Placeholder: real `strlen` reads guest memory to find the NUL terminator,
/// which this stub cannot do. Always reports length `0`.
fn hle_strlen(args: &[u64]) -> u64 {
    debug!("strlen(s={:#x}) [placeholder: cannot read guest memory]", args.first().copied().unwrap_or(0));
    0
}

/// Placeholder: real `strcmp` reads both guest strings; this stub always
/// reports "equal" (`0`).
fn hle_strcmp(args: &[u64]) -> u64 {
    debug!(
        "strcmp(s1={:#x}, s2={:#x}) [placeholder: cannot read guest memory]",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    0
}

fn hle_strcpy(args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    debug!(
        "strcpy(dst={:#x}, src={:#x}) [placeholder: no bytes actually copied]",
        dst,
        args.get(1).copied().unwrap_or(0)
    );
    dst
}

fn hle_strncpy(args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    debug!(
        "strncpy(dst={:#x}, src={:#x}, n={:#x}) [placeholder: no bytes actually copied]",
        dst,
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    dst
}

/// Placeholder: real `snprintf` returns the number of characters that
/// would've been written; this stub reports `0` since it does no formatting.
fn hle_snprintf(args: &[u64]) -> u64 {
    debug!(
        "snprintf(buf={:#x}, size={:#x}, fmt={:#x}) [placeholder: no formatting performed]",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    0
}

fn hle_printf(args: &[u64]) -> u64 {
    debug!(
        "printf(fmt={:#x}) [placeholder: no formatting performed]",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_puts(args: &[u64]) -> u64 {
    debug!(
        "puts(s={:#x}) [placeholder: cannot read guest memory]",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_abort(_args: &[u64]) -> u64 {
    // Real `abort` never returns (raises SIGABRT). The stub cannot terminate
    // the guest process from here, so it just logs and returns.
    debug!("abort() [stub: does not actually terminate the process]");
    0
}

fn hle_exit(args: &[u64]) -> u64 {
    // Real `exit` never returns. Same limitation as `abort` above.
    debug!(
        "exit(code={}) [stub: does not actually terminate the process]",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_stack_chk_fail(_args: &[u64]) -> u64 {
    // Real `__stack_chk_fail` aborts the process on stack-smash detection.
    // The stub just logs — it cannot terminate the guest process.
    debug!("__stack_chk_fail() [stub: does not actually terminate the process]");
    0
}

fn hle_memalign(args: &[u64]) -> u64 {
    debug!(
        "memalign(alignment={:#x}, size={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    FAKE_HEAP_ADDR
}

fn hle_posix_memalign(args: &[u64]) -> u64 {
    // Real function writes the allocated pointer through `*memptr`; not
    // writable from this stub. Report success (0) only.
    debug!(
        "posix_memalign(memptr={:#x}, alignment={:#x}, size={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_adds_expected_functions() {
        let registry = HleRegistry::new();
        for name in [
            "malloc",
            "free",
            "calloc",
            "realloc",
            "memcpy",
            "memset",
            "memmove",
            "strlen",
            "strcmp",
            "strcpy",
            "strncpy",
            "snprintf",
            "printf",
            "puts",
            "abort",
            "exit",
            "__stack_chk_fail",
            "memalign",
            "posix_memalign",
        ] {
            assert!(registry.is_implemented("libc", name), "missing libc::{name}");
            registry.call("libc", name, &[1, 2, 3]);
        }
    }
}
