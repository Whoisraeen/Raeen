//! HLE libc — Standard C library re-implementation.
//!
//! Clean-room re-implementation of the PS5 `libc.sprx` exports. Function
//! *names* below are factual, standard C API identifiers (not copyrightable);
//! every implementation is original.
//!
//! ## Stub status
//!
//! Every HLE call now gets an [`crate::HleContext`] with guest-memory
//! access, so string/buffer functions can actually read/write guest bytes.
//! `memcpy`, `memset`, and `strlen` do the real operation below, bounds
//! -checked through [`crate::GuestMemory`]. The rest (`malloc`/`strcpy`/
//! `printf`/...) still log the call and return a plausible value — real
//! heap allocation and string/format handling are future work, not blocked
//! on the dispatch signature anymore.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

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

/// A fake, always-the-same non-null "heap" address. No real allocator backs
/// it yet — `malloc`/`calloc`/`realloc`/`memalign` don't reserve any actual
/// guest memory (that needs a heap allocator on top of
/// `ctx.kernel.memory`, a later milestone).
const FAKE_HEAP_ADDR: u64 = 0x0000_7000_0000_0000;

/// Cap on how far [`hle_strlen`] will scan looking for a NUL terminator, so
/// a wild/unterminated guest pointer can't spin forever. Arbitrary but
/// generous.
const STRLEN_MAX_SCAN: u64 = 1 << 20; // 1 MiB

fn hle_malloc(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("malloc(size={:#x})", args.first().copied().unwrap_or(0));
    FAKE_HEAP_ADDR
}

fn hle_free(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!("free(ptr={:#x})", args.first().copied().unwrap_or(0));
    0
}

fn hle_calloc(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "calloc(nmemb={}, size={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    FAKE_HEAP_ADDR
}

fn hle_realloc(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "realloc(ptr={:#x}, size={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    FAKE_HEAP_ADDR
}

/// Real `memcpy` returns `dst` unchanged. Now actually copies: reads `n`
/// bytes from `ctx.mem` at `src` and writes them at `dst`, bounds-checked —
/// if either side is out of bounds, logs and returns `dst` without moving
/// any bytes (never a panic or OOB host access).
fn hle_memcpy(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    let n = args.get(2).copied().unwrap_or(0);
    debug!("memcpy(dst={dst:#x}, src={src:#x}, n={n:#x})");

    let Ok(len) = usize::try_from(n) else {
        warn!("memcpy: n={n:#x} does not fit a host usize");
        return dst;
    };
    let mut buf = vec![0u8; len];
    if !ctx.mem.read(src, &mut buf) {
        warn!("memcpy: src {src:#x} (len {n:#x}) out of bounds");
        return dst;
    }
    if !ctx.mem.write(dst, &buf) {
        warn!("memcpy: dst {dst:#x} (len {n:#x}) out of bounds");
    }
    dst
}

/// Real `memset` returns `dst` unchanged. Now actually fills `n` bytes at
/// `dst` with `c`, bounds-checked through `ctx.mem`.
fn hle_memset(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let value = args.get(1).copied().unwrap_or(0) as u8;
    let n = args.get(2).copied().unwrap_or(0);
    debug!("memset(s={dst:#x}, c={value}, n={n:#x})");

    let Ok(len) = usize::try_from(n) else {
        warn!("memset: n={n:#x} does not fit a host usize");
        return dst;
    };
    let buf = vec![value; len];
    if !ctx.mem.write(dst, &buf) {
        warn!("memset: dst {dst:#x} (len {n:#x}) out of bounds");
    }
    dst
}

fn hle_memmove(_ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    debug!(
        "memmove(dst={:#x}, src={:#x}, n={:#x}) [placeholder: no bytes actually moved]",
        dst,
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    dst
}

/// Walks `ctx.mem` from `s` counting bytes until a NUL terminator or
/// [`STRLEN_MAX_SCAN`] bytes have been scanned (guarding against a
/// wild/unterminated pointer spinning forever). Returns the length found so
/// far if the scan runs off the end of mapped memory or hits the cap.
fn hle_strlen(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    debug!("strlen(s={s:#x})");

    let mut len: u64 = 0;
    let mut byte = [0u8; 1];
    while len < STRLEN_MAX_SCAN {
        let Some(addr) = s.checked_add(len) else {
            warn!("strlen: s {s:#x} + len {len} overflowed");
            break;
        };
        if !ctx.mem.read(addr, &mut byte) {
            warn!("strlen: s {s:#x} out of bounds after {len} bytes");
            break;
        }
        if byte[0] == 0 {
            break;
        }
        len += 1;
    }
    len
}

/// Placeholder: real `strcmp` reads both guest strings; this stub always
/// reports "equal" (`0`).
fn hle_strcmp(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "strcmp(s1={:#x}, s2={:#x}) [placeholder: cannot read guest memory]",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    0
}

fn hle_strcpy(_ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    debug!(
        "strcpy(dst={:#x}, src={:#x}) [placeholder: no bytes actually copied]",
        dst,
        args.get(1).copied().unwrap_or(0)
    );
    dst
}

fn hle_strncpy(_ctx: &HleContext, args: &[u64]) -> u64 {
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
fn hle_snprintf(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "snprintf(buf={:#x}, size={:#x}, fmt={:#x}) [placeholder: no formatting performed]",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    0
}

fn hle_printf(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "printf(fmt={:#x}) [placeholder: no formatting performed]",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_puts(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "puts(s={:#x}) [placeholder: cannot read guest memory]",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_abort(_ctx: &HleContext, _args: &[u64]) -> u64 {
    // Real `abort` never returns (raises SIGABRT). The stub cannot terminate
    // the guest process from here, so it just logs and returns.
    debug!("abort() [stub: does not actually terminate the process]");
    0
}

fn hle_exit(_ctx: &HleContext, args: &[u64]) -> u64 {
    // Real `exit` never returns. Same limitation as `abort` above.
    debug!(
        "exit(code={}) [stub: does not actually terminate the process]",
        args.first().copied().unwrap_or(0)
    );
    0
}

fn hle_stack_chk_fail(_ctx: &HleContext, _args: &[u64]) -> u64 {
    // Real `__stack_chk_fail` aborts the process on stack-smash detection.
    // The stub just logs — it cannot terminate the guest process.
    debug!("__stack_chk_fail() [stub: does not actually terminate the process]");
    0
}

fn hle_memalign(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "memalign(alignment={:#x}, size={:#x})",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0)
    );
    FAKE_HEAP_ADDR
}

fn hle_posix_memalign(_ctx: &HleContext, args: &[u64]) -> u64 {
    // Real function writes the allocated pointer through `*memptr`; not
    // wired up here yet. Report success (0) only.
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
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn register_adds_expected_functions() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let ctx = test_ctx(&kernel, &mem);
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
            registry.call(&ctx, "libc", name, &[1, 2, 3]);
        }
    }

    #[test]
    fn memcpy_actually_moves_bytes_in_guest_memory() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let ctx = test_ctx(&kernel, &mem);

        let src: u64 = 0x10;
        let dst: u64 = 0x50;
        let payload = [0xDEu8, 0xAD, 0xBE, 0xEF];
        assert!(mem.write(src, &payload));

        let result = registry.call(&ctx, "libc", "memcpy", &[dst, src, payload.len() as u64]).unwrap();
        assert_eq!(result, dst);

        let mut copied = [0u8; 4];
        assert!(mem.read(dst, &mut copied));
        assert_eq!(copied, payload, "memcpy must actually move the bytes, not just return dst");
    }

    #[test]
    fn memcpy_out_of_bounds_src_does_not_panic_and_leaves_dst_alone() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x20);
        let ctx = test_ctx(&kernel, &mem);

        // src is entirely outside the 0x20-byte test memory.
        let result = registry.call(&ctx, "libc", "memcpy", &[0x0, 0xFFFF, 8]).unwrap();
        assert_eq!(result, 0x0);
    }

    #[test]
    fn memset_actually_fills_bytes_in_guest_memory() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let ctx = test_ctx(&kernel, &mem);

        let dst: u64 = 0x20;
        let result = registry.call(&ctx, "libc", "memset", &[dst, 0xAB, 6]).unwrap();
        assert_eq!(result, dst);

        let mut filled = [0u8; 6];
        assert!(mem.read(dst, &mut filled));
        assert_eq!(filled, [0xAB; 6]);
    }

    #[test]
    fn strlen_measures_a_real_nul_terminated_guest_string() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let ctx = test_ctx(&kernel, &mem);

        let s: u64 = 0x8;
        assert!(mem.write(s, b"hello\0garbage"));

        let result = registry.call(&ctx, "libc", "strlen", &[s]).unwrap();
        assert_eq!(result, 5);
    }

    #[test]
    fn strlen_on_unmapped_pointer_stops_without_panicking() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let ctx = test_ctx(&kernel, &mem);

        // Pointer entirely outside the 0x10-byte test memory: strlen should
        // report 0 (nothing readable), not panic or spin.
        let result = registry.call(&ctx, "libc", "strlen", &[0xFFFF]).unwrap();
        assert_eq!(result, 0);
    }
}
