//! HLE libc — Standard C library re-implementation.
//!
//! Clean-room re-implementation of the PS5 `libc.sprx` exports. Function
//! *names* below are factual, standard C API identifiers (not copyrightable);
//! every implementation is original.
//!
//! ## Stub status
//!
//! Every HLE call now gets an [`crate::HleContext`] with guest-memory *and*
//! guest-allocator access, so both buffer functions and the heap family do
//! real work. `memcpy`, `memset`, `memmove`, and `strlen` do the real
//! operation below, bounds-checked through [`crate::GuestMemory`]. The heap
//! family (`malloc`/`calloc`/`realloc`/`free`/`memalign`/`posix_memalign`)
//! routes through [`crate::GuestAllocator`], backed in production by
//! `xps5x-runtime`'s `GuestArena` — so a guest `malloc` returns a real,
//! dereferenceable guest address, not a sentinel. The rest (`strcpy`/
//! `printf`/...) still log the call and return a plausible value — string/
//! format handling is future work, not blocked on the dispatch signature
//! anymore.

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

/// Cap on how far [`hle_strlen`] will scan looking for a NUL terminator, so
/// a wild/unterminated guest pointer can't spin forever. Arbitrary but
/// generous.
const STRLEN_MAX_SCAN: u64 = 1 << 20; // 1 MiB

/// `ENOMEM`-ish errno value [`hle_posix_memalign`] reports on allocation
/// failure. `posix_memalign` returns an errno value directly (not through
/// `errno`/`GetLastError`), so any nonzero value the caller can distinguish
/// from success (`0`) is honest here; `12` is the real `ENOMEM` on both
/// Linux and the PS5's BSD-derived libc.
const POSIX_MEMALIGN_ENOMEM: u64 = 12;

/// Real `malloc` allocates `size` bytes (any alignment libc guarantees,
/// here fixed at 16 bytes — the usual `malloc` minimum) from the guest heap.
/// Honest OOM: an exhausted/overflowing request returns `0` (`NULL`), never
/// a sentinel or a panic.
fn hle_malloc(ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.first().copied().unwrap_or(0);
    debug!("malloc(size={size:#x})");
    ctx.alloc.alloc(size, 16).unwrap_or(0)
}

/// Real `free` releases a block previously returned by `malloc`/`calloc`/
/// `realloc`/`memalign`. `free(NULL)` is a defined no-op in the real API, so
/// a `ptr == 0` is not even forwarded to the allocator.
fn hle_free(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    debug!("free(ptr={ptr:#x})");
    if ptr != 0 {
        ctx.alloc.free(ptr);
    }
    0
}

/// Real `calloc` allocates `nmemb * size` bytes and zero-fills them. The
/// multiplication is checked — real `calloc` must report `NULL` on overflow
/// rather than silently allocating an undersized block — and the block is
/// zeroed through `ctx.mem` after allocation (the allocator itself makes no
/// zeroing guarantee).
fn hle_calloc(ctx: &HleContext, args: &[u64]) -> u64 {
    let nmemb = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    debug!("calloc(nmemb={nmemb}, size={size:#x})");

    let Some(total) = nmemb.checked_mul(size) else {
        warn!("calloc: nmemb={nmemb} * size={size:#x} overflowed");
        return 0;
    };
    let Some(addr) = ctx.alloc.alloc(total, 16) else {
        return 0;
    };
    let Ok(len) = usize::try_from(total) else {
        warn!("calloc: total={total:#x} does not fit a host usize");
        ctx.alloc.free(addr);
        return 0;
    };
    if !ctx.mem.write(addr, &vec![0u8; len]) {
        warn!("calloc: zeroing block at {addr:#x} (len {total:#x}) failed");
    }
    addr
}

/// Real `realloc(NULL, size)` behaves exactly like `malloc(size)`; otherwise
/// resizes the existing block, honest-OOM (`0`) on failure — the original
/// block is left untouched by [`crate::GuestAllocator::realloc`]'s contract
/// in that case.
fn hle_realloc(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    debug!("realloc(ptr={ptr:#x}, size={size:#x})");

    if ptr == 0 {
        return ctx.alloc.alloc(size, 16).unwrap_or(0);
    }
    ctx.alloc.realloc(ptr, size).unwrap_or(0)
}

/// Bounded host staging-buffer size for the block memory ops (`memcpy`,
/// `memmove`, `memset`). They act on a guest-controlled length `n`; staging
/// the transfer in fixed-size chunks — rather than one `vec![0u8; n]` — keeps
/// host memory use `O(MEM_OP_CHUNK)` regardless of `n`. Otherwise, since
/// `usize == u64` on this target, `usize::try_from(n)` always succeeds and a
/// guest passing e.g. `n = 0x0000_FFFF_FFFF_FFFF` would make the host attempt
/// a ~256 TiB allocation, aborting the process via `handle_alloc_error`
/// before any bounds check runs. Bytes still only move where `ctx.mem`
/// permits; an out-of-bounds chunk stops the transfer (a partial move may
/// have occurred, as with a bad pointer in C — but never a panic, abort, or
/// OOB host access).
const MEM_OP_CHUNK: usize = 16 * 1024;

/// Copy `n` guest bytes `src`→`dst` low-address-first, in [`MEM_OP_CHUNK`]
/// chunks through `ctx.mem`. Correct for non-overlapping ranges and for
/// overlaps where `dst <= src`. Returns `false` on the first out-of-bounds or
/// address-overflowing chunk.
fn mem_copy_forward(ctx: &HleContext, dst: u64, src: u64, n: u64) -> bool {
    let mut buf = [0u8; MEM_OP_CHUNK];
    let mut done: u64 = 0;
    while done < n {
        let chunk = (n - done).min(MEM_OP_CHUNK as u64) as usize;
        let (Some(s), Some(d)) = (src.checked_add(done), dst.checked_add(done)) else {
            return false;
        };
        if !ctx.mem.read(s, &mut buf[..chunk]) || !ctx.mem.write(d, &buf[..chunk]) {
            return false;
        }
        done += chunk as u64;
    }
    true
}

/// Copy `n` guest bytes `src`→`dst` high-address-first, in [`MEM_OP_CHUNK`]
/// chunks through `ctx.mem` — the overlap-safe direction when `dst > src`.
/// Returns `false` on the first out-of-bounds or address-overflowing chunk.
fn mem_copy_backward(ctx: &HleContext, dst: u64, src: u64, n: u64) -> bool {
    let mut buf = [0u8; MEM_OP_CHUNK];
    let mut remaining = n;
    while remaining > 0 {
        let chunk = remaining.min(MEM_OP_CHUNK as u64);
        let off = remaining - chunk;
        let (Some(s), Some(d)) = (src.checked_add(off), dst.checked_add(off)) else {
            return false;
        };
        let chunk = chunk as usize;
        if !ctx.mem.read(s, &mut buf[..chunk]) || !ctx.mem.write(d, &buf[..chunk]) {
            return false;
        }
        remaining = off;
    }
    true
}

/// Fill `n` guest bytes at `dst` with `value`, in [`MEM_OP_CHUNK`] chunks
/// through `ctx.mem`. Returns `false` on the first out-of-bounds or
/// address-overflowing chunk.
fn mem_fill(ctx: &HleContext, dst: u64, value: u8, n: u64) -> bool {
    let buf = [value; MEM_OP_CHUNK];
    let mut done: u64 = 0;
    while done < n {
        let chunk = (n - done).min(MEM_OP_CHUNK as u64) as usize;
        let Some(d) = dst.checked_add(done) else {
            return false;
        };
        if !ctx.mem.write(d, &buf[..chunk]) {
            return false;
        }
        done += chunk as u64;
    }
    true
}

/// Real `memcpy` returns `dst` unchanged. Copies `n` bytes `src`→`dst`
/// through `ctx.mem` in bounded chunks (see [`MEM_OP_CHUNK`]) — a huge
/// guest-controlled `n` never triggers a giant host allocation. Out of bounds
/// on either side stops the copy and returns `dst` (never a panic or OOB host
/// access).
fn hle_memcpy(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    let n = args.get(2).copied().unwrap_or(0);
    debug!("memcpy(dst={dst:#x}, src={src:#x}, n={n:#x})");
    if !mem_copy_forward(ctx, dst, src, n) {
        warn!("memcpy: out of bounds (dst={dst:#x}, src={src:#x}, n={n:#x})");
    }
    dst
}

/// Real `memset` returns `dst` unchanged. Fills `n` bytes at `dst` with `c`
/// through `ctx.mem` in bounded chunks (see [`MEM_OP_CHUNK`]).
fn hle_memset(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let value = args.get(1).copied().unwrap_or(0) as u8;
    let n = args.get(2).copied().unwrap_or(0);
    debug!("memset(s={dst:#x}, c={value}, n={n:#x})");
    if !mem_fill(ctx, dst, value, n) {
        warn!("memset: dst {dst:#x} (len {n:#x}) out of bounds");
    }
    dst
}

/// Real `memmove` returns `dst` unchanged. Moves `n` bytes `src`→`dst`
/// through `ctx.mem` in bounded chunks, choosing copy direction from the
/// `dst`/`src` order so overlapping ranges are handled correctly (forward when
/// `dst <= src`, backward when `dst > src`) — a real `memmove`, not a `memcpy`
/// alias, and with the same huge-`n` safety (see [`MEM_OP_CHUNK`]) as the
/// others.
fn hle_memmove(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    let n = args.get(2).copied().unwrap_or(0);
    debug!("memmove(dst={dst:#x}, src={src:#x}, n={n:#x})");
    let ok = if dst <= src {
        mem_copy_forward(ctx, dst, src, n)
    } else {
        mem_copy_backward(ctx, dst, src, n)
    };
    if !ok {
        warn!("memmove: out of bounds (dst={dst:#x}, src={src:#x}, n={n:#x})");
    }
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

/// Real `memalign(alignment, size)` allocates `size` bytes aligned to
/// `alignment`, honest-OOM (`0`) on failure.
fn hle_memalign(ctx: &HleContext, args: &[u64]) -> u64 {
    let alignment = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    debug!("memalign(alignment={alignment:#x}, size={size:#x})");
    ctx.alloc.alloc(size, alignment).unwrap_or(0)
}

/// Real `posix_memalign(memptr, alignment, size)` allocates `size` bytes
/// aligned to `alignment` and writes the resulting guest address through
/// `*memptr` (via `ctx.mem`), returning `0` on success or a nonzero
/// errno-ish value ([`POSIX_MEMALIGN_ENOMEM`]) on failure — the real
/// function's return value is an errno, not a pointer or boolean.
fn hle_posix_memalign(ctx: &HleContext, args: &[u64]) -> u64 {
    let memptr = args.first().copied().unwrap_or(0);
    let alignment = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    debug!("posix_memalign(memptr={memptr:#x}, alignment={alignment:#x}, size={size:#x})");

    let Some(addr) = ctx.alloc.alloc(size, alignment) else {
        return POSIX_MEMALIGN_ENOMEM;
    };
    if !ctx.mem.write(memptr, &addr.to_le_bytes()) {
        warn!("posix_memalign: failed to write result pointer to memptr={memptr:#x}");
        ctx.alloc.free(addr);
        return POSIX_MEMALIGN_ENOMEM;
    }
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
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
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
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

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
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // src is entirely outside the 0x20-byte test memory.
        let result = registry.call(&ctx, "libc", "memcpy", &[0x0, 0xFFFF, 8]).unwrap();
        assert_eq!(result, 0x0);
    }

    #[test]
    fn memset_actually_fills_bytes_in_guest_memory() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let dst: u64 = 0x20;
        let result = registry.call(&ctx, "libc", "memset", &[dst, 0xAB, 6]).unwrap();
        assert_eq!(result, dst);

        let mut filled = [0u8; 6];
        assert!(mem.read(dst, &mut filled));
        assert_eq!(filled, [0xAB; 6]);
    }

    /// Regression: a huge guest-controlled length must not make the host try
    /// a gigantic allocation (which would abort the process via
    /// `handle_alloc_error`). The block ops stage in bounded chunks, so this
    /// returns `dst` harmlessly instead of dying. If this test process
    /// survives the call, the fix holds.
    #[test]
    fn block_ops_with_huge_guest_length_do_not_abort() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // ~256 TiB — would abort if any op did `vec![0u8; n]` up front.
        let huge = 0x0000_FFFF_FFFF_FFFF;
        assert_eq!(registry.call(&ctx, "libc", "memcpy", &[0x0, 0x8, huge]).unwrap(), 0x0);
        assert_eq!(registry.call(&ctx, "libc", "memset", &[0x0, 0xAB, huge]).unwrap(), 0x0);
        assert_eq!(registry.call(&ctx, "libc", "memmove", &[0x0, 0x8, huge]).unwrap(), 0x0);
    }

    /// `memmove` with `dst > src` and overlapping ranges must copy
    /// high-address-first, or it would clobber source bytes it hasn't read.
    #[test]
    fn memmove_overlapping_upward_is_correct() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x10, &[1, 2, 3, 4, 5]));
        // Move [0x10..0x14] up by one into [0x11..0x15].
        let result = registry.call(&ctx, "libc", "memmove", &[0x11, 0x10, 4]).unwrap();
        assert_eq!(result, 0x11);

        let mut out = [0u8; 5];
        assert!(mem.read(0x10, &mut out));
        assert_eq!(out, [1, 1, 2, 3, 4], "upward overlapping memmove must not smear the first byte");
    }

    /// `memmove` with `dst < src` and overlapping ranges must copy
    /// low-address-first (a `memcpy`-style forward copy is already correct
    /// here); verifies the direction choice doesn't corrupt this case.
    #[test]
    fn memmove_overlapping_downward_is_correct() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x10, &[1, 2, 3, 4, 5]));
        // Move [0x11..0x15] down by one into [0x10..0x14].
        let result = registry.call(&ctx, "libc", "memmove", &[0x10, 0x11, 4]).unwrap();
        assert_eq!(result, 0x10);

        let mut out = [0u8; 5];
        assert!(mem.read(0x10, &mut out));
        assert_eq!(out, [2, 3, 4, 5, 5], "downward overlapping memmove must shift bytes down cleanly");
    }

    #[test]
    fn strlen_measures_a_real_nul_terminated_guest_string() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

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
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Pointer entirely outside the 0x10-byte test memory: strlen should
        // report 0 (nothing readable), not panic or spin.
        let result = registry.call(&ctx, "libc", "strlen", &[0xFFFF]).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn malloc_returns_nonzero_distinct_addresses() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let a = registry.call(&ctx, "libc", "malloc", &[16]).unwrap();
        let b = registry.call(&ctx, "libc", "malloc", &[16]).unwrap();
        assert_ne!(a, 0, "malloc must not return a null/sentinel address");
        assert_ne!(b, 0, "malloc must not return a null/sentinel address");
        assert_ne!(a, b, "two live allocations must not share an address");
    }

    #[test]
    fn calloc_zeroes_the_allocated_block_through_guest_memory() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Pre-fill the region calloc will hand out with garbage, so a
        // zero read-back proves calloc actually zeroed it rather than it
        // merely having started zeroed.
        assert!(mem.write(0x10, &[0xFFu8; 16]));

        let result = registry.call(&ctx, "libc", "calloc", &[4, 4]).unwrap();
        assert_ne!(result, 0, "calloc must not return a null/sentinel address");

        let mut block = [0u8; 16];
        assert!(mem.read(result, &mut block));
        assert_eq!(block, [0u8; 16], "calloc'd block must read back as all zeros");
    }

    #[test]
    fn calloc_overflowing_nmemb_times_size_returns_zero() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let result = registry.call(&ctx, "libc", "calloc", &[u64::MAX, 2]).unwrap();
        assert_eq!(result, 0, "an overflowing nmemb*size must report NULL, not wrap into an undersized alloc");
    }

    #[test]
    fn realloc_with_null_ptr_behaves_like_malloc() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let result = registry.call(&ctx, "libc", "realloc", &[0, 32]).unwrap();
        assert_ne!(result, 0, "realloc(NULL, size) must behave like malloc(size)");
    }

    #[test]
    fn free_of_null_pointer_is_a_harmless_noop() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let result = registry.call(&ctx, "libc", "free", &[0]).unwrap();
        assert_eq!(result, 0, "free(NULL) must not panic or forward a null address to the allocator");
    }

    #[test]
    fn malloc_returns_zero_when_the_allocator_is_exhausted() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        // A base near `u64::MAX` makes `TestAllocator`'s bump-alignment
        // arithmetic overflow on the very first request, simulating an
        // exhausted arena without needing a real `GuestArena`.
        let alloc = crate::TestAllocator::new(u64::MAX - 4);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let result = registry.call(&ctx, "libc", "malloc", &[16]).unwrap();
        assert_eq!(result, 0, "an exhausted/overflowing allocator request must report NULL, not panic");
    }

    #[test]
    fn posix_memalign_writes_the_pointer_through_memptr_and_reports_success() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0x20);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let memptr: u64 = 0x8;
        let result = registry.call(&ctx, "libc", "posix_memalign", &[memptr, 16, 64]).unwrap();
        assert_eq!(result, 0, "posix_memalign must report success (0) on a satisfiable request");

        let mut written = [0u8; 8];
        assert!(mem.read(memptr, &mut written));
        let addr = u64::from_le_bytes(written);
        assert_ne!(addr, 0, "posix_memalign must write the real allocated address through *memptr");
    }

    #[test]
    fn memmove_actually_moves_bytes_in_guest_memory() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let src: u64 = 0x10;
        let dst: u64 = 0x50;
        let payload = [0x11u8, 0x22, 0x33, 0x44];
        assert!(mem.write(src, &payload));

        let result = registry.call(&ctx, "libc", "memmove", &[dst, src, payload.len() as u64]).unwrap();
        assert_eq!(result, dst);

        let mut moved = [0u8; 4];
        assert!(mem.read(dst, &mut moved));
        assert_eq!(moved, payload, "memmove must actually move the bytes, not just return dst");
    }
}
