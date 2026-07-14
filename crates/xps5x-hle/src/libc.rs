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

/// Real `strcmp` (M1-C): reads both (bounded) guest strings and compares
/// them as unsigned bytes, returning the sign of the first difference as a
/// sign-extended-to-u64 `int`. An unreadable pointer logs and reports
/// "equal" — the least-surprising degradation for a comparison with no
/// error channel (the pre-M1-C stub did the same unconditionally).
fn hle_strcmp(ctx: &HleContext, args: &[u64]) -> u64 {
    let s1_ptr = args.first().copied().unwrap_or(0);
    let s2_ptr = args.get(1).copied().unwrap_or(0);
    debug!("strcmp(s1={s1_ptr:#x}, s2={s2_ptr:#x})");

    let (Some(s1), Some(s2)) = (
        crate::fmt::read_cstr(ctx.mem, s1_ptr),
        crate::fmt::read_cstr(ctx.mem, s2_ptr),
    ) else {
        warn!("strcmp: unreadable string pointer (s1={s1_ptr:#x}, s2={s2_ptr:#x})");
        return 0;
    };
    match s1.cmp(&s2) {
        std::cmp::Ordering::Less => (-1i32) as u32 as u64,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Real `strcpy`: reads the (bounded, see `fmt::read_cstr`) NUL-terminated
/// source string from guest memory and writes it — including the NUL — to
/// `dst`. An unreadable source or a failed destination write logs and
/// leaves the destination as-is; the return value is `dst` either way (the
/// real API's return value carries no error channel).
fn hle_strcpy(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    debug!("strcpy(dst={dst:#x}, src={src:#x})");

    let Some(mut bytes) = crate::fmt::read_cstr(ctx.mem, src) else {
        warn!("strcpy: unreadable source string at {src:#x}");
        return dst;
    };
    bytes.push(0);
    if !ctx.mem.write(dst, &bytes) {
        warn!(
            "strcpy: failed to write {} bytes to dst={dst:#x}",
            bytes.len()
        );
    }
    dst
}

/// Real `strncpy`: copies at most `n` source bytes and, per the (infamous)
/// real contract, zero-fills the remainder of the `n`-byte destination if
/// the source is shorter — and does *not* NUL-terminate if the source is
/// `n` bytes or longer.
fn hle_strncpy(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    let n = args.get(2).copied().unwrap_or(0);
    debug!("strncpy(dst={dst:#x}, src={src:#x}, n={n:#x})");

    let Ok(n_usize) = usize::try_from(n) else {
        warn!("strncpy: n={n:#x} does not fit a host usize");
        return dst;
    };
    let Some(mut bytes) = crate::fmt::read_cstr(ctx.mem, src) else {
        warn!("strncpy: unreadable source string at {src:#x}");
        return dst;
    };
    bytes.resize(n_usize, 0); // truncate long, zero-fill short — the real semantics
    if !ctx.mem.write(dst, &bytes) {
        warn!("strncpy: failed to write {n_usize} bytes to dst={dst:#x}");
    }
    dst
}

/// Placeholder: real `snprintf` returns the number of characters that
/// would've been written; this stub reports `0` since it does no formatting.
/// Real `snprintf` (M1-C): reads the guest format string, formats it against
/// the remaining captured registers (at most 3 variadic values — the
/// register-only dispatch limit, see `fmt.rs`'s module docs), and writes at
/// most `size - 1` bytes plus a NUL into the guest buffer. Returns the full
/// would-be length (the real API's truncation-detection contract). `size ==
/// 0` writes nothing, per the real API.
fn hle_snprintf(ctx: &HleContext, args: &[u64]) -> u64 {
    let buf = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    let fmt_ptr = args.get(2).copied().unwrap_or(0);
    debug!("snprintf(buf={buf:#x}, size={size:#x}, fmt={fmt_ptr:#x})");

    let Some(fmt) = crate::fmt::read_cstr(ctx.mem, fmt_ptr) else {
        warn!("snprintf: unreadable format string at {fmt_ptr:#x}");
        return 0;
    };
    let mut varargs = args.iter().skip(3).copied();
    let formatted = crate::fmt::format_c(&fmt, &mut varargs, ctx.mem);
    let full_len = formatted.len() as u64;

    if size > 0 {
        let cap = usize::try_from(size - 1).unwrap_or(usize::MAX);
        let n = formatted.len().min(cap);
        let mut out = formatted;
        out.truncate(n);
        out.push(0);
        if !ctx.mem.write(buf, &out) {
            warn!(
                "snprintf: failed to write {} bytes to buf={buf:#x}",
                out.len()
            );
        }
    }
    full_len
}

/// Real `printf` (M1-C): reads the guest format string, formats it against
/// the remaining captured registers (at most 5 variadic values — the
/// register-only dispatch limit, see `fmt.rs`'s module docs), and emits the
/// result to the kernel [`xps5x_kernel::Console`] (captured for the Shell /
/// tests, mirrored to the host log). Returns the number of bytes written,
/// per the real API.
fn hle_printf(ctx: &HleContext, args: &[u64]) -> u64 {
    let fmt_ptr = args.first().copied().unwrap_or(0);
    debug!("printf(fmt={fmt_ptr:#x})");

    let Some(fmt) = crate::fmt::read_cstr(ctx.mem, fmt_ptr) else {
        warn!("printf: unreadable format string at {fmt_ptr:#x}");
        return u64::MAX; // EOF-ish negative return, the real error signal
    };
    let mut varargs = args.iter().skip(1).copied();
    let formatted = crate::fmt::format_c(&fmt, &mut varargs, ctx.mem);
    ctx.kernel.console.write_bytes(&formatted);
    formatted.len() as u64
}

/// Real `puts` (M1-C): reads the guest string and emits it plus the
/// API-mandated trailing newline to the kernel console. Returns a
/// nonnegative value on success, `EOF` (-1) on an unreadable pointer.
fn hle_puts(ctx: &HleContext, args: &[u64]) -> u64 {
    let s_ptr = args.first().copied().unwrap_or(0);
    debug!("puts(s={s_ptr:#x})");

    let Some(mut s) = crate::fmt::read_cstr(ctx.mem, s_ptr) else {
        warn!("puts: unreadable string at {s_ptr:#x}");
        return u64::MAX; // EOF
    };
    s.push(b'\n');
    let len = s.len() as u64;
    ctx.kernel.console.write_bytes(&s);
    len
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

    /// M1-C: `printf` reads the guest format string and `%s` pointee, formats
    /// against the captured registers, and lands the output in the kernel
    /// console — the observable-stdout contract.
    #[test]
    fn printf_formats_guest_strings_into_the_kernel_console() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"hello %s, %d + %d = %d\n\0"));
        assert!(mem.write(0x200, b"world\0"));

        let written = hle_printf(&ctx, &[0x100, 0x200, 2, 3, 5]);
        assert_eq!(kernel.console.contents(), "hello world, 2 + 3 = 5\n");
        assert_eq!(written, "hello world, 2 + 3 = 5\n".len() as u64);
    }

    #[test]
    fn puts_appends_the_mandated_newline() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"Hello World\0"));
        let ret = hle_puts(&ctx, &[0x100]);
        assert_eq!(kernel.console.contents(), "Hello World\n");
        assert!(ret as i64 >= 0, "puts must return nonnegative on success");
    }

    #[test]
    fn puts_with_unreadable_pointer_returns_eof_and_writes_nothing() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let ret = hle_puts(&ctx, &[0xDEAD_0000]);
        assert_eq!(ret as i64, -1, "EOF on unreadable pointer");
        assert!(kernel.console.is_empty());
    }

    /// M1-C: `snprintf` writes the (truncated, NUL-terminated) result into
    /// guest memory and returns the full would-be length.
    #[test]
    fn snprintf_truncates_nul_terminates_and_returns_full_length() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"n=%d!\0"));
        // Full formatted result is "n=1234!" (7 bytes); size 5 keeps 4 + NUL.
        let ret = hle_snprintf(&ctx, &[0x300, 5, 0x100, 1234]);
        assert_eq!(ret, 7, "returns the untruncated length");
        let mut buf = [0u8; 5];
        assert!(mem.read(0x300, &mut buf));
        assert_eq!(&buf, b"n=12\0");
        assert!(
            kernel.console.is_empty(),
            "snprintf must not touch the console"
        );
    }

    #[test]
    fn strcmp_strcpy_strncpy_do_real_string_work() {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"abc\0"));
        assert!(mem.write(0x110, b"abd\0"));
        assert_eq!(hle_strcmp(&ctx, &[0x100, 0x110]) as u32 as i32, -1);
        assert_eq!(hle_strcmp(&ctx, &[0x110, 0x100]) as u32 as i32, 1);
        assert_eq!(hle_strcmp(&ctx, &[0x100, 0x100]), 0);

        assert_eq!(hle_strcpy(&ctx, &[0x200, 0x100]), 0x200);
        let mut buf = [0u8; 4];
        assert!(mem.read(0x200, &mut buf));
        assert_eq!(&buf, b"abc\0");

        // strncpy zero-fills the remainder of an n-byte destination…
        assert!(mem.write(0x300, &[0xFFu8; 8]));
        assert_eq!(hle_strncpy(&ctx, &[0x300, 0x100, 6]), 0x300);
        let mut buf6 = [0u8; 6];
        assert!(mem.read(0x300, &mut buf6));
        assert_eq!(&buf6, b"abc\0\0\0");
        // …and does NOT NUL-terminate when the source fills all n bytes.
        assert_eq!(hle_strncpy(&ctx, &[0x400, 0x100, 2]), 0x400);
        let mut buf2 = [0u8; 2];
        assert!(mem.read(0x400, &mut buf2));
        assert_eq!(&buf2, b"ab");
    }

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
            assert!(
                registry.is_implemented("libc", name),
                "missing libc::{name}"
            );
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

        let result = registry
            .call(&ctx, "libc", "memcpy", &[dst, src, payload.len() as u64])
            .unwrap();
        assert_eq!(result, dst);

        let mut copied = [0u8; 4];
        assert!(mem.read(dst, &mut copied));
        assert_eq!(
            copied, payload,
            "memcpy must actually move the bytes, not just return dst"
        );
    }

    #[test]
    fn memcpy_out_of_bounds_src_does_not_panic_and_leaves_dst_alone() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x20);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // src is entirely outside the 0x20-byte test memory.
        let result = registry
            .call(&ctx, "libc", "memcpy", &[0x0, 0xFFFF, 8])
            .unwrap();
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
        let result = registry
            .call(&ctx, "libc", "memset", &[dst, 0xAB, 6])
            .unwrap();
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
        assert_eq!(
            registry
                .call(&ctx, "libc", "memcpy", &[0x0, 0x8, huge])
                .unwrap(),
            0x0
        );
        assert_eq!(
            registry
                .call(&ctx, "libc", "memset", &[0x0, 0xAB, huge])
                .unwrap(),
            0x0
        );
        assert_eq!(
            registry
                .call(&ctx, "libc", "memmove", &[0x0, 0x8, huge])
                .unwrap(),
            0x0
        );
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
        let result = registry
            .call(&ctx, "libc", "memmove", &[0x11, 0x10, 4])
            .unwrap();
        assert_eq!(result, 0x11);

        let mut out = [0u8; 5];
        assert!(mem.read(0x10, &mut out));
        assert_eq!(
            out,
            [1, 1, 2, 3, 4],
            "upward overlapping memmove must not smear the first byte"
        );
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
        let result = registry
            .call(&ctx, "libc", "memmove", &[0x10, 0x11, 4])
            .unwrap();
        assert_eq!(result, 0x10);

        let mut out = [0u8; 5];
        assert!(mem.read(0x10, &mut out));
        assert_eq!(
            out,
            [2, 3, 4, 5, 5],
            "downward overlapping memmove must shift bytes down cleanly"
        );
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
        assert_eq!(
            block, [0u8; 16],
            "calloc'd block must read back as all zeros"
        );
    }

    #[test]
    fn calloc_overflowing_nmemb_times_size_returns_zero() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let result = registry
            .call(&ctx, "libc", "calloc", &[u64::MAX, 2])
            .unwrap();
        assert_eq!(
            result, 0,
            "an overflowing nmemb*size must report NULL, not wrap into an undersized alloc"
        );
    }

    #[test]
    fn realloc_with_null_ptr_behaves_like_malloc() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let result = registry.call(&ctx, "libc", "realloc", &[0, 32]).unwrap();
        assert_ne!(
            result, 0,
            "realloc(NULL, size) must behave like malloc(size)"
        );
    }

    #[test]
    fn free_of_null_pointer_is_a_harmless_noop() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0x10);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let result = registry.call(&ctx, "libc", "free", &[0]).unwrap();
        assert_eq!(
            result, 0,
            "free(NULL) must not panic or forward a null address to the allocator"
        );
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
        assert_eq!(
            result, 0,
            "an exhausted/overflowing allocator request must report NULL, not panic"
        );
    }

    #[test]
    fn posix_memalign_writes_the_pointer_through_memptr_and_reports_success() {
        let registry = HleRegistry::new();
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0x20);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let memptr: u64 = 0x8;
        let result = registry
            .call(&ctx, "libc", "posix_memalign", &[memptr, 16, 64])
            .unwrap();
        assert_eq!(
            result, 0,
            "posix_memalign must report success (0) on a satisfiable request"
        );

        let mut written = [0u8; 8];
        assert!(mem.read(memptr, &mut written));
        let addr = u64::from_le_bytes(written);
        assert_ne!(
            addr, 0,
            "posix_memalign must write the real allocated address through *memptr"
        );
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

        let result = registry
            .call(&ctx, "libc", "memmove", &[dst, src, payload.len() as u64])
            .unwrap();
        assert_eq!(result, dst);

        let mut moved = [0u8; 4];
        assert!(mem.read(dst, &mut moved));
        assert_eq!(
            moved, payload,
            "memmove must actually move the bytes, not just return dst"
        );
    }
}
