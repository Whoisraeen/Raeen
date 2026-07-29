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
//! `raeen-runtime`'s `GuestArena` — so a guest `malloc` returns a real,
//! dereferenceable guest address, not a sentinel. The string, `printf`-
//! family, and strtol functions are real too, as is the NID-fill batch
//! documented at `register` (Dinkumware CRT plumbing, stdio FILE streams,
//! the time/scanf/localeconv families, and the float-returning libm family
//! — those return their bits through the registry's float-return channel,
//! which the runtime delivers to guest XMM0 on both dispatch paths).

use crate::{HleContext, HleFunction, HleRegistry};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Register libc HLE functions.
pub fn register(registry: &HleRegistry) {
    register_abi(registry, "malloc", hle_malloc);
    register_abi(registry, "free", hle_free);
    register_abi(registry, "calloc", hle_calloc);
    register_abi(registry, "realloc", hle_realloc);
    register_abi(registry, "memcpy", hle_memcpy);
    register_abi(registry, "memset", hle_memset);
    register_abi(registry, "memmove", hle_memmove);
    register_abi(registry, "strlen", hle_strlen);
    register_abi(registry, "strcmp", hle_strcmp);
    register_abi(registry, "strcpy", hle_strcpy);
    register_abi(registry, "strncpy", hle_strncpy);
    // M1 hardening batch (real guest-memory behavior; ported with reference
    // to SharpEmu's KernelMemoryCompatExports + Kyty libc): the string/buffer
    // functions crt0 and ordinary homebrew hit constantly.
    register_abi(registry, "memcmp", hle_memcmp);
    register_abi(registry, "memchr", hle_memchr);
    register_abi(registry, "strncmp", hle_strncmp);
    register_abi(registry, "strnlen", hle_strnlen);
    register_abi(registry, "strchr", hle_strchr);
    register_abi(registry, "strrchr", hle_strrchr);
    register_abi(registry, "strcspn", hle_strcspn);
    register_abi(registry, "wcslen", hle_wcslen);
    register_abi(registry, "wcscpy", hle_wcscpy);
    register_abi(registry, "sincosf", hle_sincosf);
    register_abi(registry, "vsnprintf", hle_vsnprintf);
    register_abi(registry, "strtok", hle_strtok);
    register_abi(registry, "strcat", hle_strcat);
    register_abi(registry, "strncat", hle_strncat);
    register_abi(registry, "strstr", hle_strstr);
    // String → integer parsing (real behavior): arg/config parsing.
    register_abi(registry, "atoi", hle_atoi);
    register_abi(registry, "atol", hle_atol);
    register_abi(registry, "strtol", hle_strtol);
    register_abi(registry, "strtoul", hle_strtoul);
    // crt0 / C++ static-init registration: record-and-succeed. Real homebrew
    // registers atexit/global-dtor callbacks during startup; failing these
    // aborts init before `main`.
    register_abi(registry, "atexit", hle_atexit);
    register_abi(registry, "__cxa_atexit", hle_cxa_atexit);
    register_abi(registry, "snprintf", hle_snprintf);
    register_abi(registry, "printf", hle_printf);
    register_abi(registry, "puts", hle_puts);
    register_abi(registry, "abort", hle_abort);
    register_abi(registry, "exit", hle_exit);
    register_abi(registry, "__stack_chk_fail", hle_stack_chk_fail);
    // Retail modules also import the stack-protector abort naming libkernel
    // (same name-hash NID) — the measured Minecraft eboot does.
    registry.register("libkernel", "__stack_chk_fail", hle_stack_chk_fail);
    register_abi(registry, "memalign", hle_memalign);
    register_abi(registry, "posix_memalign", hle_posix_memalign);
    register_abi(registry, "_init_env", hle_init_env);
    // C++ function-local static guards (Itanium ABI). Measured: A Plague Tale
    // Requiem imports `__cxa_guard_acquire` from `libc`; release/abort are its
    // mandatory partners and are registered with it.
    register_abi(registry, "__cxa_guard_acquire", hle_cxa_guard_acquire);
    register_abi(registry, "__cxa_guard_release", hle_cxa_guard_release);
    register_abi(registry, "__cxa_guard_abort", hle_cxa_guard_abort);

    // ------------------------------------------------------------------
    // NID-fill batch: the 140 libc imports measured missing from retail
    // titles (artifacts/compat/nid-coverage.json; list in
    // scratch/nid-fill/missing-libc.txt). Implemented below with real,
    // host-backed behavior — grouped here by family. The skipped remainder
    // is listed with reasons in the big comment above `kernel_key`: the C++
    // unwinder family, data objects (vtables/typeinfo/locale ids), and the
    // Dinkumware locale/iostream internals whose object layouts this build
    // cannot verify. (qsort left this list 2026-07-27 when
    // GuestCallScheduler::call_guest gained synchronous guest callbacks.)
    // ------------------------------------------------------------------
    // Sorting with a REAL synchronous guest comparator (checklist item 7).
    register_abi(registry, "qsort", hle_qsort);
    // Dinkumware CRT internals (Sony's libc is Dinkumware-derived).
    register_abi(registry, "_Assert", hle_dinkum_assert);
    register_abi(registry, "_Atomic_fetch_add_4", hle_atomic_fetch_add_4);
    register_abi(registry, "_Atomic_fetch_sub_4", hle_atomic_fetch_sub_4);
    register_abi(registry, "_Getpctype", hle_getpctype);
    register_abi(registry, "_Getptolower", hle_getptolower);
    register_abi(registry, "_Getptoupper", hle_getptoupper);
    register_abi(registry, "_Locksyslock", hle_locksyslock);
    register_abi(registry, "_Unlocksyslock", hle_unlocksyslock);
    register_abi(registry, "_Mtx_init", hle_mtx_init);
    register_abi(registry, "_Mtx_lock", hle_mtx_lock);
    register_abi(registry, "_Mtx_unlock", hle_mtx_unlock);
    register_abi(registry, "_Mtx_destroy", hle_mtx_destroy);
    register_abi(registry, "_Thrd_id", hle_thrd_id);
    register_abi(registry, "_Thrd_join", hle_thrd_join);
    register_abi(registry, "_Thrd_sleep", hle_thrd_sleep);
    register_abi(registry, "_Xtime_get_ticks", hle_xtime_get_ticks);
    // stdio FILE streams (unbuffered, routed to the kernel console).
    register_abi(registry, "_Stdout", hle_stdout);
    register_abi(registry, "_Stderr", hle_stderr);
    register_abi(registry, "fflush", hle_fflush);
    register_abi(registry, "fputc", hle_fputc);
    register_abi(registry, "fputs", hle_fputs);
    register_abi(registry, "fwrite", hle_fwrite);
    register_abi(registry, "vfprintf", hle_vfprintf);
    // Formatted output/input.
    register_abi(registry, "sprintf", hle_sprintf);
    register_abi(registry, "sprintf_s", hle_sprintf_s);
    register_abi(registry, "vswprintf", hle_vswprintf);
    register_abi(registry, "sscanf", hle_sscanf);
    // String/scan extras.
    register_abi(registry, "strpbrk", hle_strpbrk);
    register_abi(registry, "wcscat", hle_wcscat);
    register_abi(registry, "mbsrtowcs", hle_mbsrtowcs);
    // Integer parsing: the 64-bit strto family (same ABI as strtol).
    register_abi(registry, "strtoimax", hle_strtoimax);
    register_abi(registry, "strtoull", hle_strtoull);
    register_abi(registry, "strtoumax", hle_strtoull);
    register_abi(registry, "_Stoull", hle_strtoull);
    // Random, heap query.
    register_abi(registry, "rand", hle_rand);
    register_abi(registry, "srand", hle_srand);
    register_abi(registry, "malloc_usable_size", hle_malloc_usable_size);
    // Time and locale.
    register_abi(registry, "time", hle_time);
    register_abi(registry, "gmtime", hle_gmtime);
    register_abi(registry, "gmtime_s", hle_gmtime_s);
    register_abi(registry, "localtime", hle_localtime);
    register_abi(registry, "mktime", hle_mktime);
    register_abi(registry, "strftime", hle_strftime);
    register_abi(registry, "localeconv", hle_localeconv);
    // Math: void with out-pointers (sincos, like the existing sincosf) and
    // int-returning classification (__isnan/__isfinite) ride the ordinary
    // channel...
    register_abi(registry, "sincos", hle_sincos);
    register_abi(registry, "__isfinite", hle_isfinite);
    register_abi(registry, "__isfinitef", hle_isfinitef);
    register_abi(registry, "__isnan", hle_isnan);
    register_abi(registry, "__isnanf", hle_isnanf);
    // ...while the float/double-RETURNING libm family uses the registry's
    // float-return marker: the handler hands back the result's bit pattern
    // and the runtime delivers it to guest XMM0 on both dispatch paths
    // (VEH writeback + the direct gateway's float bridge).
    register_abi_float(registry, "acosf", hle_acosf);
    register_abi_float(registry, "asin", hle_asin);
    register_abi_float(registry, "asinf", hle_asinf);
    register_abi_float(registry, "atan", hle_atan);
    register_abi_float(registry, "atan2", hle_atan2);
    register_abi_float(registry, "atan2f", hle_atan2f);
    register_abi_float(registry, "atanf", hle_atanf);
    register_abi_float(registry, "atof", hle_atof);
    register_abi_float(registry, "cos", hle_cos);
    register_abi_float(registry, "cosf", hle_cosf);
    register_abi_float(registry, "difftime", hle_difftime);
    register_abi_float(registry, "exp", hle_exp);
    register_abi_float(registry, "exp2", hle_exp2);
    register_abi_float(registry, "exp2f", hle_exp2f);
    register_abi_float(registry, "expf", hle_expf);
    register_abi_float(registry, "fmod", hle_fmod);
    register_abi_float(registry, "fmodf", hle_fmodf);
    register_abi_float(registry, "ldexpf", hle_ldexpf);
    register_abi_float(registry, "log", hle_log);
    register_abi_float(registry, "log10", hle_log10);
    register_abi_float(registry, "log10f", hle_log10f);
    register_abi_float(registry, "log2", hle_log2);
    register_abi_float(registry, "log2f", hle_log2f);
    register_abi_float(registry, "logf", hle_logf);
    register_abi_float(registry, "nextafterf", hle_nextafterf);
    register_abi_float(registry, "pow", hle_pow);
    register_abi_float(registry, "powf", hle_powf);
    register_abi_float(registry, "sin", hle_sin);
    register_abi_float(registry, "sinf", hle_sinf);
    register_abi_float(registry, "strtod", hle_strtod);
    register_abi_float(registry, "tan", hle_tan);
    register_abi_float(registry, "tanf", hle_tanf);
    register_abi_float(registry, "tanhf", hle_tanhf);
    // crt0 process exit.
    register_abi(registry, "catchReturnFromMain", hle_catch_return_from_main);
    // C++ ABI: exception-object storage is real; the unwinder itself is not
    // implemented (skipped names listed above `kernel_key`).
    register_abi(
        registry,
        "__cxa_allocate_exception",
        hle_cxa_allocate_exception,
    );
    register_abi(registry, "__cxa_free_exception", hle_cxa_free_exception);
    register_abi(registry, "__cxa_thread_atexit", hle_cxa_thread_atexit);
    register_abi(registry, "__cxa_bad_cast", hle_cxa_bad_cast);
    register_abi(
        registry,
        "_ZSt18uncaught_exceptionv",
        hle_std_uncaught_exception,
    );
    // C++ throw-path helpers: noreturn throw functions Raeen cannot unwind.
    // Each logs the exact throw (message + caller) and returns 0 — the same
    // defined-failure pattern as `hle_cxa_pure_virtual` / `hle_abort`.
    register_abi(registry, "_ZSt11_Xbad_allocv", hle_std_xbad_alloc);
    register_abi(registry, "_ZSt14_Throw_C_errori", hle_std_throw_c_error);
    register_abi(registry, "_ZSt14_Xlength_errorPKc", hle_std_xlength_error);
    register_abi(registry, "_ZSt14_Xout_of_rangePKc", hle_std_xout_of_range);
    register_abi(registry, "_ZSt16_Throw_Cpp_errori", hle_std_throw_cpp_error);
    register_abi(
        registry,
        "_ZSt19_Xbad_function_callv",
        hle_std_xbad_function_call,
    );
    register_abi(registry, "_ZSt9terminatev", hle_std_terminate);
    register_abi(
        registry,
        "_ZNKSt9exception6_RaiseEv",
        hle_std_exception_raise,
    );
    register_abi(
        registry,
        "_ZNKSt9exception8_DoraiseEv",
        hle_std_exception_doraise,
    );
}

/// In-flight `__cxa_guard_acquire` claims, keyed by guest guard address, with a
/// condvar so a second guest thread waits for the initializer instead of racing
/// it. Empty and untouched unless a guest actually uses function-local statics.
static CXA_GUARDS: std::sync::LazyLock<(
    std::sync::Mutex<std::collections::HashSet<u64>>,
    std::sync::Condvar,
)> = std::sync::LazyLock::new(|| {
    (
        std::sync::Mutex::new(std::collections::HashSet::new()),
        std::sync::Condvar::new(),
    )
});

/// How long a waiter re-checks the guard flag before giving up and reporting
/// "already initialized". Only reached if the owning thread never released,
/// which a well-formed C++ program cannot do (`release`/`abort` always follow).
const CXA_GUARD_WAIT_SLICE: std::time::Duration = std::time::Duration::from_millis(50);

/// `__cxa_guard_acquire(guard) -> int`: the Itanium C++ ABI entry protecting a
/// function-local `static`. Returns **1** when this caller must run the
/// initializer (and now owns the guard), **0** when the object is already
/// constructed.
///
/// The guard object's first byte is the "initialized" flag. Guest threads can
/// reach the same static concurrently, so the read-check-claim is made atomic
/// under a host mutex: a thread that finds another already initializing waits
/// for the flag rather than constructing the object a second time — running a
/// C++ static constructor twice is exactly the corruption this guard exists to
/// prevent.
///
/// Measured: A Plague Tale Requiem stops its boot on this import.
fn hle_cxa_guard_acquire(ctx: &HleContext, args: &[u64]) -> u64 {
    let guard = args.first().copied().unwrap_or(0);
    if guard == 0 {
        return 0;
    }
    let (lock, condvar) = &*CXA_GUARDS;
    let mut in_progress = lock.lock().unwrap_or_else(|p| p.into_inner());
    loop {
        let mut flag = [0u8; 1];
        if !ctx.mem.read(guard, &mut flag) {
            // An unreadable guard cannot be claimed safely; report "done" so the
            // guest skips construction rather than building into bad memory.
            return 0;
        }
        if flag[0] != 0 {
            return 0;
        }
        if in_progress.insert(guard) {
            return 1;
        }
        // Another guest thread owns this guard: wait for its release/abort.
        let (guard_set, timeout) = condvar
            .wait_timeout(in_progress, CXA_GUARD_WAIT_SLICE)
            .unwrap_or_else(|p| p.into_inner());
        in_progress = guard_set;
        if timeout.timed_out() && !in_progress.contains(&guard) {
            // Owner released while we were not scheduled; loop re-reads the flag.
            continue;
        }
    }
}

/// `__cxa_guard_release(guard)`: the initializer finished — mark the static
/// constructed and wake anyone waiting on it.
fn hle_cxa_guard_release(ctx: &HleContext, args: &[u64]) -> u64 {
    let guard = args.first().copied().unwrap_or(0);
    if guard == 0 {
        return 0;
    }
    let (lock, condvar) = &*CXA_GUARDS;
    let mut in_progress = lock.lock().unwrap_or_else(|p| p.into_inner());
    let _ = ctx.mem.write(guard, &[1u8]);
    in_progress.remove(&guard);
    condvar.notify_all();
    0
}

/// `__cxa_guard_abort(guard)`: the initializer threw — release the claim WITHOUT
/// marking the static constructed, so the next caller retries it.
fn hle_cxa_guard_abort(_ctx: &HleContext, args: &[u64]) -> u64 {
    let guard = args.first().copied().unwrap_or(0);
    if guard == 0 {
        return 0;
    }
    let (lock, condvar) = &*CXA_GUARDS;
    let mut in_progress = lock.lock().unwrap_or_else(|p| p.into_inner());
    in_progress.remove(&guard);
    condvar.notify_all();
    0
}

/// `_init_env()`: libc's pre-`main` environment initialiser.
///
/// Raeen builds the process environment itself — `build_process_stack` lays out
/// `argc`/`argv`/`envp`/`auxv` before `_start` — so there is nothing left for
/// the guest CRT to set up here. Succeeding with zero matches both references:
/// SharpEmu returns `rax = 0` / OK, and shadPS4 stubs it.
///
/// Measured: A Plague Tale Requiem stops its boot on this import.
fn hle_init_env(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0
}

/// The public C ABI is exposed through both the generic libc view used by
/// homebrew fixtures and the provider name carried by retail PRX imports.
/// Keeping the alias explicit preserves provider-aware NID collision safety.
fn register_abi(registry: &HleRegistry, function: &str, implementation: HleFunction) {
    registry.register("libc", function, implementation);
    registry.register("libSceLibcInternal", function, implementation);
}

/// The float-returning twin of [`register_abi`]: registers under both
/// provider views AND marks the function float-returning
/// ([`HleRegistry::register_float`]) so the runtime delivers the handler's
/// `u64` result bits to guest XMM0 (SysV) instead of only RAX.
fn register_abi_float(registry: &HleRegistry, function: &str, implementation: HleFunction) {
    registry.register_float("libc", function, implementation);
    registry.register_float("libSceLibcInternal", function, implementation);
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
/// Opt-in heap poison (`RAEEN_POISON_HEAP=1`). Ported in spirit from SharpEmu,
/// which fills fresh allocations so an uninitialized read surfaces as a
/// recognizable byte pattern (`0xCDCDCDCD…`) in the crash dump instead of a
/// silent zero that corrupts millions of ops downstream. Off by default: a
/// title that (buggily) relies on `malloc` returning zeroed memory keeps
/// working, and the poison is a deliberate debugging choice, not a behaviour
/// change. Read once — the env var never changes mid-run.
fn poison_heap_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RAEEN_POISON_HEAP").is_some())
}

/// The uninitialized-allocation poison byte (SharpEmu / MSVC debug-CRT `0xCD`).
const HEAP_ALLOC_POISON: u8 = 0xCD;

fn hle_malloc(ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.first().copied().unwrap_or(0);
    debug!("malloc(size={size:#x})");
    let addr = ctx.alloc.alloc(size, 16).unwrap_or(0);
    track_alloc(ctx, addr, size);
    // Poison the requested span so a read before write is visible. Only the
    // `size` bytes the caller asked for are touched — exactly what a real
    // program may read — and never on a failed (0) allocation.
    if addr != 0 && size != 0 && poison_heap_enabled() {
        let _ = mem_fill(ctx, addr, HEAP_ALLOC_POISON, size);
    }
    addr
}

/// Real `free` releases a block previously returned by `malloc`/`calloc`/
/// `realloc`/`memalign`. `free(NULL)` is a defined no-op in the real API, so
/// a `ptr == 0` is not even forwarded to the allocator.
fn hle_free(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    debug!("free(ptr={ptr:#x})");
    if ptr != 0 {
        track_free(ctx, ptr);
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
    if !crate::zero_guest_range(ctx.mem, addr, total) {
        warn!("calloc: zeroing block at {addr:#x} (len {total:#x}) failed");
        ctx.alloc.free(addr);
        return 0;
    }
    track_alloc(ctx, addr, total);
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
        let addr = ctx.alloc.alloc(size, 16).unwrap_or(0);
        track_alloc(ctx, addr, size);
        return addr;
    }
    let addr = ctx.alloc.realloc(ptr, size).unwrap_or(0);
    if addr != 0 {
        track_free(ctx, ptr);
        track_alloc(ctx, addr, size);
    }
    addr
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

    if n > crate::MAX_HLE_BULK_BYTES {
        warn!("strncpy: n={n:#x} exceeds the bounded HLE bulk-operation limit");
        return dst;
    }
    let Some(bytes) = crate::fmt::read_cstr(ctx.mem, src) else {
        warn!("strncpy: unreadable source string at {src:#x}");
        return dst;
    };
    let Some(range) = crate::GuestRange::new(crate::GuestAddress::new(dst), n) else {
        warn!("strncpy: destination range overflows");
        return dst;
    };
    if crate::ValidatedGuestRange::validate(ctx.mem, range, crate::GuestAccess::Write).is_none() {
        warn!("strncpy: destination range is not writable");
        return dst;
    }
    let copy_len = bytes.len().min(n as usize);
    if copy_len > 0 && !ctx.mem.write(dst, &bytes[..copy_len]) {
        warn!("strncpy: failed to write source bytes to dst={dst:#x}");
        return dst;
    }
    let zero_len = n - copy_len as u64;
    if zero_len > 0 && !crate::zero_guest_range(ctx.mem, dst + copy_len as u64, zero_len) {
        warn!("strncpy: failed to zero-fill destination at dst={dst:#x}");
    }
    dst
}

/// Cap on how many bytes a single comparison/scan reads out of guest memory,
/// so a wild `n` can't balloon a host buffer. Matches `STRLEN_MAX_SCAN`'s
/// rationale (both are 1 MiB).
const CMP_MAX_BYTES: u64 = 1 << 20;

/// Read `n` guest bytes at `addr` (capped by [`CMP_MAX_BYTES`]) into an owned
/// buffer, or `None` if the range isn't fully readable. Used by the compare/
/// scan functions below, which need a concrete byte window rather than a
/// NUL-terminated string.
fn read_guest_bytes(ctx: &HleContext, addr: u64, n: u64) -> Option<Vec<u8>> {
    let capped = n.min(CMP_MAX_BYTES);
    if capped < n {
        // Truncating a comparison/scan over *valid* memory yields a wrong
        // answer (a false "equal" / "not found" past the cap), so warn —
        // matching `read_cstr`'s own truncation warning — rather than
        // silently returning a partial result.
        warn!("read_guest_bytes: {n:#x} bytes at {addr:#x} capped to {capped:#x} (CMP_MAX_BYTES)");
    }
    let len = usize::try_from(capped).ok()?;
    let mut buf = vec![0u8; len];
    if len == 0 {
        return Some(buf);
    }
    if ctx.mem.read(addr, &mut buf) {
        Some(buf)
    } else {
        None
    }
}

/// Real `memcmp(a, b, n)`: reads `n` bytes from each pointer and returns the
/// sign of the first differing byte pair (unsigned), `0` if equal. An
/// unreadable range logs and reports `0` (equal) — the least-surprising
/// degradation for a comparison with no error channel.
fn hle_memcmp(ctx: &HleContext, args: &[u64]) -> u64 {
    let a = args.first().copied().unwrap_or(0);
    let b = args.get(1).copied().unwrap_or(0);
    let n = args.get(2).copied().unwrap_or(0);
    debug!("memcmp(a={a:#x}, b={b:#x}, n={n:#x})");

    let (Some(ba), Some(bb)) = (read_guest_bytes(ctx, a, n), read_guest_bytes(ctx, b, n)) else {
        warn!("memcmp: unreadable range (a={a:#x}, b={b:#x}, n={n:#x})");
        return 0;
    };
    for (x, y) in ba.iter().zip(bb.iter()) {
        if x != y {
            return if x < y { (-1i32) as u32 as u64 } else { 1 };
        }
    }
    0
}

/// Real `memchr(s, c, n)`: returns the guest address of the first byte equal
/// to `c` (low 8 bits) within the first `n` bytes of `s`, or `0` (`NULL`) if
/// not found or the range is unreadable.
fn hle_memchr(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    let c = (args.get(1).copied().unwrap_or(0) & 0xFF) as u8;
    let n = args.get(2).copied().unwrap_or(0);
    debug!("memchr(s={s:#x}, c={c:#x}, n={n:#x})");

    let Some(bytes) = read_guest_bytes(ctx, s, n) else {
        warn!("memchr: unreadable range (s={s:#x}, n={n:#x})");
        return 0;
    };
    match bytes.iter().position(|&b| b == c) {
        Some(off) => s.wrapping_add(off as u64),
        None => 0,
    }
}

/// Real `strncmp(a, b, n)`: compares up to `n` bytes of the two guest
/// strings, stopping at the first NUL, unsigned. Unreadable pointer → `0`.
fn hle_strncmp(ctx: &HleContext, args: &[u64]) -> u64 {
    let a = args.first().copied().unwrap_or(0);
    let b = args.get(1).copied().unwrap_or(0);
    let n = args.get(2).copied().unwrap_or(0);
    debug!("strncmp(a={a:#x}, b={b:#x}, n={n:#x})");

    let (Some(sa), Some(sb)) = (
        crate::fmt::read_cstr(ctx.mem, a),
        crate::fmt::read_cstr(ctx.mem, b),
    ) else {
        warn!("strncmp: unreadable string (a={a:#x}, b={b:#x})");
        return 0;
    };
    let limit = usize::try_from(n).unwrap_or(usize::MAX);
    let ta = &sa[..sa.len().min(limit)];
    let tb = &sb[..sb.len().min(limit)];
    match ta.cmp(tb) {
        std::cmp::Ordering::Less => (-1i32) as u32 as u64,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// Real `strnlen(s, maxlen)`: length of the guest string, capped at `maxlen`
/// (and at `read_cstr`'s own 1 MiB scan bound). Unreadable → `0`.
fn hle_strnlen(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    let maxlen = args.get(1).copied().unwrap_or(0);
    debug!("strnlen(s={s:#x}, maxlen={maxlen:#x})");

    // Real `strnlen` examines at most `maxlen` bytes — read exactly that
    // window (bounded by CMP_MAX_BYTES) rather than scanning the whole
    // string first, so the read footprint matches the C contract.
    let window = maxlen.min(CMP_MAX_BYTES);
    let Some(bytes) = read_guest_bytes(ctx, s, window) else {
        warn!("strnlen: unreadable window at {s:#x}");
        return 0;
    };
    match bytes.iter().position(|&b| b == 0) {
        Some(nul) => nul as u64,
        None => window, // no NUL within maxlen → maxlen (capped)
    }
}

/// Real `strchr(s, c)`: guest address of the first byte equal to `c` (low 8
/// bits) in the string, or `0`. Per the C contract, `c == 0` matches (and
/// returns the address of) the terminating NUL.
fn hle_strchr(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    let c = (args.get(1).copied().unwrap_or(0) & 0xFF) as u8;
    debug!("strchr(s={s:#x}, c={c:#x})");

    let Some(bytes) = crate::fmt::read_cstr(ctx.mem, s) else {
        warn!("strchr: unreadable string at {s:#x}");
        return 0;
    };
    if c == 0 {
        return s.wrapping_add(bytes.len() as u64); // the NUL terminator
    }
    match bytes.iter().position(|&b| b == c) {
        Some(off) => s.wrapping_add(off as u64),
        None => 0,
    }
}

/// Saved scan position for [`hle_strtok`] — C's `strtok` is stateful across
/// calls. Classic `strtok` keeps one global cursor (that is exactly why it is
/// not re-entrant), and this mirrors it rather than inventing per-thread state
/// the guest would not expect.
static STRTOK_SAVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Real `strtok(str, delim)`: return the next token in `str`, writing a NUL over
/// the delimiter that ends it and remembering where to resume.
///
/// `str == NULL` continues the previous string, per the C contract. Returns
/// `NULL` once the string is exhausted. The token is delimited **in guest
/// memory** — this function genuinely mutates the caller's buffer, as the real
/// one does.
///
/// Measured: A Plague Tale Requiem stops its boot on this import.
fn hle_strtok(ctx: &HleContext, args: &[u64]) -> u64 {
    use std::sync::atomic::Ordering;

    let s = args.first().copied().unwrap_or(0);
    let delim_ptr = args.get(1).copied().unwrap_or(0);
    debug!("strtok(str={s:#x}, delim={delim_ptr:#x})");

    let start = if s != 0 {
        s
    } else {
        STRTOK_SAVE.load(Ordering::Relaxed)
    };
    if start == 0 {
        return 0;
    }
    let Some(delims) = crate::fmt::read_cstr(ctx.mem, delim_ptr) else {
        warn!("strtok: unreadable delimiter set at {delim_ptr:#x}");
        return 0;
    };

    let read_byte = |addr: u64| -> Option<u8> {
        let mut b = [0u8; 1];
        ctx.mem.read(addr, &mut b).then_some(b[0])
    };

    // Skip leading delimiters.
    let mut p = start;
    let mut scanned = 0u64;
    let token = loop {
        if scanned >= STRLEN_MAX_SCAN {
            STRTOK_SAVE.store(0, Ordering::Relaxed);
            return 0;
        }
        match read_byte(p) {
            None | Some(0) => {
                STRTOK_SAVE.store(0, Ordering::Relaxed);
                return 0;
            }
            Some(b) if delims.contains(&b) => {
                p = p.wrapping_add(1);
                scanned += 1;
            }
            Some(_) => break p,
        }
    };

    // Run to the delimiter that ends this token (or the string's end).
    loop {
        if scanned >= STRLEN_MAX_SCAN {
            break;
        }
        match read_byte(p) {
            None | Some(0) => {
                STRTOK_SAVE.store(0, Ordering::Relaxed);
                return token;
            }
            Some(b) if delims.contains(&b) => {
                // Terminate the token in place and resume after it.
                let _ = ctx.mem.write(p, &[0u8]);
                STRTOK_SAVE.store(p.wrapping_add(1), Ordering::Relaxed);
                return token;
            }
            Some(_) => {
                p = p.wrapping_add(1);
                scanned += 1;
            }
        }
    }
    STRTOK_SAVE.store(0, Ordering::Relaxed);
    token
}

/// A SysV `va_list` walked out of guest memory, yielding the integer varargs in
/// call order so it can drive [`crate::fmt::format_c`] exactly like the
/// register slice a plain `printf` hands it.
///
/// The ABI's `va_list` is a 24-byte object: `gp_offset` (u32), `fp_offset`
/// (u32), `overflow_arg_area` (ptr), `reg_save_area` (ptr). Integer varargs
/// come from `reg_save_area + gp_offset` while `gp_offset < 48` (the six GP
/// registers), and from `overflow_arg_area` — the caller's stack — afterwards.
///
/// Like the register-slice path, only *integer* varargs are walked; a `%f`
/// would need the XMM save area, which no caller has required yet.
#[derive(Clone, Copy)]
struct GuestVaList<'a> {
    mem: &'a dyn crate::GuestMemory,
    gp_offset: u32,
    /// Byte offset of the next unconsumed XMM slot in the register save area.
    /// Variadic floats travel in XMM registers, tracked independently of
    /// `gp_offset` — see [`GuestVaListFloats`].
    fp_offset: u32,
    overflow_arg_area: u64,
    reg_save_area: u64,
}

/// The floating-point half of a `va_list`, walked alongside the integer half.
///
/// SysV splits variadic arguments across two register files, so a `va_list`
/// carries two independent cursors. Yielding both from one iterator would
/// desynchronize them; this borrows the same list and advances only
/// `fp_offset`.
struct GuestVaListFloats<'a, 'b>(&'b mut GuestVaList<'a>);

impl Iterator for GuestVaListFloats<'_, '_> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        /// The XMM save area follows the six GP slots...
        const FP_SAVE_START: u32 = 48;
        /// ...and holds eight 16-byte slots.
        const FP_SAVE_END: u32 = FP_SAVE_START + 8 * 16;

        let list = &mut *self.0;
        let addr = if list.fp_offset < FP_SAVE_END {
            let addr = list.reg_save_area.checked_add(u64::from(list.fp_offset))?;
            list.fp_offset += 16;
            addr
        } else {
            // Spilled to the stack: doubles occupy one 8-byte slot each.
            let addr = list.overflow_arg_area;
            list.overflow_arg_area = addr.checked_add(8)?;
            addr
        };
        let mut word = [0u8; 8];
        if !list.mem.read(addr, &mut word) {
            return None;
        }
        Some(f64::from_bits(u64::from_le_bytes(word)))
    }
}

impl<'a> GuestVaList<'a> {
    /// Read the `va_list` object at guest address `ap`.
    fn read(mem: &'a dyn crate::GuestMemory, ap: u64) -> Option<Self> {
        let mut head = [0u8; 24];
        if ap == 0 || !mem.read(ap, &mut head) {
            return None;
        }
        Some(Self {
            mem,
            gp_offset: u32::from_le_bytes(head[0..4].try_into().ok()?),
            fp_offset: u32::from_le_bytes(head[4..8].try_into().ok()?),
            overflow_arg_area: u64::from_le_bytes(head[8..16].try_into().ok()?),
            reg_save_area: u64::from_le_bytes(head[16..24].try_into().ok()?),
        })
    }
}

impl Iterator for GuestVaList<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        /// Six 8-byte GP register slots live at the head of the save area.
        const GP_SAVE_BYTES: u32 = 48;

        let addr = if self.gp_offset < GP_SAVE_BYTES {
            let addr = self.reg_save_area.checked_add(u64::from(self.gp_offset))?;
            self.gp_offset += 8;
            addr
        } else {
            let addr = self.overflow_arg_area;
            self.overflow_arg_area = addr.checked_add(8)?;
            addr
        };
        let mut word = [0u8; 8];
        if !self.mem.read(addr, &mut word) {
            return None;
        }
        Some(u64::from_le_bytes(word))
    }
}

/// Real `vsnprintf(str, size, format, ap)`: [`hle_snprintf`] with the varargs
/// taken from a guest `va_list` instead of the caller's registers. Truncation
/// and the "length that *would* have been written" return value follow the same
/// C contract.
///
/// Measured: A Plague Tale Requiem stops its boot on this import.
fn hle_vsnprintf(ctx: &HleContext, args: &[u64]) -> u64 {
    let buf = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    let fmt_ptr = args.get(2).copied().unwrap_or(0);
    let ap = args.get(3).copied().unwrap_or(0);
    debug!("vsnprintf(buf={buf:#x}, size={size:#x}, fmt={fmt_ptr:#x}, ap={ap:#x})");

    let Some(fmt) = crate::fmt::read_cstr(ctx.mem, fmt_ptr) else {
        warn!("vsnprintf: unreadable format string at {fmt_ptr:#x}");
        return 0;
    };
    let Some(mut varargs) = GuestVaList::read(ctx.mem, ap) else {
        warn!("vsnprintf: unreadable va_list at {ap:#x}");
        return 0;
    };
    // The two register files are walked by independent cursors, so the float
    // view gets its own copy of the list rather than aliasing the integer one.
    //
    // LIMIT: both copies own an `overflow_arg_area` cursor. Arguments that fit
    // in registers — the first six integers and eight floats, which covers
    // essentially every real `printf` — are exact; a call that spills BOTH
    // kinds to the stack would have the two cursors read the same slots. Left
    // as-is deliberately: the alternative is a shared-cursor rewrite of
    // `format_c`'s signature for a case no measured title reaches.
    let mut float_list = varargs;
    let mut floats = GuestVaListFloats(&mut float_list);
    let formatted = crate::fmt::format_c(&fmt, &mut varargs, &mut floats, ctx.mem);
    let full_len = formatted.len() as u64;

    if size > 0 {
        let cap = usize::try_from(size - 1).unwrap_or(usize::MAX);
        let n = formatted.len().min(cap);
        let mut out = formatted;
        out.truncate(n);
        out.push(0);
        if !ctx.mem.write(buf, &out) {
            warn!(
                "vsnprintf: failed to write {} bytes to buf={buf:#x}",
                out.len()
            );
        }
    }
    full_len
}

/// Real `sincosf(float x, float *sinp, float *cosp)`: write `sin(x)` and
/// `cos(x)` through the two out-pointers.
///
/// `x` arrives in **XMM0**, not in an integer register, so it is read from
/// [`HleContext::float_arg_f32`]; the two pointers are the first two integer
/// arguments. Computing this without the real `x` would hand the title silently
/// wrong trigonometry, so the float-argument channel exists precisely for
/// functions like this one.
///
/// Measured: A Plague Tale Requiem stops its boot on this import.
fn hle_sincosf(ctx: &HleContext, args: &[u64]) -> u64 {
    let sin_out = args.first().copied().unwrap_or(0);
    let cos_out = args.get(1).copied().unwrap_or(0);
    let x = ctx.float_arg_f32(0);
    debug!("sincosf(x={x}, sinp={sin_out:#x}, cosp={cos_out:#x})");

    if sin_out != 0 && !ctx.mem.write(sin_out, &x.sin().to_le_bytes()) {
        warn!("sincosf: sin out-ptr {sin_out:#x} not writable");
    }
    if cos_out != 0 && !ctx.mem.write(cos_out, &x.cos().to_le_bytes()) {
        warn!("sincosf: cos out-ptr {cos_out:#x} not writable");
    }
    0
}

/// Real `wcslen(s)`: number of wide characters before the terminating null.
///
/// `wchar_t` is **32-bit** on the PS5's BSD/LLVM ABI (not 16-bit as on
/// Windows), so this counts 4-byte units. The scan is bounded by
/// [`STRLEN_MAX_SCAN`] characters, like [`hle_strlen`], so a wild or
/// unterminated guest pointer cannot spin forever; an unreadable unit ends the
/// scan and returns the count so far.
///
/// Measured: A Plague Tale Requiem stops its boot on this import.
fn hle_wcslen(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    debug!("wcslen(s={s:#x})");
    if s == 0 {
        return 0;
    }
    let mut count = 0u64;
    let mut unit = [0u8; 4];
    while count < STRLEN_MAX_SCAN {
        let Some(addr) = s.checked_add(count.wrapping_mul(4)) else {
            break;
        };
        if !ctx.mem.read(addr, &mut unit) {
            warn!("wcslen: unreadable wide char at {addr:#x} (after {count})");
            break;
        }
        if u32::from_le_bytes(unit) == 0 {
            break;
        }
        count += 1;
    }
    count
}

/// Real `wcscpy(dst, src)`: copy the wide string at `src` — including its
/// 4-byte null terminator — to `dst`, returning `dst`.
///
/// Uses [`hle_wcslen`]'s bounded scan to size the copy, so an unterminated
/// source cannot run away. Mirrors [`hle_strcpy`]: an unreadable source or
/// unwritable destination is logged and `dst` is still returned, because the
/// ABI has no way to report failure.
///
/// Measured: A Plague Tale Requiem stops its boot on this import.
fn hle_wcscpy(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    debug!("wcscpy(dst={dst:#x}, src={src:#x})");
    if dst == 0 || src == 0 {
        return dst;
    }

    // Characters, then bytes including the terminator.
    let chars = hle_wcslen(ctx, &[src]);
    let Ok(bytes) = usize::try_from((chars + 1).saturating_mul(4)) else {
        return dst;
    };
    let mut buf = vec![0u8; bytes];
    if !ctx.mem.read(src, &mut buf) {
        warn!("wcscpy: unreadable source at {src:#x} ({chars} wide chars)");
        return dst;
    }
    // `wcslen` stops at the terminator, so force it rather than trusting the
    // trailing unit we just read.
    buf[bytes - 4..].fill(0);
    if !ctx.mem.write(dst, &buf) {
        warn!("wcscpy: failed to write {bytes} bytes to dst={dst:#x}");
    }
    dst
}

/// Real `strcspn(s, reject)`: the length of the initial segment of `s` made up
/// of bytes **not** in `reject` — i.e. the offset of the first rejected byte,
/// or the whole length if none match.
///
/// An unreadable operand yields `0`, the conservative answer: the caller then
/// behaves as though `s` was rejected immediately rather than scanning past
/// memory we could not validate.
///
/// Measured: A Plague Tale Requiem stops its boot on this import.
fn hle_strcspn(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    let reject = args.get(1).copied().unwrap_or(0);
    debug!("strcspn(s={s:#x}, reject={reject:#x})");

    let (Some(bytes), Some(set)) = (
        crate::fmt::read_cstr(ctx.mem, s),
        crate::fmt::read_cstr(ctx.mem, reject),
    ) else {
        warn!("strcspn: unreadable operand (s={s:#x}, reject={reject:#x})");
        return 0;
    };
    bytes
        .iter()
        .position(|byte| set.contains(byte))
        .unwrap_or(bytes.len()) as u64
}

/// Real `strrchr(s, c)`: guest address of the *last* byte equal to `c`, or
/// `0`. `c == 0` matches the terminating NUL.
fn hle_strrchr(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    let c = (args.get(1).copied().unwrap_or(0) & 0xFF) as u8;
    debug!("strrchr(s={s:#x}, c={c:#x})");

    let Some(bytes) = crate::fmt::read_cstr(ctx.mem, s) else {
        warn!("strrchr: unreadable string at {s:#x}");
        return 0;
    };
    if c == 0 {
        return s.wrapping_add(bytes.len() as u64);
    }
    match bytes.iter().rposition(|&b| b == c) {
        Some(off) => s.wrapping_add(off as u64),
        None => 0,
    }
}

/// Real `strcat(dst, src)`: appends the `src` string to `dst` (writing a new
/// NUL), returning `dst`. Reads `dst`'s current length, then writes
/// `src + NUL` at `dst + len`.
fn hle_strcat(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    debug!("strcat(dst={dst:#x}, src={src:#x})");

    let (Some(dst_bytes), Some(mut src_bytes)) = (
        crate::fmt::read_cstr(ctx.mem, dst),
        crate::fmt::read_cstr(ctx.mem, src),
    ) else {
        warn!("strcat: unreadable string (dst={dst:#x}, src={src:#x})");
        return dst;
    };
    let append_at = dst.wrapping_add(dst_bytes.len() as u64);
    src_bytes.push(0);
    if !ctx.mem.write(append_at, &src_bytes) {
        warn!(
            "strcat: failed to append {} bytes at {append_at:#x}",
            src_bytes.len()
        );
    }
    dst
}

/// Real `strncat(dst, src, n)`: appends at most `n` bytes of `src` to `dst`,
/// always writing a terminating NUL after them.
fn hle_strncat(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    let n = args.get(2).copied().unwrap_or(0);
    debug!("strncat(dst={dst:#x}, src={src:#x}, n={n:#x})");

    let (Some(dst_bytes), Some(src_bytes)) = (
        crate::fmt::read_cstr(ctx.mem, dst),
        crate::fmt::read_cstr(ctx.mem, src),
    ) else {
        warn!("strncat: unreadable string (dst={dst:#x}, src={src:#x})");
        return dst;
    };
    let take = usize::try_from(n)
        .unwrap_or(usize::MAX)
        .min(src_bytes.len());
    let mut out = src_bytes[..take].to_vec();
    out.push(0);
    let append_at = dst.wrapping_add(dst_bytes.len() as u64);
    if !ctx.mem.write(append_at, &out) {
        warn!(
            "strncat: failed to append {} bytes at {append_at:#x}",
            out.len()
        );
    }
    dst
}

/// Real `strstr(haystack, needle)`: guest address of the first occurrence of
/// `needle` in `haystack`, or `0`. An empty needle returns `haystack` (the C
/// contract).
fn hle_strstr(ctx: &HleContext, args: &[u64]) -> u64 {
    let haystack = args.first().copied().unwrap_or(0);
    let needle = args.get(1).copied().unwrap_or(0);
    debug!("strstr(haystack={haystack:#x}, needle={needle:#x})");

    let (Some(hay), Some(ndl)) = (
        crate::fmt::read_cstr(ctx.mem, haystack),
        crate::fmt::read_cstr(ctx.mem, needle),
    ) else {
        warn!("strstr: unreadable string (haystack={haystack:#x}, needle={needle:#x})");
        return 0;
    };
    if ndl.is_empty() {
        return haystack;
    }
    // `memchr::memmem::find` is linear (Two-Way + SIMD prefilter), so a
    // hostile "aaaa…" haystack + "aaaa…b" needle can't force the O(n·m)
    // blowup a naive `windows().position()` scan would (both inputs are
    // guest-controlled, each up to 1 MiB from `read_cstr`).
    match memchr::memmem::find(&hay, &ndl) {
        Some(off) => haystack.wrapping_add(off as u64),
        None => 0,
    }
}

/// Parse a C integer out of a guest string per `strtol`/`strtoul` rules:
/// skip leading ASCII whitespace, optional `+`/`-` sign, then digits in
/// `base` (base 0 auto-detects `0x`/`0X` → 16, leading `0` → 8, else 10).
/// Returns `(value_as_i128, bytes_consumed_from_string_start)`; the caller
/// clamps/casts the value and computes the `endptr`. Stops at the first
/// non-convertible character (the real API's behavior). Overflow saturates.
fn parse_c_integer(s: &[u8], mut base: u32) -> (i128, usize) {
    let mut i = 0usize;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    let mut negative = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        negative = s[i] == b'-';
        i += 1;
    }
    // Base auto-detection / `0x` prefix consumption.
    if (base == 0 || base == 16)
        && i + 1 < s.len()
        && s[i] == b'0'
        && (s[i + 1] == b'x' || s[i + 1] == b'X')
        && s.get(i + 2).is_some_and(|c| c.is_ascii_hexdigit())
    {
        base = 16;
        i += 2;
    } else if base == 0 && i < s.len() && s[i] == b'0' {
        base = 8;
    } else if base == 0 {
        base = 10;
    }

    let mut value: i128 = 0;
    let mut any = false;
    while i < s.len() {
        let Some(digit) = (s[i] as char).to_digit(base) else {
            break;
        };
        any = true;
        value = value
            .saturating_mul(base as i128)
            .saturating_add(digit as i128);
        i += 1;
    }
    if !any {
        return (0, i); // no digits converted → 0, endptr at the sign/start
    }
    (if negative { -value } else { value }, i)
}

/// Real `atoi(nptr)`: `strtol(nptr, NULL, 10)` truncated to a 32-bit `int`.
fn hle_atoi(ctx: &HleContext, args: &[u64]) -> u64 {
    let nptr = args.first().copied().unwrap_or(0);
    debug!("atoi(nptr={nptr:#x})");
    let Some(bytes) = crate::fmt::read_cstr(ctx.mem, nptr) else {
        warn!("atoi: unreadable string at {nptr:#x}");
        return 0;
    };
    let (v, _) = parse_c_integer(&bytes, 10);
    (v as i32) as u32 as u64
}

/// Real `atol(nptr)`: `strtol(nptr, NULL, 10)` as a 64-bit `long`.
fn hle_atol(ctx: &HleContext, args: &[u64]) -> u64 {
    let nptr = args.first().copied().unwrap_or(0);
    debug!("atol(nptr={nptr:#x})");
    let Some(bytes) = crate::fmt::read_cstr(ctx.mem, nptr) else {
        warn!("atol: unreadable string at {nptr:#x}");
        return 0;
    };
    let (v, _) = parse_c_integer(&bytes, 10);
    (v as i64) as u64
}

/// Real `strtol(nptr, endptr, base)`: parse a `long`, and if `endptr != NULL`
/// write the guest address of the first unconverted character through it.
/// Value saturates to the `long` range on overflow (approximating the real
/// `LONG_MIN`/`LONG_MAX` + `ERANGE` clamp, without the `errno` write).
fn hle_strtol(ctx: &HleContext, args: &[u64]) -> u64 {
    let nptr = args.first().copied().unwrap_or(0);
    let endptr = args.get(1).copied().unwrap_or(0);
    let base = args.get(2).copied().unwrap_or(0) as u32;
    debug!("strtol(nptr={nptr:#x}, endptr={endptr:#x}, base={base})");
    strtol_impl(ctx, nptr, endptr, base, false)
}

/// Real `strtoul(nptr, endptr, base)`: like `strtol` but unsigned. Shared
/// with `libSceLibcInternal`'s `_Stoul` (the Dinkumware STL's strtoul core,
/// same `(nptr, endptr, base)` signature) — see `libsce_libc_internal`.
pub(crate) fn hle_strtoul(ctx: &HleContext, args: &[u64]) -> u64 {
    let nptr = args.first().copied().unwrap_or(0);
    let endptr = args.get(1).copied().unwrap_or(0);
    let base = args.get(2).copied().unwrap_or(0) as u32;
    debug!("strtoul(nptr={nptr:#x}, endptr={endptr:#x}, base={base})");
    strtol_impl(ctx, nptr, endptr, base, true)
}

/// Shared `strtol`/`strtoul` body: parse, write `endptr`, clamp to the
/// signed or unsigned 64-bit range.
fn strtol_impl(ctx: &HleContext, nptr: u64, endptr: u64, base: u32, unsigned: bool) -> u64 {
    if base != 0 && !(2..=36).contains(&base) {
        warn!("strtol: invalid base {base}");
        return 0;
    }
    let Some(bytes) = crate::fmt::read_cstr(ctx.mem, nptr) else {
        warn!("strtol: unreadable string at {nptr:#x}");
        return 0;
    };
    let (v, consumed) = parse_c_integer(&bytes, base);
    if endptr != 0 {
        let end_addr = nptr.wrapping_add(consumed as u64);
        if !ctx.mem.write(endptr, &end_addr.to_le_bytes()) {
            warn!("strtol: failed to write endptr at {endptr:#x}");
        }
    }
    if unsigned {
        v.clamp(0, u64::MAX as i128) as u64
    } else {
        v.clamp(i64::MIN as i128, i64::MAX as i128) as i64 as u64
    }
}

/// `atexit(fn)`: record-and-succeed. A real libc runs registered callbacks at
/// `exit`; Raeen's `exit` HLE ends the process without running them (honest —
/// no atexit dispatch yet), but the registration itself must *succeed* (`0`)
/// or crt0/C++ static init aborts before `main`.
fn hle_atexit(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "atexit(fn={:#x}) [registered; not dispatched at exit yet]",
        args.first().copied().unwrap_or(0)
    );
    0
}

/// `__cxa_atexit(fn, arg, dso)`: the C++ ABI variant of `atexit` for global/
/// static destructors. Same record-and-succeed contract (`0`).
fn hle_cxa_atexit(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "__cxa_atexit(fn={:#x}, arg={:#x}, dso={:#x}) [registered; not dispatched at exit yet]",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    0
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
    // Variadic floats arrived in XMM0-7, captured separately from the GP args.
    let mut floats = ctx.float_args.iter().map(|bits| f64::from_bits(*bits));
    let formatted = crate::fmt::format_c(&fmt, &mut varargs, &mut floats, ctx.mem);
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
/// result to the kernel [`raeen_kernel::Console`] (captured for the Shell /
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
    // Variadic floats arrived in XMM0-7, captured separately from the GP args.
    let mut floats = ctx.float_args.iter().map(|bits| f64::from_bits(*bits));
    let formatted = crate::fmt::format_c(&fmt, &mut varargs, &mut floats, ctx.mem);
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

/// Shared actionable-report body for the guest-fatal noreturn handlers
/// (`abort`, `__stack_chk_fail`): the dying thread's recent HLE calls (the
/// most reliable "what was it doing" signal — host threads are pooled), a
/// stack code-address chain naming the call path INTO the fatal site, and an
/// EOWNERDEAD-style release of any spin-loop mutexes the thread holds so the
/// rest of the process keeps making observable progress (same recovery as
/// the DebugRaiseException path).
fn report_fatal_thread_diagnostics(ctx: &HleContext, thread: u64, name: &str) {
    if let Some(ring) = ctx.kernel.recent_hle_calls.get(&thread) {
        let recent = ring.lock().iter().cloned().collect::<Vec<_>>().join(" ");
        if !recent.is_empty() {
            tracing::error!("  recent HLE calls: {recent}");
        }
    }
    let chain = crate::guest_stack_code_addrs(ctx);
    if !chain.is_empty() {
        tracing::error!("  fatal-thread stack code-addrs: {}", chain.join(" "));
    }
    let released = ctx.kernel.release_mutexes_owned_by(thread);
    if released > 0 {
        warn!("  released {released} mutex(es) held by dying thread {thread} ('{name}')");
    }
}

/// Real `abort()` never returns: it raises SIGABRT and the compiler treats
/// the call as `noreturn`, so an HLE handler that returns 0 makes the guest
/// execute whatever bytes follow the call site — the same walk-into-garbage
/// hazard `hle_stack_chk_fail` had (measured on Until Dawn: the walk-off
/// landed in a UD2 and masked the real fatal cause). Report everything
/// actionable, then unwind the calling guest thread exactly like the other
/// guest-fatal handlers: `request_exit` makes the dispatcher restore the
/// recovery context, so control never returns to the aborting frame.
fn hle_abort(ctx: &HleContext, _args: &[u64]) -> u64 {
    let thread = ctx.guest_threads.current_thread();
    let name = ctx
        .kernel
        .thread_names
        .get(&thread)
        .map_or_else(|| "<unnamed>".to_owned(), |entry| entry.clone());
    tracing::error!(
        "abort(): guest called abort on thread {thread} ('{name}'), guest ra={:#x} — \
         terminating the calling guest thread (exit code {:#x}); the caller's fatal \
         path (assert/panic/terminate) decided the process state was invalid",
        ctx.caller_return_addr,
        crate::ABORT_EXIT_CODE,
    );
    report_fatal_thread_diagnostics(ctx, thread, &name);
    if !ctx.guest_threads.request_exit(crate::ABORT_EXIT_CODE) {
        // No per-thread unwind available (should not happen on the runtime
        // dispatch path): escalate to process termination rather than hand
        // control back to a frame that believes abort() cannot return.
        ctx.guest_threads
            .request_process_exit(crate::ABORT_EXIT_CODE);
    }
    // Unreachable by the guest: the dispatcher observes the exit request at
    // the HLE boundary and restores the recovery context instead of
    // delivering this value.
    crate::ABORT_EXIT_CODE
}

/// Real `exit(status)` never returns: it terminates the whole process with
/// the guest's own `status`. The old stub logged and returned 0, walking the
/// guest into whatever bytes follow the (noreturn) call site. On the
/// trampoline path `libc`/`libSceLibcInternal` `exit` is intercepted by the
/// runtime's terminating-function table before this handler runs; this
/// implementation is the defense-in-depth for every other route (direct
/// dispatch, in-tree callers, future provider aliases): record the status
/// process-wide so all workers stop at their next safe point, then unwind
/// the calling thread immediately. Unlike `abort`, this is an orderly
/// termination — the status carried is the guest's, not a fatal-family code.
fn hle_exit(ctx: &HleContext, args: &[u64]) -> u64 {
    let status = args.first().copied().unwrap_or(0);
    info!(
        "exit(status={status}) on thread {} — terminating the guest process",
        ctx.guest_threads.current_thread()
    );
    // Real exit() ends the whole process: every guest worker observes the
    // termination flag at its next HLE/fault safe point, and the process
    // outcome carries the guest's own status.
    ctx.guest_threads.request_process_exit(status);
    // Unwind THIS thread now so control never returns to the exiting frame.
    ctx.guest_threads.request_exit(status);
    // Unreachable by the guest (dispatcher restores the recovery context).
    status
}

/// Real `__stack_chk_fail` never returns: the compiler emits the call as the
/// last instruction of a smashed function's epilogue, so returning "executes"
/// whatever bytes follow — measured on Until Dawn, that walk-off landed in a
/// UD2 and got reported as OUR fault while the real cause (the corrupted
/// canary) vanished from the log. Report everything actionable, then unwind
/// the calling guest thread exactly like the other guest-fatal handlers
/// (`__cxa_throw` trap, `sceKernelDebugRaiseException`): `request_exit`
/// makes the dispatcher restore the recovery context, so control never
/// returns to the guest frame.
fn hle_stack_chk_fail(ctx: &HleContext, _args: &[u64]) -> u64 {
    let thread = ctx.guest_threads.current_thread();
    let name = ctx
        .kernel
        .thread_names
        .get(&thread)
        .map_or_else(|| "<unnamed>".to_owned(), |entry| entry.clone());
    tracing::error!(
        "__stack_chk_fail: guest stack canary smashed on thread {thread} ('{name}'), \
         guest ra={:#x} — terminating the calling guest thread (exit code {:#x}); \
         the frame that called this is the one that overflowed",
        ctx.caller_return_addr,
        crate::STACK_CHK_FAIL_EXIT_CODE,
    );
    report_fatal_thread_diagnostics(ctx, thread, &name);
    if !ctx
        .guest_threads
        .request_exit(crate::STACK_CHK_FAIL_EXIT_CODE)
    {
        // No per-thread unwind available (should not happen on the runtime
        // dispatch path): escalate to process termination rather than hand
        // control back to a smashed frame.
        ctx.guest_threads
            .request_process_exit(crate::STACK_CHK_FAIL_EXIT_CODE);
    }
    // Unreachable by the guest: the dispatcher observes the exit request at
    // the HLE boundary and restores the recovery context instead of
    // delivering this value.
    crate::STACK_CHK_FAIL_EXIT_CODE
}

/// Real `memalign(alignment, size)` allocates `size` bytes aligned to
/// `alignment`, honest-OOM (`0`) on failure.
fn hle_memalign(ctx: &HleContext, args: &[u64]) -> u64 {
    let alignment = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    debug!("memalign(alignment={alignment:#x}, size={size:#x})");
    let addr = ctx.alloc.alloc(size, alignment).unwrap_or(0);
    track_alloc(ctx, addr, size);
    addr
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
    track_alloc(ctx, addr, size);
    0
}

// ---------------------------------------------------------------------------
// NID-fill batch (measured: artifacts/compat/nid-coverage.json; list in
// scratch/nid-fill/missing-libc.txt): the libc internals retail titles
// import — Dinkumware CRT plumbing (_Getpctype/_Mtx_*/_Thrd_*), stdio FILE
// streams, the strto/scanf/time/locale families, the float-returning libm
// family, and the C++ throw-path helpers.
//
// Deliberately NOT implemented — left unresolved rather than registered as a
// lie (the unresolved-trampoline path already logs the NID loudly and the
// report back lists every one):
//
// * the C++ unwinder (_Unwind_Resume, __cxa_begin_catch, __cxa_end_catch,
//   __cxa_rethrow, __gxx_personality_v0) and __dynamic_cast: real stack
//   unwinding and an RTTI graph walk are runtime machinery HLE cannot
//   supply from here; a wrong answer corrupts silently.
// * data objects (Need_sceLibc, the _ZTV*/_ZTI* vtables and typeinfo,
//   std::_Fpz, std::_BADOFF, std::_sceLibcClassicLocale, the locale facet
//   `id` objects): not functions — HLE only traps calls.
// * Dinkumware locale/iostream internals (_Locinfo ctor/dtor, std::_Pad,
//   locale::_Getgloballocale/facet::_Register, ios_base and exception
//   family destructors, iostream_category, time_get::get/_Getcat): they
//   operate on C++ object layouts this build cannot verify, so any
//   implementation would be a guess that corrupts or double-frees.
// ---------------------------------------------------------------------------

/// Upper bound on `nmemb * size` for [`hle_qsort`]. A garbage length from a
/// confused guest must fail loudly here, not spin the sort across terabytes
/// of unmapped address space (every element access is still bounds-checked
/// individually; this cap just keeps the failure immediate and attributable).
const QSORT_MAX_TOTAL_BYTES: u64 = 1 << 30; // 1 GiB

/// Upper bound on a single element's `size` for [`hle_qsort`] — bounds the
/// two host-side swap buffers.
const QSORT_MAX_ELEM_BYTES: u64 = 1 << 24; // 16 MiB

/// Why an in-flight qsort had to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QsortAbort {
    /// The synchronous comparator dispatch was refused (test double, direct
    /// gateway). A comparator that *starts* and then faults or unwinds never
    /// produces this — the runtime's recovery machinery abandons the whole
    /// handler instead (see `GuestCallScheduler::call_guest`).
    Call(crate::GuestCallError),
    /// A guest element read/write at this address failed the bounds check.
    Memory(u64),
}

/// `void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const
/// void *, const void *))` — a REAL sort with a REAL guest comparator,
/// dispatched synchronously mid-sort through
/// [`crate::GuestCallScheduler::call_guest`] (checklist item 7's first
/// consumer).
///
/// In-place heapsort over guest memory: every comparator invocation receives
/// genuine pointers **into the live array** (maximum ABI fidelity — a
/// comparator that inspects addresses sees exactly what hardware would show
/// it), swaps move `size`-byte elements through bounds-checked guest
/// reads/writes, and no scratch guest allocation is needed. Heapsort keeps
/// the O(n log n) bound without qsort's recursion (comparator calls are
/// exception round-trips through the runtime, so the comparison count
/// matters) and is trivially abortable mid-flight.
///
/// Failure honesty: a null/refused comparator logs loudly and leaves the
/// array untouched (the refusal is deterministic, so it always precedes the
/// first swap); a mid-sort memory failure logs the address and leaves the
/// array partially heapified — never silently "sorted". A comparator that
/// faults or triggers a fatal unwind never returns control here at all.
fn hle_qsort(ctx: &HleContext, args: &[u64]) -> u64 {
    let base = args[0];
    let nmemb = args[1];
    let size = args[2];
    let compar = args[3];
    if nmemb < 2 || size == 0 {
        return 0; // Nothing to reorder; C says this is a no-op.
    }
    if compar == 0 {
        warn!("qsort(base={base:#x}, nmemb={nmemb}, size={size}): null comparator — untouched");
        return 0;
    }
    let total = nmemb.checked_mul(size);
    if size > QSORT_MAX_ELEM_BYTES
        || total.is_none_or(|t| t > QSORT_MAX_TOTAL_BYTES)
        || base.checked_add(total.unwrap_or(u64::MAX)).is_none()
    {
        warn!("qsort(base={base:#x}, nmemb={nmemb}, size={size}): implausible extent — untouched");
        return 0;
    }
    match qsort_heapsort(ctx, base, nmemb, size, compar) {
        Ok(comparator_calls) => {
            debug!(
                "qsort(base={base:#x}, nmemb={nmemb}, size={size}, compar={compar:#x}): sorted \
                 with {comparator_calls} guest comparator calls"
            );
        }
        Err(QsortAbort::Call(err)) => {
            tracing::error!(
                "qsort(base={base:#x}, nmemb={nmemb}, size={size}): synchronous guest \
                 comparator dispatch refused ({err:?}) — this dispatch path cannot re-enter \
                 guest code; array left unsorted"
            );
        }
        Err(QsortAbort::Memory(addr)) => {
            warn!(
                "qsort(base={base:#x}, nmemb={nmemb}, size={size}): guest element access at \
                 {addr:#x} failed mid-sort — array left partially reordered"
            );
        }
    }
    0
}

/// The heapsort behind [`hle_qsort`]. Returns the number of guest comparator
/// calls made. Ordering follows C `qsort`: the comparator's `int` return
/// (RAX's low 32 bits, sign-interpreted) — negative means `a < b`.
fn qsort_heapsort(
    ctx: &HleContext,
    base: u64,
    nmemb: u64,
    size: u64,
    compar: u64,
) -> Result<u64, QsortAbort> {
    let elem = |i: u64| base + i * size;

    // compare(elements at indices i, j) via the guest comparator.
    let mut comparator_calls = 0u64;
    let mut cmp = |i: u64, j: u64| -> Result<i32, QsortAbort> {
        comparator_calls += 1;
        let raw = ctx
            .guest_calls
            .call_guest(compar, [elem(i), elem(j), 0, 0, 0, 0])
            .map_err(QsortAbort::Call)?;
        // SysV int return: the callee's meaningful result is EAX.
        Ok(raw as u32 as i32)
    };
    let swap = |i: u64, j: u64| -> Result<(), QsortAbort> {
        let mut a = vec![0u8; size as usize];
        let mut b = vec![0u8; size as usize];
        if !ctx.mem.read(elem(i), &mut a) {
            return Err(QsortAbort::Memory(elem(i)));
        }
        if !ctx.mem.read(elem(j), &mut b) {
            return Err(QsortAbort::Memory(elem(j)));
        }
        if !ctx.mem.write(elem(i), &b) {
            return Err(QsortAbort::Memory(elem(i)));
        }
        if !ctx.mem.write(elem(j), &a) {
            return Err(QsortAbort::Memory(elem(j)));
        }
        Ok(())
    };
    // Sift the max-heap property down from `root` over the heap `0..=end`.
    let mut sift_down = |mut root: u64, end: u64| -> Result<(), QsortAbort> {
        loop {
            let child = 2 * root + 1;
            if child > end {
                return Ok(());
            }
            let mut largest = root;
            if cmp(largest, child)? < 0 {
                largest = child;
            }
            if child < end && cmp(largest, child + 1)? < 0 {
                largest = child + 1;
            }
            if largest == root {
                return Ok(());
            }
            swap(root, largest)?;
            root = largest;
        }
    };

    // Build the max-heap, then repeatedly move the maximum to the tail.
    let mut start = (nmemb - 2) / 2;
    loop {
        sift_down(start, nmemb - 1)?;
        if start == 0 {
            break;
        }
        start -= 1;
    }
    let mut end = nmemb - 1;
    while end > 0 {
        swap(0, end)?;
        end -= 1;
        sift_down(0, end)?;
    }
    Ok(comparator_calls)
}

/// Process-scoped key for the per-process tables below: the live kernel's
/// host address. One guest process per host process (the RT0 invariant the
/// rest of this crate relies on) makes it unique and stable for the
/// process's lifetime; table entries self-heal if a fresh process reuses the
/// address (see `ctype_tables`' sentinel check).
fn kernel_key(ctx: &HleContext) -> u64 {
    ctx.kernel as *const raeen_kernel::OrbisKernel as u64
}

/// Requested sizes of live blocks handed out by the `malloc` family, per
/// process — the backing data for `malloc_usable_size`. The guest allocator
/// does not expose block sizes, so the family tracks its own, exactly what a
/// real malloc's chunk headers provide.
static MALLOC_SIZES: std::sync::LazyLock<dashmap::DashMap<(u64, u64), u64>> =
    std::sync::LazyLock::new(dashmap::DashMap::new);

/// Record a live allocation for `malloc_usable_size`. No-op on the null
/// (failed) address.
pub(crate) fn track_alloc(ctx: &HleContext, addr: u64, size: u64) {
    if addr != 0 {
        MALLOC_SIZES.insert((kernel_key(ctx), addr), size);
    }
}

/// Drop a released allocation from the `malloc_usable_size` table.
pub(crate) fn track_free(ctx: &HleContext, addr: u64) {
    if addr != 0 {
        MALLOC_SIZES.remove(&(kernel_key(ctx), addr));
    }
}

/// Real `malloc_usable_size(ptr)`: the usable size of a live malloc-family
/// block. Every block this libc hands out is usable for at least the
/// requested size, so the requested size is a conforming (conservative)
/// answer. A pointer the family never handed out reports 0 with a warning —
/// querying a foreign pointer is undefined in the real API; here it is at
/// least diagnosed.
fn hle_malloc_usable_size(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    let size = MALLOC_SIZES
        .get(&(kernel_key(ctx), ptr))
        .map(|entry| *entry)
        .unwrap_or(0);
    if size == 0 && ptr != 0 {
        warn!("malloc_usable_size({ptr:#x}): not a live malloc-family block");
    }
    size
}

/// Contention cap for the `_Atomic_fetch_*` CAS loop — after this many lost
/// races something pathological is happening; report failure instead of
/// spinning inside an HLE trap forever.
const ATOMIC_FETCH_MAX_RETRIES: u32 = 1024;

/// Shared read-modify-write for the 32-bit `_Atomic_fetch_*` family: a CAS
/// loop until the update sticks, returning the value the guest word held
/// BEFORE the update (the fetch_* contract). An unreadable/unwritable word
/// reports 0 with a warning — the ABI has no error channel.
fn atomic_fetch_op(ctx: &HleContext, ptr: u64, op: impl Fn(u32) -> u32) -> u64 {
    for _ in 0..ATOMIC_FETCH_MAX_RETRIES {
        let Some(old) = ctx.mem.atomic_load_u32(ptr) else {
            warn!("_Atomic_fetch_*: {ptr:#x} is not a readable guest word");
            return 0;
        };
        match ctx.mem.atomic_compare_exchange_u32(ptr, old, op(old)) {
            Some(observed) if observed == old => return u64::from(old),
            Some(_) => continue, // lost a race with another guest thread: retry
            None => {
                warn!("_Atomic_fetch_*: {ptr:#x} is not a writable guest word");
                return 0;
            }
        }
    }
    warn!("_Atomic_fetch_*: {ptr:#x} contended after {ATOMIC_FETCH_MAX_RETRIES} retries");
    0
}

/// Real `_Atomic_fetch_add_4(ptr, value, order)`: Dinkumware's C11-atomics
/// core behind 32-bit `atomic_fetch_add_explicit` — atomically add `value`
/// to the guest u32 at `ptr` and return the OLD value. The `order` argument
/// is accepted and ignored: the guest CPU is the host CPU, so every guest
/// atomic is already a real host atomic of the strongest kind.
fn hle_atomic_fetch_add_4(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    let value = args.get(1).copied().unwrap_or(0) as u32;
    debug!("_Atomic_fetch_add_4(ptr={ptr:#x}, value={value:#x})");
    atomic_fetch_op(ctx, ptr, |old| old.wrapping_add(value))
}

/// Real `_Atomic_fetch_sub_4(ptr, value, order)`: the subtract twin.
fn hle_atomic_fetch_sub_4(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    let value = args.get(1).copied().unwrap_or(0) as u32;
    debug!("_Atomic_fetch_sub_4(ptr={ptr:#x}, value={value:#x})");
    atomic_fetch_op(ctx, ptr, |old| old.wrapping_sub(value))
}

// Dinkumware ctype bitmask layout (the CRT Sony's libc derives from).
// Behavior cross-checked against SharpEmu's LibcStdioExports (GPL-2.0),
// which documents why the table must be Dinkumware-shaped and NOT MSVC/UCRT
// -shaped: a UCRT table makes a title's bundled printf engine misparse
// directives and its preprocessor drop 'a'-'f' from identifiers.
const CTYPE_XD: u16 = 0x001; // _XD: '0'-'9', 'A'-'F', 'a'-'f'
const CTYPE_UP: u16 = 0x002; // _UP: 'A'-'Z'
const CTYPE_SP: u16 = 0x004; // _SP: ' '
const CTYPE_PU: u16 = 0x008; // _PU: punctuation
const CTYPE_LO: u16 = 0x010; // _LO: 'a'-'z'
const CTYPE_DI: u16 = 0x020; // _DI: '0'-'9'
const CTYPE_CN: u16 = 0x040; // _CN: '\t'-'\r' (space-like control)
const CTYPE_BB: u16 = 0x080; // _BB: other control characters
const CTYPE_XB: u16 = 0x400; // _XB: ' ' and '\t' (blank)

/// The tables cover indexes in [-128, 255] so both signed- and unsigned-char
/// indexing work; the pointer handed to the guest addresses the `c == 0`
/// slot, i.e. index -128 lands 128 entries (256 bytes) into the allocation.
const CTYPE_TABLE_LOWER: i32 = -128;
const CTYPE_TABLE_ENTRIES: usize = 384; // -128..=255
const CTYPE_TABLE_BYTES: u64 = (CTYPE_TABLE_ENTRIES * 2) as u64;

/// Dinkumware category bits for one code in the C locale (ASCII).
fn ctype_flags(c: i32) -> u16 {
    if !(0..=0x7F).contains(&c) {
        return 0;
    }
    let upper = (0x41..=0x5A).contains(&c);
    let lower = (0x61..=0x7A).contains(&c);
    let digit = (0x30..=0x39).contains(&c);
    let mut flags = 0;
    if upper {
        flags |= CTYPE_UP;
    }
    if lower {
        flags |= CTYPE_LO;
    }
    if digit {
        flags |= CTYPE_DI;
    }
    if digit || (0x41..=0x46).contains(&c) || (0x61..=0x66).contains(&c) {
        flags |= CTYPE_XD;
    }
    if c == 0x20 {
        flags |= CTYPE_SP | CTYPE_XB;
    }
    if c == 0x09 {
        flags |= CTYPE_XB;
    }
    if (0x09..=0x0D).contains(&c) {
        flags |= CTYPE_CN;
    }
    if (0x20..=0x7E).contains(&c) && !upper && !lower && !digit && c != 0x20 {
        flags |= CTYPE_PU;
    }
    if c <= 0x08 || (0x0E..=0x1F).contains(&c) || c == 0x7F {
        flags |= CTYPE_BB;
    }
    flags
}

/// The tolower/toupper table value for code `c`: identity outside A-Z/a-z;
/// EOF (-1) maps to EOF so `tolower(EOF) == EOF` round-trips; other
/// negative codes (a signed char misused as an index) read 0.
fn ctype_case_map(c: i32, to_lower: bool) -> u16 {
    if c == -1 {
        return 0xFFFF;
    }
    let mapped = match (to_lower, c) {
        (true, 0x41..=0x5A) => c + 0x20,
        (false, 0x61..=0x7A) => c - 0x20,
        _ if c < 0 => 0,
        _ => c,
    };
    (mapped as u32 & 0xFFFF) as u16
}

/// Byte offset, inside a table built by [`ctype_class_table_bytes`], of the
/// `c == 0` entry — i.e. how far into the allocation the pointer a guest
/// indexes with `table[c]` must sit. 128 negative entries × 2 bytes.
pub const CTYPE_TABLE_ZERO_SLOT_OFFSET: u64 = (-CTYPE_TABLE_LOWER) as u64 * 2;

/// The C-locale Dinkumware **classification** table as raw little-endian
/// bytes: 384 `u16` entries covering codes `-128..=255`.
///
/// This is the one generator behind both spellings of the table, so they can
/// never disagree:
///
/// * `_Getpctype()` — the *function* form, which copies these bytes into the
///   guest heap on demand (see [`write_ctype_table`]); and
/// * `_Ctype` — the *data-object* form, which `raeen-firmware` embeds in the
///   HLE data page so a data relocation can point straight at it.
///
/// A guest that reaches the table either way must see identical bytes; a title
/// that resolves `_Ctype` in one translation unit and calls `_Getpctype()` in
/// another would otherwise classify the same character two different ways.
/// Callers publishing this as `_Ctype` must offset the exported symbol address
/// by [`CTYPE_TABLE_ZERO_SLOT_OFFSET`].
pub fn ctype_class_table_bytes() -> Vec<u8> {
    ctype_table_bytes(ctype_flags)
}

/// Serialize one 384-entry `u16` table from `f`, in guest (little-endian)
/// byte order, indexed from [`CTYPE_TABLE_LOWER`].
fn ctype_table_bytes(f: impl Fn(i32) -> u16) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(CTYPE_TABLE_BYTES as usize);
    for i in 0..CTYPE_TABLE_ENTRIES as i32 {
        bytes.extend_from_slice(&f(i + CTYPE_TABLE_LOWER).to_le_bytes());
    }
    bytes
}

/// The three process-global C-locale classification/conversion tables.
#[derive(Clone, Copy, Default)]
struct CtypeTables {
    pctype: u64,
    ptolower: u64,
    ptoupper: u64,
}

/// Per-process cache: the returned pointers are guest addresses, so they
/// would dangle across address spaces.
static CTYPE_TABLES: std::sync::LazyLock<dashmap::DashMap<u64, CtypeTables>> =
    std::sync::LazyLock::new(dashmap::DashMap::new);

/// Allocate one 384-entry u16 guest table from `f` and return the guest
/// address of its c==0 slot (or `None` on allocation/write failure).
fn write_ctype_table(ctx: &HleContext, f: impl Fn(i32) -> u16) -> Option<u64> {
    let bytes = ctype_table_bytes(f);
    let base = ctx.alloc.alloc(CTYPE_TABLE_BYTES, 2)?;
    if !ctx.mem.write(base, &bytes) {
        ctx.alloc.free(base);
        return None;
    }
    Some(base + CTYPE_TABLE_ZERO_SLOT_OFFSET)
}

/// The process's ctype tables, built once and then validated on every call
/// with known-good sentinel entries — a stale cached pointer into a fresh
/// address space (tests, a restarted process at the same kernel host
/// address) reads back zeros and is rebuilt rather than silently serving an
/// all-zero table.
fn ctype_tables(ctx: &HleContext) -> CtypeTables {
    let key = kernel_key(ctx);
    if let Some(tables) = CTYPE_TABLES.get(&key).map(|entry| *entry) {
        let mut upper = [0u8; 2];
        let mut mapped = [0u8; 2];
        if ctx.mem.read(tables.pctype + 0x41 * 2, &mut upper)
            && ctx.mem.read(tables.ptolower + 0x61 * 2, &mut mapped)
            && u16::from_le_bytes(upper) == CTYPE_UP | CTYPE_XD
            && u16::from_le_bytes(mapped) == 0x61
        {
            return tables;
        }
    }
    let tables = CtypeTables {
        pctype: write_ctype_table(ctx, ctype_flags).unwrap_or(0),
        ptolower: write_ctype_table(ctx, |c| ctype_case_map(c, true)).unwrap_or(0),
        ptoupper: write_ctype_table(ctx, |c| ctype_case_map(c, false)).unwrap_or(0),
    };
    CTYPE_TABLES.insert(key, tables);
    tables
}

/// Real `_Getpctype()`: Dinkumware's ctype-table accessor — returns a guest
/// pointer to the c==0 slot of the C-locale category table (indexable over
/// [-128, 255]). `0` only if the guest arena is exhausted.
fn hle_getpctype(ctx: &HleContext, _args: &[u64]) -> u64 {
    let tables = ctype_tables(ctx);
    if tables.pctype == 0 {
        warn!("_Getpctype: guest arena exhausted for the ctype table");
    }
    tables.pctype
}

/// Real `_Getptolower()`: the tolower conversion table, same indexing.
fn hle_getptolower(ctx: &HleContext, _args: &[u64]) -> u64 {
    let tables = ctype_tables(ctx);
    if tables.ptolower == 0 {
        warn!("_Getptolower: guest arena exhausted for the tolower table");
    }
    tables.ptolower
}

/// Real `_Getptoupper()`: the toupper conversion table, same indexing.
fn hle_getptoupper(ctx: &HleContext, _args: &[u64]) -> u64 {
    let tables = ctype_tables(ctx);
    if tables.ptoupper == 0 {
        warn!("_Getptoupper: guest arena exhausted for the toupper table");
    }
    tables.ptoupper
}

/// A host recursive mutex for the Dinkumware lock plumbing (`_Locksyslock`,
/// `_Mtx_lock`). Guest threads are host threads under Raeen's native
/// runtime, so real blocking mutual exclusion here IS real mutual exclusion
/// between guest threads. Ownership is tracked by guest thread id so
/// recursive locks re-enter correctly and a foreign unlock is an error
/// rather than a silent pass.
struct HostRecursiveLock {
    state: std::sync::Mutex<(u64, u32)>,
    condvar: std::sync::Condvar,
}

impl HostRecursiveLock {
    const fn new() -> Self {
        Self {
            state: std::sync::Mutex::new((0, 0)),
            condvar: std::sync::Condvar::new(),
        }
    }

    /// Lock, blocking until free. With `recursive`, the owning thread
    /// re-enters (its unlock count grows) instead of deadlocking.
    fn lock(&self, tid: u64, recursive: bool) {
        let mut guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            let (owner, count) = *guard;
            if owner == 0 {
                *guard = (tid, 1);
                return;
            }
            if recursive && owner == tid {
                *guard = (tid, count.saturating_add(1));
                return;
            }
            guard = self.condvar.wait(guard).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Unlock one level. Returns `false` when the caller does not hold the
    /// lock — real unlock of an unheld mutex is undefined; reporting the
    /// misuse is the honest failure mode.
    fn unlock(&self, tid: u64) -> bool {
        let mut guard = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let (owner, count) = *guard;
        if owner != tid {
            return false;
        }
        if count <= 1 {
            *guard = (0, 0);
            self.condvar.notify_all();
        } else {
            *guard = (owner, count - 1);
        }
        true
    }
}

/// Dinkumware's `_Locksyslock(i)`/`_Unlocksyslock(i)` take one of a small
/// fixed set of process locks (stream, locale, malloc, debug, ...). Eight
/// slots cover every index their headers assign; an out-of-range index is a
/// guest bug, logged and ignored rather than corrupting silently.
const SYSLOCK_COUNT: usize = 8;

/// The syslocks are recursive in Dinkumware (its stdio locks a stream the
/// caller may already hold).
static SYSLOCKS: [HostRecursiveLock; SYSLOCK_COUNT] =
    [const { HostRecursiveLock::new() }; SYSLOCK_COUNT];

/// Real `_Locksyslock(index)`: take process syslock `index` (recursively).
fn hle_locksyslock(ctx: &HleContext, args: &[u64]) -> u64 {
    let index = args.first().copied().unwrap_or(0);
    debug!("_Locksyslock(index={index})");
    let Some(lock) = SYSLOCKS.get(usize::try_from(index).unwrap_or(usize::MAX)) else {
        warn!("_Locksyslock: index {index} out of range (have {SYSLOCK_COUNT})");
        return 0;
    };
    lock.lock(ctx.guest_threads.current_thread(), true);
    0
}

/// Real `_Unlocksyslock(index)`: release one level of syslock `index`.
fn hle_unlocksyslock(ctx: &HleContext, args: &[u64]) -> u64 {
    let index = args.first().copied().unwrap_or(0);
    debug!("_Unlocksyslock(index={index})");
    let Some(lock) = SYSLOCKS.get(usize::try_from(index).unwrap_or(usize::MAX)) else {
        warn!("_Unlocksyslock: index {index} out of range (have {SYSLOCK_COUNT})");
        return 0;
    };
    if !lock.unlock(ctx.guest_threads.current_thread()) {
        warn!("_Unlocksyslock({index}): not held by this thread");
    }
    0
}

/// The recursive flag in `_Mtx_init`'s internal type encoding (Dinkumware's
/// `mtx_init` maps the C11 `mtx_recursive` onto it).
const MTX_TYPE_RECURSIVE: u64 = 0x100;

/// One live Dinkumware mutex: the host lock plus whether `_Mtx_init` asked
/// for recursive semantics.
struct MtxState {
    recursive: bool,
    lock: HostRecursiveLock,
}

/// Live Dinkumware mutexes, keyed by (process, guest `mtx_t` address). C11's
/// `mtx_t` is opaque to the program — it only ever reaches HLE as a pointer
/// — so the address alone identifies the mutex and no storage layout has to
/// be assumed (or written into guest memory at all).
static MTX_LOCKS: std::sync::LazyLock<dashmap::DashMap<(u64, u64), Arc<MtxState>>> =
    std::sync::LazyLock::new(dashmap::DashMap::new);

/// Look up a live mutex without holding the map shard across a blocking
/// lock operation.
fn mtx_lookup(ctx: &HleContext, mtx: u64) -> Option<Arc<MtxState>> {
    MTX_LOCKS
        .get(&(kernel_key(ctx), mtx))
        .map(|entry| Arc::clone(&entry))
}

/// Real `_Mtx_init(mtx, type)`: create a mutex at address `mtx`. `type`
/// carries Dinkumware's internal flags (`0x100` = recursive). Returns 0.
fn hle_mtx_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let mtx = args.first().copied().unwrap_or(0);
    let kind = args.get(1).copied().unwrap_or(0);
    debug!("_Mtx_init(mtx={mtx:#x}, type={kind:#x})");
    MTX_LOCKS.insert(
        (kernel_key(ctx), mtx),
        Arc::new(MtxState {
            recursive: kind & MTX_TYPE_RECURSIVE != 0,
            lock: HostRecursiveLock::new(),
        }),
    );
    0
}

/// Real `_Mtx_lock(mtx)`: lock, blocking until held. A mutex `_Mtx_init`
/// never created reports 1 (Dinkumware's error code) with a warning — real
/// use of an uninitialized mutex is undefined.
fn hle_mtx_lock(ctx: &HleContext, args: &[u64]) -> u64 {
    let mtx = args.first().copied().unwrap_or(0);
    debug!("_Mtx_lock(mtx={mtx:#x})");
    let Some(state) = mtx_lookup(ctx, mtx) else {
        warn!("_Mtx_lock({mtx:#x}): mutex was never _Mtx_init'd");
        return 1;
    };
    state
        .lock
        .lock(ctx.guest_threads.current_thread(), state.recursive);
    0
}

/// Real `_Mtx_unlock(mtx)`: release one level; 1 on an unknown mutex or a
/// foreign unlock.
fn hle_mtx_unlock(ctx: &HleContext, args: &[u64]) -> u64 {
    let mtx = args.first().copied().unwrap_or(0);
    debug!("_Mtx_unlock(mtx={mtx:#x})");
    let Some(state) = mtx_lookup(ctx, mtx) else {
        warn!("_Mtx_unlock({mtx:#x}): mutex was never _Mtx_init'd");
        return 1;
    };
    if !state.lock.unlock(ctx.guest_threads.current_thread()) {
        warn!("_Mtx_unlock({mtx:#x}): not held by this thread");
        return 1;
    }
    0
}

/// Real `_Mtx_destroy(mtx)`: retire the mutex. Destroying a locked mutex is
/// the guest's bug (undefined in C); the entry is simply dropped.
fn hle_mtx_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let mtx = args.first().copied().unwrap_or(0);
    debug!("_Mtx_destroy(mtx={mtx:#x})");
    MTX_LOCKS.remove(&(kernel_key(ctx), mtx));
    0
}

/// Real `_Thrd_id()`: Dinkumware's current-thread identity behind
/// `thrd_current`/`thrd_equal` — the guest thread handle, unique per thread.
fn hle_thrd_id(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.guest_threads.current_thread()
}

/// Real `_Thrd_join(thrd, code)`: join a C11 thread. Dinkumware's POSIX
/// xthreads typedef `_Thrd_t` to `pthread_t`, so the by-value argument IS a
/// scePthread handle and the kernel's join machinery applies directly. One
/// ABI shim: the scheduler writes an 8-byte `void*` retval, but `code` is a
/// 4-byte `int*` — stage through an 8-byte guest scratch and copy the low
/// half so the write cannot overrun the caller's int.
fn hle_thrd_join(ctx: &HleContext, args: &[u64]) -> u64 {
    let thrd = args.first().copied().unwrap_or(0);
    let code = args.get(1).copied().unwrap_or(0);
    debug!("_Thrd_join(thrd={thrd:#x}, code={code:#x})");
    let report = |rc: u64| {
        if rc == 0 {
            0
        } else {
            warn!("_Thrd_join({thrd:#x}): join failed ({rc:#x})");
            1
        }
    };
    if code == 0 {
        return report(ctx.guest_threads.join(thrd, 0));
    }
    let Some(scratch) = ctx.alloc.alloc(8, 8) else {
        warn!("_Thrd_join: no guest memory for the retval scratch");
        return 1;
    };
    let rc = ctx.guest_threads.join(thrd, scratch);
    if rc != 0 {
        ctx.alloc.free(scratch);
        return report(rc);
    }
    let mut value = [0u8; 8];
    let ok = ctx.mem.read(scratch, &mut value) && ctx.mem.write(code, &value[..4]);
    ctx.alloc.free(scratch);
    if !ok {
        warn!("_Thrd_join: failed to deliver the thread return code to {code:#x}");
        return 1;
    }
    0
}

/// Real `_Thrd_sleep(timespec*)`: block the calling (host) thread for the
/// requested duration — guest threads are host threads, so a host sleep IS
/// a guest sleep. Returns 0; -1 when the timespec is unreadable.
fn hle_thrd_sleep(ctx: &HleContext, args: &[u64]) -> u64 {
    let ts = args.first().copied().unwrap_or(0);
    debug!("_Thrd_sleep(ts={ts:#x})");
    let mut raw = [0u8; 16];
    if ts == 0 || !ctx.mem.read(ts, &mut raw) {
        warn!("_Thrd_sleep: unreadable timespec at {ts:#x}");
        return u64::MAX; // -1
    }
    let sec = i64::from_le_bytes(raw[..8].try_into().unwrap_or([0; 8]));
    let nsec = i64::from_le_bytes(raw[8..].try_into().unwrap_or([0; 8]));
    let sec = sec.max(0) as u64;
    let nsec = nsec.clamp(0, 999_999_999) as u32;
    std::thread::sleep(Duration::new(sec, nsec));
    0
}

/// Real `_Xtime_get_ticks()`: Dinkumware's monotonic tick source behind
/// `xtime_get`/`timespec_get`/`std::chrono::steady_clock`, in 100 ns units
/// (the 10 MHz tick every Dinkumware port uses). Anchored to the host wall
/// clock at first call and advanced strictly monotonically (Instant)
/// afterwards, so both steady durations and UTC-ish conversions behave —
/// the anchor is documented because a real RTC can jump where this cannot.
fn hle_xtime_get_ticks(_ctx: &HleContext, _args: &[u64]) -> u64 {
    static ORIGIN: std::sync::OnceLock<(Instant, i64)> = std::sync::OnceLock::new();
    let (start, wall_ticks) = ORIGIN.get_or_init(|| {
        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| (d.as_nanos() / 100) as i64)
            .unwrap_or(0);
        (Instant::now(), wall)
    });
    let elapsed = (start.elapsed().as_nanos() / 100) as i64;
    (wall_ticks + elapsed) as u64
}

/// Magic tag at the head of every FILE object Raeen hands out, so the stdio
/// family recognizes its own streams — and, just as importantly, REJECTS a
/// foreign `FILE*` (a title that conjured one from its own bundled libc)
/// instead of dereferencing it blindly.
const STDIO_FILE_MAGIC: u64 = 0x5241_4545_4E5F_4649; // "RAEEN_FI"

/// FILE object: magic, then the stream fd (1 = stdout, 2 = stderr).
const STDIO_FILE_BYTES: u64 = 16;

/// Allocate one std stream object. Returns 0 (NULL) on arena exhaustion.
fn new_stdio_file(ctx: &HleContext, fd: u64) -> u64 {
    let Some(addr) = ctx.alloc.alloc(STDIO_FILE_BYTES, 8) else {
        return 0;
    };
    if !ctx.mem.write(addr, &STDIO_FILE_MAGIC.to_le_bytes())
        || !ctx.mem.write(addr + 8, &fd.to_le_bytes())
    {
        ctx.alloc.free(addr);
        return 0;
    }
    addr
}

/// The fd of a Raeen stream object, or `None` when `stream` is not one of
/// ours.
fn stdio_file_fd(ctx: &HleContext, stream: u64) -> Option<u64> {
    let mut head = [0u8; STDIO_FILE_BYTES as usize];
    if stream == 0 || !ctx.mem.read(stream, &mut head) {
        return None;
    }
    if u64::from_le_bytes(head[..8].try_into().ok()?) != STDIO_FILE_MAGIC {
        return None;
    }
    Some(u64::from_le_bytes(head[8..].try_into().ok()?))
}

/// Route bytes to a stream's destination. Raeen's std streams are
/// unbuffered and share the kernel console (stdout and stderr both reach
/// the user-visible log, as on a real terminal; the fd is kept for
/// diagnostics). Returns `false` for a foreign FILE*.
fn stdio_write(ctx: &HleContext, stream: u64, bytes: &[u8]) -> bool {
    let Some(fd) = stdio_file_fd(ctx, stream) else {
        warn!("stdio write: {stream:#x} is not a Raeen FILE object");
        return false;
    };
    if fd == 2 {
        debug!("stdio: {} byte(s) to stderr", bytes.len());
    }
    ctx.kernel.console.write_bytes(bytes);
    true
}

/// Real `_Stdout()` / `_Stderr()`: Dinkumware's stream accessors, returning
/// the process's stdout/stderr FILE*. A fresh object per call (16 bytes;
/// titles cache the result) keeps every returned pointer valid for the
/// process's lifetime without caching guest addresses across processes.
fn hle_stdout(ctx: &HleContext, _args: &[u64]) -> u64 {
    new_stdio_file(ctx, 1)
}

/// See [`hle_stdout`].
fn hle_stderr(ctx: &HleContext, _args: &[u64]) -> u64 {
    new_stdio_file(ctx, 2)
}

/// Real `fflush(stream)`: a no-op success because Raeen's streams are
/// unbuffered — there is never anything buffered to flush. `fflush(NULL)`
/// (flush all) succeeds the same way. A foreign FILE* reports EOF.
fn hle_fflush(ctx: &HleContext, args: &[u64]) -> u64 {
    let stream = args.first().copied().unwrap_or(0);
    debug!("fflush(stream={stream:#x})");
    if stream == 0 {
        return 0;
    }
    if stdio_file_fd(ctx, stream).is_none() {
        warn!("fflush: {stream:#x} is not a Raeen FILE object");
        return u64::MAX; // EOF
    }
    0
}

/// Real `fputc(c, stream)`: write one byte; return it (as the unsigned
/// char) or EOF.
fn hle_fputc(ctx: &HleContext, args: &[u64]) -> u64 {
    let c = (args.first().copied().unwrap_or(0) & 0xFF) as u8;
    let stream = args.get(1).copied().unwrap_or(0);
    debug!("fputc(c={c:#x}, stream={stream:#x})");
    if !stdio_write(ctx, stream, &[c]) {
        return u64::MAX; // EOF
    }
    u64::from(c)
}

/// Real `fputs(s, stream)`: write the string WITHOUT a newline; return a
/// non-negative value on success, EOF on an unreadable string or a foreign
/// stream.
fn hle_fputs(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    let stream = args.get(1).copied().unwrap_or(0);
    debug!("fputs(s={s:#x}, stream={stream:#x})");
    let Some(bytes) = crate::fmt::read_cstr(ctx.mem, s) else {
        warn!("fputs: unreadable string at {s:#x}");
        return u64::MAX; // EOF
    };
    if !stdio_write(ctx, stream, &bytes) {
        return u64::MAX; // EOF
    }
    1
}

/// Real `fwrite(ptr, size, nmemb, stream)`: write `size * nmemb` bytes in
/// bounded chunks, returning the number of complete ITEMS written (short on
/// failure, like the real API).
fn hle_fwrite(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    let nmemb = args.get(2).copied().unwrap_or(0);
    let stream = args.get(3).copied().unwrap_or(0);
    debug!("fwrite(ptr={ptr:#x}, size={size:#x}, nmemb={nmemb:#x}, stream={stream:#x})");
    let Some(total) = size.checked_mul(nmemb) else {
        warn!("fwrite: size*nmemb overflowed");
        return 0;
    };
    if total == 0 {
        return 0;
    }
    let mut buf = [0u8; MEM_OP_CHUNK];
    let mut done = 0u64;
    while done < total {
        let chunk = (total - done).min(MEM_OP_CHUNK as u64) as usize;
        let Some(from) = ptr.checked_add(done) else {
            return done / size.max(1);
        };
        if !ctx.mem.read(from, &mut buf[..chunk]) || !stdio_write(ctx, stream, &buf[..chunk]) {
            warn!("fwrite: stopped after {done}/{total} bytes");
            return done / size.max(1);
        }
        done += chunk as u64;
    }
    nmemb
}

/// Real `vfprintf(stream, fmt, ap)`: format against the guest `va_list` and
/// write to the stream. Returns the byte count, or a negative value (EOF)
/// on an unreadable format/list or a foreign stream.
fn hle_vfprintf(ctx: &HleContext, args: &[u64]) -> u64 {
    let stream = args.first().copied().unwrap_or(0);
    let fmt_ptr = args.get(1).copied().unwrap_or(0);
    let ap = args.get(2).copied().unwrap_or(0);
    debug!("vfprintf(stream={stream:#x}, fmt={fmt_ptr:#x}, ap={ap:#x})");
    if stdio_file_fd(ctx, stream).is_none() {
        warn!("vfprintf: {stream:#x} is not a Raeen FILE object");
        return u64::MAX;
    }
    let Some(fmt) = crate::fmt::read_cstr(ctx.mem, fmt_ptr) else {
        warn!("vfprintf: unreadable format string at {fmt_ptr:#x}");
        return u64::MAX;
    };
    let Some(mut varargs) = GuestVaList::read(ctx.mem, ap) else {
        warn!("vfprintf: unreadable va_list at {ap:#x}");
        return u64::MAX;
    };
    let mut float_list = varargs;
    let mut floats = GuestVaListFloats(&mut float_list);
    let formatted = crate::fmt::format_c(&fmt, &mut varargs, &mut floats, ctx.mem);
    ctx.kernel.console.write_bytes(&formatted);
    formatted.len() as u64
}

/// Real `sprintf(buf, fmt, ...)`: format into the guest buffer (unbounded,
/// like the real API — the caller owns the overflow risk), NUL-terminated.
/// Returns the byte count excluding the NUL; a negative value (EOF) on an
/// unreadable format or failed write.
fn hle_sprintf(ctx: &HleContext, args: &[u64]) -> u64 {
    let buf = args.first().copied().unwrap_or(0);
    let fmt_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sprintf(buf={buf:#x}, fmt={fmt_ptr:#x})");
    let Some(fmt) = crate::fmt::read_cstr(ctx.mem, fmt_ptr) else {
        warn!("sprintf: unreadable format string at {fmt_ptr:#x}");
        return u64::MAX;
    };
    let mut varargs = args.iter().skip(2).copied();
    let mut floats = ctx.float_args.iter().map(|bits| f64::from_bits(*bits));
    let mut formatted = crate::fmt::format_c(&fmt, &mut varargs, &mut floats, ctx.mem);
    let len = formatted.len() as u64;
    formatted.push(0);
    if !ctx.mem.write(buf, &formatted) {
        warn!(
            "sprintf: failed to write {} bytes to buf={buf:#x}",
            formatted.len()
        );
        return u64::MAX;
    }
    len
}

/// Real `sprintf_s(buf, n, fmt, ...)` with C11 Annex K semantics: on a
/// runtime-constraint violation (null buffer, zero size, or output that
/// would not fit) store `'\0'` in `buf[0]` when possible and return a
/// negative value; otherwise behave like `sprintf` (count excluding NUL).
fn hle_sprintf_s(ctx: &HleContext, args: &[u64]) -> u64 {
    let buf = args.first().copied().unwrap_or(0);
    let n = args.get(1).copied().unwrap_or(0);
    let fmt_ptr = args.get(2).copied().unwrap_or(0);
    debug!("sprintf_s(buf={buf:#x}, n={n:#x}, fmt={fmt_ptr:#x})");
    let fail = |ctx: &HleContext| {
        if n > 0 && buf != 0 {
            let _ = ctx.mem.write(buf, &[0u8]);
        }
        u64::MAX
    };
    if buf == 0 || n == 0 {
        warn!("sprintf_s: constraint violation (buf={buf:#x}, n={n:#x})");
        return u64::MAX;
    }
    let Some(fmt) = crate::fmt::read_cstr(ctx.mem, fmt_ptr) else {
        warn!("sprintf_s: unreadable format string at {fmt_ptr:#x}");
        return fail(ctx);
    };
    let mut varargs = args.iter().skip(3).copied();
    let mut floats = ctx.float_args.iter().map(|bits| f64::from_bits(*bits));
    let formatted = crate::fmt::format_c(&fmt, &mut varargs, &mut floats, ctx.mem);
    if formatted.len() as u64 >= n {
        warn!(
            "sprintf_s: output ({} bytes) does not fit n={n:#x}",
            formatted.len()
        );
        return fail(ctx);
    }
    let mut out = formatted;
    let len = out.len() as u64;
    out.push(0);
    if !ctx.mem.write(buf, &out) {
        warn!(
            "sprintf_s: failed to write {} bytes to buf={buf:#x}",
            out.len()
        );
        return fail(ctx);
    }
    len
}

/// Read a NUL-terminated guest wide string (`wchar_t` is 32-bit on this
/// ABI), bounded by [`STRLEN_MAX_SCAN`] units like the narrow readers.
/// `None` only when the very first unit is unreadable.
fn read_wide_cstr(mem: &dyn crate::GuestMemory, addr: u64) -> Option<Vec<u32>> {
    let mut out = Vec::new();
    let mut unit = [0u8; 4];
    let mut off = 0u64;
    loop {
        let cur = addr.checked_add(off.checked_mul(4)?)?;
        if !mem.read(cur, &mut unit) {
            if off == 0 {
                return None;
            }
            warn!(
                "read_wide_cstr: string at {addr:#x} ran out of readable memory after {off} units"
            );
            break;
        }
        let u = u32::from_le_bytes(unit);
        if u == 0 {
            break;
        }
        out.push(u);
        off += 1;
        if off >= STRLEN_MAX_SCAN {
            warn!("read_wide_cstr: string at {addr:#x} unterminated; truncating");
            break;
        }
    }
    Some(out)
}

/// Append `units` to `out` padded to `width` with spaces (or zeros for the
/// zero-pad flag, keeping a leading '-' first) — the wide twin of
/// `fmt::pad`.
fn pad_wide(out: &mut Vec<u32>, units: &[u32], width: usize, left_align: bool, zero_pad: bool) {
    let pad_n = width.saturating_sub(units.len());
    if pad_n == 0 {
        out.extend_from_slice(units);
        return;
    }
    if left_align {
        out.extend_from_slice(units);
        out.extend(std::iter::repeat_n(0x20, pad_n));
    } else if zero_pad {
        if let Some((&first, rest)) = units.split_first()
            && first == u32::from(b'-')
        {
            out.push(first);
            out.extend(std::iter::repeat_n(0x30, pad_n));
            out.extend_from_slice(rest);
            return;
        }
        out.extend(std::iter::repeat_n(0x30, pad_n));
        out.extend_from_slice(units);
    } else {
        out.extend(std::iter::repeat_n(0x20, pad_n));
        out.extend_from_slice(units);
    }
}

/// The wide twin of `format_c` for `vswprintf`: the same directive grammar
/// over 32-bit wchar units, with the C wide-printf rule that `%s`/`%c`
/// take WIDE arguments (narrow needs the `h` length: `%hs`/`%hc`). The
/// narrow numeric conversions (`%d`/`%u`/`%x`/`%p`/`%f`/...) delegate to
/// `format_c` through the original directive text so flag/width/precision/
/// length behavior stays identical, then widen into the output. `%o` and
/// the wide `%s`/`%c` are rendered here. Anything else is emitted verbatim
/// with a warning, matching `format_c`'s contract.
fn format_wide(
    fmt: &[u32],
    args: &mut dyn Iterator<Item = u64>,
    floats: &mut dyn Iterator<Item = f64>,
    mem: &dyn crate::GuestMemory,
) -> Vec<u32> {
    /// How a length modifier reshapes an integer argument (mirrors
    /// `fmt.rs`'s private `Length`, needed for the locally-rendered `%o`).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum WideLen {
        Int,
        Char,
        Short,
        Long,
    }

    let mut out = Vec::with_capacity(fmt.len());
    let mut i = 0usize;
    while i < fmt.len() {
        if fmt[i] != u32::from(b'%') {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        let spec_start = i;
        i += 1;

        let mut zero_pad = false;
        let mut left_align = false;
        while i < fmt.len() {
            match fmt[i] {
                u if u == u32::from(b'0') => zero_pad = true,
                u if u == u32::from(b'-') => left_align = true,
                u if u == u32::from(b'+') || u == u32::from(b' ') || u == u32::from(b'#') => {}
                _ => break,
            }
            i += 1;
        }
        let mut width = 0usize;
        while i < fmt.len() {
            let u = fmt[i];
            if !(u32::from(b'0')..=u32::from(b'9')).contains(&u) {
                break;
            }
            width = width
                .saturating_mul(10)
                .saturating_add((u - u32::from(b'0')) as usize);
            i += 1;
        }
        let mut precision: Option<usize> = None;
        if i < fmt.len() && fmt[i] == u32::from(b'.') {
            i += 1;
            let mut p = 0usize;
            while i < fmt.len() {
                let u = fmt[i];
                if !(u32::from(b'0')..=u32::from(b'9')).contains(&u) {
                    break;
                }
                p = p
                    .saturating_mul(10)
                    .saturating_add((u - u32::from(b'0')) as usize);
                i += 1;
            }
            precision = Some(p);
        }
        let mut length = WideLen::Int;
        let mut narrow = false;
        while i < fmt.len() {
            match fmt[i] {
                u if u == u32::from(b'l')
                    || u == u32::from(b'z')
                    || u == u32::from(b't')
                    || u == u32::from(b'j') =>
                {
                    length = WideLen::Long;
                }
                u if u == u32::from(b'h') => {
                    narrow = true;
                    length = if length == WideLen::Short {
                        WideLen::Char
                    } else {
                        WideLen::Short
                    };
                }
                _ => break,
            }
            i += 1;
        }
        let Some(&conv_u) = fmt.get(i) else {
            out.extend_from_slice(&fmt[spec_start..]);
            break;
        };
        i += 1;
        let Ok(conv) = u8::try_from(conv_u) else {
            warn!("vswprintf: non-ASCII conversion emitted verbatim");
            out.extend_from_slice(&fmt[spec_start..i]);
            continue;
        };

        match conv {
            b'%' => out.push(u32::from(b'%')),
            b'c' => {
                let v = args.next().unwrap_or(0);
                let unit = if narrow { (v & 0xFF) as u32 } else { v as u32 };
                pad_wide(&mut out, &[unit], width, left_align, false);
            }
            b's' => {
                let ptr = args.next().unwrap_or(0);
                let mut units: Vec<u32> = if narrow {
                    match crate::fmt::read_cstr(mem, ptr) {
                        Some(bytes) => bytes.into_iter().map(u32::from).collect(),
                        None => {
                            warn!("vswprintf %%hs: unreadable guest string pointer {ptr:#x}");
                            format!("<bad ptr {ptr:#x}>")
                                .bytes()
                                .map(u32::from)
                                .collect()
                        }
                    }
                } else {
                    match read_wide_cstr(mem, ptr) {
                        Some(units) => units,
                        None => {
                            warn!("vswprintf %%s: unreadable guest wide string pointer {ptr:#x}");
                            format!("<bad ptr {ptr:#x}>")
                                .bytes()
                                .map(u32::from)
                                .collect()
                        }
                    }
                };
                if let Some(p) = precision {
                    units.truncate(p);
                }
                pad_wide(&mut out, &units, width, left_align, false);
            }
            b'o' => {
                let raw = args.next().unwrap_or(0);
                let v: u64 = match length {
                    WideLen::Int => raw as u32 as u64,
                    WideLen::Char => raw as u8 as u64,
                    WideLen::Short => raw as u16 as u64,
                    WideLen::Long => raw,
                };
                let rendered: Vec<u32> = format!("{v:o}").bytes().map(u32::from).collect();
                pad_wide(&mut out, &rendered, width, left_align, zero_pad);
            }
            b'd' | b'i' | b'u' | b'x' | b'X' | b'p' | b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
                // Delegate to the narrow engine with the exact original
                // directive text (every unit so far is ASCII by construction).
                let spec: Vec<u8> = fmt[spec_start..i]
                    .iter()
                    .map(|&u| u8::try_from(u).unwrap_or(b'?'))
                    .collect();
                let rendered = if matches!(conv, b'f' | b'F' | b'e' | b'E' | b'g' | b'G') {
                    let mut no_args = std::iter::empty();
                    let mut one_float = std::iter::once(floats.next().unwrap_or(0.0));
                    crate::fmt::format_c(&spec, &mut no_args, &mut one_float, mem)
                } else {
                    let mut one_arg = std::iter::once(args.next().unwrap_or(0));
                    let mut no_floats = std::iter::empty();
                    crate::fmt::format_c(&spec, &mut one_arg, &mut no_floats, mem)
                };
                out.extend(rendered.into_iter().map(u32::from));
            }
            other => {
                warn!(
                    "vswprintf: unsupported conversion '%{}' emitted verbatim",
                    (other as char).escape_default()
                );
                out.extend_from_slice(&fmt[spec_start..i]);
            }
        }
    }
    out
}

/// Write wide `units` (plus a terminating NUL unit when `terminate`) to the
/// guest buffer.
fn write_wide_units(ctx: &HleContext, buf: u64, units: &[u32], terminate: bool) -> bool {
    let mut bytes = Vec::with_capacity(units.len() * 4 + 4);
    for &u in units {
        bytes.extend_from_slice(&u.to_le_bytes());
    }
    if terminate {
        bytes.extend_from_slice(&[0; 4]);
    }
    ctx.mem.write(buf, &bytes)
}

/// Real `vswprintf(buf, n, fmt, ap)`: the wide `vsnprintf`. C's `swprintf`
/// contract (unlike `vsnprintf`): output that would reach `n` units is a
/// failure — write the fitting prefix (n-1 units + NUL) and return a
/// negative value; otherwise return the unit count excluding the NUL.
fn hle_vswprintf(ctx: &HleContext, args: &[u64]) -> u64 {
    let buf = args.first().copied().unwrap_or(0);
    let n = args.get(1).copied().unwrap_or(0);
    let fmt_ptr = args.get(2).copied().unwrap_or(0);
    let ap = args.get(3).copied().unwrap_or(0);
    debug!("vswprintf(buf={buf:#x}, n={n:#x}, fmt={fmt_ptr:#x}, ap={ap:#x})");
    let Some(fmt) = read_wide_cstr(ctx.mem, fmt_ptr) else {
        warn!("vswprintf: unreadable wide format string at {fmt_ptr:#x}");
        return u64::MAX;
    };
    let Some(mut varargs) = GuestVaList::read(ctx.mem, ap) else {
        warn!("vswprintf: unreadable va_list at {ap:#x}");
        return u64::MAX;
    };
    let mut float_list = varargs;
    let mut floats = GuestVaListFloats(&mut float_list);
    let formatted = format_wide(&fmt, &mut varargs, &mut floats, ctx.mem);
    let full_len = formatted.len() as u64;
    if full_len >= n {
        if n > 0 && buf != 0 {
            let keep = usize::try_from(n - 1)
                .unwrap_or(usize::MAX)
                .min(formatted.len());
            if !write_wide_units(ctx, buf, &formatted[..keep], true) {
                warn!("vswprintf: failed to write truncated output to buf={buf:#x}");
            }
        }
        return u64::MAX; // -1: output would not fit
    }
    if !write_wide_units(ctx, buf, &formatted, true) {
        warn!(
            "vswprintf: failed to write {} units to buf={buf:#x}",
            formatted.len()
        );
        return u64::MAX;
    }
    full_len
}

/// How a scanf length modifier sizes the STORE through the argument
/// pointer — the write-width half of the scanf contract (wrong width =
/// corrupted guest stack).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScanLen {
    /// No modifier: `int` / `float` (32-bit store).
    Int,
    /// `hh`: `signed char` (8-bit store).
    Char,
    /// `h`: `short` (16-bit store).
    Short,
    /// `l`/`ll`/`z`/`t`/`j`: 64-bit store (`long`, `double`).
    Long,
}

/// Store one scanned integer through a guest pointer, truncated to the
/// width the length modifier promises.
fn store_scan_int(mem: &dyn crate::GuestMemory, ptr: u64, value: i128, len: ScanLen) -> bool {
    match len {
        ScanLen::Int => mem.write(ptr, &(value as i32).to_le_bytes()),
        ScanLen::Char => mem.write(ptr, &(value as i8).to_le_bytes()),
        ScanLen::Short => mem.write(ptr, &(value as i16).to_le_bytes()),
        ScanLen::Long => mem.write(ptr, &(value as i64).to_le_bytes()),
    }
}

/// Parse a C `strtod`-style floating literal at the start of `s`:
/// whitespace, sign, then `inf`/`infinity`/`nan(...)` (case-insensitive), a
/// hex float (`0x1.8p3`), or a decimal with optional fraction/exponent.
/// Returns `(value, bytes_consumed_including_leading_whitespace)`; a
/// zero `consumed` means nothing converted. Rust's own float parser handles
/// the decimal body (correctly rounded); the hex form is accumulated
/// manually.
fn parse_c_float(s: &[u8]) -> (f64, usize) {
    let mut i = 0usize;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    let num_start = i;
    let mut negative = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        negative = s[i] == b'-';
        i += 1;
    }
    let rest = &s[i..];
    if rest.len() >= 3 && rest[..3].eq_ignore_ascii_case(b"inf") {
        let extra = if rest.len() >= 8 && rest[3..8].eq_ignore_ascii_case(b"inity") {
            8
        } else {
            3
        };
        return (
            if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            },
            i + extra,
        );
    }
    if rest.len() >= 3 && rest[..3].eq_ignore_ascii_case(b"nan") {
        let mut end = i + 3;
        if s.get(end) == Some(&b'(') {
            let mut j = end + 1;
            while j < s.len() && (s[j].is_ascii_alphanumeric() || s[j] == b'_') {
                j += 1;
            }
            if s.get(j) == Some(&b')') {
                end = j + 1;
            }
        }
        return (f64::NAN, end);
    }
    // Hex float: 0x H[.H][p[+-]D].
    if rest.len() >= 2 && rest[0] == b'0' && rest[1] | 0x20 == b'x' {
        let mut j = i + 2;
        let mut value = 0.0f64;
        let mut any = false;
        while j < s.len() && s[j].is_ascii_hexdigit() {
            let digit = (s[j] as char).to_digit(16).unwrap_or(0);
            value = value * 16.0 + f64::from(digit);
            j += 1;
            any = true;
        }
        if s.get(j) == Some(&b'.') {
            let mut frac_base = 1.0f64 / 16.0;
            let mut k = j + 1;
            let mut any_frac = false;
            while k < s.len() && s[k].is_ascii_hexdigit() {
                let digit = (s[k] as char).to_digit(16).unwrap_or(0);
                value += frac_base * f64::from(digit);
                frac_base /= 16.0;
                k += 1;
                any_frac = true;
            }
            if any_frac {
                j = k;
            }
        }
        if !any {
            // "0x" with no hex digits: the subject is just "0".
            return (0.0, i + 1);
        }
        if s.get(j).map(|c| c | 0x20) == Some(b'p') {
            let mut k = j + 1;
            let mut exp_neg = false;
            if k < s.len() && (s[k] == b'+' || s[k] == b'-') {
                exp_neg = s[k] == b'-';
                k += 1;
            }
            let digits_start = k;
            let mut exp = 0i32;
            while k < s.len() && s[k].is_ascii_digit() {
                exp = exp.saturating_mul(10).saturating_add((s[k] - b'0') as i32);
                k += 1;
            }
            if k > digits_start {
                value *= 2f64.powi(if exp_neg { -exp } else { exp }.clamp(-4096, 4096));
                j = k;
            }
        }
        return (if negative { -value } else { value }, j);
    }
    // Decimal: D[.D][eE[+-]D].
    let mut j = i;
    while j < s.len() && s[j].is_ascii_digit() {
        j += 1;
    }
    let mut saw_digit = j > i;
    if s.get(j) == Some(&b'.') {
        let mut k = j + 1;
        while k < s.len() && s[k].is_ascii_digit() {
            k += 1;
        }
        if k > j + 1 {
            saw_digit = true;
        }
        if saw_digit {
            j = k;
        }
    }
    if !saw_digit {
        return (0.0, 0);
    }
    if s.get(j).map(|c| c | 0x20) == Some(b'e') {
        let mut k = j + 1;
        if k < s.len() && (s[k] == b'+' || s[k] == b'-') {
            k += 1;
        }
        if k < s.len() && s[k].is_ascii_digit() {
            while k < s.len() && s[k].is_ascii_digit() {
                k += 1;
            }
            j = k;
        }
    }
    let value = std::str::from_utf8(&s[num_start..j])
        .ok()
        .and_then(|text| text.parse::<f64>().ok())
        .unwrap_or(0.0);
    (value, j)
}

/// The `sscanf` engine: walk the format against the input, storing each
/// conversion through its argument pointer. Returns the number of items
/// assigned, or -1 when input failure strikes before the first assignment
/// (the C EOF contract).
fn scan_c(
    mem: &dyn crate::GuestMemory,
    input: &[u8],
    fmt: &[u8],
    args: &mut dyn Iterator<Item = u64>,
) -> i32 {
    let mut assigned = 0i32;
    let mut pos = 0usize;
    let mut fi = 0usize;
    'outer: while fi < fmt.len() {
        let fb = fmt[fi];
        if fb.is_ascii_whitespace() {
            while pos < input.len() && input[pos].is_ascii_whitespace() {
                pos += 1;
            }
            fi += 1;
            continue;
        }
        if fb != b'%' {
            if pos < input.len() && input[pos] == fb {
                pos += 1;
                fi += 1;
                continue;
            }
            break;
        }
        fi += 1;
        if fmt.get(fi) == Some(&b'%') {
            if pos < input.len() && input[pos] == b'%' {
                pos += 1;
                fi += 1;
                continue;
            }
            break;
        }
        let mut suppress = false;
        if fmt.get(fi) == Some(&b'*') {
            suppress = true;
            fi += 1;
        }
        let mut width = 0usize;
        while fi < fmt.len() && fmt[fi].is_ascii_digit() {
            width = width
                .saturating_mul(10)
                .saturating_add((fmt[fi] - b'0') as usize);
            fi += 1;
        }
        let mut len = ScanLen::Int;
        while fi < fmt.len() {
            match fmt[fi] {
                b'l' | b'z' | b't' | b'j' => len = ScanLen::Long,
                b'h' => {
                    len = if len == ScanLen::Short {
                        ScanLen::Char
                    } else {
                        ScanLen::Short
                    };
                }
                _ => break,
            }
            fi += 1;
        }
        let Some(&conv) = fmt.get(fi) else {
            break;
        };
        fi += 1;
        // Every conversion except %c/%[/%n skips leading whitespace first.
        if !matches!(conv, b'c' | b'[' | b'n') {
            while pos < input.len() && input[pos].is_ascii_whitespace() {
                pos += 1;
            }
        }
        match conv {
            b'd' | b'i' | b'u' | b'x' | b'X' | b'o' | b'p' => {
                let base = match conv {
                    b'd' | b'u' => 10,
                    b'i' => 0,
                    b'x' | b'X' | b'p' => 16,
                    _ => 8,
                };
                let (value, consumed) = parse_c_integer(&input[pos..], base);
                if consumed == 0 {
                    break 'outer; // matching failure
                }
                pos += consumed;
                if !suppress {
                    let Some(ptr) = args.next() else { break 'outer };
                    // %p always stores a full pointer, whatever the modifier.
                    let store_len = if conv == b'p' { ScanLen::Long } else { len };
                    if !store_scan_int(mem, ptr, value, store_len) {
                        warn!("sscanf: result pointer {ptr:#x} not writable");
                    }
                    assigned += 1;
                }
            }
            b'f' | b'e' | b'E' | b'g' | b'G' => {
                let (value, consumed) = parse_c_float(&input[pos..]);
                if consumed == 0 {
                    break 'outer;
                }
                pos += consumed;
                if !suppress {
                    let Some(ptr) = args.next() else { break 'outer };
                    let ok = if len == ScanLen::Long {
                        mem.write(ptr, &value.to_le_bytes())
                    } else {
                        mem.write(ptr, &(value as f32).to_le_bytes())
                    };
                    if !ok {
                        warn!("sscanf: float result pointer {ptr:#x} not writable");
                    }
                    assigned += 1;
                }
            }
            b's' => {
                let limit = if width == 0 { usize::MAX } else { width };
                let start = pos;
                while pos < input.len() && !input[pos].is_ascii_whitespace() && pos - start < limit
                {
                    pos += 1;
                }
                if pos == start {
                    break 'outer;
                }
                if !suppress {
                    let Some(ptr) = args.next() else { break 'outer };
                    let mut bytes = input[start..pos].to_vec();
                    bytes.push(0);
                    if !mem.write(ptr, &bytes) {
                        warn!("sscanf %%s: result pointer {ptr:#x} not writable");
                    }
                    assigned += 1;
                }
            }
            b'c' => {
                let count = width.max(1);
                if pos + count > input.len() {
                    break 'outer; // input failure
                }
                if !suppress {
                    let Some(ptr) = args.next() else { break 'outer };
                    if !mem.write(ptr, &input[pos..pos + count]) {
                        warn!("sscanf %%c: result pointer {ptr:#x} not writable");
                    }
                    assigned += 1;
                }
                pos += count;
            }
            b'[' => {
                // Scanset: '^' negates; a ']' first (after any '^') is a
                // member; 'a-z' ranges work between two members.
                let mut negate = false;
                if fmt.get(fi) == Some(&b'^') {
                    negate = true;
                    fi += 1;
                }
                let mut members = [false; 256];
                let mut first = true;
                let mut prev: Option<u8> = None;
                while fi < fmt.len() {
                    let c = fmt[fi];
                    if c == b']' && !first {
                        fi += 1;
                        break;
                    }
                    first = false;
                    if c == b'-' && prev.is_some() && fi + 1 < fmt.len() && fmt[fi + 1] != b']' {
                        let lo = prev.unwrap_or(0) as usize;
                        let hi = fmt[fi + 1] as usize;
                        for member in members.iter_mut().take(hi + 1).skip(lo) {
                            *member = true;
                        }
                        prev = None;
                        fi += 2;
                        continue;
                    }
                    members[c as usize] = true;
                    prev = Some(c);
                    fi += 1;
                }
                let limit = if width == 0 { usize::MAX } else { width };
                let start = pos;
                while pos < input.len()
                    && members[input[pos] as usize] != negate
                    && pos - start < limit
                {
                    pos += 1;
                }
                if pos == start {
                    break 'outer;
                }
                if !suppress {
                    let Some(ptr) = args.next() else { break 'outer };
                    let mut bytes = input[start..pos].to_vec();
                    bytes.push(0);
                    if !mem.write(ptr, &bytes) {
                        warn!("sscanf %%[...]: result pointer {ptr:#x} not writable");
                    }
                    assigned += 1;
                }
            }
            b'n' => {
                // %n reports the consumed count and does NOT count as an
                // assignment (the C contract).
                if !suppress {
                    let Some(ptr) = args.next() else { break 'outer };
                    if !store_scan_int(mem, ptr, pos as i128, len) {
                        warn!("sscanf %%n: result pointer {ptr:#x} not writable");
                    }
                }
            }
            other => {
                warn!(
                    "sscanf: unsupported conversion '%{}'; scan stops here",
                    (other as char).escape_default()
                );
                break 'outer;
            }
        }
    }
    if assigned == 0 && pos >= input.len() {
        -1 // input failure before the first conversion: EOF
    } else {
        assigned
    }
}

/// Real `sscanf(s, fmt, ...)`: scan from the guest string. Returns the
/// assigned-item count, or EOF (-1) on an unreadable input/format or input
/// failure before the first assignment.
fn hle_sscanf(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    let fmt_ptr = args.get(1).copied().unwrap_or(0);
    debug!("sscanf(s={s:#x}, fmt={fmt_ptr:#x})");
    let Some(input) = crate::fmt::read_cstr(ctx.mem, s) else {
        warn!("sscanf: unreadable input string at {s:#x}");
        return u64::MAX; // EOF
    };
    let Some(fmt) = crate::fmt::read_cstr(ctx.mem, fmt_ptr) else {
        warn!("sscanf: unreadable format string at {fmt_ptr:#x}");
        return u64::MAX;
    };
    let mut varargs = args.iter().skip(2).copied();
    (scan_c(ctx.mem, &input, &fmt, &mut varargs) as i64) as u64
}

/// Real `strtoimax(nptr, endptr, base)`: the `intmax_t` (64-bit signed)
/// spelling of `strtol`.
fn hle_strtoimax(ctx: &HleContext, args: &[u64]) -> u64 {
    let nptr = args.first().copied().unwrap_or(0);
    let endptr = args.get(1).copied().unwrap_or(0);
    let base = args.get(2).copied().unwrap_or(0) as u32;
    debug!("strtoimax(nptr={nptr:#x}, endptr={endptr:#x}, base={base})");
    strtol_impl(ctx, nptr, endptr, base, false)
}

/// Real `strtoull(nptr, endptr, base)` / `strtoumax` / Dinkumware's
/// `_Stoull` STL core: the unsigned 64-bit spelling of `strtol` (identical
/// ABI on LP64).
fn hle_strtoull(ctx: &HleContext, args: &[u64]) -> u64 {
    let nptr = args.first().copied().unwrap_or(0);
    let endptr = args.get(1).copied().unwrap_or(0);
    let base = args.get(2).copied().unwrap_or(0) as u32;
    debug!("strtoull(nptr={nptr:#x}, endptr={endptr:#x}, base={base})");
    strtol_impl(ctx, nptr, endptr, base, true)
}

/// Real `strpbrk(s, accept)`: guest address of the first byte of `s` that
/// is also in `accept`, or 0.
fn hle_strpbrk(ctx: &HleContext, args: &[u64]) -> u64 {
    let s = args.first().copied().unwrap_or(0);
    let accept = args.get(1).copied().unwrap_or(0);
    debug!("strpbrk(s={s:#x}, accept={accept:#x})");
    let (Some(bytes), Some(set)) = (
        crate::fmt::read_cstr(ctx.mem, s),
        crate::fmt::read_cstr(ctx.mem, accept),
    ) else {
        warn!("strpbrk: unreadable operand (s={s:#x}, accept={accept:#x})");
        return 0;
    };
    match bytes.iter().position(|byte| set.contains(byte)) {
        Some(off) => s.wrapping_add(off as u64),
        None => 0,
    }
}

/// Real `wcscat(dst, src)`: append the wide string `src` (32-bit units) to
/// `dst`, NUL-terminated, returning `dst`.
fn hle_wcscat(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src = args.get(1).copied().unwrap_or(0);
    debug!("wcscat(dst={dst:#x}, src={src:#x})");
    if dst == 0 || src == 0 {
        return dst;
    }
    let dst_chars = hle_wcslen(ctx, &[dst]);
    let src_chars = hle_wcslen(ctx, &[src]);
    let Ok(bytes) = usize::try_from((src_chars + 1).saturating_mul(4)) else {
        return dst;
    };
    let mut buf = vec![0u8; bytes];
    if !ctx.mem.read(src, &mut buf) {
        warn!("wcscat: unreadable source at {src:#x} ({src_chars} wide chars)");
        return dst;
    }
    // wcslen stops at the terminator; force it rather than trusting the
    // trailing unit just read.
    buf[bytes - 4..].fill(0);
    let append_at = dst.wrapping_add(dst_chars.saturating_mul(4));
    if !ctx.mem.write(append_at, &buf) {
        warn!("wcscat: failed to append {bytes} bytes at {append_at:#x}");
    }
    dst
}

/// Real `mbsrtowcs(dst, src, len, ps)` in the C locale: every byte is one
/// complete multibyte character (no shift states), so conversion is a
/// byte-to-u32 widening. Stops after `len` wide chars or at the source NUL
/// (which is converted, `*src` becomes NULL, and is not counted). `ps` is
/// accepted and ignored — the C locale is stateless. Returns the count, or
/// (size_t)-1 when the source pointer is unreadable.
fn hle_mbsrtowcs(ctx: &HleContext, args: &[u64]) -> u64 {
    let dst = args.first().copied().unwrap_or(0);
    let src_slot = args.get(1).copied().unwrap_or(0);
    let len = args.get(2).copied().unwrap_or(0);
    debug!("mbsrtowcs(dst={dst:#x}, src={src_slot:#x}, len={len:#x})");
    let mut slot = [0u8; 8];
    if src_slot == 0 || !ctx.mem.read(src_slot, &mut slot) {
        warn!("mbsrtowcs: unreadable src-pointer slot at {src_slot:#x}");
        return u64::MAX; // (size_t)-1
    }
    let src = u64::from_le_bytes(slot);
    let set_src = |ctx: &HleContext, value: u64| {
        let _ = ctx.mem.write(src_slot, &value.to_le_bytes());
    };
    // Counting mode (dst == NULL): scan to the NUL, bounded like the other
    // string scans.
    if dst == 0 {
        let mut count = 0u64;
        let mut byte = [0u8; 1];
        while count < STRLEN_MAX_SCAN {
            let Some(addr) = src.checked_add(count) else {
                break;
            };
            if !ctx.mem.read(addr, &mut byte) {
                warn!("mbsrtowcs: unreadable source at {addr:#x}");
                return u64::MAX;
            }
            if byte[0] == 0 {
                return count;
            }
            count += 1;
        }
        return count;
    }
    let mut written = 0u64;
    while written < len.min(STRLEN_MAX_SCAN) {
        let Some(addr) = src.checked_add(written) else {
            break;
        };
        let mut byte = [0u8; 1];
        if !ctx.mem.read(addr, &mut byte) {
            warn!("mbsrtowcs: unreadable source at {addr:#x}");
            return u64::MAX;
        }
        let Some(waddr) = dst.checked_add(written.saturating_mul(4)) else {
            break;
        };
        if !ctx.mem.write(waddr, &u32::from(byte[0]).to_le_bytes()) {
            warn!("mbsrtowcs: destination {waddr:#x} not writable");
            return u64::MAX;
        }
        if byte[0] == 0 {
            // The terminating NUL was converted: *src becomes NULL and the
            // NUL itself is not counted.
            set_src(ctx, 0);
            return written;
        }
        written += 1;
    }
    // Ran out of room before any NUL: the source advances past what was
    // consumed.
    set_src(ctx, src.wrapping_add(written));
    written
}

/// Process-global `rand` state. C's `rand` is an implementation-defined
/// PRNG — only the `[0, RAND_MAX]` range (2^31-1 here, the BSD/glibc value)
/// and `srand` reproducibility are contractual — so a Knuth MMIX LCG over
/// one state word is a conforming, honest generator. Process-global like
/// the real one (and like this file's `STRTOK_SAVE`).
static RAND_STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Real `rand()`: the next LCG output in [0, 2^31).
fn hle_rand(_ctx: &HleContext, _args: &[u64]) -> u64 {
    use std::sync::atomic::Ordering;
    let mut state = RAND_STATE.load(Ordering::Relaxed);
    loop {
        let next = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        match RAND_STATE.compare_exchange_weak(state, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return (next >> 33) & 0x7FFF_FFFF,
            Err(current) => state = current,
        }
    }
}

/// Real `srand(seed)`: reseed the generator.
fn hle_srand(_ctx: &HleContext, args: &[u64]) -> u64 {
    use std::sync::atomic::Ordering;
    let seed = args.first().copied().unwrap_or(0);
    debug!("srand(seed={seed:#x})");
    RAND_STATE.store(seed, Ordering::Relaxed);
    0
}

/// Real `sincos(double x, double *sinp, double *cosp)`: the f64 twin of the
/// existing `sincosf` handler — `x` arrives in XMM0 ([`HleContext`]'s float
/// channel); both out-pointers are integer arguments.
fn hle_sincos(ctx: &HleContext, args: &[u64]) -> u64 {
    let sin_out = args.first().copied().unwrap_or(0);
    let cos_out = args.get(1).copied().unwrap_or(0);
    let x = ctx.float_arg_f64(0);
    debug!("sincos(x={x}, sinp={sin_out:#x}, cosp={cos_out:#x})");
    if sin_out != 0 && !ctx.mem.write(sin_out, &x.sin().to_le_bytes()) {
        warn!("sincos: sin out-ptr {sin_out:#x} not writable");
    }
    if cos_out != 0 && !ctx.mem.write(cos_out, &x.cos().to_le_bytes()) {
        warn!("sincos: cos out-ptr {cos_out:#x} not writable");
    }
    0
}

/// Real `__isfinite(double)`: 1 when the XMM0 argument is finite, else 0.
/// Integer return, so the RAX result channel carries the answer honestly.
fn hle_isfinite(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f64(0).is_finite())
}

/// Real `__isfinitef(float)`: the f32 twin.
fn hle_isfinitef(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).is_finite())
}

/// Real `__isnan(double)`: 1 when the XMM0 argument is NaN, else 0.
fn hle_isnan(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f64(0).is_nan())
}

/// Real `__isnanf(float)`: the f32 twin.
fn hle_isnanf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).is_nan())
}

// ---------------------------------------------------------------------------
// Float/double-RETURNING libm. Every handler below is registered through
// `register_abi_float`, so its `u64` return is the result's BIT PATTERN
// (`f64::to_bits`, or zero-extended `f32::to_bits`), which the runtime's
// float-return channel delivers to guest XMM0 on both dispatch paths. The
// float/double ARGUMENTS arrive in the XMM channel
// ([`HleContext::float_arg_f32`]/[`HleContext::float_arg_f64`]) — SysV
// never puts them in the integer slice. All math is the host's own
// (Rust's libm methods on f32/f64), within the ordinary accuracy contract
// of a real libm.
// ---------------------------------------------------------------------------

/// Real `acosf(float)`: host `f32::acos` (NaN out of domain, like C).
fn hle_acosf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).acos().to_bits())
}

/// Real `asin(double)`: host `f64::asin`.
fn hle_asin(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).asin().to_bits()
}

/// Real `asinf(float)`: host `f32::asin`.
fn hle_asinf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).asin().to_bits())
}

/// Real `atan(double)`: host `f64::atan`.
fn hle_atan(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).atan().to_bits()
}

/// Real `atan2(double y, double x)`: host `f64::atan2` — two arguments in
/// XMM0/XMM1.
fn hle_atan2(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).atan2(ctx.float_arg_f64(1)).to_bits()
}

/// Real `atan2f(float, float)`: the f32 twin.
fn hle_atan2f(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).atan2(ctx.float_arg_f32(1)).to_bits())
}

/// Real `atanf(float)`: host `f32::atan`.
fn hle_atanf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).atan().to_bits())
}

/// Real `cos(double)`: host `f64::cos`.
fn hle_cos(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).cos().to_bits()
}

/// Real `cosf(float)`: host `f32::cos`.
fn hle_cosf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).cos().to_bits())
}

/// Real `difftime(time_t t1, time_t t0)`: the difference in seconds as a
/// double. Both arguments are INTEGER (the integer slice); computing in
/// f64 from the raw values avoids any i64 subtraction overflow.
fn hle_difftime(_ctx: &HleContext, args: &[u64]) -> u64 {
    let t1 = args.first().copied().unwrap_or(0) as i64;
    let t0 = args.get(1).copied().unwrap_or(0) as i64;
    debug!("difftime(t1={t1}, t0={t0})");
    ((t1 as f64) - (t0 as f64)).to_bits()
}

/// Real `exp(double)`: host `f64::exp`.
fn hle_exp(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).exp().to_bits()
}

/// Real `exp2(double)`: host `f64::exp2`.
fn hle_exp2(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).exp2().to_bits()
}

/// Real `exp2f(float)`: host `f32::exp2`.
fn hle_exp2f(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).exp2().to_bits())
}

/// Real `expf(float)`: host `f32::exp`.
fn hle_expf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).exp().to_bits())
}

/// Real `fmod(double x, double y)`: C remainder — Rust's `%` on floats is
/// exactly `fmod` (truncated quotient, result takes x's sign).
fn hle_fmod(ctx: &HleContext, _args: &[u64]) -> u64 {
    (ctx.float_arg_f64(0) % ctx.float_arg_f64(1)).to_bits()
}

/// Real `fmodf(float, float)`: the f32 twin.
fn hle_fmodf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from((ctx.float_arg_f32(0) % ctx.float_arg_f32(1)).to_bits())
}

/// Real `ldexpf(float x, int exp)`: `x * 2^exp`. The float arrives in XMM0;
/// the int exponent is the first INTEGER argument. An out-of-range exponent
/// saturates to 0/inf through `powi`, matching C's overflow/underflow
/// behavior.
fn hle_ldexpf(ctx: &HleContext, args: &[u64]) -> u64 {
    let x = ctx.float_arg_f32(0);
    let exp = (args.first().copied().unwrap_or(0) as i64)
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    debug!("ldexpf(x={x}, exp={exp})");
    u64::from((x * 2f32.powi(exp)).to_bits())
}

/// Real `log(double)`: host `f64::ln` (natural log; −inf at 0, NaN below,
/// like C).
fn hle_log(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).ln().to_bits()
}

/// Real `log10(double)`: host `f64::log10`.
fn hle_log10(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).log10().to_bits()
}

/// Real `log10f(float)`: host `f32::log10`.
fn hle_log10f(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).log10().to_bits())
}

/// Real `log2(double)`: host `f64::log2`.
fn hle_log2(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).log2().to_bits()
}

/// Real `log2f(float)`: host `f32::log2`.
fn hle_log2f(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).log2().to_bits())
}

/// Real `logf(float)`: host `f32::ln`.
fn hle_logf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).ln().to_bits())
}

/// `nextafterf(x, y)` computed by bit-walking — the next representable f32
/// from `x` toward `y`, per the C contract (NaN propagates; `x == y`
/// returns `y`; from zero, the least subnormal of `y`'s sign). Rust has no
/// stable `next_after`, and a hand-rolled bit walk is exact here (no
/// rounding anywhere).
fn next_after_f32(x: f32, y: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if y.is_nan() {
        return y;
    }
    if x == y {
        return y;
    }
    let bits = x.to_bits();
    let next = if x == 0.0 {
        (y.to_bits() & 0x8000_0000) | 1
    } else if (x > 0.0) == (y > x) {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next)
}

/// Real `nextafterf(float x, float y)`: both arguments in XMM0/XMM1.
fn hle_nextafterf(ctx: &HleContext, _args: &[u64]) -> u64 {
    let (x, y) = (ctx.float_arg_f32(0), ctx.float_arg_f32(1));
    u64::from(next_after_f32(x, y).to_bits())
}

/// Real `pow(double x, double y)`: host `f64::powf`.
fn hle_pow(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).powf(ctx.float_arg_f64(1)).to_bits()
}

/// Real `powf(float, float)`: host `f32::powf`.
fn hle_powf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).powf(ctx.float_arg_f32(1)).to_bits())
}

/// Real `sin(double)`: host `f64::sin`.
fn hle_sin(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).sin().to_bits()
}

/// Real `sinf(float)`: host `f32::sin`.
fn hle_sinf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).sin().to_bits())
}

/// Real `tan(double)`: host `f64::tan`.
fn hle_tan(ctx: &HleContext, _args: &[u64]) -> u64 {
    ctx.float_arg_f64(0).tan().to_bits()
}

/// Real `tanf(float)`: host `f32::tan`.
fn hle_tanf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).tan().to_bits())
}

/// Real `tanhf(float)`: host `f32::tanh`.
fn hle_tanhf(ctx: &HleContext, _args: &[u64]) -> u64 {
    u64::from(ctx.float_arg_f32(0).tanh().to_bits())
}

/// Shared `strtod`/`atof` body: parse with [`parse_c_float`] (the same C
/// literal grammar sscanf uses — inf/nan, hex floats, decimal with
/// exponent), returning the value's f64 bits. On an unreadable string or a
/// non-converting one the value is 0.0 and `endptr` (when given) points at
/// `nptr` — the real no-conversion contract.
fn strtod_impl(ctx: &HleContext, nptr: u64, endptr: u64) -> u64 {
    let store_end = |ctx: &HleContext, consumed: u64| {
        if endptr != 0 {
            let end_addr = nptr.wrapping_add(consumed);
            if !ctx.mem.write(endptr, &end_addr.to_le_bytes()) {
                warn!("strtod: failed to write endptr at {endptr:#x}");
            }
        }
    };
    let Some(bytes) = crate::fmt::read_cstr(ctx.mem, nptr) else {
        warn!("strtod: unreadable string at {nptr:#x}");
        store_end(ctx, 0);
        return 0.0f64.to_bits();
    };
    let (value, consumed) = parse_c_float(&bytes);
    store_end(ctx, consumed as u64);
    value.to_bits()
}

/// Real `strtod(nptr, endptr)`: C double parsing; the value travels back in
/// XMM0 via the float-return channel.
fn hle_strtod(ctx: &HleContext, args: &[u64]) -> u64 {
    let nptr = args.first().copied().unwrap_or(0);
    let endptr = args.get(1).copied().unwrap_or(0);
    debug!("strtod(nptr={nptr:#x}, endptr={endptr:#x})");
    strtod_impl(ctx, nptr, endptr)
}

/// Real `atof(nptr)`: `strtod(nptr, NULL)`.
fn hle_atof(ctx: &HleContext, args: &[u64]) -> u64 {
    let nptr = args.first().copied().unwrap_or(0);
    debug!("atof(nptr={nptr:#x})");
    strtod_impl(ctx, nptr, 0)
}

/// Seconds since the Unix epoch from the host wall clock.
fn host_unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Days since 1970-01-01 for the proleptic Gregorian `y`-`m`-`d` — Howard
/// Hinnant's well-known public-domain `days_from_civil` algorithm, written
/// out here in original Rust (no library code copied).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400; // [0, 399]
    let mp = (i64::from(m) + 9).rem_euclid(12); // March = 0
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of [`days_from_civil`]: `(year, month, day)` for a day count
/// since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

/// The nine standard `struct tm` fields (all `int`), the prefix every
/// `struct tm` layout on this platform shares.
#[derive(Clone, Copy, Default)]
struct GuestTm {
    sec: i32,
    min: i32,
    hour: i32,
    mday: i32,
    mon: i32,
    year: i32, // years since 1900
    wday: i32, // 0 = Sunday
    yday: i32, // 0-based day of year
    isdst: i32,
}

/// Break a UTC second count down into calendar fields.
fn break_down_utc(t: i64) -> GuestTm {
    let days = t.div_euclid(86_400);
    let secs = t.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let yday = days - days_from_civil(year, 1, 1);
    GuestTm {
        sec: (secs % 60) as i32,
        min: ((secs / 60) % 60) as i32,
        hour: (secs / 3600) as i32,
        mday: day as i32,
        mon: month as i32 - 1,
        year: (year - 1900) as i32,
        wday: (days + 4).rem_euclid(7) as i32, // 1970-01-01 was a Thursday
        yday: yday as i32,
        isdst: 0,
    }
}

/// Read the nine standard `struct tm` ints from guest memory.
fn read_tm(mem: &dyn crate::GuestMemory, addr: u64) -> Option<GuestTm> {
    let mut raw = [0u8; 36];
    if addr == 0 || !mem.read(addr, &mut raw) {
        return None;
    }
    let field = |i: usize| i32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap_or([0; 4]));
    Some(GuestTm {
        sec: field(0),
        min: field(1),
        hour: field(2),
        mday: field(3),
        mon: field(4),
        year: field(5),
        wday: field(6),
        yday: field(7),
        isdst: field(8),
    })
}

/// Write the nine standard `struct tm` ints to guest memory.
fn write_tm(mem: &dyn crate::GuestMemory, addr: u64, tm: &GuestTm) -> bool {
    let fields = [
        tm.sec, tm.min, tm.hour, tm.mday, tm.mon, tm.year, tm.wday, tm.yday, tm.isdst,
    ];
    let mut raw = [0u8; 36];
    for (i, field) in fields.iter().enumerate() {
        raw[i * 4..i * 4 + 4].copy_from_slice(&field.to_le_bytes());
    }
    mem.write(addr, &raw)
}

/// The host's local-time zone: UTC offset (local = UTC + offset), current
/// daylight-saving state, and zone name. Same `GetTimeZoneInformation`
/// source `libsce_rtc` already uses; non-Windows hosts fall back to UTC
/// with an empty name (the runtime is Windows-only today).
struct HostZone {
    offset_secs: i64,
    is_dst: bool,
    name: String,
}

fn host_zone() -> &'static HostZone {
    static ZONE: std::sync::OnceLock<HostZone> = std::sync::OnceLock::new();
    ZONE.get_or_init(|| {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Time::{GetTimeZoneInformation, TIME_ZONE_INFORMATION};
            // GetTimeZoneInformation's return: 0 = UNKNOWN, 1 = STANDARD,
            // 2 = DAYLIGHT. The named constant lives in
            // Win32::System::SystemServices, a windows-sys feature this
            // workspace does not enable — the literal is the same value.
            const TIME_ZONE_ID_DAYLIGHT: u32 = 2;
            // SAFETY: plain out-param POD on a live thread; no aliasing.
            let mut info: TIME_ZONE_INFORMATION = unsafe { std::mem::zeroed() };
            let id = unsafe { GetTimeZoneInformation(&mut info) };
            let is_dst = id == TIME_ZONE_ID_DAYLIGHT;
            // UTC = local + bias ⇒ local = UTC − bias; the daylight/standard
            // bias adds on top of the base bias for the current state.
            let bias_minutes = i64::from(info.Bias)
                + if is_dst {
                    i64::from(info.DaylightBias)
                } else {
                    i64::from(info.StandardBias)
                };
            let name_src = if is_dst {
                &info.DaylightName
            } else {
                &info.StandardName
            };
            let end = name_src
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(name_src.len());
            HostZone {
                offset_secs: -bias_minutes * 60,
                is_dst,
                name: String::from_utf16_lossy(&name_src[..end]),
            }
        }
        #[cfg(not(windows))]
        {
            HostZone {
                offset_secs: 0,
                is_dst: false,
                name: String::from("UTC"),
            }
        }
    })
}

/// Size of the `struct tm` storage `gmtime`/`localtime` hand out: the nine
/// standard ints, padding, then the BSD/glibc extension fields
/// (`long tm_gmtoff` at 40, `char *tm_zone` at 48) — the superset layout,
/// so a title built for either shape reads correct values (a 9-int consumer
/// simply never looks past 36).
const TM_STRUCT_BYTES: u64 = 56;

/// Build a broken-down-time object in guest memory: the 56-byte struct plus
/// a NUL-terminated zone name right after it. Returns the struct address,
/// or 0 on arena exhaustion.
fn broken_down_tm_object(ctx: &HleContext, t_utc: i64, local: bool) -> u64 {
    let zone = host_zone();
    let (tm, gmtoff, zone_name) = if local {
        let mut tm = break_down_utc(t_utc.saturating_add(zone.offset_secs));
        // Approximation, documented: the CURRENT host DST state, not the
        // DST rule at `t_utc` (GetTimeZoneInformation carries no history).
        tm.isdst = i32::from(zone.is_dst);
        (tm, zone.offset_secs, zone.name.as_str())
    } else {
        (break_down_utc(t_utc), 0, "UTC")
    };
    let Some(base) = ctx.alloc.alloc(TM_STRUCT_BYTES + 16, 16) else {
        return 0;
    };
    let zone_ptr = base + TM_STRUCT_BYTES;
    let name_bytes = {
        let mut v = zone_name.as_bytes().to_vec();
        v.truncate(15);
        v.push(0);
        v
    };
    if !write_tm(ctx.mem, base, &tm)
        || !ctx.mem.write(base + 40, &gmtoff.to_le_bytes())
        || !ctx.mem.write(base + 48, &zone_ptr.to_le_bytes())
        || !ctx.mem.write(zone_ptr, &name_bytes)
    {
        warn!("broken_down_tm_object: guest buffer {base:#x} not writable");
        ctx.alloc.free(base);
        return 0;
    }
    base
}

/// Real `time(time_t*)`: seconds since the Unix epoch from the host clock,
/// also stored through the out-pointer when non-NULL.
fn hle_time(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    let secs = host_unix_seconds();
    debug!("time(t={out:#x}) -> {secs}");
    if out != 0 && !ctx.mem.write(out, &secs.to_le_bytes()) {
        warn!("time: out-ptr {out:#x} not writable");
    }
    secs as u64
}

/// Shared body of `gmtime`/`localtime`: read the `time_t`, build the
/// broken-down object. Returns 0 (NULL) on an unreadable argument or an
/// unrepresentable year — the real functions' failure answer.
fn gmtime_impl(ctx: &HleContext, timer: u64, local: bool) -> u64 {
    let mut raw = [0u8; 8];
    if timer == 0 || !ctx.mem.read(timer, &mut raw) {
        warn!("gmtime/localtime: unreadable time_t at {timer:#x}");
        return 0;
    }
    let t = i64::from_le_bytes(raw);
    let zone = host_zone();
    let shifted = if local {
        t.saturating_add(zone.offset_secs)
    } else {
        t
    };
    let (year, _, _) = civil_from_days(shifted.div_euclid(86_400));
    if i32::try_from(year - 1900).is_err() {
        warn!("gmtime/localtime: year {year} is not representable");
        return 0;
    }
    broken_down_tm_object(ctx, t, local)
}

/// Real `gmtime(time_t*)`: UTC broken-down time in guest storage (fresh per
/// call; the real API's static buffer is allowed to be overwritten, never
/// required to be).
fn hle_gmtime(ctx: &HleContext, args: &[u64]) -> u64 {
    let timer = args.first().copied().unwrap_or(0);
    debug!("gmtime(timer={timer:#x})");
    gmtime_impl(ctx, timer, false)
}

/// Real `localtime(time_t*)`: broken-down time in the host's local zone
/// (offset and name from `GetTimeZoneInformation`).
fn hle_localtime(ctx: &HleContext, args: &[u64]) -> u64 {
    let timer = args.first().copied().unwrap_or(0);
    debug!("localtime(timer={timer:#x})");
    gmtime_impl(ctx, timer, true)
}

/// Real `gmtime_s(timer, result)` with C11 Annex K argument order (Dinkumware
/// authored Annex K; this is NOT Microsoft's reversed order). Fills the
/// caller's `struct tm` with the UTC breakdown of `*timer` — the nine
/// standard ints only, the prefix every `struct tm` shape shares — and
/// returns 0, or EINVAL (22) on a constraint violation.
fn hle_gmtime_s(ctx: &HleContext, args: &[u64]) -> u64 {
    let timer = args.first().copied().unwrap_or(0);
    let result = args.get(1).copied().unwrap_or(0);
    debug!("gmtime_s(timer={timer:#x}, result={result:#x})");
    const EINVAL: u64 = 22;
    let mut raw = [0u8; 8];
    if timer == 0 || result == 0 || !ctx.mem.read(timer, &mut raw) {
        warn!("gmtime_s: constraint violation (timer={timer:#x}, result={result:#x})");
        return EINVAL;
    }
    let t = i64::from_le_bytes(raw);
    let (year, _, _) = civil_from_days(t.div_euclid(86_400));
    if i32::try_from(year - 1900).is_err() {
        return EINVAL;
    }
    let tm = break_down_utc(t);
    if !write_tm(ctx.mem, result, &tm) {
        warn!("gmtime_s: result {result:#x} not writable");
        return EINVAL;
    }
    0
}

/// Real `mktime(struct tm*)`: interpret the fields as LOCAL time, normalize
/// out-of-range fields in place (real mktime contract: wday/yday are
/// recomputed and stored back), and return the corresponding `time_t`, or
/// (time_t)-1 when unrepresentable. The local→UTC conversion uses the
/// current host offset (same documented approximation as `libsce_rtc`'s
/// UTC↔local tick conversion).
fn hle_mktime(ctx: &HleContext, args: &[u64]) -> u64 {
    let tm_ptr = args.first().copied().unwrap_or(0);
    debug!("mktime(tm={tm_ptr:#x})");
    let Some(tm) = read_tm(ctx.mem, tm_ptr) else {
        warn!("mktime: unreadable struct tm at {tm_ptr:#x}");
        return u64::MAX; // (time_t)-1
    };
    let year = i64::from(tm.year) + 1900 + i64::from(tm.mon).div_euclid(12);
    let mon0 = i64::from(tm.mon).rem_euclid(12);
    let days = days_from_civil(year, (mon0 + 1) as u32, 1) + i64::from(tm.mday) - 1;
    let local_secs = days
        .saturating_mul(86_400)
        .saturating_add(i64::from(tm.hour) * 3600 + i64::from(tm.min) * 60 + i64::from(tm.sec));
    let zone = host_zone();
    let utc_secs = local_secs.saturating_sub(zone.offset_secs);
    let mut normalized = break_down_utc(local_secs);
    normalized.isdst = if tm.isdst < 0 {
        i32::from(zone.is_dst)
    } else {
        tm.isdst
    };
    if i32::try_from(civil_from_days(local_secs.div_euclid(86_400)).0 - 1900).is_err() {
        return u64::MAX;
    }
    if !write_tm(ctx.mem, tm_ptr, &normalized) {
        warn!("mktime: could not write normalized fields to {tm_ptr:#x}");
        return u64::MAX;
    }
    utc_secs as u64
}

/// The `strftime` engine for the C locale. Covers the standard directives
/// (compound ones recurse); ISO-week (`%g`/`%G`/`%V`) and anything else
/// unknown is emitted verbatim with a warning, matching `format_c`'s
/// visibly-wrong-beats-silently-dropped contract. `%z` reports +0000 and
/// `%Z` the empty string: the 9-int `struct tm` prefix carries no zone.
fn strftime_c(fmt: &[u8], tm: &GuestTm) -> Vec<u8> {
    const ABDAY: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const DAY: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    const ABMON: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const MON: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];

    fn num(out: &mut Vec<u8>, value: i64, width: usize, pad: u8) {
        let text = value.to_string();
        if text.len() < width {
            out.extend(std::iter::repeat_n(pad, width - text.len()));
        }
        out.extend_from_slice(text.as_bytes());
    }

    let mut out = Vec::new();
    let year = i64::from(tm.year) + 1900;
    let wday_ok = (0..7).contains(&tm.wday);
    let mon_ok = (0..12).contains(&tm.mon);
    let mut i = 0usize;
    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        let Some(&conv) = fmt.get(i) else {
            out.extend_from_slice(&fmt[start..]);
            break;
        };
        i += 1;
        match conv {
            b'a' => out.extend_from_slice(
                if wday_ok {
                    ABDAY[tm.wday as usize]
                } else {
                    "?"
                }
                .as_bytes(),
            ),
            b'A' => {
                out.extend_from_slice(if wday_ok { DAY[tm.wday as usize] } else { "?" }.as_bytes())
            }
            b'b' | b'h' => {
                out.extend_from_slice(if mon_ok { ABMON[tm.mon as usize] } else { "?" }.as_bytes());
            }
            b'B' => {
                out.extend_from_slice(if mon_ok { MON[tm.mon as usize] } else { "?" }.as_bytes())
            }
            b'c' => out.extend_from_slice(&strftime_c(b"%a %b %e %H:%M:%S %Y", tm)),
            b'C' => num(&mut out, year.div_euclid(100), 2, b'0'),
            b'd' => num(&mut out, i64::from(tm.mday), 2, b'0'),
            b'D' => out.extend_from_slice(&strftime_c(b"%m/%d/%y", tm)),
            b'e' => num(&mut out, i64::from(tm.mday), 2, b' '),
            b'F' => out.extend_from_slice(&strftime_c(b"%Y-%m-%d", tm)),
            b'H' => num(&mut out, i64::from(tm.hour), 2, b'0'),
            b'I' => {
                let h12 = tm.hour.rem_euclid(12);
                num(
                    &mut out,
                    i64::from(if h12 == 0 { 12 } else { h12 }),
                    2,
                    b'0',
                );
            }
            b'j' => num(&mut out, i64::from(tm.yday) + 1, 3, b'0'),
            b'm' => num(&mut out, i64::from(tm.mon) + 1, 2, b'0'),
            b'M' => num(&mut out, i64::from(tm.min), 2, b'0'),
            b'n' => out.push(b'\n'),
            b'p' => out.extend_from_slice(if tm.hour < 12 { b"AM" } else { b"PM" }),
            b'r' => out.extend_from_slice(&strftime_c(b"%I:%M:%S %p", tm)),
            b'R' => out.extend_from_slice(&strftime_c(b"%H:%M", tm)),
            b'S' => num(&mut out, i64::from(tm.sec), 2, b'0'),
            b't' => out.push(b'\t'),
            b'T' => out.extend_from_slice(&strftime_c(b"%H:%M:%S", tm)),
            b'u' => num(
                &mut out,
                i64::from((tm.wday + 6).rem_euclid(7)) + 1,
                1,
                b'0',
            ),
            b'U' => num(&mut out, i64::from(tm.yday + 7 - tm.wday) / 7, 2, b'0'),
            b'w' => num(&mut out, i64::from(tm.wday), 1, b'0'),
            b'W' => num(
                &mut out,
                i64::from(tm.yday + 7 - (tm.wday + 6).rem_euclid(7)) / 7,
                2,
                b'0',
            ),
            b'x' => out.extend_from_slice(&strftime_c(b"%m/%d/%y", tm)),
            b'X' => out.extend_from_slice(&strftime_c(b"%H:%M:%S", tm)),
            b'y' => num(&mut out, year.rem_euclid(100), 2, b'0'),
            b'Y' => out.extend_from_slice(year.to_string().as_bytes()),
            b'z' => out.extend_from_slice(b"+0000"),
            b'Z' => {}
            b'%' => out.push(b'%'),
            other => {
                warn!(
                    "strftime: unsupported conversion '%{}' emitted verbatim",
                    (other as char).escape_default()
                );
                out.extend_from_slice(&fmt[start..i]);
            }
        }
    }
    out
}

/// Real `strftime(buf, maxsize, fmt, tm)`: format the broken-down time,
/// returning the byte count excluding the NUL — or 0 when the result does
/// not fit `maxsize` (the C contract; a terminated prefix is still left in
/// the buffer for debuggability, which C permits as "indeterminate").
fn hle_strftime(ctx: &HleContext, args: &[u64]) -> u64 {
    let buf = args.first().copied().unwrap_or(0);
    let maxsize = args.get(1).copied().unwrap_or(0);
    let fmt_ptr = args.get(2).copied().unwrap_or(0);
    let tm_ptr = args.get(3).copied().unwrap_or(0);
    debug!("strftime(buf={buf:#x}, maxsize={maxsize:#x}, fmt={fmt_ptr:#x}, tm={tm_ptr:#x})");
    let Some(fmt) = crate::fmt::read_cstr(ctx.mem, fmt_ptr) else {
        warn!("strftime: unreadable format string at {fmt_ptr:#x}");
        return 0;
    };
    let Some(tm) = read_tm(ctx.mem, tm_ptr) else {
        warn!("strftime: unreadable struct tm at {tm_ptr:#x}");
        return 0;
    };
    let out = strftime_c(&fmt, &tm);
    if maxsize == 0 || out.len() as u64 >= maxsize {
        if maxsize > 0 && buf != 0 {
            let keep = usize::try_from(maxsize - 1)
                .unwrap_or(usize::MAX)
                .min(out.len());
            let mut truncated = out[..keep].to_vec();
            truncated.push(0);
            let _ = ctx.mem.write(buf, &truncated);
        }
        return 0;
    }
    let mut bytes = out;
    let len = bytes.len() as u64;
    bytes.push(0);
    if !ctx.mem.write(buf, &bytes) {
        warn!(
            "strftime: failed to write {} bytes to buf={buf:#x}",
            bytes.len()
        );
        return 0;
    }
    len
}

/// Real `localeconv()`: the C locale's `struct lconv` — `decimal_point` is
/// ".", every other string empty, and every char field CHAR_MAX ("value not
/// available in the C locale"), exactly what the C standard specifies.
///
/// Layout (LP64): **ten** `char *` fields at 0..80, then **fourteen** `char`
/// fields at 80..94 — `int_frac_digits`, `frac_digits`, `p_cs_precedes`,
/// `p_sep_by_space`, `n_cs_precedes`, `n_sep_by_space`, `p_sign_posn`,
/// `n_sign_posn`, then the six `int_*` variants C99 added
/// (`int_p_cs_precedes`, `int_n_cs_precedes`, `int_p_sep_by_space`,
/// `int_n_sep_by_space`, `int_p_sign_posn`, `int_n_sign_posn`) — padded to 96.
/// The two strings live **after** the struct, never inside it: an earlier
/// version treated the char block as eight fields ending at 88 and put `"."`
/// at offset 88, which is `int_p_cs_precedes`. A guest reading that field saw
/// `'.'` (46) instead of `CHAR_MAX`, and a guest *writing* any `int_*` field
/// (they are the caller's to modify in a copied `lconv`) corrupted the
/// `decimal_point` string the struct still pointed at.
fn hle_localeconv(ctx: &HleContext, _args: &[u64]) -> u64 {
    /// `sizeof(struct lconv)`: 94 bytes of fields, padded to the 8-byte
    /// alignment its leading pointers require.
    const LCONV_BYTES: usize = 96;
    /// `"." NUL` + `"" NUL`.
    const STRING_BYTES: usize = 3;
    let Some(base) = ctx.alloc.alloc((LCONV_BYTES + STRING_BYTES) as u64, 8) else {
        warn!("localeconv: guest arena exhausted for struct lconv");
        return 0;
    };
    // Both strings sit past the end of the struct: `"."` then the shared empty
    // string (the NUL terminating `"."` is not reused, so a guest that walks
    // `empty` backwards cannot reach `'.'`).
    let dot = base + LCONV_BYTES as u64;
    let empty = dot + 2;
    let mut bytes = vec![0u8; LCONV_BYTES + STRING_BYTES];
    for (i, ptr) in [
        dot, empty, empty, empty, empty, empty, empty, empty, empty, empty,
    ]
    .iter()
    .enumerate()
    {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&ptr.to_le_bytes());
    }
    // All fourteen char fields, `int_*` variants included.
    for byte in &mut bytes[80..94] {
        *byte = 0x7F; // CHAR_MAX: "not available in the C locale"
    }
    bytes[LCONV_BYTES] = b'.';
    if !ctx.mem.write(base, &bytes) {
        warn!("localeconv: guest buffer {base:#x} not writable");
        ctx.alloc.free(base);
        return 0;
    }
    base
}

/// `catchReturnFromMain(status)`: crt0's bridge from `main`'s return value
/// to process termination (the real one runs `exit(status)` and never
/// returns). Raeen's designed exit path is the scheduler's
/// `request_process_exit`: the dispatch loop turns it into a clean
/// `RunOutcome::Exited` for the Shell instead of a trap — SharpEmu's
/// catchReturnFromMain does the same via RequestCurrentEntryExit (GPL-2.0).
fn hle_catch_return_from_main(ctx: &HleContext, args: &[u64]) -> u64 {
    let status = args.first().copied().unwrap_or(0);
    debug!("catchReturnFromMain(status={status})");
    ctx.guest_threads.request_process_exit(status);
    0
}

/// `_Assert(expr, file, line)`: Dinkumware's `assert()` core — prints
/// "Assertion failed: ..." and aborts. The message goes to the kernel
/// console (stderr's home in this process) and the host log with the caller
/// address. Deliberately still returns 0 so diagnostics keep flowing: the
/// real `_Assert` calls `abort()`, but its call sites (the `assert()` macro
/// expansion) have well-formed code after the call, so returning is safe
/// here — unlike `abort`/`exit`/`__stack_chk_fail`, which are emitted as
/// `noreturn` tail calls and now unwind the guest thread instead.
fn hle_dinkum_assert(ctx: &HleContext, args: &[u64]) -> u64 {
    let expr = args.first().copied().unwrap_or(0);
    let file = args.get(1).copied().unwrap_or(0);
    let line = args.get(2).copied().unwrap_or(0) as u32;
    let read = |ptr: u64| {
        crate::fmt::read_cstr(ctx.mem, ptr)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_else(|| format!("<bad ptr {ptr:#x}>"))
    };
    let message = format!(
        "Assertion failed: {}, file {}, line {line}\n",
        read(expr),
        read(file)
    );
    ctx.kernel.console.write_bytes(message.as_bytes());
    warn!(
        caller = format_args!("{:#x}", ctx.caller_return_addr),
        "guest assertion failed (real libc aborts here): {message}"
    );
    0
}

/// Bytes reserved before the thrown object for the ABI's private exception
/// header (`__cxa_exception`: the `_Unwind_Exception` plus the vendor's
/// handler state), so `__cxa_free_exception` can recover the allocation
/// base from the public pointer.
const CXA_EXCEPTION_HEADER_BYTES: u64 = 128;

/// Real `__cxa_allocate_exception(size)`: carve `size` bytes for a thrown
/// object out of the guest arena behind the private header and return the
/// object pointer. The header itself would be filled by `__cxa_throw` (not
/// HLE'd — the unwinder is out of scope), but the allocation contract is
/// real: titles link the whole ABI and only exercise this on a throw.
/// Returns 0 on exhaustion (the real one calls `std::terminate` there).
fn hle_cxa_allocate_exception(ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.first().copied().unwrap_or(0);
    debug!("__cxa_allocate_exception(size={size:#x})");
    let Some(total) = size.checked_add(CXA_EXCEPTION_HEADER_BYTES) else {
        return 0;
    };
    let Some(base) = ctx.alloc.alloc(total, 16) else {
        warn!("__cxa_allocate_exception({size:#x}): arena exhausted (real ABI terminates here)");
        return 0;
    };
    base + CXA_EXCEPTION_HEADER_BYTES
}

/// Real `__cxa_free_exception(ptr)`: release a block from
/// [`hle_cxa_allocate_exception`]. Unknown pointers are ignored by the
/// allocator, matching `free`'s tolerance here.
fn hle_cxa_free_exception(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    debug!("__cxa_free_exception(ptr={ptr:#x})");
    if ptr >= CXA_EXCEPTION_HEADER_BYTES {
        ctx.alloc.free(ptr - CXA_EXCEPTION_HEADER_BYTES);
    }
    0
}

/// `__cxa_thread_atexit(fn, arg, dso)`: the TLS-destructor twin of
/// `__cxa_atexit` — same record-and-succeed contract (registration must
/// succeed or C++ TLS init aborts; Raeen does not dispatch registered
/// destructors at thread exit yet — documented at `hle_cxa_atexit`).
fn hle_cxa_thread_atexit(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "__cxa_thread_atexit(fn={:#x}, arg={:#x}, dso={:#x}) [registered; not dispatched at thread exit yet]",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0)
    );
    0
}

/// Shared defined-failure for the Dinkumware/C++-ABI throw helpers
/// (`_ZSt14_Xlength_error`, `_ZSt9terminatev`, `__cxa_bad_cast`, ...): the
/// real functions construct an exception and throw — a noreturn path Raeen
/// cannot honor without a guest unwinder. Following this crate's
/// `hle_cxa_pure_virtual` pattern, the handler logs the exact throw
/// (message and caller address) and returns 0 so diagnostics keep flowing;
/// the guest continues past what should have been a throw, which is loud
/// and diagnosable rather than a silent lie.
fn throw_path_failure(ctx: &HleContext, name: &str, detail: &str) -> u64 {
    warn!(
        caller = format_args!("{:#x}", ctx.caller_return_addr),
        "{name}: guest tried to throw ({detail}) — no guest unwinder; \
         continuing after logging (defined failure, real libc never returns)"
    );
    0
}

/// Read a guest C string for a throw-helper message, tolerating a bad
/// pointer in the log text.
fn read_guest_message(ctx: &HleContext, ptr: u64) -> String {
    crate::fmt::read_cstr(ctx.mem, ptr)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_else(|| format!("<bad ptr {ptr:#x}>"))
}

/// `std::_Xlength_error(const char*)`: would throw `std::length_error`.
fn hle_std_xlength_error(ctx: &HleContext, args: &[u64]) -> u64 {
    let msg = read_guest_message(ctx, args.first().copied().unwrap_or(0));
    throw_path_failure(ctx, "std::_Xlength_error", &msg)
}

/// `std::_Xout_of_range(const char*)`: would throw `std::out_of_range`.
fn hle_std_xout_of_range(ctx: &HleContext, args: &[u64]) -> u64 {
    let msg = read_guest_message(ctx, args.first().copied().unwrap_or(0));
    throw_path_failure(ctx, "std::_Xout_of_range", &msg)
}

/// `std::_Xbad_alloc()`: would throw `std::bad_alloc`.
fn hle_std_xbad_alloc(ctx: &HleContext, _args: &[u64]) -> u64 {
    throw_path_failure(ctx, "std::_Xbad_alloc", "allocation failed")
}

/// `std::_Xbad_function_call()`: would throw `std::bad_function_call`.
fn hle_std_xbad_function_call(ctx: &HleContext, _args: &[u64]) -> u64 {
    throw_path_failure(
        ctx,
        "std::_Xbad_function_call",
        "empty std::function invoked",
    )
}

/// `std::terminate()`: would abort the process after a noexcept/unwinding
/// violation.
fn hle_std_terminate(ctx: &HleContext, _args: &[u64]) -> u64 {
    throw_path_failure(
        ctx,
        "std::terminate",
        "noexcept violation or missing handler",
    )
}

/// `std::_Throw_C_error(int)`: would throw `std::system_error` from a C
/// error code.
fn hle_std_throw_c_error(ctx: &HleContext, args: &[u64]) -> u64 {
    let code = args.first().copied().unwrap_or(0);
    throw_path_failure(ctx, "std::_Throw_C_error", &format!("code={code}"))
}

/// `std::_Throw_Cpp_error(int)`: would throw `std::system_error` from a
/// `std::errc` value.
fn hle_std_throw_cpp_error(ctx: &HleContext, args: &[u64]) -> u64 {
    let code = args.first().copied().unwrap_or(0);
    throw_path_failure(ctx, "std::_Throw_Cpp_error", &format!("errc={code}"))
}

/// `std::exception::_Raise() const`: Dinkumware's rethrow-as-this-type
/// helper.
fn hle_std_exception_raise(ctx: &HleContext, _args: &[u64]) -> u64 {
    throw_path_failure(ctx, "std::exception::_Raise", "rethrow")
}

/// `std::exception::_Doraise() const`: Dinkumware's throw-this helper.
fn hle_std_exception_doraise(ctx: &HleContext, _args: &[u64]) -> u64 {
    throw_path_failure(ctx, "std::exception::_Doraise", "throw")
}

/// `__cxa_bad_cast()`: the ABI's trap for a failed `dynamic_cast&` — would
/// throw `std::bad_cast`.
fn hle_cxa_bad_cast(ctx: &HleContext, _args: &[u64]) -> u64 {
    throw_path_failure(ctx, "__cxa_bad_cast", "failed dynamic_cast")
}

/// Real `std::uncaught_exception()`: whether stack unwinding is in flight.
/// Raeen has no guest unwinder and no throw path starts one, so "false" is
/// the TRUE state of this runtime — a real answer, not a default.
fn hle_std_uncaught_exception(_ctx: &HleContext, _args: &[u64]) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    /// A host-side stand-in for the runtime's synchronous guest-callback
    /// dispatch: "calls" the comparator by reading the two pointed-at `u64`s
    /// straight from the shared test memory and comparing them — exactly the
    /// observable behavior of the guest comparator fixture the runtime
    /// acceptance test uses.
    struct HostComparator<'a> {
        mem: &'a crate::TestMemory,
        expected_entry: u64,
        calls: std::cell::Cell<u64>,
    }

    impl crate::GuestCallScheduler for HostComparator<'_> {
        fn request(&self, _request: crate::GuestCallRequest) -> bool {
            false
        }

        fn call_guest(&self, entry: u64, args: [u64; 6]) -> Result<u64, crate::GuestCallError> {
            assert_eq!(
                entry, self.expected_entry,
                "qsort must dispatch the comparator the guest supplied"
            );
            self.calls.set(self.calls.get() + 1);
            let read = |addr: u64| {
                let mut bytes = [0u8; 8];
                assert!(
                    self.mem.read(addr, &mut bytes),
                    "comparator argument {addr:#x} must point into the live array"
                );
                u64::from_le_bytes(bytes)
            };
            let (a, b) = (read(args[0]), read(args[1]));
            Ok(match a.cmp(&b) {
                std::cmp::Ordering::Less => (-1i32) as u32 as u64,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            })
        }
    }

    /// `qsort` end-to-end at the HLE level: an in-memory `u64` array is
    /// sorted ascending through synchronous comparator dispatches that
    /// receive REAL element addresses inside the array, and the comparator
    /// runs at least `nmemb - 1` times (no sort can verify order with
    /// fewer comparisons).
    #[test]
    fn qsort_sorts_a_guest_array_through_a_synchronous_comparator() {
        const BASE: u64 = 0x100;
        const CMP_ENTRY: u64 = 0x9000;
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let values: [u64; 7] = [5, 1, 4, 2, 6, 3, 3];
        for (i, v) in values.iter().enumerate() {
            assert!(mem.write(BASE + i as u64 * 8, &v.to_le_bytes()));
        }
        let comparator = HostComparator {
            mem: &mem,
            expected_entry: CMP_ENTRY,
            calls: std::cell::Cell::new(0),
        };
        let ctx = crate::test_ctx_with_guest_calls(&kernel, &mem, &alloc, &comparator);

        let ret = hle_qsort(&ctx, &[BASE, values.len() as u64, 8, CMP_ENTRY, 0, 0]);
        assert_eq!(ret, 0, "qsort returns void");

        let mut sorted = [0u64; 7];
        for (i, slot) in sorted.iter_mut().enumerate() {
            let mut bytes = [0u8; 8];
            assert!(mem.read(BASE + i as u64 * 8, &mut bytes));
            *slot = u64::from_le_bytes(bytes);
        }
        assert_eq!(
            sorted,
            [1, 2, 3, 3, 4, 5, 6],
            "ascending per the comparator"
        );
        assert!(
            comparator.calls.get() >= values.len() as u64 - 1,
            "a real sort cannot order {} elements with only {} comparator calls",
            values.len(),
            comparator.calls.get()
        );
    }

    /// A dispatch context that cannot re-enter guest code (the default
    /// `call_guest` — test doubles, the direct gateway) must leave the array
    /// byte-for-byte untouched and still return: refusal happens before the
    /// first swap, never mid-sort.
    #[test]
    fn qsort_without_call_guest_support_leaves_the_array_untouched() {
        const BASE: u64 = 0x100;
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let values: [u64; 4] = [4, 3, 2, 1];
        for (i, v) in values.iter().enumerate() {
            assert!(mem.write(BASE + i as u64 * 8, &v.to_le_bytes()));
        }
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_qsort(&ctx, &[BASE, 4, 8, 0x9000, 0, 0]), 0);

        for (i, v) in values.iter().enumerate() {
            let mut bytes = [0u8; 8];
            assert!(mem.read(BASE + i as u64 * 8, &mut bytes));
            assert_eq!(
                u64::from_le_bytes(bytes),
                *v,
                "refused dispatch must not have moved element {i}"
            );
        }
    }

    /// Degenerate inputs are C-standard no-ops, and a null comparator is
    /// refused without touching memory.
    #[test]
    fn qsort_degenerate_inputs_are_no_ops() {
        const BASE: u64 = 0x100;
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        assert!(mem.write(BASE, &7u64.to_le_bytes()));
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert_eq!(hle_qsort(&ctx, &[BASE, 0, 8, 0x9000, 0, 0]), 0); // nmemb 0
        assert_eq!(hle_qsort(&ctx, &[BASE, 1, 8, 0x9000, 0, 0]), 0); // nmemb 1
        assert_eq!(hle_qsort(&ctx, &[BASE, 4, 0, 0x9000, 0, 0]), 0); // size 0
        assert_eq!(hle_qsort(&ctx, &[BASE, 4, 8, 0, 0, 0]), 0); // null comparator
        assert_eq!(hle_qsort(&ctx, &[BASE, u64::MAX, 8, 0x9000, 0, 0]), 0); // overflow

        let mut bytes = [0u8; 8];
        assert!(mem.read(BASE, &mut bytes));
        assert_eq!(u64::from_le_bytes(bytes), 7, "no degenerate call may write");
    }

    /// M1-C: `printf` reads the guest format string and `%s` pointee, formats
    /// against the captured registers, and lands the output in the kernel
    /// console — the observable-stdout contract.
    /// Heap poison fills exactly the requested span with `0xCD` so an
    /// uninitialized read is visible. Exercises the fill mechanism directly
    /// (the `RAEEN_POISON_HEAP` gate is a cached `OnceLock`, unfriendly to a
    /// per-test env toggle); poisoning is just `mem_fill(_, 0xCD, size)`.
    #[test]
    fn heap_poison_fills_the_requested_span_with_cd() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let size = 0x20u64;
        let addr = ctx.alloc.alloc(size, 16).expect("test alloc");
        assert!(mem_fill(&ctx, addr, HEAP_ALLOC_POISON, size));

        let mut buf = vec![0u8; size as usize];
        assert!(mem.read(addr, &mut buf));
        assert!(
            buf.iter().all(|&b| b == 0xCD),
            "every poisoned byte reads back 0xCD"
        );
        // The byte just past the requested span is untouched (fill is exact).
        let mut tail = [0u8; 1];
        assert!(mem.read(addr + size, &mut tail));
        assert_eq!(tail[0], 0, "poison does not spill past `size`");
    }

    /// `calloc` still zeroes after the poison change — a regression guard that
    /// the poison path did not leak into the zero-on-allocate contract.
    #[test]
    fn calloc_still_zeroes_the_block() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let addr = hle_calloc(&ctx, &[4, 8]); // 32 bytes
        assert_ne!(addr, 0);
        let mut buf = [0xFFu8; 32];
        assert!(mem.read(addr, &mut buf));
        assert!(buf.iter().all(|&b| b == 0), "calloc zeroes its block");
    }

    /// `__stack_chk_fail` is `noreturn` on hardware: the handler must unwind
    /// the calling guest thread (via `request_exit`, which the dispatcher
    /// turns into a recovery-context restore) rather than hand a return value
    /// back to a smashed frame. The old stub returned 0, so the guest walked
    /// off the call site into UD2 and the smash was misreported as a wild
    /// jump (measured: Until Dawn after getdents on /app0/deepfiles).
    #[test]
    fn stack_chk_fail_requests_guest_thread_exit_instead_of_returning() {
        use crate::{
            GpuSubmissionSubsystem, GuestCallRequest, GuestCallScheduler, GuestThreadScheduler,
            HleContext, STACK_CHK_FAIL_EXIT_CODE,
        };

        struct NoGpu;
        impl GpuSubmissionSubsystem for NoGpu {
            fn submit(&self, _words: Vec<u32>, _queue: raeen_core::subsystems::GpuQueue) {}
            fn map_shader_metadata(
                &self,
                _code_address: u64,
                _data: raeen_core::subsystems::ShaderMappedData,
            ) {
            }
            fn present_scanout(
                &self,
                _address: u64,
                _descriptor: Option<raeen_core::subsystems::ScanoutDescriptor>,
            ) {
            }
            fn wait_idle(&self) {}
            fn stats(&self) -> raeen_core::subsystems::GpuSubmissionStats {
                raeen_core::subsystems::GpuSubmissionStats::default()
            }
        }
        struct NoGuestCalls;
        impl GuestCallScheduler for NoGuestCalls {
            fn request(&self, _request: GuestCallRequest) -> bool {
                false
            }
        }
        /// Records the exit request the handler must make.
        struct RecordingThreads {
            exit_requested: std::cell::Cell<Option<u64>>,
        }
        impl GuestThreadScheduler for RecordingThreads {
            fn create(&self, _thread_out: u64, _attr: u64, _entry: u64, _arg: u64) -> u64 {
                0x8002_000B
            }
            fn join(&self, _thread: u64, _retval_out: u64) -> u64 {
                0x8002_0003
            }
            fn detach(&self, _thread: u64) -> u64 {
                0x8002_0003
            }
            fn request_exit(&self, retval: u64) -> bool {
                self.exit_requested.set(Some(retval));
                true
            }
            fn current_thread(&self) -> u64 {
                1
            }
            fn request_process_exit(&self, _code: u64) {
                panic!("with a working thread scheduler the handler must not escalate");
            }
            fn process_is_terminating(&self) -> bool {
                false
            }
        }

        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let gpu = NoGpu;
        let guest_calls = NoGuestCalls;
        let threads = RecordingThreads {
            exit_requested: std::cell::Cell::new(None),
        };
        let ctx = HleContext {
            kernel: &kernel,
            services: &kernel,
            gpu: &gpu,
            mem: &mem,
            alloc: &alloc,
            guest_calls: &guest_calls,
            guest_threads: &threads,
            caller_return_addr: 0,
            caller_rsp: 0,
            float_args: [0; 8],
            caller_gprs: None,
        };

        let ret = hle_stack_chk_fail(&ctx, &[]);
        assert_eq!(
            threads.exit_requested.get(),
            Some(STACK_CHK_FAIL_EXIT_CODE),
            "the handler must unwind the guest thread with the fatal exit code"
        );
        assert_ne!(
            ret, 0,
            "the (never-delivered) return value must not look like success"
        );
    }

    /// Test doubles for the noreturn-handler tests (`abort`, `exit`): a GPU
    /// subsystem and guest-call scheduler that do nothing, plus a thread
    /// scheduler that records the exit requests the handlers must make.
    struct FatalNoGpu;
    impl crate::GpuSubmissionSubsystem for FatalNoGpu {
        fn submit(&self, _words: Vec<u32>, _queue: raeen_core::subsystems::GpuQueue) {}
        fn map_shader_metadata(
            &self,
            _code_address: u64,
            _data: raeen_core::subsystems::ShaderMappedData,
        ) {
        }
        fn present_scanout(
            &self,
            _address: u64,
            _descriptor: Option<raeen_core::subsystems::ScanoutDescriptor>,
        ) {
        }
        fn wait_idle(&self) {}
        fn stats(&self) -> raeen_core::subsystems::GpuSubmissionStats {
            raeen_core::subsystems::GpuSubmissionStats::default()
        }
    }
    struct FatalNoGuestCalls;
    impl crate::GuestCallScheduler for FatalNoGuestCalls {
        fn request(&self, _request: crate::GuestCallRequest) -> bool {
            false
        }
    }
    /// Records both the per-thread unwind and any process-exit request.
    struct FatalRecordingThreads {
        exit_requested: std::cell::Cell<Option<u64>>,
        process_exit_requested: std::cell::Cell<Option<u64>>,
    }
    impl FatalRecordingThreads {
        fn new() -> Self {
            Self {
                exit_requested: std::cell::Cell::new(None),
                process_exit_requested: std::cell::Cell::new(None),
            }
        }
    }
    impl crate::GuestThreadScheduler for FatalRecordingThreads {
        fn create(&self, _thread_out: u64, _attr: u64, _entry: u64, _arg: u64) -> u64 {
            0x8002_000B
        }
        fn join(&self, _thread: u64, _retval_out: u64) -> u64 {
            0x8002_0003
        }
        fn detach(&self, _thread: u64) -> u64 {
            0x8002_0003
        }
        fn request_exit(&self, retval: u64) -> bool {
            self.exit_requested.set(Some(retval));
            true
        }
        fn current_thread(&self) -> u64 {
            1
        }
        fn request_process_exit(&self, code: u64) {
            self.process_exit_requested.set(Some(code));
        }
        fn process_is_terminating(&self) -> bool {
            false
        }
    }
    fn fatal_test_ctx<'a>(
        kernel: &'a raeen_kernel::OrbisKernel,
        mem: &'a crate::TestMemory,
        alloc: &'a crate::TestAllocator,
        gpu: &'a FatalNoGpu,
        guest_calls: &'a FatalNoGuestCalls,
        threads: &'a FatalRecordingThreads,
    ) -> crate::HleContext<'a> {
        crate::HleContext {
            kernel,
            services: kernel,
            gpu,
            mem,
            alloc,
            guest_calls,
            guest_threads: threads,
            caller_return_addr: 0,
            caller_rsp: 0,
            float_args: [0; 8],
            caller_gprs: None,
        }
    }

    /// `abort()` is `noreturn` on hardware. Like `__stack_chk_fail`, the
    /// handler must unwind the calling guest thread with its own fatal code
    /// (distinct from the canary-smash code) rather than return 0 into a
    /// frame the compiler compiled as never resuming.
    #[test]
    fn abort_requests_guest_thread_exit_instead_of_returning() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let gpu = FatalNoGpu;
        let guest_calls = FatalNoGuestCalls;
        let threads = FatalRecordingThreads::new();
        let ctx = fatal_test_ctx(&kernel, &mem, &alloc, &gpu, &guest_calls, &threads);

        let ret = hle_abort(&ctx, &[]);
        assert_eq!(
            threads.exit_requested.get(),
            Some(crate::ABORT_EXIT_CODE),
            "abort must unwind the guest thread with the abort fatal code"
        );
        assert_eq!(
            threads.process_exit_requested.get(),
            None,
            "with a working per-thread unwind, abort must not escalate to process exit"
        );
        assert_ne!(
            crate::ABORT_EXIT_CODE,
            crate::STACK_CHK_FAIL_EXIT_CODE,
            "a deliberate abort must be distinguishable from a canary smash"
        );
        assert_ne!(
            ret, 0,
            "the (never-delivered) return value must not look like success"
        );
    }

    /// `exit(status)` is `noreturn` on hardware and terminates the whole
    /// process with the guest's own status: the handler must record the
    /// status process-wide (so every worker stops at its next safe point)
    /// AND unwind the calling thread immediately — carrying `status`, not a
    /// fatal-family code.
    #[test]
    fn exit_requests_process_and_thread_exit_with_the_guest_status() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let gpu = FatalNoGpu;
        let guest_calls = FatalNoGuestCalls;
        let threads = FatalRecordingThreads::new();
        let ctx = fatal_test_ctx(&kernel, &mem, &alloc, &gpu, &guest_calls, &threads);

        let ret = hle_exit(&ctx, &[42]);
        assert_eq!(
            threads.process_exit_requested.get(),
            Some(42),
            "exit must request process termination with the guest's own status"
        );
        assert_eq!(
            threads.exit_requested.get(),
            Some(42),
            "exit must unwind the calling guest thread with the guest's own status"
        );
        assert_eq!(ret, 42, "the (never-delivered) value carries the status");
    }

    #[test]
    fn printf_formats_guest_strings_into_the_kernel_console() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"hello %s, %d + %d = %d\n\0"));
        assert!(mem.write(0x200, b"world\0"));

        let written = hle_printf(&ctx, &[0x100, 0x200, 2, 3, 5]);
        assert_eq!(kernel.console.contents(), "hello world, 2 + 3 = 5\n");
        assert_eq!(written, "hello world, 2 + 3 = 5\n".len() as u64);
    }

    /// `sincosf` takes its input in XMM0, which reaches a handler through
    /// `HleContext::float_args` rather than the integer slice. `test_ctx` zeroes
    /// that channel, so `x == 0.0` — which makes the expected results exact
    /// (`sin 0 = 0`, `cos 0 = 1`) and proves both out-params are written through
    /// the right pointers, in the right order.
    #[test]
    fn sincosf_reads_the_float_channel_and_writes_both_out_params() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        const SIN_OUT: u64 = 0x400;
        const COS_OUT: u64 = 0x410;

        // Poison both slots so a missing write is visible.
        assert!(mem.write(SIN_OUT, &0xDEAD_BEEFu32.to_le_bytes()));
        assert!(mem.write(COS_OUT, &0xDEAD_BEEFu32.to_le_bytes()));

        assert_eq!(
            ctx.float_arg_f32(0),
            0.0,
            "test_ctx zeroes the float channel"
        );
        hle_sincosf(&ctx, &[SIN_OUT, COS_OUT]);

        let mut buf = [0u8; 4];
        assert!(mem.read(SIN_OUT, &mut buf));
        assert_eq!(f32::from_le_bytes(buf), 0.0, "sin(0) written to arg0");
        assert!(mem.read(COS_OUT, &mut buf));
        assert_eq!(f32::from_le_bytes(buf), 1.0, "cos(0) written to arg1");

        // Null out-pointers must not fault.
        hle_sincosf(&ctx, &[0, 0]);
    }

    /// The Itanium C++ static-guard contract: the first caller is told to run
    /// the initializer (1), and once it releases, every later caller is told to
    /// skip it (0). Getting this backwards double-constructs C++ statics.
    #[test]
    fn cxa_guard_runs_a_static_initializer_exactly_once() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        const GUARD: u64 = 0x300;

        assert!(mem.write(GUARD, &[0u8; 8]), "guard starts zeroed");
        assert_eq!(
            hle_cxa_guard_acquire(&ctx, &[GUARD]),
            1,
            "the first caller must run the initializer"
        );
        hle_cxa_guard_release(&ctx, &[GUARD]);
        assert_eq!(
            hle_cxa_guard_acquire(&ctx, &[GUARD]),
            0,
            "after release the static is constructed; later callers must skip"
        );

        // An aborted initializer leaves the static unconstructed, so the next
        // caller must be told to retry it.
        const RETRY: u64 = 0x320;
        assert!(mem.write(RETRY, &[0u8; 8]));
        assert_eq!(hle_cxa_guard_acquire(&ctx, &[RETRY]), 1);
        hle_cxa_guard_abort(&ctx, &[RETRY]);
        assert_eq!(
            hle_cxa_guard_acquire(&ctx, &[RETRY]),
            1,
            "an aborted construction must be retried, not skipped"
        );

        // A null guard is not a crash.
        assert_eq!(hle_cxa_guard_acquire(&ctx, &[0]), 0);
    }

    #[test]
    fn puts_appends_the_mandated_newline() {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
    fn huge_strncpy_and_calloc_lengths_fail_without_host_allocation() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x800);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x100, b"abc\0"));
        assert!(mem.write(0x200, &[0xAA; 4]));

        assert_eq!(hle_strncpy(&ctx, &[0x200, 0x100, u64::MAX]), 0x200);
        let mut unchanged = [0u8; 4];
        assert!(mem.read(0x200, &mut unchanged));
        assert_eq!(unchanged, [0xAA; 4]);
        assert_eq!(hle_calloc(&ctx, &[1, crate::MAX_HLE_BULK_BYTES + 1]), 0);
    }

    /// M1 hardening batch: the string/buffer functions do real guest-memory
    /// work — compare, scan, concatenate — not lie.
    #[test]
    fn string_and_buffer_batch_do_real_work() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // memcmp / memchr
        assert!(mem.write(0x100, b"abcd"));
        assert!(mem.write(0x110, b"abce"));
        assert_eq!(hle_memcmp(&ctx, &[0x100, 0x110, 4]) as u32 as i32, -1);
        assert_eq!(
            hle_memcmp(&ctx, &[0x100, 0x110, 3]),
            0,
            "first 3 bytes equal"
        );
        assert_eq!(hle_memchr(&ctx, &[0x100, b'c' as u64, 4]), 0x102);
        assert_eq!(
            hle_memchr(&ctx, &[0x100, b'z' as u64, 4]),
            0,
            "not found → NULL"
        );

        // strncmp / strnlen
        assert!(mem.write(0x200, b"hello\0"));
        assert!(mem.write(0x210, b"help\0"));
        assert_eq!(hle_strncmp(&ctx, &[0x200, 0x210, 3]), 0, "first 3 equal");
        assert_eq!(
            hle_strncmp(&ctx, &[0x200, 0x210, 4]) as u32 as i32,
            -1,
            "'l' < 'p'"
        );
        assert_eq!(hle_strnlen(&ctx, &[0x200, 100]), 5);
        assert_eq!(hle_strnlen(&ctx, &[0x200, 3]), 3, "capped at maxlen");

        // strchr / strrchr
        assert!(mem.write(0x300, b"a/b/c\0"));
        assert_eq!(hle_strchr(&ctx, &[0x300, b'/' as u64]), 0x301);
        assert_eq!(hle_strrchr(&ctx, &[0x300, b'/' as u64]), 0x303);
        assert_eq!(hle_strchr(&ctx, &[0x300, 0]), 0x305, "c==0 matches the NUL");
        assert_eq!(hle_strchr(&ctx, &[0x300, b'z' as u64]), 0);

        // strstr
        assert!(mem.write(0x400, b"foobarbaz\0"));
        assert!(mem.write(0x420, b"bar\0"));
        assert_eq!(hle_strstr(&ctx, &[0x400, 0x420]), 0x403);
        assert!(mem.write(0x430, b"\0"));
        assert_eq!(
            hle_strstr(&ctx, &[0x400, 0x430]),
            0x400,
            "empty needle → haystack"
        );

        // strcat / strncat
        assert!(mem.write(0x500, b"foo\0"));
        assert!(mem.write(0x520, b"bar\0"));
        assert_eq!(hle_strcat(&ctx, &[0x500, 0x520]), 0x500);
        let mut buf = [0u8; 7];
        assert!(mem.read(0x500, &mut buf));
        assert_eq!(&buf, b"foobar\0");

        assert!(mem.write(0x600, b"x\0"));
        assert!(mem.write(0x620, b"yzABC\0"));
        assert_eq!(hle_strncat(&ctx, &[0x600, 0x620, 2]), 0x600);
        let mut buf2 = [0u8; 4];
        assert!(mem.read(0x600, &mut buf2));
        assert_eq!(&buf2, b"xyz\0", "only 2 src bytes appended + NUL");

        // strncat with n >= src.len() appends the whole source.
        assert!(mem.write(0x700, b"p\0"));
        assert!(mem.write(0x720, b"qr\0"));
        assert_eq!(hle_strncat(&ctx, &[0x700, 0x720, 10]), 0x700);
        let mut buf3 = [0u8; 4];
        assert!(mem.read(0x700, &mut buf3));
        assert_eq!(&buf3, b"pqr\0");
    }

    /// The comparison functions' positive/Greater branch and the miss/
    /// degradation contracts the batch documents (reviewer-requested gaps).
    #[test]
    fn compare_positive_branches_miss_and_unreadable_degradation() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"abd"));
        assert!(mem.write(0x110, b"abc"));
        assert_eq!(hle_memcmp(&ctx, &[0x100, 0x110, 3]), 1, "'d' > 'c' → +1");
        assert_eq!(hle_memcmp(&ctx, &[0x100, 0x110, 0]), 0, "n==0 → equal");

        assert!(mem.write(0x200, b"help\0"));
        assert!(mem.write(0x210, b"hello\0"));
        assert_eq!(hle_strncmp(&ctx, &[0x200, 0x210, 4]), 1, "'p' > 'l' → +1");

        // strstr not-found → 0
        assert!(mem.write(0x300, b"foobar\0"));
        assert!(mem.write(0x320, b"xyz\0"));
        assert_eq!(hle_strstr(&ctx, &[0x300, 0x320]), 0);

        // Unreadable-pointer degradations (documented): comparisons report
        // 0 (equal); scans report 0 (NULL); no panic, no host OOB.
        assert_eq!(hle_memcmp(&ctx, &[0xDEAD_0000, 0x100, 4]), 0);
        assert_eq!(hle_memchr(&ctx, &[0xDEAD_0000, b'a' as u64, 4]), 0);
        assert_eq!(hle_strncmp(&ctx, &[0xDEAD_0000, 0x200, 4]), 0);
        assert_eq!(hle_strchr(&ctx, &[0xDEAD_0000, b'a' as u64]), 0);
        assert_eq!(hle_strstr(&ctx, &[0xDEAD_0000, 0x320]), 0);
        assert_eq!(hle_strnlen(&ctx, &[0xDEAD_0000, 8]), 0);
    }

    /// M1 hardening: string→integer parsing does real conversion with base
    /// detection, sign, endptr, and saturation.
    #[test]
    fn atoi_atol_strtol_parse_real_integers() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"  -42abc\0"));
        assert_eq!(
            hle_atoi(&ctx, &[0x100]) as u32 as i32,
            -42,
            "skips ws, sign, stops at 'a'"
        );

        assert!(mem.write(0x120, b"9000000000\0")); // > i32, fits i64
        assert_eq!(hle_atol(&ctx, &[0x120]) as i64, 9_000_000_000);

        // strtol base 16 with 0x prefix + endptr.
        assert!(mem.write(0x140, b"0x1F!\0"));
        assert_eq!(hle_strtol(&ctx, &[0x140, 0x200, 16]) as i64, 0x1F);
        let mut ep = [0u8; 8];
        assert!(mem.read(0x200, &mut ep));
        assert_eq!(u64::from_le_bytes(ep), 0x140 + 4, "endptr points at '!'");

        // base 0 auto-detect: octal.
        assert!(mem.write(0x160, b"010\0"));
        assert_eq!(hle_strtol(&ctx, &[0x160, 0, 0]) as i64, 8);

        // strtoul clamps a negative to a large unsigned, and parses big values.
        assert!(mem.write(0x180, b"4294967295\0"));
        assert_eq!(hle_strtoul(&ctx, &[0x180, 0, 10]), 4_294_967_295);

        // No digits → 0, endptr at start.
        assert!(mem.write(0x1A0, b"xyz\0"));
        assert_eq!(hle_strtol(&ctx, &[0x1A0, 0x220, 10]), 0);
        let mut ep2 = [0u8; 8];
        assert!(mem.read(0x220, &mut ep2));
        assert_eq!(
            u64::from_le_bytes(ep2),
            0x1A0,
            "endptr == nptr when nothing converts"
        );
    }

    #[test]
    fn atexit_family_registers_and_succeeds() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_atexit(&ctx, &[0x1234]), 0);
        assert_eq!(hle_cxa_atexit(&ctx, &[0x1234, 0, 0]), 0);
    }

    #[test]
    fn register_adds_expected_functions() {
        let registry = HleRegistry::new();
        let kernel = raeen_kernel::OrbisKernel::new();
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
            "memcmp",
            "memchr",
            "strncmp",
            "strnlen",
            "strchr",
            "strrchr",
            "strcat",
            "strncat",
            "strstr",
            "atoi",
            "atol",
            "strtol",
            "strtoul",
            "atexit",
            "__cxa_atexit",
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
            assert!(
                registry.is_implemented("libSceLibcInternal", name),
                "missing libSceLibcInternal::{name} ABI alias"
            );
            registry.call(&ctx, "libc", name, &[1, 2, 3]);
        }

        // Retail modules import the stack-protector abort naming libkernel —
        // same name-hash NID, provider-aware resolution.
        assert!(
            registry.is_implemented("libkernel", "__stack_chk_fail"),
            "missing libkernel::__stack_chk_fail provider alias"
        );
    }

    #[test]
    fn memcpy_actually_moves_bytes_in_guest_memory() {
        let registry = HleRegistry::new();
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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
        let kernel = raeen_kernel::OrbisKernel::new();
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

    // ------------------------------------------------------------------
    // NID-fill batch tests.
    // ------------------------------------------------------------------

    /// The ctype tables carry the Dinkumware C-locale layout: category bits
    /// at `table[c]`, EOF(-1) handled, case tables mapping A-Z/a-z.
    #[test]
    fn ctype_tables_classify_and_map_the_c_locale() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let pctype = hle_getpctype(&ctx, &[]);
        assert_ne!(pctype, 0);
        let entry = |c: i64| {
            let mut b = [0u8; 2];
            assert!(mem.read((pctype as i64 + c * 2) as u64, &mut b));
            u16::from_le_bytes(b)
        };
        assert_eq!(entry(0x41), CTYPE_UP | CTYPE_XD, "'A'");
        assert_eq!(entry(0x61), CTYPE_LO | CTYPE_XD, "'a'");
        assert_eq!(entry(0x35), CTYPE_DI | CTYPE_XD, "'5'");
        assert_eq!(entry(0x20), CTYPE_SP | CTYPE_XB, "' '");
        assert_eq!(entry(0x09), CTYPE_CN | CTYPE_XB, "'\\t'");
        assert_eq!(entry(0x2E), CTYPE_PU, "'.' is punctuation");
        assert_eq!(entry(0x01), CTYPE_BB, "SOH is plain control");
        assert_eq!(entry(-1), 0, "EOF has no category bits");
        assert_eq!(entry(0x80), 0, "beyond ASCII: zero");

        let ptolower = hle_getptolower(&ctx, &[]);
        let ptoupper = hle_getptoupper(&ctx, &[]);
        assert_ne!(ptolower, 0);
        assert_ne!(ptoupper, 0);
        let lower = |c: i64| {
            let mut b = [0u8; 2];
            assert!(mem.read((ptolower as i64 + c * 2) as u64, &mut b));
            u16::from_le_bytes(b)
        };
        let upper = |c: i64| {
            let mut b = [0u8; 2];
            assert!(mem.read((ptoupper as i64 + c * 2) as u64, &mut b));
            u16::from_le_bytes(b)
        };
        assert_eq!(lower(0x41), 0x61, "tolower('A') == 'a'");
        assert_eq!(lower(0x35), 0x35, "tolower('5') == '5'");
        assert_eq!(upper(0x61), 0x41, "toupper('a') == 'A'");
        assert_eq!(lower(-1), 0xFFFF, "tolower(EOF) == EOF");
        assert_eq!(upper(-1), 0xFFFF, "toupper(EOF) == EOF");

        // Same process, second call: the cached table comes back unchanged.
        assert_eq!(hle_getpctype(&ctx, &[]), pctype);
    }

    /// `_Atomic_fetch_add_4`/`_Atomic_fetch_sub_4` update the guest word
    /// atomically and return the OLD value.
    #[test]
    fn atomic_fetch_add_and_sub_update_the_guest_word() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x20, &10u32.to_le_bytes()));
        assert_eq!(hle_atomic_fetch_add_4(&ctx, &[0x20, 5, 6]), 10);
        assert_eq!(hle_atomic_fetch_sub_4(&ctx, &[0x20, 3, 6]), 15);
        let mut word = [0u8; 4];
        assert!(mem.read(0x20, &mut word));
        assert_eq!(u32::from_le_bytes(word), 12);
        // An unmapped word reports 0 instead of faulting.
        assert_eq!(hle_atomic_fetch_add_4(&ctx, &[0xDEAD_0000, 1, 6]), 0);
    }

    /// The stdio family writes real bytes to the kernel console through the
    /// FILE objects `_Stdout`/`_Stderr` hand out, and rejects foreign FILEs.
    #[test]
    fn stdio_streams_route_real_bytes_to_the_console() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x300);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let stdout = hle_stdout(&ctx, &[]);
        assert_ne!(stdout, 0);
        assert!(mem.write(0x100, b"hello\0"));

        assert_eq!(hle_fputs(&ctx, &[0x100, stdout]), 1);
        assert_eq!(hle_fputc(&ctx, &[b'!' as u64, stdout]), b'!' as u64);
        assert!(mem.write(0x110, b"abab"));
        assert_eq!(hle_fwrite(&ctx, &[0x110, 2, 2, stdout]), 2);
        assert_eq!(hle_fflush(&ctx, &[stdout]), 0);
        assert_eq!(hle_fflush(&ctx, &[0]), 0, "fflush(NULL) flushes all");

        // vfprintf with a guest va_list: all GP slots spilled, one arg at
        // the overflow area.
        assert!(mem.write(0x120, b" %d\0"));
        let mut va = [0u8; 24];
        va[0..4].copy_from_slice(&48u32.to_le_bytes()); // gp_offset
        va[4..8].copy_from_slice(&176u32.to_le_bytes()); // fp_offset
        va[8..16].copy_from_slice(&0x600u64.to_le_bytes()); // overflow_arg_area
        va[16..24].copy_from_slice(&0x700u64.to_le_bytes()); // reg_save_area
        assert!(mem.write(0x500, &va));
        assert!(mem.write(0x600, &42u64.to_le_bytes()));
        assert_eq!(hle_vfprintf(&ctx, &[stdout, 0x120, 0x500]), 3);

        assert_eq!(kernel.console.contents(), "hello!abab 42");

        // stderr is a distinct object that also reaches the console.
        let stderr = hle_stderr(&ctx, &[]);
        assert_ne!(stderr, 0);
        assert_ne!(stderr, stdout);
        assert_eq!(hle_fputs(&ctx, &[0x100, stderr]), 1);

        // A foreign FILE* is refused, not dereferenced.
        assert_eq!(hle_fputs(&ctx, &[0x100, 0x80]), u64::MAX);
        assert_eq!(hle_fputc(&ctx, &[b'x' as u64, 0x80]), u64::MAX);
        assert_eq!(hle_fflush(&ctx, &[0x80]), u64::MAX);
        assert_eq!(hle_vfprintf(&ctx, &[0x80, 0x120, 0x500]), u64::MAX);
    }

    /// `sprintf`/`sprintf_s` format through the real engine; the `_s` variant
    /// enforces its Annex K size contract (negative + NUL on truncation).
    #[test]
    fn sprintf_family_formats_and_enforces_size_contracts() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"%d-%s\0"));
        assert!(mem.write(0x120, b"x\0"));
        assert_eq!(hle_sprintf(&ctx, &[0x200, 0x100, 7, 0x120]), 3);
        let mut buf = [0u8; 4];
        assert!(mem.read(0x200, &mut buf));
        assert_eq!(&buf, b"7-x\0");

        // sprintf_s success path.
        assert_eq!(hle_sprintf_s(&ctx, &[0x300, 16, 0x100, 7, 0x120]), 3);
        assert!(mem.read(0x300, &mut buf));
        assert_eq!(&buf, b"7-x\0");

        // sprintf_s truncation: negative result, buf[0] = NUL.
        assert_eq!(hle_sprintf_s(&ctx, &[0x300, 3, 0x100, 7, 0x120]), u64::MAX);
        let mut nul = [0u8; 1];
        assert!(mem.read(0x300, &mut nul));
        assert_eq!(nul[0], 0);
    }

    /// `vswprintf` formats wide output (32-bit units) and reports -1 when
    /// the output would reach `n` (the C swprintf contract).
    #[test]
    fn vswprintf_formats_wide_and_reports_truncation() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Wide fmt "n=%d" and a va_list with one spilled GP argument.
        let fmt_units: Vec<u8> = [b'n' as u32, b'=' as u32, b'%' as u32, b'd' as u32, 0]
            .iter()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        assert!(mem.write(0x100, &fmt_units));
        let mut va = [0u8; 24];
        va[0..4].copy_from_slice(&48u32.to_le_bytes());
        va[4..8].copy_from_slice(&176u32.to_le_bytes());
        va[8..16].copy_from_slice(&0x600u64.to_le_bytes());
        va[16..24].copy_from_slice(&0x700u64.to_le_bytes());
        assert!(mem.write(0x500, &va));
        assert!(mem.write(0x600, &42u64.to_le_bytes()));

        assert_eq!(hle_vswprintf(&ctx, &[0x800, 16, 0x100, 0x500]), 4);
        let mut out = [0u8; 20];
        assert!(mem.read(0x800, &mut out));
        let units: Vec<u32> = out
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4-byte chunks")))
            .collect();
        assert_eq!(
            units,
            vec![b'n' as u32, b'=' as u32, b'4' as u32, b'2' as u32, 0]
        );

        // "n=42" is 4 units; n=4 cannot hold it plus the NUL: -1 + prefix.
        assert_eq!(hle_vswprintf(&ctx, &[0x800, 4, 0x100, 0x500]), u64::MAX);
        assert!(mem.read(0x800, &mut out[..16]));
        let units: Vec<u32> = out[..16]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4-byte chunks")))
            .collect();
        assert_eq!(units, vec![b'n' as u32, b'=' as u32, b'4' as u32, 0]);
    }

    /// `sscanf` parses the common conversions and stores through guest
    /// pointers with the width each length modifier promises.
    #[test]
    fn sscanf_parses_and_stores_through_guest_pointers() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"42 0x1F hello 3.5\0"));
        assert!(mem.write(0x120, b"%d %i %s %f %n\0"));
        let assigned = hle_sscanf(&ctx, &[0x100, 0x120, 0x200, 0x204, 0x208, 0x210, 0x218]);
        assert_eq!(assigned, 4, "%n does not count as an assignment");
        let mut v32 = [0u8; 4];
        assert!(mem.read(0x200, &mut v32));
        assert_eq!(i32::from_le_bytes(v32), 42);
        assert!(mem.read(0x204, &mut v32));
        assert_eq!(i32::from_le_bytes(v32), 31, "%i auto-detects 0x hex");
        let mut s = [0u8; 6];
        assert!(mem.read(0x208, &mut s));
        assert_eq!(&s, b"hello\0");
        assert!(mem.read(0x210, &mut v32));
        assert_eq!(f32::from_le_bytes(v32), 3.5);
        assert!(mem.read(0x218, &mut v32));
        assert_eq!(i32::from_le_bytes(v32), 17, "%%n reports consumed bytes");

        // Suppression, %x, %o, and a matching failure.
        assert!(mem.write(0x300, b"1 2\0"));
        assert!(mem.write(0x310, b"%*d %d\0"));
        assert_eq!(hle_sscanf(&ctx, &[0x300, 0x310, 0x320]), 1);
        assert!(mem.read(0x320, &mut v32));
        assert_eq!(i32::from_le_bytes(v32), 2);

        assert!(mem.write(0x340, b"1f 17\0"));
        assert!(mem.write(0x350, b"%x %o\0"));
        assert_eq!(hle_sscanf(&ctx, &[0x340, 0x350, 0x360, 0x364]), 2);
        assert!(mem.read(0x360, &mut v32));
        assert_eq!(i32::from_le_bytes(v32), 31);
        assert!(mem.read(0x364, &mut v32));
        assert_eq!(i32::from_le_bytes(v32), 15);

        // Matching failure reports 0; input failure before any assignment
        // reports EOF (-1).
        assert!(mem.write(0x380, b"abc\0"));
        assert!(mem.write(0x390, b"%d\0"));
        assert_eq!(hle_sscanf(&ctx, &[0x380, 0x390, 0x3A0]), 0);
        assert!(mem.write(0x3C0, b"\0"));
        assert_eq!(hle_sscanf(&ctx, &[0x3C0, 0x390, 0x3A0]), u64::MAX);
    }

    /// The 64-bit strto family shares the strtol engine (with endptr).
    #[test]
    fn strto_family_parses_64_bit_values() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"18446744073709551615\0"));
        assert_eq!(hle_strtoull(&ctx, &[0x100, 0, 10]), u64::MAX);

        assert!(mem.write(0x120, b"-9223372036854775808\0"));
        assert_eq!(hle_strtoimax(&ctx, &[0x120, 0, 10]), i64::MIN as u64);

        assert!(mem.write(0x140, b"0xFFz\0"));
        assert_eq!(hle_strtoull(&ctx, &[0x140, 0x200, 16]), 0xFF);
        let mut ep = [0u8; 8];
        assert!(mem.read(0x200, &mut ep));
        assert_eq!(u64::from_le_bytes(ep), 0x140 + 4, "endptr at 'z'");
    }

    /// `time`/`gmtime`/`gmtime_s`/`localtime`/`mktime` do real calendar math
    /// (Hinnant civil arithmetic) with the host clock and zone.
    #[test]
    fn time_family_breaks_down_and_rebuilds_dates() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x2000);
        let alloc = crate::TestAllocator::new(0x400);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // time() writes through the out-pointer and returns the same value.
        let now = hle_time(&ctx, &[0x100]);
        let mut secs = [0u8; 8];
        assert!(mem.read(0x100, &mut secs));
        assert_eq!(u64::from_le_bytes(secs), now);
        assert!(now > 1_700_000_000, "host clock is past Nov 2023");

        // gmtime(1_700_000_000) == 2023-11-14 22:13:20 UTC (a Tuesday).
        assert!(mem.write(0x108, &1_700_000_000i64.to_le_bytes()));
        let tm_addr = hle_gmtime(&ctx, &[0x108]);
        assert_ne!(tm_addr, 0);
        let tm = read_tm(&mem, tm_addr).expect("gmtime returns a readable tm");
        assert_eq!(
            (tm.year, tm.mon, tm.mday, tm.hour, tm.min, tm.sec),
            (123, 10, 14, 22, 13, 20)
        );
        assert_eq!((tm.wday, tm.yday), (2, 317), "Tuesday, day 317 (0-based)");
        let mut gmtoff = [0u8; 8];
        assert!(mem.read(tm_addr + 40, &mut gmtoff));
        assert_eq!(i64::from_le_bytes(gmtoff), 0, "UTC has no offset");
        let mut zone_ptr = [0u8; 8];
        assert!(mem.read(tm_addr + 48, &mut zone_ptr));
        let zone_name = crate::fmt::read_cstr(&mem, u64::from_le_bytes(zone_ptr));
        assert_eq!(zone_name.as_deref(), Some(b"UTC".as_slice()));

        // gmtime_s: Annex K order (timer, result) into the caller's struct.
        assert_eq!(hle_gmtime_s(&ctx, &[0x108, 0x180]), 0);
        let tm = read_tm(&mem, 0x180).expect("gmtime_s fills the caller tm");
        assert_eq!((tm.year, tm.mday, tm.hour), (123, 14, 22));

        // localtime carries the host zone offset.
        let local_addr = hle_localtime(&ctx, &[0x108]);
        assert_ne!(local_addr, 0);
        assert!(mem.read(local_addr + 40, &mut gmtoff));
        assert_eq!(i64::from_le_bytes(gmtoff), host_zone().offset_secs);

        // mktime round-trips gmtime's fields (through the local offset), and
        // normalizes out-of-range fields in place.
        let tm = break_down_utc(1_700_000_000);
        assert!(write_tm(&mem, 0x200, &tm));
        let rebuilt = hle_mktime(&ctx, &[0x200]) as i64;
        assert_eq!(rebuilt, 1_700_000_000 - host_zone().offset_secs);

        let mut weird = GuestTm {
            year: 123,
            mon: 13,  // out of range: 2024-02
            mday: 32, // out of range: rolls into March
            ..GuestTm::default()
        };
        weird.hour = 1;
        assert!(write_tm(&mem, 0x240, &weird));
        assert_ne!(hle_mktime(&ctx, &[0x240]), u64::MAX);
        let normalized = read_tm(&mem, 0x240).expect("mktime normalizes in place");
        assert_eq!(
            (normalized.year, normalized.mon, normalized.mday),
            (124, 2, 3),
            "2023 month 13 day 32 normalizes to 2024-03-03"
        );
    }

    /// `strftime` renders the C locale's calendar names and numerics.
    #[test]
    fn strftime_formats_the_c_locale_calendar() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let tm = GuestTm {
            sec: 20,
            min: 13,
            hour: 22,
            mday: 14,
            mon: 10,
            year: 123,
            wday: 2,
            yday: 317,
            isdst: 0,
        };
        assert!(write_tm(&mem, 0x300, &tm));

        assert!(mem.write(0x100, b"%Y-%m-%d %H:%M:%S\0"));
        assert_eq!(hle_strftime(&ctx, &[0x400, 64, 0x100, 0x300]), 19);
        let mut buf = [0u8; 20];
        assert!(mem.read(0x400, &mut buf));
        assert_eq!(&buf, b"2023-11-14 22:13:20\0");

        assert!(mem.write(0x100, b"%a %A %b %B %p %j %y %U %W\0"));
        assert_eq!(hle_strftime(&ctx, &[0x400, 64, 0x100, 0x300]), 40);
        let text = crate::fmt::read_cstr(&mem, 0x400).expect("strftime wrote a string");
        assert_eq!(
            String::from_utf8(text).expect("ASCII"),
            "Tue Tuesday Nov November PM 318 23 46 46"
        );

        assert!(mem.write(0x100, b"%c\0"));
        assert_eq!(hle_strftime(&ctx, &[0x400, 64, 0x100, 0x300]), 24);
        let text = crate::fmt::read_cstr(&mem, 0x400).expect("strftime wrote a string");
        assert_eq!(
            String::from_utf8(text).expect("ASCII"),
            "Tue Nov 14 22:13:20 2023"
        );

        // Overflow: return 0, still a terminated prefix.
        assert!(mem.write(0x100, b"%Y-%m-%d\0"));
        assert_eq!(hle_strftime(&ctx, &[0x400, 5, 0x100, 0x300]), 0);
        let mut buf5 = [0u8; 5];
        assert!(mem.read(0x400, &mut buf5));
        assert_eq!(&buf5, b"2023\0");
    }

    /// `rand` is reproducible per `srand` seed and bounded by RAND_MAX.
    #[test]
    fn rand_is_seeded_reproducible_and_bounded() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        hle_srand(&ctx, &[42]);
        let a = hle_rand(&ctx, &[]);
        hle_srand(&ctx, &[42]);
        let b = hle_rand(&ctx, &[]);
        assert_eq!(a, b, "same seed, same sequence");
        let c = hle_rand(&ctx, &[]);
        assert_ne!(b, c, "the sequence advances without a reseed");
        assert!(c < 0x8000_0000, "within [0, RAND_MAX]");
    }

    /// Dinkumware locks/mutexes do real recursive round-trips; `_Thrd_*` map
    /// onto the guest thread scheduler; `_Xtime_get_ticks` is monotonic.
    #[test]
    fn dinkumware_sync_and_thread_primitives_work() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0x80);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Recursive mutex: lock twice, unwind twice.
        assert_eq!(hle_mtx_init(&ctx, &[0x40, MTX_TYPE_RECURSIVE]), 0);
        assert_eq!(hle_mtx_lock(&ctx, &[0x40]), 0);
        assert_eq!(hle_mtx_lock(&ctx, &[0x40]), 0, "recursive re-entry");
        assert_eq!(hle_mtx_unlock(&ctx, &[0x40]), 0);
        assert_eq!(hle_mtx_unlock(&ctx, &[0x40]), 0);
        assert_eq!(hle_mtx_destroy(&ctx, &[0x40]), 0);
        // Plain mutex round-trip; unknown mutexes are an error, not a crash.
        assert_eq!(hle_mtx_init(&ctx, &[0x48, 0x1]), 0);
        assert_eq!(hle_mtx_lock(&ctx, &[0x48]), 0);
        assert_eq!(hle_mtx_unlock(&ctx, &[0x48]), 0);
        assert_eq!(hle_mtx_destroy(&ctx, &[0x48]), 0);
        assert_eq!(hle_mtx_lock(&ctx, &[0x900]), 1, "never _Mtx_init'd");
        assert_eq!(hle_mtx_unlock(&ctx, &[0x900]), 1);

        // Syslocks: recursive, bounded indexes.
        assert_eq!(hle_locksyslock(&ctx, &[2]), 0);
        assert_eq!(hle_locksyslock(&ctx, &[2]), 0);
        assert_eq!(hle_unlocksyslock(&ctx, &[2]), 0);
        assert_eq!(hle_unlocksyslock(&ctx, &[2]), 0);
        assert_eq!(hle_locksyslock(&ctx, &[99]), 0, "out-of-range is ignored");

        // Thread identity/sleep/join.
        assert_eq!(hle_thrd_id(&ctx, &[]), 1, "the test scheduler's thread");
        let mut ts = [0u8; 16];
        assert!(mem.write(0x60, &ts));
        assert_eq!(hle_thrd_sleep(&ctx, &[0x60]), 0);
        ts = [0xFF; 16];
        let _ = ts;
        assert_eq!(hle_thrd_sleep(&ctx, &[0xDEAD_0000]), u64::MAX);
        assert_eq!(
            hle_thrd_join(&ctx, &[0xABC, 0]),
            1,
            "the test scheduler's join error maps to a Dinkumware failure"
        );

        // Monotonic, wall-anchored 100ns ticks.
        let t1 = hle_xtime_get_ticks(&ctx, &[]);
        let t2 = hle_xtime_get_ticks(&ctx, &[]);
        assert!(t2 >= t1, "monotonic");
        assert!(t1 > 17_000_000_000_000_000, "wall-anchored past 2023");
    }

    /// The throw-path helpers log and return the defined failure (0);
    /// exception-object storage is a real allocation; `_Assert` prints.
    #[test]
    fn throw_helpers_and_assert_fail_loudly_but_defined() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"vector too long\0"));
        assert_eq!(hle_std_xlength_error(&ctx, &[0x100]), 0);
        assert_eq!(hle_std_xout_of_range(&ctx, &[0x100]), 0);
        assert_eq!(hle_std_xbad_alloc(&ctx, &[]), 0);
        assert_eq!(hle_std_xbad_function_call(&ctx, &[]), 0);
        assert_eq!(hle_std_terminate(&ctx, &[]), 0);
        assert_eq!(hle_std_throw_c_error(&ctx, &[22]), 0);
        assert_eq!(hle_std_throw_cpp_error(&ctx, &[1]), 0);
        assert_eq!(hle_std_exception_raise(&ctx, &[0x300]), 0);
        assert_eq!(hle_std_exception_doraise(&ctx, &[0x300]), 0);
        assert_eq!(hle_cxa_bad_cast(&ctx, &[]), 0);
        assert_eq!(hle_std_uncaught_exception(&ctx, &[]), 0);
        assert_eq!(hle_cxa_thread_atexit(&ctx, &[0x1234, 0, 0]), 0);

        // Exception-object storage is a real allocation.
        let exc = hle_cxa_allocate_exception(&ctx, &[64]);
        assert_ne!(exc, 0);
        hle_cxa_free_exception(&ctx, &[exc]);

        // crt0's return-from-main bridge returns the defined value (the test
        // scheduler records no real exit).
        assert_eq!(hle_catch_return_from_main(&ctx, &[7]), 0);

        // _Assert prints its message to the console and returns 0.
        assert!(mem.write(0x200, b"x > 0\0"));
        assert!(mem.write(0x220, b"main.c\0"));
        assert_eq!(hle_dinkum_assert(&ctx, &[0x200, 0x220, 42]), 0);
        assert!(
            kernel
                .console
                .contents()
                .contains("Assertion failed: x > 0, file main.c, line 42"),
            "the assert message reaches the console"
        );
    }

    /// `__isnan`/`__isfinite` read the XMM channel and answer in RAX;
    /// `sincos` writes both f64 out-params.
    #[test]
    fn isnan_isfinite_and_sincos_use_the_float_channel() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let mut ctx = test_ctx(&kernel, &mem, &alloc);

        ctx.float_args[0] = f64::NAN.to_bits();
        assert_eq!(hle_isnan(&ctx, &[]), 1);
        assert_eq!(hle_isfinite(&ctx, &[]), 0);
        ctx.float_args[0] = f64::INFINITY.to_bits();
        assert_eq!(hle_isnan(&ctx, &[]), 0);
        assert_eq!(hle_isfinite(&ctx, &[]), 0);
        ctx.float_args[0] = 1.5f64.to_bits();
        assert_eq!(hle_isnan(&ctx, &[]), 0);
        assert_eq!(hle_isfinite(&ctx, &[]), 1);

        ctx.float_args[0] = u64::from(f32::NAN.to_bits());
        assert_eq!(hle_isnanf(&ctx, &[]), 1);
        ctx.float_args[0] = u64::from(2.5f32.to_bits());
        assert_eq!(hle_isfinitef(&ctx, &[]), 1);

        // sincos(0): sin 0 = 0, cos 0 = 1, as f64 writes.
        ctx.float_args[0] = 0.0f64.to_bits();
        hle_sincos(&ctx, &[0x400, 0x410]);
        let mut buf = [0u8; 8];
        assert!(mem.read(0x400, &mut buf));
        assert_eq!(f64::from_le_bytes(buf), 0.0);
        assert!(mem.read(0x410, &mut buf));
        assert_eq!(f64::from_le_bytes(buf), 1.0);
        hle_sincos(&ctx, &[0, 0]); // null out-pointers must not fault
    }

    /// `malloc_usable_size` reports the tracked requested size of live
    /// malloc-family blocks and 0 after free.
    #[test]
    fn malloc_usable_size_reports_tracked_blocks() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let a = hle_malloc(&ctx, &[24]);
        assert_ne!(a, 0);
        assert_eq!(hle_malloc_usable_size(&ctx, &[a]), 24);

        let b = hle_realloc(&ctx, &[a, 48]);
        assert_ne!(b, 0);
        assert_eq!(hle_malloc_usable_size(&ctx, &[b]), 48);
        hle_free(&ctx, &[b]);
        assert_eq!(
            hle_malloc_usable_size(&ctx, &[b]),
            0,
            "freed blocks report 0"
        );
    }

    /// `mbsrtowcs`, `wcscat`, and `strpbrk` do real byte/wide work.
    #[test]
    fn mbsrtowcs_wcscat_and_strpbrk_do_real_work() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // mbsrtowcs: "ab" -> [97, 98, 0], *src becomes NULL, count 2.
        assert!(mem.write(0x100, b"ab\0"));
        assert!(mem.write(0x110, &0x100u64.to_le_bytes()));
        assert_eq!(hle_mbsrtowcs(&ctx, &[0x200, 0x110, 8, 0]), 2);
        let mut units = [0u8; 12];
        assert!(mem.read(0x200, &mut units));
        let wide: Vec<u32> = units
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4-byte chunks")))
            .collect();
        assert_eq!(wide, vec![97, 98, 0]);
        let mut slot = [0u8; 8];
        assert!(mem.read(0x110, &mut slot));
        assert_eq!(u64::from_le_bytes(slot), 0, "*src is NULL after the NUL");
        // Counting mode (dst == NULL).
        assert!(mem.write(0x110, &0x100u64.to_le_bytes()));
        assert_eq!(hle_mbsrtowcs(&ctx, &[0, 0x110, 0, 0]), 2);

        // wcscat: "ab" + "cd" (32-bit units).
        let wide_of =
            |s: &[u8]| -> Vec<u8> { s.iter().flat_map(|&b| u32::from(b).to_le_bytes()).collect() };
        assert!(mem.write(0x300, &wide_of(b"ab\0")));
        assert!(mem.write(0x320, &wide_of(b"cd\0")));
        assert_eq!(hle_wcscat(&ctx, &[0x300, 0x320]), 0x300);
        let mut joined = [0u8; 20];
        assert!(mem.read(0x300, &mut joined));
        let joined: Vec<u32> = joined
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4-byte chunks")))
            .collect();
        assert_eq!(joined, vec![97, 98, 99, 100, 0]);

        // strpbrk finds the first byte of the accept set.
        assert!(mem.write(0x400, b"hello\0"));
        assert!(mem.write(0x410, b"ol\0"));
        assert_eq!(hle_strpbrk(&ctx, &[0x400, 0x410]), 0x402);
        assert!(mem.write(0x410, b"xyz\0"));
        assert_eq!(hle_strpbrk(&ctx, &[0x400, 0x410]), 0, "no match -> NULL");
    }

    /// `localeconv` serves the C locale's struct lconv.
    #[test]
    fn localeconv_serves_the_c_locale() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let lconv = hle_localeconv(&ctx, &[]);
        assert_ne!(lconv, 0);
        let mut ptr = [0u8; 8];
        assert!(mem.read(lconv, &mut ptr));
        let decimal_point = crate::fmt::read_cstr(&mem, u64::from_le_bytes(ptr));
        assert_eq!(decimal_point.as_deref(), Some(b".".as_slice()));
        assert!(mem.read(lconv + 8, &mut ptr));
        let thousands_sep = crate::fmt::read_cstr(&mem, u64::from_le_bytes(ptr));
        assert_eq!(thousands_sep.as_deref(), Some(b"".as_slice()));
        let mut frac = [0u8; 1];
        assert!(mem.read(lconv + 81, &mut frac));
        assert_eq!(frac[0], 0x7F, "frac_digits is CHAR_MAX in the C locale");

        // All FOURTEEN char fields are CHAR_MAX — the six C99 `int_*` variants
        // at 88..94 included. They used to hold the `"."` string, so
        // `int_p_cs_precedes` read back as `'.'` and a guest writing those
        // fields corrupted `decimal_point`.
        let mut chars = [0u8; 14];
        assert!(mem.read(lconv + 80, &mut chars));
        assert!(
            chars.iter().all(|&b| b == 0x7F),
            "lconv's char block is 80..94, not 80..88: {chars:?}"
        );
        // The strings therefore live past the struct, not inside it.
        assert!(
            u64::from_le_bytes(ptr) >= lconv + 96,
            "lconv strings must sit after sizeof(struct lconv)"
        );
    }

    /// Every name in the NID-fill batch resolves under both provider views.
    #[test]
    fn register_adds_the_nid_fill_batch() {
        let registry = HleRegistry::new();
        for name in [
            "_Assert",
            "_Atomic_fetch_add_4",
            "_Atomic_fetch_sub_4",
            "_Getpctype",
            "_Getptolower",
            "_Getptoupper",
            "_Locksyslock",
            "_Unlocksyslock",
            "_Mtx_init",
            "_Mtx_lock",
            "_Mtx_unlock",
            "_Mtx_destroy",
            "_Thrd_id",
            "_Thrd_join",
            "_Thrd_sleep",
            "_Xtime_get_ticks",
            "_Stdout",
            "_Stderr",
            "fflush",
            "fputc",
            "fputs",
            "fwrite",
            "vfprintf",
            "sprintf",
            "sprintf_s",
            "vswprintf",
            "sscanf",
            "strpbrk",
            "wcscat",
            "mbsrtowcs",
            "strtoimax",
            "strtoull",
            "strtoumax",
            "_Stoull",
            "rand",
            "srand",
            "malloc_usable_size",
            "time",
            "gmtime",
            "gmtime_s",
            "localtime",
            "mktime",
            "strftime",
            "localeconv",
            "sincos",
            "__isfinite",
            "__isfinitef",
            "__isnan",
            "__isnanf",
            "catchReturnFromMain",
            "__cxa_allocate_exception",
            "__cxa_free_exception",
            "__cxa_thread_atexit",
            "__cxa_bad_cast",
            "_ZSt18uncaught_exceptionv",
            "_ZSt11_Xbad_allocv",
            "_ZSt14_Throw_C_errori",
            "_ZSt14_Xlength_errorPKc",
            "_ZSt14_Xout_of_rangePKc",
            "_ZSt16_Throw_Cpp_errori",
            "_ZSt19_Xbad_function_callv",
            "_ZSt9terminatev",
            "_ZNKSt9exception6_RaiseEv",
            "_ZNKSt9exception8_DoraiseEv",
        ] {
            assert!(
                registry.is_implemented("libc", name),
                "missing libc::{name}"
            );
            assert!(
                registry.is_implemented("libSceLibcInternal", name),
                "missing libSceLibcInternal::{name} ABI alias"
            );
        }
    }

    /// The float-returning libm handlers compute real values from the XMM
    /// argument channel and hand back the exact result bits the runtime
    /// writes into guest XMM0.
    #[test]
    fn libm_handlers_compute_real_values_through_the_float_channel() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let mut ctx = test_ctx(&kernel, &mem, &alloc);
        let f32_args = |ctx: &mut HleContext, a: f32, b: f32| {
            ctx.float_args[0] = u64::from(a.to_bits());
            ctx.float_args[1] = u64::from(b.to_bits());
        };
        let f64_args = |ctx: &mut HleContext, a: f64, b: f64| {
            ctx.float_args[0] = a.to_bits();
            ctx.float_args[1] = b.to_bits();
        };

        // Exactly-representable cases: bit-exact assertions.
        f32_args(&mut ctx, 0.0, 0.0);
        assert_eq!(f32::from_bits(hle_cosf(&ctx, &[]) as u32), 1.0);
        assert_eq!(f32::from_bits(hle_sinf(&ctx, &[]) as u32), 0.0);
        assert_eq!(f32::from_bits(hle_tanf(&ctx, &[]) as u32), 0.0);
        assert_eq!(f32::from_bits(hle_tanhf(&ctx, &[]) as u32), 0.0);
        assert_eq!(
            f32::from_bits(hle_acosf(&ctx, &[]) as u32),
            std::f32::consts::FRAC_PI_2
        );

        f64_args(&mut ctx, 0.0, 0.0);
        assert_eq!(hle_cos(&ctx, &[]), 1.0f64.to_bits());
        assert_eq!(hle_sin(&ctx, &[]), 0.0f64.to_bits());
        assert_eq!(hle_tan(&ctx, &[]), 0.0f64.to_bits());
        assert_eq!(hle_exp(&ctx, &[]), 1.0f64.to_bits());
        assert_eq!(hle_asin(&ctx, &[]), 0.0f64.to_bits());
        assert_eq!(hle_atan(&ctx, &[]), 0.0f64.to_bits());

        f64_args(&mut ctx, 2.0, 10.0);
        assert_eq!(hle_pow(&ctx, &[]), 1024.0f64.to_bits());
        f32_args(&mut ctx, 2.0, 3.0);
        assert_eq!(f32::from_bits(hle_powf(&ctx, &[]) as u32), 8.0);

        f64_args(&mut ctx, 5.5, 2.0);
        assert_eq!(hle_fmod(&ctx, &[]), 1.5f64.to_bits());
        f32_args(&mut ctx, -5.5, 2.0);
        assert_eq!(
            f32::from_bits(hle_fmodf(&ctx, &[]) as u32),
            -1.5,
            "fmod takes the dividend's sign"
        );

        f64_args(&mut ctx, 3.0, 0.0);
        assert_eq!(hle_exp2(&ctx, &[]), 8.0f64.to_bits());
        f32_args(&mut ctx, 10.0, 0.0);
        assert_eq!(f32::from_bits(hle_exp2f(&ctx, &[]) as u32), 1024.0);

        f64_args(&mut ctx, 8.0, 0.0);
        assert_eq!(hle_log2(&ctx, &[]), 3.0f64.to_bits());
        f64_args(&mut ctx, 1000.0, 0.0);
        assert_eq!(hle_log10(&ctx, &[]), 3.0f64.to_bits());
        f64_args(&mut ctx, std::f64::consts::E, 0.0);
        assert!((f64::from_bits(hle_log(&ctx, &[])) - 1.0).abs() < 1e-15);
        f32_args(&mut ctx, 8.0, 0.0);
        assert_eq!(f32::from_bits(hle_log2f(&ctx, &[]) as u32), 3.0);
        f32_args(&mut ctx, 1000.0, 0.0);
        assert!((f32::from_bits(hle_log10f(&ctx, &[]) as u32) - 3.0).abs() < 1e-6);
        f32_args(&mut ctx, 1.0, 0.0);
        assert_eq!(f32::from_bits(hle_logf(&ctx, &[]) as u32), 0.0);
        f32_args(&mut ctx, 1.0, 0.0);
        assert!((f32::from_bits(hle_expf(&ctx, &[]) as u32) - std::f32::consts::E).abs() < 1e-7);

        // atan2's quadrant handling: (0, -1) → +pi.
        f64_args(&mut ctx, 0.0, -1.0);
        assert!((f64::from_bits(hle_atan2(&ctx, &[])) - std::f64::consts::PI).abs() < 1e-15);
        f32_args(&mut ctx, 1.0, 1.0);
        assert!(
            (f32::from_bits(hle_atan2f(&ctx, &[]) as u32) - std::f32::consts::FRAC_PI_4).abs()
                < 1e-7
        );
        f32_args(&mut ctx, 1.0, 0.0);
        assert!(
            (f32::from_bits(hle_atanf(&ctx, &[]) as u32) - std::f32::consts::FRAC_PI_4).abs()
                < 1e-7
        );
        f32_args(&mut ctx, 0.5, 0.0);
        assert!(
            (f32::from_bits(hle_asinf(&ctx, &[]) as u32) - std::f32::consts::FRAC_PI_6).abs()
                < 1e-7
        );

        // ldexpf: float in XMM0, int exponent in the integer slice.
        f32_args(&mut ctx, 1.5, 0.0);
        assert_eq!(f32::from_bits(hle_ldexpf(&ctx, &[3]) as u32), 12.0);
        assert_eq!(
            f32::from_bits(hle_ldexpf(&ctx, &[(-1i64) as u64]) as u32),
            0.75
        );

        // nextafterf: exact bit steps in both directions, plus the zero case.
        f32_args(&mut ctx, 1.0, 2.0);
        assert_eq!(hle_nextafterf(&ctx, &[]), u64::from(0x3F80_0001u32));
        f32_args(&mut ctx, 1.0, 0.0);
        assert_eq!(hle_nextafterf(&ctx, &[]), u64::from(0x3F7F_FFFFu32));
        f32_args(&mut ctx, 0.0, -1.0);
        assert_eq!(
            hle_nextafterf(&ctx, &[]),
            u64::from(0x8000_0001u32),
            "from +0 toward -1: the least negative subnormal"
        );

        // difftime: integer args, double result.
        assert_eq!(hle_difftime(&ctx, &[20, 5]), 15.0f64.to_bits());
        assert_eq!(hle_difftime(&ctx, &[5, 20]), (-15.0f64).to_bits());
    }

    /// `strtod`/`atof` parse real C doubles (including hex floats and
    /// inf/nan) and honor the endptr contract.
    #[test]
    fn strtod_and_atof_parse_doubles_with_endptr() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(0x100, b"  1.5x\0"));
        assert_eq!(f64::from_bits(hle_strtod(&ctx, &[0x100, 0x200])), 1.5);
        let mut ep = [0u8; 8];
        assert!(mem.read(0x200, &mut ep));
        assert_eq!(u64::from_le_bytes(ep), 0x100 + 5, "endptr at 'x'");

        // Hex float: 0x1.8p1 == 3.0.
        assert!(mem.write(0x120, b"0x1.8p1\0"));
        assert_eq!(f64::from_bits(hle_strtod(&ctx, &[0x120, 0])), 3.0);

        // Case-insensitive infinity, with sign.
        assert!(mem.write(0x140, b"-INF\0"));
        assert_eq!(
            f64::from_bits(hle_strtod(&ctx, &[0x140, 0])),
            f64::NEG_INFINITY
        );

        // No conversion: 0.0 and endptr == nptr.
        assert!(mem.write(0x160, b"abc\0"));
        assert_eq!(f64::from_bits(hle_strtod(&ctx, &[0x160, 0x208])), 0.0);
        assert!(mem.read(0x208, &mut ep));
        assert_eq!(u64::from_le_bytes(ep), 0x160);

        // atof is strtod without the endptr.
        assert!(mem.write(0x180, b"2.5\0"));
        assert_eq!(f64::from_bits(hle_atof(&ctx, &[0x180])), 2.5);
    }

    /// The float-returning batch is registered under both provider views AND
    /// carries the float marker the runtime consults; integer-returning
    /// neighbors must NOT carry it.
    #[test]
    fn register_marks_the_float_batch_float_returning() {
        let registry = HleRegistry::new();
        for name in [
            "acosf",
            "asin",
            "asinf",
            "atan",
            "atan2",
            "atan2f",
            "atanf",
            "atof",
            "cos",
            "cosf",
            "difftime",
            "exp",
            "exp2",
            "exp2f",
            "expf",
            "fmod",
            "fmodf",
            "ldexpf",
            "log",
            "log10",
            "log10f",
            "log2",
            "log2f",
            "logf",
            "nextafterf",
            "pow",
            "powf",
            "sin",
            "sinf",
            "strtod",
            "tan",
            "tanf",
            "tanhf",
        ] {
            assert!(
                registry.is_implemented("libc", name),
                "missing libc::{name}"
            );
            assert!(
                registry.is_implemented("libSceLibcInternal", name),
                "missing libSceLibcInternal::{name} ABI alias"
            );
            assert!(
                registry.returns_float("libc", name),
                "libc::{name} must be marked float-returning"
            );
            assert!(
                registry.returns_float("libSceLibcInternal", name),
                "libSceLibcInternal::{name} must be marked float-returning"
            );
        }
        // Integer/void-returning functions stay on the plain RAX channel.
        for name in ["strlen", "__isnan", "sincos", "time", "sprintf"] {
            assert!(
                !registry.returns_float("libc", name),
                "libc::{name} must NOT be marked float-returning"
            );
        }
        assert!(
            !registry.returns_float("libc", "definitely_not_registered"),
            "unknown names report false"
        );
    }
}
