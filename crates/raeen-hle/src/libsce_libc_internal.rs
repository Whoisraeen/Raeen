//! HLE libSceLibcInternal / libSceLibcInternalExt — libc.prx's private
//! heap-instrumentation hooks.
//!
//! Real libc.prx wires its allocator's tracing/diagnostic plumbing through
//! these during `module_start`, and a real title (Minecraft) faults on
//! `sceLibcHeapGetTraceInfo` right there. Semantics cross-checked against
//! SharpEmu's `LibcInternalExports` (GPL-2.0) and Kyty's
//! `LibcInternalExt::LibcHeapGetTraceInfo` (MIT):
//!
//! * `sceLibcHeapGetTraceInfo(info)` — `info` is a 32-byte block
//!   `{ u64 size; u64 _; u64 *mask; u64 *table }`; the callee validates
//!   `size == 32` and writes pointers to persistent zeroed storage (an
//!   8-byte atomic-id mask + a 64-entry mstate table) through `+16`/`+24`.
//!   libc then uses that storage for its own bookkeeping, so it must be
//!   real guest memory, not a fake address.
//! * `sceLibcInternalHeapErrorReportForGame` / `BacktraceForGame` — the
//!   game's own heap instrumentation reporting problems; a diagnostic sink
//!   that logs and returns success.
//!
//! The trace storage is a process-global (one guest process per host
//! process under the runtime's RT0 single-active-execution invariant),
//! lazily carved from the guest arena on first request.

use crate::{HleContext, HleFunction, HleRegistry};
use std::sync::atomic::Ordering;
use tracing::{debug, warn};

const SCE_OK: u64 = 0;
const ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

/// The `info` block's mandatory `size` field value.
const TRACE_INFO_SIZE: u64 = 32;
/// 8-byte mspace atomic-id mask, then a 64-entry (`u64`) mstate table.
const TRACE_TABLE_ENTRIES: u64 = 64;
const TRACE_STORAGE_BYTES: u64 = 8 + TRACE_TABLE_ENTRIES * 8;

/// Register the libSceLibcInternal(-Ext) functions.
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libSceLibcInternalExt",
        "sceLibcHeapGetTraceInfo",
        hle_heap_get_trace_info,
    );
    // Resolution is provider-aware: the measured ASTRO.BOT imports these two
    // naming `libSceLibcInternalExt`, so both provider spellings must be
    // registered (same implementations).
    for library in ["libSceLibcInternal", "libSceLibcInternalExt"] {
        registry.register(
            library,
            "sceLibcInternalHeapErrorReportForGame",
            hle_heap_error_report,
        );
        registry.register(
            library,
            "sceLibcInternalBacktraceForGame",
            hle_backtrace_for_game,
        );
    }
    // C++ ABI plumbing libc.prx re-exports through libSceLibcInternal.
    registry.register("libSceLibcInternal", "__cxa_finalize", hle_cxa_finalize);
    registry.register(
        "libSceLibcInternal",
        "__cxa_pure_virtual",
        hle_cxa_pure_virtual,
    );
    // C++ `operator new`/`operator delete` and all their overloads. Titles
    // import these from EITHER `libc` (ASTRO.BOT PPSA21564 imports `_Znwm` from
    // libc) or `libSceLibcInternal` (libc.prx re-exports them), so register
    // under both — resolution keys on the importing symbol's library.
    for lib in ["libc", "libSceLibcInternal"] {
        // operator new / new[] (throwing + nothrow): plain size arg.
        for name in [
            "_Znwm",                        // operator new(size_t)
            "_Znam",                        // operator new[](size_t)
            "_ZnwmRKSt9nothrow_t",          // operator new(size_t, nothrow_t)
            "_ZnamRKSt9nothrow_t",          // operator new[](size_t, nothrow_t)
        ] {
            registry.register(lib, name, hle_operator_new);
        }
        // Aligned operator new / new[]: (size, align[, nothrow]).
        for name in [
            "_ZnwmSt11align_val_t",
            "_ZnamSt11align_val_t",
            "_ZnwmSt11align_val_tRKSt9nothrow_t",
            "_ZnamSt11align_val_tRKSt9nothrow_t",
        ] {
            registry.register(lib, name, hle_operator_new_aligned);
        }
        // operator delete / delete[] (plain, sized, aligned, array): free the
        // pointer, ignore the advisory size/align argument.
        for name in [
            "_ZdlPv",                       // operator delete(void*)
            "_ZdaPv",                       // operator delete[](void*)
            "_ZdlPvm",                      // sized operator delete(void*, size_t)
            "_ZdaPvm",
            "_ZdlPvSt11align_val_t",        // aligned operator delete
            "_ZdaPvSt11align_val_t",
            "_ZdlPvmSt11align_val_t",       // sized+aligned
            "_ZdaPvmSt11align_val_t",
            "_ZdlPvRKSt9nothrow_t",         // nothrow operator delete
            "_ZdaPvRKSt9nothrow_t",
        ] {
            registry.register(lib, name, hle_operator_delete);
        }
    }
    // `_Stoul` is the Dinkumware STL's strtoul core — identical
    // `(nptr, endptr, base)` ABI to POSIX strtoul, so the real libc parser
    // serves it.
    registry.register("libSceLibcInternal", "_Stoul", crate::libc::hle_strtoul);
    registry.register(
        "libSceLibcInternal",
        "sceLibcMspaceCreate",
        hle_mspace_create,
    );
    registry.register(
        "libSceLibcInternal",
        "sceLibcMspaceDestroy",
        hle_mspace_destroy,
    );
    registry.register("libSceLibcInternal", "sceLibcMspaceFree", hle_mspace_free);
    registry.register(
        "libSceLibcInternal",
        "sceLibcMspaceMalloc",
        hle_mspace_malloc,
    );
    registry.register(
        "libSceLibcInternal",
        "sceLibcMspaceMemalign",
        hle_mspace_memalign,
    );
    registry.register(
        "libSceLibcInternal",
        "sceLibcMspaceMallocUsableSize",
        hle_mspace_malloc_usable_size,
    );
    registry.register(
        "libSceLibcInternal",
        "sceLibcMspaceMallocStats",
        hle_mspace_malloc_stats,
    );
    registry.register(
        "libSceLibcInternal",
        "sceLibcMspaceMallocStatsFast",
        hle_mspace_malloc_stats,
    );
    registry.register(
        "libSceLibcInternal",
        "sceLibcMspaceRealloc",
        hle_mspace_realloc,
    );
    // The `sceLibcMspace*` heap family, `_Stoul`, and `__cxa_pure_virtual` are
    // re-exported by `libc.prx`, and titles import them from `libc` (ASTRO.BOT
    // PPSA21564 imports the whole mspace heap from libc — its `new`/STL heap
    // rides on an mspace). Resolution keys on the importing library, so mirror
    // every one under `libc` too. `MallocStatsFast` shares the stats handler.
    let mspace_family: [(&str, HleFunction); 11] = [
        ("sceLibcMspaceCreate", hle_mspace_create),
        ("sceLibcMspaceDestroy", hle_mspace_destroy),
        ("sceLibcMspaceFree", hle_mspace_free),
        ("sceLibcMspaceMalloc", hle_mspace_malloc),
        ("sceLibcMspaceCalloc", hle_mspace_calloc),
        ("sceLibcMspaceMemalign", hle_mspace_memalign),
        ("sceLibcMspaceRealloc", hle_mspace_realloc),
        ("sceLibcMspaceMallocUsableSize", hle_mspace_malloc_usable_size),
        ("sceLibcMspaceMallocStats", hle_mspace_malloc_stats),
        ("sceLibcMspaceMallocStatsFast", hle_mspace_malloc_stats),
        ("__cxa_pure_virtual", hle_cxa_pure_virtual),
    ];
    for (name, func) in mspace_family {
        registry.register("libc", name, func);
    }
    // Calloc is new under libSceLibcInternal too; _Stoul under libc.
    registry.register("libSceLibcInternal", "sceLibcMspaceCalloc", hle_mspace_calloc);
    registry.register("libc", "_Stoul", crate::libc::hle_strtoul);
    // `std::_Random_device()` (Dinkumware's `random_device` core) returns a
    // fresh unsigned int the STL uses to seed a PRNG. No host entropy source is
    // wired in; an xorshift32 gives distinct successive values, enough to seed.
    registry.register("libc", "_ZSt14_Random_devicev", hle_std_random_device);
    registry.register(
        "libSceLibcInternal",
        "_ZSt14_Random_devicev",
        hle_std_random_device,
    );
}

/// `sceLibcMspaceCalloc(msp, nmemb, size)`: allocate `nmemb * size` zeroed
/// bytes from the mspace. Overflow in the product yields a null return, matching
/// `calloc`. Mirrors `hle_mspace_malloc` then zero-fills the block.
fn hle_mspace_calloc(ctx: &HleContext, args: &[u64]) -> u64 {
    let nmemb = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    let Some(total) = nmemb.checked_mul(size) else {
        return 0;
    };
    let addr = hle_mspace_malloc(ctx, &[args.first().copied().unwrap_or(0), total]);
    if addr != 0 && total != 0 {
        let _ = ctx.mem.write(addr, &vec![0u8; total as usize]);
    }
    addr
}

/// `std::_Random_device()`: return a fresh 32-bit value (xorshift32 so the
/// STL's repeated calls while seeding a generator each differ).
fn hle_std_random_device(_ctx: &HleContext, _args: &[u64]) -> u64 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static STATE: AtomicU32 = AtomicU32::new(0x9e37_79b9);
    let mut x = STATE.load(Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    STATE.store(x, Ordering::Relaxed);
    u64::from(x)
}

/// `__cxa_finalize(void *dso)`: run the atexit/destructor chain registered
/// for one DSO. Raeen records `__cxa_atexit` callbacks but never dispatches
/// them (see `libc::hle_cxa_atexit` — process teardown is host-driven), so
/// finalize has nothing to run and succeeds. Itanium C++ ABI; NULL means
/// "all DSOs" and is equally a no-op here.
fn hle_cxa_finalize(_ctx: &HleContext, args: &[u64]) -> u64 {
    debug!(
        "__cxa_finalize(dso={:#x}) -> no registered destructors dispatched",
        args.first().copied().unwrap_or(0)
    );
    SCE_OK
}

/// `__cxa_pure_virtual()`: the Itanium C++ ABI's trap for calling a pure
/// virtual through a partially-constructed/destroyed object. Reaching it is
/// a GUEST BUG (the real one aborts the process) — surface it loudly with
/// the caller address so `--dump-vaddr` can find the broken vtable call,
/// then return 0 to give diagnostics a chance to keep flowing rather than
/// tearing the process down inside an HLE trap.
fn hle_cxa_pure_virtual(ctx: &HleContext, _args: &[u64]) -> u64 {
    warn!(
        caller = format_args!("{:#x}", ctx.caller_return_addr),
        "__cxa_pure_virtual called — GUEST BUG: pure virtual call through a \
         partially-constructed or destroyed object (real libc aborts here)"
    );
    0
}

/// `_Znwm`/`_Znam` — `operator new(size_t)` / `operator new[](size_t)` and the
/// nothrow overloads (`...RKSt9nothrow_t`): allocate `size` bytes from the same
/// arena as `malloc`/`operator delete`. C++ requires a UNIQUE non-null pointer
/// even for a zero-size request, so 0 allocates 1 byte. A failed allocation
/// should throw `std::bad_alloc`; with no C++ exception machinery the HLE hands
/// back null (the caller faults visibly rather than corrupting silently).
/// A C++ game cannot construct a single object without this — it is the very
/// first thing a title's runtime touches, so a missing `_Znwm` halts boot at
/// crt0 (measured on ASTRO.BOT PPSA21564).
fn hle_operator_new(ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.first().copied().unwrap_or(0).max(1);
    ctx.alloc.alloc(size, 16).unwrap_or(0)
}

/// `_ZnwmSt11align_val_t` and friends — the aligned `operator new` overloads.
/// The requested alignment arrives in the 2nd integer argument; honour it (a
/// minimum of 16 matches the default `malloc` alignment).
fn hle_operator_new_aligned(ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.first().copied().unwrap_or(0).max(1);
    let align = args.get(1).copied().unwrap_or(16).max(16);
    ctx.alloc.alloc(size, align).unwrap_or(0)
}

/// `_ZdlPv` — `operator delete(void*)`: releases storage that came from the
/// allocator behind `malloc`/`operator new` (this HLE's `ctx.alloc`, the
/// same model `libc::hle_free` frees into). `delete nullptr` is defined as a
/// no-op. The sized/aligned/array overloads (`_ZdlPvm`, `_ZdlPvSt11align_val_t`,
/// `_ZdaPv`, ...) release the same way — the extra size/align argument is
/// advisory and ignored, exactly like `free`.
fn hle_operator_delete(ctx: &HleContext, args: &[u64]) -> u64 {
    let ptr = args.first().copied().unwrap_or(0);
    debug!("operator delete(_ZdlPv)(ptr={ptr:#x})");
    if ptr != 0 {
        ctx.alloc.free(ptr);
    }
    0
}

/// `sceLibcMspaceRealloc(msp, ptr, size)`: resize an mspace allocation.
/// dlmalloc semantics (the engine behind Sony's mspaces): NULL ptr acts as
/// malloc, size 0 acts as free, otherwise allocate-copy-free within the SAME
/// mspace — the copy moves `min(old, new)` bytes through guest memory, and a
/// failed allocation leaves the original block untouched and returns 0.
fn hle_mspace_realloc(ctx: &HleContext, args: &[u64]) -> u64 {
    let mspace = args.first().copied().unwrap_or(0);
    let ptr = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    if ptr == 0 {
        return hle_mspace_malloc(ctx, &[mspace, size]);
    }
    if size == 0 {
        let _ = hle_mspace_free(ctx, &[mspace, ptr]);
        return 0;
    }
    let Some(old_size) = ctx
        .kernel
        .libc_mspace_allocations
        .get(&ptr)
        .filter(|allocation| allocation.mspace == mspace)
        .map(|allocation| allocation.size)
    else {
        warn!("sceLibcMspaceRealloc(msp={mspace:#x}, ptr={ptr:#x}): unknown allocation");
        return 0;
    };
    let new_ptr = hle_mspace_malloc(ctx, &[mspace, size]);
    if new_ptr == 0 {
        return 0; // old block untouched, dlmalloc contract
    }
    // Chunked guest-to-guest copy of the surviving bytes.
    let mut remaining = old_size.min(size);
    let mut offset = 0u64;
    let mut chunk = [0u8; 4096];
    while remaining > 0 {
        let take = remaining.min(chunk.len() as u64) as usize;
        if !ctx.mem.read(ptr + offset, &mut chunk[..take])
            || !ctx.mem.write(new_ptr + offset, &chunk[..take])
        {
            warn!("sceLibcMspaceRealloc: copy fault at +{offset:#x} — abandoning the resize");
            let _ = hle_mspace_free(ctx, &[mspace, new_ptr]);
            return 0;
        }
        offset += take as u64;
        remaining -= take as u64;
    }
    let _ = hle_mspace_free(ctx, &[mspace, ptr]);
    new_ptr
}

fn read_cstring(ctx: &HleContext, address: u64, max: usize) -> Option<String> {
    if address == 0 {
        return None;
    }
    let mut bytes = Vec::new();
    for offset in 0..max {
        let mut byte = [0u8; 1];
        if !ctx.mem.read(address + offset as u64, &mut byte) {
            return None;
        }
        if byte[0] == 0 {
            return Some(String::from_utf8_lossy(&bytes).into_owned());
        }
        bytes.push(byte[0]);
    }
    None
}

/// The alignment `sceLibcMspaceMalloc` allocates at: the x86-64 SysV ABI's
/// `malloc` guarantee, suitable for any type with fundamental alignment.
const MSPACE_DEFAULT_ALIGN: u64 = 16;

/// Create a fixed-capacity mspace over memory already owned by the guest.
fn hle_mspace_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let name_address = args.first().copied().unwrap_or(0);
    let base = args.get(1).copied().unwrap_or(0);
    let capacity = args.get(2).copied().unwrap_or(0);
    let flag = args.get(3).copied().unwrap_or(0);
    // Returning 0 here hands the guest a null heap, and a title does not
    // necessarily say so: ASTRO.BOT reports "Out of Global Heap Memory" from its
    // own configured numbers (Current/Max: 0 / 2038431744) and never calls
    // malloc at all — a failure that reads as an allocator bug when it is
    // actually a rejected create. Say which guard rejected it.
    if base == 0 || capacity < 0x100 || flag > 1 {
        warn!(
            "sceLibcMspaceCreate(base={base:#x}, capacity={capacity:#x}, flag={flag}): rejected \
             by argument guard — returning a NULL mspace"
        );
        return 0;
    }
    // A NULL name is VALID — the mspace is simply anonymous, and titles do pass
    // NULL (ASTRO.BOT does). Only a non-NULL but unreadable pointer is suspect;
    // treat that as anonymous too rather than failing the create — a rejected
    // create hands the title a null heap it later dereferences (measured:
    // ASTRO.BOT's sceLibcMspaceFree faulting on a null mspace).
    let name = if name_address == 0 {
        String::new()
    } else {
        read_cstring(ctx, name_address, 256).unwrap_or_else(|| {
            warn!(
                "sceLibcMspaceCreate(base={base:#x}, capacity={capacity:#x}): name at \
                 {name_address:#x} unreadable — using an anonymous name"
            );
            String::new()
        })
    };
    ctx.kernel.libc_mspaces.insert(
        base,
        raeen_kernel::LibcMspace {
            base,
            capacity,
            next_offset: 0x100,
            peak_offset: 0x100,
            active_bytes: 0,
            name,
            free_list: Vec::new(),
        },
    );
    base
}

fn hle_mspace_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let mspace = args.first().copied().unwrap_or(0);
    if ctx.kernel.libc_mspaces.remove(&mspace).is_none() {
        return 0;
    }
    ctx.kernel
        .libc_mspace_allocations
        .retain(|_, allocation| allocation.mspace != mspace);
    1
}

/// `sceLibcMspaceMalloc(msp, size)` — the same storage as
/// [`hle_mspace_memalign`], at the ABI's default alignment.
///
/// # Why its absence broke a whole title
///
/// Implementing half of a paired API is worse than implementing none of it. The
/// resolver picks HLE or LLE **per symbol** (`ModulePolicy::PreferHle`, the
/// default: try HLE, fall back to the module's own export). So with `Create`
/// implemented here and `Malloc` not, a title bundling its own `libc.prx` got
/// its mspace *created* by us — which hands back `base` as an opaque handle and
/// keeps the bookkeeping in our kernel — and then *allocated* by the real
/// dlmalloc inside its libc, which reads `base` expecting an initialised
/// `malloc_state` there, finds uninitialised mapped memory, and reports the heap
/// exhausted.
///
/// Measured on ASTRO.BOT, which prints exactly that and then aborts:
///
/// ```text
/// ASSERT: ...\engine\app\Common\Memory.cpp:69
/// Out of Global Heap Memory
/// Allocator Size : 72
/// Current/Max    : 0 / 2038431744 (0.00%)
/// ```
///
/// A heap 0.00% full cannot be out of memory — those are the game's configured
/// numbers, printed because it never got a working allocator at all.
///
/// The general rule this encodes: whoever owns `Create` must own every operation
/// on what it returns. A half-HLE'd library is split-brained by construction.
fn hle_mspace_malloc(ctx: &HleContext, args: &[u64]) -> u64 {
    let mspace = args.first().copied().unwrap_or(0);
    // The platform mspace allocator may return a unique minimum allocation for
    // malloc(0). Measured ASTRO.BOT relies on that behavior: returning NULL for
    // its zero-byte request is interpreted as a fatal global-heap OOM.
    let size = args.get(1).copied().unwrap_or(0).max(1);
    hle_mspace_memalign(ctx, &[mspace, MSPACE_DEFAULT_ALIGN, size])
}

fn hle_mspace_memalign(ctx: &HleContext, args: &[u64]) -> u64 {
    let mspace = args.first().copied().unwrap_or(0);
    let alignment = args.get(1).copied().unwrap_or(0).max(8);
    let size = args.get(2).copied().unwrap_or(0);
    if size == 0 {
        warn!(
            "sceLibcMspaceMemalign(msp={mspace:#x}, align={alignment:#x}, size=0): \
             returning NULL (caller={:#x})",
            ctx.caller_return_addr
        );
        return 0;
    }
    if !alignment.is_power_of_two() {
        warn!(
            "sceLibcMspaceMemalign(msp={mspace:#x}, align={alignment:#x}, size={size:#x}): \
             alignment is not a power of two"
        );
        return 0;
    }
    let Some(mut state) = ctx.kernel.libc_mspaces.get_mut(&mspace) else {
        warn!(
            "sceLibcMspaceMemalign(msp={mspace:#x}, align={alignment:#x}, size={size:#x}): \
             mspace is not registered"
        );
        return 0;
    };
    let base = state.base;

    // First-fit over reclaimed blocks: reuse before bumping so churn can't
    // exhaust a fixed-capacity mspace (native dlmalloc reclaims; we must too).
    let mut chosen = None;
    for (index, &(block_off, block_size)) in state.free_list.iter().enumerate() {
        // Align the block's start address up (alignment is a power of two).
        let aligned_off = (((base + block_off) + alignment - 1) & !(alignment - 1)) - base;
        if aligned_off + size <= block_off + block_size {
            chosen = Some((index, block_off, block_size, aligned_off));
            break;
        }
    }
    if let Some((index, block_off, block_size, aligned_off)) = chosen {
        state.free_list.remove(index);
        let block_end = block_off + block_size;
        let alloc_end = aligned_off + size;
        // Return the alignment gap and any tail to the free list.
        if aligned_off > block_off {
            insert_and_coalesce(&mut state.free_list, block_off, aligned_off - block_off);
        }
        if block_end > alloc_end {
            insert_and_coalesce(&mut state.free_list, alloc_end, block_end - alloc_end);
        }
        state.active_bytes = state.active_bytes.saturating_add(size);
        drop(state);
        let address = base + aligned_off;
        ctx.kernel
            .libc_mspace_allocations
            .insert(address, raeen_kernel::LibcMspaceAllocation { mspace, size });
        return address;
    }

    // No reusable block: bump the high-water mark.
    let Some(aligned) = state
        .next_offset
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
    else {
        return 0;
    };
    let Some(end) = aligned.checked_add(size) else {
        warn!(
            "sceLibcMspaceMemalign(msp={mspace:#x}, align={alignment:#x}, size={size:#x}): \
             allocation end overflowed"
        );
        return 0;
    };
    if end > state.capacity {
        let free_bytes = state
            .free_list
            .iter()
            .fold(0u64, |total, &(_, bytes)| total.saturating_add(bytes));
        let largest_free = state
            .free_list
            .iter()
            .map(|&(_, bytes)| bytes)
            .max()
            .unwrap_or(0);
        warn!(
            "sceLibcMspaceMemalign(msp={mspace:#x}, align={alignment:#x}, size={size:#x}): OOM \
             capacity={:#x} next={:#x} active={:#x} free_bytes={free_bytes:#x} \
             largest_free={largest_free:#x} free_blocks={}",
            state.capacity,
            state.next_offset,
            state.active_bytes,
            state.free_list.len()
        );
        return 0;
    }
    let Some(address) = base.checked_add(aligned) else {
        return 0;
    };
    state.next_offset = end;
    state.peak_offset = state.peak_offset.max(end);
    state.active_bytes = state.active_bytes.saturating_add(size);
    drop(state);
    ctx.kernel
        .libc_mspace_allocations
        .insert(address, raeen_kernel::LibcMspaceAllocation { mspace, size });
    address
}

/// Insert a `(offset, size)` free block, keeping the list sorted by offset and
/// merging any block adjacent to its neighbours so churn cannot shred the mspace
/// into unusable slivers.
fn insert_and_coalesce(list: &mut Vec<(u64, u64)>, offset: u64, size: u64) {
    if size == 0 {
        return;
    }
    list.push((offset, size));
    list.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(list.len());
    for &(off, sz) in list.iter() {
        match merged.last_mut() {
            Some(last) if last.0 + last.1 == off => last.1 += sz,
            _ => merged.push((off, sz)),
        }
    }
    *list = merged;
}

fn hle_mspace_free(ctx: &HleContext, args: &[u64]) -> u64 {
    let mspace = args.first().copied().unwrap_or(0);
    let address = args.get(1).copied().unwrap_or(0);
    let Some((_, allocation)) = ctx.kernel.libc_mspace_allocations.remove(&address) else {
        return 0;
    };
    if allocation.mspace != mspace {
        ctx.kernel
            .libc_mspace_allocations
            .insert(address, allocation);
        return 0;
    }
    if let Some(mut state) = ctx.kernel.libc_mspaces.get_mut(&mspace) {
        state.active_bytes = state.active_bytes.saturating_sub(allocation.size);
        // Reclaim the block so a later malloc can reuse it instead of bumping.
        let offset = address - state.base;
        insert_and_coalesce(&mut state.free_list, offset, allocation.size);
    }
    1
}

fn hle_mspace_malloc_usable_size(ctx: &HleContext, args: &[u64]) -> u64 {
    let address = args.first().copied().unwrap_or(0);
    ctx.kernel
        .libc_mspace_allocations
        .get(&address)
        .map_or(0, |allocation| allocation.size)
}

fn hle_mspace_malloc_stats(ctx: &HleContext, args: &[u64]) -> u64 {
    let mspace = args.first().copied().unwrap_or(0);
    let output = args.get(1).copied().unwrap_or(0);
    let Some(state) = ctx.kernel.libc_mspaces.get(&mspace) else {
        return 0;
    };
    if output == 0 {
        return 0;
    }
    // `SceLibcMallocManagedSize` has an 8-byte header the CALLER fills — `sz:u16`,
    // `ver:u16`, `reserved:u32` — followed by four `size_t` fields at 0x08 / 0x10
    // / 0x18 / 0x20. Writing from offset 0 clobbered the header AND shifted every
    // field one slot early, so the guest read `maxSystemSize` as our tiny
    // `next_offset` (0x100). ASTRO.BOT's GpuMemory then saw a ~256-byte "system
    // size" and reported "Out of graphics memory [Onion]" for a 2 MiB request on
    // an otherwise-empty 184 MiB pool (`GpuMemory.cpp:155`).
    if std::env::var_os("RAEEN_TRACE_MSPACE").is_some() {
        warn!(
            "MSPACE-STATS mspace={mspace:#x} name={:?} cap={:#x} next={:#x} peak={:#x} active={:#x}",
            state.name, state.capacity, state.next_offset, state.peak_offset, state.active_bytes
        );
    }
    let fields = [
        (0x08u64, state.capacity),  // maxSystemSize: whole fixed region
        (0x10, state.capacity),     // currentSystemSize: fully committed
        (0x18, state.peak_offset),  // maxInuseSize: high-water mark
        (0x20, state.active_bytes), // currentInuseSize: live bytes now
    ];
    for (offset, value) in fields {
        if !ctx.mem.write(output + offset, &value.to_le_bytes()) {
            return ERROR_MEMORY_FAULT;
        }
    }
    SCE_OK
}

/// Lazily allocate the persistent zeroed trace storage, returning its guest
/// address (0 if the arena is exhausted).
fn trace_storage(ctx: &HleContext) -> u64 {
    let existing = ctx.kernel.libc_trace_storage.load(Ordering::Acquire);
    if existing != 0 {
        return existing;
    }
    let Some(storage) = ctx.alloc.alloc(TRACE_STORAGE_BYTES, 8) else {
        return 0;
    };
    if !crate::zero_guest_range(ctx.mem, storage, TRACE_STORAGE_BYTES) {
        ctx.alloc.free(storage);
        return 0;
    }
    match ctx.kernel.libc_trace_storage.compare_exchange(
        0,
        storage,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => storage,
        Err(winner) => {
            ctx.alloc.free(storage);
            winner
        }
    }
}

/// `sceLibcHeapGetTraceInfo(info)`.
fn hle_heap_get_trace_info(ctx: &HleContext, args: &[u64]) -> u64 {
    let info = args.first().copied().unwrap_or(0);
    let mut size = [0u8; 8];
    if info == 0 || !ctx.mem.read(info, &mut size) {
        return ERROR_INVALID_ARGUMENT;
    }
    let size = u64::from_le_bytes(size);
    if size != TRACE_INFO_SIZE {
        warn!("sceLibcHeapGetTraceInfo: unexpected info size {size} (want {TRACE_INFO_SIZE})");
        return ERROR_INVALID_ARGUMENT;
    }
    let storage = trace_storage(ctx);
    if storage == 0 {
        warn!("sceLibcHeapGetTraceInfo: guest arena exhausted for trace storage");
        return ERROR_MEMORY_FAULT;
    }
    let mask_addr = storage;
    let table_addr = storage + 8;
    if !ctx.mem.write(info + 16, &mask_addr.to_le_bytes())
        || !ctx.mem.write(info + 24, &table_addr.to_le_bytes())
    {
        return ERROR_MEMORY_FAULT;
    }
    debug!("sceLibcHeapGetTraceInfo -> mask={mask_addr:#x} table={table_addr:#x}");
    SCE_OK
}

/// `sceLibcInternalHeapErrorReportForGame(...)`: the guest's heap
/// instrumentation reporting a bad block — log what it said and let it
/// continue.
fn hle_heap_error_report(_ctx: &HleContext, args: &[u64]) -> u64 {
    warn!(
        "sceLibcInternalHeapErrorReportForGame({:#x}, {:#x}, {:#x}, {:#x}) — guest reported a \
         heap error",
        args.first().copied().unwrap_or(0),
        args.get(1).copied().unwrap_or(0),
        args.get(2).copied().unwrap_or(0),
        args.get(3).copied().unwrap_or(0),
    );
    SCE_OK
}

/// `sceLibcInternalBacktraceForGame(...)`: no unwinder yet — zero frames.
fn hle_backtrace_for_game(_ctx: &HleContext, _args: &[u64]) -> u64 {
    debug!("sceLibcInternalBacktraceForGame() -> 0 frames (no unwinder)");
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    #[test]
    fn trace_info_writes_real_guest_storage_pointers() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x2000);
        let alloc = crate::TestAllocator::new(0x100);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        let info = 0x40u64;
        // Wrong size is rejected.
        assert!(mem.write(info, &16u64.to_le_bytes()));
        assert_eq!(
            hle_heap_get_trace_info(&ctx, &[info]),
            ERROR_INVALID_ARGUMENT
        );
        // NULL info is rejected.
        assert_eq!(hle_heap_get_trace_info(&ctx, &[0]), ERROR_INVALID_ARGUMENT);

        // size == 32 gets mask/table pointers into real, writable guest memory.
        assert!(mem.write(info, &TRACE_INFO_SIZE.to_le_bytes()));
        assert_eq!(hle_heap_get_trace_info(&ctx, &[info]), SCE_OK);
        let mut mask_ptr = [0u8; 8];
        let mut table_ptr = [0u8; 8];
        assert!(mem.read(info + 16, &mut mask_ptr));
        assert!(mem.read(info + 24, &mut table_ptr));
        let mask_addr = u64::from_le_bytes(mask_ptr);
        let table_addr = u64::from_le_bytes(table_ptr);
        assert_ne!(mask_addr, 0);
        assert_eq!(table_addr, mask_addr + 8, "table follows the 8-byte mask");
        // The guest can write its bookkeeping there.
        assert!(mem.write(mask_addr, &u64::MAX.to_le_bytes()));
        assert!(mem.write(
            table_addr + (TRACE_TABLE_ENTRIES - 1) * 8,
            &1u64.to_le_bytes()
        ));

        // A second call hands back the SAME storage (persistent, not re-carved).
        let info2 = 0x80u64;
        assert!(mem.write(info2, &TRACE_INFO_SIZE.to_le_bytes()));
        assert_eq!(hle_heap_get_trace_info(&ctx, &[info2]), SCE_OK);
        let mut mask2 = [0u8; 8];
        assert!(mem.read(info2 + 16, &mut mask2));
        assert_eq!(u64::from_le_bytes(mask2), mask_addr);
    }

    #[test]
    fn diagnostic_sinks_return_success() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_heap_error_report(&ctx, &[1, 2, 3, 4]), SCE_OK);
        assert_eq!(hle_backtrace_for_game(&ctx, &[]), 0);
    }

    #[test]
    fn mspace_create_memalign_stats_free_and_destroy_are_process_local() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x3000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x40, b"renderer\0"));

        let mspace = hle_mspace_create(&ctx, &[0x40, 0x400, 0x1000, 0]);
        assert_eq!(mspace, 0x400);
        let allocation = hle_mspace_memalign(&ctx, &[mspace, 0x100, 0x180]);
        assert_ne!(allocation, 0);
        assert_eq!(allocation & 0xFF, 0);
        assert_eq!(hle_mspace_malloc_usable_size(&ctx, &[allocation]), 0x180);

        // SceLibcMallocManagedSize: 8-byte header + size_t fields at 0x08/0x10/
        // 0x18/0x20. maxSystemSize (0x08) = capacity; currentInuseSize (0x20) =
        // live bytes. Writing these correctly is what stops ASTRO.BOT reading a
        // 256-byte "system size" and reporting Out-of-graphics-memory.
        assert_eq!(hle_mspace_malloc_stats(&ctx, &[mspace, 0x100]), SCE_OK);
        let mut value = [0u8; 8];
        assert!(mem.read(0x108, &mut value)); // maxSystemSize
        assert_eq!(u64::from_le_bytes(value), 0x1000);
        assert!(mem.read(0x120, &mut value)); // currentInuseSize
        assert_eq!(u64::from_le_bytes(value), 0x180);

        assert_eq!(hle_mspace_free(&ctx, &[mspace, allocation]), 1);
        assert_eq!(hle_mspace_malloc_usable_size(&ctx, &[allocation]), 0);
        assert_eq!(hle_mspace_destroy(&ctx, &[mspace]), 1);
        assert!(!kernel.libc_mspaces.contains_key(&mspace));

        let registry = HleRegistry::new();
        for function in [
            "sceLibcMspaceCreate",
            "sceLibcMspaceDestroy",
            "sceLibcMspaceFree",
            "sceLibcMspaceMemalign",
            "sceLibcMspaceMallocStats",
            "sceLibcMspaceMallocStatsFast",
            "sceLibcMspaceMallocUsableSize",
        ] {
            assert!(
                registry
                    .call(&ctx, "libSceLibcInternal", function, &[0])
                    .is_some(),
                "{function} must be HLE-reachable"
            );
        }
    }

    /// `sceLibcMspaceRealloc` follows dlmalloc: NULL→malloc, 0→free,
    /// otherwise the bytes survive the move and the old block is reclaimed.
    #[test]
    fn mspace_realloc_moves_bytes_and_reclaims() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x3000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mspace = hle_mspace_create(&ctx, &[0, 0x1000, 0x1000, 0]);
        assert_eq!(mspace, 0x1000);

        // NULL ptr acts as malloc.
        let a = hle_mspace_realloc(&ctx, &[mspace, 0, 0x40]);
        assert_ne!(a, 0);
        assert!(mem.write(a, b"payload!"));

        // Growing preserves the payload and retires the old block.
        let b = hle_mspace_realloc(&ctx, &[mspace, a, 0x80]);
        assert_ne!(b, 0);
        let mut copied = [0u8; 8];
        assert!(mem.read(b, &mut copied));
        assert_eq!(&copied, b"payload!");
        assert_eq!(
            hle_mspace_malloc_usable_size(&ctx, &[a]),
            0,
            "old block was freed"
        );
        assert_eq!(hle_mspace_malloc_usable_size(&ctx, &[b]), 0x80);

        // Size 0 acts as free.
        assert_eq!(hle_mspace_realloc(&ctx, &[mspace, b, 0]), 0);
        assert_eq!(hle_mspace_malloc_usable_size(&ctx, &[b]), 0);

        // Foreign pointer: refused, nothing freed.
        assert_eq!(hle_mspace_realloc(&ctx, &[mspace, 0xDEAD, 0x10]), 0);

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libSceLibcInternal", "sceLibcMspaceRealloc"));
        assert!(registry.is_implemented("libSceLibcInternal", "_ZdlPv"));
        assert!(registry.is_implemented("libSceLibcInternal", "_Stoul"));
        assert!(registry.is_implemented("libSceLibcInternal", "__cxa_finalize"));
        assert!(registry.is_implemented("libSceLibcInternal", "__cxa_pure_virtual"));
        assert!(
            registry.is_implemented("libSceLibcInternalExt", "sceLibcInternalBacktraceForGame")
        );
        assert!(registry.is_implemented(
            "libSceLibcInternalExt",
            "sceLibcInternalHeapErrorReportForGame"
        ));
    }

    /// ASTRO.BOT's global allocator issues a zero-byte mspace allocation and
    /// treats NULL as a fatal OOM. The platform allocator must return distinct,
    /// freeable minimum allocations for that compatibility case.
    #[test]
    fn mspace_malloc_zero_returns_distinct_non_null_allocations() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let mspace = hle_mspace_create(&ctx, &[0, 0x1000, 0x1000, 0]);
        assert_eq!(mspace, 0x1000);

        let first = hle_mspace_malloc(&ctx, &[mspace, 0]);
        let second = hle_mspace_malloc(&ctx, &[mspace, 0]);
        assert_ne!(first, 0, "malloc(0) must not report a false OOM");
        assert_ne!(second, 0, "malloc(0) must not report a false OOM");
        assert_ne!(first, second, "live minimum allocations must be distinct");
        assert_eq!(hle_mspace_free(&ctx, &[mspace, first]), 1);
        assert_eq!(hle_mspace_free(&ctx, &[mspace, second]), 1);
    }

    /// The free list must let malloc/free churn recycle memory instead of the
    /// bump pointer marching to capacity — the ASTRO.BOT "Out of Global Heap
    /// Memory" regression (a fixed-capacity heap OOMing after enough turnover).
    #[test]
    fn mspace_free_list_reclaims_so_churn_does_not_exhaust() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert!(mem.write(0x40, b"heap\0"));
        // 4 KiB mspace; a bump-only allocator exhausts after ~2 of these 0x800
        // allocations, so 200 alloc/free cycles prove the block is reclaimed.
        let mspace = hle_mspace_create(&ctx, &[0x40, 0x1000, 0x1000, 0]);
        assert_eq!(mspace, 0x1000);
        for _ in 0..200 {
            let p = hle_mspace_memalign(&ctx, &[mspace, 0x10, 0x800]);
            assert_ne!(p, 0, "a reclaimed block must be reused, not exhausted");
            assert_eq!(hle_mspace_free(&ctx, &[mspace, p]), 1);
        }
        // After the final free the heap is empty again.
        assert_eq!(kernel.libc_mspaces.get(&mspace).unwrap().active_bytes, 0);
    }
}
