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

use crate::{HleContext, HleRegistry};
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
    registry.register(
        "libSceLibcInternal",
        "sceLibcInternalHeapErrorReportForGame",
        hle_heap_error_report,
    );
    registry.register(
        "libSceLibcInternal",
        "sceLibcInternalBacktraceForGame",
        hle_backtrace_for_game,
    );
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
        xps5x_kernel::LibcMspace {
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
    let size = args.get(1).copied().unwrap_or(0);
    hle_mspace_memalign(ctx, &[mspace, MSPACE_DEFAULT_ALIGN, size])
}

fn hle_mspace_memalign(ctx: &HleContext, args: &[u64]) -> u64 {
    let mspace = args.first().copied().unwrap_or(0);
    let alignment = args.get(1).copied().unwrap_or(0).max(8);
    let size = args.get(2).copied().unwrap_or(0);
    if size == 0 || !alignment.is_power_of_two() {
        return 0;
    }
    let Some(mut state) = ctx.kernel.libc_mspaces.get_mut(&mspace) else {
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
            .insert(address, xps5x_kernel::LibcMspaceAllocation { mspace, size });
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
        return 0;
    };
    if end > state.capacity {
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
        .insert(address, xps5x_kernel::LibcMspaceAllocation { mspace, size });
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
    if std::env::var_os("XPS5X_TRACE_MSPACE").is_some() {
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
        let kernel = xps5x_kernel::OrbisKernel::new();
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
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x100);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_heap_error_report(&ctx, &[1, 2, 3, 4]), SCE_OK);
        assert_eq!(hle_backtrace_for_game(&ctx, &[]), 0);
    }

    #[test]
    fn mspace_create_memalign_stats_free_and_destroy_are_process_local() {
        let kernel = xps5x_kernel::OrbisKernel::new();
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

    /// The free list must let malloc/free churn recycle memory instead of the
    /// bump pointer marching to capacity — the ASTRO.BOT "Out of Global Heap
    /// Memory" regression (a fixed-capacity heap OOMing after enough turnover).
    #[test]
    fn mspace_free_list_reclaims_so_churn_does_not_exhaust() {
        let kernel = xps5x_kernel::OrbisKernel::new();
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
