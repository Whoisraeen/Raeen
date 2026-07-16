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
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, warn};

const SCE_OK: u64 = 0;
const ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

/// The `info` block's mandatory `size` field value.
const TRACE_INFO_SIZE: u64 = 32;
/// 8-byte mspace atomic-id mask, then a 64-entry (`u64`) mstate table.
const TRACE_TABLE_ENTRIES: u64 = 64;
const TRACE_STORAGE_BYTES: u64 = 8 + TRACE_TABLE_ENTRIES * 8;

/// Guest address of the persistent trace storage (0 = not yet allocated).
static TRACE_STORAGE: AtomicU64 = AtomicU64::new(0);

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
    let Some(name) = read_cstring(ctx, name_address, 256) else {
        warn!(
            "sceLibcMspaceCreate(base={base:#x}, capacity={capacity:#x}): name at \
             {name_address:#x} unreadable — returning a NULL mspace"
        );
        return 0;
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
    let Some(address) = state.base.checked_add(aligned) else {
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
    let values = [
        state.capacity,
        state.next_offset,
        state.peak_offset,
        state.active_bytes,
    ];
    for (index, value) in values.into_iter().enumerate() {
        if !ctx
            .mem
            .write(output + index as u64 * 8, &value.to_le_bytes())
        {
            return 0;
        }
    }
    1
}

/// Lazily allocate the persistent zeroed trace storage, returning its guest
/// address (0 if the arena is exhausted).
fn trace_storage(ctx: &HleContext) -> u64 {
    let existing = TRACE_STORAGE.load(Ordering::Acquire);
    if existing != 0 {
        return existing;
    }
    let Some(storage) = ctx.alloc.alloc(TRACE_STORAGE_BYTES, 8) else {
        return 0;
    };
    let _ = ctx.mem.write(storage, &[0u8; TRACE_STORAGE_BYTES as usize]);
    match TRACE_STORAGE.compare_exchange(0, storage, Ordering::AcqRel, Ordering::Acquire) {
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

        assert_eq!(hle_mspace_malloc_stats(&ctx, &[mspace, 0x100]), 1);
        let mut value = [0u8; 8];
        assert!(mem.read(0x100, &mut value));
        assert_eq!(u64::from_le_bytes(value), 0x1000);
        assert!(mem.read(0x118, &mut value));
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
}
