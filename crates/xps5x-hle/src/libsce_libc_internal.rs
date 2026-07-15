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
}
