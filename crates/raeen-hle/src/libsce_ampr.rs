//! HLE libSceAmpr — the AMPR (async processor) command-buffer lifecycle.
//!
//! A faithful Rust port of the command-buffer object management from SharpEmu's
//! `AmprExports` (GPL-2.0). An AMPR command buffer is a small guest struct
//! (self ptr @0x00, data ptr @0x08, size @0x10, aux @0x18/0x20) plus a
//! host-tracked write cursor (`OrbisKernel::ampr_write_offsets`, keyed by the
//! command-buffer address). This ports the construct/destruct/get/set/reset
//! lifecycle, the record writers (`WriteKernelEventQueue`/
//! `WriteAddressOnCompletion`), and `AprCommandBufferReadFile`, whose file
//! read is EAGER at record-append time (SharpEmu AmprExports.cs:255-293) —
//! submission/completion (`raeen_hle::libkernel::apr_complete_command_buffer`)
//! then only services the equeue/write-address records.
//!
//! The GTA V batch (the 46 measured-missing NIDs, 2026-07-27) is a
//! behavioral port of KytyPS5's `src/libs/libAmpr.cpp` (Kyty/MIT lineage):
//! nop/marker/wait commands append inert **self-sizing skip records**
//! (`[type=4][total_size]`, no completion effect — KytyPS5 appends the same
//! zeroed no-ops), the `WriteAddress*` family appends the existing type-3
//! write-address record, the `ReadFileGather`/`Scatter`/`GatherScatter`
//! family continues a host-tracked gather/scatter stream
//! (`OrbisKernel::ampr_gather_scatter`) with the same eager-read model as
//! `ReadFile`, and the APR `MapBegin`/`MapEnd` window is validated and
//! flag-tracked (`OrbisKernel::ampr_type_flags`) without any actual mapping,
//! exactly like KytyPS5. Record sizes are **Raeen's**, not the console's:
//! every `MeasureCommandSize*` returns exactly the byte count its paired
//! writer advances, which is the only invariant a title sizing its buffer by
//! measure calls can observe. Knowingly degraded semantics (waits dropped,
//! counters unmodeled) are registered via `register_incomplete` so coverage
//! tooling never mistakes them for fully working.

use crate::{HleContext, HleRegistry};
use tracing::{debug, warn};

const OK: u64 = 0;
const SCE_ERROR_PERMISSION_DENIED: u64 = 0x8002_0001;
const SCE_ERROR_NOT_FOUND: u64 = 0x8002_0002;
const SCE_ERROR_INVALID_ARGUMENT: u64 = 0x8002_0016;
const SCE_ERROR_MEMORY_FAULT: u64 = 0x8002_000E;

// AmprCommandBuffer struct field offsets (bytes).
const CB_SELF_OFFSET: u64 = 0x00;
const CB_DATA_OFFSET: u64 = 0x08;
const CB_SIZE_OFFSET: u64 = 0x10;
const CB_AUX0_OFFSET: u64 = 0x18;
const CB_AUX1_OFFSET: u64 = 0x20;

// Per-command record sizes reported by the `MeasureCommandSize*` calls.
const READ_FILE_RECORD_SIZE: u64 = 0x30;
const KERNEL_EVENT_QUEUE_RECORD_SIZE: u64 = 0x30;
const WRITE_ADDRESS_RECORD_SIZE: u64 = 0x20;

// --- GTA V batch (KytyPS5 libAmpr.cpp behavioral port) ------------------

/// Self-sizing skip record: `[type:u32 = 4][total_size:u32][payload…]`,
/// `total_size` includes the 8-byte header, always 4-aligned and ≥ 8.
/// Completion (`apr_complete_command_buffer`) skips it — nops, markers,
/// waits, and map bookkeeping have no completion effect (KytyPS5 parity:
/// its `AppendNoOpCommand` writes zeroed bytes and executes nothing).
const NOP_RECORD_TYPE: u32 = 4;
const NOP_RECORD_HEADER: u64 = 8;
/// Wait/counter commands occupy one fixed 0x20-byte skip record (KytyPS5
/// appends 0x20 for `WaitOnAddress`/`WaitOnCounter`/`WriteCounter` too).
const WAIT_RECORD_SIZE: u64 = 0x20;
const POP_MARKER_RECORD_SIZE: u64 = NOP_RECORD_HEADER + 4;
const RESET_GATHER_SCATTER_RECORD_SIZE: u64 = NOP_RECORD_HEADER + 4;
const MAP_BEGIN_RECORD_SIZE: u64 = NOP_RECORD_HEADER + 0xc;
const MAP_DIRECT_BEGIN_RECORD_SIZE: u64 = NOP_RECORD_HEADER + 0x10;
const MAP_END_RECORD_SIZE: u64 = NOP_RECORD_HEADER + 4;
/// KytyPS5 rejects command buffers larger than 64 MiB (`SetBuffer`); the
/// same bound caps a single record so `ConstructNop(bytes=…)` can never ask
/// the host to build a multi-GiB record.
const COMMAND_BUFFER_SIZE_MAX: u64 = 64 * 1024 * 1024;
/// AMM/APR map arguments are 16 KiB-page granular (KytyPS5 `AMM_PAGE_SIZE`).
const AMM_PAGE_SIZE: u64 = 0x4000;
/// APR read-argument validity bounds (KytyPS5 `APR_MAX_*`).
const APR_MAX_READ_LENGTH: u64 = 0x0000_0001_0000_0000;
const APR_MAX_FILE_OFFSET: u64 = 0x0000_0100_0000_0000;
const APR_MAX_APP_ADDRESS: u64 = 0x0000_f000_0000_0000;
/// `sceAmprCommandBufferGetType` flag bits (KytyPS5 keeps these in the guest
/// header's type word; Raeen tracks them in `OrbisKernel::ampr_type_flags`).
const APR_TYPE_GATHER_SCATTER_VALID: u32 = 0x0001_0000;
const APR_TYPE_MAP_ACTIVE: u32 = 0x0002_0000;

/// Register the libSceAmpr command-buffer lifecycle functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAmpr", "sceAmprCommandBufferConstructor", hle_ctor);
    registry.register("libSceAmpr", "sceAmprAprCommandBufferConstructor", hle_ctor);
    registry.register("libSceAmpr", "sceAmprCommandBufferDestructor", hle_dtor);
    registry.register("libSceAmpr", "sceAmprAprCommandBufferDestructor", hle_dtor);
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferSetBuffer",
        hle_set_buffer,
    );
    registry.register("libSceAmpr", "sceAmprCommandBufferReset", hle_reset);
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferClearBuffer",
        hle_clear_buffer,
    );
    registry.register("libSceAmpr", "sceAmprCommandBufferGetSize", hle_get_size);
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferGetCurrentOffset",
        hle_get_current_offset,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferGetNumCommands",
        hle_get_num_commands,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprAprCommandBufferReadFile",
        hle_apr_read_file,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferWriteKernelEventQueue_04_00",
        hle_write_equeue_record,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferWriteAddressOnCompletion",
        hle_write_address_record,
    );
    // MeasureCommandSize* report a command record's byte size.
    registry.register("libSceAmpr", "sceAmprMeasureCommandSizeReadFile", |_, _| {
        READ_FILE_RECORD_SIZE
    });
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWriteKernelEventQueue_04_00",
        |_, _| KERNEL_EVENT_QUEUE_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWriteAddressOnCompletion",
        |_, _| WRITE_ADDRESS_RECORD_SIZE,
    );

    // --- GTA V batch: the 46 measured-missing NIDs (KytyPS5 port) --------
    // Nop / marker family: inert skip records, real cursor/count effects.
    registry.register("libSceAmpr", "sceAmprCommandBufferNop", hle_nop);
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferNopWithData",
        hle_nop_with_data,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferConstructNop",
        hle_construct_nop,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferConstructMarker",
        hle_construct_marker,
    );
    registry.register("libSceAmpr", "sceAmprCommandBufferSetMarker", hle_marker);
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferSetMarkerWithColor",
        hle_marker_with_color,
    );
    registry.register("libSceAmpr", "sceAmprCommandBufferPushMarker", hle_marker);
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferPushMarkerWithColor",
        hle_marker_with_color,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferPopMarker",
        hle_pop_marker,
    );
    // Getters over the visible struct / host-tracked flags.
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferGetType",
        hle_get_type_flags,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferGetBufferBaseAddress",
        hle_get_buffer_base_address,
    );
    // WriteAddress: the real completion effect (type-3 record, performed by
    // apr_complete_command_buffer).
    registry.register(
        "libSceAmpr",
        "sceAmprCommandBufferWriteAddress_04_00",
        hle_write_address_04_00,
    );
    // Waits and counters: KytyPS5 parity — the record is appended (cursor,
    // count, measure size all real) but the wait/counter semantic is
    // dropped, because submissions complete synchronously at submit time
    // and AMPR counters are unmodeled. Honest-error registration so the
    // coverage report keeps naming them.
    registry.register_incomplete(
        "libSceAmpr",
        "sceAmprCommandBufferWaitOnAddress_04_00",
        hle_wait_or_counter_nop,
        "wait dropped: submissions complete synchronously; the address condition is not re-checked (KytyPS5 parity)",
    );
    registry.register_incomplete(
        "libSceAmpr",
        "sceAmprCommandBufferWaitOnCounter_04_00",
        hle_wait_or_counter_nop,
        "wait dropped: submissions complete synchronously; AMPR counters are unmodeled (KytyPS5 parity)",
    );
    registry.register_incomplete(
        "libSceAmpr",
        "sceAmprCommandBufferWriteCounter_04_00",
        hle_wait_or_counter_nop,
        "AMPR counters unmodeled: the counter write is dropped (KytyPS5 parity)",
    );
    registry.register_incomplete(
        "libSceAmpr",
        "sceAmprCommandBufferWriteAddressFromCounter_04_00",
        hle_write_address_from_source,
        "writes 0 at completion: AMPR counters unmodeled (KytyPS5 parity)",
    );
    registry.register_incomplete(
        "libSceAmpr",
        "sceAmprCommandBufferWriteAddressFromCounterPair_04_00",
        hle_write_address_from_source,
        "writes 0 at completion: AMPR counter pairs unmodeled (KytyPS5 parity)",
    );
    registry.register_incomplete(
        "libSceAmpr",
        "sceAmprCommandBufferWriteAddressFromTimeCounter_04_00",
        hle_write_address_from_source,
        "writes 0 at completion: the AMPR time counter is unmodeled (KytyPS5 parity)",
    );
    // Gather/scatter file reads: real data movement (eager, like ReadFile)
    // continuing the host-tracked stream state.
    registry.register(
        "libSceAmpr",
        "sceAmprAprCommandBufferReadFileGather",
        hle_apr_read_file_gather,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprAprCommandBufferReadFileScatter",
        hle_apr_read_file_scatter,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprAprCommandBufferReadFileGatherScatter",
        hle_apr_read_file_gather_scatter,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprAprCommandBufferResetGatherScatterState",
        hle_apr_reset_gather_scatter_state,
    );
    // APR map window: argument validation + window flag are real; nothing is
    // actually mapped (KytyPS5 parity — its records execute no mapping
    // either). Honest-error so a title that depends on the mapping shows up.
    registry.register_incomplete(
        "libSceAmpr",
        "sceAmprAprCommandBufferMapBegin",
        hle_apr_map_begin,
        "validated and window-flagged, but no memory is actually mapped (KytyPS5 parity)",
    );
    registry.register_incomplete(
        "libSceAmpr",
        "sceAmprAprCommandBufferMapDirectBegin",
        hle_apr_map_direct_begin,
        "validated and window-flagged, but no memory is actually mapped (KytyPS5 parity)",
    );
    registry.register_incomplete(
        "libSceAmpr",
        "sceAmprAprCommandBufferMapEnd",
        hle_apr_map_end,
        "closes the window flag only; no mapping was performed (KytyPS5 parity)",
    );
    // MeasureCommandSize*: every one returns exactly the byte count its
    // paired writer advances (the buffer-sizing invariant).
    registry.register("libSceAmpr", "sceAmprMeasureCommandSizeNop", |_, args| {
        measure_nop_size(args.first().copied().unwrap_or(0) as u32)
    });
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeNopWithData",
        |_, args| nop_with_data_record_size(args.first().copied().unwrap_or(0) as u32),
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeSetMarker",
        hle_measure_marker,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeSetMarkerWithColor",
        hle_measure_marker_with_color,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizePushMarker",
        hle_measure_marker,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizePushMarkerWithColor",
        hle_measure_marker_with_color,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizePopMarker",
        |_, _| POP_MARKER_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWriteAddress_04_00",
        |_, _| WRITE_ADDRESS_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWriteAddressFromCounter_04_00",
        |_, _| WRITE_ADDRESS_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWriteAddressFromCounterPair_04_00",
        |_, _| WRITE_ADDRESS_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWriteAddressFromTimeCounter_04_00",
        |_, _| WRITE_ADDRESS_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWaitOnAddress_04_00",
        |_, _| WAIT_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWaitOnCounter_04_00",
        |_, _| WAIT_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeWriteCounter_04_00",
        |_, _| WAIT_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeReadFileGather",
        hle_measure_read_file_gather,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeReadFileScatter",
        hle_measure_read_file_scatter,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeReadFileGatherScatter",
        hle_measure_read_file_gather_scatter,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeResetGatherScatterState",
        |_, _| RESET_GATHER_SCATTER_RECORD_SIZE,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeMapBegin",
        hle_measure_map_begin,
    );
    registry.register(
        "libSceAmpr",
        "sceAmprMeasureCommandSizeMapDirectBegin",
        hle_measure_map_direct_begin,
    );
    registry.register("libSceAmpr", "sceAmprMeasureCommandSizeMapEnd", |_, _| {
        MAP_END_RECORD_SIZE
    });
}

/// Write the command-buffer struct fields (self/data/size/aux) and set the
/// write cursor to `write_offset`.
///
/// The three load-bearing fields (`self`, `data`, `size`) are always written:
/// every later call reads `data`/`size` back out of the guest struct. The two
/// **aux** slots at 0x18/0x20 are different — nothing in Raeen ever reads them,
/// they exist only to zero what SharpEmu zeroes, and `sizeof(SceAmprCommandBuffer)`
/// is not established by anything in-tree. Titles declare the buffer as a stack
/// local (`SceAmprCommandBuffer cb; sceAmprCommandBufferConstructor(&cb, …);`),
/// so if the real struct ends at 0x18 or 0x20 those speculative stores land on
/// the caller's frame — every construct and every reset, i.e. once per command
/// buffer per frame. They are therefore written only where the struct provably
/// is not a caller frame ([`crate::out_buffer`]); on a frame the two slots are
/// left alone, which is exactly as informative as zeroing them.
fn write_cb(ctx: &HleContext, cb: u64, buffer: u64, size: u64, write_offset: u64) -> bool {
    let ok = ctx.mem.write(cb + CB_SELF_OFFSET, &cb.to_le_bytes())
        && ctx.mem.write(cb + CB_DATA_OFFSET, &buffer.to_le_bytes())
        && ctx.mem.write(cb + CB_SIZE_OFFSET, &size.to_le_bytes())
        && ctx.zero_out_object(
            "libSceAmpr::sceAmprCommandBufferConstructor(aux)",
            cb + CB_AUX0_OFFSET,
            (CB_AUX1_OFFSET - CB_AUX0_OFFSET + 8) as usize,
            0,
        );
    if ok {
        ctx.kernel.ampr_write_offsets.insert(cb, write_offset);
        // Construct/reset both rewind through here: the appended-record count
        // starts over with the cursor (SharpEmu zeroes `CommandCount` in its
        // constructor and reset paths), and the gather/scatter continuation
        // state is invalidated (KytyPS5 `WriteCommandBufferPointers` clears
        // it on both paths; the guest-visible type flags survive a reset).
        ctx.kernel.ampr_command_counts.insert(cb, 0);
        ctx.kernel.ampr_gather_scatter.remove(&cb);
    }
    ok
}

/// `sceAmprCommandBufferConstructor(cb, buffer, size)`: initialize the command
/// buffer over `[buffer, buffer+size)` with the cursor at 0. A NULL `cb` is a
/// benign no-op (returns 0). Returns the command-buffer pointer on success.
fn hle_ctor(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let buffer = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    if !write_cb(ctx, cb, buffer, size, 0) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    // Construction zeroes the visible header, type flags included (KytyPS5
    // `InitializeBaseCommandBuffer` memsets the whole header).
    ctx.kernel.ampr_type_flags.remove(&cb);
    debug!("sceAmprCommandBufferConstructor(cb={cb:#x}, buffer={buffer:#x}, size={size:#x})");
    cb
}

/// `sceAmprCommandBufferDestructor(cb)`: drop the tracked write cursor.
fn hle_dtor(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb != 0 {
        ctx.kernel.ampr_write_offsets.remove(&cb);
        ctx.kernel.ampr_command_counts.remove(&cb);
        ctx.kernel.ampr_gather_scatter.remove(&cb);
        ctx.kernel.ampr_type_flags.remove(&cb);
    }
    0
}

/// `sceAmprCommandBufferGetNumCommands(cb)`: return the number of command
/// records appended since construct/reset (SharpEmu's
/// `CommandBufferGetNumCommands`, which reads `state.CommandCount`). A null
/// `cb` is `SCE_ERROR_INVALID_ARGUMENT`; an untracked one is
/// `SCE_ERROR_MEMORY_FAULT` — measured: ASTRO.BOT's async .skel loader calls
/// this on a buffer it just built, and died on the unresolved-import stub
/// before this existed.
fn hle_get_num_commands(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    match ctx.kernel.ampr_command_counts.get(&cb) {
        Some(count) => *count,
        None => SCE_ERROR_MEMORY_FAULT,
    }
}

/// `sceAmprCommandBufferSetBuffer(cb, buffer, size)`: rebind the buffer.
fn hle_set_buffer(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let buffer = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !ctx.mem.write(cb + CB_DATA_OFFSET, &buffer.to_le_bytes())
        || !ctx.mem.write(cb + CB_SIZE_OFFSET, &size.to_le_bytes())
    {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferReset(cb)`: rewind the cursor to 0 (keeping the buffer).
fn hle_reset(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let (mut data, mut size) = ([0u8; 8], [0u8; 8]);
    if !ctx.mem.read(cb + CB_DATA_OFFSET, &mut data)
        || !ctx.mem.read(cb + CB_SIZE_OFFSET, &mut size)
    {
        return SCE_ERROR_MEMORY_FAULT;
    }
    if !write_cb(
        ctx,
        cb,
        u64::from_le_bytes(data),
        u64::from_le_bytes(size),
        0,
    ) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferClearBuffer(cb)`: zero the visible buffer/size
/// pointers in the struct and return the previously-bound buffer pointer.
fn hle_clear_buffer(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut data = [0u8; 8];
    if !ctx.mem.read(cb + CB_DATA_OFFSET, &mut data) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    let old_buffer = u64::from_le_bytes(data);
    if !ctx.mem.write(cb + CB_DATA_OFFSET, &0u64.to_le_bytes())
        || !ctx.mem.write(cb + CB_SIZE_OFFSET, &0u64.to_le_bytes())
    {
        return SCE_ERROR_MEMORY_FAULT;
    }
    old_buffer
}

/// `sceAmprCommandBufferGetSize(cb)`: return the buffer size (in `rax`).
fn hle_get_size(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut buf = [0u8; 8];
    if !ctx.mem.read(cb + CB_SIZE_OFFSET, &mut buf) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    u64::from_le_bytes(buf)
}

/// `sceAmprCommandBufferGetCurrentOffset(cb)`: return the write cursor.
fn hle_get_current_offset(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    ctx.kernel
        .ampr_write_offsets
        .get(&cb)
        .map(|o| *o)
        .unwrap_or(0)
}

/// Append one command record to a command buffer's visible buffer at the
/// host-tracked cursor, advancing the cursor (SharpEmu's
/// `AppendCommandBufferRecord`).
fn append_record(ctx: &HleContext, cb: u64, record: &[u8]) -> bool {
    let mut data = [0u8; 8];
    let mut size = [0u8; 8];
    if !ctx.mem.read(cb + CB_DATA_OFFSET, &mut data)
        || !ctx.mem.read(cb + CB_SIZE_OFFSET, &mut size)
    {
        return false;
    }
    let (buffer, buf_size) = (u64::from_le_bytes(data), u64::from_le_bytes(size));
    let offset = ctx
        .kernel
        .ampr_write_offsets
        .get(&cb)
        .map(|o| *o)
        .unwrap_or(0);
    let record_len = record.len() as u64;
    if buffer == 0 || offset > buf_size || record_len > buf_size - offset {
        return false;
    }
    if !ctx.mem.write(buffer + offset, record) {
        return false;
    }
    ctx.kernel
        .ampr_write_offsets
        .insert(cb, offset + record_len);
    // SharpEmu `AppendCommandBufferRecord` bumps `state.CommandCount` after a
    // successful append; `sceAmprCommandBufferGetNumCommands` reports it.
    ctx.kernel
        .ampr_command_counts
        .entry(cb)
        .and_modify(|count| *count += 1)
        .or_insert(1);
    true
}

/// `sceAmprCommandBufferWriteKernelEventQueue_04_00(cb, equeue, ident, userData)`:
/// append a completion-event record (type 2, 0x30 bytes) that the kernel
/// fires when the buffer completes. SharpEmu `AppendKernelEventQueueRecord`.
fn hle_write_equeue_record(ctx: &HleContext, args: &[u64]) -> u64 {
    const AMPR_FILTER: i16 = -0x64; // KernelEventFilterAmpr (SharpEmu)
    let cb = args.first().copied().unwrap_or(0);
    let equeue = args.get(1).copied().unwrap_or(0);
    let ident = args.get(2).copied().unwrap_or(0);
    let user_data = args.get(3).copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut record = [0u8; 0x30];
    record[0x00..0x04].copy_from_slice(&2u32.to_le_bytes());
    record[0x04..0x06].copy_from_slice(&AMPR_FILTER.to_le_bytes());
    record[0x08..0x10].copy_from_slice(&equeue.to_le_bytes());
    record[0x10..0x18].copy_from_slice(&ident.to_le_bytes());
    record[0x18..0x20].copy_from_slice(&user_data.to_le_bytes());
    record[0x20..0x28].copy_from_slice(&user_data.to_le_bytes());
    if !append_record(ctx, cb, &record) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferWriteAddressOnCompletion(cb, address, value)`:
/// append a write-address record (type 3, 0x20 bytes) the kernel performs at
/// completion. SharpEmu `AppendWriteAddressRecord`.
fn hle_write_address_record(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let address = args.get(1).copied().unwrap_or(0);
    let value = args.get(2).copied().unwrap_or(0);
    if cb == 0 || address == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !append_write_address_record(ctx, cb, address, value) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// Append the type-3 write-address record (0x20 bytes: address @0x08, value
/// @0x10) that `apr_complete_command_buffer` performs at completion. Shared
/// by `WriteAddressOnCompletion`, `WriteAddress_04_00`, and the
/// `WriteAddressFrom*` family.
fn append_write_address_record(ctx: &HleContext, cb: u64, address: u64, value: u64) -> bool {
    let mut record = [0u8; 0x20];
    record[0x00..0x04].copy_from_slice(&3u32.to_le_bytes());
    record[0x08..0x10].copy_from_slice(&address.to_le_bytes());
    record[0x10..0x18].copy_from_slice(&value.to_le_bytes());
    append_record(ctx, cb, &record)
}

// --- GTA V batch implementation (KytyPS5 libAmpr.cpp behavioral port) ----

/// Round up to the AMPR command stream's 4-byte granularity.
fn align4(value: u64) -> u64 {
    (value + 3) & !3u64
}

/// Total size of a self-sizing skip record carrying `payload` bytes.
fn nop_record_size(payload: u64) -> u64 {
    NOP_RECORD_HEADER + align4(payload)
}

/// Set/clear bits in a command buffer's host-tracked type flag word.
fn update_type_flags(ctx: &HleContext, cb: u64, set: u32, clear: u32) {
    let mut entry = ctx.kernel.ampr_type_flags.entry(cb).or_insert(0);
    *entry = (*entry | set) & !clear;
}

/// Append a [`NOP_RECORD_TYPE`] skip record of `total` bytes (header
/// included; must be ≥ 8, 4-aligned, and within the 64 MiB command-buffer
/// bound), embedding up to `total - 8` bytes of `payload` (markers keep
/// their message text; the rest is zeros). No completion effect.
fn append_nop_record(ctx: &HleContext, cb: u64, total: u64, payload: &[u8]) -> bool {
    if total < NOP_RECORD_HEADER || (total & 3) != 0 || total > COMMAND_BUFFER_SIZE_MAX {
        return false;
    }
    let mut record = vec![0u8; total as usize];
    record[0..4].copy_from_slice(&NOP_RECORD_TYPE.to_le_bytes());
    record[4..8].copy_from_slice(&(total as u32).to_le_bytes());
    let n = payload.len().min((total - NOP_RECORD_HEADER) as usize);
    record[8..8 + n].copy_from_slice(&payload[..n]);
    append_record(ctx, cb, &record)
}

// KytyPS5 argument-validity predicates (`IsValidApr*`, `ValidateAmmMapArgs`).
fn is_valid_apr_file_offset(file_offset: u64) -> bool {
    file_offset < APR_MAX_FILE_OFFSET
}
fn is_valid_apr_read_size(size: u64) -> bool {
    size != 0 && size <= APR_MAX_READ_LENGTH
}
fn is_valid_apr_read_range(destination: u64, size: u64) -> bool {
    is_valid_apr_read_size(size)
        && destination <= APR_MAX_APP_ADDRESS
        && APR_MAX_APP_ADDRESS - destination >= size
}
fn is_valid_amm_map_args(va: u64, size: u64) -> bool {
    va != 0
        && size != 0
        && (va & (AMM_PAGE_SIZE - 1)) == 0
        && (size & (AMM_PAGE_SIZE - 1)) == 0
        && va.checked_add(size).is_some()
}

/// `sceAmprCommandBufferNop(cb, numDwords)`: append `numDwords` (1..=16)
/// dwords of inert padding (KytyPS5 `CommandBufferNop`). Our record adds the
/// 8-byte self-sizing header — [`measure_nop_size`] reports the same total.
fn hle_nop(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let num_u32 = args.get(1).copied().unwrap_or(0) as u32;
    if cb == 0 || num_u32 == 0 || num_u32 > 16 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !append_nop_record(ctx, cb, nop_record_size(u64::from(num_u32) * 4), &[]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// The size [`hle_nop`] advances for `num_u32` dwords — `MeasureCommandSizeNop`.
/// KytyPS5 measures one dword for `num == 0` (its writer rejects 0 too).
fn measure_nop_size(num_u32: u32) -> u64 {
    nop_record_size(u64::from(num_u32.max(1)) * 4)
}

/// Record size shared by `NopWithData`'s writer and measure: the payload is
/// `num_u32` dwords plus the one-dword packet header KytyPS5 accounts for.
fn nop_with_data_record_size(num_u32: u32) -> u64 {
    nop_record_size((u64::from(num_u32) + 1) * 4)
}

/// `sceAmprCommandBufferNopWithData(cb, numDwords, data)`: inert padding
/// carrying `numDwords` (0..=15) dwords of caller data. KytyPS5 drops the
/// payload; we embed it in the skip record when readable (real NOP-with-data
/// packets carry their annotation payload) — either way it has no effect.
fn hle_nop_with_data(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let num_u32 = args.get(1).copied().unwrap_or(0) as u32;
    let data_ptr = args.get(2).copied().unwrap_or(0);
    if cb == 0 || num_u32 > 15 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let mut payload = vec![0u8; num_u32 as usize * 4];
    if data_ptr != 0 && !payload.is_empty() && !ctx.mem.read(data_ptr, &mut payload) {
        payload.fill(0); // unreadable annotation data: keep the record inert
    }
    if !append_nop_record(ctx, cb, nop_with_data_record_size(num_u32), &payload) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferConstructNop(cb, _, _, sizeBytes, _)`: append an
/// inert command of `sizeBytes` payload bytes plus the one-dword packet
/// header (KytyPS5 `CommandBufferConstructNop`).
fn hle_construct_nop(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let bytes = args.get(3).copied().unwrap_or(0) as u32;
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !append_nop_record(ctx, cb, nop_record_size(4 + u64::from(bytes)), &[]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// Byte length of a marker command: one dword of packet header (two with a
/// color word) plus the NUL-terminated message — KytyPS5
/// `MarkerCommandSize` — under our 8-byte record header. Measure and writer
/// MUST read the string identically, so both funnel through here.
fn marker_record_size_and_text(ctx: &HleContext, msg_ptr: u64, with_color: bool) -> (u64, Vec<u8>) {
    let text = if msg_ptr == 0 {
        Vec::new()
    } else {
        crate::fmt::read_cstr(ctx.mem, msg_ptr).unwrap_or_default()
    };
    let msg_size = if text.is_empty() && msg_ptr == 0 {
        0
    } else {
        text.len() as u64 + 1
    };
    let header: u64 = if with_color { 8 } else { 4 };
    (nop_record_size(header + msg_size), text)
}

/// `sceAmprCommandBufferSetMarker` / `PushMarker` `(cb, msg)`: inert debug
/// annotation; the message text is embedded in the skip record.
fn hle_marker(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let msg = args.get(1).copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let (total, text) = marker_record_size_and_text(ctx, msg, false);
    if !append_nop_record(ctx, cb, total, &text) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferSetMarkerWithColor` / `PushMarkerWithColor`
/// `(cb, msg, color)`.
fn hle_marker_with_color(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let msg = args.get(1).copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let (total, text) = marker_record_size_and_text(ctx, msg, true);
    if !append_nop_record(ctx, cb, total, &text) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferConstructMarker(cb, _, msg, colorPtr)`: marker with
/// the color word present iff `colorPtr` is non-null (KytyPS5
/// `CommandBufferConstructMarker`).
fn hle_construct_marker(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let msg = args.get(2).copied().unwrap_or(0);
    let color_ptr = args.get(3).copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let (total, text) = marker_record_size_and_text(ctx, msg, color_ptr != 0);
    if !append_nop_record(ctx, cb, total, &text) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferPopMarker(cb)`: one-dword marker-stack pop.
fn hle_pop_marker(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !append_nop_record(ctx, cb, POP_MARKER_RECORD_SIZE, &[]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `MeasureCommandSizeSetMarker` / `PushMarker` `(msg)`.
fn hle_measure_marker(ctx: &HleContext, args: &[u64]) -> u64 {
    marker_record_size_and_text(ctx, args.first().copied().unwrap_or(0), false).0
}

/// `MeasureCommandSize{Set,Push}MarkerWithColor(msg, color)`.
fn hle_measure_marker_with_color(ctx: &HleContext, args: &[u64]) -> u64 {
    marker_record_size_and_text(ctx, args.first().copied().unwrap_or(0), true).0
}

/// `sceAmprCommandBufferWaitOnAddress_04_00` / `WaitOnCounter_04_00` /
/// `WriteCounter_04_00`: the record is appended (cursor, count, and the
/// paired measure size are all real) but the semantic is dropped — this
/// runtime completes submissions synchronously at submit time, and AMPR
/// counters are unmodeled (KytyPS5 appends the same 0x20-byte no-op).
fn hle_wait_or_counter_nop(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !append_nop_record(ctx, cb, WAIT_RECORD_SIZE, &[]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferWriteAddress_04_00(cb, address, value, flags)`: the
/// completion writes `value` to `address` — same type-3 record (and same
/// completion effect) as `WriteAddressOnCompletion` (KytyPS5 routes both
/// through `AppendWriteAddressCommand`).
fn hle_write_address_04_00(ctx: &HleContext, args: &[u64]) -> u64 {
    hle_write_address_record(ctx, args)
}

/// `sceAmprCommandBufferWriteAddressFrom{Counter,CounterPair,TimeCounter}
/// _04_00(cb, address, …)`: completion writes **0** to `address` because the
/// counter source is unmodeled (KytyPS5 parity — it appends the same
/// write-address command with value 0). Registered incomplete.
fn hle_write_address_from_source(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let address = args.get(1).copied().unwrap_or(0);
    if cb == 0 || address == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !append_write_address_record(ctx, cb, address, 0) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
}

/// `sceAmprCommandBufferGetType(cb)`: the host-tracked type flag word (see
/// [`APR_TYPE_GATHER_SCATTER_VALID`] / [`APR_TYPE_MAP_ACTIVE`]). KytyPS5
/// reads the same bits from its guest header; a null or never-flagged
/// buffer reads 0.
fn hle_get_type_flags(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    ctx.kernel
        .ampr_type_flags
        .get(&cb)
        .map(|flags| u64::from(*flags))
        .unwrap_or(0)
}

/// `sceAmprCommandBufferGetBufferBaseAddress(cb)`: the bound backing-buffer
/// pointer (the struct's data field); null/unreadable reads 0.
fn hle_get_buffer_base_address(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return 0;
    }
    let mut buf = [0u8; 8];
    if !ctx.mem.read(cb + CB_DATA_OFFSET, &mut buf) {
        return 0;
    }
    u64::from_le_bytes(buf)
}

/// `sceAmprAprCommandBufferReadFileGather(cb, _, _, size, fileOffset)`: read
/// `size` bytes at the NEW `fileOffset` into the destination where the
/// previous read left off (KytyPS5 `AprCommandBufferReadFileGather`).
/// Requires live gather/scatter state (a prior `ReadFile*` on this buffer).
fn hle_apr_read_file_gather(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let size = args.get(3).copied().unwrap_or(0);
    let file_offset = args.get(4).copied().unwrap_or(0);
    if cb == 0 || !is_valid_apr_read_size(size) || !is_valid_apr_file_offset(file_offset) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let Some(gs) = ctx.kernel.ampr_gather_scatter.get(&cb).map(|g| *g) else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    if !is_valid_apr_read_range(gs.next_destination, size) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    apr_read_and_append(ctx, cb, gs.file_id, gs.next_destination, size, file_offset)
}

/// `sceAmprAprCommandBufferReadFileScatter(cb, _, _, destination, size)`:
/// continue the file stream sequentially (offset where the previous read
/// left off) into a NEW destination (KytyPS5 `AprCommandBufferReadFileScatter`).
fn hle_apr_read_file_scatter(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let destination = args.get(3).copied().unwrap_or(0);
    let size = args.get(4).copied().unwrap_or(0);
    if cb == 0 || !is_valid_apr_read_range(destination, size) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let Some(gs) = ctx.kernel.ampr_gather_scatter.get(&cb).map(|g| *g) else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    if !is_valid_apr_file_offset(gs.next_file_offset) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    apr_read_and_append(ctx, cb, gs.file_id, destination, size, gs.next_file_offset)
}

/// `sceAmprAprCommandBufferReadFileGatherScatter(cb, _, _, destination,
/// size, fileOffset)`: both sides given; only the file id continues from the
/// gather/scatter state (KytyPS5 `AprCommandBufferReadFileGatherScatter`).
fn hle_apr_read_file_gather_scatter(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let destination = args.get(3).copied().unwrap_or(0);
    let size = args.get(4).copied().unwrap_or(0);
    let file_offset = args.get(5).copied().unwrap_or(0);
    if cb == 0
        || !is_valid_apr_read_range(destination, size)
        || !is_valid_apr_file_offset(file_offset)
    {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    let Some(gs) = ctx.kernel.ampr_gather_scatter.get(&cb).map(|g| *g) else {
        return SCE_ERROR_INVALID_ARGUMENT;
    };
    apr_read_and_append(ctx, cb, gs.file_id, destination, size, file_offset)
}

/// `sceAmprAprCommandBufferResetGatherScatterState(cb, _, _)`: append the
/// one-dword reset command, invalidate the continuation state, and lower
/// the `GATHER_SCATTER_VALID` type flag.
fn hle_apr_reset_gather_scatter_state(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !append_nop_record(ctx, cb, RESET_GATHER_SCATTER_RECORD_SIZE, &[]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    ctx.kernel.ampr_gather_scatter.remove(&cb);
    update_type_flags(ctx, cb, 0, APR_TYPE_GATHER_SCATTER_VALID);
    OK
}

/// `sceAmprAprCommandBufferMapBegin(cb, va, size, type, prot)`: validate the
/// 16 KiB-granular window and raise `MAP_ACTIVE`. No memory is actually
/// mapped (KytyPS5 parity); registered incomplete.
fn hle_apr_map_begin(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let va = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    if cb == 0 || !is_valid_amm_map_args(va, size) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !append_nop_record(ctx, cb, MAP_BEGIN_RECORD_SIZE, &[]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    update_type_flags(ctx, cb, APR_TYPE_MAP_ACTIVE, 0);
    OK
}

/// `sceAmprAprCommandBufferMapDirectBegin(cb, va, dmemOffset, size, type,
/// prot)`: like MapBegin plus the direct-memory offset (also page-granular).
fn hle_apr_map_direct_begin(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let va = args.get(1).copied().unwrap_or(0);
    let dmem_offset = args.get(2).copied().unwrap_or(0);
    let size = args.get(3).copied().unwrap_or(0);
    if cb == 0 || !is_valid_amm_map_args(va, size) || (dmem_offset & (AMM_PAGE_SIZE - 1)) != 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    if !append_nop_record(ctx, cb, MAP_DIRECT_BEGIN_RECORD_SIZE, &[]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    update_type_flags(ctx, cb, APR_TYPE_MAP_ACTIVE, 0);
    OK
}

/// `sceAmprAprCommandBufferMapEnd(cb)`: closes an open map window — EPERM
/// when none is active (KytyPS5 `AprCommandBufferMapEnd`).
fn hle_apr_map_end(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb == 0 {
        return SCE_ERROR_PERMISSION_DENIED;
    }
    let active = ctx
        .kernel
        .ampr_type_flags
        .get(&cb)
        .map(|flags| *flags & APR_TYPE_MAP_ACTIVE != 0)
        .unwrap_or(false);
    if !active {
        return SCE_ERROR_PERMISSION_DENIED;
    }
    if !append_nop_record(ctx, cb, MAP_END_RECORD_SIZE, &[]) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    update_type_flags(ctx, cb, 0, APR_TYPE_MAP_ACTIVE);
    OK
}

/// `MeasureCommandSizeReadFileGather(size, fileOffset)` — validated like the
/// writer; the size is Raeen's 0x30 ReadFile record.
fn hle_measure_read_file_gather(_ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.first().copied().unwrap_or(0);
    let file_offset = args.get(1).copied().unwrap_or(0);
    if !is_valid_apr_read_size(size) || !is_valid_apr_file_offset(file_offset) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    READ_FILE_RECORD_SIZE
}

/// `MeasureCommandSizeReadFileScatter(destination, size)`.
fn hle_measure_read_file_scatter(_ctx: &HleContext, args: &[u64]) -> u64 {
    let destination = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    if !is_valid_apr_read_range(destination, size) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    READ_FILE_RECORD_SIZE
}

/// `MeasureCommandSizeReadFileGatherScatter(destination, size, fileOffset)`.
fn hle_measure_read_file_gather_scatter(_ctx: &HleContext, args: &[u64]) -> u64 {
    let destination = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    let file_offset = args.get(2).copied().unwrap_or(0);
    if !is_valid_apr_read_range(destination, size) || !is_valid_apr_file_offset(file_offset) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    READ_FILE_RECORD_SIZE
}

/// `MeasureCommandSizeMapBegin(va, size, type, prot)` — EINVAL (as the
/// return value, KytyPS5 `MeasureAprCommandSizeMapBegin`) on bad alignment.
fn hle_measure_map_begin(_ctx: &HleContext, args: &[u64]) -> u64 {
    let va = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0);
    if !is_valid_amm_map_args(va, size) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    MAP_BEGIN_RECORD_SIZE
}

/// `MeasureCommandSizeMapDirectBegin(va, dmemOffset, size, type, prot)`.
fn hle_measure_map_direct_begin(_ctx: &HleContext, args: &[u64]) -> u64 {
    let va = args.first().copied().unwrap_or(0);
    let dmem_offset = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    if !is_valid_amm_map_args(va, size) || (dmem_offset & (AMM_PAGE_SIZE - 1)) != 0 {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    MAP_DIRECT_BEGIN_RECORD_SIZE
}

/// One positional read at an absolute file offset (pread-style), so a cached
/// handle's shared cursor is never disturbed. SharpEmu uses
/// `RandomAccess.Read` (AmprExports.cs:796-799).
#[cfg(windows)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

/// `read_at` for Unix hosts (`pread`).
#[cfg(unix)]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

/// `read_at` fallback for hosts without a positional-read ext trait: a
/// private clone of the handle seeks without disturbing the cached one.
#[cfg(not(any(windows, unix)))]
fn read_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::io::{Read, Seek, SeekFrom};
    let mut clone = file.try_clone()?;
    clone.seek(SeekFrom::Start(offset))?;
    clone.read(buf)
}

/// Map a host I/O error to the SCE error SharpEmu returns for it
/// (`TryGetCachedHostFile`/`TryReadFileToGuestMemory`: UnauthorizedAccess →
/// PERMISSION_DENIED, other I/O → NOT_FOUND; AmprExports.cs:854-865,
/// 813-821).
fn apr_io_error(err: &std::io::Error) -> u64 {
    match err.kind() {
        std::io::ErrorKind::PermissionDenied => SCE_ERROR_PERMISSION_DENIED,
        _ => SCE_ERROR_NOT_FOUND,
    }
}

/// Read `size` bytes of the host file backing APR `file_id` at `file_offset`
/// into guest `destination`, eagerly, with read-EXACT semantics: loop
/// positional reads until the request is filled or EOF (a short count is OK
/// only at EOF or an offset past end-of-file). Returns `Ok(bytes_read)` or
/// the SCE error to hand the guest. Faithful port of SharpEmu
/// `AmprExports.TryReadFileToGuestMemory` (AmprExports.cs:748-828) with its
/// open-handle cache (`TryGetCachedHostFile`, AmprExports.cs:830-866) keyed
/// here by APR id (`OrbisKernel::appr_file_handles`).
fn apr_read_file_into_guest(
    ctx: &HleContext,
    file_id: u32,
    host_path: &str,
    file_offset: u64,
    destination: u64,
    size: u64,
) -> Result<u64, u64> {
    if size == 0 {
        return Ok(0);
    }
    if file_offset > i64::MAX as u64 {
        return Err(SCE_ERROR_INVALID_ARGUMENT);
    }
    // Cached open handle per APR id (SharpEmu `_hostFileCache`): a title
    // re-reading one asset must not re-open it per command record.
    if !ctx.kernel.appr_file_handles.contains_key(&file_id) {
        match std::fs::File::open(host_path) {
            Ok(file) => {
                ctx.kernel.appr_file_handles.insert(file_id, file);
            }
            Err(err) => return Err(apr_io_error(&err)),
        }
    }
    let file = ctx
        .kernel
        .appr_file_handles
        .get(&file_id)
        .ok_or(SCE_ERROR_NOT_FOUND)?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if file_offset >= file_len {
        return Ok(0);
    }
    let mut bytes_read = 0u64;
    let mut chunk = vec![0u8; size.min(1024 * 1024) as usize];
    while bytes_read < size {
        let absolute = file_offset
            .checked_add(bytes_read)
            .ok_or(SCE_ERROR_INVALID_ARGUMENT)?;
        if absolute > i64::MAX as u64 {
            return Err(SCE_ERROR_INVALID_ARGUMENT);
        }
        let want = ((size - bytes_read).min(chunk.len() as u64)) as usize;
        let read = match read_at(&file, &mut chunk[..want], absolute) {
            Ok(0) => break, // EOF
            Ok(n) => n,
            Err(err) => return Err(apr_io_error(&err)),
        };
        if !ctx.mem.write(destination + bytes_read, &chunk[..read]) {
            return Err(SCE_ERROR_MEMORY_FAULT);
        }
        bytes_read += read as u64;
    }
    Ok(bytes_read)
}

/// The once-per-fileId "name the miss" diagnostic: an unregistered APR id
/// means path resolution failed (or the title never resolved the path), and
/// the asset behind this id is silently lost unless the log names it. This
/// is the diagnostic whose absence made the ASTRO.BOT pause-menu asset loss
/// take days to find.
fn warn_missing_apr_file_once(ctx: &HleContext, file_id: u32) {
    if ctx.kernel.appr_missing_warned.insert(file_id, ()).is_none() {
        warn!(
            "sceAmprAprCommandBufferReadFile: fileId {file_id:#010x} has no registered host path \
             (APR path resolution failed or was never requested) — returning NOT_FOUND with guest \
             memory untouched (SharpEmu AmprExports.cs:272-276); the asset for this id is LOST"
        );
    }
}

/// `sceAmprAprCommandBufferReadFile(cb, _, _, fileId, destination, size,
/// fileOffset)`: read `size` bytes of APR file `fileId` at `fileOffset` into
/// guest `destination` and append a ReadFile command record.
/// `fileOffset` is SysV arg7 (`args[6]`, on the stack at `[Rsp+8]`,
/// captured by the runtime dispatch).
///
/// Faithful port of SharpEmu `AprCommandBufferReadFile`
/// (AmprExports.cs:255-293): the read happens EAGERLY at record-append time
/// — the bytes land in guest memory before any submit — and `bytesRead` is
/// recorded in the record at append (@0x20; `AppendReadFileRecord`,
/// AmprExports.cs:868-887). Completion later skips the record because the
/// data is already in place. An unregistered id is NOT_FOUND (no zero-fill,
/// no record, guest memory untouched; AmprExports.cs:272-276) plus a
/// once-per-id warn naming the lost asset.
pub(crate) fn hle_apr_read_file(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    let file_id = args.get(3).copied().unwrap_or(0) as u32;
    let destination = args.get(4).copied().unwrap_or(0);
    let size = args.get(5).copied().unwrap_or(0);
    let file_offset = args.get(6).copied().unwrap_or(0);
    if cb == 0 || (destination == 0 && size != 0) {
        return SCE_ERROR_INVALID_ARGUMENT;
    }
    apr_read_and_append(ctx, cb, file_id, destination, size, file_offset)
}

/// Shared body of every `AprCommandBufferReadFile*` writer: resolve the APR
/// id, read EAGERLY into guest memory, append the 0x30 ReadFile record with
/// `bytesRead` @0x20, then advance the gather/scatter continuation state
/// (KytyPS5 `AppendReadFileRecord`: the file id sticks, destination and file
/// offset continue past the bytes just requested) and raise the
/// `GATHER_SCATTER_VALID` type flag.
fn apr_read_and_append(
    ctx: &HleContext,
    cb: u64,
    file_id: u32,
    destination: u64,
    size: u64,
    file_offset: u64,
) -> u64 {
    let Some(host_path) = ctx.kernel.appr_host_path(file_id) else {
        warn_missing_apr_file_once(ctx, file_id);
        return SCE_ERROR_NOT_FOUND;
    };
    let bytes_read =
        match apr_read_file_into_guest(ctx, file_id, &host_path, file_offset, destination, size) {
            Ok(n) => n,
            Err(err) => return err,
        };
    let mut record = [0u8; 0x30];
    record[0x00..0x04].copy_from_slice(&1u32.to_le_bytes());
    record[0x04..0x08].copy_from_slice(&file_id.to_le_bytes());
    record[0x08..0x10].copy_from_slice(&destination.to_le_bytes());
    record[0x10..0x18].copy_from_slice(&size.to_le_bytes());
    record[0x18..0x20].copy_from_slice(&file_offset.to_le_bytes());
    record[0x20..0x28].copy_from_slice(&bytes_read.to_le_bytes());
    if !append_record(ctx, cb, &record) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    // KytyPS5 advances the continuation by the REQUESTED size, not by
    // bytes_read; wrapping adds are unreachable behind the validated APR
    // bounds but must never panic on the legacy (unvalidated) ReadFile path.
    ctx.kernel.ampr_gather_scatter.insert(
        cb,
        raeen_kernel::AmprGatherScatterState {
            file_id,
            next_destination: destination.wrapping_add(size),
            next_file_offset: file_offset.wrapping_add(size),
        },
    );
    update_type_flags(ctx, cb, APR_TYPE_GATHER_SCATTER_VALID, 0);
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_ctx;

    fn ctx_env() -> (
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    fn read_u64(ctx: &HleContext, addr: u64) -> u64 {
        let mut b = [0u8; 8];
        assert!(ctx.mem.read(addr, &mut b));
        u64::from_le_bytes(b)
    }

    #[test]
    fn num_commands_tracks_appends_and_resets() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x100;
        // Null and untracked buffers follow SharpEmu's error mapping.
        assert_eq!(hle_get_num_commands(&ctx, &[0]), SCE_ERROR_INVALID_ARGUMENT);
        assert_eq!(hle_get_num_commands(&ctx, &[cb]), SCE_ERROR_MEMORY_FAULT);
        // Construct starts the count at zero (buffer inside the 0x400-byte
        // test memory so the appended records are writable).
        assert_eq!(hle_ctor(&ctx, &[cb, 0x200, 0x100]), cb);
        assert_eq!(hle_get_num_commands(&ctx, &[cb]), 0);
        // Each appended record bumps it (two equeue records here).
        assert_eq!(hle_write_equeue_record(&ctx, &[cb, 0x9000, 1, 2]), OK);
        assert_eq!(hle_write_equeue_record(&ctx, &[cb, 0x9000, 3, 4]), OK);
        assert_eq!(hle_get_num_commands(&ctx, &[cb]), 2);
        // Reset rewinds the count with the cursor.
        assert_eq!(hle_reset(&ctx, &[cb]), OK);
        assert_eq!(hle_get_num_commands(&ctx, &[cb]), 0);
        // Destructor drops the state entirely.
        hle_dtor(&ctx, &[cb]);
        assert_eq!(hle_get_num_commands(&ctx, &[cb]), SCE_ERROR_MEMORY_FAULT);
    }

    #[test]
    fn construct_get_set_reset_lifecycle() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x100;
        // Construct binds buffer/size + zeroes the cursor, returns cb.
        assert_eq!(hle_ctor(&ctx, &[cb, 0x1000, 0x800]), cb);
        assert_eq!(read_u64(&ctx, cb + CB_SELF_OFFSET), cb);
        assert_eq!(read_u64(&ctx, cb + CB_DATA_OFFSET), 0x1000);
        assert_eq!(hle_get_size(&ctx, &[cb]), 0x800);
        assert_eq!(hle_get_current_offset(&ctx, &[cb]), 0);
        // Advance the cursor (as a writer would), then Reset rewinds it.
        kernel.ampr_write_offsets.insert(cb, 0x40);
        assert_eq!(hle_get_current_offset(&ctx, &[cb]), 0x40);
        assert_eq!(hle_reset(&ctx, &[cb]), OK);
        assert_eq!(hle_get_current_offset(&ctx, &[cb]), 0);
        // SetBuffer rebinds.
        assert_eq!(hle_set_buffer(&ctx, &[cb, 0x2000, 0x400]), OK);
        assert_eq!(hle_get_size(&ctx, &[cb]), 0x400);
        assert_eq!(read_u64(&ctx, cb + CB_DATA_OFFSET), 0x2000);
        // ClearBuffer returns the bound buffer and zeroes the visible pointers.
        assert_eq!(hle_clear_buffer(&ctx, &[cb]), 0x2000);
        assert_eq!(read_u64(&ctx, cb + CB_DATA_OFFSET), 0);
        assert_eq!(read_u64(&ctx, cb + CB_SIZE_OFFSET), 0);
        // Destruct drops the cursor state; NULL cb constructor is a no-op.
        assert_eq!(hle_dtor(&ctx, &[cb]), 0);
        assert!(!kernel.ampr_write_offsets.contains_key(&cb));
        assert_eq!(hle_ctor(&ctx, &[0, 0, 0]), 0);
        // Getters reject a NULL command buffer.
        assert_eq!(hle_get_size(&ctx, &[0]), SCE_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn measure_command_sizes_report_record_bytes() {
        let reg = HleRegistry::new();
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x10);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            reg.call(&ctx, "libSceAmpr", "sceAmprMeasureCommandSizeReadFile", &[]),
            Some(READ_FILE_RECORD_SIZE)
        );
        assert_eq!(
            reg.call(
                &ctx,
                "libSceAmpr",
                "sceAmprMeasureCommandSizeWriteAddressOnCompletion",
                &[]
            ),
            Some(WRITE_ADDRESS_RECORD_SIZE)
        );
    }

    /// Write a patterned temp host file and return its path (caller deletes).
    fn temp_host_file(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let host = std::env::temp_dir().join(format!(
            "raeen_ampr_test_{}_{}_{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&host, bytes).expect("temp file");
        host
    }

    /// The ReadFile read is EAGER: the guest destination must hold the file
    /// bytes the moment `sceAmprAprCommandBufferReadFile` returns, before any
    /// submit/completion (SharpEmu `AprCommandBufferReadFile`,
    /// AmprExports.cs:255-293).
    #[test]
    fn apr_read_file_populates_guest_memory_at_append_not_submit() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let data: Vec<u8> = (0..256u32).map(|i| (i * 7 % 251) as u8).collect();
        let host = temp_host_file("eager", &data);
        let id = kernel.appr_register_file("/app0/assets/eager.bin", host.display().to_string());

        let cb: u64 = 0x40;
        let buf: u64 = 0x100;
        let dst: u64 = 0x300;
        assert_eq!(hle_ctor(&ctx, &[cb, buf, 0x100]), cb);
        assert_eq!(
            hle_apr_read_file(&ctx, &[cb, 0, 0, u64::from(id), dst, 256, 0]),
            OK
        );
        // BEFORE any submit: the guest destination already holds the bytes.
        let mut guest = vec![0u8; 256];
        assert!(ctx.mem.read(dst, &mut guest));
        assert_eq!(guest, data, "the read must be eager at record-append time");
        // bytesRead is recorded in the command record at append time (@0x20).
        assert_eq!(read_u64(&ctx, buf + 0x20), 256);
        let _ = std::fs::remove_file(&host);
    }

    /// Read-EXACT semantics: a read larger than any single host `read` call
    /// would deliver must still fill the entire destination (loop until full
    /// or EOF — SharpEmu `TryReadFileToGuestMemory`, AmprExports.cs:782-812).
    #[test]
    fn apr_read_file_exact_read_fills_full_destination() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x40000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // > 64 KiB: a naive single small read would come back short.
        let n = 100_000usize;
        let data: Vec<u8> = (0..n as u32).map(|i| (i * 31 % 256) as u8).collect();
        let host = temp_host_file("exact", &data);
        let id = kernel.appr_register_file("/app0/assets/exact.bin", host.display().to_string());

        let cb: u64 = 0x100;
        let buf: u64 = 0x1000;
        let dst: u64 = 0x10000;
        assert_eq!(hle_ctor(&ctx, &[cb, buf, 0x1000]), cb);
        assert_eq!(
            hle_apr_read_file(&ctx, &[cb, 0, 0, u64::from(id), dst, n as u64, 0]),
            OK
        );
        let mut guest = vec![0u8; n];
        assert!(ctx.mem.read(dst, &mut guest));
        assert_eq!(guest, data, "all {n} bytes must land (read-exact loop)");
        assert_eq!(read_u64(&ctx, buf + 0x20), n as u64);

        // A request running past EOF records the short count and stops
        // (SharpEmu: the loop breaks on a zero read; OK with partial bytes).
        assert_eq!(
            hle_apr_read_file(
                &ctx,
                &[cb, 0, 0, u64::from(id), dst, 0x2000, (n - 0x1000) as u64]
            ),
            OK
        );
        // Second record at buf+0x30; bytesRead must be the 0x1000 available.
        assert_eq!(read_u64(&ctx, buf + 0x30 + 0x20), 0x1000);
        let _ = std::fs::remove_file(&host);
    }

    /// Unresolvable fileId: SharpEmu returns NOT_FOUND and does NOT touch
    /// guest memory or append a record (AmprExports.cs:272-276) — no silent
    /// zero-fill. A once-per-fileId warn names the lost asset.
    #[test]
    fn apr_read_file_missing_file_logs_and_matches_sharpemu_semantics() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x100;
        let dst: u64 = 0x200;
        assert_eq!(hle_ctor(&ctx, &[cb, 0x300, 0x100]), cb);
        // Pre-dirty the destination: a missing file must leave it alone.
        assert!(ctx.mem.write(dst, &[0xEEu8; 16]));
        assert_eq!(
            hle_apr_read_file(&ctx, &[cb, 0, 0, 7, dst, 16, 0]),
            SCE_ERROR_NOT_FOUND
        );
        let mut probe = [0u8; 16];
        assert!(ctx.mem.read(dst, &mut probe));
        assert_eq!(
            probe, [0xEEu8; 16],
            "a missing file must not zero-fill (SharpEmu parity)"
        );
        // No record was appended.
        assert_eq!(hle_get_current_offset(&ctx, &[cb]), 0);
        // The once-per-fileId name-the-miss warn fired (the rate-limit set is
        // the observable artifact; the log line itself names the id).
        assert!(kernel.appr_missing_warned.contains_key(&7));
        // A repeat stays NOT_FOUND without re-warning.
        assert_eq!(
            hle_apr_read_file(&ctx, &[cb, 0, 0, 7, dst, 16, 0]),
            SCE_ERROR_NOT_FOUND
        );
        assert_eq!(kernel.appr_missing_warned.len(), 1);
        // Argument validation is unchanged, and checked before the registry.
        assert_eq!(
            hle_apr_read_file(&ctx, &[0, 0, 0, 7, dst, 16, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_apr_read_file(&ctx, &[cb, 0, 0, 7, 0, 16, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        // SharpEmu checks the registry before the size: an unregistered id is
        // NOT_FOUND even for a zero-size read.
        assert_eq!(
            hle_apr_read_file(&ctx, &[cb, 0, 0, 7, 0, 0, 0]),
            SCE_ERROR_NOT_FOUND
        );
    }

    // --- GTA V batch (the 46 measured-missing NIDs) ----------------------

    /// The 46 `libSceAmpr` names GTA V (PPSA04264 v01.005.000) imports and
    /// could not resolve (artifacts/compat/nid-coverage.json, 2026-07-27).
    const GTA5_MISSING_AMPR_NIDS: [&str; 46] = [
        "sceAmprAprCommandBufferReadFileGatherScatter",
        "sceAmprMeasureCommandSizeWaitOnCounter_04_00",
        "sceAmprCommandBufferWaitOnAddress_04_00",
        "sceAmprMeasureCommandSizeReadFileGatherScatter",
        "sceAmprAprCommandBufferMapBegin",
        "sceAmprCommandBufferConstructNop",
        "sceAmprMeasureCommandSizeWriteCounter_04_00",
        "sceAmprMeasureCommandSizeWriteAddressFromCounter_04_00",
        "sceAmprAprCommandBufferReadFileScatter",
        "sceAmprMeasureCommandSizeNop",
        "sceAmprCommandBufferGetBufferBaseAddress",
        "sceAmprCommandBufferGetType",
        "sceAmprMeasureCommandSizeSetMarker",
        "sceAmprMeasureCommandSizeNopWithData",
        "sceAmprAprCommandBufferMapEnd",
        "sceAmprAprCommandBufferResetGatherScatterState",
        "sceAmprAprCommandBufferMapDirectBegin",
        "sceAmprCommandBufferWriteAddressFromTimeCounter_04_00",
        "sceAmprCommandBufferWaitOnCounter_04_00",
        "sceAmprCommandBufferPushMarker",
        "sceAmprCommandBufferWriteAddressFromCounterPair_04_00",
        "sceAmprCommandBufferPushMarkerWithColor",
        "sceAmprMeasureCommandSizeWriteAddressFromTimeCounter_04_00",
        "sceAmprMeasureCommandSizeMapEnd",
        "sceAmprCommandBufferWriteCounter_04_00",
        "sceAmprCommandBufferWriteAddress_04_00",
        "sceAmprMeasureCommandSizeMapBegin",
        "sceAmprAprCommandBufferReadFileGather",
        "sceAmprCommandBufferPopMarker",
        "sceAmprCommandBufferNopWithData",
        "sceAmprMeasureCommandSizePopMarker",
        "sceAmprMeasureCommandSizeReadFileGather",
        "sceAmprMeasureCommandSizeMapDirectBegin",
        "sceAmprMeasureCommandSizeResetGatherScatterState",
        "sceAmprCommandBufferSetMarkerWithColor",
        "sceAmprCommandBufferNop",
        "sceAmprMeasureCommandSizeSetMarkerWithColor",
        "sceAmprCommandBufferWriteAddressFromCounter_04_00",
        "sceAmprMeasureCommandSizeWaitOnAddress_04_00",
        "sceAmprMeasureCommandSizePushMarker",
        "sceAmprMeasureCommandSizeWriteAddressFromCounterPair_04_00",
        "sceAmprMeasureCommandSizePushMarkerWithColor",
        "sceAmprCommandBufferConstructMarker",
        "sceAmprMeasureCommandSizeWriteAddress_04_00",
        "sceAmprCommandBufferSetMarker",
        "sceAmprMeasureCommandSizeReadFileScatter",
    ];

    /// Registration audit: every one of GTA V's 46 measured-missing
    /// `libSceAmpr` NIDs must resolve.
    #[test]
    fn gta5_batch_all_46_missing_nids_resolve() {
        let registry = HleRegistry::new();
        for name in GTA5_MISSING_AMPR_NIDS {
            assert!(
                registry.is_implemented("libSceAmpr", name),
                "GTA V measured-missing import must resolve: libSceAmpr::{name}"
            );
        }
    }

    /// The knowingly degraded entries (waits dropped, counters unmodeled,
    /// map window unmapped) must be tagged incomplete so coverage tooling
    /// keeps naming them instead of reporting them as working.
    #[test]
    fn gta5_batch_degraded_semantics_are_tagged_incomplete() {
        let registry = HleRegistry::new();
        let incomplete: Vec<String> = registry
            .incomplete_registrations()
            .into_iter()
            .filter(|(library, _, _)| library == "libSceAmpr")
            .map(|(_, function, _)| function)
            .collect();
        for name in [
            "sceAmprCommandBufferWaitOnAddress_04_00",
            "sceAmprCommandBufferWaitOnCounter_04_00",
            "sceAmprCommandBufferWriteCounter_04_00",
            "sceAmprCommandBufferWriteAddressFromCounter_04_00",
            "sceAmprCommandBufferWriteAddressFromCounterPair_04_00",
            "sceAmprCommandBufferWriteAddressFromTimeCounter_04_00",
            "sceAmprAprCommandBufferMapBegin",
            "sceAmprAprCommandBufferMapDirectBegin",
            "sceAmprAprCommandBufferMapEnd",
        ] {
            assert!(
                incomplete.iter().any(|f| f == name),
                "degraded-semantics entry must be register_incomplete: {name}"
            );
        }
    }

    /// The buffer-sizing invariant: every `MeasureCommandSize*` must return
    /// exactly the byte count its paired writer advances the cursor by —
    /// that is the only size a title reserving space by measure calls can
    /// rely on.
    #[test]
    fn measure_sizes_match_writer_advances() {
        let registry = HleRegistry::new();
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x2000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x40;
        assert_eq!(hle_ctor(&ctx, &[cb, 0x200, 0x1000]), cb);

        // A guest marker string for the marker family.
        let msg: u64 = 0x100;
        assert!(ctx.mem.write(msg, b"gta5\0"));

        // MapEnd needs an open window: run MapBegin first (16 KiB-aligned
        // va/size; nothing is mapped).
        let va: u64 = 0x4000;
        let sz: u64 = 0x4000;

        // (measure name, measure args, writer name, writer args)
        let cases: &[(&str, Vec<u64>, &str, Vec<u64>)] = &[
            (
                "sceAmprMeasureCommandSizeNop",
                vec![3],
                "sceAmprCommandBufferNop",
                vec![cb, 3],
            ),
            (
                "sceAmprMeasureCommandSizeNopWithData",
                vec![2, msg],
                "sceAmprCommandBufferNopWithData",
                vec![cb, 2, msg],
            ),
            (
                "sceAmprMeasureCommandSizeSetMarker",
                vec![msg],
                "sceAmprCommandBufferSetMarker",
                vec![cb, msg],
            ),
            (
                "sceAmprMeasureCommandSizeSetMarkerWithColor",
                vec![msg, 0xFF00FF],
                "sceAmprCommandBufferSetMarkerWithColor",
                vec![cb, msg, 0xFF00FF],
            ),
            (
                "sceAmprMeasureCommandSizePushMarker",
                vec![msg],
                "sceAmprCommandBufferPushMarker",
                vec![cb, msg],
            ),
            (
                "sceAmprMeasureCommandSizePushMarkerWithColor",
                vec![msg, 1],
                "sceAmprCommandBufferPushMarkerWithColor",
                vec![cb, msg, 1],
            ),
            (
                "sceAmprMeasureCommandSizePopMarker",
                vec![],
                "sceAmprCommandBufferPopMarker",
                vec![cb],
            ),
            (
                "sceAmprMeasureCommandSizeWriteAddress_04_00",
                vec![0x300, 7],
                "sceAmprCommandBufferWriteAddress_04_00",
                vec![cb, 0x300, 7, 0],
            ),
            (
                "sceAmprMeasureCommandSizeWriteAddressFromCounter_04_00",
                vec![0x300, 0],
                "sceAmprCommandBufferWriteAddressFromCounter_04_00",
                vec![cb, 0x300, 0, 0],
            ),
            (
                "sceAmprMeasureCommandSizeWriteAddressFromCounterPair_04_00",
                vec![0x300, 0],
                "sceAmprCommandBufferWriteAddressFromCounterPair_04_00",
                vec![cb, 0x300, 0, 0],
            ),
            (
                "sceAmprMeasureCommandSizeWriteAddressFromTimeCounter_04_00",
                vec![0x300],
                "sceAmprCommandBufferWriteAddressFromTimeCounter_04_00",
                vec![cb, 0x300, 0],
            ),
            (
                "sceAmprMeasureCommandSizeWaitOnAddress_04_00",
                vec![0x300, 1, 0, 0],
                "sceAmprCommandBufferWaitOnAddress_04_00",
                vec![cb, 0x300, 1, 0, 0],
            ),
            (
                "sceAmprMeasureCommandSizeWaitOnCounter_04_00",
                vec![],
                "sceAmprCommandBufferWaitOnCounter_04_00",
                vec![cb, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                "sceAmprMeasureCommandSizeWriteCounter_04_00",
                vec![],
                "sceAmprCommandBufferWriteCounter_04_00",
                vec![cb, 0, 0, 0, 0, 0],
            ),
            (
                "sceAmprMeasureCommandSizeResetGatherScatterState",
                vec![],
                "sceAmprAprCommandBufferResetGatherScatterState",
                vec![cb, 0, 0],
            ),
            (
                "sceAmprMeasureCommandSizeMapBegin",
                vec![va, sz, 0, 0],
                "sceAmprAprCommandBufferMapBegin",
                vec![cb, va, sz, 0, 0],
            ),
            (
                "sceAmprMeasureCommandSizeMapDirectBegin",
                vec![va, 0, sz, 0, 0],
                "sceAmprAprCommandBufferMapDirectBegin",
                vec![cb, va, 0, sz, 0, 0],
            ),
            (
                "sceAmprMeasureCommandSizeMapEnd",
                vec![],
                "sceAmprAprCommandBufferMapEnd",
                vec![cb],
            ),
        ];
        for (measure, margs, writer, wargs) in cases {
            let measured = registry
                .call(&ctx, "libSceAmpr", measure, margs)
                .unwrap_or_else(|| panic!("{measure} must resolve"));
            assert!(
                measured < 0x8000_0000,
                "{measure} returned an error for valid args: {measured:#x}"
            );
            let before = hle_get_current_offset(&ctx, &[cb]);
            let ret = registry
                .call(&ctx, "libSceAmpr", writer, wargs)
                .unwrap_or_else(|| panic!("{writer} must resolve"));
            assert_eq!(ret, OK, "{writer} must append for valid args");
            let after = hle_get_current_offset(&ctx, &[cb]);
            assert_eq!(
                after - before,
                measured,
                "{measure} ({measured:#x}) must equal {writer}'s cursor advance"
            );
        }
    }

    /// A submitted buffer full of the new skip records must complete
    /// cleanly through the libkernel completion walker — and the one record
    /// with a real completion effect (WriteAddress_04_00) must perform it.
    #[test]
    fn submitted_batch_records_complete_and_write_address_fires() {
        let registry = HleRegistry::new();
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x2000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x40;
        let target: u64 = 0x300;
        assert_eq!(hle_ctor(&ctx, &[cb, 0x400, 0x800]), cb);
        assert!(ctx.mem.write(0x100, b"marker\0"));
        assert!(ctx.mem.write(target, &u64::MAX.to_le_bytes()));

        assert_eq!(hle_nop(&ctx, &[cb, 4]), OK);
        assert_eq!(hle_marker(&ctx, &[cb, 0x100]), OK);
        assert_eq!(hle_wait_or_counter_nop(&ctx, &[cb]), OK);
        assert_eq!(hle_write_address_04_00(&ctx, &[cb, target, 0x1234, 0]), OK);
        assert_eq!(hle_pop_marker(&ctx, &[cb]), OK);

        // Nothing fires before submit.
        assert_eq!(read_u64(&ctx, target), u64::MAX);
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelAprSubmitCommandBuffer",
                &[cb, 0]
            ),
            Some(0),
            "a buffer of skip + write-address records must complete cleanly"
        );
        assert_eq!(
            read_u64(&ctx, target),
            0x1234,
            "WriteAddress_04_00 must perform its completion write"
        );
    }

    /// `WriteAddressFrom{Counter,TimeCounter}` complete by writing 0 (the
    /// counter source is unmodeled — KytyPS5 parity), never by leaving the
    /// destination untouched or faking a plausible value.
    #[test]
    fn write_address_from_counter_writes_zero_at_completion() {
        let registry = HleRegistry::new();
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x40;
        let target: u64 = 0x300;
        assert_eq!(hle_ctor(&ctx, &[cb, 0x200, 0x100]), cb);
        assert!(ctx.mem.write(target, &0xDEAD_BEEFu64.to_le_bytes()));
        assert_eq!(hle_write_address_from_source(&ctx, &[cb, target, 0]), OK);
        assert_eq!(
            registry.call(
                &ctx,
                "libkernel",
                "sceKernelAprSubmitCommandBuffer",
                &[cb, 0]
            ),
            Some(0)
        );
        assert_eq!(read_u64(&ctx, target), 0);
    }

    /// Gather/scatter reads continue the stream state a prior `ReadFile`
    /// seeded: Scatter keeps reading the file sequentially into a new
    /// destination, Gather reads a new file offset into the continued
    /// destination, and Reset invalidates the state (further gather/scatter
    /// is EINVAL until a new ReadFile).
    #[test]
    fn gather_scatter_reads_continue_the_stream() {
        let kernel = raeen_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let data: Vec<u8> = (0..128u32).map(|i| i as u8).collect();
        let host = temp_host_file("gs", &data);
        let id = kernel.appr_register_file("/app0/assets/gs.bin", host.display().to_string());

        let cb: u64 = 0x40;
        let dst1: u64 = 0x1000;
        let dst2: u64 = 0x2000;
        assert_eq!(hle_ctor(&ctx, &[cb, 0x400, 0x400]), cb);

        // Gather/scatter before any ReadFile: no stream state yet.
        assert_eq!(
            hle_apr_read_file_gather(&ctx, &[cb, 0, 0, 8, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );

        // Seed: bytes 0..16 -> dst1. Stream continues at (dst1+16, offset 16).
        assert_eq!(
            hle_apr_read_file(&ctx, &[cb, 0, 0, u64::from(id), dst1, 16, 0]),
            OK
        );
        assert_eq!(
            hle_get_type_flags(&ctx, &[cb]) as u32 & APR_TYPE_GATHER_SCATTER_VALID,
            APR_TYPE_GATHER_SCATTER_VALID
        );

        // Scatter: next 16 file bytes (16..32) into dst2.
        assert_eq!(hle_apr_read_file_scatter(&ctx, &[cb, 0, 0, dst2, 16]), OK);
        let mut got = [0u8; 16];
        assert!(ctx.mem.read(dst2, &mut got));
        assert_eq!(got, data[16..32], "scatter continues the file sequentially");

        // Gather: file bytes 64..72 into the continued destination (dst2+16).
        assert_eq!(hle_apr_read_file_gather(&ctx, &[cb, 0, 0, 8, 64]), OK);
        let mut got8 = [0u8; 8];
        assert!(ctx.mem.read(dst2 + 16, &mut got8));
        assert_eq!(
            got8,
            data[64..72],
            "gather lands at the continued destination"
        );

        // GatherScatter: explicit destination and offset, file id continued.
        assert_eq!(
            hle_apr_read_file_gather_scatter(&ctx, &[cb, 0, 0, dst1 + 0x100, 8, 96]),
            OK
        );
        assert!(ctx.mem.read(dst1 + 0x100, &mut got8));
        assert_eq!(got8, data[96..104]);

        // Reset invalidates the stream and lowers the type flag.
        assert_eq!(hle_apr_reset_gather_scatter_state(&ctx, &[cb, 0, 0]), OK);
        assert_eq!(
            hle_get_type_flags(&ctx, &[cb]) as u32 & APR_TYPE_GATHER_SCATTER_VALID,
            0
        );
        assert_eq!(
            hle_apr_read_file_scatter(&ctx, &[cb, 0, 0, dst2, 8]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        let _ = std::fs::remove_file(&host);
    }

    /// The APR map window is a bracketed state machine: MapEnd without an
    /// open window is EPERM, unaligned windows are EINVAL (writer and
    /// measure agree), and Begin/End raise/lower the `MAP_ACTIVE` type flag.
    #[test]
    fn map_window_requires_begin_before_end() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x40;
        assert_eq!(hle_ctor(&ctx, &[cb, 0x200, 0x200]), cb);

        assert_eq!(hle_apr_map_end(&ctx, &[cb]), SCE_ERROR_PERMISSION_DENIED);
        // Unaligned va/size: EINVAL from writer AND measure.
        assert_eq!(
            hle_apr_map_begin(&ctx, &[cb, 0x123, 0x4000, 0, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_measure_map_begin(&ctx, &[0x123, 0x4000, 0, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            hle_measure_map_direct_begin(&ctx, &[0x4000, 0x123, 0x4000, 0, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );

        assert_eq!(hle_apr_map_begin(&ctx, &[cb, 0x4000, 0x8000, 0, 0]), OK);
        assert_eq!(
            hle_get_type_flags(&ctx, &[cb]) as u32 & APR_TYPE_MAP_ACTIVE,
            APR_TYPE_MAP_ACTIVE
        );
        assert_eq!(hle_apr_map_end(&ctx, &[cb]), OK);
        assert_eq!(
            hle_get_type_flags(&ctx, &[cb]) as u32 & APR_TYPE_MAP_ACTIVE,
            0
        );
        // The window is closed again.
        assert_eq!(hle_apr_map_end(&ctx, &[cb]), SCE_ERROR_PERMISSION_DENIED);
    }

    /// `GetBufferBaseAddress` reads the bound backing buffer; `GetType` on a
    /// fresh or null buffer reads 0; construct clears stale flags.
    #[test]
    fn get_type_and_buffer_base_address_basics() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x40;
        assert_eq!(hle_get_buffer_base_address(&ctx, &[0]), 0);
        assert_eq!(hle_get_type_flags(&ctx, &[0]), 0);
        assert_eq!(hle_ctor(&ctx, &[cb, 0x200, 0x100]), cb);
        assert_eq!(hle_get_buffer_base_address(&ctx, &[cb]), 0x200);
        assert_eq!(hle_get_type_flags(&ctx, &[cb]), 0);
        // Raise a flag, then re-construct: flags must not survive.
        assert_eq!(hle_apr_map_begin(&ctx, &[cb, 0x4000, 0x4000, 0, 0]), OK);
        assert_ne!(hle_get_type_flags(&ctx, &[cb]), 0);
        assert_eq!(hle_ctor(&ctx, &[cb, 0x200, 0x100]), cb);
        assert_eq!(hle_get_type_flags(&ctx, &[cb]), 0);
    }

    /// The nop family's writers reject the KytyPS5 argument bounds
    /// (`Nop`: 1..=16 dwords, `NopWithData`: ≤ 15 dwords).
    #[test]
    fn nop_writers_enforce_dword_bounds() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cb: u64 = 0x40;
        assert_eq!(hle_ctor(&ctx, &[cb, 0x200, 0x200]), cb);
        assert_eq!(hle_nop(&ctx, &[cb, 0]), SCE_ERROR_INVALID_ARGUMENT);
        assert_eq!(hle_nop(&ctx, &[cb, 17]), SCE_ERROR_INVALID_ARGUMENT);
        assert_eq!(
            hle_nop_with_data(&ctx, &[cb, 16, 0]),
            SCE_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(hle_nop(&ctx, &[0, 1]), SCE_ERROR_INVALID_ARGUMENT);
        // In-bounds appends land and count as commands.
        assert_eq!(hle_nop(&ctx, &[cb, 16]), OK);
        assert_eq!(hle_nop_with_data(&ctx, &[cb, 15, 0]), OK);
        assert_eq!(hle_get_num_commands(&ctx, &[cb]), 2);
    }
}
