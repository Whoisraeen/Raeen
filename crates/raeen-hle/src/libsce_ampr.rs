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
}

/// Write the command-buffer struct fields (self/data/size/aux) and set the
/// write cursor to `write_offset`.
fn write_cb(ctx: &HleContext, cb: u64, buffer: u64, size: u64, write_offset: u64) -> bool {
    let ok = ctx.mem.write(cb + CB_SELF_OFFSET, &cb.to_le_bytes())
        && ctx.mem.write(cb + CB_DATA_OFFSET, &buffer.to_le_bytes())
        && ctx.mem.write(cb + CB_SIZE_OFFSET, &size.to_le_bytes())
        && ctx.mem.write(cb + CB_AUX0_OFFSET, &0u64.to_le_bytes())
        && ctx.mem.write(cb + CB_AUX1_OFFSET, &0u64.to_le_bytes());
    if ok {
        ctx.kernel.ampr_write_offsets.insert(cb, write_offset);
        // Construct/reset both rewind through here: the appended-record count
        // starts over with the cursor (SharpEmu zeroes `CommandCount` in its
        // constructor and reset paths).
        ctx.kernel.ampr_command_counts.insert(cb, 0);
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
    debug!("sceAmprCommandBufferConstructor(cb={cb:#x}, buffer={buffer:#x}, size={size:#x})");
    cb
}

/// `sceAmprCommandBufferDestructor(cb)`: drop the tracked write cursor.
fn hle_dtor(ctx: &HleContext, args: &[u64]) -> u64 {
    let cb = args.first().copied().unwrap_or(0);
    if cb != 0 {
        ctx.kernel.ampr_write_offsets.remove(&cb);
        ctx.kernel.ampr_command_counts.remove(&cb);
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
    let mut record = [0u8; 0x20];
    record[0x00..0x04].copy_from_slice(&3u32.to_le_bytes());
    record[0x08..0x10].copy_from_slice(&address.to_le_bytes());
    record[0x10..0x18].copy_from_slice(&value.to_le_bytes());
    if !append_record(ctx, cb, &record) {
        return SCE_ERROR_MEMORY_FAULT;
    }
    OK
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
}
