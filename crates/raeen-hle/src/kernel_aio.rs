//! HLE libkernel **asynchronous file I/O** (`sceKernelAio*`).
//!
//! Backed by the real host-threadpool engine in [`raeen_kernel::aio`]:
//! submit returns immediately with a request id, worker threads perform the
//! positional reads/writes through the same VFS descriptor table the
//! synchronous `read`/`pread`/`pwrite` path uses, wait blocks (with timeout),
//! poll never blocks, cancel aborts not-yet-started requests, delete retires
//! the id.
//!
//! **Guest-memory discipline:** workers never touch guest memory. Write
//! payloads are captured from the guest buffer at submit time (on the guest
//! thread, through `ctx.mem` — the same access layer the sync path uses) and
//! read completions are copied back into the guest buffer when this module
//! *drains* completions on a guest thread (any wait/poll/cancel/delete call
//! for that id). A guest that checks request state through this API — the
//! documented contract — observes buffers and `SceKernelAioResult` structs
//! filled in exactly when the API first reports the terminal state.
//!
//! ## Guest ABI (re-derived from the public C declarations, cross-checked
//! against shadPS4 `src/core/libraries/kernel/aio.h` — layout re-derived,
//! no code ported)
//!
//! ```text
//! SceKernelAioRWRequest (0x28 bytes):
//!   0x00  s64   offset
//!   0x08  s64   nbyte
//!   0x10  void* buf
//!   0x18  SceKernelAioResult* result
//!   0x20  s32   fd            (+4 pad)
//! SceKernelAioResult (0x10 bytes):
//!   0x00  s64   returnValue   (bytes transferred, or negative SCE error)
//!   0x08  u32   state         (+4 pad)
//! SceKernelAioSubmitId: s32 (>= 1; 0 is the "no request" sentinel)
//! ```
//!
//! Measured demand (phase1 NID coverage): Until Dawn and Dragon Ball
//! Sparking Zero import `SubmitReadCommands`, `SubmitWriteCommands`,
//! `WaitRequest`, `PollRequests`, `DeleteRequest`. The sibling spellings
//! (`*Multiple`, plural/singular forms, cancel) share the same machinery and
//! are registered alongside.

use crate::{GuestAccess, GuestAddress, GuestRange, HleContext, HleRegistry, MAX_HLE_BULK_BYTES};
use raeen_kernel::aio::{
    AIO_STATE_PROCESSING, AIO_STATE_SUBMITTED, AioCompletion, AioOp, AioRequest, AioWaitError,
};
use tracing::{debug, warn};

const OK: u64 = 0;
const SCE_KERNEL_ERROR_ESRCH: u64 = 0x8002_0003;
const SCE_KERNEL_ERROR_EFAULT: u64 = 0x8002_000E;
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;
const SCE_KERNEL_ERROR_ETIMEDOUT: u64 = 0x8002_003C;

/// Guest `SceKernelAioRWRequest` stride.
const RW_REQUEST_SIZE: u64 = 0x28;
/// Requests per submit call cap (`SCE_KERNEL_AIO_MAX_REQUESTS`-class bound;
/// also bounds the id/state array walks of the plural entry points).
const MAX_AIO_BATCH: u64 = 128;
/// Slice for interruptible blocking waits: re-check process termination at
/// this cadence so an infinite AIO wait cannot outlive its guest process.
const WAIT_SLICE: std::time::Duration = std::time::Duration::from_millis(50);

/// Register the kernel AIO HLE functions. `sceKernelAioInitialize{Param,Impl}`
/// stay in `libkernel.rs` (they predate this module and are pure init).
pub fn register(registry: &HleRegistry) {
    registry.register(
        "libkernel",
        "sceKernelAioSubmitReadCommands",
        hle_submit_read_commands,
    );
    registry.register(
        "libkernel",
        "sceKernelAioSubmitReadCommandsMultiple",
        hle_submit_read_commands_multiple,
    );
    registry.register(
        "libkernel",
        "sceKernelAioSubmitWriteCommands",
        hle_submit_write_commands,
    );
    registry.register(
        "libkernel",
        "sceKernelAioSubmitWriteCommandsMultiple",
        hle_submit_write_commands_multiple,
    );
    registry.register("libkernel", "sceKernelAioWaitRequest", hle_wait_request);
    registry.register("libkernel", "sceKernelAioWaitRequests", hle_wait_requests);
    registry.register("libkernel", "sceKernelAioPollRequest", hle_poll_request);
    registry.register("libkernel", "sceKernelAioPollRequests", hle_poll_requests);
    registry.register("libkernel", "sceKernelAioCancelRequest", hle_cancel_request);
    registry.register(
        "libkernel",
        "sceKernelAioCancelRequests",
        hle_cancel_requests,
    );
    registry.register("libkernel", "sceKernelAioDeleteRequest", hle_delete_request);
    registry.register(
        "libkernel",
        "sceKernelAioDeleteRequests",
        hle_delete_requests,
    );
}

/// One decoded guest `SceKernelAioRWRequest`.
struct RwRequest {
    offset: i64,
    nbyte: i64,
    buf: u64,
    result: u64,
    fd: i32,
}

fn read_rw_request(ctx: &HleContext, addr: u64) -> Option<RwRequest> {
    let mut bytes = [0u8; RW_REQUEST_SIZE as usize];
    if !ctx.mem.read(addr, &mut bytes) {
        return None;
    }
    Some(RwRequest {
        offset: i64::from_le_bytes(bytes[0x00..0x08].try_into().unwrap()),
        nbyte: i64::from_le_bytes(bytes[0x08..0x10].try_into().unwrap()),
        buf: u64::from_le_bytes(bytes[0x10..0x18].try_into().unwrap()),
        result: u64::from_le_bytes(bytes[0x18..0x20].try_into().unwrap()),
        fd: i32::from_le_bytes(bytes[0x20..0x24].try_into().unwrap()),
    })
}

/// Write a guest `SceKernelAioResult`.
fn write_result(ctx: &HleContext, addr: u64, return_value: i64, state: u32) -> bool {
    let mut bytes = [0u8; 0x10];
    bytes[0x00..0x08].copy_from_slice(&return_value.to_le_bytes());
    bytes[0x08..0x0C].copy_from_slice(&state.to_le_bytes());
    ctx.mem.write(addr, &bytes)
}

/// Deliver drained completions to the guest: staged read bytes into the
/// request's buffer, then the result struct. Both go through `ctx.mem` —
/// the same guest-memory layer the synchronous read path writes through.
/// Ranges were validated at submit time; a failure here (guest unmapped the
/// buffer mid-flight) is logged and skipped, never a host fault.
fn deliver_completions(ctx: &HleContext, completions: &[AioCompletion]) {
    for completion in completions {
        if let Some(data) = completion.data.as_deref()
            && !data.is_empty()
            && !ctx.mem.write(completion.guest_buf, data)
        {
            warn!(
                buf = format_args!("{:#x}", completion.guest_buf),
                len = data.len(),
                "AIO completion: guest buffer no longer writable — data dropped"
            );
        }
        if completion.guest_result != 0
            && !write_result(
                ctx,
                completion.guest_result,
                completion.return_value,
                completion.state,
            )
        {
            warn!(
                result = format_args!("{:#x}", completion.guest_result),
                "AIO completion: result struct no longer writable"
            );
        }
    }
}

/// Drain and deliver everything terminal for `id`.
fn drain_and_deliver(ctx: &HleContext, id: i32) {
    let completions = ctx.kernel.aio.drain_completions(id);
    deliver_completions(ctx, &completions);
}

/// Decode + validate one guest request into an engine [`AioRequest`],
/// writing the initial `SUBMITTED` result state. `Err` carries the SCE
/// error to return from the submit call.
fn build_request(ctx: &HleContext, addr: u64, is_read: bool) -> Result<AioRequest, u64> {
    let Some(request) = read_rw_request(ctx, addr) else {
        return Err(SCE_KERNEL_ERROR_EFAULT);
    };
    if request.offset < 0 || request.nbyte < 0 || request.nbyte as u64 > MAX_HLE_BULK_BYTES {
        return Err(SCE_KERNEL_ERROR_EINVAL);
    }
    let nbyte = request.nbyte as u64;
    if nbyte > 0 {
        let Some(range) = GuestRange::new(GuestAddress::new(request.buf), nbyte) else {
            return Err(SCE_KERNEL_ERROR_EFAULT);
        };
        let access = if is_read {
            GuestAccess::Write
        } else {
            GuestAccess::Read
        };
        if !ctx.mem.validate_range(range, access) {
            return Err(SCE_KERNEL_ERROR_EFAULT);
        }
    }
    let op = if is_read {
        AioOp::Read {
            fd: request.fd,
            offset: request.offset as u64,
            nbyte,
        }
    } else {
        // Capture the write payload NOW, on the guest thread, through the
        // same guest-memory layer the sync write path reads through — the
        // worker never touches guest memory.
        let mut data = vec![0u8; nbyte as usize];
        if !ctx.mem.read(request.buf, &mut data) {
            return Err(SCE_KERNEL_ERROR_EFAULT);
        }
        AioOp::Write {
            fd: request.fd,
            offset: request.offset as u64,
            data,
        }
    };
    // The request is accepted: its result struct starts life SUBMITTED.
    if request.result != 0 && !write_result(ctx, request.result, 0, AIO_STATE_SUBMITTED) {
        return Err(SCE_KERNEL_ERROR_EFAULT);
    }
    Ok(AioRequest {
        op,
        guest_buf: request.buf,
        guest_result: request.result,
    })
}

/// Shared body of `sceKernelAioSubmit{Read,Write}Commands(req[], size, prio,
/// id*)`: one submit id covers the whole batch.
fn submit_commands(ctx: &HleContext, args: &[u64], is_read: bool) -> u64 {
    let req = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0) as i32;
    let _prio = args.get(2).copied().unwrap_or(0);
    let id_out = args.get(3).copied().unwrap_or(0);
    if req == 0 || id_out == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if size <= 0 || size as u64 > MAX_AIO_BATCH {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let mut requests = Vec::with_capacity(size as usize);
    for index in 0..size as u64 {
        match build_request(ctx, req + index * RW_REQUEST_SIZE, is_read) {
            Ok(request) => requests.push(request),
            Err(error) => return error,
        }
    }
    let id = ctx.kernel.aio.submit(requests);
    debug!(
        id,
        size,
        kind = if is_read { "read" } else { "write" },
        "sceKernelAioSubmitCommands"
    );
    if !ctx.mem.write(id_out, &id.to_le_bytes()) {
        // The batch is already queued; retire it so nothing leaks.
        let _ = ctx.kernel.aio.delete(id);
        return SCE_KERNEL_ERROR_EFAULT;
    }
    OK
}

/// Shared body of the `*Multiple` forms: each request gets its OWN submit id,
/// written to the id array.
fn submit_commands_multiple(ctx: &HleContext, args: &[u64], is_read: bool) -> u64 {
    let req = args.first().copied().unwrap_or(0);
    let size = args.get(1).copied().unwrap_or(0) as i32;
    let _prio = args.get(2).copied().unwrap_or(0);
    let ids_out = args.get(3).copied().unwrap_or(0);
    if req == 0 || ids_out == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if size <= 0 || size as u64 > MAX_AIO_BATCH {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    for index in 0..size as u64 {
        let request = match build_request(ctx, req + index * RW_REQUEST_SIZE, is_read) {
            Ok(request) => request,
            Err(error) => return error,
        };
        let id = ctx.kernel.aio.submit(vec![request]);
        if !ctx.mem.write(ids_out + index * 4, &id.to_le_bytes()) {
            let _ = ctx.kernel.aio.delete(id);
            return SCE_KERNEL_ERROR_EFAULT;
        }
    }
    OK
}

fn hle_submit_read_commands(ctx: &HleContext, args: &[u64]) -> u64 {
    submit_commands(ctx, args, true)
}

fn hle_submit_read_commands_multiple(ctx: &HleContext, args: &[u64]) -> u64 {
    submit_commands_multiple(ctx, args, true)
}

fn hle_submit_write_commands(ctx: &HleContext, args: &[u64]) -> u64 {
    submit_commands(ctx, args, false)
}

fn hle_submit_write_commands_multiple(ctx: &HleContext, args: &[u64]) -> u64 {
    submit_commands_multiple(ctx, args, false)
}

/// Block until `id` is terminal, slicing the engine wait so a terminating
/// guest process is never held hostage by an in-flight AIO wait.
/// `usec = None` (or 0) waits indefinitely. `Ok(state)` on terminal;
/// `Err(ETIMEDOUT-or-ESRCH)` otherwise.
fn wait_terminal(ctx: &HleContext, id: i32, usec: Option<u64>) -> Result<u32, u64> {
    let deadline = usec
        .filter(|&usec| usec > 0)
        .map(|usec| std::time::Instant::now() + std::time::Duration::from_micros(usec));
    loop {
        let slice = match deadline {
            Some(deadline) => {
                let now = std::time::Instant::now();
                if now >= deadline {
                    return Err(SCE_KERNEL_ERROR_ETIMEDOUT);
                }
                (deadline - now).min(WAIT_SLICE)
            }
            None => WAIT_SLICE,
        };
        match ctx.kernel.aio.wait(id, Some(slice)) {
            Ok(state) => return Ok(state),
            Err(AioWaitError::Unknown) => return Err(SCE_KERNEL_ERROR_ESRCH),
            Err(AioWaitError::TimedOut(_)) => {
                if ctx.guest_threads.process_is_terminating() {
                    return Err(SCE_KERNEL_ERROR_ETIMEDOUT);
                }
            }
        }
    }
}

/// Write one s32 `state` (or delete-`ret`) array element.
fn write_s32(ctx: &HleContext, addr: u64, value: i32) -> bool {
    ctx.mem.write(addr, &value.to_le_bytes())
}

/// `sceKernelAioWaitRequest(id, state*, usec*)`: block until the request is
/// terminal (or `*usec` microseconds elapse; NULL/0 = forever), deliver its
/// completions, and report the state. Timeout → `ETIMEDOUT` with the current
/// state still written.
fn hle_wait_request(ctx: &HleContext, args: &[u64]) -> u64 {
    let id = args.first().copied().unwrap_or(0) as i32;
    let state_out = args.get(1).copied().unwrap_or(0);
    let usec_ptr = args.get(2).copied().unwrap_or(0);
    if state_out == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let usec = if usec_ptr == 0 {
        None
    } else {
        let mut bytes = [0u8; 4];
        if !ctx.mem.read(usec_ptr, &mut bytes) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        Some(u64::from(u32::from_le_bytes(bytes)))
    };
    match wait_terminal(ctx, id, usec) {
        Ok(state) => {
            drain_and_deliver(ctx, id);
            if !write_s32(ctx, state_out, state as i32) {
                return SCE_KERNEL_ERROR_EFAULT;
            }
            OK
        }
        Err(SCE_KERNEL_ERROR_ETIMEDOUT) => {
            drain_and_deliver(ctx, id);
            let state = ctx.kernel.aio.poll(id).unwrap_or(AIO_STATE_PROCESSING);
            let _ = write_s32(ctx, state_out, state as i32);
            SCE_KERNEL_ERROR_ETIMEDOUT
        }
        Err(error) => error,
    }
}

/// `sceKernelAioWaitRequests(id[], num, state[], mode, usec*)`: wait on a set
/// of ids under one deadline. `mode` bit 0x02 = "wait any": return as soon as
/// one request is terminal; otherwise wait for all.
fn hle_wait_requests(ctx: &HleContext, args: &[u64]) -> u64 {
    const WAIT_ANY: u64 = 0x02;
    let ids_ptr = args.first().copied().unwrap_or(0);
    let num = args.get(1).copied().unwrap_or(0) as i32;
    let states_ptr = args.get(2).copied().unwrap_or(0);
    let mode = args.get(3).copied().unwrap_or(0);
    let usec_ptr = args.get(4).copied().unwrap_or(0);
    if ids_ptr == 0 || states_ptr == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if num <= 0 || num as u64 > MAX_AIO_BATCH {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let usec = if usec_ptr == 0 {
        None
    } else {
        let mut bytes = [0u8; 4];
        if !ctx.mem.read(usec_ptr, &mut bytes) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        Some(u64::from(u32::from_le_bytes(bytes)))
    };
    let mut ids = Vec::with_capacity(num as usize);
    for index in 0..num as u64 {
        let mut bytes = [0u8; 4];
        if !ctx.mem.read(ids_ptr + index * 4, &mut bytes) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        ids.push(i32::from_le_bytes(bytes));
    }

    let mut timed_out = false;
    let wait_any = mode & WAIT_ANY != 0;
    let mut any_terminal = false;
    for &id in &ids {
        if timed_out || (wait_any && any_terminal) {
            break; // remaining ids only get their current state reported
        }
        match wait_terminal(ctx, id, usec) {
            Ok(_) => any_terminal = true,
            Err(SCE_KERNEL_ERROR_ETIMEDOUT) => timed_out = true,
            Err(error) => return error,
        }
    }
    for (index, &id) in ids.iter().enumerate() {
        drain_and_deliver(ctx, id);
        let state = ctx.kernel.aio.poll(id).unwrap_or(AIO_STATE_PROCESSING);
        if !write_s32(ctx, states_ptr + index as u64 * 4, state as i32) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
    }
    if timed_out {
        SCE_KERNEL_ERROR_ETIMEDOUT
    } else {
        OK
    }
}

/// `sceKernelAioPollRequest(id, state*)`: report the current state without
/// ever blocking, delivering any terminal completions first.
fn hle_poll_request(ctx: &HleContext, args: &[u64]) -> u64 {
    let id = args.first().copied().unwrap_or(0) as i32;
    let state_out = args.get(1).copied().unwrap_or(0);
    if state_out == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    let Some(state) = ctx.kernel.aio.poll(id) else {
        return SCE_KERNEL_ERROR_ESRCH;
    };
    drain_and_deliver(ctx, id);
    if !write_s32(ctx, state_out, state as i32) {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    OK
}

/// `sceKernelAioPollRequests(id[], num, state[])`: the plural form of
/// [`hle_poll_request`] — one state per id, never blocking.
fn hle_poll_requests(ctx: &HleContext, args: &[u64]) -> u64 {
    let ids_ptr = args.first().copied().unwrap_or(0);
    let num = args.get(1).copied().unwrap_or(0) as i32;
    let states_ptr = args.get(2).copied().unwrap_or(0);
    if ids_ptr == 0 || states_ptr == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if num <= 0 || num as u64 > MAX_AIO_BATCH {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    for index in 0..num as u64 {
        let mut bytes = [0u8; 4];
        if !ctx.mem.read(ids_ptr + index * 4, &mut bytes) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        let id = i32::from_le_bytes(bytes);
        let Some(state) = ctx.kernel.aio.poll(id) else {
            return SCE_KERNEL_ERROR_ESRCH;
        };
        drain_and_deliver(ctx, id);
        if !write_s32(ctx, states_ptr + index * 4, state as i32) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
    }
    OK
}

/// Cancel one id and report the state after the attempt. `id == 0` is the
/// "no request" sentinel: nothing to cancel, reported as `PROCESSING`
/// (matching the console's observed behavior for the null id).
fn cancel_one(ctx: &HleContext, id: i32) -> Result<u32, u64> {
    if id == 0 {
        return Ok(AIO_STATE_PROCESSING);
    }
    let Some(state) = ctx.kernel.aio.cancel(id) else {
        return Err(SCE_KERNEL_ERROR_ESRCH);
    };
    drain_and_deliver(ctx, id);
    Ok(state)
}

/// `sceKernelAioCancelRequest(id, state*)`: abort what has not started;
/// requests already running finish normally. Cancel of a completed request
/// is a no-op that reports `COMPLETED`.
fn hle_cancel_request(ctx: &HleContext, args: &[u64]) -> u64 {
    let id = args.first().copied().unwrap_or(0) as i32;
    let state_out = args.get(1).copied().unwrap_or(0);
    if state_out == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    match cancel_one(ctx, id) {
        Ok(state) => {
            if !write_s32(ctx, state_out, state as i32) {
                return SCE_KERNEL_ERROR_EFAULT;
            }
            OK
        }
        Err(error) => error,
    }
}

/// `sceKernelAioCancelRequests(id[], num, state[])`.
fn hle_cancel_requests(ctx: &HleContext, args: &[u64]) -> u64 {
    let ids_ptr = args.first().copied().unwrap_or(0);
    let num = args.get(1).copied().unwrap_or(0) as i32;
    let states_ptr = args.get(2).copied().unwrap_or(0);
    if ids_ptr == 0 || states_ptr == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if num <= 0 || num as u64 > MAX_AIO_BATCH {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    for index in 0..num as u64 {
        let mut bytes = [0u8; 4];
        if !ctx.mem.read(ids_ptr + index * 4, &mut bytes) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        let state = match cancel_one(ctx, i32::from_le_bytes(bytes)) {
            Ok(state) => state,
            Err(error) => return error,
        };
        if !write_s32(ctx, states_ptr + index * 4, state as i32) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
    }
    OK
}

/// Delete one id: deliver everything terminal, then retire it. `Ok` carries
/// the delete return value for the guest's `ret` slot (always 0 here).
fn delete_one(ctx: &HleContext, id: i32) -> Result<i32, u64> {
    let Some((_state, completions)) = ctx.kernel.aio.delete(id) else {
        return Err(SCE_KERNEL_ERROR_ESRCH);
    };
    deliver_completions(ctx, &completions);
    Ok(0)
}

/// `sceKernelAioDeleteRequest(id, ret*)`: release the request, delivering
/// any not-yet-drained completions first. A deleted id still answers polls
/// with its final state (bounded retire ring in the engine).
fn hle_delete_request(ctx: &HleContext, args: &[u64]) -> u64 {
    let id = args.first().copied().unwrap_or(0) as i32;
    let ret_out = args.get(1).copied().unwrap_or(0);
    if ret_out == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    match delete_one(ctx, id) {
        Ok(ret) => {
            if !write_s32(ctx, ret_out, ret) {
                return SCE_KERNEL_ERROR_EFAULT;
            }
            OK
        }
        Err(error) => error,
    }
}

/// `sceKernelAioDeleteRequests(id[], num, ret[])`.
fn hle_delete_requests(ctx: &HleContext, args: &[u64]) -> u64 {
    let ids_ptr = args.first().copied().unwrap_or(0);
    let num = args.get(1).copied().unwrap_or(0) as i32;
    let rets_ptr = args.get(2).copied().unwrap_or(0);
    if ids_ptr == 0 || rets_ptr == 0 {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    if num <= 0 || num as u64 > MAX_AIO_BATCH {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    for index in 0..num as u64 {
        let mut bytes = [0u8; 4];
        if !ctx.mem.read(ids_ptr + index * 4, &mut bytes) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        let ret = match delete_one(ctx, i32::from_le_bytes(bytes)) {
            Ok(ret) => ret,
            Err(error) => return error,
        };
        if !write_s32(ctx, rets_ptr + index * 4, ret) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};
    use raeen_kernel::aio::{AIO_STATE_ABORTED, AIO_STATE_COMPLETED};
    use raeen_kernel::filesystem::open_flags::{O_CREAT, O_RDONLY, O_RDWR};
    use std::path::PathBuf;

    // Guest layout used by every test.
    const REQ: u64 = 0x100; // SceKernelAioRWRequest array
    const RESULT: u64 = 0x300; // SceKernelAioResult array (0x10 stride)
    const BUF: u64 = 0x500; // data buffers
    const ID_OUT: u64 = 0x800; // submit id / id arrays
    const STATE_OUT: u64 = 0x900; // state / ret arrays
    const USEC: u64 = 0xA00;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("raeen-hle-aio-{tag}-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn kernel_with_file(tag: &str, contents: &[u8]) -> (raeen_kernel::OrbisKernel, i32) {
        let dir = temp_dir(tag);
        std::fs::write(dir.join("asset.bin"), contents).unwrap();
        let kernel = raeen_kernel::OrbisKernel::new();
        kernel.filesystem.set_game_directory(&dir);
        let fd = kernel
            .filesystem
            .open("/app0/asset.bin", O_RDONLY, 0)
            .unwrap();
        (kernel, fd)
    }

    /// Write one guest `SceKernelAioRWRequest` at `addr`.
    fn write_request(
        mem: &crate::TestMemory,
        addr: u64,
        offset: i64,
        nbyte: i64,
        buf: u64,
        result: u64,
        fd: i32,
    ) {
        let mut bytes = [0u8; RW_REQUEST_SIZE as usize];
        bytes[0x00..0x08].copy_from_slice(&offset.to_le_bytes());
        bytes[0x08..0x10].copy_from_slice(&nbyte.to_le_bytes());
        bytes[0x10..0x18].copy_from_slice(&buf.to_le_bytes());
        bytes[0x18..0x20].copy_from_slice(&result.to_le_bytes());
        bytes[0x20..0x24].copy_from_slice(&fd.to_le_bytes());
        assert!(mem.write(addr, &bytes));
    }

    fn read_u32(mem: &crate::TestMemory, addr: u64) -> u32 {
        let mut bytes = [0u8; 4];
        assert!(mem.read(addr, &mut bytes));
        u32::from_le_bytes(bytes)
    }

    fn read_i32(mem: &crate::TestMemory, addr: u64) -> i32 {
        read_u32(mem, addr) as i32
    }

    fn read_i64(mem: &crate::TestMemory, addr: u64) -> i64 {
        let mut bytes = [0u8; 8];
        assert!(mem.read(addr, &mut bytes));
        i64::from_le_bytes(bytes)
    }

    #[test]
    fn submit_read_then_wait_fills_buffer_and_result_struct() {
        let (kernel, fd) = kernel_with_file("read", b"asynchronous!");
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        write_request(&mem, REQ, 2, 4, BUF, RESULT, fd);
        assert_eq!(hle_submit_read_commands(&ctx, &[REQ, 1, 0, ID_OUT]), OK);
        let id = read_i32(&mem, ID_OUT);
        assert!(id >= 1);
        // Workers never touch guest memory: until this thread drains through
        // the API, the result struct still says SUBMITTED.
        assert_eq!(read_u32(&mem, RESULT + 0x08), AIO_STATE_SUBMITTED);

        // Wait (NULL usec = infinite): terminal, buffer + result delivered.
        assert_eq!(hle_wait_request(&ctx, &[id as u64, STATE_OUT, 0]), OK);
        assert_eq!(read_i32(&mem, STATE_OUT), AIO_STATE_COMPLETED as i32);
        let mut data = [0u8; 4];
        assert!(mem.read(BUF, &mut data));
        assert_eq!(&data, b"ynch");
        assert_eq!(read_i64(&mem, RESULT), 4); // returnValue = bytes read
        assert_eq!(read_u32(&mem, RESULT + 0x08), AIO_STATE_COMPLETED);
    }

    #[test]
    fn submit_write_persists_through_the_shared_descriptor_table() {
        let dir = temp_dir("write");
        let kernel = raeen_kernel::OrbisKernel::new();
        kernel.filesystem.set_game_directory(&dir);
        let fd = kernel
            .filesystem
            .open("/app0/save.bin", O_RDWR | O_CREAT, 0o644)
            .unwrap();
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        assert!(mem.write(BUF, b"SAVE"));
        write_request(&mem, REQ, 0, 4, BUF, RESULT, fd);
        assert_eq!(hle_submit_write_commands(&ctx, &[REQ, 1, 0, ID_OUT]), OK);
        let id = read_i32(&mem, ID_OUT);
        assert_eq!(hle_wait_request(&ctx, &[id as u64, STATE_OUT, 0]), OK);
        assert_eq!(read_i64(&mem, RESULT), 4);
        assert_eq!(read_u32(&mem, RESULT + 0x08), AIO_STATE_COMPLETED);
        // Visible through the same descriptor the sync path reads.
        assert_eq!(kernel.filesystem.pread(fd, 4, 0).unwrap(), b"SAVE");
    }

    #[test]
    fn poll_requests_reports_states_without_blocking() {
        let (kernel, fd) = kernel_with_file("poll", b"0123456789");
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Two independent single-request submissions (the Multiple form).
        write_request(&mem, REQ, 0, 2, BUF, RESULT, fd);
        write_request(
            &mem,
            REQ + RW_REQUEST_SIZE,
            4,
            2,
            BUF + 0x10,
            RESULT + 0x10,
            fd,
        );
        assert_eq!(
            hle_submit_read_commands_multiple(&ctx, &[REQ, 2, 0, ID_OUT]),
            OK
        );
        let id0 = read_i32(&mem, ID_OUT);
        let id1 = read_i32(&mem, ID_OUT + 4);
        assert!(id0 >= 1 && id1 >= 1 && id0 != id1);

        // Let both complete, then poll the pair: states written, buffers
        // delivered, and the call itself never blocked on anything pending.
        assert_eq!(
            hle_wait_request(&ctx, &[id0 as u64, STATE_OUT + 0x20, 0]),
            OK
        );
        assert_eq!(
            hle_wait_request(&ctx, &[id1 as u64, STATE_OUT + 0x24, 0]),
            OK
        );
        assert_eq!(hle_poll_requests(&ctx, &[ID_OUT, 2, STATE_OUT]), OK);
        assert_eq!(read_i32(&mem, STATE_OUT), AIO_STATE_COMPLETED as i32);
        assert_eq!(read_i32(&mem, STATE_OUT + 4), AIO_STATE_COMPLETED as i32);
        let mut b = [0u8; 2];
        assert!(mem.read(BUF, &mut b));
        assert_eq!(&b, b"01");
        assert!(mem.read(BUF + 0x10, &mut b));
        assert_eq!(&b, b"45");

        // Unknown id in the array → ESRCH.
        assert!(mem.write(ID_OUT + 4, &0x7AAA_AAAAu32.to_le_bytes()));
        assert_eq!(
            hle_poll_requests(&ctx, &[ID_OUT, 2, STATE_OUT]),
            SCE_KERNEL_ERROR_ESRCH
        );
    }

    #[test]
    fn wait_request_times_out_and_still_reports_a_state() {
        let (kernel, fd) = kernel_with_file("wait-timeout", b"abc");
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        write_request(&mem, REQ, 0, 3, BUF, RESULT, fd);
        assert_eq!(hle_submit_read_commands(&ctx, &[REQ, 1, 0, ID_OUT]), OK);
        let id = read_i32(&mem, ID_OUT);
        // 1 µs timeout: either the request already completed (fast worker)
        // or the call reports ETIMEDOUT with a live state — never a hang.
        assert!(mem.write(USEC, &1u32.to_le_bytes()));
        let result = hle_wait_request(&ctx, &[id as u64, STATE_OUT, USEC]);
        let state = read_i32(&mem, STATE_OUT);
        if result == OK {
            assert_eq!(state, AIO_STATE_COMPLETED as i32);
        } else {
            assert_eq!(result, SCE_KERNEL_ERROR_ETIMEDOUT);
            assert!((AIO_STATE_SUBMITTED as i32..=AIO_STATE_ABORTED as i32).contains(&state));
            // And it still completes on a real wait afterwards.
            assert_eq!(hle_wait_request(&ctx, &[id as u64, STATE_OUT, 0]), OK);
        }
        // Unknown id → ESRCH.
        assert_eq!(
            hle_wait_request(&ctx, &[0x7BBB_BBBB, STATE_OUT, 0]),
            SCE_KERNEL_ERROR_ESRCH
        );
    }

    #[test]
    fn delete_request_writes_ret_retires_id_and_double_delete_is_esrch() {
        let (kernel, fd) = kernel_with_file("delete", b"retire me");
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        write_request(&mem, REQ, 0, 6, BUF, RESULT, fd);
        assert_eq!(hle_submit_read_commands(&ctx, &[REQ, 1, 0, ID_OUT]), OK);
        let id = read_i32(&mem, ID_OUT);
        assert_eq!(hle_wait_request(&ctx, &[id as u64, STATE_OUT, 0]), OK);

        assert!(mem.write(STATE_OUT, &0x5555_5555u32.to_le_bytes()));
        assert_eq!(hle_delete_request(&ctx, &[id as u64, STATE_OUT]), OK);
        assert_eq!(read_i32(&mem, STATE_OUT), 0); // *ret = 0

        // Retired: a late poll still answers with the final state...
        assert_eq!(hle_poll_request(&ctx, &[id as u64, STATE_OUT]), OK);
        assert_eq!(read_i32(&mem, STATE_OUT), AIO_STATE_COMPLETED as i32);
        // ...but a second delete is ESRCH.
        assert_eq!(
            hle_delete_request(&ctx, &[id as u64, STATE_OUT]),
            SCE_KERNEL_ERROR_ESRCH
        );
    }

    #[test]
    fn delete_before_wait_still_delivers_the_finished_read() {
        let (kernel, fd) = kernel_with_file("del-deliver", b"deliver!");
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        write_request(&mem, REQ, 0, 8, BUF, RESULT, fd);
        assert_eq!(hle_submit_read_commands(&ctx, &[REQ, 1, 0, ID_OUT]), OK);
        let id = read_i32(&mem, ID_OUT);
        // Ensure the worker finished (engine wait, no drain — the guest has
        // not observed anything yet).
        assert_eq!(
            kernel
                .aio
                .wait(id, Some(std::time::Duration::from_secs(10))),
            Ok(AIO_STATE_COMPLETED)
        );
        assert_eq!(read_u32(&mem, RESULT + 0x08), AIO_STATE_SUBMITTED);
        // Delete without ever waiting/polling: the completed read must still
        // land in the guest buffer + result struct on the way out.
        assert_eq!(hle_delete_request(&ctx, &[id as u64, STATE_OUT]), OK);
        let mut data = [0u8; 8];
        assert!(mem.read(BUF, &mut data));
        assert_eq!(&data, b"deliver!");
        assert_eq!(read_i64(&mem, RESULT), 8);
        assert_eq!(read_u32(&mem, RESULT + 0x08), AIO_STATE_COMPLETED);
    }

    #[test]
    fn cancel_request_null_id_reports_processing_and_completed_is_noop() {
        let (kernel, fd) = kernel_with_file("cancel", b"xyzw");
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // id 0 sentinel: nothing to cancel, PROCESSING reported.
        assert_eq!(hle_cancel_request(&ctx, &[0, STATE_OUT]), OK);
        assert_eq!(read_i32(&mem, STATE_OUT), AIO_STATE_PROCESSING as i32);

        // Cancel after completion: no-op, COMPLETED, result untouched by the
        // cancel (already delivered by the wait).
        write_request(&mem, REQ, 0, 4, BUF, RESULT, fd);
        assert_eq!(hle_submit_read_commands(&ctx, &[REQ, 1, 0, ID_OUT]), OK);
        let id = read_i32(&mem, ID_OUT);
        assert_eq!(hle_wait_request(&ctx, &[id as u64, STATE_OUT, 0]), OK);
        assert_eq!(hle_cancel_request(&ctx, &[id as u64, STATE_OUT]), OK);
        assert_eq!(read_i32(&mem, STATE_OUT), AIO_STATE_COMPLETED as i32);
        assert_eq!(read_u32(&mem, RESULT + 0x08), AIO_STATE_COMPLETED);

        // Unknown id → ESRCH.
        assert_eq!(
            hle_cancel_request(&ctx, &[0x7CCC_CCCC, STATE_OUT]),
            SCE_KERNEL_ERROR_ESRCH
        );
    }

    #[test]
    fn submit_argument_validation() {
        let (kernel, fd) = kernel_with_file("validate", b"abc");
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        // Null request array / null id out → EFAULT.
        assert_eq!(
            hle_submit_read_commands(&ctx, &[0, 1, 0, ID_OUT]),
            SCE_KERNEL_ERROR_EFAULT
        );
        assert_eq!(
            hle_submit_read_commands(&ctx, &[REQ, 1, 0, 0]),
            SCE_KERNEL_ERROR_EFAULT
        );
        // size <= 0 → EINVAL.
        assert_eq!(
            hle_submit_read_commands(&ctx, &[REQ, 0, 0, ID_OUT]),
            SCE_KERNEL_ERROR_EINVAL
        );
        // Negative offset / nbyte → EINVAL.
        write_request(&mem, REQ, -1, 4, BUF, RESULT, fd);
        assert_eq!(
            hle_submit_read_commands(&ctx, &[REQ, 1, 0, ID_OUT]),
            SCE_KERNEL_ERROR_EINVAL
        );
        write_request(&mem, REQ, 0, -4, BUF, RESULT, fd);
        assert_eq!(
            hle_submit_read_commands(&ctx, &[REQ, 1, 0, ID_OUT]),
            SCE_KERNEL_ERROR_EINVAL
        );
        // Read buffer outside guest memory → EFAULT.
        write_request(&mem, REQ, 0, 4, 0xFFFF_0000, RESULT, fd);
        assert_eq!(
            hle_submit_read_commands(&ctx, &[REQ, 1, 0, ID_OUT]),
            SCE_KERNEL_ERROR_EFAULT
        );
        // Wait/poll state pointer null → EFAULT.
        assert_eq!(hle_wait_request(&ctx, &[1, 0, 0]), SCE_KERNEL_ERROR_EFAULT);
        assert_eq!(hle_poll_request(&ctx, &[1, 0]), SCE_KERNEL_ERROR_EFAULT);
        assert_eq!(hle_delete_request(&ctx, &[1, 0]), SCE_KERNEL_ERROR_EFAULT);
    }

    #[test]
    fn failed_read_reports_negative_sce_return_value_in_the_result_struct() {
        let (kernel, _fd) = kernel_with_file("badfd", b"abc");
        let mem = crate::TestMemory::new(0x1000);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(&kernel, &mem, &alloc);

        write_request(&mem, REQ, 0, 4, BUF, RESULT, 0x7FFF_0002);
        assert_eq!(hle_submit_read_commands(&ctx, &[REQ, 1, 0, ID_OUT]), OK);
        let id = read_i32(&mem, ID_OUT);
        assert_eq!(hle_wait_request(&ctx, &[id as u64, STATE_OUT, 0]), OK);
        // The batch completed (it was processed); the request itself aborted
        // with a negative EBADF returnValue.
        assert_eq!(read_i32(&mem, STATE_OUT), AIO_STATE_COMPLETED as i32);
        assert_eq!(read_u32(&mem, RESULT + 0x08), AIO_STATE_ABORTED);
        let rv = read_i64(&mem, RESULT);
        assert!(rv < 0, "returnValue {rv:#x} should be negative");
        assert_eq!(rv as i32 as u32, 0x8002_0009); // EBADF
    }

    #[test]
    fn registry_registers_the_measured_family() {
        let registry = HleRegistry::new();
        for name in [
            "sceKernelAioSubmitReadCommands",
            "sceKernelAioSubmitWriteCommands",
            "sceKernelAioWaitRequest",
            "sceKernelAioPollRequests",
            "sceKernelAioDeleteRequest",
            "sceKernelAioSubmitReadCommandsMultiple",
            "sceKernelAioSubmitWriteCommandsMultiple",
            "sceKernelAioWaitRequests",
            "sceKernelAioPollRequest",
            "sceKernelAioCancelRequest",
            "sceKernelAioCancelRequests",
            "sceKernelAioDeleteRequests",
        ] {
            assert!(
                registry.is_implemented("libkernel", name),
                "{name} is not registered"
            );
        }
    }
}
