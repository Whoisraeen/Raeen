//! HLE libkernel **event queues** (`sceKernelCreateEqueue` + user events).
//!
//! A kqueue-like event-notification primitive. This is a faithful Rust port of
//! the **user-event** core of SharpEmu's `KernelEventQueueCompatExports`
//! (GPL-2.0): a title creates a queue, registers user events on it
//! (`AddUserEvent`), triggers them (`TriggerUserEvent`), and collects pending
//! events with `WaitEqueue` (which fills an array of 32-byte `SceKernelEvent`
//! structs). The `GetEvent*` accessors read fields back out of a delivered
//! event.
//!
//! Registration/trigger/delivery is **fully correct** under XPS5X's
//! single-active-execution model. `WaitEqueue` blocks a real thread when no
//! event is pending; with one guest thread nothing else can trigger, so it
//! delivers immediately when events are pending and otherwise reports a
//! timeout. AMPR/graphics events and true blocking waits need the M1-E/M2
//! infrastructure. State lives in the kernel (`kernel_equeues` /
//! `kernel_equeue_events`).

use crate::{HleContext, HleRegistry};
use tracing::debug;

const OK: u64 = 0;
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;
const SCE_KERNEL_ERROR_ESRCH: u64 = 0x8002_0003;
const SCE_KERNEL_ERROR_EFAULT: u64 = 0x8002_000E;
const SCE_KERNEL_ERROR_ETIMEDOUT: u64 = 0x8002_003C;

/// `EVFILT_USER`, the filter reported for user events.
const KERNEL_EVENT_FILTER_USER: i16 = -11;
/// Size of a `SceKernelEvent` struct.
const KERNEL_EVENT_SIZE: u64 = 0x20;

/// Register the event-queue HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "sceKernelCreateEqueue", hle_create);
    registry.register("libkernel", "sceKernelDeleteEqueue", hle_delete);
    registry.register("libkernel", "sceKernelAddUserEvent", hle_add_user_event);
    registry.register("libkernel", "sceKernelAddUserEventEdge", hle_add_user_event);
    registry.register(
        "libkernel",
        "sceKernelDeleteUserEvent",
        hle_delete_user_event,
    );
    registry.register(
        "libkernel",
        "sceKernelTriggerUserEvent",
        hle_trigger_user_event,
    );
    registry.register("libkernel", "sceKernelWaitEqueue", hle_wait);
    registry.register("libkernel", "sceKernelGetEventId", hle_get_event_id);
    registry.register("libkernel", "sceKernelGetEventFilter", hle_get_event_filter);
    registry.register("libkernel", "sceKernelGetEventData", hle_get_event_data);
    registry.register(
        "libkernel",
        "sceKernelGetEventUserData",
        hle_get_event_user_data,
    );
}

/// `sceKernelCreateEqueue(out, name)`: allocate a queue, write its handle.
fn hle_create(ctx: &HleContext, args: &[u64]) -> u64 {
    let out = args.first().copied().unwrap_or(0);
    if out == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }
    let handle = ctx.kernel.create_equeue(0);
    if !ctx.mem.write(out, &handle.to_le_bytes()) {
        ctx.kernel.kernel_equeues.remove(&handle);
        return SCE_KERNEL_ERROR_EFAULT;
    }
    debug!("sceKernelCreateEqueue -> handle {handle:#x}");
    OK
}

/// `sceKernelDeleteEqueue(eq)`: drop the queue and its registered events.
fn hle_delete(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    if ctx.kernel.kernel_equeues.remove(&eq).is_none() {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    ctx.kernel.kernel_equeue_events.retain(|k, _| k.0 != eq);
    OK
}

/// `sceKernelAddUserEvent[Edge](eq, id)`: register an (initially un-triggered)
/// user event on the queue.
fn hle_add_user_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    let id = args.get(1).copied().unwrap_or(0);
    if !ctx.kernel.kernel_equeues.contains_key(&eq) {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    ctx.kernel
        .kernel_equeue_events
        .insert((eq, id), xps5x_kernel::EqueueUserEvent::default());
    OK
}

/// `sceKernelDeleteUserEvent(eq, id)`.
fn hle_delete_user_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    let id = args.get(1).copied().unwrap_or(0);
    if ctx.kernel.kernel_equeue_events.remove(&(eq, id)).is_none() {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    OK
}

/// `sceKernelTriggerUserEvent(eq, id, udata)`: mark the user event pending.
fn hle_trigger_user_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    let id = args.get(1).copied().unwrap_or(0);
    let udata = args.get(2).copied().unwrap_or(0);
    let Some(mut ev) = ctx.kernel.kernel_equeue_events.get_mut(&(eq, id)) else {
        return SCE_KERNEL_ERROR_ESRCH;
    };
    ev.triggered = true;
    ev.udata = udata;
    ev.fflags += 1;
    OK
}

/// `sceKernelWaitEqueue(eq, events, num, out_count, timeout)`: deliver up to
/// `num` pending events (edge-clearing them) as `SceKernelEvent` structs, and
/// write the delivered count. No pending events → timeout.
fn hle_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    let events_ptr = args.get(1).copied().unwrap_or(0);
    let num = args.get(2).copied().unwrap_or(0);
    let out_count = args.get(3).copied().unwrap_or(0);

    if !ctx.kernel.kernel_equeues.contains_key(&eq) {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    if events_ptr == 0 || num == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }

    // Collect pending (ident, udata, fflags) for this queue, then edge-clear.
    let mut pending: Vec<(u64, u64, u32)> = Vec::new();
    for entry in ctx.kernel.kernel_equeue_events.iter() {
        let (q, id) = *entry.key();
        if q == eq && entry.triggered && (pending.len() as u64) < num {
            pending.push((id, entry.udata, entry.fflags));
        }
    }

    if pending.is_empty() {
        // Nothing pending; report zero and time out (no other thread can fire).
        if out_count != 0 {
            let _ = ctx.mem.write(out_count, &0u32.to_le_bytes());
        }
        return SCE_KERNEL_ERROR_ETIMEDOUT;
    }

    for (i, &(id, udata, fflags)) in pending.iter().enumerate() {
        let addr = events_ptr + i as u64 * KERNEL_EVENT_SIZE;
        if !write_kernel_event(ctx, addr, id, udata, fflags) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        // Edge-triggered: clear the pending flag now that it's delivered.
        if let Some(mut ev) = ctx.kernel.kernel_equeue_events.get_mut(&(eq, id)) {
            ev.triggered = false;
        }
    }
    if out_count != 0
        && !ctx
            .mem
            .write(out_count, &(pending.len() as u32).to_le_bytes())
    {
        return SCE_KERNEL_ERROR_EFAULT;
    }
    OK
}

/// Write a `SceKernelEvent` (32 bytes) for a delivered user event.
fn write_kernel_event(ctx: &HleContext, addr: u64, ident: u64, udata: u64, fflags: u32) -> bool {
    let mut b = [0u8; KERNEL_EVENT_SIZE as usize];
    b[0x00..0x08].copy_from_slice(&ident.to_le_bytes());
    b[0x08..0x0A].copy_from_slice(&KERNEL_EVENT_FILTER_USER.to_le_bytes());
    // flags (0x0A) left 0.
    b[0x0C..0x10].copy_from_slice(&fflags.to_le_bytes());
    // data (0x10) left 0.
    b[0x18..0x20].copy_from_slice(&udata.to_le_bytes());
    ctx.mem.write(addr, &b)
}

/// Read an 8-byte field at `event_ptr + off`, or 0 if unreadable.
fn read_event_u64(ctx: &HleContext, event_ptr: u64, off: u64) -> u64 {
    let mut b = [0u8; 8];
    if event_ptr != 0 && ctx.mem.read(event_ptr + off, &mut b) {
        u64::from_le_bytes(b)
    } else {
        0
    }
}

/// `sceKernelGetEventId(ev)`: the event's `ident` (offset 0x00).
fn hle_get_event_id(ctx: &HleContext, args: &[u64]) -> u64 {
    read_event_u64(ctx, args.first().copied().unwrap_or(0), 0x00)
}

/// `sceKernelGetEventFilter(ev)`: the event's `filter` (offset 0x08, i16).
fn hle_get_event_filter(ctx: &HleContext, args: &[u64]) -> u64 {
    let ev = args.first().copied().unwrap_or(0);
    let mut b = [0u8; 2];
    if ev != 0 && ctx.mem.read(ev + 0x08, &mut b) {
        // Sign-extend the i16 filter to the return register.
        i64::from(i16::from_le_bytes(b)) as u64
    } else {
        0
    }
}

/// `sceKernelGetEventData(ev)`: the event's `data` (offset 0x10).
fn hle_get_event_data(ctx: &HleContext, args: &[u64]) -> u64 {
    read_event_u64(ctx, args.first().copied().unwrap_or(0), 0x10)
}

/// `sceKernelGetEventUserData(ev)`: the event's `udata` (offset 0x18).
fn hle_get_event_user_data(ctx: &HleContext, args: &[u64]) -> u64 {
    read_event_u64(ctx, args.first().copied().unwrap_or(0), 0x18)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, test_ctx};

    fn ctx_env() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = xps5x_kernel::OrbisKernel::new();
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        (kernel, mem, alloc)
    }

    fn create(ctx: &HleContext) -> u64 {
        assert_eq!(hle_create(ctx, &[0x100]), OK);
        let mut b = [0u8; 8];
        assert!(ctx.mem.read(0x100, &mut b));
        u64::from_le_bytes(b)
    }

    #[test]
    fn trigger_then_wait_delivers_the_event() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = create(&ctx);
        // Register user event id=5, trigger with udata=0xABCD.
        assert_eq!(hle_add_user_event(&ctx, &[eq, 5]), OK);
        assert_eq!(hle_trigger_user_event(&ctx, &[eq, 5, 0xABCD]), OK);
        // Wait: 1 event delivered at 0x200, count at 0x1F0.
        assert_eq!(hle_wait(&ctx, &[eq, 0x200, 4, 0x1F0]), OK);
        let mut cnt = [0u8; 4];
        assert!(mem.read(0x1F0, &mut cnt));
        assert_eq!(u32::from_le_bytes(cnt), 1);
        // Read the event fields back via the accessors.
        assert_eq!(hle_get_event_id(&ctx, &[0x200]), 5);
        assert_eq!(hle_get_event_filter(&ctx, &[0x200]), (-11i64) as u64);
        assert_eq!(hle_get_event_user_data(&ctx, &[0x200]), 0xABCD);
        // Edge-cleared: a second wait finds nothing pending → timeout.
        assert_eq!(
            hle_wait(&ctx, &[eq, 0x200, 4, 0x1F0]),
            SCE_KERNEL_ERROR_ETIMEDOUT
        );
    }

    #[test]
    fn wait_with_no_pending_events_times_out() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = create(&ctx);
        hle_add_user_event(&ctx, &[eq, 7]); // registered but not triggered
        assert_eq!(
            hle_wait(&ctx, &[eq, 0x200, 4, 0x1F0]),
            SCE_KERNEL_ERROR_ETIMEDOUT
        );
        let mut cnt = [0u8; 4];
        assert!(mem.read(0x1F0, &mut cnt));
        assert_eq!(u32::from_le_bytes(cnt), 0);
    }

    #[test]
    fn lifecycle_and_error_paths() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_create(&ctx, &[0]), SCE_KERNEL_ERROR_EINVAL);
        let eq = create(&ctx);
        // Adding/triggering on unknown queue / event → ESRCH.
        assert_eq!(
            hle_add_user_event(&ctx, &[0xDEAD, 1]),
            SCE_KERNEL_ERROR_ESRCH
        );
        assert_eq!(
            hle_trigger_user_event(&ctx, &[eq, 99, 0]),
            SCE_KERNEL_ERROR_ESRCH
        );
        // Delete removes the queue + its events; second delete → ESRCH.
        hle_add_user_event(&ctx, &[eq, 1]);
        assert_eq!(hle_delete(&ctx, &[eq]), OK);
        assert_eq!(hle_delete(&ctx, &[eq]), SCE_KERNEL_ERROR_ESRCH);
        assert!(!kernel.kernel_equeue_events.contains_key(&(eq, 1)));
        let _ = mem; // silence unused in this path
    }
}
