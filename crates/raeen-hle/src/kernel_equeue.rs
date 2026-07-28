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
//! `WaitEqueue` uses Raeen's shared wait service, so a NULL guest timeout means
//! a real indefinite wait while finite deadlines remain guest-visible. User,
//! VideoOut, APR, and AGC producers wake blocked waiters after publishing their
//! events. State lives in the kernel (`kernel_equeues` /
//! `kernel_equeue_events`).

use crate::{HleContext, HleRegistry};
use raeen_core::subsystems::{WaitKey, WaitOutcome, WakeReason};
use tracing::debug;

const OK: u64 = 0;
const SCE_KERNEL_ERROR_EINVAL: u64 = 0x8002_0016;
const SCE_KERNEL_ERROR_ESRCH: u64 = 0x8002_0003;
const SCE_KERNEL_ERROR_EFAULT: u64 = 0x8002_000E;
const SCE_KERNEL_ERROR_ETIMEDOUT: u64 = 0x8002_003C;

/// Size of a `SceKernelEvent` struct.
const KERNEL_EVENT_SIZE: u64 = 0x20;

/// Register the event-queue HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libkernel", "sceKernelCreateEqueue", hle_create);
    registry.register("libkernel", "sceKernelDeleteEqueue", hle_delete);
    registry.register("libkernel", "sceKernelAddUserEvent", hle_add_user_event);
    registry.register("libkernel", "sceKernelAddUserEventEdge", hle_add_user_event);
    registry.register("libkernel", "sceKernelAddAmprEvent", hle_add_ampr_event);
    registry.register("libkernel", "sceKernelAddReadEvent", hle_add_read_event);
    registry.register(
        "libkernel",
        "sceKernelDeleteReadEvent",
        hle_delete_read_event,
    );
    registry.register("libkernel", "sceKernelAddWriteEvent", hle_add_write_event);
    registry.register(
        "libkernel",
        "sceKernelDeleteWriteEvent",
        hle_delete_read_event,
    );
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
    wake_equeue(ctx, eq, WakeReason::Deleted);
    OK
}

/// Wake threads blocked in `sceKernelWaitEqueue(eq, ...)`.
///
/// All producers use the Raeen-owned wait contract instead of reaching into
/// the kernel's host synchronization primitive directly.
pub(crate) fn wake_equeue(ctx: &HleContext, eq: u64, reason: WakeReason) {
    ctx.services.wake(
        WaitKey {
            class: "kernel-equeue",
            object: eq,
            guest_thread: ctx.guest_threads.current_thread(),
        },
        reason,
    );
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
        .insert((eq, id), raeen_kernel::EqueueUserEvent::default());
    debug!(eq, id, "registered kernel user event");
    OK
}

/// `sceKernelAddAmprEvent(eq, id, data)`: register an AMPR event — SharpEmu
/// `KernelEventQueueCompatExports.KernelAddAmprEvent`. Same queue model as a
/// user event (the filter distinction is internal to the kernel; a later
/// trigger fires it either way), with `data` as the event's udata. Measured:
/// Dragon Ball right after PlayGo init.
fn hle_add_ampr_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    let id = args.get(1).copied().unwrap_or(0);
    let data = args.get(2).copied().unwrap_or(0);
    if !ctx.kernel.kernel_equeues.contains_key(&eq) {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    ctx.kernel.kernel_equeue_events.insert(
        (eq, id),
        raeen_kernel::EqueueUserEvent {
            udata: data,
            ..Default::default()
        },
    );
    debug!(eq, id, data, "registered kernel AMPR event");
    OK
}

/// `sceKernelAddReadEvent(eq, fd, udata)`: attach an offline socket read
/// interest to an event queue. It remains untriggered until a socket backend
/// receives data; Raeen deliberately has no host-network backend.
fn hle_add_read_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    let fd = args.get(1).copied().unwrap_or(0) as i32;
    let udata = args.get(2).copied().unwrap_or(0);
    if !ctx.kernel.kernel_equeues.contains_key(&eq) || !ctx.kernel.kernel_sockets.contains_key(&fd)
    {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    ctx.kernel.kernel_equeue_events.insert(
        (eq, fd as u32 as u64),
        raeen_kernel::EqueueUserEvent {
            udata,
            ..Default::default()
        },
    );
    debug!(eq, fd, udata, "registered offline socket read event");
    OK
}

/// `sceKernelAddWriteEvent(eq, fd, udata)`: attach a write-readiness interest
/// to an event queue. Accepted for offline sockets and VFS descriptors, but
/// registered **untriggered** and never fired: with no host-network backend a
/// socket never becomes writable, and no measured title yet waits on file
/// writability. Delivering a fake "writable" event would make a title write
/// into a connection that does not exist.
fn hle_add_write_event(ctx: &HleContext, args: &[u64]) -> u64 {
    const EVFILT_WRITE: i16 = -2;
    let eq = args.first().copied().unwrap_or(0);
    let fd = args.get(1).copied().unwrap_or(0) as i32;
    let udata = args.get(2).copied().unwrap_or(0);
    let known_fd =
        ctx.kernel.kernel_sockets.contains_key(&fd) || ctx.kernel.filesystem.flags(fd).is_some();
    if !ctx.kernel.kernel_equeues.contains_key(&eq) || !known_fd {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    ctx.kernel.kernel_equeue_events.insert(
        (eq, fd as u32 as u64),
        raeen_kernel::EqueueUserEvent {
            udata,
            filter: EVFILT_WRITE,
            ..Default::default()
        },
    );
    debug!(
        eq,
        fd, udata, "registered write event (never fires: offline)"
    );
    OK
}

/// `sceKernelDeleteReadEvent(eq, fd)` / `sceKernelDeleteWriteEvent(eq, fd)`:
/// remove a descriptor interest from an event queue. The event identity is
/// the descriptor, matching the registration performed by
/// [`hle_add_read_event`] / [`hle_add_write_event`].
fn hle_delete_read_event(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    let fd = args.get(1).copied().unwrap_or(0) as i32;
    if !ctx.kernel.kernel_equeues.contains_key(&eq) {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    if ctx
        .kernel
        .kernel_equeue_events
        .remove(&(eq, fd as u32 as u64))
        .is_none()
    {
        return SCE_KERNEL_ERROR_ESRCH;
    }
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
#[allow(clippy::needless_return)]
fn hle_trigger_user_event(ctx: &HleContext, args: &[u64]) -> u64 {
    if std::env::var_os("RAEEN_TRACE_EQUEUE").is_some() {
        tracing::warn!(
            eq = format_args!("{:#x}", args.first().copied().unwrap_or(0)),
            id = format_args!("{:#x}", args.get(1).copied().unwrap_or(0)),
            known = ctx.kernel.kernel_equeue_events.contains_key(&(
                args.first().copied().unwrap_or(0),
                args.get(1).copied().unwrap_or(0)
            )),
            "TRACE_EQUEUE: TriggerUserEvent called"
        );
    }
    hle_trigger_user_event_inner(ctx, args)
}

fn hle_trigger_user_event_inner(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    let id = args.get(1).copied().unwrap_or(0);
    let udata = args.get(2).copied().unwrap_or(0);
    let Some(mut ev) = ctx.kernel.kernel_equeue_events.get_mut(&(eq, id)) else {
        return SCE_KERNEL_ERROR_ESRCH;
    };
    ev.triggered = true;
    ev.udata = udata;
    ev.fflags += 1;
    drop(ev);
    wake_equeue(ctx, eq, WakeReason::Signal);
    OK
}

/// `sceKernelWaitEqueue(eq, events, num, out_count, timeout)`: deliver up to
/// `num` pending events (edge-clearing them) as `SceKernelEvent` structs, and
/// write the delivered count. No pending events → timeout.
/// How long a single host park may last: the internal slice, never overshooting
/// the caller's own deadline.
///
/// A NULL guest timeout (`deadline == None`) always parks a full slice — it is
/// an indefinite wait, and the slice exists only so process teardown is noticed
/// promptly. A finite deadline parks `min(remaining, slice)` so the wait neither
/// overshoots the interval the guest asked for nor blocks past teardown.
///
/// Pure so the slice/deadline arithmetic is testable against a synthetic clock:
/// the wall-clock version of this test raced under parallel load.
fn equeue_park_slice(
    deadline: Option<std::time::Instant>,
    now: std::time::Instant,
    slice: std::time::Duration,
) -> std::time::Duration {
    deadline.map_or(slice, |dl| dl.saturating_duration_since(now).min(slice))
}

/// Whether a park that timed out is the **guest's** timeout, or merely an
/// internal slice expiring.
///
/// This is the whole contract the equeue wait had to be corrected to honour.
/// Returning `true` for a slice expiry is exactly the old bug: Dragon Ball's AGC
/// workers took a fabricated `ETIMEDOUT` after 50 ms and entered the title's
/// fatal-reporting path before their first submission. So:
///
/// * NULL timeout — **never** a guest timeout, however many slices elapse; and
/// * finite timeout — a guest timeout only once the real deadline has arrived,
///   not at the first internal slice boundary.
fn equeue_deadline_reached(deadline: Option<std::time::Instant>, now: std::time::Instant) -> bool {
    deadline.is_some_and(|dl| now >= dl)
}

fn hle_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    let eq = args.first().copied().unwrap_or(0);
    let events_ptr = args.get(1).copied().unwrap_or(0);
    let num = args.get(2).copied().unwrap_or(0);
    let out_count = args.get(3).copied().unwrap_or(0);
    let timeout_ptr = args.get(4).copied().unwrap_or(0);

    if !ctx.kernel.kernel_equeues.contains_key(&eq) {
        return SCE_KERNEL_ERROR_ESRCH;
    }
    if events_ptr == 0 || num == 0 {
        return SCE_KERNEL_ERROR_EINVAL;
    }

    // The 5th arg is `SceKernelUseconds*` (NULL = wait forever). Ignoring it and
    // reporting an instant timeout turned every guest event loop into a hot spin:
    // measured 2.29 MILLION `sceKernelWaitEqueue` calls in one Minecraft run,
    // two threads at 100% CPU, starving the threads that had real work — while
    // the queue's producer was firing events correctly all along (169 triggers).
    // Waiting for the interval the caller asked for is both the ABI and what
    // stops the spin.
    let timeout_us = if timeout_ptr == 0 {
        None
    } else {
        let mut buf = [0u8; 4];
        if !ctx.mem.read(timeout_ptr, &mut buf) {
            return SCE_KERNEL_ERROR_EFAULT;
        }
        Some(u64::from(u32::from_le_bytes(buf)))
    };

    // NULL means FOREVER. The host wait remains sliced so process teardown is
    // observed promptly, but an internal slice expiry is never exposed to the
    // guest as ETIMEDOUT. The old implementation did exactly that after 50 ms:
    // Dragon Ball's AGC workers entered their fatal path after a fabricated
    // timeout. This mirrors the corrected WaitEventFlag contract.
    const WAIT_SLICE: std::time::Duration = std::time::Duration::from_millis(50);
    // The deferred worker queue has no kernel handle with which to notify this
    // condvar. Keep its observation latency bounded without returning the
    // slice as a guest timeout; normal eager/event producers wake immediately.
    let wait_slice = if raeen_gpu::ordered_side_effects::defer_gpu_side_effects() {
        std::time::Duration::from_millis(1)
    } else {
        WAIT_SLICE
    };
    let deadline =
        timeout_us.map(|us| std::time::Instant::now() + std::time::Duration::from_micros(us));

    loop {
        // Deliver anything the GPU worker executed in-stream since the last
        // poll (events/EOP/flips under `RAEEN_DEFER_GPU_SIDE_EFFECTS`; a
        // relaxed-load no-op otherwise) — a waiter parked here IS the
        // observer those effects must reach.
        crate::libsce_agc::apply_ordered_gpu_side_effects(ctx);
        let delivered = std::cell::Cell::new(None);
        let deleted = std::cell::Cell::new(false);
        let mut ready = || {
            if !ctx.kernel.kernel_equeues.contains_key(&eq) {
                deleted.set(true);
                return true;
            }
            // Collect and deliver while the wait subsystem's notification lock
            // is held. Producers take that same lock in `wake_equeue`, closing
            // the check-then-sleep lost-wakeup race.
            let mut pending: Vec<(u64, u64, u32, i16, i64)> = Vec::new();
            for entry in ctx.kernel.kernel_equeue_events.iter() {
                let (q, id) = *entry.key();
                if q == eq && entry.triggered && (pending.len() as u64) < num {
                    pending.push((id, entry.udata, entry.fflags, entry.filter, entry.data));
                }
            }
            if pending.is_empty() {
                false
            } else {
                delivered.set(Some(deliver_events(
                    ctx, eq, events_ptr, out_count, &pending,
                )));
                true
            }
        };

        let wait = equeue_park_slice(deadline, std::time::Instant::now(), wait_slice);
        let outcome = ctx.services.wait_until(
            WaitKey {
                class: "kernel-equeue",
                object: eq,
                guest_thread: ctx.guest_threads.current_thread(),
            },
            wait,
            &|| ctx.guest_threads.process_is_terminating(),
            &mut ready,
        );
        if deleted.get() {
            return SCE_KERNEL_ERROR_ESRCH;
        }
        if let Some(result) = delivered.get() {
            return result;
        }
        if outcome == WaitOutcome::Terminating {
            return OK;
        }
        if outcome == WaitOutcome::TimedOut
            && equeue_deadline_reached(deadline, std::time::Instant::now())
        {
            break;
        }
        // An internal 50 ms slice elapsed. A NULL-timeout wait loops forever;
        // a finite wait loops until its actual deadline.
    }

    // Waited out the caller's interval with nothing pending: report zero.
    if out_count != 0 {
        let _ = ctx.mem.write(out_count, &0u32.to_le_bytes());
    }
    if std::env::var_os("RAEEN_TRACE_EQUEUE").is_some() {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEEN: AtomicU32 = AtomicU32::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 24 {
            let registered: Vec<String> = ctx
                .kernel
                .kernel_equeue_events
                .iter()
                .filter(|e| e.key().0 == eq)
                .map(|e| {
                    format!(
                        "id={:#x},filter={},trig={}",
                        e.key().1,
                        e.filter,
                        e.triggered
                    )
                })
                .collect();
            tracing::warn!(
                eq = format_args!("{eq:#x}"),
                want = num,
                waited_us = timeout_us.unwrap_or(0),
                registered_count = registered.len(),
                registered = ?registered,
                "TRACE_EQUEUE: wait timed out"
            );
        }
    }
    debug!(eq, num, "kernel event wait timed out");
    SCE_KERNEL_ERROR_ETIMEDOUT
}

/// Write the pending events into the guest's array, edge-clear them, and report
/// the delivered count.
fn deliver_events(
    ctx: &HleContext,
    eq: u64,
    events_ptr: u64,
    out_count: u64,
    pending: &[(u64, u64, u32, i16, i64)],
) -> u64 {
    for (i, &(id, udata, fflags, filter, data)) in pending.iter().enumerate() {
        let addr = events_ptr + i as u64 * KERNEL_EVENT_SIZE;
        if !write_kernel_event(ctx, addr, id, udata, fflags, filter, data) {
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
    debug!(eq, delivered = pending.len(), "delivered kernel events");
    OK
}

/// Write a `SceKernelEvent` (32 bytes) for a delivered user event.
fn write_kernel_event(
    ctx: &HleContext,
    addr: u64,
    ident: u64,
    udata: u64,
    fflags: u32,
    filter: i16,
    data: i64,
) -> bool {
    let mut b = [0u8; KERNEL_EVENT_SIZE as usize];
    b[0x00..0x08].copy_from_slice(&ident.to_le_bytes());
    b[0x08..0x0A].copy_from_slice(&filter.to_le_bytes());
    // flags (0x0A) left 0.
    b[0x0C..0x10].copy_from_slice(&fflags.to_le_bytes());
    b[0x10..0x18].copy_from_slice(&data.to_le_bytes());
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
        raeen_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        let kernel = raeen_kernel::OrbisKernel::new();
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
        assert!(mem.write(0x120, &1_000u32.to_le_bytes()));
        assert_eq!(
            hle_wait(&ctx, &[eq, 0x200, 4, 0x1F0, 0x120]),
            SCE_KERNEL_ERROR_ETIMEDOUT
        );
    }

    /// Ordered GPU side effects (checklist item 5, step 4): an in-stream
    /// `EVENT_WRITE` the GPU worker published is delivered by the wait loop's
    /// drain — a guest parked in `sceKernelWaitEqueue` IS the observation
    /// point those effects must reach.
    #[test]
    fn wait_delivers_worker_published_gpu_events() {
        // The hand-off queue is process-global: serialize with every other
        // test that touches it.
        let _guard = crate::SIDEFX_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ = raeen_gpu::ordered_side_effects::drain();
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = create(&ctx);
        assert_eq!(hle_add_user_event(&ctx, &[eq, 0x2A]), OK);
        raeen_gpu::ordered_side_effects::publish([
            raeen_gpu::ordered_side_effects::OrderedGpuSideEffect::EventWrite { event_id: 0x2A },
        ]);
        assert_eq!(hle_wait(&ctx, &[eq, 0x200, 4, 0x1F0]), OK);
        assert_eq!(hle_get_event_id(&ctx, &[0x200]), 0x2A);
        let mut cnt = [0u8; 4];
        assert!(mem.read(0x1F0, &mut cnt));
        assert_eq!(u32::from_le_bytes(cnt), 1);
    }

    #[test]
    fn wait_with_no_pending_events_times_out() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = create(&ctx);
        hle_add_user_event(&ctx, &[eq, 7]); // registered but not triggered
        assert!(mem.write(0x120, &1_000u32.to_le_bytes()));
        assert_eq!(
            hle_wait(&ctx, &[eq, 0x200, 4, 0x1F0, 0x120]),
            SCE_KERNEL_ERROR_ETIMEDOUT
        );
        let mut cnt = [0u8; 4];
        assert!(mem.read(0x1F0, &mut cnt));
        assert_eq!(u32::from_le_bytes(cnt), 0);
    }

    /// A NULL timeout is an indefinite wait, not a 50 ms timeout, and a finite
    /// timeout is the caller's interval — not the internal slice. Dragon Ball's
    /// AGC workers use both shapes; the old single-slice implementation returned
    /// ETIMEDOUT at 50 ms and sent every worker into the title's fatal-reporting
    /// path before its first submission.
    ///
    /// Driven against a **synthetic clock** rather than real elapsed time. The
    /// previous versions of these two tests spawned a producer that slept 80 ms
    /// and asserted a 200 ms guest deadline had not expired; under a full
    /// parallel `cargo test --workspace` the producer could miss its slot and
    /// the deadline would pass first, so they failed under load and passed in
    /// isolation. The property being protected is arithmetic, not timing, so it
    /// is tested as arithmetic — which also makes it exhaustive rather than
    /// probabilistic.
    #[test]
    fn an_internal_slice_expiry_is_never_reported_as_the_guests_timeout() {
        use std::time::{Duration, Instant};
        const SLICE: Duration = Duration::from_millis(50);
        let t0 = Instant::now();

        // NULL timeout: parks a full slice, forever, and no number of elapsed
        // slices ever becomes a guest timeout.
        assert_eq!(equeue_park_slice(None, t0, SLICE), SLICE);
        for slices in [1u32, 2, 100, 10_000] {
            let later = t0 + SLICE * slices;
            assert_eq!(
                equeue_park_slice(None, later, SLICE),
                SLICE,
                "a NULL timeout must keep parking full slices"
            );
            assert!(
                !equeue_deadline_reached(None, later),
                "a NULL timeout must never report a guest timeout ({slices} slices in)"
            );
        }

        // Finite 200 ms deadline vs a 50 ms slice — the exact shape that used to
        // race. It parks in slices and stays un-expired across each of the first
        // three boundaries.
        let deadline = t0 + Duration::from_millis(200);
        for slices in 0..3u32 {
            let now = t0 + SLICE * slices;
            assert_eq!(
                equeue_park_slice(Some(deadline), now, SLICE),
                SLICE,
                "a 200 ms deadline must park a full 50 ms slice at boundary {slices}"
            );
            assert!(
                !equeue_deadline_reached(Some(deadline), now),
                "slice {slices} of a 200 ms deadline must not be a guest timeout"
            );
        }

        // The last park is clamped to what remains, never overshooting the
        // interval the guest asked for...
        assert_eq!(
            equeue_park_slice(Some(deadline), t0 + Duration::from_millis(180), SLICE),
            Duration::from_millis(20),
            "the final park must not overshoot the caller's deadline"
        );
        // ...and only the real deadline ends the wait.
        assert!(!equeue_deadline_reached(
            Some(deadline),
            t0 + Duration::from_millis(199)
        ));
        assert!(equeue_deadline_reached(Some(deadline), deadline));
        assert!(equeue_deadline_reached(
            Some(deadline),
            t0 + Duration::from_millis(201)
        ));
        assert_eq!(
            equeue_park_slice(Some(deadline), t0 + Duration::from_millis(201), SLICE),
            Duration::ZERO,
            "past the deadline there is nothing left to park"
        );

        // A deadline shorter than one slice is honoured as-is, not rounded up.
        let short = t0 + Duration::from_millis(5);
        assert_eq!(
            equeue_park_slice(Some(short), t0, SLICE),
            Duration::from_millis(5)
        );
    }

    /// End-to-end: a producer on another thread wakes a parked waiter and the
    /// event's id and payload arrive intact.
    ///
    /// No sleeps and no elapsed-time assertion. The guest deadline is
    /// deliberately enormous (30 s) so no scheduling delay this test could ever
    /// see can expire it — the slice/deadline arithmetic itself is covered
    /// exhaustively above.
    #[test]
    fn a_producer_thread_wakes_a_parked_equeue_waiter() {
        let kernel = std::sync::Arc::new(raeen_kernel::OrbisKernel::new());
        let mem = crate::TestMemory::new(0x400);
        let alloc = crate::TestAllocator::new(0);
        let ctx = test_ctx(kernel.as_ref(), &mem, &alloc);
        let eq = create(&ctx);
        assert_eq!(hle_add_user_event(&ctx, &[eq, 0x21]), OK);
        assert!(mem.write(0x120, &30_000_000u32.to_le_bytes()));

        let producer = std::thread::spawn({
            let kernel = std::sync::Arc::clone(&kernel);
            move || {
                let mem = crate::TestMemory::new(0x100);
                let alloc = crate::TestAllocator::new(0);
                let ctx = test_ctx(kernel.as_ref(), &mem, &alloc);
                assert_eq!(hle_trigger_user_event(&ctx, &[eq, 0x21, 0xBEEF]), OK);
            }
        });

        assert_eq!(
            hle_wait(&ctx, &[eq, 0x200, 1, 0x1F0, 0x120]),
            OK,
            "the waiter must receive the producer's event"
        );
        producer.join().unwrap();
        assert_eq!(hle_get_event_id(&ctx, &[0x200]), 0x21);
        assert_eq!(hle_get_event_user_data(&ctx, &[0x200]), 0xBEEF);
    }

    #[test]
    fn unreadable_timeout_pointer_is_efault_not_an_infinite_wait() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = create(&ctx);
        assert_eq!(
            hle_wait(&ctx, &[eq, 0x200, 1, 0x1F0, 0xFFFF]),
            SCE_KERNEL_ERROR_EFAULT
        );
    }

    #[test]
    fn offline_socket_read_event_registers_without_becoming_ready() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let eq = create(&ctx);
        let fd = kernel.create_socket().expect("socket quota available");
        assert_eq!(hle_add_read_event(&ctx, &[eq, fd as u64, 0xCAFE]), OK);
        let event = kernel
            .kernel_equeue_events
            .get(&(eq, fd as u32 as u64))
            .unwrap();
        assert_eq!(event.udata, 0xCAFE);
        assert!(!event.triggered);
        drop(event);
        assert_eq!(hle_delete_read_event(&ctx, &[eq, fd as u64]), OK);
        assert!(
            !kernel
                .kernel_equeue_events
                .contains_key(&(eq, fd as u32 as u64))
        );
        assert_eq!(
            hle_delete_read_event(&ctx, &[eq, fd as u64]),
            SCE_KERNEL_ERROR_ESRCH
        );

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libkernel", "sceKernelAddReadEvent"));
        assert!(registry.is_implemented("libkernel", "sceKernelDeleteReadEvent"));
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
