//! Host-backed pthread condition variables for concurrent native guest
//! workers. Wait atomically releases the associated guest mutex while holding
//! the condition generation lock, sleeps, then reacquires before returning.

use tracing::debug;

use crate::{HleContext, HleRegistry};

/// POSIX success. These entry points return errno directly (0 = success),
/// matching `pthread_sync`.
const OK: u64 = 0;
const EINVAL: u64 = 22;
const ETIMEDOUT: u64 = 60;
const COND_OBJECT_SIZE: u64 = 0x100;

pub fn register(registry: &HleRegistry) {
    for library in ["libScePosix", "libkernel"] {
        registry.register(library, "pthread_cond_init", hle_cond_init);
        registry.register(library, "pthread_cond_destroy", hle_cond_destroy);
        registry.register(library, "pthread_cond_wait", hle_cond_wait);
        registry.register(library, "pthread_cond_timedwait", hle_cond_timedwait);
        registry.register(library, "pthread_cond_signal", hle_cond_signal);
        registry.register(library, "pthread_cond_broadcast", hle_cond_broadcast);
    }
    registry.register("libkernel", "scePthreadCondInit", hle_cond_init);
    registry.register("libkernel", "scePthreadCondWait", hle_cond_wait);
    registry.register(
        "libkernel",
        "scePthreadCondTimedwait",
        hle_sce_cond_timedwait,
    );
    registry.register("libkernel", "scePthreadCondSignal", hle_cond_signal);
    registry.register("libkernel", "scePthreadCondBroadcast", hle_cond_broadcast);
    // The SCE spelling of destroy belongs with the rest of the real state
    // machine. `libkernel` previously bound it to a no-op that shared its
    // handler with `CondInit`, so destroying a condition left its state (and any
    // waiters' generation) behind.
    registry.register("libkernel", "scePthreadCondDestroy", hle_cond_destroy);
    registry.register("libScePosix", "pthread_condattr_init", hle_condattr_init);
    registry.register(
        "libScePosix",
        "pthread_condattr_destroy",
        hle_condattr_destroy,
    );
    registry.register(
        "libScePosix",
        "pthread_condattr_setclock",
        hle_condattr_setclock,
    );
    // The SCE-namespaced twins (measured: ASTRO.BOT calls scePthreadCondattrInit
    // from libkernel, nid 0x9b9ff66ec35fbfbb).
    registry.register("libkernel", "scePthreadCondattrInit", hle_condattr_init);
    registry.register(
        "libkernel",
        "scePthreadCondattrDestroy",
        hle_condattr_destroy,
    );
    registry.register(
        "libkernel",
        "scePthreadCondattrSetclock",
        hle_condattr_setclock,
    );
}

/// `pthread_cond_init(cond, attr)`. Orbis condition variables are opaque
/// pointer handles: initialize `*cond`, and retain the same host state under
/// both the guest pointer slot and its allocated handle. Guest libc inspects
/// the slot directly, so leaving it zero after reporting success can make it
/// mistake its own initialized condition for a static/uninitialized object.
fn hle_cond_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    let attr = args.get(1).copied().unwrap_or(0);
    debug!("pthread_cond_init(cond={cond:#x}, attr={attr:#x})");
    if cond == 0 {
        EINVAL
    } else {
        let Some(handle) = ctx.alloc.alloc(COND_OBJECT_SIZE, 0x10) else {
            return EINVAL;
        };
        if !ctx.mem.write(cond, &handle.to_le_bytes()) {
            ctx.alloc.free(handle);
            return EINVAL;
        }
        // Bake the attr's clock into the condition now. POSIX attrs are inputs
        // to init, not live links: a later change to the attr must not reach a
        // cond already built from it, and the attr is usually destroyed
        // immediately after this call anyway.
        let monotonic = ctx
            .kernel
            .pthread_condattr_clocks
            .get(&attr)
            .is_some_and(|clock| *clock == crate::libkernel::CLOCK_MONOTONIC);
        let state = std::sync::Arc::new(xps5x_kernel::PthreadCond::default());
        state
            .monotonic
            .store(monotonic, std::sync::atomic::Ordering::Relaxed);
        ctx.kernel.pthread_conds.insert(cond, state.clone());
        ctx.kernel.pthread_conds.insert(handle, state);
        OK
    }
}

/// `pthread_cond_destroy(cond)`.
fn hle_cond_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    debug!("pthread_cond_destroy(cond={cond:#x})");
    if cond == 0 {
        EINVAL
    } else {
        let handle = read_handle(ctx, cond);
        ctx.kernel.pthread_conds.remove(&cond);
        if let Some(handle) = handle {
            ctx.kernel.pthread_conds.remove(&handle);
            ctx.alloc.free(handle);
        }
        let _ = ctx.mem.write(cond, &0u64.to_le_bytes());
        OK
    }
}

fn read_handle(ctx: &HleContext, cond: u64) -> Option<u64> {
    let mut bytes = [0u8; 8];
    ctx.mem
        .read(cond, &mut bytes)
        .then(|| u64::from_le_bytes(bytes))
        .filter(|handle| *handle != 0)
}

/// Resolve an initialized condition, lazily materializing a zero-initialized
/// static object with the same opaque-handle ABI as `pthread_cond_init`.
fn condition(ctx: &HleContext, cond: u64) -> Option<std::sync::Arc<xps5x_kernel::PthreadCond>> {
    if let Some(state) = ctx.kernel.pthread_conds.get(&cond) {
        return Some(state.clone());
    }
    if let Some(handle) = read_handle(ctx, cond)
        && let Some(state) = ctx.kernel.pthread_conds.get(&handle)
    {
        return Some(state.clone());
    }

    // Implicit creation for a statically-initialized cond, and it MUST be
    // atomic. A waiter and its signaler both land here for the same address, and
    // a plain check-then-insert lets both miss, both create, and the second
    // overwrite the first: the waiter then blocks on one object while the signal
    // goes to another, and it never wakes. That is a silent, permanent lost
    // wakeup — the exact shape of a title stalling with every worker idle.
    // `entry` serializes the miss so only one state is ever published.
    let state = match ctx.kernel.pthread_conds.entry(cond) {
        dashmap::mapref::entry::Entry::Occupied(existing) => return Some(existing.get().clone()),
        dashmap::mapref::entry::Entry::Vacant(slot) => {
            let handle = ctx.alloc.alloc(COND_OBJECT_SIZE, 0x10)?;
            if !ctx.mem.write(cond, &handle.to_le_bytes()) {
                ctx.alloc.free(handle);
                return None;
            }
            let state = std::sync::Arc::new(xps5x_kernel::PthreadCond::default());
            slot.insert(state.clone());
            // Alias the opaque handle to the SAME state, but only after the
            // entry guard is dropped: inserting into this map while holding one
            // of its shard guards deadlocks when both keys hash to that shard.
            (state, handle)
        }
    };
    let (state, handle) = state;
    ctx.kernel.pthread_conds.insert(handle, state.clone());
    Some(state)
}

fn hle_cond_wait(ctx: &HleContext, args: &[u64]) -> u64 {
    wait_core(ctx, args, None)
}

/// `SCE_KERNEL_ERROR_ETIMEDOUT` — the SCE-facing timeout code (SharpEmu's
/// `ORBIS_GEN2_ERROR_TIMED_OUT`, shadPS4's `ORBIS_KERNEL_ERROR_ETIMEDOUT`).
const SCE_KERNEL_ERROR_ETIMEDOUT: u64 = 0x8002_003C;

/// Orbis `scePthreadCondTimedwait(cond, mutex, SceKernelUseconds usec)`:
/// the third argument is a RELATIVE microsecond count passed by value, not a
/// pointer to an absolute timespec (cross-checked against SharpEmu +
/// shadPS4). On timeout this SCE entry point returns the SCE error code
/// `0x8002003C`, NOT POSIX `ETIMEDOUT` (60): a PS5 title's own pthread wrapper
/// recognizes the SCE code as "timed out" (and maps it to a benign cv_status);
/// handed the POSIX 60 it can't classify it, maps it to EINVAL, and its C++
/// std::condition_variable throws std::system_error("invalid argument") —
/// measured as the uncaught exception that killed Minecraft's worker threads.
fn hle_sce_cond_timedwait(ctx: &HleContext, args: &[u64]) -> u64 {
    let usec = args.get(2).copied().unwrap_or(0);
    let timeout = Some(std::time::Duration::from_micros(usec));
    match wait_core(ctx, args, timeout) {
        ETIMEDOUT => SCE_KERNEL_ERROR_ETIMEDOUT,
        other => other,
    }
}

/// POSIX `pthread_cond_timedwait` — returns errno directly (`ETIMEDOUT` = 60
/// on timeout), the errno convention libc's own POSIX wrappers expect.
/// POSIX `pthread_cond_timedwait(cond, mutex, const struct timespec *abstime)`.
///
/// The 3rd argument is a POINTER to an ABSOLUTE deadline — NOT a relative
/// microsecond count. That is the *SCE* spelling's ABI (see
/// [`hle_sce_cond_timedwait`]), and the two are entirely different calls that
/// happen to share a state machine.
///
/// Reading the pointer as a duration is catastrophic rather than merely wrong:
/// Minecraft's MAIN THREAD passes `abstime` at e.g. `0x1_0000_4be8_e8`, which as
/// microseconds is ~12.7 DAYS. The thread never woke to re-check its predicate,
/// every worker then idled waiting on it, and boot stalled on the black loading
/// screen — measured as ~30 threads parked at one host wait with the main
/// thread's call ring frozen on `pthread_cond_timedwait`.
fn hle_cond_timedwait(ctx: &HleContext, args: &[u64]) -> u64 {
    // The deadline is on the clock this cond was initialized with, so it can
    // only be read against that clock's own origin.
    let monotonic = condition(ctx, args.first().copied().unwrap_or(0))
        .is_some_and(|state| state.monotonic.load(std::sync::atomic::Ordering::Relaxed));
    let timeout = abstime_to_relative(ctx, args.get(2).copied().unwrap_or(0), monotonic);
    wait_core(ctx, args, timeout)
}

/// Convert a guest POSIX `struct timespec` (16 bytes: `time_t tv_sec`,
/// `long tv_nsec`) into "how long from now".
///
/// `monotonic` selects the origin the deadline is measured from, and it is not
/// optional: `CLOCK_MONOTONIC` counts from process start, `CLOCK_REALTIME` from
/// the Unix epoch, so the same `tv_sec` means two times ~1.78e9 seconds apart.
/// Reading a monotonic deadline against the epoch makes it permanently
/// "expired" — `wait_core` then returns `ETIMEDOUT` without ever waiting, and a
/// title that re-arms its wait spins as fast as the CPU allows instead of
/// sleeping. Measured on Minecraft: ~20,000 instant timeouts per second, main
/// thread never advancing past the boot loop.
///
/// `None` (a null `abstime`) means wait forever, matching POSIX. A deadline that
/// has genuinely passed yields `ZERO`, which `wait_core` reports as `ETIMEDOUT`
/// immediately — also what POSIX requires. An unreadable pointer is treated as
/// already-expired rather than as "forever": a bad deadline must not park a
/// thread for the rest of the run.
fn abstime_to_relative(
    ctx: &HleContext,
    abstime: u64,
    monotonic: bool,
) -> Option<std::time::Duration> {
    if abstime == 0 {
        return None;
    }
    let mut buf = [0u8; 16];
    if !ctx.mem.read(abstime, &mut buf) {
        tracing::warn!(
            abstime = format_args!("{abstime:#x}"),
            "pthread_cond_timedwait: abstime unreadable — treating as expired"
        );
        return Some(std::time::Duration::ZERO);
    }
    let tv_sec = i64::from_le_bytes(buf[0..8].try_into().unwrap_or_default());
    let tv_nsec = i64::from_le_bytes(buf[8..16].try_into().unwrap_or_default());
    let deadline = std::time::Duration::new(
        u64::try_from(tv_sec).unwrap_or(0),
        u32::try_from(tv_nsec.clamp(0, 999_999_999)).unwrap_or(0),
    );
    // Read "now" on the SAME clock the guest built the deadline on. The
    // monotonic arm must use `process_start`, the identical origin
    // `sceKernelClockGettime(CLOCK_MONOTONIC)` reports to the guest — if these
    // two ever disagree, every deadline is skewed by the difference.
    let now = if monotonic {
        crate::libkernel::process_start().elapsed()
    } else {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
    };
    Some(deadline.saturating_sub(now))
}

fn wait_core(ctx: &HleContext, args: &[u64], timeout: Option<std::time::Duration>) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    let mutex = args.get(1).copied().unwrap_or(0);
    if cond == 0 || mutex == 0 {
        return EINVAL;
    }
    let Some(state) = condition(ctx, cond) else {
        return EINVAL;
    };
    let mut generation = state.generation.lock();
    let observed = *generation;
    let unlock = crate::pthread_sync::mutex_unlock_for_cond(ctx, mutex);
    if unlock != OK {
        return unlock;
    }
    let started = std::time::Instant::now();
    let mut timed_out = false;
    let mut reported = false;
    while *generation == observed && !ctx.guest_threads.process_is_terminating() {
        // Forensic: a waiter that never sees its generation move is waiting on a
        // condition nobody signals. Name the cond + the waiter so it can be
        // matched against XPS5X_TRACE_COND's signal side.
        if !reported
            && started.elapsed() >= std::time::Duration::from_secs(3)
            && std::env::var_os("XPS5X_TRACE_COND").is_some()
        {
            reported = true;
            let name = ctx
                .kernel
                .thread_names
                .get(&ctx.guest_threads.current_thread())
                .map_or_else(|| "<unnamed>".to_owned(), |n| n.clone());
            tracing::warn!(
                cond = format_args!("{cond:#x}"),
                waiter = ctx.guest_threads.current_thread(),
                waiter_name = %name,
                generation = observed,
                "TRACE_COND: waiting >3s — this cond has not been signalled"
            );
        }
        let slice = timeout
            .map(|limit| limit.saturating_sub(started.elapsed()))
            .unwrap_or(std::time::Duration::from_millis(10))
            .min(std::time::Duration::from_millis(10));
        if timeout.is_some() && slice.is_zero() {
            timed_out = true;
            break;
        }
        let wait = state.changed.wait_for(&mut generation, slice);
        // POSIX explicitly permits spurious condition-variable wakeups. Treat
        // the bounded host wait as one so an orphaned/stale guest waiter can
        // re-check its own predicate, while still polling process termination
        // without pinning a host thread forever inside the VEH.
        if timeout.is_none() && wait.timed_out() {
            break;
        }
    }
    let woken = *generation != observed;
    drop(generation);
    note_wait_outcome(ctx, cond, woken);
    let relock = crate::pthread_sync::mutex_lock_for_cond(ctx, mutex);
    if relock != OK {
        return relock;
    }
    if timed_out { ETIMEDOUT } else { OK }
}

/// How long a waiter must go without its generation ever moving before it is
/// reported as starved.
const COND_STARVED_AFTER: std::time::Duration = std::time::Duration::from_secs(8);

/// Per-`(cond, thread)` start of the current starvation streak: when this waiter
/// last began waiting *without* having been genuinely woken since.
type CondStreaks = std::collections::HashMap<(u64, u64), (std::time::Instant, u64, bool)>;
static COND_STREAKS: std::sync::LazyLock<std::sync::Mutex<CondStreaks>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Forensic: report a condition variable a thread keeps re-waiting on but is
/// **never genuinely woken from**.
///
/// The single-call check in [`wait_core`] cannot see this: an infinite wait
/// deliberately returns after a 10 ms slice as a permitted spurious wakeup, so
/// no individual call ever looks long. A starved waiter is therefore a *streak*
/// of calls that never observe a generation change — which is exactly the shape
/// of a stalled engine task graph (every worker parked, producer waiting on a
/// completion nobody posts). Reported once per (cond, thread).
fn note_wait_outcome(ctx: &HleContext, cond: u64, woken: bool) {
    if std::env::var_os("XPS5X_TRACE_COND").is_none() {
        return;
    }
    let thread = ctx.guest_threads.current_thread();
    let mut streaks = COND_STREAKS.lock().unwrap_or_else(|p| p.into_inner());
    if woken {
        // A real wake ends the streak.
        streaks.remove(&(cond, thread));
        return;
    }
    let now = std::time::Instant::now();
    let entry = streaks.entry((cond, thread)).or_insert((now, 0, false));
    entry.1 += 1;
    if !entry.2 && now.duration_since(entry.0) >= COND_STARVED_AFTER {
        entry.2 = true;
        let name = ctx
            .kernel
            .thread_names
            .get(&thread)
            .map_or_else(|| "<unnamed>".to_owned(), |n| n.clone());
        tracing::warn!(
            cond = format_args!("{cond:#x}"),
            waiter = thread,
            waiter_name = %name,
            waits = entry.1,
            secs = now.duration_since(entry.0).as_secs(),
            // The guest instruction that called wait: `--dump-vaddr` turns this
            // into the code around the handshake, which is what identifies
            // *what* the thread is waiting to be told.
            caller = format_args!("{:#x}", ctx.caller_return_addr),
            "TRACE_COND: STARVED — re-waited this cond with no genuine wake"
        );
    }
}

/// `pthread_cond_signal(cond)` — wake one waiter.
///
/// Correct, not a stub: with one guest thread there are no waiters to wake, and
/// POSIX defines signalling a condition variable with no waiters as a no-op
/// returning success.
fn hle_cond_signal(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    if cond == 0 {
        return EINVAL;
    }
    let Some(state) = condition(ctx, cond) else {
        return EINVAL;
    };
    *state.generation.lock() += 1;
    state.changed.notify_one();
    trace_signal(ctx, cond, "signal");
    OK
}

/// Forensic: record which conds actually get signalled, so a waiter that never
/// wakes can be matched against the signal side (its cond simply never appears).
fn trace_signal(ctx: &HleContext, cond: u64, kind: &str) {
    if std::env::var_os("XPS5X_TRACE_COND").is_none() {
        return;
    }
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if SEEN.fetch_add(1, Ordering::Relaxed) < 4000 {
        let name = ctx
            .kernel
            .thread_names
            .get(&ctx.guest_threads.current_thread())
            .map_or_else(|| "<unnamed>".to_owned(), |n| n.clone());
        tracing::warn!(
            cond = format_args!("{cond:#x}"),
            by = ctx.guest_threads.current_thread(),
            by_name = %name,
            kind,
            "TRACE_COND: signalled"
        );
    }
}

/// `pthread_cond_broadcast(cond)` — wake all waiters. Same reasoning as
/// [`hle_cond_signal`].
fn hle_cond_broadcast(ctx: &HleContext, args: &[u64]) -> u64 {
    let cond = args.first().copied().unwrap_or(0);
    if cond == 0 {
        return EINVAL;
    }
    let Some(state) = condition(ctx, cond) else {
        return EINVAL;
    };
    *state.generation.lock() += 1;
    state.changed.notify_all();
    trace_signal(ctx, cond, "broadcast");
    OK
}

/// `pthread_condattr_init/destroy/setclock` — attribute objects carry nothing
/// that affects behaviour while there are no waiters.
/// `pthread_condattr_setclock(attr, clock_id)` — records the clock so
/// [`hle_cond_init`] can fix it on the condition it creates.
///
/// Recording it is not optional: the clock decides how that condition's
/// `pthread_cond_timedwait` deadlines are read (see [`abstime_to_relative`]),
/// and dropping it silently makes every monotonic deadline expire on arrival.
fn hle_condattr_setclock(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    let clock_id = args.get(1).copied().unwrap_or(0);
    if attr == 0 {
        return EINVAL;
    }
    debug!("pthread_condattr_setclock(attr={attr:#x}, clock_id={clock_id})");
    ctx.kernel.pthread_condattr_clocks.insert(attr, clock_id);
    OK
}

/// `pthread_condattr_init(attr)` — a fresh attr carries the POSIX default
/// clock.
///
/// This must clear any recorded clock rather than leave one behind: attrs are
/// short-lived stack objects, so a later attr at a recycled address would
/// otherwise inherit the previous one's `CLOCK_MONOTONIC` and mis-read its
/// realtime deadlines.
fn hle_condattr_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    if attr == 0 {
        return EINVAL;
    }
    ctx.kernel.pthread_condattr_clocks.remove(&attr);
    OK
}

/// `pthread_condattr_destroy(attr)` — drops the recorded clock with the attr.
fn hle_condattr_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    if attr == 0 {
        return EINVAL;
    }
    ctx.kernel.pthread_condattr_clocks.remove(&attr);
    OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestMemory, TestAllocator, TestMemory, test_ctx};

    fn fixture() -> (xps5x_kernel::OrbisKernel, TestMemory, TestAllocator) {
        (
            xps5x_kernel::OrbisKernel::new(),
            TestMemory::new(0x4000),
            TestAllocator::new(0x2000),
        )
    }

    #[test]
    fn signal_and_broadcast_succeed_with_no_waiters() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_cond_signal(&ctx, &[0x1000]), OK);
        assert_eq!(hle_cond_broadcast(&ctx, &[0x1000]), OK);
    }

    #[test]
    fn init_destroy_round_trip() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_cond_init(&ctx, &[0x2000, 0]), OK);
        assert_eq!(hle_cond_destroy(&ctx, &[0x2000]), OK);
    }

    #[test]
    fn null_cond_is_einval_not_a_silent_success() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_cond_init(&ctx, &[0]), EINVAL);
        assert_eq!(hle_cond_destroy(&ctx, &[0]), EINVAL);
        assert_eq!(hle_cond_signal(&ctx, &[0]), EINVAL);
        assert_eq!(hle_cond_broadcast(&ctx, &[0]), EINVAL);
        assert_eq!(hle_condattr_init(&ctx, &[0]), EINVAL);
        assert_eq!(hle_condattr_destroy(&ctx, &[0]), EINVAL);
        assert_eq!(
            hle_condattr_setclock(&ctx, &[0, crate::libkernel::CLOCK_MONOTONIC]),
            EINVAL
        );
    }

    /// Write a 16-byte guest `struct timespec` at `addr`.
    fn write_timespec(mem: &TestMemory, addr: u64, secs: u64, nanos: u32) {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&i64::try_from(secs).unwrap().to_le_bytes());
        buf[8..16].copy_from_slice(&i64::from(nanos).to_le_bytes());
        assert!(mem.write(addr, &buf));
    }

    /// The bug this whole clock-tracking path exists for.
    ///
    /// A `CLOCK_MONOTONIC` deadline a few seconds after process start must read
    /// as a few seconds *away*, not as expired. Read against the Unix epoch it
    /// lands ~1.78e9 seconds in the past, `wait_core` returns `ETIMEDOUT`
    /// without waiting, and a title that re-arms spins at ~20k timeouts/sec —
    /// measured on Minecraft's boot, which never advanced past it.
    #[test]
    fn monotonic_deadline_is_not_read_as_already_expired() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Exactly the shape measured from the title: a small tv_sec, which is
        // only sane on the monotonic clock.
        let deadline = crate::libkernel::process_start().elapsed().as_secs() + 30;
        write_timespec(&mem, 0x3000, deadline, 0);

        let as_monotonic = abstime_to_relative(&ctx, 0x3000, true).expect("non-null abstime");
        assert!(
            as_monotonic > std::time::Duration::from_secs(25),
            "a monotonic deadline ~30s out must be ~30s away, got {as_monotonic:?}"
        );

        let as_realtime = abstime_to_relative(&ctx, 0x3000, false).expect("non-null abstime");
        assert_eq!(
            as_realtime,
            std::time::Duration::ZERO,
            "the same bytes read against the epoch are the bug: instantly expired"
        );
    }

    /// `pthread_condattr_setclock(CLOCK_MONOTONIC)` must reach the cond that
    /// `pthread_cond_init` builds from that attr — dropping it is what made
    /// every deadline expire on arrival.
    #[test]
    fn condattr_setclock_monotonic_reaches_the_cond() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_condattr_init(&ctx, &[0x1500]), OK);
        assert_eq!(
            hle_condattr_setclock(&ctx, &[0x1500, crate::libkernel::CLOCK_MONOTONIC]),
            OK
        );
        assert_eq!(hle_cond_init(&ctx, &[0x2000, 0x1500]), OK);

        let state = condition(&ctx, 0x2000).expect("cond was just initialized");
        assert!(
            state.monotonic.load(std::sync::atomic::Ordering::Relaxed),
            "cond built from a CLOCK_MONOTONIC attr must wait on the monotonic clock"
        );
        // Destroying the attr must not disturb a cond already built from it.
        assert_eq!(hle_condattr_destroy(&ctx, &[0x1500]), OK);
        assert!(
            condition(&ctx, 0x2000)
                .expect("cond outlives its attr")
                .monotonic
                .load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    /// The POSIX default, and what a static `PTHREAD_COND_INITIALIZER` gets.
    #[test]
    fn cond_without_attr_defaults_to_realtime() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(hle_cond_init(&ctx, &[0x2000, 0]), OK);
        assert!(
            !condition(&ctx, 0x2000)
                .expect("cond was just initialized")
                .monotonic
                .load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    /// Attrs are short-lived stack objects, so addresses get recycled. A fresh
    /// `pthread_condattr_init` at a used address must not inherit the previous
    /// attr's clock, or an unrelated realtime cond silently waits on the wrong
    /// one.
    #[test]
    fn condattr_init_clears_a_recycled_addresss_clock() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        assert_eq!(
            hle_condattr_setclock(&ctx, &[0x1500, crate::libkernel::CLOCK_MONOTONIC]),
            OK
        );
        assert_eq!(hle_condattr_destroy(&ctx, &[0x1500]), OK);
        // Same address, fresh attr, no setclock: the POSIX default applies.
        assert_eq!(hle_condattr_init(&ctx, &[0x1500]), OK);
        assert_eq!(hle_cond_init(&ctx, &[0x2000, 0x1500]), OK);
        assert!(
            !condition(&ctx, 0x2000)
                .expect("cond was just initialized")
                .monotonic
                .load(std::sync::atomic::Ordering::Relaxed),
            "a recycled attr address must not leak CLOCK_MONOTONIC into the next cond"
        );
    }

    #[test]
    fn wait_is_registered_for_real_guest_workers() {
        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libScePosix", "pthread_cond_wait"));
        assert!(registry.is_implemented("libScePosix", "pthread_cond_timedwait"));
        assert!(registry.is_implemented("libScePosix", "pthread_cond_broadcast"));
        assert!(registry.is_implemented("libScePosix", "pthread_cond_signal"));
    }

    #[test]
    fn static_wait_materializes_an_opaque_handle_and_can_wake_spuriously() {
        let (kernel, mem, alloc) = fixture();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let cond = 0x100;
        let mutex = 0x200;
        assert_eq!(crate::pthread_sync::mutex_lock_for_cond(&ctx, mutex), OK);

        let started = std::time::Instant::now();
        assert_eq!(hle_cond_wait(&ctx, &[cond, mutex]), OK);
        assert!(started.elapsed() < std::time::Duration::from_millis(100));

        let mut bytes = [0u8; 8];
        assert!(mem.read(cond, &mut bytes));
        let handle = u64::from_le_bytes(bytes);
        assert_ne!(handle, 0);
        assert!(kernel.pthread_conds.contains_key(&cond));
        assert!(kernel.pthread_conds.contains_key(&handle));
        // POSIX requires the mutex to be reacquired before wait returns.
        assert_eq!(crate::pthread_sync::mutex_unlock_for_cond(&ctx, mutex), OK);
    }
}
