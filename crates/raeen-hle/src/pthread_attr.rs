//! HLE libkernel pthread **thread-attribute** objects (`scePthreadAttr*`).
//!
//! A faithful Rust port of SharpEmu's thread-attribute state
//! (`PthreadAttrState`, GPL-2.0). A title configures a `pthread_attr_t` — stack
//! size, detach state, guard size, scheduling params — before
//! `scePthreadCreate`, then reads fields back. This is pure configuration data
//! with **no runtime dependency**, so it ports completely (unlike thread
//! creation itself, which needs the M1-E scheduler). State lives in the kernel
//! (`OrbisKernel::pthread_attrs`), keyed by both the guest `pthread_attr_t`
//! address and the opaque handle `Init` allocates and writes into `*attr`.

use crate::{HleContext, HleRegistry};
use raeen_kernel::PthreadAttr;
use tracing::{debug, info, warn};

const OK: u64 = 0;
const EINVAL: u64 = 22;

/// How many `scePthreadAttrGet` answers are logged at `info` before dropping to
/// `debug`.
///
/// A collector queries every thread it registers, so this floods at `info` if
/// unbounded; the first few are the ones that prove the reported stack is the
/// stack the kernel actually mapped, and that is exactly what a single
/// retail-title run needs to show without turning on `debug` for everything.
/// Same shape as `crate::exception`'s delivery counter.
const VERBOSE_ATTR_GETS: u64 = 6;
static ATTR_GETS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many `scePthreadAttrGet` calls found **no** registered stack for the
/// thread they were asked about, and therefore could not report a real one.
///
/// Diagnostics: a non-zero value means some guest thread learned its stack
/// extent from a configured default instead of the mapping, which is the defect
/// [`hle_attr_get`] exists to remove. Surfaced so a crash report can say so
/// rather than leaving it to a log grep.
static ATTR_GETS_WITHOUT_A_STACK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// How many `scePthreadAttrGet` calls could not report the thread's real stack.
#[must_use]
pub fn attr_get_without_stack_count() -> u64 {
    ATTR_GETS_WITHOUT_A_STACK.load(std::sync::atomic::Ordering::Relaxed)
}

/// Size of the opaque attribute object handed to the guest.
const ATTR_OBJECT_SIZE: u64 = 0x40;

/// Register the pthread thread-attribute HLE functions.
pub fn register(registry: &HleRegistry) {
    register_posix(registry);

    registry.register("libkernel", "scePthreadAttrInit", hle_attr_init);
    registry.register("libkernel", "scePthreadAttrDestroy", hle_attr_destroy);
    registry.register(
        "libkernel",
        "scePthreadAttrSetdetachstate",
        hle_set_detachstate,
    );
    registry.register(
        "libkernel",
        "scePthreadAttrGetdetachstate",
        hle_get_detachstate,
    );
    registry.register("libkernel", "scePthreadAttrSetstacksize", hle_set_stacksize);
    registry.register("libkernel", "scePthreadAttrGetstacksize", hle_get_stacksize);
    registry.register("libkernel", "scePthreadAttrSetstack", hle_set_stack);
    registry.register("libkernel", "scePthreadAttrGetstack", hle_get_stack);
    registry.register("libkernel", "scePthreadAttrGetstackaddr", hle_get_stackaddr);
    registry.register("libkernel", "scePthreadAttrGet", hle_attr_get);
    registry.register(
        "libkernel",
        "scePthreadAttrSetsolosched",
        hle_set_solo_sched,
    );
    registry.register("libkernel", "scePthreadAttrSetguardsize", hle_set_guardsize);
    registry.register("libkernel", "scePthreadAttrGetguardsize", hle_get_guardsize);
    registry.register(
        "libkernel",
        "scePthreadAttrSetschedpolicy",
        hle_set_schedpolicy,
    );
}

/// The POSIX spellings, under `libScePosix`.
///
/// Different NIDs from the `scePthread*` forms (a NID hashes the name alone),
/// and these are the ones a real title imports. Same signatures and the same
/// POSIX return convention (0 / positive errno), so the implementations alias
/// directly.
fn register_posix(registry: &HleRegistry) {
    registry.register("libScePosix", "pthread_attr_init", hle_attr_init);
    registry.register("libScePosix", "pthread_attr_destroy", hle_attr_destroy);
    registry.register(
        "libScePosix",
        "pthread_attr_setdetachstate",
        hle_set_detachstate,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_getdetachstate",
        hle_get_detachstate,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_setstacksize",
        hle_set_stacksize,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_getstacksize",
        hle_get_stacksize,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_setguardsize",
        hle_set_guardsize,
    );
    registry.register(
        "libScePosix",
        "pthread_attr_getguardsize",
        hle_get_guardsize,
    );
    // The `_np` spelling is how libkernel names the non-POSIX-standard
    // solo-scheduler knob for middleware compiled against plain headers —
    // a distinct NID (a NID hashes the name alone) over the same body.
    registry.register(
        "libScePosix",
        "pthread_attr_setsolosched_np",
        hle_set_solo_sched,
    );
    // `pthread_attr_get_np(pthread_t, pthread_attr_t *)` is the FreeBSD name for
    // what `scePthreadAttrGet` does, and shadPS4 exports the same body under
    // both `libScePosix` and `libkernel` (NID `Ucsu-OK+els`) as well as under
    // the SCE spelling. Registered here so a title that reaches for the POSIX
    // name gets the real live-thread answer instead of an unresolved NID —
    // Blasphemous II imports only the SCE spelling, but this is the *other* way
    // a guest can learn a thread's stack extent and it must not be a hole.
    registry.register("libScePosix", "pthread_attr_get_np", hle_attr_get);
    registry.register("libkernel", "pthread_attr_get_np", hle_attr_get);
}

/// Resolve the attr-state key for a guest `pthread_attr_t` address: the address
/// if registered, else the handle it points at.
fn resolve_key(ctx: &HleContext, attr_addr: u64) -> Option<u64> {
    if ctx.kernel.pthread_attrs.contains_key(&attr_addr) {
        return Some(attr_addr);
    }
    let mut buf = [0u8; 8];
    if ctx.mem.read(attr_addr, &mut buf) {
        let handle = u64::from_le_bytes(buf);
        if handle != 0 && ctx.kernel.pthread_attrs.contains_key(&handle) {
            return Some(handle);
        }
    }
    None
}

/// `scePthreadAttrInit(attr)`: allocate an opaque object, write its handle into
/// `*attr`, and register default attribute state.
fn hle_attr_init(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr_addr = args.first().copied().unwrap_or(0);
    if attr_addr == 0 {
        return EINVAL;
    }
    let Some(handle) = ctx.alloc.alloc(ATTR_OBJECT_SIZE, 0x10) else {
        return EINVAL;
    };
    if !ctx.mem.write(attr_addr, &handle.to_le_bytes()) {
        return EINVAL;
    }
    let state = PthreadAttr::default();
    ctx.kernel.pthread_attrs.insert(attr_addr, state);
    ctx.kernel.pthread_attrs.insert(handle, state);
    debug!("scePthreadAttrInit(attr={attr_addr:#x}) -> handle {handle:#x}");
    OK
}

/// `scePthreadAttrDestroy(attr)`.
fn hle_attr_destroy(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr_addr = args.first().copied().unwrap_or(0);
    if attr_addr == 0 {
        return EINVAL;
    }
    let Some(key) = resolve_key(ctx, attr_addr) else {
        return EINVAL;
    };
    ctx.kernel.pthread_attrs.remove(&key);
    if key != attr_addr {
        ctx.kernel.pthread_attrs.remove(&attr_addr);
    }
    let _ = ctx.mem.write(attr_addr, &0u64.to_le_bytes());
    OK
}

/// Apply `update` to the attr state for `attr_addr` (creating a default if the
/// title never called `Init` — a static `pthread_attr_t`).
fn with_attr(ctx: &HleContext, attr_addr: u64, update: impl FnOnce(&mut PthreadAttr)) -> u64 {
    if attr_addr == 0 {
        return EINVAL;
    }
    let key = resolve_key(ctx, attr_addr).unwrap_or_else(|| {
        ctx.kernel
            .pthread_attrs
            .insert(attr_addr, PthreadAttr::default());
        attr_addr
    });
    let mut entry = ctx.kernel.pthread_attrs.get_mut(&key).unwrap();
    update(&mut entry);
    OK
}

/// Read the attr state for `attr_addr`, or the default if unset.
fn read_attr(ctx: &HleContext, attr_addr: u64) -> Option<PthreadAttr> {
    let key = resolve_key(ctx, attr_addr)?;
    ctx.kernel.pthread_attrs.get(&key).map(|e| *e)
}

fn hle_set_detachstate(ctx: &HleContext, args: &[u64]) -> u64 {
    let state = args.get(1).copied().unwrap_or(0) as i32;
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.detach_state = state
    })
}

fn hle_set_stacksize(ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.get(1).copied().unwrap_or(0);
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.stack_size = size
    })
}

/// `scePthreadAttrSetstack(attr, stackAddr, stackSize)` — the title supplies its
/// own stack region (base + size). We do not honor a guest-chosen stack BASE
/// (the scheduler owns stack allocation), but the call MUST succeed: ASTRO.BOT
/// asserts (engine `Thread.cpp:120`) and then faults if it returns an error.
/// Record the size like `scePthreadAttrSetstacksize` and return OK.
fn hle_set_stack(ctx: &HleContext, args: &[u64]) -> u64 {
    let address = args.get(1).copied().unwrap_or(0);
    let size = args.get(2).copied().unwrap_or(0);
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.stack_address = address;
        a.stack_size = size;
    })
}

fn hle_set_guardsize(ctx: &HleContext, args: &[u64]) -> u64 {
    let size = args.get(1).copied().unwrap_or(0);
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.guard_size = size
    })
}

fn hle_set_schedpolicy(ctx: &HleContext, args: &[u64]) -> u64 {
    let policy = args.get(1).copied().unwrap_or(0) as i32;
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.sched_policy = policy
    })
}

/// `scePthreadAttrGetdetachstate(attr, int *out)`.
fn hle_get_detachstate(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    let Some(state) = read_attr(ctx, attr) else {
        return EINVAL;
    };
    if out == 0 || !ctx.mem.write(out, &state.detach_state.to_le_bytes()) {
        return EINVAL;
    }
    OK
}

/// `scePthreadAttrGetstacksize(attr, size_t *out)`.
fn hle_get_stacksize(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    let Some(state) = read_attr(ctx, attr) else {
        return EINVAL;
    };
    if out == 0 || !ctx.mem.write(out, &state.stack_size.to_le_bytes()) {
        return EINVAL;
    }
    OK
}

/// `scePthreadAttrGetguardsize(attr, size_t *out)`.
fn hle_get_guardsize(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    let Some(state) = read_attr(ctx, attr) else {
        return EINVAL;
    };
    if out == 0 || !ctx.mem.write(out, &state.guard_size.to_le_bytes()) {
        return EINVAL;
    }
    OK
}

/// `scePthreadAttrGetstack(attr, void **stackAddrOut, size_t *stackSizeOut)`:
/// read back the stack configuration recorded by `scePthreadAttrSetstack` /
/// `scePthreadAttrSetstacksize`.
///
/// The attribute object preserves both requested fields. Thread creation may
/// still choose a scheduler-owned stack; that execution policy does not change
/// the ABI requirement that attribute getters return what the guest configured.
fn hle_get_stack(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    let addr_out = args.get(1).copied().unwrap_or(0);
    let size_out = args.get(2).copied().unwrap_or(0);
    if addr_out == 0 || size_out == 0 {
        return EINVAL;
    }
    let Some(state) = read_attr(ctx, attr) else {
        return EINVAL;
    };
    if !ctx.mem.write(addr_out, &state.stack_address.to_le_bytes())
        || !ctx.mem.write(size_out, &state.stack_size.to_le_bytes())
    {
        return EINVAL;
    }
    OK
}

/// `scePthreadAttrGetstackaddr(attr, void **stackAddrOut)`: the single-field
/// form used by Unity's pthread wrapper.
fn hle_get_stackaddr(ctx: &HleContext, args: &[u64]) -> u64 {
    let attr = args.first().copied().unwrap_or(0);
    let out = args.get(1).copied().unwrap_or(0);
    let Some(state) = read_attr(ctx, attr) else {
        return EINVAL;
    };
    if out == 0 || !ctx.mem.write(out, &state.stack_address.to_le_bytes()) {
        return EINVAL;
    }
    OK
}

/// `scePthreadAttrGet(ScePthread thread, ScePthreadAttr *attr)`: fill `*attr`
/// from the **live** thread named by `thread` — *not* from whatever the guest
/// had configured on that attribute object.
///
/// This is FreeBSD's `pthread_attr_get_np` under an SCE name; shadPS4 maps the
/// very NID this title imports (`x1X76arYMxU`) straight onto
/// `posix_pthread_attr_get_np`, which copies the running thread's own
/// `PthreadAttr` — stack base and size included — into the destination
/// (`core/libraries/kernel/threads/pthread_attr.cpp`, GPL-2.0; behaviour
/// studied, re-implemented in Rust).
///
/// # Why the previous `hle_ok_stub` was not a harmless stub
///
/// It returned `SCE_OK` and wrote **nothing**, so a caller that did the standard
///
/// ```c
/// scePthreadAttrInit(&attr);
/// scePthreadAttrGet(scePthreadSelf(), &attr);      /* reported success */
/// scePthreadAttrGetstackaddr(&attr, &base);        /* -> 0, the DEFAULT */
/// scePthreadAttrGetstacksize(&attr, &size);        /* -> 1 MiB, the DEFAULT */
/// bounds->stack_top = (char *)base + size;         /* -> 0x100000, garbage */
/// ```
///
/// walked away believing its own stack lived at a low address it never mapped.
/// Measured on Blasphemous II (Unity/IL2CPP, PPSA13580), whose Boehm collector
/// does exactly this: `Il2CppUserAssemblies.prx+0x2B8F10`
/// (`GC_push_all_stacks`) loads `lo = p->stop_info.stack_ptr` from `[p+0x18]`
/// and `hi = p->stack_end` from `[p+0x100]`, then either
///
/// * aborts at `+0x2B908E` with `"GC_push_all_stacks: sp not set!"` when the
///   bound is zero, or
/// * pushes the bogus range and faults reading it in the mark loop at
///   `+0x2B24AB` (`mov r12,[rax+0x10]`) at a low address with no high dword.
///
/// Both were observed from the same title on consecutive runs; they are one
/// cause with two shapes. The title imports `scePthreadAttrGet`,
/// `scePthreadAttrGetstackaddr` and `scePthreadAttrGetstacksize` and no
/// `Setstack*` form, so the configured base is *always* the default 0 and this
/// call is the only place the real base can come from.
///
/// # What is reported, and what is not
///
/// Reported from live state: the thread's real mapped stack
/// ([`raeen_kernel::OrbisKernel::guest_stack_of`] — `[base, top)`, so
/// `base + size` is exactly the top of the stack the thread is running on), its
/// scheduling priority, and its scheduling policy.
///
/// **Not** reported: `detach_state` and `guard_size`, which Raeen tracks in the
/// runtime's own thread table rather than the kernel, so there is no live value
/// to copy; those fields keep whatever the destination attr already held.
///
/// An unknown or already-reaped thread keeps the previous behaviour — `SCE_OK`
/// with the attr untouched — rather than the ABI's `ESRCH`, because a collector
/// that treats a non-zero return as fatal would abort on it. It is logged once
/// per thread at `warn` instead of passing silently.
fn hle_attr_get(ctx: &HleContext, args: &[u64]) -> u64 {
    let thread = args.first().copied().unwrap_or(0);
    let attr_addr = args.get(1).copied().unwrap_or(0);
    if thread == 0 || attr_addr == 0 {
        return EINVAL;
    }
    let stack = ctx.kernel.guest_stack_of(thread);
    let priority = ctx.kernel.thread_priorities.get(&thread).map(|e| *e);
    let policy = ctx.kernel.thread_sched_policies.get(&thread).map(|e| *e);
    let result = with_attr(ctx, attr_addr, |a| {
        if let Some((base, top)) = stack {
            a.stack_address = base;
            a.stack_size = top - base;
        }
        if let Some(priority) = priority {
            a.sched_priority = priority;
        }
        if let Some(policy) = policy {
            a.sched_policy = policy;
        }
    });
    match stack {
        Some((base, top)) => {
            let seen = ATTR_GETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let size = top - base;
            if seen < VERBOSE_ATTR_GETS {
                // At `info` so ONE retail run shows, without enabling `debug`
                // everywhere, that the reported bounds are the mapped ones and
                // carry their high dword. `base + size` is the value a Boehm
                // collector scans up to.
                info!(
                    thread,
                    stack_base = format_args!("{base:#x}"),
                    stack_size = format_args!("{size:#x}"),
                    stack_top = format_args!("{top:#x}"),
                    "scePthreadAttrGet: reporting the thread's real mapped stack"
                );
            } else {
                debug!(
                    "scePthreadAttrGet(thread={thread:#x}, attr={attr_addr:#x}) -> stack \
                     [{base:#x}, {top:#x})"
                );
            }
        }
        None if result == OK => {
            ATTR_GETS_WITHOUT_A_STACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            warn!(
                "scePthreadAttrGet(thread={thread:#x}, attr={attr_addr:#x}): no stack is \
                 registered for that guest thread, so the attr keeps its configured stack \
                 base/size. A caller computing `base + size` as the thread's stack top will get a \
                 bogus address"
            );
        }
        None => {}
    }
    result
}

/// `scePthreadAttrSetsolosched(attr, int solo)` /
/// `pthread_attr_setsolosched_np(attr, solo)`: the SCE-specific "solo
/// scheduler" attr flag — the title asks that the thread be scheduled on its
/// own context rather than sharing one. Raeen maps guest threads onto host
/// threads, which are already independently scheduled, so there is no host
/// action to take; the flag is pure attr bookkeeping and reads back exactly
/// as set (same storage class as `sched_policy` above).
fn hle_set_solo_sched(ctx: &HleContext, args: &[u64]) -> u64 {
    let solo = args.get(1).copied().unwrap_or(0) != 0;
    with_attr(ctx, args.first().copied().unwrap_or(0), |a| {
        a.solo_sched = solo
    })
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
        let mem = crate::TestMemory::new(0x4000);
        let alloc = crate::TestAllocator::new(0x1000);
        (kernel, mem, alloc)
    }

    #[test]
    fn init_writes_a_handle_and_defaults() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x100;
        assert_eq!(hle_attr_init(&ctx, &[attr]), OK);
        let mut buf = [0u8; 8];
        assert!(mem.read(attr, &mut buf));
        assert!(u64::from_le_bytes(buf) != 0, "init writes a handle");
        // Default stack size (1 MiB) is readable back.
        let out = 0x200;
        assert_eq!(hle_get_stacksize(&ctx, &[attr, out]), OK);
        let mut sz = [0u8; 8];
        assert!(mem.read(out, &mut sz));
        assert_eq!(u64::from_le_bytes(sz), 0x10_0000);
        assert_eq!(hle_attr_init(&ctx, &[0]), EINVAL);
    }

    #[test]
    fn set_then_get_round_trips_stacksize_and_detachstate() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x300;
        hle_attr_init(&ctx, &[attr]);
        // Stack size round-trips.
        assert_eq!(hle_set_stacksize(&ctx, &[attr, 0x20_0000]), OK);
        let out = 0x400;
        assert_eq!(hle_get_stacksize(&ctx, &[attr, out]), OK);
        let mut sz = [0u8; 8];
        assert!(mem.read(out, &mut sz));
        assert_eq!(u64::from_le_bytes(sz), 0x20_0000);
        // Detach state round-trips (1 = detached).
        assert_eq!(hle_set_detachstate(&ctx, &[attr, 1]), OK);
        let dout = 0x408;
        assert_eq!(hle_get_detachstate(&ctx, &[attr, dout]), OK);
        let mut ds = [0u8; 4];
        assert!(mem.read(dout, &mut ds));
        assert_eq!(i32::from_le_bytes(ds), 1);
    }

    #[test]
    fn get_on_unknown_attr_errors_and_destroy_clears() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // Get before init → EINVAL (no state, and *attr reads 0).
        assert_eq!(hle_get_stacksize(&ctx, &[0x500, 0x600]), EINVAL);
        let attr = 0x700;
        hle_attr_init(&ctx, &[attr]);
        assert_eq!(hle_attr_destroy(&ctx, &[attr]), OK);
        assert!(!kernel.pthread_attrs.contains_key(&attr));
        let mut buf = [0u8; 8];
        assert!(mem.read(attr, &mut buf));
        assert_eq!(u64::from_le_bytes(buf), 0, "destroy zeroes the handle");
    }

    #[test]
    fn getstack_family_round_trips_the_recorded_address_and_size() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x300;
        hle_attr_init(&ctx, &[attr]);
        // Attribute getters round-trip both configured fields.
        assert_eq!(hle_set_stack(&ctx, &[attr, 0xDEAD_0000, 0x20_0000]), OK);
        let addr_out = 0x400;
        let size_out = 0x408;
        assert_eq!(hle_get_stack(&ctx, &[attr, addr_out, size_out]), OK);
        let mut a = [0u8; 8];
        let mut s = [0u8; 8];
        assert!(mem.read(addr_out, &mut a));
        assert!(mem.read(size_out, &mut s));
        assert_eq!(u64::from_le_bytes(a), 0xDEAD_0000);
        assert_eq!(u64::from_le_bytes(s), 0x20_0000);
        assert_eq!(hle_get_stackaddr(&ctx, &[attr, addr_out]), OK);
        assert!(mem.read(addr_out, &mut a));
        assert_eq!(u64::from_le_bytes(a), 0xDEAD_0000);
        // NULL out-pointers and unknown attrs are EINVAL.
        assert_eq!(hle_get_stack(&ctx, &[attr, 0, size_out]), EINVAL);
        assert_eq!(hle_get_stack(&ctx, &[0x900, addr_out, size_out]), EINVAL);

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libkernel", "scePthreadAttrGetstack"));
        assert!(registry.is_implemented("libkernel", "scePthreadAttrGetstackaddr"));
    }

    /// **The Blasphemous II collector fault, as a round-trip assertion.**
    ///
    /// `scePthreadAttrGet` is how a title learns another thread's stack extent,
    /// and a Boehm collector turns the answer into `base + size` — the top of
    /// the range it scans. What the kernel MAPPED must therefore be exactly what
    /// the guest reads back, at full 64-bit width, through both the two-field
    /// and the single-field getter.
    ///
    /// The old registration was `hle_ok_stub`: success, nothing written, so the
    /// getters returned the default base of **0** and the collector computed a
    /// low garbage top. The `0xDEAD_0000` poison below is the *configured* base;
    /// a fix that only zero-filled, or that kept round-tripping the configured
    /// value, leaves it in place and fails here.
    #[test]
    fn attr_get_reports_the_real_mapped_stack_of_the_named_thread() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        // A full 64-bit guest stack: the arena base is above 2^32, so a
        // truncating write anywhere on this path loses the high dword.
        const BASE: u64 = 0x1000_4000_8A20;
        const TOP: u64 = BASE + 0x12_0000;
        kernel.guest_thread_stacks.insert(15, (BASE, TOP));
        kernel.thread_priorities.insert(15, 256);
        kernel.thread_sched_policies.insert(15, 2);

        let attr = 0x300;
        assert_eq!(hle_attr_init(&ctx, &[attr]), OK);
        // Poison with a *configured* base/size, so "kept what the guest set"
        // cannot pass as "reported what the kernel mapped".
        assert_eq!(hle_set_stack(&ctx, &[attr, 0xDEAD_0000, 0x1000]), OK);

        assert_eq!(hle_attr_get(&ctx, &[15, attr]), OK);

        let addr_out = 0x400;
        let size_out = 0x408;
        assert_eq!(hle_get_stack(&ctx, &[attr, addr_out, size_out]), OK);
        let mut a = [0u8; 8];
        let mut s = [0u8; 8];
        assert!(mem.read(addr_out, &mut a));
        assert!(mem.read(size_out, &mut s));
        let base = u64::from_le_bytes(a);
        let size = u64::from_le_bytes(s);
        assert_eq!(
            base, BASE,
            "the base must be the stack the kernel really mapped, at full width"
        );
        assert_eq!(size, TOP - BASE);
        assert_eq!(
            base + size,
            TOP,
            "base + size is what a collector scans up to; it must be the real stack top"
        );

        // The single-field getter Unity's wrapper uses must agree.
        assert!(mem.write(addr_out, &0u64.to_le_bytes()));
        assert_eq!(hle_get_stackaddr(&ctx, &[attr, addr_out]), OK);
        assert!(mem.read(addr_out, &mut a));
        assert_eq!(u64::from_le_bytes(a), BASE);

        // Live scheduling state is copied too.
        let state = kernel.pthread_attrs.get(&attr).unwrap();
        assert_eq!(state.sched_priority, 256);
        assert_eq!(state.sched_policy, 2);
    }

    /// An unknown thread must not become a fatal return code: a collector that
    /// treats non-zero as fatal aborts on it. It stays `SCE_OK` with the attr
    /// untouched (and a `warn` names it), and a null thread or attr is `EINVAL`.
    #[test]
    fn attr_get_validates_its_arguments_and_survives_an_unknown_thread() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x300;
        assert_eq!(hle_attr_init(&ctx, &[attr]), OK);
        assert_eq!(hle_set_stack(&ctx, &[attr, 0xDEAD_0000, 0x1000]), OK);

        assert_eq!(hle_attr_get(&ctx, &[0, attr]), EINVAL);
        assert_eq!(hle_attr_get(&ctx, &[15, 0]), EINVAL);
        let unreported = attr_get_without_stack_count();
        assert_eq!(
            hle_attr_get(&ctx, &[99, attr]),
            OK,
            "an unknown thread must not hand the guest a code it treats as fatal"
        );
        assert_eq!(
            attr_get_without_stack_count(),
            unreported + 1,
            "a query that could not report a real stack must be counted, so a crash report can \
             say the guest computed its scan bounds from a default"
        );
        let state = kernel.pthread_attrs.get(&attr).unwrap();
        assert_eq!(
            (state.stack_address, state.stack_size),
            (0xDEAD_0000, 0x1000),
            "with no live stack to report, the configured values must be left alone"
        );

        // Every spelling a guest can reach a live thread's stack extent through
        // must resolve to the real implementation, not a success-and-write-nothing
        // stub.
        let registry = HleRegistry::new();
        for (lib, name) in [
            ("libkernel", "scePthreadAttrGet"),
            ("libkernel", "pthread_attr_get_np"),
            ("libScePosix", "pthread_attr_get_np"),
        ] {
            assert!(
                registry.is_implemented(lib, name),
                "{lib}::{name} must be registered"
            );
        }
    }

    /// `scePthreadAttrGet` describes a *live thread*; the plain setters still
    /// describe a *configuration*. Fixing the former must not turn the latter
    /// into a report of some thread's stack.
    #[test]
    fn attr_get_does_not_change_what_the_plain_setters_round_trip() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        kernel
            .guest_thread_stacks
            .insert(1, (0x1000_8000_0000, 0x1000_A000_0000));
        let attr = 0x300;
        assert_eq!(hle_attr_init(&ctx, &[attr]), OK);
        assert_eq!(hle_set_stack(&ctx, &[attr, 0xDEAD_0000, 0x20_0000]), OK);
        let addr_out = 0x400;
        let size_out = 0x408;
        assert_eq!(hle_get_stack(&ctx, &[attr, addr_out, size_out]), OK);
        let mut a = [0u8; 8];
        let mut s = [0u8; 8];
        assert!(mem.read(addr_out, &mut a));
        assert!(mem.read(size_out, &mut s));
        assert_eq!(u64::from_le_bytes(a), 0xDEAD_0000);
        assert_eq!(u64::from_le_bytes(s), 0x20_0000);
    }

    #[test]
    fn setsolosched_records_the_flag_on_the_attr() {
        let (kernel, mem, alloc) = ctx_env();
        let ctx = test_ctx(&kernel, &mem, &alloc);
        let attr = 0x300;
        hle_attr_init(&ctx, &[attr]);
        // Default off; set → stored; cleared → stored.
        assert!(!kernel.pthread_attrs.get(&attr).unwrap().solo_sched);
        assert_eq!(hle_set_solo_sched(&ctx, &[attr, 1]), OK);
        assert!(kernel.pthread_attrs.get(&attr).unwrap().solo_sched);
        assert_eq!(hle_set_solo_sched(&ctx, &[attr, 0]), OK);
        assert!(!kernel.pthread_attrs.get(&attr).unwrap().solo_sched);
        // A null attr address is EINVAL.
        assert_eq!(hle_set_solo_sched(&ctx, &[0, 1]), EINVAL);

        let registry = HleRegistry::new();
        assert!(registry.is_implemented("libkernel", "scePthreadAttrSetsolosched"));
        assert!(registry.is_implemented("libScePosix", "pthread_attr_setsolosched_np"));
    }
}
