//! The guest-thread-stack registry is the **only** place Raeen knows where a
//! guest stack is: stacks are arena-owned, so they appear in no VMM region and
//! nothing about their address distinguishes them from an ordinary heap object.
//!
//! Everything that answers "where is this thread's stack?" for the guest reads
//! it — `scePthreadAttrGet` (FreeBSD's `pthread_attr_get_np`, which a Boehm
//! collector turns into `base + size` = the top of the range it scans),
//! `sceKernelIsStack`, `sceKernelVirtualQuery`, and the HLE out-buffer guard's
//! caller-local test. These pin the lookup semantics at full 64-bit width.

use raeen_kernel::OrbisKernel;

/// Real guest stacks live above 2^32 (the arena base is ~17 TiB), so every
/// bound here is chosen so a truncating read or write loses the high dword and
/// fails visibly.
const BASE: u64 = 0x1000_4000_8A20;
const TOP: u64 = BASE + 0x12_0000;

#[test]
fn a_registered_stack_is_reported_by_thread_and_by_address_at_full_width() {
    let kernel = OrbisKernel::new();
    kernel.guest_thread_stacks.insert(15, (BASE, TOP));

    assert_eq!(kernel.guest_stack_of(15), Some((BASE, TOP)));
    assert_eq!(kernel.guest_stack_containing(BASE), Some((BASE, TOP)));
    assert_eq!(
        kernel.guest_stack_containing(BASE + 0x1234),
        Some((BASE, TOP))
    );
    // `base + size` must be exactly the top — this is the value a collector
    // scans up to, so an off-by-a-page here is an out-of-bounds scan.
    let (base, top) = kernel.guest_stack_of(15).unwrap();
    assert_eq!(base + (top - base), TOP);
}

#[test]
fn the_range_is_half_open_and_unknown_threads_report_nothing() {
    let kernel = OrbisKernel::new();
    kernel.guest_thread_stacks.insert(15, (BASE, TOP));

    // Half-open: the base is inside, the top is not.
    assert_eq!(kernel.guest_stack_containing(TOP), None);
    assert_eq!(kernel.guest_stack_containing(BASE - 1), None);
    // Truncating the high dword must NOT match — the whole point of the width.
    assert_eq!(kernel.guest_stack_containing(BASE & 0xFFFF_FFFF), None);

    assert_eq!(kernel.guest_stack_of(1), None, "unregistered thread");
    assert_eq!(kernel.guest_stack_of(0), None);
}

#[test]
fn an_empty_or_inverted_registration_is_refused_rather_than_reported() {
    let kernel = OrbisKernel::new();
    // A degenerate entry must never be handed to the guest as a stack: a
    // collector would compute a zero-length or negative-length scan range.
    kernel.guest_thread_stacks.insert(2, (TOP, BASE));
    kernel.guest_thread_stacks.insert(3, (BASE, BASE));
    assert_eq!(kernel.guest_stack_of(2), None);
    assert_eq!(kernel.guest_stack_of(3), None);
    assert_eq!(kernel.guest_stack_containing(BASE), None);
}

#[test]
fn several_live_threads_are_each_matched_to_their_own_stack() {
    let kernel = OrbisKernel::new();
    let stacks = [
        (2u64, (0x1000_4000_0000u64, 0x1000_4012_0000u64)),
        (3, (0x1000_4012_0000, 0x1000_4024_0000)),
        (1, (0x1000_8000_0000, 0x1000_A000_0000)),
    ];
    for (thread, bounds) in stacks {
        kernel.guest_thread_stacks.insert(thread, bounds);
    }
    for (thread, (base, top)) in stacks {
        assert_eq!(kernel.guest_stack_of(thread), Some((base, top)));
        assert_eq!(kernel.guest_stack_containing(base), Some((base, top)));
        assert_eq!(kernel.guest_stack_containing(top - 1), Some((base, top)));
    }
    // Adjacent stacks must not bleed into one another.
    assert_eq!(
        kernel.guest_stack_containing(0x1000_4012_0000),
        Some((0x1000_4012_0000, 0x1000_4024_0000)),
        "a shared boundary belongs to the region that starts there"
    );
}
