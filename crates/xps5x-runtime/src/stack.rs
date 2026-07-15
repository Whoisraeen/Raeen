//! Native entry into guest code on a caller-prepared guest stack.
//!
//! Guest execution never returns through this Rust frame. The caller places
//! either a guarded return-trampoline address or the process parameter block
//! at `guest_rsp`, and [`enter_guest`] switches RSP then jumps to the entry.
//! A normal function return therefore faults at the guarded trampoline and
//! `dispatch` restores its previously captured host `CONTEXT`. This avoids
//! keeping any host stack pointer in process-global or guest-visible state.

use core::arch::asm;

/// Enter mapped guest code with the six SysV integer argument registers.
///
/// # Safety
///
/// - `entry` must name executable guest code in the live `GuestArena`.
/// - `guest_rsp` must point into a committed writable guest stack and have
///   the alignment required by the selected entry ABI.
/// - The caller must have armed `dispatch`'s recovery context and VEH before
///   calling this function. Control leaves the guest only through that
///   recovery path, so this function intentionally never returns.
pub(crate) unsafe fn enter_guest(entry: u64, guest_rsp: u64, args: [u64; 6]) -> ! {
    // SAFETY: the caller guarantees that `entry` and `guest_rsp` are live
    // guest addresses and that dispatch recovery is armed. No host register
    // or host-stack address needs to survive the jump: the return-trampoline,
    // exit, and fault paths all restore the complete captured host CONTEXT.
    unsafe {
        asm!(
            "mov rsp, {guest_rsp}",
            "jmp {entry}",
            guest_rsp = in(reg) guest_rsp,
            entry = in(reg) entry,
            in("rdi") args[0],
            in("rsi") args[1],
            in("rdx") args[2],
            in("rcx") args[3],
            in("r8") args[4],
            in("r9") args[5],
            options(noreturn),
        )
    }
}
