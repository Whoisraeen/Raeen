//! [`call_on_guest_stack`]: the RSP-switch trampoline that runs a guest
//! `sysv64` entry point on the arena's dedicated guest stack region instead
//! of the host thread's own stack (design doc §2, RT2c-a). This is the most
//! delicate `unsafe` in the runtime — see the function doc comment below for
//! the full ABI/robustness argument, and `dispatch.rs`'s module doc comment
//! for why this does not disturb the VEH trampoline path or the RT1a
//! genuine-fault recovery.

use core::arch::asm;
use core::cell::UnsafeCell;

/// The single host-RSP save slot used by [`call_on_guest_stack`], addressed
/// **RIP-relative** from the asm below. Storing the host stack pointer here
/// (rather than in a register or a register-addressed slot) is what makes the
/// restore after the guest returns depend on **no** general-purpose register
/// surviving the guest `call`: `[rip + slot]` needs only RIP, which the guest
/// cannot influence.
///
/// `UnsafeCell` (a mutable static) is required so the slot lives in writable
/// memory — a plain immutable `static` would be placed in read-only `.rodata`
/// and the asm's store into it would fault.
///
/// A single process-wide slot is sound because all guest execution is
/// serialized by `dispatch::CALL_LOCK` (design doc §2/§9): exactly one
/// `call_on_guest_stack` is ever in flight, so the slot is never accessed
/// concurrently despite the `Sync` impl, and never needs to nest.
struct HostRspSlot(
    // Only ever accessed by the asm below via its `sym` operand (RIP-relative),
    // never through this field in Rust, so dead-code analysis can't see the use.
    #[allow(dead_code)] UnsafeCell<u64>,
);

// SAFETY: access is serialized process-wide by `dispatch::CALL_LOCK` — only
// one guest execution (hence one write/read of this slot) happens at a time.
unsafe impl Sync for HostRspSlot {}

static HOST_RSP_SLOT: HostRspSlot = HostRspSlot(UnsafeCell::new(0));

/// Call `entry` with `args` (the SysV integer argument registers, in order)
/// on the guest stack whose top (highest address) is `guest_rsp_top`,
/// returning `entry`'s `RAX`.
///
/// # Safety
/// - `entry` must be a valid `sysv64` function pointer, safe to call with
///   `args` right now — the same contract [`crate::dispatch::run`]'s `entry`
///   parameter carries (a mapped, executable guest function from the LM1
///   pipeline).
/// - `guest_rsp_top` must be a **16-byte-aligned** address that is the top
///   of a committed, writable memory region big enough for `entry` (and
///   anything it calls) to use as a downward-growing stack —
///   [`crate::arena::GuestArena::stack_top`]. 16-alignment matters because
///   the `call` instruction below pushes an 8-byte return address, so
///   `entry` observes `rsp ≡ 8 mod 16` on entry — exactly what the SysV ABI
///   requires a called function to see. A non-16-aligned `guest_rsp_top` is
///   a caller bug; debug builds catch it via `debug_assert!` below.
/// - Must be called with `dispatch::CALL_LOCK` held (as `execute_linked`
///   does), so the single [`HOST_RSP_SLOT`] is never used re-entrantly.
///
/// # Mechanism (robust host-RSP save/restore)
///
/// The host RSP is saved to the process-static [`HOST_RSP_SLOT`] **before**
/// RSP is switched, and restored from it **after** the guest returns — both
/// via RIP-relative addressing (`[rip + slot]`):
///
/// ```text
/// mov [rip + slot], rsp   ; save host RSP (no GP register involved)
/// mov rsp, guest_rsp_top  ; switch to the guest stack
/// call entry              ; run guest code (may clobber any register)
/// mov rsp, [rip + slot]   ; restore host RSP (no GP register involved)
/// ```
///
/// The crucial property: the **restore depends on no general-purpose register
/// surviving the `call`.** A tempting simpler design stashes the host RSP
/// (or a pointer to it) in a callee-saved register such as `r15` across the
/// call — but that relies on `entry` honoring the SysV callee-saved
/// convention for that register. `entry` is guest code from the LM1 pipeline,
/// not necessarily a compiler-emitted function with a textbook epilogue; a
/// hand-written or non-conforming routine can return normally yet leave `r15`
/// (or any callee-saved register) altered, and the restore would then load a
/// guest-controlled value into the **host** RSP — silently corrupting the host
/// thread on an otherwise "successful" call. Addressing the save slot
/// RIP-relative removes that dependency entirely: the host RSP is recovered no
/// matter what `entry` did to the general-purpose registers, meeting the
/// design doc §7 requirement to be robust against a guest that does not
/// perfectly honor the SysV ABI. (A guest can still corrupt its own execution
/// — its return address, or, since it runs natively on an identity map, host
/// memory it was never handed a pointer to — but that is the same residual
/// trust boundary the whole runtime already operates under, design doc §6/§7,
/// not a fragility of this trampoline.)
///
/// `clobber_abi("sysv64")` tells the compiler the internal `call` clobbers the
/// full SysV caller-saved register/XMM set; combined with the explicit
/// `in("...")` operands binding `args` directly to the SysV integer argument
/// registers, this trampoline clobbers exactly what an ordinary `sysv64` call
/// through `entry` would, so no Rust-visible state is corrupted beyond what an
/// ordinary call already implies.
pub(crate) unsafe fn call_on_guest_stack(
    entry: unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64,
    args: [u64; 6],
    guest_rsp_top: u64,
) -> u64 {
    debug_assert_eq!(guest_rsp_top % 16, 0, "guest_rsp_top must be 16-byte aligned (SysV call ABI)");

    // Function pointers aren't directly usable as `asm!` register operands;
    // `entry as usize` is the address value the `call` below needs. This is
    // a plain integer cast of a function pointer, not a dereference.
    let entry_addr = entry as usize;
    let result: u64;

    // SAFETY: see this function's doc comment for the full ABI/robustness
    // contract. Register usage in this block:
    //  - rdi/rsi/rdx/rcx/r8/r9: bound directly to `args[0..6]` by the
    //    operand list below (SysV integer argument registers) — `entry`
    //    reads its arguments from exactly these registers, per its
    //    `extern "sysv64"` signature.
    //  - `{entry_reg}`/`{guest_rsp}`: compiler-chosen scratch registers
    //    (disjoint from the reserved rdi/rsi/rdx/rcx/r8/r9) holding `entry`'s
    //    address and the target RSP value; both are consumed before/at the
    //    `call` and need not survive it.
    //  - rax: the guest's return value, read out as `result` after the call.
    //  - No general-purpose register carries the host RSP (or a pointer to
    //    it) across the `call` — it is saved to / restored from the static
    //    `HOST_RSP_SLOT` via RIP-relative addressing, so the restore is
    //    correct regardless of how `entry` treats the registers.
    unsafe {
        asm!(
            // Save the host RSP to the static slot *before* switching stacks,
            // addressed RIP-relative (no GP register), since rsp is about to
            // change and no callee-saved register can be trusted to survive
            // the guest call.
            "mov qword ptr [rip + {slot}], rsp",
            // Switch to the guest stack. `guest_rsp_top` is 16-aligned (see
            // this function's `debug_assert!` and doc comment); the `call`
            // immediately below pushes an 8-byte return address, so `entry`
            // observes rsp ≡ 8 mod 16 on entry, exactly as SysV requires.
            "mov rsp, {guest_rsp}",
            "call {entry_reg}",
            // Restore the host RSP from the same static slot, again
            // RIP-relative — this is the key line: it does not depend on any
            // general-purpose register having survived the guest call.
            "mov rsp, qword ptr [rip + {slot}]",
            slot = sym HOST_RSP_SLOT,
            guest_rsp = in(reg) guest_rsp_top,
            entry_reg = in(reg) entry_addr,
            in("rdi") args[0],
            in("rsi") args[1],
            in("rdx") args[2],
            in("rcx") args[3],
            in("r8") args[4],
            in("r9") args[5],
            lateout("rax") result,
            clobber_abi("sysv64"),
        );
    }

    result
}
