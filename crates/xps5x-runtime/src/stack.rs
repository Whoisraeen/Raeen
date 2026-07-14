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
    debug_assert_eq!(
        guest_rsp_top % 16,
        0,
        "guest_rsp_top must be 16-byte aligned (SysV call ABI)"
    );

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

/// Enter `entry` as a process's `_start` (design doc §3, wall #1 W1a): set
/// `rsp = process_rsp` and transfer control with a **`jmp`, not a `call`** —
/// no return address is pushed, so `entry`'s very first instruction sees
/// `[rsp]` exactly as `process_rsp` left it (`argc`, per
/// [`crate::process::build_process_stack`]'s layout), matching what a real
/// kernel-invoked `_start` observes.
///
/// # Safety
/// - `entry` must be a valid function pointer into mapped, executable guest
///   code (the same contract [`call_on_guest_stack`]'s `entry` parameter
///   carries), even though it is never actually *called* with SysV register
///   arguments — see the control-flow note below for why its Rust type is
///   reused as-is regardless.
/// - `process_rsp` must be a **16-byte-aligned** address pointing at the top
///   of a fully-built process stack inside a committed, writable memory
///   region — exactly [`crate::process::build_process_stack`]'s return
///   value. 16-alignment matters because, unlike `call_on_guest_stack`, no
///   `call` here pushes an 8-byte return address to shift it to `8 mod 16`:
///   `_start`'s ABI requires `rsp ≡ 0 mod 16` at its very first instruction,
///   so `process_rsp` must already be exactly that.
/// - Must be called with `dispatch::CALL_LOCK` held (as `execute_process`
///   does), for the same [`HOST_RSP_SLOT`] reentrancy reason as
///   `call_on_guest_stack`.
///
/// # Control flow (why a `jmp`, and why the "restore" line is unreachable)
///
/// A well-formed `_start` never returns to this function at all: it ends the
/// program via `exit`/`exit_group`/`_exit`, which `dispatch::veh_callback`
/// recognizes and answers with the *same* `RtlCaptureContext`-based longjmp
/// RT1a uses for a genuine fault (design doc §4) — that longjmp overwrites
/// the OS-delivered `CONTEXT` with the snapshot `dispatch::run` captured
/// *before* this function was ever called, so execution resumes directly
/// inside `run`, never passing back through this function's own asm at all.
/// A malformed `_start` that instead executes a plain `ret` doesn't return
/// here either: `ret` pops `[rsp]` (which holds `argc`, not a return
/// address this function ever pushed — there is none, by design, since this
/// is a `jmp`) and jumps to *that* value as if it were code, which reliably
/// faults on essentially any real `argc`/pointer value; that fault is then
/// recovered the same RT1a way. So the `mov rsp, [rip + slot]` line below,
/// immediately after the `jmp`, is unreachable in every path this runtime
/// actually exercises — it exists purely as defense in depth (design doc
/// §3: "the asm itself may simply also restore-on-return for the malformed
/// case"), so that even a control-flow path nobody has anticipated still
/// can't leave the *host* `rsp` pointing into guest memory. Because it's a
/// `jmp` (not a `call`), `entry`'s incoming SysV argument registers are
/// never set up — a real `_start` reads `argc`/`argv`/`envp` off the stack,
/// not out of registers, exactly like a kernel-invoked entry would.
///
/// `clobber_abi("sysv64")` is kept for the same reason as
/// `call_on_guest_stack`: the guest can do anything to the caller-saved
/// register/XMM set before it (if ever) reaches the unreachable tail below.
pub(crate) unsafe fn enter_guest_at_start(
    entry: unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64,
    process_rsp: u64,
) -> u64 {
    debug_assert_eq!(
        process_rsp % 16,
        0,
        "process_rsp must be 16-byte aligned (_start ABI)"
    );

    // Function pointers aren't directly usable as `asm!` register operands;
    // `entry as usize` is the address value the `jmp` below needs. This is a
    // plain integer cast of a function pointer, not a dereference.
    let entry_addr = entry as usize;
    let result: u64;

    // SAFETY: see this function's doc comment for the full ABI/control-flow
    // argument. Register usage:
    //  - `{entry_reg}`/`{guest_rsp}`: compiler-chosen scratch registers
    //    holding `entry`'s address and `process_rsp`; both are consumed by
    //    the `mov rsp, ...`/`jmp` sequence and need not survive it.
    //  - rax: only meaningful if the unreachable tail is somehow reached
    //    (see the doc comment); bound as `lateout` so the compiler doesn't
    //    assume it holds a valid value on the (never-taken) fallthrough.
    //  - No general-purpose register carries the host RSP across the `jmp`
    //    — saved to / restored from the RIP-relative static [`HOST_RSP_SLOT`]
    //    exactly like `call_on_guest_stack`.
    unsafe {
        asm!(
            // Save the host RSP before switching stacks, RIP-relative (no
            // GP register) — same reasoning as `call_on_guest_stack`.
            "mov qword ptr [rip + {slot}], rsp",
            // Switch to the process stack. `process_rsp` is 16-aligned (see
            // this function's `debug_assert!` and doc comment) and points at
            // `argc`, per `_start`'s ABI.
            "mov rsp, {guest_rsp}",
            // Transfer control WITHOUT pushing a return address: `entry`'s
            // first instruction sees `[rsp] == argc`, unperturbed.
            "jmp {entry_reg}",
            // Unreachable on every path this runtime exercises (see this
            // function's doc comment) — present only as defense in depth.
            "mov rsp, qword ptr [rip + {slot}]",
            slot = sym HOST_RSP_SLOT,
            guest_rsp = in(reg) process_rsp,
            entry_reg = in(reg) entry_addr,
            lateout("rax") result,
            clobber_abi("sysv64"),
        );
    }

    result
}
