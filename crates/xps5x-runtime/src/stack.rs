//! [`call_on_guest_stack`]: the RSP-switch trampoline that runs a guest
//! `sysv64` entry point on the arena's dedicated guest stack region instead
//! of the host thread's own stack (design doc §2, RT2c-a). This is the most
//! delicate `unsafe` in the runtime — see the function doc comment below for
//! the full ABI/robustness argument, and `dispatch.rs`'s module doc comment
//! for why this does not disturb the VEH trampoline path or the RT1a
//! genuine-fault recovery.

use core::arch::asm;

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
///   a caller bug; debug builds catch it via `debug_assert!` below rather
///   than silently mis-aligning the guest's stack.
/// - `host_rsp_save` must point at 8 bytes of writable memory, valid for the
///   entire duration of this call, that guest code cannot reach through any
///   pointer it might construct (i.e. it must lie outside the arena) — the
///   caller ([`crate::dispatch::run`]) passes `ActiveContext::host_rsp`'s
///   address, a host-stack-local field, never exposed to guest memory.
///
/// # Mechanism (why a memory save, not a register save)
///
/// The current (host) RSP is saved to `*host_rsp_save` **before** RSP is
/// overwritten, addressed via a register holding the *pointer* (`r15`) —
/// not `rsp`/`rbp`-relative addressing, since RSP is exactly what is about
/// to change. After `entry` returns, RSP is restored by reading back through
/// that same pointer.
///
/// This is deliberately a **memory** save of the RSP *value*, not a
/// register save. The tempting alternative — stash the raw host RSP value
/// itself in a single callee-saved register (say `r15`) across the call —
/// relies on `entry` honoring the SysV callee-saved convention for that
/// register. `entry` is guest code produced by the LM1 pipeline, not
/// necessarily a compiler-emitted function with a textbook
/// prologue/epilogue; a hand-written or otherwise non-conforming leaf
/// routine can clobber *any* general-purpose register it likes without
/// saving/restoring it, since nothing calls into it expecting SysV
/// callee-saved discipline to hold across a single "leaf" execution. If the
/// raw RSP value were trusted to survive in such a register, a guest that
/// merely *uses* that register (not even maliciously — just carelessly)
/// would silently corrupt the host's stack pointer on return, which is
/// exactly the kind of fragility this milestone must not ship (design doc
/// §7: "must be robust against a guest that does not perfectly honor the
/// SysV ABI").
///
/// Saving the value to memory instead removes that dependency entirely: the
/// saved RSP survives no matter what `entry` does to the general-purpose
/// registers, *except* the one register (`r15`) that carries the memory
/// slot's address across the call, and except a wild write through
/// `*host_rsp_save` itself. Both of those residual cases are already
/// governed by this function's safety contract and by `entry`'s existing
/// trust boundary (design doc §6/§7): only LM1-pipeline (clean-room
/// re-implemented) images are ever executed here, never arbitrary or
/// adversarial machine code, so a guest that goes out of its way to clobber
/// `r15` (a register nothing in the ABI obligates it to touch at all,
/// SysV-callee-saved or not) or to scribble over host memory it was never
/// given a pointer to is already outside what this runtime promises to
/// survive gracefully — the same as a guest that corrupts its own return
/// address or jumps to an arbitrary instruction. It is categorically more
/// robust than a register-only save, which a perfectly ordinary
/// (non-adversarial, just not compiler-conventional) guest routine could
/// defeat by accident.
///
/// `clobber_abi("sysv64")` tells the compiler that the internal `call`
/// clobbers the full SysV caller-saved register/XMM set; combined with the
/// explicit `in("...")` operands binding `args` directly to the SysV integer
/// argument registers (rdi/rsi/rdx/rcx/r8/r9 — no `mov`-into-place
/// instructions are needed in the body at all), this trampoline clobbers
/// exactly what an ordinary `sysv64` call through `entry` would, so no
/// Rust-visible state is corrupted beyond what a ordinary call already
/// implies.
pub(crate) unsafe fn call_on_guest_stack(
    entry: unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64,
    args: [u64; 6],
    guest_rsp_top: u64,
    host_rsp_save: *mut u64,
) -> u64 {
    debug_assert_eq!(guest_rsp_top % 16, 0, "guest_rsp_top must be 16-byte aligned (SysV call ABI)");
    debug_assert!(!host_rsp_save.is_null(), "host_rsp_save must be a valid memory slot");

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
    //  - r15: holds `host_rsp_save` (a pointer, not the RSP value itself),
    //    used both before and after the `call` to save/restore the host RSP
    //    through memory.
    //  - `{entry_reg}`/`{guest_rsp}`: compiler-chosen scratch registers
    //    (guaranteed disjoint from rdi/rsi/rdx/rcx/r8/r9/r15, all of which
    //    are explicitly reserved above) holding `entry`'s address and the
    //    target RSP value; neither needs to survive past the `call`.
    //  - rax: the guest's return value, read out as `result` after the call.
    unsafe {
        asm!(
            // Save the host RSP to memory *before* switching stacks —
            // addressed via r15 (a pointer), never rsp/rbp-relative, since
            // rsp is what's about to change.
            "mov [r15], rsp",
            // Switch to the guest stack. `guest_rsp_top` is 16-aligned (see
            // this function's `debug_assert!` and doc comment); the `call`
            // immediately below pushes an 8-byte return address, so `entry`
            // observes rsp ≡ 8 mod 16 on entry, exactly as SysV requires.
            "mov rsp, {guest_rsp}",
            "call {entry_reg}",
            // Restore the host RSP from the same memory slot, still
            // addressed via r15 (see this function's doc comment for why a
            // guest that clobbers r15 is out of scope / UB, same as the
            // rest of `entry`'s trust boundary).
            "mov rsp, [r15]",
            guest_rsp = in(reg) guest_rsp_top,
            entry_reg = in(reg) entry_addr,
            in("rdi") args[0],
            in("rsi") args[1],
            in("rdx") args[2],
            in("rcx") args[3],
            in("r8") args[4],
            in("r9") args[5],
            in("r15") host_rsp_save,
            lateout("rax") result,
            clobber_abi("sysv64"),
        );
    }

    result
}
