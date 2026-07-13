//! FSGSBASE-based guest TLS (RT2c-b, design doc §3): the raw `RDFSBASE`/
//! `WRFSBASE` instruction wrappers [`dispatch::run`](crate::dispatch::run)
//! uses to point the FS segment base at a guest TCB around a guest call. The
//! TCB itself is set up by [`crate::arena::GuestArena::setup_main_tcb`].
//!
//! # Why raw instruction bytes, not the `rdfsbase`/`wrfsbase` mnemonics
//!
//! `core::arch::asm!` only assembles the `rdfsbase`/`wrfsbase` mnemonics if
//! the compilation unit's target features include `fsgsbase` (LLVM gates the
//! mnemonic on it). Enabling that crate-wide would need a
//! `-C target-feature=+fsgsbase` rustflag (via `.cargo/config.toml`), which
//! this project's constraints forbid (no crate-wide rustflags/codegen
//! changes — the RT2c-b spike used exactly that flag in a throwaway scratch
//! project, precisely so this crate doesn't have to). Emitting the fixed
//! instruction encoding directly via `.byte` sidesteps the codegen-flag
//! requirement entirely: the CPU either has FSGSBASE and executes the bytes
//! (gated by [`fsgsbase_available`]), or it doesn't and this code never
//! reaches them — independent of what LLVM believes the compile-time target
//! supports.
//!
//! # Why FSGSBASE at all, and why this is safe on Windows
//!
//! See the RT2c-b spike report
//! (`docs/superpowers/specs/2026-07-13-xps5x-guest-stack-tls-design.md` §3)
//! and the design doc: 64-bit Windows never uses the FS segment (it uses GS
//! for the TEB) and leaves the FS base at 0, and user-mode
//! `RDFSBASE`/`WRFSBASE` are permitted (no `#UD`) once `CR4.FSGSBASE` is set
//! by the OS (Windows 10 1709+, confirmed set on this build via a spike). The
//! spike also confirmed the FS base survives the VEH + `RtlCaptureContext`
//! fault-recovery round trip `dispatch::run` relies on: the x64 `CONTEXT`
//! structure has no FS-base field, so nothing in that mechanism has a slot
//! to reset it through. Neither function here ever touches GS.

use core::arch::asm;
use core::arch::x86_64::__cpuid_count;
use std::sync::OnceLock;

/// Cached result of the `CPUID.(EAX=7,ECX=0):EBX[bit 0]` FSGSBASE
/// availability probe (the CPU's capability can't change while the process
/// runs, so this is computed at most once).
static FSGSBASE_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Whether this CPU reports FSGSBASE support (`CPUID.(7,0):EBX[0]`).
/// [`dispatch::run`](crate::dispatch::run) and [`crate::execute_linked`] gate
/// every `read_fsbase`/`write_fsbase` call, and TCB setup, on this — on a CPU
/// without FSGSBASE, those functions are simply never called, so the guest
/// runs without TLS (honest degradation) rather than ever executing an
/// instruction that could raise `#UD`.
///
/// There is no user-mode way to directly probe whether the OS has actually
/// set `CR4.FSGSBASE` (the bit this capability check doesn't cover); the
/// RT2c-b spike confirmed empirically that this CPUID bit being set
/// correlated with `RDFSBASE`/`WRFSBASE` actually being permitted on the
/// Windows build tested.
pub(crate) fn fsgsbase_available() -> bool {
    *FSGSBASE_AVAILABLE.get_or_init(|| {
        // `__cpuid_count` is a safe fn on this target/rustc (the `CPUID`
        // instruction is unconditionally available on every x86-64 CPU this
        // crate can run on -- Windows x86-64 only -- no target-feature
        // precondition, unlike the leaf-7 bit it queries). Leaf 7, sub-leaf
        // 0 is a standard, always-queryable "Extended Features" leaf.
        let regs = __cpuid_count(7, 0);
        (regs.ebx & 1) != 0
    })
}

/// Read the current thread's FS segment base via `RDFSBASE`.
///
/// # Safety
/// The caller must have confirmed [`fsgsbase_available`] returns `true`
/// before calling this — on a CPU without FSGSBASE, `RDFSBASE` is not a
/// valid instruction and executing it raises `#UD`. Reads only the FS base;
/// never touches GS or any other segment/register beyond `rax`.
pub(crate) unsafe fn read_fsbase() -> u64 {
    let value: u64;
    // SAFETY: per this function's contract, the caller has already confirmed
    // FSGSBASE is available. `F3 48 0F AE C0` is `RDFSBASE RAX`: `F3` is the
    // mandatory instruction-selecting prefix, `48` is REX.W (64-bit
    // destination), `0F AE` is the opcode, and `C0` is ModRM
    // (mod=11, reg=/0, rm=000=RAX) -- exactly the encoding given in the
    // RT2c-b task brief. The result is bound to `rax` via `out("rax")`; the
    // instruction reads no memory and (per the Intel SDM) does not affect
    // `EFLAGS`.
    unsafe {
        asm!(
            ".byte 0xf3, 0x48, 0x0f, 0xae, 0xc0", // rdfsbase rax
            out("rax") value,
            options(nomem, nostack, preserves_flags),
        );
    }
    value
}

/// Write `value` into the current thread's FS segment base via `WRFSBASE`.
///
/// # Safety
/// The caller must have confirmed [`fsgsbase_available`] returns `true`
/// before calling this — on a CPU without FSGSBASE, `WRFSBASE` is not a
/// valid instruction and executing it raises `#UD`. Never touches GS or any
/// other segment/register beyond `rax`. `value` becomes visible to any
/// subsequent `fs:`-prefixed memory access on this thread (including guest
/// code, if called with a guest TCB address) until this is called again.
pub(crate) unsafe fn write_fsbase(value: u64) {
    // SAFETY: per this function's contract, the caller has already confirmed
    // FSGSBASE is available. `F3 48 0F AE D0` is `WRFSBASE RAX`: same
    // prefix/REX.W reasoning as `read_fsbase`, with ModRM `D0`
    // (mod=11, reg=/2, rm=000=RAX) selecting the write form -- exactly the
    // encoding given in the RT2c-b task brief. `value` is bound to `rax` via
    // `in("rax")`; the instruction itself reads/writes no memory (it only
    // updates the hidden FS-base descriptor-cache state) and does not affect
    // `EFLAGS`.
    unsafe {
        asm!(
            ".byte 0xf3, 0x48, 0x0f, 0xae, 0xd0", // wrfsbase rax
            in("rax") value,
            options(nomem, nostack, preserves_flags),
        );
    }
}
