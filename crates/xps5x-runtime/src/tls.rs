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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// **Platform reality, pinned (M1-E spec §6 I3 spike — it FAILED):**
    /// a user-set FS base does **NOT** survive a Windows context switch. It is
    /// reset to `0` by the first preemption.
    ///
    /// The module doc above only ever established that the FS base survives the
    /// VEH + `RtlCaptureContext` round trip (the x64 `CONTEXT` has no FS-base
    /// field, so *that* mechanism has no slot to reset it through). That is a
    /// different question from preemption, and the original RT2c-b spike never
    /// tested preemption at all.
    ///
    /// Measured here (probe output): `write_fsbase(MAGIC)` reads back correctly
    /// immediately and across a `yield_now`, but comes back `0x0` after a
    /// `sleep` **and** after a pure user-mode busy-wait with no syscall
    /// whatsoever. So it is a genuine timer-interrupt context switch — not a
    /// kernel transition — that clears it: Windows saves/restores the FS base
    /// from its own notion of the thread's base (`0` for native x64 threads),
    /// discarding any user-mode `WRFSBASE`. (Linux preserves it; Windows does
    /// not. `CR4.FSGSBASE` being set only makes the *instruction* legal — it
    /// does not make the value survive scheduling.)
    ///
    /// **Consequence:** a raw `WRFSBASE` guest TCB is only valid until the next
    /// quantum (~15 ms). Any guest that touches TLS or the `fs:0x28` canary
    /// after being preempted reads a near-null address, which the VEH sees as a
    /// genuine wild fault → the title reports `Faulted`. Guest TLS therefore
    /// cannot rely on `WRFSBASE` alone; it needs the FS base re-armed after
    /// each preemption (see `dispatch`'s fault path).
    ///
    /// This test pins the platform behaviour so the assumption is never quietly
    /// re-introduced. If it ever starts failing, Windows changed and the
    /// re-arm machinery may be simplifiable.
    #[test]
    fn fsbase_does_not_survive_preemption_on_windows() {
        if !fsgsbase_available() {
            // Honest skip: this CPU/OS never executes the RDFSBASE/WRFSBASE
            // path at all, so there is nothing to pin.
            return;
        }

        // Canonical (high bits clear) and obviously not a real base.
        const MAGIC: u64 = 0x0000_1234_5678_9AB0;

        // SAFETY: `fsgsbase_available()` is true. 64-bit Windows never uses
        // the FS segment (it uses GS for the TEB) and leaves the FS base at 0,
        // so temporarily repointing *this* thread's FS base cannot disturb the
        // host runtime; it is restored below before any assertion can unwind.
        let orig = unsafe { read_fsbase() };

        // Confine this thread + one spinner to a single CPU (two runnable
        // threads, one core → the scheduler must preempt this one), leaving
        // every other core free so the parallel test suite is not starved.
        // Stops+joins the spinner and restores affinity on drop, so an
        // assertion panic below cannot leak the busy-loop.
        struct Spinners {
            stop: Arc<AtomicBool>,
            handle: Option<std::thread::JoinHandle<()>>,
            prev_affinity: usize,
        }
        impl Drop for Spinners {
            fn drop(&mut self) {
                use windows_sys::Win32::System::Threading::{
                    GetCurrentThread, SetThreadAffinityMask,
                };
                self.stop.store(true, Ordering::Relaxed);
                if let Some(h) = self.handle.take() {
                    let _ = h.join();
                }
                if self.prev_affinity != 0 {
                    // SAFETY: restore the caller's original affinity; Drop runs
                    // on the same thread that constructed this.
                    unsafe { SetThreadAffinityMask(GetCurrentThread(), self.prev_affinity) };
                }
            }
        }
        let stop = Arc::new(AtomicBool::new(false));
        let _spinners = {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
            // SAFETY: pin the current thread to CPU 0; returns the previous
            // mask (0 on failure), restored on Drop.
            let prev_affinity = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1) };
            let handle = {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    // SAFETY: pin this spinner to CPU 0 too.
                    unsafe { SetThreadAffinityMask(GetCurrentThread(), 1) };
                    while !stop.load(Ordering::Relaxed) {
                        std::hint::spin_loop();
                    }
                })
            };
            Spinners {
                stop: Arc::clone(&stop),
                handle: Some(handle),
                prev_affinity,
            }
        };

        // SAFETY: as above — available, and restored before returning.
        unsafe { write_fsbase(MAGIC) };
        // Diagnostic: isolate exactly where (if anywhere) the base is lost.
        // SAFETY: as above.
        let immediately = unsafe { read_fsbase() };
        std::thread::yield_now();
        // SAFETY: as above.
        let after_yield = unsafe { read_fsbase() };
        std::thread::sleep(std::time::Duration::from_millis(1));
        // SAFETY: as above.
        let after_sleep = unsafe { read_fsbase() };
        // Distinguish "a kernel transition (syscall) resets it" from "a bare
        // timer-interrupt preemption resets it". Real guest code makes no
        // syscalls, so only the latter would threaten a running guest — and it
        // is the latter.
        // SAFETY: as above.
        unsafe { write_fsbase(MAGIC) };
        let spin_start = std::time::Instant::now();
        // Pure user-mode busy-wait, no syscall: long enough (with the machine
        // saturated by spinners) that a timer-interrupt preemption is certain.
        while spin_start.elapsed() < std::time::Duration::from_millis(120) {
            std::hint::spin_loop();
        }
        // SAFETY: as above.
        let after_pure_spin = unsafe { read_fsbase() };

        // Restore *before* asserting so a failure can't leave this test thread
        // with a bogus FS base. `_spinners` stops+joins on drop at end of scope.
        // SAFETY: as above.
        unsafe { write_fsbase(orig) };

        // The write itself works, and survives while we stay scheduled...
        assert_eq!(
            immediately, MAGIC,
            "WRFSBASE did not take effect at all (probe: yield={after_yield:#x})"
        );
        // ...but a real context switch discards it. Both a syscall-driven
        // deschedule and a bare timer-interrupt preemption clear it to 0,
        // which is what makes this a scheduling property rather than a
        // kernel-transition artifact.
        assert_eq!(
            after_sleep, 0,
            "FS base unexpectedly SURVIVED a sleep/deschedule — Windows behaviour \
             changed; the dispatch re-arm machinery may be simplifiable"
        );
        assert_eq!(
            after_pure_spin, 0,
            "FS base unexpectedly SURVIVED a pure user-mode preemption — Windows \
             behaviour changed; the dispatch re-arm machinery may be simplifiable"
        );
    }
}
