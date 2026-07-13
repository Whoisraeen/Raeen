//! # XPS5X Runtime — RT0
//!
//! Executes a [`xps5x_firmware::LinkedModule`] natively: maps its `image`
//! into host memory, guards the HLE trampoline region the LM1 linker
//! addressed relocation slots against, arms a Vectored Exception Handler,
//! and calls the guest entry function directly on the host thread as a
//! foreign `extern "sysv64"` function pointer. See the design doc
//! (`docs/superpowers/specs/2026-07-13-xps5x-runtime-design.md`) for the
//! full mechanism, ABI boundary, and safety/trust-boundary discussion.
//!
//! RT0 is Windows-first (design doc §7/§9): [`execute_linked`]'s mechanism
//! (`mem`/`trampoline`/`dispatch`) is Win32-API-specific and gated
//! `#[cfg(target_os = "windows")]`, but the public function signature is
//! platform-independent so a POSIX `sigaction`/`SIGSEGV` backend can slot in
//! at a later milestone without callers changing.
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "windows")]
mod dispatch;
#[cfg(target_os = "windows")]
mod mem;
#[cfg(target_os = "windows")]
mod trampoline;

use thiserror::Error;
use xps5x_firmware::LinkedModule;
use xps5x_hle::HleRegistry;

/// Errors [`execute_linked`] can return (design doc §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RuntimeError {
    /// Guest memory (the image mapping or the trampoline guard region)
    /// could not be established, or `entry_offset` did not point within the
    /// mapped image.
    #[error("failed to map guest memory")]
    MapFailed,
    /// A guest `call` hit a trampoline slot with no corresponding
    /// [`xps5x_firmware::HleTrampoline`] entry — surfaced, not silently
    /// ignored (design doc §5). The faulting address is reported for
    /// diagnostics.
    #[error("call to unresolved HLE trampoline at {0:#x}")]
    UnresolvedTrampoline(u64),
    /// A genuine guest fault (an access violation) outside the trampoline
    /// guard region — e.g. a wild pointer dereference in guest code, not an
    /// HLE call. Recovered rather than crashing the process (RT1a): the VEH
    /// restores a pre-call register snapshot taken via `RtlCaptureContext`
    /// (see `dispatch.rs`'s module doc comment for the exact mechanism).
    /// `addr` is the faulting instruction's `Rip`.
    #[error("guest fault at {addr:#x}")]
    Faulted { addr: u64 },
    /// More than 6 integer/pointer arguments were requested — RT0 only
    /// marshals the SysV integer argument registers (design doc §3).
    #[error("more than 6 arguments requested (RT0 marshals only the SysV integer registers)")]
    TooManyArgs,
}

/// Integer/pointer arguments RT0 marshals: SysV RDI, RSI, RDX, RCX, R8, R9
/// (design doc §3).
const MAX_ARGS: usize = 6;

/// Run `module`'s function at `entry_offset` (an offset into
/// `module.image`) natively, passing `args` (up to 6 integer/pointer
/// values, SysV) and servicing every HLE trampoline call it makes through
/// `hle`. Returns the guest function's `RAX` on success. See the design doc
/// §2 for the full trap-and-dispatch mechanism and §5 for this signature.
#[cfg(target_os = "windows")]
pub fn execute_linked(
    module: &LinkedModule,
    hle: &HleRegistry,
    entry_offset: u64,
    args: &[u64],
) -> Result<u64, RuntimeError> {
    if args.len() > MAX_ARGS {
        return Err(RuntimeError::TooManyArgs);
    }
    let mut padded = [0u64; MAX_ARGS];
    padded[..args.len()].copy_from_slice(args);

    // RT0 supports exactly one active native guest execution at a time
    // (design doc §4/§6/§9); held for this entire function, not just the
    // guest call inside `dispatch::run` below — see `dispatch::CALL_LOCK`'s
    // doc comment for why the trampoline guard reservation just below also
    // needs this.
    let _call_lock = dispatch::call_lock();

    let image = mem::MappedImage::map(&module.image)?;
    let entry_ptr = image.entry_ptr(entry_offset)?;
    let guard = trampoline::TrampolineGuard::reserve(module.hle_trampolines.len())?;

    // SAFETY: `entry_ptr` is a host address inside `image`'s
    // `PAGE_EXECUTE_READWRITE` mapping, at the caller-specified
    // `entry_offset` into `module.image` — code the LM1 pipeline produced,
    // and the only thing this crate ever executes (design doc §6).
    // Transmuting a data pointer to an `extern "sysv64"` function pointer
    // matches the guest ABI (design doc §3); actually calling it happens
    // inside `dispatch::run`, guarded by the VEH armed there.
    let entry: unsafe extern "sysv64" fn(u64, u64, u64, u64, u64, u64) -> u64 =
        unsafe { core::mem::transmute::<*const u8, _>(entry_ptr) };

    // SAFETY: `entry` is exactly the function pointer `dispatch::run`'s
    // safety contract requires (a valid `sysv64` pointer into the
    // `MappedImage` we just built); `module.hle_trampolines` and `hle`
    // outlive this call (both borrowed for its entire duration); `guard`'s
    // region covers every address `module.hle_trampolines` can resolve (it
    // was sized from that same table, immediately above).
    unsafe { dispatch::run(entry, padded, &module.hle_trampolines, hle, &guard) }
}

/// RT0 is Windows-first; a POSIX backend lands at a later milestone without
/// changing this signature (design doc §7/§9).
#[cfg(not(target_os = "windows"))]
pub fn execute_linked(
    _module: &LinkedModule,
    _hle: &HleRegistry,
    _entry_offset: u64,
    _args: &[u64],
) -> Result<u64, RuntimeError> {
    Err(RuntimeError::MapFailed)
}
