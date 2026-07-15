//! The trampoline guard region: a `PAGE_NOACCESS` reservation over
//! `[HLE_TRAMPOLINE_BASE, ..)` so a guest `call [import_slot]` through an
//! HLE-resolved relocation slot faults deterministically (design doc §2/§4),
//! plus the faulting-address → [`HleTrampoline`] lookup the VEH uses to
//! service that fault.

use core::ffi::c_void;

use windows_sys::Win32::System::Memory::{
    MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree,
};

use xps5x_firmware::{HLE_TRAMPOLINE_BASE, HleTrampoline};

use crate::RuntimeError;

/// x86-64 Windows' page size (see `mem.rs`'s equivalent constant).
const PAGE_SIZE: u64 = 4096;

/// A `PAGE_NOACCESS` reservation covering `trampoline_count` HLE trampoline
/// slots (8 bytes each) starting at [`HLE_TRAMPOLINE_BASE`], plus an invalid
/// diagnostic sentinel at index `count` and a controlled return trampoline at
/// index `count + 1`. Reserved, not committed — unmapped and reserved memory
/// already faults on access exactly like committed `PAGE_NOACCESS` memory,
/// so no commit is needed for this guard's purpose. Freed via `VirtualFree`
/// on `Drop`.
pub(crate) struct TrampolineGuard {
    base: u64,
    /// Actual reserved length in bytes (page-rounded; always `>=` the
    /// logical `trampoline_count * 8 + 16` span).
    len: u64,
    return_trampoline: u64,
}

impl TrampolineGuard {
    /// Reserve the guard region for a module with `trampoline_count`
    /// distinct HLE trampolines.
    pub(crate) fn reserve(trampoline_count: usize) -> Result<Self, RuntimeError> {
        // Keep index `count` as the invalid-trampoline diagnostic sentinel.
        // A normal guest return targets the following guarded slot.
        let return_trampoline = HLE_TRAMPOLINE_BASE + (trampoline_count as u64 + 1) * 8;
        let logical_len = (trampoline_count as u64) * 8 + 16;
        let reserved_len = logical_len.div_ceil(PAGE_SIZE) * PAGE_SIZE;

        // SAFETY: `HLE_TRAMPOLINE_BASE` is a fixed, page-aligned (indeed
        // 64 KiB-aligned) high sentinel address (design doc §2/§7), passed
        // as an explicit non-null `lpAddress`. `VirtualAlloc` with an
        // explicit address either reserves exactly that address range or
        // fails (returns null) — it never silently relocates elsewhere.
        // `MEM_RESERVE` (no `MEM_COMMIT`) with `PAGE_NOACCESS` is a
        // well-formed request; the reservation is freed exactly once in
        // `Drop`.
        let raw = unsafe {
            VirtualAlloc(
                HLE_TRAMPOLINE_BASE as *const c_void,
                reserved_len as usize,
                MEM_RESERVE,
                PAGE_NOACCESS,
            )
        };
        if raw.is_null() {
            // Per design doc §2 step 2: report if the fixed address is
            // unusable rather than silently relocating (which would break
            // the deterministic HLE_TRAMPOLINE_BASE-relative addressing the
            // LM1 linker already baked into the module's relocation slots).
            return Err(RuntimeError::MapFailed);
        }
        debug_assert_eq!(
            raw as u64, HLE_TRAMPOLINE_BASE,
            "VirtualAlloc with an explicit lpAddress must return that exact address on success"
        );

        Ok(Self {
            base: HLE_TRAMPOLINE_BASE,
            len: reserved_len,
            return_trampoline,
        })
    }

    /// The guard region's base address (== [`HLE_TRAMPOLINE_BASE`] on a
    /// successful `reserve`).
    pub(crate) fn base(&self) -> u64 {
        self.base
    }

    /// The guard region's actual (page-rounded) length in bytes.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Dedicated guarded address used as the return target for guest calls.
    pub(crate) fn return_trampoline(&self) -> u64 {
        self.return_trampoline
    }
}

impl Drop for TrampolineGuard {
    fn drop(&mut self) {
        // SAFETY: `self.base` is the exact pointer `VirtualAlloc` returned
        // in `reserve`, freed exactly once here with `dwSize = 0` as
        // `MEM_RELEASE` requires.
        unsafe {
            VirtualFree(self.base as *mut c_void, 0, MEM_RELEASE);
        }
    }
}

/// Map a faulting address within the trampoline region to the
/// [`HleTrampoline`] it names: `idx = (fault_addr - HLE_TRAMPOLINE_BASE) /
/// 8`, indexed into `trampolines`. Returns `None` if `fault_addr` precedes
/// [`HLE_TRAMPOLINE_BASE`] or names an index past `trampolines.len()` (an
/// unmapped/unresolved trampoline slot).
pub(crate) fn resolve(fault_addr: u64, trampolines: &[HleTrampoline]) -> Option<&HleTrampoline> {
    let offset = fault_addr.checked_sub(HLE_TRAMPOLINE_BASE)?;
    let idx = usize::try_from(offset / 8).ok()?;
    trampolines.get(idx)
}
