//! Port of Kyty's `Core::VirtualMemory`
//! (`reference/kyty/source/lib/Core/src/VirtualMemory.cpp` +
//! `include/Kyty/Core/VirtualMemory.h`).
//!
//! On Windows — XPS5X's target — Kyty's `Core::VirtualMemory` free functions
//! are 1:1 forwarders to the `Sys` layer (`Alloc` → `sys_virtual_alloc`,
//! etc.), so this module is a thin faithful re-export over [`crate::sys_virtual`],
//! preserving Kyty's two-layer shape (a `Sys::sys_virtual_*` primitive and a
//! `Core::VirtualMemory::*` public wrapper). Callers that want the
//! Kyty-Core-named API use these; the primitives remain available directly.
//!
//! # Not ported: `ExceptionHandler`
//!
//! Kyty's `VirtualMemory::ExceptionHandler` (a Win32 Vectored Exception
//! Handler that traps guest access violations and dispatches to a handler
//! callback) is deliberately **not** ported here: XPS5X's runtime already
//! owns that responsibility with its own, more capable VEH machinery
//! (`xps5x-runtime`'s `dispatch.rs` — trampoline-guard dispatch + RT1a
//! genuine-fault recovery + exit-longjmp). Adding a second, parallel VEH in
//! `kyty-core` would violate the project's "don't invent parallel
//! architectures" rule. If a Kyty-Graphics port later needs the exact
//! `ExceptionInfo` shape, it should adapt `xps5x-runtime`'s handler rather
//! than resurrect this one.

pub use crate::sys_virtual::{Mode, SystemInfo};

/// `Core::GetSystemInfo()` — the host processor description.
#[must_use]
pub fn get_system_info() -> SystemInfo {
    crate::sys_virtual::sys_get_system_info()
}

/// `VirtualMemory::Init()`.
pub fn init() {
    crate::sys_virtual::sys_virtual_init();
}

/// `VirtualMemory::Alloc(address, size, mode)` — reserve+commit `size` bytes
/// (near `address` if nonzero) with protection `mode`; returns the base
/// address or `0` on failure.
#[must_use]
pub fn alloc(address: u64, size: u64, mode: Mode) -> u64 {
    crate::sys_virtual::sys_virtual_alloc(address, size, mode)
}

/// `VirtualMemory::AllocAligned(address, size, mode, alignment)`.
#[must_use]
pub fn alloc_aligned(address: u64, size: u64, mode: Mode, alignment: u64) -> u64 {
    crate::sys_virtual::sys_virtual_alloc_aligned(address, size, mode, alignment)
}

/// `VirtualMemory::AllocFixed(address, size, mode)` — commit at exactly
/// `address`; `false` if that range is unavailable.
#[must_use]
pub fn alloc_fixed(address: u64, size: u64, mode: Mode) -> bool {
    crate::sys_virtual::sys_virtual_alloc_fixed(address, size, mode)
}

/// `VirtualMemory::Free(address)`.
pub fn free(address: u64) -> bool {
    crate::sys_virtual::sys_virtual_free(address)
}

/// `VirtualMemory::Protect(address, size, mode, old_mode)` — change page
/// protection; writes the previous mode through `old_mode` when provided.
pub fn protect(address: u64, size: u64, mode: Mode, old_mode: Option<&mut Mode>) -> bool {
    crate::sys_virtual::sys_virtual_protect(address, size, mode, old_mode)
}

/// `VirtualMemory::FlushInstructionCache(address, size)`.
pub fn flush_instruction_cache(address: u64, size: u64) -> bool {
    crate::sys_virtual::sys_virtual_flush_instruction_cache(address, size)
}

/// `VirtualMemory::PatchReplace(vaddr, value)` — write `value` at `vaddr`,
/// temporarily making the page writable; `true` if the stored value changed.
pub fn patch_replace(vaddr: u64, value: u64) -> bool {
    crate::sys_virtual::sys_virtual_patch_replace(vaddr, value)
}

/// `VirtualMemory::IsExecute(mode)`.
#[must_use]
pub fn is_execute(mode: Mode) -> bool {
    mode.is_execute()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Core wrapper forwards to the Sys layer: an alloc/free round-trip
    /// through the Core-named API behaves exactly like the primitive.
    #[test]
    fn core_alloc_free_round_trip_forwards_to_sys() {
        let addr = alloc(0, 0x1000, Mode::ReadWrite);
        assert_ne!(
            addr, 0,
            "Core::VirtualMemory::alloc should return a base address"
        );
        assert!(free(addr), "Core::VirtualMemory::free should release it");
    }

    #[test]
    fn core_is_execute_matches_mode() {
        assert!(is_execute(Mode::ExecuteReadWrite));
        assert!(!is_execute(Mode::ReadWrite));
    }

    #[test]
    fn core_get_system_info_reports_a_processor() {
        assert!(!get_system_info().processor_name.is_empty());
    }
}
