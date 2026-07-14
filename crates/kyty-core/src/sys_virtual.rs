//! Port of Kyty's `Sys::SysVirtual` Windows backend
//! (`reference/kyty/source/include/Kyty/Sys/Windows/SysWindowsVirtual.h` +
//! `reference/kyty/source/lib/Sys/src/SysWindowsVirtual.cpp`), the
//! OS-abstraction layer that `Core::VirtualMemory`
//! (`include/Kyty/Core/VirtualMemory.h`) sits on top of.
//!
//! XPS5X targets Windows first (see the crate-wide porting conventions), so
//! this whole module is `#[cfg(windows)]`-gated and backed by `windows-sys`
//! FFI where Kyty relies on Windows-specific behavior that `std` does not
//! expose: `VirtualAlloc`/`VirtualAlloc2`/`VirtualFree`/`VirtualProtect` for
//! page-granular reserve/commit/protect/free (including the dynamically
//! resolved `VirtualAlloc2`, used to satisfy the PS4-address-space alignment
//! hints the emulator needs), and `FlushInstructionCache` for self-modifying
//! (JIT-patched) code.
//!
//! Two Core types the header depends on (`Core::SystemInfo` and
//! `Core::VirtualMemory::Mode`) are not yet ported as their own module, so
//! minimal faithful definitions live here; when `Core::VirtualMemory` is
//! ported it should reuse these rather than redeclaring them.
//!
//! Naming: Kyty's free functions here are already `snake_case`
//! (`sys_virtual_alloc`, ...), so names are preserved verbatim. The two
//! `get_protection_flag` C++ overloads (Mode -> DWORD and DWORD -> Mode)
//! become two distinctly-named private helpers, since Rust has no overload
//! resolution: [`mode_to_protection_flag`] and [`protection_flag_to_mode`].
//! `sys_get_system_info` returns an owned [`SystemInfo`] instead of writing
//! through an out-pointer, per this crate's convention of not transliterating
//! C++ manual-pointer idioms; and rather than pulling in the external
//! `cpuinfo` library Kyty uses, the processor name is read directly via the
//! `CPUID` brand-string leaves (`std::arch::x86_64::__cpuid`), which needs no
//! extra dependency and no per-process initialization, so
//! [`sys_virtual_init`] is a no-op faithful stand-in for
//! `cpuinfo_initialize()`.

#![cfg(windows)]

use crate::String;
use std::ffi::c_void;
use std::sync::OnceLock;
use windows_sys::Win32::Foundation::{
    ERROR_INVALID_ADDRESS, ERROR_INVALID_PARAMETER, GetLastError, HANDLE, HMODULE,
};
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
use windows_sys::Win32::System::Memory::{
    MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_EXTENDED_PARAMETER_0,
    MEM_EXTENDED_PARAMETER_1, MEM_RELEASE, MEM_RESERVE, MemExtendedParameterAddressRequirements,
    PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS, PAGE_READONLY,
    PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Kyty `Core::SystemInfo` (`VirtualMemory.h`): the subset of host info Sys
/// exposes. Only `ProcessorName` is populated by `sys_get_system_info` on
/// Windows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemInfo {
    pub processor_name: String,
}

/// Kyty `Core::VirtualMemory::Mode`: page protection mode, mirroring the
/// original's bit-flag-valued enumerators (`ReadWrite = Read | Write`, etc.)
/// as precomputed discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Mode {
    #[default]
    NoAccess = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3,
    Execute = 4,
    ExecuteRead = 5,
    ExecuteWrite = 6,
    ExecuteReadWrite = 7,
}

impl Mode {
    /// Kyty `Core::VirtualMemory::IsExecute(Mode)`.
    #[must_use]
    pub fn is_execute(self) -> bool {
        matches!(
            self,
            Mode::Execute | Mode::ExecuteRead | Mode::ExecuteWrite | Mode::ExecuteReadWrite
        )
    }
}

/// Kyty `get_protection_flag(VirtualMemory::Mode)` (the C++ file has two
/// overloads; this is the Mode -> Win32-flag direction).
fn mode_to_protection_flag(mode: Mode) -> u32 {
    match mode {
        Mode::Read => PAGE_READONLY,
        Mode::Write | Mode::ReadWrite => PAGE_READWRITE,
        Mode::Execute => PAGE_EXECUTE,
        Mode::ExecuteRead => PAGE_EXECUTE_READ,
        Mode::ExecuteWrite | Mode::ExecuteReadWrite => PAGE_EXECUTE_READWRITE,
        Mode::NoAccess => PAGE_NOACCESS,
    }
}

/// Kyty `get_protection_flag(DWORD)` (the Win32-flag -> Mode direction).
fn protection_flag_to_mode(protect: u32) -> Mode {
    match protect {
        PAGE_NOACCESS => Mode::NoAccess,
        PAGE_READONLY => Mode::Read,
        PAGE_READWRITE => Mode::ReadWrite,
        PAGE_EXECUTE => Mode::Execute,
        PAGE_EXECUTE_READ => Mode::ExecuteRead,
        PAGE_EXECUTE_READWRITE => Mode::ExecuteReadWrite,
        _ => Mode::NoAccess,
    }
}

/// SAFETY helper: `GetLastError` takes no arguments and only reads the
/// calling thread's last-error slot; it has no failure mode.
fn last_error() -> u32 {
    // SAFETY: no preconditions; pure TLS read.
    unsafe { GetLastError() }
}

/// Kyty `sys_virtual_init()`: in the original this calls `cpuinfo_initialize()`
/// before `sys_get_system_info` can query the CPU package. This port reads
/// the processor brand string directly via `CPUID` (see module docs), which
/// needs no separate initialization step, so this is a faithful no-op stand-in
/// kept for API parity with call sites that invoke it before
/// [`sys_get_system_info`].
pub fn sys_virtual_init() {}

#[cfg(target_arch = "x86_64")]
fn cpu_brand_string() -> Vec<u8> {
    use std::arch::x86_64::__cpuid;

    // `__cpuid` is a safe intrinsic on this toolchain (the brand-string
    // leaves 0x8000_0002..=0x8000_0004 are available on every x86-64 CPU
    // shipped since the early 2000s; it just runs `cpuid` and returns the
    // four output registers verbatim).
    let leaves = [__cpuid(0x8000_0002), __cpuid(0x8000_0003), __cpuid(0x8000_0004)];

    let mut bytes = Vec::with_capacity(48);
    for leaf in leaves {
        bytes.extend_from_slice(&leaf.eax.to_le_bytes());
        bytes.extend_from_slice(&leaf.ebx.to_le_bytes());
        bytes.extend_from_slice(&leaf.ecx.to_le_bytes());
        bytes.extend_from_slice(&leaf.edx.to_le_bytes());
    }
    // The brand string is NUL-padded/terminated by the CPU; trim trailing
    // NULs so `String::from_utf8_bytes` doesn't see them.
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    bytes
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_brand_string() -> Vec<u8> {
    Vec::new()
}

/// Kyty `sys_get_system_info(SystemInfo*)`, adapted to return an owned value
/// (see module docs) rather than writing through an out-pointer.
#[must_use]
pub fn sys_get_system_info() -> SystemInfo {
    SystemInfo {
        processor_name: String::from_utf8_bytes(&cpu_brand_string()),
    }
}

/// Kyty `sys_virtual_alloc()`: reserve+commit `size` bytes with protection
/// `mode`, at `address` if nonzero, otherwise anywhere the OS picks (via the
/// aligned path with `alignment = 1`).
#[must_use]
pub fn sys_virtual_alloc(address: u64, size: u64, mode: Mode) -> u64 {
    let ptr: u64 = if address == 0 {
        sys_virtual_alloc_aligned(address, size, mode, 1)
    } else {
        // SAFETY: reserves+commits `size` bytes at the caller-specified
        // address in this process's address space. Only the resulting
        // address is observed as an opaque `u64`; the caller is responsible
        // for eventually releasing it via `sys_virtual_free`.
        (unsafe {
            VirtualAlloc(
                address as *const c_void,
                size as usize,
                MEM_COMMIT | MEM_RESERVE,
                mode_to_protection_flag(mode),
            )
        }) as u64
    };

    if ptr == 0 {
        let err = last_error();
        if err != ERROR_INVALID_ADDRESS {
            eprintln!("VirtualAlloc() failed: {err:#010x}");
        } else {
            return sys_virtual_alloc_aligned(address, size, mode, 1);
        }
    }
    ptr
}

type VirtualAlloc2Func = unsafe extern "system" fn(
    HANDLE,
    *const c_void,
    usize,
    u32,
    u32,
    *mut MEM_EXTENDED_PARAMETER,
    u32,
) -> *mut c_void;

/// Kyty `ResolveVirtualAlloc2()`: `VirtualAlloc2` is only available on
/// Windows 10 1803+, so Kyty resolves it dynamically via `GetProcAddress`
/// instead of linking it directly.
fn resolve_virtual_alloc2() -> Option<VirtualAlloc2Func> {
    static CACHE: OnceLock<Option<VirtualAlloc2Func>> = OnceLock::new();
    *CACHE.get_or_init(|| {
        // SAFETY: `GetModuleHandleA`/`GetProcAddress` take NUL-terminated
        // ANSI strings and query the process's already-loaded module table
        // (KernelBase.dll is always loaded); both return values are checked
        // for null/None before use. The `transmute` re-types the resolved
        // `FARPROC` (an untyped function pointer) to `VirtualAlloc2`'s known
        // signature, mirroring the C++ `reinterpret_cast` to
        // `VirtualAlloc2_func_t`.
        unsafe {
            let h: HMODULE = GetModuleHandleA(c"KernelBase".as_ptr().cast());
            if h.is_null() {
                return None;
            }
            let proc = GetProcAddress(h, c"VirtualAlloc2".as_ptr().cast());
            proc.map(|f| std::mem::transmute::<unsafe extern "system" fn() -> isize, VirtualAlloc2Func>(f))
        }
    })
}

fn align_up(addr: u64, alignment: u64) -> u64 {
    (addr + alignment - 1) & !(alignment - 1)
}

/// Kyty `sys_virtual_alloc_aligned()`: reserve+commit memory with a specific
/// alignment via the dynamically-resolved `VirtualAlloc2`, biasing the
/// search range toward the "system managed" range when `address == 0` and
/// otherwise toward the requested address (falling back to the general user
/// range), doubling `alignment` and retrying on `ERROR_INVALID_PARAMETER`.
#[must_use]
pub fn sys_virtual_alloc_aligned(address: u64, size: u64, mode: Mode, alignment: u64) -> u64 {
    if alignment == 0 {
        eprintln!("VirtualAlloc2 failed: {:#010x}", last_error());
        return 0;
    }

    const SYSTEM_MANAGED_MIN: u64 = 0x00_00_04_00_00;
    const SYSTEM_MANAGED_MAX: u64 = 0x07_FF_FF_BF_FF;
    const USER_MIN: u64 = 0x10_00_00_00_00;
    const USER_MAX: u64 = 0xFB_FF_FF_FF_FF;

    let mut req = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (if address == 0 {
            SYSTEM_MANAGED_MIN
        } else {
            align_up(address, alignment)
        }) as *mut c_void,
        HighestEndingAddress: (if address == 0 { SYSTEM_MANAGED_MAX } else { USER_MAX }) as *mut c_void,
        Alignment: alignment as usize,
    };
    let mut param = MEM_EXTENDED_PARAMETER {
        Anonymous1: MEM_EXTENDED_PARAMETER_0 {
            _bitfield: MemExtendedParameterAddressRequirements as u64,
        },
        Anonymous2: MEM_EXTENDED_PARAMETER_1 {
            Pointer: std::ptr::addr_of_mut!(req).cast(),
        },
    };

    let mut req2 = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (if address == 0 {
            USER_MIN
        } else {
            align_up(address, alignment)
        }) as *mut c_void,
        HighestEndingAddress: USER_MAX as *mut c_void,
        Alignment: alignment as usize,
    };
    let mut param2 = MEM_EXTENDED_PARAMETER {
        Anonymous1: MEM_EXTENDED_PARAMETER_0 {
            _bitfield: MemExtendedParameterAddressRequirements as u64,
        },
        Anonymous2: MEM_EXTENDED_PARAMETER_1 {
            Pointer: std::ptr::addr_of_mut!(req2).cast(),
        },
    };

    let Some(virtual_alloc2) = resolve_virtual_alloc2() else {
        // Kyty: `EXIT_NOT_IMPLEMENTED(virtual_alloc2 == nullptr)`; we are
        // already in the `None` case, so this is unconditional here.
        crate::not_implemented!()
    };

    // SAFETY: `virtual_alloc2` was resolved from `KernelBase.dll` and
    // re-typed to the documented `VirtualAlloc2` signature; `param`/`param2`
    // point at address requirement structs that outlive the call (they are
    // locals of this stack frame, and the call is synchronous).
    let mut ptr = (unsafe {
        virtual_alloc2(
            GetCurrentProcess(),
            std::ptr::null(),
            size as usize,
            MEM_COMMIT | MEM_RESERVE,
            mode_to_protection_flag(mode),
            &mut param,
            1,
        )
    }) as u64;

    if ptr == 0 {
        // SAFETY: same contract as above, with the fallback (general user
        // range) address requirement.
        ptr = (unsafe {
            virtual_alloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                size as usize,
                MEM_COMMIT | MEM_RESERVE,
                mode_to_protection_flag(mode),
                &mut param2,
                1,
            )
        }) as u64;
    }

    if ptr == 0 {
        let err = last_error();
        if err != ERROR_INVALID_PARAMETER {
            eprintln!("VirtualAlloc2(alignment = {alignment:#018x}) failed: {err:#010x}");
        } else {
            return sys_virtual_alloc_aligned(address, size, mode, alignment << 1u64);
        }
    }
    ptr
}

/// Kyty `sys_virtual_alloc_fixed()`: reserve+commit memory at exactly
/// `address`, releasing it and failing if the OS placed it elsewhere.
#[must_use]
pub fn sys_virtual_alloc_fixed(address: u64, size: u64, mode: Mode) -> bool {
    // SAFETY: reserves+commits `size` bytes at `address`; the resulting
    // pointer is only compared/observed as a `u64`, and released below via
    // `VirtualFree` if it doesn't match the request.
    let ptr = (unsafe {
        VirtualAlloc(
            address as *const c_void,
            size as usize,
            MEM_COMMIT | MEM_RESERVE,
            mode_to_protection_flag(mode),
        )
    }) as u64;

    if ptr == 0 {
        eprintln!("VirtualAlloc() failed: {:#010x}", last_error());
        return false;
    }

    if ptr != address {
        eprintln!("VirtualAlloc() failed: wrong address");
        // SAFETY: `ptr` is the address just returned by `VirtualAlloc`
        // above (a full reserved region), which is exactly what `VirtualFree`
        // with `MEM_RELEASE` requires.
        unsafe {
            VirtualFree(ptr as *mut c_void, 0, MEM_RELEASE);
        }
        return false;
    }

    true
}

/// Kyty `sys_virtual_free()`: release a region previously reserved by one of
/// the `sys_virtual_alloc*` functions.
pub fn sys_virtual_free(address: u64) -> bool {
    // SAFETY: `MEM_RELEASE` requires `address` be the base address of a
    // region previously reserved by `VirtualAlloc`/`VirtualAlloc2`, which
    // callers of this function are required to pass (matching Kyty's
    // contract).
    let ok = unsafe { VirtualFree(address as *mut c_void, 0, MEM_RELEASE) };
    if ok == 0 {
        eprintln!("VirtualFree() failed: {:#010x}", last_error());
        return false;
    }
    true
}

/// Kyty `sys_virtual_protect()`: change the protection of an already-mapped
/// region, optionally reporting the previous mode back through `old_mode`.
pub fn sys_virtual_protect(address: u64, size: u64, mode: Mode, old_mode: Option<&mut Mode>) -> bool {
    let mut old_protect: u32 = 0;
    // SAFETY: `address`/`size` must describe an already-mapped region within
    // this process, per `VirtualProtect`'s contract (the same contract Kyty
    // places on its callers); `old_protect` is a valid local receiving the
    // previous protection.
    let ok = unsafe {
        VirtualProtect(
            address as *const c_void,
            size as usize,
            mode_to_protection_flag(mode),
            &mut old_protect,
        )
    };
    if ok == 0 {
        eprintln!("VirtualProtect() failed: {:#010x}", last_error());
        return false;
    }
    if let Some(old_mode) = old_mode {
        *old_mode = protection_flag_to_mode(old_protect);
    }
    true
}

/// Kyty `sys_virtual_flush_instruction_cache()`: flush the CPU instruction
/// cache for a range of self-modified/JIT-patched code.
pub fn sys_virtual_flush_instruction_cache(address: u64, size: u64) -> bool {
    // SAFETY: `FlushInstructionCache` only reads the given range to flush
    // cache lines; it does not require the range to be executable, only
    // readable and valid, which is the same precondition Kyty places on
    // callers.
    let ok = unsafe { FlushInstructionCache(GetCurrentProcess(), address as *const c_void, size as usize) };
    if ok == 0 {
        eprintln!("FlushInstructionCache() failed: {:#010x}", last_error());
        return false;
    }
    true
}

/// Kyty `sys_virtual_patch_replace()`: hot-patch an 8-byte value at `vaddr`
/// (e.g. a JIT'd instruction/pointer slot), temporarily making the page
/// writable and restoring its previous protection afterward, flushing the
/// instruction cache if that previous protection was executable. Returns
/// whether the value actually changed.
pub fn sys_virtual_patch_replace(vaddr: u64, value: u64) -> bool {
    let mut old_mode = Mode::NoAccess;
    sys_virtual_protect(vaddr, 8, Mode::ReadWrite, Some(&mut old_mode));

    let ptr = vaddr as *mut u64;
    // SAFETY: caller guarantees `vaddr` names a valid 8-byte region within
    // this process (the same contract Kyty places on `sys_virtual_patch_replace`
    // callers); we just made it read-write above. `read_unaligned`/
    // `write_unaligned` are used instead of a direct dereference (as the C++
    // does) so this does not require natural alignment, matching the
    // original's runtime behavior on x86 without invoking Rust-level
    // misaligned-access UB.
    let changed = unsafe { ptr.read_unaligned() != value };
    unsafe {
        ptr.write_unaligned(value);
    }

    sys_virtual_protect(vaddr, 8, old_mode, None);

    if old_mode.is_execute() {
        sys_virtual_flush_instruction_cache(vaddr, 8);
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_is_execute_matches_kyty_isexecute() {
        assert!(!Mode::NoAccess.is_execute());
        assert!(!Mode::Read.is_execute());
        assert!(!Mode::Write.is_execute());
        assert!(!Mode::ReadWrite.is_execute());
        assert!(Mode::Execute.is_execute());
        assert!(Mode::ExecuteRead.is_execute());
        assert!(Mode::ExecuteWrite.is_execute());
        assert!(Mode::ExecuteReadWrite.is_execute());
    }

    #[test]
    fn mode_discriminants_match_kyty_bit_values() {
        assert_eq!(Mode::NoAccess as u32, 0);
        assert_eq!(Mode::Read as u32, 1);
        assert_eq!(Mode::Write as u32, 2);
        assert_eq!(Mode::ReadWrite as u32, 3);
        assert_eq!(Mode::Execute as u32, 4);
        assert_eq!(Mode::ExecuteRead as u32, 5);
        assert_eq!(Mode::ExecuteWrite as u32, 6);
        assert_eq!(Mode::ExecuteReadWrite as u32, 7);
    }

    #[test]
    fn protection_flag_round_trips_through_mode() {
        for mode in [
            Mode::NoAccess,
            Mode::Read,
            Mode::ReadWrite,
            Mode::Execute,
            Mode::ExecuteRead,
            Mode::ExecuteReadWrite,
        ] {
            let flag = mode_to_protection_flag(mode);
            assert_eq!(protection_flag_to_mode(flag), mode);
        }
        // The Write-only and ExecuteWrite-only variants collapse onto
        // ReadWrite/ExecuteReadWrite on Win32 (there is no write-only page
        // protection), matching Kyty's mapping exactly.
        assert_eq!(mode_to_protection_flag(Mode::Write), mode_to_protection_flag(Mode::ReadWrite));
        assert_eq!(
            mode_to_protection_flag(Mode::ExecuteWrite),
            mode_to_protection_flag(Mode::ExecuteReadWrite)
        );
    }

    #[test]
    fn alloc_and_free_round_trip() {
        let size = 0x1000u64;
        let addr = sys_virtual_alloc(0, size, Mode::ReadWrite);
        assert_ne!(addr, 0, "sys_virtual_alloc should succeed for a fresh mapping");
        assert_eq!(addr % 0x1000, 0, "VirtualAlloc always returns page-aligned addresses");

        // The region must actually be writable.
        let ptr = addr as *mut u64;
        unsafe {
            ptr.write_unaligned(0xDEAD_BEEF_CAFE_F00D);
            assert_eq!(ptr.read_unaligned(), 0xDEAD_BEEF_CAFE_F00D);
        }

        assert!(sys_virtual_free(addr));
    }

    #[test]
    fn alloc_aligned_honors_alignment() {
        let alignment = 0x10000u64; // 64 KiB, the Windows allocation granularity.
        let addr = sys_virtual_alloc_aligned(0, 0x1000, Mode::ReadWrite, alignment);
        assert_ne!(addr, 0);
        assert_eq!(addr % alignment, 0, "returned address must satisfy the requested alignment");
        assert!(sys_virtual_free(addr));
    }

    #[test]
    fn protect_reports_previous_mode() {
        let size = 0x1000u64;
        let addr = sys_virtual_alloc(0, size, Mode::ReadWrite);
        assert_ne!(addr, 0);

        let mut old_mode = Mode::NoAccess;
        assert!(sys_virtual_protect(addr, size, Mode::Read, Some(&mut old_mode)));
        assert_eq!(old_mode, Mode::ReadWrite);

        // Restore so the region can be freed cleanly (not strictly required
        // by VirtualFree, but keeps the test symmetric).
        assert!(sys_virtual_protect(addr, size, Mode::ReadWrite, None));
        assert!(sys_virtual_free(addr));
    }

    #[test]
    fn patch_replace_reports_change_and_writes_value() {
        let size = 0x1000u64;
        let addr = sys_virtual_alloc(0, size, Mode::ReadWrite);
        assert_ne!(addr, 0);

        unsafe {
            (addr as *mut u64).write_unaligned(0);
        }

        assert!(
            sys_virtual_patch_replace(addr, 0x1122_3344_5566_7788),
            "value differs from the initial 0, so this must report a change"
        );
        let readback = unsafe { (addr as *const u64).read_unaligned() };
        assert_eq!(readback, 0x1122_3344_5566_7788);

        assert!(
            !sys_virtual_patch_replace(addr, 0x1122_3344_5566_7788),
            "patching in the same value again must report no change"
        );

        assert!(sys_virtual_free(addr));
    }

    #[test]
    fn flush_instruction_cache_succeeds_on_valid_range() {
        let size = 0x1000u64;
        let addr = sys_virtual_alloc(0, size, Mode::ExecuteReadWrite);
        assert_ne!(addr, 0);
        assert!(sys_virtual_flush_instruction_cache(addr, size));
        assert!(sys_virtual_free(addr));
    }

    #[test]
    fn alloc_fixed_fails_gracefully_for_a_bad_hint() {
        // Address 0 is never a valid hint for VirtualAlloc; this must return
        // false rather than panicking.
        assert!(!sys_virtual_alloc_fixed(0, 0x1000, Mode::ReadWrite));
    }

    #[test]
    fn get_system_info_reports_a_processor_name() {
        sys_virtual_init();
        let info = sys_get_system_info();
        assert!(!info.processor_name.is_empty(), "CPUID brand string should be non-empty on x86-64");
    }
}
