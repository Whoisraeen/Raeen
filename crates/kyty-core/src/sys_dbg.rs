//! Port of Kyty's `Sys/Windows/SysWindowsDbg` (Windows implementation of the
//! cross-platform `Kyty::sys_dbg_*` API declared in
//! `include/Kyty/Sys/SysDbg.h` -> `include/Kyty/Sys/Windows/SysWindowsDbg.h`),
//! from `reference/kyty/source/{include/Kyty/Sys/Windows/SysWindowsDbg.h,
//! lib/Sys/src/SysWindowsDbg.cpp}` (MIT (c) 2021 InoriRus; see
//! `/THIRD_PARTY_NOTICES.md`).
//!
//! Windows-only debug/diagnostics helpers used by the emulator's fault and
//! crash-reporting machinery: call-stack walking, stack-region introspection
//! (reserved / guard / committed extents), main-module base+size lookup, and
//! installing a process-wide unhandled-exception filter.
//!
//! Mapping to std / windows-sys (this is an OS-abstraction leaf, so unlike the
//! pure-std `Core` modules, direct Win32 FFI is expected here):
//! - [`sys_stack_walk`] faithfully ports the 64-bit `WalkStack` helper
//!   (`RtlCaptureContext` + `RtlLookupFunctionEntry` + `RtlVirtualUnwind`).
//!   Kyty's legacy 32-bit EBP-chain walker (`#if KYTY_BITNESS == 32`) is not
//!   ported: XPS5X only targets x86_64.
//! - [`sys_stack_usage`] / [`sys_stack_usage_print`] walk `VirtualQuery`
//!   regions to find the reserved/guard/committed extents of the calling
//!   thread's stack, exactly as the C++ does (querying the region containing
//!   the local `MEMORY_BASIC_INFORMATION` itself to find the stack's
//!   `AllocationBase`, then stepping forward region by region).
//! - [`sys_get_code_info`] reports the main executable module's base address
//!   and size via `GetModuleInformation` (psapi). That API is declared
//!   manually below (as `K32GetModuleInformation`, exported by `kernel32.dll`
//!   since Windows Vista) rather than by adding the `Win32_System_ProcessStatus`
//!   windows-sys feature, following the precedent set in `dbg_assert.rs` for
//!   `IsDebuggerPresent`.
//! - [`sys_set_exception_filter`] installs a `SetUnhandledExceptionFilter`
//!   trampoline that forwards the faulting instruction address to a
//!   caller-supplied Rust callback.
//!
//! All Win32 calls are gated behind `#[cfg(windows)]`; on other targets this
//! module compiles to an empty shell (see the module-level `cfg` below), so
//! the crate still builds cross-platform.

#![cfg(windows)]

use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Mutex;

use windows_sys::Win32::Foundation::{HANDLE, HMODULE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    RtlCaptureContext, RtlLookupFunctionEntry, RtlVirtualUnwind, SetUnhandledExceptionFilter,
    CONTEXT, EXCEPTION_EXECUTE_HANDLER, EXCEPTION_POINTERS,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleA;
use windows_sys::Win32::System::Memory::{
    VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, PAGE_NOCACHE, PAGE_READONLY, PAGE_READWRITE,
    PAGE_WRITECOPY,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// `sys_dbg_stack_info_t` — reserved/guard/committed extents of a thread
/// stack, as reported by [`sys_stack_usage`]. Field names and meaning match
/// the Kyty struct exactly.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SysDbgStackInfo {
    pub addr: usize,

    pub reserved_addr: usize,
    pub reserved_size: usize,
    pub guard_addr: usize,
    pub guard_size: usize,
    pub commited_addr: usize,
    pub commited_size: usize,

    pub total_size: usize,
}

/// `exception_filter_func_t` — a callback invoked with the faulting
/// instruction address when the filter installed by
/// [`sys_set_exception_filter`] fires.
pub type ExceptionFilterFunc = fn(*mut c_void);

/// Manually declared because `Win32_System_ProcessStatus` (the windows-sys
/// feature gating `GetModuleInformation`/`MODULEINFO`) is not among this
/// workspace's enabled windows-sys features. `K32GetModuleInformation` is the
/// same function, exported directly by `kernel32.dll` since Windows Vista
/// (psapi.h defines `GetModuleInformation` as a macro to it on modern SDKs),
/// so this avoids widening the shared feature list for one call.
#[repr(C)]
struct ModuleInfo {
    lp_base_of_dll: *mut c_void,
    size_of_image: u32,
    entry_point: *mut c_void,
}

unsafe extern "system" {
    fn K32GetModuleInformation(
        h_process: HANDLE,
        h_module: HMODULE,
        lp_mod_info: *mut ModuleInfo,
        cb: u32,
    ) -> i32;
}

const READABLE: u32 = PAGE_EXECUTE_READ
    | PAGE_EXECUTE_READWRITE
    | PAGE_EXECUTE_WRITECOPY
    | PAGE_READONLY
    | PAGE_READWRITE
    | PAGE_WRITECOPY;
const PROTECTED: u32 = PAGE_GUARD | PAGE_NOCACHE | PAGE_NOACCESS;

static G_EXCEPTION_FILTER_FUNC: Mutex<Option<ExceptionFilterFunc>> = Mutex::new(None);

/// `sys_mem_read_allowed` — true if `ptr` refers to committed, readable
/// memory that is not guard/no-access/no-cache protected.
///
/// Deliberately a *safe* function taking a raw pointer, mirroring Kyty's
/// `bool sys_mem_read_allowed(const void*)`: `VirtualQuery` only inspects
/// the VA region metadata containing `ptr` and never dereferences it, so
/// any address (valid or not) is a sound argument — which is the whole
/// point of the query.
#[must_use]
#[allow(clippy::not_unsafe_ptr_arg_deref)] // VirtualQuery inspects, never derefs, `ptr` (see doc + SAFETY below)
pub fn sys_mem_read_allowed(ptr: *const c_void) -> bool {
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };

    // SAFETY: `mbi` is a valid, appropriately sized out-buffer for
    // `VirtualQuery`; `ptr` may be any address (VirtualQuery itself never
    // dereferences it, it only inspects the containing VA region).
    let s = unsafe { VirtualQuery(ptr, &mut mbi, size_of::<MEMORY_BASIC_INFORMATION>()) };
    crate::exit_if!(s == 0);

    (mbi.Protect & PROTECTED) == 0
        && (mbi.State & MEM_COMMIT) != 0
        && (mbi.AllocationProtect & READABLE) != 0
}

/// `sys_stack_walk` — walks the current call stack starting at the caller,
/// filling `stack` with up to `stack.len()` return addresses (innermost
/// first) and returning the number of frames actually captured. Faithful
/// port of the x86_64 `WalkStack`/`sys_stack_walk` pair: captures the current
/// register context with `RtlCaptureContext`, then repeatedly looks up the
/// unwind info for the current `Rip` with `RtlLookupFunctionEntry` and steps
/// one frame with `RtlVirtualUnwind`, stopping when there is no further
/// unwind info or the unwound `Rip` is zero.
#[must_use]
pub fn sys_stack_walk(stack: &mut [usize]) -> usize {
    // SAFETY: `context` is a plain register-snapshot POD struct;
    // `RtlCaptureContext` fills it in-place and cannot fail.
    // `RtlLookupFunctionEntry`/`RtlVirtualUnwind` are the standard Windows x64
    // unwinder pair, called with valid local out-parameters for each step;
    // this mirrors the reference `WalkStack` loop exactly.
    unsafe {
        let mut context: CONTEXT = std::mem::zeroed();
        RtlCaptureContext(&mut context);

        let mut frame = 0usize;
        while frame < stack.len() {
            stack[frame] = context.Rip as usize;
            frame += 1;

            let mut image_base: u64 = 0;
            let runtime_function =
                RtlLookupFunctionEntry(context.Rip, &mut image_base, std::ptr::null_mut());
            if runtime_function.is_null() {
                break;
            }

            let mut handler_data: *mut c_void = std::ptr::null_mut();
            let mut establisher_frame: u64 = 0;
            let _ = RtlVirtualUnwind(
                0, // UNW_FLAG_NHANDLER
                image_base,
                context.Rip,
                runtime_function,
                &mut context,
                &mut handler_data,
                &mut establisher_frame,
                std::ptr::null_mut(),
            );

            if context.Rip == 0 {
                break;
            }
        }
        frame
    }
}

/// `sys_stack_usage_print` — prints a stack info block in the same
/// `(reserved) + (guard) + (committed)` layout as the reference `printf`.
pub fn sys_stack_usage_print(stack: &SysDbgStackInfo) {
    println!(
        "stack: (0x{:x}, {}) + (0x{:x}, {}) + (0x{:x}, {})",
        stack.reserved_addr,
        stack.reserved_size,
        stack.guard_addr,
        stack.guard_size,
        stack.commited_addr,
        stack.commited_size
    );
}

/// `sys_stack_usage` — determines the reserved, guard-page, and committed
/// extents of the calling thread's stack by walking `VirtualQuery` regions
/// forward from the region that contains `s` itself (a local on the current
/// stack), exactly as the reference implementation does.
pub fn sys_stack_usage(s: &mut SysDbgStackInfo) {
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };

    // SAFETY: each `VirtualQuery` call writes into `mbi`, a validly sized
    // local out-buffer; the queried address in each step is either `&mbi`
    // itself (a valid stack address) or a `BaseAddress`/`AllocationBase`
    // previously reported by `VirtualQuery`, offset by the previously
    // reported `RegionSize` — i.e. the start of the next VA region, which
    // `VirtualQuery` accepts for any address (committed or not).
    unsafe {
        let ss = VirtualQuery(
            std::ptr::addr_of!(mbi).cast(),
            &mut mbi,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        );
        crate::exit_if!(ss == 0);
        let reserved = mbi.AllocationBase;

        let ss = VirtualQuery(reserved, &mut mbi, size_of::<MEMORY_BASIC_INFORMATION>());
        crate::exit_if!(ss == 0);
        let reserved_size = mbi.RegionSize;

        let ss = VirtualQuery(
            reserved.cast::<u8>().add(reserved_size).cast(),
            &mut mbi,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        );
        crate::exit_if!(ss == 0);
        let guard_page = mbi.BaseAddress;
        let guard_page_size = mbi.RegionSize;

        let ss = VirtualQuery(
            guard_page.cast::<u8>().add(guard_page_size).cast(),
            &mut mbi,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        );
        crate::exit_if!(ss == 0);
        let commited = mbi.BaseAddress;
        let commited_size = mbi.RegionSize;

        s.reserved_addr = reserved as usize;
        s.reserved_size = reserved_size;
        s.guard_addr = guard_page as usize;
        s.guard_size = guard_page_size;
        s.commited_addr = commited as usize;
        s.commited_size = commited_size;

        s.addr = s.reserved_addr;
        s.total_size = s.reserved_size + s.guard_size + s.commited_size;
    }
}

/// `sys_get_code_info` — reports the base address and size of the main
/// executable module (`GetModuleHandle(nullptr)`) via `GetModuleInformation`.
pub fn sys_get_code_info(addr: &mut usize, size: &mut usize) {
    let mut info: ModuleInfo = unsafe { std::mem::zeroed() };

    // SAFETY: `GetCurrentProcess` and `GetModuleHandleA(null)` (main module)
    // never fail; `info` is a validly sized out-buffer matching the psapi
    // `MODULEINFO` layout for `K32GetModuleInformation`.
    unsafe {
        let h_process = GetCurrentProcess();
        let h_module = GetModuleHandleA(std::ptr::null());
        K32GetModuleInformation(
            h_process,
            h_module,
            &mut info,
            size_of::<ModuleInfo>() as u32,
        );
    }

    *addr = info.lp_base_of_dll as usize;
    *size = info.size_of_image as usize;
}

/// `sys_set_exception_filter` — installs `func` as the process-wide
/// unhandled-exception filter: on an unhandled Win32 exception, `func` is
/// called with the faulting instruction address, then the exception is
/// handled (`EXCEPTION_EXECUTE_HANDLER`), matching the reference behavior.
pub fn sys_set_exception_filter(func: ExceptionFilterFunc) {
    *G_EXCEPTION_FILTER_FUNC.lock().unwrap() = Some(func);

    // SAFETY: `exception_filter` matches the required
    // `unsafe extern "system" fn(*const EXCEPTION_POINTERS) -> i32` signature
    // and is valid for the process lifetime (a `fn` item, not a closure).
    unsafe {
        SetUnhandledExceptionFilter(Some(exception_filter));
    }
}

/// `ExceptionFilter` — the installed top-level exception filter trampoline;
/// forwards the faulting address to the registered Rust callback and always
/// requests immediate handling, exactly as the reference `ExceptionFilter`.
unsafe extern "system" fn exception_filter(exception: *const EXCEPTION_POINTERS) -> i32 {
    // SAFETY: Windows guarantees `exception` and its `ExceptionRecord` are
    // valid, live pointers for the duration of this callback.
    let addr = unsafe { (*(*exception).ExceptionRecord).ExceptionAddress };

    if let Some(f) = *G_EXCEPTION_FILTER_FUNC.lock().unwrap() {
        f(addr);
    }

    EXCEPTION_EXECUTE_HANDLER
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_RECORD;

    #[test]
    fn sys_mem_read_allowed_true_for_stack_local() {
        let x = 5_i32;
        assert!(sys_mem_read_allowed(std::ptr::addr_of!(x).cast()));
    }

    #[test]
    fn sys_mem_read_allowed_false_for_null() {
        // The null-page region is reserved/no-access, never committed+readable.
        assert!(!sys_mem_read_allowed(std::ptr::null()));
    }

    #[test]
    fn sys_stack_walk_captures_this_frame() {
        let mut stack = [0usize; 16];
        let depth = sys_stack_walk(&mut stack);
        assert!(depth > 0, "expected at least one captured frame");
        assert!(depth <= stack.len());
        // Every captured return address should be non-null.
        for &addr in &stack[..depth] {
            assert_ne!(addr, 0);
        }
    }

    #[test]
    fn sys_stack_walk_respects_capacity() {
        let mut stack = [0usize; 2];
        let depth = sys_stack_walk(&mut stack);
        assert!(depth <= 2);
    }

    #[test]
    fn sys_stack_usage_reports_nonzero_regions() {
        let mut info = SysDbgStackInfo::default();
        sys_stack_usage(&mut info);

        assert_eq!(info.addr, info.reserved_addr);
        assert!(info.reserved_size > 0);
        assert!(info.guard_size > 0 || info.commited_size > 0);
        assert_eq!(
            info.total_size,
            info.reserved_size + info.guard_size + info.commited_size
        );
    }

    #[test]
    fn sys_stack_usage_print_does_not_panic() {
        let mut info = SysDbgStackInfo::default();
        sys_stack_usage(&mut info);
        sys_stack_usage_print(&info);
    }

    #[test]
    fn sys_get_code_info_reports_main_module() {
        let mut addr = 0usize;
        let mut size = 0usize;
        sys_get_code_info(&mut addr, &mut size);
        assert_ne!(addr, 0);
        assert!(size > 0);
    }

    #[test]
    fn sys_set_exception_filter_registers_callback() {
        static SEEN: AtomicUsize = AtomicUsize::new(0);
        fn callback(addr: *mut std::ffi::c_void) {
            SEEN.store(addr as usize, Ordering::SeqCst);
        }

        sys_set_exception_filter(callback);
        assert!(G_EXCEPTION_FILTER_FUNC.lock().unwrap().is_some());

        // Drive the installed trampoline directly (without raising a real
        // structured exception) to verify it forwards the faulting address
        // and requests immediate handling, matching `ExceptionFilter` in the
        // reference implementation.
        let mut fake_addr = 0x1234_usize;
        let record = EXCEPTION_RECORD {
            ExceptionCode: 0,
            ExceptionFlags: 0,
            ExceptionRecord: std::ptr::null_mut(),
            ExceptionAddress: std::ptr::addr_of_mut!(fake_addr).cast(),
            NumberParameters: 0,
            ExceptionInformation: [0; 15],
        };
        let pointers = EXCEPTION_POINTERS {
            ExceptionRecord: std::ptr::addr_of!(record).cast_mut(),
            ContextRecord: std::ptr::null_mut(),
        };

        // SAFETY: `pointers` and `record` are valid locals for this call.
        let result = unsafe { exception_filter(std::ptr::addr_of!(pointers)) };

        assert_eq!(result, EXCEPTION_EXECUTE_HANDLER);
        assert_eq!(
            SEEN.load(Ordering::SeqCst),
            std::ptr::addr_of!(fake_addr) as usize
        );
    }
}
