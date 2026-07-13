//! Port of Kyty's `Sys::SysCS` / `sys_sleep`
//! (`reference/kyty/source/include/Kyty/Sys/SysSync.h` — the cross-platform
//! interface header re-exported per-OS — implemented here from
//! `reference/kyty/source/include/Kyty/Sys/Windows/SysWindowsSync.h`, the
//! Windows implementation, which is XPS5X's only target platform).
//!
//! `SysCS` is a thin wrapper around a Win32 `CRITICAL_SECTION`: a
//! recursive, spin-then-block mutex. Unlike `std::sync::Mutex`, Kyty's API
//! is *not* RAII/guard-based — `Init`/`Delete` bracket the object's
//! lifetime (separately from Rust's own construction/destruction) and
//! `Enter`/`TryEnter`/`Leave` are independent calls with no borrow tying a
//! `Leave()` to its matching `Enter()`. `std::sync::Mutex` cannot express
//! that shape (its guard is tied to the borrow that produced it) and is not
//! recursive, so this is one of the FFI exceptions called out by the port
//! conventions: it binds `windows-sys`'s `CRITICAL_SECTION` functions
//! directly, mirroring the original 1:1.
//!
//! Kyty's `m_check_ptr` invariant (null until `Init()`, `this` afterwards,
//! null again after `Delete()`) is preserved verbatim as a run-time guard,
//! using this crate's `exit_if!` macro in place of Kyty's `EXIT_IF` — both
//! halt on a violated precondition (double-`Init`, `Enter`/`Leave` before
//! `Init`/after `Delete`, or dropping a `SysCS` that was never `delete()`d).
//!
//! `sys_sleep(ms)` maps to Win32 `Sleep`; since Kyty's Windows sleep and
//! Rust's `std::thread::sleep` both ultimately rest on the same OS
//! primitive and have equivalent observable behavior (sleep at least the
//! requested duration), this part of the port uses `std` rather than FFI.
//!
//! Method names are the `snake_case` equivalents of Kyty's `PascalCase` API
//! (`Init` -> `init`, `TryEnter` -> `try_enter`, ...), per this crate's
//! porting convention.

#![cfg(windows)]

use crate::exit_if;
use std::cell::UnsafeCell;
use std::ptr;
use std::time::Duration;
use windows_sys::Win32::System::Threading::{
    DeleteCriticalSection, EnterCriticalSection, InitializeCriticalSectionAndSpinCount,
    LeaveCriticalSection, TryEnterCriticalSection, CRITICAL_SECTION,
};

/// `SYS_CS_SPIN_COUNT` — the spin count `SysCS::init` passes to
/// `InitializeCriticalSectionAndSpinCount`, verbatim from Kyty.
const SYS_CS_SPIN_COUNT: u32 = 16;

/// Rust port of `Kyty::SysCS`. A recursive critical section with an
/// explicit `init`/`delete` lifecycle (distinct from Rust construction/
/// drop) matching the original's non-RAII API.
///
/// # Notes
/// Like the C++ original (whose `CRITICAL_SECTION` address is registered
/// with the OS on `init()`), a `SysCS` must not be moved after `init()` is
/// called and before `delete()` — keep it behind a stable address (a
/// struct field, `Box`, etc.).
pub struct SysCS {
    /// Mirrors Kyty's `m_check_ptr`: null before `init`/after `delete`,
    /// `self` while live. Guards every operation against misuse.
    check_ptr: UnsafeCell<*const SysCS>,
    cs: UnsafeCell<CRITICAL_SECTION>,
}

// SAFETY: `SysCS` forwards all interior mutability to the Win32
// `CRITICAL_SECTION` APIs, which are explicitly designed for concurrent
// cross-thread use (that is the entire point of a critical section); the
// `check_ptr` invariant is likewise only ever read/written under the same
// external synchronization discipline Kyty relies on (init/delete are not
// meant to race with enter/leave, exactly as in the original).
unsafe impl Send for SysCS {}
unsafe impl Sync for SysCS {}

impl Default for SysCS {
    fn default() -> Self {
        Self {
            check_ptr: UnsafeCell::new(ptr::null()),
            cs: UnsafeCell::new(CRITICAL_SECTION {
                DebugInfo: ptr::null_mut(),
                LockCount: 0,
                RecursionCount: 0,
                OwningThread: ptr::null_mut(),
                LockSemaphore: ptr::null_mut(),
                SpinCount: 0,
            }),
        }
    }
}

impl SysCS {
    /// `SysCS()` — default-constructs an un-initialized critical section.
    /// Call [`SysCS::init`] before use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads the current `check_ptr` value (Kyty's `m_check_ptr`).
    fn check_ptr(&self) -> *const SysCS {
        // SAFETY: reading a `Copy` raw pointer through `UnsafeCell::get`.
        // `check_ptr` is only ever written by `init`/`delete`; like Kyty's
        // own unsynchronized field access, callers are responsible for not
        // racing those against each other or against this read.
        unsafe { *self.check_ptr.get() }
    }

    /// `Init()` — brings the critical section to life. Fatal if already
    /// initialized (`m_check_ptr != nullptr`).
    pub fn init(&self) {
        exit_if!(!self.check_ptr().is_null());
        unsafe {
            *self.check_ptr.get() = self as *const SysCS;
            // SAFETY: `self.cs` is a valid, stable `CRITICAL_SECTION`
            // (guaranteed not to move while initialized, see the struct's
            // doc comment); `InitializeCriticalSectionAndSpinCount` is the
            // documented Win32 API for setting it up.
            InitializeCriticalSectionAndSpinCount(self.cs.get(), SYS_CS_SPIN_COUNT);
        }
    }

    /// `Delete()` — tears the critical section down. Fatal unless it was
    /// `init()`-ed on `self` (`m_check_ptr != this`).
    pub fn delete(&self) {
        exit_if!(self.check_ptr() != self as *const SysCS);
        unsafe {
            *self.check_ptr.get() = ptr::null();
            // SAFETY: `self.cs` was initialized by `init()` (checked above
            // via `check_ptr`) and is still at a stable address.
            DeleteCriticalSection(self.cs.get());
        }
    }

    /// `Enter()` — blocking acquire (recursive: the same thread may call
    /// this more than once). Fatal unless initialized.
    pub fn enter(&self) {
        exit_if!(self.check_ptr() != self as *const SysCS);
        // SAFETY: initialization checked above; `EnterCriticalSection` is
        // the documented Win32 API for acquiring the section.
        unsafe { EnterCriticalSection(self.cs.get()) };
    }

    /// `TryEnter()` — non-blocking acquire attempt; `true` on success.
    /// Fatal unless initialized.
    #[must_use]
    pub fn try_enter(&self) -> bool {
        exit_if!(self.check_ptr() != self as *const SysCS);
        // SAFETY: initialization checked above.
        unsafe { TryEnterCriticalSection(self.cs.get()) != 0 }
    }

    /// `Leave()` — releases one level of acquisition. Fatal unless
    /// initialized.
    pub fn leave(&self) {
        exit_if!(self.check_ptr() != self as *const SysCS);
        // SAFETY: initialization checked above.
        unsafe { LeaveCriticalSection(self.cs.get()) };
    }
}

impl Drop for SysCS {
    /// `~SysCS()` — fatal if `delete()` was never called
    /// (`m_check_ptr != nullptr`), exactly like the original destructor.
    fn drop(&mut self) {
        exit_if!(!self.check_ptr().is_null());
    }
}

/// `sys_sleep(ms)` — suspend the current thread for (at least) `ms`
/// milliseconds. Ported from Kyty's inline `Sleep(ms)` wrapper; `std`
/// fully covers this behavior so no FFI is used here.
pub fn sys_sleep(ms: u32) {
    std::thread::sleep(Duration::from_millis(u64::from(ms)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_default_is_uninitialized() {
        let cs = SysCS::new();
        // Not yet init()-ed: check_ptr must be null so Drop doesn't panic.
        assert!(cs.check_ptr().is_null());
    }

    #[test]
    fn init_then_delete_round_trips_cleanly() {
        let cs = SysCS::new();
        cs.init();
        assert_eq!(cs.check_ptr(), &cs as *const SysCS);
        cs.delete();
        assert!(cs.check_ptr().is_null());
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn double_init_panics() {
        let cs = SysCS::new();
        cs.init();
        cs.init(); // already initialized -> fatal
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn enter_before_init_panics() {
        let cs = SysCS::new();
        cs.enter();
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn delete_without_init_panics() {
        let cs = SysCS::new();
        cs.delete();
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn drop_without_delete_panics() {
        let cs = SysCS::new();
        cs.init();
        // `cs` drops here without a matching `delete()` -> Drop panics.
    }

    #[test]
    fn enter_is_recursive_on_same_thread() {
        let cs = SysCS::new();
        cs.init();
        cs.enter();
        // A real CRITICAL_SECTION allows the owning thread to re-enter;
        // this would deadlock with a plain (non-recursive) mutex.
        assert!(cs.try_enter());
        cs.leave();
        cs.leave();
        cs.delete();
    }

    #[test]
    fn try_enter_succeeds_when_uncontended() {
        let cs = SysCS::new();
        cs.init();
        assert!(cs.try_enter());
        cs.leave();
        cs.delete();
    }

    #[test]
    fn enter_leave_blocks_across_threads() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let cs = Arc::new(SysCS::new());
        cs.init();
        let counter = Arc::new(AtomicU32::new(0));

        cs.enter();
        let cs2 = Arc::clone(&cs);
        let counter2 = Arc::clone(&counter);
        let handle = std::thread::spawn(move || {
            cs2.enter();
            counter2.fetch_add(1, Ordering::SeqCst);
            cs2.leave();
        });

        // Give the spawned thread a chance to block on Enter.
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        cs.leave();
        handle.join().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Unwrap the Arc (now sole owner again) so we can delete() before drop.
        let cs = Arc::try_unwrap(cs).unwrap_or_else(|_| panic!("still shared"));
        cs.delete();
    }

    #[test]
    fn sys_sleep_sleeps_at_least_requested_duration() {
        let start = std::time::Instant::now();
        sys_sleep(20);
        assert!(start.elapsed() >= Duration::from_millis(15));
    }
}
