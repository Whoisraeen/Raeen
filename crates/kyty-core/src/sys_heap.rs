//! Port of Kyty's `Sys/Heap` layer.
//!
//! Kyty source:
//! - `reference/kyty/source/include/Kyty/Sys/SysHeap.h` — the cross-platform
//!   entry point; it does not declare its own API, it just re-exports
//!   whichever platform header (`Windows/SysWindowsHeap.h` or
//!   `Linux/SysLinuxHeap.h`) matches `KYTY_PLATFORM` (`IWYU pragma: export`).
//! - `reference/kyty/source/include/Kyty/Sys/Windows/SysWindowsHeap.h` — the
//!   Windows implementation. Its `#if 0` branch (a real `HeapCreate`/
//!   `HeapAlloc`/`HeapReAlloc`/`HeapFree` wrapper) is dead code; the `#else`
//!   branch that actually compiles is a plain `malloc`/`realloc`/`free`
//!   wrapper where `sys_heap_id_t = SysCS*` (a critical-section pointer) is
//!   `nullptr` for the unsynchronized API (`sys_heap_create`,
//!   `sys_heap_deafult` [sic — Kyty's own typo, preserved here since it is
//!   part of the ported public API], `sys_heap_alloc`, `sys_heap_realloc`,
//!   `sys_heap_free`) and a real critical section for the synchronized API
//!   (`sys_heap_create_s`, `sys_heap_alloc_s`, `sys_heap_realloc_s`,
//!   `sys_heap_free_s`), which enters/leaves the section around each
//!   operation. That is the branch ported here.
//!
//! std mapping: `malloc`/`realloc`/`free` map to Rust's
//! [`std::alloc`] allocator API (`alloc`/`realloc`/`dealloc` +
//! [`Layout`]), which is std's direct equivalent of a raw, size-tracked heap
//! allocator — this is the "prefer std where it fully covers the behavior"
//! case, not a manual-memory transliteration. Because `std::alloc::dealloc`/
//! `realloc` require the *exact* `Layout` used at allocation time (unlike C's
//! `free`/`realloc`, which recover the block size internally), allocation
//! sizes are tracked in a small side table keyed by pointer address — the
//! same bookkeeping C allocators do internally, just made explicit. Kyty's
//! `SysCS` critical section maps to [`std::sync::Mutex`].

// Windows-first: XPS5X's only current target. This whole module is gated so
// the crate keeps building on non-Windows hosts until a Linux counterpart
// (ported from `Kyty/Sys/Linux/SysLinuxHeap.h`) is added under its own module.
#![cfg(windows)]

use std::alloc::{alloc, dealloc, realloc as std_realloc, Layout};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Alignment used for every heap allocation. Kyty relies on the platform
/// `malloc`'s natural alignment guarantee (suitable for any object type); 16
/// bytes matches the typical `malloc` guarantee on both Windows (MSVC CRT)
/// and Linux (glibc).
const HEAP_ALIGN: usize = 16;

/// Kyty's `sys_heap_id_t` (`using sys_heap_id_t = SysCS*`). `None` is Kyty's
/// `nullptr` heap id (the unsynchronized default heap); `Some(_)` is a heap
/// created by [`sys_heap_create_s`], carrying the critical section that
/// `sys_heap_alloc_s`/`sys_heap_realloc_s`/`sys_heap_free_s` enter and leave
/// around each operation.
#[derive(Debug)]
pub struct SysHeapId(Option<Mutex<()>>);

impl SysHeapId {
    /// `SysCS::Enter()` — held only while the guard is alive (RAII stands in
    /// for Kyty's explicit `Enter`/`Leave` pair). `None` for unsynchronized
    /// heap ids, matching the non-`_s` functions ignoring `heap_id` entirely.
    fn enter(&self) -> Option<MutexGuard<'_, ()>> {
        self.0.as_ref().map(|cs| cs.lock().unwrap())
    }
}

fn size_table() -> &'static Mutex<HashMap<usize, usize>> {
    static TABLE: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn layout_for(size: usize) -> Layout {
    // size.max(1): `Layout` (and `std::alloc::alloc`) requires a non-zero
    // size; C's `malloc(0)` is implementation-defined but typically returns
    // a unique non-null pointer, which this matches.
    Layout::from_size_align(size.max(1), HEAP_ALIGN).expect("sys_heap: invalid layout")
}

fn raw_alloc(size: usize) -> *mut u8 {
    let layout = layout_for(size);
    // SAFETY: `layout` has non-zero size (`size.max(1)`) and a valid
    // (power-of-two) alignment, satisfying `alloc`'s contract.
    let ptr = unsafe { alloc(layout) };
    crate::exit_if!(ptr.is_null());
    size_table().lock().unwrap().insert(ptr as usize, size);
    ptr
}

fn raw_realloc(ptr: *mut u8, new_size: usize) -> *mut u8 {
    if ptr.is_null() {
        return raw_alloc(new_size);
    }
    let old_size = size_table()
        .lock()
        .unwrap()
        .remove(&(ptr as usize))
        .expect("sys_heap_realloc: pointer was not allocated by this heap");
    let old_layout = layout_for(old_size);
    // SAFETY: `ptr` was returned by a previous `raw_alloc`/`raw_realloc` call
    // using `old_layout` (same size class via `layout_for` + `HEAP_ALIGN`),
    // it has not been freed (it was just removed from the live-size table),
    // and `new_size.max(1)` inside `layout_for` keeps the new size non-zero
    // — satisfying `realloc`'s contract.
    let new_ptr = unsafe { std_realloc(ptr, old_layout, new_size.max(1)) };
    if new_ptr.is_null() {
        // Matches Kyty's `EXIT_IF(m == 0)` on realloc failure. Per
        // `std::alloc::realloc`'s contract the original block is left
        // valid/unmoved on failure, so re-track it under its old size to
        // keep the table consistent (defensive; `exit_if!` halts next).
        size_table().lock().unwrap().insert(ptr as usize, old_size);
    } else {
        size_table().lock().unwrap().insert(new_ptr as usize, new_size);
    }
    crate::exit_if!(new_ptr.is_null());
    new_ptr
}

fn raw_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let size = size_table().lock().unwrap().remove(&(ptr as usize));
    match size {
        // SAFETY: `ptr` was returned by `raw_alloc`/`raw_realloc` with
        // exactly this size (tracked in `size_table`, so `layout_for` yields
        // the identical `Layout` used at allocation time), and this is the
        // single point that removes it from the table, so it is freed once.
        Some(size) => unsafe { dealloc(ptr, layout_for(size)) },
        // Matches Kyty's `EXIT_IF(!r)` when `HeapFree`/`free` is asked to
        // free a pointer this heap never allocated.
        None => crate::exit_if!(true),
    }
}

/// `sys_heap_create()` — creates an unsynchronized heap id (Kyty: `nullptr`,
/// since the active implementation is a bare `malloc` wrapper).
#[must_use]
pub fn sys_heap_create() -> SysHeapId {
    SysHeapId(None)
}

/// `sys_heap_deafult()` — the process's default heap id. Name (including the
/// `deafult` typo) preserved verbatim from Kyty's public API.
#[must_use]
pub fn sys_heap_deafult() -> SysHeapId {
    SysHeapId(None)
}

/// `sys_heap_alloc(heap_id, size)` — unsynchronized allocation; `heap_id` is
/// accepted (matching the C signature) but unused, exactly as in Kyty.
#[must_use]
pub fn sys_heap_alloc(_heap_id: &SysHeapId, size: usize) -> *mut u8 {
    raw_alloc(size)
}

/// `sys_heap_realloc(heap_id, p, size)` — unsynchronized reallocation;
/// allocates if `p` is null, exactly as in Kyty.
#[must_use]
pub fn sys_heap_realloc(_heap_id: &SysHeapId, p: *mut u8, size: usize) -> *mut u8 {
    raw_realloc(p, size)
}

/// `sys_heap_free(heap_id, p)` — unsynchronized free.
pub fn sys_heap_free(_heap_id: &SysHeapId, p: *mut u8) {
    raw_free(p);
}

/// `sys_heap_create_s()` — creates a synchronized heap id backed by a fresh
/// critical section (Kyty: `new SysCS; cs->Init(); return cs;`).
#[must_use]
pub fn sys_heap_create_s() -> SysHeapId {
    SysHeapId(Some(Mutex::new(())))
}

/// `sys_heap_alloc_s(heap_id, size)` — allocation under `heap_id`'s critical
/// section.
#[must_use]
pub fn sys_heap_alloc_s(heap_id: &SysHeapId, size: usize) -> *mut u8 {
    let _guard = heap_id.enter();
    raw_alloc(size)
}

/// `sys_heap_realloc_s(heap_id, p, size)` — reallocation under `heap_id`'s
/// critical section.
#[must_use]
pub fn sys_heap_realloc_s(heap_id: &SysHeapId, p: *mut u8, size: usize) -> *mut u8 {
    let _guard = heap_id.enter();
    raw_realloc(p, size)
}

/// `sys_heap_free_s(heap_id, p)` — free under `heap_id`'s critical section.
pub fn sys_heap_free_s(heap_id: &SysHeapId, p: *mut u8) {
    let _guard = heap_id.enter();
    raw_free(p);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn create_and_deafult_are_unsynchronized() {
        let a = sys_heap_create();
        let b = sys_heap_deafult();
        assert!(a.enter().is_none());
        assert!(b.enter().is_none());
    }

    #[test]
    fn create_s_is_synchronized() {
        let h = sys_heap_create_s();
        assert!(h.enter().is_some());
    }

    #[test]
    fn alloc_write_read_free_roundtrip() {
        let heap = sys_heap_create();
        let p = sys_heap_alloc(&heap, 64);
        assert!(!p.is_null());
        // SAFETY: `p` was just allocated by `sys_heap_alloc` with size 64.
        unsafe {
            for i in 0..64u8 {
                p.add(i as usize).write(i);
            }
            for i in 0..64u8 {
                assert_eq!(p.add(i as usize).read(), i);
            }
        }
        sys_heap_free(&heap, p);
    }

    #[test]
    fn alloc_zero_size_is_non_null_and_freeable() {
        let heap = sys_heap_create();
        let p = sys_heap_alloc(&heap, 0);
        assert!(!p.is_null());
        sys_heap_free(&heap, p);
    }

    #[test]
    fn realloc_from_null_behaves_like_alloc() {
        let heap = sys_heap_create();
        let p = sys_heap_realloc(&heap, std::ptr::null_mut(), 32);
        assert!(!p.is_null());
        sys_heap_free(&heap, p);
    }

    #[test]
    fn realloc_growing_preserves_prefix() {
        let heap = sys_heap_create();
        let p = sys_heap_alloc(&heap, 8);
        // SAFETY: `p` was allocated with size 8 above.
        unsafe {
            for i in 0..8u8 {
                p.add(i as usize).write(i + 1);
            }
        }
        let p2 = sys_heap_realloc(&heap, p, 128);
        assert!(!p2.is_null());
        // SAFETY: `p2` is the (possibly moved) live allocation, now sized
        // 128, and the first 8 bytes are guaranteed preserved by realloc.
        unsafe {
            for i in 0..8u8 {
                assert_eq!(p2.add(i as usize).read(), i + 1);
            }
        }
        sys_heap_free(&heap, p2);
    }

    #[test]
    fn synchronized_alloc_free_roundtrip() {
        let heap = sys_heap_create_s();
        let p = sys_heap_alloc_s(&heap, 16);
        assert!(!p.is_null());
        let p2 = sys_heap_realloc_s(&heap, p, 256);
        assert!(!p2.is_null());
        sys_heap_free_s(&heap, p2);
    }

    #[test]
    fn synchronized_heap_survives_concurrent_use() {
        use std::sync::Arc;

        let heap = Arc::new(sys_heap_create_s());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let heap = Arc::clone(&heap);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    let p = sys_heap_alloc_s(&heap, 24);
                    assert!(!p.is_null());
                    let p = sys_heap_realloc_s(&heap, p, 48);
                    assert!(!p.is_null());
                    sys_heap_free_s(&heap, p);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    #[should_panic]
    fn free_of_untracked_pointer_panics() {
        let heap = sys_heap_create();
        // A stack address is never something this heap allocated.
        let mut stack_byte: u8 = 0;
        sys_heap_free(&heap, &mut stack_byte as *mut u8);
    }
}
