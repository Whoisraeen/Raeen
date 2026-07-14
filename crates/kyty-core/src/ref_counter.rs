//! Port of Kyty's `include/Kyty/Core/RefCounter.h` (`Kyty::Core::RefCounter<MutexPolicy>`).
//!
//! Kyty's original is an intrusive, manually-managed reference counter meant
//! to be a base class for objects handled through raw pointers: `Release()`
//! decrements a `uint32_t` refcount under a policy-selected mutex and
//! `delete this` when it hits zero; `CopyPtr` assigns a raw pointer and bumps
//! the count; `CopyOnWrite` clones the pointee (`new PtrType(**data)`) when
//! shared.
//!
//! That whole pattern — manual `delete this`, raw `PtrType**` pointers, a
//! pluggable mutex policy to make the same code single- or multi-threaded —
//! is scaffolding Rust's ownership model makes unnecessary:
//!   - shared ownership + automatic deallocation at refcount zero is
//!     `std::sync::Arc<T>` (or `Rc<T>` for the non-atomic/single-threaded
//!     case Kyty's `MutexPolicy = MutexNone` would have produced);
//!   - `CopyPtr` (assign + `AddRef`) is `Arc::clone`;
//!   - `CopyOnWrite` (clone the pointee when `Refs() > 1`) is exactly
//!     `Arc::make_mut`, which clones-on-write automatically and is even
//!     race-free (no separate lock/check/clone steps).
//!
//! So this module does not reproduce the intrusive base class. Instead it
//! documents the mapping above for downstream ported types (e.g. Kyty's
//! `SharedPtr`/`Ptr` wrappers) to use directly:
//!
//! | Kyty `RefCounter<M>` member          | Rust equivalent                     |
//! |---------------------------------------|--------------------------------------|
//! | `RefCounter()` / refcount starts at 1  | `Arc::new(value)`                    |
//! | `AddRef()` / `CopyPtr(dst, src)`       | `Arc::clone(&src)`                   |
//! | `Refs()`                               | `Arc::strong_count(&a)`               |
//! | `DecRef()` + `delete this` at 0        | drop the last `Arc` (automatic)       |
//! | `Release()`                            | drop the `Arc` (automatic)            |
//! | `CopyOnWrite(data)`                    | `Arc::make_mut(&mut data)`            |
//!
//! [`RefCount`] below is kept only as a minimal, safe, non-intrusive counter
//! for the rare case a caller needs the raw count semantics (`refs`,
//! `add_ref`, `release` returning whether the count reached zero) without
//! pulling in a full `Arc<T>` — e.g. bridging code that mirrors Kyty's
//! `Refs()`-based assertions during the port. It uses
//! `std::sync::atomic::AtomicU32`, i.e. Kyty's thread-safe `MutexPolicy`
//! instantiation; there is no separate single-threaded variant since
//! `Arc`/`Rc` already cover that split.
use crate::exit_if;
use std::sync::atomic::{AtomicU32, Ordering};

/// A minimal, safe stand-in for Kyty's `RefCounter<MutexPolicy>` counting
/// primitive. New code should prefer `Arc<T>` (see module docs); this exists
/// only for call sites that need the bare counter semantics.
#[derive(Debug, Default)]
pub struct RefCounter {
    refs: AtomicU32,
}

impl RefCounter {
    /// Equivalent to Kyty's `RefCounter()` — starts the count at 1.
    pub fn new() -> Self {
        Self {
            refs: AtomicU32::new(1),
        }
    }

    /// Equivalent to `RefCounter::Refs()` (private in Kyty, exposed here
    /// since there is no `Lock`/`Unlock` needed with an atomic).
    pub fn refs(&self) -> u32 {
        self.refs.load(Ordering::Acquire)
    }

    /// Equivalent to `RefCounter::AddRef()`.
    pub fn add_ref(&self) -> u32 {
        self.refs.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Equivalent to `RefCounter::Release()`. Returns `true` when the count
    /// reached zero (i.e. where Kyty would `delete this`); the caller is
    /// responsible for actually dropping/freeing in that case, since this
    /// type does not own a payload.
    pub fn release(&self) -> bool {
        exit_if!(self.refs.load(Ordering::Acquire) == 0);
        self.refs.fetch_sub(1, Ordering::AcqRel) - 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn new_starts_at_one() {
        let rc = RefCounter::new();
        assert_eq!(rc.refs(), 1);
    }

    #[test]
    fn add_ref_increments_and_returns_new_count() {
        let rc = RefCounter::new();
        assert_eq!(rc.add_ref(), 2);
        assert_eq!(rc.add_ref(), 3);
        assert_eq!(rc.refs(), 3);
    }

    #[test]
    fn release_decrements_and_signals_zero() {
        let rc = RefCounter::new();
        rc.add_ref(); // refs = 2
        assert!(!rc.release()); // refs = 1, not the last
        assert!(rc.release()); // refs = 0, last release
    }

    #[test]
    #[should_panic]
    fn release_past_zero_panics_like_exit_if() {
        let rc = RefCounter::new();
        assert!(rc.release()); // refs -> 0
        rc.release(); // EXIT_IF(m_refs == 0) equivalent
    }

    #[test]
    fn concurrent_add_ref_is_race_free() {
        let rc = Arc::new(RefCounter::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let rc = Arc::clone(&rc);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    rc.add_ref();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // initial 1 + 8*1000 add_refs
        assert_eq!(rc.refs(), 1 + 8 * 1000);
    }

    #[test]
    fn arc_covers_copy_ptr_and_copy_on_write_semantics() {
        // Demonstrates the mapping documented above: Arc::clone replaces
        // CopyPtr, Arc::make_mut replaces CopyOnWrite.
        let a = Arc::new(vec![1, 2, 3]);
        let b = Arc::clone(&a); // CopyPtr(dst, src) + AddRef
        assert_eq!(Arc::strong_count(&a), 2); // Refs()

        let mut c = Arc::clone(&a);
        Arc::make_mut(&mut c).push(4); // CopyOnWrite: clones since shared
        assert_eq!(*a, vec![1, 2, 3]);
        assert_eq!(*c, vec![1, 2, 3, 4]);
        drop(b);
    }
}
