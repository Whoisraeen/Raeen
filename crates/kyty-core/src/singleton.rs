//! Port of Kyty's `Core/Singleton.h`.
//!
//! Kyty's `Singleton<T>` lazily `malloc`s + placement-`new`s a `T` on first
//! `Instance()` and hands back a raw `T*` (never freed — a leaked process-
//! lifetime global). The faithful Rust equivalent is a lazily-initialized
//! process-lifetime global backed by [`std::sync::OnceLock`], which gives the
//! same "constructed once on first access, lives until process exit"
//! semantics without the raw pointer or manual allocation.
//!
//! Kyty singletons require `T: Default` here (Kyty default-constructs the
//! instance via `new (p) T`). Ported call sites use
//! `Singleton::<T>::instance()` where Kyty wrote `Singleton<T>::Instance()`.

use std::sync::OnceLock;

/// A process-lifetime, lazily-initialized singleton — the Rust analog of
/// Kyty's `Kyty::Core::Singleton<T>`.
///
/// Unlike the C++ original there is no inherited base class; a ported type
/// that was `class Foo : public Singleton<Foo>` instead keeps a private
/// `static SINGLETON: Singleton<Foo>` (or is accessed through this type
/// directly). `instance()` returns a shared reference valid for the rest of
/// the process, matching the original's returned pointer's lifetime.
pub struct Singleton<T> {
    cell: OnceLock<T>,
}

impl<T: Default> Singleton<T> {
    /// Create an as-yet-uninitialized singleton slot. `const` so it can back a
    /// `static`, mirroring Kyty's `static inline T* g_m_instance`.
    #[must_use]
    pub const fn new() -> Self {
        Self { cell: OnceLock::new() }
    }

    /// Kyty `Instance()`: construct the `T` (via `Default`) on first call and
    /// return a shared reference to it thereafter.
    pub fn instance(&self) -> &T {
        self.cell.get_or_init(T::default)
    }
}

impl<T: Default> Default for Singleton<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static CONSTRUCTIONS: AtomicU32 = AtomicU32::new(0);

    #[derive(Default)]
    struct Counter {
        value: u32,
    }

    impl Counter {
        fn make() -> Self {
            CONSTRUCTIONS.fetch_add(1, Ordering::SeqCst);
            Counter { value: 42 }
        }
    }

    #[test]
    fn instance_constructs_once_and_is_stable() {
        static S: Singleton<Counter> = Singleton::new();
        // First access constructs (via Default → value 0).
        let a = S.instance();
        let b = S.instance();
        assert_eq!(a.value, 0);
        // Same object both times (same address).
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn constructed_exactly_once_under_repeated_access() {
        let s: Singleton<Counter> = Singleton::new();
        // get_or_init with an explicit ctor to observe construction count.
        let first = s.cell.get_or_init(Counter::make);
        assert_eq!(first.value, 42);
        let _second = s.cell.get_or_init(Counter::make);
        assert_eq!(CONSTRUCTIONS.load(Ordering::SeqCst), 1);
    }
}
