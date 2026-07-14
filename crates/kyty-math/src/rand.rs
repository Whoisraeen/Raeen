//! Port of Kyty's `Math::Rand`
//! (`reference/kyty/source/include/Kyty/Math/Rand.h` +
//! `lib/Math/src/Rand.cpp`).
//!
//! Kyty's `Rand` is a `std::mt19937` behind a set of static helpers producing
//! uniform integers/floats over inclusive/exclusive ranges. The idiomatic
//! Rust equivalent is the [`rand`] crate (workspace-crate convention). This
//! is **not** bit-identical to Kyty's stream — XPS5X is a clean-room reimpl,
//! not running Kyty's code, so the exact PRNG sequence is not load-bearing;
//! only the API shape and the distribution semantics (inclusive vs
//! exclusive bounds) are preserved.
//!
//! Kyty's `Rand` is a process-global singleton (`Init()` allocates a static
//! context). This port keeps that shape with a thread-local generator so the
//! free functions match Kyty's `Rand::Uint()`-style static call sites without
//! threading a generator through every caller.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;

thread_local! {
    /// The per-thread generator backing the `Rand::*` free functions. Seeded
    /// deterministically by default (like a freshly-`Init()`-ed Kyty context
    /// before any `Seed()`), re-seeded by [`seed`].
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0x4D65_7473_656E_6E65));
}

/// `Rand::Init()` — Kyty allocates its global context here. Nothing to do in
/// the Rust port (the thread-local generator is created on first use); kept
/// for API parity so ported call sites can call it verbatim.
pub fn init() {}

/// `Rand::Seed(s)` — reseed the generator.
pub fn seed(s: u64) {
    RNG.with(|r| *r.borrow_mut() = StdRng::seed_from_u64(s));
}

/// `Rand::Uint()` — uniform in `[0, 2^32)`.
pub fn uint() -> u32 {
    RNG.with(|r| r.borrow_mut().r#gen())
}

/// `Rand::Int()` — uniform over the whole `i32` range.
pub fn int() -> i32 {
    RNG.with(|r| r.borrow_mut().r#gen())
}

/// `Rand::Double()` — uniform in `[0.0, 1.0)`.
pub fn double() -> f64 {
    RNG.with(|r| r.borrow_mut().gen_range(0.0..1.0))
}

/// `Rand::DoubleInclusive()` — uniform in `[0.0, 1.0]`.
pub fn double_inclusive() -> f64 {
    RNG.with(|r| r.borrow_mut().gen_range(0.0..=1.0))
}

/// `Rand::DoubleRange(from, to)` — uniform in `[from, to)`. Requires
/// `from < to` (Kyty `EXIT_IF`s otherwise); this port panics on a bad range.
pub fn double_range(from_incl: f64, to_excl: f64) -> f64 {
    assert!(from_incl < to_excl, "double_range requires from < to");
    RNG.with(|r| r.borrow_mut().gen_range(from_incl..to_excl))
}

/// `Rand::DoubleInclusiveRange(from, to)` — uniform in `[from, to]`. `from ==
/// to` returns `from` (matching Kyty); `from > to` panics.
pub fn double_inclusive_range(from_incl: f64, to_incl: f64) -> f64 {
    assert!(
        from_incl <= to_incl,
        "double_inclusive_range requires from <= to"
    );
    if from_incl == to_incl {
        return from_incl;
    }
    RNG.with(|r| r.borrow_mut().gen_range(from_incl..=to_incl))
}

/// `Rand::Float()` — uniform in `[0.0, 1.0)`.
pub fn float() -> f32 {
    RNG.with(|r| r.borrow_mut().gen_range(0.0..1.0))
}

/// `Rand::FloatInclusive()` — uniform in `[0.0, 1.0]`.
pub fn float_inclusive() -> f32 {
    RNG.with(|r| r.borrow_mut().gen_range(0.0..=1.0))
}

/// `Rand::FloatRange(from, to)` — uniform in `[from, to)`.
pub fn float_range(from_incl: f32, to_excl: f32) -> f32 {
    assert!(from_incl < to_excl, "float_range requires from < to");
    RNG.with(|r| r.borrow_mut().gen_range(from_incl..to_excl))
}

/// `Rand::FloatInclusiveRange(from, to)` — uniform in `[from, to]`.
pub fn float_inclusive_range(from_incl: f32, to_incl: f32) -> f32 {
    assert!(
        from_incl <= to_incl,
        "float_inclusive_range requires from <= to"
    );
    if from_incl == to_incl {
        return from_incl;
    }
    RNG.with(|r| r.borrow_mut().gen_range(from_incl..=to_incl))
}

/// `Rand::UintInclusiveRange(from, to)` — uniform integer in `[from, to]`.
pub fn uint_inclusive_range(from_incl: u32, to_incl: u32) -> u32 {
    assert!(
        from_incl <= to_incl,
        "uint_inclusive_range requires from <= to"
    );
    RNG.with(|r| r.borrow_mut().gen_range(from_incl..=to_incl))
}

/// `Rand::IntInclusiveRange(from, to)` — uniform integer in `[from, to]`.
pub fn int_inclusive_range(from_incl: i32, to_incl: i32) -> i32 {
    assert!(
        from_incl <= to_incl,
        "int_inclusive_range requires from <= to"
    );
    RNG.with(|r| r.borrow_mut().gen_range(from_incl..=to_incl))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_stay_within_bounds() {
        init();
        for _ in 0..1000 {
            let d = double();
            assert!((0.0..1.0).contains(&d));
            let di = double_inclusive();
            assert!((0.0..=1.0).contains(&di));
            let f = float();
            assert!((0.0..1.0).contains(&f));
            assert!((5..=7).contains(&uint_inclusive_range(5, 7)));
            assert!((-3..=3).contains(&int_inclusive_range(-3, 3)));
            let r = double_range(10.0, 20.0);
            assert!((10.0..20.0).contains(&r));
        }
    }

    #[test]
    fn degenerate_inclusive_range_returns_the_bound() {
        assert_eq!(double_inclusive_range(4.0, 4.0), 4.0);
        assert_eq!(float_inclusive_range(2.5, 2.5), 2.5);
    }

    #[test]
    fn seed_makes_the_sequence_reproducible() {
        seed(42);
        let a: Vec<u32> = (0..5).map(|_| uint()).collect();
        seed(42);
        let b: Vec<u32> = (0..5).map(|_| uint()).collect();
        assert_eq!(a, b, "same seed must reproduce the same sequence");
    }

    #[test]
    #[should_panic(expected = "from < to")]
    fn inverted_range_panics() {
        double_range(5.0, 1.0);
    }
}
