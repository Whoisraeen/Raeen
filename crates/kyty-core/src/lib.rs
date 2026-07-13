//! `kyty-core` — a faithful Rust port of Kyty's `lib/Core` foundation
//! (`reference/kyty/source/lib/Core` + `include/Kyty/Core`, MIT © 2021
//! InoriRus; see `/THIRD_PARTY_NOTICES.md`).
//!
//! Porting conventions (see the master roadmap,
//! `docs/superpowers/plans/2026-07-13-kyty-full-port.md`): types that merely
//! re-implement `std` (`Vector`, `String`, `Hashmap`, …) are provided here as
//! thin wrappers over `std` that expose Kyty's public API, so downstream
//! ported subsystems compile against the same names and semantics.
//!
//! Modules are added as each Core unit is ported, bottom-up in dependency
//! order.

#![forbid(unsafe_op_in_unsafe_fn)]

// Phase 1, module 1 — foundation leaves. The assertion macros
// (`exit!`, `exit_if!`, `assert_kyty!`, `exit_not_implemented!`,
// `not_implemented!`) are exported crate-wide via `#[macro_export]`, so later
// Core modules and downstream `kyty-*` crates can use them directly.
pub mod common;
pub mod dbg_assert;
pub mod safe_delete;
pub mod singleton;

pub use singleton::Singleton;
