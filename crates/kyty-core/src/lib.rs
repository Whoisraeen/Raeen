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
pub mod array_wrapper;
pub mod byte_buffer;
pub mod common;
pub mod compression;
pub mod date_time;
pub mod dbg_assert;
pub mod hash;
pub mod hashmap;
pub mod json_reader;
pub mod language;
pub mod link_list;
pub mod magic_enum;
pub mod ref_counter;
pub mod safe_delete;
pub mod simple_array;
pub mod singleton;
pub mod string;
pub mod string8;
pub mod timer;
pub mod vector;

pub use array_wrapper::{Array, Array2, Array3};
pub use byte_buffer::ByteBuffer;
pub use date_time::{Date, DateTime, Jd, Time};
pub use hashmap::Hashmap;
pub use json_reader::{Json, JsonType};
pub use language::LanguageId;
pub use link_list::{List, ListIndex, ListSet};
pub use magic_enum::{enum_name, enum_name8, enum_value, MagicEnum};
pub use ref_counter::RefCounter;
pub use simple_array::SimpleArray;
pub use singleton::Singleton;
pub use string::{String, StringList};
pub use string8::String8;
pub use timer::Timer;
pub use vector::{Vector, INVALID_INDEX};
