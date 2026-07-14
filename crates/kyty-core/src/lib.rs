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

// Kyty `lib/Sys` port (Windows-targeted, thin FFI over win32 — see each
// module's header doc). Ported in a prior batch but left undeclared here;
// wired in now so they actually compile and test as part of the crate.
#[cfg(windows)]
pub mod sys_dbg;
#[cfg(windows)]
pub mod sys_file_io;
#[cfg(windows)]
pub mod sys_heap;
#[cfg(windows)]
pub mod sys_stdio;
#[cfg(windows)]
pub mod sys_stdlib;
#[cfg(windows)]
pub mod sys_swap_byte_order;
#[cfg(windows)]
pub mod sys_sync;
#[cfg(windows)]
pub mod sys_timer;
#[cfg(windows)]
pub mod sys_virtual;

// Kyty `lib/Core` wrappers over the Sys layer (Core::VirtualMemory forwards
// 1:1 to Sys on Windows — see the module doc).
#[cfg(windows)]
pub mod virtual_memory;

pub use array_wrapper::{Array, Array2, Array3};
pub use byte_buffer::ByteBuffer;
pub use date_time::{Date, DateTime, Jd, Time};
pub use hashmap::Hashmap;
pub use json_reader::{Json, JsonType};
pub use language::LanguageId;
pub use link_list::{List, ListIndex, ListSet};
pub use magic_enum::{MagicEnum, enum_name, enum_name8, enum_value};
pub use ref_counter::RefCounter;
pub use simple_array::SimpleArray;
pub use singleton::Singleton;
pub use string::{String, StringList};
pub use string8::String8;
pub use timer::Timer;
pub use vector::{INVALID_INDEX, Vector};
