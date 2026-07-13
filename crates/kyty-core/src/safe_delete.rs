//! Port of Kyty's `Core/SafeDelete.h`.
//!
//! `SafeDelete.h` exists to make C++ manual memory management less dangerous:
//! `Delete(T*& p)` / `DeleteArray(T*& p)` and the `DeleteProtected` macros
//! `delete` a pointer, then overwrite it with a `0x1` "deadbeef" sentinel so a
//! subsequent double-delete is caught (`EXIT("already deleted")`) instead of
//! corrupting the heap.
//!
//! **This module is intentionally empty of code.** The entire problem it
//! solves — dangling pointers, double-free, use-after-free — is prevented by
//! construction in safe Rust: ownership and `Drop` free a value exactly once,
//! at a statically-known point, and the borrow checker forbids using it
//! afterward. Transliterating `Delete(T*&)` into Rust would mean *introducing*
//! raw pointers and `unsafe` solely to re-create a hazard Rust otherwise makes
//! impossible — the opposite of a faithful port, which preserves the intent
//! (safe reclamation), not the mechanism.
//!
//! Port mapping for call sites:
//! - `Delete(p)` / `DeleteProtected(p)` → drop the owner (end its scope, or
//!   `drop(x)`); for owned heap data hold it in `Box<T>`/`Vec<T>` and let
//!   `Drop` reclaim it.
//! - `DeleteArray(p)` → the owning `Vec<T>`/`Box<[T]>` drops its buffer.
//! - The "already deleted" guard has no analog: the borrow checker rejects a
//!   use-after-move at compile time.
