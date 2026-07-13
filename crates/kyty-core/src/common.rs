//! Port of Kyty's `Core/Common.h`.
//!
//! `Common.h` is almost entirely C++ preprocessor machinery with **no Rust
//! analog**, ported here as documentation of non-applicability rather than
//! code — a faithful port preserves intent, and reproducing these as Rust
//! would manufacture constructs the language makes unnecessary:
//!
//! - `KYTY_CLASS_NO_COPY` / `KYTY_CLASS_DEFAULT_COPY` — C++ copy/move-control
//!   boilerplate. Rust is move-by-default and never copies implicitly; a type
//!   is non-copyable unless it derives `Clone`/`Copy`. No macro needed.
//! - `KYTY_FORCE_LINK_THIS/THAT` — defeats the C++ linker's dead-code
//!   stripping of translation units. Rust's module/crate model has no
//!   equivalent problem.
//! - `KYTY_FORMAT_PRINTF` — GCC/Clang `printf`-format attribute. Rust's
//!   `format!`/`format_args!` are checked by the compiler intrinsically.
//! - `KYTY_LOGI`/`KYTY_LOGE` — map to `println!`/`eprintln!` (or `tracing`)
//!   at each ported call site.
//! - The compiler/platform detection (`KYTY_COMPILER`, `KYTY_PLATFORM`) maps
//!   to `cfg!(...)` / `#[cfg(...)]`.
//!
//! The fixed-width integer aliases `Common.h` re-exports from `<cstdint>` are
//! Rust's built-in `u8`/`i32`/`u64`/… so no aliases are defined here either.

// Intentionally empty of code: see the module doc comment. This module exists
// so the mapping is discoverable by anyone porting a file that `#include`d
// `Common.h`.
