//! # kyty-math
//!
//! Port of Kyty's `lib/Math` (`reference/kyty/source/lib/Math`,
//! `include/Kyty/Math`) into idiomatic Rust.
//!
//! Per the Kyty-port conventions (heavily-templated 3rd-party-shaped C++ →
//! a maintained workspace crate rather than a hand transliteration):
//!
//! - **VectorAndMatrix** (`vec2/3/4`, `mat2/3/4`) → thin Kyty-named aliases
//!   and helpers over [`glam`]'s SIMD types (see [`vector_and_matrix`]).
//! - **Rand** (`std::mt19937` + uniform distributions) → the [`rand`] crate,
//!   preserving Kyty's `Rand` API (see [`rand`]).
//! - **Crypto** (AES / Hash) → **not ported here**: it maps to the RustCrypto
//!   crates (`aes`/`cbc`/`sha1`), which `xps5x-firmware` already uses for SELF
//!   decryption. Re-implementing it in `kyty-math` would duplicate that.

pub mod rand;
pub mod vector_and_matrix;
