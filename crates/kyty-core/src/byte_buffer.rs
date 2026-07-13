//! Port of Kyty's `Kyty::Core::ByteBuffer`
//! (`reference/kyty/source/include/Kyty/Core/ByteBuffer.h`).
//!
//! In C++, `ByteBuffer` is a thin subclass of `Vector<std::byte>` (Kyty's
//! own re-implementation of a growable array) that adds two convenience
//! constructors: one from an `initializer_list<uint8_t>` and one that copies
//! a raw `(ptr, size)` buffer.
//!
//! Std mapping: `Vector<Byte>` -> `Vec<u8>` (Rust has no `std::byte`; a
//! buffer of raw bytes is idiomatically `u8`). `ByteBuffer` is implemented
//! here as a thin newtype wrapper around `Vec<u8>`, exposing Kyty's method
//! names (`GetData`/`Size`/etc. equivalents) via `Deref`/`DerefMut` to `Vec<u8>`
//! plus the two constructors from the C++ header, translated to idiomatic
//! Rust constructors (`from_slice`, `From<Vec<u8>>`, `From<&[u8]>`,
//! `FromIterator<u8>`). No unsafe is needed: the C++ version manually
//! allocates and `memcpy`s into an uninitialized buffer, which Rust's
//! `Vec::extend_from_slice` / `Vec::from` handle safely.

use std::ops::{Deref, DerefMut};

/// A buffer of raw bytes. Thin wrapper over `Vec<u8>` mirroring Kyty's
/// `ByteBuffer` (itself a thin subclass of `Vector<std::byte>`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ByteBuffer(Vec<u8>);

impl ByteBuffer {
    /// Equivalent to the default constructor `ByteBuffer()`.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Equivalent to `ByteBuffer(const void* buf, uint32_t size)`: copies
    /// `size` bytes starting at `buf` into a freshly allocated buffer.
    pub fn from_slice(buf: &[u8]) -> Self {
        Self(buf.to_vec())
    }

    /// Number of bytes currently stored (`Vector::Size()`).
    pub fn size(&self) -> usize {
        self.0.len()
    }

    /// Whether the buffer holds no bytes (`Vector::IsEmpty()`).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Raw pointer to the underlying data (`Vector::GetData()` /
    /// `Vector::GetDataConst()`).
    pub fn get_data(&self) -> &[u8] {
        &self.0
    }

    /// Mutable raw pointer to the underlying data (`Vector::GetData()`).
    pub fn get_data_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }

    /// Unwrap into the underlying `Vec<u8>`.
    pub fn into_vec(self) -> Vec<u8> {
        self.0
    }
}

impl Deref for ByteBuffer {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ByteBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<u8>> for ByteBuffer {
    fn from(v: Vec<u8>) -> Self {
        Self(v)
    }
}

impl From<&[u8]> for ByteBuffer {
    fn from(v: &[u8]) -> Self {
        Self(v.to_vec())
    }
}

impl FromIterator<u8> for ByteBuffer {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        Self(Vec::from_iter(iter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let b = ByteBuffer::new();
        assert!(b.is_empty());
        assert_eq!(b.size(), 0);
    }

    #[test]
    fn from_slice_copies_bytes() {
        let data = [1u8, 2, 3, 4, 5];
        let b = ByteBuffer::from_slice(&data);
        assert_eq!(b.size(), 5);
        assert_eq!(b.get_data(), &data);
    }

    #[test]
    fn from_initializer_list_equivalent() {
        // Mirrors ByteBuffer(std::initializer_list<uint8_t> list)
        let b: ByteBuffer = [0xDE, 0xAD, 0xBE, 0xEF].into_iter().collect();
        assert_eq!(b.size(), 4);
        assert_eq!(b.get_data(), &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn from_vec_and_slice_conversions() {
        let v = vec![9u8, 8, 7];
        let b: ByteBuffer = v.clone().into();
        assert_eq!(b.into_vec(), v);

        let s: &[u8] = &[1, 2, 3];
        let b2: ByteBuffer = s.into();
        assert_eq!(b2.get_data(), s);
    }

    #[test]
    fn deref_gives_vec_api() {
        let mut b = ByteBuffer::new();
        b.push(42);
        b.push(43);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0], 42);
    }

    #[test]
    fn deref_mut_allows_mutation() {
        let mut b = ByteBuffer::from_slice(&[1, 2, 3]);
        b.get_data_mut()[0] = 99;
        assert_eq!(b.get_data(), &[99, 2, 3]);
        b.clear();
        assert!(b.is_empty());
    }

    #[test]
    fn clone_and_eq() {
        let a = ByteBuffer::from_slice(&[1, 2, 3]);
        let c = a.clone();
        assert_eq!(a, c);
    }
}
