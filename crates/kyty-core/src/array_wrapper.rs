//! Port of Kyty's `Core/ArrayWrapper.h` (`Kyty::Core::Array`, `Array2`,
//! `Array3`).
//!
//! Kyty's `Array<Type, Num>` (and its `Array2`/`Array3` nestings) is a
//! fixed-size, stack-allocated array wrapper: a hand-rolled substitute for
//! what Rust's native `[T; N]` const-generic array already provides directly
//! in the language (fixed size known at compile time, `Index`/`IndexMut`
//! with bounds-checked panics, `Copy`/`Clone`, iteration).
//!
//! Std mapping used here: `Array<T, N>` is a thin newtype wrapping `[T; N]`
//! (const generic, `N: usize`), with `Kyty`'s method names (`Size`, `GetPtr`,
//! `ByteSize`) preserved as `size`, `get_ptr`, `byte_size` for API
//! compatibility with downstream ported code. `Array2`/`Array3` are the same
//! wrapper nested (`Array<Array<T, N2>, N1>`, `Array<Array<Array<T, N3>, N2>,
//! N1>`), matching Kyty's own definitions in terms of the previous one.
//! Bounds-checked indexing is provided via `EXIT_IF`-equivalent panics
//! (`exit_if!`) through `Index`/`IndexMut`, mirroring Kyty's explicit range
//! check rather than relying only on the implicit one native arrays already
//! have, so the panic message matches the ported call sites' expectations.

use crate::exit_if;

/// Fixed-size array wrapper, 1:1 with `Kyty::Core::Array<Type, Num>`.
///
/// `Type` must implement `Default` so `Array::new()` can value-initialize
/// every element, mirroring `Array() = default` / `m_ptr {}` zero-init in
/// the C++ source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array<T, const N: usize> {
    data: [T; N],
}

impl<T: Default, const N: usize> Default for Array<T, N> {
    fn default() -> Self {
        Self { data: std::array::from_fn(|_| T::default()) }
    }
}

impl<T: Default, const N: usize> Array<T, N> {
    /// Kyty: `Array()` (default ctor, zero/value-initializes all elements).
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, const N: usize> Array<T, N> {
    /// Kyty: `Array(std::initializer_list<Type> list)`.
    pub fn from_list(list: [T; N]) -> Self {
        Self { data: list }
    }

    /// Kyty: `int Size() const`.
    #[must_use]
    pub fn size(&self) -> i32 {
        N as i32
    }

    /// Kyty: `Type* GetPtr()`.
    pub fn get_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Kyty: `const Type* GetPtr() const`.
    #[must_use]
    pub fn get_ptr_const(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Kyty: `size_t ByteSize() const`.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        std::mem::size_of::<T>() * N
    }

    /// Kyty: `begin()`/`end()` (mutable iteration).
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Kyty: `begin() const`/`end() const` (and `cbegin()`/`cend()`).
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Kyty: `operator Type const*() const` / `operator Type*()`.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Kyty: `operator Type*()`.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T, const N: usize> std::ops::Index<i32> for Array<T, N> {
    type Output = T;

    /// Kyty: `const Type& operator[](int index) const` — `EXIT_IF(index < 0
    /// || index >= Num)`.
    fn index(&self, index: i32) -> &T {
        exit_if!(index < 0 || index as usize >= N);
        &self.data[index as usize]
    }
}

impl<T, const N: usize> std::ops::IndexMut<i32> for Array<T, N> {
    /// Kyty: `Type& operator[](int index)` — `EXIT_IF(index < 0 || index >=
    /// Num)`.
    fn index_mut(&mut self, index: i32) -> &mut T {
        exit_if!(index < 0 || index as usize >= N);
        &mut self.data[index as usize]
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Array<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Array<T, N> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

/// Kyty: `Kyty::Core::Array2<Type, Num1, Num2>` — an array of arrays.
pub type Array2<T, const N1: usize, const N2: usize> = Array<Array<T, N2>, N1>;

/// Kyty: `Kyty::Core::Array3<Type, Num1, Num2, Num3>` — an array of arrays of
/// arrays.
pub type Array3<T, const N1: usize, const N2: usize, const N3: usize> = Array<Array<Array<T, N3>, N2>, N1>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zero_initializes() {
        let a: Array<i32, 4> = Array::new();
        assert_eq!(a.size(), 4);
        for v in a.iter() {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn from_list_and_index() {
        let a: Array<i32, 3> = Array::from_list([1, 2, 3]);
        assert_eq!(a[0], 1);
        assert_eq!(a[1], 2);
        assert_eq!(a[2], 3);
    }

    #[test]
    fn index_mut_writes_through() {
        let mut a: Array<i32, 3> = Array::from_list([1, 2, 3]);
        a[1] = 42;
        assert_eq!(a[1], 42);
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn out_of_bounds_index_panics() {
        let a: Array<i32, 3> = Array::from_list([1, 2, 3]);
        let _ = a[3];
    }

    #[test]
    #[should_panic(expected = "KYTY EXIT_IF failed")]
    fn negative_index_panics() {
        let a: Array<i32, 3> = Array::from_list([1, 2, 3]);
        let _ = a[-1];
    }

    #[test]
    fn byte_size_matches_type_and_len() {
        let a: Array<i32, 5> = Array::new();
        assert_eq!(a.byte_size(), std::mem::size_of::<i32>() * 5);
    }

    #[test]
    fn as_slice_and_ptr_roundtrip() {
        let mut a: Array<i32, 3> = Array::from_list([10, 20, 30]);
        assert_eq!(a.as_slice(), &[10, 20, 30]);
        unsafe {
            assert_eq!(*a.get_ptr(), 10);
        }
    }

    #[test]
    fn iteration_via_for_loop() {
        let a: Array<i32, 3> = Array::from_list([1, 2, 3]);
        let sum: i32 = (&a).into_iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn array2_nested_indexing() {
        let mut a: Array2<i32, 2, 3> = Array::new();
        a[0][1] = 99;
        assert_eq!(a[0][1], 99);
        assert_eq!(a.size(), 2);
        assert_eq!(a[0].size(), 3);
    }

    #[test]
    fn array3_nested_indexing() {
        let mut a: Array3<i32, 2, 2, 2> = Array::new();
        a[1][1][1] = 7;
        assert_eq!(a[1][1][1], 7);
    }
}
