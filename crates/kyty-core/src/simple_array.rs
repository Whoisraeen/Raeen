//! Port of Kyty's `Core/SimpleArray.h` (`Kyty::Core::SimpleArray<T>`).
//!
//! `SimpleArray<T>` in Kyty is a hand-rolled, manually-allocated dynamic
//! array (custom `mem_alloc`/`mem_realloc`, placement-new element
//! construction, a growth factor of `3/2`, a lazily-computed content hash,
//! and hand-written quicksort variants). All of that manual-memory
//! scaffolding exists only because C++ has no owning growable-array type
//! with move semantics built in — Rust's `Vec<T>` already provides
//! allocation, growth, drop-safety and iteration, so this is ported as a
//! **thin wrapper over `Vec<T>`** that exposes Kyty's method names
//! (`Size`, `Add`, `InsertAt`, `Find`, `RemoveAt`, `GetData`, `Sort`, …)
//! rather than reimplementing manual allocation/placement-new in `unsafe`
//! Rust.
//!
//! Notable behavioral choices carried over from the C++:
//! - `INVALID_INDEX` is `u32::MAX`, returned by `Find`/`Find` variants when
//!   no match exists (matches the C++ `static_cast<uint32_t>(-1)`).
//! - `operator[]` (`index`/`index_mut`) panics (`exit_if!`) on out-of-range
//!   access, matching `EXIT_IF(index >= m_values_num)`.
//! - `Hash()` is a lazily-computed, cached hash of the array's contents
//!   (mapped to Rust's `std::hash::Hash`/`DefaultHasher` rather than
//!   Kyty's custom `Core::hash()` byte-hash, since we no longer have a raw
//!   byte buffer to hash over).
//! - The move constructor/move-assignment are `= delete`d in the C++ (only
//!   copy and default-construct are allowed); this port mirrors that by
//!   deriving `Clone` but not exposing a public move-out API beyond normal
//!   Rust move semantics (which are always safe/available in Rust, unlike
//!   C++, so no `= delete` equivalent is needed there).

use std::hash::{Hash, Hasher};

/// Sentinel returned by [`SimpleArray::find`] and friends when no element matches.
pub const INVALID_INDEX: u32 = u32::MAX;

/// Faithful port of `Kyty::Core::SimpleArray<T>`, backed by `Vec<T>`.
#[derive(Debug, Clone, Default)]
pub struct SimpleArray<T> {
    values: Vec<T>,
    /// Cached content hash; `None` means "needs recompute", mirroring the
    /// C++ `m_hash == 0` sentinel.
    hash_cache: std::cell::Cell<Option<u64>>,
}

impl<T> SimpleArray<T> {
    /// `SimpleArray()` — empty array.
    pub fn new() -> Self {
        Self { values: Vec::new(), hash_cache: std::cell::Cell::new(None) }
    }

    /// `Size()`.
    pub fn size(&self) -> u32 {
        self.values.len() as u32
    }

    /// `Capacity()`.
    pub fn capacity(&self) -> u32 {
        self.values.capacity() as u32
    }

    /// `Clear()`.
    pub fn clear(&mut self) {
        self.values.clear();
        self.hash_cache.set(None);
    }

    /// `Free()` — in Kyty this also releases the backing allocation;
    /// `Vec::clear` retains capacity, so mirror `Free`'s "empty and give
    /// memory back" intent with `Vec::clear` + `shrink_to_fit`.
    pub fn free(&mut self) {
        self.values.clear();
        self.values.shrink_to_fit();
        self.hash_cache.set(None);
    }

    /// `Expand(num)` — reserve capacity for `num` more elements.
    pub fn expand(&mut self, num: u32) {
        self.values.reserve(num as usize);
    }

    /// `Add(const T&)` / `Add(T&&)`.
    pub fn add(&mut self, val: T) {
        self.values.push(val);
        self.hash_cache.set(None);
    }

    /// `Add(const T* val, uint32_t num)`.
    pub fn add_slice(&mut self, val: &[T])
    where
        T: Clone,
    {
        self.values.extend_from_slice(val);
        self.hash_cache.set(None);
    }

    /// `InsertAt(index, val)`.
    pub fn insert_at(&mut self, index: u32, val: T) {
        if self.index_valid(index) {
            self.values.insert(index as usize, val);
        } else {
            self.values.push(val);
        }
        self.hash_cache.set(None);
    }

    /// `Find(const T&)`.
    pub fn find(&self, t: &T) -> u32
    where
        T: PartialEq,
    {
        self.values
            .iter()
            .position(|v| v == t)
            .map(|i| i as u32)
            .unwrap_or(INVALID_INDEX)
    }

    /// `Find(const T&, OP&& op_eq)` and the `T2`-typed overload.
    pub fn find_by<F: Fn(&T) -> bool>(&self, op_eq: F) -> u32 {
        self.values
            .iter()
            .position(op_eq)
            .map(|i| i as u32)
            .unwrap_or(INVALID_INDEX)
    }

    /// `FindAll(t, op_eq, ret)` — collects all matching indices.
    pub fn find_all_by<F: Fn(&T) -> bool>(&self, op_eq: F, ret: &mut SimpleArray<u32>) {
        for (index, v) in self.values.iter().enumerate() {
            if op_eq(v) {
                ret.add(index as u32);
            }
        }
    }

    /// `Remove(const T&)`.
    pub fn remove(&mut self, t: &T) -> bool
    where
        T: PartialEq,
    {
        let index = self.find(t);
        if index == INVALID_INDEX {
            return false;
        }
        self.remove_at(index, 1)
    }

    /// `RemoveAt(index, count = 1)`.
    pub fn remove_at(&mut self, index: u32, count: u32) -> bool {
        let size = self.values.len() as u32;
        if index >= size {
            return false;
        }
        let count = if index + count > size { size - index } else { count };
        let start = index as usize;
        let end = (index + count) as usize;
        self.values.drain(start..end);
        self.hash_cache.set(None);
        true
    }

    /// `IndexValid(index)`.
    pub fn index_valid(&self, index: u32) -> bool {
        index < self.size()
    }

    /// `operator[](uint32_t)` (mutable form).
    ///
    /// Kept as an inherent method rather than a `std::ops::IndexMut` impl: it
    /// takes a `u32` (matching Kyty's `operator[]`) and must invalidate the
    /// lazy hash cache on mutable access, which the trait signature cannot do.
    #[allow(clippy::should_implement_trait)]
    pub fn index_mut(&mut self, index: u32) -> &mut T {
        crate::exit_if!(index >= self.size());
        self.hash_cache.set(None);
        &mut self.values[index as usize]
    }

    /// `operator[](uint32_t) const`.
    ///
    /// Inherent method rather than a `std::ops::Index` impl to mirror Kyty's
    /// `u32`-typed `operator[]` and to pair with the cache-invalidating
    /// [`Self::index_mut`].
    #[allow(clippy::should_implement_trait)]
    pub fn index(&self, index: u32) -> &T {
        crate::exit_if!(index >= self.size());
        &self.values[index as usize]
    }

    /// `At(index) const`.
    pub fn at(&self, index: u32) -> &T {
        crate::exit_if!(index >= self.size());
        &self.values[index as usize]
    }

    /// `GetData()` (mutable).
    pub fn get_data_mut(&mut self) -> &mut [T] {
        self.hash_cache.set(None);
        &mut self.values
    }

    /// `GetData() const` / `GetDataConst() const`.
    pub fn get_data(&self) -> &[T] {
        &self.values
    }

    /// `Sort()` — sorts by `Ord`, matching the C++ default `operator<` sort.
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.values.sort();
        self.hash_cache.set(None);
    }

    /// `Sort(SortCompareFunc)` / `Sort(OP&& comp_func)` — sort with a
    /// strict-weak-order predicate `comp_func(a, b)` meaning "a < b".
    pub fn sort_by<F: FnMut(&T, &T) -> bool>(&mut self, mut comp_func: F) {
        self.values
            .sort_by(|a, b| if comp_func(a, b) { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater });
        self.hash_cache.set(None);
    }

    /// `begin()`/`end()` (mutable iteration).
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.hash_cache.set(None);
        self.values.iter_mut()
    }

    /// `begin() const`/`end() const` (immutable iteration).
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    /// `Hash()` — lazily computed, cached content hash. Uses Rust's
    /// `Hash`/`DefaultHasher` rather than Kyty's raw-byte `Core::hash()`.
    pub fn hash(&self) -> u64
    where
        T: Hash,
    {
        if let Some(h) = self.hash_cache.get() {
            return h;
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for v in &self.values {
            v.hash(&mut hasher);
        }
        let h = hasher.finish();
        self.hash_cache.set(Some(h));
        h
    }
}

impl<T> std::ops::Index<u32> for SimpleArray<T> {
    type Output = T;
    fn index(&self, index: u32) -> &T {
        SimpleArray::index(self, index)
    }
}

impl<T> std::ops::IndexMut<u32> for SimpleArray<T> {
    fn index_mut(&mut self, index: u32) -> &mut T {
        SimpleArray::index_mut(self, index)
    }
}

impl<T: PartialEq> PartialEq for SimpleArray<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<T> FromIterator<T> for SimpleArray<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self { values: Vec::from_iter(iter), hash_cache: std::cell::Cell::new(None) }
    }
}

impl<T> IntoIterator for SimpleArray<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<T, const N: usize> From<[T; N]> for SimpleArray<T> {
    fn from(list: [T; N]) -> Self {
        Self { values: Vec::from(list), hash_cache: std::cell::Cell::new(None) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_array_is_empty() {
        let a: SimpleArray<i32> = SimpleArray::new();
        assert_eq!(a.size(), 0);
    }

    #[test]
    fn add_and_index() {
        let mut a: SimpleArray<i32> = SimpleArray::new();
        a.add(10);
        a.add(20);
        a.add(30);
        assert_eq!(a.size(), 3);
        assert_eq!(*a.index(0), 10);
        assert_eq!(*a.index(2), 30);
        assert_eq!(a[1], 20);
    }

    #[test]
    fn add_slice() {
        let mut a: SimpleArray<i32> = SimpleArray::new();
        a.add_slice(&[1, 2, 3]);
        assert_eq!(a.size(), 3);
        assert_eq!(a.get_data(), &[1, 2, 3]);
    }

    #[test]
    fn insert_at_shifts_elements() {
        let mut a = SimpleArray::from([1, 2, 4]);
        a.insert_at(2, 3);
        assert_eq!(a.get_data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn insert_at_out_of_range_appends() {
        let mut a = SimpleArray::from([1, 2]);
        a.insert_at(100, 3);
        assert_eq!(a.get_data(), &[1, 2, 3]);
    }

    #[test]
    fn find_returns_invalid_index_when_missing() {
        let a = SimpleArray::from([1, 2, 3]);
        assert_eq!(a.find(&2), 1);
        assert_eq!(a.find(&99), INVALID_INDEX);
    }

    #[test]
    fn find_by_predicate() {
        let a = SimpleArray::from([1, 2, 3, 4]);
        assert_eq!(a.find_by(|v| *v % 2 == 0), 1);
    }

    #[test]
    fn find_all_by_collects_indices() {
        let a = SimpleArray::from([1, 2, 3, 4, 6]);
        let mut out: SimpleArray<u32> = SimpleArray::new();
        a.find_all_by(|v| *v % 2 == 0, &mut out);
        assert_eq!(out.get_data(), &[1, 3, 4]);
    }

    #[test]
    fn remove_removes_first_match() {
        let mut a = SimpleArray::from([1, 2, 3]);
        assert!(a.remove(&2));
        assert_eq!(a.get_data(), &[1, 3]);
        assert!(!a.remove(&999));
    }

    #[test]
    fn remove_at_with_count() {
        let mut a = SimpleArray::from([1, 2, 3, 4, 5]);
        assert!(a.remove_at(1, 2));
        assert_eq!(a.get_data(), &[1, 4, 5]);
    }

    #[test]
    fn remove_at_out_of_range_fails() {
        let mut a = SimpleArray::from([1, 2, 3]);
        assert!(!a.remove_at(10, 1));
    }

    #[test]
    fn index_valid_checks_bounds() {
        let a = SimpleArray::from([1, 2, 3]);
        assert!(a.index_valid(2));
        assert!(!a.index_valid(3));
    }

    #[test]
    #[should_panic]
    fn index_out_of_range_panics() {
        let a: SimpleArray<i32> = SimpleArray::from([1, 2]);
        let _ = a.index(5);
    }

    #[test]
    fn sort_default_order() {
        let mut a = SimpleArray::from([3, 1, 2]);
        a.sort();
        assert_eq!(a.get_data(), &[1, 2, 3]);
    }

    #[test]
    fn sort_by_custom_predicate_descending() {
        let mut a = SimpleArray::from([1, 3, 2]);
        a.sort_by(|x, y| x > y);
        assert_eq!(a.get_data(), &[3, 2, 1]);
    }

    #[test]
    fn equality_by_contents() {
        let a = SimpleArray::from([1, 2, 3]);
        let b = SimpleArray::from([1, 2, 3]);
        let c = SimpleArray::from([1, 2, 4]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn hash_is_stable_and_content_based() {
        let a = SimpleArray::from([1, 2, 3]);
        let b = SimpleArray::from([1, 2, 3]);
        let c = SimpleArray::from([1, 2, 4]);
        assert_eq!(a.hash(), a.hash());
        assert_eq!(a.hash(), b.hash());
        assert_ne!(a.hash(), c.hash());
    }

    #[test]
    fn clear_empties_but_keeps_type() {
        let mut a = SimpleArray::from([1, 2, 3]);
        a.clear();
        assert_eq!(a.size(), 0);
    }

    #[test]
    fn iteration_matches_contents() {
        let a = SimpleArray::from([1, 2, 3]);
        let collected: Vec<i32> = a.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn clone_is_independent() {
        let mut a = SimpleArray::from([1, 2, 3]);
        let b = a.clone();
        a.add(4);
        assert_eq!(a.size(), 4);
        assert_eq!(b.size(), 3);
    }
}
