//! Port of Kyty's `Kyty::Vector<T>` / `Kyty::Core::VectorBase<T, A>`
//! (`reference/kyty/source/include/Kyty/Core/Vector.h`).
//!
//! In C++, `Vector<T>` is Kyty's own growable-array re-implementation: a
//! thin subclass of `VectorBase<T, SimpleArray<T>>` that adds copy-on-write
//! (refcounted, shared `SimpleArray<T>*` storage, deep-copied lazily on the
//! first mutating access) on top of a hand-rolled dynamic array
//! (`SimpleArray<T>`, see `Kyty/Core/SimpleArray.h`) that manually manages
//! allocation, growth, element construction/destruction and a custom
//! quicksort.
//!
//! Std mapping: `Vector<T>` -> `Vec<T>`. Rust's `Vec<T>` already provides
//! safe, correct growable-array storage with value semantics; the
//! copy-on-write scheme in `VectorBase` exists only to make C++ value-copies
//! of `Vector` cheap until a mutation happens. `Vec::clone()` deep-copies
//! eagerly instead, which is observably equivalent for a value type (same
//! final contents, same mutation semantics — just without the laziness
//! optimization) and needs no unsafe code, no manual refcounting and no
//! custom allocator calls. `SimpleArray<T>` itself is therefore not ported
//! as a separate type: its behavior (growth, element storage, sorting,
//! find/remove/insert) is what `Vec<T>` + slice methods already provide, and
//! `Vector<T>` here wraps `Vec<T>` directly rather than wrapping a
//! from-scratch reimplementation of it.
//!
//! Kyty's custom quicksort (`SimpleArray::sort` / `sort_with_compare_func`)
//! is ported as calls to `slice::sort_by`/`sort_unstable_by`: the sorting
//! *algorithm* differs, but the observable contract (`Sort()` leaves the
//! vector in ascending order per the given comparator) is preserved, which
//! is what "faithful port" means for an algorithm-replaceable helper method.
//! `SortSwapFunc` (a custom swap hook, used historically so callers could
//! keep a side index in sync with element swaps) is ported as
//! `sort_with_swap`, taking a safe `FnMut(&mut [T], usize, usize)` closure
//! instead of a C `fn(T*, i32, i32, void*)` + `void*` argument pointer — the
//! `void*` user-data parameter is unnecessary in Rust because the closure
//! can simply capture what it needs.

use std::cmp::Ordering;

/// Sentinel returned by `find`/`Find` when no element matches
/// (`VectorBase::INVALID_INDEX`, `static_cast<uint32_t>(-1)`).
pub const INVALID_INDEX: u32 = u32::MAX;

/// Kyty's `Vector<T>`: a thin wrapper around `Vec<T>` exposing Kyty's
/// `VectorBase<T, SimpleArray<T>>` method names.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T> Vector<T> {
    /// `VectorBase()`: empty vector.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// `explicit VectorBase(uint32_t size, bool ctor = true)`: preallocate
    /// `size` default-constructed elements. Kyty's `ctor = false` variant
    /// (reserve capacity without constructing elements) has no safe
    /// equivalent in Rust and is not exposed; use `with_capacity` instead.
    pub fn with_size(size: u32) -> Self
    where
        T: Default,
    {
        let mut data = Vec::with_capacity(size as usize);
        data.resize_with(size as usize, T::default);
        Self { data }
    }

    /// Reserve capacity for at least `capacity` elements without
    /// constructing them (Kyty's `VectorBase(size, /*ctor=*/false)`).
    pub fn with_capacity(capacity: u32) -> Self {
        Self {
            data: Vec::with_capacity(capacity as usize),
        }
    }

    /// `VectorBase(std::initializer_list<T> list)`.
    pub fn from_list<const N: usize>(list: [T; N]) -> Self {
        Self {
            data: Vec::from(list),
        }
    }

    /// `Size()`.
    pub fn size(&self) -> u32 {
        self.data.len() as u32
    }

    /// `Capacity()`.
    pub fn capacity(&self) -> u32 {
        self.data.capacity() as u32
    }

    /// `IsEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `Clear()`: destroys all elements, keeps the allocation.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// `Free()`: destroys all elements and releases the allocation.
    pub fn free(&mut self) {
        self.data = Vec::new();
    }

    /// `Expand(uint32_t num)`: ensure room for `num` more elements.
    pub fn expand(&mut self, num: u32) {
        self.data.reserve(num as usize);
    }

    /// `Add(const T&)` / `Add(T&&)`: append one element.
    pub fn add(&mut self, val: T) {
        self.data.push(val);
    }

    /// `Add(const T* val, uint32_t num)`: append `num` elements copied from
    /// `val`.
    pub fn add_slice(&mut self, val: &[T])
    where
        T: Clone,
    {
        self.data.extend_from_slice(val);
    }

    /// `Add(const VectorType&)`: append the contents of another `Vector`.
    pub fn add_vector(&mut self, other: &Vector<T>)
    where
        T: Clone,
    {
        self.data.extend_from_slice(&other.data);
    }

    /// `InsertAt(uint32_t index, const T&)` / `InsertAt(uint32_t, T&&)`.
    pub fn insert_at(&mut self, index: u32, val: T) {
        self.data.insert(index as usize, val);
    }

    /// `Find(const T&)`: index of the first element equal to `t`, or
    /// `INVALID_INDEX`.
    pub fn find(&self, t: &T) -> u32
    where
        T: PartialEq,
    {
        match self.data.iter().position(|x| x == t) {
            Some(i) => i as u32,
            None => INVALID_INDEX,
        }
    }

    /// `Find(const T&, OP&& op_eq)`: index of the first element for which
    /// `op_eq(element, t)` holds, or `INVALID_INDEX`.
    pub fn find_by<F>(&self, t: &T, mut op_eq: F) -> u32
    where
        F: FnMut(&T, &T) -> bool,
    {
        match self.data.iter().position(|x| op_eq(x, t)) {
            Some(i) => i as u32,
            None => INVALID_INDEX,
        }
    }

    /// `Contains(const T&)`.
    pub fn contains(&self, t: &T) -> bool
    where
        T: PartialEq,
    {
        self.find(t) != INVALID_INDEX
    }

    /// `Contains(const T2&, OP&& op_eq)`.
    pub fn contains_by<F>(&self, t: &T, op_eq: F) -> bool
    where
        F: FnMut(&T, &T) -> bool,
    {
        self.find_by(t, op_eq) != INVALID_INDEX
    }

    /// `Remove(const T&)`: removes the first element equal to `t`, if any.
    /// Returns whether an element was removed.
    pub fn remove(&mut self, t: &T) -> bool
    where
        T: PartialEq,
    {
        match self.data.iter().position(|x| x == t) {
            Some(i) => {
                self.data.remove(i);
                true
            }
            None => false,
        }
    }

    /// `RemoveAt(uint32_t index, uint32_t count = 1)`: removes `count`
    /// elements starting at `index`. Returns whether the range was valid.
    pub fn remove_at(&mut self, index: u32, count: u32) -> bool {
        let index = index as usize;
        let count = count as usize;
        if count == 0
            || index
                .checked_add(count)
                .is_none_or(|end| end > self.data.len())
        {
            return false;
        }
        self.data.drain(index..index + count);
        true
    }

    /// `IndexValid(uint32_t index)`.
    pub fn index_valid(&self, index: u32) -> bool {
        (index as usize) < self.data.len()
    }

    /// `At(uint32_t index)` (const access; panics like Kyty's `EXIT_IF` on
    /// out-of-range, via normal slice indexing).
    pub fn at(&self, index: u32) -> &T {
        &self.data[index as usize]
    }

    /// `GetData()` (mutable slice view).
    pub fn get_data(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// `GetDataConst()` / const `GetData()`.
    pub fn get_data_const(&self) -> &[T] {
        &self.data
    }

    /// `Memset(int c)`: sets every byte of the backing storage to `c`, as
    /// the C++ `memset` does for POD element types. Bounded on
    /// [`bytemuck::Pod`] — Kyty only ever `Memset`s POD `Vector`s (bytes,
    /// trivial structs), and `Pod` is exactly the Rust guarantee that makes a
    /// raw byte fill sound: every bit pattern is a valid `T` and `T` has no
    /// padding/uninitialized bytes. (`T: Copy` alone is *not* enough — e.g.
    /// `bool`/fieldless enums are `Copy` but only some bit patterns are valid,
    /// so a byte fill could produce an invalid value = UB.)
    pub fn memset(&mut self, c: u8)
    where
        T: bytemuck::Pod,
    {
        // `bytemuck::fill_zeroes`-style, but for an arbitrary byte value:
        // reinterpret the initialized element storage as bytes (sound because
        // `T: Pod`) and fill. No unsafe needed — `bytemuck::cast_slice_mut`
        // discharges the reinterpretation safely.
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut self.data);
        bytes.fill(c);
    }

    /// `Sort()`: ascending sort using `T`'s natural order.
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.data.sort();
    }

    /// `Sort(SortCompareFunc comp_func)` / templated `Sort(OP&& comp_func)`:
    /// sort using a "less-than" predicate, matching Kyty's
    /// `bool (*)(const T&, const T&)` comparator convention.
    pub fn sort_by<F>(&mut self, mut less: F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        self.data.sort_by(|a, b| {
            if less(a, b) {
                Ordering::Less
            } else if less(b, a) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
    }

    /// `Sort(SortSwapFunc swap_func, void* swap_arg = nullptr)`: sort using
    /// natural order (`T: Ord`), but perform swaps through the caller's
    /// `swap_func` instead of Rust's default in-place swap, so a caller can
    /// keep external state (e.g. a parallel index array) synchronized with
    /// element moves. Takes a plain closure instead of Kyty's
    /// `fn(T*, i32, i32, void*) + void* arg` pair since a Rust closure can
    /// simply capture whatever state it needs.
    pub fn sort_with_swap<F>(&mut self, mut swap_func: F)
    where
        T: Ord,
        F: FnMut(&mut [T], usize, usize),
    {
        // Simple insertion sort expressed purely in terms of adjacent
        // swaps, so every reordering goes through `swap_func`.
        let len = self.data.len();
        for i in 1..len {
            let mut j = i;
            while j > 0 && self.data[j] < self.data[j - 1] {
                swap_func(&mut self.data, j - 1, j);
                j -= 1;
            }
        }
    }

    /// `begin()`/`end()` (mutable iteration).
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// `cbegin()`/`cend()` (const iteration).
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Unwrap into the underlying `Vec<T>`.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> std::ops::Index<u32> for Vector<T> {
    type Output = T;

    /// `operator[](uint32_t index) const`.
    fn index(&self, index: u32) -> &T {
        &self.data[index as usize]
    }
}

impl<T> std::ops::IndexMut<u32> for Vector<T> {
    /// `operator[](uint32_t index)`.
    fn index_mut(&mut self, index: u32) -> &mut T {
        &mut self.data[index as usize]
    }
}

impl<T> From<Vec<T>> for Vector<T> {
    fn from(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            data: Vec::from_iter(iter),
        }
    }
}

impl<T> IntoIterator for Vector<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Vector<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let v: Vector<i32> = Vector::new();
        assert_eq!(v.size(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn with_size_default_fills() {
        let v: Vector<i32> = Vector::with_size(3);
        assert_eq!(v.size(), 3);
        assert_eq!(v.at(0), &0);
        assert_eq!(v.at(2), &0);
    }

    #[test]
    fn add_and_index() {
        let mut v: Vector<i32> = Vector::new();
        v.add(1);
        v.add(2);
        v.add(3);
        assert_eq!(v.size(), 3);
        assert_eq!(v[0], 1);
        assert_eq!(v[2], 3);
        v[1] = 42;
        assert_eq!(*v.at(1), 42);
    }

    #[test]
    fn add_slice_and_add_vector() {
        let mut v: Vector<i32> = Vector::new();
        v.add_slice(&[1, 2, 3]);
        assert_eq!(v.size(), 3);

        let other = Vector::from(vec![4, 5]);
        v.add_vector(&other);
        assert_eq!(v.size(), 5);
        assert_eq!(v.get_data_const(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn insert_at() {
        let mut v: Vector<i32> = Vector::from(vec![1, 2, 4]);
        v.insert_at(2, 3);
        assert_eq!(v.get_data_const(), &[1, 2, 3, 4]);
    }

    #[test]
    fn find_and_contains() {
        let v: Vector<i32> = Vector::from(vec![10, 20, 30]);
        assert_eq!(v.find(&20), 1);
        assert_eq!(v.find(&99), INVALID_INDEX);
        assert!(v.contains(&30));
        assert!(!v.contains(&99));
    }

    #[test]
    fn find_by_predicate() {
        let v: Vector<i32> = Vector::from(vec![10, 20, 30]);
        // op_eq(element, t): find element whose "double" equals t.
        let idx = v.find_by(&40, |elem, t| elem * 2 == *t);
        assert_eq!(idx, 1);
    }

    #[test]
    fn remove_and_remove_at() {
        let mut v: Vector<i32> = Vector::from(vec![1, 2, 3, 4, 5]);
        assert!(v.remove(&3));
        assert_eq!(v.get_data_const(), &[1, 2, 4, 5]);
        assert!(!v.remove(&99));

        assert!(v.remove_at(1, 2));
        assert_eq!(v.get_data_const(), &[1, 5]);
        assert!(!v.remove_at(0, 10));
    }

    #[test]
    fn index_valid() {
        let v: Vector<i32> = Vector::from(vec![1, 2]);
        assert!(v.index_valid(0));
        assert!(v.index_valid(1));
        assert!(!v.index_valid(2));
    }

    #[test]
    fn clear_and_free() {
        let mut v: Vector<i32> = Vector::from(vec![1, 2, 3]);
        v.clear();
        assert_eq!(v.size(), 0);

        let mut v2: Vector<i32> = Vector::from(vec![1, 2, 3]);
        v2.free();
        assert_eq!(v2.size(), 0);
    }

    #[test]
    fn sort_default_order() {
        let mut v: Vector<i32> = Vector::from(vec![3, 1, 2]);
        v.sort();
        assert_eq!(v.get_data_const(), &[1, 2, 3]);
    }

    #[test]
    fn sort_by_custom_predicate_descending() {
        let mut v: Vector<i32> = Vector::from(vec![1, 3, 2]);
        v.sort_by(|a, b| a > b);
        assert_eq!(v.get_data_const(), &[3, 2, 1]);
    }

    #[test]
    fn sort_with_swap_tracks_moves() {
        let mut v: Vector<i32> = Vector::from(vec![3, 1, 2]);
        let mut swap_count = 0;
        v.sort_with_swap(|data, i, j| {
            data.swap(i, j);
            swap_count += 1;
        });
        assert_eq!(v.get_data_const(), &[1, 2, 3]);
        assert!(swap_count > 0);
    }

    #[test]
    fn memset_zeroes_bytes() {
        let mut v: Vector<i32> = Vector::from(vec![1, 2, 3]);
        v.memset(0);
        assert_eq!(v.get_data_const(), &[0, 0, 0]);
    }

    #[test]
    fn iteration() {
        let v: Vector<i32> = Vector::from(vec![1, 2, 3]);
        let sum: i32 = v.iter().sum();
        assert_eq!(sum, 6);

        let mut v2: Vector<i32> = Vector::from(vec![1, 2, 3]);
        for x in v2.iter_mut() {
            *x += 1;
        }
        assert_eq!(v2.get_data_const(), &[2, 3, 4]);

        let collected: Vec<i32> = (&v).into_iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn from_iterator_and_into_iterator() {
        let v: Vector<i32> = (1..=3).collect();
        assert_eq!(v.get_data_const(), &[1, 2, 3]);

        let collected: Vec<i32> = v.into_iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn equality() {
        let a: Vector<i32> = Vector::from(vec![1, 2, 3]);
        let b: Vector<i32> = Vector::from(vec![1, 2, 3]);
        let c: Vector<i32> = Vector::from(vec![1, 2, 4]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn with_capacity_reserves_without_elements() {
        let v: Vector<i32> = Vector::with_capacity(10);
        assert_eq!(v.size(), 0);
        assert!(v.capacity() >= 10);
    }
}
