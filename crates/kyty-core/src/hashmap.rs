//! Port of Kyty's `Kyty::Core::Hashmap<K, V>` / `HashmapBase` / `HashmapPrivate`
//! (`reference/kyty/source/include/Kyty/Core/Hashmap.h` +
//! `reference/kyty/source/lib/Core/src/Hashmap.cpp`).
//!
//! In C++, `Hashmap<K, V>` is a type-erased wrapper: `HashmapBase` stores
//! `void*` blobs and per-key `hash_calc`/`hash_key_equals`/copy/free function
//! pointers, while `HashmapPrivate` implements a hand-rolled open-hashing
//! table (fixed-size `uint8_t key[32]` / `value[16]` byte arrays inside each
//! `Entry`, manual `new`/placement-new/`Delete` bucket-chain management,
//! power-of-two bucket-count growth at a 3/4 load factor, and a stateful
//! `loop_index`/`loop_entry` cursor for `Start`/`End`/`Next`/`Value`/`Key`
//! iteration, driven by the `FOR_HASH` macro).
//!
//! Std mapping: `Hashmap<K, V>` -> `std::collections::HashMap<K, V>`. Rust's
//! `HashMap` already provides a safe, correct, generically-hashed table, so
//! this type is a thin wrapper over it that exposes Kyty's method names
//! (`Put`/`Get`/`Find`/`Contains`/`Remove`/`Clear`/`Size`/`GetOrPutDef`/
//! `ForEach`/the `Start`/`End`/`Next`/`Value`/`Key` iteration cursor/
//! `CollisionsCount`/`operator==`) rather than a from-scratch bucket-chain
//! reimplementation. `HashmapBase`'s `void*` + function-pointer type erasure
//! and `HashmapPrivate`'s manual entry storage are not ported: Rust generics
//! (`K: Eq + Hash + Clone`) give the same "one implementation, many key/value
//! types" behavior without unsafe casts or placement-new.
//!
//! `Start`/`End`/`Next`/`Value`/`Key` are `const` (`&self`) iteration methods
//! in Kyty, driven by mutable cursor state hidden behind `mutable` members.
//! Here that cursor (a snapshot of the current keys plus a position) lives in
//! `Cell`/`RefCell` so the same `&self` signatures work. `Key()` returns
//! `const K&` in C++; here it returns an owned `K` (`K: Clone`) since the
//! snapshot, not the live map, owns the traversed key list — cheaper to clone
//! than to fight the borrow checker for a reference whose target may be
//! removed mid-iteration. `Value()` still borrows directly from the backing
//! map, matching `const V&`.
//!
//! `CollisionsCount()` reported bucket-chain collisions in Kyty's hand-rolled
//! table; `std::collections::HashMap` does not expose its internal bucket
//! layout, so this is a diagnostic-only method here and always returns `0`
//! (documented behavior difference, not a semantic one — nothing in Kyty's
//! public contract depends on the actual count besides logging/metrics).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::Hash;

/// Kyty's `Hashmap<K, V>`: a thin wrapper around `std::collections::HashMap<K, V>`
/// exposing Kyty's method names. Kyty's `Hashmap` is `KYTY_CLASS_NO_COPY`
/// (non-copyable); this type intentionally does not derive `Clone`.
pub struct Hashmap<K, V> {
    map: HashMap<K, V>,
    // Cursor state for the `Start`/`End`/`Next`/`Value`/`Key` iteration API
    // (Kyty's `loop_index`/`loop_entry`, `mutable` in C++, hence `Cell`/
    // `RefCell` here so iteration works through `&self`).
    iter_keys: RefCell<Vec<K>>,
    iter_pos: Cell<usize>,
}

impl<K, V> Hashmap<K, V> {
    /// `Hashmap()`: empty map.
    pub fn new() -> Self {
        Self { map: HashMap::new(), iter_keys: RefCell::new(Vec::new()), iter_pos: Cell::new(0) }
    }

    /// `Size()`.
    pub fn size(&self) -> u32 {
        self.map.len() as u32
    }
}

impl<K, V> Default for Hashmap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Clone, V> Hashmap<K, V> {
    /// `Clear()`.
    pub fn clear(&mut self) {
        self.map.clear();
        self.iter_keys.borrow_mut().clear();
        self.iter_pos.set(0);
    }

    /// `Put(const K& key, const V& value)`: insert, or overwrite the value of
    /// an existing key.
    pub fn put(&mut self, key: K, value: V) {
        self.map.insert(key, value);
    }

    /// `Find(const K& key)`: `const V*`, `nullptr` if absent.
    pub fn find(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// `Contains(const K& key)`.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// `Remove(const K& key)`.
    pub fn remove(&mut self, key: &K) {
        self.map.remove(key);
    }

    /// `GetOrPutDef(const K& key, const V& def)`: return a mutable reference
    /// to the value for `key`, inserting `def` first if it was absent.
    pub fn get_or_put_def(&mut self, key: &K, def: V) -> &mut V {
        self.map.entry(key.clone()).or_insert(def)
    }

    /// `operator[](const K& key)`: like `GetOrPutDef` but with `V::default()`
    /// (Kyty constructs `V def {};`) as the inserted default.
    pub fn operator_square_brackets(&mut self, key: &K) -> &mut V
    where
        V: Default,
    {
        self.map.entry(key.clone()).or_default()
    }

    /// `Start()`: begin iteration (`FOR_HASH`). Snapshots the current keys so
    /// `Next`/`End`/`Value`/`Key` behave consistently even if the map is not
    /// mutated during the loop (matching Kyty's live bucket-chain cursor for
    /// the read-only-during-iteration case).
    pub fn start(&self) {
        *self.iter_keys.borrow_mut() = self.map.keys().cloned().collect();
        self.iter_pos.set(0);
    }

    /// `End()`.
    pub fn end(&self) -> bool {
        self.iter_pos.get() >= self.iter_keys.borrow().len()
    }

    /// `Next()`.
    pub fn next(&self) {
        self.iter_pos.set(self.iter_pos.get() + 1);
    }

    /// `Key()`: `const K&` in C++; returns an owned clone here (see module
    /// doc comment).
    pub fn key(&self) -> K {
        self.iter_keys.borrow()[self.iter_pos.get()].clone()
    }

    /// `Value()`: `const V&`.
    pub fn value(&self) -> &V {
        let pos = self.iter_pos.get();
        let key = self.iter_keys.borrow()[pos].clone();
        self.map.get(&key).expect("Hashmap::value(): iterator key missing from map (removed during iteration?)")
    }

    /// `ForEach(callback, arg)`: visits every (key, value) pair via the
    /// `Start`/`End`/`Next`/`Key`/`Value` cursor, stopping early if
    /// `callback` returns `false` (Kyty's `hash_callback_func_t` contract).
    pub fn for_each<F>(&self, mut callback: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        self.start();
        while !self.end() {
            let k = self.key();
            if !callback(&k, self.value()) {
                break;
            }
            self.next();
        }
    }

    /// `CollisionsCount()`: always `0` here (see module doc comment).
    pub fn collisions_count(&self) -> u32 {
        0
    }
}

impl<K: Eq + Hash + Clone, V: Clone> Hashmap<K, V> {
    /// `Get(const K& key, const V& default_value = V())`.
    pub fn get(&self, key: &K, default_value: V) -> V {
        self.map.get(key).cloned().unwrap_or(default_value)
    }
}

impl<K: Eq + Hash + Clone, V: Default + Clone> Hashmap<K, V> {
    /// `Get(const K& key)` with the default-value parameter omitted
    /// (`default_value = V()`).
    pub fn get_or_default(&self, key: &K) -> V {
        self.map.get(key).cloned().unwrap_or_default()
    }
}

impl<K: Eq + Hash + Clone, V: PartialEq> PartialEq for Hashmap<K, V> {
    /// `operator==`: same size, and every key in `self` maps to the same
    /// value in `other`.
    fn eq(&self, other: &Self) -> bool {
        if self.size() != other.size() {
            return false;
        }
        self.map.iter().all(|(k, v)| other.find(k) == Some(v))
    }
}

impl<K: Eq + Hash + Clone, V: Eq> Eq for Hashmap<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn new_map_is_empty() {
        let m: Hashmap<i32, i32> = Hashmap::new();
        assert_eq!(m.size(), 0);
        assert!(!m.contains(&1));
        assert_eq!(m.find(&1), None);
    }

    #[test]
    fn default_matches_new() {
        let m: Hashmap<i32, i32> = Hashmap::default();
        assert_eq!(m.size(), 0);
    }

    #[test]
    fn put_get_find_contains() {
        let mut m: Hashmap<String, i32> = Hashmap::new();
        m.put("a".to_string(), 1);
        m.put("b".to_string(), 2);

        assert_eq!(m.size(), 2);
        assert!(m.contains(&"a".to_string()));
        assert_eq!(m.find(&"a".to_string()), Some(&1));
        assert_eq!(m.get(&"b".to_string(), -1), 2);
        assert_eq!(m.get(&"missing".to_string(), -1), -1);
        assert_eq!(m.get_or_default(&"missing".to_string()), 0);
    }

    #[test]
    fn put_overwrites_existing_key() {
        let mut m: Hashmap<i32, i32> = Hashmap::new();
        m.put(1, 10);
        m.put(1, 20);
        assert_eq!(m.size(), 1);
        assert_eq!(m.find(&1), Some(&20));
    }

    #[test]
    fn remove_deletes_key() {
        let mut m: Hashmap<i32, i32> = Hashmap::new();
        m.put(1, 10);
        m.put(2, 20);
        m.remove(&1);
        assert_eq!(m.size(), 1);
        assert!(!m.contains(&1));
        assert!(m.contains(&2));

        // Removing an absent key is a no-op, not an error.
        m.remove(&999);
        assert_eq!(m.size(), 1);
    }

    #[test]
    fn clear_empties_map_and_resets_iteration() {
        let mut m: Hashmap<i32, i32> = Hashmap::new();
        m.put(1, 10);
        m.put(2, 20);
        m.clear();
        assert_eq!(m.size(), 0);
        assert_eq!(m.find(&1), None);
        assert!(m.end());
    }

    #[test]
    fn get_or_put_def_inserts_once_then_returns_existing() {
        let mut m: Hashmap<i32, i32> = Hashmap::new();
        {
            let v = m.get_or_put_def(&1, 42);
            assert_eq!(*v, 42);
            *v += 1;
        }
        assert_eq!(m.find(&1), Some(&43));

        // Second call with a different default must not overwrite.
        let v2 = m.get_or_put_def(&1, 999);
        assert_eq!(*v2, 43);
    }

    #[test]
    fn operator_square_brackets_inserts_default_and_is_mutable() {
        let mut m: Hashmap<i32, i32> = Hashmap::new();
        *m.operator_square_brackets(&5) += 7;
        assert_eq!(m.find(&5), Some(&7));

        *m.operator_square_brackets(&5) += 1;
        assert_eq!(m.find(&5), Some(&8));
    }

    #[test]
    fn iteration_visits_every_entry_exactly_once() {
        let mut m: Hashmap<i32, i32> = Hashmap::new();
        for i in 0..10 {
            m.put(i, i * i);
        }

        let mut seen = HashSet::new();
        m.start();
        while !m.end() {
            let k = m.key();
            let v = *m.value();
            assert_eq!(v, k * k);
            assert!(seen.insert(k), "key {k} visited twice");
            m.next();
        }
        assert_eq!(seen.len(), 10);
    }

    #[test]
    fn for_each_visits_all_pairs() {
        let mut m: Hashmap<i32, i32> = Hashmap::new();
        for i in 0..5 {
            m.put(i, i * 10);
        }

        let mut sum = 0;
        m.for_each(|_k, v| {
            sum += v;
            true
        });
        assert_eq!(sum, 10 + 20 + 30 + 40);
    }

    #[test]
    fn for_each_stops_early_when_callback_returns_false() {
        let mut m: Hashmap<i32, i32> = Hashmap::new();
        for i in 0..10 {
            m.put(i, i);
        }

        let mut visited = 0;
        m.for_each(|_k, _v| {
            visited += 1;
            visited < 3
        });
        assert_eq!(visited, 3);
    }

    #[test]
    fn empty_map_iteration_ends_immediately() {
        let m: Hashmap<i32, i32> = Hashmap::new();
        m.start();
        assert!(m.end());
    }

    #[test]
    fn collisions_count_is_zero() {
        let m: Hashmap<i32, i32> = Hashmap::new();
        assert_eq!(m.collisions_count(), 0);
    }

    #[test]
    fn equality_ignores_insertion_order() {
        let mut a: Hashmap<i32, i32> = Hashmap::new();
        a.put(1, 10);
        a.put(2, 20);

        let mut b: Hashmap<i32, i32> = Hashmap::new();
        b.put(2, 20);
        b.put(1, 10);

        assert!(a == b);
        assert!(!(a != b));
    }

    #[test]
    fn equality_detects_size_and_value_differences() {
        let mut a: Hashmap<i32, i32> = Hashmap::new();
        a.put(1, 10);

        let mut b: Hashmap<i32, i32> = Hashmap::new();
        b.put(1, 10);
        b.put(2, 20);
        assert!(a != b);

        let mut c: Hashmap<i32, i32> = Hashmap::new();
        c.put(1, 999);
        assert!(a != c);
    }
}
