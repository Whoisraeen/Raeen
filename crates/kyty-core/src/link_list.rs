//! Ports `Kyty::Core::List` / `Kyty::Core::ListNode` / `Kyty::Core::ListSet`
//! from `reference/kyty/source/include/Kyty/Core/LinkList.h` (header-only,
//! no matching `.cpp`).
//!
//! Kyty's `List<T>` is an intrusive circular doubly-linked list: every
//! element lives in its own heap-allocated `ListNode<T>`, and callers address
//! elements through a `ListIndex` (a raw `void*` to the node) that stays
//! valid across insertions and is only invalidated when *its own* node is
//! removed. `ListSet<T>` is a subclass that overrides `Add` to be a no-op
//! when the value is already present.
//!
//! Rust mapping: raw self-referential node pointers are unnecessary here —
//! Rust's ownership model can express the same "stable handle until its
//! element is removed" contract safely with a slab. This port stores
//! elements in a `Vec<Option<Node<T>>>` (a std-backed free-list slab) with
//! explicit `head`/`tail` slot links replacing Kyty's circular pointer ring.
//! [`ListIndex`] wraps an `Option<usize>` slot key in place of the C++
//! `void*`, preserving identical semantics (`IndexValid`, `First`/`Last`,
//! `Next`/`Prev`, `Find`, `Remove` returning the following index, …) without
//! unsafe code or manual memory management. [`ListSet`] is ported as a thin
//! wrapper around [`List`] exposing `add_unique` in place of the C++
//! virtual-dispatch override of `Add`.

use std::fmt;

/// Stable handle to an element of a [`List`]. Mirrors Kyty's `ListIndex`
/// (a `void*` to the node); here it is a slot key into the list's slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ListIndex(Option<usize>);

impl ListIndex {
    /// The always-invalid index, matching a Kyty `ListIndex{nullptr}`.
    pub const fn invalid() -> Self {
        ListIndex(None)
    }
}

struct Node<T> {
    value: T,
    next: Option<usize>,
    prev: Option<usize>,
}

/// Faithful port of `Kyty::Core::List<T>`.
pub struct List<T> {
    slots: Vec<Option<Node<T>>>,
    free: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
    size: u32,
}

impl<T> Default for List<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> List<T> {
    /// Kyty: `List()`.
    pub fn new() -> Self {
        List { slots: Vec::new(), free: Vec::new(), head: None, tail: None, size: 0 }
    }

    /// Kyty: `Size()`.
    #[must_use]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Kyty: `Clear()`.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.head = None;
        self.tail = None;
        self.size = 0;
    }

    /// Kyty: `Add(const T& v)`. Appends at the tail (matches Kyty's
    /// `InsertBefore(m_head)` on a circular list, which places the new node
    /// just before the head, i.e. at the end of iteration order).
    pub fn add(&mut self, v: T) -> ListIndex {
        let node = Node { value: v, next: None, prev: self.tail };
        let key = if let Some(k) = self.free.pop() {
            self.slots[k] = Some(node);
            k
        } else {
            self.slots.push(Some(node));
            self.slots.len() - 1
        };

        if let Some(t) = self.tail {
            self.slots[t].as_mut().unwrap().next = Some(key);
        } else {
            self.head = Some(key);
        }
        self.tail = Some(key);
        self.size += 1;

        ListIndex(Some(key))
    }

    /// Kyty: `IndexValid(Index index)`.
    #[must_use]
    pub fn index_valid(&self, index: ListIndex) -> bool {
        index.0.is_some()
    }

    /// Kyty: `First()`.
    #[must_use]
    pub fn first(&self) -> ListIndex {
        ListIndex(self.head)
    }

    /// Kyty: `Last()`.
    #[must_use]
    pub fn last(&self) -> ListIndex {
        ListIndex(self.tail)
    }

    /// Kyty: `Next(Index index)`.
    #[must_use]
    pub fn next(&self, index: ListIndex) -> ListIndex {
        match index.0 {
            Some(k) => ListIndex(self.slots[k].as_ref().and_then(|n| n.next)),
            None => ListIndex(None),
        }
    }

    /// Kyty: `Prev(Index index)`.
    #[must_use]
    pub fn prev(&self, index: ListIndex) -> ListIndex {
        match index.0 {
            Some(k) => ListIndex(self.slots[k].as_ref().and_then(|n| n.prev)),
            None => ListIndex(None),
        }
    }

    /// Kyty: `operator[](Index index)` (const and non-const) / `At(Index)`.
    #[must_use]
    pub fn at(&self, index: ListIndex) -> &T {
        crate::exit_if!(!self.index_valid(index));
        &self.slots[index.0.unwrap()].as_ref().unwrap().value
    }

    /// Kyty: `operator[](Index index)` (mutable).
    pub fn at_mut(&mut self, index: ListIndex) -> &mut T {
        crate::exit_if!(!self.index_valid(index));
        &mut self.slots[index.0.unwrap()].as_mut().unwrap().value
    }

    /// Kyty: `Find(const T2&, OP&&)` (predicate form).
    pub fn find_by<F: Fn(&T) -> bool>(&self, pred: F) -> ListIndex {
        let mut idx = self.first();
        while self.index_valid(idx) {
            if pred(self.at(idx)) {
                return idx;
            }
            idx = self.next(idx);
        }
        ListIndex(None)
    }

    /// Kyty: `Remove(Index index)`.
    pub fn remove_at(&mut self, index: ListIndex) -> ListIndex {
        let Some(k) = index.0 else {
            return ListIndex(None);
        };

        let (prev, next) = {
            let node = self.slots[k].as_ref().unwrap();
            (node.prev, node.next)
        };

        match prev {
            Some(p) => self.slots[p].as_mut().unwrap().next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.slots[n].as_mut().unwrap().prev = prev,
            None => self.tail = prev,
        }

        self.slots[k] = None;
        self.free.push(k);
        self.size -= 1;

        ListIndex(next)
    }

    /// Iterate over `&T` in list order (a safe convenience not present
    /// verbatim in Kyty, built atop `First`/`Next`).
    pub fn iter(&self) -> ListIter<'_, T> {
        ListIter { list: self, cur: self.first() }
    }
}

impl<T: PartialEq> List<T> {
    /// Kyty: `Find(const T& value)`.
    #[must_use]
    pub fn find(&self, value: &T) -> ListIndex {
        self.find_by(|v| v == value)
    }

    /// Kyty: `Contains(const T& value)`.
    #[must_use]
    pub fn contains(&self, value: &T) -> bool {
        self.index_valid(self.find(value))
    }

    /// Kyty: `Remove(const T& value)`.
    pub fn remove(&mut self, value: &T) -> ListIndex {
        let idx = self.find(value);
        self.remove_at(idx)
    }
}

impl<T: Clone> Clone for List<T> {
    /// Kyty: `List(const List<T>& list)` / `operator=(const List<T>&)`.
    fn clone(&self) -> Self {
        let mut out = List::new();
        for v in self.iter() {
            out.add(v.clone());
        }
        out
    }
}

impl<T: fmt::Debug> fmt::Debug for List<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Iterator returned by [`List::iter`].
pub struct ListIter<'a, T> {
    list: &'a List<T>,
    cur: ListIndex,
}

impl<'a, T> Iterator for ListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if !self.list.index_valid(self.cur) {
            return None;
        }
        let v = self.list.at(self.cur);
        self.cur = self.list.next(self.cur);
        Some(v)
    }
}

/// Faithful port of `Kyty::Core::ListSet<T>`: a [`List`] whose `Add` is
/// idempotent (a no-op returning the existing index when the value is
/// already present).
#[derive(Default)]
pub struct ListSet<T>(List<T>);

impl<T> ListSet<T> {
    pub fn new() -> Self {
        ListSet(List::new())
    }

    #[must_use]
    pub fn size(&self) -> u32 {
        self.0.size()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T: PartialEq> ListSet<T> {
    /// Kyty: `ListSet<T>::Add` override — inserts only if not already
    /// present, returning the (new or existing) index either way.
    pub fn add_unique(&mut self, v: T) -> ListIndex {
        let idx = self.0.find(&v);
        if self.0.index_valid(idx) {
            idx
        } else {
            self.0.add(v)
        }
    }
}

impl<T> std::ops::Deref for ListSet<T> {
    type Target = List<T>;
    fn deref(&self) -> &List<T> {
        &self.0
    }
}

impl<T> std::ops::DerefMut for ListSet<T> {
    fn deref_mut(&mut self) -> &mut List<T> {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_iterate_preserves_order() {
        let mut list: List<i32> = List::new();
        list.add(1);
        list.add(2);
        list.add(3);
        assert_eq!(list.size(), 3);
        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn first_last_next_prev_walk_the_list() {
        let mut list: List<i32> = List::new();
        list.add(10);
        list.add(20);
        list.add(30);

        let first = list.first();
        assert!(list.index_valid(first));
        assert_eq!(*list.at(first), 10);

        let last = list.last();
        assert_eq!(*list.at(last), 30);

        let mid = list.next(first);
        assert_eq!(*list.at(mid), 20);
        assert_eq!(*list.at(list.prev(last)), 20);

        let past_last = list.next(last);
        assert!(!list.index_valid(past_last));

        let before_first = list.prev(first);
        assert!(!list.index_valid(before_first));
    }

    #[test]
    fn find_and_contains() {
        let mut list: List<&str> = List::new();
        list.add("a");
        list.add("b");
        list.add("c");

        assert!(list.contains(&"b"));
        assert!(!list.contains(&"z"));

        let idx = list.find(&"b");
        assert!(list.index_valid(idx));
        assert_eq!(*list.at(idx), "b");
    }

    #[test]
    fn remove_by_index_relinks_neighbors_and_returns_next() {
        let mut list: List<i32> = List::new();
        list.add(1);
        let idx2 = list.add(2);
        list.add(3);

        let next = list.remove_at(idx2);
        assert_eq!(list.size(), 2);
        assert_eq!(*list.at(next), 3);

        let collected: Vec<i32> = list.iter().copied().collect();
        assert_eq!(collected, vec![1, 3]);
    }

    #[test]
    fn remove_head_and_tail() {
        let mut list: List<i32> = List::new();
        let head_idx = list.add(1);
        list.add(2);
        let tail_idx = list.add(3);

        let next_after_head = list.remove_at(head_idx);
        assert_eq!(*list.at(next_after_head), 2);
        assert_eq!(*list.at(list.first()), 2);

        let next_after_tail = list.remove_at(tail_idx);
        assert!(!list.index_valid(next_after_tail));
        assert_eq!(*list.at(list.last()), 2);
    }

    #[test]
    fn remove_by_value() {
        let mut list: List<i32> = List::new();
        list.add(5);
        list.add(6);
        list.add(7);

        list.remove(&6);
        assert_eq!(list.size(), 2);
        assert!(!list.contains(&6));
    }

    #[test]
    fn remove_last_element_empties_list() {
        let mut list: List<i32> = List::new();
        list.add(42);
        list.remove(&42);
        assert_eq!(list.size(), 0);
        assert!(!list.index_valid(list.first()));
        assert!(!list.index_valid(list.last()));
    }

    #[test]
    fn clear_resets_list() {
        let mut list: List<i32> = List::new();
        list.add(1);
        list.add(2);
        list.clear();
        assert_eq!(list.size(), 0);
        assert!(!list.index_valid(list.first()));
    }

    #[test]
    fn slot_reuse_after_removal() {
        let mut list: List<i32> = List::new();
        let a = list.add(1);
        list.remove_at(a);
        let b = list.add(2);
        // Slot key is reused, but the old index for 'a' should no longer
        // resolve to a valid node in the new list ordering.
        assert_eq!(list.size(), 1);
        assert_eq!(*list.at(b), 2);
    }

    #[test]
    fn clone_produces_independent_copy_with_same_order() {
        let mut list: List<i32> = List::new();
        list.add(1);
        list.add(2);
        let mut cloned = list.clone();
        cloned.add(3);

        assert_eq!(list.size(), 2);
        assert_eq!(cloned.size(), 3);
        assert_eq!(list.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(cloned.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn list_set_add_unique_deduplicates() {
        let mut set: ListSet<i32> = ListSet::new();
        let i1 = set.add_unique(1);
        let i2 = set.add_unique(2);
        let i1_again = set.add_unique(1);

        assert_eq!(set.size(), 2);
        assert_eq!(i1, i1_again);
        assert_ne!(i1, i2);

        let collected: Vec<i32> = set.iter().copied().collect();
        assert_eq!(collected, vec![1, 2]);
    }
}
