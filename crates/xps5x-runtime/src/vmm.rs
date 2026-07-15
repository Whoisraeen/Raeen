//! The guest virtual-address map: which ranges are free, reserved, or mapped.
//!
//! # Why this exists
//!
//! [`crate::arena::GuestArena`] hands out addresses from monotonic bumps
//! (`heap_bump`, `mmap_bump`, `reserve_bump`) with no free path. Measured
//! consequence on retail titles: Until Dawn opens with a single 512 GiB
//! `sceKernelReserveVirtualRange`, which permanently consumed the shared
//! `reserve_bump`; every later allocation then failed, all the way down to
//! 64 KiB (`sceKernelAllocateDirectMemory: arena mmap failed`), and the title
//! died in its own crash reporter. Dragon Ball fails identically. A bump cannot
//! recover from that, because nothing ever returns.
//!
//! This module is the replacement: one interval map covering the whole guest
//! address space, where *every* address belongs to exactly one
//! [`Vma`] and freeing coalesces back into large blocks. It is a port of
//! shadPS4's `MemoryManager::vma_map` design (`reference/shadps4`,
//! GPL-2.0-or-later, © shadPS4 Emulator Project — see `THIRD_PARTY_NOTICES.md`),
//! which boots commercial titles with this structure. Notably shadPS4 keeps
//! reservations in the *same* map as ordinary allocations: the defect was never
//! that they shared an address space, it was the bump and the missing free path.
//!
//! Deliberately **not** taken from Kyty, whose `PhysicalMemory::Alloc` is the
//! same monotonic-bump design that is failing here.
//!
//! # Invariant
//!
//! The map always tiles `[min, max)` exactly: no gaps, no overlaps. A gap would
//! be an address that [`VmaMap::find`] cannot answer for, and every caller
//! relies on `find` being total over the range.

use std::collections::BTreeMap;

/// What a range of guest address space is being used for.
///
/// Mirrors shadPS4's `VMAType` (`core/memory.h:94`). The discriminants are our
/// own — nothing serializes them — but the set is kept faithful so the port
/// stays readable against the reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmaType {
    /// Nothing owns this range; [`VmaMap::search_free`] may hand it out.
    Free,
    /// Guest reserved the addresses via `sceKernelReserveVirtualRange` but no
    /// memory backs them. Carries no host mapping: a touch is answered by
    /// demand-commit, and a later `MAP_FIXED` may take the range over.
    Reserved,
    /// Backed by direct ("physical") memory at [`Vma::phys_base`].
    Direct,
    /// Backed by flexible memory.
    Flexible,
    /// Pooled memory (`sceKernelMapNamedSystemFlexibleMemory` family).
    Pooled,
    /// Reserved for the pool but not yet committed.
    PoolReserved,
    /// A guest thread stack.
    Stack,
    /// Loaded module image.
    Code,
    /// The emulator's own mapping, not the guest's to unmap.
    System,
}

impl VmaType {
    /// Whether this range is available to hand out.
    pub fn is_free(self) -> bool {
        matches!(self, VmaType::Free)
    }

    /// Whether the range has host memory behind it. `Reserved` deliberately
    /// does not: shadPS4 skips the host mapping for reservations
    /// (`core/memory.cpp:665`), which is what makes a 512 GiB reservation cost
    /// address space rather than 512 GiB of RAM.
    pub fn is_host_backed(self) -> bool {
        !matches!(
            self,
            VmaType::Free | VmaType::Reserved | VmaType::PoolReserved
        )
    }
}

/// PS5 memory protection bits, as the guest passes them.
///
/// Mirrors shadPS4's `MemoryProt` (`core/memory.h:33`). GPU bits are carried
/// through rather than interpreted — the CPU mapping ignores them, but a title
/// reads back what it set.
pub mod prot {
    pub const NO_ACCESS: u32 = 0;
    pub const CPU_READ: u32 = 1;
    pub const CPU_WRITE: u32 = 2;
    pub const CPU_READ_WRITE: u32 = 3;
    pub const CPU_EXEC: u32 = 4;
    pub const GPU_READ: u32 = 16;
    pub const GPU_WRITE: u32 = 32;
    pub const GPU_READ_WRITE: u32 = 48;
}

/// `mmap`-style flags the guest passes to the map calls.
///
/// Mirrors shadPS4's `MemoryMapFlags` (`core/memory.h:45`). Only the two that
/// change placement semantics are load-bearing today.
pub mod map_flags {
    pub const NO_FLAGS: u32 = 0;
    pub const SHARED: u32 = 1;
    pub const PRIVATE: u32 = 2;
    /// The guest's address is binding, not a hint. Measured: Minecraft passes an
    /// address to `sceKernelMapDirectMemory` and then writes to that literal
    /// address without reading the out-param back, so honoring it is mandatory.
    pub const FIXED: u32 = 0x10;
    /// With [`FIXED`], fail instead of clobbering whatever already occupies the
    /// range.
    pub const NO_OVERWRITE: u32 = 0x80;
    pub const VOID: u32 = 0x100;
    pub const STACK: u32 = 0x400;
    pub const ANON: u32 = 0x1000;
    pub const SYSTEM: u32 = 0x2000;
}

/// One contiguous run of guest address space with uniform state.
#[derive(Debug, Clone)]
pub struct Vma {
    pub base: u64,
    pub size: u64,
    pub kind: VmaType,
    pub prot: u32,
    /// Offset into direct memory backing this range, when [`VmaType::Direct`].
    /// Splitting a `Direct` range advances this by the split offset so both
    /// halves keep pointing at their own bytes.
    pub phys_base: Option<u64>,
    /// The `name` a title passes to the Named* map calls; purely diagnostic, but
    /// titles do read it back.
    pub name: String,
    /// Set for ranges that must never be coalesced with a neighbour even when
    /// otherwise identical (the guest asked for them separately and may unmap
    /// them separately).
    pub disallow_merge: bool,
}

impl Vma {
    fn free(base: u64, size: u64) -> Self {
        Self {
            base,
            size,
            kind: VmaType::Free,
            prot: prot::NO_ACCESS,
            phys_base: None,
            name: String::new(),
            disallow_merge: false,
        }
    }

    /// Exclusive end of the range.
    pub fn end(&self) -> u64 {
        self.base + self.size
    }

    /// Whether `[addr, addr + size)` lies wholly inside this range.
    pub fn contains(&self, addr: u64, size: u64) -> bool {
        addr >= self.base && addr.saturating_add(size) <= self.end()
    }

    /// Whether this range and `[addr, addr + size)` share any address.
    pub fn overlaps(&self, addr: u64, size: u64) -> bool {
        addr < self.end() && addr.saturating_add(size) > self.base
    }

    /// Whether two adjacent ranges describe the same thing and may fuse.
    /// `phys_base` must be contiguous too, or fusing would claim bytes the
    /// second half never had.
    fn mergeable_with(&self, next: &Vma) -> bool {
        if self.disallow_merge || next.disallow_merge {
            return false;
        }
        if self.kind != next.kind || self.prot != next.prot || self.name != next.name {
            return false;
        }
        match (self.phys_base, next.phys_base) {
            (None, None) => true,
            (Some(a), Some(b)) => a + self.size == b,
            _ => false,
        }
    }
}

/// The interval map. Tiles `[min, max)` with no gaps or overlaps.
#[derive(Debug)]
pub struct VmaMap {
    map: BTreeMap<u64, Vma>,
    min: u64,
    max: u64,
}

impl VmaMap {
    /// A map whose whole range is free.
    ///
    /// # Panics
    ///
    /// If `max <= min` — an empty address space is always a construction bug,
    /// and every later operation would have to answer for a range that cannot
    /// contain anything.
    pub fn new(min: u64, max: u64) -> Self {
        assert!(max > min, "empty address space [{min:#x}, {max:#x})");
        let mut map = BTreeMap::new();
        map.insert(min, Vma::free(min, max - min));
        Self { map, min, max }
    }

    pub fn min(&self) -> u64 {
        self.min
    }

    pub fn max(&self) -> u64 {
        self.max
    }

    /// The VMA containing `addr`, or `None` if `addr` is outside `[min, max)`.
    /// Total over the range, by the tiling invariant.
    pub fn find(&self, addr: u64) -> Option<&Vma> {
        if addr < self.min || addr >= self.max {
            return None;
        }
        self.map.range(..=addr).next_back().map(|(_, vma)| vma)
    }

    /// Every VMA, in address order.
    pub fn iter(&self) -> impl Iterator<Item = &Vma> {
        self.map.values()
    }

    /// Split the VMA containing `at` so that a VMA begins exactly at `at`.
    /// No-op when one already does, or when `at` is outside the map.
    fn split_at(&mut self, at: u64) {
        if at <= self.min || at >= self.max {
            return;
        }
        let Some((&base, vma)) = self.map.range(..=at).next_back() else {
            return;
        };
        if base == at {
            return;
        }
        let mut left = vma.clone();
        let mut right = vma.clone();
        let left_size = at - base;
        left.size = left_size;
        right.base = at;
        right.size = vma.size - left_size;
        // A split Direct range must keep both halves pointing at their own
        // physical bytes, or the tail would silently alias the head.
        right.phys_base = vma.phys_base.map(|p| p + left_size);
        self.map.insert(base, left);
        self.map.insert(at, right);
    }

    /// Make `[addr, addr + size)` line up on VMA boundaries, so it is covered by
    /// whole entries and can be replaced wholesale.
    pub fn carve(&mut self, addr: u64, size: u64) {
        self.split_at(addr);
        self.split_at(addr + size);
    }

    /// Fuse the VMA at `addr` with its neighbours where they describe the same
    /// thing. Without this, a reserve/release cycle would shred the map into
    /// unusable slivers and large requests would start failing even with the
    /// space free — the bump's failure mode, reintroduced.
    pub fn merge_adjacent(&mut self, addr: u64) {
        // Merge forward first: the backward merge may delete this entry, and its
        // key would then be gone.
        let next_base = self
            .map
            .range(addr + 1..)
            .next()
            .map(|(&base, _)| base)
            .filter(|&next_base| {
                let (Some(cur), Some(next)) = (self.map.get(&addr), self.map.get(&next_base))
                else {
                    return false;
                };
                cur.end() == next.base && cur.mergeable_with(next)
            });
        if let Some(next_base) = next_base {
            let next_size = self.map.remove(&next_base).expect("just located").size;
            if let Some(cur) = self.map.get_mut(&addr) {
                cur.size += next_size;
            }
        }

        let prev_base = self
            .map
            .range(..addr)
            .next_back()
            .map(|(&base, _)| base)
            .filter(|&prev_base| {
                let (Some(prev), Some(cur)) = (self.map.get(&prev_base), self.map.get(&addr))
                else {
                    return false;
                };
                prev.end() == cur.base && prev.mergeable_with(cur)
            });
        if let Some(prev_base) = prev_base {
            let cur_size = self.map.remove(&addr).expect("current entry").size;
            if let Some(prev) = self.map.get_mut(&prev_base) {
                prev.size += cur_size;
            }
        }
    }

    /// Claim `[addr, addr + size)` as `kind`, replacing whatever is there.
    /// The caller is responsible for having checked overwrite policy.
    #[allow(clippy::too_many_arguments)]
    pub fn map_range(
        &mut self,
        addr: u64,
        size: u64,
        kind: VmaType,
        prot: u32,
        phys_base: Option<u64>,
        name: &str,
        disallow_merge: bool,
    ) {
        self.carve(addr, size);
        // Drop every entry the range now covers exactly.
        let covered: Vec<u64> = self
            .map
            .range(addr..addr + size)
            .map(|(&base, _)| base)
            .collect();
        for base in covered {
            self.map.remove(&base);
        }
        self.map.insert(
            addr,
            Vma {
                base: addr,
                size,
                kind,
                prot,
                phys_base,
                name: name.to_owned(),
                disallow_merge,
            },
        );
        self.merge_adjacent(addr);
    }

    /// Return `[addr, addr + size)` to [`VmaType::Free`] and coalesce. This is
    /// the path a bump allocator never had.
    pub fn unmap_range(&mut self, addr: u64, size: u64) {
        self.map_range(addr, size, VmaType::Free, prot::NO_ACCESS, None, "", false);
    }

    /// Whether every address in `[addr, addr + size)` is free.
    pub fn range_is_free(&self, addr: u64, size: u64) -> bool {
        if size == 0 || addr < self.min || addr.saturating_add(size) > self.max {
            return false;
        }
        let mut cursor = addr;
        while cursor < addr + size {
            let Some(vma) = self.find(cursor) else {
                return false;
            };
            if !vma.kind.is_free() {
                return false;
            }
            cursor = vma.end();
        }
        true
    }

    /// First address at or after `hint` where `size` bytes are free at `align`.
    ///
    /// Mirrors shadPS4's `MemoryManager::SearchFree` (`core/memory.cpp:1383`):
    /// start at the hint, then walk free VMAs in address order. Returns `None`
    /// when nothing fits — which, unlike a bump running dry, now means the
    /// address space is genuinely full rather than merely spent.
    pub fn search_free(&self, hint: u64, size: u64, align: u64) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let align = align.max(1);
        let start = hint.max(self.min);
        if start >= self.max {
            return None;
        }

        // The hint itself, if it happens to land in a free run that fits.
        let aligned = align_up(start, align)?;
        if let Some(vma) = self.find(aligned)
            && vma.kind.is_free()
            && vma.contains(aligned, size)
        {
            return Some(aligned);
        }

        // Otherwise the first free VMA at or after the hint that fits once
        // aligned. Alignment can consume more than the VMA holds, so the fit is
        // rechecked after aligning rather than assumed from `vma.size`.
        for vma in self.map.range(aligned..).map(|(_, vma)| vma) {
            if !vma.kind.is_free() {
                continue;
            }
            let Some(candidate) = align_up(vma.base.max(aligned), align) else {
                continue;
            };
            if vma.contains(candidate, size) {
                return Some(candidate);
            }
        }
        None
    }
}

/// Round `value` up to a multiple of `align`, or `None` on overflow.
fn align_up(value: u64, align: u64) -> Option<u64> {
    if align <= 1 {
        return Some(value);
    }
    let rem = value % align;
    if rem == 0 {
        return Some(value);
    }
    value.checked_add(align - rem)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: u64 = 0x1000;
    const MAX: u64 = 0x1_0000_0000;

    fn map() -> VmaMap {
        VmaMap::new(MIN, MAX)
    }

    /// The invariant every other operation depends on: the map tiles its range
    /// with no gaps or overlaps, so `find` is total.
    fn assert_tiles(m: &VmaMap) {
        let mut expected = m.min();
        for vma in m.iter() {
            assert_eq!(vma.base, expected, "gap or overlap at {:#x}", vma.base);
            assert!(vma.size > 0, "zero-sized vma at {:#x}", vma.base);
            expected = vma.end();
        }
        assert_eq!(expected, m.max(), "map does not reach max");
    }

    #[test]
    fn a_fresh_map_is_one_free_range_covering_everything() {
        let m = map();
        assert_tiles(&m);
        assert!(m.range_is_free(MIN, MAX - MIN));
        assert_eq!(m.iter().count(), 1);
    }

    #[test]
    fn find_is_total_inside_the_range_and_refuses_outside() {
        let m = map();
        assert!(m.find(MIN).is_some());
        assert!(m.find(MAX - 1).is_some());
        assert!(m.find(MIN - 1).is_none());
        assert!(m.find(MAX).is_none());
    }

    #[test]
    fn mapping_a_range_splits_the_free_space_around_it() {
        let mut m = map();
        m.map_range(
            0x10000,
            0x1000,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0),
            "d",
            false,
        );
        assert_tiles(&m);
        let vma = m.find(0x10000).expect("mapped");
        assert_eq!(vma.kind, VmaType::Direct);
        assert_eq!(vma.size, 0x1000);
        assert!(!m.range_is_free(0x10000, 0x1000));
        assert!(m.range_is_free(0x11000, 0x1000));
    }

    /// The whole point of the module. A bump could not do this: after the
    /// reservation is released the space must be handed out again.
    #[test]
    fn released_space_is_reusable_which_a_bump_allocator_could_never_do() {
        let mut m = map();
        let big = 0x8000_0000u64; // 2 GiB, then released, twice over
        for _ in 0..4 {
            let addr = m
                .search_free(MIN, big, 0x1000)
                .expect("a released range must be handed out again");
            m.map_range(
                addr,
                big,
                VmaType::Reserved,
                prot::NO_ACCESS,
                None,
                "",
                false,
            );
            m.unmap_range(addr, big);
            assert_tiles(&m);
        }
        // Fully coalesced back to a single free range.
        assert_eq!(m.iter().count(), 1);
        assert!(m.range_is_free(MIN, MAX - MIN));
    }

    /// Until Dawn's measured failure, reduced: reserve far more than any single
    /// later allocation needs, release it, and keep allocating. The old
    /// `reserve_bump` consumed the span permanently and then failed a 64 KiB
    /// request; this must not.
    #[test]
    fn a_huge_reservation_then_released_does_not_starve_later_allocations() {
        let mut m = map();
        let huge = (MAX - MIN) / 2;
        let addr = m.search_free(MIN, huge, 0x1000).expect("huge reservation");
        m.map_range(
            addr,
            huge,
            VmaType::Reserved,
            prot::NO_ACCESS,
            None,
            "",
            false,
        );
        m.unmap_range(addr, huge);

        // The exact size that failed on the real title.
        let small = m
            .search_free(MIN, 0x10000, 0x1000)
            .expect("64 KiB must still be available after a released reservation");
        m.map_range(
            small,
            0x10000,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0),
            "",
            false,
        );
        assert_tiles(&m);
    }

    #[test]
    fn adjacent_identical_ranges_coalesce_but_differing_ones_do_not() {
        let mut m = map();
        m.map_range(
            0x10000,
            0x1000,
            VmaType::Reserved,
            prot::NO_ACCESS,
            None,
            "",
            false,
        );
        m.map_range(
            0x11000,
            0x1000,
            VmaType::Reserved,
            prot::NO_ACCESS,
            None,
            "",
            false,
        );
        assert_eq!(m.find(0x10000).expect("merged").size, 0x2000);

        // A different protection must stay its own range.
        m.map_range(
            0x12000,
            0x1000,
            VmaType::Reserved,
            prot::CPU_READ,
            None,
            "",
            false,
        );
        assert_eq!(m.find(0x10000).expect("still merged").size, 0x2000);
        assert_eq!(m.find(0x12000).expect("distinct").size, 0x1000);
        assert_tiles(&m);
    }

    /// Fusing two Direct ranges whose physical bytes are not contiguous would
    /// silently claim bytes the tail never had.
    #[test]
    fn direct_ranges_only_coalesce_when_their_physical_backing_is_contiguous() {
        let mut m = map();
        m.map_range(
            0x10000,
            0x1000,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0x0),
            "",
            false,
        );
        m.map_range(
            0x11000,
            0x1000,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0x1000),
            "",
            false,
        );
        assert_eq!(
            m.find(0x10000).expect("contiguous phys merges").size,
            0x2000
        );

        let mut m = map();
        m.map_range(
            0x10000,
            0x1000,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0x0),
            "",
            false,
        );
        m.map_range(
            0x11000,
            0x1000,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0x9000),
            "",
            false,
        );
        assert_eq!(
            m.find(0x10000).expect("disjoint phys stays split").size,
            0x1000
        );
        assert_tiles(&m);
    }

    /// Splitting a Direct range must keep each half pointing at its own bytes.
    #[test]
    fn splitting_a_direct_range_advances_the_physical_base_of_the_tail() {
        let mut m = map();
        m.map_range(
            0x10000,
            0x4000,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0x8000),
            "",
            false,
        );
        // Unmap the middle, forcing splits on both sides.
        m.unmap_range(0x11000, 0x1000);
        assert_tiles(&m);
        assert_eq!(m.find(0x10000).expect("head").phys_base, Some(0x8000));
        assert_eq!(m.find(0x12000).expect("tail").phys_base, Some(0xA000));
    }

    #[test]
    fn search_free_honors_alignment() {
        let m = map();
        let addr = m.search_free(MIN, 0x1000, 0x10000).expect("aligned fit");
        assert_eq!(addr % 0x10000, 0);
    }

    #[test]
    fn search_free_skips_occupied_ranges() {
        let mut m = map();
        m.map_range(
            MIN,
            0x10_0000,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0),
            "",
            false,
        );
        let addr = m
            .search_free(MIN, 0x1000, 0x1000)
            .expect("fit after the mapping");
        assert!(addr >= MIN + 0x10_0000);
    }

    #[test]
    fn search_free_returns_none_when_nothing_fits() {
        let mut m = map();
        m.map_range(
            MIN,
            MAX - MIN,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0),
            "",
            false,
        );
        assert!(m.search_free(MIN, 0x1000, 0x1000).is_none());
    }

    /// A mapping laid over several existing ranges must replace all of them and
    /// leave the map tiled — this is what MAP_FIXED does over a reservation.
    #[test]
    fn a_fixed_style_overwrite_replaces_every_range_it_covers() {
        let mut m = map();
        m.map_range(
            0x10000,
            0x1000,
            VmaType::Reserved,
            prot::NO_ACCESS,
            None,
            "a",
            false,
        );
        m.map_range(
            0x11000,
            0x1000,
            VmaType::Direct,
            prot::CPU_READ,
            Some(0),
            "b",
            false,
        );
        m.map_range(
            0x12000,
            0x1000,
            VmaType::Reserved,
            prot::NO_ACCESS,
            None,
            "c",
            false,
        );

        m.map_range(
            0x10000,
            0x3000,
            VmaType::Direct,
            prot::CPU_READ_WRITE,
            Some(0x5000),
            "over",
            false,
        );
        assert_tiles(&m);
        let vma = m.find(0x10000).expect("overwritten");
        assert_eq!(vma.size, 0x3000);
        assert_eq!(vma.name, "over");
        assert_eq!(m.find(0x11000).expect("same range").base, 0x10000);
    }

    #[test]
    fn disallow_merge_keeps_a_range_separately_unmappable() {
        let mut m = map();
        m.map_range(
            0x10000,
            0x1000,
            VmaType::Reserved,
            prot::NO_ACCESS,
            None,
            "",
            true,
        );
        m.map_range(
            0x11000,
            0x1000,
            VmaType::Reserved,
            prot::NO_ACCESS,
            None,
            "",
            true,
        );
        assert_eq!(m.find(0x10000).expect("unmerged").size, 0x1000);
        assert_eq!(m.find(0x11000).expect("unmerged").size, 0x1000);
        assert_tiles(&m);
    }

    #[test]
    fn reservations_carry_no_host_memory_but_mapped_kinds_do() {
        assert!(!VmaType::Reserved.is_host_backed());
        assert!(!VmaType::Free.is_host_backed());
        assert!(!VmaType::PoolReserved.is_host_backed());
        assert!(VmaType::Direct.is_host_backed());
        assert!(VmaType::Flexible.is_host_backed());
        assert!(VmaType::Code.is_host_backed());
    }
}
