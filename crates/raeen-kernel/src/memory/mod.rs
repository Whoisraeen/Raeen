//! Virtual memory manager for the emulated PS5 address space.
//!
//! Manages the PS5's flat 48-bit virtual address space,
//! translating mmap/munmap/mprotect to host memory allocations.
//! Handles PS5-specific memory types (GARLIC/ONION) for CPU↔GPU coherency.

pub mod gpu_memory;
pub mod virtual_memory;

use parking_lot::RwLock;
use raeen_core::error::KernelError;
use raeen_core::types::{MappingKind, MemoryProtection, MemoryRegion, Ps5MemoryType, VAddr};
use std::collections::BTreeMap;
use tracing::{debug, info, warn};

/// Manages the emulated PS5 virtual address space.
pub struct VirtualMemoryManager {
    /// Active memory mappings, keyed by base virtual address.
    regions: RwLock<BTreeMap<VAddr, MemoryRegion>>,
    /// Backing storage for mapped regions.
    /// Maps VAddr → host heap allocation.
    backing: RwLock<BTreeMap<VAddr, Vec<u8>>>,
    /// Next available address for anonymous mappings.
    next_anon_addr: RwLock<VAddr>,
}

impl Default for VirtualMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualMemoryManager {
    /// Create a new virtual memory manager.
    pub fn new() -> Self {
        info!("Initializing virtual memory manager");
        Self {
            regions: RwLock::new(BTreeMap::new()),
            backing: RwLock::new(BTreeMap::new()),
            // Start anonymous mappings at a high address to avoid
            // conflicts with ELF segment addresses.
            next_anon_addr: RwLock::new(0x0000_2000_0000_0000),
        }
    }

    /// Map a region of memory (mmap equivalent).
    pub fn mmap(
        &self,
        addr: u64,
        length: u64,
        prot: u32,
        flags: u32,
        _fd: i32,
        _offset: u64,
    ) -> Result<u64, KernelError> {
        let aligned_length = align_up(length, raeen_core::PS5_PAGE_SIZE as u64);

        // Determine the mapping address.
        let map_addr = if addr != 0 && (flags & 0x10 != 0) {
            // MAP_FIXED — use the requested address.
            addr
        } else if addr != 0 {
            // Hint address provided, try to use it.
            addr
        } else {
            // Anonymous: allocate at the next available address.
            let mut next = self.next_anon_addr.write();
            let result = *next;
            *next += aligned_length;
            result
        };

        let protection = MemoryProtection::from_bits_truncate(prot);

        let region = MemoryRegion {
            vaddr: map_addr,
            size: aligned_length,
            protection,
            memory_type: Ps5MemoryType::Onion, // Default to CPU-cached.
            name: None,
            kind: MappingKind::Anonymous,
            direct_offset: 0,
            direct_memory_type: 0,
        };

        debug!(
            "mmap: mapping {:#x}..{:#x} ({} bytes, prot={:?})",
            map_addr,
            map_addr + aligned_length,
            aligned_length,
            protection
        );

        // Create backing storage.
        let backing_data = vec![0u8; aligned_length as usize];

        self.regions.write().insert(map_addr, region);
        self.backing.write().insert(map_addr, backing_data);

        Ok(map_addr)
    }

    /// Unmap a region of memory (munmap equivalent).
    pub fn munmap(&self, addr: u64, length: u64) -> Result<(), KernelError> {
        let aligned_length = align_up(length, raeen_core::PS5_PAGE_SIZE as u64);

        debug!(
            "munmap: unmapping {:#x}..{:#x}",
            addr,
            addr + aligned_length
        );

        self.regions.write().remove(&addr);
        self.backing.write().remove(&addr);

        Ok(())
    }

    /// Record region metadata for a mapping whose bytes are backed
    /// *externally* — i.e. by the runtime's `GuestArena`, not a `Vec` owned
    /// by this VMM (RT2 Task 5, design doc §5). Inserts a [`MemoryRegion`]
    /// into `regions` only; deliberately does **not** touch `backing`.
    ///
    /// Because there is no `backing` entry, this VMM's own [`Self::read`]/
    /// [`Self::write`] will not find bytes for `addr` — that is intentional:
    /// the real bytes live in the arena, and callers must go through the
    /// same [`raeen_hle::GuestMemory`]-style view that owns them (the
    /// arena itself), not through this VMM. `is_mapped`/`region_containing`
    /// still work, since those only ever consult `regions`.
    pub fn record_mapping(&self, addr: VAddr, size: u64, prot: u32) {
        self.record_mapping_of_kind(addr, size, prot, MappingKind::Anonymous, 0, 0);
    }

    /// [`Self::record_mapping`], but recording what kind of mapping this is and
    /// (for direct memory) the physical offset and allocation type it came
    /// from, so `sceKernelVirtualQuery` can report the region the way the
    /// guest mapped it.
    ///
    /// Titles read their own mappings back. Minecraft's embedded V8 maps direct
    /// memory and immediately queries the range to confirm the kernel agrees;
    /// answering "anonymous, offset 0, type 0" for memory it had just mapped as
    /// direct type 12 violated its invariant and tripped `UNREACHABLE()`.
    pub fn record_mapping_of_kind(
        &self,
        addr: VAddr,
        size: u64,
        prot: u32,
        kind: MappingKind,
        direct_offset: u64,
        direct_memory_type: i32,
    ) {
        let aligned_size = align_up(size, raeen_core::PS5_PAGE_SIZE as u64);
        let protection = MemoryProtection::from_bits_truncate(prot);

        let region = MemoryRegion {
            vaddr: addr,
            size: aligned_size,
            protection,
            memory_type: Ps5MemoryType::Onion,
            name: Some("arena_mmap".to_string()),
            kind,
            direct_offset,
            direct_memory_type,
        };

        debug!(
            "record_mapping: {:#x}..{:#x} ({} bytes, prot={:?}) [externally backed]",
            addr,
            addr + aligned_size,
            aligned_size,
            protection
        );

        self.regions.write().insert(addr, region);
    }

    /// Remove metadata previously recorded by [`Self::record_mapping`].
    /// Like `record_mapping`, this only touches `regions` — there is no
    /// `backing` entry to remove, since this VMM never owned the bytes.
    pub fn remove_mapping(&self, addr: VAddr) {
        debug!("remove_mapping: {:#x}", addr);
        self.regions.write().remove(&addr);
    }

    /// Change memory protection (mprotect equivalent).
    pub fn mprotect(&self, addr: u64, _length: u64, prot: u32) -> Result<(), KernelError> {
        let protection = MemoryProtection::from_bits_truncate(prot);

        if let Some(region) = self.regions.write().get_mut(&addr) {
            debug!(
                "mprotect: {:#x} {:?} -> {:?}",
                addr, region.protection, protection
            );
            region.protection = protection;
            Ok(())
        } else {
            warn!("mprotect: no mapping found at {:#x}", addr);
            // Don't fail — some games mprotect regions we don't track.
            Ok(())
        }
    }

    /// Read bytes from emulated memory.
    ///
    /// The read must fall entirely within a single mapped region; it may
    /// start anywhere inside that region, not only at its base.
    pub fn read(&self, addr: VAddr, size: usize) -> Result<Vec<u8>, KernelError> {
        let backing = self.backing.read();

        // Find the region with the greatest base address <= addr, then
        // verify the whole read fits within it (O(log n) lookup).
        if let Some((base, data)) = backing.range(..=addr).next_back() {
            let end = base + data.len() as u64;
            if addr
                .checked_add(size as u64)
                .is_some_and(|read_end| read_end <= end)
            {
                let offset = (addr - base) as usize;
                return Ok(data[offset..offset + size].to_vec());
            }
        }

        Err(KernelError::InvalidMemoryAccess(addr))
    }

    /// Write bytes to emulated memory.
    ///
    /// The write must fall entirely within a single mapped region.
    pub fn write(&self, addr: VAddr, data: &[u8]) -> Result<(), KernelError> {
        let mut backing = self.backing.write();

        if let Some((base, storage)) = backing.range_mut(..=addr).next_back() {
            let end = *base + storage.len() as u64;
            if addr
                .checked_add(data.len() as u64)
                .is_some_and(|write_end| write_end <= end)
            {
                let offset = (addr - *base) as usize;
                storage[offset..offset + data.len()].copy_from_slice(data);
                return Ok(());
            }
        }

        Err(KernelError::InvalidMemoryAccess(addr))
    }

    /// Find the mapped region containing `addr`, if any.
    ///
    /// Returns a clone of the region metadata (not the backing bytes).
    /// The `type` a direct-memory range was allocated with, looked up by its
    /// physical (direct-memory) address. `None` if nothing is recorded there.
    #[must_use]
    pub fn direct_allocation_type(&self, phys: VAddr) -> Option<i32> {
        self.region_containing(phys)
            .map(|region| region.direct_memory_type)
    }

    pub fn region_containing(&self, addr: VAddr) -> Option<MemoryRegion> {
        self.regions
            .read()
            .range(..=addr)
            .next_back()
            .filter(|(base, region)| addr < **base + region.size)
            .map(|(_, region)| region.clone())
    }

    /// Find the first mapped region whose base is at or after `addr`.
    /// This backs `sceKernelVirtualQuery(..., FIND_NEXT, ...)` without
    /// exposing the VMM's internal map.
    pub fn region_at_or_after(&self, addr: VAddr) -> Option<MemoryRegion> {
        self.regions
            .read()
            .range(addr..)
            .next()
            .map(|(_, region)| region.clone())
    }

    /// Whether `addr` falls within any mapped region.
    pub fn is_mapped(&self, addr: VAddr) -> bool {
        self.region_containing(addr).is_some()
    }

    /// Load a binary segment into emulated memory.
    pub fn load_segment(
        &self,
        vaddr: VAddr,
        data: &[u8],
        mem_size: u64,
        prot: MemoryProtection,
    ) -> Result<(), KernelError> {
        let aligned_size = align_up(mem_size, raeen_core::PS5_PAGE_SIZE as u64);

        let region = MemoryRegion {
            vaddr,
            size: aligned_size,
            protection: prot,
            // Loaded segments currently all use Onion (CPU-cached) memory.
            // TODO: place GPU-facing data segments in Garlic once the loader
            // distinguishes resource segments from code/data.
            memory_type: Ps5MemoryType::Onion,
            name: Some("loaded_segment".to_string()),
            kind: MappingKind::Anonymous,
            direct_offset: 0,
            direct_memory_type: 0,
        };

        let mut backing_data = vec![0u8; aligned_size as usize];
        let copy_len = data.len().min(aligned_size as usize);
        backing_data[..copy_len].copy_from_slice(&data[..copy_len]);

        info!(
            "Loaded segment: {:#x}..{:#x} ({} bytes data, {} bytes total, prot={:?})",
            vaddr,
            vaddr + aligned_size,
            data.len(),
            aligned_size,
            prot
        );

        self.regions.write().insert(vaddr, region);
        self.backing.write().insert(vaddr, backing_data);

        Ok(())
    }

    /// Get information about all mapped regions (for debugging).
    pub fn dump_regions(&self) -> Vec<MemoryRegion> {
        self.regions.read().values().cloned().collect()
    }
}

/// Align `value` up to `alignment`, which **must** be a non-zero power of two.
///
/// The mask form silently produces garbage for a non-power-of-two, and
/// `alignment == 0` underflows `alignment - 1` to `u64::MAX` (wrapping in
/// release, panicking in debug). Every caller today passes the
/// [`raeen_core::PS5_PAGE_SIZE`] constant, so this cannot fire — the assertion
/// exists so the day a *guest-supplied* alignment is threaded through here it
/// fails loudly in tests instead of corrupting a mapping in a retail run.
fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(
        alignment.is_power_of_two(),
        "align_up alignment must be a non-zero power of two, got {alignment}"
    );
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 0x4000), 0);
        assert_eq!(align_up(1, 0x4000), 0x4000);
        assert_eq!(align_up(0x4000, 0x4000), 0x4000);
        assert_eq!(align_up(0x4001, 0x4000), 0x8000);
    }

    #[test]
    fn test_mmap_and_read_write() {
        let vmm = VirtualMemoryManager::new();
        let addr = vmm.mmap(0, 0x10000, 0x7, 0x22, -1, 0).unwrap();
        assert_ne!(addr, 0);

        // Write and read back.
        vmm.write(addr, &[1, 2, 3, 4]).unwrap();
        let data = vmm.read(addr, 4).unwrap();
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_munmap() {
        let vmm = VirtualMemoryManager::new();
        let addr = vmm.mmap(0, 0x4000, 0x7, 0x22, -1, 0).unwrap();
        vmm.munmap(addr, 0x4000).unwrap();

        // Reading from unmapped memory should fail.
        assert!(vmm.read(addr, 4).is_err());
    }

    #[test]
    fn test_read_from_middle_of_region() {
        let vmm = VirtualMemoryManager::new();
        let addr = vmm.mmap(0, 0x10000, 0x7, 0x22, -1, 0).unwrap();

        // Write at an offset inside the region, then read it back.
        vmm.write(addr + 0x100, &[0xAA, 0xBB, 0xCC]).unwrap();
        let data = vmm.read(addr + 0x100, 3).unwrap();
        assert_eq!(data, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_read_spanning_region_end_fails() {
        let vmm = VirtualMemoryManager::new();
        let addr = vmm.mmap(0, 0x4000, 0x7, 0x22, -1, 0).unwrap();

        // A read that starts in-bounds but extends past the region end fails.
        assert!(vmm.read(addr + 0x3FFF, 4).is_err());
    }

    #[test]
    fn test_region_containing_and_is_mapped() {
        let vmm = VirtualMemoryManager::new();
        let addr = vmm.mmap(0, 0x8000, 0x5, 0x22, -1, 0).unwrap();

        assert!(vmm.is_mapped(addr));
        assert!(vmm.is_mapped(addr + 0x7FFF));
        assert!(!vmm.is_mapped(addr + 0x8000)); // One past the end.
        assert!(!vmm.is_mapped(addr - 1));

        let region = vmm.region_containing(addr + 0x10).unwrap();
        assert_eq!(region.vaddr, addr);
        assert_eq!(region.size, 0x8000);
    }

    #[test]
    fn record_mapping_and_remove_mapping_are_metadata_only() {
        let vmm = VirtualMemoryManager::new();
        let addr = 0x0000_1000_A000_0000;

        vmm.record_mapping(addr, 0x4000, 0x3);

        assert!(vmm.is_mapped(addr));
        assert!(vmm.is_mapped(addr + 0x1234)); // Somewhere in the middle.
        assert!(!vmm.is_mapped(addr + 0x4000)); // One past the end.

        let region = vmm.region_containing(addr + 0x1234).unwrap();
        assert_eq!(region.vaddr, addr);
        assert_eq!(region.size, 0x4000);

        // No `backing` entry was created — reads/writes through this VMM
        // must not find bytes for an externally (arena-)backed region.
        assert!(vmm.read(addr, 4).is_err());

        vmm.remove_mapping(addr);
        assert!(!vmm.is_mapped(addr));
    }

    #[test]
    fn region_at_or_after_finds_the_next_mapping() {
        let vmm = VirtualMemoryManager::new();
        vmm.record_mapping(0x2000, 0x1000, 0x3);
        vmm.record_mapping(0x5000, 0x1000, 0x3);
        assert_eq!(vmm.region_at_or_after(0x3000).unwrap().vaddr, 0x5000);
        assert!(vmm.region_at_or_after(0x6000).is_none());
    }
}
