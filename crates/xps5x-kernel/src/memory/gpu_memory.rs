//! GPU memory management.
//!
//! Handles the PS5's unified memory model where GPU and CPU share
//! the same 16 GB GDDR6 pool with two access modes:
//!
//! - **GARLIC**: GPU-cached, CPU write-combined. Used for render targets,
//!   textures, vertex buffers — anything the GPU reads frequently.
//! - **ONION**: CPU-cached, GPU uncached. Used for game state, CPU-side
//!   computation, and data the CPU reads back from GPU.
//! - **ONION+**: CPU-cached, GPU coherent. For shared data.

use parking_lot::RwLock;
use std::collections::BTreeMap;
use tracing::{debug, info};
use xps5x_core::types::{GpuAddr, MemSize, VAddr};

/// GPU memory region descriptor.
#[derive(Debug, Clone)]
pub struct GpuMemoryRegion {
    /// GPU virtual address.
    pub gpu_addr: GpuAddr,
    /// CPU virtual address (in the emulated PS5 address space).
    pub cpu_addr: VAddr,
    /// Size of the region.
    pub size: MemSize,
    /// Memory type (affects caching behavior).
    pub memory_type: GpuMemoryType,
    /// Label for debugging.
    pub label: Option<String>,
}

/// GPU memory types mapping to PS5's unified memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryType {
    /// GARLIC — GPU-cached, optimal for GPU reads.
    Garlic,
    /// ONION — CPU-cached, optimal for CPU reads.
    Onion,
    /// ONION+ — Coherent between CPU and GPU.
    OnionPlus,
}

/// Manages GPU memory mappings in the emulated PS5.
pub struct GpuMemoryManager {
    /// GPU address → region mapping.
    regions: RwLock<BTreeMap<GpuAddr, GpuMemoryRegion>>,
    /// Next available GPU virtual address.
    next_gpu_addr: RwLock<GpuAddr>,
    /// Total allocated GPU memory in bytes.
    total_allocated: RwLock<MemSize>,
}

impl Default for GpuMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuMemoryManager {
    /// Create a new GPU memory manager.
    pub fn new() -> Self {
        info!("Initializing GPU memory manager (PS5 unified memory model)");
        Self {
            regions: RwLock::new(BTreeMap::new()),
            next_gpu_addr: RwLock::new(0x0000_0001_0000_0000), // GPU VA space starts here.
            total_allocated: RwLock::new(0),
        }
    }

    /// Allocate a GPU memory region.
    pub fn allocate(
        &self,
        size: MemSize,
        memory_type: GpuMemoryType,
        cpu_addr: VAddr,
        label: Option<String>,
    ) -> GpuAddr {
        let aligned_size = (size + 0xFFFF) & !0xFFFF; // 64 KiB alignment for GPU.

        let mut next = self.next_gpu_addr.write();
        let gpu_addr = *next;
        *next += aligned_size;

        let region = GpuMemoryRegion {
            gpu_addr,
            cpu_addr,
            size: aligned_size,
            memory_type,
            label: label.clone(),
        };

        debug!(
            "GPU alloc: gpu={:#x}, cpu={:#x}, size={:#x}, type={:?}, label={:?}",
            gpu_addr, cpu_addr, aligned_size, memory_type, label
        );

        self.regions.write().insert(gpu_addr, region);
        *self.total_allocated.write() += aligned_size;

        gpu_addr
    }

    /// Free a GPU memory region.
    pub fn free(&self, gpu_addr: GpuAddr) {
        if let Some(region) = self.regions.write().remove(&gpu_addr) {
            *self.total_allocated.write() -= region.size;
            debug!("GPU free: gpu={:#x}, size={:#x}", gpu_addr, region.size);
        }
    }

    /// Look up the CPU address corresponding to a GPU address.
    pub fn gpu_to_cpu(&self, gpu_addr: GpuAddr) -> Option<VAddr> {
        let regions = self.regions.read();
        for (base, region) in regions.iter() {
            if gpu_addr >= *base && gpu_addr < *base + region.size {
                let offset = gpu_addr - base;
                return Some(region.cpu_addr + offset);
            }
        }
        None
    }

    /// Get total allocated GPU memory.
    pub fn total_allocated(&self) -> MemSize {
        *self.total_allocated.read()
    }
}
