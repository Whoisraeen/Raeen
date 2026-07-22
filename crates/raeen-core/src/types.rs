//! Shared types used across Raeen crates.
//!
//! Defines common type aliases, address types, and PS5-specific
//! data structures shared between the kernel, GPU, and loader.

use bitflags::bitflags;

/// A PS5 virtual address (48-bit, stored as u64).
pub type VAddr = u64;

/// A PS5 physical address.
pub type PAddr = u64;

/// A PS5 GPU virtual address.
pub type GpuAddr = u64;

/// Size type for memory regions.
pub type MemSize = u64;

/// PS5 process ID.
pub type Pid = u32;

/// PS5 thread ID.
pub type Tid = u64;

/// PS5 file descriptor.
pub type Fd = i32;

/// A handle to a kernel object (event flag, semaphore, etc.).
pub type KernelHandle = i32;

bitflags! {
    /// Memory protection flags (matching POSIX mmap).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MemoryProtection: u32 {
        /// No access.
        const NONE    = 0;
        /// Read access.
        const READ    = 1 << 0;
        /// Write access.
        const WRITE   = 1 << 1;
        /// Execute access.
        const EXEC    = 1 << 2;
        /// GPU read access (PS5 specific).
        const GPU_READ  = 1 << 4;
        /// GPU write access (PS5 specific).
        const GPU_WRITE = 1 << 5;
    }
}

bitflags! {
    /// Memory mapping flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MapFlags: u32 {
        /// Mapping is shared between processes.
        const SHARED    = 1 << 0;
        /// Mapping is private (copy-on-write).
        const PRIVATE   = 1 << 1;
        /// Map at a fixed address.
        const FIXED     = 1 << 4;
        /// Anonymous mapping (not backed by a file).
        const ANONYMOUS = 1 << 5;
    }
}

/// PS5 memory types — the PS5 uses a unified memory model with
/// two access modes for CPU ↔ GPU coherency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ps5MemoryType {
    /// GARLIC memory — GPU-cached, CPU write-combined.
    /// Used for GPU resources (textures, buffers, render targets).
    Garlic,
    /// ONION memory — CPU-cached, GPU uncached.
    /// Used for CPU-side data (game state, CPU buffers).
    Onion,
    /// ONION+ memory — CPU-cached, GPU coherent.
    /// Used for shared data that both CPU and GPU read.
    OnionPlus,
}

/// Represents a mapped memory region in the emulated PS5 address space.
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Virtual address in PS5 address space.
    pub vaddr: VAddr,
    /// Size of the mapping in bytes.
    pub size: MemSize,
    /// Memory protection flags.
    pub protection: MemoryProtection,
    /// Memory type.
    pub memory_type: Ps5MemoryType,
    /// Optional name/label for debugging.
    pub name: Option<String>,
}

/// PS5 module information (loaded .sprx or .elf).
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    /// Module ID.
    pub id: u32,
    /// Module name (e.g., "libkernel", "libSceGnmDriver").
    pub name: String,
    /// Base virtual address where the module is loaded.
    pub base_address: VAddr,
    /// Total size of all loaded segments.
    pub size: MemSize,
    /// Entry point address (for executables).
    pub entry_point: Option<VAddr>,
    /// Whether this module has been initialized.
    pub initialized: bool,
}
