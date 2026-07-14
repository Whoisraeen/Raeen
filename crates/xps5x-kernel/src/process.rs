//! Executable image loading — bridges the binary loader and memory manager.
//!
//! The [`xps5x_loader`] crate parses an ELF/SELF into a
//! [`LoadedBinary`](xps5x_loader::LoadedBinary) (entry point + a list of
//! program segments). This module maps those segments into the emulated
//! virtual address space via the [`VirtualMemoryManager`], registers the
//! result as a kernel module, and returns a [`ProcessImage`] describing the
//! loaded executable. This is the path the emulator takes to boot a
//! homebrew ELF (milestone M1).

use crate::OrbisKernel;
use tracing::{debug, info};
use xps5x_core::error::{KernelError, XPS5XError};
use xps5x_core::types::{MemoryProtection, ModuleInfo, VAddr};
use xps5x_loader::{LoadedBinary, LoadedSegment, self_format};

/// A loaded executable image in the emulated address space.
#[derive(Debug, Clone)]
pub struct ProcessImage {
    /// Kernel module ID assigned to this image.
    pub module_id: u32,
    /// Entry point virtual address.
    pub entry_point: VAddr,
    /// Lowest virtual address across all loaded segments.
    pub base_address: VAddr,
    /// Span from the lowest to the highest segment end, in bytes.
    pub image_size: u64,
    /// Number of loadable segments mapped.
    pub segment_count: usize,
    /// Whether the image is dynamically linked.
    pub is_dynamic: bool,
}

/// Translate a segment's read/write/execute flags into [`MemoryProtection`].
fn segment_protection(segment: &LoadedSegment) -> MemoryProtection {
    let mut prot = MemoryProtection::NONE;
    if segment.readable {
        prot |= MemoryProtection::READ;
    }
    if segment.writable {
        prot |= MemoryProtection::WRITE;
    }
    if segment.executable {
        prot |= MemoryProtection::EXEC;
    }
    prot
}

impl OrbisKernel {
    /// Map an already-parsed binary into the emulated address space.
    ///
    /// Each loadable segment is copied into memory at its virtual address
    /// with the appropriate protection. The image is registered as a
    /// kernel module and a [`ProcessImage`] is returned.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] if a segment fails to map, or
    /// [`KernelError::MmapFailed`] if the binary has no loadable segments.
    pub fn load_executable(&self, binary: &LoadedBinary) -> Result<ProcessImage, KernelError> {
        info!(
            "Loading executable '{}': entry={:#x}, {} segment(s), dynamic={}",
            binary.module_name,
            binary.entry_point,
            binary.segments.len(),
            binary.is_dynamic
        );

        if binary.segments.is_empty() {
            return Err(KernelError::MmapFailed {
                address: binary.entry_point,
                size: 0,
            });
        }

        let mut lowest = u64::MAX;
        let mut highest = 0u64;

        for segment in &binary.segments {
            let prot = segment_protection(segment);
            self.memory
                .load_segment(segment.vaddr, &segment.data, segment.mem_size, prot)?;

            lowest = lowest.min(segment.vaddr);
            highest = highest.max(segment.vaddr + segment.mem_size);

            debug!(
                "  mapped segment {:#x}..{:#x} ({:?})",
                segment.vaddr,
                segment.vaddr + segment.mem_size,
                prot
            );
        }

        let image_size = highest - lowest;

        let module_id = self.register_module(ModuleInfo {
            id: 0, // Assigned by register_module.
            name: binary.module_name.clone(),
            base_address: lowest,
            size: image_size,
            entry_point: Some(binary.entry_point),
            initialized: false,
        });

        info!(
            "Executable loaded: module_id={}, base={:#x}, size={:#x}, entry={:#x}",
            module_id, lowest, image_size, binary.entry_point
        );

        Ok(ProcessImage {
            module_id,
            entry_point: binary.entry_point,
            base_address: lowest,
            image_size,
            segment_count: binary.segments.len(),
            is_dynamic: binary.is_dynamic,
        })
    }

    /// Parse raw bytes (ELF or decrypted SELF) and map the result.
    ///
    /// Convenience wrapper that runs the loader's format auto-detection and
    /// then [`OrbisKernel::load_executable`].
    ///
    /// # Errors
    ///
    /// Returns [`XPS5XError::Loader`] if parsing fails or
    /// [`XPS5XError::Kernel`] if mapping fails.
    pub fn load_executable_from_bytes(&self, data: &[u8]) -> Result<ProcessImage, XPS5XError> {
        let binary = self_format::load_binary(data)?;
        Ok(self.load_executable(&binary)?)
    }
}
