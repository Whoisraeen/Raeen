//! # Raeen Loader
//!
//! Binary loader for PS5 executables and game packages.
//!
//! Supports three file formats:
//! - **ELF** — Standard ELF64 binaries (homebrew, decrypted executables)
//! - **SELF** — Sony's Signed ELF format (encrypted wrapper around ELF)
//! - **PKG** — Sony's package container format (game distribution archives)
//!
//! The loader parses these formats, extracts program segments, and loads
//! them into the emulated PS5 virtual address space.

pub mod elf;
pub mod pkg;
pub mod self_format;

use raeen_core::types::VAddr;

/// Result of loading a PS5 binary.
#[derive(Debug)]
pub struct LoadedBinary {
    /// The entry point virtual address.
    pub entry_point: VAddr,
    /// Loaded program segments.
    pub segments: Vec<LoadedSegment>,
    /// Dynamic libraries required by this binary.
    pub needed_libraries: Vec<String>,
    /// Module name (from the binary metadata).
    pub module_name: String,
    /// Whether this is a dynamically-linked executable.
    pub is_dynamic: bool,
}

/// A loaded program segment in the emulated memory space.
#[derive(Debug)]
pub struct LoadedSegment {
    /// Virtual address where this segment is loaded.
    pub vaddr: VAddr,
    /// Size of the segment in memory.
    pub mem_size: u64,
    /// Size of the segment in the file.
    pub file_size: u64,
    /// Segment data (copied from the binary).
    pub data: Vec<u8>,
    /// Whether this segment is readable.
    pub readable: bool,
    /// Whether this segment is writable.
    pub writable: bool,
    /// Whether this segment is executable.
    pub executable: bool,
}
