//! # Raeen Core
//!
//! Core engine module for the Raeen PlayStation 5 emulator.
//! Provides configuration management, logging infrastructure,
//! error types, and shared constants used across all crates.

pub mod blockers;
pub mod config;
pub mod diagnostics;
pub mod error;
pub mod frame_path;
pub mod host_sleep;
pub mod logging;
pub mod subsystems;
pub mod trophies;
pub mod types;

/// Raeen version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The PS5's page size (16 KiB, matching AMD Zen 2 large page support).
pub const PS5_PAGE_SIZE: usize = 0x4000; // 16 KiB

/// PS5 main memory size (16 GB GDDR6).
pub const PS5_MAIN_MEMORY_SIZE: u64 = 16 * 1024 * 1024 * 1024;

/// PS5 CPU core count.
pub const PS5_CPU_CORES: usize = 8;

/// PS5 GPU compute unit count.
pub const PS5_GPU_COMPUTE_UNITS: u32 = 36;

/// PS5 GPU clock speed in MHz.
pub const PS5_GPU_CLOCK_MHZ: u32 = 2233;
