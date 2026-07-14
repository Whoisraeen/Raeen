//! # XPS5X I/O
//!
//! Emulates the PS5's custom I/O complex, which provides:
//! - Hardware-accelerated decompression (Kraken/Oodle)
//! - DMA transfers between SSD, memory, and GPU
//! - Guaranteed I/O bandwidth for asset streaming

pub mod decompression;
pub mod dma;
pub mod ssd;
