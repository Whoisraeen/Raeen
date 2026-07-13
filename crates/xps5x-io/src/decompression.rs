//! Decompression engine — software replacement for PS5's hardware Kraken decompressor.
//!
//! The PS5 has dedicated silicon for decompressing Kraken (Oodle) streams.
//! On PC, we do this in software using LZ4 and Zstd as approximations
//! (Oodle is proprietary and not available for free use).

use tracing::{debug, warn};

/// Supported decompression algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression.
    None,
    /// LZ4 fast decompression.
    Lz4,
    /// Zstandard decompression.
    Zstd,
    /// Oodle Kraken (PS5 native — requires proprietary library).
    Kraken,
}

/// Decompress a data block.
pub fn decompress(
    data: &[u8],
    compression: CompressionType,
    uncompressed_size: usize,
) -> Result<Vec<u8>, xps5x_core::error::IoError> {
    match compression {
        CompressionType::None => Ok(data.to_vec()),
        CompressionType::Lz4 => {
            debug!("LZ4 decompress: {} -> {} bytes", data.len(), uncompressed_size);
            lz4_flex::decompress(data, uncompressed_size).map_err(|e| {
                xps5x_core::error::IoError::DecompressionFailed(format!("LZ4: {}", e))
            })
        }
        CompressionType::Zstd => {
            debug!("Zstd decompress: {} bytes", data.len());
            zstd::stream::decode_all(data).map_err(|e| {
                xps5x_core::error::IoError::DecompressionFailed(format!("Zstd: {}", e))
            })
        }
        CompressionType::Kraken => {
            warn!("Kraken decompression not available — Oodle is proprietary");
            Err(xps5x_core::error::IoError::DecompressionFailed(
                "Oodle Kraken decompression requires proprietary library".to_string(),
            ))
        }
    }
}
