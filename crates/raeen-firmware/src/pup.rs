//! Firmware package access — opens an SLB2/PUP container and exposes entries.
//!
//! The real `PS5UPDATE.PUP` is ~1.2 GB, so [`Firmware::open`] memory-maps the
//! file rather than reading it into RAM. Reads return borrowed slices into the
//! mapping; no entry payload is copied or decrypted here.

use crate::slb2::{Slb2Entry, parse_slb2};
use raeen_core::error::{FirmwareError, LoaderError};
use std::path::Path;

enum Backing {
    Mmap(memmap2::Mmap),
    Bytes(Vec<u8>),
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => m,
            Backing::Bytes(b) => b,
        }
    }
}

/// An opened PS5 firmware package (SLB2 container).
pub struct Firmware {
    backing: Backing,
    entries: Vec<Slb2Entry>,
}

impl Firmware {
    /// Open a firmware package from a file path (memory-mapped, read-only).
    ///
    /// # Errors
    ///
    /// I/O failures surface as [`FirmwareError::Loader`]; a bad container
    /// surfaces as [`FirmwareError::InvalidPupMagic`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FirmwareError> {
        let file = std::fs::File::open(path.as_ref()).map_err(LoaderError::from)?;
        // SAFETY: `Mmap::map` is unsafe because the mapping aliases the file's
        // bytes: if the backing file is modified or truncated by another
        // process while mapped, reads can observe torn data or fault (SIGBUS).
        // Raeen accepts this risk for a user-supplied, locally-owned firmware
        // file it only reads; it never writes through the mapping.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(LoaderError::from)?;
        let entries = parse_slb2(&mmap)?;
        Ok(Self {
            backing: Backing::Mmap(mmap),
            entries,
        })
    }

    /// Construct from an in-memory buffer (tests, piped input).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, FirmwareError> {
        let entries = parse_slb2(&bytes)?;
        Ok(Self {
            backing: Backing::Bytes(bytes),
            entries,
        })
    }

    /// The container's entries.
    pub fn entries(&self) -> &[Slb2Entry] {
        &self.entries
    }

    /// Read the raw (possibly encrypted) bytes of entry `index`.
    ///
    /// # Errors
    ///
    /// [`FirmwareError::PupEntryOutOfBounds`] if `index` is invalid or the
    /// entry's declared range falls outside the container.
    pub fn read_entry(&self, index: usize) -> Result<&[u8], FirmwareError> {
        let entry = self
            .entries
            .get(index)
            .ok_or(FirmwareError::PupEntryOutOfBounds { index })?;
        let data = self.backing.as_slice();
        let start = entry.offset as usize;
        let end = start
            .checked_add(entry.size as usize)
            .filter(|&e| e <= data.len())
            .ok_or(FirmwareError::PupEntryOutOfBounds { index })?;
        Ok(&data[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raeen_core::error::FirmwareError;

    /// SLB2 with one entry whose 4-byte payload ("DATA") sits at block 2.
    fn synthetic_firmware() -> Vec<u8> {
        let payload_off = 2 * 0x200usize;
        let mut buf = vec![0u8; payload_off + 4];
        buf[0..4].copy_from_slice(b"SLB2");
        buf[0x0C..0x10].copy_from_slice(&1u32.to_le_bytes()); // file_count = 1
        buf[0x20..0x24].copy_from_slice(&2u32.to_le_bytes()); // block_offset = 2
        buf[0x24..0x28].copy_from_slice(&4u32.to_le_bytes()); // size = 4
        buf[0x30..0x30 + 12].copy_from_slice(b"PS5UPDATE1.P"); // truncated name ok
        buf[payload_off..payload_off + 4].copy_from_slice(b"DATA");
        buf
    }

    #[test]
    fn from_bytes_enumerates_entries() {
        let fw = Firmware::from_bytes(synthetic_firmware()).unwrap();
        assert_eq!(fw.entries().len(), 1);
        assert_eq!(fw.entries()[0].size, 4);
    }

    #[test]
    fn read_entry_returns_payload() {
        let fw = Firmware::from_bytes(synthetic_firmware()).unwrap();
        assert_eq!(fw.read_entry(0).unwrap(), b"DATA");
    }

    #[test]
    fn read_entry_out_of_range_index_errors() {
        let fw = Firmware::from_bytes(synthetic_firmware()).unwrap();
        assert!(matches!(
            fw.read_entry(5),
            Err(FirmwareError::PupEntryOutOfBounds { index: 5 })
        ));
    }
}
