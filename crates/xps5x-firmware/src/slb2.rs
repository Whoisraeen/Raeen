//! SLB2 firmware container parser.
//!
//! PS5 update packages are wrapped in a plaintext "SLB2" container: a fixed
//! `0x20`-byte header followed by a table of `0x30`-byte entries, each naming
//! an inner file (e.g. `PS5UPDATE1.PUP`) with its block offset and byte size.
//! Entry *contents* may be encrypted; this parser only reads the container
//! structure and never attempts decryption.

use xps5x_core::error::FirmwareError;

const SLB2_MAGIC: [u8; 4] = *b"SLB2";
const SLB2_BLOCK_SIZE: u64 = 0x200; // 512 bytes
const SLB2_HEADER_SIZE: usize = 0x20;
const SLB2_ENTRY_SIZE: usize = 0x30;
const SLB2_NAME_OFFSET: usize = 0x10; // within an entry
const SLB2_NAME_LEN: usize = 0x20;

/// A parsed SLB2 container entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slb2Entry {
    /// Inner file name (e.g. "PS5UPDATE1.PUP").
    pub name: String,
    /// Byte offset of the entry data within the container.
    pub offset: u64,
    /// Size of the entry data in bytes.
    pub size: u64,
}

/// Parse the SLB2 header and entry table from `data`.
///
/// `data` must contain at least the header and full entry table; entry
/// payloads may lie beyond the slice and are not required here.
///
/// # Errors
///
/// - [`FirmwareError::InvalidPupMagic`] if the magic is not `"SLB2"`.
/// - [`FirmwareError::PupEntryOutOfBounds`] if an entry record is truncated.
pub fn parse_slb2(data: &[u8]) -> Result<Vec<Slb2Entry>, FirmwareError> {
    if data.len() < SLB2_HEADER_SIZE || data[0..4] != SLB2_MAGIC {
        let magic = if data.len() >= 4 {
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        } else {
            0
        };
        return Err(FirmwareError::InvalidPupMagic(magic));
    }

    let file_count = u32::from_le_bytes(data[0x0C..0x10].try_into().unwrap()) as usize;

    // A valid container's entry table must fit within the data we were given.
    // Guard against a malformed `file_count` (attacker-controlled, read from
    // the header) driving a huge pre-allocation that would abort the process.
    let max_entries = (data.len() - SLB2_HEADER_SIZE) / SLB2_ENTRY_SIZE;
    if file_count > max_entries {
        return Err(FirmwareError::PupEntryOutOfBounds { index: max_entries });
    }

    let mut entries = Vec::with_capacity(file_count);
    for index in 0..file_count {
        let base = SLB2_HEADER_SIZE + index * SLB2_ENTRY_SIZE;
        if base + SLB2_ENTRY_SIZE > data.len() {
            return Err(FirmwareError::PupEntryOutOfBounds { index });
        }
        let block_offset = u32::from_le_bytes(data[base..base + 4].try_into().unwrap()) as u64;
        let size = u32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap()) as u64;
        let name_start = base + SLB2_NAME_OFFSET;
        let name = data[name_start..name_start + SLB2_NAME_LEN]
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as char)
            .collect::<String>();
        entries.push(Slb2Entry {
            name,
            offset: block_offset * SLB2_BLOCK_SIZE,
            size,
        });
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xps5x_core::error::FirmwareError;

    /// Build a synthetic SLB2 container with one entry.
    fn synthetic_slb2() -> Vec<u8> {
        let mut buf = vec![0u8; 0x20 + 0x30];
        buf[0..4].copy_from_slice(b"SLB2");
        buf[4..8].copy_from_slice(&3u32.to_le_bytes()); // version
        buf[0x0C..0x10].copy_from_slice(&1u32.to_le_bytes()); // file_count = 1
        // entry 0 at 0x20
        buf[0x20..0x24].copy_from_slice(&2u32.to_le_bytes()); // block_offset = 2
        buf[0x24..0x28].copy_from_slice(&0x100u32.to_le_bytes()); // size = 256
        let name = b"PS5UPDATE1.PUP";
        buf[0x30..0x30 + name.len()].copy_from_slice(name);
        buf
    }

    #[test]
    fn parses_single_entry() {
        let entries = parse_slb2(&synthetic_slb2()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "PS5UPDATE1.PUP");
        assert_eq!(entries[0].offset, 2 * 0x200);
        assert_eq!(entries[0].size, 0x100);
    }

    #[test]
    fn rejects_bad_magic() {
        let data = [0u8; 0x20];
        assert!(matches!(
            parse_slb2(&data),
            Err(FirmwareError::InvalidPupMagic(_))
        ));
    }

    #[test]
    fn rejects_truncated_entry_table() {
        let mut buf = synthetic_slb2();
        buf.truncate(0x20 + 0x10); // header + half an entry
        assert!(matches!(
            parse_slb2(&buf),
            Err(FirmwareError::PupEntryOutOfBounds { index: 0 })
        ));
    }

    #[test]
    fn rejects_absurd_file_count_without_aborting() {
        // Valid magic but a malformed, attacker-controlled file_count that far
        // exceeds what the buffer can hold. Must return a clean error rather
        // than pre-allocating ~170 GB and aborting the process.
        let mut buf = vec![0u8; 0x20];
        buf[0..4].copy_from_slice(b"SLB2");
        buf[0x0C..0x10].copy_from_slice(&u32::MAX.to_le_bytes()); // file_count = 0xFFFFFFFF
        assert!(matches!(
            parse_slb2(&buf),
            Err(FirmwareError::PupEntryOutOfBounds { .. })
        ));
    }
}
