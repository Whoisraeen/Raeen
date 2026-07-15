//! SELF (Signed ELF) parser for PS5 executables.
//!
//! PS5 retail executables are distributed as SELF files — an encrypted
//! and signed wrapper around a standard ELF64 binary. The SELF format
//! adds authentication headers, segment encryption metadata, and
//! digital signatures.
//!
//! XPS5X can only load **decrypted** SELF files (or the inner ELF
//! extracted from a decrypted SELF). Retail encrypted SELFs require
//! the user to decrypt them externally using their own hardware keys.

use crate::LoadedBinary;
use tracing::{debug, info, warn};
use xps5x_core::error::LoaderError;

/// SELF file magic: 0x4F15D17E ("OISED" / "SELF" rearranged). This project's
/// in-tree fixtures use it; see [`is_self_magic`] for the real-hardware ones.
pub const SELF_MAGIC: u32 = 0x4F15D17E;

/// The SCE SELF magic used by PS4/PS5 titles (`4F 15 3D 1D` on disk).
pub const SELF_MAGIC_SCE: u32 = 0x1D3D154F;

/// The SELF magic observed on **PS5 (Prospero) title** `eboot.bin`/`.prx`
/// files (`54 14 F5 EE` on disk).
///
/// Same container layout as every other SELF — a 0x20-byte header, then
/// `num_entries` 0x20-byte segment entries, then the inner ELF — only the
/// magic differs, so it is accepted as a container-format discriminator, not
/// special-cased anywhere downstream. (Verified against a real title: header
/// `num_entries = 12`, and `0x20 + 12*0x20 = 0x1A0` lands exactly on the
/// inner `\x7fELF`.) This is public container-format information, not a key.
pub const SELF_MAGIC_PROSPERO: u32 = 0xEEF5_1454;

/// Whether `magic` names a SELF container this loader understands.
///
/// All variants share one layout; only the discriminator differs. Accepting
/// the real-hardware magics (not just the in-tree fixture one) is what lets
/// actual titles be parsed at all — nothing here decrypts anything.
pub fn is_self_magic(magic: u32) -> bool {
    matches!(magic, SELF_MAGIC | SELF_MAGIC_SCE | SELF_MAGIC_PROSPERO)
}

#[cfg(test)]
mod real_format_tests {
    use super::*;

    /// Pins the real PS4/PS5 `properties` bit layout against a **real title's**
    /// observed values, so the flags can never silently revert to being read as
    /// an enum again.
    ///
    /// `0x2804` and `0x110004` are verbatim from a retail PS5 `eboot.bin`
    /// (Prospero SELF, 12 entries). Both are *plaintext* — neither sets bit 1 —
    /// but both set bit 2 (`signed`). The previous accessor tested bits 1-3
    /// together, so it called them encrypted and the loader demanded keys that
    /// cannot exist for an already-decrypted dump.
    #[test]
    fn real_title_property_bits_decode_as_independent_flags() {
        // A real blocked *data* segment: signed + blocked, phdr index 0.
        let data_seg = SelfEntry {
            properties: 0x2804,
            offset: 0x5c930,
            compressed_size: 0xb7b44ec,
            uncompressed_size: 0xb7b44ec,
        };
        assert!(!data_seg.is_encrypted(), "bit 1 clear => plaintext");
        assert!(!data_seg.is_compressed(), "bit 3 clear => uncompressed");
        assert!(data_seg.is_blocked(), "bit 11 set => blocked segment data");
        assert_eq!(data_seg.segment_index(), 0, "bits 20+ => ELF phdr index");

        // A real block/digest table entry: signed, NOT blocked.
        let table = SelfEntry {
            properties: 0x110004,
            offset: 0xb70,
            compressed_size: 0x5bdc0,
            uncompressed_size: 0x5bdc0,
        };
        assert!(!table.is_encrypted());
        assert!(
            !table.is_blocked(),
            "not blocked => a table, not segment data"
        );
        assert_eq!(table.segment_index(), 1);
    }

    /// The real-hardware magics must be accepted, not just the fixture one —
    /// otherwise every actual title is rejected at byte 0.
    #[test]
    fn real_self_magics_are_accepted() {
        assert!(is_self_magic(SELF_MAGIC), "in-tree fixture magic");
        assert!(is_self_magic(SELF_MAGIC_SCE), "PS4/PS5 SCE SELF magic");
        assert!(
            is_self_magic(SELF_MAGIC_PROSPERO),
            "PS5 title magic (observed on a real eboot.bin)"
        );
        assert!(!is_self_magic(0x464C457F), "a bare ELF is not a SELF");
        assert!(!is_self_magic(0xDEAD_BEEF));
    }
}

/// SELF header structure.
/// This is the outer container that wraps the inner ELF.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SelfHeader {
    /// Magic number (0x4F15D17E).
    pub magic: u32,
    /// SELF version.
    pub version: u8,
    /// Mode (0 = PS5).
    pub mode: u8,
    /// Endianness (1 = little-endian).
    pub endian: u8,
    /// Attributes.
    pub attributes: u8,
    /// Key type.
    pub key_type: u32,
    /// Header size (total size of SELF headers before ELF data).
    pub header_size: u16,
    /// Metadata size.
    pub meta_size: u16,
    /// File size (total SELF file size).
    pub file_size: u64,
    /// Number of SELF segments (entries).
    pub num_entries: u16,
    /// Flags.
    pub flags: u16,
    /// Padding.
    pub _padding: u32,
}

/// SELF segment entry — describes how each ELF segment is stored.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SelfEntry {
    /// Properties (encrypted, compressed, etc.).
    pub properties: u64,
    /// Offset of segment data within the SELF file.
    pub offset: u64,
    /// Compressed size (or file size if uncompressed).
    pub compressed_size: u64,
    /// Uncompressed size.
    pub uncompressed_size: u64,
}

impl SelfEntry {
    /// Whether this segment's data is encrypted — real SELF `properties`
    /// **bit 1**.
    ///
    /// The bit layout is the actual PS4/PS5 one: bit 0 `ordered`, **bit 1
    /// `encrypted`**, bit 2 `signed`, bit 3 `compressed`, bit 11 `blocked`,
    /// bits 20+ the segment id ([`Self::segment_index`]). These are
    /// *independent flags*, not an enum.
    ///
    /// This previously tested `(properties >> 1) & 0x7`, i.e. bits 1-3 —
    /// folding `signed` and `compressed` into "encrypted". That only ever
    /// looked right because every in-tree fixture used `properties = 0`. Real
    /// titles set `signed` on plaintext segments (e.g. `0x2804` = signed +
    /// blocked, id 0), so the old test reported them as encrypted and the
    /// loader demanded keys that do not exist for an already-decrypted dump —
    /// making every real SELF unloadable.
    pub fn is_encrypted(&self) -> bool {
        self.properties & 0x2 != 0
    }

    /// Whether this segment's data is compressed — real SELF `properties`
    /// **bit 3** (was previously read from bit 8, which is not the
    /// compression bit). See [`Self::is_encrypted`] for the full layout.
    pub fn is_compressed(&self) -> bool {
        self.properties & 0x8 != 0
    }

    /// Whether this segment is stored "blocked" — real SELF `properties`
    /// **bit 11**. Blocked segments are accompanied by a separate block/digest
    /// table entry; see [`Self::is_encrypted`] for the full layout.
    pub fn is_blocked(&self) -> bool {
        self.properties & 0x800 != 0
    }

    /// Check if this segment has data (non-zero size).
    pub fn has_data(&self) -> bool {
        self.compressed_size > 0 || self.uncompressed_size > 0
    }

    /// Get the segment index from properties.
    pub fn segment_index(&self) -> u64 {
        self.properties >> 20
    }
}

/// Parse a SELF (Signed ELF) file.
///
/// This function handles both:
/// 1. **Decrypted SELF** — Extracts the inner ELF and delegates to the ELF parser.
/// 2. **Encrypted SELF** — Returns an error indicating decryption is required.
///
/// # Errors
///
/// Returns `LoaderError::InvalidSelfMagic` if the magic doesn't match,
/// or `LoaderError::EncryptedSelf` if the file contains encrypted segments.
pub fn parse_self(data: &[u8]) -> Result<LoadedBinary, LoaderError> {
    if data.len() < std::mem::size_of::<SelfHeader>() {
        return Err(LoaderError::InvalidSelfMagic(0));
    }

    // Read and validate the magic.
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if !is_self_magic(magic) {
        return Err(LoaderError::InvalidSelfMagic(magic));
    }

    info!("Parsing SELF file ({} bytes)", data.len());

    // Parse the header manually (avoiding unsafe transmute).
    let version = data[4];
    let mode = data[5];
    let _endian = data[6];
    let _attributes = data[7];
    let _key_type = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let header_size = u16::from_le_bytes([data[12], data[13]]);
    let _meta_size = u16::from_le_bytes([data[14], data[15]]);
    let file_size = u64::from_le_bytes(data[16..24].try_into().unwrap());
    let num_entries = u16::from_le_bytes([data[24], data[25]]);
    let _flags = u16::from_le_bytes([data[26], data[27]]);

    debug!(
        "SELF header: version={}, mode={}, entries={}, header_size={:#x}, file_size={:#x}",
        version, mode, num_entries, header_size, file_size
    );

    // Parse segment entries.
    let entry_offset = 32usize; // After the 32-byte header.
    let entry_size = 32usize; // Each SelfEntry is 32 bytes.
    let mut has_encrypted_segments = false;

    for i in 0..num_entries as usize {
        let base = entry_offset + i * entry_size;
        if base + entry_size > data.len() {
            warn!("SELF entry {} extends beyond file bounds", i);
            break;
        }

        let properties = u64::from_le_bytes(data[base..base + 8].try_into().unwrap());
        let offset = u64::from_le_bytes(data[base + 8..base + 16].try_into().unwrap());
        let compressed_size = u64::from_le_bytes(data[base + 16..base + 24].try_into().unwrap());
        let uncompressed_size = u64::from_le_bytes(data[base + 24..base + 32].try_into().unwrap());

        let entry = SelfEntry {
            properties,
            offset,
            compressed_size,
            uncompressed_size,
        };

        debug!(
            "  Entry {}: idx={}, offset={:#x}, compressed={:#x}, uncompressed={:#x}, encrypted={}, compressed={}",
            i,
            entry.segment_index(),
            entry.offset,
            entry.compressed_size,
            entry.uncompressed_size,
            entry.is_encrypted(),
            entry.is_compressed()
        );

        if entry.is_encrypted() && entry.has_data() {
            has_encrypted_segments = true;
        }
    }

    if has_encrypted_segments {
        warn!("SELF file contains encrypted segments — these must be decrypted externally");
        return Err(LoaderError::EncryptedSelf);
    }

    // The inner ELF starts after the SELF headers.
    let elf_offset = header_size as usize;
    if elf_offset >= data.len() {
        return Err(LoaderError::SegmentLoadFailed {
            address: 0,
            size: 0,
            reason: format!(
                "SELF header_size ({:#x}) exceeds file size ({:#x})",
                elf_offset,
                data.len()
            ),
        });
    }

    info!("Extracting inner ELF from SELF at offset {:#x}", elf_offset);
    let inner_elf_data = &data[elf_offset..];

    // Delegate to the ELF parser.
    crate::elf::parse_elf(inner_elf_data)
}

/// Auto-detect file format and parse accordingly.
///
/// Checks the magic bytes to determine if the file is a SELF or plain ELF,
/// then dispatches to the appropriate parser.
pub fn load_binary(data: &[u8]) -> Result<LoadedBinary, LoaderError> {
    if data.len() < 4 {
        return Err(LoaderError::InvalidElfMagic(0));
    }

    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);

    match magic {
        m if is_self_magic(m) => {
            info!("Detected SELF format (magic {m:#010x})");
            parse_self(data)
        }
        0x464C457F => {
            // 0x7F 'E' 'L' 'F' in little-endian.
            info!("Detected ELF format");
            crate::elf::parse_elf(data)
        }
        _ => {
            // Try PKG format.
            if data.len() >= 4
                && data[0] == 0x7F
                && data[1] == b'C'
                && data[2] == b'N'
                && data[3] == b'T'
            {
                info!("Detected PKG format — use pkg::parse_pkg() for extraction first");
                Err(LoaderError::SegmentLoadFailed {
                    address: 0,
                    size: 0,
                    reason:
                        "PKG files must be extracted before loading. Use pkg::parse_pkg() first."
                            .to_string(),
                })
            } else {
                Err(LoaderError::InvalidElfMagic(magic))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_self_magic() {
        let data = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let result = parse_self(&data);
        assert!(matches!(result, Err(LoaderError::InvalidSelfMagic(_))));
    }

    #[test]
    fn test_auto_detect_elf() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
        data[4] = 2; // ELFCLASS64
        // This will fail at the ELF parse stage but proves auto-detection works.
        let result = load_binary(&data);
        // We expect an ELF parse error (not InvalidSelfMagic), proving correct dispatch.
        assert!(!matches!(result, Err(LoaderError::InvalidSelfMagic(_))));
    }
}
