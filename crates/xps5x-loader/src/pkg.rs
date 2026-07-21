//! PKG container parser for PS5 game packages.
//!
//! PS5 games are distributed as `.pkg` files — encrypted archive
//! containers holding game executables, assets, metadata, and DLC.
//!
//! The PKG format uses a header with magic `\x7FCNT` and contains
//! an internal file table pointing to encrypted/compressed entries.
//!
//! **Note:** XPS5X only handles decrypted/extracted PKG contents.
//! Users must extract PKG files externally using appropriate tools.

use tracing::{debug, info, warn};
use xps5x_core::error::LoaderError;

/// PKG magic bytes: 0x7F 'C' 'N' 'T'.
const PKG_MAGIC: [u8; 4] = [0x7F, b'C', b'N', b'T'];

/// Finalized image header magic: 0x7F 'F' 'I' 'H'.
#[allow(dead_code)] // reserved: FIH (finalized image) parsing not yet implemented
const FIH_MAGIC: [u8; 4] = [0x7F, b'F', b'I', b'H'];

/// PKG content type IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PkgContentType {
    /// Game data (GD).
    GameData = 0x1A,
    /// Game patch/update.
    Patch = 0x1B,
    /// Additional content (DLC).
    AdditionalContent = 0x1C,
    /// Unknown content type.
    Unknown = 0xFF,
}

impl From<u32> for PkgContentType {
    fn from(value: u32) -> Self {
        match value {
            0x1A => Self::GameData,
            0x1B => Self::Patch,
            0x1C => Self::AdditionalContent,
            _ => Self::Unknown,
        }
    }
}

/// Parsed PKG header information.
#[derive(Debug, Clone)]
pub struct PkgHeader {
    /// Content ID (e.g., "UP9000-PPSA01411_00-YOURPSKGCONTENTID").
    pub content_id: String,
    /// Content type.
    pub content_type: PkgContentType,
    /// Total package size in bytes.
    pub pkg_size: u64,
    /// Number of entries in the file table.
    pub entry_count: u32,
    /// Offset to the entry table.
    pub entry_table_offset: u64,
}

/// A single entry in the PKG file table.
#[derive(Debug, Clone)]
pub struct PkgEntry {
    /// Entry ID / type.
    pub id: u32,
    /// Offset of the entry data within the PKG.
    pub offset: u64,
    /// Size of the entry data.
    pub size: u64,
    /// Filename (if available).
    pub name: Option<String>,
}

/// Parsed PKG metadata (PARAM.SFO fields).
#[derive(Debug, Clone, Default)]
pub struct PkgMetadata {
    /// Game title.
    pub title: String,
    /// Title ID (e.g., "PPSA01411").
    pub title_id: String,
    /// Application version.
    pub app_version: String,
    /// Minimum firmware version required.
    pub system_version: String,
    /// Content ID.
    pub content_id: String,
    /// Category (e.g., "gd" for game data).
    pub category: String,
}

/// Parse a PKG file header to extract metadata.
///
/// This does NOT extract or decrypt the PKG contents — it only reads
/// the header and metadata to identify the package.
///
/// # Errors
///
/// Returns `LoaderError::InvalidPkgMagic` if the magic doesn't match.
pub fn parse_pkg_header(data: &[u8]) -> Result<PkgHeader, LoaderError> {
    if data.len() < 4 {
        return Err(LoaderError::InvalidPkgMagic(0));
    }

    // Validate magic.
    if data[0..4] != PKG_MAGIC {
        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        return Err(LoaderError::InvalidPkgMagic(magic));
    }

    info!("Parsing PKG header ({} bytes)", data.len());

    // PKG header is big-endian in the outer container.
    // The header layout varies between PS4 and PS5 PKGs, but the
    // core fields are at known offsets.

    // For now, extract what we can from the header.
    let pkg_size = if data.len() >= 24 {
        u64::from_be_bytes(data[16..24].try_into().unwrap_or([0u8; 8]))
    } else {
        data.len() as u64
    };

    let entry_count = if data.len() >= 28 {
        u32::from_be_bytes(data[24..28].try_into().unwrap_or([0u8; 4]))
    } else {
        0
    };

    let entry_table_offset = if data.len() >= 48 {
        u64::from_be_bytes(data[40..48].try_into().unwrap_or([0u8; 8]))
    } else {
        0
    };

    // Content ID is typically at offset 0x40, 36 bytes.
    let content_id = if data.len() >= 0x40 + 36 {
        let raw = &data[0x40..0x40 + 36];
        String::from_utf8_lossy(raw)
            .trim_end_matches('\0')
            .to_string()
    } else {
        String::new()
    };

    debug!(
        "PKG: content_id='{}', size={:#x}, entries={}, table_offset={:#x}",
        content_id, pkg_size, entry_count, entry_table_offset
    );

    Ok(PkgHeader {
        content_id,
        content_type: PkgContentType::GameData, // Default; parsed from metadata later.
        pkg_size,
        entry_count,
        entry_table_offset,
    })
}

/// Scan a directory for extracted PKG contents and identify games.
///
/// Looks for `eboot.bin` (the main executable) and `sce_sys/param.json`
/// (game metadata) within the given directory.
pub fn scan_game_directory(path: &std::path::Path) -> Result<PkgMetadata, LoaderError> {
    info!("Scanning game directory: {}", path.display());

    let eboot_path = path.join("eboot.bin");
    if !eboot_path.exists() {
        warn!("eboot.bin not found in {}", path.display());
    }

    // Look for param.json or param.sfo.
    let param_json_path = path.join("sce_sys").join("param.json");
    let param_sfo_path = path.join("sce_sys").join("param.sfo");

    let metadata = if param_json_path.exists() {
        debug!("Found param.json at {}", param_json_path.display());
        // Parse JSON metadata.
        parse_param_json(&param_json_path)?
    } else if param_sfo_path.exists() {
        debug!("Found param.sfo at {}", param_sfo_path.display());
        // Parse SFO metadata (binary format).
        parse_param_sfo(&param_sfo_path)?
    } else {
        warn!("No param.json or param.sfo found in {}", path.display());
        PkgMetadata {
            title: path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "Unknown".to_string()),
            title_id: "UNKNOWN00000".to_string(),
            ..Default::default()
        }
    };

    info!("Game: '{}' ({})", metadata.title, metadata.title_id);
    Ok(metadata)
}

fn parse_param_json(path: &std::path::Path) -> Result<PkgMetadata, LoaderError> {
    let contents = std::fs::read_to_string(path)?;
    // Simple JSON parsing without pulling in serde_json.
    // Extract key fields with basic string matching.
    let mut metadata = PkgMetadata::default();

    // `localizedParameters` lists one `titleName` per locale, and the
    // document-level default may be any language (ASTRO.BOT's is Japanese).
    // Prefer the `en-US` entry when one exists: the first `titleName` after
    // an `"en-US"` key wins and locks the title; otherwise the last plain
    // `titleName` stands.
    let mut await_en_us_title = false;
    let mut title_locked = false;

    for line in contents.lines() {
        let line = line.trim();
        if line.contains("\"en-US\"") {
            await_en_us_title = true;
        }
        if line.contains("\"titleName\"") || line.contains("\"title\"") {
            if let Some(value) = extract_json_string_value(line) {
                if await_en_us_title {
                    metadata.title = value;
                    title_locked = true;
                    await_en_us_title = false;
                } else if !title_locked {
                    metadata.title = value;
                }
            }
        } else if line.contains("\"titleId\"") {
            if let Some(value) = extract_json_string_value(line) {
                metadata.title_id = value;
            }
        } else if line.contains("\"contentId\"") {
            if let Some(value) = extract_json_string_value(line) {
                metadata.content_id = value;
            }
        } else if line.contains("\"contentVersion\"") || line.contains("\"masterVersion\"") {
            if metadata.app_version.is_empty()
                && let Some(value) = extract_json_string_value(line)
            {
                metadata.app_version = value;
            }
        } else if line.contains("\"applicationCategoryType\"")
            && let Some(value) = extract_json_string_value(line)
        {
            metadata.category = value;
        }
    }

    Ok(metadata)
}

fn parse_param_sfo(path: &std::path::Path) -> Result<PkgMetadata, LoaderError> {
    let data = std::fs::read(path)?;
    let mut metadata = PkgMetadata::default();

    // SFO magic: 0x00505346 ("\0PSF").
    if data.len() < 20 || &data[0..4] != b"\x00PSF" {
        warn!("Invalid SFO magic in {}", path.display());
        return Ok(metadata);
    }

    // Parse SFO header.
    let key_table_offset = u32::from_le_bytes(data[8..12].try_into().unwrap_or([0; 4])) as usize;
    let data_table_offset = u32::from_le_bytes(data[12..16].try_into().unwrap_or([0; 4])) as usize;
    let num_entries = u32::from_le_bytes(data[16..20].try_into().unwrap_or([0; 4])) as usize;

    for i in 0..num_entries {
        let entry_offset = 20 + i * 16;
        if entry_offset + 16 > data.len() {
            break;
        }

        let key_offset = u16::from_le_bytes(
            data[entry_offset..entry_offset + 2]
                .try_into()
                .unwrap_or([0; 2]),
        ) as usize;
        let data_offset = u32::from_le_bytes(
            data[entry_offset + 12..entry_offset + 16]
                .try_into()
                .unwrap_or([0; 4]),
        ) as usize;

        let key_start = key_table_offset + key_offset;
        let data_start = data_table_offset + data_offset;

        if key_start >= data.len() || data_start >= data.len() {
            continue;
        }

        // Read null-terminated key string.
        let key_end = data[key_start..].iter().position(|&b| b == 0).unwrap_or(0) + key_start;
        let key = String::from_utf8_lossy(&data[key_start..key_end]).to_string();

        // Read null-terminated value string.
        let value_end = data[data_start..].iter().position(|&b| b == 0).unwrap_or(0) + data_start;
        let value = String::from_utf8_lossy(&data[data_start..value_end]).to_string();

        match key.as_str() {
            "TITLE" => metadata.title = value,
            "TITLE_ID" => metadata.title_id = value,
            "CONTENT_ID" => metadata.content_id = value,
            "APP_VER" => metadata.app_version = value,
            "SYSTEM_VER" | "PSP2_SYSTEM_VER" => metadata.system_version = value,
            "CATEGORY" => metadata.category = value,
            _ => {}
        }
    }

    Ok(metadata)
}

fn extract_json_string_value(line: &str) -> Option<String> {
    // Find the value after the colon.
    let after_colon = line.split(':').nth(1)?;
    let trimmed = after_colon
        .trim()
        .trim_matches(|c| c == '"' || c == ',' || c == ' ');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_pkg_magic() {
        let data = [0x00, 0x01, 0x02, 0x03];
        let result = parse_pkg_header(&data);
        assert!(matches!(result, Err(LoaderError::InvalidPkgMagic(_))));
    }

    #[test]
    fn test_valid_pkg_magic() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&PKG_MAGIC);
        let result = parse_pkg_header(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_json_value() {
        assert_eq!(
            extract_json_string_value(r#""title": "Demon's Souls","#),
            Some("Demon's Souls".to_string())
        );
    }
}
