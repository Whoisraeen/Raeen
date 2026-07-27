//! Memory-mapped file reads for large guest images.
//!
//! Game eboots, NEEDED `.prx`/`.sprx` modules, and packages run to hundreds
//! of megabytes; `fs::read` copies all of it through a heap buffer before a
//! single header byte is parsed. [`MappedFile`] maps the file instead and
//! dereferences to `&[u8]`, so parse/link code is byte-for-byte unchanged
//! while the OS pages in only what is touched. Mapping failures (exotic
//! filesystems, zero-length files) quietly fall back to a buffered read —
//! never a new failure mode.

use std::ops::Deref;
use std::path::Path;

/// File contents, memory-mapped when possible, read into memory otherwise.
/// Either way it derefs to `&[u8]` for the lifetime of the value.
pub enum MappedFile {
    Mapped(memmap2::Mmap),
    Buffered(Vec<u8>),
}

impl MappedFile {
    /// Open and map `path` read-only, falling back to `fs::read`.
    ///
    /// The map stays valid only while the underlying file is left alone —
    /// truncating or rewriting a game file mid-launch invalidates pages the
    /// OS may still fault in. That is the standard mmap contract and an
    /// acceptable trust assumption for the user's own installed games; the
    /// buffered fallback path has no such caveat.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        match std::fs::File::open(path) {
            Ok(file) => {
                // SAFETY: read-only shared map of a file we just opened; see
                // the doc comment for the external-mutation contract.
                match unsafe { memmap2::Mmap::map(&file) } {
                    Ok(map) => Ok(Self::Mapped(map)),
                    // Zero-length files and unmappable filesystems land here.
                    Err(_) => std::fs::read(path).map(Self::Buffered),
                }
            }
            Err(e) => Err(e),
        }
    }
}

impl Deref for MappedFile {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map,
            Self::Buffered(bytes) => bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_a_real_file_and_derefs_to_its_bytes() {
        let path = std::env::temp_dir().join(format!("raeen-mapped-{}.bin", std::process::id()));
        std::fs::write(&path, b"\x7fELF-mapped-test").expect("write fixture");
        let mapped = MappedFile::open(&path).expect("open");
        assert_eq!(&mapped[..4], b"\x7fELF");
        assert_eq!(mapped.len(), 16);
        drop(mapped);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_file_falls_back_to_buffered_without_error() {
        let path = std::env::temp_dir().join(format!("raeen-mapped-empty-{}", std::process::id()));
        std::fs::write(&path, b"").expect("write fixture");
        let mapped = MappedFile::open(&path).expect("open");
        assert!(mapped.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_an_io_error() {
        assert!(MappedFile::open(Path::new("no/such/file/raeen")).is_err());
    }
}
