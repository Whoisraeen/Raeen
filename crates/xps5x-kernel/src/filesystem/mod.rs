//! Virtual filesystem for the emulated PS5.
//!
//! Maps PS5 mount points to host directories:
//! - `/app0/`      → Game data directory
//! - `/savedata0/` → Save data directory
//! - `/system/`    → Firmware modules
//! - `/dev/`       → Device files (stubbed)
//! - `/proc/`      → Process info (stubbed)

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{debug, info, warn};
use xps5x_core::types::Fd;

/// A mount point mapping PS5 paths to host paths.
#[derive(Debug, Clone)]
struct MountPoint {
    /// PS5 path prefix (e.g., "/app0/").
    ps5_prefix: String,
    /// Host directory this maps to.
    host_path: PathBuf,
}

/// Orbis/BSD `open` flags (the subset the VFS honors). Values match the
/// PS5's FreeBSD-derived libkernel.
pub mod open_flags {
    pub const O_RDONLY: i32 = 0x0000;
    pub const O_WRONLY: i32 = 0x0001;
    pub const O_RDWR: i32 = 0x0002;
    pub const O_ACCMODE: i32 = 0x0003;
    pub const O_APPEND: i32 = 0x0008;
    pub const O_CREAT: i32 = 0x0200;
    pub const O_TRUNC: i32 = 0x0400;
}

/// An open file descriptor.
#[derive(Debug)]
#[allow(dead_code)] // fd/ps5_path recorded for fstat, re-open, and debug tooling
struct OpenFile {
    /// File descriptor number.
    fd: Fd,
    /// Host file path.
    host_path: PathBuf,
    /// PS5 path (for debugging).
    ps5_path: String,
    /// Current file position.
    position: u64,
    /// File data (cached in memory; the write-back buffer for a writable fd).
    data: Option<Vec<u8>>,
    /// Whether the fd was opened for writing (`O_WRONLY`/`O_RDWR`).
    writable: bool,
    /// Whether `data` has unflushed writes to persist to `host_path` on close.
    dirty: bool,
}

/// Virtual filesystem mapping PS5 paths to host directories.
pub struct VirtualFileSystem {
    /// Registered mount points.
    mounts: RwLock<Vec<MountPoint>>,
    /// Open file descriptors.
    open_files: RwLock<HashMap<Fd, OpenFile>>,
    /// Next file descriptor to assign.
    next_fd: RwLock<Fd>,
}

impl Default for VirtualFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualFileSystem {
    /// Create a new VFS with default mount points.
    pub fn new() -> Self {
        info!("Initializing virtual filesystem");
        let vfs = Self {
            mounts: RwLock::new(Vec::new()),
            open_files: RwLock::new(HashMap::new()),
            next_fd: RwLock::new(3), // 0=stdin, 1=stdout, 2=stderr.
        };

        // Register standard PS5 mount points with default paths.
        vfs.mount("/app0/", "games/current");
        vfs.mount("/savedata0/", "savedata");
        vfs.mount("/system/", "firmware");
        vfs.mount("/dev/", ""); // Stubbed.
        vfs.mount("/proc/", ""); // Stubbed.

        vfs
    }

    /// Register a mount point.
    pub fn mount(&self, ps5_prefix: &str, host_path: &str) {
        debug!("VFS mount: '{}' -> '{}'", ps5_prefix, host_path);
        self.mounts.write().push(MountPoint {
            ps5_prefix: ps5_prefix.to_string(),
            host_path: PathBuf::from(host_path),
        });
    }

    /// Set the game directory for /app0/.
    pub fn set_game_directory(&self, path: &std::path::Path) {
        let mut mounts = self.mounts.write();
        for mount in mounts.iter_mut() {
            if mount.ps5_prefix == "/app0/" {
                mount.host_path = path.to_path_buf();
                info!("VFS: /app0/ -> {}", path.display());
                return;
            }
        }
    }

    /// Resolve a PS5 path to a host path.
    pub fn resolve_path(&self, ps5_path: &str) -> Option<PathBuf> {
        let mounts = self.mounts.read();
        for mount in mounts.iter() {
            if ps5_path.starts_with(&mount.ps5_prefix) {
                let relative = &ps5_path[mount.ps5_prefix.len()..];
                return Some(mount.host_path.join(relative));
            }
        }
        None
    }

    /// Open a file, honoring the `open`-flag subset in [`open_flags`].
    ///
    /// Write confinement: a guest path containing a `..` component is
    /// rejected (`PermissionDenied`) so a writable open can't escape its
    /// mount's host directory via traversal — guest paths are otherwise
    /// untrusted. Read-only opens of existing files are unaffected.
    pub fn open(&self, path: &str, flags: i32, _mode: u32) -> Result<Fd, std::io::Error> {
        use open_flags::*;

        let writable = matches!(flags & O_ACCMODE, O_WRONLY | O_RDWR);
        let create = flags & O_CREAT != 0;
        let truncate = flags & O_TRUNC != 0;
        let append = flags & O_APPEND != 0;

        // Reject path traversal on any writable open (defense against a guest
        // writing outside its mount via "../"). Read-only opens don't persist
        // anything, so they don't need this guard.
        if writable && path.split(['/', '\\']).any(|c| c == "..") {
            warn!("VFS open: refusing writable open of traversing path '{path}'");
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "path traversal",
            ));
        }

        let host_path = self
            .resolve_path(path)
            .unwrap_or_else(|| PathBuf::from(path));

        let exists = host_path.exists();
        let data = if exists && !truncate {
            Some(std::fs::read(&host_path)?)
        } else if exists || create {
            // Truncated existing file, or a new file being created: start empty.
            Some(Vec::new())
        } else {
            debug!("VFS: file not found on host: {}", host_path.display());
            None
        };

        // O_CREAT of a missing file: ensure the parent dir exists so the
        // eventual flush-on-close succeeds.
        if create
            && !exists
            && let Some(parent) = host_path.parent()
        {
            let _ = std::fs::create_dir_all(parent);
        }

        let mut next = self.next_fd.write();
        let fd = *next;
        *next += 1;

        let position = if append {
            data.as_ref().map_or(0, |d| d.len() as u64)
        } else {
            0
        };
        // A newly-created or truncated writable file is dirty immediately so
        // it persists even if closed with zero bytes written.
        let dirty = writable && ((create && !exists) || truncate);

        let file = OpenFile {
            fd,
            host_path,
            ps5_path: path.to_string(),
            position,
            data,
            writable,
            dirty,
        };

        debug!("VFS open: '{path}' -> fd={fd} (writable={writable}, create={create})");
        self.open_files.write().insert(fd, file);
        Ok(fd)
    }

    /// Read from an open file descriptor.
    pub fn read(&self, fd: Fd, count: usize) -> Result<Vec<u8>, std::io::Error> {
        let mut files = self.open_files.write();
        if let Some(file) = files.get_mut(&fd) {
            if let Some(ref data) = file.data {
                let pos = file.position as usize;
                let end = (pos + count).min(data.len());
                let result = data[pos..end].to_vec();
                file.position = end as u64;
                Ok(result)
            } else {
                Ok(Vec::new())
            }
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {} not open", fd),
            ))
        }
    }

    /// Write `bytes` to an open, writable descriptor at its current position
    /// (extending the file as needed), advancing the position and marking the
    /// fd dirty so [`close`](Self::close) flushes it to the host. A read-only
    /// fd is rejected with `PermissionDenied`; an unknown fd with `NotFound`.
    pub fn write(&self, fd: Fd, bytes: &[u8]) -> Result<usize, std::io::Error> {
        let mut files = self.open_files.write();
        let Some(file) = files.get_mut(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        if !file.writable {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "fd not opened for writing",
            ));
        }
        let buf = file.data.get_or_insert_with(Vec::new);
        let pos = file.position as usize;
        if pos > buf.len() {
            buf.resize(pos, 0); // sparse gap → zero-fill
        }
        let end = pos + bytes.len();
        if end > buf.len() {
            buf.resize(end, 0);
        }
        buf[pos..end].copy_from_slice(bytes);
        file.position = end as u64;
        file.dirty = true;
        debug!("VFS write: fd={fd}, {} bytes at {pos}", bytes.len());
        Ok(bytes.len())
    }

    /// Reposition an open file descriptor. `whence` follows POSIX:
    /// `SEEK_SET` (0) = absolute, `SEEK_CUR` (1) = relative to current,
    /// `SEEK_END` (2) = relative to end-of-file. Returns the new absolute
    /// position, or an error for a bad fd / negative resulting offset.
    pub fn seek(&self, fd: Fd, offset: i64, whence: i32) -> Result<u64, std::io::Error> {
        let mut files = self.open_files.write();
        let Some(file) = files.get_mut(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        let size = file.data.as_ref().map_or(0u64, |d| d.len() as u64);
        let base = match whence {
            0 => 0i64,                 // SEEK_SET
            1 => file.position as i64, // SEEK_CUR
            2 => size as i64,          // SEEK_END
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "bad whence",
                ));
            }
        };
        let target = base
            .checked_add(offset)
            .filter(|t| *t >= 0)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "seek before start of file",
                )
            })?;
        file.position = target as u64;
        Ok(file.position)
    }

    /// The size in bytes of an open file's backing data (0 if the host file
    /// was absent at open time). Used by `fstat`/`lseek(SEEK_END)`.
    pub fn file_size(&self, fd: Fd) -> Option<u64> {
        self.open_files
            .read()
            .get(&fd)
            .map(|f| f.data.as_ref().map_or(0, |d| d.len() as u64))
    }

    /// Close a file descriptor, flushing a dirty writable fd's buffer back to
    /// its host file. A flush failure is logged but does not fail the close
    /// (the fd is still removed), matching the pragmatic behavior most guests
    /// expect from `close`.
    pub fn close(&self, fd: Fd) -> Result<(), std::io::Error> {
        let Some(file) = self.open_files.write().remove(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        if file.dirty && file.writable {
            if let Some(ref data) = file.data {
                match std::fs::write(&file.host_path, data) {
                    Ok(()) => debug!(
                        "VFS close: flushed {} bytes -> {}",
                        data.len(),
                        file.host_path.display()
                    ),
                    Err(e) => warn!(
                        "VFS close: failed to persist {}: {e}",
                        file.host_path.display()
                    ),
                }
            }
        } else {
            debug!("VFS close: fd={fd}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xps5x-vfs-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn create_write_close_persists_to_host_then_reads_back() {
        use open_flags::*;
        let dir = temp_dir("persist");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        // Create + write "save data", close → must persist to the host file.
        let fd = vfs
            .open("/app0/save.bin", O_WRONLY | O_CREAT | O_TRUNC, 0o644)
            .unwrap();
        assert_eq!(vfs.write(fd, b"SAVEDATA").unwrap(), 8);
        vfs.close(fd).unwrap();
        assert_eq!(std::fs::read(dir.join("save.bin")).unwrap(), b"SAVEDATA");

        // Re-open read-only and read it back through the VFS.
        let fd2 = vfs.open("/app0/save.bin", O_RDONLY, 0).unwrap();
        assert_eq!(vfs.read(fd2, 8).unwrap(), b"SAVEDATA");
        vfs.close(fd2).unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_to_readonly_fd_is_rejected() {
        use open_flags::*;
        let dir = temp_dir("ro");
        std::fs::write(dir.join("f.bin"), b"x").unwrap();
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        let fd = vfs.open("/app0/f.bin", O_RDONLY, 0).unwrap();
        assert!(
            vfs.write(fd, b"nope").is_err(),
            "read-only fd must reject writes"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writable_open_of_traversing_path_is_refused() {
        use open_flags::*;
        let vfs = VirtualFileSystem::new();
        let err = vfs
            .open("/app0/../../escape.bin", O_WRONLY | O_CREAT, 0o644)
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn seek_then_write_extends_with_zero_fill() {
        use open_flags::*;
        let dir = temp_dir("sparse");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        let fd = vfs
            .open("/app0/sparse.bin", O_WRONLY | O_CREAT | O_TRUNC, 0o644)
            .unwrap();
        assert_eq!(vfs.seek(fd, 4, 0).unwrap(), 4); // SEEK_SET past start
        vfs.write(fd, b"AB").unwrap();
        vfs.close(fd).unwrap();
        // 4 zero bytes + "AB".
        assert_eq!(
            std::fs::read(dir.join("sparse.bin")).unwrap(),
            b"\0\0\0\0AB"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
