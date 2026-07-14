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
use tracing::{debug, info};
use xps5x_core::types::Fd;

/// A mount point mapping PS5 paths to host paths.
#[derive(Debug, Clone)]
struct MountPoint {
    /// PS5 path prefix (e.g., "/app0/").
    ps5_prefix: String,
    /// Host directory this maps to.
    host_path: PathBuf,
}

/// An open file descriptor.
#[derive(Debug)]
#[allow(dead_code)] // fd/host_path/ps5_path recorded for pending fstat, re-open, and debug tooling
struct OpenFile {
    /// File descriptor number.
    fd: Fd,
    /// Host file path.
    host_path: PathBuf,
    /// PS5 path (for debugging).
    ps5_path: String,
    /// Current file position.
    position: u64,
    /// File data (cached in memory for simplicity).
    data: Option<Vec<u8>>,
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

    /// Open a file.
    pub fn open(&self, path: &str, _flags: i32, _mode: u32) -> Result<Fd, std::io::Error> {
        let host_path = self
            .resolve_path(path)
            .unwrap_or_else(|| PathBuf::from(path));

        let mut next = self.next_fd.write();
        let fd = *next;
        *next += 1;

        let data = if host_path.exists() {
            Some(std::fs::read(&host_path)?)
        } else {
            debug!("VFS: file not found on host: {}", host_path.display());
            None
        };

        let file = OpenFile {
            fd,
            host_path,
            ps5_path: path.to_string(),
            position: 0,
            data,
        };

        debug!("VFS open: '{}' -> fd={}", path, fd);
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

    /// Write to an open file descriptor.
    pub fn write(&self, fd: Fd, data: &[u8]) -> Result<usize, std::io::Error> {
        // For now, just acknowledge the write.
        debug!("VFS write: fd={}, {} bytes", fd, data.len());
        Ok(data.len())
    }

    /// Close a file descriptor.
    pub fn close(&self, fd: Fd) -> Result<(), std::io::Error> {
        if self.open_files.write().remove(&fd).is_some() {
            debug!("VFS close: fd={}", fd);
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {} not open", fd),
            ))
        }
    }
}
