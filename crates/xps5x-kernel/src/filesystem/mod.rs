//! Virtual filesystem for the emulated PS5.
//!
//! Maps PS5 mount points to host directories:
//! - `/app0/`      → Game data directory
//! - `/savedata0/` → Save data directory
//! - `/download0/` → Per-title downloaded/bootstrap data
//! - `/temp0/`     → Per-title temporary data
//! - `/system/`    → Firmware modules
//! - `/dev/`       → Device files (stubbed)
//! - `/proc/`      → Process info (stubbed)

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
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
    /// Original Orbis open flags (queried/updated through `fcntl`).
    flags: i32,
    /// Sorted host directory entries for directory descriptors.
    directory_entries: Option<Vec<DirectoryEntry>>,
    /// Next directory entry returned by `getdents`.
    directory_index: usize,
}

#[derive(Debug, Clone)]
struct DirectoryEntry {
    name: String,
    is_directory: bool,
}

/// Virtual filesystem mapping PS5 paths to host directories.
pub struct VirtualFileSystem {
    /// Registered mount points.
    mounts: RwLock<Vec<MountPoint>>,
    /// Stable per-title save root. `/savedata0` is temporarily remapped below
    /// this root while a save slot is mounted, so save-slot enumeration must
    /// not derive its root from the current guest mount.
    savedata_root: RwLock<PathBuf>,
    /// Active save-slot mounts: index `N` holds the slot directory mounted at
    /// `/savedataN`. The real service hands out up to 16 concurrent mount
    /// points, and titles (Minecraft) hold several mounted at once from
    /// different threads — a single rebound `/savedata0` corrupts them all.
    savedata_mounts: RwLock<[Option<String>; 16]>,
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
            savedata_root: RwLock::new(PathBuf::from("savedata")),
            savedata_mounts: RwLock::new(std::array::from_fn(|_| None)),
            open_files: RwLock::new(HashMap::new()),
            next_fd: RwLock::new(3), // 0=stdin, 1=stdout, 2=stderr.
        };

        // Register standard PS5 mount points with default paths.
        vfs.mount("/app0/", "games/current");
        vfs.mount("/savedata0/", "savedata");
        vfs.mount("/download0/", "downloads/current");
        vfs.mount(
            "/temp0/",
            &std::env::temp_dir()
                .join("xps5x")
                .join("current")
                .to_string_lossy(),
        );
        vfs.mount("/system/", "firmware");
        vfs.mount("/dev/", ""); // Stubbed.
        vfs.mount("/proc/", ""); // Stubbed.

        vfs
    }

    /// Register or replace a mount point.
    ///
    /// Guest roots are stored without a trailing slash, so both `/temp0`
    /// and `/temp0/file` resolve through the same mount. Re-registering a
    /// root updates it in place instead of leaving an unreachable duplicate.
    pub fn mount(&self, ps5_prefix: &str, host_path: &str) {
        let prefix = normalize_mount_root(ps5_prefix);
        debug!("VFS mount: '{}' -> '{}'", prefix, host_path);
        let mut mounts = self.mounts.write();
        if let Some(existing) = mounts.iter_mut().find(|mount| mount.ps5_prefix == prefix) {
            existing.host_path = PathBuf::from(host_path);
        } else {
            mounts.push(MountPoint {
                ps5_prefix: prefix,
                host_path: PathBuf::from(host_path),
            });
            mounts.sort_by_key(|mount| std::cmp::Reverse(mount.ps5_prefix.len()));
        }
    }

    /// Set the game directory for /app0/.
    pub fn set_game_directory(&self, path: &std::path::Path) {
        let mut mounts = self.mounts.write();
        for mount in mounts.iter_mut() {
            if mount.ps5_prefix == "/app0" {
                mount.host_path = path.to_path_buf();
                info!("VFS: /app0/ -> {}", path.display());
                return;
            }
        }
    }

    /// Set the process-private writable directory exposed at `/temp0/`.
    pub fn set_temp_directory(&self, path: &std::path::Path) {
        let mut mounts = self.mounts.write();
        for mount in mounts.iter_mut() {
            if mount.ps5_prefix == "/temp0" {
                mount.host_path = path.to_path_buf();
                info!("VFS: /temp0/ -> {}", path.display());
                return;
            }
        }
    }

    /// Set the process-private downloaded/bootstrap-data directory exposed at
    /// `/download0/`.
    pub fn set_download_directory(&self, path: &std::path::Path) {
        self.set_mount_directory("/download0", path);
    }

    /// Set the process-private title save root and expose it at `/savedata0/`
    /// until a concrete save slot is mounted.
    pub fn set_savedata_directory(&self, path: &std::path::Path) {
        *self.savedata_root.write() = path.to_path_buf();
        self.set_mount_directory("/savedata0", path);
    }

    /// Return the stable per-title save root, independent of the currently
    /// mounted `/savedata0` slot.
    pub fn savedata_root(&self) -> PathBuf {
        self.savedata_root.read().clone()
    }

    /// Resolve a single save-slot name below the per-title root. Save-data
    /// directory names are opaque guest strings, but they may never escape
    /// the title root or introduce nested host paths.
    pub fn savedata_slot_path(&self, slot_name: &str) -> Result<PathBuf, std::io::Error> {
        let mut components = Path::new(slot_name).components();
        let valid = matches!(components.next(), Some(Component::Normal(_)))
            && components.next().is_none()
            && !slot_name.is_empty()
            && !slot_name.contains(['/', '\\'])
            && slot_name != "."
            && slot_name != "..";
        if !valid {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "save-data slot must be one path component",
            ));
        }
        Ok(self.savedata_root().join(slot_name))
    }

    /// Mount one validated save slot at the first free `/savedataN` point
    /// (N in 0..16, matching the real service) and return that prefix plus
    /// the host path. Mounting an already-mounted slot returns its existing
    /// point — the real API errors BUSY there, and idempotency is the safer
    /// HLE degradation. Directory creation and mount-mode policy remain the
    /// save-data service's responsibility.
    pub fn mount_savedata_slot(
        &self,
        slot_name: &str,
    ) -> Result<(String, PathBuf), std::io::Error> {
        let path = self.savedata_slot_path(slot_name)?;
        let mut slots = self.savedata_mounts.write();
        if let Some(index) = slots
            .iter()
            .position(|slot| slot.as_deref() == Some(slot_name))
        {
            return Ok((format!("/savedata{index}"), path));
        }
        let Some(index) = slots.iter().position(Option::is_none) else {
            return Err(std::io::Error::other(
                "all 16 save-data mount points are in use",
            ));
        };
        slots[index] = Some(slot_name.to_owned());
        drop(slots);
        let prefix = format!("/savedata{index}");
        self.set_mount_directory(&prefix, &path);
        Ok((prefix, path))
    }

    /// Unmount the save slot at `prefix` (`/savedataN`). Returns whether a
    /// slot was actually mounted there. `/savedata0` reverts to the
    /// title-level root (its boot-time mapping); higher points are removed.
    pub fn unmount_savedata_slot(&self, prefix: &str) -> bool {
        let prefix = normalize_mount_root(prefix);
        let Some(index) = prefix
            .strip_prefix("/savedata")
            .and_then(|digits| digits.parse::<usize>().ok())
        else {
            return false;
        };
        let mut slots = self.savedata_mounts.write();
        if index >= slots.len() || slots[index].is_none() {
            return false;
        }
        slots[index] = None;
        drop(slots);
        if index == 0 {
            let root = self.savedata_root();
            self.set_mount_directory("/savedata0", &root);
        } else {
            let mut mounts = self.mounts.write();
            mounts.retain(|mount| mount.ps5_prefix != prefix);
        }
        true
    }

    /// Active save-slot mount prefixes, for whole-service operations
    /// (commit) that flush every mounted container.
    pub fn savedata_mount_prefixes(&self) -> Vec<String> {
        self.savedata_mounts
            .read()
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_some())
            .map(|(index, _)| format!("/savedata{index}"))
            .collect()
    }

    fn set_mount_directory(&self, guest_root: &str, path: &std::path::Path) {
        let root = normalize_mount_root(guest_root);
        let mut mounts = self.mounts.write();
        if let Some(mount) = mounts.iter_mut().find(|mount| mount.ps5_prefix == root) {
            mount.host_path = path.to_path_buf();
            info!("VFS: {root}/ -> {}", path.display());
            return;
        }
        drop(mounts);
        self.mount(&root, &path.to_string_lossy());
    }

    /// Resolve a PS5 path to a host path.
    pub fn resolve_path(&self, ps5_path: &str) -> Option<PathBuf> {
        let mounts = self.mounts.read();
        for mount in mounts.iter() {
            let exact_root = ps5_path == mount.ps5_prefix;
            let under_root = ps5_path
                .strip_prefix(&mount.ps5_prefix)
                .is_some_and(|suffix| suffix.starts_with('/'));
            if exact_root || under_root {
                let relative = ps5_path[mount.ps5_prefix.len()..].trim_start_matches('/');
                if relative
                    .split(['/', '\\'])
                    .any(|component| component == "..")
                {
                    warn!("VFS resolve: refusing traversing guest path '{ps5_path}'");
                    return None;
                }
                return Some(mount.host_path.join(relative));
            }
        }
        None
    }

    /// Read host metadata for a mounted guest path. Unmounted paths are never
    /// interpreted as host paths.
    pub fn metadata(&self, ps5_path: &str) -> Result<std::fs::Metadata, std::io::Error> {
        let host_path = self.resolve_path(ps5_path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "path is not mounted")
        })?;
        std::fs::metadata(host_path)
    }

    /// Create a directory (and missing parents) beneath a mounted guest root.
    pub fn create_dir_all(&self, ps5_path: &str) -> Result<(), std::io::Error> {
        let host_path = self.resolve_path(ps5_path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "path is not mounted")
        })?;
        std::fs::create_dir_all(host_path)
    }

    /// Remove a directory tree beneath a mounted guest root. Missing paths
    /// are already in the desired state and succeed.
    pub fn remove_dir_all(&self, ps5_path: &str) -> Result<(), std::io::Error> {
        let host_path = self.resolve_path(ps5_path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "path is not mounted")
        })?;
        if host_path.exists() {
            std::fs::remove_dir_all(host_path)
        } else {
            Ok(())
        }
    }

    /// Remove a single file beneath a mounted guest root (`unlink`).
    pub fn remove_file(&self, ps5_path: &str) -> Result<(), std::io::Error> {
        let host_path = self.resolve_path(ps5_path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "path is not mounted")
        })?;
        std::fs::remove_file(host_path)
    }

    /// Remove an empty directory beneath a mounted guest root (`rmdir`).
    /// A non-empty directory fails, matching POSIX `rmdir` (use
    /// [`remove_dir_all`](Self::remove_dir_all) for recursive removal).
    pub fn remove_dir(&self, ps5_path: &str) -> Result<(), std::io::Error> {
        let host_path = self.resolve_path(ps5_path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "path is not mounted")
        })?;
        std::fs::remove_dir(host_path)
    }

    /// Rename `from` to `to`, both beneath mounted guest roots. Traversing or
    /// unmounted paths on either side fail rather than touching the host.
    pub fn rename(&self, from: &str, to: &str) -> Result<(), std::io::Error> {
        let host_from = self.resolve_path(from).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "source path is not mounted")
        })?;
        let host_to = self.resolve_path(to).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "destination path is not mounted",
            )
        })?;
        std::fs::rename(host_from, host_to)
    }

    /// Truncate (or zero-extend) the file at a mounted guest path to `len`
    /// bytes (`truncate`). The file must already exist.
    pub fn truncate(&self, ps5_path: &str, len: u64) -> Result<(), std::io::Error> {
        let host_path = self.resolve_path(ps5_path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "path is not mounted")
        })?;
        std::fs::OpenOptions::new()
            .write(true)
            .open(host_path)?
            .set_len(len)
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

        if path == "/" {
            if writable {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "virtual root opened for writing",
                ));
            }
            let mut entries: Vec<DirectoryEntry> = self
                .mounts
                .read()
                .iter()
                .filter_map(|mount| {
                    mount
                        .ps5_prefix
                        .strip_prefix('/')
                        .filter(|name| !name.is_empty())
                        .map(|name| DirectoryEntry {
                            name: name.to_string(),
                            is_directory: true,
                        })
                })
                .collect();
            entries.sort_by_key(|entry| entry.name.to_ascii_lowercase());
            entries.dedup_by(|a, b| a.name.eq_ignore_ascii_case(&b.name));
            let mut next = self.next_fd.write();
            let fd = *next;
            *next += 1;
            self.open_files.write().insert(
                fd,
                OpenFile {
                    fd,
                    host_path: PathBuf::new(),
                    ps5_path: path.to_string(),
                    position: 0,
                    data: None,
                    writable: false,
                    dirty: false,
                    flags,
                    directory_entries: Some(entries),
                    directory_index: 0,
                },
            );
            return Ok(fd);
        }

        let host_path = self.resolve_path(path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "path is not mounted")
        })?;

        let exists = host_path.exists();
        let is_directory = host_path.is_dir();
        if is_directory && writable {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "directory opened for writing",
            ));
        }
        let directory_entries = if is_directory {
            let mut entries = std::fs::read_dir(&host_path)?
                .filter_map(Result::ok)
                .filter_map(|entry| {
                    let name = entry.file_name().into_string().ok()?;
                    Some(DirectoryEntry {
                        is_directory: entry.file_type().ok()?.is_dir(),
                        name,
                    })
                })
                .collect::<Vec<_>>();
            entries.sort_by_key(|entry| entry.name.to_ascii_lowercase());
            // Real Orbis `getdents` yields `.` and `..` first; some
            // directory iterators depend on their presence (or explicitly
            // skip them and mis-handle a listing that omits them). Prepend
            // them so the guest sees a POSIX-shaped directory.
            let mut with_dots = Vec::with_capacity(entries.len() + 2);
            with_dots.push(DirectoryEntry {
                is_directory: true,
                name: ".".to_string(),
            });
            with_dots.push(DirectoryEntry {
                is_directory: true,
                name: "..".to_string(),
            });
            with_dots.extend(entries);
            Some(with_dots)
        } else {
            None
        };
        let data = if is_directory {
            None
        } else if exists && !truncate {
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
            flags,
            directory_entries,
            directory_index: 0,
        };

        debug!("VFS open: '{path}' -> fd={fd} (writable={writable}, create={create})");
        self.open_files.write().insert(fd, file);
        Ok(fd)
    }

    /// Read from an open file descriptor.
    pub fn read(&self, fd: Fd, count: usize) -> Result<Vec<u8>, std::io::Error> {
        let mut files = self.open_files.write();
        if let Some(file) = files.get_mut(&fd) {
            if file.directory_entries.is_some() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "fd is a directory",
                ));
            }
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

    /// Return the descriptor's Orbis open flags.
    pub fn flags(&self, fd: Fd) -> Option<i32> {
        self.open_files.read().get(&fd).map(|file| file.flags)
    }

    /// Update status flags while preserving the descriptor's access mode.
    pub fn set_status_flags(&self, fd: Fd, flags: i32) -> Result<(), std::io::Error> {
        use open_flags::O_ACCMODE;
        let mut files = self.open_files.write();
        let Some(file) = files.get_mut(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        file.flags = (file.flags & O_ACCMODE) | (flags & !O_ACCMODE);
        Ok(())
    }

    /// Return one fixed-size Gen5 dirent per call.
    ///
    /// PS5's kernel-facing directory ABI uses a 512-byte record even when the
    /// caller supplies a larger buffer. Advancing one entry at a time is
    /// important: retail code commonly treats every successful call as one
    /// complete record rather than walking packed BSD-style variable records.
    pub fn getdents(&self, fd: Fd, requested: usize) -> Result<(Vec<u8>, usize), std::io::Error> {
        const DIRENT_SIZE: usize = 512;
        if requested < DIRENT_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "directory buffer is smaller than one record",
            ));
        }
        let mut files = self.open_files.write();
        let Some(file) = files.get_mut(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        let Some(entries) = file.directory_entries.as_ref() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "fd is not a directory",
            ));
        };
        let base = file.directory_index;
        let Some(entry) = entries.get(file.directory_index) else {
            return Ok((Vec::new(), base));
        };
        let name = entry.name.as_bytes();
        let name_len = name.len().min(255);
        // Orbis `dirent`: d_fileno(u32), d_reclen(u16), d_type(u8),
        // d_namlen(u8), d_name[]. `d_fileno` must be NON-ZERO — a real
        // filesystem never hands out inode 0, and code that treats 0 as
        // "invalid entry" will mis-parse it. fnv1a of the name can be 0 for
        // some inputs, so force the low bit.
        let fileno = fnv1a32(&name[..name_len]) | 1;
        let mut record = vec![0u8; DIRENT_SIZE];
        record[0..4].copy_from_slice(&fileno.to_le_bytes());
        record[4..6].copy_from_slice(&(DIRENT_SIZE as u16).to_le_bytes());
        record[6] = if entry.is_directory { 4 } else { 8 };
        record[7] = name_len as u8;
        record[8..8 + name_len].copy_from_slice(&name[..name_len]);
        file.directory_index += 1;
        file.position = file.directory_index as u64;
        tracing::debug!(
            "getdents fd={fd}: entry[{base}] '{}' type={} -> 1 record",
            String::from_utf8_lossy(&name[..name_len]),
            if entry.is_directory { "dir" } else { "file" }
        );
        Ok((record, base))
    }

    /// Persist an open descriptor's dirty write-back buffer without closing
    /// it. Read-only descriptors succeed; unknown descriptors return
    /// `NotFound`. This backs the guest's `fsync` durability boundary.
    pub fn sync(&self, fd: Fd) -> Result<(), std::io::Error> {
        let mut files = self.open_files.write();
        let Some(file) = files.get_mut(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        flush_open_file(file)?;
        Ok(())
    }

    /// Persist every dirty writable descriptor below one guest mount.
    ///
    /// Save-data commit APIs operate on a mount point rather than individual
    /// file descriptors. Keeping this operation in the VFS makes that
    /// durability boundary real while ensuring a save commit cannot flush or
    /// otherwise affect descriptors opened below unrelated mounts.
    pub fn sync_mount(&self, guest_root: &str) -> Result<usize, std::io::Error> {
        let root = normalize_mount_root(guest_root);
        if self.resolve_path(&root).is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("mount {root} is not registered"),
            ));
        }

        let mut files = self.open_files.write();
        let mut flushed = 0usize;
        let mut first_error = None;
        for file in files.values_mut().filter(|file| {
            file.ps5_path == root
                || file
                    .ps5_path
                    .strip_prefix(&root)
                    .is_some_and(|suffix| suffix.starts_with('/'))
        }) {
            match flush_open_file(file) {
                Ok(true) => flushed += 1,
                Ok(false) => {}
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(flushed)
        }
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

/// Flush one open file's write-back buffer and report whether it was dirty.
fn flush_open_file(file: &mut OpenFile) -> Result<bool, std::io::Error> {
    if !file.dirty || !file.writable {
        return Ok(false);
    }
    if let Some(ref data) = file.data {
        std::fs::write(&file.host_path, data)?;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&file.host_path)?
            .sync_all()?;
        debug!(
            "VFS sync: flushed {} bytes -> {}",
            data.len(),
            file.host_path.display()
        );
    }
    file.dirty = false;
    Ok(true)
}

fn normalize_mount_root(prefix: &str) -> String {
    let normalized = prefix.replace('\\', "/");
    let root = normalized.trim_end_matches('/');
    if root.is_empty() {
        "/".to_string()
    } else {
        root.to_string()
    }
}

fn fnv1a32(bytes: &[u8]) -> u32 {
    bytes.iter().fold(2_166_136_261u32, |hash, byte| {
        (hash ^ u32::from(*byte)).wrapping_mul(16_777_619)
    })
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

    #[test]
    fn sync_persists_without_closing_and_rejects_unknown_fd() {
        use open_flags::*;
        let dir = temp_dir("sync");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        let fd = vfs
            .open("/app0/live.bin", O_WRONLY | O_CREAT | O_TRUNC, 0o644)
            .unwrap();
        vfs.write(fd, b"LIVE").unwrap();
        vfs.sync(fd).unwrap();
        assert_eq!(std::fs::read(dir.join("live.bin")).unwrap(), b"LIVE");
        assert_eq!(
            vfs.sync(0x7fff).unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        );
        vfs.close(fd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sync_mount_flushes_only_descriptors_below_that_guest_mount() {
        use open_flags::*;
        let save_dir = temp_dir("sync-mount-save");
        let game_dir = temp_dir("sync-mount-game");
        let vfs = VirtualFileSystem::new();
        vfs.set_savedata_directory(&save_dir);
        vfs.set_game_directory(&game_dir);

        let save_fd = vfs
            .open(
                "/savedata0/slot/live.bin",
                O_WRONLY | O_CREAT | O_TRUNC,
                0o644,
            )
            .unwrap();
        let game_fd = vfs
            .open(
                "/app0/should-stay-buffered.bin",
                O_WRONLY | O_CREAT | O_TRUNC,
                0o644,
            )
            .unwrap();
        vfs.write(save_fd, b"SAVE").unwrap();
        vfs.write(game_fd, b"GAME").unwrap();

        assert_eq!(vfs.sync_mount("/savedata0").unwrap(), 1);
        assert_eq!(
            std::fs::read(save_dir.join("slot/live.bin")).unwrap(),
            b"SAVE"
        );
        assert!(!game_dir.join("should-stay-buffered.bin").exists());

        vfs.close(save_fd).unwrap();
        vfs.close(game_fd).unwrap();
        let _ = std::fs::remove_dir_all(&save_dir);
        let _ = std::fs::remove_dir_all(&game_dir);
    }

    #[test]
    fn directory_open_getdents_and_fcntl_flags_are_stateful() {
        use open_flags::*;
        let dir = temp_dir("getdents");
        std::fs::write(dir.join("stone.txt"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("packs")).unwrap();
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        let fd = vfs.open("/app0", O_RDONLY, 0).unwrap();
        assert_eq!(vfs.flags(fd), Some(O_RDONLY));
        // Four entries now: the two synthetic dot-dirs Orbis always yields
        // first (`.`, `..`), then the real `packs` and `stone.txt`.
        let mut kinds_and_names = Vec::new();
        for expected_base in 0..4 {
            let (bytes, base) = vfs.getdents(fd, 1024).unwrap();
            assert_eq!(bytes.len(), 512);
            assert_eq!(base, expected_base);
            assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 512);
            // Every d_fileno must be non-zero.
            assert_ne!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0);
            let name_len = bytes[7] as usize;
            kinds_and_names.push((
                bytes[6],
                std::str::from_utf8(&bytes[8..8 + name_len])
                    .unwrap()
                    .to_string(),
            ));
        }
        kinds_and_names.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(
            kinds_and_names,
            [
                (4, ".".to_string()),
                (4, "..".to_string()),
                (4, "packs".to_string()),
                (8, "stone.txt".to_string()),
            ]
        );
        let (eof, base) = vfs.getdents(fd, 512).unwrap();
        assert!(eof.is_empty());
        assert_eq!(base, 4);
        vfs.set_status_flags(fd, O_APPEND).unwrap();
        assert_eq!(vfs.flags(fd), Some(O_APPEND | O_RDONLY));
        vfs.close(fd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn virtual_root_enumerates_guest_mounts_without_exposing_host_root() {
        let vfs = VirtualFileSystem::new();
        let fd = vfs.open("/", open_flags::O_RDONLY, 0).unwrap();
        let mut bytes = Vec::new();
        loop {
            let (record, _) = vfs.getdents(fd, 1024).unwrap();
            if record.is_empty() {
                break;
            }
            bytes.extend_from_slice(&record);
        }
        let names = ["app0", "savedata0", "download0", "temp0", "system"];
        for name in names {
            assert!(
                bytes
                    .windows(name.len())
                    .any(|window| window == name.as_bytes()),
                "virtual root must contain {name}"
            );
        }
        assert!(!bytes.windows(7).any(|window| window == b"Windows"));
    }

    #[test]
    fn mount_root_and_children_resolve_without_prefix_collisions() {
        let root = temp_dir("mount-root");
        let nested = temp_dir("mount-nested");
        let vfs = VirtualFileSystem::new();
        vfs.mount("/temp0/", &root.to_string_lossy());
        vfs.mount("/temp0/cache/", &nested.to_string_lossy());

        assert_eq!(vfs.resolve_path("/temp0"), Some(root.clone()));
        assert_eq!(
            vfs.resolve_path("/temp0/file.bin"),
            Some(root.join("file.bin"))
        );
        assert_eq!(
            vfs.resolve_path("/temp0/cache/index.bin"),
            Some(nested.join("index.bin"))
        );
        assert_eq!(vfs.resolve_path("/temp01/file.bin"), None);
        assert_eq!(vfs.resolve_path("/temp0/../escape.bin"), None);

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(nested);
    }

    #[test]
    fn per_title_download_and_savedata_mounts_are_replaceable() {
        let download = temp_dir("download");
        let savedata = temp_dir("savedata");
        let vfs = VirtualFileSystem::new();
        vfs.set_download_directory(&download);
        vfs.set_savedata_directory(&savedata);

        assert_eq!(vfs.resolve_path("/download0"), Some(download.clone()));
        assert_eq!(
            vfs.resolve_path("/download0/bootstrap.json"),
            Some(download.join("bootstrap.json"))
        );
        assert_eq!(vfs.resolve_path("/savedata0"), Some(savedata.clone()));

        let _ = std::fs::remove_dir_all(download);
        let _ = std::fs::remove_dir_all(savedata);
    }

    #[test]
    fn savedata_slot_mount_keeps_a_stable_title_root() {
        let root = temp_dir("savedata-slots");
        let vfs = VirtualFileSystem::new();
        vfs.set_savedata_directory(&root);

        let (prefix, slot_path) = vfs.mount_savedata_slot("slot00000001@world").unwrap();
        assert_eq!(prefix, "/savedata0");
        assert_eq!(slot_path, root.join("slot00000001@world"));
        assert_eq!(vfs.savedata_root(), root);
        assert_eq!(
            vfs.resolve_path("/savedata0/level.dat"),
            Some(root.join("slot00000001@world/level.dat"))
        );
        assert!(vfs.mount_savedata_slot("../escape").is_err());
        assert!(vfs.mount_savedata_slot("nested/escape").is_err());

        assert!(vfs.unmount_savedata_slot("/savedata0"));
        assert_eq!(vfs.resolve_path("/savedata0"), Some(root.clone()));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn concurrent_savedata_slots_get_distinct_mount_points() {
        let root = temp_dir("savedata-multi");
        let vfs = VirtualFileSystem::new();
        vfs.set_savedata_directory(&root);

        let (settings, _) = vfs
            .mount_savedata_slot("BedrockUserSettingsStorage")
            .unwrap();
        let (cache, _) = vfs.mount_savedata_slot("BedrockLevelInfoCache").unwrap();
        assert_eq!(settings, "/savedata0");
        assert_eq!(cache, "/savedata1");
        // Each point resolves into its own container — the second mount must
        // not have rebound the first.
        assert_eq!(
            vfs.resolve_path("/savedata0/options.txt"),
            Some(root.join("BedrockUserSettingsStorage/options.txt"))
        );
        assert_eq!(
            vfs.resolve_path("/savedata1/cache.bin"),
            Some(root.join("BedrockLevelInfoCache/cache.bin"))
        );
        // Re-mounting a mounted slot is idempotent, not a fresh point.
        let (again, _) = vfs.mount_savedata_slot("BedrockLevelInfoCache").unwrap();
        assert_eq!(again, "/savedata1");
        assert_eq!(
            vfs.savedata_mount_prefixes(),
            vec!["/savedata0".to_owned(), "/savedata1".to_owned()]
        );

        // Unmounting one point leaves the other intact.
        assert!(vfs.unmount_savedata_slot("/savedata1"));
        assert!(vfs.resolve_path("/savedata1/cache.bin").is_none());
        assert_eq!(
            vfs.resolve_path("/savedata0/options.txt"),
            Some(root.join("BedrockUserSettingsStorage/options.txt"))
        );
        assert!(!vfs.unmount_savedata_slot("/savedata1"));
        let _ = std::fs::remove_dir_all(root);
    }
}
