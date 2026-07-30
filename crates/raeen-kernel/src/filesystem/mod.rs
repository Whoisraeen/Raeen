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
//!
//! One guest path family is deliberately *not* a host path: the `memory:` URI
//! scheme, served straight out of guest memory by [`memory_scheme`]. It is
//! routed off before host-path resolution ever sees it.

pub mod memory_scheme;

use parking_lot::{Mutex, RwLock};
use raeen_core::types::Fd;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Weak};
use tracing::{debug, info, warn};

pub use memory_scheme::{GuestByteSource, MemoryUri, MemoryUriError};

/// Why the `memory:` scheme could not reach a guest address space.
///
/// Two variants, not one, because they have opposite fixes and a single
/// "no address space bound" message cannot tell them apart. Measured: GTA V's
/// first run on this scheme reported that message four times, and the cause was
/// [`NeverBound`](Self::NeverBound) — the `--run-eboot` entry point constructed
/// its arena inline and never called
/// [`set_guest_byte_source`](VirtualFileSystem::set_guest_byte_source) — not
/// [`Dropped`](Self::Dropped), which is what a reader would assume from a
/// nine-seconds-into-the-run failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressSpaceMiss {
    /// No runtime entry point ever bound an address space. A missing call site.
    NeverBound,
    /// An address space was bound, but the arena behind the [`Weak`] is gone. A
    /// lifetime problem.
    Dropped,
}

impl AddressSpaceMiss {
    /// Stable blocker key suffix, so the two cases aggregate separately.
    #[must_use]
    pub fn blocker_reason(self) -> &'static str {
        match self {
            Self::NeverBound => "no-address-space",
            Self::Dropped => "address-space-dropped",
        }
    }

    /// Human-readable cause, used verbatim in both the blocker detail and the
    /// `io::Error` the guest's `open`/`read` maps to `EIO`.
    #[must_use]
    pub fn explanation(self) -> &'static str {
        match self {
            Self::NeverBound => {
                "no guest address space was ever bound to the VFS (a runtime entry point is \
                 missing its bind call)"
            }
            Self::Dropped => {
                "the bound guest address space has been dropped (its arena outlived by the VFS)"
            }
        }
    }
}

/// A mount point mapping PS5 paths to host paths.
#[derive(Debug, Clone)]
struct MountPoint {
    /// PS5 path prefix (e.g., "/app0/").
    ps5_prefix: String,
    /// Host directory this maps to.
    host_path: PathBuf,
    /// Canonical mount identity captured when the mapping is installed.
    ///
    /// Resolving every asset used to canonicalize this unchanged root again.
    /// Minecraft probes thousands of optional resource-pack paths while
    /// holding a title streaming mutex, so the redundant host syscall was a
    /// measurable load-time multiplier. The candidate/deepest-existing path
    /// is still canonicalized on every access and compared with this identity,
    /// preserving the fail-closed reparse-point boundary.
    canonical_root: Option<PathBuf>,
}

impl MountPoint {
    fn new(ps5_prefix: String, host_path: PathBuf) -> Self {
        let canonical_root = canonicalize_mount_root(&host_path);
        Self {
            ps5_prefix,
            host_path,
            canonical_root,
        }
    }

    fn set_host_path(&mut self, host_path: PathBuf) {
        self.canonical_root = canonicalize_mount_root(&host_path);
        self.host_path = host_path;
    }
}

fn canonicalize_mount_root(host_path: &Path) -> Option<PathBuf> {
    if host_path.as_os_str().is_empty() {
        None
    } else {
        std::fs::canonicalize(host_path).ok()
    }
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
///
/// Fields split by mutability: the immutable-after-open ones live directly on
/// the struct; the per-fd state mutated during I/O lives in [`OpenFileMut`]
/// behind a per-file `Mutex`. That split is what lets `read`/`pread`/`write`/
/// `seek` take only the *shared* `open_files` READ lock and then lock this one
/// file — two different fds hold two different mutexes, so their I/O runs
/// genuinely concurrently instead of serializing on a single map-wide WRITE
/// lock. Same fd → same mutex → serialized, preserving today's per-descriptor
/// cursor semantics. Measured motivation: a WRITE lock on the whole open-files
/// map taken on *every* read compounded over the asset streamer's thousands of
/// small reads into multi-second loads (ASTRO.BOT
/// "loading time: data/prein/haptics : 180.169").
#[derive(Debug)]
#[allow(dead_code)] // fd/ps5_path recorded for fstat, re-open, and debug tooling
struct OpenFile {
    /// File descriptor number.
    fd: Fd,
    /// Host file path.
    host_path: PathBuf,
    /// PS5 path (for debugging).
    ps5_path: String,
    /// True for `/dev/random` and `/dev/urandom`. Reads are supplied directly
    /// by the host OS entropy source rather than a host file.
    random_device: bool,
    /// Present for a `memory:` pseudo-file (see [`memory_scheme`]). Reads are
    /// served out of the guest address range this URI declared, re-validated on
    /// every access; there is no host file and no `data` buffer. Mutually
    /// exclusive with `reader`, `dirents`, and `random_device`, and always
    /// paired with `writable == false`.
    memory_region: Option<MemoryUri>,
    /// Lazy read-only backing: the host file handle and its length. Present
    /// instead of `data` for a read-only fd of an existing file, so a large
    /// file streams on demand rather than being slurped whole into memory at
    /// `open`. The eager `std::fs::read` OOMs under host commit pressure —
    /// measured: ASTRO.BOT's 6.7 MiB `game_text.xml` failed to open, returned a
    /// null fd, and the game null-dereferenced it (logs/raeen.txt).
    ///
    /// Read as `&File` (shared), never `&mut`: reads are now positional
    /// (`seek_read`/`read_at`), which move no cursor, so many reads can share
    /// one handle concurrently under the map read lock without a per-call clone.
    reader: Option<(std::fs::File, u64)>,
    /// Whether the fd was opened for writing (`O_WRONLY`/`O_RDWR`).
    writable: bool,
    /// Packed Orbis dirent blocks for directory descriptors (see
    /// [`pack_dirents`]), snapshot at open. Immutable after open; the walk
    /// cursor is the byte offset in [`OpenFileMut::position`], exactly like a
    /// regular file — which is also what `lseek` on a directory fd moves.
    dirents: Option<Vec<u8>>,
    /// Per-fd mutable I/O state. Its own `Mutex` so concurrent I/O on
    /// *different* fds never contends (see the struct doc).
    inner: Mutex<OpenFileMut>,
}

/// The mutable-during-I/O state of one open descriptor, guarded by
/// [`OpenFile::inner`]. Deliberately small: the per-file lock is held only for
/// cursor bookkeeping, the write-back buffer, and `fcntl`/`getdents` cursors.
#[derive(Debug)]
struct OpenFileMut {
    /// Current file position.
    position: u64,
    /// File data (cached in memory; the write-back buffer for a writable fd).
    data: Option<Vec<u8>>,
    /// Whether `data` has unflushed writes to persist to `host_path` on close.
    dirty: bool,
    /// Original Orbis open flags (queried/updated through `fcntl`).
    flags: i32,
}

#[derive(Debug, Clone)]
struct DirectoryEntry {
    name: String,
    is_directory: bool,
}

/// One directory block: real Orbis `getdents` (FreeBSD `DIRBLKSIZ`) returns
/// whole 512-byte blocks of packed variable-length records.
const DIRENT_BLOCK: usize = 512;
/// Fixed dirent header: `d_fileno`(u32) `d_reclen`(u16) `d_type`(u8)
/// `d_namlen`(u8), followed by the NUL-terminated `d_name`.
const DIRENT_HEADER: usize = 8;

/// Pack directory entries into Orbis dirent blocks, mirroring shadPS4's
/// `NormalDirectory::RebuildDirents` (itself mirroring FreeBSD directory
/// blocks): each record is `d_reclen = align4(8 + namlen + 1)` bytes, records
/// never cross a 512-byte block boundary, and the last record of each block
/// absorbs the block's slack into its `d_reclen` so records tile every block
/// exactly. The total is always a multiple of 512 — this is also the
/// `st_size` a directory fd reports.
///
/// The previous model (one 512-byte record per call with `d_reclen == 512`)
/// overflowed real guests: `sizeof(dirent)` is 264, and a parser that copies
/// a record by its `d_reclen` into a stack-allocated `dirent` writes 248
/// bytes past it — measured as Until Dawn's deterministic
/// `__stack_chk_fail` after listing an empty `/app0/deepfiles`.
fn pack_dirents(entries: &[DirectoryEntry]) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut last_reclen_at: Option<usize> = None;
    for entry in entries {
        let name = entry.name.as_bytes();
        let namlen = name.len().min(255);
        let reclen = (DIRENT_HEADER + namlen + 1).next_multiple_of(4);
        let mut offset = bin.len();
        let block_end = (offset / DIRENT_BLOCK + 1) * DIRENT_BLOCK;
        if offset + reclen > block_end {
            // Would cross a block boundary: extend the previous record over
            // the slack and start this one at the next block.
            if let Some(at) = last_reclen_at {
                let old = u16::from_le_bytes([bin[at], bin[at + 1]]);
                let padded = old + (block_end - offset) as u16;
                bin[at..at + 2].copy_from_slice(&padded.to_le_bytes());
            }
            bin.resize(block_end, 0);
            offset = block_end;
        }
        bin.resize(offset + reclen, 0);
        // `d_fileno` must be NON-ZERO — a real filesystem never hands out
        // inode 0, and code that treats 0 as "invalid entry" will mis-parse
        // it. fnv1a of the name can be 0 for some inputs, so force a bit.
        let fileno = fnv1a32(&name[..namlen]) | 1;
        bin[offset..offset + 4].copy_from_slice(&fileno.to_le_bytes());
        bin[offset + 4..offset + 6].copy_from_slice(&(reclen as u16).to_le_bytes());
        bin[offset + 6] = if entry.is_directory { 4 } else { 8 }; // DT_DIR / DT_REG
        bin[offset + 7] = namlen as u8;
        bin[offset + 8..offset + 8 + namlen].copy_from_slice(&name[..namlen]);
        // The NUL terminator and align-to-4 padding are already zero.
        last_reclen_at = Some(offset + 4);
    }
    // Round the final block up and let its last record absorb the slack.
    if let Some(at) = last_reclen_at {
        let ceiling = bin.len().next_multiple_of(DIRENT_BLOCK);
        let slack = (ceiling - bin.len()) as u16;
        let old = u16::from_le_bytes([bin[at], bin[at + 1]]);
        bin[at..at + 2].copy_from_slice(&(old + slack).to_le_bytes());
        bin.resize(ceiling, 0);
    }
    bin
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
    /// Guest-memory reader for the `memory:` pseudo-file scheme, installed by
    /// the runtime once the guest address space exists (see
    /// [`set_guest_byte_source`](Self::set_guest_byte_source)).
    ///
    /// Held **weakly** on purpose: the source is the process's guest arena, and
    /// a strong reference here would keep its reservations alive past the end of
    /// the run. A dead (or never-installed) source makes a `memory:` open a
    /// named, counted refusal rather than a silent empty file.
    guest_bytes: RwLock<Option<Weak<dyn GuestByteSource>>>,
    /// How many times [`set_guest_byte_source`](VirtualFileSystem::set_guest_byte_source)
    /// has run against this instance — see
    /// [`guest_byte_source_binding_count`](VirtualFileSystem::guest_byte_source_binding_count).
    guest_byte_source_bindings: std::sync::atomic::AtomicU64,
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
            guest_bytes: RwLock::new(None),
            guest_byte_source_bindings: std::sync::atomic::AtomicU64::new(0),
        };

        // Register standard PS5 mount points with default paths.
        vfs.mount("/app0/", "games/current");
        vfs.mount("/savedata0/", "savedata");
        vfs.mount("/download0/", "downloads/current");
        vfs.mount(
            "/temp0/",
            &std::env::temp_dir()
                .join("raeen")
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
            existing.set_host_path(PathBuf::from(host_path));
        } else {
            mounts.push(MountPoint::new(prefix, PathBuf::from(host_path)));
            mounts.sort_by_key(|mount| std::cmp::Reverse(mount.ps5_prefix.len()));
        }
    }

    /// Set the game directory for /app0/.
    pub fn set_game_directory(&self, path: &std::path::Path) {
        let mut mounts = self.mounts.write();
        for mount in mounts.iter_mut() {
            if mount.ps5_prefix == "/app0" {
                mount.set_host_path(path.to_path_buf());
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
                mount.set_host_path(path.to_path_buf());
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
            mount.set_host_path(path.to_path_buf());
            info!("VFS: {root}/ -> {}", path.display());
            return;
        }
        drop(mounts);
        self.mount(&root, &path.to_string_lossy());
    }

    /// Install the guest-memory reader that backs the `memory:` pseudo-file
    /// scheme, replacing any previous one.
    ///
    /// Called by the runtime once the guest address space exists. Only a
    /// [`Weak`] is retained, so this never extends the arena's lifetime; when the
    /// process ends and the arena drops, `memory:` opens revert to a named
    /// refusal on their own with no explicit teardown.
    pub fn set_guest_byte_source(&self, source: &Arc<dyn GuestByteSource>) {
        *self.guest_bytes.write() = Some(Arc::downgrade(source));
        self.guest_byte_source_bindings
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        debug!("VFS: memory: scheme bound to a guest address space");
    }

    /// How many times an address space has been bound to **this** VFS.
    ///
    /// Per-instance, not process-global, on purpose: a global counter is racy
    /// under the parallel test harness (measured — it read 46 where 35 was
    /// expected because sibling tests bound their own arenas concurrently), and a
    /// flaky assertion about a wiring bug is worse than none. Each `OrbisKernel`
    /// owns one VFS, so a test that builds its own kernel observes only its own
    /// bindings.
    ///
    /// This exists because the failure it detects is not a wrong value but an
    /// *unexecuted* one: `execute_process_shared` — the entry point every retail
    /// title uses — never called
    /// [`set_guest_byte_source`](Self::set_guest_byte_source) at all.
    #[must_use]
    pub fn guest_byte_source_binding_count(&self) -> u64 {
        self.guest_byte_source_bindings
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Drop the `memory:` guest-memory reader.
    pub fn clear_guest_byte_source(&self) {
        *self.guest_bytes.write() = None;
    }

    /// The live guest-memory reader.
    ///
    /// Upgraded to a strong `Arc` for the duration of one operation so the arena
    /// cannot drop mid-read, and the `guest_bytes` lock is released before any
    /// guest memory is touched (the source never re-enters the VFS, but holding
    /// two locks across a foreign call is not worth the risk).
    ///
    /// The two failures are reported **separately** on purpose. A single
    /// "no address space" refusal cannot distinguish a runtime entry point that
    /// never performed the binding from a binding whose arena has since dropped,
    /// and those have opposite fixes — wire the missing call site, versus keep a
    /// strong reference alive. That exact ambiguity sent this scheme's first
    /// measurement down the wrong path (`--run-eboot` had simply never bound),
    /// and it is the same class of ambiguity that cost four rounds on
    /// Blasphemous II's `scePthreadAttrGet`.
    fn guest_byte_source(&self) -> Result<Arc<dyn GuestByteSource>, AddressSpaceMiss> {
        match self.guest_bytes.read().as_ref() {
            None => Err(AddressSpaceMiss::NeverBound),
            Some(weak) => weak.upgrade().ok_or(AddressSpaceMiss::Dropped),
        }
    }

    /// Parse and validate a `memory:` URI, without opening a descriptor.
    ///
    /// This is what `stat` needs: the declared length, proven to name a mapped
    /// guest range, with no fd allocated. Every refusal is named and counted.
    pub fn memory_file_len(&self, path: &str) -> Result<u64, std::io::Error> {
        Ok(self.parse_and_validate_memory_uri(path)?.len)
    }

    /// The shared `memory:` gate: parse strictly, then prove the whole declared
    /// range is mapped and readable *before* anything is served.
    ///
    /// Refusals are `NotFound` (the guest's ENOENT) except a missing address
    /// space, which is a host-side wiring failure and reported as such.
    fn parse_and_validate_memory_uri(&self, path: &str) -> Result<MemoryUri, std::io::Error> {
        let uri = memory_scheme::parse(path).map_err(|error| {
            memory_scheme::refuse(error.name(), 0, || {
                format!("malformed memory: URI '{path}' ({error})")
            });
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("malformed memory: URI ({error})"),
            )
        })?;
        let source = self.guest_byte_source().map_err(|miss| {
            memory_scheme::refuse(miss.blocker_reason(), uri.addr, || {
                format!(
                    "memory: URI '{}' requested but {}",
                    uri.display_name,
                    miss.explanation()
                )
            });
            std::io::Error::other(miss.explanation())
        })?;
        // The one place a guest-supplied pointer stops being an integer. Prove
        // the WHOLE declared range is mapped and readable before a single byte
        // is served; reads re-check as well, so this is a fail-fast at open
        // rather than the only guard.
        if !source.guest_range_readable(uri.addr, uri.len) {
            memory_scheme::refuse("unmapped-range", uri.addr, || {
                format!(
                    "memory: URI '{}' declares [{:#x}, {:#x}) which is not fully mapped readable \
                     guest memory",
                    uri.display_name,
                    uri.addr,
                    uri.end().unwrap_or(u64::MAX)
                )
            });
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "memory: range is not mapped readable guest memory",
            ));
        }
        Ok(uri)
    }

    /// Serve one read of a `memory:` descriptor at absolute `offset`.
    ///
    /// Clamps to the declared length (a read past EOF is a short/empty read, as
    /// POSIX requires) and then copies through [`GuestByteSource`], which
    /// re-validates: the guest may have unmapped the buffer since `open`, and a
    /// stale range must fail, never serve host bytes.
    fn read_memory_region(
        &self,
        region: &MemoryUri,
        out: &mut [u8],
        offset: u64,
    ) -> Result<usize, std::io::Error> {
        let start = offset.min(region.len);
        let want = usize::try_from((region.len - start).min(out.len() as u64)).unwrap_or(0);
        if want == 0 {
            return Ok(0);
        }
        let source = self.guest_byte_source().map_err(|miss| {
            memory_scheme::refuse(miss.blocker_reason(), region.addr, || {
                format!(
                    "read of memory: file '{}' but {}",
                    region.display_name,
                    miss.explanation()
                )
            });
            std::io::Error::other(miss.explanation())
        })?;
        // `start <= region.len` and `addr + len` was proven non-overflowing at
        // parse time, so this cannot wrap.
        let at = region.addr + start;
        if !source.read_guest_bytes(at, &mut out[..want]) {
            memory_scheme::refuse("unmapped-read", at, || {
                format!(
                    "read of {want} bytes at {at:#x} in memory: file '{}' hit unmapped guest memory",
                    region.display_name
                )
            });
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "memory: read hit unmapped guest memory",
            ));
        }
        Ok(want)
    }

    /// Resolve a PS5 path to a host path.
    ///
    /// This is the guest->host sandbox boundary: every file syscall maps a guest
    /// path here and hands the result straight to the host filesystem. A path
    /// that matches no mount, or that would escape its mount root, resolves to
    /// `None` (default-deny / fail-closed) — see [`combine_within_mount`].
    pub fn resolve_path(&self, ps5_path: &str) -> Option<PathBuf> {
        // A `memory:` URI is not a host path and must never be walked as one.
        // Refusing it here (rather than letting it reach `combine_within_mount`
        // and trip the drive-qualifier guard) keeps the log honest: the guard's
        // warning describes a sandbox escape attempt, which this is not. The
        // guard itself is untouched — an ORDINARY path containing `:` still
        // reaches it and is still refused.
        if memory_scheme::claims(ps5_path) {
            debug!("VFS resolve: '{ps5_path}' is a memory: pseudo-file, not a host path");
            return None;
        }
        // Mount matching is a literal prefix compare, so it must run on a
        // normalized spelling — see [`normalize_guest_path`].
        let normalized = normalize_guest_path(ps5_path);
        let mounts = self.mounts.read();
        for mount in mounts.iter() {
            let exact_root = normalized == mount.ps5_prefix;
            let under_root = normalized
                .strip_prefix(&mount.ps5_prefix)
                .is_some_and(|suffix| suffix.starts_with('/'));
            if exact_root || under_root {
                let relative = normalized[mount.ps5_prefix.len()..].trim_start_matches('/');
                return combine_within_mount(
                    &mount.host_path,
                    mount.canonical_root.as_deref(),
                    ps5_path,
                    relative,
                );
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
    /// Write confinement is enforced by [`combine_within_mount`], which
    /// normalizes `..` with a clamp at the mount root and then proves lexical
    /// and canonical containment. There is deliberately no extra "reject any
    /// path containing `..`" guard here: that rule denied the
    /// `../../../`-prefixed paths every Unreal Engine title uses for its own
    /// content and save trees, while adding no confinement the resolver does not
    /// already guarantee. An unresolvable path still fails closed below.
    pub fn open(&self, path: &str, flags: i32, _mode: u32) -> Result<Fd, std::io::Error> {
        use open_flags::*;

        let writable = matches!(flags & O_ACCMODE, O_WRONLY | O_RDWR);
        let create = flags & O_CREAT != 0;
        let truncate = flags & O_TRUNC != 0;
        let append = flags & O_APPEND != 0;

        // --- `memory:` pseudo-file, BEFORE any host-path resolution. ---
        //
        // Ordering mirrors shadPS4's `/dev/` branch (`file_system.cpp:119`):
        // after the access-mode decode, before `GetHostPath`. Once the scheme
        // claims the path the decision is authoritative — a malformed URI or an
        // unmapped range is a named refusal, never a fall-through that would
        // turn a bad URI into a host filesystem probe.
        if memory_scheme::claims(path) {
            if writable || create || truncate {
                // The guest already owns this buffer and can store to it
                // directly; a writable file handle would be a second aliasing
                // path into it. EACCES, matching `/dev/random`'s refusal and the
                // Orbis read-only-object convention.
                memory_scheme::refuse("write-open", 0, || {
                    format!("memory: URI '{path}' opened for writing (flags {flags:#x})")
                });
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "memory: pseudo-files are read-only",
                ));
            }
            let region = self.parse_and_validate_memory_uri(path)?;
            let mut next = self.next_fd.write();
            let fd = *next;
            *next += 1;
            debug!(
                "VFS open: '{}' -> fd={fd} ({} bytes of guest memory at {:#x}, flags {})",
                region.display_name, region.len, region.addr, region.flags
            );
            self.open_files.write().insert(
                fd,
                OpenFile {
                    fd,
                    host_path: PathBuf::new(),
                    ps5_path: path.to_string(),
                    random_device: false,
                    memory_region: Some(region),
                    reader: None,
                    writable: false,
                    dirents: None,
                    inner: Mutex::new(OpenFileMut {
                        position: 0,
                        data: None,
                        dirty: false,
                        flags,
                    }),
                },
            );
            return Ok(fd);
        }

        if matches!(path, "/dev/random" | "/dev/urandom") {
            if writable {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "random device opened for writing",
                ));
            }
            let mut next = self.next_fd.write();
            let fd = *next;
            *next += 1;
            self.open_files.write().insert(
                fd,
                OpenFile {
                    fd,
                    host_path: PathBuf::new(),
                    ps5_path: path.to_string(),
                    random_device: true,
                    memory_region: None,
                    reader: None,
                    writable: false,
                    dirents: None,
                    inner: Mutex::new(OpenFileMut {
                        position: 0,
                        data: None,
                        dirty: false,
                        flags,
                    }),
                },
            );
            debug!("VFS open: '{path}' -> fd={fd} (host entropy device)");
            return Ok(fd);
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
            // Same dot-entry prefix a real directory listing carries.
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
            let entries = with_dots;
            let mut next = self.next_fd.write();
            let fd = *next;
            *next += 1;
            self.open_files.write().insert(
                fd,
                OpenFile {
                    fd,
                    host_path: PathBuf::new(),
                    ps5_path: path.to_string(),
                    random_device: false,
                    memory_region: None,
                    reader: None,
                    writable: false,
                    dirents: Some(pack_dirents(&entries)),
                    inner: Mutex::new(OpenFileMut {
                        position: 0,
                        data: None,
                        dirty: false,
                        flags,
                    }),
                },
            );
            return Ok(fd);
        }

        let host_path = self.resolve_path(path).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "path is not mounted")
        })?;

        let exists = host_path.exists();
        let is_directory = host_path.is_dir();
        if !exists && !create {
            debug!("VFS: file not found on host: {}", host_path.display());
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{} does not exist", host_path.display()),
            ));
        }
        if is_directory && writable {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "directory opened for writing",
            ));
        }
        let dirents = if is_directory {
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
            Some(pack_dirents(&with_dots))
        } else {
            None
        };
        // Read-only opens of an existing file stream lazily from the host file
        // instead of buffering the whole thing at open time. Slurping the file
        // into a `Vec` up front allocates its full size in one shot, which fails
        // with `OutOfMemory` under host commit pressure (the paging file being
        // too small) — the open then returns an error a title reads as "the file
        // does not exist", and titles that skip the null check crash. Writable
        // opens still load the content: the write-back model rewrites the whole
        // file on close, so it needs the buffer.
        let mut reader = None;
        let data = if is_directory {
            None
        } else if exists && !truncate && !writable {
            let handle = std::fs::File::open(&host_path)?;
            let len = handle.metadata().map(|m| m.len()).unwrap_or(0);
            reader = Some((handle, len));
            None
        } else if exists && !truncate {
            Some(std::fs::read(&host_path)?)
        } else {
            // Truncated existing file, or a new file being created: start empty.
            Some(Vec::new())
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
            random_device: false,
            memory_region: None,
            reader,
            writable,
            dirents,
            inner: Mutex::new(OpenFileMut {
                position,
                data,
                dirty,
                flags,
            }),
        };

        debug!("VFS open: '{path}' -> fd={fd} (writable={writable}, create={create})");
        self.open_files.write().insert(fd, file);
        Ok(fd)
    }

    /// Read from an open file descriptor, advancing its cursor.
    ///
    /// Takes only the SHARED map read lock plus this one fd's mutex, so reads on
    /// different fds run concurrently (the old whole-map WRITE lock serialized
    /// every read across all fds and threads). The per-file lock is held across
    /// the positional read because `position` must advance coherently — same-fd
    /// reads therefore chunk sequentially, exactly as before.
    pub fn read(&self, fd: Fd, count: usize) -> Result<Vec<u8>, std::io::Error> {
        let mut bytes = vec![0u8; count];
        let read = self.read_into(fd, &mut bytes)?;
        bytes.truncate(read);
        Ok(bytes)
    }

    /// Allocation-free sequential read into caller-owned storage.
    pub fn read_into(&self, fd: Fd, out: &mut [u8]) -> Result<usize, std::io::Error> {
        let files = self.open_files.read();
        let Some(file) = files.get(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        if file.dirents.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "fd is a directory",
            ));
        }
        if file.random_device {
            getrandom::fill(out)
                .map_err(|error| std::io::Error::other(format!("host entropy failed: {error}")))?;
            return Ok(out.len());
        }
        if let Some(region) = file.memory_region.as_ref() {
            // Hold the per-fd lock across the guest copy so `position` advances
            // coherently, exactly as the `data`/`reader` branches below do.
            let mut inner = file.inner.lock();
            let pos = inner.position;
            let read = self.read_memory_region(region, out, pos)?;
            inner.position = pos + read as u64;
            return Ok(read);
        }
        let mut inner = file.inner.lock();
        let pos = inner.position;
        if let Some(data) = inner.data.as_ref() {
            let start = (pos as usize).min(data.len());
            let end = start.saturating_add(out.len()).min(data.len());
            let read = end - start;
            out[..read].copy_from_slice(&data[start..end]);
            // Advance by the bytes actually read (mirrors the reader branch
            // below). Setting `position = end` rewound the cursor to EOF when it
            // started past EOF, corrupting a following write's offset on an
            // O_RDWR fd; POSIX leaves the offset unchanged on a 0-byte EOF read.
            inner.position = pos + read as u64;
            Ok(read)
        } else if let Some((handle, len)) = file.reader.as_ref() {
            // Serve at most the bytes remaining to EOF (`count` is already capped
            // by the caller via READ_MAX_BYTES). Positional read: no `try_clone`
            // (a per-call Windows `DuplicateHandle`/Unix `dup`) and no cursor
            // seek — `seek_read`/`read_at` take `&File` and move nothing, so the
            // one shared handle serves this fd's sequential stream directly.
            let want =
                usize::try_from((*len).saturating_sub(pos).min(out.len() as u64)).unwrap_or(0);
            let read = positional_read_into(handle, pos, &mut out[..want])?;
            inner.position = pos + read as u64;
            Ok(read)
        } else {
            Ok(0)
        }
    }

    /// Write `bytes` to an open, writable descriptor at its current position
    /// (extending the file as needed), advancing the position and marking the
    /// fd dirty so [`close`](Self::close) flushes it to the host. A read-only
    /// fd is rejected with `PermissionDenied`; an unknown fd with `NotFound`.
    pub fn write(&self, fd: Fd, bytes: &[u8]) -> Result<usize, std::io::Error> {
        let files = self.open_files.read();
        let Some(file) = files.get(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        // A `memory:` fd is never `writable`, so the check below would already
        // refuse this. Naming the scheme first turns a bare EACCES into a log
        // line a compatibility report can act on.
        if let Some(region) = file.memory_region.as_ref() {
            memory_scheme::refuse("write-attempt", region.addr, || {
                format!(
                    "write to read-only memory: file '{}' (fd {fd})",
                    region.display_name
                )
            });
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "memory: pseudo-files are read-only",
            ));
        }
        if !file.writable {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "fd not opened for writing",
            ));
        }
        let mut inner = file.inner.lock();
        let pos = inner.position as usize;
        let buf = inner.data.get_or_insert_with(Vec::new);
        if pos > buf.len() {
            buf.resize(pos, 0); // sparse gap → zero-fill
        }
        let end = pos + bytes.len();
        if end > buf.len() {
            buf.resize(end, 0);
        }
        buf[pos..end].copy_from_slice(bytes);
        inner.position = end as u64;
        inner.dirty = true;
        debug!("VFS write: fd={fd}, {} bytes at {pos}", bytes.len());
        Ok(bytes.len())
    }

    /// Positional read (`pread`): read up to `count` bytes at absolute
    /// `offset` WITHOUT moving the descriptor's position — the whole point of
    /// pread is that concurrent streaming readers never disturb each other's
    /// cursor. Reads past end-of-file return the short (possibly empty) tail,
    /// like POSIX.
    pub fn pread(&self, fd: Fd, count: usize, offset: u64) -> Result<Vec<u8>, std::io::Error> {
        let mut bytes = vec![0u8; count];
        let read = self.pread_into(fd, &mut bytes, offset)?;
        bytes.truncate(read);
        Ok(bytes)
    }

    /// Allocation-free positional read into caller-owned storage.
    pub fn pread_into(&self, fd: Fd, out: &mut [u8], offset: u64) -> Result<usize, std::io::Error> {
        let files = self.open_files.read();
        let Some(file) = files.get(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        if file.dirents.is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "fd is a directory",
            ));
        }
        if file.random_device {
            getrandom::fill(out)
                .map_err(|error| std::io::Error::other(format!("host entropy failed: {error}")))?;
            return Ok(out.len());
        }
        if let Some(region) = file.memory_region.as_ref() {
            // A `memory:` region is immutable after open and this touches no
            // per-fd state, so no per-file lock is needed — concurrent preads
            // and the sequential stream on one such fd never serialize.
            return self.read_memory_region(region, out, offset);
        }
        // In-memory branch: read the write-back buffer coherently with a
        // concurrent same-fd `write`, still WITHOUT touching `position`.
        let inner = file.inner.lock();
        if let Some(data) = inner.data.as_ref() {
            let pos = usize::try_from(offset)
                .unwrap_or(usize::MAX)
                .min(data.len());
            let end = pos.saturating_add(out.len()).min(data.len());
            let read = end - pos;
            out[..read].copy_from_slice(&data[pos..end]);
            return Ok(read);
        }
        drop(inner);
        // Reader branch: `reader` is immutable after open and `positional_read`
        // touches no per-fd state, so release the per-file lock before the read.
        // Concurrent preads (and the sequential stream) on this same read-only fd
        // then run without serializing on that mutex — the asset streamer issues
        // many positional reads against one open archive handle. `seek_read`/
        // `read_at` need neither a `try_clone` (a `DuplicateHandle`/`dup` syscall
        // per call) nor a cursor move, so a shared `&File` is safe under the map
        // read lock.
        if let Some((handle, len)) = file.reader.as_ref() {
            let start = offset.min(*len);
            let want =
                usize::try_from((*len).saturating_sub(start).min(out.len() as u64)).unwrap_or(0);
            positional_read_into(handle, start, &mut out[..want])
        } else {
            Ok(0)
        }
    }

    /// Positional write (`pwrite`): write `bytes` at absolute `offset` WITHOUT
    /// moving the descriptor's position — the write-side twin of
    /// [`pread`](Self::pread), for streaming loaders that issue ordered writes
    /// against one shared fd from several threads. A read-only fd is rejected
    /// with `PermissionDenied`; an unknown fd with `NotFound`.
    ///
    /// Extends the write-back buffer (zero-filling any sparse gap, like POSIX)
    /// and marks the fd dirty so [`close`](Self::close) persists it, exactly as
    /// [`write`](Self::write) does — only the cursor handling differs.
    pub fn pwrite(&self, fd: Fd, bytes: &[u8], offset: u64) -> Result<usize, std::io::Error> {
        let files = self.open_files.read();
        let Some(file) = files.get(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        // A `memory:` fd is never `writable`, so the check below would already
        // refuse this. Naming the scheme first turns a bare EACCES into a log
        // line a compatibility report can act on.
        if let Some(region) = file.memory_region.as_ref() {
            memory_scheme::refuse("write-attempt", region.addr, || {
                format!(
                    "write to read-only memory: file '{}' (fd {fd})",
                    region.display_name
                )
            });
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "memory: pseudo-files are read-only",
            ));
        }
        if !file.writable {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "fd not opened for writing",
            ));
        }
        let pos = usize::try_from(offset).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "offset too large")
        })?;
        let mut inner = file.inner.lock();
        let buf = inner.data.get_or_insert_with(Vec::new);
        if pos > buf.len() {
            buf.resize(pos, 0); // sparse gap → zero-fill
        }
        let end = pos.checked_add(bytes.len()).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "offset overflow")
        })?;
        if end > buf.len() {
            buf.resize(end, 0);
        }
        buf[pos..end].copy_from_slice(bytes);
        inner.dirty = true;
        debug!("VFS pwrite: fd={fd}, {} bytes at {pos}", bytes.len());
        Ok(bytes.len())
    }

    /// Truncate (or zero-extend) an OPEN descriptor to `len` bytes
    /// (`ftruncate`). Resizes the fd's write-back buffer so the new length
    /// survives the flush-on-close: shrinking drops the tail, extending
    /// zero-fills — both matching POSIX. A read-only fd is rejected with
    /// `PermissionDenied`; an unknown fd with `NotFound`.
    ///
    /// Note the extension is materialized in the buffer (the same allocation
    /// exposure `write` already accepts when a guest seeks far past EOF and
    /// writes): this VFS has no sparse-file representation.
    pub fn ftruncate(&self, fd: Fd, len: u64) -> Result<(), std::io::Error> {
        let files = self.open_files.read();
        let Some(file) = files.get(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        // A `memory:` fd is never `writable`, so the check below would already
        // refuse this. Naming the scheme first turns a bare EACCES into a log
        // line a compatibility report can act on.
        if let Some(region) = file.memory_region.as_ref() {
            memory_scheme::refuse("write-attempt", region.addr, || {
                format!(
                    "write to read-only memory: file '{}' (fd {fd})",
                    region.display_name
                )
            });
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "memory: pseudo-files are read-only",
            ));
        }
        if !file.writable {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "fd not opened for writing",
            ));
        }
        let len = usize::try_from(len).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "length too large")
        })?;
        let mut inner = file.inner.lock();
        let buf = inner.data.get_or_insert_with(Vec::new);
        buf.resize(len, 0);
        // A truncation past the cursor leaves the cursor where it was (POSIX
        // keeps the file offset unchanged), so no `position` update here.
        inner.dirty = true;
        debug!("VFS ftruncate: fd={fd} -> {len} bytes");
        Ok(())
    }

    /// Reposition an open file descriptor. `whence` follows POSIX:
    /// `SEEK_SET` (0) = absolute, `SEEK_CUR` (1) = relative to current,
    /// `SEEK_END` (2) = relative to end-of-file. Returns the new absolute
    /// position, or an error for a bad fd / negative resulting offset.
    pub fn seek(&self, fd: Fd, offset: i64, whence: i32) -> Result<u64, std::io::Error> {
        let files = self.open_files.read();
        let Some(file) = files.get(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        let mut inner = file.inner.lock();
        let size = backing_len(file, &inner);
        let base = match whence {
            0 => 0i64,                  // SEEK_SET
            1 => inner.position as i64, // SEEK_CUR
            2 => size as i64,           // SEEK_END
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
        // A directory fd's enumeration cursor IS `position` (the byte offset
        // into its packed dirent blocks), so `lseek(dirfd, 0, SEEK_SET)`
        // (rewinddir) restarts the walk with no extra bookkeeping.
        inner.position = target as u64;
        Ok(inner.position)
    }

    /// The size in bytes of an open file's backing data (0 if the host file
    /// was absent at open time). Used by `fstat`/`lseek(SEEK_END)`.
    pub fn file_size(&self, fd: Fd) -> Option<u64> {
        let files = self.open_files.read();
        let file = files.get(&fd)?;
        let inner = file.inner.lock();
        Some(backing_len(file, &inner))
    }

    /// Whether `fd` names a `memory:` pseudo-file (see [`memory_scheme`]).
    ///
    /// Exposed so `fstat` and diagnostics can tell a guest-memory-backed regular
    /// file apart from a host-backed one; both report `S_IFREG`, which is what
    /// the title expects, so this is informational rather than a mode switch.
    #[must_use]
    pub fn is_memory_file(&self, fd: Fd) -> bool {
        self.open_files
            .read()
            .get(&fd)
            .is_some_and(|file| file.memory_region.is_some())
    }

    /// Whether `fd` is an open directory descriptor.
    ///
    /// HLE `fstat` needs this: a directory must report `S_IFDIR` in
    /// `st_mode`, not pose as a zero-length regular file — UE5-style
    /// directory walkers branch on it between "list further" and "read".
    #[must_use]
    pub fn is_directory(&self, fd: Fd) -> bool {
        self.open_files
            .read()
            .get(&fd)
            .is_some_and(|file| file.dirents.is_some())
    }

    /// Whether `fd` names one of the process-local entropy character devices.
    ///
    /// HLE `fstat` needs this distinction: reporting `/dev/random` as a
    /// regular file makes runtimes reject an otherwise successful entropy
    /// read.
    #[must_use]
    pub fn is_random_device(&self, fd: Fd) -> bool {
        self.open_files
            .read()
            .get(&fd)
            .is_some_and(|file| file.random_device)
    }

    /// Guest path associated with an open descriptor.
    ///
    /// This is intentionally a cloned string: callers use it only for
    /// diagnostics, and retaining a map-lock-backed reference across I/O would
    /// unnecessarily serialize descriptor operations.
    #[must_use]
    pub fn open_path(&self, fd: Fd) -> Option<String> {
        self.open_files
            .read()
            .get(&fd)
            .map(|file| file.ps5_path.clone())
    }

    /// Return the descriptor's Orbis open flags.
    pub fn flags(&self, fd: Fd) -> Option<i32> {
        let files = self.open_files.read();
        let file = files.get(&fd)?;
        Some(file.inner.lock().flags)
    }

    /// Update status flags while preserving the descriptor's access mode.
    pub fn set_status_flags(&self, fd: Fd, flags: i32) -> Result<(), std::io::Error> {
        use open_flags::O_ACCMODE;
        let files = self.open_files.read();
        let Some(file) = files.get(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        let mut inner = file.inner.lock();
        inner.flags = (inner.flags & O_ACCMODE) | (flags & !O_ACCMODE);
        Ok(())
    }

    /// Return as many whole 512-byte directory blocks of packed Orbis dirent
    /// records as fit the caller's buffer (see [`pack_dirents`] for the
    /// record/block layout), advancing the descriptor's byte cursor. Returns
    /// `(payload, base)` where `base` is the byte offset before this call —
    /// the value `getdirentries` stores through `basep`. An exhausted
    /// directory returns an empty payload (guest return 0), never a synthetic
    /// record.
    pub fn getdents(&self, fd: Fd, requested: usize) -> Result<(Vec<u8>, usize), std::io::Error> {
        if requested < DIRENT_BLOCK {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "directory buffer is smaller than one directory block",
            ));
        }
        let files = self.open_files.read();
        let Some(file) = files.get(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        let Some(dirents) = file.dirents.as_ref() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "fd is not a directory",
            ));
        };
        let mut inner = file.inner.lock();
        let base = usize::try_from(inner.position).unwrap_or(usize::MAX);
        if base >= dirents.len() {
            return Ok((Vec::new(), base));
        }
        // Whole blocks only: records are padded to block boundaries, so any
        // 512-multiple is also a record boundary.
        let take = (dirents.len() - base).min(requested / DIRENT_BLOCK * DIRENT_BLOCK);
        let payload = dirents[base..base + take].to_vec();
        inner.position += take as u64;
        tracing::debug!(
            "getdents fd={fd}: {take} bytes of packed dirents at offset {base} (of {})",
            dirents.len()
        );
        Ok((payload, base))
    }

    /// Persist an open descriptor's dirty write-back buffer without closing
    /// it. Read-only descriptors succeed; unknown descriptors return
    /// `NotFound`. This backs the guest's `fsync` durability boundary.
    pub fn sync(&self, fd: Fd) -> Result<(), std::io::Error> {
        let files = self.open_files.read();
        let Some(file) = files.get(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        let mut inner = file.inner.lock();
        flush_open_file(&file.host_path, file.writable, &mut inner)?;
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

        let files = self.open_files.read();
        let mut flushed = 0usize;
        let mut first_error = None;
        for file in files.values().filter(|file| {
            file.ps5_path == root
                || file
                    .ps5_path
                    .strip_prefix(&root)
                    .is_some_and(|suffix| suffix.starts_with('/'))
        }) {
            let mut inner = file.inner.lock();
            match flush_open_file(&file.host_path, file.writable, &mut inner) {
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
        // `remove` needs the map WRITE lock — one of the few operations that
        // still does. Once removed, we own the `OpenFile`, so the per-file mutex
        // can be consumed without locking.
        let Some(file) = self.open_files.write().remove(&fd) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("fd {fd} not open"),
            ));
        };
        let inner = file.inner.into_inner();
        if let Some(region) = file.memory_region.as_ref() {
            // Nothing to flush and nothing to unmap: the bytes belong to the
            // guest, which still owns that buffer after the handle is gone.
            // `host_path` is empty for these, so falling into the flush branch
            // below would try to `std::fs::write("")`.
            debug!(
                "VFS close: fd={fd} (memory: file '{}', {} bytes at {:#x})",
                region.display_name, region.len, region.addr
            );
            return Ok(());
        }
        if inner.dirty && file.writable {
            if let Some(ref data) = inner.data {
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

/// The length of whichever backing an open descriptor has.
///
/// One definition shared by `lseek(SEEK_END)` and `fstat`, because the two
/// disagreeing is how a title ends up reading past what a read will serve.
/// Ordered by authority:
/// * a `memory:` region's **declared** length — never re-derived, since that is
///   the number the title itself supplied and will size its own buffer from;
/// * the in-memory `data` buffer (a writable fd's write-back store);
/// * the lazy host `reader`'s length;
/// * a directory's packed dirent listing, always a multiple of 512, so
///   `lseek(SEEK_END)` reports exactly what `getdents` walks — mirroring
///   shadPS4's `NormalDirectory`.
fn backing_len(file: &OpenFile, inner: &OpenFileMut) -> u64 {
    file.memory_region
        .as_ref()
        .map(|region| region.len)
        .or_else(|| inner.data.as_ref().map(|data| data.len() as u64))
        .or_else(|| file.reader.as_ref().map(|(_, len)| *len))
        .or_else(|| file.dirents.as_ref().map(|bin| bin.len() as u64))
        .unwrap_or(0)
}

/// Flush one open file's write-back buffer and report whether it was dirty.
/// Takes the immutable `host_path`/`writable` plus the locked mutable state, so
/// callers can drive it while holding only the map READ lock and this fd's mutex.
fn flush_open_file(
    host_path: &Path,
    writable: bool,
    inner: &mut OpenFileMut,
) -> Result<bool, std::io::Error> {
    if !inner.dirty || !writable {
        return Ok(false);
    }
    if let Some(ref data) = inner.data {
        std::fs::write(host_path, data)?;
        std::fs::OpenOptions::new()
            .write(true)
            .open(host_path)?
            .sync_all()?;
        debug!(
            "VFS sync: flushed {} bytes -> {}",
            data.len(),
            host_path.display()
        );
    }
    inner.dirty = false;
    Ok(true)
}

/// Positional read of up to `want` bytes at absolute `offset` from a shared
/// read-only handle, looping over short reads until `want` bytes or EOF.
///
/// This is the per-read cost the hot path sheds. The old streaming/pread dance
/// was `seek(offset)` + `take(want).read_to_end()` — and for `pread` a
/// `try_clone()` first (a Windows `DuplicateHandle` / Unix `dup` syscall) purely
/// to get an independent cursor. `seek_read`/`read_at` take `&File` and read at
/// an explicit offset WITHOUT moving the handle's cursor, so they need neither
/// the clone nor the seek and are safe to issue concurrently on one shared
/// handle: the win compounds over the streamer's thousands of small reads.
fn positional_read_into(
    handle: &std::fs::File,
    offset: u64,
    out: &mut [u8],
) -> Result<usize, std::io::Error> {
    let mut filled = 0usize;
    while filled < out.len() {
        // `seek_read`/`read_at` may return short; loop until `want` or EOF.
        let n = loop {
            #[cfg(windows)]
            let result = {
                use std::os::windows::fs::FileExt;
                handle.seek_read(&mut out[filled..], offset + filled as u64)
            };
            #[cfg(unix)]
            let result = {
                use std::os::unix::fs::FileExt;
                handle.read_at(&mut out[filled..], offset + filled as u64)
            };
            match result {
                Ok(n) => break n,
                // A signal can interrupt the read mid-syscall; retry, as
                // `read_to_end` did internally.
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        };
        if n == 0 {
            break; // EOF before `want` — return the short tail, like POSIX.
        }
        filled += n;
    }
    Ok(filled)
}

/// The guest directory a **relative** guest path is resolved against.
///
/// Orbis titles run with their app root as the process working directory
/// (KytyPS5 loads the executable as `/app0/eboot.bin` and mounts the app
/// directory at both `/app0` and `/hostapp` — `reference/kytyps5`,
/// `src/main.cpp:138` / `src/emulator.cpp:201`). Raeen has no `chdir`/`getcwd`
/// HLE, so this is fixed rather than per-process; when one is added, this
/// becomes its default.
pub const GUEST_WORKING_DIRECTORY: &str = "/app0";

/// The eboot name [`guest_argv0`] falls back to when the host file's own name
/// cannot be spelled as a guest path component.
const DEFAULT_EBOOT_NAME: &str = "eboot.bin";

/// The environment a title is entered with (`envp`, laid out by the runtime's
/// `build_process_stack` between `argv`'s terminator and the auxv).
///
/// An Orbis process has a real, FreeBSD-derived environment — the PS5 SDK's own
/// rtld resolves shared libraries by reading `LD_LIBRARY_PATH` out of it
/// (`reference/ps5-payload-sdk`, `crt/rtld.c:223`) — so an empty `envp` is a
/// shape no process has on hardware.
///
/// This is a *description*, not a control knob: Raeen resolves `DT_NEEDED`
/// itself and never reads this back. The value it states is true anyway —
/// `sce_module/` under the app root is exactly where the loader looks for a
/// title's shipped system modules (`raeen-firmware`'s `DEPENDENCY_SUBDIRS`) —
/// so a title that parses `envp` for its own reasons is told the truth. Kept
/// deliberately minimal for the same reason: every entry here is a string a
/// guest may act on, and inventing variables no hardware sets would be a lie
/// with a larger blast radius than an empty environment.
///
/// The path is spelled out because [`GUEST_WORKING_DIRECTORY`] cannot be
/// concatenated in a `const`; `guest_envp_paths_live_under_the_app_mount`
/// pins the two together.
pub const GUEST_ENVP: &[&str] = &["LD_LIBRARY_PATH=/app0/sce_module"];

/// The `argv[0]` a title is entered with, for the eboot at `host_eboot`.
///
/// `argv[0]` is the one string a crt0 hands straight to the title, and titles
/// print it, parse it, and open it — so it must name a path the guest can
/// actually reach. Measured on Blasphemous II (PPSA13580): while the runner
/// passed the raw host path, the title's own launcher banner printed
/// `Arg 0 = E:\PS5\PPSA13580-app\eboot.bin` — a spelling no guest API can open,
/// since the file is mounted at [`GUEST_WORKING_DIRECTORY`].
///
/// So: the eboot's *basename* under `/app0`, which is the path the app mount
/// resolves back to that same file. shadPS4 builds `argv[0]` identically
/// (`reference/shadps4`, `src/emulator.cpp:285`: `"/app0/" + eboot_name`);
/// KytyPS5 passes a bare `"KytyEmu"` (`src/loader/runtimeLinker.cpp:1359`),
/// which keeps the host out of guest memory but names nothing openable.
///
/// A basename that cannot serve as a guest path component (absent, non-UTF-8,
/// empty, or carrying host path syntax of its own) falls back to `eboot.bin`
/// rather than passing the odd spelling through.
pub fn guest_argv0(host_eboot: &Path) -> String {
    let name = host_eboot
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && !looks_like_host_path(name))
        .unwrap_or(DEFAULT_EBOOT_NAME);
    format!("{GUEST_WORKING_DIRECTORY}/{name}")
}

/// Whether `s` carries *host* path syntax — a Windows drive letter (`E:\…`,
/// `E:/…`) or any backslash separator. Guest paths are `/`-rooted and contain
/// neither.
///
/// This is a syntactic check, and on a POSIX host it cannot be more than that:
/// `/home/user/Games/x/eboot.bin` is shaped exactly like a guest path. What
/// closes that gap is rooting every guest-visible path at
/// [`GUEST_WORKING_DIRECTORY`] (see [`guest_argv0`]) rather than detecting the
/// host spelling after the fact. Used to keep host paths out of `argv`/`envp`,
/// and to make the leak loud if one ever gets that far.
pub fn looks_like_host_path(s: &str) -> bool {
    if s.contains('\\') {
        return true;
    }
    let mut chars = s.chars();
    matches!(
        (chars.next(), chars.next()),
        (Some(drive), Some(':')) if drive.is_ascii_alphabetic()
    )
}

/// Normalize a guest path into the absolute, `.`-free spelling that mount
/// prefix matching can compare literally.
///
/// Mount matching is a literal string prefix compare, so any spelling the
/// guest is entitled to use but that does not *look* like `/<mount>/...` was
/// reported as "path is not mounted". Measured on Blasphemous II (PPSA13580):
/// `sceKernelMkdir('././')` failed that way immediately before the title
/// stalled. `'././'` is a legal spelling of the current directory and must
/// resolve to it.
///
/// Three normalizations, all of which only ever *shorten* the path:
///
/// 1. `.` and empty components are dropped, which also collapses the `//`
///    double slashes shadPS4 corrects for the same reason (`reference/shadps4`,
///    `src/core/file_sys/fs.cpp:46`: *"Evil games like Turok2 pass double
///    slashes"*).
/// 2. A relative path is anchored at [`GUEST_WORKING_DIRECTORY`]. Without this
///    a relative guest path matched no mount at all. This is the *sandboxed*
///    form of what KytyPS5 does — its `MountPoints::GetRealFilename`
///    (`src/kernel/fileSystem.cpp:226-245`) returns an unmatched guest path
///    verbatim, which on the host resolves against the emulator's own working
///    directory. Deliberately **not** copied: a raw guest string used as a host
///    path is the escape `combine_within_mount` exists to close.
/// 3. Trailing slashes disappear with the empty components, so `/app0/` and
///    `/app0` are the same path.
///
/// `..` is deliberately left **in place**. Resolving it here would change which
/// mount a path matches; it belongs to [`combine_within_mount`], which pops it
/// with a clamp at the mount root so no amount of `..` escapes (and which a
/// UE title's `../../../Content/...` depends on). An empty input stays empty so
/// `open("")` keeps failing rather than silently becoming the app root.
fn normalize_guest_path(ps5_path: &str) -> String {
    if ps5_path.is_empty() {
        return String::new();
    }
    let absolute = ps5_path.starts_with('/');
    let mut normalized = if absolute {
        String::new()
    } else {
        normalize_mount_root(GUEST_WORKING_DIRECTORY)
    };
    for segment in ps5_path.split('/') {
        if segment.is_empty() || segment == "." {
            continue;
        }
        normalized.push('/');
        normalized.push_str(segment);
    }
    if normalized.is_empty() {
        // Every component was `.` or empty on an absolute path: the root.
        "/".to_string()
    } else {
        normalized
    }
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

/// The longest single path component the host filesystems in scope accept
/// (`NAME_MAX` on Linux, comfortably under `MAX_PATH` on Windows). A guest
/// segment longer than this cannot name a real host file, and long enough
/// segments make the host path calls below error; reject up front.
const MAX_HOST_NAME_LEN: usize = 255;

/// Combine a mount-relative guest path onto a mount's host root, refusing any
/// input that would escape that root. Returns `None` (fail-closed) on denial;
/// callers map that to `NOT_FOUND`/`PermissionDenied`.
///
/// Ported from SharpEmu (GPL-2.0, commit e01092a) —
/// `KernelMemoryCompatExports.CombineWithinMount` + `EscapesMountViaReparsePoint`
/// + `NormalizeMountRelativePath`, pinned by `KernelSandboxEscapeTests`.
///
/// The escape this closes: `NormalizeMountRelativePath`-style stripping of
/// `.`/`..` splits only on separators, so a drive-qualified token like `C:`
/// survives as a segment. `Path::join` then DISCARDS the mount root because its
/// argument is drive-rooted, yielding a raw host path such as `C:\Windows\...`.
/// On Windows a tail like `/app0/C:/Windows/System32/...` is therefore absolute
/// and grants arbitrary host read/write/delete. Lexical containment alone also
/// does not follow symlinks/junctions, so a reparse point planted inside the
/// mount could redirect out of it.
///
/// # Why `..` is normalized rather than refused (UE4/UE5 boot)
///
/// Refusing any path with a `..` component is a *stricter* rule than SharpEmu's
/// and it breaks Unreal Engine titles outright. A UE game's base directory is
/// `<app>/binaries/<platform>`, so the engine addresses its own content with
/// `../../../`-prefixed paths that, on real hardware, land back inside `/app0`.
/// Denying them means the title enumerates nothing and never finds its `.pak`
/// files — SharpEmu measured exactly that (`NormalizeMountRelativePath`,
/// GPL-2.0: *"Combining those raw against the app0 root walked out of the game
/// folder entirely, so the title enumerated an unrelated host directory and
/// never found its .pak files."*). Until Dawn is a UE5 title.
///
/// So `..` **pops** the last resolved segment and is **silently dropped at the
/// mount root** — the resolved path is clamped to the mount, never escaping it.
/// This is not a weakening: popping can only ever shorten the segment list, so
/// the by-construction lexical containment below is unchanged, and the
/// drive-qualifier, reparse-point, and canonical-containment defenses all still
/// run on the normalized result.
///
/// Defense, in order:
/// 1. Sanitize each segment — normalize `.`/`..` with a clamp at the mount root,
///    refuse any absolute segment, any segment containing `:` (drive/ADS
///    qualifier), and NUL/over-long names.
///    After this the assembled path is lexically contained *by construction*.
/// 2. Refuse a final symlink explicitly, including a dangling one that
///    `canonicalize` cannot resolve. This keeps `O_CREAT` from following a
///    dangling file link outside the sandbox.
/// 3. Canonicalize the candidate or its nearest existing ancestor and ASSERT it
///    stays under the canonical mount identity. This resolves every existing
///    intermediate symlink/junction in one host operation and fails closed on
///    errors other than a genuinely-missing tail.
fn combine_within_mount(
    mount_root: &Path,
    cached_canonical_root: Option<&Path>,
    ps5_path: &str,
    relative: &str,
) -> Option<PathBuf> {
    // A mount with an empty host root (the stubbed `/dev`, `/proc`) has no
    // backing directory. Building a candidate from an empty root produces a
    // CWD-relative host path AND skips the canonical-containment assertion
    // below (`canonicalize("")` errors), letting a guest `open("/dev/<name>")`
    // read or create `./<name>` in the emulator's working directory. Fail
    // closed: an unbacked mount resolves to nothing.
    if mount_root.as_os_str().is_empty() {
        warn!("VFS resolve: refusing guest path '{ps5_path}' under an unbacked (empty-root) mount");
        return None;
    }
    // --- 1. Segment sanitation. ---
    let mut segments: Vec<&str> = Vec::new();
    for segment in relative.split(['/', '\\']) {
        if segment.is_empty() || segment == "." {
            continue;
        }
        if segment == ".." {
            // Pop one resolved segment, or drop the `..` when already at the
            // mount root (the clamp). `Vec::pop` on an empty vec is a no-op, so
            // no amount of leading `..` can walk above the root — which is what
            // makes a UE title's `../../../Content/...` resolve back INTO the
            // mount instead of being denied.
            segments.pop();
            continue;
        }
        // A ':' is a Windows drive or alternate-data-stream qualifier, and an
        // absolute segment would make `Path::join` discard the mount root; both
        // are escapes.
        if segment.contains(':') || Path::new(segment).is_absolute() {
            warn!(
                "VFS resolve: refusing drive-qualified/absolute segment in guest path '{ps5_path}'"
            );
            return None;
        }
        // A NUL or an over-long name cannot name a real host file and makes the
        // canonicalization calls below error; deny explicitly rather than lean
        // on those to fail closed.
        if segment.len() > MAX_HOST_NAME_LEN || segment.as_bytes().contains(&0) {
            warn!("VFS resolve: refusing malformed segment in guest path '{ps5_path}'");
            return None;
        }
        segments.push(segment);
    }

    // Assemble the candidate. Built only from plain-name segments, so it is
    // lexically under `mount_root` by construction.
    let mut candidate = mount_root.to_path_buf();
    for segment in &segments {
        candidate.push(segment);
    }

    // --- 2. Final-component symlink check. ---
    // `canonicalize` returns NotFound for a dangling symlink. If this is an
    // O_CREAT target, treating it as an ordinary missing tail would let the
    // subsequent host open follow the link. One metadata lookup on the exact
    // candidate closes that case without walking every component.
    match std::fs::symlink_metadata(&candidate) {
        Ok(meta) if meta.file_type().is_symlink() => {
            warn!("VFS resolve: refusing symlink/reparse target in guest path '{ps5_path}'");
            return None;
        }
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => {
            warn!("VFS resolve: unreadable target in guest path '{ps5_path}' ({err}); denying");
            return None;
        }
    }

    // --- 3. Canonical containment assertion. ---
    // Mount mappings normally point at existing roots, whose canonical identity
    // was cached when installed. A root that did not exist then is re-checked
    // here so a later-created directory gains the same containment guard.
    let discovered_root = cached_canonical_root
        .is_none()
        .then(|| canonicalize_mount_root(mount_root))
        .flatten();
    let canonical_root = cached_canonical_root.or(discovered_root.as_deref());

    if let Some(canonical_root) = canonical_root {
        let mut probe = candidate.as_path();
        loop {
            match std::fs::canonicalize(probe) {
                Ok(real) if real.starts_with(canonical_root) => break,
                Ok(_) => {
                    warn!(
                        "VFS resolve: guest path '{ps5_path}' resolves outside its mount; denying"
                    );
                    return None;
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    // A missing O_CREAT/optional-file tail has nothing to
                    // resolve. Climb until an existing parent can prove that
                    // the complete existing prefix stays inside the mount.
                    if probe == mount_root {
                        warn!(
                            "VFS resolve: mount root disappeared while resolving guest path \
                             '{ps5_path}'; denying"
                        );
                        return None;
                    }
                    let Some(parent) = probe.parent() else {
                        warn!(
                            "VFS resolve: cannot verify containment of guest path '{ps5_path}'; \
                             denying"
                        );
                        return None;
                    };
                    if !parent.starts_with(mount_root) {
                        warn!(
                            "VFS resolve: guest path '{ps5_path}' escaped while checking its \
                             existing prefix; denying"
                        );
                        return None;
                    }
                    probe = parent;
                }
                Err(err) => {
                    warn!(
                        "VFS resolve: cannot verify containment of guest path '{ps5_path}' \
                         ({err}); denying"
                    );
                    return None;
                }
            }
        }
    }

    Some(candidate)
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
        let d = std::env::temp_dir().join(format!("raeen-vfs-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    // ---------------------------------------------------------------------
    // `memory:` pseudo-file scheme
    // ---------------------------------------------------------------------

    /// The five URIs GTA V was measured requesting. Addresses and lengths are
    /// as logged; the *contents* below are fabricated by [`pattern_byte`], so no
    /// game bytes are involved.
    const GTA_V_URIS: [&str; 5] = [
        "memory:$1085a08000,9156,0:00005_initial_interactive_screen_ps5.gfx",
        "memory:$1546320000,110250,0:00010_game_stream.gfx",
        "memory:$1546320000,63305,0:00011_generic_instructional_buttons.gfx",
        "memory:$1559e00000,198674,0:00002_font_lib_efigs_ps5.gfx",
        "memory:$1559ec0000,114275,0:00015_loadingscreen_startup.gfx",
    ];

    /// Deterministic content as a pure function of the guest address, so a read
    /// at any offset can be verified without holding a reference copy.
    fn pattern_byte(addr: u64) -> u8 {
        (addr.wrapping_mul(31).wrapping_add(7) >> 3) as u8
    }

    /// A fabricated guest address space: a set of readable regions and nothing
    /// else. Deliberately does NOT override `guest_range_readable`, so the
    /// trait's default page probe is exercised too.
    struct FakeGuestMemory {
        regions: Vec<(u64, u64)>,
        /// Set to model the guest unmapping its buffers while a descriptor is
        /// still open — every subsequent read must fail closed.
        revoked: std::sync::atomic::AtomicBool,
    }

    impl FakeGuestMemory {
        fn over_regions(regions: &[(u64, u64)]) -> Arc<dyn GuestByteSource> {
            Arc::new(Self {
                regions: regions.to_vec(),
                revoked: std::sync::atomic::AtomicBool::new(false),
            })
        }

        /// The four distinct buffers the five GTA V URIs address. Two URIs share
        /// base `0x1546320000` with different lengths, exactly as measured — one
        /// buffer, two views — so the larger length covers both.
        fn gta_v() -> Arc<dyn GuestByteSource> {
            Self::over_regions(&[
                (0x10_85A0_8000, 9156),
                (0x15_4632_0000, 110_250),
                (0x15_59E0_0000, 198_674),
                (0x15_59EC_0000, 114_275),
            ])
        }
    }

    impl GuestByteSource for FakeGuestMemory {
        fn read_guest_bytes(&self, addr: u64, out: &mut [u8]) -> bool {
            if self.revoked.load(std::sync::atomic::Ordering::Relaxed) {
                return false;
            }
            let Some(end) = addr.checked_add(out.len() as u64) else {
                return false;
            };
            let covered = self
                .regions
                .iter()
                .any(|&(base, len)| addr >= base && end <= base + len);
            if !covered {
                return false;
            }
            for (index, slot) in out.iter_mut().enumerate() {
                *slot = pattern_byte(addr + index as u64);
            }
            true
        }
    }

    /// A VFS with a real `/app0` mount (so a plain path still resolves) and the
    /// fabricated GTA V address space bound. The `Arc` is returned so the caller
    /// keeps the weakly-held source alive.
    fn vfs_with_gta_v_memory() -> (VirtualFileSystem, Arc<dyn GuestByteSource>, PathBuf) {
        let dir = temp_dir("memory-scheme");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        let source = FakeGuestMemory::gta_v();
        vfs.set_guest_byte_source(&source);
        (vfs, source, dir)
    }

    #[test]
    fn the_five_gta_v_scaleform_uris_open_and_read_back_their_whole_declared_length() {
        use open_flags::O_RDONLY;
        let (vfs, _source, _dir) = vfs_with_gta_v_memory();

        for uri in GTA_V_URIS {
            let expected = memory_scheme::parse(uri).expect("fixture URI parses");
            let fd = vfs
                .open(uri, O_RDONLY, 0)
                .unwrap_or_else(|e| panic!("{uri} must open, got {e}"));
            assert!(vfs.is_memory_file(fd), "{uri} must be a memory: descriptor");
            assert_eq!(
                vfs.file_size(fd),
                Some(expected.len),
                "{uri} must report its DECLARED length, not a re-derived one"
            );

            // Whole file through the sequential cursor, in one call.
            let bytes = vfs.read(fd, expected.len as usize + 64).expect("read");
            assert_eq!(
                bytes.len(),
                expected.len as usize,
                "{uri}: a read past EOF must return the short tail, not over-read"
            );
            for (offset, got) in bytes.iter().enumerate() {
                assert_eq!(
                    *got,
                    pattern_byte(expected.addr + offset as u64),
                    "{uri}: byte at offset {offset} came from the wrong guest address"
                );
            }
            // Cursor is at EOF; a further read is empty and does not error.
            assert_eq!(vfs.read(fd, 16).expect("read at eof").len(), 0);
            vfs.close(fd).expect("close");
        }
    }

    #[test]
    fn memory_backed_reads_serve_offset_zero_mid_file_and_a_tail_that_straddles_the_end() {
        use open_flags::O_RDONLY;
        let (vfs, _source, _dir) = vfs_with_gta_v_memory();
        let uri = "memory:$1559e00000,198674,0:00002_font_lib_efigs_ps5.gfx";
        let base = 0x15_59E0_0000u64;
        let len = 198_674u64;
        let fd = vfs.open(uri, O_RDONLY, 0).expect("open");

        // Offset 0.
        let mut head = [0xEEu8; 32];
        assert_eq!(vfs.pread_into(fd, &mut head, 0).expect("pread head"), 32);
        for (i, got) in head.iter().enumerate() {
            assert_eq!(*got, pattern_byte(base + i as u64), "head byte {i}");
        }

        // Mid-file.
        let mid = len / 2;
        let mut middle = [0xEEu8; 64];
        assert_eq!(
            vfs.pread_into(fd, &mut middle, mid).expect("pread mid"),
            64,
            "a mid-file positional read must be complete"
        );
        for (i, got) in middle.iter().enumerate() {
            assert_eq!(*got, pattern_byte(base + mid + i as u64), "mid byte {i}");
        }

        // Straddling the end: only the bytes before EOF are served, and the ones
        // past it are left untouched rather than zero-filled.
        let mut tail = [0xEEu8; 100];
        let from = len - 40;
        assert_eq!(
            vfs.pread_into(fd, &mut tail, from).expect("pread tail"),
            40,
            "a straddling read must be short, not padded"
        );
        for (i, got) in tail[..40].iter().enumerate() {
            assert_eq!(*got, pattern_byte(base + from + i as u64), "tail byte {i}");
        }
        assert!(
            tail[40..].iter().all(|b| *b == 0xEE),
            "bytes past EOF must not be written at all"
        );

        // Wholly past the end: empty, not an error.
        let mut past = [0xEEu8; 16];
        assert_eq!(vfs.pread_into(fd, &mut past, len + 4096).unwrap(), 0);
        assert!(past.iter().all(|b| *b == 0xEE));

        // pread never disturbed the sequential cursor.
        let mut sequential = [0u8; 8];
        assert_eq!(vfs.read_into(fd, &mut sequential).unwrap(), 8);
        for (i, got) in sequential.iter().enumerate() {
            assert_eq!(*got, pattern_byte(base + i as u64), "cursor byte {i}");
        }
        vfs.close(fd).expect("close");
    }

    #[test]
    fn lseek_on_a_memory_file_uses_the_declared_length_and_reads_nothing_past_eof() {
        use open_flags::O_RDONLY;
        let (vfs, _source, _dir) = vfs_with_gta_v_memory();
        let uri = "memory:$1085a08000,9156,0:00005_initial_interactive_screen_ps5.gfx";
        let base = 0x10_85A0_8000u64;
        let len = 9156u64;
        let fd = vfs.open(uri, O_RDONLY, 0).expect("open");

        // SEEK_END(0) is the size probe Scaleform uses before allocating.
        assert_eq!(vfs.seek(fd, 0, 2).expect("SEEK_END"), len);
        assert_eq!(vfs.read(fd, 16).expect("read at eof").len(), 0);

        // SEEK_SET into the middle, then a relative SEEK_CUR.
        assert_eq!(vfs.seek(fd, 100, 0).expect("SEEK_SET"), 100);
        assert_eq!(vfs.seek(fd, 28, 1).expect("SEEK_CUR"), 128);
        let mut at128 = [0u8; 4];
        assert_eq!(vfs.read_into(fd, &mut at128).unwrap(), 4);
        for (i, got) in at128.iter().enumerate() {
            assert_eq!(*got, pattern_byte(base + 128 + i as u64));
        }

        // Seeking PAST eof is legal (POSIX) and leaves the cursor there; the
        // read that follows is empty and must not move it backwards.
        assert_eq!(vfs.seek(fd, 1_000_000, 0).expect("past eof"), 1_000_000);
        assert_eq!(vfs.read(fd, 32).expect("read past eof").len(), 0);
        assert_eq!(
            vfs.seek(fd, 0, 1).expect("cursor unchanged"),
            1_000_000,
            "a 0-byte EOF read must leave the offset alone"
        );
        assert_eq!(vfs.seek(fd, -8, 2).expect("SEEK_END back"), len - 8);
        assert_eq!(vfs.read(fd, 64).expect("read tail").len(), 8);

        // Negative absolute offsets are still refused.
        assert_eq!(
            vfs.seek(fd, -1, 0).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        vfs.close(fd).expect("close");
    }

    #[test]
    fn a_memory_uri_naming_unmapped_guest_memory_is_refused_without_allocating_a_descriptor() {
        use open_flags::O_RDONLY;
        let (vfs, _source, _dir) = vfs_with_gta_v_memory();
        let before = *vfs.next_fd.read();

        for (uri, why) in [
            // Right length, address nowhere near a mapped region.
            ("memory:$dead0000,4096,0:phantom.gfx", "an unmapped range"),
            // Correct base, one byte longer than the mapped buffer — the case a
            // lexical range check without a map would wave through.
            (
                "memory:$1085a08000,9157,0:one_too_far.gfx",
                "a range one byte past the mapping",
            ),
        ] {
            let error = vfs
                .open(uri, O_RDONLY, 0)
                .expect_err("{why} must not become a descriptor");
            assert_eq!(error.kind(), std::io::ErrorKind::NotFound, "{why}: {uri}");
            // Must be THIS scheme's refusal, naming the range — not the generic
            // "path is not mounted" a host-path walk produces.
            assert!(
                error.to_string().contains("memory:"),
                "{why} must be refused by the memory: scheme, got '{error}'"
            );
        }

        assert_eq!(
            *vfs.next_fd.read(),
            before,
            "a refused memory: open must not consume a guest descriptor"
        );
    }

    #[test]
    fn a_memory_uri_without_a_bound_address_space_is_refused_rather_than_served_empty() {
        use open_flags::O_RDONLY;
        let dir = temp_dir("memory-no-space");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        // --- Never bound: a missing runtime call site. ---
        assert_eq!(
            vfs.guest_byte_source().err(),
            Some(AddressSpaceMiss::NeverBound)
        );
        let error = vfs
            .open(GTA_V_URIS[0], O_RDONLY, 0)
            .expect_err("no address space must be an error, not a zero-length file");
        assert_eq!(error.kind(), std::io::ErrorKind::Other);
        assert!(
            error.to_string().contains("missing its bind call"),
            "must point at the missing call site, not merely report an absent binding — got \
             '{error}'"
        );

        // --- Bound, then dropped: a lifetime problem, NOT a missing call site.
        // These two report differently on purpose: one measurement of the
        // ambiguous message sent this scheme's first GTA V run at the wrong
        // hypothesis. ---
        let source = FakeGuestMemory::gta_v();
        vfs.set_guest_byte_source(&source);
        let weak = Arc::downgrade(&source);
        assert!(vfs.open(GTA_V_URIS[0], O_RDONLY, 0).is_ok());
        drop(source);
        assert_eq!(
            weak.strong_count(),
            0,
            "the VFS must hold the guest address space weakly, never extend its life"
        );
        assert_eq!(
            vfs.guest_byte_source().err(),
            Some(AddressSpaceMiss::Dropped)
        );
        let error = vfs
            .open(GTA_V_URIS[1], O_RDONLY, 0)
            .expect_err("a dead address space must refuse");
        assert_eq!(error.kind(), std::io::ErrorKind::Other);
        assert!(
            error.to_string().contains("dropped"),
            "a dropped arena must not be reported as never bound — got '{error}'"
        );

        // The two blocker keys must differ, or the digest aggregates a wiring
        // bug and a lifetime bug into one indistinguishable count.
        assert_ne!(
            AddressSpaceMiss::NeverBound.blocker_reason(),
            AddressSpaceMiss::Dropped.blocker_reason()
        );
    }

    #[test]
    fn a_memory_backed_read_is_revalidated_so_a_range_unmapped_after_open_cannot_be_read() {
        use open_flags::O_RDONLY;
        let dir = temp_dir("memory-revoke");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        let concrete = Arc::new(FakeGuestMemory {
            regions: vec![(0x10_85A0_8000, 9156)],
            revoked: std::sync::atomic::AtomicBool::new(false),
        });
        let source: Arc<dyn GuestByteSource> = concrete.clone();
        vfs.set_guest_byte_source(&source);

        let fd = vfs.open(GTA_V_URIS[0], O_RDONLY, 0).expect("open");
        assert_eq!(vfs.read(fd, 16).expect("first read").len(), 16);

        // The guest frees its buffer while the handle is still open.
        concrete
            .revoked
            .store(true, std::sync::atomic::Ordering::Relaxed);

        let mut out = [0u8; 16];
        assert_eq!(
            vfs.read_into(fd, &mut out).unwrap_err().kind(),
            std::io::ErrorKind::NotFound,
            "validation at open must not be the only guard — every read re-checks"
        );
        assert_eq!(
            vfs.pread_into(fd, &mut out, 0).unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        );
        // A size probe still answers from the URI; only byte access faults.
        assert_eq!(vfs.file_size(fd), Some(9156));
        vfs.close(fd).expect("close still succeeds");
    }

    #[test]
    fn every_malformed_memory_uri_is_refused_by_name_and_never_reaches_host_resolution() {
        use open_flags::O_RDONLY;
        let (vfs, _source, _dir) = vfs_with_gta_v_memory();
        let before = *vfs.next_fd.read();
        for uri in [
            "memory:",                                  // nothing at all
            "memory:1085a08000,9156,0:x.gfx",           // missing `$`
            "memory:$,9156,0:x.gfx",                    // empty address
            "memory:$zzzz,9156,0:x.gfx",                // non-hex address
            "memory:$0,9156,0:x.gfx",                   // null address
            "memory:$1085a08000",                       // missing length field
            "memory:$1085a08000,9156",                  // missing flags field
            "memory:$1085a08000,nine,0:x.gfx",          // non-decimal length
            "memory:$1085a08000,0,0:x.gfx",             // zero length
            "memory:$1085a08000,999999999999,0:x.gfx",  // absurd length
            "memory:$1085a08000,9156,zero:x.gfx",       // non-decimal flags
            "memory:$1085a08000,9156,0",                // missing display name
            "memory:$1085a08000,9156,0:",               // empty display name
            "memory:$ffffffffffffffff,9156,0:wrap.gfx", // addr+len wraps
        ] {
            let Err(error) = vfs.open(uri, O_RDONLY, 0) else {
                panic!("{uri} must be refused, not opened");
            };
            assert_eq!(
                error.kind(),
                std::io::ErrorKind::NotFound,
                "{uri} must be a named ENOENT refusal"
            );
            assert!(
                error.to_string().contains("memory:"),
                "{uri} refusal must name the scheme, got '{error}'"
            );
            // Never a host path, so never a host filesystem probe.
            assert_eq!(vfs.resolve_path(uri), None, "{uri} must not resolve");
        }
        assert_eq!(
            *vfs.next_fd.read(),
            before,
            "no malformed URI may consume a descriptor"
        );
    }

    #[test]
    fn an_ordinary_guest_path_containing_a_colon_is_still_refused_by_the_drive_qualifier_guard() {
        use open_flags::*;
        let (vfs, _source, dir) = vfs_with_gta_v_memory();

        // The escape the guard exists for: a drive-qualified segment. Still
        // denied — routing `memory:` off did not relax it.
        for hostile in [
            "/app0/C:/Windows/System32/drivers/etc/hosts",
            "/app0/sub/D:evil.bin",
            "/app0/stream:ads",
            "memoryfoo:$1000,4,0:not-the-scheme.gfx",
        ] {
            assert_eq!(
                vfs.resolve_path(hostile),
                None,
                "{hostile} must not resolve to a host path"
            );
            assert!(
                vfs.open(hostile, O_RDONLY, 0).is_err(),
                "{hostile} must not open read-only"
            );
            assert!(
                vfs.open(hostile, O_WRONLY | O_CREAT, 0o644).is_err(),
                "{hostile} must not be creatable"
            );
        }

        // And an ordinary colon-free path in the same mount still works, so the
        // guard was not turned into a blanket denial.
        std::fs::write(dir.join("ordinary.bin"), b"REAL").unwrap();
        let fd = vfs.open("/app0/ordinary.bin", O_RDONLY, 0).expect("open");
        assert_eq!(vfs.read(fd, 4).unwrap(), b"REAL");
        vfs.close(fd).unwrap();
        let _ = std::fs::remove_file(dir.join("ordinary.bin"));
    }

    #[test]
    fn a_memory_backed_descriptor_refuses_every_write_path_and_touches_no_host_file() {
        use open_flags::*;
        let (vfs, _source, dir) = vfs_with_gta_v_memory();

        // Opening for write never yields a descriptor at all.
        let before = *vfs.next_fd.read();
        for flags in [
            O_WRONLY,
            O_RDWR,
            O_RDONLY | O_CREAT,
            O_RDONLY | O_TRUNC,
            O_WRONLY | O_CREAT | O_TRUNC,
        ] {
            let Err(error) = vfs.open(GTA_V_URIS[0], flags, 0o644) else {
                panic!("flags {flags:#x} must be refused, not opened");
            };
            assert_eq!(
                error.kind(),
                std::io::ErrorKind::PermissionDenied,
                "flags {flags:#x} must be EACCES, never a silent success"
            );
        }
        assert_eq!(*vfs.next_fd.read(), before);

        // And every fd-side mutation on a read-only handle is refused too.
        let fd = vfs.open(GTA_V_URIS[0], O_RDONLY, 0).expect("open");
        assert_eq!(
            vfs.write(fd, b"NOPE").unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(
            vfs.pwrite(fd, b"NOPE", 0).unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(
            vfs.ftruncate(fd, 0).unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        // A refused write must not have altered what a read serves.
        let mut check = [0u8; 4];
        assert_eq!(vfs.pread_into(fd, &mut check, 0).unwrap(), 4);
        for (i, got) in check.iter().enumerate() {
            assert_eq!(*got, pattern_byte(0x10_85A0_8000 + i as u64));
        }
        assert_eq!(vfs.file_size(fd), Some(9156), "length is unchanged");

        // fsync succeeds (nothing to flush) and close writes no host file.
        vfs.sync(fd).expect("fsync on a read-only handle succeeds");
        let host_files_before = std::fs::read_dir(&dir).unwrap().count();
        vfs.close(fd).expect("close");
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            host_files_before,
            "closing a memory: handle must not create or flush a host file"
        );
    }

    #[test]
    fn a_memory_backed_descriptor_is_not_a_directory_and_not_an_entropy_device() {
        use open_flags::O_RDONLY;
        let (vfs, _source, _dir) = vfs_with_gta_v_memory();
        let fd = vfs.open(GTA_V_URIS[3], O_RDONLY, 0).expect("open");
        assert!(!vfs.is_directory(fd));
        assert!(!vfs.is_random_device(fd));
        assert!(vfs.is_memory_file(fd));
        assert_eq!(
            vfs.open_path(fd).as_deref(),
            Some(GTA_V_URIS[3]),
            "the URI is retained verbatim for diagnostics"
        );
        assert_eq!(
            vfs.getdents(fd, 512).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput,
            "getdents on a regular memory: file must be ENOTDIR-shaped, not a panic"
        );
        assert_eq!(vfs.flags(fd), Some(O_RDONLY));
        vfs.close(fd).expect("close");
        assert!(!vfs.is_memory_file(fd), "a closed fd is no longer known");
    }

    #[test]
    fn memory_file_len_answers_stat_sized_queries_without_opening_a_descriptor() {
        let (vfs, _source, _dir) = vfs_with_gta_v_memory();
        let before = *vfs.next_fd.read();
        for uri in GTA_V_URIS {
            let expected = memory_scheme::parse(uri).unwrap().len;
            assert_eq!(
                vfs.memory_file_len(uri)
                    .unwrap_or_else(|e| panic!("{uri}: {e}")),
                expected,
                "{uri}: stat must report the declared length"
            );
        }
        // Unmapped and malformed are both refusals here too, not a size of 0.
        assert!(vfs.memory_file_len("memory:$dead0000,16,0:x.gfx").is_err());
        assert!(vfs.memory_file_len("memory:$1085a08000,0,0:x.gfx").is_err());
        assert!(vfs.memory_file_len("/app0/plain.bin").is_err());
        assert_eq!(*vfs.next_fd.read(), before, "stat must allocate no fd");
    }

    #[test]
    fn two_descriptors_over_one_aliased_guest_buffer_keep_independent_cursors() {
        use open_flags::O_RDONLY;
        let (vfs, _source, _dir) = vfs_with_gta_v_memory();
        // The two measured URIs that share base 0x1546320000 with different
        // declared lengths — one buffer, two views.
        let long = vfs
            .open(GTA_V_URIS[1], O_RDONLY, 0)
            .expect("open long view");
        let short = vfs
            .open(GTA_V_URIS[2], O_RDONLY, 0)
            .expect("open short view");
        assert_eq!(vfs.file_size(long), Some(110_250));
        assert_eq!(vfs.file_size(short), Some(63_305));

        assert_eq!(vfs.read(long, 1000).unwrap().len(), 1000);
        assert_eq!(
            vfs.seek(short, 0, 1).unwrap(),
            0,
            "reading one view must not move the other's cursor"
        );
        // The short view's EOF is its own declared length, not the long one's.
        assert_eq!(vfs.seek(short, 0, 2).unwrap(), 63_305);
        assert_eq!(vfs.read(short, 16).unwrap().len(), 0);
        assert_eq!(vfs.seek(long, 0, 2).unwrap(), 110_250);
        vfs.close(long).unwrap();
        vfs.close(short).unwrap();
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
    fn missing_read_only_open_returns_not_found_without_allocating_fd() {
        use open_flags::O_RDONLY;
        let dir = temp_dir("missing-open");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        let error = vfs
            .open("/app0/does-not-exist.bin", O_RDONLY, 0)
            .expect_err("a missing file without O_CREAT must not become an empty descriptor");
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        assert_eq!(
            *vfs.next_fd.read(),
            3,
            "a failed open must not consume a guest descriptor"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_into_and_pread_into_fill_callers_buffer_without_cursor_interference() {
        use open_flags::O_RDONLY;
        let dir = temp_dir("read-into");
        std::fs::write(dir.join("stream.bin"), b"0123456789").unwrap();
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        let fd = vfs.open("/app0/stream.bin", O_RDONLY, 0).unwrap();

        let mut first = [0xEE; 4];
        assert_eq!(vfs.read_into(fd, &mut first).unwrap(), 4);
        assert_eq!(&first, b"0123");

        let mut positional = [0xEE; 3];
        assert_eq!(vfs.pread_into(fd, &mut positional, 7).unwrap(), 3);
        assert_eq!(&positional, b"789");

        let mut second = [0xEE; 4];
        assert_eq!(vfs.read_into(fd, &mut second).unwrap(), 4);
        assert_eq!(
            &second, b"4567",
            "pread_into must not disturb the sequential cursor"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pwrite_writes_at_an_offset_without_moving_the_cursor() {
        use open_flags::*;
        let dir = temp_dir("pwrite");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        let fd = vfs
            .open("/app0/blob.bin", O_RDWR | O_CREAT | O_TRUNC, 0o644)
            .unwrap();
        assert_eq!(vfs.write(fd, b"aaaa").unwrap(), 4);
        // Positional write at offset 1 leaves the cursor at 4.
        assert_eq!(vfs.pwrite(fd, b"ZZ", 1).unwrap(), 2);
        assert_eq!(vfs.seek(fd, 0, 1).unwrap(), 4, "cursor unmoved by pwrite");
        // pwrite into the buffered content reads back through pread...
        assert_eq!(vfs.pread(fd, 4, 0).unwrap(), b"aZZa");
        // ...and persists through the flush-on-close.
        vfs.close(fd).unwrap();
        assert_eq!(std::fs::read(dir.join("blob.bin")).unwrap(), b"aZZa");

        // A sparse pwrite zero-fills the gap, like POSIX.
        let fd = vfs
            .open("/app0/sparse.bin", O_RDWR | O_CREAT | O_TRUNC, 0o644)
            .unwrap();
        assert_eq!(vfs.pwrite(fd, b"end", 6).unwrap(), 3);
        assert_eq!(vfs.pread(fd, 9, 0).unwrap(), b"\0\0\0\0\0\0end");
        vfs.close(fd).unwrap();

        // Read-only and unknown fds are rejected.
        let fd = vfs.open("/app0/blob.bin", O_RDONLY, 0).unwrap();
        assert_eq!(
            vfs.pwrite(fd, b"x", 0).unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        vfs.close(fd).unwrap();
        assert_eq!(
            vfs.pwrite(0x7fff, b"x", 0).unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ftruncate_shrinks_and_zero_extends_the_open_descriptor() {
        use open_flags::*;
        let dir = temp_dir("ftruncate");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        let fd = vfs
            .open("/app0/file.bin", O_RDWR | O_CREAT | O_TRUNC, 0o644)
            .unwrap();
        assert_eq!(vfs.write(fd, b"0123456789").unwrap(), 10);
        // Shrink drops the tail and survives the flush.
        vfs.ftruncate(fd, 4).unwrap();
        assert_eq!(vfs.file_size(fd), Some(4));
        vfs.close(fd).unwrap();
        assert_eq!(std::fs::read(dir.join("file.bin")).unwrap(), b"0123");

        // Extend zero-fills; POSIX leaves the file offset alone.
        let fd = vfs.open("/app0/file.bin", O_RDWR, 0).unwrap();
        vfs.ftruncate(fd, 6).unwrap();
        assert_eq!(vfs.file_size(fd), Some(6));
        assert_eq!(vfs.pread(fd, 6, 0).unwrap(), b"0123\0\0");
        vfs.close(fd).unwrap();
        assert_eq!(std::fs::read(dir.join("file.bin")).unwrap(), b"0123\0\0");

        // Read-only and unknown fds are rejected.
        let fd = vfs.open("/app0/file.bin", O_RDONLY, 0).unwrap();
        assert_eq!(
            vfs.ftruncate(fd, 1).unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        vfs.close(fd).unwrap();
        assert_eq!(
            vfs.ftruncate(0x7fff, 1).unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn random_devices_stream_host_entropy_without_a_host_mount() {
        use open_flags::*;
        let vfs = VirtualFileSystem::new();

        for path in ["/dev/random", "/dev/urandom"] {
            let fd = vfs.open(path, O_RDONLY, 0).expect("open random device");
            let first = vfs.read(fd, 32).expect("first entropy read");
            let second = vfs.pread(fd, 32, 0).expect("positional entropy read");
            assert_eq!(first.len(), 32);
            assert_eq!(second.len(), 32);
            assert!(vfs.is_random_device(fd));
            assert_ne!(
                first, second,
                "independent reads should supply fresh entropy"
            );
            vfs.close(fd).expect("close random device");
            assert!(!vfs.is_random_device(fd));
        }

        assert_eq!(
            vfs.open("/dev/urandom", O_WRONLY, 0).unwrap_err().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        assert_eq!(
            vfs.open("/dev/not-a-device", O_RDONLY, 0)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::NotFound
        );
    }

    /// A read-only open must stream a large file lazily instead of buffering it
    /// whole at open. The regression this guards: ASTRO.BOT's 6.7 MiB
    /// `game_text.xml` was slurped into a `Vec` at open, which OOMed under host
    /// commit pressure — the open failed, the title read that as "file absent",
    /// got a null fd, and crashed. Chunked read, seek, size, and pread must all
    /// work off the lazy handle and return the exact bytes.
    #[test]
    fn readonly_open_streams_a_large_file_lazily() {
        use open_flags::*;
        let dir = temp_dir("lazy-read");
        // Several MiB of deterministic content, larger than one read.
        let content: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(dir.join("big.bin"), &content).unwrap();

        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        let fd = vfs.open("/app0/big.bin", O_RDONLY, 0).unwrap();

        // SEEK_END and fstat report the on-disk length without buffering it.
        assert_eq!(vfs.seek(fd, 0, 2).unwrap(), content.len() as u64);
        assert_eq!(vfs.file_size(fd), Some(content.len() as u64));
        assert_eq!(vfs.seek(fd, 0, 0).unwrap(), 0);

        // Sequential chunked reads reassemble the file byte-for-byte and stop
        // cleanly at EOF.
        let mut got = Vec::new();
        loop {
            let chunk = vfs.read(fd, 100_000).unwrap();
            if chunk.is_empty() {
                break;
            }
            got.extend_from_slice(&chunk);
        }
        assert_eq!(got, content, "streamed read must match the file exactly");

        // A positional read reads the right window and does not disturb the
        // sequential cursor (which is now at EOF).
        let mid = vfs.pread(fd, 16, 1_000_000).unwrap();
        assert_eq!(mid, content[1_000_000..1_000_016]);

        vfs.close(fd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two DIFFERENT read-only fds must read concurrently. Before the
    /// interior-mutability refactor, `read()` took the whole-map WRITE lock, so
    /// no two reads could ever be in flight at once — that global serialization
    /// is what compounded over the streamer's thousands of small reads into
    /// multi-second asset loads. Different fds now hold different per-file
    /// mutexes; the barrier maximizes overlap, and both streams must reassemble
    /// their files byte-for-byte under it.
    #[test]
    fn two_fds_read_concurrently_and_reassemble_correctly() {
        use open_flags::*;
        use std::sync::{Arc, Barrier};
        let dir = temp_dir("concurrent-fds");
        let a: Vec<u8> = (0..800_000u32).map(|i| (i % 251) as u8).collect();
        let b: Vec<u8> = (0..800_000u32)
            .map(|i| ((i.wrapping_mul(7) + 3) % 251) as u8)
            .collect();
        std::fs::write(dir.join("a.bin"), &a).unwrap();
        std::fs::write(dir.join("b.bin"), &b).unwrap();

        let vfs = Arc::new(VirtualFileSystem::new());
        vfs.set_game_directory(&dir);
        let fda = vfs.open("/app0/a.bin", O_RDONLY, 0).unwrap();
        let fdb = vfs.open("/app0/b.bin", O_RDONLY, 0).unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let stream = |vfs: Arc<VirtualFileSystem>, fd: Fd, barrier: Arc<Barrier>| {
            std::thread::spawn(move || {
                barrier.wait();
                let mut got = Vec::new();
                loop {
                    let chunk = vfs.read(fd, 4096).unwrap();
                    if chunk.is_empty() {
                        break;
                    }
                    got.extend_from_slice(&chunk);
                }
                got
            })
        };
        let ta = stream(vfs.clone(), fda, barrier.clone());
        let tb = stream(vfs.clone(), fdb, barrier.clone());
        assert_eq!(ta.join().unwrap(), a, "fd A must reassemble exactly");
        assert_eq!(tb.join().unwrap(), b, "fd B must reassemble exactly");

        vfs.close(fda).unwrap();
        vfs.close(fdb).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A positional `pread` must never disturb a concurrent streaming `read`'s
    /// cursor, even on the SAME fd. The streamer's sequential reads advance
    /// `position` and must still reassemble the whole file, while thousands of
    /// preads hammer fixed offsets in parallel — each returning its exact window.
    /// (`pread` is cursor-free by design; this pins that it stays so under real
    /// concurrency, not just sequentially.)
    #[test]
    fn concurrent_pread_does_not_disturb_a_streaming_reads_cursor() {
        use open_flags::*;
        use std::sync::Arc;
        let dir = temp_dir("pread-vs-stream");
        let content: Arc<Vec<u8>> = Arc::new((0..1_000_000u32).map(|i| (i % 251) as u8).collect());
        std::fs::write(dir.join("big.bin"), content.as_slice()).unwrap();

        let vfs = Arc::new(VirtualFileSystem::new());
        vfs.set_game_directory(&dir);
        let fd = vfs.open("/app0/big.bin", O_RDONLY, 0).unwrap();

        let streamer = {
            let (vfs, expect) = (vfs.clone(), content.clone());
            std::thread::spawn(move || {
                let mut got = Vec::new();
                loop {
                    let chunk = vfs.read(fd, 3333).unwrap();
                    if chunk.is_empty() {
                        break;
                    }
                    got.extend_from_slice(&chunk);
                }
                assert_eq!(
                    got, *expect,
                    "streamed read must reassemble the file exactly"
                );
            })
        };
        let preader = {
            let (vfs, expect) = (vfs.clone(), content.clone());
            std::thread::spawn(move || {
                for _ in 0..3000 {
                    for off in [0u64, 100_000, 500_000, 999_990] {
                        let want = ((expect.len() as u64 - off).min(16)) as usize;
                        let got = vfs.pread(fd, 16, off).unwrap();
                        assert_eq!(
                            got,
                            expect[off as usize..off as usize + want],
                            "pread window at {off} must be exact"
                        );
                    }
                }
            })
        };
        streamer.join().unwrap();
        preader.join().unwrap();

        vfs.close(fd).unwrap();
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

    /// A writable open of a traversing path is **confined**, not refused: the
    /// `..` clamps at the mount root, so the write lands inside the mount. With
    /// no game directory mounted there is nothing to resolve against, so it
    /// still fails closed — just not with a blanket traversal rejection that
    /// would also have denied every Unreal `Saved/` write.
    #[test]
    fn writable_open_of_traversing_path_is_confined_not_refused() {
        use open_flags::*;
        let dir = temp_dir("traverse-write");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        let fd = vfs
            .open("/app0/../../escape.bin", O_WRONLY | O_CREAT, 0o644)
            .expect("clamped writable open must succeed");
        vfs.write(fd, b"x").unwrap();
        vfs.close(fd).unwrap();
        assert!(
            dir.join("escape.bin").is_file(),
            "the write must be clamped INTO the mount, not escape above it"
        );
        assert!(
            !dir.parent().unwrap().join("escape.bin").exists(),
            "nothing may be created above the mount root"
        );
        let _ = std::fs::remove_dir_all(&dir);
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
    fn read_past_eof_leaves_the_cursor_in_place() {
        use open_flags::*;
        let dir = temp_dir("eof-cursor");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        let fd = vfs
            .open("/app0/eof.bin", O_RDWR | O_CREAT | O_TRUNC, 0o644)
            .unwrap();
        vfs.write(fd, b"aaaa").unwrap(); // 4-byte file; cursor now at 4.
        assert_eq!(vfs.seek(fd, 10, 0).unwrap(), 10); // SEEK_SET past EOF.
        assert!(
            vfs.read(fd, 8).unwrap().is_empty(),
            "a read starting past EOF returns nothing"
        );
        // POSIX: a 0-byte EOF read must not move the offset. The bug rewound it
        // to EOF (4), which then corrupted a following write on an O_RDWR fd.
        assert_eq!(
            vfs.seek(fd, 0, 1).unwrap(),
            10,
            "read past EOF must leave the cursor at 10, not rewind to EOF"
        );
        vfs.close(fd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rewinddir_via_lseek_restarts_enumeration() {
        use open_flags::*;
        let dir = temp_dir("rewinddir");
        std::fs::write(dir.join("a.bin"), b"a").unwrap();
        std::fs::write(dir.join("b.bin"), b"b").unwrap();
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        let fd = vfs.open("/app0", O_RDONLY, 0).unwrap();

        let drain = |fd| {
            let mut n = 0;
            loop {
                let (bytes, _) = vfs.getdents(fd, 4096).unwrap();
                if bytes.is_empty() {
                    break;
                }
                n += parse_dirent_blocks(&bytes).len();
            }
            n
        };
        let first = drain(fd);
        assert!(
            first >= 4,
            "expected the two dot entries plus the two files, got {first}"
        );

        // rewinddir: `lseek(dirfd, 0, SEEK_SET)` must restart the walk. Before
        // the fix it only reset `position`, not `directory_index`, so this
        // re-enumeration yielded nothing.
        vfs.seek(fd, 0, 0).unwrap();
        assert_eq!(drain(fd), first, "rewound enumeration must repeat");
        vfs.close(fd).unwrap();
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

    /// Walk packed Orbis dirent records: `d_fileno`(u32) `d_reclen`(u16)
    /// `d_type`(u8) `d_namlen`(u8) `d_name[]`, records padded so none crosses
    /// a 512-byte block boundary (FreeBSD `DIRBLKSIZ` semantics, mirrored by
    /// shadPS4's `NormalDirectory`). Asserts the structural invariants a real
    /// guest parser depends on, then returns `(d_type, name, d_reclen)`.
    fn parse_dirent_blocks(bytes: &[u8]) -> Vec<(u8, String, u16)> {
        assert!(
            bytes.len().is_multiple_of(512),
            "getdents returns whole 512-byte directory blocks, got {}",
            bytes.len()
        );
        let mut records = Vec::new();
        let mut offset = 0usize;
        while offset < bytes.len() {
            let fileno = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            let reclen =
                u16::from_le_bytes(bytes[offset + 4..offset + 6].try_into().unwrap()) as usize;
            let d_type = bytes[offset + 6];
            let namlen = bytes[offset + 7] as usize;
            assert_ne!(fileno, 0, "d_fileno must be non-zero");
            assert!(reclen > 8 + namlen, "d_reclen must cover the record");
            // The overflow that smashed Until Dawn's canary: a record whose
            // d_reclen walks past the bytes the call actually returned.
            assert!(
                offset + reclen <= bytes.len(),
                "d_reclen {reclen} at offset {offset} exceeds returned payload {}",
                bytes.len()
            );
            // Records never cross a 512-byte directory-block boundary.
            assert_eq!(
                offset / 512,
                (offset + reclen - 1) / 512,
                "record at {offset} (reclen {reclen}) crosses a 512-byte block boundary"
            );
            assert_eq!(
                bytes[offset + 8 + namlen],
                0,
                "d_name must be NUL-terminated"
            );
            records.push((
                d_type,
                std::str::from_utf8(&bytes[offset + 8..offset + 8 + namlen])
                    .unwrap()
                    .to_string(),
                reclen as u16,
            ));
            offset += reclen;
        }
        assert_eq!(offset, bytes.len(), "records must tile the blocks exactly");
        records
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
        // Four entries: the two synthetic dot-dirs Orbis always yields first
        // (`.`, `..`), then the real `packs` and `stone.txt` — all packed
        // into ONE 512-byte block returned by ONE call.
        let (bytes, base) = vfs.getdents(fd, 1024).unwrap();
        assert_eq!(base, 0, "basep is the byte offset before the call");
        assert_eq!(bytes.len(), 512);
        let mut kinds_and_names: Vec<(u8, String)> = parse_dirent_blocks(&bytes)
            .into_iter()
            .map(|(d_type, name, _)| (d_type, name))
            .collect();
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
        assert_eq!(base, 512, "EOF basep reports the final byte offset");
        // lseek(SEEK_END) on a directory reports the 512-aligned dirent size.
        assert_eq!(vfs.seek(fd, 0, 2).unwrap(), 512);
        vfs.seek(fd, 0, 0).unwrap();
        vfs.set_status_flags(fd, O_APPEND).unwrap();
        assert_eq!(vfs.flags(fd), Some(O_APPEND | O_RDONLY));
        vfs.close(fd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The Until Dawn regression (ledger 2026-07-25/26 ITEM 2): an EMPTY
    /// `/app0/deepfiles` directory returned 0x200 bytes TWICE — one 512-byte
    /// record per dot entry, each claiming `d_reclen == 512`. A guest that
    /// copies a record by `d_reclen` into a `sizeof(dirent)`-sized (264-byte)
    /// stack struct overflows by 248 bytes and trips `__stack_chk_fail`.
    /// Real Orbis (and shadPS4) packs `.` and `..` into ONE 512-byte block —
    /// so: one 0x200 return, then 0.
    #[test]
    fn empty_directory_returns_one_dot_block_then_eof() {
        use open_flags::*;
        let dir = temp_dir("getdents-empty");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        let fd = vfs.open("/app0", O_RDONLY, 0).unwrap();

        let (first, base) = vfs.getdents(fd, 0x1000).unwrap();
        assert_eq!(base, 0);
        assert_eq!(first.len(), 512, "dot entries pack into one block");
        let records = parse_dirent_blocks(&first);
        assert_eq!(records.len(), 2, "exactly `.` and `..`");
        assert_eq!(records[0].0, 4);
        assert_eq!(records[0].1, ".");
        assert_eq!(records[1].0, 4);
        assert_eq!(records[1].1, "..");
        // The final record absorbs the block slack into its d_reclen
        // (FreeBSD directory-block semantics): 12 + 500 = 512.
        assert_eq!(records[0].2, 12);
        assert_eq!(records[1].2, 500);

        let (second, base) = vfs.getdents(fd, 0x1000).unwrap();
        assert!(
            second.is_empty(),
            "an exhausted directory must return 0 bytes, not another 0x200"
        );
        assert_eq!(base, 512);
        vfs.close(fd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Directories larger than one block: records never cross a 512-byte
    /// boundary, each block's last record absorbs the slack, and a 512-byte
    /// caller buffer walks the listing one block per call with byte basep.
    #[test]
    fn getdents_packs_multiple_blocks_without_boundary_crossings() {
        use open_flags::*;
        let dir = temp_dir("getdents-blocks");
        // 8 names of ~120 chars: reclen align4(8+120+1)=132; ~3 per block →
        // guarantees several blocks together with the dot entries.
        let mut expected: Vec<String> = (0..8).map(|i| format!("{i}{}", "x".repeat(119))).collect();
        for name in &expected {
            std::fs::write(dir.join(name), b"y").unwrap();
        }
        expected.push(".".to_string());
        expected.push("..".to_string());
        expected.sort();

        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        let fd = vfs.open("/app0", O_RDONLY, 0).unwrap();

        let total = vfs.seek(fd, 0, 2).unwrap();
        assert!(total > 512, "fixture must span multiple blocks");
        assert!(total.is_multiple_of(512));
        vfs.seek(fd, 0, 0).unwrap();

        let mut names = Vec::new();
        let mut expected_base = 0u64;
        loop {
            let (bytes, base) = vfs.getdents(fd, 512).unwrap();
            if bytes.is_empty() {
                break;
            }
            assert_eq!(base as u64, expected_base, "basep advances by bytes");
            assert_eq!(bytes.len(), 512, "a 512-byte buffer gets one block");
            for (_, name, _) in parse_dirent_blocks(&bytes) {
                names.push(name);
            }
            expected_base += 512;
        }
        assert_eq!(expected_base, total);
        names.sort();
        assert_eq!(names, expected);
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
        // `..` clamps at the mount root instead of escaping (or being denied).
        assert_eq!(
            vfs.resolve_path("/temp0/../escape.bin"),
            Some(root.join("escape.bin"))
        );

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(nested);
    }

    /// `'././'` is a legal spelling of the current directory and must resolve
    /// to it, not fail as "path is not mounted".
    ///
    /// Measured on Blasphemous II (PPSA13580): `sceKernelMkdir('././')` failed
    /// that way in the last seconds before the title stalled, because mount
    /// matching is a literal prefix compare and `'././'` neither starts with a
    /// mount prefix nor is absolute at all.
    #[test]
    fn dot_relative_guest_paths_resolve_against_the_app_root() {
        let app = temp_dir("dot-relative-app");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&app);

        // The exact measured call, plus the other spellings of the same idea.
        for spelling in ["././", ".", "./", "./.", "././."] {
            assert_eq!(
                vfs.resolve_path(spelling),
                Some(app.clone()),
                "guest path {spelling:?} must resolve to the app root"
            );
        }
        // A relative *name* lands inside the app root, not outside any mount.
        assert_eq!(
            vfs.resolve_path("./cache/state.bin"),
            Some(app.join("cache").join("state.bin"))
        );
        assert_eq!(
            vfs.resolve_path("cache/state.bin"),
            Some(app.join("cache").join("state.bin"))
        );
        // `mkdir` of the current directory is a no-op that must succeed.
        assert!(vfs.create_dir_all("././").is_ok());

        let _ = std::fs::remove_dir_all(app);
    }

    /// `guest_argv0` maps a host eboot path to the guest spelling of the same
    /// file: rooted at the app mount, carrying the eboot's own basename, and
    /// free of every trace of the host layout it came from. The Windows path
    /// here is the exact one Blasphemous II's launcher banner echoed back as
    /// `Arg 0 = E:\PS5\PPSA13580-app\eboot.bin`.
    #[test]
    fn guest_argv0_is_the_app_mount_spelling_never_the_host_path() {
        for host in [
            r"E:\PS5\PPSA13580-app\eboot.bin",
            r"C:\Users\someone\Documents\Raeen\Games\Title\eboot.bin",
            "/home/someone/Games/Title/eboot.bin",
            "eboot.bin",
        ] {
            let argv0 = guest_argv0(Path::new(host));
            assert_eq!(argv0, "/app0/eboot.bin", "host path {host:?}");
        }

        // A differently-named main module keeps its own name — argv[0] must
        // still resolve through the mount to the file that was launched.
        // Forward slashes so the basename is the basename on either host:
        // Windows accepts `/` as a separator, POSIX does not accept `\`.
        assert_eq!(
            guest_argv0(Path::new("E:/PS5/Homebrew/hello.elf")),
            "/app0/hello.elf"
        );

        // Nothing usable as a guest path component falls back rather than
        // passing an odd spelling through.
        assert_eq!(guest_argv0(Path::new("")), "/app0/eboot.bin");
        assert_eq!(guest_argv0(Path::new("/")), "/app0/eboot.bin");
    }

    /// No `argv[0]` this produces carries host path syntax, whatever the host
    /// layout — the property the guest side depends on, asserted directly.
    #[test]
    fn guest_argv0_never_carries_a_drive_letter_or_backslash() {
        for host in [
            r"E:\PS5\PPSA13580-app\eboot.bin",
            r"\\server\share\Title\eboot.bin",
            r"C:eboot.bin",
            "/mnt/games/Title/eboot.bin",
        ] {
            let argv0 = guest_argv0(Path::new(host));
            assert!(
                !looks_like_host_path(&argv0),
                "argv[0] {argv0:?} from host path {host:?} still reads as a host path"
            );
            assert!(
                argv0.starts_with(&format!("{GUEST_WORKING_DIRECTORY}/")),
                "argv[0] {argv0:?} must be rooted at the app mount"
            );
        }
    }

    /// The environment describes guest paths only, and every path in it lives
    /// under the app mount — so `GUEST_ENVP`'s spelled-out `/app0` cannot drift
    /// away from [`GUEST_WORKING_DIRECTORY`].
    #[test]
    fn guest_envp_paths_live_under_the_app_mount() {
        for entry in GUEST_ENVP {
            let (name, value) = entry
                .split_once('=')
                .unwrap_or_else(|| panic!("envp entry {entry:?} must be NAME=VALUE"));
            assert!(!name.is_empty(), "envp entry {entry:?} has an empty name");
            assert!(
                !looks_like_host_path(value),
                "envp entry {entry:?} leaks host path syntax"
            );
            assert!(
                value.starts_with(&format!("{GUEST_WORKING_DIRECTORY}/")),
                "envp entry {entry:?} must name a path under the app mount"
            );
        }
    }

    /// The host-path predicate the argv/envp checks rest on: drive letters and
    /// backslashes are host syntax; `/`-rooted guest paths are not.
    #[test]
    fn looks_like_host_path_flags_drive_letters_and_backslashes() {
        assert!(looks_like_host_path(r"E:\PS5\PPSA13580-app\eboot.bin"));
        assert!(looks_like_host_path("E:/PS5/eboot.bin"));
        assert!(looks_like_host_path(r"C:eboot.bin"));
        assert!(looks_like_host_path(r"\\server\share\eboot.bin"));
        assert!(looks_like_host_path(r"Games\Title\eboot.bin"));

        assert!(!looks_like_host_path("/app0/eboot.bin"));
        assert!(!looks_like_host_path("LD_LIBRARY_PATH=/app0/sce_module"));
        assert!(!looks_like_host_path("eboot.bin"));
        assert!(!looks_like_host_path(""));
        // A lone colon is not a drive letter: only `<alpha>:` at the front is.
        assert!(!looks_like_host_path("/app0/odd:name.bin"));
        assert!(!looks_like_host_path(":8080"));
    }

    /// Interior `.` components and doubled slashes collapse before mount
    /// matching; `..` does **not**, because resolving it here would change
    /// which mount a path matches (see `normalize_guest_path`).
    #[test]
    fn dot_and_double_slash_components_collapse_but_dotdot_survives() {
        assert_eq!(
            normalize_guest_path("/app0/./sce_sys/./param.json"),
            "/app0/sce_sys/param.json"
        );
        assert_eq!(
            normalize_guest_path("/app0//Media//level0"),
            "/app0/Media/level0"
        );
        assert_eq!(normalize_guest_path("/app0/"), "/app0");
        assert_eq!(normalize_guest_path("/"), "/");
        assert_eq!(normalize_guest_path("/./"), "/");
        // `..` is preserved verbatim for `combine_within_mount` to clamp.
        assert_eq!(
            normalize_guest_path("/app0/../../escape.bin"),
            "/app0/../../escape.bin"
        );
        // An empty path stays empty so `open("")` keeps failing.
        assert_eq!(normalize_guest_path(""), "");
    }

    /// Normalization must not open an escape: a relative path is anchored at
    /// the app root and then clamped there, so `..` still cannot walk out.
    #[test]
    fn normalizing_relative_paths_does_not_weaken_the_dotdot_clamp() {
        let app = temp_dir("dot-relative-clamp");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&app);

        for escape in ["../escape.bin", "./../../escape.bin", "../../../escape.bin"] {
            assert_eq!(
                vfs.resolve_path(escape),
                Some(app.join("escape.bin")),
                "{escape:?} must clamp INTO the app root"
            );
        }
        // A drive qualifier is still refused, relative spelling included.
        assert_eq!(
            vfs.resolve_path("./C:/Windows/System32/drivers/etc/hosts"),
            None
        );

        let _ = std::fs::remove_dir_all(app);
    }

    /// An unbacked mount stays unbacked: anchoring relative paths must not make
    /// a stubbed mount resolvable, and an empty path must not become a mount.
    #[test]
    fn normalization_does_not_resolve_unbacked_or_unknown_roots() {
        let vfs = VirtualFileSystem::new();
        assert_eq!(vfs.resolve_path("/dev/./null"), None);
        assert_eq!(vfs.resolve_path(""), None);
        // `/devlog` is devkit-only and absent on retail hardware: it must keep
        // resolving to nothing rather than gain an invented mount.
        assert_eq!(vfs.resolve_path("/devlog/app/debug.log"), None);
        assert_eq!(vfs.resolve_path("/devlog/./app/./debug.log"), None);
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
    fn mount_updates_refresh_the_cached_canonical_root() {
        let first = temp_dir("canonical-root-first");
        let second = temp_dir("canonical-root-second");
        let vfs = VirtualFileSystem::new();

        vfs.set_game_directory(&first);
        let cached_first = vfs
            .mounts
            .read()
            .iter()
            .find(|mount| mount.ps5_prefix == "/app0")
            .and_then(|mount| mount.canonical_root.clone());
        assert_eq!(cached_first, std::fs::canonicalize(&first).ok());

        vfs.set_game_directory(&second);
        let cached_second = vfs
            .mounts
            .read()
            .iter()
            .find(|mount| mount.ps5_prefix == "/app0")
            .and_then(|mount| mount.canonical_root.clone());
        assert_eq!(cached_second, std::fs::canonicalize(&second).ok());
        assert_ne!(cached_first, cached_second);

        let _ = std::fs::remove_dir_all(first);
        let _ = std::fs::remove_dir_all(second);
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

    // ---- Sandbox-escape regression tests (SharpEmu e01092a port). ----
    //
    // `resolve_path` is the guest->host boundary. A mount-relative guest path
    // must never resolve to a host path outside its mount root, and a malformed
    // path must fail closed (`None`) rather than throw or escape.

    #[cfg(windows)]
    fn try_symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_dir(target, link)
    }

    #[cfg(unix)]
    fn try_symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[test]
    fn drive_letter_injection_cannot_escape_mount() {
        // The core Windows escape: "/app0/C:/Windows/..." carries a drive-rooted
        // tail that `Path::join` would let REPLACE the mount root, granting raw
        // host access. The ':' segment is refused, so every form denies.
        let dir = temp_dir("sandbox-drive");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        for guest in [
            "/app0/C:/Windows/System32/drivers/etc/hosts",
            "/app0/C:/Windows/Temp/evil.dll",
            "/app0/data/C:/secret",
        ] {
            let resolved = vfs.resolve_path(guest);
            assert!(
                resolved.is_none(),
                "drive-qualified guest path {guest} must be denied, got {resolved:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `..` is normalized with a clamp at the mount root, not refused — so a
    /// traversing tail resolves back INSIDE the mount rather than escaping it
    /// and rather than being denied. Denial is the wrong contract: it breaks
    /// every Unreal title (see `combine_within_mount`'s UE section).
    #[test]
    fn parent_traversal_tail_is_clamped_to_the_mount_root() {
        let dir = temp_dir("sandbox-traverse");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        // `resolve_path` returns the mount root joined with the normalized
        // segments (not canonicalized), so compare against `dir` as mounted.
        let root = dir.clone();

        for guest in [
            "/app0/../escape.bin",
            "/app0/a/../../escape.bin",
            "/app0/..\\escape.bin",
            "/app0/../../../../../../escape.bin",
        ] {
            let resolved = vfs
                .resolve_path(guest)
                .unwrap_or_else(|| panic!("{guest} must resolve (clamped), not be denied"));
            assert_eq!(
                resolved,
                root.join("escape.bin"),
                "{guest} must clamp to the mount root, landing inside it"
            );
            assert!(
                resolved.starts_with(&root),
                "{guest} escaped the mount root: {resolved:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **UE4/UE5 boot acceptance (SharpEmu daeb-family fix).** An Unreal title
    /// runs with its base directory at `<app>/binaries/<platform>` and addresses
    /// its own content with `../../../`-prefixed paths. Those must resolve back
    /// into `/app0` and actually find the file, or the title enumerates nothing
    /// and never loads its `.pak`s. Until Dawn is UE5.
    #[test]
    fn unreal_project_relative_paths_resolve_back_into_app0() {
        let dir = temp_dir("sandbox-unreal");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        // Lay down the content the engine is looking for.
        let pak_dir = dir.join("Project").join("Content").join("Paks");
        std::fs::create_dir_all(&pak_dir).unwrap();
        std::fs::write(pak_dir.join("pakchunk0.pak"), b"PAK!").unwrap();

        // Exactly the shape UE emits from binaries/<platform>.
        let guest = "/app0/binaries/prospero/../../../Project/Content/Paks/pakchunk0.pak";
        let resolved = vfs
            .resolve_path(guest)
            .expect("UE project-relative path must resolve, not be denied");
        assert_eq!(resolved, pak_dir.join("pakchunk0.pak"));
        assert_eq!(
            std::fs::read(&resolved).unwrap(),
            b"PAK!",
            "the clamped path must reach the real content file"
        );

        // And the same shape must work for a writable open (UE's Saved/ tree),
        // which a blanket "reject any `..`" guard used to refuse outright.
        use open_flags::*;
        let fd = vfs
            .open(
                "/app0/binaries/prospero/../../../Project/Saved/save.dat",
                O_WRONLY | O_CREAT | O_TRUNC,
                0o644,
            )
            .expect("UE-relative writable open must be permitted (clamped, not escaping)");
        assert_eq!(vfs.write(fd, b"SAVE").unwrap(), 4);
        vfs.close(fd).unwrap();
        assert_eq!(
            std::fs::read(dir.join("Project").join("Saved").join("save.dat")).unwrap(),
            b"SAVE",
            "the write must land inside the mount, at the clamped location"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nul_embedded_segment_fails_closed() {
        let dir = temp_dir("sandbox-nul");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        assert!(
            vfs.resolve_path("/app0/bad\0name").is_none(),
            "a NUL-embedded segment must be denied, never handed to the host FS"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn over_long_segment_fails_closed() {
        let dir = temp_dir("sandbox-long");
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);
        let guest = format!("/app0/{}", "a".repeat(40_000));
        assert!(
            vfs.resolve_path(&guest).is_none(),
            "an over-long segment must be denied rather than error the host FS call"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legitimate_in_mount_path_still_resolves() {
        let dir = temp_dir("sandbox-legit");
        std::fs::create_dir_all(dir.join("a").join("b")).unwrap();
        std::fs::write(dir.join("a").join("b").join("c.bin"), b"ok").unwrap();
        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&dir);

        let resolved = vfs
            .resolve_path("/app0/a/b/c.bin")
            .expect("a real nested in-mount file must still resolve");
        assert_eq!(resolved, dir.join("a").join("b").join("c.bin"));
        assert!(resolved.starts_with(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reparse_point_inside_mount_is_denied() {
        // A dump can plant a symlink/junction inside the mount that redirects
        // outside it; lexical containment alone would follow it onto the host FS.
        let outer = temp_dir("sandbox-reparse");
        let inside = outer.join("app0");
        let outside = outer.join("outside");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.bin"), [1, 2, 3]).unwrap();

        // Creating a symlink can require privilege (Windows without Developer
        // Mode); if it fails there is nothing to assert.
        if try_symlink_dir(&outside, &inside.join("link")).is_err() {
            let _ = std::fs::remove_dir_all(&outer);
            return;
        }

        let vfs = VirtualFileSystem::new();
        vfs.set_game_directory(&inside);
        // Sanity: the link genuinely redirects out of the mount.
        assert!(inside.join("link").join("secret.bin").exists());

        assert!(
            vfs.resolve_path("/app0/link/secret.bin").is_none(),
            "a guest path traversing a reparse point out of the mount must be denied"
        );
        let _ = std::fs::remove_dir_all(&outer);
    }
}
