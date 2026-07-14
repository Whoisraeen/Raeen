//! # XPS5X Kernel
//!
//! High-Level Emulation (HLE) of the PS5's Orbis OS kernel.
//!
//! The PS5 runs a heavily modified FreeBSD kernel (Orbis OS). This crate
//! translates PS5 syscalls to host OS equivalents, manages the emulated
//! virtual address space, implements PS5 threading primitives, and provides
//! a virtual filesystem mapping PS5 paths to host directories.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │              PS5 Game (x86-64 binary)            │
//! │                  ↓ syscall                        │
//! ├──────────────────────────────────────────────────┤
//! │            Syscall Dispatcher                     │
//! │   ┌──────────┬──────────┬───────────┬─────────┐  │
//! │   │  File    │  Memory  │  Thread   │  Other  │  │
//! │   │  Ops     │  Ops     │  Ops      │         │  │
//! │   └──────────┴──────────┴───────────┴─────────┘  │
//! ├──────────────────────────────────────────────────┤
//! │           Host OS (Win32 / POSIX)                │
//! └──────────────────────────────────────────────────┘
//! ```

pub mod filesystem;
pub mod hypervisor;
pub mod memory;
pub mod process;
pub mod syscalls;
pub mod threading;

pub use process::ProcessImage;

use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use xps5x_core::types::ModuleInfo;

/// Captured guest console output (M1-C): everything the guest writes via
/// `printf`/`puts`/`write(1|2, ...)` lands here, byte-for-byte, so the
/// Shell can surface real guest stdout and tests can assert on it. Each
/// write is also mirrored to the host log (`tracing`, target `"guest"`)
/// line-buffered, so `cargo run` output shows guest prints live.
///
/// Bounded: once `MAX_CONSOLE_BYTES` is exceeded the oldest bytes are
/// dropped (keeping the tail) — a guest print loop must not grow host
/// memory without bound.
pub struct Console {
    buf: parking_lot::Mutex<Vec<u8>>,
    /// Bytes of the current, not-yet-newline-terminated line, staged for
    /// the host-log mirror only (the raw `buf` above is unaffected).
    pending_line: parking_lot::Mutex<Vec<u8>>,
}

/// Cap on [`Console`]'s retained output. 4 MiB of text is far more than any
/// M1-era homebrew prints; the tail is what a user debugging a title needs.
const MAX_CONSOLE_BYTES: usize = 4 << 20;

impl Console {
    fn new() -> Self {
        Self {
            buf: parking_lot::Mutex::new(Vec::new()),
            pending_line: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Append raw guest output bytes. Complete lines are mirrored to the
    /// host log (target `"guest"`) as they form.
    pub fn write_bytes(&self, bytes: &[u8]) {
        {
            let mut buf = self.buf.lock();
            buf.extend_from_slice(bytes);
            if buf.len() > MAX_CONSOLE_BYTES {
                let drop_n = buf.len() - MAX_CONSOLE_BYTES;
                buf.drain(..drop_n);
            }
        }

        let mut pending = self.pending_line.lock();
        pending.extend_from_slice(bytes);
        while let Some(nl) = pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = pending.drain(..=nl).collect();
            tracing::info!(target: "guest", "{}", String::from_utf8_lossy(&line[..line.len() - 1]));
        }
        // Bound the pending fragment too (a guest spewing without newlines).
        if pending.len() > MAX_CONSOLE_BYTES {
            let drop_n = pending.len() - MAX_CONSOLE_BYTES;
            pending.drain(..drop_n);
        }
    }

    /// The captured output so far, lossily decoded as UTF-8.
    pub fn contents(&self) -> String {
        String::from_utf8_lossy(&self.buf.lock()).into_owned()
    }

    /// Total captured bytes (post-truncation).
    pub fn len(&self) -> usize {
        self.buf.lock().len()
    }

    /// Whether nothing has been captured yet.
    pub fn is_empty(&self) -> bool {
        self.buf.lock().is_empty()
    }
}

/// The emulated PS5 kernel state.
///
/// Holds all kernel-level state: memory map, file descriptors,
/// threads, loaded modules, and system configuration.
pub struct OrbisKernel {
    /// Captured guest console output (stdout/stderr + libc print family).
    pub console: Console,
    /// Virtual memory manager.
    pub memory: Arc<memory::VirtualMemoryManager>,
    /// Thread manager.
    pub threads: Arc<threading::ThreadManager>,
    /// Virtual filesystem.
    pub filesystem: Arc<filesystem::VirtualFileSystem>,
    /// Loaded modules (.sprx / .elf).
    pub modules: DashMap<u32, ModuleInfo>,
    /// Next module ID to assign.
    next_module_id: RwLock<u32>,
    /// Syscall statistics (for debugging).
    pub syscall_stats: DashMap<u64, u64>,
}

impl OrbisKernel {
    /// Create a new kernel instance with default configuration.
    pub fn new() -> Self {
        tracing::info!("Initializing Orbis kernel HLE");
        Self {
            console: Console::new(),
            memory: Arc::new(memory::VirtualMemoryManager::new()),
            threads: Arc::new(threading::ThreadManager::new()),
            filesystem: Arc::new(filesystem::VirtualFileSystem::new()),
            modules: DashMap::new(),
            next_module_id: RwLock::new(1),
            syscall_stats: DashMap::new(),
        }
    }

    /// Register a loaded module with the kernel.
    pub fn register_module(&self, mut info: ModuleInfo) -> u32 {
        let mut next_id = self.next_module_id.write();
        let id = *next_id;
        info.id = id;
        *next_id += 1;
        tracing::info!(
            "Registered module: id={}, name='{}', base={:#x}, size={:#x}",
            id,
            info.name,
            info.base_address,
            info.size
        );
        self.modules.insert(id, info);
        id
    }

    /// Look up a module by name.
    pub fn find_module(&self, name: &str) -> Option<ModuleInfo> {
        self.modules
            .iter()
            .find(|entry| entry.value().name == name)
            .map(|entry| entry.value().clone())
    }

    /// Dispatch a syscall.
    pub fn dispatch_syscall(
        &self,
        number: u64,
        args: &[u64],
    ) -> Result<u64, xps5x_core::error::KernelError> {
        // Track syscall statistics.
        self.syscall_stats
            .entry(number)
            .and_modify(|count| *count += 1)
            .or_insert(1);

        syscalls::dispatch(self, number, args)
    }
}

impl Default for OrbisKernel {
    fn default() -> Self {
        Self::new()
    }
}
