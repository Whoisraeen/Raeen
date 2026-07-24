//! Thread manager for the emulated PS5.
//!
//! Maps PS5 threads to host OS threads, emulates thread priorities
//! and affinity, and implements synchronization primitives.

pub mod sync;

use parking_lot::RwLock;
use raeen_core::error::KernelError;
use raeen_core::types::Tid;
use std::collections::HashMap;
use tracing::{debug, info};

/// Information about an emulated PS5 thread.
#[derive(Debug, Clone)]
pub struct ThreadInfo {
    /// PS5 thread ID.
    pub tid: Tid,
    /// Thread name (for debugging).
    pub name: String,
    /// Thread priority (PS5 range: 0-767, lower = higher priority).
    pub priority: i32,
    /// CPU affinity mask (8 cores).
    pub affinity: u8,
    /// Thread state.
    pub state: ThreadState,
    /// Entry point address.
    pub entry_point: u64,
    /// Stack base address.
    pub stack_base: u64,
    /// Stack size.
    pub stack_size: u64,
    /// TLS (Thread-Local Storage) base address.
    pub tls_base: u64,
}

/// Thread execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
    /// Thread is ready to run.
    Ready,
    /// Thread is currently executing.
    Running,
    /// Thread is blocked (waiting on sync primitive).
    Blocked,
    /// Thread is sleeping.
    Sleeping,
    /// Thread has exited.
    Exited,
}

/// Manages all emulated PS5 threads.
pub struct ThreadManager {
    /// Active threads, keyed by TID.
    threads: RwLock<HashMap<Tid, ThreadInfo>>,
    /// Next TID to assign.
    next_tid: RwLock<Tid>,
}

impl Default for ThreadManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadManager {
    /// Create a new thread manager.
    pub fn new() -> Self {
        info!("Initializing thread manager");
        Self {
            threads: RwLock::new(HashMap::new()),
            next_tid: RwLock::new(1),
        }
    }

    /// Create a new emulated thread.
    ///
    /// In a full implementation, this would spawn a host OS thread.
    pub fn create_thread(&self, param_addr: u64) -> Result<Tid, KernelError> {
        let mut next = self.next_tid.write();
        let tid = *next;
        *next += 1;

        let info = ThreadInfo {
            tid,
            name: format!("thread_{}", tid),
            priority: 700,  // Default PS5 priority.
            affinity: 0xFF, // All 8 cores.
            state: ThreadState::Ready,
            entry_point: param_addr,
            stack_base: 0,
            stack_size: 0x80000, // 512 KiB default stack.
            tls_base: 0,
        };

        debug!("Created thread: tid={}, entry={:#x}", tid, param_addr);
        self.threads.write().insert(tid, info);

        Ok(tid)
    }

    /// Get the main thread (TID 1).
    pub fn main_thread(&self) -> Option<ThreadInfo> {
        self.threads.read().get(&1).cloned()
    }

    /// Set thread state.
    pub fn set_state(&self, tid: Tid, state: ThreadState) {
        if let Some(thread) = self.threads.write().get_mut(&tid) {
            debug!("Thread {} state: {:?} -> {:?}", tid, thread.state, state);
            thread.state = state;
        }
    }

    /// Get all active thread IDs.
    pub fn active_threads(&self) -> Vec<Tid> {
        self.threads
            .read()
            .iter()
            .filter(|(_, t)| t.state != ThreadState::Exited)
            .map(|(tid, _)| *tid)
            .collect()
    }

    /// Get thread count.
    pub fn count(&self) -> usize {
        self.threads.read().len()
    }
}
