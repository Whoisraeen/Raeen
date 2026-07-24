//! Synchronization primitives for emulated PS5 threads.
//!
//! Implements PS5-specific sync objects: event flags, semaphores,
//! mutexes, and read-write locks.

use parking_lot::Mutex;
use raeen_core::types::KernelHandle;
use std::collections::HashMap;
use tracing::debug;

/// Emulated PS5 event flag.
#[derive(Debug)]
pub struct EventFlag {
    pub handle: KernelHandle,
    pub name: String,
    pub bits: u64,
    pub waiters: Vec<u64>, // Thread IDs waiting on this event.
}

/// Emulated PS5 semaphore.
#[derive(Debug)]
pub struct Semaphore {
    pub handle: KernelHandle,
    pub name: String,
    pub count: i32,
    pub max_count: i32,
}

/// Registry of all synchronization objects.
pub struct SyncRegistry {
    event_flags: Mutex<HashMap<KernelHandle, EventFlag>>,
    semaphores: Mutex<HashMap<KernelHandle, Semaphore>>,
    next_handle: Mutex<KernelHandle>,
}

impl SyncRegistry {
    pub fn new() -> Self {
        Self {
            event_flags: Mutex::new(HashMap::new()),
            semaphores: Mutex::new(HashMap::new()),
            next_handle: Mutex::new(1),
        }
    }

    /// Create a new event flag.
    pub fn create_event_flag(&self, name: &str, init_pattern: u64) -> KernelHandle {
        let mut next = self.next_handle.lock();
        let handle = *next;
        *next += 1;

        let ef = EventFlag {
            handle,
            name: name.to_string(),
            bits: init_pattern,
            waiters: Vec::new(),
        };

        debug!("Created event flag: handle={}, name='{}'", handle, name);
        self.event_flags.lock().insert(handle, ef);
        handle
    }

    /// Create a new semaphore.
    pub fn create_semaphore(&self, name: &str, init_count: i32, max_count: i32) -> KernelHandle {
        let mut next = self.next_handle.lock();
        let handle = *next;
        *next += 1;

        let sem = Semaphore {
            handle,
            name: name.to_string(),
            count: init_count,
            max_count,
        };

        debug!(
            "Created semaphore: handle={}, name='{}', count={}/{}",
            handle, name, init_count, max_count
        );
        self.semaphores.lock().insert(handle, sem);
        handle
    }
}

impl Default for SyncRegistry {
    fn default() -> Self {
        Self::new()
    }
}
