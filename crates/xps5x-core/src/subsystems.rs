//! XPS5X-owned contracts between guest ABI adapters and emulator subsystems.
//!
//! These traits intentionally expose neither Kyty types nor HLE function
//! signatures. HLE translates the guest ABI into these operations; kernel and
//! GPU crates retain ownership of their implementations and lifecycle.

use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitKey {
    pub class: &'static str,
    pub object: u64,
    pub guest_thread: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitOutcome {
    Ready,
    TimedOut,
    Terminating,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeReason {
    Set,
    Clear,
    Cancel,
    Signal,
    Broadcast,
    Deleted,
    SubmissionComplete,
}

pub trait TimeSubsystem: Send + Sync {
    fn monotonic_elapsed(&self) -> Duration;
    fn wall_clock(&self) -> SystemTime;
    fn sleep(&self, duration: Duration);
}

/// Host blocking isolated behind one contract. `ready` is evaluated while the
/// same notification lock used by `wake` is held, preventing the classic
/// check-then-sleep lost-wakeup race.
pub trait WaitSubsystem: Send + Sync {
    fn wait_until(
        &self,
        key: WaitKey,
        timeout: Duration,
        terminating: &dyn Fn() -> bool,
        ready: &mut dyn FnMut() -> bool,
    ) -> WaitOutcome;

    fn wake(&self, key: WaitKey, reason: WakeReason);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventUpdate {
    Set(u64),
    Keep(u64),
    Replace(u64),
}

pub trait EventSubsystem: Send + Sync {
    fn create_event(&self, attributes: u32, initial: u64) -> u64;
    fn delete_event(&self, handle: u64) -> bool;
    fn update_event(&self, handle: u64, update: EventUpdate) -> Option<u64>;
    fn event_bits(&self, handle: u64) -> Option<u64>;
}

pub trait VfsSubsystem: Send + Sync {
    fn open(&self, path: &str, flags: i32, mode: u32) -> std::io::Result<i32>;
    fn read(&self, fd: i32, count: usize) -> std::io::Result<Vec<u8>>;
    fn write(&self, fd: i32, bytes: &[u8]) -> std::io::Result<usize>;
    fn sync(&self, fd: i32) -> std::io::Result<()>;
    fn close(&self, fd: i32) -> std::io::Result<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkMode {
    Offline,
    Host,
}

pub trait NetworkSubsystem: Send + Sync {
    fn mode(&self) -> NetworkMode;
    fn create_socket(&self) -> i32;
    fn socket_exists(&self, fd: i32) -> bool;
}

/// Kernel-backed services carried together through one HLE context.
pub trait KernelSubsystems:
    TimeSubsystem + WaitSubsystem + EventSubsystem + VfsSubsystem + NetworkSubsystem
{
}

impl<T> KernelSubsystems for T where
    T: TimeSubsystem + WaitSubsystem + EventSubsystem + VfsSubsystem + NetworkSubsystem
{
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuQueue {
    Graphics,
    AsyncCompute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuSubmissionStats {
    pub submitted: u64,
    pub completed_draws: u64,
    pub skipped_shaders: u64,
}

pub trait GpuSubmissionSubsystem: Send + Sync {
    fn submit(&self, words: Vec<u32>, queue: GpuQueue);
    fn wait_idle(&self);
    fn stats(&self) -> GpuSubmissionStats;
}
