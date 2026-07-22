//! Raeen-owned contracts between guest ABI adapters and emulator subsystems.
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
    /// Create a process-owned event, or report resource exhaustion. The
    /// fallible return is part of the boundary: guest-controlled handle tables
    /// must never imply an unbounded host allocation.
    fn create_event(&self, attributes: u32, initial: u64) -> Option<u64>;
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
    /// Create a process-owned socket, or report descriptor-table exhaustion.
    fn create_socket(&self) -> Option<i32>;
    fn socket_exists(&self, fd: i32) -> bool;
    fn close_socket(&self, fd: i32) -> bool;
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

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderSharp {
    pub raw: u16,
}

impl ShaderSharp {
    #[must_use]
    pub const fn new(offset_dw: u16, size: u16) -> Self {
        Self {
            raw: (offset_dw & 0x7fff) | ((size & 1) << 15),
        }
    }

    #[must_use]
    pub const fn offset_dw(self) -> u16 {
        self.raw & 0x7fff
    }

    #[must_use]
    pub const fn size(self) -> u16 {
        self.raw >> 15
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderUserData {
    pub direct_resource_offset: Vec<u16>,
    pub sharp_resource_offset: [Vec<ShaderSharp>; 4],
    pub eud_size_dw: u16,
    pub srt_size_dw: u16,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderSemantic {
    pub raw: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShaderMappedData {
    pub user_data: Option<ShaderUserData>,
    pub input_semantics: Vec<ShaderSemantic>,
}

/// Layout of a guest display buffer for present-from-guest-memory (M3).
///
/// A 2D title that CPU-draws pixels into its display buffer and flips never
/// routes those pixels through a GPU draw, so the flipped buffer is absent from
/// the GPU's render-target map. This descriptor — populated by the HLE from the
/// VideoOut buffer attribute the title registered
/// (`sceVideoOutRegisterBuffers2`) — lets the GPU present those bytes directly.
///
/// SharpEmu `VulkanVideoPresenter.cs:1643-1660` (`GuestImageWantsInitialData`):
/// PS5 render targets alias CPU-visible memory; first-use images are seeded
/// from guest memory, which is how CPU-written pixels become visible without
/// any GPU draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanoutDescriptor {
    /// Visible width in pixels.
    pub width: u32,
    /// Visible height in pixels.
    pub height: u32,
    /// Row stride in pixels (defaults to `width` for a tightly-packed linear
    /// buffer when the attribute carries no separate pitch).
    pub pitch_pixels: u32,
    /// Raw `SceVideoOutPixelFormat`.
    pub pixel_format: u64,
    /// Raw `SceVideoOutTilingMode` (0 = tiled, 1 = linear).
    pub tiling_mode: u32,
}

pub trait GpuSubmissionSubsystem: Send + Sync {
    fn submit(&self, words: Vec<u32>, queue: GpuQueue);
    fn map_shader_metadata(&self, code_address: u64, data: ShaderMappedData);
    /// Present the guest display buffer at `address` the title flipped to. When
    /// `descriptor` is provided and no GPU-drawn target exists at `address`, the
    /// backend may read the guest bytes there as pixels (CPU-drawn 2D, M3).
    fn present_scanout(&self, address: u64, descriptor: Option<ScanoutDescriptor>);
    fn wait_idle(&self);
    fn stats(&self) -> GpuSubmissionStats;
    /// `sceSystemServiceHideSplashScreen`: the title says its own rendering is
    /// ready, so the system boot splash must come down. Default no-op for
    /// backends with no presentation surface.
    fn hide_splash(&self) {}
}
