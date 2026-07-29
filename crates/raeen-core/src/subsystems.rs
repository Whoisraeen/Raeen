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

/// How long a single host park inside a sliced guest wait may last.
///
/// Every blocking kernel primitive (event flag, equeue, semaphore) parks in
/// bounded slices rather than for the guest's whole interval, so process
/// teardown is noticed promptly and an infinite wait stays escapable. The slice
/// length is the *internal* bound; the guest's own deadline still wins when it
/// is nearer.
///
/// A `None` deadline is an indefinite wait and always parks a full slice. A
/// finite deadline parks `min(remaining, slice)`, so the wait neither overshoots
/// the interval the guest asked for nor blocks past teardown.
///
/// Pure so the arithmetic is testable against a synthetic clock: the wall-clock
/// version of this test raced under parallel load. Shared by all three
/// primitives, which previously each carried their own copy — and one copy
/// disagreeing with the others is how a fabricated `ETIMEDOUT` gets shipped.
#[must_use]
pub fn park_slice(
    deadline: Option<std::time::Instant>,
    now: std::time::Instant,
    slice: Duration,
) -> Duration {
    deadline.map_or(slice, |dl| dl.saturating_duration_since(now).min(slice))
}

/// Whether a park that timed out is the **guest's** timeout, or merely an
/// internal slice expiring.
///
/// This is the whole contract the sliced waits had to be corrected to honour.
/// Reporting a slice expiry as the guest's timeout is a fabricated
/// `ETIMEDOUT`: Dragon Ball's AGC workers took one after 50 ms and entered the
/// title's fatal-reporting path before their first submission. So:
///
/// * a `None` timeout is **never** a guest timeout, however many slices elapse;
///   and
/// * a finite timeout is a guest timeout only once the real deadline has
///   arrived, not at the first internal slice boundary.
#[must_use]
pub fn guest_deadline_reached(
    deadline: Option<std::time::Instant>,
    now: std::time::Instant,
) -> bool {
    deadline.is_some_and(|dl| now >= dl)
}

#[cfg(test)]
mod wait_slice_tests {
    use super::{guest_deadline_reached, park_slice};
    use std::time::{Duration, Instant};

    /// The slice/deadline decision, on a synthetic clock. This is the shared
    /// arithmetic behind every kernel wait, so its edges are asserted here once
    /// instead of three times with three chances to drift.
    #[test]
    fn park_slice_prefers_the_nearer_of_slice_and_guest_deadline() {
        const SLICE: Duration = Duration::from_millis(50);
        let t0 = Instant::now();

        // Indefinite: always a full slice, no matter how much time passes.
        assert_eq!(park_slice(None, t0, SLICE), SLICE);
        assert_eq!(
            park_slice(None, t0 + Duration::from_secs(3600), SLICE),
            SLICE
        );

        // Finite and far away: the slice bounds it.
        let far = t0 + Duration::from_millis(200);
        assert_eq!(park_slice(Some(far), t0, SLICE), SLICE);

        // Finite and nearer than the slice: the guest's remaining time wins, so
        // the wait cannot overshoot what the caller asked for.
        assert_eq!(
            park_slice(Some(far), t0 + Duration::from_millis(180), SLICE),
            Duration::from_millis(20)
        );
        let near = t0 + Duration::from_millis(5);
        assert_eq!(park_slice(Some(near), t0, SLICE), Duration::from_millis(5));

        // Already expired: zero, never a wrapped/huge duration.
        assert_eq!(park_slice(Some(near), near, SLICE), Duration::ZERO);
        assert_eq!(
            park_slice(Some(near), t0 + Duration::from_secs(10), SLICE),
            Duration::ZERO
        );

        // A zero slice degenerates to a poll rather than an unbounded park.
        assert_eq!(park_slice(None, t0, Duration::ZERO), Duration::ZERO);
    }

    /// An internal slice expiry must never be reported as the guest's timeout.
    #[test]
    fn only_a_real_deadline_counts_as_the_guests_timeout() {
        let t0 = Instant::now();

        // Indefinite waits never time out — this is the fabricated-ETIMEDOUT bug.
        assert!(!guest_deadline_reached(None, t0));
        assert!(!guest_deadline_reached(
            None,
            t0 + Duration::from_secs(86_400)
        ));

        let deadline = t0 + Duration::from_millis(200);
        // Before the deadline — including well past the first 50 ms slice.
        assert!(!guest_deadline_reached(Some(deadline), t0));
        assert!(!guest_deadline_reached(
            Some(deadline),
            t0 + Duration::from_millis(50)
        ));
        assert!(!guest_deadline_reached(
            Some(deadline),
            t0 + Duration::from_millis(199)
        ));
        // At and after it.
        assert!(guest_deadline_reached(Some(deadline), deadline));
        assert!(guest_deadline_reached(
            Some(deadline),
            t0 + Duration::from_millis(201)
        ));
    }
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
    /// Fill caller-owned storage from `fd`, advancing its cursor.
    ///
    /// Backends should override this to avoid allocating an intermediate
    /// `Vec`. The default preserves compatibility with small test doubles.
    fn read_into(&self, fd: i32, out: &mut [u8]) -> std::io::Result<usize> {
        let bytes = self.read(fd, out.len())?;
        let count = bytes.len().min(out.len());
        out[..count].copy_from_slice(&bytes[..count]);
        Ok(count)
    }
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
