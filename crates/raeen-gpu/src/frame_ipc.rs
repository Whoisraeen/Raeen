//! Crash-isolated runner -> Shell frame transport.
//!
//! The retail-title runner lives in a child process so a guest or driver crash
//! cannot take down the Shell. Process-local [`crate::AgcGpuSession`] state
//! therefore cannot be observed by the Shell directly. This module provides a
//! single-producer/single-consumer, pagefile-backed shared-memory slot for the
//! latest complete RGBA8 frame.
//!
//! A sequence lock keeps the hot path to one frame copy in the child and one
//! copy in the Shell. If the child begins replacing a frame while the Shell is
//! reading it, the Shell discards that copy and keeps its cached last complete
//! frame. It never uploads a torn frame.

use crate::RenderedImage;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

/// Environment variable carrying the unique Windows mapping name into the
/// isolated runner.
pub const FRAME_IPC_ENV: &str = "RAEEN_FRAME_IPC";

/// A complete frame copied out of the isolated runner's shared-memory slot.
#[derive(Clone)]
pub struct RemoteFrame {
    pub epoch: u64,
    pub image: Arc<RenderedImage>,
}

#[cfg(target_os = "windows")]
mod platform {
    use super::*;
    use std::io;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
        OpenFileMappingW, PAGE_READWRITE, UnmapViewOfFile,
    };

    const MAGIC: u32 = u32::from_le_bytes(*b"RAEF");
    const VERSION: u32 = 1;
    const HEADER_BYTES: usize = 64;
    // Covers an 8K RGBA8 frame (132.7 MiB) with room to spare. The mapping is
    // pagefile-backed virtual address space; only touched pages consume commit.
    const PIXEL_CAPACITY: usize = 256 * 1024 * 1024;
    const MAPPING_BYTES: usize = HEADER_BYTES + PIXEL_CAPACITY;

    const MAGIC_OFFSET: usize = 0;
    const VERSION_OFFSET: usize = 4;
    const SEQUENCE_OFFSET: usize = 8;
    const WIDTH_OFFSET: usize = 16;
    const HEIGHT_OFFSET: usize = 20;
    const LENGTH_OFFSET: usize = 24;
    const BPP_OFFSET: usize = 32;

    struct Mapping {
        handle: HANDLE,
        view: *mut u8,
    }

    // SAFETY: the mapping address remains valid until Drop. All shared header
    // synchronization is atomic, payload publication is guarded by the
    // sequence lock, and callers receive owned copies of payload bytes.
    unsafe impl Send for Mapping {}
    // SAFETY: see Send. There is one publisher process and receiver reads are
    // serialized by FrameIpcReceiver::cache.
    unsafe impl Sync for Mapping {}

    impl Mapping {
        fn create(name: &str) -> io::Result<Self> {
            let wide = wide_name(name);
            let size = MAPPING_BYTES as u64;
            // SAFETY: INVALID_HANDLE_VALUE requests pagefile backing; the
            // security-attributes pointer is null; `wide` is NUL-terminated.
            let handle = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    core::ptr::null(),
                    PAGE_READWRITE,
                    (size >> 32) as u32,
                    size as u32,
                    wide.as_ptr(),
                )
            };
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            match Self::map_handle(handle) {
                Ok(mapping) => Ok(mapping),
                Err(error) => {
                    // SAFETY: `handle` was returned by CreateFileMappingW and
                    // has not been transferred to a Mapping.
                    unsafe { CloseHandle(handle) };
                    Err(error)
                }
            }
        }

        fn open(name: &str) -> io::Result<Self> {
            let wide = wide_name(name);
            // SAFETY: `wide` is NUL-terminated and names the mapping created by
            // the Shell. The child needs read/write access for the seqlock.
            let handle = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, wide.as_ptr()) };
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            match Self::map_handle(handle) {
                Ok(mapping) => Ok(mapping),
                Err(error) => {
                    // SAFETY: `handle` was returned by OpenFileMappingW and has
                    // not been transferred to a Mapping.
                    unsafe { CloseHandle(handle) };
                    Err(error)
                }
            }
        }

        fn map_handle(handle: HANDLE) -> io::Result<Self> {
            // SAFETY: `handle` is a valid file-mapping handle and the requested
            // view length is the size used by both protocol peers.
            let mapped = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, MAPPING_BYTES) };
            if mapped.Value.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                handle,
                view: mapped.Value.cast(),
            })
        }

        fn atomic_u32(&self, offset: usize) -> &AtomicU32 {
            debug_assert!(offset.is_multiple_of(core::mem::align_of::<AtomicU32>()));
            // SAFETY: every header offset is aligned, inside the live mapping,
            // and initialized before another process opens it.
            unsafe { &*self.view.add(offset).cast::<AtomicU32>() }
        }

        fn atomic_u64(&self, offset: usize) -> &AtomicU64 {
            debug_assert!(offset.is_multiple_of(core::mem::align_of::<AtomicU64>()));
            // SAFETY: every header offset is aligned, inside the live mapping,
            // and initialized before another process opens it.
            unsafe { &*self.view.add(offset).cast::<AtomicU64>() }
        }

        fn pixels(&self) -> *mut u8 {
            // SAFETY: HEADER_BYTES is inside the live MAPPING_BYTES view.
            unsafe { self.view.add(HEADER_BYTES) }
        }
    }

    impl Drop for Mapping {
        fn drop(&mut self) {
            // SAFETY: both resources were acquired together by Mapping and are
            // released exactly once here after all references are gone.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.view.cast(),
                });
                CloseHandle(self.handle);
            }
        }
    }

    fn wide_name(name: &str) -> Vec<u16> {
        name.encode_utf16().chain(core::iter::once(0)).collect()
    }

    /// Shell-owned receiver and mapping lifetime.
    pub struct FrameIpcReceiver {
        name: String,
        mapping: Mapping,
        cache: Mutex<Option<RemoteFrame>>,
    }

    impl FrameIpcReceiver {
        pub fn create() -> io::Result<Self> {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let name = format!("Local\\RaeenFrame-{}-{id}", std::process::id());
            let mapping = Mapping::create(&name)?;
            mapping
                .atomic_u32(MAGIC_OFFSET)
                .store(MAGIC, Ordering::Relaxed);
            mapping
                .atomic_u32(VERSION_OFFSET)
                .store(VERSION, Ordering::Relaxed);
            mapping
                .atomic_u64(SEQUENCE_OFFSET)
                .store(0, Ordering::Release);
            Ok(Self {
                name,
                mapping,
                cache: Mutex::new(None),
            })
        }

        pub fn name(&self) -> &str {
            &self.name
        }

        /// Return the newest complete frame, or the cached prior frame if the
        /// producer is currently replacing the shared slot.
        pub fn latest(&self) -> Option<RemoteFrame> {
            let mut cache = self.cache.lock();
            let first = self
                .mapping
                .atomic_u64(SEQUENCE_OFFSET)
                .load(Ordering::Acquire);
            if first == 0 || first & 1 != 0 {
                return cache.clone();
            }
            if cache.as_ref().is_some_and(|frame| frame.epoch == first) {
                return cache.clone();
            }
            if self
                .mapping
                .atomic_u32(MAGIC_OFFSET)
                .load(Ordering::Relaxed)
                != MAGIC
                || self
                    .mapping
                    .atomic_u32(VERSION_OFFSET)
                    .load(Ordering::Relaxed)
                    != VERSION
            {
                return cache.clone();
            }

            let width = self
                .mapping
                .atomic_u32(WIDTH_OFFSET)
                .load(Ordering::Relaxed);
            let height = self
                .mapping
                .atomic_u32(HEIGHT_OFFSET)
                .load(Ordering::Relaxed);
            let len = self
                .mapping
                .atomic_u64(LENGTH_OFFSET)
                .load(Ordering::Relaxed) as usize;
            let bpp = self.mapping.atomic_u32(BPP_OFFSET).load(Ordering::Relaxed);
            let expected = (width as usize)
                .checked_mul(height as usize)
                .and_then(|pixels| pixels.checked_mul(bpp as usize));
            if width == 0
                || height == 0
                || bpp != 4
                || len == 0
                || len > PIXEL_CAPACITY
                || expected != Some(len)
            {
                return cache.clone();
            }

            let mut pixels = vec![0; len];
            // SAFETY: the validated length is within PIXEL_CAPACITY, both
            // buffers are valid for `len` bytes and do not overlap. The
            // sequence re-check below rejects a copy concurrent with a write.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.mapping.pixels().cast_const(),
                    pixels.as_mut_ptr(),
                    len,
                );
            }
            std::sync::atomic::fence(Ordering::Acquire);
            let second = self
                .mapping
                .atomic_u64(SEQUENCE_OFFSET)
                .load(Ordering::Acquire);
            if first != second || second & 1 != 0 {
                return cache.clone();
            }

            let frame = RemoteFrame {
                epoch: second,
                image: Arc::new(RenderedImage {
                    width,
                    height,
                    pixels,
                    bytes_per_pixel: bpp,
                }),
            };
            if cache.is_none() {
                tracing::info!(
                    width,
                    height,
                    bytes = len,
                    epoch = second,
                    "Shell received first complete frame from isolated runner"
                );
            }
            *cache = Some(frame.clone());
            Some(frame)
        }
    }

    /// Child-owned publisher.
    pub(crate) struct FrameIpcPublisher {
        mapping: Mapping,
    }

    impl FrameIpcPublisher {
        pub(crate) fn open_from_env() -> Option<Self> {
            let name = std::env::var(FRAME_IPC_ENV).ok()?;
            match Mapping::open(&name) {
                Ok(mapping) => {
                    tracing::info!(mapping = %name, "isolated runner connected to Shell frame IPC");
                    Some(Self { mapping })
                }
                Err(error) => {
                    tracing::error!(%error, mapping = %name, "cannot open Shell frame IPC mapping");
                    None
                }
            }
        }

        pub(crate) fn publish(&self, image: &RenderedImage) {
            if image.bytes_per_pixel != 4
                || image.pixels.is_empty()
                || image.pixels.len() > PIXEL_CAPACITY
                || (image.width as usize)
                    .checked_mul(image.height as usize)
                    .and_then(|pixels| pixels.checked_mul(4))
                    != Some(image.pixels.len())
            {
                tracing::warn!(
                    width = image.width,
                    height = image.height,
                    bytes_per_pixel = image.bytes_per_pixel,
                    bytes = image.pixels.len(),
                    "complete frame does not fit the runner frame IPC contract"
                );
                return;
            }

            let sequence = self
                .mapping
                .atomic_u64(SEQUENCE_OFFSET)
                .load(Ordering::Relaxed);
            let writing = if sequence & 1 == 0 {
                sequence.wrapping_add(1)
            } else {
                sequence.wrapping_add(2)
            };
            self.mapping
                .atomic_u64(SEQUENCE_OFFSET)
                .store(writing, Ordering::Release);
            self.mapping
                .atomic_u32(WIDTH_OFFSET)
                .store(image.width, Ordering::Relaxed);
            self.mapping
                .atomic_u32(HEIGHT_OFFSET)
                .store(image.height, Ordering::Relaxed);
            self.mapping
                .atomic_u64(LENGTH_OFFSET)
                .store(image.pixels.len() as u64, Ordering::Relaxed);
            self.mapping
                .atomic_u32(BPP_OFFSET)
                .store(image.bytes_per_pixel, Ordering::Relaxed);
            // SAFETY: the image length was validated against PIXEL_CAPACITY,
            // both buffers are live and non-overlapping. The odd sequence keeps
            // the receiver from accepting this slot until the release below.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    image.pixels.as_ptr(),
                    self.mapping.pixels(),
                    image.pixels.len(),
                );
            }
            std::sync::atomic::fence(Ordering::Release);
            self.mapping
                .atomic_u64(SEQUENCE_OFFSET)
                .store(writing.wrapping_add(1), Ordering::Release);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn complete_frame_round_trips_between_runner_and_shell_mapping() {
            let receiver = FrameIpcReceiver::create().expect("create mapping");
            let publisher = FrameIpcPublisher {
                mapping: Mapping::open(receiver.name()).expect("open child mapping"),
            };
            let image = RenderedImage {
                width: 2,
                height: 2,
                pixels: vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 8, 7, 255],
                bytes_per_pixel: 4,
            };
            publisher.publish(&image);
            let remote = receiver.latest().expect("published frame");
            assert_eq!(remote.image.width, 2);
            assert_eq!(remote.image.height, 2);
            assert_eq!(remote.image.pixels, image.pixels);
            assert_eq!(receiver.latest().unwrap().epoch, remote.epoch);
        }

        #[test]
        fn invalid_frame_never_replaces_the_last_complete_frame() {
            let receiver = FrameIpcReceiver::create().expect("create mapping");
            let publisher = FrameIpcPublisher {
                mapping: Mapping::open(receiver.name()).expect("open child mapping"),
            };
            publisher.publish(&RenderedImage {
                width: 1,
                height: 1,
                pixels: vec![1, 2, 3, 255],
                bytes_per_pixel: 4,
            });
            let accepted = receiver.latest().expect("first frame");
            publisher.publish(&RenderedImage {
                width: 2,
                height: 2,
                pixels: vec![0; 3],
                bytes_per_pixel: 4,
            });
            let kept = receiver.latest().expect("cached frame");
            assert_eq!(kept.epoch, accepted.epoch);
            assert_eq!(kept.image.pixels, accepted.image.pixels);
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    use super::*;
    use std::io;

    pub struct FrameIpcReceiver;

    impl FrameIpcReceiver {
        pub fn create() -> io::Result<Self> {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "runner frame IPC is Windows-only",
            ))
        }

        pub fn name(&self) -> &str {
            ""
        }

        pub fn latest(&self) -> Option<RemoteFrame> {
            None
        }
    }

    pub(crate) struct FrameIpcPublisher;

    impl FrameIpcPublisher {
        pub(crate) fn open_from_env() -> Option<Self> {
            None
        }

        pub(crate) fn publish(&self, _image: &RenderedImage) {}
    }
}

pub(crate) use platform::FrameIpcPublisher;
pub use platform::FrameIpcReceiver;

fn active_receiver() -> &'static RwLock<Option<Arc<FrameIpcReceiver>>> {
    static ACTIVE: std::sync::OnceLock<RwLock<Option<Arc<FrameIpcReceiver>>>> =
        std::sync::OnceLock::new();
    ACTIVE.get_or_init(|| RwLock::new(None))
}

/// Install the receiver for the Shell's one active isolated title.
pub fn install_receiver(receiver: Arc<FrameIpcReceiver>) {
    *active_receiver().write() = Some(receiver);
}

/// Clear `receiver` without allowing an older exiting worker to remove a newer
/// title's bridge.
pub fn clear_receiver(receiver: &Arc<FrameIpcReceiver>) {
    let mut active = active_receiver().write();
    if active
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, receiver))
    {
        *active = None;
    }
}

/// Latest complete child-runner frame, cached so an unchanged frame costs only
/// an Arc bump on each Shell repaint.
pub fn latest_remote_frame() -> Option<RemoteFrame> {
    active_receiver()
        .read()
        .as_ref()
        .and_then(|receiver| receiver.latest())
}
