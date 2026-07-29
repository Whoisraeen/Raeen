//! Crash-isolated runner -> Shell frame transport.
//!
//! The retail-title runner lives in a child process so a guest or driver crash
//! cannot take down the Shell. Process-local [`crate::AgcGpuSession`] state
//! therefore cannot be observed by the Shell directly. This module provides a
//! pagefile-backed shared-memory slots for the latest complete RGBA8 frame and
//! the Shell's latest 12-byte Orbis pad snapshot.
//!
//! A sequence lock plus two shared slots keeps the hot path to one frame copy
//! in the child and one copy in the Shell. The child always writes the slot
//! opposite the previously-published frame, so a normal Shell upload can run
//! concurrently with the next guest frame instead of racing one hot slot. If
//! the child laps the Shell, the sequence re-check discards that copy and keeps
//! the cached last complete frame. It never uploads a torn frame.

use crate::{RenderedImage, agc_exec::PresentTiming};
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
    pub timing: PresentTiming,
}

/// The guest runner's diagnostics state, as the Shell sees it across the
/// process boundary.
///
/// The Shell's F10 console is fed by a process-local tracing ring and the guest
/// runs in a separate child, so this struct is the *only* path by which the
/// console can say anything about a running or stalled title.
///
/// Every field is honest about not knowing: [`Self::default`] — which is also
/// what a never-published or mid-write block reads as — carries `seq == 0` and
/// `None` stages, which a renderer must show as "no report yet", never as
/// "stage 0" or "0 blockers".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionStatus {
    /// Publication counter. `0` = the child has never published.
    pub seq: u64,
    /// Furthest `raeen_core::frame_path::Stage` index reached, if any.
    pub stage: Option<u8>,
    /// Furthest `raeen_core::frame_path::Phase` index reached, if any.
    pub phase: Option<u8>,
    /// Distinct blockers retained by the child's table.
    pub distinct_blockers: u64,
    /// Occurrences across every retained blocker.
    pub total_events: u64,
    /// Distinct blockers the child dropped at its per-category cap.
    pub dropped_distinct: u64,
    /// Guest-process CPU time. Read against `wall_ms`, this separates a title
    /// parked in a wait (CPU much less than wall) from one spinning — the
    /// distinction that split the measured silent-zero-frame cluster into two
    /// different bugs.
    pub cpu_ms: u64,
    /// Guest-process wall-clock time.
    pub wall_ms: u64,
    /// Human-readable digest: the frame-path summary and the ranked blocker
    /// lines, truncated to the channel's byte budget on a `char` boundary.
    pub digest: String,
}

impl SessionStatus {
    /// Whether the child has ever published. `false` means "no report yet",
    /// which is not the same as a report saying nothing happened.
    #[must_use]
    pub fn published(&self) -> bool {
        self.seq > 0
    }
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
    // v6: the header was padded from 104 B to a full page so every pixel slot
    // starts page-aligned (see `HEADER_BYTES`). Field offsets are unchanged —
    // only the padding after them grew — but a v5 peer would read pixels at
    // the wrong offset, so the version gate must reject it.
    const VERSION: u32 = 6;
    // Input occupies bytes 48..60, the timing counters run to 104, and the
    // child → Shell rumble word sits at 104..112; the rest of this page is
    // padding. (Adding the rumble word did NOT bump VERSION: it lives in
    // previously-zeroed padding, no existing field moved, and a peer without
    // it simply reads/writes 0 — which decodes as "no rumble source" — so
    // mismatched builds degrade to "no rumble" instead of "no video".)
    //
    // A FULL PAGE, not the 104 B the fields need, so that each pixel slot
    // begins on a page boundary. `VK_EXT_external_memory_host` can only import
    // a host pointer aligned to `minImportedHostPointerAlignment` (measured:
    // 4096 B on a Radeon 760M), and phase 1 of the GPU-resident present plan
    // imports a slot so the GPU can copy the finished frame straight into it
    // instead of going image -> staging buffer -> Vec -> memcpy here. At the
    // old 104 B header every slot started at 104 mod 4096 and was therefore
    // unimportable. `PIXEL_CAPACITY` is itself a multiple of the page size, so
    // aligning the header aligns every slot.
    const HEADER_BYTES: usize = 4096;
    // One slot covers an 8K RGBA8 frame (126.6 MiB) with room to spare. Keeping
    // each slot tight makes the two-slot mapping only slightly larger than the
    // old single 256-MiB slot.
    const PIXEL_CAPACITY: usize = 136 * 1024 * 1024;
    const FRAME_SLOTS: usize = 2;
    const MAPPING_BYTES: usize = HEADER_BYTES + PIXEL_CAPACITY * FRAME_SLOTS;

    /// Host-pointer alignment every pixel slot satisfies. Kept as a named
    /// constant so the import path can assert against it and so a future
    /// header change that breaks alignment fails the test below, not a driver
    /// call at runtime.
    // Read by the alignment test today; the phase-1 host-pointer import path
    // (gpu-resident present plan) is the non-test consumer.
    #[allow(dead_code)]
    pub(crate) const SLOT_ALIGNMENT: usize = 4096;

    const MAGIC_OFFSET: usize = 0;
    const VERSION_OFFSET: usize = 4;
    const SEQUENCE_OFFSET: usize = 8;
    const WIDTH_OFFSET: usize = 16;
    const HEIGHT_OFFSET: usize = 20;
    const LENGTH_OFFSET: usize = 24;
    const BPP_OFFSET: usize = 32;
    const INPUT_SEQUENCE_OFFSET: usize = 40;
    const INPUT_DATA_OFFSET: usize = 48;
    const INPUT_BYTES: usize = 12;
    const PRESENT_COUNT_OFFSET: usize = 64;
    const WORKER_DRAIN_US_OFFSET: usize = 72;
    const FENCE_WAIT_US_OFFSET: usize = 80;
    const READBACK_US_OFFSET: usize = 88;
    const SRGB_ENCODE_US_OFFSET: usize = 96;
    /// Child → Shell vibration request: one atomic `u64` in the wire format
    /// of `raeen_input::rumble::encode_word` (seq<<16 | large<<8 | small).
    /// A single aligned atomic needs no seqlock; 0 means "never set".
    const RUMBLE_WORD_OFFSET: usize = 104;

    // ---- Child -> Shell session status (diagnostics) ----------------------
    //
    // The Shell's F10 console is fed by a *process-local* tracing ring, and the
    // guest runs in a child spawned with inherited (not piped) stdio -- so
    // without this block the console cannot see a single guest event, which is
    // exactly the case a user hits when a title stalls. This page is already
    // mapped between the two processes on every launch and has ~3,984 zeroed
    // bytes spare after the rumble word, so the bridge costs no new mapping,
    // no new thread and no new handle.
    //
    // Like the rumble word, this deliberately does NOT bump `VERSION`: it lives
    // in previously-zeroed padding, moves no existing field, and a peer without
    // it reads `STATUS_SEQ == 0`, which decodes as "never published". Bumping
    // the version instead would hard-fail mismatched peers into "no video" --
    // turning a diagnostics addition into a black screen.
    /// Seqlock for the status block. Odd = a write is in flight; 0 = never
    /// published (never "measured zero").
    const STATUS_SEQ_OFFSET: usize = 112;
    /// Furthest `frame_path::Stage` index **plus one**; 0 = no stage reached.
    const STATUS_STAGE_OFFSET: usize = 120;
    /// Furthest `frame_path::Phase` index **plus one**; 0 = no phase reached.
    const STATUS_PHASE_OFFSET: usize = 128;
    const STATUS_BLOCKERS_OFFSET: usize = 136;
    const STATUS_EVENTS_OFFSET: usize = 144;
    const STATUS_DROPPED_OFFSET: usize = 152;
    const STATUS_CPU_MS_OFFSET: usize = 160;
    const STATUS_WALL_MS_OFFSET: usize = 168;
    const STATUS_TEXT_LEN_OFFSET: usize = 176;
    const STATUS_TEXT_OFFSET: usize = 184;
    /// Text budget for the human-readable digest. Ends at 3,984, inside the
    /// 4,096-byte header -- asserted by a test, not by arithmetic in a comment.
    const STATUS_TEXT_CAPACITY: usize = 3_800;

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

        fn pixels(&self, slot: usize) -> *mut u8 {
            debug_assert!(slot < FRAME_SLOTS);
            // SAFETY: both fixed-capacity slots are inside MAPPING_BYTES.
            unsafe { self.view.add(HEADER_BYTES + slot * PIXEL_CAPACITY) }
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
            mapping
                .atomic_u64(INPUT_SEQUENCE_OFFSET)
                .store(0, Ordering::Release);
            mapping
                .atomic_u64(PRESENT_COUNT_OFFSET)
                .store(0, Ordering::Release);
            for offset in [
                WORKER_DRAIN_US_OFFSET,
                FENCE_WAIT_US_OFFSET,
                READBACK_US_OFFSET,
                SRGB_ENCODE_US_OFFSET,
                RUMBLE_WORD_OFFSET,
                // Zeroing the status seqlock is what makes "never published"
                // distinguishable from "published stage 0" on a reused page.
                STATUS_SEQ_OFFSET,
                STATUS_TEXT_LEN_OFFSET,
            ] {
                mapping.atomic_u64(offset).store(0, Ordering::Relaxed);
            }
            Ok(Self {
                name,
                mapping,
                cache: Mutex::new(None),
            })
        }

        pub fn name(&self) -> &str {
            &self.name
        }

        /// Publish the Shell's merged native/gilrs/keyboard snapshot for the
        /// isolated runner. A separate seqlock keeps this bidirectional field
        /// independent from the child-owned frame slot.
        pub fn publish_pad_state(&self, state: [u8; INPUT_BYTES]) {
            let sequence = self
                .mapping
                .atomic_u64(INPUT_SEQUENCE_OFFSET)
                .load(Ordering::Relaxed);
            let writing = if sequence & 1 == 0 {
                sequence.wrapping_add(1)
            } else {
                sequence.wrapping_add(2)
            };
            self.mapping
                .atomic_u64(INPUT_SEQUENCE_OFFSET)
                .store(writing, Ordering::Release);
            // SAFETY: the input field is a fixed 12-byte range fully contained
            // in the header and does not overlap an atomic field.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    state.as_ptr(),
                    self.mapping.view.add(INPUT_DATA_OFFSET),
                    INPUT_BYTES,
                );
            }
            std::sync::atomic::fence(Ordering::Release);
            self.mapping
                .atomic_u64(INPUT_SEQUENCE_OFFSET)
                .store(writing.wrapping_add(1), Ordering::Release);
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
            let timing = PresentTiming {
                worker_drain_us: self
                    .mapping
                    .atomic_u64(WORKER_DRAIN_US_OFFSET)
                    .load(Ordering::Relaxed),
                fence_wait_us: self
                    .mapping
                    .atomic_u64(FENCE_WAIT_US_OFFSET)
                    .load(Ordering::Relaxed),
                readback_us: self
                    .mapping
                    .atomic_u64(READBACK_US_OFFSET)
                    .load(Ordering::Relaxed),
                srgb_encode_us: self
                    .mapping
                    .atomic_u64(SRGB_ENCODE_US_OFFSET)
                    .load(Ordering::Relaxed),
                egui_upload_us: 0,
            };
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

            // Recycle the previous frame's buffer when one is retained: this
            // runs on the Shell's UI thread once per published frame, and a
            // fresh 8 MB mapping costs ~870 us of page faults alone (see
            // [`crate::frame_pool`]). The copy below fills every byte, so the
            // frame is identical either way.
            let mut pixels = crate::frame_pool::take(len).unwrap_or_default();
            if pixels.capacity() < len && pixels.try_reserve_exact(len).is_err() {
                tracing::warn!(
                    bytes = len,
                    "Shell frame IPC allocation failed under host memory pressure; \
                     preserving the last complete frame"
                );
                return cache.clone();
            }
            pixels.resize(len, 0);
            let slot = ((first / 2) as usize) % FRAME_SLOTS;
            // SAFETY: the validated length is within PIXEL_CAPACITY, both
            // buffers are valid for `len` bytes and do not overlap. The
            // sequence re-check below rejects a copy concurrent with a write.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.mapping.pixels(slot).cast_const(),
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
                timing,
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

        /// The child guest's newest vibration request as an encoded rumble
        /// word (`raeen_input::rumble` wire format), or `None` when the
        /// mapping is not a live protocol peer. `Some(0)` means the peer is
        /// alive but no title ever called `scePadSetVibration` — the decoder
        /// treats that as "no rumble source".
        pub fn latest_rumble_word(&self) -> Option<u64> {
            if self
                .mapping
                .atomic_u32(MAGIC_OFFSET)
                .load(Ordering::Acquire)
                != MAGIC
                || self
                    .mapping
                    .atomic_u32(VERSION_OFFSET)
                    .load(Ordering::Relaxed)
                    != VERSION
            {
                return None;
            }
            Some(
                self.mapping
                    .atomic_u64(RUMBLE_WORD_OFFSET)
                    .load(Ordering::Acquire),
            )
        }

        /// Child VideoOut flips observed across the process boundary. Unlike
        /// the frame-slot sequence, this advances even when the newest complete
        /// image is byte-identical to the prior one or presentation temporarily
        /// reuses the last good frame.
        pub fn present_count(&self) -> Option<u64> {
            if self
                .mapping
                .atomic_u32(MAGIC_OFFSET)
                .load(Ordering::Acquire)
                != MAGIC
                || self
                    .mapping
                    .atomic_u32(VERSION_OFFSET)
                    .load(Ordering::Relaxed)
                    != VERSION
            {
                return None;
            }
            Some(
                self.mapping
                    .atomic_u64(PRESENT_COUNT_OFFSET)
                    .load(Ordering::Acquire),
            )
        }

        /// The guest runner's newest diagnostics status, or `None` when the
        /// mapping is not a live protocol peer.
        ///
        /// `Some(status)` with `seq == 0` means the bridge is alive but the
        /// child has not published yet — deliberately distinct from "the child
        /// reported no progress", which is `seq > 0` with `stage: None`.
        pub fn latest_status(&self) -> Option<SessionStatus> {
            if self
                .mapping
                .atomic_u32(MAGIC_OFFSET)
                .load(Ordering::Acquire)
                != MAGIC
                || self
                    .mapping
                    .atomic_u32(VERSION_OFFSET)
                    .load(Ordering::Relaxed)
                    != VERSION
            {
                return None;
            }
            let first = self
                .mapping
                .atomic_u64(STATUS_SEQ_OFFSET)
                .load(Ordering::Acquire);
            // Never published, or a write is in flight: report the bridge as
            // alive but say nothing about its contents rather than read a torn
            // block or invent a measurement.
            if first == 0 || first & 1 != 0 {
                return Some(SessionStatus::default());
            }

            let read = |offset| self.mapping.atomic_u64(offset).load(Ordering::Relaxed);
            let stage = read(STATUS_STAGE_OFFSET);
            let phase = read(STATUS_PHASE_OFFSET);
            let distinct_blockers = read(STATUS_BLOCKERS_OFFSET);
            let total_events = read(STATUS_EVENTS_OFFSET);
            let dropped_distinct = read(STATUS_DROPPED_OFFSET);
            let cpu_ms = read(STATUS_CPU_MS_OFFSET);
            let wall_ms = read(STATUS_WALL_MS_OFFSET);
            let len = (read(STATUS_TEXT_LEN_OFFSET) as usize).min(STATUS_TEXT_CAPACITY);
            let mut bytes = vec![0u8; len];
            // SAFETY: `len` is clamped to the text capacity, so the source
            // range lies wholly inside the header and overlaps no atomic field.
            // The sequence re-check below rejects a concurrent write.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.mapping.view.add(STATUS_TEXT_OFFSET).cast_const(),
                    bytes.as_mut_ptr(),
                    len,
                );
            }
            std::sync::atomic::fence(Ordering::Acquire);
            let second = self
                .mapping
                .atomic_u64(STATUS_SEQ_OFFSET)
                .load(Ordering::Acquire);
            if first != second || second & 1 != 0 {
                return Some(SessionStatus::default());
            }
            Some(SessionStatus {
                seq: second,
                // The `+1` encoding is what keeps 0 free for "none reached",
                // so a guest that reached nothing never reads as stage 0.
                stage: (stage > 0).then(|| (stage - 1) as u8),
                phase: (phase > 0).then(|| (phase - 1) as u8),
                distinct_blockers,
                total_events,
                dropped_distinct,
                cpu_ms,
                wall_ms,
                // The writer only ever copies whole UTF-8, but a torn or
                // foreign page must degrade to readable text, not a panic.
                digest: String::from_utf8_lossy(&bytes).into_owned(),
            })
        }
    }

    /// Child-owned reader for the Shell's merged controller snapshot.
    pub struct FrameIpcInputReader {
        mapping: Mapping,
        cached: Mutex<Option<(u64, [u8; INPUT_BYTES])>>,
    }

    impl FrameIpcInputReader {
        pub fn open_from_env() -> Option<Self> {
            let name = std::env::var(FRAME_IPC_ENV).ok()?;
            match Mapping::open(&name) {
                Ok(mapping) => {
                    tracing::info!(
                        mapping = %name,
                        "isolated runner connected to Shell input IPC"
                    );
                    Some(Self {
                        mapping,
                        cached: Mutex::new(None),
                    })
                }
                Err(error) => {
                    tracing::error!(
                        %error,
                        mapping = %name,
                        "cannot open Shell input IPC mapping"
                    );
                    None
                }
            }
        }

        /// Return the newest complete snapshot, retaining the prior snapshot
        /// if the Shell is updating the slot concurrently.
        pub fn latest(&self) -> Option<[u8; INPUT_BYTES]> {
            let mut cached = self.cached.lock();
            let first = self
                .mapping
                .atomic_u64(INPUT_SEQUENCE_OFFSET)
                .load(Ordering::Acquire);
            if first == 0 || first & 1 != 0 {
                return cached.as_ref().map(|(_, state)| *state);
            }
            if cached
                .as_ref()
                .is_some_and(|(sequence, _)| *sequence == first)
            {
                return cached.as_ref().map(|(_, state)| *state);
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
                return cached.as_ref().map(|(_, state)| *state);
            }

            let mut state = [0; INPUT_BYTES];
            // SAFETY: the source and destination are valid non-overlapping
            // 12-byte ranges. The sequence re-check rejects concurrent writes.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.mapping.view.add(INPUT_DATA_OFFSET).cast_const(),
                    state.as_mut_ptr(),
                    INPUT_BYTES,
                );
            }
            std::sync::atomic::fence(Ordering::Acquire);
            let second = self
                .mapping
                .atomic_u64(INPUT_SEQUENCE_OFFSET)
                .load(Ordering::Acquire);
            if first != second || second & 1 != 0 {
                return cached.as_ref().map(|(_, state)| *state);
            }
            *cached = Some((second, state));
            Some(state)
        }

        /// Publish the child guest's newest vibration request for the Shell
        /// (the reverse direction of this bidirectional pad channel). The
        /// word is the `raeen_input::rumble` wire format; publishing the
        /// same word repeatedly is idempotent and cheap (one atomic store).
        pub fn publish_rumble_word(&self, word: u64) {
            self.mapping
                .atomic_u64(RUMBLE_WORD_OFFSET)
                .store(word, Ordering::Release);
        }

        /// Publish the guest runner's diagnostics status for the Shell.
        ///
        /// This is what puts a stalled title's state in front of a user: the
        /// Shell reads it every frame and renders it in the F10 Status tab,
        /// and it keeps working after a hard `child.kill()` because the Shell
        /// owns the mapping.
        pub fn publish_status(&self, status: &SessionStatus) {
            let sequence = self
                .mapping
                .atomic_u64(STATUS_SEQ_OFFSET)
                .load(Ordering::Relaxed);
            let writing = if sequence & 1 == 0 {
                sequence.wrapping_add(1)
            } else {
                sequence.wrapping_add(2)
            };
            self.mapping
                .atomic_u64(STATUS_SEQ_OFFSET)
                .store(writing, Ordering::Release);

            let store = |offset, value: u64| {
                self.mapping
                    .atomic_u64(offset)
                    .store(value, Ordering::Relaxed);
            };
            store(
                STATUS_STAGE_OFFSET,
                status.stage.map_or(0, |s| u64::from(s) + 1),
            );
            store(
                STATUS_PHASE_OFFSET,
                status.phase.map_or(0, |p| u64::from(p) + 1),
            );
            store(STATUS_BLOCKERS_OFFSET, status.distinct_blockers);
            store(STATUS_EVENTS_OFFSET, status.total_events);
            store(STATUS_DROPPED_OFFSET, status.dropped_distinct);
            store(STATUS_CPU_MS_OFFSET, status.cpu_ms);
            store(STATUS_WALL_MS_OFFSET, status.wall_ms);

            let text = truncate_on_char_boundary(&status.digest, STATUS_TEXT_CAPACITY);
            store(STATUS_TEXT_LEN_OFFSET, text.len() as u64);
            // SAFETY: `text.len()` is bounded by STATUS_TEXT_CAPACITY, so the
            // destination range lies wholly inside the header and overlaps no
            // atomic field. The seqlock guards concurrent readers.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    text.as_ptr(),
                    self.mapping.view.add(STATUS_TEXT_OFFSET),
                    text.len(),
                );
            }
            std::sync::atomic::fence(Ordering::Release);
            self.mapping
                .atomic_u64(STATUS_SEQ_OFFSET)
                .store(writing.wrapping_add(1), Ordering::Release);
        }
    }

    /// Longest prefix of `text` that fits `max_bytes` without splitting a
    /// `char`. Byte-truncating instead would hand the Shell invalid UTF-8 the
    /// moment a digest carried a non-ASCII character (the report's own
    /// em-dashes do).
    fn truncate_on_char_boundary(text: &str, max_bytes: usize) -> &str {
        if text.len() <= max_bytes {
            return text;
        }
        let mut end = max_bytes;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        &text[..end]
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

        pub(crate) fn publish(&self, image: &RenderedImage, timing: PresentTiming) {
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
            let complete = writing.wrapping_add(1);
            let slot = ((complete / 2) as usize) % FRAME_SLOTS;
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
            self.mapping
                .atomic_u64(WORKER_DRAIN_US_OFFSET)
                .store(timing.worker_drain_us, Ordering::Relaxed);
            self.mapping
                .atomic_u64(FENCE_WAIT_US_OFFSET)
                .store(timing.fence_wait_us, Ordering::Relaxed);
            self.mapping
                .atomic_u64(READBACK_US_OFFSET)
                .store(timing.readback_us, Ordering::Relaxed);
            self.mapping
                .atomic_u64(SRGB_ENCODE_US_OFFSET)
                .store(timing.srgb_encode_us, Ordering::Relaxed);
            // SAFETY: the image length was validated against PIXEL_CAPACITY,
            // both buffers are live and non-overlapping. The odd sequence keeps
            // the receiver from accepting this slot until the release below.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    image.pixels.as_ptr(),
                    self.mapping.pixels(slot),
                    image.pixels.len(),
                );
            }
            std::sync::atomic::fence(Ordering::Release);
            self.mapping
                .atomic_u64(SEQUENCE_OFFSET)
                .store(complete, Ordering::Release);
        }

        /// Record one guest VideoOut presentation independently of image
        /// publication. This keeps the Shell FPS counter alive for static or
        /// repeatedly-reused images.
        pub(crate) fn mark_presented(&self) {
            self.mapping
                .atomic_u64(PRESENT_COUNT_OFFSET)
                .fetch_add(1, Ordering::Release);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Every pixel slot must start on a `SLOT_ALIGNMENT` boundary, or
        /// `VK_EXT_external_memory_host` cannot import it and the GPU-resident
        /// present fast path silently degrades to the buffered copy.
        ///
        /// This is a layout invariant, so it is asserted on the constants
        /// rather than on a live mapping: a future header edit that adds a
        /// field and pushes `HEADER_BYTES` off a page boundary fails here
        /// instead of at a driver call on someone's machine.
        #[test]
        fn every_pixel_slot_is_import_aligned() {
            assert!(
                SLOT_ALIGNMENT.is_power_of_two(),
                "alignment must be a power of two"
            );
            assert_eq!(
                HEADER_BYTES % SLOT_ALIGNMENT,
                0,
                "header must be a whole number of pages so slot 0 is aligned"
            );
            assert_eq!(
                PIXEL_CAPACITY % SLOT_ALIGNMENT,
                0,
                "slot stride must be a multiple of the alignment so every later slot stays aligned"
            );
            for slot in 0..FRAME_SLOTS {
                let offset = HEADER_BYTES + slot * PIXEL_CAPACITY;
                assert_eq!(
                    offset % SLOT_ALIGNMENT,
                    0,
                    "slot {slot} starts at {offset}, which is not {SLOT_ALIGNMENT}-aligned"
                );
            }
            // The header fields must still fit in the padded header (checked
            // at compile time; both operands are constants).
            const {
                assert!(
                    RUMBLE_WORD_OFFSET + 8 <= HEADER_BYTES,
                    "header fields overflow the padded header"
                );
                assert!(
                    RUMBLE_WORD_OFFSET >= SRGB_ENCODE_US_OFFSET + 8,
                    "rumble word overlaps the timing counters"
                );
            }
        }

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
            let timing = PresentTiming {
                worker_drain_us: 11,
                fence_wait_us: 22,
                readback_us: 33,
                srgb_encode_us: 44,
                egui_upload_us: 0,
            };
            publisher.publish(&image, timing);
            let remote = receiver.latest().expect("published frame");
            assert_eq!(remote.image.width, 2);
            assert_eq!(remote.image.height, 2);
            assert_eq!(remote.image.pixels, image.pixels);
            assert_eq!(remote.timing, timing);
            assert_eq!(receiver.latest().unwrap().epoch, remote.epoch);
        }

        #[test]
        fn present_counter_advances_without_republishing_pixels() {
            let receiver = FrameIpcReceiver::create().expect("create mapping");
            let publisher = FrameIpcPublisher {
                mapping: Mapping::open(receiver.name()).expect("open child mapping"),
            };
            assert_eq!(receiver.present_count(), Some(0));
            publisher.mark_presented();
            publisher.mark_presented();
            assert_eq!(receiver.present_count(), Some(2));
            assert!(receiver.latest().is_none());
        }

        #[test]
        fn invalid_frame_never_replaces_the_last_complete_frame() {
            let receiver = FrameIpcReceiver::create().expect("create mapping");
            let publisher = FrameIpcPublisher {
                mapping: Mapping::open(receiver.name()).expect("open child mapping"),
            };
            publisher.publish(
                &RenderedImage {
                    width: 1,
                    height: 1,
                    pixels: vec![1, 2, 3, 255],
                    bytes_per_pixel: 4,
                },
                PresentTiming::default(),
            );
            let accepted = receiver.latest().expect("first frame");
            publisher.publish(
                &RenderedImage {
                    width: 2,
                    height: 2,
                    pixels: vec![0; 3],
                    bytes_per_pixel: 4,
                },
                PresentTiming::default(),
            );
            let kept = receiver.latest().expect("cached frame");
            assert_eq!(kept.epoch, accepted.epoch);
            assert_eq!(kept.image.pixels, accepted.image.pixels);
        }

        #[test]
        fn double_buffer_publishes_the_newest_complete_slot() {
            let receiver = FrameIpcReceiver::create().expect("create mapping");
            let publisher = FrameIpcPublisher {
                mapping: Mapping::open(receiver.name()).expect("open child mapping"),
            };
            for value in [1, 2, 3] {
                publisher.publish(
                    &RenderedImage {
                        width: 1,
                        height: 1,
                        pixels: vec![value, 0, 0, 255],
                        bytes_per_pixel: 4,
                    },
                    PresentTiming::default(),
                );
                let remote = receiver.latest().expect("published frame");
                assert_eq!(remote.image.pixels, [value, 0, 0, 255]);
            }
        }

        #[test]
        fn shell_pad_snapshot_round_trips_to_isolated_runner() {
            let receiver = FrameIpcReceiver::create().expect("create mapping");
            let reader = FrameIpcInputReader {
                mapping: Mapping::open(receiver.name()).expect("open child mapping"),
                cached: Mutex::new(None),
            };
            assert_eq!(reader.latest(), None);

            let mut pressed = [0u8; INPUT_BYTES];
            pressed[0..4].copy_from_slice(&0x0000_4000u32.to_le_bytes());
            pressed[4..8].copy_from_slice(&[0, 255, 128, 128]);
            receiver.publish_pad_state(pressed);
            assert_eq!(reader.latest(), Some(pressed));

            receiver.publish_pad_state([0; INPUT_BYTES]);
            assert_eq!(reader.latest(), Some([0; INPUT_BYTES]));
        }

        /// The reverse pad channel: the child's guest vibration word reaches
        /// the Shell through the same mapping. A live-but-silent peer reads
        /// as `Some(0)` ("never set"), never `None`.
        #[test]
        fn child_rumble_word_round_trips_to_shell() {
            let receiver = FrameIpcReceiver::create().expect("create mapping");
            let reader = FrameIpcInputReader {
                mapping: Mapping::open(receiver.name()).expect("open child mapping"),
                cached: Mutex::new(None),
            };
            assert_eq!(receiver.latest_rumble_word(), Some(0), "never set");

            // seq=1, large=255, small=128 in the rumble wire format.
            let word = (1u64 << 16) | (255 << 8) | 128;
            reader.publish_rumble_word(word);
            assert_eq!(receiver.latest_rumble_word(), Some(word));

            reader.publish_rumble_word(2 << 16);
            assert_eq!(receiver.latest_rumble_word(), Some(2 << 16));
        }

        /// The status block must fit the header page and sit clear of the
        /// rumble word. Arithmetic in a comment is not a guarantee.
        ///
        /// Every operand is a `const`, so these are checked when the crate is
        /// compiled rather than when the test runs — a bad layout fails the
        /// build instead of waiting for `cargo test`. That is strictly stronger
        /// than the runtime assertions this replaced, and it is why clippy's
        /// `assertions_on_constants` was firing here.
        #[test]
        fn the_status_block_fits_the_header_page_and_clears_the_rumble_word() {
            const _: () = assert!(
                STATUS_SEQ_OFFSET >= RUMBLE_WORD_OFFSET + 8,
                "status block overlaps the rumble word"
            );
            const _: () = assert!(
                STATUS_TEXT_OFFSET >= STATUS_TEXT_LEN_OFFSET + 8,
                "status text overlaps its own length field"
            );
            const _: () = assert!(
                STATUS_TEXT_OFFSET + STATUS_TEXT_CAPACITY <= HEADER_BYTES,
                "status text runs past the header into pixel slot 0"
            );
        }

        /// The whole point of the bridge: what the child knows must arrive in
        /// the Shell intact, and what it has not said must not be invented.
        #[test]
        fn child_status_round_trips_to_shell_and_silence_is_not_a_measurement() {
            let receiver = FrameIpcReceiver::create().expect("create mapping");
            let reader = FrameIpcInputReader {
                mapping: Mapping::open(receiver.name()).expect("open child mapping"),
                cached: Mutex::new(None),
            };

            // A live bridge that has never published is `Some`, not `None` —
            // and reads as "no report", not as "stage 0, zero blockers".
            let fresh = receiver.latest_status().expect("bridge is alive");
            assert_eq!(fresh, SessionStatus::default());
            assert!(!fresh.published(), "seq 0 means the child never published");
            assert_eq!(fresh.stage, None, "never publish a fabricated stage 0");

            // A digest longer than the channel budget, with multi-byte chars on
            // the truncation boundary — byte-truncating here would hand the
            // Shell invalid UTF-8.
            let long_digest = "é".repeat(5_000);
            let status = SessionStatus {
                seq: 0, // ignored by the writer; the seqlock owns it
                stage: Some(4),
                phase: Some(3),
                distinct_blockers: 2,
                total_events: 30_736,
                dropped_distinct: 8,
                cpu_ms: 3_200,
                wall_ms: 180_100,
                digest: long_digest.clone(),
            };
            reader.publish_status(&status);

            let read = receiver.latest_status().expect("bridge is alive");
            assert!(read.published(), "seq must advance on publication");
            assert_eq!(read.stage, Some(4));
            assert_eq!(read.phase, Some(3));
            assert_eq!(read.distinct_blockers, 2);
            assert_eq!(read.total_events, 30_736);
            assert_eq!(read.dropped_distinct, 8);
            assert_eq!(read.cpu_ms, 3_200);
            assert_eq!(read.wall_ms, 180_100);
            assert!(
                read.digest.len() <= STATUS_TEXT_CAPACITY,
                "budget respected"
            );
            assert!(
                long_digest.starts_with(&read.digest),
                "truncation must keep a valid prefix, not corrupt the text"
            );
            assert!(!read.digest.is_empty());

            // A stage the guest genuinely has not reached stays `None` across
            // the wire — the `+1` encoding is what keeps 0 free for "none".
            let none_reached = SessionStatus {
                stage: None,
                phase: Some(0),
                digest: "frame path: reached=nothing phase=process_loaded".to_string(),
                ..SessionStatus::default()
            };
            reader.publish_status(&none_reached);
            let read = receiver.latest_status().expect("bridge is alive");
            assert_eq!(read.stage, None, "no stage reached must survive as None");
            assert_eq!(read.phase, Some(0), "phase index 0 is a real phase");
            assert!(read.digest.contains("reached=nothing"));
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

        pub fn present_count(&self) -> Option<u64> {
            None
        }

        pub fn publish_pad_state(&self, _state: [u8; 12]) {}

        pub fn latest_rumble_word(&self) -> Option<u64> {
            None
        }

        pub fn latest_status(&self) -> Option<SessionStatus> {
            None
        }
    }

    pub struct FrameIpcInputReader;

    impl FrameIpcInputReader {
        pub fn open_from_env() -> Option<Self> {
            None
        }

        pub fn latest(&self) -> Option<[u8; 12]> {
            None
        }

        pub fn publish_rumble_word(&self, _word: u64) {}

        pub fn publish_status(&self, _status: &SessionStatus) {}
    }

    pub(crate) struct FrameIpcPublisher;

    impl FrameIpcPublisher {
        pub(crate) fn open_from_env() -> Option<Self> {
            None
        }

        pub(crate) fn publish(&self, _image: &RenderedImage, _timing: PresentTiming) {}

        pub(crate) fn mark_presented(&self) {}
    }
}

pub(crate) use platform::FrameIpcPublisher;
pub use platform::{FrameIpcInputReader, FrameIpcReceiver};

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

/// Latest child VideoOut flip count, independent of whether frame pixels
/// changed. Used by the Shell's FPS badge for isolated retail runners.
pub fn latest_remote_present_count() -> Option<u64> {
    active_receiver()
        .read()
        .as_ref()
        .and_then(|receiver| receiver.present_count())
}

/// Latest child guest vibration request as an encoded rumble word
/// (`raeen_input::rumble` wire format), or `None` when no isolated-runner
/// bridge is installed. `Some(0)` = bridge alive, no vibration ever set.
pub fn latest_remote_rumble_word() -> Option<u64> {
    active_receiver()
        .read()
        .as_ref()
        .and_then(|receiver| receiver.latest_rumble_word())
}

/// The isolated runner's newest diagnostics status, or `None` when no bridge
/// is installed (no title running, or an in-process launch).
///
/// This is what the Shell's F10 Status tab renders. `Some(status)` with
/// `status.published() == false` means "a title is running but has not reported
/// yet" — distinct from "it reported no progress".
pub fn latest_remote_status() -> Option<SessionStatus> {
    active_receiver()
        .read()
        .as_ref()
        .and_then(|receiver| receiver.latest_status())
}

/// Publish one merged Shell controller snapshot to the active isolated title.
/// Returns false when no child frame/input bridge is installed.
pub fn publish_pad_state(state: [u8; 12]) -> bool {
    let active = active_receiver().read();
    let Some(receiver) = active.as_ref() else {
        return false;
    };
    receiver.publish_pad_state(state);
    true
}
