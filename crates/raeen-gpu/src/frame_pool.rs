//! Recycled full-frame pixel buffers.
//!
//! Every full-frame CPU crossing in the present path ends in
//! `Vec<u8>::try_reserve_exact(size)` followed by a copy of `size` bytes:
//! the render-target flush readback
//! ([`crate::vulkan::offscreen::readback_to_vec_fallible`]) and the Shell's
//! copy out of the frame-IPC slot ([`crate::frame_ipc`]). Both allocate a
//! **fresh** buffer per frame and free it when the frame is replaced.
//!
//! At 1080p RGBA that allocation is 8.29 MB, which the OS hands back as fresh
//! pages — so the copy that follows also takes a soft page fault per 4 KiB.
//! Measured on this dev machine (200 iterations, 8.29 MB, copying from cached
//! DRAM so only the allocation differs):
//!
//! | destination            | per frame |
//! |------------------------|-----------|
//! | fresh `Vec` (before)   | 1106 us   |
//! | recycled `Vec` (after) |  239 us   |
//!
//! ~870 us/frame, 4.6x, and it scales with resolution (~3.5 ms at 4K). The
//! real readback copies from a mapped `HOST_CACHED` Vulkan buffer, so the copy
//! half is slower than the benchmark's — but the allocation half is identical,
//! and it is pure waste.
//!
//! Recycling is therefore worth a global pool. [`RenderedImage`]'s `Drop`
//! offers its buffer here, and the two allocation sites take from here first.
//!
//! **This changes no pixel.** A taken buffer is always filled with exactly the
//! same `size` bytes the fresh buffer would have received: [`take`] returns it
//! drained (`len == 0`) with `capacity() >= size`, and every caller writes all
//! `size` bytes before anyone reads them. `Vec::len` — what every consumer of
//! `RenderedImage::pixels` uses — is unchanged; only the spare capacity behind
//! it may differ, and no consumer can observe that.
//!
//! [`RenderedImage`]: crate::vulkan::offscreen::RenderedImage

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Buffers smaller than this are not worth pooling: the win is the page-fault
/// cost of a fresh multi-megabyte mapping, and the tiny images the test suite
/// builds by the thousand would otherwise churn the pool for no benefit.
const MIN_POOLED_BYTES: usize = 1 << 20;

/// A title cycles through a handful of distinct frame sizes (its display-buffer
/// ring, an intermediate, a plugin output). Past this the pool is not caching,
/// it is leaking.
const MAX_POOLED_BUFFERS: usize = 8;

/// Hard cap on retained bytes, so a 4K title plus a resize cannot pin an
/// unbounded amount of host memory in the pool.
const MAX_POOLED_BYTES: usize = 512 << 20;

/// Retained buffers, drained (`len == 0`), smallest-capacity first.
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static RETURNS: AtomicU64 = AtomicU64::new(0);
static DECLINED: AtomicU64 = AtomicU64::new(0);

/// Pool effectiveness. `hits + misses` is how many full-frame buffers the
/// present path asked for; `hits` is how many cost no allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FramePoolStats {
    /// Requests served from the pool (no allocation).
    pub hits: u64,
    /// Requests the pool could not serve (the caller allocated).
    pub misses: u64,
    /// Buffers accepted back into the pool.
    pub returns: u64,
    /// Buffers offered back and dropped (too small, or the pool was full).
    pub declined: u64,
    /// Buffers currently retained.
    pub buffers: usize,
    /// Bytes of capacity currently retained.
    pub bytes: usize,
}

/// Counters for the pool, for diagnostics and tests.
#[must_use]
pub fn stats() -> FramePoolStats {
    let pool = POOL.lock();
    FramePoolStats {
        hits: HITS.load(Ordering::Relaxed),
        misses: MISSES.load(Ordering::Relaxed),
        returns: RETURNS.load(Ordering::Relaxed),
        declined: DECLINED.load(Ordering::Relaxed),
        buffers: pool.len(),
        bytes: pool.iter().map(Vec::capacity).sum(),
    }
}

/// A drained buffer with `capacity() >= size`, or `None` when the pool holds
/// nothing suitable and the caller must allocate.
///
/// Picks the smallest buffer that fits, so an 8 MB request cannot consume a
/// retained 33 MB (4K) buffer and leave the large one's capacity stranded
/// behind a small frame.
#[must_use]
pub(crate) fn take(size: usize) -> Option<Vec<u8>> {
    if size < MIN_POOLED_BYTES {
        return None;
    }
    let mut pool = POOL.lock();
    let index = pool
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.capacity() >= size)
        .min_by_key(|(_, buffer)| buffer.capacity())
        .map(|(index, _)| index);
    match index {
        Some(index) => {
            let mut buffer = pool.swap_remove(index);
            drop(pool);
            // Belt and braces: `give` already drained it. `clear` cannot
            // reallocate, so the capacity contract above still holds.
            buffer.clear();
            HITS.fetch_add(1, Ordering::Relaxed);
            Some(buffer)
        }
        None => {
            drop(pool);
            MISSES.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

/// Offer a buffer back. Dropped rather than retained when it is too small to be
/// worth recycling or the pool is at its bound — so this is always safe to
/// call, including from `Drop`.
pub(crate) fn give(mut buffer: Vec<u8>) {
    let capacity = buffer.capacity();
    if capacity < MIN_POOLED_BYTES {
        DECLINED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    buffer.clear();
    let mut pool = POOL.lock();
    let retained: usize = pool.iter().map(Vec::capacity).sum();
    if pool.len() >= MAX_POOLED_BUFFERS || retained.saturating_add(capacity) > MAX_POOLED_BYTES {
        drop(pool);
        DECLINED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    pool.push(buffer);
    drop(pool);
    RETURNS.fetch_add(1, Ordering::Relaxed);
}

/// Drop every retained buffer. For tests that assert on pool occupancy, and for
/// a title teardown that should not leave a 4K frame ring resident.
pub fn clear() {
    let drained: Vec<Vec<u8>> = std::mem::take(&mut *POOL.lock());
    drop(drained);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pool is process-global, so these tests must not run concurrently
    /// with each other.
    static SERIAL: Mutex<()> = Mutex::new(());

    const BIG: usize = 4 << 20;

    #[test]
    fn a_returned_buffer_serves_the_next_request_without_allocating() {
        let _serial = SERIAL.lock();
        clear();
        let first = Vec::<u8>::with_capacity(BIG);
        let address = first.as_ptr() as usize;
        give(first);
        assert_eq!(stats().buffers, 1);

        let taken = take(BIG).expect("a retained buffer of exactly this size fits");
        assert_eq!(
            taken.as_ptr() as usize,
            address,
            "the recycled buffer must be the SAME allocation, not a fresh one"
        );
        assert_eq!(taken.len(), 0, "take hands back a drained buffer");
        assert!(taken.capacity() >= BIG);
        clear();
    }

    #[test]
    fn a_request_larger_than_anything_retained_misses() {
        let _serial = SERIAL.lock();
        clear();
        give(Vec::<u8>::with_capacity(BIG));
        assert!(
            take(BIG * 4).is_none(),
            "a buffer that does not fit must not be handed out"
        );
        assert_eq!(
            stats().buffers,
            1,
            "a miss leaves the unsuitable buffer retained"
        );
        clear();
    }

    #[test]
    fn a_small_buffer_is_never_retained() {
        let _serial = SERIAL.lock();
        clear();
        give(vec![0u8; 64]);
        assert_eq!(stats().buffers, 0);
        assert!(take(64).is_none(), "sub-threshold requests never hit");
        clear();
    }

    #[test]
    fn the_pool_is_bounded_by_count() {
        let _serial = SERIAL.lock();
        clear();
        for _ in 0..MAX_POOLED_BUFFERS * 3 {
            give(Vec::<u8>::with_capacity(BIG));
        }
        assert_eq!(
            stats().buffers,
            MAX_POOLED_BUFFERS,
            "the pool must never grow past its bound"
        );
        clear();
        assert_eq!(stats().buffers, 0);
    }

    #[test]
    fn the_pool_is_bounded_by_bytes() {
        let _serial = SERIAL.lock();
        clear();
        // Two buffers of a third of the byte cap each fit; the cap stops the
        // third even though the count bound would allow it.
        let huge = MAX_POOLED_BYTES / 3 + 1;
        for _ in 0..3 {
            give(Vec::<u8>::with_capacity(huge));
        }
        let stats = stats();
        assert!(
            stats.bytes <= MAX_POOLED_BYTES,
            "retained {} B exceeds the {MAX_POOLED_BYTES} B cap",
            stats.bytes
        );
        assert_eq!(stats.buffers, 2);
        clear();
    }

    #[test]
    fn take_picks_the_smallest_buffer_that_fits() {
        let _serial = SERIAL.lock();
        clear();
        let small = Vec::<u8>::with_capacity(BIG);
        let small_address = small.as_ptr() as usize;
        give(Vec::<u8>::with_capacity(BIG * 8));
        give(small);
        let taken = take(BIG).expect("the small buffer fits");
        assert_eq!(
            taken.as_ptr() as usize,
            small_address,
            "an 8 MB request must not strand a retained 4K-sized buffer"
        );
        clear();
    }
}
