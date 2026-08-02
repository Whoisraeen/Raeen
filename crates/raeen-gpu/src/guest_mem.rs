//! Process-authorized guest memory for the PM4 command processor.
//!
//! Several Gen5 packets carry guest pointers out-of-band — `R_*_REGS_INDIRECT`
//! register lists and indirect-draw argument buffers (see
//! `kyty_graphics::run`). Kyty dereferences those pointers directly because
//! its command processor runs inside the guest's address space; Raeen's
//! `GuestArena` is **identity-mapped**, so the same is true here: a guest
//! virtual address *is* a host address in this process.
//!
//! A committed Windows page is not necessarily guest-owned. Every access is
//! therefore routed through the active process's [`GpuGuestMemory`] authority;
//! the GPU crate never probes or dereferences arbitrary host pages itself.

use kyty_graphics::run::GuestMemory;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Address-space authority supplied by the owning guest process.
pub trait GpuGuestMemory: Send + Sync {
    fn validate_gpu_range(&self, addr: u64, len: u64, write: bool) -> bool;
    fn read_gpu(&self, addr: u64, out: &mut [u8]) -> bool;
    /// Copy guest bytes into uninitialized host storage.
    ///
    /// # Safety
    ///
    /// Returning `true` promises that every element of `out` was initialized.
    /// The default initializes the slice before delegating to [`Self::read_gpu`];
    /// identity-mapped process authorities may override this to copy directly.
    unsafe fn read_gpu_uninit(&self, addr: u64, out: &mut [std::mem::MaybeUninit<u8>]) -> bool {
        for byte in out.iter_mut() {
            byte.write(0);
        }
        // SAFETY: every element was initialized to zero above, and u8 accepts
        // every bit pattern.
        let initialized =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len()) };
        self.read_gpu(addr, initialized)
    }
    fn write_gpu(&self, addr: u64, data: &[u8]) -> bool;
}

pub(crate) struct DenyGpuMemory;

impl GpuGuestMemory for DenyGpuMemory {
    fn validate_gpu_range(&self, _addr: u64, _len: u64, _write: bool) -> bool {
        false
    }

    fn read_gpu(&self, _addr: u64, _out: &mut [u8]) -> bool {
        false
    }

    fn write_gpu(&self, _addr: u64, _data: &[u8]) -> bool {
        false
    }
}

thread_local! {
    static ACTIVE_MEMORY: RefCell<Option<Arc<dyn GpuGuestMemory>>> = RefCell::new(None);
    static ACTIVE_BUDGET: Cell<u64> = const { Cell::new(0) };
}

/// Guest-layout bytes produced by a compute storage image whose allocation has
/// no writable CPU mirror.
///
/// A PS5 image can be rebound through a different descriptor (for example,
/// GTA V writes a 32-bit UAV at `0x161473000`, then samples the same bytes as a
/// 2048x4096 R8 font atlas). Keeping only the Vulkan storage image cannot serve
/// that alias because the sampled view can have a different format and tiling
/// interpretation. The encoded guest-layout bytes are therefore the coherence
/// boundary: later resource reads see the GPU result exactly as if writeback
/// had reached guest memory.
struct GpuImageShadow {
    base: u64,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct GpuImageShadows {
    entries: VecDeque<GpuImageShadow>,
    bytes: usize,
}

/// Keep GPU-only shadows bounded independently of the Vulkan compute-image
/// cache. A single oversized result is retained (correctness wins for the most
/// recent image); otherwise oldest complete images are evicted first.
const MAX_GPU_IMAGE_SHADOW_BYTES: usize = 256 << 20;

static GPU_IMAGE_SHADOWS: Mutex<GpuImageShadows> = Mutex::new(GpuImageShadows {
    entries: VecDeque::new(),
    bytes: 0,
});

fn shadow_covers(base: u64, len: u64, shadow: &GpuImageShadow) -> Option<std::ops::Range<usize>> {
    let end = base.checked_add(len)?;
    let shadow_end = shadow.base.checked_add(shadow.bytes.len() as u64)?;
    if base < shadow.base || end > shadow_end {
        return None;
    }
    let start = usize::try_from(base - shadow.base).ok()?;
    let len = usize::try_from(len).ok()?;
    Some(start..start.checked_add(len)?)
}

fn read_gpu_image_shadow(base: u64, len: u64) -> Option<Vec<u8>> {
    let shadows = GPU_IMAGE_SHADOWS.lock().ok()?;
    // The back is the most recent writer. Overlapping resources therefore
    // observe the last completed compute publication.
    shadows.entries.iter().rev().find_map(|shadow| {
        let range = shadow_covers(base, len, shadow)?;
        Some(shadow.bytes[range].to_vec())
    })
}

fn gpu_image_shadow_covers(base: u64, len: u64) -> bool {
    GPU_IMAGE_SHADOWS.lock().is_ok_and(|shadows| {
        shadows
            .entries
            .iter()
            .rev()
            .any(|shadow| shadow_covers(base, len, shadow).is_some())
    })
}

fn invalidate_gpu_image_shadows(base: u64, len: u64) {
    if base == 0 || len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    if let Ok(mut shadows) = GPU_IMAGE_SHADOWS.lock() {
        let mut removed = 0usize;
        shadows.entries.retain(|shadow| {
            let shadow_end = shadow.base.saturating_add(shadow.bytes.len() as u64);
            let overlaps = base < shadow_end && shadow.base < end;
            if overlaps {
                removed = removed.saturating_add(shadow.bytes.len());
            }
            !overlaps
        });
        shadows.bytes = shadows.bytes.saturating_sub(removed);
    }
}

fn retain_gpu_image_shadow(base: u64, bytes: Vec<u8>) {
    if base == 0 || bytes.is_empty() {
        return;
    }
    let end = base.saturating_add(bytes.len() as u64);
    if let Ok(mut shadows) = GPU_IMAGE_SHADOWS.lock() {
        let mut removed = 0usize;
        shadows.entries.retain(|shadow| {
            let shadow_end = shadow.base.saturating_add(shadow.bytes.len() as u64);
            let overlaps = base < shadow_end && shadow.base < end;
            if overlaps {
                removed = removed.saturating_add(shadow.bytes.len());
            }
            !overlaps
        });
        shadows.bytes = shadows.bytes.saturating_sub(removed);
        shadows.bytes = shadows.bytes.saturating_add(bytes.len());
        shadows.entries.push_back(GpuImageShadow { base, bytes });
        while shadows.bytes > MAX_GPU_IMAGE_SHADOW_BYTES && shadows.entries.len() > 1 {
            if let Some(evicted) = shadows.entries.pop_front() {
                shadows.bytes = shadows.bytes.saturating_sub(evicted.bytes.len());
            }
        }
    }
}

/// Cumulative guest bytes one submission may read through resource fetches
/// (vertex/index/storage buffers, textures) before further reads are refused.
///
/// This is a *total-work* ceiling, not a peak-memory one: every resource read is
/// transient (allocated, copied out, freed), and each INDIVIDUAL read is already
/// bounded to [`MAX_RESOURCE_READ_DWORDS`] (256 MiB) AND validated against
/// committed guest memory — that per-read cap is the real wraparound / mis-decode
/// guard. So this cumulative budget only needs to stop a pathological stream from
/// doing unbounded cumulative reads; it must not refuse a legitimately
/// texture-heavy frame.
///
/// 256 MiB was too tight and refused real frames: Minecraft's menu submission
/// samples its panorama skybox (measured ~25 MiB and ~63 MiB textures) across
/// several draws in ONE submission, so the cumulative total clears 256 MiB even
/// though no single allocation is large — the reads came back
/// `not fully readable (readable prefix == size)`, i.e. refused by THIS cap, not
/// by a memory fault. Raised to 2 GiB: ~10x headroom for a texture-heavy menu
/// while still bounding a runaway/mis-decoded stream (a submission touching over
/// 2 GiB of distinct guest resources is not a real menu). Reset per submission
/// by [`with_guest_memory`].
const MAX_SUBMISSION_GUEST_BYTES: u64 = 2 << 30;

struct ActiveMemoryGuard {
    memory: Option<Arc<dyn GpuGuestMemory>>,
    budget: u64,
}

impl Drop for ActiveMemoryGuard {
    fn drop(&mut self) {
        ACTIVE_MEMORY.with(|active| {
            *active.borrow_mut() = self.memory.take();
        });
        ACTIVE_BUDGET.with(|budget| budget.set(self.budget));
    }
}

pub(crate) fn with_guest_memory<T>(memory: &Arc<dyn GpuGuestMemory>, f: impl FnOnce() -> T) -> T {
    let previous = ACTIVE_MEMORY.with(|active| active.borrow_mut().replace(Arc::clone(memory)));
    let previous_budget = ACTIVE_BUDGET.with(|budget| budget.replace(MAX_SUBMISSION_GUEST_BYTES));
    let _guard = ActiveMemoryGuard {
        memory: previous,
        budget: previous_budget,
    };
    f()
}

fn with_active_memory<T>(f: impl FnOnce(&dyn GpuGuestMemory) -> T) -> Option<T> {
    ACTIVE_MEMORY.with(|active| active.borrow().as_deref().map(f))
}

fn charge_guest_bytes(bytes: u64) -> bool {
    ACTIVE_BUDGET.with(|budget| {
        let remaining = budget.get();
        if bytes > remaining {
            return false;
        }
        budget.set(remaining - bytes);
        true
    })
}

/// Upper bound on a single out-of-band read. The largest legitimate consumer
/// is an indirect register list (`UC_NUM` pairs = 32 Ki dwords); anything
/// bigger is a mis-decode.
const MAX_READ_DWORDS: u32 = 0x1_0000;

/// [`GuestMemory`] over the identity-mapped guest arena.
///
/// Reads are refused (returning `None`, which the command processor turns
/// into a rate-limited warn + packet skip) when the address is null,
/// unaligned, oversized, or any page in the range is not committed readable
/// memory in this process.
pub(crate) struct IdentityGuestMemory;

impl GuestMemory for IdentityGuestMemory {
    fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
        read_dwords_checked(addr, count)
    }

    /// Chained command buffer read (`IT_INDIRECT_BUFFER` target): validated
    /// exactly like [`Self::read_dwords`] but under the RESOURCE cap, not the
    /// out-of-band POINTER cap. A chain target's length comes from the packet's
    /// own 20-bit `IB_SIZE` field, so a legal one runs to 0xF_FFFF dwords
    /// (4 MiB) — `MAX_READ_DWORDS` (0x1_0000) would refuse a valid frame's
    /// command stream as if it were a mis-decoded pointer.
    /// Chained command buffer read (`IT_INDIRECT_BUFFER` target): validated
    /// exactly like [`Self::read_dwords`] but under the RESOURCE cap, not the
    /// out-of-band POINTER cap. A chain target's length comes from the packet's
    /// own 20-bit `IB_SIZE` field, so a legal one runs to 0xF_FFFF dwords
    /// (4 MiB) — `MAX_READ_DWORDS` (0x1_0000) would refuse a valid frame's
    /// command stream as if it were a mis-decoded pointer.
    fn read_command_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
        read_dwords_validated(addr, count)
    }

    /// DMA payload read: dword-granular under the RESOURCE cap (a DMA fill of
    /// a 1080p scanout buffer is ~8 MiB — far beyond the pointer-read cap but
    /// a legitimate resource-sized transfer).
    fn read_bytes(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
        read_bytes_validated(addr, len)
    }

    fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
        // An `IT_DMA_DATA` memory→memory copy runs through here; if it targets a
        // flipped display buffer this is exactly the scene→scanout fill we are
        // hunting (SharpEmu run.rs cp_op_dma_data comment: the composite lands
        // its scanout by DMA).
        trace_scanout_fill(addr, bytes.len(), "dma_data");
        write_bytes_checked(addr, bytes)
    }
}

/// Upper bound on a single *resource* fetch (vertex/index/storage buffers,
/// textures): 256 MiB. Refuses wraparound garbage while covering any real
/// title resource — measured: Minecraft binds a 4 MiB vertex arena V#.
pub(crate) const MAX_RESOURCE_READ_DWORDS: u32 = 0x0400_0000;

/// Committed-readable prefix of a guest range, for naming a failed fetch
/// precisely (wild base ≈ 0; lazy tail ≈ a page-aligned interior cut).
pub(crate) fn readable_prefix(addr: u64, size: u64) -> u64 {
    if size == 0 {
        return 0;
    }
    if gpu_image_shadow_covers(addr, size) {
        return size;
    }
    with_active_memory(|memory| {
        if memory.validate_gpu_range(addr, size, false) {
            return size;
        }

        // `validate_gpu_range` answers only an all-or-nothing question, but
        // callers need the committed prefix when a bounded speculative read
        // crosses the end of an allocation. Range validity is monotonic for a
        // fixed base, so find the largest valid byte count without probing or
        // dereferencing memory outside the process-provided authority.
        if !memory.validate_gpu_range(addr, 1, false) {
            return 0;
        }
        let mut valid = 1;
        let mut invalid = size;
        while valid + 1 < invalid {
            let candidate = valid + (invalid - valid) / 2;
            if memory.validate_gpu_range(addr, candidate, false) {
                valid = candidate;
            } else {
                invalid = candidate;
            }
        }
        valid
    })
    .unwrap_or(0)
}

/// VirtualQuery-validated read of guest dwords (identity map). `None` when the
/// range is null/unaligned/oversized or not fully committed-readable. Shared
/// by [`IdentityGuestMemory`] and the shader fetch layer.
pub(crate) fn read_dwords_checked(addr: u64, count: u32) -> Option<Vec<u32>> {
    if count == 0 || count > MAX_READ_DWORDS {
        return None;
    }
    read_dwords_validated(addr, count)
}

/// The same validated read without the out-of-band cap, for guest *resources*
/// (vertex/storage buffers, textures) whose legitimate size runs to megabytes.
/// `MAX_READ_DWORDS` exists only to refuse mis-decoded command-processor
/// pointer reads; a V# declaring a 4 MiB vertex arena is not a mis-decode
/// (measured on Minecraft's menu draws).
pub(crate) fn read_dwords_validated(addr: u64, count: u32) -> Option<Vec<u32>> {
    if count == 0 || count > MAX_RESOURCE_READ_DWORDS || addr == 0 || !addr.is_multiple_of(4) {
        return None;
    }
    let bytes = count as usize * 4;
    if !charge_guest_bytes(bytes as u64) {
        return None;
    }
    let mut out = Vec::<u32>::new();
    out.try_reserve_exact(count as usize).ok()?;
    out.resize(count as usize, 0);
    // SAFETY: `out` owns `count * 4` initialized bytes and u32 has no invalid
    // bit patterns. The slice is used only as the destination of a bounded
    // process-authorized copy.
    let raw = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), bytes) };
    let accepted = with_active_memory(|memory| {
        memory.validate_gpu_range(addr, bytes as u64, false) && memory.read_gpu(addr, raw)
    })?;
    if !accepted {
        return None;
    }
    Some(out)
}

/// Process-authorized resource read directly into bytes.
///
/// This has the same address, size, submission-budget, and memory-authority
/// contract as [`read_dwords_validated`] but avoids the old two-allocation
/// `Vec<u32> -> Vec<u8>` conversion in texture/storage/vertex hot paths.
pub(crate) fn read_bytes_validated(addr: u64, len: u64) -> Option<Vec<u8>> {
    if len == 0 || !len.is_multiple_of(4) || addr == 0 || !addr.is_multiple_of(4) {
        return None;
    }
    let count = u32::try_from(len / 4).ok()?;
    if count == 0 || count > MAX_RESOURCE_READ_DWORDS || !charge_guest_bytes(len) {
        return None;
    }
    // Prefer the newest GPU publication over a readable-but-stale CPU mirror.
    // This also makes genuinely GPU-only allocations readable to later draws.
    if let Some(bytes) = read_gpu_image_shadow(addr, len) {
        return Some(bytes);
    }
    let bytes = usize::try_from(len).ok()?;
    let mut out = Vec::<u8>::new();
    out.try_reserve_exact(bytes).ok()?;
    let spare = &mut out.spare_capacity_mut()[..bytes];
    // SAFETY: `read_gpu_uninit` may return true only after initializing every
    // byte in `spare`; the length is exposed only on that success path.
    let accepted = with_active_memory(|memory| unsafe {
        memory.validate_gpu_range(addr, len, false) && memory.read_gpu_uninit(addr, spare)
    })?;
    if !accepted {
        return None;
    }
    // SAFETY: the accepted authority call initialized all `bytes` elements.
    unsafe { out.set_len(bytes) };
    Some(out)
}

/// Process-authorized resource read into caller-owned initialized storage.
///
/// Texture freshness probes read 64 small, potentially unaligned windows from
/// one guest allocation. Routing each window through `read_bytes_validated`
/// allocated a temporary `Vec` and then copied its aligned sub-slice again.
/// This seam preserves the same process authority, submission byte budget,
/// GPU-image-shadow preference, and resource-size ceiling while letting that
/// hot path reuse one bounded scratch buffer.
pub(crate) fn read_bytes_into_validated(addr: u64, out: &mut [u8]) -> bool {
    let len = out.len() as u64;
    if addr == 0
        || len == 0
        || len > u64::from(MAX_RESOURCE_READ_DWORDS).saturating_mul(4)
        || !charge_guest_bytes(len)
    {
        return false;
    }
    if let Some(bytes) = read_gpu_image_shadow(addr, len) {
        out.copy_from_slice(&bytes);
        return true;
    }
    with_active_memory(|memory| {
        memory.validate_gpu_range(addr, len, false) && memory.read_gpu(addr, out)
    })
    .unwrap_or(false)
}

/// Scene→scanout fill trace (SharpEmu port task #5). The title composites its
/// HDR scene into a set of render targets, then flips to a *different* display
/// buffer (e.g. ASTRO.BOT renders to 0x53a.../0x539... but flips to
/// 0x507.../0x509...). This registry lets the two guest-write chokepoints —
/// `IT_DMA_DATA` copies (`IdentityGuestMemory::write_bytes`) and Vulkan compute
/// storage writeback (`draw_translate`) — report when a write actually lands in
/// a flipped display buffer, which is the only way that buffer gets filled
/// short of a GPU render pass we already track. Gated on `RAEEN_TRACE_SCANOUT_FILL`.
static SCANOUT_WATCH: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
/// Fallback span when a flip carries no descriptor (unknown frame size): a 4K
/// RGBA/10-bit frame is ~33 MiB, a 4K RGBA16F frame ~66 MiB.
const SCANOUT_WATCH_DEFAULT_SPAN: u64 = 64 << 20;

/// Register a flipped display-buffer `[base, base + byte_len)` so later guest
/// writes into exactly that frame region are reported by [`trace_scanout_fill`]
/// — a precise span (from the flip descriptor) keeps the trace from blaming an
/// unrelated sub-allocation that merely sits nearby. Idempotent per base.
pub(crate) fn register_scanout_watch(addr: u64, byte_len: u64) {
    if addr == 0 {
        return;
    }
    let span = if byte_len == 0 {
        SCANOUT_WATCH_DEFAULT_SPAN
    } else {
        byte_len
    };
    if let Ok(mut watch) = SCANOUT_WATCH.lock() {
        if let Some(entry) = watch.iter_mut().find(|(base, _)| *base == addr) {
            entry.1 = entry.1.max(span);
        } else {
            watch.push((addr, span));
        }
    }
}

/// Whether a compute storage image can be a display/composite candidate.
///
/// Titles also use compute storage images for square atlases, mip generation,
/// and detiling. Putting every non-zero UAV into the presentation census lets
/// those resources replace the real frame. An exact VideoOut address is always
/// eligible; an intermediate at another address is eligible only when its byte
/// extent matches a known, explicitly sized scanout.
pub(crate) fn is_scanout_candidate(addr: u64, byte_len: usize) -> bool {
    if addr == 0 || byte_len == 0 {
        return false;
    }
    SCANOUT_WATCH.lock().is_ok_and(|watch| {
        watch.iter().any(|&(base, span)| {
            addr == base
                || (span != SCANOUT_WATCH_DEFAULT_SPAN
                    && (span == byte_len as u64 || span / 2 == byte_len as u64))
        })
    })
}

/// Report a guest write that lands in a watched flipped display buffer, tagged
/// with the mechanism (`dma_data`, `compute-storage`, `compute-image`). This is
/// the instrumentation that answers "how is the scanout buffer filled?": if the
/// scene never reaches the screen, this trace stays silent for the flip address
/// (nothing writes it) or fires with the mechanism that does. Rate-limited so a
/// per-frame fill does not flood the log.
pub(crate) fn trace_scanout_fill(addr: u64, len: usize, source: &str) {
    static COUNT: AtomicU64 = AtomicU64::new(0);
    if std::env::var_os("RAEEN_TRACE_SCANOUT_FILL").is_none() {
        return;
    }
    let hit = SCANOUT_WATCH.lock().ok().and_then(|watch| {
        watch
            .iter()
            .find(|&&(base, span)| addr >= base && addr < base.saturating_add(span))
            .map(|&(base, _)| base)
    });
    let Some(base) = hit else {
        return;
    };
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if n < 64 || n.is_power_of_two() {
        tracing::info!(
            scanout_base = format_args!("{base:#x}"),
            write_addr = format_args!("{addr:#x}"),
            offset = addr - base,
            len,
            source,
            "TRACE_SCANOUT_FILL: guest write landed in a flipped display buffer"
        );
    }
}

/// Process-authorized write into GPU-visible guest memory. Used for Vulkan
/// compute storage-buffer writeback after the queue fence signals.
pub(crate) fn write_bytes_checked(addr: u64, bytes: &[u8]) -> bool {
    if addr == 0 || bytes.is_empty() || bytes.len() > MAX_RESOURCE_READ_DWORDS as usize * 4 {
        return false;
    }
    if addr.checked_add(bytes.len() as u64).is_none() {
        return false;
    }
    if !charge_guest_bytes(bytes.len() as u64) {
        return false;
    }
    let written = with_active_memory(|memory| {
        memory.validate_gpu_range(addr, bytes.len() as u64, true) && memory.write_gpu(addr, bytes)
    })
    .unwrap_or(false);
    if written {
        invalidate_gpu_image_shadows(addr, bytes.len() as u64);
    }
    written
}

/// Result of trying to mirror a completed compute storage image into guest
/// memory.
///
/// Storage images can be GPU-only allocations: later draws consume their
/// persistent Vulkan image even when no CPU-visible guest mapping exists.
/// Refusing the whole submission in that case discards a valid compositor
/// result. Storage-buffer writeback deliberately does not use this policy
/// because guest CPU code may depend on those bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComputeImageGuestMirror {
    Written,
    GpuOnly,
}

/// Best-effort CPU mirror for a compute storage image.
///
/// [`ComputeImageGuestMirror::GpuOnly`] is a successful GPU publication: the
/// caller must retain the persistent GPU/readback result for later sampling or
/// presentation. The warning is process-rate-limited (first, then powers of
/// two) because titles commonly reuse the same GPU-only image every frame.
pub(crate) fn mirror_compute_image_to_guest(
    addr: u64,
    bytes: Vec<u8>,
    source: &'static str,
) -> ComputeImageGuestMirror {
    trace_scanout_fill(addr, bytes.len(), source);
    if write_bytes_checked(addr, &bytes) {
        return ComputeImageGuestMirror::Written;
    }

    let len = bytes.len();
    retain_gpu_image_shadow(addr, bytes);
    static GPU_ONLY_IMAGES: AtomicU64 = AtomicU64::new(0);
    let count = GPU_ONLY_IMAGES.fetch_add(1, Ordering::Relaxed) + 1;
    if count == 1 || count.is_power_of_two() {
        tracing::warn!(
            count,
            base = format_args!("{addr:#x}"),
            end = format_args!("{:#x}", addr.saturating_add(len as u64)),
            len,
            source,
            "compute storage image has no writable CPU guest mirror; retaining a guest-layout \
             shadow for later GPU reads"
        );
    }
    ComputeImageGuestMirror::GpuOnly
}

/// Install a bounded host-allocation authority for unit tests that model
/// identity-mapped guest buffers with live Vec/array storage. Production code
/// cannot call this and never regains the old arbitrary-host-page probe.
#[cfg(test)]
pub(crate) fn with_test_ranges<T>(ranges: &[(u64, usize)], f: impl FnOnce() -> T) -> T {
    struct TestRanges(Vec<(u64, u64)>);

    impl GpuGuestMemory for TestRanges {
        fn validate_gpu_range(&self, addr: u64, len: u64, _write: bool) -> bool {
            self.0.iter().any(|&(start, size)| {
                addr >= start
                    && addr
                        .checked_add(len)
                        .is_some_and(|end| end <= start.saturating_add(size))
            })
        }

        fn read_gpu(&self, addr: u64, out: &mut [u8]) -> bool {
            if !self.validate_gpu_range(addr, out.len() as u64, false) {
                return false;
            }
            // SAFETY: test callers describe live allocations and the validated
            // access is wholly contained in one of those allocations for the
            // synchronous closure lifetime.
            unsafe {
                std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len())
            };
            true
        }

        fn write_gpu(&self, addr: u64, data: &[u8]) -> bool {
            if !self.validate_gpu_range(addr, data.len() as u64, true) {
                return false;
            }
            // SAFETY: same bounded live-allocation proof as `read_gpu`.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len()) };
            true
        }
    }

    let memory: Arc<dyn GpuGuestMemory> = Arc::new(TestRanges(
        ranges
            .iter()
            .map(|&(start, size)| (start, size as u64))
            .collect(),
    ));
    with_guest_memory(&memory, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct HostRange {
        start: u64,
        len: u64,
    }

    impl GpuGuestMemory for HostRange {
        fn validate_gpu_range(&self, addr: u64, len: u64, _write: bool) -> bool {
            addr >= self.start
                && addr
                    .checked_add(len)
                    .is_some_and(|end| end <= self.start + self.len)
        }

        fn read_gpu(&self, addr: u64, out: &mut [u8]) -> bool {
            if !self.validate_gpu_range(addr, out.len() as u64, false) {
                return false;
            }
            // SAFETY: this test authority is created from a live Vec and the
            // validated range stays within that Vec for the closure lifetime.
            unsafe {
                std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len())
            };
            true
        }

        fn write_gpu(&self, addr: u64, data: &[u8]) -> bool {
            if !self.validate_gpu_range(addr, data.len() as u64, true) {
                return false;
            }
            // SAFETY: same bounded test allocation argument as `read_gpu`.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len()) };
            true
        }
    }

    fn memory_for(data: &mut [u32]) -> Arc<dyn GpuGuestMemory> {
        Arc::new(HostRange {
            start: data.as_mut_ptr() as u64,
            len: std::mem::size_of_val(data) as u64,
        })
    }

    #[test]
    fn reads_committed_host_memory_identity_mapped() {
        let mut data: Vec<u32> = vec![0xAABB_CCDD, 1, 2, 3];
        let addr = data.as_ptr() as u64;
        let memory = memory_for(&mut data);
        let got = with_guest_memory(&memory, || IdentityGuestMemory.read_dwords(addr, 4));
        assert_eq!(got, Some(data.clone()));
    }

    #[test]
    fn reads_resource_bytes_without_dword_conversion() {
        let mut data: Vec<u32> = vec![0xAABB_CCDD, 0x0102_0304, 0x1122_3344];
        let addr = data.as_ptr() as u64;
        let expected: Vec<u8> = data.iter().flat_map(|word| word.to_le_bytes()).collect();
        let memory = memory_for(&mut data);
        let got = with_guest_memory(&memory, || {
            read_bytes_validated(addr, expected.len() as u64)
        });
        assert_eq!(got, Some(expected));
    }

    #[test]
    fn reads_unaligned_resource_bytes_into_reusable_storage() {
        let mut data: Vec<u32> = vec![0xAABB_CCDD, 0x0102_0304, 0x1122_3344];
        let addr = data.as_ptr() as u64;
        let all: Vec<u8> = data.iter().flat_map(|word| word.to_le_bytes()).collect();
        let memory = memory_for(&mut data);
        let mut out = [0u8; 7];
        let accepted = with_guest_memory(&memory, || read_bytes_into_validated(addr + 1, &mut out));
        assert!(accepted);
        assert_eq!(out.as_slice(), &all[1..8]);
    }

    #[test]
    fn refuses_null_unaligned_zero_and_oversized() {
        let mut data: Vec<u32> = vec![1, 2];
        let addr = data.as_ptr() as u64;
        let memory = memory_for(&mut data);
        with_guest_memory(&memory, || {
            assert_eq!(IdentityGuestMemory.read_dwords(0, 1), None, "null");
            assert_eq!(
                IdentityGuestMemory.read_dwords(addr + 2, 1),
                None,
                "unaligned"
            );
            assert_eq!(IdentityGuestMemory.read_dwords(addr, 0), None, "zero len");
            assert_eq!(
                IdentityGuestMemory.read_dwords(addr, MAX_READ_DWORDS + 1),
                None,
                "over the cap"
            );
        });
    }

    /// A chained command buffer is resource-sized, not pointer-sized: the PM4
    /// `IB_SIZE` field is 20 bits, so a legal chain target can be far larger
    /// than [`MAX_READ_DWORDS`]. Reading one through the pointer cap would
    /// refuse a valid frame's command stream as a mis-decode.
    #[test]
    fn a_command_buffer_read_uses_the_resource_cap_not_the_pointer_cap() {
        let oversized = MAX_READ_DWORDS as usize + 16;
        let mut data: Vec<u32> = (0..oversized as u32).collect();
        let addr = data.as_ptr() as u64;
        let memory = memory_for(&mut data);
        with_guest_memory(&memory, || {
            assert_eq!(
                IdentityGuestMemory.read_dwords(addr, oversized as u32),
                None,
                "the out-of-band POINTER read stays capped — a pointer that big is a mis-decode"
            );
            let got = IdentityGuestMemory.read_command_dwords(addr, oversized as u32);
            assert_eq!(
                got.as_ref().map(Vec::len),
                Some(oversized),
                "a command buffer past the pointer cap must still be readable in full"
            );
            // The same address/alignment/authority contract still applies.
            assert_eq!(IdentityGuestMemory.read_command_dwords(0, 4), None, "null");
            assert_eq!(
                IdentityGuestMemory.read_command_dwords(addr + 2, 4),
                None,
                "unaligned"
            );
            assert_eq!(
                IdentityGuestMemory.read_command_dwords(addr, 0),
                None,
                "zero length"
            );
            assert_eq!(
                IdentityGuestMemory.read_command_dwords(addr, MAX_RESOURCE_READ_DWORDS + 1),
                None,
                "past the resource cap"
            );
        });
    }

    #[test]
    fn writes_committed_host_memory_identity_mapped() {
        let mut data = vec![0u32; 4];
        let replacement = [1u32, 2, 3, 4];
        let bytes: Vec<u8> = replacement
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let addr = data.as_mut_ptr() as u64;
        let memory = memory_for(&mut data);
        assert!(with_guest_memory(&memory, || write_bytes_checked(
            addr, &bytes
        )));
        assert_eq!(data, replacement);
    }

    #[test]
    fn compute_image_guest_mirror_keeps_gpu_only_results_non_fatal() {
        let bytes = vec![0x11, 0x22, 0x33, 0x44];
        assert_eq!(
            mirror_compute_image_to_guest(0x4441_0000, bytes.clone(), "test-compute-image"),
            ComputeImageGuestMirror::GpuOnly,
            "a GPU-only image is retained even without a CPU guest mapping"
        );
        let denied: Arc<dyn GpuGuestMemory> = Arc::new(DenyGpuMemory);
        let shadow = with_guest_memory(&denied, || {
            read_bytes_validated(0x4441_0000, bytes.len() as u64)
        });
        assert_eq!(
            shadow,
            Some(bytes.clone()),
            "later GPU reads must observe the guest-layout compute result"
        );

        let mut destination = [0u8; 4];
        let addr = destination.as_mut_ptr() as u64;
        let memory: Arc<dyn GpuGuestMemory> = Arc::new(HostRange {
            start: addr,
            len: destination.len() as u64,
        });
        let publication = with_guest_memory(&memory, || {
            mirror_compute_image_to_guest(addr, bytes.clone(), "test-compute-image")
        });
        assert_eq!(publication, ComputeImageGuestMirror::Written);
        assert_eq!(destination, bytes.as_slice());
    }

    #[test]
    fn compute_presentation_rejects_non_scanout_atlases() {
        let scanout = 0x7f10_0000;
        let frame_bytes = 1920 * 1080 * 4;
        // VideoOut registers the conservative 8-B/px upper bound. Both an
        // RGBA16F image (the full span) and an RGBA8 image (half) match it.
        register_scanout_watch(scanout, (frame_bytes * 2) as u64);

        assert!(is_scanout_candidate(scanout, frame_bytes));
        assert!(
            is_scanout_candidate(0x7f90_0000, frame_bytes),
            "a full-size RGBA8 intermediate can be the missing final composite"
        );
        assert!(
            is_scanout_candidate(0x7f91_0000, frame_bytes * 2),
            "a full-size RGBA16F intermediate can be the missing final composite"
        );
        assert!(
            !is_scanout_candidate(0x7fa0_0000, 1024 * 1024 * 4),
            "a square compute atlas must never replace a widescreen frame"
        );
    }

    #[test]
    fn refuses_memory_without_process_authority() {
        let data = [1u32, 2, 3, 4];
        assert_eq!(
            IdentityGuestMemory.read_dwords(data.as_ptr() as u64, 4),
            None,
            "committed host memory outside a GuestProcess must be refused"
        );
    }

    /// A range that starts committed but runs into an uncommitted region must
    /// be refused as a whole.
    #[test]
    fn refuses_range_that_leaves_process_authority() {
        let mut data = vec![1u32, 2];
        let addr = data.as_ptr() as u64;
        let memory = memory_for(&mut data);
        with_guest_memory(&memory, || {
            assert_eq!(IdentityGuestMemory.read_dwords(addr, 4), None);
            assert!(IdentityGuestMemory.read_dwords(addr, 2).is_some());
        });
    }

    #[test]
    fn reports_the_exact_readable_prefix_before_authority_ends() {
        let mut data = vec![1u32, 2, 3];
        let addr = data.as_ptr() as u64;
        let bytes = std::mem::size_of_val(data.as_slice()) as u64;
        let memory = memory_for(&mut data);
        with_guest_memory(&memory, || {
            assert_eq!(readable_prefix(addr, bytes + 4096), bytes);
            assert_eq!(readable_prefix(addr + 4, bytes + 4096), bytes - 4);
            assert_eq!(readable_prefix(addr + bytes, 4096), 0);
            assert_eq!(readable_prefix(addr, 0), 0);
        });
    }
}
