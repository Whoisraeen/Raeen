//! AGC DCB execution against the Vulkan offscreen path.
//!
//! Two paths live here:
//!
//! - [`AgcGpuSession::execute_dcb_cp`] — **the title path**. Runs the DCB
//!   through [`kyty_graphics::run::CommandProcessor`], so the draw's extent,
//!   format, viewport, scissor, topology and shaders all come from decoded
//!   register state.
//! - [`AgcGpuSession::execute_dcb`] — **the M2 fixture path**, deprecated and
//!   retained only as the regression gate behind `tests/m2_agc_triangle.rs`. It
//!   ignores registers entirely and always renders the same hardcoded triangle.
//!
//! The fixture is deliberately still reachable: `tests/m2_agc_triangle.rs` pins
//! the M2 milestone and its DCB (two dwords, no register state) cannot draw
//! through a real command processor. Deleting the fixture path would take the
//! M2 gate with it.

use crate::agc::{self, AgcDecodeError, AgcSubmission};
use crate::backend::GpuBackend;
use crate::draw_translate::OffscreenDrawSink;
use crate::vulkan::{RenderedImage, VulkanBackend};
use kyty_graphics::pm4;
use kyty_graphics::run::{CommandProcessor, CpError, RunOutcome, SuspendedWait};
use parking_lot::{Mutex, RwLock};
use raeen_core::error::GpuError;
use raeen_core::frame_path::{self, Stage};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use thiserror::Error;
use tracing::{debug, warn};

/// Default offscreen size for PM4-triggered M2 draws.
pub const M2_DRAW_WIDTH: u32 = 64;
pub const M2_DRAW_HEIGHT: u32 = 64;

const IT_NOP: u32 = 0x10;
const R_DRAW_INDEX_AUTO: u32 = 0x04;

#[derive(Debug, Error)]
pub enum AgcExecError {
    #[error(transparent)]
    Decode(#[from] AgcDecodeError),
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error("PM4 command processor: {0}")]
    CommandProcessor(String),
    #[error("guest GPU address space is no longer available")]
    AddressSpaceUnavailable,
}

/// Always-on, low-overhead wall-clock breakdown for the latest completed
/// VideoOut frame. Values are microseconds so the child runner can copy them
/// through shared memory without floating-point or allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PresentTiming {
    pub worker_drain_us: u64,
    pub fence_wait_us: u64,
    pub readback_us: u64,
    pub srgb_encode_us: u64,
    /// Filled by the Shell after `ColorImage` construction and texture upload.
    pub egui_upload_us: u64,
}

/// How many submitted DCBs may be in flight before a submitter blocks.
///
/// Backpressure is not optional: a title submits faster than this session
/// renders, so an unbounded queue would grow without limit (a DCB is up to 4 MiB).
/// Blocking the submitter when the queue is full is also what real hardware does
/// when its ring buffer fills.
const SUBMIT_QUEUE_DEPTH: usize = 8;

/// Frames the guest's flip thread may have in flight before a flip blocks
/// (item 7 / rank 7 — THE fps lever). See [`FlipSemaphore`] and
/// [`AgcGpuSession::submit_flip_flush`].
///
/// `sceVideoOutSubmitFlip` no longer waits for the whole GPU worker to drain
/// (measured ~2.3 s per flip on ASTRO.BOT — full worker-drain + fence stall,
/// which pinned the title at ~0.4 fps). Instead the flip *enqueues* the
/// present flush behind the already-queued draws and returns, letting the
/// guest render the next frame while the worker catches up (CPU/GPU overlap).
///
/// The cap is the whole point. A naive unbounded `wait: false` was tried and
/// REVERTED: on Minecraft the unthrottled flip stream let the title's render
/// pool thread reuse display buffers whose flips had not completed, corrupting
/// the title's own flip state machine — the main thread wedged ~10 s into boot
/// on a title mutex held by that render pool thread (a pthread_sync "stuck >3s"
/// warn named it). Real hardware bounds this with a finite pool of display
/// buffers; this semaphore restores that backpressure. When `cap` flushes are
/// already outstanding the flip blocks — exactly the wait that made the old
/// synchronous form safe, but paid only when the guest is `cap` frames ahead
/// of the GPU rather than on every single flip. 2 gives double-buffered
/// pipelining (one frame presenting while the next is prepared) while still
/// bounding race-ahead tightly.
const FLIP_FRAMES_IN_FLIGHT: usize = 2;

/// Bounded async flip is the production default: the Phase 2.1 gate (three
/// consecutive 180-second Minecraft runs without the historical title-mutex
/// wedge) went green 2026-07-26 — runs `run-1785110215494` (12192 flips),
/// `run-1785111755329` (10432), and `run-1785111946064` (9920), all with
/// flips still flowing in the final window, zero "stuck >3s" warnings, and
/// per-flip worker drain of 16–24 µs. `RAEEN_ASYNC_FLIP=0` is the opt-out
/// back to the synchronous flip for A/B diagnosis of any future wedge.
fn async_flip_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    !value.is_some_and(|value| {
        matches!(
            value.to_string_lossy().trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

/// Counting semaphore gating how many flip flushes may be in flight
/// ([`FLIP_FRAMES_IN_FLIGHT`]). Permits are acquired by the guest's flip thread
/// in [`AgcGpuSession::submit_flip_flush`] and released from the GPU-completion
/// side (the worker drops the [`FlipPermit`] once the flush executes) — never
/// from the guest side, so the guest can never be the thread that must run to
/// free a permit. The worker only ever *releases*, never acquires, so there is
/// no acquire cycle and the worker always makes forward progress.
struct FlipSemaphore {
    /// Permits currently available (0..=`cap`).
    available: Mutex<usize>,
    cv: parking_lot::Condvar,
    cap: usize,
}

impl FlipSemaphore {
    fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self {
            available: Mutex::new(cap),
            cv: parking_lot::Condvar::new(),
            cap,
        })
    }

    /// Take a permit, blocking until one is free. Holds only this semaphore's
    /// own mutex while parked — never a guest-visible GPU lock — so a blocked
    /// flip thread cannot stall the worker that would release a permit.
    fn acquire(self: &Arc<Self>) -> FlipPermit {
        let mut available = self.available.lock();
        while *available == 0 {
            self.cv.wait(&mut available);
        }
        *available -= 1;
        FlipPermit(Arc::clone(self))
    }

    fn release(&self) {
        let mut available = self.available.lock();
        // Never exceed the cap: a stray double-release must not inflate the
        // in-flight budget.
        if *available < self.cap {
            *available += 1;
        }
        self.cv.notify_one();
    }
}

/// RAII permit from a [`FlipSemaphore`]. Dropping it returns the permit. It
/// rides into [`GpuWork::Flush`] and is dropped by the GPU worker after the
/// flush executes (the fence-signal equivalent on this single-consumer path);
/// on any inline fallback it drops on the calling thread instead. Either way
/// the permit is always eventually returned — even if the flush panics (the
/// permit is bound outside the worker's `catch_unwind`) or the worker shuts
/// down with the item still queued (dropping the receiver drops the item, and
/// thus the permit).
struct FlipPermit(Arc<FlipSemaphore>);

impl Drop for FlipPermit {
    fn drop(&mut self) {
        self.0.release();
    }
}

enum GpuWork {
    Submit(Vec<u32>, bool),
    /// Land pending deferred readbacks and (when `address` is `Some`) present
    /// the flipped scanout buffer. Queued through the same ordered channel as
    /// submissions so the flush runs AFTER every draw the title submitted
    /// before the flip — and on the GPU worker thread, never cross-thread.
    /// `done` is an optional rendezvous back to the requester: `wait_idle`
    /// blocks until the flush executed; a flip (`present_scanout`) passes
    /// `None` and returns immediately (stage C item 4) — the title's flip
    /// thread never pays for the readback, and presentation is at most one
    /// flip latent, which is standard swapchain behaviour.
    Flush {
        address: Option<u64>,
        /// When this ordered flush entered the worker queue. The delta at
        /// dequeue is the guest-visible worker-drain backlog.
        queued_at: std::time::Instant,
        done: Option<std::sync::mpsc::SyncSender<()>>,
        /// Frames-in-flight permit for the bounded fire-and-forget flip path
        /// (item 7 / rank 7). Held from before this flush is enqueued until
        /// the worker finishes it, then dropped on the worker thread — which
        /// releases the permit from the GPU-completion side. `None` for
        /// `wait_idle` and the synchronous fallback (they carry no budget).
        permit: Option<FlipPermit>,
    },
    #[cfg(test)]
    Panic,
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuLifecycle {
    Open,
    Closing,
    Closed,
}

/// A command buffer suspended mid-stream on an unmet `WAIT_REG_MEM` label.
///
/// Port of SharpEmu `GpuWaitRegistry.WaitingDcb` (GpuWaitRegistry.cs:19-40):
/// the buffer, where to resume in it, and the wait condition. The label is
/// NEVER force-satisfied — the buffer resumes only when a later submission's
/// writebacks (compute storage writebacks, DMA_DATA copies, WRITE_DATA) or a
/// direct guest store genuinely satisfy the condition.
struct SuspendedBuffer {
    words: Vec<u32>,
    resume_dword: usize,
    wait: kyty_graphics::run::WaitSpec,
    deferred_present: bool,
    /// How many producer re-check rounds have seen this wait still unmet —
    /// drives the rate-limited dead-wait warn.
    recheck_rounds: u64,
    /// Latched satisfied by [`AgcGpuSession::latch_produced_waits`] when a
    /// producer packet wrote a satisfying value THIS submission — even if the
    /// guest has since reset the label to arm the next frame (SharpEmu
    /// `GpuWaitRegistry.LatchSatisfiedByValue`). A live re-read at wake time
    /// would miss that transient window.
    latched: bool,
}

/// Per-queue wait state: at most one suspended buffer, plus the submissions
/// parked behind it. On hardware a wait blocks its own ring; later work on
/// the SAME queue must queue up behind it, not run ahead (SharpEmu's
/// `SubmittedDcbState.PendingSubmissions`, AgcExports.cs — validated in
/// `ValidateSubmittedQueueAndReleaseMemDecoders`).
#[derive(Default)]
struct QueueWaitState {
    suspended: Option<SuspendedBuffer>,
    /// `(words, deferred_present)` in submission order.
    pending: std::collections::VecDeque<(Vec<u32>, bool)>,
}

/// The two hardware rings this session models: graphics (DCB) and async
/// compute (ACB) — matching the two command processors.
#[derive(Default)]
struct WaitStates {
    graphics: QueueWaitState,
    compute: QueueWaitState,
}

impl WaitStates {
    fn queue_mut(&mut self, is_compute: bool) -> &mut QueueWaitState {
        if is_compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }
}

/// Fixed-point bound for the resume loop: a resumed buffer's own writebacks
/// can satisfy the other queue's wait, so re-checks loop — but never forever.
const MAX_RESUME_PASSES: u32 = 64;

/// After this many producer re-check rounds with the wait still unmet, warn
/// (then again at each doubling) so dead waits are visible in logs.
const STALE_WAIT_RECHECK_ROUNDS: u64 = 512;

/// After TWICE the "possible dead wait" window with the wait STILL unmet, the
/// label's producer is treated as one that will never run, and the buffer is
/// FORCE-RESUMED instead of left parked forever. A genuinely dead `WAIT_REG_MEM`
/// otherwise deadlocks the title's GPU submit worker and, through the mutex it
/// holds, the guest main thread parked behind it (measured: ASTRO.BOT
/// `mutex=0x300944e00 owner=21` stuck on a `dcb` label its producer never wrote).
/// Force-resuming degrades to a possible rendering glitch, never a permanent hang.
const DEAD_WAIT_FORCE_RESUME_ROUNDS: u64 = STALE_WAIT_RECHECK_ROUNDS * 2;

/// Cumulative wait/suspend counters (diagnostics + tests).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuWaitStats {
    /// Buffers that suspended on an unmet label wait.
    pub suspended: u64,
    /// Suspended buffers resumed after their label was genuinely written.
    pub resumed: u64,
    /// Buffers currently suspended (0..=2 — one per queue).
    pub currently_suspended: usize,
    /// Submissions currently parked behind a suspended queue.
    pub parked: usize,
}

impl From<CpError> for AgcExecError {
    fn from(error: CpError) -> Self {
        Self::CommandProcessor(error.to_string())
    }
}

struct InFlightCompletion<'a>(&'a AgcGpuSession);

impl Drop for InFlightCompletion<'_> {
    fn drop(&mut self) {
        self.0.finish_one();
    }
}

/// Process-global session: lazy Vulkan bring-up + last rendered image.
pub struct AgcGpuSession {
    /// Serializes admission against shutdown. Once closing begins, stale
    /// observer clones cannot enqueue work behind the worker's Shutdown marker.
    lifecycle: Mutex<GpuLifecycle>,
    /// Process-owned address-space authority. Holding this Arc keeps the guest
    /// arena alive until every asynchronous submission has completed.
    guest_memory: Mutex<Option<Arc<dyn crate::guest_mem::GpuGuestMemory>>>,
    /// DCBs handed to the GPU worker. Created on first submit; a single
    /// consumer keeps DCBs in submission order, which is required — register
    /// state carries across submissions, so reordering a state-only DCB past
    /// the draw DCB that depends on it renders with the wrong state.
    submit_queue: OnceLock<std::sync::mpsc::SyncSender<GpuWork>>,
    /// Submissions accepted but not yet executed, for [`AgcGpuSession::wait_idle`].
    in_flight: (Mutex<usize>, parking_lot::Condvar),
    backend: Mutex<Option<VulkanBackend>>,
    /// GPU register state persists across queue submissions. AGC commonly
    /// submits state-only DCBs before a later draw-only DCB.
    command_processor: Mutex<CommandProcessor>,
    /// Separate command processor for the ACB (asynchronous compute) queue. On
    /// hardware the ACE compute ring keeps its own register state, independent
    /// of the graphics DCB. Sharing one CP let an ACB `R_DISPATCH_RESET` (a
    /// compute-queue reset) zero the compute shader that graphics-DCB dispatches
    /// depend on — measured on ASTRO.BOT: DCB compute dispatches reaching
    /// translation collapsed from 1057 to 409 under a shared CP. Isolating the
    /// queues keeps each one's resets from clobbering the other.
    compute_command_processor: Mutex<CommandProcessor>,
    /// The frame the Shell presents, behind an [`Arc`] so the two hottest hand-
    /// offs cost a refcount bump instead of an 8 MB (1080p RGBA) memcpy: the
    /// flush store (`present_flipped`, on the guest flip thread's synchronous
    /// critical section) and the per-frame Shell read (`shell/present.rs` calls
    /// [`AgcGpuSession::last_image`] every repaint). The frame bytes are
    /// immutable once presented, so sharing them is sound and no consumer needs
    /// its own copy.
    last_image: Mutex<Option<Arc<RenderedImage>>>,
    /// System boot splash: the package's `sce_sys/pic0.png`, decoded at launch.
    /// While `Some`, [`AgcGpuSession::last_image`] presents it instead of any
    /// title frame — a real PS5 shows this image from launch until the title
    /// calls `sceSystemServiceHideSplashScreen`. It also comes down when the
    /// title flips to a buffer with real drawn content (SharpEmu's behavior),
    /// but NOT for the most-content present fallback: that path can surface a
    /// bare cleared render target, which is exactly what the splash exists to
    /// cover.
    ///
    /// Shared behind the same [`Arc`] as [`Self::last_image`]: the splash is a
    /// full decoded `pic0.png`, and staging + seeding + reset previously cloned
    /// it three times per launch.
    splash: Mutex<Option<Arc<RenderedImage>>>,
    draw_count: Mutex<u64>,
    /// ShaderMemory Phase 2: guest shader fetch+translate results, shared
    /// across DCBs so per-frame re-binds hit the cache instead of
    /// re-translating (and failures warn once per distinct shader, ever).
    shader_cache: Mutex<crate::shader_fetch::ShaderTranslateCache>,
    /// Draws skipped because a bound guest shader failed translation.
    shader_skip_count: Mutex<u64>,
    /// Persistent per-render-target pixels (keyed by `CB_COLOR0_BASE`), so
    /// draws compose into a frame across DCBs instead of each starting from
    /// a cleared attachment.
    framebuffers: Mutex<std::collections::HashMap<u64, Arc<RenderedImage>>>,
    /// ABI-v3 GPU-plugin results keyed by the native render-target base.
    /// Presentation may use these, but they never replace native framebuffer
    /// pixels used to seed later guest draws.
    gpu_present_overrides: Mutex<std::collections::HashMap<u64, Arc<RenderedImage>>>,
    /// Guest display-buffer address the title last flipped to
    /// (`sceVideoOutSubmitFlip` → `present_scanout`). When set and a render
    /// target with this base exists, it — not the last-drawn target — is what
    /// the Shell presents. `None` preserves the last-drawn baseline.
    scanout_address: Mutex<Option<u64>>,
    /// Layout of the last-flipped guest display buffer, for
    /// present-from-guest-memory (M3): when the flip address has no GPU-drawn
    /// render target, the raw guest bytes at that address are read as pixels
    /// using this descriptor (CPU-drawn 2D). `None` disables the guest-memory
    /// path (the flip carried no attribute).
    scanout_descriptor: Mutex<Option<raeen_core::subsystems::ScanoutDescriptor>>,
    /// Last compute shader bound on either queue, carried across submissions.
    /// The title binds it on the graphics DCB and dispatches it on the ACB
    /// (whose buffers are dispatch-only), so an ACB dispatch that arrives with a
    /// null shader falls back to this. Seeded into the sink before a submission
    /// and read back after (see `execute_dcb_cp`).
    last_compute_shader: Mutex<Option<kyty_graphics::hw_regs::ComputeShaderInfo>>,
    submission_count: AtomicU64,
    /// Cross-queue WAIT_REG_MEM suspend/resume state (SharpEmu
    /// `GpuWaitRegistry` port). See [`SuspendedBuffer`].
    wait_states: Mutex<WaitStates>,
    /// Buffers that ever suspended on an unmet label wait.
    wait_suspended_total: AtomicU64,
    /// Suspended buffers resumed by a genuine label write.
    wait_resumed_total: AtomicU64,
    /// The guest base the most-content fallback last presented. Steady state
    /// on a title whose flip address never matches a drawn target (its scanout
    /// is filled by an uncaptured copy/DMA): each flip reads back ONLY
    /// {flip address, this base} instead of every dirty target — measured on
    /// ASTRO.BOT, the difference between a 15-target 54 ms flush and a
    /// 1-target one. Re-elected by a full census every
    /// [`FALLBACK_REELECT_INTERVAL`] flip misses.
    fallback_present_base: Mutex<Option<u64>>,
    /// Flips whose address had no drawn content (drives fallback re-election).
    flip_miss_count: AtomicU64,
    /// Bounds how many flip present-flushes may be outstanding (item 7 / rank
    /// 7 — THE fps lever). A flip acquires a permit before enqueuing its async
    /// flush and the worker releases it when the flush completes, so the flip
    /// returns at vblank cadence instead of inheriting the full worker-drain
    /// latency, yet the guest can never race more than [`FLIP_FRAMES_IN_FLIGHT`]
    /// frames ahead of the GPU. See [`AgcGpuSession::submit_flip_flush`].
    flip_permits: Arc<FlipSemaphore>,
    /// Bounded async-flip switch. Default ON since 2026-07-26 (the three-run
    /// no-wedge Minecraft gate went green); `RAEEN_ASYNC_FLIP=0` opts back
    /// out to the synchronous flip for A/B diagnosis.
    async_flip: bool,
    /// Monotonic counter advanced every time a COMPLETE frame is published to
    /// [`Self::last_image`] (or a splash transition changes what
    /// [`Self::last_image`] returns). The Shell (`shell/present.rs`) refreshes
    /// its egui texture exactly when this advances, so it always shows the
    /// freshest completed frame the GPU worker has finished — instead of
    /// chasing the guest-side VideoOut flip counter, which with the bounded
    /// async flip races ahead of the frames actually read back. This is the
    /// "read the freshest completed frame" the async present path needs so it
    /// never shows a stale/black frame. See [`Self::present_epoch`].
    present_epoch: AtomicU64,
    /// Guest display-buffer byte snapshots taken on the flip thread (keyed by
    /// flip address) so the async present-from-guest-memory path reads a
    /// COMPLETE frame captured at flip time — not live guest memory the title
    /// has since begun reusing for its next frame. This is what keeps the
    /// bounded async flip from ever presenting a partially-cleared/reused
    /// buffer (the regression that black-screened the earlier fire-and-forget
    /// flip). Bounded to [`MAX_SCANOUT_SNAPSHOTS`] entries (a title cycles
    /// through only a handful of display buffers). See [`Self::present_scanout`].
    guest_scanout_snapshots: Mutex<std::collections::HashMap<u64, Arc<Vec<u8>>>>,
    /// Child-runner -> Shell shared-memory publisher. `None` for the Shell's
    /// bootstrap session and in-process tests; the isolated runner receives
    /// the mapping name through [`crate::frame_ipc::FRAME_IPC_ENV`].
    frame_publisher: Option<crate::frame_ipc::FrameIpcPublisher>,
    /// Latest completed-frame timing, stored as independent relaxed atomics so
    /// the HUD read and child IPC publish need no mutex on the flip hot path.
    present_worker_drain_us: AtomicU64,
    present_fence_wait_us: AtomicU64,
    present_readback_us: AtomicU64,
    present_srgb_encode_us: AtomicU64,
    /// One-entry-per-target cache of HDR→sRGB present encodes, keyed by the
    /// source `Arc`'s identity. A title that flips without redrawing (menus,
    /// loading screens, the remembered-fallback present path) re-presents the
    /// same `Arc<RenderedImage>` for many consecutive flips; re-encoding its
    /// 8.3 Mpx every flip was pure latency. The `Weak` validates the key
    /// against pointer reuse (ABA): a hit requires upgrading to the SAME Arc.
    /// Bounded to a handful of entries — a display-buffer ring plus the
    /// fallback target is all that recurs.
    present_encode_cache: Mutex<PresentEncodeCache>,
}

/// Cached HDR→sRGB present encodes: (source `Arc` pointer key, weak source for
/// ABA validation, encoded image). See [`AgcGpuSession::present_encode_cache`].
type PresentEncodeCache = Vec<(usize, Weak<RenderedImage>, Arc<RenderedImage>)>;

/// Cap on how many live guest-scanout snapshots
/// ([`AgcGpuSession::guest_scanout_snapshots`]) are retained. A title flips
/// through a small display-buffer ring (2–3 buffers); this bounds retired
/// buffers from accumulating. When exceeded the map is cleared, degrading at
/// most one later flip to a live read (the pre-async behaviour), never leaking.
const MAX_SCANOUT_SNAPSHOTS: usize = 8;

/// Count pixels with visible colour, optionally sampling every `stride`th
/// pixel. Alpha alone is not visible over the Shell's black background and
/// must not make a cleared RGBA target look content-bearing.
fn visible_pixel_count(image: &RenderedImage, stride: usize) -> usize {
    let bpp = image.bytes_per_pixel.max(1) as usize;
    let colour = colour_span(bpp);
    image
        .pixels
        .chunks_exact(bpp)
        .step_by(stride.max(1))
        .filter(|pixel| pixel.iter().take(colour).any(|&channel| channel != 0))
        .count()
}

/// Leading bytes of one texel that carry COLOUR, given the texel's size. Alpha
/// is excluded on purpose: alpha alone is not visible over the Shell's black
/// background and must not make a cleared target look content-bearing.
///
/// RGBA16F (`bytes_per_pixel == 8` — the HDR target [`to_presentable`] converts)
/// spends TWO bytes per channel, so its colour span is the first SIX. Testing
/// three would cover red plus only the low half of green and never look at blue
/// at all; half-float `1.0` is `0x3C00`, whose little-endian low byte is zero,
/// so a pure-green or blue-dominant HDR frame scanned as entirely black and was
/// discarded as "never drawn".
const fn colour_span(bytes_per_pixel: usize) -> usize {
    match bytes_per_pixel {
        // R16G16B16A16: R, G, B are the first three 16-bit channels.
        8 => 6,
        // RGBA8 / BGRA8: skip the alpha byte.
        4 => 3,
        // Anything else carries no alpha to exclude.
        other => other,
    }
}

fn has_visible_content(image: &RenderedImage) -> bool {
    visible_pixel_count(image, 16) != 0
}

/// Whether every colour texel is identical (alpha is ignored).
///
/// A uniform scanout can be a legitimate fade/clear, but it is not evidence
/// that a missing final composite landed. Minecraft's second screen draws a
/// light-gray clear into both VideoOut buffers while its actual menu lives in a
/// non-uniform intermediate target. Treating any non-black scanout as
/// authoritative hid that menu.
fn is_visually_uniform(image: &RenderedImage) -> bool {
    let bpp = image.bytes_per_pixel.max(1) as usize;
    let colour = colour_span(bpp).min(bpp);
    let mut pixels = image.pixels.chunks_exact(bpp);
    let Some(first) = pixels.next() else {
        return true;
    };
    pixels.all(|pixel| pixel[..colour] == first[..colour])
}

/// Elect a content-bearing, non-uniform intermediate target.
///
/// Uniform clears are deliberately excluded even when every pixel is nonzero:
/// their old visible-pixel score always beat a partially occupied UI target.
fn elect_detailed_target(
    framebuffers: &std::collections::HashMap<u64, Arc<RenderedImage>>,
    exclude: Option<u64>,
) -> Option<(u64, Arc<RenderedImage>)> {
    framebuffers
        .iter()
        .filter(|(base, image)| {
            // Election is a rare routing census, not the steady-state flip
            // path. Use an exact visibility check so a sparse UI cannot
            // disappear merely because every 16th sampled texel is black.
            Some(**base) != exclude
                && visible_pixel_count(image, 1) != 0
                && !is_visually_uniform(image)
        })
        .map(|(base, image)| (visible_pixel_count(image, 16), *base, image))
        .max_by_key(|(nonzero, _, _)| *nonzero)
        .map(|(_, base, image)| (base, Arc::clone(image)))
}

/// Select the smallest target set a flush consumer needs on the CPU.
///
/// A VideoOut flip names the new scanout directly and also needs the elected
/// intermediate target when the final copy is not decoded. A suspend point
/// (`requested_scanout == None`) needs command/compute completion but no CPU
/// pixels; selecting no targets lets it fence pending work without stealing
/// the flip's one filtered readback. Other touched targets stay GPU-side and
/// the existing flip-miss census reads them only when routing genuinely moves.
fn presentation_filter_bases(requested_scanout: Option<u64>, fallback: Option<u64>) -> Vec<u64> {
    let mut bases = Vec::with_capacity(2);
    let Some(base) = requested_scanout.filter(|base| *base != 0) else {
        return bases;
    };
    bases.push(base);
    if let Some(base) = fallback.filter(|base| *base != 0 && !bases.contains(base)) {
        bases.push(base);
    }
    bases
}

/// Whether a submission must fence compute immediately instead of letting the
/// ordered worker batch it to a real lifetime boundary.
const fn submission_compute_flush_required(queued_compute: bool, deferred_present: bool) -> bool {
    queued_compute && !deferred_present
}

/// Every Nth flip miss, the most-content fallback re-runs its full census
/// (full flush + scan) instead of trusting the remembered target, so content
/// migrating to a different render target is picked up within N flips.
const FALLBACK_REELECT_INTERVAL: u64 = 64;

/// Rate-limited warn for a flip whose display buffer uses a tiling mode or
/// pixel format the present-from-guest-memory path does not model. The frame is
/// skipped (never faked); the last presented frame stays up.
fn warn_unsupported_scanout(desc: &raeen_core::subsystems::ScanoutDescriptor) {
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_power_of_two() {
        warn!(
            tiling_mode = desc.tiling_mode,
            pixel_format = format_args!("{:#018x}", desc.pixel_format),
            width = desc.width,
            height = desc.height,
            pitch_pixels = desc.pitch_pixels,
            "present-from-guest-memory: unsupported scanout tiling/format — skipped"
        );
    }
}

/// Cloneable ownership handle for one guest process's GPU state. The Shell
/// observes the installed handle, while the runtime owns another clone; Kyty's
/// command processor remains private inside [`AgcGpuSession`].
#[derive(Clone)]
pub struct GpuProcessSession(Arc<AgcGpuSession>);

impl std::ops::Deref for GpuProcessSession {
    type Target = AgcGpuSession;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GpuProcessSession {
    #[must_use]
    fn create(memory: Arc<dyn crate::guest_mem::GpuGuestMemory>) -> Self {
        Self::create_with_frame_publisher(memory, false)
    }

    #[must_use]
    fn create_with_frame_publisher(
        memory: Arc<dyn crate::guest_mem::GpuGuestMemory>,
        publish_to_shell: bool,
    ) -> Self {
        let mut session = AgcGpuSession::new(memory);
        // Only the real process session publishes to the isolated-runner
        // bridge. The child also lazily creates a bootstrap/global session;
        // attaching both would violate the bridge's single-writer contract.
        if publish_to_shell {
            session.frame_publisher = crate::frame_ipc::FrameIpcPublisher::open_from_env();
        }
        // Seed the boot splash the launcher staged for this launch (cloned,
        // not taken: an unrelated session created concurrently must not steal
        // the splash from the launch it was staged for).
        *session.splash.lock() = pending_splash().lock().clone();
        Self(Arc::new(session))
    }

    /// Asynchronously submit through this process's bounded, ordered queue.
    pub fn submit_dcb_async(&self, words: Vec<u32>, is_compute: bool) {
        AgcGpuSession::submit_dcb_async_owned(&self.0, words, is_compute);
    }

    /// Drain and stop this process's submission worker. The runtime calls this
    /// before unmapping guest memory, so no queued PM4 work can retain guest
    /// pointers past process teardown.
    pub fn shutdown(&self) {
        {
            let mut lifecycle = self.lifecycle.lock();
            if *lifecycle != GpuLifecycle::Open {
                return;
            }
            *lifecycle = GpuLifecycle::Closing;
        }
        self.wait_idle();
        let waits = self.wait_suspend_stats();
        if waits.currently_suspended > 0 || waits.parked > 0 {
            warn!(
                suspended = waits.currently_suspended,
                parked = waits.parked,
                "GPU shutdown with label waits still unmet — their producers never ran"
            );
        }
        if let Some(sender) = self.submit_queue.get() {
            let _ = sender.send(GpuWork::Shutdown);
        }
        *self.guest_memory.lock() = None;
        *self.lifecycle.lock() = GpuLifecycle::Closed;
    }
}

impl raeen_core::subsystems::GpuSubmissionSubsystem for GpuProcessSession {
    fn submit(&self, words: Vec<u32>, queue: raeen_core::subsystems::GpuQueue) {
        self.submit_dcb_async(
            words,
            matches!(queue, raeen_core::subsystems::GpuQueue::AsyncCompute),
        );
    }

    fn map_shader_metadata(
        &self,
        code_address: u64,
        data: raeen_core::subsystems::ShaderMappedData,
    ) {
        AgcGpuSession::map_shader_metadata(&self.0, code_address, data);
    }

    fn present_scanout(
        &self,
        address: u64,
        descriptor: Option<raeen_core::subsystems::ScanoutDescriptor>,
    ) {
        AgcGpuSession::present_scanout(&self.0, address, descriptor);
    }

    fn wait_idle(&self) {
        AgcGpuSession::wait_idle(self);
    }

    fn hide_splash(&self) {
        AgcGpuSession::hide_splash(self);
    }

    fn stats(&self) -> raeen_core::subsystems::GpuSubmissionStats {
        raeen_core::subsystems::GpuSubmissionStats {
            submitted: self.submission_count.load(Ordering::Relaxed),
            completed_draws: self.draw_count(),
            skipped_shaders: self.shader_skip_count(),
        }
    }
}

impl AgcGpuSession {
    fn new(memory: Arc<dyn crate::guest_mem::GpuGuestMemory>) -> Self {
        Self::new_with_async_flip(
            memory,
            async_flip_enabled(std::env::var_os("RAEEN_ASYNC_FLIP").as_deref()),
        )
    }

    fn new_with_async_flip(
        memory: Arc<dyn crate::guest_mem::GpuGuestMemory>,
        async_flip: bool,
    ) -> Self {
        Self {
            lifecycle: Mutex::new(GpuLifecycle::Open),
            guest_memory: Mutex::new(Some(memory)),
            submit_queue: OnceLock::new(),
            in_flight: (Mutex::new(0), parking_lot::Condvar::new()),
            backend: Mutex::new(None),
            command_processor: Mutex::new(Self::new_command_processor()),
            compute_command_processor: Mutex::new(Self::new_command_processor()),
            last_image: Mutex::new(None),
            splash: Mutex::new(None),
            draw_count: Mutex::new(0),
            shader_cache: Mutex::new(crate::shader_fetch::ShaderTranslateCache::new()),
            shader_skip_count: Mutex::new(0),
            framebuffers: Mutex::new(std::collections::HashMap::new()),
            gpu_present_overrides: Mutex::new(std::collections::HashMap::new()),
            scanout_address: Mutex::new(None),
            scanout_descriptor: Mutex::new(None),
            last_compute_shader: Mutex::new(None),
            wait_states: Mutex::new(WaitStates::default()),
            wait_suspended_total: AtomicU64::new(0),
            wait_resumed_total: AtomicU64::new(0),
            submission_count: AtomicU64::new(0),
            fallback_present_base: Mutex::new(None),
            flip_miss_count: AtomicU64::new(0),
            flip_permits: FlipSemaphore::new(FLIP_FRAMES_IN_FLIGHT),
            async_flip,
            present_epoch: AtomicU64::new(0),
            guest_scanout_snapshots: Mutex::new(std::collections::HashMap::new()),
            frame_publisher: None,
            present_worker_drain_us: AtomicU64::new(0),
            present_fence_wait_us: AtomicU64::new(0),
            present_readback_us: AtomicU64::new(0),
            present_srgb_encode_us: AtomicU64::new(0),
            present_encode_cache: Mutex::new(Vec::new()),
        }
    }

    /// A session command processor, with the unified GPU clock plumbed in:
    /// the installed source consults `RAEEN_UNIFIED_GPU_CLOCK` per call and
    /// DECLINES when the gate is off, so default behavior is bit-identical to
    /// the CP's legacy process-local release counter (see `crate::gpu_clock`).
    fn new_command_processor() -> CommandProcessor {
        let mut cp = CommandProcessor::new();
        cp.set_timestamp_source(Some(crate::gpu_clock::cp_timestamp_source));
        cp
    }

    /// Create isolated GPU state for a new guest process.
    #[must_use]
    pub fn new_process(memory: Arc<dyn crate::guest_mem::GpuGuestMemory>) -> GpuProcessSession {
        GpuProcessSession::create_with_frame_publisher(memory, true)
    }

    /// Make `session` visible to Shell presentation without transferring the
    /// runtime's ownership. A new launch replaces the observer handle and does
    /// not inherit prior command-processor/register/framebuffer state.
    pub fn install_process(session: &GpuProcessSession) {
        *current_gpu_session().write() = session.clone();
    }

    /// Currently installed process session. Before the first launch this is a
    /// clean bootstrap session used by GPU acceptance tests.
    #[must_use]
    pub fn global() -> GpuProcessSession {
        current_gpu_session().read().clone()
    }

    /// How many PM4-triggered draws have completed successfully.
    pub fn draw_count(&self) -> u64 {
        *self.draw_count.lock()
    }

    /// Monotonic count of published complete frames (and splash transitions).
    /// The Shell refreshes its on-screen texture when this advances, so it
    /// always shows the freshest COMPLETE frame the GPU worker has finished —
    /// never a half-read async frame, and never chasing the guest-side flip
    /// counter that races ahead of the worker under the bounded async flip.
    /// See the [`present_epoch`](Self::present_epoch) field.
    pub fn present_epoch(&self) -> u64 {
        self.present_epoch.load(Ordering::Relaxed)
    }

    /// Timing breakdown attached to the newest completed frame.
    #[must_use]
    pub fn present_timing(&self) -> PresentTiming {
        PresentTiming {
            worker_drain_us: self.present_worker_drain_us.load(Ordering::Relaxed),
            fence_wait_us: self.present_fence_wait_us.load(Ordering::Relaxed),
            readback_us: self.present_readback_us.load(Ordering::Relaxed),
            srgb_encode_us: self.present_srgb_encode_us.load(Ordering::Relaxed),
            egui_upload_us: 0,
        }
    }

    /// Publish a fully-read-back (fence-complete) or snapshot-complete frame as
    /// the presented image and advance [`Self::present_epoch`] so the Shell
    /// refreshes. This is the ONLY way a title frame reaches `last_image`, so
    /// the invariant "every presented frame is a COMPLETE frame" holds by
    /// construction — the guarantee the async flip must not break.
    fn publish_frame(&self, presented: Arc<RenderedImage>) {
        puffin::profile_function!();
        // Offer the finished frame to the active present plugin (upscaler /
        // frame-gen). The default is a zero-cost identity — with no plugin
        // selected this returns the same `Arc`, so the "every presented frame is
        // COMPLETE" invariant and the default present cost are both unchanged.
        let presented = crate::present_plugin::apply_to_image(presented);
        self.publish_completed_frame(presented);
    }

    /// Publish a frame already processed by the active ABI-v3 GPU plugin.
    /// Re-running its CPU compatibility entry would duplicate the upscale.
    fn publish_gpu_frame(&self, presented: Arc<RenderedImage>) {
        self.publish_completed_frame(presented);
    }

    fn publish_completed_frame(&self, presented: Arc<RenderedImage>) {
        // The last rung of the frame path, and the only one that means a pixel
        // was actually produced. Every publish route funnels through here.
        frame_path::record(Stage::FramePublished);
        if let Some(publisher) = &self.frame_publisher {
            publisher.publish(&presented, self.present_timing());
        }
        *self.last_image.lock() = Some(presented);
        self.present_epoch.fetch_add(1, Ordering::Relaxed);
    }

    /// Advance the present epoch without changing `last_image` — for splash
    /// transitions, which change what [`Self::last_image`] returns (splash vs.
    /// title frame) without writing the `last_image` field, so the Shell still
    /// needs a refresh signal.
    fn bump_present_epoch(&self) {
        self.present_epoch.fetch_add(1, Ordering::Relaxed);
    }

    /// Guest shader fetch/translate counters (ShaderMemory Phase 2).
    pub fn shader_stats(&self) -> crate::shader_fetch::ShaderCacheStats {
        self.shader_cache.lock().stats()
    }

    /// Draws skipped because a bound guest shader failed translation.
    pub fn shader_skip_count(&self) -> u64 {
        *self.shader_skip_count.lock()
    }

    /// Publish the owned shader metadata produced by AGC shader creation.
    pub fn map_shader_metadata(&self, code_address: u64, data: crate::contracts::ShaderMappedData) {
        self.shader_cache
            .lock()
            .map_shader_metadata(code_address, crate::contracts::mapped_data_to_kyty(data));
    }

    /// The frame to present: the boot splash while it is up, otherwise the
    /// last image produced by a draw-bearing DCB, if any. Diagnostics
    /// (frame dumps) bypass this and always see the title's own output.
    pub fn last_image(&self) -> Option<Arc<RenderedImage>> {
        // Both clones are `Arc` refcount bumps, not frame copies — the Shell
        // calls this every repaint, so an 8 MB memcpy here was pure waste.
        if let Some(splash) = self.splash.lock().clone() {
            return Some(splash);
        }
        self.last_image.lock().clone()
    }

    /// Drop every trace of the previously-presented title and (re)apply
    /// `splash` as the frame to show. Called when a new title is staged so its
    /// launch never opens on the prior title's splash or last drawn frame —
    /// including the case where the new title fails to launch and never
    /// installs a session of its own (an encrypted retail SELF), leaving this
    /// already-installed session in place.
    fn reset_presentation(&self, splash: Option<Arc<RenderedImage>>) {
        *self.splash.lock() = splash;
        *self.last_image.lock() = None;
        self.framebuffers.lock().clear();
        self.gpu_present_overrides.lock().clear();
        *self.scanout_address.lock() = None;
        *self.draw_count.lock() = 0;
        // Drop stale flip-time snapshots from the previous title so they can
        // never be presented under the new one.
        self.guest_scanout_snapshots.lock().clear();
        // What `last_image()` returns just changed (new splash, or nothing) —
        // signal the Shell to refresh.
        self.bump_present_epoch();
    }

    /// Stage the boot splash for the next launched process (see
    /// [`pending_splash`]). Call with `None` when the package has no
    /// `pic0.png`, so the previous launch's splash cannot carry over.
    ///
    /// This also resets the *currently-installed* session's presentation right
    /// away: a launch that fails before creating its own session (e.g. an
    /// encrypted retail SELF) would otherwise keep [`global`] pointed at the
    /// previous title, leaking its splash under the new title's overlay. The
    /// new process's own session, once created, seeds the same staged splash
    /// at construction (see [`GpuProcessSession::create`]).
    ///
    /// [`global`]: AgcGpuSession::global
    pub fn set_pending_splash(image: Option<RenderedImage>) {
        // Wrap once so the pending slot and the current session's reset SHARE
        // one allocation instead of each cloning the full decoded pic0.png. The
        // public API still takes an owned `RenderedImage` — the launcher hands
        // off a freshly-decoded image and does not keep it.
        let image = image.map(Arc::new);
        *pending_splash().lock() = image.clone();
        let current = current_gpu_session().read().clone();
        current.reset_presentation(image);
    }

    /// Mirror the user's GPU settings (Settings ▸ Video ▸ Validation Layers /
    /// Resolution Scale) into the GPU crate. The Shell calls this once at
    /// startup; the values are read when the Vulkan backend is created
    /// (validation) and when each guest draw is sized (resolution scale).
    pub fn set_runtime_config(
        validation_layers: bool,
        resolution_scale: f32,
        gpu_device_index: u32,
        shader_cache: bool,
        shader_cache_dir: std::path::PathBuf,
    ) {
        *gpu_runtime_config().write() = GpuRuntimeConfig {
            validation_layers,
            resolution_scale,
            gpu_device_index,
            shader_cache,
            shader_cache_dir,
        };
    }

    /// The current process-wide GPU settings (see [`Self::set_runtime_config`]).
    pub(crate) fn runtime_config() -> GpuRuntimeConfig {
        gpu_runtime_config().read().clone()
    }

    /// Register a present-path plugin (upscaler / frame generator). Out-of-tree,
    /// user-supplied plugin crates — an FSR/XeSS pass, or a BYO DLSS shim Raeen
    /// never ships or fetches — call this to add themselves. The plugin then
    /// becomes selectable by [`Self::select_present_plugin`]. See
    /// [`crate::present_plugin`] for the clean-room/license boundary.
    pub fn register_present_plugin(plugin: Box<dyn crate::present_plugin::PresentPlugin>) {
        crate::present_plugin::register(plugin);
    }

    /// Load and register every out-of-tree present plugin in `dir`, returning
    /// the names registered. Nothing is *activated* — selection stays the
    /// user's choice via [`Self::select_present_plugin`].
    ///
    /// This is the runtime (`dlopen`/`LoadLibrary`) path: a plugin is a separate
    /// binary the **user** supplies, never linked into the distributed Raeen
    /// artifact. A missing directory is not an error (having no `plugins/` is
    /// the normal case); every refused candidate is logged with its reason.
    ///
    /// # Safety
    ///
    /// Executes arbitrary native code from `dir` inside this process. Only call
    /// this on a directory whose contents the user controls — see
    /// [`crate::present_plugin::cabi`] for the full trust boundary.
    pub unsafe fn load_present_plugins_from(dir: &std::path::Path) -> Vec<String> {
        // SAFETY: delegated to this function's caller.
        unsafe { crate::present_plugin::cabi::load_and_register_dir(dir) }
    }

    /// Activate a registered present plugin by name; returns `false` (changing
    /// nothing) if no plugin with that name is registered.
    pub fn select_present_plugin(name: &str) -> bool {
        crate::present_plugin::select(name)
    }

    /// Restore the zero-cost identity present path (no active plugin).
    pub fn clear_present_plugin() {
        crate::present_plugin::select_none();
    }

    /// `(name, capabilities)` for every registered present plugin — for a
    /// Settings ▸ Video dropdown.
    #[must_use]
    pub fn present_plugins() -> Vec<(String, crate::present_plugin::Capabilities)> {
        crate::present_plugin::list()
    }

    /// Full plugin descriptions (name, capabilities, source path) for every
    /// registered present plugin — for the Shell's Plugins UI.
    #[must_use]
    pub fn present_plugin_infos() -> Vec<crate::present_plugin::PluginInfo> {
        crate::present_plugin::list_info()
    }

    /// Why each refused candidate in the latest `plugins/` scan was not
    /// loaded, one human-readable line per refusal.
    #[must_use]
    pub fn present_plugin_load_failures() -> Vec<String> {
        crate::present_plugin::load_failures()
    }

    /// The active present plugin's name, or `None` for the identity path.
    #[must_use]
    pub fn active_present_plugin() -> Option<String> {
        crate::present_plugin::active()
    }

    /// Set the present-time upscale factor an active upscaler should target
    /// (`1.0` = native). Distinct from the *render* resolution scale.
    pub fn set_present_output_scale(scale: f32) {
        crate::present_plugin::set_output_scale(scale);
    }

    /// Take the boot splash down (`sceSystemServiceHideSplashScreen`, or a
    /// flip to a buffer with real drawn content).
    pub fn hide_splash(&self) {
        *self.splash.lock() = None;
        // The splash coming down changes what `last_image()` returns (the title
        // frame now shows through) — refresh the Shell even if `last_image`
        // itself was not just rewritten.
        self.bump_present_epoch();
    }

    /// Select the guest display buffer the title flipped to
    /// (`sceVideoOutSubmitFlip`) as the presented frame. A title composites its
    /// UI across several render targets and flips to *one* of them; the
    /// last-drawn target is often the black background, not that buffer. If a
    /// render target with this base address has already been drawn, present it
    /// now; also remember the address so a later draw to it (the async GPU
    /// worker may not have finished the composite when the flip arrives)
    /// becomes the presented frame. Never regresses the last-drawn baseline —
    /// when the buffer has no drawn content, the current image is kept.
    pub fn present_scanout(
        &self,
        address: u64,
        descriptor: Option<raeen_core::subsystems::ScanoutDescriptor>,
    ) {
        if address == 0 {
            return;
        }
        // The isolated Shell cannot observe this runner's kernel flip counter.
        // Advance a dedicated shared counter for every real VideoOut flip,
        // independently of whether frame pixels changed.
        if let Some(publisher) = &self.frame_publisher {
            publisher.mark_presented();
        }
        *self.scanout_address.lock() = Some(address);
        *self.scanout_descriptor.lock() = descriptor;
        // Watch this flipped display buffer so a later DMA copy or compute
        // writeback into its frame region is reported by the scene→scanout fill
        // trace (task #5). The span is the descriptor's frame byte size so an
        // unrelated sub-allocation sitting past the frame is not mis-blamed
        // (measured: an 88 MiB-offset compute write near a 33 MiB 4K frame).
        // Assume ≤8 B/px (covers RGBA16F); a smaller true format only overshoots.
        let watch_span = descriptor
            .map(|d| u64::from(d.pitch_pixels.max(d.width)) * u64::from(d.height) * 8)
            .unwrap_or(0);
        crate::guest_mem::register_scanout_watch(address, watch_span);
        // Capture the guest display buffer NOW, on the flip thread, while its
        // bytes are the frame the title just finished. The async flush enqueued
        // below runs LATER on the GPU worker, by which time the title may have
        // begun reusing this buffer for its next frame — so the present-from-
        // guest-memory path (a CPU-/DMA-composed scanout with no GPU render
        // target at the flip address) must read this snapshot, not live memory.
        // Snapshot only when no drawn GPU target already backs this address:
        // GPU-target titles keep their frame in a render target read back under
        // a fence (ordering-complete on the single worker), so they never need
        // it and pay no snapshot cost. This is the fix for the earlier fire-
        // and-forget flip presenting a partially-cleared/reused buffer.
        // Read `framebuffers` in its own statement so the guard is dropped
        // before the snapshot read below (never held across another lock).
        let backed_by_target = self.framebuffers.lock().contains_key(&address);
        if self.async_flip
            && let Some(desc) = descriptor
            && !backed_by_target
            && let Some(bytes) = self.read_scanout_bytes(address, &desc)
        {
            let mut snaps = self.guest_scanout_snapshots.lock();
            if snaps.len() >= MAX_SCANOUT_SNAPSHOTS {
                snaps.clear();
            }
            snaps.insert(address, Arc::new(bytes));
        }
        if self.async_flip {
            // Enqueue the present flush and return WITHOUT waiting for the GPU
            // worker to drain it — the bounded fire-and-forget flip (item 7 /
            // rank 7, THE fps lever). The guest's flip thread no longer pays
            // the per-flip readback + `wait_for_fences` stall; the worker reads
            // back and presents in the background while the guest builds the
            // next frame, bounded to `FLIP_FRAMES_IN_FLIGHT` outstanding
            // flushes. The Shell shows the most recent COMPLETE frame the
            // worker has published (`present_epoch` drives its texture
            // refresh), so it can never surface a half-read frame.
            self.submit_flip_flush(address);
        } else {
            // Synchronous completion contract, kept as the `RAEEN_ASYNC_FLIP=0`
            // A/B fallback: the flip waits for the whole worker drain.
            self.consume_flush(Some(address), true);
        }
    }

    /// Read the raw bytes of a guest display buffer for present-from-guest-
    /// memory: `pitch * height * 4` bytes at `address`, bounds- and
    /// authority-checked against guest memory. `None` when the descriptor is
    /// degenerate, the geometry is out of range, or the guest range is
    /// unreadable. Shared by the flip-time snapshot ([`Self::present_scanout`])
    /// and the live fallback ([`Self::present_from_guest_memory`]). The tiling
    /// mode and pixel format are validated by the caller (the snapshot is a raw
    /// byte copy; only the size matters here).
    fn read_scanout_bytes(
        &self,
        address: u64,
        desc: &raeen_core::subsystems::ScanoutDescriptor,
    ) -> Option<Vec<u8>> {
        if address == 0 || desc.width == 0 || desc.height == 0 || desc.tiling_mode > 1 {
            return None;
        }
        let width = desc.width;
        let pitch = if desc.pitch_pixels != 0 {
            desc.pitch_pixels
        } else {
            width
        };
        if pitch < width {
            return None;
        }
        let total = (pitch as u64 * 4).checked_mul(desc.height as u64)?;
        // Refuse an absurd read (8K x 8K x 4 = 256 MiB is the ceiling).
        if total > (256 << 20) {
            return None;
        }
        let memory = self.guest_memory.lock().clone()?;
        if !memory.validate_gpu_range(address, total, false) {
            return None;
        }
        let mut src = Vec::<u8>::new();
        src.try_reserve_exact(total as usize).ok()?;
        src.resize(total as usize, 0);
        if !memory.read_gpu(address, &mut src) {
            return None;
        }
        Some(src)
    }

    /// Enqueue the present flush for a flip and return without waiting for the
    /// worker to drain — the bounded fire-and-forget flip (item 7 / rank 7,
    /// THE fps lever).
    ///
    /// ## Why this exists
    ///
    /// The old path called `consume_flush(Some(address), true)`, which sends a
    /// [`GpuWork::Flush`] down the *same* ordered channel as draw submissions
    /// and then blocks the guest's flip thread on the rendezvous until the
    /// worker has drained every queued submit AND run the flush. A single GPU
    /// worker drains that channel, so every flip inherited the full
    /// worker-drain + fence-stall latency — measured ~2.3 s per flip on
    /// ASTRO.BOT, pinning it at ~0.4 fps. The 60 Hz vblank pacer in
    /// `libSceVideoOut` already paces flips; this synchronous wait was pure
    /// extra latency stacked on top of it.
    ///
    /// This enqueues the flush the same way (still ordered AFTER every draw the
    /// title submitted before the flip — ordering is preserved) but does NOT
    /// wait for it. The flip returns immediately, so the guest can build the
    /// next frame while the worker catches up: CPU work overlaps GPU work.
    ///
    /// ## Why it is bounded (and why unbounded wedged Minecraft)
    ///
    /// A prior naive `wait: false` attempt was UNBOUNDED and REVERTED: on
    /// Minecraft the unthrottled flip stream let the title's render pool thread
    /// reuse display buffers whose flips had not completed, corrupting the
    /// title's own flip state machine — the main thread wedged ~10 s into boot
    /// on a title mutex held by that render pool thread (a pthread_sync
    /// "stuck >3s" warn named it). Real hardware bounds a title's race-ahead
    /// with a finite pool of display buffers; the [`FlipSemaphore`] restores
    /// that backpressure.
    ///
    /// ## Deadlock-safety reasoning
    ///
    /// - The permit is acquired BEFORE any GPU lock and while holding no
    ///   guest-visible lock (only the semaphore's own mutex, briefly). A
    ///   blocked flip thread therefore cannot hold a lock the worker needs to
    ///   complete a flush and release a permit.
    /// - The permit is released from the GPU-completion side: the worker drops
    ///   the [`FlipPermit`] after the flush executes. The guest flip thread
    ///   only ever *acquires*; it is never the thread that must run to free a
    ///   permit, and the worker only ever *releases*, so there is no acquire
    ///   cycle. The worker always makes forward progress draining the channel.
    /// - This restores exactly the blocking that made the old synchronous form
    ///   safe, but pays it only when the guest is [`FLIP_FRAMES_IN_FLIGHT`]
    ///   frames ahead of the GPU — not on every flip.
    /// - The permit is always eventually returned: the worker drops it after
    ///   the flush (even if the flush panics — it is bound outside the worker's
    ///   `catch_unwind`); a shutdown that leaves the item queued drops it when
    ///   the receiver is dropped; and every inline fallback drops it here.
    fn submit_flip_flush(&self, address: u64) {
        if let Some(sender) = self.submit_queue.get() {
            // Acquire the frames-in-flight permit first, holding no GPU lock.
            // Blocks only when `FLIP_FRAMES_IN_FLIGHT` flushes are already
            // outstanding (the hardware display-buffer backpressure).
            let permit = self.flip_permits.acquire();
            let enqueued = {
                let lifecycle = self.lifecycle.lock();
                if *lifecycle == GpuLifecycle::Open {
                    // On success the permit rides into the worker and is
                    // released when the flush completes. On send failure
                    // (worker gone) `.is_ok()` drops the returned SendError —
                    // and with it the permit — right here.
                    sender
                        .send(GpuWork::Flush {
                            address: Some(address),
                            queued_at: std::time::Instant::now(),
                            done: None,
                            permit: Some(permit),
                        })
                        .is_ok()
                } else {
                    // Session closing: fall through inline. `permit` drops at
                    // the end of this `if let` block, before the inline flush.
                    false
                }
            };
            if enqueued {
                return;
            }
        }
        // No worker was ever started, the session is closing, or the send
        // failed: run the flush inline (synchronous, but nothing is queued to
        // wait behind). Any permit acquired above has already been released.
        self.present_worker_drain_us.store(0, Ordering::Relaxed);
        self.flush_and_present(Some(address));
    }

    /// Run one flush at a consumer point (flip or `wait_idle`), routed through
    /// the ordered GPU work queue when the worker exists — the flush must
    /// execute AFTER every already-queued submission (register state and draws
    /// for this flip may still be in the queue), and Vulkan work stays on the
    /// single consumer thread. With `wait` the caller blocks until the flush
    /// has executed (the `wait_idle` contract); without it the flush is queued
    /// and the caller returns (flip latency, item 4). Falls back to running
    /// inline when no worker was ever started (nothing is queued) or the
    /// session is shutting down (the worker may be gone).
    fn consume_flush(&self, address: Option<u64>, wait: bool) {
        if let Some(sender) = self.submit_queue.get() {
            let lifecycle = self.lifecycle.lock();
            if *lifecycle == GpuLifecycle::Open {
                let (done_tx, done_rx) = if wait {
                    let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
                    (Some(tx), Some(rx))
                } else {
                    (None, None)
                };
                let sent = sender
                    .send(GpuWork::Flush {
                        address,
                        queued_at: std::time::Instant::now(),
                        done: done_tx,
                        // `wait_idle` carries no flip budget — the bounded
                        // fire-and-forget permit belongs to the flip path only.
                        permit: None,
                    })
                    .is_ok();
                drop(lifecycle);
                if sent {
                    match done_rx {
                        // A dropped worker (spawn failed and the receiver is
                        // gone) errors recv rather than blocking; fall
                        // through inline.
                        Some(rx) => {
                            if rx.recv().is_ok() {
                                return;
                            }
                        }
                        None => return,
                    }
                }
            }
        }
        self.present_worker_drain_us.store(0, Ordering::Relaxed);
        self.flush_and_present(address);
    }

    /// The flush consumer body (stage C): land pending deferred readbacks —
    /// restricted to the flipped target when one is named (item 2) — then
    /// present the scanout buffer. Runs on the GPU worker thread via
    /// [`GpuWork::Flush`], or inline when no worker exists.
    fn flush_and_present(&self, address: Option<u64>) {
        self.present_fence_wait_us.store(0, Ordering::Relaxed);
        self.present_readback_us.store(0, Ordering::Relaxed);
        self.present_srgb_encode_us.store(0, Ordering::Relaxed);
        // Did this flush actually read back the flipped target — i.e. did the
        // GPU render into the buffer the title is presenting? That is the
        // honest "the title drew this frame" signal, and it is what lets
        // `present_flipped` accept a frame that is legitimately BLACK (a
        // fade-out, a night scene) instead of mistaking it for a never-drawn
        // buffer and presenting a stale target in its place.
        let mut flip_target_drawn = false;
        {
            let guard = self.backend.lock();
            if let Some(device) = guard.as_ref().and_then(|b| b.device()) {
                let mut framebuffers = self.framebuffers.lock();
                let mut gpu_overrides = self.gpu_present_overrides.lock();
                let insert_all =
                    |fb: &mut std::collections::HashMap<u64, Arc<RenderedImage>>,
                     overrides: &mut std::collections::HashMap<u64, Arc<RenderedImage>>,
                     native: Vec<(u64, RenderedImage)>,
                     plugin: Vec<(u64, RenderedImage)>,
                     drawn: &mut bool| {
                        for (base, img) in native {
                            if Some(base) == address {
                                *drawn = true;
                            }
                            overrides.remove(&base);
                            fb.insert(base, Arc::new(img));
                        }
                        overrides.extend(
                            plugin
                                .into_iter()
                                .map(|(base, image)| (base, Arc::new(image))),
                        );
                    };
                // The all-targets dump needs every target's CPU pixels, so it
                // forces a full readback; otherwise a flip reads back ONLY the
                // flipped target plus the remembered fallback target — every
                // other dirty target stays GPU-side.
                let remembered = *self.fallback_present_base.lock();
                let filter: Option<Vec<u64>> = if crate::diagnostics::gpu_env().dump_all_targets {
                    None
                } else {
                    Some(presentation_filter_bases(address, remembered))
                };
                // Deferred compute publication writes back to guest SSBOs/UAVs.
                // A flip/suspend flush runs as a separate worker item, outside
                // `execute_dcb_cp_routed`'s submission-scoped authority guard.
                // Reinstall the owning process's memory authority here or
                // every valid writeback is rejected as "not writable".
                let memory = self.guest_memory.lock().clone();
                let flushed = if let Some(memory) = memory {
                    crate::guest_mem::with_guest_memory(&memory, || {
                        crate::vulkan::offscreen::flush_deferred_draws_filtered_timed(
                            device,
                            filter.as_deref(),
                        )
                    })
                } else {
                    crate::vulkan::offscreen::flush_deferred_draws_filtered_timed(
                        device,
                        filter.as_deref(),
                    )
                };
                match flushed {
                    Ok(flushed) => {
                        self.present_fence_wait_us
                            .store(flushed.timing.fence_wait_us, Ordering::Relaxed);
                        self.present_readback_us
                            .store(flushed.timing.readback_us, Ordering::Relaxed);
                        insert_all(
                            &mut framebuffers,
                            &mut gpu_overrides,
                            flushed.images,
                            flushed.plugin_images,
                            &mut flip_target_drawn,
                        );
                    }
                    Err(e) => {
                        warn!(error = %e, "deferred-draw flush failed — presenting the last flushed frame");
                    }
                }
                // Routing miss: either the flipped address has no drawn
                // target, or it contains only a uniform clear and the CPU map
                // has no detailed target yet. The latter is Minecraft's
                // screen transition: filtered readback kept landing the gray
                // VideoOut buffers while the completed UI remained GPU-side.
                //
                // A FULL flush + census runs immediately for an unknown
                // route, then periodically. Once a detailed fallback is
                // remembered, the normal filtered path reads only it and the
                // scanout. Periodic re-election catches content moving to a
                // different target without turning solid fades into a
                // full-readback-every-frame performance regression.
                if let Some(addr) = address {
                    let scanout_missing = !framebuffers.contains_key(&addr);
                    let uniform_without_detail =
                        framebuffers.get(&addr).is_some_and(|image| {
                            has_visible_content(image) && is_visually_uniform(image)
                        }) && elect_detailed_target(&framebuffers, Some(addr)).is_none();
                    if scanout_missing || uniform_without_detail {
                        let misses = self.flip_miss_count.fetch_add(1, Ordering::Relaxed);
                        let remembered_has_detail = remembered.is_some_and(|r| {
                            framebuffers.get(&r).is_some_and(|image| {
                                has_visible_content(image) && !is_visually_uniform(image)
                            })
                        });
                        let needs_full_census = if scanout_missing {
                            !remembered_has_detail
                                || misses.is_multiple_of(FALLBACK_REELECT_INTERVAL)
                        } else {
                            misses.is_multiple_of(FALLBACK_REELECT_INTERVAL)
                        };
                        if needs_full_census {
                            match crate::vulkan::offscreen::flush_deferred_draws_filtered_timed(
                                device, None,
                            ) {
                                Ok(flushed) => {
                                    self.present_fence_wait_us
                                        .fetch_add(flushed.timing.fence_wait_us, Ordering::Relaxed);
                                    self.present_readback_us
                                        .fetch_add(flushed.timing.readback_us, Ordering::Relaxed);
                                    insert_all(
                                        &mut framebuffers,
                                        &mut gpu_overrides,
                                        flushed.images,
                                        flushed.plugin_images,
                                        &mut flip_target_drawn,
                                    );
                                }
                                Err(e) => {
                                    warn!(error = %e, "full deferred-draw flush failed at a flip miss");
                                }
                            }
                            // Force `present_flipped` to re-run its census over
                            // the freshly-landed pixels instead of trusting the
                            // remembered winner.
                            *self.fallback_present_base.lock() = None;
                        }
                    }
                }
            }
        }
        if let Some(address) = address {
            self.present_flipped(address, flip_target_drawn);
        }
    }

    /// Present the buffer the title flipped to, after the flush landed its
    /// pixels: the frame at the flip address when the GPU drew into it this
    /// flush (`flip_target_drawn`) or it has visible content, else the drawn
    /// target with the most content (the composite — the title fills its
    /// scanout buffer by a copy/DMA we do not yet capture, so the flip address
    /// is often an empty target). Never regresses `last_image` when nothing has
    /// content, and only a frame at the actual flip address takes the boot
    /// splash down.
    ///
    /// `flip_target_drawn` is what keeps a legitimately BLACK frame — a
    /// fade-out, a night scene, a dark loading screen — from being mistaken for
    /// a never-drawn buffer: pixel content alone cannot tell those apart, and
    /// judging by content froze such frames on the last bright image (and left
    /// the boot splash up for a title that opens dark).
    /// sRGB-encode an HDR target for presentation, reusing a cached encode when
    /// the SAME `Arc<RenderedImage>` is presented again (a title flipping
    /// without redrawing, or the remembered-fallback path re-presenting its
    /// target). Non-HDR images pass through uncached, as before.
    fn to_presentable_cached(&self, image: Arc<RenderedImage>) -> Option<Arc<RenderedImage>> {
        if image.bytes_per_pixel != 8 {
            return Some(image);
        }
        let key = Arc::as_ptr(&image) as usize;
        {
            let cache = self.present_encode_cache.lock();
            for (stored_key, source, encoded) in cache.iter() {
                if *stored_key == key
                    && source
                        .upgrade()
                        .is_some_and(|source| Arc::ptr_eq(&source, &image))
                {
                    return Some(Arc::clone(encoded));
                }
            }
        }
        let encoded = try_to_presentable_arc(Arc::clone(&image))?;
        let mut cache = self.present_encode_cache.lock();
        // A display-buffer ring plus the fallback target is all that recurs;
        // clear rather than grow without bound.
        if cache.len() >= 8 {
            cache.clear();
        }
        cache.push((key, Arc::downgrade(&image), Arc::clone(&encoded)));
        Some(encoded)
    }

    fn present_flipped(&self, address: u64, flip_target_drawn: bool) {
        let remembered = *self.fallback_present_base.lock();
        let (image, fallback, fallback_base, keys, flip_target_known) = {
            let fb = self.framebuffers.lock();
            let flip_target_known = fb.contains_key(&address);
            let scanout_image = fb
                .get(&address)
                .filter(|image| flip_target_drawn || has_visible_content(image))
                .cloned();
            // A uniform scanout clear yields when a richer target exists. With
            // no richer target it remains authoritative, preserving solid
            // fades. A GPU-drawn black frame always remains authoritative:
            // unlike Minecraft's opaque gray clear, it may be an intentional
            // fade/night frame and cannot be distinguished from one by pixels.
            let detailed_over_uniform = scanout_image
                .as_ref()
                .filter(|image| has_visible_content(image) && is_visually_uniform(image))
                .and_then(|_| elect_detailed_target(&fb, Some(address)));
            let image = if detailed_over_uniform.is_some() {
                None
            } else {
                scanout_image
            };
            let (fallback, fallback_base) = if image.is_none() {
                // Steady state: present the remembered fallback target while
                // it still has content — no census, no scan of other targets
                // (whose entries may be deliberately stale, kept GPU-side by
                // the filtered flush).
                let kept = detailed_over_uniform.or_else(|| {
                    remembered.and_then(|r| {
                        fb.get(&r)
                            .filter(|img| has_visible_content(img) && !is_visually_uniform(img))
                            .map(|img| (r, Arc::clone(img)))
                    })
                });
                match kept {
                    Some((base, img)) => (Some(img), Some(base)),
                    None => {
                        // Census election, sub-sampled (every 64th byte):
                        // exact counts cost a full scan of every 8 MB target
                        // per flip. Which target has the MOST content
                        // survives sub-sampling.
                        let elected = elect_detailed_target(&fb, Some(address));
                        match elected {
                            Some((base, img)) => (Some(img), Some(base)),
                            None => (None, None),
                        }
                    }
                }
            } else {
                (None, None)
            };
            // RAEEN_TRACE_FLIP shows EVERY flip — an explicit opt-in wants the
            // whole stream. The automatic nothing-to-present case is rate
            // limited exactly like the BLACK FRAME warning below it: it fires
            // on every flip otherwise, and buried each run's log under hundreds
            // of near-identical lines.
            let keys = if crate::diagnostics::gpu_env().trace_flip || {
                image.is_none() && fallback.is_none() && {
                    static SCANOUT_LOGS: AtomicU64 = AtomicU64::new(0);
                    let occurrence = SCANOUT_LOGS.fetch_add(1, Ordering::Relaxed) + 1;
                    occurrence <= 8 || occurrence.is_power_of_two()
                }
            } {
                Some(fb.keys().map(|k| format!("{k:#x}")).collect::<Vec<_>>())
            } else {
                None
            };
            (image, fallback, fallback_base, keys, flip_target_known)
        };
        if image.is_none() {
            // Remember (or clear) the fallback winner so the next flip's
            // filtered flush reads it back and this path skips the census.
            *self.fallback_present_base.lock() = fallback_base;
        }
        // Present-from-guest-memory (M3): a CPU-drawn 2D buffer never entered
        // the GPU render-target map, so with no GPU-drawn target at the flip
        // address AND no drawn fallback anywhere, read the guest bytes at the
        // flip address as pixels. Ordered last so a GPU title — whose scanout
        // is filled by an uncaptured copy/DMA while its real pixels live in
        // another render target — keeps presenting that target instead of the
        // (empty) guest scanout bytes.
        // Some Gen5 titles render the completed scene into an intermediate but
        // issue their final display copy through a PM4 path that is not decoded
        // yet. We already selected that exact content-bearing target above;
        // for compatible linear RGBA/BGRA scanouts, perform the missing final
        // copy into the guest's real flip buffer. This turns the fallback into
        // persistent scanout state (and lets subsequent CPU/GPU consumers see
        // the same bytes) instead of merely painting the wrong-address target.
        let composed_present = if image.is_none()
            && let (Some(fallback), Some(desc)) =
                (fallback.as_ref(), *self.scanout_descriptor.lock())
        {
            let encode_at = std::time::Instant::now();
            let presentable = self.to_presentable_cached(Arc::clone(fallback));
            self.present_srgb_encode_us
                .fetch_add(encode_at.elapsed().as_micros() as u64, Ordering::Relaxed);
            presentable.filter(|presentable| {
                self.compose_presentable_to_scanout(address, &desc, presentable)
            })
        } else {
            None
        };
        let guest_present = if image.is_none() {
            // A successful compatibility composite already owns the exact,
            // complete RGBA frame the Shell needs. Publishing that Arc directly
            // avoids allocating another 8 MiB buffer merely to decode the guest
            // scanout bytes we wrote one line above.
            composed_present.or_else(|| {
                let desc = *self.scanout_descriptor.lock();
                desc.and_then(|desc| self.present_from_guest_memory(address, &desc))
                    .map(Arc::new)
            })
        } else {
            None
        };
        // The guest scanout is authoritative when it actually holds pixels:
        // ASTRO-class titles compose their display buffer by compute/DMA writes
        // we do not capture as a render target, so the real frame lives at the
        // flip address in guest memory, not in any drawn target. A uniform
        // guest clear must not hide a detailed intermediate, however; Minecraft
        // leaves its registered scanouts gray while the next UI is complete in
        // another target.
        let guest_has_content = guest_present
            .as_ref()
            .is_some_and(|image| has_visible_content(image));
        let fallback_has_detail = fallback
            .as_ref()
            .is_some_and(|image| !is_visually_uniform(image));
        let guest_is_uniform = guest_present
            .as_ref()
            .is_none_or(|image| is_visually_uniform(image));
        let guest_preferred = guest_has_content && !(fallback_has_detail && guest_is_uniform);
        let guest_hit = guest_preferred;
        let fallback_hit = fallback.is_some();
        // RAEEN_TRACE_FLIP: does the buffer the title flipped to have drawn
        // content, and what render targets exist? Answers whether black frames
        // are a routing miss (content is in another target) or a genuinely
        // empty scanout (the composite that fills it never ran).
        if let Some(keys) = keys {
            tracing::info!(
                scanout = format_args!("{address:#x}"),
                target_known = flip_target_known,
                target_visible = image.is_some(),
                render_targets = ?keys,
                "present_scanout: title flipped to this buffer"
            );
        }
        let flip_address_hit = image.is_some();
        if !flip_address_hit && !guest_preferred && fallback_hit {
            static ROUTING_WARNINGS: AtomicU64 = AtomicU64::new(0);
            let occurrence = ROUTING_WARNINGS.fetch_add(1, Ordering::Relaxed) + 1;
            if occurrence <= 8 || occurrence.is_power_of_two() {
                warn!(
                    occurrence,
                    scanout = format_args!("{address:#x}"),
                    scanout_target_known = flip_target_known,
                    fallback_target = format_args!("{:#x}", fallback_base.unwrap_or(0)),
                    "PRESENT ROUTING: flipped scanout is empty or only a uniform clear; \
                     presenting a content-bearing intermediate target (final copy/composite \
                     to scanout is missing or still pending)"
                );
            }
        }
        if !flip_address_hit && !guest_preferred && !fallback_hit {
            static BLACK_FRAME_WARNINGS: AtomicU64 = AtomicU64::new(0);
            let occurrence = BLACK_FRAME_WARNINGS.fetch_add(1, Ordering::Relaxed) + 1;
            if occurrence <= 8 || occurrence.is_power_of_two() {
                let render_targets = self
                    .framebuffers
                    .lock()
                    .keys()
                    .map(|base| format!("{base:#x}"))
                    .collect::<Vec<_>>();
                warn!(
                    occurrence,
                    scanout = format_args!("{address:#x}"),
                    scanout_target_known = flip_target_known,
                    render_targets = ?render_targets,
                    completed_draws = *self.draw_count.lock(),
                    shader_skips = *self.shader_skip_count.lock(),
                    shader_cache = ?self.shader_stats(),
                    descriptor = ?*self.scanout_descriptor.lock(),
                    "BLACK FRAME: the flipped scanout, guest-memory scanout, and every \
                     available GPU render target contain no visible RGB colour; inspect \
                     preceding draw/shader skip warnings (or a missing final GPU copy)"
                );
            }
        }
        // Priority: a detailed target drawn at the flip address; then a
        // non-uniform guest scanout; then the detailed census fallback; then
        // the guest scanout as a last resort (including legitimate solid
        // fades when no detailed target exists).
        let presented = if let Some(img) = image {
            Some(img)
        } else if guest_preferred {
            guest_present
        } else {
            fallback.or(guest_present)
        };
        let Some(presented) = presented else {
            return;
        };
        let gpu_base = if flip_address_hit {
            Some(address)
        } else if guest_preferred {
            None
        } else {
            fallback_base
        };
        let gpu_presented =
            gpu_base.and_then(|base| self.gpu_present_overrides.lock().get(&base).cloned());
        let gpu_processed = gpu_presented.is_some();
        let presented = gpu_presented.unwrap_or(presented);
        // sRGB-encode an HDR float target (SharpEmu #448) to the RGBA8 the Shell
        // surface presents; already-8-bit frames pass through unchanged.
        // Move into the `Arc` once; the store below is then a refcount bump, not
        // an 8 MB copy — and this runs inside the guest flip thread's
        // synchronous flush, so the copy was pure per-flip latency.
        let encode_at = std::time::Instant::now();
        let Some(presented) = self.to_presentable_cached(presented) else {
            warn_present_allocation_failed();
            return;
        };
        self.present_srgb_encode_us
            .fetch_add(encode_at.elapsed().as_micros() as u64, Ordering::Relaxed);
        if gpu_processed {
            self.publish_gpu_frame(Arc::clone(&presented));
        } else {
            self.publish_frame(Arc::clone(&presented));
        }
        if flip_address_hit || guest_hit {
            // The title flipped to a buffer it really drew into (a GPU-drawn
            // target, or CPU-drawn pixels read straight from the flip address):
            // its own rendering has replaced the boot splash. The most-content
            // fallback must NOT take the splash down — it can surface a bare
            // cleared render target, which the splash exists to cover.
            self.hide_splash();
        }
        let present_index = PRESENT_INDEX.fetch_add(1, Ordering::Relaxed) + 1;
        if present_index <= 8 || present_index.is_power_of_two() {
            let timing = self.present_timing();
            tracing::info!(
                scanout_hit = flip_address_hit,
                present_index,
                scanout = format_args!("{address:#x}"),
                worker_drain_us = timing.worker_drain_us,
                fence_wait_us = timing.fence_wait_us,
                readback_us = timing.readback_us,
                srgb_encode_us = timing.srgb_encode_us,
                "present: dumping the scanned-out frame"
            );
        }
        maybe_dump_frame(&presented, present_index);
        if crate::diagnostics::gpu_env().dump_all_targets {
            let targets: Vec<(u64, RenderedImage)> = self
                .framebuffers
                .lock()
                .iter()
                .map(|(base, img)| (*base, (**img).clone()))
                .collect();
            maybe_dump_all_targets(&targets, present_index);
        }
    }

    /// Copy an already-presentable RGBA8 intermediate into a compatible real
    /// VideoOut buffer. Returns false for any format/layout that cannot be
    /// represented exactly; those continue through the existing fallback.
    fn compose_presentable_to_scanout(
        &self,
        address: u64,
        desc: &raeen_core::subsystems::ScanoutDescriptor,
        image: &RenderedImage,
    ) -> bool {
        if address == 0
            || desc.tiling_mode > 1
            || image.bytes_per_pixel != 4
            || image.width == 0
            || image.height == 0
            || desc.width == 0
            || desc.height == 0
            || (image.width as usize)
                .checked_mul(image.height as usize)
                .and_then(|pixels| pixels.checked_mul(4))
                != Some(image.pixels.len())
        {
            return false;
        }
        #[derive(Clone, Copy)]
        enum Order {
            Rgba,
            Bgra,
        }
        let order = match desc.pixel_format {
            0x8000_2000 | 0x8000_2200 | 0x8000_0000_2200_0000 => Order::Rgba,
            0x8000_0000 | 0x8000_0200 | 0x8000_0000_0000_0000 => Order::Bgra,
            _ => return false,
        };
        let pitch = if desc.pitch_pixels == 0 {
            desc.width
        } else {
            desc.pitch_pixels
        };
        if pitch < desc.width {
            return false;
        }
        let row_bytes = pitch as usize * 4;
        let Some(total) = row_bytes.checked_mul(desc.height as usize) else {
            return false;
        };
        if total > (256 << 20) {
            return false;
        }
        // The overwhelmingly common native RGBA case is already tightly
        // packed. Write it directly instead of allocating/copying a second
        // full scanout frame.
        let direct = matches!(order, Order::Rgba)
            && pitch == desc.width
            && (image.width, image.height) == (desc.width, desc.height);
        let scanout = if direct {
            None
        } else {
            let Some(mut scanout) = try_zeroed_bytes(total) else {
                warn_present_allocation_failed();
                return false;
            };
            for y in 0..desc.height as usize {
                let source_y = y * image.height as usize / desc.height as usize;
                let destination =
                    &mut scanout[y * row_bytes..y * row_bytes + desc.width as usize * 4];
                for x in 0..desc.width as usize {
                    let source_x = x * image.width as usize / desc.width as usize;
                    let source_at = (source_y * image.width as usize + source_x) * 4;
                    let src = &image.pixels[source_at..source_at + 4];
                    let dst = &mut destination[x * 4..x * 4 + 4];
                    match order {
                        Order::Rgba => dst.copy_from_slice(src),
                        Order::Bgra => {
                            dst.copy_from_slice(&[src[2], src[1], src[0], src[3]]);
                        }
                    }
                }
            }
            Some(scanout)
        };
        let bytes = scanout.as_deref().unwrap_or(&image.pixels);
        let wrote = self
            .guest_memory
            .lock()
            .as_ref()
            .is_some_and(|memory| memory.write_gpu(address, bytes));
        if !wrote {
            return false;
        }
        static COMPOSITE_LOGS: AtomicU64 = AtomicU64::new(0);
        let occurrence = COMPOSITE_LOGS.fetch_add(1, Ordering::Relaxed) + 1;
        if occurrence == 1 || occurrence.is_power_of_two() {
            tracing::info!(
                occurrence,
                scanout = format_args!("{address:#x}"),
                width = desc.width,
                height = desc.height,
                "completed intermediate-to-VideoOut scanout copy"
            );
        }
        true
    }

    /// Build a [`RenderedImage`] by reading the guest bytes at a flipped
    /// display buffer as pixels (present-from-guest-memory, M3). This is how
    /// CPU-drawn 2D pixels become visible without any GPU draw.
    ///
    /// SharpEmu `VulkanVideoPresenter.cs:1643-1660` (`GuestImageWantsInitialData`):
    /// PS5 render targets alias CPU-visible memory; a first-use image is seeded
    /// from guest memory. Only LINEAR tiling + 32-bit RGBA/BGRA is supported;
    /// other tile modes/formats are named in a rate-limited warn and skipped
    /// (never faked). The produced pixels are RGBA byte order to match the
    /// Shell's `from_rgba_unmultiplied` present path, swizzling B/R for the
    /// `A8R8G8B8` (memory order BGRA) formats.
    fn present_from_guest_memory(
        &self,
        address: u64,
        desc: &raeen_core::subsystems::ScanoutDescriptor,
    ) -> Option<RenderedImage> {
        if address == 0 || desc.width == 0 || desc.height == 0 {
            return None;
        }
        // The scanout is read linearly (row-major with pitch) for both the
        // TILE(0) and LINEAR(1) `tiling_mode` values: SharpEmu's software
        // presenter reads mode-0 display buffers row by row
        // (AgcExports `TrySoftwarePresent`), and ASTRO.BOT's real scanout is
        // `tiling_mode=0`. Higher modes are a genuine macro-tile we do not
        // detile yet — named in a rate-limited warn and skipped, never faked.
        if desc.tiling_mode > 1 {
            warn_unsupported_scanout(desc);
            return None;
        }
        // How to turn each 32-bit guest word into the RGBA8 the Shell presents.
        // `A8B8G8R8`/`2R8G8B8A8` words are memory R,G,B,A (no swizzle);
        // `A8R8G8B8`/`2B8G8R8A8` words are memory B,G,R,A (swap R/B). Real PS5
        // titles register the *Gen5* 64-bit `SceVideoOutBufferAttribute2` pixel
        // format (SharpEmu VideoOutExports.cs), so both the PS4-style 32-bit and
        // the Gen5 64-bit encodings are accepted. The packed 10:10:10:2 formats
        // (ASTRO.BOT's scanout is `2B10G10R10A2_SRGB` = 0x8100_0000_0000_0000)
        // unpack per SharpEmu `ConvertPacked10ToRgba8Normalized`.
        #[derive(Clone, Copy)]
        enum ScanoutConv {
            /// 8-bit word already in memory R,G,B,A order.
            Rgba8,
            /// 8-bit word in memory B,G,R,A order — swap R/B.
            Bgra8,
            /// Packed 10:10:10:2; `red_is_least` picks the 2R10.. vs 2B10.. lane.
            Packed10 { red_is_least: bool },
        }
        let conv = match desc.pixel_format {
            // PS4-style 32-bit encodings (measured on the 2D homebrew path).
            0x8000_2000 | 0x8000_2200 => ScanoutConv::Rgba8, // A8B8G8R8
            0x8000_0000 | 0x8000_0200 => ScanoutConv::Bgra8, // A8R8G8B8
            // Gen5 64-bit 8-bit encodings (real PS5 titles).
            0x8000_0000_2200_0000 => ScanoutConv::Rgba8, // 2R8G8B8A8
            0x8000_0000_0000_0000 => ScanoutConv::Bgra8, // 2B8G8R8A8
            // Gen5 64-bit packed 10:10:10:2 (SharpEmu VideoOutExports.cs:159-164).
            // 2R10G10B10A2 family: red in the least-significant 10 bits.
            0x8100_0000_2200_0000 | 0x8100_0006_2200_0000 | 0x8100_0704_2200_0000 => {
                ScanoutConv::Packed10 { red_is_least: true }
            }
            // 2B10G10R10A2 family: blue in the least-significant 10 bits.
            0x8100_0000_0000_0000 | 0x8100_0006_0000_0000 | 0x8100_0704_0000_0000 => {
                ScanoutConv::Packed10 {
                    red_is_least: false,
                }
            }
            _ => {
                warn_unsupported_scanout(desc);
                return None;
            }
        };
        let width = desc.width;
        let height = desc.height;
        let pitch = if desc.pitch_pixels != 0 {
            desc.pitch_pixels
        } else {
            width
        };
        if pitch < width {
            return None;
        }
        let row_bytes = pitch as u64 * 4;
        let total = row_bytes.checked_mul(height as u64)?;
        // Refuse an absurd read (8K x 8K x 4 = 256 MiB is the ceiling).
        if total > (256 << 20) {
            warn_unsupported_scanout(desc);
            return None;
        }
        // Prefer the snapshot captured on the flip thread — a COMPLETE frame at
        // this address, taken before the title could begin reusing the buffer —
        // so the async flush never presents a partially-cleared/reused buffer.
        // Fall back to a live read only when no snapshot was taken (a
        // `wait_idle` flush, or the flip-time heuristic saw a GPU target back
        // this address); that matches the pre-async behaviour for paths that
        // never raced.
        let src: Arc<Vec<u8>> = {
            let snap = self
                .guest_scanout_snapshots
                .lock()
                .get(&address)
                .filter(|b| b.len() == total as usize)
                .cloned();
            match snap {
                Some(bytes) => bytes,
                None => Arc::new(self.read_scanout_bytes(address, desc)?),
            }
        };
        let mut pixels = Vec::<u8>::new();
        pixels
            .try_reserve_exact(width as usize * height as usize * 4)
            .ok()?;
        pixels.resize(width as usize * height as usize * 4, 0);
        // 10-bit UNORM -> 8-bit UNORM, round-to-nearest preserving both
        // endpoints (SharpEmu `ReduceUnorm10To8`; a plain >>2 biases low
        // because the 10-bit max is 1023, not 1020).
        let reduce10to8 = |v: u32| ((v * 255 + 511) / 1023) as u8;
        for y in 0..height as usize {
            let src_row = y * row_bytes as usize;
            let dst_row = y * width as usize * 4;
            for x in 0..width as usize {
                let s = src_row + x * 4;
                let d = dst_row + x * 4;
                match conv {
                    ScanoutConv::Rgba8 => pixels[d..d + 4].copy_from_slice(&src[s..s + 4]),
                    ScanoutConv::Bgra8 => {
                        pixels[d] = src[s + 2];
                        pixels[d + 1] = src[s + 1];
                        pixels[d + 2] = src[s];
                        pixels[d + 3] = src[s + 3];
                    }
                    ScanoutConv::Packed10 { red_is_least } => {
                        let word = u32::from_le_bytes([src[s], src[s + 1], src[s + 2], src[s + 3]]);
                        let least = word & 0x3FF;
                        let green = (word >> 10) & 0x3FF;
                        let most = (word >> 20) & 0x3FF;
                        let (r10, b10) = if red_is_least {
                            (least, most)
                        } else {
                            (most, least)
                        };
                        pixels[d] = reduce10to8(r10);
                        pixels[d + 1] = reduce10to8(green);
                        pixels[d + 2] = reduce10to8(b10);
                        // 2-bit alpha -> 8-bit (0,85,170,255).
                        pixels[d + 3] = ((((word >> 30) & 0x3) * 255 + 1) / 3) as u8;
                    }
                }
            }
        }
        Some(RenderedImage {
            width,
            height,
            pixels,
            bytes_per_pixel: 4,
        })
    }

    fn ensure_backend(&self) -> Result<(), GpuError> {
        let mut slot = self.backend.lock();
        if slot.is_some() {
            return Ok(());
        }
        // Validation is opt-in on the title path. The Khronos layer costs
        // ~0.9s per vkCreateGraphicsPipelines on a real title (measured on
        // ASTRO.BOT: 1028 draws => ~15 minutes of pipeline creation alone),
        // which is the difference between reaching a presented frame and
        // never getting there. Set RAEEN_VULKAN_VALIDATION=1 to restore it
        // when debugging a specific draw.
        // Validation is on when Settings ▸ Video ▸ Validation Layers is enabled,
        // or the env override is set (a per-run toggle for debugging one draw
        // without editing the config). Read at first backend creation, so the
        // config setting applies from the launch after it was changed.
        let validation = Self::runtime_config().validation_layers
            || std::env::var_os("RAEEN_VULKAN_VALIDATION").is_some();
        let mut backend = VulkanBackend::new(validation);
        backend.init()?;
        *slot = Some(backend);
        Ok(())
    }

    /// Run a DCB through the real PM4 command processor.
    ///
    /// This is the title path: the draw is built from decoded register state,
    /// with no fixture anywhere. Returns `Ok(None)` if the DCB contained no
    /// draw.
    ///
    /// # Errors
    ///
    /// [`AgcExecError::CommandProcessor`] if a packet is unknown/truncated or a draw's
    /// registers cannot be honoured (the error names the register), or
    /// [`AgcExecError::Gpu`] if Vulkan is unavailable.
    pub fn execute_dcb_cp(
        &self,
        words: &[u32],
        is_compute: bool,
    ) -> Result<Option<RenderedImage>, AgcExecError> {
        puffin::profile_function!();
        // Frame path: this is where a guest GPU submission actually reaches the
        // command processor. A title with buffers registered but zero DCBs here
        // never asked the GPU for anything.
        frame_path::record(Stage::DcbSubmitted);
        // RenderDoc capture bracket (RAEEN_RENDERDOC_CAPTURE + running under
        // RenderDoc): each remaining budgeted DCB execution becomes one
        // capture, swapchain or not.
        let _renderdoc = crate::diagnostics::renderdoc_dcb_capture();
        // Synchronous entry (tests, inline fallback): there is no worker to
        // re-check a suspended buffer, so an unmet wait cannot park it.
        // Continue past it — the pre-suspend behaviour — rather than dropping
        // the remainder of the buffer.
        let mut start = 0usize;
        let mut image = None;
        loop {
            let (segment, suspended) =
                self.execute_dcb_cp_routed(words, start, is_compute, false)?;
            image = segment.or(image);
            let Some(suspended) = suspended else {
                return Ok(image);
            };
            debug!(
                label = format_args!("{:#x}", suspended.wait.address),
                "inline DCB execution continuing past an unmet wait (no worker to park it)"
            );
            start = suspended.resume_dword;
        }
    }

    /// [`Self::execute_dcb_cp`] with presentation routing (stage C). With
    /// `deferred_present = false` this is byte-identical to the historical
    /// behaviour: flush + presentation logic run at the end of every
    /// draw-bearing submission. With `true` (the GPU worker's title path) the
    /// submission only runs the command processor and defers readback AND
    /// presentation to the next flush consumer — one flush per flip.
    fn execute_dcb_cp_routed(
        &self,
        words: &[u32],
        start_dword: usize,
        is_compute: bool,
        deferred_present: bool,
    ) -> Result<(Option<RenderedImage>, Option<SuspendedWait>), AgcExecError> {
        let memory = self
            .guest_memory
            .lock()
            .clone()
            .ok_or(AgcExecError::AddressSpaceUnavailable)?;
        crate::guest_mem::with_guest_memory(&memory, || {
            self.execute_dcb_cp_authorized(words, start_dword, is_compute, deferred_present)
        })
    }

    fn execute_dcb_cp_authorized(
        &self,
        words: &[u32],
        start_dword: usize,
        is_compute: bool,
        deferred_present: bool,
    ) -> Result<(Option<RenderedImage>, Option<SuspendedWait>), AgcExecError> {
        let decoded = agc::decode_submission(words)?;
        // Route each queue to its own command processor: the async-compute (ACB)
        // ring keeps register/shader state independent of the graphics DCB, so a
        // reset on one queue can't zero the other's bound shader.
        let mut cp = if is_compute {
            self.compute_command_processor.lock()
        } else {
            self.command_processor.lock()
        };

        // State-only DCBs are still real GPU work. Process them without
        // forcing Vulkan initialization so their register writes are latched
        // for the next submission.
        if decoded.draw_packets == 0 && decoded.dispatch_packets == 0 {
            let mut sink = StateOnlySink;
            let outcome = cp.run_resumable(
                words,
                start_dword,
                &mut sink,
                Some(&crate::guest_mem::IdentityGuestMemory),
            )?;
            let produced = cp.take_produced_labels();
            // In-stream completion side effects (events/EOP/flips), published
            // for HLE delivery in execution order iff the defer gate is on
            // (gate off: the HLE already applied them eagerly at submit).
            crate::ordered_side_effects::publish_cp_side_effects(cp.take_side_effects());
            drop(cp);
            self.latch_produced_waits(&produced);
            return Ok((None, suspended_of(outcome)));
        }

        self.ensure_backend()?;
        let guard = self.backend.lock();
        let backend = guard
            .as_ref()
            .expect("ensure_backend left a live VulkanBackend");
        let device = backend.device().ok_or_else(|| {
            GpuError::VulkanInitFailed("backend not initialized — call init() first".to_owned())
        })?;

        let mut cache = self.shader_cache.lock();
        let mut framebuffers = self.framebuffers.lock();
        let mut sink = OffscreenDrawSink::new(device, &mut cache, &mut framebuffers);
        sink.queue_is_compute = is_compute;
        // Cross-queue compute-shader seeding: hand the sink the last compute
        // shader bound on either queue so a dispatch-only ACB buffer can fall
        // back to it (the title binds on the DCB, dispatches on the ACB).
        sink.current_compute = *self.last_compute_shader.lock();
        // Indirect register/draw packets carry guest pointers; the identity
        // map makes them host-readable (VirtualQuery-validated).
        let run = cp.run_resumable(
            words,
            start_dword,
            &mut sink,
            Some(&crate::guest_mem::IdentityGuestMemory),
        );
        if submission_compute_flush_required(sink.queued_compute, deferred_present) {
            // Synchronous/test callers have no ordered worker consumer, so
            // fence here. The title worker keeps compute work GPU-side across
            // adjacent DCB/ACB submissions and fences at the real lifetime
            // boundary (the first dependent draw, `sceAgcSuspendPoint`, flip,
            // wait_idle, or shutdown).
            // Passing an empty target filter keeps render targets GPU-side.
            crate::vulkan::offscreen::flush_deferred_draws_filtered(device, Some(&[]))?;
        }
        let produced = cp.take_produced_labels();
        // In-stream completion side effects (events/EOP/flips) — see the
        // state-only path above; a suspended walk has published nothing past
        // its unmet wait, so delivery order is PM4 execution order.
        crate::ordered_side_effects::publish_cp_side_effects(cp.take_side_effects());
        self.latch_produced_waits(&produced);
        // Carry any compute shader this submission observed forward to the next.
        if let Some(cs) = sink.current_compute {
            *self.last_compute_shader.lock() = Some(cs);
        }
        let shader_state = cp.get_sh_ctx().clone();

        let drawn = sink.draws;
        let shader_skips = sink.shader_skips;
        let draw_skips = sink.draw_skips;
        let dispatch_skips = sink.dispatch_skips;
        let draw_skip_reason = sink.last_draw_skip_reason.clone();
        let dispatch_skip_reason = sink.last_dispatch_skip_reason.clone();
        let last_target = sink.last_target;
        let mut image = sink.last.take();
        drop(sink);
        if deferred_present {
            // Stage C: no flush and no presentation here. Deferred draws stay
            // GPU-side until the next flush consumer — a flip
            // (`present_scanout`), `wait_idle`, a frame dump, or the
            // feedback-loop fallback inside the sink. This is what turns 125
            // flushes per 11 presents into ~1 per flip.
            drop(framebuffers);
            drop(cache);
            drop(guard);
            self.record_shader_skips(
                shader_skips,
                draw_skips,
                dispatch_skips,
                draw_skip_reason.as_deref(),
                dispatch_skip_reason.as_deref(),
                &shader_state,
            );
            let suspended = suspended_of(run?);
            if drawn > 0 {
                *self.draw_count.lock() += drawn;
                frame_path::record_n(Stage::Draw, drawn);
            }
            debug!(
                drawn,
                last_target = format_args!("{:#x}", last_target.unwrap_or(0)),
                "AGC DCB executed with presentation deferred to the next flip"
            );
            return Ok((image, suspended));
        }
        // Stage B flush: land every deferred readback in the framebuffer map
        // — at most one readback per touched target per SUBMISSION, instead
        // of one per draw. Runs before presentation/dump logic so everything
        // downstream (scanout lookup, most-content fallback, frame dumps)
        // sees exactly the bytes the old per-draw path produced.
        match crate::vulkan::offscreen::flush_deferred_draws(device) {
            Ok(flushed) => {
                for (base, img) in flushed {
                    framebuffers.insert(base, Arc::new(img));
                }
            }
            Err(e) => {
                warn!(error = %e, "deferred-draw flush failed — presenting the last flushed frame");
            }
        }
        // The submission's own "last image": the last-drawn target's flushed
        // pixels (deferred draws populate `last` only via the immediate
        // fallback).
        if let Some(base) = last_target
            && let Some(img) = framebuffers.get(&base)
        {
            image = Some((**img).clone());
        }
        // Snapshot every accumulated render target while the guard is still
        // held (re-locking `self.framebuffers` here would deadlock — the guard
        // lives to end of scope), for the optional all-targets dump below.
        let all_targets: Option<Vec<(u64, RenderedImage)>> = image.as_ref().and_then(|_| {
            crate::diagnostics::gpu_env().dump_all_targets.then(|| {
                framebuffers
                    .iter()
                    .map(|(base, img)| (*base, (**img).clone()))
                    .collect()
            })
        });
        // Prefer the buffer the title flipped to (VideoOut scanout) over the
        // last-drawn target — captured here while the framebuffers lock is
        // still held. Composited UIs draw their black background last, so the
        // last-drawn target is the wrong thing to present; the scanout buffer
        // is where the composite landed. `None` keeps the last-drawn baseline.
        let (flip_address_hit, scanout_image) = {
            let addr = *self.scanout_address.lock();
            let at_flip_address = addr
                .and_then(|a| framebuffers.get(&a))
                .filter(|image| has_visible_content(image))
                .cloned();
            let flip_address_hit = at_flip_address.is_some();
            let image = at_flip_address.or_else(|| {
                // The title fills its VideoOut scanout buffer by a copy/DMA we do
                // not yet capture (task #11), so that address is often an empty
                // target. Present the drawn target with the MOST content — the
                // composited frame — instead of the last-drawn one (usually a
                // black background composited last). Single non-zero-byte pass
                // per target; only runs when the flip address has no drawn image.
                framebuffers
                    .values()
                    .map(|img| (visible_pixel_count(img, 1), img))
                    .filter(|(nonzero, _)| *nonzero > 0)
                    .max_by_key(|(nonzero, _)| *nonzero)
                    .map(|(_, img)| Arc::clone(img))
            });
            (flip_address_hit, image)
        };
        drop(framebuffers);
        drop(cache);
        drop(guard);
        self.record_shader_skips(
            shader_skips,
            draw_skips,
            dispatch_skips,
            draw_skip_reason.as_deref(),
            dispatch_skip_reason.as_deref(),
            &shader_state,
        );
        let suspended = suspended_of(run?);

        if let Some(image) = image {
            // Present the VideoOut scanout buffer when the title has flipped to
            // one that has been drawn; otherwise fall back to the last-drawn
            // target (the pre-existing baseline).
            let scanout_hit = flip_address_hit;
            let presented = scanout_image.unwrap_or_else(|| Arc::new(image.clone()));
            // sRGB-encode an HDR float target (SharpEmu #448) before it reaches
            // the RGBA8 present surface / PPM dump; 8-bit frames pass through.
            // Into the `Arc` once (see `last_image`): the store is a bump.
            let encode_at = std::time::Instant::now();
            let Some(presented) = self.to_presentable_cached(presented) else {
                {
                    let mut draws = self.draw_count.lock();
                    *draws += drawn;
                }
                frame_path::record_n(Stage::Draw, drawn);
                warn_present_allocation_failed();
                return Ok((Some(image), suspended));
            };
            self.present_srgb_encode_us
                .store(encode_at.elapsed().as_micros() as u64, Ordering::Relaxed);
            self.publish_frame(Arc::clone(&presented));
            // Only a frame at the actual flip address takes the boot splash
            // down. The most-content fallback can surface a bare cleared
            // target — exactly what the splash exists to cover.
            if flip_address_hit {
                self.hide_splash();
            }
            {
                let mut draws = self.draw_count.lock();
                *draws += drawn;
            }
            frame_path::record_n(Stage::Draw, drawn);
            // Gate the dump on how many frames have been PRESENTED, not on the
            // cumulative draw count. A submission contributes its whole draw
            // total at once (`*draws += drawn`), so the draw counter jumps
            // 0 -> 21 -> 42 -> 56 and almost never lands on the "<=8 or power
            // of two" cadence: ASTRO.BOT rendered 56 real draws and every
            // single dump was skipped. One increment per present restores the
            // intended first-8-then-doubling sampling.
            let present_index = PRESENT_INDEX.fetch_add(1, Ordering::Relaxed) + 1;
            // Dump what is actually PRESENTED (the scanout/composite), not the
            // last-drawn target (often the black background composited last) —
            // otherwise the frame dump misrepresents a rendered scene as black.
            if present_index <= 8 || present_index.is_power_of_two() {
                tracing::info!(
                    scanout_hit,
                    present_index,
                    scanout = format_args!("{:#x}", self.scanout_address.lock().unwrap_or(0)),
                    "present: dumping the scanned-out frame"
                );
            }
            maybe_dump_frame(&presented, present_index);
            // A title renders its UI to several render targets and composites
            // them; the last-drawn one (often the display's black background
            // this early) is not necessarily where the content is. The
            // snapshot above (taken under the lock) lets the all-targets dump
            // surface content in a non-final target instead of discarding it.
            if let Some(targets) = all_targets {
                maybe_dump_all_targets(&targets, present_index);
            }
            return Ok((Some(image), suspended));
        }
        debug!("AGC DCB ran through the command processor without a draw");
        Ok((None, suspended))
    }

    /// Accumulate this submission's shader skips and warn with a process-wide
    /// rate limit (first occurrence, then powers of two).
    fn record_shader_skips(
        &self,
        shader_skips: u64,
        draw_skips: u64,
        dispatch_skips: u64,
        draw_skip_reason: Option<&str>,
        dispatch_skip_reason: Option<&str>,
        shader_state: &kyty_graphics::hw_regs::Shader,
    ) {
        if shader_skips == 0 {
            return;
        }
        let total = {
            let mut skips = self.shader_skip_count.lock();
            *skips += shader_skips;
            *skips
        };
        if total == shader_skips || total.is_power_of_two() {
            // Draw and compute skips are reported with SEPARATE reasons: a
            // title issues far more dispatches than draws, so a single
            // shared reason almost always shows a compute failure and masks
            // why a draw skipped. "draw_reason" is empty when no draw has
            // skipped (its shaders translate) — the wall is then elsewhere.
            warn!(
                total_shader_skips = total,
                draw_skips,
                dispatch_skips,
                draw_reason = draw_skip_reason.unwrap_or("(none — draw shaders translate)"),
                dispatch_reason = dispatch_skip_reason.unwrap_or("(none)"),
                texture_cap_skips = crate::draw_translate::stage_texture_cap_skips(),
                storage_addressing_skips = crate::draw_translate::storage_addressing_skips(),
                // A non-zero count here means a vertex shader was dropped for
                // an unsupported (component count, unified format) attribute
                // pair — the class of failure that held GTA V at 192 flips
                // with no other blocker. The `Spirv::WriteGlobalVariables`
                // error line names the exact pair.
                vertex_input_pair_skips = kyty_graphics::shader::vertex_input_pair_skips(),
                vs_addr = format_args!("{:#x}", shader_state.vs.vs_regs.data_addr),
                es_addr = format_args!("{:#x}", shader_state.vs.es_regs.data_addr),
                gs_addr = format_args!("{:#x}", shader_state.vs.gs_regs.data_addr),
                gs_checksum = format_args!("{:#x}", shader_state.vs.gs_regs.chksum),
                ps_addr = format_args!("{:#x}", shader_state.ps.ps_regs.data_addr),
                stats = ?self.shader_stats(),
                "AGC shader skips (draws + compute dispatches) — see per-path reasons"
            );
        }
    }

    /// Best-effort [`Self::execute_dcb_cp`] for the HLE submit path: a GPU
    /// fault must not become a guest-visible submit failure.
    /// Hand a DCB to the GPU worker and return, the way `sceAgcDriverSubmitDcb`
    /// behaves on hardware: the command buffer goes to the GPU and the caller
    /// carries on.
    ///
    /// Executing the DCB inline on the calling thread instead is what this
    /// exists to stop. A title calls submit from its render thread *while
    /// holding its own mutexes*, so an inline submit holds those locks for as
    /// long as the whole command buffer takes to render. Measured on Minecraft:
    /// 150 ms per submit inside `Rendering Pool(0)`, and the main thread lost
    /// 148.5 s of a 212 s run blocked in `scePthreadMutexLock` on a mutex that
    /// thread owned — 70% of the main thread, spent waiting for our renderer.
    /// The two totals tracked each other across every sample.
    ///
    /// A full queue still blocks the submitter (see [`SUBMIT_QUEUE_DEPTH`]), so
    /// this bounds the damage rather than removing it: a session that renders
    /// slower than the title submits will still stall the submitter, just at
    /// the ring's edge instead of on every single DCB.
    fn submit_dcb_async_owned(self: &Arc<Self>, words: Vec<u32>, is_compute: bool) {
        // Keep admission and enqueue in one critical section. Shutdown first
        // flips Open -> Closing, then drains every submission accepted here;
        // no sender can race a new item behind the Shutdown marker.
        let lifecycle = self.lifecycle.lock();
        if *lifecycle != GpuLifecycle::Open {
            debug!("GPU submission ignored after process teardown began");
            return;
        }
        self.submission_count.fetch_add(1, Ordering::Relaxed);
        let tx = self.submit_queue.get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<GpuWork>(SUBMIT_QUEUE_DEPTH);
            let session = Arc::clone(self);
            let spawned = std::thread::Builder::new()
                .name("raeen-gpu".to_owned())
                .spawn(move || {
                    // RAEEN_TIME_WORKER: windowed worker-occupancy split. `idle`
                    // is time blocked in `recv` (waiting for the guest to submit
                    // the next frame's work); `submit_busy`/`flush_busy` is time
                    // running the GPU path. High idle% => guest/CPU-bound (the
                    // worker starves for work); low idle% => GPU-worker-bound.
                    // Summarized every 32 flips, then the window resets to track
                    // steady state rather than smear the boot burst into it.
                    let timing = crate::diagnostics::gpu_env().time_worker;
                    let mut idle = std::time::Duration::ZERO;
                    let mut submit_busy = std::time::Duration::ZERO;
                    let mut flush_busy = std::time::Duration::ZERO;
                    let mut window_submits: u64 = 0;
                    let mut present_total: u64 = 0;
                    loop {
                        let recv_at = std::time::Instant::now();
                        let work = match rx.recv() {
                            Ok(work) => work,
                            Err(_) => break,
                        };
                        if timing {
                            idle += recv_at.elapsed();
                        }
                        match work {
                            GpuWork::Submit(words, is_compute) => {
                                let _completion = InFlightCompletion(&session);
                                let busy_at = std::time::Instant::now();
                                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    // Stage C: worker submissions defer
                                    // presentation — the flush (readback +
                                    // present) runs once per FLIP, not once
                                    // per submission. Frame dumps are taken
                                    // from that completed present below; a
                                    // diagnostic must not silently restore the
                                    // old flush-per-submission performance
                                    // wall or cease to represent real pacing.
                                    session.worker_submit(words, is_compute, true);
                                }))
                                .is_err()
                                {
                                    warn!("GPU submission panicked; dropping the DCB and keeping the worker alive");
                                }
                                if timing {
                                    submit_busy += busy_at.elapsed();
                                    window_submits += 1;
                                }
                            }
                            GpuWork::Flush {
                                address,
                                queued_at,
                                done,
                                permit,
                            } => {
                                // A suspend flush fences pending work but is not
                                // a displayed VideoOut frame. Count only named
                                // scanout flips in the frame-time denominator;
                                // Minecraft issues both, and treating suspend
                                // fences as frames made worker telemetry report
                                // roughly twice the measured shared-IPC FPS.
                                let is_present = address.is_some();
                                session.present_worker_drain_us.store(
                                    queued_at.elapsed().as_micros() as u64,
                                    Ordering::Relaxed,
                                );
                                let busy_at = std::time::Instant::now();
                                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    session.flush_and_present(address);
                                    // The flush lands deferred readbacks and
                                    // compute storage writebacks — producer
                                    // events for suspended label waits.
                                    session.recheck_suspended_waits();
                                }))
                                .is_err()
                                {
                                    warn!("GPU flush panicked; keeping the worker alive");
                                }
                                // The requester may have given up (recv error
                                // path falls back inline) or not be waiting
                                // at all (flip fire-and-forget); a dead or
                                // absent receiver is fine.
                                if let Some(done) = done {
                                    let _ = done.send(());
                                }
                                // Release the frames-in-flight permit from the
                                // GPU-completion side, AFTER the flush executed
                                // (the fence-signal equivalent on this
                                // single-consumer path). `permit` is bound
                                // outside the `catch_unwind` above, so a
                                // panicking flush still returns its budget —
                                // never a leaked permit that would wedge the
                                // flip thread. `None` for wait_idle flushes.
                                drop(permit);
                                if timing {
                                    flush_busy += busy_at.elapsed();
                                    if is_present {
                                        present_total += 1;
                                    }
                                    if is_present && present_total.is_multiple_of(32) {
                                        let wall = idle + submit_busy + flush_busy;
                                        let busy = submit_busy + flush_busy;
                                        let pct = |d: std::time::Duration| {
                                            let w = wall.as_secs_f64();
                                            if w > 0.0 {
                                                100.0 * d.as_secs_f64() / w
                                            } else {
                                                0.0
                                            }
                                        };
                                        warn!(
                                            flips = present_total,
                                            submits = window_submits,
                                            window_ms = wall.as_millis() as u64,
                                            idle_pct = format_args!("{:.0}", pct(idle)),
                                            busy_pct = format_args!("{:.0}", pct(busy)),
                                            submit_pct = format_args!("{:.0}", pct(submit_busy)),
                                            flush_pct = format_args!("{:.0}", pct(flush_busy)),
                                            frame_ms =
                                                format_args!("{:.1}", wall.as_secs_f64() * 1000.0 / 32.0),
                                            worker_ms =
                                                format_args!("{:.1}", busy.as_secs_f64() * 1000.0 / 32.0),
                                            "WORKER TIMING: idle=waiting-on-guest, busy=GPU worker; high idle_pct => guest/CPU-bound"
                                        );
                                        idle = std::time::Duration::ZERO;
                                        submit_busy = std::time::Duration::ZERO;
                                        flush_busy = std::time::Duration::ZERO;
                                        window_submits = 0;
                                    }
                                }
                            }
                            #[cfg(test)]
                            GpuWork::Panic => {
                                let _completion = InFlightCompletion(&session);
                                let _ = std::panic::catch_unwind(|| panic!("injected GPU panic"));
                            }
                            GpuWork::Shutdown => break,
                        }
                    }
                });
            if let Err(e) = &spawned {
                warn!(error = %e, "cannot start the GPU worker — DCBs will run inline");
            }
            tx
        });
        *self.in_flight.0.lock() += 1;
        // A worker that never started (spawn failed) leaves the receiver
        // dropped, so `send` errors rather than blocking forever: fall back to
        // rendering inline, which is slow but keeps the title drawing.
        if let Err(std::sync::mpsc::SendError(GpuWork::Submit(words, is_compute))) =
            tx.send(GpuWork::Submit(words, is_compute))
        {
            let _completion = InFlightCompletion(self);
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.try_execute_dcb_cp(&words, is_compute);
            }));
        }
        drop(lifecycle);
    }

    fn finish_one(&self) {
        let (lock, cvar) = &self.in_flight;
        let mut n = lock.lock();
        *n = n.saturating_sub(1);
        if *n == 0 {
            cvar.notify_all();
        }
    }

    /// Block until every submitted DCB has been executed.
    ///
    /// Submission is asynchronous, so anything reading a result of rendering
    /// (`last_image`, `draw_count`, a render-target census) must drain first or
    /// it races the worker and reads the frame before the draws land.
    pub fn wait_idle(&self) {
        let (lock, cvar) = &self.in_flight;
        {
            let mut n = lock.lock();
            while *n > 0 {
                cvar.wait(&mut n);
            }
        }
        // wait_idle is a flush consumer (stage C): anything reading a render
        // result afterwards (framebuffer census, draw pixels, shutdown) must
        // see the deferred batch's pixels, not GPU-side-only targets.
        self.consume_flush(None, true);
    }

    pub fn try_execute_dcb_cp(&self, words: &[u32], is_compute: bool) {
        match self.execute_dcb_cp(words, is_compute) {
            Ok(Some(image)) => debug!(
                width = image.width,
                height = image.height,
                "AGC DCB drove a register-state Vulkan draw"
            ),
            Ok(None) => {}
            Err(e) => warn!(error = %e, "AGC DCB draw skipped"),
        }
    }

    /// Worker-thread submission entry (WAIT_REG_MEM suspend/resume; SharpEmu
    /// AgcExports.cs:4508-4529): a queue with a suspended buffer parks new
    /// submissions behind it — in-order ring semantics — otherwise the buffer
    /// runs and may itself suspend. Afterwards every suspended wait is
    /// re-checked: this submission's writebacks (compute storage writebacks,
    /// DMA_DATA copies, WRITE_DATA) are exactly the producer events that
    /// satisfy cross-queue label waits.
    fn worker_submit(&self, words: Vec<u32>, is_compute: bool, deferred_present: bool) {
        let to_run = {
            let mut ws = self.wait_states.lock();
            let queue = ws.queue_mut(is_compute);
            if queue.suspended.is_some() {
                queue.pending.push_back((words, deferred_present));
                None
            } else {
                Some(words)
            }
        };
        if let Some(words) = to_run {
            self.run_worker_buffer(words, 0, is_compute, deferred_present);
        }
        self.recheck_suspended_waits();
    }

    /// Execute a buffer from `start` on the worker; park it when it suspends
    /// on an unmet label wait.
    fn run_worker_buffer(
        &self,
        words: Vec<u32>,
        start: usize,
        is_compute: bool,
        deferred_present: bool,
    ) {
        match self.execute_dcb_cp_routed(&words, start, is_compute, deferred_present) {
            Ok((image, None)) => {
                if let Some(image) = image {
                    debug!(
                        width = image.width,
                        height = image.height,
                        "AGC DCB drove a register-state Vulkan draw"
                    );
                }
            }
            Ok((_, Some(suspended))) => {
                let total = self.wait_suspended_total.fetch_add(1, Ordering::Relaxed) + 1;
                if total <= 8 || total.is_power_of_two() {
                    tracing::info!(
                        queue = if is_compute { "acb" } else { "dcb" },
                        label = format_args!("{:#x}", suspended.wait.address),
                        compare = suspended.wait.compare,
                        reference = format_args!("{:#x}", suspended.wait.reference),
                        resume_dword = suspended.resume_dword,
                        total_suspends = total,
                        "agc.queue_suspended: WAIT_REG_MEM unmet — buffer parked until its label is written"
                    );
                }
                let mut ws = self.wait_states.lock();
                let queue = ws.queue_mut(is_compute);
                debug_assert!(
                    queue.suspended.is_none(),
                    "a queue holds at most one suspended buffer"
                );
                queue.suspended = Some(SuspendedBuffer {
                    words,
                    resume_dword: suspended.resume_dword,
                    wait: suspended.wait,
                    deferred_present,
                    recheck_rounds: 0,
                    latched: false,
                });
            }
            Err(e) => warn!(error = %e, "AGC DCB draw skipped"),
        }
    }

    /// Mark any suspended cross-queue waiter satisfied when a producer packet
    /// (`WRITE_DATA`/`RELEASE_MEM`, drained from the CP via `take_produced_labels`)
    /// wrote a value meeting its condition this submission — the write-time latch
    /// that survives the guest resetting the label same-frame (SharpEmu
    /// `GpuWaitRegistry.RecordProduced`/`LatchSatisfiedByValue`). The label is
    /// never force-written; a genuinely dead wait still parks and is handled by
    /// the dead-wait force-resume net.
    fn latch_produced_waits(&self, produced: &[(u64, u64)]) {
        if produced.is_empty() {
            return;
        }
        let mut ws = self.wait_states.lock();
        for is_compute in [false, true] {
            let queue = ws.queue_mut(is_compute);
            if let Some(buffer) = queue.suspended.as_mut()
                && !buffer.latched
                && produced.iter().any(|&(address, value)| {
                    address == buffer.wait.address && buffer.wait.satisfied_by(value)
                })
            {
                buffer.latched = true;
            }
        }
    }

    /// Re-evaluate every suspended label wait against current guest memory,
    /// resume the satisfied ones from their suspend point, and drain the
    /// submissions parked behind them.
    ///
    /// Port of SharpEmu `DrainResumableDcbs` + `ResumeSuspendedDcb`
    /// (AgcExports.cs:4843-4950): loops to a bounded fixed point because a
    /// resumed buffer's own writebacks can satisfy the *other* queue's wait —
    /// measured there, one compute clear's guest writeback resumed 11
    /// suspended graphics queues (`agc.queue_resumed`) and a real title draw
    /// followed. Never force-satisfies; an unreadable label keeps its buffer
    /// suspended (`GpuWaitRegistry.CollectSatisfied` keeps null-reads
    /// registered), and a wait still unmet after
    /// [`STALE_WAIT_RECHECK_ROUNDS`] rounds warns (then at each doubling) so
    /// dead waits are visible.
    fn recheck_suspended_waits(&self) {
        let Some(memory) = self.guest_memory.lock().clone() else {
            return;
        };
        for _pass in 0..MAX_RESUME_PASSES {
            let mut progressed = false;
            for is_compute in [false, true] {
                let resumable = {
                    let mut ws = self.wait_states.lock();
                    let queue = ws.queue_mut(is_compute);
                    match queue.suspended.as_mut() {
                        None => None,
                        Some(buffer) => {
                            // Latched (a producer wrote a satisfying value this
                            // submission, even if the guest has since reset the
                            // label) OR the live label currently satisfies. Both
                            // are genuine — the label is never force-written
                            // (SharpEmu `LatchSatisfiedByValue` + `CollectSatisfied`).
                            let satisfied = buffer.latched
                                || crate::guest_mem::with_guest_memory(&memory, || {
                                    buffer
                                        .wait
                                        .read_label(&crate::guest_mem::IdentityGuestMemory)
                                        .is_some_and(|value| buffer.wait.satisfied_by(value))
                                });
                            if satisfied {
                                queue.suspended.take()
                            } else {
                                buffer.recheck_rounds += 1;
                                let rounds = buffer.recheck_rounds;
                                if rounds == STALE_WAIT_RECHECK_ROUNDS
                                    || (rounds > STALE_WAIT_RECHECK_ROUNDS
                                        && rounds.is_power_of_two())
                                {
                                    warn!(
                                        queue = if is_compute { "acb" } else { "dcb" },
                                        label = format_args!("{:#x}", buffer.wait.address),
                                        reference = format_args!("{:#x}", buffer.wait.reference),
                                        compare = buffer.wait.compare,
                                        rounds,
                                        parked_behind = queue.pending.len(),
                                        "suspended WAIT_REG_MEM still unmet after many \
                                         producer re-checks — possible dead wait \
                                         (its producer never ran)"
                                    );
                                }
                                if rounds >= DEAD_WAIT_FORCE_RESUME_ROUNDS {
                                    // Confirmed dead: the producer has not run in
                                    // twice the "possible dead wait" window, so it
                                    // never will. Left parked, this WAIT_REG_MEM
                                    // deadlocks the title's GPU submit worker and the
                                    // guest main thread parked behind it forever.
                                    // Force-resume so the queue drains — a possible
                                    // glitch, never a permanent hang.
                                    warn!(
                                        queue = if is_compute { "acb" } else { "dcb" },
                                        label = format_args!("{:#x}", buffer.wait.address),
                                        reference = format_args!("{:#x}", buffer.wait.reference),
                                        compare = buffer.wait.compare,
                                        rounds,
                                        parked_behind = queue.pending.len(),
                                        "FORCE-RESUMING a dead WAIT_REG_MEM to avoid a \
                                         permanent GPU deadlock — its producer never ran"
                                    );
                                    queue.suspended.take()
                                } else {
                                    None
                                }
                            }
                        }
                    }
                };
                let Some(buffer) = resumable else {
                    continue;
                };
                progressed = true;
                let resumed = self.wait_resumed_total.fetch_add(1, Ordering::Relaxed) + 1;
                if resumed <= 8 || resumed.is_power_of_two() {
                    tracing::info!(
                        queue = if is_compute { "acb" } else { "dcb" },
                        label = format_args!("{:#x}", buffer.wait.address),
                        resume_dword = buffer.resume_dword,
                        total_resumes = resumed,
                        "agc.queue_resumed: label satisfied — resuming the parked buffer"
                    );
                }
                self.run_worker_buffer(
                    buffer.words,
                    buffer.resume_dword,
                    is_compute,
                    buffer.deferred_present,
                );
                // Drain submissions parked behind the wait, in order, until
                // the queue suspends again or the backlog empties.
                loop {
                    let next = {
                        let mut ws = self.wait_states.lock();
                        let queue = ws.queue_mut(is_compute);
                        if queue.suspended.is_some() {
                            None
                        } else {
                            queue.pending.pop_front()
                        }
                    };
                    let Some((words, deferred_present)) = next else {
                        break;
                    };
                    self.run_worker_buffer(words, 0, is_compute, deferred_present);
                }
            }
            if !progressed {
                break;
            }
        }
    }

    /// Cumulative WAIT_REG_MEM suspend/resume counters.
    pub fn wait_suspend_stats(&self) -> GpuWaitStats {
        let ws = self.wait_states.lock();
        GpuWaitStats {
            suspended: self.wait_suspended_total.load(Ordering::Relaxed),
            resumed: self.wait_resumed_total.load(Ordering::Relaxed),
            currently_suspended: usize::from(ws.graphics.suspended.is_some())
                + usize::from(ws.compute.suspended.is_some()),
            parked: ws.graphics.pending.len() + ws.compute.pending.len(),
        }
    }

    /// Decode `words` and, if any draw packets are present, rasterize the M2
    /// triangle. Returns `Ok(None)` for sync/flip-only DCBs.
    #[deprecated(
        note = "fixture path: ignores register state and always draws the same triangle. \
                M2 regression gate only — the title path is execute_dcb_cp."
    )]
    pub fn execute_dcb(&self, words: &[u32]) -> Result<Option<RenderedImage>, AgcExecError> {
        let decoded = agc::decode_submission(words)?;
        #[allow(deprecated)]
        self.execute_decoded(&decoded)
    }

    /// Same as [`Self::execute_dcb`] but with a pre-decoded submission.
    #[deprecated(
        note = "fixture path: ignores register state. M2 regression gate only — \
                the title path is execute_dcb_cp."
    )]
    pub fn execute_decoded(
        &self,
        decoded: &AgcSubmission,
    ) -> Result<Option<RenderedImage>, AgcExecError> {
        if decoded.draw_packets == 0 {
            debug!("AGC DCB has no draw packets — skipping Vulkan draw");
            return Ok(None);
        }

        self.ensure_backend()?;
        let image = {
            let guard = self.backend.lock();
            let backend = guard
                .as_ref()
                .expect("ensure_backend left a live VulkanBackend");
            backend.render_m2_triangle(M2_DRAW_WIDTH, M2_DRAW_HEIGHT)?
        };

        self.publish_frame(Arc::new(image.clone()));
        *self.draw_count.lock() += 1;
        Ok(Some(image))
    }

    /// Best-effort draw for HLE: log and continue if Vulkan is unavailable.
    #[deprecated(note = "fixture path. The title path is try_execute_dcb_cp.")]
    pub fn try_execute_decoded(&self, decoded: &AgcSubmission) {
        if decoded.draw_packets == 0 {
            return;
        }
        #[allow(deprecated)]
        match self.execute_decoded(decoded) {
            Ok(Some(_)) => debug!(
                draws = decoded.draw_packets,
                "AGC DCB drove an M2 Vulkan draw"
            ),
            Ok(None) => {}
            Err(e) => warn!(error = %e, "AGC DCB draw skipped — Vulkan unavailable or failed"),
        }
    }
}

/// [`RunOutcome`] → the suspension payload, if any.
const fn suspended_of(outcome: RunOutcome) -> Option<SuspendedWait> {
    match outcome {
        RunOutcome::Completed => None,
        RunOutcome::Suspended(suspended) => Some(suspended),
    }
}

/// Boot splash staged for the NEXT process session. The launcher decodes
/// `sce_sys/pic0.png` before entering the guest (the GPU session is created
/// inside `execute_process`, after the launcher's last chance to touch it), and
/// every launch stages either `Some` or `None` so a previous title's splash can
/// never leak into the next.
fn pending_splash() -> &'static Mutex<Option<Arc<RenderedImage>>> {
    static PENDING: OnceLock<Mutex<Option<Arc<RenderedImage>>>> = OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(None))
}

/// Process-wide GPU settings, mirrored from `EmulatorConfig.graphics` by the
/// Shell (see [`AgcGpuSession::set_runtime_config`]). Read where the Vulkan
/// backend is created (validation) and where each guest draw is sized
/// (resolution scale) — the two settings the user can drive from Settings ▸
/// Video that the GPU path can honour today.
#[derive(Clone)]
pub(crate) struct GpuRuntimeConfig {
    pub validation_layers: bool,
    pub resolution_scale: f32,
    /// Physical-device selection: 0 = auto (best-scored), n ≥ 1 selects the
    /// n-th usable device (1-based), falling back to auto when out of range.
    pub gpu_device_index: u32,
    /// Persist translated SPIR-V and driver pipeline binaries between runs.
    pub shader_cache: bool,
    /// Root of the versioned shader and Vulkan pipeline caches.
    pub shader_cache_dir: std::path::PathBuf,
}

impl Default for GpuRuntimeConfig {
    fn default() -> Self {
        Self {
            validation_layers: false,
            resolution_scale: 1.0,
            gpu_device_index: 0,
            shader_cache: true,
            shader_cache_dir: std::path::PathBuf::from("shader_cache"),
        }
    }
}

fn gpu_runtime_config() -> &'static RwLock<GpuRuntimeConfig> {
    static CFG: OnceLock<RwLock<GpuRuntimeConfig>> = OnceLock::new();
    CFG.get_or_init(|| RwLock::new(GpuRuntimeConfig::default()))
}

fn current_gpu_session() -> &'static RwLock<GpuProcessSession> {
    static SESSION: OnceLock<RwLock<GpuProcessSession>> = OnceLock::new();
    SESSION.get_or_init(|| {
        RwLock::new(GpuProcessSession::create(Arc::new(
            crate::guest_mem::DenyGpuMemory,
        )))
    })
}

/// A state-only submission must never reach a draw. The standalone decoder
/// classified the DCB before this sink is installed; an unexpected draw means
/// the two walkers disagree and is a named error.
struct StateOnlySink;

impl kyty_graphics::run::DrawSink for StateOnlySink {
    fn draw_index_auto(
        &mut self,
        _ctx: &kyty_graphics::hw_regs::Context,
        _ucfg: &kyty_graphics::hw_regs::UserConfig,
        _sh: &kyty_graphics::hw_regs::Shader,
        _index_count: u32,
        _flags: u32,
    ) -> Result<(), kyty_graphics::run::DrawError> {
        Err(kyty_graphics::run::DrawError(
            "AGC decoder classified a draw-bearing DCB as state-only".to_owned(),
        ))
    }
}

/// Write draw output to disk when `RAEEN_DUMP_FRAMES` names a directory —
/// the only way to *see* what a headless `--run-eboot` title actually
/// rendered. Binary PPM (P6), alpha dropped.
///
/// Throttled: the first 8 draws, then powers of two — a title at 60 fps
/// would otherwise write gigabytes and turn the observation into the
/// bottleneck. A failed write logs and is otherwise ignored: frame dumping
/// is diagnostics, never a reason to fail a submit.
/// Dump every accumulated render target once per throttled frame, filename
/// keyed by the target's guest base address, plus a one-line non-black-pixel
/// census per target so the interesting one is greppable without opening PPMs.
fn dump_frame_due(index: u64) -> bool {
    static DUMP_TIMER: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let started = DUMP_TIMER.get_or_init(std::time::Instant::now);
    if std::env::var("RAEEN_DUMP_FRAME_AFTER_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u128>().ok())
        .is_some_and(|after_ms| started.elapsed().as_millis() < after_ms)
    {
        return false;
    }
    if index <= 8 {
        return true;
    }
    static INTERVAL: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    match INTERVAL.get_or_init(|| {
        std::env::var("RAEEN_DUMP_FRAME_INTERVAL")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|interval| *interval != 0)
    }) {
        Some(interval) => index.is_multiple_of(*interval),
        None => index.is_power_of_two(),
    }
}

fn maybe_dump_all_targets(targets: &[(u64, RenderedImage)], draw_index: u64) {
    let Some(dir) = crate::diagnostics::gpu_env().dump_frames.as_deref() else {
        return;
    };
    if dir.is_empty() || !dump_frame_due(draw_index) {
        return;
    }
    for (base, image) in targets {
        let bpp = image.bytes_per_pixel.max(1) as usize;
        let non_black = visible_pixel_count(image, 1);
        let path =
            std::path::Path::new(dir).join(format!("target_{base:012x}_{draw_index:06}.ppm"));
        let mut ppm = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
        ppm.reserve(image.pixels.len() / bpp * 3);
        // First 3 bytes of each pixel as approximate RGB — exact for the 4-byte
        // RGBA/BGRA formats; a rough low-byte view for packed/HDR targets (this
        // is a diagnostic dump, the presented frame is the RGBA8 composite).
        for px in image.pixels.chunks_exact(bpp) {
            ppm.extend_from_slice(&px[..3]);
        }
        let _ = std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, &ppm));
        tracing::info!(
            base = format_args!("{base:#x}"),
            non_black_pixels = non_black,
            total = image.pixels.len() / bpp,
            "render-target census"
        );
    }
}

/// Decode an IEEE-754 half (binary16) to `f32`. Used to unpack an
/// `R16G16B16A16_SFLOAT` HDR render target for sRGB encoding at present.
fn half_to_f32(bits: u16) -> f32 {
    let sign = f32::from((bits >> 15) & 1);
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    let magnitude = if exp == 0 {
        // Subnormal: no implicit leading 1, exponent fixed at -14.
        f32::from(frac) * 2f32.powi(-24)
    } else if exp == 0x1f {
        // Inf/NaN collapse to a large finite value — present clamps anyway.
        if frac == 0 { 65504.0 } else { 0.0 }
    } else {
        (1.0 + f32::from(frac) / 1024.0) * 2f32.powi(i32::from(exp) - 15)
    };
    (1.0 - 2.0 * sign) * magnitude
}

/// Encode a linear-light channel to an 8-bit sRGB byte (the IEC 61966-2-1
/// transfer function). PS5 float VideoOut/render targets hold linear scRGB
/// light where 1.0 is SDR white; hardware scan-out applies this transfer, so a
/// raw numeric copy into an 8-bit display crushes dim scenes to near-black.
///
/// Ported from SharpEmu #448 (327018e): "Encode linear-float flips to sRGB at
/// present" (GPL-2.0-or-later). SharpEmu performs the encode with an sRGB
/// Vulkan store; Raeen presents through the Shell's RGBA8 surface, so the
/// equivalent encode is done here on the CPU.
fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let encoded = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0 + 0.5) as u8
}

/// Lookup tables from a binary16 half's raw bit pattern to the presented byte,
/// built once from the scalar reference functions above. An HDR present
/// otherwise pays three `powf` calls per pixel on every flip (8.3 Mpx × 3 on a
/// 1080p target). The mapping is deterministic per bit pattern — including
/// NaN/Inf patterns, which collapse exactly as the scalar path does — so the
/// tables are exact, not an approximation.
struct HalfPresentLuts {
    /// half bits → sRGB-encoded byte (colour channels).
    srgb: Box<[u8; 65536]>,
    /// half bits → linear-scaled byte (alpha: coverage, not light).
    linear: Box<[u8; 65536]>,
}

fn half_present_luts() -> &'static HalfPresentLuts {
    static LUTS: OnceLock<HalfPresentLuts> = OnceLock::new();
    LUTS.get_or_init(|| {
        let mut srgb = Box::new([0u8; 65536]);
        let mut linear = Box::new([0u8; 65536]);
        for (bits, (srgb_out, linear_out)) in srgb.iter_mut().zip(linear.iter_mut()).enumerate() {
            let value = half_to_f32(bits as u16);
            *srgb_out = linear_to_srgb_u8(value);
            *linear_out = (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
        HalfPresentLuts { srgb, linear }
    })
}

/// Allocate an initialized byte buffer without invoking Rust's aborting
/// allocation-error handler. Full-HD presentation buffers are 8 MiB; under
/// memory pressure a missed frame must preserve the previous completed frame,
/// not terminate the isolated runner.
fn try_zeroed_bytes(len: usize) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).ok()?;
    bytes.resize(len, 0);
    Some(bytes)
}

/// Convert a render target to the RGBA8 bytes the Shell present surface and the
/// PPM frame dump expect. An `R16G16B16A16_SFLOAT` HDR target (8 bytes/pixel)
/// is unpacked and sRGB-encoded (SharpEmu #448); everything already 4 bytes per
/// pixel is shared unchanged.
///
/// The conversion reads the source through its `Arc` instead of cloning the
/// whole 16 MiB 1080p HDR frame before allocating the 8 MiB output.
fn try_to_presentable_arc(image: Arc<RenderedImage>) -> Option<Arc<RenderedImage>> {
    if image.bytes_per_pixel != 8 {
        return Some(image);
    }
    let px_count = (image.width as usize).checked_mul(image.height as usize)?;
    let input_bytes = px_count.checked_mul(8)?;
    let output_bytes = px_count.checked_mul(4)?;
    if image.pixels.len() < input_bytes {
        return None;
    }
    let mut pixels = try_zeroed_bytes(output_bytes)?;
    let luts = half_present_luts();
    for (texel, out) in image.pixels.chunks_exact(8).zip(pixels.chunks_exact_mut(4)) {
        let r = u16::from_le_bytes([texel[0], texel[1]]);
        let g = u16::from_le_bytes([texel[2], texel[3]]);
        let b = u16::from_le_bytes([texel[4], texel[5]]);
        let a = u16::from_le_bytes([texel[6], texel[7]]);
        out[0] = luts.srgb[r as usize];
        out[1] = luts.srgb[g as usize];
        out[2] = luts.srgb[b as usize];
        // Alpha is a coverage value, not light — keep it linear.
        out[3] = luts.linear[a as usize];
    }
    Some(Arc::new(RenderedImage {
        width: image.width,
        height: image.height,
        pixels,
        bytes_per_pixel: 4,
    }))
}

fn warn_present_allocation_failed() {
    static WARNINGS: AtomicU64 = AtomicU64::new(0);
    let occurrence = WARNINGS.fetch_add(1, Ordering::Relaxed) + 1;
    if occurrence <= 8 || occurrence.is_power_of_two() {
        warn!(
            occurrence,
            "presentation frame allocation failed under host memory pressure; \
             preserving the last complete frame"
        );
    }
}

/// Number of frames presented since process start — the frame-dump sampling
/// index (see the call site for why the draw counter cannot serve this role).
static PRESENT_INDEX: AtomicU64 = AtomicU64::new(0);

fn maybe_dump_frame(image: &RenderedImage, draw_index: u64) {
    let Some(dir) = crate::diagnostics::gpu_env().dump_frames.as_deref() else {
        return;
    };
    if dir.is_empty() || !dump_frame_due(draw_index) {
        return;
    }
    let path = std::path::Path::new(dir).join(format!("frame_{draw_index:06}.ppm"));
    let mut ppm = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
    ppm.reserve(image.pixels.len() / 4 * 3);
    for rgba in image.pixels.chunks_exact(4) {
        ppm.extend_from_slice(&rgba[..3]);
    }
    match std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, &ppm)) {
        Ok(()) => tracing::info!(
            path = %path.display(),
            width = image.width,
            height = image.height,
            "dumped rendered frame"
        ),
        Err(e) => warn!(error = %e, path = %path.display(), "frame dump failed"),
    }
}

/// Gen5 type-3 header: total packet length `dwords`, opcode, register.
fn agc_header(dwords: u32, opcode: u32, register: u32) -> u32 {
    debug_assert!(dwords >= 2);
    0xc000_0000 | ((dwords - 2) << 16) | (opcode << 8) | (register << 2)
}

/// Fixture DCB: one `DRAW_INDEX_AUTO` with vertex count 3.
///
/// Carries **no register state at all**, which is why it can only drive the
/// fixture path — through a real command processor it has no render target and
/// must not draw.
#[deprecated(note = "fixture DCB with no register state; use build_cp_draw_dcb for the CP path")]
pub fn build_m2_draw_dcb() -> Vec<u32> {
    vec![agc_header(2, IT_NOP, R_DRAW_INDEX_AUTO), 3]
}

/// Which half of the target the acceptance DCB's scissor selects.
///
/// Two DCBs differing in exactly one register value must produce mirror-image
/// output; no hardcoded renderer can do that.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScissorHalf {
    Left,
    Right,
}

/// Build a register-complete Gen5 DCB that draws a full-target clear quad via
/// Kyty's embedded shaders, scissored to one half.
///
/// Every draw parameter is programmed as a real PM4 packet — this is the DCB
/// the Phase 1 acceptance test runs, and nothing about the resulting image is
/// hardcoded on the host side:
///
/// | Packet | Register | Drives |
/// |---|---|---|
/// | `SET_CONTEXT_REG` | `CB_COLOR0_BASE` | render target bound at all |
/// | `SET_CONTEXT_REG` | `CB_COLOR0_INFO` | `R8G8B8A8_UNORM` |
/// | `SET_CONTEXT_REG` | `CB_COLOR0_ATTRIB2` | the `width` x `height` extent |
/// | `SET_CONTEXT_REG` | `CB_TARGET_MASK` | colour writes enabled |
/// | `SET_CONTEXT_REG` | `PA_CL_VPORT_XSCALE`+5 | the viewport |
/// | `SET_CONTEXT_REG` | `PA_SC_SCREEN_SCISSOR_TL/BR` | the rasterized half |
/// | `SET_UCONFIG_REG` | `VGT_PRIMITIVE_TYPE` | RectList |
/// | `NOP R_VS_EMBEDDED` / `R_PS_EMBEDDED` | — | the shaders |
/// | `NOP R_DRAW_INDEX_AUTO` | — | the draw |
#[must_use]
pub fn build_cp_draw_dcb(width: u32, height: u32, half: ScissorHalf) -> Vec<u32> {
    let mut dcb = Vec::new();

    let mut set_cx = |reg: u32, values: &[u32]| {
        dcb.push(pm4::header(
            (values.len() + 2) as u16,
            pm4::IT_SET_CONTEXT_REG,
            pm4::R_ZERO,
        ));
        dcb.push(reg);
        dcb.extend_from_slice(values);
    };

    // A non-zero base is what distinguishes "bound" from NoColorOutput; the
    // address is never dereferenced on the offscreen path.
    set_cx(pm4::CB_COLOR0_BASE, &[0x1_0000 >> 8]);
    // format=0xa (8_8_8_8), channel_type=0 (unorm), channel_order=0 (RGBA).
    set_cx(pm4::CB_COLOR0_INFO, &[0xa << 2]);
    // ATTRIB2 stores width/height minus one: MIP0_WIDTH at 14, MIP0_HEIGHT at 0.
    set_cx(
        pm4::CB_COLOR0_ATTRIB2,
        &[((width - 1) << 14) | (height - 1)],
    );
    set_cx(pm4::CB_TARGET_MASK, &[0xF]);

    // Viewport: x = xoffset - xscale = 0, w = xscale * 2 = width.
    let (hw, hh) = (width as f32 / 2.0, height as f32 / 2.0);
    set_cx(
        pm4::PA_CL_VPORT_XSCALE,
        &[
            hw.to_bits(),
            hw.to_bits(),
            hh.to_bits(),
            hh.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
        ],
    );

    let mid = width / 2;
    let (tl_x, br_x) = match half {
        ScissorHalf::Left => (0, mid),
        ScissorHalf::Right => (mid, width),
    };
    set_cx(pm4::PA_SC_SCREEN_SCISSOR_TL, &[tl_x, br_x | (height << 16)]);

    // RectList — Kyty's clear primitive, which the embedded VS expects.
    dcb.push(pm4::header(3, pm4::IT_SET_UCONFIG_REG, pm4::R_ZERO));
    dcb.push(pm4::VGT_PRIMITIVE_TYPE);
    dcb.push(17);

    // Embedded shaders: id 0 for both. Kyty declares these packets at fixed
    // lengths (29 and 40 total dwords), most of which is unread payload.
    dcb.push(pm4::header(29, pm4::IT_NOP, pm4::R_VS_EMBEDDED));
    dcb.push(0); // shader_modifier
    dcb.push(0); // id
    dcb.resize(dcb.len() + 26, 0);

    dcb.push(pm4::header(40, pm4::IT_NOP, pm4::R_PS_EMBEDDED));
    dcb.push(0); // id
    dcb.resize(dcb.len() + 38, 0);

    dcb.push(pm4::header(7, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO));
    dcb.push(3); // index_count
    dcb.push(0); // flags
    dcb.resize(dcb.len() + 4, 0);

    dcb
}

#[cfg(test)]
mod tests {
    use super::*;
    use raeen_core::subsystems::GpuSubmissionSubsystem;
    use std::ffi::OsStr;

    fn deny_memory() -> Arc<dyn crate::guest_mem::GpuGuestMemory> {
        Arc::new(crate::guest_mem::DenyGpuMemory)
    }

    #[test]
    fn async_flip_defaults_on_with_an_explicit_opt_out() {
        assert!(
            async_flip_enabled(None),
            "bounded async flip is the production default (Phase 2.1 gate green 2026-07-26)"
        );
        for enabled in ["1", "true", "YES", " on ", "invalid"] {
            assert!(
                async_flip_enabled(Some(OsStr::new(enabled))),
                "{enabled:?} should leave the bounded path enabled"
            );
        }
        for disabled in ["0", "false", "no", "off", " OFF "] {
            assert!(
                !async_flip_enabled(Some(OsStr::new(disabled))),
                "{disabled:?} must restore synchronous presentation"
            );
        }
    }

    /// The frames-in-flight semaphore (item 7 / rank 7). Proves the three
    /// properties the bounded fire-and-forget flip relies on: the cap is
    /// enforced, a blocked acquire is unblocked by a release from ANOTHER
    /// thread (the GPU-completion side, modelling a slow worker — no
    /// deadlock), and permits stay balanced and capped across acquire/release.
    #[test]
    fn flip_semaphore_caps_in_flight_and_stays_balanced() {
        let sem = FlipSemaphore::new(2);
        let p1 = sem.acquire();
        let _p2 = sem.acquire();
        assert_eq!(
            *sem.available.lock(),
            0,
            "both permits taken (cap enforced)"
        );

        // A third acquire must block until a permit is freed. Release it from
        // a separate thread AFTER a delay — the "slow worker" case. If acquire
        // deadlocked or the release path were broken, this test would hang.
        let releaser_sem = Arc::clone(&sem);
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            drop(p1); // GPU-completion side returns the permit
            drop(releaser_sem);
        });
        let p3 = sem.acquire(); // blocks ~50 ms, then proceeds
        releaser.join().expect("releaser thread");
        assert_eq!(*sem.available.lock(), 0, "reacquired the freed permit");

        drop(_p2);
        drop(p3);
        assert_eq!(*sem.available.lock(), 2, "all permits returned");

        // A stray extra release must not inflate the budget past the cap.
        sem.release();
        assert_eq!(*sem.available.lock(), 2, "release is capped at the max");
    }

    /// The whole bounded flip path through a real session: with a worker up,
    /// every `present_scanout` takes the async enqueue path, and however many
    /// flips are issued past the in-flight cap, they are all accepted (the
    /// flip thread never wedges) and every permit is returned once the worker
    /// drains. This is the permit acquire/release balance under the actual
    /// worker, plus the "no deadlock past the cap" contract.
    #[test]
    fn bounded_flip_enqueues_async_and_returns_all_permits() {
        let session = GpuProcessSession(Arc::new(AgcGpuSession::new_with_async_flip(
            deny_memory(),
            true,
        )));
        // Stand up the GPU worker; a worker-less session flushes inline and
        // would never exercise the permit path.
        session.submit_dcb_async(build_state_only_dcb(), false);
        session.wait_idle();
        // A framebuffer entry at the flip address so each flush presents
        // deterministically without a Vulkan device or a guest-memory read.
        session.framebuffers.lock().insert(
            0x1000,
            Arc::new(RenderedImage {
                width: 1,
                height: 1,
                pixels: vec![7, 8, 9, 255],
                bytes_per_pixel: 4,
            }),
        );
        // Far more flips than the cap: each acquires a permit before enqueuing
        // and blocks (on the semaphore or the full channel) only transiently
        // while the worker drains — never a wedge.
        for _ in 0..FLIP_FRAMES_IN_FLIGHT * 8 {
            session.present_scanout(0x1000, None);
        }
        // The worker is a single FIFO consumer, so this synchronous drain runs
        // behind every flip flush; by the time it returns each flush has
        // executed and dropped its permit.
        session.wait_idle();
        assert_eq!(
            *session.flip_permits.available.lock(),
            FLIP_FRAMES_IN_FLIGHT,
            "every flip permit returned after the worker drained"
        );
        session.shutdown();
    }

    /// A flip that falls back to the inline path (no worker was ever started)
    /// must still balance its permit — it acquires none it cannot return, and
    /// the flush happens synchronously on the caller.
    #[test]
    fn worker_less_flip_stays_permit_balanced() {
        let session = AgcGpuSession::new(deny_memory());
        assert_eq!(
            *session.flip_permits.available.lock(),
            FLIP_FRAMES_IN_FLIGHT
        );
        // No submit => no worker => inline flush. present_scanout to an empty
        // target simply keeps the (absent) frame; the point is permit balance.
        session.present_scanout(0xDEAD_BEEF, None);
        assert_eq!(
            *session.flip_permits.available.lock(),
            FLIP_FRAMES_IN_FLIGHT,
            "the inline fallback acquires and returns no net permits"
        );
    }

    #[test]
    fn half_to_f32_decodes_representative_values() {
        assert_eq!(half_to_f32(0x0000), 0.0, "positive zero");
        assert_eq!(half_to_f32(0x3C00), 1.0, "one");
        assert_eq!(half_to_f32(0x3800), 0.5, "half");
        assert_eq!(half_to_f32(0x4000), 2.0, "two");
        assert_eq!(half_to_f32(0xBC00), -1.0, "negative one");
    }

    /// SharpEmu #448: a linear-float HDR target (8 B/px) is sRGB-encoded to
    /// RGBA8 at present, so dim linear values are lifted by the transfer instead
    /// of copied numerically into an 8-bit display (which crushes them to black).
    #[test]
    fn to_presentable_srgb_encodes_a_float_hdr_target() {
        // One pixel: R=1.0, G=0.5, B=0.0 (linear), A=1.0, as four LE halves.
        let mut pixels = Vec::new();
        for half in [0x3C00u16, 0x3800, 0x0000, 0x3C00] {
            pixels.extend_from_slice(&half.to_le_bytes());
        }
        let hdr = RenderedImage {
            width: 1,
            height: 1,
            pixels,
            bytes_per_pixel: 8,
        };
        let out = try_to_presentable_arc(Arc::new(hdr)).expect("small HDR conversion");
        assert_eq!(out.bytes_per_pixel, 4, "float target becomes RGBA8");
        assert_eq!(out.pixels.len(), 4);
        assert_eq!(out.pixels[0], 255, "linear 1.0 -> sRGB 255");
        // Linear 0.5 sRGB-encodes to ~0.7354 -> 188, NOT the numeric 128 a raw
        // copy would give (the whole point of the encode).
        assert_eq!(out.pixels[1], 188, "linear 0.5 -> sRGB 188");
        assert!(
            out.pixels[1] > 128,
            "sRGB lifts mid-grey above the linear byte"
        );
        assert_eq!(out.pixels[2], 0, "linear 0.0 -> 0");
        assert_eq!(out.pixels[3], 255, "alpha stays linear");
    }

    /// The LUT present path must be exact, not approximate: every one of the
    /// 65536 binary16 patterns must produce the same byte as the scalar
    /// reference (half decode + transfer function), including NaN/Inf patterns.
    #[test]
    fn half_present_luts_match_the_scalar_reference_for_every_pattern() {
        let luts = half_present_luts();
        for bits in 0..=u16::MAX {
            let value = half_to_f32(bits);
            assert_eq!(
                luts.srgb[bits as usize],
                linear_to_srgb_u8(value),
                "sRGB LUT diverges at half {bits:#06x}"
            );
            assert_eq!(
                luts.linear[bits as usize],
                (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                "linear LUT diverges at half {bits:#06x}"
            );
        }
    }

    /// An already-8-bit frame passes through `to_presentable` unchanged (the
    /// common present path must not pay for a needless re-encode).
    #[test]
    fn to_presentable_passes_through_rgba8() {
        let frame = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![1, 2, 3, 255, 4, 5, 6, 255],
            bytes_per_pixel: 4,
        };
        let input = Arc::new(frame.clone());
        let out = try_to_presentable_arc(Arc::clone(&input)).expect("RGBA passthrough");
        assert!(
            Arc::ptr_eq(&out, &input),
            "the common RGBA8 path must not copy a frame"
        );
        assert_eq!(out.pixels, frame.pixels);
        assert_eq!(out.bytes_per_pixel, 4);
    }

    #[test]
    fn synchronous_flip_does_not_retain_an_async_scanout_snapshot() {
        let mut backing = vec![1u8, 2, 3, 255, 4, 5, 6, 255];
        let base = backing.as_mut_ptr() as u64;
        let session = AgcGpuSession::new_with_async_flip(
            Arc::new(HostRangeMemory {
                start: base,
                len: backing.len() as u64,
            }),
            false,
        );
        let desc = raeen_core::subsystems::ScanoutDescriptor {
            width: 2,
            height: 1,
            pitch_pixels: 2,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        };

        session.present_scanout(base, Some(desc));

        assert!(
            session.guest_scanout_snapshots.lock().is_empty(),
            "the synchronous path consumes guest memory before the flip returns; \
             retaining an async race-avoidance snapshot is an 8 MiB-per-buffer waste"
        );
    }

    #[test]
    fn fallback_composite_publishes_the_complete_intermediate_without_redecoding_it() {
        let mut scanout = vec![0u8; 8];
        let scanout_base = scanout.as_mut_ptr() as u64;
        let session = AgcGpuSession::new_with_async_flip(
            Arc::new(HostRangeMemory {
                start: scanout_base,
                len: scanout.len() as u64,
            }),
            false,
        );
        let desc = raeen_core::subsystems::ScanoutDescriptor {
            width: 2,
            height: 1,
            pitch_pixels: 2,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        };
        let intermediate = Arc::new(RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![10, 20, 30, 255, 90, 80, 70, 255],
            bytes_per_pixel: 4,
        });
        session
            .framebuffers
            .lock()
            .insert(0x1234_0000, Arc::clone(&intermediate));

        session.present_scanout(scanout_base, Some(desc));

        let presented = session.last_image().expect("fallback frame is published");
        assert!(
            Arc::ptr_eq(&presented, &intermediate),
            "the compatibility composite already owns a complete RGBA frame; reading the \
             just-written guest scanout back allocated and copied a second full frame"
        );
        assert_eq!(scanout, intermediate.pixels);
    }

    /// A DCB that only writes registers: no draw and no dispatch, so
    /// `execute_dcb_cp` takes its state-only path and never brings up Vulkan.
    fn build_state_only_dcb() -> Vec<u32> {
        vec![
            pm4::header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::CB_TARGET_MASK,
            0xF,
        ]
    }

    /// `submit_dcb_async` must not lose or deadlock on a DCB, and `wait_idle`
    /// must not return while work is outstanding — everything that reads a
    /// render result depends on that contract.
    ///
    /// Submits more than [`SUBMIT_QUEUE_DEPTH`] so the queue fills and `send`
    /// blocks: that is the backpressure path, and it must drain and finish
    /// rather than wedge the submitter against its own worker.
    ///
    /// Uses a private session rather than `global()` — the global is shared with
    /// every other test in the process, so its in-flight count is not this
    /// test's to assert on. That private session is also why the DCBs are
    /// state-only: a drawing DCB would stand up a SECOND Vulkan device beside
    /// the global session's, which raced the other tests' device and panicked
    /// inside ash under a parallel `cargo test --workspace`. The queue contract
    /// is what is under test here; rendering is covered by the M2 fixture.
    #[test]
    fn async_submit_drains_every_dcb_including_past_the_queue_depth() {
        let session = GpuProcessSession::create(deny_memory());
        let words = build_state_only_dcb();
        for _ in 0..SUBMIT_QUEUE_DEPTH * 3 {
            session.submit_dcb_async(words.clone(), false);
        }
        session.wait_idle();
        assert_eq!(
            *session.in_flight.0.lock(),
            0,
            "wait_idle returned with DCBs still in flight"
        );
        session.shutdown();
    }

    #[test]
    fn a_new_process_gets_fresh_gpu_submission_and_register_state() {
        let first = AgcGpuSession::new_process(deny_memory());
        first.submit_dcb_async(build_state_only_dcb(), false);
        first.wait_idle();
        assert_eq!(first.stats().submitted, 1);
        first.shutdown();

        let second = AgcGpuSession::new_process(deny_memory());
        AgcGpuSession::install_process(&second);
        assert_eq!(
            second.stats(),
            raeen_core::subsystems::GpuSubmissionStats::default()
        );
        assert_eq!(AgcGpuSession::global().stats().submitted, 0);
        second.shutdown();
    }

    #[test]
    fn panicking_gpu_work_still_completes_and_shutdown_returns() {
        let session = GpuProcessSession::create(deny_memory());
        session.submit_dcb_async(build_state_only_dcb(), false);
        session.wait_idle();
        *session.in_flight.0.lock() += 1;
        session
            .submit_queue
            .get()
            .expect("worker queue initialized")
            .send(GpuWork::Panic)
            .expect("worker accepts injected panic");
        session.wait_idle();
        assert_eq!(*session.in_flight.0.lock(), 0);
        session.shutdown();
    }

    #[test]
    fn concurrent_submit_and_shutdown_cannot_enqueue_behind_shutdown() {
        let session = GpuProcessSession::create(deny_memory());
        let submitter = session.clone();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let submit_barrier = Arc::clone(&barrier);
        let thread = std::thread::spawn(move || {
            submit_barrier.wait();
            submitter.submit_dcb_async(build_state_only_dcb(), false);
        });

        barrier.wait();
        session.shutdown();
        thread.join().expect("submitter must finish");
        session.wait_idle();
        let accepted = session.stats().submitted;
        assert!(accepted <= 1);

        // A stale observer clone after shutdown must be a no-op, not an item
        // queued behind the worker's terminal marker.
        session.submit_dcb_async(build_state_only_dcb(), false);
        assert_eq!(session.stats().submitted, accepted);
        session.wait_idle();
    }

    /// The present-routing contract: a title composites into several render
    /// targets and flips to one via `sceVideoOutSubmitFlip`. `present_scanout`
    /// must present the buffer the title flipped to (looked up in
    /// `framebuffers` by its guest base address), NOT the last-drawn target
    /// (often a black background composited over later) — and must never
    /// regress to a stale/black frame when the flipped buffer has no drawn
    /// content yet.
    #[test]
    fn present_scanout_prefers_the_flipped_buffer_and_never_regresses() {
        let session = AgcGpuSession::new(deny_memory());
        let content = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![10, 20, 30, 255, 40, 50, 60, 255],
            bytes_per_pixel: 4,
        };
        let black = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 255, 0, 0, 0, 255],
            bytes_per_pixel: 4,
        };
        // The GPU drew content to the render target at guest base 0x1000; the
        // last-drawn image happens to be the black background.
        session
            .framebuffers
            .lock()
            .insert(0x1000, Arc::new(content.clone()));
        *session.last_image.lock() = Some(Arc::new(black));

        // Flip to the registered content buffer -> present THAT buffer.
        session.present_scanout(0x1000, None);
        assert_eq!(session.last_image().unwrap().pixels, content.pixels);

        // Flip to a buffer with no drawn content -> keep the current frame.
        session.present_scanout(0xDEAD_BEEF, None);
        assert_eq!(session.last_image().unwrap().pixels, content.pixels);

        // Address 0 is not a flip target and is ignored.
        session.present_scanout(0, None);
        assert_eq!(session.last_image().unwrap().pixels, content.pixels);
    }

    /// A render-target entry at the flip address is not automatically a valid
    /// frame. Minecraft keeps cleared scanout targets while drawing its UI to
    /// an intermediate target; existence-only routing presented the cleared
    /// buffer and suppressed the visible UI forever.
    #[test]
    fn black_flipped_target_falls_back_to_visible_intermediate_target() {
        let session = AgcGpuSession::new(deny_memory());
        let black_scanout = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 255, 0, 0, 0, 255],
            bytes_per_pixel: 4,
        };
        let visible_ui = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![0, 40, 0, 255, 0, 0, 90, 255],
            bytes_per_pixel: 4,
        };
        session
            .framebuffers
            .lock()
            .insert(0x1000, Arc::new(black_scanout));
        session
            .framebuffers
            .lock()
            .insert(0x2000, Arc::new(visible_ui.clone()));

        session.present_scanout(0x1000, None);

        assert_eq!(
            session.last_image().expect("fallback frame").pixels,
            visible_ui.pixels,
            "a cleared scanout must not suppress a visible intermediate target"
        );
        assert_eq!(
            *session.fallback_present_base.lock(),
            Some(0x2000),
            "the content-bearing intermediate becomes the steady-state fallback"
        );
    }

    /// Minecraft's next screen clears both registered VideoOut buffers to an
    /// opaque light gray while drawing the actual menu into an intermediate.
    /// A visible-pixel score ranks that clear as a perfect frame, so routing
    /// must prefer the non-uniform UI even when the scanout was drawn this
    /// flush.
    #[test]
    fn uniform_scanout_yields_to_detailed_intermediate_target() {
        let session = AgcGpuSession::new(deny_memory());
        let gray_scanout = RenderedImage {
            width: 2,
            height: 2,
            pixels: [229, 231, 234, 255].repeat(4),
            bytes_per_pixel: 4,
        };
        let detailed_ui = RenderedImage {
            width: 2,
            height: 2,
            pixels: vec![
                0, 0, 0, 255, 240, 240, 240, 255, 20, 80, 140, 255, 0, 0, 0, 255,
            ],
            bytes_per_pixel: 4,
        };
        session
            .framebuffers
            .lock()
            .insert(0x1000, Arc::new(gray_scanout));
        session
            .framebuffers
            .lock()
            .insert(0x2000, Arc::new(detailed_ui.clone()));

        session.present_flipped(0x1000, true);

        assert_eq!(
            session.last_image().expect("detailed frame").pixels,
            detailed_ui.pixels,
            "a uniform VideoOut clear must not hide a completed UI intermediate"
        );
        assert_eq!(*session.fallback_present_base.lock(), Some(0x2000));
    }

    /// With no detailed target to replace it, a solid frame remains valid.
    /// This preserves intentional fades and loading-screen clears.
    #[test]
    fn uniform_scanout_remains_valid_without_a_detailed_target() {
        let session = AgcGpuSession::new(deny_memory());
        let solid = RenderedImage {
            width: 2,
            height: 1,
            pixels: [12, 24, 36, 255].repeat(2),
            bytes_per_pixel: 4,
        };
        session
            .framebuffers
            .lock()
            .insert(0x1000, Arc::new(solid.clone()));

        session.present_flipped(0x1000, true);

        assert_eq!(
            session.last_image().expect("solid frame").pixels,
            solid.pixels
        );
        assert_eq!(*session.fallback_present_base.lock(), None);
    }

    /// An HDR (RGBA16F) texel spends TWO bytes per channel, so its colour span
    /// is six bytes. Testing three covered red plus only the LOW half of green
    /// and never reached blue — and half-float `1.0` is `0x3C00`, whose
    /// little-endian low byte is zero — so a pure-green or blue-dominant HDR
    /// frame scanned as entirely black and was discarded as never-drawn.
    #[test]
    fn hdr_green_and_blue_frames_count_as_visible_content() {
        let one = 0x3C00u16.to_le_bytes(); // half-float 1.0
        let off = [0u8, 0];
        let texel = |r: [u8; 2], g: [u8; 2], b: [u8; 2]| RenderedImage {
            width: 1,
            height: 1,
            pixels: [r, g, b, one].concat(),
            bytes_per_pixel: 8,
        };

        assert!(
            has_visible_content(&texel(off, one, off)),
            "a pure-green HDR frame is visible colour"
        );
        assert!(
            has_visible_content(&texel(off, off, one)),
            "a pure-blue HDR frame is visible colour"
        );
        assert!(
            has_visible_content(&texel(one, off, off)),
            "a pure-red HDR frame is visible colour"
        );
        // The alpha exclusion still holds: a cleared HDR target carrying only
        // opaque alpha is not content.
        assert!(
            !has_visible_content(&texel(off, off, off)),
            "alpha alone is not visible over the Shell's black background"
        );
    }

    #[test]
    fn suspend_readback_filter_reuses_known_presentation_targets() {
        assert_eq!(
            presentation_filter_bases(None, Some(0x31bf_0000)),
            Vec::<u64>::new(),
            "suspend fences work but defers CPU pixels to the next flip"
        );
        assert_eq!(
            presentation_filter_bases(Some(0x1f7d_0000), Some(0x31bf_0000)),
            vec![0x1f7d_0000, 0x31bf_0000],
            "an explicit flip reads its scanout and the routed fallback"
        );
        assert_eq!(
            presentation_filter_bases(Some(0x31bf_0000), Some(0x31bf_0000)),
            vec![0x31bf_0000],
            "the same scanout/fallback target is read only once"
        );
        assert!(
            presentation_filter_bases(None, None).is_empty(),
            "a pre-first-flip suspend fences work without speculative target readback"
        );
    }

    #[test]
    fn worker_compute_batches_until_an_ordered_lifetime_boundary() {
        assert!(
            submission_compute_flush_required(true, false),
            "synchronous callers need submission-local guest visibility"
        );
        assert!(
            !submission_compute_flush_required(true, true),
            "the ordered worker batches adjacent compute submissions; the draw sink fences at \
             the first compute-to-graphics dependency, otherwise suspend/flip does"
        );
        assert!(
            !submission_compute_flush_required(false, false),
            "a submission with no queued compute has nothing to fence"
        );
    }

    /// A frame the GPU actually rendered into is presented even when it is
    /// entirely black — a fade-out, a night scene, a dark loading screen.
    /// Pixel content alone cannot tell "drew black" from "never drawn", so
    /// judging by content froze such frames on the last bright image.
    #[test]
    fn black_frame_the_gpu_drew_into_is_presented_not_replaced() {
        let session = AgcGpuSession::new(deny_memory());
        let drawn_black = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 255, 0, 0, 0, 255],
            bytes_per_pixel: 4,
        };
        let stale_bright = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![0, 40, 0, 255, 0, 0, 90, 255],
            bytes_per_pixel: 4,
        };
        session
            .framebuffers
            .lock()
            .insert(0x1000, Arc::new(drawn_black.clone()));
        session
            .framebuffers
            .lock()
            .insert(0x2000, Arc::new(stale_bright));

        // `true` = this flush read the flipped target back, i.e. the GPU
        // rendered into the very buffer the title is presenting.
        session.present_flipped(0x1000, true);

        assert_eq!(
            session.last_image().expect("presented frame").pixels,
            drawn_black.pixels,
            "a black frame the title actually drew must not be replaced by a stale target"
        );
    }

    /// The present hand-off shares ONE allocation end to end: the flush stores
    /// the frame, and every Shell read (`shell/present.rs` calls `last_image()`
    /// each repaint) hands back the same `Arc` — no per-frame ~8 MB copy. This
    /// is the fps fix's invariant AND its no-regression proof: the exact bytes
    /// the flush produced are the bytes the Shell paints, so the frame still
    /// reaches the window unchanged.
    #[test]
    fn present_hands_off_the_frame_without_copying_pixels() {
        let session = AgcGpuSession::new(deny_memory());
        let content = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![10, 20, 30, 255, 40, 50, 60, 255],
            bytes_per_pixel: 4,
        };
        session
            .framebuffers
            .lock()
            .insert(0x2000, Arc::new(content.clone()));
        session.hide_splash();

        // The title flips to the drawn buffer -> the flush presents it.
        session.present_scanout(0x2000, None);

        let first = session.last_image().expect("flip presented a frame");
        let second = session.last_image().expect("frame still present");
        assert!(
            Arc::ptr_eq(&first, &second),
            "last_image() must hand back a shared Arc, not a per-read clone"
        );
        assert_eq!(
            first.pixels, content.pixels,
            "the Shell paints exactly the drawn frame"
        );
    }

    /// `present_scanout` now enqueues the flush and returns WITHOUT the per-flip
    /// `wait_for_fences` stall (the fps lever), and publishes only COMPLETE
    /// frames. For a GPU-drawn target (the Minecraft path) the presented frame
    /// is exactly the drawn frame — never a partial/black one — and every
    /// publish advances `present_epoch`, the signal the Shell refreshes its
    /// on-screen texture on. This is the async flip's no-regression proof: the
    /// bytes the flush produced are the bytes that reach the window.
    #[test]
    fn async_flip_publishes_the_complete_drawn_frame_and_bumps_the_epoch() {
        let session = AgcGpuSession::new(deny_memory());
        let content = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![11, 22, 33, 255, 44, 55, 66, 255],
            bytes_per_pixel: 4,
        };
        session
            .framebuffers
            .lock()
            .insert(0x3000, Arc::new(content.clone()));
        session.hide_splash();
        let before = session.present_epoch();

        // No worker is started in-test, so the async enqueue runs the flush
        // inline — but through the same `submit_flip_flush` path the guest flip
        // thread now takes, exercising the publish + epoch invariants.
        session.present_scanout(0x3000, None);

        let img = session.last_image().expect("async flip presented a frame");
        assert_eq!(
            img.pixels, content.pixels,
            "the presented frame is exactly the drawn frame — complete, never partial"
        );
        assert!(
            session.present_epoch() > before,
            "publishing a complete frame advances the present epoch the Shell refreshes on"
        );
    }

    /// The bounded async flip returns before the GPU worker reads the frame
    /// back, so a title can begin reusing its display buffer while the worker
    /// still owes a present. The present-from-guest-memory path must therefore
    /// read the COMPLETE frame captured on the flip thread (the snapshot), never
    /// whatever the title has since written — the partially-cleared/reused
    /// buffer that black-screened the earlier fire-and-forget flip.
    #[test]
    fn async_present_from_guest_memory_reads_the_flip_time_snapshot_not_a_reused_buffer() {
        const BASE: u64 = 0x4000;
        // Live guest memory now holds the title's NEXT frame (cleared) — the
        // buffer has been reused since the flip.
        let cleared = vec![0u8, 0, 0, 0, 0, 0, 0, 0];
        let session =
            AgcGpuSession::new(Arc::new(MutableScanoutMemory::new(BASE, cleared.clone())));
        let desc = raeen_core::subsystems::ScanoutDescriptor {
            width: 2,
            height: 1,
            pitch_pixels: 2,
            pixel_format: 0x8000_2200, // A8B8G8R8 -> RGBA in memory
            tiling_mode: 0,
        };
        // The flip thread captured the frame the title had just finished.
        let frame_a = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
        session
            .guest_scanout_snapshots
            .lock()
            .insert(BASE, Arc::new(frame_a.clone()));

        let img = session
            .present_from_guest_memory(BASE, &desc)
            .expect("a snapshot present yields a frame");
        assert_eq!(
            img.pixels, frame_a,
            "present reads the flip-time snapshot, never the reused (cleared) live buffer"
        );

        // With no snapshot it falls back to a live read (the pre-async path) —
        // proving the snapshot, not some unrelated state, was the source above.
        session.guest_scanout_snapshots.lock().clear();
        let live_img = session
            .present_from_guest_memory(BASE, &desc)
            .expect("live fallback yields a frame");
        assert_eq!(
            live_img.pixels, cleared,
            "with no snapshot the live (now cleared) buffer is read"
        );
    }

    /// The staged boot splash rides into the next process session, masks
    /// fallback-presented frames, and comes down on `hide_splash`
    /// (`sceSystemServiceHideSplashScreen`).
    #[test]
    fn boot_splash_presents_until_hidden() {
        let splash = RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![9, 9, 9, 255],
            bytes_per_pixel: 4,
        };
        AgcGpuSession::set_pending_splash(Some(splash.clone()));
        let session = GpuProcessSession::create(deny_memory());
        AgcGpuSession::set_pending_splash(None);

        assert_eq!(session.last_image().unwrap().pixels, splash.pixels);

        // A frame that reached `last_image` WITHOUT hitting the flip address
        // (the most-content fallback) must not replace the splash — it can be
        // a bare cleared render target.
        let fallback = RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![64, 128, 0, 255],
            bytes_per_pixel: 4,
        };
        *session.last_image.lock() = Some(Arc::new(fallback.clone()));
        assert_eq!(session.last_image().unwrap().pixels, splash.pixels);

        // The title declares itself ready -> present its frames.
        session.hide_splash();
        assert_eq!(session.last_image().unwrap().pixels, fallback.pixels);
        session.shutdown();
    }

    /// Staging a new title resets the session's presentation: the previous
    /// title's drawn frame is dropped and the new splash (or nothing) shows.
    /// This is the fix for a new launch opening on the *previous* title's
    /// splash when the new title fails and never installs a session of its own.
    #[test]
    fn reset_presentation_drops_the_old_frame_and_applies_the_new_splash() {
        let session = AgcGpuSession::new(deny_memory());
        // Title A rendered a frame (its splash already came down).
        session.hide_splash();
        *session.last_image.lock() = Some(Arc::new(RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![7, 7, 7, 255],
            bytes_per_pixel: 4,
        }));
        assert_eq!(session.last_image().unwrap().pixels, vec![7, 7, 7, 255]);

        // Title B staged with its own splash: A's frame is gone, B's splash shows.
        session.reset_presentation(Some(Arc::new(RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![2, 2, 2, 255],
            bytes_per_pixel: 4,
        })));
        assert_eq!(session.last_image().unwrap().pixels, vec![2, 2, 2, 255]);

        // Title B has no pic0: nothing to present — crucially NOT A's old frame.
        session.reset_presentation(None);
        assert!(session.last_image().is_none());
    }

    /// A flip to a buffer the title really drew into is its own "rendering is
    /// ready" signal and takes the splash down, exactly like SharpEmu.
    #[test]
    fn flip_to_drawn_buffer_takes_the_splash_down() {
        let session = AgcGpuSession::new(deny_memory());
        *session.splash.lock() = Some(Arc::new(RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![9, 9, 9, 255],
            bytes_per_pixel: 4,
        }));
        let content = RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![10, 20, 30, 255],
            bytes_per_pixel: 4,
        };
        session
            .framebuffers
            .lock()
            .insert(0x1000, Arc::new(content.clone()));

        // Flip to an undrawn buffer: splash stays up.
        session.present_scanout(0xDEAD_BEEF, None);
        assert_eq!(session.last_image().unwrap().pixels, vec![9, 9, 9, 255]);

        // Flip to the drawn buffer: splash comes down, title frame presents.
        session.present_scanout(0x1000, None);
        assert_eq!(session.last_image().unwrap().pixels, content.pixels);
    }

    /// Present-from-guest-memory (M3): a CPU-drawn 2D display buffer with no GPU
    /// render target is presented by reading its guest bytes as pixels, using
    /// the registered VideoOut attribute — and a GPU-drawn target at the same
    /// address still wins (never regressed).
    #[test]
    fn present_scanout_reads_cpu_drawn_pixels_from_guest_memory() {
        // A 2x1 RGBA8 (A8B8G8R8, memory order RGBA) linear buffer laid out in a
        // live host allocation the session can read as guest memory.
        let backing: Vec<u8> = vec![10, 20, 30, 255, 40, 50, 60, 255];
        let base = backing.as_ptr() as u64;
        let session = AgcGpuSession::new(Arc::new(HostRangeMemory {
            start: base,
            len: backing.len() as u64,
        }));
        let desc = raeen_core::subsystems::ScanoutDescriptor {
            width: 2,
            height: 1,
            pitch_pixels: 2,
            pixel_format: 0x8000_2200, // A8B8G8R8 -> RGBA in memory
            tiling_mode: 1,            // LINEAR
        };

        // No GPU-drawn target anywhere: the guest bytes are read as the frame.
        session.present_scanout(base, Some(desc));
        let img = session.last_image().expect("guest-memory frame presented");
        assert_eq!(img.width, 2);
        assert_eq!(img.pixels, backing, "raw guest RGBA bytes become the frame");

        // A GPU-drawn target at the SAME address still wins over guest memory.
        let drawn = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![1, 2, 3, 255, 4, 5, 6, 255],
            bytes_per_pixel: 4,
        };
        session
            .framebuffers
            .lock()
            .insert(base, Arc::new(drawn.clone()));
        session.present_scanout(base, Some(desc));
        assert_eq!(
            session.last_image().unwrap().pixels,
            drawn.pixels,
            "a GPU-drawn target at the flip address takes priority over guest memory"
        );

        // A `tiling_mode = 0` display buffer is read linearly (row-major with
        // pitch), matching SharpEmu's software presenter — real PS5 titles
        // (ASTRO.BOT) register their scanout as mode 0. The guest bytes become
        // the frame just like the mode-1 case above.
        session.framebuffers.lock().clear();
        let mode0 = raeen_core::subsystems::ScanoutDescriptor {
            tiling_mode: 0,
            ..desc
        };
        session.present_scanout(base, Some(mode0));
        assert_eq!(
            session.last_image().unwrap().pixels,
            backing,
            "a tiling_mode=0 scanout is read linearly and presented"
        );

        // A genuinely unsupported tiling mode (>1, an undetiled macro-tile) is
        // skipped, never faked — the last frame stays up.
        let drawn_again = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![7, 8, 9, 255, 11, 12, 13, 255],
            bytes_per_pixel: 4,
        };
        session
            .framebuffers
            .lock()
            .insert(base, Arc::new(drawn_again.clone()));
        session.present_scanout(base, Some(desc)); // drawn wins, becomes last frame
        session.framebuffers.lock().clear();
        let tiled = raeen_core::subsystems::ScanoutDescriptor {
            tiling_mode: 2,
            ..desc
        };
        session.present_scanout(base, Some(tiled));
        assert_eq!(
            session.last_image().unwrap().pixels,
            drawn_again.pixels,
            "an unsupported tiling mode must not replace the last frame"
        );
    }

    #[test]
    fn missing_final_composite_lands_in_the_real_scanout_buffer() {
        let mut backing = vec![0u8; 8];
        let base = backing.as_mut_ptr() as u64;
        let session = AgcGpuSession::new(Arc::new(HostRangeMemory {
            start: base,
            len: backing.len() as u64,
        }));
        let desc = raeen_core::subsystems::ScanoutDescriptor {
            width: 2,
            height: 1,
            pitch_pixels: 2,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        };
        let intermediate = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![10, 20, 30, 255, 40, 50, 60, 255],
            bytes_per_pixel: 4,
        };
        assert!(session.compose_presentable_to_scanout(base, &desc, &intermediate));
        assert_eq!(
            backing, intermediate.pixels,
            "the compatibility composite must update guest-visible VideoOut memory"
        );
        let scanned = session
            .present_from_guest_memory(base, &desc)
            .expect("composed scanout");
        assert_eq!(scanned.pixels, intermediate.pixels);
    }

    #[test]
    fn final_composite_scales_internal_resolution_to_video_out() {
        let mut backing = vec![0u8; 16];
        let base = backing.as_mut_ptr() as u64;
        let session = AgcGpuSession::new(Arc::new(HostRangeMemory {
            start: base,
            len: backing.len() as u64,
        }));
        let desc = raeen_core::subsystems::ScanoutDescriptor {
            width: 4,
            height: 1,
            pitch_pixels: 4,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        };
        let internal = RenderedImage {
            width: 2,
            height: 1,
            pixels: vec![10, 20, 30, 255, 40, 50, 60, 255],
            bytes_per_pixel: 4,
        };
        assert!(session.compose_presentable_to_scanout(base, &desc, &internal));
        assert_eq!(
            backing,
            vec![
                10, 20, 30, 255, 10, 20, 30, 255, 40, 50, 60, 255, 40, 50, 60, 255
            ]
        );
    }

    /// Bounded identity-mapped guest memory over a live allocation, so worker
    /// tests can model labels and DMA-visible buffers.
    struct HostRangeMemory {
        start: u64,
        len: u64,
    }

    impl crate::guest_mem::GpuGuestMemory for HostRangeMemory {
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
            // SAFETY: the validated range lies inside a leaked live allocation
            // owned by the test for the process lifetime.
            unsafe {
                std::ptr::copy_nonoverlapping(addr as *const u8, out.as_mut_ptr(), out.len());
            }
            true
        }

        fn write_gpu(&self, addr: u64, data: &[u8]) -> bool {
            if !self.validate_gpu_range(addr, data.len() as u64, true) {
                return false;
            }
            // SAFETY: same bounded leaked-allocation proof as `read_gpu`.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len());
            }
            true
        }
    }

    /// Guest memory backing a single display buffer with an owned byte vector
    /// (no host-pointer identity), so a present test can model the title having
    /// reused the buffer for its next frame while an async flip is still owed.
    struct MutableScanoutMemory {
        base: u64,
        bytes: Mutex<Vec<u8>>,
    }

    impl MutableScanoutMemory {
        fn new(base: u64, bytes: Vec<u8>) -> Self {
            Self {
                base,
                bytes: Mutex::new(bytes),
            }
        }
    }

    impl crate::guest_mem::GpuGuestMemory for MutableScanoutMemory {
        fn validate_gpu_range(&self, addr: u64, len: u64, _write: bool) -> bool {
            addr == self.base && (len as usize) <= self.bytes.lock().len()
        }

        fn read_gpu(&self, addr: u64, out: &mut [u8]) -> bool {
            if addr != self.base {
                return false;
            }
            let bytes = self.bytes.lock();
            if out.len() > bytes.len() {
                return false;
            }
            out.copy_from_slice(&bytes[..out.len()]);
            true
        }

        fn write_gpu(&self, addr: u64, data: &[u8]) -> bool {
            if addr != self.base {
                return false;
            }
            let mut bytes = self.bytes.lock();
            if data.len() > bytes.len() {
                return false;
            }
            bytes[..data.len()].copy_from_slice(data);
            true
        }
    }

    /// `sceAgcAcbWaitRegMem` 32-bit wait: 6 dwords, body
    /// `[addr_lo, addr_hi, mask, compare, reference]`.
    fn wait32_dcb(addr: u64, compare: u32, reference: u32) -> Vec<u32> {
        vec![
            pm4::header(6, pm4::IT_NOP, pm4::R_WAIT_MEM_32),
            addr as u32,
            (addr >> 32) as u32,
            0xFFFF_FFFF,
            compare,
            reference,
        ]
    }

    /// ACB-form `R_DMA_DATA` memory→memory copy: 7 dwords, body
    /// `[dst_lo, dst_hi, src_lo, src_hi, byte_count, sel]`.
    fn dma_copy_dcb(dst: u64, src: u64, bytes: u32) -> Vec<u32> {
        vec![
            pm4::header(7, pm4::IT_NOP, pm4::R_DMA_DATA),
            dst as u32,
            (dst >> 32) as u32,
            src as u32,
            (src >> 32) as u32,
            bytes,
            0,
        ]
    }

    /// The cross-queue producer/consumer shape measured on ASTRO.BOT (and the
    /// scene-pixel gate SharpEmu closed — AgcExports.cs:4508-4529): an ACB
    /// buffer waits on a label another queue's writeback writes.
    ///
    /// 1. The ACB buffer suspends at its unmet `R_WAIT_MEM_32` — the work
    ///    behind the wait must NOT run, and the label is never force-written.
    /// 2. A second ACB submission parks behind the suspended queue (in-order
    ///    ring semantics).
    /// 3. A graphics-DCB writeback (here a `R_DMA_DATA` guest-memory copy)
    ///    writes the label; the worker's post-submission re-check resumes the
    ///    ACB from its suspend point and then drains the parked backlog in
    ///    order.
    #[test]
    fn acb_wait_resumes_when_a_dcb_writeback_writes_the_label() {
        let arena: &'static mut [u32] = Box::leak(vec![0u32; 16].into_boxed_slice());
        let base = arena.as_ptr() as u64;
        // dword layout: 0 label, 1 producer value (1), 2 consumer src (0xAA),
        // 3 consumer dst, 4 parked src (0xBB), 5 parked dst.
        arena[1] = 1;
        arena[2] = 0xAA;
        arena[4] = 0xBB;
        let memory: Arc<dyn crate::guest_mem::GpuGuestMemory> = Arc::new(HostRangeMemory {
            start: base,
            len: std::mem::size_of_val(arena) as u64,
        });
        let session = AgcGpuSession::new_process(memory);

        // ACB: wait for label == 1, then copy 0xAA into dword 3.
        let mut acb = wait32_dcb(base, 3, 1);
        acb.extend(dma_copy_dcb(base + 12, base + 8, 4));
        session.submit_dcb_async(acb, true);
        session.wait_idle();
        assert_eq!(
            session.wait_suspend_stats(),
            GpuWaitStats {
                suspended: 1,
                resumed: 0,
                currently_suspended: 1,
                parked: 0,
            },
            "unmet ACB wait must suspend its buffer"
        );
        assert_eq!(arena[3], 0, "work behind the wait must not run");
        assert_eq!(arena[0], 0, "the label must never be force-satisfied");

        // A second ACB submission parks behind the suspended queue.
        session.submit_dcb_async(dma_copy_dcb(base + 20, base + 16, 4), true);
        session.wait_idle();
        assert_eq!(arena[5], 0, "parked work must not run ahead of the wait");
        assert_eq!(session.wait_suspend_stats().parked, 1);

        // Producer: a graphics-DCB DMA writeback writes the label.
        session.submit_dcb_async(dma_copy_dcb(base, base + 4, 4), false);
        session.wait_idle();
        assert_eq!(arena[0], 1, "the producer's writeback landed");
        assert_eq!(arena[3], 0xAA, "the resumed ACB ran its post-wait work");
        assert_eq!(
            arena[5], 0xBB,
            "the parked submission drained after the resume"
        );
        assert_eq!(
            session.wait_suspend_stats(),
            GpuWaitStats {
                suspended: 1,
                resumed: 1,
                currently_suspended: 0,
                parked: 0,
            }
        );
        session.shutdown();
    }

    /// ACB-form `R_RELEASE_MEM` completion-label write: 8 dwords, body
    /// `[action, control (data_sel at bits 23:16), addr_lo, addr_hi,
    /// data_lo, data_hi, ctx]`.
    fn release_mem_dcb(addr: u64, value: u32) -> Vec<u32> {
        vec![
            pm4::header(8, pm4::IT_NOP, pm4::R_RELEASE_MEM),
            0,
            1 << 16,
            addr as u32,
            (addr >> 32) as u32,
            value,
            0,
            0,
        ]
    }

    /// The REVERSE direction of the test above: a suspended GRAPHICS wait is
    /// resumed by a compute-queue `RELEASE_MEM` producer. GTA V drives both
    /// directions — graphics fences feeding ACB waits and ACB completion
    /// labels feeding graphics waits.
    #[test]
    fn dcb_wait_resumes_when_an_acb_release_mem_writes_the_label() {
        let arena: &'static mut [u32] = Box::leak(vec![0u32; 16].into_boxed_slice());
        let base = arena.as_ptr() as u64;
        // dword layout: 0 label, 2 consumer src (0xCC), 3 consumer dst.
        arena[2] = 0xCC;
        let memory: Arc<dyn crate::guest_mem::GpuGuestMemory> = Arc::new(HostRangeMemory {
            start: base,
            len: std::mem::size_of_val(arena) as u64,
        });
        let session = AgcGpuSession::new_process(memory);

        // Graphics DCB: wait for label == 1, then copy 0xCC into dword 3.
        let mut dcb = wait32_dcb(base, 3, 1);
        dcb.extend(dma_copy_dcb(base + 12, base + 8, 4));
        session.submit_dcb_async(dcb, false);
        session.wait_idle();
        assert_eq!(
            session.wait_suspend_stats().currently_suspended,
            1,
            "unmet graphics wait must suspend its buffer"
        );
        assert_eq!(arena[3], 0, "work behind the wait must not run");
        assert_eq!(arena[0], 0, "the label must never be force-satisfied");

        // Producer: an ACB RELEASE_MEM writes the completion label.
        session.submit_dcb_async(release_mem_dcb(base, 1), true);
        session.wait_idle();
        assert_eq!(arena[0], 1, "the compute RELEASE_MEM landed");
        assert_eq!(
            arena[3], 0xCC,
            "the resumed graphics buffer ran its post-wait work"
        );
        assert_eq!(
            session.wait_suspend_stats(),
            GpuWaitStats {
                suspended: 1,
                resumed: 1,
                currently_suspended: 0,
                parked: 0,
            }
        );
        session.shutdown();
    }

    /// The state-only fixture must really be state-only, or the test above
    /// silently starts standing up Vulkan again.
    #[test]
    fn state_only_fixture_has_no_draw_or_dispatch() {
        let decoded = agc::decode_submission(&build_state_only_dcb()).expect("valid DCB");
        assert_eq!(decoded.draw_packets, 0);
        assert_eq!(decoded.dispatch_packets, 0);
    }

    /// `wait_idle` on a session that never submitted anything must return, not
    /// block on a worker that was never started.
    #[test]
    fn wait_idle_is_a_no_op_when_nothing_was_submitted() {
        let session = AgcGpuSession::new(deny_memory());
        session.wait_idle();
    }

    #[test]
    #[allow(deprecated)]
    fn fixture_dcb_decodes_as_one_draw() {
        let words = build_m2_draw_dcb();
        let decoded = agc::decode_submission(&words).expect("valid fixture");
        assert_eq!(decoded.draw_packets, 1);
        assert_eq!(decoded.dispatch_packets, 0);
    }

    /// The CP fixture DCB must still be a legal AGC stream to the standalone
    /// decoder — the two decoders are independent and should agree.
    #[test]
    fn cp_draw_dcb_decodes_as_one_draw() {
        let words = build_cp_draw_dcb(96, 48, ScissorHalf::Left);
        let decoded = agc::decode_submission(&words).expect("valid Gen5 DCB");
        assert_eq!(decoded.draw_packets, 1);
    }

    /// Walk the CP DCB with the real command processor and assert the register
    /// state it leaves behind. No Vulkan needed — this pins the decode half of
    /// the acceptance test on every machine.
    #[test]
    fn cp_draw_dcb_programs_extent_scissor_and_shaders() {
        struct Probe {
            seen: Option<(u32, u32, i32, i32, bool, bool, u32)>,
        }
        impl kyty_graphics::run::DrawSink for Probe {
            fn draw_index_auto(
                &mut self,
                ctx: &kyty_graphics::hw_regs::Context,
                ucfg: &kyty_graphics::hw_regs::UserConfig,
                sh: &kyty_graphics::hw_regs::Shader,
                _index_count: u32,
                _flags: u32,
            ) -> Result<(), kyty_graphics::run::DrawError> {
                let rt = &ctx.render_targets[0];
                let vp = &ctx.screen_viewport;
                self.seen = Some((
                    rt.attrib2.width + 1,
                    rt.attrib2.height + 1,
                    vp.screen_scissor_left,
                    vp.screen_scissor_right,
                    sh.vs.vs_embedded,
                    sh.ps.ps_embedded,
                    ucfg.prim_type,
                ));
                Ok(())
            }
        }

        let mut probe = Probe { seen: None };
        let mut cp = CommandProcessor::new();
        cp.run(&build_cp_draw_dcb(96, 48, ScissorHalf::Left), &mut probe)
            .expect("the CP must walk its own fixture DCB");
        assert_eq!(probe.seen, Some((96, 48, 0, 48, true, true, 17)));

        let mut probe = Probe { seen: None };
        let mut cp = CommandProcessor::new();
        cp.run(&build_cp_draw_dcb(96, 48, ScissorHalf::Right), &mut probe)
            .expect("mirror DCB");
        let (_, _, left, right, ..) = probe.seen.expect("draw reached the sink");
        assert_eq!((left, right), (48, 96), "one register value flips the half");
    }

    /// An eliminate-fast-clear pass (CB_COLOR_CONTROL mode 2 with FAST_CLEAR
    /// armed) must become a real direct clear: the draw is consumed and the
    /// target's framebuffer entry holds the packed CLEAR_WORD colour.
    #[test]
    fn eliminate_fast_clear_pass_clears_the_target_framebuffer() {
        const BASE: u64 = 0x1_0000;
        let complete = build_cp_draw_dcb(96, 48, ScissorHalf::Left);
        let split = complete.len() - 7;
        let mut dcb: Vec<u32> = complete[..split].to_vec();
        let mut set_cx = |reg: u32, value: u32| {
            dcb.extend([
                pm4::header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
                reg,
                value,
            ]);
        };
        // Arm fast clear on slot 0 (format 0xa RGBA8, FAST_CLEAR bit 13) and
        // program the packed clear colour.
        set_cx(pm4::CB_COLOR0_INFO, (0xa << 2) | (1 << 13));
        set_cx(pm4::CB_COLOR0_CLEAR_WORD0, 0x8040_20FF);
        // CB_COLOR_CONTROL.MODE = 2 (EliminateFastClear), OP untouched.
        set_cx(pm4::CB_COLOR_CONTROL, 2 << 4);
        dcb.extend_from_slice(&complete[split..]);

        let session = AgcGpuSession::new(deny_memory());
        match session.execute_dcb_cp(&dcb, false) {
            Ok(_) => {}
            Err(AgcExecError::Gpu(_)) => return, // Vulkan-less CI host.
            Err(other) => panic!("FCE DCB must walk: {other:?}"),
        }
        let framebuffers = session.framebuffers.lock();
        let image = framebuffers
            .get(&BASE)
            .expect("the direct clear must land in the framebuffer map");
        assert_eq!((image.width, image.height), (96, 48));
        assert!(
            image
                .pixels
                .chunks_exact(4)
                .all(|px| px == 0x8040_20FFu32.to_le_bytes()),
            "every pixel must be the packed CLEAR_WORD colour"
        );
    }

    /// A fixture DCB that binds TWO colour targets must surface the second as
    /// a real extra attachment in the translated draw state — the full chain
    /// from `SET_CONTEXT_REG` decode through `draw_state_from_regs`. No
    /// Vulkan needed; the Vulkan half is pinned by `tests/mrt_targets.rs`.
    #[test]
    fn cp_dcb_with_two_bound_targets_translates_to_an_mrt_draw_state() {
        struct Probe {
            mrt: Option<Vec<(u8, u64, ash::vk::Format)>>,
        }
        impl kyty_graphics::run::DrawSink for Probe {
            fn draw_index_auto(
                &mut self,
                ctx: &kyty_graphics::hw_regs::Context,
                ucfg: &kyty_graphics::hw_regs::UserConfig,
                _sh: &kyty_graphics::hw_regs::Shader,
                index_count: u32,
                _flags: u32,
            ) -> Result<(), kyty_graphics::run::DrawError> {
                const SPIRV: &[u32] = &[0x0723_0203];
                let state = crate::draw_translate::draw_state_from_regs(
                    ctx,
                    ucfg,
                    index_count,
                    SPIRV,
                    SPIRV,
                )?;
                self.mrt = Some(
                    state
                        .mrt
                        .iter()
                        .map(|extra| (extra.slot, extra.target_base, extra.format))
                        .collect(),
                );
                Ok(())
            }
        }

        // Splice slot-1 register writes between the fixture's state section
        // and its trailing 7-dword draw packet, AFTER the fixture's
        // CB_TARGET_MASK write (which would otherwise zero slot 1's nibble).
        let complete = build_cp_draw_dcb(96, 48, ScissorHalf::Left);
        let split = complete.len() - 7;
        let mut dcb: Vec<u32> = complete[..split].to_vec();
        let mut set_cx = |reg: u32, value: u32| {
            dcb.extend([
                pm4::header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
                reg,
                value,
            ]);
        };
        set_cx(
            pm4::CB_COLOR0_BASE + pm4::CB_COLOR_SLOT_STRIDE,
            0x2_0000 >> 8,
        );
        set_cx(pm4::CB_COLOR0_INFO + pm4::CB_COLOR_SLOT_STRIDE, 0xa << 2);
        set_cx(pm4::CB_COLOR0_ATTRIB2 + 1, (95 << 14) | 47);
        set_cx(pm4::CB_TARGET_MASK, 0xF | (0xF << 4));
        dcb.extend_from_slice(&complete[split..]);

        let mut probe = Probe { mrt: None };
        let mut cp = CommandProcessor::new();
        cp.run(&dcb, &mut probe).expect("MRT fixture DCB must walk");
        assert_eq!(
            probe.mrt.expect("the draw reached the sink"),
            vec![(1u8, 0x2_0000u64, ash::vk::Format::R8G8B8A8_UNORM)],
            "slot 1 must translate into one extra attachment"
        );
    }

    /// Register state belongs to the GPU queue, not to one submitted DCB.
    /// Retail AGC emits state-only setup buffers followed by draw-only buffers;
    /// constructing a fresh command processor per submit loses every shader and
    /// render-target bind before the draw arrives.
    #[test]
    fn gpu_session_preserves_register_state_across_submissions() {
        let complete = build_cp_draw_dcb(96, 48, ScissorHalf::Left);
        let draw_dwords = 7;
        let split = complete.len() - draw_dwords;
        let state_only = &complete[..split];
        let draw_only = &complete[split..];
        assert_eq!(
            agc::decode_submission(state_only)
                .expect("state DCB")
                .draw_packets,
            0
        );
        assert_eq!(
            agc::decode_submission(draw_only)
                .expect("draw DCB")
                .draw_packets,
            1
        );

        let session = AgcGpuSession::new(deny_memory());
        match session.execute_dcb_cp(state_only, false) {
            Ok(None) => {}
            Err(AgcExecError::Gpu(_)) => return, // Vulkan-less CI host.
            other => panic!("state-only submit should not draw: {other:?}"),
        }
        // The state-only submit above never needs a device (it draws nothing),
        // so it returns `Ok(None)` even on a Vulkan-less host and the guard
        // there does not fire. This second submit is the one that actually
        // rasterizes, so it needs its OWN guard — without it the test fails
        // only on hosts with no Vulkan 1.3 driver, i.e. exactly CI.
        let image = match session.execute_dcb_cp(draw_only, false) {
            Ok(image) => image.expect("persistent state reaches a real draw"),
            Err(AgcExecError::Gpu(_)) => return, // Vulkan-less CI host.
            Err(other) => panic!("draw-only DCB must inherit the setup DCB: {other:?}"),
        };
        assert_eq!((image.width, image.height), (96, 48));
    }

    /// The M2 fixture DCB must not reach a sink through a real command
    /// processor. It fails twice over: it is a 2-dword invention rather than
    /// Kyty's 7-dword AGC draw packet (so it truncates), and it programs no
    /// register state (so it has no render target either way).
    ///
    /// This is what "retired from the title path" means concretely — the two
    /// paths cannot be mistaken for one another.
    #[test]
    #[allow(deprecated)]
    fn fixture_dcb_cannot_draw_through_the_command_processor() {
        struct Fail;
        impl kyty_graphics::run::DrawSink for Fail {
            fn draw_index_auto(
                &mut self,
                _ctx: &kyty_graphics::hw_regs::Context,
                _ucfg: &kyty_graphics::hw_regs::UserConfig,
                _sh: &kyty_graphics::hw_regs::Shader,
                _index_count: u32,
                _flags: u32,
            ) -> Result<(), kyty_graphics::run::DrawError> {
                panic!("the register-less fixture DCB must never reach a draw sink");
            }
        }
        let mut cp = CommandProcessor::new();
        let err = cp
            .run(&build_m2_draw_dcb(), &mut Fail)
            .expect_err("a register-less DCB must not draw");
        assert!(
            matches!(err, CpError::Truncated { .. }),
            "the fixture's 2-dword draw packet is not Kyty's 7-dword AGC form; got {err:?}"
        );
    }

    /// A well-formed draw packet with no preceding register writes must be
    /// REFUSED — never a silent success or a fixture triangle. Under Fix 1 the
    /// refusal no longer aborts the walk (aborting deadlocked Minecraft's
    /// async-compute submit worker): it is named once at the handler, counted in
    /// `refused_draws`, and skipped so the completion packets after it still run.
    /// A non-zero `refused_draws` is the observable proof the draw did not
    /// silently succeed and was not served by the M2 fixture path.
    #[test]
    fn draw_without_a_bound_render_target_is_refused_and_counted() {
        struct Sink;
        impl kyty_graphics::run::DrawSink for Sink {
            fn draw_index_auto(
                &mut self,
                ctx: &kyty_graphics::hw_regs::Context,
                ucfg: &kyty_graphics::hw_regs::UserConfig,
                _sh: &kyty_graphics::hw_regs::Shader,
                index_count: u32,
                _flags: u32,
            ) -> Result<(), kyty_graphics::run::DrawError> {
                const SPIRV: &[u32] = &[0x0723_0203];
                crate::draw_translate::draw_state_from_regs(ctx, ucfg, index_count, SPIRV, SPIRV)
                    .map(|_| ())
            }
        }
        let mut dcb = vec![pm4::header(7, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO), 3, 0];
        dcb.resize(dcb.len() + 4, 0);

        let mut cp = CommandProcessor::new();
        cp.run(&dcb, &mut Sink)
            .expect("Fix 1: a refused draw must be skipped, not abort the walk");
        assert_eq!(
            cp.refused_draws(),
            1,
            "the register-less draw must be REFUSED (not silently drawn or a fixture)"
        );
    }
}
