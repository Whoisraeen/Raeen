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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tracing::{debug, warn};
use xps5x_core::error::GpuError;

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

/// How many submitted DCBs may be in flight before a submitter blocks.
///
/// Backpressure is not optional: a title submits faster than this session
/// renders, so an unbounded queue would grow without limit (a DCB is up to 4 MiB).
/// Blocking the submitter when the queue is full is also what real hardware does
/// when its ring buffer fills.
const SUBMIT_QUEUE_DEPTH: usize = 8;

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
        done: Option<std::sync::mpsc::SyncSender<()>>,
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
    last_image: Mutex<Option<RenderedImage>>,
    /// System boot splash: the package's `sce_sys/pic0.png`, decoded at launch.
    /// While `Some`, [`AgcGpuSession::last_image`] presents it instead of any
    /// title frame — a real PS5 shows this image from launch until the title
    /// calls `sceSystemServiceHideSplashScreen`. It also comes down when the
    /// title flips to a buffer with real drawn content (SharpEmu's behavior),
    /// but NOT for the most-content present fallback: that path can surface a
    /// bare cleared render target, which is exactly what the splash exists to
    /// cover.
    splash: Mutex<Option<RenderedImage>>,
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
    framebuffers: Mutex<std::collections::HashMap<u64, RenderedImage>>,
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
    scanout_descriptor: Mutex<Option<xps5x_core::subsystems::ScanoutDescriptor>>,
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
}

/// Every Nth flip miss, the most-content fallback re-runs its full census
/// (full flush + scan) instead of trusting the remembered target, so content
/// migrating to a different render target is picked up within N flips.
const FALLBACK_REELECT_INTERVAL: u64 = 64;

/// Rate-limited warn for a flip whose display buffer uses a tiling mode or
/// pixel format the present-from-guest-memory path does not model. The frame is
/// skipped (never faked); the last presented frame stays up.
fn warn_unsupported_scanout(desc: &xps5x_core::subsystems::ScanoutDescriptor) {
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_power_of_two() {
        warn!(
            tiling_mode = desc.tiling_mode,
            pixel_format = format_args!("{:#x}", desc.pixel_format),
            width = desc.width,
            height = desc.height,
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
        let session = AgcGpuSession::new(memory);
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

impl xps5x_core::subsystems::GpuSubmissionSubsystem for GpuProcessSession {
    fn submit(&self, words: Vec<u32>, queue: xps5x_core::subsystems::GpuQueue) {
        self.submit_dcb_async(
            words,
            matches!(queue, xps5x_core::subsystems::GpuQueue::AsyncCompute),
        );
    }

    fn map_shader_metadata(
        &self,
        code_address: u64,
        data: xps5x_core::subsystems::ShaderMappedData,
    ) {
        AgcGpuSession::map_shader_metadata(&self.0, code_address, data);
    }

    fn present_scanout(
        &self,
        address: u64,
        descriptor: Option<xps5x_core::subsystems::ScanoutDescriptor>,
    ) {
        AgcGpuSession::present_scanout(&self.0, address, descriptor);
    }

    fn wait_idle(&self) {
        AgcGpuSession::wait_idle(self);
    }

    fn hide_splash(&self) {
        AgcGpuSession::hide_splash(self);
    }

    fn stats(&self) -> xps5x_core::subsystems::GpuSubmissionStats {
        xps5x_core::subsystems::GpuSubmissionStats {
            submitted: self.submission_count.load(Ordering::Relaxed),
            completed_draws: self.draw_count(),
            skipped_shaders: self.shader_skip_count(),
        }
    }
}

impl AgcGpuSession {
    fn new(memory: Arc<dyn crate::guest_mem::GpuGuestMemory>) -> Self {
        Self {
            lifecycle: Mutex::new(GpuLifecycle::Open),
            guest_memory: Mutex::new(Some(memory)),
            submit_queue: OnceLock::new(),
            in_flight: (Mutex::new(0), parking_lot::Condvar::new()),
            backend: Mutex::new(None),
            command_processor: Mutex::new(CommandProcessor::new()),
            compute_command_processor: Mutex::new(CommandProcessor::new()),
            last_image: Mutex::new(None),
            splash: Mutex::new(None),
            draw_count: Mutex::new(0),
            shader_cache: Mutex::new(crate::shader_fetch::ShaderTranslateCache::new()),
            shader_skip_count: Mutex::new(0),
            framebuffers: Mutex::new(std::collections::HashMap::new()),
            scanout_address: Mutex::new(None),
            scanout_descriptor: Mutex::new(None),
            last_compute_shader: Mutex::new(None),
            wait_states: Mutex::new(WaitStates::default()),
            wait_suspended_total: AtomicU64::new(0),
            wait_resumed_total: AtomicU64::new(0),
            submission_count: AtomicU64::new(0),
            fallback_present_base: Mutex::new(None),
            flip_miss_count: AtomicU64::new(0),
        }
    }

    /// Create isolated GPU state for a new guest process.
    #[must_use]
    pub fn new_process(memory: Arc<dyn crate::guest_mem::GpuGuestMemory>) -> GpuProcessSession {
        GpuProcessSession::create(memory)
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
    pub fn last_image(&self) -> Option<RenderedImage> {
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
    fn reset_presentation(&self, splash: Option<RenderedImage>) {
        *self.splash.lock() = splash;
        *self.last_image.lock() = None;
        self.framebuffers.lock().clear();
        *self.scanout_address.lock() = None;
        *self.draw_count.lock() = 0;
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
    ) {
        *gpu_runtime_config().write() = GpuRuntimeConfig {
            validation_layers,
            resolution_scale,
            gpu_device_index,
        };
    }

    /// The current process-wide GPU settings (see [`Self::set_runtime_config`]).
    pub(crate) fn runtime_config() -> GpuRuntimeConfig {
        *gpu_runtime_config().read()
    }

    /// Take the boot splash down (`sceSystemServiceHideSplashScreen`, or a
    /// flip to a buffer with real drawn content).
    pub fn hide_splash(&self) {
        *self.splash.lock() = None;
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
        descriptor: Option<xps5x_core::subsystems::ScanoutDescriptor>,
    ) {
        if address == 0 {
            return;
        }
        *self.scanout_address.lock() = Some(address);
        *self.scanout_descriptor.lock() = descriptor;
        // Synchronous by design (item 4 status): the flip waits for its flush
        // (~1-7.5 ms measured — one fence + at most one target readback). The
        // fire-and-forget variant (`wait: false`) was tried and REVERTED: on
        // Minecraft the unthrottled flip stream wedged the title ~10 s into
        // boot (main thread deadlocked on a title mutex held by the flipping
        // render pool thread, three threads spinning, flips stopped — a
        // one-line pthread_sync "stuck >3s" warn names it), while the
        // rendezvous form ran two clean 180 s runs at the same measured flip
        // rate (the ~16.7 ms vblank pacer, not this wait, owns the flip
        // period). Re-attempt only with the wedge understood.
        self.consume_flush(Some(address), true);
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
                        done: done_tx,
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
        self.flush_and_present(address);
    }

    /// The flush consumer body (stage C): land pending deferred readbacks —
    /// restricted to the flipped target when one is named (item 2) — then
    /// present the scanout buffer. Runs on the GPU worker thread via
    /// [`GpuWork::Flush`], or inline when no worker exists.
    fn flush_and_present(&self, address: Option<u64>) {
        {
            let guard = self.backend.lock();
            if let Some(device) = guard.as_ref().and_then(|b| b.device()) {
                let mut framebuffers = self.framebuffers.lock();
                let insert_all =
                    |fb: &mut std::collections::HashMap<u64, RenderedImage>,
                     flushed: Vec<(u64, RenderedImage)>| {
                        for (base, img) in flushed {
                            fb.insert(base, img);
                        }
                    };
                // The all-targets dump needs every target's CPU pixels, so it
                // forces a full readback; otherwise a flip reads back ONLY the
                // flipped target plus the remembered fallback target — every
                // other dirty target stays GPU-side.
                let remembered = *self.fallback_present_base.lock();
                let filter: Option<Vec<u64>> =
                    if std::env::var_os("XPS5X_DUMP_ALL_TARGETS").is_some() {
                        None
                    } else {
                        address.map(|a| {
                            let mut bases = vec![a];
                            if let Some(r) = remembered
                                && r != a
                            {
                                bases.push(r);
                            }
                            bases
                        })
                    };
                match crate::vulkan::offscreen::flush_deferred_draws_filtered(
                    device,
                    filter.as_deref(),
                ) {
                    Ok(flushed) => insert_all(&mut framebuffers, flushed),
                    Err(e) => {
                        warn!(error = %e, "deferred-draw flush failed — presenting the last flushed frame");
                    }
                }
                // Flip miss: the flipped address has no drawn content. The
                // most-content fallback presents the remembered target when it
                // still has fresh content; a FULL flush + census re-election
                // runs on the first miss, whenever the remembered target went
                // dark, and every FALLBACK_REELECT_INTERVAL misses (content
                // migrating to another render target is caught within that).
                if let Some(addr) = address
                    && !framebuffers.contains_key(&addr)
                {
                    let misses = self.flip_miss_count.fetch_add(1, Ordering::Relaxed);
                    let remembered_has_content = remembered.is_some_and(|r| {
                        framebuffers
                            .get(&r)
                            .is_some_and(|img| img.pixels.iter().step_by(64).any(|&b| b != 0))
                    });
                    if !remembered_has_content || misses.is_multiple_of(FALLBACK_REELECT_INTERVAL) {
                        match crate::vulkan::offscreen::flush_deferred_draws(device) {
                            Ok(flushed) => insert_all(&mut framebuffers, flushed),
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
        if let Some(address) = address {
            self.present_flipped(address);
        }
    }

    /// Present the buffer the title flipped to, after the flush landed its
    /// pixels: the frame at the flip address when it has drawn content, else
    /// the drawn target with the most content (the composite — the title
    /// fills its scanout buffer by a copy/DMA we do not yet capture, so the
    /// flip address is often an empty target). Never regresses `last_image`
    /// when nothing has content, and only a frame at the actual flip address
    /// takes the boot splash down.
    fn present_flipped(&self, address: u64) {
        let remembered = *self.fallback_present_base.lock();
        let (image, fallback, fallback_base, keys) = {
            let fb = self.framebuffers.lock();
            let image = fb.get(&address).cloned();
            let (fallback, fallback_base) = if image.is_none() {
                // Steady state: present the remembered fallback target while
                // it still has content — no census, no scan of other targets
                // (whose entries may be deliberately stale, kept GPU-side by
                // the filtered flush).
                let kept = remembered
                    .and_then(|r| fb.get(&r).map(|img| (r, img)))
                    .filter(|(_, img)| img.pixels.iter().step_by(64).any(|&b| b != 0))
                    .map(|(r, img)| (Some(r), img.clone()));
                match kept {
                    Some((base, img)) => (Some(img), base),
                    None => {
                        // Census election, sub-sampled (every 64th byte):
                        // exact counts cost a full scan of every 8 MB target
                        // per flip. Which target has the MOST content
                        // survives sub-sampling.
                        let elected = fb
                            .iter()
                            .map(|(base, img)| {
                                let nonzero =
                                    img.pixels.iter().step_by(64).filter(|&&b| b != 0).count();
                                (nonzero, *base, img)
                            })
                            .filter(|(nonzero, _, _)| *nonzero > 0)
                            .max_by_key(|(nonzero, _, _)| *nonzero)
                            .map(|(_, base, img)| (base, img.clone()));
                        match elected {
                            Some((base, img)) => (Some(img), Some(base)),
                            None => (None, None),
                        }
                    }
                }
            } else {
                (None, None)
            };
            let keys = if std::env::var_os("XPS5X_TRACE_FLIP").is_some() {
                Some(fb.keys().map(|k| format!("{k:#x}")).collect::<Vec<_>>())
            } else {
                None
            };
            (image, fallback, fallback_base, keys)
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
        let guest_present = if image.is_none() && fallback.is_none() {
            let desc = *self.scanout_descriptor.lock();
            desc.and_then(|desc| self.present_from_guest_memory(address, &desc))
        } else {
            None
        };
        let guest_hit = guest_present.is_some();
        // XPS5X_TRACE_FLIP: does the buffer the title flipped to have drawn
        // content, and what render targets exist? Answers whether black frames
        // are a routing miss (content is in another target) or a genuinely
        // empty scanout (the composite that fills it never ran).
        if let Some(keys) = keys {
            tracing::info!(
                scanout = format_args!("{address:#x}"),
                had_content = image.is_some(),
                render_targets = ?keys,
                "present_scanout: title flipped to this buffer"
            );
        }
        let flip_address_hit = image.is_some();
        let Some(presented) = image.or(fallback).or(guest_present) else {
            return;
        };
        *self.last_image.lock() = Some(presented.clone());
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
            tracing::info!(
                scanout_hit = flip_address_hit,
                present_index,
                scanout = format_args!("{address:#x}"),
                "present: dumping the scanned-out frame"
            );
        }
        maybe_dump_frame(&presented, present_index);
        if std::env::var_os("XPS5X_DUMP_ALL_TARGETS").is_some() {
            let targets: Vec<(u64, RenderedImage)> = self
                .framebuffers
                .lock()
                .iter()
                .map(|(base, img)| (*base, img.clone()))
                .collect();
            maybe_dump_all_targets(&targets, present_index);
        }
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
        desc: &xps5x_core::subsystems::ScanoutDescriptor,
    ) -> Option<RenderedImage> {
        if address == 0 || desc.width == 0 || desc.height == 0 {
            return None;
        }
        // SCE_VIDEO_OUT_TILING_MODE_LINEAR == 1.
        if desc.tiling_mode != 1 {
            warn_unsupported_scanout(desc);
            return None;
        }
        // Byte order in guest memory (little-endian) for the two common 32-bit
        // display formats. `A8B8G8R8` word => memory R,G,B,A (matches the
        // Shell's RGBA present, no swizzle). `A8R8G8B8` word => memory B,G,R,A
        // (swap R/B). _SRGB and _UNORM variants share the same channel layout.
        let swap_rb = match desc.pixel_format {
            0x8000_2000 | 0x8000_2200 => false, // A8B8G8R8 (RGBA in memory)
            0x8000_0000 | 0x8000_0200 => true,  // A8R8G8B8 (BGRA in memory)
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
        let mut pixels = Vec::<u8>::new();
        pixels
            .try_reserve_exact(width as usize * height as usize * 4)
            .ok()?;
        pixels.resize(width as usize * height as usize * 4, 0);
        for y in 0..height as usize {
            let src_row = y * row_bytes as usize;
            let dst_row = y * width as usize * 4;
            for x in 0..width as usize {
                let s = src_row + x * 4;
                let d = dst_row + x * 4;
                if swap_rb {
                    pixels[d] = src[s + 2];
                    pixels[d + 1] = src[s + 1];
                    pixels[d + 2] = src[s];
                    pixels[d + 3] = src[s + 3];
                } else {
                    pixels[d..d + 4].copy_from_slice(&src[s..s + 4]);
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
        // never getting there. Set XPS5X_VULKAN_VALIDATION=1 to restore it
        // when debugging a specific draw.
        // Validation is on when Settings ▸ Video ▸ Validation Layers is enabled,
        // or the env override is set (a per-run toggle for debugging one draw
        // without editing the config). Read at first backend creation, so the
        // config setting applies from the launch after it was changed.
        let validation = Self::runtime_config().validation_layers
            || std::env::var_os("XPS5X_VULKAN_VALIDATION").is_some();
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
                    framebuffers.insert(base, img);
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
            image = Some(img.clone());
        }
        // Snapshot every accumulated render target while the guard is still
        // held (re-locking `self.framebuffers` here would deadlock — the guard
        // lives to end of scope), for the optional all-targets dump below.
        let all_targets: Option<Vec<(u64, RenderedImage)>> = image.as_ref().and_then(|_| {
            std::env::var_os("XPS5X_DUMP_ALL_TARGETS").map(|_| {
                framebuffers
                    .iter()
                    .map(|(base, img)| (*base, img.clone()))
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
            let at_flip_address = addr.and_then(|a| framebuffers.get(&a).cloned());
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
                    .map(|img| (img.pixels.iter().filter(|&&b| b != 0).count(), img))
                    .filter(|(nonzero, _)| *nonzero > 0)
                    .max_by_key(|(nonzero, _)| *nonzero)
                    .map(|(_, img)| img.clone())
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
            let scanout_hit = scanout_image.is_some();
            let presented = scanout_image.unwrap_or_else(|| image.clone());
            *self.last_image.lock() = Some(presented.clone());
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
                .name("xps5x-gpu".to_owned())
                .spawn(move || {
                    while let Ok(work) = rx.recv() {
                        match work {
                            GpuWork::Submit(words, is_compute) => {
                                let _completion = InFlightCompletion(&session);
                                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    // Stage C: worker submissions defer
                                    // presentation — the flush (readback +
                                    // present) runs once per FLIP, not once
                                    // per submission. XPS5X_DUMP_FRAMES keeps
                                    // the old flush-per-submission cadence so
                                    // frame-dump diagnostics stay faithful.
                                    let deferred_present =
                                        std::env::var_os("XPS5X_DUMP_FRAMES").is_none();
                                    session.worker_submit(words, is_compute, deferred_present);
                                }))
                                .is_err()
                                {
                                    warn!("GPU submission panicked; dropping the DCB and keeping the worker alive");
                                }
                            }
                            GpuWork::Flush { address, done } => {
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
                });
            }
            Err(e) => warn!(error = %e, "AGC DCB draw skipped"),
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
                            let satisfied = crate::guest_mem::with_guest_memory(&memory, || {
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
                                None
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

        *self.last_image.lock() = Some(image.clone());
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
fn pending_splash() -> &'static Mutex<Option<RenderedImage>> {
    static PENDING: OnceLock<Mutex<Option<RenderedImage>>> = OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(None))
}

/// Process-wide GPU settings, mirrored from `EmulatorConfig.graphics` by the
/// Shell (see [`AgcGpuSession::set_runtime_config`]). Read where the Vulkan
/// backend is created (validation) and where each guest draw is sized
/// (resolution scale) — the two settings the user can drive from Settings ▸
/// Video that the GPU path can honour today.
#[derive(Clone, Copy)]
pub(crate) struct GpuRuntimeConfig {
    pub validation_layers: bool,
    pub resolution_scale: f32,
    /// Physical-device selection: 0 = auto (best-scored), n ≥ 1 selects the
    /// n-th usable device (1-based), falling back to auto when out of range.
    pub gpu_device_index: u32,
}

impl Default for GpuRuntimeConfig {
    fn default() -> Self {
        Self {
            validation_layers: false,
            resolution_scale: 1.0,
            gpu_device_index: 0,
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

/// Write draw output to disk when `XPS5X_DUMP_FRAMES` names a directory —
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
fn maybe_dump_all_targets(targets: &[(u64, RenderedImage)], draw_index: u64) {
    let Ok(dir) = std::env::var("XPS5X_DUMP_FRAMES") else {
        return;
    };
    if dir.is_empty() || (draw_index > 8 && !draw_index.is_power_of_two()) {
        return;
    }
    for (base, image) in targets {
        let bpp = image.bytes_per_pixel.max(1) as usize;
        let non_black = image
            .pixels
            .chunks_exact(bpp)
            .filter(|px| px.iter().take(3).any(|&b| b != 0))
            .count();
        let path =
            std::path::Path::new(&dir).join(format!("target_{base:012x}_{draw_index:06}.ppm"));
        let mut ppm = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
        ppm.reserve(image.pixels.len() / bpp * 3);
        // First 3 bytes of each pixel as approximate RGB — exact for the 4-byte
        // RGBA/BGRA formats; a rough low-byte view for packed/HDR targets (this
        // is a diagnostic dump, the presented frame is the RGBA8 composite).
        for px in image.pixels.chunks_exact(bpp) {
            ppm.extend_from_slice(&px[..3]);
        }
        let _ = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, &ppm));
        tracing::info!(
            base = format_args!("{base:#x}"),
            non_black_pixels = non_black,
            total = image.pixels.len() / bpp,
            "render-target census"
        );
    }
}

/// Number of frames presented since process start — the frame-dump sampling
/// index (see the call site for why the draw counter cannot serve this role).
static PRESENT_INDEX: AtomicU64 = AtomicU64::new(0);

fn maybe_dump_frame(image: &RenderedImage, draw_index: u64) {
    let Ok(dir) = std::env::var("XPS5X_DUMP_FRAMES") else {
        return;
    };
    if dir.is_empty() || (draw_index > 8 && !draw_index.is_power_of_two()) {
        return;
    }
    let path = std::path::Path::new(&dir).join(format!("frame_{draw_index:06}.ppm"));
    let mut ppm = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
    ppm.reserve(image.pixels.len() / 4 * 3);
    for rgba in image.pixels.chunks_exact(4) {
        ppm.extend_from_slice(&rgba[..3]);
    }
    match std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, &ppm)) {
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
    use xps5x_core::subsystems::GpuSubmissionSubsystem;

    fn deny_memory() -> Arc<dyn crate::guest_mem::GpuGuestMemory> {
        Arc::new(crate::guest_mem::DenyGpuMemory)
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
            xps5x_core::subsystems::GpuSubmissionStats::default()
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
        session.framebuffers.lock().insert(0x1000, content.clone());
        *session.last_image.lock() = Some(black);

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
        *session.last_image.lock() = Some(fallback.clone());
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
        *session.last_image.lock() = Some(RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![7, 7, 7, 255],
            bytes_per_pixel: 4,
        });
        assert_eq!(session.last_image().unwrap().pixels, vec![7, 7, 7, 255]);

        // Title B staged with its own splash: A's frame is gone, B's splash shows.
        session.reset_presentation(Some(RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![2, 2, 2, 255],
            bytes_per_pixel: 4,
        }));
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
        *session.splash.lock() = Some(RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![9, 9, 9, 255],
            bytes_per_pixel: 4,
        });
        let content = RenderedImage {
            width: 1,
            height: 1,
            pixels: vec![10, 20, 30, 255],
            bytes_per_pixel: 4,
        };
        session.framebuffers.lock().insert(0x1000, content.clone());

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
        let desc = xps5x_core::subsystems::ScanoutDescriptor {
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
        session.framebuffers.lock().insert(base, drawn.clone());
        session.present_scanout(base, Some(desc));
        assert_eq!(
            session.last_image().unwrap().pixels,
            drawn.pixels,
            "a GPU-drawn target at the flip address takes priority over guest memory"
        );

        // An unsupported (tiled) layout is skipped, never faked — the drawn
        // frame from the previous flip stays up.
        session.framebuffers.lock().clear();
        let tiled = xps5x_core::subsystems::ScanoutDescriptor {
            tiling_mode: 0,
            ..desc
        };
        session.present_scanout(base, Some(tiled));
        assert_eq!(
            session.last_image().unwrap().pixels,
            drawn.pixels,
            "an unsupported tiling mode must not replace the last frame"
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
        let image = session
            .execute_dcb_cp(draw_only, false)
            .expect("draw-only DCB must inherit the setup DCB")
            .expect("persistent state reaches a real draw");
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

    /// A well-formed draw packet with no preceding register writes must be a
    /// named fault, not a silent success or a fixture.
    #[test]
    fn draw_without_a_bound_render_target_is_a_named_error() {
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
        match cp.run(&dcb, &mut Sink) {
            Err(CpError::Draw { source, .. }) => {
                assert!(source.0.contains("render target"), "got {source}");
            }
            other => panic!("expected a named draw fault, got {other:?}"),
        }
    }
}
