//! The PM4 command processor.
//!
//! Faithful port of Kyty `emulator/src/Graphics/GraphicsRun.cpp`
//! (MIT (c) InoriRus) — specifically `CommandProcessor::Run` (L989) and its
//! `graphics_init_jmp_tables` dispatch (L4130).
//!
//! # Scope
//!
//! Gen5/AGC only: the PS5 uses AGC, not GNM, so Kyty's Gen4 block decoders
//! (pitch/slice/view) are not ported. This slice covers what a real DCB needs
//! to reach its draws — `SET_{CONTEXT,SH,UCONFIG}_REG` (direct and indirect),
//! the embedded-shader ops, index state, `DRAW_INDEX_AUTO`, `DRAW_INDEX`, and
//! degraded indirect draws.
//!
//! # This crate cannot draw
//!
//! `kyty-graphics` has no Vulkan dependency, so unlike Kyty (whose
//! `CommandProcessor` calls straight into `GraphicsRender`) the walk here
//! terminates at the [`DrawSink`] trait. `raeen-gpu` implements it.
//!
//! # Resilience policy (deliberate deviation from Kyty)
//!
//! Kyty `EXIT()`s on any packet it does not recognize. A retail title's
//! command buffer always contains ops this processor does not know, so that
//! policy means zero draws forever. Instead:
//!
//! - An **unknown opcode, custom op, or register** logs a rate-limited warn
//!   (once per distinct op per [`CommandProcessor`] instance) naming the op,
//!   is skipped by its encoded dword length, and the walk continues.
//! - **Hard errors are reserved for structurally corrupt streams**: a packet
//!   whose declared length runs past the buffer ([`CpError::Truncated`]) or a
//!   header that is not type-2/type-3 ([`CpError::NotType3`], a desynced
//!   walk).
//! - A draw the sink cannot honour ([`CpError::Draw`]) is treated like an
//!   unknown op: named once (never-silent), counted in
//!   [`CommandProcessor::refused_draws`], skipped by its header-encoded length,
//!   and the walk CONTINUES. A refusal must never abandon the packets that
//!   follow it — the completion labels/fences and later dispatches the guest
//!   polls on for "GPU done" live there, and aborting the walk deadlocked a
//!   title's async-compute submit worker (see [`CommandProcessor::run_resumable`]).
//!   The sink call sites still surface [`CpError::Draw`] so a non-walk caller
//!   (a direct `dispatch`/test) sees the named fault.
//!
//! # Deviations from Kyty (deliberate; see the ledger)
//!
//! 1. **Type check.** Kyty's `Run` loop never checks the packet type and would
//!    misparse a type-0/2 header as type-3. We reject type-0/1 and treat type-2
//!    as the 1-dword filler it is.
//! 2. **No unsigned wrap.** Kyty does `dw -= s + 1` on an unsigned counter, so
//!    an over-long packet wraps and is only caught on the *next* iteration —
//!    after the overrun read. We bounds-check first.
//! 3. **Register batches.** Kyty's handlers assert on exact `cmd_id` constants
//!    (`EXIT_NOT_IMPLEMENTED(cmd_id != 0xC0016900)`), baking packet *length*
//!    into the test; a two-register batch over the same offset would abort.
//!    Kyty survives only because its own emitters produce exactly those
//!    constants. We decode the count and loop.
//! 4. **One NOP table.** Kyty keeps two parallel tables (`g_hw_sh_custom_func`
//!    and `g_cp_op_custom_func`) and `cp_op_nop` aborts if both or neither is
//!    set. One `match` makes that invariant unrepresentable.
//! 5. **Sparse dispatch.** Kyty uses flat function-pointer arrays sized
//!    `UC_NUM` (16384) to hold a single entry. We `match`, which compiles to
//!    the same jump table without the 64 KiB of nulls.
//! 6. **Register fallback.** Kyty reaches the PS5 per-register setters (incl.
//!    `CB_COLOR0_ATTRIB2`, the render-target extent) *only* through
//!    `R_CX_REGS_INDIRECT`, which derefs an out-of-band guest pointer. Those
//!    setters take `(offset, value)` and need no memory of their own, so
//!    [`CommandProcessor::set_context_register`] exposes them to plain
//!    `IT_SET_CONTEXT_REG` writes as well.
//! 7. **Guest memory behind a trait.** Kyty's indirect handlers
//!    (`cp_op_indirect_cx_regs` L3018 …) reinterpret guest dwords as host
//!    pointers and dereference them. Here every out-of-band read goes through
//!    [`GuestMemory`]; when no reader is supplied the packet is skipped with a
//!    rate-limited warn instead of a wild deref.
//! 8. **No trailing-NOP swallowing.** Kyty's raw draw parsers
//!    (`cp_op_draw_index` L2757, `cp_op_draw_index_auto` L2807) peek past
//!    their own packet and over-report their length to swallow the marker
//!    NOPs its emitters append. Our walker parses those NOPs as the `R_ZERO`
//!    NOPs they are, so every handler returns its header-declared length.
//! 9. **Indexed/indirect draws degrade, honestly.** Kyty fetches real index
//!    buffers in `GraphicsRender`; that layer is not ported. The default
//!    [`DrawSink::draw_index`] degrades to a vertex-count-only
//!    [`DrawSink::draw_index_auto`], and indirect draws read only the first
//!    args record ([`DrawIndirectArgs`]) to recover a count. Both paths log
//!    the degradation (rate-limited).

use crate::hw_regs::{
    ColorAttrib2, ColorAttrib3, ColorInfo, Context, CsStageRegisters, DepthShaderControl, Shader,
    UserConfig, UserSgprType,
};
use crate::pm4;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, warn};

fn trace_shader_binds_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RAEEN_TRACE_SHADER_BINDS").is_some())
}

fn trace_indirect_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RAEEN_TRACE_INDIRECT").is_some())
}

/// A process-monotonic, always-nonzero value for `RELEASE_MEM` DATA_SEL 3/4
/// (GPU timestamp). SharpEmu writes `Stopwatch.GetTimestamp()` here
/// (AgcExports.cs:5395-5397 / 5469-5471); the guest only needs a nonzero,
/// non-decreasing completion value, so a plain counter is deterministic and
/// sufficient. Starts at 1 so a "became nonzero" poll is satisfied by the first
/// release.
fn next_release_timestamp() -> u64 {
    static RELEASE_CLOCK: AtomicU64 = AtomicU64::new(0);
    RELEASE_CLOCK.fetch_add(1, Ordering::Relaxed) + 1
}

/// Which register file an unknown offset belonged to.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegFile {
    Context,
    Shader,
    UserConfig,
}

impl std::fmt::Display for RegFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Context => write!(f, "context"),
            Self::Shader => write!(f, "shader"),
            Self::UserConfig => write!(f, "user-config"),
        }
    }
}

/// Process-global rate limit for skipped-register warnings (FIX 1, log noise).
///
/// [`CommandProcessor::first`] already dedups per instance, but the graphics
/// DCB and the async-compute ACB each own a persistent `CommandProcessor`
/// (`raeen-gpu` `AgcExec`), so a register unknown to both is warned about once
/// PER QUEUE — a Minecraft run leaks ~140 lines for ~70 distinct registers.
/// This set collapses that to at most one WARN per distinct `(file, register)`
/// for the entire process, keeping the message at WARN so the
/// register-coverage gap stays visible while it stops spamming.
static WARNED_SKIP_REGS: std::sync::Mutex<BTreeSet<(RegFile, u32)>> =
    std::sync::Mutex::new(BTreeSet::new());

/// True the first time this `(file, reg)` is seen process-wide; the caller
/// emits its WARN exactly then. Recovers from a poisoned lock (a panic while
/// deduping must not re-arm the spam).
fn warn_skip_reg_once(file: RegFile, reg: u32) -> bool {
    WARNED_SKIP_REGS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert((file, reg))
}

/// A draw that could not be translated. Never silent: the message names the
/// register or resource that was missing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawError(pub String);

impl std::fmt::Display for DrawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DrawError {}

/// A command-stream fault. Every variant names the DWORD offset so a fault can
/// be pointed at a packet in a capture.
///
/// Per the resilience policy only **structural** faults are errors: unknown
/// ops and registers are logged and skipped, never returned. Typed
/// replacement for Kyty's hard `EXIT(...)`; the crate convention is a
/// hand-written `Display` rather than a `thiserror` dependency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpError {
    Truncated {
        offset: u32,
        need: u32,
        remaining: u32,
    },
    NotType3 {
        offset: u32,
        cmd_id: u32,
    },
    Draw {
        offset: u32,
        source: DrawError,
    },
}

impl std::fmt::Display for CpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated {
                offset,
                need,
                remaining,
            } => write!(
                f,
                "truncated PM4 packet at DWORD {offset}: needs {need}, has {remaining}"
            ),
            Self::NotType3 { offset, cmd_id } => {
                write!(f, "non-type-3 PM4 header at DWORD {offset}: {cmd_id:#010x}")
            }
            Self::Draw { offset, source } => {
                write!(f, "draw failed at DWORD {offset}: {source}")
            }
        }
    }
}

impl std::error::Error for CpError {}

/// Out-of-band guest-memory access for the packets that carry pointers
/// (`R_*_REGS_INDIRECT`, indirect draw args).
///
/// Kyty dereferences these pointers directly (its CP runs in the guest's
/// address space). This crate has no such assumption; the embedder supplies a
/// reader, and without one the pointer-carrying packets are skipped with a
/// rate-limited warn.
pub trait GuestMemory {
    /// Read `count` dwords at guest virtual address `addr`, or `None` if the
    /// range is not readable.
    fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>>;

    /// Read `len` bytes at guest virtual address `addr`, for DMA payload
    /// copies. Default `None`: a read-only embedder skips DMA packets with a
    /// warn instead of failing the stream.
    fn read_bytes(&self, _addr: u64, _len: u64) -> Option<Vec<u8>> {
        None
    }

    /// Write `bytes` at guest virtual address `addr`, for DMA payload copies.
    /// Default `false` (not writable) for read-only embedders.
    fn write_bytes(&self, _addr: u64, _bytes: &[u8]) -> bool {
        false
    }
}

/// The parsed condition of a wait-on-memory packet (`IT_WAIT_REG_MEM`,
/// `R_WAIT_MEM_32`, `R_WAIT_MEM_64`): suspend the queue until
/// `(label & mask) <compare> (reference & mask)` holds.
///
/// Port of SharpEmu's `GpuWaitRegistry.WaitingDcb` condition fields
/// (GpuWaitRegistry.cs:19-40) and its masked comparison
/// (`GpuWaitRegistry.Compare`, GpuWaitRegistry.cs:239-256).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WaitSpec {
    /// Guest address of the watched completion label.
    pub address: u64,
    pub mask: u64,
    pub reference: u64,
    /// 3-bit hardware compare function: 0 = always, 1 = `<`, 2 = `<=`,
    /// 3 = `==`, 4 = `!=`, 5 = `>=`, 6 = `>`, 7 = reserved (fail-open).
    pub compare: u32,
    /// 64-bit label (`R_WAIT_MEM_64`) vs 32-bit.
    pub is_64: bool,
}

impl WaitSpec {
    /// Masked comparison — SharpEmu `GpuWaitRegistry.Compare`
    /// (GpuWaitRegistry.cs:239-256). Functions 0 and 7 never block: 0 is the
    /// hardware "always" condition and reserved 7 is fail-open so a malformed
    /// packet cannot suspend a queue forever.
    #[must_use]
    pub fn satisfied_by(&self, value: u64) -> bool {
        let masked = value & self.mask;
        let reference = self.reference & self.mask;
        match self.compare {
            1 => masked < reference,
            2 => masked <= reference,
            3 => masked == reference,
            4 => masked != reference,
            5 => masked >= reference,
            6 => masked > reference,
            _ => true,
        }
    }

    /// Read the watched label. `None` when the address is not readable guest
    /// memory — the caller decides whether that means "keep waiting"
    /// (re-check path) or "do not stall" (parse path).
    pub fn read_label(&self, mem: &dyn GuestMemory) -> Option<u64> {
        let dwords = mem.read_dwords(self.address, if self.is_64 { 2 } else { 1 })?;
        Some(if self.is_64 {
            u64::from(dwords[0]) | (u64::from(dwords[1]) << 32)
        } else {
            u64::from(dwords[0])
        })
    }
}

/// A walk that stopped at an unmet wait: where to resume and what it waits on.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SuspendedWait {
    /// Dword index just past the wait packet — pass to
    /// [`CommandProcessor::run_resumable`] once the label satisfies the spec.
    pub resume_dword: usize,
    pub wait: WaitSpec,
}

/// Outcome of a resumable CP walk ([`CommandProcessor::run_resumable`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    Completed,
    /// The stream reached a `WAIT_REG_MEM`-family packet whose condition the
    /// current label value does not satisfy. Nothing past the packet ran.
    ///
    /// SharpEmu proves this is THE scene-pixel gate: suspending here and
    /// resuming when the label is genuinely written (AgcExports.cs:4508-4529,
    /// `HandleSubmittedWaitRegMem` registering into `GpuWaitRegistry`) is what
    /// lets cross-queue composites run in dependency order. NEVER
    /// force-satisfy the label — that publishes incomplete state.
    Suspended(SuspendedWait),
}

/// Which wait-packet encoding is being parsed (they differ in body layout).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum WaitForm {
    /// `IT_WAIT_REG_MEM`, body `[control, addr_lo, addr_hi, ref32, mask32,
    /// poll]` (SharpEmu `TryParseSubmittedWait` standard arm,
    /// AgcExports.cs:4550-4566).
    Standard,
    /// `IT_NOP`+`R_WAIT_MEM_32`, body `[addr_lo, addr_hi, mask32, control,
    /// ref32]` (SharpEmu AgcExports.cs:4568-4590; our own
    /// `sceAgcAcbWaitRegMem` emits the same layout).
    Mem32,
    /// `IT_NOP`+`R_WAIT_MEM_64`, body `[addr_lo, addr_hi, mask_lo, mask_hi,
    /// ref_lo, ref_hi, control, poll]`.
    Mem64,
}

/// Parameters of an indexed draw, as decoded from `R_DRAW_INDEX` /
/// `IT_DRAW_INDEX_2` (Kyty `cp_op_draw_index`, GraphicsRun.cpp L2757).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IndexedDraw {
    /// Latched `IT_INDEX_TYPE` state (Kyty `m_index_type_and_size`).
    pub index_type_and_size: u32,
    pub index_count: u32,
    /// Guest address of the index buffer (0 when unknown, e.g. a degraded
    /// indirect draw with no `INDEX_BASE` programmed).
    pub index_addr: u64,
    /// The AGC draw-modifier flags dword; 0 for the raw `IT_DRAW_INDEX_2` form.
    pub flags: u32,
    /// Kyty's `type` argument to `CommandProcessor::DrawIndex`: the AGC form's
    /// body[4]; the raw form passes 1; degraded indirect draws pass 0.
    pub index_type: u32,
}

/// First record of an indirect-draw argument buffer, AMD's
/// `VkDrawIndirectCommand`-compatible layout: `{count, instance_count, ...}`.
/// Only `count` is honoured by the degraded path.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DrawIndirectArgs {
    pub count: u32,
    pub instance_count: u32,
}

/// Where [`CommandProcessor`] sends a translated draw.
///
/// Mirrors Kyty's `GraphicsRenderDrawIndexAuto` / `GraphicsRenderDrawIndex`
/// signatures. The whole register state is passed by reference; the
/// implementor decides what it needs.
pub trait DrawSink {
    /// A PM4 packet that can write guest memory was consumed.
    ///
    /// The command processor does not know which embedder-side decoded
    /// resources alias that range, so sinks with submission-local guest-memory
    /// caches must conservatively invalidate them. The default keeps simple
    /// recording/test sinks source-compatible.
    fn guest_memory_write_boundary(&mut self) {}

    /// Kyty: `GraphicsRenderDrawIndexAuto` (GraphicsRender.cpp).
    fn draw_index_auto(
        &mut self,
        ctx: &Context,
        ucfg: &UserConfig,
        sh: &Shader,
        index_count: u32,
        flags: u32,
    ) -> Result<(), DrawError>;

    /// Kyty: `GraphicsRenderDrawIndex` (GraphicsRender.cpp).
    ///
    /// **Default degradation (documented, deliberate):** the index buffer is
    /// *not* fetched; the draw is forwarded to
    /// [`DrawSink::draw_index_auto`] with `index_count` as the vertex count.
    /// For a triangle-list of sequential indices this is exact; for anything
    /// else it draws the wrong vertices but the right amount of work — enough
    /// for first light and honest logging. A sink that can fetch guest index
    /// buffers should override this.
    fn draw_index(
        &mut self,
        ctx: &Context,
        ucfg: &UserConfig,
        sh: &Shader,
        draw: &IndexedDraw,
    ) -> Result<(), DrawError> {
        self.draw_index_auto(ctx, ucfg, sh, draw.index_count, draw.flags)
    }

    /// Kyty: `GraphicsRenderDispatchDirect` (GraphicsRender.cpp L4938).
    fn dispatch_direct(
        &mut self,
        _ctx: &Context,
        _ucfg: &UserConfig,
        _sh: &Shader,
        _groups: [u32; 3],
        _mode: u32,
    ) -> Result<(), DrawError> {
        Err(DrawError(
            "compute dispatch reached a sink without compute support".to_owned(),
        ))
    }
}

/// Rate-limit key: which distinct condition has already been warned about.
/// One warn per key per [`CommandProcessor`] instance.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SkipKey {
    /// Unknown IT opcode (header bits 15:8).
    Op(u8),
    /// Unknown AGC custom op (`IT_NOP` header bits 7:2).
    Custom(u8),
    /// Unknown or out-of-range register in a file.
    Reg(RegFile, u32),
    /// A named degradation or skipped feature.
    Note(&'static str),
}

/// Kyty: `class CommandProcessor` (GraphicsRun.cpp L~100).
#[derive(Clone, Debug, Default)]
pub struct CommandProcessor {
    ctx: Context,
    ucfg: UserConfig,
    sh_ctx: Shader,
    index_type_and_size: u32,
    num_instances: u32,
    /// `IT_INDEX_BASE`: guest address of the bound index buffer.
    index_base: u64,
    /// `IT_INDEX_BUFFER_SIZE`: dword count of the bound index buffer.
    index_buffer_size: u32,
    /// `IT_SET_BASE` select 1: base address for indirect-draw argument
    /// buffers.
    indirect_draw_base: u64,
    /// `IT_SET_BASE` select 1 with the shader-type header bit set: base
    /// address for indirect-dispatch argument buffers. KytyPS5 keeps the two
    /// bases separate (`CpOpSetBase`, pm4Handlers.cpp L2546: PM4 header bit 1
    /// encodes `Gnmp::ShaderType` — 0 = draw args, 1 = dispatch args).
    indirect_dispatch_base: u64,
    /// Latched by the `R_ZERO` 'hu' marker; types subsequent user-SGPR writes.
    user_data_marker: UserSgprType,
    /// Which distinct unknown ops/registers have already been warned about.
    /// Survives [`CommandProcessor::reset`] so a per-frame `R_DRAW_RESET`
    /// cannot turn the rate limit back into log spam.
    warned: BTreeSet<SkipKey>,
    /// Number of shader-bind trace sites visited. This survives queue resets
    /// and bounds the opt-in diagnostic on frame loops.
    shader_bind_trace_count: u64,
    /// Draws/dispatches the sink REFUSED (returned [`DrawError`]) that the walk
    /// skipped over instead of aborting on. Cumulative across queue resets — a
    /// per-frame reset must not zero the honest count of refused work. See the
    /// skip-and-continue arm in [`CommandProcessor::run_resumable`].
    refused_draws: u64,
    /// Set by the wait-packet handlers when the watched label does not satisfy
    /// the condition; consumed by the walker immediately after the packet, so
    /// [`CommandProcessor::run_resumable`] can suspend the stream there.
    pending_wait: Option<WaitSpec>,
    /// Completion labels this walk's producer packets (`WRITE_DATA`,
    /// `RELEASE_MEM`) actually wrote to guest memory, as `(address, value)`.
    /// Drained by the embedder ([`Self::take_produced_labels`]) after a walk so
    /// it can latch cross-queue `WAIT_REG_MEM` waiters against the value *at
    /// write time*.
    ///
    /// Faithful adaptation of SharpEmu `GpuWaitRegistry.RecordProduced`
    /// (GpuWaitRegistry.cs:385-400): the guest frequently resets a completion
    /// label back to 0 immediately after signalling it (to arm the next frame),
    /// so re-reading live guest memory at wake time can miss the transient
    /// satisfied window. Recording the produced value lets the embedder resume
    /// the waiter even after the label was reset. Bounded by
    /// [`Self::MAX_PRODUCED_LABELS`].
    produced_labels: Vec<(u64, u64)>,
}

impl CommandProcessor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_instances: 1,
            ..Self::default()
        }
    }

    #[must_use]
    pub const fn get_ctx(&self) -> &Context {
        &self.ctx
    }
    #[must_use]
    pub const fn get_ucfg(&self) -> &UserConfig {
        &self.ucfg
    }
    #[must_use]
    pub const fn get_sh_ctx(&self) -> &Shader {
        &self.sh_ctx
    }
    #[must_use]
    pub const fn num_instances(&self) -> u32 {
        self.num_instances
    }
    #[must_use]
    pub const fn index_type_and_size(&self) -> u32 {
        self.index_type_and_size
    }
    #[must_use]
    pub const fn index_base(&self) -> u64 {
        self.index_base
    }
    #[must_use]
    pub const fn index_buffer_size(&self) -> u32 {
        self.index_buffer_size
    }
    #[must_use]
    pub const fn indirect_draw_base(&self) -> u64 {
        self.indirect_draw_base
    }

    /// How many distinct unknown/skipped conditions this instance has warned
    /// about. Diagnostics: a growing number across frames means the DCB uses
    /// ops this processor does not yet honour.
    #[must_use]
    pub fn distinct_skips(&self) -> usize {
        self.warned.len()
    }

    /// How many draws/dispatches this processor refused (sink returned
    /// [`DrawError`]) and SKIPPED — continuing the walk rather than aborting it,
    /// so the completion packets after the refusal still executed. A growing
    /// count is the honest measure of work the title asked for that we could not
    /// render; it never means the stream desynced (that is [`CpError`]).
    #[must_use]
    pub const fn refused_draws(&self) -> u64 {
        self.refused_draws
    }

    /// Kyty: `CommandProcessor::Reset` (L519) — clears register and index
    /// state. The warn rate-limit set deliberately survives (deviation; a
    /// reset must not re-arm log spam).
    pub fn reset(&mut self) {
        let warned = std::mem::take(&mut self.warned);
        let shader_bind_trace_count = self.shader_bind_trace_count;
        let refused_draws = self.refused_draws;
        // Producer labels written before an in-stream queue reset must survive
        // it: the embedder still needs to latch waiters against them, and a
        // per-frame `R_DRAW_RESET` between the producer packet and the drain
        // must not silently drop the wakeup.
        let produced_labels = std::mem::take(&mut self.produced_labels);
        *self = Self::new();
        self.warned = warned;
        self.shader_bind_trace_count = shader_bind_trace_count;
        self.refused_draws = refused_draws;
        self.produced_labels = produced_labels;
    }

    /// Cap on completion labels retained between drains, so a pathological
    /// increment `WRITE_DATA` cannot grow the vector without bound (real
    /// completion labels are one or two dwords).
    const MAX_PRODUCED_LABELS: usize = 64;

    /// Drain the completion labels the walk(s) since the last drain produced
    /// (see [`Self::produced_labels`]). The embedder latches these against
    /// suspended cross-queue waiters *before* re-reading live guest memory, so a
    /// same-submission label reset cannot lose the wakeup.
    #[must_use]
    pub fn take_produced_labels(&mut self) -> Vec<(u64, u64)> {
        std::mem::take(&mut self.produced_labels)
    }

    /// Record a `(address, value)` a producer packet just wrote to guest memory.
    /// Ignored past [`Self::MAX_PRODUCED_LABELS`] entries (the write still
    /// happened; only the latch record is capped).
    fn record_produced(&mut self, address: u64, value: u64) {
        if self.produced_labels.len() < Self::MAX_PRODUCED_LABELS {
            self.produced_labels.push((address, value));
        }
    }

    /// True the first time `key` is seen; the caller warns exactly then.
    fn first(&mut self, key: SkipKey) -> bool {
        self.warned.insert(key)
    }

    fn trace_shader_bind(&mut self) -> bool {
        if !trace_shader_binds_enabled() {
            return false;
        }
        self.shader_bind_trace_count = self.shader_bind_trace_count.saturating_add(1);
        self.shader_bind_trace_count <= 128 || self.shader_bind_trace_count.is_power_of_two()
    }

    /// Kyty: `CommandProcessor::Run` (L989) — walk a DCB and execute it.
    ///
    /// Equivalent to [`Self::run_with_memory`] with no [`GuestMemory`]:
    /// pointer-carrying packets (indirect registers, indirect draw args) are
    /// skipped with a rate-limited warn.
    pub fn run(&mut self, data: &[u32], sink: &mut dyn DrawSink) -> Result<(), CpError> {
        self.run_with_memory(data, sink, None)
    }

    /// Walk a DCB with out-of-band guest-memory access.
    ///
    /// Each handler returns the **body** dwords it consumed; the walker adds
    /// one for the header.
    ///
    /// # Errors
    ///
    /// Only structural faults ([`CpError::Truncated`], [`CpError::NotType3`]).
    /// A refused draw ([`CpError::Draw`] from the sink) is NOT returned here —
    /// it is counted ([`Self::refused_draws`]), logged once, and skipped so the
    /// walk continues to the completion packets after it (module resilience
    /// policy). Unknown packets are likewise skipped by their encoded length.
    pub fn run_with_memory(
        &mut self,
        data: &[u32],
        sink: &mut dyn DrawSink,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<(), CpError> {
        let mut pos = 0usize;
        loop {
            match self.run_resumable(data, pos, sink, mem)? {
                RunOutcome::Completed => return Ok(()),
                RunOutcome::Suspended(suspended) => {
                    // This entry point has no way to park the buffer and
                    // re-run it later, so an unmet wait degrades to the
                    // pre-suspend behaviour: continue past it, loudly. The
                    // GPU worker uses `run_resumable` and genuinely suspends.
                    if self.first(SkipKey::Note("wait_unmet_inline_continue")) {
                        warn!(
                            label = format_args!("{:#x}", suspended.wait.address),
                            compare = suspended.wait.compare,
                            "unmet WAIT_REG_MEM on the non-resumable walk — \
                             continuing past it (no suspend support at this call site)"
                        );
                    }
                    pos = suspended.resume_dword;
                }
            }
        }
    }

    /// Walk a DCB from `start_dword`, stopping at the first wait-on-memory
    /// packet whose condition the current label value does not satisfy.
    ///
    /// Port of SharpEmu's suspend design (`HandleSubmittedWaitRegMem`,
    /// AgcExports.cs:4508-4529 / 4595-4726): an unmet wait suspends the
    /// command buffer mid-stream — [`RunOutcome::Suspended`] carries the
    /// resume dword and the [`WaitSpec`] — and the embedder re-runs from the
    /// resume point once the label memory is genuinely written by a later
    /// submission's writebacks. The label is never force-satisfied.
    ///
    /// # Errors
    ///
    /// Same structural faults as [`Self::run_with_memory`].
    pub fn run_resumable(
        &mut self,
        data: &[u32],
        start_dword: usize,
        sink: &mut dyn DrawSink,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<RunOutcome, CpError> {
        let mut pos = start_dword.min(data.len());
        while pos < data.len() {
            let cmd_id = data[pos];
            let offset = pos as u32;

            // Kyty's loop assumes type 3 unconditionally. A type-2 filler is a
            // bare 1-dword NOP and is legal in a real stream.
            if pm4::is_type2(cmd_id) {
                pos += 1;
                continue;
            }
            if !pm4::is_type3(cmd_id) {
                return Err(CpError::NotType3 { offset, cmd_id });
            }

            let remaining = (data.len() - pos) as u32;
            if remaining < 2 {
                return Err(CpError::Truncated {
                    offset,
                    need: 2,
                    remaining,
                });
            }

            let body = &data[pos + 1..];
            let consumed = match self.dispatch(cmd_id, body, offset, sink, mem) {
                Ok(consumed) => consumed,
                // A REFUSED draw/dispatch is not a stream fault. The packet is
                // well-formed — its length lives in the header, so the walk can
                // step over it — and every packet AFTER it must still run:
                // RELEASE_MEM / WRITE_DATA / EVENT_WRITE_EOP completion labels,
                // DMA_DATA copies, and the later dispatches whose storage
                // writebacks a cross-queue `WAIT_REG_MEM` (and the guest's own
                // submit worker) poll on for "GPU done". Aborting the walk here
                // abandoned all of them, which is exactly the Minecraft deadlock:
                // an async-compute dispatch bound an unsupported storage-buffer V#
                // (add-tid / swizzle / OOB), the sink refused it, this walk
                // aborted, the completion the guest's ACB submit worker (thread
                // 21) waited on never came, and ~0.7 s later the main thread
                // wedged on that thread's held mutex — 0 fps, dead pad, no more
                // assets. So SKIP the one refused packet by its header-encoded
                // length and keep walking; the draw degrades to a visual glitch,
                // never a hang. STRUCTURAL faults (`Truncated` / `NotType3`)
                // still abort — the stream is desynced and the next packet
                // boundary is unknowable.
                Err(CpError::Draw { offset, source }) => {
                    self.refused_draws = self.refused_draws.saturating_add(1);
                    if self.first(SkipKey::Note("draw_refused_skip_and_continue")) {
                        warn!(
                            offset,
                            reason = %source,
                            "refused draw/dispatch skipped — continuing the walk so the \
                             completion packets after it still run (never-silent; later \
                             refusals on this processor are counted via refused_draws, \
                             not re-logged)"
                        );
                    }
                    pm4::body_dw(cmd_id)
                }
                Err(e) => return Err(e),
            };

            // Kyty wraps here on an over-long packet and only notices next
            // iteration; bail before the overrun instead.
            let advance = consumed as usize + 1;
            if advance > data.len() - pos {
                return Err(CpError::Truncated {
                    offset,
                    need: advance as u32,
                    remaining,
                });
            }
            pos += advance;
            if let Some(wait) = self.pending_wait.take() {
                return Ok(RunOutcome::Suspended(SuspendedWait {
                    resume_dword: pos,
                    wait,
                }));
            }
        }
        Ok(RunOutcome::Completed)
    }

    /// Kyty: the `g_cp_op_func[256]` table.
    fn dispatch(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        sink: &mut dyn DrawSink,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let op = pm4::op(cmd_id);
        let guest_memory_write_boundary = matches!(
            op,
            pm4::IT_WRITE_DATA | pm4::IT_RELEASE_MEM | pm4::IT_DMA_DATA
        ) || (op == pm4::IT_NOP
            && matches!(
                pm4::r_code(cmd_id),
                pm4::R_WRITE_DATA | pm4::R_RELEASE_MEM | pm4::R_DMA_DATA
            ));
        let result = match op {
            pm4::IT_NOP => self.cp_op_nop(cmd_id, body, offset, sink, mem),
            pm4::IT_SET_CONTEXT_REG => self.cp_op_set_context_reg(cmd_id, body, offset),
            pm4::IT_SET_SH_REG => self.cp_op_set_shader_reg(cmd_id, body, offset),
            pm4::IT_SET_UCONFIG_REG => self.cp_op_set_uconfig_reg(cmd_id, body, offset),
            pm4::IT_SET_UCONFIG_REG_INDEX => self.cp_op_set_uconfig_reg_index(cmd_id, body, offset),
            pm4::IT_DRAW_INDEX_AUTO => self.cp_op_draw_index_auto(cmd_id, body, offset, sink),
            pm4::IT_DISPATCH_DIRECT => self.cp_op_dispatch_direct(cmd_id, body, offset, sink),
            pm4::IT_DISPATCH_INDIRECT => {
                self.cp_op_dispatch_indirect(cmd_id, body, offset, sink, mem)
            }
            // Kyty: cp_op_draw_index (L2757), raw IT form 0xc0042700.
            pm4::IT_DRAW_INDEX_2 => self.cp_op_draw_index(cmd_id, body, offset, sink),
            // Not in Kyty's table — and it is what Minecraft draws with. See
            // cp_op_draw_index_offset_2.
            pm4::IT_DRAW_INDEX_OFFSET_2 => {
                self.cp_op_draw_index_offset_2(cmd_id, body, offset, sink)
            }
            pm4::IT_NUM_INSTANCES => {
                // Kyty: SetNumInstances (L1036) — 0 means 1.
                let n = Self::body_at(body, 0, offset)?;
                self.num_instances = if n == 0 { 1 } else { n };
                Ok(pm4::body_dw(cmd_id))
            }
            // Kyty: cp_op_index_type (L2986) → SetIndexType. This is what
            // Gen5's `GraphicsDcbSetIndexSize` emits (Graphics.cpp L1949).
            pm4::IT_INDEX_TYPE => {
                self.index_type_and_size = Self::body_at(body, 0, offset)?;
                Ok(pm4::body_dw(cmd_id))
            }
            // Index-buffer state for indexed draws (standard PM4; Kyty tracks
            // the address only inside its draw packets).
            pm4::IT_INDEX_BASE => {
                let lo = Self::body_at(body, 0, offset)?;
                let hi = Self::body_at(body, 1, offset)?;
                self.index_base = u64::from(lo) | (u64::from(hi) << 32);
                Ok(pm4::body_dw(cmd_id))
            }
            pm4::IT_INDEX_BUFFER_SIZE => {
                self.index_buffer_size = Self::body_at(body, 0, offset)?;
                Ok(pm4::body_dw(cmd_id))
            }
            // SET_BASE select 1 = indirect argument buffer base. The PM4
            // header's bit 1 carries `Gnmp::ShaderType` (libSceAgc
            // setBaseIndirectArgs folds it in): 0 routes to the indirect-DRAW
            // base, 1 to the indirect-DISPATCH base — KytyPS5 `CpOpSetBase`
            // (pm4Handlers.cpp L2546-2567).
            pm4::IT_SET_BASE => {
                let select = Self::body_at(body, 0, offset)? & 0xf;
                let lo = Self::body_at(body, 1, offset)?;
                let hi = Self::body_at(body, 2, offset)?;
                let shader_type = (cmd_id >> 1) & 0x3;
                let base = u64::from(lo & !7) | (u64::from(hi & 0xffff) << 32);
                if select == 1 && shader_type == 0 {
                    self.indirect_draw_base = base;
                } else if select == 1 && shader_type == 1 {
                    self.indirect_dispatch_base = base;
                } else if self.first(SkipKey::Note("set_base_select")) {
                    warn!(
                        select,
                        shader_type, offset, "IT_SET_BASE with unsupported base select — ignored"
                    );
                }
                Ok(pm4::body_dw(cmd_id))
            }
            pm4::IT_DRAW_INDIRECT | pm4::IT_DRAW_INDIRECT_MULTI => {
                self.cp_op_draw_indirect(cmd_id, body, offset, sink, mem, false)
            }
            pm4::IT_DRAW_INDEX_INDIRECT | pm4::IT_DRAW_INDEX_INDIRECT_MULTI => {
                self.cp_op_draw_indirect(cmd_id, body, offset, sink, mem, true)
            }
            pm4::IT_COND_EXEC => self.cp_op_cond_exec(cmd_id, body, offset, mem),
            // The wait family is honoured: parse, evaluate against the label,
            // and suspend the walk when unmet (see cp_op_wait_mem).
            pm4::IT_WAIT_REG_MEM => {
                self.cp_op_wait_mem(cmd_id, body, offset, WaitForm::Standard, mem)
            }
            // The label PRODUCERS: a cross-queue `WAIT_REG_MEM` blocks until one
            // of these writes its completion label to guest memory. Consuming
            // them "without effect" (the old behaviour) is exactly why
            // Minecraft's cross-queue waits were never satisfied — the DCB
            // parked forever on a label no producer wrote, and only the
            // dead-wait force-resume broke it (a glitch render). Execute them.
            pm4::IT_WRITE_DATA => self.cp_op_write_data(cmd_id, body, offset, true, mem),
            pm4::IT_RELEASE_MEM => self.cp_op_release_mem(cmd_id, body, offset, true, mem),
            // The standard DMA op: a real guest-memory copy/fill a later
            // packet in the same stream observes, so it must run in PM4 order
            // here — not only in the HLE's eager submit-time decode.
            pm4::IT_DMA_DATA => self.cp_op_it_dma_data(cmd_id, body, offset, mem),
            // These carry no guest-memory completion label on the RDNA2/AGC
            // draw path (EVENT_WRITE triggers kernel event queues, handled by
            // the HLE submit layer; ACQUIRE_MEM/CLEAR_STATE/etc. are cache/state
            // ops a draw never observes). Consumed by encoded length.
            pm4::IT_ACQUIRE_MEM
            | pm4::IT_EVENT_WRITE
            | pm4::IT_EVENT_WRITE_EOP
            | pm4::IT_EVENT_WRITE_EOS
            | pm4::IT_CONTEXT_CONTROL
            | pm4::IT_CLEAR_STATE
            | pm4::IT_PFP_SYNC_ME => {
                if self.first(SkipKey::Op(op.0)) {
                    warn!(
                        cmd_id = format_args!("{cmd_id:#010x}"),
                        op = op.0,
                        offset,
                        "PM4 sync/data packet consumed without effect"
                    );
                }
                Ok(pm4::body_dw(cmd_id))
            }
            _ => {
                // Resilience policy: skip by encoded length, warn once.
                if self.first(SkipKey::Op(op.0)) {
                    warn!(
                        cmd_id = format_args!("{cmd_id:#010x}"),
                        op = format_args!("{:#04x}", op.0),
                        offset,
                        "unknown PM4 opcode — packet skipped by its encoded length"
                    );
                }
                Ok(pm4::body_dw(cmd_id))
            }
        };
        if guest_memory_write_boundary && result.is_ok() {
            // This is deliberately conservative: malformed/unwritable packets
            // may clear a cache unnecessarily, but a real write must never
            // leave stale decoded guest resources bound later in the stream.
            sink.guest_memory_write_boundary();
        }
        result
    }

    /// Conditional execution packet: a zero 32-bit label skips the following
    /// packet range, while a non-zero label falls through. The returned count
    /// includes the guarded dwords only on the skip path, so the outer stream
    /// walker resumes at the first unguarded command.
    fn cp_op_cond_exec(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let packet_body = pm4::body_dw(cmd_id);
        if packet_body < 4 {
            return Err(CpError::Truncated {
                offset,
                need: 5,
                remaining: packet_body + 1,
            });
        }
        let address = u64::from(Self::body_at(body, 0, offset)? & 0xffff_fffc)
            | (u64::from(Self::body_at(body, 1, offset)?) << 32);
        let skip_dwords = Self::body_at(body, 3, offset)? & 0x3fff;

        let Some(mem) = mem else {
            if self.first(SkipKey::Note("cond_exec_without_guest_memory")) {
                warn!(
                    address = format_args!("{address:#x}"),
                    offset,
                    "COND_EXEC label cannot be read without guest memory — guarded commands executed"
                );
            }
            return Ok(packet_body);
        };
        let Some(value) = mem
            .read_dwords(address, 1)
            .and_then(|words| words.first().copied())
        else {
            if self.first(SkipKey::Note("cond_exec_unreadable_label")) {
                warn!(
                    address = format_args!("{address:#x}"),
                    offset, "COND_EXEC label is unreadable — guarded commands executed"
                );
            }
            return Ok(packet_body);
        };

        Ok(if value == 0 {
            packet_body.saturating_add(skip_dwords)
        } else {
            packet_body
        })
    }

    /// Kyty: `cp_op_nop` (L3156) — the AGC dialect's whole custom-op space
    /// rides here, discriminated by header bits 7:2.
    fn cp_op_nop(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        sink: &mut dyn DrawSink,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let r = pm4::r_code(cmd_id);

        if r == pm4::R_ZERO {
            // Kyty: the 'hu' marker latches how later user-SGPR writes are typed.
            if body
                .first()
                .is_some_and(|w| (w & 0xffff_0000) == 0x6875_0000)
            {
                return self.cp_op_marker(cmd_id, body);
            }
            return Ok(pm4::body_dw(cmd_id));
        }

        match r {
            pm4::R_CS => {
                let rsrc1 = Self::body_at(body, 3, offset)?;
                let rsrc2 = Self::body_at(body, 4, offset)?;
                let regs = CsStageRegisters {
                    data_addr: (u64::from(Self::body_at(body, 1, offset)?) << 8)
                        | (u64::from(Self::body_at(body, 2, offset)?) << 40),
                    // Gen5 delivers the checksum via later COMPUTE_SHADER_CHKSUM
                    // register writes (push_chksum), not in the R_CS packet.
                    chksum: 0,
                    vgprs: pm4::field(rsrc1, pm4::compute_pgm_rsrc1::VGPRS) as u8,
                    sgprs: pm4::field(rsrc1, pm4::compute_pgm_rsrc1::SGPRS) as u8,
                    bulky: pm4::field(rsrc1, pm4::compute_pgm_rsrc1::BULKY) as u8,
                    scratch_en: pm4::field(rsrc2, pm4::compute_pgm_rsrc2::SCRATCH_EN) as u8,
                    user_sgpr: pm4::field(rsrc2, pm4::compute_pgm_rsrc2::USER_SGPR) as u8,
                    tgid_x_en: pm4::field(rsrc2, pm4::compute_pgm_rsrc2::TGID_X_EN) as u8,
                    tgid_y_en: pm4::field(rsrc2, pm4::compute_pgm_rsrc2::TGID_Y_EN) as u8,
                    tgid_z_en: pm4::field(rsrc2, pm4::compute_pgm_rsrc2::TGID_Z_EN) as u8,
                    tg_size_en: pm4::field(rsrc2, pm4::compute_pgm_rsrc2::TG_SIZE_EN) as u8,
                    tidig_comp_cnt: pm4::field(rsrc2, pm4::compute_pgm_rsrc2::TIDIG_COMP_CNT) as u8,
                    lds_size: pm4::field(rsrc2, pm4::compute_pgm_rsrc2::LDS_SIZE) as u8,
                    num_thread_x: Self::body_at(body, 5, offset)?,
                    num_thread_y: Self::body_at(body, 6, offset)?,
                    num_thread_z: Self::body_at(body, 7, offset)?,
                };
                self.sh_ctx.set_cs_shader(regs);
                Ok(pm4::body_dw(cmd_id))
            }
            pm4::R_DISPATCH_DIRECT => self.cp_op_dispatch_direct(cmd_id, body, offset, sink),
            pm4::R_VS_EMBEDDED => {
                // Kyty: hw_sh_set_vs_embedded (L2367). cmd_id 0xc01b1034.
                let shader_modifier = Self::body_at(body, 0, offset)?;
                let id = Self::body_at(body, 1, offset)?;
                if self.trace_shader_bind() {
                    warn!(
                        offset,
                        id, shader_modifier, "shader-bind trace: embedded VS"
                    );
                }
                self.sh_ctx.set_vs_embedded(id, shader_modifier);
                Ok(pm4::body_dw(cmd_id))
            }
            pm4::R_PS_EMBEDDED => {
                // Kyty: hw_sh_set_ps_embedded (L2264). cmd_id 0xc0261038.
                let id = Self::body_at(body, 0, offset)?;
                if self.trace_shader_bind() {
                    warn!(offset, id, "shader-bind trace: embedded PS");
                }
                self.sh_ctx.set_ps_embedded(id);
                Ok(pm4::body_dw(cmd_id))
            }
            pm4::R_DRAW_INDEX_AUTO => self.cp_op_draw_index_auto(cmd_id, body, offset, sink),
            // Kyty: cp_op_draw_index (L2757), AGC form 0xC008100C.
            pm4::R_DRAW_INDEX => self.cp_op_draw_index(cmd_id, body, offset, sink),
            // Kyty: cp_op_indirect_{cx,sh,uc}_regs (L3018/L3050/L3082).
            pm4::R_CX_REGS_INDIRECT => {
                self.cp_op_indirect_regs(cmd_id, body, offset, RegFile::Context, mem)
            }
            pm4::R_SH_REGS_INDIRECT => {
                self.cp_op_indirect_regs(cmd_id, body, offset, RegFile::Shader, mem)
            }
            pm4::R_UC_REGS_INDIRECT => {
                self.cp_op_indirect_regs(cmd_id, body, offset, RegFile::UserConfig, mem)
            }
            // Kyty: cp_op_draw_reset (L2853) → CommandProcessor::Reset. Gen5's
            // `GraphicsDcbResetQueue` emits this (Graphics.cpp L1806).
            pm4::R_DRAW_RESET => {
                if self.trace_shader_bind() {
                    warn!(
                        offset,
                        vs_addr = format_args!("{:#x}", self.sh_ctx.vs.vs_regs.data_addr),
                        es_addr = format_args!("{:#x}", self.sh_ctx.vs.es_regs.data_addr),
                        ps_addr = format_args!("{:#x}", self.sh_ctx.ps.ps_regs.data_addr),
                        "shader-bind trace: draw reset"
                    );
                }
                self.reset();
                Ok(pm4::body_dw(cmd_id))
            }
            pm4::R_PUSH_MARKER | pm4::R_POP_MARKER => Ok(pm4::body_dw(cmd_id)),
            pm4::R_DMA_DATA => self.cp_op_dma_data(cmd_id, body, offset, mem),
            pm4::R_DISPATCH_RESET => {
                self.reset();
                Ok(pm4::body_dw(cmd_id))
            }
            // Label waits: honoured — parse, evaluate, suspend when unmet.
            pm4::R_WAIT_MEM_32 => self.cp_op_wait_mem(cmd_id, body, offset, WaitForm::Mem32, mem),
            pm4::R_WAIT_MEM_64 => self.cp_op_wait_mem(cmd_id, body, offset, WaitForm::Mem64, mem),
            // The AGC (NOP-wrapped) label PRODUCERS — the async-compute queue's
            // completion signals a cross-queue `WAIT_REG_MEM` polls on. Execute
            // them so those waits are genuinely satisfied (see the IT_* forms in
            // `dispatch`).
            pm4::R_WRITE_DATA => self.cp_op_write_data(cmd_id, body, offset, false, mem),
            pm4::R_RELEASE_MEM => self.cp_op_release_mem(cmd_id, body, offset, false, mem),
            // Sync / flip ops: consumed, not honoured. A draw never observes
            // them, and their side effects (flip queues) are already applied by
            // the HLE submit decode.
            pm4::R_ACQUIRE_MEM | pm4::R_WAIT_FLIP_DONE | pm4::R_FLIP => {
                if self.first(SkipKey::Custom(r.0)) {
                    warn!(
                        cmd_id = format_args!("{cmd_id:#010x}"),
                        r = r.0,
                        offset,
                        "AGC sync/flip packet consumed without effect"
                    );
                }
                Ok(pm4::body_dw(cmd_id))
            }
            _ => {
                // Resilience policy: skip by encoded length, warn once.
                if self.first(SkipKey::Custom(r.0)) {
                    warn!(
                        cmd_id = format_args!("{cmd_id:#010x}"),
                        r = format_args!("{:#04x}", r.0),
                        offset,
                        "unknown AGC custom op — packet skipped by its encoded length"
                    );
                }
                Ok(pm4::body_dw(cmd_id))
            }
        }
    }

    /// Execute an AGC `R_DMA_DATA` payload copy — the packet the title uses to
    /// fill buffers by DMA, including (measured suspicion, task #11) the
    /// VideoOut scanout buffer its composite lands in. Until this existed the
    /// packet was skipped by length, so those copies silently never happened.
    ///
    /// Two builder layouts share the r-code, discriminated by packet length
    /// (mirroring our own `sceAgcDcbDmaData` / `sceAgcAcbDmaData` emissions,
    /// which are dword-exact ports of SharpEmu's — i.e. of what retail libSceAgc
    /// emits):
    /// - 8-dw DCB form: body `[control0, control_ext, byte_count, dst_lo,
    ///   dst_hi, src_lo, src_hi]`, `control0` = dstSel | dstCache<<8 |
    ///   srcSel<<16 | srcCache<<24.
    /// - 7-dw ACB form: body `[dst_lo, dst_hi, src_lo, src_hi, byte_count,
    ///   sel]`, `sel` = srcSel | dstSel<<8.
    ///
    /// Only memory→memory (both selectors 0) is honoured; GDS/immediate
    /// selectors, absent/read-only [`GuestMemory`], or unreadable/unwritable
    /// ranges skip with one rate-limited warn each — never a stream error.
    fn cp_op_dma_data(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        /// Builder-enforced ceiling (`sceAgc*DmaData` reject larger): 256 MiB.
        const MAX_DMA_BYTES: u64 = 256 * 1024 * 1024;
        let body_len = pm4::body_dw(cmd_id);
        let (dst, src, byte_count, src_sel, dst_sel) = match body_len {
            7 => {
                let control0 = Self::body_at(body, 0, offset)?;
                let byte_count = Self::body_at(body, 2, offset)?;
                let dst = u64::from(Self::body_at(body, 3, offset)?)
                    | (u64::from(Self::body_at(body, 4, offset)?) << 32);
                let src = u64::from(Self::body_at(body, 5, offset)?)
                    | (u64::from(Self::body_at(body, 6, offset)?) << 32);
                (
                    dst,
                    src,
                    byte_count,
                    (control0 >> 16) & 0xff,
                    control0 & 0xff,
                )
            }
            6 => {
                let dst = u64::from(Self::body_at(body, 0, offset)?)
                    | (u64::from(Self::body_at(body, 1, offset)?) << 32);
                let src = u64::from(Self::body_at(body, 2, offset)?)
                    | (u64::from(Self::body_at(body, 3, offset)?) << 32);
                let byte_count = Self::body_at(body, 4, offset)?;
                let sel = Self::body_at(body, 5, offset)?;
                (dst, src, byte_count, sel & 0xff, (sel >> 8) & 0xff)
            }
            other => {
                if self.first(SkipKey::Note("DMA_DATA unknown length")) {
                    warn!(
                        cmd_id = format_args!("{cmd_id:#010x}"),
                        body_dw = other,
                        offset,
                        "R_DMA_DATA with unrecognized packet length — skipped"
                    );
                }
                return Ok(body_len);
            }
        };
        if src_sel != 0 || dst_sel != 0 {
            if self.first(SkipKey::Note("DMA_DATA non-memory selector")) {
                warn!(
                    src_sel,
                    dst_sel, offset, "DMA_DATA with non-memory selector — skipped"
                );
            }
            return Ok(body_len);
        }
        if byte_count == 0 || u64::from(byte_count) > MAX_DMA_BYTES {
            if self.first(SkipKey::Note("DMA_DATA byte count out of range")) {
                warn!(
                    byte_count,
                    offset, "DMA_DATA byte count out of range — skipped"
                );
            }
            return Ok(body_len);
        }
        let Some(mem) = mem else {
            if self.first(SkipKey::Note("DMA_DATA needs GuestMemory")) {
                warn!(offset, "DMA_DATA needs a GuestMemory accessor — skipped");
            }
            return Ok(body_len);
        };
        match mem.read_bytes(src, u64::from(byte_count)) {
            Some(bytes) if mem.write_bytes(dst, &bytes) => {
                debug!(
                    src = format_args!("{src:#x}"),
                    dst = format_args!("{dst:#x}"),
                    byte_count,
                    "DMA_DATA copy executed"
                );
            }
            _ => {
                if self.first(SkipKey::Note("DMA_DATA range unreadable/unwritable")) {
                    warn!(
                        src = format_args!("{src:#x}"),
                        dst = format_args!("{dst:#x}"),
                        byte_count,
                        offset,
                        "DMA_DATA source/destination not accessible guest memory — skipped"
                    );
                }
            }
        }
        Ok(body_len)
    }

    /// Execute a standard `IT_DMA_DATA` (0x50) packet — the same guest-memory
    /// copy/fill the AGC `R_DMA_DATA` custom op performs, in the standard PM4
    /// encoding. Until this existed the packet was consumed without effect on
    /// this walk and only the HLE's eager submit-time decode applied it, so a
    /// packet earlier in the stream never observed the DMA'd data in PM4
    /// order.
    ///
    /// Layout (KytyPS5 `CpOpDmaData`, pm4Handlers.cpp L2241; the model the
    /// `raeen_gpu::agc::decode_submission` eager decoder already uses): body
    /// `[control, src_lo, src_hi, dst_lo, dst_hi, command]`, `command` low 21
    /// bits = `num_bytes`, `src_sel` = control>>29 & 3, `dst_sel` =
    /// control>>20 & 3. Selectors 0 and 3 both mean guest memory; `src_sel` 2
    /// is a 32-bit pattern fill with `src_lo` the value. Only the plain
    /// guest-memory copy and the pattern fill are honoured, under the same
    /// guards as the eager decoder (so the gate-off duplicate is the same
    /// write twice). GDS/immediate forms, short packets, absent/read-only
    /// [`GuestMemory`], and unreadable/unwritable ranges are consumed
    /// silently or skip with one rate-limited warn each — never a stream
    /// error (same posture as `cp_op_dma_data`).
    fn cp_op_it_dma_data(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let body_len = pm4::body_dw(cmd_id);
        // The modeled layouts need the full 6-dword body; anything shorter is
        // not a form we model — consume it by its encoded length like any
        // other unmodeled packet.
        if body_len < 6 {
            if self.first(SkipKey::Note("IT_DMA_DATA short packet")) {
                warn!(
                    cmd_id = format_args!("{cmd_id:#010x}"),
                    body_dw = body_len,
                    offset,
                    "IT_DMA_DATA shorter than the modeled layout — consumed without effect"
                );
            }
            return Ok(body_len);
        }
        let control = Self::body_at(body, 0, offset)?;
        let num_bytes = Self::body_at(body, 5, offset)? & 0x1f_ffff;
        let src_sel = (control >> 29) & 0x3;
        let dst_sel = (control >> 20) & 0x3;
        let dst = u64::from(Self::body_at(body, 3, offset)?)
            | (u64::from(Self::body_at(body, 4, offset)?) << 32);
        let fill = src_sel == 2 && matches!(dst_sel, 0 | 3);
        let copy = matches!(src_sel, 0 | 3) && matches!(dst_sel, 0 | 3);
        // Guards mirror the eager decoder exactly: an unmodeled selector form
        // (GDS, …) or a degenerate packet is consumed without effect.
        if !(fill || copy) || dst == 0 || num_bytes == 0 {
            return Ok(body_len);
        }
        let Some(mem) = mem else {
            if self.first(SkipKey::Note("IT_DMA_DATA needs GuestMemory")) {
                warn!(offset, "IT_DMA_DATA needs a GuestMemory accessor — skipped");
            }
            return Ok(body_len);
        };
        if fill {
            let value = Self::body_at(body, 1, offset)?;
            // Whole dwords only, mirroring the HLE eager fill byte-for-byte
            // (hardware requires num_bytes % 4 == 0; KytyPS5 EXITs otherwise).
            let mut bytes = Vec::with_capacity(num_bytes as usize & !3);
            for _ in 0..num_bytes / 4 {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            if mem.write_bytes(dst, &bytes) {
                debug!(
                    dst = format_args!("{dst:#x}"),
                    value = format_args!("{value:#x}"),
                    num_bytes,
                    "IT_DMA_DATA fill executed"
                );
            } else if self.first(SkipKey::Note("IT_DMA_DATA fill unwritable")) {
                warn!(
                    dst = format_args!("{dst:#x}"),
                    num_bytes,
                    offset,
                    "IT_DMA_DATA fill destination not writable guest memory — skipped"
                );
            }
            return Ok(body_len);
        }
        let src = u64::from(Self::body_at(body, 1, offset)?)
            | (u64::from(Self::body_at(body, 2, offset)?) << 32);
        if src == 0 {
            return Ok(body_len);
        }
        match mem.read_bytes(src, u64::from(num_bytes)) {
            Some(bytes) if mem.write_bytes(dst, &bytes) => {
                debug!(
                    src = format_args!("{src:#x}"),
                    dst = format_args!("{dst:#x}"),
                    num_bytes,
                    "IT_DMA_DATA copy executed"
                );
            }
            _ => {
                if self.first(SkipKey::Note("IT_DMA_DATA range unreadable/unwritable")) {
                    warn!(
                        src = format_args!("{src:#x}"),
                        dst = format_args!("{dst:#x}"),
                        num_bytes,
                        offset,
                        "IT_DMA_DATA source/destination not accessible guest memory — skipped"
                    );
                }
            }
        }
        Ok(body_len)
    }

    /// Execute a `WRITE_DATA` packet — the GPU writes an immediate payload of
    /// dwords straight to guest memory, the way a producing queue publishes a
    /// completion label a cross-queue `WAIT_REG_MEM` then polls on.
    ///
    /// Port of SharpEmu `ApplySubmittedWriteData` + `DecodeStandardWriteDataControl`
    /// / `DecodeAgcWriteDataControl` (AgcExports.cs:4494-4577). Body layout is
    /// the same for both forms: `[control, dst_lo, dst_hi, value0, value1, …]`;
    /// only the control-word decode differs (`standard` = raw `IT_WRITE_DATA`,
    /// else the AGC `IT_NOP`+`R_WRITE_DATA` byte-packed wrapper). Writes only
    /// when the destination selector picks memory (1, 2, 4 or 5); the address
    /// increments per dword unless the packet disables it (all values land on
    /// the same address then, last wins — hardware behaviour). Each written
    /// dword is recorded for the cross-queue wait latch ([`Self::record_produced`]).
    fn cp_op_write_data(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        standard: bool,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let body_len = pm4::body_dw(cmd_id);
        let control = Self::body_at(body, 0, offset)?;
        let dst = u64::from(Self::body_at(body, 1, offset)?)
            | (u64::from(Self::body_at(body, 2, offset)?) << 32);
        // dwordCount = total - 4 = body_dw - 3 (header + control + dst_lo/hi).
        let dword_count = body_len.saturating_sub(3);
        let (destination, increment) = if standard {
            // GFX10 PKT3_WRITE_DATA: DST_SEL in bits 11:8, ADDR_INCR is bit 16
            // (0 => increment). The reserved low byte must NOT be read as DST_SEL
            // (SharpEmu regression note, AgcExports.cs:4560-4563).
            ((control >> 8) & 0xF, (control & (1 << 16)) == 0)
        } else {
            // AGC byte-packed: DST_SEL is the low byte, ADDR_INCR the third byte.
            (control & 0xFF, ((control >> 16) & 0xFF) == 0)
        };
        let writes_memory = matches!(destination, 1 | 2 | 4 | 5);
        if !writes_memory || dword_count == 0 || dst == 0 {
            return Ok(body_len);
        }
        let Some(mem) = mem else {
            if self.first(SkipKey::Note("write_data_needs_memory")) {
                warn!(
                    offset,
                    dst = format_args!("{dst:#x}"),
                    "WRITE_DATA needs a GuestMemory writer — label not written"
                );
            }
            return Ok(body_len);
        };
        for index in 0..dword_count {
            let value = Self::body_at(body, 3 + index as usize, offset)?;
            let addr = if increment {
                dst + u64::from(index) * 4
            } else {
                dst
            };
            if mem.write_bytes(addr, &value.to_le_bytes()) {
                self.record_produced(addr, u64::from(value));
            } else {
                if self.first(SkipKey::Note("write_data_unwritable")) {
                    warn!(
                        offset,
                        addr = format_args!("{addr:#x}"),
                        "WRITE_DATA destination not writable guest memory — skipped"
                    );
                }
                break;
            }
        }
        Ok(body_len)
    }

    /// Execute a `RELEASE_MEM` (end-of-pipe) packet — the RDNA2/AGC way a queue
    /// signals "GPU work up to here is done" by writing a completion label to
    /// guest memory, which a cross-queue `WAIT_REG_MEM` polls on. This is the
    /// producer Minecraft's graphics queue waits on; consuming it "without
    /// effect" left the label unwritten and the wait permanently unmet.
    ///
    /// Port of SharpEmu `ApplySubmittedReleaseMem` (AGC, AgcExports.cs:5430-5496)
    /// and `ApplySubmittedStandardReleaseMem` + `DecodeStandardReleaseMemControl`
    /// (standard, AgcExports.cs:5349-5428). Both forms share the body layout
    /// `[event, control, dst_lo, dst_hi, data_lo, data_hi]` (body[0] is the
    /// event/GCR field the memory write ignores); only the control decode and
    /// the destination-selector gate differ. `DATA_SEL` picks the write width:
    /// 1 = 32-bit immediate, 2 = 64-bit immediate, 3/4 = a sampled GPU timestamp
    /// (the payload is ignored; a nonzero monotonic value is what the guest
    /// polls for). The immediate forms are recorded for the wait latch.
    fn cp_op_release_mem(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        standard: bool,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let body_len = pm4::body_dw(cmd_id);
        let control = Self::body_at(body, 1, offset)?;
        let dst = u64::from(Self::body_at(body, 2, offset)?)
            | (u64::from(Self::body_at(body, 3, offset)?) << 32);
        let (dst_sel_ok, data_sel) = if standard {
            // DST_SEL bits 17:16 (memory = 0 or 1); DATA_SEL bits 31:29.
            (
                matches!((control >> 16) & 0x3, 0 | 1),
                (control >> 29) & 0x7,
            )
        } else {
            // AGC form: DATA_SEL is the byte at bits 23:16; no DST_SEL gate.
            (true, (control >> 16) & 0xFF)
        };
        if !dst_sel_ok || dst == 0 {
            return Ok(body_len);
        }
        let Some(mem) = mem else {
            if self.first(SkipKey::Note("release_mem_needs_memory")) {
                warn!(
                    offset,
                    dst = format_args!("{dst:#x}"),
                    "RELEASE_MEM needs a GuestMemory writer — label not written"
                );
            }
            return Ok(body_len);
        };
        match data_sel {
            1 => {
                let value = Self::body_at(body, 4, offset)?;
                if mem.write_bytes(dst, &value.to_le_bytes()) {
                    self.record_produced(dst, u64::from(value));
                }
            }
            2 => {
                let value = u64::from(Self::body_at(body, 4, offset)?)
                    | (u64::from(Self::body_at(body, 5, offset)?) << 32);
                if mem.write_bytes(dst, &value.to_le_bytes()) {
                    self.record_produced(dst, value);
                }
            }
            3 | 4 => {
                // Hardware samples the GPU clock at the release point; the guest
                // uses the nonzero value as submit-completion state. A process-
                // monotonic counter is nonzero and strictly increasing, which is
                // what a "became nonzero" / ">= earlier sample" poll needs. Not
                // recorded for the equality latch (it is a counter, not a
                // specific reference the waiter compares equal to).
                let ts = next_release_timestamp();
                let _ = mem.write_bytes(dst, &ts.to_le_bytes());
            }
            _ => {
                if self.first(SkipKey::Note("release_mem_data_sel")) {
                    warn!(
                        offset,
                        data_sel, "RELEASE_MEM with no-write DATA_SEL — label not written"
                    );
                }
            }
        }
        Ok(body_len)
    }

    /// Parse and evaluate a wait-on-memory packet; arm a suspend when unmet.
    ///
    /// Port of SharpEmu `HandleSubmittedWaitRegMem` + `TryParseSubmittedWait`
    /// (AgcExports.cs:4508-4529 / 4534-4593 / 4595-4726), including its
    /// guards, in order:
    ///
    /// - compare 0 ("always") and reserved 7 are fail-open — never a waiter;
    /// - a null address, zero mask, or misaligned label is a malformed packet
    ///   and must not become a permanent waiter (warn once, continue);
    /// - no [`GuestMemory`] or an unreadable label → "cannot evaluate the
    ///   label — do not stall the DCB" (AgcExports.cs:4700-4703);
    /// - a satisfied condition keeps parsing;
    /// - an unmet condition arms [`CommandProcessor::pending_wait`]; the
    ///   walker turns that into [`RunOutcome::Suspended`] right after this
    ///   packet. The label is NEVER written to force the condition.
    fn cp_op_wait_mem(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        form: WaitForm,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let consumed = pm4::body_dw(cmd_id);
        let dword = |i: usize| Self::body_at(body, i, offset);
        let pair = |lo: u32, hi: u32| u64::from(lo) | (u64::from(hi) << 32);
        let spec = match form {
            WaitForm::Standard => WaitSpec {
                compare: dword(0)? & 0x7,
                address: pair(dword(1)?, dword(2)?),
                reference: u64::from(dword(3)?),
                mask: u64::from(dword(4)?),
                is_64: false,
            },
            WaitForm::Mem32 => WaitSpec {
                address: pair(dword(0)?, dword(1)?),
                mask: u64::from(dword(2)?),
                compare: dword(3)? & 0x7,
                reference: u64::from(dword(4)?),
                is_64: false,
            },
            WaitForm::Mem64 => WaitSpec {
                address: pair(dword(0)?, dword(1)?),
                mask: pair(dword(2)?, dword(3)?),
                reference: pair(dword(4)?, dword(5)?),
                compare: dword(6)? & 0x7,
                is_64: true,
            },
        };
        if spec.compare == 0 || spec.compare == 7 {
            return Ok(consumed);
        }
        let alignment = if spec.is_64 { 8 } else { 4 };
        if spec.address == 0 || spec.mask == 0 || spec.address % alignment != 0 {
            if self.first(SkipKey::Note("wait_mem_invalid_address_or_mask")) {
                warn!(
                    label = format_args!("{:#x}", spec.address),
                    mask = format_args!("{:#x}", spec.mask),
                    offset,
                    "WAIT_REG_MEM with null/misaligned label or zero mask — not honoured"
                );
            }
            return Ok(consumed);
        }
        let Some(mem) = mem else {
            if self.first(SkipKey::Note("wait_mem_no_memory")) {
                warn!(
                    label = format_args!("{:#x}", spec.address),
                    offset, "WAIT_REG_MEM needs a GuestMemory reader — not honoured"
                );
            }
            return Ok(consumed);
        };
        let Some(current) = spec.read_label(mem) else {
            // SharpEmu: cannot evaluate the label — do not stall the DCB.
            if self.first(SkipKey::Note("wait_mem_label_unreadable")) {
                warn!(
                    label = format_args!("{:#x}", spec.address),
                    offset, "WAIT_REG_MEM label unreadable — not honoured"
                );
            }
            return Ok(consumed);
        };
        if !spec.satisfied_by(current) {
            debug!(
                label = format_args!("{:#x}", spec.address),
                current = format_args!("{current:#x}"),
                reference = format_args!("{:#x}", spec.reference),
                mask = format_args!("{:#x}", spec.mask),
                compare = spec.compare,
                offset,
                "WAIT_REG_MEM unmet — suspending the walk after this packet"
            );
            self.pending_wait = Some(spec);
        }
        Ok(consumed)
    }

    /// Kyty: `cp_op_dispatch_direct` (GraphicsRun.cpp L2691).
    fn cp_op_dispatch_direct(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        sink: &mut dyn DrawSink,
    ) -> Result<u32, CpError> {
        let groups = [
            Self::body_at(body, 0, offset)?,
            Self::body_at(body, 1, offset)?,
            Self::body_at(body, 2, offset)?,
        ];
        let mode = Self::body_at(body, 3, offset)?;
        sink.dispatch_direct(&self.ctx, &self.ucfg, &self.sh_ctx, groups, mode)
            .map_err(|source| CpError::Draw { offset, source })?;
        Ok(pm4::body_dw(cmd_id))
    }

    /// Indirect compute dispatch: read the `[x, y, z]` thread-group counts
    /// from guest memory, then dispatch exactly like the direct form.
    ///
    /// Port of KytyPS5 `CpOpDispatchIndirect` (pm4Handlers.cpp L2009-2036) +
    /// `CommandProcessor::DispatchIndirect` (graphicsRun.cpp L1100-1113). Two
    /// encodings exist:
    /// - 3-DWORD packet, body `[data_offset, mode]`: args live at the
    ///   indirect-DISPATCH base (`IT_SET_BASE` select 1, shader type 1) plus
    ///   `data_offset`;
    /// - 4-DWORD packet, body `[addr_lo, addr_hi, mode]`: args live at the
    ///   absolute guest address.
    ///
    /// Where KytyPS5 `EXIT`s (no base programmed, unmappable args) this skips
    /// by encoded length with a once-warn, mirroring [`Self::cp_op_draw_indirect`]:
    /// the completion labels behind the dispatch must still run.
    fn cp_op_dispatch_indirect(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        sink: &mut dyn DrawSink,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let consumed = pm4::body_dw(cmd_id);
        let (args_addr, mode) = match consumed {
            2 => {
                let data_offset = u64::from(Self::body_at(body, 0, offset)?);
                let mode = Self::body_at(body, 1, offset)?;
                if self.indirect_dispatch_base == 0 {
                    if self.first(SkipKey::Note("indirect_dispatch_no_base")) {
                        warn!(
                            offset,
                            "indirect dispatch with no IT_SET_BASE(1, dispatch) programmed — skipped"
                        );
                    }
                    return Ok(consumed);
                }
                (self.indirect_dispatch_base + data_offset, mode)
            }
            3 => {
                let lo = Self::body_at(body, 0, offset)?;
                let hi = Self::body_at(body, 1, offset)?;
                let mode = Self::body_at(body, 2, offset)?;
                (u64::from(lo) | (u64::from(hi) << 32), mode)
            }
            other => {
                if self.first(SkipKey::Note("indirect_dispatch_len")) {
                    warn!(
                        body_dw = other,
                        offset, "IT_DISPATCH_INDIRECT with unknown body length — skipped"
                    );
                }
                return Ok(consumed);
            }
        };
        let Some(mem) = mem else {
            if self.first(SkipKey::Note("indirect_dispatch_no_memory")) {
                warn!(
                    offset,
                    "indirect dispatch needs a GuestMemory reader — skipped"
                );
            }
            return Ok(consumed);
        };
        let Some(groups) = mem.read_dwords(args_addr, 3) else {
            if self.first(SkipKey::Note("indirect_dispatch_unreadable_args")) {
                warn!(
                    args_addr = format_args!("{args_addr:#x}"),
                    offset, "indirect dispatch args unreadable — skipped"
                );
            }
            return Ok(consumed);
        };
        sink.dispatch_direct(
            &self.ctx,
            &self.ucfg,
            &self.sh_ctx,
            [groups[0], groups[1], groups[2]],
            mode,
        )
        .map_err(|source| CpError::Draw { offset, source })?;
        Ok(consumed)
    }

    /// Kyty: `cp_op_marker` — latches the user-data marker type.
    fn cp_op_marker(&mut self, cmd_id: u32, body: &[u32]) -> Result<u32, CpError> {
        if let Some(word) = body.first() {
            self.user_data_marker = match word & 0xff {
                0x4 => UserSgprType::Vsharp,
                0xd => UserSgprType::Region,
                _ => UserSgprType::Unknown,
            };
        }
        Ok(pm4::body_dw(cmd_id))
    }

    /// Kyty: `cp_op_draw_index_auto` (L2807).
    fn cp_op_draw_index_auto(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        sink: &mut dyn DrawSink,
    ) -> Result<u32, CpError> {
        let index_count = Self::body_at(body, 0, offset)?;
        // The AGC form (0xC0051010) carries flags; the raw IT form does not.
        let is_agc = pm4::op(cmd_id) == pm4::IT_NOP;
        let flags = if is_agc {
            Self::body_at(body, 1, offset)?
        } else {
            0
        };

        sink.draw_index_auto(&self.ctx, &self.ucfg, &self.sh_ctx, index_count, flags)
            .map_err(|source| CpError::Draw { offset, source })?;

        Ok(pm4::body_dw(cmd_id))
    }

    /// Kyty: `cp_op_draw_index` (L2757) — both encodings.
    ///
    /// - AGC form (`IT_NOP` + `R_DRAW_INDEX`, Kyty cmd 0xC008100C):
    ///   `[index_count, addr_lo, addr_hi, flags, type]`.
    /// - Raw form (`IT_DRAW_INDEX_2`, Kyty cmd 0xc0042700):
    ///   `[index_count, addr_lo, addr_hi, index_count again, 0]`; Kyty passes
    ///   `flags=0, type=1`. Kyty's duplicate-count/zero asserts are dropped
    ///   per the resilience policy (a mismatch is the guest's data, not a
    ///   stream fault).
    fn cp_op_draw_index(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        sink: &mut dyn DrawSink,
    ) -> Result<u32, CpError> {
        let is_agc = pm4::op(cmd_id) == pm4::IT_NOP;
        let index_count = Self::body_at(body, 0, offset)?;
        let lo = Self::body_at(body, 1, offset)?;
        let hi = Self::body_at(body, 2, offset)?;
        let index_addr = u64::from(lo) | (u64::from(hi) << 32);
        let (flags, index_type) = if is_agc {
            (
                Self::body_at(body, 3, offset)?,
                Self::body_at(body, 4, offset)?,
            )
        } else {
            (0, 1)
        };

        if self.first(SkipKey::Note("indexed_draw_degradation")) {
            warn!(
                index_addr = format_args!("{index_addr:#x}"),
                index_count,
                "indexed draw issued — default sinks degrade to a vertex-count-only \
                 auto draw (index buffer not fetched)"
            );
        }

        let draw = IndexedDraw {
            index_type_and_size: self.index_type_and_size,
            index_count,
            index_addr,
            flags,
            index_type,
        };
        sink.draw_index(&self.ctx, &self.ucfg, &self.sh_ctx, &draw)
            .map_err(|source| CpError::Draw { offset, source })?;

        Ok(pm4::body_dw(cmd_id))
    }

    /// `IT_DRAW_INDEX_OFFSET_2` (0x35) — an indexed draw from the *bound* index
    /// buffer, starting `INDEX_OFFSET` elements in.
    ///
    /// Kyty's `g_cp_op_func` has no entry for this opcode, so it fell to the
    /// resilience policy's default arm: warn once, skip by encoded length,
    /// forever. That is not a small gap — **it is the opcode Minecraft draws
    /// with**. Measured: 24,224 draw packets decoded in a 120 s run, of which
    /// ~64 reached Vulkan (~0.4%), every one of the rest dropped here after a
    /// single warning. The frame was black because almost nothing was drawn.
    ///
    /// Unlike [`Self::cp_op_draw_index`] the packet carries no address — the
    /// indices come from `IT_INDEX_BASE`, which is why a processor that ignores
    /// this op also silently ignores whatever that buffer holds.
    ///
    /// AMD PM4 body: `{ MAX_SIZE, INDEX_OFFSET, INDEX_COUNT, DRAW_INITIATOR }`.
    fn cp_op_draw_index_offset_2(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        sink: &mut dyn DrawSink,
    ) -> Result<u32, CpError> {
        // MAX_SIZE (body[0]) bounds the index buffer and DRAW_INITIATOR
        // (body[3]) selects the draw path; neither reaches a degraded sink.
        let index_offset = Self::body_at(body, 1, offset)?;
        let index_count = Self::body_at(body, 2, offset)?;

        let index_addr = self.index_base.saturating_add(
            u64::from(index_offset) * Self::index_element_bytes(self.index_type_and_size),
        );

        let draw = IndexedDraw {
            index_type_and_size: self.index_type_and_size,
            index_count,
            index_addr,
            // The raw (non-AGC) form carries no modifier flags, and passes the
            // same index_type as IT_DRAW_INDEX_2's raw form.
            flags: 0,
            index_type: 1,
        };
        sink.draw_index(&self.ctx, &self.ucfg, &self.sh_ctx, &draw)
            .map_err(|source| CpError::Draw { offset, source })?;

        Ok(pm4::body_dw(cmd_id))
    }

    /// Bytes per index for a latched `IT_INDEX_TYPE` dword.
    ///
    /// AMD `VGT_INDEX_TYPE` in bits 1:0: 0 = 16-bit, 1 = 32-bit, 2 = 8-bit.
    const fn index_element_bytes(index_type_and_size: u32) -> u64 {
        match index_type_and_size & 0x3 {
            0 => 2,
            2 => 1,
            // 1 (32-bit) and the reserved 3 both take the widest sane element,
            // so an offset is never computed short of where the indices are.
            _ => 4,
        }
    }

    /// Degraded indirect draw: recover a count from the first args record.
    ///
    /// Kyty has no indirect-draw handler at all (its `g_cp_op_func` leaves
    /// these opcodes null → `EXIT`). This extension reads only the first
    /// record at `indirect_draw_base + body[0]` — the AMD layout puts the
    /// vertex/index count in the first dword — and issues one degraded draw.
    /// Multi-draw counts/strides are **not** walked; that (and real per-draw
    /// state) needs `GraphicsRender`.
    fn cp_op_draw_indirect(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        sink: &mut dyn DrawSink,
        mem: Option<&dyn GuestMemory>,
        indexed: bool,
    ) -> Result<u32, CpError> {
        let consumed = pm4::body_dw(cmd_id);
        let op = pm4::op(cmd_id);
        let data_offset = u64::from(Self::body_at(body, 0, offset)?);

        if self.indirect_draw_base == 0 {
            if self.first(SkipKey::Note("indirect_draw_no_base")) {
                warn!(
                    op = format_args!("{:#04x}", op.0),
                    offset, "indirect draw with no IT_SET_BASE(1) programmed — skipped"
                );
            }
            return Ok(consumed);
        }
        let Some(mem) = mem else {
            if self.first(SkipKey::Note("indirect_draw_no_memory")) {
                warn!(
                    op = format_args!("{:#04x}", op.0),
                    offset, "indirect draw needs a GuestMemory reader — skipped"
                );
            }
            return Ok(consumed);
        };
        let args_addr = self.indirect_draw_base + data_offset;
        let Some(args) = mem.read_dwords(args_addr, 2) else {
            if self.first(SkipKey::Note("indirect_draw_unreadable_args")) {
                warn!(
                    args_addr = format_args!("{args_addr:#x}"),
                    offset, "indirect draw args unreadable — skipped"
                );
            }
            return Ok(consumed);
        };
        let args = DrawIndirectArgs {
            count: args[0],
            instance_count: args[1],
        };

        if self.first(SkipKey::Note("indirect_draw_degradation")) {
            warn!(
                op = format_args!("{:#04x}", op.0),
                count = args.count,
                instance_count = args.instance_count,
                indexed,
                "indirect draw degraded: first args record only, count-only draw \
                 (multi-draw stride/count not walked)"
            );
        }

        if args.count == 0 {
            return Ok(consumed);
        }

        let result = if indexed {
            let draw = IndexedDraw {
                index_type_and_size: self.index_type_and_size,
                index_count: args.count,
                index_addr: self.index_base,
                flags: 0,
                index_type: 0,
            };
            sink.draw_index(&self.ctx, &self.ucfg, &self.sh_ctx, &draw)
        } else {
            sink.draw_index_auto(&self.ctx, &self.ucfg, &self.sh_ctx, args.count, 0)
        };
        result.map_err(|source| CpError::Draw { offset, source })?;

        Ok(consumed)
    }

    /// Kyty: `cp_op_indirect_{cx,sh,uc}_regs` (L3018/L3050/L3082) — the body
    /// is `[num_regs, addr_lo, addr_hi]`; the pointed-to buffer holds
    /// `(offset, value)` pairs fed to the per-register setters.
    fn cp_op_indirect_regs(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        file: RegFile,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let consumed = pm4::body_dw(cmd_id);
        let num_regs = Self::body_at(body, 0, offset)?;
        let lo = Self::body_at(body, 1, offset)?;
        let hi = Self::body_at(body, 2, offset)?;
        let addr = u64::from(lo) | (u64::from(hi) << 32);

        // Kyty EXITs on nullptr/0; a garbage pointer must not kill the DCB.
        if num_regs == 0 || addr == 0 {
            if self.first(SkipKey::Note("indirect_regs_null")) {
                warn!(%file, num_regs, addr = format_args!("{addr:#x}"), offset,
                    "indirect register packet with null pointer or zero count — skipped");
            }
            return Ok(consumed);
        }
        // Defensive cap: the largest real register file is UC_NUM (16384).
        if num_regs as usize > pm4::UC_NUM {
            if self.first(SkipKey::Note("indirect_regs_count")) {
                warn!(%file, num_regs, offset,
                    "indirect register packet count exceeds the register file — skipped");
            }
            return Ok(consumed);
        }
        let Some(mem) = mem else {
            if self.first(SkipKey::Note("indirect_regs_no_memory")) {
                warn!(%file, offset,
                    "indirect register packet needs a GuestMemory reader — skipped");
            }
            return Ok(consumed);
        };
        let Some(pairs) = mem.read_dwords(addr, num_regs * 2) else {
            if self.first(SkipKey::Note("indirect_regs_unreadable")) {
                warn!(%file, addr = format_args!("{addr:#x}"), offset,
                    "indirect register buffer unreadable — skipped");
            }
            return Ok(consumed);
        };

        for pair in pairs.chunks_exact(2) {
            let (reg, value) = (pair[0], pair[1]);
            match file {
                RegFile::Context => self.set_context_register(pm4::strip_fake(reg), value),
                RegFile::Shader => self.set_shader_register(pm4::strip_fake(reg), value),
                RegFile::UserConfig => self.set_uconfig_register(reg & 0xEFFF_FFFF, value),
            }
        }
        // RAEEN_TRACE_INDIRECT: the out-of-range register warns say WHAT was
        // skipped but not WHY the table looks like vertex data. Dump the raw
        // packet and the table head for the first N packets with any
        // out-of-file offset — mis-decoded layout vs stale/raced memory is
        // decidable from this line alone.
        if trace_indirect_enabled() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEEN: AtomicU32 = AtomicU32::new(0);
            let oob = pairs
                .chunks_exact(2)
                .filter(|p| {
                    let r = pm4::strip_fake(p[0]) as usize;
                    match file {
                        RegFile::Context => r >= pm4::CX_NUM,
                        RegFile::Shader => r >= pm4::SH_NUM,
                        RegFile::UserConfig => r >= pm4::UC_NUM,
                    }
                })
                .count();
            if oob != 0 && SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(
                    %file,
                    num_regs,
                    addr = format_args!("{addr:#x}"),
                    oob,
                    head = format_args!("{:08x?}", &pairs[..pairs.len().min(16)]),
                    "TRACE_INDIRECT: indirect register table holds out-of-file offsets"
                );
            }
        }
        Ok(consumed)
    }

    /// Kyty: `cp_op_set_context_reg` (L3288).
    ///
    /// `body[0]` is a **relative flat dword index** — the driver already
    /// subtracted the context base, so no base math happens here.
    fn cp_op_set_context_reg(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
    ) -> Result<u32, CpError> {
        let reg = pm4::strip_fake(Self::body_at(body, 0, offset)?);
        let values = &body[1..];

        // Kyty's only multi-register context block on this path.
        if reg == pm4::PA_SC_SCREEN_SCISSOR_TL && values.len() >= 2 {
            return self
                .hw_ctx_set_screen_scissor(values, offset)
                .map(|n| n + 1);
        }

        let count = Self::reg_count(cmd_id, values, offset)?;
        for (i, &value) in values.iter().enumerate().take(count) {
            self.set_context_register(reg + i as u32, value);
        }
        Ok(count as u32 + 1)
    }

    /// How many registers a `SET_*_REG` packet writes, per its header.
    ///
    /// The header's count is authoritative. Clamping it to what the buffer
    /// happens to hold would turn a truncated packet into a silent success —
    /// the exact failure mode the never-silent rule exists to prevent.
    fn reg_count(cmd_id: u32, values: &[u32], offset: u32) -> Result<usize, CpError> {
        let count = pm4::body_dw(cmd_id).saturating_sub(1) as usize;
        if count > values.len() {
            return Err(CpError::Truncated {
                offset,
                need: count as u32 + 2,
                remaining: values.len() as u32 + 2,
            });
        }
        Ok(count)
    }

    /// Kyty: `hw_ctx_set_screen_scissor` (L~2700). TL and BR in one packet.
    fn hw_ctx_set_screen_scissor(&mut self, values: &[u32], _offset: u32) -> Result<u32, CpError> {
        // Direct and indirect register packets must share the same semantics.
        self.set_context_register(pm4::PA_SC_SCREEN_SCISSOR_TL, values[0]);
        self.set_context_register(pm4::PA_SC_SCREEN_SCISSOR_BR, values[1]);
        Ok(2)
    }

    /// The per-register context setters — Kyty's `g_hw_ctx_indirect_func`
    /// table (`graphics_init_jmp_tables_cx_indirect`, L3482).
    ///
    /// These take `(offset, value)` and touch no memory, so they serve direct
    /// `SET_CONTEXT_REG` writes as well as the indirect packet. An unknown or
    /// out-of-range register is a rate-limited warn and a skip, never a fault
    /// (resilience policy).
    fn set_context_register(&mut self, reg: u32, value: u32) {
        if reg as usize >= pm4::CX_NUM {
            if self.first(SkipKey::Reg(RegFile::Context, reg))
                && warn_skip_reg_once(RegFile::Context, reg)
            {
                warn!(
                    reg = format_args!("{reg:#06x}"),
                    "context register index out of range — write skipped"
                );
            }
            return;
        }
        let slot_of = |base: u32, stride: u32| ((reg - base) / stride) as usize;

        match reg {
            pm4::CB_TARGET_MASK => self.ctx.render_target_mask = value,

            // PA_SU_SC_MODE_CNTL was a HALF-WIRED feature: the pm4 constant
            // (pm4.rs), the full `ModeControl` struct (hw_regs.rs) and the
            // consumer that turns cull_front/cull_back into a Vulkan cull mode
            // (raeen-gpu draw_translate) all existed, but nothing ever decoded
            // the register — so `ctx.mode_control` stayed at its all-false
            // default and EVERY draw in EVERY title rasterized with
            // CullModeFlags::NONE. Field layout is Kyty's
            // (Pm4.h PA_SU_SC_MODE_CNTL_*_SHIFT/_MASK, L489-510).
            pm4::PA_SU_SC_MODE_CNTL => {
                let m = &mut self.ctx.mode_control;
                m.cull_front = value & 0x1 != 0;
                m.cull_back = (value >> 1) & 0x1 != 0;
                m.face = (value >> 2) & 0x1 != 0;
                m.poly_mode = ((value >> 3) & 0x3) as u8;
                m.polymode_front_ptype = ((value >> 5) & 0x7) as u8;
                m.polymode_back_ptype = ((value >> 8) & 0x7) as u8;
                m.poly_offset_front_enable = (value >> 11) & 0x1 != 0;
                m.poly_offset_back_enable = (value >> 12) & 0x1 != 0;
                m.vtx_window_offset_enable = (value >> 16) & 0x1 != 0;
                m.provoking_vtx_last = (value >> 19) & 0x1 != 0;
                m.persp_corr_dis = (value >> 20) & 0x1 != 0;
            }

            // Kyty's Gen5 primary-register lists program scissors through
            // R_CX_REGS_INDIRECT. Keep these as individual setters so direct
            // SET_CONTEXT_REG and indirect `(offset, value)` pairs converge.
            pm4::PA_SC_SCREEN_SCISSOR_TL => {
                use pm4::pa_sc_screen_scissor as f;
                self.ctx.screen_viewport.screen_scissor_left =
                    i32::from(pm4::field(value, f::TL_X) as u16 as i16);
                self.ctx.screen_viewport.screen_scissor_top =
                    i32::from(pm4::field(value, f::TL_Y) as u16 as i16);
            }
            pm4::PA_SC_SCREEN_SCISSOR_BR => {
                use pm4::pa_sc_screen_scissor as f;
                self.ctx.screen_viewport.screen_scissor_right =
                    i32::from(pm4::field(value, f::BR_X) as u16 as i16);
                self.ctx.screen_viewport.screen_scissor_bottom =
                    i32::from(pm4::field(value, f::BR_Y) as u16 as i16);
            }
            pm4::PA_SC_GENERIC_SCISSOR_TL => {
                use pm4::pa_sc_offset_scissor as f;
                let sv = &mut self.ctx.screen_viewport;
                sv.generic_scissor_left = i32::from(pm4::field(value, f::TL_X) as u16 as i16);
                sv.generic_scissor_top = i32::from(pm4::field(value, f::TL_Y) as u16 as i16);
                sv.generic_scissor_window_offset_enable =
                    pm4::field(value, f::WINDOW_OFFSET_DISABLE) == 0;
            }
            pm4::PA_SC_GENERIC_SCISSOR_BR => {
                use pm4::pa_sc_offset_scissor as f;
                let sv = &mut self.ctx.screen_viewport;
                sv.generic_scissor_right = i32::from(pm4::field(value, f::BR_X) as u16 as i16);
                sv.generic_scissor_bottom = i32::from(pm4::field(value, f::BR_Y) as u16 as i16);
            }
            r if r >= pm4::PA_SC_VPORT_SCISSOR_0_TL
                && r < pm4::PA_SC_VPORT_SCISSOR_0_TL
                    + (crate::hw_regs::ScreenViewport::VIEWPORTS_MAX as u32 * 2) =>
            {
                use pm4::pa_sc_offset_scissor as f;
                let rel = r - pm4::PA_SC_VPORT_SCISSOR_0_TL;
                let vp = &mut self.ctx.screen_viewport.viewports[(rel / 2) as usize];
                if rel & 1 == 0 {
                    vp.viewport_scissor_left = i32::from(pm4::field(value, f::TL_X) as u16 as i16);
                    vp.viewport_scissor_top = i32::from(pm4::field(value, f::TL_Y) as u16 as i16);
                    vp.viewport_scissor_window_offset_enable =
                        pm4::field(value, f::WINDOW_OFFSET_DISABLE) == 0;
                } else {
                    vp.viewport_scissor_right = i32::from(pm4::field(value, f::BR_X) as u16 as i16);
                    vp.viewport_scissor_bottom =
                        i32::from(pm4::field(value, f::BR_Y) as u16 as i16);
                }
            }

            // Shader-facing context registers (Kyty: g_hw_ctx_indirect_func,
            // GraphicsRun.cpp L3805-3825). These feed `ctx.sh_regs`, which the
            // guest-shader translation (ShaderMemory Phase 2) reads.
            pm4::SPI_VS_OUT_CONFIG => self.ctx.sh_regs.spi_vs_out_config = value,
            pm4::SPI_PS_INPUT_ENA => self.ctx.sh_regs.ps_input_ena = value,
            pm4::SPI_PS_INPUT_ADDR => self.ctx.sh_regs.ps_input_addr = value,
            pm4::SPI_PS_IN_CONTROL => self.ctx.sh_regs.ps_in_control = value,
            pm4::SPI_SHADER_COL_FORMAT => {
                for (i, mode) in self.ctx.sh_regs.target_output_mode.iter_mut().enumerate() {
                    *mode = ((value >> (i * 4)) & 0xF) as u8;
                }
            }
            r if (pm4::SPI_PS_INPUT_CNTL_0..pm4::SPI_PS_INPUT_CNTL_0 + 32).contains(&r) => {
                let slot = (r - pm4::SPI_PS_INPUT_CNTL_0) as usize;
                self.ctx.sh_regs.ps_interpolator_settings[slot] = value;
            }
            pm4::DB_SHADER_CONTROL => {
                // Kyty: DB_SHADER_CONTROL decode (GraphicsRun.cpp L3820).
                self.ctx.sh_regs.db_shader_control = DepthShaderControl {
                    other_bits: value & 0xFFFF_9B8E,
                    conservative_z_export_value: ((value >> 13) & 0x3) as u8,
                    shader_z_behavior: ((value >> 4) & 0x3) as u8,
                    shader_kill_enable: (value >> 6) & 0x1 != 0,
                    shader_z_export_enable: value & 0x1 != 0,
                    shader_execute_on_noop: (value >> 10) & 0x1 != 0,
                };
            }

            r if (pm4::CB_COLOR0_BASE..=pm4::CB_COLOR7_BASE).contains(&r)
                && (r - pm4::CB_COLOR0_BASE) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_BASE, pm4::CB_COLOR_SLOT_STRIDE);
                let base = &mut self.ctx.render_targets[slot].base;
                // Kyty preserves the high/low bits the BASE_EXT register owns.
                base.addr &= 0xFFFF_FF00_0000_00FF;
                base.addr |= u64::from(value) << 8;
            }

            r if (pm4::CB_COLOR0_INFO..=pm4::CB_COLOR7_INFO).contains(&r)
                && (r - pm4::CB_COLOR0_INFO) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_INFO, pm4::CB_COLOR_SLOT_STRIDE);
                use pm4::cb_color_info as f;
                self.ctx.render_targets[slot].info = ColorInfo {
                    format: pm4::field(value, f::FORMAT),
                    channel_type: pm4::field(value, f::NUMBER_TYPE),
                    channel_order: pm4::field(value, f::COMP_SWAP),
                    cmask_fast_clear_enable: pm4::field(value, f::FAST_CLEAR) != 0,
                    fmask_compression_enable: pm4::field(value, f::COMPRESSION) != 0,
                    blend_clamp: pm4::field(value, f::BLEND_CLAMP) != 0,
                    blend_bypass: pm4::field(value, f::BLEND_BYPASS) != 0,
                    round_mode: pm4::field(value, f::ROUND_MODE) != 0,
                    cmask_tile_mode: pm4::field(value, f::CMASK_IS_LINEAR),
                    dcc_compression_enable: pm4::field(value, f::DCC_ENABLE) != 0,
                    neo_mode: pm4::field(value, f::ALT_TILE_MODE) != 0,
                    ..ColorInfo::default()
                };
            }

            r if (pm4::CB_COLOR0_ATTRIB2..=pm4::CB_COLOR7_ATTRIB2).contains(&r) => {
                let slot = (r - pm4::CB_COLOR0_ATTRIB2) as usize;
                use pm4::cb_color_attrib2 as f;
                self.ctx.render_targets[slot].attrib2 = ColorAttrib2 {
                    height: pm4::field(value, f::MIP0_HEIGHT),
                    width: pm4::field(value, f::MIP0_WIDTH),
                    num_mip_levels: pm4::field(value, f::MAX_MIP),
                };
            }

            r if (pm4::CB_COLOR0_ATTRIB3..=pm4::CB_COLOR7_ATTRIB3).contains(&r) => {
                let slot = (r - pm4::CB_COLOR0_ATTRIB3) as usize;
                use pm4::cb_color_attrib3 as f;
                self.ctx.render_targets[slot].attrib3 = ColorAttrib3 {
                    depth: pm4::field(value, f::MIP0_DEPTH),
                    tile_mode: pm4::field(value, f::COLOR_SW_MODE),
                    dimension: pm4::field(value, f::RESOURCE_TYPE),
                    cmask_pipe_aligned: pm4::field(value, f::CMASK_PIPE_ALIGNED) != 0,
                    dcc_pipe_aligned: pm4::field(value, f::DCC_PIPE_ALIGNED) != 0,
                };
            }

            // Viewport scale/offset: six consecutive registers per viewport.
            r if r >= pm4::PA_CL_VPORT_XSCALE
                && r < pm4::PA_CL_VPORT_XSCALE
                    + pm4::PA_CL_VPORT_STRIDE
                        * crate::hw_regs::ScreenViewport::VIEWPORTS_MAX as u32 =>
            {
                let rel = r - pm4::PA_CL_VPORT_XSCALE;
                let slot = (rel / pm4::PA_CL_VPORT_STRIDE) as usize;
                let f = f32::from_bits(value);
                let vp = &mut self.ctx.screen_viewport.viewports[slot];
                match rel % pm4::PA_CL_VPORT_STRIDE {
                    0 => vp.xscale = f,
                    1 => vp.xoffset = f,
                    2 => vp.yscale = f,
                    3 => vp.yoffset = f,
                    4 => vp.zscale = f,
                    _ => vp.zoffset = f,
                }
            }

            pm4::DB_DEPTH_CONTROL => {
                let c = &mut self.ctx.depth_control;
                c.stencil_enable = (value & 0x1) != 0;
                c.z_enable = (value & 0x2) != 0;
                c.z_write_enable = (value & 0x4) != 0;
                c.depth_bounds_enable = (value & 0x8) != 0;
                c.zfunc = ((value >> 4) & 0x7) as u8;
                c.backface_enable = (value & 0x80) != 0;
                c.stencilfunc = ((value >> 8) & 0x7) as u8;
                c.stencilfunc_bf = ((value >> 20) & 0x7) as u8;
            }

            // ---- Depth/stencil surface registers (Kyty GraphicsRun.cpp) ----
            pm4::DB_RENDER_CONTROL => {
                // Kyty: hw_ctx_set_render_control (GraphicsRun.cpp L1887).
                use pm4::db_render_control as f;
                self.ctx.render_control = crate::hw_regs::RenderControl {
                    depth_clear_enable: pm4::field(value, f::DEPTH_CLEAR_ENABLE) != 0,
                    stencil_clear_enable: pm4::field(value, f::STENCIL_CLEAR_ENABLE) != 0,
                    resummarize_enable: pm4::field(value, f::RESUMMARIZE_ENABLE) != 0,
                    stencil_compress_disable: pm4::field(value, f::STENCIL_COMPRESS_DISABLE) != 0,
                    depth_compress_disable: pm4::field(value, f::DEPTH_COMPRESS_DISABLE) != 0,
                    copy_centroid: pm4::field(value, f::COPY_CENTROID) != 0,
                    copy_sample: pm4::field(value, f::COPY_SAMPLE) as u8,
                };
            }

            pm4::DB_DEPTH_VIEW => {
                // Kyty: g_hw_ctx_indirect_func[DB_DEPTH_VIEW] (GraphicsRun.cpp L3944).
                use pm4::db_depth_view as f;
                self.ctx.depth_render_target.depth_view = crate::hw_regs::DepthDepthView {
                    slice_start: pm4::field(value, f::SLICE_START)
                        + (pm4::field(value, f::SLICE_START_HI) << 11),
                    slice_max: pm4::field(value, f::SLICE_MAX)
                        + (pm4::field(value, f::SLICE_MAX_HI) << 11),
                    depth_write_disable: pm4::field(value, f::Z_READ_ONLY) != 0,
                    stencil_write_disable: pm4::field(value, f::STENCIL_READ_ONLY) != 0,
                    current_mip_level: pm4::field(value, f::MIPID) as u8,
                };
            }

            pm4::DB_DEPTH_SIZE_XY => {
                // Kyty: g_hw_ctx_indirect_func[DB_DEPTH_SIZE_XY] (GraphicsRun.cpp L3955).
                use pm4::db_depth_size_xy as f;
                self.ctx.depth_render_target.size = crate::hw_regs::DepthDepthSizeXy {
                    x_max: pm4::field(value, f::X_MAX) as u16,
                    y_max: pm4::field(value, f::Y_MAX) as u16,
                };
            }

            pm4::DB_DEPTH_BOUNDS_MIN => self.ctx.depth_bounds_min = f32::from_bits(value),
            pm4::DB_DEPTH_BOUNDS_MAX => self.ctx.depth_bounds_max = f32::from_bits(value),

            pm4::DB_STENCIL_CLEAR => {
                // Kyty: hw_ctx_set_stencil_clear (GraphicsRun.cpp L2101).
                self.ctx.stencil_clear_value =
                    pm4::field(value, pm4::db_stencil_clear::CLEAR) as u8;
            }

            pm4::DB_DEPTH_CLEAR => {
                // Kyty: hw_ctx_set_depth_clear (GraphicsRun.cpp L1597).
                self.ctx.depth_clear_value = f32::from_bits(value);
            }

            pm4::DB_DEPTH_INFO => {
                // Kyty: hw_ctx_set_depth_render_target's DB_DEPTH_INFO slice
                // (GraphicsRun.cpp L1713). Tiling metadata — the offscreen path
                // never reads guest depth memory, so nothing consumes this yet.
                use pm4::db_depth_info as f;
                self.ctx.depth_render_target.depth_info =
                    crate::hw_regs::DepthRenderTargetDepthInfo {
                        addr5_swizzle_mask: pm4::field(value, f::ADDR5_SWIZZLE_MASK),
                        array_mode: pm4::field(value, f::ARRAY_MODE),
                        pipe_config: pm4::field(value, f::PIPE_CONFIG),
                        bank_width: pm4::field(value, f::BANK_WIDTH),
                        bank_height: pm4::field(value, f::BANK_HEIGHT),
                        macro_tile_aspect: pm4::field(value, f::MACRO_TILE_ASPECT),
                        num_banks: pm4::field(value, f::NUM_BANKS),
                    };
            }

            pm4::DB_Z_INFO => {
                // Kyty: hw_ctx_set_depth_render_target's Z_INFO slice
                // (GraphicsRun.cpp L1647).
                use pm4::db_z_info as f;
                self.ctx.depth_render_target.z_info = crate::hw_regs::DepthZInfo {
                    format: pm4::field(value, f::FORMAT),
                    num_samples: pm4::field(value, f::NUM_SAMPLES),
                    embedded_sample_locations: pm4::field(value, f::ITERATE_FLUSH) != 0,
                    partially_resident: pm4::field(value, f::PARTIALLY_RESIDENT) != 0,
                    num_mip_levels: pm4::field(value, f::MAXMIP) as u8,
                    tile_mode_index: pm4::field(value, f::TILE_MODE_INDEX),
                    plane_compression: pm4::field(value, f::DECOMPRESS_ON_N_ZPLANES) as u8,
                    expclear_enabled: pm4::field(value, f::ALLOW_EXPCLEAR) != 0,
                    tile_surface_enable: pm4::field(value, f::TILE_SURFACE_ENABLE) != 0,
                    zrange_precision: pm4::field(value, f::ZRANGE_PRECISION),
                };
            }

            pm4::DB_STENCIL_INFO => {
                // Kyty: g_hw_ctx_indirect_func[DB_STENCIL_INFO] (GraphicsRun.cpp
                // L3849). Kyty's direct-write handler (L2130) reads buffer[1] — a
                // fused-packet quirk; a standalone write carries the value here.
                use pm4::db_stencil_info as f;
                self.ctx.depth_render_target.stencil_info = crate::hw_regs::DepthStencilInfo {
                    format: pm4::field(value, f::FORMAT),
                    texture_compatible_stencil: pm4::field(value, f::ITERATE_FLUSH) != 0,
                    partially_resident: pm4::field(value, f::PARTIALLY_RESIDENT) != 0,
                    tile_split: pm4::field(value, f::RESERVED_FIELD_1),
                    tile_mode_index: pm4::field(value, f::TILE_MODE_INDEX),
                    expclear_enabled: pm4::field(value, f::ALLOW_EXPCLEAR) != 0,
                    tile_stencil_disable: pm4::field(value, f::TILE_STENCIL_DISABLE) != 0,
                };
            }

            // Depth/stencil base addresses assemble exactly like Kyty's indirect
            // handlers (GraphicsRun.cpp L3864-3942): LO shifts into bits 8..40,
            // HI's low byte into 40..48.
            pm4::DB_Z_READ_BASE => {
                let base = &mut self.ctx.depth_render_target.z_read_base_addr;
                *base &= 0xFFFF_FF00_0000_00FF;
                *base |= u64::from(value) << 8;
            }
            pm4::DB_Z_READ_BASE_HI => {
                let base = &mut self.ctx.depth_render_target.z_read_base_addr;
                *base &= 0xFFFF_00FF_FFFF_FFFF;
                *base |= u64::from(value & 0xFF) << 40;
            }
            pm4::DB_STENCIL_READ_BASE => {
                let base = &mut self.ctx.depth_render_target.stencil_read_base_addr;
                *base &= 0xFFFF_FF00_0000_00FF;
                *base |= u64::from(value) << 8;
            }
            pm4::DB_STENCIL_READ_BASE_HI => {
                let base = &mut self.ctx.depth_render_target.stencil_read_base_addr;
                *base &= 0xFFFF_00FF_FFFF_FFFF;
                *base |= u64::from(value & 0xFF) << 40;
            }
            pm4::DB_Z_WRITE_BASE => {
                let base = &mut self.ctx.depth_render_target.z_write_base_addr;
                *base &= 0xFFFF_FF00_0000_00FF;
                *base |= u64::from(value) << 8;
            }
            pm4::DB_Z_WRITE_BASE_HI => {
                let base = &mut self.ctx.depth_render_target.z_write_base_addr;
                *base &= 0xFFFF_00FF_FFFF_FFFF;
                *base |= u64::from(value & 0xFF) << 40;
            }
            pm4::DB_STENCIL_WRITE_BASE => {
                let base = &mut self.ctx.depth_render_target.stencil_write_base_addr;
                *base &= 0xFFFF_FF00_0000_00FF;
                *base |= u64::from(value) << 8;
            }
            pm4::DB_STENCIL_WRITE_BASE_HI => {
                let base = &mut self.ctx.depth_render_target.stencil_write_base_addr;
                *base &= 0xFFFF_00FF_FFFF_FFFF;
                *base |= u64::from(value & 0xFF) << 40;
            }
            pm4::DB_HTILE_DATA_BASE => {
                let base = &mut self.ctx.depth_render_target.htile_data_base_addr;
                *base &= 0xFFFF_FF00_0000_00FF;
                *base |= u64::from(value) << 8;
            }
            pm4::DB_HTILE_DATA_BASE_HI => {
                let base = &mut self.ctx.depth_render_target.htile_data_base_addr;
                *base &= 0xFFFF_00FF_FFFF_FFFF;
                *base |= u64::from(value & 0xFF) << 40;
            }

            pm4::DB_DEPTH_SIZE => {
                use pm4::db_depth_size as f;
                let z = &mut self.ctx.depth_render_target;
                z.pitch_div8_minus1 = pm4::field(value, f::PITCH_TILE_MAX);
                z.height_div8_minus1 = pm4::field(value, f::HEIGHT_TILE_MAX);
            }
            pm4::DB_DEPTH_SLICE => {
                self.ctx.depth_render_target.slice_div64_minus1 =
                    pm4::field(value, pm4::db_depth_slice::SLICE_TILE_MAX);
            }

            pm4::DB_STENCIL_CONTROL => {
                // Kyty: hw_ctx_set_stencil_control (GraphicsRun.cpp L2111).
                use pm4::db_stencil_control as f;
                self.ctx.stencil_control = crate::hw_regs::StencilControl {
                    stencil_fail: pm4::field(value, f::STENCILFAIL) as u8,
                    stencil_zpass: pm4::field(value, f::STENCILZPASS) as u8,
                    stencil_zfail: pm4::field(value, f::STENCILZFAIL) as u8,
                    stencil_fail_bf: pm4::field(value, f::STENCILFAIL_BF) as u8,
                    stencil_zpass_bf: pm4::field(value, f::STENCILZPASS_BF) as u8,
                    stencil_zfail_bf: pm4::field(value, f::STENCILZFAIL_BF) as u8,
                };
            }

            pm4::DB_STENCILREFMASK => {
                // Kyty: hw_ctx_set_stencil_mask's front half (GraphicsRun.cpp L2157).
                use pm4::db_stencilrefmask as f;
                let m = &mut self.ctx.stencil_mask;
                m.stencil_testval = pm4::field(value, f::STENCILTESTVAL) as u8;
                m.stencil_mask = pm4::field(value, f::STENCILMASK) as u8;
                m.stencil_writemask = pm4::field(value, f::STENCILWRITEMASK) as u8;
                m.stencil_opval = pm4::field(value, f::STENCILOPVAL) as u8;
            }
            pm4::DB_STENCILREFMASK_BF => {
                // Same layout, back face (Kyty Pm4.h L338-346).
                use pm4::db_stencilrefmask as f;
                let m = &mut self.ctx.stencil_mask;
                m.stencil_testval_bf = pm4::field(value, f::STENCILTESTVAL) as u8;
                m.stencil_mask_bf = pm4::field(value, f::STENCILMASK) as u8;
                m.stencil_writemask_bf = pm4::field(value, f::STENCILWRITEMASK) as u8;
                m.stencil_opval_bf = pm4::field(value, f::STENCILOPVAL) as u8;
            }

            pm4::DB_HTILE_SURFACE => {
                // Kyty: hw_ctx_set_depth_render_target's HTile slice
                // (GraphicsRun.cpp L1734). Tracked; HTile is not implemented.
                use pm4::db_htile_surface as f;
                self.ctx.depth_render_target.htile_surface =
                    crate::hw_regs::DepthRenderTargetHTileSurface {
                        linear: pm4::field(value, f::LINEAR),
                        full_cache: pm4::field(value, f::FULL_CACHE),
                        htile_uses_preload_win: pm4::field(value, f::HTILE_USES_PRELOAD_WIN),
                        preload: pm4::field(value, f::PRELOAD),
                        prefetch_width: pm4::field(value, f::PREFETCH_WIDTH),
                        prefetch_height: pm4::field(value, f::PREFETCH_HEIGHT),
                        dst_outside_zero_to_one: pm4::field(value, f::DST_OUTSIDE_ZERO_TO_ONE),
                    };
            }

            pm4::CB_COLOR_CONTROL => {
                self.ctx.color_control.mode = ((value >> 4) & 0x7) as u8;
                self.ctx.color_control.op = ((value >> 16) & 0xff) as u8;
            }

            // Kyty Pm4.h CB_BLEND0_CONTROL_* field layout. Per-slot alpha
            // blending state — Minecraft's UI writes these.
            r if (pm4::CB_BLEND0_CONTROL..pm4::CB_BLEND0_CONTROL + pm4::CB_BLEND_CONTROL_SLOTS)
                .contains(&r) =>
            {
                let slot = (r - pm4::CB_BLEND0_CONTROL) as usize;
                self.ctx.blend_control[slot] = crate::hw_regs::BlendControl {
                    color_srcblend: (value & 0x1f) as u8,
                    color_comb_fcn: ((value >> 5) & 0x7) as u8,
                    color_destblend: ((value >> 8) & 0x1f) as u8,
                    alpha_srcblend: ((value >> 16) & 0x1f) as u8,
                    alpha_comb_fcn: ((value >> 21) & 0x7) as u8,
                    alpha_destblend: ((value >> 24) & 0x1f) as u8,
                    separate_alpha_blend: (value >> 29) & 0x1 != 0,
                    enable: (value >> 30) & 0x1 != 0,
                };
            }

            pm4::CB_BLEND_RED => self.ctx.blend_color.red = f32::from_bits(value),
            pm4::CB_BLEND_GREEN => self.ctx.blend_color.green = f32::from_bits(value),
            pm4::CB_BLEND_BLUE => self.ctx.blend_color.blue = f32::from_bits(value),
            pm4::CB_BLEND_ALPHA => self.ctx.blend_color.alpha = f32::from_bits(value),

            _ => {
                if self.first(SkipKey::Reg(RegFile::Context, reg))
                    && warn_skip_reg_once(RegFile::Context, reg)
                {
                    warn!(
                        reg = format_args!("{reg:#06x}"),
                        "unknown context register — write skipped"
                    );
                }
            }
        }
    }

    /// Kyty: `cp_op_set_shader_reg` (L3311).
    fn cp_op_set_shader_reg(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
    ) -> Result<u32, CpError> {
        let reg = pm4::strip_fake(Self::body_at(body, 0, offset)?);
        let values = &body[1..];
        let count = Self::reg_count(cmd_id, values, offset)?;
        for (i, &value) in values.iter().enumerate().take(count) {
            self.set_shader_register(reg + i as u32, value);
        }
        Ok(count as u32 + 1)
    }

    fn set_shader_register(&mut self, reg: u32, value: u32) {
        // 32 user-SGPR registers per GRAPHICS stage on Gen5 (VS/PS/GS below);
        // compute is handled separately by its own 16-wide
        // COMPUTE_USER_DATA_0..15 range. Widened from Kyty's PS4-era 16 after
        // measuring ASTRO.BOT pixel shaders declaring up to 32 — see
        // `UserSgprInfo::SGPRS_MAX`. No SH-register collision: PS user data
        // starts at 0x0C with the next SH register (VS_0) at 0x4C, and GS_0
        // at 0x8C runs to ES_LO at 0xC8.
        const SGPRS: u32 = 32;
        if reg as usize >= pm4::SH_NUM {
            if self.first(SkipKey::Reg(RegFile::Shader, reg))
                && warn_skip_reg_once(RegFile::Shader, reg)
            {
                warn!(
                    reg = format_args!("{reg:#06x}"),
                    "shader register index out of range — write skipped"
                );
            }
            return;
        }
        if matches!(
            reg,
            pm4::SPI_SHADER_PGM_LO_PS
                | pm4::SPI_SHADER_PGM_HI_PS
                | pm4::SPI_SHADER_PGM_CHKSUM_PS
                | pm4::SPI_SHADER_PGM_LO_ES
                | pm4::SPI_SHADER_PGM_HI_ES
                | pm4::SPI_SHADER_PGM_CHKSUM_GS
        ) && self.trace_shader_bind()
        {
            warn!(
                reg = format_args!("{reg:#06x}"),
                value = format_args!("{value:#010x}"),
                "shader-bind trace: SH register write"
            );
        }
        let marker = self.user_data_marker;
        match reg {
            r if (pm4::SPI_SHADER_USER_DATA_VS_0..pm4::SPI_SHADER_USER_DATA_VS_0 + SGPRS)
                .contains(&r) =>
            {
                let id = r - pm4::SPI_SHADER_USER_DATA_VS_0;
                self.sh_ctx.vs.vs_user_sgpr.set(id, value, marker);
            }
            r if (pm4::SPI_SHADER_USER_DATA_PS_0..pm4::SPI_SHADER_USER_DATA_PS_0 + SGPRS)
                .contains(&r) =>
            {
                let id = r - pm4::SPI_SHADER_USER_DATA_PS_0;
                tracing::debug!(
                    id,
                    value = format_args!("{value:#010x}"),
                    "PS user SGPR write"
                );
                self.sh_ctx.ps.ps_user_sgpr.set(id, value, marker);
            }
            r if (pm4::SPI_SHADER_USER_DATA_GS_0..pm4::SPI_SHADER_USER_DATA_GS_0 + SGPRS)
                .contains(&r) =>
            {
                // Kyty: hw_sh_set_gs_user_sgpr (GraphicsRun.cpp L2456) — the
                // Gen5 vertex stage runs as GS, so its user data lands here.
                let id = r - pm4::SPI_SHADER_USER_DATA_GS_0;
                self.sh_ctx.vs.gs_user_sgpr.set(id, value, marker);
            }
            r if (pm4::COMPUTE_USER_DATA_0..=pm4::COMPUTE_USER_DATA_15).contains(&r) => {
                let id = r - pm4::COMPUTE_USER_DATA_0;
                self.sh_ctx.cs.cs_user_sgpr.set(id, value, marker);
            }
            // Gen5 shader binds: plain SH-register writes (Kyty's
            // g_hw_sh_indirect_func table, GraphicsRun.cpp L3995-4100).
            // Address registers merge into the 40-bit code base the same way
            // Kyty does: LO shifts into bits 8..40, HI's low byte into 40..48.
            pm4::SPI_SHADER_PGM_LO_PS => {
                let base = self.sh_ctx.ps.ps_regs.data_addr;
                self.sh_ctx
                    .set_ps_shader_base((base & 0xFFFF_FF00_0000_00FF) | (u64::from(value) << 8));
            }
            pm4::SPI_SHADER_PGM_HI_PS => {
                let base = self.sh_ctx.ps.ps_regs.data_addr;
                self.sh_ctx.set_ps_shader_base(
                    (base & 0xFFFF_00FF_FFFF_FFFF) | (u64::from(value & 0xFF) << 40),
                );
            }
            pm4::SPI_SHADER_PGM_CHKSUM_PS => self.sh_ctx.ps.ps_regs.push_chksum(value),
            pm4::SPI_SHADER_PGM_RSRC2_PS => {
                let user_sgpr = pm4::field(value, pm4::spi_shader_pgm_rsrc2::USER_SGPR)
                    + (pm4::field(value, pm4::spi_shader_pgm_rsrc2::USER_SGPR_MSB) << 5);
                self.sh_ctx.set_ps_rsrc2_user_sgpr(user_sgpr as u8);
            }
            pm4::SPI_SHADER_PGM_LO_ES => {
                let base = self.sh_ctx.vs.es_regs.data_addr;
                self.sh_ctx
                    .set_es_shader_base((base & 0xFFFF_FF00_0000_00FF) | (u64::from(value) << 8));
            }
            pm4::SPI_SHADER_PGM_HI_ES => {
                let base = self.sh_ctx.vs.es_regs.data_addr;
                self.sh_ctx.set_es_shader_base(
                    (base & 0xFFFF_00FF_FFFF_FFFF) | (u64::from(value & 0xFF) << 40),
                );
            }
            pm4::SPI_SHADER_PGM_CHKSUM_GS => self.sh_ctx.push_gs_chksum(value),
            pm4::SPI_SHADER_PGM_RSRC2_GS => {
                let user_sgpr = pm4::field(value, pm4::spi_shader_pgm_rsrc2::USER_SGPR)
                    + (pm4::field(value, pm4::spi_shader_pgm_rsrc2::USER_SGPR_MSB) << 5);
                self.sh_ctx.set_gs_rsrc2_user_sgpr(user_sgpr as u8);
            }
            // RSRC1 carries VGPR/SGPR allocation and float-mode bits nothing
            // on the ported parse path reads; consumed knowingly, not unknown.
            pm4::SPI_SHADER_PGM_RSRC1_PS | pm4::SPI_SHADER_PGM_RSRC1_GS => {}
            pm4::COMPUTE_PGM_LO => {
                let base = self.sh_ctx.cs.cs_regs.data_addr;
                self.sh_ctx.cs.cs_regs.data_addr =
                    (base & 0xFFFF_FF00_0000_00FF) | (u64::from(value) << 8);
            }
            pm4::COMPUTE_PGM_HI => {
                let base = self.sh_ctx.cs.cs_regs.data_addr;
                self.sh_ctx.cs.cs_regs.data_addr =
                    (base & 0xFFFF_00FF_FFFF_FFFF) | (u64::from(value & 0xFF) << 40);
            }
            pm4::COMPUTE_PGM_RSRC1 => {
                self.sh_ctx.cs.cs_regs.vgprs =
                    pm4::field(value, pm4::compute_pgm_rsrc1::VGPRS) as u8;
                self.sh_ctx.cs.cs_regs.sgprs =
                    pm4::field(value, pm4::compute_pgm_rsrc1::SGPRS) as u8;
                self.sh_ctx.cs.cs_regs.bulky =
                    pm4::field(value, pm4::compute_pgm_rsrc1::BULKY) as u8;
            }
            pm4::COMPUTE_PGM_RSRC2 => {
                let regs = &mut self.sh_ctx.cs.cs_regs;
                regs.scratch_en = pm4::field(value, pm4::compute_pgm_rsrc2::SCRATCH_EN) as u8;
                regs.user_sgpr = pm4::field(value, pm4::compute_pgm_rsrc2::USER_SGPR) as u8;
                regs.tgid_x_en = pm4::field(value, pm4::compute_pgm_rsrc2::TGID_X_EN) as u8;
                regs.tgid_y_en = pm4::field(value, pm4::compute_pgm_rsrc2::TGID_Y_EN) as u8;
                regs.tgid_z_en = pm4::field(value, pm4::compute_pgm_rsrc2::TGID_Z_EN) as u8;
                regs.tg_size_en = pm4::field(value, pm4::compute_pgm_rsrc2::TG_SIZE_EN) as u8;
                regs.tidig_comp_cnt =
                    pm4::field(value, pm4::compute_pgm_rsrc2::TIDIG_COMP_CNT) as u8;
                regs.lds_size = pm4::field(value, pm4::compute_pgm_rsrc2::LDS_SIZE) as u8;
            }
            pm4::COMPUTE_NUM_THREAD_X => self.sh_ctx.cs.cs_regs.num_thread_x = value,
            pm4::COMPUTE_NUM_THREAD_Y => self.sh_ctx.cs.cs_regs.num_thread_y = value,
            pm4::COMPUTE_NUM_THREAD_Z => self.sh_ctx.cs.cs_regs.num_thread_z = value,
            // Start coordinates, checksum, and RSRC3 do not shape the ported
            // analyzer or Vulkan pipeline yet, but are consumed knowingly.
            pm4::COMPUTE_START_X
            | pm4::COMPUTE_START_Y
            | pm4::COMPUTE_START_Z
            | pm4::COMPUTE_PGM_RSRC3 => {}
            pm4::COMPUTE_SHADER_CHKSUM => self.sh_ctx.cs.cs_regs.push_chksum(value),
            _ => {
                if self.first(SkipKey::Reg(RegFile::Shader, reg))
                    && warn_skip_reg_once(RegFile::Shader, reg)
                {
                    warn!(
                        reg = format_args!("{reg:#06x}"),
                        "unknown shader register — write skipped"
                    );
                }
            }
        }
    }

    /// Kyty: `cp_op_set_uconfig_reg` (L3332). Bit 28 is the "neo" flag and is
    /// masked off the register index.
    fn cp_op_set_uconfig_reg(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
    ) -> Result<u32, CpError> {
        let reg = Self::body_at(body, 0, offset)? & 0xEFFF_FFFF;
        let values = &body[1..];
        let count = Self::reg_count(cmd_id, values, offset)?;
        for (i, &value) in values.iter().enumerate().take(count) {
            self.set_uconfig_register(reg + i as u32, value);
        }
        Ok(count as u32 + 1)
    }

    /// `IT_SET_UCONFIG_REG_INDEX`: indexed user-config writes encode the
    /// register in the low 16 bits of the first body DWORD and carry the
    /// index/control selector in its high bits. Minecraft receives this packet
    /// from KytyPS5's `GraphicsUnknownKRzWekV120`.
    fn cp_op_set_uconfig_reg_index(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
    ) -> Result<u32, CpError> {
        let reg = Self::body_at(body, 0, offset)? & 0xffff;
        let values = &body[1..];
        let count = Self::reg_count(cmd_id, values, offset)?;
        for (i, &value) in values.iter().enumerate().take(count) {
            self.set_uconfig_register(reg + i as u32, value);
        }
        Ok(count as u32 + 1)
    }

    /// Kyty: `g_hw_uc_func` / `g_hw_uc_indirect_func` — one entry
    /// (`VGT_PRIMITIVE_TYPE`).
    fn set_uconfig_register(&mut self, reg: u32, value: u32) {
        if reg as usize >= pm4::UC_NUM {
            if self.first(SkipKey::Reg(RegFile::UserConfig, reg))
                && warn_skip_reg_once(RegFile::UserConfig, reg)
            {
                warn!(
                    reg = format_args!("{reg:#06x}"),
                    "user-config register index out of range — write skipped"
                );
            }
            return;
        }
        match reg {
            pm4::VGT_PRIMITIVE_TYPE => self.ucfg.prim_type = value,
            pm4::VGT_INDEX_TYPE => self.index_type_and_size = value,
            _ => {
                if self.first(SkipKey::Reg(RegFile::UserConfig, reg))
                    && warn_skip_reg_once(RegFile::UserConfig, reg)
                {
                    warn!(
                        reg = format_args!("{reg:#06x}"),
                        "unknown user-config register — write skipped"
                    );
                }
            }
        }
    }

    fn body_at(body: &[u32], idx: usize, offset: u32) -> Result<u32, CpError> {
        body.get(idx).copied().ok_or(CpError::Truncated {
            offset,
            need: idx as u32 + 2,
            remaining: body.len() as u32 + 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pm4::{ItOp, RCode, header};

    #[derive(Default)]
    struct RecordingSink {
        draws: Vec<(u32, u32, u32, bool, bool)>,
        dispatches: Vec<RecordedDispatch>,
        guest_memory_write_boundaries: u32,
        fail: Option<String>,
    }

    /// (group_xyz, unused, direct_address, dims, mode, tag) recorded per dispatch.
    type RecordedDispatch = ([u32; 3], u32, u64, [u32; 3], u8, u32);

    impl DrawSink for RecordingSink {
        fn guest_memory_write_boundary(&mut self) {
            self.guest_memory_write_boundaries += 1;
        }

        fn draw_index_auto(
            &mut self,
            _ctx: &Context,
            ucfg: &UserConfig,
            sh: &Shader,
            index_count: u32,
            flags: u32,
        ) -> Result<(), DrawError> {
            if let Some(m) = &self.fail {
                return Err(DrawError(m.clone()));
            }
            self.draws.push((
                index_count,
                flags,
                ucfg.prim_type,
                sh.vs.vs_embedded,
                sh.ps.ps_embedded,
            ));
            Ok(())
        }

        fn dispatch_direct(
            &mut self,
            _ctx: &Context,
            _ucfg: &UserConfig,
            sh: &Shader,
            groups: [u32; 3],
            mode: u32,
        ) -> Result<(), DrawError> {
            self.dispatches.push((
                groups,
                mode,
                sh.cs.cs_regs.data_addr,
                [
                    sh.cs.cs_regs.num_thread_x,
                    sh.cs.cs_regs.num_thread_y,
                    sh.cs.cs_regs.num_thread_z,
                ],
                sh.cs.cs_regs.user_sgpr,
                sh.cs.cs_user_sgpr.value[3],
            ));
            Ok(())
        }
    }

    #[test]
    fn cond_exec_skips_the_guarded_dwords_when_the_label_is_zero() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let label_addr = 0x1000u64;
        let mem = BufMem {
            base: label_addr,
            words: vec![0],
        };
        let dcb = vec![
            header(5, pm4::IT_COND_EXEC, pm4::R_ZERO),
            label_addr as u32,
            (label_addr >> 32) as u32,
            0,
            3,
            header(3, pm4::IT_DRAW_INDEX_AUTO, pm4::R_ZERO),
            111,
            0,
            header(3, pm4::IT_DRAW_INDEX_AUTO, pm4::R_ZERO),
            222,
            0,
        ];

        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("conditional stream must execute");
        assert_eq!(
            sink.draws.iter().map(|draw| draw.0).collect::<Vec<_>>(),
            vec![222],
            "zero label must skip exactly the guarded command"
        );

        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mem = BufMem {
            base: label_addr,
            words: vec![1],
        };
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("non-zero conditional stream must execute");
        assert_eq!(
            sink.draws.iter().map(|draw| draw.0).collect::<Vec<_>>(),
            vec![111, 222],
            "non-zero label must execute the guarded command"
        );
    }

    /// A sink that records the full [`IndexedDraw`], overriding the default
    /// degradation.
    #[derive(Default)]
    struct IndexedSink {
        auto_draws: Vec<u32>,
        indexed: Vec<IndexedDraw>,
    }

    impl DrawSink for IndexedSink {
        fn draw_index_auto(
            &mut self,
            _ctx: &Context,
            _ucfg: &UserConfig,
            _sh: &Shader,
            index_count: u32,
            _flags: u32,
        ) -> Result<(), DrawError> {
            self.auto_draws.push(index_count);
            Ok(())
        }
        fn draw_index(
            &mut self,
            _ctx: &Context,
            _ucfg: &UserConfig,
            _sh: &Shader,
            draw: &IndexedDraw,
        ) -> Result<(), DrawError> {
            self.indexed.push(*draw);
            Ok(())
        }
    }

    /// The opcode Minecraft draws with must reach the sink.
    ///
    /// It was absent from the dispatch table, so it fell to the default arm —
    /// warn once, skip by encoded length, forever. Measured on the title: 24,224
    /// draw packets decoded, ~64 executed. A regression here is not a missing
    /// feature, it is ~99.6% of a game's draws vanishing after one log line.
    ///
    /// `cmd_id` is the exact header captured from Minecraft: `0xc0033500`.
    #[test]
    fn draw_index_offset_2_reaches_the_sink_and_is_not_skipped() {
        let mut cp = CommandProcessor::new();
        let mut sink = IndexedSink {
            auto_draws: Vec::new(),
            indexed: Vec::new(),
        };
        // { MAX_SIZE, INDEX_OFFSET, INDEX_COUNT, DRAW_INITIATOR }
        let dcb = vec![0xc003_3500, 4096, 6, 300, 0];
        cp.run(&dcb, &mut sink).expect("packet must not fault");

        assert_eq!(
            sink.indexed.len(),
            1,
            "IT_DRAW_INDEX_OFFSET_2 was skipped instead of drawn"
        );
        assert_eq!(sink.indexed[0].index_count, 300, "INDEX_COUNT is body[2]");
        assert_eq!(
            cp.distinct_skips(),
            0,
            "the opcode must be handled, not skipped-with-a-warn"
        );
    }

    /// The packet carries no address: indices come from `IT_INDEX_BASE`, offset
    /// by `INDEX_OFFSET` *elements* — so the element size has to be right or the
    /// draw reads from the wrong place in the buffer.
    #[test]
    fn draw_index_offset_2_offsets_from_the_bound_index_base() {
        for (index_type, bytes) in [(0u32, 2u64), (1, 4), (2, 1)] {
            let mut cp = CommandProcessor::new();
            let mut sink = IndexedSink {
                auto_draws: Vec::new(),
                indexed: Vec::new(),
            };
            let dcb = vec![
                // IT_INDEX_BASE { lo, hi } = 0x1_0000_0000
                pm4::header(3, pm4::IT_INDEX_BASE, pm4::R_ZERO),
                0x0000_0000,
                0x0000_0001,
                // IT_INDEX_TYPE { type }
                pm4::header(2, pm4::IT_INDEX_TYPE, pm4::R_ZERO),
                index_type,
                // DRAW_INDEX_OFFSET_2 { MAX_SIZE, INDEX_OFFSET=10, INDEX_COUNT, INITIATOR }
                0xc003_3500,
                4096,
                10,
                300,
                0,
            ];
            cp.run(&dcb, &mut sink).expect("packets must not fault");
            assert_eq!(sink.indexed.len(), 1);
            assert_eq!(
                sink.indexed[0].index_addr,
                0x1_0000_0000 + 10 * bytes,
                "index_type {index_type} must offset by {bytes} bytes per index"
            );
        }
    }

    /// Guest memory backed by a plain buffer at a fixed base address.
    struct BufMem {
        base: u64,
        words: Vec<u32>,
    }

    impl GuestMemory for BufMem {
        fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
            let rel = addr.checked_sub(self.base)?;
            if rel % 4 != 0 {
                return None;
            }
            let start = usize::try_from(rel / 4).ok()?;
            let end = start.checked_add(count as usize)?;
            self.words.get(start..end).map(<[u32]>::to_vec)
        }
    }

    /// Body dwords the AGC embedded/draw packets declare, as padding.
    fn pad(n: usize) -> Vec<u32> {
        vec![0; n]
    }

    /// The AGC draw packet plus the register state that makes it meaningful:
    /// prim type 17 then `R_DRAW_INDEX_AUTO` with count 3.
    fn state_and_draw() -> Vec<u32> {
        let mut dcb = vec![
            header(3, pm4::IT_SET_UCONFIG_REG, pm4::R_ZERO),
            pm4::VGT_PRIMITIVE_TYPE,
            17,
        ];
        dcb.extend_from_slice(&[header(7, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO), 3, 0]);
        dcb.extend(pad(4));
        dcb
    }

    #[test]
    fn r_ps_embedded_sets_flag_and_id() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = vec![header(40, pm4::IT_NOP, pm4::R_PS_EMBEDDED), 0];
        dcb.extend(pad(38));
        cp.run(&dcb, &mut sink).expect("embedded PS packet");
        assert!(cp.get_sh_ctx().ps.ps_embedded);
        assert_eq!(cp.get_sh_ctx().ps.ps_embedded_id, 0);
    }

    /// One `IT_SET_SH_REG` packet: `reg` then `values`.
    fn set_sh(reg: u32, values: &[u32]) -> Vec<u32> {
        let mut dcb = vec![
            header((values.len() + 2) as u16, pm4::IT_SET_SH_REG, pm4::R_ZERO),
            reg,
        ];
        dcb.extend_from_slice(values);
        dcb
    }

    /// Gen5 binds the PS stage by writing `SPI_SHADER_PGM_LO/HI_PS` +
    /// `CHKSUM_PS` + `RSRC2_PS` as plain SH registers (Kyty's
    /// `g_hw_sh_indirect_func`). The CP must compose the 40-bit code address,
    /// accumulate the checksum across both writes, decode `USER_SGPR` with its
    /// MSB extension, and clear the embedded flag.
    #[test]
    fn sh_pgm_ps_registers_bind_a_real_pixel_shader() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = vec![header(40, pm4::IT_NOP, pm4::R_PS_EMBEDDED), 0];
        dcb.extend(pad(38)); // embedded first, so the bind must *clear* it
        dcb.extend(set_sh(pm4::SPI_SHADER_PGM_LO_PS, &[0x00C0_FFEE]));
        dcb.extend(set_sh(pm4::SPI_SHADER_PGM_HI_PS, &[0x12])); // low byte only
        dcb.extend(set_sh(pm4::SPI_SHADER_PGM_CHKSUM_PS, &[0xAAAA_0001]));
        dcb.extend(set_sh(pm4::SPI_SHADER_PGM_CHKSUM_PS, &[0xBBBB_0002]));
        // USER_SGPR = 0x1F at bit 1, MSB at bit 27 -> 0x1F + 0x20 = 0x3F.
        dcb.extend(set_sh(
            pm4::SPI_SHADER_PGM_RSRC2_PS,
            &[(0x1F << 1) | (1 << 27)],
        ));
        cp.run(&dcb, &mut sink).expect("SH shader-bind packets");

        let ps = &cp.get_sh_ctx().ps;
        assert!(!ps.ps_embedded, "a real bind clears the embedded flag");
        assert_eq!(
            ps.ps_regs.data_addr,
            (0x00C0_FFEEu64 << 8) | (0x12u64 << 40)
        );
        assert_eq!(ps.ps_regs.chksum, 0xAAAA_0001_BBBB_0002);
        assert_eq!(ps.ps_regs.rsrc2.user_sgpr, 0x3F);
    }

    /// The Gen5 vertex stage rides the ES/GS registers: `PGM_LO/HI_ES` set the
    /// code base, `CHKSUM_GS` carries the identity `shader_parse_vs` reads in
    /// next-gen mode, `RSRC2_GS` the user-SGPR count, and user data lands in
    /// the GS slots. Together they form exactly the "gs instead of vs" state.
    #[test]
    fn sh_pgm_es_gs_registers_bind_a_real_vertex_stage() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = vec![header(29, pm4::IT_NOP, pm4::R_VS_EMBEDDED), 0, 0];
        dcb.extend(pad(26));
        dcb.extend(set_sh(pm4::SPI_SHADER_PGM_LO_ES, &[0x0000_1000]));
        dcb.extend(set_sh(pm4::SPI_SHADER_PGM_HI_ES, &[0x00]));
        dcb.extend(set_sh(pm4::SPI_SHADER_PGM_CHKSUM_GS, &[0x1234_5678]));
        dcb.extend(set_sh(pm4::SPI_SHADER_PGM_CHKSUM_GS, &[0x9ABC_DEF0]));
        dcb.extend(set_sh(pm4::SPI_SHADER_PGM_RSRC2_GS, &[0x4 << 1]));
        dcb.extend(set_sh(pm4::SPI_SHADER_USER_DATA_GS_0 + 2, &[0xFEED]));
        cp.run(&dcb, &mut sink).expect("ES/GS shader-bind packets");

        let vs = &cp.get_sh_ctx().vs;
        assert!(!vs.vs_embedded, "a real bind clears the embedded flag");
        assert_eq!(
            vs.vs_regs.data_addr, 0,
            "vs base stays 0 (gs-instead-of-vs)"
        );
        assert_eq!(vs.es_regs.data_addr, 0x0000_1000u64 << 8);
        assert_eq!(vs.gs_regs.chksum, 0x1234_5678_9ABC_DEF0);
        assert_eq!(vs.gs_regs.rsrc2.user_sgpr, 4);
        assert_eq!(vs.gs_user_sgpr.value[2], 0xFEED);
        assert_eq!(vs.gs_user_sgpr.count, 3, "high-water mark from slot 2");
    }

    #[test]
    fn r_vs_embedded_reads_modifier_then_id() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = vec![header(29, pm4::IT_NOP, pm4::R_VS_EMBEDDED), 0xAA, 7];
        dcb.extend(pad(26));
        cp.run(&dcb, &mut sink).expect("embedded VS packet");
        assert!(cp.get_sh_ctx().vs.vs_embedded);
        assert_eq!(cp.get_sh_ctx().vs.vs_embedded_id, 7, "id is body[1]");
    }

    #[test]
    fn set_context_reg_writes_render_target_attrib2() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::CB_COLOR0_ATTRIB2,
            (95 << 14) | 47,
        ];
        cp.run(&dcb, &mut sink).expect("attrib2 write");
        let rt = &cp.get_ctx().render_targets[0];
        assert_eq!(rt.attrib2.width, 95);
        assert_eq!(rt.attrib2.height, 47);
    }

    #[test]
    fn set_context_reg_writes_color_base_shifted_by_8() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::CB_COLOR0_BASE,
            0x1_0000 >> 8,
        ];
        cp.run(&dcb, &mut sink).expect("base write");
        assert_eq!(cp.get_ctx().render_targets[0].base.addr, 0x1_0000);
    }

    #[test]
    fn set_context_reg_decodes_screen_scissor_tl_and_br() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(4, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::PA_SC_SCREEN_SCISSOR_TL,
            0,
            48 | (48 << 16),
        ];
        cp.run(&dcb, &mut sink).expect("scissor write");
        let vp = &cp.get_ctx().screen_viewport;
        assert_eq!(
            (
                vp.screen_scissor_left,
                vp.screen_scissor_top,
                vp.screen_scissor_right,
                vp.screen_scissor_bottom
            ),
            (0, 0, 48, 48)
        );
    }

    #[test]
    fn set_uconfig_reg_strips_neo_bit_and_writes_prim_type() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(3, pm4::IT_SET_UCONFIG_REG, pm4::R_ZERO),
            pm4::VGT_PRIMITIVE_TYPE | 0x1000_0000,
            17,
        ];
        cp.run(&dcb, &mut sink).expect("uconfig write");
        assert_eq!(cp.get_ucfg().prim_type, 17);
    }

    #[test]
    fn set_sh_reg_writes_vs_user_sgpr_high_water_mark() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(3, pm4::IT_SET_SH_REG, pm4::R_ZERO),
            pm4::SPI_SHADER_USER_DATA_VS_0 + 3,
            0xDEAD,
        ];
        cp.run(&dcb, &mut sink).expect("sh write");
        let sgpr = &cp.get_sh_ctx().vs.vs_user_sgpr;
        assert_eq!(sgpr.count, 4);
        assert_eq!(sgpr.value[3], 0xDEAD);
    }

    /// Kyty bakes packet length into an exact-cmd_id assert and would abort
    /// here; a batch must write every register in it.
    ///
    /// `ATTRIB2` has slot stride 1, so consecutive registers are consecutive
    /// *slots* — `CB_COLOR0_ATTRIB2 + 1` is colour slot 1, not `ATTRIB3`.
    #[test]
    fn set_context_reg_writes_a_multi_register_batch() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(4, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::CB_COLOR0_ATTRIB2,
            (95 << 14) | 47,
            (31 << 14) | 15,
        ];
        cp.run(&dcb, &mut sink).expect("batched write");
        let rts = &cp.get_ctx().render_targets;
        assert_eq!((rts[0].attrib2.width, rts[0].attrib2.height), (95, 47));
        assert_eq!(
            (rts[1].attrib2.width, rts[1].attrib2.height),
            (31, 15),
            "the second register in the batch must land in slot 1"
        );
    }

    #[test]
    fn r_draw_index_auto_invokes_sink_with_prim_type_and_count() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        cp.run(&state_and_draw(), &mut sink).expect("draw");
        assert_eq!(sink.draws, [(3, 0, 17, false, false)]);
    }

    #[test]
    fn r_cs_and_dispatch_direct_deliver_compute_state_to_sink() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();

        let rsrc1 = 5 | (7 << 6) | (1 << 24);
        let rsrc2 = (9 << 1) | (1 << 7) | (1 << 9) | (2 << 11) | (3 << 15);
        let mut dcb = vec![
            header(25, pm4::IT_NOP, pm4::R_CS),
            0x1234,
            0x2345_6789,
            0x12,
            rsrc1,
            rsrc2,
            8,
            4,
            2,
        ];
        dcb.extend(pad(16));
        dcb.extend_from_slice(&[
            header(3, pm4::IT_SET_SH_REG, pm4::R_ZERO),
            pm4::COMPUTE_USER_DATA_0 + 3,
            0xCAFE_BABE,
            header(9, pm4::IT_NOP, pm4::R_DISPATCH_DIRECT),
            11,
            12,
            13,
            0,
        ]);
        dcb.extend(pad(4));

        cp.run(&dcb, &mut sink).expect("compute dispatch");
        assert_eq!(
            sink.dispatches,
            [([11, 12, 13], 0, 0x1223_4567_8900, [8, 4, 2], 9, 0xCAFE_BABE,)]
        );
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.vgprs, 5);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.sgprs, 7);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.bulky, 1);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.tgid_x_en, 1);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.tgid_z_en, 1);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.tidig_comp_cnt, 2);
    }

    /// The sink call site still names the refusal ([`CpError::Draw`]) for a
    /// direct (non-walk) caller — never-silent for draws is preserved at the
    /// packet handler.
    #[test]
    fn draw_error_from_sink_is_named_at_the_handler() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink {
            fail: Some("no bound render target".into()),
            ..Default::default()
        };
        // Drive the draw handler directly (bypassing the walk's skip-and-continue
        // policy) to prove the refusal is still surfaced as a named CpError::Draw.
        let err = cp
            .cp_op_draw_index_auto(
                header(3, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO),
                &[3, 0],
                0,
                &mut sink,
            )
            .expect_err("sink refused the draw");
        match err {
            CpError::Draw { source, .. } => assert!(source.0.contains("render target")),
            other => panic!("expected a named draw fault, got {other:?}"),
        }
    }

    /// FIX 1 invariant (the Minecraft async-compute unblock): a sink that
    /// REFUSES a draw/dispatch must not abort the command-buffer walk. Every
    /// packet AFTER the refusal — the completion labels/fences and the later
    /// dispatches whose writebacks the guest polls on for "GPU done" — must
    /// still run, or the title's async-compute submit worker hangs forever on a
    /// completion that never arrives (measured: a refused ACB dispatch wedged
    /// the whole game ~0.7 s later on a held mutex).
    #[test]
    fn refused_draw_is_skipped_so_the_walk_continues_to_completion_packets() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink {
            fail: Some("no bound render target".into()),
            ..Default::default()
        };
        // [refused draw][NUM_INSTANCES = 5]. The register write stands in for the
        // completion packets that follow a real dispatch: if the walk aborted at
        // the refusal (the old behaviour that deadlocked Minecraft), it would
        // never execute.
        let dcb = vec![
            header(3, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO),
            3, // index_count
            0, // flags
            header(2, pm4::IT_NUM_INSTANCES, pm4::R_ZERO),
            5, // num instances
        ];
        cp.run(&dcb, &mut sink)
            .expect("a refused draw must be skipped, not abort the walk");
        assert_eq!(
            cp.num_instances(),
            5,
            "the packet after a refused draw must still execute (completion invariant)"
        );
        assert_eq!(cp.refused_draws(), 1, "the refusal must be counted");
    }

    /// Kyty's `dw -= s + 1` wraps on an over-long packet and reads past the end
    /// before noticing. This test fails against a verbatim transliteration.
    #[test]
    fn truncated_packet_errors_instead_of_wrapping() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        // COUNT claims 8 total dwords; only 2 are present.
        let dcb = vec![header(8, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO), 0];
        let err = cp.run(&dcb, &mut sink).expect_err("must not overrun");
        assert!(matches!(err, CpError::Truncated { .. }), "got {err:?}");
    }

    /// PA_SU_SC_MODE_CNTL decodes into `ctx.mode_control`. This register was
    /// previously undecoded while its struct and its Vulkan cull-mode consumer
    /// both existed, so every draw rasterized with culling disabled. Bit
    /// positions are Kyty's (Pm4.h L489-510); each field is given a DISTINCT
    /// value so a copy-paste shift error cannot pass.
    #[test]
    fn pa_su_sc_mode_cntl_decodes_every_field() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        // cull_front=1, cull_back=0, face=1, poly_mode=2, front_ptype=5,
        // back_ptype=3, offset_front=1, offset_back=0, vtx_window=1,
        // provoking_last=1, persp_corr_dis=0 (zero-valued fields omitted).
        let value =
            1 | (1 << 2) | (2 << 3) | (5 << 5) | (3 << 8) | (1 << 11) | (1 << 16) | (1 << 19);
        let dcb = vec![
            header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::PA_SU_SC_MODE_CNTL,
            value,
        ];
        cp.run(&dcb, &mut sink).expect("mode cntl decodes");
        let m = cp.get_ctx().mode_control;
        assert!(m.cull_front, "cull_front is bit 0");
        assert!(!m.cull_back, "cull_back is bit 1");
        assert!(m.face, "face is bit 2");
        assert_eq!(m.poly_mode, 2, "poly_mode is bits 4:3");
        assert_eq!(m.polymode_front_ptype, 5, "front ptype is bits 7:5");
        assert_eq!(m.polymode_back_ptype, 3, "back ptype is bits 10:8");
        assert!(m.poly_offset_front_enable, "bit 11");
        assert!(!m.poly_offset_back_enable, "bit 12");
        assert!(m.vtx_window_offset_enable, "bit 16");
        assert!(m.provoking_vtx_last, "bit 19");
        assert!(!m.persp_corr_dis, "bit 20");
    }

    /// Resilience policy: an unknown register is a rate-limited warn and a
    /// skipped write — and the rest of the batch still lands.
    #[test]
    fn unknown_register_is_skipped_and_the_batch_continues() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        // Two-register batch: 0x8D is unknown, 0x8E is CB_TARGET_MASK.
        let dcb = vec![
            header(4, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::CB_TARGET_MASK - 1,
            0xAAAA,
            0xF,
        ];
        cp.run(&dcb, &mut sink).expect("unknown register must skip");
        assert_eq!(
            cp.get_ctx().render_target_mask,
            0xF,
            "the known register after the unknown one must still be written"
        );
        assert_eq!(cp.distinct_skips(), 1);
    }

    /// The priority-1 resilience test: a DCB of [unknown op, then a valid
    /// DRAW_INDEX_AUTO with render state] still produces the draw.
    #[test]
    fn unknown_opcode_is_skipped_and_the_draw_still_lands() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = vec![header(3, ItOp(0xEE), pm4::R_ZERO), 0, 0];
        dcb.extend(state_and_draw());
        cp.run(&dcb, &mut sink)
            .expect("an unknown op must not kill the DCB");
        assert_eq!(sink.draws, [(3, 0, 17, false, false)]);
        assert_eq!(cp.distinct_skips(), 1);
    }

    #[test]
    fn unknown_custom_op_is_skipped_and_the_draw_still_lands() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        // RCode 0x3F is unassigned in Kyty's table.
        let mut dcb = vec![header(4, pm4::IT_NOP, RCode(0x3F)), 0, 0, 0];
        dcb.extend(state_and_draw());
        cp.run(&dcb, &mut sink)
            .expect("an unknown custom op must not kill the DCB");
        assert_eq!(sink.draws.len(), 1);
    }

    /// Byte-addressed read/write test memory for DMA_DATA copies; dwords are
    /// read byte-accurately so the same fixture also backs wait labels.
    struct DmaMem {
        base: u64,
        bytes: std::cell::RefCell<Vec<u8>>,
    }

    impl DmaMem {
        fn range(&self, addr: u64, len: u64) -> Option<std::ops::Range<usize>> {
            let start = usize::try_from(addr.checked_sub(self.base)?).ok()?;
            let end = start.checked_add(usize::try_from(len).ok()?)?;
            (end <= self.bytes.borrow().len()).then_some(start..end)
        }
    }

    impl GuestMemory for DmaMem {
        fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
            let range = self.range(addr, u64::from(count) * 4)?;
            Some(
                self.bytes.borrow()[range]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
                    .collect(),
            )
        }

        fn read_bytes(&self, addr: u64, len: u64) -> Option<Vec<u8>> {
            let range = self.range(addr, len)?;
            Some(self.bytes.borrow()[range].to_vec())
        }

        fn write_bytes(&self, addr: u64, data: &[u8]) -> bool {
            match self.range(addr, data.len() as u64) {
                Some(range) => {
                    self.bytes.borrow_mut()[range].copy_from_slice(data);
                    true
                }
                None => false,
            }
        }
    }

    /// Both DMA_DATA builder layouts execute a real memory→memory copy, and a
    /// non-memory selector skips without touching the destination.
    #[test]
    fn dma_data_executes_memory_copies_in_both_layouts() {
        let mem = DmaMem {
            base: 0x9000,
            bytes: std::cell::RefCell::new((0u8..192).collect()),
        };
        // src = base (bytes 0..16), dst regions initially hold 64.. and 128..
        let (src, dst_a, dst_b) = (0x9000u64, 0x9040u64, 0x9080u64);

        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            // 8-dw DCB form: control0 (mem→mem), control_ext, byte_count,
            // dst lo/hi, src lo/hi.
            header(8, pm4::IT_NOP, pm4::R_DMA_DATA),
            0,
            0,
            16,
            dst_a as u32,
            (dst_a >> 32) as u32,
            src as u32,
            (src >> 32) as u32,
            // 7-dw ACB form: dst lo/hi, src lo/hi, byte_count, sel (mem→mem).
            header(7, pm4::IT_NOP, pm4::R_DMA_DATA),
            dst_b as u32,
            (dst_b >> 32) as u32,
            src as u32,
            (src >> 32) as u32,
            16,
            0,
            // ACB form again but srcSel=2 (immediate/GDS): must be skipped.
            header(7, pm4::IT_NOP, pm4::R_DMA_DATA),
            0x9060,
            0,
            src as u32,
            (src >> 32) as u32,
            16,
            2,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("DMA packets must not kill the DCB");

        let bytes = mem.bytes.borrow();
        let pattern: Vec<u8> = (0u8..16).collect();
        assert_eq!(&bytes[0x40..0x50], &pattern[..], "DCB-form copy landed");
        assert_eq!(&bytes[0x80..0x90], &pattern[..], "ACB-form copy landed");
        assert_eq!(
            bytes[0x60..0x70],
            (96u8..112).collect::<Vec<u8>>()[..],
            "non-memory selector must not write"
        );
        assert_eq!(
            sink.guest_memory_write_boundaries, 3,
            "every potentially writing packet conservatively invalidates sink caches"
        );
    }

    /// Without a GuestMemory accessor a DMA_DATA packet is skipped (one warn),
    /// never a stream error — read-only embedders keep working.
    #[test]
    fn dma_data_without_memory_is_skipped() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = vec![
            header(7, pm4::IT_NOP, pm4::R_DMA_DATA),
            0x9040,
            0,
            0x9000,
            0,
            16,
            0,
        ];
        dcb.extend(state_and_draw());
        cp.run(&dcb, &mut sink)
            .expect("DMA without memory must not kill the DCB");
        assert_eq!(sink.draws.len(), 1, "the stream continues to the draw");
    }

    /// The standard `IT_DMA_DATA` layout executes its guest-memory copy and
    /// 32-bit pattern fill in-stream; an unmodeled selector form (GDS dst) is
    /// consumed without effect.
    #[test]
    fn it_dma_data_executes_copy_and_fill_in_stream() {
        let mem = DmaMem {
            base: 0x9000,
            bytes: std::cell::RefCell::new((0u8..192).collect()),
        };
        // src = base (bytes 0..16); dst regions initially hold 64.., 96.., 128..
        let (src, dst_copy, dst_fill, dst_gds) = (0x9000u64, 0x9040u64, 0x9080u64, 0x9060u64);

        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            // Copy, mem→mem (src_sel 0, dst_sel 0): both selectors zero.
            header(7, pm4::IT_DMA_DATA, pm4::R_ZERO),
            0,
            src as u32,
            (src >> 32) as u32,
            dst_copy as u32,
            (dst_copy >> 32) as u32,
            16, // command: num_bytes = 16
            // Fill (src_sel 2, dst_sel 3 = MemoryUsingL2): src_lo is the pattern.
            header(7, pm4::IT_DMA_DATA, pm4::R_ZERO),
            (2 << 29) | (3 << 20),
            0xABAB_ABAB,
            0,
            dst_fill as u32,
            (dst_fill >> 32) as u32,
            16,
            // GDS destination (dst_sel 1): not modeled — consumed silently.
            header(7, pm4::IT_DMA_DATA, pm4::R_ZERO),
            1 << 20,
            src as u32,
            (src >> 32) as u32,
            dst_gds as u32,
            (dst_gds >> 32) as u32,
            16,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("DMA packets must not kill the DCB");

        let bytes = mem.bytes.borrow();
        let pattern: Vec<u8> = (0u8..16).collect();
        assert_eq!(&bytes[0x40..0x50], &pattern[..], "standard copy landed");
        assert_eq!(
            &bytes[0x80..0x90],
            &[0xABu8; 16][..],
            "standard fill landed"
        );
        assert_eq!(
            bytes[0x60..0x70],
            (96u8..112).collect::<Vec<u8>>()[..],
            "GDS form must not write"
        );
    }

    /// Without a GuestMemory accessor a standard `IT_DMA_DATA` packet is
    /// skipped (one warn), never a stream error — read-only embedders keep
    /// working.
    #[test]
    fn it_dma_data_without_memory_is_skipped() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = vec![
            header(7, pm4::IT_DMA_DATA, pm4::R_ZERO),
            0,
            0x9000,
            0,
            0x9040,
            0,
            16,
        ];
        dcb.extend(state_and_draw());
        cp.run(&dcb, &mut sink)
            .expect("DMA without memory must not kill the DCB");
        assert_eq!(sink.draws.len(), 1, "the stream continues to the draw");
    }

    /// In-order proof: a standard `IT_DMA_DATA` copy lands before the packets
    /// after it evaluate. One stream stages a label with WRITE_DATA, DMA-copies
    /// it onto the address a later WAIT_MEM_32 polls, then draws — the walk
    /// completes with no suspend. The control stream without the DMA suspends
    /// on the same wait, proving the DMA'd data is what satisfied it.
    #[test]
    fn it_dma_data_runs_in_pm4_order_before_a_later_wait() {
        let mem = DmaMem {
            base: 0x9000,
            bytes: std::cell::RefCell::new(vec![0u8; 0x200]),
        };
        let (label, staging) = (0x9000u64, 0x9100u64);

        // Control: the wait on label == 0x2A with the label never written
        // suspends — the wait genuinely gates on memory contents.
        let mut control = wait32(label, !0, 3, 0x2A);
        control.extend(state_and_draw());
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let outcome = cp
            .run_resumable(&control, 0, &mut sink, Some(&mem))
            .expect("control wait must not fault");
        assert!(
            matches!(outcome, RunOutcome::Suspended(_)),
            "unmet wait suspends"
        );
        assert!(sink.draws.is_empty(), "work behind the wait must not run");

        // Full stream: WRITE_DATA stages 0x2A, the DMA copies it onto the
        // label, the wait then observes it and the draw behind the wait runs.
        let mut dcb = write_data_agc(staging, 0x2A);
        dcb.extend([
            header(7, pm4::IT_DMA_DATA, pm4::R_ZERO),
            0, // mem→mem
            staging as u32,
            (staging >> 32) as u32,
            label as u32,
            (label >> 32) as u32,
            4, // command: num_bytes = 4
        ]);
        dcb.extend(wait32(label, !0, 3, 0x2A));
        dcb.extend(state_and_draw());

        let mut cp = CommandProcessor::new();
        let outcome = cp
            .run_resumable(&dcb, 0, &mut sink, Some(&mem))
            .expect("in-order stream must not fault");
        assert_eq!(
            outcome,
            RunOutcome::Completed,
            "the DMA'd label satisfied the wait in-stream"
        );
        assert_eq!(sink.draws.len(), 1, "work behind the wait ran");
    }

    /// Once per distinct op per instance: the same unknown op twice warns once
    /// and both packets are skipped.
    #[test]
    fn unknown_op_warns_once_per_distinct_op() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(2, ItOp(0xEE), pm4::R_ZERO),
            0,
            header(2, ItOp(0xEE), pm4::R_ZERO),
            0,
            header(2, ItOp(0xEF), pm4::R_ZERO),
            0,
        ];
        cp.run(&dcb, &mut sink).expect("skips");
        assert_eq!(
            cp.distinct_skips(),
            2,
            "two distinct ops => two rate-limit keys, repeats coalesce"
        );
    }

    /// An unknown op whose declared length runs past the buffer is still a
    /// hard structural error — skip-by-length must not read past the end.
    #[test]
    fn unknown_op_with_overlong_length_is_still_truncated() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![header(16, ItOp(0xEE), pm4::R_ZERO), 0];
        let err = cp.run(&dcb, &mut sink).expect_err("must not overrun");
        assert!(matches!(err, CpError::Truncated { .. }), "got {err:?}");
    }

    #[test]
    fn type2_filler_is_skipped_and_type0_is_rejected() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        // A lone type-2 filler is legal and advances one dword.
        cp.run(&[0x8000_0000], &mut sink).expect("type-2 filler");
        // Kyty would misparse a type-0 header as type-3.
        match cp.run(&[0x0000_0000, 0], &mut sink) {
            Err(CpError::NotType3 { .. }) => {}
            other => panic!("expected NotType3, got {other:?}"),
        }
    }

    #[test]
    fn viewport_scale_offset_registers_decode_as_floats() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(7, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::PA_CL_VPORT_XSCALE,
            48.0f32.to_bits(),
            48.0f32.to_bits(),
            24.0f32.to_bits(),
            24.0f32.to_bits(),
            1.0f32.to_bits(),
        ];
        cp.run(&dcb, &mut sink).expect("viewport write");
        let vp = &cp.get_ctx().screen_viewport.viewports[0];
        assert_eq!((vp.xscale, vp.xoffset), (48.0, 48.0));
        assert_eq!((vp.yscale, vp.yoffset), (24.0, 24.0));
        assert_eq!(vp.zscale, 1.0);
    }

    #[test]
    fn new_starts_with_one_instance() {
        assert_eq!(CommandProcessor::new().num_instances(), 1);
    }

    // ---- index state -----------------------------------------------------

    /// Gen5 `GraphicsDcbSetIndexSize` emits `IT_INDEX_TYPE` (Graphics.cpp
    /// L1949); the CP latches it like Kyty's `SetIndexType`.
    #[test]
    fn index_type_base_and_size_are_tracked() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(2, pm4::IT_INDEX_TYPE, pm4::R_ZERO),
            2, // 32-bit indices
            header(3, pm4::IT_INDEX_BASE, pm4::R_ZERO),
            0x5000,
            0x1,
            header(2, pm4::IT_INDEX_BUFFER_SIZE, pm4::R_ZERO),
            600,
        ];
        cp.run(&dcb, &mut sink).expect("index state");
        assert_eq!(cp.index_type_and_size(), 2);
        assert_eq!(cp.index_base(), 0x1_0000_5000);
        assert_eq!(cp.index_buffer_size(), 600);
    }

    #[test]
    fn indexed_uconfig_packet_updates_vgt_index_type() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(3, pm4::IT_SET_UCONFIG_REG_INDEX, pm4::R_ZERO),
            0x2000_0000 | pm4::VGT_INDEX_TYPE,
            0x4483,
        ];
        cp.run(&dcb, &mut sink)
            .expect("indexed user-config register write");
        assert_eq!(cp.index_type_and_size(), 0x4483);
        assert_eq!(
            cp.distinct_skips(),
            0,
            "the packet must not degrade to skip"
        );
    }

    /// Kyty: `cp_op_draw_reset` → `CommandProcessor::Reset`. Register and
    /// index state clear; the warn rate-limit survives.
    #[test]
    fn r_draw_reset_clears_state_but_keeps_the_rate_limit() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(2, pm4::IT_INDEX_TYPE, pm4::R_ZERO),
            2,
            header(2, ItOp(0xEE), pm4::R_ZERO), // arm the rate limit
            0,
            header(2, pm4::IT_NOP, pm4::R_DRAW_RESET),
            0,
        ];
        cp.run(&dcb, &mut sink).expect("reset packet");
        assert_eq!(cp.index_type_and_size(), 0, "reset must clear index state");
        assert_eq!(cp.num_instances(), 1);
        assert_eq!(
            cp.distinct_skips(),
            1,
            "the rate limit must survive a reset"
        );
    }

    // ---- indexed draws ----------------------------------------------------

    /// AGC form (Kyty cmd 0xC008100C): `[count, addr_lo, addr_hi, flags,
    /// type]`. A sink that implements `draw_index` sees the full parameters.
    #[test]
    fn r_draw_index_delivers_the_full_indexed_draw() {
        let mut cp = CommandProcessor::new();
        let mut sink = IndexedSink::default();
        let mut dcb = vec![header(2, pm4::IT_INDEX_TYPE, pm4::R_ZERO), 2];
        dcb.extend_from_slice(&[
            header(10, pm4::IT_NOP, pm4::R_DRAW_INDEX),
            6,      // index_count
            0x5000, // addr lo
            0x1,    // addr hi
            0xA,    // flags
            0x2,    // type
        ]);
        dcb.extend(pad(4));
        cp.run(&dcb, &mut sink).expect("indexed draw");
        assert_eq!(
            sink.indexed,
            [IndexedDraw {
                index_type_and_size: 2,
                index_count: 6,
                index_addr: 0x1_0000_5000,
                flags: 0xA,
                index_type: 0x2,
            }]
        );
    }

    /// The default `draw_index` degrades to a vertex-count-only auto draw —
    /// this is what `OffscreenDrawSink` inherits.
    #[test]
    fn indexed_draw_degrades_to_auto_draw_by_default() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = vec![
            header(3, pm4::IT_SET_UCONFIG_REG, pm4::R_ZERO),
            pm4::VGT_PRIMITIVE_TYPE,
            4,
        ];
        dcb.extend_from_slice(&[
            header(10, pm4::IT_NOP, pm4::R_DRAW_INDEX),
            6,
            0x5000,
            0,
            0,
            0,
        ]);
        dcb.extend(pad(4));
        cp.run(&dcb, &mut sink).expect("degraded indexed draw");
        assert_eq!(
            sink.draws,
            [(6, 0, 4, false, false)],
            "index_count becomes the vertex count"
        );
    }

    /// Raw `IT_DRAW_INDEX_2` form (Kyty cmd 0xc0042700): count, addr, dup
    /// count, zero — Kyty passes flags=0, type=1.
    #[test]
    fn it_draw_index_2_is_an_indexed_draw_with_type_1() {
        let mut cp = CommandProcessor::new();
        let mut sink = IndexedSink::default();
        let dcb = vec![
            header(6, pm4::IT_DRAW_INDEX_2, pm4::R_ZERO),
            3,      // index_count
            0x2000, // addr lo
            0,      // addr hi
            3,      // Kyty asserts this duplicates the count
            0,
        ];
        cp.run(&dcb, &mut sink).expect("raw indexed draw");
        assert_eq!(
            sink.indexed,
            [IndexedDraw {
                index_type_and_size: 0,
                index_count: 3,
                index_addr: 0x2000,
                flags: 0,
                index_type: 1,
            }]
        );
    }

    // ---- indirect draws ---------------------------------------------------

    /// `SET_BASE(1)` + `DRAW_INDIRECT` with a readable args buffer degrades to
    /// an auto draw whose count comes from guest memory.
    #[test]
    fn draw_indirect_reads_count_from_guest_args() {
        let mem = BufMem {
            base: 0x9000,
            // args records: {count, instance_count, first, first_instance}
            words: vec![12, 1, 0, 0, 24, 1, 0, 0],
        };
        let mut cp = CommandProcessor::new();
        let mut sink = IndexedSink::default();
        let dcb = vec![
            header(4, pm4::IT_SET_BASE, pm4::R_ZERO),
            1, // base select: indirect args
            0x9000,
            0,
            header(5, pm4::IT_DRAW_INDIRECT, pm4::R_ZERO),
            16, // data offset -> second record
            0,
            0,
            0,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("indirect draw");
        assert_eq!(cp.indirect_draw_base(), 0x9000);
        assert_eq!(
            sink.auto_draws,
            [24],
            "count comes from the args record at base+offset"
        );
    }

    /// The indexed indirect form routes through `draw_index` with the tracked
    /// index-buffer state.
    #[test]
    fn draw_index_indirect_degrades_to_an_indexed_draw() {
        let mem = BufMem {
            base: 0x9000,
            words: vec![36, 1, 0, 0, 0],
        };
        let mut cp = CommandProcessor::new();
        let mut sink = IndexedSink::default();
        let dcb = vec![
            header(3, pm4::IT_INDEX_BASE, pm4::R_ZERO),
            0x5000,
            0,
            header(4, pm4::IT_SET_BASE, pm4::R_ZERO),
            1,
            0x9000,
            0,
            header(6, pm4::IT_DRAW_INDEX_INDIRECT_MULTI, pm4::R_ZERO),
            0, // data offset
            0,
            0,
            0,
            0,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("indexed indirect draw");
        assert_eq!(sink.indexed.len(), 1);
        assert_eq!(sink.indexed[0].index_count, 36);
        assert_eq!(sink.indexed[0].index_addr, 0x5000);
    }

    /// Without a memory reader (or without SET_BASE) an indirect draw is a
    /// logged skip, never an error and never a draw.
    #[test]
    fn draw_indirect_without_memory_or_base_is_skipped() {
        let mut cp = CommandProcessor::new();
        let mut sink = IndexedSink::default();
        // No SET_BASE.
        let dcb = vec![header(5, pm4::IT_DRAW_INDIRECT, pm4::R_ZERO), 0, 0, 0, 0];
        cp.run(&dcb, &mut sink).expect("skip");
        assert!(sink.auto_draws.is_empty());

        // SET_BASE but no memory reader.
        let dcb = vec![
            header(4, pm4::IT_SET_BASE, pm4::R_ZERO),
            1,
            0x9000,
            0,
            header(5, pm4::IT_DRAW_INDIRECT, pm4::R_ZERO),
            0,
            0,
            0,
            0,
        ];
        cp.run(&dcb, &mut sink).expect("skip without memory");
        assert!(sink.auto_draws.is_empty());
        assert!(cp.distinct_skips() >= 2);
    }

    // ---- indirect dispatches (async-compute ACB arm) -----------------------

    /// KytyPS5 `CpOpDispatchIndirect` 3-DWORD form: `SET_BASE(1, dispatch)` +
    /// `DISPATCH_INDIRECT [data_offset, mode]` reads the thread-group counts
    /// from guest memory at base + offset. The shader-type header bit routes
    /// the base to the DISPATCH slot without disturbing the draw slot.
    #[test]
    fn dispatch_indirect_reads_groups_from_the_dispatch_base() {
        let mem = BufMem {
            base: 0x9000,
            // two args records: {x, y, z}
            words: vec![4, 5, 6, 7, 8, 9],
        };
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            // libSceAgc setBaseIndirectArgs with ShaderType compute → bit 1.
            header(4, pm4::IT_SET_BASE, pm4::R_ZERO) | (1 << 1),
            1, // base select: indirect args
            0x9000,
            0,
            header(3, pm4::IT_DISPATCH_INDIRECT, pm4::R_ZERO),
            12,   // data offset -> second record
            0x2A, // mode/initiator
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("indirect dispatch");
        assert_eq!(
            cp.indirect_draw_base(),
            0,
            "compute SET_BASE must not clobber the draw-args base"
        );
        assert_eq!(sink.dispatches.len(), 1);
        let (groups, mode, ..) = sink.dispatches[0];
        assert_eq!(
            groups,
            [7, 8, 9],
            "groups come from the record at base+offset"
        );
        assert_eq!(mode, 0x2A);
    }

    /// The 4-DWORD form carries the absolute args address inline
    /// (KytyPS5 pm4Handlers.cpp L2013-2028) and needs no SET_BASE.
    #[test]
    fn dispatch_indirect_absolute_address_form_needs_no_base() {
        let mem = BufMem {
            base: 0x6000,
            words: vec![2, 3, 4],
        };
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(4, pm4::IT_DISPATCH_INDIRECT, pm4::R_ZERO),
            0x6000,
            0,
            1, // mode
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("absolute indirect dispatch");
        assert_eq!(sink.dispatches.len(), 1);
        let (groups, mode, ..) = sink.dispatches[0];
        assert_eq!(groups, [2, 3, 4]);
        assert_eq!(mode, 1);
    }

    /// Without a programmed dispatch base (or without a memory reader) the
    /// packet is a logged skip — the walk continues to the completion labels
    /// behind it, mirroring the indirect-draw degrade policy.
    #[test]
    fn dispatch_indirect_without_base_or_memory_is_skipped() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        // No SET_BASE at all.
        let dcb = vec![header(3, pm4::IT_DISPATCH_INDIRECT, pm4::R_ZERO), 0, 0];
        cp.run(&dcb, &mut sink).expect("skip without base");
        assert!(sink.dispatches.is_empty());

        // A DRAW base alone must not satisfy the dispatch form.
        let dcb = vec![
            header(4, pm4::IT_SET_BASE, pm4::R_ZERO),
            1,
            0x9000,
            0,
            header(3, pm4::IT_DISPATCH_INDIRECT, pm4::R_ZERO),
            0,
            0,
        ];
        let mem = BufMem {
            base: 0x9000,
            words: vec![1, 1, 1],
        };
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("skip with only a draw base");
        assert!(sink.dispatches.is_empty());
    }

    // ---- indirect registers -----------------------------------------------

    /// Kyty: `cp_op_indirect_cx_regs` — `(offset, value)` pairs fetched from
    /// guest memory feed the same per-register setters as direct writes.
    #[test]
    fn cx_regs_indirect_feeds_the_context_setters() {
        let mem = BufMem {
            base: 0x4000,
            words: vec![
                pm4::CB_COLOR0_ATTRIB2,
                (95 << 14) | 47,
                pm4::CB_TARGET_MASK,
                0xF,
            ],
        };
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(4, pm4::IT_NOP, pm4::R_CX_REGS_INDIRECT),
            2, // num_regs
            0x4000,
            0,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("indirect cx regs");
        let rt = &cp.get_ctx().render_targets[0];
        assert_eq!((rt.attrib2.width, rt.attrib2.height), (95, 47));
        assert_eq!(cp.get_ctx().render_target_mask, 0xF);
    }

    #[test]
    fn cx_regs_indirect_decodes_the_measured_gen5_scissor_defaults() {
        let mem = BufMem {
            base: 0x4000,
            words: vec![
                // Minecraft PPSA17221 primary-register state, matching the
                // AGC/Kyty Gen5 defaults carried by the title's indirect list.
                pm4::PA_SC_SCREEN_SCISSOR_TL,
                0,
                pm4::PA_SC_SCREEN_SCISSOR_BR,
                0x4000_4000,
                pm4::PA_SC_GENERIC_SCISSOR_TL,
                0x8000_0000,
                pm4::PA_SC_GENERIC_SCISSOR_BR,
                0x4000_4000,
                pm4::PA_SC_VPORT_SCISSOR_0_TL,
                0x8000_0000,
                pm4::PA_SC_VPORT_SCISSOR_0_BR,
                0x4000_4000,
            ],
        };
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(4, pm4::IT_NOP, pm4::R_CX_REGS_INDIRECT),
            6,
            0x4000,
            0,
        ];

        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("measured indirect scissor defaults");
        let sv = &cp.get_ctx().screen_viewport;
        assert_eq!(
            (
                sv.screen_scissor_left,
                sv.screen_scissor_top,
                sv.screen_scissor_right,
                sv.screen_scissor_bottom,
            ),
            (0, 0, 0x4000, 0x4000)
        );
        assert_eq!(
            (
                sv.generic_scissor_left,
                sv.generic_scissor_top,
                sv.generic_scissor_right,
                sv.generic_scissor_bottom,
                sv.generic_scissor_window_offset_enable,
            ),
            (0, 0, 0x4000, 0x4000, false)
        );
        let vp = &sv.viewports[0];
        assert_eq!(
            (
                vp.viewport_scissor_left,
                vp.viewport_scissor_top,
                vp.viewport_scissor_right,
                vp.viewport_scissor_bottom,
                vp.viewport_scissor_window_offset_enable,
            ),
            (0, 0, 0x4000, 0x4000, false)
        );
    }

    #[test]
    fn sh_and_uc_regs_indirect_feed_their_setters() {
        let mem = BufMem {
            base: 0x4000,
            words: vec![
                // sh pairs at 0x4000
                pm4::SPI_SHADER_USER_DATA_VS_0,
                0xBEEF,
                // uc pairs at 0x4008
                pm4::VGT_PRIMITIVE_TYPE,
                17,
            ],
        };
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(4, pm4::IT_NOP, pm4::R_SH_REGS_INDIRECT),
            1,
            0x4000,
            0,
            header(4, pm4::IT_NOP, pm4::R_UC_REGS_INDIRECT),
            1,
            0x4008,
            0,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("indirect sh/uc regs");
        assert_eq!(cp.get_sh_ctx().vs.vs_user_sgpr.value[0], 0xBEEF);
        assert_eq!(cp.get_ucfg().prim_type, 17);
    }

    /// Without a memory reader the indirect-register packet is skipped (warn
    /// once), not a fault — and the rest of the DCB still executes.
    #[test]
    fn regs_indirect_without_memory_skips_and_continues() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = vec![
            header(4, pm4::IT_NOP, pm4::R_CX_REGS_INDIRECT),
            2,
            0x4000,
            0,
        ];
        dcb.extend(state_and_draw());
        cp.run(&dcb, &mut sink).expect("skip indirect regs");
        assert_eq!(sink.draws.len(), 1, "the draw after the skip still lands");
        assert_eq!(cp.distinct_skips(), 1);
    }

    /// An unreadable pointer is guest data, not a stream fault: warn + skip.
    #[test]
    fn regs_indirect_with_unreadable_pointer_skips() {
        let mem = BufMem {
            base: 0x4000,
            words: vec![0; 2],
        };
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(4, pm4::IT_NOP, pm4::R_CX_REGS_INDIRECT),
            8, // more pairs than the buffer holds
            0x4000,
            0,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("unreadable pointer skips");
        assert_eq!(cp.distinct_skips(), 1);
    }

    /// `CB_BLEND0_CONTROL` decodes per-slot with the Kyty field layout (the
    /// title's UI blending reaches Vulkan through this); the blend-colour
    /// registers land as floats. Measured on Minecraft: writes to 0x1E0.
    #[test]
    fn blend_control_registers_decode_into_context() {
        let mut cp = CommandProcessor::new();
        // enable | separate-alpha | color SrcAlpha/ADD/OneMinusSrcAlpha,
        // alpha One/ADD/Zero.
        let value = (1 << 30) | (1 << 29) | 0x04 | (0x05 << 8) | (0x01 << 16);
        cp.set_context_register(pm4::CB_BLEND0_CONTROL, value);
        let bc = &cp.ctx.blend_control[0];
        assert!(bc.enable);
        assert!(bc.separate_alpha_blend);
        assert_eq!(bc.color_srcblend, 0x04);
        assert_eq!(bc.color_comb_fcn, 0);
        assert_eq!(bc.color_destblend, 0x05);
        assert_eq!(bc.alpha_srcblend, 0x01);
        assert_eq!(bc.alpha_destblend, 0x00);
        // Each slot lands in its own entry.
        cp.set_context_register(pm4::CB_BLEND0_CONTROL + 3, 1 << 30);
        assert!(cp.ctx.blend_control[3].enable);
        assert!(!cp.ctx.blend_control[1].enable);
        // Blend colour arrives as raw float bits.
        cp.set_context_register(pm4::CB_BLEND_RED, 0x3f80_0000);
        assert_eq!(cp.ctx.blend_color.red, 1.0);
    }

    /// Depth register writes decode into the context: surface format, bases,
    /// extent, and clear values. Field layouts are Kyty's Pm4.h — a title's
    /// z-prepass reaches the Vulkan depth attachment through these.
    #[test]
    fn depth_registers_decode_into_context() {
        let mut cp = CommandProcessor::new();

        // DB_Z_INFO: format 3 (Z32F) | zrange_precision 1.
        cp.set_context_register(pm4::DB_Z_INFO, 3 | (1 << 31));
        cp.set_context_register(pm4::DB_STENCIL_INFO, 0);
        // Bases assemble LO<<8, HI low byte <<40 (Kyty GraphicsRun.cpp L3896).
        cp.set_context_register(pm4::DB_Z_WRITE_BASE, 0x200);
        cp.set_context_register(pm4::DB_Z_WRITE_BASE_HI, 0x1);
        cp.set_context_register(pm4::DB_Z_READ_BASE, 0x200);
        // DB_DEPTH_SIZE_XY: x_max | y_max<<16.
        cp.set_context_register(pm4::DB_DEPTH_SIZE_XY, 63 | (63 << 16));
        // Clear values: depth is a float, stencil an 8-bit value.
        cp.set_context_register(pm4::DB_DEPTH_CLEAR, 1.0f32.to_bits());
        cp.set_context_register(pm4::DB_STENCIL_CLEAR, 0x2A);
        // DB_RENDER_CONTROL: depth clear enable.
        cp.set_context_register(pm4::DB_RENDER_CONTROL, 1);

        let z = &cp.ctx.depth_render_target;
        assert_eq!(z.z_info.format, 3);
        assert_eq!(z.z_info.zrange_precision, 1);
        assert_eq!(z.stencil_info.format, 0);
        assert_eq!(z.z_write_base_addr, (0x1 << 40) | (0x200 << 8));
        assert_eq!(z.z_read_base_addr, 0x200 << 8);
        assert_eq!((z.size.x_max, z.size.y_max), (63, 63));
        assert_eq!(cp.ctx.depth_clear_value, 1.0);
        assert_eq!(cp.ctx.stencil_clear_value, 0x2A);
        assert!(cp.ctx.render_control.depth_clear_enable);
        assert!(!cp.ctx.render_control.stencil_clear_enable);
    }

    /// `DB_STENCIL_CONTROL`'s six ops and the front/back refmask registers.
    #[test]
    fn stencil_registers_decode_into_context() {
        let mut cp = CommandProcessor::new();
        // fail=1 (Zero) | zpass=3 (ReplaceTest) | zfail=0 (Keep) | bf fail=8 (AddWrap).
        cp.set_context_register(pm4::DB_STENCIL_CONTROL, 1 | (3 << 4) | (8 << 12));
        let sc = &cp.ctx.stencil_control;
        assert_eq!(sc.stencil_fail, 1);
        assert_eq!(sc.stencil_zpass, 3);
        assert_eq!(sc.stencil_zfail, 0);
        assert_eq!(sc.stencil_fail_bf, 8);
        assert_eq!(sc.stencil_zpass_bf, 0);

        cp.set_context_register(
            pm4::DB_STENCILREFMASK,
            0x11 | (0x22 << 8) | (0x33 << 16) | (0x44 << 24),
        );
        cp.set_context_register(
            pm4::DB_STENCILREFMASK_BF,
            0x55 | (0x66 << 8) | (0x77 << 16) | (0x88 << 24),
        );
        let sm = &cp.ctx.stencil_mask;
        assert_eq!(sm.stencil_testval, 0x11);
        assert_eq!(sm.stencil_mask, 0x22);
        assert_eq!(sm.stencil_writemask, 0x33);
        assert_eq!(sm.stencil_opval, 0x44);
        assert_eq!(sm.stencil_testval_bf, 0x55);
        assert_eq!(sm.stencil_mask_bf, 0x66);
        assert_eq!(sm.stencil_writemask_bf, 0x77);
        assert_eq!(sm.stencil_opval_bf, 0x88);
    }

    /// The batched form a Gen5 driver emits: one SET_CONTEXT_REG packet writes
    /// DB_Z_INFO through DB_DEPTH_SLICE — every register in the run must land.
    #[test]
    fn depth_surface_batch_writes_all_registers() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(10, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::DB_Z_INFO,
            3,              // DB_Z_INFO: Z32F
            0,              // DB_STENCIL_INFO
            0x200,          // DB_Z_READ_BASE
            0,              // DB_STENCIL_READ_BASE
            0x200,          // DB_Z_WRITE_BASE
            0,              // DB_STENCIL_WRITE_BASE
            (63 << 11) | 7, // DB_DEPTH_SIZE: pitch 7, height 63 tile max
            0x3FF,          // DB_DEPTH_SLICE: slice tile max
        ];
        cp.run(&dcb, &mut sink).expect("depth batch");
        let z = &cp.ctx.depth_render_target;
        assert_eq!(z.z_info.format, 3);
        assert_eq!(z.z_read_base_addr, 0x200 << 8);
        assert_eq!(z.z_write_base_addr, 0x200 << 8);
        assert_eq!(z.pitch_div8_minus1, 7);
        assert_eq!(z.height_div8_minus1, 63);
        assert_eq!(z.slice_div64_minus1, 0x3FF);
    }

    // ---- WAIT_REG_MEM suspend/resume (SharpEmu AgcExports.cs:4508-4529) ----

    /// Guest memory with a mutable label, so a test can play the producer.
    struct LabelMem {
        base: u64,
        words: std::cell::RefCell<Vec<u32>>,
    }

    impl GuestMemory for LabelMem {
        fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
            let rel = addr.checked_sub(self.base)?;
            if rel % 4 != 0 {
                return None;
            }
            let start = usize::try_from(rel / 4).ok()?;
            let end = start.checked_add(count as usize)?;
            self.words.borrow().get(start..end).map(<[u32]>::to_vec)
        }
    }

    /// `sceAgcAcbWaitRegMem` 32-bit layout: total 6 dwords,
    /// body `[addr_lo, addr_hi, mask32, compare, ref32]`.
    fn wait32(addr: u64, mask: u32, compare: u32, reference: u32) -> Vec<u32> {
        vec![
            header(6, pm4::IT_NOP, pm4::R_WAIT_MEM_32),
            addr as u32,
            (addr >> 32) as u32,
            mask,
            compare,
            reference,
        ]
    }

    /// An unmet 32-bit label wait suspends the walk mid-stream: nothing past
    /// the packet runs, the outcome names the label and the resume dword, and
    /// re-running from there after the label is genuinely written executes
    /// the remainder. The label itself is never modified by the CP.
    #[test]
    fn wait_mem32_suspends_and_resumes_where_it_stopped() {
        let mem = LabelMem {
            base: 0x9000,
            words: std::cell::RefCell::new(vec![0]),
        };
        let mut dcb = wait32(0x9000, 0xFFFF_FFFF, 3, 1); // wait for label == 1
        // Work "behind" the wait: prim type write + a draw.
        dcb.extend(state_and_draw());
        let resume_at = 6usize;

        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let outcome = cp
            .run_resumable(&dcb, 0, &mut sink, Some(&mem))
            .expect("wait packet must not fault");
        match outcome {
            RunOutcome::Suspended(s) => {
                assert_eq!(s.resume_dword, resume_at);
                assert_eq!(s.wait.address, 0x9000);
                assert_eq!(s.wait.compare, 3);
                assert_eq!(s.wait.reference, 1);
                assert!(!s.wait.is_64);
            }
            RunOutcome::Completed => panic!("unmet wait must suspend"),
        }
        assert!(sink.draws.is_empty(), "work behind the wait must not run");
        assert_eq!(
            mem.words.borrow()[0],
            0,
            "the CP must never write the label"
        );

        // Producer writes the label; the re-check would now pass.
        mem.words.borrow_mut()[0] = 1;
        let spec = match outcome {
            RunOutcome::Suspended(s) => s.wait,
            RunOutcome::Completed => unreachable!(),
        };
        assert_eq!(spec.read_label(&mem), Some(1));
        assert!(spec.satisfied_by(1));

        let outcome = cp
            .run_resumable(&dcb, resume_at, &mut sink, Some(&mem))
            .expect("resumed walk");
        assert_eq!(outcome, RunOutcome::Completed);
        assert_eq!(sink.draws.len(), 1, "the resumed remainder must draw");
    }

    /// The 64-bit form: total 9 dwords, body `[addr_lo, addr_hi, mask_lo,
    /// mask_hi, ref_lo, ref_hi, compare, poll]`; masked `>=` comparison.
    #[test]
    fn wait_mem64_parses_the_wide_layout_and_compares_masked() {
        let mem = LabelMem {
            base: 0x9000,
            words: std::cell::RefCell::new(vec![0x5, 0x0]), // label = 0x0000_0000_0000_0005
        };
        let dcb = vec![
            header(9, pm4::IT_NOP, pm4::R_WAIT_MEM_64),
            0x9000,
            0,
            0xFFFF_FFFF, // mask lo
            0xFFFF_FFFF, // mask hi
            0x10,        // ref lo
            0,           // ref hi
            5,           // compare: >=
            0,           // poll cycles
        ];
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let outcome = cp.run_resumable(&dcb, 0, &mut sink, Some(&mem)).unwrap();
        let spec = match outcome {
            RunOutcome::Suspended(s) => {
                assert!(s.wait.is_64);
                assert_eq!(s.wait.reference, 0x10);
                s.wait
            }
            RunOutcome::Completed => panic!("5 >= 0x10 is false — must suspend"),
        };
        // Label reaches the reference: satisfied.
        mem.words.borrow_mut()[0] = 0x10;
        assert_eq!(spec.read_label(&mem), Some(0x10));
        assert!(spec.satisfied_by(0x10));
        // Masked comparison: bits outside the mask are invisible.
        let masked = WaitSpec { mask: 0xFF, ..spec };
        assert!(masked.satisfied_by(0xAB00_0010));
    }

    /// The standard `IT_WAIT_REG_MEM` form: body `[control, addr_lo, addr_hi,
    /// ref32, mask32, poll]`, compare in control bits 2:0.
    #[test]
    fn standard_wait_reg_mem_suspends_on_unmet_equal() {
        let mem = LabelMem {
            base: 0x9000,
            words: std::cell::RefCell::new(vec![7]),
        };
        let dcb = vec![
            header(7, pm4::IT_WAIT_REG_MEM, pm4::R_ZERO),
            3, // compare ==
            0x9000,
            0,
            1,           // reference
            0xFFFF_FFFF, // mask
            2,           // poll
        ];
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        match cp.run_resumable(&dcb, 0, &mut sink, Some(&mem)).unwrap() {
            RunOutcome::Suspended(s) => {
                assert_eq!(s.wait.address, 0x9000);
                assert_eq!(s.wait.reference, 1);
                assert_eq!(s.resume_dword, 7);
            }
            RunOutcome::Completed => panic!("7 != 1 — must suspend"),
        }
    }

    /// Fail-open guards (SharpEmu AgcExports.cs:4620-4644 / 4700-4703): the
    /// "always" and reserved compare functions, a null/zero-mask packet, a
    /// missing memory reader, an unreadable label, and an already-satisfied
    /// condition must all keep parsing — suspension only for a genuine,
    /// evaluable, unmet wait.
    #[test]
    fn wait_mem_fail_open_cases_never_suspend() {
        let mem = LabelMem {
            base: 0x9000,
            words: std::cell::RefCell::new(vec![0]),
        };
        let mut sink = RecordingSink::default();
        let run = |dcb: &[u32], mem: Option<&dyn GuestMemory>, sink: &mut RecordingSink| {
            CommandProcessor::new()
                .run_resumable(dcb, 0, sink, mem)
                .expect("no structural fault")
        };
        // compare 0 = always, 7 = reserved.
        for compare in [0, 7] {
            let dcb = wait32(0x9000, !0, compare, 1);
            assert_eq!(
                run(&dcb, Some(&mem), &mut sink),
                RunOutcome::Completed,
                "compare {compare} is fail-open"
            );
        }
        // Null address / zero mask / misaligned label.
        for dcb in [
            wait32(0, !0, 3, 1),
            wait32(0x9000, 0, 3, 1),
            wait32(0x9002, !0, 3, 1),
        ] {
            assert_eq!(run(&dcb, Some(&mem), &mut sink), RunOutcome::Completed);
        }
        // No memory reader; unreadable label.
        let dcb = wait32(0x9000, !0, 3, 1);
        assert_eq!(run(&dcb, None, &mut sink), RunOutcome::Completed);
        let unreadable = wait32(0xDEAD_0000, !0, 3, 1);
        assert_eq!(
            run(&unreadable, Some(&mem), &mut sink),
            RunOutcome::Completed
        );
        // Already satisfied.
        mem.words.borrow_mut()[0] = 1;
        let dcb = wait32(0x9000, !0, 3, 1);
        assert_eq!(run(&dcb, Some(&mem), &mut sink), RunOutcome::Completed);
    }

    /// `run_with_memory` (the non-resumable entry) must keep its historical
    /// contract: an unmet wait cannot wedge it — it warns and continues.
    #[test]
    fn non_resumable_walk_continues_past_an_unmet_wait() {
        let mem = LabelMem {
            base: 0x9000,
            words: std::cell::RefCell::new(vec![0]),
        };
        let mut dcb = wait32(0x9000, !0, 3, 1);
        dcb.extend(state_and_draw());
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("legacy walk");
        assert_eq!(sink.draws.len(), 1, "legacy walk still reaches the draw");
        assert_eq!(mem.words.borrow()[0], 0, "and never writes the label");
    }

    // ---- Label PRODUCERS: WRITE_DATA / RELEASE_MEM (SharpEmu
    // ApplySubmittedWriteData / ApplySubmittedReleaseMem) ----

    /// Guest memory that both reads dwords (for `WAIT_REG_MEM`) and writes bytes
    /// (for the `WRITE_DATA`/`RELEASE_MEM` producers), so one test can play both
    /// the consumer and the producer.
    struct RwMem {
        base: u64,
        words: std::cell::RefCell<Vec<u32>>,
    }

    impl RwMem {
        fn new(base: u64, len: usize) -> Self {
            Self {
                base,
                words: std::cell::RefCell::new(vec![0u32; len]),
            }
        }
        fn word(&self, index: usize) -> u32 {
            self.words.borrow()[index]
        }
    }

    impl GuestMemory for RwMem {
        fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
            let rel = addr.checked_sub(self.base)?;
            if rel % 4 != 0 {
                return None;
            }
            let start = usize::try_from(rel / 4).ok()?;
            let end = start.checked_add(count as usize)?;
            self.words.borrow().get(start..end).map(<[u32]>::to_vec)
        }

        fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
            let Some(rel) = addr.checked_sub(self.base) else {
                return false;
            };
            if rel % 4 != 0 || bytes.len() % 4 != 0 {
                return false;
            }
            let start = usize::try_from(rel / 4).expect("in range");
            let mut words = self.words.borrow_mut();
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                let Some(slot) = words.get_mut(start + i) else {
                    return false;
                };
                *slot = u32::from_le_bytes(chunk.try_into().expect("4 bytes"));
            }
            true
        }
    }

    /// AGC `IT_NOP`+`R_WRITE_DATA`: `[control, dst_lo, dst_hi, value]`. Control
    /// low byte is DST_SEL (1 = memory), third byte is ADDR_INCR (0 = increment).
    fn write_data_agc(addr: u64, value: u32) -> Vec<u32> {
        vec![
            header(5, pm4::IT_NOP, pm4::R_WRITE_DATA),
            1, // dst_sel = 1 (memory), addr-increment enabled
            addr as u32,
            (addr >> 32) as u32,
            value,
        ]
    }

    /// AGC `IT_NOP`+`R_RELEASE_MEM`: `[event, control, dst_lo, dst_hi, data_lo,
    /// data_hi]`. Control DATA_SEL byte (bits 23:16) = 1 → 32-bit immediate.
    fn release_mem_agc(addr: u64, value: u32) -> Vec<u32> {
        vec![
            header(7, pm4::IT_NOP, pm4::R_RELEASE_MEM),
            0,       // event/GCR field (ignored by the memory write)
            1 << 16, // control: DATA_SEL = 1
            addr as u32,
            (addr >> 32) as u32,
            value, // data_lo
            0,     // data_hi
        ]
    }

    /// Both `WRITE_DATA` forms write the label to guest memory and record it for
    /// the cross-queue wait latch. The old behaviour consumed the packet without
    /// effect, which is why cross-queue waits were never satisfied.
    #[test]
    fn write_data_writes_the_label_in_both_forms() {
        // AGC form.
        let mem = RwMem::new(0x9000, 4);
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&write_data_agc(0x9000, 0x2A), &mut sink, Some(&mem))
            .expect("producer must not fault");
        assert_eq!(mem.word(0), 0x2A, "AGC WRITE_DATA wrote the label");
        assert_eq!(cp.take_produced_labels(), vec![(0x9000, 0x2A)]);

        // Standard IT_WRITE_DATA: DST_SEL in bits 11:8 (1 = memory), ADDR_INCR
        // bit 16 clear = increment.
        let mem = RwMem::new(0x9000, 4);
        let dcb = vec![
            header(5, pm4::IT_WRITE_DATA, pm4::R_ZERO),
            1 << 8, // DST_SEL = 1
            0x9004u32,
            0,
            0x5B,
        ];
        let mut cp = CommandProcessor::new();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("producer must not fault");
        assert_eq!(mem.word(1), 0x5B, "standard WRITE_DATA wrote the label");
        assert_eq!(cp.take_produced_labels(), vec![(0x9004, 0x5B)]);
    }

    /// Both `RELEASE_MEM` forms write a 32-bit immediate completion label.
    #[test]
    fn release_mem_writes_the_label_in_both_forms() {
        // AGC form.
        let mem = RwMem::new(0x9000, 4);
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&release_mem_agc(0x9000, 7), &mut sink, Some(&mem))
            .expect("producer must not fault");
        assert_eq!(mem.word(0), 7, "AGC RELEASE_MEM wrote the label");
        assert_eq!(cp.take_produced_labels(), vec![(0x9000, 7)]);

        // Standard IT_RELEASE_MEM: DST_SEL bits 17:16 = 0 (memory), DATA_SEL
        // bits 31:29 = 1 (32-bit immediate). 8 total dwords (the trailing
        // INT_CTXID dword the decoder ignores).
        let mem = RwMem::new(0x9000, 4);
        let dcb = vec![
            header(8, pm4::IT_RELEASE_MEM, pm4::R_ZERO),
            0,          // event
            1u32 << 29, // control: DATA_SEL = 1, DST_SEL = 0 (memory)
            0x9008u32,
            0,
            9, // data_lo
            0, // data_hi
            0, // int_ctxid (ignored)
        ];
        let mut cp = CommandProcessor::new();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("producer must not fault");
        assert_eq!(mem.word(2), 9, "standard RELEASE_MEM wrote the label");
        assert_eq!(cp.take_produced_labels(), vec![(0x9008, 9)]);
    }

    /// A 64-bit `RELEASE_MEM` (DATA_SEL 2) writes both dwords and records the
    /// 64-bit value; a timestamp form (DATA_SEL 3) writes a nonzero value but is
    /// not recorded for the equality latch.
    #[test]
    fn release_mem_data_sel_variants() {
        let mem = RwMem::new(0x9000, 4);
        let dcb = vec![
            header(7, pm4::IT_NOP, pm4::R_RELEASE_MEM),
            0,
            2 << 16, // DATA_SEL = 2 (64-bit)
            0x9000u32,
            0,
            0xDEAD_BEEF,
            0x0000_0001,
        ];
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem)).unwrap();
        assert_eq!(mem.word(0), 0xDEAD_BEEF, "64-bit low dword");
        assert_eq!(mem.word(1), 0x0000_0001, "64-bit high dword");
        assert_eq!(
            cp.take_produced_labels(),
            vec![(0x9000, 0x1_DEAD_BEEF)],
            "64-bit value recorded for the latch"
        );

        let mem = RwMem::new(0x9000, 4);
        let dcb = vec![
            header(7, pm4::IT_NOP, pm4::R_RELEASE_MEM),
            0,
            3 << 16, // DATA_SEL = 3 (GPU timestamp)
            0x9000u32,
            0,
            0,
            0,
        ];
        let mut cp = CommandProcessor::new();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem)).unwrap();
        assert_ne!(mem.word(0), 0, "timestamp form writes a nonzero value");
        assert!(
            cp.take_produced_labels().is_empty(),
            "timestamp is a counter, not an equality-latched label"
        );
    }

    /// A producer with no `GuestMemory` writer is skipped (one warn), never a
    /// stream fault — read-only embedders keep working.
    #[test]
    fn producers_without_memory_are_skipped() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let mut dcb = write_data_agc(0x9000, 1);
        dcb.extend(release_mem_agc(0x9000, 1));
        dcb.extend(state_and_draw());
        cp.run(&dcb, &mut sink)
            .expect("producers without memory must not kill the DCB");
        assert_eq!(sink.draws.len(), 1, "the stream still reaches the draw");
        assert!(cp.take_produced_labels().is_empty());
    }

    /// End to end at the CP layer: a `WAIT_REG_MEM` suspends, a producer buffer
    /// writes the label, and re-running from the resume point completes — no
    /// force-satisfy anywhere. This is the exact cross-queue gate that black-
    /// screened Minecraft, proven on one command processor.
    #[test]
    fn wait_then_producer_write_lets_the_walk_resume() {
        let mem = RwMem::new(0x9000, 4);
        // Consumer: wait for label(0x9000) == 1, then a draw.
        let mut consumer = wait32(0x9000, !0, 3, 1);
        consumer.extend(state_and_draw());
        let resume_at = consumer.len() - state_and_draw().len();

        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let outcome = cp
            .run_resumable(&consumer, 0, &mut sink, Some(&mem))
            .expect("wait must not fault");
        assert!(
            matches!(outcome, RunOutcome::Suspended(_)),
            "unmet wait suspends"
        );
        assert!(sink.draws.is_empty(), "work behind the wait must not run");
        assert_eq!(mem.word(0), 0, "the label is never force-written");

        // Producer: a WRITE_DATA on the same memory writes the label.
        let mut producer_cp = CommandProcessor::new();
        producer_cp
            .run_with_memory(&write_data_agc(0x9000, 1), &mut sink, Some(&mem))
            .expect("producer must not fault");
        assert_eq!(mem.word(0), 1, "the producer wrote the label");

        // Resume from where the consumer stopped: it now completes and draws.
        let outcome = cp
            .run_resumable(&consumer, resume_at, &mut sink, Some(&mem))
            .expect("resumed walk");
        assert_eq!(outcome, RunOutcome::Completed);
        assert_eq!(sink.draws.len(), 1, "the resumed remainder draws");
    }
}
