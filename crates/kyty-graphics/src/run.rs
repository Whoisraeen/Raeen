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

/// One guest-visible completion side effect a walked packet requested that
/// this command processor cannot apply itself — kernel event queues and
/// VideoOut flips live on the embedder's side of the seam. Recorded in
/// **stream order** ([`CommandProcessor::take_side_effects`]) so the embedder
/// can deliver them in PM4 submission order: a flip or event behind an unmet
/// `WAIT_REG_MEM` is only recorded once the walk genuinely passes the wait,
/// never early.
///
/// The field extraction mirrors the eager submit-time decoder
/// (`raeen_gpu::agc::decode_submission`) exactly, so the gate-off eager
/// duplicate and this in-order record describe the same effect.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SideEffect {
    /// Standard `IT_EVENT_WRITE`: signal kernel equeue events keyed by this id.
    EventWrite {
        /// Event type (low 6 bits of the packet's first body dword).
        event_id: u32,
    },
    /// AGC-form `RELEASE_MEM` end-of-pipe interrupt request: the kernel
    /// delivers it to registered graphics-core events.
    EopInterrupt {
        /// The packet's INT_CTXID dword (0 when absent).
        context_id: u32,
    },
    /// AGC flip packet (`IT_NOP` + `R_FLIP`): a VideoOut flip embedded in the
    /// command stream.
    Flip {
        /// VideoOut handle the title opened.
        video_out_handle: u32,
        /// Registered display-buffer slot to scan out.
        display_buffer_index: u32,
        /// `SceVideoOutFlipMode`.
        flip_mode: u32,
        /// The title's opaque completion argument.
        flip_arg: u64,
    },
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

/// One process-wide note that the CB compression-metadata registers
/// (DCC/CMASK/FMASK addresses, slices, and DCC_CONTROL) are decoded into
/// named `RenderTarget` fields but deliberately NOT emulated: every target
/// renders uncompressed, and no path reads or writes the metadata surfaces.
/// This replaces per-register "unknown context register" warnings for the
/// whole block — the skip is intentional, so it is an INFO, once.
fn note_compression_metadata_ignored() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            "CB compression metadata (DCC/CMASK/FMASK) decoded but ignored — \
             colour targets render uncompressed by design"
        );
    });
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

    /// Read a chained command buffer — the target of an `IT_INDIRECT_BUFFER`
    /// packet.
    ///
    /// Deliberately its own method rather than a [`Self::read_dwords`] call:
    /// that one is the *pointer* read used by indirect register lists and
    /// indirect draw arguments, and embedders cap it accordingly (`raeen-gpu`
    /// refuses over 0x1_0000 dwords there, because anything larger is a
    /// mis-decoded pointer). A command buffer is resource-sized — the PM4
    /// `IB_SIZE` field is 20 bits, so a legal chain target runs to 0xF_FFFF
    /// dwords / 4 MiB — and would be refused by that cap even when it is
    /// perfectly valid. Default delegates, so a read-only embedder keeps
    /// today's behaviour.
    fn read_command_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
        self.read_dwords(addr, count)
    }

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
    /// `writes` carries the EXACT `(base, len)` guest ranges the packet wrote,
    /// coalesced. Sinks holding guest-memory-derived caches (decoded
    /// descriptors, analyzed shaders, texture content hashes) should invalidate
    /// only the entries those ranges can have changed.
    ///
    /// An EMPTY slice means the packet was write-capable but wrote nothing (a
    /// non-memory destination selector, a null address, a refused range, or no
    /// [`GuestMemory`] at all) — there is nothing to invalidate. The
    /// notification still fires so a sink that wants the old blanket-clear
    /// policy can keep it.
    ///
    /// The default keeps simple recording/test sinks source-compatible.
    fn guest_memory_write_boundary(&mut self, writes: &[(u64, u64)]) {
        let _ = writes;
    }

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
    /// The MOST RECENT refusal reason ([`DrawError`] text) behind
    /// [`Self::refused_draws`]. The warn at the skip-and-continue arm is
    /// rate-limited to once per processor, so without this the 2nd..Nth refusals
    /// — and every refusal on a processor that already warned — carried no
    /// recoverable reason at all. The embedder reads it
    /// ([`Self::last_refusal`]) so a black-frame diagnostic can NAME why the
    /// draws did not land instead of reporting a bare `draws=0`.
    ///
    /// Cumulative across queue resets, like `refused_draws`.
    last_refusal: Option<String>,
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
    /// Completion side effects (events, EOP interrupts, flips) the walk
    /// executed, in stream order, for the embedder to deliver — see
    /// [`SideEffect`]. Drained by [`Self::take_side_effects`]; bounded by
    /// [`Self::MAX_SIDE_EFFECTS`].
    side_effects: Vec<SideEffect>,
    /// Guest byte ranges the CURRENT packet actually wrote, as `(base, len)`.
    ///
    /// Drained into [`DrawSink::guest_memory_write_boundary`] after each
    /// write-capable packet and cleared before the next one, so a sink can
    /// invalidate exactly the guest-memory-derived caches the write can have
    /// changed instead of throwing all of them away. Coalesced (see
    /// [`Self::record_guest_write`]) so a `WRITE_DATA` increment loop reports
    /// one span, and bounded by [`Self::MAX_PACKET_WRITE_RANGES`].
    packet_guest_writes: Vec<(u64, u64)>,
    /// Overriding clock for `RELEASE_MEM` DATA_SEL 3/4 GPU-timestamp writes.
    /// `None` from the source (or no source at all) falls back to the legacy
    /// process-local counter ([`next_release_timestamp`]) — bit-identical
    /// default behavior. The embedder installs a source that consults the
    /// `RAEEN_UNIFIED_GPU_CLOCK` gate per call, so both the eager submit-time
    /// writer and this in-stream writer share ONE authoritative clock when
    /// the gate is on (the two-clock double-write was measured to disagree).
    timestamp_source: Option<fn() -> Option<u64>>,
    /// Per-packet-class census of the walk, for the embedder's timing report.
    /// Always counted (a `u64` increment); the `_ns` fields are only filled
    /// when [`Self::set_walk_timing`] is on.
    census: WalkCensus,
    /// Whether to wrap the expensive packet classes in a clock — see
    /// [`WalkCensus`].
    time_walk: bool,
    /// Chain-packet census — see [`ChainCensus`]. Always counted, and (like
    /// `refused_draws`) it survives [`Self::reset`] so a per-frame
    /// `R_DRAW_RESET` cannot zero the honest chain count mid-submission.
    chain_census: ChainCensus,
    /// Whether to WALK `IT_INDIRECT_BUFFER` targets rather than only counting
    /// them. Embedder configuration, so it survives [`Self::reset`]: a
    /// mid-stream queue reset must not silently turn the follower off and drop
    /// the rest of the frame's chained work.
    follow_chains: bool,
    /// Raised by [`Self::cp_op_indirect_buffer`] and consumed immediately after
    /// the packet by whichever walk loop is running it.
    pending_chain: Option<ChainRequest>,
    /// Chained buffers followed since the current top-level walk started —
    /// the termination guarantee for a wide chain graph that never repeats a
    /// buffer on any single path. Bounded by [`Self::MAX_CHAIN_BUFFERS`].
    chain_buffers_followed: u64,
}

/// What the PM4 walk spent its time on, split by packet class.
///
/// `walk_us` in the embedder's `SUBMISSION PHASES` report is the whole walk; it
/// once read 98% of a Dead Cells submission with only a third of that in
/// measurable draw work, and there was no way to say which packet class owned
/// the rest. This names them.
///
/// The whole census is opt-in ([`CommandProcessor::set_walk_timing`]): a
/// `SET_CONTEXT_REG` packet walks in ~10 ns and a `SET_SH_REG` register in
/// ~2.7 ns (`kyty-graphics/tests/pm4_walk_cost.rs`), and even
/// classify-and-increment measured +1-2.5 ns on that. With the census on,
/// register writes are still only COUNTED, never timed — `Instant::now()` costs
/// more than the packet does, so timing them would report the probe rather than
/// the work. The timed classes each cost hundreds of ns or more, where a clock
/// read is noise.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WalkCensus {
    /// `SET_CONTEXT_REG` / `SET_SH_REG` / `SET_UCONFIG_REG*` packets…
    pub reg_packets: u64,
    /// …and the individual registers they wrote (count-only, see above).
    pub regs: u64,
    /// Draw packets, and the nanoseconds their [`DrawSink`] calls took.
    pub draws: u64,
    pub draw_ns: u64,
    /// Dispatch packets and their sink time.
    pub dispatches: u64,
    pub dispatch_ns: u64,
    /// `WRITE_DATA` / `RELEASE_MEM` / `DMA_DATA` packets — the completion
    /// labels a title interleaves with its draws.
    pub write_packets: u64,
    pub write_ns: u64,
    /// The [`DrawSink::guest_memory_write_boundary`] notifications those
    /// packets raised, and what the sink spent invalidating its
    /// guest-memory-derived caches. This is the field that found Dead Cells'
    /// missing 4.7 ms.
    pub boundaries: u64,
    pub boundary_ns: u64,
    /// `WAIT_REG_MEM` and the AGC `R_WAIT_MEM_*` forms.
    pub waits: u64,
    pub wait_ns: u64,
    /// Indirect register/draw/dispatch packets, which re-enter the walk.
    ///
    /// CONFLATED on purpose (it is a cost bucket, not a feature count): this
    /// counts `IT_INDIRECT_BUFFER` / `IT_INDIRECT_BUFFER_CNST` together with the
    /// AGC `R_CX_REGS_INDIRECT` / `R_SH_REGS_INDIRECT` / `R_UC_REGS_INDIRECT`
    /// register lists, which every measured title emits constantly. A non-zero
    /// `indirects` therefore says NOTHING about whether a title chains its
    /// command stream — [`ChainCensus`] is the chain-only measurement.
    pub indirects: u64,
    pub indirect_ns: u64,
    /// Packets consumed by encoded length with no state change: type-2 filler,
    /// markers, `CONTEXT_CONTROL`, and the unknown-opcode skip path.
    pub inert_packets: u64,
}

/// The [`WalkCensus`] bucket one packet belongs to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PacketClass {
    RegWrite,
    Draw,
    Dispatch,
    Write,
    Wait,
    Indirect,
    Inert,
}

impl PacketClass {
    /// Whether a clock read is cheap relative to this class's own cost. See
    /// [`WalkCensus`]: register writes are ~3-10 ns, so timing them would
    /// measure the probe.
    const fn is_timed(self) -> bool {
        matches!(
            self,
            Self::Draw | Self::Dispatch | Self::Write | Self::Wait | Self::Indirect
        )
    }
}

impl WalkCensus {
    /// Attribute one dispatched packet. `consumed` is the handler's dword count
    /// on success — for a register packet that is `registers + 1`, which is how
    /// [`Self::regs`] is counted without touching the setters.
    fn record(
        &mut self,
        class: PacketClass,
        clock: Option<std::time::Instant>,
        consumed: Option<u32>,
    ) {
        let ns = clock.map_or(0, |at| at.elapsed().as_nanos() as u64);
        match class {
            PacketClass::RegWrite => {
                self.reg_packets += 1;
                self.regs += u64::from(consumed.unwrap_or(1).saturating_sub(1));
            }
            PacketClass::Draw => {
                self.draws += 1;
                self.draw_ns += ns;
            }
            PacketClass::Dispatch => {
                self.dispatches += 1;
                self.dispatch_ns += ns;
            }
            PacketClass::Write => {
                self.write_packets += 1;
                self.write_ns += ns;
            }
            PacketClass::Wait => {
                self.waits += 1;
                self.wait_ns += ns;
            }
            PacketClass::Indirect => {
                self.indirects += 1;
                self.indirect_ns += ns;
            }
            PacketClass::Inert => self.inert_packets += 1,
        }
    }
}

/// Which chain form an `IT_INDIRECT_BUFFER` family packet carries.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ChainForm {
    /// 4-dword `IT_INDIRECT_BUFFER`: unconditional chain into one buffer.
    /// `raeen-hle`'s `sceAgcDcbJump` emits exactly this.
    Jump,
    /// 14-dword `IT_INDIRECT_BUFFER`: conditional chain, then-buffer or
    /// else-buffer selected by a masked compare against a guest label.
    /// `raeen-hle`'s `sceAgcCbBranch` emits exactly this; KytyPS5 routes the
    /// same length to `CpOpBranch` (pm4Handlers.cpp L2574).
    Branch,
    /// `IT_INDIRECT_BUFFER_CNST` (0x33) — the CONSTANT-ENGINE ring, not the
    /// graphics/compute ring this processor models. Counted, never followed;
    /// see [`CommandProcessor::cp_op_indirect_buffer`].
    Const,
}

/// One observed chain packet, for the embedder's report. Bounded by
/// [`ChainCensus::MAX_SAMPLES`] — the point is to NAME a few real targets, not
/// to log a frame's worth.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ChainSample {
    /// DWORD offset of the packet inside the buffer that carried it.
    pub offset: u32,
    /// Chain target (0 for a `Const` packet, which is not decoded).
    pub address: u64,
    pub size_dwords: u32,
    /// Raw control dword (`IB_SIZE` | `CHAIN` | `VMID`), for the 4-dword form.
    pub control: u32,
    pub form: ChainForm,
}

/// Whether a submitted DCB chains its frame into other command buffers, and
/// what this processor did about each one.
///
/// This exists because `WalkCensus::indirects` cannot answer the question: it
/// counts `IT_INDIRECT_BUFFER` together with the AGC `R_CX_REGS_INDIRECT` /
/// `R_SH_REGS_INDIRECT` / `R_UC_REGS_INDIRECT` register-list packets, which
/// every measured title emits constantly. A non-zero `indirects` therefore says
/// nothing at all about chaining. Every field here is chain-only.
///
/// Always counted (a handful of `u64` increments on packets that are rare by
/// construction), independent of [`CommandProcessor::set_walk_timing`] — a run
/// with no diagnostics enabled must still be able to say whether chains exist.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChainCensus {
    /// 4-dword unconditional chain packets seen.
    pub jump_packets: u64,
    /// 14-dword conditional chain packets seen.
    pub branch_packets: u64,
    /// `IT_INDIRECT_BUFFER_CNST` packets seen (never followed).
    pub const_packets: u64,
    /// Dwords the chain targets *claim*, summed over every chain packet seen —
    /// how much command stream lives outside the submitted buffer.
    pub target_dwords: u64,
    /// Chained buffers actually walked, and their dword total.
    pub followed: u64,
    pub followed_dwords: u64,
    /// Chain packets decoded but not walked because the follower is off. This
    /// is the field that separates "no chains" from "chains we ignore".
    pub refused_disabled: u64,
    /// Named refusals. Each has a `warn!` at its site; these are the counts so
    /// the 2nd..Nth are never silent.
    pub refused_depth: u64,
    pub refused_cycle: u64,
    pub refused_budget: u64,
    pub refused_unreadable: u64,
    pub refused_malformed: u64,
    pub refused_no_memory: u64,
    /// Chain packets whose `CHAIN` control bit (bit 20) was set. Both
    /// references ignore that bit; this counts it so a stream that uses it
    /// cannot pass unnoticed. See [`CommandProcessor::cp_op_indirect_buffer`].
    pub chain_bit_set: u64,
    /// Wait packets encountered INSIDE a chained buffer, which this processor
    /// cannot suspend on (the resume point is a top-level dword offset) and so
    /// continues past, loudly.
    pub waits_dropped: u64,
    /// A bounded sample of the chain packets seen, newest dropped once full.
    pub samples: Vec<ChainSample>,
}

impl ChainCensus {
    /// How many [`ChainSample`]s are retained per drain.
    pub const MAX_SAMPLES: usize = 8;

    /// Any chain packet at all, of any form. The one-line answer to "does this
    /// title chain its command stream?".
    #[must_use]
    pub const fn packets(&self) -> u64 {
        self.jump_packets + self.branch_packets + self.const_packets
    }

    /// Every named refusal, summed.
    #[must_use]
    pub const fn refusals(&self) -> u64 {
        self.refused_depth
            + self.refused_cycle
            + self.refused_budget
            + self.refused_unreadable
            + self.refused_malformed
            + self.refused_no_memory
    }

    /// Fold `other` in, for an embedder accumulating across submissions.
    pub fn absorb(&mut self, other: &Self) {
        self.jump_packets += other.jump_packets;
        self.branch_packets += other.branch_packets;
        self.const_packets += other.const_packets;
        self.target_dwords += other.target_dwords;
        self.followed += other.followed;
        self.followed_dwords += other.followed_dwords;
        self.refused_disabled += other.refused_disabled;
        self.refused_depth += other.refused_depth;
        self.refused_cycle += other.refused_cycle;
        self.refused_budget += other.refused_budget;
        self.refused_unreadable += other.refused_unreadable;
        self.refused_malformed += other.refused_malformed;
        self.refused_no_memory += other.refused_no_memory;
        self.chain_bit_set += other.chain_bit_set;
        self.waits_dropped += other.waits_dropped;
        for sample in &other.samples {
            if self.samples.len() >= Self::MAX_SAMPLES {
                break;
            }
            self.samples.push(*sample);
        }
    }
}

/// A chain the current packet asked the walk to follow, consumed by
/// [`CommandProcessor::run_resumable`] (top level) or by the chain work-list
/// (nested). Only ever `Some` when the follower is enabled.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ChainRequest {
    /// DWORD offset of the packet that raised it, for the refusal messages.
    offset: u32,
    address: u64,
    size_dwords: u32,
}

/// One open chained command buffer on [`CommandProcessor::run_chain`]'s
/// work-list: the guest bytes, already read and length-checked, plus the cursor
/// into them. `address`/`size_dwords` are retained for the cycle test and for
/// naming the buffer in a refusal.
struct ChainFrame {
    address: u64,
    size_dwords: u32,
    words: Vec<u32>,
    pos: usize,
}

impl CommandProcessor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_instances: 1,
            ..Self::default()
        }
    }

    /// Take the [`WalkCensus`] — classify and count every packet, and clock the
    /// expensive classes. Off by default; see [`WalkCensus`] for why even
    /// counting is gated.
    pub const fn set_walk_timing(&mut self, on: bool) {
        self.time_walk = on;
    }

    /// Drain the per-packet-class census accumulated since the last drain.
    #[must_use]
    pub fn take_walk_census(&mut self) -> WalkCensus {
        std::mem::take(&mut self.census)
    }

    /// Walk `IT_INDIRECT_BUFFER` chain targets instead of only counting them.
    ///
    /// Off by default. With it off, a chain packet is decoded into the
    /// [`ChainCensus`] and then consumed by its encoded length — byte-for-byte
    /// the behaviour before the follower existed, except that the packet is now
    /// named in the log instead of landing in the anonymous unknown-opcode arm.
    pub const fn set_follow_chains(&mut self, on: bool) {
        self.follow_chains = on;
    }

    /// Whether chain following is enabled.
    #[must_use]
    pub const fn follows_chains(&self) -> bool {
        self.follow_chains
    }

    /// Drain the chain census accumulated since the last drain.
    #[must_use]
    pub fn take_chain_census(&mut self) -> ChainCensus {
        std::mem::take(&mut self.chain_census)
    }

    /// Read the chain census without draining it.
    #[must_use]
    pub const fn chain_census(&self) -> &ChainCensus {
        &self.chain_census
    }

    /// `Instant::now()` only when timing is on.
    fn walk_clock(&self) -> Option<std::time::Instant> {
        self.time_walk.then(std::time::Instant::now)
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

    /// Why the most recent refused draw/dispatch was refused — see
    /// [`Self::last_refusal`].
    ///
    /// `Some` exactly when [`Self::refused_draws`] is non-zero. Pair the two in
    /// any "nothing was drawn" diagnostic: a non-zero count with this reason is
    /// the difference between "the GPU dropped work and here is why" and a bare
    /// `draws=0` that names nothing (the state Dead Cells presented in, where
    /// every draw was refused for `indexed draw with no index buffer: addr=0x0`
    /// but the frame path reported only `draws=0 draw_skips=0`).
    #[must_use]
    pub fn last_refusal(&self) -> Option<&str> {
        self.last_refusal.as_deref()
    }

    /// Kyty: `CommandProcessor::Reset` (L519) — clears register and index
    /// state. The warn rate-limit set deliberately survives (deviation; a
    /// reset must not re-arm log spam).
    pub fn reset(&mut self) {
        let warned = std::mem::take(&mut self.warned);
        let shader_bind_trace_count = self.shader_bind_trace_count;
        let refused_draws = self.refused_draws;
        let last_refusal = self.last_refusal.take();
        // Producer labels written before an in-stream queue reset must survive
        // it: the embedder still needs to latch waiters against them, and a
        // per-frame `R_DRAW_RESET` between the producer packet and the drain
        // must not silently drop the wakeup. The same holds for undelivered
        // side effects, and the timestamp source is embedder configuration —
        // a queue reset must not silently fork the clock back to the counter.
        let produced_labels = std::mem::take(&mut self.produced_labels);
        let side_effects = std::mem::take(&mut self.side_effects);
        let timestamp_source = self.timestamp_source;
        // Chain following is embedder configuration and the chain census is a
        // cumulative diagnostic: an in-stream `R_DRAW_RESET` between a chain
        // packet and the embedder's drain must neither turn the follower off
        // (silently dropping the rest of the frame's chained work) nor zero the
        // count of chains already seen. Same argument as `refused_draws`.
        let chain_census = std::mem::take(&mut self.chain_census);
        let follow_chains = self.follow_chains;
        *self = Self::new();
        self.warned = warned;
        self.shader_bind_trace_count = shader_bind_trace_count;
        self.refused_draws = refused_draws;
        self.last_refusal = last_refusal;
        self.produced_labels = produced_labels;
        self.side_effects = side_effects;
        self.timestamp_source = timestamp_source;
        self.chain_census = chain_census;
        self.follow_chains = follow_chains;
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

    /// Cap on distinct write spans reported for ONE packet. Past it the spans
    /// collapse into a single covering range: still correct (a superset
    /// over-invalidates), still bounded, and never a silent omission.
    const MAX_PACKET_WRITE_RANGES: usize = 16;

    /// Note that this packet wrote `len` bytes at `addr`.
    ///
    /// Called at every guest-memory write site in the walk, so
    /// [`DrawSink::guest_memory_write_boundary`] receives the exact extent of
    /// what changed. Spans that abut or overlap the previous one are merged —
    /// `WRITE_DATA` writes dword by dword (incrementing, or repeatedly to the
    /// same address), and reporting one span per dword would defeat the cap.
    fn record_guest_write(&mut self, addr: u64, len: u64) {
        if len == 0 {
            return;
        }
        if let Some(last) = self.packet_guest_writes.last_mut() {
            let (base, span) = *last;
            let end = base.saturating_add(span);
            if addr >= base && addr <= end {
                last.1 = end.max(addr.saturating_add(len)).saturating_sub(base);
                return;
            }
        }
        if self.packet_guest_writes.len() >= Self::MAX_PACKET_WRITE_RANGES {
            let start = self
                .packet_guest_writes
                .iter()
                .map(|&(base, _)| base)
                .chain(std::iter::once(addr))
                .min()
                .unwrap_or(addr);
            let end = self
                .packet_guest_writes
                .iter()
                .map(|&(base, span)| base.saturating_add(span))
                .chain(std::iter::once(addr.saturating_add(len)))
                .max()
                .unwrap_or_else(|| addr.saturating_add(len));
            self.packet_guest_writes.clear();
            self.packet_guest_writes
                .push((start, end.saturating_sub(start)));
            return;
        }
        self.packet_guest_writes.push((addr, len));
    }

    /// [`GuestMemory::write_bytes`] that records the written extent for the
    /// packet's [`DrawSink::guest_memory_write_boundary`] report.
    ///
    /// Every guest write in the walk goes through here: a new write site that
    /// bypassed it would leave a sink's guest-memory caches stale, so this is
    /// the one chokepoint rather than seven bookkeeping calls.
    fn guest_write(&mut self, mem: &dyn GuestMemory, addr: u64, bytes: &[u8]) -> bool {
        if mem.write_bytes(addr, bytes) {
            self.record_guest_write(addr, bytes.len() as u64);
            true
        } else {
            false
        }
    }

    /// Cap on undelivered [`SideEffect`]s retained between drains — real
    /// streams carry a handful of events and at most one flip per frame, so
    /// hitting this means the embedder stopped draining (warned once).
    const MAX_SIDE_EFFECTS: usize = 1024;

    /// Drain the completion side effects (events, EOP interrupts, flips) the
    /// walk(s) since the last drain executed, in stream order. The embedder
    /// delivers them to the kernel/VideoOut layer; a suspended walk has NOT
    /// recorded anything past its unmet wait, so delivering after every
    /// (partial) walk preserves PM4 submission order.
    #[must_use]
    pub fn take_side_effects(&mut self) -> Vec<SideEffect> {
        std::mem::take(&mut self.side_effects)
    }

    /// Record one completion side effect for the embedder, bounded by
    /// [`Self::MAX_SIDE_EFFECTS`].
    fn record_side_effect(&mut self, effect: SideEffect) {
        if self.side_effects.len() < Self::MAX_SIDE_EFFECTS {
            self.side_effects.push(effect);
        } else if self.first(SkipKey::Note("side_effects_capped")) {
            warn!(
                cap = Self::MAX_SIDE_EFFECTS,
                "completion side effects capped — the embedder is not draining take_side_effects"
            );
        }
    }

    /// Install (or clear) the overriding GPU-timestamp clock for `RELEASE_MEM`
    /// DATA_SEL 3/4 writes — see [`Self::timestamp_source`]. A source
    /// returning `None` falls back to the legacy process-local counter, so an
    /// installed-but-gated-off source is bit-identical to no source.
    pub fn set_timestamp_source(&mut self, source: Option<fn() -> Option<u64>>) {
        self.timestamp_source = source;
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
        // Per-walk chain budget (see `MAX_CHAIN_BUFFERS`). Reset here rather
        // than accumulated: it is a termination bound on one walk, not an
        // accounting total — the honest totals live in `chain_census`.
        self.chain_buffers_followed = 0;
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
                    // Record the reason on EVERY refusal, not only the one that
                    // logs. The warn below is rate-limited to once per
                    // processor, so a title whose every draw is refused used to
                    // leave the embedder with a bare `draws=0` and nothing to
                    // name — see `last_refusal`.
                    self.last_refusal = Some(source.0.clone());
                    if self.first(SkipKey::Note("draw_refused_skip_and_continue")) {
                        warn!(
                            offset,
                            reason = %source,
                            "refused draw/dispatch skipped — continuing the walk so the \
                             completion packets after it still run (never-silent; later \
                             refusals on this processor are counted via refused_draws \
                             and named via last_refusal, not re-logged)"
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
            // A chain raised by the packet just executed runs HERE — after the
            // parent advanced past the chain packet, before the parent's next
            // packet. That is the call semantics both references implement, and
            // it is what makes any register the child writes visible to the rest
            // of the parent buffer. `run_chain` never fails: a fault inside a
            // chained buffer abandons that buffer by name, so the follower can
            // never turn a working title's walk into a structural abort.
            // `follow_chains` first: `pending_chain` can only be `Some` when the
            // follower is on, so a bool test keeps the default path from paying
            // an `Option<ChainRequest>` take on every packet. The walk costs
            // ~2.7-10 ns per packet (`tests/pm4_walk_cost.rs`) and Minecraft's
            // frame is thousands of them.
            if self.follow_chains
                && let Some(request) = self.pending_chain.take()
            {
                self.run_chain(request, sink, mem);
            }
            if let Some(wait) = self.pending_wait.take() {
                return Ok(RunOutcome::Suspended(SuspendedWait {
                    resume_dword: pos,
                    wait,
                }));
            }
        }
        Ok(RunOutcome::Completed)
    }

    /// Walk chained command buffers depth-first from `first`.
    ///
    /// An explicit work-list, not recursion: the walk depth is guest-controlled
    /// (the chain graph lives in guest memory), and a corrupt or hostile stream
    /// must not be able to reach the host stack at all. Both references recurse
    /// — KytyPS5 through an unbounded `m_buffer_stack`, shadPS4 through a nested
    /// coroutine per level — and neither bounds the depth or validates the
    /// target address.
    ///
    /// Every frame is opened through [`Self::open_chain_frame`], which is the
    /// single place a guest-supplied chain address is validated and read. Faults
    /// abandon the offending frame with a named, counted refusal; they never
    /// propagate, and they never touch host memory outside the embedder's
    /// [`GuestMemory`] authority.
    fn run_chain(
        &mut self,
        first: ChainRequest,
        sink: &mut dyn DrawSink,
        mem: Option<&dyn GuestMemory>,
    ) {
        let Some(mem) = mem else {
            self.chain_census.refused_no_memory += 1;
            if self.first(SkipKey::Note("chain_without_guest_memory")) {
                warn!(
                    address = format_args!("{:#x}", first.address),
                    size_dwords = first.size_dwords,
                    offset = first.offset,
                    "IT_INDIRECT_BUFFER target cannot be read without a guest memory reader — \
                     chain not walked, counted (ChainCensus::refused_no_memory)"
                );
            }
            return;
        };
        let mut stack: Vec<ChainFrame> = Vec::new();
        if let Some(frame) = self.open_chain_frame(&stack, first, mem) {
            stack.push(frame);
        }
        while let Some(top) = stack.len().checked_sub(1) {
            let pos = stack[top].pos;
            if pos >= stack[top].words.len() {
                stack.pop();
                continue;
            }
            let cmd_id = stack[top].words[pos];
            let offset = pos as u32;
            if pm4::is_type2(cmd_id) {
                stack[top].pos = pos + 1;
                continue;
            }
            // A desynced chained buffer is not a desynced SUBMISSION: the parent
            // stream's next packet boundary is still known exactly (the chain
            // packet carried its own length). Abandon this buffer only.
            let short = stack[top].words.len() - pos < 2;
            if !pm4::is_type3(cmd_id) || short {
                let (address, size_dwords) = (stack[top].address, stack[top].size_dwords);
                let reason = if short {
                    "a type-3 header is the last dword of the chained buffer — no body follows"
                } else {
                    "not a type-3 PM4 packet"
                };
                self.refuse_chain_frame(address, size_dwords, offset, reason);
                stack.pop();
                continue;
            }
            let outcome = {
                let words = &stack[top].words;
                self.dispatch(cmd_id, &words[pos + 1..], offset, sink, Some(mem))
            };
            let consumed = match outcome {
                Ok(consumed) => consumed,
                // Same policy as the top-level walk: a refused draw is skipped
                // by its encoded length so the completion packets after it in
                // the CHAINED buffer still run.
                Err(CpError::Draw { offset, source }) => {
                    self.refused_draws = self.refused_draws.saturating_add(1);
                    self.last_refusal = Some(source.0.clone());
                    if self.first(SkipKey::Note("chained_draw_refused_skip_and_continue")) {
                        warn!(
                            offset,
                            reason = %source,
                            "refused draw/dispatch inside a CHAINED command buffer skipped — \
                             continuing that buffer's walk (counted via refused_draws, named via \
                             last_refusal)"
                        );
                    }
                    pm4::body_dw(cmd_id)
                }
                Err(error) => {
                    let (address, size_dwords) = (stack[top].address, stack[top].size_dwords);
                    self.refuse_chain_frame(address, size_dwords, offset, &error.to_string());
                    stack.pop();
                    continue;
                }
            };
            let advance = consumed as usize + 1;
            if advance > stack[top].words.len() - pos {
                let (address, size_dwords) = (stack[top].address, stack[top].size_dwords);
                self.refuse_chain_frame(
                    address,
                    size_dwords,
                    offset,
                    "packet declares more dwords than the chained buffer holds",
                );
                stack.pop();
                continue;
            }
            stack[top].pos = pos + advance;
            // A wait inside a chained buffer cannot suspend the stream:
            // `RunOutcome::Suspended` carries a resume point as a dword offset
            // into the SUBMITTED buffer, which cannot name a position inside a
            // buffer the embedder never submitted. Degrade to the same
            // behaviour `run_with_memory` uses for an unmet wait it cannot park
            // — continue past it — but count it, so a title that genuinely
            // needs cross-queue ordering inside a chain shows up as a number
            // rather than as a glitch.
            if let Some(wait) = self.pending_wait.take() {
                self.chain_census.waits_dropped += 1;
                if self.first(SkipKey::Note("chain_wait_cannot_suspend")) {
                    warn!(
                        label = format_args!("{:#x}", wait.address),
                        compare = wait.compare,
                        offset,
                        "unmet wait inside a CHAINED command buffer — continuing past it \
                         (a suspend resume point can only name a dword in the submitted buffer). \
                         Counted (ChainCensus::waits_dropped)"
                    );
                }
            }
            if self.follow_chains
                && let Some(request) = self.pending_chain.take()
                && let Some(frame) = self.open_chain_frame(&stack, request, mem)
            {
                stack.push(frame);
            }
        }
    }

    /// Validate a chain target and read it. The ONLY place a guest-supplied
    /// chain address is dereferenced.
    ///
    /// Refuses, each by name and with its own [`ChainCensus`] counter:
    /// * nesting past [`Self::MAX_CHAIN_DEPTH`];
    /// * more than [`Self::MAX_CHAIN_BUFFERS`] buffers in one top-level walk;
    /// * a cycle — the same ADDRESS already open on the active path, which
    ///   covers a self-chain, an A→B→A loop, and an A→B→A' loop that re-enters
    ///   one base at a different declared size;
    /// * a null, non-dword-aligned, empty, over-long or address-space-wrapping
    ///   target;
    /// * a target the embedder's [`GuestMemory`] will not read in full.
    fn open_chain_frame(
        &mut self,
        stack: &[ChainFrame],
        request: ChainRequest,
        mem: &dyn GuestMemory,
    ) -> Option<ChainFrame> {
        let ChainRequest {
            offset,
            address,
            size_dwords,
        } = request;
        if stack.len() >= Self::MAX_CHAIN_DEPTH {
            self.chain_census.refused_depth += 1;
            if self.first(SkipKey::Note("chain_depth_exceeded")) {
                warn!(
                    address = format_args!("{address:#x}"),
                    size_dwords,
                    offset,
                    depth = stack.len(),
                    limit = Self::MAX_CHAIN_DEPTH,
                    "IT_INDIRECT_BUFFER chain nests deeper than the limit — target refused and \
                     counted (ChainCensus::refused_depth); the walk does not recurse into it"
                );
            }
            return None;
        }
        if self.chain_buffers_followed >= Self::MAX_CHAIN_BUFFERS {
            self.chain_census.refused_budget += 1;
            if self.first(SkipKey::Note("chain_buffer_budget_exhausted")) {
                warn!(
                    address = format_args!("{address:#x}"),
                    size_dwords,
                    offset,
                    limit = Self::MAX_CHAIN_BUFFERS,
                    "this walk has already followed the maximum number of chained command \
                     buffers — target refused and counted (ChainCensus::refused_budget)"
                );
            }
            return None;
        }
        // Keyed on the ADDRESS alone, not on `(address, size)`. A buffer
        // re-entered on the active path is a cycle whatever length the second
        // packet claims for it, and keying on the pair leaves an escape hatch:
        // A→B→A' where A' names the same base with size+1 is not a repeat of the
        // pair, so a stream could walk ~2^20 distinct "sizes" of one buffer. The
        // depth bound and `MAX_CHAIN_BUFFERS` still force termination there, but
        // refusing on the address is both stricter and simpler, and no real
        // stream re-enters one base at two lengths on one path.
        if stack.iter().any(|frame| frame.address == address) {
            self.chain_census.refused_cycle += 1;
            if self.first(SkipKey::Note("chain_cycle")) {
                warn!(
                    address = format_args!("{address:#x}"),
                    size_dwords,
                    offset,
                    depth = stack.len(),
                    "IT_INDIRECT_BUFFER chains back into a buffer already open on this path — \
                     cycle refused and counted (ChainCensus::refused_cycle)"
                );
            }
            return None;
        }
        let bytes = u64::from(size_dwords) * 4;
        let wraps = address.checked_add(bytes).is_none();
        // `%` rather than `u64::is_multiple_of`: this crate's MSRV is 1.85 and
        // that method is stable only since 1.87.
        if address == 0
            || address % 4 != 0
            || size_dwords == 0
            || size_dwords > Self::MAX_CHAIN_DWORDS
            || wraps
        {
            self.chain_census.refused_malformed += 1;
            if self.first(SkipKey::Note("chain_target_malformed")) {
                warn!(
                    address = format_args!("{address:#x}"),
                    size_dwords,
                    offset,
                    "IT_INDIRECT_BUFFER target is null, not DWORD-aligned, empty, over-long or \
                     wraps the address space — refused and counted \
                     (ChainCensus::refused_malformed) WITHOUT reading it"
                );
            }
            return None;
        }
        // The whole `[address, address + size_dwords * 4)` range must come back
        // in one read: the embedder's authority validates the full extent, so a
        // partially-mapped chain target is refused rather than half-walked.
        let Some(words) = mem.read_command_dwords(address, size_dwords) else {
            self.chain_census.refused_unreadable += 1;
            if self.first(SkipKey::Note("chain_target_unreadable")) {
                warn!(
                    address = format_args!("{address:#x}"),
                    end = format_args!("{:#x}", address.saturating_add(bytes)),
                    size_dwords,
                    offset,
                    "IT_INDIRECT_BUFFER target is not fully readable guest memory — refused and \
                     counted (ChainCensus::refused_unreadable); no host memory was touched \
                     outside the guest-memory authority"
                );
            }
            return None;
        };
        // A short read would silently truncate the child's command stream.
        if words.len() != size_dwords as usize {
            self.chain_census.refused_unreadable += 1;
            if self.first(SkipKey::Note("chain_target_short_read")) {
                warn!(
                    address = format_args!("{address:#x}"),
                    size_dwords,
                    got = words.len(),
                    offset,
                    "IT_INDIRECT_BUFFER target read returned fewer dwords than the packet \
                     declared — refused and counted (ChainCensus::refused_unreadable)"
                );
            }
            return None;
        }
        self.chain_buffers_followed = self.chain_buffers_followed.saturating_add(1);
        self.chain_census.followed += 1;
        self.chain_census.followed_dwords += u64::from(size_dwords);
        Some(ChainFrame {
            address,
            size_dwords,
            words,
            pos: 0,
        })
    }

    /// Abandon one chained buffer with a named, counted refusal.
    fn refuse_chain_frame(&mut self, address: u64, size_dwords: u32, offset: u32, reason: &str) {
        self.chain_census.refused_malformed += 1;
        if self.first(SkipKey::Note("chain_frame_abandoned")) {
            warn!(
                address = format_args!("{address:#x}"),
                size_dwords,
                offset,
                reason,
                "chained command buffer abandoned mid-walk — counted \
                 (ChainCensus::refused_malformed). The SUBMITTED buffer keeps walking: its next \
                 packet boundary is still known, because the chain packet carried its own length"
            );
        }
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
        // ONE branch for the whole census: classification, counting and the
        // clock all live behind it. Measured on the synthetic probe, an
        // unconditional class-and-count added 1-2.5 ns to an
        // `IT_NOP`/`SET_CONTEXT_REG` packet whose whole cost is ~10 ns, so a
        // diagnostic must not tax the default path.
        let clock = self.walk_clock();
        // Per-PACKET scope: the ranges this packet writes, never the previous
        // one's. Cheap (a `clear` on an already-empty Vec) for the overwhelming
        // majority of packets, which write nothing.
        self.packet_guest_writes.clear();
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
            // The AGC multi-instanced indexed draw Raeen's own HLE emits. See
            // cp_op_draw_index_multi_instanced.
            pm4::IT_DISPATCH_DRAW_PREAMBLE => {
                self.cp_op_draw_index_multi_instanced(cmd_id, body, offset, sink)
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
            // Chained command buffers. Decoded and CENSUSED unconditionally;
            // walked only when the follower is enabled. See
            // `cp_op_indirect_buffer` — before it existed these two opcodes
            // landed in the anonymous unknown-opcode arm below, so a title that
            // links its frame through Jump/Branch produced one rate-limited
            // "unknown PM4 opcode" line and no evidence at all.
            pm4::IT_INDIRECT_BUFFER | pm4::IT_INDIRECT_BUFFER_CNST => {
                self.cp_op_indirect_buffer(cmd_id, body, offset, mem)
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
            // The kernel-event side effect: recorded in stream order for the
            // embedder to deliver (it owns the equeue state) — see
            // [`SideEffect::EventWrite`].
            pm4::IT_EVENT_WRITE => self.cp_op_event_write(cmd_id, body, offset),
            // These carry no guest-memory completion label on the RDNA2/AGC
            // draw path (ACQUIRE_MEM/CLEAR_STATE/etc. are cache/state
            // ops a draw never observes). Consumed by encoded length.
            pm4::IT_ACQUIRE_MEM
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
            // Draw opcodes `raeen-gpu`'s `agc::decode_submission` COUNTS
            // (`draw_packets`) that this processor cannot translate. Falling to
            // the default arm below made them the worst kind of drop: the
            // submission reported a draw, the walk reported neither a draw nor a
            // skip nor a refusal, and the only trace was one rate-limited
            // "unknown PM4 opcode" line — the same shape as the Dead Cells
            // `draws=0` blocker. A NAMED, COUNTED refusal instead, so the drop
            // lands in `refused_draws` / `last_refusal` like every other
            // untranslatable draw.
            //
            // Refused rather than implemented deliberately: no reference walks
            // either body. KytyPS5's opcode table (pm4Dispatch.cpp L212) wires
            // only `IT_DISPATCH_DRAW_PREAMBLE` (0x3A), not `IT_DISPATCH_DRAW`
            // (0x8D); shadPS4 names `DrawIndexMultiAuto` (0x30) in
            // `pm4_opcodes.h` but its liverpool `switch` has no case for it; and
            // Mesa's `ac_gather_context_rolls` only classifies 0x30 as
            // context-busy without decoding a body. Guessing a body layout would
            // issue a WRONG draw that looks like a working one — strictly worse
            // than an honest counted refusal. Replace this arm with a real
            // handler the moment a title in evidence emits either opcode; the
            // refusal count and reason are what will show that it does.
            pm4::IT_DRAW_INDEX_MULTI_AUTO | pm4::IT_DISPATCH_DRAW => {
                let name = if op == pm4::IT_DRAW_INDEX_MULTI_AUTO {
                    "IT_DRAW_INDEX_MULTI_AUTO"
                } else {
                    "IT_DISPATCH_DRAW"
                };
                // Once per distinct opcode per processor, like the unknown-op
                // arm — the refusal warn in `run_resumable` is rate-limited
                // across ALL refusals, so without this a processor that already
                // refused something would never log this opcode's name.
                if self.first(SkipKey::Op(op.0)) {
                    warn!(
                        cmd_id = format_args!("{cmd_id:#010x}"),
                        op = format_args!("{:#04x}", op.0),
                        name,
                        offset,
                        "draw opcode counted by the AGC decoder has no command-processor \
                         handler — refused and counted (refused_draws), not silently skipped"
                    );
                }
                Err(CpError::Draw {
                    offset,
                    source: DrawError(format!(
                        "{name} ({:#04x}) is counted as a draw by the AGC decoder but has no \
                         command-processor handler — packet skipped by its encoded length",
                        op.0
                    )),
                })
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
        if let Some(at) = clock {
            let class = Self::packet_class(cmd_id, op);
            self.census.record(
                class,
                class.is_timed().then_some(at),
                result.as_ref().ok().copied(),
            );
        }
        // Notify on the write-capable opcodes (even when they wrote nothing, so
        // a sink may keep a blanket-clear policy) AND on any packet that
        // actually wrote guest memory, so a future write site can never bypass
        // invalidation silently. `packet_guest_writes` names the exact extent:
        // a completion label written next to a texture must not throw away the
        // analysis of every shader in the frame.
        if (guest_memory_write_boundary || !self.packet_guest_writes.is_empty()) && result.is_ok() {
            let at = self.walk_clock();
            sink.guest_memory_write_boundary(&self.packet_guest_writes);
            if self.time_walk {
                self.census.boundaries += 1;
                if let Some(at) = at {
                    self.census.boundary_ns += at.elapsed().as_nanos() as u64;
                }
            }
        }
        result
    }

    /// Which [`WalkCensus`] bucket a packet belongs to.
    ///
    /// Classified from the header alone — one `matches!` chain per packet, no
    /// change to the dispatch table itself, so the walk's behaviour is
    /// untouched.
    fn packet_class(cmd_id: u32, op: pm4::ItOp) -> PacketClass {
        match op {
            pm4::IT_SET_CONTEXT_REG
            | pm4::IT_SET_SH_REG
            | pm4::IT_SET_UCONFIG_REG
            | pm4::IT_SET_UCONFIG_REG_INDEX
            | pm4::IT_SET_CONFIG_REG => PacketClass::RegWrite,
            pm4::IT_DRAW_INDEX_AUTO
            | pm4::IT_DRAW_INDEX_2
            | pm4::IT_DRAW_INDEX_OFFSET_2
            | pm4::IT_DRAW_INDIRECT
            | pm4::IT_DRAW_INDIRECT_MULTI
            | pm4::IT_DRAW_INDEX_INDIRECT
            | pm4::IT_DRAW_INDEX_INDIRECT_MULTI
            // The AGC multi-instanced indexed draw — translated, not refused.
            | pm4::IT_DISPATCH_DRAW_PREAMBLE
            // Refused, not translated (see the unimplemented-draw arm in
            // `dispatch`) — but still draw PACKETS, and `decode_submission`
            // counts both in `draw_packets`. Classing them Inert would make the
            // census disagree with the submission count all over again.
            | pm4::IT_DRAW_INDEX_MULTI_AUTO
            | pm4::IT_DISPATCH_DRAW => PacketClass::Draw,
            pm4::IT_DISPATCH_DIRECT | pm4::IT_DISPATCH_INDIRECT => PacketClass::Dispatch,
            pm4::IT_WRITE_DATA | pm4::IT_RELEASE_MEM | pm4::IT_DMA_DATA => PacketClass::Write,
            pm4::IT_WAIT_REG_MEM => PacketClass::Wait,
            pm4::IT_INDIRECT_BUFFER | pm4::IT_INDIRECT_BUFFER_CNST => PacketClass::Indirect,
            // The AGC dialect rides on `IT_NOP`, discriminated by the R code.
            pm4::IT_NOP => match pm4::r_code(cmd_id) {
                pm4::R_DRAW_INDEX | pm4::R_DRAW_INDEX_AUTO => PacketClass::Draw,
                pm4::R_DISPATCH_DIRECT => PacketClass::Dispatch,
                pm4::R_WRITE_DATA | pm4::R_RELEASE_MEM | pm4::R_DMA_DATA => PacketClass::Write,
                pm4::R_WAIT_MEM_32 | pm4::R_WAIT_MEM_64 | pm4::R_WAIT_FLIP_DONE => {
                    PacketClass::Wait
                }
                pm4::R_CX_REGS_INDIRECT | pm4::R_SH_REGS_INDIRECT | pm4::R_UC_REGS_INDIRECT => {
                    PacketClass::Indirect
                }
                _ => PacketClass::Inert,
            },
            _ => PacketClass::Inert,
        }
    }

    /// Deepest chain nesting [`Self::run_chain`] will walk, below the submitted
    /// buffer. A malformed or hostile stream must not be able to grow the
    /// work-list without bound; neither reference bounds this at all
    /// (KytyPS5 `ProcessIndirectBuffer` pushes onto `m_buffer_stack` freely,
    /// shadPS4 spawns a nested coroutine per level).
    ///
    /// 8 is generous against measured practice: PAL-style chunked command
    /// streams and the AGC Jump/Branch forms nest one or two levels (a frame
    /// buffer chaining a pass buffer chaining a state block).
    pub const MAX_CHAIN_DEPTH: usize = 8;

    /// Chained buffers one top-level walk will follow in total.
    ///
    /// The depth bound alone does not guarantee termination: a chain graph that
    /// never repeats a buffer on any single path is a DAG, and a DAG can still
    /// be walked exponentially in its depth. This is the flat ceiling that makes
    /// termination unconditional.
    pub const MAX_CHAIN_BUFFERS: u64 = 4096;

    /// Largest chain target this processor will read: the PM4 `IB_SIZE` field is
    /// 20 bits, so this is the field's own maximum (~4 MiB of command stream).
    /// A larger value cannot have come out of a well-formed packet.
    const MAX_CHAIN_DWORDS: u32 = 0x000f_ffff;

    /// `IT_INDIRECT_BUFFER` (0x3F) / `IT_INDIRECT_BUFFER_CNST` (0x33) — a
    /// chained command buffer.
    ///
    /// Always decodes the packet into the [`ChainCensus`]; walks the target only
    /// when [`Self::set_follow_chains`] is on, by raising [`Self::pending_chain`]
    /// for the walk loop to drain immediately after this packet.
    ///
    /// **CALL, not jump** — the parent buffer resumes at the packet after this
    /// one once the child completes. Evidence, from both references that
    /// implement it:
    ///
    /// * KytyPS5 `CpOpIndirectBuffer` (pm4Handlers.cpp L2569-2612) calls
    ///   `cp.ProcessIndirectBuffer(...)` and then `return 3` — the parent cursor
    ///   advances past the 4-dword packet and keeps walking. `ProcessIndirectBuffer`
    ///   (graphicsRun.cpp L625) pushes the child onto `m_buffer_stack` and runs
    ///   `ProcessPm4(execution, stop_depth)` down to the depth it started at, so
    ///   control returns to the parent frame by construction.
    /// * shadPS4 `liverpool.cpp` L830 runs a nested `ProcessGraphics` task to
    ///   completion, `break`s, and its loop then advances by the chain packet's
    ///   own `NumWords() + 1`.
    ///
    /// The control dword's bit 20 is AMD's `CHAIN` flag (shadPS4
    /// `PM4CmdIndirectBuffer::chain`, pm4_cmds.h L881), which on hardware makes
    /// the transfer a jump — the parent is abandoned. Neither reference honours
    /// it, and KytyPS5's own logging of observed PS5 control values
    /// (`control & 0x0fe00000` expected `0x0f200000`, i.e. bit 21 plus
    /// `VMID = 0xf`) shows bit 20 CLEAR in the streams it has seen. So this
    /// implements the call form both references implement, and merely COUNTS
    /// (`chain_bit_set`) plus names a set chain bit rather than guessing a jump
    /// — a wrong guess in that direction would drop the parent's remaining
    /// draws, which is the exact bug class this handler exists to fix.
    ///
    /// `IT_INDIRECT_BUFFER_CNST` is deliberately NOT followed: it addresses the
    /// **constant engine** ring, a separate queue with its own register shadow.
    /// shadPS4 handles 0x33 only inside `ProcessCeUpdate` (liverpool.cpp L195),
    /// never in the graphics or compute walk, and KytyPS5 refuses it outright
    /// (`EXIT_NOT_IMPLEMENTED` on any opcode but 0x3F, pm4Handlers.cpp L2572).
    /// Feeding a CE buffer to the graphics processor would execute it against
    /// the wrong register file.
    fn cp_op_indirect_buffer(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let body_dw = pm4::body_dw(cmd_id);
        if pm4::op(cmd_id) == pm4::IT_INDIRECT_BUFFER_CNST {
            self.chain_census.const_packets += 1;
            self.record_chain_sample(ChainSample {
                offset,
                address: 0,
                size_dwords: 0,
                control: 0,
                form: ChainForm::Const,
            });
            if self.first(SkipKey::Note("indirect_buffer_const_not_followed")) {
                warn!(
                    cmd_id = format_args!("{cmd_id:#010x}"),
                    offset,
                    "IT_INDIRECT_BUFFER_CNST addresses the CONSTANT-ENGINE ring, which this \
                     processor does not model — counted (ChainCensus::const_packets) and \
                     consumed by its encoded length, never walked against the graphics \
                     register file"
                );
            }
            return Ok(body_dw);
        }
        // KytyPS5 discriminates the two IT_INDIRECT_BUFFER layouts purely by
        // packet length (pm4Handlers.cpp L2574/L2578): 14 dwords total = the
        // conditional branch, 4 = the unconditional chain.
        if body_dw == 13 {
            return self.cp_op_chain_branch(cmd_id, body, offset, mem);
        }
        if body_dw != 3 {
            self.chain_census.refused_malformed += 1;
            if self.first(SkipKey::Note("indirect_buffer_bad_length")) {
                warn!(
                    cmd_id = format_args!("{cmd_id:#010x}"),
                    body_dw,
                    offset,
                    "IT_INDIRECT_BUFFER with neither the 4-dword chain nor the 14-dword branch \
                     length — refused and counted (ChainCensus::refused_malformed), consumed by \
                     its encoded length"
                );
            }
            return Ok(body_dw);
        }
        let lo = Self::body_at(body, 0, offset)?;
        let hi = Self::body_at(body, 1, offset)?;
        let control = Self::body_at(body, 2, offset)?;
        // shadPS4 `PM4CmdIndirectBuffer`: `ibase_hi` is 16 bits, `ib_size` 20.
        let address = u64::from(lo) | (u64::from(hi & 0xffff) << 32);
        let size_dwords = control & Self::MAX_CHAIN_DWORDS;
        self.chain_census.jump_packets += 1;
        self.chain_census.target_dwords += u64::from(size_dwords);
        if control & (1 << 20) != 0 {
            self.chain_census.chain_bit_set += 1;
            if self.first(SkipKey::Note("indirect_buffer_chain_bit")) {
                warn!(
                    control = format_args!("{control:#010x}"),
                    offset,
                    "IT_INDIRECT_BUFFER has the CHAIN control bit set — hardware would JUMP \
                     (abandoning the rest of this buffer); both references and this processor \
                     treat it as a CALL. Counted (ChainCensus::chain_bit_set)"
                );
            }
        }
        self.record_chain_sample(ChainSample {
            offset,
            address,
            size_dwords,
            control,
            form: ChainForm::Jump,
        });
        self.request_chain(offset, address, size_dwords);
        Ok(body_dw)
    }

    /// The 14-dword conditional chain: `IT_INDIRECT_BUFFER` carrying a masked
    /// 64-bit compare and two targets.
    ///
    /// Body layout is KytyPS5 `CpOpBranch` (pm4Handlers.cpp L2140-2156), and is
    /// exactly what `raeen-hle`'s `sceAgcCbBranch` (`hle_cb_branch`) already
    /// emits: `[0]` mode | function << 8, `[1..3]` compare address, `[3..5]`
    /// mask, `[5..7]` reference, `[7..9]` then-target, `[9]` then size,
    /// `[10..12]` else-target, `[12]` else size.
    fn cp_op_chain_branch(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<u32, CpError> {
        let body_dw = pm4::body_dw(cmd_id);
        self.chain_census.branch_packets += 1;
        let selector = Self::body_at(body, 0, offset)?;
        let mode = selector & 0x3;
        let function = (selector >> 8) & 0x7;
        let compare_address = u64::from(Self::body_at(body, 1, offset)? & 0xffff_fff8)
            | (u64::from(Self::body_at(body, 2, offset)?) << 32);
        let mask = u64::from(Self::body_at(body, 3, offset)?)
            | (u64::from(Self::body_at(body, 4, offset)?) << 32);
        let reference = u64::from(Self::body_at(body, 5, offset)?)
            | (u64::from(Self::body_at(body, 6, offset)?) << 32);
        let then_address = u64::from(Self::body_at(body, 7, offset)? & 0xffff_fffc)
            | (u64::from(Self::body_at(body, 8, offset)?) << 32);
        let then_dwords = Self::body_at(body, 9, offset)? & Self::MAX_CHAIN_DWORDS;
        let else_address = u64::from(Self::body_at(body, 10, offset)? & 0xffff_fffc)
            | (u64::from(Self::body_at(body, 11, offset)?) << 32);
        let else_dwords = Self::body_at(body, 12, offset)? & Self::MAX_CHAIN_DWORDS;
        self.chain_census.target_dwords += u64::from(then_dwords);
        let spec = WaitSpec {
            address: compare_address,
            mask,
            reference,
            compare: function,
            is_64: true,
        };
        // The compare label lives in guest memory. Without a reader, or with an
        // unreadable label, there is no honest way to pick a branch — refuse by
        // name rather than guessing the then-branch (which would execute work
        // the title asked to be skipped).
        let Some(mem) = mem else {
            self.chain_census.refused_no_memory += 1;
            if self.first(SkipKey::Note("chain_branch_without_guest_memory")) {
                warn!(
                    address = format_args!("{compare_address:#x}"),
                    offset,
                    "conditional IT_INDIRECT_BUFFER needs the compare label but no guest memory \
                     reader is installed — neither branch taken, counted \
                     (ChainCensus::refused_no_memory)"
                );
            }
            return Ok(body_dw);
        };
        let Some(value) = spec.read_label(mem) else {
            self.chain_census.refused_unreadable += 1;
            if self.first(SkipKey::Note("chain_branch_unreadable_label")) {
                warn!(
                    address = format_args!("{compare_address:#x}"),
                    offset,
                    "conditional IT_INDIRECT_BUFFER compare label is unreadable — neither branch \
                     taken, counted (ChainCensus::refused_unreadable)"
                );
            }
            return Ok(body_dw);
        };
        // Mode 1 = then-only, mode 2 = then/else (KytyPS5 refuses any other).
        let (address, size_dwords) = if spec.satisfied_by(value) {
            (then_address, then_dwords)
        } else if mode == 2 {
            (else_address, else_dwords)
        } else {
            (0, 0)
        };
        self.record_chain_sample(ChainSample {
            offset,
            address,
            size_dwords,
            control: selector,
            form: ChainForm::Branch,
        });
        if size_dwords != 0 {
            self.request_chain(offset, address, size_dwords);
        }
        Ok(body_dw)
    }

    fn record_chain_sample(&mut self, sample: ChainSample) {
        if self.chain_census.samples.len() < ChainCensus::MAX_SAMPLES {
            self.chain_census.samples.push(sample);
        }
    }

    /// Ask the running walk loop to follow `address`, or record that the
    /// follower is off. Never dereferences anything — validation happens in
    /// [`Self::open_chain_frame`], at the point the target would be read.
    fn request_chain(&mut self, offset: u32, address: u64, size_dwords: u32) {
        if !self.follow_chains {
            self.chain_census.refused_disabled += 1;
            if self.first(SkipKey::Note("indirect_buffer_follower_disabled")) {
                warn!(
                    address = format_args!("{address:#x}"),
                    size_dwords,
                    offset,
                    "IT_INDIRECT_BUFFER chain target NOT walked — the chain follower is off. \
                     Counted (ChainCensus::refused_disabled); the target's draws, dispatches and \
                     completion packets do not run"
                );
            }
            return;
        }
        self.pending_chain = Some(ChainRequest {
            offset,
            address,
            size_dwords,
        });
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
            // The embedded VideoOut flip: recorded in stream order for the
            // embedder to deliver, so a flip behind an unmet wait cannot
            // become visible early. Field layout mirrors the eager decoder
            // (`decode_submission`): body `[handle, index, mode, arg_lo,
            // arg_hi, …]`, honoured only at the modeled >= 6-dword length.
            pm4::R_FLIP => {
                if pm4::body_dw(cmd_id) >= 5 {
                    let flip_arg = u64::from(Self::body_at(body, 3, offset)?)
                        | (u64::from(Self::body_at(body, 4, offset)?) << 32);
                    let effect = SideEffect::Flip {
                        video_out_handle: Self::body_at(body, 0, offset)?,
                        display_buffer_index: Self::body_at(body, 1, offset)?,
                        flip_mode: Self::body_at(body, 2, offset)?,
                        flip_arg,
                    };
                    self.record_side_effect(effect);
                }
                Ok(pm4::body_dw(cmd_id))
            }
            // Sync ops: consumed, not honoured — a draw never observes them.
            pm4::R_ACQUIRE_MEM | pm4::R_WAIT_FLIP_DONE => {
                if self.first(SkipKey::Custom(r.0)) {
                    warn!(
                        cmd_id = format_args!("{cmd_id:#010x}"),
                        r = r.0,
                        offset,
                        "AGC sync packet consumed without effect"
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
            Some(bytes) if self.guest_write(mem, dst, &bytes) => {
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
            if self.guest_write(mem, dst, &bytes) {
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
            Some(bytes) if self.guest_write(mem, dst, &bytes) => {
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
    /// Execute a standard `IT_EVENT_WRITE` packet by recording its kernel-event
    /// side effect for the embedder ([`SideEffect::EventWrite`]) in stream
    /// order. Event-id extraction mirrors the eager decoder: the low 6 bits of
    /// the first body dword, honoured only at the modeled >= 2-dword length.
    fn cp_op_event_write(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
    ) -> Result<u32, CpError> {
        let body_len = pm4::body_dw(cmd_id);
        if body_len >= 1 {
            let event_id = Self::body_at(body, 0, offset)? & 0x3f;
            self.record_side_effect(SideEffect::EventWrite { event_id });
        }
        Ok(body_len)
    }

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
            if self.guest_write(mem, addr, &value.to_le_bytes()) {
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
        if !standard {
            // AGC-form end-of-pipe interrupt request (byte at bits 31:24),
            // recorded for the embedder BEFORE the destination gates: an
            // interrupt-only completion carries no memory write (dst == 0).
            // Mirrors the eager decoder, which likewise extracts interrupts
            // only from the AGC form.
            let interrupt = (control >> 24) & 0xFF;
            if interrupt != 0 {
                self.record_side_effect(SideEffect::EopInterrupt {
                    context_id: body.get(6).copied().unwrap_or(0),
                });
            }
        }
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
                if self.guest_write(mem, dst, &value.to_le_bytes()) {
                    self.record_produced(dst, u64::from(value));
                }
            }
            2 => {
                let value = u64::from(Self::body_at(body, 4, offset)?)
                    | (u64::from(Self::body_at(body, 5, offset)?) << 32);
                if self.guest_write(mem, dst, &value.to_le_bytes()) {
                    self.record_produced(dst, value);
                }
            }
            3 | 4 => {
                // Hardware samples the GPU clock at the release point; the guest
                // uses the nonzero value as submit-completion state. A process-
                // monotonic counter is nonzero and strictly increasing, which is
                // what a "became nonzero" / ">= earlier sample" poll needs. Not
                // recorded for the equality latch (it is a counter, not a
                // specific reference the waiter compares equal to). An
                // installed timestamp source overrides the counter (the
                // embedder's unified GPU clock — see `set_timestamp_source`);
                // a source returning `None` is the legacy counter, exactly.
                let ts = self
                    .timestamp_source
                    .and_then(|source| source())
                    .unwrap_or_else(next_release_timestamp);
                let _ = self.guest_write(mem, dst, &ts.to_le_bytes());
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

    /// Body dwords in the only modeled `IT_DISPATCH_DRAW_PREAMBLE` layout —
    /// KytyPS5's `0xC0073A00`, i.e. a 9-dword total packet.
    const DISPATCH_DRAW_PREAMBLE_BODY_DW: u32 = 8;

    /// `IT_DISPATCH_DRAW_PREAMBLE` (0x3A) — the AGC multi-instanced indexed
    /// draw.
    ///
    /// KytyPS5 routes this opcode to the *same* handler as `IT_DRAW_INDEX_2`
    /// (`MakeOpcodeDispatchTable`, pm4Dispatch.cpp L212) and discriminates the
    /// two layouts on the exact `cmd_id`: `CpOpDrawIndex` (pm4Handlers.cpp
    /// L2276-2297) decodes `0xC0073A00` as
    ///
    /// ```text
    /// [index_count, addr_lo, addr_hi, max_instance_count,
    ///  obj_lo, obj_hi, instance_count, flags]
    /// ```
    ///
    /// and returns 8 body dwords.
    ///
    /// Unlike every other opcode here this is not a speculative decode of some
    /// title's stream: Raeen's own `sceAgcDcbDrawIndexMultiInstanced`
    /// (`raeen-hle::hle_dcb_draw_index_multi_instanced`) emits exactly this
    /// packet, so the emitter and this handler are the two ends of one in-tree
    /// contract —
    /// `multi_instanced_draw_emission_reaches_the_command_processor` pins it
    /// from the emitter's side. Before this handler existed the opcode fell to
    /// the anonymous unknown-opcode arm and every such draw vanished with no
    /// counter and no named reason anywhere.
    ///
    /// **Degradation (named, deliberate):** `instance_count` and the
    /// `object_ids` buffer are not forwarded — [`IndexedDraw`] carries neither
    /// and no sink implements instanced draws — so instances `2..N` are not
    /// rendered. The first instance still lands, which is a visual glitch
    /// rather than a dropped draw.
    fn cp_op_draw_index_multi_instanced(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
        sink: &mut dyn DrawSink,
    ) -> Result<u32, CpError> {
        let consumed = pm4::body_dw(cmd_id);
        // KytyPS5 `EXIT`s on any cmd_id but 0xC0073A00. The resilience policy
        // REFUSES instead — a short packet is still a draw being dropped, so it
        // must land in `refused_draws` / `last_refusal` rather than be skipped
        // anonymously. The guard is also load-bearing for memory safety of the
        // reads below: `body_at` bounds-checks against the rest of the BUFFER,
        // not against this packet, so without it a truncated 0x3A would silently
        // read the following packets' dwords as its own draw fields.
        if consumed < Self::DISPATCH_DRAW_PREAMBLE_BODY_DW {
            return Err(CpError::Draw {
                offset,
                source: DrawError(format!(
                    "IT_DISPATCH_DRAW_PREAMBLE with a {consumed}-dword body — the only \
                     modeled layout is KytyPS5's {}-dword 0xC0073A00 form, so the draw \
                     fields cannot be located",
                    Self::DISPATCH_DRAW_PREAMBLE_BODY_DW
                )),
            });
        }

        let index_count = Self::body_at(body, 0, offset)?;
        let lo = Self::body_at(body, 1, offset)?;
        let hi = Self::body_at(body, 2, offset)?;
        let index_addr = u64::from(lo) | (u64::from(hi) << 32);
        let instance_count = Self::body_at(body, 6, offset)?;
        let flags = Self::body_at(body, 7, offset)?;

        if instance_count > 1 && self.first(SkipKey::Note("multi_instanced_draw_degradation")) {
            warn!(
                instance_count,
                index_count,
                offset,
                "multi-instanced indexed draw degraded to ONE instance — IndexedDraw carries \
                 no instance count and no sink implements instanced draws, so instances 2..N \
                 and the object-id buffer are dropped (the draw itself still lands)"
            );
        }

        let draw = IndexedDraw {
            index_type_and_size: self.index_type_and_size,
            index_count,
            index_addr,
            flags,
            // KytyPS5 passes the same `type` argument here as for the raw
            // IT_DRAW_INDEX_2 form: `cp.DrawIndex(.., 0, 1, ..)`
            // (pm4Handlers.cpp L2296 vs L2311).
            index_type: 1,
        };
        sink.draw_index(&self.ctx, &self.ucfg, &self.sh_ctx, &draw)
            .map_err(|source| CpError::Draw { offset, source })?;

        Ok(consumed)
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
                    fmask_data_compression_disable: pm4::field(value, f::FMASK_COMPRESSION_DISABLE)
                        != 0,
                    fmask_one_frag_mode: pm4::field(value, f::FMASK_COMPRESS_1FRAG_ONLY) != 0,
                    dcc_compression_enable: pm4::field(value, f::DCC_ENABLE) != 0,
                    cmask_tile_mode_neo: pm4::field(value, f::CMASK_ADDR_TYPE),
                    neo_mode: pm4::field(value, f::ALT_TILE_MODE) != 0,
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

            // ---- Remaining per-slot CB_COLOR{n} sub-registers (stride 15) ----
            // Kyty: g_hw_ctx_indirect_func loops (GraphicsRun.cpp L3522-3700).
            // VIEW and the CLEAR_WORDs are live feature state (array-slice
            // window / fast-clear colour); ATTRIB and the DCC/CMASK/FMASK
            // block are compression metadata — decoded into named fields so
            // they never log as unknown, but deliberately not emulated.
            r if (pm4::CB_COLOR0_VIEW..=pm4::CB_COLOR7_VIEW).contains(&r)
                && (r - pm4::CB_COLOR0_VIEW) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_VIEW, pm4::CB_COLOR_SLOT_STRIDE);
                use pm4::cb_color_view as f;
                self.ctx.render_targets[slot].view = crate::hw_regs::ColorView {
                    base_array_slice_index: pm4::field(value, f::SLICE_START),
                    last_array_slice_index: pm4::field(value, f::SLICE_MAX),
                    current_mip_level: pm4::field(value, f::MIP_LEVEL),
                };
            }

            r if (pm4::CB_COLOR0_ATTRIB..=pm4::CB_COLOR7_ATTRIB).contains(&r)
                && (r - pm4::CB_COLOR0_ATTRIB) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_ATTRIB, pm4::CB_COLOR_SLOT_STRIDE);
                use pm4::cb_color_attrib as f;
                self.ctx.render_targets[slot].attrib = crate::hw_regs::ColorAttrib {
                    force_dest_alpha_to_one: pm4::field(value, f::FORCE_DST_ALPHA_1) != 0,
                    tile_mode: pm4::field(value, f::TILE_MODE_INDEX),
                    fmask_tile_mode: pm4::field(value, f::FMASK_TILE_MODE_INDEX),
                    num_samples: pm4::field(value, f::NUM_SAMPLES),
                    num_fragments: pm4::field(value, f::NUM_FRAGMENTS),
                };
            }

            r if (pm4::CB_COLOR0_DCC_CONTROL..=pm4::CB_COLOR7_DCC_CONTROL).contains(&r)
                && (r - pm4::CB_COLOR0_DCC_CONTROL) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_DCC_CONTROL, pm4::CB_COLOR_SLOT_STRIDE);
                use pm4::cb_color_dcc_control as f;
                self.ctx.render_targets[slot].dcc = crate::hw_regs::ColorDccControl {
                    overwrite_combiner_disable: pm4::field(value, f::OVERWRITE_COMBINER_DISABLE)
                        != 0,
                    dcc_clear_key_enable: pm4::field(value, f::KEY_CLEAR_ENABLE) != 0,
                    max_uncompressed_block_size: pm4::field(value, f::MAX_UNCOMPRESSED_BLOCK_SIZE),
                    min_compressed_block_size: pm4::field(value, f::MIN_COMPRESSED_BLOCK_SIZE),
                    max_compressed_block_size: pm4::field(value, f::MAX_COMPRESSED_BLOCK_SIZE),
                    color_transform: pm4::field(value, f::COLOR_TRANSFORM),
                    independent_64b_blocks: pm4::field(value, f::INDEPENDENT_64B_BLOCKS) != 0,
                    data_write_on_dcc_clear_to_reg: pm4::field(
                        value,
                        f::ENABLE_CONSTANT_ENCODE_REG_WRITE,
                    ) != 0,
                    independent_128b_blocks: pm4::field(value, f::INDEPENDENT_128B_BLOCKS) != 0,
                };
                note_compression_metadata_ignored();
            }

            r if (pm4::CB_COLOR0_CMASK..=pm4::CB_COLOR7_CMASK).contains(&r)
                && (r - pm4::CB_COLOR0_CMASK) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_CMASK, pm4::CB_COLOR_SLOT_STRIDE);
                let addr = &mut self.ctx.render_targets[slot].cmask.addr;
                *addr &= 0xFFFF_FF00_0000_00FF;
                *addr |= u64::from(value) << 8;
                note_compression_metadata_ignored();
            }

            r if (pm4::CB_COLOR0_CMASK_SLICE..=pm4::CB_COLOR7_CMASK_SLICE).contains(&r)
                && (r - pm4::CB_COLOR0_CMASK_SLICE) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_CMASK_SLICE, pm4::CB_COLOR_SLOT_STRIDE);
                self.ctx.render_targets[slot].cmask_slice.slice_minus1 = value;
                note_compression_metadata_ignored();
            }

            r if (pm4::CB_COLOR0_FMASK..=pm4::CB_COLOR7_FMASK).contains(&r)
                && (r - pm4::CB_COLOR0_FMASK) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_FMASK, pm4::CB_COLOR_SLOT_STRIDE);
                let addr = &mut self.ctx.render_targets[slot].fmask.addr;
                *addr &= 0xFFFF_FF00_0000_00FF;
                *addr |= u64::from(value) << 8;
                note_compression_metadata_ignored();
            }

            r if (pm4::CB_COLOR0_FMASK_SLICE..=pm4::CB_COLOR7_FMASK_SLICE).contains(&r)
                && (r - pm4::CB_COLOR0_FMASK_SLICE) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_FMASK_SLICE, pm4::CB_COLOR_SLOT_STRIDE);
                self.ctx.render_targets[slot].fmask_slice.slice_minus1 = value;
                note_compression_metadata_ignored();
            }

            r if (pm4::CB_COLOR0_CLEAR_WORD0..=pm4::CB_COLOR7_CLEAR_WORD0).contains(&r)
                && (r - pm4::CB_COLOR0_CLEAR_WORD0) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_CLEAR_WORD0, pm4::CB_COLOR_SLOT_STRIDE);
                self.ctx.render_targets[slot].clear_word0.word0 = value;
            }

            r if (pm4::CB_COLOR0_CLEAR_WORD1..=pm4::CB_COLOR7_CLEAR_WORD1).contains(&r)
                && (r - pm4::CB_COLOR0_CLEAR_WORD1) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_CLEAR_WORD1, pm4::CB_COLOR_SLOT_STRIDE);
                self.ctx.render_targets[slot].clear_word1.word1 = value;
            }

            r if (pm4::CB_COLOR0_DCC_BASE..=pm4::CB_COLOR7_DCC_BASE).contains(&r)
                && (r - pm4::CB_COLOR0_DCC_BASE) % pm4::CB_COLOR_SLOT_STRIDE == 0 =>
            {
                let slot = slot_of(pm4::CB_COLOR0_DCC_BASE, pm4::CB_COLOR_SLOT_STRIDE);
                let addr = &mut self.ctx.render_targets[slot].dcc_addr.addr;
                *addr &= 0xFFFF_FF00_0000_00FF;
                *addr |= u64::from(value) << 8;
                note_compression_metadata_ignored();
            }

            // ---- Gen5 `_EXT` high-address-byte blocks (stride 1) ----
            // Kyty: GraphicsRun.cpp L3609-3688 — the low byte of the value is
            // bits 40..48 of the matching address.
            r if (pm4::CB_COLOR0_BASE_EXT..=pm4::CB_COLOR7_BASE_EXT).contains(&r) => {
                let slot = (r - pm4::CB_COLOR0_BASE_EXT) as usize;
                let addr = &mut self.ctx.render_targets[slot].base.addr;
                *addr &= 0xFFFF_00FF_FFFF_FFFF;
                *addr |= u64::from(value & 0xFF) << 40;
            }
            r if (pm4::CB_COLOR0_CMASK_BASE_EXT..=pm4::CB_COLOR7_CMASK_BASE_EXT).contains(&r) => {
                let slot = (r - pm4::CB_COLOR0_CMASK_BASE_EXT) as usize;
                let addr = &mut self.ctx.render_targets[slot].cmask.addr;
                *addr &= 0xFFFF_00FF_FFFF_FFFF;
                *addr |= u64::from(value & 0xFF) << 40;
                note_compression_metadata_ignored();
            }
            r if (pm4::CB_COLOR0_FMASK_BASE_EXT..=pm4::CB_COLOR7_FMASK_BASE_EXT).contains(&r) => {
                let slot = (r - pm4::CB_COLOR0_FMASK_BASE_EXT) as usize;
                let addr = &mut self.ctx.render_targets[slot].fmask.addr;
                *addr &= 0xFFFF_00FF_FFFF_FFFF;
                *addr |= u64::from(value & 0xFF) << 40;
                note_compression_metadata_ignored();
            }
            r if (pm4::CB_COLOR0_DCC_BASE_EXT..=pm4::CB_COLOR7_DCC_BASE_EXT).contains(&r) => {
                let slot = (r - pm4::CB_COLOR0_DCC_BASE_EXT) as usize;
                let addr = &mut self.ctx.render_targets[slot].dcc_addr.addr;
                *addr &= 0xFFFF_00FF_FFFF_FFFF;
                *addr |= u64::from(value & 0xFF) << 40;
                note_compression_metadata_ignored();
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
                // Gated: this is the single hottest register range in the walk
                // (a title rewrites up to 32 PS user SGPRs per draw for its
                // per-object constants), and an UNCONDITIONAL per-register
                // `debug!` puts a tracing dispatch on every one of them —
                // free-ish at WARN, ruinous the moment anyone runs at DEBUG.
                // The env gate is a cached `OnceLock`, so the diagnostic is
                // still available on request and costs one load otherwise.
                if trace_shader_binds_enabled() {
                    tracing::debug!(
                        id,
                        value = format_args!("{value:#010x}"),
                        "PS user SGPR write"
                    );
                }
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
        /// Every `(base, len)` span reported across all boundaries, in order.
        guest_writes: Vec<(u64, u64)>,
        fail: Option<String>,
    }

    /// (group_xyz, unused, direct_address, dims, mode, tag) recorded per dispatch.
    type RecordedDispatch = ([u32; 3], u32, u64, [u32; 3], u8, u32);

    impl DrawSink for RecordingSink {
        fn guest_memory_write_boundary(&mut self, writes: &[(u64, u64)]) {
            self.guest_memory_write_boundaries += 1;
            self.guest_writes.extend_from_slice(writes);
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

    /// Every CB_COLOR{n} sub-register family must land in the right slot's
    /// named field — nothing in the 0x318..=0x3BF block may fall through to
    /// "unknown context register" any more.
    #[test]
    fn set_context_reg_decodes_cb_color_sub_registers_for_every_slot() {
        for slot in [0u32, 3, 7] {
            let mut cp = CommandProcessor::new();
            let mut sink = RecordingSink::default();
            let s15 = slot * pm4::CB_COLOR_SLOT_STRIDE;
            let view = (2 << 26) | (5 << 13) | 1; // mip 2, slice_max 5, start 1
            let attrib = (1 << 17) | (2 << 15) | (3 << 12) | (9 << 5) | 4;
            let dcc_control = 1 | (1 << 1) | (2 << 2) | (1 << 4) | (1 << 5) | (1 << 9) | (1 << 20);
            let writes: Vec<(u32, u32)> = vec![
                (pm4::CB_COLOR0_VIEW + s15, view),
                (pm4::CB_COLOR0_ATTRIB + s15, attrib),
                (pm4::CB_COLOR0_DCC_CONTROL + s15, dcc_control),
                (pm4::CB_COLOR0_CMASK + s15, 0xAB_CDEF),
                (pm4::CB_COLOR0_CMASK_SLICE + s15, 0x3F),
                (pm4::CB_COLOR0_FMASK + s15, 0x12_3456),
                (pm4::CB_COLOR0_FMASK_SLICE + s15, 0x7F),
                (pm4::CB_COLOR0_CLEAR_WORD0 + s15, 0xFF80_4020),
                (pm4::CB_COLOR0_CLEAR_WORD1 + s15, 0x0000_00FF),
                (pm4::CB_COLOR0_DCC_BASE + s15, 0x77_8899),
                (pm4::CB_COLOR0_CMASK_BASE_EXT + slot, 0xAA),
                (pm4::CB_COLOR0_FMASK_BASE_EXT + slot, 0xBB),
                (pm4::CB_COLOR0_DCC_BASE_EXT + slot, 0xCC),
            ];
            let mut dcb = Vec::new();
            for (reg, value) in writes {
                dcb.extend([header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO), reg, value]);
            }
            cp.run(&dcb, &mut sink).expect("CB sub-register writes");
            let rt = &cp.get_ctx().render_targets[slot as usize];
            assert_eq!(
                (
                    rt.view.base_array_slice_index,
                    rt.view.last_array_slice_index,
                    rt.view.current_mip_level
                ),
                (1, 5, 2),
                "slot {slot} VIEW"
            );
            assert_eq!(
                (
                    rt.attrib.tile_mode,
                    rt.attrib.fmask_tile_mode,
                    rt.attrib.num_samples,
                    rt.attrib.num_fragments,
                    rt.attrib.force_dest_alpha_to_one
                ),
                (4, 9, 3, 2, true),
                "slot {slot} ATTRIB"
            );
            assert!(rt.dcc.overwrite_combiner_disable && rt.dcc.dcc_clear_key_enable);
            assert_eq!(rt.dcc.max_uncompressed_block_size, 2);
            assert_eq!(rt.dcc.min_compressed_block_size, 1);
            assert_eq!(rt.dcc.max_compressed_block_size, 1);
            assert!(rt.dcc.independent_64b_blocks && rt.dcc.independent_128b_blocks);
            // Metadata addresses assemble exactly like colour BASE: low dword
            // shifted by 8, `_EXT` low byte into bits 40..48.
            assert_eq!(rt.cmask.addr, (0xAB_CDEFu64 << 8) | (0xAAu64 << 40));
            assert_eq!(rt.fmask.addr, (0x12_3456u64 << 8) | (0xBBu64 << 40));
            assert_eq!(rt.dcc_addr.addr, (0x77_8899u64 << 8) | (0xCCu64 << 40));
            assert_eq!(rt.cmask_slice.slice_minus1, 0x3F);
            assert_eq!(rt.fmask_slice.slice_minus1, 0x7F);
            assert_eq!(
                (rt.clear_word0.word0, rt.clear_word1.word1),
                (0xFF80_4020, 0x0000_00FF),
                "slot {slot} fast-clear words"
            );
        }
    }

    /// `CB_COLOR{n}_BASE_EXT` carries bits 40..48 of the colour base, exactly
    /// like the depth `_HI` registers (Kyty GraphicsRun.cpp L3609).
    #[test]
    fn set_context_reg_color_base_ext_sets_high_address_byte() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::CB_COLOR0_BASE + 2 * pm4::CB_COLOR_SLOT_STRIDE,
            0x1_0000 >> 8,
            header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::CB_COLOR0_BASE_EXT + 2,
            0xFF12, // only the low byte may land
        ];
        cp.run(&dcb, &mut sink).expect("base + base_ext writes");
        assert_eq!(
            cp.get_ctx().render_targets[2].base.addr,
            0x1_0000 | (0x12u64 << 40)
        );
    }

    /// The INFO decode carries the full Kyty field set, including the FMASK
    /// compression flags and the NEO cmask address type.
    #[test]
    fn set_context_reg_color_info_decodes_compression_flags() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let value = (1u32 << 13) // FAST_CLEAR
            | (1 << 26) // FMASK_COMPRESSION_DISABLE
            | (1 << 27) // FMASK_COMPRESS_1FRAG_ONLY
            | (2 << 29); // CMASK_ADDR_TYPE
        let dcb = vec![
            header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::CB_COLOR0_INFO + pm4::CB_COLOR_SLOT_STRIDE,
            value,
        ];
        cp.run(&dcb, &mut sink).expect("info write");
        let info = &cp.get_ctx().render_targets[1].info;
        assert!(info.cmask_fast_clear_enable);
        assert!(info.fmask_data_compression_disable);
        assert!(info.fmask_one_frag_mode);
        assert_eq!(info.cmask_tile_mode_neo, 2);
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

    /// A counted refusal must also be a NAMED one, for every refusal — not only
    /// the first.
    ///
    /// The Dead Cells black frame was diagnosable in principle and undiagnosable
    /// in practice: every draw was refused for `indexed draw with no index
    /// buffer: addr=0x0`, but the walk's warn is rate-limited to once per
    /// processor and the reason was then dropped on the floor. The embedder saw
    /// `draws=0` with `draw_skips=0` (the sink's own counter — a refusal never
    /// touches it) and concluded the command processor "reports neither a draw
    /// nor a reason". [`CommandProcessor::last_refusal`] closes that: the reason
    /// survives past the one log line, and past a queue reset.
    #[test]
    fn every_refused_draw_leaves_a_recoverable_reason() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink {
            fail: Some("indexed draw with no index buffer: addr=0x0 count=6".into()),
            ..Default::default()
        };
        assert_eq!(cp.refused_draws(), 0);
        assert_eq!(cp.last_refusal(), None, "no refusal yet → nothing to name");

        // Three draws, all refused. Only the FIRST emits a log line.
        let one_draw = [header(3, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO), 3, 0];
        let dcb: Vec<u32> = one_draw
            .iter()
            .chain(one_draw.iter())
            .chain(one_draw.iter())
            .copied()
            .collect();
        cp.run(&dcb, &mut sink)
            .expect("refusals are skipped, not stream faults");

        assert_eq!(cp.refused_draws(), 3, "every refusal counts");
        assert_eq!(
            cp.last_refusal(),
            Some("indexed draw with no index buffer: addr=0x0 count=6"),
            "the reason must outlive the rate-limited warn"
        );
        // A per-frame queue reset must not erase the evidence, exactly as it
        // does not erase `refused_draws`.
        cp.reset();
        assert_eq!(cp.refused_draws(), 3, "reset must not zero the count");
        assert_eq!(
            cp.last_refusal(),
            Some("indexed draw with no index buffer: addr=0x0 count=6"),
            "reset must not erase the reason"
        );
    }

    /// A sink that only accepts indexed draws, so a silent degrade to
    /// `draw_index_auto` (which would lose the index buffer) cannot pass as a
    /// success.
    #[derive(Default)]
    struct IndexOnlySink {
        indexed: Vec<IndexedDraw>,
    }

    impl DrawSink for IndexOnlySink {
        fn draw_index_auto(
            &mut self,
            _ctx: &Context,
            _ucfg: &UserConfig,
            _sh: &Shader,
            _index_count: u32,
            _flags: u32,
        ) -> Result<(), DrawError> {
            Err(DrawError("expected an indexed draw".to_owned()))
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

    /// `IT_DISPATCH_DRAW_PREAMBLE` (0x3A) decodes KytyPS5's `0xC0073A00` layout.
    ///
    /// This opcode is the one Raeen's own HLE emits
    /// (`sceAgcDcbDrawIndexMultiInstanced`), and it had no arm at all: the
    /// packet fell to the anonymous unknown-opcode skip, so every such draw
    /// disappeared with no counter and no named reason — the same drift class as
    /// 0x30/0x8d, but pointing the other way (`decode_submission` did not count
    /// it either, so the submission UNDER-reported its draws).
    ///
    /// Field order is pinned against KytyPS5 `CpOpDrawIndex`
    /// (pm4Handlers.cpp L2281-2297): a transposition here is exactly the
    /// zero-index-base failure that produced Dead Cells' `draws=0`.
    #[test]
    fn dispatch_draw_preamble_decodes_the_kytyps5_multi_instanced_layout() {
        let mut cp = CommandProcessor::new();
        let mut sink = IndexOnlySink::default();
        let index_addr = 0x00AB_CDEF_1234_5678u64;
        let object_ids = 0x0000_7777_8888_9999u64;

        let dcb = vec![
            header(2, pm4::IT_INDEX_TYPE, pm4::R_ZERO),
            1, // 32-bit indices
            header(9, pm4::IT_DISPATCH_DRAW_PREAMBLE, pm4::R_ZERO),
            6,                         // index_count
            index_addr as u32,         // addr_lo
            (index_addr >> 32) as u32, // addr_hi
            4,                         // max_instance_count
            object_ids as u32,         // obj_lo
            (object_ids >> 32) as u32, // obj_hi
            4,                         // instance_count
            0xA0,                      // flags (KytyPS5 asserts flags & ~0xa0 == 0)
            header(2, pm4::IT_NUM_INSTANCES, pm4::R_ZERO),
            5,
        ];
        cp.run(&dcb, &mut sink)
            .expect("the emitted layout must walk cleanly");

        assert_eq!(cp.refused_draws(), 0, "the draw must not be refused");
        assert_eq!(sink.indexed.len(), 1, "exactly one indexed draw must land");
        let draw = sink.indexed[0];
        assert_eq!(draw.index_count, 6, "body[0] is the index count");
        assert_eq!(
            draw.index_addr, index_addr,
            "body[1..2] is the index buffer — a zero here is the Dead Cells failure"
        );
        assert_eq!(draw.flags, 0xA0, "body[7] is the flags dword");
        assert_eq!(draw.index_type_and_size, 1, "the latched IT_INDEX_TYPE");
        assert_eq!(
            cp.num_instances(),
            5,
            "the packet after the draw must still execute"
        );
    }

    /// The packet's own declared length is what bounds the field reads.
    ///
    /// `body_at` bounds-checks against the rest of the BUFFER, not against the
    /// current packet, so a short `IT_DISPATCH_DRAW_PREAMBLE` would happily read
    /// the FOLLOWING packets' dwords as its index buffer and instance count.
    /// That must be a counted refusal instead — and the walk must resume at the
    /// right boundary.
    #[test]
    fn a_short_dispatch_draw_preamble_is_refused_not_read_past_its_own_length() {
        let mut cp = CommandProcessor::new();
        // Records auto draws rather than refusing them, so the packet AFTER the
        // short one is observable as a draw and not just as a second refusal.
        let mut sink = RecordingSink::default();
        // A 4-dword 0x3A (body 3) followed by a real auto draw. If the handler
        // read its 8 modeled body dwords it would swallow the auto draw's header
        // and count, and the walk would desync.
        let dcb = vec![
            header(4, pm4::IT_DISPATCH_DRAW_PREAMBLE, pm4::R_ZERO),
            6,
            0x1234,
            0,
            header(3, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO),
            9, // index_count
            0, // flags
        ];
        cp.run(&dcb, &mut sink)
            .expect("a short packet is a refusal, not a stream fault");

        assert_eq!(
            cp.refused_draws(),
            1,
            "the malformed draw must be COUNTED — and it is the ONLY refusal, so the \
             handler did not also mangle the packet behind it"
        );
        let reason = cp.last_refusal().expect("and NAMED");
        assert!(
            reason.contains("IT_DISPATCH_DRAW_PREAMBLE") && reason.contains("3-dword"),
            "the reason must name the opcode and the length it got, got {reason:?}"
        );
        // The auto draw after it must be parsed as a packet, which only holds if
        // the refusal advanced by the short packet's own encoded length.
        assert_eq!(
            sink.draws.iter().map(|d| d.0).collect::<Vec<_>>(),
            vec![9],
            "the walk must resume at the next packet boundary and reach the auto draw"
        );
    }

    /// A draw opcode the AGC decoder counts but this processor cannot translate
    /// must be REFUSED (named + counted), never dropped in the anonymous
    /// unknown-opcode arm.
    ///
    /// `IT_DRAW_INDEX_MULTI_AUTO` (0x30) and `IT_DISPATCH_DRAW` (0x8D) are both
    /// counted in `agc::decode_submission`'s `draw_packets` and neither had a
    /// constant here, so both fell through to the default arm: the packet
    /// inflated the submission's draw count while the walk incremented NEITHER
    /// `sink.draws`, NOR `sink.draw_skips`, NOR `refused_draws` — one
    /// rate-limited "unknown PM4 opcode" line was the entire trace. That is the
    /// Dead Cells `draws=0` failure shape, and this is the assertion that keeps
    /// the drop accountable.
    #[test]
    fn unimplemented_draw_opcodes_are_refused_by_name_not_anonymously_skipped() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        // [MULTI_AUTO][DISPATCH_DRAW][NUM_INSTANCES = 5]. The trailing register
        // write proves the walk continued past both refusals.
        let dcb = vec![
            header(5, pm4::IT_DRAW_INDEX_MULTI_AUTO, pm4::R_ZERO),
            0x100, // MAX_SIZE
            0,     // INDEX_OFFSET
            3,     // INDEX_COUNT
            0,     // DRAW_INITIATOR
            header(3, pm4::IT_DISPATCH_DRAW, pm4::R_ZERO),
            0,
            0,
            header(2, pm4::IT_NUM_INSTANCES, pm4::R_ZERO),
            5,
        ];
        cp.run(&dcb, &mut sink)
            .expect("an unimplemented draw opcode is a refusal, not a stream fault");

        assert_eq!(
            cp.refused_draws(),
            2,
            "both unimplemented draw opcodes must be COUNTED, not silently skipped"
        );
        let reason = cp.last_refusal().expect("a counted refusal must be named");
        assert!(
            reason.contains("IT_DISPATCH_DRAW") && reason.contains("0x8d"),
            "the refusal must name the opcode that was dropped, got {reason:?}"
        );
        assert!(
            sink.draws.is_empty(),
            "a refused draw must not reach the sink"
        );
        assert_eq!(
            cp.num_instances(),
            5,
            "the packet after a refusal must still execute (completion invariant)"
        );
    }

    /// The refusal is skipped by the packet's own encoded length — a wrong
    /// advance would desync the walk and turn a missing handler into a stream
    /// fault (or, worse, misparse the following packet's body as headers).
    #[test]
    fn a_refused_unimplemented_draw_advances_by_its_encoded_length() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        // A deliberately long MULTI_AUTO body whose dwords would each decode as
        // a valid-looking type-3 header if the walk advanced by anything less.
        let mut dcb = vec![header(8, pm4::IT_DRAW_INDEX_MULTI_AUTO, pm4::R_ZERO)];
        dcb.extend(std::iter::repeat_n(
            header(2, pm4::IT_NUM_INSTANCES, pm4::R_ZERO),
            7,
        ));
        dcb.push(header(3, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO));
        dcb.push(9); // index_count
        dcb.push(0); // flags

        cp.run(&dcb, &mut sink).expect("the walk must stay in sync");
        assert_eq!(cp.refused_draws(), 1);
        assert_eq!(
            sink.draws.first().map(|d| d.0),
            Some(9),
            "the packet AFTER the refused one must be parsed as a packet, which \
             only holds if the refusal advanced by its full encoded length"
        );
        assert_eq!(
            cp.num_instances(),
            1,
            "the swallowed body dwords must NOT have executed as NUM_INSTANCES packets"
        );
    }

    /// The emitter/walker contract for `sceAgcDcbDrawIndex`, from the walker's
    /// side: a `DRAW_INDEX_2` whose body carries a zero base is REFUSABLE, and
    /// the same packet with the real base is not. `raeen-hle`'s
    /// `draw_index_emission_reaches_the_command_processor_with_the_real_base`
    /// asserts the emitter produces the second shape; this asserts the walker
    /// distinguishes them, so neither side can drift back alone.
    ///
    /// The zero-base body is what `hle_dcb_draw_index` emitted before the fix.
    /// The `IT_INDEX_BASE` preamble does NOT rescue it: `DRAW_INDEX_2` carries
    /// its own base per AMD PM4, and only `DRAW_INDEX_OFFSET_2` reads the bound
    /// one.
    #[test]
    fn draw_index_2_takes_its_base_from_its_own_body_not_the_index_base_preamble() {
        let base = 0x00AB_CDEF_1234_5678u64;
        // The exact shape hle_dcb_draw_index emits: INDEX_BASE, then
        // INDEX_BUFFER_SIZE, then DRAW_INDEX_2.
        let preamble = [
            header(3, pm4::IT_INDEX_BASE, pm4::R_ZERO),
            base as u32,
            (base >> 32) as u32,
            header(2, pm4::IT_INDEX_BUFFER_SIZE, pm4::R_ZERO),
            6,
        ];

        // Pre-fix body: base zeroed. The preamble is present and ignored.
        let mut zero_base = preamble.to_vec();
        zero_base.extend_from_slice(&[
            header(6, pm4::IT_DRAW_INDEX_2, pm4::R_ZERO),
            6, // MAX_SIZE
            0, // BASE_LO — the bug
            0, // BASE_HI — the bug
            6, // INDEX_COUNT
            0, // DRAW_INITIATOR
        ]);
        let mut cp = CommandProcessor::new();
        let mut sink = IndexedSink::default();
        cp.run(&zero_base, &mut sink).expect("well-formed stream");
        assert_eq!(
            sink.indexed[0].index_addr, 0,
            "a zero-base DRAW_INDEX_2 must NOT silently inherit the bound \
             INDEX_BASE — a real sink refuses it, which is the Dead Cells \
             draws=0 signature"
        );

        // Post-fix body: the real base rides in the packet.
        let mut real_base = preamble.to_vec();
        real_base.extend_from_slice(&[
            header(6, pm4::IT_DRAW_INDEX_2, pm4::R_ZERO),
            6,
            base as u32,
            (base >> 32) as u32,
            6,
            0,
        ]);
        let mut cp = CommandProcessor::new();
        let mut sink = IndexedSink::default();
        cp.run(&real_base, &mut sink).expect("well-formed stream");
        assert_eq!(
            sink.indexed[0].index_addr, base,
            "the walker must deliver the packet's own base"
        );
        assert_eq!(sink.indexed[0].index_count, 6);
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
            "every potentially writing packet notifies the sink"
        );
        assert_eq!(
            sink.guest_writes,
            vec![(dst_a, 16), (dst_b, 16)],
            "each copy reports its exact destination span; the skipped \
             non-memory-selector packet reports none"
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

    /// The write boundary carries the EXACT span each packet wrote.
    ///
    /// A sink's guest-memory caches (analyzed shaders, decoded descriptors,
    /// texture content hashes) are invalidated from this. Reporting "a write
    /// happened, somewhere" forced the embedder to throw all of them away on
    /// every completion label a title interleaves with its draws — which is why
    /// the resolved-shader memo never hit and every draw re-ran the full VS+PS
    /// resource analysis.
    #[test]
    fn guest_write_boundary_reports_the_exact_written_span() {
        // Three incrementing dwords at 0x9000 are ONE span, not three entries.
        let mem = RwMem::new(0x9000, 8);
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(7, pm4::IT_NOP, pm4::R_WRITE_DATA),
            1, // dst_sel = memory, addr-increment enabled
            0x9000,
            0,
            0xA,
            0xB,
            0xC,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("producer must not fault");
        assert_eq!(
            sink.guest_writes,
            vec![(0x9000, 12)],
            "three consecutive dwords coalesce into one 12-byte span"
        );

        // A 32-bit RELEASE_MEM label is a 4-byte span at its own address.
        let mut sink = RecordingSink::default();
        let mut cp = CommandProcessor::new();
        cp.run_with_memory(&release_mem_agc(0x9004, 7), &mut sink, Some(&mem))
            .expect("producer must not fault");
        assert_eq!(sink.guest_writes, vec![(0x9004, 4)]);
    }

    /// `ADDR_INCR` disabled makes every payload dword land on the SAME address
    /// (hardware behaviour, last wins). That is one 4-byte span, not N.
    #[test]
    fn guest_write_boundary_coalesces_a_non_incrementing_write_data() {
        let mem = RwMem::new(0x9000, 8);
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(7, pm4::IT_NOP, pm4::R_WRITE_DATA),
            1 | (1 << 16), // dst_sel = memory, addr-increment DISABLED
            0x9000,
            0,
            0xA,
            0xB,
            0xC,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("producer must not fault");
        assert_eq!(mem.word(0), 0xC, "last write wins at the fixed address");
        assert_eq!(sink.guest_writes, vec![(0x9000, 4)]);
    }

    /// A write-capable packet that wrote NOTHING (non-memory destination
    /// selector) still notifies — so a sink may keep a blanket-clear policy —
    /// but reports no span, so a range-precise sink invalidates nothing.
    #[test]
    fn guest_write_boundary_reports_no_span_when_the_packet_wrote_nothing() {
        let mem = RwMem::new(0x9000, 4);
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(5, pm4::IT_NOP, pm4::R_WRITE_DATA),
            0, // dst_sel = 0: not a memory destination
            0x9000,
            0,
            0x2A,
        ];
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("producer must not fault");
        assert_eq!(mem.word(0), 0, "nothing was written");
        assert_eq!(
            sink.guest_memory_write_boundaries, 1,
            "the notification still fires"
        );
        assert!(
            sink.guest_writes.is_empty(),
            "no span means nothing to invalidate"
        );
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

    // ---- Ordered completion side effects (events / EOP interrupts / flips)
    // and the injectable RELEASE_MEM timestamp clock (checklist item 5,
    // steps 3-5) ----

    /// AGC flip packet (`IT_NOP` + `R_FLIP`), the layout the eager decoder
    /// honours: body `[handle, index, mode, arg_lo, arg_hi, 0]`.
    fn flip_agc(handle: u32, index: u32, mode: u32, arg: u64) -> Vec<u32> {
        vec![
            header(7, pm4::IT_NOP, pm4::R_FLIP),
            handle,
            index,
            mode,
            arg as u32,
            (arg >> 32) as u32,
            0,
        ]
    }

    /// Events, flips and AGC EOP interrupts are recorded in STREAM order for
    /// the embedder; draining empties the record; a queue reset must not drop
    /// undelivered effects.
    #[test]
    fn events_flips_and_eop_interrupts_are_recorded_in_stream_order() {
        let mem = RwMem::new(0x9000, 4);
        // Standard EVENT_WRITE (event id = low 6 bits of the first body word),
        // then a flip, then an interrupt-only AGC RELEASE_MEM (no dst).
        let mut dcb = vec![header(3, pm4::IT_EVENT_WRITE, pm4::R_ZERO), 0x2A, 0];
        dcb.extend(flip_agc(1, 2, 3, 0x0123_4567_89AB_CDEF));
        dcb.push(header(8, pm4::IT_NOP, pm4::R_RELEASE_MEM));
        dcb.extend_from_slice(&[0, 2 << 24, 0, 0, 0, 0, 0x55]);
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem)).unwrap();
        assert_eq!(
            cp.take_side_effects(),
            vec![
                SideEffect::EventWrite { event_id: 0x2A },
                SideEffect::Flip {
                    video_out_handle: 1,
                    display_buffer_index: 2,
                    flip_mode: 3,
                    flip_arg: 0x0123_4567_89AB_CDEF,
                },
                SideEffect::EopInterrupt { context_id: 0x55 },
            ],
            "all three side effects, in PM4 stream order"
        );
        assert!(cp.take_side_effects().is_empty(), "the drain empties");

        // Undelivered effects survive an in-stream queue reset.
        cp.run_with_memory(&flip_agc(1, 0, 0, 9), &mut sink, Some(&mem))
            .unwrap();
        cp.reset();
        assert_eq!(
            cp.take_side_effects(),
            vec![SideEffect::Flip {
                video_out_handle: 1,
                display_buffer_index: 0,
                flip_mode: 0,
                flip_arg: 9,
            }],
            "a queue reset must not drop undelivered side effects"
        );
    }

    /// Parity with the eager decoder: the STANDARD `RELEASE_MEM` form carries
    /// no modeled interrupt extraction, and a short `R_FLIP` is unmodeled —
    /// neither records a side effect.
    #[test]
    fn standard_release_mem_and_short_flips_record_no_side_effects() {
        let mem = RwMem::new(0x9000, 4);
        let mut dcb = vec![
            header(8, pm4::IT_RELEASE_MEM, pm4::R_ZERO),
            0,
            (1u32 << 29) | (2 << 24), // DATA_SEL 1; junk in the byte at 31:24
            0x9000u32,
            0,
            7,
            0,
            0,
        ];
        // 5-total-dword R_FLIP: shorter than the modeled layout.
        dcb.extend_from_slice(&[header(5, pm4::IT_NOP, pm4::R_FLIP), 1, 0, 0, 0]);
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem)).unwrap();
        assert_eq!(mem.word(0), 7, "the standard label write still lands");
        assert!(
            cp.take_side_effects().is_empty(),
            "no interrupt from the standard form, no flip from a short packet"
        );
    }

    /// Step 5's ordering property at the CP layer: a flip queued behind an
    /// unexecuted wait is NOT recorded (so the embedder cannot deliver it
    /// early); it appears only once the resumed walk genuinely passes the wait.
    #[test]
    fn a_flip_behind_an_unmet_wait_is_not_recorded_until_the_wait_passes() {
        let mem = RwMem::new(0x9000, 4);
        let mut dcb = wait32(0x9000, !0, 3, 1); // wait for label == 1 (it is 0)
        dcb.extend(flip_agc(1, 0, 0, 7));
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let outcome = cp.run_resumable(&dcb, 0, &mut sink, Some(&mem)).unwrap();
        let RunOutcome::Suspended(suspended) = outcome else {
            panic!("the unmet wait must suspend the walk");
        };
        assert!(
            cp.take_side_effects().is_empty(),
            "the flip must not become visible before its wait executes"
        );
        // The producer writes the label; the resumed walk reaches the flip.
        mem.words.borrow_mut()[0] = 1;
        assert_eq!(
            cp.run_resumable(&dcb, suspended.resume_dword, &mut sink, Some(&mem))
                .unwrap(),
            RunOutcome::Completed
        );
        assert_eq!(
            cp.take_side_effects(),
            vec![SideEffect::Flip {
                video_out_handle: 1,
                display_buffer_index: 0,
                flip_mode: 0,
                flip_arg: 7,
            }],
            "the flip is recorded exactly once, after the wait passes"
        );
    }

    fn fixed_ts() -> Option<u64> {
        Some(0x0000_1234_5678_9ABC)
    }
    fn declined_ts() -> Option<u64> {
        None
    }

    /// `RELEASE_MEM` DATA_SEL 3 timestamp packet targeting `addr`.
    fn ts_packet(addr: u64) -> Vec<u32> {
        vec![
            header(7, pm4::IT_NOP, pm4::R_RELEASE_MEM),
            0,
            3 << 16,
            addr as u32,
            (addr >> 32) as u32,
            0,
            0,
        ]
    }

    /// Step 3 at the CP layer: an installed timestamp source overrides the
    /// legacy release counter (and survives a queue reset — it is embedder
    /// configuration); a source that DECLINES (`None` — the gate off) is the
    /// legacy counter, exactly.
    #[test]
    fn an_installed_timestamp_source_overrides_the_release_clock() {
        let mem = RwMem::new(0x9000, 4);
        let mut cp = CommandProcessor::new();
        cp.set_timestamp_source(Some(fixed_ts));
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&ts_packet(0x9000), &mut sink, Some(&mem))
            .unwrap();
        assert_eq!(mem.word(0), 0x5678_9ABC, "source low dword");
        assert_eq!(mem.word(1), 0x0000_1234, "source high dword");
        // The source survives a queue reset.
        let mem = RwMem::new(0x9000, 4);
        cp.reset();
        cp.run_with_memory(&ts_packet(0x9000), &mut sink, Some(&mem))
            .unwrap();
        assert_eq!(mem.word(0), 0x5678_9ABC, "source survives reset");

        // A declining source falls back to the nonzero, advancing counter.
        let mem = RwMem::new(0x9000, 4);
        let mut cp = CommandProcessor::new();
        cp.set_timestamp_source(Some(declined_ts));
        cp.run_with_memory(&ts_packet(0x9000), &mut sink, Some(&mem))
            .unwrap();
        let first = u64::from(mem.word(0)) | (u64::from(mem.word(1)) << 32);
        assert_ne!(first, 0, "declined source falls back to the counter");
        cp.run_with_memory(&ts_packet(0x9000), &mut sink, Some(&mem))
            .unwrap();
        let second = u64::from(mem.word(0)) | (u64::from(mem.word(1)) << 32);
        assert!(second > first, "counter advances: {first} -> {second}");
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

    /// Every packet the walk consumes lands in exactly one [`WalkCensus`]
    /// bucket, and the register bucket counts REGISTERS, not packets.
    ///
    /// `walk_us` alone could not say which packet class owned a submission. This
    /// pins that the split is exhaustive — a packet that fell through every arm
    /// would leave the buckets short of the packet count, which is the failure
    /// mode that made the old report un-actionable.
    #[test]
    fn walk_census_attributes_every_packet_class() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let label = 0x2000u64;
        let mem = BufMem {
            base: label,
            words: vec![0; 8],
        };

        let mut dcb = Vec::new();
        // 1 context-register packet writing 4 registers.
        dcb.extend_from_slice(&[
            header(6, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            pm4::SPI_PS_INPUT_CNTL_0,
            1,
            2,
            3,
            4,
        ]);
        // 1 shader-register packet writing 2 registers.
        dcb.extend_from_slice(&set_sh(pm4::SPI_SHADER_USER_DATA_PS_0, &[7, 8]));
        // 1 user-config packet writing 1 register.
        dcb.extend_from_slice(&[
            header(3, pm4::IT_SET_UCONFIG_REG, pm4::R_ZERO),
            pm4::VGT_PRIMITIVE_TYPE,
            17,
        ]);
        // 1 draw.
        dcb.extend_from_slice(&[header(7, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO), 3, 0]);
        dcb.extend(pad(4));
        // 1 WRITE_DATA completion label (2 payload dwords into memory).
        dcb.extend_from_slice(&[
            header(6, pm4::IT_WRITE_DATA, pm4::R_ZERO),
            1 << 8,
            label as u32,
            (label >> 32) as u32,
            0xabcd,
            0,
        ]);
        // 1 inert packet: CONTEXT_CONTROL.
        dcb.extend_from_slice(&[header(3, pm4::IT_CONTEXT_CONTROL, pm4::R_ZERO), 0, 0]);

        cp.set_walk_timing(true);
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("census stream must walk");
        let census = cp.take_walk_census();

        assert_eq!(census.reg_packets, 3, "three register packets");
        assert_eq!(census.regs, 7, "4 + 2 + 1 individual registers");
        assert_eq!(census.draws, 1);
        assert_eq!(census.write_packets, 1);
        assert_eq!(census.inert_packets, 1);
        assert_eq!(census.dispatches, 0);
        assert_eq!(census.waits, 0);
        assert_eq!(census.indirects, 0);
        // The write packet raised exactly one sink boundary notification.
        assert_eq!(census.boundaries, 1);
        assert_eq!(
            sink.guest_memory_write_boundaries, 1,
            "the census must agree with the sink"
        );

        // EXHAUSTIVE: every packet is in exactly one bucket.
        let counted = census.reg_packets
            + census.draws
            + census.dispatches
            + census.write_packets
            + census.waits
            + census.indirects
            + census.inert_packets;
        assert_eq!(counted, 6, "6 packets in, 6 packets attributed");

        // Timing is opt-in and only for the classes where a clock read is noise.
        assert!(census.draw_ns > 0, "a timed class must report time");
        assert!(census.write_ns > 0);

        // With the census OFF nothing is recorded at all — not even a count.
        // That is deliberate: classify-and-increment measured +1-2.5 ns on a
        // ~10 ns packet, and the default path must not pay for a diagnostic.
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("census stream must walk");
        assert_eq!(
            cp.take_walk_census(),
            WalkCensus::default(),
            "no packet may be classified, counted, or clocked unless the census \
             was asked for"
        );
        assert_eq!(
            sink.guest_memory_write_boundaries, 1,
            "the walk itself must behave identically either way"
        );
    }

    // ─── IT_INDIRECT_BUFFER chains ──────────────────────────────────────────

    /// Several disjoint guest buffers at chosen addresses, for chain targets.
    /// Unlike [`BufMem`] this can hold a whole chain graph, and it REFUSES any
    /// read that is not wholly inside one registered range — the property the
    /// "unmapped target" test depends on.
    #[derive(Default)]
    struct ChainMem {
        ranges: Vec<(u64, Vec<u32>)>,
        /// Every `(addr, count)` this memory was asked for, in order. A refusal
        /// test asserts against this that no read was even attempted.
        reads: std::cell::RefCell<Vec<(u64, u32)>>,
    }

    impl ChainMem {
        fn with(mut self, base: u64, words: Vec<u32>) -> Self {
            self.ranges.push((base, words));
            self
        }

        fn read_count(&self) -> usize {
            self.reads.borrow().len()
        }
    }

    impl GuestMemory for ChainMem {
        fn read_dwords(&self, addr: u64, count: u32) -> Option<Vec<u32>> {
            self.reads.borrow_mut().push((addr, count));
            if count == 0 || addr % 4 != 0 {
                return None;
            }
            self.ranges.iter().find_map(|(base, words)| {
                let rel = addr.checked_sub(*base)?;
                let start = usize::try_from(rel / 4).ok()?;
                let end = start.checked_add(count as usize)?;
                words.get(start..end).map(<[u32]>::to_vec)
            })
        }
    }

    /// A 4-dword unconditional chain packet (`sceAgcDcbJump`'s emission).
    fn chain_packet(target: u64, size_dwords: u32) -> Vec<u32> {
        vec![
            header(4, pm4::IT_INDIRECT_BUFFER, pm4::R_ZERO),
            target as u32,
            ((target >> 32) & 0xffff) as u32,
            size_dwords & 0x000f_ffff,
        ]
    }

    /// One `IT_DRAW_INDEX_AUTO` with `index_count`, so a draw can be identified
    /// by which buffer it came from.
    fn tagged_draw(index_count: u32) -> Vec<u32> {
        vec![
            header(3, pm4::IT_DRAW_INDEX_AUTO, pm4::R_ZERO),
            index_count,
            0,
        ]
    }

    fn drawn(sink: &RecordingSink) -> Vec<u32> {
        sink.draws.iter().map(|draw| draw.0).collect()
    }

    /// The step-1 measurement: a chain packet must be DECODED and counted with
    /// its target and size even when the follower is off, and must not be
    /// followed. Before `cp_op_indirect_buffer` existed, 0x3F fell into the
    /// anonymous unknown-opcode arm and the only trace of a chained frame was
    /// one rate-limited "unknown PM4 opcode" line.
    #[test]
    fn a_chain_packet_is_counted_with_its_target_even_when_not_followed() {
        let child = 0x8000_1000u64;
        let mem = ChainMem::default().with(child, tagged_draw(77));
        let mut dcb = chain_packet(child, 3);
        dcb.extend(tagged_draw(11));

        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("a chain packet must never fault the walk");

        let census = cp.take_chain_census();
        assert_eq!(census.jump_packets, 1, "the 4-dword chain form was seen");
        assert_eq!(census.packets(), 1);
        assert_eq!(
            census.target_dwords, 3,
            "the census must report how much command stream lives outside the submitted buffer"
        );
        assert_eq!(
            census.refused_disabled, 1,
            "with the follower off the target is a NAMED refusal, not a silent skip"
        );
        assert_eq!(census.followed, 0);
        assert_eq!(
            census.samples,
            vec![ChainSample {
                offset: 0,
                address: child,
                size_dwords: 3,
                control: 3,
                form: ChainForm::Jump,
            }],
            "the sample must name the target so a run can be diagnosed from the log"
        );
        assert_eq!(
            drawn(&sink),
            vec![11],
            "the follower is off: only the submitted buffer's draw runs"
        );
        assert_eq!(
            mem.read_count(),
            0,
            "counting a chain must not read the target"
        );
    }

    /// One level: the child's draws execute, and the PARENT resumes after the
    /// chain packet. Call semantics, per KytyPS5 `CpOpIndirectBuffer` returning
    /// 3 after `ProcessIndirectBuffer` and shadPS4 `liverpool.cpp` L830
    /// advancing by the packet's own length after the nested task completes.
    #[test]
    fn a_followed_chain_executes_the_childs_draws_then_returns_to_the_parent() {
        let child = 0x8000_1000u64;
        let mem = ChainMem::default().with(child, tagged_draw(77));
        let mut dcb = tagged_draw(11);
        dcb.extend(chain_packet(child, 3));
        dcb.extend(tagged_draw(22));

        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("a chained stream must walk");

        assert_eq!(
            drawn(&sink),
            vec![11, 77, 22],
            "the child runs AT the chain packet and the parent's remaining draws still run — a \
             CALL, not a jump (77 missing = chain not followed; 22 missing = treated as a jump)"
        );
        let census = cp.take_chain_census();
        assert_eq!(census.followed, 1);
        assert_eq!(census.followed_dwords, 3);
        assert_eq!(census.refusals(), 0);
    }

    /// Registers a chained buffer writes must be visible to the rest of the
    /// PARENT — the property that makes call-vs-jump ordering load-bearing
    /// rather than cosmetic.
    #[test]
    fn a_chained_buffer_state_write_is_visible_to_the_parents_later_draws() {
        let child = 0x8000_2000u64;
        let mem = ChainMem::default().with(
            child,
            vec![
                header(3, pm4::IT_SET_UCONFIG_REG, pm4::R_ZERO),
                pm4::VGT_PRIMITIVE_TYPE,
                17,
            ],
        );
        let mut dcb = chain_packet(child, 3);
        dcb.extend(tagged_draw(11));

        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("a chained state stream must walk");

        assert_eq!(
            sink.draws.first().map(|draw| draw.2),
            Some(17),
            "the parent's draw must see the primitive type the CHILD set"
        );
    }

    /// Nesting to exactly [`CommandProcessor::MAX_CHAIN_DEPTH`] works, and the
    /// buffer one level past it is refused BY NAME rather than growing the
    /// work-list. Both references bound this at nothing at all.
    #[test]
    fn a_chain_nests_to_the_configured_depth_and_refuses_one_deeper() {
        // depth-1 .. depth-N buffers, each chaining into the next; the deepest
        // one draws.
        let base = 0x8001_0000u64;
        let stride = 0x1000u64;
        let depth = CommandProcessor::MAX_CHAIN_DEPTH;
        let address = |level: usize| base + stride * level as u64;

        // Every level is exactly 4 dwords so one declared size fits them all —
        // the deepest one pads its 3-dword draw with a type-2 filler. Without
        // that the parent's declared size would overrun the child's real extent
        // and the test would measure a short read instead of the depth bound.
        let build = |levels: usize| {
            let mut mem = ChainMem::default();
            for level in 0..levels {
                let words = if level + 1 == levels {
                    let mut deepest = tagged_draw(900 + level as u32);
                    deepest.push(0x8000_0000);
                    deepest
                } else {
                    chain_packet(address(level + 1), 4)
                };
                assert_eq!(words.len(), 4, "every chain level must be 4 dwords");
                mem = mem.with(address(level), words);
            }
            mem
        };

        // Exactly at the limit: the submitted buffer chains into level 0, which
        // is work-list depth 1, so `depth` levels fit.
        let mem = build(depth);
        let dcb = chain_packet(address(0), 4);
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("a chain at the depth limit must walk");
        let census = cp.take_chain_census();
        assert_eq!(
            census.followed, depth as u64,
            "every level up to the limit must be walked"
        );
        assert_eq!(census.refused_depth, 0);
        assert_eq!(
            drawn(&sink),
            vec![900 + depth as u32 - 1],
            "the deepest buffer's draw must reach the sink"
        );

        // One deeper: the last buffer is refused, and everything above it still
        // ran.
        let mem = build(depth + 1);
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("an over-deep chain must be refused, not fault the walk");
        let census = cp.take_chain_census();
        assert_eq!(
            census.refused_depth, 1,
            "the buffer past MAX_CHAIN_DEPTH must be a named counted refusal"
        );
        assert_eq!(census.followed, depth as u64);
        assert!(
            drawn(&sink).is_empty(),
            "the refused level held the only draw"
        );
    }

    /// A buffer that chains to itself, and an A→B→A loop, must terminate.
    #[test]
    fn a_self_referential_and_a_two_buffer_chain_cycle_both_terminate() {
        let a = 0x8002_0000u64;
        let b = 0x8002_1000u64;

        // Self-chain: A chains into A.
        let mut a_words = chain_packet(a, 4);
        a_words.extend(tagged_draw(1));
        let mem = ChainMem::default().with(a, a_words);
        let dcb = chain_packet(a, 7);
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("a self-referential chain must terminate");
        let census = cp.take_chain_census();
        assert_eq!(census.refused_cycle, 1, "A→A must be refused as a cycle");
        assert_eq!(
            drawn(&sink),
            vec![1],
            "the cycle refusal must not abandon the rest of A"
        );

        // A→B→A: A is 4 dwords (one chain packet into B), B is 7 (a chain packet
        // back into A, then a draw).
        let mut b_words = chain_packet(a, 4);
        b_words.extend(tagged_draw(2));
        let mem = ChainMem::default()
            .with(a, chain_packet(b, 7))
            .with(b, b_words);
        let dcb = chain_packet(a, 4);
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("an A→B→A chain must terminate");
        let census = cp.take_chain_census();
        assert_eq!(census.refused_cycle, 1, "A→B→A must be refused as a cycle");
        assert_eq!(census.followed, 2, "A and B each ran once");
        assert_eq!(drawn(&sink), vec![2]);

        // The same loop with a DIFFERENT declared size for A on the way back.
        // Keying the cycle test on `(address, size)` would miss this; keying it
        // on the address does not.
        let mut b_words = chain_packet(a, 3);
        b_words.extend(tagged_draw(3));
        let mem = ChainMem::default()
            .with(a, chain_packet(b, 7))
            .with(b, b_words);
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("an A→B→A' chain must terminate");
        let census = cp.take_chain_census();
        assert_eq!(
            census.refused_cycle, 1,
            "re-entering A at a different declared size is still a cycle"
        );
        assert_eq!(drawn(&sink), vec![3]);
    }

    /// An unmapped or misaligned target is a named refusal, and the misaligned
    /// one is refused WITHOUT the memory authority ever being asked to read it.
    #[test]
    fn an_unmapped_or_misaligned_chain_target_is_refused_without_touching_host_memory() {
        let mapped = 0x8003_0000u64;
        let mem = ChainMem::default().with(mapped, tagged_draw(5));

        // Unmapped: well-formed packet, target the authority will not read.
        let dcb = chain_packet(0x1234_5000, 4);
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("an unmapped chain target must be refused, not fault");
        let census = cp.take_chain_census();
        assert_eq!(census.refused_unreadable, 1);
        assert_eq!(census.followed, 0);
        assert!(drawn(&sink).is_empty());

        // Misaligned: refused BEFORE any read is attempted.
        let mem = ChainMem::default().with(mapped, tagged_draw(5));
        let dcb = chain_packet(mapped + 2, 3);
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("a misaligned chain target must be refused, not fault");
        let census = cp.take_chain_census();
        assert_eq!(census.refused_malformed, 1);
        assert_eq!(
            mem.read_count(),
            0,
            "a misaligned target must never be handed to the guest-memory authority"
        );

        // Null target, and a size the 20-bit IB_SIZE field cannot hold.
        for (address, size) in [(0u64, 4u32), (mapped, 0)] {
            let dcb = chain_packet(address, size);
            let mut cp = CommandProcessor::new();
            cp.set_follow_chains(true);
            let mut sink = RecordingSink::default();
            let mem = ChainMem::default().with(mapped, tagged_draw(5));
            cp.run_with_memory(&dcb, &mut sink, Some(&mem))
                .expect("a malformed chain target must be refused, not fault");
            assert_eq!(
                cp.take_chain_census().refused_malformed,
                1,
                "null/empty target {address:#x}/{size}"
            );
            assert_eq!(mem.read_count(), 0);
        }
    }

    /// A chained buffer whose stream is desynced must abandon THAT buffer only.
    /// The submitted buffer's next packet boundary is still known, because the
    /// chain packet carried its own length.
    #[test]
    fn a_desynced_chained_buffer_is_abandoned_without_aborting_the_submission() {
        let child = 0x8004_0000u64;
        // A type-0 header: not type 2, not type 3.
        let mem = ChainMem::default().with(child, vec![0x0000_0000, 0, 0]);
        let mut dcb = chain_packet(child, 3);
        dcb.extend(tagged_draw(42));

        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("a desynced CHILD must not fault the submitted buffer");
        assert_eq!(
            drawn(&sink),
            vec![42],
            "the submitted buffer must keep walking after a bad child"
        );
        assert_eq!(cp.take_chain_census().refused_malformed, 1);
    }

    /// `IT_INDIRECT_BUFFER_CNST` (0x33) addresses the constant-engine ring.
    /// Counted and named, never walked against the graphics register file —
    /// shadPS4 handles it only in `ProcessCeUpdate` (liverpool.cpp L195) and
    /// KytyPS5 refuses any opcode but 0x3F (pm4Handlers.cpp L2572).
    #[test]
    fn indirect_buffer_const_is_counted_but_never_followed() {
        let child = 0x8005_0000u64;
        let mem = ChainMem::default().with(child, tagged_draw(99));
        let mut dcb = chain_packet(child, 3);
        dcb[0] = header(4, pm4::IT_INDIRECT_BUFFER_CNST, pm4::R_ZERO);
        dcb.extend(tagged_draw(1));

        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("a CE chain packet must not fault the walk");
        let census = cp.take_chain_census();
        assert_eq!(census.const_packets, 1);
        assert_eq!(census.followed, 0, "the CE ring must never be walked here");
        assert_eq!(
            drawn(&sink),
            vec![1],
            "only the submitted buffer's own draw runs"
        );
        assert_eq!(mem.read_count(), 0);
    }

    /// The 14-dword conditional form (`sceAgcCbBranch` / KytyPS5 `CpOpBranch`):
    /// the compare picks then- or else-target, and mode 1 has no else.
    #[test]
    fn the_conditional_chain_takes_the_then_or_else_buffer_by_its_compare() {
        let label = 0x8006_0000u64;
        let then_buffer = 0x8006_1000u64;
        let else_buffer = 0x8006_2000u64;

        // mode | function << 8. Function 3 = equal.
        let branch = |mode: u32, function: u32| {
            vec![
                header(14, pm4::IT_INDIRECT_BUFFER, pm4::R_ZERO),
                mode | (function << 8),
                label as u32,
                (label >> 32) as u32,
                0xffff_ffff,
                0xffff_ffff,
                7,
                0,
                then_buffer as u32,
                (then_buffer >> 32) as u32,
                3,
                else_buffer as u32,
                (else_buffer >> 32) as u32,
                3,
            ]
        };
        let memory = |label_value: u32| {
            ChainMem::default()
                .with(label, vec![label_value, 0])
                .with(then_buffer, tagged_draw(700))
                .with(else_buffer, tagged_draw(800))
        };

        // Label == reference: then-branch.
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&branch(2, 3), &mut sink, Some(&memory(7)))
            .expect("a conditional chain must walk");
        assert_eq!(drawn(&sink), vec![700], "satisfied compare takes then");
        assert_eq!(cp.take_chain_census().branch_packets, 1);

        // Label != reference, mode 2: else-branch.
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&branch(2, 3), &mut sink, Some(&memory(9)))
            .expect("a conditional chain must walk");
        assert_eq!(drawn(&sink), vec![800], "unsatisfied compare takes else");

        // Label != reference, mode 1: no else target exists.
        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&branch(1, 3), &mut sink, Some(&memory(9)))
            .expect("a then-only conditional chain must walk");
        assert!(
            drawn(&sink).is_empty(),
            "mode 1 has no else buffer — neither branch runs"
        );
        assert_eq!(cp.take_chain_census().followed, 0);
    }

    /// An in-stream `R_DRAW_RESET` must not turn the follower off or zero the
    /// census: a per-frame queue reset between a chain packet and the
    /// embedder's drain would otherwise silently drop the rest of the frame.
    #[test]
    fn a_queue_reset_preserves_the_follower_and_the_chain_census() {
        let child = 0x8007_0000u64;
        let mem = ChainMem::default().with(child, tagged_draw(31));
        let mut dcb = chain_packet(child, 3);
        dcb.extend_from_slice(&[header(2, pm4::IT_NOP, pm4::R_DRAW_RESET), 0]);
        dcb.extend(chain_packet(child, 3));

        let mut cp = CommandProcessor::new();
        cp.set_follow_chains(true);
        let mut sink = RecordingSink::default();
        cp.run_with_memory(&dcb, &mut sink, Some(&mem))
            .expect("a reset between chains must walk");
        assert!(cp.follows_chains(), "a queue reset is not a config change");
        let census = cp.take_chain_census();
        assert_eq!(
            census.followed, 2,
            "the chain AFTER the reset must still be followed"
        );
        assert_eq!(drawn(&sink), vec![31, 31]);
    }

    /// THE REGRESSION GUARD for the working titles: a submission with no chain
    /// packets must behave identically with the follower on and off, and must
    /// not touch the chain census at all. Minecraft draws with
    /// `DRAW_INDEX_OFFSET_2` and is the M5 acceptance path.
    #[test]
    fn a_submission_without_chains_is_identical_with_the_follower_on_or_off() {
        let label = 0x8008_0000u64;
        let mem = BufMem {
            base: label,
            words: vec![0x1234_5678, 0],
        };
        let mut dcb = state_and_draw();
        dcb.extend_from_slice(&[header(3, pm4::IT_DRAW_INDEX_OFFSET_2, pm4::R_ZERO), 9, 0]);
        dcb.extend_from_slice(&[
            header(4, pm4::IT_NOP, pm4::R_CX_REGS_INDIRECT),
            label as u32,
            (label >> 32) as u32,
            1,
        ]);
        dcb.extend_from_slice(&[header(2, pm4::IT_NOP, pm4::R_PUSH_MARKER), 0]);

        /// Everything a chainless walk can observably produce.
        #[derive(Debug, PartialEq)]
        struct WalkResult {
            draws: Vec<(u32, u32, u32, bool, bool)>,
            dispatches: Vec<RecordedDispatch>,
            boundaries: u32,
            guest_writes: Vec<(u64, u64)>,
            ctx: Context,
            ucfg: UserConfig,
            refused_draws: u64,
            reg_packets: u64,
            regs: u64,
            walk_draws: u64,
            indirects: u64,
            inert_packets: u64,
            chain: ChainCensus,
        }

        let run = |follow: bool| {
            let mut cp = CommandProcessor::new();
            cp.set_follow_chains(follow);
            cp.set_walk_timing(true);
            let mut sink = RecordingSink::default();
            cp.run_with_memory(&dcb, &mut sink, Some(&mem))
                .expect("a chainless stream must walk");
            let walk = cp.take_walk_census();
            WalkResult {
                draws: sink.draws.clone(),
                dispatches: sink.dispatches.clone(),
                boundaries: sink.guest_memory_write_boundaries,
                guest_writes: sink.guest_writes.clone(),
                ctx: cp.get_ctx().clone(),
                ucfg: cp.get_ucfg().clone(),
                refused_draws: cp.refused_draws(),
                reg_packets: walk.reg_packets,
                regs: walk.regs,
                walk_draws: walk.draws,
                indirects: walk.indirects,
                inert_packets: walk.inert_packets,
                chain: cp.take_chain_census(),
            }
        };
        let off = run(false);
        let on = run(true);
        assert_eq!(
            off, on,
            "enabling the chain follower must change NOTHING for a submission that carries no \
             chain packets — this is what protects Minecraft / Dead Cells / Blasphemous II"
        );
        assert_eq!(
            off.chain,
            ChainCensus::default(),
            "a chainless submission must not touch the chain census"
        );
    }
}
