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
//! terminates at the [`DrawSink`] trait. `xps5x-gpu` implements it.
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
//! - A draw the sink cannot honour is still a named error
//!   ([`CpError::Draw`]) — never-silent applies to draws, and the caller
//!   decides whether to continue the submit.
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
use tracing::warn;

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
    /// Latched by the `R_ZERO` 'hu' marker; types subsequent user-SGPR writes.
    user_data_marker: UserSgprType,
    /// Which distinct unknown ops/registers have already been warned about.
    /// Survives [`CommandProcessor::reset`] so a per-frame `R_DRAW_RESET`
    /// cannot turn the rate limit back into log spam.
    warned: BTreeSet<SkipKey>,
    /// Number of shader-bind trace sites visited. This survives queue resets
    /// and bounds the opt-in diagnostic on frame loops.
    shader_bind_trace_count: u64,
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

    /// Kyty: `CommandProcessor::Reset` (L519) — clears register and index
    /// state. The warn rate-limit set deliberately survives (deviation; a
    /// reset must not re-arm log spam).
    pub fn reset(&mut self) {
        let warned = std::mem::take(&mut self.warned);
        let shader_bind_trace_count = self.shader_bind_trace_count;
        *self = Self::new();
        self.warned = warned;
        self.shader_bind_trace_count = shader_bind_trace_count;
    }

    /// True the first time `key` is seen; the caller warns exactly then.
    fn first(&mut self, key: SkipKey) -> bool {
        self.warned.insert(key)
    }

    fn trace_shader_bind(&mut self) -> bool {
        if std::env::var_os("XPS5X_TRACE_SHADER_BINDS").is_none() {
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
    /// Only structural faults ([`CpError::Truncated`], [`CpError::NotType3`])
    /// and refused draws ([`CpError::Draw`]) — unknown packets are skipped per
    /// the module-level resilience policy.
    pub fn run_with_memory(
        &mut self,
        data: &[u32],
        sink: &mut dyn DrawSink,
        mem: Option<&dyn GuestMemory>,
    ) -> Result<(), CpError> {
        let mut pos = 0usize;
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
            let consumed = self.dispatch(cmd_id, body, offset, sink, mem)?;

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
        }
        Ok(())
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
        match op {
            pm4::IT_NOP => self.cp_op_nop(cmd_id, body, offset, sink, mem),
            pm4::IT_SET_CONTEXT_REG => self.cp_op_set_context_reg(cmd_id, body, offset),
            pm4::IT_SET_SH_REG => self.cp_op_set_shader_reg(cmd_id, body, offset),
            pm4::IT_SET_UCONFIG_REG => self.cp_op_set_uconfig_reg(cmd_id, body, offset),
            pm4::IT_DRAW_INDEX_AUTO => self.cp_op_draw_index_auto(cmd_id, body, offset, sink),
            pm4::IT_DISPATCH_DIRECT => self.cp_op_dispatch_direct(cmd_id, body, offset, sink),
            // Kyty: cp_op_draw_index (L2757), raw IT form 0xc0042700.
            pm4::IT_DRAW_INDEX_2 => self.cp_op_draw_index(cmd_id, body, offset, sink),
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
            // SET_BASE select 1 = indirect-draw argument buffer base.
            pm4::IT_SET_BASE => {
                let select = Self::body_at(body, 0, offset)?;
                let lo = Self::body_at(body, 1, offset)?;
                let hi = Self::body_at(body, 2, offset)?;
                if select == 1 {
                    self.indirect_draw_base = u64::from(lo) | (u64::from(hi) << 32);
                } else if self.first(SkipKey::Note("set_base_select")) {
                    warn!(
                        select,
                        offset, "IT_SET_BASE with unsupported base select — ignored"
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
            // Kyty ports these with 22 EXIT_NOT_IMPLEMENTED sites between them;
            // nothing on the minimal draw path observes their effects. Their
            // side effects (waits, label writes) are handled by the HLE submit
            // layer where needed.
            pm4::IT_ACQUIRE_MEM
            | pm4::IT_RELEASE_MEM
            | pm4::IT_EVENT_WRITE
            | pm4::IT_EVENT_WRITE_EOP
            | pm4::IT_EVENT_WRITE_EOS
            | pm4::IT_WAIT_REG_MEM
            | pm4::IT_WRITE_DATA
            | pm4::IT_DMA_DATA
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
        }
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
            pm4::R_DISPATCH_RESET => {
                self.reset();
                Ok(pm4::body_dw(cmd_id))
            }
            // Sync / flip / memory ops: consumed, not honoured. A draw never
            // observes them, and their side effects (flip queues, label
            // waits) are already applied by the HLE submit decode.
            pm4::R_WAIT_MEM_32
            | pm4::R_WAIT_MEM_64
            | pm4::R_WRITE_DATA
            | pm4::R_ACQUIRE_MEM
            | pm4::R_RELEASE_MEM
            | pm4::R_WAIT_FLIP_DONE
            | pm4::R_FLIP => {
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
            if self.first(SkipKey::Reg(RegFile::Context, reg)) {
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

            pm4::CB_COLOR_CONTROL => {
                self.ctx.color_control.mode = ((value >> 4) & 0x7) as u8;
                self.ctx.color_control.op = ((value >> 16) & 0xff) as u8;
            }

            _ => {
                if self.first(SkipKey::Reg(RegFile::Context, reg)) {
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
        const SGPRS: u32 = 16;
        if reg as usize >= pm4::SH_NUM {
            if self.first(SkipKey::Reg(RegFile::Shader, reg)) {
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
                tracing::debug!(id, value = format_args!("{value:#010x}"), "PS user SGPR write");
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
                if self.first(SkipKey::Reg(RegFile::Shader, reg)) {
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

    /// Kyty: `g_hw_uc_func` / `g_hw_uc_indirect_func` — one entry
    /// (`VGT_PRIMITIVE_TYPE`).
    fn set_uconfig_register(&mut self, reg: u32, value: u32) {
        if reg as usize >= pm4::UC_NUM {
            if self.first(SkipKey::Reg(RegFile::UserConfig, reg)) {
                warn!(
                    reg = format_args!("{reg:#06x}"),
                    "user-config register index out of range — write skipped"
                );
            }
            return;
        }
        match reg {
            pm4::VGT_PRIMITIVE_TYPE => self.ucfg.prim_type = value,
            _ => {
                if self.first(SkipKey::Reg(RegFile::UserConfig, reg)) {
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
        dispatches: Vec<([u32; 3], u32, u64, [u32; 3], u8, u32)>,
        fail: Option<String>,
    }

    impl DrawSink for RecordingSink {
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
            [(
                [11, 12, 13],
                0,
                0x12_2345_6789_00,
                [8, 4, 2],
                9,
                0xCAFE_BABE,
            )]
        );
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.vgprs, 5);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.sgprs, 7);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.bulky, 1);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.tgid_x_en, 1);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.tgid_z_en, 1);
        assert_eq!(cp.get_sh_ctx().cs.cs_regs.tidig_comp_cnt, 2);
    }

    #[test]
    fn draw_error_from_sink_is_propagated_and_named() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink {
            fail: Some("no bound render target".into()),
            ..Default::default()
        };
        let mut dcb = vec![header(7, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO), 3, 0];
        dcb.extend(pad(4));
        let err = cp.run(&dcb, &mut sink).expect_err("sink refused the draw");
        match err {
            CpError::Draw { source, .. } => assert!(source.0.contains("render target")),
            other => panic!("expected a named draw fault, got {other:?}"),
        }
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
}
