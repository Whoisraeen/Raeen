//! The PM4 command processor.
//!
//! Faithful port of Kyty `emulator/src/Graphics/GraphicsRun.cpp`
//! (MIT (c) InoriRus) — specifically `CommandProcessor::Run` (L989) and its
//! `graphics_init_jmp_tables` dispatch (L4130).
//!
//! # Scope
//!
//! Gen5/AGC only: the PS5 uses AGC, not GNM, so Kyty's Gen4 block decoders
//! (pitch/slice/view) are not ported. This slice covers what a minimal draw
//! needs — `SET_{CONTEXT,SH,UCONFIG}_REG`, the embedded-shader ops, and
//! `DRAW_INDEX_AUTO`.
//!
//! # This crate cannot draw
//!
//! `kyty-graphics` has no Vulkan dependency, so unlike Kyty (whose
//! `CommandProcessor` calls straight into `GraphicsRender`) the walk here
//! terminates at the [`DrawSink`] trait. `xps5x-gpu` implements it.
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
//!    `IT_SET_CONTEXT_REG` writes as well. This is what lets a draw resolve its
//!    render target without a guest-memory reader.

use crate::hw_regs::{
    ColorAttrib2, ColorAttrib3, ColorInfo, Context, Shader, UserConfig, UserSgprType,
};
use crate::pm4::{self, ItOp, RCode};
use tracing::warn;

/// Which register file an unknown offset belonged to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
/// Typed replacement for Kyty's hard `EXIT(...)`; the crate convention is a
/// hand-written `Display` rather than a `thiserror` dependency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpError {
    UnknownOp {
        offset: u32,
        cmd_id: u32,
        op: ItOp,
    },
    UnknownCustomOp {
        offset: u32,
        cmd_id: u32,
        r: RCode,
    },
    UnknownRegister {
        offset: u32,
        file: RegFile,
        reg: u32,
    },
    RegisterOutOfRange {
        offset: u32,
        file: RegFile,
        reg: u32,
    },
    Truncated {
        offset: u32,
        need: u32,
        remaining: u32,
    },
    NotType3 {
        offset: u32,
        cmd_id: u32,
    },
    Unimplemented {
        offset: u32,
        cmd_id: u32,
        what: &'static str,
    },
    Draw {
        offset: u32,
        source: DrawError,
    },
}

impl std::fmt::Display for CpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownOp { offset, cmd_id, op } => write!(
                f,
                "unknown PM4 opcode {:#04x} at DWORD {offset} (cmd_id {cmd_id:#010x})",
                op.0
            ),
            Self::UnknownCustomOp { offset, cmd_id, r } => write!(
                f,
                "unknown AGC custom op R_{:#04x} at DWORD {offset} (cmd_id {cmd_id:#010x})",
                r.0
            ),
            Self::UnknownRegister { offset, file, reg } => {
                write!(f, "unknown {file} register {reg:#06x} at DWORD {offset}")
            }
            Self::RegisterOutOfRange { offset, file, reg } => write!(
                f,
                "register index {reg:#x} out of range for the {file} file at DWORD {offset}"
            ),
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
            Self::Unimplemented {
                offset,
                cmd_id,
                what,
            } => write!(
                f,
                "unimplemented at DWORD {offset} ({what}, cmd_id {cmd_id:#010x})"
            ),
            Self::Draw { offset, source } => {
                write!(f, "draw failed at DWORD {offset}: {source}")
            }
        }
    }
}

impl std::error::Error for CpError {}

/// Where [`CommandProcessor`] sends a translated draw.
///
/// Mirrors Kyty's `GraphicsRenderDrawIndexAuto` signature. The whole register
/// state is passed by reference; the implementor decides what it needs.
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
}

/// Kyty: `class CommandProcessor` (GraphicsRun.cpp L~100).
#[derive(Clone, Debug, Default)]
pub struct CommandProcessor {
    ctx: Context,
    ucfg: UserConfig,
    sh_ctx: Shader,
    index_type_and_size: u32,
    num_instances: u32,
    /// Latched by the `R_ZERO` 'hu' marker; types subsequent user-SGPR writes.
    user_data_marker: UserSgprType,
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

    /// Kyty: `CommandProcessor::Reset` (L519).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Kyty: `CommandProcessor::Run` (L989) — walk a DCB and execute it.
    ///
    /// Each handler returns the **body** dwords it consumed; the walker adds
    /// one for the header. That return is authoritative over the header's own
    /// length field: Kyty's draw parsers deliberately over-report in order to
    /// swallow trailing marker NOPs, and "fixing" that into header-driven
    /// advancement desyncs the walk.
    pub fn run(&mut self, data: &[u32], sink: &mut dyn DrawSink) -> Result<(), CpError> {
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
            let consumed = self.dispatch(cmd_id, body, offset, sink)?;

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
    ) -> Result<u32, CpError> {
        let op = pm4::op(cmd_id);
        match op {
            pm4::IT_NOP => self.cp_op_nop(cmd_id, body, offset, sink),
            pm4::IT_SET_CONTEXT_REG => self.cp_op_set_context_reg(cmd_id, body, offset),
            pm4::IT_SET_SH_REG => self.cp_op_set_shader_reg(cmd_id, body, offset),
            pm4::IT_SET_UCONFIG_REG => self.cp_op_set_uconfig_reg(cmd_id, body, offset),
            pm4::IT_DRAW_INDEX_AUTO => self.cp_op_draw_index_auto(cmd_id, body, offset, sink),
            pm4::IT_NUM_INSTANCES => {
                self.num_instances = *body.first().ok_or(CpError::Truncated {
                    offset,
                    need: 2,
                    remaining: 1,
                })?;
                Ok(1)
            }
            pm4::IT_INDEX_TYPE => {
                self.index_type_and_size = *body.first().ok_or(CpError::Truncated {
                    offset,
                    need: 2,
                    remaining: 1,
                })?;
                Ok(1)
            }
            // Kyty ports these with 22 EXIT_NOT_IMPLEMENTED sites between them;
            // nothing on the minimal draw path observes their effects.
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
                warn!(
                    cmd_id = format_args!("{cmd_id:#010x}"),
                    op = op.0,
                    offset,
                    "PM4 sync/data packet consumed without effect (Phase 1)"
                );
                Ok(pm4::body_dw(cmd_id))
            }
            _ => Err(CpError::UnknownOp { offset, cmd_id, op }),
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
            pm4::R_VS_EMBEDDED => {
                // Kyty: hw_sh_set_vs_embedded (L2367). cmd_id 0xc01b1034.
                let shader_modifier = Self::body_at(body, 0, offset)?;
                let id = Self::body_at(body, 1, offset)?;
                self.sh_ctx.set_vs_embedded(id, shader_modifier);
                Ok(pm4::body_dw(cmd_id))
            }
            pm4::R_PS_EMBEDDED => {
                // Kyty: hw_sh_set_ps_embedded (L2264). cmd_id 0xc0261038.
                let id = Self::body_at(body, 0, offset)?;
                self.sh_ctx.set_ps_embedded(id);
                Ok(pm4::body_dw(cmd_id))
            }
            pm4::R_DRAW_INDEX_AUTO => self.cp_op_draw_index_auto(cmd_id, body, offset, sink),
            pm4::R_PUSH_MARKER | pm4::R_POP_MARKER => Ok(pm4::body_dw(cmd_id)),
            // Sync / flip / memory ops: consumed, not honoured. A draw never
            // observes them, and stubbing beats a wrong transliteration.
            pm4::R_WAIT_MEM_32
            | pm4::R_WAIT_MEM_64
            | pm4::R_WRITE_DATA
            | pm4::R_ACQUIRE_MEM
            | pm4::R_RELEASE_MEM
            | pm4::R_WAIT_FLIP_DONE
            | pm4::R_FLIP
            | pm4::R_DRAW_RESET
            | pm4::R_DISPATCH_RESET => {
                warn!(
                    cmd_id = format_args!("{cmd_id:#010x}"),
                    r = r.0,
                    offset,
                    "AGC sync/flip packet consumed without effect (Phase 1)"
                );
                Ok(pm4::body_dw(cmd_id))
            }
            _ => Err(CpError::UnknownCustomOp { offset, cmd_id, r }),
        }
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

    /// Kyty: `cp_op_draw_index_auto` (L1071 / L3xxx).
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
        if reg as usize >= pm4::CX_NUM {
            return Err(CpError::RegisterOutOfRange {
                offset,
                file: RegFile::Context,
                reg,
            });
        }
        let values = &body[1..];

        // Kyty's only multi-register context block on this path.
        if reg == pm4::PA_SC_SCREEN_SCISSOR_TL && values.len() >= 2 {
            return self
                .hw_ctx_set_screen_scissor(values, offset)
                .map(|n| n + 1);
        }

        let count = Self::reg_count(cmd_id, values, offset)?;
        for (i, &value) in values.iter().enumerate().take(count) {
            self.set_context_register(reg + i as u32, value, offset)?;
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
        let sixteen = |v: u32, f: (u32, u32)| i32::from(pm4::field(v, f) as u16 as i16);
        let vp = &mut self.ctx.screen_viewport;
        vp.screen_scissor_left = sixteen(values[0], pm4::pa_sc_screen_scissor::TL_X);
        vp.screen_scissor_top = sixteen(values[0], pm4::pa_sc_screen_scissor::TL_Y);
        vp.screen_scissor_right = sixteen(values[1], pm4::pa_sc_screen_scissor::BR_X);
        vp.screen_scissor_bottom = sixteen(values[1], pm4::pa_sc_screen_scissor::BR_Y);
        Ok(2)
    }

    /// The per-register context setters — Kyty's `g_hw_ctx_indirect_func`
    /// table (`graphics_init_jmp_tables_cx_indirect`, L3482).
    ///
    /// These take `(offset, value)` and touch no memory, so they serve direct
    /// `SET_CONTEXT_REG` writes as well as the indirect packet. This is the
    /// only route to the PS5 extent registers.
    fn set_context_register(&mut self, reg: u32, value: u32, offset: u32) -> Result<(), CpError> {
        let slot_of = |base: u32, stride: u32| ((reg - base) / stride) as usize;

        match reg {
            pm4::CB_TARGET_MASK => self.ctx.render_target_mask = value,

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
                return Err(CpError::UnknownRegister {
                    offset,
                    file: RegFile::Context,
                    reg,
                });
            }
        }
        Ok(())
    }

    /// Kyty: `cp_op_set_shader_reg` (L3311).
    fn cp_op_set_shader_reg(
        &mut self,
        cmd_id: u32,
        body: &[u32],
        offset: u32,
    ) -> Result<u32, CpError> {
        let reg = pm4::strip_fake(Self::body_at(body, 0, offset)?);
        if reg as usize >= pm4::SH_NUM {
            return Err(CpError::RegisterOutOfRange {
                offset,
                file: RegFile::Shader,
                reg,
            });
        }
        let values = &body[1..];
        let count = Self::reg_count(cmd_id, values, offset)?;
        for (i, &value) in values.iter().enumerate().take(count) {
            self.set_shader_register(reg + i as u32, value, offset)?;
        }
        Ok(count as u32 + 1)
    }

    fn set_shader_register(&mut self, reg: u32, value: u32, offset: u32) -> Result<(), CpError> {
        const SGPRS: u32 = 16;
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
                self.sh_ctx.ps.ps_user_sgpr.set(id, value, marker);
            }
            _ => {
                return Err(CpError::UnknownRegister {
                    offset,
                    file: RegFile::Shader,
                    reg,
                });
            }
        }
        Ok(())
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
        if reg as usize >= pm4::UC_NUM {
            return Err(CpError::RegisterOutOfRange {
                offset,
                file: RegFile::UserConfig,
                reg,
            });
        }
        let values = &body[1..];
        let count = Self::reg_count(cmd_id, values, offset)?;
        for (i, &value) in values.iter().enumerate().take(count) {
            match reg + i as u32 {
                pm4::VGT_PRIMITIVE_TYPE => self.ucfg.prim_type = value,
                other => {
                    return Err(CpError::UnknownRegister {
                        offset,
                        file: RegFile::UserConfig,
                        reg: other,
                    });
                }
            }
        }
        Ok(count as u32 + 1)
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
    use crate::pm4::header;

    #[derive(Default)]
    struct RecordingSink {
        draws: Vec<(u32, u32, u32, bool, bool)>,
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
    }

    /// Body dwords the AGC embedded/draw packets declare, as padding.
    fn pad(n: usize) -> Vec<u32> {
        vec![0; n]
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
        let mut dcb = vec![
            header(3, pm4::IT_SET_UCONFIG_REG, pm4::R_ZERO),
            pm4::VGT_PRIMITIVE_TYPE,
            17,
        ];
        dcb.extend_from_slice(&[header(7, pm4::IT_NOP, pm4::R_DRAW_INDEX_AUTO), 3, 0]);
        dcb.extend(pad(4));
        cp.run(&dcb, &mut sink).expect("draw");
        assert_eq!(sink.draws, [(3, 0, 17, false, false)]);
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
        assert!(
            matches!(
                err,
                CpError::Truncated { .. } | CpError::UnknownRegister { .. }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn unknown_register_is_a_named_error_not_a_panic() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![
            header(3, pm4::IT_SET_CONTEXT_REG, pm4::R_ZERO),
            0x3FE,
            0xAAAA,
        ];
        let err = cp.run(&dcb, &mut sink).expect_err("unknown register");
        match err {
            CpError::UnknownRegister { file, reg, .. } => {
                assert_eq!(file, RegFile::Context);
                assert_eq!(reg, 0x3FE);
            }
            other => panic!("expected UnknownRegister, got {other:?}"),
        }
    }

    #[test]
    fn unknown_opcode_names_the_op() {
        let mut cp = CommandProcessor::new();
        let mut sink = RecordingSink::default();
        let dcb = vec![header(2, ItOp(0xEE), pm4::R_ZERO), 0];
        match cp.run(&dcb, &mut sink) {
            Err(CpError::UnknownOp { op, .. }) => assert_eq!(op, ItOp(0xEE)),
            other => panic!("expected UnknownOp, got {other:?}"),
        }
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
}
