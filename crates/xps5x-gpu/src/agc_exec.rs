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
use kyty_graphics::run::{CommandProcessor, CpError};
use parking_lot::Mutex;
use std::sync::OnceLock;
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
    Cp(#[from] CpError),
}

/// Process-global session: lazy Vulkan bring-up + last rendered image.
pub struct AgcGpuSession {
    backend: Mutex<Option<VulkanBackend>>,
    /// GPU register state persists across queue submissions. AGC commonly
    /// submits state-only DCBs before a later draw-only DCB.
    command_processor: Mutex<CommandProcessor>,
    last_image: Mutex<Option<RenderedImage>>,
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
}

impl AgcGpuSession {
    fn new() -> Self {
        Self {
            backend: Mutex::new(None),
            command_processor: Mutex::new(CommandProcessor::new()),
            last_image: Mutex::new(None),
            draw_count: Mutex::new(0),
            shader_cache: Mutex::new(crate::shader_fetch::ShaderTranslateCache::new()),
            shader_skip_count: Mutex::new(0),
            framebuffers: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Shared process-wide session (HLE submit + acceptance tests).
    pub fn global() -> &'static AgcGpuSession {
        static SESSION: OnceLock<AgcGpuSession> = OnceLock::new();
        SESSION.get_or_init(AgcGpuSession::new)
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
    pub fn map_shader_metadata(
        &self,
        code_address: u64,
        data: kyty_graphics::shader::ShaderMappedData,
    ) {
        self.shader_cache
            .lock()
            .map_shader_metadata(code_address, data);
    }

    /// Last image produced by a draw-bearing DCB, if any.
    pub fn last_image(&self) -> Option<RenderedImage> {
        self.last_image.lock().clone()
    }

    fn ensure_backend(&self) -> Result<(), GpuError> {
        let mut slot = self.backend.lock();
        if slot.is_some() {
            return Ok(());
        }
        let mut backend = VulkanBackend::new(true);
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
    /// [`AgcExecError::Cp`] if a packet is unknown/truncated or a draw's
    /// registers cannot be honoured (the error names the register), or
    /// [`AgcExecError::Gpu`] if Vulkan is unavailable.
    pub fn execute_dcb_cp(&self, words: &[u32]) -> Result<Option<RenderedImage>, AgcExecError> {
        let decoded = agc::decode_submission(words)?;
        let mut cp = self.command_processor.lock();

        // State-only DCBs are still real GPU work. Process them without
        // forcing Vulkan initialization so their register writes are latched
        // for the next submission.
        if decoded.draw_packets == 0 && decoded.dispatch_packets == 0 {
            let mut sink = StateOnlySink;
            cp.run_with_memory(
                words,
                &mut sink,
                Some(&crate::guest_mem::IdentityGuestMemory),
            )?;
            return Ok(None);
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
        // Indirect register/draw packets carry guest pointers; the identity
        // map makes them host-readable (VirtualQuery-validated).
        let run = cp.run_with_memory(
            words,
            &mut sink,
            Some(&crate::guest_mem::IdentityGuestMemory),
        );
        let shader_state = cp.get_sh_ctx().clone();

        let drawn = sink.draws;
        let shader_skips = sink.shader_skips;
        let sink_skip_reason = sink.last_shader_skip_reason.clone();
        let image = sink.last.take();
        drop(sink);
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
        drop(framebuffers);
        drop(cache);
        drop(guard);
        if shader_skips > 0 {
            let total = {
                let mut skips = self.shader_skip_count.lock();
                *skips += shader_skips;
                *skips
            };
            if total == shader_skips || total.is_power_of_two() {
                warn!(
                    total_shader_skips = total,
                    reason = sink_skip_reason.as_deref().unwrap_or("unknown"),
                    vs_addr = format_args!("{:#x}", shader_state.vs.vs_regs.data_addr),
                    es_addr = format_args!("{:#x}", shader_state.vs.es_regs.data_addr),
                    gs_addr = format_args!("{:#x}", shader_state.vs.gs_regs.data_addr),
                    gs_checksum = format_args!("{:#x}", shader_state.vs.gs_regs.chksum),
                    ps_addr = format_args!("{:#x}", shader_state.ps.ps_regs.data_addr),
                    stats = ?self.shader_stats(),
                    "AGC draws skipped because bound shader state is not renderable"
                );
            }
        }
        run?;

        if let Some(image) = image {
            *self.last_image.lock() = Some(image.clone());
            let count = {
                let mut draws = self.draw_count.lock();
                *draws += drawn;
                *draws
            };
            maybe_dump_frame(&image, count);
            // A title renders its UI to several render targets and composites
            // them; the last-drawn one (often the display's black background
            // this early) is not necessarily where the content is. The
            // snapshot above (taken under the lock) lets the all-targets dump
            // surface content in a non-final target instead of discarding it.
            if let Some(targets) = all_targets {
                maybe_dump_all_targets(&targets, count);
            }
            return Ok(Some(image));
        }
        debug!("AGC DCB ran through the command processor without a draw");
        Ok(None)
    }

    /// Best-effort [`Self::execute_dcb_cp`] for the HLE submit path: a GPU
    /// fault must not become a guest-visible submit failure.
    pub fn try_execute_dcb_cp(&self, words: &[u32]) {
        match self.execute_dcb_cp(words) {
            Ok(Some(image)) => debug!(
                width = image.width,
                height = image.height,
                "AGC DCB drove a register-state Vulkan draw"
            ),
            Ok(None) => {}
            Err(e) => warn!(error = %e, "AGC DCB draw skipped"),
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
        let non_black = image
            .pixels
            .chunks_exact(4)
            .filter(|px| px[0] != 0 || px[1] != 0 || px[2] != 0)
            .count();
        let path =
            std::path::Path::new(&dir).join(format!("target_{base:012x}_{draw_index:06}.ppm"));
        let mut ppm = format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
        ppm.reserve(image.pixels.len() / 4 * 3);
        for rgba in image.pixels.chunks_exact(4) {
            ppm.extend_from_slice(&rgba[..3]);
        }
        let _ = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, &ppm));
        tracing::info!(
            base = format_args!("{base:#x}"),
            non_black_pixels = non_black,
            total = image.pixels.len() / 4,
            "render-target census"
        );
    }
}

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

        let session = AgcGpuSession::new();
        match session.execute_dcb_cp(state_only) {
            Ok(None) => {}
            Err(AgcExecError::Gpu(_)) => return, // Vulkan-less CI host.
            other => panic!("state-only submit should not draw: {other:?}"),
        }
        let image = session
            .execute_dcb_cp(draw_only)
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
