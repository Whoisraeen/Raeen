//! Process-lifetime GPU diagnostic switches.
//!
//! Raeen sets these before a title starts. Reading the Windows environment in
//! every draw/dispatch takes a global lock and allocates an `OsString`, so the
//! hot path snapshots the switches once on first GPU use.

use std::sync::OnceLock;

pub(crate) struct GpuEnv {
    pub allow_known_device_loss_cs: bool,
    pub dump_all_targets: bool,
    pub dump_draw_state: Option<String>,
    pub dump_draw_target: Option<String>,
    pub dump_frames: Option<String>,
    pub dump_gpu_resources: Option<String>,
    pub follow_ib_chains: bool,
    pub force_clear: bool,
    pub no_cull: bool,
    pub no_defer: bool,
    pub no_defer_compute: bool,
    pub relax_simple_compute_tdr: bool,
    pub no_depth: bool,
    pub no_mip_chain: bool,
    pub no_stencil: bool,
    pub no_tex_cache: bool,
    pub time_compute: bool,
    pub time_draw: bool,
    pub time_worker: bool,
    pub trace_draw_state: bool,
    pub trace_draws: bool,
    pub trace_flip: bool,
    pub trace_model: bool,
    pub trace_textures: bool,
    pub skip_cs: Option<String>,
    pub skip_shader_addr: Option<String>,
    pub solid_ps_addr: Option<String>,
    pub trace_shader_addr: Option<String>,
    pub trace_shader_words: Option<String>,
}

impl GpuEnv {
    fn capture() -> Self {
        let on = |name| std::env::var_os(name).is_some();
        Self {
            allow_known_device_loss_cs: on("RAEEN_ALLOW_KNOWN_DEVICE_LOSS_CS"),
            dump_all_targets: on("RAEEN_DUMP_ALL_TARGETS"),
            dump_draw_state: std::env::var("RAEEN_DUMP_DRAW_STATE").ok(),
            dump_draw_target: std::env::var("RAEEN_DUMP_DRAW_TARGET").ok(),
            dump_frames: std::env::var("RAEEN_DUMP_FRAMES").ok(),
            dump_gpu_resources: std::env::var("RAEEN_DUMP_GPU_RESOURCES").ok(),
            // Walk `IT_INDIRECT_BUFFER` chain targets instead of only counting
            // them. Default OFF: it dereferences guest-supplied addresses and
            // changes which command stream executes, so the working titles
            // (Minecraft Bedrock, Dead Cells, Blasphemous II) must be A/B-able
            // against it in one run. With it off, chain packets are still
            // decoded and counted — see `CHAIN CENSUS` and
            // `kyty_graphics::run::ChainCensus`.
            follow_ib_chains: on("RAEEN_FOLLOW_IB_CHAINS"),
            force_clear: on("RAEEN_FORCE_CLEAR"),
            no_cull: on("RAEEN_NO_CULL"),
            no_defer: on("RAEEN_NO_DEFER"),
            no_defer_compute: on("RAEEN_NO_DEFER_COMPUTE"),
            // The default one-billion weighted-instruction TDR budget remains
            // deliberately conservative. This opt-in raises the budget only
            // for small compute modules so retail A/B runs can validate
            // whether batching them is safe before changing the default.
            relax_simple_compute_tdr: on("RAEEN_RELAX_SIMPLE_COMPUTE_TDR"),
            no_depth: on("RAEEN_NO_DEPTH"),
            // A GFX10 mip chain is stored smallest-first, so mip 0 sits at the
            // END of the allocation; `decode_texture` relocates its tiled read
            // accordingly (SharpEmu #470). `RAEEN_NO_MIP_CHAIN=1` restores the
            // read-at-descriptor-base behaviour for A/B bisecting a title whose
            // MAX_MIP field turns out not to describe its allocation.
            no_mip_chain: on("RAEEN_NO_MIP_CHAIN"),
            no_stencil: on("RAEEN_NO_STENCIL"),
            // Persistent sampled textures are the title-path default. A
            // 2026-07-25 Minecraft A/B after the compute-to-graphics
            // publication fix showed the cached path rendering the panorama
            // and menu correctly while reducing steady worker frame time from
            // ~100-115 ms to ~23-26 ms. Keep an explicit correctness/perf
            // bisection switch: `RAEEN_NO_TEX_CACHE=1` restores per-draw guest
            // decode and upload.
            no_tex_cache: on("RAEEN_NO_TEX_CACHE"),
            time_compute: on("RAEEN_TIME_COMPUTE"),
            time_draw: on("RAEEN_TIME_DRAW"),
            time_worker: on("RAEEN_TIME_WORKER"),
            trace_draw_state: on("RAEEN_TRACE_DRAW_STATE"),
            trace_draws: on("RAEEN_TRACE_DRAWS"),
            trace_flip: on("RAEEN_TRACE_FLIP"),
            trace_model: on("RAEEN_TRACE_MODEL"),
            trace_textures: on("RAEEN_TRACE_TEXTURES"),
            skip_cs: std::env::var("RAEEN_SKIP_CS").ok(),
            skip_shader_addr: std::env::var("RAEEN_SKIP_SHADER_ADDR").ok(),
            solid_ps_addr: std::env::var("RAEEN_SOLID_PS_ADDR").ok(),
            trace_shader_addr: std::env::var("RAEEN_TRACE_SHADER_ADDR").ok(),
            trace_shader_words: std::env::var("RAEEN_TRACE_SHADER_WORDS").ok(),
        }
    }
}

pub(crate) fn gpu_env() -> &'static GpuEnv {
    static ENV: OnceLock<GpuEnv> = OnceLock::new();
    ENV.get_or_init(GpuEnv::capture)
}

// ─── RenderDoc programmatic captures ────────────────────────────────────────
//
// `RAEEN_RENDERDOC_CAPTURE=N` (with Raeen launched *under RenderDoc*) wraps
// the next N DCB executions in explicit Start/EndFrameCapture calls. Raeen
// renders offscreen — there is no swapchain "frame" for RenderDoc to latch
// onto by itself — so the bracket is what turns a headless draw into a
// capturable event. Without the env var, or outside RenderDoc, everything
// here is a no-op.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

fn renderdoc_api() -> &'static Mutex<Option<renderdoc::RenderDoc<renderdoc::V141>>> {
    static API: OnceLock<Mutex<Option<renderdoc::RenderDoc<renderdoc::V141>>>> = OnceLock::new();
    API.get_or_init(|| {
        let api = match renderdoc::RenderDoc::<renderdoc::V141>::new() {
            Ok(api) => Some(api),
            Err(e) => {
                tracing::warn!(
                    "RAEEN_RENDERDOC_CAPTURE set but the RenderDoc API is unavailable \
                     (launch Raeen from RenderDoc): {e}"
                );
                None
            }
        };
        Mutex::new(api)
    })
}

/// Remaining capture budget from `RAEEN_RENDERDOC_CAPTURE` (`0` = disabled).
fn renderdoc_budget() -> &'static AtomicU64 {
    static BUDGET: OnceLock<AtomicU64> = OnceLock::new();
    BUDGET.get_or_init(|| {
        AtomicU64::new(
            std::env::var("RAEEN_RENDERDOC_CAPTURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        )
    })
}

/// RAII guard: ends the RenderDoc capture when the wrapped DCB execution
/// returns (any path — success or error).
pub(crate) struct RenderdocCapture(());

impl Drop for RenderdocCapture {
    fn drop(&mut self) {
        if let Some(api) = renderdoc_api().lock().unwrap().as_mut() {
            api.end_frame_capture(std::ptr::null(), std::ptr::null());
            tracing::info!("RenderDoc capture ended for one DCB execution");
        }
    }
}

/// Begin a RenderDoc capture for one DCB execution if budget remains.
/// Returns `None` (no-op) when disabled, exhausted, or not under RenderDoc.
pub(crate) fn renderdoc_dcb_capture() -> Option<RenderdocCapture> {
    let budget = renderdoc_budget();
    if budget.load(Ordering::Relaxed) == 0 {
        return None;
    }
    let mut api = renderdoc_api().lock().unwrap();
    let api = api.as_mut()?;
    // Claim one unit of budget; a concurrent claimant losing the race skips.
    if budget
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
        .is_err()
    {
        return None;
    }
    api.start_frame_capture(std::ptr::null(), std::ptr::null());
    Some(RenderdocCapture(()))
}
