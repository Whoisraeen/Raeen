//! Process-lifetime GPU diagnostic switches.
//!
//! Raeen sets these before a title starts. Reading the Windows environment in
//! every draw/dispatch takes a global lock and allocates an `OsString`, so the
//! hot path snapshots the switches once on first GPU use.

use std::sync::OnceLock;

pub(crate) struct GpuEnv {
    pub dump_all_targets: bool,
    pub dump_draw_state: Option<String>,
    pub dump_draw_target: Option<String>,
    pub dump_frames: Option<String>,
    pub dump_gpu_resources: Option<String>,
    pub force_clear: bool,
    pub no_cull: bool,
    pub no_defer: bool,
    pub no_defer_compute: bool,
    pub no_depth: bool,
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
    pub solid_ps_addr: Option<String>,
    pub trace_shader_addr: Option<String>,
    pub trace_shader_words: Option<String>,
}

impl GpuEnv {
    fn capture() -> Self {
        let on = |name| std::env::var_os(name).is_some();
        Self {
            dump_all_targets: on("RAEEN_DUMP_ALL_TARGETS"),
            dump_draw_state: std::env::var("RAEEN_DUMP_DRAW_STATE").ok(),
            dump_draw_target: std::env::var("RAEEN_DUMP_DRAW_TARGET").ok(),
            dump_frames: std::env::var("RAEEN_DUMP_FRAMES").ok(),
            dump_gpu_resources: std::env::var("RAEEN_DUMP_GPU_RESOURCES").ok(),
            force_clear: on("RAEEN_FORCE_CLEAR"),
            no_cull: on("RAEEN_NO_CULL"),
            no_defer: on("RAEEN_NO_DEFER"),
            no_defer_compute: on("RAEEN_NO_DEFER_COMPUTE"),
            no_depth: on("RAEEN_NO_DEPTH"),
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
