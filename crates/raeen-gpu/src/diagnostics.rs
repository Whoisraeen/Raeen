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
            // The persistent sampled-texture cache is OFF unless explicitly
            // asked for. It is a measured correctness regression: with the
            // cache on, Minecraft's panorama and menu render as flat
            // untextured geometry and 5 of 12 sampled-texture probes read
            // all-zero; with it off the same frame renders completely
            // (textured panorama, logo, skin, buttons) and 0 of 12 read
            // all-zero. Frame captures both ways, 2026-07-24.
            //
            // This is the A/B `docs/rendering-blockers-and-port-plan-2026-07-22.md`
            // demanded be settled before further perf work — it regresses, so
            // the default flips to correctness while the invalidation bug is
            // found. The mechanism is NOT yet understood and the obvious
            // theories are already disproved: the sample hash is whole-range
            // (exact) for ranges up to 4 KiB, which covers the measured 1 KiB
            // texture, and `sampled_render_target` is consulted BEFORE
            // `decode_texture`, so the cache does not short-circuit a live
            // render-target bind. Do not re-enable by default until a frame
            // capture shows the cache on rendering identically to the cache
            // off. `RAEEN_TEX_CACHE=1` restores it for that comparison.
            no_tex_cache: !on("RAEEN_TEX_CACHE"),
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
        }
    }
}

pub(crate) fn gpu_env() -> &'static GpuEnv {
    static ENV: OnceLock<GpuEnv> = OnceLock::new();
    ENV.get_or_init(GpuEnv::capture)
}
