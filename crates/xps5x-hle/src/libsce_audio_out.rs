//! HLE libSceAudioOut — audio output management.
//!
//! A title's audio thread loops: fill a buffer, call `sceAudioOutOutput`
//! (which on real hardware *blocks* for roughly one buffer's playback time
//! so the ring buffer paces itself), repeat. XPS5X does not yet play audio,
//! but this stub must (a) never hang that thread and (b) not let it spin at
//! 100% CPU. So `Output` acknowledges the buffer, sleeps ~one buffer period
//! (grain ÷ frequency, bounded), and returns the sample count — real pacing
//! without real playback (CLAUDE.md M3: "audio stub must not hang the
//! title"). Export set cross-checked against SharpEmu.

use crate::{HleContext, HleRegistry};
use std::time::Duration;
use tracing::debug;

/// `SCE_OK`.
const SCE_OK: u64 = 0;
/// The process already owns the maximum number of emulator AudioOut ports.
const SCE_AUDIO_OUT_ERROR_PORT_FULL: u64 = 0x8026_0004;
/// Upper bound on how long one `Output` blocks — a wild grain/frequency
/// can't wedge the audio thread for more than this.
const OUTPUT_MAX_SLEEP: Duration = Duration::from_millis(100);

/// Register libSceAudioOut HLE functions.
pub fn register(registry: &HleRegistry) {
    registry.register("libSceAudioOut", "sceAudioOutInit", hle_ok);
    registry.register("libSceAudioOut", "sceAudioOutOpen", hle_open);
    registry.register("libSceAudioOut", "sceAudioOutOutput", hle_output);
    registry.register("libSceAudioOut", "sceAudioOutClose", hle_close);
    registry.register("libSceAudioOut", "sceAudioOutSetVolume", hle_ok);
}

fn hle_ok(_ctx: &HleContext, _args: &[u64]) -> u64 {
    SCE_OK
}

/// `sceAudioOutOpen(userId, type, index, length, freq, param)`: records the
/// port's grain (`length`, samples per `Output`) and `freq`, returns a new
/// positive port handle.
fn hle_open(ctx: &HleContext, args: &[u64]) -> u64 {
    let grain = args.get(3).copied().unwrap_or(256) as u32;
    let freq = args.get(4).copied().unwrap_or(48_000) as u32;
    let Some(handle) = ctx.kernel.open_audio_out_port(grain, freq) else {
        return SCE_AUDIO_OUT_ERROR_PORT_FULL;
    };
    debug!("sceAudioOutOpen(grain={grain}, freq={freq}) -> handle {handle}");
    handle as u64
}

/// `sceAudioOutOutput(handle, ptr)`: acknowledge the submitted buffer, sleep
/// ~one buffer period (grain ÷ freq, capped) so the audio thread paces
/// instead of spinning, and return the sample count. No samples are actually
/// played yet.
fn hle_output(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    let (grain, freq) = ctx.kernel.audio_out_port(handle).unwrap_or((256, 48_000));
    if freq > 0 {
        let period = Duration::from_secs_f64(grain as f64 / freq as f64).min(OUTPUT_MAX_SLEEP);
        std::thread::sleep(period);
    }
    grain as u64
}

/// `sceAudioOutClose(handle)`: drop the port's recorded pacing state.
fn hle_close(ctx: &HleContext, args: &[u64]) -> u64 {
    let handle = args.first().copied().unwrap_or(0) as u32;
    debug!("sceAudioOutClose(handle={handle})");
    ctx.kernel.close_audio_out_port(handle);
    SCE_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_bits() -> (
        xps5x_kernel::OrbisKernel,
        crate::TestMemory,
        crate::TestAllocator,
    ) {
        (
            xps5x_kernel::OrbisKernel::new(),
            crate::TestMemory::new(0x100),
            crate::TestAllocator::new(0),
        )
    }

    #[test]
    fn open_returns_distinct_positive_handles() {
        let (k, m, a) = ctx_bits();
        let ctx = crate::test_ctx(&k, &m, &a);
        let h1 = hle_open(&ctx, &[0, 0, 0, 256, 48_000, 0]);
        let h2 = hle_open(&ctx, &[0, 0, 0, 256, 48_000, 0]);
        assert!(
            h1 > 0 && h2 > 0 && h1 != h2,
            "handles must be positive and distinct"
        );
    }

    #[test]
    fn output_returns_grain_and_paces_without_hanging() {
        let (k, m, a) = ctx_bits();
        let ctx = crate::test_ctx(&k, &m, &a);
        let h = hle_open(&ctx, &[0, 0, 0, 256, 48_000, 0]);
        let t0 = std::time::Instant::now();
        let ret = hle_output(&ctx, &[h, 0x1000]);
        assert_eq!(ret, 256, "returns the grain (samples submitted)");
        // 256/48000 ≈ 5.3ms; it slept *something* but nowhere near the cap.
        assert!(t0.elapsed() < OUTPUT_MAX_SLEEP * 2, "must not hang");
        assert_eq!(hle_close(&ctx, &[h]), SCE_OK);
    }

    #[test]
    fn output_on_unknown_handle_uses_defaults_and_returns() {
        let (k, m, a) = ctx_bits();
        let ctx = crate::test_ctx(&k, &m, &a);
        // Never opened: falls back to default grain, still returns promptly.
        assert_eq!(hle_output(&ctx, &[9999, 0x1000]), 256);
    }

    #[test]
    fn port_table_is_bounded_and_close_releases_capacity() {
        let (k, m, a) = ctx_bits();
        let ctx = crate::test_ctx(&k, &m, &a);
        let mut handles = Vec::new();
        for _ in 0..xps5x_kernel::OrbisKernel::MAX_AUDIO_OUT_PORTS {
            let handle = hle_open(&ctx, &[0, 0, 0, 256, 48_000, 0]);
            assert_ne!(handle, SCE_AUDIO_OUT_ERROR_PORT_FULL);
            handles.push(handle);
        }
        assert_eq!(
            hle_open(&ctx, &[0, 0, 0, 256, 48_000, 0]),
            SCE_AUDIO_OUT_ERROR_PORT_FULL
        );
        assert_eq!(hle_close(&ctx, &[handles[0]]), SCE_OK);
        assert_ne!(
            hle_open(&ctx, &[0, 0, 0, 256, 48_000, 0]),
            SCE_AUDIO_OUT_ERROR_PORT_FULL
        );
    }

    #[test]
    fn separate_process_kernels_do_not_share_audio_ports() {
        let (first, first_mem, first_alloc) = ctx_bits();
        let first_ctx = crate::test_ctx(&first, &first_mem, &first_alloc);
        let first_handle = hle_open(&first_ctx, &[0, 0, 0, 256, 48_000, 0]);

        let (second, second_mem, second_alloc) = ctx_bits();
        let second_ctx = crate::test_ctx(&second, &second_mem, &second_alloc);
        let second_handle = hle_open(&second_ctx, &[0, 0, 0, 256, 48_000, 0]);
        assert_eq!(first_handle, 1);
        assert_eq!(second_handle, 1);
        assert!(first.audio_out_port(first_handle as u32).is_some());
        assert!(second.audio_out_port(second_handle as u32).is_some());
    }
}
